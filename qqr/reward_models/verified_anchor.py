"""
Verified Anchor-Based Ranking Group Reward Model

This module extends the anchor-based ranking with verification layer support
for the VaaS (Verifier-as-a-Service) architecture.

Includes Verification-Weighted Ranking where trajectories that pass "Hard Checks"
(e.g., successful code execution, terraform plan) are weighted so heavily that
they can almost never lose to trajectories that only "look good" to the LLM.
"""

import asyncio
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd
import torch

from qqr import registers
from qqr.schemas import GroupRewardModel, LLMJudge, ValidationResult, VerificationStatus
from qqr.verification import CompositeJudge, VerificationPlugin


class VerificationWeight(Enum):
    """Weight categories for verification results."""
    HARD_PASS = "hard_pass"  # Passed execution/deployment test
    SOFT_PASS = "soft_pass"  # Passed linting/static analysis
    PARTIAL = "partial"  # Some checks passed
    SOFT_FAIL = "soft_fail"  # Failed soft checks only
    HARD_FAIL = "hard_fail"  # Failed hard checks


@dataclass
class WeightedVerificationResult:
    """Verification result with weighting information."""
    passed: bool
    weight_category: VerificationWeight
    hard_checks_passed: int
    hard_checks_total: int
    soft_checks_passed: int
    soft_checks_total: int
    execution_verified: bool = False
    security_verified: bool = False
    score_multiplier: float = 1.0
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def hard_check_ratio(self) -> float:
        """Ratio of hard checks passed."""
        if self.hard_checks_total == 0:
            return 1.0
        return self.hard_checks_passed / self.hard_checks_total

    def compute_weight(
        self,
        hard_pass_weight: float = 10.0,
        execution_bonus: float = 5.0,
        security_bonus: float = 2.0,
    ) -> float:
        """
        Compute the final weight for this verification result.

        Hard passes receive massive bonuses to ensure they almost never
        lose to trajectories that only "look good" to the LLM.
        """
        base_weight = 1.0

        if self.weight_category == VerificationWeight.HARD_PASS:
            base_weight = hard_pass_weight
        elif self.weight_category == VerificationWeight.SOFT_PASS:
            base_weight = 3.0
        elif self.weight_category == VerificationWeight.PARTIAL:
            base_weight = 1.5
        elif self.weight_category == VerificationWeight.SOFT_FAIL:
            base_weight = 0.5
        elif self.weight_category == VerificationWeight.HARD_FAIL:
            base_weight = 0.1

        # Apply bonuses
        if self.execution_verified:
            base_weight *= (1.0 + execution_bonus)

        if self.security_verified:
            base_weight *= (1.0 + security_bonus)

        return base_weight * self.score_multiplier


@registers.reward_model("verified_anchor")
class VerifiedAnchorGroupRewardModel(GroupRewardModel):
    """
    Anchor-based ranking with verification layer integration.

    This model extends the standard anchor-based tournament with:
    1. Pre-tournament verification via plugins
    2. Dead on Arrival (DoA) handling for failed trajectories
    3. Score modification based on verification results

    The verification layer ensures constraint satisfaction before
    expensive LLM-based comparisons occur.
    """

    floor_reward: float = -1.0

    def __init__(
        self,
        llm_judge: LLMJudge,
        plugins: list[VerificationPlugin] | None = None,
        skip_failed_in_tournament: bool = True,
    ):
        """
        Initialize the verified anchor reward model.

        Args:
            llm_judge: The underlying LLM judge for comparisons
            plugins: Optional verification plugins for hard checks
            skip_failed_in_tournament: If True, DoA trajectories don't participate
        """
        super().__init__()

        if plugins:
            self.judge = CompositeJudge(llm_judge, plugins)
        else:
            self.judge = llm_judge

        self.plugins = plugins or []
        self.skip_failed_in_tournament = skip_failed_in_tournament

    async def validate_batch(
        self, predictions: list[list[dict]], metadata: list[dict[str, Any]] | None = None
    ) -> list[ValidationResult]:
        """
        Validate all predictions using verification plugins.

        Args:
            predictions: List of trajectory message lists
            metadata: Optional metadata for each trajectory

        Returns:
            List of ValidationResult for each prediction
        """
        if not self.plugins:
            return [
                ValidationResult(status=VerificationStatus.VALID)
                for _ in predictions
            ]

        if isinstance(self.judge, CompositeJudge):
            results = await self.judge.batch_verify(predictions, metadata)
            return [
                ValidationResult(
                    status=VerificationStatus.VALID if r.all_passed else VerificationStatus.INVALID,
                    message="; ".join(r.failure_messages) if r.failure_messages else "Passed",
                    details=r.to_dict(),
                )
                for r in results
            ]

        return [
            ValidationResult(status=VerificationStatus.VALID)
            for _ in predictions
        ]

    async def compute(
        self,
        predictions: list[list[dict]],
        query: str,
        metadata: list[dict[str, Any]] | None = None,
    ) -> list[float]:
        """
        Compute rewards with verification layer integration.

        Args:
            predictions: List of trajectory message lists
            query: The query being evaluated
            metadata: Optional metadata for each trajectory

        Returns:
            List of rewards with floor values for failed verifications
        """
        group_size = len(predictions)

        validation_results = await self.validate_batch(predictions, metadata)

        valid_indices = [
            i for i, v in enumerate(validation_results) if v.is_valid
        ]
        failed_indices = [
            i for i, v in enumerate(validation_results) if v.is_doa
        ]

        if len(valid_indices) == 0:
            return [self.floor_reward] * group_size

        if len(valid_indices) == 1:
            rewards = [self.floor_reward] * group_size
            rewards[valid_indices[0]] = 1.0
            return rewards

        if self.skip_failed_in_tournament:
            valid_predictions = [predictions[i] for i in valid_indices]
            valid_rewards = await self._compute_tournament(valid_predictions, query)

            rewards = [self.floor_reward] * group_size
            for idx, reward in zip(valid_indices, valid_rewards):
                rewards[idx] = reward
            return rewards
        else:
            rewards = await self._compute_tournament(predictions, query)
            return self.apply_floor_rewards(rewards, validation_results)

    async def _compute_tournament(
        self, predictions: list[list[dict]], query: str
    ) -> list[float]:
        """
        Run the anchor-based tournament on valid predictions.

        Args:
            predictions: List of valid trajectory message lists
            query: The query being evaluated

        Returns:
            List of normalized rewards
        """
        group_size = len(predictions)

        if group_size < 2:
            return [0.0] * group_size

        pivot_idx = 0
        pivot_prediction = predictions[pivot_idx]
        pivot_scores = [5.0] * group_size
        other_scores = [5.0] * group_size

        tasks = []
        async with asyncio.TaskGroup() as tg:
            for idx in range(1, group_size):
                task = tg.create_task(
                    self.judge.bidirectional_compare(
                        predictions[idx], pivot_prediction, query=query, idx=idx
                    )
                )
                tasks.append(task)

        for task in tasks:
            other_score, pivot_score, metadata = task.result()
            idx = metadata.get("idx", tasks.index(task) + 1)
            other_scores[idx] = other_score
            pivot_scores[idx] = pivot_score

        pivot_scores = pivot_scores[1:]
        pivot_mean_score = np.mean(pivot_scores)
        scores = [pivot_mean_score] + other_scores[1:]
        ranks = pd.Series(scores).rank(method="min", ascending=False).tolist()
        max_rank = max(ranks)

        if max_rank == 1:
            group_rewards = [0.0] * group_size
        else:
            group_rewards = [(max_rank - r) / (max_rank - 1) for r in ranks]

        group_rewards = torch.tensor(group_rewards, dtype=torch.float)
        mean = group_rewards.mean(dim=-1, keepdim=True)
        std = group_rewards.std(dim=-1, keepdim=True)
        group_rewards = (group_rewards - mean) / (std + 1e-6)
        group_rewards = group_rewards.flatten().tolist()

        return group_rewards


@registers.reward_model("verification_weighted")
class VerificationWeightedRankingModel(GroupRewardModel):
    """
    Verification-Weighted Ranking Group Reward Model.

    This model implements "Commercial Ground Truth Anchors" where:
    - Trajectories that pass Hard Checks (execution, deployment) receive
      massive weight bonuses that make them almost impossible to lose to
      trajectories that only "look good" to the LLM judge
    - LLM judge scores are multiplied by verification weights
    - Creates a two-tier system: verified responses vs unverified

    This is crucial for high-stakes infrastructure engineering where
    the LLM Judge can still be wrong, but terraform plan passing is
    a definitive signal.

    Configuration:
        hard_pass_weight: float = 10.0 - Weight multiplier for hard passes
        execution_bonus: float = 5.0 - Extra bonus for execution verification
        security_bonus: float = 2.0 - Extra bonus for security verification
        soft_pass_weight: float = 3.0 - Weight for soft passes
        floor_reward: float = -10.0 - Floor reward for hard failures
    """

    floor_reward: float = -10.0

    def __init__(
        self,
        llm_judge: LLMJudge,
        plugins: list[VerificationPlugin] | None = None,
        hard_pass_weight: float = 10.0,
        execution_bonus: float = 5.0,
        security_bonus: float = 2.0,
        hard_check_plugins: list[str] | None = None,
        execution_plugins: list[str] | None = None,
        security_plugins: list[str] | None = None,
    ):
        """
        Initialize the verification-weighted ranking model.

        Args:
            llm_judge: The underlying LLM judge for comparisons
            plugins: Verification plugins
            hard_pass_weight: Weight multiplier for hard passes
            execution_bonus: Extra bonus for execution verification
            security_bonus: Extra bonus for security verification
            hard_check_plugins: Names of plugins that are hard checks
            execution_plugins: Names of plugins that verify execution
            security_plugins: Names of plugins that verify security
        """
        super().__init__()

        if plugins:
            self.judge = CompositeJudge(llm_judge, plugins)
        else:
            self.judge = llm_judge

        self.plugins = plugins or []
        self.hard_pass_weight = hard_pass_weight
        self.execution_bonus = execution_bonus
        self.security_bonus = security_bonus

        # Plugin categorization
        self.hard_check_plugins = set(hard_check_plugins or [
            "runtime_verifier",
            "terraform_verifier",
            "infra_sandbox",
            "code_linter",
        ])
        self.execution_plugins = set(execution_plugins or [
            "runtime_verifier",
            "terraform_verifier",
        ])
        self.security_plugins = set(security_plugins or [
            "safety_filter",
            "leash_verifier",
            "opa_verifier",
        ])

    def _categorize_verification(
        self,
        validation_result: ValidationResult,
    ) -> WeightedVerificationResult:
        """
        Categorize a validation result into weighted categories.

        Examines the detailed plugin results to determine:
        - How many hard vs soft checks passed
        - Whether execution was verified
        - Whether security was verified
        """
        details = validation_result.details or {}
        results = details.get("results", [])

        hard_passed = 0
        hard_total = 0
        soft_passed = 0
        soft_total = 0
        execution_verified = False
        security_verified = False

        for result in results:
            plugin_name = result.get("plugin_name", "")
            status = result.get("status", "")
            passed = status == "passed"

            is_hard = plugin_name in self.hard_check_plugins
            is_execution = plugin_name in self.execution_plugins
            is_security = plugin_name in self.security_plugins

            if is_hard:
                hard_total += 1
                if passed:
                    hard_passed += 1
            else:
                soft_total += 1
                if passed:
                    soft_passed += 1

            if is_execution and passed:
                execution_verified = True

            if is_security and passed:
                security_verified = True

        # Determine weight category
        if hard_total > 0:
            if hard_passed == hard_total:
                category = VerificationWeight.HARD_PASS
            elif hard_passed > 0:
                category = VerificationWeight.PARTIAL
            else:
                category = VerificationWeight.HARD_FAIL
        else:
            if soft_total > 0 and soft_passed == soft_total:
                category = VerificationWeight.SOFT_PASS
            elif soft_passed > 0:
                category = VerificationWeight.PARTIAL
            else:
                category = VerificationWeight.SOFT_FAIL

        return WeightedVerificationResult(
            passed=validation_result.is_valid,
            weight_category=category,
            hard_checks_passed=hard_passed,
            hard_checks_total=hard_total,
            soft_checks_passed=soft_passed,
            soft_checks_total=soft_total,
            execution_verified=execution_verified,
            security_verified=security_verified,
            score_multiplier=details.get("combined_score_modifier", 1.0),
            details=details,
        )

    async def validate_batch(
        self,
        predictions: list[list[dict]],
        metadata: list[dict[str, Any]] | None = None,
    ) -> list[WeightedVerificationResult]:
        """
        Validate all predictions and return weighted results.
        """
        if not self.plugins:
            return [
                WeightedVerificationResult(
                    passed=True,
                    weight_category=VerificationWeight.SOFT_PASS,
                    hard_checks_passed=0,
                    hard_checks_total=0,
                    soft_checks_passed=0,
                    soft_checks_total=0,
                )
                for _ in predictions
            ]

        if isinstance(self.judge, CompositeJudge):
            results = await self.judge.batch_verify(predictions, metadata)
            return [
                self._categorize_verification(
                    ValidationResult(
                        status=VerificationStatus.VALID if r.all_passed else VerificationStatus.INVALID,
                        message="; ".join(r.failure_messages) if r.failure_messages else "Passed",
                        details=r.to_dict(),
                    )
                )
                for r in results
            ]

        return [
            WeightedVerificationResult(
                passed=True,
                weight_category=VerificationWeight.SOFT_PASS,
                hard_checks_passed=0,
                hard_checks_total=0,
                soft_checks_passed=0,
                soft_checks_total=0,
            )
            for _ in predictions
        ]

    async def compute(
        self,
        predictions: list[list[dict]],
        query: str,
        metadata: list[dict[str, Any]] | None = None,
    ) -> list[float]:
        """
        Compute verification-weighted rewards.

        The final reward is: LLM_score * verification_weight

        This ensures that trajectories with hard verification passes
        have a massive advantage over those that only look good to the LLM.
        """
        group_size = len(predictions)

        # Get weighted verification results
        weighted_results = await self.validate_batch(predictions, metadata)

        # Compute verification weights
        weights = [
            r.compute_weight(
                hard_pass_weight=self.hard_pass_weight,
                execution_bonus=self.execution_bonus,
                security_bonus=self.security_bonus,
            )
            for r in weighted_results
        ]

        # Separate by verification status
        valid_indices = [
            i for i, r in enumerate(weighted_results)
            if r.passed
        ]
        failed_indices = [
            i for i, r in enumerate(weighted_results)
            if not r.passed
        ]

        # If all failed, return floor rewards
        if len(valid_indices) == 0:
            return [self.floor_reward] * group_size

        # If only one valid, it wins
        if len(valid_indices) == 1:
            rewards = [self.floor_reward] * group_size
            rewards[valid_indices[0]] = 1.0 * weights[valid_indices[0]]
            return rewards

        # Run LLM tournament on valid predictions
        valid_predictions = [predictions[i] for i in valid_indices]
        valid_weights = [weights[i] for i in valid_indices]

        llm_scores = await self._compute_tournament(valid_predictions, query)

        # Apply verification weights to LLM scores
        weighted_scores = [
            score * weight
            for score, weight in zip(llm_scores, valid_weights)
        ]

        # Normalize weighted scores
        if weighted_scores:
            max_score = max(weighted_scores)
            min_score = min(weighted_scores)
            score_range = max_score - min_score

            if score_range > 0:
                weighted_scores = [
                    (s - min_score) / score_range * 2 - 1  # Normalize to [-1, 1]
                    for s in weighted_scores
                ]

        # Build final rewards
        rewards = [self.floor_reward] * group_size
        for idx, score in zip(valid_indices, weighted_scores):
            rewards[idx] = score

        return rewards

    async def _compute_tournament(
        self,
        predictions: list[list[dict]],
        query: str,
    ) -> list[float]:
        """Run the LLM-based tournament."""
        group_size = len(predictions)

        if group_size < 2:
            return [0.0] * group_size

        pivot_idx = 0
        pivot_prediction = predictions[pivot_idx]
        pivot_scores = [5.0] * group_size
        other_scores = [5.0] * group_size

        tasks = []
        async with asyncio.TaskGroup() as tg:
            for idx in range(1, group_size):
                task = tg.create_task(
                    self.judge.bidirectional_compare(
                        predictions[idx], pivot_prediction, query=query, idx=idx
                    )
                )
                tasks.append(task)

        for task in tasks:
            other_score, pivot_score, metadata = task.result()
            idx = metadata.get("idx", tasks.index(task) + 1)
            other_scores[idx] = other_score
            pivot_scores[idx] = pivot_score

        pivot_scores = pivot_scores[1:]
        pivot_mean_score = np.mean(pivot_scores)
        scores = [pivot_mean_score] + other_scores[1:]

        return scores

    def get_weight_explanation(
        self,
        weighted_result: WeightedVerificationResult,
    ) -> str:
        """Get a human-readable explanation of the weight calculation."""
        weight = weighted_result.compute_weight(
            hard_pass_weight=self.hard_pass_weight,
            execution_bonus=self.execution_bonus,
            security_bonus=self.security_bonus,
        )

        lines = [
            f"Category: {weighted_result.weight_category.value}",
            f"Hard checks: {weighted_result.hard_checks_passed}/{weighted_result.hard_checks_total}",
            f"Soft checks: {weighted_result.soft_checks_passed}/{weighted_result.soft_checks_total}",
            f"Execution verified: {weighted_result.execution_verified}",
            f"Security verified: {weighted_result.security_verified}",
            f"Final weight: {weight:.2f}x",
        ]

        return "\n".join(lines)
