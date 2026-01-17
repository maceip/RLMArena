"""
Verified Anchor-Based Ranking Group Reward Model

This module extends the anchor-based ranking with verification layer support
for the VaaS (Verifier-as-a-Service) architecture.
"""

import asyncio
from typing import Any

import numpy as np
import pandas as pd
import torch

from qqr import registers
from qqr.schemas import GroupRewardModel, LLMJudge, ValidationResult, VerificationStatus
from qqr.verification import CompositeJudge, VerificationPlugin


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
