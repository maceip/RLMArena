"""
CompositeJudge: The VaaS Quality Guarantor Engine

Combines deterministic verification plugins with the stochastic arena tournament
to provide reliable, constraint-satisfying LLM outputs.

Architecture:
1. Hard Verification Layer - Deterministic plugins run first
2. Arena Tournament Layer - LLM-based relative ranking
3. Score Aggregation - Combines verification status with tournament scores
"""

import asyncio
from typing import Any

from qqr.schemas import LLMJudge

from .plugin import (
    AggregateVerificationResult,
    VerificationPlugin,
    VerificationResult,
    VerificationStatus,
)


class CompositeJudge(LLMJudge):
    """
    The 'Guarantor' engine combining deterministic plugins with arena judges.

    This judge implements a two-phase evaluation:
    1. Hard Verification: All plugins run deterministic checks
    2. Arena Tournament: If verification passes, proceed to LLM-based comparison

    Dead on Arrival (DoA) Logic:
    - If a trajectory fails a Hard Check, it receives floor_reward immediately
    - The arena tournament is bypassed for DoA trajectories
    - This saves compute costs and ensures constraint satisfaction

    Usage:
        plugins = [CodeLinterPlugin(), SafetyFilterPlugin()]
        arena_judge = TravelLLMJudge()
        judge = CompositeJudge(arena_judge, plugins)

        # Pairwise comparison with verification
        score_a, score_b, meta = await judge.bidirectional_compare(
            messages_a, messages_b, query="Plan a trip..."
        )
    """

    FLOOR_REWARD = -10.0
    VERIFICATION_BONUS = 1.0

    def __init__(
        self,
        arena_judge: LLMJudge,
        plugins: list[VerificationPlugin],
        floor_reward: float = FLOOR_REWARD,
        skip_arena_on_failure: bool = True,
        parallel_verification: bool = True,
    ):
        """
        Initialize the CompositeJudge.

        Args:
            arena_judge: The underlying LLM-based judge for relative ranking
            plugins: List of verification plugins for hard checks
            floor_reward: Score assigned to trajectories that fail verification
            skip_arena_on_failure: If True, skip arena comparison when one fails
            parallel_verification: If True, run plugins concurrently
        """
        self.arena_judge = arena_judge
        self.plugins = plugins
        self.floor_reward = floor_reward
        self.skip_arena_on_failure = skip_arena_on_failure
        self.parallel_verification = parallel_verification

    async def verify_trajectory(
        self,
        messages: list[dict],
        metadata: dict[str, Any] | None = None
    ) -> AggregateVerificationResult:
        """
        Run all verification plugins on a trajectory.

        Args:
            messages: The conversation/trajectory messages
            metadata: Optional metadata (tools, query, etc.)

        Returns:
            AggregateVerificationResult with all plugin results
        """
        if self.parallel_verification:
            tasks = [
                plugin.verify(messages, metadata)
                for plugin in self.plugins
                if not plugin.should_skip(messages, metadata)
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            processed_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    processed_results.append(VerificationResult(
                        status=VerificationStatus.ERROR,
                        plugin_name=self.plugins[i].name,
                        message=f"Plugin error: {str(result)}",
                    ))
                else:
                    processed_results.append(result)
        else:
            processed_results = []
            for plugin in self.plugins:
                if plugin.should_skip(messages, metadata):
                    continue
                try:
                    result = await plugin.verify(messages, metadata)
                    processed_results.append(result)
                except Exception as e:
                    processed_results.append(VerificationResult(
                        status=VerificationStatus.ERROR,
                        plugin_name=plugin.name,
                        message=f"Plugin error: {str(e)}",
                    ))

        return AggregateVerificationResult(results=processed_results)

    async def compare(
        self,
        messages_a: list[dict],
        messages_b: list[dict],
        *args,
        **kwargs
    ) -> tuple[float, float]:
        """
        Compare two trajectories with verification layer.

        Args:
            messages_a: First trajectory messages
            messages_b: Second trajectory messages
            *args, **kwargs: Passed to arena judge

        Returns:
            Tuple of (score_a, score_b)
        """
        score_a, score_b, _ = await self.bidirectional_compare(
            messages_a, messages_b, *args, **kwargs
        )
        return score_a, score_b

    async def bidirectional_compare(
        self,
        messages_a: list[dict],
        messages_b: list[dict],
        *args,
        **kwargs
    ) -> tuple[float, float, dict]:
        """
        Bidirectional comparison with verification layer.

        Logic:
        1. Run verification on both trajectories concurrently
        2. If one fails and other passes, the passer wins by default
        3. If both fail, both get floor rewards
        4. If both pass, proceed to arena tournament

        Args:
            messages_a: First trajectory messages
            messages_b: Second trajectory messages
            *args, **kwargs: Passed to arena judge (e.g., query)

        Returns:
            Tuple of (score_a, score_b, metadata)
        """
        metadata_a = kwargs.pop("metadata_a", None)
        metadata_b = kwargs.pop("metadata_b", None)

        verification_a, verification_b = await asyncio.gather(
            self.verify_trajectory(messages_a, metadata_a),
            self.verify_trajectory(messages_b, metadata_b),
        )

        result_metadata = {
            "verification_a": verification_a.to_dict(),
            "verification_b": verification_b.to_dict(),
            "arena_comparison": None,
        }

        a_passed = verification_a.all_passed
        b_passed = verification_b.all_passed

        if not a_passed and not b_passed:
            return (
                self.floor_reward,
                self.floor_reward,
                {**result_metadata, "outcome": "both_failed_verification"},
            )

        if not a_passed:
            return (
                self.floor_reward,
                self.VERIFICATION_BONUS,
                {**result_metadata, "outcome": "a_failed_verification"},
            )

        if not b_passed:
            return (
                self.VERIFICATION_BONUS,
                self.floor_reward,
                {**result_metadata, "outcome": "b_failed_verification"},
            )

        arena_score_a, arena_score_b, arena_meta = await self.arena_judge.bidirectional_compare(
            messages_a, messages_b, *args, **kwargs
        )

        final_score_a = arena_score_a * verification_a.combined_score_modifier
        final_score_b = arena_score_b * verification_b.combined_score_modifier

        result_metadata["arena_comparison"] = arena_meta
        result_metadata["outcome"] = "arena_comparison"
        result_metadata["arena_scores"] = {"a": arena_score_a, "b": arena_score_b}
        result_metadata["final_scores"] = {"a": final_score_a, "b": final_score_b}

        return final_score_a, final_score_b, result_metadata

    async def batch_verify(
        self,
        trajectories: list[list[dict]],
        metadata: list[dict[str, Any]] | None = None
    ) -> list[AggregateVerificationResult]:
        """
        Verify multiple trajectories concurrently.

        Args:
            trajectories: List of message lists
            metadata: Optional list of metadata dicts

        Returns:
            List of AggregateVerificationResult
        """
        metadata = metadata or [None] * len(trajectories)
        tasks = [
            self.verify_trajectory(traj, meta)
            for traj, meta in zip(trajectories, metadata)
        ]
        return await asyncio.gather(*tasks)

    def get_doomed_indices(
        self, verification_results: list[AggregateVerificationResult]
    ) -> list[int]:
        """
        Get indices of trajectories that failed verification (DoA).

        Args:
            verification_results: Results from batch_verify

        Returns:
            List of indices that failed hard checks
        """
        return [
            i for i, result in enumerate(verification_results)
            if result.any_hard_failure
        ]
