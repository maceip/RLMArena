from abc import ABC, abstractmethod
from typing import Any


class LLMJudge(ABC):
    """
    Abstract base class for LLM-based judges.

    Judges perform pairwise comparisons of trajectories to determine
    relative quality scores. They form the core of the arena tournament
    system for ranking LLM outputs.
    """

    @abstractmethod
    async def compare(
        self, messages_a: list[dict], messages_b: list[dict], *args, **kwargs
    ) -> tuple[float, float]:
        """
        One-way comparison of two trajectories.

        Args:
            messages_a: First trajectory messages
            messages_b: Second trajectory messages
            *args, **kwargs: Additional arguments (e.g., query)

        Returns:
            Tuple of (score_a, score_b)
        """
        ...

    @abstractmethod
    async def bidirectional_compare(
        self, messages_a: list[dict], messages_b: list[dict], *args, **kwargs
    ) -> tuple[float, float, dict]:
        """
        Two-way comparison with positional bias correction.

        Performs A vs B and B vs A comparisons, aggregating results
        to reduce positional bias in LLM judgments.

        Args:
            messages_a: First trajectory messages
            messages_b: Second trajectory messages
            *args, **kwargs: Additional arguments (e.g., query)

        Returns:
            Tuple of (score_a, score_b, metadata)
        """
        ...

    async def validate(
        self, messages: list[dict], metadata: dict[str, Any] | None = None
    ) -> tuple[bool, str, dict]:
        """
        Optional validation hook for pre-tournament checks.

        Override this method to add custom validation logic that runs
        before the arena tournament. This is called by the verification
        layer when integrated with CompositeJudge.

        Args:
            messages: Trajectory messages to validate
            metadata: Optional metadata about the trajectory

        Returns:
            Tuple of (is_valid, message, details)
        """
        return True, "Validation not implemented", {}

    async def batch_compare(
        self, trajectories: list[list[dict]], query: str, **kwargs
    ) -> list[tuple[float, float]]:
        """
        Compare all pairs in a batch for tournament efficiency.

        Override for optimized batch processing in tournament systems.

        Args:
            trajectories: List of trajectory message lists
            query: The query/prompt being evaluated
            **kwargs: Additional arguments

        Returns:
            List of (score_a, score_b) for each comparison
        """
        raise NotImplementedError("batch_compare not implemented")
