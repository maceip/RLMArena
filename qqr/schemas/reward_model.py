from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any


class VerificationStatus(Enum):
    """Status codes for trajectory verification"""
    VALID = "valid"
    INVALID = "invalid"
    SKIPPED = "skipped"
    ERROR = "error"


@dataclass
class ValidationResult:
    """Result of trajectory validation"""
    status: VerificationStatus
    message: str = ""
    details: dict[str, Any] | None = None
    floor_reward: float | None = None

    @property
    def is_valid(self) -> bool:
        return self.status == VerificationStatus.VALID

    @property
    def is_doa(self) -> bool:
        """Dead on Arrival - failed hard checks"""
        return self.status == VerificationStatus.INVALID


class RewardModel(ABC):
    """
    Abstract base class for single-sample reward computation.

    Reward models evaluate individual predictions against optional
    references to produce scalar or multi-dimensional rewards.
    """

    floor_reward: float = -1.0

    async def __call__(self, *args, **kwargs) -> float | dict[str, float]:
        return await self.compute(*args, **kwargs)

    @abstractmethod
    async def compute(
        self, prediction, reference=None, *args, **kwargs
    ) -> float | dict[str, float]:
        """
        Compute reward for a single prediction.

        Args:
            prediction: The model output to evaluate
            reference: Optional reference/ground truth
            *args, **kwargs: Additional arguments

        Returns:
            Scalar reward or dict of reward components
        """
        ...

    async def validate(
        self, prediction, metadata: dict[str, Any] | None = None
    ) -> ValidationResult:
        """
        Validate a prediction before reward computation.

        Override this method to implement hard checks that run before
        the expensive reward computation. Failed validation results in
        immediate assignment of floor_reward.

        Args:
            prediction: The prediction to validate
            metadata: Optional metadata about the prediction

        Returns:
            ValidationResult indicating pass/fail with details
        """
        return ValidationResult(
            status=VerificationStatus.VALID,
            message="Validation not implemented - defaulting to valid",
        )


class GroupRewardModel(ABC):
    """
    Abstract base class for group/batch reward computation.

    Group reward models compare multiple predictions simultaneously,
    typically using tournament-style ranking algorithms to produce
    relative rewards across the group.
    """

    floor_reward: float = -1.0

    async def __call__(self, *args, **kwargs) -> list[float] | list[dict[str, float]]:
        return await self.compute(*args, **kwargs)

    @abstractmethod
    async def compute(
        self, predictions: list, reference=None, *args, **kwargs
    ) -> list[float] | list[dict[str, float]]:
        """
        Compute rewards for a group of predictions.

        Args:
            predictions: List of model outputs to evaluate
            reference: Optional reference/ground truth
            *args, **kwargs: Additional arguments

        Returns:
            List of rewards matching the prediction count
        """
        ...

    async def validate_batch(
        self, predictions: list, metadata: list[dict[str, Any]] | None = None
    ) -> list[ValidationResult]:
        """
        Validate all predictions before group reward computation.

        Args:
            predictions: List of predictions to validate
            metadata: Optional list of metadata dicts

        Returns:
            List of ValidationResult for each prediction
        """
        metadata = metadata or [None] * len(predictions)
        return [
            ValidationResult(
                status=VerificationStatus.VALID,
                message="Validation not implemented",
            )
            for _ in predictions
        ]

    def apply_floor_rewards(
        self,
        rewards: list[float],
        validation_results: list[ValidationResult]
    ) -> list[float]:
        """
        Apply floor rewards to predictions that failed validation.

        Args:
            rewards: Original computed rewards
            validation_results: Validation results for each prediction

        Returns:
            Rewards with floor values applied for failed validations
        """
        return [
            self.floor_reward if v.is_doa else r
            for r, v in zip(rewards, validation_results)
        ]
