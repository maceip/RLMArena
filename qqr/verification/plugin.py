"""
Verification Plugin Infrastructure for VaaS

This module defines the base classes for deterministic verification plugins
that run before the LLM-based tournament judging. Plugins implement "Hard Checks"
that can immediately disqualify trajectories that fail constraint satisfaction.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class VerificationStatus(Enum):
    """Status of a verification check"""
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    ERROR = "error"


@dataclass
class VerificationResult:
    """Result of a verification plugin check"""
    status: VerificationStatus
    plugin_name: str
    message: str = ""
    details: dict[str, Any] = field(default_factory=dict)
    score_modifier: float = 1.0

    @property
    def passed(self) -> bool:
        return self.status == VerificationStatus.PASSED

    @property
    def failed(self) -> bool:
        return self.status == VerificationStatus.FAILED

    def to_dict(self) -> dict:
        return {
            "status": self.status.value,
            "plugin_name": self.plugin_name,
            "message": self.message,
            "details": self.details,
            "score_modifier": self.score_modifier,
        }

    @staticmethod
    def from_dict(data: dict) -> "VerificationResult":
        return VerificationResult(
            status=VerificationStatus(data["status"]),
            plugin_name=data["plugin_name"],
            message=data.get("message", ""),
            details=data.get("details", {}),
            score_modifier=data.get("score_modifier", 1.0),
        )


class VerificationPlugin(ABC):
    """
    Base class for all Hard Verifiers in the VaaS system.

    Verification plugins perform deterministic checks on trajectories
    before they reach the LLM-based tournament. If a trajectory fails
    a Hard Check, it receives a floor reward and bypasses expensive
    tournament logic.

    Plugins can be:
    - Code linters (syntax validation)
    - API schema validators
    - Safety filters
    - Tool call validators
    - Custom constraint checkers
    """

    name: str = "base_plugin"
    description: str = "Base verification plugin"
    is_hard_check: bool = True
    floor_reward: float = -1.0

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}

    @abstractmethod
    async def verify(
        self,
        messages: list[dict],
        metadata: dict[str, Any] | None = None
    ) -> VerificationResult:
        """
        Verify a trajectory against this plugin's constraints.

        Args:
            messages: The conversation/trajectory messages in OpenAI format
            metadata: Optional metadata about the trajectory (tools, query, etc.)

        Returns:
            VerificationResult with status, message, and optional details
        """
        pass

    async def verify_batch(
        self,
        trajectories: list[list[dict]],
        metadata: list[dict[str, Any]] | None = None
    ) -> list[VerificationResult]:
        """
        Verify multiple trajectories. Override for batch-optimized implementations.

        Args:
            trajectories: List of message lists
            metadata: Optional list of metadata dicts (one per trajectory)

        Returns:
            List of VerificationResults
        """
        metadata = metadata or [{}] * len(trajectories)
        results = []
        for msgs, meta in zip(trajectories, metadata):
            result = await self.verify(msgs, meta)
            results.append(result)
        return results

    def should_skip(self, messages: list[dict], metadata: dict[str, Any] | None = None) -> bool:
        """
        Determine if this plugin should be skipped for a given trajectory.
        Override to implement conditional verification logic.
        """
        return False


@dataclass
class AggregateVerificationResult:
    """Aggregated result from multiple verification plugins"""
    results: list[VerificationResult]

    @property
    def all_passed(self) -> bool:
        return all(r.passed or r.status == VerificationStatus.SKIPPED for r in self.results)

    @property
    def any_hard_failure(self) -> bool:
        return any(r.failed for r in self.results)

    @property
    def combined_score_modifier(self) -> float:
        modifier = 1.0
        for r in self.results:
            if r.passed:
                modifier *= r.score_modifier
            elif r.failed:
                return 0.0
        return modifier

    @property
    def failure_messages(self) -> list[str]:
        return [r.message for r in self.results if r.failed]

    def to_dict(self) -> dict:
        return {
            "results": [r.to_dict() for r in self.results],
            "all_passed": self.all_passed,
            "any_hard_failure": self.any_hard_failure,
            "combined_score_modifier": self.combined_score_modifier,
        }
