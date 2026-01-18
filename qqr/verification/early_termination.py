"""
Early Termination Module for Cost Optimization

Implements early termination strategies for the ShadowArena to save tokens
and compute costs:

1. If a trajectory fails the HardVerifier, immediately kill the other N-1
   parallel samples for that branch
2. Progressive termination based on verification confidence
3. Budget-aware termination when token limits are approached
4. Quality-threshold termination when a "good enough" response is found

This can save 50-80% of token costs in typical scenarios where early
verification failures are common.
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Optional


class TerminationReason(Enum):
    """Reasons for early termination."""
    HARD_VERIFICATION_FAILURE = "hard_verification_failure"
    QUALITY_THRESHOLD_MET = "quality_threshold_met"
    BUDGET_EXCEEDED = "budget_exceeded"
    TIMEOUT = "timeout"
    ALL_FAILED = "all_failed"
    SUFFICIENT_VALID = "sufficient_valid"
    USER_REQUESTED = "user_requested"


@dataclass
class TerminationConfig:
    """Configuration for early termination."""
    enable_hard_fail_termination: bool = True
    enable_quality_termination: bool = True
    enable_budget_termination: bool = True
    quality_threshold: float = 0.9  # Score threshold for "good enough"
    min_valid_before_termination: int = 1  # Minimum valid responses needed
    max_token_budget: int = 100000  # Maximum tokens to spend
    termination_delay_ms: float = 100.0  # Delay before killing tasks
    aggressive_mode: bool = False  # If True, terminate on first hard fail


@dataclass
class TerminationEvent:
    """Record of a termination event."""
    reason: TerminationReason
    terminated_count: int
    saved_tokens_estimate: int
    timestamp: datetime = field(default_factory=datetime.utcnow)
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "reason": self.reason.value,
            "terminated_count": self.terminated_count,
            "saved_tokens_estimate": self.saved_tokens_estimate,
            "timestamp": self.timestamp.isoformat(),
            "details": self.details,
        }


@dataclass
class TerminationStats:
    """Statistics for early termination."""
    total_requests: int = 0
    total_terminated: int = 0
    total_tokens_saved: int = 0
    termination_events: list[TerminationEvent] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "total_requests": self.total_requests,
            "total_terminated": self.total_terminated,
            "total_tokens_saved": self.total_tokens_saved,
            "termination_rate": (
                self.total_terminated / max(1, self.total_requests)
            ),
            "recent_events": [e.to_dict() for e in self.termination_events[-10:]],
        }


class EarlyTerminator:
    """
    Manages early termination of parallel generation tasks.

    Monitors verification results and terminates remaining tasks when:
    - A hard verification failure is detected
    - Quality threshold is met
    - Budget is exceeded
    """

    def __init__(self, config: Optional[TerminationConfig] = None):
        self.config = config or TerminationConfig()
        self._stats = TerminationStats()
        self._active_tasks: dict[str, list[asyncio.Task]] = {}
        self._verification_results: dict[str, list[Any]] = {}
        self._terminated_sessions: set[str] = set()

    def register_session(
        self,
        session_id: str,
        tasks: list[asyncio.Task],
    ) -> None:
        """Register a session with its parallel tasks."""
        self._active_tasks[session_id] = tasks
        self._verification_results[session_id] = []
        self._stats.total_requests += 1

    def report_verification_result(
        self,
        session_id: str,
        task_index: int,
        result: Any,
        is_hard_check: bool = True,
    ) -> Optional[TerminationEvent]:
        """
        Report a verification result and check for termination.

        Returns TerminationEvent if termination is triggered, None otherwise.
        """
        if session_id not in self._active_tasks:
            return None

        if session_id in self._terminated_sessions:
            return None

        results = self._verification_results.get(session_id, [])

        # Store result
        while len(results) <= task_index:
            results.append(None)
        results[task_index] = result
        self._verification_results[session_id] = results

        # Check for termination conditions
        if is_hard_check and self.config.enable_hard_fail_termination:
            # Check if this is a hard failure
            passed = self._check_result_passed(result)

            if not passed:
                return self._trigger_termination(
                    session_id=session_id,
                    exclude_index=task_index,
                    reason=TerminationReason.HARD_VERIFICATION_FAILURE,
                    details={"failed_index": task_index},
                )

        # Check for quality threshold
        if self.config.enable_quality_termination:
            score = self._extract_score(result)
            if score is not None and score >= self.config.quality_threshold:
                valid_count = sum(
                    1 for r in results
                    if r is not None and self._check_result_passed(r)
                )
                if valid_count >= self.config.min_valid_before_termination:
                    return self._trigger_termination(
                        session_id=session_id,
                        exclude_index=task_index,
                        reason=TerminationReason.QUALITY_THRESHOLD_MET,
                        details={
                            "score": score,
                            "threshold": self.config.quality_threshold,
                        },
                    )

        return None

    def report_token_usage(
        self,
        session_id: str,
        tokens_used: int,
    ) -> Optional[TerminationEvent]:
        """
        Report token usage and check for budget termination.
        """
        if not self.config.enable_budget_termination:
            return None

        if session_id not in self._active_tasks:
            return None

        if session_id in self._terminated_sessions:
            return None

        if tokens_used >= self.config.max_token_budget:
            return self._trigger_termination(
                session_id=session_id,
                exclude_index=-1,  # Terminate all
                reason=TerminationReason.BUDGET_EXCEEDED,
                details={"tokens_used": tokens_used},
            )

        return None

    def _check_result_passed(self, result: Any) -> bool:
        """Check if a verification result passed."""
        if result is None:
            return False

        if hasattr(result, "all_passed"):
            return result.all_passed
        if hasattr(result, "passed"):
            return result.passed
        if isinstance(result, dict):
            return result.get("all_passed", result.get("passed", False))
        if isinstance(result, bool):
            return result

        return False

    def _extract_score(self, result: Any) -> Optional[float]:
        """Extract score from a verification result."""
        if result is None:
            return None

        if hasattr(result, "score"):
            return result.score
        if hasattr(result, "combined_score_modifier"):
            return result.combined_score_modifier
        if isinstance(result, dict):
            return result.get("score", result.get("combined_score_modifier"))

        return None

    def _trigger_termination(
        self,
        session_id: str,
        exclude_index: int,
        reason: TerminationReason,
        details: dict,
    ) -> TerminationEvent:
        """Trigger early termination for a session."""
        tasks = self._active_tasks.get(session_id, [])

        terminated_count = 0
        saved_tokens = 0

        for i, task in enumerate(tasks):
            if i == exclude_index:
                continue

            if not task.done() and not task.cancelled():
                task.cancel()
                terminated_count += 1
                # Estimate 1000 tokens per incomplete task
                saved_tokens += 1000

        self._terminated_sessions.add(session_id)

        event = TerminationEvent(
            reason=reason,
            terminated_count=terminated_count,
            saved_tokens_estimate=saved_tokens,
            details=details,
        )

        self._stats.total_terminated += terminated_count
        self._stats.total_tokens_saved += saved_tokens
        self._stats.termination_events.append(event)

        return event

    def cleanup_session(self, session_id: str) -> None:
        """Clean up session data."""
        self._active_tasks.pop(session_id, None)
        self._verification_results.pop(session_id, None)
        self._terminated_sessions.discard(session_id)

    def get_stats(self) -> dict:
        """Get termination statistics."""
        return self._stats.to_dict()


class BranchTerminator:
    """
    Terminates parallel branches based on verification results.

    This is used within brackets to kill sibling branches when
    a hard failure is detected.
    """

    def __init__(
        self,
        config: Optional[TerminationConfig] = None,
        on_termination: Optional[Callable[[TerminationEvent], None]] = None,
    ):
        self.config = config or TerminationConfig()
        self.on_termination = on_termination
        self._branch_tasks: dict[str, dict[int, asyncio.Task]] = {}
        self._branch_results: dict[str, dict[int, Any]] = {}

    def register_branch(
        self,
        query_id: str,
        branch_id: int,
        task: asyncio.Task,
    ) -> None:
        """Register a branch task."""
        if query_id not in self._branch_tasks:
            self._branch_tasks[query_id] = {}
            self._branch_results[query_id] = {}

        self._branch_tasks[query_id][branch_id] = task

    async def on_verification_complete(
        self,
        query_id: str,
        branch_id: int,
        result: Any,
    ) -> Optional[TerminationEvent]:
        """
        Handle verification completion for a branch.

        If the branch failed hard verification, terminate siblings.
        """
        if query_id not in self._branch_tasks:
            return None

        self._branch_results[query_id][branch_id] = result

        # Check if this is a hard failure
        passed = self._check_passed(result)

        if not passed and self.config.enable_hard_fail_termination:
            # In aggressive mode, terminate all other branches
            if self.config.aggressive_mode:
                event = await self._terminate_siblings(
                    query_id=query_id,
                    keep_branch_id=branch_id,
                    reason=TerminationReason.HARD_VERIFICATION_FAILURE,
                )
                if event and self.on_termination:
                    self.on_termination(event)
                return event

        return None

    async def on_high_quality_result(
        self,
        query_id: str,
        branch_id: int,
        score: float,
    ) -> Optional[TerminationEvent]:
        """
        Handle a high-quality result that may trigger early termination.
        """
        if not self.config.enable_quality_termination:
            return None

        if score >= self.config.quality_threshold:
            event = await self._terminate_siblings(
                query_id=query_id,
                keep_branch_id=branch_id,
                reason=TerminationReason.QUALITY_THRESHOLD_MET,
            )
            if event and self.on_termination:
                self.on_termination(event)
            return event

        return None

    def _check_passed(self, result: Any) -> bool:
        """Check if verification passed."""
        if result is None:
            return False
        if hasattr(result, "all_passed"):
            return result.all_passed
        if hasattr(result, "passed"):
            return result.passed
        if isinstance(result, dict):
            return result.get("all_passed", result.get("passed", False))
        return False

    async def _terminate_siblings(
        self,
        query_id: str,
        keep_branch_id: int,
        reason: TerminationReason,
    ) -> TerminationEvent:
        """Terminate all sibling branches."""
        branches = self._branch_tasks.get(query_id, {})
        terminated_count = 0
        saved_tokens = 0

        for branch_id, task in branches.items():
            if branch_id == keep_branch_id:
                continue

            if not task.done() and not task.cancelled():
                task.cancel()
                terminated_count += 1
                saved_tokens += 1000  # Estimate

        # Wait briefly for cancellation
        await asyncio.sleep(self.config.termination_delay_ms / 1000)

        return TerminationEvent(
            reason=reason,
            terminated_count=terminated_count,
            saved_tokens_estimate=saved_tokens,
            details={
                "query_id": query_id,
                "kept_branch": keep_branch_id,
            },
        )

    def cleanup_query(self, query_id: str) -> None:
        """Clean up query data."""
        self._branch_tasks.pop(query_id, None)
        self._branch_results.pop(query_id, None)


async def run_with_early_termination(
    tasks: list[Callable],
    verify_fn: Callable,
    config: Optional[TerminationConfig] = None,
) -> tuple[list[Any], TerminationStats]:
    """
    Run tasks with early termination support.

    Args:
        tasks: List of async callables to run
        verify_fn: Function to verify each result (returns bool)
        config: Termination configuration

    Returns:
        Tuple of (results, stats)
    """
    config = config or TerminationConfig()
    terminator = EarlyTerminator(config)
    session_id = f"session_{id(tasks)}"

    # Create tasks
    async_tasks = [asyncio.create_task(t()) for t in tasks]
    terminator.register_session(session_id, async_tasks)

    results = [None] * len(tasks)
    completed = 0

    try:
        for coro in asyncio.as_completed(async_tasks):
            try:
                result = await coro
                task_idx = async_tasks.index(coro._coro.__self__)  # type: ignore

                results[task_idx] = result
                completed += 1

                # Verify and check for termination
                verification_result = await verify_fn(result)
                event = terminator.report_verification_result(
                    session_id=session_id,
                    task_index=task_idx,
                    result=verification_result,
                )

                if event:
                    # Termination triggered
                    break

            except asyncio.CancelledError:
                pass
            except Exception as e:
                # Task failed, report as failed verification
                task_idx = next(
                    i for i, t in enumerate(async_tasks)
                    if t.done() and not results[i]
                )
                results[task_idx] = e
                terminator.report_verification_result(
                    session_id=session_id,
                    task_index=task_idx,
                    result={"passed": False, "error": str(e)},
                )

    finally:
        terminator.cleanup_session(session_id)

    return results, terminator._stats
