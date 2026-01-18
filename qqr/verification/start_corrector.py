"""
START Self-Correction Engine.

Implements the Self-Taught Reasoner with Tools (START) approach for
automatic correction of failed trajectories based on verification feedback.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Optional, Protocol
import asyncio
import json
import uuid


class CorrectionStatus(Enum):
    """Status of a correction attempt."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"
    MAX_ATTEMPTS_REACHED = "max_attempts"


@dataclass
class CorrectionConfig:
    """Configuration for the self-correction engine."""
    max_attempts: int = 3
    backtrack_on_hard_failure: bool = True
    include_rationale: bool = True
    feedback_format: str = "structured"  # "structured", "natural", "minimal"
    correction_temperature: float = 0.3
    allow_partial_success: bool = True
    timeout_per_attempt_seconds: float = 30.0


@dataclass
class FailureContext:
    """Context about a verification failure for correction."""
    plugin_name: str
    failure_message: str
    failure_details: dict[str, Any]
    trajectory_point: Optional[int] = None
    severity: int = 5  # 1-10 scale
    suggested_fix: Optional[str] = None


@dataclass
class CorrectionAttempt:
    """Record of a single correction attempt."""
    attempt_number: int
    timestamp: datetime
    failure_contexts: list[FailureContext]
    correction_prompt: str
    corrected_messages: list[dict[str, Any]]
    verification_passed: bool
    remaining_failures: list[FailureContext] = field(default_factory=list)
    latency_ms: float = 0.0


@dataclass
class CorrectionTrajectory:
    """Complete trajectory of correction attempts."""
    id: str
    original_messages: list[dict[str, Any]]
    attempts: list[CorrectionAttempt]
    final_status: CorrectionStatus
    final_messages: Optional[list[dict[str, Any]]] = None
    total_attempts: int = 0
    total_latency_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_training_data(self) -> Optional[dict[str, Any]]:
        """Convert to training data format if correction was successful."""
        if self.final_status not in (CorrectionStatus.SUCCESS, CorrectionStatus.PARTIAL):
            return None

        return {
            "id": self.id,
            "original": self.original_messages,
            "corrected": self.final_messages,
            "attempts": self.total_attempts,
            "failure_types": [
                ctx.plugin_name
                for attempt in self.attempts
                for ctx in attempt.failure_contexts
            ],
            "correction_chain": [
                {
                    "failures": [ctx.failure_message for ctx in attempt.failure_contexts],
                    "prompt": attempt.correction_prompt,
                    "success": attempt.verification_passed,
                }
                for attempt in self.attempts
            ],
        }


class BacktrackHook:
    """
    Hook that triggers on hard verifier failures.

    Intercepts failed trajectories and initiates the correction process.
    """

    def __init__(
        self,
        corrector: "STARTCorrectionEngine",
        trigger_plugins: Optional[list[str]] = None,
    ):
        self.corrector = corrector
        self.trigger_plugins = trigger_plugins or [
            "leash_verifier",
            "opa_verifier",
            "code_linter",
            "runtime_verifier",
        ]

    def should_trigger(self, verification_result: dict[str, Any]) -> bool:
        """Check if this failure should trigger correction."""
        if verification_result.get("status") != "failed":
            return False

        plugin_name = verification_result.get("plugin_name", "")
        return plugin_name in self.trigger_plugins

    async def on_failure(
        self,
        messages: list[dict[str, Any]],
        verification_results: list[dict[str, Any]],
    ) -> Optional[CorrectionTrajectory]:
        """Handle verification failure by initiating correction."""
        # Extract failure contexts
        failure_contexts = []
        for result in verification_results:
            if result.get("status") == "failed":
                failure_contexts.append(FailureContext(
                    plugin_name=result.get("plugin_name", "unknown"),
                    failure_message=result.get("message", "Unknown failure"),
                    failure_details=result.get("details", {}),
                ))

        if not failure_contexts:
            return None

        return await self.corrector.correct(messages, failure_contexts)


class CorrectionPromptBuilder:
    """Builds correction prompts from failure contexts."""

    def __init__(self, format: str = "structured"):
        self.format = format

    def build(
        self,
        failure_contexts: list[FailureContext],
        attempt_number: int,
    ) -> str:
        """Build a correction prompt from failure contexts."""
        if self.format == "minimal":
            return self._build_minimal(failure_contexts)
        elif self.format == "natural":
            return self._build_natural(failure_contexts, attempt_number)
        else:
            return self._build_structured(failure_contexts, attempt_number)

    def _build_structured(
        self,
        failure_contexts: list[FailureContext],
        attempt_number: int,
    ) -> str:
        """Build a structured correction prompt."""
        prompt = f"""[CORRECTION REQUIRED - Attempt {attempt_number}]

The previous response failed verification. Please correct the following issues:

"""
        for i, ctx in enumerate(failure_contexts, 1):
            prompt += f"""Issue {i}: {ctx.plugin_name}
- Error: {ctx.failure_message}
- Severity: {ctx.severity}/10
"""
            if ctx.suggested_fix:
                prompt += f"- Suggested Fix: {ctx.suggested_fix}\n"

            if ctx.failure_details:
                details_str = json.dumps(ctx.failure_details, indent=2)
                if len(details_str) < 500:
                    prompt += f"- Details: {details_str}\n"

            prompt += "\n"

        prompt += """Please provide a corrected response that addresses all issues above.
Focus on fixing the errors while preserving the correct parts of your previous response."""

        return prompt

    def _build_natural(
        self,
        failure_contexts: list[FailureContext],
        attempt_number: int,
    ) -> str:
        """Build a natural language correction prompt."""
        issues = [ctx.failure_message for ctx in failure_contexts]
        issues_str = ", ".join(issues)

        if attempt_number == 1:
            return f"I noticed some issues with my previous response: {issues_str}. Let me correct that."
        else:
            return f"My correction still has problems: {issues_str}. Let me try again more carefully."

    def _build_minimal(
        self,
        failure_contexts: list[FailureContext],
    ) -> str:
        """Build a minimal correction prompt."""
        errors = [ctx.failure_message for ctx in failure_contexts]
        return f"[FIX ERRORS: {'; '.join(errors)}]"


class ModelGenerator(Protocol):
    """Protocol for model generation interface."""

    async def generate(
        self,
        messages: list[dict[str, Any]],
        temperature: float = 0.7,
    ) -> list[dict[str, Any]]:
        ...


class MockModelGenerator:
    """Mock model generator for testing."""

    async def generate(
        self,
        messages: list[dict[str, Any]],
        temperature: float = 0.7,
    ) -> list[dict[str, Any]]:
        """Generate a mock corrected response."""
        # Find the last assistant message
        last_assistant = None
        for msg in reversed(messages):
            if msg.get("role") == "assistant":
                last_assistant = msg
                break

        # Simulate a correction
        corrected_content = "Here is the corrected response addressing all verification issues."

        if last_assistant:
            # Append correction marker
            corrected_content = f"{last_assistant.get('content', '')}\n\n[CORRECTED]: Fixed the identified issues."

        return messages + [{"role": "assistant", "content": corrected_content}]


class Verifier(Protocol):
    """Protocol for verification interface."""

    async def verify(
        self,
        messages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        ...


class MockVerifier:
    """Mock verifier that simulates decreasing failure rate."""

    def __init__(self):
        self._call_count = 0

    async def verify(
        self,
        messages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Verify messages with simulated improvement."""
        self._call_count += 1

        # Simulate improving success rate with each attempt
        if self._call_count >= 2:
            return [{"status": "passed", "plugin_name": "mock", "message": "All checks passed"}]

        return [
            {
                "status": "failed",
                "plugin_name": "mock_verifier",
                "message": "Mock verification failure",
                "details": {"attempt": self._call_count},
            }
        ]


class STARTCorrectionEngine:
    """
    Self-Taught Reasoner with Tools (START) correction engine.

    Implements recursive self-correction when hard verifiers identify errors:
    1. Captures verification failure messages
    2. Injects them as correction prompts
    3. Triggers re-generation
    4. Re-verifies until success or max attempts
    5. Collects correction trajectories for training
    """

    def __init__(
        self,
        config: Optional[CorrectionConfig] = None,
        generator: Optional[ModelGenerator] = None,
        verifier: Optional[Verifier] = None,
        prompt_builder: Optional[CorrectionPromptBuilder] = None,
    ):
        self.config = config or CorrectionConfig()
        self._generator = generator or MockModelGenerator()
        self._verifier = verifier or MockVerifier()
        self._prompt_builder = prompt_builder or CorrectionPromptBuilder(
            self.config.feedback_format
        )
        self._trajectories: list[CorrectionTrajectory] = []

    async def correct(
        self,
        messages: list[dict[str, Any]],
        failure_contexts: list[FailureContext],
    ) -> CorrectionTrajectory:
        """
        Attempt to correct a failed trajectory.

        Args:
            messages: Original messages that failed verification
            failure_contexts: Information about why verification failed

        Returns:
            CorrectionTrajectory with all attempts and final result
        """
        trajectory_id = str(uuid.uuid4())
        attempts = []
        current_messages = messages.copy()
        current_failures = failure_contexts.copy()
        total_latency = 0.0

        for attempt_num in range(1, self.config.max_attempts + 1):
            start_time = datetime.utcnow()

            # Build correction prompt
            correction_prompt = self._prompt_builder.build(
                current_failures, attempt_num
            )

            # Inject correction prompt
            corrected_messages = self._inject_correction(
                current_messages, correction_prompt
            )

            # Generate corrected response
            try:
                corrected_messages = await asyncio.wait_for(
                    self._generator.generate(
                        corrected_messages,
                        temperature=self.config.correction_temperature,
                    ),
                    timeout=self.config.timeout_per_attempt_seconds,
                )
            except asyncio.TimeoutError:
                attempt = CorrectionAttempt(
                    attempt_number=attempt_num,
                    timestamp=start_time,
                    failure_contexts=current_failures,
                    correction_prompt=correction_prompt,
                    corrected_messages=[],
                    verification_passed=False,
                    latency_ms=self.config.timeout_per_attempt_seconds * 1000,
                )
                attempts.append(attempt)
                break

            # Verify corrected response
            verification_results = await self._verifier.verify(corrected_messages)

            # Check for remaining failures
            remaining_failures = self._extract_failures(verification_results)
            passed = len(remaining_failures) == 0

            latency = (datetime.utcnow() - start_time).total_seconds() * 1000
            total_latency += latency

            attempt = CorrectionAttempt(
                attempt_number=attempt_num,
                timestamp=start_time,
                failure_contexts=current_failures,
                correction_prompt=correction_prompt,
                corrected_messages=corrected_messages,
                verification_passed=passed,
                remaining_failures=remaining_failures,
                latency_ms=latency,
            )
            attempts.append(attempt)

            if passed:
                # Success!
                trajectory = CorrectionTrajectory(
                    id=trajectory_id,
                    original_messages=messages,
                    attempts=attempts,
                    final_status=CorrectionStatus.SUCCESS,
                    final_messages=corrected_messages,
                    total_attempts=attempt_num,
                    total_latency_ms=total_latency,
                )
                self._trajectories.append(trajectory)
                return trajectory

            # Check for partial success
            if (
                self.config.allow_partial_success
                and len(remaining_failures) < len(current_failures)
            ):
                # Made progress, continue
                current_messages = corrected_messages
                current_failures = remaining_failures
            else:
                # No progress or got worse
                current_messages = corrected_messages
                current_failures = remaining_failures

        # Max attempts reached
        final_status = CorrectionStatus.MAX_ATTEMPTS_REACHED

        # Check if we at least partially succeeded
        if attempts and len(attempts[-1].remaining_failures) < len(failure_contexts):
            final_status = CorrectionStatus.PARTIAL
            final_messages = attempts[-1].corrected_messages
        else:
            final_status = CorrectionStatus.FAILED
            final_messages = None

        trajectory = CorrectionTrajectory(
            id=trajectory_id,
            original_messages=messages,
            attempts=attempts,
            final_status=final_status,
            final_messages=final_messages,
            total_attempts=len(attempts),
            total_latency_ms=total_latency,
        )
        self._trajectories.append(trajectory)
        return trajectory

    def _inject_correction(
        self,
        messages: list[dict[str, Any]],
        correction_prompt: str,
    ) -> list[dict[str, Any]]:
        """Inject correction prompt into messages."""
        # Add as a system message or tool result
        correction_message = {
            "role": "user",
            "content": correction_prompt,
        }

        return messages + [correction_message]

    def _extract_failures(
        self,
        verification_results: list[dict[str, Any]],
    ) -> list[FailureContext]:
        """Extract failure contexts from verification results."""
        failures = []
        for result in verification_results:
            if result.get("status") == "failed":
                failures.append(FailureContext(
                    plugin_name=result.get("plugin_name", "unknown"),
                    failure_message=result.get("message", "Unknown failure"),
                    failure_details=result.get("details", {}),
                ))
        return failures

    def get_trajectories(self) -> list[CorrectionTrajectory]:
        """Get all recorded correction trajectories."""
        return self._trajectories.copy()

    def get_successful_trajectories(self) -> list[CorrectionTrajectory]:
        """Get only successful correction trajectories."""
        return [
            t for t in self._trajectories
            if t.final_status in (CorrectionStatus.SUCCESS, CorrectionStatus.PARTIAL)
        ]

    def export_training_data(self) -> list[dict[str, Any]]:
        """Export successful corrections as training data."""
        training_data = []
        for trajectory in self.get_successful_trajectories():
            data = trajectory.to_training_data()
            if data:
                training_data.append(data)
        return training_data

    def get_stats(self) -> dict[str, Any]:
        """Get correction statistics."""
        total = len(self._trajectories)
        if total == 0:
            return {"total": 0}

        success = sum(1 for t in self._trajectories if t.final_status == CorrectionStatus.SUCCESS)
        partial = sum(1 for t in self._trajectories if t.final_status == CorrectionStatus.PARTIAL)
        failed = sum(1 for t in self._trajectories if t.final_status == CorrectionStatus.FAILED)

        avg_attempts = sum(t.total_attempts for t in self._trajectories) / total

        return {
            "total": total,
            "success": success,
            "partial": partial,
            "failed": failed,
            "success_rate": (success + partial) / total,
            "avg_attempts": avg_attempts,
        }


class TrainingDataCollector:
    """
    Collects and manages training data from correction trajectories.

    Converts failed → corrected trajectories into training datasets
    for model fine-tuning.
    """

    def __init__(self):
        self._data: list[dict[str, Any]] = []
        self._metadata: dict[str, Any] = {
            "created_at": datetime.utcnow().isoformat(),
            "version": "1.0",
        }

    def add_trajectory(self, trajectory: CorrectionTrajectory) -> bool:
        """Add a correction trajectory to the collection."""
        data = trajectory.to_training_data()
        if data:
            self._data.append(data)
            return True
        return False

    def add_from_engine(self, engine: STARTCorrectionEngine) -> int:
        """Add all successful trajectories from an engine."""
        added = 0
        for trajectory in engine.get_successful_trajectories():
            if self.add_trajectory(trajectory):
                added += 1
        return added

    def export_jsonl(self) -> str:
        """Export as JSONL format."""
        lines = [json.dumps(item) for item in self._data]
        return "\n".join(lines)

    def export_dpo_pairs(self) -> list[dict[str, Any]]:
        """Export as DPO training pairs (chosen/rejected)."""
        pairs = []
        for item in self._data:
            pairs.append({
                "prompt": item["original"][0].get("content", "") if item["original"] else "",
                "chosen": item["corrected"],
                "rejected": item["original"],
                "metadata": {
                    "correction_attempts": item["attempts"],
                    "failure_types": item["failure_types"],
                },
            })
        return pairs

    def get_stats(self) -> dict[str, Any]:
        """Get collection statistics."""
        if not self._data:
            return {"count": 0}

        failure_types: dict[str, int] = {}
        for item in self._data:
            for ft in item.get("failure_types", []):
                failure_types[ft] = failure_types.get(ft, 0) + 1

        return {
            "count": len(self._data),
            "failure_types": failure_types,
            "avg_attempts": sum(item["attempts"] for item in self._data) / len(self._data),
        }
