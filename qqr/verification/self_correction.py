"""
Recursive Self-Correction Module

Provides mechanisms for LLM agents to self-correct when verification
identifies constraint or syntax errors in their outputs.
"""

from dataclasses import dataclass, field
from typing import Any

from .plugin import AggregateVerificationResult, VerificationResult, VerificationStatus


@dataclass
class CorrectionContext:
    """Context for a self-correction attempt"""
    attempt_number: int
    max_attempts: int
    original_response: str
    verification_result: AggregateVerificationResult | VerificationResult
    correction_history: list[dict] = field(default_factory=list)

    @property
    def should_retry(self) -> bool:
        return self.attempt_number < self.max_attempts

    def add_to_history(self, response: str, result: AggregateVerificationResult | VerificationResult):
        self.correction_history.append({
            "attempt": self.attempt_number,
            "response": response,
            "result": result.to_dict() if hasattr(result, "to_dict") else str(result),
        })


class SelfCorrectionEngine:
    """
    Engine for recursive self-correction of LLM outputs.

    When verification identifies errors, this engine:
    1. Formats error feedback as a system/tool message
    2. Appends feedback to conversation
    3. Triggers re-generation
    4. Re-runs verification
    5. Repeats until pass or max attempts reached
    """

    DEFAULT_CORRECTION_PROMPT = """Your previous response failed verification checks. Please correct the following issues:

{errors}

Provide a corrected response that addresses all the issues above."""

    SYSTEM_CORRECTION_PROMPT = """[VERIFICATION FAILED]
The following issues were detected in your response:
{errors}

Please provide a corrected response."""

    def __init__(
        self,
        max_correction_attempts: int = 3,
        correction_prompt_template: str | None = None,
        use_system_message: bool = False,
        include_error_details: bool = True,
    ):
        """
        Initialize the self-correction engine.

        Args:
            max_correction_attempts: Maximum number of correction attempts
            correction_prompt_template: Custom template for correction prompts
            use_system_message: If True, use system role; else use tool role
            include_error_details: If True, include detailed error info
        """
        self.max_correction_attempts = max_correction_attempts
        self.correction_prompt_template = (
            correction_prompt_template or self.DEFAULT_CORRECTION_PROMPT
        )
        self.use_system_message = use_system_message
        self.include_error_details = include_error_details

    def format_errors(
        self, verification_result: AggregateVerificationResult | VerificationResult
    ) -> str:
        """Format verification errors into a human-readable string"""
        if isinstance(verification_result, VerificationResult):
            if verification_result.failed:
                error_lines = [f"- {verification_result.plugin_name}: {verification_result.message}"]
                if self.include_error_details and verification_result.details:
                    for key, value in verification_result.details.items():
                        if key != "errors":
                            continue
                        if isinstance(value, list):
                            for err in value[:3]:
                                error_lines.append(f"  - {err}")
                return "\n".join(error_lines)
            return ""

        error_lines = []
        for result in verification_result.results:
            if result.failed:
                error_lines.append(f"- {result.plugin_name}: {result.message}")
                if self.include_error_details and result.details:
                    errors = result.details.get("errors", [])
                    for err in errors[:3]:
                        if isinstance(err, dict):
                            error_lines.append(f"  - {err.get('error', str(err))}")
                        else:
                            error_lines.append(f"  - {err}")

        return "\n".join(error_lines)

    def create_correction_message(
        self,
        verification_result: AggregateVerificationResult | VerificationResult,
        custom_context: str | None = None,
    ) -> dict:
        """
        Create a message to append for self-correction.

        Args:
            verification_result: The failed verification result
            custom_context: Optional custom context to include

        Returns:
            Message dict in OpenAI format
        """
        errors = self.format_errors(verification_result)
        content = self.correction_prompt_template.format(errors=errors)

        if custom_context:
            content = f"{custom_context}\n\n{content}"

        if self.use_system_message:
            return {
                "role": "system",
                "content": self.SYSTEM_CORRECTION_PROMPT.format(errors=errors),
            }
        else:
            return {
                "role": "tool",
                "content": content,
                "tool_call_id": "verification_feedback",
            }

    def should_attempt_correction(
        self,
        verification_result: AggregateVerificationResult | VerificationResult,
        attempt_number: int,
    ) -> bool:
        """
        Determine if correction should be attempted.

        Args:
            verification_result: The verification result
            attempt_number: Current attempt number (0-indexed)

        Returns:
            True if correction should be attempted
        """
        if attempt_number >= self.max_correction_attempts:
            return False

        if isinstance(verification_result, VerificationResult):
            return verification_result.failed

        return verification_result.any_hard_failure

    def create_context(
        self,
        original_response: str,
        verification_result: AggregateVerificationResult | VerificationResult,
        attempt_number: int = 0,
    ) -> CorrectionContext:
        """
        Create a correction context for tracking attempts.

        Args:
            original_response: The original LLM response
            verification_result: The verification result
            attempt_number: Starting attempt number

        Returns:
            CorrectionContext instance
        """
        return CorrectionContext(
            attempt_number=attempt_number,
            max_attempts=self.max_correction_attempts,
            original_response=original_response,
            verification_result=verification_result,
        )


def inject_correction_into_messages(
    messages: list[dict],
    correction_message: dict,
    insert_before_last_assistant: bool = True,
) -> list[dict]:
    """
    Inject a correction message into the conversation.

    Args:
        messages: Current conversation messages
        correction_message: The correction message to inject
        insert_before_last_assistant: If True, insert before last assistant msg

    Returns:
        Updated messages list
    """
    messages = messages.copy()

    if insert_before_last_assistant:
        for i in range(len(messages) - 1, -1, -1):
            if messages[i].get("role") == "assistant":
                messages.insert(i, correction_message)
                return messages

    messages.append(correction_message)
    return messages
