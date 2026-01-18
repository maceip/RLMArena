"""
Code Linter Verification Plugin

Validates Python code snippets in LLM responses for syntax errors.
"""

import ast
import re
from typing import Any

from ..plugin import VerificationPlugin, VerificationResult, VerificationStatus


class CodeLinterPlugin(VerificationPlugin):
    """
    Verifies Python code snippets in responses are syntactically valid.

    This is a Hard Check - trajectories with syntax errors in their
    code outputs are considered Dead on Arrival (DoA).
    """

    name = "code_linter"
    description = "Validates Python code syntax in responses"
    is_hard_check = True
    floor_reward = -1.0

    CODE_BLOCK_PATTERN = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL)

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        self.strict_mode = self.config.get("strict_mode", False)
        self.check_imports = self.config.get("check_imports", False)

    def _extract_code_blocks(self, content: str) -> list[str]:
        """Extract Python code blocks from markdown-formatted content"""
        matches = self.CODE_BLOCK_PATTERN.findall(content)
        return [m.strip() for m in matches if m.strip()]

    def _validate_syntax(self, code: str) -> tuple[bool, str]:
        """Validate Python syntax using ast.parse"""
        try:
            ast.parse(code)
            return True, ""
        except SyntaxError as e:
            return False, f"Syntax error at line {e.lineno}: {e.msg}"
        except Exception as e:
            return False, f"Parse error: {str(e)}"

    async def verify(
        self,
        messages: list[dict],
        metadata: dict[str, Any] | None = None
    ) -> VerificationResult:
        """
        Verify all Python code blocks in assistant messages.

        Args:
            messages: Conversation messages
            metadata: Optional metadata

        Returns:
            VerificationResult indicating pass/fail with details
        """
        all_code_blocks = []
        errors = []

        for msg in messages:
            if msg.get("role") != "assistant":
                continue

            content = msg.get("content", "")
            if isinstance(content, dict):
                content = content.get("content", "")

            code_blocks = self._extract_code_blocks(content)
            all_code_blocks.extend(code_blocks)

            for i, code in enumerate(code_blocks):
                is_valid, error_msg = self._validate_syntax(code)
                if not is_valid:
                    errors.append({
                        "block_index": i,
                        "code_preview": code[:100] + "..." if len(code) > 100 else code,
                        "error": error_msg,
                    })

        if not all_code_blocks:
            return VerificationResult(
                status=VerificationStatus.SKIPPED,
                plugin_name=self.name,
                message="No Python code blocks found",
                details={"blocks_checked": 0},
            )

        if errors:
            return VerificationResult(
                status=VerificationStatus.FAILED,
                plugin_name=self.name,
                message=f"Found {len(errors)} syntax error(s) in code blocks",
                details={
                    "blocks_checked": len(all_code_blocks),
                    "errors": errors,
                },
                score_modifier=0.0,
            )

        return VerificationResult(
            status=VerificationStatus.PASSED,
            plugin_name=self.name,
            message=f"All {len(all_code_blocks)} code block(s) have valid syntax",
            details={"blocks_checked": len(all_code_blocks)},
            score_modifier=1.0,
        )
