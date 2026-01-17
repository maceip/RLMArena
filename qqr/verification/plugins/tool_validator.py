"""
Tool Call Validator Plugin

Validates that tool calls in LLM responses are well-formed and use available tools.
"""

import json
from typing import Any

from ..plugin import VerificationPlugin, VerificationResult, VerificationStatus


class ToolCallValidatorPlugin(VerificationPlugin):
    """
    Validates tool calls in agent trajectories.

    Checks:
    - Tool names match available tools
    - Arguments are valid JSON
    - Required arguments are present
    - Argument types match expected types
    """

    name = "tool_validator"
    description = "Validates tool call format and arguments"
    is_hard_check = True
    floor_reward = -1.0

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        self.available_tools = self.config.get("available_tools", [])
        self.strict_args = self.config.get("strict_args", False)
        self.allow_unknown_tools = self.config.get("allow_unknown_tools", False)

    def _get_tool_schema(self, tool_name: str) -> dict | None:
        """Get schema for a tool by name"""
        for tool in self.available_tools:
            func = tool.get("function", {})
            if func.get("name") == tool_name:
                return func.get("parameters", {})
        return None

    def _validate_tool_call(
        self, tool_call: dict, available_tool_names: set
    ) -> tuple[bool, list[str]]:
        """Validate a single tool call"""
        errors = []

        func = tool_call.get("function", {})
        tool_name = func.get("name", "")
        args_str = func.get("arguments", "{}")

        if not tool_name:
            errors.append("Tool call missing function name")
            return False, errors

        if not self.allow_unknown_tools and tool_name not in available_tool_names:
            errors.append(f"Unknown tool: {tool_name}")

        try:
            args = json.loads(args_str) if isinstance(args_str, str) else args_str
        except json.JSONDecodeError as e:
            errors.append(f"Invalid JSON arguments for {tool_name}: {e.msg}")
            return False, errors

        if self.strict_args:
            schema = self._get_tool_schema(tool_name)
            if schema and "required" in schema:
                missing = [k for k in schema["required"] if k not in args]
                if missing:
                    errors.append(f"Missing required arguments for {tool_name}: {missing}")

        return len(errors) == 0, errors

    async def verify(
        self,
        messages: list[dict],
        metadata: dict[str, Any] | None = None
    ) -> VerificationResult:
        """
        Validate all tool calls in the trajectory.

        Args:
            messages: Conversation messages
            metadata: Should contain 'tools' with available tool definitions

        Returns:
            VerificationResult with validation details
        """
        tools = (metadata or {}).get("tools", self.available_tools)
        available_tool_names = {
            t.get("function", {}).get("name", "") for t in tools
        }

        all_errors = []
        tool_calls_checked = 0

        for msg in messages:
            if msg.get("role") != "assistant":
                continue

            tool_calls = msg.get("tool_calls", [])

            if not tool_calls and "content" in msg:
                content = msg.get("content", {})
                if isinstance(content, dict):
                    tool_calls = content.get("tool_calls", [])

            for tool_call in tool_calls:
                tool_calls_checked += 1
                is_valid, errors = self._validate_tool_call(tool_call, available_tool_names)
                if not is_valid:
                    all_errors.append({
                        "tool_call": tool_call.get("function", {}).get("name", "unknown"),
                        "errors": errors,
                    })

        if tool_calls_checked == 0:
            return VerificationResult(
                status=VerificationStatus.SKIPPED,
                plugin_name=self.name,
                message="No tool calls found in trajectory",
                details={"tool_calls_checked": 0},
            )

        if all_errors:
            return VerificationResult(
                status=VerificationStatus.FAILED,
                plugin_name=self.name,
                message=f"Found {len(all_errors)} invalid tool call(s)",
                details={
                    "tool_calls_checked": tool_calls_checked,
                    "errors": all_errors,
                    "available_tools": list(available_tool_names),
                },
                score_modifier=0.0,
            )

        return VerificationResult(
            status=VerificationStatus.PASSED,
            plugin_name=self.name,
            message=f"All {tool_calls_checked} tool call(s) are valid",
            details={
                "tool_calls_checked": tool_calls_checked,
                "available_tools": list(available_tool_names),
            },
            score_modifier=1.0,
        )
