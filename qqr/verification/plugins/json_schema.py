"""
JSON Schema Verification Plugin

Validates that LLM responses conform to expected JSON schemas.
"""

import json
import re
from typing import Any

from ..plugin import VerificationPlugin, VerificationResult, VerificationStatus


class JSONSchemaPlugin(VerificationPlugin):
    """
    Verifies JSON outputs in responses conform to specified schemas.

    Useful for validating structured outputs from LLMs, API responses,
    and tool call arguments.
    """

    name = "json_schema"
    description = "Validates JSON outputs against schemas"
    is_hard_check = True
    floor_reward = -1.0

    JSON_BLOCK_PATTERN = re.compile(r"```(?:json)?\s*\n(.*?)```", re.DOTALL)
    INLINE_JSON_PATTERN = re.compile(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}")

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        self.schemas = self.config.get("schemas", {})
        self.require_valid_json = self.config.get("require_valid_json", True)
        self.extract_inline = self.config.get("extract_inline", False)

    def _extract_json_blocks(self, content: str) -> list[str]:
        """Extract JSON from markdown code blocks"""
        matches = self.JSON_BLOCK_PATTERN.findall(content)
        blocks = [m.strip() for m in matches if m.strip()]

        if self.extract_inline and not blocks:
            inline_matches = self.INLINE_JSON_PATTERN.findall(content)
            blocks.extend(inline_matches)

        return blocks

    def _validate_json(self, json_str: str) -> tuple[bool, Any, str]:
        """Parse and validate JSON string"""
        try:
            parsed = json.loads(json_str)
            return True, parsed, ""
        except json.JSONDecodeError as e:
            return False, None, f"Invalid JSON: {e.msg} at position {e.pos}"

    def _validate_against_schema(
        self, data: Any, schema: dict[str, Any]
    ) -> tuple[bool, list[str]]:
        """Simple schema validation without jsonschema dependency"""
        errors = []

        if "type" in schema:
            expected_type = schema["type"]
            type_map = {
                "object": dict,
                "array": list,
                "string": str,
                "number": (int, float),
                "integer": int,
                "boolean": bool,
                "null": type(None),
            }
            if expected_type in type_map:
                if not isinstance(data, type_map[expected_type]):
                    errors.append(f"Expected type {expected_type}, got {type(data).__name__}")
                    return False, errors

        if isinstance(data, dict) and "required" in schema:
            missing = [k for k in schema["required"] if k not in data]
            if missing:
                errors.append(f"Missing required fields: {missing}")

        if isinstance(data, dict) and "properties" in schema:
            for key, prop_schema in schema["properties"].items():
                if key in data:
                    valid, prop_errors = self._validate_against_schema(data[key], prop_schema)
                    errors.extend([f"{key}.{e}" for e in prop_errors])

        return len(errors) == 0, errors

    async def verify(
        self,
        messages: list[dict],
        metadata: dict[str, Any] | None = None
    ) -> VerificationResult:
        """
        Verify JSON content in assistant messages.

        Args:
            messages: Conversation messages
            metadata: May contain 'expected_schema' for validation

        Returns:
            VerificationResult with validation details
        """
        expected_schema = (metadata or {}).get("expected_schema", self.schemas.get("default"))
        all_json_blocks = []
        errors = []
        parsed_objects = []

        for msg in messages:
            if msg.get("role") != "assistant":
                continue

            content = msg.get("content", "")
            if isinstance(content, dict):
                content = content.get("content", "")

            json_blocks = self._extract_json_blocks(content)
            all_json_blocks.extend(json_blocks)

            for i, json_str in enumerate(json_blocks):
                is_valid, parsed, error_msg = self._validate_json(json_str)

                if not is_valid:
                    errors.append({
                        "block_index": i,
                        "type": "parse_error",
                        "error": error_msg,
                        "preview": json_str[:100] + "..." if len(json_str) > 100 else json_str,
                    })
                    continue

                parsed_objects.append(parsed)

                if expected_schema:
                    schema_valid, schema_errors = self._validate_against_schema(
                        parsed, expected_schema
                    )
                    if not schema_valid:
                        errors.append({
                            "block_index": i,
                            "type": "schema_error",
                            "errors": schema_errors,
                        })

        if not all_json_blocks:
            return VerificationResult(
                status=VerificationStatus.SKIPPED,
                plugin_name=self.name,
                message="No JSON blocks found",
                details={"blocks_checked": 0},
            )

        if errors:
            return VerificationResult(
                status=VerificationStatus.FAILED,
                plugin_name=self.name,
                message=f"Found {len(errors)} JSON validation error(s)",
                details={
                    "blocks_checked": len(all_json_blocks),
                    "errors": errors,
                },
                score_modifier=0.0,
            )

        return VerificationResult(
            status=VerificationStatus.PASSED,
            plugin_name=self.name,
            message=f"All {len(all_json_blocks)} JSON block(s) are valid",
            details={
                "blocks_checked": len(all_json_blocks),
                "objects_parsed": len(parsed_objects),
            },
            score_modifier=1.0,
        )
