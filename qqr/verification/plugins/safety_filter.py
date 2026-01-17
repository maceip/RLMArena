"""
Safety Filter Verification Plugin

Checks LLM outputs for potentially harmful or inappropriate content.
"""

import re
from typing import Any

from ..plugin import VerificationPlugin, VerificationResult, VerificationStatus


class SafetyFilterPlugin(VerificationPlugin):
    """
    Filters responses for potentially harmful content patterns.

    This plugin performs keyword and pattern-based safety checks.
    For production use, integrate with dedicated safety APIs
    (e.g., OpenAI moderation, Perspective API, custom classifiers).
    """

    name = "safety_filter"
    description = "Checks for harmful or inappropriate content"
    is_hard_check = True
    floor_reward = -1.0

    DEFAULT_PATTERNS = [
        r"(?i)\b(hack|exploit|attack)\s+(the\s+)?(system|server|network)",
        r"(?i)\b(sql|code)\s*injection",
        r"(?i)\b(bypass|circumvent)\s+(security|authentication|authorization)",
        r"(?i)\b(steal|exfiltrate)\s+(data|credentials|information)",
        r"(?i)\bmalware\b",
        r"(?i)\b(ddos|dos)\s+attack",
    ]

    DEFAULT_BLOCKLIST = [
        "rm -rf /",
        "DROP TABLE",
        "DELETE FROM",
        "; DROP",
        "TRUNCATE TABLE",
    ]

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        self.patterns = [
            re.compile(p) for p in self.config.get("patterns", self.DEFAULT_PATTERNS)
        ]
        self.blocklist = self.config.get("blocklist", self.DEFAULT_BLOCKLIST)
        self.check_tool_calls = self.config.get("check_tool_calls", True)
        self.external_api_url = self.config.get("external_api_url")

    def _check_patterns(self, text: str) -> list[dict]:
        """Check text against regex patterns"""
        matches = []
        for pattern in self.patterns:
            found = pattern.findall(text)
            if found:
                matches.append({
                    "pattern": pattern.pattern,
                    "matches": found[:5],
                })
        return matches

    def _check_blocklist(self, text: str) -> list[str]:
        """Check for blocklisted strings"""
        found = []
        text_lower = text.lower()
        for item in self.blocklist:
            if item.lower() in text_lower:
                found.append(item)
        return found

    async def verify(
        self,
        messages: list[dict],
        metadata: dict[str, Any] | None = None
    ) -> VerificationResult:
        """
        Check messages for safety concerns.

        Args:
            messages: Conversation messages
            metadata: Optional metadata

        Returns:
            VerificationResult with safety check details
        """
        all_violations = []
        messages_checked = 0

        for msg in messages:
            role = msg.get("role", "")
            if role not in ["assistant", "tool"]:
                continue

            messages_checked += 1
            content = msg.get("content", "")
            if isinstance(content, dict):
                content = content.get("content", "")

            pattern_matches = self._check_patterns(content)
            if pattern_matches:
                all_violations.append({
                    "type": "pattern_match",
                    "role": role,
                    "matches": pattern_matches,
                })

            blocklist_matches = self._check_blocklist(content)
            if blocklist_matches:
                all_violations.append({
                    "type": "blocklist_match",
                    "role": role,
                    "matches": blocklist_matches,
                })

            if self.check_tool_calls and "tool_calls" in msg:
                for tool_call in msg.get("tool_calls", []):
                    args = tool_call.get("function", {}).get("arguments", "")
                    arg_blocklist = self._check_blocklist(args)
                    if arg_blocklist:
                        all_violations.append({
                            "type": "tool_call_blocklist",
                            "tool_name": tool_call.get("function", {}).get("name"),
                            "matches": arg_blocklist,
                        })

        if messages_checked == 0:
            return VerificationResult(
                status=VerificationStatus.SKIPPED,
                plugin_name=self.name,
                message="No assistant or tool messages to check",
                details={"messages_checked": 0},
            )

        if all_violations:
            return VerificationResult(
                status=VerificationStatus.FAILED,
                plugin_name=self.name,
                message=f"Found {len(all_violations)} safety violation(s)",
                details={
                    "messages_checked": messages_checked,
                    "violations": all_violations,
                },
                score_modifier=0.0,
            )

        return VerificationResult(
            status=VerificationStatus.PASSED,
            plugin_name=self.name,
            message=f"Passed safety checks ({messages_checked} messages checked)",
            details={"messages_checked": messages_checked},
            score_modifier=1.0,
        )
