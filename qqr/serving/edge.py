"""
Edge Verifier for zero-latency local verification.

Runs static analysis and policy checks on-device or at VPC edge
without requiring cloud round-trips.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional
import asyncio
import re
import time


class EdgeCheckType(Enum):
    """Types of edge verification checks."""
    STATIC_ANALYSIS = "static_analysis"
    POLICY_CHECK = "policy_check"
    FORMAT_VALIDATION = "format_validation"
    BLOCKLIST_CHECK = "blocklist_check"
    RATE_CHECK = "rate_check"


@dataclass
class EdgeConfig:
    """Configuration for edge verifier."""
    enable_static_analysis: bool = True
    enable_policy_checks: bool = True
    enable_format_validation: bool = True
    enable_blocklist: bool = True
    max_content_length: int = 100000
    blocked_patterns: list[str] = field(default_factory=list)
    allowed_domains: Optional[list[str]] = None
    max_tool_calls_per_turn: int = 10
    timeout_ms: float = 100.0


@dataclass
class EdgeCheckResult:
    """Result of an edge verification check."""
    check_type: EdgeCheckType
    passed: bool
    message: str
    latency_ms: float
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class EdgeVerificationResult:
    """Complete edge verification result."""
    passed: bool
    checks: list[EdgeCheckResult]
    total_latency_ms: float
    blocked_reason: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "checks": [
                {
                    "type": c.check_type.value,
                    "passed": c.passed,
                    "message": c.message,
                    "latency_ms": c.latency_ms,
                }
                for c in self.checks
            ],
            "total_latency_ms": self.total_latency_ms,
            "blocked_reason": self.blocked_reason,
            "timestamp": self.timestamp.isoformat(),
        }


class StaticAnalyzer:
    """Lightweight static analyzer for edge deployment."""

    def __init__(self):
        self._python_patterns = [
            (r"\beval\s*\(", "Use of eval()"),
            (r"\bexec\s*\(", "Use of exec()"),
            (r"\b__import__\s*\(", "Dynamic import"),
            (r"os\.system\s*\(", "Shell command execution"),
            (r"subprocess\.(run|call|Popen)", "Subprocess execution"),
        ]

        self._js_patterns = [
            (r"\beval\s*\(", "Use of eval()"),
            (r"innerHTML\s*=", "Direct innerHTML assignment"),
            (r"document\.write\s*\(", "Use of document.write"),
        ]

    def analyze(self, code: str, language: str = "python") -> list[dict[str, Any]]:
        """Analyze code for issues."""
        issues = []
        patterns = self._python_patterns if language == "python" else self._js_patterns

        for pattern, description in patterns:
            matches = list(re.finditer(pattern, code))
            for match in matches:
                line_num = code[:match.start()].count('\n') + 1
                issues.append({
                    "line": line_num,
                    "pattern": pattern,
                    "description": description,
                    "severity": "warning",
                })

        return issues


class PolicyChecker:
    """Lightweight policy checker for Cedar policies."""

    def __init__(self, policies: Optional[list[dict[str, Any]]] = None):
        self._policies = policies or []
        self._default_blocked_hosts = [
            r".*\.internal\.",
            r"localhost",
            r"127\.0\.0\.1",
            r"10\.\d+\.\d+\.\d+",
            r"192\.168\.\d+\.\d+",
        ]

    def check_url(self, url: str) -> tuple[bool, Optional[str]]:
        """Check if URL is allowed by policy."""
        for pattern in self._default_blocked_hosts:
            if re.search(pattern, url, re.IGNORECASE):
                return False, f"URL matches blocked pattern: {pattern}"
        return True, None

    def check_content(self, content: str) -> tuple[bool, Optional[str]]:
        """Check content against policies."""
        # Check for sensitive data patterns
        sensitive_patterns = [
            (r"(?i)password\s*[:=]\s*['\"][^'\"]+['\"]", "Hardcoded password"),
            (r"AKIA[0-9A-Z]{16}", "AWS access key"),
            (r"(?i)api[_-]?key\s*[:=]\s*['\"][^'\"]+['\"]", "Hardcoded API key"),
        ]

        for pattern, description in sensitive_patterns:
            if re.search(pattern, content):
                return False, f"Sensitive data detected: {description}"

        return True, None


class EdgeVerifier:
    """
    Edge verifier for zero-latency local checks.

    Runs lightweight verification that can execute on-device
    or at the VPC edge without cloud round-trips.
    """

    def __init__(self, config: Optional[EdgeConfig] = None):
        self.config = config or EdgeConfig()
        self._static_analyzer = StaticAnalyzer()
        self._policy_checker = PolicyChecker()
        self._blocklist_patterns = [re.compile(p) for p in self.config.blocked_patterns]

    async def verify(
        self,
        messages: list[dict[str, Any]],
        metadata: Optional[dict[str, Any]] = None,
    ) -> EdgeVerificationResult:
        """
        Run edge verification on messages.

        Executes all enabled checks and returns aggregated result.
        """
        start_time = time.time()
        checks = []
        blocked_reason = None

        # Run checks in parallel
        tasks = []

        if self.config.enable_format_validation:
            tasks.append(self._check_format(messages))

        if self.config.enable_blocklist:
            tasks.append(self._check_blocklist(messages))

        if self.config.enable_static_analysis:
            tasks.append(self._check_static_analysis(messages))

        if self.config.enable_policy_checks:
            tasks.append(self._check_policies(messages))

        try:
            results = await asyncio.wait_for(
                asyncio.gather(*tasks),
                timeout=self.config.timeout_ms / 1000.0,
            )
            checks.extend(results)
        except asyncio.TimeoutError:
            checks.append(EdgeCheckResult(
                check_type=EdgeCheckType.RATE_CHECK,
                passed=False,
                message="Edge verification timed out",
                latency_ms=self.config.timeout_ms,
            ))

        # Determine overall result
        passed = all(c.passed for c in checks)
        if not passed:
            for check in checks:
                if not check.passed:
                    blocked_reason = check.message
                    break

        total_latency = (time.time() - start_time) * 1000

        return EdgeVerificationResult(
            passed=passed,
            checks=checks,
            total_latency_ms=total_latency,
            blocked_reason=blocked_reason,
        )

    async def _check_format(
        self,
        messages: list[dict[str, Any]],
    ) -> EdgeCheckResult:
        """Check message format validity."""
        start_time = time.time()

        for i, msg in enumerate(messages):
            if not isinstance(msg, dict):
                return EdgeCheckResult(
                    check_type=EdgeCheckType.FORMAT_VALIDATION,
                    passed=False,
                    message=f"Message {i} is not a dict",
                    latency_ms=(time.time() - start_time) * 1000,
                )

            if "role" not in msg:
                return EdgeCheckResult(
                    check_type=EdgeCheckType.FORMAT_VALIDATION,
                    passed=False,
                    message=f"Message {i} missing 'role'",
                    latency_ms=(time.time() - start_time) * 1000,
                )

            content = msg.get("content", "")
            if isinstance(content, str) and len(content) > self.config.max_content_length:
                return EdgeCheckResult(
                    check_type=EdgeCheckType.FORMAT_VALIDATION,
                    passed=False,
                    message=f"Message {i} content exceeds max length",
                    latency_ms=(time.time() - start_time) * 1000,
                )

        return EdgeCheckResult(
            check_type=EdgeCheckType.FORMAT_VALIDATION,
            passed=True,
            message="Format validation passed",
            latency_ms=(time.time() - start_time) * 1000,
        )

    async def _check_blocklist(
        self,
        messages: list[dict[str, Any]],
    ) -> EdgeCheckResult:
        """Check messages against blocklist patterns."""
        start_time = time.time()

        for msg in messages:
            content = msg.get("content", "")
            if not isinstance(content, str):
                continue

            for pattern in self._blocklist_patterns:
                if pattern.search(content):
                    return EdgeCheckResult(
                        check_type=EdgeCheckType.BLOCKLIST_CHECK,
                        passed=False,
                        message=f"Content matches blocked pattern",
                        latency_ms=(time.time() - start_time) * 1000,
                        details={"pattern": pattern.pattern},
                    )

        return EdgeCheckResult(
            check_type=EdgeCheckType.BLOCKLIST_CHECK,
            passed=True,
            message="Blocklist check passed",
            latency_ms=(time.time() - start_time) * 1000,
        )

    async def _check_static_analysis(
        self,
        messages: list[dict[str, Any]],
    ) -> EdgeCheckResult:
        """Run static analysis on code in messages."""
        start_time = time.time()
        all_issues = []

        for msg in messages:
            content = msg.get("content", "")
            if not isinstance(content, str):
                continue

            # Extract code blocks
            code_pattern = r"```(\w+)?\s*\n(.*?)```"
            for match in re.finditer(code_pattern, content, re.DOTALL):
                language = match.group(1) or "python"
                code = match.group(2)

                issues = self._static_analyzer.analyze(code, language)
                all_issues.extend(issues)

        if all_issues:
            return EdgeCheckResult(
                check_type=EdgeCheckType.STATIC_ANALYSIS,
                passed=True,  # Warnings don't block
                message=f"Static analysis found {len(all_issues)} warnings",
                latency_ms=(time.time() - start_time) * 1000,
                details={"issues": all_issues[:10]},  # Limit details
            )

        return EdgeCheckResult(
            check_type=EdgeCheckType.STATIC_ANALYSIS,
            passed=True,
            message="Static analysis passed",
            latency_ms=(time.time() - start_time) * 1000,
        )

    async def _check_policies(
        self,
        messages: list[dict[str, Any]],
    ) -> EdgeCheckResult:
        """Check policy compliance."""
        start_time = time.time()

        for msg in messages:
            content = msg.get("content", "")
            if not isinstance(content, str):
                continue

            # Check content for sensitive data
            passed, reason = self._policy_checker.check_content(content)
            if not passed:
                return EdgeCheckResult(
                    check_type=EdgeCheckType.POLICY_CHECK,
                    passed=False,
                    message=reason or "Policy check failed",
                    latency_ms=(time.time() - start_time) * 1000,
                )

            # Check URLs in content
            url_pattern = r'https?://[^\s<>"\')\]}`]+'
            urls = re.findall(url_pattern, content)
            for url in urls:
                passed, reason = self._policy_checker.check_url(url)
                if not passed:
                    return EdgeCheckResult(
                        check_type=EdgeCheckType.POLICY_CHECK,
                        passed=False,
                        message=reason or "URL policy check failed",
                        latency_ms=(time.time() - start_time) * 1000,
                        details={"url": url},
                    )

        return EdgeCheckResult(
            check_type=EdgeCheckType.POLICY_CHECK,
            passed=True,
            message="Policy checks passed",
            latency_ms=(time.time() - start_time) * 1000,
        )

    def add_blocklist_pattern(self, pattern: str) -> None:
        """Add a pattern to the blocklist."""
        self._blocklist_patterns.append(re.compile(pattern))

    def get_stats(self) -> dict[str, Any]:
        """Get verifier statistics."""
        return {
            "blocklist_patterns": len(self._blocklist_patterns),
            "config": {
                "static_analysis": self.config.enable_static_analysis,
                "policy_checks": self.config.enable_policy_checks,
                "format_validation": self.config.enable_format_validation,
                "blocklist": self.config.enable_blocklist,
            },
        }
