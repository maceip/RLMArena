"""
Leash Verifier Plugin for kernel-level network policy enforcement.

This plugin integrates with strongdm/leash (or a mock) to enforce Cedar policies
on agent network access. Unlike container-level isolation, Leash operates in the
kernel network stack with <1ms overhead.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional
import json
import re

from qqr.verification.plugin import VerificationPlugin, VerificationResult, VerificationStatus


class CedarEffect(Enum):
    """Cedar policy effect."""
    PERMIT = "permit"
    FORBID = "forbid"


class CedarOperator(Enum):
    """Cedar policy condition operators."""
    EQUALS = "=="
    NOT_EQUALS = "!="
    IN = "in"
    CONTAINS = "contains"
    MATCHES = "matches"
    LESS_THAN = "<"
    GREATER_THAN = ">"


@dataclass
class CedarCondition:
    """A condition in a Cedar policy."""
    attribute: str
    operator: CedarOperator
    value: Any

    def evaluate(self, context: dict[str, Any]) -> bool:
        """Evaluate this condition against a context."""
        actual = context.get(self.attribute)

        if self.operator == CedarOperator.EQUALS:
            return actual == self.value
        elif self.operator == CedarOperator.NOT_EQUALS:
            return actual != self.value
        elif self.operator == CedarOperator.IN:
            return actual in self.value
        elif self.operator == CedarOperator.CONTAINS:
            if isinstance(actual, str) and isinstance(self.value, str):
                return self.value in actual
            elif isinstance(actual, (list, set)):
                return self.value in actual
            return False
        elif self.operator == CedarOperator.MATCHES:
            if isinstance(actual, str):
                return bool(re.match(self.value, actual))
            return False
        elif self.operator == CedarOperator.LESS_THAN:
            return actual < self.value if actual is not None else False
        elif self.operator == CedarOperator.GREATER_THAN:
            return actual > self.value if actual is not None else False
        return False

    def to_cedar_string(self) -> str:
        """Convert to Cedar policy language string."""
        op_map = {
            CedarOperator.EQUALS: "==",
            CedarOperator.NOT_EQUALS: "!=",
            CedarOperator.IN: "in",
            CedarOperator.CONTAINS: ".contains",
            CedarOperator.MATCHES: ".matches",
            CedarOperator.LESS_THAN: "<",
            CedarOperator.GREATER_THAN: ">",
        }
        op = op_map[self.operator]

        if self.operator in (CedarOperator.CONTAINS, CedarOperator.MATCHES):
            return f'{self.attribute}{op}("{self.value}")'
        else:
            value_str = json.dumps(self.value) if isinstance(self.value, str) else str(self.value)
            return f"{self.attribute} {op} {value_str}"


@dataclass
class CedarPolicy:
    """
    A Cedar policy for network/resource access control.

    Cedar is a policy language developed by AWS that enables fine-grained
    authorization with formal verification guarantees.
    """
    id: str
    effect: CedarEffect
    principal: str  # e.g., "Agent::*" or "Agent::\"agent-123\""
    action: str  # e.g., "Action::\"connect\"" or "Action::\"read\""
    resource: str  # e.g., "Host::\"internal-api.com\"" or "Host::*"
    conditions: list[CedarCondition] = field(default_factory=list)
    description: Optional[str] = None

    def matches(
        self,
        principal: str,
        action: str,
        resource: str,
        context: Optional[dict[str, Any]] = None,
    ) -> bool:
        """Check if this policy matches the given request."""
        context = context or {}

        # Check principal match
        if not self._matches_pattern(self.principal, principal):
            return False

        # Check action match
        if not self._matches_pattern(self.action, action):
            return False

        # Check resource match
        if not self._matches_pattern(self.resource, resource):
            return False

        # Check conditions
        for condition in self.conditions:
            if not condition.evaluate(context):
                return False

        return True

    def _matches_pattern(self, pattern: str, value: str) -> bool:
        """Check if a pattern matches a value (supports wildcards)."""
        if pattern.endswith("::*"):
            prefix = pattern[:-2]
            return value.startswith(prefix)
        return pattern == value

    def to_cedar_string(self) -> str:
        """Convert to Cedar policy language."""
        effect = self.effect.value
        lines = [f"{effect} ("]
        lines.append(f"  principal {self.principal},")
        lines.append(f"  action {self.action},")
        lines.append(f"  resource {self.resource}")
        lines.append(")")

        if self.conditions:
            lines.append("when {")
            cond_strs = [f"  {c.to_cedar_string()}" for c in self.conditions]
            lines.append(" &&\n".join(cond_strs))
            lines.append("};")
        else:
            lines.append(";")

        return "\n".join(lines)


@dataclass
class NetworkRequest:
    """A network request to be verified against policies."""
    principal: str
    action: str
    resource: str
    context: dict[str, Any] = field(default_factory=dict)
    source_message_index: Optional[int] = None
    source_tool_call: Optional[str] = None


class LeashPolicyEngine:
    """
    Mock Leash policy engine for Cedar policy evaluation.

    In production, this would interface with the leash binary.
    This mock provides the same API for testing and development.
    """

    def __init__(self, policies: Optional[list[CedarPolicy]] = None):
        self.policies = policies or []
        self._default_deny = True

    def add_policy(self, policy: CedarPolicy) -> None:
        """Add a policy to the engine."""
        self.policies.append(policy)

    def remove_policy(self, policy_id: str) -> bool:
        """Remove a policy by ID."""
        original_len = len(self.policies)
        self.policies = [p for p in self.policies if p.id != policy_id]
        return len(self.policies) < original_len

    def evaluate(self, request: NetworkRequest) -> tuple[bool, list[str]]:
        """
        Evaluate a request against all policies.

        Returns (allowed, matched_policy_ids).
        Uses default-deny: if no PERMIT matches, request is denied.
        """
        matched_permits = []
        matched_forbids = []

        for policy in self.policies:
            if policy.matches(
                request.principal,
                request.action,
                request.resource,
                request.context,
            ):
                if policy.effect == CedarEffect.PERMIT:
                    matched_permits.append(policy.id)
                else:
                    matched_forbids.append(policy.id)

        # FORBID takes precedence over PERMIT
        if matched_forbids:
            return False, matched_forbids

        # If any PERMIT matched, allow
        if matched_permits:
            return True, matched_permits

        # Default deny
        return not self._default_deny, []

    def get_applicable_policies(self, request: NetworkRequest) -> list[CedarPolicy]:
        """Get all policies that match a request."""
        return [
            policy for policy in self.policies
            if policy.matches(
                request.principal,
                request.action,
                request.resource,
                request.context,
            )
        ]


class LeashVerifierPlugin(VerificationPlugin):
    """
    Verification plugin that enforces Cedar policies on agent network access.

    This plugin:
    1. Extracts network-related actions from trajectory
    2. Evaluates them against Cedar policies
    3. Fails trajectory if any policy is violated
    """

    def __init__(
        self,
        policies: Optional[list[CedarPolicy]] = None,
        agent_id: str = "default-agent",
        extract_urls: bool = True,
        extract_tool_targets: bool = True,
        blocked_hosts: Optional[list[str]] = None,
        allowed_hosts: Optional[list[str]] = None,
    ):
        self._policies = policies or []
        self._agent_id = agent_id
        self._extract_urls = extract_urls
        self._extract_tool_targets = extract_tool_targets
        self._blocked_hosts = blocked_hosts or []
        self._allowed_hosts = allowed_hosts

        # Initialize policy engine
        self._engine = LeashPolicyEngine(self._policies)

        # Add default blocked host policies
        for host in self._blocked_hosts:
            self._engine.add_policy(CedarPolicy(
                id=f"block-{host}",
                effect=CedarEffect.FORBID,
                principal="Agent::*",
                action="Action::*",
                resource=f'Host::"{host}"',
                description=f"Block access to {host}",
            ))

        # Add allowed host policies if whitelist mode
        if self._allowed_hosts is not None:
            for host in self._allowed_hosts:
                self._engine.add_policy(CedarPolicy(
                    id=f"allow-{host}",
                    effect=CedarEffect.PERMIT,
                    principal="Agent::*",
                    action="Action::*",
                    resource=f'Host::"{host}"',
                    description=f"Allow access to {host}",
                ))

    @property
    def name(self) -> str:
        return "leash_verifier"

    @property
    def description(self) -> str:
        return "Enforces Cedar network policies on agent actions using kernel-level Leash engine"

    @property
    def is_hard_check(self) -> bool:
        return True

    @property
    def floor_reward(self) -> float:
        return -1.0

    def add_policy(self, policy: CedarPolicy) -> None:
        """Add a Cedar policy at runtime."""
        self._engine.add_policy(policy)

    def _extract_network_requests(
        self,
        messages: list[dict[str, Any]],
        metadata: Optional[dict[str, Any]] = None,
    ) -> list[NetworkRequest]:
        """Extract network requests from trajectory messages."""
        requests = []
        principal = f'Agent::"{self._agent_id}"'

        for i, msg in enumerate(messages):
            content = msg.get("content", "")
            role = msg.get("role", "")

            # Extract URLs from content
            if self._extract_urls and isinstance(content, str):
                urls = self._extract_urls_from_text(content)
                for url in urls:
                    host = self._extract_host(url)
                    if host:
                        requests.append(NetworkRequest(
                            principal=principal,
                            action='Action::"connect"',
                            resource=f'Host::"{host}"',
                            context={"url": url, "role": role},
                            source_message_index=i,
                        ))

            # Extract from tool calls
            if self._extract_tool_targets and role == "assistant":
                tool_calls = msg.get("tool_calls", [])
                for tool_call in tool_calls:
                    tool_name = tool_call.get("function", {}).get("name", "")
                    args = tool_call.get("function", {}).get("arguments", "{}")

                    try:
                        args_dict = json.loads(args) if isinstance(args, str) else args
                    except json.JSONDecodeError:
                        args_dict = {}

                    # Check for URL/host in arguments
                    for key, value in args_dict.items():
                        if isinstance(value, str):
                            if key in ("url", "host", "endpoint", "target", "address"):
                                host = self._extract_host(value)
                                if host:
                                    requests.append(NetworkRequest(
                                        principal=principal,
                                        action=f'Action::"{tool_name}"',
                                        resource=f'Host::"{host}"',
                                        context={"tool": tool_name, "argument": key},
                                        source_message_index=i,
                                        source_tool_call=tool_name,
                                    ))

        return requests

    def _extract_urls_from_text(self, text: str) -> list[str]:
        """Extract URLs from text content."""
        url_pattern = r'https?://[^\s<>"\')\]}`]+'
        return re.findall(url_pattern, text)

    def _extract_host(self, url_or_host: str) -> Optional[str]:
        """Extract hostname from URL or return as-is if already a host."""
        # Already a hostname
        if not url_or_host.startswith(("http://", "https://")):
            # Basic hostname validation
            if re.match(r'^[\w.-]+$', url_or_host):
                return url_or_host
            return None

        # Extract from URL
        match = re.match(r'https?://([^/:]+)', url_or_host)
        if match:
            return match.group(1)
        return None

    async def verify(
        self,
        messages: list[dict[str, Any]],
        metadata: Optional[dict[str, Any]] = None,
    ) -> VerificationResult:
        """Verify trajectory against Cedar network policies."""
        requests = self._extract_network_requests(messages, metadata)

        if not requests:
            return VerificationResult(
                status=VerificationStatus.PASSED,
                plugin_name=self.name,
                message="No network requests detected in trajectory",
            )

        violations = []
        for request in requests:
            allowed, matched_policies = self._engine.evaluate(request)

            if not allowed:
                violation = {
                    "resource": request.resource,
                    "action": request.action,
                    "message_index": request.source_message_index,
                    "tool_call": request.source_tool_call,
                    "blocking_policies": matched_policies,
                }
                violations.append(violation)

        if violations:
            return VerificationResult(
                status=VerificationStatus.FAILED,
                plugin_name=self.name,
                message=f"Policy violations detected: {len(violations)} forbidden network request(s)",
                details={
                    "violations": violations,
                    "total_requests": len(requests),
                },
                score_modifier=-1.0,
            )

        return VerificationResult(
            status=VerificationStatus.PASSED,
            plugin_name=self.name,
            message=f"All {len(requests)} network requests permitted by policy",
            details={
                "total_requests": len(requests),
                "verified_hosts": list(set(r.resource for r in requests)),
            },
        )


# Preset policy sets for common use cases
STRICT_INTERNAL_POLICY = [
    CedarPolicy(
        id="forbid-internal",
        effect=CedarEffect.FORBID,
        principal="Agent::*",
        action="Action::*",
        resource="Host::*",
        conditions=[
            CedarCondition(
                attribute="host",
                operator=CedarOperator.MATCHES,
                value=r".*\.internal\..*",
            )
        ],
        description="Forbid access to internal domains",
    ),
    CedarPolicy(
        id="forbid-localhost",
        effect=CedarEffect.FORBID,
        principal="Agent::*",
        action="Action::*",
        resource='Host::"localhost"',
        description="Forbid access to localhost",
    ),
    CedarPolicy(
        id="forbid-private-ip",
        effect=CedarEffect.FORBID,
        principal="Agent::*",
        action="Action::*",
        resource="Host::*",
        conditions=[
            CedarCondition(
                attribute="host",
                operator=CedarOperator.MATCHES,
                value=r"^(10\.|172\.(1[6-9]|2[0-9]|3[01])\.|192\.168\.).*",
            )
        ],
        description="Forbid access to private IP ranges",
    ),
]

ALLOW_PUBLIC_ONLY_POLICY = [
    CedarPolicy(
        id="permit-https",
        effect=CedarEffect.PERMIT,
        principal="Agent::*",
        action='Action::"connect"',
        resource="Host::*",
        conditions=[
            CedarCondition(
                attribute="url",
                operator=CedarOperator.MATCHES,
                value=r"^https://.*",
            )
        ],
        description="Permit HTTPS connections",
    ),
] + STRICT_INTERNAL_POLICY
