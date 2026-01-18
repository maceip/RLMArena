"""
Mock Infrastructure Components.

Simulates external infrastructure services for testing and development
without requiring actual deployment of these systems.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Optional
import asyncio
import hashlib
import json
import random
import time
import uuid


@dataclass
class InfrastructureConfig:
    """Configuration for mock infrastructure."""
    simulate_latency: bool = True
    latency_base_ms: float = 10.0
    latency_variance_ms: float = 5.0
    failure_rate: float = 0.0  # 0.0 to 1.0
    enable_logging: bool = False


class MockSGLang:
    """
    Mock sglang router with RadixAttention simulation.

    Simulates:
    - Prefix caching with RadixAttention
    - Load balancing across workers
    - Batch inference
    """

    def __init__(
        self,
        config: Optional[InfrastructureConfig] = None,
        num_workers: int = 4,
    ):
        self.config = config or InfrastructureConfig()
        self.num_workers = num_workers
        self._prefix_cache: dict[str, dict[str, Any]] = {}
        self._worker_loads: dict[int, int] = {i: 0 for i in range(num_workers)}
        self._stats = {
            "total_requests": 0,
            "cache_hits": 0,
            "cache_misses": 0,
        }

    async def generate(
        self,
        messages: list[dict[str, Any]],
        max_tokens: int = 1024,
        temperature: float = 0.7,
        n: int = 1,
    ) -> list[dict[str, Any]]:
        """Generate responses with RadixAttention prefix caching."""
        self._stats["total_requests"] += 1

        # Simulate latency
        if self.config.simulate_latency:
            latency = self.config.latency_base_ms + random.uniform(
                -self.config.latency_variance_ms,
                self.config.latency_variance_ms,
            )
            await asyncio.sleep(latency / 1000.0)

        # Simulate failure
        if random.random() < self.config.failure_rate:
            raise RuntimeError("Mock sglang generation failed")

        # Check prefix cache
        prefix_key = self._compute_prefix_key(messages)
        cache_hit = prefix_key in self._prefix_cache

        if cache_hit:
            self._stats["cache_hits"] += 1
        else:
            self._stats["cache_misses"] += 1
            self._prefix_cache[prefix_key] = {
                "messages": messages,
                "cached_at": datetime.utcnow(),
            }

        # Generate mock responses
        responses = []
        for i in range(n):
            response_content = self._generate_mock_response(messages, temperature)
            responses.append({
                "role": "assistant",
                "content": response_content,
                "finish_reason": "stop",
                "usage": {
                    "prompt_tokens": sum(len(m.get("content", "")) // 4 for m in messages),
                    "completion_tokens": len(response_content) // 4,
                },
                "cache_hit": cache_hit,
            })

        return responses

    def _compute_prefix_key(self, messages: list[dict[str, Any]]) -> str:
        """Compute cache key for message prefix."""
        content = json.dumps(messages, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def _generate_mock_response(
        self,
        messages: list[dict[str, Any]],
        temperature: float,
    ) -> str:
        """Generate a mock response."""
        last_user_msg = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                last_user_msg = msg.get("content", "")[:100]
                break

        response_templates = [
            f"Here's my response to your query about {last_user_msg[:30]}...",
            f"Based on your request, I've analyzed the situation and here's what I found...",
            f"Let me help you with that. The solution involves the following steps...",
        ]

        base_response = random.choice(response_templates)

        # Add variability based on temperature
        if temperature > 0.5:
            base_response += f" [Generated with temp={temperature:.2f}]"

        return base_response

    def get_stats(self) -> dict[str, Any]:
        """Get router statistics."""
        cache_total = self._stats["cache_hits"] + self._stats["cache_misses"]
        return {
            **self._stats,
            "cache_hit_rate": (
                self._stats["cache_hits"] / cache_total if cache_total > 0 else 0.0
            ),
            "cache_size": len(self._prefix_cache),
            "num_workers": self.num_workers,
        }


class MockFirecracker:
    """
    Mock Firecracker MicroVM for code execution.

    Simulates:
    - VM creation and destruction
    - Isolated code execution
    - Resource limits
    """

    def __init__(
        self,
        config: Optional[InfrastructureConfig] = None,
        pool_size: int = 4,
    ):
        self.config = config or InfrastructureConfig()
        self.pool_size = pool_size
        self._available_vms = list(range(pool_size))
        self._active_executions: dict[str, int] = {}
        self._stats = {
            "total_executions": 0,
            "successful_executions": 0,
            "failed_executions": 0,
        }

    async def execute(
        self,
        code: str,
        language: str = "python",
        timeout_seconds: float = 30.0,
        memory_mb: int = 512,
    ) -> dict[str, Any]:
        """Execute code in a mock MicroVM."""
        execution_id = str(uuid.uuid4())[:8]
        self._stats["total_executions"] += 1

        # Simulate latency
        if self.config.simulate_latency:
            latency = self.config.latency_base_ms * 5 + random.uniform(0, 50)
            await asyncio.sleep(latency / 1000.0)

        # Simulate failure
        if random.random() < self.config.failure_rate:
            self._stats["failed_executions"] += 1
            return {
                "execution_id": execution_id,
                "status": "error",
                "exit_code": -1,
                "stdout": "",
                "stderr": "Mock execution failed",
                "execution_time_ms": 0.0,
            }

        # Check for obvious errors in code
        has_syntax_error = self._check_syntax_errors(code, language)

        if has_syntax_error:
            self._stats["failed_executions"] += 1
            return {
                "execution_id": execution_id,
                "status": "failure",
                "exit_code": 1,
                "stdout": "",
                "stderr": "SyntaxError: invalid syntax",
                "execution_time_ms": 10.0,
            }

        self._stats["successful_executions"] += 1
        return {
            "execution_id": execution_id,
            "status": "success",
            "exit_code": 0,
            "stdout": f"Mock execution of {language} code completed successfully\n",
            "stderr": "",
            "execution_time_ms": random.uniform(50, 200),
            "memory_used_mb": random.uniform(10, memory_mb * 0.5),
        }

    def _check_syntax_errors(self, code: str, language: str) -> bool:
        """Check for obvious syntax errors."""
        if language == "python":
            try:
                compile(code, "<string>", "exec")
                return False
            except SyntaxError:
                return True
        return False

    def get_stats(self) -> dict[str, Any]:
        """Get execution statistics."""
        return {
            **self._stats,
            "pool_size": self.pool_size,
            "available_vms": len(self._available_vms),
            "success_rate": (
                self._stats["successful_executions"] / self._stats["total_executions"]
                if self._stats["total_executions"] > 0 else 0.0
            ),
        }


class MockLeash:
    """
    Mock Leash kernel-level network enforcement.

    Simulates:
    - Cedar policy evaluation
    - Network access control
    - Audit logging
    """

    def __init__(
        self,
        config: Optional[InfrastructureConfig] = None,
        default_deny: bool = True,
    ):
        self.config = config or InfrastructureConfig()
        self.default_deny = default_deny
        self._policies: list[dict[str, Any]] = []
        self._blocked_hosts = [
            "internal-api.com",
            "localhost",
            "127.0.0.1",
        ]
        self._allowed_hosts = [
            "api.github.com",
            "api.openai.com",
            "*.amazonaws.com",
        ]
        self._audit_log: list[dict[str, Any]] = []

    def add_policy(self, policy: dict[str, Any]) -> None:
        """Add a Cedar policy."""
        self._policies.append(policy)

    def block_host(self, host: str) -> None:
        """Block a specific host."""
        self._blocked_hosts.append(host)

    def allow_host(self, host: str) -> None:
        """Allow a specific host."""
        self._allowed_hosts.append(host)

    async def evaluate(
        self,
        principal: str,
        action: str,
        resource: str,
        context: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Evaluate a network request against policies."""
        # Simulate latency (very low for kernel-level)
        if self.config.simulate_latency:
            await asyncio.sleep(0.001)  # 1ms max

        # Extract host from resource
        host = resource.replace('Host::"', "").rstrip('"')

        # Check blocked hosts
        for blocked in self._blocked_hosts:
            if blocked in host:
                result = {
                    "allowed": False,
                    "reason": f"Host {host} is blocked by policy",
                    "policy_id": "block-list",
                }
                self._audit_log.append({
                    "timestamp": datetime.utcnow().isoformat(),
                    **result,
                    "principal": principal,
                    "action": action,
                    "resource": resource,
                })
                return result

        # Check allowed hosts
        for allowed in self._allowed_hosts:
            if allowed.startswith("*"):
                if host.endswith(allowed[1:]):
                    return {"allowed": True, "reason": "Matched wildcard allow"}
            elif allowed == host:
                return {"allowed": True, "reason": "Matched explicit allow"}

        # Default deny
        if self.default_deny:
            return {
                "allowed": False,
                "reason": "No matching allow policy (default deny)",
            }

        return {"allowed": True, "reason": "Default allow"}

    def get_audit_log(self) -> list[dict[str, Any]]:
        """Get audit log of blocked requests."""
        return self._audit_log.copy()

    def get_stats(self) -> dict[str, Any]:
        """Get enforcement statistics."""
        return {
            "policies": len(self._policies),
            "blocked_hosts": len(self._blocked_hosts),
            "allowed_hosts": len(self._allowed_hosts),
            "audit_log_size": len(self._audit_log),
        }


class MockOPA:
    """
    Mock Open Policy Agent for Rego policy evaluation.

    Simulates:
    - Rego policy evaluation
    - Infrastructure compliance checking
    """

    def __init__(
        self,
        config: Optional[InfrastructureConfig] = None,
    ):
        self.config = config or InfrastructureConfig()
        self._policies: dict[str, str] = {}
        self._builtin_checks = {
            "no_public_s3": self._check_no_public_s3,
            "encryption_required": self._check_encryption,
            "tagging_required": self._check_tagging,
        }

    def add_policy(self, name: str, rego: str) -> None:
        """Add a Rego policy."""
        self._policies[name] = rego

    async def evaluate(
        self,
        input_data: dict[str, Any],
        policy_name: Optional[str] = None,
    ) -> dict[str, Any]:
        """Evaluate input against policies."""
        # Simulate latency
        if self.config.simulate_latency:
            await asyncio.sleep(self.config.latency_base_ms / 1000.0)

        violations = []

        # Run built-in checks
        for check_name, check_fn in self._builtin_checks.items():
            result = check_fn(input_data)
            violations.extend(result)

        return {
            "allow": len(violations) == 0,
            "violations": violations,
            "policies_evaluated": list(self._builtin_checks.keys()),
        }

    def _check_no_public_s3(self, data: dict[str, Any]) -> list[dict[str, Any]]:
        """Check for public S3 buckets."""
        violations = []
        resources = data.get("resource", {})
        buckets = resources.get("aws_s3_bucket", {})

        for name, config in buckets.items():
            if config.get("acl") in ("public-read", "public-read-write"):
                violations.append({
                    "rule": "no_public_s3",
                    "resource": f"aws_s3_bucket.{name}",
                    "message": "S3 bucket has public ACL",
                })

        return violations

    def _check_encryption(self, data: dict[str, Any]) -> list[dict[str, Any]]:
        """Check for unencrypted resources."""
        violations = []
        resources = data.get("resource", {})

        # Check EBS volumes
        volumes = resources.get("aws_ebs_volume", {})
        for name, config in volumes.items():
            if not config.get("encrypted"):
                violations.append({
                    "rule": "encryption_required",
                    "resource": f"aws_ebs_volume.{name}",
                    "message": "EBS volume is not encrypted",
                })

        # Check RDS instances
        rds = resources.get("aws_db_instance", {})
        for name, config in rds.items():
            if not config.get("storage_encrypted"):
                violations.append({
                    "rule": "encryption_required",
                    "resource": f"aws_db_instance.{name}",
                    "message": "RDS instance storage is not encrypted",
                })

        return violations

    def _check_tagging(self, data: dict[str, Any]) -> list[dict[str, Any]]:
        """Check for required tags."""
        violations = []
        resources = data.get("resource", {})
        required_tags = ["Environment", "Owner"]

        taggable_types = ["aws_instance", "aws_s3_bucket", "aws_vpc"]

        for resource_type in taggable_types:
            type_resources = resources.get(resource_type, {})
            for name, config in type_resources.items():
                tags = config.get("tags", {})
                missing = [t for t in required_tags if t not in tags]
                if missing:
                    violations.append({
                        "rule": "tagging_required",
                        "resource": f"{resource_type}.{name}",
                        "message": f"Missing required tags: {missing}",
                    })

        return violations

    def get_stats(self) -> dict[str, Any]:
        """Get OPA statistics."""
        return {
            "custom_policies": len(self._policies),
            "builtin_checks": len(self._builtin_checks),
        }


class MockTfsec:
    """
    Mock tfsec scanner for Terraform security analysis.

    Simulates:
    - Security rule evaluation
    - Severity classification
    - Remediation suggestions
    """

    def __init__(
        self,
        config: Optional[InfrastructureConfig] = None,
        severity_threshold: str = "HIGH",
    ):
        self.config = config or InfrastructureConfig()
        self.severity_threshold = severity_threshold
        self._excluded_rules: set[str] = set()

    def exclude_rule(self, rule_id: str) -> None:
        """Exclude a rule from scanning."""
        self._excluded_rules.add(rule_id)

    async def scan(
        self,
        terraform_config: dict[str, Any],
    ) -> dict[str, Any]:
        """Scan Terraform configuration for security issues."""
        # Simulate latency
        if self.config.simulate_latency:
            await asyncio.sleep(self.config.latency_base_ms * 2 / 1000.0)

        findings = []
        resources = terraform_config.get("resource", {})

        # Check S3 buckets
        for name, config in resources.get("aws_s3_bucket", {}).items():
            if not config.get("versioning"):
                findings.append({
                    "rule_id": "AWS017",
                    "description": "S3 bucket does not have versioning enabled",
                    "severity": "MEDIUM",
                    "resource": f"aws_s3_bucket.{name}",
                })

            if config.get("acl") in ("public-read", "public-read-write"):
                findings.append({
                    "rule_id": "AWS001",
                    "description": "S3 bucket has public ACL",
                    "severity": "CRITICAL",
                    "resource": f"aws_s3_bucket.{name}",
                })

        # Check security groups
        for name, config in resources.get("aws_security_group", {}).items():
            for rule in config.get("ingress", []):
                if "0.0.0.0/0" in str(rule.get("cidr_blocks", [])):
                    if rule.get("from_port") == 22:
                        findings.append({
                            "rule_id": "AWS006",
                            "description": "SSH exposed to the internet",
                            "severity": "CRITICAL",
                            "resource": f"aws_security_group.{name}",
                        })

        # Filter excluded rules
        findings = [f for f in findings if f["rule_id"] not in self._excluded_rules]

        # Calculate summary
        critical = sum(1 for f in findings if f["severity"] == "CRITICAL")
        high = sum(1 for f in findings if f["severity"] == "HIGH")
        medium = sum(1 for f in findings if f["severity"] == "MEDIUM")

        return {
            "passed": critical == 0 and high == 0,
            "findings": findings,
            "summary": {
                "critical": critical,
                "high": high,
                "medium": medium,
                "low": sum(1 for f in findings if f["severity"] == "LOW"),
            },
            "resources_scanned": sum(len(v) for v in resources.values()),
        }

    def get_stats(self) -> dict[str, Any]:
        """Get scanner statistics."""
        return {
            "excluded_rules": len(self._excluded_rules),
            "severity_threshold": self.severity_threshold,
        }
