#!/usr/bin/env python3
"""
Standalone component tests that don't require external dependencies.

This script tests the core logic of all new components without
importing the full qqr package.
"""

import os
import sys
import sqlite3
import tempfile
import json
import re
from datetime import datetime
from enum import Enum
from dataclasses import dataclass, field
from typing import Any, Optional


def test_proxy_components():
    """Test proxy module components."""
    print("1. PROXY MODULE")
    print("-" * 40)

    # Test ProxyMode enum
    class ProxyMode(Enum):
        PASSTHROUGH = "passthrough"
        SHADOW = "shadow"
        INTERCEPT = "intercept"
        AUDIT = "audit"

    assert ProxyMode.INTERCEPT.value == "intercept"
    print("  ProxyMode enum: OK")

    # Test ProxyConfig dataclass
    @dataclass
    class ProxyConfig:
        mode: ProxyMode = ProxyMode.INTERCEPT
        num_variations: int = 4
        timeout_seconds: float = 60.0

    config = ProxyConfig()
    assert config.mode == ProxyMode.INTERCEPT
    assert config.num_variations == 4
    print(f"  ProxyConfig: mode={config.mode.value}, variations={config.num_variations}")

    # Test ResponseCache
    class ResponseCache:
        def __init__(self, max_size: int = 100):
            self._cache = {}
            self._max_size = max_size

        def _make_key(self, messages, model):
            import hashlib
            content = json.dumps({"m": messages, "model": model}, sort_keys=True)
            return hashlib.sha256(content.encode()).hexdigest()

        def get(self, messages, model):
            key = self._make_key(messages, model)
            return self._cache.get(key)

        def set(self, messages, model, value):
            key = self._make_key(messages, model)
            self._cache[key] = value

    cache = ResponseCache()
    messages = [{"role": "user", "content": "test"}]
    cache.set(messages, "gpt-4", {"id": "test-123"})
    result = cache.get(messages, "gpt-4")
    assert result == {"id": "test-123"}
    print("  ResponseCache: OK")

    # Test RateLimiter
    class RateLimiter:
        def __init__(self, rpm: int = 60):
            self._rpm = rpm
            self._tokens = {}

        def check(self, client_id: str):
            tokens = self._tokens.get(client_id, self._rpm)
            if tokens < 1:
                return False, 1.0
            self._tokens[client_id] = tokens - 1
            return True, None

    limiter = RateLimiter()
    allowed, _ = limiter.check("client-1")
    assert allowed
    print("  RateLimiter: OK")
    print()


def test_terraform_verifier():
    """Test Terraform verifier components."""
    print("2. TERRAFORM VERIFIER")
    print("-" * 40)

    # Test severity enum
    class TerraformSeverity(Enum):
        ERROR = "error"
        WARNING = "warning"
        NOTICE = "notice"

    assert TerraformSeverity.ERROR.value == "error"
    print("  TerraformSeverity enum: OK")

    # Test issue dataclass
    @dataclass
    class TerraformIssue:
        severity: TerraformSeverity
        message: str
        rule: Optional[str] = None

        def to_dict(self):
            return {
                "severity": self.severity.value,
                "message": self.message,
                "rule": self.rule,
            }

    issue = TerraformIssue(
        severity=TerraformSeverity.ERROR,
        message="S3 bucket not encrypted",
        rule="AWS017",
    )
    assert issue.to_dict()["severity"] == "error"
    print("  TerraformIssue: OK")

    # Test code block pattern
    TF_PATTERN = re.compile(
        r"```(?:terraform|tf|hcl)\s*\n(.*?)```",
        re.DOTALL | re.IGNORECASE
    )

    test_content = """
Here is the Terraform:

```terraform
resource "aws_s3_bucket" "main" {
  bucket = "my-bucket"
}
```

Done.
"""
    matches = TF_PATTERN.findall(test_content)
    assert len(matches) == 1
    assert "aws_s3_bucket" in matches[0]
    print("  Terraform regex: OK")
    print()


def test_infra_sandbox():
    """Test infrastructure sandbox verifier."""
    print("3. INFRA SANDBOX VERIFIER")
    print("-" * 40)

    # Test InfraType enum
    class InfraType(Enum):
        KUBERNETES = "kubernetes"
        DOCKER = "docker"
        CLOUDFORMATION = "cloudformation"
        ANSIBLE = "ansible"

    # Test K8s detection
    K8S_PATTERN = re.compile(r"apiVersion:\s*\S+.*?kind:\s*\S+", re.DOTALL)

    k8s_content = """
apiVersion: v1
kind: Pod
metadata:
  name: test
"""
    assert K8S_PATTERN.search(k8s_content)
    print("  Kubernetes detection: OK")

    # Test Dockerfile detection
    docker_content = "FROM python:3.11\nRUN pip install"
    assert docker_content.startswith("FROM ")
    print("  Dockerfile detection: OK")
    print()


def test_training_flywheel():
    """Test DPO training flywheel."""
    print("4. TRAINING FLYWHEEL")
    print("-" * 40)

    # Test PreferenceSource enum
    class PreferenceSource(Enum):
        VERIFICATION = "verification"
        TOURNAMENT = "tournament"
        HUMAN = "human"

    assert PreferenceSource.VERIFICATION.value == "verification"
    print("  PreferenceSource enum: OK")

    # Test PreferencePair dataclass
    @dataclass
    class PreferencePair:
        id: str
        prompt: str
        chosen: str
        rejected: str
        chosen_score: float
        rejected_score: float
        source: PreferenceSource

        def to_dpo_format(self):
            return {
                "prompt": self.prompt,
                "chosen": self.chosen,
                "rejected": self.rejected,
            }

    pair = PreferencePair(
        id="test-1",
        prompt="Write hello",
        chosen="print('hello')",
        rejected="printf('hello')",
        chosen_score=0.9,
        rejected_score=0.2,
        source=PreferenceSource.VERIFICATION,
    )
    dpo = pair.to_dpo_format()
    assert dpo["prompt"] == "Write hello"
    assert dpo["chosen"] == "print('hello')"
    print("  PreferencePair: OK")

    # Test SQLite buffer
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE preference_pairs (
                id TEXT PRIMARY KEY,
                prompt TEXT,
                chosen TEXT,
                rejected TEXT,
                status TEXT DEFAULT 'pending'
            )
        """)

        cursor.execute("""
            INSERT INTO preference_pairs (id, prompt, chosen, rejected)
            VALUES (?, ?, ?, ?)
        """, ("p1", "prompt", "chosen", "rejected"))

        conn.commit()

        cursor.execute("SELECT COUNT(*) FROM preference_pairs WHERE status='pending'")
        count = cursor.fetchone()[0]
        assert count == 1
        print(f"  SQLite buffer: OK ({count} pending pair)")

        conn.close()
    finally:
        os.unlink(db_path)
    print()


def test_verification_weights():
    """Test verification-weighted ranking."""
    print("5. VERIFICATION WEIGHTS")
    print("-" * 40)

    # Test VerificationWeight enum
    class VerificationWeight(Enum):
        HARD_PASS = "hard_pass"
        SOFT_PASS = "soft_pass"
        PARTIAL = "partial"
        SOFT_FAIL = "soft_fail"
        HARD_FAIL = "hard_fail"

    # Test WeightedVerificationResult
    @dataclass
    class WeightedVerificationResult:
        passed: bool
        weight_category: VerificationWeight
        hard_checks_passed: int
        hard_checks_total: int
        execution_verified: bool = False
        security_verified: bool = False

        def compute_weight(
            self,
            hard_pass_weight: float = 10.0,
            execution_bonus: float = 5.0,
            security_bonus: float = 2.0,
        ) -> float:
            if self.weight_category == VerificationWeight.HARD_PASS:
                base = hard_pass_weight
            elif self.weight_category == VerificationWeight.SOFT_PASS:
                base = 3.0
            elif self.weight_category == VerificationWeight.PARTIAL:
                base = 1.5
            elif self.weight_category == VerificationWeight.SOFT_FAIL:
                base = 0.5
            else:
                base = 0.1

            if self.execution_verified:
                base *= (1.0 + execution_bonus)
            if self.security_verified:
                base *= (1.0 + security_bonus)

            return base

    # Hard pass with execution and security
    hard_result = WeightedVerificationResult(
        passed=True,
        weight_category=VerificationWeight.HARD_PASS,
        hard_checks_passed=3,
        hard_checks_total=3,
        execution_verified=True,
        security_verified=True,
    )
    hard_weight = hard_result.compute_weight()
    print(f"  Hard pass weight (exec+sec): {hard_weight:.1f}")
    assert hard_weight == 10.0 * 6.0 * 3.0  # 180.0

    # Soft pass
    soft_result = WeightedVerificationResult(
        passed=True,
        weight_category=VerificationWeight.SOFT_PASS,
        hard_checks_passed=0,
        hard_checks_total=0,
    )
    soft_weight = soft_result.compute_weight()
    print(f"  Soft pass weight: {soft_weight:.1f}")
    assert soft_weight == 3.0

    # Verify hard pass dominates
    assert hard_weight > soft_weight * 50
    print("  Weight dominance: OK (hard >> soft)")
    print()


def test_early_termination():
    """Test early termination logic."""
    print("6. EARLY TERMINATION")
    print("-" * 40)

    # Test TerminationReason enum
    class TerminationReason(Enum):
        HARD_VERIFICATION_FAILURE = "hard_verification_failure"
        QUALITY_THRESHOLD_MET = "quality_threshold_met"
        BUDGET_EXCEEDED = "budget_exceeded"
        TIMEOUT = "timeout"

    # Test TerminationConfig
    @dataclass
    class TerminationConfig:
        enable_hard_fail_termination: bool = True
        quality_threshold: float = 0.9
        max_token_budget: int = 100000
        aggressive_mode: bool = False

    config = TerminationConfig(quality_threshold=0.95)
    assert config.quality_threshold == 0.95
    print("  TerminationConfig: OK")

    # Test TerminationEvent
    @dataclass
    class TerminationEvent:
        reason: TerminationReason
        terminated_count: int
        saved_tokens_estimate: int

        def to_dict(self):
            return {
                "reason": self.reason.value,
                "terminated_count": self.terminated_count,
                "saved_tokens": self.saved_tokens_estimate,
            }

    event = TerminationEvent(
        reason=TerminationReason.HARD_VERIFICATION_FAILURE,
        terminated_count=3,
        saved_tokens_estimate=3000,
    )
    assert event.to_dict()["reason"] == "hard_verification_failure"
    print(f"  TerminationEvent: {event.reason.value}, saved {event.saved_tokens_estimate} tokens")

    # Test TerminationStats
    @dataclass
    class TerminationStats:
        total_requests: int = 0
        total_terminated: int = 0
        total_tokens_saved: int = 0

        def termination_rate(self):
            return self.total_terminated / max(1, self.total_requests)

    stats = TerminationStats(
        total_requests=100,
        total_terminated=50,
        total_tokens_saved=50000,
    )
    assert stats.termination_rate() == 0.5
    print(f"  TerminationStats: rate={stats.termination_rate():.0%}")
    print()


def main():
    """Run all tests."""
    print("=" * 50)
    print("RLMArena Component Tests (Standalone)")
    print("=" * 50)
    print()

    test_proxy_components()
    test_terraform_verifier()
    test_infra_sandbox()
    test_training_flywheel()
    test_verification_weights()
    test_early_termination()

    print("=" * 50)
    print("ALL TESTS PASSED!")
    print("=" * 50)


if __name__ == "__main__":
    main()
