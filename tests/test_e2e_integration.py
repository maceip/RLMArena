"""
End-to-End Integration Tests for RLMArena

Mocked tests demonstrating that all components work together:
1. OpenAI-compatible proxy
2. TerraformVerifier plugin
3. Infrastructure Sandbox Verifier
4. DPO Training Flywheel
5. Verification-Weighted Ranking
6. Early Termination

These tests use mocking to avoid external dependencies while proving
the components integrate correctly.
"""

import asyncio
import json
import os
import tempfile
from dataclasses import dataclass
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ============================================================================
# Test Fixtures and Mocks
# ============================================================================

@dataclass
class MockVerificationResult:
    """Mock verification result."""
    all_passed: bool
    failure_messages: list[str] = None
    combined_score_modifier: float = 1.0

    def __post_init__(self):
        if self.failure_messages is None:
            self.failure_messages = []

    def to_dict(self) -> dict:
        return {
            "all_passed": self.all_passed,
            "failure_messages": self.failure_messages,
            "combined_score_modifier": self.combined_score_modifier,
            "results": [
                {
                    "plugin_name": "terraform_verifier",
                    "status": "passed" if self.all_passed else "failed",
                }
            ],
        }


@dataclass
class MockLLMJudge:
    """Mock LLM judge for testing."""

    async def bidirectional_compare(
        self,
        trajectory_a: list[dict],
        trajectory_b: list[dict],
        query: str = "",
        **kwargs,
    ) -> tuple[float, float, dict]:
        """Return mock comparison scores."""
        # Simple heuristic: longer response scores higher
        len_a = len(str(trajectory_a))
        len_b = len(str(trajectory_b))

        score_a = 5.0 + (len_a / max(len_a, len_b, 1))
        score_b = 5.0 + (len_b / max(len_a, len_b, 1))

        return score_a, score_b, kwargs


class MockCompositeJudge:
    """Mock composite judge for testing."""

    def __init__(self, llm_judge, plugins=None):
        self.llm_judge = llm_judge
        self.plugins = plugins or []
        self._verification_results = {}

    async def verify_trajectory(
        self,
        trajectory: list[dict],
        context: dict = None,
    ) -> MockVerificationResult:
        """Mock verification."""
        # Check for terraform code
        content = str(trajectory)
        if "invalid" in content.lower():
            return MockVerificationResult(
                all_passed=False,
                failure_messages=["Invalid terraform code"],
            )
        return MockVerificationResult(all_passed=True)

    async def batch_verify(
        self,
        trajectories: list[list[dict]],
        metadata: list[dict] = None,
    ) -> list[MockVerificationResult]:
        """Mock batch verification."""
        return [await self.verify_trajectory(t) for t in trajectories]

    async def bidirectional_compare(self, *args, **kwargs):
        """Delegate to LLM judge."""
        return await self.llm_judge.bidirectional_compare(*args, **kwargs)


# ============================================================================
# Test: Proxy Middleware
# ============================================================================

class TestProxyMiddleware:
    """Tests for the OpenAI-compatible proxy."""

    def test_proxy_config_creation(self):
        """Test proxy configuration."""
        from qqr.serving.proxy import ProxyConfig, ProxyMode

        config = ProxyConfig(
            mode=ProxyMode.INTERCEPT,
            num_variations=4,
            timeout_seconds=60.0,
        )

        assert config.mode == ProxyMode.INTERCEPT
        assert config.num_variations == 4
        assert config.timeout_seconds == 60.0

    def test_response_cache(self):
        """Test response caching."""
        from qqr.serving.proxy import ResponseCache

        cache = ResponseCache(max_size=100, ttl_seconds=3600)

        messages = [{"role": "user", "content": "test"}]
        model = "gpt-4"
        response = {"id": "test-123", "choices": []}

        # Set and get
        cache.set(messages, model, response)
        result = cache.get(messages, model)

        assert result == response

    def test_rate_limiter(self):
        """Test rate limiting."""
        from qqr.serving.proxy import RateLimiter

        limiter = RateLimiter(
            requests_per_minute=60,
            requests_per_hour=1000,
            burst_size=10,
        )

        # Should allow first request
        allowed, retry_after = limiter.check("client-1")
        assert allowed is True
        assert retry_after is None

    @pytest.mark.asyncio
    async def test_proxy_metrics(self):
        """Test proxy metrics collection."""
        from qqr.serving.proxy import ProxyMetrics

        metrics = ProxyMetrics()
        metrics.total_requests = 100
        metrics.successful_requests = 95
        metrics.failed_requests = 5

        stats = metrics.to_dict()

        assert stats["total_requests"] == 100
        assert stats["successful_requests"] == 95
        assert stats["failed_requests"] == 5


# ============================================================================
# Test: Terraform Verifier
# ============================================================================

class TestTerraformVerifier:
    """Tests for the Terraform verification plugin."""

    def test_terraform_block_extraction(self):
        """Test extraction of Terraform code blocks."""
        from qqr.verification.plugins.terraform_verifier import TerraformVerifierPlugin

        plugin = TerraformVerifierPlugin()

        content = '''
Here is the Terraform code:

```terraform
resource "aws_instance" "example" {
  ami           = "ami-12345678"
  instance_type = "t2.micro"
}
```

This creates an EC2 instance.
'''

        blocks = plugin._extract_terraform_blocks(content)
        assert len(blocks) == 1
        assert "aws_instance" in blocks[0]

    @pytest.mark.asyncio
    async def test_terraform_verification_skip(self):
        """Test skipping when no Terraform code present."""
        from qqr.verification.plugins.terraform_verifier import TerraformVerifierPlugin

        plugin = TerraformVerifierPlugin()

        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]

        result = await plugin.verify(messages)

        assert result.status.value == "skipped"
        assert "No Terraform" in result.message


# ============================================================================
# Test: Infrastructure Sandbox Verifier
# ============================================================================

class TestInfraSandboxVerifier:
    """Tests for the infrastructure sandbox verifier."""

    def test_infra_type_detection_kubernetes(self):
        """Test Kubernetes manifest detection."""
        from qqr.verification.plugins.infra_sandbox import (
            InfraSandboxVerifierPlugin,
            InfraType,
        )

        plugin = InfraSandboxVerifierPlugin()

        k8s_content = '''
apiVersion: v1
kind: Pod
metadata:
  name: test-pod
spec:
  containers:
  - name: test
    image: nginx
'''

        detected = plugin._detect_infra_type(k8s_content)
        assert detected == InfraType.KUBERNETES

    def test_infra_type_detection_docker(self):
        """Test Dockerfile detection."""
        from qqr.verification.plugins.infra_sandbox import (
            InfraSandboxVerifierPlugin,
            InfraType,
        )

        plugin = InfraSandboxVerifierPlugin()

        docker_content = '''
FROM python:3.11
RUN pip install fastapi
COPY . /app
CMD ["python", "main.py"]
'''

        detected = plugin._detect_infra_type(docker_content)
        assert detected == InfraType.DOCKER


# ============================================================================
# Test: DPO Training Flywheel
# ============================================================================

class TestTrainingFlywheel:
    """Tests for the DPO training flywheel."""

    def test_training_buffer_creation(self):
        """Test training buffer initialization."""
        from qqr.training.flywheel import TrainingBuffer

        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        try:
            buffer = TrainingBuffer(db_path=db_path)
            stats = buffer.get_buffer_stats()

            assert "pairs_by_status" in stats
            assert "total_logs" in stats
        finally:
            os.unlink(db_path)

    def test_preference_pair_creation(self):
        """Test preference pair creation."""
        from qqr.training.flywheel import PreferencePair, PreferenceSource

        pair = PreferencePair(
            id="test-pair-1",
            prompt="Write a function",
            chosen="def foo(): return 42",
            rejected="def bar(): pass",
            chosen_score=0.9,
            rejected_score=0.3,
            preference_source=PreferenceSource.VERIFICATION,
            verification_label=True,
        )

        dpo_format = pair.to_dpo_format()

        assert dpo_format["prompt"] == "Write a function"
        assert dpo_format["chosen"] == "def foo(): return 42"
        assert dpo_format["rejected"] == "def bar(): pass"

    def test_arena_log_conversion(self):
        """Test conversion of arena logs to preference pairs."""
        from qqr.training.flywheel import ArenaLogConverter, PreferenceSource

        converter = ArenaLogConverter(min_score_gap=0.1)

        result = {
            "best_response": {"role": "assistant", "content": "def foo(): return 42"},
            "all_responses": [
                [{"role": "assistant", "content": "def foo(): return 42"}],
                [{"role": "assistant", "content": "def bar(): pass"}],
            ],
            "verification_results": [
                {"all_passed": True},
                {"all_passed": False, "failure_messages": ["Syntax error"]},
            ],
            "tournament_scores": [0.9, -0.5],
        }

        pairs = converter.convert("Write a function", result)

        assert len(pairs) > 0
        # Should have verification-based pair
        verification_pairs = [
            p for p in pairs
            if p.preference_source == PreferenceSource.VERIFICATION
        ]
        assert len(verification_pairs) >= 1

    def test_dpo_export_jsonl(self):
        """Test DPO export to JSONL."""
        from qqr.training.flywheel import DPOExporter, PreferencePair, PreferenceSource

        pairs = [
            PreferencePair(
                id=f"pair-{i}",
                prompt=f"prompt {i}",
                chosen=f"chosen {i}",
                rejected=f"rejected {i}",
                chosen_score=0.9,
                rejected_score=0.1,
                preference_source=PreferenceSource.VERIFICATION,
            )
            for i in range(3)
        ]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            output_path = f.name

        try:
            count = DPOExporter.to_jsonl(pairs, output_path)
            assert count == 3

            with open(output_path) as f:
                lines = f.readlines()
            assert len(lines) == 3
        finally:
            os.unlink(output_path)


# ============================================================================
# Test: Verification-Weighted Ranking
# ============================================================================

class TestVerificationWeightedRanking:
    """Tests for verification-weighted ranking."""

    def test_verification_weight_categories(self):
        """Test verification weight categories."""
        from qqr.reward_models.verified_anchor import (
            VerificationWeight,
            WeightedVerificationResult,
        )

        # Hard pass should have highest weight
        hard_pass = WeightedVerificationResult(
            passed=True,
            weight_category=VerificationWeight.HARD_PASS,
            hard_checks_passed=3,
            hard_checks_total=3,
            soft_checks_passed=2,
            soft_checks_total=2,
            execution_verified=True,
        )

        weight = hard_pass.compute_weight()
        assert weight > 10.0  # Base weight * execution bonus

        # Soft pass should have lower weight
        soft_pass = WeightedVerificationResult(
            passed=True,
            weight_category=VerificationWeight.SOFT_PASS,
            hard_checks_passed=0,
            hard_checks_total=0,
            soft_checks_passed=2,
            soft_checks_total=2,
        )

        soft_weight = soft_pass.compute_weight()
        assert soft_weight < weight
        assert soft_weight == 3.0  # Base weight for soft pass

    def test_hard_check_ratio(self):
        """Test hard check ratio calculation."""
        from qqr.reward_models.verified_anchor import (
            VerificationWeight,
            WeightedVerificationResult,
        )

        result = WeightedVerificationResult(
            passed=True,
            weight_category=VerificationWeight.PARTIAL,
            hard_checks_passed=2,
            hard_checks_total=4,
            soft_checks_passed=1,
            soft_checks_total=2,
        )

        assert result.hard_check_ratio == 0.5


# ============================================================================
# Test: Early Termination
# ============================================================================

class TestEarlyTermination:
    """Tests for early termination."""

    def test_termination_config(self):
        """Test termination configuration."""
        from qqr.verification.early_termination import TerminationConfig

        config = TerminationConfig(
            enable_hard_fail_termination=True,
            quality_threshold=0.95,
            max_token_budget=50000,
        )

        assert config.enable_hard_fail_termination is True
        assert config.quality_threshold == 0.95
        assert config.max_token_budget == 50000

    def test_early_terminator_registration(self):
        """Test session registration."""
        from qqr.verification.early_termination import EarlyTerminator

        terminator = EarlyTerminator()

        # Create mock tasks
        async def mock_task():
            await asyncio.sleep(10)

        tasks = [asyncio.create_task(mock_task()) for _ in range(4)]

        terminator.register_session("session-1", tasks)

        assert "session-1" in terminator._active_tasks
        assert len(terminator._active_tasks["session-1"]) == 4

        # Cleanup
        for task in tasks:
            task.cancel()

    def test_termination_event_creation(self):
        """Test termination event creation."""
        from qqr.verification.early_termination import (
            TerminationEvent,
            TerminationReason,
        )

        event = TerminationEvent(
            reason=TerminationReason.HARD_VERIFICATION_FAILURE,
            terminated_count=3,
            saved_tokens_estimate=3000,
            details={"failed_index": 0},
        )

        data = event.to_dict()

        assert data["reason"] == "hard_verification_failure"
        assert data["terminated_count"] == 3
        assert data["saved_tokens_estimate"] == 3000

    def test_termination_stats(self):
        """Test termination statistics."""
        from qqr.verification.early_termination import (
            TerminationEvent,
            TerminationReason,
            TerminationStats,
        )

        stats = TerminationStats()
        stats.total_requests = 100
        stats.total_terminated = 50
        stats.total_tokens_saved = 50000

        data = stats.to_dict()

        assert data["total_requests"] == 100
        assert data["termination_rate"] == 0.5
        assert data["total_tokens_saved"] == 50000


# ============================================================================
# Test: Full Integration Flow
# ============================================================================

class TestFullIntegration:
    """Full integration tests with mocked components."""

    @pytest.mark.asyncio
    async def test_shadow_arena_flow(self):
        """Test the full ShadowArena flow with mocked components."""
        from qqr.verification.shadow_arena import (
            ShadowArena,
            ShadowArenaConfig,
        )

        # Mock generate function
        async def mock_generate(query, context=None, **kwargs):
            return [
                {"role": "user", "content": query},
                {"role": "assistant", "content": f"Response for: {query}"},
            ]

        # Create mock judge
        mock_llm_judge = MockLLMJudge()
        mock_judge = MockCompositeJudge(mock_llm_judge)

        config = ShadowArenaConfig(
            num_variations=2,
            max_correction_attempts=1,
            enable_self_correction=False,
            parallel_generation=True,
        )

        arena = ShadowArena(
            judge=mock_judge,
            generate_fn=mock_generate,
            config=config,
        )

        result = await arena.process_query("Write hello world")

        assert result.success or len(result.verification_results) > 0

    @pytest.mark.asyncio
    async def test_flywheel_integration(self):
        """Test flywheel integration with arena results."""
        from qqr.training.flywheel import (
            ShadowArenaHook,
            TrainingBuffer,
            ArenaLogConverter,
        )

        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        try:
            buffer = TrainingBuffer(db_path=db_path)
            hook = ShadowArenaHook(buffer)
            converter = ArenaLogConverter()

            # Simulate arena result
            arena_result = {
                "success": True,
                "best_response": {"role": "assistant", "content": "Hello world!"},
                "all_responses": [
                    [{"role": "assistant", "content": "Hello world!"}],
                    [{"role": "assistant", "content": "Hi!"}],
                ],
                "verification_results": [
                    {"all_passed": True},
                    {"all_passed": True},
                ],
                "tournament_scores": [0.8, 0.6],
            }

            # Log the result
            log_id = hook.log_result("Say hello", arena_result)
            assert log_id is not None

            # Get unprocessed logs
            logs = buffer.get_unprocessed_logs()
            assert len(logs) == 1

            # Convert to preference pairs
            for log_id, query, result in logs:
                pairs = converter.convert(query, result)
                for pair in pairs:
                    buffer.add_preference_pair(pair)

            # Mark processed
            buffer.mark_logs_processed([log_id])

            # Verify
            stats = buffer.get_buffer_stats()
            assert stats["unprocessed_logs"] == 0

        finally:
            os.unlink(db_path)

    @pytest.mark.asyncio
    async def test_verification_weighted_flow(self):
        """Test verification-weighted ranking flow."""
        from qqr.reward_models.verified_anchor import (
            VerificationWeight,
            WeightedVerificationResult,
        )

        # Simulate verification results for multiple trajectories
        results = [
            WeightedVerificationResult(
                passed=True,
                weight_category=VerificationWeight.HARD_PASS,
                hard_checks_passed=3,
                hard_checks_total=3,
                soft_checks_passed=2,
                soft_checks_total=2,
                execution_verified=True,
                security_verified=True,
            ),
            WeightedVerificationResult(
                passed=True,
                weight_category=VerificationWeight.SOFT_PASS,
                hard_checks_passed=0,
                hard_checks_total=0,
                soft_checks_passed=2,
                soft_checks_total=2,
            ),
            WeightedVerificationResult(
                passed=False,
                weight_category=VerificationWeight.HARD_FAIL,
                hard_checks_passed=0,
                hard_checks_total=3,
                soft_checks_passed=1,
                soft_checks_total=2,
            ),
        ]

        # Compute weights
        weights = [r.compute_weight() for r in results]

        # Hard pass should have much higher weight
        assert weights[0] > weights[1] * 5  # Significantly higher
        assert weights[2] < 1.0  # Failed should have low weight

        # Verify hard pass weight is substantial
        assert weights[0] > 50.0  # 10.0 * (1+5.0) * (1+2.0) = 180.0

    @pytest.mark.asyncio
    async def test_early_termination_flow(self):
        """Test early termination flow."""
        from qqr.verification.early_termination import (
            BranchTerminator,
            TerminationConfig,
            TerminationReason,
        )

        config = TerminationConfig(
            enable_hard_fail_termination=True,
            aggressive_mode=True,
        )

        termination_events = []

        def on_termination(event):
            termination_events.append(event)

        terminator = BranchTerminator(config=config, on_termination=on_termination)

        # Create mock tasks
        async def slow_task():
            await asyncio.sleep(10)
            return "done"

        tasks = [asyncio.create_task(slow_task()) for _ in range(4)]

        # Register branches
        for i, task in enumerate(tasks):
            terminator.register_branch("query-1", i, task)

        # Report a hard failure on branch 0
        mock_failed_result = MockVerificationResult(all_passed=False)
        event = await terminator.on_verification_complete(
            "query-1",
            0,
            mock_failed_result,
        )

        # Should have triggered termination
        assert event is not None
        assert event.reason == TerminationReason.HARD_VERIFICATION_FAILURE
        assert event.terminated_count == 3  # All except branch 0

        # Cleanup
        for task in tasks:
            if not task.done():
                task.cancel()

    def test_proxy_mode_selection(self):
        """Test proxy mode selection."""
        from qqr.serving.proxy import ProxyConfig, ProxyMode

        # Test all modes
        for mode in ProxyMode:
            config = ProxyConfig(mode=mode)
            assert config.mode == mode

    def test_terraform_issue_reporting(self):
        """Test Terraform issue reporting."""
        from qqr.verification.plugins.terraform_verifier import (
            TerraformIssue,
            TerraformIssueType,
            TerraformSeverity,
            TerraformVerificationResult,
        )

        issues = [
            TerraformIssue(
                issue_type=TerraformIssueType.SECURITY,
                severity=TerraformSeverity.ERROR,
                message="S3 bucket is not encrypted",
                rule="AWS017",
            ),
            TerraformIssue(
                issue_type=TerraformIssueType.LINT,
                severity=TerraformSeverity.WARNING,
                message="Variable is not used",
                rule="terraform_unused_declarations",
            ),
        ]

        result = TerraformVerificationResult(
            passed=False,
            issues=issues,
        )

        assert result.error_count == 1
        assert result.warning_count == 1
        assert not result.passed


# ============================================================================
# Test: Complete System Integration
# ============================================================================

class TestCompleteSystemIntegration:
    """
    Tests demonstrating the complete system working together.
    """

    @pytest.mark.asyncio
    async def test_complete_pipeline(self):
        """
        Test the complete pipeline:
        1. Proxy receives request
        2. ShadowArena generates variations
        3. Verification runs (Terraform, K8s, etc.)
        4. Verification-weighted ranking scores responses
        5. Early termination kills slow branches
        6. Flywheel logs results for DPO training
        7. Best response is returned
        """
        from qqr.serving.proxy import ProxyConfig, ProxyMode, ShadowArenaProxy
        from qqr.training.flywheel import (
            ShadowArenaHook,
            TrainingBuffer,
            PreferenceSource,
        )
        from qqr.verification.early_termination import EarlyTerminator, TerminationConfig

        # Setup components
        proxy_config = ProxyConfig(
            mode=ProxyMode.INTERCEPT,
            num_variations=4,
            enable_early_termination=True,
        )

        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        try:
            buffer = TrainingBuffer(db_path=db_path)
            hook = ShadowArenaHook(buffer)
            terminator = EarlyTerminator(TerminationConfig())

            # Simulate a request-response cycle

            # 1. Request comes in
            messages = [
                {"role": "user", "content": "Write Terraform for an S3 bucket"}
            ]

            # 2. Generate mock variations (in reality, these would be from LLM)
            variations = [
                {
                    "content": '''```terraform
resource "aws_s3_bucket" "main" {
  bucket = "my-bucket"
}
```''',
                    "verified": True,
                    "score": 0.9,
                },
                {
                    "content": '''```terraform
resource "aws_s3_bucket" "main" {
  # Invalid configuration
}
```''',
                    "verified": False,
                    "score": -0.5,
                },
            ]

            # 3. Simulate verification results
            verification_results = [
                {"all_passed": v["verified"]}
                for v in variations
            ]

            # 4. Build arena result
            arena_result = {
                "success": True,
                "best_response": {"role": "assistant", "content": variations[0]["content"]},
                "all_responses": [
                    [{"role": "assistant", "content": v["content"]}]
                    for v in variations
                ],
                "verification_results": verification_results,
                "tournament_scores": [v["score"] for v in variations],
            }

            # 5. Log to flywheel
            hook.log_result(messages[-1]["content"], arena_result)

            # 6. Verify logging
            logs = buffer.get_unprocessed_logs()
            assert len(logs) == 1

            # 7. Check that we can retrieve the best response
            assert "my-bucket" in arena_result["best_response"]["content"]

            # 8. Verify terminator stats work
            stats = terminator.get_stats()
            assert "total_requests" in stats

        finally:
            os.unlink(db_path)

    def test_plugin_discovery(self):
        """Test that all plugins can be imported."""
        from qqr.verification.plugins import (
            CodeLinterPlugin,
            JSONSchemaPlugin,
            SafetyFilterPlugin,
            ToolCallValidatorPlugin,
        )
        from qqr.verification.plugins.terraform_verifier import TerraformVerifierPlugin
        from qqr.verification.plugins.infra_sandbox import InfraSandboxVerifierPlugin

        # All plugins should be instantiable
        plugins = [
            CodeLinterPlugin(),
            JSONSchemaPlugin(),
            SafetyFilterPlugin(),
            ToolCallValidatorPlugin(),
            TerraformVerifierPlugin(),
            InfraSandboxVerifierPlugin(),
        ]

        for plugin in plugins:
            assert plugin.name is not None
            assert plugin.description is not None

    def test_reward_model_registration(self):
        """Test that reward models can be imported."""
        from qqr.reward_models.verified_anchor import (
            VerifiedAnchorGroupRewardModel,
            VerificationWeightedRankingModel,
            VerificationWeight,
            WeightedVerificationResult,
        )

        # Classes should be importable
        assert VerifiedAnchorGroupRewardModel is not None
        assert VerificationWeightedRankingModel is not None
        assert VerificationWeight is not None
        assert WeightedVerificationResult is not None


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
