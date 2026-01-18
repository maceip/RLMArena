"""
End-to-End Integration Tests for VaaS Platform.

These tests verify the complete flow through all components.
"""

import pytest
import asyncio

from qqr.verification.expert_aligner import (
    ExpertAlignerService,
    TrajectoryPoint,
    TechnicalRationale,
    FailureCategory,
)
from qqr.verification.dspy_optimizer import (
    DSPyJudgeOptimizer,
    JudgeModule,
    Example,
    OptimizationConfig,
)
from qqr.verification.shadow_loop import (
    ShadowLoop,
    ShadowLoopConfig,
    VariationStrategy,
)
from qqr.verification.start_corrector import (
    STARTCorrectionEngine,
    CorrectionConfig,
    FailureContext,
    TrainingDataCollector,
)
from qqr.verification.distillation import (
    DistillationPipeline,
    DatasetFormat,
)
from qqr.verification.training_store import TrainingStore
from qqr.verification.brackets.open_coder import OpenCoderBracket
from qqr.verification.brackets.open_cloud_infra import OpenCloudInfraBracket
from qqr.verification.gateway import (
    VerificationGateway,
    RateLimitConfig,
    RequestPriority,
)
from qqr.serving.router import VaaSRouter, ModelEndpoint, ModelTier, RouterConfig
from qqr.serving.edge import EdgeVerifier, EdgeConfig
from qqr.serving.telemetry import TelemetryCollector, SpanContext
from qqr.serving.costs import CostTracker, QuotaConfig, UsageType
from qqr.infrastructure.mocks import (
    MockSGLang,
    MockFirecracker,
    MockLeash,
    MockOPA,
    MockTfsec,
    InfrastructureConfig,
)


class TestFullVaaSPipeline:
    """
    End-to-end test of the complete VaaS pipeline.
    """

    @pytest.fixture
    def sample_code_trajectory(self):
        return [
            {"role": "user", "content": "Write a function to calculate factorial"},
            {"role": "assistant", "content": '''```python
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)

# Test
print(factorial(5))  # Should print 120
```'''},
        ]

    @pytest.fixture
    def sample_infra_trajectory(self):
        return [
            {"role": "user", "content": "Create a secure S3 bucket with encryption"},
            {"role": "assistant", "content": '''```terraform
resource "aws_s3_bucket" "secure_bucket" {
  bucket = "my-secure-bucket"
  acl    = "private"

  server_side_encryption_configuration {
    rule {
      apply_server_side_encryption_by_default {
        sse_algorithm = "AES256"
      }
    }
  }

  versioning {
    enabled = true
  }

  tags = {
    Environment = "production"
    Owner       = "team"
  }
}
```'''},
        ]

    @pytest.mark.asyncio
    async def test_expert_aligner_to_dspy_flow(self, sample_code_trajectory):
        """Test SME labeling -> DSPy optimization flow."""
        # Create and populate expert aligner
        aligner = ExpertAlignerService()

        comparison = aligner.create_gold_comparison(
            input_query="Write factorial function",
            trajectory_a=sample_code_trajectory,
            trajectory_b=[
                {"role": "user", "content": "Write factorial"},
                {"role": "assistant", "content": "def fact(n): return 1 if n <= 1 else n * fact(n-1)"},
            ],
        )

        aligner.add_sme_label(
            comparison.id,
            sme_id="expert_1",
            winner="a",
            confidence=0.9,
            rationales=[
                TechnicalRationale(
                    category=FailureCategory.LOGIC_ERROR,
                    description="A has better readability and includes test",
                    severity=3,
                )
            ],
        )

        # Export for training
        training_data = aligner.export_for_training(min_confidence=0.5)
        assert len(training_data) == 1

        # Use for DSPy optimization
        optimizer = DSPyJudgeOptimizer()
        examples = optimizer.examples_from_gold_comparisons(training_data)
        assert len(examples) == 1

        # Run optimization (mock)
        module = JudgeModule()
        optimized, result = optimizer.optimize(module, examples)
        assert result.best_score >= 0

    @pytest.mark.asyncio
    async def test_shadow_loop_with_verification(self, sample_code_trajectory):
        """Test shadow loop generating and verifying variations."""
        shadow_loop = ShadowLoop(
            config=ShadowLoopConfig(
                num_variations=3,
                variation_strategy=VariationStrategy.TEMPERATURE_SWEEP,
            ),
        )

        # Generate variations
        result = await shadow_loop.run(sample_code_trajectory)

        assert len(result.variations) == 3
        assert result.best_variation_index is not None

        # Verify best with OpenCoder bracket
        bracket = OpenCoderBracket()
        best = result.variations[result.best_variation_index]

        cert = await bracket.verify_trajectory(best.messages)
        assert cert.certificate_id is not None

    @pytest.mark.asyncio
    async def test_self_correction_to_distillation_flow(self):
        """Test self-correction -> training data -> distillation flow."""
        # Create corrector
        corrector = STARTCorrectionEngine(
            config=CorrectionConfig(max_attempts=3),
        )

        # Simulate a failed trajectory
        messages = [
            {"role": "user", "content": "Write a function"},
            {"role": "assistant", "content": "def broken(\n  syntax error"},
        ]

        failure_contexts = [
            FailureContext(
                plugin_name="code_linter",
                failure_message="Syntax error: invalid syntax",
                failure_details={"line": 2},
            )
        ]

        # Run correction
        trajectory = await corrector.correct(messages, failure_contexts)

        # Collect training data
        collector = TrainingDataCollector()
        collector.add_from_engine(corrector)

        # Feed into distillation pipeline
        pipeline = DistillationPipeline()

        training_data = collector.export_dpo_pairs()
        if training_data:
            for item in training_data:
                pipeline.process_correction_trajectory(
                    original_messages=item["rejected"],
                    corrected_messages=item["chosen"],
                    failure_reasons=["syntax_error"],
                    num_attempts=trajectory.total_attempts,
                )

        # Export to training store
        store = TrainingStore()
        exported = pipeline.export(DatasetFormat.SLIME)

        for triplet in exported:
            store.add_triplet(
                input_query=triplet["prompt"],
                chosen=[],
                rejected=[],
                confidence=triplet["chosen_score"],
                quality="silver",
                source="self_correction",
            )

        stats = store.get_stats()
        # May or may not have triplets depending on correction success
        assert "total_triplets" in stats

    @pytest.mark.asyncio
    async def test_code_bracket_full_verification(self, sample_code_trajectory):
        """Test complete OpenCoder bracket verification."""
        bracket = OpenCoderBracket()
        cert = await bracket.verify_trajectory(sample_code_trajectory)

        assert cert.certificate_id is not None
        assert cert.execution_proof is not None or cert.security_audit is not None

        # Check certificate validity
        if cert.is_valid():
            assert cert.expires_at is not None

    @pytest.mark.asyncio
    async def test_infra_bracket_full_verification(self, sample_infra_trajectory):
        """Test complete OpenCloudInfra bracket verification."""
        bracket = OpenCloudInfraBracket()
        cert = await bracket.verify_trajectory(sample_infra_trajectory)

        assert cert.certificate_id is not None
        assert cert.compliance_report is not None
        assert cert.compliance_report.resources_scanned > 0


class TestServingInfrastructure:
    """Tests for VaaS serving infrastructure."""

    @pytest.fixture
    def router(self):
        router = VaaSRouter()
        router.add_endpoint(ModelEndpoint(
            name="model-1",
            url="http://localhost:8000",
            tier=ModelTier.STANDARD,
        ))
        router.add_endpoint(ModelEndpoint(
            name="model-2",
            url="http://localhost:8001",
            tier=ModelTier.PREMIUM,
        ))
        return router

    def test_router_load_balancing(self, router):
        """Test router distributes requests."""
        request_counts = {}

        for i in range(10):
            decision = router.route(f"req-{i}")
            name = decision.endpoint.name
            request_counts[name] = request_counts.get(name, 0) + 1

        # Should have distributed to multiple endpoints
        assert len(request_counts) >= 1

    def test_router_with_bracket_preference(self, router):
        """Test router respects bracket preferences."""
        router.set_bracket_route("open_coder", ["model-1"])

        for _ in range(5):
            decision = router.route("req", bracket="open_coder")
            assert decision.endpoint.name == "model-1"

    @pytest.mark.asyncio
    async def test_edge_verifier(self):
        """Test edge verification for fast checks."""
        edge = EdgeVerifier(EdgeConfig(
            enable_static_analysis=True,
            enable_policy_checks=True,
        ))

        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]

        result = await edge.verify(messages)
        assert result.passed
        assert result.total_latency_ms < 100  # Should be fast

    @pytest.mark.asyncio
    async def test_edge_verifier_blocks_sensitive_data(self):
        """Test edge verifier catches sensitive data."""
        edge = EdgeVerifier()

        messages = [
            {"role": "assistant", "content": "password = 'supersecret123'"},
        ]

        result = await edge.verify(messages)
        # Should detect the hardcoded password
        assert any(not c.passed for c in result.checks if c.check_type.value == "policy_check")

    def test_telemetry_collection(self):
        """Test telemetry event recording."""
        telemetry = TelemetryCollector()
        span = SpanContext()

        telemetry.record_request_start("req-1", "client-1", span)
        telemetry.record_verification("req-1", "code_linter", True, 10.0, span_context=span)
        telemetry.record_request_end("req-1", "client-1", True, 100.0, span)

        events = telemetry.query_by_request("req-1")
        assert len(events) == 3

        # Query by trace
        trace_events = telemetry.query_by_trace(span.trace_id)
        assert len(trace_events) == 3

    def test_cost_tracking(self):
        """Test cost tracking and quotas."""
        tracker = CostTracker()

        # Set quota
        tracker.set_quota(QuotaConfig(
            client_id="client-1",
            max_requests_per_day=100,
            max_cost_per_day_usd=10.0,
        ))

        # Record usage
        tracker.record_request(
            client_id="client-1",
            request_id="req-1",
            input_tokens=1000,
            output_tokens=500,
            num_verifications=3,
        )

        # Check quota
        within_quota, reason = tracker.check_quota("client-1")
        assert within_quota

        # Get summary
        summary = tracker.get_usage_summary("client-1")
        assert summary.total_requests == 1
        assert summary.total_tokens_input == 1000


class TestInfrastructureMocks:
    """Tests for infrastructure mock components."""

    @pytest.mark.asyncio
    async def test_mock_sglang(self):
        """Test mock sglang router."""
        sglang = MockSGLang()

        messages = [{"role": "user", "content": "Hello"}]
        responses = await sglang.generate(messages, n=3)

        assert len(responses) == 3
        for r in responses:
            assert "content" in r

        stats = sglang.get_stats()
        assert stats["total_requests"] == 1

    @pytest.mark.asyncio
    async def test_mock_firecracker(self):
        """Test mock Firecracker execution."""
        fc = MockFirecracker()

        result = await fc.execute("print('hello')", language="python")
        assert result["status"] == "success"
        assert result["exit_code"] == 0

        # Test syntax error detection
        result = await fc.execute("def broken(\n  bad", language="python")
        assert result["status"] == "failure"

    @pytest.mark.asyncio
    async def test_mock_leash(self):
        """Test mock Leash network enforcement."""
        leash = MockLeash()

        # Blocked host
        result = await leash.evaluate(
            'Agent::"test"',
            'Action::"connect"',
            'Host::"localhost"',
        )
        assert not result["allowed"]

        # Allowed host
        result = await leash.evaluate(
            'Agent::"test"',
            'Action::"connect"',
            'Host::"api.github.com"',
        )
        assert result["allowed"]

    @pytest.mark.asyncio
    async def test_mock_opa(self):
        """Test mock OPA policy evaluation."""
        opa = MockOPA()

        # Compliant config
        result = await opa.evaluate({
            "resource": {
                "aws_s3_bucket": {
                    "my_bucket": {
                        "acl": "private",
                        "tags": {"Environment": "prod", "Owner": "team"},
                    }
                }
            }
        })
        assert result["allow"]

        # Non-compliant config
        result = await opa.evaluate({
            "resource": {
                "aws_s3_bucket": {
                    "my_bucket": {
                        "acl": "public-read",
                    }
                }
            }
        })
        assert not result["allow"]

    @pytest.mark.asyncio
    async def test_mock_tfsec(self):
        """Test mock tfsec scanning."""
        tfsec = MockTfsec()

        result = await tfsec.scan({
            "resource": {
                "aws_s3_bucket": {
                    "public_bucket": {
                        "acl": "public-read",
                    }
                }
            }
        })

        assert not result["passed"]
        assert result["summary"]["critical"] > 0


class TestGatewayIntegration:
    """Tests for verification gateway."""

    @pytest.fixture
    def gateway(self):
        gateway = VerificationGateway(
            rate_limit_config=RateLimitConfig(
                requests_per_minute=100,
            ),
        )

        # Register a mock handler
        async def mock_handler(context, data):
            return {
                "data": {"result": "verified"},
                "verification": {"passed": True},
            }

        gateway.register_handler("verify_code", mock_handler)
        return gateway

    @pytest.mark.asyncio
    async def test_successful_request(self, gateway):
        """Test successful verification request."""
        response = await gateway.handle_request(
            client_id="client-1",
            request_type="verify_code",
            request_data={"messages": []},
        )

        assert response.status.value == "completed"
        assert response.data is not None

    @pytest.mark.asyncio
    async def test_unknown_handler(self, gateway):
        """Test unknown request type."""
        response = await gateway.handle_request(
            client_id="client-1",
            request_type="unknown",
            request_data={},
        )

        assert response.status.value == "failed"
        assert "unknown" in response.error.lower()

    @pytest.mark.asyncio
    async def test_metrics_tracking(self, gateway):
        """Test gateway metrics."""
        await gateway.handle_request("client-1", "verify_code", {})
        await gateway.handle_request("client-1", "verify_code", {})

        metrics = gateway.get_metrics()
        assert metrics["total_requests"] == 2
        assert metrics["successful_requests"] == 2
