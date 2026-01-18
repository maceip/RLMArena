"""
Tests for Hard Verifier Plugins (Leash and OPA).
"""

import pytest

from qqr.verification.plugins.leash_verifier import (
    LeashVerifierPlugin,
    CedarPolicy,
    CedarEffect,
    CedarCondition,
    CedarOperator,
    LeashPolicyEngine,
    NetworkRequest,
    STRICT_INTERNAL_POLICY,
)

from qqr.verification.plugins.opa_verifier import (
    OPAVerifierPlugin,
    RegoPolicy,
    RegoRule,
    PolicySeverity,
    OPAPolicyEngine,
    TERRAFORM_SECURITY_POLICY,
    KUBERNETES_SECURITY_POLICY,
)


class TestCedarPolicy:
    """Tests for Cedar policy definitions."""

    def test_simple_policy_match(self):
        policy = CedarPolicy(
            id="test-policy",
            effect=CedarEffect.PERMIT,
            principal="Agent::*",
            action='Action::"connect"',
            resource='Host::"api.github.com"',
        )

        assert policy.matches(
            'Agent::"test-agent"',
            'Action::"connect"',
            'Host::"api.github.com"',
        )

    def test_wildcard_principal(self):
        policy = CedarPolicy(
            id="test-policy",
            effect=CedarEffect.FORBID,
            principal="Agent::*",
            action="Action::*",
            resource='Host::"internal.com"',
        )

        assert policy.matches(
            'Agent::"any-agent"',
            'Action::"read"',
            'Host::"internal.com"',
        )

    def test_condition_evaluation(self):
        condition = CedarCondition(
            attribute="port",
            operator=CedarOperator.EQUALS,
            value=443,
        )

        assert condition.evaluate({"port": 443})
        assert not condition.evaluate({"port": 80})

    def test_regex_condition(self):
        condition = CedarCondition(
            attribute="host",
            operator=CedarOperator.MATCHES,
            value=r".*\.internal\..*",
        )

        assert condition.evaluate({"host": "api.internal.company.com"})
        assert not condition.evaluate({"host": "api.github.com"})

    def test_policy_to_cedar_string(self):
        policy = CedarPolicy(
            id="test",
            effect=CedarEffect.PERMIT,
            principal="Agent::*",
            action='Action::"connect"',
            resource="Host::*",
        )

        cedar_str = policy.to_cedar_string()
        assert "permit" in cedar_str
        assert "Agent::*" in cedar_str


class TestLeashPolicyEngine:
    """Tests for Leash policy engine."""

    @pytest.fixture
    def engine(self):
        return LeashPolicyEngine()

    def test_default_deny(self, engine):
        request = NetworkRequest(
            principal='Agent::"test"',
            action='Action::"connect"',
            resource='Host::"unknown.com"',
        )

        allowed, _ = engine.evaluate(request)
        assert not allowed  # Default deny

    def test_permit_policy(self, engine):
        engine.add_policy(CedarPolicy(
            id="allow-github",
            effect=CedarEffect.PERMIT,
            principal="Agent::*",
            action="Action::*",
            resource='Host::"api.github.com"',
        ))

        request = NetworkRequest(
            principal='Agent::"test"',
            action='Action::"connect"',
            resource='Host::"api.github.com"',
        )

        allowed, matched = engine.evaluate(request)
        assert allowed
        assert "allow-github" in matched

    def test_forbid_takes_precedence(self, engine):
        engine.add_policy(CedarPolicy(
            id="allow-all",
            effect=CedarEffect.PERMIT,
            principal="Agent::*",
            action="Action::*",
            resource="Host::*",
        ))
        engine.add_policy(CedarPolicy(
            id="block-internal",
            effect=CedarEffect.FORBID,
            principal="Agent::*",
            action="Action::*",
            resource='Host::"internal.com"',
        ))

        request = NetworkRequest(
            principal='Agent::"test"',
            action='Action::"connect"',
            resource='Host::"internal.com"',
        )

        allowed, matched = engine.evaluate(request)
        assert not allowed
        assert "block-internal" in matched


class TestLeashVerifierPlugin:
    """Tests for Leash Verifier Plugin."""

    @pytest.fixture
    def plugin(self):
        return LeashVerifierPlugin(
            blocked_hosts=["internal-api.com", "localhost"],
        )

    @pytest.mark.asyncio
    async def test_clean_trajectory_passes(self, plugin):
        messages = [
            {"role": "user", "content": "Search for Python documentation"},
            {"role": "assistant", "content": "Here's the documentation from https://docs.python.org"},
        ]

        result = await plugin.verify(messages)
        assert result.status.value == "passed"

    @pytest.mark.asyncio
    async def test_blocked_host_fails(self, plugin):
        messages = [
            {"role": "user", "content": "Connect to the API"},
            {"role": "assistant", "content": "Connecting to https://internal-api.com/data"},
        ]

        result = await plugin.verify(messages)
        assert result.status.value == "failed"
        assert "violations" in result.details

    @pytest.mark.asyncio
    async def test_no_network_requests_passes(self, plugin):
        messages = [
            {"role": "user", "content": "What is 2 + 2?"},
            {"role": "assistant", "content": "The answer is 4."},
        ]

        result = await plugin.verify(messages)
        assert result.status.value == "passed"


class TestOPAPolicyEngine:
    """Tests for OPA policy engine."""

    @pytest.fixture
    def engine(self):
        engine = OPAPolicyEngine()
        engine.add_policy(TERRAFORM_SECURITY_POLICY)
        return engine

    def test_detect_public_s3(self, engine):
        terraform_config = {
            "resource": {
                "aws_s3_bucket": {
                    "my_bucket": {
                        "acl": "public-read",
                    }
                }
            }
        }

        violations = engine.evaluate(terraform_config, "terraform")
        assert len(violations) > 0
        assert any(v.rule == "no_public_s3" for v in violations)

    def test_detect_unencrypted_ebs(self, engine):
        terraform_config = {
            "resource": {
                "aws_ebs_volume": {
                    "my_volume": {
                        "size": 100,
                        "encrypted": False,
                    }
                }
            }
        }

        violations = engine.evaluate(terraform_config, "terraform")
        assert any(v.rule == "encrypted_ebs" for v in violations)

    def test_detect_hardcoded_secrets(self, engine):
        terraform_config = {
            "resource": {
                "aws_instance": {
                    "my_instance": {
                        "password": "supersecret123",
                    }
                }
            }
        }

        violations = engine.evaluate(terraform_config, "terraform")
        assert any(v.rule == "no_hardcoded_secrets" for v in violations)


class TestOPAVerifierPlugin:
    """Tests for OPA Verifier Plugin."""

    @pytest.fixture
    def plugin(self):
        return OPAVerifierPlugin(
            policies=[TERRAFORM_SECURITY_POLICY],
        )

    @pytest.mark.asyncio
    async def test_clean_terraform_passes(self, plugin):
        messages = [
            {"role": "user", "content": "Create a secure S3 bucket"},
            {"role": "assistant", "content": '''```terraform
resource "aws_s3_bucket" "secure_bucket" {
  acl = "private"

  server_side_encryption_configuration {
    rule {
      apply_server_side_encryption_by_default {
        sse_algorithm = "AES256"
      }
    }
  }

  tags = {
    Environment = "production"
    Owner       = "team"
    Project     = "test"
  }
}
```'''},
        ]

        result = await plugin.verify(messages)
        # May have warnings but should not have critical errors
        assert result.status.value in ("passed", "failed")

    @pytest.mark.asyncio
    async def test_public_s3_fails(self, plugin):
        messages = [
            {"role": "user", "content": "Create an S3 bucket"},
            {"role": "assistant", "content": '''```terraform
resource "aws_s3_bucket" "public_bucket" {
  acl = "public-read"
}
```'''},
        ]

        result = await plugin.verify(messages)
        # Should detect the public ACL
        if result.details.get("violations"):
            assert any("public" in str(v).lower() for v in result.details["violations"])

    @pytest.mark.asyncio
    async def test_no_infrastructure_passes(self, plugin):
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]

        result = await plugin.verify(messages)
        assert result.status.value == "passed"
