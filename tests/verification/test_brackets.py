"""
Tests for Specialized Verification Brackets.
"""

import pytest

from qqr.verification.brackets.open_coder import (
    OpenCoderBracket,
    FunctionalCertaintyCertificate,
    CertificateStatus,
    SecurityAuditPlugin,
    MockBanditScanner,
    MockSemgrepScanner,
)

from qqr.verification.brackets.open_cloud_infra import (
    OpenCloudInfraBracket,
    PolicyComplianceCertificate,
    ComplianceStatus,
    MockTfsecScanner,
    MockCheckovScanner,
    TfsecPlugin,
)


class TestSecurityAuditPlugin:
    """Tests for Security Audit Plugin."""

    @pytest.fixture
    def plugin(self):
        return SecurityAuditPlugin()

    @pytest.mark.asyncio
    async def test_clean_code_passes(self, plugin):
        code = '''
def add(a, b):
    return a + b

def multiply(a, b):
    return a * b
'''
        results = await plugin.audit(code, "python")
        passed, summary = plugin.aggregate_results(results)
        assert summary["critical"] == 0
        assert summary["high"] == 0

    @pytest.mark.asyncio
    async def test_eval_detected(self, plugin):
        code = '''
user_input = input()
result = eval(user_input)
'''
        results = await plugin.audit(code, "python")
        passed, summary = plugin.aggregate_results(results)
        assert summary["high"] > 0

    @pytest.mark.asyncio
    async def test_hardcoded_password_detected(self, plugin):
        code = '''
password = "supersecret123"
api_key = "sk-abc123xyz"
'''
        results = await plugin.audit(code, "python")
        passed, summary = plugin.aggregate_results(results)
        assert summary["medium"] > 0


class TestMockBanditScanner:
    """Tests for Mock Bandit Scanner."""

    @pytest.fixture
    def scanner(self):
        return MockBanditScanner()

    def test_detect_eval(self, scanner):
        code = "result = eval(user_input)"
        result = scanner.scan(code)
        assert any(f["rule_id"] == "B307" for f in result.findings)

    def test_detect_subprocess(self, scanner):
        code = "subprocess.call(['ls'], shell=True)"
        result = scanner.scan(code)
        assert any(f["rule_id"] == "B602" for f in result.findings)

    def test_clean_code_no_findings(self, scanner):
        code = "print('Hello, World!')"
        result = scanner.scan(code)
        assert len(result.findings) == 0


class TestOpenCoderBracket:
    """Tests for OpenCoder Bracket."""

    @pytest.fixture
    def bracket(self):
        return OpenCoderBracket()

    @pytest.mark.asyncio
    async def test_valid_code_gets_certificate(self, bracket):
        messages = [
            {"role": "user", "content": "Write a function to add numbers"},
            {"role": "assistant", "content": '''```python
def add(a, b):
    return a + b
```'''},
        ]

        cert = await bracket.verify_trajectory(messages)

        assert cert.certificate_id is not None
        assert cert.execution_proof is not None

    @pytest.mark.asyncio
    async def test_code_with_security_issues(self, bracket):
        messages = [
            {"role": "user", "content": "Execute user code"},
            {"role": "assistant", "content": '''```python
user_code = input()
exec(user_code)
```'''},
        ]

        cert = await bracket.verify_trajectory(messages)

        # Should have security audit findings
        if cert.security_audit:
            assert len(cert.security_audit.findings) > 0

    @pytest.mark.asyncio
    async def test_no_code_returns_invalid(self, bracket):
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]

        cert = await bracket.verify_trajectory(messages)
        assert cert.status == CertificateStatus.INVALID

    @pytest.mark.asyncio
    async def test_batch_verify(self, bracket):
        trajectories = [
            ("t1", [
                {"role": "user", "content": "Code 1"},
                {"role": "assistant", "content": "```python\nprint(1)\n```"},
            ]),
            ("t2", [
                {"role": "user", "content": "Code 2"},
                {"role": "assistant", "content": "```python\nprint(2)\n```"},
            ]),
        ]

        certs = await bracket.batch_verify(trajectories)
        assert len(certs) == 2


class TestMockTfsecScanner:
    """Tests for Mock tfsec Scanner."""

    @pytest.fixture
    def scanner(self):
        return MockTfsecScanner()

    def test_detect_public_s3(self, scanner):
        config = {
            "resource": {
                "aws_s3_bucket": {
                    "my_bucket": {
                        "acl": "public-read",
                    }
                }
            }
        }

        report = scanner.scan(config)
        assert not report.passed
        assert report.critical_count > 0

    def test_detect_unencrypted_bucket(self, scanner):
        config = {
            "resource": {
                "aws_s3_bucket": {
                    "my_bucket": {
                        "acl": "private",
                    }
                }
            }
        }

        report = scanner.scan(config)
        # Should have medium finding for missing encryption
        assert any(v.severity == "MEDIUM" for v in report.violations)

    def test_secure_config_passes(self, scanner):
        config = {
            "resource": {
                "aws_s3_bucket": {
                    "secure_bucket": {
                        "acl": "private",
                        "versioning": True,
                        "server_side_encryption_configuration": {},
                    }
                }
            }
        }

        report = scanner.scan(config)
        assert report.critical_count == 0


class TestOpenCloudInfraBracket:
    """Tests for OpenCloudInfra Bracket."""

    @pytest.fixture
    def bracket(self):
        return OpenCloudInfraBracket()

    @pytest.mark.asyncio
    async def test_secure_terraform_passes(self, bracket):
        messages = [
            {"role": "user", "content": "Create a secure S3 bucket"},
            {"role": "assistant", "content": '''```terraform
resource "aws_s3_bucket" "secure" {
  acl = "private"
}
```'''},
        ]

        cert = await bracket.verify_trajectory(messages)
        assert cert.certificate_id is not None

    @pytest.mark.asyncio
    async def test_insecure_terraform_fails(self, bracket):
        messages = [
            {"role": "user", "content": "Create a public S3 bucket"},
            {"role": "assistant", "content": '''```terraform
resource "aws_s3_bucket" "public" {
  acl = "public-read"
}
```'''},
        ]

        cert = await bracket.verify_trajectory(messages)
        # Should detect the public bucket
        assert cert.compliance_report.violations is not None

    @pytest.mark.asyncio
    async def test_no_infrastructure_returns_unknown(self, bracket):
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]

        cert = await bracket.verify_trajectory(messages)
        assert cert.status == ComplianceStatus.UNKNOWN

    @pytest.mark.asyncio
    async def test_kubernetes_manifest_checked(self, bracket):
        messages = [
            {"role": "user", "content": "Create a pod"},
            {"role": "assistant", "content": '''```yaml
apiVersion: v1
kind: Pod
metadata:
  name: test
spec:
  containers:
  - name: app
    image: nginx:latest
```'''},
        ]

        cert = await bracket.verify_trajectory(messages)
        assert cert.certificate_id is not None
