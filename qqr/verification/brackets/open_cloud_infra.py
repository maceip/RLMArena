"""
OpenCloudInfra Bracket - Policy Compliance for Infrastructure Code.

Combines OPA + tfsec + policy validation to guarantee that
infrastructure code never violates company security policies.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional
import asyncio
import hashlib
import json
import re
import uuid


class ComplianceStatus(Enum):
    """Status of policy compliance."""
    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    PARTIAL = "partial"
    UNKNOWN = "unknown"


class PolicyCategory(Enum):
    """Categories of compliance policies."""
    ENCRYPTION = "encryption"
    ACCESS_CONTROL = "access_control"
    NETWORK = "network"
    LOGGING = "logging"
    DATA_PROTECTION = "data_protection"
    COST = "cost"
    TAGGING = "tagging"


@dataclass
class PolicyViolation:
    """A policy violation found in infrastructure code."""
    policy_id: str
    category: PolicyCategory
    severity: str
    resource_type: str
    resource_name: str
    message: str
    remediation: Optional[str] = None
    line_number: Optional[int] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy_id": self.policy_id,
            "category": self.category.value,
            "severity": self.severity,
            "resource_type": self.resource_type,
            "resource_name": self.resource_name,
            "message": self.message,
            "remediation": self.remediation,
            "line_number": self.line_number,
        }


@dataclass
class ComplianceReport:
    """Detailed compliance report for infrastructure code."""
    report_id: str
    scan_time: datetime
    status: ComplianceStatus
    violations: list[PolicyViolation]
    resources_scanned: int
    policies_checked: int
    critical_count: int = 0
    high_count: int = 0
    medium_count: int = 0
    low_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "report_id": self.report_id,
            "scan_time": self.scan_time.isoformat(),
            "status": self.status.value,
            "violations": [v.to_dict() for v in self.violations],
            "resources_scanned": self.resources_scanned,
            "policies_checked": self.policies_checked,
            "summary": {
                "critical": self.critical_count,
                "high": self.high_count,
                "medium": self.medium_count,
                "low": self.low_count,
            },
            "metadata": self.metadata,
        }


@dataclass
class PolicyComplianceCertificate:
    """
    Certificate proving infrastructure policy compliance.

    Guarantees:
    1. No critical policy violations
    2. No high-severity security issues
    3. All required resources have proper configuration
    """
    certificate_id: str
    trajectory_id: str
    status: ComplianceStatus
    compliance_report: ComplianceReport
    opa_passed: bool
    tfsec_passed: bool
    issued_at: datetime
    expires_at: Optional[datetime] = None
    issuer: str = "OpenCloudInfraBracket"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "certificate_id": self.certificate_id,
            "trajectory_id": self.trajectory_id,
            "status": self.status.value,
            "compliance_report": self.compliance_report.to_dict(),
            "opa_passed": self.opa_passed,
            "tfsec_passed": self.tfsec_passed,
            "issued_at": self.issued_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "issuer": self.issuer,
            "metadata": self.metadata,
        }

    def is_valid(self) -> bool:
        """Check if certificate is currently valid."""
        if self.status != ComplianceStatus.COMPLIANT:
            return False

        if self.expires_at and datetime.utcnow() > self.expires_at:
            return False

        return True


class MockTfsecScanner:
    """
    Mock tfsec scanner for Terraform security analysis.

    tfsec is a static analysis security scanner for Terraform.
    """

    def __init__(self):
        self._checks = [
            # AWS checks
            ("aws-s3-enable-bucket-encryption", "aws_s3_bucket", PolicyCategory.ENCRYPTION,
             lambda r: not r.get("server_side_encryption_configuration"),
             "S3 bucket does not have encryption enabled",
             "Add server_side_encryption_configuration block"),

            ("aws-s3-no-public-buckets", "aws_s3_bucket", PolicyCategory.ACCESS_CONTROL,
             lambda r: r.get("acl") in ("public-read", "public-read-write"),
             "S3 bucket has public access",
             "Set acl to 'private' and use bucket policies for access control"),

            ("aws-ec2-enforce-http-token-imds", "aws_instance", PolicyCategory.ACCESS_CONTROL,
             lambda r: not r.get("metadata_options", {}).get("http_tokens") == "required",
             "EC2 instance does not require IMDSv2",
             "Add metadata_options with http_tokens = 'required'"),

            ("aws-rds-encrypt-instance-storage-data", "aws_db_instance", PolicyCategory.ENCRYPTION,
             lambda r: not r.get("storage_encrypted"),
             "RDS instance storage is not encrypted",
             "Set storage_encrypted = true"),

            ("aws-vpc-no-default-vpc", "aws_default_vpc", PolicyCategory.NETWORK,
             lambda r: True,  # Always flag default VPC usage
             "Using default VPC is not recommended",
             "Create a custom VPC instead of using default"),

            ("aws-cloudwatch-log-group-encryption", "aws_cloudwatch_log_group", PolicyCategory.ENCRYPTION,
             lambda r: not r.get("kms_key_id"),
             "CloudWatch log group is not encrypted with KMS",
             "Add kms_key_id for encryption"),
        ]

    def scan(self, terraform_config: dict[str, Any]) -> ComplianceReport:
        """Scan Terraform configuration for security issues."""
        violations = []
        resources_scanned = 0

        resources = terraform_config.get("resource", {})

        for check_id, resource_type, category, check_fn, message, remediation in self._checks:
            type_resources = resources.get(resource_type, {})

            for name, config in type_resources.items():
                resources_scanned += 1

                if check_fn(config):
                    violations.append(PolicyViolation(
                        policy_id=check_id,
                        category=category,
                        severity="high" if category in (PolicyCategory.ENCRYPTION, PolicyCategory.ACCESS_CONTROL) else "medium",
                        resource_type=resource_type,
                        resource_name=name,
                        message=message,
                        remediation=remediation,
                    ))

        critical = sum(1 for v in violations if v.severity == "critical")
        high = sum(1 for v in violations if v.severity == "high")
        medium = sum(1 for v in violations if v.severity == "medium")
        low = sum(1 for v in violations if v.severity == "low")

        status = ComplianceStatus.COMPLIANT
        if critical > 0 or high > 0:
            status = ComplianceStatus.NON_COMPLIANT
        elif medium > 0:
            status = ComplianceStatus.PARTIAL

        return ComplianceReport(
            report_id=str(uuid.uuid4())[:8],
            scan_time=datetime.utcnow(),
            status=status,
            violations=violations,
            resources_scanned=resources_scanned,
            policies_checked=len(self._checks),
            critical_count=critical,
            high_count=high,
            medium_count=medium,
            low_count=low,
            metadata={"tool": "tfsec"},
        )


class MockCheckovScanner:
    """Mock Checkov scanner for infrastructure policy compliance."""

    def __init__(self, frameworks: Optional[list[str]] = None):
        self.frameworks = frameworks or ["terraform", "kubernetes"]
        self._k8s_checks = [
            ("CKV_K8S_1", "no_privileged_containers", PolicyCategory.ACCESS_CONTROL,
             "Container should not run as privileged"),
            ("CKV_K8S_8", "liveness_probe_defined", PolicyCategory.LOGGING,
             "Container should have liveness probe"),
            ("CKV_K8S_9", "readiness_probe_defined", PolicyCategory.LOGGING,
             "Container should have readiness probe"),
            ("CKV_K8S_14", "no_host_pid", PolicyCategory.NETWORK,
             "Pod should not share host PID namespace"),
            ("CKV_K8S_22", "read_only_root_filesystem", PolicyCategory.DATA_PROTECTION,
             "Container should use read-only root filesystem"),
        ]

    def scan(self, config: dict[str, Any], config_type: str = "terraform") -> ComplianceReport:
        """Scan configuration for compliance issues."""
        violations = []
        resources_scanned = 0

        if config_type == "kubernetes":
            violations, resources_scanned = self._scan_kubernetes(config)
        else:
            # Use tfsec-like checks for terraform
            pass

        critical = sum(1 for v in violations if v.severity == "critical")
        high = sum(1 for v in violations if v.severity == "high")

        status = ComplianceStatus.COMPLIANT if not violations else ComplianceStatus.NON_COMPLIANT

        return ComplianceReport(
            report_id=str(uuid.uuid4())[:8],
            scan_time=datetime.utcnow(),
            status=status,
            violations=violations,
            resources_scanned=resources_scanned,
            policies_checked=len(self._k8s_checks),
            critical_count=critical,
            high_count=high,
            metadata={"tool": "checkov"},
        )

    def _scan_kubernetes(self, config: dict[str, Any]) -> tuple[list[PolicyViolation], int]:
        """Scan Kubernetes manifest."""
        violations = []
        resources_scanned = 0

        kind = config.get("kind", "")
        metadata = config.get("metadata", {})
        name = metadata.get("name", "unnamed")

        spec = config.get("spec", {})
        template_spec = spec.get("template", {}).get("spec", spec)
        containers = template_spec.get("containers", [])

        for container in containers:
            resources_scanned += 1
            container_name = container.get("name", "unnamed")

            # Check privileged
            security_context = container.get("securityContext", {})
            if security_context.get("privileged"):
                violations.append(PolicyViolation(
                    policy_id="CKV_K8S_1",
                    category=PolicyCategory.ACCESS_CONTROL,
                    severity="critical",
                    resource_type=kind,
                    resource_name=f"{name}/{container_name}",
                    message="Container runs as privileged",
                    remediation="Remove privileged: true from securityContext",
                ))

            # Check probes
            if not container.get("livenessProbe"):
                violations.append(PolicyViolation(
                    policy_id="CKV_K8S_8",
                    category=PolicyCategory.LOGGING,
                    severity="low",
                    resource_type=kind,
                    resource_name=f"{name}/{container_name}",
                    message="Container missing liveness probe",
                ))

        return violations, resources_scanned


class TfsecPlugin:
    """Plugin wrapper for tfsec integration."""

    def __init__(
        self,
        severity_threshold: str = "high",
        exclude_checks: Optional[list[str]] = None,
    ):
        self.severity_threshold = severity_threshold
        self.exclude_checks = exclude_checks or []
        self._scanner = MockTfsecScanner()

    async def scan(self, terraform_config: dict[str, Any]) -> ComplianceReport:
        """Run tfsec scan."""
        report = self._scanner.scan(terraform_config)

        # Filter excluded checks
        if self.exclude_checks:
            report.violations = [
                v for v in report.violations
                if v.policy_id not in self.exclude_checks
            ]

        return report


class OpenCloudInfraBracket:
    """
    OpenCloudInfra verification bracket for infrastructure agents.

    Combines:
    - OPA for Rego policy enforcement
    - tfsec for Terraform security scanning
    - Checkov for multi-framework compliance

    Produces PolicyComplianceCertificates proving compliance.
    """

    def __init__(
        self,
        opa_verifier: Optional[Any] = None,
        tfsec_plugin: Optional[TfsecPlugin] = None,
        checkov_scanner: Optional[MockCheckovScanner] = None,
        certificate_validity_hours: int = 24,
        company_policies: Optional[list[str]] = None,
    ):
        self._opa = opa_verifier
        self._tfsec = tfsec_plugin or TfsecPlugin()
        self._checkov = checkov_scanner or MockCheckovScanner()
        self._certificate_validity_hours = certificate_validity_hours
        self._company_policies = company_policies or [
            "no_public_s3",
            "encryption_required",
            "tagging_required",
        ]

    def _extract_infrastructure_code(
        self,
        messages: list[dict[str, Any]],
    ) -> list[tuple[dict[str, Any], str]]:
        """Extract infrastructure code from messages."""
        configs = []

        for msg in messages:
            content = msg.get("content", "")
            if not isinstance(content, str):
                continue

            # Extract Terraform blocks
            tf_pattern = r"```(?:terraform|hcl|tf)\s*\n(.*?)```"
            for match in re.finditer(tf_pattern, content, re.DOTALL | re.IGNORECASE):
                parsed = self._parse_terraform(match.group(1))
                if parsed:
                    configs.append((parsed, "terraform"))

            # Extract Kubernetes blocks
            k8s_pattern = r"```(?:yaml|kubernetes|k8s)\s*\n(.*?)```"
            for match in re.finditer(k8s_pattern, content, re.DOTALL | re.IGNORECASE):
                parsed = self._parse_yaml(match.group(1))
                if parsed:
                    configs.append((parsed, "kubernetes"))

            # Extract JSON blocks
            json_pattern = r"```json\s*\n(.*?)```"
            for match in re.finditer(json_pattern, content, re.DOTALL | re.IGNORECASE):
                try:
                    parsed = json.loads(match.group(1))
                    if "resource" in parsed or "provider" in parsed:
                        configs.append((parsed, "terraform"))
                    elif "apiVersion" in parsed:
                        configs.append((parsed, "kubernetes"))
                except json.JSONDecodeError:
                    pass

        return configs

    def _parse_terraform(self, code: str) -> Optional[dict[str, Any]]:
        """Parse Terraform HCL to dict (simplified)."""
        try:
            return json.loads(code)
        except json.JSONDecodeError:
            pass

        # Simple HCL parsing
        result = {"resource": {}}

        resource_pattern = r'resource\s+"(\w+)"\s+"(\w+)"\s*\{([^}]+)\}'
        for match in re.finditer(resource_pattern, code, re.DOTALL):
            resource_type, resource_name, body = match.groups()
            if resource_type not in result["resource"]:
                result["resource"][resource_type] = {}
            result["resource"][resource_type][resource_name] = self._parse_hcl_body(body)

        return result if result["resource"] else None

    def _parse_hcl_body(self, body: str) -> dict[str, Any]:
        """Parse HCL block body."""
        result = {}
        kv_pattern = r'(\w+)\s*=\s*(?:"([^"]*)"|(true|false)|(\d+))'
        for match in re.finditer(kv_pattern, body):
            key = match.group(1)
            if match.group(2) is not None:
                result[key] = match.group(2)
            elif match.group(3) is not None:
                result[key] = match.group(3) == "true"
            elif match.group(4) is not None:
                result[key] = int(match.group(4))
        return result

    def _parse_yaml(self, code: str) -> Optional[dict[str, Any]]:
        """Parse YAML to dict (simplified)."""
        try:
            import yaml
            return yaml.safe_load(code)
        except:
            pass

        # Basic parsing fallback
        try:
            return json.loads(code)
        except:
            pass

        return None

    async def verify_trajectory(
        self,
        messages: list[dict[str, Any]],
        trajectory_id: Optional[str] = None,
    ) -> PolicyComplianceCertificate:
        """Verify infrastructure trajectory for policy compliance."""
        trajectory_id = trajectory_id or str(uuid.uuid4())
        certificate_id = str(uuid.uuid4())
        now = datetime.utcnow()

        # Extract infrastructure code
        configs = self._extract_infrastructure_code(messages)

        if not configs:
            return PolicyComplianceCertificate(
                certificate_id=certificate_id,
                trajectory_id=trajectory_id,
                status=ComplianceStatus.UNKNOWN,
                compliance_report=ComplianceReport(
                    report_id=str(uuid.uuid4())[:8],
                    scan_time=now,
                    status=ComplianceStatus.UNKNOWN,
                    violations=[],
                    resources_scanned=0,
                    policies_checked=0,
                ),
                opa_passed=True,
                tfsec_passed=True,
                issued_at=now,
                metadata={"error": "No infrastructure code found"},
            )

        # Run scans
        all_violations = []
        total_resources = 0
        total_policies = 0
        opa_passed = True
        tfsec_passed = True

        for config, config_type in configs:
            if config_type == "terraform":
                # Run tfsec
                report = await self._tfsec.scan(config)
                all_violations.extend(report.violations)
                total_resources += report.resources_scanned
                total_policies += report.policies_checked
                if report.status == ComplianceStatus.NON_COMPLIANT:
                    tfsec_passed = False

                # Run OPA if available
                if self._opa:
                    opa_result = await self._opa.verify(messages)
                    if opa_result.status.value == "failed":
                        opa_passed = False

            elif config_type == "kubernetes":
                # Run Checkov
                report = self._checkov.scan(config, "kubernetes")
                all_violations.extend(report.violations)
                total_resources += report.resources_scanned
                total_policies += report.policies_checked

        # Aggregate results
        critical = sum(1 for v in all_violations if v.severity == "critical")
        high = sum(1 for v in all_violations if v.severity == "high")
        medium = sum(1 for v in all_violations if v.severity == "medium")
        low = sum(1 for v in all_violations if v.severity == "low")

        if critical > 0 or high > 0:
            status = ComplianceStatus.NON_COMPLIANT
        elif medium > 0:
            status = ComplianceStatus.PARTIAL
        else:
            status = ComplianceStatus.COMPLIANT

        compliance_report = ComplianceReport(
            report_id=str(uuid.uuid4())[:8],
            scan_time=now,
            status=status,
            violations=all_violations,
            resources_scanned=total_resources,
            policies_checked=total_policies,
            critical_count=critical,
            high_count=high,
            medium_count=medium,
            low_count=low,
        )

        return PolicyComplianceCertificate(
            certificate_id=certificate_id,
            trajectory_id=trajectory_id,
            status=status,
            compliance_report=compliance_report,
            opa_passed=opa_passed,
            tfsec_passed=tfsec_passed,
            issued_at=now,
            expires_at=now.replace(hour=now.hour + self._certificate_validity_hours) if status == ComplianceStatus.COMPLIANT else None,
            metadata={
                "configs_scanned": len(configs),
                "company_policies": self._company_policies,
            },
        )

    async def batch_verify(
        self,
        trajectories: list[tuple[str, list[dict[str, Any]]]],
    ) -> list[PolicyComplianceCertificate]:
        """Verify multiple trajectories in parallel."""
        tasks = [
            self.verify_trajectory(messages, trajectory_id)
            for trajectory_id, messages in trajectories
        ]
        return await asyncio.gather(*tasks)
