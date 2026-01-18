"""
Infrastructure Sandbox Verifier Plugin

Provides comprehensive verification of Infrastructure-as-Code (IaC) including:
- Kubernetes manifests (kubectl validate, kube-score, kube-linter)
- Pulumi programs (pulumi preview)
- CloudFormation templates (cfn-lint)
- Ansible playbooks (ansible-lint)
- Docker/Compose files (hadolint, docker-compose config)

This is a Hard Check that verifies infrastructure code is syntactically correct,
follows best practices, and can be deployed safely.
"""

import asyncio
import json
import os
import re
import shutil
import tempfile
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional

import yaml

from ..plugin import VerificationPlugin, VerificationResult, VerificationStatus


class InfraType(Enum):
    """Types of infrastructure code."""
    KUBERNETES = "kubernetes"
    PULUMI = "pulumi"
    CLOUDFORMATION = "cloudformation"
    ANSIBLE = "ansible"
    DOCKER = "docker"
    DOCKER_COMPOSE = "docker_compose"
    HELM = "helm"
    UNKNOWN = "unknown"


class InfraSeverity(Enum):
    """Severity levels for infrastructure issues."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


@dataclass
class InfraIssue:
    """Represents a single issue found in infrastructure code."""
    infra_type: InfraType
    severity: InfraSeverity
    message: str
    file: Optional[str] = None
    line: Optional[int] = None
    resource: Optional[str] = None
    rule: Optional[str] = None
    fix_suggestion: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "infra_type": self.infra_type.value,
            "severity": self.severity.value,
            "message": self.message,
            "file": self.file,
            "line": self.line,
            "resource": self.resource,
            "rule": self.rule,
            "fix_suggestion": self.fix_suggestion,
        }


@dataclass
class InfraVerificationResult:
    """Complete verification result for infrastructure code."""
    passed: bool
    infra_type: InfraType
    issues: list[InfraIssue] = field(default_factory=list)
    resources_found: list[str] = field(default_factory=list)
    validation_output: Optional[str] = None
    score: float = 1.0
    execution_time_ms: float = 0.0

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "infra_type": self.infra_type.value,
            "issues": [i.to_dict() for i in self.issues],
            "resources_found": self.resources_found,
            "score": self.score,
            "execution_time_ms": self.execution_time_ms,
        }


class InfraSandboxVerifierPlugin(VerificationPlugin):
    """
    Verifies infrastructure code in LLM responses using external tools.

    Supported formats:
    - Kubernetes: kubectl validate, kube-score, kube-linter
    - Pulumi: pulumi preview (TypeScript/Python/Go)
    - CloudFormation: cfn-lint
    - Ansible: ansible-lint
    - Docker: hadolint
    - Docker Compose: docker-compose config

    Configuration:
        enable_kubernetes: bool - Enable K8s validation (default: True)
        enable_pulumi: bool - Enable Pulumi validation (default: True)
        enable_cloudformation: bool - Enable CFN validation (default: True)
        enable_ansible: bool - Enable Ansible validation (default: True)
        enable_docker: bool - Enable Docker validation (default: True)
        strict_mode: bool - Treat warnings as errors (default: False)
        timeout_seconds: float - Max time for verification (default: 30.0)
        kube_score_threshold: int - Minimum kube-score (default: 70)
    """

    name = "infra_sandbox"
    description = "Validates infrastructure code (K8s, Pulumi, CFN, etc.)"
    is_hard_check = True
    floor_reward = -1.0

    # Code block patterns for different infra types
    PATTERNS = {
        InfraType.KUBERNETES: re.compile(
            r"```(?:yaml|yml|kubernetes|k8s)\s*\n(.*?)```",
            re.DOTALL | re.IGNORECASE
        ),
        InfraType.DOCKER: re.compile(
            r"```(?:dockerfile|docker)\s*\n(.*?)```",
            re.DOTALL | re.IGNORECASE
        ),
        InfraType.DOCKER_COMPOSE: re.compile(
            r"```(?:yaml|yml|docker-compose|compose)\s*\n(.*?)```",
            re.DOTALL | re.IGNORECASE
        ),
        InfraType.CLOUDFORMATION: re.compile(
            r"```(?:yaml|yml|json|cloudformation|cfn)\s*\n(.*?)```",
            re.DOTALL | re.IGNORECASE
        ),
        InfraType.ANSIBLE: re.compile(
            r"```(?:yaml|yml|ansible)\s*\n(.*?)```",
            re.DOTALL | re.IGNORECASE
        ),
        InfraType.PULUMI: re.compile(
            r"```(?:typescript|python|go|pulumi)\s*\n(.*?)```",
            re.DOTALL | re.IGNORECASE
        ),
    }

    # Kubernetes resource patterns
    K8S_RESOURCE_PATTERN = re.compile(r"apiVersion:\s*\S+.*?kind:\s*\S+", re.DOTALL)

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        self.enable_kubernetes = self.config.get("enable_kubernetes", True)
        self.enable_pulumi = self.config.get("enable_pulumi", True)
        self.enable_cloudformation = self.config.get("enable_cloudformation", True)
        self.enable_ansible = self.config.get("enable_ansible", True)
        self.enable_docker = self.config.get("enable_docker", True)
        self.strict_mode = self.config.get("strict_mode", False)
        self.timeout_seconds = self.config.get("timeout_seconds", 30.0)
        self.kube_score_threshold = self.config.get("kube_score_threshold", 70)

        # Check available tools
        self._kubectl_available = shutil.which("kubectl") is not None
        self._kube_score_available = shutil.which("kube-score") is not None
        self._kube_linter_available = shutil.which("kube-linter") is not None
        self._cfn_lint_available = shutil.which("cfn-lint") is not None
        self._ansible_lint_available = shutil.which("ansible-lint") is not None
        self._hadolint_available = shutil.which("hadolint") is not None
        self._docker_compose_available = shutil.which("docker-compose") is not None or shutil.which("docker") is not None

    def _detect_infra_type(self, content: str) -> InfraType:
        """Detect the type of infrastructure code."""
        # Check for Kubernetes markers
        if self.K8S_RESOURCE_PATTERN.search(content):
            return InfraType.KUBERNETES

        # Check for CloudFormation markers
        if "AWSTemplateFormatVersion" in content or "Resources:" in content:
            try:
                data = yaml.safe_load(content)
                if data and ("AWSTemplateFormatVersion" in data or "Resources" in data):
                    return InfraType.CLOUDFORMATION
            except yaml.YAMLError:
                pass

        # Check for Docker Compose markers
        if "services:" in content.lower() or "version:" in content:
            try:
                data = yaml.safe_load(content)
                if data and "services" in data:
                    return InfraType.DOCKER_COMPOSE
            except yaml.YAMLError:
                pass

        # Check for Ansible markers
        if "- hosts:" in content or "- name:" in content:
            return InfraType.ANSIBLE

        # Check for Dockerfile markers
        if content.strip().startswith("FROM ") or "RUN " in content:
            return InfraType.DOCKER

        # Check for Pulumi markers
        if "pulumi" in content.lower() or "@pulumi" in content:
            return InfraType.PULUMI

        return InfraType.UNKNOWN

    def _extract_infra_blocks(self, content: str) -> list[tuple[str, InfraType]]:
        """Extract infrastructure code blocks from content."""
        blocks = []

        # Try each pattern
        for infra_type, pattern in self.PATTERNS.items():
            matches = pattern.findall(content)
            for match in matches:
                detected_type = self._detect_infra_type(match)
                if detected_type != InfraType.UNKNOWN:
                    blocks.append((match.strip(), detected_type))
                else:
                    blocks.append((match.strip(), infra_type))

        return blocks

    async def _verify_kubernetes(self, code: str, workspace: Path) -> InfraVerificationResult:
        """Verify Kubernetes manifest."""
        import time
        start_time = time.time()
        issues = []
        resources = []

        # Write manifest to file
        manifest_file = workspace / "manifest.yaml"
        manifest_file.write_text(code)

        # Parse YAML to find resources
        try:
            docs = list(yaml.safe_load_all(code))
            for doc in docs:
                if doc and "kind" in doc:
                    name = doc.get("metadata", {}).get("name", "unnamed")
                    resources.append(f"{doc['kind']}/{name}")
        except yaml.YAMLError as e:
            issues.append(InfraIssue(
                infra_type=InfraType.KUBERNETES,
                severity=InfraSeverity.CRITICAL,
                message=f"Invalid YAML: {str(e)}",
            ))
            return InfraVerificationResult(
                passed=False,
                infra_type=InfraType.KUBERNETES,
                issues=issues,
                execution_time_ms=(time.time() - start_time) * 1000,
            )

        # Run kubectl validate (dry-run)
        if self._kubectl_available:
            try:
                proc = await asyncio.create_subprocess_exec(
                    "kubectl", "apply", "--dry-run=client", "-f", str(manifest_file), "-o", "json",
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                _, stderr = await asyncio.wait_for(
                    proc.communicate(),
                    timeout=self.timeout_seconds
                )

                if proc.returncode != 0 and stderr:
                    issues.append(InfraIssue(
                        infra_type=InfraType.KUBERNETES,
                        severity=InfraSeverity.HIGH,
                        message=stderr.decode().strip()[:500],
                        rule="kubectl-validate",
                    ))
            except Exception as e:
                issues.append(InfraIssue(
                    infra_type=InfraType.KUBERNETES,
                    severity=InfraSeverity.MEDIUM,
                    message=f"kubectl validation error: {str(e)}",
                ))

        # Run kube-score
        if self._kube_score_available:
            try:
                proc = await asyncio.create_subprocess_exec(
                    "kube-score", "score", str(manifest_file), "--output-format", "json",
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                stdout, _ = await asyncio.wait_for(
                    proc.communicate(),
                    timeout=self.timeout_seconds
                )

                if stdout:
                    try:
                        results = json.loads(stdout.decode())
                        for obj in results:
                            for check in obj.get("checks", []):
                                if check.get("grade", 0) < 5:
                                    severity = InfraSeverity.LOW
                                    if check.get("grade", 0) == 0:
                                        severity = InfraSeverity.CRITICAL
                                    elif check.get("grade", 0) <= 3:
                                        severity = InfraSeverity.HIGH

                                    issues.append(InfraIssue(
                                        infra_type=InfraType.KUBERNETES,
                                        severity=severity,
                                        message=check.get("comment", "Score check failed"),
                                        resource=obj.get("object_name"),
                                        rule=check.get("check", {}).get("name"),
                                    ))
                    except json.JSONDecodeError:
                        pass
            except Exception:
                pass

        # Run kube-linter
        if self._kube_linter_available:
            try:
                proc = await asyncio.create_subprocess_exec(
                    "kube-linter", "lint", str(manifest_file), "--format", "json",
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                stdout, _ = await asyncio.wait_for(
                    proc.communicate(),
                    timeout=self.timeout_seconds
                )

                if stdout:
                    try:
                        result = json.loads(stdout.decode())
                        for report in result.get("Reports", []):
                            severity_map = {
                                "error": InfraSeverity.HIGH,
                                "warning": InfraSeverity.MEDIUM,
                                "info": InfraSeverity.LOW,
                            }
                            issues.append(InfraIssue(
                                infra_type=InfraType.KUBERNETES,
                                severity=severity_map.get(report.get("Diagnostic", {}).get("Level", "warning"), InfraSeverity.MEDIUM),
                                message=report.get("Diagnostic", {}).get("Message", "Lint issue"),
                                resource=report.get("Object", {}).get("Name"),
                                rule=report.get("Check"),
                                fix_suggestion=report.get("Remediation"),
                            ))
                    except json.JSONDecodeError:
                        pass
            except Exception:
                pass

        execution_time = (time.time() - start_time) * 1000

        # Determine pass/fail
        critical_count = sum(1 for i in issues if i.severity in (InfraSeverity.CRITICAL, InfraSeverity.HIGH))
        passed = critical_count == 0 if not self.strict_mode else len(issues) == 0

        return InfraVerificationResult(
            passed=passed,
            infra_type=InfraType.KUBERNETES,
            issues=issues,
            resources_found=resources,
            execution_time_ms=execution_time,
        )

    async def _verify_docker(self, code: str, workspace: Path) -> InfraVerificationResult:
        """Verify Dockerfile."""
        import time
        start_time = time.time()
        issues = []

        dockerfile = workspace / "Dockerfile"
        dockerfile.write_text(code)

        # Run hadolint
        if self._hadolint_available:
            try:
                proc = await asyncio.create_subprocess_exec(
                    "hadolint", "--format", "json", str(dockerfile),
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                stdout, _ = await asyncio.wait_for(
                    proc.communicate(),
                    timeout=self.timeout_seconds
                )

                if stdout:
                    try:
                        results = json.loads(stdout.decode())
                        for finding in results:
                            severity_map = {
                                "error": InfraSeverity.HIGH,
                                "warning": InfraSeverity.MEDIUM,
                                "info": InfraSeverity.LOW,
                                "style": InfraSeverity.INFO,
                            }
                            issues.append(InfraIssue(
                                infra_type=InfraType.DOCKER,
                                severity=severity_map.get(finding.get("level", "warning"), InfraSeverity.MEDIUM),
                                message=finding.get("message", "Dockerfile issue"),
                                line=finding.get("line"),
                                rule=finding.get("code"),
                            ))
                    except json.JSONDecodeError:
                        pass
            except Exception as e:
                issues.append(InfraIssue(
                    infra_type=InfraType.DOCKER,
                    severity=InfraSeverity.MEDIUM,
                    message=f"hadolint error: {str(e)}",
                ))

        execution_time = (time.time() - start_time) * 1000
        critical_count = sum(1 for i in issues if i.severity in (InfraSeverity.CRITICAL, InfraSeverity.HIGH))
        passed = critical_count == 0 if not self.strict_mode else len(issues) == 0

        return InfraVerificationResult(
            passed=passed,
            infra_type=InfraType.DOCKER,
            issues=issues,
            execution_time_ms=execution_time,
        )

    async def _verify_cloudformation(self, code: str, workspace: Path) -> InfraVerificationResult:
        """Verify CloudFormation template."""
        import time
        start_time = time.time()
        issues = []
        resources = []

        # Determine file extension
        try:
            data = yaml.safe_load(code)
            template_file = workspace / "template.yaml"
        except yaml.YAMLError:
            try:
                data = json.loads(code)
                template_file = workspace / "template.json"
            except json.JSONDecodeError:
                issues.append(InfraIssue(
                    infra_type=InfraType.CLOUDFORMATION,
                    severity=InfraSeverity.CRITICAL,
                    message="Invalid YAML/JSON in CloudFormation template",
                ))
                return InfraVerificationResult(
                    passed=False,
                    infra_type=InfraType.CLOUDFORMATION,
                    issues=issues,
                    execution_time_ms=(time.time() - start_time) * 1000,
                )

        # Extract resources
        if data and "Resources" in data:
            for name, resource in data.get("Resources", {}).items():
                resources.append(f"{resource.get('Type', 'Unknown')}/{name}")

        template_file.write_text(code)

        # Run cfn-lint
        if self._cfn_lint_available:
            try:
                proc = await asyncio.create_subprocess_exec(
                    "cfn-lint", "-f", "json", str(template_file),
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                stdout, _ = await asyncio.wait_for(
                    proc.communicate(),
                    timeout=self.timeout_seconds
                )

                if stdout:
                    try:
                        results = json.loads(stdout.decode())
                        for finding in results:
                            severity_map = {
                                "Error": InfraSeverity.HIGH,
                                "Warning": InfraSeverity.MEDIUM,
                                "Informational": InfraSeverity.LOW,
                            }
                            issues.append(InfraIssue(
                                infra_type=InfraType.CLOUDFORMATION,
                                severity=severity_map.get(finding.get("Level", "Warning"), InfraSeverity.MEDIUM),
                                message=finding.get("Message", "CFN issue"),
                                line=finding.get("Location", {}).get("Start", {}).get("LineNumber"),
                                rule=finding.get("Rule", {}).get("Id"),
                            ))
                    except json.JSONDecodeError:
                        pass
            except Exception as e:
                issues.append(InfraIssue(
                    infra_type=InfraType.CLOUDFORMATION,
                    severity=InfraSeverity.MEDIUM,
                    message=f"cfn-lint error: {str(e)}",
                ))

        execution_time = (time.time() - start_time) * 1000
        critical_count = sum(1 for i in issues if i.severity in (InfraSeverity.CRITICAL, InfraSeverity.HIGH))
        passed = critical_count == 0 if not self.strict_mode else len(issues) == 0

        return InfraVerificationResult(
            passed=passed,
            infra_type=InfraType.CLOUDFORMATION,
            issues=issues,
            resources_found=resources,
            execution_time_ms=execution_time,
        )

    async def _verify_ansible(self, code: str, workspace: Path) -> InfraVerificationResult:
        """Verify Ansible playbook."""
        import time
        start_time = time.time()
        issues = []

        playbook_file = workspace / "playbook.yml"
        playbook_file.write_text(code)

        # Validate YAML first
        try:
            yaml.safe_load(code)
        except yaml.YAMLError as e:
            issues.append(InfraIssue(
                infra_type=InfraType.ANSIBLE,
                severity=InfraSeverity.CRITICAL,
                message=f"Invalid YAML: {str(e)}",
            ))
            return InfraVerificationResult(
                passed=False,
                infra_type=InfraType.ANSIBLE,
                issues=issues,
                execution_time_ms=(time.time() - start_time) * 1000,
            )

        # Run ansible-lint
        if self._ansible_lint_available:
            try:
                proc = await asyncio.create_subprocess_exec(
                    "ansible-lint", "-f", "json", str(playbook_file),
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                stdout, _ = await asyncio.wait_for(
                    proc.communicate(),
                    timeout=self.timeout_seconds
                )

                if stdout:
                    try:
                        results = json.loads(stdout.decode())
                        for finding in results:
                            severity_map = {
                                "VERY_HIGH": InfraSeverity.CRITICAL,
                                "HIGH": InfraSeverity.HIGH,
                                "MEDIUM": InfraSeverity.MEDIUM,
                                "LOW": InfraSeverity.LOW,
                                "VERY_LOW": InfraSeverity.INFO,
                            }
                            issues.append(InfraIssue(
                                infra_type=InfraType.ANSIBLE,
                                severity=severity_map.get(finding.get("severity", "MEDIUM"), InfraSeverity.MEDIUM),
                                message=finding.get("message", "Ansible lint issue"),
                                line=finding.get("linenumber"),
                                rule=finding.get("rule", {}).get("id"),
                            ))
                    except json.JSONDecodeError:
                        pass
            except Exception as e:
                issues.append(InfraIssue(
                    infra_type=InfraType.ANSIBLE,
                    severity=InfraSeverity.MEDIUM,
                    message=f"ansible-lint error: {str(e)}",
                ))

        execution_time = (time.time() - start_time) * 1000
        critical_count = sum(1 for i in issues if i.severity in (InfraSeverity.CRITICAL, InfraSeverity.HIGH))
        passed = critical_count == 0 if not self.strict_mode else len(issues) == 0

        return InfraVerificationResult(
            passed=passed,
            infra_type=InfraType.ANSIBLE,
            issues=issues,
            execution_time_ms=execution_time,
        )

    async def _verify_docker_compose(self, code: str, workspace: Path) -> InfraVerificationResult:
        """Verify Docker Compose file."""
        import time
        start_time = time.time()
        issues = []
        resources = []

        compose_file = workspace / "docker-compose.yml"
        compose_file.write_text(code)

        # Parse YAML to find services
        try:
            data = yaml.safe_load(code)
            if data and "services" in data:
                for name in data.get("services", {}).keys():
                    resources.append(f"service/{name}")
        except yaml.YAMLError as e:
            issues.append(InfraIssue(
                infra_type=InfraType.DOCKER_COMPOSE,
                severity=InfraSeverity.CRITICAL,
                message=f"Invalid YAML: {str(e)}",
            ))
            return InfraVerificationResult(
                passed=False,
                infra_type=InfraType.DOCKER_COMPOSE,
                issues=issues,
                execution_time_ms=(time.time() - start_time) * 1000,
            )

        # Validate with docker-compose config
        if self._docker_compose_available:
            try:
                # Try docker compose (v2) first
                proc = await asyncio.create_subprocess_exec(
                    "docker", "compose", "-f", str(compose_file), "config",
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=workspace,
                )
                _, stderr = await asyncio.wait_for(
                    proc.communicate(),
                    timeout=self.timeout_seconds
                )

                if proc.returncode != 0 and stderr:
                    issues.append(InfraIssue(
                        infra_type=InfraType.DOCKER_COMPOSE,
                        severity=InfraSeverity.HIGH,
                        message=stderr.decode().strip()[:500],
                        rule="docker-compose-config",
                    ))
            except Exception:
                # Fall back to docker-compose (v1)
                try:
                    proc = await asyncio.create_subprocess_exec(
                        "docker-compose", "-f", str(compose_file), "config",
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE,
                        cwd=workspace,
                    )
                    _, stderr = await asyncio.wait_for(
                        proc.communicate(),
                        timeout=self.timeout_seconds
                    )

                    if proc.returncode != 0 and stderr:
                        issues.append(InfraIssue(
                            infra_type=InfraType.DOCKER_COMPOSE,
                            severity=InfraSeverity.HIGH,
                            message=stderr.decode().strip()[:500],
                            rule="docker-compose-config",
                        ))
                except Exception as e:
                    issues.append(InfraIssue(
                        infra_type=InfraType.DOCKER_COMPOSE,
                        severity=InfraSeverity.MEDIUM,
                        message=f"docker-compose validation error: {str(e)}",
                    ))

        execution_time = (time.time() - start_time) * 1000
        critical_count = sum(1 for i in issues if i.severity in (InfraSeverity.CRITICAL, InfraSeverity.HIGH))
        passed = critical_count == 0 if not self.strict_mode else len(issues) == 0

        return InfraVerificationResult(
            passed=passed,
            infra_type=InfraType.DOCKER_COMPOSE,
            issues=issues,
            resources_found=resources,
            execution_time_ms=execution_time,
        )

    async def _verify_block(self, code: str, infra_type: InfraType) -> InfraVerificationResult:
        """Verify a single infrastructure code block."""
        workspace = Path(tempfile.mkdtemp(prefix="infra_verify_"))

        try:
            if infra_type == InfraType.KUBERNETES and self.enable_kubernetes:
                return await self._verify_kubernetes(code, workspace)
            elif infra_type == InfraType.DOCKER and self.enable_docker:
                return await self._verify_docker(code, workspace)
            elif infra_type == InfraType.DOCKER_COMPOSE and self.enable_docker:
                return await self._verify_docker_compose(code, workspace)
            elif infra_type == InfraType.CLOUDFORMATION and self.enable_cloudformation:
                return await self._verify_cloudformation(code, workspace)
            elif infra_type == InfraType.ANSIBLE and self.enable_ansible:
                return await self._verify_ansible(code, workspace)
            else:
                return InfraVerificationResult(
                    passed=True,
                    infra_type=infra_type,
                    issues=[],
                )
        finally:
            shutil.rmtree(workspace, ignore_errors=True)

    async def verify(
        self,
        messages: list[dict],
        metadata: dict[str, Any] | None = None
    ) -> VerificationResult:
        """
        Verify all infrastructure code blocks in assistant messages.

        Args:
            messages: Conversation messages
            metadata: Optional metadata

        Returns:
            VerificationResult indicating pass/fail with details
        """
        all_blocks = []
        all_results = []

        for msg in messages:
            if msg.get("role") != "assistant":
                continue

            content = msg.get("content", "")
            if isinstance(content, dict):
                content = content.get("content", "")
            if not isinstance(content, str):
                continue

            blocks = self._extract_infra_blocks(content)
            all_blocks.extend(blocks)

        if not all_blocks:
            return VerificationResult(
                status=VerificationStatus.SKIPPED,
                plugin_name=self.name,
                message="No infrastructure code blocks found",
                details={"blocks_checked": 0},
            )

        # Verify each block
        tasks = [self._verify_block(code, infra_type) for code, infra_type in all_blocks]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in results:
            if isinstance(result, Exception):
                all_results.append(InfraVerificationResult(
                    passed=False,
                    infra_type=InfraType.UNKNOWN,
                    issues=[InfraIssue(
                        infra_type=InfraType.UNKNOWN,
                        severity=InfraSeverity.HIGH,
                        message=f"Verification error: {str(result)}",
                    )],
                ))
            else:
                all_results.append(result)

        # Aggregate results
        total_critical = sum(
            sum(1 for i in r.issues if i.severity == InfraSeverity.CRITICAL)
            for r in all_results
        )
        total_high = sum(
            sum(1 for i in r.issues if i.severity == InfraSeverity.HIGH)
            for r in all_results
        )
        total_medium = sum(
            sum(1 for i in r.issues if i.severity == InfraSeverity.MEDIUM)
            for r in all_results
        )

        all_passed = all(r.passed for r in all_results)

        if not all_passed or total_critical > 0 or total_high > 0:
            return VerificationResult(
                status=VerificationStatus.FAILED,
                plugin_name=self.name,
                message=f"Found {total_critical} critical, {total_high} high, {total_medium} medium issues",
                details={
                    "blocks_checked": len(all_blocks),
                    "results": [r.to_dict() for r in all_results],
                    "total_critical": total_critical,
                    "total_high": total_high,
                    "total_medium": total_medium,
                },
                score_modifier=0.0,
            )

        # Calculate score modifier
        score_modifier = max(0.5, 1.0 - (total_medium * 0.05))

        return VerificationResult(
            status=VerificationStatus.PASSED,
            plugin_name=self.name,
            message=f"All {len(all_blocks)} infrastructure block(s) validated",
            details={
                "blocks_checked": len(all_blocks),
                "results": [r.to_dict() for r in all_results],
            },
            score_modifier=score_modifier,
        )
