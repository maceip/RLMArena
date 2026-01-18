"""
Terraform Infrastructure Verifier Plugin

Provides deterministic verification of Terraform code using external CLI tools:
- tflint for linting and best practices
- terraform validate for syntax validation
- terraform plan for deployability checking
- tfsec for security scanning

This is a Hard Check - trajectories with invalid/insecure infrastructure code
are considered Dead on Arrival (DoA) for OpenCloudInfra brackets.
"""

import asyncio
import hashlib
import json
import os
import re
import shutil
import tempfile
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional

from ..plugin import VerificationPlugin, VerificationResult, VerificationStatus


class TerraformSeverity(Enum):
    """Severity levels for Terraform issues."""
    ERROR = "error"
    WARNING = "warning"
    NOTICE = "notice"
    INFO = "info"


class TerraformIssueType(Enum):
    """Types of Terraform issues."""
    SYNTAX = "syntax"
    LINT = "lint"
    SECURITY = "security"
    VALIDATION = "validation"
    PLAN = "plan"
    COMPLIANCE = "compliance"


@dataclass
class TerraformIssue:
    """Represents a single issue found in Terraform code."""
    issue_type: TerraformIssueType
    severity: TerraformSeverity
    message: str
    file: Optional[str] = None
    line: Optional[int] = None
    column: Optional[int] = None
    rule: Optional[str] = None
    fix_suggestion: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "type": self.issue_type.value,
            "severity": self.severity.value,
            "message": self.message,
            "file": self.file,
            "line": self.line,
            "column": self.column,
            "rule": self.rule,
            "fix_suggestion": self.fix_suggestion,
        }


@dataclass
class TerraformVerificationResult:
    """Complete verification result for Terraform code."""
    passed: bool
    issues: list[TerraformIssue] = field(default_factory=list)
    plan_summary: Optional[dict] = None
    security_findings: list[dict] = field(default_factory=list)
    resources_to_create: int = 0
    resources_to_update: int = 0
    resources_to_destroy: int = 0
    execution_time_ms: float = 0.0

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "issues": [i.to_dict() for i in self.issues],
            "plan_summary": self.plan_summary,
            "security_findings": self.security_findings,
            "resources": {
                "create": self.resources_to_create,
                "update": self.resources_to_update,
                "destroy": self.resources_to_destroy,
            },
            "execution_time_ms": self.execution_time_ms,
        }

    @property
    def error_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == TerraformSeverity.ERROR)

    @property
    def warning_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == TerraformSeverity.WARNING)


class TerraformVerifierPlugin(VerificationPlugin):
    """
    Verifies Terraform code in LLM responses for correctness and security.

    This plugin extracts Terraform code blocks from assistant messages and
    validates them using external tools:

    1. tflint - Linting and best practices
    2. terraform validate - Syntax validation
    3. terraform plan - Deployability check (optional, requires providers)
    4. tfsec - Security scanning

    Configuration:
        enable_tflint: bool - Enable tflint checks (default: True)
        enable_validate: bool - Enable terraform validate (default: True)
        enable_plan: bool - Enable terraform plan (default: False)
        enable_tfsec: bool - Enable tfsec security scanning (default: True)
        strict_mode: bool - Treat warnings as errors (default: False)
        timeout_seconds: float - Max time for verification (default: 30.0)
        temp_dir: str - Custom temp directory for Terraform files
        required_providers: dict - Provider versions to use
    """

    name = "terraform_verifier"
    description = "Validates Terraform infrastructure code"
    is_hard_check = True
    floor_reward = -1.0

    # Patterns to extract Terraform code blocks
    TF_BLOCK_PATTERN = re.compile(
        r"```(?:terraform|tf|hcl)\s*\n(.*?)```",
        re.DOTALL | re.IGNORECASE
    )

    # Also detect inline HCL blocks that might not be marked
    HCL_RESOURCE_PATTERN = re.compile(
        r'(resource\s+"[^"]+"\s+"[^"]+"\s*\{.*?\n\})',
        re.DOTALL
    )

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        self.enable_tflint = self.config.get("enable_tflint", True)
        self.enable_validate = self.config.get("enable_validate", True)
        self.enable_plan = self.config.get("enable_plan", False)
        self.enable_tfsec = self.config.get("enable_tfsec", True)
        self.strict_mode = self.config.get("strict_mode", False)
        self.timeout_seconds = self.config.get("timeout_seconds", 30.0)
        self.temp_dir = self.config.get("temp_dir", None)
        self.required_providers = self.config.get("required_providers", {
            "aws": "~> 5.0",
            "azurerm": "~> 3.0",
            "google": "~> 5.0",
        })

        # Check for available tools
        self._tflint_available = shutil.which("tflint") is not None
        self._terraform_available = shutil.which("terraform") is not None
        self._tfsec_available = shutil.which("tfsec") is not None

    def _extract_terraform_blocks(self, content: str) -> list[str]:
        """Extract Terraform code blocks from content."""
        blocks = []

        # Extract fenced code blocks
        matches = self.TF_BLOCK_PATTERN.findall(content)
        blocks.extend([m.strip() for m in matches if m.strip()])

        return blocks

    def _create_temp_workspace(self, terraform_code: str) -> Path:
        """Create a temporary workspace with Terraform files."""
        if self.temp_dir:
            workspace = Path(tempfile.mkdtemp(dir=self.temp_dir))
        else:
            workspace = Path(tempfile.mkdtemp(prefix="tf_verify_"))

        # Write main.tf
        main_tf = workspace / "main.tf"
        main_tf.write_text(terraform_code)

        # Generate versions.tf with required providers
        versions_content = self._generate_versions_tf()
        versions_tf = workspace / "versions.tf"
        versions_tf.write_text(versions_content)

        return workspace

    def _generate_versions_tf(self) -> str:
        """Generate a versions.tf file with required providers."""
        providers = []
        for name, version in self.required_providers.items():
            namespace = "hashicorp"
            if name == "azurerm":
                namespace = "hashicorp"
            elif name == "google":
                namespace = "hashicorp"

            providers.append(f'''    {name} = {{
      source  = "{namespace}/{name}"
      version = "{version}"
    }}''')

        return f'''terraform {{
  required_version = ">= 1.0"
  required_providers {{
{chr(10).join(providers)}
  }}
}}
'''

    async def _run_tflint(self, workspace: Path) -> list[TerraformIssue]:
        """Run tflint on the workspace."""
        issues = []

        if not self._tflint_available:
            return issues

        try:
            # Initialize tflint first
            init_proc = await asyncio.create_subprocess_exec(
                "tflint", "--init",
                cwd=workspace,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await asyncio.wait_for(init_proc.communicate(), timeout=self.timeout_seconds)

            # Run tflint with JSON output
            proc = await asyncio.create_subprocess_exec(
                "tflint", "--format=json",
                cwd=workspace,
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
                    for issue in result.get("issues", []):
                        severity = TerraformSeverity.WARNING
                        if issue.get("rule", {}).get("severity") == "error":
                            severity = TerraformSeverity.ERROR

                        issues.append(TerraformIssue(
                            issue_type=TerraformIssueType.LINT,
                            severity=severity,
                            message=issue.get("message", "Unknown issue"),
                            file=issue.get("range", {}).get("filename"),
                            line=issue.get("range", {}).get("start", {}).get("line"),
                            rule=issue.get("rule", {}).get("name"),
                        ))
                except json.JSONDecodeError:
                    pass

        except asyncio.TimeoutError:
            issues.append(TerraformIssue(
                issue_type=TerraformIssueType.LINT,
                severity=TerraformSeverity.ERROR,
                message="tflint timed out",
            ))
        except Exception as e:
            issues.append(TerraformIssue(
                issue_type=TerraformIssueType.LINT,
                severity=TerraformSeverity.WARNING,
                message=f"tflint error: {str(e)}",
            ))

        return issues

    async def _run_terraform_validate(self, workspace: Path) -> list[TerraformIssue]:
        """Run terraform validate on the workspace."""
        issues = []

        if not self._terraform_available:
            return issues

        try:
            # Initialize terraform first
            init_proc = await asyncio.create_subprocess_exec(
                "terraform", "init", "-backend=false",
                cwd=workspace,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            _, stderr = await asyncio.wait_for(
                init_proc.communicate(),
                timeout=self.timeout_seconds
            )

            if init_proc.returncode != 0:
                # Parse init errors
                issues.append(TerraformIssue(
                    issue_type=TerraformIssueType.VALIDATION,
                    severity=TerraformSeverity.ERROR,
                    message=f"terraform init failed: {stderr.decode()[:500]}",
                ))
                return issues

            # Run validate with JSON output
            proc = await asyncio.create_subprocess_exec(
                "terraform", "validate", "-json",
                cwd=workspace,
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
                    if not result.get("valid", False):
                        for diag in result.get("diagnostics", []):
                            severity = TerraformSeverity.ERROR
                            if diag.get("severity") == "warning":
                                severity = TerraformSeverity.WARNING

                            issues.append(TerraformIssue(
                                issue_type=TerraformIssueType.VALIDATION,
                                severity=severity,
                                message=diag.get("summary", "Validation error"),
                                file=diag.get("range", {}).get("filename"),
                                line=diag.get("range", {}).get("start", {}).get("line"),
                            ))
                except json.JSONDecodeError:
                    pass

        except asyncio.TimeoutError:
            issues.append(TerraformIssue(
                issue_type=TerraformIssueType.VALIDATION,
                severity=TerraformSeverity.ERROR,
                message="terraform validate timed out",
            ))
        except Exception as e:
            issues.append(TerraformIssue(
                issue_type=TerraformIssueType.VALIDATION,
                severity=TerraformSeverity.WARNING,
                message=f"terraform validate error: {str(e)}",
            ))

        return issues

    async def _run_tfsec(self, workspace: Path) -> tuple[list[TerraformIssue], list[dict]]:
        """Run tfsec security scanner on the workspace."""
        issues = []
        findings = []

        if not self._tfsec_available:
            return issues, findings

        try:
            proc = await asyncio.create_subprocess_exec(
                "tfsec", str(workspace), "--format=json",
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
                    for finding in result.get("results", []):
                        severity_map = {
                            "CRITICAL": TerraformSeverity.ERROR,
                            "HIGH": TerraformSeverity.ERROR,
                            "MEDIUM": TerraformSeverity.WARNING,
                            "LOW": TerraformSeverity.NOTICE,
                        }
                        severity = severity_map.get(
                            finding.get("severity", "MEDIUM"),
                            TerraformSeverity.WARNING
                        )

                        issues.append(TerraformIssue(
                            issue_type=TerraformIssueType.SECURITY,
                            severity=severity,
                            message=finding.get("description", "Security issue"),
                            file=finding.get("location", {}).get("filename"),
                            line=finding.get("location", {}).get("start_line"),
                            rule=finding.get("rule_id"),
                            fix_suggestion=finding.get("resolution"),
                        ))

                        findings.append({
                            "rule_id": finding.get("rule_id"),
                            "severity": finding.get("severity"),
                            "description": finding.get("description"),
                            "impact": finding.get("impact"),
                            "resolution": finding.get("resolution"),
                        })
                except json.JSONDecodeError:
                    pass

        except asyncio.TimeoutError:
            issues.append(TerraformIssue(
                issue_type=TerraformIssueType.SECURITY,
                severity=TerraformSeverity.WARNING,
                message="tfsec timed out",
            ))
        except Exception as e:
            issues.append(TerraformIssue(
                issue_type=TerraformIssueType.SECURITY,
                severity=TerraformSeverity.WARNING,
                message=f"tfsec error: {str(e)}",
            ))

        return issues, findings

    async def _verify_terraform_code(self, code: str) -> TerraformVerificationResult:
        """Verify a single block of Terraform code."""
        import time
        start_time = time.time()

        workspace = self._create_temp_workspace(code)
        all_issues = []
        security_findings = []

        try:
            # Run all enabled checks in parallel
            tasks = []

            if self.enable_tflint:
                tasks.append(self._run_tflint(workspace))

            if self.enable_validate:
                tasks.append(self._run_terraform_validate(workspace))

            if self.enable_tfsec:
                tasks.append(self._run_tfsec(workspace))

            results = await asyncio.gather(*tasks, return_exceptions=True)

            for result in results:
                if isinstance(result, Exception):
                    all_issues.append(TerraformIssue(
                        issue_type=TerraformIssueType.VALIDATION,
                        severity=TerraformSeverity.WARNING,
                        message=f"Verification error: {str(result)}",
                    ))
                elif isinstance(result, tuple):
                    # tfsec returns (issues, findings)
                    all_issues.extend(result[0])
                    security_findings.extend(result[1])
                elif isinstance(result, list):
                    all_issues.extend(result)

        finally:
            # Cleanup workspace
            shutil.rmtree(workspace, ignore_errors=True)

        execution_time = (time.time() - start_time) * 1000

        # Determine if passed
        if self.strict_mode:
            passed = len(all_issues) == 0
        else:
            passed = all(
                i.severity not in (TerraformSeverity.ERROR,)
                for i in all_issues
            )

        return TerraformVerificationResult(
            passed=passed,
            issues=all_issues,
            security_findings=security_findings,
            execution_time_ms=execution_time,
        )

    async def verify(
        self,
        messages: list[dict],
        metadata: dict[str, Any] | None = None
    ) -> VerificationResult:
        """
        Verify all Terraform code blocks in assistant messages.

        Args:
            messages: Conversation messages
            metadata: Optional metadata

        Returns:
            VerificationResult indicating pass/fail with details
        """
        all_terraform_blocks = []
        all_results = []

        # Extract Terraform code from assistant messages
        for msg in messages:
            if msg.get("role") != "assistant":
                continue

            content = msg.get("content", "")
            if isinstance(content, dict):
                content = content.get("content", "")
            if not isinstance(content, str):
                continue

            blocks = self._extract_terraform_blocks(content)
            all_terraform_blocks.extend(blocks)

        if not all_terraform_blocks:
            return VerificationResult(
                status=VerificationStatus.SKIPPED,
                plugin_name=self.name,
                message="No Terraform code blocks found",
                details={"blocks_checked": 0},
            )

        # Verify each block
        for block in all_terraform_blocks:
            result = await self._verify_terraform_code(block)
            all_results.append(result)

        # Aggregate results
        total_errors = sum(r.error_count for r in all_results)
        total_warnings = sum(r.warning_count for r in all_results)
        all_security_findings = []
        for r in all_results:
            all_security_findings.extend(r.security_findings)

        all_passed = all(r.passed for r in all_results)

        if not all_passed or total_errors > 0:
            return VerificationResult(
                status=VerificationStatus.FAILED,
                plugin_name=self.name,
                message=f"Found {total_errors} error(s) and {total_warnings} warning(s) in Terraform code",
                details={
                    "blocks_checked": len(all_terraform_blocks),
                    "results": [r.to_dict() for r in all_results],
                    "security_findings": all_security_findings,
                    "total_errors": total_errors,
                    "total_warnings": total_warnings,
                },
                score_modifier=0.0,
            )

        # Calculate score modifier based on warnings
        score_modifier = max(0.5, 1.0 - (total_warnings * 0.1))

        return VerificationResult(
            status=VerificationStatus.PASSED,
            plugin_name=self.name,
            message=f"All {len(all_terraform_blocks)} Terraform block(s) validated successfully",
            details={
                "blocks_checked": len(all_terraform_blocks),
                "results": [r.to_dict() for r in all_results],
                "total_warnings": total_warnings,
            },
            score_modifier=score_modifier,
        )

    def should_skip(
        self,
        messages: list[dict],
        metadata: dict[str, Any] | None = None
    ) -> bool:
        """Skip if no Terraform code is present."""
        for msg in messages:
            if msg.get("role") != "assistant":
                continue
            content = msg.get("content", "")
            if isinstance(content, str) and self.TF_BLOCK_PATTERN.search(content):
                return False
        return True
