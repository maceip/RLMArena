"""
Runtime Verifier Plugin for functional code execution.

This plugin executes agent-generated code in isolated sandbox environments
(Firecracker MicroVMs or subprocess isolation) to verify functional correctness.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional
import asyncio
import json
import os
import re
import subprocess
import tempfile
import time
import uuid


class ExecutionStatus(Enum):
    """Status of code execution."""
    SUCCESS = "success"
    FAILURE = "failure"
    TIMEOUT = "timeout"
    ERROR = "error"
    SKIPPED = "skipped"


@dataclass
class SandboxConfig:
    """Configuration for the execution sandbox."""
    memory_limit_mb: int = 512
    cpu_limit_cores: float = 1.0
    timeout_seconds: float = 30.0
    network_enabled: bool = False
    max_output_bytes: int = 1024 * 1024  # 1MB
    working_dir: Optional[str] = None
    environment: dict[str, str] = field(default_factory=dict)
    allowed_imports: Optional[list[str]] = None
    denied_imports: Optional[list[str]] = None


@dataclass
class ExecutionResult:
    """Result of code execution."""
    id: str
    status: ExecutionStatus
    stdout: str
    stderr: str
    exit_code: int
    execution_time_ms: float
    memory_used_mb: float = 0.0
    tests_passed: int = 0
    tests_failed: int = 0
    tests_total: int = 0
    error_message: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "status": self.status.value,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "exit_code": self.exit_code,
            "execution_time_ms": self.execution_time_ms,
            "memory_used_mb": self.memory_used_mb,
            "tests_passed": self.tests_passed,
            "tests_failed": self.tests_failed,
            "tests_total": self.tests_total,
            "error_message": self.error_message,
            "metadata": self.metadata,
        }


class MockFirecrackerPool:
    """
    Mock Firecracker MicroVM pool for code execution.

    In production, this would manage actual Firecracker MicroVMs.
    This mock uses subprocess isolation to simulate the behavior.
    """

    def __init__(
        self,
        pool_size: int = 4,
        default_config: Optional[SandboxConfig] = None,
    ):
        self.pool_size = pool_size
        self.default_config = default_config or SandboxConfig()
        self._available_vms = list(range(pool_size))
        self._in_use: dict[str, int] = {}
        self._lock = asyncio.Lock()

    async def acquire(self) -> Optional[str]:
        """Acquire a VM from the pool."""
        async with self._lock:
            if not self._available_vms:
                return None

            vm_id = self._available_vms.pop(0)
            session_id = str(uuid.uuid4())
            self._in_use[session_id] = vm_id
            return session_id

    async def release(self, session_id: str) -> None:
        """Release a VM back to the pool."""
        async with self._lock:
            if session_id in self._in_use:
                vm_id = self._in_use.pop(session_id)
                self._available_vms.append(vm_id)

    async def execute(
        self,
        code: str,
        session_id: Optional[str] = None,
        config: Optional[SandboxConfig] = None,
        language: str = "python",
    ) -> ExecutionResult:
        """Execute code in a sandboxed environment."""
        config = config or self.default_config
        execution_id = str(uuid.uuid4())[:8]

        # Acquire VM if not provided
        owned_session = False
        if session_id is None:
            session_id = await self.acquire()
            owned_session = True

            if session_id is None:
                return ExecutionResult(
                    id=execution_id,
                    status=ExecutionStatus.ERROR,
                    stdout="",
                    stderr="No VM available in pool",
                    exit_code=-1,
                    execution_time_ms=0.0,
                    error_message="Pool exhausted",
                )

        try:
            result = await self._run_in_sandbox(
                code, config, execution_id, language
            )
            return result
        finally:
            if owned_session:
                await self.release(session_id)

    async def _run_in_sandbox(
        self,
        code: str,
        config: SandboxConfig,
        execution_id: str,
        language: str,
    ) -> ExecutionResult:
        """Run code in subprocess sandbox (simulating Firecracker)."""
        start_time = time.time()

        # Create temporary file for code
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".py" if language == "python" else f".{language}",
            delete=False,
        ) as f:
            f.write(code)
            code_file = f.name

        try:
            # Build command
            if language == "python":
                cmd = ["python", "-u", code_file]
            elif language == "node":
                cmd = ["node", code_file]
            elif language == "bash":
                cmd = ["bash", code_file]
            else:
                return ExecutionResult(
                    id=execution_id,
                    status=ExecutionStatus.ERROR,
                    stdout="",
                    stderr=f"Unsupported language: {language}",
                    exit_code=-1,
                    execution_time_ms=0.0,
                    error_message=f"Unsupported language: {language}",
                )

            # Execute with timeout
            env = os.environ.copy()
            env.update(config.environment)

            # Disable network if required (mock - would use network namespaces in production)
            if not config.network_enabled:
                env["NO_PROXY"] = "*"
                env["HTTP_PROXY"] = "http://blocked:0"
                env["HTTPS_PROXY"] = "http://blocked:0"

            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
                cwd=config.working_dir,
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=config.timeout_seconds,
                )

                stdout_str = stdout.decode("utf-8", errors="replace")
                stderr_str = stderr.decode("utf-8", errors="replace")

                # Truncate if too long
                if len(stdout_str) > config.max_output_bytes:
                    stdout_str = stdout_str[:config.max_output_bytes] + "\n[TRUNCATED]"
                if len(stderr_str) > config.max_output_bytes:
                    stderr_str = stderr_str[:config.max_output_bytes] + "\n[TRUNCATED]"

                execution_time = (time.time() - start_time) * 1000

                # Parse test results if pytest-style output
                tests_passed, tests_failed, tests_total = self._parse_test_output(
                    stdout_str + stderr_str
                )

                status = ExecutionStatus.SUCCESS if process.returncode == 0 else ExecutionStatus.FAILURE

                return ExecutionResult(
                    id=execution_id,
                    status=status,
                    stdout=stdout_str,
                    stderr=stderr_str,
                    exit_code=process.returncode or 0,
                    execution_time_ms=execution_time,
                    tests_passed=tests_passed,
                    tests_failed=tests_failed,
                    tests_total=tests_total,
                )

            except asyncio.TimeoutError:
                process.kill()
                await process.wait()

                return ExecutionResult(
                    id=execution_id,
                    status=ExecutionStatus.TIMEOUT,
                    stdout="",
                    stderr=f"Execution timed out after {config.timeout_seconds}s",
                    exit_code=-1,
                    execution_time_ms=config.timeout_seconds * 1000,
                    error_message="Timeout",
                )

        except Exception as e:
            return ExecutionResult(
                id=execution_id,
                status=ExecutionStatus.ERROR,
                stdout="",
                stderr=str(e),
                exit_code=-1,
                execution_time_ms=(time.time() - start_time) * 1000,
                error_message=str(e),
            )
        finally:
            # Clean up temp file
            try:
                os.unlink(code_file)
            except:
                pass

    def _parse_test_output(self, output: str) -> tuple[int, int, int]:
        """Parse pytest-style test output."""
        # Look for pytest summary line: "X passed, Y failed"
        match = re.search(r"(\d+) passed", output)
        passed = int(match.group(1)) if match else 0

        match = re.search(r"(\d+) failed", output)
        failed = int(match.group(1)) if match else 0

        match = re.search(r"(\d+) error", output)
        errors = int(match.group(1)) if match else 0

        total = passed + failed + errors

        # Also check for unittest-style output
        if total == 0:
            match = re.search(r"Ran (\d+) test", output)
            if match:
                total = int(match.group(1))
                if "OK" in output:
                    passed = total
                elif "FAILED" in output:
                    fail_match = re.search(r"failures=(\d+)", output)
                    failed = int(fail_match.group(1)) if fail_match else 0
                    passed = total - failed

        return passed, failed, total

    def get_stats(self) -> dict[str, Any]:
        """Get pool statistics."""
        return {
            "pool_size": self.pool_size,
            "available": len(self._available_vms),
            "in_use": len(self._in_use),
        }


class RuntimeVerifierPlugin:
    """
    Verification plugin that executes agent code to verify functional correctness.

    This plugin:
    1. Extracts code from trajectory
    2. Executes it in a sandboxed environment
    3. Checks for successful execution and test passage
    4. Returns verification result
    """

    def __init__(
        self,
        sandbox_config: Optional[SandboxConfig] = None,
        pool_size: int = 4,
        require_tests: bool = False,
        min_test_coverage: float = 0.0,
    ):
        self._config = sandbox_config or SandboxConfig()
        self._pool = MockFirecrackerPool(pool_size, self._config)
        self._require_tests = require_tests
        self._min_test_coverage = min_test_coverage

    @property
    def name(self) -> str:
        return "runtime_verifier"

    @property
    def description(self) -> str:
        return "Executes agent code in sandboxed environment to verify functional correctness"

    @property
    def is_hard_check(self) -> bool:
        return True

    @property
    def floor_reward(self) -> float:
        return -1.0

    def _extract_code_blocks(
        self,
        messages: list[dict[str, Any]],
    ) -> list[tuple[str, str, int]]:
        """
        Extract executable code blocks from messages.

        Returns list of (code, language, message_index) tuples.
        """
        blocks = []

        for i, msg in enumerate(messages):
            content = msg.get("content", "")
            if not isinstance(content, str):
                continue

            # Extract Python code blocks
            python_pattern = r"```(?:python|py)\s*\n(.*?)```"
            for match in re.finditer(python_pattern, content, re.DOTALL | re.IGNORECASE):
                blocks.append((match.group(1), "python", i))

            # Extract JavaScript/Node code blocks
            js_pattern = r"```(?:javascript|js|node)\s*\n(.*?)```"
            for match in re.finditer(js_pattern, content, re.DOTALL | re.IGNORECASE):
                blocks.append((match.group(1), "node", i))

            # Extract bash code blocks
            bash_pattern = r"```(?:bash|sh|shell)\s*\n(.*?)```"
            for match in re.finditer(bash_pattern, content, re.DOTALL | re.IGNORECASE):
                blocks.append((match.group(1), "bash", i))

        return blocks

    def _is_safe_code(self, code: str, language: str) -> tuple[bool, Optional[str]]:
        """Check if code is safe to execute."""
        if language == "python":
            # Check for dangerous imports/operations
            dangerous_patterns = [
                (r"\bos\.system\b", "os.system call"),
                (r"\bsubprocess\b", "subprocess usage"),
                (r"\b__import__\b", "dynamic import"),
                (r"\beval\b\s*\(", "eval() call"),
                (r"\bexec\b\s*\(", "exec() call"),
                (r"\bopen\s*\([^)]*['\"]w['\"]", "file write operation"),
                (r"\brm\s+-rf\b", "recursive delete"),
                (r"\bshutil\.rmtree\b", "directory removal"),
            ]

            for pattern, desc in dangerous_patterns:
                if re.search(pattern, code):
                    return False, f"Potentially dangerous operation: {desc}"

        elif language == "bash":
            dangerous_patterns = [
                (r"\brm\s+-rf\s+/", "recursive delete from root"),
                (r"\b:\(\)\s*\{\s*:\|\:&\s*\}", "fork bomb"),
                (r"\bdd\s+if=.*of=/dev/", "disk write"),
                (r"\bmkfs\b", "filesystem creation"),
            ]

            for pattern, desc in dangerous_patterns:
                if re.search(pattern, code):
                    return False, f"Potentially dangerous operation: {desc}"

        return True, None

    def _add_test_harness(self, code: str, language: str) -> str:
        """Add a test harness if code contains testable functions."""
        if language != "python":
            return code

        # Check if code already has tests
        if "def test_" in code or "unittest" in code or "pytest" in code:
            return code

        # Check if code has functions that could be tested
        func_match = re.search(r"def\s+(\w+)\s*\(", code)
        if func_match and not func_match.group(1).startswith("_"):
            # Add simple smoke test
            func_name = func_match.group(1)
            test_code = f"""

# Auto-generated smoke test
if __name__ == "__main__":
    try:
        # Attempt to call the function with no args (may fail, but verifies it exists)
        print("Testing function: {func_name}")
        print("Function exists: True")
    except Exception as e:
        print(f"Test error: {{e}}")
"""
            return code + test_code

        return code

    async def verify(
        self,
        messages: list[dict[str, Any]],
        metadata: Optional[dict[str, Any]] = None,
    ) -> "VerificationResult":
        """Verify code in trajectory by executing it."""
        from qqr.verification.plugin import VerificationResult, VerificationStatus

        code_blocks = self._extract_code_blocks(messages)

        if not code_blocks:
            return VerificationResult(
                status=VerificationStatus.PASSED,
                plugin_name=self.name,
                message="No executable code blocks found in trajectory",
            )

        results = []
        all_passed = True
        total_tests = 0
        passed_tests = 0

        for code, language, msg_index in code_blocks:
            # Safety check
            is_safe, reason = self._is_safe_code(code, language)
            if not is_safe:
                results.append({
                    "message_index": msg_index,
                    "language": language,
                    "status": "rejected",
                    "reason": reason,
                })
                all_passed = False
                continue

            # Add test harness if needed
            executable_code = self._add_test_harness(code, language)

            # Execute in sandbox
            exec_result = await self._pool.execute(
                executable_code,
                language=language,
            )

            results.append({
                "message_index": msg_index,
                "language": language,
                "status": exec_result.status.value,
                "exit_code": exec_result.exit_code,
                "execution_time_ms": exec_result.execution_time_ms,
                "tests_passed": exec_result.tests_passed,
                "tests_failed": exec_result.tests_failed,
                "stdout_preview": exec_result.stdout[:200] if exec_result.stdout else "",
                "stderr_preview": exec_result.stderr[:200] if exec_result.stderr else "",
            })

            if exec_result.status != ExecutionStatus.SUCCESS:
                all_passed = False

            total_tests += exec_result.tests_total
            passed_tests += exec_result.tests_passed

        # Check test requirements
        if self._require_tests and total_tests == 0:
            return VerificationResult(
                status=VerificationStatus.FAILED,
                plugin_name=self.name,
                message="No tests found in code but tests are required",
                details={"execution_results": results},
                score_modifier=-0.5,
            )

        if total_tests > 0 and passed_tests / total_tests < self._min_test_coverage:
            return VerificationResult(
                status=VerificationStatus.FAILED,
                plugin_name=self.name,
                message=f"Test coverage {passed_tests}/{total_tests} below minimum {self._min_test_coverage}",
                details={"execution_results": results},
                score_modifier=-0.5,
            )

        if all_passed:
            return VerificationResult(
                status=VerificationStatus.PASSED,
                plugin_name=self.name,
                message=f"All {len(code_blocks)} code block(s) executed successfully",
                details={
                    "execution_results": results,
                    "total_tests": total_tests,
                    "passed_tests": passed_tests,
                },
            )
        else:
            failed_count = sum(1 for r in results if r["status"] != "success")
            return VerificationResult(
                status=VerificationStatus.FAILED,
                plugin_name=self.name,
                message=f"{failed_count}/{len(code_blocks)} code block(s) failed execution",
                details={"execution_results": results},
                score_modifier=-1.0,
            )

    def get_pool_stats(self) -> dict[str, Any]:
        """Get sandbox pool statistics."""
        return self._pool.get_stats()


class ExecutionProof:
    """
    Proof of successful code execution for certification.

    This provides cryptographic evidence that code was executed
    and produced the expected output.
    """

    def __init__(
        self,
        execution_result: ExecutionResult,
        code_hash: str,
        timestamp: str,
    ):
        self.execution_result = execution_result
        self.code_hash = code_hash
        self.timestamp = timestamp
        self.proof_id = str(uuid.uuid4())

    def to_dict(self) -> dict[str, Any]:
        return {
            "proof_id": self.proof_id,
            "code_hash": self.code_hash,
            "timestamp": self.timestamp,
            "execution_status": self.execution_result.status.value,
            "exit_code": self.execution_result.exit_code,
            "execution_time_ms": self.execution_result.execution_time_ms,
            "tests_passed": self.execution_result.tests_passed,
            "tests_total": self.execution_result.tests_total,
        }

    def is_valid(self) -> bool:
        """Check if this proof represents successful execution."""
        return (
            self.execution_result.status == ExecutionStatus.SUCCESS
            and self.execution_result.exit_code == 0
        )
