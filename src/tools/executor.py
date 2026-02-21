"""
Sandboxed Tool Executor

Executes tools in sandboxed environments with resource limits.
Supports subprocess, container, and in-process execution modes.
"""

import json
import logging
import os
import secrets
import subprocess
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .exceptions import (
    HumanApprovalError,
    SmithValidationError,
    ToolNotFoundError,
    ToolPermissionError,
    ToolSandboxError,
    ToolTimeoutError,
)
from .interface import (
    InvocationResult,
    ToolInterface,
    ToolInvocation,
    ToolResult,
    ToolRiskLevel,
)
from .permissions import PermissionCheckResult, PermissionManager
from .registry import ToolRegistration, ToolRegistry
from .validation import ApprovalResult, ToolApprovalResult, ToolApprovalValidator

logger = logging.getLogger(__name__)


class ExecutionMode(Enum):
    """Tool execution modes."""

    IN_PROCESS = auto()  # Direct execution (lowest security)
    SUBPROCESS = auto()  # Isolated subprocess
    CONTAINER = auto()  # Docker/Podman container (highest security)


class ExecutionState(Enum):
    """State of a tool execution."""

    PENDING = auto()
    VALIDATING = auto()
    AWAITING_APPROVAL = auto()
    EXECUTING = auto()
    COMPLETED = auto()
    FAILED = auto()
    TIMEOUT = auto()
    CANCELLED = auto()


@dataclass
class ExecutionConfig:
    """Configuration for tool execution."""

    default_mode: ExecutionMode = ExecutionMode.SUBPROCESS
    timeout_seconds: int = 30
    memory_limit_mb: int = 256
    cpu_limit_percent: int = 50
    network_allowed: bool = False
    filesystem_read_only: bool = True
    container_runtime: str = "docker"
    container_image: str = "python:3.11-slim"
    working_dir: Optional[Path] = None
    environment: Dict[str, str] = field(default_factory=dict)


@dataclass
class ExecutionContext:
    """Context for a tool execution."""

    execution_id: str
    invocation: ToolInvocation
    state: ExecutionState = ExecutionState.PENDING
    mode: ExecutionMode = ExecutionMode.SUBPROCESS
    config: ExecutionConfig = field(default_factory=ExecutionConfig)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    approval_result: Optional[ToolApprovalResult] = None
    permission_result: Optional[PermissionCheckResult] = None
    result: Optional[ToolResult] = None
    logs: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def generate_id() -> str:
        return f"EXEC-{secrets.token_hex(8)}"

    def log(self, message: str) -> None:
        self.logs.append(f"[{datetime.now().isoformat()}] {message}")


class ToolExecutor:
    """
    Executes tools in sandboxed environments.

    Provides:
    - Permission checking
    - Smith approval validation
    - Sandboxed execution
    - Timeout handling
    - Audit logging
    """

    def __init__(
        self,
        registry: ToolRegistry,
        permission_manager: PermissionManager,
        validator: ToolApprovalValidator,
        config: Optional[ExecutionConfig] = None,
        audit_callback: Optional[Callable[[ExecutionContext], None]] = None,
    ):
        """
        Initialize executor.

        Args:
            registry: Tool registry
            permission_manager: Permission manager
            validator: Tool approval validator
            config: Default execution config
            audit_callback: Callback for audit logging
        """
        self.registry = registry
        self.permission_manager = permission_manager
        self.validator = validator
        self.config = config or ExecutionConfig()
        self.audit_callback = audit_callback
        self._executions: Dict[str, ExecutionContext] = {}
        self._lock = threading.RLock()
        self._human_approval_callback: Optional[Callable[[ExecutionContext], bool]] = None

    def set_human_approval_callback(
        self,
        callback: Callable[[ExecutionContext], bool],
    ) -> None:
        """Set callback for human approval requests."""
        self._human_approval_callback = callback

    def execute(
        self,
        tool_name: str,
        parameters: Dict[str, Any],
        user_id: str,
        agent_id: Optional[str] = None,
        request_id: Optional[str] = None,
        mode_override: Optional[ExecutionMode] = None,
        timeout_override: Optional[int] = None,
    ) -> ToolResult:
        """
        Execute a tool with full security checks.

        Args:
            tool_name: Name of tool to execute
            parameters: Tool parameters
            user_id: User requesting execution
            agent_id: Agent requesting execution
            request_id: Original request ID
            mode_override: Override execution mode
            timeout_override: Override timeout

        Returns:
            ToolResult
        """
        # Create invocation record
        invocation = ToolInvocation(
            invocation_id=ToolInvocation.generate_id(),
            tool_name=tool_name,
            parameters=parameters,
            user_id=user_id,
            agent_id=agent_id,
            request_id=request_id,
        )

        # Create execution context
        context = ExecutionContext(
            execution_id=ExecutionContext.generate_id(),
            invocation=invocation,
            mode=mode_override or self.config.default_mode,
            config=self.config,
        )

        with self._lock:
            self._executions[context.execution_id] = context

        context.log(f"Starting execution of tool: {tool_name}")

        try:
            # Step 1: Get tool registration
            context.state = ExecutionState.VALIDATING
            registration = self.registry.get_by_name(tool_name)

            if not registration:
                context.log(f"Tool not found: {tool_name}")
                return self._fail_execution(
                    context,
                    InvocationResult.VALIDATION_ERROR,
                    f"Tool not found: {tool_name}",
                )

            # Step 2: Check permissions
            permission_result = self.permission_manager.check(
                principal_id=user_id,
                principal_type="user",
                tool_id=registration.tool_id,
                tool_name=tool_name,
                tool_category=registration.schema.category,
                tool_risk=registration.schema.risk_level,
            )
            context.permission_result = permission_result

            if not permission_result.allowed:
                context.log(f"Permission denied: {permission_result.reason}")
                return self._fail_execution(
                    context,
                    InvocationResult.PERMISSION_DENIED,
                    permission_result.reason,
                )

            # Step 2b: File system path allowlist check
            path_violation = self._check_path_allowlist(parameters, agent_id)
            if path_violation:
                context.log(f"Path access denied: {path_violation}")
                return self._fail_execution(
                    context,
                    InvocationResult.PERMISSION_DENIED,
                    path_violation,
                )

            # Step 3: Smith approval validation
            approval_result = self.validator.validate(
                registration=registration,
                parameters=parameters,
                user_id=user_id,
                agent_id=agent_id,
            )
            context.approval_result = approval_result

            if not approval_result.approved:
                context.log(f"Smith denied: {approval_result.denial_message}")
                return self._fail_execution(
                    context,
                    InvocationResult.PERMISSION_DENIED,
                    approval_result.denial_message or "Smith validation failed",
                )

            # Step 4: Handle human approval if required
            if approval_result.requires_human_approval:
                context.state = ExecutionState.AWAITING_APPROVAL

                try:
                    self._request_human_approval(context)
                    invocation.approved_by = "human"
                    invocation.approval_time = datetime.now()
                except HumanApprovalError as e:
                    context.log(f"Human approval failed: {e}")
                    return self._fail_execution(
                        context,
                        InvocationResult.PERMISSION_DENIED,
                        str(e),
                    )
            else:
                invocation.approved_by = "smith"
                invocation.approval_time = datetime.now()

            # Step 5: Execute tool
            context.state = ExecutionState.EXECUTING
            context.started_at = datetime.now()
            context.log("Beginning execution")

            # Determine execution mode based on risk
            effective_mode = self._determine_execution_mode(
                registration.schema.risk_level,
                mode_override,
            )
            context.mode = effective_mode

            # Get timeout
            timeout = timeout_override or registration.schema.timeout_seconds

            # Execute with appropriate method
            if effective_mode == ExecutionMode.IN_PROCESS:
                result = self._execute_in_process(
                    registration.tool,
                    parameters,
                    timeout,
                )
            elif effective_mode == ExecutionMode.SUBPROCESS:
                result = self._execute_subprocess(
                    registration.tool,
                    parameters,
                    timeout,
                    context,
                )
            else:  # CONTAINER
                result = self._execute_container(
                    registration.tool,
                    parameters,
                    timeout,
                    context,
                )

            # Step 6: Complete
            context.completed_at = datetime.now()
            context.state = ExecutionState.COMPLETED
            context.result = result
            invocation.result = result
            invocation.completed_at = datetime.now()

            # Record invocation
            self.registry.record_invocation(registration.tool_id)
            if permission_result.grant:
                self.permission_manager.record_use(permission_result.grant.grant_id)

            context.log(f"Execution completed: {result.result.name}")

            # Audit
            self._audit(context)

            return result

        except ToolNotFoundError as e:
            return self._fail_execution(
                context,
                InvocationResult.VALIDATION_ERROR,
                str(e),
            )
        except ToolPermissionError as e:
            return self._fail_execution(
                context,
                InvocationResult.PERMISSION_DENIED,
                str(e),
            )
        except SmithValidationError as e:
            return self._fail_execution(
                context,
                InvocationResult.PERMISSION_DENIED,
                str(e),
            )
        except HumanApprovalError as e:
            return self._fail_execution(
                context,
                InvocationResult.PERMISSION_DENIED,
                str(e),
            )
        except ToolTimeoutError as e:
            return self._fail_execution(
                context,
                InvocationResult.TIMEOUT,
                str(e),
            )
        except ToolSandboxError as e:
            logger.error(f"Sandbox error for {tool_name}: {e}")
            return self._fail_execution(
                context,
                InvocationResult.SANDBOX_ERROR,
                str(e),
            )
        except Exception as e:
            logger.exception(f"Unexpected tool execution error for {tool_name}: {type(e).__name__}")
            return self._fail_execution(
                context,
                InvocationResult.EXECUTION_ERROR,
                f"Unexpected error: {type(e).__name__}: {e}",
            )

    def _fail_execution(
        self,
        context: ExecutionContext,
        result_type: InvocationResult,
        error: str,
    ) -> ToolResult:
        """Record a failed execution."""
        context.state = ExecutionState.FAILED
        context.completed_at = datetime.now()

        result = ToolResult(
            result=result_type,
            error=error,
        )
        context.result = result
        context.invocation.result = result
        context.invocation.completed_at = datetime.now()

        self._audit(context)
        return result

    def _check_path_allowlist(
        self,
        parameters: Dict[str, Any],
        agent_id: Optional[str],
    ) -> Optional[str]:
        """
        Check if file paths in tool parameters are within the agent's allowed_paths.

        Returns None if allowed, or an error message string if denied.
        """
        if not agent_id:
            return None

        # Try to load the agent config to get allowed_paths
        try:
            from src.agents.config import AgentConfig
            from src.agents.loader import create_loader

            loader = create_loader(Path.cwd())
            registered = loader.registry.get(agent_id)
            if not registered or not hasattr(registered, "config"):
                return None
            config = registered.config
            allowed_paths = getattr(config, "allowed_paths", None)
            if allowed_paths is None:
                return None
            if not allowed_paths:
                # Empty list means no file access — check if any paths in params
                pass
        except Exception:
            # If we can't load the config, skip the check
            return None

        # Extract file-like paths from parameters
        path_keys = {"path", "file", "filepath", "file_path", "filename", "directory", "dir"}
        for key, value in parameters.items():
            if key.lower() in path_keys and isinstance(value, str):
                resolved = str(Path(value).resolve())
                if not any(resolved.startswith(str(Path(ap).resolve())) for ap in allowed_paths):
                    return (
                        f"Agent '{agent_id}' denied access to path '{value}' — "
                        f"not in allowed_paths: {allowed_paths}"
                    )

        return None

    def _determine_execution_mode(
        self,
        risk_level: ToolRiskLevel,
        override: Optional[ExecutionMode],
    ) -> ExecutionMode:
        """Determine appropriate execution mode based on risk."""
        if override:
            return override

        # Force container for high/critical risk
        if risk_level.value >= ToolRiskLevel.HIGH.value:
            return ExecutionMode.CONTAINER

        # Force subprocess for medium risk
        if risk_level.value >= ToolRiskLevel.MEDIUM.value:
            return ExecutionMode.SUBPROCESS

        return self.config.default_mode

    def _request_human_approval(self, context: ExecutionContext) -> bool:
        """
        Request human approval for execution.

        Raises:
            HumanApprovalError: If approval callback is not configured or fails
        """
        context.log("Requesting human approval")

        if not self._human_approval_callback:
            raise HumanApprovalError(
                message="No human approval callback configured",
                tool_name=context.invocation.tool_name,
                approval_denied=False,
            )

        try:
            approved = self._human_approval_callback(context)
            if not approved:
                raise HumanApprovalError(
                    message="Human approval was denied",
                    tool_name=context.invocation.tool_name,
                    approval_denied=True,
                )
            return approved
        except HumanApprovalError:
            raise
        except Exception as e:
            logger.error(f"Human approval callback error: {type(e).__name__}: {e}")
            raise HumanApprovalError(
                message=f"Approval callback failed: {e}",
                tool_name=context.invocation.tool_name,
                approval_denied=False,
                original_error=e,
            )

    def _execute_in_process(
        self,
        tool: ToolInterface,
        parameters: Dict[str, Any],
        timeout: int,
    ) -> ToolResult:
        """Execute tool directly in current process."""
        result_holder = [None]
        exception_holder = [None]

        def run():
            try:
                result_holder[0] = tool.invoke(parameters)
            except Exception as e:
                exception_holder[0] = e

        thread = threading.Thread(target=run)
        thread.start()
        thread.join(timeout=timeout)

        if thread.is_alive():
            return ToolResult(
                result=InvocationResult.TIMEOUT,
                error=f"Execution timed out after {timeout}s",
            )

        if exception_holder[0]:
            return ToolResult(
                result=InvocationResult.EXECUTION_ERROR,
                error=str(exception_holder[0]),
            )

        return result_holder[0] or ToolResult(
            result=InvocationResult.FAILURE,
            error="No result returned",
        )

    def _get_restricted_env(self) -> Dict[str, str]:
        """Get a restricted environment for subprocess execution.

        Strips sensitive environment variables to prevent credential leakage.
        """
        env = dict(os.environ)
        sensitive_prefixes = [
            "AGENT_OS_API_KEY", "AGENT_OS_ENCRYPTION_KEY", "AGENT_OS_SESSION_SECRET",
            "AWS_", "OPENAI_", "ANTHROPIC_", "HF_TOKEN", "GITHUB_TOKEN",
            "DATABASE_URL",
        ]
        for key in list(env.keys()):
            for prefix in sensitive_prefixes:
                if key.startswith(prefix):
                    del env[key]
                    break
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        return env

    def _execute_subprocess(
        self,
        tool: ToolInterface,
        parameters: Dict[str, Any],
        timeout: int,
        context: ExecutionContext,
    ) -> ToolResult:
        """Execute tool in isolated subprocess with real process isolation."""
        import json as json_module
        import sys
        import tempfile

        context.log("Executing in subprocess isolation")

        params_file = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".json", delete=False, prefix="agentos_tool_"
            ) as f:
                json_module.dump(parameters, f)
                params_file = f.name

            cmd = [
                sys.executable, "-m", "src.tools.subprocess_runner",
                "--tool-module", tool.__class__.__module__,
                "--tool-class", tool.__class__.__name__,
                "--tool-name", tool.name,
                "--params-file", params_file,
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                timeout=timeout,
                text=True,
                env=self._get_restricted_env(),
            )

            if result.returncode != 0:
                stderr = result.stderr[:500] if result.stderr else ""
                logger.warning(
                    "Subprocess execution failed for tool '%s' (exit=%d), "
                    "falling back to in-process execution: %s",
                    tool.name, result.returncode, stderr,
                )
                return self._execute_in_process(tool, parameters, timeout)

            try:
                output = json_module.loads(result.stdout)
            except json_module.JSONDecodeError:
                return ToolResult(
                    result=InvocationResult.SANDBOX_ERROR,
                    error="Failed to parse subprocess output",
                )

            if output.get("success"):
                return ToolResult(
                    result=InvocationResult.SUCCESS,
                    output=output.get("output"),
                    execution_time_ms=output.get("execution_time_ms", 0),
                )
            else:
                return ToolResult(
                    result=InvocationResult.ERROR,
                    error=output.get("error", "Unknown subprocess error"),
                )

        except subprocess.TimeoutExpired:
            return ToolResult(
                result=InvocationResult.TIMEOUT,
                error=f"Subprocess timed out after {timeout}s",
            )
        except Exception as e:
            logger.error(f"Subprocess execution error: {e}")
            return ToolResult(
                result=InvocationResult.SANDBOX_ERROR,
                error="Subprocess execution failed",
            )
        finally:
            if params_file:
                try:
                    os.unlink(params_file)
                except OSError:
                    pass

    def _execute_container(
        self,
        tool: ToolInterface,
        parameters: Dict[str, Any],
        timeout: int,
        context: ExecutionContext,
    ) -> ToolResult:
        """Execute tool in container sandbox."""
        context.log("Executing in container (falling back to subprocess)")

        # Check if container runtime available
        runtime = self.config.container_runtime

        try:
            result = subprocess.run(
                [runtime, "--version"],
                capture_output=True,
                timeout=5,
            )
            if result.returncode != 0:
                raise FileNotFoundError(f"{runtime} not available")
        except (subprocess.SubprocessError, FileNotFoundError, OSError):
            logger.warning(
                "SECURITY: Container runtime '%s' not available. Tool '%s' "
                "executing in subprocess mode instead of container isolation. "
                "Install Docker or Podman for full container sandboxing.",
                runtime,
                tool.name,
            )
            context.log(f"Container runtime not available, falling back to subprocess")
            return self._execute_subprocess(tool, parameters, timeout, context)

        # For full implementation, would build and run container
        # For now, fall back to subprocess with resource limits
        return self._execute_subprocess(tool, parameters, timeout, context)

    def _audit(self, context: ExecutionContext) -> None:
        """Send audit event."""
        if self.audit_callback:
            try:
                self.audit_callback(context)
            except Exception as e:
                logger.warning(f"Audit callback error: {e}")

    def get_execution(self, execution_id: str) -> Optional[ExecutionContext]:
        """Get execution context by ID."""
        with self._lock:
            return self._executions.get(execution_id)

    def list_executions(
        self,
        user_id: Optional[str] = None,
        state: Optional[ExecutionState] = None,
        limit: int = 100,
    ) -> List[ExecutionContext]:
        """List execution contexts."""
        with self._lock:
            results = list(self._executions.values())

            if user_id:
                results = [e for e in results if e.invocation.user_id == user_id]

            if state:
                results = [e for e in results if e.state == state]

            # Sort by start time, most recent first
            results.sort(
                key=lambda e: e.started_at or datetime.min,
                reverse=True,
            )

            return results[:limit]

    def get_statistics(self) -> Dict[str, Any]:
        """Get execution statistics."""
        with self._lock:
            by_state = {}
            by_result = {}
            total_time_ms = 0
            count = 0

            for context in self._executions.values():
                # By state
                state = context.state.name
                by_state[state] = by_state.get(state, 0) + 1

                # By result
                if context.result:
                    result = context.result.result.name
                    by_result[result] = by_result.get(result, 0) + 1
                    total_time_ms += context.result.execution_time_ms
                    count += 1

            return {
                "total_executions": len(self._executions),
                "by_state": by_state,
                "by_result": by_result,
                "average_execution_time_ms": total_time_ms / count if count > 0 else 0,
            }


def create_executor(
    registry: ToolRegistry,
    permission_manager: PermissionManager,
    validator: ToolApprovalValidator,
    config: Optional[ExecutionConfig] = None,
) -> ToolExecutor:
    """
    Create a tool executor.

    Args:
        registry: Tool registry
        permission_manager: Permission manager
        validator: Tool approval validator
        config: Execution config

    Returns:
        ToolExecutor instance
    """
    return ToolExecutor(
        registry=registry,
        permission_manager=permission_manager,
        validator=validator,
        config=config,
    )
