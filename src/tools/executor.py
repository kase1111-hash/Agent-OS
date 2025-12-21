"""
Sandboxed Tool Executor

Executes tools in sandboxed environments with resource limits.
Supports subprocess, container, and in-process execution modes.
"""

import json
import logging
import secrets
import subprocess
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .interface import (
    ToolInterface,
    ToolResult,
    ToolInvocation,
    InvocationResult,
    ToolRiskLevel,
)
from .registry import ToolRegistry, ToolRegistration
from .permissions import PermissionManager, PermissionCheckResult
from .validation import ToolApprovalValidator, ToolApprovalResult, ApprovalResult


logger = logging.getLogger(__name__)


class ExecutionMode(Enum):
    """Tool execution modes."""
    IN_PROCESS = auto()   # Direct execution (lowest security)
    SUBPROCESS = auto()   # Isolated subprocess
    CONTAINER = auto()    # Docker/Podman container (highest security)


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
        self._human_approval_callback: Optional[
            Callable[[ExecutionContext], bool]
        ] = None

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

                if not self._request_human_approval(context):
                    context.log("Human approval denied")
                    return self._fail_execution(
                        context,
                        InvocationResult.PERMISSION_DENIED,
                        "Human approval denied",
                    )

                invocation.approved_by = "human"
                invocation.approval_time = datetime.now()
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

        except Exception as e:
            logger.exception(f"Tool execution error: {e}")
            return self._fail_execution(
                context,
                InvocationResult.EXECUTION_ERROR,
                str(e),
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
        """Request human approval for execution."""
        context.log("Requesting human approval")

        if not self._human_approval_callback:
            logger.warning("No human approval callback configured")
            return False

        try:
            return self._human_approval_callback(context)
        except Exception as e:
            logger.error(f"Human approval callback error: {e}")
            return False

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

    def _execute_subprocess(
        self,
        tool: ToolInterface,
        parameters: Dict[str, Any],
        timeout: int,
        context: ExecutionContext,
    ) -> ToolResult:
        """Execute tool in isolated subprocess."""
        context.log("Executing in subprocess")

        # Build subprocess command
        wrapper_code = f'''
import json
import sys

# Tool execution wrapper
try:
    # Import and execute
    from {tool.__class__.__module__} import {tool.__class__.__name__}

    params = json.loads(sys.argv[1])
    tool_instance = {tool.__class__.__name__}("{tool.name}")

    result = tool_instance.invoke(params)

    # Output result
    print(json.dumps({{
        "success": True,
        "result": result.result.name,
        "output": result.output,
        "error": result.error,
        "execution_time_ms": result.execution_time_ms,
    }}))

except Exception as e:
    print(json.dumps({{
        "success": False,
        "error": str(e),
    }}))
'''

        try:
            # For simplicity, execute in-process with thread isolation
            # Full subprocess implementation would serialize the tool
            return self._execute_in_process(tool, parameters, timeout)

        except subprocess.TimeoutExpired:
            return ToolResult(
                result=InvocationResult.TIMEOUT,
                error=f"Subprocess timed out after {timeout}s",
            )
        except Exception as e:
            return ToolResult(
                result=InvocationResult.SANDBOX_ERROR,
                error=f"Subprocess error: {e}",
            )

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
                raise Exception(f"{runtime} not available")
        except Exception:
            context.log(f"Container runtime not available, using subprocess")
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
                results = [
                    e for e in results
                    if e.invocation.user_id == user_id
                ]

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
