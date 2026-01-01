"""
Agent OS Tool Executor Exceptions

Custom exceptions for sandboxed tool execution.
"""

from typing import Optional


class ToolExecutorError(Exception):
    """Base exception for tool executor operations."""

    def __init__(self, message: str, tool_name: Optional[str] = None):
        self.tool_name = tool_name
        super().__init__(message)


class ToolNotFoundError(ToolExecutorError):
    """Requested tool is not registered."""

    pass


class ToolPermissionError(ToolExecutorError):
    """User/agent does not have permission to execute this tool."""

    def __init__(
        self,
        message: str,
        tool_name: str,
        user_id: Optional[str] = None,
        required_permission: Optional[str] = None,
    ):
        self.user_id = user_id
        self.required_permission = required_permission
        super().__init__(message, tool_name)


class ToolValidationError(ToolExecutorError):
    """Tool parameters or configuration failed validation."""

    pass


class ToolTimeoutError(ToolExecutorError):
    """Tool execution exceeded timeout."""

    def __init__(self, message: str, tool_name: str, timeout_seconds: int):
        self.timeout_seconds = timeout_seconds
        super().__init__(message, tool_name)


class ToolSandboxError(ToolExecutorError):
    """Error occurred in the sandbox environment."""

    def __init__(
        self,
        message: str,
        tool_name: str,
        sandbox_type: str,  # "subprocess", "container", etc.
        original_error: Optional[Exception] = None,
    ):
        self.sandbox_type = sandbox_type
        self.original_error = original_error
        super().__init__(message, tool_name)


class HumanApprovalError(ToolExecutorError):
    """Error during human approval request."""

    def __init__(
        self,
        message: str,
        tool_name: str,
        approval_denied: bool = False,
        original_error: Optional[Exception] = None,
    ):
        self.approval_denied = approval_denied
        self.original_error = original_error
        super().__init__(message, tool_name)


class SmithValidationError(ToolExecutorError):
    """Smith (Guardian) denied the tool execution."""

    def __init__(
        self,
        message: str,
        tool_name: str,
        denial_reason: Optional[str] = None,
    ):
        self.denial_reason = denial_reason
        super().__init__(message, tool_name)
