"""
Agent OS Tool Integration Framework

Provides secure tool execution with sandboxing, permissions, and Smith validation.

Components:
- ToolInterface: Base interface for all tools
- ToolRegistry: Tool registration and lifecycle management
- PermissionManager: Role-based access control for tools
- ToolApprovalValidator: Smith-based security validation
- ToolExecutor: Sandboxed tool execution
- ToolsClient: High-level client for Agent-OS integration

Usage:
    from src.tools import ToolsClient, create_tools_client

    # Create client
    client = create_tools_client()

    # Register a tool
    from src.tools import BaseTool, ToolCategory, ToolRiskLevel

    class MyTool(BaseTool):
        def __init__(self):
            super().__init__(
                name="my_tool",
                description="Does something useful",
                category=ToolCategory.UTILITY,
                risk_level=ToolRiskLevel.LOW,
            )
            self.add_parameter("input", "string", "Input value")

        def execute(self, parameters):
            result = process(parameters["input"])
            return ToolResult(result=InvocationResult.SUCCESS, output=result)

    registration = client.register_tool(MyTool(), auto_approve=True)

    # Grant permission
    client.grant_permission("user123")

    # Invoke tool
    result = client.invoke(
        tool_name="my_tool",
        parameters={"input": "hello"},
        user_id="user123",
    )

    if result.is_success:
        print(result.output)
"""

# Interface and types
from .interface import (
    ToolInterface,
    BaseTool,
    FunctionTool,
    ToolSchema,
    ToolParameter,
    ToolResult,
    ToolInvocation,
    ToolCategory,
    ToolRiskLevel,
    ToolStatus,
    InvocationResult,
    create_function_tool,
)

# Registry
from .registry import (
    ToolRegistry,
    ToolRegistration,
    ToolQuery,
    create_registry,
    get_registry,
    set_registry,
)

# Permissions
from .permissions import (
    PermissionManager,
    PermissionGrant,
    PermissionDenial,
    PermissionLevel,
    GrantType,
    RiskLimitPolicy,
    PermissionCheckResult,
    create_permission_manager,
)

# Validation (Smith integration)
from .validation import (
    ToolApprovalValidator,
    ToolApprovalResult,
    ApprovalResult,
    DenialReason,
    SecurityCheck,
    create_tool_validator,
)

# Executor
from .executor import (
    ToolExecutor,
    ExecutionConfig,
    ExecutionMode,
    ExecutionState,
    ExecutionContext,
    create_executor,
)

# Client
from .client import (
    ToolsClient,
    ToolsClientConfig,
    create_tools_client,
    get_default_client,
    set_default_client,
    register_tool,
    register_function,
    invoke,
    list_tools,
    can_use,
)


__all__ = [
    # Interface
    "ToolInterface",
    "BaseTool",
    "FunctionTool",
    "ToolSchema",
    "ToolParameter",
    "ToolResult",
    "ToolInvocation",
    "ToolCategory",
    "ToolRiskLevel",
    "ToolStatus",
    "InvocationResult",
    "create_function_tool",
    # Registry
    "ToolRegistry",
    "ToolRegistration",
    "ToolQuery",
    "create_registry",
    "get_registry",
    "set_registry",
    # Permissions
    "PermissionManager",
    "PermissionGrant",
    "PermissionDenial",
    "PermissionLevel",
    "GrantType",
    "RiskLimitPolicy",
    "PermissionCheckResult",
    "create_permission_manager",
    # Validation
    "ToolApprovalValidator",
    "ToolApprovalResult",
    "ApprovalResult",
    "DenialReason",
    "SecurityCheck",
    "create_tool_validator",
    # Executor
    "ToolExecutor",
    "ExecutionConfig",
    "ExecutionMode",
    "ExecutionState",
    "ExecutionContext",
    "create_executor",
    # Client
    "ToolsClient",
    "ToolsClientConfig",
    "create_tools_client",
    "get_default_client",
    "set_default_client",
    "register_tool",
    "register_function",
    "invoke",
    "list_tools",
    "can_use",
]
