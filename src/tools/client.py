"""
Tools Client for Agent-OS Integration

Provides a unified high-level interface for tool management and execution.
This is the primary entry point for agents to use tools.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set

from .executor import (
    ExecutionConfig,
    ExecutionContext,
    ExecutionMode,
    ToolExecutor,
    create_executor,
)
from .interface import (
    BaseTool,
    FunctionTool,
    InvocationResult,
    ToolCategory,
    ToolInterface,
    ToolParameter,
    ToolResult,
    ToolRiskLevel,
    ToolSchema,
    create_function_tool,
)
from .permissions import (
    GrantType,
    PermissionGrant,
    PermissionLevel,
    PermissionManager,
    RiskLimitPolicy,
    create_permission_manager,
)
from .registry import (
    ToolQuery,
    ToolRegistration,
    ToolRegistry,
    create_registry,
)
from .validation import (
    ToolApprovalResult,
    ToolApprovalValidator,
    create_tool_validator,
)

logger = logging.getLogger(__name__)


@dataclass
class ToolsClientConfig:
    """Configuration for the tools client."""

    storage_path: Optional[Path] = None
    auto_approve_low_risk: bool = False
    default_execution_mode: ExecutionMode = ExecutionMode.SUBPROCESS
    default_timeout_seconds: int = 30
    rate_limit_per_minute: int = 60
    require_human_above: ToolRiskLevel = ToolRiskLevel.HIGH
    blocked_categories: Set[ToolCategory] = field(default_factory=set)


class ToolsClient:
    """
    High-level client for tool management and execution.

    Provides:
    - Tool registration and discovery
    - Permission management
    - Safe tool execution with Smith validation
    - Audit trail
    """

    def __init__(self, config: Optional[ToolsClientConfig] = None):
        """
        Initialize tools client.

        Args:
            config: Client configuration
        """
        self.config = config or ToolsClientConfig()

        # Initialize components
        storage_path = self.config.storage_path
        registry_path = storage_path / "registry" if storage_path else None
        permissions_path = storage_path / "permissions" if storage_path else None

        self._registry = create_registry(
            storage_path=registry_path,
            auto_approve_low_risk=self.config.auto_approve_low_risk,
        )

        self._permissions = create_permission_manager(
            storage_path=permissions_path,
        )

        self._validator = create_tool_validator(
            require_human_above=self.config.require_human_above,
            blocked_categories=self.config.blocked_categories,
            rate_limit_per_minute=self.config.rate_limit_per_minute,
        )

        exec_config = ExecutionConfig(
            default_mode=self.config.default_execution_mode,
            timeout_seconds=self.config.default_timeout_seconds,
        )

        self._executor = create_executor(
            registry=self._registry,
            permission_manager=self._permissions,
            validator=self._validator,
            config=exec_config,
        )

        self._audit_handlers: List[Callable[[ExecutionContext], None]] = []

    # =========================================================================
    # Tool Registration
    # =========================================================================

    def register_tool(
        self,
        tool: ToolInterface,
        auto_approve: bool = False,
    ) -> ToolRegistration:
        """
        Register a new tool.

        Args:
            tool: Tool to register
            auto_approve: Force auto-approval

        Returns:
            ToolRegistration
        """
        return self._registry.register(
            tool=tool,
            auto_approve=auto_approve,
        )

    def register_function(
        self,
        name: str,
        description: str,
        func: Callable[..., Any],
        parameters: Optional[List[ToolParameter]] = None,
        category: ToolCategory = ToolCategory.UTILITY,
        risk_level: ToolRiskLevel = ToolRiskLevel.LOW,
        auto_approve: bool = False,
    ) -> ToolRegistration:
        """
        Register a function as a tool.

        Args:
            name: Tool name
            description: Tool description
            func: Function to wrap
            parameters: Parameter definitions
            category: Tool category
            risk_level: Risk level
            auto_approve: Force auto-approval

        Returns:
            ToolRegistration
        """
        tool = create_function_tool(
            name=name,
            description=description,
            func=func,
            parameters=parameters,
            category=category,
            risk_level=risk_level,
        )
        return self.register_tool(tool, auto_approve=auto_approve)

    def approve_tool(self, tool_name: str, approved_by: str = "admin") -> bool:
        """
        Approve a pending tool.

        Args:
            tool_name: Tool name
            approved_by: ID of approver

        Returns:
            True if approved
        """
        registration = self._registry.get_by_name(tool_name)
        if not registration:
            return False
        return self._registry.approve(registration.tool_id, approved_by)

    def disable_tool(self, tool_name: str, reason: str) -> bool:
        """
        Disable a tool.

        Args:
            tool_name: Tool name
            reason: Reason for disabling

        Returns:
            True if disabled
        """
        registration = self._registry.get_by_name(tool_name)
        if not registration:
            return False
        return self._registry.disable(registration.tool_id, reason)

    def enable_tool(self, tool_name: str) -> bool:
        """
        Re-enable a disabled tool.

        Args:
            tool_name: Tool name

        Returns:
            True if enabled
        """
        registration = self._registry.get_by_name(tool_name)
        if not registration:
            return False
        return self._registry.enable(registration.tool_id)

    # =========================================================================
    # Tool Discovery
    # =========================================================================

    def get_tool(self, tool_name: str) -> Optional[ToolRegistration]:
        """Get a tool by name."""
        return self._registry.get_by_name(tool_name)

    def list_tools(
        self,
        available_only: bool = True,
        category: Optional[ToolCategory] = None,
        risk_level: Optional[ToolRiskLevel] = None,
    ) -> List[ToolRegistration]:
        """
        List registered tools.

        Args:
            available_only: Only return available tools
            category: Filter by category
            risk_level: Filter by risk level

        Returns:
            List of tool registrations
        """
        query = ToolQuery(available_only=available_only)

        if category:
            query.categories = {category}

        if risk_level:
            query.risk_levels = {risk_level}

        return self._registry.query(query)

    def list_pending_tools(self) -> List[ToolRegistration]:
        """List tools awaiting approval."""
        return self._registry.list_pending()

    def get_tool_schema(self, tool_name: str) -> Optional[ToolSchema]:
        """Get tool schema by name."""
        registration = self._registry.get_by_name(tool_name)
        if registration:
            return registration.schema
        return None

    # =========================================================================
    # Permissions
    # =========================================================================

    def grant_permission(
        self,
        user_id: str,
        tool_name: Optional[str] = None,
        category: Optional[ToolCategory] = None,
        permission: PermissionLevel = PermissionLevel.USE,
        grant_type: GrantType = GrantType.PERMANENT,
        duration_hours: Optional[int] = None,
        granted_by: Optional[str] = None,
    ) -> PermissionGrant:
        """
        Grant tool permission to a user.

        Args:
            user_id: User ID
            tool_name: Specific tool (None = all)
            category: Tool category (None = all)
            permission: Permission level
            grant_type: Type of grant
            duration_hours: Duration for temporary grants
            granted_by: ID of granter

        Returns:
            PermissionGrant
        """
        return self._permissions.grant(
            principal_id=user_id,
            principal_type="user",
            tool_name=tool_name,
            category=category,
            permission=permission,
            grant_type=grant_type,
            duration_hours=duration_hours,
            granted_by=granted_by,
        )

    def grant_agent_permission(
        self,
        agent_id: str,
        tool_name: Optional[str] = None,
        category: Optional[ToolCategory] = None,
        permission: PermissionLevel = PermissionLevel.INVOKE,
    ) -> PermissionGrant:
        """
        Grant tool permission to an agent.

        Args:
            agent_id: Agent ID
            tool_name: Specific tool (None = all)
            category: Tool category (None = all)
            permission: Permission level

        Returns:
            PermissionGrant
        """
        return self._permissions.grant(
            principal_id=agent_id,
            principal_type="agent",
            tool_name=tool_name,
            category=category,
            permission=permission,
        )

    def deny_permission(
        self,
        user_id: str,
        tool_name: Optional[str] = None,
        category: Optional[ToolCategory] = None,
        reason: str = "",
    ) -> None:
        """
        Deny tool permission to a user.

        Args:
            user_id: User ID
            tool_name: Specific tool
            category: Tool category
            reason: Reason for denial
        """
        self._permissions.deny(
            principal_id=user_id,
            principal_type="user",
            tool_name=tool_name,
            category=category,
            reason=reason,
        )

    def set_risk_policy(
        self,
        user_id: str,
        max_risk: ToolRiskLevel = ToolRiskLevel.MEDIUM,
        require_confirmation_above: ToolRiskLevel = ToolRiskLevel.MEDIUM,
        require_human_above: ToolRiskLevel = ToolRiskLevel.HIGH,
    ) -> None:
        """
        Set risk limit policy for a user.

        Args:
            user_id: User ID
            max_risk: Maximum allowed risk level
            require_confirmation_above: Require confirmation above this level
            require_human_above: Require human approval above this level
        """
        policy = RiskLimitPolicy(
            max_risk_level=max_risk,
            require_confirmation_above=require_confirmation_above,
            require_human_approval_above=require_human_above,
        )
        self._permissions.set_policy(user_id, policy)

    def can_use_tool(
        self,
        user_id: str,
        tool_name: str,
    ) -> bool:
        """
        Check if user can use a tool.

        Args:
            user_id: User ID
            tool_name: Tool name

        Returns:
            True if user can use the tool
        """
        registration = self._registry.get_by_name(tool_name)
        if not registration:
            return False

        # Tool must be available (approved and enabled)
        if not registration.is_available:
            return False

        result = self._permissions.check(
            principal_id=user_id,
            principal_type="user",
            tool_id=registration.tool_id,
            tool_name=tool_name,
            tool_category=registration.schema.category,
            tool_risk=registration.schema.risk_level,
        )

        return result.allowed

    # =========================================================================
    # Tool Execution
    # =========================================================================

    def invoke(
        self,
        tool_name: str,
        parameters: Dict[str, Any],
        user_id: str,
        agent_id: Optional[str] = None,
        timeout: Optional[int] = None,
    ) -> ToolResult:
        """
        Invoke a tool.

        Args:
            tool_name: Tool name
            parameters: Tool parameters
            user_id: User ID
            agent_id: Agent ID (if called by agent)
            timeout: Timeout override

        Returns:
            ToolResult
        """
        return self._executor.execute(
            tool_name=tool_name,
            parameters=parameters,
            user_id=user_id,
            agent_id=agent_id,
            timeout_override=timeout,
        )

    def set_human_approval_callback(
        self,
        callback: Callable[[ExecutionContext], bool],
    ) -> None:
        """
        Set callback for human approval requests.

        The callback receives an ExecutionContext and should return
        True if approved, False if denied.

        Args:
            callback: Approval callback function
        """
        self._executor.set_human_approval_callback(callback)

    def add_audit_handler(
        self,
        handler: Callable[[ExecutionContext], None],
    ) -> None:
        """
        Add an audit handler for tool executions.

        Args:
            handler: Handler function
        """
        self._audit_handlers.append(handler)
        self._executor.audit_callback = self._dispatch_audit

    def _dispatch_audit(self, context: ExecutionContext) -> None:
        """Dispatch to all audit handlers."""
        for handler in self._audit_handlers:
            try:
                handler(context)
            except Exception as e:
                logger.warning(f"Audit handler error: {e}")

    # =========================================================================
    # Statistics
    # =========================================================================

    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics."""
        return {
            "registry": self._registry.get_statistics(),
            "executor": self._executor.get_statistics(),
        }


# Default client instance
_default_client: Optional[ToolsClient] = None


def get_default_client() -> ToolsClient:
    """Get the default tools client."""
    global _default_client
    if _default_client is None:
        _default_client = ToolsClient()
    return _default_client


def set_default_client(client: ToolsClient) -> None:
    """Set the default tools client."""
    global _default_client
    _default_client = client


def create_tools_client(
    config: Optional[ToolsClientConfig] = None,
) -> ToolsClient:
    """
    Create a tools client.

    Args:
        config: Client configuration

    Returns:
        ToolsClient instance
    """
    return ToolsClient(config)


# =========================================================================
# Convenience Functions
# =========================================================================


def register_tool(
    tool: ToolInterface,
    auto_approve: bool = False,
) -> ToolRegistration:
    """Register a tool using the default client."""
    return get_default_client().register_tool(tool, auto_approve)


def register_function(
    name: str,
    description: str,
    func: Callable[..., Any],
    parameters: Optional[List[ToolParameter]] = None,
    category: ToolCategory = ToolCategory.UTILITY,
    risk_level: ToolRiskLevel = ToolRiskLevel.LOW,
    auto_approve: bool = False,
) -> ToolRegistration:
    """Register a function as a tool using the default client."""
    return get_default_client().register_function(
        name=name,
        description=description,
        func=func,
        parameters=parameters,
        category=category,
        risk_level=risk_level,
        auto_approve=auto_approve,
    )


def invoke(
    tool_name: str,
    parameters: Dict[str, Any],
    user_id: str,
    agent_id: Optional[str] = None,
) -> ToolResult:
    """Invoke a tool using the default client."""
    return get_default_client().invoke(
        tool_name=tool_name,
        parameters=parameters,
        user_id=user_id,
        agent_id=agent_id,
    )


def list_tools(
    available_only: bool = True,
    category: Optional[ToolCategory] = None,
) -> List[ToolRegistration]:
    """List tools using the default client."""
    return get_default_client().list_tools(
        available_only=available_only,
        category=category,
    )


def can_use(user_id: str, tool_name: str) -> bool:
    """Check if user can use a tool using the default client."""
    return get_default_client().can_use_tool(user_id, tool_name)
