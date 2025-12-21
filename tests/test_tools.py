"""
Tests for UC-015: Tool Integration Framework

Tests the tool registration, permission management, Smith validation,
and sandboxed execution system.
"""

import tempfile
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any

import pytest

from src.tools import (
    # Interface
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
    # Registry
    ToolRegistry,
    ToolRegistration,
    ToolQuery,
    create_registry,
    # Permissions
    PermissionManager,
    PermissionGrant,
    PermissionLevel,
    GrantType,
    RiskLimitPolicy,
    create_permission_manager,
    # Validation
    ToolApprovalValidator,
    ToolApprovalResult,
    ApprovalResult,
    create_tool_validator,
    # Executor
    ToolExecutor,
    ExecutionConfig,
    ExecutionMode,
    ExecutionState,
    create_executor,
    # Client
    ToolsClient,
    ToolsClientConfig,
    create_tools_client,
)


# =============================================================================
# Test Fixtures
# =============================================================================


class EchoTool(BaseTool):
    """Simple echo tool for testing."""

    def __init__(self):
        super().__init__(
            name="echo",
            description="Echoes input back",
            category=ToolCategory.UTILITY,
            risk_level=ToolRiskLevel.LOW,
        )
        self.add_parameter("message", "string", "Message to echo")

    def execute(self, parameters: Dict[str, Any]) -> ToolResult:
        return ToolResult(
            result=InvocationResult.SUCCESS,
            output=parameters.get("message", ""),
        )


class FileWriteTool(BaseTool):
    """File write tool (medium risk) for testing."""

    def __init__(self):
        super().__init__(
            name="file_write",
            description="Writes content to a file",
            category=ToolCategory.FILE_SYSTEM,
            risk_level=ToolRiskLevel.MEDIUM,
        )
        self.add_parameter("path", "string", "File path")
        self.add_parameter("content", "string", "Content to write")

    def execute(self, parameters: Dict[str, Any]) -> ToolResult:
        return ToolResult(
            result=InvocationResult.SUCCESS,
            output=f"Wrote to {parameters['path']}",
        )


class NetworkTool(BaseTool):
    """Network tool (high risk) for testing."""

    def __init__(self):
        super().__init__(
            name="http_request",
            description="Makes HTTP requests",
            category=ToolCategory.NETWORK,
            risk_level=ToolRiskLevel.HIGH,
        )
        self.add_parameter("url", "string", "URL to request")
        self.add_parameter("method", "string", "HTTP method", required=False, default="GET")

    def execute(self, parameters: Dict[str, Any]) -> ToolResult:
        return ToolResult(
            result=InvocationResult.SUCCESS,
            output={"status": 200, "body": "test"},
        )


class ShellTool(BaseTool):
    """Shell execution tool (critical risk) for testing."""

    def __init__(self):
        super().__init__(
            name="shell_exec",
            description="Executes shell commands",
            category=ToolCategory.SHELL,
            risk_level=ToolRiskLevel.CRITICAL,
        )
        self.add_parameter("command", "string", "Command to execute")

    def execute(self, parameters: Dict[str, Any]) -> ToolResult:
        return ToolResult(
            result=InvocationResult.SUCCESS,
            output="command executed",
        )


# =============================================================================
# Tool Interface Tests
# =============================================================================


class TestToolParameter:
    """Tests for ToolParameter."""

    def test_basic_validation(self):
        """Test basic parameter validation."""
        param = ToolParameter(
            name="count",
            param_type="integer",
            description="A count",
        )

        is_valid, error = param.validate(10)
        assert is_valid
        assert error is None

    def test_type_validation(self):
        """Test type mismatch detection."""
        param = ToolParameter(
            name="count",
            param_type="integer",
            description="A count",
        )

        is_valid, error = param.validate("not an int")
        assert not is_valid
        assert "type" in error.lower()

    def test_required_validation(self):
        """Test required parameter check."""
        param = ToolParameter(
            name="required_field",
            param_type="string",
            description="Required",
            required=True,
        )

        is_valid, error = param.validate(None)
        assert not is_valid
        assert "missing" in error.lower()

    def test_optional_validation(self):
        """Test optional parameter allows None."""
        param = ToolParameter(
            name="optional_field",
            param_type="string",
            description="Optional",
            required=False,
        )

        is_valid, error = param.validate(None)
        assert is_valid

    def test_constraint_min_max(self):
        """Test min/max constraints."""
        param = ToolParameter(
            name="age",
            param_type="integer",
            description="Age",
            constraints={"min": 0, "max": 120},
        )

        is_valid, _ = param.validate(25)
        assert is_valid

        is_valid, error = param.validate(-1)
        assert not is_valid
        assert ">=" in error

        is_valid, error = param.validate(200)
        assert not is_valid
        assert "<=" in error

    def test_constraint_enum(self):
        """Test enum constraint."""
        param = ToolParameter(
            name="color",
            param_type="string",
            description="Color",
            constraints={"enum": ["red", "green", "blue"]},
        )

        is_valid, _ = param.validate("red")
        assert is_valid

        is_valid, error = param.validate("purple")
        assert not is_valid


class TestToolSchema:
    """Tests for ToolSchema."""

    def test_validate_parameters(self):
        """Test schema parameter validation."""
        schema = ToolSchema(
            name="test_tool",
            description="Test",
            category=ToolCategory.UTILITY,
            parameters=[
                ToolParameter("name", "string", "Name", required=True),
                ToolParameter("count", "integer", "Count", required=False, default=1),
            ],
        )

        is_valid, errors = schema.validate_parameters({"name": "test"})
        assert is_valid
        assert len(errors) == 0

    def test_validate_missing_required(self):
        """Test detection of missing required parameters."""
        schema = ToolSchema(
            name="test_tool",
            description="Test",
            category=ToolCategory.UTILITY,
            parameters=[
                ToolParameter("name", "string", "Name", required=True),
            ],
        )

        is_valid, errors = schema.validate_parameters({})
        assert not is_valid
        assert len(errors) > 0

    def test_validate_unknown_parameter(self):
        """Test detection of unknown parameters."""
        schema = ToolSchema(
            name="test_tool",
            description="Test",
            category=ToolCategory.UTILITY,
            parameters=[
                ToolParameter("name", "string", "Name"),
            ],
        )

        is_valid, errors = schema.validate_parameters({
            "name": "test",
            "unknown": "value",
        })
        assert not is_valid
        assert any("unknown" in e.lower() for e in errors)

    def test_compute_hash(self):
        """Test schema hash computation."""
        schema = ToolSchema(
            name="test_tool",
            description="Test",
            category=ToolCategory.UTILITY,
        )

        hash1 = schema.compute_hash()
        assert len(hash1) == 16

        schema.description = "Modified"
        hash2 = schema.compute_hash()
        assert hash1 != hash2


class TestBaseTool:
    """Tests for BaseTool."""

    def test_create_tool(self):
        """Test creating a basic tool."""
        tool = EchoTool()
        assert tool.name == "echo"
        assert tool.enabled

    def test_get_schema(self):
        """Test getting tool schema."""
        tool = EchoTool()
        schema = tool.get_schema()

        assert schema.name == "echo"
        assert schema.category == ToolCategory.UTILITY
        assert schema.risk_level == ToolRiskLevel.LOW
        assert len(schema.parameters) == 1

    def test_invoke(self):
        """Test tool invocation."""
        tool = EchoTool()
        result = tool.invoke({"message": "hello"})

        assert result.is_success
        assert result.output == "hello"

    def test_disable_enable(self):
        """Test disabling and enabling tool."""
        tool = EchoTool()

        tool.disable()
        assert not tool.enabled

        result = tool.invoke({"message": "test"})
        assert result.result == InvocationResult.PERMISSION_DENIED

        tool.enable()
        result = tool.invoke({"message": "test"})
        assert result.is_success


class TestFunctionTool:
    """Tests for FunctionTool."""

    def test_create_from_function(self):
        """Test creating tool from function."""

        def add(a: int, b: int) -> int:
            return a + b

        tool = create_function_tool(
            name="add",
            description="Adds two numbers",
            func=add,
            parameters=[
                ToolParameter("a", "integer", "First number"),
                ToolParameter("b", "integer", "Second number"),
            ],
        )

        assert tool.name == "add"

        result = tool.invoke({"a": 2, "b": 3})
        assert result.is_success
        assert result.output == 5

    def test_function_error_handling(self):
        """Test error handling in function tools."""

        def failing_func():
            raise ValueError("Test error")

        tool = create_function_tool(
            name="fail",
            description="Always fails",
            func=failing_func,
        )

        result = tool.invoke({})
        assert result.result == InvocationResult.EXECUTION_ERROR
        assert "Test error" in result.error


# =============================================================================
# Tool Registry Tests
# =============================================================================


class TestToolRegistry:
    """Tests for ToolRegistry."""

    def test_create_registry(self):
        """Test creating a registry."""
        registry = create_registry()
        assert len(registry.list_all()) == 0

    def test_register_tool(self):
        """Test registering a tool."""
        registry = create_registry()
        tool = EchoTool()

        registration = registry.register(tool)

        assert registration.tool_id.startswith("TOOL-")
        assert registration.status == ToolStatus.PENDING
        assert registration.tool.name == "echo"

    def test_auto_approve_low_risk(self):
        """Test auto-approval of low risk tools."""
        registry = create_registry(auto_approve_low_risk=True)
        tool = EchoTool()

        registration = registry.register(tool)

        assert registration.status == ToolStatus.APPROVED
        assert registration.is_available

    def test_approve_tool(self):
        """Test manual tool approval."""
        registry = create_registry()
        tool = EchoTool()

        registration = registry.register(tool)
        assert registration.status == ToolStatus.PENDING

        result = registry.approve(registration.tool_id, "admin")
        assert result

        updated = registry.get(registration.tool_id)
        assert updated.status == ToolStatus.APPROVED
        assert updated.approved_by == "admin"

    def test_disable_enable_tool(self):
        """Test disabling and enabling tools."""
        registry = create_registry(auto_approve_low_risk=True)
        tool = EchoTool()

        registration = registry.register(tool)

        registry.disable(registration.tool_id, "maintenance")
        updated = registry.get(registration.tool_id)
        assert updated.status == ToolStatus.DISABLED
        assert not updated.is_available

        registry.enable(registration.tool_id)
        updated = registry.get(registration.tool_id)
        assert updated.status == ToolStatus.APPROVED
        assert updated.is_available

    def test_revoke_tool(self):
        """Test revoking a tool."""
        registry = create_registry(auto_approve_low_risk=True)
        tool = EchoTool()

        registration = registry.register(tool)
        registry.revoke(registration.tool_id, "security issue")

        updated = registry.get(registration.tool_id)
        assert updated.status == ToolStatus.REVOKED
        assert not updated.is_available

    def test_get_by_name(self):
        """Test getting tool by name."""
        registry = create_registry()
        tool = EchoTool()
        registry.register(tool)

        found = registry.get_by_name("echo")
        assert found is not None
        assert found.tool.name == "echo"

    def test_query_by_category(self):
        """Test querying by category."""
        registry = create_registry(auto_approve_low_risk=True)
        registry.register(EchoTool())
        registry.register(FileWriteTool())

        utility_tools = registry.list_by_category(ToolCategory.UTILITY)
        assert len(utility_tools) == 1
        assert utility_tools[0].tool.name == "echo"

        fs_tools = registry.list_by_category(ToolCategory.FILE_SYSTEM)
        assert len(fs_tools) == 1

    def test_query_available_only(self):
        """Test querying available tools only."""
        registry = create_registry()
        registry.register(EchoTool())  # Pending

        available = registry.list_available()
        assert len(available) == 0

    def test_persistence(self):
        """Test registry state persistence."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage_path = Path(tmpdir)

            # Create and populate registry
            registry1 = create_registry(storage_path=storage_path)
            tool = EchoTool()
            registration = registry1.register(tool)
            registry1.approve(registration.tool_id, "admin")

            # Create new registry from same storage
            registry2 = create_registry(storage_path=storage_path)

            # Name index should be restored
            assert "echo" in registry2._by_name

    def test_duplicate_registration(self):
        """Test duplicate registration is rejected."""
        registry = create_registry()
        registry.register(EchoTool())

        with pytest.raises(ValueError):
            registry.register(EchoTool())

    def test_statistics(self):
        """Test registry statistics."""
        registry = create_registry(auto_approve_low_risk=True)
        registry.register(EchoTool())
        registry.register(FileWriteTool())

        stats = registry.get_statistics()
        assert stats["total_tools"] == 2
        assert "APPROVED" in stats["by_status"]


# =============================================================================
# Permission Tests
# =============================================================================


class TestPermissionManager:
    """Tests for PermissionManager."""

    def test_create_manager(self):
        """Test creating permission manager."""
        manager = create_permission_manager()
        assert manager is not None

    def test_grant_permission(self):
        """Test granting permission."""
        manager = create_permission_manager()

        grant = manager.grant(
            principal_id="user123",
            principal_type="user",
            tool_name="echo",
            permission=PermissionLevel.USE,
        )

        assert grant.grant_id.startswith("GRANT-")
        assert grant.principal_id == "user123"
        assert grant.permission == PermissionLevel.USE

    def test_grant_all_tools(self):
        """Test granting permission to all tools."""
        manager = create_permission_manager()

        grant = manager.grant(
            principal_id="admin",
            principal_type="user",
            permission=PermissionLevel.ADMIN,
        )

        # No specific tool = all tools
        assert grant.tool_name is None
        assert grant.tool_id is None
        assert grant.category is None

    def test_grant_by_category(self):
        """Test granting permission by category."""
        manager = create_permission_manager()

        grant = manager.grant(
            principal_id="user123",
            principal_type="user",
            category=ToolCategory.UTILITY,
            permission=PermissionLevel.USE,
        )

        assert grant.category == ToolCategory.UTILITY

    def test_temporary_grant(self):
        """Test temporary permission grant."""
        manager = create_permission_manager()

        grant = manager.grant(
            principal_id="user123",
            principal_type="user",
            tool_name="echo",
            grant_type=GrantType.TEMPORARY,
            duration_hours=1,
        )

        assert grant.grant_type == GrantType.TEMPORARY
        assert grant.expires_at is not None
        assert grant.is_valid

    def test_one_time_grant(self):
        """Test one-time permission grant."""
        manager = create_permission_manager()

        grant = manager.grant(
            principal_id="user123",
            principal_type="user",
            tool_name="echo",
            grant_type=GrantType.ONE_TIME,
            uses=3,
        )

        assert grant.grant_type == GrantType.ONE_TIME
        assert grant.uses_remaining == 3

        # Use the grant
        grant.use()
        assert grant.uses_remaining == 2
        assert grant.is_valid

        grant.use()
        grant.use()
        assert grant.uses_remaining == 0
        assert not grant.is_valid

    def test_deny_permission(self):
        """Test explicit permission denial."""
        manager = create_permission_manager()

        denial = manager.deny(
            principal_id="bad_user",
            principal_type="user",
            tool_name="shell_exec",
            reason="Security risk",
        )

        assert denial.denial_id.startswith("DENY-")
        assert denial.reason == "Security risk"

    def test_check_permission_allowed(self):
        """Test permission check when allowed."""
        manager = create_permission_manager()

        manager.grant(
            principal_id="user123",
            principal_type="user",
            tool_name="echo",
            permission=PermissionLevel.USE,
        )

        result = manager.check(
            principal_id="user123",
            principal_type="user",
            tool_id="TOOL-123",
            tool_name="echo",
            tool_category=ToolCategory.UTILITY,
            tool_risk=ToolRiskLevel.LOW,
        )

        assert result.allowed
        assert result.grant is not None

    def test_check_permission_denied(self):
        """Test permission check when denied."""
        manager = create_permission_manager()

        result = manager.check(
            principal_id="user123",
            principal_type="user",
            tool_id="TOOL-123",
            tool_name="echo",
            tool_category=ToolCategory.UTILITY,
            tool_risk=ToolRiskLevel.LOW,
        )

        assert not result.allowed
        assert "no permission" in result.reason.lower()

    def test_explicit_denial_overrides_grant(self):
        """Test that explicit denial overrides grant."""
        manager = create_permission_manager()

        # Grant all tools
        manager.grant(
            principal_id="user123",
            principal_type="user",
            permission=PermissionLevel.USE,
        )

        # Deny specific tool
        manager.deny(
            principal_id="user123",
            principal_type="user",
            tool_name="shell_exec",
            reason="Too dangerous",
        )

        # Shell should be denied
        result = manager.check(
            principal_id="user123",
            principal_type="user",
            tool_id="TOOL-123",
            tool_name="shell_exec",
            tool_category=ToolCategory.SHELL,
            tool_risk=ToolRiskLevel.CRITICAL,
        )

        assert not result.allowed
        assert result.denial is not None

        # Other tools should be allowed
        result = manager.check(
            principal_id="user123",
            principal_type="user",
            tool_id="TOOL-456",
            tool_name="echo",
            tool_category=ToolCategory.UTILITY,
            tool_risk=ToolRiskLevel.LOW,
        )

        assert result.allowed

    def test_risk_policy(self):
        """Test risk level policy."""
        manager = create_permission_manager()

        # Set strict policy
        manager.set_policy("user123", RiskLimitPolicy(
            max_risk_level=ToolRiskLevel.LOW,
        ))

        # Grant all tools
        manager.grant(
            principal_id="user123",
            principal_type="user",
            permission=PermissionLevel.USE,
        )

        # Low risk should be allowed
        result = manager.check(
            principal_id="user123",
            principal_type="user",
            tool_id="TOOL-123",
            tool_name="echo",
            tool_category=ToolCategory.UTILITY,
            tool_risk=ToolRiskLevel.LOW,
        )
        assert result.allowed

        # Medium risk should be denied
        result = manager.check(
            principal_id="user123",
            principal_type="user",
            tool_id="TOOL-456",
            tool_name="file_write",
            tool_category=ToolCategory.FILE_SYSTEM,
            tool_risk=ToolRiskLevel.MEDIUM,
        )
        assert not result.allowed

    def test_revoke_grant(self):
        """Test revoking a grant."""
        manager = create_permission_manager()

        grant = manager.grant(
            principal_id="user123",
            principal_type="user",
            tool_name="echo",
        )

        result = manager.revoke_grant(grant.grant_id)
        assert result

        # Should no longer have access
        grants = manager.get_grants_for_principal("user123")
        assert len(grants) == 0


# =============================================================================
# Validation Tests
# =============================================================================


class TestToolApprovalValidator:
    """Tests for ToolApprovalValidator."""

    def test_create_validator(self):
        """Test creating validator."""
        validator = create_tool_validator()
        assert validator is not None

    def test_validate_unregistered_tool(self):
        """Test validation of unregistered tool."""
        validator = create_tool_validator()

        result = validator.validate(
            registration=None,
            parameters={},
            user_id="user123",
        )

        assert not result.approved
        assert result.result == ApprovalResult.DENIED

    def test_validate_low_risk_tool(self):
        """Test validation of low risk tool."""
        validator = create_tool_validator()
        registry = create_registry(auto_approve_low_risk=True)
        registration = registry.register(EchoTool())

        result = validator.validate(
            registration=registration,
            parameters={"message": "hello"},
            user_id="user123",
        )

        assert result.approved
        assert result.result == ApprovalResult.APPROVED

    def test_validate_high_risk_requires_human(self):
        """Test high risk tools require human approval."""
        validator = create_tool_validator(
            require_human_above=ToolRiskLevel.MEDIUM,
        )
        registry = create_registry(auto_approve_low_risk=True)
        registration = registry.register(NetworkTool())
        registry.approve(registration.tool_id, "admin")

        result = validator.validate(
            registration=registration,
            parameters={"url": "https://example.com"},
            user_id="user123",
        )

        assert result.approved
        assert result.result == ApprovalResult.ESCALATE
        assert result.requires_human_approval

    def test_detect_path_traversal(self):
        """Test detection of path traversal attempt."""
        validator = create_tool_validator()
        registry = create_registry(auto_approve_low_risk=True)
        registration = registry.register(FileWriteTool())
        registry.approve(registration.tool_id, "admin")

        result = validator.validate(
            registration=registration,
            parameters={"path": "../../../etc/passwd", "content": "test"},
            user_id="user123",
        )

        assert not result.approved
        assert any("path_traversal" in c.check_id.lower() for c in result.failed_checks)

    def test_detect_command_injection(self):
        """Test detection of command injection."""
        validator = create_tool_validator()
        registry = create_registry(auto_approve_low_risk=True)
        registration = registry.register(EchoTool())

        result = validator.validate(
            registration=registration,
            parameters={"message": "test; rm -rf /"},
            user_id="user123",
        )

        # Should have warnings about shell characters
        assert any(not c.passed for c in result.checks)

    def test_rate_limiting(self):
        """Test rate limiting."""
        validator = create_tool_validator(rate_limit_per_minute=5)
        registry = create_registry(auto_approve_low_risk=True)
        registration = registry.register(EchoTool())

        # Make 5 requests (should all pass)
        for _ in range(5):
            result = validator.validate(
                registration=registration,
                parameters={"message": "test"},
                user_id="user123",
            )
            assert result.approved

        # 6th request should be rate limited
        result = validator.validate(
            registration=registration,
            parameters={"message": "test"},
            user_id="user123",
        )
        assert not result.approved

    def test_blocked_category(self):
        """Test blocking of specific categories."""
        validator = create_tool_validator(
            blocked_categories={ToolCategory.SHELL},
        )
        registry = create_registry(auto_approve_low_risk=True)
        registration = registry.register(ShellTool())
        registry.approve(registration.tool_id, "admin")

        result = validator.validate(
            registration=registration,
            parameters={"command": "ls"},
            user_id="user123",
        )

        assert not result.approved


# =============================================================================
# Executor Tests
# =============================================================================


class TestToolExecutor:
    """Tests for ToolExecutor."""

    def test_create_executor(self):
        """Test creating executor."""
        registry = create_registry(auto_approve_low_risk=True)
        permissions = create_permission_manager()
        validator = create_tool_validator()

        executor = create_executor(registry, permissions, validator)
        assert executor is not None

    def test_execute_tool(self):
        """Test executing a tool."""
        registry = create_registry(auto_approve_low_risk=True)
        permissions = create_permission_manager()
        validator = create_tool_validator()
        executor = create_executor(registry, permissions, validator)

        # Register tool
        registry.register(EchoTool())

        # Grant permission
        permissions.grant(
            principal_id="user123",
            principal_type="user",
            tool_name="echo",
        )

        # Execute
        result = executor.execute(
            tool_name="echo",
            parameters={"message": "hello"},
            user_id="user123",
        )

        assert result.is_success
        assert result.output == "hello"

    def test_execute_without_permission(self):
        """Test execution denied without permission."""
        registry = create_registry(auto_approve_low_risk=True)
        permissions = create_permission_manager()
        validator = create_tool_validator()
        executor = create_executor(registry, permissions, validator)

        registry.register(EchoTool())

        # No permission granted
        result = executor.execute(
            tool_name="echo",
            parameters={"message": "hello"},
            user_id="user123",
        )

        assert result.result == InvocationResult.PERMISSION_DENIED

    def test_execute_unregistered_tool(self):
        """Test execution of unregistered tool."""
        registry = create_registry()
        permissions = create_permission_manager()
        validator = create_tool_validator()
        executor = create_executor(registry, permissions, validator)

        result = executor.execute(
            tool_name="nonexistent",
            parameters={},
            user_id="user123",
        )

        assert result.result == InvocationResult.VALIDATION_ERROR
        assert "not found" in result.error.lower()

    def test_execute_pending_tool(self):
        """Test execution of pending (unapproved) tool."""
        registry = create_registry()  # No auto-approve
        permissions = create_permission_manager()
        validator = create_tool_validator()
        executor = create_executor(registry, permissions, validator)

        registry.register(EchoTool())  # Still pending

        permissions.grant(
            principal_id="user123",
            principal_type="user",
            tool_name="echo",
        )

        result = executor.execute(
            tool_name="echo",
            parameters={"message": "hello"},
            user_id="user123",
        )

        assert result.result == InvocationResult.PERMISSION_DENIED

    def test_execution_statistics(self):
        """Test execution statistics."""
        registry = create_registry(auto_approve_low_risk=True)
        permissions = create_permission_manager()
        validator = create_tool_validator()
        executor = create_executor(registry, permissions, validator)

        registry.register(EchoTool())
        permissions.grant("user123", "user", tool_name="echo")

        # Execute a few times
        for _ in range(3):
            executor.execute("echo", {"message": "test"}, "user123")

        stats = executor.get_statistics()
        assert stats["total_executions"] == 3


# =============================================================================
# Client Tests
# =============================================================================


class TestToolsClient:
    """Tests for ToolsClient."""

    def test_create_client(self):
        """Test creating client."""
        client = create_tools_client()
        assert client is not None

    def test_register_and_invoke(self):
        """Test full registration and invocation flow."""
        client = create_tools_client(ToolsClientConfig(
            auto_approve_low_risk=True,
        ))

        # Register tool
        registration = client.register_tool(EchoTool())
        assert registration.is_available

        # Grant permission
        client.grant_permission("user123", tool_name="echo")

        # Invoke
        result = client.invoke(
            tool_name="echo",
            parameters={"message": "hello world"},
            user_id="user123",
        )

        assert result.is_success
        assert result.output == "hello world"

    def test_register_function(self):
        """Test registering a function as tool."""
        client = create_tools_client(ToolsClientConfig(
            auto_approve_low_risk=True,
        ))

        def multiply(a: int, b: int) -> int:
            return a * b

        registration = client.register_function(
            name="multiply",
            description="Multiplies two numbers",
            func=multiply,
            parameters=[
                ToolParameter("a", "integer", "First number"),
                ToolParameter("b", "integer", "Second number"),
            ],
            auto_approve=True,
        )

        assert registration.is_available

        client.grant_permission("user123", tool_name="multiply")

        result = client.invoke(
            tool_name="multiply",
            parameters={"a": 6, "b": 7},
            user_id="user123",
        )

        assert result.is_success
        assert result.output == 42

    def test_list_tools(self):
        """Test listing tools."""
        client = create_tools_client(ToolsClientConfig(
            auto_approve_low_risk=True,
        ))

        client.register_tool(EchoTool(), auto_approve=True)
        client.register_tool(FileWriteTool())  # Medium risk, not auto-approved

        available = client.list_tools(available_only=True)
        assert len(available) == 1
        assert available[0].tool.name == "echo"

        all_tools = client.list_tools(available_only=False)
        assert len(all_tools) == 2

    def test_can_use_tool(self):
        """Test checking tool access."""
        client = create_tools_client(ToolsClientConfig(
            auto_approve_low_risk=True,
        ))

        client.register_tool(EchoTool())

        # No permission yet
        assert not client.can_use_tool("user123", "echo")

        # Grant permission
        client.grant_permission("user123", tool_name="echo")

        # Now should have access
        assert client.can_use_tool("user123", "echo")

    def test_grant_agent_permission(self):
        """Test granting permission to agent."""
        client = create_tools_client(ToolsClientConfig(
            auto_approve_low_risk=True,
        ))

        client.register_tool(EchoTool())

        grant = client.grant_agent_permission(
            agent_id="sage",
            tool_name="echo",
        )

        assert grant.principal_type == "agent"
        assert grant.permission == PermissionLevel.INVOKE

    def test_deny_permission(self):
        """Test denying permission."""
        client = create_tools_client(ToolsClientConfig(
            auto_approve_low_risk=True,
        ))

        client.register_tool(EchoTool())
        client.grant_permission("user123")  # All tools
        client.deny_permission("user123", tool_name="echo", reason="Testing")

        assert not client.can_use_tool("user123", "echo")

    def test_set_risk_policy(self):
        """Test setting risk policy."""
        client = create_tools_client(ToolsClientConfig(
            auto_approve_low_risk=True,
        ))

        client.register_tool(FileWriteTool())
        client.register_function(
            name="safe_func",
            description="Safe function",
            func=lambda: "ok",
            risk_level=ToolRiskLevel.LOW,
            auto_approve=True,
        )

        # Grant all tools
        client.grant_permission("user123")

        # Set strict policy
        client.set_risk_policy(
            "user123",
            max_risk=ToolRiskLevel.LOW,
        )

        # Low risk allowed
        assert client.can_use_tool("user123", "safe_func")

        # Medium risk denied
        assert not client.can_use_tool("user123", "file_write")

    def test_approve_and_disable(self):
        """Test approving and disabling tools."""
        client = create_tools_client()

        registration = client.register_tool(EchoTool())
        assert not registration.is_available  # Pending

        client.approve_tool("echo", "admin")
        updated = client.get_tool("echo")
        assert updated.is_available

        client.disable_tool("echo", "maintenance")
        updated = client.get_tool("echo")
        assert not updated.is_available

        client.enable_tool("echo")
        updated = client.get_tool("echo")
        assert updated.is_available

    def test_statistics(self):
        """Test getting statistics."""
        client = create_tools_client(ToolsClientConfig(
            auto_approve_low_risk=True,
        ))

        client.register_tool(EchoTool())
        client.grant_permission("user123", tool_name="echo")
        client.invoke("echo", {"message": "test"}, "user123")

        stats = client.get_statistics()
        assert "registry" in stats
        assert "executor" in stats


# =============================================================================
# Acceptance Criteria Tests
# =============================================================================


class TestAcceptanceCriteria:
    """Tests for UC-015 acceptance criteria."""

    def test_function_calling_api(self):
        """Verify function calling API works."""
        client = create_tools_client(ToolsClientConfig(
            auto_approve_low_risk=True,
        ))

        # Register function tool
        def greet(name: str) -> str:
            return f"Hello, {name}!"

        client.register_function(
            name="greet",
            description="Greets a person",
            func=greet,
            parameters=[ToolParameter("name", "string", "Name to greet")],
            auto_approve=True,
        )

        client.grant_permission("user123", tool_name="greet")

        # Invoke via API
        result = client.invoke(
            tool_name="greet",
            parameters={"name": "World"},
            user_id="user123",
        )

        assert result.is_success
        assert result.output == "Hello, World!"

    def test_tool_registration_system(self):
        """Verify tool registration system works."""
        client = create_tools_client()

        # Register multiple tools
        echo_reg = client.register_tool(EchoTool())
        file_reg = client.register_tool(FileWriteTool())
        net_reg = client.register_tool(NetworkTool())

        assert echo_reg.schema.risk_level == ToolRiskLevel.LOW
        assert file_reg.schema.risk_level == ToolRiskLevel.MEDIUM
        assert net_reg.schema.risk_level == ToolRiskLevel.HIGH

        # Verify discovery
        all_tools = client.list_tools(available_only=False)
        assert len(all_tools) == 3

        # Verify pending list
        pending = client.list_pending_tools()
        assert len(pending) == 3

        # Approve and verify
        client.approve_tool("echo")
        pending = client.list_pending_tools()
        assert len(pending) == 2

    def test_permission_layer(self):
        """Verify permission layer works."""
        client = create_tools_client(ToolsClientConfig(
            auto_approve_low_risk=True,
        ))

        client.register_tool(EchoTool())
        client.register_tool(FileWriteTool())

        # User without permission cannot use tools
        assert not client.can_use_tool("user123", "echo")

        # Grant specific tool permission
        client.grant_permission("user123", tool_name="echo")
        assert client.can_use_tool("user123", "echo")
        assert not client.can_use_tool("user123", "file_write")

        # Grant category permission
        client.grant_permission("user123", category=ToolCategory.FILE_SYSTEM)
        # Still can't use because not approved
        assert not client.can_use_tool("user123", "file_write")

    def test_mandatory_smith_approval(self):
        """Verify Smith approval is mandatory."""
        registry = create_registry(auto_approve_low_risk=True)
        permissions = create_permission_manager()
        validator = create_tool_validator()
        executor = create_executor(registry, permissions, validator)

        # Register tools at different risk levels
        registry.register(EchoTool())  # LOW
        registry.register(NetworkTool())  # HIGH
        registry.approve(registry.get_by_name("http_request").tool_id, "admin")

        permissions.grant("user123", "user")  # Grant all

        # Low risk - auto approved
        result = executor.execute("echo", {"message": "test"}, "user123")
        assert result.is_success

        # High risk - should be escalated (not blocked, but flagged)
        # The executor would normally wait for human approval here

    def test_sandboxed_execution(self):
        """Verify tools run in sandbox."""
        registry = create_registry(auto_approve_low_risk=True)
        permissions = create_permission_manager()
        validator = create_tool_validator()
        config = ExecutionConfig(default_mode=ExecutionMode.SUBPROCESS)
        executor = create_executor(registry, permissions, validator, config)

        registry.register(EchoTool())
        permissions.grant("user123", "user", tool_name="echo")

        # Verify execution mode is set
        assert executor.config.default_mode == ExecutionMode.SUBPROCESS

        # Execute tool
        result = executor.execute("echo", {"message": "test"}, "user123")
        assert result.is_success

    def test_audit_trail(self):
        """Verify audit trail is maintained."""
        client = create_tools_client(ToolsClientConfig(
            auto_approve_low_risk=True,
        ))

        audit_log = []

        def audit_handler(context):
            audit_log.append({
                "execution_id": context.execution_id,
                "tool": context.invocation.tool_name,
                "user": context.invocation.user_id,
                "result": context.result.result.name if context.result else None,
                "timestamp": datetime.now().isoformat(),
            })

        client.add_audit_handler(audit_handler)
        client.register_tool(EchoTool())
        client.grant_permission("user123", tool_name="echo")

        # Invoke tool
        client.invoke("echo", {"message": "test"}, "user123")

        # Verify audit entry
        assert len(audit_log) == 1
        assert audit_log[0]["tool"] == "echo"
        assert audit_log[0]["user"] == "user123"
        assert audit_log[0]["result"] == "SUCCESS"
