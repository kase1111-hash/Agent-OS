"""
Tool Integration Interface

Defines the interface and types for external tool integration in Agent OS.
All tools must implement the ToolInterface abstract class.
"""

import hashlib
import secrets
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set, Union
import logging


logger = logging.getLogger(__name__)


class ToolCategory(Enum):
    """Categories of tools."""
    FILE_SYSTEM = "file_system"       # File read/write operations
    NETWORK = "network"               # HTTP, API calls
    SHELL = "shell"                   # Shell command execution
    CODE = "code"                     # Code execution/evaluation
    DATABASE = "database"             # Database operations
    MESSAGING = "messaging"           # Email, SMS, notifications
    SYSTEM = "system"                 # System configuration
    UTILITY = "utility"               # General utilities
    CUSTOM = "custom"                 # User-defined tools


class ToolRiskLevel(Enum):
    """Risk level for tools."""
    LOW = auto()      # Read-only, no side effects
    MEDIUM = auto()   # Writes data, reversible
    HIGH = auto()     # External communication, system changes
    CRITICAL = auto() # Irreversible actions, security-sensitive


class ToolStatus(Enum):
    """Tool registration status."""
    PENDING = auto()       # Awaiting approval
    APPROVED = auto()      # Approved and usable
    DISABLED = auto()      # Temporarily disabled
    REVOKED = auto()       # Permanently revoked
    DEPRECATED = auto()    # No longer recommended


class InvocationResult(Enum):
    """Result of a tool invocation."""
    SUCCESS = auto()
    FAILURE = auto()
    TIMEOUT = auto()
    PERMISSION_DENIED = auto()
    VALIDATION_ERROR = auto()
    EXECUTION_ERROR = auto()
    SANDBOX_ERROR = auto()


@dataclass
class ToolParameter:
    """Definition of a tool parameter."""
    name: str
    param_type: str  # "string", "integer", "boolean", "array", "object"
    description: str
    required: bool = True
    default: Any = None
    constraints: Dict[str, Any] = field(default_factory=dict)  # min, max, pattern, enum
    sensitive: bool = False  # Contains sensitive data

    def validate(self, value: Any) -> tuple[bool, Optional[str]]:
        """Validate a value against this parameter definition."""
        if value is None:
            if self.required:
                return False, f"Required parameter '{self.name}' is missing"
            return True, None

        # Type validation
        type_validators = {
            "string": lambda v: isinstance(v, str),
            "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
            "boolean": lambda v: isinstance(v, bool),
            "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
            "array": lambda v: isinstance(v, list),
            "object": lambda v: isinstance(v, dict),
        }

        validator = type_validators.get(self.param_type)
        if validator and not validator(value):
            return False, f"Parameter '{self.name}' must be of type {self.param_type}"

        # Constraint validation
        if "min" in self.constraints and value < self.constraints["min"]:
            return False, f"Parameter '{self.name}' must be >= {self.constraints['min']}"

        if "max" in self.constraints and value > self.constraints["max"]:
            return False, f"Parameter '{self.name}' must be <= {self.constraints['max']}"

        if "pattern" in self.constraints and self.param_type == "string":
            import re
            if not re.match(self.constraints["pattern"], value):
                return False, f"Parameter '{self.name}' does not match required pattern"

        if "enum" in self.constraints and value not in self.constraints["enum"]:
            return False, f"Parameter '{self.name}' must be one of: {self.constraints['enum']}"

        if "max_length" in self.constraints and len(value) > self.constraints["max_length"]:
            return False, f"Parameter '{self.name}' exceeds max length {self.constraints['max_length']}"

        return True, None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "type": self.param_type,
            "description": self.description,
            "required": self.required,
            "default": self.default,
            "constraints": self.constraints,
            "sensitive": self.sensitive,
        }


@dataclass
class ToolSchema:
    """Schema definition for a tool."""
    name: str
    description: str
    category: ToolCategory
    version: str = "1.0.0"
    parameters: List[ToolParameter] = field(default_factory=list)
    returns: Dict[str, Any] = field(default_factory=dict)
    examples: List[Dict[str, Any]] = field(default_factory=list)
    risk_level: ToolRiskLevel = ToolRiskLevel.LOW
    requires_confirmation: bool = False
    timeout_seconds: int = 30
    idempotent: bool = True
    tags: List[str] = field(default_factory=list)

    def validate_parameters(self, params: Dict[str, Any]) -> tuple[bool, List[str]]:
        """Validate parameters against schema."""
        errors = []

        # Check required parameters
        for param in self.parameters:
            if param.required and param.name not in params:
                errors.append(f"Required parameter '{param.name}' is missing")
                continue

            value = params.get(param.name, param.default)
            is_valid, error = param.validate(value)
            if not is_valid:
                errors.append(error)

        # Check for unknown parameters
        known_params = {p.name for p in self.parameters}
        for key in params:
            if key not in known_params:
                errors.append(f"Unknown parameter: '{key}'")

        return len(errors) == 0, errors

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "category": self.category.value,
            "version": self.version,
            "parameters": [p.to_dict() for p in self.parameters],
            "returns": self.returns,
            "examples": self.examples,
            "risk_level": self.risk_level.name,
            "requires_confirmation": self.requires_confirmation,
            "timeout_seconds": self.timeout_seconds,
            "idempotent": self.idempotent,
            "tags": self.tags,
        }

    def compute_hash(self) -> str:
        """Compute hash of schema for versioning."""
        import json
        content = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()[:16]


@dataclass
class ToolResult:
    """Result of a tool invocation."""
    result: InvocationResult
    output: Any = None
    error: Optional[str] = None
    execution_time_ms: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    logs: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def is_success(self) -> bool:
        return self.result == InvocationResult.SUCCESS

    @property
    def is_failure(self) -> bool:
        return self.result in (
            InvocationResult.FAILURE,
            InvocationResult.EXECUTION_ERROR,
            InvocationResult.SANDBOX_ERROR,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "result": self.result.name,
            "output": self.output,
            "error": self.error,
            "execution_time_ms": self.execution_time_ms,
            "metadata": self.metadata,
            "logs": self.logs,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class ToolInvocation:
    """Record of a tool invocation."""
    invocation_id: str
    tool_name: str
    parameters: Dict[str, Any]
    user_id: str
    agent_id: Optional[str] = None
    request_id: Optional[str] = None
    result: Optional[ToolResult] = None
    approved_by: Optional[str] = None  # Smith or human
    approval_time: Optional[datetime] = None
    started_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def generate_id() -> str:
        """Generate unique invocation ID."""
        return f"TOOL-{secrets.token_hex(8)}"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "invocation_id": self.invocation_id,
            "tool_name": self.tool_name,
            "parameters": {
                k: "***REDACTED***" if self._is_sensitive_param(k) else v
                for k, v in self.parameters.items()
            },
            "user_id": self.user_id,
            "agent_id": self.agent_id,
            "request_id": self.request_id,
            "result": self.result.to_dict() if self.result else None,
            "approved_by": self.approved_by,
            "approval_time": self.approval_time.isoformat() if self.approval_time else None,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "metadata": self.metadata,
        }

    def _is_sensitive_param(self, param_name: str) -> bool:
        """Check if parameter is sensitive."""
        sensitive_patterns = ["password", "secret", "key", "token", "credential", "auth"]
        param_lower = param_name.lower()
        return any(pattern in param_lower for pattern in sensitive_patterns)


class ToolInterface(ABC):
    """
    Abstract base class for all tools.

    Tools must implement:
    1. get_schema() -> ToolSchema
    2. execute(params) -> ToolResult
    3. validate(params) -> (bool, errors)
    """

    def __init__(self, name: str):
        """Initialize tool."""
        self._name = name
        self._enabled = True
        self._invocation_count = 0
        self._last_invocation: Optional[datetime] = None

    @property
    def name(self) -> str:
        return self._name

    @property
    def enabled(self) -> bool:
        return self._enabled

    def enable(self) -> None:
        self._enabled = True

    def disable(self) -> None:
        self._enabled = False

    @abstractmethod
    def get_schema(self) -> ToolSchema:
        """Get tool schema definition."""
        pass

    @abstractmethod
    def execute(self, parameters: Dict[str, Any]) -> ToolResult:
        """
        Execute the tool with given parameters.

        Args:
            parameters: Tool parameters

        Returns:
            ToolResult with execution result
        """
        pass

    def validate(self, parameters: Dict[str, Any]) -> tuple[bool, List[str]]:
        """
        Validate parameters before execution.

        Default implementation uses schema validation.
        Override for custom validation.
        """
        return self.get_schema().validate_parameters(parameters)

    def pre_execute_hook(self, parameters: Dict[str, Any]) -> Optional[str]:
        """
        Hook called before execution.

        Returns:
            Error message if execution should be blocked, None otherwise
        """
        return None

    def post_execute_hook(self, parameters: Dict[str, Any], result: ToolResult) -> None:
        """Hook called after execution."""
        pass

    def invoke(self, parameters: Dict[str, Any]) -> ToolResult:
        """
        Full invocation lifecycle with hooks.

        Args:
            parameters: Tool parameters

        Returns:
            ToolResult
        """
        if not self._enabled:
            return ToolResult(
                result=InvocationResult.PERMISSION_DENIED,
                error="Tool is disabled",
            )

        # Validate
        is_valid, errors = self.validate(parameters)
        if not is_valid:
            return ToolResult(
                result=InvocationResult.VALIDATION_ERROR,
                error="; ".join(errors),
            )

        # Pre-execute hook
        block_reason = self.pre_execute_hook(parameters)
        if block_reason:
            return ToolResult(
                result=InvocationResult.PERMISSION_DENIED,
                error=block_reason,
            )

        # Execute
        start = datetime.now()
        try:
            result = self.execute(parameters)
            result.execution_time_ms = int((datetime.now() - start).total_seconds() * 1000)
        except Exception as e:
            result = ToolResult(
                result=InvocationResult.EXECUTION_ERROR,
                error=str(e),
                execution_time_ms=int((datetime.now() - start).total_seconds() * 1000),
            )

        # Post-execute hook
        self.post_execute_hook(parameters, result)

        # Update stats
        self._invocation_count += 1
        self._last_invocation = datetime.now()

        return result


class BaseTool(ToolInterface):
    """
    Base tool with common functionality.

    Extend this class for simpler tool implementations.
    """

    def __init__(
        self,
        name: str,
        description: str,
        category: ToolCategory = ToolCategory.UTILITY,
        risk_level: ToolRiskLevel = ToolRiskLevel.LOW,
    ):
        super().__init__(name)
        self._description = description
        self._category = category
        self._risk_level = risk_level
        self._parameters: List[ToolParameter] = []

    def add_parameter(
        self,
        name: str,
        param_type: str,
        description: str,
        required: bool = True,
        default: Any = None,
        **constraints,
    ) -> "BaseTool":
        """Add a parameter definition. Returns self for chaining."""
        self._parameters.append(ToolParameter(
            name=name,
            param_type=param_type,
            description=description,
            required=required,
            default=default,
            constraints=constraints,
        ))
        return self

    def get_schema(self) -> ToolSchema:
        return ToolSchema(
            name=self._name,
            description=self._description,
            category=self._category,
            parameters=self._parameters,
            risk_level=self._risk_level,
        )

    def execute(self, parameters: Dict[str, Any]) -> ToolResult:
        """Must be overridden by subclasses."""
        return ToolResult(
            result=InvocationResult.FAILURE,
            error="execute() not implemented",
        )


class FunctionTool(BaseTool):
    """
    Tool wrapping a Python function.

    Convenient for simple tools that don't need full class implementation.
    """

    def __init__(
        self,
        name: str,
        description: str,
        func: Callable[..., Any],
        category: ToolCategory = ToolCategory.UTILITY,
        risk_level: ToolRiskLevel = ToolRiskLevel.LOW,
    ):
        super().__init__(name, description, category, risk_level)
        self._func = func

    def execute(self, parameters: Dict[str, Any]) -> ToolResult:
        try:
            output = self._func(**parameters)
            return ToolResult(
                result=InvocationResult.SUCCESS,
                output=output,
            )
        except Exception as e:
            return ToolResult(
                result=InvocationResult.EXECUTION_ERROR,
                error=str(e),
            )


def create_function_tool(
    name: str,
    description: str,
    func: Callable[..., Any],
    parameters: Optional[List[ToolParameter]] = None,
    category: ToolCategory = ToolCategory.UTILITY,
    risk_level: ToolRiskLevel = ToolRiskLevel.LOW,
) -> FunctionTool:
    """
    Create a tool from a function.

    Args:
        name: Tool name
        description: Tool description
        func: Function to wrap
        parameters: Parameter definitions
        category: Tool category
        risk_level: Risk level

    Returns:
        FunctionTool instance
    """
    tool = FunctionTool(name, description, func, category, risk_level)
    if parameters:
        tool._parameters = parameters
    return tool
