"""
Agent OS Agent Interface

Abstract base class defining the mandatory interface for all Agent OS agents.
Every agent in the system MUST implement this interface.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set

from src.core.models import Rule
from src.messaging.models import (
    FlowRequest,
    FlowResponse,
    MessageStatus,
)

logger = logging.getLogger(__name__)


# Import constitution loader (lazy to avoid circular imports)
def _get_constitution_loader():
    from src.agents.constitution_loader import get_constitution_loader

    return get_constitution_loader()


class AgentState(Enum):
    """Agent lifecycle states."""

    UNINITIALIZED = auto()
    INITIALIZING = auto()
    READY = auto()
    PROCESSING = auto()
    SHUTTING_DOWN = auto()
    SHUTDOWN = auto()
    ERROR = auto()


class CapabilityType(Enum):
    """Types of capabilities an agent can have."""

    REASONING = "reasoning"  # Complex reasoning and analysis
    GENERATION = "generation"  # Content generation
    RETRIEVAL = "retrieval"  # Memory/document retrieval
    VALIDATION = "validation"  # Input validation and filtering
    ROUTING = "routing"  # Intent routing and orchestration
    MEMORY = "memory"  # Persistent memory management
    TOOL_USE = "tool_use"  # Tool/function calling
    CREATIVE = "creative"  # Creative content generation
    CODE = "code"  # Code generation/analysis


@dataclass
class AgentCapabilities:
    """Agent capabilities declaration."""

    name: str  # Agent name (e.g., "sage", "whisper")
    version: str  # Agent version
    description: str  # What this agent does
    capabilities: Set[CapabilityType] = field(default_factory=set)
    supported_intents: List[str] = field(default_factory=list)  # Intent patterns handled
    model: Optional[str] = None  # Model used (if any)
    context_window: int = 4096  # Max context tokens
    max_output_tokens: int = 2048  # Max output tokens
    requires_constitution: bool = True  # Needs constitutional enforcement
    requires_memory: bool = False  # Uses memory system
    can_escalate: bool = True  # Can escalate to human
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert capabilities to dictionary."""
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "capabilities": [c.value for c in self.capabilities],
            "supported_intents": self.supported_intents,
            "model": self.model,
            "context_window": self.context_window,
            "max_output_tokens": self.max_output_tokens,
            "requires_constitution": self.requires_constitution,
            "requires_memory": self.requires_memory,
            "can_escalate": self.can_escalate,
            "metadata": self.metadata,
        }


@dataclass
class RequestValidationResult:
    """Result of validating an incoming request."""

    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    modified_request: Optional[FlowRequest] = None  # If request was sanitized
    applicable_rules: List[Rule] = field(default_factory=list)
    requires_escalation: bool = False
    escalation_reason: Optional[str] = None

    def add_error(self, message: str) -> None:
        self.errors.append(message)
        self.is_valid = False

    def add_warning(self, message: str) -> None:
        self.warnings.append(message)


@dataclass
class AgentMetrics:
    """Runtime metrics for an agent."""

    requests_processed: int = 0
    requests_succeeded: int = 0
    requests_failed: int = 0
    requests_refused: int = 0
    total_tokens_consumed: int = 0
    total_inference_time_ms: int = 0
    average_response_time_ms: float = 0.0
    last_request_time: Optional[datetime] = None
    uptime_seconds: float = 0.0
    errors: List[Dict[str, Any]] = field(default_factory=list)


class AgentInterface(ABC):
    """
    Abstract base class for all Agent OS agents.

    Every agent MUST implement these 5 mandatory methods:
    1. initialize(config) → bool
    2. validate_request(request) → RequestValidationResult
    3. process(request) → FlowResponse
    4. get_capabilities() → AgentCapabilities
    5. shutdown() → bool

    The interface also provides common functionality for:
    - Constitutional boundary enforcement
    - Request/response logging
    - Metrics collection
    - Error handling
    """

    def __init__(self, name: str):
        """
        Initialize base agent.

        Args:
            name: Agent name identifier
        """
        self._name = name
        self._state = AgentState.UNINITIALIZED
        self._config: Dict[str, Any] = {}
        self._metrics = AgentMetrics()
        self._start_time: Optional[datetime] = None
        self._capabilities: Optional[AgentCapabilities] = None
        self._constitutional_rules: List[Rule] = []
        self._callbacks: Dict[str, List[Callable]] = {
            "on_request": [],
            "on_response": [],
            "on_error": [],
            "on_shutdown": [],
        }

    @property
    def name(self) -> str:
        """Get agent name."""
        return self._name

    @property
    def state(self) -> AgentState:
        """Get current agent state."""
        return self._state

    @property
    def is_ready(self) -> bool:
        """Check if agent is ready to process requests."""
        return self._state == AgentState.READY

    @property
    def metrics(self) -> AgentMetrics:
        """Get agent metrics."""
        if self._start_time:
            self._metrics.uptime_seconds = (datetime.now() - self._start_time).total_seconds()
        return self._metrics

    # ==========================================================================
    # MANDATORY METHODS (must be implemented by all agents)
    # ==========================================================================

    @abstractmethod
    def initialize(self, config: Dict[str, Any]) -> bool:
        """
        Initialize the agent with configuration.

        This method should:
        - Load model/resources
        - Set up connections
        - Load constitutional rules
        - Prepare for request processing

        Args:
            config: Agent configuration dictionary

        Returns:
            True if initialization successful, False otherwise
        """
        pass

    @abstractmethod
    def validate_request(self, request: FlowRequest) -> RequestValidationResult:
        """
        Validate an incoming request against constitutional rules.

        This method should:
        - Check request format/content
        - Validate against constitutional boundaries
        - Check if agent can handle this request
        - Determine if escalation is needed

        Args:
            request: Incoming FlowRequest

        Returns:
            RequestValidationResult with validation status
        """
        pass

    @abstractmethod
    def process(self, request: FlowRequest) -> FlowResponse:
        """
        Process a validated request and generate a response.

        This method should:
        - Generate response using agent's capabilities
        - Handle errors gracefully
        - Include response metadata
        - Log processing for audit

        Args:
            request: Validated FlowRequest

        Returns:
            FlowResponse with agent's response
        """
        pass

    @abstractmethod
    def get_capabilities(self) -> AgentCapabilities:
        """
        Get agent capabilities declaration.

        Returns:
            AgentCapabilities describing what this agent can do
        """
        pass

    @abstractmethod
    def shutdown(self) -> bool:
        """
        Gracefully shutdown the agent.

        This method should:
        - Complete any pending requests
        - Release resources
        - Save state if needed
        - Notify callbacks

        Returns:
            True if shutdown successful, False otherwise
        """
        pass

    # ==========================================================================
    # COMMON IMPLEMENTATION (inherited by all agents)
    # ==========================================================================

    def handle_request(self, request: FlowRequest) -> FlowResponse:
        """
        Main entry point for request handling with full lifecycle management.

        This wraps validate_request and process with:
        - State checking
        - Metrics collection
        - Error handling
        - Callback invocation

        Args:
            request: Incoming FlowRequest

        Returns:
            FlowResponse
        """
        start_time = datetime.now()

        # Check state
        if not self.is_ready:
            return self._create_error_response(
                request, f"Agent {self.name} is not ready (state: {self.state.name})"
            )

        try:
            self._state = AgentState.PROCESSING

            # Invoke pre-request callbacks
            for callback in self._callbacks["on_request"]:
                try:
                    callback(request)
                except Exception as e:
                    logger.warning(f"Request callback error: {e}")

            # Validate request
            validation = self.validate_request(request)

            if not validation.is_valid:
                self._metrics.requests_refused += 1
                return self._create_refused_response(
                    request, validation.errors, validation.escalation_reason
                )

            if validation.requires_escalation:
                return self._create_escalation_response(
                    request, validation.escalation_reason or "Escalation required"
                )

            # Use modified request if validation sanitized it
            effective_request = validation.modified_request or request

            # Process request
            response = self.process(effective_request)

            # Update metrics
            elapsed_ms = int((datetime.now() - start_time).total_seconds() * 1000)
            self._update_metrics(response, elapsed_ms)

            # Invoke post-response callbacks
            for callback in self._callbacks["on_response"]:
                try:
                    callback(request, response)
                except Exception as e:
                    logger.warning(f"Response callback error: {e}")

            return response

        except Exception as e:
            self._metrics.requests_failed += 1
            self._metrics.errors.append(
                {
                    "timestamp": datetime.now().isoformat(),
                    "request_id": str(request.request_id),
                    "error": str(e),
                }
            )

            # Invoke error callbacks
            for callback in self._callbacks["on_error"]:
                try:
                    callback(request, e)
                except Exception as cb_err:
                    logger.warning(f"Error callback error: {cb_err}")

            logger.exception(f"Agent {self.name} error processing request")
            return self._create_error_response(request, str(e))

        finally:
            if self._state == AgentState.PROCESSING:
                self._state = AgentState.READY

    def set_constitutional_rules(self, rules: List[Rule]) -> None:
        """
        Set constitutional rules for this agent.

        Args:
            rules: List of applicable constitutional rules
        """
        self._constitutional_rules = rules
        logger.info(f"Agent {self.name}: loaded {len(rules)} constitutional rules")

    def register_callback(self, event: str, callback: Callable) -> None:
        """
        Register a callback for agent events.

        Args:
            event: Event name (on_request, on_response, on_error, on_shutdown)
            callback: Callback function
        """
        if event in self._callbacks:
            self._callbacks[event].append(callback)
        else:
            raise ValueError(f"Unknown event: {event}")

    def get_applicable_rules(self, request: FlowRequest) -> List[Rule]:
        """
        Get constitutional rules applicable to a request.

        Args:
            request: The request to check

        Returns:
            List of applicable rules
        """
        applicable = []
        content_lower = request.content.prompt.lower()
        intent_lower = request.intent.lower()

        for rule in self._constitutional_rules:
            # Check if rule keywords match content or intent
            for keyword in rule.keywords:
                if keyword in content_lower or keyword in intent_lower:
                    applicable.append(rule)
                    break

        return applicable

    def _update_metrics(self, response: FlowResponse, elapsed_ms: int) -> None:
        """Update metrics after processing a request."""
        self._metrics.requests_processed += 1
        self._metrics.last_request_time = datetime.now()
        self._metrics.total_inference_time_ms += elapsed_ms

        if response.is_success():
            self._metrics.requests_succeeded += 1
        elif response.was_refused():
            self._metrics.requests_refused += 1
        elif response.is_error():
            self._metrics.requests_failed += 1

        if response.content.metadata.tokens_consumed:
            self._metrics.total_tokens_consumed += response.content.metadata.tokens_consumed

        # Update average
        if self._metrics.requests_processed > 0:
            self._metrics.average_response_time_ms = (
                self._metrics.total_inference_time_ms / self._metrics.requests_processed
            )

    def _create_error_response(self, request: FlowRequest, error: str) -> FlowResponse:
        """Create an error response."""
        return request.create_response(
            source=self.name,
            status=MessageStatus.ERROR,
            output="",
            errors=[error],
        )

    def _create_refused_response(
        self, request: FlowRequest, errors: List[str], reason: Optional[str] = None
    ) -> FlowResponse:
        """Create a refused response (constitutional violation)."""
        return request.create_response(
            source=self.name,
            status=MessageStatus.REFUSED,
            output="Request refused due to constitutional constraints.",
            reasoning=reason or "; ".join(errors),
        )

    def _create_escalation_response(self, request: FlowRequest, reason: str) -> FlowResponse:
        """Create a response requiring human escalation."""
        response = request.create_response(
            source=self.name,
            status=MessageStatus.PARTIAL,
            output="This request requires human approval.",
            reasoning=reason,
        )
        response.next_actions.append(
            {
                "action": "escalate_to_human",
                "reason": reason,
                "request_id": str(request.request_id),
            }
        )
        return response

    def _do_initialize(self, config: Dict[str, Any]) -> bool:
        """
        Common initialization logic.

        Call this at the start of your initialize() implementation.
        """
        self._state = AgentState.INITIALIZING
        self._config = config
        self._start_time = datetime.now()
        self._capabilities = self.get_capabilities()

        logger.info(f"Agent {self.name} initializing with config: {list(config.keys())}")
        return True

    def _do_shutdown(self) -> bool:
        """
        Common shutdown logic.

        Call this at the end of your shutdown() implementation.
        """
        self._state = AgentState.SHUTTING_DOWN

        # Invoke shutdown callbacks
        for callback in self._callbacks["on_shutdown"]:
            try:
                callback()
            except Exception as e:
                logger.warning(f"Shutdown callback error: {e}")

        self._state = AgentState.SHUTDOWN
        logger.info(f"Agent {self.name} shutdown complete")
        return True


class BaseAgent(AgentInterface):
    """
    Base agent with default implementations.

    Extend this class for simpler agent implementations that
    only need to override specific methods.
    """

    def __init__(
        self,
        name: str,
        description: str = "",
        version: str = "0.1.0",
        capabilities: Optional[Set[CapabilityType]] = None,
        supported_intents: Optional[List[str]] = None,
    ):
        super().__init__(name)
        self._description = description
        self._version = version
        self._capability_types = capabilities or set()
        self._supported_intents = supported_intents or []

    def initialize(self, config: Dict[str, Any]) -> bool:
        """Default initialization."""
        self._do_initialize(config)
        self._state = AgentState.READY
        return True

    def validate_request(self, request: FlowRequest) -> RequestValidationResult:
        """Default validation - check constitutional rules."""
        result = RequestValidationResult(is_valid=True)

        # Get applicable rules
        applicable_rules = self.get_applicable_rules(request)
        result.applicable_rules = applicable_rules

        # Check for prohibitions
        from src.core.models import RuleType

        for rule in applicable_rules:
            if rule.rule_type == RuleType.PROHIBITION:
                result.add_error(f"Request may violate prohibition: {rule.content[:100]}")

        # Check for escalation requirements
        for rule in applicable_rules:
            if rule.rule_type == RuleType.ESCALATION:
                result.requires_escalation = True
                result.escalation_reason = rule.content

        return result

    def process(self, request: FlowRequest) -> FlowResponse:
        """Default processing - must be overridden for real agents."""
        return request.create_response(
            source=self.name,
            status=MessageStatus.ERROR,
            output="",
            errors=["Agent.process() not implemented"],
        )

    def get_capabilities(self) -> AgentCapabilities:
        """Return agent capabilities."""
        return AgentCapabilities(
            name=self.name,
            version=self._version,
            description=self._description,
            capabilities=self._capability_types,
            supported_intents=self._supported_intents,
        )

    def shutdown(self) -> bool:
        """Default shutdown."""
        return self._do_shutdown()

    # ==========================================================================
    # CONSTITUTIONAL CONTEXT HELPERS
    # ==========================================================================

    def load_constitutional_context(
        self,
        include_supreme: bool = True,
    ) -> str:
        """
        Load constitutional context for this agent.

        This loads:
        - The supreme CONSTITUTION.md (if include_supreme=True)
        - This agent's specific constitution.md

        Other agents' constitutions are NOT loaded.

        Args:
            include_supreme: Whether to include the supreme constitution

        Returns:
            Formatted constitutional context string
        """
        try:
            loader = _get_constitution_loader()
            context = loader.load_for_agent(self.name, include_supreme)
            return context.combined_prompt
        except Exception as e:
            logger.warning(f"Could not load constitutional context for {self.name}: {e}")
            return ""

    def build_system_prompt_with_constitution(
        self,
        base_prompt: str,
        include_supreme: bool = True,
    ) -> str:
        """
        Build a complete system prompt with constitutional context.

        This prepends the constitutional rules to the base prompt,
        ensuring the LLM has access to governance rules in its context.

        Args:
            base_prompt: The agent's base system prompt (from prompt.md)
            include_supreme: Whether to include supreme constitution

        Returns:
            Combined system prompt with constitutional context
        """
        try:
            loader = _get_constitution_loader()
            return loader.get_system_prompt_with_constitution(
                agent_name=self.name,
                base_prompt=base_prompt,
                include_supreme=include_supreme,
            )
        except Exception as e:
            logger.warning(f"Could not build constitutional prompt for {self.name}: {e}")
            return base_prompt

    def _load_prompt_file(self, prompt_filename: str = "prompt.md") -> str:
        """
        Load a prompt file from the agent's directory.

        Args:
            prompt_filename: Name of the prompt file

        Returns:
            Content of the prompt file, or empty string if not found
        """
        try:
            # Try standard location: agents/<name>/prompt.md
            loader = _get_constitution_loader()
            prompt_path = loader.project_root / "agents" / self.name / prompt_filename

            if prompt_path.exists():
                return prompt_path.read_text(encoding="utf-8")

            logger.debug(f"Prompt file not found at {prompt_path}")
            return ""
        except Exception as e:
            logger.warning(f"Could not load prompt file for {self.name}: {e}")
            return ""

    def get_full_system_prompt(
        self,
        include_constitution: bool = True,
        include_supreme: bool = True,
        fallback_prompt: str = "",
    ) -> str:
        """
        Get the complete system prompt with constitutional context.

        This is the recommended method for agents to get their system prompt.
        It:
        1. Loads the agent's prompt.md file
        2. Prepends constitutional context (supreme + agent-specific)
        3. Returns the combined prompt

        Args:
            include_constitution: Whether to include constitutional context
            include_supreme: Whether to include the supreme constitution
            fallback_prompt: Fallback prompt if prompt.md not found

        Returns:
            Complete system prompt ready for LLM use
        """
        # Load base prompt
        base_prompt = self._load_prompt_file() or fallback_prompt

        if not base_prompt:
            logger.warning(f"No system prompt found for agent {self.name}")
            base_prompt = f"You are {self.name}, an Agent-OS agent. {self._description}"

        # Add constitutional context if requested
        if include_constitution:
            return self.build_system_prompt_with_constitution(
                base_prompt=base_prompt,
                include_supreme=include_supreme,
            )

        return base_prompt
