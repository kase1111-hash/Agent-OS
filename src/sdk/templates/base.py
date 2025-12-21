"""
Base Agent Template

Provides the foundational template for all Agent OS agents.
Simplifies agent creation with sensible defaults and helper methods.
"""

import logging
from abc import abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Set, Union

from src.agents.interface import (
    AgentInterface,
    BaseAgent,
    AgentState,
    AgentCapabilities,
    CapabilityType,
    RequestValidationResult,
    AgentMetrics,
)
from src.messaging.models import FlowRequest, FlowResponse, MessageStatus


logger = logging.getLogger(__name__)


@dataclass
class AgentConfig:
    """Configuration for an agent template."""
    name: str
    version: str = "1.0.0"
    description: str = ""
    capabilities: Set[CapabilityType] = field(default_factory=set)
    supported_intents: List[str] = field(default_factory=list)
    model: Optional[str] = None
    context_window: int = 4096
    max_output_tokens: int = 2048
    requires_constitution: bool = True
    requires_memory: bool = False
    can_escalate: bool = True
    timeout_seconds: int = 30
    max_retries: int = 3
    metadata: Dict[str, Any] = field(default_factory=dict)


class AgentTemplate(BaseAgent):
    """
    Enhanced base agent template with additional functionality.

    Provides:
    - Simplified configuration
    - Request/response logging
    - Error handling with retries
    - Pre/post processing hooks
    - Metrics collection
    """

    def __init__(self, config: AgentConfig):
        """
        Initialize agent from config.

        Args:
            config: Agent configuration
        """
        super().__init__(
            name=config.name,
            description=config.description,
            version=config.version,
            capabilities=config.capabilities,
            supported_intents=config.supported_intents,
        )
        self.agent_config = config
        self._pre_process_hooks: List[Callable[[FlowRequest], FlowRequest]] = []
        self._post_process_hooks: List[Callable[[FlowResponse], FlowResponse]] = []
        self._error_handlers: List[Callable[[Exception, FlowRequest], Optional[FlowResponse]]] = []

    def get_capabilities(self) -> AgentCapabilities:
        """Get agent capabilities from config."""
        return AgentCapabilities(
            name=self.agent_config.name,
            version=self.agent_config.version,
            description=self.agent_config.description,
            capabilities=self.agent_config.capabilities,
            supported_intents=self.agent_config.supported_intents,
            model=self.agent_config.model,
            context_window=self.agent_config.context_window,
            max_output_tokens=self.agent_config.max_output_tokens,
            requires_constitution=self.agent_config.requires_constitution,
            requires_memory=self.agent_config.requires_memory,
            can_escalate=self.agent_config.can_escalate,
            metadata=self.agent_config.metadata,
        )

    def add_pre_process_hook(
        self,
        hook: Callable[[FlowRequest], FlowRequest],
    ) -> "AgentTemplate":
        """
        Add a pre-processing hook.

        The hook receives the request and can modify it before processing.

        Args:
            hook: Hook function

        Returns:
            Self for chaining
        """
        self._pre_process_hooks.append(hook)
        return self

    def add_post_process_hook(
        self,
        hook: Callable[[FlowResponse], FlowResponse],
    ) -> "AgentTemplate":
        """
        Add a post-processing hook.

        The hook receives the response and can modify it after processing.

        Args:
            hook: Hook function

        Returns:
            Self for chaining
        """
        self._post_process_hooks.append(hook)
        return self

    def add_error_handler(
        self,
        handler: Callable[[Exception, FlowRequest], Optional[FlowResponse]],
    ) -> "AgentTemplate":
        """
        Add an error handler.

        The handler receives the exception and request, and can return
        a response to use instead of re-raising the error.

        Args:
            handler: Handler function

        Returns:
            Self for chaining
        """
        self._error_handlers.append(handler)
        return self

    def process(self, request: FlowRequest) -> FlowResponse:
        """
        Process a request with hooks and error handling.

        Override handle_request() instead of this method.
        """
        # Apply pre-process hooks
        for hook in self._pre_process_hooks:
            try:
                request = hook(request)
            except Exception as e:
                logger.warning(f"Pre-process hook error: {e}")

        # Process with error handling and retries
        last_error = None
        for attempt in range(self.agent_config.max_retries):
            try:
                response = self.handle_request(request)

                # Apply post-process hooks
                for hook in self._post_process_hooks:
                    try:
                        response = hook(response)
                    except Exception as e:
                        logger.warning(f"Post-process hook error: {e}")

                return response

            except Exception as e:
                last_error = e
                logger.warning(
                    f"Agent {self.name} attempt {attempt + 1}/{self.agent_config.max_retries} "
                    f"failed: {e}"
                )

                # Try error handlers
                for handler in self._error_handlers:
                    try:
                        response = handler(e, request)
                        if response:
                            return response
                    except Exception as he:
                        logger.warning(f"Error handler failed: {he}")

        # All retries exhausted
        logger.error(f"Agent {self.name} failed after {self.agent_config.max_retries} attempts")
        return request.create_response(
            source=self.name,
            status=MessageStatus.ERROR,
            output="",
            errors=[str(last_error)],
        )

    @abstractmethod
    def handle_request(self, request: FlowRequest) -> FlowResponse:
        """
        Handle a request. Override this method in subclasses.

        Args:
            request: The request to handle

        Returns:
            FlowResponse
        """
        pass


class SimpleAgent(AgentTemplate):
    """
    Simple agent that uses a handler function.

    Convenient for quick agent creation without subclassing.
    """

    def __init__(
        self,
        config: AgentConfig,
        handler: Callable[[FlowRequest], Union[str, Dict[str, Any], FlowResponse]],
    ):
        """
        Initialize simple agent.

        Args:
            config: Agent configuration
            handler: Handler function that processes requests
        """
        super().__init__(config)
        self._handler = handler

    def handle_request(self, request: FlowRequest) -> FlowResponse:
        """Process request using handler function."""
        result = self._handler(request)

        if isinstance(result, FlowResponse):
            return result

        if isinstance(result, dict):
            return request.create_response(
                source=self.name,
                status=MessageStatus.SUCCESS,
                output=result.get("output", ""),
                reasoning=result.get("reasoning"),
            )

        # Assume string output
        return request.create_response(
            source=self.name,
            status=MessageStatus.SUCCESS,
            output=str(result),
        )


def create_simple_agent(
    name: str,
    handler: Callable[[FlowRequest], Union[str, Dict[str, Any], FlowResponse]],
    description: str = "",
    capabilities: Optional[Set[CapabilityType]] = None,
    intents: Optional[List[str]] = None,
    **kwargs,
) -> SimpleAgent:
    """
    Create a simple agent from a handler function.

    Example:
        def my_handler(request):
            return f"Hello! You said: {request.content.prompt}"

        agent = create_simple_agent(
            name="greeter",
            handler=my_handler,
            description="A friendly greeter agent",
        )

    Args:
        name: Agent name
        handler: Handler function
        description: Agent description
        capabilities: Agent capabilities
        intents: Supported intent patterns
        **kwargs: Additional config options

    Returns:
        SimpleAgent instance
    """
    config = AgentConfig(
        name=name,
        description=description,
        capabilities=capabilities or set(),
        supported_intents=intents or [],
        **kwargs,
    )
    return SimpleAgent(config, handler)
