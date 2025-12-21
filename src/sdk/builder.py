"""
Agent Builder

Provides a fluent API for building agents with a declarative syntax.
Simplifies agent creation without subclassing.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Type, Union

from src.agents.interface import (
    BaseAgent,
    AgentCapabilities,
    CapabilityType,
    RequestValidationResult,
    AgentState,
)
from src.messaging.models import FlowRequest, FlowResponse, MessageStatus
from src.core.models import Rule

from .templates.base import AgentTemplate, AgentConfig


logger = logging.getLogger(__name__)


class AgentBuilder:
    """
    Fluent builder for creating agents.

    Example:
        agent = (
            AgentBuilder("my_agent")
            .with_description("Does amazing things")
            .with_capability(CapabilityType.REASONING)
            .with_intent("analyze.*")
            .with_handler(my_handler_function)
            .with_pre_hook(log_request)
            .with_post_hook(format_response)
            .with_error_handler(handle_errors)
            .build()
        )
    """

    def __init__(self, name: str):
        """
        Start building an agent.

        Args:
            name: Agent name
        """
        self._name = name
        self._version = "1.0.0"
        self._description = ""
        self._capabilities: Set[CapabilityType] = set()
        self._intents: List[str] = []
        self._model: Optional[str] = None
        self._context_window = 4096
        self._max_output_tokens = 2048
        self._requires_constitution = True
        self._requires_memory = False
        self._can_escalate = True
        self._timeout = 30
        self._max_retries = 3
        self._metadata: Dict[str, Any] = {}

        self._handler: Optional[Callable] = None
        self._validator: Optional[Callable] = None
        self._initializer: Optional[Callable] = None
        self._shutdown_handler: Optional[Callable] = None

        self._pre_hooks: List[Callable] = []
        self._post_hooks: List[Callable] = []
        self._error_handlers: List[Callable] = []
        self._rules: List[Rule] = []

    def with_version(self, version: str) -> "AgentBuilder":
        """Set agent version."""
        self._version = version
        return self

    def with_description(self, description: str) -> "AgentBuilder":
        """Set agent description."""
        self._description = description
        return self

    def with_capability(self, capability: CapabilityType) -> "AgentBuilder":
        """Add a capability."""
        self._capabilities.add(capability)
        return self

    def with_capabilities(self, *capabilities: CapabilityType) -> "AgentBuilder":
        """Add multiple capabilities."""
        self._capabilities.update(capabilities)
        return self

    def with_intent(self, pattern: str) -> "AgentBuilder":
        """Add a supported intent pattern."""
        self._intents.append(pattern)
        return self

    def with_intents(self, *patterns: str) -> "AgentBuilder":
        """Add multiple intent patterns."""
        self._intents.extend(patterns)
        return self

    def with_model(self, model: str) -> "AgentBuilder":
        """Set the model to use."""
        self._model = model
        return self

    def with_context_window(self, tokens: int) -> "AgentBuilder":
        """Set context window size."""
        self._context_window = tokens
        return self

    def with_max_output(self, tokens: int) -> "AgentBuilder":
        """Set max output tokens."""
        self._max_output_tokens = tokens
        return self

    def requires_constitution(self, required: bool = True) -> "AgentBuilder":
        """Set whether constitutional enforcement is required."""
        self._requires_constitution = required
        return self

    def requires_memory(self, required: bool = True) -> "AgentBuilder":
        """Set whether memory system is required."""
        self._requires_memory = required
        return self

    def can_escalate(self, can: bool = True) -> "AgentBuilder":
        """Set whether agent can escalate to human."""
        self._can_escalate = can
        return self

    def with_timeout(self, seconds: int) -> "AgentBuilder":
        """Set request timeout."""
        self._timeout = seconds
        return self

    def with_retries(self, count: int) -> "AgentBuilder":
        """Set max retry count."""
        self._max_retries = count
        return self

    def with_metadata(self, key: str, value: Any) -> "AgentBuilder":
        """Add metadata."""
        self._metadata[key] = value
        return self

    def with_handler(
        self,
        handler: Callable[[FlowRequest], Union[str, Dict, FlowResponse]],
    ) -> "AgentBuilder":
        """
        Set the request handler function.

        Handler receives FlowRequest and should return:
        - str: Used as output
        - dict: Must contain 'output' key
        - FlowResponse: Used directly
        """
        self._handler = handler
        return self

    def with_validator(
        self,
        validator: Callable[[FlowRequest], RequestValidationResult],
    ) -> "AgentBuilder":
        """Set custom request validator."""
        self._validator = validator
        return self

    def with_initializer(
        self,
        initializer: Callable[[Dict[str, Any]], bool],
    ) -> "AgentBuilder":
        """Set initialization function."""
        self._initializer = initializer
        return self

    def with_shutdown(
        self,
        handler: Callable[[], bool],
    ) -> "AgentBuilder":
        """Set shutdown handler."""
        self._shutdown_handler = handler
        return self

    def with_pre_hook(
        self,
        hook: Callable[[FlowRequest], FlowRequest],
    ) -> "AgentBuilder":
        """Add pre-processing hook."""
        self._pre_hooks.append(hook)
        return self

    def with_post_hook(
        self,
        hook: Callable[[FlowResponse], FlowResponse],
    ) -> "AgentBuilder":
        """Add post-processing hook."""
        self._post_hooks.append(hook)
        return self

    def with_error_handler(
        self,
        handler: Callable[[Exception, FlowRequest], Optional[FlowResponse]],
    ) -> "AgentBuilder":
        """Add error handler."""
        self._error_handlers.append(handler)
        return self

    def with_rule(self, rule: Rule) -> "AgentBuilder":
        """Add a constitutional rule."""
        self._rules.append(rule)
        return self

    def build(self) -> BaseAgent:
        """
        Build the agent.

        Returns:
            Configured agent instance
        """
        if not self._handler:
            raise ValueError("Handler is required. Use .with_handler()")

        # Create config
        config = AgentConfig(
            name=self._name,
            version=self._version,
            description=self._description,
            capabilities=self._capabilities,
            supported_intents=self._intents,
            model=self._model,
            context_window=self._context_window,
            max_output_tokens=self._max_output_tokens,
            requires_constitution=self._requires_constitution,
            requires_memory=self._requires_memory,
            can_escalate=self._can_escalate,
            timeout_seconds=self._timeout,
            max_retries=self._max_retries,
            metadata=self._metadata,
        )

        # Build agent class
        builder = self

        class BuiltAgent(AgentTemplate):
            def __init__(self):
                super().__init__(config)

                # Add hooks
                for hook in builder._pre_hooks:
                    self.add_pre_process_hook(hook)
                for hook in builder._post_hooks:
                    self.add_post_process_hook(hook)
                for handler in builder._error_handlers:
                    self.add_error_handler(handler)

                # Set rules
                if builder._rules:
                    self.set_constitutional_rules(builder._rules)

            def initialize(self, config: Dict[str, Any]) -> bool:
                result = super().initialize(config)
                if builder._initializer:
                    return builder._initializer(config)
                return result

            def validate_request(self, request: FlowRequest) -> RequestValidationResult:
                if builder._validator:
                    return builder._validator(request)
                return super().validate_request(request)

            def handle_request(self, request: FlowRequest) -> FlowResponse:
                result = builder._handler(request)

                if isinstance(result, FlowResponse):
                    return result

                if isinstance(result, dict):
                    return request.create_response(
                        source=self.name,
                        status=MessageStatus.SUCCESS,
                        output=result.get("output", ""),
                        reasoning=result.get("reasoning"),
                    )

                return request.create_response(
                    source=self.name,
                    status=MessageStatus.SUCCESS,
                    output=str(result),
                )

            def shutdown(self) -> bool:
                if builder._shutdown_handler:
                    return builder._shutdown_handler()
                return super().shutdown()

        return BuiltAgent()


def agent(name: str) -> AgentBuilder:
    """
    Start building an agent with fluent API.

    Convenience function for AgentBuilder(name).

    Example:
        my_agent = (
            agent("analyzer")
            .with_description("Analyzes text")
            .with_capability(CapabilityType.REASONING)
            .with_handler(analyze_text)
            .build()
        )

    Args:
        name: Agent name

    Returns:
        AgentBuilder instance
    """
    return AgentBuilder(name)
