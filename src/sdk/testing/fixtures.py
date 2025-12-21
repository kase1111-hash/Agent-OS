"""
Test Fixtures for Agent Testing

Provides factory functions and builders for creating test objects.
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Set

from src.messaging.models import (
    FlowRequest,
    FlowResponse,
    RequestContent,
    RequestMetadata,
    MessageStatus,
)
from src.agents.interface import BaseAgent, AgentCapabilities, CapabilityType


@dataclass
class TestContext:
    """Context for a test scenario."""
    user_id: str = "test_user"
    session_id: str = field(default_factory=lambda: f"test-session-{uuid.uuid4().hex[:8]}")
    conversation_id: str = field(default_factory=lambda: f"test-conv-{uuid.uuid4().hex[:8]}")
    agent_id: str = "test_agent"
    metadata: Dict[str, Any] = field(default_factory=dict)


def create_test_context(
    user_id: str = "test_user",
    session_id: Optional[str] = None,
    **kwargs,
) -> TestContext:
    """
    Create a test context.

    Args:
        user_id: User ID
        session_id: Session ID (auto-generated if not provided)
        **kwargs: Additional context fields

    Returns:
        TestContext instance
    """
    return TestContext(
        user_id=user_id,
        session_id=session_id or f"test-session-{uuid.uuid4().hex[:8]}",
        **kwargs,
    )


def create_test_request(
    prompt: str,
    intent: str = "general",
    source: str = "test_user",
    destination: str = "test_agent",
    context: Optional[List[Dict[str, Any]]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> FlowRequest:
    """
    Create a test request.

    Args:
        prompt: User prompt
        intent: Intent classification
        source: Source of request
        destination: Destination agent
        context: Additional context (list of context items)
        metadata: Additional metadata

    Returns:
        FlowRequest instance
    """
    request_metadata = RequestMetadata()
    if metadata:
        for key, value in metadata.items():
            if hasattr(request_metadata, key):
                setattr(request_metadata, key, value)

    return FlowRequest(
        request_id=uuid.uuid4(),
        source=source,
        destination=destination,
        intent=intent,
        content=RequestContent(
            prompt=prompt,
            context=context or [],
            metadata=request_metadata,
        ),
    )


def create_test_response(
    request: FlowRequest,
    output: str,
    status: MessageStatus = MessageStatus.SUCCESS,
    source: str = "test_agent",
    reasoning: Optional[str] = None,
    errors: Optional[List[str]] = None,
) -> FlowResponse:
    """
    Create a test response.

    Args:
        request: Original request
        output: Response output
        status: Response status
        source: Responding agent
        reasoning: Reasoning explanation
        errors: Error messages

    Returns:
        FlowResponse instance
    """
    kwargs = {}
    if reasoning is not None:
        kwargs["reasoning"] = reasoning
    if errors is not None:
        kwargs["errors"] = errors

    return request.create_response(
        source=source,
        status=status,
        output=output,
        **kwargs,
    )


class TestRequestBuilder:
    """
    Builder for creating test requests with fluent API.

    Example:
        request = (
            TestRequestBuilder()
            .with_prompt("Hello, how are you?")
            .with_intent("greeting")
            .from_user("alice")
            .with_context_item("role", "Previous conversation...")
            .build()
        )
    """

    def __init__(self):
        self._prompt = ""
        self._intent = "general"
        self._source = "test_user"
        self._destination = "test_agent"
        self._context: List[Dict[str, Any]] = []
        self._metadata: Dict[str, Any] = {}
        self._history: List[Dict[str, str]] = []

    def with_prompt(self, prompt: str) -> "TestRequestBuilder":
        """Set the prompt."""
        self._prompt = prompt
        return self

    def with_intent(self, intent: str) -> "TestRequestBuilder":
        """Set the intent."""
        self._intent = intent
        return self

    def from_user(self, user_id: str) -> "TestRequestBuilder":
        """Set the source user."""
        self._source = user_id
        return self

    def to_agent(self, agent_id: str) -> "TestRequestBuilder":
        """Set the destination agent."""
        self._destination = agent_id
        return self

    def with_context_item(self, role: str, content: str) -> "TestRequestBuilder":
        """Add a context item."""
        self._context.append({"role": role, "content": content})
        return self

    def with_metadata(self, key: str, value: Any) -> "TestRequestBuilder":
        """Add metadata."""
        self._metadata[key] = value
        return self

    def with_history(self, role: str, content: str) -> "TestRequestBuilder":
        """Add conversation history entry."""
        self._history.append({"role": role, "content": content})
        return self

    def build(self) -> FlowRequest:
        """Build the request."""
        metadata = dict(self._metadata)
        if self._history:
            metadata["history"] = self._history

        return create_test_request(
            prompt=self._prompt,
            intent=self._intent,
            source=self._source,
            destination=self._destination,
            context=self._context,
            metadata=metadata,
        )


class AgentTestCase:
    """
    Base class for agent test cases.

    Provides setup/teardown and helper methods for testing agents.

    Example:
        class TestMyAgent(AgentTestCase):
            def setup_agent(self):
                return MyAgent()

            def test_greeting(self):
                request = self.create_request("Hello!")
                response = self.agent.handle_request(request)
                self.assert_success(response)
                self.assert_contains(response, "Hello")
    """

    def __init__(self):
        self.agent: Optional[BaseAgent] = None
        self.context = create_test_context()

    def setup(self) -> None:
        """Called before each test."""
        self.agent = self.setup_agent()
        if self.agent:
            self.agent.initialize({})

    def teardown(self) -> None:
        """Called after each test."""
        if self.agent:
            self.agent.shutdown()

    def setup_agent(self) -> BaseAgent:
        """
        Create and return the agent to test.

        Override this method in subclasses.
        """
        raise NotImplementedError("Override setup_agent() to return your agent")

    def create_request(
        self,
        prompt: str,
        intent: str = "general",
        destination: str = "test_agent",
        **kwargs,
    ) -> FlowRequest:
        """Create a test request using test context."""
        return create_test_request(
            prompt=prompt,
            intent=intent,
            source=self.context.user_id,
            destination=destination,
            **kwargs,
        )

    def request_builder(self) -> TestRequestBuilder:
        """Get a request builder with test context."""
        return TestRequestBuilder().from_user(self.context.user_id)

    def send_request(self, request: FlowRequest) -> FlowResponse:
        """Send request to agent."""
        if not self.agent:
            raise RuntimeError("No agent configured. Override setup_agent()")
        return self.agent.handle_request(request)

    def send(self, prompt: str, intent: str = "general") -> FlowResponse:
        """Convenience method to send a simple prompt."""
        request = self.create_request(prompt, intent)
        return self.send_request(request)

    # Assertions

    def assert_success(self, response: FlowResponse) -> None:
        """Assert response was successful."""
        assert response.is_success(), f"Expected success, got {response.status.name}"

    def assert_refused(self, response: FlowResponse) -> None:
        """Assert response was refused."""
        assert response.was_refused(), f"Expected refused, got {response.status.name}"

    def assert_error(self, response: FlowResponse) -> None:
        """Assert response was error."""
        assert response.is_error(), f"Expected error, got {response.status.name}"

    def assert_contains(self, response: FlowResponse, text: str) -> None:
        """Assert response output contains text."""
        assert text in response.content.output, (
            f"Expected output to contain '{text}', got: {response.content.output}"
        )

    def assert_not_contains(self, response: FlowResponse, text: str) -> None:
        """Assert response output does not contain text."""
        assert text not in response.content.output, (
            f"Expected output not to contain '{text}'"
        )

    def assert_output_matches(self, response: FlowResponse, pattern: str) -> None:
        """Assert response output matches regex pattern."""
        import re
        assert re.search(pattern, response.content.output), (
            f"Expected output to match '{pattern}'"
        )

    def assert_has_reasoning(self, response: FlowResponse) -> None:
        """Assert response has reasoning."""
        assert response.content.reasoning, "Expected response to have reasoning"

    def assert_capability(self, capability: CapabilityType) -> None:
        """Assert agent has capability."""
        if not self.agent:
            raise RuntimeError("No agent configured")
        caps = self.agent.get_capabilities()
        assert capability in caps.capabilities, (
            f"Expected agent to have capability {capability.value}"
        )
