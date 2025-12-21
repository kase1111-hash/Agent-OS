"""
Tests for Agent OS Agent Interface
"""

import pytest
from datetime import datetime
from typing import Dict, Any
from uuid import uuid4

from src.agents.interface import (
    AgentInterface,
    BaseAgent,
    AgentState,
    CapabilityType,
    AgentCapabilities,
    RequestValidationResult,
    AgentMetrics,
)
from src.messaging.models import (
    FlowRequest,
    FlowResponse,
    RequestContent,
    MessageStatus,
    create_request,
)


class TestAgentCapabilities:
    """Tests for AgentCapabilities model."""

    def test_create_capabilities(self):
        """Create agent capabilities."""
        caps = AgentCapabilities(
            name="sage",
            version="1.0.0",
            description="Reasoning agent",
            capabilities={CapabilityType.REASONING, CapabilityType.GENERATION},
            supported_intents=["query.factual", "query.analytical"],
        )

        assert caps.name == "sage"
        assert caps.version == "1.0.0"
        assert CapabilityType.REASONING in caps.capabilities
        assert "query.factual" in caps.supported_intents

    def test_capabilities_to_dict(self):
        """Convert capabilities to dictionary."""
        caps = AgentCapabilities(
            name="test",
            version="1.0.0",
            description="Test agent",
            capabilities={CapabilityType.GENERATION},
            context_window=8192,
        )

        d = caps.to_dict()

        assert d["name"] == "test"
        assert d["context_window"] == 8192
        assert "generation" in d["capabilities"]


class TestRequestValidationResult:
    """Tests for RequestValidationResult."""

    def test_valid_result(self):
        """Create valid result."""
        result = RequestValidationResult(is_valid=True)

        assert result.is_valid is True
        assert len(result.errors) == 0

    def test_add_error(self):
        """Adding error invalidates result."""
        result = RequestValidationResult(is_valid=True)
        result.add_error("Validation failed")

        assert result.is_valid is False
        assert "Validation failed" in result.errors

    def test_add_warning(self):
        """Adding warning doesn't invalidate."""
        result = RequestValidationResult(is_valid=True)
        result.add_warning("Minor issue")

        assert result.is_valid is True
        assert "Minor issue" in result.warnings


class TestBaseAgent:
    """Tests for BaseAgent class."""

    def test_create_agent(self):
        """Create a base agent."""
        agent = BaseAgent(
            name="test-agent",
            description="A test agent",
            version="0.1.0",
        )

        assert agent.name == "test-agent"
        assert agent.state == AgentState.UNINITIALIZED
        assert agent.is_ready is False

    def test_agent_initialize(self):
        """Initialize an agent."""
        agent = BaseAgent(name="test")
        config = {"setting": "value"}

        result = agent.initialize(config)

        assert result is True
        assert agent.state == AgentState.READY
        assert agent.is_ready is True

    def test_agent_shutdown(self):
        """Shutdown an agent."""
        agent = BaseAgent(name="test")
        agent.initialize({})

        result = agent.shutdown()

        assert result is True
        assert agent.state == AgentState.SHUTDOWN
        assert agent.is_ready is False

    def test_get_capabilities(self):
        """Get agent capabilities."""
        agent = BaseAgent(
            name="test",
            description="Test agent",
            version="1.0.0",
            capabilities={CapabilityType.REASONING},
            supported_intents=["query.*"],
        )

        caps = agent.get_capabilities()

        assert caps.name == "test"
        assert caps.version == "1.0.0"
        assert CapabilityType.REASONING in caps.capabilities

    def test_agent_metrics(self):
        """Agent metrics are tracked."""
        agent = BaseAgent(name="test")
        agent.initialize({})

        metrics = agent.metrics

        assert metrics.requests_processed == 0
        assert metrics.uptime_seconds > 0

    def test_validate_request_default(self):
        """Default validation passes."""
        agent = BaseAgent(name="test")
        agent.initialize({})

        request = create_request(
            source="user",
            destination="test",
            intent="query.factual",
            prompt="What is AI?",
        )

        result = agent.validate_request(request)

        # Default validation should pass (no rules loaded)
        assert result.is_valid is True


class SampleAgent(BaseAgent):
    """Sample agent for testing."""

    def __init__(self, name: str):
        super().__init__(
            name=name,
            description="Sample agent for testing",
            version="1.0.0",
            capabilities={CapabilityType.GENERATION},
            supported_intents=["test.*"],
        )
        self.processed_count = 0

    def process(self, request: FlowRequest) -> FlowResponse:
        """Process a request."""
        self.processed_count += 1
        return request.create_response(
            source=self.name,
            status=MessageStatus.SUCCESS,
            output=f"Processed: {request.content.prompt}",
        )


class TestCustomAgent:
    """Tests for custom agent implementations."""

    def test_custom_agent_process(self):
        """Custom agent processes requests."""
        agent = SampleAgent(name="sample")
        agent.initialize({})

        request = create_request(
            source="user",
            destination="sample",
            intent="test.echo",
            prompt="Hello world",
        )

        response = agent.process(request)

        assert response.status == MessageStatus.SUCCESS
        assert "Processed: Hello world" in response.content.output
        assert agent.processed_count == 1

    def test_handle_request_lifecycle(self):
        """handle_request manages full lifecycle."""
        agent = SampleAgent(name="sample")
        agent.initialize({})

        request = create_request(
            source="user",
            destination="sample",
            intent="test.echo",
            prompt="Test",
        )

        response = agent.handle_request(request)

        assert response.status == MessageStatus.SUCCESS
        assert agent.metrics.requests_processed == 1
        assert agent.metrics.requests_succeeded == 1

    def test_handle_request_not_ready(self):
        """handle_request fails if not ready."""
        agent = SampleAgent(name="sample")
        # Don't initialize

        request = create_request(
            source="user",
            destination="sample",
            intent="test.echo",
            prompt="Test",
        )

        response = agent.handle_request(request)

        assert response.status == MessageStatus.ERROR
        assert "not ready" in response.content.errors[0].lower()


class FailingAgent(BaseAgent):
    """Agent that fails during processing."""

    def process(self, request: FlowRequest) -> FlowResponse:
        raise RuntimeError("Processing failed intentionally")


class TestAgentErrorHandling:
    """Tests for error handling."""

    def test_process_error_handling(self):
        """Errors during processing are handled."""
        agent = FailingAgent(name="failing")
        agent.initialize({})

        request = create_request(
            source="user",
            destination="failing",
            intent="test",
            prompt="Test",
        )

        response = agent.handle_request(request)

        assert response.status == MessageStatus.ERROR
        assert agent.metrics.requests_failed == 1
        assert len(agent.metrics.errors) == 1


class TestAgentCallbacks:
    """Tests for callback system."""

    def test_on_request_callback(self):
        """on_request callback is invoked."""
        agent = SampleAgent(name="sample")
        agent.initialize({})

        callback_called = []

        def on_request(request):
            callback_called.append(request)

        agent.register_callback("on_request", on_request)

        request = create_request(
            source="user",
            destination="sample",
            intent="test",
            prompt="Test",
        )

        agent.handle_request(request)

        assert len(callback_called) == 1
        assert callback_called[0] == request

    def test_on_response_callback(self):
        """on_response callback is invoked."""
        agent = SampleAgent(name="sample")
        agent.initialize({})

        responses = []

        def on_response(request, response):
            responses.append((request, response))

        agent.register_callback("on_response", on_response)

        request = create_request(
            source="user",
            destination="sample",
            intent="test",
            prompt="Test",
        )

        agent.handle_request(request)

        assert len(responses) == 1
        assert responses[0][1].status == MessageStatus.SUCCESS

    def test_on_error_callback(self):
        """on_error callback is invoked on errors."""
        agent = FailingAgent(name="failing")
        agent.initialize({})

        errors = []

        def on_error(request, error):
            errors.append((request, error))

        agent.register_callback("on_error", on_error)

        request = create_request(
            source="user",
            destination="failing",
            intent="test",
            prompt="Test",
        )

        agent.handle_request(request)

        assert len(errors) == 1
        assert "intentionally" in str(errors[0][1])

    def test_on_shutdown_callback(self):
        """on_shutdown callback is invoked."""
        agent = SampleAgent(name="sample")
        agent.initialize({})

        shutdown_called = []

        def on_shutdown():
            shutdown_called.append(True)

        agent.register_callback("on_shutdown", on_shutdown)

        agent.shutdown()

        assert len(shutdown_called) == 1

    def test_invalid_callback_event(self):
        """Invalid callback event raises error."""
        agent = SampleAgent(name="sample")

        with pytest.raises(ValueError):
            agent.register_callback("invalid_event", lambda: None)


class TestAgentMetrics:
    """Tests for metrics tracking."""

    def test_metrics_increments(self):
        """Metrics increment properly."""
        agent = SampleAgent(name="sample")
        agent.initialize({})

        for i in range(5):
            request = create_request(
                source="user",
                destination="sample",
                intent="test",
                prompt=f"Test {i}",
            )
            agent.handle_request(request)

        metrics = agent.metrics

        assert metrics.requests_processed == 5
        assert metrics.requests_succeeded == 5
        assert metrics.requests_failed == 0
        # Response time may be 0 if processing is very fast
        assert metrics.average_response_time_ms >= 0

    def test_uptime_tracking(self):
        """Uptime is tracked."""
        agent = SampleAgent(name="sample")
        agent.initialize({})

        import time
        time.sleep(0.1)

        metrics = agent.metrics

        assert metrics.uptime_seconds >= 0.1
