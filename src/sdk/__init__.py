"""
Agent OS SDK

A comprehensive SDK for developing Agent OS agents.

Components:
- Templates: Pre-built agent templates for common patterns
- Builder: Fluent API for building agents declaratively
- Testing: Test fixtures, mocks, and assertions
- Decorators: Function decorators for common patterns
- Lifecycle: Agent lifecycle management

Quick Start:
    # Using the builder
    from src.sdk import agent, CapabilityType

    my_agent = (
        agent("my_agent")
        .with_description("My custom agent")
        .with_capability(CapabilityType.REASONING)
        .with_handler(lambda req: f"Hello, {req.content.prompt}!")
        .build()
    )

    # Using templates
    from src.sdk.templates import create_simple_agent

    agent = create_simple_agent(
        name="greeter",
        handler=lambda req: f"Hello, {req.source}!",
        description="A friendly greeter",
    )

    # Testing
    from src.sdk.testing import AgentTestCase, assert_response_success

    class TestMyAgent(AgentTestCase):
        def setup_agent(self):
            return MyAgent()

        def test_greeting(self):
            response = self.send("Hello!")
            assert_response_success(response)
"""

# Core types from agents interface
from src.agents.interface import (
    AgentCapabilities,
    AgentInterface,
    AgentMetrics,
    AgentState,
    BaseAgent,
    CapabilityType,
    RequestValidationResult,
)

# Builder
from .builder import (
    AgentBuilder,
    agent,
)

# Decorators
from .decorators import (
    cache_response,
    catch_errors,
    log_requests,
    log_responses,
    measure_time,
    rate_limit,
    require_capability,
    retry,
    validate_request,
)

# Lifecycle
from .lifecycle import (
    AgentLifecycleManager,
    AgentPool,
    AgentStats,
    HealthCheck,
    HealthStatus,
    get_manager,
    register_agent,
    shutdown_all,
)

# Templates
from .templates import (
    AgentTemplate,
    GenerationAgentTemplate,
    ReasoningAgentTemplate,
    SimpleAgent,
    ToolUseAgentTemplate,
    ValidationAgentTemplate,
    create_generation_agent,
    create_reasoning_agent,
    create_simple_agent,
    create_tool_use_agent,
    create_validation_agent,
)

# Testing
from .testing import (
    AgentTestCase,
    AgentTestRunner,
    MockAgent,
    MockConstitution,
    MockMemoryStore,
    MockToolsClient,
    TestRequestBuilder,
    TestResult,
    TestSuite,
    assert_agent_capability,
    assert_response_contains,
    assert_response_error,
    assert_response_matches,
    assert_response_refused,
    assert_response_success,
    assert_validation_failed,
    assert_validation_passed,
    create_test_context,
    create_test_request,
    create_test_response,
    run_agent_tests,
)

__all__ = [
    # Core types
    "AgentInterface",
    "BaseAgent",
    "AgentState",
    "AgentCapabilities",
    "CapabilityType",
    "RequestValidationResult",
    "AgentMetrics",
    # Templates
    "AgentTemplate",
    "SimpleAgent",
    "create_simple_agent",
    "ReasoningAgentTemplate",
    "create_reasoning_agent",
    "GenerationAgentTemplate",
    "create_generation_agent",
    "ValidationAgentTemplate",
    "create_validation_agent",
    "ToolUseAgentTemplate",
    "create_tool_use_agent",
    # Builder
    "AgentBuilder",
    "agent",
    # Testing
    "AgentTestCase",
    "create_test_request",
    "create_test_response",
    "create_test_context",
    "TestRequestBuilder",
    "MockAgent",
    "MockMemoryStore",
    "MockToolsClient",
    "MockConstitution",
    "assert_response_success",
    "assert_response_refused",
    "assert_response_error",
    "assert_response_contains",
    "assert_response_matches",
    "assert_agent_capability",
    "assert_validation_passed",
    "assert_validation_failed",
    "AgentTestRunner",
    "TestResult",
    "TestSuite",
    "run_agent_tests",
    # Decorators
    "log_requests",
    "log_responses",
    "measure_time",
    "retry",
    "catch_errors",
    "rate_limit",
    "cache_response",
    "validate_request",
    "require_capability",
    # Lifecycle
    "AgentLifecycleManager",
    "AgentPool",
    "HealthCheck",
    "HealthStatus",
    "AgentStats",
    "register_agent",
    "get_manager",
    "shutdown_all",
]
