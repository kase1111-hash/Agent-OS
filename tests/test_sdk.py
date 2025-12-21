"""
Tests for UC-016: Agent SDK

Tests the agent development templates, builder, testing framework,
decorators, and lifecycle management.
"""

import time
from typing import Dict, Any, List

import pytest

from src.sdk import (
    # Core types
    BaseAgent,
    AgentState,
    CapabilityType,
    # Templates
    AgentTemplate,
    SimpleAgent,
    create_simple_agent,
    ReasoningAgentTemplate,
    create_reasoning_agent,
    GenerationAgentTemplate,
    create_generation_agent,
    ValidationAgentTemplate,
    create_validation_agent,
    # Builder
    AgentBuilder,
    agent,
    # Testing
    AgentTestCase,
    create_test_request,
    create_test_response,
    create_test_context,
    TestRequestBuilder,
    MockAgent,
    MockMemoryStore,
    MockToolsClient,
    MockConstitution,
    assert_response_success,
    assert_response_refused,
    assert_response_error,
    assert_response_contains,
    assert_response_matches,
    assert_agent_capability,
    assert_validation_passed,
    assert_validation_failed,
    AgentTestRunner,
    TestSuite,
    run_agent_tests,
    # Decorators
    log_requests,
    measure_time,
    retry,
    catch_errors,
    rate_limit,
    cache_response,
    validate_request,
    # Lifecycle
    AgentLifecycleManager,
    AgentPool,
    HealthCheck,
    HealthStatus,
    register_agent,
)
from src.sdk.templates.base import AgentConfig
from src.sdk.templates.reasoning import ReasoningResult, ReasoningConfig
from src.sdk.templates.generation import GenerationResult, GenerationConfig
from src.sdk.templates.validation import (
    ValidationResult,
    ValidationIssue,
    ValidationSeverity,
    ValidationRule,
)
from src.messaging.models import FlowRequest, FlowResponse, MessageStatus


# =============================================================================
# Template Tests
# =============================================================================


class TestSimpleAgent:
    """Tests for SimpleAgent template."""

    def test_create_simple_agent(self):
        """Test creating a simple agent."""
        def handler(request):
            return f"Echo: {request.content.prompt}"

        agent = create_simple_agent(
            name="echo",
            handler=handler,
            description="Echo agent",
        )

        assert agent.name == "echo"
        assert agent.state == AgentState.UNINITIALIZED

    def test_simple_agent_string_response(self):
        """Test simple agent with string response."""
        agent = create_simple_agent(
            name="test",
            handler=lambda req: "Hello!",
        )
        agent.initialize({})

        request = create_test_request("Hi")
        response = agent.handle_request(request)

        assert response.is_success()
        assert response.content.output == "Hello!"

    def test_simple_agent_dict_response(self):
        """Test simple agent with dict response."""
        agent = create_simple_agent(
            name="test",
            handler=lambda req: {"output": "Hello!", "reasoning": "Greeting"},
        )
        agent.initialize({})

        request = create_test_request("Hi")
        response = agent.handle_request(request)

        assert response.is_success()
        assert response.content.output == "Hello!"
        assert response.content.reasoning == "Greeting"

    def test_simple_agent_with_hooks(self):
        """Test simple agent with hooks."""
        log = []

        def pre_hook(request):
            log.append("pre")
            return request

        def post_hook(response):
            log.append("post")
            return response

        agent = create_simple_agent(
            name="test",
            handler=lambda req: "OK",
        )
        agent.add_pre_process_hook(pre_hook)
        agent.add_post_process_hook(post_hook)
        agent.initialize({})

        request = create_test_request("Test")
        agent.process(request)

        assert log == ["pre", "post"]


class TestReasoningAgent:
    """Tests for ReasoningAgentTemplate."""

    def test_create_reasoning_agent(self):
        """Test creating a reasoning agent."""
        def reasoner(problem, request):
            return ReasoningResult(
                conclusion="The answer is 42",
                confidence=0.95,
            )

        agent = create_reasoning_agent(
            name="analyzer",
            reason_fn=reasoner,
            description="Analyzes problems",
        )

        assert agent.name == "analyzer"
        caps = agent.get_capabilities()
        assert CapabilityType.REASONING in caps.capabilities

    def test_reasoning_with_steps(self):
        """Test reasoning with step tracking."""
        def reasoner(problem, request):
            return ReasoningResult(
                conclusion="Solved",
                confidence=0.9,
                steps=[],
            )

        agent = create_reasoning_agent(
            name="solver",
            reason_fn=reasoner,
        )
        agent.initialize({})

        request = create_test_request("Solve this")
        response = agent.handle_request(request)

        assert response.is_success()
        assert "Solved" in response.content.output

    def test_low_confidence_escalation(self):
        """Test that low confidence triggers escalation."""
        def low_confidence_reasoner(problem, request):
            return ReasoningResult(
                conclusion="Maybe",
                confidence=0.3,  # Below default threshold
            )

        agent = create_reasoning_agent(
            name="uncertain",
            reason_fn=low_confidence_reasoner,
            min_confidence=0.7,
        )
        agent.initialize({})

        request = create_test_request("What is this?")
        response = agent.handle_request(request)

        # Should escalate due to low confidence
        assert response.status == MessageStatus.PARTIAL


class TestGenerationAgent:
    """Tests for GenerationAgentTemplate."""

    def test_create_generation_agent(self):
        """Test creating a generation agent."""
        def generator(params, request):
            return GenerationResult(
                content=f"Generated from: {params['prompt']}",
                quality_score=0.9,
            )

        agent = create_generation_agent(
            name="writer",
            generate_fn=generator,
            description="Writes content",
        )

        assert agent.name == "writer"
        caps = agent.get_capabilities()
        assert CapabilityType.GENERATION in caps.capabilities

    def test_generation_with_template(self):
        """Test generation with templates."""
        def generator(params, request):
            return GenerationResult(
                content="Test content",
                quality_score=0.8,
            )

        agent = create_generation_agent(
            name="templated",
            generate_fn=generator,
        )
        agent.register_template("greeting", "Hello, {name}!")
        agent.initialize({})

        # Use template
        result = agent.apply_template("greeting", {"name": "World"})
        assert result == "Hello, World!"


class TestValidationAgent:
    """Tests for ValidationAgentTemplate."""

    def test_create_validation_agent(self):
        """Test creating a validation agent."""
        agent = create_validation_agent(
            name="validator",
            description="Validates input",
        )

        assert agent.name == "validator"
        caps = agent.get_capabilities()
        assert CapabilityType.VALIDATION in caps.capabilities

    def test_pattern_validation(self):
        """Test pattern-based validation."""
        agent = create_validation_agent(
            name="profanity_filter",
            rules=[
                ValidationRule(
                    code="BADWORD",
                    description="Contains bad word",
                    pattern=r"\bbad\b",
                    severity=ValidationSeverity.ERROR,
                ),
            ],
        )
        agent.initialize({})

        # Clean input
        request = create_test_request("This is good")
        response = agent.handle_request(request)
        assert response.is_success()

        # Violating input
        request = create_test_request("This is bad")
        response = agent.handle_request(request)
        assert response.was_refused()

    def test_checker_validation(self):
        """Test checker-based validation."""
        agent = create_validation_agent(
            name="length_checker",
        )
        agent.add_checker_rule(
            code="TOO_SHORT",
            checker=lambda s: len(s) < 5,
            description="Input too short",
            severity=ValidationSeverity.ERROR,
        )
        agent.initialize({})

        # Valid input
        request = create_test_request("Hello World")
        response = agent.handle_request(request)
        assert response.is_success()

        # Too short
        request = create_test_request("Hi")
        response = agent.handle_request(request)
        assert response.was_refused()


# =============================================================================
# Builder Tests
# =============================================================================


class TestAgentBuilder:
    """Tests for AgentBuilder."""

    def test_basic_builder(self):
        """Test basic agent building."""
        my_agent = (
            agent("test_agent")
            .with_description("Test agent")
            .with_handler(lambda req: "Hello!")
            .build()
        )

        assert my_agent.name == "test_agent"
        my_agent.initialize({})

        request = create_test_request("Hi")
        response = my_agent.handle_request(request)
        assert response.content.output == "Hello!"

    def test_builder_with_capabilities(self):
        """Test building with capabilities."""
        my_agent = (
            agent("capable")
            .with_capabilities(CapabilityType.REASONING, CapabilityType.GENERATION)
            .with_handler(lambda req: "OK")
            .build()
        )

        caps = my_agent.get_capabilities()
        assert CapabilityType.REASONING in caps.capabilities
        assert CapabilityType.GENERATION in caps.capabilities

    def test_builder_with_intents(self):
        """Test building with intent patterns."""
        my_agent = (
            agent("intent_handler")
            .with_intents("analyze.*", "query.*")
            .with_handler(lambda req: "Handled")
            .build()
        )

        caps = my_agent.get_capabilities()
        assert "analyze.*" in caps.supported_intents
        assert "query.*" in caps.supported_intents

    def test_builder_with_hooks(self):
        """Test building with hooks."""
        log = []

        my_agent = (
            agent("hooked")
            .with_handler(lambda req: "OK")
            .with_pre_hook(lambda req: (log.append("pre"), req)[1])
            .with_post_hook(lambda res: (log.append("post"), res)[1])
            .build()
        )
        my_agent.initialize({})

        request = create_test_request("Test")
        my_agent.process(request)

        assert "pre" in log
        assert "post" in log

    def test_builder_without_handler_fails(self):
        """Test that building without handler fails."""
        with pytest.raises(ValueError):
            agent("no_handler").build()


# =============================================================================
# Testing Framework Tests
# =============================================================================


class TestTestFixtures:
    """Tests for test fixtures."""

    def test_create_test_request(self):
        """Test creating test requests."""
        request = create_test_request(
            prompt="Hello",
            intent="greeting",
            source="user123",
        )

        assert request.content.prompt == "Hello"
        assert request.intent == "greeting"
        assert request.source == "user123"

    def test_create_test_response(self):
        """Test creating test responses."""
        request = create_test_request("Test")
        response = create_test_response(
            request=request,
            output="Response",
            status=MessageStatus.SUCCESS,
        )

        assert response.content.output == "Response"
        assert response.status == MessageStatus.SUCCESS

    def test_request_builder(self):
        """Test request builder fluent API."""
        request = (
            TestRequestBuilder()
            .with_prompt("Hello there")
            .with_intent("greeting")
            .from_user("alice")
            .with_context_item("system", "Previous chat")
            .build()
        )

        assert request.content.prompt == "Hello there"
        assert request.intent == "greeting"
        assert request.source == "alice"
        assert len(request.content.context) == 1
        assert request.content.context[0]["content"] == "Previous chat"


class TestMocks:
    """Tests for mock objects."""

    def test_mock_agent(self):
        """Test mock agent."""
        mock = MockAgent(default_response="Default")
        mock.set_response("greeting", "Hello!")
        mock.initialize({})

        # Default response
        request = create_test_request("Hi", intent="other")
        response = mock.handle_request(request)
        assert response.content.output == "Default"

        # Intent-specific response
        request = create_test_request("Hi", intent="greeting")
        response = mock.handle_request(request)
        assert response.content.output == "Hello!"

        # Recorded requests
        assert len(mock.received_requests) == 2

    def test_mock_agent_error_mode(self):
        """Test mock agent in error mode."""
        mock = MockAgent().should_error(True, "Test error")
        mock.initialize({})

        request = create_test_request("Test")
        response = mock.handle_request(request)

        assert response.is_error()
        assert "Test error" in response.content.errors

    def test_mock_memory_store(self):
        """Test mock memory store."""
        store = MockMemoryStore()

        # Store and recall
        entry_id = store.store("Important fact", {"type": "fact"})
        assert entry_id.startswith("mem-")

        results = store.recall("Important")
        assert len(results) == 1
        assert "Important fact" in results[0].content

    def test_mock_tools_client(self):
        """Test mock tools client."""
        client = MockToolsClient()
        client.register_tool("calculator", result=42)

        result = client.invoke(
            tool_name="calculator",
            parameters={"expr": "6 * 7"},
            user_id="user123",
        )

        assert result["success"]
        assert result["output"] == 42
        assert len(client.get_invocations()) == 1


class TestAssertions:
    """Tests for custom assertions."""

    def test_assert_response_success(self):
        """Test success assertion."""
        request = create_test_request("Test")
        response = create_test_response(request, "OK")

        # Should pass
        assert_response_success(response)

    def test_assert_response_success_fails(self):
        """Test success assertion fails on error."""
        request = create_test_request("Test")
        response = create_test_response(
            request, "", status=MessageStatus.ERROR
        )

        with pytest.raises(AssertionError):
            assert_response_success(response)

    def test_assert_response_contains(self):
        """Test contains assertion."""
        request = create_test_request("Test")
        response = create_test_response(request, "Hello World")

        assert_response_contains(response, "World")

        with pytest.raises(AssertionError):
            assert_response_contains(response, "Goodbye")

    def test_assert_response_matches(self):
        """Test regex match assertion."""
        request = create_test_request("Test")
        response = create_test_response(request, "The answer is 42")

        assert_response_matches(response, r"\d+")

        with pytest.raises(AssertionError):
            assert_response_matches(response, r"[A-Z]{10}")


class TestTestRunner:
    """Tests for test runner."""

    def test_run_suite(self):
        """Test running a test suite."""
        suite = TestSuite("Test Suite")
        suite.add_test("test_pass", lambda: None)
        suite.add_test("test_also_pass", lambda: None)

        result = suite.run(verbose=False)

        assert result.total_count == 2
        assert result.passed_count == 2
        assert result.all_passed

    def test_suite_with_failure(self):
        """Test suite with failing test."""
        def failing_test():
            raise AssertionError("Expected failure")

        suite = TestSuite("Failing Suite")
        suite.add_test("test_fail", failing_test)

        result = suite.run(verbose=False)

        assert result.failed_count == 1
        assert not result.all_passed


# =============================================================================
# Decorator Tests
# =============================================================================


class TestDecorators:
    """Tests for agent decorators."""

    def test_retry_decorator(self):
        """Test retry decorator."""
        attempts = [0]

        @retry(max_attempts=3, delay_seconds=0.01)
        def flaky_function():
            attempts[0] += 1
            if attempts[0] < 3:
                raise ValueError("Not yet")
            return "Success"

        result = flaky_function()
        assert result == "Success"
        assert attempts[0] == 3

    def test_retry_exhausted(self):
        """Test retry exhaustion."""
        @retry(max_attempts=2, delay_seconds=0.01)
        def always_fails():
            raise ValueError("Always fails")

        with pytest.raises(ValueError):
            always_fails()

    def test_measure_time(self):
        """Test timing decorator."""
        timings = []

        @measure_time(callback=lambda n, t: timings.append((n, t)))
        def slow_function():
            time.sleep(0.01)
            return "Done"

        result = slow_function()
        assert result == "Done"
        assert len(timings) == 1
        assert timings[0][1] >= 10  # At least 10ms


# =============================================================================
# Lifecycle Tests
# =============================================================================


class TestLifecycleManager:
    """Tests for lifecycle management."""

    def test_start_stop(self):
        """Test starting and stopping agent."""
        test_agent = create_simple_agent(
            name="lifecycle_test",
            handler=lambda req: "OK",
        )

        manager = AgentLifecycleManager(test_agent)

        assert not manager.is_running()

        manager.start()
        assert manager.is_running()
        assert test_agent.state == AgentState.READY

        manager.stop()
        assert not manager.is_running()

    def test_health_check(self):
        """Test health checking."""
        test_agent = create_simple_agent(
            name="healthy",
            handler=lambda req: "OK",
        )

        manager = AgentLifecycleManager(test_agent)
        manager.start()

        health = manager.check_health()
        assert health.status == HealthStatus.HEALTHY

        manager.stop()

    def test_custom_health_check(self):
        """Test custom health check."""
        test_agent = create_simple_agent(
            name="custom_health",
            handler=lambda req: "OK",
        )

        manager = AgentLifecycleManager(test_agent)
        manager.add_health_check(
            lambda: HealthCheck(
                status=HealthStatus.DEGRADED,
                message="Memory high",
            )
        )
        manager.start()

        health = manager.check_health()
        assert health.status == HealthStatus.DEGRADED

    def test_stats_recording(self):
        """Test statistics recording."""
        test_agent = create_simple_agent(
            name="stats_test",
            handler=lambda req: "OK",
        )

        manager = AgentLifecycleManager(test_agent)
        manager.start()

        # Record some activity
        manager.record_request(success=True, latency_ms=100)
        manager.record_request(success=True, latency_ms=200)
        manager.record_request(success=False, latency_ms=50)

        stats = manager.get_stats()
        assert stats.requests_processed == 3
        assert stats.requests_succeeded == 2
        assert stats.requests_failed == 1
        assert stats.average_latency_ms == pytest.approx(116.67, rel=0.1)


class TestAgentPool:
    """Tests for agent pool."""

    def test_pool_creation(self):
        """Test creating an agent pool."""
        pool = AgentPool(
            agent_factory=lambda: create_simple_agent(
                name="pooled",
                handler=lambda req: "OK",
            ),
            pool_size=3,
        )

        result = pool.start()
        assert result

        stats = pool.get_all_stats()
        assert len(stats) == 3

        pool.stop()

    def test_pool_get_agent(self):
        """Test getting agents from pool."""
        pool = AgentPool(
            agent_factory=lambda: create_simple_agent(
                name="pool_member",
                handler=lambda req: "OK",
            ),
            pool_size=2,
        )
        pool.start()

        agent1 = pool.get_agent()
        agent2 = pool.get_agent()

        assert agent1 is not None
        assert agent2 is not None

        pool.stop()


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests for SDK components."""

    def test_full_agent_workflow(self):
        """Test complete agent workflow."""
        # Build agent
        my_agent = (
            agent("full_workflow")
            .with_description("Full workflow test")
            .with_capability(CapabilityType.REASONING)
            .with_handler(lambda req: f"Processed: {req.content.prompt}")
            .build()
        )

        # Register with lifecycle manager
        manager = AgentLifecycleManager(my_agent)
        manager.start()

        # Create and send request
        request = (
            TestRequestBuilder()
            .with_prompt("Test input")
            .with_intent("test")
            .from_user("integration_test")
            .build()
        )

        response = my_agent.handle_request(request)

        # Assert
        assert_response_success(response)
        assert_response_contains(response, "Test input")
        assert_agent_capability(my_agent, CapabilityType.REASONING)

        manager.stop()

    def test_agent_test_case_pattern(self):
        """Test the AgentTestCase pattern."""
        class MyTestCase(AgentTestCase):
            def setup_agent(self):
                return create_simple_agent(
                    name="test_subject",
                    handler=lambda req: f"Echo: {req.content.prompt}",
                )

        test = MyTestCase()
        test.setup()

        response = test.send("Hello!")
        test.assert_success(response)
        test.assert_contains(response, "Echo:")

        test.teardown()


# =============================================================================
# Acceptance Criteria Tests
# =============================================================================


class TestAcceptanceCriteria:
    """Tests for UC-016 acceptance criteria."""

    def test_agent_development_templates(self):
        """Verify agent development templates work."""
        # Simple agent template
        simple = create_simple_agent(
            name="simple",
            handler=lambda req: "OK",
        )
        assert simple is not None

        # Reasoning template
        reasoning = create_reasoning_agent(
            name="reasoning",
            reason_fn=lambda p, r: ReasoningResult("Answer", 0.9),
        )
        caps = reasoning.get_capabilities()
        assert CapabilityType.REASONING in caps.capabilities

        # Generation template
        generation = create_generation_agent(
            name="generation",
            generate_fn=lambda p, r: GenerationResult("Content", 0.8),
        )
        caps = generation.get_capabilities()
        assert CapabilityType.GENERATION in caps.capabilities

        # Validation template
        validation = create_validation_agent(name="validation")
        caps = validation.get_capabilities()
        assert CapabilityType.VALIDATION in caps.capabilities

    def test_testing_framework(self):
        """Verify testing framework works."""
        # Test fixtures
        request = create_test_request("Test prompt")
        assert request.content.prompt == "Test prompt"

        # Mock objects
        mock = MockAgent()
        mock.initialize({})
        response = mock.handle_request(request)
        assert response.is_success()

        # Assertions
        assert_response_success(response)

        # Test runner
        suite = TestSuite("Acceptance")
        suite.add_test("pass", lambda: None)
        result = suite.run(verbose=False)
        assert result.all_passed

    def test_fluent_builder_api(self):
        """Verify fluent builder API works."""
        my_agent = (
            agent("fluent_test")
            .with_version("2.0.0")
            .with_description("Fluent API test")
            .with_capability(CapabilityType.GENERATION)
            .with_intent("create.*")
            .with_model("test-model")
            .with_timeout(60)
            .with_retries(5)
            .with_handler(lambda req: "Generated content")
            .build()
        )

        caps = my_agent.get_capabilities()
        assert caps.version == "2.0.0"
        assert caps.description == "Fluent API test"
        assert CapabilityType.GENERATION in caps.capabilities
        assert "create.*" in caps.supported_intents

    def test_lifecycle_management(self):
        """Verify lifecycle management works."""
        test_agent = create_simple_agent(
            name="lifecycle",
            handler=lambda req: "OK",
        )

        manager = AgentLifecycleManager(test_agent)

        # Start
        result = manager.start()
        assert result
        assert manager.is_running()

        # Health check
        health = manager.check_health()
        assert health.status == HealthStatus.HEALTHY

        # Stats
        stats = manager.get_stats()
        assert stats.uptime_seconds > 0

        # Stop
        manager.stop()
        assert not manager.is_running()
