"""
Agent Testing Framework

Provides utilities for testing Agent OS agents including:
- Test fixtures for common scenarios
- Mock objects for dependencies
- Custom assertions for agent behavior
- Test runner with reporting
"""

from .fixtures import (
    AgentTestCase,
    create_test_request,
    create_test_response,
    create_test_context,
    TestRequestBuilder,
)
from .mocks import (
    MockAgent,
    MockMemoryStore,
    MockToolsClient,
    MockConstitution,
)
from .assertions import (
    assert_response_success,
    assert_response_refused,
    assert_response_error,
    assert_response_contains,
    assert_response_matches,
    assert_agent_capability,
    assert_validation_passed,
    assert_validation_failed,
)
from .runner import (
    AgentTestRunner,
    TestResult,
    TestSuite,
    run_agent_tests,
)


__all__ = [
    # Fixtures
    "AgentTestCase",
    "create_test_request",
    "create_test_response",
    "create_test_context",
    "TestRequestBuilder",
    # Mocks
    "MockAgent",
    "MockMemoryStore",
    "MockToolsClient",
    "MockConstitution",
    # Assertions
    "assert_response_success",
    "assert_response_refused",
    "assert_response_error",
    "assert_response_contains",
    "assert_response_matches",
    "assert_agent_capability",
    "assert_validation_passed",
    "assert_validation_failed",
    # Runner
    "AgentTestRunner",
    "TestResult",
    "TestSuite",
    "run_agent_tests",
]
