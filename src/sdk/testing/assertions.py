"""
Custom Assertions for Agent Testing

Provides assertion functions for testing agent behavior.
"""

import re
from typing import Any, List, Optional, Pattern, Set, Union

from src.agents.interface import (
    BaseAgent,
    AgentCapabilities,
    CapabilityType,
    RequestValidationResult,
)
from src.messaging.models import FlowResponse, MessageStatus


# Use the builtin AssertionError for compatibility with pytest


def assert_response_success(
    response: FlowResponse,
    message: Optional[str] = None,
) -> None:
    """
    Assert that response indicates success.

    Args:
        response: The response to check
        message: Optional custom message
    """
    if not response.is_success():
        msg = message or f"Expected successful response, got {response.status.name}"
        if response.content.errors:
            msg += f". Errors: {response.content.errors}"
        raise AssertionError(msg)


def assert_response_refused(
    response: FlowResponse,
    message: Optional[str] = None,
) -> None:
    """
    Assert that response was refused.

    Args:
        response: The response to check
        message: Optional custom message
    """
    if not response.was_refused():
        msg = message or f"Expected refused response, got {response.status.name}"
        raise AssertionError(msg)


def assert_response_error(
    response: FlowResponse,
    message: Optional[str] = None,
) -> None:
    """
    Assert that response indicates error.

    Args:
        response: The response to check
        message: Optional custom message
    """
    if not response.is_error():
        msg = message or f"Expected error response, got {response.status.name}"
        raise AssertionError(msg)


def assert_response_status(
    response: FlowResponse,
    expected: MessageStatus,
    message: Optional[str] = None,
) -> None:
    """
    Assert response has specific status.

    Args:
        response: The response to check
        expected: Expected status
        message: Optional custom message
    """
    if response.status != expected:
        msg = message or f"Expected status {expected.name}, got {response.status.name}"
        raise AssertionError(msg)


def assert_response_contains(
    response: FlowResponse,
    text: str,
    case_sensitive: bool = True,
    message: Optional[str] = None,
) -> None:
    """
    Assert response output contains text.

    Args:
        response: The response to check
        text: Text to find
        case_sensitive: Whether to match case
        message: Optional custom message
    """
    output = response.content.output
    search_text = text
    search_output = output

    if not case_sensitive:
        search_text = text.lower()
        search_output = output.lower()

    if search_text not in search_output:
        msg = message or f"Expected output to contain '{text}'"
        msg += f"\nActual output: {output[:200]}..."
        raise AssertionError(msg)


def assert_response_not_contains(
    response: FlowResponse,
    text: str,
    case_sensitive: bool = True,
    message: Optional[str] = None,
) -> None:
    """
    Assert response output does not contain text.

    Args:
        response: The response to check
        text: Text that should not be present
        case_sensitive: Whether to match case
        message: Optional custom message
    """
    output = response.content.output
    search_text = text
    search_output = output

    if not case_sensitive:
        search_text = text.lower()
        search_output = output.lower()

    if search_text in search_output:
        msg = message or f"Expected output NOT to contain '{text}'"
        raise AssertionError(msg)


def assert_response_matches(
    response: FlowResponse,
    pattern: Union[str, Pattern],
    message: Optional[str] = None,
) -> None:
    """
    Assert response output matches regex pattern.

    Args:
        response: The response to check
        pattern: Regex pattern
        message: Optional custom message
    """
    output = response.content.output

    if isinstance(pattern, str):
        pattern = re.compile(pattern)

    if not pattern.search(output):
        msg = message or f"Expected output to match pattern '{pattern.pattern}'"
        msg += f"\nActual output: {output[:200]}..."
        raise AssertionError(msg)


def assert_response_not_matches(
    response: FlowResponse,
    pattern: Union[str, Pattern],
    message: Optional[str] = None,
) -> None:
    """
    Assert response output does not match regex pattern.

    Args:
        response: The response to check
        pattern: Regex pattern that should not match
        message: Optional custom message
    """
    output = response.content.output

    if isinstance(pattern, str):
        pattern = re.compile(pattern)

    if pattern.search(output):
        msg = message or f"Expected output NOT to match pattern '{pattern.pattern}'"
        raise AssertionError(msg)


def assert_response_has_reasoning(
    response: FlowResponse,
    message: Optional[str] = None,
) -> None:
    """
    Assert response has reasoning explanation.

    Args:
        response: The response to check
        message: Optional custom message
    """
    if not response.content.reasoning:
        msg = message or "Expected response to have reasoning"
        raise AssertionError(msg)


def assert_response_has_errors(
    response: FlowResponse,
    min_count: int = 1,
    message: Optional[str] = None,
) -> None:
    """
    Assert response has error messages.

    Args:
        response: The response to check
        min_count: Minimum number of errors expected
        message: Optional custom message
    """
    error_count = len(response.content.errors)
    if error_count < min_count:
        msg = message or f"Expected at least {min_count} errors, got {error_count}"
        raise AssertionError(msg)


def assert_response_error_contains(
    response: FlowResponse,
    text: str,
    message: Optional[str] = None,
) -> None:
    """
    Assert response errors contain text.

    Args:
        response: The response to check
        text: Text to find in errors
        message: Optional custom message
    """
    errors = " ".join(response.content.errors)
    if text not in errors:
        msg = message or f"Expected errors to contain '{text}'"
        msg += f"\nActual errors: {response.content.errors}"
        raise AssertionError(msg)


def assert_agent_capability(
    agent: BaseAgent,
    capability: CapabilityType,
    message: Optional[str] = None,
) -> None:
    """
    Assert agent has capability.

    Args:
        agent: The agent to check
        capability: Expected capability
        message: Optional custom message
    """
    caps = agent.get_capabilities()
    if capability not in caps.capabilities:
        msg = message or f"Expected agent to have capability {capability.value}"
        msg += f"\nActual capabilities: {[c.value for c in caps.capabilities]}"
        raise AssertionError(msg)


def assert_agent_capabilities(
    agent: BaseAgent,
    capabilities: Set[CapabilityType],
    message: Optional[str] = None,
) -> None:
    """
    Assert agent has all capabilities.

    Args:
        agent: The agent to check
        capabilities: Expected capabilities
        message: Optional custom message
    """
    caps = agent.get_capabilities()
    missing = capabilities - caps.capabilities
    if missing:
        msg = message or f"Agent missing capabilities: {[c.value for c in missing]}"
        raise AssertionError(msg)


def assert_agent_intent(
    agent: BaseAgent,
    intent: str,
    message: Optional[str] = None,
) -> None:
    """
    Assert agent supports intent pattern.

    Args:
        agent: The agent to check
        intent: Intent pattern
        message: Optional custom message
    """
    caps = agent.get_capabilities()
    if intent not in caps.supported_intents:
        msg = message or f"Expected agent to support intent '{intent}'"
        msg += f"\nSupported intents: {caps.supported_intents}"
        raise AssertionError(msg)


def assert_validation_passed(
    result: RequestValidationResult,
    message: Optional[str] = None,
) -> None:
    """
    Assert validation passed.

    Args:
        result: Validation result
        message: Optional custom message
    """
    if not result.is_valid:
        msg = message or "Expected validation to pass"
        if result.errors:
            msg += f"\nErrors: {result.errors}"
        raise AssertionError(msg)


def assert_validation_failed(
    result: RequestValidationResult,
    message: Optional[str] = None,
) -> None:
    """
    Assert validation failed.

    Args:
        result: Validation result
        message: Optional custom message
    """
    if result.is_valid:
        msg = message or "Expected validation to fail"
        raise AssertionError(msg)


def assert_validation_error_contains(
    result: RequestValidationResult,
    text: str,
    message: Optional[str] = None,
) -> None:
    """
    Assert validation errors contain text.

    Args:
        result: Validation result
        text: Text to find
        message: Optional custom message
    """
    errors = " ".join(result.errors)
    if text not in errors:
        msg = message or f"Expected validation errors to contain '{text}'"
        msg += f"\nActual errors: {result.errors}"
        raise AssertionError(msg)


def assert_requires_escalation(
    result: RequestValidationResult,
    message: Optional[str] = None,
) -> None:
    """
    Assert validation requires escalation.

    Args:
        result: Validation result
        message: Optional custom message
    """
    if not result.requires_escalation:
        msg = message or "Expected validation to require escalation"
        raise AssertionError(msg)


def assert_response_time_under(
    response: FlowResponse,
    max_ms: int,
    message: Optional[str] = None,
) -> None:
    """
    Assert response time is under threshold.

    Args:
        response: The response to check
        max_ms: Maximum allowed time in milliseconds
        message: Optional custom message
    """
    inference_time = response.content.metadata.inference_time_ms
    if inference_time and inference_time > max_ms:
        msg = message or f"Expected response time under {max_ms}ms, got {inference_time}ms"
        raise AssertionError(msg)
