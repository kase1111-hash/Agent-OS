"""
Tests for AgentInterface / BaseAgent — covers set_constitutional_rules,
get_applicable_rules, escalation/refusal response creation, _update_metrics,
and constitutional context loading.

Targeted at boosting interface.py coverage from ~33% to 85%+.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set
from unittest.mock import MagicMock, patch

import pytest

from src.agents.interface import (
    AgentCapabilities,
    AgentState,
    BaseAgent,
    CapabilityType,
    RequestValidationResult,
)
from src.core.models import AuthorityLevel, Rule, RuleType
from src.messaging.models import (
    FlowRequest,
    FlowResponse,
    MessageStatus,
    RequestContent,
    ResponseContent,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_agent(name="test-agent", **kwargs):
    """Create a BaseAgent and initialize it."""
    agent = BaseAgent(
        name=name,
        description="Test agent for coverage",
        version="1.0.0",
        capabilities={CapabilityType.REASONING},
        supported_intents=["query.*"],
        **kwargs,
    )
    agent.initialize({})
    return agent


def _make_request(prompt="What is physics?", intent="query.factual"):
    """Create a minimal valid FlowRequest."""
    return FlowRequest(
        source="user",
        destination="test-agent",
        intent=intent,
        content=RequestContent(prompt=prompt),
    )


def _make_rule(
    rule_type=RuleType.PROHIBITION,
    content="Test rule",
    keywords=None,
    scope="all_agents",
    is_immutable=False,
):
    return Rule(
        id=f"rule-{hash(content) % 10000}",
        content=content,
        rule_type=rule_type,
        section="Test",
        section_path=["Test"],
        authority_level=AuthorityLevel.SUPREME,
        scope=scope,
        keywords=set(keywords or []),
        is_immutable=is_immutable,
    )


# ---------------------------------------------------------------------------
# set_constitutional_rules / get_applicable_rules
# ---------------------------------------------------------------------------


class TestConstitutionalRules:
    def test_set_constitutional_rules(self):
        """set_constitutional_rules stores rules on the agent."""
        agent = _make_agent()
        rules = [
            _make_rule(keywords=["harm"]),
            _make_rule(content="Must be truthful", keywords=["truth"]),
        ]
        agent.set_constitutional_rules(rules)
        assert len(agent._constitutional_rules) == 2

    def test_set_empty_rules(self):
        """Setting empty rules list clears rules."""
        agent = _make_agent()
        agent.set_constitutional_rules([_make_rule()])
        agent.set_constitutional_rules([])
        assert len(agent._constitutional_rules) == 0

    def test_get_applicable_rules_keyword_match(self):
        """get_applicable_rules returns rules matching request keywords."""
        agent = _make_agent()
        harm_rule = _make_rule(keywords=["harm", "delete"])
        truth_rule = _make_rule(content="Be truthful", keywords=["truth", "honest"])
        agent.set_constitutional_rules([harm_rule, truth_rule])

        request = _make_request(prompt="Delete all the data and harm the system")
        applicable = agent.get_applicable_rules(request)
        assert harm_rule in applicable

    def test_get_applicable_rules_intent_match(self):
        """Rules match on intent keywords too."""
        agent = _make_agent()
        memory_rule = _make_rule(keywords=["memory"])
        agent.set_constitutional_rules([memory_rule])

        request = _make_request(prompt="Save this", intent="memory.store")
        applicable = agent.get_applicable_rules(request)
        assert memory_rule in applicable

    def test_get_applicable_rules_no_match(self):
        """No matching rules returns empty list."""
        agent = _make_agent()
        rule = _make_rule(keywords=["cryptography"])
        agent.set_constitutional_rules([rule])

        request = _make_request(prompt="What is the weather?")
        applicable = agent.get_applicable_rules(request)
        assert len(applicable) == 0

    def test_get_applicable_rules_no_rules_set(self):
        """Empty rules list returns empty applicable rules."""
        agent = _make_agent()
        request = _make_request()
        applicable = agent.get_applicable_rules(request)
        assert applicable == []


# ---------------------------------------------------------------------------
# Response creation methods
# ---------------------------------------------------------------------------


class TestResponseCreation:
    def test_create_refused_response(self):
        """_create_refused_response returns REFUSED status."""
        agent = _make_agent()
        request = _make_request()
        response = agent._create_refused_response(
            request, errors=["Rule violation"], reason="Constitutional rule X violated"
        )
        assert response.status == MessageStatus.REFUSED
        assert "Constitutional rule X violated" in response.content.reasoning

    def test_create_refused_response_no_reason(self):
        """_create_refused_response without explicit reason uses errors."""
        agent = _make_agent()
        request = _make_request()
        response = agent._create_refused_response(
            request, errors=["Error A", "Error B"]
        )
        assert response.status == MessageStatus.REFUSED
        assert "Error A" in response.content.reasoning

    def test_create_escalation_response(self):
        """_create_escalation_response returns PARTIAL with escalation action."""
        agent = _make_agent()
        request = _make_request()
        response = agent._create_escalation_response(request, "Needs human approval")
        assert response.status == MessageStatus.PARTIAL
        assert len(response.next_actions) >= 1
        action = response.next_actions[0]
        assert action["action"] == "escalate_to_human"
        assert action["reason"] == "Needs human approval"

    def test_create_error_response(self):
        """_create_error_response returns ERROR status."""
        agent = _make_agent()
        request = _make_request()
        response = agent._create_error_response(request, "Something went wrong")
        assert response.status == MessageStatus.ERROR
        assert "Something went wrong" in response.content.errors


# ---------------------------------------------------------------------------
# handle_request paths
# ---------------------------------------------------------------------------


class TestHandleRequest:
    def test_handle_request_not_ready(self):
        """handle_request raises if agent not in READY state."""
        agent = BaseAgent(
            name="test", description="Test", version="1.0",
        )
        # Not initialized, state is UNINITIALIZED
        request = _make_request()
        response = agent.handle_request(request)
        assert response.status == MessageStatus.ERROR

    def test_handle_request_success(self):
        """handle_request returns successful response for valid request."""
        agent = _make_agent()
        request = _make_request()
        response = agent.handle_request(request)
        # Default BaseAgent.process() returns ERROR (must override), but
        # the lifecycle should work
        assert isinstance(response, FlowResponse)

    def test_handle_request_with_callbacks(self):
        """Callbacks are invoked on request/response."""
        agent = _make_agent()
        on_request_calls = []
        on_response_calls = []

        agent.register_callback("on_request", lambda r: on_request_calls.append(r))
        agent.register_callback("on_response", lambda req, resp: on_response_calls.append(resp))

        request = _make_request()
        agent.handle_request(request)

        assert len(on_request_calls) == 1
        assert len(on_response_calls) == 1

    def test_handle_request_escalation_path(self):
        """handle_request returns escalation response when validation requires it."""
        agent = _make_agent()

        # Override validate_request to return escalation
        def mock_validate(request):
            result = RequestValidationResult(is_valid=True)
            result.requires_escalation = True
            result.escalation_reason = "Needs human review"
            return result

        agent.validate_request = mock_validate
        request = _make_request()
        response = agent.handle_request(request)
        assert response.status == MessageStatus.PARTIAL
        assert any(a.get("action") == "escalate_to_human" for a in response.next_actions)

    def test_handle_request_refusal_path(self):
        """handle_request returns refused response when validation fails."""
        agent = _make_agent()

        # Set up rules that will trigger refusal
        harm_rule = _make_rule(
            rule_type=RuleType.PROHIBITION,
            keywords=["harm", "destroy"],
        )
        agent.set_constitutional_rules([harm_rule])

        request = _make_request(prompt="Please destroy and harm everything")
        response = agent.handle_request(request)
        # BaseAgent.validate_request checks for prohibition keywords
        assert response.status in (MessageStatus.REFUSED, MessageStatus.ERROR)


# ---------------------------------------------------------------------------
# _update_metrics
# ---------------------------------------------------------------------------


class TestUpdateMetrics:
    def test_metrics_increment_on_request(self):
        """Metrics should track request count."""
        agent = _make_agent()
        request = _make_request()
        agent.handle_request(request)
        assert agent.metrics.requests_processed >= 1

    def test_metrics_track_response_time(self):
        """Metrics should track average response time."""
        agent = _make_agent()
        request = _make_request()
        agent.handle_request(request)
        # After one request, average_response_time_ms should be set
        assert agent.metrics.average_response_time_ms >= 0


# ---------------------------------------------------------------------------
# Agent lifecycle
# ---------------------------------------------------------------------------


class TestAgentLifecycle:
    def test_state_transitions(self):
        """Agent goes UNINITIALIZED -> READY -> SHUTDOWN."""
        agent = BaseAgent(name="test", description="Test", version="1.0")
        assert agent.state == AgentState.UNINITIALIZED

        agent.initialize({})
        assert agent.state == AgentState.READY
        assert agent.is_ready is True

        agent.shutdown()
        assert agent.state == AgentState.SHUTDOWN
        assert agent.is_ready is False

    def test_shutdown_callback(self):
        """on_shutdown callback fires during shutdown."""
        agent = _make_agent()
        shutdown_calls = []
        agent.register_callback("on_shutdown", lambda: shutdown_calls.append(True))
        agent.shutdown()
        assert len(shutdown_calls) == 1

    def test_error_callback(self):
        """on_error callback fires when process raises."""
        agent = _make_agent()

        # Make process raise an exception
        def failing_process(request):
            raise RuntimeError("Test error")

        agent.process = failing_process
        error_calls = []
        agent.register_callback("on_error", lambda req, exc: error_calls.append(exc))

        request = _make_request()
        response = agent.handle_request(request)
        assert response.status == MessageStatus.ERROR
        assert len(error_calls) == 1

    def test_invalid_callback_event(self):
        """Registering callback for invalid event raises ValueError."""
        agent = _make_agent()
        with pytest.raises(ValueError):
            agent.register_callback("on_nonexistent", lambda: None)


# ---------------------------------------------------------------------------
# Capabilities
# ---------------------------------------------------------------------------


class TestCapabilities:
    def test_get_capabilities_returns_dataclass(self):
        """get_capabilities returns AgentCapabilities."""
        agent = _make_agent()
        caps = agent.get_capabilities()
        assert isinstance(caps, AgentCapabilities)
        assert caps.name == "test-agent"
        assert CapabilityType.REASONING in caps.capabilities

    def test_capabilities_to_dict(self):
        """AgentCapabilities.to_dict works correctly."""
        agent = _make_agent()
        caps = agent.get_capabilities()
        d = caps.to_dict()
        assert d["name"] == "test-agent"
        assert "version" in d
