"""
Tests for Agent OS Message Protocol Models
"""

import pytest
from datetime import datetime
from uuid import UUID

from src.messaging.models import (
    FlowRequest,
    FlowResponse,
    RequestMetadata,
    ResponseMetadata,
    RequestContent,
    ResponseContent,
    ConstitutionalCheck,
    MemoryRequest,
    MessageStatus,
    CheckStatus,
    MessagePriority,
    DeadLetterMessage,
    AuditLogEntry,
    create_request,
    create_response,
)


class TestFlowRequest:
    """Tests for FlowRequest model."""

    def test_create_basic_request(self):
        """Create a basic flow request."""
        request = FlowRequest(
            source="user",
            destination="whisper",
            intent="query.factual",
            content=RequestContent(prompt="What is AI?"),
        )

        assert request.source == "user"
        assert request.destination == "whisper"
        assert request.intent == "query.factual"
        assert request.content.prompt == "What is AI?"
        assert isinstance(request.request_id, UUID)
        assert isinstance(request.timestamp, datetime)

    def test_request_id_auto_generated(self):
        """Request ID should be auto-generated."""
        request = FlowRequest(
            source="user",
            destination="sage",
            intent="query.factual",
            content=RequestContent(prompt="Test"),
        )

        assert request.request_id is not None
        assert isinstance(request.request_id, UUID)

    def test_request_with_metadata(self):
        """Create request with metadata."""
        request = FlowRequest(
            source="user",
            destination="sage",
            intent="query.factual",
            content=RequestContent(
                prompt="Explain quantum computing",
                metadata=RequestMetadata(
                    user_id="user123",
                    session_id="session456",
                    requires_memory=True,
                    priority=MessagePriority.HIGH,
                ),
            ),
        )

        assert request.content.metadata.user_id == "user123"
        assert request.content.metadata.requires_memory is True
        assert request.content.metadata.priority == MessagePriority.HIGH

    def test_request_with_context(self):
        """Create request with conversation context."""
        request = FlowRequest(
            source="user",
            destination="sage",
            intent="query.factual",
            content=RequestContent(
                prompt="What else?",
                context=[
                    {"role": "user", "content": "Tell me about AI"},
                    {"role": "assistant", "content": "AI is..."},
                ],
            ),
        )

        assert len(request.content.context) == 2

    def test_constitutional_check_default(self):
        """Constitutional check should default to pending."""
        request = FlowRequest(
            source="user",
            destination="sage",
            intent="query.factual",
            content=RequestContent(prompt="Test"),
        )

        assert request.constitutional_check.status == CheckStatus.PENDING

    def test_is_approved(self):
        """Check if request is approved."""
        request = FlowRequest(
            source="user",
            destination="sage",
            intent="query.factual",
            content=RequestContent(prompt="Test"),
            constitutional_check=ConstitutionalCheck(
                status=CheckStatus.APPROVED,
            ),
        )

        assert request.is_approved() is True

    def test_add_hop(self):
        """Track agent hops."""
        request = FlowRequest(
            source="user",
            destination="whisper",
            intent="query.factual",
            content=RequestContent(prompt="Test"),
        )

        assert request.hop_count == 0
        assert len(request.route_history) == 0

        request.add_hop("whisper")
        request.add_hop("sage")

        assert request.hop_count == 2
        assert request.route_history == ["whisper", "sage"]

    def test_create_response_from_request(self):
        """Create response from a request."""
        request = FlowRequest(
            source="user",
            destination="sage",
            intent="query.factual",
            content=RequestContent(prompt="Test"),
        )

        response = request.create_response(
            source="sage",
            status=MessageStatus.SUCCESS,
            output="This is the answer",
        )

        assert response.request_id == request.request_id
        assert response.source == "sage"
        assert response.destination == "user"
        assert response.status == MessageStatus.SUCCESS

    def test_agent_name_validation(self):
        """Agent names should be normalized and validated."""
        request = FlowRequest(
            source="  USER  ",
            destination="  SAGE  ",
            intent="query",
            content=RequestContent(prompt="Test"),
        )

        assert request.source == "user"
        assert request.destination == "sage"

    def test_empty_agent_name_fails(self):
        """Empty agent names should fail validation."""
        with pytest.raises(ValueError):
            FlowRequest(
                source="",
                destination="sage",
                intent="query",
                content=RequestContent(prompt="Test"),
            )

    def test_json_serialization(self):
        """Request should serialize to JSON."""
        request = FlowRequest(
            source="user",
            destination="sage",
            intent="query.factual",
            content=RequestContent(prompt="Test"),
        )

        json_str = request.model_dump_json()
        assert isinstance(json_str, str)
        assert "user" in json_str
        assert "sage" in json_str


class TestFlowResponse:
    """Tests for FlowResponse model."""

    def test_create_success_response(self):
        """Create a success response."""
        response = FlowResponse(
            request_id="550e8400-e29b-41d4-a716-446655440000",
            source="sage",
            destination="user",
            status=MessageStatus.SUCCESS,
            content=ResponseContent(output="The answer is 42"),
        )

        assert response.status == MessageStatus.SUCCESS
        assert response.content.output == "The answer is 42"
        assert response.is_success() is True
        assert response.is_error() is False

    def test_create_error_response(self):
        """Create an error response."""
        response = FlowResponse(
            request_id="550e8400-e29b-41d4-a716-446655440000",
            source="sage",
            destination="user",
            status=MessageStatus.ERROR,
            content=ResponseContent(
                output="",
                errors=["Model not available"],
            ),
        )

        assert response.is_error() is True
        assert len(response.content.errors) == 1

    def test_create_refused_response(self):
        """Create a refused response (constitutional violation)."""
        response = FlowResponse(
            request_id="550e8400-e29b-41d4-a716-446655440000",
            source="smith",
            destination="user",
            status=MessageStatus.REFUSED,
            content=ResponseContent(
                output="Request violates constitutional boundaries",
                reasoning="Rule S1: Role boundary violation",
            ),
        )

        assert response.was_refused() is True
        assert response.content.reasoning is not None

    def test_response_with_metadata(self):
        """Response with inference metadata."""
        response = FlowResponse(
            request_id="550e8400-e29b-41d4-a716-446655440000",
            source="sage",
            destination="user",
            status=MessageStatus.SUCCESS,
            content=ResponseContent(
                output="Answer",
                metadata=ResponseMetadata(
                    model_used="mistral-7b-instruct",
                    tokens_consumed=150,
                    inference_time_ms=450,
                ),
            ),
        )

        assert response.content.metadata.model_used == "mistral-7b-instruct"
        assert response.content.metadata.tokens_consumed == 150

    def test_response_with_memory_request(self):
        """Response with memory storage request."""
        response = FlowResponse(
            request_id="550e8400-e29b-41d4-a716-446655440000",
            source="seshat",
            destination="whisper",
            status=MessageStatus.SUCCESS,
            content=ResponseContent(output="Stored"),
            memory_request=MemoryRequest(
                requested=True,
                content="User preference: dark mode",
                justification="User explicitly requested to remember",
                consent_required=True,
            ),
        )

        assert response.memory_request.requested is True
        assert response.memory_request.consent_required is True

    def test_response_confidence(self):
        """Response confidence should be validated."""
        response = FlowResponse(
            request_id="550e8400-e29b-41d4-a716-446655440000",
            source="sage",
            destination="user",
            status=MessageStatus.SUCCESS,
            content=ResponseContent(
                output="Answer",
                confidence=0.85,
            ),
        )

        assert response.content.confidence == 0.85

    def test_invalid_confidence_fails(self):
        """Confidence outside 0-1 should fail."""
        with pytest.raises(ValueError):
            ResponseContent(output="Test", confidence=1.5)


class TestConstitutionalCheck:
    """Tests for ConstitutionalCheck model."""

    def test_default_check(self):
        """Default check should be pending."""
        check = ConstitutionalCheck()

        assert check.status == CheckStatus.PENDING
        assert check.validated_by == "smith"

    def test_approved_check(self):
        """Approved constitutional check."""
        check = ConstitutionalCheck(
            status=CheckStatus.APPROVED,
            constraints=["memory_access_restricted"],
            rule_ids=["rule-001", "rule-002"],
        )

        assert check.status == CheckStatus.APPROVED
        assert len(check.constraints) == 1
        assert len(check.rule_ids) == 2

    def test_denied_check(self):
        """Denied constitutional check."""
        check = ConstitutionalCheck(
            status=CheckStatus.DENIED,
            reason="Violates human sovereignty principle",
        )

        assert check.status == CheckStatus.DENIED
        assert "sovereignty" in check.reason


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_create_request_function(self):
        """Test create_request convenience function."""
        request = create_request(
            source="user",
            destination="sage",
            intent="query.factual",
            prompt="What is the meaning of life?",
            user_id="user123",
            priority=MessagePriority.HIGH,
        )

        assert request.source == "user"
        assert request.destination == "sage"
        assert request.content.prompt == "What is the meaning of life?"
        assert request.content.metadata.user_id == "user123"
        assert request.content.metadata.priority == MessagePriority.HIGH

    def test_create_response_function(self):
        """Test create_response convenience function."""
        request = create_request(
            source="user",
            destination="sage",
            intent="query.factual",
            prompt="Test",
        )

        response = create_response(
            request=request,
            source="sage",
            status=MessageStatus.SUCCESS,
            output="The answer",
            reasoning="Based on knowledge",
            confidence=0.95,
        )

        assert response.request_id == request.request_id
        assert response.source == "sage"
        assert response.destination == "user"
        assert response.content.confidence == 0.95


class TestDeadLetterMessage:
    """Tests for DeadLetterMessage model."""

    def test_create_dead_letter(self):
        """Create a dead letter message."""
        request = create_request(
            source="user",
            destination="sage",
            intent="query",
            prompt="Test",
        )

        dead_letter = DeadLetterMessage(
            original_message=request,
            failure_reason="No subscribers",
            destination_channel="agent.sage",
        )

        assert dead_letter.retry_count == 0
        assert dead_letter.failure_reason == "No subscribers"


class TestAuditLogEntry:
    """Tests for AuditLogEntry model."""

    def test_create_audit_entry(self):
        """Create an audit log entry."""
        entry = AuditLogEntry(
            message_type="request",
            message_id="550e8400-e29b-41d4-a716-446655440000",
            source="user",
            destination="sage",
            intent="query.factual",
            constitutional_status="approved",
        )

        assert entry.message_type == "request"
        assert entry.source == "user"
        assert isinstance(entry.timestamp, datetime)
