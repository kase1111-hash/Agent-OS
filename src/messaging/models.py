"""
Agent OS Message Protocol Models

Defines FlowRequest and FlowResponse schemas for inter-agent communication.
All messages in Agent OS conform to these schemas.
"""

from datetime import datetime
from enum import Enum
from typing import Optional, List, Dict, Any, Union
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, field_validator, ConfigDict


class MessagePriority(int, Enum):
    """Message priority levels (1-10, higher = more urgent)."""
    LOWEST = 1
    LOW = 3
    NORMAL = 5
    HIGH = 7
    URGENT = 9
    CRITICAL = 10


class CheckStatus(str, Enum):
    """Constitutional check status."""
    APPROVED = "approved"
    DENIED = "denied"
    CONDITIONAL = "conditional"
    PENDING = "pending"


class MessageStatus(str, Enum):
    """Response status."""
    SUCCESS = "success"
    PARTIAL = "partial"
    REFUSED = "refused"
    ERROR = "error"


class RequestMetadata(BaseModel):
    """Metadata for a flow request."""
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    requires_memory: bool = False
    priority: MessagePriority = MessagePriority.NORMAL
    tags: List[str] = Field(default_factory=list)
    correlation_id: Optional[str] = None  # For tracking related messages
    ttl_seconds: Optional[int] = None  # Time-to-live for message


class ConstitutionalCheck(BaseModel):
    """Result of constitutional validation by Guardian (Smith)."""
    validated_by: str = "smith"
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    status: CheckStatus = CheckStatus.PENDING
    constraints: List[str] = Field(default_factory=list)
    rule_ids: List[str] = Field(default_factory=list)  # IDs of applicable rules
    reason: Optional[str] = None


class RequestContent(BaseModel):
    """Content of a flow request."""
    prompt: str
    context: List[Dict[str, Any]] = Field(default_factory=list)
    metadata: RequestMetadata = Field(default_factory=RequestMetadata)
    attachments: List[Dict[str, Any]] = Field(default_factory=list)


class FlowRequest(BaseModel):
    """
    Standard request message schema for inter-agent communication.

    Every request to an agent MUST conform to this schema.
    """
    request_id: UUID = Field(default_factory=uuid4)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    source: str  # Agent name or 'user'
    destination: str  # Target agent name
    intent: str  # Classified intent type
    content: RequestContent
    constitutional_check: ConstitutionalCheck = Field(
        default_factory=ConstitutionalCheck
    )
    # Routing metadata
    hop_count: int = 0  # Number of agents this request has passed through
    route_history: List[str] = Field(default_factory=list)  # Path taken

    @field_validator('source', 'destination')
    @classmethod
    def validate_agent_name(cls, v: str) -> str:
        """Validate agent names are non-empty."""
        if not v or not v.strip():
            raise ValueError("Agent name cannot be empty")
        return v.strip().lower()

    @field_validator('intent')
    @classmethod
    def validate_intent(cls, v: str) -> str:
        """Validate intent is non-empty."""
        if not v or not v.strip():
            raise ValueError("Intent cannot be empty")
        return v.strip()

    def is_approved(self) -> bool:
        """Check if request has been approved by constitutional check."""
        return self.constitutional_check.status == CheckStatus.APPROVED

    def add_hop(self, agent_name: str) -> None:
        """Record that this request passed through an agent."""
        self.hop_count += 1
        self.route_history.append(agent_name)

    def create_response(
        self,
        source: str,
        status: MessageStatus,
        output: Union[str, Dict[str, Any]],
        **kwargs
    ) -> "FlowResponse":
        """Create a response to this request."""
        return FlowResponse(
            request_id=self.request_id,
            source=source,
            destination=self.source,  # Send back to requester
            status=status,
            content=ResponseContent(output=output, **kwargs),
        )

    model_config = ConfigDict(
        ser_json_timedelta='iso8601',
    )


class ResponseMetadata(BaseModel):
    """Metadata about how a response was generated."""
    model_used: Optional[str] = None
    tokens_consumed: Optional[int] = None
    inference_time_ms: Optional[int] = None
    cache_hit: bool = False


class ResponseContent(BaseModel):
    """Content of a flow response."""
    output: Union[str, Dict[str, Any]]
    reasoning: Optional[str] = None
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    metadata: ResponseMetadata = Field(default_factory=ResponseMetadata)
    errors: List[str] = Field(default_factory=list)


class MemoryRequest(BaseModel):
    """Request to store information in memory."""
    requested: bool = False
    content: Optional[Union[str, Dict[str, Any]]] = None
    justification: Optional[str] = None
    consent_required: bool = True
    retention_days: Optional[int] = None  # How long to keep
    memory_type: str = "ephemeral"  # ephemeral, working, long_term


class FlowResponse(BaseModel):
    """
    Standard response message schema for inter-agent communication.

    Every agent response MUST conform to this schema.
    """
    response_id: UUID = Field(default_factory=uuid4)
    request_id: UUID  # Reference to original request
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    source: str  # Agent that generated this response
    destination: str  # Where to send response
    status: MessageStatus
    content: ResponseContent
    next_actions: List[Dict[str, Any]] = Field(default_factory=list)
    memory_request: MemoryRequest = Field(default_factory=MemoryRequest)

    @field_validator('source', 'destination')
    @classmethod
    def validate_agent_name(cls, v: str) -> str:
        """Validate agent names are non-empty."""
        if not v or not v.strip():
            raise ValueError("Agent name cannot be empty")
        return v.strip().lower()

    def is_success(self) -> bool:
        """Check if response indicates success."""
        return self.status == MessageStatus.SUCCESS

    def is_error(self) -> bool:
        """Check if response indicates an error."""
        return self.status == MessageStatus.ERROR

    def was_refused(self) -> bool:
        """Check if request was refused (constitutional violation)."""
        return self.status == MessageStatus.REFUSED

    model_config = ConfigDict(
        ser_json_timedelta='iso8601',
    )


class DeadLetterMessage(BaseModel):
    """A message that failed delivery and was sent to the dead letter queue."""
    original_message: Union[FlowRequest, FlowResponse]
    failure_reason: str
    failure_timestamp: datetime = Field(default_factory=datetime.utcnow)
    retry_count: int = 0
    last_error: Optional[str] = None
    destination_channel: str


class AuditLogEntry(BaseModel):
    """Audit log entry for message tracking."""
    entry_id: UUID = Field(default_factory=uuid4)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    message_type: str  # "request" or "response"
    message_id: UUID
    source: str
    destination: str
    intent: Optional[str] = None
    status: Optional[str] = None
    constitutional_status: Optional[str] = None
    processing_time_ms: Optional[int] = None
    error: Optional[str] = None


def create_request(
    source: str,
    destination: str,
    intent: str,
    prompt: str,
    **kwargs
) -> FlowRequest:
    """
    Convenience function to create a FlowRequest.

    Args:
        source: Source agent or 'user'
        destination: Target agent
        intent: Classified intent type
        prompt: The actual request text
        **kwargs: Additional metadata

    Returns:
        FlowRequest instance
    """
    metadata = RequestMetadata(**{k: v for k, v in kwargs.items()
                                  if k in RequestMetadata.model_fields})

    context = kwargs.get('context', [])

    return FlowRequest(
        source=source,
        destination=destination,
        intent=intent,
        content=RequestContent(
            prompt=prompt,
            context=context,
            metadata=metadata,
        ),
    )


def create_response(
    request: FlowRequest,
    source: str,
    status: MessageStatus,
    output: Union[str, Dict[str, Any]],
    **kwargs
) -> FlowResponse:
    """
    Convenience function to create a FlowResponse.

    Args:
        request: The original request
        source: Agent generating response
        status: Response status
        output: Response content
        **kwargs: Additional response data

    Returns:
        FlowResponse instance
    """
    return FlowResponse(
        request_id=request.request_id,
        source=source,
        destination=request.source,
        status=status,
        content=ResponseContent(
            output=output,
            reasoning=kwargs.get('reasoning'),
            confidence=kwargs.get('confidence', 1.0),
        ),
        next_actions=kwargs.get('next_actions', []),
        memory_request=MemoryRequest(**{
            k: v for k, v in kwargs.items()
            if k in MemoryRequest.model_fields
        }),
    )
