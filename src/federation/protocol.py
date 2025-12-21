"""
Federation Protocol

Provides inter-instance communication protocol including:
- Message types and formats
- Request/response handling
- Protocol negotiation
"""

import asyncio
import json
import logging
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Awaitable, Callable, Dict, List, Optional, Union

from .identity import Identity, PublicKey

logger = logging.getLogger(__name__)


# =============================================================================
# Message Types
# =============================================================================


class MessageType(str, Enum):
    """Federation message types."""

    # Handshake
    HELLO = "hello"
    HELLO_ACK = "hello_ack"
    HANDSHAKE_COMPLETE = "handshake_complete"

    # Identity
    IDENTITY_REQUEST = "identity_request"
    IDENTITY_RESPONSE = "identity_response"
    IDENTITY_VERIFY = "identity_verify"

    # Permission
    PERMISSION_REQUEST = "permission_request"
    PERMISSION_RESPONSE = "permission_response"
    PERMISSION_REVOKE = "permission_revoke"

    # Data
    DATA_REQUEST = "data_request"
    DATA_RESPONSE = "data_response"
    DATA_PUSH = "data_push"

    # Action
    ACTION_REQUEST = "action_request"
    ACTION_RESPONSE = "action_response"

    # Control
    PING = "ping"
    PONG = "pong"
    ERROR = "error"
    DISCONNECT = "disconnect"

    # Sync
    SYNC_REQUEST = "sync_request"
    SYNC_RESPONSE = "sync_response"


class MessagePriority(str, Enum):
    """Message priority levels."""

    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"


class ErrorCode(str, Enum):
    """Protocol error codes."""

    UNKNOWN_ERROR = "unknown_error"
    INVALID_MESSAGE = "invalid_message"
    AUTHENTICATION_FAILED = "authentication_failed"
    PERMISSION_DENIED = "permission_denied"
    NOT_FOUND = "not_found"
    TIMEOUT = "timeout"
    PROTOCOL_ERROR = "protocol_error"
    INTERNAL_ERROR = "internal_error"


# =============================================================================
# Message Models
# =============================================================================


@dataclass
class FederationMessage:
    """Federation protocol message."""

    message_id: str
    message_type: MessageType
    sender_id: str
    recipient_id: str
    payload: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.utcnow)
    correlation_id: Optional[str] = None  # For request/response matching
    priority: MessagePriority = MessagePriority.NORMAL
    encrypted: bool = False
    signature: Optional[bytes] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        message_type: MessageType,
        sender_id: str,
        recipient_id: str,
        payload: Optional[Dict[str, Any]] = None,
        correlation_id: Optional[str] = None,
        priority: MessagePriority = MessagePriority.NORMAL,
    ) -> "FederationMessage":
        """Create a new message."""
        return cls(
            message_id=str(uuid.uuid4()),
            message_type=message_type,
            sender_id=sender_id,
            recipient_id=recipient_id,
            payload=payload or {},
            correlation_id=correlation_id,
            priority=priority,
        )

    @classmethod
    def create_response(
        cls,
        request: "FederationMessage",
        message_type: MessageType,
        payload: Optional[Dict[str, Any]] = None,
    ) -> "FederationMessage":
        """Create a response to a request."""
        return cls.create(
            message_type=message_type,
            sender_id=request.recipient_id,
            recipient_id=request.sender_id,
            payload=payload,
            correlation_id=request.message_id,
        )

    @classmethod
    def create_error(
        cls,
        request: "FederationMessage",
        error_code: ErrorCode,
        error_message: str,
    ) -> "FederationMessage":
        """Create an error response."""
        return cls.create_response(
            request,
            MessageType.ERROR,
            {
                "error_code": error_code.value,
                "error_message": error_message,
            },
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "message_id": self.message_id,
            "message_type": self.message_type.value,
            "sender_id": self.sender_id,
            "recipient_id": self.recipient_id,
            "payload": self.payload,
            "timestamp": self.timestamp.isoformat(),
            "correlation_id": self.correlation_id,
            "priority": self.priority.value,
            "encrypted": self.encrypted,
            "metadata": self.metadata,
        }

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FederationMessage":
        """Create from dictionary."""
        return cls(
            message_id=data["message_id"],
            message_type=MessageType(data["message_type"]),
            sender_id=data["sender_id"],
            recipient_id=data["recipient_id"],
            payload=data.get("payload", {}),
            timestamp=datetime.fromisoformat(data.get("timestamp", datetime.utcnow().isoformat())),
            correlation_id=data.get("correlation_id"),
            priority=MessagePriority(data.get("priority", "normal")),
            encrypted=data.get("encrypted", False),
            metadata=data.get("metadata", {}),
        )

    @classmethod
    def from_json(cls, json_str: str) -> "FederationMessage":
        """Create from JSON string."""
        return cls.from_dict(json.loads(json_str))


# =============================================================================
# Protocol Handler Interface
# =============================================================================


class ProtocolHandler(ABC):
    """
    Abstract base class for protocol message handlers.
    """

    @abstractmethod
    async def handle_message(
        self,
        message: FederationMessage,
    ) -> Optional[FederationMessage]:
        """
        Handle an incoming message.

        Args:
            message: Incoming message

        Returns:
            Optional response message
        """
        pass

    @abstractmethod
    def get_handled_types(self) -> List[MessageType]:
        """Get list of message types this handler processes."""
        pass


class DefaultHandler(ProtocolHandler):
    """Default handler for common message types."""

    def __init__(self, node_id: str):
        self.node_id = node_id

    async def handle_message(
        self,
        message: FederationMessage,
    ) -> Optional[FederationMessage]:
        """Handle common messages."""
        if message.message_type == MessageType.PING:
            return FederationMessage.create_response(
                message,
                MessageType.PONG,
                {"timestamp": time.time()},
            )

        return None

    def get_handled_types(self) -> List[MessageType]:
        """Get handled message types."""
        return [MessageType.PING]


# =============================================================================
# Federation Protocol
# =============================================================================


class FederationProtocol:
    """
    Federation protocol manager.

    Handles message routing, serialization, and protocol negotiation.
    """

    VERSION = "1.0"
    SUPPORTED_VERSIONS = ["1.0"]

    def __init__(
        self,
        node_id: str,
        identity: Optional[Identity] = None,
    ):
        self.node_id = node_id
        self.identity = identity

        # Message handlers
        self._handlers: Dict[MessageType, List[ProtocolHandler]] = {}
        self._default_handler = DefaultHandler(node_id)

        # Pending requests
        self._pending_requests: Dict[str, asyncio.Future] = {}

        # Message queue
        self._outgoing_queue: asyncio.Queue = asyncio.Queue()
        self._incoming_queue: asyncio.Queue = asyncio.Queue()

        # Callbacks
        self._on_message_callbacks: List[Callable[[FederationMessage], None]] = []
        self._on_error_callbacks: List[Callable[[str, Exception], None]] = []

        # Stats
        self._messages_sent = 0
        self._messages_received = 0
        self._errors = 0

    def register_handler(
        self,
        handler: ProtocolHandler,
    ) -> None:
        """Register a message handler."""
        for msg_type in handler.get_handled_types():
            if msg_type not in self._handlers:
                self._handlers[msg_type] = []
            self._handlers[msg_type].append(handler)
            logger.debug(f"Registered handler for {msg_type.value}")

    def unregister_handler(
        self,
        handler: ProtocolHandler,
    ) -> None:
        """Unregister a message handler."""
        for msg_type in handler.get_handled_types():
            if msg_type in self._handlers:
                self._handlers[msg_type].remove(handler)

    def on_message(
        self,
        callback: Callable[[FederationMessage], None],
    ) -> None:
        """Register callback for all incoming messages."""
        self._on_message_callbacks.append(callback)

    def on_error(
        self,
        callback: Callable[[str, Exception], None],
    ) -> None:
        """Register callback for errors."""
        self._on_error_callbacks.append(callback)

    async def send_message(
        self,
        message: FederationMessage,
    ) -> None:
        """Send a message."""
        await self._outgoing_queue.put(message)
        self._messages_sent += 1
        logger.debug(f"Queued message {message.message_id} to {message.recipient_id}")

    async def send_request(
        self,
        message: FederationMessage,
        timeout: float = 30.0,
    ) -> FederationMessage:
        """
        Send a request and wait for response.

        Args:
            message: Request message
            timeout: Timeout in seconds

        Returns:
            Response message

        Raises:
            asyncio.TimeoutError: If no response within timeout
        """
        # Create future for response
        future = asyncio.get_event_loop().create_future()
        self._pending_requests[message.message_id] = future

        try:
            await self.send_message(message)
            response = await asyncio.wait_for(future, timeout)
            return response
        finally:
            self._pending_requests.pop(message.message_id, None)

    async def receive_message(
        self,
        message: FederationMessage,
    ) -> Optional[FederationMessage]:
        """
        Process an incoming message.

        Args:
            message: Incoming message

        Returns:
            Optional response message
        """
        self._messages_received += 1

        # Notify callbacks
        for callback in self._on_message_callbacks:
            try:
                callback(message)
            except Exception as e:
                logger.error(f"Message callback error: {e}")

        # Check if this is a response to a pending request
        if message.correlation_id and message.correlation_id in self._pending_requests:
            future = self._pending_requests.pop(message.correlation_id)
            if not future.done():
                future.set_result(message)
            return None

        # Route to handlers
        response = await self._route_message(message)
        return response

    async def _route_message(
        self,
        message: FederationMessage,
    ) -> Optional[FederationMessage]:
        """Route message to appropriate handlers."""
        handlers = self._handlers.get(message.message_type, [])

        if not handlers:
            # Try default handler
            response = await self._default_handler.handle_message(message)
            if response:
                return response

            # Unknown message type
            logger.warning(f"No handler for message type: {message.message_type.value}")
            return FederationMessage.create_error(
                message,
                ErrorCode.PROTOCOL_ERROR,
                f"Unknown message type: {message.message_type.value}",
            )

        # Call handlers (first response wins)
        for handler in handlers:
            try:
                response = await handler.handle_message(message)
                if response:
                    return response
            except Exception as e:
                logger.error(f"Handler error: {e}")
                self._errors += 1

        return None

    async def get_outgoing_message(self) -> FederationMessage:
        """Get next outgoing message from queue."""
        return await self._outgoing_queue.get()

    def create_hello(
        self,
        recipient_id: str,
    ) -> FederationMessage:
        """Create a hello message for handshake."""
        return FederationMessage.create(
            MessageType.HELLO,
            self.node_id,
            recipient_id,
            {
                "protocol_version": self.VERSION,
                "supported_versions": self.SUPPORTED_VERSIONS,
                "capabilities": self._get_capabilities(),
                "identity": self.identity.to_dict() if self.identity else None,
            },
        )

    def create_hello_ack(
        self,
        request: FederationMessage,
    ) -> FederationMessage:
        """Create hello acknowledgment."""
        return FederationMessage.create_response(
            request,
            MessageType.HELLO_ACK,
            {
                "protocol_version": self.VERSION,
                "accepted": True,
                "capabilities": self._get_capabilities(),
                "identity": self.identity.to_dict() if self.identity else None,
            },
        )

    def _get_capabilities(self) -> List[str]:
        """Get node capabilities."""
        return [
            "messaging",
            "data_sync",
            "permission_negotiation",
            "encryption",
        ]

    def get_stats(self) -> Dict[str, Any]:
        """Get protocol statistics."""
        return {
            "messages_sent": self._messages_sent,
            "messages_received": self._messages_received,
            "pending_requests": len(self._pending_requests),
            "errors": self._errors,
        }


# =============================================================================
# Factory Function
# =============================================================================


def create_protocol(
    node_id: str,
    identity: Optional[Identity] = None,
) -> FederationProtocol:
    """
    Create a federation protocol instance.

    Args:
        node_id: Local node ID
        identity: Local identity

    Returns:
        FederationProtocol instance
    """
    return FederationProtocol(node_id, identity)
