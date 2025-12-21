"""
Chat API Routes

WebSocket-based chat interface for Agent OS.
"""

import asyncio
import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set

from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)
router = APIRouter()


# =============================================================================
# Models
# =============================================================================


class MessageRole(str, Enum):
    """Message role in conversation."""

    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class ChatMessage(BaseModel):
    """A chat message."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    role: MessageRole
    content: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ChatRequest(BaseModel):
    """Request to send a chat message."""

    message: str
    conversation_id: Optional[str] = None
    context: Optional[Dict[str, Any]] = None


class ChatResponse(BaseModel):
    """Response from chat."""

    message: ChatMessage
    conversation_id: str
    processing_time_ms: Optional[int] = None


class ConversationSummary(BaseModel):
    """Summary of a conversation."""

    id: str
    title: str
    message_count: int
    created_at: datetime
    updated_at: datetime


class StreamChunk(BaseModel):
    """A chunk of streaming response."""

    type: str  # "content", "thinking", "done", "error"
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


# =============================================================================
# Connection Manager
# =============================================================================


@dataclass
class Connection:
    """A WebSocket connection."""

    id: str
    websocket: WebSocket
    user_id: Optional[str] = None
    conversation_id: Optional[str] = None
    connected_at: datetime = field(default_factory=datetime.utcnow)
    last_activity: datetime = field(default_factory=datetime.utcnow)


class ConnectionManager:
    """Manages WebSocket connections."""

    def __init__(self):
        self.connections: Dict[str, Connection] = {}
        self.conversations: Dict[str, List[ChatMessage]] = {}
        self._message_handlers: List[Callable] = []

    async def connect(
        self,
        websocket: WebSocket,
        connection_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> Connection:
        """Accept a new WebSocket connection."""
        await websocket.accept()

        conn_id = connection_id or str(uuid.uuid4())
        connection = Connection(
            id=conn_id,
            websocket=websocket,
            user_id=user_id,
        )
        self.connections[conn_id] = connection

        logger.info(f"WebSocket connection established: {conn_id}")
        return connection

    async def disconnect(self, connection_id: str) -> None:
        """Close and remove a connection."""
        if connection_id in self.connections:
            connection = self.connections.pop(connection_id)
            try:
                await connection.websocket.close()
            except Exception:
                pass
            logger.info(f"WebSocket connection closed: {connection_id}")

    async def send_message(
        self,
        connection_id: str,
        message: Dict[str, Any],
    ) -> bool:
        """Send a message to a specific connection."""
        if connection_id not in self.connections:
            return False

        connection = self.connections[connection_id]
        try:
            await connection.websocket.send_json(message)
            connection.last_activity = datetime.utcnow()
            return True
        except Exception as e:
            logger.error(f"Failed to send message to {connection_id}: {e}")
            return False

    async def broadcast(
        self,
        message: Dict[str, Any],
        exclude: Optional[Set[str]] = None,
    ) -> int:
        """Broadcast a message to all connections."""
        exclude = exclude or set()
        sent_count = 0

        for conn_id in list(self.connections.keys()):
            if conn_id in exclude:
                continue
            if await self.send_message(conn_id, message):
                sent_count += 1

        return sent_count

    def get_connection(self, connection_id: str) -> Optional[Connection]:
        """Get a connection by ID."""
        return self.connections.get(connection_id)

    def get_conversation(self, conversation_id: str) -> List[ChatMessage]:
        """Get conversation history."""
        if conversation_id not in self.conversations:
            self.conversations[conversation_id] = []
        return self.conversations[conversation_id]

    def add_message(
        self,
        conversation_id: str,
        message: ChatMessage,
    ) -> None:
        """Add a message to a conversation."""
        if conversation_id not in self.conversations:
            self.conversations[conversation_id] = []
        self.conversations[conversation_id].append(message)

    def register_handler(self, handler: Callable) -> None:
        """Register a message handler."""
        self._message_handlers.append(handler)

    async def process_message(
        self,
        connection_id: str,
        message: str,
        conversation_id: str,
    ) -> ChatMessage:
        """
        Process an incoming chat message.

        This is a placeholder that simulates agent processing.
        In production, this would route to Whisper/Smith/agents.
        """
        # Create user message
        user_message = ChatMessage(
            role=MessageRole.USER,
            content=message,
        )
        self.add_message(conversation_id, user_message)

        # Simulate processing time
        await asyncio.sleep(0.1)

        # Generate response (placeholder)
        # In production, this would invoke the agent system
        response_content = self._generate_response(message, conversation_id)

        assistant_message = ChatMessage(
            role=MessageRole.ASSISTANT,
            content=response_content,
            metadata={"model": "agent-os", "tokens": len(response_content.split())},
        )
        self.add_message(conversation_id, assistant_message)

        return assistant_message

    def _generate_response(self, message: str, conversation_id: str) -> str:
        """
        Generate a response to a message.

        This is a placeholder. In production, this would:
        1. Route through Whisper for intent classification
        2. Pass through Smith for constitutional checking
        3. Execute via appropriate agent(s)
        4. Return aggregated response
        """
        # Simple echo response for demonstration
        history = self.get_conversation(conversation_id)
        turn_count = len([m for m in history if m.role == MessageRole.USER])

        return (
            f"I received your message: '{message}'\n\n"
            f"This is turn {turn_count} in our conversation.\n\n"
            "Note: This is a placeholder response. In production, "
            "this would be processed by the Agent OS agent system."
        )


# Global connection manager
_manager: Optional[ConnectionManager] = None


def get_manager() -> ConnectionManager:
    """Get the connection manager."""
    global _manager
    if _manager is None:
        _manager = ConnectionManager()
    return _manager


# =============================================================================
# WebSocket Endpoint
# =============================================================================


@router.websocket("/ws")
async def chat_websocket(
    websocket: WebSocket,
    conversation_id: Optional[str] = None,
):
    """
    WebSocket endpoint for real-time chat.

    Protocol:
    - Send: {"type": "message", "content": "Hello"}
    - Receive: {"type": "response", "message": {...}}
    - Receive: {"type": "stream", "chunk": {...}}  # For streaming
    - Receive: {"type": "error", "message": "..."}
    """
    manager = get_manager()
    connection = await manager.connect(websocket)

    # Use provided conversation_id or create new one
    conv_id = conversation_id or str(uuid.uuid4())
    connection.conversation_id = conv_id

    # Send connection acknowledgment
    await manager.send_message(
        connection.id,
        {
            "type": "connected",
            "connection_id": connection.id,
            "conversation_id": conv_id,
        },
    )

    try:
        while True:
            # Receive message
            data = await websocket.receive_json()
            msg_type = data.get("type", "message")

            if msg_type == "message":
                content = data.get("content", "")
                if not content:
                    await manager.send_message(
                        connection.id,
                        {"type": "error", "message": "Empty message"},
                    )
                    continue

                # Process message
                try:
                    response = await manager.process_message(
                        connection.id,
                        content,
                        conv_id,
                    )

                    await manager.send_message(
                        connection.id,
                        {
                            "type": "response",
                            "message": {
                                "id": response.id,
                                "role": response.role.value,
                                "content": response.content,
                                "timestamp": response.timestamp.isoformat(),
                                "metadata": response.metadata,
                            },
                        },
                    )
                except Exception as e:
                    logger.error(f"Error processing message: {e}")
                    await manager.send_message(
                        connection.id,
                        {"type": "error", "message": str(e)},
                    )

            elif msg_type == "ping":
                await manager.send_message(
                    connection.id,
                    {"type": "pong", "timestamp": datetime.utcnow().isoformat()},
                )

            elif msg_type == "history":
                # Send conversation history
                history = manager.get_conversation(conv_id)
                await manager.send_message(
                    connection.id,
                    {
                        "type": "history",
                        "messages": [
                            {
                                "id": m.id,
                                "role": m.role.value,
                                "content": m.content,
                                "timestamp": m.timestamp.isoformat(),
                            }
                            for m in history
                        ],
                    },
                )

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: {connection.id}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        await manager.disconnect(connection.id)


# =============================================================================
# REST Endpoints
# =============================================================================


@router.post("/send", response_model=ChatResponse)
async def send_message(
    request: ChatRequest,
    manager: ConnectionManager = Depends(get_manager),
) -> ChatResponse:
    """
    Send a message via REST API.

    For simple request/response without WebSocket.
    """
    import time

    start_time = time.time()

    conversation_id = request.conversation_id or str(uuid.uuid4())

    # Create a temporary connection ID for processing
    temp_conn_id = f"rest-{uuid.uuid4()}"

    response = await manager.process_message(
        temp_conn_id,
        request.message,
        conversation_id,
    )

    processing_time = int((time.time() - start_time) * 1000)

    return ChatResponse(
        message=response,
        conversation_id=conversation_id,
        processing_time_ms=processing_time,
    )


@router.get("/conversations", response_model=List[ConversationSummary])
async def list_conversations(
    limit: int = 20,
    manager: ConnectionManager = Depends(get_manager),
) -> List[ConversationSummary]:
    """List recent conversations."""
    summaries = []

    for conv_id, messages in manager.conversations.items():
        if not messages:
            continue

        # Get first user message as title
        user_messages = [m for m in messages if m.role == MessageRole.USER]
        title = user_messages[0].content[:50] if user_messages else "Untitled"
        if len(title) == 50:
            title += "..."

        summaries.append(
            ConversationSummary(
                id=conv_id,
                title=title,
                message_count=len(messages),
                created_at=messages[0].timestamp,
                updated_at=messages[-1].timestamp,
            )
        )

    # Sort by updated_at descending
    summaries.sort(key=lambda x: x.updated_at, reverse=True)

    return summaries[:limit]


@router.get("/conversations/{conversation_id}", response_model=List[ChatMessage])
async def get_conversation(
    conversation_id: str,
    manager: ConnectionManager = Depends(get_manager),
) -> List[ChatMessage]:
    """Get messages from a conversation."""
    messages = manager.get_conversation(conversation_id)
    return messages


@router.delete("/conversations/{conversation_id}")
async def delete_conversation(
    conversation_id: str,
    manager: ConnectionManager = Depends(get_manager),
) -> Dict[str, str]:
    """Delete a conversation."""
    if conversation_id in manager.conversations:
        del manager.conversations[conversation_id]
        return {"status": "deleted", "conversation_id": conversation_id}
    raise HTTPException(status_code=404, detail="Conversation not found")


@router.get("/status")
async def get_chat_status(
    manager: ConnectionManager = Depends(get_manager),
) -> Dict[str, Any]:
    """Get chat system status."""
    return {
        "active_connections": len(manager.connections),
        "total_conversations": len(manager.conversations),
        "status": "operational",
    }
