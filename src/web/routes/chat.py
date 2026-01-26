"""
Chat API Routes

WebSocket-based chat interface for Agent OS.
Integrates with Ollama for local LLM inference.
"""

import asyncio
import concurrent.futures
import json
import logging
import os
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field

from src.boundary import BoundaryMode, SmithClient, create_smith_client
from src.web.conversation_store import (
    Conversation,
    ConversationStore,
)
from src.web.conversation_store import MessageRole as StoredMessageRole
from src.web.conversation_store import (
    StoredMessage,
    get_conversation_store,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Security: Endpoint Validation
# =============================================================================

import ipaddress
from urllib.parse import urlparse


class SSRFProtectionError(Exception):
    """Raised when a URL fails SSRF protection validation."""

    pass


def validate_ollama_endpoint(url: str) -> str:
    """
    Validate an Ollama endpoint URL to prevent SSRF attacks.

    Args:
        url: The URL to validate

    Returns:
        The validated URL

    Raises:
        SSRFProtectionError: If the URL fails validation
    """
    if not url:
        raise SSRFProtectionError("Empty URL")

    try:
        parsed = urlparse(url)
    except Exception as e:
        raise SSRFProtectionError(f"Invalid URL format: {e}")

    # Validate scheme
    if parsed.scheme not in ("http", "https"):
        raise SSRFProtectionError(f"Invalid URL scheme: {parsed.scheme}")

    # Validate host exists
    if not parsed.hostname:
        raise SSRFProtectionError("URL has no hostname")

    hostname = parsed.hostname.lower()

    # Check for suspicious internal hostnames
    suspicious_patterns = [
        "metadata.",  # Cloud metadata services
        "169.254.",  # AWS metadata IP range
        "internal.",  # Internal services
    ]

    for pattern in suspicious_patterns:
        if pattern in hostname:
            raise SSRFProtectionError(f"Suspicious hostname pattern: {hostname}")

    # Reject .internal, .local, .corp, .lan TLDs (except localhost)
    if hostname not in ("localhost", "127.0.0.1", "::1"):
        suspicious_tlds = [".internal", ".local", ".corp", ".lan"]
        for tld in suspicious_tlds:
            if hostname.endswith(tld):
                raise SSRFProtectionError(f"Suspicious hostname pattern: {hostname}")

    # Check for IP addresses
    try:
        ip = ipaddress.ip_address(hostname)
        # Allow localhost/loopback for local Ollama
        if not ip.is_loopback:
            if ip.is_private:
                # Allow common private network ranges for local Ollama deployments
                # but block link-local, multicast, reserved
                pass
            if ip.is_link_local or ip.is_multicast or ip.is_reserved:
                raise SSRFProtectionError(f"Reserved IP address not allowed: {hostname}")
    except ValueError:
        # Not an IP address, it's a hostname - that's fine
        pass

    return url


# Ollama configuration with SSRF protection
_raw_ollama_endpoint = os.environ.get("OLLAMA_ENDPOINT", "http://localhost:11434")
try:
    OLLAMA_ENDPOINT = validate_ollama_endpoint(_raw_ollama_endpoint)
except SSRFProtectionError as e:
    logger.error(f"Invalid OLLAMA_ENDPOINT configuration: {e}")
    OLLAMA_ENDPOINT = "http://localhost:11434"  # Fall back to safe default

OLLAMA_TIMEOUT = float(os.environ.get("OLLAMA_TIMEOUT", "120"))

# System prompt for the AI assistant
SYSTEM_PROMPT = """You are Agent OS, a helpful AI assistant running locally on the user's machine.
You are powered by Ollama and respect user privacy - all conversations stay on this device.
Be concise, helpful, and friendly. If you don't know something, say so honestly."""


class ModelManager:
    """Manages the current Ollama model and available models."""

    def __init__(self):
        self.current_model = os.environ.get("OLLAMA_MODEL", "mistral")
        self._available_models: List[str] = []
        self._last_refresh = None

    def get_current_model(self) -> str:
        return self.current_model

    def set_model(self, model_name: str) -> bool:
        """Set the current model. Returns True if model exists."""
        available = self.get_available_models()
        # Check exact match or partial match
        for m in available:
            if m == model_name or m.startswith(model_name + ":"):
                self.current_model = m
                return True
        # If not found but user specified it, try anyway
        if model_name:
            self.current_model = model_name
            return True
        return False

    def get_available_models(self) -> List[str]:
        """Get list of available models from Ollama."""
        try:
            from src.agents.ollama import OllamaClient

            client = OllamaClient(endpoint=OLLAMA_ENDPOINT, timeout=5)
            models = client.list_models()
            client.close()
            self._available_models = [m.name for m in models]
            return self._available_models
        except Exception as e:
            logger.error(f"Failed to get models: {e}")
            return self._available_models  # Return cached list

    def find_model(self, query: str) -> Optional[str]:
        """Find a model matching the query."""
        query_lower = query.lower().strip()
        available = self.get_available_models()

        # Exact match
        for m in available:
            if m.lower() == query_lower:
                return m

        # Partial match (model name starts with query)
        for m in available:
            if m.lower().startswith(query_lower):
                return m

        # Fuzzy match (query is contained in model name)
        for m in available:
            if query_lower in m.lower():
                return m

        return None


# Global model manager
_model_manager: Optional[ModelManager] = None


def get_model_manager() -> ModelManager:
    """Get the model manager instance."""
    global _model_manager
    if _model_manager is None:
        _model_manager = ModelManager()
    return _model_manager


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
    """Manages WebSocket connections with persistent conversation storage."""

    def __init__(
        self,
        store: Optional[ConversationStore] = None,
        boundary_client: Optional[SmithClient] = None,
    ):
        self.connections: Dict[str, Connection] = {}
        self._conversations_cache: Dict[str, List[ChatMessage]] = {}  # In-memory cache
        self._message_handlers: List[Callable] = []
        self._store = store  # Persistent storage
        self._boundary_client = boundary_client  # Boundary daemon client

    @property
    def store(self) -> ConversationStore:
        """Get the conversation store, initializing if needed."""
        if self._store is None:
            self._store = get_conversation_store()
        return self._store

    @property
    def boundary(self) -> SmithClient:
        """Get the boundary client, initializing if needed."""
        if self._boundary_client is None:
            # Allow network access for local Ollama by default
            # Can be disabled via AGENT_OS_BOUNDARY_NETWORK_ALLOWED=false
            network_allowed = os.environ.get(
                "AGENT_OS_BOUNDARY_NETWORK_ALLOWED", "true"
            ).lower() in ("true", "1", "yes")

            self._boundary_client = create_smith_client(
                embedded=True,
                initial_mode=BoundaryMode.RESTRICTED,
                fail_closed=True,
                network_allowed=network_allowed,
            )
        return self._boundary_client

    @property
    def conversations(self) -> Dict[str, List[ChatMessage]]:
        """Property for backwards compatibility - returns cached conversations."""
        return self._conversations_cache

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
            except Exception as e:
                logger.debug(f"Error closing websocket {connection_id}: {e}")
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
        """Get conversation history from cache or load from store."""
        # Check boundary permission for memory access
        if not self.boundary.check_memory_access("chat:conversation", conversation_id):
            logger.warning(
                f"Boundary daemon denied memory access to conversation {conversation_id}"
            )
            return []

        # Check cache first
        if conversation_id in self._conversations_cache:
            return self._conversations_cache[conversation_id]

        # Load from persistent store
        try:
            stored_messages = self.store.get_messages(conversation_id)
            messages = [
                ChatMessage(
                    id=m.id,
                    role=MessageRole(m.role.value),
                    content=m.content,
                    timestamp=m.timestamp,
                    metadata=m.metadata,
                )
                for m in stored_messages
            ]
            self._conversations_cache[conversation_id] = messages
            return messages
        except Exception as e:
            logger.warning(f"Failed to load conversation {conversation_id}: {e}")
            self._conversations_cache[conversation_id] = []
            return self._conversations_cache[conversation_id]

    def add_message(
        self,
        conversation_id: str,
        message: ChatMessage,
    ) -> None:
        """Add a message to a conversation (both cache and persistent store)."""
        # Add to cache
        if conversation_id not in self._conversations_cache:
            self._conversations_cache[conversation_id] = []
        self._conversations_cache[conversation_id].append(message)

        # Persist to store
        try:
            self.store.add_message(
                conversation_id=conversation_id,
                role=StoredMessageRole(message.role.value),
                content=message.content,
                message_id=message.id,
                metadata=message.metadata,
            )
        except Exception as e:
            logger.error(f"Failed to persist message: {e}")

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
        Process an incoming chat message using Ollama.

        Routes the message through the local Ollama instance for LLM inference.
        Handles special commands for model management via Whisper-style intent detection.
        """
        # Create user message
        user_message = ChatMessage(
            role=MessageRole.USER,
            content=message,
        )
        self.add_message(conversation_id, user_message)

        # Check for model management commands (Whisper intent detection)
        command_response = self._handle_model_command(message)
        if command_response:
            response_content, metadata = command_response
        else:
            # Generate response using Ollama
            response_content, metadata = await self._generate_ollama_response(
                message, conversation_id
            )

        assistant_message = ChatMessage(
            role=MessageRole.ASSISTANT,
            content=response_content,
            metadata=metadata,
        )
        self.add_message(conversation_id, assistant_message)

        return assistant_message

    def _handle_model_command(self, message: str) -> Optional[Tuple[str, Dict[str, Any]]]:
        """
        Handle model management commands via Whisper-style intent detection.

        Supported commands:
        - "list models" / "show models" / "what models"
        - "use <model>" / "switch to <model>" / "change model to <model>"
        - "current model" / "what model"
        """
        msg_lower = message.lower().strip()
        model_manager = get_model_manager()

        # List models command
        list_patterns = [
            r"^list\s+models?$",
            r"^show\s+models?$",
            r"^what\s+models?\s+(are\s+)?(available|do\s+i\s+have)",
            r"^available\s+models?$",
            r"^models?\s+list$",
        ]
        for pattern in list_patterns:
            if re.match(pattern, msg_lower):
                models = model_manager.get_available_models()
                current = model_manager.get_current_model()
                if models:
                    model_list = "\n".join(
                        f"  • {m}" + (" ← current" if m == current else "") for m in models
                    )
                    response = f"**Available Models:**\n{model_list}\n\nTo switch models, say: `use <model_name>`"
                else:
                    response = "No models found. Make sure Ollama is running and you have models pulled.\nTry: `ollama pull mistral`"
                return response, {"command": "list_models", "models": models}

        # Current model command
        current_patterns = [
            r"^(what|which)\s+(is\s+)?(the\s+)?current\s+model",
            r"^current\s+model",
            r"^what\s+model\s+(am\s+i|are\s+you)\s+using",
        ]
        for pattern in current_patterns:
            if re.match(pattern, msg_lower):
                current = model_manager.get_current_model()
                return f"Currently using: **{current}**", {
                    "command": "current_model",
                    "model": current,
                }

        # Switch model command
        switch_patterns = [
            r"^(?:use|switch\s+to|change\s+(?:model\s+)?to|set\s+model\s+(?:to)?)\s+(.+)$",
            r"^model\s+(.+)$",
        ]
        for pattern in switch_patterns:
            match = re.match(pattern, msg_lower)
            if match:
                requested_model = match.group(1).strip()
                found_model = model_manager.find_model(requested_model)

                if found_model:
                    model_manager.set_model(found_model)
                    response = f"✓ Switched to model: **{found_model}**"
                    return response, {"command": "switch_model", "model": found_model}
                else:
                    available = model_manager.get_available_models()
                    if available:
                        suggestions = ", ".join(available[:5])
                        response = f"Model '{requested_model}' not found.\n\nAvailable models: {suggestions}"
                    else:
                        response = f"Model '{requested_model}' not found. No models available - is Ollama running?"
                    return response, {"command": "switch_model", "error": "not_found"}

        return None

    async def _generate_ollama_response(
        self, message: str, conversation_id: str
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Generate a response using Ollama.

        Runs the synchronous Ollama client in a thread pool to avoid blocking.
        Checks boundary permissions before making network calls.
        """
        # Update dreaming status
        dreaming = None
        try:
            from src.web.dreaming import get_dreaming

            dreaming = get_dreaming()
            dreaming.start("Processing chat request")
        except Exception as e:
            # Log but continue - dreaming is not critical
            logger.debug(f"Dreaming service unavailable: {e}")
            dreaming = None

        # Check boundary permission for network access to Ollama
        if not self.boundary.check_network_access("chat:ollama", OLLAMA_ENDPOINT):
            logger.warning("Boundary daemon denied network access to Ollama")
            if dreaming:
                dreaming.complete("Boundary denied access")
            return (
                "I'm unable to respond right now. Network access to the AI backend "
                "has been restricted by the security boundary.",
                {"error": "boundary_denied", "request_type": "network_access"},
            )

        # Build conversation history for context
        try:
            history = self.get_conversation(conversation_id)

            # Convert history to Ollama message format
            messages = []
            for msg in history[-10:]:  # Keep last 10 messages for context
                messages.append(
                    {
                        "role": msg.role.value,
                        "content": msg.content,
                    }
                )

            # Add the current message
            messages.append({"role": "user", "content": message})

            # Run Ollama in thread pool (since httpx client is synchronous)
            loop = asyncio.get_event_loop()

            # Update dreaming status to running
            if dreaming:
                dreaming.running("Generating response")

            with concurrent.futures.ThreadPoolExecutor() as executor:
                result = await loop.run_in_executor(
                    executor,
                    self._call_ollama,
                    messages,
                )
                # Mark as complete
                if dreaming:
                    dreaming.complete("Chat response generated")
                return result

        except Exception as e:
            logger.error(f"Ollama error: {e}")
            if dreaming:
                dreaming.complete("Chat request failed")
            model = get_model_manager().get_current_model()
            return (
                f"I'm sorry, I encountered an error connecting to Ollama: {str(e)}\n\n"
                "Please make sure Ollama is running (`ollama serve`) and you have a model pulled "
                f"(`ollama pull {model}`).",
                {"model": model, "error": str(e)},
            )

    def _call_ollama(self, messages: List[Dict[str, str]]) -> Tuple[str, Dict[str, Any]]:
        """
        Call Ollama API synchronously.

        This runs in a thread pool to avoid blocking the async event loop.
        Uses /api/generate for broader compatibility with Ollama versions.
        """
        try:
            from src.agents.ollama import OllamaClient

            model = get_model_manager().get_current_model()

            client = OllamaClient(
                endpoint=OLLAMA_ENDPOINT,
                timeout=OLLAMA_TIMEOUT,
            )

            # Build prompt from conversation history
            prompt_parts = []
            for msg in messages:
                role = msg["role"]
                content = msg["content"]
                if role == "user":
                    prompt_parts.append(f"User: {content}")
                elif role == "assistant":
                    prompt_parts.append(f"Assistant: {content}")

            prompt = "\n\n".join(prompt_parts)
            prompt += "\n\nAssistant:"

            # Call Ollama generate API (more compatible than chat)
            response = client.generate(
                model=model,
                prompt=prompt,
                system=SYSTEM_PROMPT,
                options={
                    "temperature": 0.7,
                    "top_p": 0.9,
                },
            )

            client.close()

            metadata = {
                "model": model,
                "tokens_generated": response.tokens_generated,
                "tokens_per_second": round(response.tokens_per_second, 1),
                "total_duration_ms": response.total_duration_ms,
            }

            return response.content.strip(), metadata

        except ImportError as e:
            logger.error(f"Failed to import Ollama client: {e}")
            return (
                "Ollama client not available. Please check the installation.",
                {"error": str(e)},
            )
        except Exception as e:
            logger.error(f"Ollama API error: {e}")
            raise


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
    search: Optional[str] = None,
    manager: ConnectionManager = Depends(get_manager),
) -> List[ConversationSummary]:
    """List recent conversations from persistent storage."""
    try:
        # Get from persistent store
        conversations = manager.store.list_conversations(limit=limit, search=search)
        return [
            ConversationSummary(
                id=conv.id,
                title=conv.title,
                message_count=conv.message_count,
                created_at=conv.created_at,
                updated_at=conv.updated_at,
            )
            for conv in conversations
        ]
    except Exception as e:
        logger.error(f"Failed to list conversations: {e}")
        # Fallback to cache
        summaries = []
        for conv_id, messages in manager.conversations.items():
            if not messages:
                continue
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
    """Delete a conversation from both cache and persistent storage."""
    # Remove from cache
    if conversation_id in manager._conversations_cache:
        del manager._conversations_cache[conversation_id]

    # Delete from persistent store
    try:
        deleted = manager.store.delete_conversation(conversation_id)
        if deleted:
            return {"status": "deleted", "conversation_id": conversation_id}
    except Exception as e:
        logger.error(f"Failed to delete conversation: {e}")

    raise HTTPException(status_code=404, detail="Conversation not found")


@router.get("/status")
async def get_chat_status(
    manager: ConnectionManager = Depends(get_manager),
) -> Dict[str, Any]:
    """Get chat system status including Ollama connectivity."""
    # Check Ollama status
    ollama_status = "unknown"
    ollama_models = []

    try:
        from src.agents.ollama import OllamaClient

        client = OllamaClient(endpoint=OLLAMA_ENDPOINT, timeout=5)
        if client.is_healthy():
            ollama_status = "connected"
            models = client.list_models()
            ollama_models = [m.name for m in models]
        else:
            ollama_status = "disconnected"
        client.close()
    except Exception as e:
        ollama_status = f"error: {str(e)}"

    model_manager = get_model_manager()
    return {
        "active_connections": len(manager.connections),
        "total_conversations": len(manager.conversations),
        "status": "operational",
        "ollama": {
            "endpoint": OLLAMA_ENDPOINT,
            "model": model_manager.get_current_model(),
            "status": ollama_status,
            "available_models": ollama_models,
        },
    }


# =============================================================================
# Model Management Endpoints
# =============================================================================


class ModelSwitchRequest(BaseModel):
    """Request to switch models."""

    model: str


@router.get("/models")
async def list_models() -> Dict[str, Any]:
    """List available Ollama models."""
    model_manager = get_model_manager()
    models = model_manager.get_available_models()
    current = model_manager.get_current_model()

    return {
        "current_model": current,
        "available_models": models,
    }


@router.get("/models/current")
async def get_current_model() -> Dict[str, str]:
    """Get the currently active model."""
    model_manager = get_model_manager()
    return {"model": model_manager.get_current_model()}


@router.post("/models/switch")
async def switch_model(request: ModelSwitchRequest) -> Dict[str, Any]:
    """Switch to a different model."""
    model_manager = get_model_manager()
    found_model = model_manager.find_model(request.model)

    if found_model:
        model_manager.set_model(found_model)
        return {
            "status": "success",
            "model": found_model,
            "message": f"Switched to {found_model}",
        }
    else:
        available = model_manager.get_available_models()
        raise HTTPException(
            status_code=404,
            detail={
                "error": f"Model '{request.model}' not found",
                "available_models": available,
            },
        )


# =============================================================================
# Conversation Export/Import Endpoints
# =============================================================================


class ImportRequest(BaseModel):
    """Request to import conversations."""

    data: Dict[str, Any]
    overwrite: bool = False


@router.get("/conversations/{conversation_id}/export")
async def export_conversation(
    conversation_id: str,
    manager: ConnectionManager = Depends(get_manager),
) -> Dict[str, Any]:
    """Export a single conversation with all messages."""
    try:
        export_data = manager.store.export_conversation(conversation_id)
        if export_data:
            return export_data
    except Exception as e:
        logger.error(f"Failed to export conversation: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    raise HTTPException(status_code=404, detail="Conversation not found")


@router.get("/export")
async def export_all_conversations(
    manager: ConnectionManager = Depends(get_manager),
) -> Dict[str, Any]:
    """Export all conversations for backup."""
    try:
        return manager.store.export_all()
    except Exception as e:
        logger.error(f"Failed to export conversations: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/import")
async def import_conversations(
    request: ImportRequest,
    manager: ConnectionManager = Depends(get_manager),
) -> Dict[str, Any]:
    """Import conversations from backup."""
    try:
        data = request.data
        imported = []

        # Handle single conversation or multiple
        if "conversations" in data:
            # Multiple conversations
            for conv_data in data["conversations"]:
                conv_id = manager.store.import_conversation(conv_data, overwrite=request.overwrite)
                if conv_id:
                    imported.append(conv_id)
        elif "conversation" in data:
            # Single conversation
            conv_id = manager.store.import_conversation(data, overwrite=request.overwrite)
            if conv_id:
                imported.append(conv_id)

        # Clear cache to force reload from store
        for conv_id in imported:
            if conv_id in manager._conversations_cache:
                del manager._conversations_cache[conv_id]

        return {
            "status": "success",
            "imported_count": len(imported),
            "conversation_ids": imported,
        }
    except Exception as e:
        logger.error(f"Failed to import conversations: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/storage/stats")
async def get_storage_stats(
    manager: ConnectionManager = Depends(get_manager),
) -> Dict[str, Any]:
    """Get conversation storage statistics."""
    try:
        return manager.store.get_statistics()
    except Exception as e:
        logger.error(f"Failed to get storage stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/conversations/{conversation_id}/archive")
async def archive_conversation(
    conversation_id: str,
    manager: ConnectionManager = Depends(get_manager),
) -> Dict[str, str]:
    """Archive a conversation."""
    try:
        success = manager.store.update_conversation(conversation_id, archived=True)
        if success:
            # Remove from cache
            if conversation_id in manager._conversations_cache:
                del manager._conversations_cache[conversation_id]
            return {"status": "archived", "conversation_id": conversation_id}
    except Exception as e:
        logger.error(f"Failed to archive conversation: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    raise HTTPException(status_code=404, detail="Conversation not found")


@router.post("/conversations/{conversation_id}/unarchive")
async def unarchive_conversation(
    conversation_id: str,
    manager: ConnectionManager = Depends(get_manager),
) -> Dict[str, str]:
    """Unarchive a conversation."""
    try:
        success = manager.store.update_conversation(conversation_id, archived=False)
        if success:
            return {"status": "unarchived", "conversation_id": conversation_id}
    except Exception as e:
        logger.error(f"Failed to unarchive conversation: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    raise HTTPException(status_code=404, detail="Conversation not found")


@router.get("/search")
async def search_messages(
    q: str,
    conversation_id: Optional[str] = None,
    limit: int = 20,
    manager: ConnectionManager = Depends(get_manager),
) -> List[Dict[str, Any]]:
    """Search messages across conversations."""
    try:
        messages = manager.store.search_messages(
            query=q,
            conversation_id=conversation_id,
            limit=limit,
        )
        return [m.to_dict() for m in messages]
    except Exception as e:
        logger.error(f"Failed to search messages: {e}")
        raise HTTPException(status_code=500, detail=str(e))
