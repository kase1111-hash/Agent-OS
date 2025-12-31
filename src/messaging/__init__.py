"""
Agent OS Messaging - Message Protocol & Bus

This module provides inter-agent communication infrastructure:
- FlowRequest/FlowResponse message schemas
- Message bus implementations (in-memory, Redis)
- Channel management and routing
- Audit logging and dead letter queue
"""

from .bus import (
    ChannelRouter,
    ChannelStats,
    InMemoryMessageBus,
    MessageBus,
    MessageHandler,
    Subscription,
)
from .models import (
    AuditLogEntry,
    CheckStatus,
    ConstitutionalCheck,
    DeadLetterMessage,
    FlowRequest,
    FlowResponse,
    MemoryRequest,
    MessagePriority,
    MessageStatus,
    RequestMetadata,
    ResponseMetadata,
    create_request,
    create_response,
)

# Optional Redis support
try:
    from .redis_bus import RedisMessageBus, create_redis_bus

    REDIS_AVAILABLE = True
except ImportError:
    RedisMessageBus = None
    create_redis_bus = None
    REDIS_AVAILABLE = False

__all__ = [
    # Models
    "FlowRequest",
    "FlowResponse",
    "RequestMetadata",
    "ResponseMetadata",
    "ConstitutionalCheck",
    "MemoryRequest",
    "MessageStatus",
    "CheckStatus",
    "MessagePriority",
    "DeadLetterMessage",
    "AuditLogEntry",
    "create_request",
    "create_response",
    # Bus
    "MessageBus",
    "InMemoryMessageBus",
    "RedisMessageBus",
    "create_redis_bus",
    "MessageHandler",
    "Subscription",
    "ChannelRouter",
    "ChannelStats",
    "REDIS_AVAILABLE",
]
