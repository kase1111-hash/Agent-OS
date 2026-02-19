"""
Agent OS Message Bus

Provides pub/sub messaging infrastructure for inter-agent communication.
Supports multiple backends: in-memory (development) and Redis (production).
"""

import asyncio
import json
import logging
import threading
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from queue import Empty, Queue
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    List,
    Optional,
    Set,
    Union,
)
from uuid import UUID, uuid4

from .exceptions import (
    BusShutdownError,
    HandlerExecutionError,
    MessageDeliveryError,
)
from .models import (
    AuditLogEntry,
    DeadLetterMessage,
    FlowRequest,
    FlowResponse,
    MessageStatus,
)

logger = logging.getLogger(__name__)

# Type for message handlers
MessageHandler = Callable[
    [Union[FlowRequest, FlowResponse]], Optional[Union[FlowResponse, Awaitable[FlowResponse]]]
]


@dataclass
class Subscription:
    """Represents a subscription to a channel."""

    subscription_id: str
    channel: str
    handler: MessageHandler
    subscriber_name: str
    created_at: datetime = field(default_factory=datetime.utcnow)
    message_count: int = 0
    is_active: bool = True

    def increment_count(self) -> None:
        """Increment message count."""
        self.message_count += 1


@dataclass
class ChannelStats:
    """Statistics for a message channel."""

    channel: str
    messages_published: int = 0
    messages_delivered: int = 0
    messages_failed: int = 0
    subscriber_count: int = 0
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_message_at: Optional[datetime] = None


class MessageBus(ABC):
    """
    Abstract base class for message bus implementations.

    Provides pub/sub messaging with:
    - Channel-based routing
    - Message persistence (optional)
    - Audit logging
    - Dead letter queue
    """

    @abstractmethod
    def publish(
        self,
        channel: str,
        message: Union[FlowRequest, FlowResponse],
    ) -> bool:
        """
        Publish a message to a channel.

        Args:
            channel: Target channel name
            message: Message to publish

        Returns:
            True if published successfully
        """
        pass

    @abstractmethod
    def subscribe(
        self,
        channel: str,
        handler: MessageHandler,
        subscriber_name: str,
    ) -> Subscription:
        """
        Subscribe to a channel.

        Args:
            channel: Channel to subscribe to
            handler: Function to call when message received
            subscriber_name: Name of the subscriber

        Returns:
            Subscription object
        """
        pass

    @abstractmethod
    def unsubscribe(self, subscription: Subscription) -> bool:
        """
        Unsubscribe from a channel.

        Args:
            subscription: Subscription to cancel

        Returns:
            True if unsubscribed successfully
        """
        pass

    @abstractmethod
    def get_channel_stats(self, channel: str) -> Optional[ChannelStats]:
        """Get statistics for a channel."""
        pass

    @abstractmethod
    def get_dead_letters(self, limit: int = 100) -> List[DeadLetterMessage]:
        """Get messages from the dead letter queue."""
        pass

    @abstractmethod
    def get_audit_log(
        self,
        limit: int = 100,
        channel: Optional[str] = None,
    ) -> List[AuditLogEntry]:
        """Get audit log entries."""
        pass

    @abstractmethod
    def shutdown(self) -> None:
        """Shutdown the message bus."""
        pass


class InMemoryMessageBus(MessageBus):
    """
    In-memory message bus implementation for development and testing.

    Features:
    - Thread-safe message delivery
    - FIFO ordering per channel
    - Dead letter queue
    - Audit logging
    """

    def __init__(
        self,
        max_queue_size: int = 10000,
        max_audit_entries: int = 10000,
        max_dead_letters: int = 1000,
        delivery_timeout_seconds: float = 30.0,
        max_messages_per_minute: int = 1000,
        identity_registry: object = None,
        channel_acls: Optional[Dict[str, set]] = None,
    ):
        """
        Initialize the in-memory message bus.

        Args:
            max_queue_size: Maximum messages per channel queue
            max_audit_entries: Maximum audit log entries to keep
            max_dead_letters: Maximum dead letter messages to keep
            delivery_timeout_seconds: Timeout for message delivery
            max_messages_per_minute: Rate limit per agent (V10-1)
            identity_registry: AgentIdentityRegistry for sender verification (V4-2)
            channel_acls: Channel-level access control lists (V4-4)
        """
        self._subscriptions: Dict[str, List[Subscription]] = defaultdict(list)
        self._channel_stats: Dict[str, ChannelStats] = {}
        self._dead_letters: List[DeadLetterMessage] = []
        self._audit_log: List[AuditLogEntry] = []

        self._max_queue_size = max_queue_size
        self._max_audit_entries = max_audit_entries
        self._max_dead_letters = max_dead_letters
        self._delivery_timeout = delivery_timeout_seconds

        self._lock = threading.RLock()
        self._running = True

        # Message queues for async processing
        self._queues: Dict[str, Queue] = defaultdict(Queue)

        # Worker threads for processing messages
        self._workers: Dict[str, threading.Thread] = {}

        # V4-2: Agent identity verification
        self._identity_registry = identity_registry

        # V4-4: Channel-level access control lists
        # Maps channel name -> set of agent names allowed to publish
        self._channel_acls: Dict[str, set] = channel_acls or {}

        # V10-1: Per-agent rate limiting
        self._max_messages_per_minute = max_messages_per_minute
        self._agent_message_times: Dict[str, list] = defaultdict(list)

        logger.info("InMemoryMessageBus initialized")

    def _check_agent_rate_limit(self, source: str) -> bool:
        """Check per-agent rate limit (V10-1). Returns True if within limit."""
        import time

        now = time.time()
        times = self._agent_message_times[source]
        # Remove entries older than 60s
        while times and times[0] < now - 60:
            times.pop(0)
        if len(times) >= self._max_messages_per_minute:
            return False
        times.append(now)
        return True

    def _check_channel_acl(self, channel: str, source: str) -> bool:
        """Check channel-level ACL (V4-4). Returns True if allowed."""
        if not self._channel_acls:
            return True
        allowed = self._channel_acls.get(channel)
        if allowed is None:
            return True  # No ACL for this channel = open
        return source in allowed

    def _scan_message_for_secrets(
        self,
        message: Union[FlowRequest, FlowResponse],
        channel: str,
        source: str,
    ) -> None:
        """
        V2-3: Scan message content for secrets before delivery.

        Extracts text from the message payload and runs it through the
        SecretScanner. Raises SecretLeakageError if secrets are found,
        which prevents the message from being delivered.
        """
        from src.security.secret_scanner import SecretLeakageError, get_secret_scanner

        scanner = get_secret_scanner()

        # Collect text fields from the message
        text_parts: list[str] = []

        if isinstance(message, FlowRequest):
            text_parts.append(message.content.prompt)
            for ctx in message.content.context:
                for v in ctx.values():
                    if isinstance(v, str):
                        text_parts.append(v)
        elif isinstance(message, FlowResponse):
            output = message.content.output
            if isinstance(output, str):
                text_parts.append(output)
            elif isinstance(output, dict):
                for v in output.values():
                    if isinstance(v, str):
                        text_parts.append(v)
            if message.content.reasoning:
                text_parts.append(message.content.reasoning)

        combined = "\n".join(text_parts)

        # scan_and_block raises SecretLeakageError if secrets found
        try:
            scanner.scan_and_block(combined, source_agent=source)
        except SecretLeakageError:
            self._log_audit(message, channel, error="BLOCKED: secret detected in message payload")
            raise

    def _sign_message(
        self,
        message: Union[FlowRequest, FlowResponse],
        source: str,
    ) -> None:
        """
        V4-2: Sign a message with the sender's Ed25519 identity.

        Computes a SHA-256 hash of the message content and signs it
        using the agent's registered keypair. The signature and hash
        are stored on the message for verification at delivery time.
        """
        if not self._identity_registry:
            return

        registry = self._identity_registry
        if not hasattr(registry, "is_registered") or not registry.is_registered(source):
            return

        import hashlib

        # Build content hash from message payload
        if isinstance(message, FlowRequest):
            content_str = message.content.prompt
            msg_id = str(message.request_id)
        else:
            output = message.content.output
            content_str = output if isinstance(output, str) else json.dumps(output)
            msg_id = str(message.response_id)

        content_hash = hashlib.sha256(content_str.encode()).hexdigest()

        # Sign using the identity registry
        identity = registry.get_identity(source)
        if identity:
            signature = identity.sign_message_payload(msg_id, source, content_hash)
            message.signature_hex = signature.hex()
            message.content_hash = content_hash

    def _verify_message_signature(
        self,
        message: Union[FlowRequest, FlowResponse],
    ) -> bool:
        """
        V4-2: Verify a message's cryptographic signature.

        Returns True if:
        - No identity registry is configured (signing disabled)
        - The source agent is not registered (unsigned messages allowed from unregistered agents)
        - The signature is valid

        Returns False only if the message has a signature that fails verification.
        """
        if not self._identity_registry:
            return True

        sig_hex = getattr(message, "signature_hex", None)
        content_hash = getattr(message, "content_hash", None)
        source = getattr(message, "source", "unknown")

        # No signature present — allow if agent is not registered
        if not sig_hex or not content_hash:
            registry = self._identity_registry
            if hasattr(registry, "is_registered") and registry.is_registered(source):
                logger.warning(
                    f"REJECTED: Registered agent '{source}' sent unsigned message"
                )
                return False
            return True

        # Verify the signature
        if isinstance(message, FlowRequest):
            msg_id = str(message.request_id)
        else:
            msg_id = str(message.response_id)

        registry = self._identity_registry
        if hasattr(registry, "verify_message_payload"):
            valid = registry.verify_message_payload(
                source, msg_id, source, content_hash, bytes.fromhex(sig_hex)
            )
            if not valid:
                logger.warning(
                    f"REJECTED: Message from '{source}' failed signature verification"
                )
            return valid

        return True

    def publish(
        self,
        channel: str,
        message: Union[FlowRequest, FlowResponse],
    ) -> bool:
        """Publish a message to a channel with security checks."""
        if not self._running:
            logger.warning("Cannot publish: bus is shutting down")
            return False

        # Extract source from message
        source = getattr(message, "source", "unknown")

        # V10-1: Check per-agent rate limit
        if not self._check_agent_rate_limit(source):
            logger.warning(
                f"Rate limit exceeded for agent '{source}' on channel '{channel}' "
                f"({self._max_messages_per_minute}/min)"
            )
            return False

        # V4-4: Check channel ACL
        if not self._check_channel_acl(channel, source):
            logger.warning(
                f"Agent '{source}' not authorized to publish to channel '{channel}'"
            )
            return False

        # V2-3: Scan message content for secrets before delivery
        try:
            self._scan_message_for_secrets(message, channel, source)
        except Exception:
            # Message blocked — already logged by _scan_message_for_secrets
            return False

        # V4-2: Sign message with sender's Ed25519 identity
        self._sign_message(message, source)

        # V4-2: Verify signature before delivery
        if not self._verify_message_signature(message):
            self._log_audit(
                message, channel,
                error=f"REJECTED: signature verification failed for agent '{source}'",
            )
            return False

        with self._lock:
            # Ensure channel stats exist
            if channel not in self._channel_stats:
                self._channel_stats[channel] = ChannelStats(channel=channel)

            stats = self._channel_stats[channel]
            stats.messages_published += 1
            stats.last_message_at = datetime.utcnow()

            # Get subscribers for this channel
            subscribers = self._subscriptions.get(channel, [])
            active_subscribers = [s for s in subscribers if s.is_active]

            if not active_subscribers:
                logger.debug(f"No subscribers for channel: {channel}")
                self._log_audit(message, channel, error="No subscribers")
                return True  # Not an error, just no one listening

            # Deliver to all subscribers
            delivered = 0
            delivery_errors: List[str] = []
            for subscription in active_subscribers:
                try:
                    self._deliver_message(subscription, message)
                    subscription.increment_count()
                    stats.messages_delivered += 1
                    delivered += 1
                except HandlerExecutionError as e:
                    error_msg = f"{subscription.subscriber_name}: {e}"
                    logger.error(f"Handler error during delivery: {error_msg}")
                    stats.messages_failed += 1
                    delivery_errors.append(error_msg)
                    self._add_dead_letter(message, channel, str(e))
                except Exception as e:
                    error_msg = f"{subscription.subscriber_name}: {type(e).__name__}: {e}"
                    logger.error(f"Unexpected delivery error: {error_msg}")
                    stats.messages_failed += 1
                    delivery_errors.append(error_msg)
                    self._add_dead_letter(message, channel, str(e))

            self._log_audit(message, channel)

            # If all deliveries failed, log and return False
            # (failed messages are already in the dead letter queue)
            if delivered == 0 and delivery_errors:
                logger.error(
                    f"All deliveries failed for channel {channel}: "
                    f"{'; '.join(delivery_errors)}"
                )

            return delivered > 0

    def _deliver_message(
        self,
        subscription: Subscription,
        message: Union[FlowRequest, FlowResponse],
    ) -> None:
        """Deliver a message to a subscriber."""
        try:
            result = subscription.handler(message)

            # Handle async handlers — always block until the coroutine completes
            if asyncio.iscoroutine(result):
                try:
                    # Use asyncio.run() for clean event loop lifecycle.
                    # This works whether or not a loop exists in the current thread,
                    # because asyncio.run() creates its own loop.
                    asyncio.run(result)
                except RuntimeError:
                    # Already inside a running event loop (e.g., during async tests)
                    # Fall back to creating a new loop in a thread-safe way
                    import concurrent.futures
                    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                        future = pool.submit(asyncio.run, result)
                        future.result()  # Block until complete, propagates exceptions
                except Exception as e:
                    logger.error(
                        f"Async handler error for {subscription.subscriber_name}: {e}"
                    )
                    raise HandlerExecutionError(
                        message=f"Async handler execution failed: {e}",
                        channel=subscription.channel,
                        handler_name=subscription.subscriber_name,
                        original_error=e,
                    )

        except HandlerExecutionError:
            # Already wrapped, re-raise
            raise
        except Exception as e:
            logger.error(f"Handler error for {subscription.subscriber_name}: {e}")
            raise HandlerExecutionError(
                message=f"Handler execution failed: {e}",
                channel=subscription.channel,
                handler_name=subscription.subscriber_name,
                original_error=e,
            )

    def subscribe(
        self,
        channel: str,
        handler: MessageHandler,
        subscriber_name: str,
    ) -> Subscription:
        """Subscribe to a channel."""
        with self._lock:
            subscription = Subscription(
                subscription_id=str(uuid4()),
                channel=channel,
                handler=handler,
                subscriber_name=subscriber_name,
            )

            self._subscriptions[channel].append(subscription)

            # Update channel stats
            if channel not in self._channel_stats:
                self._channel_stats[channel] = ChannelStats(channel=channel)
            self._channel_stats[channel].subscriber_count += 1

            logger.info(f"Subscribed {subscriber_name} to channel: {channel}")

            return subscription

    def unsubscribe(self, subscription: Subscription) -> bool:
        """Unsubscribe from a channel."""
        with self._lock:
            channel = subscription.channel
            if channel in self._subscriptions:
                subs = self._subscriptions[channel]
                for i, s in enumerate(subs):
                    if s.subscription_id == subscription.subscription_id:
                        s.is_active = False
                        subs.pop(i)
                        if channel in self._channel_stats:
                            self._channel_stats[channel].subscriber_count -= 1
                        logger.info(
                            f"Unsubscribed {subscription.subscriber_name} "
                            f"from channel: {channel}"
                        )
                        return True

            return False

    def get_channel_stats(self, channel: str) -> Optional[ChannelStats]:
        """Get statistics for a channel."""
        with self._lock:
            return self._channel_stats.get(channel)

    def get_all_channel_stats(self) -> Dict[str, ChannelStats]:
        """Get statistics for all channels."""
        with self._lock:
            return dict(self._channel_stats)

    def get_dead_letters(self, limit: int = 100) -> List[DeadLetterMessage]:
        """Get messages from the dead letter queue."""
        with self._lock:
            return list(self._dead_letters[-limit:])

    def retry_dead_letter(self, index: int) -> bool:
        """
        Retry a message from the dead letter queue.

        Args:
            index: Index of message in dead letter queue

        Returns:
            True if retry was successful
        """
        with self._lock:
            if index < 0 or index >= len(self._dead_letters):
                return False

            dead_letter = self._dead_letters[index]
            dead_letter.retry_count += 1

            # Try to publish again
            success = self.publish(
                dead_letter.destination_channel,
                dead_letter.original_message,
            )

            if success:
                self._dead_letters.pop(index)

            return success

    def get_audit_log(
        self,
        limit: int = 100,
        channel: Optional[str] = None,
    ) -> List[AuditLogEntry]:
        """Get audit log entries."""
        with self._lock:
            if channel:
                filtered = [
                    e for e in self._audit_log if e.destination == channel or e.source == channel
                ]
                return list(filtered[-limit:])
            return list(self._audit_log[-limit:])

    def _add_dead_letter(
        self,
        message: Union[FlowRequest, FlowResponse],
        channel: str,
        error: str,
    ) -> None:
        """Add a message to the dead letter queue (thread-safe)."""
        dead_letter = DeadLetterMessage(
            original_message=message,
            failure_reason=error,
            destination_channel=channel,
            last_error=error,
        )

        # SECURITY FIX: Use lock for thread-safe dead letter queue operations
        with self._lock:
            self._dead_letters.append(dead_letter)

            # Trim if over limit - done atomically within lock
            if len(self._dead_letters) > self._max_dead_letters:
                # Remove oldest entries to stay within limit
                excess = len(self._dead_letters) - self._max_dead_letters
                self._dead_letters = self._dead_letters[excess:]

    def _log_audit(
        self,
        message: Union[FlowRequest, FlowResponse],
        channel: str,
        error: Optional[str] = None,
    ) -> None:
        """Log a message to the audit trail."""
        if isinstance(message, FlowRequest):
            entry = AuditLogEntry(
                message_type="request",
                message_id=message.request_id,
                source=message.source,
                destination=channel,
                intent=message.intent,
                constitutional_status=message.constitutional_check.status.value,
                error=error,
            )
        else:
            entry = AuditLogEntry(
                message_type="response",
                message_id=message.response_id,
                source=message.source,
                destination=channel,
                status=message.status.value,
                error=error,
            )

        self._audit_log.append(entry)

        # Trim if over limit
        if len(self._audit_log) > self._max_audit_entries:
            self._audit_log = self._audit_log[-self._max_audit_entries :]

    def get_subscriber_count(self, channel: str) -> int:
        """Get number of active subscribers for a channel."""
        with self._lock:
            subs = self._subscriptions.get(channel, [])
            return len([s for s in subs if s.is_active])

    def get_all_channels(self) -> List[str]:
        """Get list of all channels with subscriptions."""
        with self._lock:
            return list(self._subscriptions.keys())

    def get_message_log(
        self,
        channel: Optional[str] = None,
        agent: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        Return recent inter-agent message log for human inspection.

        V2-3: All inter-agent communications are logged for human principal
        visibility. Content is redacted to prevent secret exposure in logs.

        Args:
            channel: Filter by channel name (optional)
            agent: Filter by source agent name (optional)
            limit: Maximum entries to return

        Returns:
            List of audit log entry dicts with redacted content summaries
        """
        with self._lock:
            entries = list(self._audit_log)

        if channel:
            entries = [e for e in entries if e.destination == channel]
        if agent:
            entries = [e for e in entries if e.source == agent]

        # Return most recent entries first
        entries = entries[-limit:]
        entries.reverse()

        return [
            {
                "entry_id": str(e.entry_id),
                "timestamp": e.timestamp.isoformat(),
                "message_type": e.message_type,
                "message_id": str(e.message_id),
                "source": e.source,
                "destination": e.destination,
                "intent": e.intent,
                "status": e.status,
                "constitutional_status": e.constitutional_status,
                "error": e.error,
            }
            for e in entries
        ]

    def clear_audit_log(self) -> int:
        """Clear the audit log. Returns number of entries cleared."""
        with self._lock:
            count = len(self._audit_log)
            self._audit_log.clear()
            return count

    def clear_dead_letters(self) -> int:
        """Clear the dead letter queue. Returns number of messages cleared."""
        with self._lock:
            count = len(self._dead_letters)
            self._dead_letters.clear()
            return count

    def shutdown(self) -> None:
        """Shutdown the message bus."""
        logger.info("Shutting down InMemoryMessageBus")
        self._running = False

        with self._lock:
            # Mark all subscriptions as inactive
            for channel, subs in self._subscriptions.items():
                for sub in subs:
                    sub.is_active = False

        logger.info("InMemoryMessageBus shutdown complete")


class ChannelRouter:
    """
    Routes messages to appropriate channels based on agent names and intent.

    Provides:
    - Agent-specific channels
    - Intent-based routing
    - Broadcast channels
    """

    # Standard channel patterns
    AGENT_CHANNEL_PREFIX = "agent."
    INTENT_CHANNEL_PREFIX = "intent."
    BROADCAST_CHANNEL = "broadcast"
    SYSTEM_CHANNEL = "system"

    def __init__(self, bus: MessageBus):
        """
        Initialize the router.

        Args:
            bus: Message bus to use for routing
        """
        self._bus = bus

    @classmethod
    def agent_channel(cls, agent_name: str) -> str:
        """Get the channel name for an agent."""
        return f"{cls.AGENT_CHANNEL_PREFIX}{agent_name.lower()}"

    @classmethod
    def intent_channel(cls, intent: str) -> str:
        """Get the channel name for an intent type."""
        return f"{cls.INTENT_CHANNEL_PREFIX}{intent.lower()}"

    def route_request(self, request: FlowRequest) -> bool:
        """
        Route a request to the appropriate channel(s).

        Args:
            request: Request to route

        Returns:
            True if routed successfully
        """
        # Primary route: destination agent channel
        agent_channel = self.agent_channel(request.destination)
        success = self._bus.publish(agent_channel, request)

        # Also publish to intent channel for monitoring
        intent_channel = self.intent_channel(request.intent)
        self._bus.publish(intent_channel, request)

        return success

    def route_response(self, response: FlowResponse) -> bool:
        """
        Route a response to the appropriate channel.

        Args:
            response: Response to route

        Returns:
            True if routed successfully
        """
        # Route to destination agent channel
        agent_channel = self.agent_channel(response.destination)
        return self._bus.publish(agent_channel, response)

    def broadcast(self, message: Union[FlowRequest, FlowResponse]) -> bool:
        """
        Broadcast a message to all subscribers.

        Args:
            message: Message to broadcast

        Returns:
            True if broadcasted successfully
        """
        return self._bus.publish(self.BROADCAST_CHANNEL, message)

    def subscribe_agent(
        self,
        agent_name: str,
        handler: MessageHandler,
    ) -> Subscription:
        """
        Subscribe an agent to its dedicated channel.

        Args:
            agent_name: Name of the agent
            handler: Handler for incoming messages

        Returns:
            Subscription object
        """
        channel = self.agent_channel(agent_name)
        return self._bus.subscribe(channel, handler, agent_name)

    def subscribe_intent(
        self,
        intent: str,
        handler: MessageHandler,
        subscriber_name: str,
    ) -> Subscription:
        """
        Subscribe to messages with a specific intent.

        Args:
            intent: Intent type to subscribe to
            handler: Handler for matching messages
            subscriber_name: Name of subscriber

        Returns:
            Subscription object
        """
        channel = self.intent_channel(intent)
        return self._bus.subscribe(channel, handler, subscriber_name)

    def subscribe_broadcast(
        self,
        handler: MessageHandler,
        subscriber_name: str,
    ) -> Subscription:
        """
        Subscribe to broadcast messages.

        Args:
            handler: Handler for broadcast messages
            subscriber_name: Name of subscriber

        Returns:
            Subscription object
        """
        return self._bus.subscribe(self.BROADCAST_CHANNEL, handler, subscriber_name)
