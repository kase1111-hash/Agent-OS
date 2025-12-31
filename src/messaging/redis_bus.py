"""
Agent OS Redis Message Bus

Production-ready message bus implementation using Redis pub/sub.
Provides persistent messaging with high throughput.
"""

import asyncio
import json
import logging
import threading
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Union
from uuid import UUID, uuid4

from .bus import (
    ChannelStats,
    MessageBus,
    MessageHandler,
    Subscription,
)
from .models import (
    AuditLogEntry,
    DeadLetterMessage,
    FlowRequest,
    FlowResponse,
)

logger = logging.getLogger(__name__)

# Optional Redis import - gracefully handle if not installed
try:
    import redis
    from redis import Redis
    from redis.client import PubSub

    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    Redis = None
    PubSub = None


class RedisMessageBus(MessageBus):
    """
    Redis-based message bus implementation for production use.

    Features:
    - High-throughput pub/sub messaging
    - Persistent message storage (optional)
    - Distributed dead letter queue
    - Centralized audit logging
    - Cluster support
    """

    # Redis key prefixes
    AUDIT_KEY = "agent_os:audit"
    DEAD_LETTER_KEY = "agent_os:dead_letters"
    STATS_KEY = "agent_os:stats"
    CHANNEL_PREFIX = "agent_os:channel:"

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        password: Optional[str] = None,
        max_audit_entries: int = 10000,
        max_dead_letters: int = 1000,
        message_ttl_seconds: int = 86400,  # 24 hours
        connection_pool_size: int = 10,
    ):
        """
        Initialize Redis message bus.

        Args:
            host: Redis host
            port: Redis port
            db: Redis database number
            password: Redis password (optional)
            max_audit_entries: Maximum audit log entries
            max_dead_letters: Maximum dead letter messages
            message_ttl_seconds: TTL for persisted messages
            connection_pool_size: Size of connection pool
        """
        if not REDIS_AVAILABLE:
            raise ImportError("Redis is not installed. Install with: pip install redis")

        self._host = host
        self._port = port
        self._db = db
        self._password = password
        self._max_audit_entries = max_audit_entries
        self._max_dead_letters = max_dead_letters
        self._message_ttl = message_ttl_seconds

        # Create connection pool
        self._pool = redis.ConnectionPool(
            host=host,
            port=port,
            db=db,
            password=password,
            max_connections=connection_pool_size,
            decode_responses=True,
        )

        self._client: Redis = redis.Redis(connection_pool=self._pool)

        # Track subscriptions locally
        self._subscriptions: Dict[str, List[Subscription]] = {}
        self._pubsub_threads: Dict[str, threading.Thread] = {}
        self._pubsubs: Dict[str, PubSub] = {}

        self._running = True
        self._lock = threading.RLock()

        # Verify connection
        try:
            self._client.ping()
            logger.info(f"RedisMessageBus connected to {host}:{port}")
        except redis.ConnectionError as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise

    def publish(
        self,
        channel: str,
        message: Union[FlowRequest, FlowResponse],
    ) -> bool:
        """Publish a message to a Redis channel."""
        if not self._running:
            logger.warning("Cannot publish: bus is shutting down")
            return False

        try:
            # Serialize message to JSON
            message_data = message.model_dump_json()

            # Publish to Redis channel
            redis_channel = f"{self.CHANNEL_PREFIX}{channel}"
            subscriber_count = self._client.publish(redis_channel, message_data)

            # Update stats
            self._update_stats(channel, published=1, delivered=subscriber_count)

            # Log to audit
            self._log_audit(message, channel)

            logger.debug(f"Published to {channel}, {subscriber_count} subscribers received")

            return True

        except redis.RedisError as e:
            logger.error(f"Redis publish error: {e}")
            self._add_dead_letter(message, channel, str(e))
            return False

    def subscribe(
        self,
        channel: str,
        handler: MessageHandler,
        subscriber_name: str,
    ) -> Subscription:
        """Subscribe to a Redis channel."""
        with self._lock:
            subscription = Subscription(
                subscription_id=str(uuid4()),
                channel=channel,
                handler=handler,
                subscriber_name=subscriber_name,
            )

            # Add to local tracking
            if channel not in self._subscriptions:
                self._subscriptions[channel] = []
                self._start_pubsub(channel)

            self._subscriptions[channel].append(subscription)

            # Update stats
            stats_key = f"{self.STATS_KEY}:{channel}"
            self._client.hincrby(stats_key, "subscriber_count", 1)

            logger.info(f"Subscribed {subscriber_name} to channel: {channel}")

            return subscription

    def _start_pubsub(self, channel: str) -> None:
        """Start a pubsub listener for a channel."""
        redis_channel = f"{self.CHANNEL_PREFIX}{channel}"

        pubsub = self._client.pubsub()
        pubsub.subscribe(redis_channel)

        def listener():
            """Listen for messages on the channel."""
            try:
                for message in pubsub.listen():
                    if not self._running:
                        break

                    if message["type"] == "message":
                        self._handle_message(channel, message["data"])

            except redis.RedisError as e:
                logger.error(f"PubSub listener error: {e}")
            finally:
                pubsub.close()

        thread = threading.Thread(
            target=listener,
            name=f"pubsub-{channel}",
            daemon=True,
        )
        thread.start()

        self._pubsubs[channel] = pubsub
        self._pubsub_threads[channel] = thread

    def _handle_message(self, channel: str, data: str) -> None:
        """Handle an incoming message from Redis."""
        try:
            # Deserialize message
            message_dict = json.loads(data)

            # Determine message type and deserialize
            if "request_id" in message_dict and "intent" in message_dict:
                message = FlowRequest.model_validate(message_dict)
            else:
                message = FlowResponse.model_validate(message_dict)

            # Deliver to all local subscribers
            subscribers = self._subscriptions.get(channel, [])
            for subscription in subscribers:
                if subscription.is_active:
                    try:
                        subscription.handler(message)
                        subscription.increment_count()
                    except Exception as e:
                        logger.error(f"Handler error for {subscription.subscriber_name}: {e}")

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse message: {e}")
        except Exception as e:
            logger.error(f"Message handling error: {e}")

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

                        # Update stats
                        stats_key = f"{self.STATS_KEY}:{channel}"
                        self._client.hincrby(stats_key, "subscriber_count", -1)

                        # Stop pubsub if no more subscribers
                        if not subs:
                            self._stop_pubsub(channel)

                        logger.info(
                            f"Unsubscribed {subscription.subscriber_name} "
                            f"from channel: {channel}"
                        )
                        return True

            return False

    def _stop_pubsub(self, channel: str) -> None:
        """Stop the pubsub listener for a channel."""
        if channel in self._pubsubs:
            self._pubsubs[channel].unsubscribe()
            del self._pubsubs[channel]

        if channel in self._pubsub_threads:
            del self._pubsub_threads[channel]

        if channel in self._subscriptions:
            del self._subscriptions[channel]

    def get_channel_stats(self, channel: str) -> Optional[ChannelStats]:
        """Get statistics for a channel from Redis."""
        try:
            stats_key = f"{self.STATS_KEY}:{channel}"
            data = self._client.hgetall(stats_key)

            if not data:
                return None

            return ChannelStats(
                channel=channel,
                messages_published=int(data.get("published", 0)),
                messages_delivered=int(data.get("delivered", 0)),
                messages_failed=int(data.get("failed", 0)),
                subscriber_count=int(data.get("subscriber_count", 0)),
            )

        except redis.RedisError as e:
            logger.error(f"Failed to get stats: {e}")
            return None

    def _update_stats(
        self,
        channel: str,
        published: int = 0,
        delivered: int = 0,
        failed: int = 0,
    ) -> None:
        """Update channel statistics in Redis."""
        try:
            stats_key = f"{self.STATS_KEY}:{channel}"
            pipe = self._client.pipeline()

            if published:
                pipe.hincrby(stats_key, "published", published)
            if delivered:
                pipe.hincrby(stats_key, "delivered", delivered)
            if failed:
                pipe.hincrby(stats_key, "failed", failed)

            pipe.hset(stats_key, "last_message_at", datetime.utcnow().isoformat())
            pipe.execute()

        except redis.RedisError as e:
            logger.error(f"Failed to update stats: {e}")

    def get_dead_letters(self, limit: int = 100) -> List[DeadLetterMessage]:
        """Get messages from the Redis dead letter queue."""
        try:
            # Get last N entries from the list
            data = self._client.lrange(self.DEAD_LETTER_KEY, -limit, -1)

            dead_letters = []
            for item in data:
                try:
                    dl_dict = json.loads(item)
                    dead_letters.append(DeadLetterMessage.model_validate(dl_dict))
                except Exception as e:
                    logger.error(f"Failed to parse dead letter: {e}")

            return dead_letters

        except redis.RedisError as e:
            logger.error(f"Failed to get dead letters: {e}")
            return []

    def _add_dead_letter(
        self,
        message: Union[FlowRequest, FlowResponse],
        channel: str,
        error: str,
    ) -> None:
        """Add a message to the Redis dead letter queue."""
        try:
            dead_letter = DeadLetterMessage(
                original_message=message,
                failure_reason=error,
                destination_channel=channel,
                last_error=error,
            )

            self._client.rpush(
                self.DEAD_LETTER_KEY,
                dead_letter.model_dump_json(),
            )

            # Trim to max size
            self._client.ltrim(
                self.DEAD_LETTER_KEY,
                -self._max_dead_letters,
                -1,
            )

        except redis.RedisError as e:
            logger.error(f"Failed to add dead letter: {e}")

    def get_audit_log(
        self,
        limit: int = 100,
        channel: Optional[str] = None,
    ) -> List[AuditLogEntry]:
        """Get audit log entries from Redis."""
        try:
            # Get last N entries
            data = self._client.lrange(self.AUDIT_KEY, -limit, -1)

            entries = []
            for item in data:
                try:
                    entry_dict = json.loads(item)
                    entry = AuditLogEntry.model_validate(entry_dict)

                    # Filter by channel if specified
                    if channel and entry.destination != channel:
                        continue

                    entries.append(entry)
                except Exception as e:
                    logger.error(f"Failed to parse audit entry: {e}")

            return entries

        except redis.RedisError as e:
            logger.error(f"Failed to get audit log: {e}")
            return []

    def _log_audit(
        self,
        message: Union[FlowRequest, FlowResponse],
        channel: str,
        error: Optional[str] = None,
    ) -> None:
        """Log a message to the Redis audit trail."""
        try:
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

            self._client.rpush(self.AUDIT_KEY, entry.model_dump_json())

            # Trim to max size
            self._client.ltrim(self.AUDIT_KEY, -self._max_audit_entries, -1)

            # Set TTL on audit key
            self._client.expire(self.AUDIT_KEY, self._message_ttl)

        except redis.RedisError as e:
            logger.error(f"Failed to log audit: {e}")

    def shutdown(self) -> None:
        """Shutdown the Redis message bus."""
        logger.info("Shutting down RedisMessageBus")
        self._running = False

        with self._lock:
            # Close all pubsub connections
            for channel in list(self._pubsubs.keys()):
                self._stop_pubsub(channel)

            # Close connection pool
            self._pool.disconnect()

        logger.info("RedisMessageBus shutdown complete")

    def health_check(self) -> bool:
        """Check if Redis connection is healthy."""
        try:
            return self._client.ping()
        except redis.RedisError:
            return False


def create_redis_bus(host: str = "localhost", port: int = 6379, **kwargs) -> RedisMessageBus:
    """
    Create a Redis message bus instance.

    Args:
        host: Redis host
        port: Redis port
        **kwargs: Additional configuration

    Returns:
        RedisMessageBus instance

    Raises:
        ImportError: If redis package is not installed
        redis.ConnectionError: If connection fails
    """
    return RedisMessageBus(host=host, port=port, **kwargs)
