"""
Tests for Agent OS Message Bus
"""

import pytest
import threading
import time
from typing import List
from uuid import uuid4

from src.messaging.models import (
    FlowRequest,
    FlowResponse,
    RequestContent,
    ResponseContent,
    MessageStatus,
    CheckStatus,
    create_request,
)
from src.messaging.bus import (
    InMemoryMessageBus,
    ChannelRouter,
    Subscription,
    ChannelStats,
)


class TestInMemoryMessageBus:
    """Tests for InMemoryMessageBus."""

    @pytest.fixture
    def bus(self):
        """Create a message bus instance."""
        bus = InMemoryMessageBus()
        yield bus
        bus.shutdown()

    @pytest.fixture
    def sample_request(self):
        """Create a sample request."""
        return create_request(
            source="user",
            destination="sage",
            intent="query.factual",
            prompt="What is AI?",
        )

    def test_create_bus(self, bus):
        """Create an in-memory message bus."""
        assert bus is not None

    def test_subscribe(self, bus):
        """Subscribe to a channel."""
        received = []

        def handler(msg):
            received.append(msg)

        subscription = bus.subscribe("test-channel", handler, "test-subscriber")

        assert subscription is not None
        assert subscription.channel == "test-channel"
        assert subscription.subscriber_name == "test-subscriber"
        assert subscription.is_active is True

    def test_publish_and_receive(self, bus, sample_request):
        """Publish and receive a message."""
        received = []

        def handler(msg):
            received.append(msg)

        bus.subscribe("test-channel", handler, "test-subscriber")

        success = bus.publish("test-channel", sample_request)

        assert success is True
        assert len(received) == 1
        assert received[0].source == "user"

    def test_multiple_subscribers(self, bus, sample_request):
        """Multiple subscribers receive the same message."""
        received_1 = []
        received_2 = []

        def handler_1(msg):
            received_1.append(msg)

        def handler_2(msg):
            received_2.append(msg)

        bus.subscribe("test-channel", handler_1, "subscriber-1")
        bus.subscribe("test-channel", handler_2, "subscriber-2")

        bus.publish("test-channel", sample_request)

        assert len(received_1) == 1
        assert len(received_2) == 1

    def test_unsubscribe(self, bus, sample_request):
        """Unsubscribe stops message delivery."""
        received = []

        def handler(msg):
            received.append(msg)

        subscription = bus.subscribe("test-channel", handler, "test-subscriber")

        # Receive first message
        bus.publish("test-channel", sample_request)
        assert len(received) == 1

        # Unsubscribe
        result = bus.unsubscribe(subscription)
        assert result is True

        # Should not receive second message
        bus.publish("test-channel", sample_request)
        assert len(received) == 1

    def test_publish_to_empty_channel(self, bus, sample_request):
        """Publishing to channel with no subscribers should succeed."""
        success = bus.publish("empty-channel", sample_request)

        # Not an error, just no one listening
        assert success is True

    def test_channel_stats(self, bus, sample_request):
        """Track channel statistics."""
        received = []

        def handler(msg):
            received.append(msg)

        bus.subscribe("test-channel", handler, "test-subscriber")

        bus.publish("test-channel", sample_request)
        bus.publish("test-channel", sample_request)

        stats = bus.get_channel_stats("test-channel")

        assert stats is not None
        assert stats.messages_published == 2
        assert stats.messages_delivered == 2
        assert stats.subscriber_count == 1

    def test_audit_log(self, bus, sample_request):
        """Messages are logged to audit trail."""
        bus.subscribe("test-channel", lambda m: None, "test-subscriber")
        bus.publish("test-channel", sample_request)

        audit = bus.get_audit_log()

        assert len(audit) >= 1
        assert audit[-1].source == "user"
        assert audit[-1].message_type == "request"

    def test_dead_letter_queue(self, bus, sample_request):
        """Failed messages go to dead letter queue."""

        def failing_handler(msg):
            raise Exception("Handler failed")

        bus.subscribe("test-channel", failing_handler, "failing-subscriber")
        bus.publish("test-channel", sample_request)

        dead_letters = bus.get_dead_letters()

        assert len(dead_letters) >= 1
        assert "Handler failed" in dead_letters[-1].failure_reason

    def test_retry_dead_letter(self, bus, sample_request):
        """Retry a dead letter message."""
        attempt_count = [0]

        def sometimes_failing_handler(msg):
            attempt_count[0] += 1
            if attempt_count[0] == 1:
                raise Exception("First attempt failed")
            # Second attempt succeeds

        bus.subscribe("test-channel", sometimes_failing_handler, "test-subscriber")

        # First publish fails
        bus.publish("test-channel", sample_request)

        dead_letters = bus.get_dead_letters()
        assert len(dead_letters) == 1

        # Retry should succeed
        success = bus.retry_dead_letter(0)
        assert success is True

        # Dead letter should be removed
        dead_letters = bus.get_dead_letters()
        assert len(dead_letters) == 0

    def test_get_subscriber_count(self, bus):
        """Get subscriber count for a channel."""
        bus.subscribe("test-channel", lambda m: None, "sub-1")
        bus.subscribe("test-channel", lambda m: None, "sub-2")

        count = bus.get_subscriber_count("test-channel")

        assert count == 2

    def test_get_all_channels(self, bus):
        """Get list of all channels."""
        bus.subscribe("channel-1", lambda m: None, "sub-1")
        bus.subscribe("channel-2", lambda m: None, "sub-2")

        channels = bus.get_all_channels()

        assert "channel-1" in channels
        assert "channel-2" in channels

    def test_clear_audit_log(self, bus, sample_request):
        """Clear the audit log."""
        bus.subscribe("test-channel", lambda m: None, "test-subscriber")
        bus.publish("test-channel", sample_request)

        assert len(bus.get_audit_log()) > 0

        cleared = bus.clear_audit_log()

        assert cleared > 0
        assert len(bus.get_audit_log()) == 0

    def test_shutdown(self, bus):
        """Shutdown the bus."""
        bus.subscribe("test-channel", lambda m: None, "test-subscriber")

        bus.shutdown()

        # Should not be able to publish after shutdown
        request = create_request(
            source="user",
            destination="sage",
            intent="query",
            prompt="Test",
        )
        success = bus.publish("test-channel", request)

        assert success is False

    def test_response_message(self, bus):
        """Publish and receive a response message."""
        received = []

        def handler(msg):
            received.append(msg)

        bus.subscribe("test-channel", handler, "test-subscriber")

        response = FlowResponse(
            request_id=uuid4(),
            source="sage",
            destination="user",
            status=MessageStatus.SUCCESS,
            content=ResponseContent(output="The answer"),
        )

        bus.publish("test-channel", response)

        assert len(received) == 1
        assert received[0].status == MessageStatus.SUCCESS

    def test_thread_safety(self, bus):
        """Bus should be thread-safe."""
        received = []
        lock = threading.Lock()

        def handler(msg):
            with lock:
                received.append(msg)

        bus.subscribe("test-channel", handler, "test-subscriber")

        def publisher():
            for i in range(100):
                request = create_request(
                    source="user",
                    destination="sage",
                    intent="query",
                    prompt=f"Message {i}",
                )
                bus.publish("test-channel", request)

        threads = [threading.Thread(target=publisher) for _ in range(5)]

        for t in threads:
            t.start()

        for t in threads:
            t.join()

        # Should have received all 500 messages
        assert len(received) == 500


class TestChannelRouter:
    """Tests for ChannelRouter."""

    @pytest.fixture
    def bus(self):
        """Create a message bus."""
        bus = InMemoryMessageBus()
        yield bus
        bus.shutdown()

    @pytest.fixture
    def router(self, bus):
        """Create a channel router."""
        return ChannelRouter(bus)

    def test_agent_channel_name(self):
        """Get agent channel name."""
        channel = ChannelRouter.agent_channel("sage")
        assert channel == "agent.sage"

    def test_intent_channel_name(self):
        """Get intent channel name."""
        channel = ChannelRouter.intent_channel("query.factual")
        assert channel == "intent.query.factual"

    def test_route_request(self, router, bus):
        """Route a request to an agent."""
        received = []

        bus.subscribe("agent.sage", lambda m: received.append(m), "sage")

        request = create_request(
            source="user",
            destination="sage",
            intent="query.factual",
            prompt="Test",
        )

        success = router.route_request(request)

        assert success is True
        assert len(received) == 1

    def test_route_response(self, router, bus):
        """Route a response to an agent."""
        received = []

        bus.subscribe("agent.user", lambda m: received.append(m), "user")

        response = FlowResponse(
            request_id=uuid4(),
            source="sage",
            destination="user",
            status=MessageStatus.SUCCESS,
            content=ResponseContent(output="Answer"),
        )

        success = router.route_response(response)

        assert success is True
        assert len(received) == 1

    def test_subscribe_agent(self, router):
        """Subscribe an agent to its channel."""
        received = []

        subscription = router.subscribe_agent("sage", lambda m: received.append(m))

        assert subscription.channel == "agent.sage"
        assert subscription.subscriber_name == "sage"

    def test_subscribe_intent(self, router, bus):
        """Subscribe to an intent channel."""
        received = []

        router.subscribe_intent(
            "query.factual",
            lambda m: received.append(m),
            "monitor",
        )

        request = create_request(
            source="user",
            destination="sage",
            intent="query.factual",
            prompt="Test",
        )

        router.route_request(request)

        # Intent channel should receive the message
        assert len(received) == 1

    def test_broadcast(self, router, bus):
        """Broadcast a message to all subscribers."""
        received = []

        router.subscribe_broadcast(lambda m: received.append(m), "monitor")

        request = create_request(
            source="system",
            destination="all",
            intent="system.announcement",
            prompt="System going offline",
        )

        router.broadcast(request)

        assert len(received) == 1


class TestMessageFlow:
    """Integration tests for complete message flows."""

    @pytest.fixture
    def bus(self):
        """Create a message bus."""
        bus = InMemoryMessageBus()
        yield bus
        bus.shutdown()

    def test_request_response_flow(self, bus):
        """Complete request-response flow."""
        router = ChannelRouter(bus)

        # Simulate user request to whisper (orchestrator)
        user_request = create_request(
            source="user",
            destination="whisper",
            intent="query.factual",
            prompt="What is machine learning?",
        )

        # Whisper receives and routes to sage
        whisper_received = []
        sage_received = []
        user_received = []

        router.subscribe_agent("whisper", lambda m: whisper_received.append(m))
        router.subscribe_agent("sage", lambda m: sage_received.append(m))
        router.subscribe_agent("user", lambda m: user_received.append(m))

        # User sends request
        router.route_request(user_request)
        assert len(whisper_received) == 1

        # Whisper routes to sage
        sage_request = create_request(
            source="whisper",
            destination="sage",
            intent="query.factual",
            prompt=user_request.content.prompt,
        )
        router.route_request(sage_request)
        assert len(sage_received) == 1

        # Sage sends response back to whisper
        sage_response = sage_received[0].create_response(
            source="sage",
            status=MessageStatus.SUCCESS,
            output="Machine learning is...",
        )
        sage_response.destination = "whisper"
        router.route_response(sage_response)

        # Whisper sends final response to user
        user_response = FlowResponse(
            request_id=user_request.request_id,
            source="whisper",
            destination="user",
            status=MessageStatus.SUCCESS,
            content=ResponseContent(output="Machine learning is..."),
        )
        router.route_response(user_response)

        assert len(user_received) == 1
        assert user_received[0].content.output == "Machine learning is..."
