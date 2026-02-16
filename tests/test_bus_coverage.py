"""
Tests for InMemoryMessageBus — covers unsubscribe, channel stats,
all channel stats, and shutdown edge cases.

Targeted at boosting bus.py coverage from ~84% to 90%+.
"""

import pytest

from src.messaging.bus import InMemoryMessageBus
from src.messaging.exceptions import BusShutdownError
from src.messaging.models import FlowRequest, RequestContent


def _make_request(prompt="test"):
    return FlowRequest(
        source="test",
        destination="test_agent",
        intent="query.factual",
        content=RequestContent(prompt=prompt),
    )


class TestBusUnsubscribe:
    def test_unsubscribe_existing(self):
        """Unsubscribing active subscription returns True."""
        bus = InMemoryMessageBus()
        sub = bus.subscribe("ch1", lambda m: None, "sub1")
        result = bus.unsubscribe(sub)
        assert result is True
        bus.shutdown()

    def test_unsubscribe_nonexistent(self):
        """Unsubscribing unknown subscription returns False."""
        bus = InMemoryMessageBus()
        sub = bus.subscribe("ch1", lambda m: None, "sub1")
        bus.unsubscribe(sub)
        # Second unsubscribe should return False
        result = bus.unsubscribe(sub)
        assert result is False
        bus.shutdown()


class TestBusChannelStats:
    def test_get_channel_stats(self):
        """Channel stats track published messages."""
        bus = InMemoryMessageBus()
        bus.subscribe("ch1", lambda m: None, "sub1")
        bus.publish("ch1", _make_request())
        stats = bus.get_channel_stats("ch1")
        assert stats is not None
        assert stats.messages_published >= 1
        bus.shutdown()

    def test_get_channel_stats_nonexistent(self):
        """Stats for unsubscribed channel returns None."""
        bus = InMemoryMessageBus()
        stats = bus.get_channel_stats("nonexistent")
        assert stats is None
        bus.shutdown()

    def test_get_all_channel_stats(self):
        """get_all_channel_stats returns dict of all channels."""
        bus = InMemoryMessageBus()
        bus.subscribe("ch1", lambda m: None, "sub1")
        bus.subscribe("ch2", lambda m: None, "sub2")
        bus.publish("ch1", _make_request())
        bus.publish("ch2", _make_request())
        all_stats = bus.get_all_channel_stats()
        assert "ch1" in all_stats
        assert "ch2" in all_stats
        bus.shutdown()


class TestBusShutdown:
    def test_publish_after_shutdown_returns_false(self):
        """Publishing after shutdown should return False."""
        bus = InMemoryMessageBus()
        bus.shutdown()
        result = bus.publish("ch1", _make_request())
        assert result is False

    def test_double_shutdown(self):
        """Double shutdown should not raise."""
        bus = InMemoryMessageBus()
        bus.shutdown()
        bus.shutdown()  # Should not raise


class TestBusMultipleSubscribers:
    def test_multiple_subscribers_all_receive(self):
        """All subscribers on a channel receive the message."""
        bus = InMemoryMessageBus()
        results_a = []
        results_b = []
        bus.subscribe("ch1", lambda m: results_a.append(m), "sub_a")
        bus.subscribe("ch1", lambda m: results_b.append(m), "sub_b")
        bus.publish("ch1", _make_request())
        assert len(results_a) == 1
        assert len(results_b) == 1
        bus.shutdown()

    def test_partial_failure_delivers_to_others(self):
        """If one subscriber fails, others still receive."""
        bus = InMemoryMessageBus()
        results = []

        def failing(m):
            raise ValueError("fail")

        def succeeding(m):
            results.append(m)

        bus.subscribe("ch1", failing, "failer")
        bus.subscribe("ch1", succeeding, "succeeder")

        # Should not raise — partial delivery is OK
        bus.publish("ch1", _make_request())
        assert len(results) == 1
        bus.shutdown()

    def test_no_subscribers_still_succeeds(self):
        """Publishing to a channel with no subscribers returns True."""
        bus = InMemoryMessageBus()
        result = bus.publish("empty_channel", _make_request())
        assert result is True
        bus.shutdown()
