"""Tests for the dreaming status service (src/web/dreaming.py)."""

import threading
import time

from src.web.dreaming import DreamingService


def _get_status_with_timeout(service: DreamingService, timeout: float = 2.0) -> dict:
    """Call get_status in a thread so a deadlock fails the test instead of hanging it."""
    result = {}
    worker = threading.Thread(target=lambda: result.update(service.get_status()), daemon=True)
    worker.start()
    worker.join(timeout)
    assert not worker.is_alive(), "get_status() deadlocked"
    return result


class TestDreamingService:
    def test_initial_status_is_idle(self):
        status = DreamingService().get_status()
        assert status["phase"] == "idle"
        assert status["message"] == "Idle"
        assert status["operations_count"] == 0

    def test_first_update_applies_immediately(self):
        service = DreamingService(throttle_interval=60)
        service.start("chat")
        status = service.get_status()
        assert status["phase"] == "starting"
        assert status["operation"] == "chat"

    def test_throttled_update_is_held_back(self):
        service = DreamingService(throttle_interval=60)
        service.start("chat")
        service.running("chat")
        assert service.get_status()["phase"] == "starting"

    def test_pending_update_applied_after_throttle_without_deadlock(self):
        # Regression: get_status() held the lock while applying the pending
        # update, which re-acquired the same non-reentrant lock and froze the
        # event loop thread (the UI polls this endpoint every few seconds).
        service = DreamingService(throttle_interval=0.05)
        service.start("chat")
        service.complete("chat")  # throttled -> pending
        time.sleep(0.06)

        status = _get_status_with_timeout(service)

        assert status["phase"] == "completed"
        assert status["message"] == "Completed: chat"
        assert status["operations_count"] == 1

    def test_completed_returns_to_idle_after_delay(self):
        service = DreamingService(throttle_interval=0)
        service.IDLE_DELAY = 0.01
        service.start("chat")
        service.complete("chat")
        time.sleep(0.02)
        assert _get_status_with_timeout(service)["phase"] == "idle"

    def test_reset(self):
        service = DreamingService(throttle_interval=0)
        service.start("chat")
        service.complete("chat")
        service.reset()
        status = service.get_status()
        assert status["phase"] == "idle"
        assert status["operations_count"] == 0
