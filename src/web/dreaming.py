"""
Dreaming Status Service

Provides a lightweight status indicator showing internal system activity.
Updates are throttled to every 5 seconds to minimize performance impact.

Usage:
    from src.web.dreaming import dreaming

    # Mark operation start
    dreaming.start("Processing user request")

    # Mark operation complete
    dreaming.complete("Processing user request")

    # Get current status
    status = dreaming.get_status()
"""

import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)

# Throttle interval in seconds
THROTTLE_INTERVAL = 5.0


@dataclass
class DreamingStatus:
    """Current dreaming status."""

    message: str = "Idle"
    phase: str = "idle"  # idle, starting, running, completed
    operation: Optional[str] = None
    updated_at: datetime = field(default_factory=datetime.utcnow)
    operations_count: int = 0


class DreamingService:
    """
    Lightweight service for tracking internal system activity.

    Designed to have minimal performance impact:
    - Updates throttled to every 5 seconds
    - Non-blocking operations
    - Simple state machine (no complex tracking)
    - Auto-returns to idle after completion
    """

    # How long to show "Completed" before returning to idle
    IDLE_DELAY = 3.0

    def __init__(self, throttle_interval: float = THROTTLE_INTERVAL):
        self._status = DreamingStatus()
        self._lock = threading.Lock()
        self._throttle_interval = throttle_interval
        self._last_update_time: float = 0
        self._pending_message: Optional[str] = None
        self._pending_phase: Optional[str] = None
        self._completed_at: Optional[float] = None  # Track when completed

    def _can_update(self) -> bool:
        """Check if enough time has passed for an update."""
        current_time = time.monotonic()
        return (current_time - self._last_update_time) >= self._throttle_interval

    def _do_update(self, message: str, phase: str, operation: Optional[str] = None) -> None:
        """Perform the actual status update."""
        with self._lock:
            self._status.message = message
            self._status.phase = phase
            self._status.operation = operation
            self._status.updated_at = datetime.utcnow()
            if phase == "completed":
                self._status.operations_count += 1
            self._last_update_time = time.monotonic()
            self._pending_message = None
            self._pending_phase = None

    def start(self, operation: str) -> None:
        """
        Mark an operation as starting.

        Args:
            operation: Description of the operation
        """
        if self._can_update():
            self._do_update(
                message=f"Starting: {operation}",
                phase="starting",
                operation=operation
            )
        else:
            # Store for potential later update
            self._pending_message = f"Starting: {operation}"
            self._pending_phase = "starting"

    def running(self, operation: str) -> None:
        """
        Mark an operation as running.

        Args:
            operation: Description of the operation
        """
        if self._can_update():
            self._do_update(
                message=f"Running: {operation}",
                phase="running",
                operation=operation
            )
        else:
            self._pending_message = f"Running: {operation}"
            self._pending_phase = "running"

    def complete(self, operation: str) -> None:
        """
        Mark an operation as complete.

        Args:
            operation: Description of the operation
        """
        if self._can_update():
            self._do_update(
                message=f"Completed: {operation}",
                phase="completed",
                operation=operation
            )
            self._completed_at = time.monotonic()
        else:
            self._pending_message = f"Completed: {operation}"
            self._pending_phase = "completed"
            self._completed_at = time.monotonic()

    def idle(self) -> None:
        """Mark system as idle."""
        if self._can_update():
            self._do_update(message="Idle", phase="idle")

    def get_status(self) -> dict:
        """
        Get the current dreaming status.

        Returns:
            Dictionary with current status information
        """
        with self._lock:
            # If we have pending updates and can update now, apply them
            if self._pending_message and self._can_update():
                self._do_update(
                    self._pending_message,
                    self._pending_phase or "running",
                    self._status.operation
                )

            # Auto-return to idle after completion delay
            if (self._status.phase == "completed" and
                self._completed_at is not None and
                (time.monotonic() - self._completed_at) >= self.IDLE_DELAY):
                self._status.message = "Idle"
                self._status.phase = "idle"
                self._status.operation = None
                self._completed_at = None

            return {
                "message": self._status.message,
                "phase": self._status.phase,
                "operation": self._status.operation,
                "updated_at": self._status.updated_at.isoformat(),
                "operations_count": self._status.operations_count,
                "throttle_interval": self._throttle_interval,
            }

    def reset(self) -> None:
        """Reset the dreaming status to initial state."""
        with self._lock:
            self._status = DreamingStatus()
            self._last_update_time = 0
            self._pending_message = None
            self._pending_phase = None
            self._completed_at = None


# Global singleton instance
_dreaming: Optional[DreamingService] = None


def get_dreaming() -> DreamingService:
    """Get the global dreaming service instance."""
    global _dreaming
    if _dreaming is None:
        _dreaming = DreamingService()
    return _dreaming


# Convenience alias
dreaming = get_dreaming()
