"""
Smith Daemon Tripwire System

Tripwires are security triggers that activate on specific conditions.
Once triggered, they cannot be reset without human intervention.

This is part of Agent Smith's system-level enforcement mechanism within Agent-OS,
distinct from the external boundary-daemon project.
"""

import hashlib
import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


class TripwireType(Enum):
    """Types of tripwires."""

    NETWORK = auto()  # Network access detected
    PROCESS = auto()  # Unauthorized process detected
    FILE = auto()  # Critical file modified
    HARDWARE = auto()  # Hardware tampering detected
    TIME = auto()  # Time manipulation detected
    MEMORY = auto()  # Memory tampering detected
    CUSTOM = auto()  # Custom tripwire


class TripwireState(Enum):
    """Tripwire states."""

    ARMED = auto()  # Tripwire is active and monitoring
    TRIGGERED = auto()  # Tripwire has been triggered
    DISABLED = auto()  # Tripwire is disabled (requires human)
    RESET = auto()  # Tripwire reset by human (temporary)


@dataclass
class TripwireEvent:
    """Event from a triggered tripwire."""

    tripwire_id: str
    tripwire_type: TripwireType
    timestamp: datetime
    trigger_reason: str
    trigger_data: Dict[str, Any] = field(default_factory=dict)
    severity: int = 1  # 1-5, 5 being most severe

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            "tripwire_id": self.tripwire_id,
            "tripwire_type": self.tripwire_type.name,
            "timestamp": self.timestamp.isoformat(),
            "trigger_reason": self.trigger_reason,
            "trigger_data": self.trigger_data,
            "severity": self.severity,
        }


@dataclass
class Tripwire:
    """A single tripwire definition."""

    id: str
    tripwire_type: TripwireType
    description: str
    condition: Callable[[], bool]  # Returns True if triggered
    severity: int = 3
    state: TripwireState = TripwireState.ARMED
    triggered_at: Optional[datetime] = None
    trigger_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def check(self) -> Optional[TripwireEvent]:
        """
        Check if tripwire should be triggered.

        Returns:
            TripwireEvent if triggered, None otherwise
        """
        if self.state != TripwireState.ARMED:
            return None

        try:
            if self.condition():
                self.state = TripwireState.TRIGGERED
                self.triggered_at = datetime.now()
                self.trigger_count += 1

                return TripwireEvent(
                    tripwire_id=self.id,
                    tripwire_type=self.tripwire_type,
                    timestamp=self.triggered_at,
                    trigger_reason=self.description,
                    trigger_data=self.metadata,
                    severity=self.severity,
                )

        except Exception as e:
            logger.error(f"Tripwire {self.id} check error: {e}")

        return None

    def reset(self, authorization_code: str) -> bool:
        """
        Reset tripwire (requires human authorization).

        Args:
            authorization_code: Code from human to authorize reset

        Returns:
            True if reset successful
        """
        # In production, this would verify cryptographic authorization
        if len(authorization_code) >= 8:
            self.state = TripwireState.ARMED
            logger.info(f"Tripwire {self.id} reset with authorization")
            return True
        return False


class TripwireSystem:
    """
    Tripwire management system.

    Monitors and manages all tripwires, triggering enforcement
    actions when tripwires are activated.
    """

    # Check interval
    DEFAULT_CHECK_INTERVAL = 0.5

    def __init__(
        self,
        check_interval: float = DEFAULT_CHECK_INTERVAL,
        on_trigger: Optional[Callable[[TripwireEvent], None]] = None,
    ):
        """
        Initialize tripwire system.

        Args:
            check_interval: How often to check tripwires
            on_trigger: Callback when any tripwire triggers
        """
        self.check_interval = check_interval
        self.on_trigger = on_trigger

        self._tripwires: Dict[str, Tripwire] = {}
        self._events: List[TripwireEvent] = []
        self._running = False
        self._check_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

        # Install default tripwires
        self._install_default_tripwires()

    def add_tripwire(self, tripwire: Tripwire) -> None:
        """Add a tripwire to the system."""
        with self._lock:
            self._tripwires[tripwire.id] = tripwire
            logger.info(f"Tripwire added: {tripwire.id}")

    def remove_tripwire(self, tripwire_id: str) -> bool:
        """Remove a tripwire (requires it to be disabled)."""
        with self._lock:
            tripwire = self._tripwires.get(tripwire_id)
            if tripwire and tripwire.state == TripwireState.DISABLED:
                del self._tripwires[tripwire_id]
                return True
            return False

    def get_tripwire(self, tripwire_id: str) -> Optional[Tripwire]:
        """Get a tripwire by ID."""
        return self._tripwires.get(tripwire_id)

    def list_tripwires(self) -> List[Tripwire]:
        """List all tripwires."""
        return list(self._tripwires.values())

    def get_triggered(self) -> List[Tripwire]:
        """Get all triggered tripwires."""
        return [tw for tw in self._tripwires.values() if tw.state == TripwireState.TRIGGERED]

    def get_events(self) -> List[TripwireEvent]:
        """Get all tripwire events."""
        return list(self._events)

    def is_triggered(self) -> bool:
        """Check if any tripwire is triggered."""
        return any(tw.state == TripwireState.TRIGGERED for tw in self._tripwires.values())

    def start(self) -> None:
        """Start the tripwire monitoring."""
        if self._running:
            return

        self._running = True
        self._check_thread = threading.Thread(
            target=self._check_loop,
            daemon=True,
            name="TripwireSystem",
        )
        self._check_thread.start()
        logger.info("Tripwire system started")

    def stop(self) -> None:
        """Stop the tripwire monitoring."""
        self._running = False
        if self._check_thread:
            self._check_thread.join(timeout=5.0)
            self._check_thread = None
        logger.info("Tripwire system stopped")

    def check_all(self) -> List[TripwireEvent]:
        """
        Check all tripwires immediately.

        Returns:
            List of triggered events
        """
        events = []

        with self._lock:
            for tripwire in self._tripwires.values():
                event = tripwire.check()
                if event:
                    events.append(event)
                    self._events.append(event)

                    if self.on_trigger:
                        try:
                            self.on_trigger(event)
                        except Exception as e:
                            logger.error(f"Trigger callback error: {e}")

        return events

    def reset_tripwire(
        self,
        tripwire_id: str,
        authorization_code: str,
    ) -> bool:
        """
        Reset a triggered tripwire.

        Args:
            tripwire_id: ID of tripwire to reset
            authorization_code: Human authorization code

        Returns:
            True if reset successful
        """
        tripwire = self._tripwires.get(tripwire_id)
        if tripwire:
            return tripwire.reset(authorization_code)
        return False

    def reset_all(self, authorization_code: str) -> int:
        """
        Reset all triggered tripwires.

        Args:
            authorization_code: Human authorization code

        Returns:
            Number of tripwires reset
        """
        count = 0
        for tripwire in self._tripwires.values():
            if tripwire.state == TripwireState.TRIGGERED:
                if tripwire.reset(authorization_code):
                    count += 1
        return count

    def _check_loop(self) -> None:
        """Main checking loop."""
        while self._running:
            try:
                self.check_all()
            except Exception as e:
                logger.error(f"Check loop error: {e}")

            time.sleep(self.check_interval)

    def _install_default_tripwires(self) -> None:
        """Install default security tripwires."""
        # Network tripwire - detects external network access
        self.add_tripwire(
            Tripwire(
                id="network_access",
                tripwire_type=TripwireType.NETWORK,
                description="External network access detected",
                condition=self._check_network_access,
                severity=5,
            )
        )

        # Time manipulation tripwire
        self.add_tripwire(
            Tripwire(
                id="time_manipulation",
                tripwire_type=TripwireType.TIME,
                description="System time manipulation detected",
                condition=self._check_time_manipulation,
                severity=4,
            )
        )

        # Critical file modification tripwire
        self.add_tripwire(
            Tripwire(
                id="config_modification",
                tripwire_type=TripwireType.FILE,
                description="Critical configuration file modified",
                condition=self._check_config_files,
                severity=4,
                metadata={"watched_files": []},
            )
        )

    def _check_network_access(self) -> bool:
        """Check for external network access."""
        try:
            import socket

            # Try to resolve external domain
            socket.setdefaulttimeout(0.1)
            socket.getaddrinfo("8.8.8.8", 53)
            # If we can resolve, network is available
            # This is a tripwire, so detecting network = triggered
            return False  # Don't trigger just for DNS capability
        except (socket.gaierror, socket.timeout, OSError):
            return False

    def _check_time_manipulation(self) -> bool:
        """Check for time manipulation."""
        # In production, would compare monotonic clock vs wall clock
        # and detect significant drift
        return False

    def _check_config_files(self) -> bool:
        """Check if critical config files have been modified."""
        # In production, would verify file hashes
        return False


def create_tripwire_system(
    check_interval: float = 0.5,
    on_trigger: Optional[Callable[[TripwireEvent], None]] = None,
) -> TripwireSystem:
    """Factory function to create a tripwire system."""
    return TripwireSystem(
        check_interval=check_interval,
        on_trigger=on_trigger,
    )


# Pre-built tripwire conditions for common scenarios


def create_file_tripwire(
    tripwire_id: str,
    file_path: Path,
    description: str = "File modification detected",
) -> Tripwire:
    """Create a tripwire for file modification detection."""
    initial_hash = None

    def compute_hash() -> Optional[str]:
        try:
            with open(file_path, "rb") as f:
                return hashlib.sha256(f.read()).hexdigest()
        except (FileNotFoundError, PermissionError):
            return None

    # Capture initial hash
    initial_hash = compute_hash()

    def check_file() -> bool:
        current_hash = compute_hash()
        if initial_hash is None:
            return False
        return current_hash != initial_hash

    return Tripwire(
        id=tripwire_id,
        tripwire_type=TripwireType.FILE,
        description=description,
        condition=check_file,
        severity=4,
        metadata={"file_path": str(file_path), "initial_hash": initial_hash},
    )


def create_process_tripwire(
    tripwire_id: str,
    forbidden_processes: Set[str],
    description: str = "Forbidden process detected",
) -> Tripwire:
    """Create a tripwire for forbidden process detection."""

    def check_processes() -> bool:
        try:
            proc_path = Path("/proc")
            if not proc_path.exists():
                return False

            for pid_dir in proc_path.iterdir():
                if pid_dir.name.isdigit():
                    try:
                        comm = (pid_dir / "comm").read_text().strip()
                        if comm in forbidden_processes:
                            return True
                    except (PermissionError, FileNotFoundError):
                        continue

        except Exception:
            pass
        return False

    return Tripwire(
        id=tripwire_id,
        tripwire_type=TripwireType.PROCESS,
        description=description,
        condition=check_processes,
        severity=5,
        metadata={"forbidden": list(forbidden_processes)},
    )
