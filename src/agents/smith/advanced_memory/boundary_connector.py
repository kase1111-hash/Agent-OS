"""
Boundary Daemon Connector

Integrates with Boundary-Daemon (https://github.com/kase1111-hash/boundary-daemon-)
for policy decisions, audit events, and tripwire alerts.

Features:
- Event ingestion from Boundary-Daemon
- Policy decision tracking
- Tripwire alert monitoring
- Mode transition handling
- Cryptographic audit verification
"""

import hashlib
import logging
import queue
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set

from .store import IntelligenceEntry, IntelligenceType, SecurityIntelligenceStore

logger = logging.getLogger(__name__)


class BoundaryMode(Enum):
    """Boundary daemon operating modes."""

    OPEN = auto()  # Full access
    RESTRICTED = auto()  # Limited access
    TRUSTED = auto()  # Trusted operations only
    AIRGAP = auto()  # No network access
    COLDROOM = auto()  # Minimal operation
    LOCKDOWN = auto()  # Emergency lockdown


class PolicyAction(Enum):
    """Policy decision actions."""

    ALLOW = auto()
    DENY = auto()
    ESCALATE = auto()
    LOG_ONLY = auto()


class MemoryClassification(Enum):
    """Memory classification levels from Boundary-Daemon."""

    PUBLIC = 1
    INTERNAL = 2
    CONFIDENTIAL = 3
    SECRET = 4
    CROWN_JEWEL = 5


@dataclass
class BoundaryEvent:
    """Event from Boundary-Daemon."""

    event_id: str
    timestamp: datetime
    event_type: str
    source: str

    # Context
    current_mode: BoundaryMode
    target_resource: str = ""
    action_requested: str = ""

    # Cryptographic verification
    signature: str = ""
    prev_hash: str = ""  # Hash chain for tamper evidence

    # Details
    details: Dict[str, Any] = field(default_factory=dict)
    tags: Set[str] = field(default_factory=set)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "event_id": self.event_id,
            "timestamp": self.timestamp.isoformat(),
            "event_type": self.event_type,
            "source": self.source,
            "current_mode": self.current_mode.name,
            "target_resource": self.target_resource,
            "action_requested": self.action_requested,
            "signature": self.signature,
            "prev_hash": self.prev_hash,
            "details": self.details,
            "tags": list(self.tags),
        }

    def to_intelligence_entry(self, severity: int = 2) -> IntelligenceEntry:
        """Convert to intelligence entry for storage."""
        return IntelligenceEntry(
            entry_id=f"boundary_{self.event_id}",
            entry_type=IntelligenceType.BOUNDARY_EVENT,
            timestamp=self.timestamp,
            source="boundary-daemon",
            severity=severity,
            category=self.event_type,
            summary=f"[{self.current_mode.name}] {self.event_type}: {self.action_requested} on {self.target_resource}",
            content=self.to_dict(),
            tags=self.tags | {"boundary-daemon"},
        )


@dataclass
class PolicyDecision:
    """A policy decision from Boundary-Daemon."""

    decision_id: str
    timestamp: datetime
    action: PolicyAction
    policy_rule: str

    # Request context
    requesting_agent: str
    target_resource: str
    operation: str
    current_mode: BoundaryMode

    # Decision details
    reason: str = ""
    constraints: List[str] = field(default_factory=list)
    memory_classification: Optional[MemoryClassification] = None

    # Audit
    requires_human_approval: bool = False
    cooldown_seconds: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "decision_id": self.decision_id,
            "timestamp": self.timestamp.isoformat(),
            "action": self.action.name,
            "policy_rule": self.policy_rule,
            "requesting_agent": self.requesting_agent,
            "target_resource": self.target_resource,
            "operation": self.operation,
            "current_mode": self.current_mode.name,
            "reason": self.reason,
            "constraints": self.constraints,
            "memory_classification": self.memory_classification.name if self.memory_classification else None,
            "requires_human_approval": self.requires_human_approval,
            "cooldown_seconds": self.cooldown_seconds,
        }

    def to_intelligence_entry(self) -> IntelligenceEntry:
        """Convert to intelligence entry for storage."""
        severity = {
            PolicyAction.ALLOW: 1,
            PolicyAction.DENY: 3,
            PolicyAction.ESCALATE: 4,
            PolicyAction.LOG_ONLY: 2,
        }.get(self.action, 2)

        return IntelligenceEntry(
            entry_id=f"policy_{self.decision_id}",
            entry_type=IntelligenceType.POLICY_DECISION,
            timestamp=self.timestamp,
            source="boundary-daemon",
            severity=severity,
            category="policy_decision",
            summary=f"[{self.action.name}] {self.operation} on {self.target_resource} by {self.requesting_agent}",
            content=self.to_dict(),
            tags={"policy", self.action.name.lower()},
        )


@dataclass
class TripwireAlert:
    """Tripwire alert from Boundary-Daemon."""

    alert_id: str
    timestamp: datetime
    tripwire_name: str
    severity: int  # 1-5

    # Trigger details
    trigger_source: str
    trigger_action: str
    violated_policy: str

    # Response
    response_action: str  # e.g., "LOCKDOWN", "ALERT", "LOG"
    auto_contained: bool = False

    # Evidence
    evidence: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "alert_id": self.alert_id,
            "timestamp": self.timestamp.isoformat(),
            "tripwire_name": self.tripwire_name,
            "severity": self.severity,
            "trigger_source": self.trigger_source,
            "trigger_action": self.trigger_action,
            "violated_policy": self.violated_policy,
            "response_action": self.response_action,
            "auto_contained": self.auto_contained,
            "evidence": self.evidence,
        }

    def to_intelligence_entry(self) -> IntelligenceEntry:
        """Convert to intelligence entry for storage."""
        return IntelligenceEntry(
            entry_id=f"tripwire_{self.alert_id}",
            entry_type=IntelligenceType.TRIPWIRE_ALERT,
            timestamp=self.timestamp,
            source="boundary-daemon",
            severity=self.severity,
            category="tripwire",
            summary=f"TRIPWIRE [{self.tripwire_name}]: {self.trigger_action} by {self.trigger_source}",
            content=self.to_dict(),
            tags={"tripwire", "security-violation", self.response_action.lower()},
        )


class BoundaryDaemonConnector:
    """
    Connects to Boundary-Daemon for security event ingestion.

    Supports:
    - REST API polling
    - Webhook receivers
    - Direct event injection (for testing)
    - Cryptographic audit verification
    """

    def __init__(
        self,
        store: SecurityIntelligenceStore,
        endpoint: Optional[str] = None,
        api_key: Optional[str] = None,
        on_event: Optional[Callable[[BoundaryEvent], None]] = None,
        on_policy: Optional[Callable[[PolicyDecision], None]] = None,
        on_tripwire: Optional[Callable[[TripwireAlert], None]] = None,
    ):
        """
        Initialize Boundary-Daemon connector.

        Args:
            store: Intelligence store for persisting events
            endpoint: Boundary-Daemon API endpoint
            api_key: API authentication key
            on_event: Callback for events
            on_policy: Callback for policy decisions
            on_tripwire: Callback for tripwire alerts
        """
        self.store = store
        self.endpoint = endpoint
        self.api_key = api_key
        self.on_event = on_event
        self.on_policy = on_policy
        self.on_tripwire = on_tripwire

        # Current state
        self._current_mode: BoundaryMode = BoundaryMode.OPEN
        self._connected = False

        # Event queue for processing
        self._event_queue: queue.Queue = queue.Queue()

        # Hash chain for audit verification
        self._last_hash: str = ""
        self._event_counter = 0

        # Thread safety
        self._lock = threading.RLock()

        # Background threads
        self._running = False
        self._poll_thread: Optional[threading.Thread] = None
        self._process_thread: Optional[threading.Thread] = None

        # Statistics
        self._stats = {
            "events_received": 0,
            "policy_decisions": 0,
            "tripwire_alerts": 0,
            "mode_changes": 0,
            "connection_errors": 0,
        }

    def connect(self) -> bool:
        """
        Connect to Boundary-Daemon.

        Returns:
            True if connection successful
        """
        if not self.endpoint:
            logger.info("No endpoint configured, running in local mode")
            self._connected = True
            return True

        try:
            import requests
        except ImportError:
            logger.error("requests library required for Boundary-Daemon connector")
            return False

        try:
            # Test connection with health check
            headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
            response = requests.get(
                f"{self.endpoint}/api/v1/health",
                headers=headers,
                timeout=10,
            )

            if response.status_code == 200:
                data = response.json()
                self._current_mode = BoundaryMode[data.get("mode", "OPEN")]
                self._connected = True
                logger.info(
                    f"Connected to Boundary-Daemon at {self.endpoint} "
                    f"(mode: {self._current_mode.name})"
                )
                return True
            else:
                logger.error(f"Boundary-Daemon connection failed: {response.status_code}")
                return False

        except Exception as e:
            logger.error(f"Boundary-Daemon connection error: {e}")
            self._stats["connection_errors"] += 1
            return False

    def disconnect(self) -> None:
        """Disconnect from Boundary-Daemon."""
        self._connected = False
        self.stop()
        logger.info("Disconnected from Boundary-Daemon")

    def start(self, poll_interval: int = 30) -> None:
        """
        Start background polling and processing.

        Args:
            poll_interval: Seconds between polls
        """
        if self._running:
            return

        self._running = True

        # Start event processing thread
        self._process_thread = threading.Thread(
            target=self._process_loop,
            daemon=True,
            name="BoundaryEventProcessor",
        )
        self._process_thread.start()

        # Start polling thread if endpoint configured
        if self.endpoint:
            self._poll_thread = threading.Thread(
                target=self._poll_loop,
                args=(poll_interval,),
                daemon=True,
                name="BoundaryPoller",
            )
            self._poll_thread.start()

        logger.info("Boundary-Daemon connector started")

    def stop(self) -> None:
        """Stop background threads."""
        self._running = False

        if self._poll_thread:
            self._poll_thread.join(timeout=5.0)
            self._poll_thread = None

        if self._process_thread:
            self._event_queue.put(None)  # Signal to stop
            self._process_thread.join(timeout=5.0)
            self._process_thread = None

        logger.info("Boundary-Daemon connector stopped")

    def inject_event(self, event: BoundaryEvent) -> None:
        """
        Inject an event for processing (for testing or direct integration).

        Args:
            event: Event to inject
        """
        self._event_queue.put(("event", event))

    def inject_policy_decision(self, decision: PolicyDecision) -> None:
        """
        Inject a policy decision for processing.

        Args:
            decision: Policy decision to inject
        """
        self._event_queue.put(("policy", decision))

    def inject_tripwire_alert(self, alert: TripwireAlert) -> None:
        """
        Inject a tripwire alert for processing.

        Args:
            alert: Tripwire alert to inject
        """
        self._event_queue.put(("tripwire", alert))

    def set_mode(self, mode: BoundaryMode) -> None:
        """
        Set the current boundary mode (for local tracking).

        Args:
            mode: New boundary mode
        """
        with self._lock:
            old_mode = self._current_mode
            self._current_mode = mode

            if old_mode != mode:
                self._stats["mode_changes"] += 1
                logger.info(f"Boundary mode changed: {old_mode.name} -> {mode.name}")

                # Create mode change event
                event = BoundaryEvent(
                    event_id=f"mode_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    timestamp=datetime.now(),
                    event_type="mode_transition",
                    source="boundary-daemon",
                    current_mode=mode,
                    details={"old_mode": old_mode.name, "new_mode": mode.name},
                    tags={"mode-change"},
                )
                self.inject_event(event)

    def get_mode(self) -> BoundaryMode:
        """Get current boundary mode."""
        return self._current_mode

    def verify_audit_chain(self, events: List[BoundaryEvent]) -> bool:
        """
        Verify the hash chain of audit events.

        Args:
            events: Events to verify (must be in order)

        Returns:
            True if chain is valid
        """
        if not events:
            return True

        prev_hash = ""
        for event in events:
            if event.prev_hash != prev_hash:
                logger.error(f"Audit chain broken at event {event.event_id}")
                return False

            # Calculate hash for this event
            hash_input = f"{event.event_id}:{event.timestamp.isoformat()}:{event.event_type}:{prev_hash}"
            current_hash = hashlib.sha256(hash_input.encode()).hexdigest()

            # Would verify signature here with Ed25519 in production
            # For now, just check chain continuity

            prev_hash = current_hash

        return True

    def _poll_loop(self, interval: int) -> None:
        """Background polling loop."""
        last_poll = datetime.now() - timedelta(seconds=interval)

        while self._running:
            try:
                now = datetime.now()

                # Fetch new events
                events = self._fetch_events(since=last_poll)
                for event in events:
                    self._event_queue.put(("event", event))

                # Fetch new policy decisions
                decisions = self._fetch_policy_decisions(since=last_poll)
                for decision in decisions:
                    self._event_queue.put(("policy", decision))

                # Fetch new tripwire alerts
                alerts = self._fetch_tripwire_alerts(since=last_poll)
                for alert in alerts:
                    self._event_queue.put(("tripwire", alert))

                last_poll = now

            except Exception as e:
                logger.error(f"Boundary-Daemon poll error: {e}")
                self._stats["connection_errors"] += 1

            # Sleep
            for _ in range(interval):
                if not self._running:
                    break
                time.sleep(1)

    def _fetch_events(self, since: datetime) -> List[BoundaryEvent]:
        """Fetch events from Boundary-Daemon API."""
        if not self.endpoint:
            return []

        try:
            import requests
            headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}

            response = requests.get(
                f"{self.endpoint}/api/v1/events",
                headers=headers,
                params={"since": since.isoformat()},
                timeout=10,
            )

            if response.status_code == 200:
                data = response.json()
                return [self._parse_event(e) for e in data.get("events", [])]

        except Exception as e:
            logger.error(f"Error fetching events: {e}")

        return []

    def _fetch_policy_decisions(self, since: datetime) -> List[PolicyDecision]:
        """Fetch policy decisions from Boundary-Daemon API."""
        if not self.endpoint:
            return []

        try:
            import requests
            headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}

            response = requests.get(
                f"{self.endpoint}/api/v1/policy/decisions",
                headers=headers,
                params={"since": since.isoformat()},
                timeout=10,
            )

            if response.status_code == 200:
                data = response.json()
                return [self._parse_policy_decision(d) for d in data.get("decisions", [])]

        except Exception as e:
            logger.error(f"Error fetching policy decisions: {e}")

        return []

    def _fetch_tripwire_alerts(self, since: datetime) -> List[TripwireAlert]:
        """Fetch tripwire alerts from Boundary-Daemon API."""
        if not self.endpoint:
            return []

        try:
            import requests
            headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}

            response = requests.get(
                f"{self.endpoint}/api/v1/tripwires/alerts",
                headers=headers,
                params={"since": since.isoformat()},
                timeout=10,
            )

            if response.status_code == 200:
                data = response.json()
                return [self._parse_tripwire_alert(a) for a in data.get("alerts", [])]

        except Exception as e:
            logger.error(f"Error fetching tripwire alerts: {e}")

        return []

    def _parse_event(self, data: Dict[str, Any]) -> BoundaryEvent:
        """Parse event from API response."""
        return BoundaryEvent(
            event_id=data.get("id", ""),
            timestamp=datetime.fromisoformat(data.get("timestamp", datetime.now().isoformat())),
            event_type=data.get("type", "unknown"),
            source=data.get("source", "boundary-daemon"),
            current_mode=BoundaryMode[data.get("mode", "OPEN")],
            target_resource=data.get("target", ""),
            action_requested=data.get("action", ""),
            signature=data.get("signature", ""),
            prev_hash=data.get("prev_hash", ""),
            details=data.get("details", {}),
            tags=set(data.get("tags", [])),
        )

    def _parse_policy_decision(self, data: Dict[str, Any]) -> PolicyDecision:
        """Parse policy decision from API response."""
        return PolicyDecision(
            decision_id=data.get("id", ""),
            timestamp=datetime.fromisoformat(data.get("timestamp", datetime.now().isoformat())),
            action=PolicyAction[data.get("action", "DENY")],
            policy_rule=data.get("rule", ""),
            requesting_agent=data.get("agent", ""),
            target_resource=data.get("target", ""),
            operation=data.get("operation", ""),
            current_mode=BoundaryMode[data.get("mode", "OPEN")],
            reason=data.get("reason", ""),
            constraints=data.get("constraints", []),
            memory_classification=MemoryClassification[data["classification"]] if data.get("classification") else None,
            requires_human_approval=data.get("requires_approval", False),
            cooldown_seconds=data.get("cooldown", 0),
        )

    def _parse_tripwire_alert(self, data: Dict[str, Any]) -> TripwireAlert:
        """Parse tripwire alert from API response."""
        return TripwireAlert(
            alert_id=data.get("id", ""),
            timestamp=datetime.fromisoformat(data.get("timestamp", datetime.now().isoformat())),
            tripwire_name=data.get("name", ""),
            severity=data.get("severity", 3),
            trigger_source=data.get("source", ""),
            trigger_action=data.get("action", ""),
            violated_policy=data.get("policy", ""),
            response_action=data.get("response", "LOG"),
            auto_contained=data.get("contained", False),
            evidence=data.get("evidence", {}),
        )

    def _process_loop(self) -> None:
        """Background event processing loop."""
        while self._running:
            try:
                item = self._event_queue.get(timeout=1.0)

                if item is None:
                    break

                item_type, data = item

                if item_type == "event":
                    self._process_event(data)
                elif item_type == "policy":
                    self._process_policy_decision(data)
                elif item_type == "tripwire":
                    self._process_tripwire_alert(data)

            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error processing boundary event: {e}")

    def _process_event(self, event: BoundaryEvent) -> None:
        """Process a boundary event."""
        # Determine severity based on mode and event type
        severity = 2
        if event.current_mode in (BoundaryMode.LOCKDOWN, BoundaryMode.COLDROOM):
            severity = 4
        elif event.current_mode == BoundaryMode.AIRGAP:
            severity = 3
        elif "violation" in event.event_type.lower():
            severity = 3

        # Convert to intelligence entry and store
        entry = event.to_intelligence_entry(severity=severity)
        self.store.add(entry)

        self._stats["events_received"] += 1

        # Notify callback
        if self.on_event:
            self.on_event(event)

    def _process_policy_decision(self, decision: PolicyDecision) -> None:
        """Process a policy decision."""
        # Store as intelligence entry
        entry = decision.to_intelligence_entry()
        self.store.add(entry)

        self._stats["policy_decisions"] += 1

        # Notify callback
        if self.on_policy:
            self.on_policy(decision)

    def _process_tripwire_alert(self, alert: TripwireAlert) -> None:
        """Process a tripwire alert."""
        # Store as intelligence entry
        entry = alert.to_intelligence_entry()
        self.store.add(entry)

        self._stats["tripwire_alerts"] += 1

        # Notify callback
        if self.on_tripwire:
            self.on_tripwire(alert)

    def get_stats(self) -> Dict[str, Any]:
        """Get connector statistics."""
        return {
            **self._stats,
            "connected": self._connected,
            "current_mode": self._current_mode.name,
            "is_running": self._running,
        }


def create_boundary_connector(
    store: SecurityIntelligenceStore,
    endpoint: Optional[str] = None,
    api_key: Optional[str] = None,
    auto_start: bool = True,
) -> BoundaryDaemonConnector:
    """
    Factory function to create a Boundary-Daemon connector.

    Args:
        store: Intelligence store
        endpoint: Boundary-Daemon API endpoint
        api_key: API key
        auto_start: Start polling automatically

    Returns:
        Configured BoundaryDaemonConnector
    """
    connector = BoundaryDaemonConnector(
        store=store,
        endpoint=endpoint,
        api_key=api_key,
    )

    if auto_start:
        connector.connect()
        connector.start()

    return connector
