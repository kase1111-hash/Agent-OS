"""
Security Intelligence Store

A tiered storage system for security intelligence with configurable retention policies.
Modeled after Boundary-SIEM's ClickHouse-based storage with hot/warm/cold tiers.

Features:
- Hot tier (7 days): Immediate access, full detail retention
- Warm tier (30 days): Aggregated data, reduced detail
- Cold tier (365 days): Summary data, archival storage
- Query optimization with indexes on common fields
- Automatic tier migration based on age
"""

import json
import logging
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


class IntelligenceType(Enum):
    """Types of intelligence entries."""

    # Events from external sources
    SIEM_EVENT = auto()  # Event from Boundary-SIEM
    BOUNDARY_EVENT = auto()  # Event from Boundary-Daemon
    TRIPWIRE_ALERT = auto()  # Tripwire violation
    POLICY_DECISION = auto()  # Policy enforcement decision

    # Internal events
    ATTACK_DETECTED = auto()  # Attack detection event
    INCIDENT = auto()  # Security incident
    VALIDATION_FAILURE = auto()  # Pre/post validation failure
    REFUSAL = auto()  # Request refusal

    # Synthesized intelligence
    PATTERN = auto()  # Detected pattern
    TREND = auto()  # Trend analysis result
    CORRELATION = auto()  # Correlated events
    ANOMALY = auto()  # Behavioral anomaly

    # Reference data
    THREAT_INTEL = auto()  # Threat intelligence indicator
    BASELINE = auto()  # Behavioral baseline snapshot


class RetentionTier(Enum):
    """Data retention tiers."""

    HOT = "hot"  # Full detail, immediate access (7 days)
    WARM = "warm"  # Aggregated, reduced detail (30 days)
    COLD = "cold"  # Summary, archival (365 days)
    PERMANENT = "permanent"  # Never expires


@dataclass
class IntelligenceEntry:
    """A single intelligence entry in the store."""

    entry_id: str
    entry_type: IntelligenceType
    timestamp: datetime
    source: str  # Source system (e.g., "boundary-siem", "boundary-daemon")
    severity: int  # 1-5 severity score
    category: str
    summary: str

    # Detailed content (reduced in warm/cold tiers)
    content: Dict[str, Any] = field(default_factory=dict)

    # Correlation and enrichment
    indicators: List[str] = field(default_factory=list)  # IOCs
    mitre_tactics: List[str] = field(default_factory=list)
    related_entries: List[str] = field(default_factory=list)
    tags: Set[str] = field(default_factory=set)

    # Metadata
    tier: RetentionTier = RetentionTier.HOT
    created_at: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None
    access_count: int = 0
    last_accessed: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "entry_id": self.entry_id,
            "entry_type": self.entry_type.name,
            "timestamp": self.timestamp.isoformat(),
            "source": self.source,
            "severity": self.severity,
            "category": self.category,
            "summary": self.summary,
            "content": self.content,
            "indicators": self.indicators,
            "mitre_tactics": self.mitre_tactics,
            "related_entries": self.related_entries,
            "tags": list(self.tags),
            "tier": self.tier.value,
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "access_count": self.access_count,
            "last_accessed": self.last_accessed.isoformat() if self.last_accessed else None,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "IntelligenceEntry":
        """Create from dictionary."""
        return cls(
            entry_id=data["entry_id"],
            entry_type=IntelligenceType[data["entry_type"]],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            source=data["source"],
            severity=data["severity"],
            category=data["category"],
            summary=data["summary"],
            content=data.get("content", {}),
            indicators=data.get("indicators", []),
            mitre_tactics=data.get("mitre_tactics", []),
            related_entries=data.get("related_entries", []),
            tags=set(data.get("tags", [])),
            tier=RetentionTier(data.get("tier", "hot")),
            created_at=datetime.fromisoformat(data.get("created_at", datetime.now().isoformat())),
            expires_at=datetime.fromisoformat(data["expires_at"]) if data.get("expires_at") else None,
            access_count=data.get("access_count", 0),
            last_accessed=datetime.fromisoformat(data["last_accessed"]) if data.get("last_accessed") else None,
        )

    def compact_for_warm(self) -> "IntelligenceEntry":
        """Create a compacted version for warm tier storage."""
        return IntelligenceEntry(
            entry_id=self.entry_id,
            entry_type=self.entry_type,
            timestamp=self.timestamp,
            source=self.source,
            severity=self.severity,
            category=self.category,
            summary=self.summary,
            # Reduce content to essential fields only
            content={
                k: v for k, v in self.content.items()
                if k in {"event_type", "source_ip", "dest_ip", "action", "result"}
            },
            indicators=self.indicators[:10],  # Keep top 10 indicators
            mitre_tactics=self.mitre_tactics,
            related_entries=self.related_entries[:5],
            tags=self.tags,
            tier=RetentionTier.WARM,
            created_at=self.created_at,
            access_count=self.access_count,
        )

    def compact_for_cold(self) -> "IntelligenceEntry":
        """Create a minimal version for cold tier storage."""
        return IntelligenceEntry(
            entry_id=self.entry_id,
            entry_type=self.entry_type,
            timestamp=self.timestamp,
            source=self.source,
            severity=self.severity,
            category=self.category,
            summary=self.summary,
            # Only keep count/summary info
            content={"original_size": len(json.dumps(self.content))},
            indicators=self.indicators[:5],
            mitre_tactics=self.mitre_tactics,
            related_entries=[],
            tags=self.tags,
            tier=RetentionTier.COLD,
            created_at=self.created_at,
            access_count=self.access_count,
        )


@dataclass
class RetentionPolicy:
    """Retention policy configuration."""

    hot_days: int = 7
    warm_days: int = 30
    cold_days: int = 365

    # Severity-based adjustments
    critical_retention_multiplier: float = 2.0  # Keep critical events longer
    high_access_retention_multiplier: float = 1.5  # Keep frequently accessed longer

    # Auto-migration settings
    auto_migrate: bool = True
    migration_batch_size: int = 1000

    def get_hot_expiry(self, entry: IntelligenceEntry) -> datetime:
        """Calculate hot tier expiry for an entry."""
        multiplier = 1.0
        if entry.severity >= 4:
            multiplier = self.critical_retention_multiplier
        elif entry.access_count > 10:
            multiplier = self.high_access_retention_multiplier
        return entry.created_at + timedelta(days=int(self.hot_days * multiplier))

    def get_warm_expiry(self, entry: IntelligenceEntry) -> datetime:
        """Calculate warm tier expiry for an entry."""
        multiplier = 1.0
        if entry.severity >= 4:
            multiplier = self.critical_retention_multiplier
        return entry.created_at + timedelta(days=int(self.warm_days * multiplier))

    def get_cold_expiry(self, entry: IntelligenceEntry) -> datetime:
        """Calculate cold tier expiry for an entry."""
        return entry.created_at + timedelta(days=self.cold_days)


class SecurityIntelligenceStore:
    """
    Security Intelligence Store with tiered retention.

    Provides:
    - Fast in-memory access for hot tier
    - SQLite persistence for warm/cold tiers
    - Automatic tier migration
    - Query optimization with indexes
    - Thread-safe operations
    """

    def __init__(
        self,
        storage_path: Optional[Path] = None,
        retention_policy: Optional[RetentionPolicy] = None,
        on_tier_migrate: Optional[Callable[[IntelligenceEntry, RetentionTier], None]] = None,
    ):
        """
        Initialize the intelligence store.

        Args:
            storage_path: Path to SQLite database (None for in-memory)
            retention_policy: Retention policy configuration
            on_tier_migrate: Callback when entry migrates to new tier
        """
        self.storage_path = storage_path
        self.retention_policy = retention_policy or RetentionPolicy()
        self.on_tier_migrate = on_tier_migrate

        # Hot tier: in-memory for fast access
        self._hot_store: Dict[str, IntelligenceEntry] = {}
        self._hot_index_by_type: Dict[IntelligenceType, Set[str]] = {}
        self._hot_index_by_source: Dict[str, Set[str]] = {}
        self._hot_index_by_category: Dict[str, Set[str]] = {}
        self._hot_index_by_severity: Dict[int, Set[str]] = {}

        # Thread safety
        self._lock = threading.RLock()

        # SQLite connection for warm/cold tiers
        self._db: Optional[sqlite3.Connection] = None
        self._init_database()

        # Background maintenance
        self._running = False
        self._maintenance_thread: Optional[threading.Thread] = None

        # Statistics
        self._stats = {
            "entries_added": 0,
            "entries_retrieved": 0,
            "entries_migrated": 0,
            "entries_expired": 0,
            "queries_executed": 0,
        }

        # Entry counter for ID generation
        self._entry_counter = 0

    def _init_database(self) -> None:
        """Initialize SQLite database for warm/cold storage."""
        if self.storage_path:
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)
            db_path = str(self.storage_path)
        else:
            db_path = ":memory:"

        self._db = sqlite3.connect(db_path, check_same_thread=False)
        self._db.row_factory = sqlite3.Row

        # Create tables
        self._db.executescript("""
            CREATE TABLE IF NOT EXISTS intelligence (
                entry_id TEXT PRIMARY KEY,
                entry_type TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                source TEXT NOT NULL,
                severity INTEGER NOT NULL,
                category TEXT NOT NULL,
                summary TEXT NOT NULL,
                content TEXT NOT NULL,
                indicators TEXT NOT NULL,
                mitre_tactics TEXT NOT NULL,
                related_entries TEXT NOT NULL,
                tags TEXT NOT NULL,
                tier TEXT NOT NULL,
                created_at TEXT NOT NULL,
                expires_at TEXT,
                access_count INTEGER DEFAULT 0,
                last_accessed TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_type ON intelligence(entry_type);
            CREATE INDEX IF NOT EXISTS idx_source ON intelligence(source);
            CREATE INDEX IF NOT EXISTS idx_category ON intelligence(category);
            CREATE INDEX IF NOT EXISTS idx_severity ON intelligence(severity);
            CREATE INDEX IF NOT EXISTS idx_timestamp ON intelligence(timestamp);
            CREATE INDEX IF NOT EXISTS idx_tier ON intelligence(tier);
            CREATE INDEX IF NOT EXISTS idx_expires ON intelligence(expires_at);

            -- Aggregation table for analytics
            CREATE TABLE IF NOT EXISTS hourly_aggregates (
                hour TEXT NOT NULL,
                source TEXT NOT NULL,
                category TEXT NOT NULL,
                entry_type TEXT NOT NULL,
                count INTEGER DEFAULT 0,
                severity_sum INTEGER DEFAULT 0,
                severity_max INTEGER DEFAULT 0,
                PRIMARY KEY (hour, source, category, entry_type)
            );

            -- Indicator tracking for threat intelligence
            CREATE TABLE IF NOT EXISTS indicators (
                indicator TEXT PRIMARY KEY,
                indicator_type TEXT NOT NULL,
                first_seen TEXT NOT NULL,
                last_seen TEXT NOT NULL,
                occurrence_count INTEGER DEFAULT 1,
                associated_entries TEXT NOT NULL,
                risk_score INTEGER DEFAULT 0
            );
        """)
        self._db.commit()

    def start(self) -> None:
        """Start background maintenance thread."""
        if self._running:
            return

        self._running = True
        self._maintenance_thread = threading.Thread(
            target=self._maintenance_loop,
            daemon=True,
            name="IntelligenceStoreMaintenance",
        )
        self._maintenance_thread.start()
        logger.info("Intelligence store maintenance started")

    def stop(self) -> None:
        """Stop background maintenance thread."""
        self._running = False
        if self._maintenance_thread:
            self._maintenance_thread.join(timeout=5.0)
            self._maintenance_thread = None
        logger.info("Intelligence store maintenance stopped")

    def add(self, entry: IntelligenceEntry) -> str:
        """
        Add an intelligence entry.

        Args:
            entry: Entry to add

        Returns:
            Entry ID
        """
        with self._lock:
            # Generate ID if not provided
            if not entry.entry_id:
                self._entry_counter += 1
                entry.entry_id = f"intel_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{self._entry_counter:06d}"

            # Set expiry based on tier
            if entry.tier == RetentionTier.HOT:
                entry.expires_at = self.retention_policy.get_hot_expiry(entry)
            elif entry.tier == RetentionTier.WARM:
                entry.expires_at = self.retention_policy.get_warm_expiry(entry)
            elif entry.tier == RetentionTier.COLD:
                entry.expires_at = self.retention_policy.get_cold_expiry(entry)

            # Add to hot tier (in-memory)
            if entry.tier == RetentionTier.HOT:
                self._add_to_hot(entry)
            else:
                # Add directly to database for warm/cold
                self._add_to_db(entry)

            # Update aggregates
            self._update_aggregates(entry)

            # Update indicator tracking
            self._update_indicators(entry)

            self._stats["entries_added"] += 1

            return entry.entry_id

    def _add_to_hot(self, entry: IntelligenceEntry) -> None:
        """Add entry to hot tier."""
        self._hot_store[entry.entry_id] = entry

        # Update indexes
        if entry.entry_type not in self._hot_index_by_type:
            self._hot_index_by_type[entry.entry_type] = set()
        self._hot_index_by_type[entry.entry_type].add(entry.entry_id)

        if entry.source not in self._hot_index_by_source:
            self._hot_index_by_source[entry.source] = set()
        self._hot_index_by_source[entry.source].add(entry.entry_id)

        if entry.category not in self._hot_index_by_category:
            self._hot_index_by_category[entry.category] = set()
        self._hot_index_by_category[entry.category].add(entry.entry_id)

        if entry.severity not in self._hot_index_by_severity:
            self._hot_index_by_severity[entry.severity] = set()
        self._hot_index_by_severity[entry.severity].add(entry.entry_id)

    def _remove_from_hot(self, entry_id: str) -> Optional[IntelligenceEntry]:
        """Remove entry from hot tier."""
        entry = self._hot_store.pop(entry_id, None)
        if entry:
            self._hot_index_by_type.get(entry.entry_type, set()).discard(entry_id)
            self._hot_index_by_source.get(entry.source, set()).discard(entry_id)
            self._hot_index_by_category.get(entry.category, set()).discard(entry_id)
            self._hot_index_by_severity.get(entry.severity, set()).discard(entry_id)
        return entry

    def _add_to_db(self, entry: IntelligenceEntry) -> None:
        """Add entry to SQLite database."""
        data = entry.to_dict()
        self._db.execute(
            """
            INSERT OR REPLACE INTO intelligence (
                entry_id, entry_type, timestamp, source, severity, category,
                summary, content, indicators, mitre_tactics, related_entries,
                tags, tier, created_at, expires_at, access_count, last_accessed
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                data["entry_id"],
                data["entry_type"],
                data["timestamp"],
                data["source"],
                data["severity"],
                data["category"],
                data["summary"],
                json.dumps(data["content"]),
                json.dumps(data["indicators"]),
                json.dumps(data["mitre_tactics"]),
                json.dumps(data["related_entries"]),
                json.dumps(data["tags"]),
                data["tier"],
                data["created_at"],
                data["expires_at"],
                data["access_count"],
                data["last_accessed"],
            ),
        )
        self._db.commit()

    def _update_aggregates(self, entry: IntelligenceEntry) -> None:
        """Update hourly aggregates."""
        hour = entry.timestamp.strftime("%Y-%m-%d %H:00:00")
        self._db.execute(
            """
            INSERT INTO hourly_aggregates (hour, source, category, entry_type, count, severity_sum, severity_max)
            VALUES (?, ?, ?, ?, 1, ?, ?)
            ON CONFLICT(hour, source, category, entry_type) DO UPDATE SET
                count = count + 1,
                severity_sum = severity_sum + excluded.severity_sum,
                severity_max = MAX(severity_max, excluded.severity_max)
            """,
            (hour, entry.source, entry.category, entry.entry_type.name, entry.severity, entry.severity),
        )
        self._db.commit()

    def _update_indicators(self, entry: IntelligenceEntry) -> None:
        """Update indicator tracking."""
        now = datetime.now().isoformat()
        for indicator in entry.indicators:
            # Parse indicator type (e.g., "ip:1.2.3.4" -> type="ip", value="1.2.3.4")
            if ":" in indicator:
                ind_type, _ind_value = indicator.split(":", 1)
            else:
                ind_type, _ind_value = "unknown", indicator

            self._db.execute(
                """
                INSERT INTO indicators (indicator, indicator_type, first_seen, last_seen, occurrence_count, associated_entries, risk_score)
                VALUES (?, ?, ?, ?, 1, ?, ?)
                ON CONFLICT(indicator) DO UPDATE SET
                    last_seen = excluded.last_seen,
                    occurrence_count = occurrence_count + 1,
                    associated_entries = associated_entries || ',' || ?,
                    risk_score = MIN(10, risk_score + 1)
                """,
                (indicator, ind_type, now, now, entry.entry_id, entry.severity, entry.entry_id),
            )
        self._db.commit()

    def get(self, entry_id: str) -> Optional[IntelligenceEntry]:
        """
        Get an entry by ID.

        Args:
            entry_id: Entry ID to retrieve

        Returns:
            Entry or None if not found
        """
        with self._lock:
            self._stats["entries_retrieved"] += 1

            # Check hot tier first
            if entry_id in self._hot_store:
                entry = self._hot_store[entry_id]
                entry.access_count += 1
                # Only update timestamp if last access was >60s ago (reduces overhead)
                now = datetime.now()
                if not entry.last_accessed or (now - entry.last_accessed).total_seconds() > 60:
                    entry.last_accessed = now
                return entry

            # Check database
            row = self._db.execute(
                "SELECT * FROM intelligence WHERE entry_id = ?",
                (entry_id,),
            ).fetchone()

            if row:
                entry = self._row_to_entry(row)
                # Only update if last_accessed was >60s ago (reduces write overhead)
                now = datetime.now()
                should_update_time = (
                    not entry.last_accessed
                    or (now - entry.last_accessed).total_seconds() > 60
                )
                if should_update_time:
                    self._db.execute(
                        "UPDATE intelligence SET access_count = access_count + 1, last_accessed = ? WHERE entry_id = ?",
                        (now.isoformat(), entry_id),
                    )
                else:
                    self._db.execute(
                        "UPDATE intelligence SET access_count = access_count + 1 WHERE entry_id = ?",
                        (entry_id,),
                    )
                self._db.commit()
                return entry

            return None

    def query(
        self,
        entry_types: Optional[List[IntelligenceType]] = None,
        sources: Optional[List[str]] = None,
        categories: Optional[List[str]] = None,
        severity_min: Optional[int] = None,
        severity_max: Optional[int] = None,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
        tiers: Optional[List[RetentionTier]] = None,
        indicators: Optional[List[str]] = None,
        mitre_tactics: Optional[List[str]] = None,
        tags: Optional[Set[str]] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[IntelligenceEntry]:
        """
        Query intelligence entries.

        Args:
            entry_types: Filter by entry types
            sources: Filter by sources
            categories: Filter by categories
            severity_min: Minimum severity (inclusive)
            severity_max: Maximum severity (inclusive)
            since: Entries after this time
            until: Entries before this time
            tiers: Filter by retention tiers
            indicators: Filter by indicators (any match)
            mitre_tactics: Filter by MITRE tactics (any match)
            tags: Filter by tags (all must match)
            limit: Maximum results
            offset: Result offset

        Returns:
            List of matching entries
        """
        with self._lock:
            self._stats["queries_executed"] += 1
            results = []

            # Query hot tier (in-memory)
            hot_results = self._query_hot(
                entry_types, sources, categories, severity_min, severity_max,
                since, until, indicators, mitre_tactics, tags,
            )

            # Query database for warm/cold if needed
            if not tiers or any(t != RetentionTier.HOT for t in tiers):
                db_results = self._query_db(
                    entry_types, sources, categories, severity_min, severity_max,
                    since, until, tiers, indicators, mitre_tactics, tags,
                    limit, offset,
                )
            else:
                db_results = []

            # Combine and sort
            results = hot_results + db_results
            results.sort(key=lambda e: e.timestamp, reverse=True)

            return results[offset:offset + limit]

    def _query_hot(
        self,
        entry_types: Optional[List[IntelligenceType]],
        sources: Optional[List[str]],
        categories: Optional[List[str]],
        severity_min: Optional[int],
        severity_max: Optional[int],
        since: Optional[datetime],
        until: Optional[datetime],
        indicators: Optional[List[str]],
        mitre_tactics: Optional[List[str]],
        tags: Optional[Set[str]],
    ) -> List[IntelligenceEntry]:
        """Query hot tier."""
        # Start with all IDs or filtered set
        candidate_ids: Optional[Set[str]] = None

        if entry_types:
            type_ids: Set[str] = set()
            for et in entry_types:
                type_ids.update(self._hot_index_by_type.get(et, set()))
            candidate_ids = type_ids if candidate_ids is None else candidate_ids & type_ids

        if sources:
            source_ids: Set[str] = set()
            for src in sources:
                source_ids.update(self._hot_index_by_source.get(src, set()))
            candidate_ids = source_ids if candidate_ids is None else candidate_ids & source_ids

        if categories:
            cat_ids: Set[str] = set()
            for cat in categories:
                cat_ids.update(self._hot_index_by_category.get(cat, set()))
            candidate_ids = cat_ids if candidate_ids is None else candidate_ids & cat_ids

        if severity_min is not None or severity_max is not None:
            sev_ids: Set[str] = set()
            for sev in range(severity_min or 1, (severity_max or 5) + 1):
                sev_ids.update(self._hot_index_by_severity.get(sev, set()))
            candidate_ids = sev_ids if candidate_ids is None else candidate_ids & sev_ids

        # If no filters, use all
        if candidate_ids is None:
            candidate_ids = set(self._hot_store.keys())

        # Filter by remaining criteria
        results = []
        for entry_id in candidate_ids:
            entry = self._hot_store.get(entry_id)
            if not entry:
                continue

            # Time filters
            if since and entry.timestamp < since:
                continue
            if until and entry.timestamp > until:
                continue

            # Indicator filter
            if indicators:
                if not any(ind in entry.indicators for ind in indicators):
                    continue

            # MITRE filter
            if mitre_tactics:
                if not any(tac in entry.mitre_tactics for tac in mitre_tactics):
                    continue

            # Tags filter (all must match)
            if tags:
                if not tags.issubset(entry.tags):
                    continue

            results.append(entry)

        return results

    def _query_db(
        self,
        entry_types: Optional[List[IntelligenceType]],
        sources: Optional[List[str]],
        categories: Optional[List[str]],
        severity_min: Optional[int],
        severity_max: Optional[int],
        since: Optional[datetime],
        until: Optional[datetime],
        tiers: Optional[List[RetentionTier]],
        indicators: Optional[List[str]],
        mitre_tactics: Optional[List[str]],
        tags: Optional[Set[str]],
        limit: int,
        offset: int,
    ) -> List[IntelligenceEntry]:
        """Query database for warm/cold entries."""
        conditions = []
        params = []

        if entry_types:
            placeholders = ",".join("?" * len(entry_types))
            conditions.append(f"entry_type IN ({placeholders})")
            params.extend([et.name for et in entry_types])

        if sources:
            placeholders = ",".join("?" * len(sources))
            conditions.append(f"source IN ({placeholders})")
            params.extend(sources)

        if categories:
            placeholders = ",".join("?" * len(categories))
            conditions.append(f"category IN ({placeholders})")
            params.extend(categories)

        if severity_min is not None:
            conditions.append("severity >= ?")
            params.append(severity_min)

        if severity_max is not None:
            conditions.append("severity <= ?")
            params.append(severity_max)

        if since:
            conditions.append("timestamp >= ?")
            params.append(since.isoformat())

        if until:
            conditions.append("timestamp <= ?")
            params.append(until.isoformat())

        if tiers:
            placeholders = ",".join("?" * len(tiers))
            conditions.append(f"tier IN ({placeholders})")
            params.extend([t.value for t in tiers])
        else:
            # Exclude hot tier (already queried)
            conditions.append("tier != 'hot'")

        where_clause = " AND ".join(conditions) if conditions else "1=1"

        query = f"""
            SELECT * FROM intelligence
            WHERE {where_clause}
            ORDER BY timestamp DESC
            LIMIT ? OFFSET ?
        """
        params.extend([limit, offset])

        rows = self._db.execute(query, params).fetchall()
        results = [self._row_to_entry(row) for row in rows]

        # Post-filter for JSON fields (indicators, tactics, tags)
        if indicators or mitre_tactics or tags:
            filtered = []
            for entry in results:
                if indicators and not any(ind in entry.indicators for ind in indicators):
                    continue
                if mitre_tactics and not any(tac in entry.mitre_tactics for tac in mitre_tactics):
                    continue
                if tags and not tags.issubset(entry.tags):
                    continue
                filtered.append(entry)
            results = filtered

        return results

    def _row_to_entry(self, row: sqlite3.Row) -> IntelligenceEntry:
        """Convert database row to entry."""
        return IntelligenceEntry(
            entry_id=row["entry_id"],
            entry_type=IntelligenceType[row["entry_type"]],
            timestamp=datetime.fromisoformat(row["timestamp"]),
            source=row["source"],
            severity=row["severity"],
            category=row["category"],
            summary=row["summary"],
            content=json.loads(row["content"]),
            indicators=json.loads(row["indicators"]),
            mitre_tactics=json.loads(row["mitre_tactics"]),
            related_entries=json.loads(row["related_entries"]),
            tags=set(json.loads(row["tags"])),
            tier=RetentionTier(row["tier"]),
            created_at=datetime.fromisoformat(row["created_at"]),
            expires_at=datetime.fromisoformat(row["expires_at"]) if row["expires_at"] else None,
            access_count=row["access_count"],
            last_accessed=datetime.fromisoformat(row["last_accessed"]) if row["last_accessed"] else None,
        )

    def get_aggregates(
        self,
        since: datetime,
        until: Optional[datetime] = None,
        group_by: str = "hour",  # hour, source, category, entry_type
    ) -> List[Dict[str, Any]]:
        """
        Get aggregated statistics.

        Args:
            since: Start time
            until: End time
            group_by: Grouping field

        Returns:
            List of aggregate records
        """
        if until is None:
            until = datetime.now()

        query = f"""
            SELECT
                {group_by},
                SUM(count) as total_count,
                SUM(severity_sum) as severity_sum,
                MAX(severity_max) as severity_max,
                AVG(severity_sum * 1.0 / count) as avg_severity
            FROM hourly_aggregates
            WHERE hour >= ? AND hour <= ?
            GROUP BY {group_by}
            ORDER BY total_count DESC
        """

        rows = self._db.execute(
            query,
            (since.strftime("%Y-%m-%d %H:00:00"), until.strftime("%Y-%m-%d %H:00:00")),
        ).fetchall()

        return [dict(row) for row in rows]

    def get_top_indicators(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get top threat indicators by occurrence and risk."""
        rows = self._db.execute(
            """
            SELECT indicator, indicator_type, first_seen, last_seen,
                   occurrence_count, risk_score
            FROM indicators
            ORDER BY risk_score DESC, occurrence_count DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()

        return [dict(row) for row in rows]

    def count(
        self,
        tier: Optional[RetentionTier] = None,
        since: Optional[datetime] = None,
    ) -> int:
        """Count entries."""
        with self._lock:
            hot_count = len(self._hot_store)

            conditions = []
            params = []

            if tier and tier != RetentionTier.HOT:
                conditions.append("tier = ?")
                params.append(tier.value)
            elif tier == RetentionTier.HOT:
                return hot_count

            if since:
                conditions.append("timestamp >= ?")
                params.append(since.isoformat())

            where_clause = " AND ".join(conditions) if conditions else "1=1"

            row = self._db.execute(
                f"SELECT COUNT(*) as cnt FROM intelligence WHERE {where_clause}",
                params,
            ).fetchone()

            db_count = row["cnt"] if row else 0

            if tier:
                return db_count
            return hot_count + db_count

    def get_stats(self) -> Dict[str, Any]:
        """Get store statistics."""
        with self._lock:
            hot_count = len(self._hot_store)

            db_stats = self._db.execute(
                """
                SELECT tier, COUNT(*) as cnt
                FROM intelligence
                GROUP BY tier
                """
            ).fetchall()

            tier_counts = {row["tier"]: row["cnt"] for row in db_stats}

            return {
                **self._stats,
                "hot_entries": hot_count,
                "warm_entries": tier_counts.get("warm", 0),
                "cold_entries": tier_counts.get("cold", 0),
                "total_entries": hot_count + sum(tier_counts.values()),
            }

    def _maintenance_loop(self) -> None:
        """Background maintenance loop."""
        while self._running:
            try:
                self._perform_maintenance()
            except Exception as e:
                logger.error(f"Maintenance error: {e}")

            # Sleep for 5 minutes between maintenance runs
            for _ in range(300):
                if not self._running:
                    break
                time.sleep(1)

    def _perform_maintenance(self) -> None:
        """Perform maintenance tasks."""
        now = datetime.now()

        with self._lock:
            # Migrate hot -> warm
            hot_expired = [
                entry_id
                for entry_id, entry in self._hot_store.items()
                if entry.expires_at and entry.expires_at <= now
            ]

            for entry_id in hot_expired[:self.retention_policy.migration_batch_size]:
                entry = self._remove_from_hot(entry_id)
                if entry:
                    warm_entry = entry.compact_for_warm()
                    warm_entry.expires_at = self.retention_policy.get_warm_expiry(entry)
                    self._add_to_db(warm_entry)
                    self._stats["entries_migrated"] += 1

                    if self.on_tier_migrate:
                        self.on_tier_migrate(entry, RetentionTier.WARM)

            # Migrate warm -> cold
            warm_expired = self._db.execute(
                """
                SELECT entry_id FROM intelligence
                WHERE tier = 'warm' AND expires_at <= ?
                LIMIT ?
                """,
                (now.isoformat(), self.retention_policy.migration_batch_size),
            ).fetchall()

            for row in warm_expired:
                entry = self.get(row["entry_id"])
                if entry:
                    cold_entry = entry.compact_for_cold()
                    cold_entry.expires_at = self.retention_policy.get_cold_expiry(entry)
                    self._add_to_db(cold_entry)
                    self._stats["entries_migrated"] += 1

                    if self.on_tier_migrate:
                        self.on_tier_migrate(entry, RetentionTier.COLD)

            # Delete expired cold entries
            deleted = self._db.execute(
                """
                DELETE FROM intelligence
                WHERE tier = 'cold' AND expires_at <= ?
                """,
                (now.isoformat(),),
            ).rowcount
            self._db.commit()

            if deleted:
                self._stats["entries_expired"] += deleted
                logger.info(f"Expired {deleted} cold tier entries")

    def close(self) -> None:
        """Close the store."""
        self.stop()
        if self._db:
            self._db.close()
            self._db = None


def create_intelligence_store(
    storage_path: Optional[str] = None,
    retention_policy: Optional[RetentionPolicy] = None,
    auto_start: bool = True,
) -> SecurityIntelligenceStore:
    """
    Factory function to create an intelligence store.

    Args:
        storage_path: Path to storage file (None for in-memory)
        retention_policy: Custom retention policy
        auto_start: Start maintenance thread automatically

    Returns:
        Configured SecurityIntelligenceStore
    """
    store = SecurityIntelligenceStore(
        storage_path=Path(storage_path) if storage_path else None,
        retention_policy=retention_policy,
    )

    if auto_start:
        store.start()

    return store
