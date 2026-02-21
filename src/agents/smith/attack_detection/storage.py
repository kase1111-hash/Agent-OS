"""
Persistent Storage for Attack Detection System

Provides durable storage for:
- Detected attacks and their status
- Fix recommendations and approval workflow
- Patches and test results
- Vulnerability analysis reports
- SIEM event history

Supports both SQLite (persistent) and in-memory backends.
"""

import hashlib
import json
import logging
import sqlite3
import threading
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Type, TypeVar, Union

logger = logging.getLogger(__name__)

# Type variable for generic storage operations
T = TypeVar("T")


# =============================================================================
# Storage Enums
# =============================================================================


class StorageBackend(Enum):
    """Supported storage backends."""

    SQLITE = "sqlite"
    MEMORY = "memory"


# =============================================================================
# Storage Models
# =============================================================================


@dataclass
class StoredAttack:
    """
    Persisted attack record.

    Stores all information about a detected attack including
    its current status and any linked recommendations.
    """

    attack_id: str
    attack_type: str
    severity: int
    severity_name: str
    confidence: float
    description: str
    source: str
    target: Optional[str]
    detected_at: datetime
    indicators: List[str] = field(default_factory=list)
    mitre_tactics: List[str] = field(default_factory=list)
    raw_data: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Status tracking
    status: str = "detected"  # detected, analyzing, mitigated, false_positive, ignored
    mitigated_at: Optional[datetime] = None
    mitigation_notes: str = ""

    # Linked entities
    recommendation_id: Optional[str] = None
    vulnerability_ids: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = asdict(self)
        result["detected_at"] = self.detected_at.isoformat()
        if self.mitigated_at:
            result["mitigated_at"] = self.mitigated_at.isoformat()
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StoredAttack":
        """Create from dictionary."""
        data = data.copy()
        if isinstance(data.get("detected_at"), str):
            data["detected_at"] = datetime.fromisoformat(data["detected_at"])
        if data.get("mitigated_at") and isinstance(data["mitigated_at"], str):
            data["mitigated_at"] = datetime.fromisoformat(data["mitigated_at"])
        # Handle JSON fields
        for field_name in ["indicators", "mitre_tactics", "vulnerability_ids"]:
            if isinstance(data.get(field_name), str):
                data[field_name] = json.loads(data[field_name])
        for field_name in ["raw_data", "metadata"]:
            if isinstance(data.get(field_name), str):
                data[field_name] = json.loads(data[field_name])
        return cls(**data)


@dataclass
class StoredRecommendation:
    """
    Persisted fix recommendation.

    Stores recommendation details including approval workflow state.
    """

    recommendation_id: str
    attack_id: str
    title: str
    description: str
    priority: int
    priority_name: str
    created_at: datetime
    created_by: str = "system"

    # Status tracking
    status: str = "pending"  # pending, approved, rejected, applied, superseded
    reviewed_at: Optional[datetime] = None
    reviewed_by: Optional[str] = None
    review_comments: str = ""

    # Application tracking
    applied_at: Optional[datetime] = None
    applied_by: Optional[str] = None
    application_result: str = ""

    # Linked entities
    patch_ids: List[str] = field(default_factory=list)
    vulnerability_ids: List[str] = field(default_factory=list)

    # Additional data
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = asdict(self)
        result["created_at"] = self.created_at.isoformat()
        if self.reviewed_at:
            result["reviewed_at"] = self.reviewed_at.isoformat()
        if self.applied_at:
            result["applied_at"] = self.applied_at.isoformat()
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StoredRecommendation":
        """Create from dictionary."""
        data = data.copy()
        if isinstance(data.get("created_at"), str):
            data["created_at"] = datetime.fromisoformat(data["created_at"])
        if data.get("reviewed_at") and isinstance(data["reviewed_at"], str):
            data["reviewed_at"] = datetime.fromisoformat(data["reviewed_at"])
        if data.get("applied_at") and isinstance(data["applied_at"], str):
            data["applied_at"] = datetime.fromisoformat(data["applied_at"])
        for field_name in ["patch_ids", "vulnerability_ids"]:
            if isinstance(data.get(field_name), str):
                data[field_name] = json.loads(data[field_name])
        if isinstance(data.get("metadata"), str):
            data["metadata"] = json.loads(data["metadata"])
        return cls(**data)


@dataclass
class StoredPatch:
    """
    Persisted patch record.

    Stores patch content, test results, and application status.
    """

    patch_id: str
    recommendation_id: str
    file_path: str
    patch_type: str  # add, modify, delete
    description: str
    created_at: datetime

    # Patch content
    original_content: str = ""
    patched_content: str = ""
    diff: str = ""

    # Status tracking
    status: str = "pending"  # pending, tested, applied, failed, reverted
    tested_at: Optional[datetime] = None
    test_passed: bool = False
    test_output: str = ""

    applied_at: Optional[datetime] = None
    applied_by: Optional[str] = None
    reverted_at: Optional[datetime] = None

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = asdict(self)
        result["created_at"] = self.created_at.isoformat()
        if self.tested_at:
            result["tested_at"] = self.tested_at.isoformat()
        if self.applied_at:
            result["applied_at"] = self.applied_at.isoformat()
        if self.reverted_at:
            result["reverted_at"] = self.reverted_at.isoformat()
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StoredPatch":
        """Create from dictionary."""
        data = data.copy()
        if isinstance(data.get("created_at"), str):
            data["created_at"] = datetime.fromisoformat(data["created_at"])
        if data.get("tested_at") and isinstance(data["tested_at"], str):
            data["tested_at"] = datetime.fromisoformat(data["tested_at"])
        if data.get("applied_at") and isinstance(data["applied_at"], str):
            data["applied_at"] = datetime.fromisoformat(data["applied_at"])
        if data.get("reverted_at") and isinstance(data["reverted_at"], str):
            data["reverted_at"] = datetime.fromisoformat(data["reverted_at"])
        if isinstance(data.get("metadata"), str):
            data["metadata"] = json.loads(data["metadata"])
        return cls(**data)


@dataclass
class StoredVulnerability:
    """
    Persisted vulnerability finding.

    Stores code location and analysis details.
    """

    vulnerability_id: str
    attack_id: str
    file_path: str
    line_start: int
    line_end: int
    vulnerability_type: str
    description: str
    severity: int
    created_at: datetime

    # Code context
    code_snippet: str = ""
    suggested_fix: str = ""

    # Status
    status: str = "open"  # open, fixed, false_positive, wont_fix
    fixed_at: Optional[datetime] = None
    fixed_by: Optional[str] = None

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = asdict(self)
        result["created_at"] = self.created_at.isoformat()
        if self.fixed_at:
            result["fixed_at"] = self.fixed_at.isoformat()
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StoredVulnerability":
        """Create from dictionary."""
        data = data.copy()
        if isinstance(data.get("created_at"), str):
            data["created_at"] = datetime.fromisoformat(data["created_at"])
        if data.get("fixed_at") and isinstance(data["fixed_at"], str):
            data["fixed_at"] = datetime.fromisoformat(data["fixed_at"])
        if isinstance(data.get("metadata"), str):
            data["metadata"] = json.loads(data["metadata"])
        return cls(**data)


@dataclass
class StoredSIEMEvent:
    """
    Persisted SIEM event for historical analysis.
    """

    event_id: str
    provider: str
    timestamp: datetime
    source: str
    event_type: str
    severity: int
    category: str
    description: str
    raw_data: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    indicators: List[str] = field(default_factory=list)

    # Processing status
    processed: bool = False
    attack_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = asdict(self)
        result["timestamp"] = self.timestamp.isoformat()
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StoredSIEMEvent":
        """Create from dictionary."""
        data = data.copy()
        if isinstance(data.get("timestamp"), str):
            data["timestamp"] = datetime.fromisoformat(data["timestamp"])
        for field_name in ["raw_data", "metadata"]:
            if isinstance(data.get(field_name), str):
                data[field_name] = json.loads(data[field_name])
        if isinstance(data.get("indicators"), str):
            data["indicators"] = json.loads(data["indicators"])
        return cls(**data)


# =============================================================================
# Abstract Storage Interface
# =============================================================================


class AttackStorage(ABC):
    """
    Abstract interface for attack detection storage.

    Implementations must provide persistent storage for all
    attack detection entities.
    """

    # -------------------------------------------------------------------------
    # Attack Operations
    # -------------------------------------------------------------------------

    @abstractmethod
    def save_attack(self, attack: StoredAttack) -> bool:
        """Save or update an attack record."""
        pass

    @abstractmethod
    def get_attack(self, attack_id: str) -> Optional[StoredAttack]:
        """Get an attack by ID."""
        pass

    @abstractmethod
    def list_attacks(
        self,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
        status: Optional[str] = None,
        severity_min: Optional[int] = None,
        attack_type: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[StoredAttack]:
        """List attacks with optional filters."""
        pass

    @abstractmethod
    def update_attack_status(
        self,
        attack_id: str,
        status: str,
        notes: str = "",
    ) -> bool:
        """Update attack status."""
        pass

    @abstractmethod
    def count_attacks(
        self,
        since: Optional[datetime] = None,
        status: Optional[str] = None,
    ) -> int:
        """Count attacks matching criteria."""
        pass

    # -------------------------------------------------------------------------
    # Recommendation Operations
    # -------------------------------------------------------------------------

    @abstractmethod
    def save_recommendation(self, rec: StoredRecommendation) -> bool:
        """Save or update a recommendation."""
        pass

    @abstractmethod
    def get_recommendation(self, rec_id: str) -> Optional[StoredRecommendation]:
        """Get a recommendation by ID."""
        pass

    @abstractmethod
    def list_recommendations(
        self,
        status: Optional[str] = None,
        attack_id: Optional[str] = None,
        priority_min: Optional[int] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[StoredRecommendation]:
        """List recommendations with optional filters."""
        pass

    @abstractmethod
    def update_recommendation_status(
        self,
        rec_id: str,
        status: str,
        reviewer: Optional[str] = None,
        comments: str = "",
    ) -> bool:
        """Update recommendation status."""
        pass

    # -------------------------------------------------------------------------
    # Patch Operations
    # -------------------------------------------------------------------------

    @abstractmethod
    def save_patch(self, patch: StoredPatch) -> bool:
        """Save or update a patch."""
        pass

    @abstractmethod
    def get_patch(self, patch_id: str) -> Optional[StoredPatch]:
        """Get a patch by ID."""
        pass

    @abstractmethod
    def list_patches(
        self,
        recommendation_id: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 100,
    ) -> List[StoredPatch]:
        """List patches with optional filters."""
        pass

    @abstractmethod
    def update_patch_status(
        self,
        patch_id: str,
        status: str,
        test_passed: Optional[bool] = None,
        test_output: str = "",
    ) -> bool:
        """Update patch status."""
        pass

    # -------------------------------------------------------------------------
    # Vulnerability Operations
    # -------------------------------------------------------------------------

    @abstractmethod
    def save_vulnerability(self, vuln: StoredVulnerability) -> bool:
        """Save or update a vulnerability."""
        pass

    @abstractmethod
    def get_vulnerability(self, vuln_id: str) -> Optional[StoredVulnerability]:
        """Get a vulnerability by ID."""
        pass

    @abstractmethod
    def list_vulnerabilities(
        self,
        attack_id: Optional[str] = None,
        file_path: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 100,
    ) -> List[StoredVulnerability]:
        """List vulnerabilities with optional filters."""
        pass

    # -------------------------------------------------------------------------
    # SIEM Event Operations
    # -------------------------------------------------------------------------

    @abstractmethod
    def save_siem_event(self, event: StoredSIEMEvent) -> bool:
        """Save a SIEM event."""
        pass

    @abstractmethod
    def list_siem_events(
        self,
        since: Optional[datetime] = None,
        provider: Optional[str] = None,
        processed: Optional[bool] = None,
        limit: int = 100,
    ) -> List[StoredSIEMEvent]:
        """List SIEM events with optional filters."""
        pass

    @abstractmethod
    def mark_siem_events_processed(
        self,
        event_ids: List[str],
        attack_id: Optional[str] = None,
    ) -> int:
        """Mark SIEM events as processed."""
        pass

    # -------------------------------------------------------------------------
    # Statistics
    # -------------------------------------------------------------------------

    @abstractmethod
    def get_statistics(self) -> Dict[str, Any]:
        """Get storage statistics."""
        pass

    # -------------------------------------------------------------------------
    # Maintenance
    # -------------------------------------------------------------------------

    @abstractmethod
    def cleanup_old_records(
        self,
        older_than: datetime,
        keep_unresolved: bool = True,
    ) -> Dict[str, int]:
        """Remove old records."""
        pass

    @abstractmethod
    def close(self) -> None:
        """Close storage connection."""
        pass


# =============================================================================
# SQLite Storage Implementation
# =============================================================================


class SQLiteStorage(AttackStorage):
    """
    SQLite-based persistent storage.

    Stores all attack detection data in a SQLite database
    for durability across restarts.
    """

    SCHEMA_VERSION = 1

    def __init__(
        self,
        db_path: Union[str, Path] = "attack_detection.db",
        auto_migrate: bool = True,
    ):
        """
        Initialize SQLite storage.

        Args:
            db_path: Path to SQLite database file
            auto_migrate: Automatically apply schema migrations
        """
        self.db_path = Path(db_path)
        self._lock = threading.Lock()
        self._local = threading.local()

        # Create database directory if needed
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize schema
        self._init_schema()

        if auto_migrate:
            self._migrate_schema()

        logger.info(f"SQLite storage initialized: {self.db_path}")

    @contextmanager
    def _get_connection(self) -> Generator[sqlite3.Connection, None, None]:
        """Get a thread-local database connection."""
        if not hasattr(self._local, "connection"):
            conn = sqlite3.connect(
                str(self.db_path),
                detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES,
            )
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA foreign_keys = ON")
            conn.execute("PRAGMA journal_mode = WAL")
            self._local.connection = conn

        yield self._local.connection

    def _init_schema(self) -> None:
        """Initialize database schema."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Schema version tracking
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS schema_version (
                    version INTEGER PRIMARY KEY,
                    applied_at TEXT NOT NULL
                )
            """)

            # Attacks table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS attacks (
                    attack_id TEXT PRIMARY KEY,
                    attack_type TEXT NOT NULL,
                    severity INTEGER NOT NULL,
                    severity_name TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    description TEXT NOT NULL,
                    source TEXT NOT NULL,
                    target TEXT,
                    detected_at TEXT NOT NULL,
                    indicators TEXT DEFAULT '[]',
                    mitre_tactics TEXT DEFAULT '[]',
                    raw_data TEXT DEFAULT '{}',
                    metadata TEXT DEFAULT '{}',
                    status TEXT DEFAULT 'detected',
                    mitigated_at TEXT,
                    mitigation_notes TEXT DEFAULT '',
                    recommendation_id TEXT,
                    vulnerability_ids TEXT DEFAULT '[]'
                )
            """)

            # Recommendations table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS recommendations (
                    recommendation_id TEXT PRIMARY KEY,
                    attack_id TEXT NOT NULL,
                    title TEXT NOT NULL,
                    description TEXT NOT NULL,
                    priority INTEGER NOT NULL,
                    priority_name TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    created_by TEXT DEFAULT 'system',
                    status TEXT DEFAULT 'pending',
                    reviewed_at TEXT,
                    reviewed_by TEXT,
                    review_comments TEXT DEFAULT '',
                    applied_at TEXT,
                    applied_by TEXT,
                    application_result TEXT DEFAULT '',
                    patch_ids TEXT DEFAULT '[]',
                    vulnerability_ids TEXT DEFAULT '[]',
                    metadata TEXT DEFAULT '{}',
                    FOREIGN KEY (attack_id) REFERENCES attacks(attack_id)
                )
            """)

            # Patches table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS patches (
                    patch_id TEXT PRIMARY KEY,
                    recommendation_id TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    patch_type TEXT NOT NULL,
                    description TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    original_content TEXT DEFAULT '',
                    patched_content TEXT DEFAULT '',
                    diff TEXT DEFAULT '',
                    status TEXT DEFAULT 'pending',
                    tested_at TEXT,
                    test_passed INTEGER DEFAULT 0,
                    test_output TEXT DEFAULT '',
                    applied_at TEXT,
                    applied_by TEXT,
                    reverted_at TEXT,
                    metadata TEXT DEFAULT '{}',
                    FOREIGN KEY (recommendation_id) REFERENCES recommendations(recommendation_id)
                )
            """)

            # Vulnerabilities table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS vulnerabilities (
                    vulnerability_id TEXT PRIMARY KEY,
                    attack_id TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    line_start INTEGER NOT NULL,
                    line_end INTEGER NOT NULL,
                    vulnerability_type TEXT NOT NULL,
                    description TEXT NOT NULL,
                    severity INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    code_snippet TEXT DEFAULT '',
                    suggested_fix TEXT DEFAULT '',
                    status TEXT DEFAULT 'open',
                    fixed_at TEXT,
                    fixed_by TEXT,
                    metadata TEXT DEFAULT '{}',
                    FOREIGN KEY (attack_id) REFERENCES attacks(attack_id)
                )
            """)

            # SIEM events table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS siem_events (
                    event_id TEXT PRIMARY KEY,
                    provider TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    source TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    severity INTEGER NOT NULL,
                    category TEXT NOT NULL,
                    description TEXT NOT NULL,
                    raw_data TEXT DEFAULT '{}',
                    metadata TEXT DEFAULT '{}',
                    indicators TEXT DEFAULT '[]',
                    processed INTEGER DEFAULT 0,
                    attack_id TEXT,
                    FOREIGN KEY (attack_id) REFERENCES attacks(attack_id)
                )
            """)

            # Create indexes
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_attacks_detected_at
                ON attacks(detected_at)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_attacks_status
                ON attacks(status)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_attacks_severity
                ON attacks(severity)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_recommendations_status
                ON recommendations(status)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_recommendations_attack
                ON recommendations(attack_id)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_patches_recommendation
                ON patches(recommendation_id)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_vulnerabilities_attack
                ON vulnerabilities(attack_id)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_siem_events_timestamp
                ON siem_events(timestamp)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_siem_events_processed
                ON siem_events(processed)
            """)

            conn.commit()

    def _migrate_schema(self) -> None:
        """Apply schema migrations if needed."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Get current version
            cursor.execute("SELECT MAX(version) FROM schema_version")
            row = cursor.fetchone()
            current_version = row[0] if row[0] else 0

            if current_version < self.SCHEMA_VERSION:
                # Apply migrations
                for version in range(current_version + 1, self.SCHEMA_VERSION + 1):
                    self._apply_migration(conn, version)

                logger.info(f"Schema migrated from v{current_version} to v{self.SCHEMA_VERSION}")

    def _apply_migration(self, conn: sqlite3.Connection, version: int) -> None:
        """Apply a specific migration."""
        cursor = conn.cursor()

        # Future migrations would be added here
        # if version == 2:
        #     cursor.execute("ALTER TABLE attacks ADD COLUMN new_field TEXT")

        # Record migration
        cursor.execute(
            "INSERT INTO schema_version (version, applied_at) VALUES (?, ?)",
            (version, datetime.now().isoformat()),
        )
        conn.commit()

    # -------------------------------------------------------------------------
    # Attack Operations
    # -------------------------------------------------------------------------

    def save_attack(self, attack: StoredAttack) -> bool:
        """Save or update an attack record."""
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                try:
                    cursor.execute(
                        """
                        INSERT OR REPLACE INTO attacks (
                            attack_id, attack_type, severity, severity_name,
                            confidence, description, source, target, detected_at,
                            indicators, mitre_tactics, raw_data, metadata,
                            status, mitigated_at, mitigation_notes,
                            recommendation_id, vulnerability_ids
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            attack.attack_id,
                            attack.attack_type,
                            attack.severity,
                            attack.severity_name,
                            attack.confidence,
                            attack.description,
                            attack.source,
                            attack.target,
                            attack.detected_at.isoformat(),
                            json.dumps(attack.indicators),
                            json.dumps(attack.mitre_tactics),
                            json.dumps(attack.raw_data),
                            json.dumps(attack.metadata),
                            attack.status,
                            attack.mitigated_at.isoformat() if attack.mitigated_at else None,
                            attack.mitigation_notes,
                            attack.recommendation_id,
                            json.dumps(attack.vulnerability_ids),
                        ),
                    )
                    conn.commit()
                    return True
                except Exception as e:
                    logger.error(f"Error saving attack: {e}")
                    return False

    def get_attack(self, attack_id: str) -> Optional[StoredAttack]:
        """Get an attack by ID."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM attacks WHERE attack_id = ?", (attack_id,))
            row = cursor.fetchone()
            if row:
                return StoredAttack.from_dict(dict(row))
            return None

    def list_attacks(
        self,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
        status: Optional[str] = None,
        severity_min: Optional[int] = None,
        attack_type: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[StoredAttack]:
        """List attacks with optional filters."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            query = "SELECT * FROM attacks WHERE 1=1"
            params: List[Any] = []

            if since:
                query += " AND detected_at >= ?"
                params.append(since.isoformat())
            if until:
                query += " AND detected_at <= ?"
                params.append(until.isoformat())
            if status:
                query += " AND status = ?"
                params.append(status)
            if severity_min is not None:
                query += " AND severity >= ?"
                params.append(severity_min)
            if attack_type:
                query += " AND attack_type = ?"
                params.append(attack_type)

            query += " ORDER BY detected_at DESC LIMIT ? OFFSET ?"
            params.extend([limit, offset])

            cursor.execute(query, params)
            return [StoredAttack.from_dict(dict(row)) for row in cursor.fetchall()]

    def update_attack_status(
        self,
        attack_id: str,
        status: str,
        notes: str = "",
    ) -> bool:
        """Update attack status."""
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                try:
                    mitigated_at = datetime.now().isoformat() if status == "mitigated" else None
                    cursor.execute(
                        """
                        UPDATE attacks
                        SET status = ?, mitigation_notes = ?, mitigated_at = ?
                        WHERE attack_id = ?
                        """,
                        (status, notes, mitigated_at, attack_id),
                    )
                    conn.commit()
                    return cursor.rowcount > 0
                except Exception as e:
                    logger.error(f"Error updating attack status: {e}")
                    return False

    def count_attacks(
        self,
        since: Optional[datetime] = None,
        status: Optional[str] = None,
    ) -> int:
        """Count attacks matching criteria."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            query = "SELECT COUNT(*) FROM attacks WHERE 1=1"
            params: List[Any] = []

            if since:
                query += " AND detected_at >= ?"
                params.append(since.isoformat())
            if status:
                query += " AND status = ?"
                params.append(status)

            cursor.execute(query, params)
            return cursor.fetchone()[0]

    # -------------------------------------------------------------------------
    # Recommendation Operations
    # -------------------------------------------------------------------------

    def save_recommendation(self, rec: StoredRecommendation) -> bool:
        """Save or update a recommendation."""
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                try:
                    cursor.execute(
                        """
                        INSERT OR REPLACE INTO recommendations (
                            recommendation_id, attack_id, title, description,
                            priority, priority_name, created_at, created_by,
                            status, reviewed_at, reviewed_by, review_comments,
                            applied_at, applied_by, application_result,
                            patch_ids, vulnerability_ids, metadata
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            rec.recommendation_id,
                            rec.attack_id,
                            rec.title,
                            rec.description,
                            rec.priority,
                            rec.priority_name,
                            rec.created_at.isoformat(),
                            rec.created_by,
                            rec.status,
                            rec.reviewed_at.isoformat() if rec.reviewed_at else None,
                            rec.reviewed_by,
                            rec.review_comments,
                            rec.applied_at.isoformat() if rec.applied_at else None,
                            rec.applied_by,
                            rec.application_result,
                            json.dumps(rec.patch_ids),
                            json.dumps(rec.vulnerability_ids),
                            json.dumps(rec.metadata),
                        ),
                    )
                    conn.commit()
                    return True
                except Exception as e:
                    logger.error(f"Error saving recommendation: {e}")
                    return False

    def get_recommendation(self, rec_id: str) -> Optional[StoredRecommendation]:
        """Get a recommendation by ID."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM recommendations WHERE recommendation_id = ?",
                (rec_id,),
            )
            row = cursor.fetchone()
            if row:
                return StoredRecommendation.from_dict(dict(row))
            return None

    def list_recommendations(
        self,
        status: Optional[str] = None,
        attack_id: Optional[str] = None,
        priority_min: Optional[int] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[StoredRecommendation]:
        """List recommendations with optional filters."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            query = "SELECT * FROM recommendations WHERE 1=1"
            params: List[Any] = []

            if status:
                query += " AND status = ?"
                params.append(status)
            if attack_id:
                query += " AND attack_id = ?"
                params.append(attack_id)
            if priority_min is not None:
                query += " AND priority >= ?"
                params.append(priority_min)

            query += " ORDER BY priority DESC, created_at DESC LIMIT ? OFFSET ?"
            params.extend([limit, offset])

            cursor.execute(query, params)
            return [StoredRecommendation.from_dict(dict(row)) for row in cursor.fetchall()]

    def update_recommendation_status(
        self,
        rec_id: str,
        status: str,
        reviewer: Optional[str] = None,
        comments: str = "",
    ) -> bool:
        """Update recommendation status."""
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                try:
                    reviewed_at = datetime.now().isoformat() if reviewer else None
                    cursor.execute(
                        """
                        UPDATE recommendations
                        SET status = ?, reviewed_at = ?, reviewed_by = ?, review_comments = ?
                        WHERE recommendation_id = ?
                        """,
                        (status, reviewed_at, reviewer, comments, rec_id),
                    )
                    conn.commit()
                    return cursor.rowcount > 0
                except Exception as e:
                    logger.error(f"Error updating recommendation status: {e}")
                    return False

    # -------------------------------------------------------------------------
    # Patch Operations
    # -------------------------------------------------------------------------

    def save_patch(self, patch: StoredPatch) -> bool:
        """Save or update a patch."""
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                try:
                    cursor.execute(
                        """
                        INSERT OR REPLACE INTO patches (
                            patch_id, recommendation_id, file_path, patch_type,
                            description, created_at, original_content, patched_content,
                            diff, status, tested_at, test_passed, test_output,
                            applied_at, applied_by, reverted_at, metadata
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            patch.patch_id,
                            patch.recommendation_id,
                            patch.file_path,
                            patch.patch_type,
                            patch.description,
                            patch.created_at.isoformat(),
                            patch.original_content,
                            patch.patched_content,
                            patch.diff,
                            patch.status,
                            patch.tested_at.isoformat() if patch.tested_at else None,
                            1 if patch.test_passed else 0,
                            patch.test_output,
                            patch.applied_at.isoformat() if patch.applied_at else None,
                            patch.applied_by,
                            patch.reverted_at.isoformat() if patch.reverted_at else None,
                            json.dumps(patch.metadata),
                        ),
                    )
                    conn.commit()
                    return True
                except Exception as e:
                    logger.error(f"Error saving patch: {e}")
                    return False

    def get_patch(self, patch_id: str) -> Optional[StoredPatch]:
        """Get a patch by ID."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM patches WHERE patch_id = ?", (patch_id,))
            row = cursor.fetchone()
            if row:
                data = dict(row)
                data["test_passed"] = bool(data.get("test_passed", 0))
                return StoredPatch.from_dict(data)
            return None

    def list_patches(
        self,
        recommendation_id: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 100,
    ) -> List[StoredPatch]:
        """List patches with optional filters."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            query = "SELECT * FROM patches WHERE 1=1"
            params: List[Any] = []

            if recommendation_id:
                query += " AND recommendation_id = ?"
                params.append(recommendation_id)
            if status:
                query += " AND status = ?"
                params.append(status)

            query += " ORDER BY created_at DESC LIMIT ?"
            params.append(limit)

            cursor.execute(query, params)
            patches = []
            for row in cursor.fetchall():
                data = dict(row)
                data["test_passed"] = bool(data.get("test_passed", 0))
                patches.append(StoredPatch.from_dict(data))
            return patches

    def update_patch_status(
        self,
        patch_id: str,
        status: str,
        test_passed: Optional[bool] = None,
        test_output: str = "",
    ) -> bool:
        """Update patch status."""
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                try:
                    tested_at = datetime.now().isoformat() if test_passed is not None else None
                    cursor.execute(
                        """
                        UPDATE patches
                        SET status = ?, tested_at = ?, test_passed = ?, test_output = ?
                        WHERE patch_id = ?
                        """,
                        (
                            status,
                            tested_at,
                            1 if test_passed else 0 if test_passed is not None else None,
                            test_output,
                            patch_id,
                        ),
                    )
                    conn.commit()
                    return cursor.rowcount > 0
                except Exception as e:
                    logger.error(f"Error updating patch status: {e}")
                    return False

    # -------------------------------------------------------------------------
    # Vulnerability Operations
    # -------------------------------------------------------------------------

    def save_vulnerability(self, vuln: StoredVulnerability) -> bool:
        """Save or update a vulnerability."""
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                try:
                    cursor.execute(
                        """
                        INSERT OR REPLACE INTO vulnerabilities (
                            vulnerability_id, attack_id, file_path, line_start, line_end,
                            vulnerability_type, description, severity, created_at,
                            code_snippet, suggested_fix, status, fixed_at, fixed_by, metadata
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            vuln.vulnerability_id,
                            vuln.attack_id,
                            vuln.file_path,
                            vuln.line_start,
                            vuln.line_end,
                            vuln.vulnerability_type,
                            vuln.description,
                            vuln.severity,
                            vuln.created_at.isoformat(),
                            vuln.code_snippet,
                            vuln.suggested_fix,
                            vuln.status,
                            vuln.fixed_at.isoformat() if vuln.fixed_at else None,
                            vuln.fixed_by,
                            json.dumps(vuln.metadata),
                        ),
                    )
                    conn.commit()
                    return True
                except Exception as e:
                    logger.error(f"Error saving vulnerability: {e}")
                    return False

    def get_vulnerability(self, vuln_id: str) -> Optional[StoredVulnerability]:
        """Get a vulnerability by ID."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM vulnerabilities WHERE vulnerability_id = ?",
                (vuln_id,),
            )
            row = cursor.fetchone()
            if row:
                return StoredVulnerability.from_dict(dict(row))
            return None

    def list_vulnerabilities(
        self,
        attack_id: Optional[str] = None,
        file_path: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 100,
    ) -> List[StoredVulnerability]:
        """List vulnerabilities with optional filters."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            query = "SELECT * FROM vulnerabilities WHERE 1=1"
            params: List[Any] = []

            if attack_id:
                query += " AND attack_id = ?"
                params.append(attack_id)
            if file_path:
                query += " AND file_path = ?"
                params.append(file_path)
            if status:
                query += " AND status = ?"
                params.append(status)

            query += " ORDER BY severity DESC, created_at DESC LIMIT ?"
            params.append(limit)

            cursor.execute(query, params)
            return [StoredVulnerability.from_dict(dict(row)) for row in cursor.fetchall()]

    # -------------------------------------------------------------------------
    # SIEM Event Operations
    # -------------------------------------------------------------------------

    def save_siem_event(self, event: StoredSIEMEvent) -> bool:
        """Save a SIEM event."""
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                try:
                    cursor.execute(
                        """
                        INSERT OR REPLACE INTO siem_events (
                            event_id, provider, timestamp, source, event_type,
                            severity, category, description, raw_data, metadata,
                            indicators, processed, attack_id
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            event.event_id,
                            event.provider,
                            event.timestamp.isoformat(),
                            event.source,
                            event.event_type,
                            event.severity,
                            event.category,
                            event.description,
                            json.dumps(event.raw_data),
                            json.dumps(event.metadata),
                            json.dumps(event.indicators),
                            1 if event.processed else 0,
                            event.attack_id,
                        ),
                    )
                    conn.commit()
                    return True
                except Exception as e:
                    logger.error(f"Error saving SIEM event: {e}")
                    return False

    def list_siem_events(
        self,
        since: Optional[datetime] = None,
        provider: Optional[str] = None,
        processed: Optional[bool] = None,
        limit: int = 100,
    ) -> List[StoredSIEMEvent]:
        """List SIEM events with optional filters."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            query = "SELECT * FROM siem_events WHERE 1=1"
            params: List[Any] = []

            if since:
                query += " AND timestamp >= ?"
                params.append(since.isoformat())
            if provider:
                query += " AND provider = ?"
                params.append(provider)
            if processed is not None:
                query += " AND processed = ?"
                params.append(1 if processed else 0)

            query += " ORDER BY timestamp DESC LIMIT ?"
            params.append(limit)

            cursor.execute(query, params)
            events = []
            for row in cursor.fetchall():
                data = dict(row)
                data["processed"] = bool(data.get("processed", 0))
                events.append(StoredSIEMEvent.from_dict(data))
            return events

    def mark_siem_events_processed(
        self,
        event_ids: List[str],
        attack_id: Optional[str] = None,
    ) -> int:
        """Mark SIEM events as processed."""
        if not event_ids:
            return 0

        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                placeholders = ",".join("?" * len(event_ids))
                try:
                    cursor.execute(
                        f"""
                        UPDATE siem_events
                        SET processed = 1, attack_id = ?
                        WHERE event_id IN ({placeholders})
                        """,
                        [attack_id] + event_ids,
                    )
                    conn.commit()
                    return cursor.rowcount
                except Exception as e:
                    logger.error(f"Error marking SIEM events processed: {e}")
                    return 0

    # -------------------------------------------------------------------------
    # Statistics
    # -------------------------------------------------------------------------

    def get_statistics(self) -> Dict[str, Any]:
        """Get storage statistics."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            stats = {
                "database_path": str(self.db_path),
                "schema_version": self.SCHEMA_VERSION,
            }

            # Attack counts by status
            cursor.execute(
                "SELECT status, COUNT(*) FROM attacks GROUP BY status"
            )
            stats["attacks_by_status"] = dict(cursor.fetchall())

            # Total attacks
            cursor.execute("SELECT COUNT(*) FROM attacks")
            stats["total_attacks"] = cursor.fetchone()[0]

            # Recommendation counts by status
            cursor.execute(
                "SELECT status, COUNT(*) FROM recommendations GROUP BY status"
            )
            stats["recommendations_by_status"] = dict(cursor.fetchall())

            # Total recommendations
            cursor.execute("SELECT COUNT(*) FROM recommendations")
            stats["total_recommendations"] = cursor.fetchone()[0]

            # Patch counts by status
            cursor.execute(
                "SELECT status, COUNT(*) FROM patches GROUP BY status"
            )
            stats["patches_by_status"] = dict(cursor.fetchall())

            # Vulnerability counts by status
            cursor.execute(
                "SELECT status, COUNT(*) FROM vulnerabilities GROUP BY status"
            )
            stats["vulnerabilities_by_status"] = dict(cursor.fetchall())

            # SIEM event counts
            cursor.execute("SELECT COUNT(*) FROM siem_events")
            stats["total_siem_events"] = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM siem_events WHERE processed = 0")
            stats["unprocessed_siem_events"] = cursor.fetchone()[0]

            # Recent activity
            cursor.execute(
                "SELECT COUNT(*) FROM attacks WHERE detected_at >= ?",
                ((datetime.now() - timedelta(hours=24)).isoformat(),),
            )
            stats["attacks_last_24h"] = cursor.fetchone()[0]

            return stats

    # -------------------------------------------------------------------------
    # Maintenance
    # -------------------------------------------------------------------------

    def cleanup_old_records(
        self,
        older_than: datetime,
        keep_unresolved: bool = True,
    ) -> Dict[str, int]:
        """Remove old records."""
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                deleted = {}
                cutoff = older_than.isoformat()

                try:
                    # SIEM events (always clean old ones)
                    cursor.execute(
                        "DELETE FROM siem_events WHERE timestamp < ?",
                        (cutoff,),
                    )
                    deleted["siem_events"] = cursor.rowcount

                    # Attacks
                    if keep_unresolved:
                        cursor.execute(
                            """
                            DELETE FROM attacks
                            WHERE detected_at < ? AND status IN ('mitigated', 'false_positive', 'ignored')
                            """,
                            (cutoff,),
                        )
                    else:
                        cursor.execute(
                            "DELETE FROM attacks WHERE detected_at < ?",
                            (cutoff,),
                        )
                    deleted["attacks"] = cursor.rowcount

                    # Recommendations (cascade with attacks)
                    cursor.execute(
                        """
                        DELETE FROM recommendations
                        WHERE attack_id NOT IN (SELECT attack_id FROM attacks)
                        """,
                    )
                    deleted["recommendations"] = cursor.rowcount

                    # Patches (cascade with recommendations)
                    cursor.execute(
                        """
                        DELETE FROM patches
                        WHERE recommendation_id NOT IN (SELECT recommendation_id FROM recommendations)
                        """,
                    )
                    deleted["patches"] = cursor.rowcount

                    # Vulnerabilities (cascade with attacks)
                    cursor.execute(
                        """
                        DELETE FROM vulnerabilities
                        WHERE attack_id NOT IN (SELECT attack_id FROM attacks)
                        """,
                    )
                    deleted["vulnerabilities"] = cursor.rowcount

                    conn.commit()

                    # Vacuum to reclaim space
                    cursor.execute("VACUUM")

                    logger.info(f"Cleanup completed: {deleted}")
                    return deleted

                except Exception as e:
                    logger.error(f"Error during cleanup: {e}")
                    return {}

    def close(self) -> None:
        """Close storage connection."""
        if hasattr(self._local, "connection"):
            try:
                self._local.connection.close()
            except Exception:
                logger.debug("Error closing SQLite connection", exc_info=True)
            delattr(self._local, "connection")
        logger.info("SQLite storage closed")


# =============================================================================
# In-Memory Storage Implementation
# =============================================================================


class MemoryStorage(AttackStorage):
    """
    In-memory storage for testing and development.

    Data is lost when the process exits.
    """

    def __init__(self):
        self._attacks: Dict[str, StoredAttack] = {}
        self._recommendations: Dict[str, StoredRecommendation] = {}
        self._patches: Dict[str, StoredPatch] = {}
        self._vulnerabilities: Dict[str, StoredVulnerability] = {}
        self._siem_events: Dict[str, StoredSIEMEvent] = {}
        self._lock = threading.Lock()

        logger.info("In-memory storage initialized")

    # -------------------------------------------------------------------------
    # Attack Operations
    # -------------------------------------------------------------------------

    def save_attack(self, attack: StoredAttack) -> bool:
        with self._lock:
            self._attacks[attack.attack_id] = attack
            return True

    def get_attack(self, attack_id: str) -> Optional[StoredAttack]:
        return self._attacks.get(attack_id)

    def list_attacks(
        self,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
        status: Optional[str] = None,
        severity_min: Optional[int] = None,
        attack_type: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[StoredAttack]:
        attacks = list(self._attacks.values())

        if since:
            attacks = [a for a in attacks if a.detected_at >= since]
        if until:
            attacks = [a for a in attacks if a.detected_at <= until]
        if status:
            attacks = [a for a in attacks if a.status == status]
        if severity_min is not None:
            attacks = [a for a in attacks if a.severity >= severity_min]
        if attack_type:
            attacks = [a for a in attacks if a.attack_type == attack_type]

        attacks.sort(key=lambda a: a.detected_at, reverse=True)
        return attacks[offset : offset + limit]

    def update_attack_status(
        self,
        attack_id: str,
        status: str,
        notes: str = "",
    ) -> bool:
        with self._lock:
            if attack_id in self._attacks:
                attack = self._attacks[attack_id]
                attack.status = status
                attack.mitigation_notes = notes
                if status == "mitigated":
                    attack.mitigated_at = datetime.now()
                return True
            return False

    def count_attacks(
        self,
        since: Optional[datetime] = None,
        status: Optional[str] = None,
    ) -> int:
        attacks = list(self._attacks.values())
        if since:
            attacks = [a for a in attacks if a.detected_at >= since]
        if status:
            attacks = [a for a in attacks if a.status == status]
        return len(attacks)

    # -------------------------------------------------------------------------
    # Recommendation Operations
    # -------------------------------------------------------------------------

    def save_recommendation(self, rec: StoredRecommendation) -> bool:
        with self._lock:
            self._recommendations[rec.recommendation_id] = rec
            return True

    def get_recommendation(self, rec_id: str) -> Optional[StoredRecommendation]:
        return self._recommendations.get(rec_id)

    def list_recommendations(
        self,
        status: Optional[str] = None,
        attack_id: Optional[str] = None,
        priority_min: Optional[int] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[StoredRecommendation]:
        recs = list(self._recommendations.values())

        if status:
            recs = [r for r in recs if r.status == status]
        if attack_id:
            recs = [r for r in recs if r.attack_id == attack_id]
        if priority_min is not None:
            recs = [r for r in recs if r.priority >= priority_min]

        recs.sort(key=lambda r: (r.priority, r.created_at), reverse=True)
        return recs[offset : offset + limit]

    def update_recommendation_status(
        self,
        rec_id: str,
        status: str,
        reviewer: Optional[str] = None,
        comments: str = "",
    ) -> bool:
        with self._lock:
            if rec_id in self._recommendations:
                rec = self._recommendations[rec_id]
                rec.status = status
                if reviewer:
                    rec.reviewed_at = datetime.now()
                    rec.reviewed_by = reviewer
                rec.review_comments = comments
                return True
            return False

    # -------------------------------------------------------------------------
    # Patch Operations
    # -------------------------------------------------------------------------

    def save_patch(self, patch: StoredPatch) -> bool:
        with self._lock:
            self._patches[patch.patch_id] = patch
            return True

    def get_patch(self, patch_id: str) -> Optional[StoredPatch]:
        return self._patches.get(patch_id)

    def list_patches(
        self,
        recommendation_id: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 100,
    ) -> List[StoredPatch]:
        patches = list(self._patches.values())

        if recommendation_id:
            patches = [p for p in patches if p.recommendation_id == recommendation_id]
        if status:
            patches = [p for p in patches if p.status == status]

        patches.sort(key=lambda p: p.created_at, reverse=True)
        return patches[:limit]

    def update_patch_status(
        self,
        patch_id: str,
        status: str,
        test_passed: Optional[bool] = None,
        test_output: str = "",
    ) -> bool:
        with self._lock:
            if patch_id in self._patches:
                patch = self._patches[patch_id]
                patch.status = status
                if test_passed is not None:
                    patch.tested_at = datetime.now()
                    patch.test_passed = test_passed
                patch.test_output = test_output
                return True
            return False

    # -------------------------------------------------------------------------
    # Vulnerability Operations
    # -------------------------------------------------------------------------

    def save_vulnerability(self, vuln: StoredVulnerability) -> bool:
        with self._lock:
            self._vulnerabilities[vuln.vulnerability_id] = vuln
            return True

    def get_vulnerability(self, vuln_id: str) -> Optional[StoredVulnerability]:
        return self._vulnerabilities.get(vuln_id)

    def list_vulnerabilities(
        self,
        attack_id: Optional[str] = None,
        file_path: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 100,
    ) -> List[StoredVulnerability]:
        vulns = list(self._vulnerabilities.values())

        if attack_id:
            vulns = [v for v in vulns if v.attack_id == attack_id]
        if file_path:
            vulns = [v for v in vulns if v.file_path == file_path]
        if status:
            vulns = [v for v in vulns if v.status == status]

        vulns.sort(key=lambda v: (v.severity, v.created_at), reverse=True)
        return vulns[:limit]

    # -------------------------------------------------------------------------
    # SIEM Event Operations
    # -------------------------------------------------------------------------

    def save_siem_event(self, event: StoredSIEMEvent) -> bool:
        with self._lock:
            self._siem_events[event.event_id] = event
            return True

    def list_siem_events(
        self,
        since: Optional[datetime] = None,
        provider: Optional[str] = None,
        processed: Optional[bool] = None,
        limit: int = 100,
    ) -> List[StoredSIEMEvent]:
        events = list(self._siem_events.values())

        if since:
            events = [e for e in events if e.timestamp >= since]
        if provider:
            events = [e for e in events if e.provider == provider]
        if processed is not None:
            events = [e for e in events if e.processed == processed]

        events.sort(key=lambda e: e.timestamp, reverse=True)
        return events[:limit]

    def mark_siem_events_processed(
        self,
        event_ids: List[str],
        attack_id: Optional[str] = None,
    ) -> int:
        count = 0
        with self._lock:
            for event_id in event_ids:
                if event_id in self._siem_events:
                    self._siem_events[event_id].processed = True
                    self._siem_events[event_id].attack_id = attack_id
                    count += 1
        return count

    # -------------------------------------------------------------------------
    # Statistics
    # -------------------------------------------------------------------------

    def get_statistics(self) -> Dict[str, Any]:
        stats = {
            "backend": "memory",
            "total_attacks": len(self._attacks),
            "total_recommendations": len(self._recommendations),
            "total_patches": len(self._patches),
            "total_vulnerabilities": len(self._vulnerabilities),
            "total_siem_events": len(self._siem_events),
        }

        # Counts by status
        stats["attacks_by_status"] = {}
        for attack in self._attacks.values():
            stats["attacks_by_status"][attack.status] = (
                stats["attacks_by_status"].get(attack.status, 0) + 1
            )

        stats["recommendations_by_status"] = {}
        for rec in self._recommendations.values():
            stats["recommendations_by_status"][rec.status] = (
                stats["recommendations_by_status"].get(rec.status, 0) + 1
            )

        return stats

    # -------------------------------------------------------------------------
    # Maintenance
    # -------------------------------------------------------------------------

    def cleanup_old_records(
        self,
        older_than: datetime,
        keep_unresolved: bool = True,
    ) -> Dict[str, int]:
        deleted = {}
        with self._lock:
            # SIEM events
            old_events = [
                e for e in self._siem_events.values() if e.timestamp < older_than
            ]
            for event in old_events:
                del self._siem_events[event.event_id]
            deleted["siem_events"] = len(old_events)

            # Attacks
            resolved_statuses = {"mitigated", "false_positive", "ignored"}
            if keep_unresolved:
                old_attacks = [
                    a
                    for a in self._attacks.values()
                    if a.detected_at < older_than and a.status in resolved_statuses
                ]
            else:
                old_attacks = [
                    a for a in self._attacks.values() if a.detected_at < older_than
                ]
            attack_ids = {a.attack_id for a in old_attacks}
            for attack in old_attacks:
                del self._attacks[attack.attack_id]
            deleted["attacks"] = len(old_attacks)

            # Cascade deletes
            old_recs = [
                r for r in self._recommendations.values() if r.attack_id in attack_ids
            ]
            rec_ids = {r.recommendation_id for r in old_recs}
            for rec in old_recs:
                del self._recommendations[rec.recommendation_id]
            deleted["recommendations"] = len(old_recs)

            old_patches = [
                p for p in self._patches.values() if p.recommendation_id in rec_ids
            ]
            for patch in old_patches:
                del self._patches[patch.patch_id]
            deleted["patches"] = len(old_patches)

            old_vulns = [
                v for v in self._vulnerabilities.values() if v.attack_id in attack_ids
            ]
            for vuln in old_vulns:
                del self._vulnerabilities[vuln.vulnerability_id]
            deleted["vulnerabilities"] = len(old_vulns)

        return deleted

    def close(self) -> None:
        """No-op for memory storage."""
        pass


# =============================================================================
# Factory Functions
# =============================================================================


def create_storage(
    backend: StorageBackend = StorageBackend.SQLITE,
    db_path: Optional[Union[str, Path]] = None,
    **kwargs,
) -> AttackStorage:
    """
    Create a storage instance.

    Args:
        backend: Storage backend type
        db_path: Path to database (for SQLite)
        **kwargs: Additional backend-specific arguments

    Returns:
        AttackStorage implementation
    """
    if backend == StorageBackend.SQLITE:
        path = db_path or Path.home() / ".agent-os" / "attack_detection.db"
        return SQLiteStorage(db_path=path, **kwargs)
    elif backend == StorageBackend.MEMORY:
        return MemoryStorage()
    else:
        raise ValueError(f"Unknown storage backend: {backend}")


def create_sqlite_storage(
    db_path: Union[str, Path] = "attack_detection.db",
    auto_migrate: bool = True,
) -> SQLiteStorage:
    """Create SQLite storage instance."""
    return SQLiteStorage(db_path=db_path, auto_migrate=auto_migrate)


def create_memory_storage() -> MemoryStorage:
    """Create in-memory storage instance."""
    return MemoryStorage()
