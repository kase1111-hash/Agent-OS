"""
Intent Log Module

Tracks user intents and actions per user for audit and analytics purposes.
Each user has their own isolated intent log.
"""

import hashlib
import json
import logging
import secrets
import sqlite3
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class IntentType(Enum):
    """Types of user intents."""

    CHAT_MESSAGE = auto()  # User sent a chat message
    COMMAND = auto()  # User executed a command
    NAVIGATION = auto()  # User navigated to a view
    CONTRACT_CREATE = auto()  # User created a contract
    CONTRACT_REVOKE = auto()  # User revoked a contract
    CONTRACT_VIEW = auto()  # User viewed contracts
    MEMORY_CREATE = auto()  # User created a memory
    MEMORY_DELETE = auto()  # User deleted a memory
    MEMORY_SEARCH = auto()  # User searched memories
    AGENT_INTERACT = auto()  # User interacted with an agent
    RULE_CREATE = auto()  # User created a rule
    RULE_UPDATE = auto()  # User updated a rule
    RULE_DELETE = auto()  # User deleted a rule
    AUTH_LOGIN = auto()  # User logged in
    AUTH_LOGOUT = auto()  # User logged out
    AUTH_REGISTER = auto()  # User registered
    SETTINGS_CHANGE = auto()  # User changed settings
    SYSTEM_ACTION = auto()  # Other system action
    EXPORT = auto()  # User exported data
    IMPORT = auto()  # User imported data


@dataclass
class IntentLogEntry:
    """A single intent log entry."""

    entry_id: str
    user_id: str
    intent_type: IntentType
    description: str
    details: Dict[str, Any] = field(default_factory=dict)
    session_id: Optional[str] = None
    related_entity_type: Optional[str] = None  # e.g., "conversation", "contract", "memory"
    related_entity_id: Optional[str] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "entry_id": self.entry_id,
            "user_id": self.user_id,
            "intent_type": self.intent_type.name,
            "description": self.description,
            "details": self.details,
            "session_id": self.session_id,
            "related_entity_type": self.related_entity_type,
            "related_entity_id": self.related_entity_id,
            "ip_address": self.ip_address,
            "user_agent": self.user_agent,
            "created_at": self.created_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "IntentLogEntry":
        """Create from dictionary."""
        return cls(
            entry_id=data["entry_id"],
            user_id=data["user_id"],
            intent_type=IntentType[data["intent_type"]],
            description=data["description"],
            details=data.get("details", {}),
            session_id=data.get("session_id"),
            related_entity_type=data.get("related_entity_type"),
            related_entity_id=data.get("related_entity_id"),
            ip_address=data.get("ip_address"),
            user_agent=data.get("user_agent"),
            created_at=datetime.fromisoformat(data["created_at"]),
        )


@dataclass
class IntentLogQuery:
    """Query parameters for finding intent log entries."""

    user_id: Optional[str] = None
    intent_type: Optional[IntentType] = None
    intent_types: Optional[List[IntentType]] = None
    session_id: Optional[str] = None
    related_entity_type: Optional[str] = None
    related_entity_id: Optional[str] = None
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    search_text: Optional[str] = None
    limit: int = 100
    offset: int = 0


@dataclass
class ChainVerificationResult:
    """V6-3: Result of audit log hash-chain verification."""

    entries_checked: int
    chain_breaks: List[str]
    is_valid: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entries_checked": self.entries_checked,
            "chain_breaks": self.chain_breaks,
            "is_valid": self.is_valid,
        }


class IntentLogStore:
    """
    Persistent store for intent log entries.

    Uses SQLite for storage with user-scoped data isolation.
    """

    def __init__(self, db_path: Optional[Path] = None):
        """
        Initialize intent log store.

        Args:
            db_path: Path to SQLite database (None for in-memory)
        """
        self.db_path = db_path
        self._lock = threading.Lock()
        self._connection: Optional[sqlite3.Connection] = None
        self._initialized = False
        # V6-3: Hash chain state — the hash of the most recent entry
        self._latest_hash: str = "GENESIS"

    def initialize(self) -> bool:
        """Initialize the intent log store."""
        try:
            if self.db_path:
                self.db_path.parent.mkdir(parents=True, exist_ok=True)

            db_str = str(self.db_path) if self.db_path else ":memory:"
            self._connection = sqlite3.connect(db_str, check_same_thread=False)
            self._connection.row_factory = sqlite3.Row

            # V6-6: Enable WAL mode for better concurrent access
            self._connection.execute("PRAGMA journal_mode=WAL")

            self._create_tables()

            # V6-6: Set restrictive file permissions (owner read/write only)
            if self.db_path:
                try:
                    import os
                    os.chmod(self.db_path, 0o600)
                    # Also restrict the WAL and SHM files if they exist
                    for suffix in ("-wal", "-shm"):
                        wal_path = Path(str(self.db_path) + suffix)
                        if wal_path.exists():
                            os.chmod(wal_path, 0o600)
                except OSError as e:
                    logger.warning(f"Could not set file permissions on {self.db_path}: {e}")
            self._initialized = True

            # V6-3: Restore hash chain head from existing entries
            try:
                cursor = self._connection.cursor()
                cursor.execute(
                    "SELECT entry_hash FROM intent_log "
                    "WHERE entry_hash IS NOT NULL "
                    "ORDER BY created_at DESC LIMIT 1"
                )
                row = cursor.fetchone()
                if row and row[0]:
                    self._latest_hash = row[0]
            except (KeyError, sqlite3.Error) as e:
                logger.warning(f"Failed to read latest hash from intent log: {e}")

            logger.info(f"Intent log store initialized: {db_str}")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize intent log store: {e}")
            return False

    def close(self) -> None:
        """Close the store."""
        if self._connection:
            self._connection.close()
            self._connection = None
        self._initialized = False

    def log_intent(
        self,
        user_id: str,
        intent_type: IntentType,
        description: str,
        details: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None,
        related_entity_type: Optional[str] = None,
        related_entity_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
    ) -> IntentLogEntry:
        """
        Log a user intent.

        Args:
            user_id: User performing the action
            intent_type: Type of intent
            description: Human-readable description
            details: Additional details
            session_id: User session ID
            related_entity_type: Type of related entity
            related_entity_id: ID of related entity
            ip_address: Client IP address
            user_agent: Client user agent

        Returns:
            Created IntentLogEntry
        """
        if not self._initialized:
            raise RuntimeError("Intent log store not initialized")

        entry = IntentLogEntry(
            entry_id=f"INT-{secrets.token_hex(8)}",
            user_id=user_id,
            intent_type=intent_type,
            description=description,
            details=details or {},
            session_id=session_id,
            related_entity_type=related_entity_type,
            related_entity_id=related_entity_id,
            ip_address=ip_address,
            user_agent=user_agent,
        )

        with self._lock:
            self._insert_entry(entry)

        logger.debug(f"Logged intent: {entry.entry_id} - {intent_type.name} for user {user_id}")
        return entry

    def get_entry(self, entry_id: str) -> Optional[IntentLogEntry]:
        """Get an entry by ID."""
        with self._lock:
            return self._get_entry(entry_id)

    def query_entries(self, query: IntentLogQuery) -> List[IntentLogEntry]:
        """Query intent log entries."""
        with self._lock:
            return self._query_entries(query)

    def get_user_entries(
        self,
        user_id: str,
        limit: int = 100,
        offset: int = 0,
    ) -> List[IntentLogEntry]:
        """Get all entries for a user."""
        query = IntentLogQuery(user_id=user_id, limit=limit, offset=offset)
        return self.query_entries(query)

    def get_recent_entries(
        self,
        user_id: str,
        hours: int = 24,
        limit: int = 100,
    ) -> List[IntentLogEntry]:
        """Get recent entries for a user."""
        query = IntentLogQuery(
            user_id=user_id,
            start_date=datetime.now() - timedelta(hours=hours),
            limit=limit,
        )
        return self.query_entries(query)

    def get_session_entries(
        self,
        user_id: str,
        session_id: str,
    ) -> List[IntentLogEntry]:
        """Get all entries for a specific session."""
        query = IntentLogQuery(user_id=user_id, session_id=session_id)
        return self.query_entries(query)

    def get_statistics(self, user_id: str) -> Dict[str, Any]:
        """Get statistics for a user's intent log."""
        with self._lock:
            cursor = self._connection.cursor()

            stats = {
                "user_id": user_id,
                "total_entries": 0,
                "by_type": {},
                "recent_24h": 0,
                "recent_7d": 0,
            }

            # Total entries
            cursor.execute("SELECT COUNT(*) FROM intent_log WHERE user_id = ?", (user_id,))
            stats["total_entries"] = cursor.fetchone()[0]

            # By type
            cursor.execute(
                """
                SELECT intent_type, COUNT(*)
                FROM intent_log
                WHERE user_id = ?
                GROUP BY intent_type
            """,
                (user_id,),
            )
            for row in cursor.fetchall():
                stats["by_type"][row[0]] = row[1]

            # Recent 24h
            cursor.execute(
                """
                SELECT COUNT(*) FROM intent_log
                WHERE user_id = ? AND created_at > ?
            """,
                (user_id, (datetime.now() - timedelta(hours=24)).isoformat()),
            )
            stats["recent_24h"] = cursor.fetchone()[0]

            # Recent 7d
            cursor.execute(
                """
                SELECT COUNT(*) FROM intent_log
                WHERE user_id = ? AND created_at > ?
            """,
                (user_id, (datetime.now() - timedelta(days=7)).isoformat()),
            )
            stats["recent_7d"] = cursor.fetchone()[0]

            return stats

    def delete_user_entries(self, user_id: str) -> int:
        """Delete all entries for a user. Returns count deleted."""
        with self._lock:
            cursor = self._connection.cursor()
            cursor.execute("DELETE FROM intent_log WHERE user_id = ?", (user_id,))
            self._connection.commit()
            count = cursor.rowcount
            logger.info(f"Deleted {count} intent log entries for user {user_id}")
            return count

    def delete_old_entries(self, days: int = 90) -> int:
        """Delete entries older than specified days. Returns count deleted."""
        with self._lock:
            cursor = self._connection.cursor()
            cutoff = (datetime.now() - timedelta(days=days)).isoformat()
            cursor.execute("DELETE FROM intent_log WHERE created_at < ?", (cutoff,))
            self._connection.commit()
            count = cursor.rowcount
            logger.info(f"Deleted {count} intent log entries older than {days} days")
            return count

    def _create_tables(self) -> None:
        """Create database tables."""
        cursor = self._connection.cursor()

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS intent_log (
                entry_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                intent_type TEXT NOT NULL,
                description TEXT NOT NULL,
                details_json TEXT,
                session_id TEXT,
                related_entity_type TEXT,
                related_entity_id TEXT,
                ip_address TEXT,
                user_agent TEXT,
                created_at TEXT NOT NULL,
                entry_hash TEXT,
                previous_hash TEXT
            )
        """
        )

        # V6-3: Migration — add hash columns if table already exists without them
        try:
            cursor.execute(
                "ALTER TABLE intent_log ADD COLUMN entry_hash TEXT"
            )
        except sqlite3.OperationalError:
            pass  # Column already exists
        try:
            cursor.execute(
                "ALTER TABLE intent_log ADD COLUMN previous_hash TEXT"
            )
        except sqlite3.OperationalError:
            pass  # Column already exists

        # Indexes for common queries
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_intent_log_user_id
            ON intent_log(user_id)
        """
        )

        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_intent_log_user_created
            ON intent_log(user_id, created_at DESC)
        """
        )

        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_intent_log_session
            ON intent_log(session_id)
        """
        )

        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_intent_log_type
            ON intent_log(intent_type)
        """
        )

        self._connection.commit()

    def _compute_entry_hash(self, entry: IntentLogEntry, previous_hash: str) -> str:
        """V6-3: Compute a tamper-evident hash for a log entry."""
        details_json = json.dumps(entry.details, sort_keys=True) if entry.details else ""
        payload = (
            f"{previous_hash}|{entry.entry_id}|{entry.user_id}|"
            f"{entry.intent_type.name}|{entry.created_at.isoformat()}|{details_json}"
        )
        return hashlib.sha256(payload.encode()).hexdigest()

    def verify_chain(self, limit: int = 0) -> "ChainVerificationResult":
        """
        V6-3: Walk the hash chain and report any breaks.

        Args:
            limit: Maximum entries to verify (0 = all)

        Returns:
            ChainVerificationResult with verification status
        """
        with self._lock:
            cursor = self._connection.cursor()
            query = (
                "SELECT entry_id, user_id, intent_type, description, "
                "details_json, created_at, entry_hash, previous_hash "
                "FROM intent_log ORDER BY created_at ASC"
            )
            if limit > 0:
                query += f" LIMIT {int(limit)}"
            cursor.execute(query)
            rows = cursor.fetchall()

            verified = 0
            breaks: List[str] = []
            expected_prev = "GENESIS"

            for row in rows:
                entry_hash = row["entry_hash"]
                previous_hash = row["previous_hash"]

                if entry_hash is None:
                    # Pre-migration entry, skip
                    continue

                if previous_hash != expected_prev:
                    breaks.append(
                        f"Chain break at {row['entry_id']}: "
                        f"expected previous_hash={expected_prev!r}, "
                        f"got {previous_hash!r}"
                    )

                # Recompute hash
                details = json.loads(row["details_json"]) if row["details_json"] else {}
                payload = (
                    f"{previous_hash}|{row['entry_id']}|{row['user_id']}|"
                    f"{row['intent_type']}|{row['created_at']}|"
                    f"{json.dumps(details, sort_keys=True) if details else ''}"
                )
                recomputed = hashlib.sha256(payload.encode()).hexdigest()
                if recomputed != entry_hash:
                    breaks.append(
                        f"Hash mismatch at {row['entry_id']}: "
                        f"stored={entry_hash[:16]}..., "
                        f"computed={recomputed[:16]}..."
                    )

                verified += 1
                expected_prev = entry_hash

            return ChainVerificationResult(
                entries_checked=verified,
                chain_breaks=breaks,
                is_valid=len(breaks) == 0,
            )

    def _insert_entry(self, entry: IntentLogEntry) -> None:
        """Insert an entry into the database."""
        # V6-3: Compute hash chain
        previous_hash = self._latest_hash
        entry_hash = self._compute_entry_hash(entry, previous_hash)

        cursor = self._connection.cursor()
        cursor.execute(
            """
            INSERT INTO intent_log (
                entry_id, user_id, intent_type, description, details_json,
                session_id, related_entity_type, related_entity_id,
                ip_address, user_agent, created_at, entry_hash, previous_hash
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                entry.entry_id,
                entry.user_id,
                entry.intent_type.name,
                entry.description,
                json.dumps(entry.details) if entry.details else None,
                entry.session_id,
                entry.related_entity_type,
                entry.related_entity_id,
                entry.ip_address,
                entry.user_agent,
                entry.created_at.isoformat(),
                entry_hash,
                previous_hash,
            ),
        )
        self._connection.commit()
        # V6-3: Advance the chain head
        self._latest_hash = entry_hash

    def _get_entry(self, entry_id: str) -> Optional[IntentLogEntry]:
        """Get an entry from the database."""
        cursor = self._connection.cursor()
        cursor.execute("SELECT * FROM intent_log WHERE entry_id = ?", (entry_id,))
        row = cursor.fetchone()
        if not row:
            return None
        return self._row_to_entry(row)

    def _query_entries(self, query: IntentLogQuery) -> List[IntentLogEntry]:
        """Query entries from the database."""
        sql = "SELECT * FROM intent_log WHERE 1=1"
        params = []

        if query.user_id:
            sql += " AND user_id = ?"
            params.append(query.user_id)

        if query.intent_type:
            sql += " AND intent_type = ?"
            params.append(query.intent_type.name)

        if query.intent_types:
            placeholders = ",".join("?" * len(query.intent_types))
            sql += f" AND intent_type IN ({placeholders})"
            params.extend(t.name for t in query.intent_types)

        if query.session_id:
            sql += " AND session_id = ?"
            params.append(query.session_id)

        if query.related_entity_type:
            sql += " AND related_entity_type = ?"
            params.append(query.related_entity_type)

        if query.related_entity_id:
            sql += " AND related_entity_id = ?"
            params.append(query.related_entity_id)

        if query.start_date:
            sql += " AND created_at >= ?"
            params.append(query.start_date.isoformat())

        if query.end_date:
            sql += " AND created_at <= ?"
            params.append(query.end_date.isoformat())

        if query.search_text:
            sql += " AND (description LIKE ? OR details_json LIKE ?)"
            search_pattern = f"%{query.search_text}%"
            params.extend([search_pattern, search_pattern])

        sql += f" ORDER BY created_at DESC LIMIT ? OFFSET ?"
        params.extend([query.limit, query.offset])

        cursor = self._connection.cursor()
        cursor.execute(sql, params)

        return [self._row_to_entry(row) for row in cursor.fetchall()]

    def _row_to_entry(self, row: sqlite3.Row) -> IntentLogEntry:
        """Convert database row to IntentLogEntry."""
        return IntentLogEntry(
            entry_id=row["entry_id"],
            user_id=row["user_id"],
            intent_type=IntentType[row["intent_type"]],
            description=row["description"],
            details=json.loads(row["details_json"]) if row["details_json"] else {},
            session_id=row["session_id"],
            related_entity_type=row["related_entity_type"],
            related_entity_id=row["related_entity_id"],
            ip_address=row["ip_address"],
            user_agent=row["user_agent"],
            created_at=datetime.fromisoformat(row["created_at"]),
        )


# Global store instance
_intent_log_store: Optional[IntentLogStore] = None


def get_intent_log_store() -> IntentLogStore:
    """Get or create the global intent log store."""
    global _intent_log_store
    if _intent_log_store is None:
        from .config import get_config

        config = get_config()
        db_path = config.data_dir / "intent_log.db"
        _intent_log_store = IntentLogStore(db_path=db_path)
        _intent_log_store.initialize()
    return _intent_log_store


def log_user_intent(
    user_id: str,
    intent_type: IntentType,
    description: str,
    details: Optional[Dict[str, Any]] = None,
    session_id: Optional[str] = None,
    related_entity_type: Optional[str] = None,
    related_entity_id: Optional[str] = None,
    ip_address: Optional[str] = None,
    user_agent: Optional[str] = None,
) -> IntentLogEntry:
    """
    Convenience function to log a user intent.

    Args:
        user_id: User performing the action
        intent_type: Type of intent
        description: Human-readable description
        details: Additional details
        session_id: User session ID
        related_entity_type: Type of related entity
        related_entity_id: ID of related entity
        ip_address: Client IP address
        user_agent: Client user agent

    Returns:
        Created IntentLogEntry
    """
    store = get_intent_log_store()
    return store.log_intent(
        user_id=user_id,
        intent_type=intent_type,
        description=description,
        details=details,
        session_id=session_id,
        related_entity_type=related_entity_type,
        related_entity_id=related_entity_id,
        ip_address=ip_address,
        user_agent=user_agent,
    )
