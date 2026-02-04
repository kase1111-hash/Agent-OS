"""
Agent OS Memory Vault Index Database

SQLite-based index for efficient vault queries:
- Blob metadata indexing
- Consent tracking
- Access logging
- Deletion propagation
"""

import json
import logging
import sqlite3
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .profiles import EncryptionTier
from .storage import BlobMetadata, BlobStatus, BlobType

logger = logging.getLogger(__name__)


class AccessType(Enum):
    """Type of access event."""

    READ = auto()
    WRITE = auto()
    DELETE = auto()
    SEAL = auto()
    UNSEAL = auto()
    PROMOTE = auto()
    QUERY = auto()


@dataclass
class AccessLogEntry:
    """Entry in the access log."""

    log_id: int
    blob_id: str
    access_type: AccessType
    accessor: str
    timestamp: datetime
    success: bool
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "log_id": self.log_id,
            "blob_id": self.blob_id,
            "access_type": self.access_type.name,
            "accessor": self.accessor,
            "timestamp": self.timestamp.isoformat(),
            "success": self.success,
            "details": self.details,
        }


@dataclass
class ConsentRecord:
    """Record of consent for memory operations."""

    consent_id: str
    granted_by: str
    granted_at: datetime
    expires_at: Optional[datetime]
    scope: str  # What the consent covers
    operations: List[str]  # Allowed operations
    active: bool = True
    revoked_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "consent_id": self.consent_id,
            "granted_by": self.granted_by,
            "granted_at": self.granted_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "scope": self.scope,
            "operations": self.operations,
            "active": self.active,
            "revoked_at": self.revoked_at.isoformat() if self.revoked_at else None,
            "metadata": self.metadata,
        }


class VaultIndex:
    """
    SQLite-based index for the Memory Vault.

    Provides:
    - Efficient metadata queries
    - Access logging
    - Consent tracking
    - Deletion cascade support
    """

    SCHEMA_VERSION = 1

    def __init__(self, db_path: Path):
        """
        Initialize vault index.

        Args:
            db_path: Path to SQLite database
        """
        self._db_path = db_path
        self._local = threading.local()
        self._lock = threading.RLock()

        self._initialize_database()

    def _get_connection(self) -> sqlite3.Connection:
        """Get thread-local database connection."""
        if not hasattr(self._local, "connection") or self._local.connection is None:
            self._local.connection = sqlite3.connect(
                str(self._db_path),
                check_same_thread=False,
            )
            self._local.connection.row_factory = sqlite3.Row
            # Enable foreign keys
            self._local.connection.execute("PRAGMA foreign_keys = ON")
        return self._local.connection

    @contextmanager
    def _transaction(self):
        """Context manager for transactions."""
        conn = self._get_connection()
        try:
            yield conn
            conn.commit()
        except sqlite3.Error:
            conn.rollback()
            raise

    def _initialize_database(self) -> None:
        """Initialize database schema."""
        self._db_path.parent.mkdir(parents=True, exist_ok=True)

        with self._transaction() as conn:
            # Schema version table
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS schema_version (
                    version INTEGER PRIMARY KEY,
                    applied_at TEXT NOT NULL
                )
            """
            )

            # Check current version
            cursor = conn.execute("SELECT MAX(version) FROM schema_version")
            current_version = cursor.fetchone()[0] or 0

            if current_version < self.SCHEMA_VERSION:
                self._apply_migrations(conn, current_version)

    def _apply_migrations(self, conn: sqlite3.Connection, from_version: int) -> None:
        """Apply database migrations."""
        if from_version < 1:
            # Initial schema
            conn.executescript(
                """
                -- Blob metadata index
                CREATE TABLE IF NOT EXISTS blobs (
                    blob_id TEXT PRIMARY KEY,
                    key_id TEXT NOT NULL,
                    tier TEXT NOT NULL,
                    blob_type TEXT NOT NULL,
                    size_bytes INTEGER NOT NULL,
                    encrypted_size INTEGER NOT NULL,
                    content_hash TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    modified_at TEXT NOT NULL,
                    accessed_at TEXT,
                    access_count INTEGER DEFAULT 0,
                    status TEXT NOT NULL DEFAULT 'ACTIVE',
                    consent_id TEXT,
                    ttl_seconds INTEGER,
                    tags TEXT,  -- JSON array
                    custom_metadata TEXT,  -- JSON object
                    FOREIGN KEY (consent_id) REFERENCES consents(consent_id)
                );

                CREATE INDEX IF NOT EXISTS idx_blobs_tier ON blobs(tier);
                CREATE INDEX IF NOT EXISTS idx_blobs_status ON blobs(status);
                CREATE INDEX IF NOT EXISTS idx_blobs_consent ON blobs(consent_id);
                CREATE INDEX IF NOT EXISTS idx_blobs_created ON blobs(created_at);
                CREATE INDEX IF NOT EXISTS idx_blobs_hash ON blobs(content_hash);

                -- Consent records
                CREATE TABLE IF NOT EXISTS consents (
                    consent_id TEXT PRIMARY KEY,
                    granted_by TEXT NOT NULL,
                    granted_at TEXT NOT NULL,
                    expires_at TEXT,
                    scope TEXT NOT NULL,
                    operations TEXT NOT NULL,  -- JSON array
                    active INTEGER DEFAULT 1,
                    revoked_at TEXT,
                    metadata TEXT  -- JSON object
                );

                CREATE INDEX IF NOT EXISTS idx_consents_active ON consents(active);
                CREATE INDEX IF NOT EXISTS idx_consents_scope ON consents(scope);
                CREATE INDEX IF NOT EXISTS idx_consents_expires ON consents(expires_at);

                -- Access log
                CREATE TABLE IF NOT EXISTS access_log (
                    log_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    blob_id TEXT NOT NULL,
                    access_type TEXT NOT NULL,
                    accessor TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    success INTEGER NOT NULL,
                    details TEXT  -- JSON object
                );

                CREATE INDEX IF NOT EXISTS idx_access_blob ON access_log(blob_id);
                CREATE INDEX IF NOT EXISTS idx_access_time ON access_log(timestamp);
                CREATE INDEX IF NOT EXISTS idx_access_type ON access_log(access_type);
                CREATE INDEX IF NOT EXISTS idx_access_accessor ON access_log(accessor);

                -- Deletion queue for right-to-delete propagation
                CREATE TABLE IF NOT EXISTS deletion_queue (
                    queue_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    consent_id TEXT NOT NULL,
                    requested_at TEXT NOT NULL,
                    completed_at TEXT,
                    blob_count INTEGER,
                    status TEXT DEFAULT 'PENDING',
                    error_message TEXT
                );

                CREATE INDEX IF NOT EXISTS idx_deletion_status ON deletion_queue(status);

                -- Genesis proofs
                CREATE TABLE IF NOT EXISTS genesis_proofs (
                    proof_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    proof_type TEXT NOT NULL,
                    proof_data TEXT NOT NULL,  -- JSON
                    verified INTEGER DEFAULT 0
                );

                -- Tags index for efficient tag queries
                CREATE TABLE IF NOT EXISTS blob_tags (
                    blob_id TEXT NOT NULL,
                    tag TEXT NOT NULL,
                    PRIMARY KEY (blob_id, tag),
                    FOREIGN KEY (blob_id) REFERENCES blobs(blob_id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_tags_tag ON blob_tags(tag);

                -- Record schema version
                INSERT INTO schema_version (version, applied_at)
                VALUES (1, datetime('now'));
            """
            )

        logger.info(f"Database migrated to version {self.SCHEMA_VERSION}")

    def index_blob(self, metadata: BlobMetadata) -> None:
        """
        Index blob metadata.

        Args:
            metadata: Blob metadata to index
        """
        with self._lock:
            with self._transaction() as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO blobs (
                        blob_id, key_id, tier, blob_type, size_bytes,
                        encrypted_size, content_hash, created_at, modified_at,
                        accessed_at, access_count, status, consent_id,
                        ttl_seconds, tags, custom_metadata
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        metadata.blob_id,
                        metadata.key_id,
                        metadata.tier.name,
                        metadata.blob_type.name,
                        metadata.size_bytes,
                        metadata.encrypted_size,
                        metadata.content_hash,
                        metadata.created_at.isoformat(),
                        metadata.modified_at.isoformat(),
                        metadata.accessed_at.isoformat() if metadata.accessed_at else None,
                        metadata.access_count,
                        metadata.status.name,
                        metadata.consent_id,
                        metadata.ttl_seconds,
                        json.dumps(metadata.tags),
                        json.dumps(metadata.custom_metadata),
                    ),
                )

                # Update tags table
                conn.execute("DELETE FROM blob_tags WHERE blob_id = ?", (metadata.blob_id,))
                if metadata.tags:
                    conn.executemany(
                        "INSERT INTO blob_tags (blob_id, tag) VALUES (?, ?)",
                        [(metadata.blob_id, tag) for tag in metadata.tags],
                    )

    def remove_blob_index(self, blob_id: str) -> None:
        """Remove blob from index."""
        with self._lock:
            with self._transaction() as conn:
                conn.execute("DELETE FROM blobs WHERE blob_id = ?", (blob_id,))

    def get_blob(self, blob_id: str) -> Optional[BlobMetadata]:
        """Get blob metadata from index."""
        conn = self._get_connection()
        cursor = conn.execute("SELECT * FROM blobs WHERE blob_id = ?", (blob_id,))
        row = cursor.fetchone()
        return self._row_to_metadata(row) if row else None

    def query_blobs(
        self,
        tier: Optional[EncryptionTier] = None,
        status: Optional[BlobStatus] = None,
        consent_id: Optional[str] = None,
        tags: Optional[List[str]] = None,
        content_hash: Optional[str] = None,
        created_after: Optional[datetime] = None,
        created_before: Optional[datetime] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[BlobMetadata]:
        """
        Query blobs with filters.

        Args:
            tier: Filter by tier
            status: Filter by status
            consent_id: Filter by consent
            tags: Filter by tags (any match)
            content_hash: Filter by content hash
            created_after: Filter by creation time
            created_before: Filter by creation time
            limit: Maximum results
            offset: Result offset

        Returns:
            List of matching blob metadata
        """
        conn = self._get_connection()

        query = "SELECT * FROM blobs WHERE 1=1"
        params: List[Any] = []

        if tier:
            query += " AND tier = ?"
            params.append(tier.name)

        if status:
            query += " AND status = ?"
            params.append(status.name)

        if consent_id:
            query += " AND consent_id = ?"
            params.append(consent_id)

        if content_hash:
            query += " AND content_hash = ?"
            params.append(content_hash)

        if created_after:
            query += " AND created_at > ?"
            params.append(created_after.isoformat())

        if created_before:
            query += " AND created_at < ?"
            params.append(created_before.isoformat())

        if tags:
            placeholders = ",".join("?" * len(tags))
            query += f"""
                AND blob_id IN (
                    SELECT blob_id FROM blob_tags
                    WHERE tag IN ({placeholders})
                )
            """
            params.extend(tags)

        query += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        cursor = conn.execute(query, params)
        return [self._row_to_metadata(row) for row in cursor.fetchall()]

    def record_consent(self, consent: ConsentRecord) -> None:
        """Record a consent grant."""
        with self._lock:
            with self._transaction() as conn:
                conn.execute(
                    """
                    INSERT INTO consents (
                        consent_id, granted_by, granted_at, expires_at,
                        scope, operations, active, revoked_at, metadata
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        consent.consent_id,
                        consent.granted_by,
                        consent.granted_at.isoformat(),
                        consent.expires_at.isoformat() if consent.expires_at else None,
                        consent.scope,
                        json.dumps(consent.operations),
                        1 if consent.active else 0,
                        consent.revoked_at.isoformat() if consent.revoked_at else None,
                        json.dumps(consent.metadata),
                    ),
                )

    def get_consent(self, consent_id: str) -> Optional[ConsentRecord]:
        """Get consent record."""
        conn = self._get_connection()
        cursor = conn.execute("SELECT * FROM consents WHERE consent_id = ?", (consent_id,))
        row = cursor.fetchone()
        return self._row_to_consent(row) if row else None

    def revoke_consent(self, consent_id: str) -> bool:
        """Revoke a consent."""
        with self._lock:
            with self._transaction() as conn:
                cursor = conn.execute(
                    """
                    UPDATE consents
                    SET active = 0, revoked_at = ?
                    WHERE consent_id = ? AND active = 1
                """,
                    (datetime.now().isoformat(), consent_id),
                )
                return cursor.rowcount > 0

    def get_active_consents(
        self,
        scope: Optional[str] = None,
    ) -> List[ConsentRecord]:
        """Get active consents, optionally filtered by scope."""
        conn = self._get_connection()

        query = "SELECT * FROM consents WHERE active = 1"
        params: List[Any] = []

        if scope:
            query += " AND scope = ?"
            params.append(scope)

        # Exclude expired
        query += " AND (expires_at IS NULL OR expires_at > ?)"
        params.append(datetime.now().isoformat())

        cursor = conn.execute(query, params)
        return [self._row_to_consent(row) for row in cursor.fetchall()]

    def log_access(
        self,
        blob_id: str,
        access_type: AccessType,
        accessor: str,
        success: bool,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log an access event."""
        with self._lock:
            with self._transaction() as conn:
                conn.execute(
                    """
                    INSERT INTO access_log (
                        blob_id, access_type, accessor, timestamp, success, details
                    ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                    (
                        blob_id,
                        access_type.name,
                        accessor,
                        datetime.now().isoformat(),
                        1 if success else 0,
                        json.dumps(details or {}),
                    ),
                )

    def get_access_log(
        self,
        blob_id: Optional[str] = None,
        access_type: Optional[AccessType] = None,
        accessor: Optional[str] = None,
        since: Optional[datetime] = None,
        limit: int = 100,
    ) -> List[AccessLogEntry]:
        """Get access log entries."""
        conn = self._get_connection()

        query = "SELECT * FROM access_log WHERE 1=1"
        params: List[Any] = []

        if blob_id:
            query += " AND blob_id = ?"
            params.append(blob_id)

        if access_type:
            query += " AND access_type = ?"
            params.append(access_type.name)

        if accessor:
            query += " AND accessor = ?"
            params.append(accessor)

        if since:
            query += " AND timestamp > ?"
            params.append(since.isoformat())

        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        cursor = conn.execute(query, params)

        entries = []
        for row in cursor.fetchall():
            entries.append(
                AccessLogEntry(
                    log_id=row["log_id"],
                    blob_id=row["blob_id"],
                    access_type=AccessType[row["access_type"]],
                    accessor=row["accessor"],
                    timestamp=datetime.fromisoformat(row["timestamp"]),
                    success=bool(row["success"]),
                    details=json.loads(row["details"]) if row["details"] else {},
                )
            )
        return entries

    def queue_deletion(self, consent_id: str) -> int:
        """
        Queue deletion for all blobs associated with a consent.

        Args:
            consent_id: Consent ID to delete blobs for

        Returns:
            Queue ID for tracking
        """
        with self._lock:
            with self._transaction() as conn:
                # Count affected blobs
                cursor = conn.execute(
                    "SELECT COUNT(*) FROM blobs WHERE consent_id = ?", (consent_id,)
                )
                blob_count = cursor.fetchone()[0]

                # Create queue entry
                cursor = conn.execute(
                    """
                    INSERT INTO deletion_queue (
                        consent_id, requested_at, blob_count, status
                    ) VALUES (?, ?, ?, 'PENDING')
                """,
                    (consent_id, datetime.now().isoformat(), blob_count),
                )

                return cursor.lastrowid

    def get_pending_deletions(self) -> List[Dict[str, Any]]:
        """Get pending deletion requests."""
        conn = self._get_connection()
        cursor = conn.execute(
            """
            SELECT * FROM deletion_queue
            WHERE status = 'PENDING'
            ORDER BY requested_at ASC
        """
        )

        return [dict(row) for row in cursor.fetchall()]

    def complete_deletion(
        self,
        queue_id: int,
        success: bool,
        error_message: Optional[str] = None,
    ) -> None:
        """Mark a deletion request as complete."""
        with self._lock:
            with self._transaction() as conn:
                conn.execute(
                    """
                    UPDATE deletion_queue
                    SET completed_at = ?, status = ?, error_message = ?
                    WHERE queue_id = ?
                """,
                    (
                        datetime.now().isoformat(),
                        "COMPLETED" if success else "FAILED",
                        error_message,
                        queue_id,
                    ),
                )

    def record_genesis_proof(
        self,
        proof_id: str,
        proof_type: str,
        proof_data: Dict[str, Any],
    ) -> None:
        """Record a genesis proof."""
        with self._lock:
            with self._transaction() as conn:
                conn.execute(
                    """
                    INSERT INTO genesis_proofs (
                        proof_id, created_at, proof_type, proof_data
                    ) VALUES (?, ?, ?, ?)
                """,
                    (
                        proof_id,
                        datetime.now().isoformat(),
                        proof_type,
                        json.dumps(proof_data),
                    ),
                )

    def get_genesis_proof(self, proof_id: str) -> Optional[Dict[str, Any]]:
        """Get a genesis proof."""
        conn = self._get_connection()
        cursor = conn.execute("SELECT * FROM genesis_proofs WHERE proof_id = ?", (proof_id,))
        row = cursor.fetchone()
        if row:
            return {
                "proof_id": row["proof_id"],
                "created_at": row["created_at"],
                "proof_type": row["proof_type"],
                "proof_data": json.loads(row["proof_data"]),
                "verified": bool(row["verified"]),
            }
        return None

    def verify_genesis_proof(self, proof_id: str) -> bool:
        """Mark a genesis proof as verified."""
        with self._lock:
            with self._transaction() as conn:
                cursor = conn.execute(
                    """
                    UPDATE genesis_proofs
                    SET verified = 1
                    WHERE proof_id = ?
                """,
                    (proof_id,),
                )
                return cursor.rowcount > 0

    def get_statistics(self) -> Dict[str, Any]:
        """Get index statistics."""
        conn = self._get_connection()

        stats = {}

        # Blob stats
        cursor = conn.execute("SELECT COUNT(*) FROM blobs")
        stats["total_blobs"] = cursor.fetchone()[0]

        cursor = conn.execute(
            """
            SELECT tier, COUNT(*) FROM blobs GROUP BY tier
        """
        )
        stats["blobs_by_tier"] = {row[0]: row[1] for row in cursor.fetchall()}

        cursor = conn.execute(
            """
            SELECT status, COUNT(*) FROM blobs GROUP BY status
        """
        )
        stats["blobs_by_status"] = {row[0]: row[1] for row in cursor.fetchall()}

        cursor = conn.execute("SELECT SUM(size_bytes), SUM(encrypted_size) FROM blobs")
        row = cursor.fetchone()
        stats["total_size_bytes"] = row[0] or 0
        stats["total_encrypted_bytes"] = row[1] or 0

        # Consent stats
        cursor = conn.execute("SELECT COUNT(*) FROM consents WHERE active = 1")
        stats["active_consents"] = cursor.fetchone()[0]

        # Access log stats
        cursor = conn.execute("SELECT COUNT(*) FROM access_log")
        stats["total_access_logs"] = cursor.fetchone()[0]

        return stats

    def _row_to_metadata(self, row: sqlite3.Row) -> BlobMetadata:
        """Convert database row to BlobMetadata."""
        return BlobMetadata(
            blob_id=row["blob_id"],
            key_id=row["key_id"],
            tier=EncryptionTier[row["tier"]],
            blob_type=BlobType[row["blob_type"]],
            size_bytes=row["size_bytes"],
            encrypted_size=row["encrypted_size"],
            content_hash=row["content_hash"],
            created_at=datetime.fromisoformat(row["created_at"]),
            modified_at=datetime.fromisoformat(row["modified_at"]),
            accessed_at=(
                datetime.fromisoformat(row["accessed_at"]) if row["accessed_at"] else None
            ),
            access_count=row["access_count"],
            status=BlobStatus[row["status"]],
            consent_id=row["consent_id"],
            ttl_seconds=row["ttl_seconds"],
            tags=json.loads(row["tags"]) if row["tags"] else [],
            custom_metadata=json.loads(row["custom_metadata"]) if row["custom_metadata"] else {},
        )

    def _row_to_consent(self, row: sqlite3.Row) -> ConsentRecord:
        """Convert database row to ConsentRecord."""
        return ConsentRecord(
            consent_id=row["consent_id"],
            granted_by=row["granted_by"],
            granted_at=datetime.fromisoformat(row["granted_at"]),
            expires_at=(datetime.fromisoformat(row["expires_at"]) if row["expires_at"] else None),
            scope=row["scope"],
            operations=json.loads(row["operations"]),
            active=bool(row["active"]),
            revoked_at=(datetime.fromisoformat(row["revoked_at"]) if row["revoked_at"] else None),
            metadata=json.loads(row["metadata"]) if row["metadata"] else {},
        )

    def close(self) -> None:
        """Close database connection."""
        if hasattr(self._local, "connection") and self._local.connection:
            self._local.connection.close()
            self._local.connection = None
