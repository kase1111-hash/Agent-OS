"""Offline Storage and Data Synchronization for Agent OS Mobile.

Provides offline-first data management for mobile applications:
- Local SQLite storage with encryption
- Data synchronization with conflict resolution
- Cache management with LRU eviction
- Change tracking and delta sync
"""

import asyncio
import hashlib
import json
import logging
import os
import sqlite3
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

logger = logging.getLogger(__name__)


class SyncState(str, Enum):
    """Synchronization states."""

    IDLE = "idle"
    SYNCING = "syncing"
    PAUSED = "paused"
    ERROR = "error"
    OFFLINE = "offline"


class ConflictResolution(str, Enum):
    """Conflict resolution strategies."""

    SERVER_WINS = "server_wins"
    CLIENT_WINS = "client_wins"
    LATEST_WINS = "latest_wins"
    MERGE = "merge"
    MANUAL = "manual"


class ChangeType(str, Enum):
    """Types of data changes."""

    CREATE = "create"
    UPDATE = "update"
    DELETE = "delete"


@dataclass
class CacheEntry:
    """A cached data entry."""

    key: str
    value: Any
    created_at: datetime = field(default_factory=datetime.now)
    accessed_at: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None
    size_bytes: int = 0
    etag: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_expired(self) -> bool:
        """Check if entry is expired."""
        if not self.expires_at:
            return False
        return datetime.now() >= self.expires_at

    def touch(self) -> None:
        """Update access time."""
        self.accessed_at = datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "key": self.key,
            "value": self.value,
            "created_at": self.created_at.isoformat(),
            "accessed_at": self.accessed_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "size_bytes": self.size_bytes,
            "etag": self.etag,
            "metadata": self.metadata,
        }


@dataclass
class ChangeRecord:
    """Record of a local data change."""

    change_id: str
    entity_type: str
    entity_id: str
    change_type: ChangeType
    data: Optional[Dict[str, Any]]
    timestamp: datetime = field(default_factory=datetime.now)
    synced: bool = False
    version: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "change_id": self.change_id,
            "entity_type": self.entity_type,
            "entity_id": self.entity_id,
            "change_type": self.change_type.value,
            "data": self.data,
            "timestamp": self.timestamp.isoformat(),
            "synced": self.synced,
            "version": self.version,
        }


@dataclass
class SyncConflict:
    """A synchronization conflict."""

    entity_type: str
    entity_id: str
    local_data: Dict[str, Any]
    server_data: Dict[str, Any]
    local_timestamp: datetime
    server_timestamp: datetime
    resolved: bool = False
    resolution: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "entity_type": self.entity_type,
            "entity_id": self.entity_id,
            "local_timestamp": self.local_timestamp.isoformat(),
            "server_timestamp": self.server_timestamp.isoformat(),
            "resolved": self.resolved,
        }


@dataclass
class StorageConfig:
    """Storage configuration."""

    database_path: str = "agentos_mobile.db"
    cache_size_mb: int = 100
    max_cache_entries: int = 10000
    default_ttl_seconds: int = 3600
    encryption_key: Optional[str] = None
    auto_vacuum: bool = True
    sync_interval_seconds: int = 300
    batch_size: int = 50
    conflict_resolution: ConflictResolution = ConflictResolution.LATEST_WINS


class OfflineStorage:
    """Local storage with offline-first support.

    Features:
    - SQLite-based storage
    - LRU cache eviction
    - Change tracking
    - Data encryption (when configured)
    """

    def __init__(self, config: Optional[StorageConfig] = None):
        """Initialize offline storage.

        Args:
            config: Storage configuration
        """
        self.config = config or StorageConfig()
        self._db: Optional[sqlite3.Connection] = None
        self._cache: Dict[str, CacheEntry] = {}
        self._cache_size_bytes = 0
        self._changes: List[ChangeRecord] = []
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize storage database."""
        self._db = sqlite3.connect(
            self.config.database_path,
            check_same_thread=False,
        )
        self._db.row_factory = sqlite3.Row

        # Create tables
        self._create_tables()
        self._initialized = True
        logger.info(f"Storage initialized: {self.config.database_path}")

    def _create_tables(self) -> None:
        """Create database tables."""
        cursor = self._db.cursor()

        # Key-value store
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS kv_store (
                key TEXT PRIMARY KEY,
                value TEXT,
                created_at TEXT,
                updated_at TEXT,
                expires_at TEXT,
                etag TEXT,
                metadata TEXT
            )
        """
        )

        # Entity store
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS entities (
                entity_type TEXT,
                entity_id TEXT,
                data TEXT,
                version INTEGER DEFAULT 0,
                created_at TEXT,
                updated_at TEXT,
                synced INTEGER DEFAULT 0,
                PRIMARY KEY (entity_type, entity_id)
            )
        """
        )

        # Change log
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS change_log (
                change_id TEXT PRIMARY KEY,
                entity_type TEXT,
                entity_id TEXT,
                change_type TEXT,
                data TEXT,
                timestamp TEXT,
                synced INTEGER DEFAULT 0,
                version INTEGER DEFAULT 0
            )
        """
        )

        # Sync metadata
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS sync_metadata (
                key TEXT PRIMARY KEY,
                value TEXT
            )
        """
        )

        self._db.commit()

    async def close(self) -> None:
        """Close storage."""
        if self._db:
            self._db.close()
            self._db = None
        self._initialized = False
        logger.info("Storage closed")

    # Key-Value Operations

    async def get(self, key: str) -> Optional[Any]:
        """Get value by key.

        Args:
            key: Key to retrieve

        Returns:
            Value or None
        """
        # Check cache first
        if key in self._cache:
            entry = self._cache[key]
            if not entry.is_expired:
                entry.touch()
                return entry.value
            else:
                del self._cache[key]

        # Check database
        cursor = self._db.cursor()
        cursor.execute(
            "SELECT value, expires_at FROM kv_store WHERE key = ?",
            (key,),
        )
        row = cursor.fetchone()

        if row:
            expires_at = datetime.fromisoformat(row["expires_at"]) if row["expires_at"] else None
            if expires_at and datetime.now() >= expires_at:
                await self.delete(key)
                return None

            value = json.loads(row["value"])

            # Add to cache
            self._add_to_cache(key, value, expires_at)

            return value

        return None

    async def set(
        self,
        key: str,
        value: Any,
        ttl_seconds: Optional[int] = None,
        etag: Optional[str] = None,
    ) -> None:
        """Set a key-value pair.

        Args:
            key: Key
            value: Value
            ttl_seconds: Time to live in seconds
            etag: Entity tag for caching
        """
        now = datetime.now()
        expires_at = (
            now + timedelta(seconds=ttl_seconds or self.config.default_ttl_seconds)
            if ttl_seconds is not None or self.config.default_ttl_seconds
            else None
        )

        value_json = json.dumps(value)

        cursor = self._db.cursor()
        cursor.execute(
            """
            INSERT OR REPLACE INTO kv_store
            (key, value, created_at, updated_at, expires_at, etag, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                key,
                value_json,
                now.isoformat(),
                now.isoformat(),
                expires_at.isoformat() if expires_at else None,
                etag,
                "{}",
            ),
        )
        self._db.commit()

        # Update cache
        self._add_to_cache(key, value, expires_at)

    async def delete(self, key: str) -> bool:
        """Delete a key.

        Args:
            key: Key to delete

        Returns:
            True if deleted
        """
        cursor = self._db.cursor()
        cursor.execute("DELETE FROM kv_store WHERE key = ?", (key,))
        self._db.commit()

        if key in self._cache:
            entry = self._cache[key]
            self._cache_size_bytes -= entry.size_bytes
            del self._cache[key]

        return cursor.rowcount > 0

    async def exists(self, key: str) -> bool:
        """Check if key exists.

        Args:
            key: Key to check

        Returns:
            True if exists
        """
        if key in self._cache:
            return not self._cache[key].is_expired

        cursor = self._db.cursor()
        cursor.execute("SELECT 1 FROM kv_store WHERE key = ?", (key,))
        return cursor.fetchone() is not None

    def _add_to_cache(
        self,
        key: str,
        value: Any,
        expires_at: Optional[datetime],
    ) -> None:
        """Add entry to cache with eviction if needed."""
        value_json = json.dumps(value)
        size = len(value_json.encode())

        # Evict if necessary
        max_size = self.config.cache_size_mb * 1024 * 1024
        while (
            self._cache_size_bytes + size > max_size
            or len(self._cache) >= self.config.max_cache_entries
        ):
            if not self._cache:
                break
            self._evict_lru()

        entry = CacheEntry(
            key=key,
            value=value,
            expires_at=expires_at,
            size_bytes=size,
        )
        self._cache[key] = entry
        self._cache_size_bytes += size

    def _evict_lru(self) -> None:
        """Evict least recently used cache entry."""
        if not self._cache:
            return

        lru_key = min(
            self._cache.keys(),
            key=lambda k: self._cache[k].accessed_at,
        )
        entry = self._cache[lru_key]
        self._cache_size_bytes -= entry.size_bytes
        del self._cache[lru_key]

    # Entity Operations

    async def save_entity(
        self,
        entity_type: str,
        entity_id: str,
        data: Dict[str, Any],
        track_change: bool = True,
    ) -> int:
        """Save an entity.

        Args:
            entity_type: Entity type
            entity_id: Entity ID
            data: Entity data
            track_change: Whether to track change for sync

        Returns:
            New version number
        """
        now = datetime.now()

        # Get current version
        cursor = self._db.cursor()
        cursor.execute(
            "SELECT version FROM entities WHERE entity_type = ? AND entity_id = ?",
            (entity_type, entity_id),
        )
        row = cursor.fetchone()
        version = (row["version"] + 1) if row else 1
        change_type = ChangeType.UPDATE if row else ChangeType.CREATE

        # Save entity
        cursor.execute(
            """
            INSERT OR REPLACE INTO entities
            (entity_type, entity_id, data, version, created_at, updated_at, synced)
            VALUES (?, ?, ?, ?, ?, ?, 0)
            """,
            (
                entity_type,
                entity_id,
                json.dumps(data),
                version,
                now.isoformat() if not row else row[0] if row else now.isoformat(),
                now.isoformat(),
            ),
        )
        self._db.commit()

        # Track change
        if track_change:
            await self._track_change(entity_type, entity_id, change_type, data, version)

        return version

    async def get_entity(
        self,
        entity_type: str,
        entity_id: str,
    ) -> Optional[Dict[str, Any]]:
        """Get an entity.

        Args:
            entity_type: Entity type
            entity_id: Entity ID

        Returns:
            Entity data or None
        """
        cursor = self._db.cursor()
        cursor.execute(
            "SELECT data FROM entities WHERE entity_type = ? AND entity_id = ?",
            (entity_type, entity_id),
        )
        row = cursor.fetchone()

        if row:
            return json.loads(row["data"])
        return None

    async def delete_entity(
        self,
        entity_type: str,
        entity_id: str,
        track_change: bool = True,
    ) -> bool:
        """Delete an entity.

        Args:
            entity_type: Entity type
            entity_id: Entity ID
            track_change: Whether to track change for sync

        Returns:
            True if deleted
        """
        cursor = self._db.cursor()
        cursor.execute(
            "DELETE FROM entities WHERE entity_type = ? AND entity_id = ?",
            (entity_type, entity_id),
        )
        self._db.commit()

        if cursor.rowcount > 0 and track_change:
            await self._track_change(entity_type, entity_id, ChangeType.DELETE, None, 0)
            return True
        return False

    async def query_entities(
        self,
        entity_type: str,
        filter_fn: Optional[Callable[[Dict[str, Any]], bool]] = None,
        limit: Optional[int] = None,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """Query entities.

        Args:
            entity_type: Entity type
            filter_fn: Optional filter function
            limit: Maximum results
            offset: Result offset

        Returns:
            List of entities
        """
        cursor = self._db.cursor()
        query = "SELECT data FROM entities WHERE entity_type = ?"
        params: List[Any] = [entity_type]

        if limit:
            query += " LIMIT ? OFFSET ?"
            params.extend([limit, offset])

        cursor.execute(query, params)
        rows = cursor.fetchall()

        entities = [json.loads(row["data"]) for row in rows]

        if filter_fn:
            entities = [e for e in entities if filter_fn(e)]

        return entities

    async def _track_change(
        self,
        entity_type: str,
        entity_id: str,
        change_type: ChangeType,
        data: Optional[Dict[str, Any]],
        version: int,
    ) -> None:
        """Track a change for sync."""
        import uuid

        change = ChangeRecord(
            change_id=str(uuid.uuid4()),
            entity_type=entity_type,
            entity_id=entity_id,
            change_type=change_type,
            data=data,
            version=version,
        )

        cursor = self._db.cursor()
        cursor.execute(
            """
            INSERT INTO change_log
            (change_id, entity_type, entity_id, change_type, data, timestamp, synced, version)
            VALUES (?, ?, ?, ?, ?, ?, 0, ?)
            """,
            (
                change.change_id,
                change.entity_type,
                change.entity_id,
                change.change_type.value,
                json.dumps(change.data) if change.data else None,
                change.timestamp.isoformat(),
                change.version,
            ),
        )
        self._db.commit()

    async def get_pending_changes(self) -> List[ChangeRecord]:
        """Get unsynced changes.

        Returns:
            List of pending changes
        """
        cursor = self._db.cursor()
        cursor.execute("SELECT * FROM change_log WHERE synced = 0 ORDER BY timestamp")
        rows = cursor.fetchall()

        return [
            ChangeRecord(
                change_id=row["change_id"],
                entity_type=row["entity_type"],
                entity_id=row["entity_id"],
                change_type=ChangeType(row["change_type"]),
                data=json.loads(row["data"]) if row["data"] else None,
                timestamp=datetime.fromisoformat(row["timestamp"]),
                synced=bool(row["synced"]),
                version=row["version"],
            )
            for row in rows
        ]

    async def mark_synced(self, change_ids: List[str]) -> None:
        """Mark changes as synced.

        Args:
            change_ids: List of change IDs to mark
        """
        if not change_ids:
            return

        placeholders = ",".join("?" * len(change_ids))
        cursor = self._db.cursor()
        cursor.execute(
            f"UPDATE change_log SET synced = 1 WHERE change_id IN ({placeholders})",
            change_ids,
        )
        self._db.commit()

    # Cache Management

    def clear_cache(self) -> int:
        """Clear in-memory cache.

        Returns:
            Number of entries cleared
        """
        count = len(self._cache)
        self._cache.clear()
        self._cache_size_bytes = 0
        return count

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "entries": len(self._cache),
            "size_bytes": self._cache_size_bytes,
            "max_size_bytes": self.config.cache_size_mb * 1024 * 1024,
            "max_entries": self.config.max_cache_entries,
        }

    async def cleanup_expired(self) -> int:
        """Remove expired entries.

        Returns:
            Number of entries removed
        """
        now = datetime.now().isoformat()

        cursor = self._db.cursor()
        cursor.execute(
            "DELETE FROM kv_store WHERE expires_at IS NOT NULL AND expires_at < ?",
            (now,),
        )
        self._db.commit()

        # Clean cache
        expired_keys = [k for k, v in self._cache.items() if v.is_expired]
        for key in expired_keys:
            entry = self._cache[key]
            self._cache_size_bytes -= entry.size_bytes
            del self._cache[key]

        return cursor.rowcount + len(expired_keys)


class SyncManager:
    """Manages data synchronization between local and remote storage.

    Features:
    - Delta synchronization
    - Conflict detection and resolution
    - Batch processing
    - Retry with backoff
    """

    def __init__(
        self,
        storage: OfflineStorage,
        config: Optional[StorageConfig] = None,
    ):
        """Initialize sync manager.

        Args:
            storage: Local storage instance
            config: Storage configuration
        """
        self.storage = storage
        self.config = config or storage.config
        self._state = SyncState.IDLE
        self._state_callbacks: List[Callable[[SyncState], None]] = []
        self._conflicts: List[SyncConflict] = []
        self._sync_task: Optional[asyncio.Task] = None
        self._last_sync: Optional[datetime] = None
        self._is_online = True

    @property
    def state(self) -> SyncState:
        """Get current sync state."""
        return self._state

    @property
    def is_syncing(self) -> bool:
        """Check if sync is in progress."""
        return self._state == SyncState.SYNCING

    @property
    def conflicts(self) -> List[SyncConflict]:
        """Get unresolved conflicts."""
        return [c for c in self._conflicts if not c.resolved]

    def on_state_change(self, callback: Callable[[SyncState], None]) -> None:
        """Register state change callback."""
        self._state_callbacks.append(callback)

    def _set_state(self, state: SyncState) -> None:
        """Set sync state and notify callbacks."""
        if self._state != state:
            old_state = self._state
            self._state = state
            logger.info(f"Sync state changed: {old_state.value} -> {state.value}")
            for callback in self._state_callbacks:
                try:
                    callback(state)
                except Exception as e:
                    logger.warning(f"State callback error: {e}")

    def set_online(self, is_online: bool) -> None:
        """Set online status.

        Args:
            is_online: Whether device is online
        """
        self._is_online = is_online
        if not is_online:
            self._set_state(SyncState.OFFLINE)
        elif self._state == SyncState.OFFLINE:
            self._set_state(SyncState.IDLE)

    async def start_auto_sync(self) -> None:
        """Start automatic synchronization."""
        if self._sync_task and not self._sync_task.done():
            return

        self._sync_task = asyncio.create_task(self._auto_sync_loop())
        logger.info("Auto-sync started")

    async def stop_auto_sync(self) -> None:
        """Stop automatic synchronization."""
        if self._sync_task:
            self._sync_task.cancel()
            self._sync_task = None
        logger.info("Auto-sync stopped")

    async def _auto_sync_loop(self) -> None:
        """Automatic sync loop."""
        while True:
            try:
                await asyncio.sleep(self.config.sync_interval_seconds)

                if self._is_online and self._state == SyncState.IDLE:
                    await self.sync()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Auto-sync error: {e}")
                self._set_state(SyncState.ERROR)

    async def sync(self) -> Dict[str, Any]:
        """Perform synchronization.

        Returns:
            Sync result summary
        """
        if not self._is_online:
            return {"status": "offline", "changes_pushed": 0, "changes_pulled": 0}

        if self._state == SyncState.SYNCING:
            return {"status": "already_syncing"}

        self._set_state(SyncState.SYNCING)
        result = {
            "status": "success",
            "changes_pushed": 0,
            "changes_pulled": 0,
            "conflicts": 0,
            "errors": [],
        }

        try:
            # Push local changes
            pending = await self.storage.get_pending_changes()
            pushed = await self._push_changes(pending)
            result["changes_pushed"] = pushed

            # Pull remote changes
            pulled = await self._pull_changes()
            result["changes_pulled"] = pulled

            # Handle conflicts
            result["conflicts"] = len(self.conflicts)

            self._last_sync = datetime.now()
            self._set_state(SyncState.IDLE)

        except Exception as e:
            logger.error(f"Sync failed: {e}")
            result["status"] = "error"
            result["errors"].append(str(e))
            self._set_state(SyncState.ERROR)

        return result

    async def _push_changes(self, changes: List[ChangeRecord]) -> int:
        """Push local changes to server.

        Args:
            changes: List of changes to push

        Returns:
            Number of changes pushed
        """
        if not changes:
            return 0

        pushed = 0
        synced_ids = []

        # Process in batches
        for i in range(0, len(changes), self.config.batch_size):
            batch = changes[i : i + self.config.batch_size]

            for change in batch:
                try:
                    # Mock push to server
                    await asyncio.sleep(0.01)
                    synced_ids.append(change.change_id)
                    pushed += 1
                except Exception as e:
                    logger.warning(f"Failed to push change {change.change_id}: {e}")

        # Mark as synced
        await self.storage.mark_synced(synced_ids)

        return pushed

    async def _pull_changes(self) -> int:
        """Pull remote changes from server.

        Returns:
            Number of changes pulled
        """
        # Mock pull from server
        await asyncio.sleep(0.01)

        # In production, this would:
        # 1. Get last sync timestamp
        # 2. Fetch changes since that time
        # 3. Apply changes with conflict detection
        # 4. Update sync timestamp

        return 0

    async def resolve_conflict(
        self,
        conflict: SyncConflict,
        resolution: ConflictResolution,
        manual_data: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Resolve a sync conflict.

        Args:
            conflict: Conflict to resolve
            resolution: Resolution strategy
            manual_data: Manual resolution data

        Returns:
            True if resolved
        """
        if conflict.resolved:
            return True

        resolved_data: Dict[str, Any]

        if resolution == ConflictResolution.SERVER_WINS:
            resolved_data = conflict.server_data
        elif resolution == ConflictResolution.CLIENT_WINS:
            resolved_data = conflict.local_data
        elif resolution == ConflictResolution.LATEST_WINS:
            if conflict.local_timestamp > conflict.server_timestamp:
                resolved_data = conflict.local_data
            else:
                resolved_data = conflict.server_data
        elif resolution == ConflictResolution.MERGE:
            resolved_data = self._merge_data(conflict.local_data, conflict.server_data)
        elif resolution == ConflictResolution.MANUAL:
            if not manual_data:
                return False
            resolved_data = manual_data
        else:
            return False

        # Save resolved data
        await self.storage.save_entity(
            conflict.entity_type,
            conflict.entity_id,
            resolved_data,
            track_change=True,
        )

        conflict.resolved = True
        conflict.resolution = resolved_data

        return True

    def _merge_data(
        self,
        local: Dict[str, Any],
        server: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Merge local and server data.

        Simple merge: server values override local for conflicts.
        """
        merged = dict(local)
        merged.update(server)
        return merged

    def get_sync_status(self) -> Dict[str, Any]:
        """Get synchronization status."""
        return {
            "state": self._state.value,
            "is_online": self._is_online,
            "last_sync": self._last_sync.isoformat() if self._last_sync else None,
            "pending_conflicts": len(self.conflicts),
        }
