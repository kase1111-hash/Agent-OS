"""
Memory Management API Routes

Provides endpoints for viewing and managing agent memory.
Each user has their own isolated memory vault.
"""

import json
import logging
import sqlite3
import threading
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Cookie, HTTPException, Query, Request
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)
router = APIRouter()

# Try to import real Seshat memory agent
try:
    from src.agents.seshat import create_seshat_agent, SeshatAgent
    from src.agents.seshat.retrieval import ContextType
    REAL_MEMORY_AVAILABLE = True
except ImportError:
    REAL_MEMORY_AVAILABLE = False
    logger.warning("Real Seshat memory agent not available, using mock data")


# =============================================================================
# Models
# =============================================================================


class MemoryType(str, Enum):
    """Type of memory."""

    EPHEMERAL = "ephemeral"
    WORKING = "working"
    LONG_TERM = "long_term"
    SEMANTIC = "semantic"


class MemoryEntry(BaseModel):
    """A memory entry."""

    id: str
    user_id: str = "default"
    content: str
    memory_type: MemoryType
    source: str = "user"  # user, agent, system
    tags: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    accessed_at: datetime = Field(default_factory=datetime.utcnow)
    access_count: int = 0
    relevance_score: float = 1.0
    consent_given: bool = True
    retention_days: Optional[int] = None


class MemoryCreate(BaseModel):
    """Request to create a memory."""

    content: str
    memory_type: MemoryType = MemoryType.WORKING
    tags: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    consent_given: bool = True
    retention_days: Optional[int] = None


class MemoryUpdate(BaseModel):
    """Request to update a memory."""

    content: Optional[str] = None
    tags: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None


class SearchResult(BaseModel):
    """A search result."""

    entry: MemoryEntry
    similarity_score: float
    highlights: List[str] = Field(default_factory=list)


class MemoryStats(BaseModel):
    """Memory statistics."""

    total_entries: int
    entries_by_type: Dict[str, int]
    total_size_bytes: int
    oldest_entry: Optional[datetime] = None
    newest_entry: Optional[datetime] = None
    consent_rate: float = 100.0


# =============================================================================
# Memory Store (Real + Mock)
# =============================================================================


class MemoryStore:
    """
    Memory store with SQLite backend and user isolation.

    Each user has their own memory vault - memories are stored with user_id.
    """

    def __init__(self, db_path: Optional[Path] = None):
        self._db_path = db_path
        self._lock = threading.Lock()
        self._connection: Optional[sqlite3.Connection] = None
        self._initialized = False

    def initialize(self) -> bool:
        """Initialize the memory store."""
        try:
            if self._db_path:
                self._db_path.parent.mkdir(parents=True, exist_ok=True)

            db_str = str(self._db_path) if self._db_path else ":memory:"
            self._connection = sqlite3.connect(db_str, check_same_thread=False)
            self._connection.row_factory = sqlite3.Row

            self._create_tables()
            self._initialized = True
            logger.info(f"Memory store initialized: {db_str}")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize memory store: {e}")
            return False

    def close(self) -> None:
        """Close the store."""
        if self._connection:
            self._connection.close()
            self._connection = None
        self._initialized = False

    def _create_tables(self) -> None:
        """Create database tables."""
        cursor = self._connection.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                content TEXT NOT NULL,
                memory_type TEXT NOT NULL,
                source TEXT DEFAULT 'user',
                tags_json TEXT,
                metadata_json TEXT,
                created_at TEXT NOT NULL,
                accessed_at TEXT NOT NULL,
                access_count INTEGER DEFAULT 0,
                relevance_score REAL DEFAULT 1.0,
                consent_given INTEGER DEFAULT 1,
                retention_days INTEGER
            )
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_memories_user_id
            ON memories(user_id)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_memories_user_type
            ON memories(user_id, memory_type)
        """)

        self._connection.commit()

    def _row_to_entry(self, row: sqlite3.Row) -> MemoryEntry:
        """Convert database row to MemoryEntry."""
        return MemoryEntry(
            id=row["id"],
            user_id=row["user_id"],
            content=row["content"],
            memory_type=MemoryType(row["memory_type"]),
            source=row["source"] or "user",
            tags=json.loads(row["tags_json"]) if row["tags_json"] else [],
            metadata=json.loads(row["metadata_json"]) if row["metadata_json"] else {},
            created_at=datetime.fromisoformat(row["created_at"]),
            accessed_at=datetime.fromisoformat(row["accessed_at"]),
            access_count=row["access_count"] or 0,
            relevance_score=row["relevance_score"] or 1.0,
            consent_given=bool(row["consent_given"]),
            retention_days=row["retention_days"],
        )

    def get_all(self, user_id: str) -> List[MemoryEntry]:
        """Get all memory entries for a user."""
        if not self._initialized:
            return []

        with self._lock:
            cursor = self._connection.cursor()
            cursor.execute(
                "SELECT * FROM memories WHERE user_id = ? ORDER BY created_at DESC",
                (user_id,)
            )
            return [self._row_to_entry(row) for row in cursor.fetchall()]

    def get(self, entry_id: str, user_id: str) -> Optional[MemoryEntry]:
        """Get a memory entry by ID (must belong to user)."""
        if not self._initialized:
            return None

        with self._lock:
            cursor = self._connection.cursor()
            cursor.execute(
                "SELECT * FROM memories WHERE id = ? AND user_id = ?",
                (entry_id, user_id)
            )
            row = cursor.fetchone()
            if not row:
                return None

            # Update access stats
            now = datetime.utcnow().isoformat()
            cursor.execute(
                "UPDATE memories SET accessed_at = ?, access_count = access_count + 1 WHERE id = ?",
                (now, entry_id)
            )
            self._connection.commit()

            return self._row_to_entry(row)

    def create(self, request: MemoryCreate, user_id: str) -> MemoryEntry:
        """Create a new memory entry for a user."""
        import uuid

        if not self._initialized:
            raise RuntimeError("Memory store not initialized")

        entry = MemoryEntry(
            id=str(uuid.uuid4()),
            user_id=user_id,
            content=request.content,
            memory_type=request.memory_type,
            tags=request.tags,
            metadata=request.metadata,
            consent_given=request.consent_given,
            retention_days=request.retention_days,
        )

        with self._lock:
            cursor = self._connection.cursor()
            cursor.execute("""
                INSERT INTO memories (
                    id, user_id, content, memory_type, source, tags_json, metadata_json,
                    created_at, accessed_at, access_count, relevance_score,
                    consent_given, retention_days
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                entry.id,
                entry.user_id,
                entry.content,
                entry.memory_type.value,
                entry.source,
                json.dumps(entry.tags),
                json.dumps(entry.metadata),
                entry.created_at.isoformat(),
                entry.accessed_at.isoformat(),
                entry.access_count,
                entry.relevance_score,
                1 if entry.consent_given else 0,
                entry.retention_days,
            ))
            self._connection.commit()

        return entry

    def update(self, entry_id: str, request: MemoryUpdate, user_id: str) -> Optional[MemoryEntry]:
        """Update a memory entry (must belong to user)."""
        if not self._initialized:
            return None

        with self._lock:
            cursor = self._connection.cursor()

            # Check ownership
            cursor.execute(
                "SELECT * FROM memories WHERE id = ? AND user_id = ?",
                (entry_id, user_id)
            )
            row = cursor.fetchone()
            if not row:
                return None

            # Build update
            updates = []
            params = []

            if request.content is not None:
                updates.append("content = ?")
                params.append(request.content)
            if request.tags is not None:
                updates.append("tags_json = ?")
                params.append(json.dumps(request.tags))
            if request.metadata is not None:
                # Merge with existing metadata
                existing_metadata = json.loads(row["metadata_json"]) if row["metadata_json"] else {}
                existing_metadata.update(request.metadata)
                updates.append("metadata_json = ?")
                params.append(json.dumps(existing_metadata))

            if not updates:
                return self._row_to_entry(row)

            params.append(entry_id)
            cursor.execute(
                f"UPDATE memories SET {', '.join(updates)} WHERE id = ?",
                params
            )
            self._connection.commit()

            # Fetch updated entry
            cursor.execute("SELECT * FROM memories WHERE id = ?", (entry_id,))
            return self._row_to_entry(cursor.fetchone())

    def delete(self, entry_id: str, user_id: str) -> bool:
        """Delete a memory entry (must belong to user)."""
        if not self._initialized:
            return False

        with self._lock:
            cursor = self._connection.cursor()
            cursor.execute(
                "DELETE FROM memories WHERE id = ? AND user_id = ?",
                (entry_id, user_id)
            )
            self._connection.commit()
            return cursor.rowcount > 0

    def delete_all(self, user_id: str, memory_type: Optional[MemoryType] = None) -> int:
        """Delete all memories for a user (optionally filtered by type)."""
        if not self._initialized:
            return 0

        with self._lock:
            cursor = self._connection.cursor()
            if memory_type:
                cursor.execute(
                    "DELETE FROM memories WHERE user_id = ? AND memory_type = ?",
                    (user_id, memory_type.value)
                )
            else:
                cursor.execute(
                    "DELETE FROM memories WHERE user_id = ?",
                    (user_id,)
                )
            self._connection.commit()
            return cursor.rowcount

    def search(
        self,
        query: str,
        user_id: str,
        memory_type: Optional[MemoryType] = None,
        tags: Optional[List[str]] = None,
        limit: int = 10,
    ) -> List[SearchResult]:
        """Search memory entries for a user."""
        entries = self.get_all(user_id)

        query_lower = query.lower()
        results = []

        for entry in entries:
            # Filter by type
            if memory_type and entry.memory_type != memory_type:
                continue

            # Filter by tags
            if tags and not any(t in entry.tags for t in tags):
                continue

            # Calculate similarity score (simple text matching)
            content_lower = entry.content.lower()
            score = 0.0

            if query_lower in content_lower:
                score = 1.0
            else:
                # Word overlap
                query_words = set(query_lower.split())
                content_words = set(content_lower.split())
                overlap = len(query_words & content_words)
                if overlap > 0:
                    score = overlap / len(query_words)

            if score > 0:
                # Find highlights
                highlights = []
                for word in query_lower.split():
                    if word in content_lower:
                        idx = content_lower.find(word)
                        start = max(0, idx - 20)
                        end = min(len(entry.content), idx + len(word) + 20)
                        highlight = entry.content[start:end]
                        if start > 0:
                            highlight = "..." + highlight
                        if end < len(entry.content):
                            highlight = highlight + "..."
                        highlights.append(highlight)

                results.append(
                    SearchResult(
                        entry=entry,
                        similarity_score=score,
                        highlights=highlights[:3],
                    )
                )

        results.sort(key=lambda r: r.similarity_score, reverse=True)
        return results[:limit]

    def get_stats(self, user_id: str) -> MemoryStats:
        """Get memory statistics for a user."""
        if not self._initialized:
            return MemoryStats(
                total_entries=0,
                entries_by_type={},
                total_size_bytes=0,
            )

        with self._lock:
            cursor = self._connection.cursor()

            # Count by type
            cursor.execute("""
                SELECT memory_type, COUNT(*) FROM memories
                WHERE user_id = ? GROUP BY memory_type
            """, (user_id,))
            type_counts = {row[0]: row[1] for row in cursor.fetchall()}

            # Total size and consent
            cursor.execute("""
                SELECT COUNT(*), SUM(LENGTH(content)), SUM(consent_given),
                       MIN(created_at), MAX(created_at)
                FROM memories WHERE user_id = ?
            """, (user_id,))
            row = cursor.fetchone()

            total = row[0] or 0
            total_size = row[1] or 0
            consent_count = row[2] or 0
            oldest = datetime.fromisoformat(row[3]) if row[3] else None
            newest = datetime.fromisoformat(row[4]) if row[4] else None

            return MemoryStats(
                total_entries=total,
                entries_by_type=type_counts,
                total_size_bytes=total_size,
                oldest_entry=oldest,
                newest_entry=newest,
                consent_rate=(consent_count / total * 100) if total > 0 else 100.0,
            )


# Global store instance
_store: Optional[MemoryStore] = None


def get_store() -> MemoryStore:
    """Get the memory store."""
    global _store
    if _store is None:
        from ..config import get_config
        config = get_config()
        db_path = config.data_dir / "memory.db"
        _store = MemoryStore(db_path=db_path)
        _store.initialize()
    return _store


# =============================================================================
# Authentication Helper
# =============================================================================


def get_current_user_id(request: Request, session_token: Optional[str] = None) -> str:
    """
    Get the current user ID from the session.

    Returns the authenticated user's ID.
    Raises HTTPException 401 if not authenticated.
    """
    try:
        from ..auth import get_user_store

        # Get token from cookie or header
        token = session_token
        if not token:
            auth_header = request.headers.get("authorization")
            if auth_header and auth_header.startswith("Bearer "):
                token = auth_header[7:]

        if token:
            store = get_user_store()
            user = store.validate_session(token)
            if user:
                return user.user_id

    except Exception as e:
        logger.debug(f"Auth check failed: {e}")

    raise HTTPException(status_code=401, detail="Authentication required")


# =============================================================================
# Endpoints
# =============================================================================


@router.get("/", response_model=List[MemoryEntry])
async def list_memories(
    request: Request,
    memory_type: Optional[MemoryType] = None,
    tag: Optional[str] = None,
    limit: int = Query(default=50, le=100),
    session_token: Optional[str] = Cookie(None),
) -> List[MemoryEntry]:
    """List memory entries for the current user with optional filtering."""
    user_id = get_current_user_id(request, session_token)
    store = get_store()
    entries = store.get_all(user_id)

    if memory_type:
        entries = [e for e in entries if e.memory_type == memory_type]

    if tag:
        entries = [e for e in entries if tag in e.tags]

    # Sort by created_at descending
    entries.sort(key=lambda e: e.created_at, reverse=True)

    return entries[:limit]


@router.get("/stats", response_model=MemoryStats)
async def get_memory_stats(
    request: Request,
    session_token: Optional[str] = Cookie(None),
) -> MemoryStats:
    """Get memory statistics for the current user."""
    user_id = get_current_user_id(request, session_token)
    store = get_store()
    return store.get_stats(user_id)


@router.get("/search", response_model=List[SearchResult])
async def search_memories(
    request: Request,
    query: str,
    memory_type: Optional[MemoryType] = None,
    tags: Optional[str] = None,
    limit: int = Query(default=10, le=50),
    session_token: Optional[str] = Cookie(None),
) -> List[SearchResult]:
    """Search memory entries for the current user."""
    user_id = get_current_user_id(request, session_token)
    store = get_store()

    tag_list = tags.split(",") if tags else None

    # Log intent
    try:
        from ..intent_log import IntentType, log_user_intent
        log_user_intent(
            user_id=user_id,
            intent_type=IntentType.MEMORY_SEARCH,
            description=f"Searched memories: {query}",
            details={"query": query, "memory_type": memory_type.value if memory_type else None},
        )
    except Exception as e:
        logger.debug(f"Failed to log intent: {e}")

    return store.search(
        query=query,
        user_id=user_id,
        memory_type=memory_type,
        tags=tag_list,
        limit=limit,
    )


@router.get("/export")
async def export_memories(
    request: Request,
    memory_type: Optional[MemoryType] = None,
    session_token: Optional[str] = Cookie(None),
) -> Dict[str, Any]:
    """Export memories as JSON for the current user."""
    user_id = get_current_user_id(request, session_token)
    store = get_store()
    entries = store.get_all(user_id)

    if memory_type:
        entries = [e for e in entries if e.memory_type == memory_type]

    return {
        "user_id": user_id,
        "exported_at": datetime.utcnow().isoformat(),
        "count": len(entries),
        "entries": [e.model_dump() for e in entries],
    }


@router.get("/{entry_id}", response_model=MemoryEntry)
async def get_memory(
    entry_id: str,
    request: Request,
    session_token: Optional[str] = Cookie(None),
) -> MemoryEntry:
    """Get a specific memory entry (must belong to current user)."""
    user_id = get_current_user_id(request, session_token)
    store = get_store()
    entry = store.get(entry_id, user_id)

    if not entry:
        raise HTTPException(status_code=404, detail=f"Memory not found: {entry_id}")

    return entry


@router.post("/", response_model=MemoryEntry)
async def create_memory(
    request: Request,
    body: MemoryCreate,
    session_token: Optional[str] = Cookie(None),
) -> MemoryEntry:
    """Create a new memory entry for the current user."""
    if not body.consent_given:
        raise HTTPException(
            status_code=403,
            detail="Consent is required to store memory",
        )

    user_id = get_current_user_id(request, session_token)
    store = get_store()
    entry = store.create(body, user_id)

    # Log intent
    try:
        from ..intent_log import IntentType, log_user_intent
        log_user_intent(
            user_id=user_id,
            intent_type=IntentType.MEMORY_CREATE,
            description=f"Created {body.memory_type.value} memory",
            details={"memory_id": entry.id, "memory_type": body.memory_type.value, "tags": body.tags},
            related_entity_type="memory",
            related_entity_id=entry.id,
        )
    except Exception as e:
        logger.debug(f"Failed to log intent: {e}")

    return entry


@router.put("/{entry_id}", response_model=MemoryEntry)
async def update_memory(
    entry_id: str,
    request: Request,
    body: MemoryUpdate,
    session_token: Optional[str] = Cookie(None),
) -> MemoryEntry:
    """Update a memory entry (must belong to current user)."""
    user_id = get_current_user_id(request, session_token)
    store = get_store()
    entry = store.update(entry_id, body, user_id)

    if not entry:
        raise HTTPException(status_code=404, detail=f"Memory not found: {entry_id}")

    return entry


@router.delete("/{entry_id}")
async def delete_memory(
    entry_id: str,
    request: Request,
    session_token: Optional[str] = Cookie(None),
) -> Dict[str, str]:
    """Delete a memory entry (must belong to current user)."""
    user_id = get_current_user_id(request, session_token)
    store = get_store()

    if store.delete(entry_id, user_id):
        # Log intent
        try:
            from ..intent_log import IntentType, log_user_intent
            log_user_intent(
                user_id=user_id,
                intent_type=IntentType.MEMORY_DELETE,
                description=f"Deleted memory: {entry_id}",
                details={"memory_id": entry_id},
                related_entity_type="memory",
                related_entity_id=entry_id,
            )
        except Exception as e:
            logger.debug(f"Failed to log intent: {e}")

        return {"status": "deleted", "entry_id": entry_id}

    raise HTTPException(status_code=404, detail=f"Memory not found: {entry_id}")


@router.delete("/")
async def clear_memories(
    request: Request,
    memory_type: Optional[MemoryType] = None,
    confirm: bool = False,
    session_token: Optional[str] = Cookie(None),
) -> Dict[str, Any]:
    """
    Clear memory entries for the current user.

    Requires confirmation to proceed.
    """
    if not confirm:
        raise HTTPException(
            status_code=400,
            detail="Confirmation required. Set confirm=true to proceed.",
        )

    user_id = get_current_user_id(request, session_token)
    store = get_store()
    deleted_count = store.delete_all(user_id, memory_type)

    # Log intent
    try:
        from ..intent_log import IntentType, log_user_intent
        log_user_intent(
            user_id=user_id,
            intent_type=IntentType.MEMORY_DELETE,
            description=f"Cleared {deleted_count} memories",
            details={"deleted_count": deleted_count, "memory_type": memory_type.value if memory_type else "all"},
        )
    except Exception as e:
        logger.debug(f"Failed to log intent: {e}")

    return {
        "status": "cleared",
        "deleted_count": deleted_count,
        "memory_type": memory_type.value if memory_type else "all",
    }
