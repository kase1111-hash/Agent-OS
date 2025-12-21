"""
Memory Management API Routes

Provides endpoints for viewing and managing agent memory.
"""

import logging
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)
router = APIRouter()


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
# Mock Memory Store
# =============================================================================


class MemoryStore:
    """
    Mock memory store for the web interface.

    In production, this would integrate with the actual MemoryVault.
    """

    def __init__(self):
        self._entries: Dict[str, MemoryEntry] = {}
        self._initialize_mock_data()

    def _initialize_mock_data(self):
        """Initialize with some mock memories."""
        import uuid

        mock_memories = [
            MemoryEntry(
                id=str(uuid.uuid4()),
                content="User prefers dark mode for all applications",
                memory_type=MemoryType.LONG_TERM,
                source="user",
                tags=["preferences", "ui"],
                metadata={"preference_type": "display"},
                consent_given=True,
            ),
            MemoryEntry(
                id=str(uuid.uuid4()),
                content="User's name is Alice",
                memory_type=MemoryType.LONG_TERM,
                source="user",
                tags=["identity", "personal"],
                metadata={"data_type": "personal_info"},
                consent_given=True,
            ),
            MemoryEntry(
                id=str(uuid.uuid4()),
                content="User is working on a Python project called 'data-analyzer'",
                memory_type=MemoryType.WORKING,
                source="system",
                tags=["context", "project"],
                metadata={"project": "data-analyzer", "language": "python"},
                consent_given=True,
            ),
            MemoryEntry(
                id=str(uuid.uuid4()),
                content="Agent recommended using pandas for data processing",
                memory_type=MemoryType.EPHEMERAL,
                source="agent",
                tags=["recommendation", "technical"],
                metadata={"agent": "oracle"},
                consent_given=True,
                retention_days=7,
            ),
            MemoryEntry(
                id=str(uuid.uuid4()),
                content="The weather API key is stored in environment variable WEATHER_API_KEY",
                memory_type=MemoryType.SEMANTIC,
                source="user",
                tags=["technical", "configuration"],
                metadata={"type": "config_info"},
                consent_given=True,
            ),
        ]

        for memory in mock_memories:
            self._entries[memory.id] = memory

    def get_all(self) -> List[MemoryEntry]:
        """Get all memory entries."""
        return list(self._entries.values())

    def get(self, entry_id: str) -> Optional[MemoryEntry]:
        """Get a memory entry by ID."""
        entry = self._entries.get(entry_id)
        if entry:
            entry.accessed_at = datetime.utcnow()
            entry.access_count += 1
        return entry

    def create(self, request: MemoryCreate) -> MemoryEntry:
        """Create a new memory entry."""
        import uuid

        entry = MemoryEntry(
            id=str(uuid.uuid4()),
            content=request.content,
            memory_type=request.memory_type,
            tags=request.tags,
            metadata=request.metadata,
            consent_given=request.consent_given,
            retention_days=request.retention_days,
        )
        self._entries[entry.id] = entry
        return entry

    def update(self, entry_id: str, request: MemoryUpdate) -> Optional[MemoryEntry]:
        """Update a memory entry."""
        if entry_id not in self._entries:
            return None

        entry = self._entries[entry_id]

        if request.content is not None:
            entry.content = request.content
        if request.tags is not None:
            entry.tags = request.tags
        if request.metadata is not None:
            entry.metadata.update(request.metadata)

        return entry

    def delete(self, entry_id: str) -> bool:
        """Delete a memory entry."""
        if entry_id in self._entries:
            del self._entries[entry_id]
            return True
        return False

    def search(
        self,
        query: str,
        memory_type: Optional[MemoryType] = None,
        tags: Optional[List[str]] = None,
        limit: int = 10,
    ) -> List[SearchResult]:
        """Search memory entries."""
        query_lower = query.lower()
        results = []

        for entry in self._entries.values():
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
                        # Find context around the word
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

        # Sort by score descending
        results.sort(key=lambda r: r.similarity_score, reverse=True)
        return results[:limit]

    def get_stats(self) -> MemoryStats:
        """Get memory statistics."""
        entries = list(self._entries.values())

        if not entries:
            return MemoryStats(
                total_entries=0,
                entries_by_type={},
                total_size_bytes=0,
            )

        type_counts = {}
        total_size = 0
        consent_count = 0

        for entry in entries:
            type_key = entry.memory_type.value
            type_counts[type_key] = type_counts.get(type_key, 0) + 1
            total_size += len(entry.content.encode())
            if entry.consent_given:
                consent_count += 1

        dates = [e.created_at for e in entries]

        return MemoryStats(
            total_entries=len(entries),
            entries_by_type=type_counts,
            total_size_bytes=total_size,
            oldest_entry=min(dates),
            newest_entry=max(dates),
            consent_rate=(consent_count / len(entries)) * 100,
        )


# Global store instance
_store: Optional[MemoryStore] = None


def get_store() -> MemoryStore:
    """Get the memory store."""
    global _store
    if _store is None:
        _store = MemoryStore()
    return _store


# =============================================================================
# Endpoints
# =============================================================================


@router.get("/", response_model=List[MemoryEntry])
async def list_memories(
    memory_type: Optional[MemoryType] = None,
    tag: Optional[str] = None,
    limit: int = Query(default=50, le=100),
) -> List[MemoryEntry]:
    """List memory entries with optional filtering."""
    store = get_store()
    entries = store.get_all()

    if memory_type:
        entries = [e for e in entries if e.memory_type == memory_type]

    if tag:
        entries = [e for e in entries if tag in e.tags]

    # Sort by created_at descending
    entries.sort(key=lambda e: e.created_at, reverse=True)

    return entries[:limit]


@router.get("/stats", response_model=MemoryStats)
async def get_memory_stats() -> MemoryStats:
    """Get memory statistics."""
    store = get_store()
    return store.get_stats()


@router.get("/search", response_model=List[SearchResult])
async def search_memories(
    query: str,
    memory_type: Optional[MemoryType] = None,
    tags: Optional[str] = None,
    limit: int = Query(default=10, le=50),
) -> List[SearchResult]:
    """Search memory entries."""
    store = get_store()

    tag_list = tags.split(",") if tags else None

    return store.search(
        query=query,
        memory_type=memory_type,
        tags=tag_list,
        limit=limit,
    )


@router.get("/export")
async def export_memories(
    memory_type: Optional[MemoryType] = None,
) -> Dict[str, Any]:
    """Export memories as JSON."""
    store = get_store()
    entries = store.get_all()

    if memory_type:
        entries = [e for e in entries if e.memory_type == memory_type]

    return {
        "exported_at": datetime.utcnow().isoformat(),
        "count": len(entries),
        "entries": [e.model_dump() for e in entries],
    }


@router.get("/{entry_id}", response_model=MemoryEntry)
async def get_memory(entry_id: str) -> MemoryEntry:
    """Get a specific memory entry."""
    store = get_store()
    entry = store.get(entry_id)

    if not entry:
        raise HTTPException(status_code=404, detail=f"Memory not found: {entry_id}")

    return entry


@router.post("/", response_model=MemoryEntry)
async def create_memory(request: MemoryCreate) -> MemoryEntry:
    """Create a new memory entry."""
    if not request.consent_given:
        raise HTTPException(
            status_code=403,
            detail="Consent is required to store memory",
        )

    store = get_store()
    return store.create(request)


@router.put("/{entry_id}", response_model=MemoryEntry)
async def update_memory(entry_id: str, request: MemoryUpdate) -> MemoryEntry:
    """Update a memory entry."""
    store = get_store()
    entry = store.update(entry_id, request)

    if not entry:
        raise HTTPException(status_code=404, detail=f"Memory not found: {entry_id}")

    return entry


@router.delete("/{entry_id}")
async def delete_memory(entry_id: str) -> Dict[str, str]:
    """Delete a memory entry."""
    store = get_store()

    if store.delete(entry_id):
        return {"status": "deleted", "entry_id": entry_id}

    raise HTTPException(status_code=404, detail=f"Memory not found: {entry_id}")


@router.delete("/")
async def clear_memories(
    memory_type: Optional[MemoryType] = None,
    confirm: bool = False,
) -> Dict[str, Any]:
    """
    Clear memory entries.

    Requires confirmation to proceed.
    """
    if not confirm:
        raise HTTPException(
            status_code=400,
            detail="Confirmation required. Set confirm=true to proceed.",
        )

    store = get_store()
    entries = store.get_all()

    if memory_type:
        entries = [e for e in entries if e.memory_type == memory_type]

    deleted_count = 0
    for entry in entries:
        if store.delete(entry.id):
            deleted_count += 1

    return {
        "status": "cleared",
        "deleted_count": deleted_count,
        "memory_type": memory_type.value if memory_type else "all",
    }
