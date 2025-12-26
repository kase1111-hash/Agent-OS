"""
Intent Log API Routes

REST API endpoints for viewing and managing user intent logs.
All endpoints are user-scoped - users can only see their own logs.
"""

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Cookie, HTTPException, Query, Request
from pydantic import BaseModel, Field

from ..intent_log import (
    IntentLogEntry,
    IntentLogQuery,
    IntentType,
    get_intent_log_store,
)

logger = logging.getLogger(__name__)

router = APIRouter()


# =============================================================================
# Helper Functions
# =============================================================================


def get_current_user_id(
    request: Request,
    session_token: Optional[str] = None,
) -> str:
    """
    Get the current user ID from the session.

    Returns the authenticated user's ID.
    Raises HTTPException 401 if not authenticated.
    """
    try:
        from ..auth import get_user_store

        token = session_token
        if not token:
            # Try to get from Authorization header
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
# Request/Response Models
# =============================================================================


class IntentLogEntryResponse(BaseModel):
    """Response model for intent log entry."""
    entry_id: str
    user_id: str
    intent_type: str
    description: str
    details: Dict[str, Any] = Field(default_factory=dict)
    session_id: Optional[str] = None
    related_entity_type: Optional[str] = None
    related_entity_id: Optional[str] = None
    created_at: datetime


class IntentLogStatsResponse(BaseModel):
    """Response model for intent log statistics."""
    user_id: str
    total_entries: int
    by_type: Dict[str, int]
    recent_24h: int
    recent_7d: int


class IntentLogListResponse(BaseModel):
    """Response model for list of intent log entries."""
    entries: List[IntentLogEntryResponse]
    total: int
    limit: int
    offset: int


# =============================================================================
# API Endpoints
# =============================================================================


@router.get("", response_model=IntentLogListResponse)
async def list_intent_logs(
    request: Request,
    session_token: Optional[str] = Cookie(None),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    intent_type: Optional[str] = Query(None, description="Filter by intent type"),
    session_id: Optional[str] = Query(None, description="Filter by session ID"),
    related_entity_type: Optional[str] = Query(None, description="Filter by related entity type"),
    related_entity_id: Optional[str] = Query(None, description="Filter by related entity ID"),
    start_date: Optional[datetime] = Query(None, description="Filter from this date"),
    end_date: Optional[datetime] = Query(None, description="Filter until this date"),
    search: Optional[str] = Query(None, description="Search in description"),
) -> IntentLogListResponse:
    """
    List intent log entries for the current user.

    Returns entries sorted by creation date (newest first).
    """
    user_id = get_current_user_id(request, session_token)

    # Build query
    query = IntentLogQuery(
        user_id=user_id,
        limit=limit,
        offset=offset,
        session_id=session_id,
        related_entity_type=related_entity_type,
        related_entity_id=related_entity_id,
        start_date=start_date,
        end_date=end_date,
        search_text=search,
    )

    # Parse intent type if provided
    if intent_type:
        try:
            query.intent_type = IntentType[intent_type.upper()]
        except KeyError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid intent type: {intent_type}"
            )

    store = get_intent_log_store()
    entries = store.query_entries(query)

    # Get total count (without limit/offset)
    count_query = IntentLogQuery(
        user_id=user_id,
        intent_type=query.intent_type,
        session_id=session_id,
        related_entity_type=related_entity_type,
        related_entity_id=related_entity_id,
        start_date=start_date,
        end_date=end_date,
        search_text=search,
        limit=10000,  # Large limit to count
        offset=0,
    )
    total = len(store.query_entries(count_query))

    return IntentLogListResponse(
        entries=[
            IntentLogEntryResponse(
                entry_id=e.entry_id,
                user_id=e.user_id,
                intent_type=e.intent_type.name,
                description=e.description,
                details=e.details,
                session_id=e.session_id,
                related_entity_type=e.related_entity_type,
                related_entity_id=e.related_entity_id,
                created_at=e.created_at,
            )
            for e in entries
        ],
        total=total,
        limit=limit,
        offset=offset,
    )


@router.get("/stats", response_model=IntentLogStatsResponse)
async def get_intent_log_stats(
    request: Request,
    session_token: Optional[str] = Cookie(None),
) -> IntentLogStatsResponse:
    """Get statistics for the current user's intent log."""
    user_id = get_current_user_id(request, session_token)
    store = get_intent_log_store()
    stats = store.get_statistics(user_id)

    return IntentLogStatsResponse(**stats)


@router.get("/types")
async def list_intent_types() -> Dict[str, List[str]]:
    """List all available intent types."""
    return {
        "intent_types": [t.name for t in IntentType]
    }


@router.get("/recent")
async def get_recent_intents(
    request: Request,
    session_token: Optional[str] = Cookie(None),
    hours: int = Query(24, ge=1, le=168),
    limit: int = Query(50, ge=1, le=200),
) -> IntentLogListResponse:
    """
    Get recent intent log entries for the current user.

    Default is last 24 hours.
    """
    user_id = get_current_user_id(request, session_token)
    store = get_intent_log_store()
    entries = store.get_recent_entries(user_id, hours=hours, limit=limit)

    return IntentLogListResponse(
        entries=[
            IntentLogEntryResponse(
                entry_id=e.entry_id,
                user_id=e.user_id,
                intent_type=e.intent_type.name,
                description=e.description,
                details=e.details,
                session_id=e.session_id,
                related_entity_type=e.related_entity_type,
                related_entity_id=e.related_entity_id,
                created_at=e.created_at,
            )
            for e in entries
        ],
        total=len(entries),
        limit=limit,
        offset=0,
    )


@router.get("/session/{session_id}")
async def get_session_intents(
    request: Request,
    session_id: str,
    session_token: Optional[str] = Cookie(None),
) -> IntentLogListResponse:
    """Get all intent log entries for a specific session."""
    user_id = get_current_user_id(request, session_token)
    store = get_intent_log_store()
    entries = store.get_session_entries(user_id, session_id)

    return IntentLogListResponse(
        entries=[
            IntentLogEntryResponse(
                entry_id=e.entry_id,
                user_id=e.user_id,
                intent_type=e.intent_type.name,
                description=e.description,
                details=e.details,
                session_id=e.session_id,
                related_entity_type=e.related_entity_type,
                related_entity_id=e.related_entity_id,
                created_at=e.created_at,
            )
            for e in entries
        ],
        total=len(entries),
        limit=len(entries),
        offset=0,
    )


@router.get("/{entry_id}", response_model=IntentLogEntryResponse)
async def get_intent_log_entry(
    request: Request,
    entry_id: str,
    session_token: Optional[str] = Cookie(None),
) -> IntentLogEntryResponse:
    """Get a specific intent log entry."""
    user_id = get_current_user_id(request, session_token)
    store = get_intent_log_store()
    entry = store.get_entry(entry_id)

    if not entry:
        raise HTTPException(status_code=404, detail="Entry not found")

    # Verify ownership
    if entry.user_id != user_id and user_id != "default":
        raise HTTPException(status_code=403, detail="Access denied")

    return IntentLogEntryResponse(
        entry_id=entry.entry_id,
        user_id=entry.user_id,
        intent_type=entry.intent_type.name,
        description=entry.description,
        details=entry.details,
        session_id=entry.session_id,
        related_entity_type=entry.related_entity_type,
        related_entity_id=entry.related_entity_id,
        created_at=entry.created_at,
    )


@router.delete("")
async def clear_intent_log(
    request: Request,
    session_token: Optional[str] = Cookie(None),
) -> Dict[str, Any]:
    """Clear all intent log entries for the current user."""
    user_id = get_current_user_id(request, session_token)

    if user_id == "default":
        raise HTTPException(
            status_code=401,
            detail="Authentication required to clear intent log"
        )

    store = get_intent_log_store()
    count = store.delete_user_entries(user_id)

    return {
        "status": "cleared",
        "entries_deleted": count,
    }


@router.post("/export")
async def export_intent_log(
    request: Request,
    session_token: Optional[str] = Cookie(None),
    start_date: Optional[datetime] = Query(None),
    end_date: Optional[datetime] = Query(None),
) -> Dict[str, Any]:
    """Export intent log entries for the current user."""
    user_id = get_current_user_id(request, session_token)

    query = IntentLogQuery(
        user_id=user_id,
        start_date=start_date,
        end_date=end_date,
        limit=10000,  # Large limit for export
    )

    store = get_intent_log_store()
    entries = store.query_entries(query)

    return {
        "user_id": user_id,
        "exported_at": datetime.now().isoformat(),
        "entry_count": len(entries),
        "entries": [e.to_dict() for e in entries],
    }
