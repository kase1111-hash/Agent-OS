"""
Consolidated Authentication Helpers for Agent OS Web Routes.

Provides reusable FastAPI dependencies for authentication and authorization.
All route modules should import from here instead of defining their own auth logic.
"""

import logging
from typing import Optional

from fastapi import Cookie, HTTPException, Request

logger = logging.getLogger(__name__)


def _extract_token(request: Request, session_token: Optional[str] = None) -> Optional[str]:
    """Extract session token from cookie or Authorization header."""
    token = session_token
    if not token:
        token = request.cookies.get("session_token")
    if not token:
        auth_header = request.headers.get("authorization") or request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            token = auth_header[7:]
    return token


def require_authenticated_user(
    request: Request,
    session_token: Optional[str] = Cookie(None),
) -> str:
    """
    FastAPI dependency: require any authenticated user.

    Returns the user_id if authenticated.
    Raises HTTPException 401 if not authenticated.

    Usage:
        @router.get("/example")
        async def example(user_id: str = Depends(require_authenticated_user)):
            ...
    """
    from .auth import get_user_store

    token = _extract_token(request, session_token)

    if not token:
        raise HTTPException(
            status_code=401,
            detail="Authentication required",
        )

    store = get_user_store()
    user = store.validate_session(token)

    if not user:
        raise HTTPException(
            status_code=401,
            detail="Session expired or invalid",
        )

    return user.user_id


def require_admin_user(
    request: Request,
    session_token: Optional[str] = Cookie(None),
) -> str:
    """
    FastAPI dependency: require an authenticated admin user.

    Returns the user_id if authenticated and has admin role.
    Raises HTTPException 401 if not authenticated, 403 if not admin.

    Usage:
        @router.post("/admin-action")
        async def admin_action(admin_id: str = Depends(require_admin_user)):
            ...
    """
    from .auth import UserRole, get_user_store

    token = _extract_token(request, session_token)

    if not token:
        raise HTTPException(
            status_code=401,
            detail="Authentication required for admin operations",
        )

    store = get_user_store()
    user = store.validate_session(token)

    if not user:
        raise HTTPException(
            status_code=401,
            detail="Session expired or invalid",
        )

    if user.role != UserRole.ADMIN:
        raise HTTPException(
            status_code=403,
            detail="Admin privileges required for this operation",
        )

    return user.user_id
