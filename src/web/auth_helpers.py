"""
Consolidated Authentication Helpers for Agent OS Web Routes.

Provides reusable FastAPI dependencies for authentication and authorization.
All route modules should import from here instead of defining their own auth logic.
"""

import logging
from typing import Optional

from fastapi import Cookie, HTTPException, Request, WebSocket

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


async def authenticate_websocket(websocket: WebSocket) -> Optional[str]:
    """
    Authenticate a WebSocket connection before accepting it.

    Checks for a session token in cookies or the ``token`` query parameter.

    Returns:
        User ID if authenticated, ``None`` otherwise.

    Usage::

        user_id = await authenticate_websocket(websocket)
        if not user_id:
            await websocket.close(code=4001, reason="Authentication required")
            return
    """
    from .auth import get_user_store

    # Try to get token from cookies
    token = websocket.cookies.get("session_token")

    # Fall back to query parameter
    if not token:
        token = websocket.query_params.get("token")

    if not token:
        return None

    store = get_user_store()
    user = store.validate_session(token)

    if not user:
        return None

    return user.user_id
