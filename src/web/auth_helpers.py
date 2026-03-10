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


def _try_api_key_auth(request: Request) -> Optional[tuple]:
    """Try to authenticate via scoped API key. Returns (user_id, scoped_key) or None."""
    auth_header = request.headers.get("authorization") or request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        return None

    token = auth_header[7:]
    if not token.startswith("aos_"):
        return None

    from .auth import get_user_store

    store = get_user_store()
    api_key = store.validate_api_key(token)
    if api_key:
        return (api_key.user_id, api_key)
    return None


def require_authenticated_user(
    request: Request,
    session_token: Optional[str] = Cookie(None),
) -> str:
    """
    FastAPI dependency: require any authenticated user.

    Supports both session tokens and scoped API keys.
    Returns the user_id if authenticated.
    Raises HTTPException 401 if not authenticated.
    """
    # Try scoped API key first
    api_result = _try_api_key_auth(request)
    if api_result:
        request.state.api_key_scopes = api_result[1].scopes
        return api_result[0]

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

    request.state.api_key_scopes = None
    return user.user_id


def require_admin_user(
    request: Request,
    session_token: Optional[str] = Cookie(None),
) -> str:
    """
    FastAPI dependency: require an authenticated admin user.

    Supports both session tokens (checks ADMIN role) and scoped API keys
    (checks admin scope).
    """
    # Try scoped API key first
    api_result = _try_api_key_auth(request)
    if api_result:
        from .auth import ApiKeyScope

        if ApiKeyScope.ADMIN.value not in api_result[1].scopes:
            raise HTTPException(
                status_code=403,
                detail="API key lacks admin scope",
            )
        request.state.api_key_scopes = api_result[1].scopes
        return api_result[0]

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

    request.state.api_key_scopes = None
    return user.user_id


def require_scope(scope: str):
    """
    FastAPI dependency factory: require a specific API key scope.

    For session-authenticated users, all scopes are granted.
    For API key users, checks if the key has the required scope.

    Usage:
        @router.post("/chat")
        async def send_chat(
            user_id: str = Depends(require_authenticated_user),
            _scope: None = Depends(require_scope("write:chat")),
        ):
            ...
    """

    def _check_scope(request: Request):
        scopes = getattr(request.state, "api_key_scopes", None)
        if scopes is None:
            return  # Session auth — all scopes granted
        from .auth import ApiKeyScope

        if ApiKeyScope.ADMIN.value in scopes or scope in scopes:
            return
        raise HTTPException(
            status_code=403,
            detail=f"API key lacks required scope: {scope}",
        )

    return _check_scope


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
