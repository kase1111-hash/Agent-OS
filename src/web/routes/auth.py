"""
Authentication API Routes

Provides endpoints for user authentication:
- User registration
- Login/logout
- Session management
- Current user info
"""

import logging
from datetime import datetime
from typing import Any, Dict, Optional

from fastapi import APIRouter, Cookie, HTTPException, Request, Response
from pydantic import BaseModel, EmailStr, Field

logger = logging.getLogger(__name__)
router = APIRouter()


# =============================================================================
# Models
# =============================================================================


class RegisterRequest(BaseModel):
    """Request to register a new user."""

    username: str = Field(..., min_length=3, max_length=50)
    password: str = Field(..., min_length=6)
    email: Optional[str] = None
    display_name: Optional[str] = None


class LoginRequest(BaseModel):
    """Request to log in."""

    username: str = Field(..., description="Username or email")
    password: str = Field(..., min_length=1)
    remember_me: bool = False


class LoginResponse(BaseModel):
    """Response after successful login."""

    success: bool
    user: Dict[str, Any]
    token: str
    expires_at: Optional[str] = None
    message: str = "Login successful"


class UserResponse(BaseModel):
    """User information response."""

    user_id: str
    username: str
    email: Optional[str]
    display_name: str
    role: str
    created_at: str
    last_login: Optional[str]


class ChangePasswordRequest(BaseModel):
    """Request to change password."""

    current_password: str
    new_password: str = Field(..., min_length=6)


class UpdateProfileRequest(BaseModel):
    """Request to update user profile."""

    display_name: Optional[str] = None
    email: Optional[str] = None


class AuthStatusResponse(BaseModel):
    """Authentication status response."""

    authenticated: bool
    user: Optional[Dict[str, Any]] = None
    require_auth: bool = True


# =============================================================================
# Helper Functions
# =============================================================================


def get_user_store():
    """Get the user store instance."""
    from ..auth import get_user_store
    return get_user_store()


def get_client_info(request: Request) -> tuple[Optional[str], Optional[str]]:
    """Get client IP and user agent from request."""
    ip = request.client.host if request.client else None
    user_agent = request.headers.get("user-agent")
    return ip, user_agent


def get_token_from_request(request: Request, session_token: Optional[str] = None) -> Optional[str]:
    """Extract session token from request (cookie or header)."""
    # Try cookie first
    if session_token:
        return session_token

    # Try Authorization header
    auth_header = request.headers.get("authorization")
    if auth_header and auth_header.startswith("Bearer "):
        return auth_header[7:]

    return None


# =============================================================================
# Routes
# =============================================================================


@router.post("/register", response_model=LoginResponse)
async def register(
    request: Request,
    response: Response,
    body: RegisterRequest,
):
    """
    Register a new user account.

    Creates a new user and automatically logs them in.
    """
    store = get_user_store()

    try:
        from ..auth import AuthError

        # Create the user
        user = store.create_user(
            username=body.username,
            password=body.password,
            email=body.email,
            display_name=body.display_name,
        )

        # Create session
        ip, user_agent = get_client_info(request)
        session = store.create_session(
            user_id=user.user_id,
            duration_hours=24,
            ip_address=ip,
            user_agent=user_agent,
        )

        # Set session cookie
        response.set_cookie(
            key="session_token",
            value=session.token,
            httponly=True,
            secure=False,  # Set to True in production with HTTPS
            samesite="lax",
            max_age=86400,  # 24 hours
        )

        logger.info(f"User registered: {user.username}")

        return LoginResponse(
            success=True,
            user=user.to_dict(),
            token=session.token,
            expires_at=session.expires_at.isoformat() if session.expires_at else None,
            message="Registration successful",
        )

    except AuthError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Registration failed: {e}")
        raise HTTPException(status_code=500, detail="Registration failed")


@router.post("/login", response_model=LoginResponse)
async def login(
    request: Request,
    response: Response,
    body: LoginRequest,
):
    """
    Log in with username/email and password.

    Returns a session token for subsequent requests.
    """
    store = get_user_store()

    try:
        # Authenticate user
        user = store.authenticate(body.username, body.password)

        if not user:
            raise HTTPException(
                status_code=401,
                detail="Invalid username or password"
            )

        # Create session with longer duration if "remember me" is checked
        duration = 24 * 30 if body.remember_me else 24  # 30 days or 24 hours
        ip, user_agent = get_client_info(request)

        session = store.create_session(
            user_id=user.user_id,
            duration_hours=duration,
            ip_address=ip,
            user_agent=user_agent,
        )

        # Set session cookie
        max_age = duration * 3600
        response.set_cookie(
            key="session_token",
            value=session.token,
            httponly=True,
            secure=False,  # Set to True in production with HTTPS
            samesite="lax",
            max_age=max_age,
        )

        logger.info(f"User logged in: {user.username}")

        return LoginResponse(
            success=True,
            user=user.to_dict(),
            token=session.token,
            expires_at=session.expires_at.isoformat() if session.expires_at else None,
            message="Login successful",
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Login failed: {e}")
        raise HTTPException(status_code=500, detail="Login failed")


@router.post("/logout")
async def logout(
    request: Request,
    response: Response,
    session_token: Optional[str] = Cookie(None),
):
    """
    Log out the current user.

    Invalidates the session and clears the cookie.
    """
    store = get_user_store()
    token = get_token_from_request(request, session_token)

    if token:
        store.invalidate_session_by_token(token)
        logger.info("User logged out")

    # Clear the session cookie
    response.delete_cookie(key="session_token")

    return {"success": True, "message": "Logged out successfully"}


@router.get("/me", response_model=UserResponse)
async def get_current_user(
    request: Request,
    session_token: Optional[str] = Cookie(None),
):
    """
    Get the currently authenticated user's information.
    """
    store = get_user_store()
    token = get_token_from_request(request, session_token)

    if not token:
        raise HTTPException(status_code=401, detail="Not authenticated")

    user = store.validate_session(token)
    if not user:
        raise HTTPException(status_code=401, detail="Session expired or invalid")

    return UserResponse(
        user_id=user.user_id,
        username=user.username,
        email=user.email,
        display_name=user.display_name or user.username,
        role=user.role.value,
        created_at=user.created_at.isoformat(),
        last_login=user.last_login.isoformat() if user.last_login else None,
    )


@router.get("/status", response_model=AuthStatusResponse)
async def get_auth_status(
    request: Request,
    session_token: Optional[str] = Cookie(None),
):
    """
    Check authentication status.

    Returns whether the user is authenticated and their info if so.
    Also indicates whether authentication is required.
    """
    from ..config import get_config
    config = get_config()

    store = get_user_store()
    token = get_token_from_request(request, session_token)

    user = None
    authenticated = False

    if token:
        user_obj = store.validate_session(token)
        if user_obj:
            authenticated = True
            user = user_obj.to_dict()

    return AuthStatusResponse(
        authenticated=authenticated,
        user=user,
        require_auth=config.require_auth,
    )


@router.put("/profile")
async def update_profile(
    request: Request,
    body: UpdateProfileRequest,
    session_token: Optional[str] = Cookie(None),
):
    """
    Update the current user's profile.
    """
    store = get_user_store()
    token = get_token_from_request(request, session_token)

    if not token:
        raise HTTPException(status_code=401, detail="Not authenticated")

    user = store.validate_session(token)
    if not user:
        raise HTTPException(status_code=401, detail="Session expired or invalid")

    try:
        from ..auth import AuthError

        store.update_user(
            user_id=user.user_id,
            display_name=body.display_name,
            email=body.email,
        )

        # Fetch updated user
        updated_user = store.get_user(user.user_id)

        return {
            "success": True,
            "message": "Profile updated",
            "user": updated_user.to_dict() if updated_user else None,
        }

    except AuthError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Profile update failed: {e}")
        raise HTTPException(status_code=500, detail="Update failed")


@router.post("/change-password")
async def change_password(
    request: Request,
    body: ChangePasswordRequest,
    session_token: Optional[str] = Cookie(None),
):
    """
    Change the current user's password.
    """
    store = get_user_store()
    token = get_token_from_request(request, session_token)

    if not token:
        raise HTTPException(status_code=401, detail="Not authenticated")

    user = store.validate_session(token)
    if not user:
        raise HTTPException(status_code=401, detail="Session expired or invalid")

    # Verify current password
    from ..auth import PasswordHasher
    if not PasswordHasher.verify_password(
        body.current_password,
        user.password_hash,
        user.salt
    ):
        raise HTTPException(status_code=400, detail="Current password is incorrect")

    try:
        from ..auth import AuthError

        store.change_password(user.user_id, body.new_password)

        # Optionally invalidate all other sessions
        # store.invalidate_all_user_sessions(user.user_id)

        return {"success": True, "message": "Password changed successfully"}

    except AuthError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Password change failed: {e}")
        raise HTTPException(status_code=500, detail="Password change failed")


@router.get("/sessions")
async def get_sessions(
    request: Request,
    session_token: Optional[str] = Cookie(None),
):
    """
    Get all active sessions for the current user.
    """
    store = get_user_store()
    token = get_token_from_request(request, session_token)

    if not token:
        raise HTTPException(status_code=401, detail="Not authenticated")

    user = store.validate_session(token)
    if not user:
        raise HTTPException(status_code=401, detail="Session expired or invalid")

    sessions = store.get_user_sessions(user.user_id)

    return {
        "sessions": [s.to_dict() for s in sessions],
        "count": len(sessions),
    }


@router.delete("/sessions/{session_id}")
async def invalidate_session(
    session_id: str,
    request: Request,
    session_token: Optional[str] = Cookie(None),
):
    """
    Invalidate a specific session.
    """
    store = get_user_store()
    token = get_token_from_request(request, session_token)

    if not token:
        raise HTTPException(status_code=401, detail="Not authenticated")

    user = store.validate_session(token)
    if not user:
        raise HTTPException(status_code=401, detail="Session expired or invalid")

    # Verify the session belongs to this user
    sessions = store.get_user_sessions(user.user_id)
    session_ids = [s.session_id for s in sessions]

    if session_id not in session_ids:
        raise HTTPException(status_code=404, detail="Session not found")

    store.invalidate_session(session_id)

    return {"success": True, "message": "Session invalidated"}


@router.post("/logout-all")
async def logout_all_sessions(
    request: Request,
    response: Response,
    session_token: Optional[str] = Cookie(None),
):
    """
    Log out from all sessions.
    """
    store = get_user_store()
    token = get_token_from_request(request, session_token)

    if not token:
        raise HTTPException(status_code=401, detail="Not authenticated")

    user = store.validate_session(token)
    if not user:
        raise HTTPException(status_code=401, detail="Session expired or invalid")

    count = store.invalidate_all_user_sessions(user.user_id)
    response.delete_cookie(key="session_token")

    logger.info(f"User logged out from all sessions: {user.username}")

    return {
        "success": True,
        "message": f"Logged out from {count} session(s)",
        "sessions_invalidated": count,
    }


# =============================================================================
# Admin Routes (for future use)
# =============================================================================


@router.get("/users/count")
async def get_user_count(
    request: Request,
    session_token: Optional[str] = Cookie(None),
):
    """
    Get the total number of registered users.
    """
    store = get_user_store()
    token = get_token_from_request(request, session_token)

    if token:
        user = store.validate_session(token)
        if user and user.role.value == "admin":
            return {"count": store.get_user_count()}

    # Return count even for unauthenticated requests (public info)
    return {"count": store.get_user_count()}
