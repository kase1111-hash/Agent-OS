"""Mobile Authentication Module for Agent OS.

Provides authentication services for mobile applications:
- Device token management
- Biometric authentication (Face ID, Touch ID, Fingerprint)
- Session management
- Token refresh and rotation
"""

import asyncio
import hashlib
import hmac
import logging
import os
import secrets
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


def _is_production_mode() -> bool:
    """Check if running in production mode.

    Production mode is enabled when AGENT_OS_PRODUCTION is set to
    '1', 'true', or 'yes' (case-insensitive).
    """
    env_value = os.environ.get("AGENT_OS_PRODUCTION", "").lower()
    return env_value in ("1", "true", "yes")


class AuthState(str, Enum):
    """Authentication states."""

    UNAUTHENTICATED = "unauthenticated"
    AUTHENTICATING = "authenticating"
    AUTHENTICATED = "authenticated"
    REFRESHING = "refreshing"
    EXPIRED = "expired"
    LOCKED = "locked"


class BiometricType(str, Enum):
    """Types of biometric authentication."""

    NONE = "none"
    FACE_ID = "face_id"
    TOUCH_ID = "touch_id"
    FINGERPRINT = "fingerprint"
    IRIS = "iris"


from src.core.exceptions import AuthError


@dataclass
class AuthToken:
    """Authentication token."""

    access_token: str
    token_type: str = "Bearer"
    expires_at: Optional[datetime] = None
    refresh_token: Optional[str] = None
    refresh_expires_at: Optional[datetime] = None
    scope: List[str] = field(default_factory=list)
    issued_at: datetime = field(default_factory=datetime.now)

    @property
    def is_expired(self) -> bool:
        """Check if access token is expired."""
        if not self.expires_at:
            return False
        return datetime.now() >= self.expires_at

    @property
    def is_refresh_expired(self) -> bool:
        """Check if refresh token is expired."""
        if not self.refresh_expires_at:
            return False
        return datetime.now() >= self.refresh_expires_at

    @property
    def expires_in(self) -> int:
        """Get seconds until expiry."""
        if not self.expires_at:
            return -1
        delta = self.expires_at - datetime.now()
        return max(0, int(delta.total_seconds()))

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "access_token": self.access_token,
            "token_type": self.token_type,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "refresh_token": self.refresh_token,
            "refresh_expires_at": (
                self.refresh_expires_at.isoformat() if self.refresh_expires_at else None
            ),
            "scope": self.scope,
            "issued_at": self.issued_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AuthToken":
        """Create from dictionary."""
        if isinstance(data.get("expires_at"), str):
            data["expires_at"] = datetime.fromisoformat(data["expires_at"])
        if isinstance(data.get("refresh_expires_at"), str):
            data["refresh_expires_at"] = datetime.fromisoformat(data["refresh_expires_at"])
        if isinstance(data.get("issued_at"), str):
            data["issued_at"] = datetime.fromisoformat(data["issued_at"])
        return cls(**data)

    @classmethod
    def from_response(cls, response: Dict[str, Any]) -> "AuthToken":
        """Create from OAuth response."""
        now = datetime.now()
        expires_in = response.get("expires_in", 3600)
        refresh_expires_in = response.get("refresh_expires_in", 86400)

        return cls(
            access_token=response["access_token"],
            token_type=response.get("token_type", "Bearer"),
            expires_at=now + timedelta(seconds=expires_in),
            refresh_token=response.get("refresh_token"),
            refresh_expires_at=(
                now + timedelta(seconds=refresh_expires_in)
                if response.get("refresh_token")
                else None
            ),
            scope=response.get("scope", "").split() if response.get("scope") else [],
            issued_at=now,
        )


@dataclass
class DeviceToken:
    """Device registration token."""

    device_id: str
    token: str
    platform: str
    created_at: datetime = field(default_factory=datetime.now)
    last_used: datetime = field(default_factory=datetime.now)
    is_trusted: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "device_id": self.device_id,
            "token": self.token,
            "platform": self.platform,
            "created_at": self.created_at.isoformat(),
            "last_used": self.last_used.isoformat(),
            "is_trusted": self.is_trusted,
            "metadata": self.metadata,
        }


@dataclass
class AuthConfig:
    """Authentication configuration."""

    client_id: str = ""
    client_secret: str = ""
    auth_url: str = "https://auth.agentos.local"
    token_endpoint: str = "/oauth/token"
    device_endpoint: str = "/device/register"
    biometric_enabled: bool = True
    auto_refresh: bool = True
    refresh_threshold: int = 300  # Refresh when < 5 minutes left
    max_failed_attempts: int = 5
    lockout_duration: int = 300  # 5 minutes
    session_timeout: int = 3600  # 1 hour
    require_biometric_confirmation: bool = False


class BiometricAuth:
    """Biometric authentication handler.

    Provides unified interface for:
    - Face ID (iOS)
    - Touch ID (iOS)
    - Fingerprint (Android)

    WARNING: This implementation simulates biometric authentication.
    In production mode (AGENT_OS_PRODUCTION=1), simulated authentication
    is disabled and requires real OS biometric integration.
    """

    def __init__(self, biometric_type: BiometricType = BiometricType.NONE):
        """Initialize biometric auth.

        Args:
            biometric_type: Type of biometric authentication
        """
        self.biometric_type = biometric_type
        self._is_enrolled = False
        self._auth_callbacks: List[Callable[[bool], None]] = []
        self._production_mode = _is_production_mode()

        if self._production_mode and biometric_type != BiometricType.NONE:
            logger.warning(
                "BiometricAuth running in production mode - " "simulated authentication is disabled"
            )

    @property
    def is_available(self) -> bool:
        """Check if biometric auth is available."""
        return self.biometric_type != BiometricType.NONE

    @property
    def is_enrolled(self) -> bool:
        """Check if user has enrolled biometrics."""
        return self._is_enrolled

    def enroll(self) -> bool:
        """Enroll biometric authentication.

        Returns:
            True if enrollment successful
        """
        if not self.is_available:
            return False

        # In production, this would trigger OS biometric enrollment
        self._is_enrolled = True
        logger.info(f"Biometric enrolled: {self.biometric_type.value}")
        return True

    async def authenticate(self, reason: str = "Authenticate") -> bool:
        """Perform biometric authentication.

        Args:
            reason: Reason shown to user

        Returns:
            True if authentication successful

        Raises:
            RuntimeError: If called in production mode without real OS integration
        """
        if not self.is_available or not self._is_enrolled:
            return False

        if self._production_mode:
            # In production mode, simulated biometrics are not allowed
            # Real OS integration would be required here
            logger.error(
                "Simulated biometric authentication is disabled in production. "
                "Real OS biometric integration is required."
            )
            raise RuntimeError(
                "Simulated biometric authentication is disabled in production mode. "
                "Real OS biometric integration is required."
            )

        # Simulate biometric prompt (development/testing only)
        await asyncio.sleep(0.1)

        # In development, simulate success
        success = True

        for callback in self._auth_callbacks:
            try:
                callback(success)
            except Exception as e:
                logger.warning(f"Auth callback error: {e}")

        return success

    def on_auth_result(self, callback: Callable[[bool], None]) -> None:
        """Register authentication result callback."""
        self._auth_callbacks.append(callback)

    def get_supported_types(self) -> List[BiometricType]:
        """Get list of supported biometric types for this device."""
        # In production, query OS for available types
        return [BiometricType.FACE_ID, BiometricType.TOUCH_ID, BiometricType.FINGERPRINT]


class MobileAuth:
    """Mobile authentication manager.

    Features:
    - Device registration
    - OAuth token management
    - Biometric authentication
    - Session management
    """

    def __init__(self, config: Optional[AuthConfig] = None):
        """Initialize mobile auth.

        Args:
            config: Authentication configuration
        """
        self.config = config or AuthConfig()
        self._state = AuthState.UNAUTHENTICATED
        self._state_callbacks: List[Callable[[AuthState], None]] = []
        self._token: Optional[AuthToken] = None
        self._device_token: Optional[DeviceToken] = None
        self._biometric: Optional[BiometricAuth] = None
        self._failed_attempts = 0
        self._lockout_until: Optional[datetime] = None
        self._refresh_task: Optional[asyncio.Task] = None
        self._session_start: Optional[datetime] = None

    @property
    def state(self) -> AuthState:
        """Get current auth state."""
        return self._state

    @property
    def is_authenticated(self) -> bool:
        """Check if user is authenticated."""
        return self._state == AuthState.AUTHENTICATED and self._token is not None

    @property
    def is_locked(self) -> bool:
        """Check if account is locked."""
        if self._lockout_until:
            if datetime.now() < self._lockout_until:
                return True
            self._lockout_until = None
            self._failed_attempts = 0
        return False

    @property
    def current_token(self) -> Optional[AuthToken]:
        """Get current auth token."""
        return self._token

    def on_state_change(self, callback: Callable[[AuthState], None]) -> None:
        """Register state change callback."""
        self._state_callbacks.append(callback)

    def _set_state(self, state: AuthState) -> None:
        """Set auth state and notify callbacks."""
        if self._state != state:
            old_state = self._state
            self._state = state
            logger.info(f"Auth state changed: {old_state.value} -> {state.value}")
            for callback in self._state_callbacks:
                try:
                    callback(state)
                except Exception as e:
                    logger.warning(f"State callback error: {e}")

    async def register_device(
        self,
        device_id: str,
        platform: str,
        push_token: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> DeviceToken:
        """Register device with server.

        Args:
            device_id: Unique device identifier
            platform: Platform type (ios/android)
            push_token: Push notification token
            metadata: Additional device metadata

        Returns:
            Device token
        """
        # Generate device token
        token = secrets.token_urlsafe(32)

        self._device_token = DeviceToken(
            device_id=device_id,
            token=token,
            platform=platform,
            metadata=metadata or {},
        )

        if push_token:
            self._device_token.metadata["push_token"] = push_token

        logger.info(f"Device registered: {device_id}")
        return self._device_token

    async def authenticate(
        self,
        username: Optional[str] = None,
        password: Optional[str] = None,
        device_token: Optional[str] = None,
        use_biometric: bool = False,
    ) -> AuthToken:
        """Authenticate user.

        Args:
            username: Username (optional if using device token)
            password: Password (optional if using device token)
            device_token: Device token for trusted device auth
            use_biometric: Use biometric authentication

        Returns:
            Auth token

        Raises:
            AuthError: If authentication fails
        """
        if self.is_locked:
            remaining = (self._lockout_until - datetime.now()).total_seconds()
            raise AuthError(
                f"Account locked. Try again in {int(remaining)} seconds",
                error_code="ACCOUNT_LOCKED",
                is_recoverable=False,
            )

        self._set_state(AuthState.AUTHENTICATING)

        try:
            # Biometric authentication
            if use_biometric and self._biometric:
                if not await self._biometric.authenticate("Sign in to Agent OS"):
                    self._record_failed_attempt()
                    raise AuthError(
                        "Biometric authentication failed",
                        error_code="BIOMETRIC_FAILED",
                    )

            # Device token authentication
            if device_token:
                token = await self._auth_with_device_token(device_token)
            # Username/password authentication
            elif username and password:
                token = await self._auth_with_credentials(username, password)
            else:
                raise AuthError(
                    "No authentication method provided",
                    error_code="NO_AUTH_METHOD",
                )

            self._token = token
            self._failed_attempts = 0
            self._session_start = datetime.now()
            self._set_state(AuthState.AUTHENTICATED)

            # Start auto-refresh
            if self.config.auto_refresh:
                self._start_refresh_task()

            logger.info("Authentication successful")
            return token

        except AuthError:
            self._set_state(AuthState.UNAUTHENTICATED)
            raise

        except Exception as e:
            self._record_failed_attempt()
            self._set_state(AuthState.UNAUTHENTICATED)
            raise AuthError(f"Authentication failed: {e}")

    async def _auth_with_credentials(self, username: str, password: str) -> AuthToken:
        """Authenticate with username/password.

        Returns:
            Auth token
        """
        # Mock authentication (in production, call auth server)
        await asyncio.sleep(0.1)

        # Generate mock token
        return AuthToken(
            access_token=secrets.token_urlsafe(32),
            expires_at=datetime.now() + timedelta(hours=1),
            refresh_token=secrets.token_urlsafe(32),
            refresh_expires_at=datetime.now() + timedelta(days=7),
            scope=["read", "write"],
        )

    async def _auth_with_device_token(self, device_token: str) -> AuthToken:
        """Authenticate with device token.

        Returns:
            Auth token
        """
        # Mock authentication (in production, verify device token)
        await asyncio.sleep(0.1)

        return AuthToken(
            access_token=secrets.token_urlsafe(32),
            expires_at=datetime.now() + timedelta(hours=1),
            refresh_token=secrets.token_urlsafe(32),
            refresh_expires_at=datetime.now() + timedelta(days=30),
            scope=["read", "write", "device"],
        )

    async def refresh_token(self) -> AuthToken:
        """Refresh the access token.

        Returns:
            New auth token

        Raises:
            AuthError: If refresh fails
        """
        if not self._token or not self._token.refresh_token:
            raise AuthError(
                "No refresh token available",
                error_code="NO_REFRESH_TOKEN",
            )

        if self._token.is_refresh_expired:
            self._set_state(AuthState.EXPIRED)
            raise AuthError(
                "Refresh token expired",
                error_code="REFRESH_EXPIRED",
                is_recoverable=False,
            )

        self._set_state(AuthState.REFRESHING)

        try:
            # Mock token refresh (in production, call auth server)
            await asyncio.sleep(0.1)

            new_token = AuthToken(
                access_token=secrets.token_urlsafe(32),
                expires_at=datetime.now() + timedelta(hours=1),
                refresh_token=self._token.refresh_token,
                refresh_expires_at=self._token.refresh_expires_at,
                scope=self._token.scope,
            )

            self._token = new_token
            self._set_state(AuthState.AUTHENTICATED)
            logger.info("Token refreshed successfully")
            return new_token

        except Exception as e:
            self._set_state(AuthState.EXPIRED)
            raise AuthError(f"Token refresh failed: {e}")

    def _start_refresh_task(self) -> None:
        """Start automatic token refresh task."""
        if self._refresh_task and not self._refresh_task.done():
            return

        self._refresh_task = asyncio.create_task(self._auto_refresh_loop())

    async def _auto_refresh_loop(self) -> None:
        """Automatically refresh token before expiry."""
        while self._state == AuthState.AUTHENTICATED and self._token:
            try:
                # Check time until expiry
                expires_in = self._token.expires_in

                if expires_in <= 0:
                    # Token expired, try to refresh
                    await self.refresh_token()
                elif expires_in <= self.config.refresh_threshold:
                    # Refresh before expiry
                    await self.refresh_token()
                else:
                    # Wait until refresh threshold
                    wait_time = expires_in - self.config.refresh_threshold
                    await asyncio.sleep(min(wait_time, 60))

            except asyncio.CancelledError:
                break
            except AuthError as e:
                logger.warning(f"Auto-refresh failed: {e}")
                break
            except Exception as e:
                logger.error(f"Auto-refresh error: {e}")
                await asyncio.sleep(60)

    def _record_failed_attempt(self) -> None:
        """Record a failed authentication attempt."""
        self._failed_attempts += 1

        if self._failed_attempts >= self.config.max_failed_attempts:
            self._lockout_until = datetime.now() + timedelta(seconds=self.config.lockout_duration)
            self._set_state(AuthState.LOCKED)
            logger.warning(f"Account locked after {self._failed_attempts} failed attempts")

    async def logout(self) -> None:
        """Log out and clear tokens."""
        # Cancel refresh task
        if self._refresh_task:
            self._refresh_task.cancel()
            self._refresh_task = None

        # Clear tokens
        self._token = None
        self._session_start = None

        self._set_state(AuthState.UNAUTHENTICATED)
        logger.info("Logged out")

    def setup_biometric(self, biometric_type: BiometricType) -> BiometricAuth:
        """Set up biometric authentication.

        Args:
            biometric_type: Type of biometric

        Returns:
            BiometricAuth instance
        """
        self._biometric = BiometricAuth(biometric_type)
        return self._biometric

    def get_biometric(self) -> Optional[BiometricAuth]:
        """Get biometric auth instance."""
        return self._biometric

    def get_authorization_header(self) -> Optional[str]:
        """Get Authorization header value."""
        if self._token:
            return f"{self._token.token_type} {self._token.access_token}"
        return None

    def get_session_info(self) -> Dict[str, Any]:
        """Get current session information."""
        return {
            "state": self._state.value,
            "is_authenticated": self.is_authenticated,
            "is_locked": self.is_locked,
            "session_start": self._session_start.isoformat() if self._session_start else None,
            "token_expires_in": self._token.expires_in if self._token else None,
            "failed_attempts": self._failed_attempts,
            "biometric_available": self._biometric.is_available if self._biometric else False,
            "device_registered": self._device_token is not None,
        }

    def validate_token(self, token: str) -> bool:
        """Validate an access token.

        Args:
            token: Token to validate

        Returns:
            True if valid
        """
        if not self._token:
            return False

        if self._token.access_token != token:
            return False

        if self._token.is_expired:
            return False

        return True


class SessionManager:
    """Manages user sessions with timeout and activity tracking."""

    def __init__(
        self,
        timeout_seconds: int = 3600,
        require_reauth_for_sensitive: bool = True,
    ):
        """Initialize session manager.

        Args:
            timeout_seconds: Session timeout in seconds
            require_reauth_for_sensitive: Require re-auth for sensitive operations
        """
        self.timeout_seconds = timeout_seconds
        self.require_reauth_for_sensitive = require_reauth_for_sensitive
        self._last_activity: Optional[datetime] = None
        self._sensitive_auth_time: Optional[datetime] = None
        self._sensitive_timeout = 300  # 5 minutes for sensitive operations

    def record_activity(self) -> None:
        """Record user activity to extend session."""
        self._last_activity = datetime.now()

    def is_session_valid(self) -> bool:
        """Check if session is still valid."""
        if not self._last_activity:
            return False

        elapsed = (datetime.now() - self._last_activity).total_seconds()
        return elapsed < self.timeout_seconds

    def needs_sensitive_reauth(self) -> bool:
        """Check if re-authentication is needed for sensitive operations."""
        if not self.require_reauth_for_sensitive:
            return False

        if not self._sensitive_auth_time:
            return True

        elapsed = (datetime.now() - self._sensitive_auth_time).total_seconds()
        return elapsed >= self._sensitive_timeout

    def record_sensitive_auth(self) -> None:
        """Record successful authentication for sensitive operations."""
        self._sensitive_auth_time = datetime.now()

    def get_remaining_time(self) -> int:
        """Get remaining session time in seconds."""
        if not self._last_activity:
            return 0

        elapsed = (datetime.now() - self._last_activity).total_seconds()
        remaining = self.timeout_seconds - elapsed
        return max(0, int(remaining))
