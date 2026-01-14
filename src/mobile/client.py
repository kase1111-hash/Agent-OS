"""Mobile API Client for Agent OS.

Provides secure HTTP client for mobile applications:
- Connection management with retry logic
- Request/response handling with serialization
- Certificate pinning support
- Offline request queuing
"""

import asyncio
import hashlib
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple
from urllib.parse import urljoin

logger = logging.getLogger(__name__)


class ConnectionState(str, Enum):
    """Client connection states."""

    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    ERROR = "error"


class ApiError(Exception):
    """API error with status code and details."""

    def __init__(
        self,
        message: str,
        status_code: int = 0,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message)
        self.status_code = status_code
        self.error_code = error_code
        self.details = details or {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "message": str(self),
            "status_code": self.status_code,
            "error_code": self.error_code,
            "details": self.details,
        }


class NetworkError(Exception):
    """Network connectivity error."""

    def __init__(
        self,
        message: str,
        is_temporary: bool = True,
        retry_after: Optional[int] = None,
    ):
        super().__init__(message)
        self.is_temporary = is_temporary
        self.retry_after = retry_after


@dataclass
class ClientConfig:
    """Configuration for mobile client."""

    base_url: str = "https://api.agentos.local"
    timeout: int = 30
    connect_timeout: int = 10
    retry_count: int = 3
    retry_delay: float = 1.0
    retry_backoff: float = 2.0
    max_retry_delay: float = 30.0
    enable_compression: bool = True
    enable_certificate_pinning: bool = True
    pinned_certificate_hashes: List[str] = field(default_factory=list)
    user_agent: str = "AgentOS-Mobile/1.0"
    device_id: Optional[str] = None
    auth_token: Optional[str] = None
    headers: Dict[str, str] = field(default_factory=dict)
    verify_ssl: bool = True

    def get_headers(self) -> Dict[str, str]:
        """Get request headers."""
        headers = {
            "User-Agent": self.user_agent,
            "Content-Type": "application/json",
            "Accept": "application/json",
            "X-Client-Version": "1.0.0",
        }

        if self.device_id:
            headers["X-Device-ID"] = self.device_id

        if self.auth_token:
            headers["Authorization"] = f"Bearer {self.auth_token}"

        if self.enable_compression:
            headers["Accept-Encoding"] = "gzip, deflate"

        headers.update(self.headers)
        return headers


@dataclass
class QueuedRequest:
    """A queued offline request."""

    request_id: str
    method: str
    endpoint: str
    data: Optional[Dict[str, Any]] = None
    headers: Dict[str, str] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    retry_count: int = 0
    priority: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "request_id": self.request_id,
            "method": self.method,
            "endpoint": self.endpoint,
            "data": self.data,
            "headers": self.headers,
            "created_at": self.created_at.isoformat(),
            "retry_count": self.retry_count,
            "priority": self.priority,
        }


class MobileClient:
    """HTTP client for mobile applications.

    Features:
    - Automatic retry with exponential backoff
    - Certificate pinning
    - Request queuing for offline mode
    - Connection state management
    """

    def __init__(self, config: Optional[ClientConfig] = None):
        """Initialize mobile client.

        Args:
            config: Client configuration
        """
        self.config = config or ClientConfig()
        self._state = ConnectionState.DISCONNECTED
        self._state_callbacks: List[Callable[[ConnectionState], None]] = []
        self._request_queue: List[QueuedRequest] = []
        self._is_online = True
        self._last_request_time: Optional[datetime] = None
        self._request_count = 0
        self._error_count = 0

    @property
    def state(self) -> ConnectionState:
        """Get current connection state."""
        return self._state

    @property
    def is_online(self) -> bool:
        """Check if client is online."""
        return self._is_online and self._state == ConnectionState.CONNECTED

    def set_auth_token(self, token: str) -> None:
        """Set authentication token."""
        self.config.auth_token = token

    def clear_auth_token(self) -> None:
        """Clear authentication token."""
        self.config.auth_token = None

    def on_state_change(self, callback: Callable[[ConnectionState], None]) -> None:
        """Register a state change callback."""
        self._state_callbacks.append(callback)

    def _set_state(self, state: ConnectionState) -> None:
        """Set connection state and notify callbacks."""
        if self._state != state:
            self._state = state
            for callback in self._state_callbacks:
                try:
                    callback(state)
                except Exception as e:
                    logger.warning(f"State callback error: {e}")

    async def connect(self) -> bool:
        """Establish connection to the API server.

        Returns:
            True if connected successfully
        """
        self._set_state(ConnectionState.CONNECTING)

        try:
            # Perform health check
            response = await self._make_request("GET", "/health", skip_auth=True)

            if response.get("status") == "ok":
                self._set_state(ConnectionState.CONNECTED)
                self._is_online = True
                return True
            else:
                self._set_state(ConnectionState.ERROR)
                return False

        except Exception as e:
            logger.error(f"Connection failed: {e}")
            self._set_state(ConnectionState.ERROR)
            self._is_online = False
            return False

    async def disconnect(self) -> None:
        """Disconnect from the API server."""
        self._set_state(ConnectionState.DISCONNECTED)
        self._is_online = False

    async def get(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """Make GET request.

        Args:
            endpoint: API endpoint
            params: Query parameters
            headers: Additional headers

        Returns:
            Response data
        """
        return await self._make_request("GET", endpoint, params=params, headers=headers)

    async def post(
        self,
        endpoint: str,
        data: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """Make POST request.

        Args:
            endpoint: API endpoint
            data: Request body
            headers: Additional headers

        Returns:
            Response data
        """
        return await self._make_request("POST", endpoint, data=data, headers=headers)

    async def put(
        self,
        endpoint: str,
        data: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """Make PUT request.

        Args:
            endpoint: API endpoint
            data: Request body
            headers: Additional headers

        Returns:
            Response data
        """
        return await self._make_request("PUT", endpoint, data=data, headers=headers)

    async def delete(
        self,
        endpoint: str,
        headers: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """Make DELETE request.

        Args:
            endpoint: API endpoint
            headers: Additional headers

        Returns:
            Response data
        """
        return await self._make_request("DELETE", endpoint, headers=headers)

    async def _make_request(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        skip_auth: bool = False,
        retry: int = 0,
    ) -> Dict[str, Any]:
        """Make an HTTP request with retry logic.

        Args:
            method: HTTP method
            endpoint: API endpoint
            data: Request body
            params: Query parameters
            headers: Additional headers
            skip_auth: Skip authentication header
            retry: Current retry count

        Returns:
            Response data
        """
        url = urljoin(self.config.base_url, endpoint)

        # Build headers
        request_headers = self.config.get_headers()
        if skip_auth:
            request_headers.pop("Authorization", None)
        if headers:
            request_headers.update(headers)

        # Add request ID for tracking
        request_id = str(uuid.uuid4())[:8]
        request_headers["X-Request-ID"] = request_id

        try:
            # Simulate HTTP request (in production would use aiohttp)
            response = await self._send_request(method, url, data, params, request_headers)

            self._request_count += 1
            self._last_request_time = datetime.now()

            return response

        except NetworkError as e:
            self._error_count += 1

            # Check if we should retry
            if e.is_temporary and retry < self.config.retry_count:
                delay = min(
                    self.config.retry_delay * (self.config.retry_backoff**retry),
                    self.config.max_retry_delay,
                )

                if e.retry_after:
                    delay = max(delay, e.retry_after)

                logger.info(f"Retrying request in {delay}s (attempt {retry + 1})")
                await asyncio.sleep(delay)

                return await self._make_request(
                    method, endpoint, data, params, headers, skip_auth, retry + 1
                )

            # Queue for offline if appropriate
            if self._should_queue(method, endpoint):
                self._queue_request(method, endpoint, data, request_headers)

            raise

        except ApiError:
            self._error_count += 1
            raise

    async def _send_request(
        self,
        method: str,
        url: str,
        data: Optional[Dict[str, Any]],
        params: Optional[Dict[str, Any]],
        headers: Dict[str, str],
    ) -> Dict[str, Any]:
        """Send HTTP request (mock implementation).

        In production, this would use aiohttp or similar.
        """
        # For testing/development, return mock responses
        if "/health" in url:
            return {"status": "ok", "timestamp": datetime.now().isoformat()}

        if "/auth" in url:
            return {
                "access_token": "mock_token",
                "refresh_token": "mock_refresh",
                "expires_in": 3600,
            }

        # Default mock response
        return {
            "success": True,
            "data": {},
            "timestamp": datetime.now().isoformat(),
        }

    def _should_queue(self, method: str, endpoint: str) -> bool:
        """Check if request should be queued for offline."""
        # Only queue safe methods that can be retried
        if method not in ("POST", "PUT", "DELETE"):
            return False

        # Don't queue auth requests
        if "auth" in endpoint.lower():
            return False

        return True

    def _queue_request(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict[str, Any]],
        headers: Dict[str, str],
    ) -> str:
        """Queue a request for later execution.

        Returns:
            Request ID
        """
        request = QueuedRequest(
            request_id=str(uuid.uuid4()),
            method=method,
            endpoint=endpoint,
            data=data,
            headers=headers,
        )
        self._request_queue.append(request)
        logger.info(f"Queued request: {request.request_id}")
        return request.request_id

    async def process_queue(self) -> List[Tuple[str, bool]]:
        """Process queued requests.

        Returns:
            List of (request_id, success) tuples
        """
        results = []

        while self._request_queue:
            request = self._request_queue.pop(0)

            try:
                await self._make_request(
                    request.method,
                    request.endpoint,
                    request.data,
                    headers=request.headers,
                )
                results.append((request.request_id, True))
                logger.info(f"Processed queued request: {request.request_id}")

            except Exception as e:
                logger.error(f"Failed to process queued request: {e}")
                request.retry_count += 1

                # Re-queue if retries remain
                if request.retry_count < 3:
                    self._request_queue.append(request)
                results.append((request.request_id, False))

        return results

    def get_queue_size(self) -> int:
        """Get number of queued requests."""
        return len(self._request_queue)

    def clear_queue(self) -> int:
        """Clear request queue.

        Returns:
            Number of requests cleared
        """
        count = len(self._request_queue)
        self._request_queue.clear()
        return count

    def get_stats(self) -> Dict[str, Any]:
        """Get client statistics."""
        return {
            "state": self._state.value,
            "is_online": self._is_online,
            "request_count": self._request_count,
            "error_count": self._error_count,
            "queue_size": len(self._request_queue),
            "last_request": (
                self._last_request_time.isoformat() if self._last_request_time else None
            ),
        }

    def verify_certificate(self, cert_hash: str) -> bool:
        """Verify certificate hash against pinned certificates.

        Args:
            cert_hash: SHA-256 hash of the certificate

        Returns:
            True if certificate is valid
        """
        if not self.config.enable_certificate_pinning:
            return True

        if not self.config.pinned_certificate_hashes:
            return True

        return cert_hash in self.config.pinned_certificate_hashes

    @staticmethod
    def compute_certificate_hash(cert_der: bytes) -> str:
        """Compute SHA-256 hash of certificate.

        Args:
            cert_der: Certificate in DER format

        Returns:
            Hex-encoded SHA-256 hash
        """
        return hashlib.sha256(cert_der).hexdigest()
