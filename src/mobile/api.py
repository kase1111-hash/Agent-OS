"""Mobile API Module for Agent OS.

Provides high-level API interface for mobile applications:
- RESTful endpoints abstraction
- Request/response models
- API versioning
- Rate limiting
"""

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, Generic, List, Optional, TypeVar

from .client import ApiError, ClientConfig, MobileClient

logger = logging.getLogger(__name__)

T = TypeVar("T")


class HttpMethod(str, Enum):
    """HTTP methods."""

    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    PATCH = "PATCH"
    DELETE = "DELETE"


class APIVersion(str, Enum):
    """API versions."""

    V1 = "v1"
    V2 = "v2"


@dataclass
class APIEndpoint:
    """API endpoint definition."""

    path: str
    method: HttpMethod = HttpMethod.GET
    version: APIVersion = APIVersion.V1
    requires_auth: bool = True
    rate_limit: Optional[int] = None  # requests per minute
    timeout: Optional[int] = None
    description: str = ""

    @property
    def full_path(self) -> str:
        """Get full path with version."""
        return f"/api/{self.version.value}{self.path}"


@dataclass
class APIRequest:
    """API request."""

    endpoint: APIEndpoint
    params: Dict[str, Any] = field(default_factory=dict)
    data: Optional[Dict[str, Any]] = None
    headers: Dict[str, str] = field(default_factory=dict)
    request_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "request_id": self.request_id,
            "endpoint": self.endpoint.full_path,
            "method": self.endpoint.method.value,
            "params": self.params,
            "data": self.data,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class APIResponse(Generic[T]):
    """API response."""

    success: bool
    data: Optional[T] = None
    error: Optional[str] = None
    error_code: Optional[str] = None
    request_id: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "data": self.data,
            "error": self.error,
            "error_code": self.error_code,
            "request_id": self.request_id,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }


class RateLimiter:
    """Simple rate limiter for API calls."""

    def __init__(self, requests_per_minute: int = 60):
        """Initialize rate limiter.

        Args:
            requests_per_minute: Maximum requests per minute
        """
        self.requests_per_minute = requests_per_minute
        self._requests: List[datetime] = []

    def can_proceed(self) -> bool:
        """Check if request can proceed."""
        now = datetime.now()
        cutoff = now - timedelta(minutes=1)

        # Remove old requests
        self._requests = [r for r in self._requests if r > cutoff]

        return len(self._requests) < self.requests_per_minute

    def record_request(self) -> None:
        """Record a request."""
        self._requests.append(datetime.now())

    def wait_time(self) -> float:
        """Get time to wait before next request."""
        if self.can_proceed():
            return 0.0

        oldest = min(self._requests)
        wait = (oldest + timedelta(minutes=1) - datetime.now()).total_seconds()
        return max(0.0, wait)


class MobileAPI:
    """High-level Mobile API interface.

    Provides typed methods for common API operations.
    """

    # Common endpoints
    ENDPOINTS = {
        # Auth endpoints
        "login": APIEndpoint("/auth/login", HttpMethod.POST, requires_auth=False),
        "logout": APIEndpoint("/auth/logout", HttpMethod.POST),
        "refresh": APIEndpoint("/auth/refresh", HttpMethod.POST, requires_auth=False),
        "register": APIEndpoint("/auth/register", HttpMethod.POST, requires_auth=False),
        # User endpoints
        "get_profile": APIEndpoint("/users/me", HttpMethod.GET),
        "update_profile": APIEndpoint("/users/me", HttpMethod.PUT),
        "get_preferences": APIEndpoint("/users/me/preferences", HttpMethod.GET),
        "update_preferences": APIEndpoint("/users/me/preferences", HttpMethod.PUT),
        # Agent endpoints
        "list_agents": APIEndpoint("/agents", HttpMethod.GET),
        "get_agent": APIEndpoint("/agents/{agent_id}", HttpMethod.GET),
        "create_agent": APIEndpoint("/agents", HttpMethod.POST),
        "update_agent": APIEndpoint("/agents/{agent_id}", HttpMethod.PUT),
        "delete_agent": APIEndpoint("/agents/{agent_id}", HttpMethod.DELETE),
        "run_agent": APIEndpoint("/agents/{agent_id}/run", HttpMethod.POST),
        # Task endpoints
        "list_tasks": APIEndpoint("/tasks", HttpMethod.GET),
        "get_task": APIEndpoint("/tasks/{task_id}", HttpMethod.GET),
        "create_task": APIEndpoint("/tasks", HttpMethod.POST),
        "cancel_task": APIEndpoint("/tasks/{task_id}/cancel", HttpMethod.POST),
        # Workflow endpoints
        "list_workflows": APIEndpoint("/workflows", HttpMethod.GET),
        "get_workflow": APIEndpoint("/workflows/{workflow_id}", HttpMethod.GET),
        "start_workflow": APIEndpoint("/workflows/{workflow_id}/start", HttpMethod.POST),
        # Device endpoints
        "register_device": APIEndpoint("/devices", HttpMethod.POST),
        "update_device": APIEndpoint("/devices/{device_id}", HttpMethod.PUT),
        "unregister_device": APIEndpoint("/devices/{device_id}", HttpMethod.DELETE),
        # Notification endpoints
        "get_notifications": APIEndpoint("/notifications", HttpMethod.GET),
        "mark_read": APIEndpoint("/notifications/{notification_id}/read", HttpMethod.POST),
        "update_push_token": APIEndpoint("/notifications/token", HttpMethod.PUT),
        # Sync endpoints
        "sync_pull": APIEndpoint("/sync/pull", HttpMethod.POST),
        "sync_push": APIEndpoint("/sync/push", HttpMethod.POST),
        # Health endpoints
        "health": APIEndpoint("/health", HttpMethod.GET, requires_auth=False),
        "version": APIEndpoint("/version", HttpMethod.GET, requires_auth=False),
    }

    def __init__(
        self,
        client: Optional[MobileClient] = None,
        config: Optional[ClientConfig] = None,
        rate_limit: int = 60,
    ):
        """Initialize Mobile API.

        Args:
            client: HTTP client instance
            config: Client configuration
            rate_limit: Requests per minute limit
        """
        self.client = client or MobileClient(config)
        self._rate_limiter = RateLimiter(rate_limit)
        self._request_callbacks: List[Callable[[APIRequest], None]] = []
        self._response_callbacks: List[Callable[[APIResponse], None]] = []

    def on_request(self, callback: Callable[[APIRequest], None]) -> None:
        """Register request callback."""
        self._request_callbacks.append(callback)

    def on_response(self, callback: Callable[[APIResponse], None]) -> None:
        """Register response callback."""
        self._response_callbacks.append(callback)

    async def _execute(
        self,
        endpoint: APIEndpoint,
        path_params: Optional[Dict[str, str]] = None,
        query_params: Optional[Dict[str, Any]] = None,
        data: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> APIResponse:
        """Execute an API request.

        Args:
            endpoint: API endpoint
            path_params: Path parameters to substitute
            query_params: Query parameters
            data: Request body
            headers: Additional headers

        Returns:
            API response
        """
        # Check rate limit
        if not self._rate_limiter.can_proceed():
            wait_time = self._rate_limiter.wait_time()
            await asyncio.sleep(wait_time)

        # Build path
        path = endpoint.full_path
        if path_params:
            for key, value in path_params.items():
                path = path.replace(f"{{{key}}}", value)

        # Create request
        request = APIRequest(
            endpoint=endpoint,
            params=query_params or {},
            data=data,
            headers=headers or {},
        )

        # Notify callbacks
        for callback in self._request_callbacks:
            try:
                callback(request)
            except Exception as e:
                logger.warning(f"Request callback error: {e}")

        try:
            # Execute based on method
            if endpoint.method == HttpMethod.GET:
                response_data = await self.client.get(path, params=query_params, headers=headers)
            elif endpoint.method == HttpMethod.POST:
                response_data = await self.client.post(path, data=data, headers=headers)
            elif endpoint.method == HttpMethod.PUT:
                response_data = await self.client.put(path, data=data, headers=headers)
            elif endpoint.method == HttpMethod.DELETE:
                response_data = await self.client.delete(path, headers=headers)
            else:
                raise ValueError(f"Unsupported method: {endpoint.method}")

            self._rate_limiter.record_request()

            response = APIResponse(
                success=True,
                data=response_data,
                request_id=request.request_id,
            )

        except ApiError as e:
            response = APIResponse(
                success=False,
                error=str(e),
                error_code=e.error_code,
                request_id=request.request_id,
            )

        except Exception as e:
            response = APIResponse(
                success=False,
                error=str(e),
                request_id=request.request_id,
            )

        # Notify callbacks
        for callback in self._response_callbacks:
            try:
                callback(response)
            except Exception as e:
                logger.warning(f"Response callback error: {e}")

        return response

    # Auth methods

    async def login(
        self,
        username: str,
        password: str,
        device_id: Optional[str] = None,
    ) -> APIResponse:
        """Log in user.

        Args:
            username: Username
            password: Password
            device_id: Device identifier

        Returns:
            API response with tokens
        """
        return await self._execute(
            self.ENDPOINTS["login"],
            data={
                "username": username,
                "password": password,
                "device_id": device_id,
            },
        )

    async def logout(self) -> APIResponse:
        """Log out user."""
        return await self._execute(self.ENDPOINTS["logout"])

    async def refresh_token(self, refresh_token: str) -> APIResponse:
        """Refresh access token.

        Args:
            refresh_token: Refresh token

        Returns:
            API response with new tokens
        """
        return await self._execute(
            self.ENDPOINTS["refresh"],
            data={"refresh_token": refresh_token},
        )

    # User methods

    async def get_profile(self) -> APIResponse:
        """Get current user profile."""
        return await self._execute(self.ENDPOINTS["get_profile"])

    async def update_profile(self, data: Dict[str, Any]) -> APIResponse:
        """Update user profile.

        Args:
            data: Profile data to update
        """
        return await self._execute(self.ENDPOINTS["update_profile"], data=data)

    # Agent methods

    async def list_agents(
        self,
        page: int = 1,
        limit: int = 20,
        status: Optional[str] = None,
    ) -> APIResponse:
        """List agents.

        Args:
            page: Page number
            limit: Items per page
            status: Filter by status
        """
        params = {"page": page, "limit": limit}
        if status:
            params["status"] = status
        return await self._execute(
            self.ENDPOINTS["list_agents"],
            query_params=params,
        )

    async def get_agent(self, agent_id: str) -> APIResponse:
        """Get agent by ID.

        Args:
            agent_id: Agent ID
        """
        return await self._execute(
            self.ENDPOINTS["get_agent"],
            path_params={"agent_id": agent_id},
        )

    async def create_agent(self, data: Dict[str, Any]) -> APIResponse:
        """Create a new agent.

        Args:
            data: Agent data
        """
        return await self._execute(self.ENDPOINTS["create_agent"], data=data)

    async def run_agent(
        self,
        agent_id: str,
        input_data: Optional[Dict[str, Any]] = None,
    ) -> APIResponse:
        """Run an agent.

        Args:
            agent_id: Agent ID
            input_data: Input data for the agent
        """
        return await self._execute(
            self.ENDPOINTS["run_agent"],
            path_params={"agent_id": agent_id},
            data=input_data,
        )

    # Task methods

    async def list_tasks(
        self,
        page: int = 1,
        limit: int = 20,
        status: Optional[str] = None,
    ) -> APIResponse:
        """List tasks.

        Args:
            page: Page number
            limit: Items per page
            status: Filter by status
        """
        params = {"page": page, "limit": limit}
        if status:
            params["status"] = status
        return await self._execute(
            self.ENDPOINTS["list_tasks"],
            query_params=params,
        )

    async def get_task(self, task_id: str) -> APIResponse:
        """Get task by ID.

        Args:
            task_id: Task ID
        """
        return await self._execute(
            self.ENDPOINTS["get_task"],
            path_params={"task_id": task_id},
        )

    async def create_task(self, data: Dict[str, Any]) -> APIResponse:
        """Create a new task.

        Args:
            data: Task data
        """
        return await self._execute(self.ENDPOINTS["create_task"], data=data)

    async def cancel_task(self, task_id: str) -> APIResponse:
        """Cancel a task.

        Args:
            task_id: Task ID
        """
        return await self._execute(
            self.ENDPOINTS["cancel_task"],
            path_params={"task_id": task_id},
        )

    # Workflow methods

    async def list_workflows(self, page: int = 1, limit: int = 20) -> APIResponse:
        """List workflows.

        Args:
            page: Page number
            limit: Items per page
        """
        return await self._execute(
            self.ENDPOINTS["list_workflows"],
            query_params={"page": page, "limit": limit},
        )

    async def start_workflow(
        self,
        workflow_id: str,
        input_data: Optional[Dict[str, Any]] = None,
    ) -> APIResponse:
        """Start a workflow.

        Args:
            workflow_id: Workflow ID
            input_data: Input data
        """
        return await self._execute(
            self.ENDPOINTS["start_workflow"],
            path_params={"workflow_id": workflow_id},
            data=input_data,
        )

    # Device methods

    async def register_device(
        self,
        device_id: str,
        platform: str,
        push_token: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> APIResponse:
        """Register device.

        Args:
            device_id: Device ID
            platform: Platform type
            push_token: Push notification token
            metadata: Device metadata
        """
        return await self._execute(
            self.ENDPOINTS["register_device"],
            data={
                "device_id": device_id,
                "platform": platform,
                "push_token": push_token,
                "metadata": metadata or {},
            },
        )

    async def update_push_token(
        self,
        device_id: str,
        push_token: str,
    ) -> APIResponse:
        """Update push notification token.

        Args:
            device_id: Device ID
            push_token: New push token
        """
        return await self._execute(
            self.ENDPOINTS["update_push_token"],
            data={"device_id": device_id, "push_token": push_token},
        )

    # Notification methods

    async def get_notifications(
        self,
        page: int = 1,
        limit: int = 20,
        unread_only: bool = False,
    ) -> APIResponse:
        """Get notifications.

        Args:
            page: Page number
            limit: Items per page
            unread_only: Only unread notifications
        """
        params = {"page": page, "limit": limit}
        if unread_only:
            params["unread"] = True
        return await self._execute(
            self.ENDPOINTS["get_notifications"],
            query_params=params,
        )

    async def mark_notification_read(self, notification_id: str) -> APIResponse:
        """Mark notification as read.

        Args:
            notification_id: Notification ID
        """
        return await self._execute(
            self.ENDPOINTS["mark_read"],
            path_params={"notification_id": notification_id},
        )

    # Sync methods

    async def sync_pull(
        self,
        entity_types: List[str],
        since: Optional[datetime] = None,
    ) -> APIResponse:
        """Pull changes from server.

        Args:
            entity_types: Entity types to sync
            since: Get changes since this time
        """
        return await self._execute(
            self.ENDPOINTS["sync_pull"],
            data={
                "entity_types": entity_types,
                "since": since.isoformat() if since else None,
            },
        )

    async def sync_push(self, changes: List[Dict[str, Any]]) -> APIResponse:
        """Push changes to server.

        Args:
            changes: List of changes to push
        """
        return await self._execute(
            self.ENDPOINTS["sync_push"],
            data={"changes": changes},
        )

    # Health methods

    async def health_check(self) -> APIResponse:
        """Check API health."""
        return await self._execute(self.ENDPOINTS["health"])

    async def get_version(self) -> APIResponse:
        """Get API version."""
        return await self._execute(self.ENDPOINTS["version"])
