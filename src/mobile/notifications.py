"""Push Notification Services for Agent OS Mobile.

Provides push notification support for mobile applications:
- Apple Push Notification service (APNs) for iOS
- Firebase Cloud Messaging (FCM) for Android
- Notification scheduling and management
- Rich notifications with actions
"""

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


class NotificationPriority(str, Enum):
    """Notification priority levels."""

    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


class NotificationCategory(str, Enum):
    """Notification categories."""

    GENERAL = "general"
    AGENT = "agent"
    TASK = "task"
    ALERT = "alert"
    MESSAGE = "message"
    SYSTEM = "system"


class DeliveryStatus(str, Enum):
    """Notification delivery status."""

    PENDING = "pending"
    SENT = "sent"
    DELIVERED = "delivered"
    READ = "read"
    FAILED = "failed"
    EXPIRED = "expired"


@dataclass
class NotificationAction:
    """Action button for notification."""

    action_id: str
    title: str
    destructive: bool = False
    requires_auth: bool = False
    foreground: bool = True
    icon: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "action_id": self.action_id,
            "title": self.title,
            "destructive": self.destructive,
            "requires_auth": self.requires_auth,
            "foreground": self.foreground,
            "icon": self.icon,
        }


@dataclass
class NotificationPayload:
    """Notification payload data."""

    title: str
    body: str
    subtitle: Optional[str] = None
    badge: Optional[int] = None
    sound: str = "default"
    category: NotificationCategory = NotificationCategory.GENERAL
    priority: NotificationPriority = NotificationPriority.NORMAL
    actions: List[NotificationAction] = field(default_factory=list)
    data: Dict[str, Any] = field(default_factory=dict)
    image_url: Optional[str] = None
    thread_id: Optional[str] = None
    collapse_key: Optional[str] = None
    mutable_content: bool = False
    content_available: bool = False
    ttl: int = 86400  # 24 hours

    def to_apns_payload(self) -> Dict[str, Any]:
        """Convert to APNs payload format."""
        aps = {
            "alert": {
                "title": self.title,
                "body": self.body,
            },
            "sound": self.sound,
        }

        if self.subtitle:
            aps["alert"]["subtitle"] = self.subtitle

        if self.badge is not None:
            aps["badge"] = self.badge

        if self.thread_id:
            aps["thread-id"] = self.thread_id

        if self.category != NotificationCategory.GENERAL:
            aps["category"] = self.category.value

        if self.mutable_content:
            aps["mutable-content"] = 1

        if self.content_available:
            aps["content-available"] = 1

        payload = {"aps": aps}
        payload.update(self.data)

        return payload

    def to_fcm_payload(self) -> Dict[str, Any]:
        """Convert to FCM payload format."""
        message = {
            "notification": {
                "title": self.title,
                "body": self.body,
            },
            "android": {
                "priority": (
                    "high"
                    if self.priority
                    in (
                        NotificationPriority.HIGH,
                        NotificationPriority.CRITICAL,
                    )
                    else "normal"
                ),
                "notification": {
                    "sound": self.sound,
                    "click_action": f"CATEGORY_{self.category.value.upper()}",
                },
                "ttl": f"{self.ttl}s",
            },
        }

        if self.image_url:
            message["notification"]["image"] = self.image_url

        if self.collapse_key:
            message["android"]["collapse_key"] = self.collapse_key

        if self.data:
            message["data"] = {k: str(v) for k, v in self.data.items()}

        return message


@dataclass
class Notification:
    """A notification instance."""

    notification_id: str
    device_id: str
    payload: NotificationPayload
    status: DeliveryStatus = DeliveryStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    sent_at: Optional[datetime] = None
    delivered_at: Optional[datetime] = None
    read_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    retry_count: int = 0
    error_message: Optional[str] = None

    def __post_init__(self):
        """Set expiry if not provided."""
        if not self.expires_at:
            self.expires_at = self.created_at + timedelta(seconds=self.payload.ttl)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "notification_id": self.notification_id,
            "device_id": self.device_id,
            "payload": {
                "title": self.payload.title,
                "body": self.payload.body,
                "category": self.payload.category.value,
                "priority": self.payload.priority.value,
            },
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "sent_at": self.sent_at.isoformat() if self.sent_at else None,
            "delivered_at": self.delivered_at.isoformat() if self.delivered_at else None,
            "read_at": self.read_at.isoformat() if self.read_at else None,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
        }


@dataclass
class NotificationConfig:
    """Push notification configuration."""

    # APNs configuration
    apns_key_id: str = ""
    apns_team_id: str = ""
    apns_bundle_id: str = ""
    apns_key_path: str = ""
    apns_production: bool = False

    # FCM configuration
    fcm_project_id: str = ""
    fcm_credentials_path: str = ""

    # General settings
    max_retry_count: int = 3
    retry_delay: float = 1.0
    batch_size: int = 100
    rate_limit: int = 1000  # per second


class APNsProvider:
    """Apple Push Notification service provider.

    Handles sending notifications to iOS devices via APNs.
    """

    def __init__(self, config: NotificationConfig):
        """Initialize APNs provider.

        Args:
            config: Notification configuration
        """
        self.config = config
        self._token: Optional[str] = None
        self._token_expires: Optional[datetime] = None
        self._connected = False

    @property
    def is_configured(self) -> bool:
        """Check if APNs is configured."""
        return bool(
            self.config.apns_key_id and self.config.apns_team_id and self.config.apns_bundle_id
        )

    async def connect(self) -> bool:
        """Establish connection to APNs.

        Returns:
            True if connected
        """
        if not self.is_configured:
            logger.warning("APNs not configured")
            return False

        # Generate JWT token for APNs
        self._token = self._generate_token()
        self._token_expires = datetime.now() + timedelta(minutes=50)
        self._connected = True

        logger.info("Connected to APNs")
        return True

    async def disconnect(self) -> None:
        """Disconnect from APNs."""
        self._token = None
        self._connected = False
        logger.info("Disconnected from APNs")

    def _generate_token(self) -> str:
        """Generate JWT token for APNs authentication."""
        # In production, use proper JWT with ES256
        import base64

        header = base64.b64encode(
            json.dumps({"alg": "ES256", "kid": self.config.apns_key_id}).encode()
        ).decode()
        payload = base64.b64encode(
            json.dumps(
                {
                    "iss": self.config.apns_team_id,
                    "iat": int(time.time()),
                }
            ).encode()
        ).decode()
        # Mock signature
        signature = base64.b64encode(b"mock_signature").decode()
        return f"{header}.{payload}.{signature}"

    async def send(
        self,
        device_token: str,
        payload: NotificationPayload,
    ) -> Tuple[bool, Optional[str]]:
        """Send notification to iOS device.

        Args:
            device_token: Device push token
            payload: Notification payload

        Returns:
            Tuple of (success, error_message)
        """
        if not self._connected:
            return False, "Not connected to APNs"

        # Refresh token if expired
        if self._token_expires and datetime.now() >= self._token_expires:
            self._token = self._generate_token()
            self._token_expires = datetime.now() + timedelta(minutes=50)

        # Convert payload
        apns_payload = payload.to_apns_payload()

        # Mock send (in production, use HTTP/2 to APNs)
        await asyncio.sleep(0.01)

        logger.debug(f"Sent APNs notification to {device_token[:8]}...")
        return True, None

    async def send_batch(
        self,
        notifications: List[Tuple[str, NotificationPayload]],
    ) -> List[Tuple[str, bool, Optional[str]]]:
        """Send batch of notifications.

        Args:
            notifications: List of (device_token, payload) tuples

        Returns:
            List of (device_token, success, error_message) tuples
        """
        results = []
        for device_token, payload in notifications:
            success, error = await self.send(device_token, payload)
            results.append((device_token, success, error))
        return results


class FCMProvider:
    """Firebase Cloud Messaging provider.

    Handles sending notifications to Android devices via FCM.
    """

    def __init__(self, config: NotificationConfig):
        """Initialize FCM provider.

        Args:
            config: Notification configuration
        """
        self.config = config
        self._access_token: Optional[str] = None
        self._token_expires: Optional[datetime] = None
        self._connected = False

    @property
    def is_configured(self) -> bool:
        """Check if FCM is configured."""
        return bool(self.config.fcm_project_id)

    async def connect(self) -> bool:
        """Establish connection to FCM.

        Returns:
            True if connected
        """
        if not self.is_configured:
            logger.warning("FCM not configured")
            return False

        # Get OAuth access token
        self._access_token = await self._get_access_token()
        self._token_expires = datetime.now() + timedelta(minutes=55)
        self._connected = True

        logger.info("Connected to FCM")
        return True

    async def disconnect(self) -> None:
        """Disconnect from FCM."""
        self._access_token = None
        self._connected = False
        logger.info("Disconnected from FCM")

    async def _get_access_token(self) -> str:
        """Get OAuth access token for FCM."""
        # In production, use service account credentials
        return "mock_access_token"

    async def send(
        self,
        device_token: str,
        payload: NotificationPayload,
    ) -> Tuple[bool, Optional[str]]:
        """Send notification to Android device.

        Args:
            device_token: Device FCM token
            payload: Notification payload

        Returns:
            Tuple of (success, error_message)
        """
        if not self._connected:
            return False, "Not connected to FCM"

        # Refresh token if expired
        if self._token_expires and datetime.now() >= self._token_expires:
            self._access_token = await self._get_access_token()
            self._token_expires = datetime.now() + timedelta(minutes=55)

        # Convert payload
        fcm_payload = payload.to_fcm_payload()
        fcm_payload["token"] = device_token

        # Mock send (in production, use FCM HTTP v1 API)
        await asyncio.sleep(0.01)

        logger.debug(f"Sent FCM notification to {device_token[:8]}...")
        return True, None

    async def send_batch(
        self,
        notifications: List[Tuple[str, NotificationPayload]],
    ) -> List[Tuple[str, bool, Optional[str]]]:
        """Send batch of notifications.

        Args:
            notifications: List of (device_token, payload) tuples

        Returns:
            List of (device_token, success, error_message) tuples
        """
        results = []
        for device_token, payload in notifications:
            success, error = await self.send(device_token, payload)
            results.append((device_token, success, error))
        return results

    async def send_to_topic(
        self,
        topic: str,
        payload: NotificationPayload,
    ) -> Tuple[bool, Optional[str]]:
        """Send notification to topic subscribers.

        Args:
            topic: Topic name
            payload: Notification payload

        Returns:
            Tuple of (success, error_message)
        """
        if not self._connected:
            return False, "Not connected to FCM"

        # Mock send
        await asyncio.sleep(0.01)

        logger.debug(f"Sent FCM notification to topic: {topic}")
        return True, None


class PushNotificationService:
    """Unified push notification service.

    Features:
    - Multi-platform support (iOS/Android)
    - Notification queuing and batching
    - Delivery tracking
    - Retry with backoff
    """

    def __init__(self, config: Optional[NotificationConfig] = None):
        """Initialize push notification service.

        Args:
            config: Notification configuration
        """
        self.config = config or NotificationConfig()
        self._apns = APNsProvider(self.config)
        self._fcm = FCMProvider(self.config)
        self._notifications: Dict[str, Notification] = {}
        self._device_tokens: Dict[str, Dict[str, str]] = {}  # device_id -> {platform, token}
        self._callbacks: Dict[str, List[Callable[[Notification], None]]] = {
            "sent": [],
            "delivered": [],
            "failed": [],
        }
        self._running = False
        self._queue: asyncio.Queue = asyncio.Queue()
        self._worker_task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        """Start the notification service."""
        await self._apns.connect()
        await self._fcm.connect()
        self._running = True
        self._worker_task = asyncio.create_task(self._process_queue())
        logger.info("Push notification service started")

    async def stop(self) -> None:
        """Stop the notification service."""
        self._running = False
        if self._worker_task:
            self._worker_task.cancel()
            self._worker_task = None
        await self._apns.disconnect()
        await self._fcm.disconnect()
        logger.info("Push notification service stopped")

    def register_device(
        self,
        device_id: str,
        platform: str,
        push_token: str,
    ) -> None:
        """Register a device for push notifications.

        Args:
            device_id: Device identifier
            platform: Platform type (ios/android)
            push_token: Push notification token
        """
        self._device_tokens[device_id] = {
            "platform": platform.lower(),
            "token": push_token,
        }
        logger.info(f"Registered device for push: {device_id}")

    def unregister_device(self, device_id: str) -> None:
        """Unregister a device from push notifications.

        Args:
            device_id: Device identifier
        """
        if device_id in self._device_tokens:
            del self._device_tokens[device_id]
            logger.info(f"Unregistered device from push: {device_id}")

    async def send(
        self,
        device_id: str,
        payload: NotificationPayload,
    ) -> Notification:
        """Send a push notification.

        Args:
            device_id: Target device ID
            payload: Notification payload

        Returns:
            Notification instance
        """
        notification = Notification(
            notification_id=str(uuid.uuid4()),
            device_id=device_id,
            payload=payload,
        )
        self._notifications[notification.notification_id] = notification

        # Queue for delivery
        await self._queue.put(notification)

        return notification

    async def send_to_many(
        self,
        device_ids: List[str],
        payload: NotificationPayload,
    ) -> List[Notification]:
        """Send notification to multiple devices.

        Args:
            device_ids: List of device IDs
            payload: Notification payload

        Returns:
            List of notification instances
        """
        notifications = []
        for device_id in device_ids:
            notification = await self.send(device_id, payload)
            notifications.append(notification)
        return notifications

    async def _process_queue(self) -> None:
        """Process notification queue."""
        while self._running:
            try:
                # Get notification from queue
                notification = await asyncio.wait_for(self._queue.get(), timeout=1.0)

                await self._deliver(notification)

            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Queue processing error: {e}")

    async def _deliver(self, notification: Notification) -> None:
        """Deliver a notification.

        Args:
            notification: Notification to deliver
        """
        device_info = self._device_tokens.get(notification.device_id)

        if not device_info:
            notification.status = DeliveryStatus.FAILED
            notification.error_message = "Device not registered"
            self._notify_callbacks("failed", notification)
            return

        platform = device_info["platform"]
        token = device_info["token"]

        try:
            if platform == "ios":
                success, error = await self._apns.send(token, notification.payload)
            elif platform == "android":
                success, error = await self._fcm.send(token, notification.payload)
            else:
                success, error = False, f"Unknown platform: {platform}"

            if success:
                notification.status = DeliveryStatus.SENT
                notification.sent_at = datetime.now()
                self._notify_callbacks("sent", notification)
            else:
                notification.retry_count += 1
                if notification.retry_count < self.config.max_retry_count:
                    # Re-queue for retry
                    await asyncio.sleep(self.config.retry_delay * notification.retry_count)
                    await self._queue.put(notification)
                else:
                    notification.status = DeliveryStatus.FAILED
                    notification.error_message = error
                    self._notify_callbacks("failed", notification)

        except Exception as e:
            notification.status = DeliveryStatus.FAILED
            notification.error_message = str(e)
            self._notify_callbacks("failed", notification)

    def on_sent(self, callback: Callable[[Notification], None]) -> None:
        """Register callback for sent notifications."""
        self._callbacks["sent"].append(callback)

    def on_delivered(self, callback: Callable[[Notification], None]) -> None:
        """Register callback for delivered notifications."""
        self._callbacks["delivered"].append(callback)

    def on_failed(self, callback: Callable[[Notification], None]) -> None:
        """Register callback for failed notifications."""
        self._callbacks["failed"].append(callback)

    def _notify_callbacks(self, event: str, notification: Notification) -> None:
        """Notify registered callbacks.

        Args:
            event: Event type
            notification: Notification instance
        """
        for callback in self._callbacks.get(event, []):
            try:
                callback(notification)
            except Exception as e:
                logger.warning(f"Callback error: {e}")

    def mark_delivered(self, notification_id: str) -> bool:
        """Mark notification as delivered.

        Args:
            notification_id: Notification ID

        Returns:
            True if marked
        """
        notification = self._notifications.get(notification_id)
        if notification:
            notification.status = DeliveryStatus.DELIVERED
            notification.delivered_at = datetime.now()
            self._notify_callbacks("delivered", notification)
            return True
        return False

    def mark_read(self, notification_id: str) -> bool:
        """Mark notification as read.

        Args:
            notification_id: Notification ID

        Returns:
            True if marked
        """
        notification = self._notifications.get(notification_id)
        if notification:
            notification.status = DeliveryStatus.READ
            notification.read_at = datetime.now()
            return True
        return False

    def get_notification(self, notification_id: str) -> Optional[Notification]:
        """Get notification by ID."""
        return self._notifications.get(notification_id)

    def get_notifications_for_device(
        self,
        device_id: str,
        status: Optional[DeliveryStatus] = None,
    ) -> List[Notification]:
        """Get notifications for a device.

        Args:
            device_id: Device ID
            status: Filter by status

        Returns:
            List of notifications
        """
        notifications = [n for n in self._notifications.values() if n.device_id == device_id]

        if status:
            notifications = [n for n in notifications if n.status == status]

        return sorted(notifications, key=lambda n: n.created_at, reverse=True)

    def get_stats(self) -> Dict[str, Any]:
        """Get notification statistics."""
        status_counts = {}
        for status in DeliveryStatus:
            status_counts[status.value] = sum(
                1 for n in self._notifications.values() if n.status == status
            )

        return {
            "total_notifications": len(self._notifications),
            "registered_devices": len(self._device_tokens),
            "queue_size": self._queue.qsize(),
            "status_counts": status_counts,
        }

    def clear_old_notifications(self, older_than_days: int = 30) -> int:
        """Clear old notifications.

        Args:
            older_than_days: Remove notifications older than this

        Returns:
            Number of notifications removed
        """
        cutoff = datetime.now() - timedelta(days=older_than_days)
        old_ids = [n.notification_id for n in self._notifications.values() if n.created_at < cutoff]

        for notification_id in old_ids:
            del self._notifications[notification_id]

        return len(old_ids)
