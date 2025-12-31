"""Mobile Applications Module for Agent OS.

Provides backend services and client SDKs for iOS and Android applications:
- API client for secure communication with Agent OS
- VPN tunnel integration for encrypted connections
- Push notification services (APNs, FCM)
- Authentication with device tokens and biometrics
- Offline storage and data synchronization
- Platform-specific configurations
"""

from .api import (
    APIEndpoint,
    APIRequest,
    APIResponse,
    MobileAPI,
)
from .auth import (
    AuthConfig,
    AuthState,
    AuthToken,
    BiometricAuth,
    DeviceToken,
    MobileAuth,
)
from .client import (
    ApiError,
    ClientConfig,
    ConnectionState,
    MobileClient,
    NetworkError,
)
from .notifications import (
    APNsProvider,
    FCMProvider,
    Notification,
    NotificationConfig,
    NotificationPayload,
    PushNotificationService,
)
from .platform import (
    DeviceCapability,
    DeviceInfo,
    Platform,
    PlatformConfig,
    PlatformType,
    get_platform_config,
)
from .storage import (
    CacheEntry,
    OfflineStorage,
    StorageConfig,
    SyncManager,
    SyncState,
)
from .vpn import (
    TunnelProtocol,
    VPNConfig,
    VPNError,
    VPNState,
    VPNTunnel,
)

__all__ = [
    # Platform
    "Platform",
    "PlatformType",
    "DeviceInfo",
    "DeviceCapability",
    "PlatformConfig",
    "get_platform_config",
    # Client
    "MobileClient",
    "ClientConfig",
    "ConnectionState",
    "ApiError",
    "NetworkError",
    # VPN
    "VPNTunnel",
    "VPNConfig",
    "VPNState",
    "TunnelProtocol",
    "VPNError",
    # Auth
    "MobileAuth",
    "AuthConfig",
    "AuthToken",
    "DeviceToken",
    "BiometricAuth",
    "AuthState",
    # Notifications
    "PushNotificationService",
    "NotificationConfig",
    "Notification",
    "NotificationPayload",
    "APNsProvider",
    "FCMProvider",
    # Storage
    "OfflineStorage",
    "StorageConfig",
    "SyncManager",
    "SyncState",
    "CacheEntry",
    # API
    "MobileAPI",
    "APIEndpoint",
    "APIRequest",
    "APIResponse",
]
