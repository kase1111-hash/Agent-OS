"""Mobile Applications Module for Agent OS.

Provides backend services and client SDKs for iOS and Android applications:
- API client for secure communication with Agent OS
- VPN tunnel integration for encrypted connections
- Push notification services (APNs, FCM)
- Authentication with device tokens and biometrics
- Offline storage and data synchronization
- Platform-specific configurations
"""

from .platform import (
    Platform,
    PlatformType,
    DeviceInfo,
    DeviceCapability,
    PlatformConfig,
    get_platform_config,
)
from .client import (
    MobileClient,
    ClientConfig,
    ConnectionState,
    ApiError,
    NetworkError,
)
from .vpn import (
    VPNTunnel,
    VPNConfig,
    VPNState,
    TunnelProtocol,
    VPNError,
)
from .auth import (
    MobileAuth,
    AuthConfig,
    AuthToken,
    DeviceToken,
    BiometricAuth,
    AuthState,
)
from .notifications import (
    PushNotificationService,
    NotificationConfig,
    Notification,
    NotificationPayload,
    APNsProvider,
    FCMProvider,
)
from .storage import (
    OfflineStorage,
    StorageConfig,
    SyncManager,
    SyncState,
    CacheEntry,
)
from .api import (
    MobileAPI,
    APIEndpoint,
    APIRequest,
    APIResponse,
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
