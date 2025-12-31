"""Platform Abstraction Layer for Mobile Applications.

Provides unified interfaces for iOS and Android platforms:
- Device information and capabilities
- Platform-specific configurations
- Feature detection
"""

import logging
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


class PlatformType(str, Enum):
    """Supported mobile platforms."""

    IOS = "ios"
    ANDROID = "android"
    WEB = "web"  # Progressive Web App
    UNKNOWN = "unknown"


class DeviceCapability(str, Enum):
    """Device capabilities."""

    BIOMETRICS = "biometrics"
    FACE_ID = "face_id"
    TOUCH_ID = "touch_id"
    FINGERPRINT = "fingerprint"
    PUSH_NOTIFICATIONS = "push_notifications"
    BACKGROUND_REFRESH = "background_refresh"
    SECURE_ENCLAVE = "secure_enclave"
    NFC = "nfc"
    BLUETOOTH = "bluetooth"
    CAMERA = "camera"
    MICROPHONE = "microphone"
    LOCATION = "location"
    ACCELEROMETER = "accelerometer"
    GYROSCOPE = "gyroscope"


@dataclass
class DeviceInfo:
    """Information about the mobile device."""

    device_id: str
    platform: PlatformType
    os_version: str
    app_version: str
    device_model: str = ""
    device_name: str = ""
    manufacturer: str = ""
    screen_width: int = 0
    screen_height: int = 0
    screen_density: float = 1.0
    language: str = "en"
    timezone: str = "UTC"
    capabilities: Set[DeviceCapability] = field(default_factory=set)
    push_token: Optional[str] = None
    is_tablet: bool = False
    is_jailbroken: bool = False
    is_emulator: bool = False
    registered_at: datetime = field(default_factory=datetime.now)
    last_seen: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate device info."""
        if not self.device_id:
            self.device_id = str(uuid.uuid4())
        if isinstance(self.platform, str):
            self.platform = PlatformType(self.platform)
        if isinstance(self.capabilities, (list, tuple)):
            self.capabilities = {
                DeviceCapability(c) if isinstance(c, str) else c for c in self.capabilities
            }

    def has_capability(self, capability: DeviceCapability) -> bool:
        """Check if device has a capability."""
        return capability in self.capabilities

    def supports_biometrics(self) -> bool:
        """Check if device supports biometric authentication."""
        return any(
            cap in self.capabilities
            for cap in [
                DeviceCapability.BIOMETRICS,
                DeviceCapability.FACE_ID,
                DeviceCapability.TOUCH_ID,
                DeviceCapability.FINGERPRINT,
            ]
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "device_id": self.device_id,
            "platform": self.platform.value,
            "os_version": self.os_version,
            "app_version": self.app_version,
            "device_model": self.device_model,
            "device_name": self.device_name,
            "manufacturer": self.manufacturer,
            "screen_width": self.screen_width,
            "screen_height": self.screen_height,
            "screen_density": self.screen_density,
            "language": self.language,
            "timezone": self.timezone,
            "capabilities": [c.value for c in self.capabilities],
            "push_token": self.push_token,
            "is_tablet": self.is_tablet,
            "is_jailbroken": self.is_jailbroken,
            "is_emulator": self.is_emulator,
            "registered_at": self.registered_at.isoformat(),
            "last_seen": self.last_seen.isoformat(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DeviceInfo":
        """Create from dictionary."""
        if isinstance(data.get("registered_at"), str):
            data["registered_at"] = datetime.fromisoformat(data["registered_at"])
        if isinstance(data.get("last_seen"), str):
            data["last_seen"] = datetime.fromisoformat(data["last_seen"])
        if isinstance(data.get("capabilities"), list):
            data["capabilities"] = {DeviceCapability(c) for c in data["capabilities"]}
        return cls(**data)

    @classmethod
    def from_user_agent(cls, user_agent: str, device_id: Optional[str] = None) -> "DeviceInfo":
        """Parse device info from User-Agent header."""
        platform = PlatformType.UNKNOWN
        os_version = ""
        device_model = ""

        ua_lower = user_agent.lower()

        # Detect iOS
        if "iphone" in ua_lower or "ipad" in ua_lower:
            platform = PlatformType.IOS
            # Extract iOS version
            ios_match = re.search(r"os (\d+[_\.]\d+[_\.]?\d*)", ua_lower)
            if ios_match:
                os_version = ios_match.group(1).replace("_", ".")
            # Detect model
            if "ipad" in ua_lower:
                device_model = "iPad"
            else:
                device_model = "iPhone"

        # Detect Android
        elif "android" in ua_lower:
            platform = PlatformType.ANDROID
            # Extract Android version
            android_match = re.search(r"android (\d+\.?\d*\.?\d*)", ua_lower)
            if android_match:
                os_version = android_match.group(1)
            # Try to get device model
            model_match = re.search(r";\s*([^;)]+)\s*build", ua_lower)
            if model_match:
                device_model = model_match.group(1).strip()

        return cls(
            device_id=device_id or str(uuid.uuid4()),
            platform=platform,
            os_version=os_version,
            app_version="1.0.0",
            device_model=device_model,
            is_tablet="ipad" in ua_lower or "tablet" in ua_lower,
        )


@dataclass
class PlatformConfig:
    """Platform-specific configuration."""

    platform: PlatformType
    api_base_url: str = "https://api.agentos.local"
    vpn_server: str = ""
    vpn_port: int = 443
    min_os_version: str = ""
    min_app_version: str = "1.0.0"
    push_enabled: bool = True
    background_refresh_interval: int = 900  # 15 minutes
    cache_size_mb: int = 100
    sync_interval: int = 300  # 5 minutes
    timeout_seconds: int = 30
    retry_count: int = 3
    certificate_pinning: bool = True
    pinned_certificates: List[str] = field(default_factory=list)
    feature_flags: Dict[str, bool] = field(default_factory=dict)
    custom_settings: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Set platform-specific defaults."""
        if isinstance(self.platform, str):
            self.platform = PlatformType(self.platform)

        if not self.min_os_version:
            if self.platform == PlatformType.IOS:
                self.min_os_version = "14.0"
            elif self.platform == PlatformType.ANDROID:
                self.min_os_version = "8.0"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "platform": self.platform.value,
            "api_base_url": self.api_base_url,
            "vpn_server": self.vpn_server,
            "vpn_port": self.vpn_port,
            "min_os_version": self.min_os_version,
            "min_app_version": self.min_app_version,
            "push_enabled": self.push_enabled,
            "background_refresh_interval": self.background_refresh_interval,
            "cache_size_mb": self.cache_size_mb,
            "sync_interval": self.sync_interval,
            "timeout_seconds": self.timeout_seconds,
            "retry_count": self.retry_count,
            "certificate_pinning": self.certificate_pinning,
            "pinned_certificates": self.pinned_certificates,
            "feature_flags": self.feature_flags,
            "custom_settings": self.custom_settings,
        }


class Platform:
    """Platform abstraction for mobile applications."""

    # Default configurations for each platform
    _configs: Dict[PlatformType, PlatformConfig] = {}

    @classmethod
    def register_config(cls, config: PlatformConfig) -> None:
        """Register a platform configuration."""
        cls._configs[config.platform] = config

    @classmethod
    def get_config(cls, platform: PlatformType) -> PlatformConfig:
        """Get configuration for a platform."""
        if platform not in cls._configs:
            cls._configs[platform] = PlatformConfig(platform=platform)
        return cls._configs[platform]

    @classmethod
    def is_supported(cls, device: DeviceInfo) -> bool:
        """Check if a device is supported."""
        if device.platform == PlatformType.UNKNOWN:
            return False

        config = cls.get_config(device.platform)

        # Check minimum OS version
        if config.min_os_version and device.os_version:
            if not cls._version_gte(device.os_version, config.min_os_version):
                return False

        # Check minimum app version
        if config.min_app_version and device.app_version:
            if not cls._version_gte(device.app_version, config.min_app_version):
                return False

        # Check for jailbreak/root (configurable)
        if device.is_jailbroken and not config.feature_flags.get("allow_jailbroken", False):
            return False

        # Check for emulator (configurable)
        if device.is_emulator and not config.feature_flags.get("allow_emulator", True):
            return False

        return True

    @staticmethod
    def _version_gte(version: str, required: str) -> bool:
        """Check if version is greater than or equal to required."""
        try:

            def parse_version(v: str) -> List[int]:
                return [int(x) for x in re.split(r"[._-]", v) if x.isdigit()]

            v_parts = parse_version(version)
            r_parts = parse_version(required)

            # Pad with zeros
            while len(v_parts) < len(r_parts):
                v_parts.append(0)
            while len(r_parts) < len(v_parts):
                r_parts.append(0)

            return v_parts >= r_parts
        except Exception:
            return True  # Default to allowing if parsing fails

    @classmethod
    def get_required_capabilities(cls, platform: PlatformType) -> Set[DeviceCapability]:
        """Get required capabilities for a platform."""
        # Base required capabilities
        required = {DeviceCapability.PUSH_NOTIFICATIONS}

        if platform == PlatformType.IOS:
            required.add(DeviceCapability.SECURE_ENCLAVE)
        elif platform == PlatformType.ANDROID:
            pass  # Android has different security model

        return required

    @classmethod
    def get_optional_capabilities(cls, platform: PlatformType) -> Set[DeviceCapability]:
        """Get optional capabilities that enhance functionality."""
        return {
            DeviceCapability.BIOMETRICS,
            DeviceCapability.FACE_ID,
            DeviceCapability.TOUCH_ID,
            DeviceCapability.FINGERPRINT,
            DeviceCapability.BACKGROUND_REFRESH,
            DeviceCapability.CAMERA,
            DeviceCapability.MICROPHONE,
            DeviceCapability.LOCATION,
        }


def get_platform_config(platform: PlatformType) -> PlatformConfig:
    """Get platform configuration (convenience function)."""
    return Platform.get_config(platform)


# Register default configurations
Platform.register_config(
    PlatformConfig(
        platform=PlatformType.IOS,
        min_os_version="14.0",
        feature_flags={
            "allow_jailbroken": False,
            "allow_emulator": True,
            "biometric_login": True,
            "offline_mode": True,
        },
    )
)

Platform.register_config(
    PlatformConfig(
        platform=PlatformType.ANDROID,
        min_os_version="8.0",
        feature_flags={
            "allow_jailbroken": False,
            "allow_emulator": True,
            "biometric_login": True,
            "offline_mode": True,
        },
    )
)
