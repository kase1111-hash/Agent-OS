"""Tests for the Mobile Applications Module."""

import asyncio
import os
import tempfile
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

from freezegun import freeze_time

import pytest
import pytest_asyncio

from src.mobile import (
    # Platform
    Platform,
    PlatformType,
    DeviceInfo,
    DeviceCapability,
    PlatformConfig,
    get_platform_config,
    # Client
    MobileClient,
    ClientConfig,
    ConnectionState,
    ApiError,
    NetworkError,
    # VPN
    VPNTunnel,
    VPNConfig,
    VPNState,
    TunnelProtocol,
    VPNError,
    # Auth
    MobileAuth,
    AuthConfig,
    AuthToken,
    DeviceToken,
    BiometricAuth,
    AuthState,
    # Notifications
    PushNotificationService,
    NotificationConfig,
    Notification,
    NotificationPayload,
    APNsProvider,
    FCMProvider,
    # Storage
    OfflineStorage,
    StorageConfig,
    SyncManager,
    SyncState,
    CacheEntry,
    # API
    MobileAPI,
    APIEndpoint,
    APIRequest,
    APIResponse,
)
from src.mobile.auth import BiometricType, SessionManager
from src.mobile.notifications import NotificationCategory, NotificationPriority, DeliveryStatus
from src.mobile.storage import ChangeType, ConflictResolution, ChangeRecord
from src.mobile.api import HttpMethod, APIVersion, RateLimiter
from src.mobile.vpn import TunnelStats, VPNManager


# ==================== Platform Tests ====================

class TestPlatformType:
    """Tests for PlatformType enum."""

    def test_platform_values(self):
        """Test platform enum values."""
        assert PlatformType.IOS.value == "ios"
        assert PlatformType.ANDROID.value == "android"
        assert PlatformType.WEB.value == "web"
        assert PlatformType.UNKNOWN.value == "unknown"


class TestDeviceCapability:
    """Tests for DeviceCapability enum."""

    def test_capability_values(self):
        """Test capability enum values."""
        assert DeviceCapability.BIOMETRICS.value == "biometrics"
        assert DeviceCapability.FACE_ID.value == "face_id"
        assert DeviceCapability.PUSH_NOTIFICATIONS.value == "push_notifications"


class TestDeviceInfo:
    """Tests for DeviceInfo dataclass."""

    def test_create_device_info(self):
        """Test creating device info."""
        device = DeviceInfo(
            device_id="test-device-123",
            platform=PlatformType.IOS,
            os_version="16.0",
            app_version="1.0.0",
            device_model="iPhone 14",
        )

        assert device.device_id == "test-device-123"
        assert device.platform == PlatformType.IOS
        assert device.os_version == "16.0"
        assert device.device_model == "iPhone 14"

    def test_auto_generate_device_id(self):
        """Test auto-generation of device ID."""
        device = DeviceInfo(
            device_id="",
            platform=PlatformType.ANDROID,
            os_version="13.0",
            app_version="1.0.0",
        )

        assert device.device_id  # Should be auto-generated

    def test_platform_from_string(self):
        """Test platform conversion from string."""
        device = DeviceInfo(
            device_id="test",
            platform="ios",  # type: ignore
            os_version="16.0",
            app_version="1.0.0",
        )

        assert device.platform == PlatformType.IOS

    def test_has_capability(self):
        """Test capability checking."""
        device = DeviceInfo(
            device_id="test",
            platform=PlatformType.IOS,
            os_version="16.0",
            app_version="1.0.0",
            capabilities={DeviceCapability.FACE_ID, DeviceCapability.PUSH_NOTIFICATIONS},
        )

        assert device.has_capability(DeviceCapability.FACE_ID)
        assert not device.has_capability(DeviceCapability.NFC)

    def test_supports_biometrics(self):
        """Test biometrics support checking."""
        device = DeviceInfo(
            device_id="test",
            platform=PlatformType.IOS,
            os_version="16.0",
            app_version="1.0.0",
            capabilities={DeviceCapability.FACE_ID},
        )

        assert device.supports_biometrics()

        device2 = DeviceInfo(
            device_id="test2",
            platform=PlatformType.ANDROID,
            os_version="13.0",
            app_version="1.0.0",
            capabilities=set(),
        )

        assert not device2.supports_biometrics()

    def test_to_dict(self):
        """Test conversion to dictionary."""
        device = DeviceInfo(
            device_id="test",
            platform=PlatformType.IOS,
            os_version="16.0",
            app_version="1.0.0",
        )

        data = device.to_dict()
        assert data["device_id"] == "test"
        assert data["platform"] == "ios"
        assert data["os_version"] == "16.0"

    def test_from_dict(self):
        """Test creation from dictionary."""
        data = {
            "device_id": "test",
            "platform": "android",
            "os_version": "13.0",
            "app_version": "1.0.0",
            "capabilities": ["push_notifications"],
            "registered_at": "2024-01-01T00:00:00",
            "last_seen": "2024-01-01T00:00:00",
            "metadata": {},
        }

        device = DeviceInfo.from_dict(data)
        assert device.device_id == "test"
        assert device.platform == PlatformType.ANDROID

    def test_from_user_agent_ios(self):
        """Test parsing iOS user agent."""
        ua = "Mozilla/5.0 (iPhone; CPU iPhone OS 16_0 like Mac OS X)"
        device = DeviceInfo.from_user_agent(ua)

        assert device.platform == PlatformType.IOS
        assert device.os_version == "16.0"
        assert device.device_model == "iPhone"

    def test_from_user_agent_android(self):
        """Test parsing Android user agent."""
        ua = "Mozilla/5.0 (Linux; Android 13; Pixel 7) Build/TQ2A"
        device = DeviceInfo.from_user_agent(ua)

        assert device.platform == PlatformType.ANDROID
        assert device.os_version == "13"


class TestPlatformConfig:
    """Tests for PlatformConfig."""

    def test_default_config(self):
        """Test default configuration."""
        config = PlatformConfig(platform=PlatformType.IOS)

        assert config.platform == PlatformType.IOS
        assert config.min_os_version == "14.0"
        assert config.certificate_pinning is True

    def test_android_defaults(self):
        """Test Android default settings."""
        config = PlatformConfig(platform=PlatformType.ANDROID)

        assert config.min_os_version == "8.0"

    def test_to_dict(self):
        """Test conversion to dictionary."""
        config = PlatformConfig(platform=PlatformType.IOS)
        data = config.to_dict()

        assert data["platform"] == "ios"
        assert "min_os_version" in data


class TestPlatform:
    """Tests for Platform class."""

    def test_register_config(self):
        """Test registering configuration."""
        config = PlatformConfig(
            platform=PlatformType.WEB,
            min_os_version="",
        )
        Platform.register_config(config)

        retrieved = Platform.get_config(PlatformType.WEB)
        assert retrieved.platform == PlatformType.WEB

    def test_is_supported(self):
        """Test device support checking."""
        device = DeviceInfo(
            device_id="test",
            platform=PlatformType.IOS,
            os_version="16.0",
            app_version="1.0.0",
        )

        assert Platform.is_supported(device)

    def test_unsupported_old_os(self):
        """Test rejection of old OS versions."""
        device = DeviceInfo(
            device_id="test",
            platform=PlatformType.IOS,
            os_version="12.0",
            app_version="1.0.0",
        )

        assert not Platform.is_supported(device)

    def test_jailbroken_rejected(self):
        """Test rejection of jailbroken devices."""
        device = DeviceInfo(
            device_id="test",
            platform=PlatformType.IOS,
            os_version="16.0",
            app_version="1.0.0",
            is_jailbroken=True,
        )

        assert not Platform.is_supported(device)


# ==================== Client Tests ====================

class TestClientConfig:
    """Tests for ClientConfig."""

    def test_default_config(self):
        """Test default configuration."""
        config = ClientConfig()

        assert config.base_url == "https://api.agentos.local"
        assert config.timeout == 30
        assert config.retry_count == 3

    def test_get_headers(self):
        """Test header generation."""
        config = ClientConfig(
            auth_token="test_token",
            device_id="device123",
        )

        headers = config.get_headers()

        assert headers["Authorization"] == "Bearer test_token"
        assert headers["X-Device-ID"] == "device123"
        assert headers["Content-Type"] == "application/json"


class TestMobileClient:
    """Tests for MobileClient."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        return MobileClient()

    def test_initial_state(self, client):
        """Test initial state is disconnected."""
        assert client.state == ConnectionState.DISCONNECTED
        assert not client.is_online

    @pytest.mark.asyncio
    async def test_connect(self, client):
        """Test connecting to server."""
        result = await client.connect()

        assert result is True
        assert client.state == ConnectionState.CONNECTED
        assert client.is_online

    @pytest.mark.asyncio
    async def test_disconnect(self, client):
        """Test disconnecting from server."""
        await client.connect()
        await client.disconnect()

        assert client.state == ConnectionState.DISCONNECTED
        assert not client.is_online

    @pytest.mark.asyncio
    async def test_get_request(self, client):
        """Test GET request."""
        await client.connect()
        response = await client.get("/health")

        assert response["status"] == "ok"

    @pytest.mark.asyncio
    async def test_post_request(self, client):
        """Test POST request."""
        await client.connect()
        response = await client.post("/auth/login", data={"user": "test"})

        assert "access_token" in response

    def test_set_auth_token(self, client):
        """Test setting auth token."""
        client.set_auth_token("new_token")
        assert client.config.auth_token == "new_token"

    def test_clear_auth_token(self, client):
        """Test clearing auth token."""
        client.set_auth_token("token")
        client.clear_auth_token()
        assert client.config.auth_token is None

    def test_state_change_callback(self, client):
        """Test state change callbacks."""
        states = []
        client.on_state_change(lambda s: states.append(s))

        client._set_state(ConnectionState.CONNECTING)
        client._set_state(ConnectionState.CONNECTED)

        assert states == [ConnectionState.CONNECTING, ConnectionState.CONNECTED]

    def test_get_stats(self, client):
        """Test getting client stats."""
        stats = client.get_stats()

        assert "state" in stats
        assert "is_online" in stats
        assert "request_count" in stats

    def test_request_queue(self, client):
        """Test request queuing."""
        request_id = client._queue_request("POST", "/test", {"data": 1}, {})

        assert request_id is not None
        assert client.get_queue_size() == 1

    def test_clear_queue(self, client):
        """Test clearing request queue."""
        client._queue_request("POST", "/test", {"data": 1}, {})
        client._queue_request("POST", "/test2", {"data": 2}, {})

        count = client.clear_queue()

        assert count == 2
        assert client.get_queue_size() == 0


class TestApiError:
    """Tests for ApiError."""

    def test_create_error(self):
        """Test creating API error."""
        error = ApiError(
            "Not found",
            status_code=404,
            error_code="NOT_FOUND",
        )

        assert str(error) == "Not found"
        assert error.status_code == 404
        assert error.error_code == "NOT_FOUND"

    def test_to_dict(self):
        """Test error to dictionary."""
        error = ApiError("Error", status_code=500)
        data = error.to_dict()

        assert data["message"] == "Error"
        assert data["status_code"] == 500


# ==================== VPN Tests ====================

class TestVPNConfig:
    """Tests for VPNConfig."""

    def test_default_config(self):
        """Test default VPN configuration."""
        config = VPNConfig(server_address="vpn.example.com")

        assert config.server_port == 51820
        assert config.protocol == TunnelProtocol.WIREGUARD
        assert config.private_key  # Should be auto-generated
        assert config.public_key  # Should be derived

    def test_key_generation(self):
        """Test key pair generation."""
        config = VPNConfig(server_address="vpn.example.com")

        assert len(config.private_key) > 0
        assert len(config.public_key) > 0

    def test_wireguard_config_generation(self):
        """Test WireGuard config file generation."""
        config = VPNConfig(
            server_address="vpn.example.com",
            server_public_key="server_pub_key",
        )

        wg_config = config.to_wireguard_config()

        assert "[Interface]" in wg_config
        assert "[Peer]" in wg_config
        assert "vpn.example.com:51820" in wg_config


class TestVPNTunnel:
    """Tests for VPNTunnel."""

    @pytest.fixture
    def tunnel(self):
        """Create test tunnel."""
        config = VPNConfig(
            server_address="vpn.example.com",
            server_public_key="test_server_key",
        )
        return VPNTunnel(config)

    def test_initial_state(self, tunnel):
        """Test initial tunnel state."""
        assert tunnel.state == VPNState.DISCONNECTED
        assert not tunnel.is_connected

    @pytest.mark.asyncio
    async def test_connect(self, tunnel):
        """Test VPN connection."""
        result = await tunnel.connect()

        assert result is True
        assert tunnel.state == VPNState.CONNECTED
        assert tunnel.is_connected

    @pytest.mark.asyncio
    async def test_disconnect(self, tunnel):
        """Test VPN disconnection."""
        await tunnel.connect()
        await tunnel.disconnect()

        assert tunnel.state == VPNState.DISCONNECTED
        assert not tunnel.is_connected

    @pytest.mark.asyncio
    async def test_reconnect(self, tunnel):
        """Test VPN reconnection."""
        await tunnel.connect()
        result = await tunnel.reconnect()

        assert result is True
        assert tunnel.is_connected

    def test_state_callback(self, tunnel):
        """Test state change callback."""
        states = []
        tunnel.on_state_change(lambda s: states.append(s))

        tunnel._set_state(VPNState.CONNECTING)
        tunnel._set_state(VPNState.CONNECTED)

        assert VPNState.CONNECTING in states
        assert VPNState.CONNECTED in states

    def test_split_tunnel_config(self, tunnel):
        """Test split tunneling configuration."""
        tunnel.set_split_tunnel(
            enabled=True,
            excluded_apps=["com.example.app"],
            excluded_routes=["192.168.1.0/24"],
        )

        assert tunnel.config.split_tunnel is True
        assert "com.example.app" in tunnel.config.excluded_apps

    def test_get_diagnostics(self, tunnel):
        """Test getting diagnostics."""
        diag = tunnel.get_diagnostics()

        assert "state" in diag
        assert "protocol" in diag
        assert "stats" in diag

    @pytest.mark.asyncio
    async def test_key_rotation(self, tunnel):
        """Test key rotation."""
        old_public = tunnel.config.public_key
        new_private, new_public = await tunnel.rotate_keys()

        assert new_public != old_public
        assert tunnel.config.public_key == new_public


class TestVPNManager:
    """Tests for VPNManager."""

    def test_add_config(self):
        """Test adding VPN configuration."""
        manager = VPNManager()
        config = VPNConfig(server_address="vpn1.example.com", server_public_key="key1")

        manager.add_config("vpn1", config)

        assert "vpn1" in manager.list_configs()

    def test_remove_config(self):
        """Test removing VPN configuration."""
        manager = VPNManager()
        config = VPNConfig(server_address="vpn1.example.com", server_public_key="key1")

        manager.add_config("vpn1", config)
        result = manager.remove_config("vpn1")

        assert result is True
        assert "vpn1" not in manager.list_configs()

    @pytest.mark.asyncio
    async def test_connect_to_named_config(self):
        """Test connecting to named configuration."""
        manager = VPNManager()
        config = VPNConfig(server_address="vpn1.example.com", server_public_key="key1")

        manager.add_config("vpn1", config)
        result = await manager.connect("vpn1")

        assert result is True
        assert manager.get_active_tunnel() is not None


# ==================== Auth Tests ====================

class TestAuthToken:
    """Tests for AuthToken."""

    def test_create_token(self):
        """Test creating auth token."""
        token = AuthToken(
            access_token="test_access_token",
            expires_at=datetime.now() + timedelta(hours=1),
        )

        assert token.access_token == "test_access_token"
        assert not token.is_expired

    def test_expired_token(self):
        """Test expired token detection."""
        token = AuthToken(
            access_token="test",
            expires_at=datetime.now() - timedelta(hours=1),
        )

        assert token.is_expired

    def test_expires_in(self):
        """Test expires_in calculation."""
        token = AuthToken(
            access_token="test",
            expires_at=datetime.now() + timedelta(minutes=5),
        )

        assert 280 < token.expires_in < 310  # Approximately 5 minutes

    def test_from_response(self):
        """Test creating token from OAuth response."""
        response = {
            "access_token": "access",
            "refresh_token": "refresh",
            "expires_in": 3600,
        }

        token = AuthToken.from_response(response)

        assert token.access_token == "access"
        assert token.refresh_token == "refresh"


class TestBiometricAuth:
    """Tests for BiometricAuth."""

    def test_create_biometric(self):
        """Test creating biometric auth."""
        bio = BiometricAuth(BiometricType.FACE_ID)

        assert bio.biometric_type == BiometricType.FACE_ID
        assert bio.is_available

    def test_not_available(self):
        """Test biometric not available."""
        bio = BiometricAuth(BiometricType.NONE)

        assert not bio.is_available

    def test_enroll(self):
        """Test biometric enrollment."""
        bio = BiometricAuth(BiometricType.TOUCH_ID)
        result = bio.enroll()

        assert result is True
        assert bio.is_enrolled

    @pytest.mark.asyncio
    async def test_authenticate(self):
        """Test biometric authentication."""
        bio = BiometricAuth(BiometricType.FINGERPRINT)
        bio.enroll()

        result = await bio.authenticate("Test auth")

        assert result is True


class TestMobileAuth:
    """Tests for MobileAuth."""

    @pytest.fixture
    def auth(self):
        """Create test auth instance."""
        return MobileAuth()

    def test_initial_state(self, auth):
        """Test initial auth state."""
        assert auth.state == AuthState.UNAUTHENTICATED
        assert not auth.is_authenticated

    @pytest.mark.asyncio
    async def test_register_device(self, auth):
        """Test device registration."""
        device_token = await auth.register_device(
            device_id="device123",
            platform="ios",
            push_token="push_token_abc",
        )

        assert device_token.device_id == "device123"
        assert device_token.platform == "ios"

    @pytest.mark.asyncio
    async def test_authenticate_with_credentials(self, auth):
        """Test authentication with username/password."""
        token = await auth.authenticate(
            username="testuser",
            password="testpass",
        )

        assert token is not None
        assert auth.is_authenticated
        assert auth.state == AuthState.AUTHENTICATED

    @pytest.mark.asyncio
    async def test_logout(self, auth):
        """Test logout."""
        await auth.authenticate(username="test", password="pass")
        await auth.logout()

        assert not auth.is_authenticated
        assert auth.state == AuthState.UNAUTHENTICATED

    @pytest.mark.asyncio
    async def test_refresh_token(self, auth):
        """Test token refresh."""
        await auth.authenticate(username="test", password="pass")
        new_token = await auth.refresh_token()

        assert new_token is not None
        assert auth.is_authenticated

    def test_get_authorization_header(self, auth):
        """Test getting authorization header."""
        # Not authenticated
        assert auth.get_authorization_header() is None

    @pytest.mark.asyncio
    async def test_failed_attempts_lockout(self, auth):
        """Test account lockout after failed attempts."""
        auth.config.max_failed_attempts = 2
        auth.config.lockout_duration = 60

        # Simulate failed attempts
        auth._record_failed_attempt()
        auth._record_failed_attempt()

        assert auth.is_locked
        assert auth.state == AuthState.LOCKED


class TestSessionManager:
    """Tests for SessionManager."""

    def test_record_activity(self):
        """Test recording activity."""
        manager = SessionManager(timeout_seconds=3600)
        manager.record_activity()

        assert manager.is_session_valid()

    def test_session_timeout(self):
        """Test session timeout."""
        manager = SessionManager(timeout_seconds=0)
        manager.record_activity()

        assert not manager.is_session_valid()

    def test_remaining_time(self):
        """Test remaining session time."""
        manager = SessionManager(timeout_seconds=3600)
        manager.record_activity()

        remaining = manager.get_remaining_time()
        assert 3590 < remaining <= 3600


# ==================== Notification Tests ====================

class TestNotificationPayload:
    """Tests for NotificationPayload."""

    def test_create_payload(self):
        """Test creating notification payload."""
        payload = NotificationPayload(
            title="Test Title",
            body="Test body message",
        )

        assert payload.title == "Test Title"
        assert payload.body == "Test body message"
        assert payload.priority == NotificationPriority.NORMAL

    def test_to_apns_payload(self):
        """Test APNs payload conversion."""
        payload = NotificationPayload(
            title="Test",
            body="Body",
            badge=5,
        )

        apns = payload.to_apns_payload()

        assert "aps" in apns
        assert apns["aps"]["alert"]["title"] == "Test"
        assert apns["aps"]["badge"] == 5

    def test_to_fcm_payload(self):
        """Test FCM payload conversion."""
        payload = NotificationPayload(
            title="Test",
            body="Body",
            priority=NotificationPriority.HIGH,
        )

        fcm = payload.to_fcm_payload()

        assert "notification" in fcm
        assert fcm["android"]["priority"] == "high"


class TestPushNotificationService:
    """Tests for PushNotificationService."""

    @pytest.fixture
    def service(self):
        """Create test notification service."""
        return PushNotificationService()

    def test_register_device(self, service):
        """Test device registration."""
        service.register_device("device1", "ios", "push_token_1")

        assert service._device_tokens["device1"]["platform"] == "ios"

    def test_unregister_device(self, service):
        """Test device unregistration."""
        service.register_device("device1", "ios", "push_token_1")
        service.unregister_device("device1")

        assert "device1" not in service._device_tokens

    @pytest.mark.asyncio
    async def test_send_notification(self, service):
        """Test sending notification."""
        service.register_device("device1", "ios", "token1")
        await service.start()

        payload = NotificationPayload(title="Test", body="Test message")
        notification = await service.send("device1", payload)

        assert notification.notification_id is not None
        assert notification.device_id == "device1"

        await service.stop()

    @pytest.mark.asyncio
    async def test_send_to_many(self, service):
        """Test sending to multiple devices."""
        service.register_device("device1", "ios", "token1")
        service.register_device("device2", "android", "token2")

        payload = NotificationPayload(title="Test", body="Broadcast")
        notifications = await service.send_to_many(["device1", "device2"], payload)

        assert len(notifications) == 2

    def test_get_stats(self, service):
        """Test getting notification stats."""
        stats = service.get_stats()

        assert "total_notifications" in stats
        assert "registered_devices" in stats


class TestAPNsProvider:
    """Tests for APNsProvider."""

    def test_not_configured(self):
        """Test provider not configured."""
        config = NotificationConfig()
        provider = APNsProvider(config)

        assert not provider.is_configured

    @pytest.mark.asyncio
    async def test_connect_when_not_configured(self):
        """Test connection when not configured."""
        config = NotificationConfig()
        provider = APNsProvider(config)

        result = await provider.connect()

        assert result is False

    @pytest.mark.asyncio
    async def test_connect_when_configured(self):
        """Test connection when configured."""
        config = NotificationConfig(
            apns_key_id="key123",
            apns_team_id="team456",
            apns_bundle_id="com.example.app",
        )
        provider = APNsProvider(config)

        result = await provider.connect()

        assert result is True


class TestFCMProvider:
    """Tests for FCMProvider."""

    def test_not_configured(self):
        """Test provider not configured."""
        config = NotificationConfig()
        provider = FCMProvider(config)

        assert not provider.is_configured

    @pytest.mark.asyncio
    async def test_connect_when_configured(self):
        """Test connection when configured."""
        config = NotificationConfig(fcm_project_id="project123")
        provider = FCMProvider(config)

        result = await provider.connect()

        assert result is True


# ==================== Storage Tests ====================

class TestCacheEntry:
    """Tests for CacheEntry."""

    def test_create_entry(self):
        """Test creating cache entry."""
        entry = CacheEntry(key="test_key", value={"data": 123})

        assert entry.key == "test_key"
        assert entry.value == {"data": 123}
        assert not entry.is_expired

    def test_expired_entry(self):
        """Test expired entry detection."""
        entry = CacheEntry(
            key="test",
            value="data",
            expires_at=datetime.now() - timedelta(hours=1),
        )

        assert entry.is_expired

    def test_touch(self):
        """Test updating access time."""
        with freeze_time("2024-01-01 12:00:00") as frozen_time:
            entry = CacheEntry(key="test", value="data")
            old_accessed = entry.accessed_at

            # Advance time by 1 second
            frozen_time.tick(timedelta(seconds=1))
            entry.touch()

            assert entry.accessed_at > old_accessed


class TestOfflineStorage:
    """Tests for OfflineStorage."""

    @pytest_asyncio.fixture
    async def storage(self, tmp_path):
        """Create test storage."""
        config = StorageConfig(
            database_path=str(tmp_path / "test.db"),
            cache_size_mb=10,
        )
        storage = OfflineStorage(config)
        await storage.initialize()
        yield storage
        await storage.close()

    @pytest.mark.asyncio
    async def test_set_and_get(self, storage):
        """Test setting and getting values."""
        await storage.set("key1", {"test": "value"})
        result = await storage.get("key1")

        assert result == {"test": "value"}

    @pytest.mark.asyncio
    async def test_get_nonexistent(self, storage):
        """Test getting nonexistent key."""
        result = await storage.get("nonexistent")

        assert result is None

    @pytest.mark.asyncio
    async def test_delete(self, storage):
        """Test deleting key."""
        await storage.set("key1", "value")
        result = await storage.delete("key1")

        assert result is True
        assert await storage.get("key1") is None

    @pytest.mark.asyncio
    async def test_exists(self, storage):
        """Test existence check."""
        await storage.set("key1", "value")

        assert await storage.exists("key1")
        assert not await storage.exists("nonexistent")

    @pytest.mark.asyncio
    async def test_save_entity(self, storage):
        """Test saving entity."""
        version = await storage.save_entity(
            "user", "user1", {"name": "Test User"}
        )

        assert version == 1

        entity = await storage.get_entity("user", "user1")
        assert entity["name"] == "Test User"

    @pytest.mark.asyncio
    async def test_update_entity_version(self, storage):
        """Test entity version increments."""
        await storage.save_entity("user", "user1", {"name": "V1"})
        version = await storage.save_entity("user", "user1", {"name": "V2"})

        assert version == 2

    @pytest.mark.asyncio
    async def test_delete_entity(self, storage):
        """Test deleting entity."""
        await storage.save_entity("user", "user1", {"name": "Test"})
        result = await storage.delete_entity("user", "user1")

        assert result is True
        assert await storage.get_entity("user", "user1") is None

    @pytest.mark.asyncio
    async def test_query_entities(self, storage):
        """Test querying entities."""
        await storage.save_entity("user", "user1", {"name": "Alice", "age": 25})
        await storage.save_entity("user", "user2", {"name": "Bob", "age": 30})

        users = await storage.query_entities("user")

        assert len(users) == 2

    @pytest.mark.asyncio
    async def test_pending_changes(self, storage):
        """Test getting pending changes."""
        await storage.save_entity("user", "user1", {"name": "Test"})
        changes = await storage.get_pending_changes()

        assert len(changes) == 1
        assert changes[0].entity_type == "user"

    @pytest.mark.asyncio
    async def test_cache_stats(self, storage):
        """Test cache statistics."""
        stats = storage.get_cache_stats()

        assert "entries" in stats
        assert "size_bytes" in stats

    @pytest.mark.asyncio
    async def test_clear_cache(self, storage):
        """Test clearing cache."""
        storage._cache["test"] = CacheEntry(key="test", value="data")
        count = storage.clear_cache()

        assert count == 1
        assert len(storage._cache) == 0


class TestSyncManager:
    """Tests for SyncManager."""

    @pytest_asyncio.fixture
    async def sync_manager(self, tmp_path):
        """Create test sync manager."""
        config = StorageConfig(database_path=str(tmp_path / "sync_test.db"))
        storage = OfflineStorage(config)
        await storage.initialize()
        manager = SyncManager(storage, config)
        yield manager
        await storage.close()

    @pytest.mark.asyncio
    async def test_initial_state(self, sync_manager):
        """Test initial sync state."""
        assert sync_manager.state == SyncState.IDLE
        assert not sync_manager.is_syncing

    @pytest.mark.asyncio
    async def test_set_online(self, sync_manager):
        """Test setting online status."""
        sync_manager.set_online(False)
        assert sync_manager.state == SyncState.OFFLINE

        sync_manager.set_online(True)
        assert sync_manager.state == SyncState.IDLE

    @pytest.mark.asyncio
    async def test_sync(self, sync_manager):
        """Test synchronization."""
        result = await sync_manager.sync()

        assert result["status"] == "success"
        assert "changes_pushed" in result

    @pytest.mark.asyncio
    async def test_sync_offline(self, sync_manager):
        """Test sync when offline."""
        sync_manager.set_online(False)
        result = await sync_manager.sync()

        assert result["status"] == "offline"

    @pytest.mark.asyncio
    async def test_get_sync_status(self, sync_manager):
        """Test getting sync status."""
        status = sync_manager.get_sync_status()

        assert "state" in status
        assert "is_online" in status


# ==================== API Tests ====================

class TestAPIEndpoint:
    """Tests for APIEndpoint."""

    def test_create_endpoint(self):
        """Test creating endpoint."""
        endpoint = APIEndpoint(
            path="/users",
            method=HttpMethod.GET,
        )

        assert endpoint.path == "/users"
        assert endpoint.method == HttpMethod.GET

    def test_full_path(self):
        """Test full path generation."""
        endpoint = APIEndpoint(path="/users", version=APIVersion.V2)

        assert endpoint.full_path == "/api/v2/users"


class TestRateLimiter:
    """Tests for RateLimiter."""

    def test_can_proceed(self):
        """Test rate limiting."""
        limiter = RateLimiter(requests_per_minute=60)

        assert limiter.can_proceed()
        limiter.record_request()
        assert limiter.can_proceed()

    def test_rate_limit_exceeded(self):
        """Test rate limit exceeded."""
        limiter = RateLimiter(requests_per_minute=2)

        limiter.record_request()
        limiter.record_request()
        assert not limiter.can_proceed()


class TestMobileAPI:
    """Tests for MobileAPI."""

    @pytest.fixture
    def api(self):
        """Create test API instance."""
        return MobileAPI()

    @pytest.mark.asyncio
    async def test_health_check(self, api):
        """Test health check endpoint."""
        await api.client.connect()
        response = await api.health_check()

        assert response.success is True

    @pytest.mark.asyncio
    async def test_login(self, api):
        """Test login endpoint."""
        await api.client.connect()
        response = await api.login("testuser", "testpass")

        assert response.success is True

    @pytest.mark.asyncio
    async def test_get_profile(self, api):
        """Test get profile endpoint."""
        await api.client.connect()
        response = await api.get_profile()

        assert response.success is True

    @pytest.mark.asyncio
    async def test_list_agents(self, api):
        """Test list agents endpoint."""
        await api.client.connect()
        response = await api.list_agents()

        assert response.success is True

    def test_request_callback(self, api):
        """Test request callback."""
        requests = []
        api.on_request(lambda r: requests.append(r))

        # Callback will be triggered on next request

    def test_response_callback(self, api):
        """Test response callback."""
        responses = []
        api.on_response(lambda r: responses.append(r))

        # Callback will be triggered on next response


class TestAPIRequest:
    """Tests for APIRequest."""

    def test_create_request(self):
        """Test creating API request."""
        endpoint = APIEndpoint(path="/test", method=HttpMethod.POST)
        request = APIRequest(
            endpoint=endpoint,
            data={"key": "value"},
        )

        assert request.endpoint == endpoint
        assert request.data == {"key": "value"}
        assert request.request_id is not None

    def test_to_dict(self):
        """Test request to dictionary."""
        endpoint = APIEndpoint(path="/test")
        request = APIRequest(endpoint=endpoint)

        data = request.to_dict()

        assert "request_id" in data
        assert "endpoint" in data
        assert "timestamp" in data


class TestAPIResponse:
    """Tests for APIResponse."""

    def test_success_response(self):
        """Test successful response."""
        response = APIResponse(success=True, data={"result": "ok"})

        assert response.success is True
        assert response.data == {"result": "ok"}

    def test_error_response(self):
        """Test error response."""
        response = APIResponse(
            success=False,
            error="Not found",
            error_code="NOT_FOUND",
        )

        assert response.success is False
        assert response.error == "Not found"

    def test_to_dict(self):
        """Test response to dictionary."""
        response = APIResponse(success=True, data={"key": "value"})
        data = response.to_dict()

        assert data["success"] is True
        assert data["data"] == {"key": "value"}


# ==================== Integration Tests ====================

class TestMobileIntegration:
    """Integration tests for mobile module."""

    @pytest.mark.asyncio
    async def test_full_auth_flow(self):
        """Test complete authentication flow."""
        # Create auth instance
        auth = MobileAuth()

        # Register device
        device_token = await auth.register_device(
            device_id="test-device",
            platform="ios",
        )
        assert device_token is not None

        # Authenticate
        token = await auth.authenticate(
            username="testuser",
            password="testpass",
        )
        assert token is not None
        assert auth.is_authenticated

        # Logout
        await auth.logout()
        assert not auth.is_authenticated

    @pytest.mark.asyncio
    async def test_vpn_with_auth(self):
        """Test VPN with authentication."""
        # Create VPN tunnel
        config = VPNConfig(
            server_address="vpn.example.com",
            server_public_key="server_key",
        )
        tunnel = VPNTunnel(config)

        # Connect
        await tunnel.connect()
        assert tunnel.is_connected

        # Verify diagnostics
        diag = tunnel.get_diagnostics()
        assert diag["state"] == "connected"

        # Disconnect
        await tunnel.disconnect()
        assert not tunnel.is_connected

    @pytest.mark.asyncio
    async def test_offline_storage_with_sync(self, tmp_path):
        """Test offline storage with sync manager."""
        config = StorageConfig(database_path=str(tmp_path / "integration.db"))
        storage = OfflineStorage(config)
        await storage.initialize()

        sync = SyncManager(storage)

        # Store some data
        await storage.save_entity("task", "task1", {"title": "Test Task"})
        await storage.save_entity("task", "task2", {"title": "Another Task"})

        # Check pending changes
        changes = await storage.get_pending_changes()
        assert len(changes) == 2

        # Sync
        result = await sync.sync()
        assert result["status"] == "success"

        await storage.close()
