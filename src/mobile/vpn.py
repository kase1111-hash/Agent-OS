"""VPN Tunnel Integration for Mobile Applications.

Provides secure VPN connectivity for mobile clients:
- WireGuard and IPSec protocol support
- Automatic connection management
- Split tunneling configuration
- Key rotation and management
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
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


class TunnelProtocol(str, Enum):
    """Supported VPN protocols."""

    WIREGUARD = "wireguard"
    IPSEC = "ipsec"
    OPENVPN = "openvpn"


class VPNState(str, Enum):
    """VPN connection states."""

    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    DISCONNECTING = "disconnecting"
    ERROR = "error"


class VPNError(Exception):
    """VPN-related error."""

    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        is_recoverable: bool = True,
    ):
        super().__init__(message)
        self.error_code = error_code
        self.is_recoverable = is_recoverable


@dataclass
class VPNConfig:
    """VPN tunnel configuration."""

    server_address: str
    server_port: int = 51820
    protocol: TunnelProtocol = TunnelProtocol.WIREGUARD
    private_key: str = ""
    public_key: str = ""
    server_public_key: str = ""
    preshared_key: Optional[str] = None
    allowed_ips: List[str] = field(default_factory=lambda: ["0.0.0.0/0", "::/0"])
    dns_servers: List[str] = field(default_factory=lambda: ["1.1.1.1", "8.8.8.8"])
    mtu: int = 1420
    persistent_keepalive: int = 25
    split_tunnel: bool = False
    excluded_apps: List[str] = field(default_factory=list)
    excluded_routes: List[str] = field(default_factory=list)
    auto_reconnect: bool = True
    reconnect_delay: float = 5.0
    max_reconnect_attempts: int = 5
    handshake_timeout: int = 30
    connection_timeout: int = 60

    def __post_init__(self):
        """Generate keys if not provided."""
        if isinstance(self.protocol, str):
            self.protocol = TunnelProtocol(self.protocol)
        if not self.private_key:
            self.private_key = self._generate_key()
        if not self.public_key:
            self.public_key = self._derive_public_key(self.private_key)

    @staticmethod
    def _generate_key() -> str:
        """Generate a WireGuard private key."""
        # Generate 32 random bytes and base64 encode
        key_bytes = secrets.token_bytes(32)
        # Clamp for Curve25519 (WireGuard uses this)
        key_list = list(key_bytes)
        key_list[0] &= 248
        key_list[31] &= 127
        key_list[31] |= 64
        import base64

        return base64.b64encode(bytes(key_list)).decode()

    @staticmethod
    def _derive_public_key(private_key: str) -> str:
        """Derive public key from private key (mock implementation)."""
        # In production, use actual Curve25519
        import base64

        hash_bytes = hashlib.sha256(private_key.encode()).digest()
        return base64.b64encode(hash_bytes).decode()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "server_address": self.server_address,
            "server_port": self.server_port,
            "protocol": self.protocol.value,
            "public_key": self.public_key,
            "server_public_key": self.server_public_key,
            "allowed_ips": self.allowed_ips,
            "dns_servers": self.dns_servers,
            "mtu": self.mtu,
            "persistent_keepalive": self.persistent_keepalive,
            "split_tunnel": self.split_tunnel,
            "excluded_apps": self.excluded_apps,
            "excluded_routes": self.excluded_routes,
            "auto_reconnect": self.auto_reconnect,
        }

    def to_wireguard_config(self) -> str:
        """Generate WireGuard configuration file content."""
        config = f"""[Interface]
PrivateKey = {self.private_key}
Address = 10.0.0.2/32
DNS = {', '.join(self.dns_servers)}
MTU = {self.mtu}

[Peer]
PublicKey = {self.server_public_key}
"""
        if self.preshared_key:
            config += f"PresharedKey = {self.preshared_key}\n"

        config += f"""AllowedIPs = {', '.join(self.allowed_ips)}
Endpoint = {self.server_address}:{self.server_port}
PersistentKeepalive = {self.persistent_keepalive}
"""
        return config


@dataclass
class TunnelStats:
    """VPN tunnel statistics."""

    bytes_sent: int = 0
    bytes_received: int = 0
    packets_sent: int = 0
    packets_received: int = 0
    last_handshake: Optional[datetime] = None
    connected_since: Optional[datetime] = None
    reconnect_count: int = 0
    current_endpoint: str = ""
    latency_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "bytes_sent": self.bytes_sent,
            "bytes_received": self.bytes_received,
            "packets_sent": self.packets_sent,
            "packets_received": self.packets_received,
            "last_handshake": self.last_handshake.isoformat() if self.last_handshake else None,
            "connected_since": self.connected_since.isoformat() if self.connected_since else None,
            "reconnect_count": self.reconnect_count,
            "current_endpoint": self.current_endpoint,
            "latency_ms": self.latency_ms,
        }


class VPNTunnel:
    """VPN tunnel manager for mobile applications.

    Features:
    - WireGuard protocol support
    - Automatic reconnection
    - Split tunneling
    - Connection statistics
    """

    def __init__(self, config: VPNConfig):
        """Initialize VPN tunnel.

        Args:
            config: VPN configuration
        """
        self.config = config
        self._state = VPNState.DISCONNECTED
        self._state_callbacks: List[Callable[[VPNState], None]] = []
        self._stats = TunnelStats()
        self._reconnect_task: Optional[asyncio.Task] = None
        self._keepalive_task: Optional[asyncio.Task] = None
        self._reconnect_attempts = 0
        self._interface_name = "wg0"

    @property
    def state(self) -> VPNState:
        """Get current VPN state."""
        return self._state

    @property
    def is_connected(self) -> bool:
        """Check if VPN is connected."""
        return self._state == VPNState.CONNECTED

    @property
    def stats(self) -> TunnelStats:
        """Get tunnel statistics."""
        return self._stats

    def on_state_change(self, callback: Callable[[VPNState], None]) -> None:
        """Register a state change callback."""
        self._state_callbacks.append(callback)

    def _set_state(self, state: VPNState) -> None:
        """Set VPN state and notify callbacks."""
        if self._state != state:
            old_state = self._state
            self._state = state
            logger.info(f"VPN state changed: {old_state.value} -> {state.value}")
            for callback in self._state_callbacks:
                try:
                    callback(state)
                except Exception as e:
                    logger.warning(f"State callback error: {e}")

    async def connect(self) -> bool:
        """Establish VPN connection.

        Returns:
            True if connection successful
        """
        if self._state == VPNState.CONNECTED:
            return True

        if self._state == VPNState.CONNECTING:
            logger.warning("Connection already in progress")
            return False

        self._set_state(VPNState.CONNECTING)
        self._reconnect_attempts = 0

        try:
            # Validate configuration
            if not self.config.server_public_key:
                raise VPNError("Server public key not configured")

            # Perform handshake
            success = await self._perform_handshake()

            if success:
                self._set_state(VPNState.CONNECTED)
                self._stats.connected_since = datetime.now()
                self._stats.last_handshake = datetime.now()
                self._stats.current_endpoint = (
                    f"{self.config.server_address}:{self.config.server_port}"
                )

                # Start keepalive
                self._keepalive_task = asyncio.create_task(self._keepalive_loop())

                logger.info("VPN connected successfully")
                return True
            else:
                self._set_state(VPNState.ERROR)
                return False

        except VPNError as e:
            logger.error(f"VPN connection failed: {e}")
            self._set_state(VPNState.ERROR)

            if e.is_recoverable and self.config.auto_reconnect:
                self._schedule_reconnect()

            raise

        except Exception as e:
            logger.error(f"Unexpected VPN error: {e}")
            self._set_state(VPNState.ERROR)
            raise VPNError(f"Connection failed: {e}")

    async def disconnect(self) -> None:
        """Disconnect VPN tunnel."""
        if self._state == VPNState.DISCONNECTED:
            return

        self._set_state(VPNState.DISCONNECTING)

        # Cancel background tasks
        if self._keepalive_task:
            self._keepalive_task.cancel()
            self._keepalive_task = None

        if self._reconnect_task:
            self._reconnect_task.cancel()
            self._reconnect_task = None

        try:
            await self._teardown_tunnel()
        except Exception as e:
            logger.warning(f"Tunnel teardown error: {e}")

        self._set_state(VPNState.DISCONNECTED)
        self._stats.connected_since = None
        logger.info("VPN disconnected")

    async def reconnect(self) -> bool:
        """Force reconnection.

        Returns:
            True if reconnection successful
        """
        await self.disconnect()
        await asyncio.sleep(1)
        return await self.connect()

    async def _perform_handshake(self) -> bool:
        """Perform VPN handshake (mock implementation).

        Returns:
            True if handshake successful
        """
        # Simulate handshake delay
        await asyncio.sleep(0.1)

        # In production, this would:
        # 1. Generate ephemeral key pair
        # 2. Send handshake initiation
        # 3. Receive and verify response
        # 4. Complete key exchange

        # Mock successful handshake
        logger.debug("VPN handshake completed")
        return True

    async def _teardown_tunnel(self) -> None:
        """Tear down VPN tunnel (mock implementation)."""
        # In production, this would:
        # 1. Send close notification
        # 2. Remove routing rules
        # 3. Delete interface
        await asyncio.sleep(0.05)
        logger.debug("VPN tunnel torn down")

    async def _keepalive_loop(self) -> None:
        """Send periodic keepalive packets."""
        while self._state == VPNState.CONNECTED:
            try:
                await asyncio.sleep(self.config.persistent_keepalive)

                if self._state != VPNState.CONNECTED:
                    break

                # Send keepalive
                await self._send_keepalive()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning(f"Keepalive error: {e}")
                if self.config.auto_reconnect:
                    self._schedule_reconnect()
                break

    async def _send_keepalive(self) -> None:
        """Send keepalive packet (mock implementation)."""
        # Update stats
        self._stats.packets_sent += 1
        self._stats.bytes_sent += 32  # Keepalive packet size

        # Simulate latency measurement
        start = time.time()
        await asyncio.sleep(0.01)  # Simulated round trip
        self._stats.latency_ms = (time.time() - start) * 1000

    def _schedule_reconnect(self) -> None:
        """Schedule reconnection attempt."""
        if self._reconnect_task and not self._reconnect_task.done():
            return

        if self._reconnect_attempts >= self.config.max_reconnect_attempts:
            logger.error("Max reconnection attempts reached")
            self._set_state(VPNState.ERROR)
            return

        self._set_state(VPNState.RECONNECTING)
        self._reconnect_task = asyncio.create_task(self._reconnect_loop())

    async def _reconnect_loop(self) -> None:
        """Handle reconnection with exponential backoff."""
        while self._reconnect_attempts < self.config.max_reconnect_attempts:
            self._reconnect_attempts += 1
            delay = self.config.reconnect_delay * (2 ** (self._reconnect_attempts - 1))

            logger.info(f"Reconnection attempt {self._reconnect_attempts} in {delay}s")
            await asyncio.sleep(delay)

            try:
                if await self._perform_handshake():
                    self._set_state(VPNState.CONNECTED)
                    self._stats.reconnect_count += 1
                    self._stats.last_handshake = datetime.now()
                    self._keepalive_task = asyncio.create_task(self._keepalive_loop())
                    logger.info("Reconnection successful")
                    return
            except Exception as e:
                logger.warning(f"Reconnection attempt failed: {e}")

        logger.error("All reconnection attempts failed")
        self._set_state(VPNState.ERROR)

    def get_interface_config(self) -> Dict[str, Any]:
        """Get network interface configuration."""
        return {
            "name": self._interface_name,
            "address": "10.0.0.2/32",
            "mtu": self.config.mtu,
            "dns": self.config.dns_servers,
            "routes": self.config.allowed_ips,
            "excluded_routes": self.config.excluded_routes,
        }

    def update_allowed_ips(self, allowed_ips: List[str]) -> None:
        """Update allowed IP ranges.

        Args:
            allowed_ips: List of CIDR ranges
        """
        self.config.allowed_ips = allowed_ips
        logger.info(f"Updated allowed IPs: {allowed_ips}")

    def set_split_tunnel(
        self,
        enabled: bool,
        excluded_apps: Optional[List[str]] = None,
        excluded_routes: Optional[List[str]] = None,
    ) -> None:
        """Configure split tunneling.

        Args:
            enabled: Enable split tunneling
            excluded_apps: Apps to exclude from VPN
            excluded_routes: Routes to exclude from VPN
        """
        self.config.split_tunnel = enabled
        if excluded_apps is not None:
            self.config.excluded_apps = excluded_apps
        if excluded_routes is not None:
            self.config.excluded_routes = excluded_routes

        logger.info(f"Split tunnel {'enabled' if enabled else 'disabled'}")

    async def rotate_keys(self) -> Tuple[str, str]:
        """Rotate VPN keys.

        Returns:
            Tuple of (new_private_key, new_public_key)
        """
        old_public = self.config.public_key

        # Generate new keys
        self.config.private_key = VPNConfig._generate_key()
        self.config.public_key = VPNConfig._derive_public_key(self.config.private_key)

        logger.info(f"Rotated VPN keys: {old_public[:8]}... -> {self.config.public_key[:8]}...")

        # Reconnect with new keys if connected
        if self._state == VPNState.CONNECTED:
            await self.reconnect()

        return self.config.private_key, self.config.public_key

    def get_diagnostics(self) -> Dict[str, Any]:
        """Get diagnostic information."""
        return {
            "state": self._state.value,
            "protocol": self.config.protocol.value,
            "server": f"{self.config.server_address}:{self.config.server_port}",
            "stats": self._stats.to_dict(),
            "config": {
                "split_tunnel": self.config.split_tunnel,
                "auto_reconnect": self.config.auto_reconnect,
                "mtu": self.config.mtu,
            },
            "reconnect_attempts": self._reconnect_attempts,
        }


class VPNManager:
    """Manages multiple VPN configurations and connections."""

    def __init__(self):
        """Initialize VPN manager."""
        self._tunnels: Dict[str, VPNTunnel] = {}
        self._active_tunnel: Optional[str] = None
        self._configs: Dict[str, VPNConfig] = {}

    def add_config(self, name: str, config: VPNConfig) -> None:
        """Add a VPN configuration.

        Args:
            name: Configuration name
            config: VPN configuration
        """
        self._configs[name] = config
        logger.info(f"Added VPN config: {name}")

    def remove_config(self, name: str) -> bool:
        """Remove a VPN configuration.

        Args:
            name: Configuration name

        Returns:
            True if removed
        """
        if name in self._configs:
            del self._configs[name]
            if name in self._tunnels:
                del self._tunnels[name]
            logger.info(f"Removed VPN config: {name}")
            return True
        return False

    def get_config(self, name: str) -> Optional[VPNConfig]:
        """Get a VPN configuration."""
        return self._configs.get(name)

    def list_configs(self) -> List[str]:
        """List all configuration names."""
        return list(self._configs.keys())

    async def connect(self, name: str) -> bool:
        """Connect to a VPN using named configuration.

        Args:
            name: Configuration name

        Returns:
            True if connected
        """
        if name not in self._configs:
            raise VPNError(f"Unknown VPN config: {name}")

        # Disconnect current if any
        if self._active_tunnel and self._active_tunnel != name:
            await self.disconnect()

        # Create tunnel if needed
        if name not in self._tunnels:
            self._tunnels[name] = VPNTunnel(self._configs[name])

        tunnel = self._tunnels[name]
        if await tunnel.connect():
            self._active_tunnel = name
            return True
        return False

    async def disconnect(self) -> None:
        """Disconnect active VPN."""
        if self._active_tunnel and self._active_tunnel in self._tunnels:
            await self._tunnels[self._active_tunnel].disconnect()
            self._active_tunnel = None

    def get_active_tunnel(self) -> Optional[VPNTunnel]:
        """Get the active tunnel instance."""
        if self._active_tunnel:
            return self._tunnels.get(self._active_tunnel)
        return None

    def get_status(self) -> Dict[str, Any]:
        """Get VPN manager status."""
        return {
            "active_config": self._active_tunnel,
            "available_configs": self.list_configs(),
            "active_tunnel": (
                self.get_active_tunnel().get_diagnostics() if self._active_tunnel else None
            ),
        }
