"""
Smith Client - Agent-OS Integration

Client for communicating with the Smith Daemon (Agent Smith's system-level enforcer).
All Agent-OS components that need to perform security-sensitive operations
must request permission through this client.

This is part of Agent Smith's internal enforcement mechanism within Agent-OS,
distinct from the external boundary-daemon project.
"""

import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .daemon import (
    BoundaryMode,
    Decision,
    RequestType,
    SmithDaemon,
    SmithDaemonConfig,
    create_smith_daemon,
)

logger = logging.getLogger(__name__)


class ConnectionState(Enum):
    """Connection state to Smith daemon."""

    DISCONNECTED = "disconnected"
    CONNECTED = "connected"
    EMBEDDED = "embedded"  # Daemon running in-process
    FAILED = "failed"


@dataclass
class SmithClientConfig:
    """Configuration for the Smith client."""

    embedded: bool = True  # Run daemon in-process
    socket_path: Optional[Path] = None  # For external daemon
    log_path: Optional[Path] = None
    initial_mode: BoundaryMode = BoundaryMode.RESTRICTED
    network_allowed: bool = False
    fail_closed: bool = True  # Deny if daemon unavailable
    cache_decisions: bool = True
    cache_ttl_seconds: float = 1.0


# Backwards compatibility alias
BoundaryClientConfig = SmithClientConfig


@dataclass
class CachedDecision:
    """Cached permission decision."""

    allowed: bool
    timestamp: datetime
    request_type: str
    source: str
    target: str


class SmithClient:
    """
    Client for Agent Smith's system-level enforcement daemon.

    Provides a simple interface for Agent-OS components to request
    permission for security-sensitive operations.

    The client can either:
    1. Run the daemon in-process (embedded mode)
    2. Connect to an external daemon via socket

    For safety, the client "fails closed" by default - if the daemon
    is unavailable, operations are denied.
    """

    def __init__(self, config: Optional[SmithClientConfig] = None):
        """
        Initialize Smith client.

        Args:
            config: Client configuration
        """
        self.config = config or SmithClientConfig()
        self._daemon: Optional[SmithDaemon] = None
        self._state = ConnectionState.DISCONNECTED
        self._lock = threading.Lock()
        self._decision_cache: Dict[str, CachedDecision] = {}
        self._request_count = 0
        self._denied_count = 0

        # Initialize based on mode
        if self.config.embedded:
            self._init_embedded()

    @property
    def is_connected(self) -> bool:
        """Check if connected to daemon."""
        return self._state in [ConnectionState.CONNECTED, ConnectionState.EMBEDDED]

    @property
    def mode(self) -> BoundaryMode:
        """Get current boundary mode."""
        if self._daemon:
            return self._daemon.mode
        return BoundaryMode.LOCKDOWN  # Fail closed

    @property
    def is_secure(self) -> bool:
        """Check if system is in secure state."""
        if self._daemon:
            return self._daemon.is_secure
        return False

    def connect(self) -> bool:
        """
        Connect to the Smith daemon.

        Returns:
            True if connection successful
        """
        with self._lock:
            if self._state == ConnectionState.EMBEDDED:
                return True

            if self.config.embedded:
                return self._init_embedded()

            # TODO: Implement socket connection
            logger.warning("Socket connection not implemented")
            return False

    def disconnect(self) -> None:
        """Disconnect from the daemon."""
        with self._lock:
            if self._daemon and self._state == ConnectionState.EMBEDDED:
                self._daemon.stop()
                self._daemon = None

            self._state = ConnectionState.DISCONNECTED

    def request_permission(
        self,
        request_type: str,
        source: str,
        target: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Request permission for an operation.

        This is the main entry point for security checks. All Agent-OS
        components must call this before performing sensitive operations.

        Args:
            request_type: Type of operation (network_access, file_write, etc.)
            source: Who is making the request (e.g., "agent:sage")
            target: What is being accessed
            metadata: Additional context

        Returns:
            True if operation is allowed

        Example:
            client = create_smith_client()

            # Check before network access
            if client.request_permission("network_access", "agent:researcher"):
                # Perform network operation
                response = requests.get(url)
            else:
                raise PermissionError("Network access denied by Smith")
        """
        self._request_count += 1

        # Check cache first
        if self.config.cache_decisions:
            cached = self._check_cache(request_type, source, target)
            if cached is not None:
                return cached

        # Check daemon
        if not self.is_connected:
            if not self.connect():
                logger.warning("Cannot connect to Smith daemon")
                if self.config.fail_closed:
                    self._denied_count += 1
                    return False
                return True  # Fail open (dangerous!)

        try:
            allowed = self._daemon.request_permission(
                request_type=request_type,
                source=source,
                target=target,
                metadata=metadata,
            )

            # Cache decision
            if self.config.cache_decisions:
                self._cache_decision(request_type, source, target, allowed)

            if not allowed:
                self._denied_count += 1

            return allowed

        except Exception as e:
            logger.error(f"Smith permission check failed: {e}")
            if self.config.fail_closed:
                self._denied_count += 1
                return False
            return True

    def check_network_access(
        self,
        source: str,
        destination: str = "",
    ) -> bool:
        """Check if network access is allowed."""
        return self.request_permission(
            "network_access",
            source,
            target=destination,
        )

    def check_file_write(
        self,
        source: str,
        file_path: str,
    ) -> bool:
        """Check if file write is allowed."""
        return self.request_permission(
            "file_write",
            source,
            target=file_path,
        )

    def check_process_spawn(
        self,
        source: str,
        process_name: str,
    ) -> bool:
        """Check if process spawn is allowed."""
        return self.request_permission(
            "process_spawn",
            source,
            target=process_name,
        )

    def check_memory_access(
        self,
        source: str,
        memory_key: str = "",
    ) -> bool:
        """Check if memory vault access is allowed."""
        return self.request_permission(
            "memory_access",
            source,
            target=memory_key,
        )

    def check_agent_activation(
        self,
        activator: str,
        agent_name: str,
    ) -> bool:
        """Check if agent activation is allowed."""
        return self.request_permission(
            "agent_activation",
            activator,
            target=agent_name,
        )

    def check_external_api(
        self,
        source: str,
        api_endpoint: str,
    ) -> bool:
        """Check if external API call is allowed."""
        return self.request_permission(
            "external_api",
            source,
            target=api_endpoint,
        )

    def lockdown(self, reason: str = "Client requested lockdown") -> bool:
        """Request system lockdown."""
        if self._daemon:
            return self._daemon.lockdown(reason)
        return False

    def set_mode(
        self,
        mode: BoundaryMode,
        reason: str = "",
        authorization: Optional[str] = None,
    ) -> bool:
        """Request mode change."""
        if self._daemon:
            return self._daemon.set_mode(mode, reason, authorization)
        return False

    def get_status(self) -> Dict[str, Any]:
        """Get Smith daemon status."""
        status = {
            "client_state": self._state.value,
            "request_count": self._request_count,
            "denied_count": self._denied_count,
            "cache_size": len(self._decision_cache),
        }

        if self._daemon:
            status["daemon"] = self._daemon.get_status()

        return status

    def add_whitelist(
        self,
        request_type: str,
        target: str,
        mode: Optional[BoundaryMode] = None,
    ) -> None:
        """Add target to whitelist."""
        if self._daemon:
            target_mode = mode or self.mode
            self._daemon.add_to_whitelist(target_mode, request_type, target)

            # Invalidate cache entries for this target
            cache_prefix = f"{request_type.lower()}:"
            keys_to_remove = [
                k
                for k in self._decision_cache.keys()
                if k.startswith(cache_prefix) and k.endswith(f":{target}")
            ]
            for key in keys_to_remove:
                del self._decision_cache[key]

    def _init_embedded(self) -> bool:
        """Initialize embedded daemon."""
        try:
            daemon_config = SmithDaemonConfig(
                initial_mode=self.config.initial_mode,
                log_path=self.config.log_path,
                network_allowed=self.config.network_allowed,
            )

            self._daemon = SmithDaemon(daemon_config)
            self._daemon.start()
            self._state = ConnectionState.EMBEDDED

            logger.info("Embedded Smith daemon started")
            return True

        except Exception as e:
            logger.error(f"Failed to start embedded Smith daemon: {e}")
            self._state = ConnectionState.FAILED
            return False

    def _check_cache(
        self,
        request_type: str,
        source: str,
        target: str,
    ) -> Optional[bool]:
        """Check decision cache."""
        cache_key = f"{request_type}:{source}:{target}"
        cached = self._decision_cache.get(cache_key)

        if cached:
            age = (datetime.now() - cached.timestamp).total_seconds()
            if age < self.config.cache_ttl_seconds:
                return cached.allowed

            # Expired, remove from cache
            del self._decision_cache[cache_key]

        return None

    def _cache_decision(
        self,
        request_type: str,
        source: str,
        target: str,
        allowed: bool,
    ) -> None:
        """Cache a decision."""
        cache_key = f"{request_type}:{source}:{target}"
        self._decision_cache[cache_key] = CachedDecision(
            allowed=allowed,
            timestamp=datetime.now(),
            request_type=request_type,
            source=source,
            target=target,
        )

        # Limit cache size
        if len(self._decision_cache) > 1000:
            # Remove oldest entries
            sorted_keys = sorted(
                self._decision_cache.keys(),
                key=lambda k: self._decision_cache[k].timestamp,
            )
            for key in sorted_keys[:100]:
                del self._decision_cache[key]


# Backwards compatibility alias
BoundaryClient = SmithClient


def create_smith_client(
    embedded: bool = True,
    initial_mode: BoundaryMode = BoundaryMode.RESTRICTED,
    fail_closed: bool = True,
    **kwargs,
) -> SmithClient:
    """
    Factory function to create a Smith client.

    Args:
        embedded: Run daemon in-process
        initial_mode: Starting boundary mode
        fail_closed: Deny operations if daemon unavailable
        **kwargs: Additional configuration

    Returns:
        Configured SmithClient instance
    """
    config = SmithClientConfig(
        embedded=embedded,
        initial_mode=initial_mode,
        fail_closed=fail_closed,
        **kwargs,
    )

    return SmithClient(config)


# Backwards compatibility alias
create_boundary_client = create_smith_client


# Convenience function for quick permission checks
def require_permission(
    request_type: str,
    source: str,
    target: str = "",
    client: Optional[SmithClient] = None,
) -> bool:
    """
    Quick permission check using default client.

    Args:
        request_type: Type of operation
        source: Requesting component
        target: Target resource
        client: Optional client (creates default if None)

    Returns:
        True if allowed
    """
    if client is None:
        client = create_smith_client()

    return client.request_permission(request_type, source, target)
