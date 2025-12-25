"""
Agent OS Boundary Module - Smith's System-Level Enforcement

Provides the integration layer between Agent-OS and Agent Smith's system-level
security enforcement (the Smith Daemon). This complements Smith's request-level
validation (S1-S12 checks in src/agents/smith/).

Note: This is distinct from the external boundary-daemon project
(https://github.com/kase1111-hash/boundary-daemon-). This module is
Agent Smith's internal enforcement mechanism within Agent-OS.

Components:
- SmithDaemon: The system-level security daemon (formerly BoundaryDaemon)
- SmithClient: Client for communicating with the Smith Daemon
- Policy Engine: Manages security modes (Lockdown/Restricted/Trusted)
- Tripwire System: Security triggers for violations
- Enforcement Layer: Executes halt/suspend/lockdown actions

Usage:
    from src.boundary import SmithClient, create_smith_client

    # Create client (connects to running daemon)
    client = create_smith_client()

    # Check if operation is allowed
    if client.request_permission("network_access", "agent:sage"):
        # Perform operation
        pass
    else:
        # Access denied
        pass

    # Backwards compatible imports also available:
    # BoundaryClient, create_boundary_client
"""

from .client import (
    # Primary exports (new naming)
    SmithClient,
    SmithClientConfig,
    create_smith_client,
    # Backwards compatibility aliases
    BoundaryClient,
    BoundaryClientConfig,
    create_boundary_client,
)

from .daemon import (
    # Primary exports (new naming)
    SmithDaemon,
    SmithDaemonConfig,
    create_smith_daemon,
    # Backwards compatibility aliases
    BoundaryDaemon,
    BoundaryConfig,
    create_boundary_daemon,
    # Shared types
    BoundaryMode,
    RequestType,
    Decision,
)

__all__ = [
    # Client (new naming)
    "SmithClient",
    "SmithClientConfig",
    "create_smith_client",
    # Client (backwards compatibility)
    "BoundaryClient",
    "BoundaryClientConfig",
    "create_boundary_client",
    # Daemon (new naming)
    "SmithDaemon",
    "SmithDaemonConfig",
    "create_smith_daemon",
    # Daemon (backwards compatibility)
    "BoundaryDaemon",
    "BoundaryConfig",
    "create_boundary_daemon",
    # Shared types
    "BoundaryMode",
    "RequestType",
    "Decision",
]
