"""
Agent OS Boundary Module

Provides boundary enforcement integration between Agent-OS and the Boundary Daemon.
The Boundary Daemon is a hard trust enforcement layer that monitors system state
and enforces security policies.

Components:
- BoundaryClient: Client for communicating with the Boundary Daemon
- BoundaryDaemon: The daemon implementation (in daemon subpackage)

Usage:
    from src.boundary import BoundaryClient, create_boundary_client

    # Create client (connects to running daemon)
    client = create_boundary_client()

    # Check if operation is allowed
    if client.request_permission("network_access", "agent:sage"):
        # Perform operation
        pass
    else:
        # Access denied
        pass
"""

from .client import (
    BoundaryClient,
    BoundaryClientConfig,
    create_boundary_client,
)

from .daemon import (
    BoundaryDaemon,
    BoundaryConfig,
    BoundaryMode,
    RequestType,
    Decision,
    create_boundary_daemon,
)

__all__ = [
    # Client
    "BoundaryClient",
    "BoundaryClientConfig",
    "create_boundary_client",
    # Daemon
    "BoundaryDaemon",
    "BoundaryConfig",
    "BoundaryMode",
    "RequestType",
    "Decision",
    "create_boundary_daemon",
]
