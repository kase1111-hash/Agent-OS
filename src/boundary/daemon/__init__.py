"""
Smith Daemon - Agent Smith's System-Level Enforcement Layer

The Smith Daemon is Agent Smith's system-level security enforcement layer that
monitors system state and enforces security policies through tripwires and
enforcement actions. This complements Smith's request-level validation (S1-S12 checks).

Note: This is distinct from the external boundary-daemon project
(https://github.com/kase1111-hash/boundary-daemon-). This module is
Agent Smith's internal enforcement mechanism within Agent-OS.

Key Components:
- StateMonitor: Monitors network, hardware, and process state
- TripwireSystem: Security triggers that activate on violations
- PolicyEngine: Manages security modes (Lockdown/Restricted/Trusted/Emergency)
- EnforcementLayer: Executes halt/suspend/isolate/lockdown actions
- ImmutableEventLog: Append-only audit log with hash chain

Usage:
    from src.boundary.daemon import SmithDaemon, create_smith_daemon

    # Create and start daemon
    daemon = create_smith_daemon(log_path=Path("/var/log/smith.log"))
    daemon.start()

    # Check current mode
    print(f"Mode: {daemon.mode}")

    # Request permission for an operation
    allowed = daemon.request_permission("network_access", "agent:sage")
    if not allowed:
        print("Access denied")

    # Stop daemon
    daemon.stop()

    # Backwards compatible aliases also available:
    # BoundaryDaemon, create_boundary_daemon
"""

from .boundary_daemon import (  # Primary exports (new naming); Backwards compatibility aliases
    BoundaryConfig,
    BoundaryDaemon,
    SmithDaemon,
    SmithDaemonConfig,
    create_boundary_daemon,
    create_smith_daemon,
)
from .enforcement import (
    EnforcementAction,
    EnforcementEvent,
    EnforcementLayer,
    EnforcementPolicy,
    EnforcementSeverity,
    create_enforcement_layer,
)
from .event_log import (
    ImmutableEventLog,
    LogEntry,
    create_event_log,
)
from .policy_engine import (
    BoundaryMode,
    Decision,
    PolicyDecision,
    PolicyEngine,
    PolicyRequest,
    PolicyRule,
    RequestType,
    create_policy_engine,
)
from .state_monitor import (
    HardwareState,
    NetworkConnection,
    NetworkState,
    ProcessState,
    StateMonitor,
    SystemState,
    create_state_monitor,
)
from .tripwires import (
    Tripwire,
    TripwireEvent,
    TripwireState,
    TripwireSystem,
    TripwireType,
    create_file_tripwire,
    create_process_tripwire,
    create_tripwire_system,
)

__all__ = [
    # Main daemon (new naming)
    "SmithDaemon",
    "SmithDaemonConfig",
    "create_smith_daemon",
    # Backwards compatibility
    "BoundaryDaemon",
    "BoundaryConfig",
    "create_boundary_daemon",
    # State monitor
    "StateMonitor",
    "SystemState",
    "NetworkState",
    "ProcessState",
    "HardwareState",
    "NetworkConnection",
    "create_state_monitor",
    # Tripwires
    "TripwireSystem",
    "Tripwire",
    "TripwireEvent",
    "TripwireType",
    "TripwireState",
    "create_tripwire_system",
    "create_file_tripwire",
    "create_process_tripwire",
    # Policy
    "PolicyEngine",
    "PolicyRequest",
    "PolicyDecision",
    "PolicyRule",
    "BoundaryMode",
    "RequestType",
    "Decision",
    "create_policy_engine",
    # Enforcement
    "EnforcementLayer",
    "EnforcementEvent",
    "EnforcementAction",
    "EnforcementSeverity",
    "EnforcementPolicy",
    "create_enforcement_layer",
    # Event log
    "ImmutableEventLog",
    "LogEntry",
    "create_event_log",
]
