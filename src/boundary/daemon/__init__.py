"""
Boundary Daemon - Hard Trust Enforcement Layer

The Boundary Daemon is a standalone security enforcement layer that monitors
system state and enforces security policies through tripwires and enforcement actions.

Key Components:
- StateMonitor: Monitors network, hardware, and process state
- TripwireSystem: Security triggers that activate on violations
- PolicyEngine: Manages boundary modes (Lockdown/Restricted/Trusted)
- EnforcementLayer: Executes halt/suspend/isolate actions
- ImmutableEventLog: Append-only audit log with hash chain

Usage:
    from src.boundary.daemon import BoundaryDaemon, create_boundary_daemon

    # Create and start daemon
    daemon = create_boundary_daemon(log_path=Path("/var/log/boundary.log"))
    daemon.start()

    # Check current mode
    print(f"Mode: {daemon.mode}")

    # Request permission for an operation
    allowed = daemon.request_permission("network_access", "agent:sage")
    if not allowed:
        print("Access denied")

    # Stop daemon
    daemon.stop()
"""

from .state_monitor import (
    StateMonitor,
    SystemState,
    NetworkState,
    ProcessState,
    HardwareState,
    NetworkConnection,
    create_state_monitor,
)

from .tripwires import (
    TripwireSystem,
    Tripwire,
    TripwireEvent,
    TripwireType,
    TripwireState,
    create_tripwire_system,
    create_file_tripwire,
    create_process_tripwire,
)

from .policy_engine import (
    PolicyEngine,
    PolicyRequest,
    PolicyDecision,
    PolicyRule,
    BoundaryMode,
    RequestType,
    Decision,
    create_policy_engine,
)

from .enforcement import (
    EnforcementLayer,
    EnforcementEvent,
    EnforcementAction,
    EnforcementSeverity,
    EnforcementPolicy,
    create_enforcement_layer,
)

from .event_log import (
    ImmutableEventLog,
    LogEntry,
    create_event_log,
)

from .boundary_daemon import (
    BoundaryDaemon,
    BoundaryConfig,
    create_boundary_daemon,
)

__all__ = [
    # Main daemon
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
