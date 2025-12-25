"""
Smith Daemon - Main Daemon Implementation

This module provides Agent Smith's system-level security enforcement layer.
It ties together all Smith daemon components into a unified security
enforcement system that monitors system state, enforces policies through
tripwires, and takes enforcement actions when violations are detected.

Note: This is distinct from the external boundary-daemon project
(https://github.com/kase1111-hash/boundary-daemon-). This module is
Agent Smith's internal enforcement mechanism within Agent-OS.
"""

import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .state_monitor import (
    StateMonitor,
    SystemState,
    NetworkState,
    create_state_monitor,
)
from .tripwires import (
    TripwireSystem,
    TripwireEvent,
    create_tripwire_system,
)
from .policy_engine import (
    PolicyEngine,
    PolicyRequest,
    PolicyDecision,
    BoundaryMode,
    RequestType,
    Decision,
    create_policy_engine,
)
from .enforcement import (
    EnforcementLayer,
    EnforcementEvent,
    EnforcementSeverity,
    create_enforcement_layer,
)
from .event_log import (
    ImmutableEventLog,
    create_event_log,
)


logger = logging.getLogger(__name__)


@dataclass
class SmithDaemonConfig:
    """Configuration for the Smith Daemon (Agent Smith's system-level enforcer)."""
    initial_mode: BoundaryMode = BoundaryMode.RESTRICTED
    log_path: Optional[Path] = None
    state_poll_interval: float = 1.0
    tripwire_check_interval: float = 0.5
    network_allowed: bool = False
    auto_lockdown_on_critical: bool = True
    verify_log_on_load: bool = True
    socket_path: Optional[Path] = None  # For local API


# Backwards compatibility alias
BoundaryConfig = SmithDaemonConfig


class SmithDaemon:
    """
    Agent Smith's system-level enforcement daemon.

    This is the "hard" security layer that operates at the system level,
    complementing Smith's request-level validation (S1-S12 checks).

    Provides:
    - System state monitoring (network, processes, hardware)
    - Tripwire-based security triggers
    - Policy-based access control (Lockdown/Restricted/Trusted modes)
    - Enforcement actions (halt/suspend/lockdown)
    - Immutable audit logging with hash chain

    Note: This is Agent Smith's internal enforcement mechanism,
    distinct from the external boundary-daemon project.
    """

    def __init__(self, config: Optional[SmithDaemonConfig] = None):
        """
        Initialize Smith Daemon.

        Args:
            config: Daemon configuration
        """
        self.config = config or SmithDaemonConfig()
        self._running = False
        self._started_at: Optional[datetime] = None

        # Initialize components
        self._event_log = create_event_log(
            log_path=self.config.log_path,
            verify_on_load=self.config.verify_log_on_load,
        )

        self._state_monitor = create_state_monitor(
            poll_interval=self.config.state_poll_interval,
            network_allowed=self.config.network_allowed,
        )

        self._tripwires = create_tripwire_system(
            check_interval=self.config.tripwire_check_interval,
            on_trigger=self._on_tripwire_triggered,
        )

        self._policy_engine = create_policy_engine(
            initial_mode=self.config.initial_mode,
            on_mode_change=self._on_mode_change,
        )

        self._enforcement = create_enforcement_layer(
            on_enforcement=self._on_enforcement,
            auto_lockdown_on_critical=self.config.auto_lockdown_on_critical,
        )

        # Register state monitor callback
        self._state_monitor.register_callback(self._on_state_change)

        # Log initialization
        self._event_log.log_event(
            "smith_daemon_init",
            mode=self.config.initial_mode.name,
            config=str(self.config),
        )

    @property
    def mode(self) -> BoundaryMode:
        """Get current boundary mode."""
        return self._policy_engine.mode

    @property
    def is_running(self) -> bool:
        """Check if daemon is running."""
        return self._running

    @property
    def is_secure(self) -> bool:
        """Check if system is in secure state."""
        state = self._state_monitor.get_current_state()
        return state.is_secure() and not self._tripwires.is_triggered()

    def start(self) -> None:
        """Start the Smith daemon."""
        if self._running:
            logger.warning("Smith daemon already running")
            return

        self._running = True
        self._started_at = datetime.now()

        # Start components
        self._state_monitor.start()
        self._tripwires.start()

        self._event_log.log_event("smith_daemon_started", mode=self.mode.name)
        logger.info(f"Smith daemon started in {self.mode.name} mode")

    def stop(self) -> None:
        """Stop the Smith daemon."""
        if not self._running:
            return

        self._running = False

        # Stop components
        self._state_monitor.stop()
        self._tripwires.stop()

        self._event_log.log_event("smith_daemon_stopped")
        logger.info("Smith daemon stopped")

    def request_permission(
        self,
        request_type: str,
        source: str,
        target: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Request permission for an operation.

        Args:
            request_type: Type of request (network_access, file_write, etc.)
            source: Who is making the request
            target: What is being accessed
            metadata: Additional context

        Returns:
            True if operation is allowed
        """
        # Check if halted
        if self._enforcement.is_halted:
            logger.warning(f"Permission denied - system halted: {request_type}")
            return False

        # Parse request type
        try:
            req_type = RequestType[request_type.upper()]
        except KeyError:
            req_type = RequestType.NETWORK_ACCESS  # Default

        # Create policy request
        request = PolicyRequest(
            request_id=str(uuid.uuid4()),
            request_type=req_type,
            source=source,
            target=target,
            metadata=metadata or {},
        )

        # Evaluate policy
        decision = self._policy_engine.evaluate(request)

        # Log decision
        self._event_log.log_policy_decision(
            request_type=request_type,
            decision=decision.decision.name,
            reason=decision.reason,
            source=source,
            target=target,
        )

        if decision.decision == Decision.ESCALATE:
            logger.info(f"Permission escalated for human approval: {request_type}")
            return False  # Deny until human approves

        return decision.decision in [Decision.ALLOW, Decision.AUDIT]

    def lockdown(self, reason: str = "Manual lockdown") -> bool:
        """
        Enter lockdown mode.

        Args:
            reason: Reason for lockdown

        Returns:
            True if lockdown successful
        """
        success = self._policy_engine.lockdown(reason)
        if success:
            self._enforcement.lockdown(reason)
            self._event_log.log_mode_change(
                old_mode=self.mode.name,
                new_mode=BoundaryMode.LOCKDOWN.name,
                reason=reason,
            )
        return success

    def set_mode(
        self,
        mode: BoundaryMode,
        reason: str = "",
        authorization: Optional[str] = None,
    ) -> bool:
        """
        Set boundary mode.

        Args:
            mode: Target mode
            reason: Reason for change
            authorization: Required for lowering security

        Returns:
            True if mode change successful
        """
        old_mode = self.mode
        success = self._policy_engine.set_mode(mode, reason, authorization)

        if success:
            # Resume from halt if moving to less restrictive mode
            if mode in [BoundaryMode.TRUSTED, BoundaryMode.RESTRICTED]:
                if authorization:
                    self._enforcement.resume(authorization)

        return success

    def reset_tripwires(self, authorization: str) -> int:
        """
        Reset all triggered tripwires.

        Args:
            authorization: Human authorization code

        Returns:
            Number of tripwires reset
        """
        count = self._tripwires.reset_all(authorization)
        if count > 0:
            self._event_log.log_event(
                "tripwires_reset",
                count=count,
            )
        return count

    def get_status(self) -> Dict[str, Any]:
        """Get daemon status."""
        state = self._state_monitor.get_current_state()
        triggered = self._tripwires.get_triggered()

        return {
            "running": self._running,
            "started_at": self._started_at.isoformat() if self._started_at else None,
            "mode": self.mode.name,
            "is_secure": self.is_secure,
            "system_state": state.to_dict(),
            "triggered_tripwires": [tw.id for tw in triggered],
            "enforcement": self._enforcement.get_status(),
            "event_count": self._event_log.count(),
        }

    def get_event_log(self, count: int = 100) -> List[Dict[str, Any]]:
        """Get recent event log entries."""
        entries = self._event_log.get_latest(count)
        return [e.to_dict() for e in entries]

    def verify_log_integrity(self) -> tuple[bool, List[str]]:
        """Verify event log integrity."""
        return self._event_log.verify_integrity()

    def add_to_whitelist(
        self,
        mode: BoundaryMode,
        request_type: str,
        target: str,
    ) -> None:
        """Add a target to the whitelist."""
        try:
            req_type = RequestType[request_type.upper()]
            self._policy_engine.whitelist(mode, req_type, target)
            self._event_log.log_event(
                "whitelist_added",
                mode=mode.name,
                request_type=request_type,
                target=target,
            )
        except KeyError:
            logger.warning(f"Unknown request type: {request_type}")

    def _on_state_change(self, state: SystemState) -> None:
        """Handle system state changes."""
        # Check for security violations
        if state.network_state == NetworkState.ONLINE and not self.config.network_allowed:
            self._event_log.log_event(
                "security_violation",
                type="network_access",
                details="External network access detected",
            )

            if self.mode != BoundaryMode.TRUSTED:
                self._enforcement.alert(
                    "External network access detected",
                    source="state_monitor",
                )

        if state.suspicious_processes:
            self._event_log.log_event(
                "security_warning",
                type="suspicious_process",
                processes=state.suspicious_processes,
            )

    def _on_tripwire_triggered(self, event: TripwireEvent) -> None:
        """Handle tripwire triggers."""
        # Log the event
        self._event_log.log_tripwire(
            tripwire_id=event.tripwire_id,
            reason=event.trigger_reason,
            severity=event.severity,
            data=event.trigger_data,
        )

        # Determine enforcement action based on severity
        severity = EnforcementSeverity(min(event.severity, 5))

        if event.severity >= 4:
            self._enforcement.lockdown(
                f"Tripwire triggered: {event.trigger_reason}",
                source=f"tripwire:{event.tripwire_id}",
            )
        elif event.severity >= 3:
            self._enforcement.suspend(
                f"Tripwire triggered: {event.trigger_reason}",
                source=f"tripwire:{event.tripwire_id}",
            )
        else:
            self._enforcement.alert(
                f"Tripwire triggered: {event.trigger_reason}",
                source=f"tripwire:{event.tripwire_id}",
            )

    def _on_mode_change(
        self,
        old_mode: BoundaryMode,
        new_mode: BoundaryMode,
    ) -> None:
        """Handle mode changes."""
        self._event_log.log_mode_change(
            old_mode=old_mode.name,
            new_mode=new_mode.name,
            reason="Mode transition",
        )

    def _on_enforcement(self, event: EnforcementEvent) -> None:
        """Handle enforcement events."""
        self._event_log.log_enforcement(
            action=event.action.name,
            reason=event.reason,
            success=event.success,
            severity=event.severity.name,
        )


def create_smith_daemon(
    initial_mode: BoundaryMode = BoundaryMode.RESTRICTED,
    log_path: Optional[Path] = None,
    network_allowed: bool = False,
    **kwargs,
) -> SmithDaemon:
    """
    Factory function to create a Smith Daemon.

    Args:
        initial_mode: Starting boundary mode
        log_path: Path for event log
        network_allowed: Whether network access is allowed
        **kwargs: Additional configuration options

    Returns:
        Configured SmithDaemon instance
    """
    config = SmithDaemonConfig(
        initial_mode=initial_mode,
        log_path=log_path,
        network_allowed=network_allowed,
        **kwargs,
    )

    return SmithDaemon(config)


# Backwards compatibility aliases
BoundaryDaemon = SmithDaemon
create_boundary_daemon = create_smith_daemon
