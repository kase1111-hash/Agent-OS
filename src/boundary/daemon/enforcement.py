"""
Smith Daemon Enforcement Layer

Executes enforcement actions when policy violations or tripwires are triggered.
Actions include: halt, suspend, isolate, alert, lockdown, shutdown.

This is part of Agent Smith's system-level enforcement mechanism within Agent-OS,
distinct from the external boundary-daemon project.
"""

import logging
import os
import signal
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


class EnforcementAction(Enum):
    """Types of enforcement actions."""

    ALERT = auto()  # Log and notify
    SUSPEND = auto()  # Suspend operations temporarily
    ISOLATE = auto()  # Isolate from external resources
    HALT = auto()  # Halt all operations
    LOCKDOWN = auto()  # Full system lockdown
    SHUTDOWN = auto()  # Graceful shutdown


class EnforcementSeverity(Enum):
    """Severity levels for enforcement."""

    LOW = 1  # Informational
    MEDIUM = 2  # Warning
    HIGH = 3  # Action required
    CRITICAL = 4  # Immediate action
    EMERGENCY = 5  # System emergency


@dataclass
class EnforcementEvent:
    """Record of an enforcement action."""

    event_id: str
    action: EnforcementAction
    severity: EnforcementSeverity
    reason: str
    timestamp: datetime = field(default_factory=datetime.now)
    source: str = ""  # What triggered this
    target: str = ""  # What was affected
    success: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            "event_id": self.event_id,
            "action": self.action.name,
            "severity": self.severity.name,
            "reason": self.reason,
            "timestamp": self.timestamp.isoformat(),
            "source": self.source,
            "target": self.target,
            "success": self.success,
            "metadata": self.metadata,
        }


@dataclass
class EnforcementPolicy:
    """Policy for automatic enforcement actions."""

    id: str
    trigger_severity: EnforcementSeverity
    action: EnforcementAction
    description: str
    enabled: bool = True
    cooldown_seconds: float = 60.0  # Minimum time between actions
    last_triggered: Optional[datetime] = None


class EnforcementLayer:
    """
    Enforcement layer for the Boundary Daemon.

    Executes security enforcement actions based on policy
    violations and tripwire triggers.
    """

    def __init__(
        self,
        on_enforcement: Optional[Callable[[EnforcementEvent], None]] = None,
        auto_lockdown_on_critical: bool = True,
    ):
        """
        Initialize enforcement layer.

        Args:
            on_enforcement: Callback when enforcement action is taken
            auto_lockdown_on_critical: Auto-lockdown on critical events
        """
        self.on_enforcement = on_enforcement
        self.auto_lockdown_on_critical = auto_lockdown_on_critical

        self._events: List[EnforcementEvent] = []
        self._policies: Dict[str, EnforcementPolicy] = {}
        self._lock = threading.Lock()
        self._suspended = False
        self._isolated = False
        self._halted = False
        self._event_counter = 0

        # Registered handlers for different actions
        self._handlers: Dict[EnforcementAction, List[Callable[[], bool]]] = {
            action: [] for action in EnforcementAction
        }

        # Install default policies
        self._install_default_policies()

    @property
    def is_suspended(self) -> bool:
        """Check if operations are suspended."""
        return self._suspended

    @property
    def is_isolated(self) -> bool:
        """Check if system is isolated."""
        return self._isolated

    @property
    def is_halted(self) -> bool:
        """Check if system is halted."""
        return self._halted

    def register_handler(
        self,
        action: EnforcementAction,
        handler: Callable[[], bool],
    ) -> None:
        """Register a handler for an enforcement action."""
        self._handlers[action].append(handler)

    def add_policy(self, policy: EnforcementPolicy) -> None:
        """Add an enforcement policy."""
        with self._lock:
            self._policies[policy.id] = policy

    def remove_policy(self, policy_id: str) -> bool:
        """Remove an enforcement policy."""
        with self._lock:
            if policy_id in self._policies:
                del self._policies[policy_id]
                return True
            return False

    def enforce(
        self,
        action: EnforcementAction,
        reason: str,
        severity: EnforcementSeverity = EnforcementSeverity.MEDIUM,
        source: str = "",
        target: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> EnforcementEvent:
        """
        Execute an enforcement action.

        Args:
            action: The action to take
            reason: Why this action is being taken
            severity: Severity level
            source: What triggered this
            target: What is being affected
            metadata: Additional context

        Returns:
            EnforcementEvent record
        """
        with self._lock:
            self._event_counter += 1
            event_id = f"ENF-{self._event_counter:06d}"

            event = EnforcementEvent(
                event_id=event_id,
                action=action,
                severity=severity,
                reason=reason,
                source=source,
                target=target,
                metadata=metadata or {},
            )

            # Execute the action
            success = self._execute_action(action)
            event.success = success

            # Record event
            self._events.append(event)

            # Log based on severity
            log_level = {
                EnforcementSeverity.LOW: logging.INFO,
                EnforcementSeverity.MEDIUM: logging.WARNING,
                EnforcementSeverity.HIGH: logging.WARNING,
                EnforcementSeverity.CRITICAL: logging.ERROR,
                EnforcementSeverity.EMERGENCY: logging.CRITICAL,
            }.get(severity, logging.INFO)

            logger.log(log_level, f"Enforcement: {action.name} - {reason} (success={success})")

            # Notify callback
            if self.on_enforcement:
                try:
                    self.on_enforcement(event)
                except Exception as e:
                    logger.error(f"Enforcement callback error: {e}")

            # Auto-lockdown on critical
            if self.auto_lockdown_on_critical and severity == EnforcementSeverity.CRITICAL:
                if action != EnforcementAction.LOCKDOWN:
                    self._execute_lockdown()

            return event

    def alert(self, reason: str, source: str = "", **kwargs) -> EnforcementEvent:
        """Issue an alert."""
        return self.enforce(
            EnforcementAction.ALERT,
            reason,
            EnforcementSeverity.LOW,
            source=source,
            **kwargs,
        )

    def suspend(self, reason: str, source: str = "", **kwargs) -> EnforcementEvent:
        """Suspend operations."""
        return self.enforce(
            EnforcementAction.SUSPEND,
            reason,
            EnforcementSeverity.HIGH,
            source=source,
            **kwargs,
        )

    def isolate(self, reason: str, source: str = "", **kwargs) -> EnforcementEvent:
        """Isolate from external resources."""
        return self.enforce(
            EnforcementAction.ISOLATE,
            reason,
            EnforcementSeverity.HIGH,
            source=source,
            **kwargs,
        )

    def halt(self, reason: str, source: str = "", **kwargs) -> EnforcementEvent:
        """Halt all operations."""
        return self.enforce(
            EnforcementAction.HALT,
            reason,
            EnforcementSeverity.CRITICAL,
            source=source,
            **kwargs,
        )

    def lockdown(self, reason: str, source: str = "", **kwargs) -> EnforcementEvent:
        """Full system lockdown."""
        return self.enforce(
            EnforcementAction.LOCKDOWN,
            reason,
            EnforcementSeverity.EMERGENCY,
            source=source,
            **kwargs,
        )

    def resume(self, authorization_code: str) -> bool:
        """
        Resume from suspended/halted state.

        Args:
            authorization_code: Human authorization

        Returns:
            True if resume successful
        """
        if len(authorization_code) < 8:
            logger.warning("Resume attempt with invalid authorization")
            return False

        with self._lock:
            self._suspended = False
            self._halted = False
            logger.info("Operations resumed with authorization")

        # Log outside the lock to avoid deadlock
        self.enforce(
            EnforcementAction.ALERT,
            "Operations resumed",
            EnforcementSeverity.MEDIUM,
            source="human",
            metadata={"action": "resume"},
        )

        return True

    def get_events(self) -> List[EnforcementEvent]:
        """Get enforcement event history."""
        return list(self._events)

    def get_status(self) -> Dict[str, Any]:
        """Get current enforcement status."""
        return {
            "suspended": self._suspended,
            "isolated": self._isolated,
            "halted": self._halted,
            "event_count": len(self._events),
            "policies": len(self._policies),
        }

    def check_policies(self, severity: EnforcementSeverity, source: str) -> None:
        """
        Check if any policies should trigger based on severity.

        Args:
            severity: Severity of the triggering event
            source: What triggered this check
        """
        now = datetime.now()

        for policy in self._policies.values():
            if not policy.enabled:
                continue

            if severity.value >= policy.trigger_severity.value:
                # Check cooldown
                if policy.last_triggered:
                    elapsed = (now - policy.last_triggered).total_seconds()
                    if elapsed < policy.cooldown_seconds:
                        continue

                policy.last_triggered = now
                self.enforce(
                    policy.action,
                    policy.description,
                    severity,
                    source=source,
                    metadata={"policy_id": policy.id},
                )

    def _execute_action(self, action: EnforcementAction) -> bool:
        """Execute an enforcement action."""
        try:
            if action == EnforcementAction.ALERT:
                return self._execute_alert()
            elif action == EnforcementAction.SUSPEND:
                return self._execute_suspend()
            elif action == EnforcementAction.ISOLATE:
                return self._execute_isolate()
            elif action == EnforcementAction.HALT:
                return self._execute_halt()
            elif action == EnforcementAction.LOCKDOWN:
                return self._execute_lockdown()
            elif action == EnforcementAction.SHUTDOWN:
                return self._execute_shutdown()

            # Execute registered handlers
            success = True
            for handler in self._handlers.get(action, []):
                try:
                    if not handler():
                        success = False
                except Exception as e:
                    logger.error(f"Handler error for {action}: {e}")
                    success = False

            return success

        except Exception as e:
            logger.error(f"Enforcement action error: {e}")
            return False

    def _execute_alert(self) -> bool:
        """Execute alert action."""
        # In production, would send notifications
        return True

    def _execute_suspend(self) -> bool:
        """Execute suspend action."""
        self._suspended = True
        return True

    def _execute_isolate(self) -> bool:
        """Execute isolate action."""
        self._isolated = True
        # In production, would disable network interfaces
        return True

    def _execute_halt(self) -> bool:
        """Execute halt action."""
        self._halted = True
        self._suspended = True
        return True

    def _execute_lockdown(self) -> bool:
        """Execute lockdown action."""
        self._halted = True
        self._suspended = True
        self._isolated = True
        return True

    def _execute_shutdown(self) -> bool:
        """Execute graceful shutdown."""
        self._halted = True
        self._suspended = True
        # In production, would signal main process to exit
        return True

    def _install_default_policies(self) -> None:
        """Install default enforcement policies."""
        # Auto-suspend on high severity
        self.add_policy(
            EnforcementPolicy(
                id="auto_suspend_high",
                trigger_severity=EnforcementSeverity.HIGH,
                action=EnforcementAction.SUSPEND,
                description="Auto-suspend on high severity event",
                cooldown_seconds=30.0,
            )
        )

        # Auto-lockdown on emergency
        self.add_policy(
            EnforcementPolicy(
                id="auto_lockdown_emergency",
                trigger_severity=EnforcementSeverity.EMERGENCY,
                action=EnforcementAction.LOCKDOWN,
                description="Auto-lockdown on emergency",
                cooldown_seconds=0.0,  # No cooldown for emergencies
            )
        )


def create_enforcement_layer(
    on_enforcement: Optional[Callable[[EnforcementEvent], None]] = None,
    auto_lockdown_on_critical: bool = True,
) -> EnforcementLayer:
    """Factory function to create an enforcement layer."""
    return EnforcementLayer(
        on_enforcement=on_enforcement,
        auto_lockdown_on_critical=auto_lockdown_on_critical,
    )
