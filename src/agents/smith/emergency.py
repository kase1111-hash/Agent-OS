"""
Agent OS Smith Emergency Controls

Implements emergency control mechanisms:
- Safe mode trigger
- System halt capability
- Emergency lockdown
- Incident logging
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List, Dict, Any, Callable
from enum import Enum, auto
import logging
import threading
import os
import signal
import json
from pathlib import Path


logger = logging.getLogger(__name__)


class SystemMode(Enum):
    """System operating modes."""
    NORMAL = auto()      # Full functionality
    RESTRICTED = auto()  # Limited functionality
    SAFE = auto()        # Minimal functionality, no external access
    LOCKDOWN = auto()    # Emergency lockdown, no operations
    HALTED = auto()      # System halted


class IncidentSeverity(Enum):
    """Severity levels for security incidents."""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4
    EMERGENCY = 5


@dataclass
class SecurityIncident:
    """Record of a security incident."""
    incident_id: str
    severity: IncidentSeverity
    category: str
    description: str
    source_agent: Optional[str] = None
    request_id: Optional[str] = None
    triggered_by: str = ""
    response_action: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "incident_id": self.incident_id,
            "severity": self.severity.name,
            "category": self.category,
            "description": self.description,
            "source_agent": self.source_agent,
            "request_id": self.request_id,
            "triggered_by": self.triggered_by,
            "response_action": self.response_action,
            "details": self.details,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class ModeTransition:
    """Record of a system mode transition."""
    from_mode: SystemMode
    to_mode: SystemMode
    reason: str
    triggered_by: str
    incident_id: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)


class EmergencyControls:
    """
    Emergency control system for Agent OS.

    Provides:
    - Safe mode activation
    - System halt capability
    - Emergency lockdown
    - Incident logging and tracking
    """

    # Thresholds for automatic mode escalation
    MODE_ESCALATION_THRESHOLDS = {
        IncidentSeverity.HIGH: (3, 60),      # 3 high incidents in 60 seconds
        IncidentSeverity.CRITICAL: (1, 300), # 1 critical incident
        IncidentSeverity.EMERGENCY: (1, 0),  # Immediate
    }

    def __init__(
        self,
        incident_log_path: Optional[str] = None,
        auto_escalate: bool = True,
        notify_callback: Optional[Callable[[SecurityIncident], None]] = None,
    ):
        """
        Initialize emergency controls.

        Args:
            incident_log_path: Path to incident log file
            auto_escalate: Automatically escalate mode on incidents
            notify_callback: Callback for incident notifications
        """
        self._current_mode = SystemMode.NORMAL
        self._mode_lock = threading.RLock()
        self._incidents: List[SecurityIncident] = []
        self._mode_history: List[ModeTransition] = []
        self._incident_counter = 0

        self.incident_log_path = incident_log_path
        self.auto_escalate = auto_escalate
        self.notify_callback = notify_callback

        # Callbacks for mode changes
        self._mode_callbacks: List[Callable[[SystemMode, SystemMode], None]] = []

        # Initialize incident log
        if self.incident_log_path:
            self._init_incident_log()

    @property
    def current_mode(self) -> SystemMode:
        """Get current system mode."""
        with self._mode_lock:
            return self._current_mode

    @property
    def is_operational(self) -> bool:
        """Check if system is operational."""
        return self._current_mode not in (SystemMode.LOCKDOWN, SystemMode.HALTED)

    @property
    def is_restricted(self) -> bool:
        """Check if system is in any restricted mode."""
        return self._current_mode != SystemMode.NORMAL

    def trigger_safe_mode(
        self,
        reason: str,
        triggered_by: str = "system",
        incident_id: Optional[str] = None,
    ) -> bool:
        """
        Trigger safe mode.

        In safe mode:
        - No external network access
        - No file system writes outside sandbox
        - No subprocess execution
        - Memory operations suspended

        Args:
            reason: Why safe mode was triggered
            triggered_by: Who/what triggered safe mode
            incident_id: Associated incident ID

        Returns:
            True if mode change successful
        """
        return self._transition_mode(
            SystemMode.SAFE,
            reason=reason,
            triggered_by=triggered_by,
            incident_id=incident_id,
        )

    def trigger_lockdown(
        self,
        reason: str,
        triggered_by: str = "system",
        incident_id: Optional[str] = None,
    ) -> bool:
        """
        Trigger emergency lockdown.

        In lockdown:
        - All agent operations suspended
        - Only emergency control commands accepted
        - All pending requests dropped

        Args:
            reason: Why lockdown was triggered
            triggered_by: Who/what triggered lockdown
            incident_id: Associated incident ID

        Returns:
            True if mode change successful
        """
        return self._transition_mode(
            SystemMode.LOCKDOWN,
            reason=reason,
            triggered_by=triggered_by,
            incident_id=incident_id,
        )

    def halt_system(
        self,
        reason: str,
        triggered_by: str = "system",
        incident_id: Optional[str] = None,
    ) -> None:
        """
        Halt the system completely.

        This is the nuclear option - the system will stop processing
        and require manual restart.

        Args:
            reason: Why system was halted
            triggered_by: Who/what triggered halt
            incident_id: Associated incident ID
        """
        logger.critical(f"SYSTEM HALT triggered by {triggered_by}: {reason}")

        # Log the incident
        incident = self.log_incident(
            severity=IncidentSeverity.EMERGENCY,
            category="system_halt",
            description=f"System halt: {reason}",
            triggered_by=triggered_by,
        )

        # Transition to halted
        self._transition_mode(
            SystemMode.HALTED,
            reason=reason,
            triggered_by=triggered_by,
            incident_id=incident.incident_id,
        )

        # Write final log entry
        self._write_halt_log(reason, triggered_by)

        # Signal termination (in production, this would stop all processes)
        # For now, we just set the mode - actual termination would be handled by supervisor

    def restore_normal(
        self,
        authorized_by: str,
        verification_code: Optional[str] = None,
    ) -> bool:
        """
        Restore system to normal operation.

        Args:
            authorized_by: Who is authorizing the restore
            verification_code: Optional verification code

        Returns:
            True if restore successful
        """
        with self._mode_lock:
            if self._current_mode == SystemMode.HALTED:
                logger.warning(
                    f"Cannot restore from HALTED mode - requires system restart"
                )
                return False

            # In production, would verify authorization
            return self._transition_mode(
                SystemMode.NORMAL,
                reason="System restored to normal operation",
                triggered_by=authorized_by,
            )

    def log_incident(
        self,
        severity: IncidentSeverity,
        category: str,
        description: str,
        source_agent: Optional[str] = None,
        request_id: Optional[str] = None,
        triggered_by: str = "",
        response_action: str = "",
        details: Optional[Dict[str, Any]] = None,
    ) -> SecurityIncident:
        """
        Log a security incident.

        Args:
            severity: Incident severity
            category: Category of incident
            description: Description of what happened
            source_agent: Agent involved
            request_id: Associated request
            triggered_by: What triggered the incident
            response_action: Action taken in response
            details: Additional details

        Returns:
            Created SecurityIncident
        """
        self._incident_counter += 1
        incident_id = f"INC-{datetime.now().strftime('%Y%m%d')}-{self._incident_counter:04d}"

        incident = SecurityIncident(
            incident_id=incident_id,
            severity=severity,
            category=category,
            description=description,
            source_agent=source_agent,
            request_id=request_id,
            triggered_by=triggered_by,
            response_action=response_action,
            details=details or {},
        )

        self._incidents.append(incident)

        # Log to file if configured
        if self.incident_log_path:
            self._write_incident(incident)

        # Notify callback if configured
        if self.notify_callback:
            try:
                self.notify_callback(incident)
            except Exception as e:
                logger.error(f"Incident notification callback failed: {e}")

        # Check for automatic mode escalation
        if self.auto_escalate:
            self._check_escalation(incident)

        logger.log(
            self._severity_to_log_level(severity),
            f"Security incident {incident_id}: [{severity.name}] {description}",
        )

        return incident

    def register_mode_callback(
        self,
        callback: Callable[[SystemMode, SystemMode], None],
    ) -> None:
        """Register a callback for mode changes."""
        self._mode_callbacks.append(callback)

    def get_incident_history(
        self,
        severity_filter: Optional[IncidentSeverity] = None,
        since: Optional[datetime] = None,
        limit: int = 100,
    ) -> List[SecurityIncident]:
        """
        Get incident history with optional filters.

        Args:
            severity_filter: Filter by minimum severity
            since: Filter by timestamp
            limit: Maximum number of incidents to return

        Returns:
            List of matching incidents
        """
        incidents = self._incidents.copy()

        if severity_filter:
            incidents = [i for i in incidents if i.severity.value >= severity_filter.value]

        if since:
            incidents = [i for i in incidents if i.timestamp >= since]

        return incidents[-limit:]

    def get_mode_history(self, limit: int = 50) -> List[ModeTransition]:
        """Get history of mode transitions."""
        return self._mode_history[-limit:]

    def get_status(self) -> Dict[str, Any]:
        """Get current emergency control status."""
        recent_incidents = [
            i for i in self._incidents
            if (datetime.now() - i.timestamp).total_seconds() < 3600
        ]

        return {
            "current_mode": self._current_mode.name,
            "is_operational": self.is_operational,
            "is_restricted": self.is_restricted,
            "total_incidents": len(self._incidents),
            "recent_incidents_1h": len(recent_incidents),
            "mode_transitions": len(self._mode_history),
            "auto_escalate_enabled": self.auto_escalate,
        }

    def _transition_mode(
        self,
        new_mode: SystemMode,
        reason: str,
        triggered_by: str,
        incident_id: Optional[str] = None,
    ) -> bool:
        """
        Internal mode transition handler.

        Returns:
            True if transition successful
        """
        with self._mode_lock:
            old_mode = self._current_mode

            # Validate transition
            if not self._is_valid_transition(old_mode, new_mode):
                logger.warning(
                    f"Invalid mode transition: {old_mode.name} -> {new_mode.name}"
                )
                return False

            # Record transition
            transition = ModeTransition(
                from_mode=old_mode,
                to_mode=new_mode,
                reason=reason,
                triggered_by=triggered_by,
                incident_id=incident_id,
            )
            self._mode_history.append(transition)

            # Apply transition
            self._current_mode = new_mode

            logger.warning(
                f"System mode changed: {old_mode.name} -> {new_mode.name} "
                f"(by {triggered_by}): {reason}"
            )

            # Notify callbacks
            for callback in self._mode_callbacks:
                try:
                    callback(old_mode, new_mode)
                except Exception as e:
                    logger.error(f"Mode callback failed: {e}")

            return True

    def _is_valid_transition(
        self,
        from_mode: SystemMode,
        to_mode: SystemMode,
    ) -> bool:
        """Check if mode transition is valid."""
        # Cannot transition from HALTED
        if from_mode == SystemMode.HALTED:
            return False

        # Can always escalate
        if to_mode.value > from_mode.value:
            return True

        # Can only de-escalate from non-halted modes
        if to_mode.value < from_mode.value:
            return from_mode not in (SystemMode.HALTED, SystemMode.LOCKDOWN)

        # Same mode transition is valid (refresh)
        return True

    def _check_escalation(self, incident: SecurityIncident) -> None:
        """Check if incident should trigger mode escalation."""
        severity = incident.severity

        if severity == IncidentSeverity.EMERGENCY:
            self.trigger_lockdown(
                reason=f"Emergency incident: {incident.description}",
                triggered_by="auto_escalation",
                incident_id=incident.incident_id,
            )
            return

        if severity == IncidentSeverity.CRITICAL:
            self.trigger_safe_mode(
                reason=f"Critical incident: {incident.description}",
                triggered_by="auto_escalation",
                incident_id=incident.incident_id,
            )
            return

        # Check threshold-based escalation
        if severity in self.MODE_ESCALATION_THRESHOLDS:
            threshold_count, window_seconds = self.MODE_ESCALATION_THRESHOLDS[severity]

            if window_seconds > 0:
                cutoff = datetime.now()
                recent = [
                    i for i in self._incidents
                    if i.severity == severity
                    and (cutoff - i.timestamp).total_seconds() < window_seconds
                ]

                if len(recent) >= threshold_count:
                    self.trigger_safe_mode(
                        reason=f"{len(recent)} {severity.name} incidents in {window_seconds}s",
                        triggered_by="auto_escalation",
                    )

    def _severity_to_log_level(self, severity: IncidentSeverity) -> int:
        """Map incident severity to logging level."""
        return {
            IncidentSeverity.LOW: logging.INFO,
            IncidentSeverity.MEDIUM: logging.WARNING,
            IncidentSeverity.HIGH: logging.WARNING,
            IncidentSeverity.CRITICAL: logging.ERROR,
            IncidentSeverity.EMERGENCY: logging.CRITICAL,
        }[severity]

    def _init_incident_log(self) -> None:
        """Initialize incident log file."""
        log_path = Path(self.incident_log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        if not log_path.exists():
            with open(log_path, 'w') as f:
                json.dump({"incidents": [], "created": datetime.now().isoformat()}, f)

    def _write_incident(self, incident: SecurityIncident) -> None:
        """Write incident to log file."""
        if not self.incident_log_path:
            return

        try:
            log_path = Path(self.incident_log_path)

            with open(log_path, 'r') as f:
                data = json.load(f)

            data["incidents"].append(incident.to_dict())

            with open(log_path, 'w') as f:
                json.dump(data, f, indent=2)

        except Exception as e:
            logger.error(f"Failed to write incident log: {e}")

    def _write_halt_log(self, reason: str, triggered_by: str) -> None:
        """Write final halt log entry."""
        halt_log = {
            "event": "SYSTEM_HALT",
            "reason": reason,
            "triggered_by": triggered_by,
            "timestamp": datetime.now().isoformat(),
            "final_mode": self._current_mode.name,
            "total_incidents": len(self._incidents),
        }

        if self.incident_log_path:
            halt_path = Path(self.incident_log_path).parent / "halt.log"
            with open(halt_path, 'a') as f:
                f.write(json.dumps(halt_log) + "\n")

        logger.critical(f"System halt logged: {halt_log}")


# Singleton instance for global access
_emergency_controls: Optional[EmergencyControls] = None


def get_emergency_controls() -> EmergencyControls:
    """Get the global emergency controls instance."""
    global _emergency_controls
    if _emergency_controls is None:
        _emergency_controls = EmergencyControls()
    return _emergency_controls


def initialize_emergency_controls(
    incident_log_path: Optional[str] = None,
    auto_escalate: bool = True,
    notify_callback: Optional[Callable[[SecurityIncident], None]] = None,
) -> EmergencyControls:
    """Initialize the global emergency controls instance."""
    global _emergency_controls
    _emergency_controls = EmergencyControls(
        incident_log_path=incident_log_path,
        auto_escalate=auto_escalate,
        notify_callback=notify_callback,
    )
    return _emergency_controls
