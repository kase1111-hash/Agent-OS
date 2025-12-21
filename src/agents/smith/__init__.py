"""
Agent OS Smith (Guardian) Module

Smith is the security validation agent responsible for:
- Pre-execution validation (S1-S5)
- Post-execution monitoring (S6-S8)
- Refusal handling (S9-S12)
- Emergency controls (safe mode, halt)
"""

from .pre_validator import (
    PreExecutionValidator,
    PreValidationResult,
    ValidationCheck,
    CheckResult,
)
from .post_monitor import (
    PostExecutionMonitor,
    PostMonitorResult,
    MonitorCheck,
    MonitorResult,
)
from .refusal_engine import (
    RefusalEngine,
    RefusalDecision,
    RefusalResponse,
    RefusalType,
)
from .emergency import (
    EmergencyControls,
    SystemMode,
    IncidentSeverity,
    SecurityIncident,
    ModeTransition,
    get_emergency_controls,
    initialize_emergency_controls,
)
from .agent import (
    SmithAgent,
    SmithMetrics,
    create_smith,
)


__all__ = [
    # Pre-Execution Validator
    "PreExecutionValidator",
    "PreValidationResult",
    "ValidationCheck",
    "CheckResult",
    # Post-Execution Monitor
    "PostExecutionMonitor",
    "PostMonitorResult",
    "MonitorCheck",
    "MonitorResult",
    # Refusal Engine
    "RefusalEngine",
    "RefusalDecision",
    "RefusalResponse",
    "RefusalType",
    # Emergency Controls
    "EmergencyControls",
    "SystemMode",
    "IncidentSeverity",
    "SecurityIncident",
    "ModeTransition",
    "get_emergency_controls",
    "initialize_emergency_controls",
    # Smith Agent
    "SmithAgent",
    "SmithMetrics",
    "create_smith",
]
