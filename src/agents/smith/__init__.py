"""
Agent OS Smith (Guardian) Module

Smith is the security validation agent responsible for:
- Pre-execution validation (S1-S5)
- Post-execution monitoring (S6-S8)
- Refusal handling (S9-S12)
- Emergency controls (safe mode, halt)
"""

from .agent import (
    SmithAgent,
    SmithMetrics,
    create_smith,
)
from .emergency import (
    EmergencyControls,
    IncidentSeverity,
    ModeTransition,
    SecurityIncident,
    SystemMode,
    get_emergency_controls,
    initialize_emergency_controls,
)
from .post_monitor import (
    MonitorCheck,
    MonitorResult,
    PostExecutionMonitor,
    PostMonitorResult,
)
from .pre_validator import (
    CheckResult,
    PreExecutionValidator,
    PreValidationResult,
    ValidationCheck,
)
from .refusal_engine import (
    RefusalDecision,
    RefusalEngine,
    RefusalResponse,
    RefusalType,
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
