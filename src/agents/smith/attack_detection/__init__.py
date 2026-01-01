"""
Agent Smith Attack Detection & Auto-Remediation System

This module provides Agent Smith with the ability to:
1. Watch boundary daemon events and SIEM feeds for attack indicators
2. Detect and classify attacks in real-time
3. Analyze attacks to identify vulnerable code paths
4. Generate patches to immunize against detected attacks
5. Test patches in isolation
6. Submit patches as recommendations for human review

Architecture:
    ┌─────────────────┐     ┌─────────────────┐
    │ Boundary Daemon │────▶│  Attack         │
    │ Event Stream    │     │  Detector       │
    └─────────────────┘     └────────┬────────┘
                                     │
    ┌─────────────────┐              │
    │ SIEM Feed       │──────────────┤
    │ (External)      │              │
    └─────────────────┘              ▼
                            ┌────────────────┐
                            │ Attack         │
                            │ Analyzer       │
                            └────────┬───────┘
                                     │
                                     ▼
                            ┌────────────────┐
                            │ Remediation    │
                            │ Engine         │
                            └────────┬───────┘
                                     │
                                     ▼
                            ┌────────────────┐
                            │ Recommendation │
                            │ System         │
                            └────────────────┘
                                     │
                                     ▼
                            ┌────────────────┐
                            │ Human Review   │
                            │ (PR/Approval)  │
                            └────────────────┘
"""

from .detector import (
    AttackDetector,
    AttackEvent,
    AttackSeverity,
    AttackType,
    create_attack_detector,
)
from .siem_connector import (
    SIEMConnector,
    SIEMConfig,
    SIEMEvent,
    SIEMProvider,
    EventSeverity,
    SIEMAdapter,
    SplunkAdapter,
    ElasticAdapter,
    SentinelAdapter,
    SyslogAdapter,
    MockSIEMAdapter,
    create_siem_connector,
)
from .patterns import (
    AttackPattern,
    PatternLibrary,
    PatternMatch,
    create_pattern_library,
)
from .analyzer import (
    AttackAnalyzer,
    VulnerabilityReport,
    CodeLocation,
    create_attack_analyzer,
)
from .remediation import (
    RemediationEngine,
    Patch,
    PatchStatus,
    RemediationPlan,
    create_remediation_engine,
)
from .recommendation import (
    RecommendationSystem,
    FixRecommendation,
    RecommendationStatus,
    create_recommendation_system,
)
from .integration import (
    connect_boundary_to_smith,
    AttackDetectionPipeline,
    setup_attack_detection_pipeline,
    create_attack_alert_handler,
)

__all__ = [
    # Detector
    "AttackDetector",
    "AttackEvent",
    "AttackSeverity",
    "AttackType",
    "create_attack_detector",
    # SIEM
    "SIEMConnector",
    "SIEMConfig",
    "SIEMEvent",
    "SIEMProvider",
    "EventSeverity",
    "SIEMAdapter",
    "SplunkAdapter",
    "ElasticAdapter",
    "SentinelAdapter",
    "SyslogAdapter",
    "MockSIEMAdapter",
    "create_siem_connector",
    # Patterns
    "AttackPattern",
    "PatternLibrary",
    "PatternMatch",
    "create_pattern_library",
    # Analyzer
    "AttackAnalyzer",
    "VulnerabilityReport",
    "CodeLocation",
    "create_attack_analyzer",
    # Remediation
    "RemediationEngine",
    "Patch",
    "PatchStatus",
    "RemediationPlan",
    "create_remediation_engine",
    # Recommendation
    "RecommendationSystem",
    "FixRecommendation",
    "RecommendationStatus",
    "create_recommendation_system",
    # Integration
    "connect_boundary_to_smith",
    "AttackDetectionPipeline",
    "setup_attack_detection_pipeline",
    "create_attack_alert_handler",
]
