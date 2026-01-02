"""
Agent Smith Advanced Memory System

A sophisticated security intelligence storage and synthesis system that integrates
with Boundary-SIEM and Boundary-Daemon to provide:

- Tiered event storage with configurable retention policies
- Cross-source threat correlation and pattern detection
- Behavioral baseline learning for anomaly detection
- Intelligence synthesis and trend analysis
- Integration with external security infrastructure

This module gives Agent Smith the ability to:
1. Remember and correlate security events over time
2. Detect patterns that span multiple sources
3. Generate actionable intelligence summaries
4. Learn normal behavior to better detect anomalies
5. Maintain a knowledge base for future reference

References:
- Boundary-Daemon: https://github.com/kase1111-hash/boundary-daemon-
- Boundary-SIEM: https://github.com/kase1111-hash/Boundary-SIEM
"""

from .store import (
    SecurityIntelligenceStore,
    IntelligenceEntry,
    IntelligenceType,
    RetentionTier,
    create_intelligence_store,
)

from .correlator import (
    ThreatCorrelator,
    CorrelationResult,
    CorrelationRule,
    ThreatCluster,
    create_threat_correlator,
)

from .synthesizer import (
    PatternSynthesizer,
    SynthesizedPattern,
    TrendAnalysis,
    IntelligenceSummary,
    create_pattern_synthesizer,
)

from .baseline import (
    BehavioralBaseline,
    BaselineMetrics,
    AnomalyScore,
    create_behavioral_baseline,
)

from .boundary_connector import (
    BoundaryDaemonConnector,
    BoundaryEvent,
    PolicyDecision,
    TripwireAlert,
    create_boundary_connector,
)

from .manager import (
    AdvancedMemoryManager,
    MemoryConfig,
    create_advanced_memory,
)

__all__ = [
    # Store
    "SecurityIntelligenceStore",
    "IntelligenceEntry",
    "IntelligenceType",
    "RetentionTier",
    "create_intelligence_store",
    # Correlator
    "ThreatCorrelator",
    "CorrelationResult",
    "CorrelationRule",
    "ThreatCluster",
    "create_threat_correlator",
    # Synthesizer
    "PatternSynthesizer",
    "SynthesizedPattern",
    "TrendAnalysis",
    "IntelligenceSummary",
    "create_pattern_synthesizer",
    # Baseline
    "BehavioralBaseline",
    "BaselineMetrics",
    "AnomalyScore",
    "create_behavioral_baseline",
    # Boundary Connector
    "BoundaryDaemonConnector",
    "BoundaryEvent",
    "PolicyDecision",
    "TripwireAlert",
    "create_boundary_connector",
    # Manager
    "AdvancedMemoryManager",
    "MemoryConfig",
    "create_advanced_memory",
]
