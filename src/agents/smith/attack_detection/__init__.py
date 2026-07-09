"""
Agent Smith Attack Detection & Auto-Remediation System

This module provides Agent Smith with the ability to:
1. Watch boundary daemon events for attack indicators
2. Detect and classify attacks in real-time
3. Analyze attacks to identify vulnerable code paths
4. Generate patches to immunize against detected attacks
5. Submit patches as recommendations for human review
"""

from .analyzer import (
    AttackAnalyzer,
    CodeLocation,
    VulnerabilityReport,
    create_attack_analyzer,
)
from .config import (
    AnalyzerConfig,
    AttackDetectionConfig,
    ConfigError,
    ConfigLoader,
    ConfigValidationError,
    DetectorConfig,
    RemediationConfig,
    SeverityLevel,
    StorageBackend,
    StorageConfig,
    create_config_loader,
    generate_default_config,
    generate_example_config,
    get_default_config,
    load_config,
    save_config,
)
from .detector import (
    AttackDetector,
    AttackEvent,
    AttackSeverity,
    AttackType,
    create_attack_detector,
)
from .integration import (
    AttackDetectionPipeline,
    connect_boundary_to_smith,
    create_attack_alert_handler,
    setup_attack_detection_pipeline,
    setup_pipeline_from_config,
)
from .llm_analyzer import (
    AnalysisConfidence,
    AttackIntent,
    CodeVulnerability,
    ImpactAssessment,
    LLMAnalysisResult,
    LLMAnalyzer,
    MITRETactic,
    create_llm_analyzer,
)
from .patterns import (
    AttackPattern,
    PatternLibrary,
    PatternMatch,
    create_pattern_library,
)
from .recommendation import (
    FixRecommendation,
    RecommendationStatus,
    RecommendationSystem,
    create_recommendation_system,
)
from .remediation import (
    Patch,
    PatchStatus,
    RemediationEngine,
    RemediationPlan,
    create_remediation_engine,
)
from .storage import (
    AttackStorage,
    MemoryStorage,
    SQLiteStorage,
    StorageBackend,
    StoredAttack,
    StoredPatch,
    StoredRecommendation,
    StoredSIEMEvent,
    StoredVulnerability,
    create_memory_storage,
    create_sqlite_storage,
    create_storage,
)
from .storage_integration import (
    StorageIntegration,
    create_storage_integration,
)

__all__ = [
    # Detector
    "AttackDetector",
    "AttackEvent",
    "AttackSeverity",
    "AttackType",
    "create_attack_detector",
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
    "setup_pipeline_from_config",
    # Storage
    "AttackStorage",
    "SQLiteStorage",
    "MemoryStorage",
    "StorageBackend",
    "StoredAttack",
    "StoredRecommendation",
    "StoredPatch",
    "StoredVulnerability",
    "StoredSIEMEvent",
    "create_storage",
    "create_sqlite_storage",
    "create_memory_storage",
    # Storage Integration
    "StorageIntegration",
    "create_storage_integration",
    # LLM Analyzer
    "LLMAnalyzer",
    "LLMAnalysisResult",
    "AttackIntent",
    "MITRETactic",
    "ImpactAssessment",
    "CodeVulnerability",
    "AnalysisConfidence",
    "create_llm_analyzer",
    # Configuration
    "AttackDetectionConfig",
    "ConfigLoader",
    "ConfigError",
    "ConfigValidationError",
    "DetectorConfig",
    "StorageConfig",
    "AnalyzerConfig",
    "RemediationConfig",
    "SeverityLevel",
    "create_config_loader",
    "load_config",
    "get_default_config",
    "save_config",
    "generate_default_config",
    "generate_example_config",
]
