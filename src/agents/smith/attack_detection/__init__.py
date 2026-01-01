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
from .storage import (
    AttackStorage,
    SQLiteStorage,
    MemoryStorage,
    StorageBackend,
    StoredAttack,
    StoredRecommendation,
    StoredPatch,
    StoredVulnerability,
    StoredSIEMEvent,
    create_storage,
    create_sqlite_storage,
    create_memory_storage,
)
from .storage_integration import (
    StorageIntegration,
    create_storage_integration,
)
from .llm_analyzer import (
    LLMAnalyzer,
    LLMAnalysisResult,
    AttackIntent,
    MITRETactic,
    ImpactAssessment,
    CodeVulnerability,
    AnalysisConfidence,
    create_llm_analyzer,
)
from .git_integration import (
    GitIntegration,
    GitProvider,
    LocalGitProvider,
    MockGitProvider,
    PatchApplicator,
    BranchInfo,
    CommitInfo,
    PullRequestInfo,
    PatchApplication,
    PRStatus,
    GitOperationError,
    PRCreationError,
    create_git_integration,
    create_local_git_provider,
    create_mock_git_provider,
    create_patch_applicator,
)
from .notifications import (
    NotificationManager,
    NotificationConfig,
    NotificationChannel,
    NotificationPriority,
    NotificationSender,
    NotificationRecord,
    SecurityAlert,
    DeliveryStatus,
    ConsoleSender,
    EmailSender,
    WebhookSender,
    SlackSender,
    PagerDutySender,
    TeamsSender,
    create_notification_manager,
    create_console_channel,
    create_slack_channel,
    create_email_channel,
    create_pagerduty_channel,
    create_webhook_channel,
    create_alert_from_attack,
)
from .config import (
    AttackDetectionConfig,
    ConfigLoader,
    ConfigError,
    ConfigValidationError,
    DetectorConfig,
    SIEMConfig,
    SIEMSourceConfig,
    NotificationsConfig,
    NotificationChannelConfig,
    StorageConfig,
    AnalyzerConfig,
    RemediationConfig,
    GitIntegrationConfig,
    SeverityLevel,
    StorageBackend,
    SIEMProviderType,
    NotificationChannelType,
    create_config_loader,
    load_config,
    get_default_config,
    save_config,
    generate_default_config,
    generate_example_config,
)
from .integration import (
    setup_pipeline_from_config,
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
    # Git Integration
    "GitIntegration",
    "GitProvider",
    "LocalGitProvider",
    "MockGitProvider",
    "PatchApplicator",
    "BranchInfo",
    "CommitInfo",
    "PullRequestInfo",
    "PatchApplication",
    "PRStatus",
    "GitOperationError",
    "PRCreationError",
    "create_git_integration",
    "create_local_git_provider",
    "create_mock_git_provider",
    "create_patch_applicator",
    # Notifications
    "NotificationManager",
    "NotificationConfig",
    "NotificationChannel",
    "NotificationPriority",
    "NotificationSender",
    "NotificationRecord",
    "SecurityAlert",
    "DeliveryStatus",
    "ConsoleSender",
    "EmailSender",
    "WebhookSender",
    "SlackSender",
    "PagerDutySender",
    "TeamsSender",
    "create_notification_manager",
    "create_console_channel",
    "create_slack_channel",
    "create_email_channel",
    "create_pagerduty_channel",
    "create_webhook_channel",
    "create_alert_from_attack",
    # Configuration
    "AttackDetectionConfig",
    "ConfigLoader",
    "ConfigError",
    "ConfigValidationError",
    "DetectorConfig",
    "SIEMConfig",
    "SIEMSourceConfig",
    "NotificationsConfig",
    "NotificationChannelConfig",
    "StorageConfig",
    "AnalyzerConfig",
    "RemediationConfig",
    "GitIntegrationConfig",
    "SeverityLevel",
    "StorageBackend",
    "SIEMProviderType",
    "NotificationChannelType",
    "create_config_loader",
    "load_config",
    "get_default_config",
    "save_config",
    "generate_default_config",
    "generate_example_config",
    "setup_pipeline_from_config",
]
