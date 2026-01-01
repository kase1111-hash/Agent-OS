"""
YAML-based Configuration System for Attack Detection

This module provides a comprehensive configuration system for the
attack detection pipeline, supporting:

- YAML configuration files
- Environment variable substitution
- Configuration validation
- Default values with overrides
- Multiple configuration sources

Example YAML config:

```yaml
attack_detection:
  enabled: true
  severity_threshold: medium

  detector:
    enable_boundary_events: true
    enable_siem_events: true
    auto_lockdown_on_critical: false

  siem:
    sources:
      - name: splunk
        provider: splunk
        endpoint: ${SPLUNK_URL}
        username: ${SPLUNK_USER}
        password: ${SPLUNK_PASS}
        poll_interval: 30

  notifications:
    channels:
      - name: slack
        type: slack
        webhook_url: ${SLACK_WEBHOOK}
        min_severity: high

  storage:
    backend: sqlite
    path: ./data/attacks.db
```
"""

import logging
import os
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, TypeVar, Union

logger = logging.getLogger(__name__)

# Try to import yaml, provide fallback
try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False
    logger.warning("PyYAML not installed. YAML config loading disabled.")


class ConfigError(Exception):
    """Configuration error."""
    pass


class ConfigValidationError(ConfigError):
    """Configuration validation error."""

    def __init__(self, errors: List[str]):
        self.errors = errors
        super().__init__(f"Configuration validation failed: {'; '.join(errors)}")


class SeverityLevel(Enum):
    """Severity level for configuration."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"
    CATASTROPHIC = "catastrophic"


class StorageBackend(Enum):
    """Storage backend type."""
    MEMORY = "memory"
    SQLITE = "sqlite"


class SIEMProviderType(Enum):
    """SIEM provider type."""
    MOCK = "mock"
    SPLUNK = "splunk"
    ELASTIC = "elastic"
    SENTINEL = "sentinel"
    SYSLOG = "syslog"


class NotificationChannelType(Enum):
    """Notification channel type."""
    CONSOLE = "console"
    EMAIL = "email"
    SLACK = "slack"
    TEAMS = "teams"
    PAGERDUTY = "pagerduty"
    WEBHOOK = "webhook"


@dataclass
class DetectorConfig:
    """Attack detector configuration."""
    enable_boundary_events: bool = True
    enable_siem_events: bool = True
    auto_lockdown_on_critical: bool = False
    lockdown_duration_seconds: int = 3600
    detection_confidence_threshold: float = 0.7
    max_events_per_minute: int = 1000
    pattern_match_timeout_ms: int = 100


@dataclass
class SIEMSourceConfig:
    """Configuration for a SIEM source."""
    name: str
    provider: SIEMProviderType
    enabled: bool = True
    endpoint: Optional[str] = None
    username: Optional[str] = None
    password: Optional[str] = None
    api_key: Optional[str] = None
    poll_interval: int = 30
    batch_size: int = 100
    verify_ssl: bool = True
    timeout: int = 30
    extra_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SIEMConfig:
    """SIEM integration configuration."""
    enabled: bool = True
    sources: List[SIEMSourceConfig] = field(default_factory=list)
    event_deduplication: bool = True
    dedup_window_seconds: int = 300
    max_queue_size: int = 10000


@dataclass
class NotificationChannelConfig:
    """Configuration for a notification channel."""
    name: str
    type: NotificationChannelType
    enabled: bool = True
    min_severity: SeverityLevel = SeverityLevel.MEDIUM
    rate_limit: int = 10
    rate_window_seconds: int = 60

    # Channel-specific settings
    webhook_url: Optional[str] = None
    channel: Optional[str] = None

    # Email settings
    smtp_host: Optional[str] = None
    smtp_port: int = 587
    from_address: Optional[str] = None
    to_addresses: List[str] = field(default_factory=list)
    use_tls: bool = True
    username: Optional[str] = None
    password: Optional[str] = None

    # PagerDuty settings
    routing_key: Optional[str] = None

    # Webhook settings
    headers: Dict[str, str] = field(default_factory=dict)
    payload_template: Optional[Dict[str, Any]] = None


@dataclass
class NotificationsConfig:
    """Notifications configuration."""
    enabled: bool = True
    channels: List[NotificationChannelConfig] = field(default_factory=list)
    aggregate_similar: bool = True
    aggregate_window_seconds: int = 300
    include_console: bool = True  # Console output for dev


@dataclass
class StorageConfig:
    """Storage configuration."""
    backend: StorageBackend = StorageBackend.SQLITE
    path: str = "./data/attack_detection.db"
    auto_migrate: bool = True
    cleanup_enabled: bool = True
    cleanup_older_than_days: int = 90
    keep_unresolved: bool = True


@dataclass
class AnalyzerConfig:
    """Attack analyzer configuration."""
    enable_llm_analysis: bool = True
    llm_timeout_seconds: int = 30
    use_sage_agent: bool = True
    fallback_to_patterns: bool = True
    max_code_context_lines: int = 50
    mitre_mapping_enabled: bool = True


@dataclass
class RemediationConfig:
    """Remediation engine configuration."""
    enabled: bool = True
    auto_generate_patches: bool = True
    require_approval: bool = True
    test_patches_in_sandbox: bool = True
    sandbox_timeout_seconds: int = 60
    max_patches_per_attack: int = 5


@dataclass
class GitIntegrationConfig:
    """Git integration configuration."""
    enabled: bool = False
    auto_create_pr: bool = False
    pr_draft_mode: bool = True
    base_branch: str = "main"
    branch_prefix: str = "security/fix"
    default_reviewers: List[str] = field(default_factory=list)
    labels: List[str] = field(default_factory=lambda: ["security", "auto-remediation"])


@dataclass
class AttackDetectionConfig:
    """Complete attack detection configuration."""
    enabled: bool = True
    severity_threshold: SeverityLevel = SeverityLevel.LOW

    detector: DetectorConfig = field(default_factory=DetectorConfig)
    siem: SIEMConfig = field(default_factory=SIEMConfig)
    notifications: NotificationsConfig = field(default_factory=NotificationsConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    analyzer: AnalyzerConfig = field(default_factory=AnalyzerConfig)
    remediation: RemediationConfig = field(default_factory=RemediationConfig)
    git: GitIntegrationConfig = field(default_factory=GitIntegrationConfig)

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return _config_to_dict(self)


T = TypeVar('T')


def _config_to_dict(obj: Any) -> Any:
    """Recursively convert dataclass to dict."""
    if hasattr(obj, '__dataclass_fields__'):
        result = {}
        for field_name in obj.__dataclass_fields__:
            value = getattr(obj, field_name)
            result[field_name] = _config_to_dict(value)
        return result
    elif isinstance(obj, list):
        return [_config_to_dict(item) for item in obj]
    elif isinstance(obj, dict):
        return {k: _config_to_dict(v) for k, v in obj.items()}
    elif isinstance(obj, Enum):
        return obj.value
    else:
        return obj


class ConfigLoader:
    """
    Configuration loader with YAML support and environment variable substitution.
    """

    ENV_VAR_PATTERN = re.compile(r'\$\{([^}]+)\}')

    def __init__(
        self,
        env_prefix: str = "ATTACK_DETECTION",
        allow_env_override: bool = True,
    ):
        """
        Initialize config loader.

        Args:
            env_prefix: Prefix for environment variable overrides
            allow_env_override: Allow environment variables to override config
        """
        self.env_prefix = env_prefix
        self.allow_env_override = allow_env_override

    def load_yaml(self, path: Union[str, Path]) -> AttackDetectionConfig:
        """
        Load configuration from a YAML file.

        Args:
            path: Path to YAML configuration file

        Returns:
            Parsed AttackDetectionConfig
        """
        if not YAML_AVAILABLE:
            raise ConfigError("PyYAML is required for YAML configuration")

        path = Path(path)
        if not path.exists():
            raise ConfigError(f"Configuration file not found: {path}")

        try:
            with open(path, 'r') as f:
                raw_config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ConfigError(f"Invalid YAML: {e}")

        if not raw_config:
            raw_config = {}

        # Extract attack_detection section if present
        if 'attack_detection' in raw_config:
            raw_config = raw_config['attack_detection']

        # Substitute environment variables
        raw_config = self._substitute_env_vars(raw_config)

        # Parse and validate
        return self._parse_config(raw_config)

    def load_dict(self, config_dict: Dict[str, Any]) -> AttackDetectionConfig:
        """
        Load configuration from a dictionary.

        Args:
            config_dict: Configuration dictionary

        Returns:
            Parsed AttackDetectionConfig
        """
        # Substitute environment variables
        config_dict = self._substitute_env_vars(config_dict)

        return self._parse_config(config_dict)

    def load_from_env(self) -> AttackDetectionConfig:
        """
        Load configuration purely from environment variables.

        Environment variable format: ATTACK_DETECTION_<SECTION>_<KEY>
        Example: ATTACK_DETECTION_DETECTOR_ENABLE_BOUNDARY_EVENTS=true

        Returns:
            AttackDetectionConfig from environment
        """
        config_dict: Dict[str, Any] = {}

        for key, value in os.environ.items():
            if key.startswith(f"{self.env_prefix}_"):
                # Parse key path
                path_parts = key[len(self.env_prefix) + 1:].lower().split('_')
                self._set_nested(config_dict, path_parts, self._parse_value(value))

        return self._parse_config(config_dict)

    def merge_configs(
        self,
        *configs: AttackDetectionConfig,
    ) -> AttackDetectionConfig:
        """
        Merge multiple configurations, later ones override earlier.

        Args:
            configs: Configuration objects to merge

        Returns:
            Merged configuration
        """
        merged_dict: Dict[str, Any] = {}

        for config in configs:
            config_dict = config.to_dict()
            merged_dict = self._deep_merge(merged_dict, config_dict)

        return self._parse_config(merged_dict)

    def _substitute_env_vars(self, obj: Any) -> Any:
        """Recursively substitute environment variables."""
        if isinstance(obj, str):
            def replace_var(match: re.Match) -> str:
                var_name = match.group(1)
                default = None

                # Support ${VAR:-default} syntax
                if ':-' in var_name:
                    var_name, default = var_name.split(':-', 1)

                value = os.environ.get(var_name)
                if value is None:
                    if default is not None:
                        return default
                    logger.warning(f"Environment variable not set: {var_name}")
                    return match.group(0)  # Keep original if not found
                return value

            return self.ENV_VAR_PATTERN.sub(replace_var, obj)

        elif isinstance(obj, dict):
            return {k: self._substitute_env_vars(v) for k, v in obj.items()}

        elif isinstance(obj, list):
            return [self._substitute_env_vars(item) for item in obj]

        return obj

    def _parse_config(self, raw: Dict[str, Any]) -> AttackDetectionConfig:
        """Parse raw config dict into AttackDetectionConfig."""
        errors: List[str] = []

        try:
            config = AttackDetectionConfig(
                enabled=raw.get('enabled', True),
                severity_threshold=self._parse_enum(
                    raw.get('severity_threshold', 'low'),
                    SeverityLevel,
                    'severity_threshold',
                    errors,
                ),
                detector=self._parse_detector(raw.get('detector', {}), errors),
                siem=self._parse_siem(raw.get('siem', {}), errors),
                notifications=self._parse_notifications(raw.get('notifications', {}), errors),
                storage=self._parse_storage(raw.get('storage', {}), errors),
                analyzer=self._parse_analyzer(raw.get('analyzer', {}), errors),
                remediation=self._parse_remediation(raw.get('remediation', {}), errors),
                git=self._parse_git(raw.get('git', {}), errors),
            )
        except Exception as e:
            errors.append(f"Configuration parsing error: {e}")
            raise ConfigValidationError(errors)

        if errors:
            raise ConfigValidationError(errors)

        return config

    def _parse_detector(self, raw: Dict[str, Any], errors: List[str]) -> DetectorConfig:
        """Parse detector configuration."""
        return DetectorConfig(
            enable_boundary_events=raw.get('enable_boundary_events', True),
            enable_siem_events=raw.get('enable_siem_events', True),
            auto_lockdown_on_critical=raw.get('auto_lockdown_on_critical', False),
            lockdown_duration_seconds=raw.get('lockdown_duration_seconds', 3600),
            detection_confidence_threshold=raw.get('detection_confidence_threshold', 0.7),
            max_events_per_minute=raw.get('max_events_per_minute', 1000),
            pattern_match_timeout_ms=raw.get('pattern_match_timeout_ms', 100),
        )

    def _parse_siem(self, raw: Dict[str, Any], errors: List[str]) -> SIEMConfig:
        """Parse SIEM configuration."""
        sources = []
        for source_raw in raw.get('sources', []):
            try:
                source = SIEMSourceConfig(
                    name=source_raw.get('name', 'unnamed'),
                    provider=self._parse_enum(
                        source_raw.get('provider', 'mock'),
                        SIEMProviderType,
                        f"siem.sources[{source_raw.get('name')}].provider",
                        errors,
                    ),
                    enabled=source_raw.get('enabled', True),
                    endpoint=source_raw.get('endpoint'),
                    username=source_raw.get('username'),
                    password=source_raw.get('password'),
                    api_key=source_raw.get('api_key'),
                    poll_interval=source_raw.get('poll_interval', 30),
                    batch_size=source_raw.get('batch_size', 100),
                    verify_ssl=source_raw.get('verify_ssl', True),
                    timeout=source_raw.get('timeout', 30),
                    extra_params=source_raw.get('extra_params', {}),
                )
                sources.append(source)
            except Exception as e:
                errors.append(f"Invalid SIEM source config: {e}")

        return SIEMConfig(
            enabled=raw.get('enabled', True),
            sources=sources,
            event_deduplication=raw.get('event_deduplication', True),
            dedup_window_seconds=raw.get('dedup_window_seconds', 300),
            max_queue_size=raw.get('max_queue_size', 10000),
        )

    def _parse_notifications(self, raw: Dict[str, Any], errors: List[str]) -> NotificationsConfig:
        """Parse notifications configuration."""
        channels = []
        for channel_raw in raw.get('channels', []):
            try:
                channel = NotificationChannelConfig(
                    name=channel_raw.get('name', 'unnamed'),
                    type=self._parse_enum(
                        channel_raw.get('type', 'console'),
                        NotificationChannelType,
                        f"notifications.channels[{channel_raw.get('name')}].type",
                        errors,
                    ),
                    enabled=channel_raw.get('enabled', True),
                    min_severity=self._parse_enum(
                        channel_raw.get('min_severity', 'medium'),
                        SeverityLevel,
                        f"notifications.channels[{channel_raw.get('name')}].min_severity",
                        errors,
                    ),
                    rate_limit=channel_raw.get('rate_limit', 10),
                    rate_window_seconds=channel_raw.get('rate_window_seconds', 60),
                    webhook_url=channel_raw.get('webhook_url'),
                    channel=channel_raw.get('channel'),
                    smtp_host=channel_raw.get('smtp_host'),
                    smtp_port=channel_raw.get('smtp_port', 587),
                    from_address=channel_raw.get('from_address'),
                    to_addresses=channel_raw.get('to_addresses', []),
                    use_tls=channel_raw.get('use_tls', True),
                    username=channel_raw.get('username'),
                    password=channel_raw.get('password'),
                    routing_key=channel_raw.get('routing_key'),
                    headers=channel_raw.get('headers', {}),
                    payload_template=channel_raw.get('payload_template'),
                )
                channels.append(channel)
            except Exception as e:
                errors.append(f"Invalid notification channel config: {e}")

        return NotificationsConfig(
            enabled=raw.get('enabled', True),
            channels=channels,
            aggregate_similar=raw.get('aggregate_similar', True),
            aggregate_window_seconds=raw.get('aggregate_window_seconds', 300),
            include_console=raw.get('include_console', True),
        )

    def _parse_storage(self, raw: Dict[str, Any], errors: List[str]) -> StorageConfig:
        """Parse storage configuration."""
        return StorageConfig(
            backend=self._parse_enum(
                raw.get('backend', 'sqlite'),
                StorageBackend,
                'storage.backend',
                errors,
            ),
            path=raw.get('path', './data/attack_detection.db'),
            auto_migrate=raw.get('auto_migrate', True),
            cleanup_enabled=raw.get('cleanup_enabled', True),
            cleanup_older_than_days=raw.get('cleanup_older_than_days', 90),
            keep_unresolved=raw.get('keep_unresolved', True),
        )

    def _parse_analyzer(self, raw: Dict[str, Any], errors: List[str]) -> AnalyzerConfig:
        """Parse analyzer configuration."""
        return AnalyzerConfig(
            enable_llm_analysis=raw.get('enable_llm_analysis', True),
            llm_timeout_seconds=raw.get('llm_timeout_seconds', 30),
            use_sage_agent=raw.get('use_sage_agent', True),
            fallback_to_patterns=raw.get('fallback_to_patterns', True),
            max_code_context_lines=raw.get('max_code_context_lines', 50),
            mitre_mapping_enabled=raw.get('mitre_mapping_enabled', True),
        )

    def _parse_remediation(self, raw: Dict[str, Any], errors: List[str]) -> RemediationConfig:
        """Parse remediation configuration."""
        return RemediationConfig(
            enabled=raw.get('enabled', True),
            auto_generate_patches=raw.get('auto_generate_patches', True),
            require_approval=raw.get('require_approval', True),
            test_patches_in_sandbox=raw.get('test_patches_in_sandbox', True),
            sandbox_timeout_seconds=raw.get('sandbox_timeout_seconds', 60),
            max_patches_per_attack=raw.get('max_patches_per_attack', 5),
        )

    def _parse_git(self, raw: Dict[str, Any], errors: List[str]) -> GitIntegrationConfig:
        """Parse git integration configuration."""
        return GitIntegrationConfig(
            enabled=raw.get('enabled', False),
            auto_create_pr=raw.get('auto_create_pr', False),
            pr_draft_mode=raw.get('pr_draft_mode', True),
            base_branch=raw.get('base_branch', 'main'),
            branch_prefix=raw.get('branch_prefix', 'security/fix'),
            default_reviewers=raw.get('default_reviewers', []),
            labels=raw.get('labels', ['security', 'auto-remediation']),
        )

    def _parse_enum(
        self,
        value: Any,
        enum_class: Type[Enum],
        field_path: str,
        errors: List[str],
    ) -> Enum:
        """Parse a value into an enum."""
        if isinstance(value, enum_class):
            return value

        try:
            # Try by value
            return enum_class(value)
        except ValueError:
            pass

        try:
            # Try by name (case insensitive)
            value_upper = str(value).upper()
            for member in enum_class:
                if member.name == value_upper or member.value == value:
                    return member
        except Exception:
            pass

        # Default to first value if not found
        default = list(enum_class)[0]
        errors.append(
            f"Invalid value '{value}' for {field_path}, "
            f"valid values: {[e.value for e in enum_class]}. Using default: {default.value}"
        )
        return default

    def _parse_value(self, value: str) -> Any:
        """Parse a string value to appropriate type."""
        # Boolean
        if value.lower() in ('true', 'yes', '1'):
            return True
        if value.lower() in ('false', 'no', '0'):
            return False

        # Integer
        try:
            return int(value)
        except ValueError:
            pass

        # Float
        try:
            return float(value)
        except ValueError:
            pass

        # List (comma-separated)
        if ',' in value:
            return [v.strip() for v in value.split(',')]

        return value

    def _set_nested(self, obj: Dict, path: List[str], value: Any) -> None:
        """Set a nested value in a dict."""
        for key in path[:-1]:
            obj = obj.setdefault(key, {})
        obj[path[-1]] = value

    def _deep_merge(self, base: Dict, override: Dict) -> Dict:
        """Deep merge two dictionaries."""
        result = base.copy()

        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value

        return result


def generate_default_config() -> str:
    """
    Generate default configuration as YAML string.

    Returns:
        YAML string with default configuration
    """
    config = AttackDetectionConfig()
    config_dict = config.to_dict()

    if YAML_AVAILABLE:
        return yaml.dump(
            {'attack_detection': config_dict},
            default_flow_style=False,
            sort_keys=False,
        )
    else:
        # Simple YAML-like output without PyYAML
        return _dict_to_yaml_string({'attack_detection': config_dict})


def _dict_to_yaml_string(obj: Any, indent: int = 0) -> str:
    """Simple dict to YAML string converter."""
    lines = []
    prefix = "  " * indent

    if isinstance(obj, dict):
        for key, value in obj.items():
            if isinstance(value, (dict, list)) and value:
                lines.append(f"{prefix}{key}:")
                lines.append(_dict_to_yaml_string(value, indent + 1))
            else:
                lines.append(f"{prefix}{key}: {_format_yaml_value(value)}")
    elif isinstance(obj, list):
        for item in obj:
            if isinstance(item, dict):
                lines.append(f"{prefix}-")
                for key, value in item.items():
                    lines.append(f"{prefix}  {key}: {_format_yaml_value(value)}")
            else:
                lines.append(f"{prefix}- {_format_yaml_value(item)}")
    else:
        lines.append(f"{prefix}{_format_yaml_value(obj)}")

    return "\n".join(lines)


def _format_yaml_value(value: Any) -> str:
    """Format a value for YAML output."""
    if value is None:
        return "null"
    elif isinstance(value, bool):
        return "true" if value else "false"
    elif isinstance(value, str):
        if any(c in value for c in ':{}[],"\''):
            return f'"{value}"'
        return value
    elif isinstance(value, list):
        if not value:
            return "[]"
        return str(value)
    elif isinstance(value, dict):
        if not value:
            return "{}"
        return str(value)
    return str(value)


def generate_example_config() -> str:
    """
    Generate example configuration with comments.

    Returns:
        YAML string with example configuration and comments
    """
    return '''# Agent Smith Attack Detection Configuration
# This file configures the attack detection and auto-remediation system.

attack_detection:
  # Enable/disable attack detection
  enabled: true

  # Minimum severity to process (low, medium, high, critical, catastrophic)
  severity_threshold: low

  # Attack detector settings
  detector:
    # Process events from boundary daemon
    enable_boundary_events: true

    # Process events from SIEM sources
    enable_siem_events: true

    # Automatically trigger lockdown on critical attacks
    auto_lockdown_on_critical: false

    # Duration of lockdown in seconds
    lockdown_duration_seconds: 3600

    # Minimum confidence for detection (0.0 - 1.0)
    detection_confidence_threshold: 0.7

  # SIEM integration settings
  siem:
    enabled: true

    # Event deduplication
    event_deduplication: true
    dedup_window_seconds: 300

    # SIEM sources
    sources:
      # Example Splunk configuration
      - name: splunk-prod
        provider: splunk
        enabled: true
        endpoint: ${SPLUNK_URL:-https://splunk.example.com:8089}
        username: ${SPLUNK_USER}
        password: ${SPLUNK_PASS}
        poll_interval: 30
        extra_params:
          index: security

      # Example Elasticsearch configuration
      - name: elastic-siem
        provider: elastic
        enabled: false
        endpoint: ${ELASTIC_URL:-https://elasticsearch.example.com:9200}
        api_key: ${ELASTIC_API_KEY}
        extra_params:
          index: "security-*"

  # Notification settings
  notifications:
    enabled: true

    # Aggregate similar alerts
    aggregate_similar: true
    aggregate_window_seconds: 300

    # Include console output (for development)
    include_console: true

    channels:
      # Slack notifications
      - name: slack-security
        type: slack
        enabled: true
        webhook_url: ${SLACK_WEBHOOK_URL}
        min_severity: high
        rate_limit: 10
        rate_window_seconds: 60

      # Email notifications for critical alerts
      - name: email-oncall
        type: email
        enabled: true
        min_severity: critical
        smtp_host: ${SMTP_HOST:-smtp.example.com}
        smtp_port: 587
        use_tls: true
        username: ${SMTP_USER}
        password: ${SMTP_PASS}
        from_address: security@example.com
        to_addresses:
          - oncall@example.com
          - security-team@example.com

      # PagerDuty for critical incidents
      - name: pagerduty-critical
        type: pagerduty
        enabled: false
        min_severity: critical
        routing_key: ${PAGERDUTY_ROUTING_KEY}

  # Persistent storage settings
  storage:
    # Backend type: memory or sqlite
    backend: sqlite

    # Database path (for sqlite)
    path: ./data/attack_detection.db

    # Auto-run migrations
    auto_migrate: true

    # Cleanup settings
    cleanup_enabled: true
    cleanup_older_than_days: 90
    keep_unresolved: true

  # Attack analyzer settings
  analyzer:
    # Use LLM for deep analysis
    enable_llm_analysis: true
    llm_timeout_seconds: 30

    # Use Sage agent for reasoning
    use_sage_agent: true

    # Fall back to pattern matching if LLM unavailable
    fallback_to_patterns: true

    # MITRE ATT&CK mapping
    mitre_mapping_enabled: true

  # Remediation settings
  remediation:
    enabled: true

    # Automatically generate patches for vulnerabilities
    auto_generate_patches: true

    # Require human approval before applying
    require_approval: true

    # Test patches in sandbox before recommending
    test_patches_in_sandbox: true
    sandbox_timeout_seconds: 60

  # Git integration for auto-PR creation
  git:
    enabled: false
    auto_create_pr: false
    pr_draft_mode: true
    base_branch: main
    branch_prefix: security/fix
    default_reviewers:
      - security-team
    labels:
      - security
      - auto-remediation
      - agent-smith
'''


# Factory functions

def create_config_loader(
    env_prefix: str = "ATTACK_DETECTION",
    allow_env_override: bool = True,
) -> ConfigLoader:
    """Create a configuration loader."""
    return ConfigLoader(
        env_prefix=env_prefix,
        allow_env_override=allow_env_override,
    )


def load_config(
    path: Optional[Union[str, Path]] = None,
    config_dict: Optional[Dict[str, Any]] = None,
    use_env: bool = True,
) -> AttackDetectionConfig:
    """
    Load configuration from file, dict, and/or environment.

    Args:
        path: Path to YAML config file
        config_dict: Configuration dictionary to merge
        use_env: Include environment variable configuration

    Returns:
        Merged AttackDetectionConfig
    """
    loader = create_config_loader()
    configs: List[AttackDetectionConfig] = []

    # Start with defaults
    configs.append(AttackDetectionConfig())

    # Load from file if provided
    if path:
        configs.append(loader.load_yaml(path))

    # Load from dict if provided
    if config_dict:
        configs.append(loader.load_dict(config_dict))

    # Load from environment
    if use_env:
        try:
            env_config = loader.load_from_env()
            # Only merge if env had any settings
            if env_config.to_dict() != AttackDetectionConfig().to_dict():
                configs.append(env_config)
        except Exception as e:
            logger.debug(f"No environment config: {e}")

    # Merge all configs
    return loader.merge_configs(*configs)


def get_default_config() -> AttackDetectionConfig:
    """Get default configuration."""
    return AttackDetectionConfig()


def save_config(
    config: AttackDetectionConfig,
    path: Union[str, Path],
) -> None:
    """
    Save configuration to a YAML file.

    Args:
        config: Configuration to save
        path: Output path
    """
    if not YAML_AVAILABLE:
        raise ConfigError("PyYAML is required to save configuration")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    config_dict = {'attack_detection': config.to_dict()}

    with open(path, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)

    logger.info(f"Configuration saved to {path}")
