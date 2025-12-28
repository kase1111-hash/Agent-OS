"""
Agent OS Agent Configuration System

Provides configuration management for agents including:
- YAML/JSON configuration loading
- Environment variable interpolation
- Configuration validation
- Model endpoint configuration
- Constitutional document binding
"""

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Dict, Any, Set, Union
import logging

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

import json


logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration for an LLM model endpoint."""
    name: str                       # Model name (e.g., "llama3:8b")
    endpoint: str = "http://localhost:11434"  # Ollama endpoint
    context_window: int = 4096      # Max context tokens
    max_output_tokens: int = 2048   # Max output tokens
    temperature: float = 0.7        # Default temperature
    top_p: float = 0.9              # Default top_p
    top_k: int = 40                 # Default top_k
    repeat_penalty: float = 1.1     # Default repeat penalty
    quantization: Optional[str] = None  # Quantization level (e.g., "Q4_K_M")
    timeout_ms: int = 30000         # Request timeout
    system_prompt: Optional[str] = None  # Default system prompt
    extra_params: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelConfig":
        """Create from dictionary.

        Raises:
            ValueError: If required fields are missing or invalid
        """
        if not isinstance(data, dict):
            raise ValueError(f"Model config must be a dictionary, got {type(data).__name__}")

        if 'name' not in data:
            raise ValueError("Model config missing required field 'name'")

        # Extract known fields
        known_fields = {
            'name', 'endpoint', 'context_window', 'max_output_tokens',
            'temperature', 'top_p', 'top_k', 'repeat_penalty',
            'quantization', 'timeout_ms', 'system_prompt'
        }

        kwargs = {k: v for k, v in data.items() if k in known_fields}
        kwargs['extra_params'] = {k: v for k, v in data.items() if k not in known_fields}

        try:
            return cls(**kwargs)
        except TypeError as e:
            raise ValueError(f"Invalid model config parameters: {e}") from e


@dataclass
class ConstitutionBinding:
    """Binding between agent and constitutional documents."""
    supreme_path: Optional[Path] = None   # Path to supreme constitution
    agent_path: Optional[Path] = None     # Path to agent-specific constitution
    role_paths: List[Path] = field(default_factory=list)  # Role constitutions
    task_paths: List[Path] = field(default_factory=list)  # Task constitutions

    def all_paths(self) -> List[Path]:
        """Get all constitution paths."""
        paths = []
        if self.supreme_path:
            paths.append(self.supreme_path)
        if self.agent_path:
            paths.append(self.agent_path)
        paths.extend(self.role_paths)
        paths.extend(self.task_paths)
        return paths


@dataclass
class AgentConfig:
    """Complete configuration for an agent."""
    # Identity
    name: str                           # Agent name
    version: str = "0.1.0"              # Agent version
    description: str = ""               # Agent description

    # Model configuration
    model: Optional[ModelConfig] = None

    # Constitutional binding
    constitution: Optional[ConstitutionBinding] = None

    # Capabilities
    capabilities: Set[str] = field(default_factory=set)
    supported_intents: List[str] = field(default_factory=list)

    # Runtime settings
    requires_memory: bool = False       # Uses memory system
    requires_smith: bool = True         # Requires Smith validation
    can_escalate: bool = True           # Can escalate to human

    # Process isolation
    isolation_level: str = "none"       # none, process, container
    resource_limits: Dict[str, Any] = field(default_factory=dict)

    # Communication
    message_bus_channel: Optional[str] = None  # Custom channel name

    # Logging/metrics
    log_level: str = "INFO"
    metrics_enabled: bool = True

    # Custom settings
    custom: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentConfig":
        """Create configuration from dictionary.

        Raises:
            ValueError: If required fields are missing or invalid
        """
        if not isinstance(data, dict):
            raise ValueError(f"Agent config must be a dictionary, got {type(data).__name__}")

        if 'name' not in data:
            raise ValueError("Agent config missing required field 'name'")

        # Parse model config
        model = None
        if 'model' in data:
            try:
                model = ModelConfig.from_dict(data['model'])
            except ValueError as e:
                raise ValueError(f"Invalid model configuration: {e}") from e

        # Parse constitution binding
        constitution = None
        if 'constitution' in data:
            try:
                const_data = data['constitution']
                if not isinstance(const_data, dict):
                    raise ValueError(f"Constitution config must be a dictionary, got {type(const_data).__name__}")
                constitution = ConstitutionBinding(
                    supreme_path=Path(const_data['supreme']) if const_data.get('supreme') else None,
                    agent_path=Path(const_data['agent']) if const_data.get('agent') else None,
                    role_paths=[Path(p) for p in const_data.get('roles', [])],
                    task_paths=[Path(p) for p in const_data.get('tasks', [])],
                )
            except (TypeError, ValueError) as e:
                raise ValueError(f"Invalid constitution binding: {e}") from e

        try:
            return cls(
                name=data['name'],
                version=data.get('version', '0.1.0'),
                description=data.get('description', ''),
                model=model,
                constitution=constitution,
                capabilities=set(data.get('capabilities', [])),
                supported_intents=data.get('supported_intents', []),
                requires_memory=data.get('requires_memory', False),
                requires_smith=data.get('requires_smith', True),
                can_escalate=data.get('can_escalate', True),
                isolation_level=data.get('isolation_level', 'none'),
                resource_limits=data.get('resource_limits', {}),
                message_bus_channel=data.get('message_bus_channel'),
                log_level=data.get('log_level', 'INFO'),
                metrics_enabled=data.get('metrics_enabled', True),
                custom=data.get('custom', {}),
            )
        except TypeError as e:
            raise ValueError(f"Invalid agent config parameters: {e}") from e

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        result = {
            'name': self.name,
            'version': self.version,
            'description': self.description,
            'capabilities': list(self.capabilities),
            'supported_intents': self.supported_intents,
            'requires_memory': self.requires_memory,
            'requires_smith': self.requires_smith,
            'can_escalate': self.can_escalate,
            'isolation_level': self.isolation_level,
            'resource_limits': self.resource_limits,
            'log_level': self.log_level,
            'metrics_enabled': self.metrics_enabled,
            'custom': self.custom,
        }

        if self.model:
            result['model'] = {
                'name': self.model.name,
                'endpoint': self.model.endpoint,
                'context_window': self.model.context_window,
                'max_output_tokens': self.model.max_output_tokens,
                'temperature': self.model.temperature,
            }

        if self.message_bus_channel:
            result['message_bus_channel'] = self.message_bus_channel

        return result


class ConfigLoader:
    """
    Configuration loader supporting YAML and JSON with env var interpolation.
    """

    # Pattern for environment variable references: ${VAR_NAME} or ${VAR_NAME:-default}
    ENV_PATTERN = re.compile(r'\$\{([^}:]+)(?::-([^}]*))?\}')

    def __init__(self, base_path: Optional[Path] = None):
        """
        Initialize config loader.

        Args:
            base_path: Base path for resolving relative paths
        """
        self.base_path = base_path or Path.cwd()

    def load(self, config_path: Union[str, Path]) -> AgentConfig:
        """
        Load agent configuration from file.

        Args:
            config_path: Path to configuration file (YAML or JSON)

        Returns:
            AgentConfig instance

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file format not supported or invalid
            IOError: If file cannot be read
        """
        path = Path(config_path)
        if not path.is_absolute():
            path = self.base_path / path

        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        try:
            content = path.read_text()
        except PermissionError as e:
            raise IOError(f"Permission denied reading config file {path}: {e}") from e
        except UnicodeDecodeError as e:
            raise ValueError(f"Config file {path} is not valid UTF-8: {e}") from e
        except OSError as e:
            raise IOError(f"Failed to read config file {path}: {e}") from e

        if not content.strip():
            raise ValueError(f"Config file {path} is empty")

        # Interpolate environment variables
        content = self._interpolate_env(content)

        # Parse based on extension
        try:
            if path.suffix in ('.yaml', '.yml'):
                if not YAML_AVAILABLE:
                    raise ValueError("YAML support requires PyYAML package")
                data = yaml.safe_load(content)
            elif path.suffix == '.json':
                data = json.loads(content)
            else:
                raise ValueError(f"Unsupported config format: {path.suffix}")
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in config file {path}: {e}") from e
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in config file {path}: {e}") from e

        if data is None:
            raise ValueError(f"Config file {path} parsed to empty data")

        try:
            return AgentConfig.from_dict(data)
        except ValueError as e:
            raise ValueError(f"Invalid configuration in {path}: {e}") from e

    def load_from_string(self, content: str, format: str = "yaml") -> AgentConfig:
        """
        Load configuration from string.

        Args:
            content: Configuration content
            format: Content format ("yaml" or "json")

        Returns:
            AgentConfig instance

        Raises:
            ValueError: If format is invalid or content cannot be parsed
        """
        if not content or not content.strip():
            raise ValueError("Configuration content is empty")

        content = self._interpolate_env(content)

        try:
            if format == "yaml":
                if not YAML_AVAILABLE:
                    raise ValueError("YAML support requires PyYAML package")
                data = yaml.safe_load(content)
            elif format == "json":
                data = json.loads(content)
            else:
                raise ValueError(f"Unsupported format: {format}. Use 'yaml' or 'json'")
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML configuration: {e}") from e
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON configuration: {e}") from e

        if data is None:
            raise ValueError("Configuration parsed to empty data")

        try:
            return AgentConfig.from_dict(data)
        except ValueError as e:
            raise ValueError(f"Invalid agent configuration: {e}") from e

    def _interpolate_env(self, content: str) -> str:
        """
        Interpolate environment variables in config content.

        Supports:
        - ${VAR_NAME} - value of VAR_NAME env var
        - ${VAR_NAME:-default} - value or default if not set
        """
        def replacer(match):
            var_name = match.group(1)
            default = match.group(2)

            value = os.environ.get(var_name)
            if value is not None:
                return value
            elif default is not None:
                return default
            else:
                logger.warning(f"Environment variable {var_name} not set and no default")
                return ""

        return self.ENV_PATTERN.sub(replacer, content)

    def validate(self, config: AgentConfig) -> List[str]:
        """
        Validate an agent configuration.

        Args:
            config: Configuration to validate

        Returns:
            List of validation errors (empty if valid)
        """
        errors = []

        # Required fields
        if not config.name:
            errors.append("Agent name is required")

        if not config.name.isidentifier():
            errors.append(f"Agent name must be a valid identifier: {config.name}")

        # Model configuration
        if config.model:
            if not config.model.name:
                errors.append("Model name is required when model is specified")

            if config.model.temperature < 0 or config.model.temperature > 2:
                errors.append("Temperature must be between 0 and 2")

            if config.model.top_p < 0 or config.model.top_p > 1:
                errors.append("top_p must be between 0 and 1")

        # Constitution binding
        if config.constitution:
            for path in config.constitution.all_paths():
                if not path.exists():
                    errors.append(f"Constitution file not found: {path}")

        # Isolation level
        valid_isolation = {'none', 'process', 'container'}
        if config.isolation_level not in valid_isolation:
            errors.append(f"Invalid isolation level: {config.isolation_level}")

        return errors


def create_default_config(
    name: str,
    model_name: Optional[str] = None,
    description: str = "",
) -> AgentConfig:
    """
    Create a default agent configuration.

    Args:
        name: Agent name
        model_name: Optional model name
        description: Agent description

    Returns:
        AgentConfig with sensible defaults
    """
    model = None
    if model_name:
        model = ModelConfig(
            name=model_name,
            endpoint=os.environ.get("OLLAMA_ENDPOINT", "http://localhost:11434"),
        )

    return AgentConfig(
        name=name,
        description=description,
        model=model,
    )


# Example configuration template
AGENT_CONFIG_TEMPLATE = """
# Agent OS Agent Configuration
# See documentation for full list of options

name: "{name}"
version: "0.1.0"
description: "{description}"

# Model configuration
model:
  name: "{model_name}"
  endpoint: "${{OLLAMA_ENDPOINT:-http://localhost:11434}}"
  context_window: 4096
  max_output_tokens: 2048
  temperature: 0.7

# Constitutional binding
constitution:
  supreme: "CONSTITUTION.md"
  agent: "agents/{name}/constitution.md"

# Capabilities
capabilities:
  - reasoning
  - generation

# Supported intents
supported_intents:
  - "query.*"

# Runtime settings
requires_memory: false
requires_smith: true
can_escalate: true

# Process isolation
isolation_level: none
resource_limits:
  memory_mb: 512
  cpu_percent: 50

# Logging
log_level: INFO
metrics_enabled: true

# Custom settings
custom:
  example_setting: "value"
"""


def generate_config_template(
    name: str,
    description: str = "",
    model_name: str = "llama3:8b",
) -> str:
    """
    Generate a configuration template for a new agent.

    Args:
        name: Agent name
        description: Agent description
        model_name: Model name

    Returns:
        Configuration template string
    """
    return AGENT_CONFIG_TEMPLATE.format(
        name=name,
        description=description,
        model_name=model_name,
    )
