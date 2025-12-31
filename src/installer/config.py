"""Installation configuration module.

Provides configuration management for Agent OS installation including:
- Installation paths and locations
- Component selection
- Installation modes
"""

import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from .platform import Platform, detect_platform

logger = logging.getLogger(__name__)


class InstallMode(str, Enum):
    """Installation mode."""

    FULL = "full"  # All components
    MINIMAL = "minimal"  # Core only
    CUSTOM = "custom"  # User-selected components
    DOCKER = "docker"  # Docker-based installation
    PORTABLE = "portable"  # Portable/no-install mode


class InstallLocation(str, Enum):
    """Predefined installation locations."""

    SYSTEM = "system"  # System-wide (/opt, Program Files)
    USER = "user"  # User directory (~/.agent-os)
    CUSTOM = "custom"  # Custom path


@dataclass
class ComponentSelection:
    """Selection of components to install."""

    core: bool = True  # Core Agent OS (required)
    agents: bool = True  # Default agents (Whisper, Smith, etc.)
    memory: bool = True  # Memory/RAG system (Seshat)
    web_ui: bool = True  # Web interface
    voice: bool = False  # Voice interaction
    multimodal: bool = False  # Multi-modal support
    federation: bool = False  # Federation protocol
    sdk: bool = False  # Developer SDK
    examples: bool = True  # Example configurations

    def to_list(self) -> List[str]:
        """Get list of selected components."""
        components = []
        if self.core:
            components.append("core")
        if self.agents:
            components.append("agents")
        if self.memory:
            components.append("memory")
        if self.web_ui:
            components.append("web_ui")
        if self.voice:
            components.append("voice")
        if self.multimodal:
            components.append("multimodal")
        if self.federation:
            components.append("federation")
        if self.sdk:
            components.append("sdk")
        if self.examples:
            components.append("examples")
        return components

    @classmethod
    def from_list(cls, components: List[str]) -> "ComponentSelection":
        """Create from list of component names."""
        return cls(
            core="core" in components,
            agents="agents" in components,
            memory="memory" in components,
            web_ui="web_ui" in components,
            voice="voice" in components,
            multimodal="multimodal" in components,
            federation="federation" in components,
            sdk="sdk" in components,
            examples="examples" in components,
        )

    @classmethod
    def full(cls) -> "ComponentSelection":
        """Create full installation selection."""
        return cls(
            core=True,
            agents=True,
            memory=True,
            web_ui=True,
            voice=True,
            multimodal=True,
            federation=True,
            sdk=True,
            examples=True,
        )

    @classmethod
    def minimal(cls) -> "ComponentSelection":
        """Create minimal installation selection."""
        return cls(
            core=True,
            agents=True,
            memory=False,
            web_ui=False,
            voice=False,
            multimodal=False,
            federation=False,
            sdk=False,
            examples=False,
        )


@dataclass
class InstallConfig:
    """Complete installation configuration."""

    # Installation target
    install_path: Path
    install_location: InstallLocation
    install_mode: InstallMode

    # Components
    components: ComponentSelection

    # Options
    create_shortcuts: bool = True
    add_to_path: bool = True
    auto_start: bool = False
    install_ollama: bool = True
    install_models: List[str] = field(default_factory=lambda: ["mistral:7b-instruct"])

    # Docker options (when mode is DOCKER)
    docker_image: str = "agentoshq/agent-os:latest"
    docker_network: str = "agent-os-network"
    docker_volumes: List[str] = field(default_factory=list)
    docker_ports: Dict[str, int] = field(default_factory=lambda: {"web": 8080, "api": 8081})

    # Advanced options
    data_path: Optional[Path] = None  # Separate data directory
    log_path: Optional[Path] = None  # Log directory
    config_path: Optional[Path] = None  # Config directory

    # Metadata
    version: str = "latest"
    source: str = "release"  # release, git, local

    def __post_init__(self):
        """Post-initialization processing."""
        # Ensure paths are Path objects
        if isinstance(self.install_path, str):
            self.install_path = Path(self.install_path)
        if isinstance(self.data_path, str):
            self.data_path = Path(self.data_path)
        if isinstance(self.log_path, str):
            self.log_path = Path(self.log_path)
        if isinstance(self.config_path, str):
            self.config_path = Path(self.config_path)

        # Set default data/log/config paths if not specified
        if self.data_path is None:
            self.data_path = self.install_path / "data"
        if self.log_path is None:
            self.log_path = self.install_path / "logs"
        if self.config_path is None:
            self.config_path = self.install_path / "config"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "install_path": str(self.install_path),
            "install_location": self.install_location.value,
            "install_mode": self.install_mode.value,
            "components": self.components.to_list(),
            "create_shortcuts": self.create_shortcuts,
            "add_to_path": self.add_to_path,
            "auto_start": self.auto_start,
            "install_ollama": self.install_ollama,
            "install_models": self.install_models,
            "docker_image": self.docker_image,
            "docker_network": self.docker_network,
            "docker_volumes": self.docker_volumes,
            "docker_ports": self.docker_ports,
            "data_path": str(self.data_path) if self.data_path else None,
            "log_path": str(self.log_path) if self.log_path else None,
            "config_path": str(self.config_path) if self.config_path else None,
            "version": self.version,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "InstallConfig":
        """Create from dictionary."""
        return cls(
            install_path=Path(data["install_path"]),
            install_location=InstallLocation(data["install_location"]),
            install_mode=InstallMode(data["install_mode"]),
            components=ComponentSelection.from_list(data.get("components", [])),
            create_shortcuts=data.get("create_shortcuts", True),
            add_to_path=data.get("add_to_path", True),
            auto_start=data.get("auto_start", False),
            install_ollama=data.get("install_ollama", True),
            install_models=data.get("install_models", ["mistral:7b-instruct"]),
            docker_image=data.get("docker_image", "agentoshq/agent-os:latest"),
            docker_network=data.get("docker_network", "agent-os-network"),
            docker_volumes=data.get("docker_volumes", []),
            docker_ports=data.get("docker_ports", {"web": 8080, "api": 8081}),
            data_path=Path(data["data_path"]) if data.get("data_path") else None,
            log_path=Path(data["log_path"]) if data.get("log_path") else None,
            config_path=Path(data["config_path"]) if data.get("config_path") else None,
            version=data.get("version", "latest"),
            source=data.get("source", "release"),
        )

    def save(self, path: Path) -> None:
        """Save configuration to file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "InstallConfig":
        """Load configuration from file."""
        with open(path) as f:
            data = json.load(f)
        return cls.from_dict(data)


def get_default_install_path(
    location: InstallLocation = InstallLocation.USER,
    platform: Optional[Platform] = None,
) -> Path:
    """Get the default installation path for a platform.

    Args:
        location: Installation location type
        platform: Target platform (auto-detected if None)

    Returns:
        Default installation path
    """
    if platform is None:
        platform = detect_platform()

    if location == InstallLocation.USER:
        if platform == Platform.WINDOWS:
            return Path.home() / "AppData" / "Local" / "AgentOS"
        elif platform == Platform.MACOS:
            return Path.home() / ".agent-os"
        else:  # Linux and others
            return Path.home() / ".agent-os"

    elif location == InstallLocation.SYSTEM:
        if platform == Platform.WINDOWS:
            return Path("C:/Program Files/AgentOS")
        elif platform == Platform.MACOS:
            return Path("/usr/local/agent-os")
        else:  # Linux
            return Path("/opt/agent-os")

    else:  # CUSTOM - return user home as default
        return Path.home() / "agent-os"


def create_install_config(
    mode: InstallMode = InstallMode.FULL,
    location: InstallLocation = InstallLocation.USER,
    custom_path: Optional[Path] = None,
    **kwargs,
) -> InstallConfig:
    """Create an installation configuration.

    Args:
        mode: Installation mode
        location: Installation location type
        custom_path: Custom installation path (when location is CUSTOM)
        **kwargs: Additional configuration options

    Returns:
        InstallConfig instance
    """
    # Determine install path
    if location == InstallLocation.CUSTOM and custom_path:
        install_path = custom_path
    else:
        install_path = get_default_install_path(location)

    # Determine components based on mode
    if mode == InstallMode.FULL:
        components = ComponentSelection.full()
    elif mode == InstallMode.MINIMAL:
        components = ComponentSelection.minimal()
    else:
        components = ComponentSelection()

    return InstallConfig(
        install_path=install_path,
        install_location=location,
        install_mode=mode,
        components=components,
        **kwargs,
    )
