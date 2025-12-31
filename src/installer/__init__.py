"""One-Click Installer Module for Agent OS.

Provides cross-platform installation support for Windows, macOS, and Linux,
with both native and Docker-based installation options.

Components:
- Platform detection and system requirements checking
- Windows installer (MSI, portable)
- macOS installer (DMG, Homebrew)
- Linux installer (deb, rpm, AppImage)
- Docker-based installation
- CLI interface for installation management
"""

from .base import (
    Installer,
    InstallerError,
    InstallerProgress,
    InstallerResult,
    ProgressCallback,
)
from .cli import (
    InstallCLI,
    run_installer,
)
from .config import (
    ComponentSelection,
    InstallConfig,
    InstallLocation,
    InstallMode,
    create_install_config,
)
from .docker import (
    DockerConfig,
    DockerInstaller,
    create_docker_installer,
)
from .linux import (
    LinuxDistro,
    LinuxInstaller,
    create_linux_installer,
)
from .macos import (
    MacOSInstaller,
    create_macos_installer,
)
from .platform import (
    Platform,
    PlatformInfo,
    RequirementCheck,
    RequirementStatus,
    SystemRequirements,
    all_requirements_met,
    check_requirements,
    detect_platform,
    get_system_info,
)
from .windows import (
    WindowsInstaller,
    create_windows_installer,
)

__all__ = [
    # Platform
    "Platform",
    "PlatformInfo",
    "SystemRequirements",
    "RequirementCheck",
    "RequirementStatus",
    "detect_platform",
    "check_requirements",
    "get_system_info",
    "all_requirements_met",
    # Config
    "InstallConfig",
    "InstallLocation",
    "InstallMode",
    "ComponentSelection",
    "create_install_config",
    # Base
    "Installer",
    "InstallerResult",
    "InstallerError",
    "InstallerProgress",
    "ProgressCallback",
    # Windows
    "WindowsInstaller",
    "create_windows_installer",
    # macOS
    "MacOSInstaller",
    "create_macos_installer",
    # Linux
    "LinuxInstaller",
    "LinuxDistro",
    "create_linux_installer",
    # Docker
    "DockerInstaller",
    "DockerConfig",
    "create_docker_installer",
    # CLI
    "InstallCLI",
    "run_installer",
]
