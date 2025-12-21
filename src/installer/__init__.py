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

from .platform import (
    Platform,
    PlatformInfo,
    SystemRequirements,
    RequirementCheck,
    RequirementStatus,
    detect_platform,
    check_requirements,
    get_system_info,
    all_requirements_met,
)
from .config import (
    InstallConfig,
    InstallLocation,
    InstallMode,
    ComponentSelection,
    create_install_config,
)
from .base import (
    Installer,
    InstallerResult,
    InstallerError,
    InstallerProgress,
    ProgressCallback,
)
from .windows import (
    WindowsInstaller,
    create_windows_installer,
)
from .macos import (
    MacOSInstaller,
    create_macos_installer,
)
from .linux import (
    LinuxInstaller,
    LinuxDistro,
    create_linux_installer,
)
from .docker import (
    DockerInstaller,
    DockerConfig,
    create_docker_installer,
)
from .cli import (
    InstallCLI,
    run_installer,
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
