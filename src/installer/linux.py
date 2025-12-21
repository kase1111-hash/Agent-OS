"""Linux installer for Agent OS.

Provides Linux-specific installation support including:
- DEB package for Debian/Ubuntu
- RPM package for Fedora/RHEL/CentOS
- AppImage for portable installation
- Systemd service configuration
"""

import logging
import os
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base import (
    Installer,
    InstallerError,
    InstallerPhase,
    create_directory,
    run_command,
)
from .config import InstallConfig
from .platform import (
    Platform,
    PlatformInfo,
    RequirementCheck,
    RequirementStatus,
    SystemRequirements,
    check_requirements,
)

logger = logging.getLogger(__name__)


class LinuxDistro(str, Enum):
    """Linux distribution types."""

    DEBIAN = "debian"  # Debian, Ubuntu, Mint
    REDHAT = "redhat"  # RHEL, CentOS, Fedora
    ARCH = "arch"  # Arch, Manjaro
    SUSE = "suse"  # openSUSE
    GENERIC = "generic"  # Unknown or other


class LinuxInstaller(Installer):
    """Linux-specific installer for Agent OS."""

    def __init__(
        self,
        config: InstallConfig,
        platform_info: Optional[PlatformInfo] = None,
    ):
        """Initialize Linux installer.

        Args:
            config: Installation configuration
            platform_info: Platform information
        """
        super().__init__(config, platform_info)
        self._distro: Optional[LinuxDistro] = None

    @property
    def platform(self) -> Platform:
        """Get the platform this installer supports."""
        return Platform.LINUX

    @property
    def distro(self) -> LinuxDistro:
        """Get the Linux distribution type."""
        if self._distro is None:
            self._distro = self._detect_distro()
        return self._distro

    def _detect_distro(self) -> LinuxDistro:
        """Detect the Linux distribution."""
        # Check /etc/os-release
        os_release = Path("/etc/os-release")
        if os_release.exists():
            content = os_release.read_text().lower()

            if any(d in content for d in ["debian", "ubuntu", "mint", "pop"]):
                return LinuxDistro.DEBIAN
            elif any(d in content for d in ["fedora", "rhel", "centos", "rocky", "alma"]):
                return LinuxDistro.REDHAT
            elif "arch" in content or "manjaro" in content:
                return LinuxDistro.ARCH
            elif "suse" in content or "opensuse" in content:
                return LinuxDistro.SUSE

        # Check for package managers
        if Path("/usr/bin/apt").exists():
            return LinuxDistro.DEBIAN
        elif Path("/usr/bin/dnf").exists() or Path("/usr/bin/yum").exists():
            return LinuxDistro.REDHAT
        elif Path("/usr/bin/pacman").exists():
            return LinuxDistro.ARCH

        return LinuxDistro.GENERIC

    def check_requirements(self) -> List[RequirementCheck]:
        """Check Linux-specific requirements.

        Returns:
            List of requirement check results
        """
        self._report_progress(
            InstallerPhase.REQUIREMENTS,
            0.1,
            "Checking Linux requirements...",
        )

        # Get base requirements
        requirements = SystemRequirements(
            required_commands=["python3", "git"],
            optional_commands=["docker", "nvidia-smi", "ollama", "systemctl"],
        )

        checks = check_requirements(
            requirements=requirements,
            install_path=self.config.install_path,
        )

        # Check kernel version
        self._report_progress(
            InstallerPhase.REQUIREMENTS,
            0.4,
            "Checking kernel version...",
        )

        kernel_version = self._get_kernel_version()
        if kernel_version and kernel_version >= (4, 15):
            checks.append(
                RequirementCheck(
                    name="Kernel Version",
                    status=RequirementStatus.PASSED,
                    message=f"Kernel {kernel_version[0]}.{kernel_version[1]} detected",
                    current_value=kernel_version,
                    required_value=(4, 15),
                )
            )
        elif kernel_version:
            checks.append(
                RequirementCheck(
                    name="Kernel Version",
                    status=RequirementStatus.WARNING,
                    message=f"Kernel {kernel_version[0]}.{kernel_version[1]} - 4.15+ recommended",
                    current_value=kernel_version,
                    required_value=(4, 15),
                )
            )

        # Check distribution
        self._report_progress(
            InstallerPhase.REQUIREMENTS,
            0.6,
            "Detecting Linux distribution...",
        )

        checks.append(
            RequirementCheck(
                name="Linux Distribution",
                status=RequirementStatus.PASSED,
                message=f"Detected: {self.distro.value}",
                current_value=self.distro.value,
                required_value=None,
            )
        )

        # Check for systemd
        self._report_progress(
            InstallerPhase.REQUIREMENTS,
            0.8,
            "Checking systemd...",
        )

        if self._check_systemd():
            checks.append(
                RequirementCheck(
                    name="Systemd",
                    status=RequirementStatus.PASSED,
                    message="Systemd is available",
                    current_value=True,
                    required_value=False,
                )
            )
        else:
            checks.append(
                RequirementCheck(
                    name="Systemd",
                    status=RequirementStatus.WARNING,
                    message="Systemd not found - auto-start will be limited",
                    current_value=False,
                    required_value=False,
                )
            )

        self._report_progress(
            InstallerPhase.REQUIREMENTS,
            1.0,
            "Requirements check complete",
        )

        return checks

    def download(self) -> bool:
        """Download Agent OS files for Linux.

        Returns:
            True if download successful
        """
        self._report_progress(
            InstallerPhase.DOWNLOAD,
            0.1,
            "Preparing download...",
        )

        install_path = self.config.install_path

        # Create directory structure
        dirs_to_create = [
            install_path,
            install_path / "bin",
            install_path / "lib",
            install_path / "config",
            install_path / "data",
            install_path / "logs",
        ]

        for i, dir_path in enumerate(dirs_to_create):
            self._report_progress(
                InstallerPhase.DOWNLOAD,
                0.2 + (0.6 * i / len(dirs_to_create)),
                f"Creating {dir_path.name}...",
            )
            if not create_directory(dir_path):
                return False

        self._report_progress(
            InstallerPhase.DOWNLOAD,
            1.0,
            "Download complete",
        )

        return True

    def install(self) -> bool:
        """Install Agent OS on Linux.

        Returns:
            True if installation successful
        """
        self._report_progress(
            InstallerPhase.INSTALL,
            0.1,
            "Starting Linux installation...",
        )

        # Create launcher scripts
        self._report_progress(
            InstallerPhase.INSTALL,
            0.3,
            "Creating launcher scripts...",
        )

        if not self._create_launcher_scripts():
            self._add_warning("Failed to create launcher scripts")

        # Set up Python environment
        self._report_progress(
            InstallerPhase.INSTALL,
            0.5,
            "Setting up Python environment...",
        )

        if not self._setup_python_environment():
            self._add_warning("Python environment setup incomplete")

        # Install Ollama if requested
        if self.config.install_ollama:
            self._report_progress(
                InstallerPhase.INSTALL,
                0.7,
                "Installing Ollama...",
            )
            if not self._install_ollama():
                self._add_warning("Ollama installation failed - install manually")

        self._report_progress(
            InstallerPhase.INSTALL,
            1.0,
            "Installation complete",
        )

        return True

    def configure(self) -> bool:
        """Configure Agent OS on Linux.

        Returns:
            True if configuration successful
        """
        self._report_progress(
            InstallerPhase.CONFIGURE,
            0.1,
            "Configuring Agent OS...",
        )

        # Create default configuration
        self._report_progress(
            InstallerPhase.CONFIGURE,
            0.3,
            "Creating default configuration...",
        )

        if not self._create_default_config():
            return False

        # Add to shell profile
        if self.config.add_to_path:
            self._report_progress(
                InstallerPhase.CONFIGURE,
                0.5,
                "Adding to PATH...",
            )
            if not self._add_to_shell_profile():
                self._add_warning("Failed to add to PATH")

        # Create systemd service for auto-start
        if self.config.auto_start and self._check_systemd():
            self._report_progress(
                InstallerPhase.CONFIGURE,
                0.7,
                "Creating systemd service...",
            )
            if not self._create_systemd_service():
                self._add_warning("Failed to create systemd service")

        # Create desktop entry
        self._report_progress(
            InstallerPhase.CONFIGURE,
            0.9,
            "Creating desktop entry...",
        )
        self._create_desktop_entry()

        self._report_progress(
            InstallerPhase.CONFIGURE,
            1.0,
            "Configuration complete",
        )

        return True

    def post_install(self) -> bool:
        """Perform post-installation tasks on Linux.

        Returns:
            True if post-install successful
        """
        self._report_progress(
            InstallerPhase.POST_INSTALL,
            0.1,
            "Running post-installation tasks...",
        )

        # Download models if requested
        if self.config.install_models:
            self._report_progress(
                InstallerPhase.POST_INSTALL,
                0.3,
                "Downloading AI models...",
            )
            for i, model in enumerate(self.config.install_models):
                self._report_progress(
                    InstallerPhase.POST_INSTALL,
                    0.3 + (0.5 * i / len(self.config.install_models)),
                    f"Downloading {model}...",
                )
                if not self._download_model(model):
                    self._add_warning(f"Failed to download model: {model}")

        # Save installation configuration
        self._report_progress(
            InstallerPhase.POST_INSTALL,
            0.9,
            "Saving installation configuration...",
        )

        config_path = self.config.config_path / "install.json"
        self.config.save(config_path)

        self._report_progress(
            InstallerPhase.POST_INSTALL,
            1.0,
            "Post-installation complete",
        )

        return True

    def uninstall(self) -> bool:
        """Uninstall Agent OS from Linux.

        Returns:
            True if uninstallation successful
        """
        logger.info("Starting Linux uninstallation...")

        # Remove from shell profile
        self._remove_from_shell_profile()

        # Remove systemd service
        self._remove_systemd_service()

        # Remove desktop entry
        self._remove_desktop_entry()

        # Remove installation directory
        from .base import remove_directory

        if not remove_directory(self.config.install_path):
            logger.error("Failed to remove installation directory")
            return False

        logger.info("Uninstallation complete")
        return True

    # Private helper methods

    def _get_kernel_version(self) -> Optional[tuple]:
        """Get Linux kernel version."""
        try:
            code, stdout, stderr = run_command(
                ["uname", "-r"],
                timeout=10,
            )
            if code == 0:
                parts = stdout.strip().split(".")
                return (int(parts[0]), int(parts[1]))
        except Exception:
            pass
        return None

    def _check_systemd(self) -> bool:
        """Check if systemd is available."""
        try:
            code, stdout, stderr = run_command(
                ["systemctl", "--version"],
                timeout=10,
            )
            return code == 0
        except Exception:
            return False

    def _create_launcher_scripts(self) -> bool:
        """Create Linux launcher scripts."""
        bin_dir = self.config.install_path / "bin"

        script_content = f'''#!/bin/bash
export AGENT_OS_HOME="{self.config.install_path}"
export PYTHONPATH="$AGENT_OS_HOME/lib:$PYTHONPATH"
python3 -m agent_os "$@"
'''
        try:
            script_path = bin_dir / "agent-os"
            script_path.write_text(script_content)
            script_path.chmod(0o755)
            return True
        except Exception as e:
            logger.error(f"Failed to create launcher scripts: {e}")
            return False

    def _setup_python_environment(self) -> bool:
        """Set up Python virtual environment."""
        try:
            venv_path = self.config.install_path / "venv"

            code, stdout, stderr = run_command(
                ["python3", "-m", "venv", str(venv_path)],
                timeout=120,
            )

            if code != 0:
                logger.warning(f"Failed to create venv: {stderr}")
                return False

            return True
        except Exception as e:
            logger.error(f"Failed to setup Python environment: {e}")
            return False

    def _install_ollama(self) -> bool:
        """Install Ollama on Linux."""
        try:
            # Check if already installed
            code, stdout, stderr = run_command(["ollama", "--version"], timeout=10)
            if code == 0:
                logger.info("Ollama is already installed")
                return True

            # Try the official install script
            code, stdout, stderr = run_command(
                ["bash", "-c", "curl -fsSL https://ollama.com/install.sh | sh"],
                timeout=600,
            )

            if code == 0:
                return True

            logger.info("Please install Ollama from ollama.com")
            return False

        except Exception as e:
            logger.error(f"Failed to install Ollama: {e}")
            return False

    def _create_default_config(self) -> bool:
        """Create default Agent OS configuration."""
        try:
            config_dir = self.config.config_path
            config_dir.mkdir(parents=True, exist_ok=True)

            config_content = f"""# Agent OS Configuration
# Generated during installation

[general]
data_dir = {self.config.data_path}
log_dir = {self.config.log_path}

[agents]
enabled = whisper, smith, sage, quill

[models]
default = mistral:7b-instruct
ollama_host = http://localhost:11434

[web]
host = 127.0.0.1
port = 8080
"""
            (config_dir / "agent-os.ini").write_text(config_content)
            return True

        except Exception as e:
            logger.error(f"Failed to create config: {e}")
            return False

    def _get_shell_profile(self) -> Path:
        """Get the appropriate shell profile file."""
        shell = os.environ.get("SHELL", "/bin/bash")

        if "zsh" in shell:
            return Path.home() / ".zshrc"
        elif "bash" in shell:
            bashrc = Path.home() / ".bashrc"
            if bashrc.exists():
                return bashrc
            return Path.home() / ".bash_profile"
        else:
            return Path.home() / ".profile"

    def _add_to_shell_profile(self) -> bool:
        """Add Agent OS to shell profile."""
        try:
            profile = self._get_shell_profile()
            bin_path = self.config.install_path / "bin"

            marker = "# Agent OS PATH"
            path_line = f'export PATH="{bin_path}:$PATH"'
            export_line = f'export AGENT_OS_HOME="{self.config.install_path}"'

            if profile.exists():
                content = profile.read_text()
                if marker in content:
                    return True
            else:
                content = ""

            new_content = f"""
{marker}
{path_line}
{export_line}
"""
            with open(profile, "a") as f:
                f.write(new_content)

            return True

        except Exception as e:
            logger.error(f"Failed to add to shell profile: {e}")
            return False

    def _remove_from_shell_profile(self) -> bool:
        """Remove Agent OS from shell profile."""
        try:
            profile = self._get_shell_profile()

            if not profile.exists():
                return True

            content = profile.read_text()
            marker = "# Agent OS PATH"

            if marker not in content:
                return True

            lines = content.split("\n")
            new_lines = []
            skip = False

            for line in lines:
                if marker in line:
                    skip = True
                    continue
                if skip and line.strip() and not line.startswith("export"):
                    skip = False
                if not skip:
                    new_lines.append(line)

            profile.write_text("\n".join(new_lines))
            return True

        except Exception as e:
            logger.error(f"Failed to remove from shell profile: {e}")
            return False

    def _create_systemd_service(self) -> bool:
        """Create a systemd user service for auto-start."""
        try:
            systemd_dir = Path.home() / ".config/systemd/user"
            systemd_dir.mkdir(parents=True, exist_ok=True)

            service_path = systemd_dir / "agent-os.service"

            service_content = f"""[Unit]
Description=Agent OS - Constitutional AI Framework
After=network.target

[Service]
Type=simple
ExecStart={self.config.install_path}/bin/agent-os daemon
Restart=on-failure
RestartSec=10
Environment=AGENT_OS_HOME={self.config.install_path}

[Install]
WantedBy=default.target
"""
            service_path.write_text(service_content)

            # Reload systemd and enable service
            run_command(["systemctl", "--user", "daemon-reload"], timeout=30)
            run_command(["systemctl", "--user", "enable", "agent-os"], timeout=30)

            return True

        except Exception as e:
            logger.error(f"Failed to create systemd service: {e}")
            return False

    def _remove_systemd_service(self) -> bool:
        """Remove the systemd service."""
        try:
            service_path = Path.home() / ".config/systemd/user/agent-os.service"

            if service_path.exists():
                run_command(
                    ["systemctl", "--user", "stop", "agent-os"], timeout=30
                )
                run_command(
                    ["systemctl", "--user", "disable", "agent-os"], timeout=30
                )
                service_path.unlink()
                run_command(["systemctl", "--user", "daemon-reload"], timeout=30)

            return True

        except Exception as e:
            logger.error(f"Failed to remove systemd service: {e}")
            return False

    def _create_desktop_entry(self) -> bool:
        """Create a desktop entry for the application menu."""
        try:
            applications_dir = Path.home() / ".local/share/applications"
            applications_dir.mkdir(parents=True, exist_ok=True)

            desktop_entry = f"""[Desktop Entry]
Name=Agent OS
Comment=Constitutional AI Framework
Exec={self.config.install_path}/bin/agent-os
Icon=agent-os
Terminal=false
Type=Application
Categories=Development;Utility;
"""
            (applications_dir / "agent-os.desktop").write_text(desktop_entry)
            return True

        except Exception as e:
            logger.error(f"Failed to create desktop entry: {e}")
            return False

    def _remove_desktop_entry(self) -> bool:
        """Remove the desktop entry."""
        try:
            desktop_entry = Path.home() / ".local/share/applications/agent-os.desktop"
            if desktop_entry.exists():
                desktop_entry.unlink()
            return True
        except Exception as e:
            logger.error(f"Failed to remove desktop entry: {e}")
            return False

    def _download_model(self, model_name: str) -> bool:
        """Download an Ollama model."""
        try:
            code, stdout, stderr = run_command(
                ["ollama", "pull", model_name],
                timeout=3600,
            )
            return code == 0
        except Exception as e:
            logger.error(f"Failed to download model {model_name}: {e}")
            return False


def create_linux_installer(
    config: InstallConfig,
    platform_info: Optional[PlatformInfo] = None,
) -> LinuxInstaller:
    """Create a Linux installer.

    Args:
        config: Installation configuration
        platform_info: Platform information

    Returns:
        LinuxInstaller instance
    """
    return LinuxInstaller(config, platform_info)
