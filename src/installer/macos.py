"""macOS installer for Agent OS.

Provides macOS-specific installation support including:
- Homebrew integration
- DMG package support
- Launch agent configuration
- Application bundle creation
"""

import logging
import os
import plistlib
import subprocess
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


class MacOSInstaller(Installer):
    """macOS-specific installer for Agent OS."""

    @property
    def platform(self) -> Platform:
        """Get the platform this installer supports."""
        return Platform.MACOS

    def check_requirements(self) -> List[RequirementCheck]:
        """Check macOS-specific requirements.

        Returns:
            List of requirement check results
        """
        self._report_progress(
            InstallerPhase.REQUIREMENTS,
            0.1,
            "Checking macOS requirements...",
        )

        # Get base requirements
        requirements = SystemRequirements(
            required_commands=["python3", "git"],
            optional_commands=["brew", "docker", "ollama"],
        )

        checks = check_requirements(
            requirements=requirements,
            install_path=self.config.install_path,
        )

        # Check macOS version
        self._report_progress(
            InstallerPhase.REQUIREMENTS,
            0.4,
            "Checking macOS version...",
        )

        macos_version = self._get_macos_version()
        if macos_version and macos_version >= (11, 0):  # Big Sur or later
            checks.append(
                RequirementCheck(
                    name="macOS Version",
                    status=RequirementStatus.PASSED,
                    message=f"macOS {macos_version[0]}.{macos_version[1]} detected",
                    current_value=macos_version,
                    required_value=(11, 0),
                )
            )
        elif macos_version and macos_version >= (10, 15):  # Catalina
            checks.append(
                RequirementCheck(
                    name="macOS Version",
                    status=RequirementStatus.WARNING,
                    message=f"macOS {macos_version[0]}.{macos_version[1]} - Big Sur or later recommended",
                    current_value=macos_version,
                    required_value=(11, 0),
                )
            )
        else:
            checks.append(
                RequirementCheck(
                    name="macOS Version",
                    status=RequirementStatus.FAILED,
                    message="macOS Catalina (10.15) or later required",
                    current_value=macos_version,
                    required_value=(10, 15),
                    is_critical=True,
                )
            )

        # Check for Xcode Command Line Tools
        self._report_progress(
            InstallerPhase.REQUIREMENTS,
            0.6,
            "Checking Xcode Command Line Tools...",
        )

        if self._check_xcode_cli():
            checks.append(
                RequirementCheck(
                    name="Xcode CLI Tools",
                    status=RequirementStatus.PASSED,
                    message="Xcode Command Line Tools installed",
                    current_value=True,
                    required_value=True,
                )
            )
        else:
            checks.append(
                RequirementCheck(
                    name="Xcode CLI Tools",
                    status=RequirementStatus.FAILED,
                    message="Xcode Command Line Tools required. Run: xcode-select --install",
                    current_value=False,
                    required_value=True,
                    is_critical=True,
                )
            )

        # Check for Homebrew
        self._report_progress(
            InstallerPhase.REQUIREMENTS,
            0.8,
            "Checking Homebrew...",
        )

        if self._check_homebrew():
            checks.append(
                RequirementCheck(
                    name="Homebrew",
                    status=RequirementStatus.PASSED,
                    message="Homebrew is installed",
                    current_value=True,
                    required_value=False,
                )
            )
        else:
            checks.append(
                RequirementCheck(
                    name="Homebrew",
                    status=RequirementStatus.WARNING,
                    message="Homebrew not found - some features may be limited",
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
        """Download Agent OS files for macOS.

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
        """Install Agent OS on macOS.

        Returns:
            True if installation successful
        """
        self._report_progress(
            InstallerPhase.INSTALL,
            0.1,
            "Starting macOS installation...",
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
        """Configure Agent OS on macOS.

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

        # Create launch agent for auto-start
        if self.config.auto_start:
            self._report_progress(
                InstallerPhase.CONFIGURE,
                0.7,
                "Creating Launch Agent...",
            )
            if not self._create_launch_agent():
                self._add_warning("Failed to create Launch Agent")

        self._report_progress(
            InstallerPhase.CONFIGURE,
            1.0,
            "Configuration complete",
        )

        return True

    def post_install(self) -> bool:
        """Perform post-installation tasks on macOS.

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
        """Uninstall Agent OS from macOS.

        Returns:
            True if uninstallation successful
        """
        logger.info("Starting macOS uninstallation...")

        # Remove from shell profile
        self._remove_from_shell_profile()

        # Remove Launch Agent
        self._remove_launch_agent()

        # Remove installation directory
        from .base import remove_directory

        if not remove_directory(self.config.install_path):
            logger.error("Failed to remove installation directory")
            return False

        logger.info("Uninstallation complete")
        return True

    # Private helper methods

    def _get_macos_version(self) -> Optional[tuple]:
        """Get macOS version."""
        try:
            code, stdout, stderr = run_command(
                ["sw_vers", "-productVersion"],
                timeout=10,
            )
            if code == 0:
                parts = stdout.strip().split(".")
                return (int(parts[0]), int(parts[1]) if len(parts) > 1 else 0)
        except Exception:
            pass
        return None

    def _check_xcode_cli(self) -> bool:
        """Check if Xcode Command Line Tools are installed."""
        try:
            code, stdout, stderr = run_command(
                ["xcode-select", "-p"],
                timeout=10,
            )
            return code == 0
        except Exception:
            return False

    def _check_homebrew(self) -> bool:
        """Check if Homebrew is installed."""
        try:
            code, stdout, stderr = run_command(
                ["brew", "--version"],
                timeout=10,
            )
            return code == 0
        except Exception:
            return False

    def _create_launcher_scripts(self) -> bool:
        """Create macOS launcher scripts."""
        bin_dir = self.config.install_path / "bin"

        # Create agent-os script
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
        """Install Ollama on macOS."""
        try:
            # Check if already installed
            code, stdout, stderr = run_command(["ollama", "--version"], timeout=10)
            if code == 0:
                logger.info("Ollama is already installed")
                return True

            # Try Homebrew
            if self._check_homebrew():
                code, stdout, stderr = run_command(
                    ["brew", "install", "ollama"],
                    timeout=600,
                )
                return code == 0

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
        shell = os.environ.get("SHELL", "/bin/zsh")

        if "zsh" in shell:
            return Path.home() / ".zshrc"
        elif "bash" in shell:
            # On macOS, use .bash_profile for login shells
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

            # Read existing content
            if profile.exists():
                content = profile.read_text()
                if marker in content:
                    return True  # Already configured
            else:
                content = ""

            # Add configuration
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

            # Remove the Agent OS section
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

    def _create_launch_agent(self) -> bool:
        """Create a macOS Launch Agent for auto-start."""
        try:
            launch_agents_dir = Path.home() / "Library/LaunchAgents"
            launch_agents_dir.mkdir(parents=True, exist_ok=True)

            plist_path = launch_agents_dir / "com.agentos.daemon.plist"

            plist_content = {
                "Label": "com.agentos.daemon",
                "ProgramArguments": [
                    str(self.config.install_path / "bin/agent-os"),
                    "daemon",
                ],
                "RunAtLoad": True,
                "KeepAlive": True,
                "StandardOutPath": str(self.config.log_path / "daemon.log"),
                "StandardErrorPath": str(self.config.log_path / "daemon.error.log"),
                "EnvironmentVariables": {
                    "AGENT_OS_HOME": str(self.config.install_path),
                },
            }

            with open(plist_path, "wb") as f:
                plistlib.dump(plist_content, f)

            # Load the Launch Agent
            run_command(["launchctl", "load", str(plist_path)], timeout=30)

            return True

        except Exception as e:
            logger.error(f"Failed to create Launch Agent: {e}")
            return False

    def _remove_launch_agent(self) -> bool:
        """Remove the macOS Launch Agent."""
        try:
            plist_path = Path.home() / "Library/LaunchAgents/com.agentos.daemon.plist"

            if plist_path.exists():
                # Unload first
                run_command(["launchctl", "unload", str(plist_path)], timeout=30)
                plist_path.unlink()

            return True

        except Exception as e:
            logger.error(f"Failed to remove Launch Agent: {e}")
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


def create_macos_installer(
    config: InstallConfig,
    platform_info: Optional[PlatformInfo] = None,
) -> MacOSInstaller:
    """Create a macOS installer.

    Args:
        config: Installation configuration
        platform_info: Platform information

    Returns:
        MacOSInstaller instance
    """
    return MacOSInstaller(config, platform_info)
