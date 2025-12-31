"""Windows installer for Agent OS.

Provides Windows-specific installation support including:
- MSI package installation
- Portable installation
- Registry configuration
- Start menu shortcuts
- PATH environment variable configuration
"""

import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Windows-only import
if sys.platform == "win32":
    import winreg
else:
    winreg = None  # type: ignore

from .base import (
    Installer,
    InstallerError,
    InstallerPhase,
    copy_directory,
    create_directory,
    run_command,
)
from .config import InstallConfig, InstallMode
from .platform import (
    Platform,
    PlatformInfo,
    RequirementCheck,
    RequirementStatus,
    SystemRequirements,
    check_requirements,
)

logger = logging.getLogger(__name__)


class WindowsInstaller(Installer):
    """Windows-specific installer for Agent OS."""

    @property
    def platform(self) -> Platform:
        """Get the platform this installer supports."""
        return Platform.WINDOWS

    def check_requirements(self) -> List[RequirementCheck]:
        """Check Windows-specific requirements.

        Returns:
            List of requirement check results
        """
        self._report_progress(
            InstallerPhase.REQUIREMENTS,
            0.1,
            "Checking Windows requirements...",
        )

        # Get base requirements
        requirements = SystemRequirements(
            required_commands=["python", "git"],
            optional_commands=["docker", "wsl", "nvidia-smi"],
        )

        checks = check_requirements(
            requirements=requirements,
            install_path=self.config.install_path,
        )

        # Check Windows version
        self._report_progress(
            InstallerPhase.REQUIREMENTS,
            0.5,
            "Checking Windows version...",
        )

        win_version = self._get_windows_version()
        if win_version and win_version >= (10, 0):
            checks.append(
                RequirementCheck(
                    name="Windows Version",
                    status=RequirementStatus.PASSED,
                    message=f"Windows {win_version[0]}.{win_version[1]} detected",
                    current_value=win_version,
                    required_value=(10, 0),
                )
            )
        else:
            checks.append(
                RequirementCheck(
                    name="Windows Version",
                    status=RequirementStatus.WARNING,
                    message="Windows 10 or later recommended",
                    current_value=win_version,
                    required_value=(10, 0),
                )
            )

        # Check for Visual C++ Redistributable
        self._report_progress(
            InstallerPhase.REQUIREMENTS,
            0.8,
            "Checking Visual C++ Redistributable...",
        )

        if self._check_vcredist():
            checks.append(
                RequirementCheck(
                    name="Visual C++ Redistributable",
                    status=RequirementStatus.PASSED,
                    message="Visual C++ Redistributable is installed",
                    current_value=True,
                    required_value=True,
                )
            )
        else:
            checks.append(
                RequirementCheck(
                    name="Visual C++ Redistributable",
                    status=RequirementStatus.WARNING,
                    message="Visual C++ Redistributable may need to be installed",
                    current_value=False,
                    required_value=True,
                )
            )

        self._report_progress(
            InstallerPhase.REQUIREMENTS,
            1.0,
            "Requirements check complete",
        )

        return checks

    def download(self) -> bool:
        """Download Agent OS files for Windows.

        Returns:
            True if download successful
        """
        self._report_progress(
            InstallerPhase.DOWNLOAD,
            0.1,
            "Preparing download...",
        )

        # For now, we simulate downloading by creating the installation structure
        # In production, this would download from release servers

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
        """Install Agent OS on Windows.

        Returns:
            True if installation successful
        """
        self._report_progress(
            InstallerPhase.INSTALL,
            0.1,
            "Starting Windows installation...",
        )

        install_path = self.config.install_path

        # Create launcher scripts
        self._report_progress(
            InstallerPhase.INSTALL,
            0.3,
            "Creating launcher scripts...",
        )

        if not self._create_launcher_scripts():
            self._add_warning("Failed to create launcher scripts")

        # Install Python dependencies
        self._report_progress(
            InstallerPhase.INSTALL,
            0.5,
            "Installing Python dependencies...",
        )

        # Create virtual environment (optional based on config)
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
        """Configure Agent OS on Windows.

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

        # Add to PATH if requested
        if self.config.add_to_path:
            self._report_progress(
                InstallerPhase.CONFIGURE,
                0.5,
                "Adding to PATH...",
            )
            if not self._add_to_path():
                self._add_warning("Failed to add to PATH")

        # Create shortcuts if requested
        if self.config.create_shortcuts:
            self._report_progress(
                InstallerPhase.CONFIGURE,
                0.7,
                "Creating shortcuts...",
            )
            if not self._create_shortcuts():
                self._add_warning("Failed to create shortcuts")

        # Register with Windows
        self._report_progress(
            InstallerPhase.CONFIGURE,
            0.9,
            "Registering with Windows...",
        )
        self._register_with_windows()

        self._report_progress(
            InstallerPhase.CONFIGURE,
            1.0,
            "Configuration complete",
        )

        return True

    def post_install(self) -> bool:
        """Perform post-installation tasks on Windows.

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
        """Uninstall Agent OS from Windows.

        Returns:
            True if uninstallation successful
        """
        logger.info("Starting Windows uninstallation...")

        # Remove from PATH
        self._remove_from_path()

        # Remove shortcuts
        self._remove_shortcuts()

        # Unregister from Windows
        self._unregister_from_windows()

        # Remove installation directory
        from .base import remove_directory

        if not remove_directory(self.config.install_path):
            logger.error("Failed to remove installation directory")
            return False

        logger.info("Uninstallation complete")
        return True

    # Private helper methods

    def _get_windows_version(self) -> Optional[tuple]:
        """Get Windows version."""
        try:
            import sys

            if hasattr(sys, "getwindowsversion"):
                ver = sys.getwindowsversion()
                return (ver.major, ver.minor)
        except Exception:
            pass
        return None

    def _check_vcredist(self) -> bool:
        """Check if Visual C++ Redistributable is installed."""
        if winreg is None:
            return False

        try:
            # Check registry for VC++ 2015-2022
            key_paths = [
                r"SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\x64",
                r"SOFTWARE\WOW6432Node\Microsoft\VisualStudio\14.0\VC\Runtimes\x64",
            ]

            for key_path in key_paths:
                try:
                    with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, key_path):
                        return True
                except FileNotFoundError:
                    continue

        except Exception:
            pass

        return False

    def _create_launcher_scripts(self) -> bool:
        """Create Windows launcher scripts."""
        bin_dir = self.config.install_path / "bin"

        # Create agent-os.cmd
        cmd_content = f"""@echo off
setlocal
set AGENT_OS_HOME={self.config.install_path}
set PYTHONPATH=%AGENT_OS_HOME%\\lib;%PYTHONPATH%
python -m agent_os %*
"""
        try:
            (bin_dir / "agent-os.cmd").write_text(cmd_content)

            # Create agent-os.ps1 for PowerShell
            ps_content = f"""$env:AGENT_OS_HOME = "{self.config.install_path}"
$env:PYTHONPATH = "$env:AGENT_OS_HOME\\lib;$env:PYTHONPATH"
python -m agent_os $args
"""
            (bin_dir / "agent-os.ps1").write_text(ps_content)

            return True
        except Exception as e:
            logger.error(f"Failed to create launcher scripts: {e}")
            return False

    def _setup_python_environment(self) -> bool:
        """Set up Python virtual environment."""
        try:
            venv_path = self.config.install_path / "venv"

            # Create virtual environment
            code, stdout, stderr = run_command(
                ["python", "-m", "venv", str(venv_path)],
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
        """Install Ollama on Windows."""
        try:
            # Check if Ollama is already installed
            code, stdout, stderr = run_command(["ollama", "--version"], timeout=10)
            if code == 0:
                logger.info("Ollama is already installed")
                return True

            # Ollama needs to be installed via Windows installer
            logger.info("Ollama not found - please install from ollama.com")
            return False

        except Exception as e:
            logger.error(f"Failed to install Ollama: {e}")
            return False

    def _create_default_config(self) -> bool:
        """Create default Agent OS configuration."""
        try:
            config_dir = self.config.config_path
            config_dir.mkdir(parents=True, exist_ok=True)

            # Create main config file
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

    def _add_to_path(self) -> bool:
        """Add Agent OS to Windows PATH."""
        if winreg is None:
            return False

        try:
            bin_path = str(self.config.install_path / "bin")

            # Get current user PATH
            with winreg.OpenKey(
                winreg.HKEY_CURRENT_USER,
                r"Environment",
                0,
                winreg.KEY_ALL_ACCESS,
            ) as key:
                try:
                    current_path, _ = winreg.QueryValueEx(key, "Path")
                except FileNotFoundError:
                    current_path = ""

                if bin_path not in current_path:
                    new_path = f"{current_path};{bin_path}" if current_path else bin_path
                    winreg.SetValueEx(key, "Path", 0, winreg.REG_EXPAND_SZ, new_path)

            # Notify system of environment change
            try:
                import ctypes

                HWND_BROADCAST = 0xFFFF
                WM_SETTINGCHANGE = 0x001A
                ctypes.windll.user32.SendMessageW(
                    HWND_BROADCAST, WM_SETTINGCHANGE, 0, "Environment"
                )
            except Exception:
                pass

            return True

        except Exception as e:
            logger.error(f"Failed to add to PATH: {e}")
            return False

    def _remove_from_path(self) -> bool:
        """Remove Agent OS from Windows PATH."""
        if winreg is None:
            return False

        try:
            bin_path = str(self.config.install_path / "bin")

            with winreg.OpenKey(
                winreg.HKEY_CURRENT_USER,
                r"Environment",
                0,
                winreg.KEY_ALL_ACCESS,
            ) as key:
                try:
                    current_path, _ = winreg.QueryValueEx(key, "Path")
                    paths = [p for p in current_path.split(";") if p != bin_path]
                    new_path = ";".join(paths)
                    winreg.SetValueEx(key, "Path", 0, winreg.REG_EXPAND_SZ, new_path)
                except FileNotFoundError:
                    pass

            return True

        except Exception as e:
            logger.error(f"Failed to remove from PATH: {e}")
            return False

    def _create_shortcuts(self) -> bool:
        """Create Windows shortcuts."""
        try:
            # Create Start Menu shortcut
            start_menu = Path(os.environ["APPDATA"]) / "Microsoft/Windows/Start Menu/Programs"
            shortcut_dir = start_menu / "Agent OS"
            shortcut_dir.mkdir(parents=True, exist_ok=True)

            # Use PowerShell to create shortcut
            shortcut_path = shortcut_dir / "Agent OS.lnk"
            target = self.config.install_path / "bin" / "agent-os.cmd"

            ps_script = f"""
$WshShell = New-Object -comObject WScript.Shell
$Shortcut = $WshShell.CreateShortcut("{shortcut_path}")
$Shortcut.TargetPath = "{target}"
$Shortcut.WorkingDirectory = "{self.config.install_path}"
$Shortcut.Description = "Agent OS - Constitutional AI Framework"
$Shortcut.Save()
"""
            code, stdout, stderr = run_command(
                ["powershell", "-Command", ps_script],
                timeout=30,
            )

            return code == 0

        except Exception as e:
            logger.error(f"Failed to create shortcuts: {e}")
            return False

    def _remove_shortcuts(self) -> bool:
        """Remove Windows shortcuts."""
        try:
            from .base import remove_directory

            start_menu = Path(os.environ["APPDATA"]) / "Microsoft/Windows/Start Menu/Programs"
            shortcut_dir = start_menu / "Agent OS"
            return remove_directory(shortcut_dir)

        except Exception as e:
            logger.error(f"Failed to remove shortcuts: {e}")
            return False

    def _register_with_windows(self) -> bool:
        """Register Agent OS with Windows (add/remove programs)."""
        if winreg is None:
            return False

        try:
            uninstall_key = r"SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall\AgentOS"

            with winreg.CreateKey(winreg.HKEY_CURRENT_USER, uninstall_key) as key:
                winreg.SetValueEx(key, "DisplayName", 0, winreg.REG_SZ, "Agent OS")
                winreg.SetValueEx(
                    key,
                    "DisplayVersion",
                    0,
                    winreg.REG_SZ,
                    self.config.version,
                )
                winreg.SetValueEx(
                    key,
                    "InstallLocation",
                    0,
                    winreg.REG_SZ,
                    str(self.config.install_path),
                )
                winreg.SetValueEx(key, "Publisher", 0, winreg.REG_SZ, "Agent OS Community")
                winreg.SetValueEx(key, "NoModify", 0, winreg.REG_DWORD, 1)
                winreg.SetValueEx(key, "NoRepair", 0, winreg.REG_DWORD, 1)

            return True

        except Exception as e:
            logger.error(f"Failed to register with Windows: {e}")
            return False

    def _unregister_from_windows(self) -> bool:
        """Unregister Agent OS from Windows."""
        if winreg is None:
            return False

        try:
            uninstall_key = r"SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall\AgentOS"
            winreg.DeleteKey(winreg.HKEY_CURRENT_USER, uninstall_key)
            return True
        except FileNotFoundError:
            return True
        except Exception as e:
            logger.error(f"Failed to unregister from Windows: {e}")
            return False

    def _download_model(self, model_name: str) -> bool:
        """Download an Ollama model."""
        try:
            code, stdout, stderr = run_command(
                ["ollama", "pull", model_name],
                timeout=3600,  # 1 hour timeout for large models
            )
            return code == 0
        except Exception as e:
            logger.error(f"Failed to download model {model_name}: {e}")
            return False


def create_windows_installer(
    config: InstallConfig,
    platform_info: Optional[PlatformInfo] = None,
) -> WindowsInstaller:
    """Create a Windows installer.

    Args:
        config: Installation configuration
        platform_info: Platform information

    Returns:
        WindowsInstaller instance
    """
    return WindowsInstaller(config, platform_info)
