"""Base installer interface and common functionality.

Provides:
- Abstract base class for platform-specific installers
- Progress tracking and callbacks
- Common installation utilities
"""

import logging
import shutil
import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from .config import InstallConfig
from .platform import Platform, PlatformInfo, RequirementCheck

logger = logging.getLogger(__name__)


class InstallerPhase(str, Enum):
    """Installation phases."""

    INIT = "init"
    REQUIREMENTS = "requirements"
    DOWNLOAD = "download"
    EXTRACT = "extract"
    INSTALL = "install"
    CONFIGURE = "configure"
    POST_INSTALL = "post_install"
    CLEANUP = "cleanup"
    COMPLETE = "complete"
    FAILED = "failed"


class InstallerError(Exception):
    """Base exception for installer errors."""

    def __init__(
        self,
        message: str,
        phase: InstallerPhase = InstallerPhase.FAILED,
        recoverable: bool = False,
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message)
        self.phase = phase
        self.recoverable = recoverable
        self.details = details or {}


@dataclass
class InstallerProgress:
    """Progress information for installation."""

    phase: InstallerPhase
    progress: float  # 0.0 to 1.0
    message: str
    current_step: int = 0
    total_steps: int = 0
    details: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    @property
    def percent(self) -> int:
        """Get progress as percentage."""
        return int(self.progress * 100)


# Type alias for progress callbacks
ProgressCallback = Callable[[InstallerProgress], None]


@dataclass
class InstallerResult:
    """Result of an installation."""

    success: bool
    config: InstallConfig
    install_path: Path
    duration_seconds: float
    phases_completed: List[InstallerPhase]
    requirement_checks: List[RequirementCheck] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "install_path": str(self.install_path),
            "duration_seconds": self.duration_seconds,
            "phases_completed": [p.value for p in self.phases_completed],
            "warnings": self.warnings,
            "errors": self.errors,
            "metadata": self.metadata,
        }


class Installer(ABC):
    """Abstract base class for platform-specific installers."""

    def __init__(
        self,
        config: InstallConfig,
        platform_info: Optional[PlatformInfo] = None,
    ):
        """Initialize installer.

        Args:
            config: Installation configuration
            platform_info: Platform information (auto-detected if None)
        """
        self.config = config
        self._platform_info = platform_info
        self._progress_callbacks: List[ProgressCallback] = []
        self._current_phase = InstallerPhase.INIT
        self._start_time: Optional[datetime] = None
        self._phases_completed: List[InstallerPhase] = []
        self._warnings: List[str] = []
        self._errors: List[str] = []

    @property
    def platform_info(self) -> PlatformInfo:
        """Get platform information."""
        if self._platform_info is None:
            from .platform import get_system_info

            self._platform_info = get_system_info()
        return self._platform_info

    @property
    @abstractmethod
    def platform(self) -> Platform:
        """Get the platform this installer supports."""
        pass

    def on_progress(self, callback: ProgressCallback) -> None:
        """Register a progress callback.

        Args:
            callback: Function to call with progress updates
        """
        self._progress_callbacks.append(callback)

    def _report_progress(
        self,
        phase: InstallerPhase,
        progress: float,
        message: str,
        current_step: int = 0,
        total_steps: int = 0,
        **details,
    ) -> None:
        """Report progress to registered callbacks.

        Args:
            phase: Current installation phase
            progress: Progress within phase (0.0 to 1.0)
            message: Human-readable status message
            current_step: Current step number
            total_steps: Total number of steps
            **details: Additional details
        """
        self._current_phase = phase
        progress_info = InstallerProgress(
            phase=phase,
            progress=progress,
            message=message,
            current_step=current_step,
            total_steps=total_steps,
            details=details,
        )

        for callback in self._progress_callbacks:
            try:
                callback(progress_info)
            except Exception as e:
                logger.warning(f"Progress callback error: {e}")

    def _add_warning(self, message: str) -> None:
        """Add a warning message."""
        self._warnings.append(message)
        logger.warning(message)

    def _add_error(self, message: str) -> None:
        """Add an error message."""
        self._errors.append(message)
        logger.error(message)

    @abstractmethod
    def check_requirements(self) -> List[RequirementCheck]:
        """Check system requirements for installation.

        Returns:
            List of requirement check results
        """
        pass

    @abstractmethod
    def download(self) -> bool:
        """Download installation files.

        Returns:
            True if download successful
        """
        pass

    @abstractmethod
    def install(self) -> bool:
        """Perform the main installation.

        Returns:
            True if installation successful
        """
        pass

    @abstractmethod
    def configure(self) -> bool:
        """Configure the installed system.

        Returns:
            True if configuration successful
        """
        pass

    @abstractmethod
    def post_install(self) -> bool:
        """Perform post-installation tasks.

        Returns:
            True if post-install successful
        """
        pass

    @abstractmethod
    def uninstall(self) -> bool:
        """Uninstall Agent OS.

        Returns:
            True if uninstallation successful
        """
        pass

    def run(self) -> InstallerResult:
        """Run the complete installation process.

        Returns:
            InstallerResult with installation outcome
        """
        self._start_time = datetime.now()
        self._phases_completed = []
        requirement_checks = []

        try:
            # Phase: Requirements
            self._report_progress(
                InstallerPhase.REQUIREMENTS,
                0.0,
                "Checking system requirements...",
            )
            requirement_checks = self.check_requirements()
            self._phases_completed.append(InstallerPhase.REQUIREMENTS)

            # Check for critical failures
            from .platform import all_requirements_met

            if not all_requirements_met(requirement_checks):
                raise InstallerError(
                    "System does not meet minimum requirements",
                    InstallerPhase.REQUIREMENTS,
                    recoverable=False,
                )

            # Phase: Download
            self._report_progress(
                InstallerPhase.DOWNLOAD,
                0.0,
                "Downloading installation files...",
            )
            if not self.download():
                raise InstallerError(
                    "Download failed",
                    InstallerPhase.DOWNLOAD,
                    recoverable=True,
                )
            self._phases_completed.append(InstallerPhase.DOWNLOAD)

            # Phase: Install
            self._report_progress(
                InstallerPhase.INSTALL,
                0.0,
                "Installing Agent OS...",
            )
            if not self.install():
                raise InstallerError(
                    "Installation failed",
                    InstallerPhase.INSTALL,
                    recoverable=True,
                )
            self._phases_completed.append(InstallerPhase.INSTALL)

            # Phase: Configure
            self._report_progress(
                InstallerPhase.CONFIGURE,
                0.0,
                "Configuring Agent OS...",
            )
            if not self.configure():
                raise InstallerError(
                    "Configuration failed",
                    InstallerPhase.CONFIGURE,
                    recoverable=True,
                )
            self._phases_completed.append(InstallerPhase.CONFIGURE)

            # Phase: Post-Install
            self._report_progress(
                InstallerPhase.POST_INSTALL,
                0.0,
                "Running post-installation tasks...",
            )
            if not self.post_install():
                self._add_warning("Some post-installation tasks failed")
            self._phases_completed.append(InstallerPhase.POST_INSTALL)

            # Phase: Complete
            self._report_progress(
                InstallerPhase.COMPLETE,
                1.0,
                "Installation complete!",
            )
            self._phases_completed.append(InstallerPhase.COMPLETE)

            duration = (datetime.now() - self._start_time).total_seconds()

            return InstallerResult(
                success=True,
                config=self.config,
                install_path=self.config.install_path,
                duration_seconds=duration,
                phases_completed=self._phases_completed,
                requirement_checks=requirement_checks,
                warnings=self._warnings,
                errors=[],
                metadata={"platform": self.platform.value},
            )

        except InstallerError as e:
            self._add_error(str(e))
            self._report_progress(
                InstallerPhase.FAILED,
                0.0,
                f"Installation failed: {e}",
                error=str(e),
            )
            duration = (datetime.now() - self._start_time).total_seconds()

            return InstallerResult(
                success=False,
                config=self.config,
                install_path=self.config.install_path,
                duration_seconds=duration,
                phases_completed=self._phases_completed,
                requirement_checks=requirement_checks,
                warnings=self._warnings,
                errors=self._errors,
                metadata={"platform": self.platform.value, "failed_phase": e.phase.value},
            )

        except Exception as e:
            logger.exception("Unexpected installation error")
            self._add_error(str(e))
            duration = (datetime.now() - self._start_time).total_seconds()

            return InstallerResult(
                success=False,
                config=self.config,
                install_path=self.config.install_path,
                duration_seconds=duration,
                phases_completed=self._phases_completed,
                requirement_checks=requirement_checks,
                warnings=self._warnings,
                errors=self._errors,
                metadata={"platform": self.platform.value, "unexpected_error": True},
            )


# Utility functions


def run_command(
    cmd: List[str],
    cwd: Optional[Path] = None,
    env: Optional[Dict[str, str]] = None,
    timeout: int = 300,
    capture: bool = True,
) -> Tuple[int, str, str]:
    """Run a shell command.

    Args:
        cmd: Command and arguments
        cwd: Working directory
        env: Environment variables
        timeout: Timeout in seconds
        capture: Whether to capture output

    Returns:
        Tuple of (return_code, stdout, stderr)
    """
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            env=env,
            timeout=timeout,
            capture_output=capture,
            text=True,
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return -1, "", "Command timed out"
    except Exception as e:
        return -1, "", str(e)


def create_directory(path: Path, parents: bool = True) -> bool:
    """Create a directory.

    Args:
        path: Directory path
        parents: Create parent directories

    Returns:
        True if successful
    """
    try:
        path.mkdir(parents=parents, exist_ok=True)
        return True
    except Exception as e:
        logger.error(f"Failed to create directory {path}: {e}")
        return False


def copy_file(src: Path, dst: Path) -> bool:
    """Copy a file.

    Args:
        src: Source path
        dst: Destination path

    Returns:
        True if successful
    """
    try:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        return True
    except Exception as e:
        logger.error(f"Failed to copy {src} to {dst}: {e}")
        return False


def copy_directory(src: Path, dst: Path) -> bool:
    """Copy a directory recursively.

    Args:
        src: Source directory
        dst: Destination directory

    Returns:
        True if successful
    """
    try:
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
        return True
    except Exception as e:
        logger.error(f"Failed to copy directory {src} to {dst}: {e}")
        return False


def remove_directory(path: Path) -> bool:
    """Remove a directory and its contents.

    Args:
        path: Directory to remove

    Returns:
        True if successful
    """
    try:
        if path.exists():
            shutil.rmtree(path)
        return True
    except Exception as e:
        logger.error(f"Failed to remove directory {path}: {e}")
        return False
