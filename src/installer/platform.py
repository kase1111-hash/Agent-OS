"""Platform detection and system requirements checking.

Provides utilities for:
- Detecting the current operating system
- Checking system requirements (CPU, RAM, disk, GPU)
- Validating dependencies
"""

import logging
import os
import platform
import shutil
import subprocess
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class Platform(str, Enum):
    """Supported platforms."""

    WINDOWS = "windows"
    MACOS = "macos"
    LINUX = "linux"
    UNKNOWN = "unknown"


class Architecture(str, Enum):
    """CPU architectures."""

    X86_64 = "x86_64"
    ARM64 = "arm64"
    X86 = "x86"
    UNKNOWN = "unknown"


class RequirementStatus(str, Enum):
    """Status of a requirement check."""

    PASSED = "passed"
    WARNING = "warning"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class RequirementCheck:
    """Result of a single requirement check."""

    name: str
    status: RequirementStatus
    message: str
    current_value: Any = None
    required_value: Any = None
    is_critical: bool = False

    @property
    def passed(self) -> bool:
        """Check if requirement passed."""
        return self.status in (RequirementStatus.PASSED, RequirementStatus.WARNING)


@dataclass
class PlatformInfo:
    """Information about the current platform."""

    platform: Platform
    architecture: Architecture
    os_name: str
    os_version: str
    os_release: str
    hostname: str
    username: str
    home_dir: Path
    is_admin: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "platform": self.platform.value,
            "architecture": self.architecture.value,
            "os_name": self.os_name,
            "os_version": self.os_version,
            "os_release": self.os_release,
            "hostname": self.hostname,
            "username": self.username,
            "home_dir": str(self.home_dir),
            "is_admin": self.is_admin,
            "metadata": self.metadata,
        }


@dataclass
class SystemRequirements:
    """Minimum system requirements for Agent OS."""

    min_ram_gb: float = 8.0
    recommended_ram_gb: float = 16.0
    min_disk_gb: float = 20.0
    recommended_disk_gb: float = 50.0
    min_cpu_cores: int = 4
    recommended_cpu_cores: int = 8
    min_vram_gb: float = 0.0  # 0 means no GPU required
    recommended_vram_gb: float = 8.0
    required_commands: List[str] = field(default_factory=lambda: ["python3", "git"])
    optional_commands: List[str] = field(default_factory=lambda: ["docker", "nvidia-smi", "ollama"])


def detect_platform() -> Platform:
    """Detect the current operating system.

    Returns:
        Platform enum value
    """
    system = platform.system().lower()

    if system == "windows":
        return Platform.WINDOWS
    elif system == "darwin":
        return Platform.MACOS
    elif system == "linux":
        return Platform.LINUX
    else:
        return Platform.UNKNOWN


def detect_architecture() -> Architecture:
    """Detect CPU architecture.

    Returns:
        Architecture enum value
    """
    machine = platform.machine().lower()

    if machine in ("x86_64", "amd64"):
        return Architecture.X86_64
    elif machine in ("arm64", "aarch64"):
        return Architecture.ARM64
    elif machine in ("i386", "i686", "x86"):
        return Architecture.X86
    else:
        return Architecture.UNKNOWN


def _check_admin() -> bool:
    """Check if running with admin/root privileges."""
    try:
        if platform.system() == "Windows":
            import ctypes

            return ctypes.windll.shell32.IsUserAnAdmin() != 0
        else:
            return os.geteuid() == 0
    except Exception:
        return False


def get_system_info() -> PlatformInfo:
    """Get comprehensive system information.

    Returns:
        PlatformInfo with current system details
    """
    plat = detect_platform()
    arch = detect_architecture()

    # Get OS-specific info
    os_name = platform.system()
    os_version = platform.version()
    os_release = platform.release()

    # User info
    try:
        username = os.getlogin()
    except OSError:
        username = os.environ.get("USER", os.environ.get("USERNAME", "unknown"))

    hostname = platform.node()
    home_dir = Path.home()
    is_admin = _check_admin()

    # Gather metadata
    metadata: Dict[str, Any] = {
        "python_version": platform.python_version(),
        "processor": platform.processor(),
        "machine": platform.machine(),
    }

    # Platform-specific metadata
    if plat == Platform.LINUX:
        try:
            import distro

            metadata["distro_name"] = distro.name()
            metadata["distro_version"] = distro.version()
            metadata["distro_id"] = distro.id()
        except ImportError:
            # Try to read from /etc/os-release
            os_release_file = Path("/etc/os-release")
            if os_release_file.exists():
                with open(os_release_file) as f:
                    for line in f:
                        if "=" in line:
                            key, value = line.strip().split("=", 1)
                            metadata[f"distro_{key.lower()}"] = value.strip('"')

    return PlatformInfo(
        platform=plat,
        architecture=arch,
        os_name=os_name,
        os_version=os_version,
        os_release=os_release,
        hostname=hostname,
        username=username,
        home_dir=home_dir,
        is_admin=is_admin,
        metadata=metadata,
    )


def get_ram_gb() -> float:
    """Get total system RAM in GB.

    Returns:
        RAM size in gigabytes
    """
    try:
        import psutil

        return psutil.virtual_memory().total / (1024**3)
    except ImportError:
        # Fallback for systems without psutil
        plat = detect_platform()

        if plat == Platform.LINUX:
            try:
                with open("/proc/meminfo") as f:
                    for line in f:
                        if line.startswith("MemTotal:"):
                            kb = int(line.split()[1])
                            return kb / (1024**2)
            except Exception:
                pass
        elif plat == Platform.MACOS:
            try:
                result = subprocess.run(
                    ["sysctl", "-n", "hw.memsize"],
                    capture_output=True,
                    text=True,
                )
                if result.returncode == 0:
                    return int(result.stdout.strip()) / (1024**3)
            except Exception:
                pass
        elif plat == Platform.WINDOWS:
            try:
                result = subprocess.run(
                    ["wmic", "os", "get", "TotalVisibleMemorySize"],
                    capture_output=True,
                    text=True,
                )
                if result.returncode == 0:
                    lines = result.stdout.strip().split("\n")
                    if len(lines) > 1:
                        kb = int(lines[1].strip())
                        return kb / (1024**2)
            except Exception:
                pass

        return 0.0


def get_disk_space_gb(path: Optional[Path] = None) -> Tuple[float, float]:
    """Get disk space for installation path.

    Args:
        path: Path to check (defaults to home directory)

    Returns:
        Tuple of (total_gb, free_gb)
    """
    if path is None:
        path = Path.home()

    try:
        import shutil

        usage = shutil.disk_usage(path)
        total_gb = usage.total / (1024**3)
        free_gb = usage.free / (1024**3)
        return total_gb, free_gb
    except Exception:
        return 0.0, 0.0


def get_cpu_cores() -> int:
    """Get number of CPU cores.

    Returns:
        Number of logical CPU cores
    """
    try:
        return os.cpu_count() or 1
    except Exception:
        return 1


def get_gpu_info() -> List[Dict[str, Any]]:
    """Get GPU information.

    Returns:
        List of GPU info dictionaries
    """
    gpus = []

    # Try nvidia-smi for NVIDIA GPUs
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            for line in result.stdout.strip().split("\n"):
                if line.strip():
                    parts = [p.strip() for p in line.split(",")]
                    if len(parts) >= 3:
                        gpus.append(
                            {
                                "name": parts[0],
                                "vram_mb": float(parts[1]),
                                "driver": parts[2],
                                "vendor": "nvidia",
                            }
                        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # Try AMD ROCm
    try:
        result = subprocess.run(
            ["rocm-smi", "--showmeminfo", "vram"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            # Parse ROCm output (simplified)
            gpus.append({"name": "AMD GPU", "vendor": "amd", "rocm": True})
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    return gpus


def check_command_exists(command: str) -> bool:
    """Check if a command is available in PATH.

    Args:
        command: Command name to check

    Returns:
        True if command exists
    """
    return shutil.which(command) is not None


def check_requirements(
    requirements: Optional[SystemRequirements] = None,
    install_path: Optional[Path] = None,
) -> List[RequirementCheck]:
    """Check if system meets installation requirements.

    Args:
        requirements: Requirements to check (uses defaults if None)
        install_path: Installation path to check disk space

    Returns:
        List of requirement check results
    """
    if requirements is None:
        requirements = SystemRequirements()

    checks: List[RequirementCheck] = []

    # Check RAM
    ram_gb = get_ram_gb()
    if ram_gb >= requirements.recommended_ram_gb:
        checks.append(
            RequirementCheck(
                name="RAM",
                status=RequirementStatus.PASSED,
                message=f"System has {ram_gb:.1f} GB RAM (recommended: {requirements.recommended_ram_gb} GB)",
                current_value=ram_gb,
                required_value=requirements.recommended_ram_gb,
            )
        )
    elif ram_gb >= requirements.min_ram_gb:
        checks.append(
            RequirementCheck(
                name="RAM",
                status=RequirementStatus.WARNING,
                message=f"System has {ram_gb:.1f} GB RAM (minimum: {requirements.min_ram_gb} GB)",
                current_value=ram_gb,
                required_value=requirements.min_ram_gb,
            )
        )
    else:
        checks.append(
            RequirementCheck(
                name="RAM",
                status=RequirementStatus.FAILED,
                message=f"System has {ram_gb:.1f} GB RAM (required: {requirements.min_ram_gb} GB)",
                current_value=ram_gb,
                required_value=requirements.min_ram_gb,
                is_critical=True,
            )
        )

    # Check disk space
    _, free_gb = get_disk_space_gb(install_path)
    if free_gb >= requirements.recommended_disk_gb:
        checks.append(
            RequirementCheck(
                name="Disk Space",
                status=RequirementStatus.PASSED,
                message=f"Available disk space: {free_gb:.1f} GB",
                current_value=free_gb,
                required_value=requirements.recommended_disk_gb,
            )
        )
    elif free_gb >= requirements.min_disk_gb:
        checks.append(
            RequirementCheck(
                name="Disk Space",
                status=RequirementStatus.WARNING,
                message=f"Available disk space: {free_gb:.1f} GB (recommended: {requirements.recommended_disk_gb} GB)",
                current_value=free_gb,
                required_value=requirements.min_disk_gb,
            )
        )
    else:
        checks.append(
            RequirementCheck(
                name="Disk Space",
                status=RequirementStatus.FAILED,
                message=f"Insufficient disk space: {free_gb:.1f} GB (required: {requirements.min_disk_gb} GB)",
                current_value=free_gb,
                required_value=requirements.min_disk_gb,
                is_critical=True,
            )
        )

    # Check CPU cores
    cores = get_cpu_cores()
    if cores >= requirements.recommended_cpu_cores:
        checks.append(
            RequirementCheck(
                name="CPU Cores",
                status=RequirementStatus.PASSED,
                message=f"System has {cores} CPU cores",
                current_value=cores,
                required_value=requirements.recommended_cpu_cores,
            )
        )
    elif cores >= requirements.min_cpu_cores:
        checks.append(
            RequirementCheck(
                name="CPU Cores",
                status=RequirementStatus.WARNING,
                message=f"System has {cores} CPU cores (recommended: {requirements.recommended_cpu_cores})",
                current_value=cores,
                required_value=requirements.min_cpu_cores,
            )
        )
    else:
        checks.append(
            RequirementCheck(
                name="CPU Cores",
                status=RequirementStatus.FAILED,
                message=f"System has {cores} CPU cores (required: {requirements.min_cpu_cores})",
                current_value=cores,
                required_value=requirements.min_cpu_cores,
                is_critical=True,
            )
        )

    # Check GPU (optional)
    gpus = get_gpu_info()
    if gpus:
        max_vram = max(g.get("vram_mb", 0) for g in gpus) / 1024  # Convert to GB
        if max_vram >= requirements.recommended_vram_gb:
            checks.append(
                RequirementCheck(
                    name="GPU",
                    status=RequirementStatus.PASSED,
                    message=f"Found GPU with {max_vram:.1f} GB VRAM",
                    current_value=max_vram,
                    required_value=requirements.recommended_vram_gb,
                )
            )
        else:
            checks.append(
                RequirementCheck(
                    name="GPU",
                    status=RequirementStatus.WARNING,
                    message=f"GPU has {max_vram:.1f} GB VRAM (recommended: {requirements.recommended_vram_gb} GB)",
                    current_value=max_vram,
                    required_value=requirements.recommended_vram_gb,
                )
            )
    else:
        checks.append(
            RequirementCheck(
                name="GPU",
                status=RequirementStatus.WARNING,
                message="No GPU detected. Agent OS will run in CPU mode.",
                current_value=None,
                required_value=requirements.recommended_vram_gb,
            )
        )

    # Check required commands
    for cmd in requirements.required_commands:
        if check_command_exists(cmd):
            checks.append(
                RequirementCheck(
                    name=f"Command: {cmd}",
                    status=RequirementStatus.PASSED,
                    message=f"{cmd} is available",
                    current_value=True,
                    required_value=True,
                )
            )
        else:
            checks.append(
                RequirementCheck(
                    name=f"Command: {cmd}",
                    status=RequirementStatus.FAILED,
                    message=f"{cmd} is required but not found",
                    current_value=False,
                    required_value=True,
                    is_critical=True,
                )
            )

    # Check optional commands
    for cmd in requirements.optional_commands:
        if check_command_exists(cmd):
            checks.append(
                RequirementCheck(
                    name=f"Optional: {cmd}",
                    status=RequirementStatus.PASSED,
                    message=f"{cmd} is available",
                    current_value=True,
                    required_value=False,
                )
            )
        else:
            checks.append(
                RequirementCheck(
                    name=f"Optional: {cmd}",
                    status=RequirementStatus.SKIPPED,
                    message=f"{cmd} not found (optional)",
                    current_value=False,
                    required_value=False,
                )
            )

    return checks


def all_requirements_met(checks: List[RequirementCheck]) -> bool:
    """Check if all critical requirements are met.

    Args:
        checks: List of requirement checks

    Returns:
        True if all critical checks passed
    """
    for check in checks:
        if check.is_critical and not check.passed:
            return False
    return True
