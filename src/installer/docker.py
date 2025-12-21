"""Docker-based installer for Agent OS.

Provides Docker container-based installation support including:
- Docker Compose configuration generation
- Container management
- Volume and network configuration
- GPU passthrough support
"""

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base import (
    Installer,
    InstallerError,
    InstallerPhase,
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
    detect_platform,
    get_gpu_info,
)

logger = logging.getLogger(__name__)


@dataclass
class DockerConfig:
    """Docker-specific configuration."""

    image: str = "agentoshq/agent-os:latest"
    container_name: str = "agent-os"
    network_name: str = "agent-os-network"
    restart_policy: str = "unless-stopped"

    # Ports
    web_port: int = 8080
    api_port: int = 8081
    ollama_port: int = 11434

    # Volumes
    data_volume: str = "agent-os-data"
    config_volume: str = "agent-os-config"
    logs_volume: str = "agent-os-logs"

    # GPU support
    enable_gpu: bool = True
    gpu_devices: List[str] = field(default_factory=lambda: ["all"])

    # Resource limits
    memory_limit: str = "8g"
    cpu_limit: float = 4.0

    # Environment
    environment: Dict[str, str] = field(default_factory=dict)

    def to_compose_dict(self) -> Dict[str, Any]:
        """Generate Docker Compose configuration."""
        service = {
            "image": self.image,
            "container_name": self.container_name,
            "restart": self.restart_policy,
            "ports": [
                f"{self.web_port}:8080",
                f"{self.api_port}:8081",
            ],
            "volumes": [
                f"{self.data_volume}:/app/data",
                f"{self.config_volume}:/app/config",
                f"{self.logs_volume}:/app/logs",
            ],
            "environment": {
                "AGENT_OS_WEB_PORT": "8080",
                "AGENT_OS_API_PORT": "8081",
                **self.environment,
            },
            "networks": [self.network_name],
        }

        # Add GPU support if enabled
        if self.enable_gpu:
            service["deploy"] = {
                "resources": {
                    "reservations": {
                        "devices": [
                            {
                                "driver": "nvidia",
                                "count": "all",
                                "capabilities": ["gpu"],
                            }
                        ]
                    }
                }
            }

        # Add resource limits
        if "deploy" not in service:
            service["deploy"] = {}
        service["deploy"]["resources"] = service["deploy"].get("resources", {})
        service["deploy"]["resources"]["limits"] = {
            "memory": self.memory_limit,
            "cpus": str(self.cpu_limit),
        }

        return {
            "version": "3.8",
            "services": {
                "agent-os": service,
                "ollama": {
                    "image": "ollama/ollama:latest",
                    "container_name": "agent-os-ollama",
                    "restart": self.restart_policy,
                    "ports": [f"{self.ollama_port}:11434"],
                    "volumes": ["ollama-models:/root/.ollama"],
                    "networks": [self.network_name],
                    **({"deploy": service.get("deploy")} if self.enable_gpu else {}),
                },
            },
            "networks": {
                self.network_name: {
                    "driver": "bridge",
                },
            },
            "volumes": {
                self.data_volume: {},
                self.config_volume: {},
                self.logs_volume: {},
                "ollama-models": {},
            },
        }


class DockerInstaller(Installer):
    """Docker-based installer for Agent OS."""

    def __init__(
        self,
        config: InstallConfig,
        platform_info: Optional[PlatformInfo] = None,
        docker_config: Optional[DockerConfig] = None,
    ):
        """Initialize Docker installer.

        Args:
            config: Installation configuration
            platform_info: Platform information
            docker_config: Docker-specific configuration
        """
        super().__init__(config, platform_info)
        self.docker_config = docker_config or DockerConfig(
            image=config.docker_image,
            network_name=config.docker_network,
            web_port=config.docker_ports.get("web", 8080),
            api_port=config.docker_ports.get("api", 8081),
        )

    @property
    def platform(self) -> Platform:
        """Get the platform this installer supports."""
        return detect_platform()

    def check_requirements(self) -> List[RequirementCheck]:
        """Check Docker installation requirements.

        Returns:
            List of requirement check results
        """
        self._report_progress(
            InstallerPhase.REQUIREMENTS,
            0.1,
            "Checking Docker requirements...",
        )

        checks: List[RequirementCheck] = []

        # Check Docker installation
        self._report_progress(
            InstallerPhase.REQUIREMENTS,
            0.3,
            "Checking Docker installation...",
        )

        docker_version = self._get_docker_version()
        if docker_version:
            checks.append(
                RequirementCheck(
                    name="Docker",
                    status=RequirementStatus.PASSED,
                    message=f"Docker {docker_version} is installed",
                    current_value=docker_version,
                    required_value="20.0",
                )
            )
        else:
            checks.append(
                RequirementCheck(
                    name="Docker",
                    status=RequirementStatus.FAILED,
                    message="Docker is not installed or not running",
                    current_value=None,
                    required_value="20.0",
                    is_critical=True,
                )
            )

        # Check Docker Compose
        self._report_progress(
            InstallerPhase.REQUIREMENTS,
            0.5,
            "Checking Docker Compose...",
        )

        compose_version = self._get_compose_version()
        if compose_version:
            checks.append(
                RequirementCheck(
                    name="Docker Compose",
                    status=RequirementStatus.PASSED,
                    message=f"Docker Compose {compose_version} is available",
                    current_value=compose_version,
                    required_value="2.0",
                )
            )
        else:
            checks.append(
                RequirementCheck(
                    name="Docker Compose",
                    status=RequirementStatus.WARNING,
                    message="Docker Compose not found - using docker-compose fallback",
                    current_value=None,
                    required_value="2.0",
                )
            )

        # Check GPU support
        self._report_progress(
            InstallerPhase.REQUIREMENTS,
            0.7,
            "Checking GPU support...",
        )

        if self.docker_config.enable_gpu:
            nvidia_docker = self._check_nvidia_docker()
            if nvidia_docker:
                checks.append(
                    RequirementCheck(
                        name="NVIDIA Container Runtime",
                        status=RequirementStatus.PASSED,
                        message="NVIDIA Docker runtime is available",
                        current_value=True,
                        required_value=False,
                    )
                )
            else:
                checks.append(
                    RequirementCheck(
                        name="NVIDIA Container Runtime",
                        status=RequirementStatus.WARNING,
                        message="NVIDIA Docker runtime not found - GPU support disabled",
                        current_value=False,
                        required_value=False,
                    )
                )
                self.docker_config.enable_gpu = False

        # Check disk space for Docker
        self._report_progress(
            InstallerPhase.REQUIREMENTS,
            0.9,
            "Checking disk space...",
        )

        from .platform import get_disk_space_gb

        _, free_gb = get_disk_space_gb()
        if free_gb >= 30:
            checks.append(
                RequirementCheck(
                    name="Disk Space (Docker)",
                    status=RequirementStatus.PASSED,
                    message=f"{free_gb:.1f} GB available for Docker images and volumes",
                    current_value=free_gb,
                    required_value=30,
                )
            )
        elif free_gb >= 15:
            checks.append(
                RequirementCheck(
                    name="Disk Space (Docker)",
                    status=RequirementStatus.WARNING,
                    message=f"{free_gb:.1f} GB available - 30+ GB recommended",
                    current_value=free_gb,
                    required_value=30,
                )
            )
        else:
            checks.append(
                RequirementCheck(
                    name="Disk Space (Docker)",
                    status=RequirementStatus.FAILED,
                    message=f"Only {free_gb:.1f} GB available - need at least 15 GB",
                    current_value=free_gb,
                    required_value=15,
                    is_critical=True,
                )
            )

        self._report_progress(
            InstallerPhase.REQUIREMENTS,
            1.0,
            "Requirements check complete",
        )

        return checks

    def download(self) -> bool:
        """Pull Docker images.

        Returns:
            True if download successful
        """
        self._report_progress(
            InstallerPhase.DOWNLOAD,
            0.1,
            "Pulling Docker images...",
        )

        # Create installation directory
        if not create_directory(self.config.install_path):
            return False

        # Pull Agent OS image
        self._report_progress(
            InstallerPhase.DOWNLOAD,
            0.2,
            f"Pulling {self.docker_config.image}...",
        )

        code, stdout, stderr = run_command(
            ["docker", "pull", self.docker_config.image],
            timeout=1800,  # 30 minutes for large images
        )

        if code != 0:
            # Try without registry prefix
            self._add_warning(f"Failed to pull {self.docker_config.image}")
            logger.warning(f"Pull failed: {stderr}")

        # Pull Ollama image
        self._report_progress(
            InstallerPhase.DOWNLOAD,
            0.6,
            "Pulling Ollama image...",
        )

        code, stdout, stderr = run_command(
            ["docker", "pull", "ollama/ollama:latest"],
            timeout=1800,
        )

        if code != 0:
            self._add_warning("Failed to pull Ollama image")

        self._report_progress(
            InstallerPhase.DOWNLOAD,
            1.0,
            "Docker images ready",
        )

        return True

    def install(self) -> bool:
        """Install Agent OS using Docker.

        Returns:
            True if installation successful
        """
        self._report_progress(
            InstallerPhase.INSTALL,
            0.1,
            "Starting Docker installation...",
        )

        # Create Docker network
        self._report_progress(
            InstallerPhase.INSTALL,
            0.2,
            "Creating Docker network...",
        )

        if not self._create_network():
            self._add_warning("Failed to create Docker network")

        # Generate docker-compose.yml
        self._report_progress(
            InstallerPhase.INSTALL,
            0.4,
            "Generating Docker Compose configuration...",
        )

        if not self._generate_compose_file():
            return False

        # Create volumes
        self._report_progress(
            InstallerPhase.INSTALL,
            0.6,
            "Creating Docker volumes...",
        )

        self._create_volumes()

        # Start containers
        self._report_progress(
            InstallerPhase.INSTALL,
            0.8,
            "Starting containers...",
        )

        if not self._start_containers():
            return False

        self._report_progress(
            InstallerPhase.INSTALL,
            1.0,
            "Docker installation complete",
        )

        return True

    def configure(self) -> bool:
        """Configure Docker deployment.

        Returns:
            True if configuration successful
        """
        self._report_progress(
            InstallerPhase.CONFIGURE,
            0.1,
            "Configuring Docker deployment...",
        )

        # Create helper scripts
        self._report_progress(
            InstallerPhase.CONFIGURE,
            0.3,
            "Creating management scripts...",
        )

        if not self._create_management_scripts():
            self._add_warning("Failed to create management scripts")

        # Wait for containers to be ready
        self._report_progress(
            InstallerPhase.CONFIGURE,
            0.6,
            "Waiting for services to be ready...",
        )

        if not self._wait_for_services():
            self._add_warning("Services may not be fully ready")

        self._report_progress(
            InstallerPhase.CONFIGURE,
            1.0,
            "Configuration complete",
        )

        return True

    def post_install(self) -> bool:
        """Perform post-installation tasks.

        Returns:
            True if post-install successful
        """
        self._report_progress(
            InstallerPhase.POST_INSTALL,
            0.1,
            "Running post-installation tasks...",
        )

        # Download models inside container
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

        config_path = self.config.install_path / "install.json"
        self.config.save(config_path)

        self._report_progress(
            InstallerPhase.POST_INSTALL,
            1.0,
            "Post-installation complete",
        )

        return True

    def uninstall(self) -> bool:
        """Uninstall Docker deployment.

        Returns:
            True if uninstallation successful
        """
        logger.info("Starting Docker uninstallation...")

        # Stop containers
        self._stop_containers()

        # Remove containers
        self._remove_containers()

        # Remove network
        self._remove_network()

        # Optionally remove volumes (with confirmation)
        # self._remove_volumes()

        # Remove installation directory
        from .base import remove_directory

        if not remove_directory(self.config.install_path):
            logger.error("Failed to remove installation directory")
            return False

        logger.info("Docker uninstallation complete")
        return True

    # Private helper methods

    def _get_docker_version(self) -> Optional[str]:
        """Get Docker version."""
        try:
            code, stdout, stderr = run_command(
                ["docker", "--version"],
                timeout=10,
            )
            if code == 0:
                # Parse "Docker version 24.0.7, build afdd53b"
                parts = stdout.strip().split()
                if len(parts) >= 3:
                    return parts[2].rstrip(",")
        except Exception:
            pass
        return None

    def _get_compose_version(self) -> Optional[str]:
        """Get Docker Compose version."""
        try:
            # Try new "docker compose" command
            code, stdout, stderr = run_command(
                ["docker", "compose", "version", "--short"],
                timeout=10,
            )
            if code == 0:
                return stdout.strip()

            # Try old "docker-compose" command
            code, stdout, stderr = run_command(
                ["docker-compose", "--version"],
                timeout=10,
            )
            if code == 0:
                parts = stdout.strip().split()
                if len(parts) >= 3:
                    return parts[2].lstrip("v")
        except Exception:
            pass
        return None

    def _check_nvidia_docker(self) -> bool:
        """Check if NVIDIA Docker runtime is available."""
        try:
            code, stdout, stderr = run_command(
                ["docker", "info", "--format", "{{.Runtimes}}"],
                timeout=10,
            )
            if code == 0 and "nvidia" in stdout.lower():
                return True

            # Also check for GPU presence
            gpus = get_gpu_info()
            if gpus:
                # Try running a test container
                code, stdout, stderr = run_command(
                    [
                        "docker",
                        "run",
                        "--rm",
                        "--gpus",
                        "all",
                        "nvidia/cuda:12.0-base",
                        "nvidia-smi",
                    ],
                    timeout=60,
                )
                return code == 0
        except Exception:
            pass
        return False

    def _create_network(self) -> bool:
        """Create Docker network."""
        try:
            code, stdout, stderr = run_command(
                ["docker", "network", "create", self.docker_config.network_name],
                timeout=30,
            )
            return code == 0 or "already exists" in stderr
        except Exception as e:
            logger.error(f"Failed to create network: {e}")
            return False

    def _remove_network(self) -> bool:
        """Remove Docker network."""
        try:
            run_command(
                ["docker", "network", "rm", self.docker_config.network_name],
                timeout=30,
            )
            return True
        except Exception:
            return False

    def _generate_compose_file(self) -> bool:
        """Generate docker-compose.yml file."""
        try:
            import yaml

            compose_path = self.config.install_path / "docker-compose.yml"
            compose_dict = self.docker_config.to_compose_dict()

            with open(compose_path, "w") as f:
                yaml.dump(compose_dict, f, default_flow_style=False)

            return True

        except ImportError:
            # Fallback without yaml
            compose_path = self.config.install_path / "docker-compose.yml"
            compose_dict = self.docker_config.to_compose_dict()

            # Simple JSON to YAML-ish conversion
            compose_content = json.dumps(compose_dict, indent=2)
            compose_path.write_text(compose_content)
            return True

        except Exception as e:
            logger.error(f"Failed to generate compose file: {e}")
            return False

    def _create_volumes(self) -> None:
        """Create Docker volumes."""
        volumes = [
            self.docker_config.data_volume,
            self.docker_config.config_volume,
            self.docker_config.logs_volume,
            "ollama-models",
        ]

        for volume in volumes:
            run_command(
                ["docker", "volume", "create", volume],
                timeout=30,
            )

    def _remove_volumes(self) -> None:
        """Remove Docker volumes."""
        volumes = [
            self.docker_config.data_volume,
            self.docker_config.config_volume,
            self.docker_config.logs_volume,
            "ollama-models",
        ]

        for volume in volumes:
            run_command(
                ["docker", "volume", "rm", volume],
                timeout=30,
            )

    def _start_containers(self) -> bool:
        """Start Docker containers using Compose."""
        try:
            compose_path = self.config.install_path / "docker-compose.yml"

            # Try new docker compose
            code, stdout, stderr = run_command(
                ["docker", "compose", "-f", str(compose_path), "up", "-d"],
                cwd=self.config.install_path,
                timeout=300,
            )

            if code == 0:
                return True

            # Try old docker-compose
            code, stdout, stderr = run_command(
                ["docker-compose", "-f", str(compose_path), "up", "-d"],
                cwd=self.config.install_path,
                timeout=300,
            )

            return code == 0

        except Exception as e:
            logger.error(f"Failed to start containers: {e}")
            return False

    def _stop_containers(self) -> bool:
        """Stop Docker containers."""
        try:
            compose_path = self.config.install_path / "docker-compose.yml"

            if compose_path.exists():
                run_command(
                    ["docker", "compose", "-f", str(compose_path), "down"],
                    cwd=self.config.install_path,
                    timeout=120,
                )

            return True

        except Exception as e:
            logger.error(f"Failed to stop containers: {e}")
            return False

    def _remove_containers(self) -> bool:
        """Remove Docker containers."""
        try:
            containers = [
                self.docker_config.container_name,
                "agent-os-ollama",
            ]

            for container in containers:
                run_command(
                    ["docker", "rm", "-f", container],
                    timeout=30,
                )

            return True

        except Exception as e:
            logger.error(f"Failed to remove containers: {e}")
            return False

    def _create_management_scripts(self) -> bool:
        """Create helper scripts for managing the deployment."""
        try:
            bin_dir = self.config.install_path
            compose_path = self.config.install_path / "docker-compose.yml"

            # Start script
            start_script = f"""#!/bin/bash
cd "{self.config.install_path}"
docker compose -f "{compose_path}" up -d
echo "Agent OS started. Access at http://localhost:{self.docker_config.web_port}"
"""
            start_path = bin_dir / "start.sh"
            start_path.write_text(start_script)
            start_path.chmod(0o755)

            # Stop script
            stop_script = f"""#!/bin/bash
cd "{self.config.install_path}"
docker compose -f "{compose_path}" down
echo "Agent OS stopped."
"""
            stop_path = bin_dir / "stop.sh"
            stop_path.write_text(stop_script)
            stop_path.chmod(0o755)

            # Logs script
            logs_script = f"""#!/bin/bash
cd "{self.config.install_path}"
docker compose -f "{compose_path}" logs -f
"""
            logs_path = bin_dir / "logs.sh"
            logs_path.write_text(logs_script)
            logs_path.chmod(0o755)

            # Status script
            status_script = f"""#!/bin/bash
cd "{self.config.install_path}"
docker compose -f "{compose_path}" ps
"""
            status_path = bin_dir / "status.sh"
            status_path.write_text(status_script)
            status_path.chmod(0o755)

            return True

        except Exception as e:
            logger.error(f"Failed to create management scripts: {e}")
            return False

    def _wait_for_services(self, timeout: int = 60) -> bool:
        """Wait for services to be ready."""
        import time

        start_time = time.time()

        while time.time() - start_time < timeout:
            # Check if Agent OS container is running
            code, stdout, stderr = run_command(
                [
                    "docker",
                    "inspect",
                    "-f",
                    "{{.State.Running}}",
                    self.docker_config.container_name,
                ],
                timeout=10,
            )

            if code == 0 and "true" in stdout.lower():
                # Check if web port is responding
                code, stdout, stderr = run_command(
                    [
                        "curl",
                        "-s",
                        "-o",
                        "/dev/null",
                        "-w",
                        "%{http_code}",
                        f"http://localhost:{self.docker_config.web_port}/health",
                    ],
                    timeout=5,
                )

                if code == 0 and stdout.strip() in ("200", "404"):
                    return True

            time.sleep(2)

        return False

    def _download_model(self, model_name: str) -> bool:
        """Download an Ollama model inside the container."""
        try:
            code, stdout, stderr = run_command(
                [
                    "docker",
                    "exec",
                    "agent-os-ollama",
                    "ollama",
                    "pull",
                    model_name,
                ],
                timeout=3600,
            )
            return code == 0
        except Exception as e:
            logger.error(f"Failed to download model {model_name}: {e}")
            return False


def create_docker_installer(
    config: InstallConfig,
    platform_info: Optional[PlatformInfo] = None,
    docker_config: Optional[DockerConfig] = None,
) -> DockerInstaller:
    """Create a Docker installer.

    Args:
        config: Installation configuration
        platform_info: Platform information
        docker_config: Docker-specific configuration

    Returns:
        DockerInstaller instance
    """
    return DockerInstaller(config, platform_info, docker_config)
