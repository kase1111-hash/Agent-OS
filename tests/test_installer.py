"""Tests for one-click installer module."""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch


class TestPlatformModule:
    """Tests for platform detection."""

    def test_platform_enum(self):
        """Test Platform enum values."""
        from src.installer.platform import Platform

        assert Platform.WINDOWS.value == "windows"
        assert Platform.MACOS.value == "macos"
        assert Platform.LINUX.value == "linux"
        assert Platform.UNKNOWN.value == "unknown"

    def test_architecture_enum(self):
        """Test Architecture enum values."""
        from src.installer.platform import Architecture

        assert Architecture.X86_64.value == "x86_64"
        assert Architecture.ARM64.value == "arm64"
        assert Architecture.X86.value == "x86"

    def test_requirement_status_enum(self):
        """Test RequirementStatus enum values."""
        from src.installer.platform import RequirementStatus

        assert RequirementStatus.PASSED.value == "passed"
        assert RequirementStatus.WARNING.value == "warning"
        assert RequirementStatus.FAILED.value == "failed"
        assert RequirementStatus.SKIPPED.value == "skipped"

    def test_requirement_check_creation(self):
        """Test RequirementCheck dataclass creation."""
        from src.installer.platform import RequirementCheck, RequirementStatus

        check = RequirementCheck(
            name="Test Check",
            status=RequirementStatus.PASSED,
            message="Test passed",
            current_value=10,
            required_value=5,
        )

        assert check.name == "Test Check"
        assert check.status == RequirementStatus.PASSED
        assert check.passed is True

    def test_requirement_check_failed(self):
        """Test RequirementCheck failed status."""
        from src.installer.platform import RequirementCheck, RequirementStatus

        check = RequirementCheck(
            name="Test Check",
            status=RequirementStatus.FAILED,
            message="Test failed",
            is_critical=True,
        )

        assert check.passed is False
        assert check.is_critical is True

    def test_platform_info_creation(self):
        """Test PlatformInfo dataclass creation."""
        from src.installer.platform import PlatformInfo, Platform, Architecture

        info = PlatformInfo(
            platform=Platform.LINUX,
            architecture=Architecture.X86_64,
            os_name="Linux",
            os_version="5.15.0",
            os_release="5.15",
            hostname="testhost",
            username="testuser",
            home_dir=Path.home(),
        )

        assert info.platform == Platform.LINUX
        assert info.architecture == Architecture.X86_64
        assert info.os_name == "Linux"

    def test_platform_info_to_dict(self):
        """Test PlatformInfo serialization."""
        from src.installer.platform import PlatformInfo, Platform, Architecture

        info = PlatformInfo(
            platform=Platform.MACOS,
            architecture=Architecture.ARM64,
            os_name="Darwin",
            os_version="23.0.0",
            os_release="14.0",
            hostname="macbook",
            username="user",
            home_dir=Path("/Users/user"),
        )

        data = info.to_dict()
        assert data["platform"] == "macos"
        assert data["architecture"] == "arm64"

    def test_system_requirements_defaults(self):
        """Test SystemRequirements default values."""
        from src.installer.platform import SystemRequirements

        req = SystemRequirements()

        assert req.min_ram_gb == 8.0
        assert req.recommended_ram_gb == 16.0
        assert req.min_disk_gb == 20.0
        assert req.min_cpu_cores == 4
        assert "python3" in req.required_commands or "python" in req.required_commands

    def test_detect_platform(self):
        """Test platform detection."""
        from src.installer.platform import detect_platform, Platform

        platform = detect_platform()
        assert platform in (Platform.WINDOWS, Platform.MACOS, Platform.LINUX, Platform.UNKNOWN)

    def test_detect_architecture(self):
        """Test architecture detection."""
        from src.installer.platform import detect_architecture, Architecture

        arch = detect_architecture()
        assert arch in (
            Architecture.X86_64,
            Architecture.ARM64,
            Architecture.X86,
            Architecture.UNKNOWN,
        )

    def test_get_system_info(self):
        """Test getting system information."""
        from src.installer.platform import get_system_info, Platform

        info = get_system_info()

        assert info.platform in Platform
        assert info.hostname is not None
        assert info.username is not None
        assert info.home_dir.exists()

    def test_get_ram_gb(self):
        """Test getting RAM size."""
        from src.installer.platform import get_ram_gb

        ram = get_ram_gb()
        assert ram >= 0  # Could be 0 if detection fails

    def test_get_disk_space_gb(self):
        """Test getting disk space."""
        from src.installer.platform import get_disk_space_gb

        total, free = get_disk_space_gb()
        assert total >= 0
        assert free >= 0
        assert free <= total

    def test_get_cpu_cores(self):
        """Test getting CPU cores."""
        from src.installer.platform import get_cpu_cores

        cores = get_cpu_cores()
        assert cores >= 1

    def test_check_command_exists(self):
        """Test command existence check."""
        from src.installer.platform import check_command_exists

        # Python should exist
        assert check_command_exists("python3") or check_command_exists("python")

        # Non-existent command
        assert check_command_exists("nonexistent_command_xyz") is False

    def test_check_requirements(self):
        """Test requirements checking."""
        from src.installer.platform import check_requirements, RequirementStatus

        checks = check_requirements()

        assert len(checks) > 0

        # Should have RAM, disk, CPU checks
        check_names = [c.name for c in checks]
        assert any("RAM" in name for name in check_names)
        assert any("Disk" in name for name in check_names)
        assert any("CPU" in name for name in check_names)

    def test_all_requirements_met(self):
        """Test all_requirements_met helper."""
        from src.installer.platform import (
            all_requirements_met,
            RequirementCheck,
            RequirementStatus,
        )

        # All passed
        passed_checks = [
            RequirementCheck("Test1", RequirementStatus.PASSED, "OK"),
            RequirementCheck("Test2", RequirementStatus.WARNING, "Warning"),
        ]
        assert all_requirements_met(passed_checks) is True

        # One critical failed
        failed_checks = [
            RequirementCheck("Test1", RequirementStatus.PASSED, "OK"),
            RequirementCheck("Test2", RequirementStatus.FAILED, "Failed", is_critical=True),
        ]
        assert all_requirements_met(failed_checks) is False


class TestConfigModule:
    """Tests for installation configuration."""

    def test_install_mode_enum(self):
        """Test InstallMode enum values."""
        from src.installer.config import InstallMode

        assert InstallMode.FULL.value == "full"
        assert InstallMode.MINIMAL.value == "minimal"
        assert InstallMode.DOCKER.value == "docker"
        assert InstallMode.CUSTOM.value == "custom"

    def test_install_location_enum(self):
        """Test InstallLocation enum values."""
        from src.installer.config import InstallLocation

        assert InstallLocation.SYSTEM.value == "system"
        assert InstallLocation.USER.value == "user"
        assert InstallLocation.CUSTOM.value == "custom"

    def test_component_selection_defaults(self):
        """Test ComponentSelection default values."""
        from src.installer.config import ComponentSelection

        components = ComponentSelection()

        assert components.core is True
        assert components.agents is True
        assert components.memory is True
        assert components.voice is False  # Optional

    def test_component_selection_to_list(self):
        """Test ComponentSelection to_list."""
        from src.installer.config import ComponentSelection

        components = ComponentSelection(core=True, agents=True, memory=False)
        lst = components.to_list()

        assert "core" in lst
        assert "agents" in lst
        assert "memory" not in lst

    def test_component_selection_from_list(self):
        """Test ComponentSelection from_list."""
        from src.installer.config import ComponentSelection

        lst = ["core", "agents", "web_ui"]
        components = ComponentSelection.from_list(lst)

        assert components.core is True
        assert components.agents is True
        assert components.web_ui is True
        assert components.memory is False

    def test_component_selection_full(self):
        """Test ComponentSelection.full factory."""
        from src.installer.config import ComponentSelection

        components = ComponentSelection.full()

        assert components.core is True
        assert components.voice is True
        assert components.multimodal is True
        assert components.sdk is True

    def test_component_selection_minimal(self):
        """Test ComponentSelection.minimal factory."""
        from src.installer.config import ComponentSelection

        components = ComponentSelection.minimal()

        assert components.core is True
        assert components.agents is True
        assert components.memory is False
        assert components.web_ui is False

    def test_install_config_creation(self):
        """Test InstallConfig creation."""
        from src.installer.config import (
            InstallConfig,
            InstallLocation,
            InstallMode,
            ComponentSelection,
        )

        config = InstallConfig(
            install_path=Path("/tmp/test"),
            install_location=InstallLocation.USER,
            install_mode=InstallMode.FULL,
            components=ComponentSelection(),
        )

        assert config.install_path == Path("/tmp/test")
        assert config.install_mode == InstallMode.FULL

    def test_install_config_post_init(self):
        """Test InstallConfig post_init."""
        from src.installer.config import (
            InstallConfig,
            InstallLocation,
            InstallMode,
            ComponentSelection,
        )

        config = InstallConfig(
            install_path="/tmp/test",  # String, not Path
            install_location=InstallLocation.USER,
            install_mode=InstallMode.FULL,
            components=ComponentSelection(),
        )

        # Should be converted to Path
        assert isinstance(config.install_path, Path)
        # Default paths should be set
        assert config.data_path is not None
        assert config.log_path is not None
        assert config.config_path is not None

    def test_install_config_to_dict(self):
        """Test InstallConfig serialization."""
        from src.installer.config import (
            InstallConfig,
            InstallLocation,
            InstallMode,
            ComponentSelection,
        )

        config = InstallConfig(
            install_path=Path("/tmp/test"),
            install_location=InstallLocation.USER,
            install_mode=InstallMode.FULL,
            components=ComponentSelection(),
        )

        data = config.to_dict()
        assert data["install_path"] == "/tmp/test"
        assert data["install_mode"] == "full"
        assert "components" in data

    def test_install_config_from_dict(self):
        """Test InstallConfig deserialization."""
        from src.installer.config import InstallConfig, InstallLocation, InstallMode

        data = {
            "install_path": "/tmp/test",
            "install_location": "user",
            "install_mode": "minimal",
            "components": ["core", "agents"],
        }

        config = InstallConfig.from_dict(data)
        assert config.install_path == Path("/tmp/test")
        assert config.install_mode == InstallMode.MINIMAL

    def test_install_config_save_load(self):
        """Test InstallConfig save and load."""
        from src.installer.config import (
            InstallConfig,
            InstallLocation,
            InstallMode,
            ComponentSelection,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            config = InstallConfig(
                install_path=Path(tmpdir),
                install_location=InstallLocation.USER,
                install_mode=InstallMode.FULL,
                components=ComponentSelection(),
            )

            config_file = Path(tmpdir) / "install.json"
            config.save(config_file)

            loaded = InstallConfig.load(config_file)
            assert loaded.install_path == config.install_path
            assert loaded.install_mode == config.install_mode

    def test_get_default_install_path(self):
        """Test get_default_install_path."""
        from src.installer.config import get_default_install_path, InstallLocation

        user_path = get_default_install_path(InstallLocation.USER)
        assert user_path is not None
        assert isinstance(user_path, Path)

        system_path = get_default_install_path(InstallLocation.SYSTEM)
        assert system_path is not None

    def test_create_install_config(self):
        """Test create_install_config factory."""
        from src.installer.config import (
            create_install_config,
            InstallMode,
            InstallLocation,
        )

        config = create_install_config(
            mode=InstallMode.FULL,
            location=InstallLocation.USER,
        )

        assert config.install_mode == InstallMode.FULL
        assert config.install_location == InstallLocation.USER
        assert config.install_path is not None


class TestBaseModule:
    """Tests for base installer functionality."""

    def test_installer_phase_enum(self):
        """Test InstallerPhase enum values."""
        from src.installer.base import InstallerPhase

        assert InstallerPhase.INIT.value == "init"
        assert InstallerPhase.REQUIREMENTS.value == "requirements"
        assert InstallerPhase.DOWNLOAD.value == "download"
        assert InstallerPhase.INSTALL.value == "install"
        assert InstallerPhase.COMPLETE.value == "complete"
        assert InstallerPhase.FAILED.value == "failed"

    def test_installer_error_creation(self):
        """Test InstallerError exception."""
        from src.installer.base import InstallerError, InstallerPhase

        error = InstallerError(
            message="Test error",
            phase=InstallerPhase.INSTALL,
            recoverable=True,
        )

        assert str(error) == "Test error"
        assert error.phase == InstallerPhase.INSTALL
        assert error.recoverable is True

    def test_installer_progress_creation(self):
        """Test InstallerProgress dataclass."""
        from src.installer.base import InstallerProgress, InstallerPhase

        progress = InstallerProgress(
            phase=InstallerPhase.DOWNLOAD,
            progress=0.5,
            message="Downloading...",
            current_step=5,
            total_steps=10,
        )

        assert progress.phase == InstallerPhase.DOWNLOAD
        assert progress.progress == 0.5
        assert progress.percent == 50

    def test_installer_result_creation(self):
        """Test InstallerResult dataclass."""
        from src.installer.base import InstallerResult, InstallerPhase
        from src.installer.config import (
            InstallConfig,
            InstallLocation,
            InstallMode,
            ComponentSelection,
        )

        config = InstallConfig(
            install_path=Path("/tmp/test"),
            install_location=InstallLocation.USER,
            install_mode=InstallMode.FULL,
            components=ComponentSelection(),
        )

        result = InstallerResult(
            success=True,
            config=config,
            install_path=Path("/tmp/test"),
            duration_seconds=120.5,
            phases_completed=[InstallerPhase.DOWNLOAD, InstallerPhase.INSTALL],
        )

        assert result.success is True
        assert len(result.phases_completed) == 2

    def test_installer_result_to_dict(self):
        """Test InstallerResult serialization."""
        from src.installer.base import InstallerResult, InstallerPhase
        from src.installer.config import (
            InstallConfig,
            InstallLocation,
            InstallMode,
            ComponentSelection,
        )

        config = InstallConfig(
            install_path=Path("/tmp/test"),
            install_location=InstallLocation.USER,
            install_mode=InstallMode.FULL,
            components=ComponentSelection(),
        )

        result = InstallerResult(
            success=True,
            config=config,
            install_path=Path("/tmp/test"),
            duration_seconds=60.0,
            phases_completed=[InstallerPhase.COMPLETE],
        )

        data = result.to_dict()
        assert data["success"] is True
        assert "install_path" in data

    def test_run_command(self):
        """Test run_command utility."""
        from src.installer.base import run_command

        # Test with echo (works on all platforms)
        code, stdout, stderr = run_command(["echo", "hello"], timeout=10)
        assert code == 0
        assert "hello" in stdout

    def test_run_command_timeout(self):
        """Test run_command timeout."""
        from src.installer.base import run_command

        # This should timeout quickly
        code, stdout, stderr = run_command(
            ["sleep", "10"],
            timeout=1,
        )
        assert code == -1
        assert "timed out" in stderr.lower() or code == -1

    def test_create_directory(self):
        """Test create_directory utility."""
        from src.installer.base import create_directory

        with tempfile.TemporaryDirectory() as tmpdir:
            test_dir = Path(tmpdir) / "test" / "nested"
            result = create_directory(test_dir)

            assert result is True
            assert test_dir.exists()

    def test_copy_file(self):
        """Test copy_file utility."""
        from src.installer.base import copy_file

        with tempfile.TemporaryDirectory() as tmpdir:
            src = Path(tmpdir) / "source.txt"
            dst = Path(tmpdir) / "dest" / "copy.txt"

            src.write_text("test content")
            result = copy_file(src, dst)

            assert result is True
            assert dst.exists()
            assert dst.read_text() == "test content"

    def test_remove_directory(self):
        """Test remove_directory utility."""
        from src.installer.base import remove_directory, create_directory

        with tempfile.TemporaryDirectory() as tmpdir:
            test_dir = Path(tmpdir) / "to_remove"
            create_directory(test_dir)
            (test_dir / "file.txt").write_text("content")

            result = remove_directory(test_dir)

            assert result is True
            assert not test_dir.exists()


class TestLinuxInstaller:
    """Tests for Linux installer."""

    def test_linux_distro_enum(self):
        """Test LinuxDistro enum values."""
        from src.installer.linux import LinuxDistro

        assert LinuxDistro.DEBIAN.value == "debian"
        assert LinuxDistro.REDHAT.value == "redhat"
        assert LinuxDistro.ARCH.value == "arch"
        assert LinuxDistro.GENERIC.value == "generic"

    def test_linux_installer_creation(self):
        """Test LinuxInstaller creation."""
        from src.installer.linux import LinuxInstaller, create_linux_installer
        from src.installer.config import (
            InstallConfig,
            InstallLocation,
            InstallMode,
            ComponentSelection,
        )
        from src.installer.platform import Platform

        config = InstallConfig(
            install_path=Path("/tmp/test"),
            install_location=InstallLocation.USER,
            install_mode=InstallMode.FULL,
            components=ComponentSelection(),
        )

        installer = create_linux_installer(config)
        assert installer.platform == Platform.LINUX

    def test_linux_installer_check_requirements(self):
        """Test Linux requirements checking."""
        from src.installer.linux import create_linux_installer
        from src.installer.config import (
            InstallConfig,
            InstallLocation,
            InstallMode,
            ComponentSelection,
        )

        config = InstallConfig(
            install_path=Path("/tmp/test"),
            install_location=InstallLocation.USER,
            install_mode=InstallMode.FULL,
            components=ComponentSelection(),
        )

        installer = create_linux_installer(config)
        checks = installer.check_requirements()

        assert len(checks) > 0


class TestDockerInstaller:
    """Tests for Docker installer."""

    def test_docker_config_creation(self):
        """Test DockerConfig creation."""
        from src.installer.docker import DockerConfig

        config = DockerConfig(
            image="test/image:latest",
            container_name="test-container",
            web_port=9090,
        )

        assert config.image == "test/image:latest"
        assert config.container_name == "test-container"
        assert config.web_port == 9090

    def test_docker_config_to_compose_dict(self):
        """Test DockerConfig to_compose_dict."""
        from src.installer.docker import DockerConfig

        config = DockerConfig(
            image="agentoshq/agent-os:latest",
            enable_gpu=False,
        )

        compose = config.to_compose_dict()

        assert "version" in compose
        assert "services" in compose
        assert "agent-os" in compose["services"]
        assert "ollama" in compose["services"]
        assert "networks" in compose
        assert "volumes" in compose

    def test_docker_installer_creation(self):
        """Test DockerInstaller creation."""
        from src.installer.docker import DockerInstaller, create_docker_installer
        from src.installer.config import (
            InstallConfig,
            InstallLocation,
            InstallMode,
            ComponentSelection,
        )

        config = InstallConfig(
            install_path=Path("/tmp/test"),
            install_location=InstallLocation.USER,
            install_mode=InstallMode.DOCKER,
            components=ComponentSelection(),
        )

        installer = create_docker_installer(config)
        assert isinstance(installer, DockerInstaller)


class TestCLIModule:
    """Tests for CLI module."""

    def test_progress_display_creation(self):
        """Test ProgressDisplay creation."""
        from src.installer.cli import ProgressDisplay

        display = ProgressDisplay(verbose=True)
        assert display.verbose is True

    def test_install_cli_creation(self):
        """Test InstallCLI creation."""
        from src.installer.cli import InstallCLI

        cli = InstallCLI()
        assert cli.parser is not None

    def test_install_cli_parse_args(self):
        """Test CLI argument parsing."""
        from src.installer.cli import InstallCLI

        cli = InstallCLI()

        args = cli.parser.parse_args(["--mode", "full", "--location", "user"])
        assert args.mode == "full"
        assert args.location == "user"

    def test_install_cli_parse_check_only(self):
        """Test CLI check-only argument."""
        from src.installer.cli import InstallCLI

        cli = InstallCLI()

        args = cli.parser.parse_args(["--check-only"])
        assert args.check_only is True

    def test_install_cli_parse_docker_mode(self):
        """Test CLI Docker mode argument."""
        from src.installer.cli import InstallCLI

        cli = InstallCLI()

        args = cli.parser.parse_args(["--mode", "docker"])
        assert args.mode == "docker"

    def test_install_cli_parse_components(self):
        """Test CLI components argument."""
        from src.installer.cli import InstallCLI

        cli = InstallCLI()

        args = cli.parser.parse_args(["--mode", "custom", "--components", "core", "agents"])
        assert args.mode == "custom"
        assert "core" in args.components
        assert "agents" in args.components


class TestModuleExports:
    """Tests for module exports."""

    def test_platform_exports(self):
        """Test platform module exports."""
        from src.installer import (
            Platform,
            PlatformInfo,
            SystemRequirements,
            RequirementCheck,
            RequirementStatus,
            detect_platform,
            check_requirements,
            get_system_info,
        )

        assert Platform is not None
        assert PlatformInfo is not None
        assert detect_platform is not None

    def test_config_exports(self):
        """Test config module exports."""
        from src.installer import (
            InstallConfig,
            InstallLocation,
            InstallMode,
            ComponentSelection,
            create_install_config,
        )

        assert InstallConfig is not None
        assert InstallMode is not None
        assert create_install_config is not None

    def test_base_exports(self):
        """Test base module exports."""
        from src.installer import (
            Installer,
            InstallerResult,
            InstallerError,
            InstallerProgress,
            ProgressCallback,
        )

        assert Installer is not None
        assert InstallerResult is not None

    def test_windows_exports(self):
        """Test Windows module exports."""
        from src.installer import WindowsInstaller, create_windows_installer

        assert WindowsInstaller is not None
        assert create_windows_installer is not None

    def test_macos_exports(self):
        """Test macOS module exports."""
        from src.installer import MacOSInstaller, create_macos_installer

        assert MacOSInstaller is not None
        assert create_macos_installer is not None

    def test_linux_exports(self):
        """Test Linux module exports."""
        from src.installer import (
            LinuxInstaller,
            LinuxDistro,
            create_linux_installer,
        )

        assert LinuxInstaller is not None
        assert LinuxDistro is not None

    def test_docker_exports(self):
        """Test Docker module exports."""
        from src.installer import (
            DockerInstaller,
            DockerConfig,
            create_docker_installer,
        )

        assert DockerInstaller is not None
        assert DockerConfig is not None

    def test_cli_exports(self):
        """Test CLI module exports."""
        from src.installer import InstallCLI, run_installer

        assert InstallCLI is not None
        assert run_installer is not None


class TestIntegration:
    """Integration tests for installer module."""

    def test_full_config_creation_flow(self):
        """Test full configuration creation flow."""
        from src.installer import (
            create_install_config,
            InstallMode,
            InstallLocation,
            detect_platform,
        )

        platform = detect_platform()
        config = create_install_config(
            mode=InstallMode.FULL,
            location=InstallLocation.USER,
        )

        assert config.install_path is not None
        assert config.components.core is True

    def test_requirements_check_flow(self):
        """Test requirements checking flow."""
        from src.installer import (
            check_requirements,
            all_requirements_met,
        )
        from src.installer.platform import all_requirements_met

        checks = check_requirements()
        result = all_requirements_met(checks)

        assert isinstance(result, bool)

    def test_installer_creation_for_current_platform(self):
        """Test creating installer for current platform."""
        from src.installer import (
            create_install_config,
            create_linux_installer,
            create_docker_installer,
            InstallMode,
            InstallLocation,
            detect_platform,
            Platform,
        )

        config = create_install_config(
            mode=InstallMode.FULL,
            location=InstallLocation.USER,
        )

        platform = detect_platform()

        if platform == Platform.LINUX:
            installer = create_linux_installer(config)
            assert installer is not None
        else:
            # Docker works on all platforms
            docker_config = create_install_config(
                mode=InstallMode.DOCKER,
                location=InstallLocation.USER,
            )
            installer = create_docker_installer(docker_config)
            assert installer is not None
