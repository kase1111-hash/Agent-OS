"""Command-line interface for Agent OS installer.

Provides an interactive CLI for:
- System requirements checking
- Installation mode selection
- Progress display
- Post-installation instructions
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Callable, List, Optional

from .base import Installer, InstallerProgress, InstallerResult
from .config import (
    ComponentSelection,
    InstallConfig,
    InstallLocation,
    InstallMode,
    create_install_config,
    get_default_install_path,
)
from .platform import (
    Platform,
    PlatformInfo,
    RequirementCheck,
    RequirementStatus,
    detect_platform,
    get_system_info,
)

logger = logging.getLogger(__name__)


class ProgressDisplay:
    """Display installation progress."""

    def __init__(self, verbose: bool = False):
        """Initialize progress display.

        Args:
            verbose: Whether to show verbose output
        """
        self.verbose = verbose
        self._last_phase = None
        self._bar_width = 40

    def __call__(self, progress: InstallerProgress) -> None:
        """Display progress update.

        Args:
            progress: Progress information
        """
        if progress.phase != self._last_phase:
            if self._last_phase is not None:
                print()  # New line after previous phase
            self._last_phase = progress.phase
            print(f"\n[{progress.phase.value.upper()}]")

        # Progress bar
        filled = int(self._bar_width * progress.progress)
        bar = "=" * filled + "-" * (self._bar_width - filled)
        print(f"\r  [{bar}] {progress.percent:3d}% {progress.message}", end="", flush=True)

        if progress.error:
            print(f"\n  ERROR: {progress.error}")

        if self.verbose and progress.details:
            for key, value in progress.details.items():
                print(f"\n    {key}: {value}")


class InstallCLI:
    """Command-line interface for installation."""

    def __init__(self):
        """Initialize CLI."""
        self.parser = self._create_parser()
        self.platform_info: Optional[PlatformInfo] = None

    def _create_parser(self) -> argparse.ArgumentParser:
        """Create argument parser."""
        parser = argparse.ArgumentParser(
            prog="agent-os-install",
            description="Agent OS One-Click Installer",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  # Interactive installation
  agent-os-install

  # Full installation to default location
  agent-os-install --mode full

  # Docker installation
  agent-os-install --mode docker

  # Custom path installation
  agent-os-install --path /opt/agent-os --location system

  # Check requirements only
  agent-os-install --check-only
""",
        )

        parser.add_argument(
            "--mode",
            "-m",
            choices=["full", "minimal", "docker", "custom"],
            default=None,
            help="Installation mode",
        )

        parser.add_argument(
            "--path",
            "-p",
            type=Path,
            default=None,
            help="Installation path (default: auto-detect)",
        )

        parser.add_argument(
            "--location",
            "-l",
            choices=["user", "system", "custom"],
            default="user",
            help="Installation location type",
        )

        parser.add_argument(
            "--components",
            "-c",
            nargs="+",
            choices=[
                "core",
                "agents",
                "memory",
                "web_ui",
                "voice",
                "multimodal",
                "federation",
                "sdk",
                "examples",
            ],
            help="Components to install (for custom mode)",
        )

        parser.add_argument(
            "--no-ollama",
            action="store_true",
            help="Skip Ollama installation",
        )

        parser.add_argument(
            "--no-models",
            action="store_true",
            help="Skip model downloads",
        )

        parser.add_argument(
            "--models",
            nargs="+",
            default=["mistral:7b-instruct"],
            help="Models to download",
        )

        parser.add_argument(
            "--no-shortcuts",
            action="store_true",
            help="Don't create desktop shortcuts",
        )

        parser.add_argument(
            "--no-path",
            action="store_true",
            help="Don't add to system PATH",
        )

        parser.add_argument(
            "--auto-start",
            action="store_true",
            help="Enable auto-start on boot",
        )

        parser.add_argument(
            "--check-only",
            action="store_true",
            help="Only check requirements, don't install",
        )

        parser.add_argument(
            "--uninstall",
            action="store_true",
            help="Uninstall Agent OS",
        )

        parser.add_argument(
            "--quiet",
            "-q",
            action="store_true",
            help="Minimal output",
        )

        parser.add_argument(
            "--verbose",
            "-v",
            action="store_true",
            help="Verbose output",
        )

        parser.add_argument(
            "--yes",
            "-y",
            action="store_true",
            help="Auto-confirm prompts",
        )

        parser.add_argument(
            "--version",
            action="version",
            version="Agent OS Installer 1.0.0",
        )

        return parser

    def run(self, args: Optional[List[str]] = None) -> int:
        """Run the CLI.

        Args:
            args: Command-line arguments

        Returns:
            Exit code (0 for success)
        """
        parsed = self.parser.parse_args(args)

        # Set up logging
        log_level = logging.WARNING if parsed.quiet else (
            logging.DEBUG if parsed.verbose else logging.INFO
        )
        logging.basicConfig(
            level=log_level,
            format="%(levelname)s: %(message)s",
        )

        # Get platform info
        self.platform_info = get_system_info()

        if not parsed.quiet:
            self._print_banner()
            self._print_system_info()

        # Check only mode
        if parsed.check_only:
            return self._check_requirements(parsed)

        # Uninstall mode
        if parsed.uninstall:
            return self._uninstall(parsed)

        # Interactive mode if no mode specified
        if parsed.mode is None and not parsed.yes:
            return self._interactive_install(parsed)

        # Non-interactive installation
        return self._run_installation(parsed)

    def _print_banner(self) -> None:
        """Print welcome banner."""
        print("""
╔══════════════════════════════════════════════════════════════╗
║                   Agent OS Installer                         ║
║           Constitutional AI Framework                        ║
╚══════════════════════════════════════════════════════════════╝
""")

    def _print_system_info(self) -> None:
        """Print detected system information."""
        info = self.platform_info
        print(f"Detected System:")
        print(f"  Platform:     {info.platform.value}")
        print(f"  Architecture: {info.architecture.value}")
        print(f"  OS:           {info.os_name} {info.os_release}")
        print(f"  User:         {info.username}")
        print()

    def _check_requirements(self, args) -> int:
        """Check system requirements.

        Returns:
            Exit code
        """
        print("Checking system requirements...\n")

        # Create a temporary installer to check requirements
        config = create_install_config(
            mode=InstallMode.FULL,
            location=InstallLocation(args.location),
            custom_path=args.path,
        )

        installer = self._create_installer(config)
        checks = installer.check_requirements()

        # Display results
        passed = 0
        warnings = 0
        failed = 0

        for check in checks:
            if check.status == RequirementStatus.PASSED:
                icon = "[OK]"
                passed += 1
            elif check.status == RequirementStatus.WARNING:
                icon = "[WARN]"
                warnings += 1
            elif check.status == RequirementStatus.FAILED:
                icon = "[FAIL]"
                failed += 1
            else:
                icon = "[SKIP]"

            print(f"  {icon:8} {check.name}: {check.message}")

        print()
        print(f"Results: {passed} passed, {warnings} warnings, {failed} failed")

        if failed > 0:
            print("\nSome critical requirements are not met.")
            return 1

        print("\nAll critical requirements passed!")
        return 0

    def _interactive_install(self, args) -> int:
        """Run interactive installation.

        Returns:
            Exit code
        """
        print("Interactive Installation\n")

        # Choose installation mode
        print("Select installation mode:")
        print("  1. Full      - All components (recommended)")
        print("  2. Minimal   - Core only")
        print("  3. Docker    - Container-based")
        print("  4. Custom    - Choose components")
        print()

        while True:
            choice = input("Enter choice [1-4] (default: 1): ").strip() or "1"
            if choice in ("1", "2", "3", "4"):
                break
            print("Invalid choice. Please enter 1-4.")

        mode_map = {
            "1": InstallMode.FULL,
            "2": InstallMode.MINIMAL,
            "3": InstallMode.DOCKER,
            "4": InstallMode.CUSTOM,
        }
        args.mode = mode_map[choice].value

        # Choose installation location
        print("\nSelect installation location:")
        default_path = get_default_install_path(InstallLocation.USER)
        print(f"  1. User directory   ({default_path})")
        system_path = get_default_install_path(InstallLocation.SYSTEM)
        print(f"  2. System directory ({system_path})")
        print("  3. Custom path")
        print()

        while True:
            choice = input("Enter choice [1-3] (default: 1): ").strip() or "1"
            if choice in ("1", "2", "3"):
                break
            print("Invalid choice. Please enter 1-3.")

        if choice == "1":
            args.location = "user"
        elif choice == "2":
            args.location = "system"
        else:
            custom_path = input("Enter custom path: ").strip()
            args.path = Path(custom_path)
            args.location = "custom"

        # Confirm
        print("\nInstallation Summary:")
        print(f"  Mode:     {args.mode}")
        print(f"  Location: {args.location}")
        if args.path:
            print(f"  Path:     {args.path}")
        print()

        confirm = input("Proceed with installation? [Y/n]: ").strip().lower()
        if confirm and confirm != "y":
            print("Installation cancelled.")
            return 1

        return self._run_installation(args)

    def _run_installation(self, args) -> int:
        """Run the installation.

        Returns:
            Exit code
        """
        # Determine mode
        mode = InstallMode(args.mode) if args.mode else InstallMode.FULL

        # Create configuration
        config = create_install_config(
            mode=mode,
            location=InstallLocation(args.location),
            custom_path=args.path,
            install_ollama=not args.no_ollama,
            install_models=[] if args.no_models else args.models,
            create_shortcuts=not args.no_shortcuts,
            add_to_path=not args.no_path,
            auto_start=args.auto_start,
        )

        # Handle custom components
        if mode == InstallMode.CUSTOM and args.components:
            config.components = ComponentSelection.from_list(args.components)

        # Create installer
        installer = self._create_installer(config)

        # Set up progress display
        if not args.quiet:
            progress_display = ProgressDisplay(verbose=args.verbose)
            installer.on_progress(progress_display)

        # Run installation
        print("\nStarting installation...\n")
        result = installer.run()

        # Display result
        print("\n")
        if result.success:
            self._print_success(result)
            return 0
        else:
            self._print_failure(result)
            return 1

    def _uninstall(self, args) -> int:
        """Uninstall Agent OS.

        Returns:
            Exit code
        """
        print("Uninstalling Agent OS...\n")

        # Load existing configuration
        config_path = args.path or get_default_install_path(
            InstallLocation(args.location)
        )
        install_config_file = config_path / "config" / "install.json"

        if not install_config_file.exists():
            print(f"Installation not found at {config_path}")
            return 1

        config = InstallConfig.load(install_config_file)
        installer = self._create_installer(config)

        if not args.yes:
            confirm = input(
                f"This will remove Agent OS from {config.install_path}. Continue? [y/N]: "
            ).strip().lower()
            if confirm != "y":
                print("Uninstallation cancelled.")
                return 1

        if installer.uninstall():
            print("Uninstallation complete!")
            return 0
        else:
            print("Uninstallation failed.")
            return 1

    def _create_installer(self, config: InstallConfig) -> Installer:
        """Create the appropriate installer for the platform.

        Args:
            config: Installation configuration

        Returns:
            Installer instance
        """
        if config.install_mode == InstallMode.DOCKER:
            from .docker import create_docker_installer

            return create_docker_installer(config, self.platform_info)

        platform = detect_platform()

        if platform == Platform.WINDOWS:
            from .windows import create_windows_installer

            return create_windows_installer(config, self.platform_info)
        elif platform == Platform.MACOS:
            from .macos import create_macos_installer

            return create_macos_installer(config, self.platform_info)
        else:  # Linux
            from .linux import create_linux_installer

            return create_linux_installer(config, self.platform_info)

    def _print_success(self, result: InstallerResult) -> None:
        """Print success message."""
        print("=" * 60)
        print("  Installation Successful!")
        print("=" * 60)
        print()
        print(f"  Install Path:  {result.install_path}")
        print(f"  Duration:      {result.duration_seconds:.1f} seconds")
        print()

        if result.warnings:
            print("  Warnings:")
            for warning in result.warnings:
                print(f"    - {warning}")
            print()

        print("  Getting Started:")
        print("    1. Open a new terminal to update PATH")
        print("    2. Run: agent-os --help")
        print("    3. Or access the web interface at http://localhost:8080")
        print()
        print("  Documentation: https://agent-os.dev/docs")
        print()

    def _print_failure(self, result: InstallerResult) -> None:
        """Print failure message."""
        print("=" * 60)
        print("  Installation Failed")
        print("=" * 60)
        print()

        if result.errors:
            print("  Errors:")
            for error in result.errors:
                print(f"    - {error}")
            print()

        if result.warnings:
            print("  Warnings:")
            for warning in result.warnings:
                print(f"    - {warning}")
            print()

        print("  Completed phases:")
        for phase in result.phases_completed:
            print(f"    - {phase.value}")
        print()

        print("  For help, please visit:")
        print("    https://github.com/kase1111-hash/Agent-OS/issues")
        print()


def run_installer(args: Optional[List[str]] = None) -> int:
    """Run the installer CLI.

    Args:
        args: Command-line arguments

    Returns:
        Exit code
    """
    cli = InstallCLI()
    return cli.run(args)


def main():
    """Main entry point."""
    sys.exit(run_installer())


if __name__ == "__main__":
    main()
