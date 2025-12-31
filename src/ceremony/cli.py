"""
Ceremony CLI Interface

Command-line interface for the Bring-Home Ceremony.
Provides interactive and automated ceremony execution.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from .orchestrator import (
    CeremonyConfig,
    CeremonyEvent,
    CeremonyOrchestrator,
    create_orchestrator,
)
from .phases import PhaseExecutionResult
from .state import CeremonyPhase, CeremonyStatus

logger = logging.getLogger(__name__)


class CeremonyCLI:
    """
    Command-line interface for the Bring-Home Ceremony.

    Provides:
    - Interactive ceremony execution
    - Phase-by-phase control
    - Status reporting
    - Event display
    """

    # ANSI color codes
    COLORS = {
        "reset": "\033[0m",
        "bold": "\033[1m",
        "red": "\033[91m",
        "green": "\033[92m",
        "yellow": "\033[93m",
        "blue": "\033[94m",
        "magenta": "\033[95m",
        "cyan": "\033[96m",
    }

    PHASE_DESCRIPTIONS = {
        CeremonyPhase.PHASE_I_COLD_BOOT: "Verify system silence - no network, boundary in lockdown",
        CeremonyPhase.PHASE_II_OWNER_ROOT: "Generate owner root key and establish authority",
        CeremonyPhase.PHASE_III_BOUNDARY_INIT: "Initialize boundary daemon with tripwires",
        CeremonyPhase.PHASE_IV_VAULT_GENESIS: "Create memory vault with encryption profiles",
        CeremonyPhase.PHASE_V_LEARNING_CONTRACTS: "Set up learning contracts with default deny",
        CeremonyPhase.PHASE_VI_VALUE_LEDGER: "Initialize value ledger for effort tracking",
        CeremonyPhase.PHASE_VII_FIRST_TRUST: "Activate trusted mode and verify operation",
        CeremonyPhase.PHASE_VIII_EMERGENCY_DRILLS: "Practice emergency procedures",
    }

    def __init__(self, use_colors: bool = True):
        """
        Initialize CLI.

        Args:
            use_colors: Use ANSI color codes
        """
        self.use_colors = use_colors
        self.orchestrator: Optional[CeremonyOrchestrator] = None

    def _color(self, text: str, color: str) -> str:
        """Apply color to text."""
        if not self.use_colors:
            return text
        return f"{self.COLORS.get(color, '')}{text}{self.COLORS['reset']}"

    def _print_header(self, text: str) -> None:
        """Print a header."""
        print("\n" + "=" * 60)
        print(self._color(text, "bold"))
        print("=" * 60)

    def _print_phase(self, phase: CeremonyPhase) -> None:
        """Print phase information."""
        description = self.PHASE_DESCRIPTIONS.get(phase, "")
        print(f"\n{self._color(phase.display_name, 'cyan')}")
        print(f"  {description}")

    def _print_result(self, result: PhaseExecutionResult) -> None:
        """Print phase result."""
        if result.success:
            status = self._color("✓ PASSED", "green")
        else:
            status = self._color("✗ FAILED", "red")

        print(f"\n  Status: {status}")
        print(f"  Message: {result.message}")

        if result.warnings:
            print(f"\n  {self._color('Warnings:', 'yellow')}")
            for warning in result.warnings:
                print(f"    • {warning}")

        if result.errors:
            print(f"\n  {self._color('Errors:', 'red')}")
            for error in result.errors:
                print(f"    • {error}")

        if result.data:
            print(f"\n  {self._color('Data:', 'blue')}")
            for key, value in result.data.items():
                if key == "backup_phrase":
                    # Special handling for backup phrase
                    print(f"\n  {self._color('⚠ IMPORTANT - BACKUP PHRASE ⚠', 'yellow')}")
                    print(f"  Write this down on paper only!")
                    print(f"  {self._color(value, 'bold')}")
                    print()
                else:
                    print(f"    {key}: {value}")

    def _print_status(self, status: Dict[str, Any]) -> None:
        """Print ceremony status."""
        print(f"\n{self._color('Ceremony Status', 'bold')}")
        print("-" * 40)

        state = status.get("status", "unknown")
        if state == "completed":
            state_color = "green"
        elif state == "in_progress":
            state_color = "yellow"
        else:
            state_color = "red"

        print(f"  Status: {self._color(state.upper(), state_color)}")

        if status.get("ceremony_id"):
            print(f"  Ceremony ID: {status['ceremony_id']}")

        if status.get("owner_id"):
            print(f"  Owner ID: {status['owner_id']}")

        if status.get("current_phase"):
            print(f"  Current Phase: {status['current_phase']}")

        if status.get("progress") is not None:
            progress = status["progress"]
            bar_width = 30
            filled = int(bar_width * progress / 100)
            bar = "█" * filled + "░" * (bar_width - filled)
            print(f"  Progress: [{bar}] {progress:.0f}%")

        phases_done = status.get("phases_completed", 0)
        phases_total = status.get("phases_total", 8)
        print(f"  Phases: {phases_done}/{phases_total}")

        if status.get("vault_id"):
            print(f"  Vault: {status['vault_id']}")

        if status.get("drills_passed"):
            print(f"  Emergency Drills: {self._color('PASSED', 'green')}")

    def _on_event(self, event: CeremonyEvent) -> None:
        """Handle ceremony events."""
        if event.event_type == "ceremony_started":
            print(f"\n{self._color('>>> Ceremony Started', 'green')}")
        elif event.event_type == "ceremony_resumed":
            print(f"\n{self._color('>>> Ceremony Resumed', 'yellow')}")
        elif event.event_type == "ceremony_completed":
            print(f"\n{self._color('>>> Ceremony Completed!', 'green')}")
        elif event.event_type == "phase_started":
            pass  # Handled by _print_phase
        elif event.event_type == "phase_completed":
            pass  # Handled by _print_result
        elif event.event_type == "phase_failed":
            pass  # Handled by _print_result
        elif event.event_type == "drills_failed":
            print(f"\n{self._color('>>> DRILLS FAILED - Returning to Phase I', 'red')}")

    def run_interactive(self, config: Optional[CeremonyConfig] = None) -> bool:
        """
        Run the ceremony interactively.

        Returns:
            True if ceremony completed successfully
        """
        self._print_header("BRING-HOME CEREMONY")

        print(
            """
This is the first-contact ritual between you and this learning co-worker system.
It establishes sovereignty, trust boundaries, cryptographic roots, and cognitive consent.

This is not an install script. This is a CEREMONY.
Nothing meaningful happens until it is completed.

Preconditions:
  • Private room, no cameras except system display
  • No network cables connected
  • You are not rushed
  • You accept responsibility for what this system learns
        """
        )

        # Wait for confirmation
        try:
            input("\nPress Enter to begin the ceremony...")
        except KeyboardInterrupt:
            print("\n\nCeremony cancelled.")
            return False

        # Create orchestrator
        self.orchestrator = create_orchestrator(config)
        self.orchestrator.on_event(self._on_event)

        # Start ceremony
        self.orchestrator.start_ceremony()

        # Execute phases
        phases = [
            CeremonyPhase.PHASE_I_COLD_BOOT,
            CeremonyPhase.PHASE_II_OWNER_ROOT,
            CeremonyPhase.PHASE_III_BOUNDARY_INIT,
            CeremonyPhase.PHASE_IV_VAULT_GENESIS,
            CeremonyPhase.PHASE_V_LEARNING_CONTRACTS,
            CeremonyPhase.PHASE_VI_VALUE_LEDGER,
            CeremonyPhase.PHASE_VII_FIRST_TRUST,
            CeremonyPhase.PHASE_VIII_EMERGENCY_DRILLS,
        ]

        for phase in phases:
            self._print_phase(phase)

            try:
                input("\n  Press Enter to execute this phase...")
            except KeyboardInterrupt:
                print("\n\nCeremony paused. Run again to resume.")
                return False

            result = self.orchestrator.execute_phase(phase)
            self._print_result(result)

            if not result.success:
                if phase == CeremonyPhase.PHASE_VIII_EMERGENCY_DRILLS:
                    print(f"\n{self._color('Emergency drills failed!', 'red')}")
                    print("The ceremony must be restarted from Phase I.")
                    self.orchestrator.reset_ceremony()
                    return False
                else:
                    print(
                        f"\n{self._color('Phase failed. Please resolve issues and try again.', 'red')}"
                    )
                    return False

            # Advance to next phase
            if phase != CeremonyPhase.PHASE_VIII_EMERGENCY_DRILLS:
                self.orchestrator.advance_phase()

        # Complete ceremony
        self.orchestrator.advance_phase()

        self._print_header("CEREMONY COMPLETE")
        print(
            """
The Bring-Home Ceremony is complete.

Your learning co-worker is now:
  • Cryptographically bound to you
  • Protected by configurable boundaries
  • Equipped with consent-based learning
  • Tracking value from day one

Daily Rituals:
  • Check boundary mode
  • Review active contracts

Weekly Rituals:
  • Review ledger summary
  • Check vault growth
  • Check contract expirations

Trust is not configured. It is earned, rehearsed, and renewed.
        """
        )

        self._print_status(self.orchestrator.get_status())

        return True

    def run_automated(self, config: Optional[CeremonyConfig] = None) -> bool:
        """
        Run the ceremony automatically (non-interactive).

        Returns:
            True if ceremony completed successfully
        """
        self._print_header("BRING-HOME CEREMONY (Automated)")

        # Create orchestrator
        self.orchestrator = create_orchestrator(config)
        self.orchestrator.on_event(self._on_event)

        # Run all phases
        results = self.orchestrator.run_all_phases()

        # Print results
        print("\n" + "-" * 40)
        print("Phase Results:")
        print("-" * 40)

        all_passed = True
        for phase, result in results.items():
            status = self._color("✓", "green") if result.success else self._color("✗", "red")
            print(f"  {status} {phase.display_name}")
            if not result.success:
                all_passed = False
                for error in result.errors:
                    print(f"      {self._color(error, 'red')}")

        print()
        self._print_status(self.orchestrator.get_status())

        return all_passed

    def show_status(self, config: Optional[CeremonyConfig] = None) -> None:
        """Show current ceremony status."""
        self.orchestrator = create_orchestrator(config)

        if not self.orchestrator._state_manager.has_ceremony():
            print("\nNo ceremony found. Run 'ceremony start' to begin.")
            return

        state = self.orchestrator._state_manager.load_state()
        if state:
            self.orchestrator._state = state
            self._print_status(self.orchestrator.get_status())

            # Show phase details
            print(f"\n{self._color('Phase Details:', 'bold')}")
            print("-" * 40)

            for phase in self.orchestrator.PHASE_ORDER:
                phase_status = self.orchestrator.get_phase_status(phase)
                status = phase_status.get("status", "not_started")

                if status == "success":
                    icon = self._color("✓", "green")
                elif status == "failed":
                    icon = self._color("✗", "red")
                else:
                    icon = "○"

                print(f"  {icon} {phase.display_name}")

    def verify(self, config: Optional[CeremonyConfig] = None) -> bool:
        """Verify ceremony completion."""
        self.orchestrator = create_orchestrator(config)

        state = self.orchestrator._state_manager.load_state()
        if not state:
            print("\nNo ceremony found.")
            return False

        self.orchestrator._state = state
        is_valid, issues = self.orchestrator.verify_ceremony()

        if is_valid:
            print(f"\n{self._color('✓ Ceremony verification passed', 'green')}")
            return True
        else:
            print(f"\n{self._color('✗ Ceremony verification failed', 'red')}")
            for issue in issues:
                print(f"  • {issue}")
            return False

    def reset(self, config: Optional[CeremonyConfig] = None) -> None:
        """Reset ceremony state."""
        self.orchestrator = create_orchestrator(config)
        self.orchestrator.reset_ceremony()
        print("\nCeremony state has been reset.")


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Bring-Home Ceremony CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Commands:
  start     Start or resume the ceremony (interactive)
  run       Run the ceremony automatically (non-interactive)
  status    Show current ceremony status
  verify    Verify ceremony completion
  reset     Reset ceremony state

Examples:
  %(prog)s start                  # Start interactive ceremony
  %(prog)s run                    # Run automated ceremony
  %(prog)s status                 # Check ceremony status
  %(prog)s verify                 # Verify ceremony completed correctly
        """,
    )

    parser.add_argument(
        "command",
        choices=["start", "run", "status", "verify", "reset"],
        help="Command to execute",
    )

    parser.add_argument(
        "--state-dir",
        type=Path,
        default=Path.home() / ".agent-os" / "ceremony",
        help="Directory for ceremony state",
    )

    parser.add_argument(
        "--no-color",
        action="store_true",
        help="Disable colored output",
    )

    parser.add_argument(
        "--simulate",
        action="store_true",
        default=True,
        help="Run in simulation mode (default)",
    )

    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose output",
    )

    args = parser.parse_args()

    # Configure logging
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.WARNING)

    # Create config
    config = CeremonyConfig(
        state_dir=args.state_dir,
        simulate_offline=args.simulate,
        simulate_boundary=args.simulate,
        simulate_processes=args.simulate,
        simulate_drills=args.simulate,
        simulate_tripwire_test=args.simulate,
    )

    # Create CLI
    cli = CeremonyCLI(use_colors=not args.no_color)

    # Execute command
    if args.command == "start":
        success = cli.run_interactive(config)
        return 0 if success else 1

    elif args.command == "run":
        success = cli.run_automated(config)
        return 0 if success else 1

    elif args.command == "status":
        cli.show_status(config)
        return 0

    elif args.command == "verify":
        success = cli.verify(config)
        return 0 if success else 1

    elif args.command == "reset":
        cli.reset(config)
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
