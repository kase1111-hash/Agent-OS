"""
Ceremony Orchestrator

Orchestrates the Bring-Home Ceremony, managing phase execution,
state transitions, and providing the main ceremony interface.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from .phases import (
    CeremonyPhaseExecutor,
    PhaseExecutionResult,
    create_phase_executor,
)
from .state import (
    CeremonyPhase,
    CeremonyState,
    CeremonyStateManager,
    CeremonyStatus,
    PhaseResult,
    create_state_manager,
)

logger = logging.getLogger(__name__)


@dataclass
class CeremonyConfig:
    """Configuration for the ceremony."""

    state_dir: Path = field(default_factory=lambda: Path.home() / ".agent-os" / "ceremony")
    vault_path: Path = field(default_factory=lambda: Path.home() / ".agent-os" / "vault")
    simulate_offline: bool = True
    simulate_boundary: bool = True
    simulate_processes: bool = True
    simulate_drills: bool = True
    simulate_tripwire_test: bool = True
    enable_hardware_binding: bool = False
    auto_advance: bool = False  # Auto-advance to next phase on success


@dataclass
class CeremonyEvent:
    """Event from ceremony execution."""

    event_type: str
    phase: Optional[CeremonyPhase] = None
    message: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    data: Dict[str, Any] = field(default_factory=dict)


class CeremonyOrchestrator:
    """
    Orchestrates the Bring-Home Ceremony.

    Manages:
    - Phase execution flow
    - State persistence
    - Event callbacks
    - Ceremony lifecycle
    """

    PHASE_ORDER = [
        CeremonyPhase.PHASE_I_COLD_BOOT,
        CeremonyPhase.PHASE_II_OWNER_ROOT,
        CeremonyPhase.PHASE_III_BOUNDARY_INIT,
        CeremonyPhase.PHASE_IV_VAULT_GENESIS,
        CeremonyPhase.PHASE_V_LEARNING_CONTRACTS,
        CeremonyPhase.PHASE_VI_VALUE_LEDGER,
        CeremonyPhase.PHASE_VII_FIRST_TRUST,
        CeremonyPhase.PHASE_VIII_EMERGENCY_DRILLS,
    ]

    def __init__(self, config: Optional[CeremonyConfig] = None):
        """
        Initialize the orchestrator.

        Args:
            config: Ceremony configuration
        """
        self.config = config or CeremonyConfig()
        self._state_manager = create_state_manager(self.config.state_dir)
        self._state: Optional[CeremonyState] = None
        self._event_handlers: List[Callable[[CeremonyEvent], None]] = []
        self._phase_callbacks: Dict[CeremonyPhase, List[Callable]] = {}

    @property
    def state(self) -> Optional[CeremonyState]:
        """Get current ceremony state."""
        return self._state

    @property
    def current_phase(self) -> CeremonyPhase:
        """Get current phase."""
        if self._state:
            return self._state.current_phase
        return CeremonyPhase.NOT_STARTED

    @property
    def is_complete(self) -> bool:
        """Check if ceremony is complete."""
        return self._state and self._state.is_complete

    @property
    def progress(self) -> float:
        """Get ceremony progress percentage."""
        if self._state:
            return self._state.progress_percent
        return 0.0

    def start_ceremony(self, owner_id: Optional[str] = None) -> CeremonyState:
        """
        Start a new ceremony or resume existing one.

        Args:
            owner_id: Optional owner identifier

        Returns:
            Ceremony state
        """
        # Check for existing ceremony
        existing = self._state_manager.load_state()
        if existing and existing.status == CeremonyStatus.IN_PROGRESS:
            self._state = existing
            self._emit_event(
                CeremonyEvent(
                    event_type="ceremony_resumed",
                    phase=existing.current_phase,
                    message=f"Resuming ceremony at {existing.current_phase.display_name}",
                )
            )
            return self._state

        # Create new ceremony
        self._state = self._state_manager.create_ceremony(owner_id)
        self._emit_event(
            CeremonyEvent(
                event_type="ceremony_started",
                message="New ceremony started",
                data={"ceremony_id": self._state.ceremony_id},
            )
        )

        return self._state

    def execute_phase(self, phase: CeremonyPhase) -> PhaseExecutionResult:
        """
        Execute a specific phase.

        Args:
            phase: Phase to execute

        Returns:
            Phase execution result
        """
        if not self._state:
            raise RuntimeError("No ceremony in progress")

        # Create phase executor
        executor = create_phase_executor(phase, self._state, self._get_phase_config())

        if not executor:
            return PhaseExecutionResult(
                success=False,
                phase=phase,
                message=f"No executor for phase: {phase.display_name}",
                errors=[f"Unknown phase: {phase.name}"],
            )

        # Check prerequisites
        prereqs_ok, missing = executor.check_prerequisites()
        if not prereqs_ok:
            return PhaseExecutionResult(
                success=False,
                phase=phase,
                message="Prerequisites not met",
                errors=[f"Missing prerequisites: {', '.join(missing)}"],
            )

        # Record phase start
        self._state.record_phase_start(phase)
        self._emit_event(
            CeremonyEvent(
                event_type="phase_started",
                phase=phase,
                message=f"Starting {phase.display_name}",
            )
        )

        # Execute phase
        result = executor.execute()

        # Record phase completion
        self._state.record_phase_complete(
            phase=phase,
            result=result.result,
            verification_hash=result.verification_hash,
            metadata=result.data,
            error_message="; ".join(result.errors) if result.errors else None,
        )

        # Emit completion event
        self._emit_event(
            CeremonyEvent(
                event_type="phase_completed" if result.success else "phase_failed",
                phase=phase,
                message=result.message,
                data=result.data,
            )
        )

        # Save state
        self._state_manager.save_state(self._state)

        # Handle phase-specific callbacks
        self._invoke_phase_callbacks(phase, result)

        # Auto-advance if configured and successful
        if result.success and self.config.auto_advance:
            next_phase = self._get_next_phase(phase)
            if next_phase and next_phase != CeremonyPhase.COMPLETED:
                self._state.advance_to_next_phase()
                self._state_manager.save_state(self._state)

        return result

    def execute_current_phase(self) -> PhaseExecutionResult:
        """Execute the current phase."""
        if not self._state:
            raise RuntimeError("No ceremony in progress")

        current = self._state.current_phase

        if current == CeremonyPhase.NOT_STARTED:
            current = CeremonyPhase.PHASE_I_COLD_BOOT

        if current == CeremonyPhase.COMPLETED:
            return PhaseExecutionResult(
                success=True,
                phase=current,
                message="Ceremony already complete",
            )

        return self.execute_phase(current)

    def advance_phase(self) -> CeremonyPhase:
        """
        Advance to the next phase.

        Returns:
            The new current phase
        """
        if not self._state:
            raise RuntimeError("No ceremony in progress")

        current = self._state.current_phase

        # Can always advance from NOT_STARTED to Phase I
        if current != CeremonyPhase.NOT_STARTED:
            current_record = self._state.get_phase_record(current)

            # Must have completed current phase to advance
            if not current_record or current_record.result != PhaseResult.SUCCESS:
                raise RuntimeError(
                    f"Cannot advance: {current.display_name} not completed successfully"
                )

        new_phase = self._state.advance_to_next_phase()
        self._state_manager.save_state(self._state)

        self._emit_event(
            CeremonyEvent(
                event_type="phase_advanced",
                phase=new_phase,
                message=f"Advanced to {new_phase.display_name}",
            )
        )

        return new_phase

    def run_all_phases(self) -> Dict[CeremonyPhase, PhaseExecutionResult]:
        """
        Run all ceremony phases in sequence.

        Returns:
            Dictionary of phase results
        """
        if not self._state:
            self.start_ceremony()

        results = {}

        for phase in self.PHASE_ORDER:
            result = self.execute_phase(phase)
            results[phase] = result

            if not result.success:
                # If emergency drills fail, we should restart
                if phase == CeremonyPhase.PHASE_VIII_EMERGENCY_DRILLS:
                    self._emit_event(
                        CeremonyEvent(
                            event_type="drills_failed",
                            phase=phase,
                            message="Emergency drills failed - ceremony must restart",
                        )
                    )
                    self._state.reset_to_phase(CeremonyPhase.PHASE_I_COLD_BOOT)
                    self._state_manager.save_state(self._state)
                break

            # Advance to next phase
            if phase != CeremonyPhase.PHASE_VIII_EMERGENCY_DRILLS:
                self._state.advance_to_next_phase()
                self._state_manager.save_state(self._state)

        # Mark complete if all passed
        if all(r.success for r in results.values()):
            self._state.advance_to_next_phase()  # Move to COMPLETED
            self._state_manager.save_state(self._state)

            self._emit_event(
                CeremonyEvent(
                    event_type="ceremony_completed",
                    message="Bring-Home Ceremony completed successfully",
                    data={
                        "ceremony_id": self._state.ceremony_id,
                        "owner_id": self._state.owner_id,
                        "vault_id": self._state.vault_id,
                    },
                )
            )

        return results

    def verify_ceremony(self) -> Tuple[bool, List[str]]:
        """
        Verify the ceremony was completed correctly.

        Returns:
            Tuple of (is_valid, list of issues)
        """
        if not self._state:
            return False, ["No ceremony state found"]

        issues = []

        # Check all phases completed
        for phase in self.PHASE_ORDER:
            record = self._state.get_phase_record(phase)
            if not record:
                issues.append(f"{phase.display_name}: Not executed")
            elif record.result != PhaseResult.SUCCESS:
                issues.append(f"{phase.display_name}: {record.result.name}")

        # Check required state
        if not self._state.owner_key_hash:
            issues.append("Owner key not established")
        if not self._state.vault_id:
            issues.append("Vault not initialized")
        if not self._state.boundary_verified:
            issues.append("Boundary not verified")
        if not self._state.contracts_initialized:
            issues.append("Contracts not initialized")
        if not self._state.ledger_initialized:
            issues.append("Ledger not initialized")
        if not self._state.drills_passed:
            issues.append("Emergency drills not passed")

        return len(issues) == 0, issues

    def get_status(self) -> Dict[str, Any]:
        """Get ceremony status summary."""
        if not self._state:
            return {
                "status": "not_started",
                "has_ceremony": self._state_manager.has_ceremony(),
            }

        return {
            "status": self._state.status.name.lower(),
            "ceremony_id": self._state.ceremony_id,
            "owner_id": self._state.owner_id,
            "current_phase": self._state.current_phase.display_name,
            "progress": self._state.progress_percent,
            "started_at": self._state.started_at.isoformat(),
            "completed_at": (
                self._state.completed_at.isoformat() if self._state.completed_at else None
            ),
            "phases_completed": sum(
                1 for r in self._state.phase_records if r.result == PhaseResult.SUCCESS
            ),
            "phases_total": len(self.PHASE_ORDER),
            "vault_id": self._state.vault_id,
            "drills_passed": self._state.drills_passed,
        }

    def get_phase_status(self, phase: CeremonyPhase) -> Dict[str, Any]:
        """Get status of a specific phase."""
        if not self._state:
            return {"status": "no_ceremony"}

        record = self._state.get_phase_record(phase)
        if not record:
            return {
                "phase": phase.display_name,
                "status": "not_started",
            }

        return {
            "phase": phase.display_name,
            "status": record.result.name.lower(),
            "attempts": record.attempts,
            "started_at": record.started_at.isoformat(),
            "completed_at": record.completed_at.isoformat() if record.completed_at else None,
            "verification_hash": record.verification_hash,
            "metadata": record.metadata,
        }

    def reset_ceremony(self) -> bool:
        """Reset the ceremony (for testing or recovery)."""
        self._state_manager.clear_ceremony()
        self._state = None

        self._emit_event(
            CeremonyEvent(
                event_type="ceremony_reset",
                message="Ceremony has been reset",
            )
        )

        return True

    def on_event(self, handler: Callable[[CeremonyEvent], None]) -> None:
        """Register an event handler."""
        self._event_handlers.append(handler)

    def on_phase_complete(
        self,
        phase: CeremonyPhase,
        callback: Callable[[PhaseExecutionResult], None],
    ) -> None:
        """Register a callback for phase completion."""
        if phase not in self._phase_callbacks:
            self._phase_callbacks[phase] = []
        self._phase_callbacks[phase].append(callback)

    def _emit_event(self, event: CeremonyEvent) -> None:
        """Emit an event to all handlers."""
        for handler in self._event_handlers:
            try:
                handler(event)
            except Exception as e:
                logger.error(f"Event handler error: {e}")

    def _invoke_phase_callbacks(
        self,
        phase: CeremonyPhase,
        result: PhaseExecutionResult,
    ) -> None:
        """Invoke callbacks for a phase."""
        callbacks = self._phase_callbacks.get(phase, [])
        for callback in callbacks:
            try:
                callback(result)
            except Exception as e:
                logger.error(f"Phase callback error: {e}")

    def _get_phase_config(self) -> Dict[str, Any]:
        """Get configuration for phase execution."""
        return {
            "vault_path": self.config.vault_path,
            "simulate_offline": self.config.simulate_offline,
            "simulate_boundary": self.config.simulate_boundary,
            "simulate_processes": self.config.simulate_processes,
            "simulate_drills": self.config.simulate_drills,
            "simulate_tripwire_test": self.config.simulate_tripwire_test,
            "enable_hardware_binding": self.config.enable_hardware_binding,
        }

    def _get_next_phase(self, current: CeremonyPhase) -> Optional[CeremonyPhase]:
        """Get the next phase after current."""
        try:
            idx = self.PHASE_ORDER.index(current)
            if idx < len(self.PHASE_ORDER) - 1:
                return self.PHASE_ORDER[idx + 1]
            return CeremonyPhase.COMPLETED
        except ValueError:
            return None


def create_orchestrator(config: Optional[CeremonyConfig] = None) -> CeremonyOrchestrator:
    """
    Factory function to create a ceremony orchestrator.

    Args:
        config: Ceremony configuration

    Returns:
        Configured CeremonyOrchestrator
    """
    return CeremonyOrchestrator(config)


def run_ceremony(config: Optional[CeremonyConfig] = None) -> Tuple[bool, Dict[str, Any]]:
    """
    Convenience function to run the complete ceremony.

    Args:
        config: Ceremony configuration

    Returns:
        Tuple of (success, results)
    """
    orchestrator = create_orchestrator(config)
    orchestrator.start_ceremony()

    results = orchestrator.run_all_phases()

    success = all(r.success for r in results.values())
    status = orchestrator.get_status()

    return success, {
        "status": status,
        "phases": {
            phase.display_name: {
                "success": result.success,
                "message": result.message,
                "data": result.data,
            }
            for phase, result in results.items()
        },
    }
