"""
Ceremony State Management

Manages the state of the Bring-Home Ceremony, tracking progress
through all 8 phases and persisting state between sessions.
"""

import hashlib
import json
import logging
import secrets
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class CeremonyPhase(Enum):
    """Phases of the Bring-Home Ceremony."""

    NOT_STARTED = auto()
    PHASE_I_COLD_BOOT = auto()
    PHASE_II_OWNER_ROOT = auto()
    PHASE_III_BOUNDARY_INIT = auto()
    PHASE_IV_VAULT_GENESIS = auto()
    PHASE_V_LEARNING_CONTRACTS = auto()
    PHASE_VI_VALUE_LEDGER = auto()
    PHASE_VII_FIRST_TRUST = auto()
    PHASE_VIII_EMERGENCY_DRILLS = auto()
    COMPLETED = auto()

    @property
    def phase_number(self) -> int:
        """Get the phase number (0-8)."""
        if self == CeremonyPhase.NOT_STARTED:
            return 0
        elif self == CeremonyPhase.COMPLETED:
            return 9
        else:
            return self.value - 1

    @property
    def display_name(self) -> str:
        """Get human-readable phase name."""
        names = {
            CeremonyPhase.NOT_STARTED: "Not Started",
            CeremonyPhase.PHASE_I_COLD_BOOT: "Phase I: Cold Boot",
            CeremonyPhase.PHASE_II_OWNER_ROOT: "Phase II: Owner Root",
            CeremonyPhase.PHASE_III_BOUNDARY_INIT: "Phase III: Boundary Init",
            CeremonyPhase.PHASE_IV_VAULT_GENESIS: "Phase IV: Vault Genesis",
            CeremonyPhase.PHASE_V_LEARNING_CONTRACTS: "Phase V: Learning Contracts",
            CeremonyPhase.PHASE_VI_VALUE_LEDGER: "Phase VI: Value Ledger",
            CeremonyPhase.PHASE_VII_FIRST_TRUST: "Phase VII: First Trust",
            CeremonyPhase.PHASE_VIII_EMERGENCY_DRILLS: "Phase VIII: Emergency Drills",
            CeremonyPhase.COMPLETED: "Completed",
        }
        return names.get(self, self.name)


class CeremonyStatus(Enum):
    """Status of a ceremony."""

    IN_PROGRESS = auto()
    COMPLETED = auto()
    FAILED = auto()
    ABANDONED = auto()


class PhaseResult(Enum):
    """Result of a phase execution."""

    SUCCESS = auto()
    FAILED = auto()
    SKIPPED = auto()
    REQUIRES_RETRY = auto()


@dataclass
class PhaseRecord:
    """Record of a phase completion."""

    phase: CeremonyPhase
    result: PhaseResult
    started_at: datetime
    completed_at: Optional[datetime] = None
    attempts: int = 1
    error_message: Optional[str] = None
    verification_hash: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "phase": self.phase.name,
            "result": self.result.name,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "attempts": self.attempts,
            "error_message": self.error_message,
            "verification_hash": self.verification_hash,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PhaseRecord":
        """Create from dictionary."""
        return cls(
            phase=CeremonyPhase[data["phase"]],
            result=PhaseResult[data["result"]],
            started_at=datetime.fromisoformat(data["started_at"]),
            completed_at=(
                datetime.fromisoformat(data["completed_at"]) if data.get("completed_at") else None
            ),
            attempts=data.get("attempts", 1),
            error_message=data.get("error_message"),
            verification_hash=data.get("verification_hash"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class CeremonyState:
    """State of the Bring-Home Ceremony."""

    ceremony_id: str
    owner_id: Optional[str] = None
    status: CeremonyStatus = CeremonyStatus.IN_PROGRESS
    current_phase: CeremonyPhase = CeremonyPhase.NOT_STARTED
    started_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    phase_records: List[PhaseRecord] = field(default_factory=list)
    owner_key_hash: Optional[str] = None
    vault_id: Optional[str] = None
    boundary_verified: bool = False
    contracts_initialized: bool = False
    ledger_initialized: bool = False
    drills_passed: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_complete(self) -> bool:
        """Check if ceremony is complete."""
        return self.status == CeremonyStatus.COMPLETED

    @property
    def progress_percent(self) -> float:
        """Get completion percentage."""
        if self.current_phase == CeremonyPhase.NOT_STARTED:
            return 0.0
        elif self.current_phase == CeremonyPhase.COMPLETED:
            return 100.0
        else:
            completed = sum(1 for r in self.phase_records if r.result == PhaseResult.SUCCESS)
            return (completed / 8) * 100

    def get_phase_record(self, phase: CeremonyPhase) -> Optional[PhaseRecord]:
        """Get record for a specific phase."""
        for record in self.phase_records:
            if record.phase == phase:
                return record
        return None

    def record_phase_start(self, phase: CeremonyPhase) -> PhaseRecord:
        """Record the start of a phase."""
        existing = self.get_phase_record(phase)
        if existing:
            existing.attempts += 1
            existing.started_at = datetime.now()
            existing.completed_at = None
            existing.result = PhaseResult.REQUIRES_RETRY
            return existing

        record = PhaseRecord(
            phase=phase,
            result=PhaseResult.REQUIRES_RETRY,
            started_at=datetime.now(),
        )
        self.phase_records.append(record)
        self.current_phase = phase
        return record

    def record_phase_complete(
        self,
        phase: CeremonyPhase,
        result: PhaseResult,
        verification_hash: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        error_message: Optional[str] = None,
    ) -> None:
        """Record phase completion."""
        record = self.get_phase_record(phase)
        if record:
            record.result = result
            record.completed_at = datetime.now()
            record.verification_hash = verification_hash
            record.error_message = error_message
            if metadata:
                record.metadata.update(metadata)

    def advance_to_next_phase(self) -> CeremonyPhase:
        """Advance to the next phase."""
        phase_order = [
            CeremonyPhase.NOT_STARTED,
            CeremonyPhase.PHASE_I_COLD_BOOT,
            CeremonyPhase.PHASE_II_OWNER_ROOT,
            CeremonyPhase.PHASE_III_BOUNDARY_INIT,
            CeremonyPhase.PHASE_IV_VAULT_GENESIS,
            CeremonyPhase.PHASE_V_LEARNING_CONTRACTS,
            CeremonyPhase.PHASE_VI_VALUE_LEDGER,
            CeremonyPhase.PHASE_VII_FIRST_TRUST,
            CeremonyPhase.PHASE_VIII_EMERGENCY_DRILLS,
            CeremonyPhase.COMPLETED,
        ]

        current_idx = phase_order.index(self.current_phase)
        if current_idx < len(phase_order) - 1:
            self.current_phase = phase_order[current_idx + 1]

            if self.current_phase == CeremonyPhase.COMPLETED:
                self.status = CeremonyStatus.COMPLETED
                self.completed_at = datetime.now()

        return self.current_phase

    def reset_to_phase(self, phase: CeremonyPhase) -> None:
        """Reset ceremony to a specific phase (for failed drills)."""
        phase_order = [
            CeremonyPhase.NOT_STARTED,
            CeremonyPhase.PHASE_I_COLD_BOOT,
            CeremonyPhase.PHASE_II_OWNER_ROOT,
            CeremonyPhase.PHASE_III_BOUNDARY_INIT,
            CeremonyPhase.PHASE_IV_VAULT_GENESIS,
            CeremonyPhase.PHASE_V_LEARNING_CONTRACTS,
            CeremonyPhase.PHASE_VI_VALUE_LEDGER,
            CeremonyPhase.PHASE_VII_FIRST_TRUST,
            CeremonyPhase.PHASE_VIII_EMERGENCY_DRILLS,
        ]

        target_idx = phase_order.index(phase)

        # Mark phases after target as skipped
        for record in self.phase_records:
            phase_idx = phase_order.index(record.phase)
            if phase_idx >= target_idx:
                record.result = PhaseResult.SKIPPED

        self.current_phase = phase
        self.status = CeremonyStatus.IN_PROGRESS

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "ceremony_id": self.ceremony_id,
            "owner_id": self.owner_id,
            "status": self.status.name,
            "current_phase": self.current_phase.name,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "phase_records": [r.to_dict() for r in self.phase_records],
            "owner_key_hash": self.owner_key_hash,
            "vault_id": self.vault_id,
            "boundary_verified": self.boundary_verified,
            "contracts_initialized": self.contracts_initialized,
            "ledger_initialized": self.ledger_initialized,
            "drills_passed": self.drills_passed,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CeremonyState":
        """Create from dictionary."""
        state = cls(
            ceremony_id=data["ceremony_id"],
            owner_id=data.get("owner_id"),
            status=CeremonyStatus[data["status"]],
            current_phase=CeremonyPhase[data["current_phase"]],
            started_at=datetime.fromisoformat(data["started_at"]),
            completed_at=(
                datetime.fromisoformat(data["completed_at"]) if data.get("completed_at") else None
            ),
            owner_key_hash=data.get("owner_key_hash"),
            vault_id=data.get("vault_id"),
            boundary_verified=data.get("boundary_verified", False),
            contracts_initialized=data.get("contracts_initialized", False),
            ledger_initialized=data.get("ledger_initialized", False),
            drills_passed=data.get("drills_passed", False),
            metadata=data.get("metadata", {}),
        )

        for record_data in data.get("phase_records", []):
            state.phase_records.append(PhaseRecord.from_dict(record_data))

        return state


class CeremonyStateManager:
    """
    Manages ceremony state persistence.

    Saves and loads ceremony state to/from disk.
    """

    def __init__(self, state_dir: Path):
        """
        Initialize state manager.

        Args:
            state_dir: Directory for state files
        """
        self.state_dir = state_dir
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self._state_file = self.state_dir / "ceremony_state.json"

    def create_ceremony(self, owner_id: Optional[str] = None) -> CeremonyState:
        """Create a new ceremony."""
        ceremony_id = f"CEREMONY-{secrets.token_hex(8)}"

        state = CeremonyState(
            ceremony_id=ceremony_id,
            owner_id=owner_id,
        )

        self.save_state(state)
        logger.info(f"Created new ceremony: {ceremony_id}")

        return state

    def load_state(self) -> Optional[CeremonyState]:
        """Load ceremony state from disk."""
        if not self._state_file.exists():
            return None

        try:
            with open(self._state_file, "r") as f:
                data = json.load(f)
            return CeremonyState.from_dict(data)
        except Exception as e:
            logger.error(f"Failed to load ceremony state: {e}")
            return None

    def save_state(self, state: CeremonyState) -> bool:
        """Save ceremony state to disk."""
        try:
            with open(self._state_file, "w") as f:
                json.dump(state.to_dict(), f, indent=2)
            return True
        except Exception as e:
            logger.error(f"Failed to save ceremony state: {e}")
            return False

    def has_ceremony(self) -> bool:
        """Check if a ceremony exists."""
        return self._state_file.exists()

    def get_or_create_ceremony(self, owner_id: Optional[str] = None) -> CeremonyState:
        """Get existing ceremony or create new one."""
        state = self.load_state()
        if state:
            return state
        return self.create_ceremony(owner_id)

    def clear_ceremony(self) -> bool:
        """Clear ceremony state (for testing)."""
        if self._state_file.exists():
            self._state_file.unlink()
            return True
        return False

    def compute_state_hash(self, state: CeremonyState) -> str:
        """Compute hash of state for integrity verification."""
        content = json.dumps(state.to_dict(), sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()


def create_state_manager(state_dir: Path) -> CeremonyStateManager:
    """Factory function to create a state manager."""
    return CeremonyStateManager(state_dir)
