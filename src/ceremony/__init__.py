"""
Agent OS Bring-Home Ceremony Module

First-contact ritual establishing cryptographic ownership via hardware-bound keys.
Implements the 8-phase ceremony for system initialization.

Phases:
- Phase I: Cold Boot - Establish silence
- Phase II: Owner Root - Generate owner key
- Phase III: Boundary Init - Configure boundaries
- Phase IV: Vault Genesis - Initialize memory vault
- Phase V: Learning Contracts - Set up consent
- Phase VI: Value Ledger - Initialize ledger
- Phase VII: First Trust - Activate trusted mode
- Phase VIII: Emergency Drills - Practice procedures

Usage:
    from src.ceremony import CeremonyOrchestrator, create_orchestrator

    # Create and run ceremony
    orchestrator = create_orchestrator()
    orchestrator.start_ceremony()

    # Execute phases
    result = orchestrator.execute_current_phase()
    if result.success:
        orchestrator.advance_phase()

    # Or run all at once
    success, results = run_ceremony()
"""

# CLI
from .cli import (
    CeremonyCLI,
)
from .cli import main as cli_main

# Orchestrator
from .orchestrator import (
    CeremonyConfig,
    CeremonyEvent,
    CeremonyOrchestrator,
    create_orchestrator,
    run_ceremony,
)

# Phase executors
from .phases import (
    BoundaryInitPhase,
    CeremonyPhaseExecutor,
    ColdBootPhase,
    EmergencyDrillsPhase,
    FirstTrustPhase,
    LearningContractsPhase,
    OwnerRootPhase,
    PhaseExecutionResult,
    ValueLedgerPhase,
    VaultGenesisPhase,
    create_phase_executor,
)

# State management
from .state import (
    CeremonyPhase,
    CeremonyState,
    CeremonyStateManager,
    CeremonyStatus,
    PhaseRecord,
    PhaseResult,
    create_state_manager,
)

__all__ = [
    # State
    "CeremonyState",
    "CeremonyStateManager",
    "CeremonyPhase",
    "CeremonyStatus",
    "PhaseResult",
    "PhaseRecord",
    "create_state_manager",
    # Phases
    "CeremonyPhaseExecutor",
    "PhaseExecutionResult",
    "ColdBootPhase",
    "OwnerRootPhase",
    "BoundaryInitPhase",
    "VaultGenesisPhase",
    "LearningContractsPhase",
    "ValueLedgerPhase",
    "FirstTrustPhase",
    "EmergencyDrillsPhase",
    "create_phase_executor",
    # Orchestrator
    "CeremonyOrchestrator",
    "CeremonyConfig",
    "CeremonyEvent",
    "create_orchestrator",
    "run_ceremony",
    # CLI
    "CeremonyCLI",
    "cli_main",
]
