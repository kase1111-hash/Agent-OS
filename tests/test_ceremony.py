"""
Tests for the Bring-Home Ceremony Module (UC-014).

Tests cover:
- Ceremony state management
- All 8 ceremony phases
- Ceremony orchestration
- Phase prerequisites and ordering
- Emergency drill handling
"""

import pytest
from datetime import datetime, timedelta
from pathlib import Path
import tempfile

from src.ceremony import (
    # State
    CeremonyState,
    CeremonyStateManager,
    CeremonyPhase,
    CeremonyStatus,
    PhaseResult,
    PhaseRecord,
    create_state_manager,
    # Phases
    CeremonyPhaseExecutor,
    PhaseExecutionResult,
    ColdBootPhase,
    OwnerRootPhase,
    BoundaryInitPhase,
    VaultGenesisPhase,
    LearningContractsPhase,
    ValueLedgerPhase,
    FirstTrustPhase,
    EmergencyDrillsPhase,
    create_phase_executor,
    # Orchestrator
    CeremonyOrchestrator,
    CeremonyConfig,
    CeremonyEvent,
    create_orchestrator,
    run_ceremony,
    # CLI
    CeremonyCLI,
)


# =============================================================================
# Ceremony State Tests
# =============================================================================

class TestCeremonyPhase:
    """Tests for CeremonyPhase enum."""

    def test_phase_numbers(self):
        """Test phase numbers are correct."""
        assert CeremonyPhase.NOT_STARTED.phase_number == 0
        assert CeremonyPhase.PHASE_I_COLD_BOOT.phase_number == 1
        assert CeremonyPhase.PHASE_II_OWNER_ROOT.phase_number == 2
        assert CeremonyPhase.PHASE_VIII_EMERGENCY_DRILLS.phase_number == 8
        assert CeremonyPhase.COMPLETED.phase_number == 9

    def test_display_names(self):
        """Test phase display names."""
        assert "Cold Boot" in CeremonyPhase.PHASE_I_COLD_BOOT.display_name
        assert "Owner Root" in CeremonyPhase.PHASE_II_OWNER_ROOT.display_name
        assert "Boundary" in CeremonyPhase.PHASE_III_BOUNDARY_INIT.display_name


class TestCeremonyState:
    """Tests for CeremonyState."""

    def test_create_state(self):
        """Test creating ceremony state."""
        state = CeremonyState(ceremony_id="TEST-001")
        assert state.ceremony_id == "TEST-001"
        assert state.current_phase == CeremonyPhase.NOT_STARTED
        assert state.status == CeremonyStatus.IN_PROGRESS
        assert not state.is_complete

    def test_progress_calculation(self):
        """Test progress percentage calculation."""
        state = CeremonyState(ceremony_id="TEST-001")
        assert state.progress_percent == 0.0

        state.current_phase = CeremonyPhase.COMPLETED
        state.status = CeremonyStatus.COMPLETED
        assert state.progress_percent == 100.0

    def test_record_phase_start(self):
        """Test recording phase start."""
        state = CeremonyState(ceremony_id="TEST-001")
        record = state.record_phase_start(CeremonyPhase.PHASE_I_COLD_BOOT)

        assert record.phase == CeremonyPhase.PHASE_I_COLD_BOOT
        assert record.attempts == 1
        assert state.current_phase == CeremonyPhase.PHASE_I_COLD_BOOT

    def test_record_phase_complete(self):
        """Test recording phase completion."""
        state = CeremonyState(ceremony_id="TEST-001")
        state.record_phase_start(CeremonyPhase.PHASE_I_COLD_BOOT)
        state.record_phase_complete(
            phase=CeremonyPhase.PHASE_I_COLD_BOOT,
            result=PhaseResult.SUCCESS,
            verification_hash="abc123",
        )

        record = state.get_phase_record(CeremonyPhase.PHASE_I_COLD_BOOT)
        assert record.result == PhaseResult.SUCCESS
        assert record.verification_hash == "abc123"
        assert record.completed_at is not None

    def test_advance_to_next_phase(self):
        """Test advancing through phases."""
        state = CeremonyState(ceremony_id="TEST-001")

        state.advance_to_next_phase()
        assert state.current_phase == CeremonyPhase.PHASE_I_COLD_BOOT

        state.advance_to_next_phase()
        assert state.current_phase == CeremonyPhase.PHASE_II_OWNER_ROOT

    def test_reset_to_phase(self):
        """Test resetting to earlier phase."""
        state = CeremonyState(ceremony_id="TEST-001")
        state.current_phase = CeremonyPhase.PHASE_V_LEARNING_CONTRACTS

        state.reset_to_phase(CeremonyPhase.PHASE_I_COLD_BOOT)

        assert state.current_phase == CeremonyPhase.PHASE_I_COLD_BOOT
        assert state.status == CeremonyStatus.IN_PROGRESS

    def test_serialization(self):
        """Test state serialization and deserialization."""
        state = CeremonyState(
            ceremony_id="TEST-001",
            owner_id="OWNER-123",
        )
        state.record_phase_start(CeremonyPhase.PHASE_I_COLD_BOOT)
        state.record_phase_complete(
            CeremonyPhase.PHASE_I_COLD_BOOT,
            PhaseResult.SUCCESS,
        )

        # Serialize
        data = state.to_dict()

        # Deserialize
        restored = CeremonyState.from_dict(data)

        assert restored.ceremony_id == state.ceremony_id
        assert restored.owner_id == state.owner_id
        assert len(restored.phase_records) == 1


class TestCeremonyStateManager:
    """Tests for CeremonyStateManager."""

    def test_create_manager(self):
        """Test creating state manager."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = create_state_manager(Path(tmpdir))
            assert manager is not None

    def test_create_ceremony(self):
        """Test creating a new ceremony."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = create_state_manager(Path(tmpdir))
            state = manager.create_ceremony("owner-123")

            assert state.ceremony_id.startswith("CEREMONY-")
            assert state.owner_id == "owner-123"

    def test_save_and_load(self):
        """Test saving and loading state."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = create_state_manager(Path(tmpdir))

            # Create and save
            state = manager.create_ceremony()
            state.owner_id = "test-owner"
            manager.save_state(state)

            # Load
            loaded = manager.load_state()
            assert loaded is not None
            assert loaded.ceremony_id == state.ceremony_id
            assert loaded.owner_id == "test-owner"

    def test_has_ceremony(self):
        """Test checking for existing ceremony."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = create_state_manager(Path(tmpdir))

            assert not manager.has_ceremony()

            manager.create_ceremony()

            assert manager.has_ceremony()

    def test_clear_ceremony(self):
        """Test clearing ceremony."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = create_state_manager(Path(tmpdir))
            manager.create_ceremony()

            assert manager.has_ceremony()

            manager.clear_ceremony()

            assert not manager.has_ceremony()


# =============================================================================
# Phase Executor Tests
# =============================================================================

class TestPhaseExecutors:
    """Tests for individual phase executors."""

    def _create_state(self) -> CeremonyState:
        """Create a test ceremony state."""
        state = CeremonyState(ceremony_id="TEST-001")
        return state

    def _get_config(self) -> dict:
        """Get test configuration."""
        return {
            "simulate_offline": True,
            "simulate_boundary": True,
            "simulate_processes": True,
            "simulate_drills": True,
            "simulate_tripwire_test": True,
            "enable_hardware_binding": False,
            "vault_path": Path("/tmp/test-vault"),
        }

    def test_cold_boot_phase(self):
        """Test Phase I: Cold Boot."""
        state = self._create_state()
        config = self._get_config()

        executor = ColdBootPhase(state, config)

        assert executor.phase == CeremonyPhase.PHASE_I_COLD_BOOT
        assert len(executor.prerequisites) == 0

        result = executor.execute()
        assert result.success
        assert result.verification_hash is not None

    def test_owner_root_phase(self):
        """Test Phase II: Owner Root."""
        state = self._create_state()
        config = self._get_config()

        # Complete prerequisite
        state.record_phase_start(CeremonyPhase.PHASE_I_COLD_BOOT)
        state.record_phase_complete(CeremonyPhase.PHASE_I_COLD_BOOT, PhaseResult.SUCCESS)

        executor = OwnerRootPhase(state, config)

        assert executor.phase == CeremonyPhase.PHASE_II_OWNER_ROOT
        assert CeremonyPhase.PHASE_I_COLD_BOOT in executor.prerequisites

        result = executor.execute()
        assert result.success
        assert "owner_key_hash" in result.data
        assert "backup_phrase" in result.data
        assert state.owner_key_hash is not None
        assert state.owner_id is not None

    def test_owner_root_backup_phrase(self):
        """Test that backup phrase is generated."""
        state = self._create_state()
        config = self._get_config()

        state.record_phase_start(CeremonyPhase.PHASE_I_COLD_BOOT)
        state.record_phase_complete(CeremonyPhase.PHASE_I_COLD_BOOT, PhaseResult.SUCCESS)

        executor = OwnerRootPhase(state, config)
        result = executor.execute()

        backup_phrase = result.data.get("backup_phrase", "")
        words = backup_phrase.split()

        assert len(words) == 24  # BIP39-style 24 words

    def test_boundary_init_phase(self):
        """Test Phase III: Boundary Init."""
        state = self._create_state()
        config = self._get_config()

        # Complete prerequisites
        state.record_phase_start(CeremonyPhase.PHASE_I_COLD_BOOT)
        state.record_phase_complete(CeremonyPhase.PHASE_I_COLD_BOOT, PhaseResult.SUCCESS)
        state.record_phase_start(CeremonyPhase.PHASE_II_OWNER_ROOT)
        state.record_phase_complete(CeremonyPhase.PHASE_II_OWNER_ROOT, PhaseResult.SUCCESS)
        state.owner_key_hash = "test_hash"

        executor = BoundaryInitPhase(state, config)
        result = executor.execute()

        assert result.success
        assert result.data.get("default_mode") == "restricted"
        assert "tripwires_enabled" in result.data
        assert state.boundary_verified

    def test_vault_genesis_phase(self):
        """Test Phase IV: Vault Genesis."""
        state = self._create_state()
        config = self._get_config()

        # Complete prerequisites
        for phase in [
            CeremonyPhase.PHASE_I_COLD_BOOT,
            CeremonyPhase.PHASE_II_OWNER_ROOT,
            CeremonyPhase.PHASE_III_BOUNDARY_INIT,
        ]:
            state.record_phase_start(phase)
            state.record_phase_complete(phase, PhaseResult.SUCCESS)

        state.owner_id = "OWNER-123"
        state.ceremony_id = "TEST-001"

        executor = VaultGenesisPhase(state, config)
        result = executor.execute()

        assert result.success
        assert "vault_id" in result.data
        assert "encryption_profiles" in result.data
        assert state.vault_id is not None

    def test_learning_contracts_phase(self):
        """Test Phase V: Learning Contracts."""
        state = self._create_state()
        config = self._get_config()

        # Complete prerequisites
        for phase in [
            CeremonyPhase.PHASE_I_COLD_BOOT,
            CeremonyPhase.PHASE_II_OWNER_ROOT,
            CeremonyPhase.PHASE_III_BOUNDARY_INIT,
            CeremonyPhase.PHASE_IV_VAULT_GENESIS,
        ]:
            state.record_phase_start(phase)
            state.record_phase_complete(phase, PhaseResult.SUCCESS)

        executor = LearningContractsPhase(state, config)
        result = executor.execute()

        assert result.success
        assert "default_contract" in result.data
        assert "prohibited_domains_count" in result.data
        assert state.contracts_initialized

    def test_value_ledger_phase(self):
        """Test Phase VI: Value Ledger."""
        state = self._create_state()
        config = self._get_config()

        # Complete prerequisites
        for phase in [
            CeremonyPhase.PHASE_I_COLD_BOOT,
            CeremonyPhase.PHASE_II_OWNER_ROOT,
            CeremonyPhase.PHASE_III_BOUNDARY_INIT,
            CeremonyPhase.PHASE_IV_VAULT_GENESIS,
            CeremonyPhase.PHASE_V_LEARNING_CONTRACTS,
        ]:
            state.record_phase_start(phase)
            state.record_phase_complete(phase, PhaseResult.SUCCESS)

        state.owner_id = "OWNER-123"
        state.ceremony_id = "TEST-001"

        executor = ValueLedgerPhase(state, config)
        result = executor.execute()

        assert result.success
        assert "ledger_id" in result.data
        assert "genesis_hash" in result.data
        assert state.ledger_initialized

    def test_first_trust_phase(self):
        """Test Phase VII: First Trust."""
        state = self._create_state()
        config = self._get_config()

        # Complete prerequisites
        for phase in [
            CeremonyPhase.PHASE_I_COLD_BOOT,
            CeremonyPhase.PHASE_II_OWNER_ROOT,
            CeremonyPhase.PHASE_III_BOUNDARY_INIT,
            CeremonyPhase.PHASE_IV_VAULT_GENESIS,
            CeremonyPhase.PHASE_V_LEARNING_CONTRACTS,
            CeremonyPhase.PHASE_VI_VALUE_LEDGER,
        ]:
            state.record_phase_start(phase)
            state.record_phase_complete(phase, PhaseResult.SUCCESS)

        executor = FirstTrustPhase(state, config)
        result = executor.execute()

        assert result.success
        assert result.data.get("boundary_mode") == "trusted"
        assert result.data.get("agent_os_enabled") == True
        assert "first_task" in result.data

    def test_emergency_drills_phase(self):
        """Test Phase VIII: Emergency Drills."""
        state = self._create_state()
        config = self._get_config()

        # Complete prerequisites
        for phase in [
            CeremonyPhase.PHASE_I_COLD_BOOT,
            CeremonyPhase.PHASE_II_OWNER_ROOT,
            CeremonyPhase.PHASE_III_BOUNDARY_INIT,
            CeremonyPhase.PHASE_IV_VAULT_GENESIS,
            CeremonyPhase.PHASE_V_LEARNING_CONTRACTS,
            CeremonyPhase.PHASE_VI_VALUE_LEDGER,
            CeremonyPhase.PHASE_VII_FIRST_TRUST,
        ]:
            state.record_phase_start(phase)
            state.record_phase_complete(phase, PhaseResult.SUCCESS)

        executor = EmergencyDrillsPhase(state, config)
        result = executor.execute()

        assert result.success
        assert result.data.get("drills_passed") == 3
        assert result.data.get("drills_total") == 3
        assert state.drills_passed

    def test_create_phase_executor(self):
        """Test phase executor factory."""
        state = self._create_state()
        config = self._get_config()

        executor = create_phase_executor(
            CeremonyPhase.PHASE_I_COLD_BOOT,
            state,
            config,
        )
        assert isinstance(executor, ColdBootPhase)

        executor = create_phase_executor(
            CeremonyPhase.PHASE_VIII_EMERGENCY_DRILLS,
            state,
            config,
        )
        assert isinstance(executor, EmergencyDrillsPhase)

        # Non-executable phases
        executor = create_phase_executor(
            CeremonyPhase.COMPLETED,
            state,
            config,
        )
        assert executor is None

    def test_prerequisite_check(self):
        """Test prerequisite checking."""
        state = self._create_state()
        config = self._get_config()

        # Try Phase II without Phase I
        executor = OwnerRootPhase(state, config)
        met, missing = executor.check_prerequisites()

        assert not met
        assert len(missing) > 0


# =============================================================================
# Ceremony Orchestrator Tests
# =============================================================================

class TestCeremonyOrchestrator:
    """Tests for CeremonyOrchestrator."""

    def _get_config(self) -> CeremonyConfig:
        """Get test configuration."""
        with tempfile.TemporaryDirectory() as tmpdir:
            return CeremonyConfig(
                state_dir=Path(tmpdir) / "ceremony",
                vault_path=Path(tmpdir) / "vault",
                simulate_offline=True,
                simulate_boundary=True,
                simulate_processes=True,
                simulate_drills=True,
            )

    def test_create_orchestrator(self):
        """Test creating orchestrator."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = CeremonyConfig(state_dir=Path(tmpdir))
            orchestrator = create_orchestrator(config)
            assert orchestrator is not None

    def test_start_ceremony(self):
        """Test starting a ceremony."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = CeremonyConfig(state_dir=Path(tmpdir))
            orchestrator = create_orchestrator(config)

            state = orchestrator.start_ceremony("owner-123")

            assert state is not None
            assert state.owner_id == "owner-123"
            assert orchestrator.state is not None

    def test_execute_phase(self):
        """Test executing a phase."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = CeremonyConfig(state_dir=Path(tmpdir))
            orchestrator = create_orchestrator(config)
            orchestrator.start_ceremony()

            result = orchestrator.execute_phase(CeremonyPhase.PHASE_I_COLD_BOOT)

            assert result.success
            assert result.phase == CeremonyPhase.PHASE_I_COLD_BOOT

    def test_execute_current_phase(self):
        """Test executing current phase."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = CeremonyConfig(state_dir=Path(tmpdir))
            orchestrator = create_orchestrator(config)
            orchestrator.start_ceremony()

            result = orchestrator.execute_current_phase()

            assert result.success
            assert result.phase == CeremonyPhase.PHASE_I_COLD_BOOT

    def test_advance_phase(self):
        """Test advancing through phases."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = CeremonyConfig(state_dir=Path(tmpdir))
            orchestrator = create_orchestrator(config)
            orchestrator.start_ceremony()

            # Execute Phase I
            orchestrator.execute_phase(CeremonyPhase.PHASE_I_COLD_BOOT)

            # Advance
            new_phase = orchestrator.advance_phase()

            assert new_phase == CeremonyPhase.PHASE_II_OWNER_ROOT
            assert orchestrator.current_phase == CeremonyPhase.PHASE_II_OWNER_ROOT

    def test_run_all_phases(self):
        """Test running all phases."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = CeremonyConfig(state_dir=Path(tmpdir))
            orchestrator = create_orchestrator(config)
            orchestrator.start_ceremony()

            results = orchestrator.run_all_phases()

            assert len(results) == 8
            assert all(r.success for r in results.values())
            assert orchestrator.is_complete

    def test_verify_ceremony(self):
        """Test ceremony verification."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = CeremonyConfig(state_dir=Path(tmpdir))
            orchestrator = create_orchestrator(config)
            orchestrator.start_ceremony()
            orchestrator.run_all_phases()

            is_valid, issues = orchestrator.verify_ceremony()

            assert is_valid
            assert len(issues) == 0

    def test_verify_incomplete_ceremony(self):
        """Test verifying incomplete ceremony."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = CeremonyConfig(state_dir=Path(tmpdir))
            orchestrator = create_orchestrator(config)
            orchestrator.start_ceremony()

            # Only execute Phase I
            orchestrator.execute_phase(CeremonyPhase.PHASE_I_COLD_BOOT)

            is_valid, issues = orchestrator.verify_ceremony()

            assert not is_valid
            assert len(issues) > 0

    def test_get_status(self):
        """Test getting ceremony status."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = CeremonyConfig(state_dir=Path(tmpdir))
            orchestrator = create_orchestrator(config)
            orchestrator.start_ceremony()
            orchestrator.execute_phase(CeremonyPhase.PHASE_I_COLD_BOOT)

            status = orchestrator.get_status()

            assert status["status"] == "in_progress"
            assert "ceremony_id" in status
            assert status["phases_completed"] == 1

    def test_get_phase_status(self):
        """Test getting phase status."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = CeremonyConfig(state_dir=Path(tmpdir))
            orchestrator = create_orchestrator(config)
            orchestrator.start_ceremony()
            orchestrator.execute_phase(CeremonyPhase.PHASE_I_COLD_BOOT)

            status = orchestrator.get_phase_status(CeremonyPhase.PHASE_I_COLD_BOOT)

            assert status["status"] == "success"
            assert "verification_hash" in status

    def test_reset_ceremony(self):
        """Test resetting ceremony."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = CeremonyConfig(state_dir=Path(tmpdir))
            orchestrator = create_orchestrator(config)
            orchestrator.start_ceremony()
            orchestrator.execute_phase(CeremonyPhase.PHASE_I_COLD_BOOT)

            orchestrator.reset_ceremony()

            assert orchestrator.state is None
            assert not orchestrator._state_manager.has_ceremony()

    def test_event_handling(self):
        """Test event handling."""
        events = []

        def handler(event):
            events.append(event)

        with tempfile.TemporaryDirectory() as tmpdir:
            config = CeremonyConfig(state_dir=Path(tmpdir))
            orchestrator = create_orchestrator(config)
            orchestrator.on_event(handler)

            orchestrator.start_ceremony()
            orchestrator.execute_phase(CeremonyPhase.PHASE_I_COLD_BOOT)

            assert len(events) > 0
            assert any(e.event_type == "ceremony_started" for e in events)

    def test_phase_callbacks(self):
        """Test phase-specific callbacks."""
        callback_results = []

        def phase_callback(result):
            callback_results.append(result)

        with tempfile.TemporaryDirectory() as tmpdir:
            config = CeremonyConfig(state_dir=Path(tmpdir))
            orchestrator = create_orchestrator(config)
            orchestrator.on_phase_complete(
                CeremonyPhase.PHASE_I_COLD_BOOT,
                phase_callback,
            )

            orchestrator.start_ceremony()
            orchestrator.execute_phase(CeremonyPhase.PHASE_I_COLD_BOOT)

            assert len(callback_results) == 1
            assert callback_results[0].success

    def test_resume_ceremony(self):
        """Test resuming an existing ceremony."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = CeremonyConfig(state_dir=Path(tmpdir))

            # Start and execute some phases
            orchestrator1 = create_orchestrator(config)
            orchestrator1.start_ceremony("owner-123")
            orchestrator1.execute_phase(CeremonyPhase.PHASE_I_COLD_BOOT)
            orchestrator1.advance_phase()

            # Create new orchestrator (simulating restart)
            orchestrator2 = create_orchestrator(config)
            state = orchestrator2.start_ceremony()

            # Should resume from Phase II
            assert state.current_phase == CeremonyPhase.PHASE_II_OWNER_ROOT
            assert state.owner_id == "owner-123"


# =============================================================================
# Run Ceremony Integration Test
# =============================================================================

class TestRunCeremony:
    """Integration tests for run_ceremony function."""

    def test_run_ceremony_success(self):
        """Test running complete ceremony."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = CeremonyConfig(
                state_dir=Path(tmpdir),
                simulate_offline=True,
                simulate_boundary=True,
                simulate_processes=True,
                simulate_drills=True,
            )

            success, results = run_ceremony(config)

            assert success
            assert results["status"]["status"] == "completed"
            assert len(results["phases"]) == 8


# =============================================================================
# CLI Tests
# =============================================================================

class TestCeremonyCLI:
    """Tests for CeremonyCLI."""

    def test_create_cli(self):
        """Test creating CLI."""
        cli = CeremonyCLI(use_colors=False)
        assert cli is not None

    def test_run_automated(self):
        """Test running automated ceremony."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = CeremonyConfig(
                state_dir=Path(tmpdir),
                simulate_offline=True,
                simulate_boundary=True,
                simulate_processes=True,
                simulate_drills=True,
            )

            cli = CeremonyCLI(use_colors=False)
            success = cli.run_automated(config)

            assert success

    def test_show_status(self):
        """Test showing status."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = CeremonyConfig(state_dir=Path(tmpdir))

            # Run ceremony first
            cli = CeremonyCLI(use_colors=False)
            cli.run_automated(config)

            # Show status (should not raise)
            cli.show_status(config)

    def test_verify(self):
        """Test verification."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = CeremonyConfig(state_dir=Path(tmpdir))

            cli = CeremonyCLI(use_colors=False)
            cli.run_automated(config)

            success = cli.verify(config)
            assert success

    def test_reset(self):
        """Test reset."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = CeremonyConfig(state_dir=Path(tmpdir))

            cli = CeremonyCLI(use_colors=False)
            cli.run_automated(config)
            cli.reset(config)

            # Verify reset
            assert not cli.orchestrator._state_manager.has_ceremony()


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================

class TestCeremonyEdgeCases:
    """Tests for edge cases and error handling."""

    def test_phase_without_ceremony(self):
        """Test executing phase without starting ceremony."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = CeremonyConfig(state_dir=Path(tmpdir))
            orchestrator = create_orchestrator(config)

            with pytest.raises(RuntimeError):
                orchestrator.execute_phase(CeremonyPhase.PHASE_I_COLD_BOOT)

    def test_advance_without_completing(self):
        """Test advancing without completing current phase."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = CeremonyConfig(state_dir=Path(tmpdir))
            orchestrator = create_orchestrator(config)
            orchestrator.start_ceremony()

            # Advance from NOT_STARTED to Phase I (allowed)
            orchestrator.advance_phase()

            # Now try to advance from Phase I without executing it (should fail)
            with pytest.raises(RuntimeError):
                orchestrator.advance_phase()

    def test_execute_phase_missing_prerequisites(self):
        """Test executing phase with missing prerequisites."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = CeremonyConfig(state_dir=Path(tmpdir))
            orchestrator = create_orchestrator(config)
            orchestrator.start_ceremony()

            # Try to execute Phase II without Phase I
            result = orchestrator.execute_phase(CeremonyPhase.PHASE_II_OWNER_ROOT)

            assert not result.success
            assert "Prerequisites not met" in result.message

    def test_completed_ceremony_execution(self):
        """Test executing on completed ceremony."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = CeremonyConfig(state_dir=Path(tmpdir))
            orchestrator = create_orchestrator(config)
            orchestrator.start_ceremony()
            orchestrator.run_all_phases()

            # Try to execute another phase
            result = orchestrator.execute_current_phase()

            assert result.success
            assert "already complete" in result.message.lower()

    def test_retry_failed_phase(self):
        """Test retrying a phase."""
        state = CeremonyState(ceremony_id="TEST-001")

        # First attempt
        record = state.record_phase_start(CeremonyPhase.PHASE_I_COLD_BOOT)
        assert record.attempts == 1

        # Mark as failed
        state.record_phase_complete(
            CeremonyPhase.PHASE_I_COLD_BOOT,
            PhaseResult.FAILED,
        )

        # Retry
        record = state.record_phase_start(CeremonyPhase.PHASE_I_COLD_BOOT)
        assert record.attempts == 2

    def test_progress_tracking(self):
        """Test progress percentage through phases."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = CeremonyConfig(state_dir=Path(tmpdir))
            orchestrator = create_orchestrator(config)
            orchestrator.start_ceremony()

            assert orchestrator.progress == 0.0

            # Complete Phase I
            orchestrator.execute_phase(CeremonyPhase.PHASE_I_COLD_BOOT)
            orchestrator.advance_phase()

            # Progress should be 12.5% (1/8 phases)
            assert orchestrator.progress == pytest.approx(12.5)


# =============================================================================
# Acceptance Criteria Tests
# =============================================================================

class TestAcceptanceCriteria:
    """Tests verifying acceptance criteria are met."""

    def test_all_phases_functional(self):
        """Verify all 8 phases are functional."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = CeremonyConfig(state_dir=Path(tmpdir))
            orchestrator = create_orchestrator(config)
            orchestrator.start_ceremony()

            results = orchestrator.run_all_phases()

            # All 8 phases should execute
            assert len(results) == 8

            # All should succeed
            for phase, result in results.items():
                assert result.success, f"{phase.display_name} failed"

    def test_owner_key_bound(self):
        """Verify owner key is established."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = CeremonyConfig(state_dir=Path(tmpdir))
            orchestrator = create_orchestrator(config)
            orchestrator.start_ceremony()
            orchestrator.run_all_phases()

            assert orchestrator.state.owner_key_hash is not None
            assert orchestrator.state.owner_id is not None

    def test_emergency_drills_pass(self):
        """Verify emergency drills are executed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = CeremonyConfig(state_dir=Path(tmpdir))
            orchestrator = create_orchestrator(config)
            orchestrator.start_ceremony()
            orchestrator.run_all_phases()

            assert orchestrator.state.drills_passed

    def test_vault_initialized(self):
        """Verify vault is initialized."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = CeremonyConfig(state_dir=Path(tmpdir))
            orchestrator = create_orchestrator(config)
            orchestrator.start_ceremony()
            orchestrator.run_all_phases()

            assert orchestrator.state.vault_id is not None

    def test_contracts_initialized(self):
        """Verify learning contracts are initialized."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = CeremonyConfig(state_dir=Path(tmpdir))
            orchestrator = create_orchestrator(config)
            orchestrator.start_ceremony()
            orchestrator.run_all_phases()

            assert orchestrator.state.contracts_initialized

    def test_boundary_verified(self):
        """Verify boundary is configured."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = CeremonyConfig(state_dir=Path(tmpdir))
            orchestrator = create_orchestrator(config)
            orchestrator.start_ceremony()
            orchestrator.run_all_phases()

            assert orchestrator.state.boundary_verified
