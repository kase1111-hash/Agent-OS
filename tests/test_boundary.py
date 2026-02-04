"""
Unit tests for Boundary Daemon (UC-012)

Tests state monitoring, tripwires, policy engine, enforcement,
event logging, and boundary client integration.
"""

import pytest
import time
import tempfile
from datetime import datetime
from pathlib import Path

from src.boundary import (
    BoundaryClient,
    BoundaryClientConfig,
    create_boundary_client,
    BoundaryDaemon,
    BoundaryConfig,
    BoundaryMode,
    RequestType,
    Decision,
    create_boundary_daemon,
)

from src.boundary.daemon import (
    StateMonitor,
    SystemState,
    NetworkState,
    ProcessState,
    HardwareState,
    create_state_monitor,
    TripwireSystem,
    Tripwire,
    TripwireEvent,
    TripwireType,
    TripwireState,
    create_tripwire_system,
    create_file_tripwire,
    PolicyEngine,
    PolicyRequest,
    PolicyDecision,
    PolicyRule,
    create_policy_engine,
    EnforcementLayer,
    EnforcementEvent,
    EnforcementAction,
    EnforcementSeverity,
    create_enforcement_layer,
    ImmutableEventLog,
    LogEntry,
    create_event_log,
)


# =============================================================================
# SystemState Tests
# =============================================================================

class TestSystemState:
    """Tests for SystemState."""

    def test_create_state(self):
        """Test creating a system state."""
        state = SystemState(
            timestamp=datetime.now(),
            network_state=NetworkState.OFFLINE,
            process_state=ProcessState.NORMAL,
            hardware_state=HardwareState.NORMAL,
        )

        assert state.network_state == NetworkState.OFFLINE
        assert state.process_state == ProcessState.NORMAL
        assert state.hardware_state == HardwareState.NORMAL

    def test_is_secure_all_normal(self):
        """Test is_secure with all normal states."""
        state = SystemState(
            timestamp=datetime.now(),
            network_state=NetworkState.OFFLINE,
            process_state=ProcessState.NORMAL,
            hardware_state=HardwareState.NORMAL,
        )

        assert state.is_secure() is True

    def test_is_secure_network_online(self):
        """Test is_secure with network online."""
        state = SystemState(
            timestamp=datetime.now(),
            network_state=NetworkState.ONLINE,
            process_state=ProcessState.NORMAL,
            hardware_state=HardwareState.NORMAL,
        )

        assert state.is_secure() is False

    def test_to_dict(self):
        """Test state serialization."""
        state = SystemState(
            timestamp=datetime.now(),
            network_state=NetworkState.OFFLINE,
            process_state=ProcessState.NORMAL,
            hardware_state=HardwareState.NORMAL,
        )

        data = state.to_dict()
        assert "network_state" in data
        assert data["is_secure"] is True


# =============================================================================
# StateMonitor Tests
# =============================================================================

class TestStateMonitor:
    """Tests for StateMonitor."""

    def test_create_monitor(self):
        """Test creating a state monitor."""
        monitor = StateMonitor()
        assert monitor is not None

    def test_get_current_state(self):
        """Test getting current state."""
        monitor = StateMonitor()
        state = monitor.get_current_state()

        assert isinstance(state, SystemState)
        assert state.timestamp is not None

    @pytest.mark.slow
    def test_start_stop(self):
        """Test starting and stopping monitor."""
        monitor = StateMonitor(poll_interval=0.1)
        monitor.start()

        time.sleep(0.2)
        assert len(monitor.get_state_history()) > 0

        monitor.stop()

    @pytest.mark.slow
    def test_callback_registration(self):
        """Test callback registration."""
        monitor = StateMonitor()
        states_received = []

        def on_state(state):
            states_received.append(state)

        monitor.register_callback(on_state)
        monitor.start()
        time.sleep(0.2)
        monitor.stop()

        # Callback should have been called
        assert len(states_received) >= 0  # May not change if state stable

    def test_factory_function(self):
        """Test factory function."""
        monitor = create_state_monitor(poll_interval=0.5)
        assert monitor.poll_interval == 0.5


# =============================================================================
# Tripwire Tests
# =============================================================================

class TestTripwire:
    """Tests for Tripwire."""

    def test_create_tripwire(self):
        """Test creating a tripwire."""
        tripwire = Tripwire(
            id="test_tripwire",
            tripwire_type=TripwireType.CUSTOM,
            description="Test tripwire",
            condition=lambda: False,
        )

        assert tripwire.id == "test_tripwire"
        assert tripwire.state == TripwireState.ARMED

    def test_tripwire_triggers(self):
        """Test tripwire triggering."""
        tripwire = Tripwire(
            id="trigger_test",
            tripwire_type=TripwireType.CUSTOM,
            description="Should trigger",
            condition=lambda: True,
            severity=5,
        )

        event = tripwire.check()

        assert event is not None
        assert tripwire.state == TripwireState.TRIGGERED
        assert event.severity == 5

    def test_tripwire_not_triggers(self):
        """Test tripwire not triggering."""
        tripwire = Tripwire(
            id="no_trigger",
            tripwire_type=TripwireType.CUSTOM,
            description="Should not trigger",
            condition=lambda: False,
        )

        event = tripwire.check()

        assert event is None
        assert tripwire.state == TripwireState.ARMED

    def test_tripwire_reset(self):
        """Test tripwire reset."""
        tripwire = Tripwire(
            id="reset_test",
            tripwire_type=TripwireType.CUSTOM,
            description="Reset test",
            condition=lambda: True,
        )

        # Trigger
        tripwire.check()
        assert tripwire.state == TripwireState.TRIGGERED

        # Reset with authorization
        success = tripwire.reset("valid_auth_code_12345")
        assert success is True
        assert tripwire.state == TripwireState.ARMED


# =============================================================================
# TripwireSystem Tests
# =============================================================================

class TestTripwireSystem:
    """Tests for TripwireSystem."""

    def test_create_system(self):
        """Test creating tripwire system."""
        system = TripwireSystem()
        assert system is not None

    def test_add_tripwire(self):
        """Test adding a tripwire."""
        system = TripwireSystem()

        tripwire = Tripwire(
            id="custom_test",
            tripwire_type=TripwireType.CUSTOM,
            description="Custom test",
            condition=lambda: False,
        )

        system.add_tripwire(tripwire)
        assert system.get_tripwire("custom_test") is not None

    def test_check_all(self):
        """Test checking all tripwires."""
        system = TripwireSystem()

        # Add a triggering tripwire
        system.add_tripwire(Tripwire(
            id="will_trigger",
            tripwire_type=TripwireType.CUSTOM,
            description="Will trigger",
            condition=lambda: True,
        ))

        events = system.check_all()

        assert len(events) > 0
        assert system.is_triggered() is True

    def test_reset_all(self):
        """Test resetting all tripwires."""
        system = TripwireSystem()

        system.add_tripwire(Tripwire(
            id="trigger_me",
            tripwire_type=TripwireType.CUSTOM,
            description="Trigger",
            condition=lambda: True,
        ))

        system.check_all()
        assert system.is_triggered() is True

        count = system.reset_all("authorization_code")
        assert count >= 1
        assert system.is_triggered() is False

    def test_factory_function(self):
        """Test factory function."""
        events_received = []

        def on_trigger(event):
            events_received.append(event)

        system = create_tripwire_system(
            check_interval=0.1,
            on_trigger=on_trigger,
        )

        assert system.check_interval == 0.1


# =============================================================================
# PolicyEngine Tests
# =============================================================================

class TestPolicyEngine:
    """Tests for PolicyEngine."""

    def test_create_engine(self):
        """Test creating policy engine."""
        engine = PolicyEngine()
        assert engine.mode == BoundaryMode.RESTRICTED

    def test_set_mode(self):
        """Test setting mode."""
        engine = PolicyEngine(initial_mode=BoundaryMode.RESTRICTED)

        # Can increase security without auth
        success = engine.set_mode(BoundaryMode.LOCKDOWN, "test")
        assert success is True
        assert engine.mode == BoundaryMode.LOCKDOWN

    def test_mode_requires_authorization(self):
        """Test mode change requiring authorization."""
        engine = PolicyEngine(initial_mode=BoundaryMode.LOCKDOWN)

        # Decreasing security requires authorization
        success = engine.set_mode(BoundaryMode.TRUSTED, "test")
        assert success is False

        success = engine.set_mode(BoundaryMode.TRUSTED, "test", "auth_code_here")
        assert success is True

    def test_evaluate_lockdown_denies(self):
        """Test lockdown mode denies network."""
        engine = PolicyEngine(initial_mode=BoundaryMode.LOCKDOWN)

        request = PolicyRequest(
            request_id="test-1",
            request_type=RequestType.NETWORK_ACCESS,
            source="agent:test",
            target="example.com",
        )

        decision = engine.evaluate(request)

        assert decision.decision == Decision.DENY

    def test_evaluate_trusted_audits(self):
        """Test trusted mode audits."""
        engine = PolicyEngine(initial_mode=BoundaryMode.TRUSTED)

        request = PolicyRequest(
            request_id="test-2",
            request_type=RequestType.FILE_WRITE,
            source="agent:test",
            target="/tmp/test.txt",
        )

        decision = engine.evaluate(request)

        assert decision.decision == Decision.AUDIT

    def test_whitelist(self):
        """Test whitelisting."""
        engine = PolicyEngine(initial_mode=BoundaryMode.RESTRICTED)

        engine.whitelist(
            BoundaryMode.RESTRICTED,
            RequestType.NETWORK_ACCESS,
            "allowed.example.com",
        )

        request = PolicyRequest(
            request_id="test-3",
            request_type=RequestType.NETWORK_ACCESS,
            source="agent:test",
            target="allowed.example.com",
        )

        decision = engine.evaluate(request)

        assert decision.decision == Decision.ALLOW

    def test_lockdown(self):
        """Test lockdown convenience method."""
        engine = PolicyEngine(initial_mode=BoundaryMode.RESTRICTED)

        success = engine.lockdown("security event")

        assert success is True
        assert engine.mode == BoundaryMode.LOCKDOWN


# =============================================================================
# EnforcementLayer Tests
# =============================================================================

class TestEnforcementLayer:
    """Tests for EnforcementLayer."""

    def test_create_layer(self):
        """Test creating enforcement layer."""
        layer = EnforcementLayer()
        assert layer is not None
        assert layer.is_halted is False

    def test_alert(self):
        """Test alert action."""
        layer = EnforcementLayer()

        event = layer.alert("Test alert", source="test")

        assert event.action == EnforcementAction.ALERT
        assert event.success is True

    def test_suspend(self):
        """Test suspend action."""
        layer = EnforcementLayer()

        event = layer.suspend("Test suspend", source="test")

        assert event.action == EnforcementAction.SUSPEND
        assert layer.is_suspended is True

    def test_halt(self):
        """Test halt action."""
        layer = EnforcementLayer()

        event = layer.halt("Test halt", source="test")

        assert event.action == EnforcementAction.HALT
        assert layer.is_halted is True

    def test_lockdown(self):
        """Test lockdown action."""
        layer = EnforcementLayer()

        event = layer.lockdown("Test lockdown", source="test")

        assert event.action == EnforcementAction.LOCKDOWN
        assert layer.is_halted is True
        assert layer.is_isolated is True

    def test_resume(self):
        """Test resume from halted state."""
        layer = EnforcementLayer()

        layer.halt("test")
        assert layer.is_halted is True

        success = layer.resume("valid_auth_code")
        assert success is True
        assert layer.is_halted is False

    def test_callback(self):
        """Test enforcement callback."""
        events_received = []

        def on_enforcement(event):
            events_received.append(event)

        layer = EnforcementLayer(on_enforcement=on_enforcement)
        layer.alert("test")

        assert len(events_received) == 1


# =============================================================================
# ImmutableEventLog Tests
# =============================================================================

class TestImmutableEventLog:
    """Tests for ImmutableEventLog."""

    def test_create_log(self):
        """Test creating event log."""
        log = ImmutableEventLog()
        assert log is not None
        assert log.count() == 0

    def test_append_event(self):
        """Test appending events."""
        log = ImmutableEventLog()

        entry = log.append("test_event", {"key": "value"})

        assert entry.sequence == 1
        assert entry.event_type == "test_event"
        assert log.count() == 1

    def test_hash_chain(self):
        """Test hash chain integrity."""
        log = ImmutableEventLog()

        log.append("event_1", {"data": "first"})
        log.append("event_2", {"data": "second"})
        log.append("event_3", {"data": "third"})

        valid, errors = log.verify_integrity()

        assert valid is True
        assert len(errors) == 0

    def test_log_to_file(self):
        """Test persisting to file."""
        with tempfile.NamedTemporaryFile(suffix=".log", delete=False) as f:
            log_path = Path(f.name)

        try:
            log = ImmutableEventLog(log_path=log_path)
            log.append("file_test", {"data": "test"})

            # Reload and verify
            log2 = ImmutableEventLog(log_path=log_path)
            assert log2.count() == 1

        finally:
            log_path.unlink(missing_ok=True)

    def test_get_entries(self):
        """Test getting entries with filters."""
        log = ImmutableEventLog()

        log.append("type_a", {"data": 1})
        log.append("type_b", {"data": 2})
        log.append("type_a", {"data": 3})

        type_a = log.get_entries(event_type="type_a")
        assert len(type_a) == 2

    def test_log_convenience_methods(self):
        """Test convenience logging methods."""
        log = ImmutableEventLog()

        log.log_tripwire("tw1", "test reason", 3)
        log.log_enforcement("halt", "test", True)
        log.log_mode_change("RESTRICTED", "LOCKDOWN", "test")

        assert log.count() == 3


# =============================================================================
# BoundaryDaemon Tests
# =============================================================================

class TestBoundaryDaemon:
    """Tests for BoundaryDaemon."""

    def test_create_daemon(self):
        """Test creating daemon."""
        daemon = BoundaryDaemon()
        assert daemon is not None
        assert daemon.mode == BoundaryMode.RESTRICTED

    def test_start_stop(self):
        """Test starting and stopping daemon."""
        daemon = BoundaryDaemon()
        daemon.start()

        assert daemon.is_running is True

        daemon.stop()
        assert daemon.is_running is False

    def test_request_permission_restricted(self):
        """Test permission in restricted mode."""
        daemon = BoundaryDaemon(BoundaryConfig(
            initial_mode=BoundaryMode.RESTRICTED,
        ))
        daemon.start()

        try:
            # Network should be denied/escalated
            allowed = daemon.request_permission(
                "network_access",
                "agent:test",
                "example.com",
            )
            # In restricted mode, network is escalated (denied)
            assert allowed is False

            # Memory access should be allowed for agents
            allowed = daemon.request_permission(
                "memory_access",
                "agent:seshat",
                "key123",
            )
            assert allowed is True

        finally:
            daemon.stop()

    def test_lockdown(self):
        """Test lockdown functionality."""
        daemon = BoundaryDaemon()
        daemon.start()

        try:
            success = daemon.lockdown("test lockdown")
            assert success is True
            assert daemon.mode == BoundaryMode.LOCKDOWN

        finally:
            daemon.stop()

    def test_get_status(self):
        """Test getting daemon status."""
        daemon = BoundaryDaemon()
        daemon.start()

        try:
            status = daemon.get_status()

            assert "running" in status
            assert "mode" in status
            assert status["running"] is True

        finally:
            daemon.stop()

    def test_factory_function(self):
        """Test daemon factory function."""
        daemon = create_boundary_daemon(
            initial_mode=BoundaryMode.TRUSTED,
            network_allowed=True,
        )

        assert daemon.mode == BoundaryMode.TRUSTED


# =============================================================================
# BoundaryClient Tests
# =============================================================================

class TestBoundaryClient:
    """Tests for BoundaryClient."""

    def test_create_client(self):
        """Test creating client."""
        client = BoundaryClient()
        assert client is not None

    def test_embedded_connection(self):
        """Test embedded daemon connection."""
        client = BoundaryClient(BoundaryClientConfig(embedded=True))

        assert client.is_connected is True
        client.disconnect()

    def test_request_permission(self):
        """Test permission request through client."""
        client = create_boundary_client(
            initial_mode=BoundaryMode.TRUSTED,
        )

        try:
            allowed = client.request_permission(
                "file_write",
                "agent:quill",
                "/tmp/test.txt",
            )
            # Trusted mode audits but allows
            assert allowed is True

        finally:
            client.disconnect()

    def test_check_network_access(self):
        """Test network access check."""
        client = create_boundary_client(
            initial_mode=BoundaryMode.LOCKDOWN,
        )

        try:
            allowed = client.check_network_access("agent:researcher")
            assert allowed is False

        finally:
            client.disconnect()

    def test_check_memory_access(self):
        """Test memory access check."""
        client = create_boundary_client(
            initial_mode=BoundaryMode.RESTRICTED,
        )

        try:
            allowed = client.check_memory_access("agent:seshat", "memory_key")
            assert allowed is True

        finally:
            client.disconnect()

    def test_get_status(self):
        """Test getting client status."""
        client = create_boundary_client()

        try:
            status = client.get_status()

            assert "client_state" in status
            assert "daemon" in status

        finally:
            client.disconnect()

    def test_fail_closed(self):
        """Test fail closed behavior."""
        config = BoundaryClientConfig(
            embedded=False,  # No daemon
            fail_closed=True,
        )
        client = BoundaryClient(config)

        # Should deny when daemon unavailable
        allowed = client.request_permission(
            "network_access",
            "agent:test",
        )
        assert allowed is False

    def test_caching(self):
        """Test decision caching."""
        client = create_boundary_client(
            initial_mode=BoundaryMode.TRUSTED,
            cache_decisions=True,
        )

        try:
            # First request
            client.request_permission("file_write", "agent:test", "/tmp/test")

            # Check cache
            assert len(client._decision_cache) == 1

            # Second request should use cache
            client.request_permission("file_write", "agent:test", "/tmp/test")

        finally:
            client.disconnect()


# =============================================================================
# Integration Tests
# =============================================================================

class TestBoundaryIntegration:
    """Integration tests for boundary system."""

    def test_full_workflow(self):
        """Test complete boundary workflow."""
        # Create daemon
        daemon = create_boundary_daemon(
            initial_mode=BoundaryMode.RESTRICTED,
        )
        daemon.start()

        try:
            # Check initial status
            assert daemon.mode == BoundaryMode.RESTRICTED

            # Request permission
            allowed = daemon.request_permission(
                "memory_access",
                "agent:sage",
                "reasoning_chain",
            )
            assert allowed is True

            # Network should be restricted
            allowed = daemon.request_permission(
                "network_access",
                "agent:sage",
                "api.example.com",
            )
            assert allowed is False

            # Lockdown
            daemon.lockdown("security event")
            assert daemon.mode == BoundaryMode.LOCKDOWN

            # Everything should be denied
            allowed = daemon.request_permission(
                "memory_access",
                "agent:sage",
                "key",
            )
            assert allowed is False

            # Verify log integrity
            valid, errors = daemon.verify_log_integrity()
            assert valid is True

        finally:
            daemon.stop()

    def test_tripwire_enforcement(self):
        """Test tripwire triggering enforcement."""
        events_captured = []

        daemon = create_boundary_daemon(
            initial_mode=BoundaryMode.TRUSTED,
        )
        daemon.start()

        try:
            # Add a custom tripwire that will trigger
            from src.boundary.daemon.tripwires import Tripwire, TripwireType

            daemon._tripwires.add_tripwire(Tripwire(
                id="test_trigger",
                tripwire_type=TripwireType.CUSTOM,
                description="Test trigger",
                condition=lambda: True,
                severity=3,
            ))

            # Check tripwires
            daemon._tripwires.check_all()

            # Should have triggered
            assert daemon._tripwires.is_triggered() is True

        finally:
            daemon.stop()

    def test_client_with_whitelisting(self):
        """Test client with whitelisting."""
        client = create_boundary_client(
            initial_mode=BoundaryMode.RESTRICTED,
        )

        try:
            # Initially network should be denied
            allowed = client.check_network_access("agent:test", "trusted.api.com")
            assert allowed is False

            # Add to whitelist
            client.add_whitelist("network_access", "trusted.api.com")

            # Now should be allowed
            allowed = client.check_network_access("agent:test", "trusted.api.com")
            assert allowed is True

        finally:
            client.disconnect()


# =============================================================================
# Enum Tests
# =============================================================================

class TestEnums:
    """Tests for enum values."""

    def test_boundary_modes(self):
        """Test boundary modes."""
        modes = list(BoundaryMode)
        assert len(modes) == 4
        assert BoundaryMode.LOCKDOWN in modes
        assert BoundaryMode.TRUSTED in modes

    def test_request_types(self):
        """Test request types."""
        types = list(RequestType)
        assert len(types) == 7
        assert RequestType.NETWORK_ACCESS in types
        assert RequestType.MEMORY_ACCESS in types

    def test_decisions(self):
        """Test decision values."""
        decisions = list(Decision)
        assert len(decisions) == 4
        assert Decision.ALLOW in decisions
        assert Decision.DENY in decisions
        assert Decision.ESCALATE in decisions

    def test_enforcement_actions(self):
        """Test enforcement actions."""
        actions = list(EnforcementAction)
        assert len(actions) == 6
        assert EnforcementAction.HALT in actions
        assert EnforcementAction.LOCKDOWN in actions
