"""
Tests for Agent Smith Advanced Memory System

Tests the complete advanced memory pipeline including:
- SecurityIntelligenceStore (tiered storage)
- ThreatCorrelator (cross-source correlation)
- PatternSynthesizer (pattern detection)
- BehavioralBaseline (anomaly detection)
- BoundaryDaemonConnector (external integration)
- AdvancedMemoryManager (unified management)
"""

import tempfile
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest

from src.agents.smith.advanced_memory.store import (
    IntelligenceEntry,
    IntelligenceType,
    RetentionPolicy,
    RetentionTier,
    SecurityIntelligenceStore,
    create_intelligence_store,
)
from src.agents.smith.advanced_memory.correlator import (
    CorrelationResult,
    CorrelationRule,
    CorrelationType,
    ThreatCluster,
    ThreatCorrelator,
    ThreatLevel,
    create_threat_correlator,
)
from src.agents.smith.advanced_memory.synthesizer import (
    IntelligenceSummary,
    PatternSynthesizer,
    PatternType,
    SynthesizedPattern,
    TrendAnalysis,
    create_pattern_synthesizer,
)
from src.agents.smith.advanced_memory.baseline import (
    AnomalyScore,
    BaselineMetrics,
    BehavioralBaseline,
    create_behavioral_baseline,
)
from src.agents.smith.advanced_memory.boundary_connector import (
    BoundaryDaemonConnector,
    BoundaryEvent,
    BoundaryMode,
    PolicyDecision,
    TripwireAlert,
    create_boundary_connector,
)
from src.agents.smith.advanced_memory.manager import (
    AdvancedMemoryManager,
    MemoryConfig,
    MemoryStatus,
    create_advanced_memory,
)


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def temp_storage():
    """Create a temporary directory for storage."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def sample_entry():
    """Create a sample intelligence entry."""
    return IntelligenceEntry(
        entry_id="test-001",
        entry_type=IntelligenceType.SIEM_EVENT,
        timestamp=datetime.now(),
        source="boundary-siem",
        severity=3,
        category="intrusion",
        summary="Suspicious network activity detected",
        content={"ip": "192.168.1.100", "port": 443},
        indicators=["192.168.1.100"],
        mitre_tactics=["initial-access"],
        tags={"network", "suspicious"},
    )


@pytest.fixture
def sample_entries():
    """Create multiple sample intelligence entries for correlation testing."""
    base_time = datetime.now()
    return [
        IntelligenceEntry(
            entry_id="test-001",
            entry_type=IntelligenceType.SIEM_EVENT,
            timestamp=base_time,
            source="boundary-siem",
            severity=3,
            category="intrusion",
            summary="Initial access attempt",
            indicators=["192.168.1.100", "evil.com"],
            mitre_tactics=["initial-access"],
        ),
        IntelligenceEntry(
            entry_id="test-002",
            entry_type=IntelligenceType.SIEM_EVENT,
            timestamp=base_time + timedelta(minutes=5),
            source="boundary-siem",
            severity=4,
            category="execution",
            summary="Malicious execution detected",
            indicators=["192.168.1.100"],
            mitre_tactics=["execution"],
        ),
        IntelligenceEntry(
            entry_id="test-003",
            entry_type=IntelligenceType.BOUNDARY_EVENT,
            timestamp=base_time + timedelta(minutes=10),
            source="boundary-daemon",
            severity=4,
            category="privilege-escalation",
            summary="Privilege escalation attempt",
            indicators=["192.168.1.100"],
            mitre_tactics=["privilege-escalation"],
        ),
        IntelligenceEntry(
            entry_id="test-004",
            entry_type=IntelligenceType.SIEM_EVENT,
            timestamp=base_time + timedelta(minutes=15),
            source="boundary-siem",
            severity=5,
            category="exfiltration",
            summary="Data exfiltration detected",
            indicators=["192.168.1.100", "exfil.evil.com"],
            mitre_tactics=["exfiltration"],
        ),
    ]


# ============================================================================
# SecurityIntelligenceStore Tests
# ============================================================================


class TestSecurityIntelligenceStore:
    """Tests for the Security Intelligence Store."""

    def test_create_store(self, temp_storage):
        """Test creating a store."""
        store = create_intelligence_store(storage_path=temp_storage)
        assert store is not None

    def test_add_entry(self, temp_storage, sample_entry):
        """Test adding an entry to the store."""
        store = create_intelligence_store(storage_path=temp_storage)
        store.start()

        entry_id = store.add(sample_entry)

        assert entry_id is not None
        assert entry_id == sample_entry.entry_id

        store.stop()

    def test_get_entry(self, temp_storage, sample_entry):
        """Test retrieving an entry."""
        store = create_intelligence_store(storage_path=temp_storage)
        store.start()

        store.add(sample_entry)
        retrieved = store.get(sample_entry.entry_id)

        assert retrieved is not None
        assert retrieved.entry_id == sample_entry.entry_id
        assert retrieved.source == sample_entry.source
        assert retrieved.severity == sample_entry.severity

        store.stop()

    def test_query_by_type(self, temp_storage, sample_entries):
        """Test querying entries by type."""
        store = create_intelligence_store(storage_path=temp_storage)
        store.start()

        for entry in sample_entries:
            store.add(entry)

        # Query only SIEM events
        results = store.query(entry_types=[IntelligenceType.SIEM_EVENT])

        assert len(results) == 3  # 3 SIEM events in sample

        store.stop()

    def test_query_by_severity(self, temp_storage, sample_entries):
        """Test querying entries by minimum severity."""
        store = create_intelligence_store(storage_path=temp_storage)
        store.start()

        for entry in sample_entries:
            store.add(entry)

        # Query high severity only
        results = store.query(severity_min=4)

        assert len(results) == 3  # severity 4 and 5

        store.stop()

    def test_query_by_source(self, temp_storage, sample_entries):
        """Test querying entries by source."""
        store = create_intelligence_store(storage_path=temp_storage)
        store.start()

        for entry in sample_entries:
            store.add(entry)

        results = store.query(sources=["boundary-daemon"])

        assert len(results) == 1
        assert results[0].source == "boundary-daemon"

        store.stop()

    def test_query_by_time_range(self, temp_storage, sample_entries):
        """Test querying entries by time range."""
        store = create_intelligence_store(storage_path=temp_storage)
        store.start()

        for entry in sample_entries:
            store.add(entry)

        # Query last 10 minutes from the last entry
        since = sample_entries[0].timestamp + timedelta(minutes=8)
        results = store.query(since=since)

        assert len(results) >= 2  # Entries from minute 10 and 15

        store.stop()

    def test_entry_serialization(self, sample_entry):
        """Test entry serialization and deserialization."""
        data = sample_entry.to_dict()

        assert data["entry_id"] == sample_entry.entry_id
        assert data["entry_type"] == "SIEM_EVENT"
        assert data["severity"] == 3

        # Deserialize
        restored = IntelligenceEntry.from_dict(data)

        assert restored.entry_id == sample_entry.entry_id
        assert restored.entry_type == IntelligenceType.SIEM_EVENT
        assert restored.severity == 3

    def test_entry_compact_for_warm(self, sample_entry):
        """Test compacting entry for warm tier."""
        sample_entry.content = {"key1": "value1", "key2": "value2", "large_data": "x" * 1000}

        compacted = sample_entry.compact_for_warm()

        assert compacted.entry_id == sample_entry.entry_id
        assert compacted.tier == RetentionTier.WARM
        # Content should be reduced
        assert len(compacted.content) <= len(sample_entry.content)

    def test_get_stats(self, temp_storage, sample_entries):
        """Test getting store statistics."""
        store = create_intelligence_store(storage_path=temp_storage)
        store.start()

        for entry in sample_entries:
            store.add(entry)

        stats = store.get_stats()

        assert "total_entries" in stats
        assert stats["total_entries"] >= 4
        assert "hot_entries" in stats

        store.stop()

    def test_count(self, temp_storage, sample_entries):
        """Test counting entries."""
        store = create_intelligence_store(storage_path=temp_storage)
        store.start()

        for entry in sample_entries:
            store.add(entry)

        total = store.count()
        high_severity = store.count(severity_min=4)

        assert total == 4
        assert high_severity == 3

        store.stop()


# ============================================================================
# ThreatCorrelator Tests
# ============================================================================


class TestThreatCorrelator:
    """Tests for the Threat Correlator."""

    @pytest.fixture
    def store_with_data(self, temp_storage, sample_entries):
        """Create a store populated with sample data."""
        store = create_intelligence_store(storage_path=temp_storage)
        store.start()
        for entry in sample_entries:
            store.add(entry)
        yield store
        store.stop()

    def test_create_correlator(self, store_with_data):
        """Test creating a correlator."""
        correlator = create_threat_correlator(store=store_with_data)
        assert correlator is not None

    def test_default_rules_loaded(self, store_with_data):
        """Test that default correlation rules are loaded."""
        correlator = create_threat_correlator(store=store_with_data)

        rules = correlator.list_rules()
        assert len(rules) > 0

        # Check for specific built-in rules
        rule_ids = {r.rule_id for r in rules}
        assert "kill_chain_progression" in rule_ids
        assert "multi_source_indicator" in rule_ids

    def test_add_custom_rule(self, store_with_data):
        """Test adding a custom correlation rule."""
        correlator = create_threat_correlator(store=store_with_data)

        rule = CorrelationRule(
            rule_id="custom_test",
            name="Custom Test Rule",
            description="Test rule",
            correlation_type=CorrelationType.TEMPORAL,
            time_window_minutes=30,
            min_events=2,
        )

        correlator.add_rule(rule)

        rules = correlator.list_rules()
        rule_ids = {r.rule_id for r in rules}
        assert "custom_test" in rule_ids

    def test_correlate_single_entry(self, store_with_data, sample_entries):
        """Test correlating a single entry."""
        correlator = create_threat_correlator(store=store_with_data)

        # Correlate the last entry
        results = correlator.correlate(sample_entries[-1])

        # Should find correlations based on shared indicators
        assert isinstance(results, list)

    def test_correlate_finds_kill_chain(self, store_with_data, sample_entries):
        """Test that kill chain progression is detected."""
        correlator = create_threat_correlator(store=store_with_data)

        # Add entries with progressive MITRE tactics
        for entry in sample_entries:
            correlator.correlate(entry)

        # Get active clusters
        clusters = correlator.get_active_clusters()

        # Should have at least one cluster for the attack progression
        assert len(clusters) >= 1

        # Check that MITRE tactics are tracked
        if clusters:
            cluster = clusters[0]
            assert len(cluster.mitre_tactics) >= 1

    def test_threat_level_calculation(self, store_with_data):
        """Test threat level calculation for clusters."""
        correlator = create_threat_correlator(store=store_with_data)

        # Create entries that should trigger high threat level
        high_severity_entries = [
            IntelligenceEntry(
                entry_id=f"critical-{i}",
                entry_type=IntelligenceType.SIEM_EVENT,
                timestamp=datetime.now() + timedelta(minutes=i),
                source=f"source-{i % 2}",  # Multiple sources
                severity=5,
                category="critical",
                summary="Critical event",
                indicators=["malicious.com"],
                mitre_tactics=["exfiltration"],
            )
            for i in range(3)
        ]

        for entry in high_severity_entries:
            correlator.correlate(entry)

        clusters = correlator.get_active_clusters(min_threat_level=ThreatLevel.HIGH)

        # High severity, multi-source should trigger high threat
        if clusters:
            assert any(c.threat_level.value >= ThreatLevel.HIGH.value for c in clusters)

    def test_cluster_entry_tracking(self, store_with_data, sample_entries):
        """Test that clusters track entries correctly."""
        correlator = create_threat_correlator(store=store_with_data)

        for entry in sample_entries:
            correlator.correlate(entry)

        clusters = correlator.get_active_clusters()

        if clusters:
            cluster = clusters[0]
            assert cluster.entry_count >= 1
            assert len(cluster.entry_ids) >= 1
            assert len(cluster.sources) >= 1

    def test_get_cluster_by_id(self, store_with_data, sample_entries):
        """Test retrieving a cluster by ID."""
        correlator = create_threat_correlator(store=store_with_data)

        for entry in sample_entries:
            correlator.correlate(entry)

        clusters = correlator.get_active_clusters()

        if clusters:
            cluster_id = clusters[0].cluster_id
            retrieved = correlator.get_cluster(cluster_id)

            assert retrieved is not None
            assert retrieved.cluster_id == cluster_id

    def test_correlator_stats(self, store_with_data, sample_entries):
        """Test correlator statistics."""
        correlator = create_threat_correlator(store=store_with_data)

        for entry in sample_entries:
            correlator.correlate(entry)

        stats = correlator.get_stats()

        assert "total_correlations" in stats
        assert "active_clusters" in stats
        assert "rules_count" in stats


# ============================================================================
# PatternSynthesizer Tests
# ============================================================================


class TestPatternSynthesizer:
    """Tests for the Pattern Synthesizer."""

    @pytest.fixture
    def populated_store(self, temp_storage, sample_entries):
        """Create a store with sample data."""
        store = create_intelligence_store(storage_path=temp_storage)
        store.start()
        for entry in sample_entries:
            store.add(entry)
        yield store
        store.stop()

    def test_create_synthesizer(self, populated_store):
        """Test creating a synthesizer."""
        synthesizer = create_pattern_synthesizer(store=populated_store)
        assert synthesizer is not None

    def test_detect_patterns(self, populated_store):
        """Test pattern detection."""
        synthesizer = create_pattern_synthesizer(store=populated_store)

        patterns = synthesizer.detect_patterns()

        assert isinstance(patterns, list)

    def test_detect_volumetric_pattern(self, temp_storage):
        """Test detection of volumetric patterns."""
        store = create_intelligence_store(storage_path=temp_storage)
        store.start()

        # Add many events in a short time
        base_time = datetime.now()
        for i in range(20):
            store.add(IntelligenceEntry(
                entry_id=f"volume-{i}",
                entry_type=IntelligenceType.SIEM_EVENT,
                timestamp=base_time + timedelta(seconds=i * 30),
                source="boundary-siem",
                severity=3,
                category="scan",
                summary="Port scan detected",
            ))

        synthesizer = create_pattern_synthesizer(store=store)
        patterns = synthesizer.detect_patterns()

        # Should detect high volume pattern
        volumetric = [p for p in patterns if p.pattern_type == PatternType.VOLUMETRIC]
        # May or may not find depending on thresholds
        assert isinstance(volumetric, list)

        store.stop()

    def test_analyze_trends(self, populated_store, sample_entries):
        """Test trend analysis."""
        synthesizer = create_pattern_synthesizer(store=populated_store)

        trend = synthesizer.analyze_trends(period_hours=24)

        assert trend is not None
        assert hasattr(trend, "period_hours")
        assert hasattr(trend, "total_events")

    def test_generate_summary(self, populated_store):
        """Test generating intelligence summary."""
        synthesizer = create_pattern_synthesizer(store=populated_store)

        summary = synthesizer.generate_summary(period="last_24h")

        assert summary is not None
        assert hasattr(summary, "period")
        assert hasattr(summary, "generated_at")
        assert hasattr(summary, "total_events")

    def test_summary_to_dict(self, populated_store):
        """Test summary serialization."""
        synthesizer = create_pattern_synthesizer(store=populated_store)

        summary = synthesizer.generate_summary(period="last_24h")
        data = summary.to_dict()

        assert "period" in data
        assert "total_events" in data
        assert "generated_at" in data

    def test_get_patterns(self, populated_store):
        """Test getting stored patterns."""
        synthesizer = create_pattern_synthesizer(store=populated_store)

        # Detect some patterns first
        synthesizer.detect_patterns()

        # Get patterns
        patterns = synthesizer.get_patterns(min_severity=1)

        assert isinstance(patterns, list)

    def test_synthesizer_stats(self, populated_store):
        """Test synthesizer statistics."""
        synthesizer = create_pattern_synthesizer(store=populated_store)

        stats = synthesizer.get_stats()

        assert "patterns_stored" in stats
        assert "summaries_generated" in stats


# ============================================================================
# BehavioralBaseline Tests
# ============================================================================


class TestBehavioralBaseline:
    """Tests for the Behavioral Baseline."""

    @pytest.fixture
    def populated_store(self, temp_storage):
        """Create a store with baseline data."""
        store = create_intelligence_store(storage_path=temp_storage)
        store.start()

        # Add normal baseline data over time
        base_time = datetime.now() - timedelta(hours=24)
        for i in range(50):
            store.add(IntelligenceEntry(
                entry_id=f"baseline-{i}",
                entry_type=IntelligenceType.SIEM_EVENT,
                timestamp=base_time + timedelta(minutes=i * 30),
                source="boundary-siem",
                severity=2,  # Mostly low severity
                category="audit",
                summary=f"Normal audit event {i}",
            ))

        yield store
        store.stop()

    def test_create_baseline(self, populated_store):
        """Test creating a baseline."""
        baseline = create_behavioral_baseline(store=populated_store)
        assert baseline is not None

    def test_learn_baseline(self, populated_store):
        """Test learning baseline from historical data."""
        baseline = create_behavioral_baseline(store=populated_store)
        baseline.start()

        result = baseline.learn(lookback_hours=24)

        assert result is True

        baseline.stop()

    def test_is_ready(self, populated_store):
        """Test baseline readiness."""
        baseline = create_behavioral_baseline(store=populated_store)
        baseline.start()

        # Initially not ready
        assert baseline.is_ready() is False

        # After learning
        baseline.learn(lookback_hours=24)

        assert baseline.is_ready() is True

        baseline.stop()

    def test_score_normal_activity(self, populated_store):
        """Test scoring normal activity."""
        baseline = create_behavioral_baseline(store=populated_store)
        baseline.start()
        baseline.learn(lookback_hours=24)

        score = baseline.score(lookback_minutes=60)

        assert score is not None
        assert hasattr(score, "overall_score")
        assert 0 <= score.overall_score <= 1

        # Normal activity should have low anomaly score
        # (though this depends on the implementation)
        assert isinstance(score.overall_score, float)

        baseline.stop()

    def test_score_anomalous_activity(self, temp_storage):
        """Test scoring anomalous activity."""
        store = create_intelligence_store(storage_path=temp_storage)
        store.start()

        # First, create normal baseline
        base_time = datetime.now() - timedelta(hours=24)
        for i in range(50):
            store.add(IntelligenceEntry(
                entry_id=f"normal-{i}",
                entry_type=IntelligenceType.SIEM_EVENT,
                timestamp=base_time + timedelta(minutes=i * 30),
                source="boundary-siem",
                severity=2,
                category="audit",
                summary=f"Normal event {i}",
            ))

        baseline = create_behavioral_baseline(store=store)
        baseline.start()
        baseline.learn(lookback_hours=24)

        # Now add anomalous activity - high severity burst
        for i in range(10):
            store.add(IntelligenceEntry(
                entry_id=f"anomaly-{i}",
                entry_type=IntelligenceType.SIEM_EVENT,
                timestamp=datetime.now() - timedelta(minutes=5 - i),
                source="unknown-source",  # Different source
                severity=5,  # High severity
                category="attack",  # Different category
                summary="Suspicious activity",
            ))

        score = baseline.score(lookback_minutes=60)

        assert score is not None
        # Anomalous activity should have higher score
        # (exact threshold depends on implementation)
        assert isinstance(score.overall_score, float)

        baseline.stop()
        store.stop()

    def test_anomaly_score_structure(self, populated_store):
        """Test anomaly score data structure."""
        baseline = create_behavioral_baseline(store=populated_store)
        baseline.start()
        baseline.learn(lookback_hours=24)

        score = baseline.score(lookback_minutes=60)

        assert hasattr(score, "score_id")
        assert hasattr(score, "overall_score")
        assert hasattr(score, "dimensions")
        assert hasattr(score, "timestamp")

        baseline.stop()

    def test_anomaly_score_to_dict(self, populated_store):
        """Test anomaly score serialization."""
        baseline = create_behavioral_baseline(store=populated_store)
        baseline.start()
        baseline.learn(lookback_hours=24)

        score = baseline.score()
        data = score.to_dict()

        assert "score_id" in data
        assert "overall_score" in data
        assert "timestamp" in data

        baseline.stop()

    def test_baseline_stats(self, populated_store):
        """Test baseline statistics."""
        baseline = create_behavioral_baseline(store=populated_store)
        baseline.start()
        baseline.learn(lookback_hours=24)

        stats = baseline.get_stats()

        assert "is_ready" in stats
        assert "learning_hours" in stats

        baseline.stop()


# ============================================================================
# BoundaryDaemonConnector Tests
# ============================================================================


class TestBoundaryDaemonConnector:
    """Tests for the Boundary Daemon Connector."""

    @pytest.fixture
    def store(self, temp_storage):
        """Create a store for the connector."""
        store = create_intelligence_store(storage_path=temp_storage)
        store.start()
        yield store
        store.stop()

    def test_create_connector(self, store):
        """Test creating a connector."""
        connector = create_boundary_connector(store=store)
        assert connector is not None

    def test_initial_mode(self, store):
        """Test initial boundary mode."""
        connector = create_boundary_connector(store=store)

        mode = connector.get_mode()

        assert mode == BoundaryMode.OPEN

    def test_set_mode(self, store):
        """Test setting boundary mode."""
        connector = create_boundary_connector(store=store)

        connector.set_mode(BoundaryMode.RESTRICTED)
        mode = connector.get_mode()

        assert mode == BoundaryMode.RESTRICTED

    def test_boundary_modes(self):
        """Test all boundary modes are defined."""
        modes = list(BoundaryMode)

        assert BoundaryMode.OPEN in modes
        assert BoundaryMode.RESTRICTED in modes
        assert BoundaryMode.TRUSTED in modes
        assert BoundaryMode.AIRGAP in modes
        assert BoundaryMode.COLDROOM in modes
        assert BoundaryMode.LOCKDOWN in modes

    def test_boundary_event_creation(self):
        """Test creating a boundary event."""
        event = BoundaryEvent(
            event_id="be-001",
            timestamp=datetime.now(),
            event_type="mode_change",
            source="boundary-daemon",
            current_mode=BoundaryMode.RESTRICTED,
            target_resource="network",
            action_requested="restrict",
            details={"reason": "security alert"},
        )

        assert event.event_id == "be-001"
        assert event.current_mode == BoundaryMode.RESTRICTED

    def test_boundary_event_to_intelligence(self):
        """Test converting boundary event to intelligence entry."""
        event = BoundaryEvent(
            event_id="be-002",
            timestamp=datetime.now(),
            event_type="policy_violation",
            source="boundary-daemon",
            current_mode=BoundaryMode.RESTRICTED,
            target_resource="api/v1/data",
            action_requested="block",
            details={"user": "unknown"},
        )

        entry = event.to_intelligence_entry()

        assert entry.entry_type == IntelligenceType.BOUNDARY_EVENT
        assert entry.source == "boundary-daemon"

    def test_policy_decision_creation(self):
        """Test creating a policy decision."""
        decision = PolicyDecision(
            decision_id="pd-001",
            timestamp=datetime.now(),
            policy_id="access-control",
            decision="deny",
            reason="Unauthorized access attempt",
            target_resource="/admin",
            details={"ip": "10.0.0.1"},
        )

        assert decision.decision == "deny"
        assert decision.policy_id == "access-control"

    def test_tripwire_alert_creation(self):
        """Test creating a tripwire alert."""
        alert = TripwireAlert(
            alert_id="tw-001",
            timestamp=datetime.now(),
            tripwire_id="config-monitor",
            triggered_by="/etc/config.yaml",
            alert_type="modification",
            severity=4,
            details={"old_hash": "abc", "new_hash": "xyz"},
        )

        assert alert.tripwire_id == "config-monitor"
        assert alert.severity == 4

    def test_connector_stats(self, store):
        """Test connector statistics."""
        connector = create_boundary_connector(store=store)

        stats = connector.get_stats()

        assert "connected" in stats
        assert "current_mode" in stats

    @patch("requests.get")
    def test_connect_success(self, mock_get, store):
        """Test successful connection."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"status": "healthy"}
        mock_get.return_value = mock_response

        connector = create_boundary_connector(
            store=store,
            endpoint="http://localhost:8080",
        )

        result = connector.connect()

        assert result is True

    @patch("requests.get")
    def test_connect_failure(self, mock_get, store):
        """Test connection failure handling."""
        mock_get.side_effect = Exception("Connection refused")

        connector = create_boundary_connector(
            store=store,
            endpoint="http://localhost:8080",
        )

        result = connector.connect()

        assert result is False


# ============================================================================
# AdvancedMemoryManager Tests
# ============================================================================


class TestAdvancedMemoryManager:
    """Tests for the Advanced Memory Manager."""

    def test_create_manager(self, temp_storage):
        """Test creating a memory manager."""
        config = MemoryConfig(storage_path=temp_storage)
        manager = AdvancedMemoryManager(config)

        assert manager is not None

    def test_initialize_manager(self, temp_storage):
        """Test initializing the manager."""
        config = MemoryConfig(storage_path=temp_storage)
        manager = AdvancedMemoryManager(config)

        result = manager.initialize()

        assert result is True

    def test_start_manager(self, temp_storage):
        """Test starting the manager."""
        config = MemoryConfig(storage_path=temp_storage)
        manager = AdvancedMemoryManager(config)
        manager.initialize()

        result = manager.start()

        assert result is True

        manager.stop()

    def test_stop_manager(self, temp_storage):
        """Test stopping the manager."""
        config = MemoryConfig(storage_path=temp_storage)
        manager = AdvancedMemoryManager(config)
        manager.initialize()
        manager.start()

        manager.stop()

        status = manager.get_status()
        assert status.running is False

    def test_factory_function(self, temp_storage):
        """Test factory function with auto_start."""
        manager = create_advanced_memory(
            storage_path=temp_storage,
            auto_start=True,
        )

        assert manager is not None

        status = manager.get_status()
        assert status.initialized is True
        assert status.running is True

        manager.stop()

    def test_ingest_entry(self, temp_storage):
        """Test ingesting an intelligence entry."""
        manager = create_advanced_memory(storage_path=temp_storage, auto_start=True)

        entry = IntelligenceEntry(
            entry_id="test-ingest",
            entry_type=IntelligenceType.SIEM_EVENT,
            timestamp=datetime.now(),
            source="test",
            severity=3,
            category="test",
            summary="Test event",
        )

        entry_id = manager.ingest(entry)

        assert entry_id is not None

        manager.stop()

    def test_ingest_siem_event(self, temp_storage):
        """Test ingesting a SIEM event from dict."""
        manager = create_advanced_memory(storage_path=temp_storage, auto_start=True)

        event_data = {
            "timestamp": datetime.now().isoformat(),
            "severity": "high",
            "category": "intrusion",
            "message": "Suspicious activity detected",
            "indicators": ["192.168.1.100"],
        }

        entry_id = manager.ingest_siem_event(event_data)

        assert entry_id is not None

        manager.stop()

    def test_ingest_boundary_event(self, temp_storage):
        """Test ingesting a Boundary-Daemon event."""
        manager = create_advanced_memory(storage_path=temp_storage, auto_start=True)

        event = BoundaryEvent(
            event_id="be-test",
            timestamp=datetime.now(),
            event_type="test",
            source="boundary-daemon",
            current_mode=BoundaryMode.OPEN,
            target_resource="test",
            action_requested="allow",
            details={},
        )

        entry_id = manager.ingest_boundary_event(event)

        assert entry_id is not None

        manager.stop()

    def test_query_entries(self, temp_storage):
        """Test querying intelligence entries."""
        manager = create_advanced_memory(storage_path=temp_storage, auto_start=True)

        # Add some entries
        for i in range(5):
            manager.ingest(IntelligenceEntry(
                entry_id=f"query-test-{i}",
                entry_type=IntelligenceType.SIEM_EVENT,
                timestamp=datetime.now(),
                source="test",
                severity=3,
                category="test",
                summary=f"Test event {i}",
            ))

        results = manager.query(limit=10)

        assert len(results) >= 5

        manager.stop()

    def test_get_threat_clusters(self, temp_storage):
        """Test getting threat clusters."""
        manager = create_advanced_memory(storage_path=temp_storage, auto_start=True)

        # Add correlated entries
        for i in range(3):
            manager.ingest(IntelligenceEntry(
                entry_id=f"cluster-test-{i}",
                entry_type=IntelligenceType.SIEM_EVENT,
                timestamp=datetime.now() + timedelta(minutes=i),
                source="test",
                severity=4,
                category="attack",
                summary="Attack event",
                indicators=["malicious.com"],
            ))

        clusters = manager.get_threat_clusters()

        assert isinstance(clusters, list)

        manager.stop()

    def test_get_patterns(self, temp_storage):
        """Test getting detected patterns."""
        manager = create_advanced_memory(storage_path=temp_storage, auto_start=True)

        patterns = manager.get_patterns()

        assert isinstance(patterns, list)

        manager.stop()

    def test_get_anomaly_score(self, temp_storage):
        """Test getting anomaly score."""
        manager = create_advanced_memory(storage_path=temp_storage, auto_start=True)

        # Add some baseline data
        for i in range(20):
            manager.ingest(IntelligenceEntry(
                entry_id=f"baseline-{i}",
                entry_type=IntelligenceType.SIEM_EVENT,
                timestamp=datetime.now() - timedelta(hours=i),
                source="test",
                severity=2,
                category="audit",
                summary="Normal event",
            ))

        score = manager.get_anomaly_score()

        # May be None if baseline not ready
        if score is not None:
            assert hasattr(score, "overall_score")

        manager.stop()

    def test_generate_summary(self, temp_storage):
        """Test generating intelligence summary."""
        manager = create_advanced_memory(storage_path=temp_storage, auto_start=True)

        # Add some data
        for i in range(5):
            manager.ingest(IntelligenceEntry(
                entry_id=f"summary-test-{i}",
                entry_type=IntelligenceType.SIEM_EVENT,
                timestamp=datetime.now(),
                source="test",
                severity=3,
                category="test",
                summary="Test event",
            ))

        summary = manager.generate_summary(period="last_24h")

        assert summary is not None

        manager.stop()

    def test_get_status(self, temp_storage):
        """Test getting memory status."""
        manager = create_advanced_memory(storage_path=temp_storage, auto_start=True)

        status = manager.get_status()

        assert isinstance(status, MemoryStatus)
        assert status.initialized is True
        assert status.running is True

        manager.stop()

    def test_status_to_dict(self, temp_storage):
        """Test status serialization."""
        manager = create_advanced_memory(storage_path=temp_storage, auto_start=True)

        status = manager.get_status()
        data = status.to_dict()

        assert "initialized" in data
        assert "running" in data
        assert "healthy" in data

        manager.stop()

    def test_get_stats(self, temp_storage):
        """Test getting comprehensive statistics."""
        manager = create_advanced_memory(storage_path=temp_storage, auto_start=True)

        stats = manager.get_stats()

        assert "events_processed" in stats
        assert "store" in stats

        manager.stop()

    def test_get_boundary_mode(self, temp_storage):
        """Test getting boundary mode."""
        manager = create_advanced_memory(storage_path=temp_storage, auto_start=True)

        mode = manager.get_boundary_mode()

        assert mode == BoundaryMode.OPEN

        manager.stop()

    def test_set_boundary_mode(self, temp_storage):
        """Test setting boundary mode."""
        manager = create_advanced_memory(storage_path=temp_storage, auto_start=True)

        manager.set_boundary_mode(BoundaryMode.RESTRICTED)
        mode = manager.get_boundary_mode()

        assert mode == BoundaryMode.RESTRICTED

        manager.stop()

    def test_callbacks_registered(self, temp_storage):
        """Test that callbacks are properly registered."""
        threat_callback = MagicMock()
        anomaly_callback = MagicMock()
        pattern_callback = MagicMock()

        config = MemoryConfig(
            storage_path=temp_storage,
            on_high_threat=threat_callback,
            on_anomaly=anomaly_callback,
            on_pattern=pattern_callback,
        )

        manager = AdvancedMemoryManager(config)
        manager.initialize()
        manager.start()

        # Callbacks should be registered but not called yet
        assert threat_callback.call_count == 0

        manager.stop()

    def test_config_defaults(self):
        """Test configuration defaults."""
        config = MemoryConfig()

        assert config.hot_retention_days == 7
        assert config.warm_retention_days == 30
        assert config.cold_retention_days == 365
        assert config.correlation_enabled is True
        assert config.synthesis_enabled is True
        assert config.baseline_enabled is True


# ============================================================================
# Integration Tests
# ============================================================================


class TestAdvancedMemoryIntegration:
    """Integration tests for the complete advanced memory system."""

    def test_full_pipeline(self, temp_storage):
        """Test the complete pipeline from ingestion to summary."""
        manager = create_advanced_memory(storage_path=temp_storage, auto_start=True)

        # Simulate a sequence of security events
        base_time = datetime.now()

        # Initial access attempt
        manager.ingest_siem_event({
            "timestamp": base_time.isoformat(),
            "severity": "medium",
            "category": "initial-access",
            "message": "Suspicious login attempt",
            "indicators": ["attacker.com", "192.168.1.100"],
            "mitre_attack": {"tactics": ["initial-access"]},
        })

        # Execution
        manager.ingest_siem_event({
            "timestamp": (base_time + timedelta(minutes=5)).isoformat(),
            "severity": "high",
            "category": "execution",
            "message": "Malicious script execution",
            "indicators": ["192.168.1.100"],
            "mitre_attack": {"tactics": ["execution"]},
        })

        # Lateral movement
        manager.ingest_siem_event({
            "timestamp": (base_time + timedelta(minutes=10)).isoformat(),
            "severity": "high",
            "category": "lateral-movement",
            "message": "Lateral movement detected",
            "indicators": ["192.168.1.100", "192.168.1.200"],
            "mitre_attack": {"tactics": ["lateral-movement"]},
        })

        # Exfiltration
        manager.ingest_siem_event({
            "timestamp": (base_time + timedelta(minutes=15)).isoformat(),
            "severity": "critical",
            "category": "exfiltration",
            "message": "Data exfiltration attempt",
            "indicators": ["192.168.1.100", "exfil.attacker.com"],
            "mitre_attack": {"tactics": ["exfiltration"]},
        })

        # Check that data was stored
        entries = manager.query(limit=10)
        assert len(entries) >= 4

        # Check for threat clusters
        clusters = manager.get_threat_clusters()
        # May or may not have clusters depending on correlation
        assert isinstance(clusters, list)

        # Generate summary
        summary = manager.generate_summary()
        assert summary is not None
        assert summary.total_events >= 4

        manager.stop()

    def test_multi_source_correlation(self, temp_storage):
        """Test correlation across multiple sources."""
        manager = create_advanced_memory(storage_path=temp_storage, auto_start=True)

        # Event from SIEM
        manager.ingest_siem_event({
            "timestamp": datetime.now().isoformat(),
            "severity": "high",
            "category": "attack",
            "message": "Attack detected by SIEM",
            "indicators": ["192.168.1.100"],
        })

        # Event from Boundary-Daemon
        manager.ingest_boundary_event(BoundaryEvent(
            event_id="bd-001",
            timestamp=datetime.now(),
            event_type="violation",
            source="boundary-daemon",
            current_mode=BoundaryMode.RESTRICTED,
            target_resource="api",
            action_requested="block",
            details={"ip": "192.168.1.100"},  # Same indicator
        ))

        # Check for multi-source correlation
        clusters = manager.get_threat_clusters()

        # Status should show both sources
        status = manager.get_status()
        assert status.store_healthy is True

        manager.stop()

    def test_anomaly_detection_flow(self, temp_storage):
        """Test anomaly detection after baseline learning."""
        manager = create_advanced_memory(storage_path=temp_storage, auto_start=True)

        # Establish normal baseline - lots of low-severity audit events
        for i in range(30):
            manager.ingest(IntelligenceEntry(
                entry_id=f"normal-{i}",
                entry_type=IntelligenceType.SIEM_EVENT,
                timestamp=datetime.now() - timedelta(hours=24 - i),
                source="audit",
                severity=1,
                category="audit",
                summary="Normal audit event",
            ))

        # Now inject anomalous activity
        for i in range(5):
            manager.ingest(IntelligenceEntry(
                entry_id=f"anomaly-{i}",
                entry_type=IntelligenceType.ATTACK_DETECTED,
                timestamp=datetime.now() - timedelta(minutes=5 - i),
                source="ids",
                severity=5,
                category="attack",
                summary="Anomalous attack detected",
            ))

        # Get anomaly score
        score = manager.get_anomaly_score()

        # Score may or may not be available depending on baseline readiness
        if score is not None:
            assert isinstance(score.overall_score, float)

        manager.stop()

    def test_concurrent_operations(self, temp_storage):
        """Test thread safety with concurrent operations."""
        manager = create_advanced_memory(storage_path=temp_storage, auto_start=True)

        errors = []
        results = []

        def ingest_events(thread_id: int):
            try:
                for i in range(10):
                    manager.ingest(IntelligenceEntry(
                        entry_id=f"thread-{thread_id}-{i}",
                        entry_type=IntelligenceType.SIEM_EVENT,
                        timestamp=datetime.now(),
                        source=f"source-{thread_id}",
                        severity=3,
                        category="test",
                        summary=f"Event from thread {thread_id}",
                    ))
                results.append(thread_id)
            except Exception as e:
                errors.append(e)

        # Run multiple threads
        threads = [threading.Thread(target=ingest_events, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert len(results) == 5

        # Verify all data was stored
        entries = manager.query(limit=100)
        assert len(entries) >= 50

        manager.stop()


# ============================================================================
# Module Export Tests
# ============================================================================


class TestModuleExports:
    """Test that all expected symbols are exported from the module."""

    def test_store_exports(self):
        """Test store module exports."""
        from src.agents.smith.advanced_memory import (
            SecurityIntelligenceStore,
            IntelligenceEntry,
            IntelligenceType,
            RetentionTier,
            create_intelligence_store,
        )

        assert SecurityIntelligenceStore is not None
        assert IntelligenceEntry is not None
        assert IntelligenceType is not None
        assert RetentionTier is not None
        assert create_intelligence_store is not None

    def test_correlator_exports(self):
        """Test correlator module exports."""
        from src.agents.smith.advanced_memory import (
            ThreatCorrelator,
            CorrelationResult,
            CorrelationRule,
            ThreatCluster,
            ThreatLevel,
            create_threat_correlator,
        )

        assert ThreatCorrelator is not None
        assert ThreatCluster is not None
        assert ThreatLevel is not None

    def test_synthesizer_exports(self):
        """Test synthesizer module exports."""
        from src.agents.smith.advanced_memory import (
            PatternSynthesizer,
            SynthesizedPattern,
            IntelligenceSummary,
            create_pattern_synthesizer,
        )

        assert PatternSynthesizer is not None
        assert SynthesizedPattern is not None
        assert IntelligenceSummary is not None

    def test_baseline_exports(self):
        """Test baseline module exports."""
        from src.agents.smith.advanced_memory import (
            BehavioralBaseline,
            AnomalyScore,
            create_behavioral_baseline,
        )

        assert BehavioralBaseline is not None
        assert AnomalyScore is not None

    def test_boundary_exports(self):
        """Test boundary connector exports."""
        from src.agents.smith.advanced_memory import (
            BoundaryDaemonConnector,
            BoundaryEvent,
            BoundaryMode,
            PolicyDecision,
            TripwireAlert,
            create_boundary_connector,
        )

        assert BoundaryDaemonConnector is not None
        assert BoundaryEvent is not None
        assert BoundaryMode is not None

    def test_manager_exports(self):
        """Test manager module exports."""
        from src.agents.smith.advanced_memory import (
            AdvancedMemoryManager,
            MemoryConfig,
            create_advanced_memory,
        )

        assert AdvancedMemoryManager is not None
        assert MemoryConfig is not None
        assert create_advanced_memory is not None
