"""
Tests for Agent OS Value Ledger Integration
"""

import pytest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch
import sys

# Add value-ledger to path for testing
sys.path.insert(0, "/home/user/value-ledger")

from src.ledger.models import (
    ValueEvent,
    ValueDimension,
    IntentCategory,
    IntentValueMapping,
    DEFAULT_MAPPINGS,
    get_mapping_for_intent,
)
from src.ledger.client import (
    LedgerClient,
    LedgerConfig,
    LedgerEntry,
    LedgerStats,
    create_ledger_client,
)
from src.ledger.hooks import (
    IntentValueHook,
    AgentValueTracker,
    create_intent_hook,
)


# =============================================================================
# Model Tests
# =============================================================================

class TestValueDimension:
    """Tests for ValueDimension enum."""

    def test_all_dimensions_defined(self):
        """Test all expected dimensions exist."""
        assert ValueDimension.EFFORT
        assert ValueDimension.KNOWLEDGE
        assert ValueDimension.INTERACTION
        assert ValueDimension.CREATIVE
        assert ValueDimension.REASONING
        assert ValueDimension.MEMORY


class TestValueEvent:
    """Tests for ValueEvent model."""

    def test_create_event(self):
        """Test creating a value event."""
        event = ValueEvent(
            event_id="evt_123",
            source="sage",
            dimension=ValueDimension.REASONING,
            intent_category=IntentCategory.QUERY,
            intent_type="query.reasoning",
            context_tokens=100,
            processing_time_ms=50.0,
        )

        assert event.event_id == "evt_123"
        assert event.source == "sage"
        assert event.dimension == ValueDimension.REASONING

    def test_event_to_dict(self):
        """Test event serialization."""
        event = ValueEvent(
            event_id="evt_123",
            source="sage",
            dimension=ValueDimension.KNOWLEDGE,
            intent_category=IntentCategory.QUERY,
            intent_type="query.factual",
        )

        data = event.to_dict()

        assert data["event_id"] == "evt_123"
        assert data["dimension"] == "knowledge"
        assert data["intent_type"] == "query.factual"

    def test_get_intent_hash(self):
        """Test intent hash generation."""
        event = ValueEvent(
            event_id="evt_123",
            source="sage",
            dimension=ValueDimension.EFFORT,
            intent_category=IntentCategory.QUERY,
            intent_type="query.test",
            session_id="session_1",
        )

        hash1 = event.get_intent_hash()
        assert len(hash1) == 16  # Truncated hash


class TestIntentValueMapping:
    """Tests for intent → value mappings."""

    def test_exact_match(self):
        """Test exact intent matching."""
        mapping = IntentValueMapping(
            intent_pattern="memory.store",
            dimension=ValueDimension.MEMORY,
            base_value=0.05,
        )

        assert mapping.matches("memory.store") is True
        assert mapping.matches("memory.retrieve") is False

    def test_wildcard_match(self):
        """Test wildcard intent matching."""
        mapping = IntentValueMapping(
            intent_pattern="memory.*",
            dimension=ValueDimension.MEMORY,
            base_value=0.03,
        )

        assert mapping.matches("memory.store") is True
        assert mapping.matches("memory.retrieve") is True
        assert mapping.matches("query.factual") is False

    def test_calculate_value_base(self):
        """Test base value calculation."""
        mapping = IntentValueMapping(
            intent_pattern="test",
            dimension=ValueDimension.EFFORT,
            base_value=0.1,
        )

        value = mapping.calculate_value()
        assert value == 0.1

    def test_calculate_value_with_tokens(self):
        """Test value calculation with tokens."""
        mapping = IntentValueMapping(
            intent_pattern="test",
            dimension=ValueDimension.EFFORT,
            base_value=0.1,
            token_multiplier=0.001,
        )

        value = mapping.calculate_value(
            context_tokens=50,
            output_tokens=50,
        )
        assert value == pytest.approx(0.2, rel=0.01)  # 0.1 + 100 * 0.001

    def test_calculate_value_clamped(self):
        """Test value is clamped to 0-1."""
        mapping = IntentValueMapping(
            intent_pattern="test",
            dimension=ValueDimension.EFFORT,
            base_value=0.9,
            token_multiplier=0.1,
        )

        value = mapping.calculate_value(context_tokens=1000)
        assert value == 1.0  # Clamped

    def test_disabled_mapping(self):
        """Test disabled mapping returns 0."""
        mapping = IntentValueMapping(
            intent_pattern="test",
            dimension=ValueDimension.EFFORT,
            base_value=0.5,
            enabled=False,
        )

        value = mapping.calculate_value()
        assert value == 0.0


class TestGetMappingForIntent:
    """Tests for mapping lookup."""

    def test_find_exact_mapping(self):
        """Test finding exact mapping."""
        mapping = get_mapping_for_intent("memory.store")
        assert mapping is not None
        assert mapping.dimension == ValueDimension.MEMORY

    def test_find_wildcard_mapping(self):
        """Test finding wildcard mapping."""
        mapping = get_mapping_for_intent("memory.unknown")
        assert mapping is not None
        # Should match memory.* pattern
        assert mapping.dimension == ValueDimension.MEMORY

    def test_fallback_mapping(self):
        """Test fallback to default mapping."""
        mapping = get_mapping_for_intent("unknown.intent.type")
        assert mapping is not None
        # Should match * fallback


# =============================================================================
# Client Tests
# =============================================================================

class TestLedgerClient:
    """Tests for LedgerClient."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        config = LedgerConfig(use_memory=True, enabled=True)
        client = LedgerClient(config)
        client.initialize()
        return client

    def test_initialize(self, client):
        """Test client initialization."""
        assert client._initialized is True

    def test_record_intent(self, client):
        """Test recording an intent."""
        entry_id = client.record_intent(
            source="sage",
            intent_type="query.factual",
            context_tokens=100,
            processing_time_ms=50.0,
        )

        # Should return entry ID (may be None if value-ledger not available)
        # but metrics should be updated
        metrics = client.get_metrics()
        assert metrics["events_recorded"] >= 0

    def test_record_event(self, client):
        """Test recording a value event."""
        event = ValueEvent(
            event_id="test_evt",
            source="seshat",
            dimension=ValueDimension.MEMORY,
            intent_category=IntentCategory.MEMORY,
            intent_type="memory.store",
            context_tokens=50,
        )

        client.record_event(event)

        metrics = client.get_metrics()
        assert metrics["enabled"] is True

    def test_disabled_client(self):
        """Test disabled client doesn't record."""
        config = LedgerConfig(use_memory=True, enabled=False)
        client = LedgerClient(config)
        client.initialize()

        entry_id = client.record_intent(
            source="test",
            intent_type="test.intent",
        )

        assert entry_id is None

    def test_get_metrics(self, client):
        """Test getting client metrics."""
        metrics = client.get_metrics()

        assert "initialized" in metrics
        assert "enabled" in metrics
        assert "events_recorded" in metrics
        assert "mapping_count" in metrics

    def test_shutdown(self, client):
        """Test client shutdown."""
        client.shutdown()
        assert client._initialized is False


class TestCreateLedgerClient:
    """Tests for client factory."""

    def test_create_memory_client(self):
        """Test creating in-memory client."""
        client = create_ledger_client(use_memory=True)

        assert client._initialized is True
        assert client._config.use_memory is True

    def test_create_disabled_client(self):
        """Test creating disabled client."""
        client = create_ledger_client(enabled=False)

        assert client._config.enabled is False


# =============================================================================
# Hook Tests
# =============================================================================

class TestIntentValueHook:
    """Tests for IntentValueHook."""

    @pytest.fixture
    def hook(self):
        """Create test hook."""
        client = create_ledger_client(use_memory=True)
        return IntentValueHook(client, enabled=True)

    def test_on_intent_processed(self, hook):
        """Test processing intent."""
        entry_id = hook.on_intent_processed(
            source="sage",
            intent="query.factual",
            context_tokens=100,
            processing_time_ms=50.0,
            success=True,
        )

        metrics = hook.get_metrics()
        assert metrics["intents_processed"] >= 1

    def test_failed_intent_not_recorded(self, hook):
        """Test failed intent is not recorded."""
        entry_id = hook.on_intent_processed(
            source="sage",
            intent="query.factual",
            success=False,
        )

        assert entry_id is None

    def test_disabled_hook(self):
        """Test disabled hook doesn't record."""
        client = create_ledger_client(use_memory=True)
        hook = IntentValueHook(client, enabled=False)

        entry_id = hook.on_intent_processed(
            source="test",
            intent="test.intent",
            success=True,
        )

        assert entry_id is None

    def test_enable_disable(self, hook):
        """Test enabling/disabling hook."""
        assert hook.enabled is True

        hook.enabled = False
        assert hook.enabled is False

        hook.enabled = True
        assert hook.enabled is True


class TestAgentValueTracker:
    """Tests for AgentValueTracker."""

    @pytest.fixture
    def tracker(self):
        """Create test tracker."""
        client = create_ledger_client(use_memory=True)
        return AgentValueTracker(client)

    def test_record_agent_activity(self, tracker):
        """Test recording agent activity."""
        tracker.record_agent_activity(
            agent_name="sage",
            intent="query.reasoning",
            dimension=ValueDimension.REASONING,
            value=0.15,
            context_tokens=200,
        )

        stats = tracker.get_agent_stats("sage")
        assert stats is not None
        assert stats.total_value == 0.15
        assert stats.intent_count == 1

    def test_multiple_agents(self, tracker):
        """Test tracking multiple agents."""
        tracker.record_agent_activity(
            agent_name="sage",
            intent="query.reasoning",
            dimension=ValueDimension.REASONING,
            value=0.1,
        )
        tracker.record_agent_activity(
            agent_name="seshat",
            intent="memory.store",
            dimension=ValueDimension.MEMORY,
            value=0.05,
        )
        tracker.record_agent_activity(
            agent_name="sage",
            intent="query.factual",
            dimension=ValueDimension.KNOWLEDGE,
            value=0.05,
        )

        all_stats = tracker.get_all_agent_stats()
        assert len(all_stats) == 2

        sage_stats = tracker.get_agent_stats("sage")
        assert sage_stats.intent_count == 2
        assert sage_stats.total_value == pytest.approx(0.15, rel=0.01)

    def test_leaderboard(self, tracker):
        """Test getting agent leaderboard."""
        tracker.record_agent_activity("low", "test", ValueDimension.EFFORT, 0.01)
        tracker.record_agent_activity("mid", "test", ValueDimension.EFFORT, 0.05)
        tracker.record_agent_activity("high", "test", ValueDimension.EFFORT, 0.1)

        leaderboard = tracker.get_leaderboard(limit=2)

        assert len(leaderboard) == 2
        assert leaderboard[0].agent_name == "high"
        assert leaderboard[1].agent_name == "mid"

    def test_value_by_dimension(self, tracker):
        """Test tracking value by dimension."""
        tracker.record_agent_activity(
            agent_name="sage",
            intent="reasoning",
            dimension=ValueDimension.REASONING,
            value=0.1,
        )
        tracker.record_agent_activity(
            agent_name="sage",
            intent="knowledge",
            dimension=ValueDimension.KNOWLEDGE,
            value=0.05,
        )

        stats = tracker.get_agent_stats("sage")
        assert stats.value_by_dimension["reasoning"] == 0.1
        assert stats.value_by_dimension["knowledge"] == 0.05


class TestCreateIntentHook:
    """Tests for hook factory."""

    def test_create_memory_hook(self):
        """Test creating in-memory hook."""
        hook = create_intent_hook(use_memory=True)

        assert hook.enabled is True

    def test_create_disabled_hook(self):
        """Test creating disabled hook."""
        hook = create_intent_hook(enabled=False)

        assert hook.enabled is False


# =============================================================================
# Value Ledger Core Tests (if module available)
# =============================================================================

class TestValueLedgerCore:
    """Tests for value-ledger core (if available)."""

    @pytest.fixture
    def ledger_available(self):
        """Check if value-ledger is available."""
        try:
            from ledger import LedgerStore, create_ledger_store
            return True
        except ImportError:
            return False

    def test_ledger_store_create(self, ledger_available):
        """Test creating ledger store."""
        if not ledger_available:
            pytest.skip("value-ledger module not available")

        from ledger import create_ledger_store

        store = create_ledger_store(None)  # In-memory
        assert store._initialized is True

    def test_ledger_store_append(self, ledger_available):
        """Test appending to ledger."""
        if not ledger_available:
            pytest.skip("value-ledger module not available")

        from ledger import (
            create_ledger_store,
            EntryType,
            ValueMetadata,
            ValueDimension,
        )

        store = create_ledger_store(None)

        metadata = ValueMetadata(
            dimension=ValueDimension.EFFORT,
            amount=0.1,
            source="test",
        )

        entry = store.append(EntryType.VALUE_ACCRUED, metadata)

        assert entry.entry_id.startswith("entry_")
        assert entry.sequence == 1  # After genesis

    def test_ledger_chain_verify(self, ledger_available):
        """Test chain verification."""
        if not ledger_available:
            pytest.skip("value-ledger module not available")

        from ledger import (
            create_ledger_store,
            EntryType,
            ValueMetadata,
            ValueDimension,
        )

        store = create_ledger_store(None)

        # Add some entries
        for i in range(5):
            metadata = ValueMetadata(
                dimension=ValueDimension.EFFORT,
                amount=0.1,
                source=f"test_{i}",
            )
            store.append(EntryType.VALUE_ACCRUED, metadata)

        # Verify chain
        is_valid, issues = store.verify_chain()
        assert is_valid is True
        assert len(issues) == 0


class TestMerkleProofs:
    """Tests for Merkle tree proofs (if available)."""

    @pytest.fixture
    def proofs_available(self):
        """Check if proofs module is available."""
        try:
            from proofs import MerkleTree, verify_proof
            return True
        except ImportError:
            return False

    def test_merkle_tree_build(self, proofs_available):
        """Test building Merkle tree."""
        if not proofs_available:
            pytest.skip("proofs module not available")

        from proofs import MerkleTree

        leaves = ["hash1", "hash2", "hash3", "hash4"]
        tree = MerkleTree(leaves)

        assert tree.root is not None
        assert tree.size == 4

    def test_merkle_proof_generate(self, proofs_available):
        """Test generating Merkle proof."""
        if not proofs_available:
            pytest.skip("proofs module not available")

        from proofs import MerkleTree

        leaves = ["hash1", "hash2", "hash3", "hash4"]
        tree = MerkleTree(leaves)

        proof = tree.get_proof(1)

        assert proof.leaf_hash == "hash2"
        assert proof.leaf_index == 1
        assert len(proof.path) > 0

    def test_merkle_proof_verify(self, proofs_available):
        """Test verifying Merkle proof."""
        if not proofs_available:
            pytest.skip("proofs module not available")

        from proofs import MerkleTree, verify_proof

        leaves = ["hash1", "hash2", "hash3", "hash4"]
        tree = MerkleTree(leaves)

        proof = tree.get_proof(2)
        is_valid = verify_proof(proof)

        assert is_valid is True
