"""
Tests for Post-Quantum Key Management

Tests the PostQuantumKeyManager and related functionality.
"""

import pytest
import tempfile
from pathlib import Path
from datetime import timedelta

from src.memory.pq_keys import (
    PostQuantumKeyManager,
    QuantumKeyType,
    PQKeyPurpose,
    PQSecurityLevel,
    PQKeyMetadata,
    PQStoredKey,
    create_pq_key_manager,
    check_pq_availability,
    PQ_AVAILABLE,
)

from src.memory.keys import KeyStatus
from src.memory.profiles import EncryptionTier


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def pq_key_manager():
    """Create a PQ key manager for testing."""
    manager = PostQuantumKeyManager()
    manager.initialize()
    yield manager
    manager.shutdown()


@pytest.fixture
def pq_key_manager_with_storage():
    """Create a PQ key manager with persistent storage."""
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = PostQuantumKeyManager(
            key_store_path=Path(tmpdir),
        )
        manager.initialize()
        yield manager
        manager.shutdown()


# =============================================================================
# Basic Tests
# =============================================================================


class TestPQKeyManagerBasics:
    """Basic tests for PostQuantumKeyManager."""

    def test_initialization(self, pq_key_manager):
        """Test manager initialization."""
        assert pq_key_manager.is_available == PQ_AVAILABLE

    def test_check_availability(self):
        """Test availability check function."""
        result = check_pq_availability()

        assert "pq_available" in result
        assert "ml_kem_available" in result
        assert "ml_dsa_available" in result
        assert "supported_levels" in result
        assert "default_level" in result

        assert result["default_level"] == "level_3"

    def test_factory_function(self):
        """Test factory function."""
        manager = create_pq_key_manager(
            security_level=PQSecurityLevel.LEVEL_5,
        )

        assert manager is not None
        assert manager._default_level == PQSecurityLevel.LEVEL_5


# =============================================================================
# KEM Key Generation Tests
# =============================================================================


class TestKEMKeyGeneration:
    """Tests for key encapsulation key generation."""

    def test_generate_hybrid_kem_keypair(self, pq_key_manager):
        """Test hybrid KEM key pair generation."""
        if not pq_key_manager.is_available:
            pytest.skip("PQ crypto not available")

        stored_key = pq_key_manager.generate_kem_keypair(
            tier=EncryptionTier.PRIVATE,
            quantum_type=QuantumKeyType.HYBRID,
        )

        assert stored_key is not None
        assert stored_key.public_key is not None
        assert stored_key.private_key is not None

        metadata = stored_key.metadata
        assert metadata.quantum_type == QuantumKeyType.HYBRID
        assert metadata.purpose == PQKeyPurpose.KEY_EXCHANGE
        assert "hybrid_kem" in metadata.key_id
        assert metadata.algorithm.startswith("x25519-ml-kem")

        # Check key sizes
        assert metadata.public_key_size > 32  # Larger than classical
        assert metadata.private_key_size > 32

    def test_generate_pq_kem_keypair(self, pq_key_manager):
        """Test pure PQ KEM key pair generation."""
        if not pq_key_manager.is_available:
            pytest.skip("PQ crypto not available")

        stored_key = pq_key_manager.generate_kem_keypair(
            tier=EncryptionTier.PRIVATE,
            quantum_type=QuantumKeyType.POST_QUANTUM,
            security_level=PQSecurityLevel.LEVEL_3,
        )

        assert stored_key is not None
        metadata = stored_key.metadata

        assert metadata.quantum_type == QuantumKeyType.POST_QUANTUM
        assert "mlkem" in metadata.key_id
        assert metadata.algorithm == "ml-kem-768"

    def test_kem_different_security_levels(self, pq_key_manager):
        """Test KEM generation at different security levels."""
        if not pq_key_manager.is_available:
            pytest.skip("PQ crypto not available")

        for level in PQSecurityLevel:
            stored_key = pq_key_manager.generate_kem_keypair(
                tier=EncryptionTier.PRIVATE,
                security_level=level,
                quantum_type=QuantumKeyType.HYBRID,
            )

            assert stored_key.metadata.security_level == level

    def test_kem_with_ttl(self, pq_key_manager):
        """Test KEM generation with TTL."""
        if not pq_key_manager.is_available:
            pytest.skip("PQ crypto not available")

        stored_key = pq_key_manager.generate_kem_keypair(
            tier=EncryptionTier.PRIVATE,
            ttl=timedelta(hours=1),
        )

        assert stored_key.metadata.expires_at is not None


# =============================================================================
# Signing Key Generation Tests
# =============================================================================


class TestSigningKeyGeneration:
    """Tests for signing key generation."""

    def test_generate_hybrid_signing_keypair(self, pq_key_manager):
        """Test hybrid signing key pair generation."""
        if not pq_key_manager.is_available:
            pytest.skip("PQ crypto not available")

        stored_key = pq_key_manager.generate_signing_keypair(
            tier=EncryptionTier.PRIVATE,
            quantum_type=QuantumKeyType.HYBRID,
        )

        assert stored_key is not None
        metadata = stored_key.metadata

        assert metadata.quantum_type == QuantumKeyType.HYBRID
        assert metadata.purpose == PQKeyPurpose.SIGNING
        assert "hybrid_sig" in metadata.key_id
        assert metadata.algorithm.startswith("ed25519-ml-dsa")

    def test_generate_pq_signing_keypair(self, pq_key_manager):
        """Test pure PQ signing key pair generation."""
        if not pq_key_manager.is_available:
            pytest.skip("PQ crypto not available")

        stored_key = pq_key_manager.generate_signing_keypair(
            tier=EncryptionTier.PRIVATE,
            quantum_type=QuantumKeyType.POST_QUANTUM,
        )

        assert stored_key is not None
        metadata = stored_key.metadata

        assert metadata.quantum_type == QuantumKeyType.POST_QUANTUM
        assert "mldsa" in metadata.key_id


# =============================================================================
# Key Retrieval Tests
# =============================================================================


class TestKeyRetrieval:
    """Tests for key retrieval."""

    def test_get_key(self, pq_key_manager):
        """Test key retrieval."""
        if not pq_key_manager.is_available:
            pytest.skip("PQ crypto not available")

        stored_key = pq_key_manager.generate_kem_keypair()
        key_id = stored_key.key_id

        retrieved = pq_key_manager.get_key(key_id)

        assert retrieved is not None
        assert retrieved.public_key == stored_key.public_key
        assert retrieved.private_key == stored_key.private_key

    def test_get_public_key(self, pq_key_manager):
        """Test public key retrieval."""
        if not pq_key_manager.is_available:
            pytest.skip("PQ crypto not available")

        stored_key = pq_key_manager.generate_kem_keypair()
        key_id = stored_key.key_id

        public_key = pq_key_manager.get_public_key(key_id)

        assert public_key is not None
        assert public_key == stored_key.public_key

    def test_get_private_key(self, pq_key_manager):
        """Test private key retrieval."""
        if not pq_key_manager.is_available:
            pytest.skip("PQ crypto not available")

        stored_key = pq_key_manager.generate_kem_keypair()
        key_id = stored_key.key_id

        private_key = pq_key_manager.get_private_key(key_id)

        assert private_key is not None
        assert private_key == stored_key.private_key

    def test_get_nonexistent_key(self, pq_key_manager):
        """Test retrieval of non-existent key."""
        result = pq_key_manager.get_key("nonexistent_key")
        assert result is None

    def test_get_metadata(self, pq_key_manager):
        """Test metadata retrieval."""
        if not pq_key_manager.is_available:
            pytest.skip("PQ crypto not available")

        stored_key = pq_key_manager.generate_kem_keypair()
        key_id = stored_key.key_id

        metadata = pq_key_manager.get_metadata(key_id)

        assert metadata is not None
        assert metadata.key_id == key_id


# =============================================================================
# Key Operations Tests
# =============================================================================


class TestKeyOperations:
    """Tests for key encapsulation/decapsulation."""

    def test_encapsulate_decapsulate_hybrid(self, pq_key_manager):
        """Test hybrid encapsulation and decapsulation."""
        if not pq_key_manager.is_available:
            pytest.skip("PQ crypto not available")

        # Generate recipient's key pair
        recipient_key = pq_key_manager.generate_kem_keypair(
            quantum_type=QuantumKeyType.HYBRID,
        )

        # Encapsulate
        result = pq_key_manager.encapsulate(recipient_key.key_id)
        assert result is not None

        shared_secret, ciphertext = result
        assert len(shared_secret) == 32
        assert len(ciphertext) > 0

        # Decapsulate
        recovered_secret = pq_key_manager.decapsulate(
            recipient_key.key_id,
            ciphertext,
        )

        assert recovered_secret is not None
        assert len(recovered_secret) == 32

    def test_encapsulate_decapsulate_pq(self, pq_key_manager):
        """Test pure PQ encapsulation and decapsulation."""
        if not pq_key_manager.is_available:
            pytest.skip("PQ crypto not available")

        recipient_key = pq_key_manager.generate_kem_keypair(
            quantum_type=QuantumKeyType.POST_QUANTUM,
        )

        result = pq_key_manager.encapsulate(recipient_key.key_id)
        assert result is not None

        shared_secret, ciphertext = result

        recovered_secret = pq_key_manager.decapsulate(
            recipient_key.key_id,
            ciphertext,
        )

        assert recovered_secret is not None


# =============================================================================
# Key Lifecycle Tests
# =============================================================================


class TestKeyLifecycle:
    """Tests for key lifecycle management."""

    def test_rotate_key(self, pq_key_manager):
        """Test key rotation."""
        if not pq_key_manager.is_available:
            pytest.skip("PQ crypto not available")

        old_key = pq_key_manager.generate_kem_keypair()
        old_key_id = old_key.key_id

        new_key = pq_key_manager.rotate_key(old_key_id)

        assert new_key is not None
        assert new_key.key_id != old_key_id

        # Check old key is marked for rotation
        old_metadata = pq_key_manager.get_metadata(old_key_id)
        assert old_metadata.status == KeyStatus.PENDING_ROTATION

        # Check new key links to old
        assert new_key.metadata.metadata.get("rotated_from") == old_key_id

    def test_revoke_key(self, pq_key_manager):
        """Test key revocation."""
        if not pq_key_manager.is_available:
            pytest.skip("PQ crypto not available")

        stored_key = pq_key_manager.generate_kem_keypair()
        key_id = stored_key.key_id

        result = pq_key_manager.revoke_key(key_id)
        assert result is True

        # Key should no longer be retrievable
        retrieved = pq_key_manager.get_key(key_id)
        assert retrieved is None

        # Metadata should show revoked
        metadata = pq_key_manager.get_metadata(key_id)
        assert metadata.status == KeyStatus.REVOKED

    def test_delete_key(self, pq_key_manager):
        """Test key deletion."""
        if not pq_key_manager.is_available:
            pytest.skip("PQ crypto not available")

        stored_key = pq_key_manager.generate_kem_keypair()
        key_id = stored_key.key_id

        result = pq_key_manager.delete_key(key_id, secure=True)
        assert result is True

        # Key should be completely gone
        assert pq_key_manager.get_key(key_id) is None
        assert pq_key_manager.get_metadata(key_id) is None


# =============================================================================
# Key Listing Tests
# =============================================================================


class TestKeyListing:
    """Tests for key listing and statistics."""

    def test_list_keys(self, pq_key_manager):
        """Test key listing."""
        if not pq_key_manager.is_available:
            pytest.skip("PQ crypto not available")

        # Generate some keys
        pq_key_manager.generate_kem_keypair(quantum_type=QuantumKeyType.HYBRID)
        pq_key_manager.generate_kem_keypair(quantum_type=QuantumKeyType.POST_QUANTUM)
        pq_key_manager.generate_signing_keypair(quantum_type=QuantumKeyType.HYBRID)

        # List all
        all_keys = pq_key_manager.list_keys()
        assert len(all_keys) == 3

        # Filter by type
        hybrid_keys = pq_key_manager.list_keys(quantum_type=QuantumKeyType.HYBRID)
        assert len(hybrid_keys) == 2

        # Filter by purpose
        kem_keys = pq_key_manager.list_keys(purpose=PQKeyPurpose.KEY_EXCHANGE)
        assert len(kem_keys) == 2

    def test_get_key_stats(self, pq_key_manager):
        """Test key statistics."""
        if not pq_key_manager.is_available:
            pytest.skip("PQ crypto not available")

        # Generate some keys
        pq_key_manager.generate_kem_keypair(quantum_type=QuantumKeyType.HYBRID)
        pq_key_manager.generate_signing_keypair(quantum_type=QuantumKeyType.POST_QUANTUM)

        stats = pq_key_manager.get_key_stats()

        assert stats["total_keys"] == 2
        assert stats["by_type"]["hybrid"] == 1
        assert stats["by_type"]["post_quantum"] == 1
        assert stats["by_purpose"]["key_exchange"] == 1
        assert stats["by_purpose"]["signing"] == 1
        assert stats["total_public_key_bytes"] > 0


# =============================================================================
# Persistence Tests
# =============================================================================


class TestPersistence:
    """Tests for key persistence."""

    def test_persist_and_load_keys(self):
        """Test key persistence and loading."""
        with tempfile.TemporaryDirectory() as tmpdir:
            key_store_path = Path(tmpdir)

            # Create manager and generate keys
            manager1 = PostQuantumKeyManager(key_store_path=key_store_path)
            manager1.initialize()

            if not manager1.is_available:
                pytest.skip("PQ crypto not available")

            stored_key = manager1.generate_kem_keypair()
            key_id = stored_key.key_id
            public_key = stored_key.public_key

            manager1.shutdown()

            # Create new manager and load keys
            manager2 = PostQuantumKeyManager(key_store_path=key_store_path)
            manager2.initialize()

            # Key should be loaded
            loaded_key = manager2.get_key(key_id)
            assert loaded_key is not None
            assert loaded_key.public_key == public_key

            manager2.shutdown()


# =============================================================================
# Metadata Tests
# =============================================================================


class TestPQKeyMetadata:
    """Tests for PQKeyMetadata."""

    def test_metadata_serialization(self):
        """Test metadata serialization."""
        metadata = PQKeyMetadata(
            key_id="test_key_123",
            quantum_type=QuantumKeyType.HYBRID,
            purpose=PQKeyPurpose.KEY_EXCHANGE,
            security_level=PQSecurityLevel.LEVEL_3,
            algorithm="x25519-ml-kem-768",
            public_key_size=1216,
            private_key_size=2432,
            tier=EncryptionTier.PRIVATE,
            binding=KeyBinding.SOFTWARE,
        )

        # Serialize
        data = metadata.to_dict()

        assert data["key_id"] == "test_key_123"
        assert data["quantum_type"] == "hybrid"
        assert data["purpose"] == "key_exchange"
        assert data["security_level"] == "level_3"

        # Deserialize
        restored = PQKeyMetadata.from_dict(data)

        assert restored.key_id == metadata.key_id
        assert restored.quantum_type == metadata.quantum_type
        assert restored.purpose == metadata.purpose
        assert restored.security_level == metadata.security_level


# Import KeyBinding for tests
from src.memory.profiles import KeyBinding


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
