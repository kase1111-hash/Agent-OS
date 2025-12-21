"""
Tests for Agent OS Memory Vault (UC-006)

Covers:
- Encryption profiles
- Key management
- Blob storage with AES-256-GCM
- Index database
- Consent verification
- Right-to-delete
- Genesis proofs
- Main vault API
"""

import os
import tempfile
import pytest
import shutil
from datetime import datetime, timedelta
from pathlib import Path

from src.memory.profiles import (
    EncryptionTier,
    EncryptionProfile,
    KeyDerivation,
    KeyBinding,
    ProfileManager,
    WORKING_PROFILE,
    PRIVATE_PROFILE,
    SEALED_PROFILE,
    VAULTED_PROFILE,
)
from src.memory.keys import (
    KeyManager,
    KeyStatus,
    DerivedKey,
)
from src.memory.storage import (
    BlobStorage,
    BlobMetadata,
    BlobType,
    BlobStatus,
)
from src.memory.index import (
    VaultIndex,
    ConsentRecord,
    AccessType,
)
from src.memory.consent import (
    ConsentManager,
    ConsentOperation,
    ConsentStatus,
    ConsentPolicy,
)
from src.memory.deletion import (
    DeletionManager,
    DeletionScope,
    DeletionStatus,
)
from src.memory.genesis import (
    GenesisProofSystem,
    GenesisRecord,
)
from src.memory.vault import (
    MemoryVault,
    VaultConfig,
    StoreResult,
    create_vault,
)


@pytest.fixture
def temp_dir():
    """Create temporary directory for tests."""
    path = Path(tempfile.mkdtemp())
    yield path
    shutil.rmtree(path)


@pytest.fixture
def profile_manager():
    """Create profile manager."""
    return ProfileManager()


@pytest.fixture
def key_manager(temp_dir):
    """Create key manager."""
    km = KeyManager(key_store_path=temp_dir / "keys")
    km.initialize()
    yield km
    km.shutdown()


@pytest.fixture
def vault_index(temp_dir):
    """Create vault index."""
    index = VaultIndex(temp_dir / "index.db")
    yield index
    index.close()


@pytest.fixture
def blob_storage(temp_dir, key_manager, profile_manager):
    """Create blob storage."""
    return BlobStorage(
        storage_path=temp_dir / "blobs",
        key_manager=key_manager,
        profile_manager=profile_manager,
    )


@pytest.fixture
def consent_manager(vault_index):
    """Create consent manager."""
    return ConsentManager(index=vault_index)


@pytest.fixture
def memory_vault(temp_dir):
    """Create and initialize a memory vault."""
    config = VaultConfig(
        vault_path=temp_dir / "vault",
        owner="test_user",
        enable_hardware_binding=False,
        enable_ttl_enforcement=False,
        strict_consent=False,  # Don't require consent for reads in tests
    )
    # Auto-grant consent for tests
    vault = MemoryVault(config, consent_prompt=lambda req: True)
    vault.initialize()
    yield vault
    vault.shutdown()


# ============================================================================
# Profile Tests
# ============================================================================

class TestEncryptionProfiles:
    """Tests for encryption profiles."""

    def test_default_profiles_exist(self, profile_manager):
        """Verify default profiles are available."""
        assert profile_manager.get_profile(EncryptionTier.WORKING) is not None
        assert profile_manager.get_profile(EncryptionTier.PRIVATE) is not None
        assert profile_manager.get_profile(EncryptionTier.SEALED) is not None
        assert profile_manager.get_profile(EncryptionTier.VAULTED) is not None

    def test_working_profile_config(self, profile_manager):
        """Verify working profile configuration."""
        profile = profile_manager.get_profile(EncryptionTier.WORKING)
        assert profile.name == "Working"
        assert profile.requires_unlock is False
        assert profile.requires_consent is False
        assert profile.key_binding == KeyBinding.SOFTWARE

    def test_private_profile_config(self, profile_manager):
        """Verify private profile configuration."""
        profile = profile_manager.get_profile(EncryptionTier.PRIVATE)
        assert profile.name == "Private"
        assert profile.requires_consent is True
        assert profile.key_derivation == KeyDerivation.ARGON2ID

    def test_sealed_profile_requires_unlock(self, profile_manager):
        """Verify sealed profile requires unlock."""
        profile = profile_manager.get_profile(EncryptionTier.SEALED)
        assert profile.requires_unlock is True
        assert profile.max_access_count is not None

    def test_vaulted_profile_hardware_bound(self, profile_manager):
        """Verify vaulted profile is hardware-bound."""
        profile = profile_manager.get_profile(EncryptionTier.VAULTED)
        assert profile.key_binding == KeyBinding.HARDWARE
        assert profile.requires_unlock is True

    def test_tier_access_hierarchy(self, profile_manager):
        """Verify tier access hierarchy."""
        assert profile_manager.can_access(EncryptionTier.VAULTED, EncryptionTier.WORKING)
        assert profile_manager.can_access(EncryptionTier.SEALED, EncryptionTier.PRIVATE)
        assert not profile_manager.can_access(EncryptionTier.WORKING, EncryptionTier.VAULTED)
        assert not profile_manager.can_access(EncryptionTier.PRIVATE, EncryptionTier.SEALED)

    def test_tier_promotion(self, profile_manager):
        """Verify tier promotion rules."""
        assert profile_manager.can_promote(EncryptionTier.WORKING, EncryptionTier.PRIVATE)
        assert profile_manager.can_promote(EncryptionTier.PRIVATE, EncryptionTier.SEALED)
        assert not profile_manager.can_promote(EncryptionTier.SEALED, EncryptionTier.PRIVATE)

    def test_get_profile_by_name(self, profile_manager):
        """Test getting profile by name."""
        profile = profile_manager.get_profile_by_name("Sealed")
        assert profile is not None
        assert profile.tier == EncryptionTier.SEALED

    def test_minimum_tier_for_sensitive_content(self, profile_manager):
        """Test minimum tier determination for sensitive content."""
        tier = profile_manager.get_minimum_tier_for_content("password_store")
        assert tier == EncryptionTier.SEALED

        tier = profile_manager.get_minimum_tier_for_content("user_notes")
        assert tier == EncryptionTier.PRIVATE


# ============================================================================
# Key Manager Tests
# ============================================================================

class TestKeyManager:
    """Tests for key management."""

    def test_initialization(self, key_manager):
        """Verify key manager initializes."""
        assert key_manager._initialized is True

    def test_generate_key(self, key_manager):
        """Test key generation."""
        key = key_manager.generate_key(EncryptionTier.PRIVATE, purpose="test")
        assert key is not None
        assert key.key_id.startswith("private_")
        assert len(key.key) == 32  # 256 bits

    def test_generate_key_for_each_tier(self, key_manager):
        """Test key generation for each tier."""
        for tier in EncryptionTier:
            key = key_manager.generate_key(tier)
            assert key is not None
            assert key.tier == tier

    def test_retrieve_key(self, key_manager):
        """Test key retrieval."""
        generated = key_manager.generate_key(EncryptionTier.PRIVATE)
        retrieved = key_manager.get_key(generated.key_id)
        assert retrieved == generated.key

    def test_key_expiry(self, key_manager):
        """Test key expiry."""
        key = key_manager.generate_key(
            EncryptionTier.WORKING,
            ttl=timedelta(seconds=-1),  # Already expired
        )
        retrieved = key_manager.get_key(key.key_id)
        assert retrieved is None

    def test_revoke_key(self, key_manager):
        """Test key revocation."""
        key = key_manager.generate_key(EncryptionTier.PRIVATE)
        assert key_manager.revoke_key(key.key_id) is True

        metadata = key_manager.get_key_metadata(key.key_id)
        assert metadata.status == KeyStatus.REVOKED

        # Cannot retrieve revoked key
        assert key_manager.get_key(key.key_id) is None

    def test_delete_key(self, key_manager):
        """Test key deletion."""
        key = key_manager.generate_key(EncryptionTier.PRIVATE)
        assert key_manager.delete_key(key.key_id) is True
        assert key_manager.get_key_metadata(key.key_id) is None

    def test_key_rotation(self, key_manager):
        """Test key rotation."""
        old_key = key_manager.generate_key(EncryptionTier.PRIVATE)
        new_key = key_manager.rotate_key(old_key.key_id)

        assert new_key is not None
        assert new_key.key_id != old_key.key_id
        assert new_key.tier == old_key.tier

        old_metadata = key_manager.get_key_metadata(old_key.key_id)
        assert old_metadata.status == KeyStatus.PENDING_ROTATION

    def test_derive_key(self, key_manager):
        """Test key derivation from password."""
        password = b"test_password"
        salt = os.urandom(32)

        key = key_manager.derive_key(password, salt, EncryptionTier.PRIVATE)
        assert key is not None
        assert len(key.key) == 32

        # Same password/salt should produce same key
        key2 = key_manager.derive_key(password, salt, EncryptionTier.PRIVATE)
        assert key.key == key2.key

    def test_list_keys(self, key_manager):
        """Test listing keys."""
        key_manager.generate_key(EncryptionTier.WORKING)
        key_manager.generate_key(EncryptionTier.PRIVATE)
        key_manager.generate_key(EncryptionTier.PRIVATE)

        all_keys = key_manager.list_keys()
        assert len(all_keys) >= 3

        private_keys = key_manager.list_keys(tier=EncryptionTier.PRIVATE)
        assert len(private_keys) >= 2


# ============================================================================
# Blob Storage Tests
# ============================================================================

class TestBlobStorage:
    """Tests for encrypted blob storage."""

    def test_store_and_retrieve(self, blob_storage):
        """Test basic store and retrieve."""
        data = b"Hello, Memory Vault!"
        metadata = blob_storage.store(data, EncryptionTier.PRIVATE)

        assert metadata is not None
        assert metadata.blob_id is not None
        assert metadata.size_bytes == len(data)

        retrieved = blob_storage.retrieve(metadata.blob_id)
        assert retrieved == data

    def test_store_text(self, blob_storage):
        """Test text storage."""
        text = "This is test text content."
        metadata = blob_storage.store_text(text, EncryptionTier.PRIVATE)

        assert metadata.blob_type == BlobType.TEXT
        retrieved = blob_storage.retrieve_text(metadata.blob_id)
        assert retrieved == text

    def test_store_json(self, blob_storage):
        """Test JSON storage."""
        data = {"key": "value", "number": 42, "nested": {"a": 1}}
        metadata = blob_storage.store_json(data, EncryptionTier.PRIVATE)

        assert metadata.blob_type == BlobType.JSON
        retrieved = blob_storage.retrieve_json(metadata.blob_id)
        assert retrieved == data

    def test_content_hash_verification(self, blob_storage):
        """Test content hash is verified on retrieval."""
        data = b"Test data for hash verification"
        metadata = blob_storage.store(data, EncryptionTier.PRIVATE)

        import hashlib
        expected_hash = hashlib.sha256(data).hexdigest()
        assert metadata.content_hash == expected_hash

    def test_store_with_tags(self, blob_storage):
        """Test storage with tags."""
        data = b"Tagged data"
        tags = ["test", "important", "demo"]
        metadata = blob_storage.store(data, EncryptionTier.PRIVATE, tags=tags)

        assert set(metadata.tags) == set(tags)

    def test_store_with_ttl(self, blob_storage):
        """Test storage with TTL."""
        data = b"Expiring data"
        metadata = blob_storage.store(
            data,
            EncryptionTier.PRIVATE,
            ttl_seconds=3600,
        )
        assert metadata.ttl_seconds == 3600

    def test_delete_blob(self, blob_storage):
        """Test blob deletion."""
        data = b"Data to delete"
        metadata = blob_storage.store(data, EncryptionTier.PRIVATE)

        assert blob_storage.delete(metadata.blob_id) is True
        assert blob_storage.retrieve(metadata.blob_id) is None

    def test_seal_and_unseal(self, blob_storage):
        """Test blob sealing."""
        data = b"Sealable data"
        metadata = blob_storage.store(data, EncryptionTier.SEALED)

        assert blob_storage.seal(metadata.blob_id) is True

        sealed_meta = blob_storage.get_metadata(metadata.blob_id)
        assert sealed_meta.status == BlobStatus.SEALED

        assert blob_storage.unseal(metadata.blob_id) is True
        retrieved = blob_storage.retrieve(metadata.blob_id)
        assert retrieved == data

    def test_encryption_different_for_each_store(self, blob_storage):
        """Verify encryption is different for same data (random nonce)."""
        data = b"Same data stored twice"
        meta1 = blob_storage.store(data, EncryptionTier.PRIVATE)
        meta2 = blob_storage.store(data, EncryptionTier.PRIVATE)

        # Different blob IDs
        assert meta1.blob_id != meta2.blob_id

        # Both should decrypt to same data
        assert blob_storage.retrieve(meta1.blob_id) == data
        assert blob_storage.retrieve(meta2.blob_id) == data

    def test_list_blobs(self, blob_storage):
        """Test listing blobs."""
        blob_storage.store(b"data1", EncryptionTier.PRIVATE, tags=["tag1"])
        blob_storage.store(b"data2", EncryptionTier.PRIVATE, tags=["tag2"])
        blob_storage.store(b"data3", EncryptionTier.SEALED)

        all_blobs = blob_storage.list_blobs()
        assert len(all_blobs) >= 3

        private_blobs = blob_storage.list_blobs(tier=EncryptionTier.PRIVATE)
        assert len(private_blobs) >= 2


# ============================================================================
# Index Tests
# ============================================================================

class TestVaultIndex:
    """Tests for vault index database."""

    def test_index_blob(self, vault_index):
        """Test indexing a blob."""
        metadata = BlobMetadata(
            blob_id="test_blob_1",
            key_id="key_1",
            tier=EncryptionTier.PRIVATE,
            blob_type=BlobType.BINARY,
            size_bytes=100,
            encrypted_size=120,
            content_hash="abc123",
            created_at=datetime.now(),
            modified_at=datetime.now(),
        )
        vault_index.index_blob(metadata)

        retrieved = vault_index.get_blob("test_blob_1")
        assert retrieved is not None
        assert retrieved.blob_id == "test_blob_1"

    def test_query_blobs_by_tier(self, vault_index):
        """Test querying blobs by tier."""
        for i in range(3):
            vault_index.index_blob(BlobMetadata(
                blob_id=f"private_{i}",
                key_id=f"key_{i}",
                tier=EncryptionTier.PRIVATE,
                blob_type=BlobType.TEXT,
                size_bytes=100,
                encrypted_size=120,
                content_hash=f"hash_{i}",
                created_at=datetime.now(),
                modified_at=datetime.now(),
            ))

        vault_index.index_blob(BlobMetadata(
            blob_id="sealed_0",
            key_id="key_s0",
            tier=EncryptionTier.SEALED,
            blob_type=BlobType.BINARY,
            size_bytes=100,
            encrypted_size=120,
            content_hash="hash_s0",
            created_at=datetime.now(),
            modified_at=datetime.now(),
        ))

        private_blobs = vault_index.query_blobs(tier=EncryptionTier.PRIVATE)
        assert len(private_blobs) == 3

        sealed_blobs = vault_index.query_blobs(tier=EncryptionTier.SEALED)
        assert len(sealed_blobs) == 1

    def test_record_consent(self, vault_index):
        """Test recording consent."""
        consent = ConsentRecord(
            consent_id="consent_1",
            granted_by="user",
            granted_at=datetime.now(),
            expires_at=datetime.now() + timedelta(days=30),
            scope="test_scope",
            operations=["read", "write"],
            active=True,
        )
        vault_index.record_consent(consent)

        retrieved = vault_index.get_consent("consent_1")
        assert retrieved is not None
        assert retrieved.scope == "test_scope"

    def test_revoke_consent(self, vault_index):
        """Test revoking consent."""
        consent = ConsentRecord(
            consent_id="consent_revoke",
            granted_by="user",
            granted_at=datetime.now(),
            expires_at=None,
            scope="revoke_test",
            operations=["read"],
            active=True,
        )
        vault_index.record_consent(consent)

        assert vault_index.revoke_consent("consent_revoke") is True

        retrieved = vault_index.get_consent("consent_revoke")
        assert retrieved.active is False

    def test_log_access(self, vault_index):
        """Test access logging."""
        vault_index.log_access(
            blob_id="test_blob",
            access_type=AccessType.READ,
            accessor="user",
            success=True,
            details={"ip": "127.0.0.1"},
        )

        logs = vault_index.get_access_log(blob_id="test_blob")
        assert len(logs) >= 1
        assert logs[0].access_type == AccessType.READ

    def test_statistics(self, vault_index):
        """Test statistics gathering."""
        for i in range(5):
            vault_index.index_blob(BlobMetadata(
                blob_id=f"stat_blob_{i}",
                key_id=f"key_{i}",
                tier=EncryptionTier.PRIVATE,
                blob_type=BlobType.TEXT,
                size_bytes=100,
                encrypted_size=120,
                content_hash=f"hash_{i}",
                created_at=datetime.now(),
                modified_at=datetime.now(),
            ))

        stats = vault_index.get_statistics()
        assert stats["total_blobs"] >= 5


# ============================================================================
# Consent Tests
# ============================================================================

class TestConsentManager:
    """Tests for consent management."""

    def test_grant_consent(self, consent_manager):
        """Test granting consent."""
        consent = consent_manager.grant_consent(
            granted_by="user",
            scope="test_scope",
            operations=[ConsentOperation.READ, ConsentOperation.WRITE],
            duration=timedelta(days=30),
        )

        assert consent is not None
        assert consent.active is True
        assert "read" in consent.operations

    def test_verify_consent(self, consent_manager):
        """Test consent verification."""
        consent = consent_manager.grant_consent(
            granted_by="user",
            scope="verify_test",
            operations=[ConsentOperation.READ],
        )

        decision = consent_manager.verify_consent(
            consent_id=consent.consent_id,
            operation=ConsentOperation.READ,
        )
        assert decision.status == ConsentStatus.GRANTED

        # Operation not covered
        decision = consent_manager.verify_consent(
            consent_id=consent.consent_id,
            operation=ConsentOperation.WRITE,
        )
        assert decision.status == ConsentStatus.INSUFFICIENT

    def test_revoke_consent(self, consent_manager):
        """Test consent revocation."""
        consent = consent_manager.grant_consent(
            granted_by="user",
            scope="revoke_test",
            operations=[ConsentOperation.READ],
        )

        assert consent_manager.revoke_consent(
            consent.consent_id,
            revoked_by="admin",
        ) is True

        decision = consent_manager.verify_consent(
            consent_id=consent.consent_id,
            operation=ConsentOperation.READ,
        )
        assert decision.status == ConsentStatus.REVOKED

    def test_consent_policy(self):
        """Test consent policy."""
        policy = ConsentPolicy()

        # Write always requires consent
        assert policy.requires_consent(ConsentOperation.WRITE, EncryptionTier.WORKING) is True

        # Learning always requires consent
        assert policy.requires_consent(ConsentOperation.LEARN, EncryptionTier.PRIVATE) is True

        # Prohibited operations
        assert policy.is_prohibited("silent_storage") is True
        assert policy.is_prohibited("normal_operation") is False


# ============================================================================
# Deletion Tests
# ============================================================================

class TestDeletionManager:
    """Tests for deletion management."""

    def test_delete_single_blob(self, blob_storage, vault_index, key_manager):
        """Test single blob deletion."""
        data = b"Data to delete"
        metadata = blob_storage.store(data, EncryptionTier.PRIVATE)
        vault_index.index_blob(metadata)

        deletion_manager = DeletionManager(
            storage=blob_storage,
            index=vault_index,
            key_manager=key_manager,
        )

        result = deletion_manager.delete_blob(metadata.blob_id, "user")
        assert result.status == DeletionStatus.COMPLETED
        assert result.blobs_deleted == 1

    def test_delete_by_consent(self, blob_storage, vault_index, key_manager, consent_manager):
        """Test deletion by consent cascade."""
        consent = consent_manager.grant_consent(
            granted_by="user",
            scope="delete_test",
            operations=[ConsentOperation.WRITE, ConsentOperation.DELETE],
        )

        # Store blobs with consent
        for i in range(3):
            metadata = blob_storage.store(
                f"Data {i}".encode(),
                EncryptionTier.PRIVATE,
                consent_id=consent.consent_id,
            )
            vault_index.index_blob(metadata)

        deletion_manager = DeletionManager(
            storage=blob_storage,
            index=vault_index,
            key_manager=key_manager,
        )

        result = deletion_manager.delete_by_consent(
            consent.consent_id,
            requested_by="user",
        )

        assert result.blobs_deleted == 3


# ============================================================================
# Genesis Proof Tests
# ============================================================================

class TestGenesisProofs:
    """Tests for genesis proof system."""

    def test_create_genesis(self, vault_index, temp_dir):
        """Test genesis record creation."""
        genesis = GenesisProofSystem(
            index=vault_index,
            proof_dir=temp_dir / "proofs",
        )

        record = genesis.create_genesis(
            vault_id="test_vault_1",
            created_by="test_user",
            encryption_profiles=["WORKING", "PRIVATE", "SEALED", "VAULTED"],
        )

        assert record is not None
        assert record.vault_id == "test_vault_1"
        assert record.proof_id.startswith("genesis_")

    def test_verify_genesis(self, vault_index, temp_dir):
        """Test genesis verification."""
        genesis = GenesisProofSystem(
            index=vault_index,
            proof_dir=temp_dir / "proofs",
        )

        genesis.create_genesis(
            vault_id="verify_test",
            created_by="user",
            encryption_profiles=["PRIVATE"],
        )

        is_valid, message = genesis.verify_genesis()
        assert is_valid is True

    def test_create_integrity_proof(self, vault_index, temp_dir):
        """Test integrity proof creation."""
        genesis = GenesisProofSystem(
            index=vault_index,
            proof_dir=temp_dir / "proofs",
        )

        genesis.create_genesis(
            vault_id="integrity_test",
            created_by="user",
            encryption_profiles=["PRIVATE"],
        )

        proof = genesis.create_integrity_proof()
        assert proof is not None
        assert proof.genesis_proof_id == genesis.get_genesis().proof_id

    def test_verify_chain(self, vault_index, temp_dir):
        """Test proof chain verification."""
        genesis = GenesisProofSystem(
            index=vault_index,
            proof_dir=temp_dir / "proofs",
        )

        genesis.create_genesis(
            vault_id="chain_test",
            created_by="user",
            encryption_profiles=["PRIVATE"],
        )

        genesis.create_integrity_proof()
        genesis.create_integrity_proof()

        is_valid, issues = genesis.verify_chain()
        assert is_valid is True
        assert len(issues) == 0


# ============================================================================
# Memory Vault Integration Tests
# ============================================================================

class TestMemoryVault:
    """Integration tests for Memory Vault."""

    def test_vault_initialization(self, memory_vault):
        """Test vault initializes correctly."""
        assert memory_vault.is_initialized is True
        assert memory_vault.vault_id is not None

    def test_store_and_retrieve(self, memory_vault):
        """Test basic store and retrieve through vault API."""
        # Grant consent for read and write
        memory_vault.grant_consent(
            scope="testing",
            operations=[ConsentOperation.READ, ConsentOperation.WRITE],
            granted_by="test_user",
        )

        result = memory_vault.store(
            data=b"Test data",
            tier=EncryptionTier.PRIVATE,
            requestor="test_user",
            purpose="testing",
        )

        assert result.success is True
        assert result.blob_id is not None

        retrieved = memory_vault.retrieve(result.blob_id)
        assert retrieved.success is True
        assert retrieved.data == b"Test data"

    def test_store_text(self, memory_vault):
        """Test text storage."""
        # Grant consent for read and write
        memory_vault.grant_consent(
            scope="text_test",
            operations=[ConsentOperation.READ, ConsentOperation.WRITE],
            granted_by="user",
        )

        result = memory_vault.store_text(
            "Hello, Vault!",
            tier=EncryptionTier.PRIVATE,
            requestor="user",
            purpose="text_test",
        )

        assert result.success is True

        retrieved = memory_vault.retrieve(result.blob_id)
        assert retrieved.text == "Hello, Vault!"

    def test_store_json(self, memory_vault):
        """Test JSON storage."""
        # Grant consent for read and write
        memory_vault.grant_consent(
            scope="json_test",
            operations=[ConsentOperation.READ, ConsentOperation.WRITE],
            granted_by="user",
        )

        data = {"name": "test", "values": [1, 2, 3]}
        result = memory_vault.store_json(
            data,
            tier=EncryptionTier.PRIVATE,
            requestor="user",
            purpose="json_test",
        )

        assert result.success is True

        retrieved = memory_vault.retrieve(result.blob_id)
        assert retrieved.json_data == data

    def test_delete(self, memory_vault):
        """Test deletion."""
        result = memory_vault.store(
            data=b"Delete me",
            tier=EncryptionTier.PRIVATE,
            requestor="user",
            purpose="delete_test",
        )

        deletion = memory_vault.delete(result.blob_id, requestor="user")
        assert deletion.status == DeletionStatus.COMPLETED

        retrieved = memory_vault.retrieve(result.blob_id)
        assert retrieved.success is False

    def test_consent_management(self, memory_vault):
        """Test consent management through vault."""
        consent = memory_vault.grant_consent(
            scope="vault_test",
            operations=[ConsentOperation.READ, ConsentOperation.WRITE],
            granted_by="user",
        )

        assert consent.active is True

        consents = memory_vault.list_consents()
        assert any(c.consent_id == consent.consent_id for c in consents)

        assert memory_vault.revoke_consent(consent.consent_id) is True

    def test_right_to_forget(self, memory_vault):
        """Test right-to-forget functionality."""
        # First grant consent and store data
        consent = memory_vault.grant_consent(
            scope="forget_test",
            operations=[ConsentOperation.WRITE, ConsentOperation.DELETE],
            granted_by="user",
        )

        # Store some data with this consent
        result1 = memory_vault.store(
            data=b"Data 1",
            tier=EncryptionTier.PRIVATE,
            requestor="user",
            purpose="forget_test",
        )

        result2 = memory_vault.store(
            data=b"Data 2",
            tier=EncryptionTier.PRIVATE,
            requestor="user",
            purpose="forget_test",
        )

        assert result1.success is True
        assert result2.success is True

        # Exercise right to forget
        deletion = memory_vault.forget(consent.consent_id, requestor="user")

        # Consent should be revoked
        consents = memory_vault.list_consents()
        assert not any(c.consent_id == consent.consent_id and c.active for c in consents)

    def test_seal_and_unseal(self, memory_vault):
        """Test blob sealing through vault."""
        result = memory_vault.store(
            data=b"Sealable",
            tier=EncryptionTier.SEALED,
            requestor="user",
            purpose="seal_test",
        )

        assert memory_vault.seal(result.blob_id) is True
        assert memory_vault.unseal(result.blob_id) is True

    def test_integrity_verification(self, memory_vault):
        """Test integrity proof creation and verification."""
        # Store some data
        memory_vault.store(
            data=b"Integrity test",
            tier=EncryptionTier.PRIVATE,
            requestor="user",
            purpose="integrity",
        )

        proof = memory_vault.create_integrity_proof()
        assert proof is not None

        is_valid, message = memory_vault.verify_integrity()
        assert is_valid is True

    def test_genesis_record(self, memory_vault):
        """Test genesis record access."""
        genesis = memory_vault.get_genesis()
        assert genesis is not None
        assert genesis.vault_id == memory_vault.vault_id

    def test_statistics(self, memory_vault):
        """Test statistics gathering."""
        memory_vault.store(
            data=b"Stats test",
            tier=EncryptionTier.PRIVATE,
            requestor="user",
            purpose="stats",
        )

        stats = memory_vault.get_statistics()
        assert "vault_id" in stats
        assert "storage" in stats
        assert "consents" in stats

    def test_list_blobs(self, memory_vault):
        """Test listing blobs."""
        memory_vault.store(
            b"Blob 1",
            EncryptionTier.PRIVATE,
            requestor="user",
            purpose="list_test",
            tags=["test"],
        )
        memory_vault.store(
            b"Blob 2",
            EncryptionTier.PRIVATE,
            requestor="user",
            purpose="list_test",
            tags=["test"],
        )

        blobs = memory_vault.list_blobs(tags=["test"])
        assert len(blobs) >= 2

    def test_create_vault_convenience_function(self, temp_dir):
        """Test create_vault convenience function."""
        vault = create_vault(temp_dir / "convenience_vault", owner="test")
        assert vault.is_initialized is True
        vault.shutdown()


# ============================================================================
# Security Tests
# ============================================================================

class TestVaultSecurity:
    """Security-focused tests."""

    def test_no_plaintext_storage(self, blob_storage, temp_dir):
        """Verify no plaintext is stored."""
        secret = b"This is a secret that should not appear in storage!"
        metadata = blob_storage.store(secret, EncryptionTier.PRIVATE)

        # Check blob file doesn't contain plaintext
        blob_path = blob_storage._get_blob_path(metadata.blob_id)
        blob_content = blob_path.read_bytes()
        assert secret not in blob_content

    def test_key_not_in_metadata(self, blob_storage):
        """Verify encryption key is not stored with metadata."""
        data = b"Test data"
        metadata = blob_storage.store(data, EncryptionTier.PRIVATE)

        # Key should not be in any metadata fields
        meta_dict = metadata.to_dict()
        meta_str = str(meta_dict)

        key = blob_storage._key_manager.get_key(metadata.key_id)
        assert key.hex() not in meta_str

    def test_secure_deletion_overwrites(self, blob_storage, temp_dir):
        """Verify secure deletion overwrites data."""
        data = b"Sensitive data to delete"
        metadata = blob_storage.store(data, EncryptionTier.PRIVATE)
        blob_path = blob_storage._get_blob_path(metadata.blob_id)

        # Store original size
        original_size = blob_path.stat().st_size

        # Secure delete
        blob_storage.delete(metadata.blob_id, secure=True)

        # File should be gone
        assert not blob_path.exists()

    def test_tier_enforcement(self, profile_manager):
        """Test that tier constraints are enforced."""
        # Cannot create VAULTED profile with software binding
        with pytest.raises(ValueError):
            EncryptionProfile(
                tier=EncryptionTier.VAULTED,
                name="Invalid",
                description="Should fail",
                key_derivation=KeyDerivation.ARGON2ID,
                key_binding=KeyBinding.SOFTWARE,
            )

    def test_content_logging_prohibited(self):
        """Verify content logging is prohibited."""
        with pytest.raises(ValueError):
            EncryptionProfile(
                tier=EncryptionTier.PRIVATE,
                name="BadProfile",
                description="Should fail",
                key_derivation=KeyDerivation.ARGON2ID,
                key_binding=KeyBinding.SOFTWARE,
                log_content=True,
            )
