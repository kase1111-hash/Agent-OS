"""
Tests for Phase 4 Production Hardening Components

Tests the HSM, audit logging, backup/recovery, and configuration
modules for post-quantum cryptography production deployment.
"""

import pytest
import tempfile
import json
from pathlib import Path
from datetime import datetime, timedelta

from src.federation.pq.hsm import (
    HSMType,
    HSMSecurityLevel,
    PQAlgorithm,
    KeyState,
    HSMKeyHandle,
    HSMConfig,
    SoftwareHSMProvider,
    create_hsm_provider,
    get_recommended_hsm_config,
)

from src.federation.pq.audit import (
    AuditEventType,
    AuditSeverity,
    ComplianceStandard,
    CryptoAuditEvent,
    AuditConfig,
    AuditMetrics,
    CryptoAuditLogger,
    create_production_audit_config,
)

from src.federation.pq.backup import (
    BackupFormat,
    RecoveryMethod,
    BackupStatus,
    KeyShare,
    KeyBackup,
    BackupConfig,
    KeyBackupManager,
    ShamirSecretSharing,
    create_backup_manager,
)

from src.federation.pq.config import (
    Environment,
    AlgorithmConfig,
    KeyPolicyConfig,
    SecurityConfig,
    PQConfig,
    get_pq_config,
    set_pq_config,
    reset_pq_config,
)


# =============================================================================
# HSM Tests
# =============================================================================


class TestSoftwareHSM:
    """Tests for Software HSM provider."""

    @pytest.fixture
    def hsm_provider(self):
        """Create software HSM for testing."""
        config = HSMConfig(hsm_type=HSMType.SOFTWARE)
        provider = SoftwareHSMProvider(config)
        provider.initialize()
        yield provider
        provider.shutdown()

    def test_initialization(self, hsm_provider):
        """Test HSM initialization."""
        assert hsm_provider.is_available()
        assert hsm_provider.security_level == HSMSecurityLevel.LEVEL_1

    def test_generate_kem_keypair(self, hsm_provider):
        """Test KEM key generation."""
        handle = hsm_provider.generate_pq_keypair(
            algorithm=PQAlgorithm.ML_KEM_768,
            label="test-kem-key",
        )

        assert handle is not None
        assert handle.algorithm == PQAlgorithm.ML_KEM_768
        assert handle.label == "test-kem-key"
        assert handle.state == KeyState.ACTIVE
        assert handle.is_valid

    def test_generate_sig_keypair(self, hsm_provider):
        """Test signature key generation."""
        handle = hsm_provider.generate_pq_keypair(
            algorithm=PQAlgorithm.ML_DSA_65,
            label="test-sig-key",
        )

        assert handle is not None
        assert handle.algorithm == PQAlgorithm.ML_DSA_65
        assert handle.is_valid

    def test_get_public_key(self, hsm_provider):
        """Test public key retrieval."""
        handle = hsm_provider.generate_pq_keypair(
            algorithm=PQAlgorithm.ML_KEM_768,
            label="test-key",
        )

        public_key = hsm_provider.get_public_key(handle)
        assert public_key is not None
        assert len(public_key) > 0

    def test_key_expiration(self, hsm_provider):
        """Test key expiration."""
        handle = hsm_provider.generate_pq_keypair(
            algorithm=PQAlgorithm.ML_KEM_768,
            label="expiring-key",
            expires_in=timedelta(seconds=-1),  # Already expired
        )

        assert not handle.is_valid

    def test_destroy_key(self, hsm_provider):
        """Test key destruction."""
        handle = hsm_provider.generate_pq_keypair(
            algorithm=PQAlgorithm.ML_KEM_768,
            label="to-destroy",
        )

        result = hsm_provider.destroy_key(handle)
        assert result

        # Key should no longer be accessible
        with pytest.raises(KeyError):
            hsm_provider.get_public_key(handle)

    def test_list_keys(self, hsm_provider):
        """Test key listing."""
        # Generate multiple keys
        hsm_provider.generate_pq_keypair(PQAlgorithm.ML_KEM_768, "key1")
        hsm_provider.generate_pq_keypair(PQAlgorithm.ML_DSA_65, "key2")

        all_keys = hsm_provider.list_keys()
        assert len(all_keys) == 2

        kem_keys = hsm_provider.list_keys(algorithm=PQAlgorithm.ML_KEM_768)
        assert len(kem_keys) == 1


class TestHSMFactory:
    """Tests for HSM factory functions."""

    def test_create_software_hsm(self):
        """Test creating software HSM."""
        provider = create_hsm_provider(HSMType.SOFTWARE)
        assert provider is not None
        provider.initialize()
        assert provider.is_available()
        provider.shutdown()

    def test_recommended_config(self):
        """Test recommended HSM configuration."""
        dev_config = get_recommended_hsm_config("development")
        assert dev_config.hsm_type == HSMType.SOFTWARE
        assert not dev_config.audit_all_operations

        prod_config = get_recommended_hsm_config("production")
        assert prod_config.hsm_type == HSMType.PKCS11
        assert prod_config.audit_all_operations


# =============================================================================
# Audit Tests
# =============================================================================


class TestCryptoAuditLogger:
    """Tests for crypto audit logging."""

    @pytest.fixture
    def audit_logger(self):
        """Create audit logger for testing."""
        config = AuditConfig(enable_console_output=False)
        logger = CryptoAuditLogger(config)
        logger.start()
        yield logger
        logger.stop()

    def test_initialization(self, audit_logger):
        """Test audit logger initialization."""
        assert audit_logger is not None

    def test_log_event(self, audit_logger):
        """Test logging events."""
        event = audit_logger.log_event(
            event_type=AuditEventType.KEY_GENERATED,
            key_id="test-key-001",
            algorithm="ml-kem-768",
            success=True,
        )

        assert event is not None
        assert event.event_type == AuditEventType.KEY_GENERATED
        assert event.key_id == "test-key-001"
        assert event.success

    def test_log_key_generation(self, audit_logger):
        """Test key generation logging."""
        event = audit_logger.log_key_generation(
            key_id="gen-key-001",
            algorithm="ml-dsa-65",
            success=True,
        )

        assert event.event_type == AuditEventType.KEY_GENERATED

    def test_log_security_alert(self, audit_logger):
        """Test security alert logging."""
        event = audit_logger.log_security_alert(
            alert_type="key_compromise",
            description="Potential key compromise detected",
            key_id="compromised-key",
        )

        assert event.event_type == AuditEventType.SECURITY_ALERT
        assert event.severity == AuditSeverity.WARNING
        assert not event.success

    def test_metrics(self, audit_logger):
        """Test audit metrics."""
        # Log several events
        audit_logger.log_key_generation("key1", "ml-kem-768", True)
        audit_logger.log_key_generation("key2", "ml-dsa-65", True)
        audit_logger.log_sign_operation("key2", "abc123", True)
        audit_logger.log_sign_operation("key2", "def456", False)

        metrics = audit_logger.get_metrics()
        assert metrics.total_events >= 4
        assert metrics.failed_operations >= 1

    def test_chain_integrity(self, audit_logger):
        """Test audit chain integrity."""
        events = []
        for i in range(5):
            event = audit_logger.log_event(
                event_type=AuditEventType.KEY_ACCESSED,
                key_id=f"key-{i}",
            )
            events.append(event)

        # Verify sequence numbers
        for i, event in enumerate(events):
            if i > 0:
                assert event.sequence_number > events[i - 1].sequence_number


class TestAuditWithStorage:
    """Tests for audit logging with file storage."""

    def test_audit_with_file_output(self):
        """Test audit logging to files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = AuditConfig(
                log_directory=Path(tmpdir),
                output_format="jsonl",
                compress_logs=False,
            )
            logger = CryptoAuditLogger(config)
            logger.start()

            # Log events
            logger.log_key_generation("key1", "ml-kem-768", True)
            logger.log_key_destroyed("key1", reason="test")

            logger.stop()

            # Check files were created
            log_files = list(Path(tmpdir).glob("crypto_audit_*.jsonl"))
            assert len(log_files) >= 1


# =============================================================================
# Backup Tests
# =============================================================================


class TestShamirSecretSharing:
    """Tests for Shamir's Secret Sharing."""

    def test_split_and_recover(self):
        """Test basic split and recover."""
        secret = b"This is a secret key material!"

        shares = ShamirSecretSharing.split(secret, total_shares=5, threshold=3)
        assert len(shares) == 5

        # Recover with minimum shares
        recovered = ShamirSecretSharing.recover(shares[:3])
        assert recovered == secret

    def test_recover_with_any_shares(self):
        """Test recovery with different share combinations."""
        secret = b"Another secret!"

        shares = ShamirSecretSharing.split(secret, total_shares=5, threshold=3)

        # Different combinations
        assert ShamirSecretSharing.recover([shares[0], shares[2], shares[4]]) == secret
        assert ShamirSecretSharing.recover([shares[1], shares[3], shares[4]]) == secret
        assert ShamirSecretSharing.recover(shares) == secret  # All shares

    def test_insufficient_shares(self):
        """Test that recovery fails with insufficient shares."""
        secret = b"Secret!"
        shares = ShamirSecretSharing.split(secret, total_shares=5, threshold=3)

        # This should fail or return wrong result with only 2 shares
        with pytest.raises(ValueError):
            ShamirSecretSharing.recover(shares[:1])


class TestKeyBackupManager:
    """Tests for key backup manager."""

    @pytest.fixture
    def backup_manager(self):
        """Create backup manager for testing."""
        return KeyBackupManager()

    def test_export_import_key(self, backup_manager):
        """Test key export and import."""
        private_key = b"mock_private_key_material_here"
        password = "secure-password-123"

        backup = backup_manager.export_key(
            key_id="test-key-001",
            private_key=private_key,
            algorithm="ml-kem-768",
            password=password,
        )

        assert backup is not None
        assert backup.key_id == "test-key-001"
        assert backup.format == BackupFormat.ENCRYPTED_JSON

        # Import
        recovered = backup_manager.import_key(backup, password)
        assert recovered == private_key

    def test_wrong_password(self, backup_manager):
        """Test import with wrong password."""
        private_key = b"secret_key_data"
        backup = backup_manager.export_key(
            key_id="key-002",
            private_key=private_key,
            algorithm="ml-dsa-65",
            password="correct-password",
        )

        with pytest.raises(ValueError):
            backup_manager.import_key(backup, "wrong-password")

    def test_split_key(self, backup_manager):
        """Test key splitting."""
        private_key = b"key_to_split_into_shares"

        shares = backup_manager.split_key(
            key_id="split-key-001",
            private_key=private_key,
            algorithm="ml-kem-768",
            total_shares=5,
            threshold=3,
        )

        assert len(shares) == 5
        assert all(s.threshold == 3 for s in shares)
        assert all(s.key_id == "split-key-001" for s in shares)

    def test_recover_from_shares(self, backup_manager):
        """Test key recovery from shares."""
        private_key = b"recoverable_key_material"

        shares = backup_manager.split_key(
            key_id="recover-key-001",
            private_key=private_key,
            algorithm="ml-dsa-65",
            total_shares=5,
            threshold=3,
        )

        # Recover with 3 shares
        recovered, key_id = backup_manager.recover_from_shares(shares[:3])
        assert recovered == private_key
        assert key_id == "recover-key-001"

    def test_share_verification(self, backup_manager):
        """Test share verification."""
        private_key = b"verify_these_shares"

        shares = backup_manager.split_key(
            key_id="verify-key",
            private_key=private_key,
            algorithm="ml-kem-768",
            total_shares=5,
            threshold=3,
        )

        # All shares should be valid
        for share in shares:
            assert share.is_valid


class TestBackupWithStorage:
    """Tests for backup with persistent storage."""

    def test_backup_persistence(self):
        """Test backup persistence to disk."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = BackupConfig(backup_directory=Path(tmpdir))
            manager = KeyBackupManager(config)

            private_key = b"persistent_key"
            backup = manager.export_key(
                key_id="persist-key",
                private_key=private_key,
                algorithm="ml-kem-768",
                password="password",
            )

            # Load from disk
            loaded = manager.load_backup(backup.backup_id)
            assert loaded is not None
            assert loaded.key_id == backup.key_id


# =============================================================================
# Configuration Tests
# =============================================================================


class TestPQConfig:
    """Tests for PQ configuration."""

    def setup_method(self):
        """Reset config before each test."""
        reset_pq_config()

    def test_default_config(self):
        """Test default configuration."""
        config = PQConfig()
        assert config.environment == Environment.DEVELOPMENT
        assert config.algorithms is not None
        assert config.security is not None

    def test_production_config(self):
        """Test production configuration."""
        config = PQConfig.for_environment(Environment.PRODUCTION)

        assert config.environment == Environment.PRODUCTION
        assert config.algorithms.require_hybrid_mode
        assert config.security.require_dual_control
        assert config.audit.enable_audit_logging
        assert config.hsm.hsm_enabled

    def test_development_config(self):
        """Test development configuration."""
        config = PQConfig.for_environment(Environment.DEVELOPMENT)

        assert config.environment == Environment.DEVELOPMENT
        assert not config.security.require_dual_control
        assert not config.hsm.hsm_enabled

    def test_config_validation(self):
        """Test configuration validation."""
        config = PQConfig.for_environment(Environment.PRODUCTION)
        errors = config.validate()
        assert len(errors) == 0  # Valid production config

        # Create invalid config
        config.algorithms.require_hybrid_mode = False
        errors = config.validate()
        assert len(errors) > 0

    def test_config_to_dict(self):
        """Test configuration serialization."""
        config = PQConfig.for_environment(Environment.STAGING)
        data = config.to_dict()

        assert data["environment"] == "staging"
        assert "algorithms" in data
        assert "security" in data
        assert "hsm" in data

    def test_algorithm_allowed(self):
        """Test algorithm allowlist."""
        config = AlgorithmConfig()

        assert config.is_algorithm_allowed("ml-kem-768")
        assert config.is_algorithm_allowed("ed25519-ml-dsa-65")
        assert not config.is_algorithm_allowed("unknown-algo")


class TestGlobalConfig:
    """Tests for global configuration management."""

    def setup_method(self):
        """Reset config before each test."""
        reset_pq_config()

    def test_get_config(self):
        """Test getting global config."""
        config = get_pq_config("development")
        assert config is not None
        assert config.environment == Environment.DEVELOPMENT

    def test_set_config(self):
        """Test setting global config."""
        custom_config = PQConfig(environment=Environment.STAGING)
        set_pq_config(custom_config)

        retrieved = get_pq_config()
        assert retrieved.environment == Environment.STAGING


# =============================================================================
# Integration Tests
# =============================================================================


class TestProductionWorkflow:
    """Integration tests for production workflows."""

    def test_full_key_lifecycle(self):
        """Test complete key lifecycle with HSM and audit."""
        # Setup
        hsm_config = HSMConfig(hsm_type=HSMType.SOFTWARE)
        hsm = SoftwareHSMProvider(hsm_config)
        hsm.initialize()

        audit_config = AuditConfig(enable_console_output=False)
        audit = CryptoAuditLogger(audit_config)
        audit.start()

        backup_manager = KeyBackupManager()

        try:
            # Generate key in HSM
            handle = hsm.generate_pq_keypair(
                algorithm=PQAlgorithm.ML_KEM_768,
                label="production-key",
            )
            audit.log_key_generation(handle.handle_id, "ml-kem-768", True)

            # Get public key
            public_key = hsm.get_public_key(handle)
            assert len(public_key) > 0

            # Create backup (normally you'd export wrapped key from HSM)
            mock_private = b"hsm_wrapped_key_material"
            shares = backup_manager.split_key(
                key_id=handle.handle_id,
                private_key=mock_private,
                algorithm="ml-kem-768",
                total_shares=5,
                threshold=3,
            )
            assert len(shares) == 5

            # Simulate key compromise and destruction
            audit.log_security_alert(
                "potential_compromise",
                "Key may have been exposed",
                key_id=handle.handle_id,
            )
            hsm.destroy_key(handle)
            audit.log_key_destroyed(handle.handle_id, reason="potential_compromise")

            # Verify metrics
            metrics = audit.get_metrics()
            assert metrics.total_events >= 3

        finally:
            audit.stop()
            hsm.shutdown()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
