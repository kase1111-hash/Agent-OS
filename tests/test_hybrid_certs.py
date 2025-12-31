"""
Tests for Hybrid Post-Quantum Certificates

Tests the HybridCertificate, HybridIdentity, and HybridIdentityManager
classes for quantum-resistant certificate operations.
"""

import pytest
import tempfile
import json
from pathlib import Path
from datetime import datetime, timedelta


def _check_liboqs_available() -> bool:
    """Check if liboqs is available for post-quantum cryptography."""
    try:
        import oqs
        return "Kyber768" in oqs.get_enabled_kem_mechanisms()
    except ImportError:
        return False


# Skip all tests if liboqs is not available
pytestmark = pytest.mark.skipif(
    not _check_liboqs_available(),
    reason="liboqs library required for post-quantum cryptography tests"
)


from src.federation.pq.hybrid_certs import (
    HybridCertificateVersion,
    CertificateType,
    HybridIdentityStatus,
    HybridIdentityKey,
    HybridCertificateSignature,
    HybridCertificate,
    HybridIdentity,
    HybridIdentityManager,
    create_hybrid_identity_manager,
    create_hybrid_identity,
)

from src.federation.pq.hybrid import HybridSigner
from src.federation.pq.ml_dsa import MLDSASecurityLevel


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def hybrid_signer():
    """Create a hybrid signer for testing."""
    return HybridSigner()


@pytest.fixture
def hybrid_identity_manager():
    """Create a hybrid identity manager for testing."""
    return HybridIdentityManager(node_id="test-node-001")


@pytest.fixture
def hybrid_identity_manager_with_storage():
    """Create a hybrid identity manager with persistent storage."""
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = HybridIdentityManager(
            node_id="test-node-002",
            storage_path=Path(tmpdir),
        )
        yield manager


# =============================================================================
# HybridIdentityKey Tests
# =============================================================================


class TestHybridIdentityKey:
    """Tests for HybridIdentityKey."""

    def test_create_identity_key(self, hybrid_signer):
        """Test creating an identity key."""
        keypair = hybrid_signer.generate_keypair()

        identity_key = HybridIdentityKey(
            classical_key=keypair.public_key.classical_key,
            pq_key=keypair.public_key.pq_key,
            algorithm=keypair.algorithm,
        )

        assert identity_key.classical_key == keypair.public_key.classical_key
        assert identity_key.pq_key == keypair.public_key.pq_key
        assert identity_key.key_id != ""
        assert identity_key.size > 32  # Larger than classical only

    def test_identity_key_serialization(self, hybrid_signer):
        """Test identity key serialization/deserialization."""
        keypair = hybrid_signer.generate_keypair()

        identity_key = HybridIdentityKey(
            classical_key=keypair.public_key.classical_key,
            pq_key=keypair.public_key.pq_key,
            algorithm=keypair.algorithm,
            security_level="level_3",
        )

        # Serialize
        data = identity_key.to_dict()
        assert "classical_key" in data
        assert "pq_key" in data
        assert data["algorithm"] == keypair.algorithm
        assert data["security_level"] == "level_3"

        # Deserialize
        restored = HybridIdentityKey.from_dict(data)
        assert restored.classical_key == identity_key.classical_key
        assert restored.pq_key == identity_key.pq_key
        assert restored.key_id == identity_key.key_id

    def test_identity_key_fingerprint(self, hybrid_signer):
        """Test identity key fingerprint generation."""
        keypair = hybrid_signer.generate_keypair()

        identity_key = HybridIdentityKey(
            classical_key=keypair.public_key.classical_key,
            pq_key=keypair.public_key.pq_key,
            algorithm=keypair.algorithm,
        )

        # Fingerprint should be consistent
        fp1 = identity_key.fingerprint
        fp2 = identity_key.fingerprint
        assert fp1 == fp2
        assert len(fp1) == 64  # SHA-256 hex

    def test_to_hybrid_public_key(self, hybrid_signer):
        """Test conversion to HybridPublicKey."""
        keypair = hybrid_signer.generate_keypair()

        identity_key = HybridIdentityKey(
            classical_key=keypair.public_key.classical_key,
            pq_key=keypair.public_key.pq_key,
            algorithm=keypair.algorithm,
        )

        hybrid_public = identity_key.to_hybrid_public_key()
        assert hybrid_public.classical_key == identity_key.classical_key
        assert hybrid_public.pq_key == identity_key.pq_key


# =============================================================================
# HybridCertificate Tests
# =============================================================================


class TestHybridCertificate:
    """Tests for HybridCertificate."""

    def test_create_certificate(self, hybrid_signer):
        """Test creating a certificate."""
        keypair = hybrid_signer.generate_keypair()
        identity_key = HybridIdentityKey(
            classical_key=keypair.public_key.classical_key,
            pq_key=keypair.public_key.pq_key,
            algorithm=keypair.algorithm,
        )

        now = datetime.utcnow()
        cert = HybridCertificate(
            subject_id="node-001",
            issuer_id="node-001",
            public_key=identity_key,
            valid_from=now,
            valid_until=now + timedelta(days=365),
        )

        assert cert.subject_id == "node-001"
        assert cert.issuer_id == "node-001"
        assert cert.is_self_signed
        assert cert.is_valid
        assert cert.is_hybrid
        assert cert.serial_number != ""

    def test_certificate_validity(self, hybrid_signer):
        """Test certificate validity checking."""
        keypair = hybrid_signer.generate_keypair()
        identity_key = HybridIdentityKey(
            classical_key=keypair.public_key.classical_key,
            pq_key=keypair.public_key.pq_key,
            algorithm=keypair.algorithm,
        )

        now = datetime.utcnow()

        # Valid certificate
        valid_cert = HybridCertificate(
            subject_id="node-001",
            issuer_id="node-001",
            public_key=identity_key,
            valid_from=now - timedelta(days=1),
            valid_until=now + timedelta(days=365),
        )
        assert valid_cert.is_valid

        # Expired certificate
        expired_cert = HybridCertificate(
            subject_id="node-001",
            issuer_id="node-001",
            public_key=identity_key,
            valid_from=now - timedelta(days=366),
            valid_until=now - timedelta(days=1),
        )
        assert not expired_cert.is_valid

        # Not yet valid certificate
        future_cert = HybridCertificate(
            subject_id="node-001",
            issuer_id="node-001",
            public_key=identity_key,
            valid_from=now + timedelta(days=1),
            valid_until=now + timedelta(days=365),
        )
        assert not future_cert.is_valid

    def test_certificate_serialization(self, hybrid_signer):
        """Test certificate serialization/deserialization."""
        keypair = hybrid_signer.generate_keypair()
        identity_key = HybridIdentityKey(
            classical_key=keypair.public_key.classical_key,
            pq_key=keypair.public_key.pq_key,
            algorithm=keypair.algorithm,
        )

        cert = HybridCertificate(
            subject_id="node-001",
            issuer_id="ca-001",
            public_key=identity_key,
            cert_type=CertificateType.NODE,
            extensions={"custom_field": "value"},
        )

        # Serialize
        data = cert.to_dict()
        assert data["subject_id"] == "node-001"
        assert data["issuer_id"] == "ca-001"
        assert data["cert_type"] == "node"

        # Deserialize
        restored = HybridCertificate.from_dict(data)
        assert restored.subject_id == cert.subject_id
        assert restored.issuer_id == cert.issuer_id
        assert restored.serial_number == cert.serial_number
        assert restored.extensions["custom_field"] == "value"

    def test_certificate_pem_format(self, hybrid_signer):
        """Test PEM-like export/import."""
        keypair = hybrid_signer.generate_keypair()
        identity_key = HybridIdentityKey(
            classical_key=keypair.public_key.classical_key,
            pq_key=keypair.public_key.pq_key,
            algorithm=keypair.algorithm,
        )

        cert = HybridCertificate(
            subject_id="node-001",
            issuer_id="node-001",
            public_key=identity_key,
        )

        # Export to PEM
        pem = cert.to_pem()
        assert "-----BEGIN HYBRID CERTIFICATE-----" in pem
        assert "-----END HYBRID CERTIFICATE-----" in pem

        # Import from PEM
        restored = HybridCertificate.from_pem(pem)
        assert restored.subject_id == cert.subject_id
        assert restored.serial_number == cert.serial_number

    def test_certificate_fingerprint(self, hybrid_signer):
        """Test certificate fingerprint."""
        keypair = hybrid_signer.generate_keypair()
        identity_key = HybridIdentityKey(
            classical_key=keypair.public_key.classical_key,
            pq_key=keypair.public_key.pq_key,
            algorithm=keypair.algorithm,
        )

        cert = HybridCertificate(
            subject_id="node-001",
            issuer_id="node-001",
            public_key=identity_key,
        )

        fp = cert.fingerprint
        assert len(fp) == 64  # SHA-256 hex


# =============================================================================
# HybridIdentity Tests
# =============================================================================


class TestHybridIdentity:
    """Tests for HybridIdentity."""

    def test_create_identity(self, hybrid_signer):
        """Test creating an identity."""
        keypair = hybrid_signer.generate_keypair()
        identity_key = HybridIdentityKey(
            classical_key=keypair.public_key.classical_key,
            pq_key=keypair.public_key.pq_key,
            algorithm=keypair.algorithm,
        )

        identity = HybridIdentity(
            node_id="node-001",
            display_name="Test Node",
            public_key=identity_key,
        )

        assert identity.node_id == "node-001"
        assert identity.display_name == "Test Node"
        assert identity.is_hybrid
        assert not identity.is_verified
        assert identity.fingerprint != ""

    def test_identity_serialization(self, hybrid_signer):
        """Test identity serialization/deserialization."""
        keypair = hybrid_signer.generate_keypair()
        identity_key = HybridIdentityKey(
            classical_key=keypair.public_key.classical_key,
            pq_key=keypair.public_key.pq_key,
            algorithm=keypair.algorithm,
        )

        identity = HybridIdentity(
            node_id="node-001",
            display_name="Test Node",
            public_key=identity_key,
            endpoints=["https://node1.example.com"],
            capabilities=["signing", "encryption"],
        )

        # Serialize
        data = identity.to_dict()
        assert data["node_id"] == "node-001"
        assert "signing" in data["capabilities"]

        # Deserialize
        restored = HybridIdentity.from_dict(data)
        assert restored.node_id == identity.node_id
        assert restored.display_name == identity.display_name
        assert restored.endpoints == identity.endpoints

    def test_factory_function(self):
        """Test create_hybrid_identity factory."""
        identity = create_hybrid_identity(
            node_id="factory-node",
            display_name="Factory Test",
        )

        assert identity.node_id == "factory-node"
        assert identity.display_name == "Factory Test"
        assert identity.is_hybrid
        assert identity.public_key is not None


# =============================================================================
# HybridIdentityManager Tests
# =============================================================================


class TestHybridIdentityManager:
    """Tests for HybridIdentityManager."""

    def test_manager_initialization(self, hybrid_identity_manager):
        """Test manager initialization."""
        manager = hybrid_identity_manager

        assert manager.identity is not None
        assert manager.identity.node_id == "test-node-001"
        assert manager.certificate is not None
        assert manager.certificate.is_self_signed
        assert manager.public_key is not None

    def test_self_signed_certificate(self, hybrid_identity_manager):
        """Test self-signed certificate generation."""
        manager = hybrid_identity_manager
        cert = manager.certificate

        assert cert.subject_id == manager.node_id
        assert cert.issuer_id == manager.node_id
        assert cert.is_self_signed
        assert cert.is_valid
        assert cert.signature is not None

        # Verify the certificate
        valid, reason = manager.verify_certificate(cert)
        assert valid, reason

    def test_sign_and_verify(self, hybrid_identity_manager):
        """Test signing and verification."""
        manager = hybrid_identity_manager

        message = b"Test message for signing"
        signature = manager.sign(message)

        assert signature is not None
        assert signature.classical_signature is not None
        assert signature.pq_signature is not None

        # Verify signature
        valid = manager.verify_signature(
            message,
            signature,
            manager.public_key,
        )
        assert valid

        # Verify wrong message fails
        valid = manager.verify_signature(
            b"Wrong message",
            signature,
            manager.public_key,
        )
        assert not valid

    def test_issue_certificate(self, hybrid_identity_manager):
        """Test issuing certificate for another identity."""
        manager = hybrid_identity_manager

        # Create peer identity
        peer_identity = create_hybrid_identity(
            node_id="peer-node",
            display_name="Peer Node",
        )

        # Issue certificate
        peer_cert = manager.issue_certificate(
            peer_identity,
            valid_days=30,
            cert_type=CertificateType.NODE,
        )

        assert peer_cert.subject_id == "peer-node"
        assert peer_cert.issuer_id == manager.node_id
        assert not peer_cert.is_self_signed
        assert peer_cert.signature is not None

        # Verify certificate
        valid, reason = manager.verify_certificate(peer_cert, manager.public_key)
        assert valid, reason

    def test_register_and_verify_identity(self, hybrid_identity_manager):
        """Test registering and verifying identities."""
        manager = hybrid_identity_manager

        # Create and register peer
        peer_identity = create_hybrid_identity(node_id="peer-001")
        peer_cert = manager.issue_certificate(peer_identity)
        peer_identity.certificate = peer_cert

        success = manager.register_identity(peer_identity)
        assert success

        # Retrieve
        retrieved = manager.get_identity("peer-001")
        assert retrieved is not None
        assert retrieved.node_id == "peer-001"

        # Verify
        valid = manager.verify_identity(peer_identity)
        assert valid
        assert peer_identity.status == HybridIdentityStatus.VERIFIED

    def test_trust_and_revoke(self, hybrid_identity_manager):
        """Test trust and revocation."""
        manager = hybrid_identity_manager

        # Create and register peer
        peer_identity = create_hybrid_identity(node_id="trusted-peer")
        peer_cert = manager.issue_certificate(peer_identity)
        peer_identity.certificate = peer_cert
        manager.register_identity(peer_identity)

        # Trust
        result = manager.trust_identity("trusted-peer")
        assert result
        assert manager.is_trusted("trusted-peer")
        assert peer_identity.status == HybridIdentityStatus.TRUSTED

        # Revoke
        result = manager.revoke_trust("trusted-peer")
        assert result
        assert not manager.is_trusted("trusted-peer")
        assert peer_identity.status == HybridIdentityStatus.REVOKED

    def test_certificate_revocation(self, hybrid_identity_manager):
        """Test certificate revocation."""
        manager = hybrid_identity_manager

        # Create and register peer
        peer_identity = create_hybrid_identity(node_id="revoked-peer")
        peer_cert = manager.issue_certificate(peer_identity)
        peer_identity.certificate = peer_cert
        manager.register_identity(peer_identity)

        # Certificate should be valid initially
        valid, _ = manager.verify_certificate(peer_cert, manager.public_key)
        assert valid

        # Revoke trust (which adds to revocation list)
        manager.revoke_trust("revoked-peer")

        # Certificate should now fail verification
        valid, reason = manager.verify_certificate(peer_cert, manager.public_key)
        assert not valid
        assert "revoked" in reason.lower()

    def test_list_identities(self, hybrid_identity_manager):
        """Test listing identities."""
        manager = hybrid_identity_manager

        # Create and register multiple peers
        for i in range(3):
            peer = create_hybrid_identity(node_id=f"peer-{i}")
            cert = manager.issue_certificate(peer)
            peer.certificate = cert
            manager.register_identity(peer)

        # List all
        all_identities = manager.list_identities()
        assert len(all_identities) == 3

        # Verify one and trust one
        manager.verify_identity(manager.get_identity("peer-0"))
        manager.trust_identity("peer-1")

        # Filter by status
        verified = manager.list_identities(status=HybridIdentityStatus.VERIFIED)
        assert len(verified) == 1

        trusted = manager.list_identities(status=HybridIdentityStatus.TRUSTED)
        assert len(trusted) == 1

    def test_attestation(self, hybrid_identity_manager):
        """Test creating and verifying attestations."""
        manager = hybrid_identity_manager

        # Register a peer so we can verify attestations from them
        peer = create_hybrid_identity(node_id="attesting-peer")
        manager.register_identity(peer)

        # Create attestation
        data = {"claim": "test_value", "timestamp": "2024-01-01T00:00:00Z"}
        attestation = manager.create_attestation(data)

        assert attestation["node_id"] == manager.node_id
        assert "signature" in attestation
        assert attestation["data"]["claim"] == "test_value"

    def test_certificate_chain(self, hybrid_identity_manager):
        """Test certificate chain operations."""
        # Create root CA
        root_ca = HybridIdentityManager(node_id="root-ca")

        # Create intermediate CA
        intermediate_identity = create_hybrid_identity(node_id="intermediate-ca")
        intermediate_cert = root_ca.issue_certificate(
            intermediate_identity,
            cert_type=CertificateType.INTERMEDIATE_CA,
        )
        intermediate_identity.certificate = intermediate_cert

        # Create leaf node
        leaf_identity = create_hybrid_identity(node_id="leaf-node")

        # Register intermediate with root
        root_ca.register_identity(intermediate_identity)

        # Get chain
        chain = root_ca.get_certificate_chain(intermediate_cert)
        assert len(chain) >= 1
        assert chain[0] == intermediate_cert


class TestHybridIdentityManagerPersistence:
    """Tests for identity manager persistence."""

    def test_save_and_load(self, hybrid_identity_manager_with_storage):
        """Test saving and loading identity."""
        manager = hybrid_identity_manager_with_storage

        # Save initial state
        original_key_id = manager.public_key.key_id
        original_cert_serial = manager.certificate.serial_number
        manager.save_identity()

        # Create new manager with same storage
        manager2 = HybridIdentityManager(
            node_id="test-node-002",
            storage_path=manager.storage_path,
        )

        # Should load existing identity
        assert manager2.public_key.key_id == original_key_id
        assert manager2.certificate.serial_number == original_cert_serial


class TestSecurityLevels:
    """Tests for different security levels."""

    @pytest.mark.parametrize("level", [
        MLDSASecurityLevel.ML_DSA_44,
        MLDSASecurityLevel.ML_DSA_65,
        MLDSASecurityLevel.ML_DSA_87,
    ])
    def test_different_security_levels(self, level):
        """Test identity manager with different security levels."""
        manager = HybridIdentityManager(
            node_id=f"test-{level.value}",
            ml_dsa_level=level,
        )

        assert manager.identity is not None
        assert manager.certificate is not None

        # Verify certificate
        valid, reason = manager.verify_certificate(manager.certificate)
        assert valid, reason

        # Test signing
        message = b"Test message"
        signature = manager.sign(message)
        assert manager.verify_signature(message, signature, manager.public_key)


class TestFactoryFunctions:
    """Tests for factory functions."""

    def test_create_hybrid_identity_manager(self):
        """Test factory function for identity manager."""
        manager = create_hybrid_identity_manager(
            node_id="factory-manager",
            ml_dsa_level=MLDSASecurityLevel.ML_DSA_65,
        )

        assert manager is not None
        assert manager.node_id == "factory-manager"
        assert manager.certificate is not None

    def test_create_hybrid_identity(self):
        """Test factory function for identity."""
        identity = create_hybrid_identity(
            node_id="factory-identity",
            display_name="Factory Created",
        )

        assert identity is not None
        assert identity.node_id == "factory-identity"
        assert identity.display_name == "Factory Created"
        assert identity.is_hybrid


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
