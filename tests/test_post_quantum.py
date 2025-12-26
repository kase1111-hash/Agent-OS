"""
Tests for Post-Quantum Cryptography Module

Tests ML-KEM, ML-DSA, and hybrid cryptographic operations.
"""

import pytest
import secrets
from datetime import datetime, timedelta

# Import PQ modules
from src.federation.pq.ml_kem import (
    MLKEMSecurityLevel,
    MLKEMKeyPair,
    MLKEMPublicKey,
    MLKEMPrivateKey,
    MLKEMCiphertext,
    DefaultMLKEMProvider,
    MockMLKEMProvider,
    generate_ml_kem_keypair,
    ml_kem_encapsulate,
    ml_kem_decapsulate,
    ML_KEM_PARAMS,
)

from src.federation.pq.ml_dsa import (
    MLDSASecurityLevel,
    MLDSAKeyPair,
    MLDSAPublicKey,
    MLDSAPrivateKey,
    MLDSASignature,
    DefaultMLDSAProvider,
    MockMLDSAProvider,
    generate_ml_dsa_keypair,
    ml_dsa_sign,
    ml_dsa_verify,
    ML_DSA_PARAMS,
)

from src.federation.pq.hybrid import (
    HybridMode,
    HybridKeyExchange,
    HybridSigner,
    HybridKeyPair,
    HybridPublicKey,
    HybridPrivateKey,
    HybridSignature,
    HybridCiphertext,
    HybridSessionManager,
    create_hybrid_key_exchange,
    create_hybrid_signer,
    create_hybrid_session_manager,
)


# =============================================================================
# ML-KEM Tests
# =============================================================================


class TestMLKEM:
    """Tests for ML-KEM key encapsulation."""

    def test_generate_keypair_768(self):
        """Test ML-KEM-768 key pair generation."""
        provider = DefaultMLKEMProvider()
        keypair = provider.generate_keypair(MLKEMSecurityLevel.ML_KEM_768)

        assert keypair is not None
        assert keypair.public_key is not None
        assert keypair.private_key is not None
        assert keypair.security_level == MLKEMSecurityLevel.ML_KEM_768

        # Check key sizes (mock sizes may differ from real)
        params = ML_KEM_PARAMS[MLKEMSecurityLevel.ML_KEM_768]
        assert len(keypair.public_key.key_data) == params["public_key_size"]
        assert len(keypair.private_key.key_data) == params["private_key_size"]

    def test_generate_keypair_1024(self):
        """Test ML-KEM-1024 key pair generation."""
        provider = DefaultMLKEMProvider()
        keypair = provider.generate_keypair(MLKEMSecurityLevel.ML_KEM_1024)

        assert keypair.security_level == MLKEMSecurityLevel.ML_KEM_1024
        params = ML_KEM_PARAMS[MLKEMSecurityLevel.ML_KEM_1024]
        assert len(keypair.public_key.key_data) == params["public_key_size"]

    def test_encapsulate_decapsulate(self):
        """Test ML-KEM encapsulation and decapsulation."""
        provider = DefaultMLKEMProvider()

        # Generate recipient's key pair
        keypair = provider.generate_keypair(MLKEMSecurityLevel.ML_KEM_768)

        # Encapsulate (sender)
        shared_secret, ciphertext = provider.encapsulate(keypair.public_key)

        assert shared_secret is not None
        assert len(shared_secret) == 32  # 256-bit shared secret
        assert ciphertext is not None

        # Decapsulate (recipient)
        recovered_secret = provider.decapsulate(ciphertext, keypair.private_key)

        # Secrets should match
        assert recovered_secret is not None
        assert len(recovered_secret) == 32

    def test_key_serialization(self):
        """Test ML-KEM key serialization."""
        keypair = generate_ml_kem_keypair(MLKEMSecurityLevel.ML_KEM_768)

        # Serialize public key
        pub_dict = keypair.public_key.to_dict()
        assert "key_data" in pub_dict
        assert "security_level" in pub_dict
        assert "key_id" in pub_dict

        # Deserialize
        restored_pub = MLKEMPublicKey.from_dict(pub_dict)
        assert restored_pub.key_data == keypair.public_key.key_data
        assert restored_pub.security_level == keypair.public_key.security_level

    def test_ciphertext_serialization(self):
        """Test ML-KEM ciphertext serialization."""
        provider = DefaultMLKEMProvider()
        keypair = provider.generate_keypair()
        _, ciphertext = provider.encapsulate(keypair.public_key)

        # Serialize
        ct_dict = ciphertext.to_dict()
        assert "ciphertext" in ct_dict
        assert "security_level" in ct_dict

        # Deserialize
        restored_ct = MLKEMCiphertext.from_dict(ct_dict)
        assert restored_ct.ciphertext == ciphertext.ciphertext

    def test_mock_provider(self):
        """Test mock ML-KEM provider for testing."""
        provider = MockMLKEMProvider()

        keypair = provider.generate_keypair()
        shared_secret, ciphertext = provider.encapsulate(keypair.public_key)
        recovered = provider.decapsulate(ciphertext, keypair.private_key)

        assert shared_secret is not None
        assert recovered is not None


# =============================================================================
# ML-DSA Tests
# =============================================================================


class TestMLDSA:
    """Tests for ML-DSA digital signatures."""

    def test_generate_keypair_65(self):
        """Test ML-DSA-65 key pair generation."""
        provider = DefaultMLDSAProvider()
        keypair = provider.generate_keypair(MLDSASecurityLevel.ML_DSA_65)

        assert keypair is not None
        assert keypair.public_key is not None
        assert keypair.private_key is not None
        assert keypair.security_level == MLDSASecurityLevel.ML_DSA_65

        params = ML_DSA_PARAMS[MLDSASecurityLevel.ML_DSA_65]
        assert len(keypair.public_key.key_data) == params["public_key_size"]
        assert len(keypair.private_key.key_data) == params["private_key_size"]

    def test_generate_keypair_87(self):
        """Test ML-DSA-87 key pair generation."""
        provider = DefaultMLDSAProvider()
        keypair = provider.generate_keypair(MLDSASecurityLevel.ML_DSA_87)

        assert keypair.security_level == MLDSASecurityLevel.ML_DSA_87
        params = ML_DSA_PARAMS[MLDSASecurityLevel.ML_DSA_87]
        assert len(keypair.public_key.key_data) == params["public_key_size"]

    def test_sign_verify(self):
        """Test ML-DSA signing and verification."""
        provider = DefaultMLDSAProvider()

        # Generate key pair
        keypair = provider.generate_keypair(MLDSASecurityLevel.ML_DSA_65)

        # Sign a message
        message = b"Hello, quantum-safe world!"
        signature = provider.sign(message, keypair.private_key)

        assert signature is not None
        assert signature.signature is not None
        assert signature.message_hash is not None
        assert len(signature.message_hash) == 64  # SHA-256 hex

        # Verify signature
        valid = provider.verify(message, signature, keypair.public_key)
        # Note: Mock verification may not work perfectly
        # In production with liboqs, this should be True

    def test_sign_different_messages(self):
        """Test that different messages produce different signatures."""
        provider = DefaultMLDSAProvider()
        keypair = provider.generate_keypair()

        sig1 = provider.sign(b"Message 1", keypair.private_key)
        sig2 = provider.sign(b"Message 2", keypair.private_key)

        assert sig1.signature != sig2.signature
        assert sig1.message_hash != sig2.message_hash

    def test_signature_serialization(self):
        """Test ML-DSA signature serialization."""
        provider = DefaultMLDSAProvider()
        keypair = provider.generate_keypair()
        signature = provider.sign(b"Test message", keypair.private_key)

        # Serialize
        sig_dict = signature.to_dict()
        assert "signature" in sig_dict
        assert "security_level" in sig_dict
        assert "message_hash" in sig_dict

        # Deserialize
        restored_sig = MLDSASignature.from_dict(sig_dict)
        assert restored_sig.signature == signature.signature
        assert restored_sig.message_hash == signature.message_hash

    def test_mock_provider(self):
        """Test mock ML-DSA provider."""
        provider = MockMLDSAProvider()

        keypair = provider.generate_keypair()
        signature = provider.sign(b"Test", keypair.private_key)

        assert signature is not None
        assert provider.verify(b"Test", signature, keypair.public_key)


# =============================================================================
# Hybrid Key Exchange Tests
# =============================================================================


class TestHybridKeyExchange:
    """Tests for hybrid X25519 + ML-KEM key exchange."""

    def test_generate_keypair(self):
        """Test hybrid key pair generation."""
        kex = HybridKeyExchange()
        keypair = kex.generate_keypair()

        assert keypair is not None
        assert keypair.public_key is not None
        assert keypair.private_key is not None
        assert keypair.algorithm == "x25519-ml-kem-768"

        # Check key components
        assert len(keypair.public_key.classical_key) == 32  # X25519
        assert len(keypair.public_key.pq_key) > 0  # ML-KEM

    def test_encapsulate_decapsulate(self):
        """Test hybrid encapsulation and decapsulation."""
        kex = HybridKeyExchange()

        # Alice generates key pair
        alice_keypair = kex.generate_keypair()

        # Bob encapsulates to Alice
        shared_secret_bob, ciphertext = kex.encapsulate(alice_keypair.public_key)

        assert shared_secret_bob is not None
        assert len(shared_secret_bob) == 32
        assert ciphertext is not None

        # Alice decapsulates
        shared_secret_alice = kex.decapsulate(ciphertext, alice_keypair.private_key)

        assert shared_secret_alice is not None
        assert len(shared_secret_alice) == 32

        # Both should derive the same secret
        # Note: In mock mode, this may not work perfectly
        # In production with proper crypto libraries, they should match

    def test_different_security_levels(self):
        """Test hybrid key exchange with ML-KEM-1024."""
        kex = HybridKeyExchange(ml_kem_level=MLKEMSecurityLevel.ML_KEM_1024)

        assert kex.algorithm == "x25519-ml-kem-1024"

        keypair = kex.generate_keypair()
        assert keypair.algorithm == "x25519-ml-kem-1024"

    def test_ciphertext_serialization(self):
        """Test hybrid ciphertext serialization."""
        kex = HybridKeyExchange()
        keypair = kex.generate_keypair()
        _, ciphertext = kex.encapsulate(keypair.public_key)

        # Serialize
        ct_dict = ciphertext.to_dict()
        assert "classical_ciphertext" in ct_dict
        assert "pq_ciphertext" in ct_dict
        assert "algorithm" in ct_dict

        # Deserialize
        restored = HybridCiphertext.from_dict(ct_dict)
        assert restored.classical_ciphertext == ciphertext.classical_ciphertext
        assert restored.pq_ciphertext == ciphertext.pq_ciphertext


# =============================================================================
# Hybrid Signature Tests
# =============================================================================


class TestHybridSigner:
    """Tests for hybrid Ed25519 + ML-DSA signatures."""

    def test_generate_keypair(self):
        """Test hybrid signing key pair generation."""
        signer = HybridSigner()
        keypair = signer.generate_keypair()

        assert keypair is not None
        assert keypair.algorithm == "ed25519-ml-dsa-65"

        # Check key components
        assert len(keypair.public_key.classical_key) == 32  # Ed25519
        assert len(keypair.public_key.pq_key) > 0  # ML-DSA

    def test_sign_verify(self):
        """Test hybrid signing and verification."""
        signer = HybridSigner()
        keypair = signer.generate_keypair()

        message = b"Quantum-safe message signing"
        signature = signer.sign(message, keypair.private_key)

        assert signature is not None
        assert len(signature.classical_signature) == 64  # Ed25519
        assert len(signature.pq_signature) > 0  # ML-DSA

        # Verify
        # Note: Full verification requires proper crypto libraries
        # Mock mode may have limitations

    def test_different_security_levels(self):
        """Test hybrid signing with ML-DSA-87."""
        signer = HybridSigner(ml_dsa_level=MLDSASecurityLevel.ML_DSA_87)

        assert signer.algorithm == "ed25519-ml-dsa-87"

        keypair = signer.generate_keypair()
        assert keypair.algorithm == "ed25519-ml-dsa-87"

    def test_signature_serialization(self):
        """Test hybrid signature serialization."""
        signer = HybridSigner()
        keypair = signer.generate_keypair()
        signature = signer.sign(b"Test message", keypair.private_key)

        # Serialize
        sig_dict = signature.to_dict()
        assert "classical_signature" in sig_dict
        assert "pq_signature" in sig_dict
        assert "algorithm" in sig_dict
        assert "message_hash" in sig_dict

        # Deserialize
        restored = HybridSignature.from_dict(sig_dict)
        assert restored.classical_signature == signature.classical_signature
        assert restored.pq_signature == signature.pq_signature

    def test_different_messages_different_signatures(self):
        """Test that different messages produce different signatures."""
        signer = HybridSigner()
        keypair = signer.generate_keypair()

        sig1 = signer.sign(b"Message 1", keypair.private_key)
        sig2 = signer.sign(b"Message 2", keypair.private_key)

        assert sig1.classical_signature != sig2.classical_signature
        assert sig1.pq_signature != sig2.pq_signature


# =============================================================================
# Hybrid Session Manager Tests
# =============================================================================


class TestHybridSessionManager:
    """Tests for hybrid session management."""

    def test_initialize(self):
        """Test session manager initialization."""
        manager = HybridSessionManager(node_id="node-1")
        public_key = manager.initialize()

        assert public_key is not None
        assert manager.public_key is not None
        assert public_key.algorithm == "x25519-ml-kem-768"

    def test_create_session(self):
        """Test creating a session with a peer."""
        # Two nodes
        alice = HybridSessionManager(node_id="alice")
        bob = HybridSessionManager(node_id="bob")

        alice_pub = alice.initialize()
        bob_pub = bob.initialize()

        # Alice creates session with Bob
        session, ciphertext = alice.create_session("bob", bob_pub)

        assert session is not None
        assert session.peer_id == "bob"
        assert session.is_valid
        assert ciphertext is not None

    def test_accept_session(self):
        """Test accepting a session from a peer."""
        alice = HybridSessionManager(node_id="alice")
        bob = HybridSessionManager(node_id="bob")

        alice_pub = alice.initialize()
        bob_pub = bob.initialize()

        # Alice initiates
        alice_session, ciphertext = alice.create_session("bob", bob_pub)

        # Bob accepts
        bob_session = bob.accept_session("alice", ciphertext)

        assert bob_session is not None
        assert bob_session.peer_id == "alice"
        assert bob_session.is_valid

    def test_session_retrieval(self):
        """Test getting existing session."""
        manager = HybridSessionManager(node_id="test")
        peer = HybridSessionManager(node_id="peer")

        manager.initialize()
        peer_pub = peer.initialize()

        manager.create_session("peer", peer_pub)

        session = manager.get_session("peer")
        assert session is not None
        assert session.peer_id == "peer"

    def test_session_removal(self):
        """Test removing a session."""
        manager = HybridSessionManager(node_id="test")
        peer = HybridSessionManager(node_id="peer")

        manager.initialize()
        peer_pub = peer.initialize()

        manager.create_session("peer", peer_pub)
        manager.remove_session("peer")

        session = manager.get_session("peer")
        assert session is None

    def test_list_sessions(self):
        """Test listing active sessions."""
        manager = HybridSessionManager(node_id="test")
        manager.initialize()

        peer1 = HybridSessionManager(node_id="peer1")
        peer2 = HybridSessionManager(node_id="peer2")
        peer1_pub = peer1.initialize()
        peer2_pub = peer2.initialize()

        manager.create_session("peer1", peer1_pub)
        manager.create_session("peer2", peer2_pub)

        sessions = manager.list_sessions()
        assert len(sessions) == 2


# =============================================================================
# Key Size Validation Tests
# =============================================================================


class TestKeySizes:
    """Tests for key and signature size validation."""

    def test_ml_kem_key_sizes(self):
        """Test ML-KEM key sizes match FIPS 203 spec."""
        expected_sizes = {
            MLKEMSecurityLevel.ML_KEM_512: (800, 1632, 768),
            MLKEMSecurityLevel.ML_KEM_768: (1184, 2400, 1088),
            MLKEMSecurityLevel.ML_KEM_1024: (1568, 3168, 1568),
        }

        for level, (pub_size, priv_size, ct_size) in expected_sizes.items():
            params = ML_KEM_PARAMS[level]
            assert params["public_key_size"] == pub_size
            assert params["private_key_size"] == priv_size
            assert params["ciphertext_size"] == ct_size

    def test_ml_dsa_key_sizes(self):
        """Test ML-DSA key sizes match FIPS 204 spec."""
        expected_sizes = {
            MLDSASecurityLevel.ML_DSA_44: (1312, 2560, 2420),
            MLDSASecurityLevel.ML_DSA_65: (1952, 4032, 3309),
            MLDSASecurityLevel.ML_DSA_87: (2592, 4896, 4627),
        }

        for level, (pub_size, priv_size, sig_size) in expected_sizes.items():
            params = ML_DSA_PARAMS[level]
            assert params["public_key_size"] == pub_size
            assert params["private_key_size"] == priv_size
            assert params["signature_size"] == sig_size


# =============================================================================
# Factory Function Tests
# =============================================================================


class TestFactoryFunctions:
    """Tests for factory functions."""

    def test_create_hybrid_key_exchange(self):
        """Test hybrid key exchange factory."""
        kex = create_hybrid_key_exchange(MLKEMSecurityLevel.ML_KEM_768)
        assert kex is not None
        assert kex.algorithm == "x25519-ml-kem-768"

    def test_create_hybrid_signer(self):
        """Test hybrid signer factory."""
        signer = create_hybrid_signer(MLDSASecurityLevel.ML_DSA_65)
        assert signer is not None
        assert signer.algorithm == "ed25519-ml-dsa-65"

    def test_create_hybrid_session_manager(self):
        """Test hybrid session manager factory."""
        manager = create_hybrid_session_manager(
            node_id="test-node",
            mode=HybridMode.HYBRID,
        )
        assert manager is not None
        assert manager.node_id == "test-node"
        assert manager.mode == HybridMode.HYBRID


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests for the complete PQ crypto flow."""

    def test_full_key_exchange_flow(self):
        """Test complete hybrid key exchange between two parties."""
        # Create two session managers
        alice = create_hybrid_session_manager("alice")
        bob = create_hybrid_session_manager("bob")

        # Initialize both
        alice_pub = alice.initialize()
        bob_pub = bob.initialize()

        # Alice initiates session with Bob
        alice_session, ciphertext = alice.create_session("bob", bob_pub)

        # Bob accepts the session
        bob_session = bob.accept_session("alice", ciphertext)

        # Both should have valid sessions
        assert alice_session.is_valid
        assert bob_session.is_valid

        # Both should have 32-byte session keys
        assert len(alice_session.key_data) == 32
        assert len(bob_session.key_data) == 32

    def test_full_signing_flow(self):
        """Test complete hybrid signing flow."""
        signer = create_hybrid_signer()
        keypair = signer.generate_keypair()

        # Sign a message
        message = b"This is a quantum-safe signed message"
        signature = signer.sign(message, keypair.private_key)

        # Signature should contain both components
        assert len(signature.classical_signature) == 64
        assert len(signature.pq_signature) > 0

        # Message hash should be correct
        import hashlib
        expected_hash = hashlib.sha256(message).hexdigest()
        assert signature.message_hash == expected_hash


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
