"""
Hybrid Post-Quantum Cryptography

Combines classical and post-quantum algorithms for defense-in-depth:
- Key Exchange: X25519 + ML-KEM (Kyber)
- Signatures: Ed25519 + ML-DSA (Dilithium)

The hybrid approach ensures security even if one algorithm is broken:
- If classical algorithms are broken by quantum computers, PQ algorithms protect
- If PQ algorithms have undiscovered weaknesses, classical algorithms protect

This follows NIST recommendations for transitional quantum-safe cryptography.

Usage:
    # Hybrid key exchange
    kex = HybridKeyExchange()
    alice_keypair = kex.generate_keypair()

    # Bob encapsulates to Alice
    shared_secret, ciphertext = kex.encapsulate(alice_keypair.public_key)

    # Alice decapsulates
    recovered_secret = kex.decapsulate(ciphertext, alice_keypair.private_key)

    # Hybrid signing
    signer = HybridSigner()
    keypair = signer.generate_keypair()
    signature = signer.sign(message, keypair.private_key)
    valid = signer.verify(message, signature, keypair.public_key)
"""

import base64
import hashlib
import hmac
import logging
import secrets
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from .ml_kem import (
    MLKEMSecurityLevel,
    MLKEMKeyPair,
    MLKEMPublicKey,
    MLKEMPrivateKey,
    MLKEMCiphertext,
    MLKEMProvider,
    DefaultMLKEMProvider,
)

from .ml_dsa import (
    MLDSASecurityLevel,
    MLDSAKeyPair,
    MLDSAPublicKey,
    MLDSAPrivateKey,
    MLDSASignature,
    MLDSAProvider,
    DefaultMLDSAProvider,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Constants
# =============================================================================


class HybridMode(str, Enum):
    """Hybrid cryptography modes."""

    CLASSICAL_ONLY = "classical"      # X25519/Ed25519 only (legacy)
    POST_QUANTUM_ONLY = "pq"          # ML-KEM/ML-DSA only
    HYBRID = "hybrid"                  # Both combined (recommended)


class HybridKEMAlgorithm(str, Enum):
    """Hybrid KEM algorithm identifiers."""

    X25519_ML_KEM_768 = "x25519-ml-kem-768"
    X25519_ML_KEM_1024 = "x25519-ml-kem-1024"


class HybridSigAlgorithm(str, Enum):
    """Hybrid signature algorithm identifiers."""

    ED25519_ML_DSA_65 = "ed25519-ml-dsa-65"
    ED25519_ML_DSA_87 = "ed25519-ml-dsa-87"


# =============================================================================
# Hybrid Key Models
# =============================================================================


@dataclass
class HybridPublicKey:
    """Combined classical and post-quantum public key."""

    classical_key: bytes          # X25519 or Ed25519 public key (32 bytes)
    pq_key: bytes                 # ML-KEM or ML-DSA public key
    algorithm: str                # Algorithm identifier
    key_id: str = ""
    created_at: datetime = field(default_factory=datetime.utcnow)

    def __post_init__(self):
        if not self.key_id:
            combined = self.classical_key + self.pq_key
            self.key_id = hashlib.sha256(combined).hexdigest()[:16]

    @property
    def size(self) -> int:
        """Total key size in bytes."""
        return len(self.classical_key) + len(self.pq_key)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "classical_key": base64.b64encode(self.classical_key).decode(),
            "pq_key": base64.b64encode(self.pq_key).decode(),
            "algorithm": self.algorithm,
            "key_id": self.key_id,
            "created_at": self.created_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HybridPublicKey":
        """Create from dictionary."""
        return cls(
            classical_key=base64.b64decode(data["classical_key"]),
            pq_key=base64.b64decode(data["pq_key"]),
            algorithm=data["algorithm"],
            key_id=data.get("key_id", ""),
            created_at=datetime.fromisoformat(
                data.get("created_at", datetime.utcnow().isoformat())
            ),
        )


@dataclass
class HybridPrivateKey:
    """Combined classical and post-quantum private key."""

    classical_key: bytes          # X25519 or Ed25519 private key
    pq_key: bytes                 # ML-KEM or ML-DSA private key
    algorithm: str
    key_id: str = ""
    created_at: datetime = field(default_factory=datetime.utcnow)

    def __post_init__(self):
        if not self.key_id:
            self.key_id = hashlib.sha256(self.classical_key[:32]).hexdigest()[:16]

    @property
    def size(self) -> int:
        """Total key size in bytes."""
        return len(self.classical_key) + len(self.pq_key)

    def secure_delete(self) -> None:
        """Securely delete key material."""
        self.classical_key = b"\x00" * len(self.classical_key)
        self.pq_key = b"\x00" * len(self.pq_key)


@dataclass
class HybridKeyPair:
    """Hybrid key pair combining classical and post-quantum."""

    public_key: HybridPublicKey
    private_key: HybridPrivateKey
    algorithm: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (public key only)."""
        return {
            "public_key": self.public_key.to_dict(),
            "algorithm": self.algorithm,
        }


@dataclass
class HybridCiphertext:
    """Hybrid ciphertext from key encapsulation."""

    classical_ciphertext: bytes    # X25519 ephemeral public key
    pq_ciphertext: bytes          # ML-KEM ciphertext
    algorithm: str
    recipient_key_id: str = ""
    created_at: datetime = field(default_factory=datetime.utcnow)

    @property
    def size(self) -> int:
        """Total ciphertext size."""
        return len(self.classical_ciphertext) + len(self.pq_ciphertext)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "classical_ciphertext": base64.b64encode(self.classical_ciphertext).decode(),
            "pq_ciphertext": base64.b64encode(self.pq_ciphertext).decode(),
            "algorithm": self.algorithm,
            "recipient_key_id": self.recipient_key_id,
            "created_at": self.created_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HybridCiphertext":
        """Create from dictionary."""
        return cls(
            classical_ciphertext=base64.b64decode(data["classical_ciphertext"]),
            pq_ciphertext=base64.b64decode(data["pq_ciphertext"]),
            algorithm=data["algorithm"],
            recipient_key_id=data.get("recipient_key_id", ""),
            created_at=datetime.fromisoformat(
                data.get("created_at", datetime.utcnow().isoformat())
            ),
        )


@dataclass
class HybridSignature:
    """Hybrid signature combining classical and post-quantum."""

    classical_signature: bytes    # Ed25519 signature (64 bytes)
    pq_signature: bytes          # ML-DSA signature
    algorithm: str
    signer_key_id: str = ""
    message_hash: str = ""
    created_at: datetime = field(default_factory=datetime.utcnow)

    @property
    def size(self) -> int:
        """Total signature size."""
        return len(self.classical_signature) + len(self.pq_signature)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "classical_signature": base64.b64encode(self.classical_signature).decode(),
            "pq_signature": base64.b64encode(self.pq_signature).decode(),
            "algorithm": self.algorithm,
            "signer_key_id": self.signer_key_id,
            "message_hash": self.message_hash,
            "created_at": self.created_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HybridSignature":
        """Create from dictionary."""
        return cls(
            classical_signature=base64.b64decode(data["classical_signature"]),
            pq_signature=base64.b64decode(data["pq_signature"]),
            algorithm=data["algorithm"],
            signer_key_id=data.get("signer_key_id", ""),
            message_hash=data.get("message_hash", ""),
            created_at=datetime.fromisoformat(
                data.get("created_at", datetime.utcnow().isoformat())
            ),
        )


# =============================================================================
# Hybrid Key Exchange
# =============================================================================


class HybridKeyExchange:
    """
    Hybrid key exchange combining X25519 and ML-KEM.

    The combined shared secret is derived as:
        shared_secret = HKDF(X25519_secret || ML-KEM_secret)

    This ensures security even if one algorithm is compromised.
    """

    def __init__(
        self,
        ml_kem_level: MLKEMSecurityLevel = MLKEMSecurityLevel.ML_KEM_768,
        ml_kem_provider: Optional[MLKEMProvider] = None,
    ):
        self.ml_kem_level = ml_kem_level
        self.ml_kem = ml_kem_provider or DefaultMLKEMProvider()
        self._has_crypto = self._check_crypto()

        # Determine algorithm identifier
        if ml_kem_level == MLKEMSecurityLevel.ML_KEM_1024:
            self.algorithm = HybridKEMAlgorithm.X25519_ML_KEM_1024.value
        else:
            self.algorithm = HybridKEMAlgorithm.X25519_ML_KEM_768.value

    def _check_crypto(self) -> bool:
        """Check if cryptography library is available."""
        try:
            from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey
            return True
        except ImportError:
            logger.warning("cryptography library not available for X25519")
            return False

    def generate_keypair(self) -> HybridKeyPair:
        """
        Generate hybrid X25519 + ML-KEM key pair.

        Returns:
            HybridKeyPair with combined public and private keys
        """
        # Generate X25519 key pair
        classical_public, classical_private = self._generate_x25519_keypair()

        # Generate ML-KEM key pair
        ml_kem_keypair = self.ml_kem.generate_keypair(self.ml_kem_level)

        public_key = HybridPublicKey(
            classical_key=classical_public,
            pq_key=ml_kem_keypair.public_key.key_data,
            algorithm=self.algorithm,
        )

        private_key = HybridPrivateKey(
            classical_key=classical_private,
            pq_key=ml_kem_keypair.private_key.key_data,
            algorithm=self.algorithm,
        )

        return HybridKeyPair(
            public_key=public_key,
            private_key=private_key,
            algorithm=self.algorithm,
        )

    def _generate_x25519_keypair(self) -> Tuple[bytes, bytes]:
        """Generate X25519 key pair."""
        if self._has_crypto:
            from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey
            from cryptography.hazmat.primitives import serialization

            private_key = X25519PrivateKey.generate()
            public_key = private_key.public_key()

            private_bytes = private_key.private_bytes(
                encoding=serialization.Encoding.Raw,
                format=serialization.PrivateFormat.Raw,
                encryption_algorithm=serialization.NoEncryption(),
            )
            public_bytes = public_key.public_bytes(
                encoding=serialization.Encoding.Raw,
                format=serialization.PublicFormat.Raw,
            )

            return public_bytes, private_bytes

        # Mock fallback
        private_bytes = secrets.token_bytes(32)
        public_bytes = hashlib.sha256(private_bytes).digest()
        return public_bytes, private_bytes

    def encapsulate(
        self,
        recipient_public_key: HybridPublicKey,
    ) -> Tuple[bytes, HybridCiphertext]:
        """
        Encapsulate: Generate shared secret using hybrid KEM.

        The process:
        1. Generate ephemeral X25519 key pair
        2. Perform X25519 key exchange
        3. Perform ML-KEM encapsulation
        4. Combine secrets using HKDF

        Args:
            recipient_public_key: Recipient's hybrid public key

        Returns:
            Tuple of (shared_secret, hybrid_ciphertext)
        """
        # X25519 key exchange
        ephemeral_public, ephemeral_private = self._generate_x25519_keypair()
        classical_secret = self._x25519_exchange(
            ephemeral_private,
            recipient_public_key.classical_key,
        )

        # ML-KEM encapsulation
        ml_kem_public = MLKEMPublicKey(
            key_data=recipient_public_key.pq_key,
            security_level=self.ml_kem_level,
        )
        pq_secret, ml_kem_ciphertext = self.ml_kem.encapsulate(ml_kem_public)

        # Combine secrets using HKDF
        combined_secret = self._combine_secrets(
            classical_secret,
            pq_secret,
            recipient_public_key.key_id,
        )

        ciphertext = HybridCiphertext(
            classical_ciphertext=ephemeral_public,
            pq_ciphertext=ml_kem_ciphertext.ciphertext,
            algorithm=self.algorithm,
            recipient_key_id=recipient_public_key.key_id,
        )

        return combined_secret, ciphertext

    def decapsulate(
        self,
        ciphertext: HybridCiphertext,
        private_key: HybridPrivateKey,
    ) -> bytes:
        """
        Decapsulate: Recover shared secret from hybrid ciphertext.

        Args:
            ciphertext: Hybrid ciphertext
            private_key: Recipient's hybrid private key

        Returns:
            Shared secret bytes
        """
        # X25519 key exchange with ephemeral public key
        classical_secret = self._x25519_exchange(
            private_key.classical_key,
            ciphertext.classical_ciphertext,
        )

        # ML-KEM decapsulation
        ml_kem_ciphertext = MLKEMCiphertext(
            ciphertext=ciphertext.pq_ciphertext,
            security_level=self.ml_kem_level,
        )
        ml_kem_private = MLKEMPrivateKey(
            key_data=private_key.pq_key,
            security_level=self.ml_kem_level,
        )
        pq_secret = self.ml_kem.decapsulate(ml_kem_ciphertext, ml_kem_private)

        # Combine secrets
        combined_secret = self._combine_secrets(
            classical_secret,
            pq_secret,
            private_key.key_id,
        )

        return combined_secret

    def _x25519_exchange(
        self,
        private_key: bytes,
        peer_public_key: bytes,
    ) -> bytes:
        """Perform X25519 key exchange."""
        if self._has_crypto:
            from cryptography.hazmat.primitives.asymmetric.x25519 import (
                X25519PrivateKey,
                X25519PublicKey,
            )

            priv = X25519PrivateKey.from_private_bytes(private_key)
            pub = X25519PublicKey.from_public_bytes(peer_public_key)
            return priv.exchange(pub)

        # Mock fallback
        return hashlib.sha256(private_key + peer_public_key).digest()

    def _combine_secrets(
        self,
        classical_secret: bytes,
        pq_secret: bytes,
        context: str,
    ) -> bytes:
        """Combine secrets using HKDF."""
        # Concatenate secrets
        combined = classical_secret + pq_secret

        # Use HKDF to derive final key
        if self._has_crypto:
            from cryptography.hazmat.primitives.kdf.hkdf import HKDF
            from cryptography.hazmat.primitives import hashes

            hkdf = HKDF(
                algorithm=hashes.SHA384(),  # Upgraded from SHA256 for PQ
                length=32,
                salt=b"hybrid-kem-v1",
                info=f"agent-os:hybrid:{context}".encode(),
            )
            return hkdf.derive(combined)

        # Fallback HKDF
        prk = hmac.new(b"hybrid-kem-v1", combined, hashlib.sha384).digest()
        okm = hmac.new(
            prk,
            f"agent-os:hybrid:{context}".encode() + b"\x01",
            hashlib.sha384,
        ).digest()
        return okm[:32]


# =============================================================================
# Hybrid Signatures
# =============================================================================


class HybridSigner:
    """
    Hybrid digital signatures combining Ed25519 and ML-DSA.

    Both signatures must verify for the overall signature to be valid.
    This provides security even if one algorithm is compromised.
    """

    def __init__(
        self,
        ml_dsa_level: MLDSASecurityLevel = MLDSASecurityLevel.ML_DSA_65,
        ml_dsa_provider: Optional[MLDSAProvider] = None,
    ):
        self.ml_dsa_level = ml_dsa_level
        self.ml_dsa = ml_dsa_provider or DefaultMLDSAProvider()
        self._has_crypto = self._check_crypto()

        # Determine algorithm identifier
        if ml_dsa_level == MLDSASecurityLevel.ML_DSA_87:
            self.algorithm = HybridSigAlgorithm.ED25519_ML_DSA_87.value
        else:
            self.algorithm = HybridSigAlgorithm.ED25519_ML_DSA_65.value

    def _check_crypto(self) -> bool:
        """Check if cryptography library is available."""
        try:
            from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
            return True
        except ImportError:
            logger.warning("cryptography library not available for Ed25519")
            return False

    def generate_keypair(self) -> HybridKeyPair:
        """
        Generate hybrid Ed25519 + ML-DSA key pair.

        Returns:
            HybridKeyPair with combined signing keys
        """
        # Generate Ed25519 key pair
        classical_public, classical_private = self._generate_ed25519_keypair()

        # Generate ML-DSA key pair
        ml_dsa_keypair = self.ml_dsa.generate_keypair(self.ml_dsa_level)

        public_key = HybridPublicKey(
            classical_key=classical_public,
            pq_key=ml_dsa_keypair.public_key.key_data,
            algorithm=self.algorithm,
        )

        private_key = HybridPrivateKey(
            classical_key=classical_private,
            pq_key=ml_dsa_keypair.private_key.key_data,
            algorithm=self.algorithm,
        )

        return HybridKeyPair(
            public_key=public_key,
            private_key=private_key,
            algorithm=self.algorithm,
        )

    def _generate_ed25519_keypair(self) -> Tuple[bytes, bytes]:
        """Generate Ed25519 key pair."""
        if self._has_crypto:
            from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
            from cryptography.hazmat.primitives import serialization

            private_key = Ed25519PrivateKey.generate()
            public_key = private_key.public_key()

            private_bytes = private_key.private_bytes(
                encoding=serialization.Encoding.Raw,
                format=serialization.PrivateFormat.Raw,
                encryption_algorithm=serialization.NoEncryption(),
            )
            public_bytes = public_key.public_bytes(
                encoding=serialization.Encoding.Raw,
                format=serialization.PublicFormat.Raw,
            )

            return public_bytes, private_bytes

        # Mock fallback
        private_bytes = secrets.token_bytes(32)
        public_bytes = hashlib.sha256(private_bytes).digest()
        return public_bytes, private_bytes

    def sign(
        self,
        message: bytes,
        private_key: HybridPrivateKey,
    ) -> HybridSignature:
        """
        Sign message with both Ed25519 and ML-DSA.

        Args:
            message: Message to sign
            private_key: Hybrid private key

        Returns:
            HybridSignature containing both signatures
        """
        # Ed25519 signature
        classical_sig = self._ed25519_sign(message, private_key.classical_key)

        # ML-DSA signature
        ml_dsa_private = MLDSAPrivateKey(
            key_data=private_key.pq_key,
            security_level=self.ml_dsa_level,
        )
        pq_sig = self.ml_dsa.sign(message, ml_dsa_private)

        return HybridSignature(
            classical_signature=classical_sig,
            pq_signature=pq_sig.signature,
            algorithm=self.algorithm,
            signer_key_id=private_key.key_id,
            message_hash=hashlib.sha256(message).hexdigest(),
        )

    def verify(
        self,
        message: bytes,
        signature: HybridSignature,
        public_key: HybridPublicKey,
    ) -> bool:
        """
        Verify hybrid signature (both must be valid).

        Args:
            message: Original message
            signature: Hybrid signature
            public_key: Signer's hybrid public key

        Returns:
            True only if BOTH signatures are valid
        """
        # Verify Ed25519 signature
        classical_valid = self._ed25519_verify(
            message,
            signature.classical_signature,
            public_key.classical_key,
        )

        if not classical_valid:
            logger.warning("Classical (Ed25519) signature verification failed")
            return False

        # Verify ML-DSA signature
        ml_dsa_public = MLDSAPublicKey(
            key_data=public_key.pq_key,
            security_level=self.ml_dsa_level,
        )
        ml_dsa_sig = MLDSASignature(
            signature=signature.pq_signature,
            security_level=self.ml_dsa_level,
            message_hash=signature.message_hash,
        )
        pq_valid = self.ml_dsa.verify(message, ml_dsa_sig, ml_dsa_public)

        if not pq_valid:
            logger.warning("Post-quantum (ML-DSA) signature verification failed")
            return False

        return True

    def _ed25519_sign(self, message: bytes, private_key: bytes) -> bytes:
        """Sign with Ed25519.

        Raises:
            ImportError: If the cryptography library is not available.
                        This is a security-critical function and MUST NOT
                        fall back to insecure alternatives.
        """
        if self._has_crypto:
            from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

            key = Ed25519PrivateKey.from_private_bytes(private_key)
            return key.sign(message)

        # SECURITY: Fail securely - do NOT use insecure mock fallbacks
        raise ImportError(
            "The 'cryptography' library is required for Ed25519 signing. "
            "Install with: pip install cryptography"
        )

    def _ed25519_verify(
        self,
        message: bytes,
        signature: bytes,
        public_key: bytes,
    ) -> bool:
        """Verify Ed25519 signature.

        Raises:
            ImportError: If the cryptography library is not available.
                        This is a security-critical function and MUST NOT
                        fall back to insecure alternatives.
        """
        if self._has_crypto:
            from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

            try:
                key = Ed25519PublicKey.from_public_bytes(public_key)
                key.verify(signature, message)
                return True
            except Exception:
                return False

        # SECURITY: Fail securely - do NOT use insecure mock fallbacks
        raise ImportError(
            "The 'cryptography' library is required for Ed25519 verification. "
            "Install with: pip install cryptography"
        )


# =============================================================================
# Hybrid Session Manager
# =============================================================================


@dataclass
class HybridSessionKey:
    """Session key from hybrid key exchange."""

    key_id: str
    key_data: bytes
    peer_id: str
    algorithm: str
    created_at: datetime = field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None
    message_count: int = 0

    def __post_init__(self):
        if not self.expires_at:
            self.expires_at = self.created_at + timedelta(hours=24)

    @property
    def is_valid(self) -> bool:
        """Check if session is still valid."""
        if not self.expires_at:
            return True
        return datetime.utcnow() < self.expires_at

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (excluding key data)."""
        return {
            "key_id": self.key_id,
            "peer_id": self.peer_id,
            "algorithm": self.algorithm,
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "message_count": self.message_count,
        }


class HybridSessionManager:
    """
    Manages hybrid encryption sessions with peers.

    Provides quantum-resistant session establishment and key management.
    """

    def __init__(
        self,
        node_id: str,
        mode: HybridMode = HybridMode.HYBRID,
        ml_kem_level: MLKEMSecurityLevel = MLKEMSecurityLevel.ML_KEM_768,
    ):
        self.node_id = node_id
        self.mode = mode
        self.kex = HybridKeyExchange(ml_kem_level=ml_kem_level)

        # Own key pair
        self._keypair: Optional[HybridKeyPair] = None

        # Session keys by peer
        self._sessions: Dict[str, HybridSessionKey] = {}

    def initialize(self) -> HybridPublicKey:
        """
        Initialize session manager and generate key pair.

        Returns:
            Our public key for sharing with peers
        """
        self._keypair = self.kex.generate_keypair()
        logger.info(
            f"Initialized hybrid session manager with algorithm: {self.kex.algorithm}"
        )
        return self._keypair.public_key

    @property
    def public_key(self) -> Optional[HybridPublicKey]:
        """Get our public key."""
        return self._keypair.public_key if self._keypair else None

    def create_session(
        self,
        peer_id: str,
        peer_public_key: HybridPublicKey,
    ) -> Tuple[HybridSessionKey, HybridCiphertext]:
        """
        Create a new session with a peer.

        Args:
            peer_id: Peer node ID
            peer_public_key: Peer's hybrid public key

        Returns:
            Tuple of (session_key, ciphertext_to_send_to_peer)
        """
        # Perform hybrid encapsulation
        shared_secret, ciphertext = self.kex.encapsulate(peer_public_key)

        # Create session
        session = HybridSessionKey(
            key_id=secrets.token_hex(8),
            key_data=shared_secret,
            peer_id=peer_id,
            algorithm=self.kex.algorithm,
        )

        self._sessions[peer_id] = session
        logger.info(f"Created hybrid session with peer: {peer_id}")

        return session, ciphertext

    def accept_session(
        self,
        peer_id: str,
        ciphertext: HybridCiphertext,
    ) -> Optional[HybridSessionKey]:
        """
        Accept a session from a peer by decapsulating their ciphertext.

        Args:
            peer_id: Peer node ID
            ciphertext: Ciphertext from peer

        Returns:
            Session key or None if decapsulation fails
        """
        if not self._keypair:
            logger.error("Session manager not initialized")
            return None

        try:
            # Perform hybrid decapsulation
            shared_secret = self.kex.decapsulate(
                ciphertext,
                self._keypair.private_key,
            )

            # Create session
            session = HybridSessionKey(
                key_id=secrets.token_hex(8),
                key_data=shared_secret,
                peer_id=peer_id,
                algorithm=self.kex.algorithm,
            )

            self._sessions[peer_id] = session
            logger.info(f"Accepted hybrid session from peer: {peer_id}")

            return session

        except Exception as e:
            logger.error(f"Failed to accept session: {e}")
            return None

    def get_session(self, peer_id: str) -> Optional[HybridSessionKey]:
        """Get session key for a peer."""
        session = self._sessions.get(peer_id)

        if session and not session.is_valid:
            logger.warning(f"Session expired for peer: {peer_id}")
            del self._sessions[peer_id]
            return None

        return session

    def remove_session(self, peer_id: str) -> None:
        """Remove session with a peer."""
        if peer_id in self._sessions:
            # Secure delete key material
            session = self._sessions[peer_id]
            session.key_data = b"\x00" * len(session.key_data)
            del self._sessions[peer_id]
            logger.info(f"Removed hybrid session with peer: {peer_id}")

    def list_sessions(self) -> List[Dict[str, Any]]:
        """List all active sessions."""
        return [
            session.to_dict()
            for session in self._sessions.values()
            if session.is_valid
        ]


# =============================================================================
# Factory Functions
# =============================================================================


def create_hybrid_key_exchange(
    ml_kem_level: MLKEMSecurityLevel = MLKEMSecurityLevel.ML_KEM_768,
) -> HybridKeyExchange:
    """Create a hybrid key exchange instance."""
    return HybridKeyExchange(ml_kem_level=ml_kem_level)


def create_hybrid_signer(
    ml_dsa_level: MLDSASecurityLevel = MLDSASecurityLevel.ML_DSA_65,
) -> HybridSigner:
    """Create a hybrid signer instance."""
    return HybridSigner(ml_dsa_level=ml_dsa_level)


def create_hybrid_session_manager(
    node_id: str,
    mode: HybridMode = HybridMode.HYBRID,
    ml_kem_level: MLKEMSecurityLevel = MLKEMSecurityLevel.ML_KEM_768,
) -> HybridSessionManager:
    """Create a hybrid session manager."""
    return HybridSessionManager(
        node_id=node_id,
        mode=mode,
        ml_kem_level=ml_kem_level,
    )
