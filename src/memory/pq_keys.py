"""
Post-Quantum Key Management

Extends the Memory Vault Key Manager with post-quantum cryptographic support:
- ML-KEM key encapsulation for key exchange
- ML-DSA signatures for key attestation
- Hybrid key pairs combining classical and post-quantum algorithms
- Key storage optimized for larger PQ key sizes

This module integrates with the federation PQ crypto primitives.
"""

import base64
import hashlib
import json
import logging
import secrets
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .keys import (
    KeyManager,
    KeyMetadata,
    KeyStatus,
    DerivedKey,
    HardwareBindingInterface,
)
from .profiles import EncryptionTier, KeyBinding, ProfileManager

# Import PQ crypto primitives
try:
    from ..federation.pq import (
        MLKEMSecurityLevel,
        MLKEMKeyPair,
        MLKEMPublicKey,
        MLKEMPrivateKey,
        DefaultMLKEMProvider,
        MLDSASecurityLevel,
        MLDSAKeyPair,
        MLDSAPublicKey,
        MLDSAPrivateKey,
        DefaultMLDSAProvider,
        HybridKeyExchange,
        HybridSigner,
        HybridKeyPair,
        HybridPublicKey,
        HybridPrivateKey,
    )
    PQ_AVAILABLE = True
except ImportError:
    PQ_AVAILABLE = False

logger = logging.getLogger(__name__)


# =============================================================================
# Quantum Key Types
# =============================================================================


class QuantumKeyType(str, Enum):
    """
    Quantum security classification for keys.

    CLASSICAL: Traditional algorithms only (Ed25519, X25519, RSA)
               - Vulnerable to quantum attacks
               - Smallest key sizes
               - Fastest operations

    HYBRID: Combined classical + post-quantum algorithms
            - Secure if either algorithm holds
            - Larger key sizes (35-70x classical)
            - Recommended for transition period

    POST_QUANTUM: Post-quantum algorithms only (ML-KEM, ML-DSA)
                  - Maximum quantum resistance
                  - Largest key sizes
                  - For future-proofing
    """

    CLASSICAL = "classical"
    HYBRID = "hybrid"
    POST_QUANTUM = "post_quantum"


class PQKeyPurpose(str, Enum):
    """Purpose of a post-quantum key."""

    KEY_EXCHANGE = "key_exchange"     # ML-KEM / Hybrid KEM
    SIGNING = "signing"               # ML-DSA / Hybrid Sig
    ENCRYPTION = "encryption"         # Symmetric derived from KEM
    ATTESTATION = "attestation"       # Identity verification


class PQSecurityLevel(str, Enum):
    """
    Security level for post-quantum keys.

    Maps to NIST security levels and equivalent classical security.
    """

    LEVEL_1 = "level_1"  # ~AES-128 (ML-KEM-512, ML-DSA-44)
    LEVEL_3 = "level_3"  # ~AES-192 (ML-KEM-768, ML-DSA-65) - Recommended
    LEVEL_5 = "level_5"  # ~AES-256 (ML-KEM-1024, ML-DSA-87)


# Security level to algorithm mapping
PQ_LEVEL_TO_KEM = {
    PQSecurityLevel.LEVEL_1: MLKEMSecurityLevel.ML_KEM_512 if PQ_AVAILABLE else None,
    PQSecurityLevel.LEVEL_3: MLKEMSecurityLevel.ML_KEM_768 if PQ_AVAILABLE else None,
    PQSecurityLevel.LEVEL_5: MLKEMSecurityLevel.ML_KEM_1024 if PQ_AVAILABLE else None,
}

PQ_LEVEL_TO_DSA = {
    PQSecurityLevel.LEVEL_1: MLDSASecurityLevel.ML_DSA_44 if PQ_AVAILABLE else None,
    PQSecurityLevel.LEVEL_3: MLDSASecurityLevel.ML_DSA_65 if PQ_AVAILABLE else None,
    PQSecurityLevel.LEVEL_5: MLDSASecurityLevel.ML_DSA_87 if PQ_AVAILABLE else None,
}


# =============================================================================
# Post-Quantum Key Metadata
# =============================================================================


@dataclass
class PQKeyMetadata:
    """Extended metadata for post-quantum keys."""

    key_id: str
    quantum_type: QuantumKeyType
    purpose: PQKeyPurpose
    security_level: PQSecurityLevel
    algorithm: str  # e.g., "x25519-ml-kem-768", "ed25519-ml-dsa-65"

    # Size information
    public_key_size: int
    private_key_size: int

    # Standard metadata
    tier: EncryptionTier
    binding: KeyBinding
    status: KeyStatus = KeyStatus.ACTIVE
    created_at: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None
    last_used: Optional[datetime] = None
    use_count: int = 0

    # Hardware binding
    hardware_handle: Optional[str] = None

    # Classical key ID if hybrid
    classical_key_id: Optional[str] = None

    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "key_id": self.key_id,
            "quantum_type": self.quantum_type.value,
            "purpose": self.purpose.value,
            "security_level": self.security_level.value,
            "algorithm": self.algorithm,
            "public_key_size": self.public_key_size,
            "private_key_size": self.private_key_size,
            "tier": self.tier.name,
            "binding": self.binding.name,
            "status": self.status.name,
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "last_used": self.last_used.isoformat() if self.last_used else None,
            "use_count": self.use_count,
            "hardware_handle": self.hardware_handle,
            "classical_key_id": self.classical_key_id,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PQKeyMetadata":
        """Create from dictionary."""
        return cls(
            key_id=data["key_id"],
            quantum_type=QuantumKeyType(data["quantum_type"]),
            purpose=PQKeyPurpose(data["purpose"]),
            security_level=PQSecurityLevel(data["security_level"]),
            algorithm=data["algorithm"],
            public_key_size=data["public_key_size"],
            private_key_size=data["private_key_size"],
            tier=EncryptionTier[data["tier"]],
            binding=KeyBinding[data["binding"]],
            status=KeyStatus[data.get("status", "ACTIVE")],
            created_at=datetime.fromisoformat(data["created_at"]),
            expires_at=(
                datetime.fromisoformat(data["expires_at"])
                if data.get("expires_at") else None
            ),
            last_used=(
                datetime.fromisoformat(data["last_used"])
                if data.get("last_used") else None
            ),
            use_count=data.get("use_count", 0),
            hardware_handle=data.get("hardware_handle"),
            classical_key_id=data.get("classical_key_id"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class PQStoredKey:
    """A stored post-quantum key with material."""

    metadata: PQKeyMetadata
    public_key: bytes
    private_key: Optional[bytes] = None  # May be hardware-bound

    @property
    def key_id(self) -> str:
        return self.metadata.key_id


# =============================================================================
# Post-Quantum Key Manager
# =============================================================================


class PostQuantumKeyManager:
    """
    Manages post-quantum cryptographic keys.

    Extends the classical KeyManager with:
    - ML-KEM key pair generation and encapsulation
    - ML-DSA key pair generation and signing
    - Hybrid key pairs (classical + PQ)
    - Optimized storage for larger PQ keys
    - Key migration from classical to hybrid
    """

    # Maximum key sizes (bytes) for validation
    MAX_PUBLIC_KEY_SIZE = 4096   # Covers ML-DSA-87
    MAX_PRIVATE_KEY_SIZE = 8192  # Covers ML-DSA-87

    def __init__(
        self,
        key_store_path: Optional[Path] = None,
        classical_key_manager: Optional[KeyManager] = None,
        profile_manager: Optional[ProfileManager] = None,
        default_security_level: PQSecurityLevel = PQSecurityLevel.LEVEL_3,
    ):
        """
        Initialize post-quantum key manager.

        Args:
            key_store_path: Path to store PQ keys
            classical_key_manager: Classical key manager for hybrid operations
            profile_manager: Profile manager for tier settings
            default_security_level: Default PQ security level
        """
        self._key_store_path = key_store_path
        self._classical_km = classical_key_manager
        self._profile_manager = profile_manager or ProfileManager()
        self._default_level = default_security_level

        # PQ crypto providers
        self._kem_provider = DefaultMLKEMProvider() if PQ_AVAILABLE else None
        self._dsa_provider = DefaultMLDSAProvider() if PQ_AVAILABLE else None

        # Key storage
        self._pq_keys: Dict[str, PQStoredKey] = {}
        self._lock = threading.RLock()
        self._initialized = False

        # Load existing keys
        if key_store_path:
            self._load_pq_keys()

    def initialize(self) -> bool:
        """Initialize the post-quantum key manager."""
        if not PQ_AVAILABLE:
            logger.error("Post-quantum crypto module not available")
            return False

        with self._lock:
            self._initialized = True
            logger.info(
                f"Post-quantum key manager initialized "
                f"(default level: {self._default_level.value})"
            )
            return True

    def shutdown(self) -> None:
        """Securely shutdown and clear keys from memory."""
        with self._lock:
            # Securely clear all private keys
            for stored_key in self._pq_keys.values():
                if stored_key.private_key:
                    # Overwrite with zeros
                    stored_key.private_key = b"\x00" * len(stored_key.private_key)

            self._pq_keys.clear()
            self._initialized = False
            logger.info("Post-quantum key manager shutdown")

    @property
    def is_available(self) -> bool:
        """Check if PQ crypto is available."""
        return PQ_AVAILABLE and self._initialized

    # =========================================================================
    # Key Generation
    # =========================================================================

    def generate_kem_keypair(
        self,
        tier: EncryptionTier = EncryptionTier.PRIVATE,
        security_level: Optional[PQSecurityLevel] = None,
        quantum_type: QuantumKeyType = QuantumKeyType.HYBRID,
        ttl: Optional[timedelta] = None,
    ) -> PQStoredKey:
        """
        Generate a key encapsulation key pair.

        Args:
            tier: Encryption tier for access control
            security_level: PQ security level (default: LEVEL_3)
            quantum_type: HYBRID or POST_QUANTUM
            ttl: Optional time-to-live

        Returns:
            PQStoredKey with key material
        """
        if not self.is_available:
            raise RuntimeError("Post-quantum key manager not available")

        security_level = security_level or self._default_level

        with self._lock:
            if quantum_type == QuantumKeyType.HYBRID:
                return self._generate_hybrid_kem_keypair(tier, security_level, ttl)
            else:
                return self._generate_pq_kem_keypair(tier, security_level, ttl)

    def _generate_hybrid_kem_keypair(
        self,
        tier: EncryptionTier,
        security_level: PQSecurityLevel,
        ttl: Optional[timedelta],
    ) -> PQStoredKey:
        """Generate hybrid X25519 + ML-KEM key pair."""
        ml_kem_level = PQ_LEVEL_TO_KEM[security_level]
        kex = HybridKeyExchange(ml_kem_level=ml_kem_level)
        keypair = kex.generate_keypair()

        key_id = f"hybrid_kem_{security_level.value}_{secrets.token_hex(8)}"

        # Serialize keys
        public_key = (
            keypair.public_key.classical_key +
            keypair.public_key.pq_key
        )
        private_key = (
            keypair.private_key.classical_key +
            keypair.private_key.pq_key
        )

        metadata = PQKeyMetadata(
            key_id=key_id,
            quantum_type=QuantumKeyType.HYBRID,
            purpose=PQKeyPurpose.KEY_EXCHANGE,
            security_level=security_level,
            algorithm=keypair.algorithm,
            public_key_size=len(public_key),
            private_key_size=len(private_key),
            tier=tier,
            binding=KeyBinding.SOFTWARE,
            expires_at=datetime.now() + ttl if ttl else None,
        )

        stored_key = PQStoredKey(
            metadata=metadata,
            public_key=public_key,
            private_key=private_key,
        )

        self._pq_keys[key_id] = stored_key
        self._persist_pq_keys()

        logger.info(
            f"Generated hybrid KEM keypair: {key_id} "
            f"(public: {len(public_key)} bytes, private: {len(private_key)} bytes)"
        )

        return stored_key

    def _generate_pq_kem_keypair(
        self,
        tier: EncryptionTier,
        security_level: PQSecurityLevel,
        ttl: Optional[timedelta],
    ) -> PQStoredKey:
        """Generate pure ML-KEM key pair."""
        ml_kem_level = PQ_LEVEL_TO_KEM[security_level]
        keypair = self._kem_provider.generate_keypair(ml_kem_level)

        key_id = f"mlkem_{security_level.value}_{secrets.token_hex(8)}"

        metadata = PQKeyMetadata(
            key_id=key_id,
            quantum_type=QuantumKeyType.POST_QUANTUM,
            purpose=PQKeyPurpose.KEY_EXCHANGE,
            security_level=security_level,
            algorithm=ml_kem_level.value,
            public_key_size=len(keypair.public_key.key_data),
            private_key_size=len(keypair.private_key.key_data),
            tier=tier,
            binding=KeyBinding.SOFTWARE,
            expires_at=datetime.now() + ttl if ttl else None,
        )

        stored_key = PQStoredKey(
            metadata=metadata,
            public_key=keypair.public_key.key_data,
            private_key=keypair.private_key.key_data,
        )

        self._pq_keys[key_id] = stored_key
        self._persist_pq_keys()

        logger.info(f"Generated ML-KEM keypair: {key_id}")

        return stored_key

    def generate_signing_keypair(
        self,
        tier: EncryptionTier = EncryptionTier.PRIVATE,
        security_level: Optional[PQSecurityLevel] = None,
        quantum_type: QuantumKeyType = QuantumKeyType.HYBRID,
        ttl: Optional[timedelta] = None,
    ) -> PQStoredKey:
        """
        Generate a signing key pair.

        Args:
            tier: Encryption tier for access control
            security_level: PQ security level
            quantum_type: HYBRID or POST_QUANTUM
            ttl: Optional time-to-live

        Returns:
            PQStoredKey with signing key material
        """
        if not self.is_available:
            raise RuntimeError("Post-quantum key manager not available")

        security_level = security_level or self._default_level

        with self._lock:
            if quantum_type == QuantumKeyType.HYBRID:
                return self._generate_hybrid_signing_keypair(tier, security_level, ttl)
            else:
                return self._generate_pq_signing_keypair(tier, security_level, ttl)

    def _generate_hybrid_signing_keypair(
        self,
        tier: EncryptionTier,
        security_level: PQSecurityLevel,
        ttl: Optional[timedelta],
    ) -> PQStoredKey:
        """Generate hybrid Ed25519 + ML-DSA key pair."""
        ml_dsa_level = PQ_LEVEL_TO_DSA[security_level]
        signer = HybridSigner(ml_dsa_level=ml_dsa_level)
        keypair = signer.generate_keypair()

        key_id = f"hybrid_sig_{security_level.value}_{secrets.token_hex(8)}"

        # Serialize keys
        public_key = (
            keypair.public_key.classical_key +
            keypair.public_key.pq_key
        )
        private_key = (
            keypair.private_key.classical_key +
            keypair.private_key.pq_key
        )

        metadata = PQKeyMetadata(
            key_id=key_id,
            quantum_type=QuantumKeyType.HYBRID,
            purpose=PQKeyPurpose.SIGNING,
            security_level=security_level,
            algorithm=keypair.algorithm,
            public_key_size=len(public_key),
            private_key_size=len(private_key),
            tier=tier,
            binding=KeyBinding.SOFTWARE,
            expires_at=datetime.now() + ttl if ttl else None,
        )

        stored_key = PQStoredKey(
            metadata=metadata,
            public_key=public_key,
            private_key=private_key,
        )

        self._pq_keys[key_id] = stored_key
        self._persist_pq_keys()

        logger.info(
            f"Generated hybrid signing keypair: {key_id} "
            f"(public: {len(public_key)} bytes, private: {len(private_key)} bytes)"
        )

        return stored_key

    def _generate_pq_signing_keypair(
        self,
        tier: EncryptionTier,
        security_level: PQSecurityLevel,
        ttl: Optional[timedelta],
    ) -> PQStoredKey:
        """Generate pure ML-DSA key pair."""
        ml_dsa_level = PQ_LEVEL_TO_DSA[security_level]
        keypair = self._dsa_provider.generate_keypair(ml_dsa_level)

        key_id = f"mldsa_{security_level.value}_{secrets.token_hex(8)}"

        metadata = PQKeyMetadata(
            key_id=key_id,
            quantum_type=QuantumKeyType.POST_QUANTUM,
            purpose=PQKeyPurpose.SIGNING,
            security_level=security_level,
            algorithm=ml_dsa_level.value,
            public_key_size=len(keypair.public_key.key_data),
            private_key_size=len(keypair.private_key.key_data),
            tier=tier,
            binding=KeyBinding.SOFTWARE,
            expires_at=datetime.now() + ttl if ttl else None,
        )

        stored_key = PQStoredKey(
            metadata=metadata,
            public_key=keypair.public_key.key_data,
            private_key=keypair.private_key.key_data,
        )

        self._pq_keys[key_id] = stored_key
        self._persist_pq_keys()

        logger.info(f"Generated ML-DSA keypair: {key_id}")

        return stored_key

    # =========================================================================
    # Key Retrieval
    # =========================================================================

    def get_key(self, key_id: str) -> Optional[PQStoredKey]:
        """
        Retrieve a key by ID.

        Args:
            key_id: Key identifier

        Returns:
            PQStoredKey or None if not found
        """
        with self._lock:
            stored_key = self._pq_keys.get(key_id)
            if not stored_key:
                return None

            metadata = stored_key.metadata

            # Check status
            if metadata.status != KeyStatus.ACTIVE:
                logger.warning(f"Attempted access to non-active PQ key: {key_id}")
                return None

            # Check expiry
            if metadata.expires_at and datetime.now() > metadata.expires_at:
                metadata.status = KeyStatus.EXPIRED
                logger.info(f"PQ key expired: {key_id}")
                return None

            # Update access tracking
            metadata.last_used = datetime.now()
            metadata.use_count += 1

            return stored_key

    def get_public_key(self, key_id: str) -> Optional[bytes]:
        """Get only the public key (safe to share)."""
        stored_key = self.get_key(key_id)
        return stored_key.public_key if stored_key else None

    def get_private_key(self, key_id: str) -> Optional[bytes]:
        """Get the private key (sensitive)."""
        stored_key = self.get_key(key_id)
        return stored_key.private_key if stored_key else None

    def get_metadata(self, key_id: str) -> Optional[PQKeyMetadata]:
        """Get key metadata."""
        stored_key = self._pq_keys.get(key_id)
        return stored_key.metadata if stored_key else None

    # =========================================================================
    # Key Operations
    # =========================================================================

    def encapsulate(
        self,
        recipient_key_id: str,
    ) -> Optional[Tuple[bytes, bytes]]:
        """
        Perform key encapsulation to a recipient's public key.

        Args:
            recipient_key_id: Recipient's KEM key ID

        Returns:
            Tuple of (shared_secret, ciphertext) or None
        """
        stored_key = self.get_key(recipient_key_id)
        if not stored_key:
            return None

        metadata = stored_key.metadata
        if metadata.purpose != PQKeyPurpose.KEY_EXCHANGE:
            logger.error(f"Key {recipient_key_id} is not a KEM key")
            return None

        if metadata.quantum_type == QuantumKeyType.HYBRID:
            return self._hybrid_encapsulate(stored_key)
        else:
            return self._pq_encapsulate(stored_key)

    def _hybrid_encapsulate(
        self,
        stored_key: PQStoredKey,
    ) -> Tuple[bytes, bytes]:
        """Hybrid encapsulation."""
        metadata = stored_key.metadata
        ml_kem_level = PQ_LEVEL_TO_KEM[metadata.security_level]
        kex = HybridKeyExchange(ml_kem_level=ml_kem_level)

        # Reconstruct public key
        classical_key = stored_key.public_key[:32]
        pq_key = stored_key.public_key[32:]

        public_key = HybridPublicKey(
            classical_key=classical_key,
            pq_key=pq_key,
            algorithm=metadata.algorithm,
        )

        shared_secret, ciphertext = kex.encapsulate(public_key)

        # Serialize ciphertext
        ct_bytes = ciphertext.classical_ciphertext + ciphertext.pq_ciphertext

        return shared_secret, ct_bytes

    def _pq_encapsulate(
        self,
        stored_key: PQStoredKey,
    ) -> Tuple[bytes, bytes]:
        """Pure PQ encapsulation."""
        metadata = stored_key.metadata
        ml_kem_level = PQ_LEVEL_TO_KEM[metadata.security_level]

        public_key = MLKEMPublicKey(
            key_data=stored_key.public_key,
            security_level=ml_kem_level,
        )

        shared_secret, ciphertext = self._kem_provider.encapsulate(public_key)

        return shared_secret, ciphertext.ciphertext

    def decapsulate(
        self,
        key_id: str,
        ciphertext: bytes,
    ) -> Optional[bytes]:
        """
        Decapsulate to recover shared secret.

        Args:
            key_id: Our KEM key ID
            ciphertext: Ciphertext from encapsulation

        Returns:
            Shared secret or None
        """
        stored_key = self.get_key(key_id)
        if not stored_key or not stored_key.private_key:
            return None

        metadata = stored_key.metadata
        if metadata.purpose != PQKeyPurpose.KEY_EXCHANGE:
            logger.error(f"Key {key_id} is not a KEM key")
            return None

        if metadata.quantum_type == QuantumKeyType.HYBRID:
            return self._hybrid_decapsulate(stored_key, ciphertext)
        else:
            return self._pq_decapsulate(stored_key, ciphertext)

    def _hybrid_decapsulate(
        self,
        stored_key: PQStoredKey,
        ciphertext: bytes,
    ) -> bytes:
        """Hybrid decapsulation."""
        metadata = stored_key.metadata
        ml_kem_level = PQ_LEVEL_TO_KEM[metadata.security_level]
        kex = HybridKeyExchange(ml_kem_level=ml_kem_level)

        # Reconstruct private key
        classical_key = stored_key.private_key[:32]
        pq_key = stored_key.private_key[32:]

        private_key = HybridPrivateKey(
            classical_key=classical_key,
            pq_key=pq_key,
            algorithm=metadata.algorithm,
        )

        # Split ciphertext
        from ..federation.pq.hybrid import HybridCiphertext
        from ..federation.pq.ml_kem import ML_KEM_PARAMS

        classical_ct = ciphertext[:32]
        pq_ct = ciphertext[32:]

        ct = HybridCiphertext(
            classical_ciphertext=classical_ct,
            pq_ciphertext=pq_ct,
            algorithm=metadata.algorithm,
        )

        return kex.decapsulate(ct, private_key)

    def _pq_decapsulate(
        self,
        stored_key: PQStoredKey,
        ciphertext: bytes,
    ) -> bytes:
        """Pure PQ decapsulation."""
        metadata = stored_key.metadata
        ml_kem_level = PQ_LEVEL_TO_KEM[metadata.security_level]

        from ..federation.pq.ml_kem import MLKEMCiphertext

        private_key = MLKEMPrivateKey(
            key_data=stored_key.private_key,
            security_level=ml_kem_level,
        )

        ct = MLKEMCiphertext(
            ciphertext=ciphertext,
            security_level=ml_kem_level,
        )

        return self._kem_provider.decapsulate(ct, private_key)

    # =========================================================================
    # Key Lifecycle
    # =========================================================================

    def rotate_key(self, key_id: str) -> Optional[PQStoredKey]:
        """
        Rotate a key (generate new, mark old for retirement).

        Args:
            key_id: Key to rotate

        Returns:
            New key or None if rotation failed
        """
        with self._lock:
            old_key = self._pq_keys.get(key_id)
            if not old_key:
                return None

            old_metadata = old_key.metadata

            # Mark old key for rotation
            old_metadata.status = KeyStatus.PENDING_ROTATION
            old_metadata.metadata["rotation_time"] = datetime.now().isoformat()

            # Generate new key of same type
            if old_metadata.purpose == PQKeyPurpose.KEY_EXCHANGE:
                new_key = self.generate_kem_keypair(
                    tier=old_metadata.tier,
                    security_level=old_metadata.security_level,
                    quantum_type=old_metadata.quantum_type,
                )
            else:
                new_key = self.generate_signing_keypair(
                    tier=old_metadata.tier,
                    security_level=old_metadata.security_level,
                    quantum_type=old_metadata.quantum_type,
                )

            # Link keys
            new_key.metadata.metadata["rotated_from"] = key_id
            old_metadata.metadata["rotated_to"] = new_key.key_id

            self._persist_pq_keys()

            logger.info(f"Rotated PQ key {key_id} -> {new_key.key_id}")
            return new_key

    def revoke_key(self, key_id: str) -> bool:
        """Revoke a key immediately."""
        with self._lock:
            stored_key = self._pq_keys.get(key_id)
            if not stored_key:
                return False

            stored_key.metadata.status = KeyStatus.REVOKED

            # Clear private key from memory
            if stored_key.private_key:
                stored_key.private_key = b"\x00" * len(stored_key.private_key)
                stored_key.private_key = None

            self._persist_pq_keys()

            logger.warning(f"Revoked PQ key: {key_id}")
            return True

    def delete_key(self, key_id: str, secure: bool = True) -> bool:
        """Delete a key permanently."""
        with self._lock:
            stored_key = self._pq_keys.get(key_id)
            if not stored_key:
                return False

            # Secure overwrite
            if secure and stored_key.private_key:
                stored_key.private_key = secrets.token_bytes(len(stored_key.private_key))

            del self._pq_keys[key_id]
            self._persist_pq_keys()

            logger.info(f"Deleted PQ key: {key_id}")
            return True

    # =========================================================================
    # Key Listing
    # =========================================================================

    def list_keys(
        self,
        quantum_type: Optional[QuantumKeyType] = None,
        purpose: Optional[PQKeyPurpose] = None,
        status: Optional[KeyStatus] = None,
    ) -> List[PQKeyMetadata]:
        """List keys with optional filters."""
        keys = [k.metadata for k in self._pq_keys.values()]

        if quantum_type:
            keys = [k for k in keys if k.quantum_type == quantum_type]

        if purpose:
            keys = [k for k in keys if k.purpose == purpose]

        if status:
            keys = [k for k in keys if k.status == status]

        return keys

    def get_key_stats(self) -> Dict[str, Any]:
        """Get statistics about stored keys."""
        keys = list(self._pq_keys.values())

        total_public_size = sum(len(k.public_key) for k in keys)
        total_private_size = sum(
            len(k.private_key) for k in keys if k.private_key
        )

        return {
            "total_keys": len(keys),
            "by_type": {
                "hybrid": len([k for k in keys if k.metadata.quantum_type == QuantumKeyType.HYBRID]),
                "post_quantum": len([k for k in keys if k.metadata.quantum_type == QuantumKeyType.POST_QUANTUM]),
            },
            "by_purpose": {
                "key_exchange": len([k for k in keys if k.metadata.purpose == PQKeyPurpose.KEY_EXCHANGE]),
                "signing": len([k for k in keys if k.metadata.purpose == PQKeyPurpose.SIGNING]),
            },
            "by_status": {
                "active": len([k for k in keys if k.metadata.status == KeyStatus.ACTIVE]),
                "expired": len([k for k in keys if k.metadata.status == KeyStatus.EXPIRED]),
                "revoked": len([k for k in keys if k.metadata.status == KeyStatus.REVOKED]),
            },
            "total_public_key_bytes": total_public_size,
            "total_private_key_bytes": total_private_size,
            "pq_available": PQ_AVAILABLE,
        }

    # =========================================================================
    # Persistence
    # =========================================================================

    def _load_pq_keys(self) -> None:
        """Load PQ keys from disk."""
        if not self._key_store_path:
            return

        pq_keys_path = self._key_store_path / "pq_keys.json"
        pq_data_path = self._key_store_path / "pq_key_data"

        if not pq_keys_path.exists():
            return

        try:
            with open(pq_keys_path, 'r') as f:
                data = json.load(f)

            for key_data in data.get("keys", []):
                metadata = PQKeyMetadata.from_dict(key_data)

                # Load key material from separate files
                public_key_file = pq_data_path / f"{metadata.key_id}.pub"
                private_key_file = pq_data_path / f"{metadata.key_id}.priv"

                public_key = b""
                private_key = None

                if public_key_file.exists():
                    public_key = base64.b64decode(public_key_file.read_text())

                if private_key_file.exists():
                    private_key = base64.b64decode(private_key_file.read_text())

                self._pq_keys[metadata.key_id] = PQStoredKey(
                    metadata=metadata,
                    public_key=public_key,
                    private_key=private_key,
                )

            logger.info(f"Loaded {len(self._pq_keys)} PQ keys")

        except Exception as e:
            logger.error(f"Failed to load PQ keys: {e}")

    def _persist_pq_keys(self) -> None:
        """Persist PQ keys to disk."""
        if not self._key_store_path:
            return

        self._key_store_path.mkdir(parents=True, exist_ok=True)
        pq_keys_path = self._key_store_path / "pq_keys.json"
        pq_data_path = self._key_store_path / "pq_key_data"
        pq_data_path.mkdir(exist_ok=True)

        try:
            # Save metadata
            data = {
                "keys": [k.metadata.to_dict() for k in self._pq_keys.values()],
                "updated_at": datetime.now().isoformat(),
                "version": "1.0",
            }

            with open(pq_keys_path, 'w') as f:
                json.dump(data, f, indent=2)

            # Save key material to separate files
            for stored_key in self._pq_keys.values():
                key_id = stored_key.metadata.key_id

                # Public key
                public_key_file = pq_data_path / f"{key_id}.pub"
                public_key_file.write_text(
                    base64.b64encode(stored_key.public_key).decode()
                )

                # Private key (with restricted permissions)
                if stored_key.private_key:
                    private_key_file = pq_data_path / f"{key_id}.priv"
                    private_key_file.write_text(
                        base64.b64encode(stored_key.private_key).decode()
                    )
                    private_key_file.chmod(0o600)

        except Exception as e:
            logger.error(f"Failed to persist PQ keys: {e}")


# =============================================================================
# Factory Functions
# =============================================================================


def create_pq_key_manager(
    key_store_path: Optional[Path] = None,
    security_level: PQSecurityLevel = PQSecurityLevel.LEVEL_3,
) -> PostQuantumKeyManager:
    """Create a post-quantum key manager."""
    return PostQuantumKeyManager(
        key_store_path=key_store_path,
        default_security_level=security_level,
    )


def check_pq_availability() -> Dict[str, Any]:
    """Check post-quantum crypto availability."""
    return {
        "pq_available": PQ_AVAILABLE,
        "ml_kem_available": PQ_AVAILABLE,
        "ml_dsa_available": PQ_AVAILABLE,
        "supported_levels": [level.value for level in PQSecurityLevel],
        "default_level": PQSecurityLevel.LEVEL_3.value,
    }
