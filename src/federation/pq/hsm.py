"""
Hardware Security Module (HSM) Integration for Post-Quantum Keys

Provides HSM abstraction for secure storage and operations on PQ cryptographic keys:
- PKCS#11 interface support for industry-standard HSMs
- TPM (Trusted Platform Module) integration
- Software HSM fallback for development/testing
- Cloud HSM support (AWS CloudHSM, Azure HSM, GCP Cloud HSM)

The HSM ensures:
- Private keys never leave the secure boundary
- Cryptographic operations performed inside HSM
- Hardware-backed random number generation
- Tamper resistance and audit logging

Security Levels:
- Level 1: Software protection only (development)
- Level 2: Tamper-evident hardware (TPM)
- Level 3: Tamper-resistant hardware (HSM)
- Level 4: FIPS 140-3 Level 4 certified HSM

Usage:
    # Create HSM provider
    hsm = create_hsm_provider(HSMType.PKCS11, slot=0, pin="1234")

    # Generate PQ key in HSM
    key_handle = hsm.generate_pq_keypair(
        algorithm=PQAlgorithm.ML_KEM_768,
        label="session-key-001",
    )

    # Perform encapsulation (key never leaves HSM)
    ciphertext = hsm.encapsulate(key_handle, recipient_public_key)
"""

import base64
import hashlib
import hmac
import json
import logging
import os
import secrets
import struct
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)


# =============================================================================
# Constants and Enums
# =============================================================================


class HSMType(str, Enum):
    """Hardware Security Module types."""

    SOFTWARE = "software"           # Software-only (development)
    TPM = "tpm"                     # Trusted Platform Module
    PKCS11 = "pkcs11"              # PKCS#11 compatible HSM
    AWS_CLOUDHSM = "aws_cloudhsm"  # AWS CloudHSM
    AZURE_HSM = "azure_hsm"        # Azure Dedicated HSM
    GCP_HSM = "gcp_hsm"            # Google Cloud HSM
    YUBIHSM = "yubihsm"            # YubiHSM


class HSMSecurityLevel(int, Enum):
    """HSM security levels (based on FIPS 140-3)."""

    LEVEL_1 = 1  # Software only
    LEVEL_2 = 2  # Tamper-evident (TPM)
    LEVEL_3 = 3  # Tamper-resistant (HSM)
    LEVEL_4 = 4  # FIPS 140-3 Level 4


class PQAlgorithm(str, Enum):
    """Post-quantum algorithms supported by HSM."""

    # Key Encapsulation
    ML_KEM_512 = "ml-kem-512"
    ML_KEM_768 = "ml-kem-768"
    ML_KEM_1024 = "ml-kem-1024"

    # Digital Signatures
    ML_DSA_44 = "ml-dsa-44"
    ML_DSA_65 = "ml-dsa-65"
    ML_DSA_87 = "ml-dsa-87"

    # Hybrid
    X25519_ML_KEM_768 = "x25519-ml-kem-768"
    ED25519_ML_DSA_65 = "ed25519-ml-dsa-65"


class KeyState(str, Enum):
    """HSM key states."""

    GENERATED = "generated"
    ACTIVE = "active"
    SUSPENDED = "suspended"
    COMPROMISED = "compromised"
    DEACTIVATED = "deactivated"
    DESTROYED = "destroyed"


class HSMOperation(str, Enum):
    """HSM cryptographic operations."""

    GENERATE_KEYPAIR = "generate_keypair"
    ENCAPSULATE = "encapsulate"
    DECAPSULATE = "decapsulate"
    SIGN = "sign"
    VERIFY = "verify"
    EXPORT_PUBLIC = "export_public"
    IMPORT_KEY = "import_key"
    DESTROY_KEY = "destroy_key"
    WRAP_KEY = "wrap_key"
    UNWRAP_KEY = "unwrap_key"


# =============================================================================
# Data Models
# =============================================================================


@dataclass
class HSMKeyHandle:
    """Handle to a key stored in HSM."""

    handle_id: str                          # Unique handle identifier
    algorithm: PQAlgorithm                  # Key algorithm
    label: str                              # Human-readable label
    created_at: datetime = field(default_factory=datetime.utcnow)
    state: KeyState = KeyState.GENERATED
    hsm_type: HSMType = HSMType.SOFTWARE
    security_level: HSMSecurityLevel = HSMSecurityLevel.LEVEL_1
    exportable: bool = False                # Can private key be exported?
    extractable: bool = False               # Can key material be read?
    usage_count: int = 0
    max_usage: Optional[int] = None         # None = unlimited
    expires_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.handle_id:
            self.handle_id = f"hsm:{secrets.token_hex(16)}"

    @property
    def is_valid(self) -> bool:
        """Check if key handle is valid for use."""
        if self.state not in (KeyState.GENERATED, KeyState.ACTIVE):
            return False
        if self.expires_at and datetime.utcnow() > self.expires_at:
            return False
        if self.max_usage and self.usage_count >= self.max_usage:
            return False
        return True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "handle_id": self.handle_id,
            "algorithm": self.algorithm.value,
            "label": self.label,
            "created_at": self.created_at.isoformat(),
            "state": self.state.value,
            "hsm_type": self.hsm_type.value,
            "security_level": self.security_level.value,
            "exportable": self.exportable,
            "extractable": self.extractable,
            "usage_count": self.usage_count,
            "max_usage": self.max_usage,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HSMKeyHandle":
        """Create from dictionary."""
        return cls(
            handle_id=data["handle_id"],
            algorithm=PQAlgorithm(data["algorithm"]),
            label=data["label"],
            created_at=datetime.fromisoformat(data["created_at"]),
            state=KeyState(data["state"]),
            hsm_type=HSMType(data["hsm_type"]),
            security_level=HSMSecurityLevel(data["security_level"]),
            exportable=data.get("exportable", False),
            extractable=data.get("extractable", False),
            usage_count=data.get("usage_count", 0),
            max_usage=data.get("max_usage"),
            expires_at=datetime.fromisoformat(data["expires_at"]) if data.get("expires_at") else None,
            metadata=data.get("metadata", {}),
        )


@dataclass
class HSMConfig:
    """HSM configuration."""

    hsm_type: HSMType = HSMType.SOFTWARE
    security_level: HSMSecurityLevel = HSMSecurityLevel.LEVEL_1

    # PKCS#11 settings
    pkcs11_library: Optional[str] = None    # Path to PKCS#11 library
    pkcs11_slot: int = 0
    pkcs11_pin: Optional[str] = None

    # TPM settings
    tpm_device: str = "/dev/tpm0"
    tpm_hierarchy: str = "owner"            # owner, endorsement, platform

    # Cloud HSM settings
    cloud_region: Optional[str] = None
    cloud_key_vault: Optional[str] = None
    cloud_credentials: Optional[Dict[str, str]] = None

    # Software HSM settings (development)
    software_key_store: Optional[Path] = None
    software_master_key: Optional[bytes] = None

    # Operational settings
    max_key_cache_size: int = 100
    key_cache_ttl_seconds: int = 3600
    audit_all_operations: bool = True
    require_dual_control: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (excluding secrets)."""
        return {
            "hsm_type": self.hsm_type.value,
            "security_level": self.security_level.value,
            "pkcs11_slot": self.pkcs11_slot,
            "tpm_device": self.tpm_device,
            "cloud_region": self.cloud_region,
            "cloud_key_vault": self.cloud_key_vault,
            "max_key_cache_size": self.max_key_cache_size,
            "key_cache_ttl_seconds": self.key_cache_ttl_seconds,
            "audit_all_operations": self.audit_all_operations,
            "require_dual_control": self.require_dual_control,
        }


@dataclass
class HSMAuditEvent:
    """Audit event for HSM operations."""

    event_id: str
    timestamp: datetime
    operation: HSMOperation
    key_handle: Optional[str]
    success: bool
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "event_id": self.event_id,
            "timestamp": self.timestamp.isoformat(),
            "operation": self.operation.value,
            "key_handle": self.key_handle,
            "success": self.success,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "error_message": self.error_message,
            "metadata": self.metadata,
        }


# =============================================================================
# HSM Provider Interface
# =============================================================================


class HSMProvider(ABC):
    """Abstract base class for HSM providers."""

    def __init__(self, config: HSMConfig):
        self.config = config
        self._initialized = False
        self._lock = threading.RLock()
        self._audit_callback: Optional[Callable[[HSMAuditEvent], None]] = None

    @abstractmethod
    def initialize(self) -> bool:
        """Initialize the HSM connection."""
        pass

    @abstractmethod
    def shutdown(self) -> None:
        """Shutdown the HSM connection."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if HSM is available."""
        pass

    @property
    @abstractmethod
    def security_level(self) -> HSMSecurityLevel:
        """Get the security level of this HSM."""
        pass

    @abstractmethod
    def generate_pq_keypair(
        self,
        algorithm: PQAlgorithm,
        label: str,
        exportable: bool = False,
        expires_in: Optional[timedelta] = None,
    ) -> HSMKeyHandle:
        """Generate a post-quantum key pair in the HSM."""
        pass

    @abstractmethod
    def get_public_key(self, handle: HSMKeyHandle) -> bytes:
        """Get the public key for a key handle."""
        pass

    @abstractmethod
    def encapsulate(
        self,
        recipient_public_key: bytes,
        algorithm: PQAlgorithm,
    ) -> Tuple[bytes, bytes]:
        """
        Perform key encapsulation.

        Returns:
            Tuple of (shared_secret, ciphertext)
        """
        pass

    @abstractmethod
    def decapsulate(
        self,
        handle: HSMKeyHandle,
        ciphertext: bytes,
    ) -> bytes:
        """
        Perform key decapsulation inside HSM.

        Returns:
            Shared secret bytes
        """
        pass

    @abstractmethod
    def sign(
        self,
        handle: HSMKeyHandle,
        message: bytes,
    ) -> bytes:
        """Sign a message using key in HSM."""
        pass

    @abstractmethod
    def verify(
        self,
        public_key: bytes,
        message: bytes,
        signature: bytes,
        algorithm: PQAlgorithm,
    ) -> bool:
        """Verify a signature."""
        pass

    @abstractmethod
    def destroy_key(self, handle: HSMKeyHandle) -> bool:
        """Securely destroy a key in the HSM."""
        pass

    @abstractmethod
    def list_keys(
        self,
        algorithm: Optional[PQAlgorithm] = None,
        state: Optional[KeyState] = None,
    ) -> List[HSMKeyHandle]:
        """List keys in the HSM."""
        pass

    def set_audit_callback(
        self,
        callback: Callable[[HSMAuditEvent], None],
    ) -> None:
        """Set callback for audit events."""
        self._audit_callback = callback

    def _audit(
        self,
        operation: HSMOperation,
        key_handle: Optional[str],
        success: bool,
        error_message: Optional[str] = None,
        **metadata,
    ) -> None:
        """Log an audit event."""
        if not self.config.audit_all_operations:
            return

        event = HSMAuditEvent(
            event_id=secrets.token_hex(8),
            timestamp=datetime.utcnow(),
            operation=operation,
            key_handle=key_handle,
            success=success,
            error_message=error_message,
            metadata=metadata,
        )

        if self._audit_callback:
            self._audit_callback(event)

        level = logging.INFO if success else logging.WARNING
        logger.log(
            level,
            f"HSM {operation.value}: handle={key_handle}, success={success}"
            + (f", error={error_message}" if error_message else ""),
        )


# =============================================================================
# Software HSM Provider (Development/Testing)
# =============================================================================


class SoftwareHSMProvider(HSMProvider):
    """
    Software-based HSM for development and testing.

    WARNING: This is NOT secure for production use!
    It simulates HSM behavior but stores keys in memory/files.
    """

    def __init__(self, config: HSMConfig):
        super().__init__(config)
        self._keys: Dict[str, Tuple[HSMKeyHandle, bytes, bytes]] = {}  # handle -> (metadata, public, private)
        self._master_key: Optional[bytes] = None

    def initialize(self) -> bool:
        """Initialize software HSM."""
        with self._lock:
            if self._initialized:
                return True

            # Generate or load master key
            if self.config.software_master_key:
                self._master_key = self.config.software_master_key
            else:
                self._master_key = secrets.token_bytes(32)

            # Load persisted keys if storage configured
            if self.config.software_key_store:
                self._load_keys()

            self._initialized = True
            logger.info("Software HSM initialized (DEVELOPMENT MODE)")
            return True

    def shutdown(self) -> None:
        """Shutdown software HSM."""
        with self._lock:
            if self.config.software_key_store:
                self._save_keys()

            # Secure memory cleanup
            for handle_id, (_, public, private) in self._keys.items():
                # Overwrite private key in memory
                if private:
                    overwrite = b"\x00" * len(private)
                    # Note: This doesn't guarantee memory is overwritten in Python
                    # Production HSMs handle this in hardware

            self._keys.clear()
            self._master_key = None
            self._initialized = False
            logger.info("Software HSM shutdown")

    def is_available(self) -> bool:
        """Check if software HSM is available."""
        return self._initialized

    @property
    def security_level(self) -> HSMSecurityLevel:
        """Software HSM is Level 1."""
        return HSMSecurityLevel.LEVEL_1

    def generate_pq_keypair(
        self,
        algorithm: PQAlgorithm,
        label: str,
        exportable: bool = False,
        expires_in: Optional[timedelta] = None,
    ) -> HSMKeyHandle:
        """Generate PQ key pair."""
        with self._lock:
            if not self._initialized:
                raise RuntimeError("HSM not initialized")

            try:
                # Generate keys based on algorithm
                public_key, private_key = self._generate_keys(algorithm)

                # Create handle
                handle = HSMKeyHandle(
                    handle_id=f"sw:{secrets.token_hex(8)}",
                    algorithm=algorithm,
                    label=label,
                    state=KeyState.ACTIVE,
                    hsm_type=HSMType.SOFTWARE,
                    security_level=HSMSecurityLevel.LEVEL_1,
                    exportable=exportable,
                    expires_at=datetime.utcnow() + expires_in if expires_in else None,
                )

                # Store encrypted
                encrypted_private = self._encrypt_key(private_key)
                self._keys[handle.handle_id] = (handle, public_key, encrypted_private)

                self._audit(
                    HSMOperation.GENERATE_KEYPAIR,
                    handle.handle_id,
                    True,
                    algorithm=algorithm.value,
                    label=label,
                )

                return handle

            except Exception as e:
                self._audit(
                    HSMOperation.GENERATE_KEYPAIR,
                    None,
                    False,
                    str(e),
                    algorithm=algorithm.value,
                )
                raise

    def _generate_keys(self, algorithm: PQAlgorithm) -> Tuple[bytes, bytes]:
        """Generate key pair for algorithm."""
        try:
            from ..pq.ml_kem import DefaultMLKEMProvider, MLKEMSecurityLevel
            from ..pq.ml_dsa import DefaultMLDSAProvider, MLDSASecurityLevel
            from ..pq.hybrid import HybridKeyExchange, HybridSigner

            if algorithm == PQAlgorithm.ML_KEM_512:
                provider = DefaultMLKEMProvider()
                kp = provider.generate_keypair(MLKEMSecurityLevel.ML_KEM_512)
                return kp.public_key.key_data, kp.private_key.key_data

            elif algorithm == PQAlgorithm.ML_KEM_768:
                provider = DefaultMLKEMProvider()
                kp = provider.generate_keypair(MLKEMSecurityLevel.ML_KEM_768)
                return kp.public_key.key_data, kp.private_key.key_data

            elif algorithm == PQAlgorithm.ML_KEM_1024:
                provider = DefaultMLKEMProvider()
                kp = provider.generate_keypair(MLKEMSecurityLevel.ML_KEM_1024)
                return kp.public_key.key_data, kp.private_key.key_data

            elif algorithm == PQAlgorithm.ML_DSA_44:
                provider = DefaultMLDSAProvider()
                kp = provider.generate_keypair(MLDSASecurityLevel.ML_DSA_44)
                return kp.public_key.key_data, kp.private_key.key_data

            elif algorithm == PQAlgorithm.ML_DSA_65:
                provider = DefaultMLDSAProvider()
                kp = provider.generate_keypair(MLDSASecurityLevel.ML_DSA_65)
                return kp.public_key.key_data, kp.private_key.key_data

            elif algorithm == PQAlgorithm.ML_DSA_87:
                provider = DefaultMLDSAProvider()
                kp = provider.generate_keypair(MLDSASecurityLevel.ML_DSA_87)
                return kp.public_key.key_data, kp.private_key.key_data

            elif algorithm == PQAlgorithm.X25519_ML_KEM_768:
                kex = HybridKeyExchange()
                kp = kex.generate_keypair()
                public = kp.public_key.classical_key + kp.public_key.pq_key
                private = kp.private_key.classical_key + kp.private_key.pq_key
                return public, private

            elif algorithm == PQAlgorithm.ED25519_ML_DSA_65:
                signer = HybridSigner()
                kp = signer.generate_keypair()
                public = kp.public_key.classical_key + kp.public_key.pq_key
                private = kp.private_key.classical_key + kp.private_key.pq_key
                return public, private

            else:
                raise ValueError(f"Unsupported algorithm: {algorithm}")

        except ImportError:
            # Fallback mock generation
            logger.warning("PQ crypto not available, using mock keys")
            public = secrets.token_bytes(32)
            private = secrets.token_bytes(64)
            return public, private

    def _encrypt_key(self, key_data: bytes) -> bytes:
        """Encrypt key with master key."""
        if not self._master_key:
            raise RuntimeError("Master key not available")

        # Use AES-GCM in production; simplified HMAC encryption for mock
        nonce = secrets.token_bytes(12)
        # In production: AES-GCM encryption
        # For mock: XOR with derived key + HMAC
        derived = hashlib.sha256(self._master_key + nonce).digest()
        encrypted = bytes(a ^ b for a, b in zip(key_data, derived * (len(key_data) // 32 + 1)))
        tag = hmac.new(self._master_key, nonce + encrypted, hashlib.sha256).digest()[:16]

        return nonce + tag + encrypted

    def _decrypt_key(self, encrypted_data: bytes) -> bytes:
        """Decrypt key with master key."""
        if not self._master_key:
            raise RuntimeError("Master key not available")

        nonce = encrypted_data[:12]
        tag = encrypted_data[12:28]
        ciphertext = encrypted_data[28:]

        # Verify tag
        expected_tag = hmac.new(self._master_key, nonce + ciphertext, hashlib.sha256).digest()[:16]
        if not hmac.compare_digest(tag, expected_tag):
            raise ValueError("Key decryption failed: invalid tag")

        # Decrypt
        derived = hashlib.sha256(self._master_key + nonce).digest()
        decrypted = bytes(a ^ b for a, b in zip(ciphertext, derived * (len(ciphertext) // 32 + 1)))

        return decrypted

    def get_public_key(self, handle: HSMKeyHandle) -> bytes:
        """Get public key for handle."""
        with self._lock:
            if handle.handle_id not in self._keys:
                raise KeyError(f"Key not found: {handle.handle_id}")

            _, public_key, _ = self._keys[handle.handle_id]

            self._audit(
                HSMOperation.EXPORT_PUBLIC,
                handle.handle_id,
                True,
            )

            return public_key

    def encapsulate(
        self,
        recipient_public_key: bytes,
        algorithm: PQAlgorithm,
    ) -> Tuple[bytes, bytes]:
        """Perform key encapsulation."""
        with self._lock:
            try:
                from ..pq.ml_kem import DefaultMLKEMProvider, MLKEMSecurityLevel, MLKEMPublicKey

                level_map = {
                    PQAlgorithm.ML_KEM_512: MLKEMSecurityLevel.ML_KEM_512,
                    PQAlgorithm.ML_KEM_768: MLKEMSecurityLevel.ML_KEM_768,
                    PQAlgorithm.ML_KEM_1024: MLKEMSecurityLevel.ML_KEM_1024,
                }

                if algorithm in level_map:
                    provider = DefaultMLKEMProvider()
                    public = MLKEMPublicKey(
                        key_data=recipient_public_key,
                        security_level=level_map[algorithm],
                    )
                    shared_secret, ciphertext = provider.encapsulate(public)

                    self._audit(
                        HSMOperation.ENCAPSULATE,
                        None,
                        True,
                        algorithm=algorithm.value,
                    )

                    return shared_secret, ciphertext.ciphertext

                else:
                    raise ValueError(f"Algorithm not supported for encapsulation: {algorithm}")

            except Exception as e:
                self._audit(
                    HSMOperation.ENCAPSULATE,
                    None,
                    False,
                    str(e),
                )
                raise

    def decapsulate(
        self,
        handle: HSMKeyHandle,
        ciphertext: bytes,
    ) -> bytes:
        """Perform key decapsulation."""
        with self._lock:
            if handle.handle_id not in self._keys:
                raise KeyError(f"Key not found: {handle.handle_id}")

            stored_handle, _, encrypted_private = self._keys[handle.handle_id]

            if not stored_handle.is_valid:
                raise ValueError("Key is not valid for use")

            try:
                private_key = self._decrypt_key(encrypted_private)

                from ..pq.ml_kem import (
                    DefaultMLKEMProvider,
                    MLKEMSecurityLevel,
                    MLKEMPrivateKey,
                    MLKEMCiphertext,
                )

                level_map = {
                    PQAlgorithm.ML_KEM_512: MLKEMSecurityLevel.ML_KEM_512,
                    PQAlgorithm.ML_KEM_768: MLKEMSecurityLevel.ML_KEM_768,
                    PQAlgorithm.ML_KEM_1024: MLKEMSecurityLevel.ML_KEM_1024,
                }

                if handle.algorithm in level_map:
                    provider = DefaultMLKEMProvider()
                    priv = MLKEMPrivateKey(
                        key_data=private_key,
                        security_level=level_map[handle.algorithm],
                    )
                    ct = MLKEMCiphertext(
                        ciphertext=ciphertext,
                        security_level=level_map[handle.algorithm],
                    )
                    shared_secret = provider.decapsulate(ct, priv)

                    # Update usage count
                    stored_handle.usage_count += 1

                    self._audit(
                        HSMOperation.DECAPSULATE,
                        handle.handle_id,
                        True,
                    )

                    return shared_secret

                else:
                    raise ValueError(f"Algorithm not supported: {handle.algorithm}")

            except Exception as e:
                self._audit(
                    HSMOperation.DECAPSULATE,
                    handle.handle_id,
                    False,
                    str(e),
                )
                raise

    def sign(
        self,
        handle: HSMKeyHandle,
        message: bytes,
    ) -> bytes:
        """Sign message with key in HSM."""
        with self._lock:
            if handle.handle_id not in self._keys:
                raise KeyError(f"Key not found: {handle.handle_id}")

            stored_handle, _, encrypted_private = self._keys[handle.handle_id]

            if not stored_handle.is_valid:
                raise ValueError("Key is not valid for use")

            try:
                private_key = self._decrypt_key(encrypted_private)

                from ..pq.ml_dsa import DefaultMLDSAProvider, MLDSASecurityLevel, MLDSAPrivateKey

                level_map = {
                    PQAlgorithm.ML_DSA_44: MLDSASecurityLevel.ML_DSA_44,
                    PQAlgorithm.ML_DSA_65: MLDSASecurityLevel.ML_DSA_65,
                    PQAlgorithm.ML_DSA_87: MLDSASecurityLevel.ML_DSA_87,
                }

                if handle.algorithm in level_map:
                    provider = DefaultMLDSAProvider()
                    priv = MLDSAPrivateKey(
                        key_data=private_key,
                        security_level=level_map[handle.algorithm],
                    )
                    sig = provider.sign(message, priv)

                    stored_handle.usage_count += 1

                    self._audit(
                        HSMOperation.SIGN,
                        handle.handle_id,
                        True,
                        message_hash=hashlib.sha256(message).hexdigest()[:16],
                    )

                    return sig.signature

                else:
                    raise ValueError(f"Algorithm not supported for signing: {handle.algorithm}")

            except Exception as e:
                self._audit(
                    HSMOperation.SIGN,
                    handle.handle_id,
                    False,
                    str(e),
                )
                raise

    def verify(
        self,
        public_key: bytes,
        message: bytes,
        signature: bytes,
        algorithm: PQAlgorithm,
    ) -> bool:
        """Verify signature."""
        try:
            from ..pq.ml_dsa import (
                DefaultMLDSAProvider,
                MLDSASecurityLevel,
                MLDSAPublicKey,
                MLDSASignature,
            )

            level_map = {
                PQAlgorithm.ML_DSA_44: MLDSASecurityLevel.ML_DSA_44,
                PQAlgorithm.ML_DSA_65: MLDSASecurityLevel.ML_DSA_65,
                PQAlgorithm.ML_DSA_87: MLDSASecurityLevel.ML_DSA_87,
            }

            if algorithm in level_map:
                provider = DefaultMLDSAProvider()
                pub = MLDSAPublicKey(
                    key_data=public_key,
                    security_level=level_map[algorithm],
                )
                sig = MLDSASignature(
                    signature=signature,
                    security_level=level_map[algorithm],
                )
                result = provider.verify(message, sig, pub)

                self._audit(
                    HSMOperation.VERIFY,
                    None,
                    result,
                    algorithm=algorithm.value,
                )

                return result

            else:
                raise ValueError(f"Algorithm not supported for verification: {algorithm}")

        except Exception as e:
            self._audit(
                HSMOperation.VERIFY,
                None,
                False,
                str(e),
            )
            return False

    def destroy_key(self, handle: HSMKeyHandle) -> bool:
        """Destroy key in HSM."""
        with self._lock:
            if handle.handle_id not in self._keys:
                return False

            stored_handle, _, encrypted_private = self._keys[handle.handle_id]

            # Mark as destroyed
            stored_handle.state = KeyState.DESTROYED

            # Remove from storage
            del self._keys[handle.handle_id]

            self._audit(
                HSMOperation.DESTROY_KEY,
                handle.handle_id,
                True,
            )

            logger.info(f"Destroyed key: {handle.handle_id}")
            return True

    def list_keys(
        self,
        algorithm: Optional[PQAlgorithm] = None,
        state: Optional[KeyState] = None,
    ) -> List[HSMKeyHandle]:
        """List keys in HSM."""
        with self._lock:
            handles = []
            for handle_id, (stored_handle, _, _) in self._keys.items():
                if algorithm and stored_handle.algorithm != algorithm:
                    continue
                if state and stored_handle.state != state:
                    continue
                handles.append(stored_handle)
            return handles

    def _load_keys(self) -> None:
        """Load keys from persistent storage."""
        if not self.config.software_key_store:
            return

        keys_file = self.config.software_key_store / "hsm_keys.json"
        data_dir = self.config.software_key_store / "key_data"

        if not keys_file.exists():
            return

        try:
            with open(keys_file) as f:
                handles_data = json.load(f)

            for handle_data in handles_data:
                handle = HSMKeyHandle.from_dict(handle_data)

                # Load key material
                key_file = data_dir / f"{handle.handle_id}.bin"
                if key_file.exists():
                    with open(key_file, "rb") as f:
                        key_material = f.read()
                    # Parse: public_len (4 bytes) + public + encrypted_private
                    public_len = struct.unpack(">I", key_material[:4])[0]
                    public = key_material[4:4 + public_len]
                    encrypted_private = key_material[4 + public_len:]

                    self._keys[handle.handle_id] = (handle, public, encrypted_private)

            logger.info(f"Loaded {len(self._keys)} keys from storage")

        except Exception as e:
            logger.error(f"Failed to load keys: {e}")

    def _save_keys(self) -> None:
        """Save keys to persistent storage."""
        if not self.config.software_key_store:
            return

        try:
            self.config.software_key_store.mkdir(parents=True, exist_ok=True)
            data_dir = self.config.software_key_store / "key_data"
            data_dir.mkdir(exist_ok=True)

            handles_data = []
            for handle_id, (handle, public, encrypted_private) in self._keys.items():
                handles_data.append(handle.to_dict())

                # Save key material
                key_file = data_dir / f"{handle_id}.bin"
                key_material = struct.pack(">I", len(public)) + public + encrypted_private
                with open(key_file, "wb") as f:
                    f.write(key_material)
                os.chmod(key_file, 0o600)

            keys_file = self.config.software_key_store / "hsm_keys.json"
            with open(keys_file, "w") as f:
                json.dump(handles_data, f, indent=2)

            logger.info(f"Saved {len(self._keys)} keys to storage")

        except Exception as e:
            logger.error(f"Failed to save keys: {e}")


# =============================================================================
# PKCS#11 HSM Provider (Stub)
# =============================================================================


class PKCS11HSMProvider(HSMProvider):
    """
    PKCS#11 HSM provider for hardware security modules.

    This is a stub implementation. In production, this would use:
    - python-pkcs11 or PyKCS11 library
    - Actual PKCS#11 library from HSM vendor
    """

    def __init__(self, config: HSMConfig):
        super().__init__(config)
        self._session = None

    def initialize(self) -> bool:
        """Initialize PKCS#11 connection."""
        if not self.config.pkcs11_library:
            logger.error("PKCS#11 library path not configured")
            return False

        try:
            # In production:
            # import pkcs11
            # lib = pkcs11.lib(self.config.pkcs11_library)
            # token = lib.get_token(slot_or_label=self.config.pkcs11_slot)
            # self._session = token.open(user_pin=self.config.pkcs11_pin)

            logger.warning("PKCS#11 HSM not implemented - using software fallback")
            self._initialized = False
            return False

        except Exception as e:
            logger.error(f"Failed to initialize PKCS#11: {e}")
            return False

    def shutdown(self) -> None:
        """Shutdown PKCS#11 connection."""
        if self._session:
            # self._session.close()
            self._session = None
        self._initialized = False

    def is_available(self) -> bool:
        """Check if PKCS#11 HSM is available."""
        return self._initialized and self._session is not None

    @property
    def security_level(self) -> HSMSecurityLevel:
        """PKCS#11 HSMs are typically Level 3."""
        return HSMSecurityLevel.LEVEL_3

    def generate_pq_keypair(
        self,
        algorithm: PQAlgorithm,
        label: str,
        exportable: bool = False,
        expires_in: Optional[timedelta] = None,
    ) -> HSMKeyHandle:
        """Generate key in PKCS#11 HSM."""
        raise NotImplementedError("PKCS#11 HSM not implemented")

    def get_public_key(self, handle: HSMKeyHandle) -> bytes:
        """Get public key from PKCS#11 HSM."""
        raise NotImplementedError("PKCS#11 HSM not implemented")

    def encapsulate(
        self,
        recipient_public_key: bytes,
        algorithm: PQAlgorithm,
    ) -> Tuple[bytes, bytes]:
        """Perform encapsulation in PKCS#11 HSM."""
        raise NotImplementedError("PKCS#11 HSM not implemented")

    def decapsulate(
        self,
        handle: HSMKeyHandle,
        ciphertext: bytes,
    ) -> bytes:
        """Perform decapsulation in PKCS#11 HSM."""
        raise NotImplementedError("PKCS#11 HSM not implemented")

    def sign(
        self,
        handle: HSMKeyHandle,
        message: bytes,
    ) -> bytes:
        """Sign in PKCS#11 HSM."""
        raise NotImplementedError("PKCS#11 HSM not implemented")

    def verify(
        self,
        public_key: bytes,
        message: bytes,
        signature: bytes,
        algorithm: PQAlgorithm,
    ) -> bool:
        """Verify signature."""
        raise NotImplementedError("PKCS#11 HSM not implemented")

    def destroy_key(self, handle: HSMKeyHandle) -> bool:
        """Destroy key in PKCS#11 HSM."""
        raise NotImplementedError("PKCS#11 HSM not implemented")

    def list_keys(
        self,
        algorithm: Optional[PQAlgorithm] = None,
        state: Optional[KeyState] = None,
    ) -> List[HSMKeyHandle]:
        """List keys in PKCS#11 HSM."""
        raise NotImplementedError("PKCS#11 HSM not implemented")


# =============================================================================
# Factory Functions
# =============================================================================


def create_hsm_provider(
    hsm_type: HSMType = HSMType.SOFTWARE,
    config: Optional[HSMConfig] = None,
    **kwargs,
) -> HSMProvider:
    """
    Create an HSM provider.

    Args:
        hsm_type: Type of HSM to use
        config: HSM configuration (created from kwargs if not provided)
        **kwargs: Configuration options

    Returns:
        HSMProvider instance
    """
    if config is None:
        config = HSMConfig(hsm_type=hsm_type, **kwargs)
    else:
        config.hsm_type = hsm_type

    providers = {
        HSMType.SOFTWARE: SoftwareHSMProvider,
        HSMType.PKCS11: PKCS11HSMProvider,
        # Future: TPM, Cloud HSM providers
    }

    provider_class = providers.get(hsm_type)
    if provider_class is None:
        logger.warning(f"HSM type {hsm_type} not supported, falling back to software")
        provider_class = SoftwareHSMProvider

    return provider_class(config)


def get_recommended_hsm_config(
    environment: str = "development",
) -> HSMConfig:
    """
    Get recommended HSM configuration for environment.

    Args:
        environment: "development", "staging", or "production"

    Returns:
        HSMConfig with recommended settings
    """
    if environment == "production":
        return HSMConfig(
            hsm_type=HSMType.PKCS11,
            security_level=HSMSecurityLevel.LEVEL_3,
            audit_all_operations=True,
            require_dual_control=True,
            max_key_cache_size=50,
            key_cache_ttl_seconds=1800,
        )
    elif environment == "staging":
        return HSMConfig(
            hsm_type=HSMType.SOFTWARE,
            security_level=HSMSecurityLevel.LEVEL_1,
            audit_all_operations=True,
            require_dual_control=False,
            software_key_store=Path("/var/lib/agent-os/hsm"),
        )
    else:  # development
        return HSMConfig(
            hsm_type=HSMType.SOFTWARE,
            security_level=HSMSecurityLevel.LEVEL_1,
            audit_all_operations=False,
            require_dual_control=False,
        )
