"""
ML-DSA (Module-Lattice-Based Digital Signature Algorithm)

Implementation of FIPS 204 (CRYSTALS-Dilithium) for post-quantum digital signatures.

ML-DSA provides EUF-CMA security against both classical and quantum adversaries.
It is based on the hardness of the Module Learning With Errors (MLWE) and
Module Short Integer Solution (MSIS) problems.

Security Levels:
- ML-DSA-44: NIST Level 2 (~AES-128)
- ML-DSA-65: NIST Level 3 (~AES-192) - Recommended
- ML-DSA-87: NIST Level 5 (~AES-256)

Key/Signature Sizes:
- ML-DSA-65: Public key 1,952 bytes, Private key 4,032 bytes, Signature 3,309 bytes
- ML-DSA-87: Public key 2,592 bytes, Private key 4,896 bytes, Signature 4,627 bytes
"""

import base64
import hashlib
import hmac
import logging
import secrets
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)


# =============================================================================
# Constants
# =============================================================================


class MLDSASecurityLevel(str, Enum):
    """ML-DSA security levels per FIPS 204."""

    ML_DSA_44 = "ml-dsa-44"  # NIST Level 2 (~AES-128)
    ML_DSA_65 = "ml-dsa-65"  # NIST Level 3 (~AES-192) - Recommended
    ML_DSA_87 = "ml-dsa-87"  # NIST Level 5 (~AES-256)


# Key/signature sizes in bytes per security level
ML_DSA_PARAMS = {
    MLDSASecurityLevel.ML_DSA_44: {
        "public_key_size": 1312,
        "private_key_size": 2560,
        "signature_size": 2420,
        "k": 4,
        "l": 4,
        "eta": 2,
        "tau": 39,
        "gamma1": 131072,
        "gamma2": 95232,
    },
    MLDSASecurityLevel.ML_DSA_65: {
        "public_key_size": 1952,
        "private_key_size": 4032,
        "signature_size": 3309,
        "k": 6,
        "l": 5,
        "eta": 4,
        "tau": 49,
        "gamma1": 524288,
        "gamma2": 261888,
    },
    MLDSASecurityLevel.ML_DSA_87: {
        "public_key_size": 2592,
        "private_key_size": 4896,
        "signature_size": 4627,
        "k": 8,
        "l": 7,
        "eta": 2,
        "tau": 60,
        "gamma1": 524288,
        "gamma2": 261888,
    },
}


# =============================================================================
# Models
# =============================================================================


@dataclass
class MLDSAPublicKey:
    """ML-DSA public key for signature verification."""

    key_data: bytes
    security_level: MLDSASecurityLevel
    key_id: str = ""
    created_at: datetime = field(default_factory=datetime.utcnow)

    def __post_init__(self):
        if not self.key_id:
            self.key_id = self._generate_key_id()

    def _generate_key_id(self) -> str:
        """Generate unique key ID from key data."""
        return hashlib.sha256(self.key_data).hexdigest()[:16]

    @property
    def size(self) -> int:
        """Get key size in bytes."""
        return len(self.key_data)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "key_data": base64.b64encode(self.key_data).decode(),
            "security_level": self.security_level.value,
            "key_id": self.key_id,
            "created_at": self.created_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MLDSAPublicKey":
        """Create from dictionary."""
        return cls(
            key_data=base64.b64decode(data["key_data"]),
            security_level=MLDSASecurityLevel(data["security_level"]),
            key_id=data.get("key_id", ""),
            created_at=datetime.fromisoformat(
                data.get("created_at", datetime.utcnow().isoformat())
            ),
        )


@dataclass
class MLDSAPrivateKey:
    """ML-DSA private key for signing."""

    key_data: bytes
    security_level: MLDSASecurityLevel
    key_id: str = ""
    created_at: datetime = field(default_factory=datetime.utcnow)

    def __post_init__(self):
        if not self.key_id:
            self.key_id = hashlib.sha256(self.key_data[:32]).hexdigest()[:16]

    @property
    def size(self) -> int:
        """Get key size in bytes."""
        return len(self.key_data)

    def secure_delete(self) -> None:
        """Securely delete key material."""
        if isinstance(self.key_data, bytearray):
            for i in range(len(self.key_data)):
                self.key_data[i] = 0
        self.key_data = b"\x00" * len(self.key_data)


@dataclass
class MLDSAKeyPair:
    """ML-DSA key pair for digital signatures."""

    public_key: MLDSAPublicKey
    private_key: MLDSAPrivateKey
    security_level: MLDSASecurityLevel = MLDSASecurityLevel.ML_DSA_65

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (public key only for safety)."""
        return {
            "public_key": self.public_key.to_dict(),
            "security_level": self.security_level.value,
        }


@dataclass
class MLDSASignature:
    """ML-DSA digital signature."""

    signature: bytes
    security_level: MLDSASecurityLevel
    signer_key_id: str = ""
    message_hash: str = ""  # SHA-256 of signed message
    created_at: datetime = field(default_factory=datetime.utcnow)

    @property
    def size(self) -> int:
        """Get signature size in bytes."""
        return len(self.signature)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "signature": base64.b64encode(self.signature).decode(),
            "security_level": self.security_level.value,
            "signer_key_id": self.signer_key_id,
            "message_hash": self.message_hash,
            "created_at": self.created_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MLDSASignature":
        """Create from dictionary."""
        return cls(
            signature=base64.b64decode(data["signature"]),
            security_level=MLDSASecurityLevel(data["security_level"]),
            signer_key_id=data.get("signer_key_id", ""),
            message_hash=data.get("message_hash", ""),
            created_at=datetime.fromisoformat(
                data.get("created_at", datetime.utcnow().isoformat())
            ),
        )


# =============================================================================
# ML-DSA Provider Interface
# =============================================================================


class MLDSAProvider(ABC):
    """
    Abstract base class for ML-DSA operations.

    Implementations must provide:
    - Key generation
    - Signing
    - Verification
    """

    @abstractmethod
    def generate_keypair(
        self,
        security_level: MLDSASecurityLevel = MLDSASecurityLevel.ML_DSA_65,
    ) -> MLDSAKeyPair:
        """
        Generate ML-DSA key pair.

        Args:
            security_level: Security level (44, 65, or 87)

        Returns:
            MLDSAKeyPair with public and private keys
        """
        pass

    @abstractmethod
    def sign(
        self,
        message: bytes,
        private_key: MLDSAPrivateKey,
    ) -> MLDSASignature:
        """
        Sign a message.

        Args:
            message: Message to sign
            private_key: Signer's private key

        Returns:
            MLDSASignature
        """
        pass

    @abstractmethod
    def verify(
        self,
        message: bytes,
        signature: MLDSASignature,
        public_key: MLDSAPublicKey,
    ) -> bool:
        """
        Verify a signature.

        Args:
            message: Original message
            signature: Signature to verify
            public_key: Signer's public key

        Returns:
            True if signature is valid
        """
        pass


# =============================================================================
# Default ML-DSA Provider (using liboqs when available)
# =============================================================================


class DefaultMLDSAProvider(MLDSAProvider):
    """
    Default ML-DSA provider using liboqs library.

    SECURITY: This provider requires the liboqs library for secure
    post-quantum digital signatures. It will NOT fall back to insecure
    mock implementations.

    Raises:
        ImportError: If liboqs is not available
    """

    def __init__(self, allow_mock: bool = False):
        """Initialize the ML-DSA provider.

        Args:
            allow_mock: If True, allow insecure mock fallback (FOR TESTING ONLY).
                       Default is False for security.

        Raises:
            ImportError: If liboqs is not available and allow_mock is False
        """
        self._has_oqs = self._check_oqs()
        self._allow_mock = allow_mock

        if self._has_oqs:
            logger.info("Using liboqs for ML-DSA operations")
        elif allow_mock:
            logger.warning(
                "SECURITY WARNING: Using mock ML-DSA - NOT SECURE FOR PRODUCTION. "
                "Install liboqs for secure post-quantum signatures."
            )
        else:
            raise ImportError(
                "The 'liboqs' library is required for secure ML-DSA operations. "
                "Install with: pip install liboqs-python"
            )

    def _check_oqs(self) -> bool:
        """Check if liboqs is available."""
        try:
            import oqs

            # Check if Dilithium is available
            if "Dilithium3" in oqs.get_enabled_sig_mechanisms():
                return True
            return False
        except ImportError:
            return False

    def _get_oqs_algorithm(self, security_level: MLDSASecurityLevel) -> str:
        """Map security level to liboqs algorithm name."""
        mapping = {
            MLDSASecurityLevel.ML_DSA_44: "Dilithium2",
            MLDSASecurityLevel.ML_DSA_65: "Dilithium3",
            MLDSASecurityLevel.ML_DSA_87: "Dilithium5",
        }
        return mapping.get(security_level, "Dilithium3")

    def generate_keypair(
        self,
        security_level: MLDSASecurityLevel = MLDSASecurityLevel.ML_DSA_65,
    ) -> MLDSAKeyPair:
        """Generate ML-DSA key pair."""
        if self._has_oqs:
            return self._generate_keypair_oqs(security_level)
        if self._allow_mock:
            return self._generate_keypair_mock(security_level)
        raise ImportError(
            "The 'liboqs' library is required for ML-DSA key generation. "
            "Install with: pip install liboqs-python"
        )

    def _generate_keypair_oqs(self, security_level: MLDSASecurityLevel) -> MLDSAKeyPair:
        """Generate key pair using liboqs."""
        import oqs

        algorithm = self._get_oqs_algorithm(security_level)
        sig = oqs.Signature(algorithm)

        public_key_bytes = sig.generate_keypair()
        private_key_bytes = sig.export_secret_key()

        return MLDSAKeyPair(
            public_key=MLDSAPublicKey(
                key_data=public_key_bytes,
                security_level=security_level,
            ),
            private_key=MLDSAPrivateKey(
                key_data=private_key_bytes,
                security_level=security_level,
            ),
            security_level=security_level,
        )

    def _generate_keypair_mock(self, security_level: MLDSASecurityLevel) -> MLDSAKeyPair:
        """Generate mock key pair for testing."""
        params = ML_DSA_PARAMS[security_level]

        seed = secrets.token_bytes(32)
        public_key_data = self._expand_seed(seed, params["public_key_size"], b"public")
        private_key_data = self._expand_seed(seed, params["private_key_size"], b"private")

        return MLDSAKeyPair(
            public_key=MLDSAPublicKey(
                key_data=public_key_data,
                security_level=security_level,
            ),
            private_key=MLDSAPrivateKey(
                key_data=private_key_data,
                security_level=security_level,
            ),
            security_level=security_level,
        )

    def sign(
        self,
        message: bytes,
        private_key: MLDSAPrivateKey,
    ) -> MLDSASignature:
        """Sign a message."""
        if self._has_oqs:
            return self._sign_oqs(message, private_key)
        if self._allow_mock:
            return self._sign_mock(message, private_key)
        raise ImportError(
            "The 'liboqs' library is required for ML-DSA signing. "
            "Install with: pip install liboqs-python"
        )

    def _sign_oqs(
        self,
        message: bytes,
        private_key: MLDSAPrivateKey,
    ) -> MLDSASignature:
        """Sign using liboqs."""
        import oqs

        algorithm = self._get_oqs_algorithm(private_key.security_level)
        sig = oqs.Signature(algorithm, private_key.key_data)

        signature_bytes = sig.sign(message)
        message_hash = hashlib.sha256(message).hexdigest()

        return MLDSASignature(
            signature=signature_bytes,
            security_level=private_key.security_level,
            signer_key_id=private_key.key_id,
            message_hash=message_hash,
        )

    def _sign_mock(
        self,
        message: bytes,
        private_key: MLDSAPrivateKey,
    ) -> MLDSASignature:
        """Mock signing for testing."""
        params = ML_DSA_PARAMS[private_key.security_level]

        # Create deterministic signature using HMAC
        message_hash = hashlib.sha256(message).hexdigest()

        # Generate signature components
        sig_data = hmac.new(
            private_key.key_data[:32],
            message,
            hashlib.sha512,
        ).digest()

        # Expand to full signature size
        signature_bytes = self._expand_seed(sig_data, params["signature_size"], message[:32])

        return MLDSASignature(
            signature=signature_bytes,
            security_level=private_key.security_level,
            signer_key_id=private_key.key_id,
            message_hash=message_hash,
        )

    def verify(
        self,
        message: bytes,
        signature: MLDSASignature,
        public_key: MLDSAPublicKey,
    ) -> bool:
        """Verify a signature."""
        if self._has_oqs:
            return self._verify_oqs(message, signature, public_key)
        if self._allow_mock:
            return self._verify_mock(message, signature, public_key)
        raise ImportError(
            "The 'liboqs' library is required for ML-DSA verification. "
            "Install with: pip install liboqs-python"
        )

    def _verify_oqs(
        self,
        message: bytes,
        signature: MLDSASignature,
        public_key: MLDSAPublicKey,
    ) -> bool:
        """Verify using liboqs."""
        import oqs

        algorithm = self._get_oqs_algorithm(public_key.security_level)
        sig = oqs.Signature(algorithm)

        try:
            return sig.verify(message, signature.signature, public_key.key_data)
        except Exception:
            return False

    def _verify_mock(
        self,
        message: bytes,
        signature: MLDSASignature,
        public_key: MLDSAPublicKey,
    ) -> bool:
        """Mock verification for testing."""
        # Check message hash
        message_hash = hashlib.sha256(message).hexdigest()
        if signature.message_hash and signature.message_hash != message_hash:
            return False

        # In mock mode, derive private key from public key (insecure!)
        # and verify the HMAC matches
        mock_private_seed = hashlib.sha256(public_key.key_data[:32]).digest()

        expected_sig_data = hmac.new(
            mock_private_seed[:32],
            message,
            hashlib.sha512,
        ).digest()

        # Check first 64 bytes match
        return hmac.compare_digest(
            signature.signature[:64],
            self._expand_seed(
                expected_sig_data,
                ML_DSA_PARAMS[signature.security_level]["signature_size"],
                message[:32],
            )[:64],
        )

    def _expand_seed(self, seed: bytes, length: int, domain: bytes) -> bytes:
        """Expand seed to desired length using SHAKE-like construction."""
        output = b""
        counter = 0
        while len(output) < length:
            block = hashlib.sha256(seed + domain + counter.to_bytes(4, "big")).digest()
            output += block
            counter += 1
        return output[:length]


# =============================================================================
# Mock ML-DSA Provider (for testing)
# =============================================================================


class MockMLDSAProvider(MLDSAProvider):
    """
    Mock ML-DSA provider for testing purposes.

    WARNING: This is NOT cryptographically secure and should
    only be used for testing and development.
    """

    def __init__(self):
        logger.warning("Using MockMLDSAProvider - NOT SECURE FOR PRODUCTION")
        self._keypairs: Dict[str, MLDSAKeyPair] = {}

    def generate_keypair(
        self,
        security_level: MLDSASecurityLevel = MLDSASecurityLevel.ML_DSA_65,
    ) -> MLDSAKeyPair:
        """Generate mock key pair."""
        params = ML_DSA_PARAMS[security_level]

        # Generate random keys
        seed = secrets.token_bytes(32)
        public_key_data = secrets.token_bytes(params["public_key_size"])
        private_key_data = seed + secrets.token_bytes(params["private_key_size"] - 32)

        keypair = MLDSAKeyPair(
            public_key=MLDSAPublicKey(
                key_data=public_key_data,
                security_level=security_level,
            ),
            private_key=MLDSAPrivateKey(
                key_data=private_key_data,
                security_level=security_level,
            ),
            security_level=security_level,
        )

        # Store for verification
        self._keypairs[keypair.public_key.key_id] = keypair

        return keypair

    def sign(
        self,
        message: bytes,
        private_key: MLDSAPrivateKey,
    ) -> MLDSASignature:
        """Mock signing."""
        params = ML_DSA_PARAMS[private_key.security_level]

        # Use HMAC for deterministic signature
        sig_data = hmac.new(
            private_key.key_data[:32],
            message,
            hashlib.sha512,
        ).digest()

        # Pad to full signature size
        signature_bytes = sig_data + secrets.token_bytes(params["signature_size"] - len(sig_data))

        return MLDSASignature(
            signature=signature_bytes,
            security_level=private_key.security_level,
            signer_key_id=private_key.key_id,
            message_hash=hashlib.sha256(message).hexdigest(),
        )

    def verify(
        self,
        message: bytes,
        signature: MLDSASignature,
        public_key: MLDSAPublicKey,
    ) -> bool:
        """Mock verification."""
        # Check if we have the keypair
        keypair = self._keypairs.get(public_key.key_id)
        if not keypair:
            return False

        # Verify using stored private key
        expected_sig = hmac.new(
            keypair.private_key.key_data[:32],
            message,
            hashlib.sha512,
        ).digest()

        return hmac.compare_digest(signature.signature[:64], expected_sig)


# =============================================================================
# Factory Functions
# =============================================================================


def create_ml_dsa_provider(
    provider_type: str = "default",
) -> MLDSAProvider:
    """
    Create an ML-DSA provider.

    Args:
        provider_type: Type of provider ("default", "mock")

    Returns:
        MLDSAProvider instance
    """
    if provider_type == "mock":
        return MockMLDSAProvider()
    return DefaultMLDSAProvider()


def generate_ml_dsa_keypair(
    security_level: MLDSASecurityLevel = MLDSASecurityLevel.ML_DSA_65,
    provider: Optional[MLDSAProvider] = None,
) -> MLDSAKeyPair:
    """
    Generate an ML-DSA key pair.

    Args:
        security_level: Security level
        provider: Optional provider instance

    Returns:
        MLDSAKeyPair
    """
    provider = provider or DefaultMLDSAProvider()
    return provider.generate_keypair(security_level)


def ml_dsa_sign(
    message: bytes,
    private_key: MLDSAPrivateKey,
    provider: Optional[MLDSAProvider] = None,
) -> MLDSASignature:
    """
    Sign a message with ML-DSA.

    Args:
        message: Message to sign
        private_key: Signer's private key
        provider: Optional provider instance

    Returns:
        MLDSASignature
    """
    provider = provider or DefaultMLDSAProvider()
    return provider.sign(message, private_key)


def ml_dsa_verify(
    message: bytes,
    signature: MLDSASignature,
    public_key: MLDSAPublicKey,
    provider: Optional[MLDSAProvider] = None,
) -> bool:
    """
    Verify an ML-DSA signature.

    Args:
        message: Original message
        signature: Signature to verify
        public_key: Signer's public key
        provider: Optional provider instance

    Returns:
        True if valid
    """
    provider = provider or DefaultMLDSAProvider()
    return provider.verify(message, signature, public_key)
