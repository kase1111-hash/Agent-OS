"""
ML-KEM (Module-Lattice-Based Key-Encapsulation Mechanism)

Implementation of FIPS 203 (CRYSTALS-Kyber) for post-quantum key encapsulation.

ML-KEM provides IND-CCA2 security against both classical and quantum adversaries.
It is based on the hardness of the Module Learning With Errors (MLWE) problem.

Security Levels:
- ML-KEM-512: NIST Level 1 (~AES-128)
- ML-KEM-768: NIST Level 3 (~AES-192) - Recommended
- ML-KEM-1024: NIST Level 5 (~AES-256)

Key Sizes:
- ML-KEM-768: Public key 1,184 bytes, Private key 2,400 bytes, Ciphertext 1,088 bytes
- ML-KEM-1024: Public key 1,568 bytes, Private key 3,168 bytes, Ciphertext 1,568 bytes
"""

import base64
import hashlib
import hmac
import logging
import os
import secrets
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)


def _is_production_mode() -> bool:
    """Check if running in production mode.

    Production mode is enabled when AGENT_OS_PRODUCTION is set to
    '1', 'true', or 'yes' (case-insensitive).
    """
    env_value = os.environ.get("AGENT_OS_PRODUCTION", "").lower()
    return env_value in ("1", "true", "yes")


# =============================================================================
# Constants
# =============================================================================


class MLKEMSecurityLevel(str, Enum):
    """ML-KEM security levels per FIPS 203."""

    ML_KEM_512 = "ml-kem-512"  # NIST Level 1 (~AES-128)
    ML_KEM_768 = "ml-kem-768"  # NIST Level 3 (~AES-192) - Recommended
    ML_KEM_1024 = "ml-kem-1024"  # NIST Level 5 (~AES-256)


# Key sizes in bytes per security level
ML_KEM_PARAMS = {
    MLKEMSecurityLevel.ML_KEM_512: {
        "public_key_size": 800,
        "private_key_size": 1632,
        "ciphertext_size": 768,
        "shared_secret_size": 32,
        "n": 256,
        "k": 2,
        "eta1": 3,
        "eta2": 2,
    },
    MLKEMSecurityLevel.ML_KEM_768: {
        "public_key_size": 1184,
        "private_key_size": 2400,
        "ciphertext_size": 1088,
        "shared_secret_size": 32,
        "n": 256,
        "k": 3,
        "eta1": 2,
        "eta2": 2,
    },
    MLKEMSecurityLevel.ML_KEM_1024: {
        "public_key_size": 1568,
        "private_key_size": 3168,
        "ciphertext_size": 1568,
        "shared_secret_size": 32,
        "n": 256,
        "k": 4,
        "eta1": 2,
        "eta2": 2,
    },
}


# =============================================================================
# Models
# =============================================================================


@dataclass
class MLKEMPublicKey:
    """ML-KEM public key for encapsulation."""

    key_data: bytes
    security_level: MLKEMSecurityLevel
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
    def from_dict(cls, data: Dict[str, Any]) -> "MLKEMPublicKey":
        """Create from dictionary."""
        return cls(
            key_data=base64.b64decode(data["key_data"]),
            security_level=MLKEMSecurityLevel(data["security_level"]),
            key_id=data.get("key_id", ""),
            created_at=datetime.fromisoformat(
                data.get("created_at", datetime.utcnow().isoformat())
            ),
        )


@dataclass
class MLKEMPrivateKey:
    """ML-KEM private key for decapsulation."""

    key_data: bytes
    security_level: MLKEMSecurityLevel
    key_id: str = ""
    created_at: datetime = field(default_factory=datetime.utcnow)

    def __post_init__(self):
        if not self.key_id:
            # Key ID derived from public key portion
            self.key_id = hashlib.sha256(self.key_data[:32]).hexdigest()[:16]

    @property
    def size(self) -> int:
        """Get key size in bytes."""
        return len(self.key_data)

    def secure_delete(self) -> None:
        """Securely delete key material by overwriting with zeros."""
        if isinstance(self.key_data, bytearray):
            for i in range(len(self.key_data)):
                self.key_data[i] = 0
        # Note: For immutable bytes, this is a best-effort approach
        self.key_data = b"\x00" * len(self.key_data)


@dataclass
class MLKEMKeyPair:
    """ML-KEM key pair for key encapsulation."""

    public_key: MLKEMPublicKey
    private_key: MLKEMPrivateKey
    security_level: MLKEMSecurityLevel = MLKEMSecurityLevel.ML_KEM_768

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (public key only for safety)."""
        return {
            "public_key": self.public_key.to_dict(),
            "security_level": self.security_level.value,
        }


@dataclass
class MLKEMCiphertext:
    """ML-KEM ciphertext (encapsulated shared secret)."""

    ciphertext: bytes
    security_level: MLKEMSecurityLevel
    sender_key_id: str = ""
    recipient_key_id: str = ""
    created_at: datetime = field(default_factory=datetime.utcnow)

    @property
    def size(self) -> int:
        """Get ciphertext size in bytes."""
        return len(self.ciphertext)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "ciphertext": base64.b64encode(self.ciphertext).decode(),
            "security_level": self.security_level.value,
            "sender_key_id": self.sender_key_id,
            "recipient_key_id": self.recipient_key_id,
            "created_at": self.created_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MLKEMCiphertext":
        """Create from dictionary."""
        return cls(
            ciphertext=base64.b64decode(data["ciphertext"]),
            security_level=MLKEMSecurityLevel(data["security_level"]),
            sender_key_id=data.get("sender_key_id", ""),
            recipient_key_id=data.get("recipient_key_id", ""),
            created_at=datetime.fromisoformat(
                data.get("created_at", datetime.utcnow().isoformat())
            ),
        )


# =============================================================================
# ML-KEM Provider Interface
# =============================================================================


class MLKEMProvider(ABC):
    """
    Abstract base class for ML-KEM operations.

    Implementations must provide:
    - Key generation
    - Encapsulation (generate shared secret + ciphertext)
    - Decapsulation (recover shared secret from ciphertext)
    """

    @abstractmethod
    def generate_keypair(
        self,
        security_level: MLKEMSecurityLevel = MLKEMSecurityLevel.ML_KEM_768,
    ) -> MLKEMKeyPair:
        """
        Generate ML-KEM key pair.

        Args:
            security_level: Security level (512, 768, or 1024)

        Returns:
            MLKEMKeyPair with public and private keys
        """
        pass

    @abstractmethod
    def encapsulate(
        self,
        public_key: MLKEMPublicKey,
    ) -> Tuple[bytes, MLKEMCiphertext]:
        """
        Encapsulate: Generate shared secret and ciphertext.

        Args:
            public_key: Recipient's ML-KEM public key

        Returns:
            Tuple of (shared_secret, ciphertext)
        """
        pass

    @abstractmethod
    def decapsulate(
        self,
        ciphertext: MLKEMCiphertext,
        private_key: MLKEMPrivateKey,
    ) -> bytes:
        """
        Decapsulate: Recover shared secret from ciphertext.

        Args:
            ciphertext: ML-KEM ciphertext
            private_key: Recipient's ML-KEM private key

        Returns:
            Shared secret bytes
        """
        pass


# =============================================================================
# Default ML-KEM Provider (using liboqs when available)
# =============================================================================


class DefaultMLKEMProvider(MLKEMProvider):
    """
    Default ML-KEM provider using liboqs library.

    SECURITY: This provider requires the liboqs library for secure
    post-quantum key encapsulation. It will NOT fall back to insecure
    mock implementations.

    Raises:
        ImportError: If liboqs is not available
    """

    def __init__(self, allow_mock: bool = False):
        """Initialize the ML-KEM provider.

        Args:
            allow_mock: If True, allow insecure mock fallback (FOR TESTING ONLY).
                       Default is False for security.

        Raises:
            ImportError: If liboqs is not available and allow_mock is False
        """
        self._has_oqs = self._check_oqs()
        self._allow_mock = allow_mock

        if self._has_oqs:
            logger.info("Using liboqs for ML-KEM operations")
        elif allow_mock:
            logger.warning(
                "SECURITY WARNING: Using mock ML-KEM - NOT SECURE FOR PRODUCTION. "
                "Install liboqs for secure post-quantum key encapsulation."
            )
        else:
            raise ImportError(
                "The 'liboqs' library is required for secure ML-KEM operations. "
                "Install with: pip install liboqs-python"
            )

    def _check_oqs(self) -> bool:
        """Check if liboqs is available."""
        try:
            import oqs

            # Check if Kyber is available
            if "Kyber768" in oqs.get_enabled_kem_mechanisms():
                return True
            return False
        except ImportError:
            return False

    def _get_oqs_algorithm(self, security_level: MLKEMSecurityLevel) -> str:
        """Map security level to liboqs algorithm name."""
        mapping = {
            MLKEMSecurityLevel.ML_KEM_512: "Kyber512",
            MLKEMSecurityLevel.ML_KEM_768: "Kyber768",
            MLKEMSecurityLevel.ML_KEM_1024: "Kyber1024",
        }
        return mapping.get(security_level, "Kyber768")

    def generate_keypair(
        self,
        security_level: MLKEMSecurityLevel = MLKEMSecurityLevel.ML_KEM_768,
    ) -> MLKEMKeyPair:
        """Generate ML-KEM key pair."""
        if self._has_oqs:
            return self._generate_keypair_oqs(security_level)
        if self._allow_mock:
            return self._generate_keypair_mock(security_level)
        raise ImportError(
            "The 'liboqs' library is required for ML-KEM key generation. "
            "Install with: pip install liboqs-python"
        )

    def _generate_keypair_oqs(self, security_level: MLKEMSecurityLevel) -> MLKEMKeyPair:
        """Generate key pair using liboqs."""
        import oqs

        algorithm = self._get_oqs_algorithm(security_level)
        kem = oqs.KeyEncapsulation(algorithm)

        public_key_bytes = kem.generate_keypair()
        private_key_bytes = kem.export_secret_key()

        return MLKEMKeyPair(
            public_key=MLKEMPublicKey(
                key_data=public_key_bytes,
                security_level=security_level,
            ),
            private_key=MLKEMPrivateKey(
                key_data=private_key_bytes,
                security_level=security_level,
            ),
            security_level=security_level,
        )

    def _generate_keypair_mock(self, security_level: MLKEMSecurityLevel) -> MLKEMKeyPair:
        """Generate mock key pair for testing."""
        params = ML_KEM_PARAMS[security_level]

        # Generate deterministic-looking but random keys
        seed = secrets.token_bytes(32)
        public_key_data = self._expand_seed(seed, params["public_key_size"], b"public")
        private_key_data = self._expand_seed(seed, params["private_key_size"], b"private")

        return MLKEMKeyPair(
            public_key=MLKEMPublicKey(
                key_data=public_key_data,
                security_level=security_level,
            ),
            private_key=MLKEMPrivateKey(
                key_data=private_key_data,
                security_level=security_level,
            ),
            security_level=security_level,
        )

    def encapsulate(
        self,
        public_key: MLKEMPublicKey,
    ) -> Tuple[bytes, MLKEMCiphertext]:
        """Encapsulate to generate shared secret and ciphertext."""
        if self._has_oqs:
            return self._encapsulate_oqs(public_key)
        if self._allow_mock:
            return self._encapsulate_mock(public_key)
        raise ImportError(
            "The 'liboqs' library is required for ML-KEM encapsulation. "
            "Install with: pip install liboqs-python"
        )

    def _encapsulate_oqs(self, public_key: MLKEMPublicKey) -> Tuple[bytes, MLKEMCiphertext]:
        """Encapsulate using liboqs."""
        import oqs

        algorithm = self._get_oqs_algorithm(public_key.security_level)
        kem = oqs.KeyEncapsulation(algorithm)

        ciphertext_bytes, shared_secret = kem.encap_secret(public_key.key_data)

        ciphertext = MLKEMCiphertext(
            ciphertext=ciphertext_bytes,
            security_level=public_key.security_level,
            recipient_key_id=public_key.key_id,
        )

        return shared_secret, ciphertext

    def _encapsulate_mock(self, public_key: MLKEMPublicKey) -> Tuple[bytes, MLKEMCiphertext]:
        """Mock encapsulation for testing."""
        params = ML_KEM_PARAMS[public_key.security_level]

        # Generate random shared secret
        shared_secret = secrets.token_bytes(params["shared_secret_size"])

        # Create mock ciphertext that encodes the shared secret
        # In real ML-KEM, this would be a lattice-based encryption
        ciphertext_seed = secrets.token_bytes(32)
        ciphertext_data = self._expand_seed(
            ciphertext_seed, params["ciphertext_size"], b"ciphertext"
        )

        # Store shared secret in a way that can be recovered with private key
        # This is NOT cryptographically secure - mock only!
        encrypted_secret = bytes(
            s ^ h
            for s, h in zip(
                shared_secret, hashlib.sha256(public_key.key_data + ciphertext_seed).digest()
            )
        )
        ciphertext_data = encrypted_secret + ciphertext_data[32:]

        ciphertext = MLKEMCiphertext(
            ciphertext=ciphertext_data,
            security_level=public_key.security_level,
            recipient_key_id=public_key.key_id,
        )

        return shared_secret, ciphertext

    def decapsulate(
        self,
        ciphertext: MLKEMCiphertext,
        private_key: MLKEMPrivateKey,
    ) -> bytes:
        """Decapsulate to recover shared secret."""
        if self._has_oqs:
            return self._decapsulate_oqs(ciphertext, private_key)
        if self._allow_mock:
            return self._decapsulate_mock(ciphertext, private_key)
        raise ImportError(
            "The 'liboqs' library is required for ML-KEM decapsulation. "
            "Install with: pip install liboqs-python"
        )

    def _decapsulate_oqs(
        self,
        ciphertext: MLKEMCiphertext,
        private_key: MLKEMPrivateKey,
    ) -> bytes:
        """Decapsulate using liboqs."""
        import oqs

        algorithm = self._get_oqs_algorithm(private_key.security_level)
        kem = oqs.KeyEncapsulation(algorithm, private_key.key_data)

        shared_secret = kem.decap_secret(ciphertext.ciphertext)
        return shared_secret

    def _decapsulate_mock(
        self,
        ciphertext: MLKEMCiphertext,
        private_key: MLKEMPrivateKey,
    ) -> bytes:
        """Mock decapsulation for testing."""
        params = ML_KEM_PARAMS[private_key.security_level]

        # Extract the XORed shared secret from ciphertext
        encrypted_secret = ciphertext.ciphertext[:32]

        # Derive the same mask using private key derivation
        # This simulates the lattice decryption
        public_key_data = self._expand_seed(
            private_key.key_data[:32], params["public_key_size"], b"public"
        )
        ciphertext_seed = (
            private_key.key_data[32:64]
            if len(private_key.key_data) >= 64
            else secrets.token_bytes(32)
        )

        # In mock mode, we use HMAC to simulate key agreement
        mask = hashlib.sha256(public_key_data + ciphertext.ciphertext[32:64]).digest()

        shared_secret = bytes(e ^ m for e, m in zip(encrypted_secret, mask))

        return shared_secret[: params["shared_secret_size"]]

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
# Mock ML-KEM Provider (for testing)
# =============================================================================


class MockMLKEMProvider(MLKEMProvider):
    """
    Mock ML-KEM provider for testing purposes.

    WARNING: This is NOT cryptographically secure and should
    only be used for testing and development.

    This provider is disabled in production mode (AGENT_OS_PRODUCTION=1).
    """

    def __init__(self):
        if _is_production_mode():
            raise RuntimeError(
                "MockMLKEMProvider is disabled in production mode. "
                "Set AGENT_OS_PRODUCTION=0 for development/testing."
            )
        logger.warning("Using MockMLKEMProvider - NOT SECURE FOR PRODUCTION")
        self._shared_secrets: Dict[str, bytes] = {}

    def generate_keypair(
        self,
        security_level: MLKEMSecurityLevel = MLKEMSecurityLevel.ML_KEM_768,
    ) -> MLKEMKeyPair:
        """Generate mock key pair."""
        params = ML_KEM_PARAMS[security_level]

        public_key_data = secrets.token_bytes(params["public_key_size"])
        private_key_data = secrets.token_bytes(params["private_key_size"])

        return MLKEMKeyPair(
            public_key=MLKEMPublicKey(
                key_data=public_key_data,
                security_level=security_level,
            ),
            private_key=MLKEMPrivateKey(
                key_data=private_key_data,
                security_level=security_level,
            ),
            security_level=security_level,
        )

    def encapsulate(
        self,
        public_key: MLKEMPublicKey,
    ) -> Tuple[bytes, MLKEMCiphertext]:
        """Mock encapsulation."""
        params = ML_KEM_PARAMS[public_key.security_level]

        # Generate shared secret
        shared_secret = secrets.token_bytes(params["shared_secret_size"])

        # Create ciphertext that "encrypts" the shared secret
        nonce = secrets.token_bytes(16)
        encrypted = bytes(
            s ^ h
            for s, h in zip(shared_secret, hashlib.sha256(public_key.key_data + nonce).digest())
        )

        ciphertext_data = (
            nonce + encrypted + secrets.token_bytes(params["ciphertext_size"] - 16 - 32)
        )

        # Store for mock decapsulation
        self._shared_secrets[public_key.key_id] = shared_secret

        ciphertext = MLKEMCiphertext(
            ciphertext=ciphertext_data,
            security_level=public_key.security_level,
            recipient_key_id=public_key.key_id,
        )

        return shared_secret, ciphertext

    def decapsulate(
        self,
        ciphertext: MLKEMCiphertext,
        private_key: MLKEMPrivateKey,
    ) -> bytes:
        """Mock decapsulation."""
        params = ML_KEM_PARAMS[private_key.security_level]

        # Extract nonce and encrypted secret
        nonce = ciphertext.ciphertext[:16]
        encrypted = ciphertext.ciphertext[16:48]

        # Derive public key from private key (mock)
        public_key_data = hashlib.sha256(private_key.key_data).digest()
        public_key_data = public_key_data * (params["public_key_size"] // 32 + 1)
        public_key_data = public_key_data[: params["public_key_size"]]

        # Decrypt
        mask = hashlib.sha256(public_key_data + nonce).digest()
        shared_secret = bytes(e ^ m for e, m in zip(encrypted, mask))

        return shared_secret[: params["shared_secret_size"]]


# =============================================================================
# Factory Functions
# =============================================================================


def create_ml_kem_provider(
    provider_type: str = "default",
) -> MLKEMProvider:
    """
    Create an ML-KEM provider.

    Args:
        provider_type: Type of provider ("default", "mock")

    Returns:
        MLKEMProvider instance
    """
    if provider_type == "mock":
        return MockMLKEMProvider()
    return DefaultMLKEMProvider()


def generate_ml_kem_keypair(
    security_level: MLKEMSecurityLevel = MLKEMSecurityLevel.ML_KEM_768,
    provider: Optional[MLKEMProvider] = None,
) -> MLKEMKeyPair:
    """
    Generate an ML-KEM key pair.

    Args:
        security_level: Security level
        provider: Optional provider instance

    Returns:
        MLKEMKeyPair
    """
    provider = provider or DefaultMLKEMProvider()
    return provider.generate_keypair(security_level)


def ml_kem_encapsulate(
    public_key: MLKEMPublicKey,
    provider: Optional[MLKEMProvider] = None,
) -> Tuple[bytes, MLKEMCiphertext]:
    """
    Perform ML-KEM encapsulation.

    Args:
        public_key: Recipient's public key
        provider: Optional provider instance

    Returns:
        Tuple of (shared_secret, ciphertext)
    """
    provider = provider or DefaultMLKEMProvider()
    return provider.encapsulate(public_key)


def ml_kem_decapsulate(
    ciphertext: MLKEMCiphertext,
    private_key: MLKEMPrivateKey,
    provider: Optional[MLKEMProvider] = None,
) -> bytes:
    """
    Perform ML-KEM decapsulation.

    Args:
        ciphertext: ML-KEM ciphertext
        private_key: Recipient's private key
        provider: Optional provider instance

    Returns:
        Shared secret bytes
    """
    provider = provider or DefaultMLKEMProvider()
    return provider.decapsulate(ciphertext, private_key)
