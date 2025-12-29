"""
End-to-End Encryption

Provides cryptographic operations for secure federation communication:
- Message encryption/decryption
- Key exchange
- Session key management
"""

import base64
import hashlib
import hmac
import logging
import os
import secrets
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

from .identity import KeyPair, KeyType, PrivateKey, PublicKey

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


class CipherSuite(str, Enum):
    """Supported cipher suites."""

    AES_256_GCM = "aes-256-gcm"
    CHACHA20_POLY1305 = "chacha20-poly1305"


class KeyExchangeMethod(str, Enum):
    """Key exchange methods."""

    X25519 = "x25519"
    ECDH_P256 = "ecdh-p256"


# =============================================================================
# Models
# =============================================================================


@dataclass
class SessionKey:
    """Session encryption key."""

    key_id: str
    key_data: bytes
    peer_id: str
    cipher_suite: CipherSuite
    created_at: datetime = field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None
    message_count: int = 0

    def __post_init__(self):
        if not self.expires_at:
            self.expires_at = self.created_at + timedelta(hours=24)

    @property
    def is_valid(self) -> bool:
        """Check if key is still valid."""
        if not self.expires_at:
            return True
        return datetime.utcnow() < self.expires_at

    @property
    def is_expired(self) -> bool:
        """Check if key has expired."""
        return not self.is_valid

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (excluding key data)."""
        return {
            "key_id": self.key_id,
            "peer_id": self.peer_id,
            "cipher_suite": self.cipher_suite.value,
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "message_count": self.message_count,
        }


@dataclass
class EncryptedMessage:
    """Encrypted message envelope."""

    ciphertext: bytes
    nonce: bytes
    key_id: str
    cipher_suite: CipherSuite
    sender_id: str
    recipient_id: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    auth_tag: Optional[bytes] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "ciphertext": base64.b64encode(self.ciphertext).decode(),
            "nonce": base64.b64encode(self.nonce).decode(),
            "key_id": self.key_id,
            "cipher_suite": self.cipher_suite.value,
            "sender_id": self.sender_id,
            "recipient_id": self.recipient_id,
            "timestamp": self.timestamp.isoformat(),
            "auth_tag": base64.b64encode(self.auth_tag).decode() if self.auth_tag else None,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EncryptedMessage":
        """Create from dictionary."""
        return cls(
            ciphertext=base64.b64decode(data["ciphertext"]),
            nonce=base64.b64decode(data["nonce"]),
            key_id=data["key_id"],
            cipher_suite=CipherSuite(data["cipher_suite"]),
            sender_id=data["sender_id"],
            recipient_id=data["recipient_id"],
            timestamp=datetime.fromisoformat(data.get("timestamp", datetime.utcnow().isoformat())),
            auth_tag=base64.b64decode(data["auth_tag"]) if data.get("auth_tag") else None,
            metadata=data.get("metadata", {}),
        )


@dataclass
class KeyExchangeData:
    """Data for key exchange."""

    public_key: bytes
    key_id: str
    method: KeyExchangeMethod
    signature: Optional[bytes] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "public_key": base64.b64encode(self.public_key).decode(),
            "key_id": self.key_id,
            "method": self.method.value,
            "signature": base64.b64encode(self.signature).decode() if self.signature else None,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "KeyExchangeData":
        """Create from dictionary."""
        return cls(
            public_key=base64.b64decode(data["public_key"]),
            key_id=data["key_id"],
            method=KeyExchangeMethod(data["method"]),
            signature=base64.b64decode(data["signature"]) if data.get("signature") else None,
        )


# =============================================================================
# Crypto Provider Interface
# =============================================================================


class CryptoProvider(ABC):
    """
    Abstract base class for cryptographic operations.
    """

    @abstractmethod
    def encrypt(
        self,
        plaintext: bytes,
        key: SessionKey,
        associated_data: Optional[bytes] = None,
    ) -> Tuple[bytes, bytes, Optional[bytes]]:
        """
        Encrypt data.

        Args:
            plaintext: Data to encrypt
            key: Session key
            associated_data: Additional authenticated data

        Returns:
            Tuple of (ciphertext, nonce, auth_tag)
        """
        pass

    @abstractmethod
    def decrypt(
        self,
        ciphertext: bytes,
        nonce: bytes,
        key: SessionKey,
        auth_tag: Optional[bytes] = None,
        associated_data: Optional[bytes] = None,
    ) -> bytes:
        """
        Decrypt data.

        Args:
            ciphertext: Encrypted data
            nonce: Encryption nonce
            key: Session key
            auth_tag: Authentication tag
            associated_data: Additional authenticated data

        Returns:
            Decrypted plaintext
        """
        pass

    @abstractmethod
    def generate_key_pair(
        self,
        method: KeyExchangeMethod = KeyExchangeMethod.X25519,
    ) -> Tuple[bytes, bytes]:
        """
        Generate key exchange key pair.

        Returns:
            Tuple of (public_key, private_key)
        """
        pass

    @abstractmethod
    def derive_shared_secret(
        self,
        private_key: bytes,
        peer_public_key: bytes,
        method: KeyExchangeMethod = KeyExchangeMethod.X25519,
    ) -> bytes:
        """
        Derive shared secret from key exchange.

        Returns:
            Shared secret bytes
        """
        pass

    @abstractmethod
    def derive_session_key(
        self,
        shared_secret: bytes,
        salt: bytes,
        info: bytes,
        key_length: int = 32,
    ) -> bytes:
        """
        Derive session key from shared secret.

        Returns:
            Session key bytes
        """
        pass


# =============================================================================
# Default Crypto Provider
# =============================================================================


class DefaultCryptoProvider(CryptoProvider):
    """
    Default cryptographic provider using cryptography library.

    Requires the 'cryptography' library - no insecure fallbacks.
    """

    def __init__(self, cipher_suite: CipherSuite = CipherSuite.AES_256_GCM):
        self.cipher_suite = cipher_suite
        self._check_crypto()  # Will raise if not available

    def _check_crypto(self) -> None:
        """Verify cryptography library is available. Raises if not."""
        try:
            import cryptography
        except ImportError:
            raise RuntimeError(
                "The 'cryptography' library is required for federation crypto. "
                "Install it with: pip install cryptography"
            )

    def encrypt(
        self,
        plaintext: bytes,
        key: SessionKey,
        associated_data: Optional[bytes] = None,
    ) -> Tuple[bytes, bytes, Optional[bytes]]:
        """Encrypt data using AES-256-GCM."""
        if self.cipher_suite != CipherSuite.AES_256_GCM:
            raise ValueError(f"Unsupported cipher suite: {self.cipher_suite}")
        return self._encrypt_aes_gcm(plaintext, key.key_data, associated_data)

    def decrypt(
        self,
        ciphertext: bytes,
        nonce: bytes,
        key: SessionKey,
        auth_tag: Optional[bytes] = None,
        associated_data: Optional[bytes] = None,
    ) -> bytes:
        """Decrypt data using AES-256-GCM."""
        if self.cipher_suite != CipherSuite.AES_256_GCM:
            raise ValueError(f"Unsupported cipher suite: {self.cipher_suite}")
        return self._decrypt_aes_gcm(ciphertext, nonce, key.key_data, auth_tag, associated_data)

    def _encrypt_aes_gcm(
        self,
        plaintext: bytes,
        key: bytes,
        associated_data: Optional[bytes],
    ) -> Tuple[bytes, bytes, bytes]:
        """Encrypt using AES-GCM."""
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM

        nonce = secrets.token_bytes(12)
        aesgcm = AESGCM(key[:32])

        ciphertext = aesgcm.encrypt(nonce, plaintext, associated_data)

        # AES-GCM includes tag in ciphertext
        return ciphertext[:-16], nonce, ciphertext[-16:]

    def _decrypt_aes_gcm(
        self,
        ciphertext: bytes,
        nonce: bytes,
        key: bytes,
        auth_tag: Optional[bytes],
        associated_data: Optional[bytes],
    ) -> bytes:
        """Decrypt using AES-GCM."""
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM

        aesgcm = AESGCM(key[:32])

        # Combine ciphertext and tag
        full_ciphertext = ciphertext + (auth_tag or b"")

        return aesgcm.decrypt(nonce, full_ciphertext, associated_data)

    def generate_key_pair(
        self,
        method: KeyExchangeMethod = KeyExchangeMethod.X25519,
    ) -> Tuple[bytes, bytes]:
        """Generate key exchange key pair using X25519."""
        if method != KeyExchangeMethod.X25519:
            raise ValueError(f"Unsupported key exchange method: {method}")

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

    def derive_shared_secret(
        self,
        private_key: bytes,
        peer_public_key: bytes,
        method: KeyExchangeMethod = KeyExchangeMethod.X25519,
    ) -> bytes:
        """Derive shared secret using X25519 key exchange."""
        if method != KeyExchangeMethod.X25519:
            raise ValueError(f"Unsupported key exchange method: {method}")

        from cryptography.hazmat.primitives.asymmetric.x25519 import (
            X25519PrivateKey,
            X25519PublicKey,
        )

        priv = X25519PrivateKey.from_private_bytes(private_key)
        pub = X25519PublicKey.from_public_bytes(peer_public_key)

        return priv.exchange(pub)

    def derive_session_key(
        self,
        shared_secret: bytes,
        salt: bytes,
        info: bytes,
        key_length: int = 32,
    ) -> bytes:
        """Derive session key using HKDF."""
        from cryptography.hazmat.primitives.kdf.hkdf import HKDF
        from cryptography.hazmat.primitives import hashes

        hkdf = HKDF(
            algorithm=hashes.SHA256(),
            length=key_length,
            salt=salt,
            info=info,
        )
        return hkdf.derive(shared_secret)


# =============================================================================
# Mock Crypto Provider
# =============================================================================


class MockCryptoProvider(CryptoProvider):
    """
    Mock crypto provider for testing.

    WARNING: This provider is NOT cryptographically secure.
    It is disabled in production mode (AGENT_OS_PRODUCTION=1).
    """

    def __init__(self):
        if _is_production_mode():
            raise RuntimeError(
                "MockCryptoProvider is disabled in production mode. "
                "Set AGENT_OS_PRODUCTION=0 for development/testing."
            )
        logger.warning("Using MockCryptoProvider - NOT SECURE FOR PRODUCTION")

    def encrypt(
        self,
        plaintext: bytes,
        key: SessionKey,
        associated_data: Optional[bytes] = None,
    ) -> Tuple[bytes, bytes, Optional[bytes]]:
        """Mock encrypt (simple base64 encoding)."""
        nonce = secrets.token_bytes(12)
        # Just prepend a marker and base64 for "encryption"
        ciphertext = b"MOCK:" + base64.b64encode(plaintext)
        auth_tag = hashlib.sha256(ciphertext + key.key_data).digest()[:16]
        return ciphertext, nonce, auth_tag

    def decrypt(
        self,
        ciphertext: bytes,
        nonce: bytes,
        key: SessionKey,
        auth_tag: Optional[bytes] = None,
        associated_data: Optional[bytes] = None,
    ) -> bytes:
        """Mock decrypt."""
        if ciphertext.startswith(b"MOCK:"):
            return base64.b64decode(ciphertext[5:])
        raise ValueError("Invalid mock ciphertext")

    def generate_key_pair(
        self,
        method: KeyExchangeMethod = KeyExchangeMethod.X25519,
    ) -> Tuple[bytes, bytes]:
        """Generate mock key pair."""
        private_key = secrets.token_bytes(32)
        public_key = hashlib.sha256(private_key).digest()
        return public_key, private_key

    def derive_shared_secret(
        self,
        private_key: bytes,
        peer_public_key: bytes,
        method: KeyExchangeMethod = KeyExchangeMethod.X25519,
    ) -> bytes:
        """Mock shared secret derivation."""
        return hashlib.sha256(private_key + peer_public_key).digest()

    def derive_session_key(
        self,
        shared_secret: bytes,
        salt: bytes,
        info: bytes,
        key_length: int = 32,
    ) -> bytes:
        """Mock session key derivation."""
        return hashlib.sha256(shared_secret + salt + info).digest()[:key_length]


# =============================================================================
# Session Manager
# =============================================================================


class SessionManager:
    """
    Manages encryption sessions with peers.
    """

    def __init__(
        self,
        node_id: str,
        crypto_provider: Optional[CryptoProvider] = None,
    ):
        self.node_id = node_id
        self.crypto = crypto_provider or DefaultCryptoProvider()

        # Session keys by peer
        self._sessions: Dict[str, SessionKey] = {}

        # Ephemeral key pairs for key exchange
        self._ephemeral_keys: Dict[str, Tuple[bytes, bytes]] = {}

    def create_session(
        self,
        peer_id: str,
        peer_public_key: bytes,
        method: KeyExchangeMethod = KeyExchangeMethod.X25519,
    ) -> SessionKey:
        """
        Create a new session with a peer.

        Args:
            peer_id: Peer node ID
            peer_public_key: Peer's public key for key exchange
            method: Key exchange method

        Returns:
            New session key
        """
        # Generate ephemeral key pair
        our_public, our_private = self.crypto.generate_key_pair(method)

        # Store ephemeral keys
        self._ephemeral_keys[peer_id] = (our_public, our_private)

        # Derive shared secret
        shared_secret = self.crypto.derive_shared_secret(
            our_private,
            peer_public_key,
            method,
        )

        # Derive session key
        salt = secrets.token_bytes(32)
        info = f"federation:{self.node_id}:{peer_id}".encode()
        key_data = self.crypto.derive_session_key(shared_secret, salt, info)

        # Create session
        session = SessionKey(
            key_id=secrets.token_hex(8),
            key_data=key_data,
            peer_id=peer_id,
            cipher_suite=CipherSuite.AES_256_GCM,
        )

        self._sessions[peer_id] = session
        logger.info(f"Created session with peer: {peer_id}")

        return session

    def get_session(self, peer_id: str) -> Optional[SessionKey]:
        """Get session key for a peer."""
        session = self._sessions.get(peer_id)

        if session and session.is_expired:
            logger.warning(f"Session expired for peer: {peer_id}")
            del self._sessions[peer_id]
            return None

        return session

    def get_ephemeral_public_key(self, peer_id: str) -> Optional[bytes]:
        """Get ephemeral public key for key exchange."""
        keys = self._ephemeral_keys.get(peer_id)
        return keys[0] if keys else None

    def remove_session(self, peer_id: str) -> None:
        """Remove session with a peer."""
        self._sessions.pop(peer_id, None)
        self._ephemeral_keys.pop(peer_id, None)
        logger.info(f"Removed session with peer: {peer_id}")

    def encrypt_for_peer(
        self,
        peer_id: str,
        plaintext: bytes,
        associated_data: Optional[bytes] = None,
    ) -> Optional[EncryptedMessage]:
        """
        Encrypt a message for a peer.

        Args:
            peer_id: Recipient peer ID
            plaintext: Message to encrypt
            associated_data: Additional authenticated data

        Returns:
            EncryptedMessage or None if no session
        """
        session = self.get_session(peer_id)
        if not session:
            logger.warning(f"No session for peer: {peer_id}")
            return None

        ciphertext, nonce, auth_tag = self.crypto.encrypt(
            plaintext,
            session,
            associated_data,
        )

        session.message_count += 1

        return EncryptedMessage(
            ciphertext=ciphertext,
            nonce=nonce,
            key_id=session.key_id,
            cipher_suite=session.cipher_suite,
            sender_id=self.node_id,
            recipient_id=peer_id,
            auth_tag=auth_tag,
        )

    def decrypt_from_peer(
        self,
        message: EncryptedMessage,
        associated_data: Optional[bytes] = None,
    ) -> Optional[bytes]:
        """
        Decrypt a message from a peer.

        Args:
            message: Encrypted message
            associated_data: Additional authenticated data

        Returns:
            Decrypted plaintext or None
        """
        session = self.get_session(message.sender_id)
        if not session:
            logger.warning(f"No session for peer: {message.sender_id}")
            return None

        if session.key_id != message.key_id:
            logger.warning(f"Key ID mismatch for peer: {message.sender_id}")
            return None

        try:
            plaintext = self.crypto.decrypt(
                message.ciphertext,
                message.nonce,
                session,
                message.auth_tag,
                associated_data,
            )
            session.message_count += 1
            return plaintext

        except Exception as e:
            logger.error(f"Decryption failed: {e}")
            return None

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


def create_crypto_provider(
    provider_type: str = "default",
    cipher_suite: CipherSuite = CipherSuite.AES_256_GCM,
) -> CryptoProvider:
    """
    Create a crypto provider.

    Args:
        provider_type: Type of provider ("default", "mock")
        cipher_suite: Cipher suite to use

    Returns:
        CryptoProvider instance
    """
    if provider_type == "mock":
        return MockCryptoProvider()

    return DefaultCryptoProvider(cipher_suite)


def encrypt_message(
    plaintext: bytes,
    session: SessionKey,
    sender_id: str,
    recipient_id: str,
    provider: Optional[CryptoProvider] = None,
) -> EncryptedMessage:
    """
    Encrypt a message.

    Args:
        plaintext: Message to encrypt
        session: Session key
        sender_id: Sender node ID
        recipient_id: Recipient node ID
        provider: Crypto provider

    Returns:
        EncryptedMessage
    """
    provider = provider or DefaultCryptoProvider()

    ciphertext, nonce, auth_tag = provider.encrypt(plaintext, session)

    return EncryptedMessage(
        ciphertext=ciphertext,
        nonce=nonce,
        key_id=session.key_id,
        cipher_suite=session.cipher_suite,
        sender_id=sender_id,
        recipient_id=recipient_id,
        auth_tag=auth_tag,
    )


def decrypt_message(
    message: EncryptedMessage,
    session: SessionKey,
    provider: Optional[CryptoProvider] = None,
) -> bytes:
    """
    Decrypt a message.

    Args:
        message: Encrypted message
        session: Session key
        provider: Crypto provider

    Returns:
        Decrypted plaintext
    """
    provider = provider or DefaultCryptoProvider()

    return provider.decrypt(
        message.ciphertext,
        message.nonce,
        session,
        message.auth_tag,
    )
