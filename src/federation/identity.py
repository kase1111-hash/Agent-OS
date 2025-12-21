"""
Identity Verification and Key Management

Provides identity management for federation nodes including:
- Key pair generation and storage
- Certificate creation and verification
- Identity attestation
"""

import base64
import hashlib
import hmac
import json
import logging
import os
import secrets
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)


# =============================================================================
# Models
# =============================================================================


class KeyType(str, Enum):
    """Cryptographic key types."""

    ED25519 = "ed25519"
    RSA_2048 = "rsa-2048"
    RSA_4096 = "rsa-4096"
    X25519 = "x25519"  # For key exchange


class IdentityStatus(str, Enum):
    """Identity verification status."""

    UNVERIFIED = "unverified"
    SELF_SIGNED = "self_signed"
    VERIFIED = "verified"
    TRUSTED = "trusted"
    REVOKED = "revoked"


@dataclass
class PublicKey:
    """Public key for identity verification."""

    key_type: KeyType
    key_data: bytes
    key_id: str = ""
    created_at: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.key_id:
            self.key_id = self._generate_key_id()

    def _generate_key_id(self) -> str:
        """Generate a unique key ID from key data."""
        return hashlib.sha256(self.key_data).hexdigest()[:16]

    def to_pem(self) -> str:
        """Export to PEM format."""
        b64 = base64.b64encode(self.key_data).decode()
        return f"-----BEGIN PUBLIC KEY-----\n{b64}\n-----END PUBLIC KEY-----"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "key_type": self.key_type.value,
            "key_data": base64.b64encode(self.key_data).decode(),
            "key_id": self.key_id,
            "created_at": self.created_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PublicKey":
        """Create from dictionary."""
        return cls(
            key_type=KeyType(data["key_type"]),
            key_data=base64.b64decode(data["key_data"]),
            key_id=data.get("key_id", ""),
            created_at=datetime.fromisoformat(data.get("created_at", datetime.utcnow().isoformat())),
        )


@dataclass
class PrivateKey:
    """Private key for signing and decryption."""

    key_type: KeyType
    key_data: bytes
    key_id: str = ""
    created_at: datetime = field(default_factory=datetime.utcnow)

    def __post_init__(self):
        if not self.key_id:
            # Generate ID from corresponding public key
            self.key_id = hashlib.sha256(self.key_data[:32]).hexdigest()[:16]

    def to_pem(self) -> str:
        """Export to PEM format (encrypted recommended)."""
        b64 = base64.b64encode(self.key_data).decode()
        return f"-----BEGIN PRIVATE KEY-----\n{b64}\n-----END PRIVATE KEY-----"


@dataclass
class KeyPair:
    """Public/private key pair."""

    public_key: PublicKey
    private_key: PrivateKey
    key_type: KeyType = KeyType.ED25519

    @classmethod
    def generate(cls, key_type: KeyType = KeyType.ED25519) -> "KeyPair":
        """Generate a new key pair."""
        try:
            if key_type == KeyType.ED25519:
                from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
                from cryptography.hazmat.primitives import serialization

                private_key_obj = Ed25519PrivateKey.generate()
                public_key_obj = private_key_obj.public_key()

                private_bytes = private_key_obj.private_bytes(
                    encoding=serialization.Encoding.Raw,
                    format=serialization.PrivateFormat.Raw,
                    encryption_algorithm=serialization.NoEncryption(),
                )
                public_bytes = public_key_obj.public_bytes(
                    encoding=serialization.Encoding.Raw,
                    format=serialization.PublicFormat.Raw,
                )

                return cls(
                    public_key=PublicKey(key_type=key_type, key_data=public_bytes),
                    private_key=PrivateKey(key_type=key_type, key_data=private_bytes),
                    key_type=key_type,
                )

            elif key_type in (KeyType.RSA_2048, KeyType.RSA_4096):
                from cryptography.hazmat.primitives.asymmetric import rsa
                from cryptography.hazmat.primitives import serialization

                key_size = 2048 if key_type == KeyType.RSA_2048 else 4096
                private_key_obj = rsa.generate_private_key(
                    public_exponent=65537,
                    key_size=key_size,
                )

                private_bytes = private_key_obj.private_bytes(
                    encoding=serialization.Encoding.DER,
                    format=serialization.PrivateFormat.PKCS8,
                    encryption_algorithm=serialization.NoEncryption(),
                )
                public_bytes = private_key_obj.public_key().public_bytes(
                    encoding=serialization.Encoding.DER,
                    format=serialization.PublicFormat.SubjectPublicKeyInfo,
                )

                return cls(
                    public_key=PublicKey(key_type=key_type, key_data=public_bytes),
                    private_key=PrivateKey(key_type=key_type, key_data=private_bytes),
                    key_type=key_type,
                )

        except ImportError:
            logger.warning("cryptography library not installed, using mock keys")

        # Fall back to mock key generation
        return cls._generate_mock(key_type)

    @classmethod
    def _generate_mock(cls, key_type: KeyType) -> "KeyPair":
        """Generate mock key pair for testing."""
        private_data = secrets.token_bytes(32)
        public_data = hashlib.sha256(private_data).digest()

        return cls(
            public_key=PublicKey(key_type=key_type, key_data=public_data),
            private_key=PrivateKey(key_type=key_type, key_data=private_data),
            key_type=key_type,
        )


@dataclass
class Certificate:
    """Identity certificate for verification."""

    subject_id: str
    issuer_id: str
    public_key: PublicKey
    valid_from: datetime
    valid_until: datetime
    signature: bytes = b""
    serial_number: str = ""
    extensions: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.serial_number:
            self.serial_number = secrets.token_hex(16)

    @property
    def is_valid(self) -> bool:
        """Check if certificate is currently valid."""
        now = datetime.utcnow()
        return self.valid_from <= now <= self.valid_until

    @property
    def is_self_signed(self) -> bool:
        """Check if certificate is self-signed."""
        return self.subject_id == self.issuer_id

    @property
    def days_until_expiry(self) -> int:
        """Get days until certificate expires."""
        delta = self.valid_until - datetime.utcnow()
        return max(0, delta.days)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "subject_id": self.subject_id,
            "issuer_id": self.issuer_id,
            "public_key": self.public_key.to_dict(),
            "valid_from": self.valid_from.isoformat(),
            "valid_until": self.valid_until.isoformat(),
            "signature": base64.b64encode(self.signature).decode(),
            "serial_number": self.serial_number,
            "extensions": self.extensions,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Certificate":
        """Create from dictionary."""
        return cls(
            subject_id=data["subject_id"],
            issuer_id=data["issuer_id"],
            public_key=PublicKey.from_dict(data["public_key"]),
            valid_from=datetime.fromisoformat(data["valid_from"]),
            valid_until=datetime.fromisoformat(data["valid_until"]),
            signature=base64.b64decode(data.get("signature", "")),
            serial_number=data.get("serial_number", ""),
            extensions=data.get("extensions", {}),
        )


@dataclass
class Identity:
    """Federation node identity."""

    node_id: str
    display_name: str
    public_key: PublicKey
    certificate: Optional[Certificate] = None
    status: IdentityStatus = IdentityStatus.UNVERIFIED
    endpoints: List[str] = field(default_factory=list)
    capabilities: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_seen: Optional[datetime] = None

    @property
    def is_verified(self) -> bool:
        """Check if identity is verified."""
        return self.status in (IdentityStatus.VERIFIED, IdentityStatus.TRUSTED)

    @property
    def fingerprint(self) -> str:
        """Get identity fingerprint."""
        data = f"{self.node_id}:{self.public_key.key_id}"
        return hashlib.sha256(data.encode()).hexdigest()[:32]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "node_id": self.node_id,
            "display_name": self.display_name,
            "public_key": self.public_key.to_dict(),
            "certificate": self.certificate.to_dict() if self.certificate else None,
            "status": self.status.value,
            "endpoints": self.endpoints,
            "capabilities": self.capabilities,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "last_seen": self.last_seen.isoformat() if self.last_seen else None,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Identity":
        """Create from dictionary."""
        return cls(
            node_id=data["node_id"],
            display_name=data["display_name"],
            public_key=PublicKey.from_dict(data["public_key"]),
            certificate=Certificate.from_dict(data["certificate"]) if data.get("certificate") else None,
            status=IdentityStatus(data.get("status", "unverified")),
            endpoints=data.get("endpoints", []),
            capabilities=data.get("capabilities", []),
            metadata=data.get("metadata", {}),
            created_at=datetime.fromisoformat(data.get("created_at", datetime.utcnow().isoformat())),
            last_seen=datetime.fromisoformat(data["last_seen"]) if data.get("last_seen") else None,
        )


# =============================================================================
# Identity Manager
# =============================================================================


class IdentityManager:
    """
    Manages node identities and verification.
    """

    def __init__(
        self,
        node_id: str,
        storage_path: Optional[Path] = None,
        key_type: KeyType = KeyType.ED25519,
    ):
        self.node_id = node_id
        self.storage_path = storage_path
        self.key_type = key_type

        # Own identity
        self._key_pair: Optional[KeyPair] = None
        self._identity: Optional[Identity] = None
        self._certificate: Optional[Certificate] = None

        # Known identities
        self._known_identities: Dict[str, Identity] = {}
        self._trusted_ids: set = set()

        # Initialize
        self._initialize()

    def _initialize(self) -> None:
        """Initialize identity manager."""
        # Try to load existing identity
        if self.storage_path and self.storage_path.exists():
            self._load_identity()
        else:
            # Generate new identity
            self._generate_identity()

    def _generate_identity(self) -> None:
        """Generate new node identity."""
        self._key_pair = KeyPair.generate(self.key_type)

        self._identity = Identity(
            node_id=self.node_id,
            display_name=self.node_id,
            public_key=self._key_pair.public_key,
            status=IdentityStatus.SELF_SIGNED,
        )

        # Create self-signed certificate
        self._certificate = self._create_self_signed_certificate()
        self._identity.certificate = self._certificate

        logger.info(f"Generated new identity for node: {self.node_id}")

    def _create_self_signed_certificate(self) -> Certificate:
        """Create a self-signed certificate."""
        now = datetime.utcnow()
        valid_until = now + timedelta(days=365)

        cert = Certificate(
            subject_id=self.node_id,
            issuer_id=self.node_id,
            public_key=self._key_pair.public_key,
            valid_from=now,
            valid_until=valid_until,
        )

        # Sign the certificate
        cert.signature = self._sign_certificate(cert)

        return cert

    def _sign_certificate(self, cert: Certificate) -> bytes:
        """Sign a certificate."""
        # Create signing data
        sign_data = json.dumps({
            "subject_id": cert.subject_id,
            "issuer_id": cert.issuer_id,
            "public_key": cert.public_key.to_dict(),
            "valid_from": cert.valid_from.isoformat(),
            "valid_until": cert.valid_until.isoformat(),
            "serial_number": cert.serial_number,
        }, sort_keys=True).encode()

        return self.sign(sign_data)

    def _load_identity(self) -> None:
        """Load identity from storage."""
        try:
            if self.storage_path:
                identity_file = self.storage_path / "identity.json"
                key_file = self.storage_path / "private_key.pem"

                if identity_file.exists():
                    with open(identity_file) as f:
                        data = json.load(f)
                    self._identity = Identity.from_dict(data)
                    self._certificate = self._identity.certificate

                if key_file.exists():
                    with open(key_file, "rb") as f:
                        key_data = f.read()
                    # Parse key data (simplified)
                    self._key_pair = KeyPair(
                        public_key=self._identity.public_key,
                        private_key=PrivateKey(
                            key_type=self._identity.public_key.key_type,
                            key_data=key_data,
                        ),
                    )

                logger.info(f"Loaded identity for node: {self.node_id}")

        except Exception as e:
            logger.error(f"Failed to load identity: {e}")
            self._generate_identity()

    def save_identity(self) -> None:
        """Save identity to storage."""
        if not self.storage_path:
            return

        try:
            self.storage_path.mkdir(parents=True, exist_ok=True)

            identity_file = self.storage_path / "identity.json"
            key_file = self.storage_path / "private_key.pem"

            with open(identity_file, "w") as f:
                json.dump(self._identity.to_dict(), f, indent=2)

            with open(key_file, "wb") as f:
                f.write(self._key_pair.private_key.key_data)

            # Restrict key file permissions
            os.chmod(key_file, 0o600)

            logger.info(f"Saved identity for node: {self.node_id}")

        except Exception as e:
            logger.error(f"Failed to save identity: {e}")

    @property
    def identity(self) -> Identity:
        """Get own identity."""
        return self._identity

    @property
    def public_key(self) -> PublicKey:
        """Get own public key."""
        return self._key_pair.public_key

    @property
    def certificate(self) -> Certificate:
        """Get own certificate."""
        return self._certificate

    def sign(self, data: bytes) -> bytes:
        """Sign data with private key."""
        if not self._key_pair:
            raise ValueError("No private key available")

        try:
            if self._key_pair.key_type == KeyType.ED25519:
                from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

                private_key = Ed25519PrivateKey.from_private_bytes(
                    self._key_pair.private_key.key_data
                )
                return private_key.sign(data)

        except ImportError:
            pass

        # Mock signature
        return hmac.new(
            self._key_pair.private_key.key_data,
            data,
            hashlib.sha256,
        ).digest()

    def verify_signature(
        self,
        data: bytes,
        signature: bytes,
        public_key: PublicKey,
    ) -> bool:
        """Verify a signature."""
        try:
            if public_key.key_type == KeyType.ED25519:
                from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

                pub_key = Ed25519PublicKey.from_public_bytes(public_key.key_data)
                pub_key.verify(signature, data)
                return True

        except ImportError:
            pass
        except Exception:
            return False

        # Mock verification
        expected = hmac.new(
            public_key.key_data,
            data,
            hashlib.sha256,
        ).digest()
        return hmac.compare_digest(signature, expected)

    def register_identity(self, identity: Identity) -> bool:
        """Register a known identity."""
        if identity.node_id in self._known_identities:
            # Update existing
            existing = self._known_identities[identity.node_id]
            if existing.public_key.key_id != identity.public_key.key_id:
                logger.warning(f"Identity key changed for {identity.node_id}")
                return False

        self._known_identities[identity.node_id] = identity
        identity.last_seen = datetime.utcnow()
        logger.info(f"Registered identity: {identity.node_id}")
        return True

    def get_identity(self, node_id: str) -> Optional[Identity]:
        """Get a known identity."""
        return self._known_identities.get(node_id)

    def verify_identity(self, identity: Identity) -> bool:
        """Verify an identity's certificate."""
        if not identity.certificate:
            return False

        cert = identity.certificate

        # Check validity period
        if not cert.is_valid:
            logger.warning(f"Certificate expired for {identity.node_id}")
            return False

        # Check key matches
        if identity.public_key.key_id != cert.public_key.key_id:
            logger.warning(f"Key mismatch for {identity.node_id}")
            return False

        # Verify signature
        sign_data = json.dumps({
            "subject_id": cert.subject_id,
            "issuer_id": cert.issuer_id,
            "public_key": cert.public_key.to_dict(),
            "valid_from": cert.valid_from.isoformat(),
            "valid_until": cert.valid_until.isoformat(),
            "serial_number": cert.serial_number,
        }, sort_keys=True).encode()

        # Get issuer's public key
        if cert.is_self_signed:
            issuer_key = cert.public_key
        else:
            issuer = self.get_identity(cert.issuer_id)
            if not issuer:
                logger.warning(f"Unknown issuer: {cert.issuer_id}")
                return False
            issuer_key = issuer.public_key

        if not self.verify_signature(sign_data, cert.signature, issuer_key):
            logger.warning(f"Invalid certificate signature for {identity.node_id}")
            return False

        identity.status = IdentityStatus.VERIFIED
        return True

    def trust_identity(self, node_id: str) -> bool:
        """Mark an identity as trusted."""
        identity = self.get_identity(node_id)
        if not identity:
            return False

        if not self.verify_identity(identity):
            return False

        identity.status = IdentityStatus.TRUSTED
        self._trusted_ids.add(node_id)
        logger.info(f"Trusted identity: {node_id}")
        return True

    def revoke_trust(self, node_id: str) -> bool:
        """Revoke trust for an identity."""
        identity = self.get_identity(node_id)
        if not identity:
            return False

        identity.status = IdentityStatus.REVOKED
        self._trusted_ids.discard(node_id)
        logger.info(f"Revoked trust for: {node_id}")
        return True

    def is_trusted(self, node_id: str) -> bool:
        """Check if an identity is trusted."""
        return node_id in self._trusted_ids

    def list_identities(
        self,
        status: Optional[IdentityStatus] = None,
    ) -> List[Identity]:
        """List known identities."""
        identities = list(self._known_identities.values())
        if status:
            identities = [i for i in identities if i.status == status]
        return identities

    def create_attestation(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a signed attestation."""
        attestation = {
            "node_id": self.node_id,
            "timestamp": datetime.utcnow().isoformat(),
            "data": data,
        }

        sign_data = json.dumps(attestation, sort_keys=True).encode()
        signature = self.sign(sign_data)

        attestation["signature"] = base64.b64encode(signature).decode()
        return attestation

    def verify_attestation(
        self,
        attestation: Dict[str, Any],
    ) -> Tuple[bool, Optional[Identity]]:
        """Verify a signed attestation."""
        node_id = attestation.get("node_id")
        signature_b64 = attestation.pop("signature", None)

        if not node_id or not signature_b64:
            return False, None

        identity = self.get_identity(node_id)
        if not identity:
            return False, None

        signature = base64.b64decode(signature_b64)
        sign_data = json.dumps(attestation, sort_keys=True).encode()

        valid = self.verify_signature(sign_data, signature, identity.public_key)
        return valid, identity if valid else None


# =============================================================================
# Factory Functions
# =============================================================================


def create_identity(
    node_id: str,
    display_name: Optional[str] = None,
    key_type: KeyType = KeyType.ED25519,
) -> Identity:
    """
    Create a new identity.

    Args:
        node_id: Unique node identifier
        display_name: Human-readable name
        key_type: Key type to use

    Returns:
        Identity instance
    """
    key_pair = KeyPair.generate(key_type)

    return Identity(
        node_id=node_id,
        display_name=display_name or node_id,
        public_key=key_pair.public_key,
        status=IdentityStatus.SELF_SIGNED,
    )


def verify_identity(
    identity: Identity,
    manager: IdentityManager,
) -> bool:
    """
    Verify an identity.

    Args:
        identity: Identity to verify
        manager: Identity manager

    Returns:
        True if verified
    """
    return manager.verify_identity(identity)
