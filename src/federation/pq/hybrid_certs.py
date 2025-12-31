"""
Hybrid Post-Quantum Certificates

Provides quantum-resistant certificates for federation identity management:
- HybridCertificate: Certificate with both Ed25519 and ML-DSA signatures
- HybridIdentity: Identity with hybrid key pairs
- HybridIdentityManager: Manages hybrid identities and certificate verification

The hybrid approach ensures security during the quantum transition:
- Classical signatures (Ed25519) maintain compatibility
- Post-quantum signatures (ML-DSA) provide future-proofing
- Both signatures must be valid for certificate verification

This follows NIST recommendations for hybrid cryptographic certificates.

Usage:
    # Create hybrid identity manager
    manager = HybridIdentityManager(node_id="node-001")

    # Get our hybrid certificate
    cert = manager.certificate

    # Sign a certificate for another node
    peer_cert = manager.issue_certificate(peer_identity)

    # Verify a certificate
    valid = manager.verify_certificate(peer_cert)
"""

import base64
import hashlib
import json
import logging
import os
import secrets
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .hybrid import (
    HybridKeyPair,
    HybridMode,
    HybridPrivateKey,
    HybridPublicKey,
    HybridSigAlgorithm,
    HybridSignature,
    HybridSigner,
)
from .ml_dsa import MLDSASecurityLevel

logger = logging.getLogger(__name__)


# =============================================================================
# Constants
# =============================================================================


class HybridCertificateVersion(str, Enum):
    """Certificate format versions."""

    V1_CLASSICAL = "v1.0"  # Classical Ed25519 only
    V2_HYBRID = "v2.0"  # Hybrid Ed25519 + ML-DSA


class CertificateType(str, Enum):
    """Certificate types."""

    SELF_SIGNED = "self_signed"
    NODE = "node"
    SERVICE = "service"
    INTERMEDIATE_CA = "intermediate_ca"
    ROOT_CA = "root_ca"


class HybridIdentityStatus(str, Enum):
    """Identity verification status."""

    UNVERIFIED = "unverified"
    SELF_SIGNED = "self_signed"
    VERIFIED = "verified"
    TRUSTED = "trusted"
    REVOKED = "revoked"
    EXPIRED = "expired"


# =============================================================================
# Hybrid Public Key for Identity
# =============================================================================


@dataclass
class HybridIdentityKey:
    """
    Combined classical and post-quantum public key for identity.

    Contains both Ed25519 and ML-DSA public keys for signature verification.
    """

    classical_key: bytes  # Ed25519 public key (32 bytes)
    pq_key: bytes  # ML-DSA public key
    algorithm: str  # Algorithm identifier
    key_id: str = ""
    security_level: str = "level_3"
    created_at: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.key_id:
            combined = self.classical_key + self.pq_key[:32]  # Use first 32 bytes
            self.key_id = hashlib.sha256(combined).hexdigest()[:16]

    @property
    def fingerprint(self) -> str:
        """Get key fingerprint."""
        combined = self.classical_key + self.pq_key
        return hashlib.sha256(combined).hexdigest()

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
            "security_level": self.security_level,
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HybridIdentityKey":
        """Create from dictionary."""
        return cls(
            classical_key=base64.b64decode(data["classical_key"]),
            pq_key=base64.b64decode(data["pq_key"]),
            algorithm=data["algorithm"],
            key_id=data.get("key_id", ""),
            security_level=data.get("security_level", "level_3"),
            created_at=datetime.fromisoformat(
                data.get("created_at", datetime.utcnow().isoformat())
            ),
            metadata=data.get("metadata", {}),
        )

    def to_hybrid_public_key(self) -> HybridPublicKey:
        """Convert to HybridPublicKey for use with HybridSigner."""
        return HybridPublicKey(
            classical_key=self.classical_key,
            pq_key=self.pq_key,
            algorithm=self.algorithm,
            key_id=self.key_id,
            created_at=self.created_at,
        )


# =============================================================================
# Hybrid Certificate Signature
# =============================================================================


@dataclass
class HybridCertificateSignature:
    """
    Combined classical and post-quantum certificate signature.

    Both signatures must be valid for the certificate to be considered valid.
    """

    classical_signature: bytes  # Ed25519 signature (64 bytes)
    pq_signature: bytes  # ML-DSA signature
    algorithm: str
    signer_key_id: str = ""
    signed_at: datetime = field(default_factory=datetime.utcnow)

    @property
    def size(self) -> int:
        """Total signature size in bytes."""
        return len(self.classical_signature) + len(self.pq_signature)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "classical_signature": base64.b64encode(self.classical_signature).decode(),
            "pq_signature": base64.b64encode(self.pq_signature).decode(),
            "algorithm": self.algorithm,
            "signer_key_id": self.signer_key_id,
            "signed_at": self.signed_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HybridCertificateSignature":
        """Create from dictionary."""
        return cls(
            classical_signature=base64.b64decode(data["classical_signature"]),
            pq_signature=base64.b64decode(data["pq_signature"]),
            algorithm=data["algorithm"],
            signer_key_id=data.get("signer_key_id", ""),
            signed_at=datetime.fromisoformat(data.get("signed_at", datetime.utcnow().isoformat())),
        )

    def to_hybrid_signature(self) -> HybridSignature:
        """Convert to HybridSignature for verification."""
        return HybridSignature(
            classical_signature=self.classical_signature,
            pq_signature=self.pq_signature,
            algorithm=self.algorithm,
            signer_key_id=self.signer_key_id,
        )


# =============================================================================
# Hybrid Certificate
# =============================================================================


@dataclass
class HybridCertificate:
    """
    Quantum-resistant certificate with hybrid signatures.

    Contains both Ed25519 and ML-DSA signatures for future-proofing.
    Compatible with classical certificate workflows while adding PQ security.
    """

    # Certificate info
    subject_id: str  # Subject node/service ID
    issuer_id: str  # Issuer (CA or self) ID
    serial_number: str = ""  # Unique serial number

    # Public key
    public_key: Optional[HybridIdentityKey] = None

    # Validity period
    valid_from: datetime = field(default_factory=datetime.utcnow)
    valid_until: datetime = field(default_factory=lambda: datetime.utcnow() + timedelta(days=365))

    # Signature
    signature: Optional[HybridCertificateSignature] = None

    # Certificate metadata
    version: HybridCertificateVersion = HybridCertificateVersion.V2_HYBRID
    cert_type: CertificateType = CertificateType.NODE
    extensions: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.serial_number:
            self.serial_number = secrets.token_hex(16)

    @property
    def is_valid(self) -> bool:
        """Check if certificate is currently valid (time-based)."""
        now = datetime.utcnow()
        return self.valid_from <= now <= self.valid_until

    @property
    def is_self_signed(self) -> bool:
        """Check if certificate is self-signed."""
        return self.subject_id == self.issuer_id

    @property
    def is_hybrid(self) -> bool:
        """Check if certificate uses hybrid cryptography."""
        return self.version == HybridCertificateVersion.V2_HYBRID

    @property
    def days_until_expiry(self) -> int:
        """Get days until certificate expires."""
        delta = self.valid_until - datetime.utcnow()
        return max(0, delta.days)

    @property
    def fingerprint(self) -> str:
        """Get certificate fingerprint."""
        data = self._get_tbs_data()
        return hashlib.sha256(data).hexdigest()

    def _get_tbs_data(self) -> bytes:
        """Get the to-be-signed certificate data."""
        tbs = {
            "version": self.version.value,
            "subject_id": self.subject_id,
            "issuer_id": self.issuer_id,
            "serial_number": self.serial_number,
            "cert_type": self.cert_type.value,
            "public_key": self.public_key.to_dict() if self.public_key else None,
            "valid_from": self.valid_from.isoformat(),
            "valid_until": self.valid_until.isoformat(),
            "extensions": self.extensions,
        }
        return json.dumps(tbs, sort_keys=True).encode()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "version": self.version.value,
            "subject_id": self.subject_id,
            "issuer_id": self.issuer_id,
            "serial_number": self.serial_number,
            "cert_type": self.cert_type.value,
            "public_key": self.public_key.to_dict() if self.public_key else None,
            "valid_from": self.valid_from.isoformat(),
            "valid_until": self.valid_until.isoformat(),
            "signature": self.signature.to_dict() if self.signature else None,
            "extensions": self.extensions,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HybridCertificate":
        """Create from dictionary."""
        return cls(
            version=HybridCertificateVersion(data.get("version", "v2.0")),
            subject_id=data["subject_id"],
            issuer_id=data["issuer_id"],
            serial_number=data.get("serial_number", ""),
            cert_type=CertificateType(data.get("cert_type", "node")),
            public_key=(
                HybridIdentityKey.from_dict(data["public_key"]) if data.get("public_key") else None
            ),
            valid_from=datetime.fromisoformat(data["valid_from"]),
            valid_until=datetime.fromisoformat(data["valid_until"]),
            signature=(
                HybridCertificateSignature.from_dict(data["signature"])
                if data.get("signature")
                else None
            ),
            extensions=data.get("extensions", {}),
        )

    def to_pem(self) -> str:
        """Export to PEM-like format."""
        data = base64.b64encode(json.dumps(self.to_dict()).encode()).decode()
        lines = [data[i : i + 64] for i in range(0, len(data), 64)]
        return (
            "-----BEGIN HYBRID CERTIFICATE-----\n"
            + "\n".join(lines)
            + "\n-----END HYBRID CERTIFICATE-----"
        )

    @classmethod
    def from_pem(cls, pem: str) -> "HybridCertificate":
        """Import from PEM-like format."""
        lines = pem.strip().split("\n")
        # Remove headers
        lines = [l for l in lines if not l.startswith("-----")]
        data = base64.b64decode("".join(lines))
        return cls.from_dict(json.loads(data))


# =============================================================================
# Hybrid Identity
# =============================================================================


@dataclass
class HybridIdentity:
    """
    Federation node identity with hybrid cryptographic keys.

    Supports both classical and post-quantum signature verification.
    """

    node_id: str
    display_name: str
    public_key: HybridIdentityKey
    certificate: Optional[HybridCertificate] = None
    status: HybridIdentityStatus = HybridIdentityStatus.UNVERIFIED
    endpoints: List[str] = field(default_factory=list)
    capabilities: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_seen: Optional[datetime] = None

    @property
    def is_verified(self) -> bool:
        """Check if identity is verified."""
        return self.status in (HybridIdentityStatus.VERIFIED, HybridIdentityStatus.TRUSTED)

    @property
    def is_hybrid(self) -> bool:
        """Check if identity uses hybrid cryptography."""
        return self.public_key is not None and len(self.public_key.pq_key) > 0

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
    def from_dict(cls, data: Dict[str, Any]) -> "HybridIdentity":
        """Create from dictionary."""
        return cls(
            node_id=data["node_id"],
            display_name=data["display_name"],
            public_key=HybridIdentityKey.from_dict(data["public_key"]),
            certificate=(
                HybridCertificate.from_dict(data["certificate"])
                if data.get("certificate")
                else None
            ),
            status=HybridIdentityStatus(data.get("status", "unverified")),
            endpoints=data.get("endpoints", []),
            capabilities=data.get("capabilities", []),
            metadata=data.get("metadata", {}),
            created_at=datetime.fromisoformat(
                data.get("created_at", datetime.utcnow().isoformat())
            ),
            last_seen=datetime.fromisoformat(data["last_seen"]) if data.get("last_seen") else None,
        )


# =============================================================================
# Hybrid Identity Manager
# =============================================================================


class HybridIdentityManager:
    """
    Manages hybrid identities and quantum-resistant certificates.

    Provides:
    - Hybrid key pair generation (Ed25519 + ML-DSA)
    - Self-signed certificate creation
    - Certificate signing (CA functionality)
    - Certificate verification with hybrid signatures
    - Identity registration and trust management
    """

    def __init__(
        self,
        node_id: str,
        storage_path: Optional[Path] = None,
        ml_dsa_level: MLDSASecurityLevel = MLDSASecurityLevel.ML_DSA_65,
        mode: HybridMode = HybridMode.HYBRID,
    ):
        self.node_id = node_id
        self.storage_path = storage_path
        self.ml_dsa_level = ml_dsa_level
        self.mode = mode

        # Hybrid signer for certificate operations
        self._signer = HybridSigner(ml_dsa_level=ml_dsa_level)

        # Own key pair and identity
        self._keypair: Optional[HybridKeyPair] = None
        self._identity: Optional[HybridIdentity] = None
        self._certificate: Optional[HybridCertificate] = None

        # Known identities
        self._known_identities: Dict[str, HybridIdentity] = {}
        self._trusted_ids: set = set()

        # Certificate revocation list
        self._revoked_serials: set = set()

        # Initialize
        self._initialize()

    def _initialize(self) -> None:
        """Initialize the identity manager."""
        if self.storage_path and self.storage_path.exists():
            self._load_identity()
        else:
            self._generate_identity()

    def _generate_identity(self) -> None:
        """Generate new hybrid identity."""
        # Generate hybrid key pair
        self._keypair = self._signer.generate_keypair()

        # Create identity key from hybrid public key
        identity_key = HybridIdentityKey(
            classical_key=self._keypair.public_key.classical_key,
            pq_key=self._keypair.public_key.pq_key,
            algorithm=self._keypair.algorithm,
            security_level=self._get_security_level_name(),
        )

        # Create identity
        self._identity = HybridIdentity(
            node_id=self.node_id,
            display_name=self.node_id,
            public_key=identity_key,
            status=HybridIdentityStatus.SELF_SIGNED,
        )

        # Create self-signed certificate
        self._certificate = self._create_self_signed_certificate()
        self._identity.certificate = self._certificate

        logger.info(f"Generated new hybrid identity for node: {self.node_id}")
        logger.info(f"Algorithm: {self._keypair.algorithm}")
        logger.info(f"Key size: {identity_key.size} bytes")

    def _get_security_level_name(self) -> str:
        """Get security level name from ML-DSA level."""
        level_map = {
            MLDSASecurityLevel.ML_DSA_44: "level_1",
            MLDSASecurityLevel.ML_DSA_65: "level_3",
            MLDSASecurityLevel.ML_DSA_87: "level_5",
        }
        return level_map.get(self.ml_dsa_level, "level_3")

    def _create_self_signed_certificate(self) -> HybridCertificate:
        """Create a self-signed hybrid certificate."""
        now = datetime.utcnow()

        cert = HybridCertificate(
            subject_id=self.node_id,
            issuer_id=self.node_id,
            public_key=self._identity.public_key,
            valid_from=now,
            valid_until=now + timedelta(days=365),
            cert_type=CertificateType.SELF_SIGNED,
            extensions={
                "hybrid_mode": self.mode.value,
                "key_usage": ["digital_signature", "key_encipherment"],
            },
        )

        # Sign the certificate
        cert.signature = self._sign_certificate(cert)

        return cert

    def _sign_certificate(self, cert: HybridCertificate) -> HybridCertificateSignature:
        """Sign a certificate with our hybrid private key."""
        if not self._keypair:
            raise ValueError("No private key available for signing")

        # Get the to-be-signed data
        tbs_data = cert._get_tbs_data()

        # Sign with hybrid signer
        hybrid_sig = self._signer.sign(tbs_data, self._keypair.private_key)

        return HybridCertificateSignature(
            classical_signature=hybrid_sig.classical_signature,
            pq_signature=hybrid_sig.pq_signature,
            algorithm=hybrid_sig.algorithm,
            signer_key_id=self._identity.public_key.key_id,
        )

    def _load_identity(self) -> None:
        """Load identity from storage."""
        try:
            if self.storage_path:
                identity_file = self.storage_path / "hybrid_identity.json"
                key_file = self.storage_path / "hybrid_private_key.bin"

                if identity_file.exists():
                    with open(identity_file) as f:
                        data = json.load(f)
                    self._identity = HybridIdentity.from_dict(data)
                    self._certificate = self._identity.certificate

                if key_file.exists():
                    with open(key_file, "rb") as f:
                        key_data = json.loads(f.read())

                    self._keypair = HybridKeyPair(
                        public_key=HybridPublicKey(
                            classical_key=base64.b64decode(key_data["classical_public"]),
                            pq_key=base64.b64decode(key_data["pq_public"]),
                            algorithm=key_data["algorithm"],
                        ),
                        private_key=HybridPrivateKey(
                            classical_key=base64.b64decode(key_data["classical_private"]),
                            pq_key=base64.b64decode(key_data["pq_private"]),
                            algorithm=key_data["algorithm"],
                        ),
                        algorithm=key_data["algorithm"],
                    )

                logger.info(f"Loaded hybrid identity for node: {self.node_id}")

        except Exception as e:
            logger.error(f"Failed to load hybrid identity: {e}")
            self._generate_identity()

    def save_identity(self) -> None:
        """Save identity to storage."""
        if not self.storage_path:
            return

        try:
            self.storage_path.mkdir(parents=True, exist_ok=True)

            identity_file = self.storage_path / "hybrid_identity.json"
            key_file = self.storage_path / "hybrid_private_key.bin"

            # Save identity
            with open(identity_file, "w") as f:
                json.dump(self._identity.to_dict(), f, indent=2)

            # Save private key (encrypted in production!)
            key_data = {
                "classical_public": base64.b64encode(
                    self._keypair.public_key.classical_key
                ).decode(),
                "classical_private": base64.b64encode(
                    self._keypair.private_key.classical_key
                ).decode(),
                "pq_public": base64.b64encode(self._keypair.public_key.pq_key).decode(),
                "pq_private": base64.b64encode(self._keypair.private_key.pq_key).decode(),
                "algorithm": self._keypair.algorithm,
            }
            with open(key_file, "wb") as f:
                f.write(json.dumps(key_data).encode())

            # Restrict key file permissions
            os.chmod(key_file, 0o600)

            logger.info(f"Saved hybrid identity for node: {self.node_id}")

        except Exception as e:
            logger.error(f"Failed to save hybrid identity: {e}")

    @property
    def identity(self) -> HybridIdentity:
        """Get own identity."""
        return self._identity

    @property
    def public_key(self) -> HybridIdentityKey:
        """Get own public key."""
        return self._identity.public_key

    @property
    def certificate(self) -> HybridCertificate:
        """Get own certificate."""
        return self._certificate

    def sign(self, data: bytes) -> HybridSignature:
        """Sign data with our hybrid private key."""
        if not self._keypair:
            raise ValueError("No private key available")

        return self._signer.sign(data, self._keypair.private_key)

    def verify_signature(
        self,
        data: bytes,
        signature: HybridSignature,
        public_key: HybridIdentityKey,
    ) -> bool:
        """
        Verify a hybrid signature.

        Both classical and PQ signatures must be valid.
        """
        hybrid_public = public_key.to_hybrid_public_key()
        return self._signer.verify(data, signature, hybrid_public)

    def issue_certificate(
        self,
        subject_identity: HybridIdentity,
        valid_days: int = 365,
        cert_type: CertificateType = CertificateType.NODE,
        extensions: Optional[Dict[str, Any]] = None,
    ) -> HybridCertificate:
        """
        Issue a certificate for another identity.

        Args:
            subject_identity: Identity to issue certificate for
            valid_days: Validity period in days
            cert_type: Certificate type
            extensions: Additional certificate extensions

        Returns:
            Signed HybridCertificate
        """
        now = datetime.utcnow()

        cert = HybridCertificate(
            subject_id=subject_identity.node_id,
            issuer_id=self.node_id,
            public_key=subject_identity.public_key,
            valid_from=now,
            valid_until=now + timedelta(days=valid_days),
            cert_type=cert_type,
            extensions=extensions or {},
        )

        cert.signature = self._sign_certificate(cert)

        logger.info(f"Issued certificate for: {subject_identity.node_id}")
        return cert

    def verify_certificate(
        self,
        certificate: HybridCertificate,
        issuer_key: Optional[HybridIdentityKey] = None,
    ) -> Tuple[bool, str]:
        """
        Verify a hybrid certificate.

        Args:
            certificate: Certificate to verify
            issuer_key: Issuer's public key (if not self-signed)

        Returns:
            Tuple of (is_valid, reason)
        """
        # Check validity period
        if not certificate.is_valid:
            return False, "Certificate has expired or is not yet valid"

        # Check if revoked
        if certificate.serial_number in self._revoked_serials:
            return False, "Certificate has been revoked"

        # Check signature exists
        if not certificate.signature:
            return False, "Certificate has no signature"

        # Get issuer key
        if certificate.is_self_signed:
            verify_key = certificate.public_key
        elif issuer_key:
            verify_key = issuer_key
        else:
            # Try to find issuer in known identities
            issuer = self.get_identity(certificate.issuer_id)
            if not issuer:
                return False, f"Unknown issuer: {certificate.issuer_id}"
            verify_key = issuer.public_key

        # Verify hybrid signature
        tbs_data = certificate._get_tbs_data()
        hybrid_sig = certificate.signature.to_hybrid_signature()
        hybrid_public = verify_key.to_hybrid_public_key()

        if not self._signer.verify(tbs_data, hybrid_sig, hybrid_public):
            return False, "Invalid certificate signature"

        # Check public key matches
        if certificate.public_key and verify_key:
            if certificate.public_key.key_id != verify_key.key_id:
                # This is expected for non-self-signed certs
                pass

        return True, "Certificate is valid"

    def register_identity(self, identity: HybridIdentity) -> bool:
        """Register a known identity."""
        if identity.node_id in self._known_identities:
            existing = self._known_identities[identity.node_id]
            if existing.public_key.key_id != identity.public_key.key_id:
                logger.warning(f"Identity key changed for {identity.node_id}")
                return False

        self._known_identities[identity.node_id] = identity
        identity.last_seen = datetime.utcnow()
        logger.info(f"Registered hybrid identity: {identity.node_id}")
        return True

    def get_identity(self, node_id: str) -> Optional[HybridIdentity]:
        """Get a known identity."""
        return self._known_identities.get(node_id)

    def verify_identity(self, identity: HybridIdentity) -> bool:
        """Verify an identity's certificate."""
        if not identity.certificate:
            return False

        valid, reason = self.verify_certificate(identity.certificate)

        if not valid:
            logger.warning(f"Identity verification failed for {identity.node_id}: {reason}")
            return False

        identity.status = HybridIdentityStatus.VERIFIED
        return True

    def trust_identity(self, node_id: str) -> bool:
        """Mark an identity as trusted."""
        identity = self.get_identity(node_id)
        if not identity:
            return False

        if not self.verify_identity(identity):
            return False

        identity.status = HybridIdentityStatus.TRUSTED
        self._trusted_ids.add(node_id)
        logger.info(f"Trusted hybrid identity: {node_id}")
        return True

    def revoke_trust(self, node_id: str) -> bool:
        """Revoke trust for an identity."""
        identity = self.get_identity(node_id)
        if not identity:
            return False

        identity.status = HybridIdentityStatus.REVOKED
        self._trusted_ids.discard(node_id)

        # Add certificate to revocation list
        if identity.certificate:
            self._revoked_serials.add(identity.certificate.serial_number)

        logger.info(f"Revoked trust for hybrid identity: {node_id}")
        return True

    def is_trusted(self, node_id: str) -> bool:
        """Check if an identity is trusted."""
        return node_id in self._trusted_ids

    def list_identities(
        self,
        status: Optional[HybridIdentityStatus] = None,
    ) -> List[HybridIdentity]:
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

        attestation["signature"] = signature.to_dict()
        return attestation

    def verify_attestation(
        self,
        attestation: Dict[str, Any],
    ) -> Tuple[bool, Optional[HybridIdentity]]:
        """Verify a signed attestation."""
        node_id = attestation.get("node_id")
        signature_data = attestation.pop("signature", None)

        if not node_id or not signature_data:
            return False, None

        identity = self.get_identity(node_id)
        if not identity:
            return False, None

        signature = HybridSignature.from_dict(signature_data)
        sign_data = json.dumps(attestation, sort_keys=True).encode()

        hybrid_public = identity.public_key.to_hybrid_public_key()
        valid = self._signer.verify(sign_data, signature, hybrid_public)

        return valid, identity if valid else None

    def get_certificate_chain(
        self,
        certificate: HybridCertificate,
    ) -> List[HybridCertificate]:
        """
        Get the certificate chain for a certificate.

        Returns list from leaf to root.
        """
        chain = [certificate]

        current = certificate
        while not current.is_self_signed:
            issuer = self.get_identity(current.issuer_id)
            if not issuer or not issuer.certificate:
                break
            chain.append(issuer.certificate)
            current = issuer.certificate

        return chain

    def verify_certificate_chain(
        self,
        chain: List[HybridCertificate],
    ) -> Tuple[bool, str]:
        """
        Verify a certificate chain.

        Args:
            chain: List of certificates from leaf to root

        Returns:
            Tuple of (is_valid, reason)
        """
        if not chain:
            return False, "Empty certificate chain"

        # Verify each certificate in the chain
        for i, cert in enumerate(chain):
            if i == len(chain) - 1:
                # Root certificate (should be self-signed)
                if not cert.is_self_signed:
                    return False, "Root certificate is not self-signed"
                issuer_key = cert.public_key
            else:
                # Get issuer from next certificate in chain
                issuer_cert = chain[i + 1]
                issuer_key = issuer_cert.public_key

            valid, reason = self.verify_certificate(cert, issuer_key)
            if not valid:
                return False, f"Certificate {i} failed: {reason}"

        return True, "Certificate chain is valid"


# =============================================================================
# Factory Functions
# =============================================================================


def create_hybrid_identity_manager(
    node_id: str,
    storage_path: Optional[Path] = None,
    ml_dsa_level: MLDSASecurityLevel = MLDSASecurityLevel.ML_DSA_65,
) -> HybridIdentityManager:
    """Create a hybrid identity manager."""
    return HybridIdentityManager(
        node_id=node_id,
        storage_path=storage_path,
        ml_dsa_level=ml_dsa_level,
    )


def create_hybrid_identity(
    node_id: str,
    display_name: Optional[str] = None,
    ml_dsa_level: MLDSASecurityLevel = MLDSASecurityLevel.ML_DSA_65,
) -> HybridIdentity:
    """
    Create a new hybrid identity.

    Args:
        node_id: Unique node identifier
        display_name: Human-readable name
        ml_dsa_level: ML-DSA security level

    Returns:
        HybridIdentity instance
    """
    signer = HybridSigner(ml_dsa_level=ml_dsa_level)
    keypair = signer.generate_keypair()

    identity_key = HybridIdentityKey(
        classical_key=keypair.public_key.classical_key,
        pq_key=keypair.public_key.pq_key,
        algorithm=keypair.algorithm,
    )

    return HybridIdentity(
        node_id=node_id,
        display_name=display_name or node_id,
        public_key=identity_key,
        status=HybridIdentityStatus.SELF_SIGNED,
    )


def upgrade_classical_certificate(
    classical_cert_data: Dict[str, Any],
    signer: HybridSigner,
    private_key: HybridPrivateKey,
) -> HybridCertificate:
    """
    Upgrade a classical certificate to hybrid format.

    This allows gradual migration from classical to hybrid certificates.

    Args:
        classical_cert_data: Classical certificate data
        signer: Hybrid signer for creating new signature
        private_key: Private key for signing

    Returns:
        HybridCertificate with hybrid signatures
    """
    # Create hybrid certificate from classical data
    # This would require converting the classical public key to hybrid format
    # which requires access to the corresponding ML-DSA public key

    raise NotImplementedError(
        "Certificate upgrade requires access to post-quantum key pair. "
        "Use create_hybrid_identity_manager for new hybrid certificates."
    )
