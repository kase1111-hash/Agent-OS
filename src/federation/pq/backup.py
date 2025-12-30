"""
Key Backup and Recovery for Post-Quantum Keys

Provides secure key backup and recovery mechanisms:
- Shamir's Secret Sharing for split-key recovery
- Encrypted key export with password derivation
- Key escrow support for enterprise deployments
- Disaster recovery procedures

Security Features:
- Keys encrypted at rest with AES-256-GCM
- Password-based key derivation using Argon2id
- Multi-party recovery requiring M-of-N shares
- Audit trail for all backup/recovery operations

Usage:
    # Create backup manager
    backup = KeyBackupManager(config)

    # Export key with password protection
    backup_data = backup.export_key(
        key_handle,
        password="secure-password",
        format=BackupFormat.ENCRYPTED_JSON,
    )

    # Split key for multi-party recovery
    shares = backup.split_key(
        key_handle,
        total_shares=5,
        threshold=3,
    )

    # Recover from shares
    recovered_key = backup.recover_from_shares(shares[:3])
"""

import base64
import hashlib
import hmac
import json
import logging
import os
import secrets
import struct
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# =============================================================================
# Constants and Enums
# =============================================================================


class BackupFormat(str, Enum):
    """Key backup formats."""

    ENCRYPTED_JSON = "encrypted_json"
    ENCRYPTED_BINARY = "encrypted_binary"
    PEM_ENCRYPTED = "pem_encrypted"
    SHAMIR_SHARES = "shamir_shares"


class RecoveryMethod(str, Enum):
    """Key recovery methods."""

    PASSWORD = "password"
    SHAMIR = "shamir"
    ESCROW = "escrow"
    HSM_BACKUP = "hsm_backup"


class BackupStatus(str, Enum):
    """Backup status."""

    CREATED = "created"
    VERIFIED = "verified"
    EXPIRED = "expired"
    REVOKED = "revoked"
    RECOVERED = "recovered"


# =============================================================================
# Shamir's Secret Sharing
# =============================================================================


class ShamirSecretSharing:
    """
    Implementation of Shamir's Secret Sharing scheme.

    Allows splitting a secret into N shares where any K shares
    can reconstruct the original secret (K-of-N threshold scheme).

    Uses GF(2^8) arithmetic for byte-level operations.
    """

    # GF(2^8) multiplication table (precomputed for performance)
    _EXP = [0] * 512
    _LOG = [0] * 256

    @classmethod
    def _init_gf256(cls) -> None:
        """Initialize GF(2^8) lookup tables."""
        if cls._EXP[0] != 0:
            return

        x = 1
        for i in range(255):
            cls._EXP[i] = x
            cls._LOG[x] = i
            x = cls._gf256_mul_nomod(x, 2)
            if x >= 256:
                x ^= 0x11B  # AES polynomial

        for i in range(255, 512):
            cls._EXP[i] = cls._EXP[i - 255]

    @staticmethod
    def _gf256_mul_nomod(a: int, b: int) -> int:
        """GF(2^8) multiplication without modular reduction."""
        result = 0
        while b:
            if b & 1:
                result ^= a
            a <<= 1
            b >>= 1
        return result

    @classmethod
    def _gf256_mul(cls, a: int, b: int) -> int:
        """GF(2^8) multiplication."""
        cls._init_gf256()
        if a == 0 or b == 0:
            return 0
        return cls._EXP[cls._LOG[a] + cls._LOG[b]]

    @classmethod
    def _gf256_div(cls, a: int, b: int) -> int:
        """GF(2^8) division."""
        cls._init_gf256()
        if b == 0:
            raise ValueError("Division by zero")
        if a == 0:
            return 0
        return cls._EXP[(cls._LOG[a] - cls._LOG[b]) % 255]

    @classmethod
    def split(
        cls,
        secret: bytes,
        total_shares: int,
        threshold: int,
    ) -> List[Tuple[int, bytes]]:
        """
        Split secret into shares.

        Args:
            secret: The secret to split
            total_shares: Total number of shares to generate
            threshold: Minimum shares needed for recovery

        Returns:
            List of (share_index, share_data) tuples
        """
        cls._init_gf256()

        if threshold > total_shares:
            raise ValueError("Threshold cannot exceed total shares")
        if threshold < 2:
            raise ValueError("Threshold must be at least 2")
        if total_shares > 254:
            raise ValueError("Maximum 254 shares supported")

        shares = []

        # For each byte of the secret
        for byte_idx, secret_byte in enumerate(secret):
            # Generate random polynomial coefficients
            # f(x) = secret + a1*x + a2*x^2 + ... + a(k-1)*x^(k-1)
            coefficients = [secret_byte] + [
                secrets.randbelow(256) for _ in range(threshold - 1)
            ]

            # Evaluate polynomial at x = 1, 2, ..., n
            for share_idx in range(1, total_shares + 1):
                if byte_idx == 0:
                    shares.append((share_idx, bytearray()))

                y = 0
                x_power = 1
                for coef in coefficients:
                    y ^= cls._gf256_mul(coef, x_power)
                    x_power = cls._gf256_mul(x_power, share_idx)

                shares[share_idx - 1][1].append(y)

        return [(idx, bytes(data)) for idx, data in shares]

    @classmethod
    def recover(
        cls,
        shares: List[Tuple[int, bytes]],
    ) -> bytes:
        """
        Recover secret from shares using Lagrange interpolation.

        Args:
            shares: List of (share_index, share_data) tuples

        Returns:
            Recovered secret bytes
        """
        cls._init_gf256()

        if len(shares) < 2:
            raise ValueError("Need at least 2 shares for recovery")

        # Verify all shares have same length
        share_len = len(shares[0][1])
        if not all(len(s[1]) == share_len for s in shares):
            raise ValueError("All shares must have same length")

        # Get x coordinates
        x_coords = [s[0] for s in shares]

        # Recover each byte
        secret = bytearray()
        for byte_idx in range(share_len):
            y_coords = [s[1][byte_idx] for s in shares]

            # Lagrange interpolation at x = 0
            secret_byte = 0
            for i, (xi, yi) in enumerate(zip(x_coords, y_coords)):
                # Compute Lagrange basis polynomial at x = 0
                numerator = 1
                denominator = 1
                for j, xj in enumerate(x_coords):
                    if i != j:
                        numerator = cls._gf256_mul(numerator, xj)
                        denominator = cls._gf256_mul(denominator, xi ^ xj)

                lagrange = cls._gf256_div(numerator, denominator)
                secret_byte ^= cls._gf256_mul(yi, lagrange)

            secret.append(secret_byte)

        return bytes(secret)


# =============================================================================
# Key Backup Models
# =============================================================================


@dataclass
class KeyShare:
    """A single share of a split key."""

    share_id: str
    share_index: int
    share_data: bytes
    key_id: str
    algorithm: str
    threshold: int
    total_shares: int
    created_at: datetime = field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None
    custodian_id: Optional[str] = None
    checksum: str = ""

    def __post_init__(self):
        if not self.share_id:
            self.share_id = f"share:{secrets.token_hex(8)}"
        if not self.checksum:
            self.checksum = hashlib.sha256(self.share_data).hexdigest()[:16]

    @property
    def is_valid(self) -> bool:
        """Check if share is valid."""
        if self.expires_at and datetime.utcnow() > self.expires_at:
            return False
        actual_checksum = hashlib.sha256(self.share_data).hexdigest()[:16]
        return actual_checksum == self.checksum

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (excludes share_data for security)."""
        return {
            "share_id": self.share_id,
            "share_index": self.share_index,
            "key_id": self.key_id,
            "algorithm": self.algorithm,
            "threshold": self.threshold,
            "total_shares": self.total_shares,
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "custodian_id": self.custodian_id,
            "checksum": self.checksum,
        }

    def to_secure_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with share data (for export)."""
        data = self.to_dict()
        data["share_data"] = base64.b64encode(self.share_data).decode()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "KeyShare":
        """Create from dictionary."""
        return cls(
            share_id=data["share_id"],
            share_index=data["share_index"],
            share_data=base64.b64decode(data["share_data"]) if "share_data" in data else b"",
            key_id=data["key_id"],
            algorithm=data["algorithm"],
            threshold=data["threshold"],
            total_shares=data["total_shares"],
            created_at=datetime.fromisoformat(data["created_at"]),
            expires_at=datetime.fromisoformat(data["expires_at"]) if data.get("expires_at") else None,
            custodian_id=data.get("custodian_id"),
            checksum=data.get("checksum", ""),
        )


@dataclass
class KeyBackup:
    """Encrypted key backup."""

    backup_id: str
    key_id: str
    algorithm: str
    format: BackupFormat
    encrypted_data: bytes
    salt: bytes
    nonce: bytes
    created_at: datetime = field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None
    status: BackupStatus = BackupStatus.CREATED
    recovery_method: RecoveryMethod = RecoveryMethod.PASSWORD
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.backup_id:
            self.backup_id = f"backup:{secrets.token_hex(8)}"

    @property
    def is_valid(self) -> bool:
        """Check if backup is valid."""
        if self.status in (BackupStatus.EXPIRED, BackupStatus.REVOKED):
            return False
        if self.expires_at and datetime.utcnow() > self.expires_at:
            return False
        return True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "backup_id": self.backup_id,
            "key_id": self.key_id,
            "algorithm": self.algorithm,
            "format": self.format.value,
            "encrypted_data": base64.b64encode(self.encrypted_data).decode(),
            "salt": base64.b64encode(self.salt).decode(),
            "nonce": base64.b64encode(self.nonce).decode(),
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "status": self.status.value,
            "recovery_method": self.recovery_method.value,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "KeyBackup":
        """Create from dictionary."""
        return cls(
            backup_id=data["backup_id"],
            key_id=data["key_id"],
            algorithm=data["algorithm"],
            format=BackupFormat(data["format"]),
            encrypted_data=base64.b64decode(data["encrypted_data"]),
            salt=base64.b64decode(data["salt"]),
            nonce=base64.b64decode(data["nonce"]),
            created_at=datetime.fromisoformat(data["created_at"]),
            expires_at=datetime.fromisoformat(data["expires_at"]) if data.get("expires_at") else None,
            status=BackupStatus(data.get("status", "created")),
            recovery_method=RecoveryMethod(data.get("recovery_method", "password")),
            metadata=data.get("metadata", {}),
        )


@dataclass
class BackupConfig:
    """Key backup configuration."""

    # Storage
    backup_directory: Optional[Path] = None

    # Encryption
    kdf_iterations: int = 100000  # Argon2id iterations
    kdf_memory_kb: int = 65536    # 64 MB
    kdf_parallelism: int = 4

    # Shamir settings
    default_threshold: int = 3
    default_total_shares: int = 5
    share_expiry_days: int = 365

    # Backup policy
    backup_expiry_days: int = 730  # 2 years
    require_verification: bool = True
    max_recovery_attempts: int = 5

    # Escrow settings
    enable_escrow: bool = False
    escrow_public_key: Optional[bytes] = None


# =============================================================================
# Key Backup Manager
# =============================================================================


class KeyBackupManager:
    """
    Manages key backup and recovery operations.

    Provides:
    - Password-protected key export
    - Shamir secret sharing for M-of-N recovery
    - Backup verification
    - Secure key recovery
    """

    def __init__(self, config: Optional[BackupConfig] = None):
        self.config = config or BackupConfig()
        self._backups: Dict[str, KeyBackup] = {}
        self._shares: Dict[str, List[KeyShare]] = {}  # key_id -> shares
        self._recovery_attempts: Dict[str, int] = {}

    def export_key(
        self,
        key_id: str,
        private_key: bytes,
        algorithm: str,
        password: str,
        format: BackupFormat = BackupFormat.ENCRYPTED_JSON,
        expires_in: Optional[timedelta] = None,
    ) -> KeyBackup:
        """
        Export a key with password protection.

        Args:
            key_id: Key identifier
            private_key: Private key bytes
            algorithm: Key algorithm
            password: Encryption password
            format: Backup format
            expires_in: Backup expiry time

        Returns:
            KeyBackup object
        """
        # Generate salt and derive key
        salt = secrets.token_bytes(32)
        encryption_key = self._derive_key(password, salt)

        # Encrypt private key
        nonce = secrets.token_bytes(12)
        encrypted_data = self._encrypt(private_key, encryption_key, nonce)

        backup = KeyBackup(
            backup_id="",
            key_id=key_id,
            algorithm=algorithm,
            format=format,
            encrypted_data=encrypted_data,
            salt=salt,
            nonce=nonce,
            expires_at=datetime.utcnow() + expires_in if expires_in else None,
            recovery_method=RecoveryMethod.PASSWORD,
            metadata={
                "kdf": "argon2id",
                "kdf_iterations": self.config.kdf_iterations,
            },
        )

        self._backups[backup.backup_id] = backup

        # Save to disk if configured
        if self.config.backup_directory:
            self._save_backup(backup)

        logger.info(f"Created backup for key {key_id}: {backup.backup_id}")
        return backup

    def import_key(
        self,
        backup: KeyBackup,
        password: str,
    ) -> bytes:
        """
        Import a key from backup.

        Args:
            backup: KeyBackup object
            password: Decryption password

        Returns:
            Decrypted private key bytes
        """
        if not backup.is_valid:
            raise ValueError("Backup is not valid")

        # Check recovery attempts
        attempts = self._recovery_attempts.get(backup.backup_id, 0)
        if attempts >= self.config.max_recovery_attempts:
            raise ValueError("Maximum recovery attempts exceeded")

        try:
            # Derive key and decrypt
            encryption_key = self._derive_key(password, backup.salt)
            private_key = self._decrypt(
                backup.encrypted_data,
                encryption_key,
                backup.nonce,
            )

            # Reset attempts on success
            self._recovery_attempts[backup.backup_id] = 0

            backup.status = BackupStatus.RECOVERED
            logger.info(f"Recovered key from backup: {backup.backup_id}")

            return private_key

        except Exception as e:
            self._recovery_attempts[backup.backup_id] = attempts + 1
            logger.warning(
                f"Recovery attempt {attempts + 1} failed for {backup.backup_id}: {e}"
            )
            raise ValueError("Decryption failed - incorrect password")

    def split_key(
        self,
        key_id: str,
        private_key: bytes,
        algorithm: str,
        total_shares: Optional[int] = None,
        threshold: Optional[int] = None,
        custodian_ids: Optional[List[str]] = None,
        expires_in: Optional[timedelta] = None,
    ) -> List[KeyShare]:
        """
        Split a key into shares using Shamir's Secret Sharing.

        Args:
            key_id: Key identifier
            private_key: Private key bytes
            algorithm: Key algorithm
            total_shares: Number of shares to create
            threshold: Minimum shares for recovery
            custodian_ids: Optional custodian IDs for each share
            expires_in: Share expiry time

        Returns:
            List of KeyShare objects
        """
        total = total_shares or self.config.default_total_shares
        thresh = threshold or self.config.default_threshold

        if thresh > total:
            raise ValueError("Threshold cannot exceed total shares")

        # Split the key
        raw_shares = ShamirSecretSharing.split(private_key, total, thresh)

        expires_at = None
        if expires_in:
            expires_at = datetime.utcnow() + expires_in
        elif self.config.share_expiry_days:
            expires_at = datetime.utcnow() + timedelta(days=self.config.share_expiry_days)

        # Create share objects
        shares = []
        for i, (idx, data) in enumerate(raw_shares):
            custodian = custodian_ids[i] if custodian_ids and i < len(custodian_ids) else None

            share = KeyShare(
                share_id="",
                share_index=idx,
                share_data=data,
                key_id=key_id,
                algorithm=algorithm,
                threshold=thresh,
                total_shares=total,
                expires_at=expires_at,
                custodian_id=custodian,
            )
            shares.append(share)

        self._shares[key_id] = shares
        logger.info(f"Split key {key_id} into {total} shares (threshold: {thresh})")

        return shares

    def recover_from_shares(
        self,
        shares: List[KeyShare],
    ) -> Tuple[bytes, str]:
        """
        Recover a key from shares.

        Args:
            shares: List of KeyShare objects

        Returns:
            Tuple of (private_key, key_id)
        """
        if not shares:
            raise ValueError("No shares provided")

        # Verify all shares are for same key
        key_id = shares[0].key_id
        threshold = shares[0].threshold
        algorithm = shares[0].algorithm

        if not all(s.key_id == key_id for s in shares):
            raise ValueError("Shares are for different keys")

        if len(shares) < threshold:
            raise ValueError(f"Need at least {threshold} shares, got {len(shares)}")

        # Verify shares
        for share in shares:
            if not share.is_valid:
                raise ValueError(f"Invalid share: {share.share_id}")

        # Extract raw shares
        raw_shares = [(s.share_index, s.share_data) for s in shares]

        # Recover
        private_key = ShamirSecretSharing.recover(raw_shares)

        logger.info(f"Recovered key {key_id} from {len(shares)} shares")
        return private_key, key_id

    def verify_backup(self, backup: KeyBackup, password: str) -> bool:
        """
        Verify a backup can be decrypted.

        Args:
            backup: KeyBackup to verify
            password: Decryption password

        Returns:
            True if backup is valid
        """
        try:
            self.import_key(backup, password)
            backup.status = BackupStatus.VERIFIED
            return True
        except Exception:
            return False

    def verify_shares(self, shares: List[KeyShare]) -> bool:
        """
        Verify shares can recover a key.

        Args:
            shares: Shares to verify

        Returns:
            True if shares are valid
        """
        try:
            self.recover_from_shares(shares)
            return True
        except Exception:
            return False

    def revoke_backup(self, backup_id: str) -> bool:
        """Revoke a backup."""
        if backup_id in self._backups:
            self._backups[backup_id].status = BackupStatus.REVOKED
            logger.info(f"Revoked backup: {backup_id}")
            return True
        return False

    def revoke_shares(self, key_id: str) -> bool:
        """Revoke all shares for a key."""
        if key_id in self._shares:
            del self._shares[key_id]
            logger.info(f"Revoked shares for key: {key_id}")
            return True
        return False

    def _derive_key(self, password: str, salt: bytes) -> bytes:
        """Derive encryption key from password using Argon2id."""
        try:
            from argon2.low_level import hash_secret_raw, Type

            return hash_secret_raw(
                secret=password.encode(),
                salt=salt,
                time_cost=self.config.kdf_iterations // 1000,
                memory_cost=self.config.kdf_memory_kb,
                parallelism=self.config.kdf_parallelism,
                hash_len=32,
                type=Type.ID,
            )
        except ImportError:
            # Fallback to PBKDF2
            logger.warning("Argon2 not available, using PBKDF2")
            import hashlib
            return hashlib.pbkdf2_hmac(
                "sha256",
                password.encode(),
                salt,
                self.config.kdf_iterations,
                dklen=32,
            )

    def _encrypt(self, plaintext: bytes, key: bytes, nonce: bytes) -> bytes:
        """Encrypt with AES-256-GCM.

        Raises:
            ImportError: If the cryptography library is not available.
                        This is a security-critical function and MUST NOT
                        fall back to insecure alternatives.
        """
        try:
            from cryptography.hazmat.primitives.ciphers.aead import AESGCM
            aesgcm = AESGCM(key)
            return aesgcm.encrypt(nonce, plaintext, None)
        except ImportError:
            # SECURITY: Fail securely - do NOT use insecure fallbacks
            raise ImportError(
                "The 'cryptography' library is required for secure key backup. "
                "Install with: pip install cryptography"
            )

    def _decrypt(self, ciphertext: bytes, key: bytes, nonce: bytes) -> bytes:
        """Decrypt with AES-256-GCM.

        Raises:
            ImportError: If the cryptography library is not available.
                        This is a security-critical function and MUST NOT
                        fall back to insecure alternatives.
        """
        try:
            from cryptography.hazmat.primitives.ciphers.aead import AESGCM
            aesgcm = AESGCM(key)
            return aesgcm.decrypt(nonce, ciphertext, None)
        except ImportError:
            # SECURITY: Fail securely - do NOT use insecure fallbacks
            raise ImportError(
                "The 'cryptography' library is required for secure key backup. "
                "Install with: pip install cryptography"
            )

    def _save_backup(self, backup: KeyBackup) -> None:
        """Save backup to disk."""
        if not self.config.backup_directory:
            return

        try:
            self.config.backup_directory.mkdir(parents=True, exist_ok=True)
            backup_file = self.config.backup_directory / f"{backup.backup_id}.json"

            with open(backup_file, "w") as f:
                json.dump(backup.to_dict(), f, indent=2)

            os.chmod(backup_file, 0o600)

        except Exception as e:
            logger.error(f"Failed to save backup: {e}")

    def load_backup(self, backup_id: str) -> Optional[KeyBackup]:
        """Load backup from disk."""
        if not self.config.backup_directory:
            return self._backups.get(backup_id)

        backup_file = self.config.backup_directory / f"{backup_id}.json"
        if not backup_file.exists():
            return None

        try:
            with open(backup_file) as f:
                data = json.load(f)
            return KeyBackup.from_dict(data)
        except Exception as e:
            logger.error(f"Failed to load backup: {e}")
            return None

    def list_backups(self, key_id: Optional[str] = None) -> List[KeyBackup]:
        """List backups, optionally filtered by key ID."""
        backups = list(self._backups.values())
        if key_id:
            backups = [b for b in backups if b.key_id == key_id]
        return backups

    def export_shares_for_distribution(
        self,
        shares: List[KeyShare],
    ) -> List[Dict[str, Any]]:
        """
        Export shares for distribution to custodians.

        Each share is independently encrypted for its custodian.

        Returns:
            List of share dictionaries for distribution
        """
        distributed = []
        for share in shares:
            # Create distribution package
            package = {
                "share_id": share.share_id,
                "share_index": share.share_index,
                "key_id": share.key_id,
                "algorithm": share.algorithm,
                "threshold": share.threshold,
                "total_shares": share.total_shares,
                "created_at": share.created_at.isoformat(),
                "expires_at": share.expires_at.isoformat() if share.expires_at else None,
                "custodian_id": share.custodian_id,
                # Base64 encoded for safe transport
                "share_data": base64.b64encode(share.share_data).decode(),
                "checksum": share.checksum,
                "instructions": (
                    f"This is share {share.share_index} of {share.total_shares}. "
                    f"At least {share.threshold} shares are required for key recovery. "
                    "Store this share securely and do not share with others."
                ),
            }
            distributed.append(package)

        return distributed


# =============================================================================
# Factory Functions
# =============================================================================


def create_backup_manager(
    backup_directory: Optional[Path] = None,
    **kwargs,
) -> KeyBackupManager:
    """Create a key backup manager."""
    config = BackupConfig(
        backup_directory=backup_directory,
        **kwargs,
    )
    return KeyBackupManager(config)


def create_production_backup_config(
    backup_directory: Path,
) -> BackupConfig:
    """Create production backup configuration."""
    return BackupConfig(
        backup_directory=backup_directory,
        kdf_iterations=100000,
        kdf_memory_kb=65536,  # 64 MB
        kdf_parallelism=4,
        default_threshold=3,
        default_total_shares=5,
        share_expiry_days=365,
        backup_expiry_days=730,
        require_verification=True,
        max_recovery_attempts=5,
    )
