"""
Agent OS Memory Vault Key Manager

Handles key generation, derivation, storage, and hardware binding.
Supports software keys and TPM/hardware security module integration.
"""

import os
import secrets
import hashlib
import base64
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from enum import Enum, auto
import threading

from .profiles import EncryptionTier, KeyDerivation, KeyBinding, ProfileManager


logger = logging.getLogger(__name__)


class KeyStatus(Enum):
    """Status of a key."""
    ACTIVE = auto()
    LOCKED = auto()
    EXPIRED = auto()
    REVOKED = auto()
    PENDING_ROTATION = auto()


@dataclass
class KeyMetadata:
    """Metadata for a managed key."""
    key_id: str
    tier: EncryptionTier
    binding: KeyBinding
    created_at: datetime
    expires_at: Optional[datetime] = None
    last_used: Optional[datetime] = None
    use_count: int = 0
    status: KeyStatus = KeyStatus.ACTIVE
    rotation_scheduled: Optional[datetime] = None
    hardware_handle: Optional[str] = None  # TPM/HSM handle if hardware-bound
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "key_id": self.key_id,
            "tier": self.tier.name,
            "binding": self.binding.name,
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "last_used": self.last_used.isoformat() if self.last_used else None,
            "use_count": self.use_count,
            "status": self.status.name,
            "rotation_scheduled": (
                self.rotation_scheduled.isoformat()
                if self.rotation_scheduled else None
            ),
            "has_hardware_binding": self.hardware_handle is not None,
            "metadata": self.metadata,
        }


@dataclass
class DerivedKey:
    """A derived encryption key."""
    key: bytes
    salt: bytes
    key_id: str
    tier: EncryptionTier
    metadata: KeyMetadata


class HardwareBindingInterface:
    """
    Interface for hardware security module operations.

    This is a stub implementation. In production, this would integrate
    with actual TPM/HSM libraries (e.g., tpm2-tools, PKCS#11).
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self._available = False
        self._config = config or {}
        self._handles: Dict[str, bytes] = {}  # Simulated hardware handles
        self._check_availability()

    def _check_availability(self) -> None:
        """Check if hardware security module is available."""
        # In production: check for TPM or HSM device
        # For now, we simulate unavailability (software fallback)
        tpm_path = Path("/dev/tpm0")
        self._available = tpm_path.exists() or self._config.get("simulate_hardware", False)

    @property
    def is_available(self) -> bool:
        return self._available

    def create_key(self, key_id: str, key_data: bytes) -> str:
        """
        Create a hardware-protected key.

        Args:
            key_id: Unique key identifier
            key_data: Key material to protect

        Returns:
            Hardware handle for the key
        """
        if not self._available:
            raise RuntimeError("Hardware security module not available")

        # In production: TPM2_Create or HSM key creation
        # For simulation: store locally with a simulated handle
        handle = f"hw:{hashlib.sha256(key_id.encode()).hexdigest()[:16]}"
        self._handles[handle] = key_data
        logger.info(f"Created hardware-protected key: {handle}")
        return handle

    def load_key(self, handle: str) -> bytes:
        """
        Load a key from hardware.

        Args:
            handle: Hardware handle

        Returns:
            Key material
        """
        if not self._available:
            raise RuntimeError("Hardware security module not available")

        if handle not in self._handles:
            raise KeyError(f"Hardware key not found: {handle}")

        return self._handles[handle]

    def delete_key(self, handle: str) -> bool:
        """
        Delete a key from hardware.

        Args:
            handle: Hardware handle

        Returns:
            True if deleted
        """
        if handle in self._handles:
            del self._handles[handle]
            logger.info(f"Deleted hardware key: {handle}")
            return True
        return False

    def seal_data(self, data: bytes, policy: Optional[Dict] = None) -> bytes:
        """
        Seal data to hardware state.

        Args:
            data: Data to seal
            policy: Optional sealing policy

        Returns:
            Sealed blob
        """
        # In production: TPM2_Seal
        # For simulation: just hash the data with a marker
        sealed = b"SEALED:" + hashlib.sha256(data).digest() + data
        return sealed

    def unseal_data(self, sealed_data: bytes) -> bytes:
        """
        Unseal data from hardware.

        Args:
            sealed_data: Sealed blob

        Returns:
            Original data
        """
        if not sealed_data.startswith(b"SEALED:"):
            raise ValueError("Invalid sealed data format")
        return sealed_data[39:]  # Skip marker and hash


class KeyManager:
    """
    Manages encryption keys for the Memory Vault.

    Responsibilities:
    - Key generation and derivation
    - Key storage and retrieval
    - Hardware binding when available
    - Key rotation and expiry
    - Secure key deletion
    """

    # Key derivation parameters
    PBKDF2_ITERATIONS = 600000
    ARGON2_TIME_COST = 3
    ARGON2_MEMORY_COST = 65536
    ARGON2_PARALLELISM = 4
    SCRYPT_N = 2**14
    SCRYPT_R = 8
    SCRYPT_P = 1

    def __init__(
        self,
        key_store_path: Optional[Path] = None,
        profile_manager: Optional[ProfileManager] = None,
        hardware_config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize key manager.

        Args:
            key_store_path: Path to key store directory
            profile_manager: Profile manager instance
            hardware_config: Hardware security module configuration
        """
        self._key_store_path = key_store_path
        self._profile_manager = profile_manager or ProfileManager()
        self._hardware = HardwareBindingInterface(hardware_config)

        self._keys: Dict[str, KeyMetadata] = {}
        self._key_cache: Dict[str, bytes] = {}  # In-memory cache (cleared on lock)
        self._master_key: Optional[bytes] = None
        self._lock = threading.RLock()

        self._initialized = False

        if key_store_path:
            self._load_key_metadata()

    def initialize(self, master_password: Optional[str] = None) -> bool:
        """
        Initialize the key manager.

        Args:
            master_password: Optional master password for key store

        Returns:
            True if initialized successfully
        """
        with self._lock:
            if master_password:
                # Derive master key from password
                salt = self._get_or_create_master_salt()
                self._master_key = self._derive_key_pbkdf2(
                    master_password.encode(),
                    salt,
                    32,
                )
            else:
                # Generate random master key (for session-only use)
                self._master_key = secrets.token_bytes(32)

            self._initialized = True
            logger.info("Key manager initialized")
            return True

    def shutdown(self) -> None:
        """Securely shutdown key manager."""
        with self._lock:
            # Clear sensitive data from memory
            self._key_cache.clear()
            if self._master_key:
                # Overwrite with zeros before clearing
                self._master_key = b'\x00' * len(self._master_key)
                self._master_key = None
            self._initialized = False
            logger.info("Key manager shutdown complete")

    def generate_key(
        self,
        tier: EncryptionTier,
        purpose: str = "encryption",
        ttl: Optional[timedelta] = None,
    ) -> DerivedKey:
        """
        Generate a new encryption key for a tier.

        Args:
            tier: Encryption tier
            purpose: Key purpose description
            ttl: Optional time-to-live

        Returns:
            DerivedKey with key material
        """
        if not self._initialized:
            raise RuntimeError("Key manager not initialized")

        with self._lock:
            profile = self._profile_manager.get_profile(tier)

            # Generate key ID
            key_id = f"{tier.name.lower()}_{secrets.token_hex(8)}"

            # Generate salt
            salt = secrets.token_bytes(32)

            # Generate key material
            raw_key = secrets.token_bytes(profile.key_bits // 8)

            # Create metadata
            metadata = KeyMetadata(
                key_id=key_id,
                tier=tier,
                binding=profile.key_binding,
                created_at=datetime.now(),
                expires_at=datetime.now() + ttl if ttl else None,
                metadata={"purpose": purpose},
            )

            # Handle hardware binding if required
            if profile.key_binding in (KeyBinding.TPM, KeyBinding.HARDWARE):
                if self._hardware.is_available:
                    metadata.hardware_handle = self._hardware.create_key(key_id, raw_key)
                else:
                    logger.warning(
                        f"Hardware binding requested but unavailable for key {key_id}. "
                        "Falling back to software binding."
                    )
                    metadata.binding = KeyBinding.SOFTWARE

            # Store key metadata
            self._keys[key_id] = metadata
            self._key_cache[key_id] = raw_key
            self._persist_key_metadata()

            logger.info(f"Generated key: {key_id} (tier={tier.name}, binding={metadata.binding.name})")

            return DerivedKey(
                key=raw_key,
                salt=salt,
                key_id=key_id,
                tier=tier,
                metadata=metadata,
            )

    def derive_key(
        self,
        password: bytes,
        salt: bytes,
        tier: EncryptionTier,
        key_id: Optional[str] = None,
    ) -> DerivedKey:
        """
        Derive a key from a password.

        Args:
            password: Password bytes
            salt: Salt bytes
            tier: Encryption tier
            key_id: Optional key ID (generated if not provided)

        Returns:
            DerivedKey with derived key material
        """
        if not self._initialized:
            raise RuntimeError("Key manager not initialized")

        profile = self._profile_manager.get_profile(tier)
        key_bits = profile.key_bits

        # Derive using appropriate method
        if profile.key_derivation == KeyDerivation.PBKDF2:
            key = self._derive_key_pbkdf2(password, salt, key_bits // 8)
        elif profile.key_derivation == KeyDerivation.ARGON2ID:
            key = self._derive_key_argon2(password, salt, key_bits // 8)
        elif profile.key_derivation == KeyDerivation.SCRYPT:
            key = self._derive_key_scrypt(password, salt, key_bits // 8)
        else:
            raise ValueError(f"Unknown key derivation: {profile.key_derivation}")

        # Generate key ID if not provided
        if not key_id:
            key_id = f"derived_{tier.name.lower()}_{secrets.token_hex(8)}"

        metadata = KeyMetadata(
            key_id=key_id,
            tier=tier,
            binding=KeyBinding.SOFTWARE,
            created_at=datetime.now(),
            metadata={"derived": True},
        )

        self._keys[key_id] = metadata

        return DerivedKey(
            key=key,
            salt=salt,
            key_id=key_id,
            tier=tier,
            metadata=metadata,
        )

    def get_key(self, key_id: str) -> Optional[bytes]:
        """
        Retrieve a key by ID.

        Args:
            key_id: Key identifier

        Returns:
            Key bytes or None if not found/expired
        """
        with self._lock:
            metadata = self._keys.get(key_id)
            if not metadata:
                return None

            # Check status
            if metadata.status != KeyStatus.ACTIVE:
                logger.warning(f"Attempted access to non-active key: {key_id}")
                return None

            # Check expiry
            if metadata.expires_at and datetime.now() > metadata.expires_at:
                metadata.status = KeyStatus.EXPIRED
                logger.info(f"Key expired: {key_id}")
                return None

            # Update access tracking
            metadata.last_used = datetime.now()
            metadata.use_count += 1

            # Retrieve from cache or hardware
            if key_id in self._key_cache:
                return self._key_cache[key_id]

            if metadata.hardware_handle and self._hardware.is_available:
                return self._hardware.load_key(metadata.hardware_handle)

            return None

    def rotate_key(self, key_id: str) -> Optional[DerivedKey]:
        """
        Rotate a key (generate new key, mark old for retirement).

        Args:
            key_id: Key to rotate

        Returns:
            New DerivedKey or None if rotation failed
        """
        with self._lock:
            old_metadata = self._keys.get(key_id)
            if not old_metadata:
                return None

            # Mark old key for pending rotation
            old_metadata.status = KeyStatus.PENDING_ROTATION
            old_metadata.rotation_scheduled = datetime.now()

            # Generate new key
            new_key = self.generate_key(
                tier=old_metadata.tier,
                purpose=old_metadata.metadata.get("purpose", "encryption"),
            )

            # Store rotation reference
            new_key.metadata.metadata["rotated_from"] = key_id

            logger.info(f"Rotated key {key_id} -> {new_key.key_id}")
            return new_key

    def revoke_key(self, key_id: str) -> bool:
        """
        Revoke a key (immediate invalidation).

        Args:
            key_id: Key to revoke

        Returns:
            True if revoked
        """
        with self._lock:
            metadata = self._keys.get(key_id)
            if not metadata:
                return False

            metadata.status = KeyStatus.REVOKED

            # Clear from cache
            if key_id in self._key_cache:
                del self._key_cache[key_id]

            # Delete from hardware if applicable
            if metadata.hardware_handle and self._hardware.is_available:
                self._hardware.delete_key(metadata.hardware_handle)

            self._persist_key_metadata()
            logger.warning(f"Key revoked: {key_id}")
            return True

    def delete_key(self, key_id: str, secure: bool = True) -> bool:
        """
        Delete a key permanently.

        Args:
            key_id: Key to delete
            secure: Perform secure deletion

        Returns:
            True if deleted
        """
        with self._lock:
            metadata = self._keys.get(key_id)
            if not metadata:
                return False

            # Clear from cache with secure overwrite
            if key_id in self._key_cache:
                if secure:
                    key_data = self._key_cache[key_id]
                    # Overwrite with random data
                    self._key_cache[key_id] = secrets.token_bytes(len(key_data))
                del self._key_cache[key_id]

            # Delete from hardware
            if metadata.hardware_handle and self._hardware.is_available:
                self._hardware.delete_key(metadata.hardware_handle)

            # Remove metadata
            del self._keys[key_id]
            self._persist_key_metadata()

            logger.info(f"Key deleted: {key_id}")
            return True

    def list_keys(
        self,
        tier: Optional[EncryptionTier] = None,
        status: Optional[KeyStatus] = None,
    ) -> List[KeyMetadata]:
        """
        List keys with optional filters.

        Args:
            tier: Filter by tier
            status: Filter by status

        Returns:
            List of key metadata
        """
        keys = list(self._keys.values())

        if tier:
            keys = [k for k in keys if k.tier == tier]

        if status:
            keys = [k for k in keys if k.status == status]

        return keys

    def get_key_metadata(self, key_id: str) -> Optional[KeyMetadata]:
        """Get metadata for a key."""
        return self._keys.get(key_id)

    def has_hardware_binding(self) -> bool:
        """Check if hardware binding is available."""
        return self._hardware.is_available

    def _derive_key_pbkdf2(
        self,
        password: bytes,
        salt: bytes,
        length: int,
    ) -> bytes:
        """Derive key using PBKDF2."""
        return hashlib.pbkdf2_hmac(
            "sha256",
            password,
            salt,
            self.PBKDF2_ITERATIONS,
            dklen=length,
        )

    def _derive_key_argon2(
        self,
        password: bytes,
        salt: bytes,
        length: int,
    ) -> bytes:
        """Derive key using Argon2id."""
        try:
            from argon2.low_level import hash_secret_raw, Type
            return hash_secret_raw(
                password,
                salt,
                time_cost=self.ARGON2_TIME_COST,
                memory_cost=self.ARGON2_MEMORY_COST,
                parallelism=self.ARGON2_PARALLELISM,
                hash_len=length,
                type=Type.ID,
            )
        except ImportError:
            # Fallback to PBKDF2 if argon2 not available
            logger.warning("argon2-cffi not available, falling back to PBKDF2")
            return self._derive_key_pbkdf2(password, salt, length)

    def _derive_key_scrypt(
        self,
        password: bytes,
        salt: bytes,
        length: int,
    ) -> bytes:
        """Derive key using scrypt."""
        return hashlib.scrypt(
            password,
            salt=salt,
            n=self.SCRYPT_N,
            r=self.SCRYPT_R,
            p=self.SCRYPT_P,
            dklen=length,
        )

    def _get_or_create_master_salt(self) -> bytes:
        """Get or create master salt for key store."""
        if not self._key_store_path:
            return secrets.token_bytes(32)

        salt_path = self._key_store_path / "master.salt"
        if salt_path.exists():
            return salt_path.read_bytes()

        salt = secrets.token_bytes(32)
        self._key_store_path.mkdir(parents=True, exist_ok=True)
        salt_path.write_bytes(salt)
        return salt

    def _load_key_metadata(self) -> None:
        """Load key metadata from disk."""
        if not self._key_store_path:
            return

        metadata_path = self._key_store_path / "keys.json"
        if not metadata_path.exists():
            return

        try:
            with open(metadata_path, 'r') as f:
                data = json.load(f)

            for key_data in data.get("keys", []):
                metadata = KeyMetadata(
                    key_id=key_data["key_id"],
                    tier=EncryptionTier[key_data["tier"]],
                    binding=KeyBinding[key_data["binding"]],
                    created_at=datetime.fromisoformat(key_data["created_at"]),
                    expires_at=(
                        datetime.fromisoformat(key_data["expires_at"])
                        if key_data.get("expires_at") else None
                    ),
                    use_count=key_data.get("use_count", 0),
                    status=KeyStatus[key_data.get("status", "ACTIVE")],
                    hardware_handle=key_data.get("hardware_handle"),
                    metadata=key_data.get("metadata", {}),
                )
                self._keys[metadata.key_id] = metadata

            logger.info(f"Loaded {len(self._keys)} key metadata records")

        except Exception as e:
            logger.error(f"Failed to load key metadata: {e}")

    def _persist_key_metadata(self) -> None:
        """Persist key metadata to disk."""
        if not self._key_store_path:
            return

        self._key_store_path.mkdir(parents=True, exist_ok=True)
        metadata_path = self._key_store_path / "keys.json"

        try:
            data = {
                "keys": [m.to_dict() for m in self._keys.values()],
                "updated_at": datetime.now().isoformat(),
            }

            with open(metadata_path, 'w') as f:
                json.dump(data, f, indent=2)

        except Exception as e:
            logger.error(f"Failed to persist key metadata: {e}")
