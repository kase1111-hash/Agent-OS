"""
Agent OS Memory Vault Encrypted Blob Storage

Implements AES-256-GCM encrypted blob storage with:
- Secure encryption/decryption
- Blob integrity verification
- Chunked storage for large files
- Secure deletion
"""

import os
import secrets
import hashlib
import json
import logging
import struct
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple, BinaryIO, Iterator
from enum import Enum, auto
import threading

from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.exceptions import InvalidTag

from .profiles import EncryptionTier, ProfileManager
from .keys import KeyManager, DerivedKey


logger = logging.getLogger(__name__)


# Constants
CHUNK_SIZE = 64 * 1024  # 64KB chunks for large files
NONCE_SIZE = 12  # 96 bits for GCM
TAG_SIZE = 16  # 128 bits for GCM authentication tag
BLOB_MAGIC = b'AGENTBLOB'
BLOB_VERSION = 1


class BlobType(Enum):
    """Type of stored blob."""
    BINARY = auto()
    TEXT = auto()
    JSON = auto()
    STREAM = auto()


class BlobStatus(Enum):
    """Status of a blob."""
    ACTIVE = auto()
    SEALED = auto()
    PENDING_DELETE = auto()
    DELETED = auto()
    CORRUPTED = auto()


@dataclass
class BlobMetadata:
    """Metadata for a stored blob."""
    blob_id: str
    key_id: str
    tier: EncryptionTier
    blob_type: BlobType
    size_bytes: int
    encrypted_size: int
    content_hash: str  # SHA-256 of plaintext
    created_at: datetime
    modified_at: datetime
    accessed_at: Optional[datetime] = None
    access_count: int = 0
    status: BlobStatus = BlobStatus.ACTIVE
    consent_id: Optional[str] = None
    ttl_seconds: Optional[int] = None
    tags: List[str] = field(default_factory=list)
    custom_metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "blob_id": self.blob_id,
            "key_id": self.key_id,
            "tier": self.tier.name,
            "blob_type": self.blob_type.name,
            "size_bytes": self.size_bytes,
            "encrypted_size": self.encrypted_size,
            "content_hash": self.content_hash,
            "created_at": self.created_at.isoformat(),
            "modified_at": self.modified_at.isoformat(),
            "accessed_at": self.accessed_at.isoformat() if self.accessed_at else None,
            "access_count": self.access_count,
            "status": self.status.name,
            "consent_id": self.consent_id,
            "ttl_seconds": self.ttl_seconds,
            "tags": self.tags,
            "custom_metadata": self.custom_metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BlobMetadata":
        return cls(
            blob_id=data["blob_id"],
            key_id=data["key_id"],
            tier=EncryptionTier[data["tier"]],
            blob_type=BlobType[data["blob_type"]],
            size_bytes=data["size_bytes"],
            encrypted_size=data["encrypted_size"],
            content_hash=data["content_hash"],
            created_at=datetime.fromisoformat(data["created_at"]),
            modified_at=datetime.fromisoformat(data["modified_at"]),
            accessed_at=(
                datetime.fromisoformat(data["accessed_at"])
                if data.get("accessed_at") else None
            ),
            access_count=data.get("access_count", 0),
            status=BlobStatus[data.get("status", "ACTIVE")],
            consent_id=data.get("consent_id"),
            ttl_seconds=data.get("ttl_seconds"),
            tags=data.get("tags", []),
            custom_metadata=data.get("custom_metadata", {}),
        )


@dataclass
class EncryptedBlob:
    """Container for encrypted blob data."""
    nonce: bytes
    ciphertext: bytes
    metadata: BlobMetadata

    def to_bytes(self) -> bytes:
        """Serialize encrypted blob to bytes."""
        meta_json = json.dumps(self.metadata.to_dict()).encode()
        meta_len = len(meta_json)

        # Format: MAGIC | VERSION | META_LEN | META | NONCE | CIPHERTEXT
        return (
            BLOB_MAGIC +
            struct.pack('<B', BLOB_VERSION) +
            struct.pack('<I', meta_len) +
            meta_json +
            self.nonce +
            self.ciphertext
        )

    @classmethod
    def from_bytes(cls, data: bytes) -> "EncryptedBlob":
        """Deserialize encrypted blob from bytes."""
        if not data.startswith(BLOB_MAGIC):
            raise ValueError("Invalid blob format: missing magic bytes")

        offset = len(BLOB_MAGIC)
        version = struct.unpack('<B', data[offset:offset + 1])[0]
        offset += 1

        if version != BLOB_VERSION:
            raise ValueError(f"Unsupported blob version: {version}")

        meta_len = struct.unpack('<I', data[offset:offset + 4])[0]
        offset += 4

        meta_json = data[offset:offset + meta_len]
        offset += meta_len

        nonce = data[offset:offset + NONCE_SIZE]
        offset += NONCE_SIZE

        ciphertext = data[offset:]

        metadata = BlobMetadata.from_dict(json.loads(meta_json))

        return cls(nonce=nonce, ciphertext=ciphertext, metadata=metadata)


class BlobStorage:
    """
    Encrypted blob storage for the Memory Vault.

    Features:
    - AES-256-GCM encryption
    - Integrity verification
    - Secure deletion
    - Large file chunking
    - Metadata persistence
    """

    def __init__(
        self,
        storage_path: Path,
        key_manager: KeyManager,
        profile_manager: Optional[ProfileManager] = None,
    ):
        """
        Initialize blob storage.

        Args:
            storage_path: Path to blob storage directory
            key_manager: Key manager instance
            profile_manager: Profile manager instance
        """
        self._storage_path = storage_path
        self._key_manager = key_manager
        self._profile_manager = profile_manager or ProfileManager()

        self._blobs_dir = storage_path / "blobs"
        self._chunks_dir = storage_path / "chunks"
        self._metadata_dir = storage_path / "metadata"

        self._lock = threading.RLock()
        self._cache: Dict[str, BlobMetadata] = {}

        self._ensure_directories()
        self._load_metadata_cache()

    def _ensure_directories(self) -> None:
        """Create storage directories if they don't exist."""
        self._blobs_dir.mkdir(parents=True, exist_ok=True)
        self._chunks_dir.mkdir(parents=True, exist_ok=True)
        self._metadata_dir.mkdir(parents=True, exist_ok=True)

    def store(
        self,
        data: bytes,
        tier: EncryptionTier,
        blob_type: BlobType = BlobType.BINARY,
        consent_id: Optional[str] = None,
        tags: Optional[List[str]] = None,
        custom_metadata: Optional[Dict[str, Any]] = None,
        ttl_seconds: Optional[int] = None,
    ) -> BlobMetadata:
        """
        Store encrypted data.

        Args:
            data: Data to store
            tier: Encryption tier
            blob_type: Type of data
            consent_id: Associated consent record
            tags: Optional tags
            custom_metadata: Optional custom metadata
            ttl_seconds: Optional time-to-live

        Returns:
            BlobMetadata for stored blob
        """
        with self._lock:
            # Generate blob ID
            blob_id = f"{tier.name.lower()}_{secrets.token_hex(16)}"

            # Generate or get key for this tier
            derived_key = self._key_manager.generate_key(tier, purpose=f"blob_{blob_id}")

            # Calculate content hash
            content_hash = hashlib.sha256(data).hexdigest()

            # Encrypt data
            nonce = secrets.token_bytes(NONCE_SIZE)
            aesgcm = AESGCM(derived_key.key)
            ciphertext = aesgcm.encrypt(nonce, data, None)

            # Create metadata
            now = datetime.now()
            metadata = BlobMetadata(
                blob_id=blob_id,
                key_id=derived_key.key_id,
                tier=tier,
                blob_type=blob_type,
                size_bytes=len(data),
                encrypted_size=len(ciphertext),
                content_hash=content_hash,
                created_at=now,
                modified_at=now,
                consent_id=consent_id,
                ttl_seconds=ttl_seconds,
                tags=tags or [],
                custom_metadata=custom_metadata or {},
            )

            # Create encrypted blob
            encrypted_blob = EncryptedBlob(
                nonce=nonce,
                ciphertext=ciphertext,
                metadata=metadata,
            )

            # Write to storage
            blob_path = self._get_blob_path(blob_id)
            blob_path.write_bytes(encrypted_blob.to_bytes())

            # Cache metadata
            self._cache[blob_id] = metadata
            self._persist_metadata(metadata)

            logger.info(f"Stored blob: {blob_id} (tier={tier.name}, size={len(data)})")
            return metadata

    def store_text(
        self,
        text: str,
        tier: EncryptionTier,
        **kwargs,
    ) -> BlobMetadata:
        """Store text as encrypted blob."""
        return self.store(
            text.encode('utf-8'),
            tier,
            blob_type=BlobType.TEXT,
            **kwargs,
        )

    def store_json(
        self,
        data: Any,
        tier: EncryptionTier,
        **kwargs,
    ) -> BlobMetadata:
        """Store JSON-serializable data as encrypted blob."""
        return self.store(
            json.dumps(data).encode('utf-8'),
            tier,
            blob_type=BlobType.JSON,
            **kwargs,
        )

    def store_stream(
        self,
        stream: BinaryIO,
        tier: EncryptionTier,
        chunk_size: int = CHUNK_SIZE,
        **kwargs,
    ) -> BlobMetadata:
        """
        Store large data from a stream using chunked encryption.

        Args:
            stream: Binary stream to read from
            tier: Encryption tier
            chunk_size: Size of chunks
            **kwargs: Additional metadata arguments

        Returns:
            BlobMetadata for stored blob
        """
        with self._lock:
            blob_id = f"{tier.name.lower()}_{secrets.token_hex(16)}"
            derived_key = self._key_manager.generate_key(tier, purpose=f"blob_{blob_id}")

            # Create chunk directory
            chunk_dir = self._chunks_dir / blob_id
            chunk_dir.mkdir(parents=True, exist_ok=True)

            total_size = 0
            encrypted_size = 0
            chunk_index = 0
            hasher = hashlib.sha256()

            aesgcm = AESGCM(derived_key.key)

            # Process chunks
            while True:
                chunk = stream.read(chunk_size)
                if not chunk:
                    break

                hasher.update(chunk)
                total_size += len(chunk)

                # Encrypt chunk
                nonce = secrets.token_bytes(NONCE_SIZE)
                ciphertext = aesgcm.encrypt(nonce, chunk, None)
                encrypted_size += len(nonce) + len(ciphertext)

                # Write chunk
                chunk_path = chunk_dir / f"{chunk_index:08d}.chunk"
                chunk_path.write_bytes(nonce + ciphertext)
                chunk_index += 1

            # Create metadata
            now = datetime.now()
            metadata = BlobMetadata(
                blob_id=blob_id,
                key_id=derived_key.key_id,
                tier=tier,
                blob_type=BlobType.STREAM,
                size_bytes=total_size,
                encrypted_size=encrypted_size,
                content_hash=hasher.hexdigest(),
                created_at=now,
                modified_at=now,
                consent_id=kwargs.get("consent_id"),
                ttl_seconds=kwargs.get("ttl_seconds"),
                tags=kwargs.get("tags", []),
                custom_metadata={
                    **kwargs.get("custom_metadata", {}),
                    "chunk_count": chunk_index,
                    "chunk_size": chunk_size,
                },
            )

            self._cache[blob_id] = metadata
            self._persist_metadata(metadata)

            logger.info(
                f"Stored stream blob: {blob_id} "
                f"(tier={tier.name}, size={total_size}, chunks={chunk_index})"
            )
            return metadata

    def retrieve(self, blob_id: str) -> Optional[bytes]:
        """
        Retrieve and decrypt a blob.

        Args:
            blob_id: Blob identifier

        Returns:
            Decrypted data or None if not found
        """
        with self._lock:
            metadata = self._cache.get(blob_id)
            if not metadata:
                return None

            if metadata.status not in (BlobStatus.ACTIVE, BlobStatus.SEALED):
                logger.warning(f"Attempted access to non-active blob: {blob_id}")
                return None

            # Get decryption key
            key = self._key_manager.get_key(metadata.key_id)
            if not key:
                logger.error(f"Key not found for blob: {blob_id}")
                return None

            try:
                if metadata.blob_type == BlobType.STREAM:
                    return self._retrieve_stream_blob(blob_id, key, metadata)
                else:
                    return self._retrieve_single_blob(blob_id, key, metadata)

            except InvalidTag:
                logger.error(f"Decryption failed (integrity check): {blob_id}")
                metadata.status = BlobStatus.CORRUPTED
                self._persist_metadata(metadata)
                return None

    def _retrieve_single_blob(
        self,
        blob_id: str,
        key: bytes,
        metadata: BlobMetadata,
    ) -> Optional[bytes]:
        """Retrieve a single (non-chunked) blob."""
        blob_path = self._get_blob_path(blob_id)
        if not blob_path.exists():
            return None

        encrypted_blob = EncryptedBlob.from_bytes(blob_path.read_bytes())

        aesgcm = AESGCM(key)
        plaintext = aesgcm.decrypt(
            encrypted_blob.nonce,
            encrypted_blob.ciphertext,
            None,
        )

        # Verify content hash
        if hashlib.sha256(plaintext).hexdigest() != metadata.content_hash:
            logger.error(f"Content hash mismatch: {blob_id}")
            metadata.status = BlobStatus.CORRUPTED
            self._persist_metadata(metadata)
            return None

        # Update access tracking
        metadata.accessed_at = datetime.now()
        metadata.access_count += 1
        self._persist_metadata(metadata)

        return plaintext

    def _retrieve_stream_blob(
        self,
        blob_id: str,
        key: bytes,
        metadata: BlobMetadata,
    ) -> Optional[bytes]:
        """Retrieve a chunked stream blob."""
        chunk_dir = self._chunks_dir / blob_id
        if not chunk_dir.exists():
            return None

        chunk_count = metadata.custom_metadata.get("chunk_count", 0)
        aesgcm = AESGCM(key)
        hasher = hashlib.sha256()
        result = bytearray()

        for i in range(chunk_count):
            chunk_path = chunk_dir / f"{i:08d}.chunk"
            if not chunk_path.exists():
                logger.error(f"Missing chunk {i} for blob: {blob_id}")
                return None

            chunk_data = chunk_path.read_bytes()
            nonce = chunk_data[:NONCE_SIZE]
            ciphertext = chunk_data[NONCE_SIZE:]

            plaintext = aesgcm.decrypt(nonce, ciphertext, None)
            hasher.update(plaintext)
            result.extend(plaintext)

        # Verify content hash
        if hasher.hexdigest() != metadata.content_hash:
            logger.error(f"Content hash mismatch: {blob_id}")
            metadata.status = BlobStatus.CORRUPTED
            self._persist_metadata(metadata)
            return None

        # Update access tracking
        metadata.accessed_at = datetime.now()
        metadata.access_count += 1
        self._persist_metadata(metadata)

        return bytes(result)

    def retrieve_text(self, blob_id: str) -> Optional[str]:
        """Retrieve and decode text blob."""
        data = self.retrieve(blob_id)
        return data.decode('utf-8') if data else None

    def retrieve_json(self, blob_id: str) -> Optional[Any]:
        """Retrieve and parse JSON blob."""
        data = self.retrieve(blob_id)
        return json.loads(data) if data else None

    def retrieve_stream(self, blob_id: str) -> Iterator[bytes]:
        """
        Retrieve blob as a stream of chunks.

        Args:
            blob_id: Blob identifier

        Yields:
            Decrypted chunks
        """
        with self._lock:
            metadata = self._cache.get(blob_id)
            if not metadata or metadata.blob_type != BlobType.STREAM:
                return

            key = self._key_manager.get_key(metadata.key_id)
            if not key:
                return

            chunk_dir = self._chunks_dir / blob_id
            chunk_count = metadata.custom_metadata.get("chunk_count", 0)
            aesgcm = AESGCM(key)

            for i in range(chunk_count):
                chunk_path = chunk_dir / f"{i:08d}.chunk"
                if not chunk_path.exists():
                    return

                chunk_data = chunk_path.read_bytes()
                nonce = chunk_data[:NONCE_SIZE]
                ciphertext = chunk_data[NONCE_SIZE:]

                try:
                    yield aesgcm.decrypt(nonce, ciphertext, None)
                except InvalidTag:
                    return

    def delete(self, blob_id: str, secure: bool = True) -> bool:
        """
        Delete a blob.

        Args:
            blob_id: Blob identifier
            secure: Perform secure deletion (overwrite before delete)

        Returns:
            True if deleted
        """
        with self._lock:
            metadata = self._cache.get(blob_id)
            if not metadata:
                return False

            # Mark as pending delete
            metadata.status = BlobStatus.PENDING_DELETE
            self._persist_metadata(metadata)

            # Delete blob file
            blob_path = self._get_blob_path(blob_id)
            if blob_path.exists():
                if secure:
                    self._secure_delete_file(blob_path)
                else:
                    blob_path.unlink()

            # Delete chunks if stream blob
            if metadata.blob_type == BlobType.STREAM:
                chunk_dir = self._chunks_dir / blob_id
                if chunk_dir.exists():
                    if secure:
                        for chunk_path in chunk_dir.iterdir():
                            self._secure_delete_file(chunk_path)
                    shutil.rmtree(chunk_dir)

            # Delete associated key
            self._key_manager.delete_key(metadata.key_id, secure=secure)

            # Update status
            metadata.status = BlobStatus.DELETED
            self._persist_metadata(metadata)

            # Remove from cache
            del self._cache[blob_id]

            logger.info(f"Deleted blob: {blob_id} (secure={secure})")
            return True

    def seal(self, blob_id: str) -> bool:
        """
        Seal a blob (prevent further access until explicitly unsealed).

        Args:
            blob_id: Blob identifier

        Returns:
            True if sealed
        """
        with self._lock:
            metadata = self._cache.get(blob_id)
            if not metadata:
                return False

            metadata.status = BlobStatus.SEALED
            self._persist_metadata(metadata)

            logger.info(f"Sealed blob: {blob_id}")
            return True

    def unseal(self, blob_id: str) -> bool:
        """
        Unseal a sealed blob.

        Args:
            blob_id: Blob identifier

        Returns:
            True if unsealed
        """
        with self._lock:
            metadata = self._cache.get(blob_id)
            if not metadata or metadata.status != BlobStatus.SEALED:
                return False

            metadata.status = BlobStatus.ACTIVE
            self._persist_metadata(metadata)

            logger.info(f"Unsealed blob: {blob_id}")
            return True

    def get_metadata(self, blob_id: str) -> Optional[BlobMetadata]:
        """Get blob metadata."""
        return self._cache.get(blob_id)

    def list_blobs(
        self,
        tier: Optional[EncryptionTier] = None,
        status: Optional[BlobStatus] = None,
        consent_id: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> List[BlobMetadata]:
        """
        List blobs with optional filters.

        Args:
            tier: Filter by tier
            status: Filter by status
            consent_id: Filter by consent
            tags: Filter by tags (any match)

        Returns:
            List of matching blob metadata
        """
        blobs = list(self._cache.values())

        if tier:
            blobs = [b for b in blobs if b.tier == tier]

        if status:
            blobs = [b for b in blobs if b.status == status]

        if consent_id:
            blobs = [b for b in blobs if b.consent_id == consent_id]

        if tags:
            tag_set = set(tags)
            blobs = [b for b in blobs if tag_set & set(b.tags)]

        return blobs

    def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage statistics."""
        stats = {
            "total_blobs": len(self._cache),
            "by_tier": {},
            "by_status": {},
            "total_size_bytes": 0,
            "encrypted_size_bytes": 0,
        }

        for metadata in self._cache.values():
            tier_name = metadata.tier.name
            status_name = metadata.status.name

            stats["by_tier"][tier_name] = stats["by_tier"].get(tier_name, 0) + 1
            stats["by_status"][status_name] = stats["by_status"].get(status_name, 0) + 1
            stats["total_size_bytes"] += metadata.size_bytes
            stats["encrypted_size_bytes"] += metadata.encrypted_size

        return stats

    def _get_blob_path(self, blob_id: str) -> Path:
        """Get path for a blob file."""
        # Use first 4 chars for sharding
        shard = blob_id[:4]
        shard_dir = self._blobs_dir / shard
        shard_dir.mkdir(parents=True, exist_ok=True)
        return shard_dir / f"{blob_id}.blob"

    def _load_metadata_cache(self) -> None:
        """Load all blob metadata into cache."""
        for meta_path in self._metadata_dir.rglob("*.json"):
            try:
                with open(meta_path, 'r') as f:
                    data = json.load(f)
                metadata = BlobMetadata.from_dict(data)
                self._cache[metadata.blob_id] = metadata
            except Exception as e:
                logger.error(f"Failed to load metadata {meta_path}: {e}")

        logger.info(f"Loaded {len(self._cache)} blob metadata records")

    def _persist_metadata(self, metadata: BlobMetadata) -> None:
        """Persist blob metadata to disk."""
        shard = metadata.blob_id[:4]
        shard_dir = self._metadata_dir / shard
        shard_dir.mkdir(parents=True, exist_ok=True)

        meta_path = shard_dir / f"{metadata.blob_id}.json"
        with open(meta_path, 'w') as f:
            json.dump(metadata.to_dict(), f, indent=2)

    def _secure_delete_file(self, path: Path) -> None:
        """Securely delete a file by overwriting before removal."""
        if not path.exists():
            return

        file_size = path.stat().st_size

        # Overwrite with random data
        with open(path, 'r+b') as f:
            f.write(secrets.token_bytes(file_size))
            f.flush()
            os.fsync(f.fileno())

        # Overwrite with zeros
        with open(path, 'r+b') as f:
            f.write(b'\x00' * file_size)
            f.flush()
            os.fsync(f.fileno())

        # Delete
        path.unlink()
