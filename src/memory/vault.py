"""
Agent OS Memory Vault API

Main facade for the Memory Vault system providing:
- Unified API for all vault operations
- Consent-gated storage
- Encrypted persistence
- Right-to-delete support
"""

import logging
import secrets
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional

from .consent import (
    ConsentDecision,
    ConsentManager,
    ConsentOperation,
    ConsentPolicy,
    ConsentStatus,
)
from .deletion import DeletionManager, DeletionResult, DeletionScope, TTLEnforcer
from .genesis import GenesisProofSystem, GenesisRecord, IntegrityProof
from .index import AccessType, ConsentRecord, VaultIndex
from .keys import KeyManager, KeyStatus
from .profiles import (
    PRIVATE_PROFILE,
    SEALED_PROFILE,
    VAULTED_PROFILE,
    WORKING_PROFILE,
    EncryptionProfile,
    EncryptionTier,
    ProfileManager,
)
from .storage import BlobMetadata, BlobStatus, BlobStorage, BlobType

logger = logging.getLogger(__name__)


@dataclass
class VaultConfig:
    """Configuration for Memory Vault."""

    vault_path: Path
    vault_id: Optional[str] = None
    owner: str = "system"
    enable_hardware_binding: bool = False
    enable_ttl_enforcement: bool = True
    ttl_check_interval: int = 3600
    strict_consent: bool = True
    constitution_path: Optional[Path] = None
    master_password: Optional[str] = None


@dataclass
class StoreResult:
    """Result of a store operation."""

    success: bool
    blob_id: Optional[str] = None
    consent_id: Optional[str] = None
    tier: Optional[EncryptionTier] = None
    error: Optional[str] = None
    metadata: Optional[BlobMetadata] = None


@dataclass
class RetrieveResult:
    """Result of a retrieve operation."""

    success: bool
    data: Optional[bytes] = None
    text: Optional[str] = None
    json_data: Optional[Any] = None
    blob_id: Optional[str] = None
    error: Optional[str] = None
    metadata: Optional[BlobMetadata] = None


class MemoryVault:
    """
    Memory Vault - Encrypted, Consent-Gated Persistent Storage.

    This is the main API for the Agent OS memory system.

    Features:
    - Four encryption tiers (Working/Private/Sealed/Vaulted)
    - AES-256-GCM encryption
    - Consent verification for all writes
    - Right-to-delete propagation
    - Genesis proofs for audit
    - Hardware key binding (when available)
    """

    def __init__(
        self,
        config: VaultConfig,
        consent_prompt: Optional[Callable[[Any], bool]] = None,
    ):
        """
        Initialize Memory Vault.

        Args:
            config: Vault configuration
            consent_prompt: Callback to prompt user for consent
        """
        self._config = config
        self._consent_prompt = consent_prompt

        # Paths
        self._vault_path = config.vault_path
        self._blobs_path = self._vault_path / "store"
        self._keys_path = self._vault_path / "crypto"
        self._index_path = self._vault_path / "index.db"
        self._proofs_path = self._vault_path / "proofs"

        # Components (initialized in initialize())
        self._profile_manager: Optional[ProfileManager] = None
        self._key_manager: Optional[KeyManager] = None
        self._storage: Optional[BlobStorage] = None
        self._index: Optional[VaultIndex] = None
        self._consent_manager: Optional[ConsentManager] = None
        self._deletion_manager: Optional[DeletionManager] = None
        self._ttl_enforcer: Optional[TTLEnforcer] = None
        self._genesis: Optional[GenesisProofSystem] = None

        self._initialized = False
        self._vault_id: Optional[str] = None

    def initialize(self) -> bool:
        """
        Initialize the vault.

        Returns:
            True if initialization successful
        """
        try:
            # Ensure directories exist
            self._vault_path.mkdir(parents=True, exist_ok=True)
            self._blobs_path.mkdir(parents=True, exist_ok=True)
            self._keys_path.mkdir(parents=True, exist_ok=True)
            self._proofs_path.mkdir(parents=True, exist_ok=True)

            # Initialize profile manager
            self._profile_manager = ProfileManager()

            # Initialize key manager
            hardware_config = {
                "simulate_hardware": self._config.enable_hardware_binding,
            }
            self._key_manager = KeyManager(
                key_store_path=self._keys_path,
                profile_manager=self._profile_manager,
                hardware_config=hardware_config,
            )
            self._key_manager.initialize(self._config.master_password)

            # Initialize index
            self._index = VaultIndex(self._index_path)

            # Initialize storage
            self._storage = BlobStorage(
                storage_path=self._blobs_path,
                key_manager=self._key_manager,
                profile_manager=self._profile_manager,
            )

            # Initialize consent manager
            policy = ConsentPolicy()
            if self._config.strict_consent:
                policy.always_require.add(ConsentOperation.READ)

            self._consent_manager = ConsentManager(
                index=self._index,
                policy=policy,
                prompt_callback=self._consent_prompt,
            )

            # Initialize deletion manager
            self._deletion_manager = DeletionManager(
                storage=self._storage,
                index=self._index,
                key_manager=self._key_manager,
            )

            # Initialize TTL enforcer
            if self._config.enable_ttl_enforcement:
                self._ttl_enforcer = TTLEnforcer(
                    deletion_manager=self._deletion_manager,
                    index=self._index,
                    check_interval=self._config.ttl_check_interval,
                )
                self._ttl_enforcer.start()

            # Initialize genesis proof system
            self._genesis = GenesisProofSystem(
                index=self._index,
                proof_dir=self._proofs_path,
            )

            # Create or verify genesis
            if not self._genesis.get_genesis():
                self._vault_id = self._config.vault_id or f"vault_{secrets.token_hex(8)}"
                self._genesis.create_genesis(
                    vault_id=self._vault_id,
                    created_by=self._config.owner,
                    encryption_profiles=["WORKING", "PRIVATE", "SEALED", "VAULTED"],
                    hardware_bound=self._config.enable_hardware_binding,
                    constitution_path=self._config.constitution_path,
                )
            else:
                self._vault_id = self._genesis.get_genesis().vault_id
                is_valid, message = self._genesis.verify_genesis()
                if not is_valid:
                    logger.error(f"Genesis verification failed: {message}")
                    return False

            self._initialized = True
            logger.info(f"Memory Vault initialized: {self._vault_id}")
            return True

        except Exception as e:
            logger.error(f"Vault initialization failed: {e}")
            return False

    def shutdown(self) -> None:
        """Shutdown the vault."""
        if self._ttl_enforcer:
            self._ttl_enforcer.stop()

        if self._key_manager:
            self._key_manager.shutdown()

        if self._index:
            self._index.close()

        self._initialized = False
        logger.info("Memory Vault shutdown complete")

    def store(
        self,
        data: bytes,
        tier: EncryptionTier = EncryptionTier.PRIVATE,
        requestor: str = "user",
        purpose: str = "storage",
        tags: Optional[List[str]] = None,
        ttl: Optional[timedelta] = None,
        custom_metadata: Optional[Dict[str, Any]] = None,
        blob_type: BlobType = BlobType.BINARY,
    ) -> StoreResult:
        """
        Store data in the vault.

        Args:
            data: Data to store
            tier: Encryption tier
            requestor: Who is storing
            purpose: Purpose of storage
            tags: Optional tags
            ttl: Optional time-to-live
            custom_metadata: Optional metadata
            blob_type: Type of blob (BINARY, TEXT, JSON)

        Returns:
            StoreResult
        """
        if not self._initialized:
            return StoreResult(success=False, error="Vault not initialized")

        # Request consent
        decision = self._consent_manager.request_consent(
            operation=ConsentOperation.WRITE,
            tier=tier,
            scope=purpose,
            requestor=requestor,
            purpose=purpose,
        )

        if decision.status != ConsentStatus.GRANTED:
            return StoreResult(
                success=False,
                error=f"Consent not granted: {decision.reason}",
            )

        try:
            # Store blob
            metadata = self._storage.store(
                data=data,
                tier=tier,
                blob_type=blob_type,
                consent_id=decision.consent_id,
                tags=tags,
                ttl_seconds=int(ttl.total_seconds()) if ttl else None,
                custom_metadata=custom_metadata,
            )

            # Index blob
            self._index.index_blob(metadata)

            # Log access
            self._index.log_access(
                blob_id=metadata.blob_id,
                access_type=AccessType.WRITE,
                accessor=requestor,
                success=True,
                details={"tier": tier.name, "size": len(data)},
            )

            return StoreResult(
                success=True,
                blob_id=metadata.blob_id,
                consent_id=decision.consent_id,
                tier=tier,
                metadata=metadata,
            )

        except Exception as e:
            logger.error(f"Store failed: {e}")
            return StoreResult(success=False, error=str(e))

    def store_text(
        self,
        text: str,
        tier: EncryptionTier = EncryptionTier.PRIVATE,
        **kwargs,
    ) -> StoreResult:
        """Store text in the vault."""
        return self.store(text.encode("utf-8"), tier, blob_type=BlobType.TEXT, **kwargs)

    def store_json(
        self,
        data: Any,
        tier: EncryptionTier = EncryptionTier.PRIVATE,
        **kwargs,
    ) -> StoreResult:
        """Store JSON data in the vault."""
        import json

        return self.store(json.dumps(data).encode("utf-8"), tier, blob_type=BlobType.JSON, **kwargs)

    def retrieve(
        self,
        blob_id: str,
        requestor: str = "user",
    ) -> RetrieveResult:
        """
        Retrieve data from the vault.

        Args:
            blob_id: Blob to retrieve
            requestor: Who is retrieving

        Returns:
            RetrieveResult
        """
        if not self._initialized:
            return RetrieveResult(success=False, error="Vault not initialized")

        # Get metadata
        metadata = self._storage.get_metadata(blob_id)
        if not metadata:
            return RetrieveResult(
                success=False,
                blob_id=blob_id,
                error="Blob not found",
            )

        # Verify consent if required
        if metadata.consent_id:
            decision = self._consent_manager.verify_consent(
                consent_id=metadata.consent_id,
                operation=ConsentOperation.READ,
            )
            if decision.status != ConsentStatus.GRANTED:
                return RetrieveResult(
                    success=False,
                    blob_id=blob_id,
                    error=f"Consent verification failed: {decision.reason}",
                )

        try:
            data = self._storage.retrieve(blob_id)
            if data is None:
                return RetrieveResult(
                    success=False,
                    blob_id=blob_id,
                    error="Failed to retrieve data",
                )

            # Log access
            self._index.log_access(
                blob_id=blob_id,
                access_type=AccessType.READ,
                accessor=requestor,
                success=True,
            )

            result = RetrieveResult(
                success=True,
                data=data,
                blob_id=blob_id,
                metadata=metadata,
            )

            # Try to decode as text/JSON
            if metadata.blob_type == BlobType.TEXT:
                try:
                    result.text = data.decode("utf-8")
                except Exception:
                    pass
            elif metadata.blob_type == BlobType.JSON:
                try:
                    import json

                    result.json_data = json.loads(data)
                    result.text = data.decode("utf-8")
                except Exception:
                    pass

            return result

        except Exception as e:
            logger.error(f"Retrieve failed: {e}")
            self._index.log_access(
                blob_id=blob_id,
                access_type=AccessType.READ,
                accessor=requestor,
                success=False,
                details={"error": str(e)},
            )
            return RetrieveResult(
                success=False,
                blob_id=blob_id,
                error=str(e),
            )

    def delete(
        self,
        blob_id: str,
        requestor: str = "user",
        secure: bool = True,
    ) -> DeletionResult:
        """
        Delete data from the vault.

        Args:
            blob_id: Blob to delete
            requestor: Who is deleting
            secure: Use secure deletion

        Returns:
            DeletionResult
        """
        if not self._initialized:
            raise RuntimeError("Vault not initialized")

        return self._deletion_manager.delete_blob(
            blob_id=blob_id,
            requested_by=requestor,
            secure=secure,
        )

    def forget(
        self,
        consent_id: str,
        requestor: str = "user",
    ) -> DeletionResult:
        """
        Exercise right-to-forget for a consent scope.

        Deletes all data associated with a consent.

        Args:
            consent_id: Consent to forget
            requestor: Who is requesting

        Returns:
            DeletionResult
        """
        if not self._initialized:
            raise RuntimeError("Vault not initialized")

        # Revoke the consent
        self._consent_manager.revoke_consent(
            consent_id=consent_id,
            revoked_by=requestor,
            reason="Right to forget exercised",
        )

        # Delete all associated data
        return self._deletion_manager.delete_by_consent(
            consent_id=consent_id,
            requested_by=requestor,
            reason="Right to forget",
        )

    def seal(self, blob_id: str) -> bool:
        """Seal a blob (prevent access until unseal)."""
        if not self._initialized:
            return False

        success = self._storage.seal(blob_id)
        if success:
            self._index.log_access(
                blob_id=blob_id,
                access_type=AccessType.SEAL,
                accessor="system",
                success=True,
            )
        return success

    def unseal(self, blob_id: str) -> bool:
        """Unseal a sealed blob."""
        if not self._initialized:
            return False

        success = self._storage.unseal(blob_id)
        if success:
            self._index.log_access(
                blob_id=blob_id,
                access_type=AccessType.UNSEAL,
                accessor="system",
                success=True,
            )
        return success

    def grant_consent(
        self,
        scope: str,
        operations: List[ConsentOperation],
        granted_by: str = "user",
        duration: Optional[timedelta] = None,
    ) -> ConsentRecord:
        """
        Grant consent for operations.

        Args:
            scope: Scope of consent
            operations: Allowed operations
            granted_by: Who is granting
            duration: Duration of consent

        Returns:
            ConsentRecord
        """
        if not self._initialized:
            raise RuntimeError("Vault not initialized")

        return self._consent_manager.grant_consent(
            granted_by=granted_by,
            scope=scope,
            operations=operations,
            duration=duration,
        )

    def revoke_consent(
        self,
        consent_id: str,
        revoked_by: str = "user",
    ) -> bool:
        """Revoke a consent."""
        if not self._initialized:
            return False

        return self._consent_manager.revoke_consent(
            consent_id=consent_id,
            revoked_by=revoked_by,
        )

    def list_blobs(
        self,
        tier: Optional[EncryptionTier] = None,
        tags: Optional[List[str]] = None,
        consent_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[BlobMetadata]:
        """List blobs with optional filters."""
        if not self._initialized:
            return []

        return self._index.query_blobs(
            tier=tier,
            tags=tags,
            consent_id=consent_id,
            limit=limit,
        )

    def list_consents(
        self,
        scope: Optional[str] = None,
        active_only: bool = True,
    ) -> List[ConsentRecord]:
        """List consents."""
        if not self._initialized:
            return []

        return self._consent_manager.list_consents(
            scope=scope,
            active_only=active_only,
        )

    def get_metadata(self, blob_id: str) -> Optional[BlobMetadata]:
        """Get blob metadata."""
        if not self._initialized:
            return None
        return self._storage.get_metadata(blob_id)

    def create_integrity_proof(self) -> IntegrityProof:
        """Create an integrity proof for current state."""
        if not self._initialized:
            raise RuntimeError("Vault not initialized")
        return self._genesis.create_integrity_proof()

    def verify_integrity(self) -> tuple:
        """Verify vault integrity."""
        if not self._initialized:
            return False, "Vault not initialized"
        return self._genesis.verify_integrity()

    def get_genesis(self) -> Optional[GenesisRecord]:
        """Get the genesis record."""
        if not self._initialized:
            return None
        return self._genesis.get_genesis()

    def get_statistics(self) -> Dict[str, Any]:
        """Get vault statistics."""
        if not self._initialized:
            return {}

        index_stats = self._index.get_statistics()
        storage_stats = self._storage.get_storage_stats()
        consent_stats = self._consent_manager.get_statistics()

        return {
            "vault_id": self._vault_id,
            "initialized": self._initialized,
            "index": index_stats,
            "storage": storage_stats,
            "consents": consent_stats,
            "hardware_binding": self._key_manager.has_hardware_binding(),
        }

    @property
    def vault_id(self) -> Optional[str]:
        """Get vault ID."""
        return self._vault_id

    @property
    def is_initialized(self) -> bool:
        """Check if vault is initialized."""
        return self._initialized


def create_vault(
    path: Path,
    owner: str = "system",
    master_password: Optional[str] = None,
    constitution_path: Optional[Path] = None,
) -> MemoryVault:
    """
    Convenience function to create and initialize a vault.

    Args:
        path: Vault storage path
        owner: Vault owner
        master_password: Optional master password
        constitution_path: Path to constitution file

    Returns:
        Initialized MemoryVault
    """
    config = VaultConfig(
        vault_path=path,
        owner=owner,
        master_password=master_password,
        constitution_path=constitution_path,
    )

    vault = MemoryVault(config)
    if not vault.initialize():
        raise RuntimeError("Failed to initialize vault")

    return vault
