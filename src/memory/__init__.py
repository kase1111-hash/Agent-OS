"""
Agent OS Memory Vault Module

Encrypted, consent-gated persistent storage with four classification tiers.

Features:
- AES-256-GCM encryption
- Four encryption tiers (Working/Private/Sealed/Vaulted)
- Hardware key binding support
- Consent verification layer
- Right-to-delete propagation
- Genesis proof system
"""

from .profiles import (
    EncryptionTier,
    EncryptionProfile,
    KeyDerivation,
    KeyBinding,
    ProfileManager,
    WORKING_PROFILE,
    PRIVATE_PROFILE,
    SEALED_PROFILE,
    VAULTED_PROFILE,
)
from .keys import (
    KeyManager,
    KeyMetadata,
    KeyStatus,
    DerivedKey,
    HardwareBindingInterface,
)
from .storage import (
    BlobStorage,
    BlobMetadata,
    BlobType,
    BlobStatus,
    EncryptedBlob,
)
from .index import (
    VaultIndex,
    ConsentRecord,
    AccessLogEntry,
    AccessType,
)
from .consent import (
    ConsentManager,
    ConsentType,
    ConsentOperation,
    ConsentStatus,
    ConsentRequest,
    ConsentDecision,
    ConsentPolicy,
    create_default_observation_contract,
    create_explicit_learning_contract,
)
from .deletion import (
    DeletionManager,
    DeletionRequest,
    DeletionResult,
    DeletionStatus,
    DeletionScope,
    TTLEnforcer,
)
from .genesis import (
    GenesisProofSystem,
    GenesisRecord,
    IntegrityProof,
)
from .vault import (
    MemoryVault,
    VaultConfig,
    StoreResult,
    RetrieveResult,
    create_vault,
)


__all__ = [
    # Profiles
    "EncryptionTier",
    "EncryptionProfile",
    "KeyDerivation",
    "KeyBinding",
    "ProfileManager",
    "WORKING_PROFILE",
    "PRIVATE_PROFILE",
    "SEALED_PROFILE",
    "VAULTED_PROFILE",
    # Keys
    "KeyManager",
    "KeyMetadata",
    "KeyStatus",
    "DerivedKey",
    "HardwareBindingInterface",
    # Storage
    "BlobStorage",
    "BlobMetadata",
    "BlobType",
    "BlobStatus",
    "EncryptedBlob",
    # Index
    "VaultIndex",
    "ConsentRecord",
    "AccessLogEntry",
    "AccessType",
    # Consent
    "ConsentManager",
    "ConsentType",
    "ConsentOperation",
    "ConsentStatus",
    "ConsentRequest",
    "ConsentDecision",
    "ConsentPolicy",
    "create_default_observation_contract",
    "create_explicit_learning_contract",
    # Deletion
    "DeletionManager",
    "DeletionRequest",
    "DeletionResult",
    "DeletionStatus",
    "DeletionScope",
    "TTLEnforcer",
    # Genesis
    "GenesisProofSystem",
    "GenesisRecord",
    "IntegrityProof",
    # Vault (main API)
    "MemoryVault",
    "VaultConfig",
    "StoreResult",
    "RetrieveResult",
    "create_vault",
]
