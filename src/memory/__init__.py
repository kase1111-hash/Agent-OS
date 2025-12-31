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

from .consent import (
    ConsentDecision,
    ConsentManager,
    ConsentOperation,
    ConsentPolicy,
    ConsentRequest,
    ConsentStatus,
    ConsentType,
    create_default_observation_contract,
    create_explicit_learning_contract,
)
from .deletion import (
    DeletionManager,
    DeletionRequest,
    DeletionResult,
    DeletionScope,
    DeletionStatus,
    TTLEnforcer,
)
from .genesis import (
    GenesisProofSystem,
    GenesisRecord,
    IntegrityProof,
)
from .index import (
    AccessLogEntry,
    AccessType,
    ConsentRecord,
    VaultIndex,
)
from .keys import (
    DerivedKey,
    HardwareBindingInterface,
    KeyManager,
    KeyMetadata,
    KeyStatus,
)
from .profiles import (
    PRIVATE_PROFILE,
    SEALED_PROFILE,
    VAULTED_PROFILE,
    WORKING_PROFILE,
    EncryptionProfile,
    EncryptionTier,
    KeyBinding,
    KeyDerivation,
    ProfileManager,
)
from .storage import (
    BlobMetadata,
    BlobStatus,
    BlobStorage,
    BlobType,
    EncryptedBlob,
)
from .vault import (
    MemoryVault,
    RetrieveResult,
    StoreResult,
    VaultConfig,
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
