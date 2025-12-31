"""
Post-Quantum Cryptography Module

Provides quantum-resistant cryptographic operations for Agent OS:
- ML-KEM (CRYSTALS-Kyber) for key encapsulation
- ML-DSA (CRYSTALS-Dilithium) for digital signatures
- Hybrid mode combining classical and post-quantum algorithms

This module implements NIST FIPS 203/204/205 standards for post-quantum security.

Security Levels:
- ML-KEM-768: ~AES-192 equivalent (recommended)
- ML-KEM-1024: ~AES-256 equivalent (high security)
- ML-DSA-65: ~AES-192 equivalent (recommended)
- ML-DSA-87: ~AES-256 equivalent (high security)

Usage:
    from federation.pq import HybridKeyExchange, HybridSigner

    # Hybrid key exchange (X25519 + ML-KEM)
    kex = HybridKeyExchange()
    public_key, private_key = kex.generate_keypair()

    # Hybrid signing (Ed25519 + ML-DSA)
    signer = HybridSigner()
    keypair = signer.generate_keypair()
    signature = signer.sign(message, keypair.private_key)
"""

from .audit import AuditConfig as CryptoAuditConfig
from .audit import (
    AuditEventType,
    AuditMetrics,
    AuditSeverity,
    ComplianceStandard,
    CryptoAuditEvent,
    CryptoAuditLogger,
    configure_audit_logger,
    create_production_audit_config,
    get_audit_logger,
)
from .backup import (
    BackupConfig,
    BackupFormat,
    BackupStatus,
    KeyBackup,
    KeyBackupManager,
    KeyShare,
    RecoveryMethod,
    ShamirSecretSharing,
    create_backup_manager,
    create_production_backup_config,
)
from .config import (
    AlgorithmConfig,
    Environment,
    HSMSettings,
    KeyPolicyConfig,
    PerformanceConfig,
    PQConfig,
    SecurityConfig,
    get_pq_config,
    reset_pq_config,
    set_pq_config,
)

# Phase 4: Production Hardening
from .hsm import (
    HSMConfig,
    HSMKeyHandle,
    HSMProvider,
    HSMSecurityLevel,
    HSMType,
    KeyState,
    PKCS11HSMProvider,
    PQAlgorithm,
    SoftwareHSMProvider,
    create_hsm_provider,
    get_recommended_hsm_config,
)
from .hybrid import (
    HybridCiphertext,
    HybridKeyExchange,
    HybridKeyPair,
    HybridMode,
    HybridPrivateKey,
    HybridPublicKey,
    HybridSessionManager,
    HybridSignature,
    HybridSigner,
)
from .hybrid_certs import (
    CertificateType,
    HybridCertificate,
    HybridCertificateSignature,
    HybridCertificateVersion,
    HybridIdentity,
    HybridIdentityKey,
    HybridIdentityManager,
    HybridIdentityStatus,
    create_hybrid_identity,
    create_hybrid_identity_manager,
)
from .ml_dsa import (
    DefaultMLDSAProvider,
    MLDSAKeyPair,
    MLDSAPrivateKey,
    MLDSAProvider,
    MLDSAPublicKey,
    MLDSASecurityLevel,
    MLDSASignature,
    MockMLDSAProvider,
)
from .ml_kem import (
    DefaultMLKEMProvider,
    MLKEMCiphertext,
    MLKEMKeyPair,
    MLKEMPrivateKey,
    MLKEMProvider,
    MLKEMPublicKey,
    MLKEMSecurityLevel,
    MockMLKEMProvider,
)

__all__ = [
    # ML-KEM
    "MLKEMSecurityLevel",
    "MLKEMKeyPair",
    "MLKEMPublicKey",
    "MLKEMPrivateKey",
    "MLKEMCiphertext",
    "MLKEMProvider",
    "DefaultMLKEMProvider",
    "MockMLKEMProvider",
    # ML-DSA
    "MLDSASecurityLevel",
    "MLDSAKeyPair",
    "MLDSAPublicKey",
    "MLDSAPrivateKey",
    "MLDSASignature",
    "MLDSAProvider",
    "DefaultMLDSAProvider",
    "MockMLDSAProvider",
    # Hybrid
    "HybridMode",
    "HybridKeyExchange",
    "HybridSigner",
    "HybridKeyPair",
    "HybridPublicKey",
    "HybridPrivateKey",
    "HybridSignature",
    "HybridCiphertext",
    "HybridSessionManager",
    # Hybrid Certificates
    "HybridCertificateVersion",
    "CertificateType",
    "HybridIdentityStatus",
    "HybridIdentityKey",
    "HybridCertificateSignature",
    "HybridCertificate",
    "HybridIdentity",
    "HybridIdentityManager",
    "create_hybrid_identity_manager",
    "create_hybrid_identity",
    # HSM
    "HSMType",
    "HSMSecurityLevel",
    "PQAlgorithm",
    "KeyState",
    "HSMKeyHandle",
    "HSMConfig",
    "HSMProvider",
    "SoftwareHSMProvider",
    "PKCS11HSMProvider",
    "create_hsm_provider",
    "get_recommended_hsm_config",
    # Audit
    "AuditEventType",
    "AuditSeverity",
    "ComplianceStandard",
    "CryptoAuditEvent",
    "CryptoAuditConfig",
    "AuditMetrics",
    "CryptoAuditLogger",
    "get_audit_logger",
    "configure_audit_logger",
    "create_production_audit_config",
    # Backup
    "BackupFormat",
    "RecoveryMethod",
    "BackupStatus",
    "KeyShare",
    "KeyBackup",
    "BackupConfig",
    "KeyBackupManager",
    "ShamirSecretSharing",
    "create_backup_manager",
    "create_production_backup_config",
    # Config
    "Environment",
    "AlgorithmConfig",
    "KeyPolicyConfig",
    "SecurityConfig",
    "PerformanceConfig",
    "HSMSettings",
    "PQConfig",
    "get_pq_config",
    "set_pq_config",
    "reset_pq_config",
]

# Version information
__version__ = "2.0.0"  # Phase 4 complete
__pq_standards__ = {
    "ml_kem": "FIPS 203 (CRYSTALS-Kyber)",
    "ml_dsa": "FIPS 204 (CRYSTALS-Dilithium)",
    "slh_dsa": "FIPS 205 (SPHINCS+) - future",
}
