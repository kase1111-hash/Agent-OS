"""
Production Configuration for Post-Quantum Cryptography

Provides centralized configuration management for PQ crypto operations:
- Environment-based configuration (development, staging, production)
- Security policy enforcement
- Performance tuning
- Compliance settings

Configuration Hierarchy:
1. Default values (secure defaults)
2. Environment variables (PQ_*)
3. Configuration file (pq_config.json)
4. Runtime overrides

Security Policies:
- Minimum key sizes
- Algorithm restrictions
- Key rotation schedules
- Audit requirements

Usage:
    # Get configuration for environment
    config = get_pq_config("production")

    # Or from environment
    config = PQConfig.from_environment()

    # Access settings
    if config.security.require_hybrid_mode:
        use_hybrid_keys()
"""

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


# =============================================================================
# Environment Enum
# =============================================================================


class Environment(str, Enum):
    """Deployment environment."""

    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"


# =============================================================================
# Algorithm Configuration
# =============================================================================


@dataclass
class AlgorithmConfig:
    """Algorithm-specific configuration."""

    # Key encapsulation
    default_kem_algorithm: str = "ml-kem-768"
    allowed_kem_algorithms: List[str] = field(
        default_factory=lambda: [
            "ml-kem-512",
            "ml-kem-768",
            "ml-kem-1024",
            "x25519-ml-kem-768",
            "x25519-ml-kem-1024",
        ]
    )
    min_kem_security_level: int = 3  # NIST Level 3

    # Digital signatures
    default_sig_algorithm: str = "ml-dsa-65"
    allowed_sig_algorithms: List[str] = field(
        default_factory=lambda: [
            "ml-dsa-44",
            "ml-dsa-65",
            "ml-dsa-87",
            "ed25519-ml-dsa-65",
            "ed25519-ml-dsa-87",
        ]
    )
    min_sig_security_level: int = 3  # NIST Level 3

    # Hybrid mode
    require_hybrid_mode: bool = True
    hybrid_classical_algorithm: str = "x25519"  # or ed25519 for signing

    def is_algorithm_allowed(self, algorithm: str) -> bool:
        """Check if algorithm is allowed."""
        return algorithm in self.allowed_kem_algorithms or algorithm in self.allowed_sig_algorithms


# =============================================================================
# Key Policy Configuration
# =============================================================================


@dataclass
class KeyPolicyConfig:
    """Key management policy configuration."""

    # Key lifecycle
    default_key_validity_days: int = 365
    max_key_validity_days: int = 730  # 2 years
    key_rotation_warning_days: int = 30
    auto_rotate_keys: bool = False
    rotation_overlap_hours: int = 24  # Time both old and new key are valid

    # Key usage
    max_key_operations: Optional[int] = None  # None = unlimited
    max_encrypt_operations: Optional[int] = 1000000
    max_sign_operations: Optional[int] = 1000000

    # Key storage
    require_hsm_for_production: bool = True
    allow_key_export: bool = False
    require_key_backup: bool = True
    backup_threshold: int = 3  # M-of-N recovery
    backup_shares: int = 5

    # Key destruction
    secure_deletion_overwrites: int = 3


# =============================================================================
# Security Configuration
# =============================================================================


@dataclass
class SecurityConfig:
    """Security policy configuration."""

    # Authentication
    require_dual_control: bool = False  # Require 2 people for sensitive ops
    session_timeout_minutes: int = 30
    max_failed_auth_attempts: int = 5
    lockout_duration_minutes: int = 15

    # Cryptographic requirements
    min_password_length: int = 16
    min_entropy_bits: int = 128
    require_hardware_rng: bool = False

    # Network security
    require_tls: bool = True
    min_tls_version: str = "1.3"
    allowed_cipher_suites: List[str] = field(
        default_factory=lambda: [
            "TLS_AES_256_GCM_SHA384",
            "TLS_CHACHA20_POLY1305_SHA256",
        ]
    )

    # Certificate validation
    require_certificate_validation: bool = True
    allowed_certificate_algorithms: List[str] = field(
        default_factory=lambda: [
            "ed25519-ml-dsa-65",
            "ed25519-ml-dsa-87",
        ]
    )
    max_certificate_chain_length: int = 5

    # Threat protection
    enable_timing_attack_protection: bool = True
    enable_side_channel_protection: bool = True


# =============================================================================
# Audit Configuration
# =============================================================================


@dataclass
class AuditConfig:
    """Audit and compliance configuration."""

    # Logging
    enable_audit_logging: bool = True
    log_all_crypto_operations: bool = True
    log_key_access: bool = True
    log_failed_operations: bool = True

    # Retention
    audit_log_retention_days: int = 365
    compliance_report_retention_days: int = 2555  # 7 years

    # Alerting
    alert_on_key_access: bool = False
    alert_on_failed_operations: bool = True
    alert_on_policy_violations: bool = True
    alert_threshold_failed_ops: int = 10  # Per hour

    # Compliance standards
    compliance_standards: List[str] = field(
        default_factory=lambda: [
            "soc2",
            "fips_140_3",
        ]
    )

    # Chain integrity
    enable_chain_verification: bool = True
    chain_verification_interval_hours: int = 1


# =============================================================================
# Performance Configuration
# =============================================================================


@dataclass
class PerformanceConfig:
    """Performance tuning configuration."""

    # Caching
    enable_key_cache: bool = True
    key_cache_size: int = 100
    key_cache_ttl_seconds: int = 3600
    enable_session_cache: bool = True
    session_cache_size: int = 1000

    # Threading
    max_concurrent_operations: int = 100
    operation_timeout_seconds: int = 30
    async_operations: bool = True

    # Batching
    enable_batch_operations: bool = True
    max_batch_size: int = 100

    # Memory
    max_memory_mb: int = 512
    precompute_tables: bool = True  # Precompute lookup tables


# =============================================================================
# HSM Configuration
# =============================================================================


@dataclass
class HSMSettings:
    """HSM-specific settings."""

    # HSM type
    hsm_type: str = "software"  # software, pkcs11, tpm, cloud
    hsm_enabled: bool = False

    # PKCS#11
    pkcs11_library_path: Optional[str] = None
    pkcs11_slot: int = 0
    pkcs11_token_label: Optional[str] = None

    # TPM
    tpm_device_path: str = "/dev/tpm0"

    # Cloud HSM
    cloud_provider: Optional[str] = None  # aws, azure, gcp
    cloud_region: Optional[str] = None
    cloud_key_vault_name: Optional[str] = None

    # Operations
    hsm_operation_timeout_seconds: int = 30
    hsm_retry_attempts: int = 3
    hsm_retry_delay_seconds: int = 1


# =============================================================================
# Main Configuration
# =============================================================================


@dataclass
class PQConfig:
    """
    Main post-quantum cryptography configuration.

    Combines all configuration sections with environment-aware defaults.
    """

    environment: Environment = Environment.DEVELOPMENT

    # Sub-configurations
    algorithms: AlgorithmConfig = field(default_factory=AlgorithmConfig)
    key_policy: KeyPolicyConfig = field(default_factory=KeyPolicyConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    audit: AuditConfig = field(default_factory=AuditConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    hsm: HSMSettings = field(default_factory=HSMSettings)

    # Paths
    config_directory: Optional[Path] = None
    key_store_path: Optional[Path] = None
    audit_log_path: Optional[Path] = None
    backup_path: Optional[Path] = None

    # Feature flags
    enable_hybrid_mode: bool = True
    enable_key_rotation: bool = True
    enable_certificate_management: bool = True
    enable_federation: bool = True

    @classmethod
    def for_environment(cls, env: Environment) -> "PQConfig":
        """Get configuration for a specific environment."""
        if env == Environment.PRODUCTION:
            return cls._production_config()
        elif env == Environment.STAGING:
            return cls._staging_config()
        elif env == Environment.TESTING:
            return cls._testing_config()
        else:
            return cls._development_config()

    @classmethod
    def _production_config(cls) -> "PQConfig":
        """Production environment configuration."""
        return cls(
            environment=Environment.PRODUCTION,
            algorithms=AlgorithmConfig(
                require_hybrid_mode=True,
                min_kem_security_level=3,
                min_sig_security_level=3,
            ),
            key_policy=KeyPolicyConfig(
                require_hsm_for_production=True,
                allow_key_export=False,
                require_key_backup=True,
                auto_rotate_keys=True,
                max_key_validity_days=365,
            ),
            security=SecurityConfig(
                require_dual_control=True,
                require_tls=True,
                min_tls_version="1.3",
                require_hardware_rng=True,
                enable_timing_attack_protection=True,
                enable_side_channel_protection=True,
            ),
            audit=AuditConfig(
                enable_audit_logging=True,
                log_all_crypto_operations=True,
                alert_on_failed_operations=True,
                alert_on_policy_violations=True,
                enable_chain_verification=True,
            ),
            performance=PerformanceConfig(
                enable_key_cache=True,
                key_cache_size=50,
                key_cache_ttl_seconds=1800,
                async_operations=True,
            ),
            hsm=HSMSettings(
                hsm_enabled=True,
                hsm_type="pkcs11",
            ),
            enable_hybrid_mode=True,
            enable_key_rotation=True,
            enable_certificate_management=True,
        )

    @classmethod
    def _staging_config(cls) -> "PQConfig":
        """Staging environment configuration."""
        return cls(
            environment=Environment.STAGING,
            algorithms=AlgorithmConfig(
                require_hybrid_mode=True,
                min_kem_security_level=3,
            ),
            key_policy=KeyPolicyConfig(
                require_hsm_for_production=False,
                allow_key_export=False,
                require_key_backup=True,
            ),
            security=SecurityConfig(
                require_dual_control=False,
                require_tls=True,
            ),
            audit=AuditConfig(
                enable_audit_logging=True,
                log_all_crypto_operations=True,
            ),
            hsm=HSMSettings(
                hsm_enabled=False,
                hsm_type="software",
            ),
            enable_hybrid_mode=True,
        )

    @classmethod
    def _testing_config(cls) -> "PQConfig":
        """Testing environment configuration."""
        return cls(
            environment=Environment.TESTING,
            algorithms=AlgorithmConfig(
                require_hybrid_mode=False,
                min_kem_security_level=1,
                min_sig_security_level=1,
            ),
            key_policy=KeyPolicyConfig(
                require_hsm_for_production=False,
                allow_key_export=True,
                require_key_backup=False,
            ),
            security=SecurityConfig(
                require_dual_control=False,
                require_tls=False,
                min_password_length=8,
            ),
            audit=AuditConfig(
                enable_audit_logging=False,
            ),
            performance=PerformanceConfig(
                enable_key_cache=False,
            ),
            hsm=HSMSettings(
                hsm_enabled=False,
            ),
            enable_hybrid_mode=False,
        )

    @classmethod
    def _development_config(cls) -> "PQConfig":
        """Development environment configuration."""
        return cls(
            environment=Environment.DEVELOPMENT,
            algorithms=AlgorithmConfig(
                require_hybrid_mode=False,
                min_kem_security_level=1,
            ),
            key_policy=KeyPolicyConfig(
                require_hsm_for_production=False,
                allow_key_export=True,
                require_key_backup=False,
                auto_rotate_keys=False,
            ),
            security=SecurityConfig(
                require_dual_control=False,
                require_tls=False,
                enable_timing_attack_protection=False,
            ),
            audit=AuditConfig(
                enable_audit_logging=True,
                log_all_crypto_operations=False,
            ),
            performance=PerformanceConfig(
                enable_key_cache=True,
                async_operations=False,
            ),
            hsm=HSMSettings(
                hsm_enabled=False,
            ),
            enable_hybrid_mode=True,
        )

    @classmethod
    def from_environment(cls) -> "PQConfig":
        """Load configuration from environment variables."""
        env_name = os.environ.get("PQ_ENVIRONMENT", "development")
        try:
            env = Environment(env_name.lower())
        except ValueError:
            logger.warning(f"Unknown environment {env_name}, using development")
            env = Environment.DEVELOPMENT

        config = cls.for_environment(env)

        # Override with environment variables
        config._apply_env_overrides()

        return config

    @classmethod
    def from_file(cls, path: Path) -> "PQConfig":
        """Load configuration from JSON file."""
        with open(path) as f:
            data = json.load(f)

        env = Environment(data.get("environment", "development"))
        config = cls.for_environment(env)

        # Apply file overrides
        config._apply_dict_overrides(data)

        return config

    def _apply_env_overrides(self) -> None:
        """Apply environment variable overrides."""
        env_mappings = {
            "PQ_REQUIRE_HYBRID": ("algorithms", "require_hybrid_mode", bool),
            "PQ_MIN_SECURITY_LEVEL": ("algorithms", "min_kem_security_level", int),
            "PQ_REQUIRE_HSM": ("key_policy", "require_hsm_for_production", bool),
            "PQ_ALLOW_EXPORT": ("key_policy", "allow_key_export", bool),
            "PQ_REQUIRE_BACKUP": ("key_policy", "require_key_backup", bool),
            "PQ_ENABLE_AUDIT": ("audit", "enable_audit_logging", bool),
            "PQ_HSM_TYPE": ("hsm", "hsm_type", str),
            "PQ_HSM_ENABLED": ("hsm", "hsm_enabled", bool),
            "PQ_KEY_STORE_PATH": ("key_store_path", None, Path),
            "PQ_AUDIT_LOG_PATH": ("audit_log_path", None, Path),
        }

        for env_var, (section, attr, type_) in env_mappings.items():
            value = os.environ.get(env_var)
            if value is not None:
                try:
                    if type_ == bool:
                        typed_value = value.lower() in ("true", "1", "yes")
                    elif type_ == Path:
                        typed_value = Path(value)
                    else:
                        typed_value = type_(value)

                    if attr is None:
                        setattr(self, section, typed_value)
                    else:
                        sub_config = getattr(self, section)
                        setattr(sub_config, attr, typed_value)

                except (ValueError, TypeError) as e:
                    logger.warning(f"Invalid value for {env_var}: {e}")

    def _apply_dict_overrides(self, data: Dict[str, Any]) -> None:
        """Apply dictionary overrides."""
        section_map = {
            "algorithms": self.algorithms,
            "key_policy": self.key_policy,
            "security": self.security,
            "audit": self.audit,
            "performance": self.performance,
            "hsm": self.hsm,
        }

        for section_name, section_data in data.items():
            if section_name in section_map and isinstance(section_data, dict):
                section = section_map[section_name]
                for key, value in section_data.items():
                    if hasattr(section, key):
                        setattr(section, key, value)

        # Top-level overrides
        for key in ["key_store_path", "audit_log_path", "backup_path", "config_directory"]:
            if key in data:
                value = data[key]
                if value is not None:
                    setattr(self, key, Path(value))

        for key in [
            "enable_hybrid_mode",
            "enable_key_rotation",
            "enable_certificate_management",
            "enable_federation",
        ]:
            if key in data:
                setattr(self, key, data[key])

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "environment": self.environment.value,
            "algorithms": {
                "default_kem_algorithm": self.algorithms.default_kem_algorithm,
                "default_sig_algorithm": self.algorithms.default_sig_algorithm,
                "require_hybrid_mode": self.algorithms.require_hybrid_mode,
                "min_kem_security_level": self.algorithms.min_kem_security_level,
                "min_sig_security_level": self.algorithms.min_sig_security_level,
            },
            "key_policy": {
                "default_key_validity_days": self.key_policy.default_key_validity_days,
                "require_hsm_for_production": self.key_policy.require_hsm_for_production,
                "allow_key_export": self.key_policy.allow_key_export,
                "require_key_backup": self.key_policy.require_key_backup,
            },
            "security": {
                "require_dual_control": self.security.require_dual_control,
                "require_tls": self.security.require_tls,
                "require_certificate_validation": self.security.require_certificate_validation,
            },
            "audit": {
                "enable_audit_logging": self.audit.enable_audit_logging,
                "log_all_crypto_operations": self.audit.log_all_crypto_operations,
                "enable_chain_verification": self.audit.enable_chain_verification,
            },
            "hsm": {
                "hsm_enabled": self.hsm.hsm_enabled,
                "hsm_type": self.hsm.hsm_type,
            },
            "paths": {
                "key_store_path": str(self.key_store_path) if self.key_store_path else None,
                "audit_log_path": str(self.audit_log_path) if self.audit_log_path else None,
                "backup_path": str(self.backup_path) if self.backup_path else None,
            },
            "features": {
                "enable_hybrid_mode": self.enable_hybrid_mode,
                "enable_key_rotation": self.enable_key_rotation,
                "enable_certificate_management": self.enable_certificate_management,
                "enable_federation": self.enable_federation,
            },
        }

    def save(self, path: Path) -> None:
        """Save configuration to file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    def validate(self) -> List[str]:
        """
        Validate configuration.

        Returns:
            List of validation errors (empty if valid)
        """
        errors = []

        # Check production requirements
        if self.environment == Environment.PRODUCTION:
            if not self.algorithms.require_hybrid_mode:
                errors.append("Hybrid mode required in production")
            if self.key_policy.allow_key_export:
                errors.append("Key export not allowed in production")
            if not self.audit.enable_audit_logging:
                errors.append("Audit logging required in production")
            if not self.security.require_tls:
                errors.append("TLS required in production")

        # Check algorithm settings
        if self.algorithms.min_kem_security_level < 1:
            errors.append("KEM security level must be at least 1")
        if self.algorithms.min_sig_security_level < 1:
            errors.append("Signature security level must be at least 1")

        # Check paths
        if self.key_store_path and not self.key_store_path.parent.exists():
            errors.append(
                f"Key store parent directory does not exist: {self.key_store_path.parent}"
            )

        return errors


# =============================================================================
# Factory Functions
# =============================================================================


_global_config: Optional[PQConfig] = None


def get_pq_config(environment: Optional[str] = None) -> PQConfig:
    """
    Get PQ configuration.

    Args:
        environment: Optional environment name (uses env var if not provided)

    Returns:
        PQConfig instance
    """
    global _global_config

    if _global_config is not None:
        return _global_config

    if environment:
        env = Environment(environment.lower())
        _global_config = PQConfig.for_environment(env)
    else:
        _global_config = PQConfig.from_environment()

    return _global_config


def set_pq_config(config: PQConfig) -> None:
    """Set global PQ configuration."""
    global _global_config
    _global_config = config


def reset_pq_config() -> None:
    """Reset global PQ configuration."""
    global _global_config
    _global_config = None
