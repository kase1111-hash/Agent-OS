"""
Agent OS Memory Vault Encryption Profiles

Defines four encryption classification tiers:
- Working: Session-scoped, ephemeral keys
- Private: User-scoped, persistent encryption
- Sealed: High-security, requires explicit unlock
- Vaulted: Maximum security, hardware-bound keys
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class EncryptionTier(Enum):
    """Encryption classification tiers (lowest to highest security)."""

    WORKING = 1  # Session-scoped, ephemeral
    PRIVATE = 2  # User-scoped, persistent
    SEALED = 3  # High-security, requires unlock
    VAULTED = 4  # Maximum security, hardware-bound


class KeyDerivation(Enum):
    """Key derivation methods."""

    PBKDF2 = auto()
    ARGON2ID = auto()
    SCRYPT = auto()


class KeyBinding(Enum):
    """Key binding types."""

    SOFTWARE = auto()  # Software-only key
    TPM = auto()  # TPM-bound
    HARDWARE = auto()  # Hardware security module


@dataclass
class EncryptionProfile:
    """Configuration for an encryption tier."""

    tier: EncryptionTier
    name: str
    description: str

    # Key configuration
    key_derivation: KeyDerivation
    key_binding: KeyBinding
    key_bits: int = 256

    # Access configuration
    requires_unlock: bool = False
    auto_lock_timeout: Optional[timedelta] = None
    max_access_count: Optional[int] = None

    # Retention configuration
    default_ttl: Optional[timedelta] = None
    allow_extension: bool = True
    requires_consent: bool = True

    # Audit configuration
    log_access: bool = True
    log_content: bool = False  # Never log actual content

    # Additional constraints
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate profile configuration."""
        if self.tier == EncryptionTier.VAULTED and self.key_binding == KeyBinding.SOFTWARE:
            raise ValueError("VAULTED tier requires hardware key binding")
        if self.log_content:
            raise ValueError("Content logging is prohibited")


# Default profile configurations
WORKING_PROFILE = EncryptionProfile(
    tier=EncryptionTier.WORKING,
    name="Working",
    description="Session-scoped ephemeral storage for active work",
    key_derivation=KeyDerivation.PBKDF2,
    key_binding=KeyBinding.SOFTWARE,
    key_bits=256,
    requires_unlock=False,
    auto_lock_timeout=None,
    default_ttl=timedelta(hours=24),
    allow_extension=True,
    requires_consent=False,  # Implicit consent for working memory
    log_access=True,
    metadata={"session_bound": True},
)

PRIVATE_PROFILE = EncryptionProfile(
    tier=EncryptionTier.PRIVATE,
    name="Private",
    description="User-scoped persistent storage with standard encryption",
    key_derivation=KeyDerivation.ARGON2ID,
    key_binding=KeyBinding.SOFTWARE,
    key_bits=256,
    requires_unlock=False,
    auto_lock_timeout=timedelta(hours=8),
    default_ttl=None,  # No expiry
    allow_extension=True,
    requires_consent=True,
    log_access=True,
    metadata={"persistent": True},
)

SEALED_PROFILE = EncryptionProfile(
    tier=EncryptionTier.SEALED,
    name="Sealed",
    description="High-security storage requiring explicit unlock",
    key_derivation=KeyDerivation.ARGON2ID,
    key_binding=KeyBinding.SOFTWARE,  # TPM preferred if available
    key_bits=256,
    requires_unlock=True,
    auto_lock_timeout=timedelta(minutes=30),
    max_access_count=10,  # Re-seal after 10 accesses
    default_ttl=None,
    allow_extension=False,
    requires_consent=True,
    log_access=True,
    metadata={"high_security": True, "explicit_unlock_required": True},
)

VAULTED_PROFILE = EncryptionProfile(
    tier=EncryptionTier.VAULTED,
    name="Vaulted",
    description="Maximum security with hardware-bound keys",
    key_derivation=KeyDerivation.ARGON2ID,
    key_binding=KeyBinding.HARDWARE,  # Requires hardware binding
    key_bits=256,
    requires_unlock=True,
    auto_lock_timeout=timedelta(minutes=5),
    max_access_count=3,  # Re-seal after 3 accesses
    default_ttl=None,
    allow_extension=False,
    requires_consent=True,
    log_access=True,
    metadata={
        "maximum_security": True,
        "hardware_bound": True,
        "explicit_unlock_required": True,
    },
)


class ProfileManager:
    """
    Manages encryption profiles for the Memory Vault.

    Provides profile lookup, validation, and tier comparison.
    """

    def __init__(self):
        """Initialize profile manager with default profiles."""
        self._profiles: Dict[EncryptionTier, EncryptionProfile] = {
            EncryptionTier.WORKING: WORKING_PROFILE,
            EncryptionTier.PRIVATE: PRIVATE_PROFILE,
            EncryptionTier.SEALED: SEALED_PROFILE,
            EncryptionTier.VAULTED: VAULTED_PROFILE,
        }
        self._custom_profiles: Dict[str, EncryptionProfile] = {}

    def get_profile(self, tier: EncryptionTier) -> EncryptionProfile:
        """
        Get the profile for a tier.

        Args:
            tier: The encryption tier

        Returns:
            EncryptionProfile for the tier
        """
        return self._profiles[tier]

    def get_profile_by_name(self, name: str) -> Optional[EncryptionProfile]:
        """
        Get a profile by name.

        Args:
            name: Profile name

        Returns:
            EncryptionProfile or None if not found
        """
        name_lower = name.lower()
        for profile in self._profiles.values():
            if profile.name.lower() == name_lower:
                return profile
        return self._custom_profiles.get(name)

    def register_custom_profile(
        self,
        name: str,
        profile: EncryptionProfile,
    ) -> None:
        """
        Register a custom encryption profile.

        Args:
            name: Unique profile name
            profile: Profile configuration
        """
        if name in self._custom_profiles:
            raise ValueError(f"Profile '{name}' already exists")
        self._custom_profiles[name] = profile
        logger.info(f"Registered custom profile: {name}")

    def can_access(
        self,
        source_tier: EncryptionTier,
        target_tier: EncryptionTier,
    ) -> bool:
        """
        Check if source tier can access target tier data.

        Higher tiers can access lower tier data, but not vice versa.

        Args:
            source_tier: Requester's tier
            target_tier: Data's tier

        Returns:
            True if access is allowed
        """
        return source_tier.value >= target_tier.value

    def can_promote(
        self,
        from_tier: EncryptionTier,
        to_tier: EncryptionTier,
    ) -> bool:
        """
        Check if data can be promoted to a higher tier.

        Args:
            from_tier: Current tier
            to_tier: Target tier

        Returns:
            True if promotion is allowed
        """
        return to_tier.value > from_tier.value

    def can_demote(
        self,
        from_tier: EncryptionTier,
        to_tier: EncryptionTier,
    ) -> bool:
        """
        Check if data can be demoted to a lower tier.

        Demotion is generally not allowed for security reasons.

        Args:
            from_tier: Current tier
            to_tier: Target tier

        Returns:
            True if demotion is allowed (usually False)
        """
        # Demotion is prohibited by default
        # Only WORKING -> PRIVATE is sometimes allowed (session promotion)
        if from_tier == EncryptionTier.WORKING and to_tier == EncryptionTier.PRIVATE:
            return True  # Session data can be persisted
        return False

    def get_minimum_tier_for_content(
        self,
        content_type: str,
        sensitivity_hints: Optional[List[str]] = None,
    ) -> EncryptionTier:
        """
        Determine minimum encryption tier for content type.

        Args:
            content_type: Type of content being stored
            sensitivity_hints: Optional hints about sensitivity

        Returns:
            Minimum required EncryptionTier
        """
        sensitivity_hints = sensitivity_hints or []

        # High sensitivity content types
        high_sensitivity = {
            "credentials",
            "password",
            "secret",
            "key",
            "token",
            "financial",
            "medical",
            "legal",
            "personal_id",
        }

        # Check content type
        content_lower = content_type.lower()
        for keyword in high_sensitivity:
            if keyword in content_lower:
                return EncryptionTier.SEALED

        # Check sensitivity hints
        hint_lower = [h.lower() for h in sensitivity_hints]
        if any(k in " ".join(hint_lower) for k in ["secret", "confidential", "sensitive"]):
            return EncryptionTier.SEALED

        if any(k in " ".join(hint_lower) for k in ["personal", "private"]):
            return EncryptionTier.PRIVATE

        # Default to private for persistent storage
        return EncryptionTier.PRIVATE

    def list_profiles(self) -> List[EncryptionProfile]:
        """List all available profiles."""
        profiles = list(self._profiles.values())
        profiles.extend(self._custom_profiles.values())
        return profiles

    def get_tier_requirements(self, tier: EncryptionTier) -> Dict[str, Any]:
        """
        Get the requirements for accessing a tier.

        Args:
            tier: The encryption tier

        Returns:
            Dict of requirements
        """
        profile = self._profiles[tier]
        return {
            "requires_unlock": profile.requires_unlock,
            "requires_consent": profile.requires_consent,
            "key_binding": profile.key_binding.name,
            "auto_lock_timeout": (
                profile.auto_lock_timeout.total_seconds() if profile.auto_lock_timeout else None
            ),
            "max_access_count": profile.max_access_count,
        }
