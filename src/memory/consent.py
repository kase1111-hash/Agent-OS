"""
Agent OS Memory Vault Consent Verification Layer

Implements consent-gated memory operations:
- Consent creation and verification
- Operation authorization
- Scope validation
- Consent lifecycle management
"""

import secrets
import hashlib
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List, Set, Callable
from enum import Enum, auto
import threading

from .profiles import EncryptionTier
from .index import VaultIndex, ConsentRecord, AccessType


logger = logging.getLogger(__name__)


class ConsentType(Enum):
    """Types of consent."""
    OBSERVATION = auto()   # Observe but don't store
    SESSION = auto()       # Store for session only
    PERSISTENT = auto()    # Store persistently
    LEARNING = auto()      # Allow learning/generalization


class ConsentOperation(Enum):
    """Operations that require consent."""
    READ = "read"
    WRITE = "write"
    DELETE = "delete"
    SHARE = "share"
    EXPORT = "export"
    LEARN = "learn"


class ConsentStatus(Enum):
    """Status of a consent check."""
    GRANTED = auto()
    DENIED = auto()
    EXPIRED = auto()
    REVOKED = auto()
    PENDING = auto()
    INSUFFICIENT = auto()


@dataclass
class ConsentRequest:
    """Request for consent."""
    request_id: str
    requestor: str
    operation: ConsentOperation
    scope: str
    tier: EncryptionTier
    purpose: str
    duration: Optional[timedelta] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class ConsentDecision:
    """Result of a consent check."""
    request: ConsentRequest
    status: ConsentStatus
    consent_id: Optional[str] = None
    reason: Optional[str] = None
    constraints: List[str] = field(default_factory=list)
    expires_at: Optional[datetime] = None
    decided_at: datetime = field(default_factory=datetime.now)


class ConsentPolicy:
    """
    Policy configuration for consent requirements.

    Defines what operations require consent and at what level.
    """

    def __init__(self):
        # Operations always requiring explicit consent
        self.always_require: Set[ConsentOperation] = {
            ConsentOperation.WRITE,
            ConsentOperation.LEARN,
            ConsentOperation.SHARE,
            ConsentOperation.EXPORT,
        }

        # Operations requiring consent by tier
        self.tier_requirements: Dict[EncryptionTier, Set[ConsentOperation]] = {
            EncryptionTier.WORKING: set(),  # Implicit consent for working memory
            EncryptionTier.PRIVATE: {ConsentOperation.READ, ConsentOperation.WRITE},
            EncryptionTier.SEALED: {
                ConsentOperation.READ,
                ConsentOperation.WRITE,
                ConsentOperation.DELETE,
            },
            EncryptionTier.VAULTED: {
                ConsentOperation.READ,
                ConsentOperation.WRITE,
                ConsentOperation.DELETE,
                ConsentOperation.SHARE,
                ConsentOperation.EXPORT,
            },
        }

        # Default consent durations by tier
        self.default_durations: Dict[EncryptionTier, Optional[timedelta]] = {
            EncryptionTier.WORKING: timedelta(hours=24),
            EncryptionTier.PRIVATE: None,  # No expiry
            EncryptionTier.SEALED: timedelta(days=30),
            EncryptionTier.VAULTED: timedelta(days=7),
        }

        # Prohibited operations (never allowed)
        self.prohibited: Set[str] = {
            "background_indexing",
            "silent_storage",
            "cross_session_tracking",
        }

    def requires_consent(
        self,
        operation: ConsentOperation,
        tier: EncryptionTier,
    ) -> bool:
        """Check if operation requires consent."""
        if operation in self.always_require:
            return True
        return operation in self.tier_requirements.get(tier, set())

    def is_prohibited(self, operation: str) -> bool:
        """Check if operation is prohibited."""
        return operation in self.prohibited


class ConsentManager:
    """
    Manages consent for memory operations.

    Responsibilities:
    - Verify consent before operations
    - Create and track consent records
    - Handle consent revocation
    - Enforce consent policies
    """

    def __init__(
        self,
        index: VaultIndex,
        policy: Optional[ConsentPolicy] = None,
        prompt_callback: Optional[Callable[[ConsentRequest], bool]] = None,
    ):
        """
        Initialize consent manager.

        Args:
            index: Vault index for persistence
            policy: Consent policy (uses default if not provided)
            prompt_callback: Callback to prompt user for consent
        """
        self._index = index
        self._policy = policy or ConsentPolicy()
        self._prompt_callback = prompt_callback

        self._active_consents: Dict[str, ConsentRecord] = {}
        self._lock = threading.RLock()

        self._load_active_consents()

    def _load_active_consents(self) -> None:
        """Load active consents from index."""
        with self._lock:
            consents = self._index.get_active_consents()
            for consent in consents:
                self._active_consents[consent.consent_id] = consent
            logger.info(f"Loaded {len(consents)} active consents")

    def request_consent(
        self,
        operation: ConsentOperation,
        tier: EncryptionTier,
        scope: str,
        requestor: str,
        purpose: str,
        duration: Optional[timedelta] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ConsentDecision:
        """
        Request consent for an operation.

        Args:
            operation: Operation type
            tier: Encryption tier
            scope: Scope of consent (e.g., "user_preferences", "project_context")
            requestor: Who is requesting
            purpose: Purpose of the request
            duration: How long consent is needed
            metadata: Additional metadata

        Returns:
            ConsentDecision with result
        """
        request = ConsentRequest(
            request_id=secrets.token_hex(16),
            requestor=requestor,
            operation=operation,
            scope=scope,
            tier=tier,
            purpose=purpose,
            duration=duration or self._policy.default_durations.get(tier),
            metadata=metadata or {},
        )

        # Check if consent is required
        if not self._policy.requires_consent(operation, tier):
            return ConsentDecision(
                request=request,
                status=ConsentStatus.GRANTED,
                reason="Consent not required for this operation",
            )

        # Check for existing consent
        existing = self._find_matching_consent(request)
        if existing:
            return ConsentDecision(
                request=request,
                status=ConsentStatus.GRANTED,
                consent_id=existing.consent_id,
                reason="Covered by existing consent",
                expires_at=existing.expires_at,
            )

        # Need to request new consent
        if self._prompt_callback:
            granted = self._prompt_callback(request)
            if granted:
                consent = self._create_consent(request)
                return ConsentDecision(
                    request=request,
                    status=ConsentStatus.GRANTED,
                    consent_id=consent.consent_id,
                    expires_at=consent.expires_at,
                )
            else:
                return ConsentDecision(
                    request=request,
                    status=ConsentStatus.DENIED,
                    reason="User denied consent",
                )

        # No callback, return pending
        return ConsentDecision(
            request=request,
            status=ConsentStatus.PENDING,
            reason="Consent required but no prompt callback available",
        )

    def verify_consent(
        self,
        consent_id: str,
        operation: ConsentOperation,
        scope: Optional[str] = None,
    ) -> ConsentDecision:
        """
        Verify an existing consent is valid for an operation.

        Args:
            consent_id: Consent ID to verify
            operation: Operation to perform
            scope: Optional scope to check

        Returns:
            ConsentDecision with verification result
        """
        with self._lock:
            consent = self._active_consents.get(consent_id)

            if not consent:
                # Try loading from index
                consent = self._index.get_consent(consent_id)
                if consent and consent.active:
                    self._active_consents[consent_id] = consent

            if not consent:
                return ConsentDecision(
                    request=ConsentRequest(
                        request_id="verify",
                        requestor="system",
                        operation=operation,
                        scope=scope or "",
                        tier=EncryptionTier.PRIVATE,
                        purpose="verification",
                    ),
                    status=ConsentStatus.DENIED,
                    reason="Consent not found",
                )

            # Check if revoked
            if not consent.active:
                return ConsentDecision(
                    request=ConsentRequest(
                        request_id="verify",
                        requestor="system",
                        operation=operation,
                        scope=scope or "",
                        tier=EncryptionTier.PRIVATE,
                        purpose="verification",
                    ),
                    status=ConsentStatus.REVOKED,
                    reason="Consent has been revoked",
                )

            # Check expiry
            if consent.expires_at and datetime.now() > consent.expires_at:
                consent.active = False
                self._index.revoke_consent(consent_id)
                return ConsentDecision(
                    request=ConsentRequest(
                        request_id="verify",
                        requestor="system",
                        operation=operation,
                        scope=scope or "",
                        tier=EncryptionTier.PRIVATE,
                        purpose="verification",
                    ),
                    status=ConsentStatus.EXPIRED,
                    reason="Consent has expired",
                )

            # Check operation is covered
            if operation.value not in consent.operations:
                return ConsentDecision(
                    request=ConsentRequest(
                        request_id="verify",
                        requestor="system",
                        operation=operation,
                        scope=scope or "",
                        tier=EncryptionTier.PRIVATE,
                        purpose="verification",
                    ),
                    status=ConsentStatus.INSUFFICIENT,
                    reason=f"Consent does not cover operation: {operation.value}",
                )

            # Check scope matches
            if scope and consent.scope != scope and consent.scope != "*":
                return ConsentDecision(
                    request=ConsentRequest(
                        request_id="verify",
                        requestor="system",
                        operation=operation,
                        scope=scope,
                        tier=EncryptionTier.PRIVATE,
                        purpose="verification",
                    ),
                    status=ConsentStatus.INSUFFICIENT,
                    reason=f"Consent scope mismatch: {consent.scope} vs {scope}",
                )

            return ConsentDecision(
                request=ConsentRequest(
                    request_id="verify",
                    requestor="system",
                    operation=operation,
                    scope=scope or consent.scope,
                    tier=EncryptionTier.PRIVATE,
                    purpose="verification",
                ),
                status=ConsentStatus.GRANTED,
                consent_id=consent_id,
                expires_at=consent.expires_at,
            )

    def grant_consent(
        self,
        granted_by: str,
        scope: str,
        operations: List[ConsentOperation],
        duration: Optional[timedelta] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ConsentRecord:
        """
        Explicitly grant consent.

        Args:
            granted_by: Who is granting consent
            scope: Scope of consent
            operations: Operations covered
            duration: Duration of consent
            metadata: Additional metadata

        Returns:
            Created ConsentRecord
        """
        consent_id = f"consent_{secrets.token_hex(16)}"
        now = datetime.now()

        consent = ConsentRecord(
            consent_id=consent_id,
            granted_by=granted_by,
            granted_at=now,
            expires_at=now + duration if duration else None,
            scope=scope,
            operations=[op.value for op in operations],
            active=True,
            metadata=metadata or {},
        )

        with self._lock:
            self._index.record_consent(consent)
            self._active_consents[consent_id] = consent

        logger.info(
            f"Consent granted: {consent_id} "
            f"(scope={scope}, ops={[op.value for op in operations]})"
        )
        return consent

    def revoke_consent(
        self,
        consent_id: str,
        revoked_by: str,
        reason: Optional[str] = None,
    ) -> bool:
        """
        Revoke a consent.

        Args:
            consent_id: Consent to revoke
            revoked_by: Who is revoking
            reason: Reason for revocation

        Returns:
            True if revoked
        """
        with self._lock:
            consent = self._active_consents.get(consent_id)
            if not consent:
                return False

            consent.active = False
            consent.revoked_at = datetime.now()
            consent.metadata["revoked_by"] = revoked_by
            if reason:
                consent.metadata["revocation_reason"] = reason

            self._index.revoke_consent(consent_id)
            del self._active_consents[consent_id]

        logger.warning(f"Consent revoked: {consent_id} (by {revoked_by})")
        return True

    def get_consent(self, consent_id: str) -> Optional[ConsentRecord]:
        """Get a consent record."""
        with self._lock:
            return self._active_consents.get(consent_id)

    def list_consents(
        self,
        scope: Optional[str] = None,
        active_only: bool = True,
    ) -> List[ConsentRecord]:
        """List consents, optionally filtered."""
        if active_only:
            consents = list(self._active_consents.values())
        else:
            consents = self._index.get_active_consents(scope)

        if scope:
            consents = [c for c in consents if c.scope == scope or c.scope == "*"]

        return consents

    def cleanup_expired(self) -> int:
        """
        Clean up expired consents.

        Returns:
            Number of consents cleaned up
        """
        count = 0
        now = datetime.now()

        with self._lock:
            expired = [
                c for c in self._active_consents.values()
                if c.expires_at and c.expires_at < now
            ]

            for consent in expired:
                consent.active = False
                self._index.revoke_consent(consent.consent_id)
                del self._active_consents[consent.consent_id]
                count += 1

        if count > 0:
            logger.info(f"Cleaned up {count} expired consents")

        return count

    def get_statistics(self) -> Dict[str, Any]:
        """Get consent statistics."""
        with self._lock:
            stats = {
                "active_consents": len(self._active_consents),
                "by_scope": {},
            }

            for consent in self._active_consents.values():
                scope = consent.scope
                stats["by_scope"][scope] = stats["by_scope"].get(scope, 0) + 1

            return stats

    def _find_matching_consent(
        self,
        request: ConsentRequest,
    ) -> Optional[ConsentRecord]:
        """Find an existing consent that covers the request."""
        with self._lock:
            for consent in self._active_consents.values():
                if not consent.active:
                    continue

                # Check expiry
                if consent.expires_at and datetime.now() > consent.expires_at:
                    continue

                # Check scope
                if consent.scope != request.scope and consent.scope != "*":
                    continue

                # Check operation
                if request.operation.value not in consent.operations:
                    continue

                return consent

            return None

    def _create_consent(self, request: ConsentRequest) -> ConsentRecord:
        """Create a consent from a request."""
        consent_id = f"consent_{secrets.token_hex(16)}"
        now = datetime.now()

        consent = ConsentRecord(
            consent_id=consent_id,
            granted_by=request.requestor,
            granted_at=now,
            expires_at=now + request.duration if request.duration else None,
            scope=request.scope,
            operations=[request.operation.value],
            active=True,
            metadata={
                "purpose": request.purpose,
                "request_id": request.request_id,
                **request.metadata,
            },
        )

        with self._lock:
            self._index.record_consent(consent)
            self._active_consents[consent_id] = consent

        logger.info(f"Created consent: {consent_id} for {request.operation.value}")
        return consent


def create_default_observation_contract() -> Dict[str, Any]:
    """
    Create the default observation contract.

    This contract allows observation but prevents storage or learning.
    """
    return {
        "type": "observation",
        "allows_storage": False,
        "allows_learning": False,
        "allows_generalization": False,
        "expires_after_session": True,
        "scope": "*",
    }


def create_explicit_learning_contract(
    scope: str,
    granted_by: str,
    duration: Optional[timedelta] = None,
) -> Dict[str, Any]:
    """
    Create an explicit learning contract.

    Requires human confirmation before learning from data.
    """
    return {
        "type": "learning",
        "scope": scope,
        "granted_by": granted_by,
        "allows_storage": True,
        "allows_learning": True,
        "allows_generalization": True,
        "requires_human_confirmation": True,
        "duration_seconds": duration.total_seconds() if duration else None,
        "created_at": datetime.now().isoformat(),
    }
