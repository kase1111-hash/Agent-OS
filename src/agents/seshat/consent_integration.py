"""
Agent OS Seshat Consent Integration

Integrates Memory Vault consent system with Seshat's retrieval pipeline.
Provides consent-aware memory operations.
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

# Import from Memory Vault
from ...memory.consent import (
    ConsentManager,
    ConsentOperation,
    ConsentStatus,
)
from ...memory.index import ConsentRecord, VaultIndex
from .retrieval import (
    ConsentVerifier,
    ContextType,
    MemoryEntry,
    RetrievalPipeline,
    RetrievalResult,
)

logger = logging.getLogger(__name__)


class SeshatConsentScope(Enum):
    """Predefined scopes for Seshat operations."""

    CONVERSATION = "seshat:conversation"
    KNOWLEDGE = "seshat:knowledge"
    EPISODIC = "seshat:episodic"
    PROCEDURAL = "seshat:procedural"
    ALL = "seshat:*"


@dataclass
class ConsentAwareConfig:
    """Configuration for consent-aware operations."""

    require_consent_for_store: bool = True
    require_consent_for_retrieve: bool = True
    require_consent_for_delete: bool = True
    default_consent_duration: Optional[timedelta] = timedelta(days=30)
    auto_request_consent: bool = True
    strict_mode: bool = False  # If True, fail operations without consent


class ConsentBridge:
    """
    Bridges Memory Vault consent system with Seshat.

    Provides:
    - Consent verification callback for retrieval
    - Consent-aware storage operations
    - Automatic consent management
    """

    def __init__(
        self,
        consent_manager: ConsentManager,
        config: Optional[ConsentAwareConfig] = None,
    ):
        """
        Initialize consent bridge.

        Args:
            consent_manager: Memory Vault consent manager
            config: Configuration for consent handling
        """
        self._consent_manager = consent_manager
        self._config = config or ConsentAwareConfig()

        # Cache for faster lookups
        self._consent_cache: Dict[str, ConsentRecord] = {}
        self._cache_ttl = timedelta(minutes=5)
        self._cache_timestamps: Dict[str, datetime] = {}

    def create_verifier(self) -> ConsentVerifier:
        """
        Create a ConsentVerifier for use with RetrievalPipeline.

        Returns:
            ConsentVerifier configured with this bridge
        """
        return ConsentVerifier(verify_callback=self.verify_access)

    def verify_access(
        self,
        consent_id: Optional[str],
        accessor: str,
    ) -> bool:
        """
        Verify consent for accessing a memory.

        Args:
            consent_id: Consent ID associated with memory
            accessor: Who is accessing

        Returns:
            True if access is allowed
        """
        if not consent_id:
            # No consent requirement
            return True

        if not self._config.require_consent_for_retrieve:
            return True

        # Check cache first
        if self._is_cached_valid(consent_id):
            cached = self._consent_cache.get(consent_id)
            if cached and cached.active:
                return ConsentOperation.READ.value in cached.operations

        # Verify with consent manager
        decision = self._consent_manager.verify_consent(
            consent_id=consent_id,
            operation=ConsentOperation.READ,
        )

        # Update cache
        consent = self._consent_manager.get_consent(consent_id)
        if consent:
            self._consent_cache[consent_id] = consent
            self._cache_timestamps[consent_id] = datetime.now()

        return decision.status == ConsentStatus.GRANTED

    def verify_store(
        self,
        scope: str,
        accessor: str,
    ) -> Optional[str]:
        """
        Verify or request consent for storing a memory.

        Args:
            scope: Scope for the memory
            accessor: Who is storing

        Returns:
            Consent ID if granted, None if denied
        """
        if not self._config.require_consent_for_store:
            return None  # No consent tracking needed

        # Check for existing consent
        consents = self._consent_manager.list_consents(scope=scope)
        for consent in consents:
            if ConsentOperation.WRITE.value in consent.operations:
                return consent.consent_id

        # Request new consent if configured
        if self._config.auto_request_consent:
            decision = self._consent_manager.request_consent(
                operation=ConsentOperation.WRITE,
                tier=self._get_tier_for_scope(scope),
                scope=scope,
                requestor=accessor,
                purpose="Store memory in Seshat",
                duration=self._config.default_consent_duration,
            )

            if decision.status == ConsentStatus.GRANTED:
                return decision.consent_id

        if self._config.strict_mode:
            logger.warning(f"Store denied: no consent for scope {scope}")
            return None

        return None

    def verify_delete(
        self,
        consent_id: Optional[str],
        accessor: str,
    ) -> bool:
        """
        Verify consent for deleting a memory.

        Args:
            consent_id: Consent ID for the memory
            accessor: Who is deleting

        Returns:
            True if deletion is allowed
        """
        if not consent_id:
            return True

        if not self._config.require_consent_for_delete:
            return True

        decision = self._consent_manager.verify_consent(
            consent_id=consent_id,
            operation=ConsentOperation.DELETE,
        )

        return decision.status == ConsentStatus.GRANTED

    def grant_seshat_consent(
        self,
        scope: SeshatConsentScope,
        granted_by: str,
        operations: Optional[List[ConsentOperation]] = None,
        duration: Optional[timedelta] = None,
    ) -> ConsentRecord:
        """
        Grant consent for Seshat operations.

        Args:
            scope: Seshat scope
            granted_by: Who is granting
            operations: Operations to grant (defaults to READ, WRITE)
            duration: Duration of consent

        Returns:
            Created ConsentRecord
        """
        if operations is None:
            operations = [ConsentOperation.READ, ConsentOperation.WRITE]

        return self._consent_manager.grant_consent(
            granted_by=granted_by,
            scope=scope.value,
            operations=operations,
            duration=duration or self._config.default_consent_duration,
            metadata={
                "source": "seshat",
                "type": "memory_access",
            },
        )

    def revoke_seshat_consent(
        self,
        consent_id: str,
        revoked_by: str,
        reason: Optional[str] = None,
    ) -> bool:
        """
        Revoke Seshat consent.

        Args:
            consent_id: Consent to revoke
            revoked_by: Who is revoking
            reason: Reason for revocation

        Returns:
            True if revoked
        """
        # Clear from cache
        if consent_id in self._consent_cache:
            del self._consent_cache[consent_id]
            del self._cache_timestamps[consent_id]

        return self._consent_manager.revoke_consent(
            consent_id=consent_id,
            revoked_by=revoked_by,
            reason=reason,
        )

    def get_active_consents(
        self,
        scope: Optional[SeshatConsentScope] = None,
    ) -> List[ConsentRecord]:
        """
        Get active consents for Seshat.

        Args:
            scope: Optional scope filter

        Returns:
            List of active consents
        """
        scope_value = scope.value if scope else None
        return self._consent_manager.list_consents(
            scope=scope_value,
            active_only=True,
        )

    def cleanup_expired(self) -> int:
        """
        Clean up expired consents.

        Returns:
            Number cleaned up
        """
        # Clear cache
        self._consent_cache.clear()
        self._cache_timestamps.clear()

        return self._consent_manager.cleanup_expired()

    def _is_cached_valid(self, consent_id: str) -> bool:
        """Check if cached consent is still valid."""
        if consent_id not in self._cache_timestamps:
            return False

        age = datetime.now() - self._cache_timestamps[consent_id]
        return age < self._cache_ttl

    def _get_tier_for_scope(self, scope: str):
        """Get encryption tier for a scope."""
        from ...memory.profiles import EncryptionTier

        # Map scopes to tiers
        if scope.startswith("seshat:conversation"):
            return EncryptionTier.WORKING
        elif scope.startswith("seshat:episodic"):
            return EncryptionTier.PRIVATE
        elif scope.startswith("seshat:procedural"):
            return EncryptionTier.PRIVATE
        elif scope.startswith("seshat:knowledge"):
            return EncryptionTier.SEALED
        else:
            return EncryptionTier.PRIVATE


class ConsentAwareRetrievalPipeline:
    """
    Wrapper around RetrievalPipeline with consent management.

    Automatically handles consent for all operations.
    """

    def __init__(
        self,
        pipeline: RetrievalPipeline,
        consent_bridge: ConsentBridge,
        default_accessor: str = "seshat",
    ):
        """
        Initialize consent-aware pipeline.

        Args:
            pipeline: Base retrieval pipeline
            consent_bridge: Consent bridge for verification
            default_accessor: Default accessor ID
        """
        self._pipeline = pipeline
        self._consent = consent_bridge
        self._default_accessor = default_accessor

    def store_memory(
        self,
        content: str,
        context_type: ContextType,
        source: str,
        scope: Optional[SeshatConsentScope] = None,
        accessor: Optional[str] = None,
        **kwargs,
    ) -> Optional[MemoryEntry]:
        """
        Store a memory with consent verification.

        Args:
            content: Memory content
            context_type: Type of context
            source: Source of memory
            scope: Consent scope
            accessor: Who is storing
            **kwargs: Additional arguments for store

        Returns:
            MemoryEntry if stored, None if consent denied
        """
        accessor = accessor or self._default_accessor

        # Determine scope
        if scope is None:
            scope = self._context_type_to_scope(context_type)

        # Verify/request consent
        consent_id = self._consent.verify_store(
            scope=scope.value,
            accessor=accessor,
        )

        # If strict mode and no consent, return None
        if self._consent._config.strict_mode and consent_id is None:
            logger.warning(f"Store denied: no consent for {context_type.name}")
            return None

        return self._pipeline.store_memory(
            content=content,
            context_type=context_type,
            source=source,
            consent_id=consent_id,
            **kwargs,
        )

    def retrieve(
        self,
        query: str,
        accessor: Optional[str] = None,
        **kwargs,
    ) -> RetrievalResult:
        """
        Retrieve memories with consent filtering.

        Args:
            query: Query text
            accessor: Who is retrieving
            **kwargs: Additional arguments

        Returns:
            RetrievalResult with consent-verified memories
        """
        accessor = accessor or self._default_accessor

        return self._pipeline.retrieve(
            query=query,
            accessor=accessor,
            **kwargs,
        )

    def delete_memory(
        self,
        memory_id: str,
        accessor: Optional[str] = None,
    ) -> bool:
        """
        Delete a memory with consent verification.

        Args:
            memory_id: Memory to delete
            accessor: Who is deleting

        Returns:
            True if deleted
        """
        accessor = accessor or self._default_accessor

        # Get memory to check consent
        memory = self._pipeline.get_memory(memory_id)
        if not memory:
            return False

        # Verify delete permission
        if not self._consent.verify_delete(memory.consent_id, accessor):
            logger.warning(f"Delete denied: no consent for {memory_id}")
            return False

        return self._pipeline.delete_memory(memory_id)

    def delete_by_consent(
        self,
        consent_id: str,
        accessor: Optional[str] = None,
    ) -> int:
        """
        Delete all memories for a consent (right to forget).

        Args:
            consent_id: Consent ID to delete
            accessor: Who is deleting

        Returns:
            Number of memories deleted
        """
        accessor = accessor or self._default_accessor

        # Verify delete permission for this consent
        if not self._consent.verify_delete(consent_id, accessor):
            logger.warning(f"Bulk delete denied for consent: {consent_id}")
            return 0

        # Revoke the consent
        self._consent.revoke_seshat_consent(
            consent_id=consent_id,
            revoked_by=accessor,
            reason="Right to forget requested",
        )

        return self._pipeline.delete_by_consent(consent_id)

    def get_statistics(self) -> Dict[str, Any]:
        """Get pipeline and consent statistics."""
        return {
            **self._pipeline.get_statistics(),
            "consent": {
                "active_consents": len(self._consent.get_active_consents()),
            },
        }

    def _context_type_to_scope(self, context_type: ContextType) -> SeshatConsentScope:
        """Map context type to consent scope."""
        mapping = {
            ContextType.CONVERSATION: SeshatConsentScope.CONVERSATION,
            ContextType.KNOWLEDGE: SeshatConsentScope.KNOWLEDGE,
            ContextType.EPISODIC: SeshatConsentScope.EPISODIC,
            ContextType.PROCEDURAL: SeshatConsentScope.PROCEDURAL,
        }
        return mapping.get(context_type, SeshatConsentScope.ALL)

    # Delegate other methods
    def retrieve_for_rag(self, *args, **kwargs):
        return self._pipeline.retrieve_for_rag(*args, **kwargs)

    def store_memories_batch(self, *args, **kwargs):
        # Note: batch operations would need consent for each entry
        return self._pipeline.store_memories_batch(*args, **kwargs)

    def consolidate_memories(self, *args, **kwargs):
        return self._pipeline.consolidate_memories(*args, **kwargs)

    def clear(self):
        return self._pipeline.clear()

    def shutdown(self):
        return self._pipeline.shutdown()


def create_consent_aware_pipeline(
    pipeline: RetrievalPipeline,
    vault_index: VaultIndex,
    consent_config: Optional[ConsentAwareConfig] = None,
    prompt_callback: Optional[Callable] = None,
) -> ConsentAwareRetrievalPipeline:
    """
    Create a consent-aware retrieval pipeline.

    Args:
        pipeline: Base retrieval pipeline
        vault_index: Memory Vault index for consent storage
        consent_config: Configuration for consent handling
        prompt_callback: Callback to prompt user for consent

    Returns:
        Configured ConsentAwareRetrievalPipeline
    """
    # Create consent manager
    consent_manager = ConsentManager(
        index=vault_index,
        prompt_callback=prompt_callback,
    )

    # Create consent bridge
    consent_bridge = ConsentBridge(
        consent_manager=consent_manager,
        config=consent_config,
    )

    # Wrap pipeline with consent awareness
    return ConsentAwareRetrievalPipeline(
        pipeline=pipeline,
        consent_bridge=consent_bridge,
    )
