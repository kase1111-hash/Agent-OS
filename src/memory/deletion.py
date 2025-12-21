"""
Agent OS Memory Vault Right-to-Delete Propagation

Implements secure deletion and right-to-forget:
- Cascading deletion by consent
- Secure data wiping
- Deletion verification
- Audit trail
"""

import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, Any, List, Callable
from enum import Enum, auto
from pathlib import Path

from .profiles import EncryptionTier
from .storage import BlobStorage, BlobStatus
from .index import VaultIndex, AccessType
from .keys import KeyManager


logger = logging.getLogger(__name__)


class DeletionStatus(Enum):
    """Status of a deletion request."""
    PENDING = auto()
    IN_PROGRESS = auto()
    COMPLETED = auto()
    PARTIAL = auto()
    FAILED = auto()
    VERIFIED = auto()


class DeletionScope(Enum):
    """Scope of deletion."""
    SINGLE_BLOB = auto()
    CONSENT_CASCADE = auto()
    TIER_PURGE = auto()
    FULL_PURGE = auto()


@dataclass
class DeletionRequest:
    """Request to delete data."""
    request_id: str
    scope: DeletionScope
    requested_by: str
    requested_at: datetime
    consent_id: Optional[str] = None
    blob_ids: List[str] = field(default_factory=list)
    tier: Optional[EncryptionTier] = None
    reason: str = ""
    secure: bool = True
    verify: bool = True


@dataclass
class DeletionResult:
    """Result of a deletion operation."""
    request: DeletionRequest
    status: DeletionStatus
    blobs_deleted: int = 0
    blobs_failed: int = 0
    keys_deleted: int = 0
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    errors: List[str] = field(default_factory=list)
    verified: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request.request_id,
            "scope": self.request.scope.name,
            "status": self.status.name,
            "blobs_deleted": self.blobs_deleted,
            "blobs_failed": self.blobs_failed,
            "keys_deleted": self.keys_deleted,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "errors": self.errors,
            "verified": self.verified,
        }


class DeletionManager:
    """
    Manages right-to-delete operations for the Memory Vault.

    Features:
    - Cascading deletion by consent
    - Secure data wiping (overwrite before delete)
    - Deletion verification
    - Audit trail maintenance
    """

    def __init__(
        self,
        storage: BlobStorage,
        index: VaultIndex,
        key_manager: KeyManager,
        verification_callback: Optional[Callable[[str], bool]] = None,
    ):
        """
        Initialize deletion manager.

        Args:
            storage: Blob storage instance
            index: Vault index instance
            key_manager: Key manager instance
            verification_callback: Optional callback to verify deletion
        """
        self._storage = storage
        self._index = index
        self._key_manager = key_manager
        self._verification_callback = verification_callback

        self._pending_requests: Dict[str, DeletionRequest] = {}
        self._results: Dict[str, DeletionResult] = {}
        self._lock = threading.RLock()

    def request_deletion(
        self,
        scope: DeletionScope,
        requested_by: str,
        consent_id: Optional[str] = None,
        blob_ids: Optional[List[str]] = None,
        tier: Optional[EncryptionTier] = None,
        reason: str = "",
        secure: bool = True,
        verify: bool = True,
    ) -> DeletionRequest:
        """
        Create a deletion request.

        Args:
            scope: Scope of deletion
            requested_by: Who is requesting
            consent_id: Consent ID for cascade deletion
            blob_ids: Specific blob IDs to delete
            tier: Tier to purge
            reason: Reason for deletion
            secure: Use secure deletion
            verify: Verify deletion

        Returns:
            Created DeletionRequest
        """
        import secrets
        request_id = f"del_{secrets.token_hex(16)}"

        request = DeletionRequest(
            request_id=request_id,
            scope=scope,
            requested_by=requested_by,
            requested_at=datetime.now(),
            consent_id=consent_id,
            blob_ids=blob_ids or [],
            tier=tier,
            reason=reason,
            secure=secure,
            verify=verify,
        )

        with self._lock:
            self._pending_requests[request_id] = request

        # Log the request
        self._index.log_access(
            blob_id="*",
            access_type=AccessType.DELETE,
            accessor=requested_by,
            success=True,
            details={
                "request_id": request_id,
                "scope": scope.name,
                "consent_id": consent_id,
                "reason": reason,
            },
        )

        logger.info(
            f"Deletion request created: {request_id} "
            f"(scope={scope.name}, by={requested_by})"
        )
        return request

    def execute_deletion(self, request_id: str) -> DeletionResult:
        """
        Execute a deletion request.

        Args:
            request_id: Request ID to execute

        Returns:
            DeletionResult with outcome
        """
        with self._lock:
            request = self._pending_requests.get(request_id)
            if not request:
                raise ValueError(f"Deletion request not found: {request_id}")

        result = DeletionResult(
            request=request,
            status=DeletionStatus.IN_PROGRESS,
            started_at=datetime.now(),
        )

        try:
            # Get blobs to delete based on scope
            blob_ids = self._get_blobs_for_deletion(request)

            logger.info(
                f"Executing deletion {request_id}: "
                f"{len(blob_ids)} blobs to delete"
            )

            # Delete each blob
            for blob_id in blob_ids:
                try:
                    success = self._storage.delete(blob_id, secure=request.secure)
                    if success:
                        result.blobs_deleted += 1
                        result.keys_deleted += 1  # Key deleted with blob
                        self._index.remove_blob_index(blob_id)
                    else:
                        result.blobs_failed += 1
                        result.errors.append(f"Failed to delete blob: {blob_id}")
                except Exception as e:
                    result.blobs_failed += 1
                    result.errors.append(f"Error deleting {blob_id}: {str(e)}")

            # Verify deletion if requested
            if request.verify and result.blobs_deleted > 0:
                result.verified = self._verify_deletion(blob_ids)

            # Determine final status
            if result.blobs_failed == 0:
                result.status = DeletionStatus.COMPLETED
            elif result.blobs_deleted > 0:
                result.status = DeletionStatus.PARTIAL
            else:
                result.status = DeletionStatus.FAILED

            result.completed_at = datetime.now()

        except Exception as e:
            logger.error(f"Deletion execution failed: {e}")
            result.status = DeletionStatus.FAILED
            result.errors.append(str(e))
            result.completed_at = datetime.now()

        with self._lock:
            del self._pending_requests[request_id]
            self._results[request_id] = result

        # Update deletion queue if consent-based
        if request.consent_id:
            queue_id = self._index.queue_deletion(request.consent_id)
            self._index.complete_deletion(
                queue_id,
                success=(result.status == DeletionStatus.COMPLETED),
                error_message="; ".join(result.errors) if result.errors else None,
            )

        logger.info(
            f"Deletion completed: {request_id} "
            f"(deleted={result.blobs_deleted}, failed={result.blobs_failed})"
        )
        return result

    def delete_by_consent(
        self,
        consent_id: str,
        requested_by: str,
        reason: str = "Right to forget",
    ) -> DeletionResult:
        """
        Delete all blobs associated with a consent (right to forget).

        Args:
            consent_id: Consent ID
            requested_by: Who is requesting
            reason: Reason for deletion

        Returns:
            DeletionResult
        """
        request = self.request_deletion(
            scope=DeletionScope.CONSENT_CASCADE,
            requested_by=requested_by,
            consent_id=consent_id,
            reason=reason,
            secure=True,
            verify=True,
        )
        return self.execute_deletion(request.request_id)

    def delete_blob(
        self,
        blob_id: str,
        requested_by: str,
        secure: bool = True,
    ) -> DeletionResult:
        """
        Delete a single blob.

        Args:
            blob_id: Blob to delete
            requested_by: Who is requesting
            secure: Use secure deletion

        Returns:
            DeletionResult
        """
        request = self.request_deletion(
            scope=DeletionScope.SINGLE_BLOB,
            requested_by=requested_by,
            blob_ids=[blob_id],
            secure=secure,
        )
        return self.execute_deletion(request.request_id)

    def purge_tier(
        self,
        tier: EncryptionTier,
        requested_by: str,
        reason: str = "Tier purge",
    ) -> DeletionResult:
        """
        Purge all blobs in a tier.

        Args:
            tier: Tier to purge
            requested_by: Who is requesting
            reason: Reason for purge

        Returns:
            DeletionResult
        """
        request = self.request_deletion(
            scope=DeletionScope.TIER_PURGE,
            requested_by=requested_by,
            tier=tier,
            reason=reason,
            secure=True,
        )
        return self.execute_deletion(request.request_id)

    def purge_all(
        self,
        requested_by: str,
        reason: str = "Full vault purge",
        confirmation_code: Optional[str] = None,
    ) -> DeletionResult:
        """
        Purge entire vault (destructive operation).

        Args:
            requested_by: Who is requesting
            reason: Reason for purge
            confirmation_code: Required confirmation code

        Returns:
            DeletionResult
        """
        # Require confirmation for full purge
        if not confirmation_code or confirmation_code != "CONFIRM_FULL_PURGE":
            raise ValueError(
                "Full purge requires confirmation code 'CONFIRM_FULL_PURGE'"
            )

        request = self.request_deletion(
            scope=DeletionScope.FULL_PURGE,
            requested_by=requested_by,
            reason=reason,
            secure=True,
        )
        return self.execute_deletion(request.request_id)

    def get_pending_requests(self) -> List[DeletionRequest]:
        """Get pending deletion requests."""
        with self._lock:
            return list(self._pending_requests.values())

    def get_result(self, request_id: str) -> Optional[DeletionResult]:
        """Get deletion result."""
        with self._lock:
            return self._results.get(request_id)

    def get_deletion_history(
        self,
        limit: int = 100,
    ) -> List[DeletionResult]:
        """Get deletion history."""
        with self._lock:
            results = list(self._results.values())
            results.sort(key=lambda r: r.completed_at or datetime.min, reverse=True)
            return results[:limit]

    def _get_blobs_for_deletion(
        self,
        request: DeletionRequest,
    ) -> List[str]:
        """Get list of blob IDs to delete based on request scope."""
        if request.scope == DeletionScope.SINGLE_BLOB:
            return request.blob_ids

        elif request.scope == DeletionScope.CONSENT_CASCADE:
            blobs = self._index.query_blobs(
                consent_id=request.consent_id,
                status=BlobStatus.ACTIVE,
                limit=10000,
            )
            return [b.blob_id for b in blobs]

        elif request.scope == DeletionScope.TIER_PURGE:
            blobs = self._index.query_blobs(
                tier=request.tier,
                status=BlobStatus.ACTIVE,
                limit=10000,
            )
            return [b.blob_id for b in blobs]

        elif request.scope == DeletionScope.FULL_PURGE:
            blobs = self._index.query_blobs(
                status=BlobStatus.ACTIVE,
                limit=100000,
            )
            return [b.blob_id for b in blobs]

        return []

    def _verify_deletion(self, blob_ids: List[str]) -> bool:
        """Verify that blobs have been deleted."""
        for blob_id in blob_ids:
            # Check storage
            metadata = self._storage.get_metadata(blob_id)
            if metadata and metadata.status == BlobStatus.ACTIVE:
                logger.error(f"Deletion verification failed: {blob_id} still active")
                return False

            # Check index
            indexed = self._index.get_blob(blob_id)
            if indexed and indexed.status == BlobStatus.ACTIVE:
                logger.error(f"Deletion verification failed: {blob_id} still indexed")
                return False

            # Custom verification callback
            if self._verification_callback:
                if not self._verification_callback(blob_id):
                    logger.error(f"Custom verification failed: {blob_id}")
                    return False

        return True


class TTLEnforcer:
    """
    Enforces time-to-live expiry on blobs.

    Runs periodically to delete expired blobs.
    """

    def __init__(
        self,
        deletion_manager: DeletionManager,
        index: VaultIndex,
        check_interval: int = 3600,  # 1 hour
    ):
        """
        Initialize TTL enforcer.

        Args:
            deletion_manager: Deletion manager instance
            index: Vault index instance
            check_interval: Check interval in seconds
        """
        self._deletion_manager = deletion_manager
        self._index = index
        self._check_interval = check_interval

        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    def start(self) -> None:
        """Start TTL enforcement."""
        if self._running:
            return

        self._running = True
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._enforcement_loop, daemon=True)
        self._thread.start()
        logger.info("TTL enforcer started")

    def stop(self) -> None:
        """Stop TTL enforcement."""
        if not self._running:
            return

        self._running = False
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)
        logger.info("TTL enforcer stopped")

    def enforce_now(self) -> int:
        """
        Enforce TTL immediately.

        Returns:
            Number of blobs deleted
        """
        now = datetime.now()
        count = 0

        # Query blobs with TTL
        blobs = self._index.query_blobs(status=BlobStatus.ACTIVE, limit=10000)

        for blob in blobs:
            if blob.ttl_seconds:
                expiry = blob.created_at.timestamp() + blob.ttl_seconds
                if now.timestamp() > expiry:
                    result = self._deletion_manager.delete_blob(
                        blob.blob_id,
                        requested_by="ttl_enforcer",
                        secure=True,
                    )
                    if result.status == DeletionStatus.COMPLETED:
                        count += 1

        if count > 0:
            logger.info(f"TTL enforcer deleted {count} expired blobs")

        return count

    def _enforcement_loop(self) -> None:
        """Main enforcement loop."""
        while not self._stop_event.is_set():
            try:
                self.enforce_now()
            except Exception as e:
                logger.error(f"TTL enforcement error: {e}")

            self._stop_event.wait(self._check_interval)
