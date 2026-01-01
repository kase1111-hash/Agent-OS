"""
Storage Integration for Attack Detection Components

Provides integration between the attack detection components
and persistent storage, enabling:
- Automatic persistence of detected attacks
- Durable recommendation workflow
- Historical analysis and reporting
- Recovery after restarts
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

from .storage import (
    AttackStorage,
    StorageBackend,
    StoredAttack,
    StoredPatch,
    StoredRecommendation,
    StoredSIEMEvent,
    StoredVulnerability,
    create_storage,
)

logger = logging.getLogger(__name__)


class StorageIntegration:
    """
    Integrates attack detection components with persistent storage.

    This class acts as a bridge between the attack detection system
    and the storage layer, automatically persisting attacks,
    recommendations, patches, and vulnerabilities.
    """

    def __init__(
        self,
        storage: Optional[AttackStorage] = None,
        backend: StorageBackend = StorageBackend.SQLITE,
        db_path: Optional[Union[str, Path]] = None,
    ):
        """
        Initialize storage integration.

        Args:
            storage: Pre-configured storage instance
            backend: Storage backend type (if storage not provided)
            db_path: Database path (for SQLite backend)
        """
        if storage:
            self._storage = storage
        else:
            self._storage = create_storage(backend=backend, db_path=db_path)

        self._attack_callbacks: List[Callable[[StoredAttack], None]] = []
        self._recommendation_callbacks: List[Callable[[StoredRecommendation], None]] = []

        logger.info(f"Storage integration initialized with {backend.value} backend")

    @property
    def storage(self) -> AttackStorage:
        """Get the underlying storage instance."""
        return self._storage

    # -------------------------------------------------------------------------
    # Attack Persistence
    # -------------------------------------------------------------------------

    def persist_attack(
        self,
        attack_id: str,
        attack_type: str,
        severity: int,
        severity_name: str,
        confidence: float,
        description: str,
        source: str,
        target: Optional[str] = None,
        indicators: Optional[List[str]] = None,
        mitre_tactics: Optional[List[str]] = None,
        raw_data: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> StoredAttack:
        """
        Persist a detected attack.

        Args:
            attack_id: Unique attack identifier
            attack_type: Type of attack (e.g., "PROMPT_INJECTION")
            severity: Numeric severity (1-5)
            severity_name: Human-readable severity
            confidence: Detection confidence (0.0-1.0)
            description: Attack description
            source: Attack source
            target: Attack target (optional)
            indicators: Attack indicators (optional)
            mitre_tactics: MITRE ATT&CK tactics (optional)
            raw_data: Raw event data (optional)
            metadata: Additional metadata (optional)

        Returns:
            The persisted StoredAttack
        """
        attack = StoredAttack(
            attack_id=attack_id,
            attack_type=attack_type,
            severity=severity,
            severity_name=severity_name,
            confidence=confidence,
            description=description,
            source=source,
            target=target,
            detected_at=datetime.now(),
            indicators=indicators or [],
            mitre_tactics=mitre_tactics or [],
            raw_data=raw_data or {},
            metadata=metadata or {},
        )

        self._storage.save_attack(attack)
        logger.info(f"Persisted attack: {attack_id}")

        # Notify callbacks
        for callback in self._attack_callbacks:
            try:
                callback(attack)
            except Exception as e:
                logger.error(f"Attack callback error: {e}")

        return attack

    def persist_attack_from_event(self, attack_event: Any) -> StoredAttack:
        """
        Persist an AttackEvent from the detector.

        Args:
            attack_event: AttackEvent from the detector

        Returns:
            The persisted StoredAttack
        """
        return self.persist_attack(
            attack_id=attack_event.attack_id,
            attack_type=attack_event.attack_type.name,
            severity=attack_event.severity.value,
            severity_name=attack_event.severity.name,
            confidence=attack_event.confidence,
            description=attack_event.description,
            source=attack_event.source,
            target=getattr(attack_event, "target", None),
            indicators=getattr(attack_event, "indicators", []),
            mitre_tactics=getattr(attack_event, "mitre_tactics", []),
            raw_data=getattr(attack_event, "raw_data", {}),
            metadata=getattr(attack_event, "metadata", {}),
        )

    def update_attack_status(
        self,
        attack_id: str,
        status: str,
        notes: str = "",
    ) -> bool:
        """
        Update attack status.

        Args:
            attack_id: Attack to update
            status: New status (detected, analyzing, mitigated, false_positive, ignored)
            notes: Status notes

        Returns:
            True if successful
        """
        return self._storage.update_attack_status(attack_id, status, notes)

    def mark_attack_mitigated(
        self,
        attack_id: str,
        recommendation_id: Optional[str] = None,
        notes: str = "",
    ) -> bool:
        """
        Mark an attack as mitigated.

        Args:
            attack_id: Attack to mark
            recommendation_id: Recommendation that mitigated it
            notes: Mitigation notes

        Returns:
            True if successful
        """
        attack = self._storage.get_attack(attack_id)
        if attack:
            attack.status = "mitigated"
            attack.mitigated_at = datetime.now()
            attack.mitigation_notes = notes
            if recommendation_id:
                attack.recommendation_id = recommendation_id
            return self._storage.save_attack(attack)
        return False

    def mark_attack_false_positive(
        self,
        attack_id: str,
        notes: str = "",
    ) -> bool:
        """
        Mark an attack as a false positive.

        Args:
            attack_id: Attack to mark
            notes: Notes explaining why it's a false positive

        Returns:
            True if successful
        """
        return self._storage.update_attack_status(attack_id, "false_positive", notes)

    # -------------------------------------------------------------------------
    # Recommendation Persistence
    # -------------------------------------------------------------------------

    def persist_recommendation(
        self,
        recommendation_id: str,
        attack_id: str,
        title: str,
        description: str,
        priority: int,
        priority_name: str,
        patch_ids: Optional[List[str]] = None,
        vulnerability_ids: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> StoredRecommendation:
        """
        Persist a fix recommendation.

        Args:
            recommendation_id: Unique recommendation ID
            attack_id: Associated attack ID
            title: Recommendation title
            description: Detailed description
            priority: Priority level (1-5)
            priority_name: Human-readable priority
            patch_ids: Associated patch IDs
            vulnerability_ids: Associated vulnerability IDs
            metadata: Additional metadata

        Returns:
            The persisted StoredRecommendation
        """
        rec = StoredRecommendation(
            recommendation_id=recommendation_id,
            attack_id=attack_id,
            title=title,
            description=description,
            priority=priority,
            priority_name=priority_name,
            created_at=datetime.now(),
            patch_ids=patch_ids or [],
            vulnerability_ids=vulnerability_ids or [],
            metadata=metadata or {},
        )

        self._storage.save_recommendation(rec)
        logger.info(f"Persisted recommendation: {recommendation_id}")

        # Update attack with recommendation link
        attack = self._storage.get_attack(attack_id)
        if attack:
            attack.recommendation_id = recommendation_id
            self._storage.save_attack(attack)

        # Notify callbacks
        for callback in self._recommendation_callbacks:
            try:
                callback(rec)
            except Exception as e:
                logger.error(f"Recommendation callback error: {e}")

        return rec

    def persist_recommendation_from_fix(self, fix_recommendation: Any) -> StoredRecommendation:
        """
        Persist a FixRecommendation from the recommendation system.

        Args:
            fix_recommendation: FixRecommendation from the system

        Returns:
            The persisted StoredRecommendation
        """
        # Extract patch IDs
        patch_ids = []
        if hasattr(fix_recommendation, "patches"):
            for patch in fix_recommendation.patches:
                patch_id = getattr(patch, "patch_id", None)
                if patch_id:
                    patch_ids.append(patch_id)

        return self.persist_recommendation(
            recommendation_id=fix_recommendation.recommendation_id,
            attack_id=fix_recommendation.attack_id,
            title=fix_recommendation.title,
            description=fix_recommendation.description,
            priority=fix_recommendation.priority.value,
            priority_name=fix_recommendation.priority.name,
            patch_ids=patch_ids,
            vulnerability_ids=getattr(fix_recommendation, "vulnerability_ids", []),
            metadata=getattr(fix_recommendation, "metadata", {}),
        )

    def approve_recommendation(
        self,
        recommendation_id: str,
        reviewer: str,
        comments: str = "",
    ) -> bool:
        """
        Approve a recommendation.

        Args:
            recommendation_id: Recommendation to approve
            reviewer: Who approved it
            comments: Approval comments

        Returns:
            True if successful
        """
        return self._storage.update_recommendation_status(
            recommendation_id, "approved", reviewer, comments
        )

    def reject_recommendation(
        self,
        recommendation_id: str,
        reviewer: str,
        comments: str = "",
    ) -> bool:
        """
        Reject a recommendation.

        Args:
            recommendation_id: Recommendation to reject
            reviewer: Who rejected it
            comments: Rejection reason

        Returns:
            True if successful
        """
        return self._storage.update_recommendation_status(
            recommendation_id, "rejected", reviewer, comments
        )

    def mark_recommendation_applied(
        self,
        recommendation_id: str,
        applied_by: str,
        result: str = "",
    ) -> bool:
        """
        Mark a recommendation as applied.

        Args:
            recommendation_id: Recommendation that was applied
            applied_by: Who applied it
            result: Application result

        Returns:
            True if successful
        """
        rec = self._storage.get_recommendation(recommendation_id)
        if rec:
            rec.status = "applied"
            rec.applied_at = datetime.now()
            rec.applied_by = applied_by
            rec.application_result = result
            if self._storage.save_recommendation(rec):
                # Also mark the attack as mitigated
                self.mark_attack_mitigated(
                    rec.attack_id, recommendation_id, f"Applied: {result}"
                )
                return True
        return False

    # -------------------------------------------------------------------------
    # Patch Persistence
    # -------------------------------------------------------------------------

    def persist_patch(
        self,
        patch_id: str,
        recommendation_id: str,
        file_path: str,
        patch_type: str,
        description: str,
        original_content: str = "",
        patched_content: str = "",
        diff: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> StoredPatch:
        """
        Persist a patch.

        Args:
            patch_id: Unique patch ID
            recommendation_id: Associated recommendation
            file_path: File being patched
            patch_type: Type of patch (add, modify, delete)
            description: Patch description
            original_content: Original file content
            patched_content: Patched file content
            diff: Unified diff
            metadata: Additional metadata

        Returns:
            The persisted StoredPatch
        """
        patch = StoredPatch(
            patch_id=patch_id,
            recommendation_id=recommendation_id,
            file_path=file_path,
            patch_type=patch_type,
            description=description,
            created_at=datetime.now(),
            original_content=original_content,
            patched_content=patched_content,
            diff=diff,
            metadata=metadata or {},
        )

        self._storage.save_patch(patch)
        logger.info(f"Persisted patch: {patch_id}")

        # Update recommendation with patch link
        rec = self._storage.get_recommendation(recommendation_id)
        if rec and patch_id not in rec.patch_ids:
            rec.patch_ids.append(patch_id)
            self._storage.save_recommendation(rec)

        return patch

    def persist_patch_from_remediation(
        self,
        patch: Any,
        recommendation_id: str,
    ) -> StoredPatch:
        """
        Persist a Patch from the remediation engine.

        Args:
            patch: Patch from remediation engine
            recommendation_id: Associated recommendation

        Returns:
            The persisted StoredPatch
        """
        return self.persist_patch(
            patch_id=patch.patch_id,
            recommendation_id=recommendation_id,
            file_path=str(patch.file_path),
            patch_type=patch.patch_type.name if hasattr(patch.patch_type, "name") else str(patch.patch_type),
            description=patch.description,
            original_content=getattr(patch, "original_content", ""),
            patched_content=getattr(patch, "patched_content", ""),
            diff=getattr(patch, "diff", ""),
            metadata=getattr(patch, "metadata", {}),
        )

    def update_patch_test_result(
        self,
        patch_id: str,
        test_passed: bool,
        test_output: str = "",
    ) -> bool:
        """
        Update patch with test result.

        Args:
            patch_id: Patch that was tested
            test_passed: Whether tests passed
            test_output: Test output

        Returns:
            True if successful
        """
        status = "tested" if test_passed else "failed"
        return self._storage.update_patch_status(patch_id, status, test_passed, test_output)

    def mark_patch_applied(
        self,
        patch_id: str,
        applied_by: str,
    ) -> bool:
        """
        Mark a patch as applied.

        Args:
            patch_id: Patch that was applied
            applied_by: Who applied it

        Returns:
            True if successful
        """
        patch = self._storage.get_patch(patch_id)
        if patch:
            patch.status = "applied"
            patch.applied_at = datetime.now()
            patch.applied_by = applied_by
            return self._storage.save_patch(patch)
        return False

    def mark_patch_reverted(self, patch_id: str) -> bool:
        """
        Mark a patch as reverted.

        Args:
            patch_id: Patch that was reverted

        Returns:
            True if successful
        """
        patch = self._storage.get_patch(patch_id)
        if patch:
            patch.status = "reverted"
            patch.reverted_at = datetime.now()
            return self._storage.save_patch(patch)
        return False

    # -------------------------------------------------------------------------
    # Vulnerability Persistence
    # -------------------------------------------------------------------------

    def persist_vulnerability(
        self,
        vulnerability_id: str,
        attack_id: str,
        file_path: str,
        line_start: int,
        line_end: int,
        vulnerability_type: str,
        description: str,
        severity: int,
        code_snippet: str = "",
        suggested_fix: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> StoredVulnerability:
        """
        Persist a vulnerability finding.

        Args:
            vulnerability_id: Unique vulnerability ID
            attack_id: Associated attack
            file_path: File containing vulnerability
            line_start: Starting line number
            line_end: Ending line number
            vulnerability_type: Type of vulnerability
            description: Vulnerability description
            severity: Severity level (1-5)
            code_snippet: Vulnerable code snippet
            suggested_fix: Suggested fix
            metadata: Additional metadata

        Returns:
            The persisted StoredVulnerability
        """
        vuln = StoredVulnerability(
            vulnerability_id=vulnerability_id,
            attack_id=attack_id,
            file_path=file_path,
            line_start=line_start,
            line_end=line_end,
            vulnerability_type=vulnerability_type,
            description=description,
            severity=severity,
            created_at=datetime.now(),
            code_snippet=code_snippet,
            suggested_fix=suggested_fix,
            metadata=metadata or {},
        )

        self._storage.save_vulnerability(vuln)
        logger.info(f"Persisted vulnerability: {vulnerability_id}")

        # Update attack with vulnerability link
        attack = self._storage.get_attack(attack_id)
        if attack and vulnerability_id not in attack.vulnerability_ids:
            attack.vulnerability_ids.append(vulnerability_id)
            self._storage.save_attack(attack)

        return vuln

    def persist_vulnerability_from_report(
        self,
        finding: Any,
        attack_id: str,
    ) -> StoredVulnerability:
        """
        Persist a vulnerability finding from analysis.

        Args:
            finding: Finding from VulnerabilityReport
            attack_id: Associated attack

        Returns:
            The persisted StoredVulnerability
        """
        return self.persist_vulnerability(
            vulnerability_id=finding.finding_id,
            attack_id=attack_id,
            file_path=str(finding.location.file_path),
            line_start=finding.location.line_start,
            line_end=finding.location.line_end,
            vulnerability_type=finding.vulnerability_type,
            description=finding.description,
            severity=finding.severity,
            code_snippet=getattr(finding, "code_snippet", ""),
            suggested_fix=getattr(finding, "suggested_fix", ""),
            metadata=getattr(finding, "metadata", {}),
        )

    def mark_vulnerability_fixed(
        self,
        vulnerability_id: str,
        fixed_by: str,
    ) -> bool:
        """
        Mark a vulnerability as fixed.

        Args:
            vulnerability_id: Vulnerability that was fixed
            fixed_by: Who fixed it

        Returns:
            True if successful
        """
        vuln = self._storage.get_vulnerability(vulnerability_id)
        if vuln:
            vuln.status = "fixed"
            vuln.fixed_at = datetime.now()
            vuln.fixed_by = fixed_by
            return self._storage.save_vulnerability(vuln)
        return False

    # -------------------------------------------------------------------------
    # SIEM Event Persistence
    # -------------------------------------------------------------------------

    def persist_siem_event(
        self,
        event_id: str,
        provider: str,
        timestamp: datetime,
        source: str,
        event_type: str,
        severity: int,
        category: str,
        description: str,
        raw_data: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        indicators: Optional[List[str]] = None,
    ) -> StoredSIEMEvent:
        """
        Persist a SIEM event.

        Args:
            event_id: Unique event ID
            provider: SIEM provider name
            timestamp: Event timestamp
            source: Event source
            event_type: Event type
            severity: Severity level
            category: Event category
            description: Event description
            raw_data: Raw event data
            metadata: Additional metadata
            indicators: Threat indicators

        Returns:
            The persisted StoredSIEMEvent
        """
        event = StoredSIEMEvent(
            event_id=event_id,
            provider=provider,
            timestamp=timestamp,
            source=source,
            event_type=event_type,
            severity=severity,
            category=category,
            description=description,
            raw_data=raw_data or {},
            metadata=metadata or {},
            indicators=indicators or [],
        )

        self._storage.save_siem_event(event)
        return event

    def persist_siem_event_from_connector(self, siem_event: Any) -> StoredSIEMEvent:
        """
        Persist a SIEMEvent from the connector.

        Args:
            siem_event: SIEMEvent from connector

        Returns:
            The persisted StoredSIEMEvent
        """
        return self.persist_siem_event(
            event_id=siem_event.event_id,
            provider=siem_event.metadata.get("provider", "unknown"),
            timestamp=siem_event.timestamp,
            source=siem_event.source,
            event_type=siem_event.event_type,
            severity=siem_event.severity.value if hasattr(siem_event.severity, "value") else int(siem_event.severity),
            category=siem_event.category,
            description=siem_event.description,
            raw_data=siem_event.raw_data,
            metadata=siem_event.metadata,
            indicators=getattr(siem_event, "indicators", []),
        )

    def mark_siem_events_processed(
        self,
        event_ids: List[str],
        attack_id: Optional[str] = None,
    ) -> int:
        """
        Mark SIEM events as processed.

        Args:
            event_ids: Events to mark
            attack_id: Associated attack (if any)

        Returns:
            Number of events marked
        """
        return self._storage.mark_siem_events_processed(event_ids, attack_id)

    # -------------------------------------------------------------------------
    # Queries
    # -------------------------------------------------------------------------

    def get_attack(self, attack_id: str) -> Optional[StoredAttack]:
        """Get an attack by ID."""
        return self._storage.get_attack(attack_id)

    def get_recommendation(self, rec_id: str) -> Optional[StoredRecommendation]:
        """Get a recommendation by ID."""
        return self._storage.get_recommendation(rec_id)

    def get_patch(self, patch_id: str) -> Optional[StoredPatch]:
        """Get a patch by ID."""
        return self._storage.get_patch(patch_id)

    def get_vulnerability(self, vuln_id: str) -> Optional[StoredVulnerability]:
        """Get a vulnerability by ID."""
        return self._storage.get_vulnerability(vuln_id)

    def list_attacks(
        self,
        since: Optional[datetime] = None,
        status: Optional[str] = None,
        severity_min: Optional[int] = None,
        limit: int = 100,
    ) -> List[StoredAttack]:
        """List attacks with optional filters."""
        return self._storage.list_attacks(
            since=since,
            status=status,
            severity_min=severity_min,
            limit=limit,
        )

    def list_pending_recommendations(self, limit: int = 100) -> List[StoredRecommendation]:
        """Get pending recommendations."""
        return self._storage.list_recommendations(status="pending", limit=limit)

    def list_approved_recommendations(self, limit: int = 100) -> List[StoredRecommendation]:
        """Get approved recommendations."""
        return self._storage.list_recommendations(status="approved", limit=limit)

    def list_patches_for_recommendation(
        self,
        recommendation_id: str,
    ) -> List[StoredPatch]:
        """Get patches for a recommendation."""
        return self._storage.list_patches(recommendation_id=recommendation_id)

    def list_vulnerabilities_for_attack(
        self,
        attack_id: str,
    ) -> List[StoredVulnerability]:
        """Get vulnerabilities for an attack."""
        return self._storage.list_vulnerabilities(attack_id=attack_id)

    def list_unprocessed_siem_events(self, limit: int = 100) -> List[StoredSIEMEvent]:
        """Get unprocessed SIEM events."""
        return self._storage.list_siem_events(processed=False, limit=limit)

    # -------------------------------------------------------------------------
    # Statistics
    # -------------------------------------------------------------------------

    def get_statistics(self) -> Dict[str, Any]:
        """Get storage statistics."""
        return self._storage.get_statistics()

    def get_attack_summary(self) -> Dict[str, Any]:
        """Get attack summary statistics."""
        stats = self._storage.get_statistics()
        return {
            "total_attacks": stats.get("total_attacks", 0),
            "attacks_by_status": stats.get("attacks_by_status", {}),
            "attacks_last_24h": stats.get("attacks_last_24h", 0),
            "total_recommendations": stats.get("total_recommendations", 0),
            "recommendations_by_status": stats.get("recommendations_by_status", {}),
        }

    # -------------------------------------------------------------------------
    # Callbacks
    # -------------------------------------------------------------------------

    def on_attack_persisted(self, callback: Callable[[StoredAttack], None]) -> None:
        """Register callback for when attacks are persisted."""
        self._attack_callbacks.append(callback)

    def on_recommendation_persisted(
        self,
        callback: Callable[[StoredRecommendation], None],
    ) -> None:
        """Register callback for when recommendations are persisted."""
        self._recommendation_callbacks.append(callback)

    # -------------------------------------------------------------------------
    # Maintenance
    # -------------------------------------------------------------------------

    def cleanup_old_records(
        self,
        days_to_keep: int = 90,
        keep_unresolved: bool = True,
    ) -> Dict[str, int]:
        """
        Clean up old records.

        Args:
            days_to_keep: Keep records newer than this
            keep_unresolved: Keep unresolved attacks regardless of age

        Returns:
            Count of deleted records by type
        """
        from datetime import timedelta

        cutoff = datetime.now() - timedelta(days=days_to_keep)
        return self._storage.cleanup_old_records(cutoff, keep_unresolved)

    def close(self) -> None:
        """Close storage connection."""
        self._storage.close()


def create_storage_integration(
    backend: StorageBackend = StorageBackend.SQLITE,
    db_path: Optional[Union[str, Path]] = None,
) -> StorageIntegration:
    """
    Create a storage integration instance.

    Args:
        backend: Storage backend type
        db_path: Database path (for SQLite)

    Returns:
        Configured StorageIntegration
    """
    return StorageIntegration(backend=backend, db_path=db_path)
