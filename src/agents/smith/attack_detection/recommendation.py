"""
Recommendation System

Packages attack analysis and remediation into human-reviewable recommendations.
Creates pull request-style recommendations that can be reviewed and approved
by human operators.

Key Features:
1. Generate comprehensive fix recommendations
2. Create PR-ready descriptions and changelogs
3. Track recommendation lifecycle
4. Support for batch approval workflows
"""

import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set

from .analyzer import VulnerabilityReport, VulnerabilityFinding, RiskLevel
from .detector import AttackEvent, AttackType, AttackSeverity
from .remediation import (
    Patch,
    PatchStatus,
    RemediationPlan,
    RemediationEngine,
    PatchType,
)
from .git_integration import (
    GitIntegration,
    PullRequestInfo,
    PRStatus,
)

logger = logging.getLogger(__name__)


class RecommendationStatus(Enum):
    """Status of a fix recommendation."""

    DRAFT = auto()  # Being prepared
    PENDING = auto()  # Waiting for review
    UNDER_REVIEW = auto()  # Being reviewed
    APPROVED = auto()  # Approved for implementation
    PARTIALLY_APPROVED = auto()  # Some patches approved
    REJECTED = auto()  # Rejected
    IMPLEMENTED = auto()  # Changes applied
    SUPERSEDED = auto()  # Replaced by newer recommendation


class Priority(Enum):
    """Priority levels for recommendations."""

    LOW = 1
    MEDIUM = 2
    HIGH = 3
    URGENT = 4
    CRITICAL = 5


@dataclass
class ReviewComment:
    """A review comment on a recommendation."""

    comment_id: str
    author: str
    content: str
    created_at: datetime
    line_ref: str = ""  # Optional file:line reference
    resolved: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "comment_id": self.comment_id,
            "author": self.author,
            "content": self.content,
            "created_at": self.created_at.isoformat(),
            "line_ref": self.line_ref,
            "resolved": self.resolved,
        }


@dataclass
class ApprovalRecord:
    """Record of an approval decision."""

    approver: str
    decision: str  # "approve", "reject", "request_changes"
    timestamp: datetime
    comments: str = ""
    patch_ids: List[str] = field(default_factory=list)  # Specific patches

    def to_dict(self) -> Dict[str, Any]:
        return {
            "approver": self.approver,
            "decision": self.decision,
            "timestamp": self.timestamp.isoformat(),
            "comments": self.comments,
            "patch_ids": self.patch_ids,
        }


@dataclass
class FixRecommendation:
    """
    A complete fix recommendation for human review.

    Packages all information about an attack and its remediation
    into a format suitable for human review and approval.
    """

    recommendation_id: str
    status: RecommendationStatus
    priority: Priority
    created_at: datetime

    # Attack context
    attack_id: str
    attack_type: AttackType
    attack_severity: AttackSeverity
    attack_summary: str

    # Analysis
    report_id: str
    vulnerability_summary: str
    risk_score: float

    # Remediation
    plan_id: str
    patches: List[Patch] = field(default_factory=list)

    # PR-style content
    title: str = ""
    description: str = ""
    change_summary: str = ""
    testing_notes: str = ""
    rollback_plan: str = ""

    # Review tracking
    assigned_reviewers: List[str] = field(default_factory=list)
    review_comments: List[ReviewComment] = field(default_factory=list)
    approvals: List[ApprovalRecord] = field(default_factory=list)

    # Metrics
    files_changed: int = 0
    lines_added: int = 0
    lines_removed: int = 0

    # Metadata
    labels: List[str] = field(default_factory=list)
    related_recommendations: List[str] = field(default_factory=list)
    updated_at: datetime = field(default_factory=datetime.now)

    # Git integration
    pr_number: Optional[int] = None
    pr_url: Optional[str] = None
    pr_branch: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "recommendation_id": self.recommendation_id,
            "status": self.status.name,
            "priority": self.priority.name,
            "created_at": self.created_at.isoformat(),
            "attack_id": self.attack_id,
            "attack_type": self.attack_type.name,
            "attack_severity": self.attack_severity.name,
            "attack_summary": self.attack_summary,
            "report_id": self.report_id,
            "vulnerability_summary": self.vulnerability_summary,
            "risk_score": self.risk_score,
            "plan_id": self.plan_id,
            "patches": [p.to_dict() for p in self.patches],
            "title": self.title,
            "description": self.description,
            "change_summary": self.change_summary,
            "testing_notes": self.testing_notes,
            "rollback_plan": self.rollback_plan,
            "assigned_reviewers": self.assigned_reviewers,
            "review_comments": [c.to_dict() for c in self.review_comments],
            "approvals": [a.to_dict() for a in self.approvals],
            "files_changed": self.files_changed,
            "lines_added": self.lines_added,
            "lines_removed": self.lines_removed,
            "labels": self.labels,
            "related_recommendations": self.related_recommendations,
            "updated_at": self.updated_at.isoformat(),
            "pr_number": self.pr_number,
            "pr_url": self.pr_url,
            "pr_branch": self.pr_branch,
        }

    def get_full_diff(self) -> str:
        """Get combined diff from all patches."""
        diffs = []
        for patch in self.patches:
            diff = patch.get_full_diff()
            if diff:
                diffs.append(f"# Patch: {patch.patch_id} - {patch.title}\n{diff}")
        return "\n\n".join(diffs)

    def to_markdown(self) -> str:
        """Generate markdown representation for human review."""
        md = []

        # Header
        md.append(f"# {self.title}")
        md.append("")
        md.append(f"**Recommendation ID:** {self.recommendation_id}")
        md.append(f"**Status:** {self.status.name}")
        md.append(f"**Priority:** {self.priority.name}")
        md.append(f"**Created:** {self.created_at.strftime('%Y-%m-%d %H:%M')}")
        md.append("")

        # Labels
        if self.labels:
            md.append(f"**Labels:** {', '.join(self.labels)}")
            md.append("")

        # Attack Context
        md.append("## Attack Context")
        md.append("")
        md.append(f"- **Attack ID:** {self.attack_id}")
        md.append(f"- **Type:** {self.attack_type.name}")
        md.append(f"- **Severity:** {self.attack_severity.name}")
        md.append("")
        md.append(self.attack_summary)
        md.append("")

        # Vulnerability Summary
        md.append("## Vulnerability Analysis")
        md.append("")
        md.append(f"**Risk Score:** {self.risk_score:.1f}/10")
        md.append("")
        md.append(self.vulnerability_summary)
        md.append("")

        # Description
        md.append("## Description")
        md.append("")
        md.append(self.description)
        md.append("")

        # Changes
        md.append("## Changes")
        md.append("")
        md.append(f"- **Files Changed:** {self.files_changed}")
        md.append(f"- **Lines Added:** +{self.lines_added}")
        md.append(f"- **Lines Removed:** -{self.lines_removed}")
        md.append("")
        md.append(self.change_summary)
        md.append("")

        # Patches
        md.append("## Patches")
        md.append("")
        for patch in self.patches:
            status_emoji = {
                PatchStatus.PENDING_REVIEW: ":hourglass:",
                PatchStatus.APPROVED: ":white_check_mark:",
                PatchStatus.REJECTED: ":x:",
                PatchStatus.TEST_PASSED: ":test_tube:",
                PatchStatus.APPLIED: ":rocket:",
            }.get(patch.status, ":question:")

            md.append(f"### {status_emoji} {patch.title}")
            md.append("")
            md.append(f"- **Patch ID:** {patch.patch_id}")
            md.append(f"- **Type:** {patch.patch_type.name}")
            md.append(f"- **Status:** {patch.status.name}")
            md.append(f"- **Risk Level:** {patch.risk_level.name}")
            md.append("")
            md.append(patch.description)
            md.append("")

            # Show diff preview
            diff = patch.get_full_diff()
            if diff:
                md.append("```diff")
                # Truncate long diffs
                if len(diff) > 2000:
                    md.append(diff[:2000])
                    md.append("... (truncated)")
                else:
                    md.append(diff)
                md.append("```")
                md.append("")

        # Testing Notes
        md.append("## Testing Notes")
        md.append("")
        md.append(self.testing_notes or "*No testing notes provided.*")
        md.append("")

        # Rollback Plan
        md.append("## Rollback Plan")
        md.append("")
        md.append(self.rollback_plan or "*Standard rollback: revert commits.*")
        md.append("")

        # Review Comments
        if self.review_comments:
            md.append("## Review Comments")
            md.append("")
            for comment in self.review_comments:
                status = ":white_check_mark:" if comment.resolved else ":speech_balloon:"
                md.append(f"{status} **{comment.author}** ({comment.created_at.strftime('%Y-%m-%d %H:%M')}):")
                if comment.line_ref:
                    md.append(f"  > Re: `{comment.line_ref}`")
                md.append(f"  {comment.content}")
                md.append("")

        # Approvals
        if self.approvals:
            md.append("## Approvals")
            md.append("")
            for approval in self.approvals:
                emoji = ":white_check_mark:" if approval.decision == "approve" else ":x:"
                md.append(f"- {emoji} **{approval.approver}**: {approval.decision}")
                if approval.comments:
                    md.append(f"  > {approval.comments}")
            md.append("")

        return "\n".join(md)


class RecommendationSystem:
    """
    System for generating and managing fix recommendations.
    """

    def __init__(
        self,
        remediation_engine: Optional[RemediationEngine] = None,
        git_integration: Optional[GitIntegration] = None,
        on_recommendation: Optional[Callable[[FixRecommendation], None]] = None,
        auto_create_pr: bool = False,
        pr_draft_mode: bool = True,
    ):
        """
        Initialize recommendation system.

        Args:
            remediation_engine: Engine for generating patches
            git_integration: Git integration for auto-creating PRs
            on_recommendation: Callback when recommendation created
            auto_create_pr: Automatically create PRs when recommendations are approved
            pr_draft_mode: Create PRs as drafts (requires manual publish)
        """
        self.remediation_engine = remediation_engine
        self.git_integration = git_integration
        self.on_recommendation = on_recommendation
        self.auto_create_pr = auto_create_pr
        self.pr_draft_mode = pr_draft_mode

        self._recommendations: Dict[str, FixRecommendation] = {}
        self._by_attack: Dict[str, str] = {}  # attack_id -> recommendation_id

    def create_recommendation(
        self,
        attack: AttackEvent,
        report: VulnerabilityReport,
        plan: RemediationPlan,
    ) -> FixRecommendation:
        """
        Create a fix recommendation from attack analysis.

        Args:
            attack: The detected attack
            report: Vulnerability analysis report
            plan: Remediation plan with patches

        Returns:
            FixRecommendation ready for review
        """
        rec_id = f"REC-{hashlib.sha256(attack.attack_id.encode()).hexdigest()[:12].upper()}"

        # Determine priority from attack severity
        priority_mapping = {
            AttackSeverity.LOW: Priority.LOW,
            AttackSeverity.MEDIUM: Priority.MEDIUM,
            AttackSeverity.HIGH: Priority.HIGH,
            AttackSeverity.CRITICAL: Priority.URGENT,
            AttackSeverity.CATASTROPHIC: Priority.CRITICAL,
        }
        priority = priority_mapping.get(attack.severity, Priority.MEDIUM)

        # Generate title
        title = f"[{attack.severity.name}] Fix for {attack.attack_type.name} attack"

        # Generate description
        description = self._generate_description(attack, report, plan)

        # Generate change summary
        change_summary = self._generate_change_summary(plan)

        # Generate testing notes
        testing_notes = self._generate_testing_notes(attack, report)

        # Generate rollback plan
        rollback_plan = self._generate_rollback_plan(plan)

        # Calculate metrics
        files_changed = len({f.file_path for p in plan.patches for f in p.files})
        lines_added = sum(
            p.patched_content.count("\n") - p.original_content.count("\n")
            for patch in plan.patches
            for p in patch.files
            if p.patched_content.count("\n") > p.original_content.count("\n")
        )
        lines_removed = sum(
            p.original_content.count("\n") - p.patched_content.count("\n")
            for patch in plan.patches
            for p in patch.files
            if p.original_content.count("\n") > p.patched_content.count("\n")
        )

        # Determine labels
        labels = self._determine_labels(attack, report)

        recommendation = FixRecommendation(
            recommendation_id=rec_id,
            status=RecommendationStatus.PENDING,
            priority=priority,
            created_at=datetime.now(),
            attack_id=attack.attack_id,
            attack_type=attack.attack_type,
            attack_severity=attack.severity,
            attack_summary=attack.description,
            report_id=report.report_id,
            vulnerability_summary=report.summary,
            risk_score=report.risk_score,
            plan_id=plan.plan_id,
            patches=plan.patches.copy(),
            title=title,
            description=description,
            change_summary=change_summary,
            testing_notes=testing_notes,
            rollback_plan=rollback_plan,
            files_changed=files_changed,
            lines_added=abs(lines_added),
            lines_removed=abs(lines_removed),
            labels=labels,
        )

        # Store recommendation
        self._recommendations[rec_id] = recommendation
        self._by_attack[attack.attack_id] = rec_id

        # Trigger callback
        if self.on_recommendation:
            try:
                self.on_recommendation(recommendation)
            except Exception as e:
                logger.error(f"Recommendation callback error: {e}")

        logger.info(
            f"Created recommendation {rec_id} for attack {attack.attack_id}"
        )

        return recommendation

    def get_recommendation(self, rec_id: str) -> Optional[FixRecommendation]:
        """Get a recommendation by ID."""
        return self._recommendations.get(rec_id)

    def get_recommendation_for_attack(self, attack_id: str) -> Optional[FixRecommendation]:
        """Get recommendation for a specific attack."""
        rec_id = self._by_attack.get(attack_id)
        if rec_id:
            return self._recommendations.get(rec_id)
        return None

    def list_recommendations(
        self,
        status: Optional[RecommendationStatus] = None,
        priority: Optional[Priority] = None,
    ) -> List[FixRecommendation]:
        """List recommendations with optional filtering."""
        recommendations = list(self._recommendations.values())

        if status:
            recommendations = [r for r in recommendations if r.status == status]

        if priority:
            recommendations = [r for r in recommendations if r.priority == priority]

        # Sort by priority (highest first), then by creation time
        return sorted(
            recommendations,
            key=lambda r: (r.priority.value, r.created_at.timestamp()),
            reverse=True,
        )

    def assign_reviewers(
        self,
        rec_id: str,
        reviewers: List[str],
    ) -> bool:
        """Assign reviewers to a recommendation."""
        rec = self._recommendations.get(rec_id)
        if not rec:
            return False

        rec.assigned_reviewers.extend(reviewers)
        rec.status = RecommendationStatus.UNDER_REVIEW
        rec.updated_at = datetime.now()

        logger.info(f"Assigned reviewers {reviewers} to {rec_id}")
        return True

    def add_comment(
        self,
        rec_id: str,
        author: str,
        content: str,
        line_ref: str = "",
    ) -> Optional[str]:
        """
        Add a review comment.

        Returns:
            Comment ID if successful
        """
        rec = self._recommendations.get(rec_id)
        if not rec:
            return None

        comment_id = hashlib.sha256(
            f"{rec_id}:{author}:{datetime.now().isoformat()}".encode()
        ).hexdigest()[:8]

        comment = ReviewComment(
            comment_id=comment_id,
            author=author,
            content=content,
            created_at=datetime.now(),
            line_ref=line_ref,
        )

        rec.review_comments.append(comment)
        rec.updated_at = datetime.now()

        return comment_id

    def resolve_comment(self, rec_id: str, comment_id: str) -> bool:
        """Mark a comment as resolved."""
        rec = self._recommendations.get(rec_id)
        if not rec:
            return False

        for comment in rec.review_comments:
            if comment.comment_id == comment_id:
                comment.resolved = True
                rec.updated_at = datetime.now()
                return True

        return False

    def approve(
        self,
        rec_id: str,
        approver: str,
        comments: str = "",
        patch_ids: Optional[List[str]] = None,
    ) -> bool:
        """
        Approve a recommendation (or specific patches).

        Args:
            rec_id: Recommendation ID
            approver: Who is approving
            comments: Approval comments
            patch_ids: Specific patches to approve (None = all)

        Returns:
            True if successful
        """
        rec = self._recommendations.get(rec_id)
        if not rec:
            return False

        target_patches = patch_ids or [p.patch_id for p in rec.patches]

        approval = ApprovalRecord(
            approver=approver,
            decision="approve",
            timestamp=datetime.now(),
            comments=comments,
            patch_ids=target_patches,
        )
        rec.approvals.append(approval)

        # Update patch statuses
        if self.remediation_engine:
            for patch_id in target_patches:
                self.remediation_engine.approve_patch(patch_id, approver, comments)

        # Update recommendation status
        if patch_ids:
            # Partial approval
            all_patch_ids = {p.patch_id for p in rec.patches}
            approved_ids = {pid for a in rec.approvals for pid in a.patch_ids if a.decision == "approve"}
            if approved_ids == all_patch_ids:
                rec.status = RecommendationStatus.APPROVED
            else:
                rec.status = RecommendationStatus.PARTIALLY_APPROVED
        else:
            rec.status = RecommendationStatus.APPROVED

        rec.updated_at = datetime.now()

        logger.info(f"Recommendation {rec_id} approved by {approver}")
        return True

    def reject(
        self,
        rec_id: str,
        rejector: str,
        reason: str,
    ) -> bool:
        """Reject a recommendation."""
        rec = self._recommendations.get(rec_id)
        if not rec:
            return False

        approval = ApprovalRecord(
            approver=rejector,
            decision="reject",
            timestamp=datetime.now(),
            comments=reason,
        )
        rec.approvals.append(approval)
        rec.status = RecommendationStatus.REJECTED
        rec.updated_at = datetime.now()

        # Update patch statuses
        if self.remediation_engine:
            for patch in rec.patches:
                self.remediation_engine.reject_patch(patch.patch_id, rejector, reason)

        logger.info(f"Recommendation {rec_id} rejected by {rejector}: {reason}")
        return True

    def request_changes(
        self,
        rec_id: str,
        reviewer: str,
        changes_requested: str,
    ) -> bool:
        """Request changes to a recommendation."""
        rec = self._recommendations.get(rec_id)
        if not rec:
            return False

        approval = ApprovalRecord(
            approver=reviewer,
            decision="request_changes",
            timestamp=datetime.now(),
            comments=changes_requested,
        )
        rec.approvals.append(approval)
        rec.status = RecommendationStatus.UNDER_REVIEW
        rec.updated_at = datetime.now()

        logger.info(f"Changes requested on {rec_id} by {reviewer}")
        return True

    def implement(self, rec_id: str) -> Tuple[bool, str]:
        """
        Implement an approved recommendation.

        Returns:
            Tuple of (success, message)
        """
        rec = self._recommendations.get(rec_id)
        if not rec:
            return False, "Recommendation not found"

        if rec.status != RecommendationStatus.APPROVED:
            return False, f"Recommendation is not approved (status: {rec.status.name})"

        if not self.remediation_engine:
            return False, "No remediation engine configured"

        # Apply all approved patches
        all_success = True
        messages = []

        for patch in rec.patches:
            success, message = self.remediation_engine.apply_patch(patch.patch_id)
            if not success:
                all_success = False
                messages.append(f"{patch.patch_id}: {message}")
            else:
                messages.append(f"{patch.patch_id}: Applied successfully")

        if all_success:
            rec.status = RecommendationStatus.IMPLEMENTED
            rec.updated_at = datetime.now()
            logger.info(f"Recommendation {rec_id} fully implemented")
            return True, "All patches applied successfully"
        else:
            logger.warning(f"Partial implementation of {rec_id}")
            return False, "; ".join(messages)

    async def create_pull_request(
        self,
        rec_id: str,
        draft: Optional[bool] = None,
        additional_reviewers: Optional[List[str]] = None,
    ) -> Tuple[bool, Optional[PullRequestInfo]]:
        """
        Create a pull request for a recommendation.

        Args:
            rec_id: Recommendation ID
            draft: Whether to create as draft (overrides pr_draft_mode)
            additional_reviewers: Additional reviewers to add

        Returns:
            Tuple of (success, PullRequestInfo or None)
        """
        rec = self._recommendations.get(rec_id)
        if not rec:
            logger.error(f"Recommendation not found: {rec_id}")
            return False, None

        if not self.git_integration:
            logger.error("Git integration not configured")
            return False, None

        if rec.status not in [
            RecommendationStatus.APPROVED,
            RecommendationStatus.PARTIALLY_APPROVED,
        ]:
            logger.warning(
                f"Recommendation {rec_id} is not approved (status: {rec.status.name})"
            )
            return False, None

        try:
            # Collect all patch content and files
            affected_files = []
            combined_patch = []
            for patch in rec.patches:
                for file_info in patch.files:
                    affected_files.append(file_info.file_path)
                    # Create unified diff content
                    combined_patch.append(
                        f"# Patch: {patch.patch_id}\n"
                        f"# File: {file_info.file_path}\n"
                        f"{patch.get_full_diff()}"
                    )

            # Get first patch file for primary application
            primary_file = affected_files[0] if affected_files else "unknown"
            primary_patch = rec.patches[0] if rec.patches else None

            if not primary_patch:
                logger.error("No patches available in recommendation")
                return False, None

            # Create the PR
            use_draft = draft if draft is not None else self.pr_draft_mode

            pr_info, commit_info = await self.git_integration.create_fix_pr_from_recommendation(
                attack_id=rec.attack_id,
                attack_type=rec.attack_type.name,
                severity=rec.attack_severity.name,
                description=rec.description,
                patch_content="\n".join(combined_patch),
                file_path=primary_file,
                recommendation_id=rec.recommendation_id,
                analysis_summary=rec.vulnerability_summary,
                draft=use_draft,
            )

            # Update recommendation with PR info
            rec.pr_number = pr_info.number
            rec.pr_url = pr_info.url
            rec.pr_branch = pr_info.branch
            rec.updated_at = datetime.now()

            logger.info(
                f"Created PR #{pr_info.number} for recommendation {rec_id}: {pr_info.url}"
            )

            return True, pr_info

        except Exception as e:
            logger.error(f"Failed to create PR for {rec_id}: {e}")
            return False, None

    async def get_pr_status(
        self,
        rec_id: str,
    ) -> Optional[PRStatus]:
        """
        Get the current status of a recommendation's pull request.

        Args:
            rec_id: Recommendation ID

        Returns:
            PRStatus or None if no PR exists
        """
        rec = self._recommendations.get(rec_id)
        if not rec or not rec.pr_number:
            return None

        if not self.git_integration:
            return None

        return await self.git_integration.get_pr_status(rec.attack_id)

    def export_recommendation(
        self,
        rec_id: str,
        output_path: Path,
        format: str = "markdown",
    ) -> bool:
        """
        Export a recommendation to a file.

        Args:
            rec_id: Recommendation ID
            output_path: Output file path
            format: "markdown" or "json"

        Returns:
            True if successful
        """
        rec = self._recommendations.get(rec_id)
        if not rec:
            return False

        try:
            if format == "markdown":
                content = rec.to_markdown()
            elif format == "json":
                content = json.dumps(rec.to_dict(), indent=2)
            else:
                logger.warning(f"Unknown format: {format}")
                return False

            output_path.write_text(content)
            logger.info(f"Exported recommendation {rec_id} to {output_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to export recommendation: {e}")
            return False

    def _generate_description(
        self,
        attack: AttackEvent,
        report: VulnerabilityReport,
        plan: RemediationPlan,
    ) -> str:
        """Generate detailed description."""
        lines = [
            "## Summary",
            "",
            f"This recommendation addresses a **{attack.attack_type.name}** attack "
            f"detected at {attack.detected_at.strftime('%Y-%m-%d %H:%M')}.",
            "",
            "## Background",
            "",
            attack.description,
            "",
            "## Impact Assessment",
            "",
            f"- **Risk Score:** {report.risk_score:.1f}/10",
            f"- **Confidence:** {attack.confidence:.0%}",
            f"- **Affected Components:** {', '.join(report.affected_components) or 'TBD'}",
            "",
        ]

        if report.attack_chain:
            lines.extend([
                "## Attack Chain",
                "",
                *[f"{i}. {step}" for i, step in enumerate(report.attack_chain, 1)],
                "",
            ])

        if attack.mitre_techniques:
            lines.extend([
                "## MITRE ATT&CK Mapping",
                "",
                *[f"- {tech}" for tech in attack.mitre_techniques],
                "",
            ])

        return "\n".join(lines)

    def _generate_change_summary(self, plan: RemediationPlan) -> str:
        """Generate summary of changes."""
        lines = []

        code_patches = [p for p in plan.patches if p.patch_type == PatchType.CODE_FIX]
        pattern_patches = [p for p in plan.patches if p.patch_type == PatchType.PATTERN_UPDATE]
        const_patches = [p for p in plan.patches if p.patch_type == PatchType.CONSTITUTIONAL_AMENDMENT]

        if code_patches:
            lines.append("### Code Changes")
            for patch in code_patches:
                lines.append(f"- {patch.title}")
            lines.append("")

        if pattern_patches:
            lines.append("### Attack Pattern Updates")
            for patch in pattern_patches:
                lines.append(f"- {patch.title}")
            lines.append("")

        if const_patches:
            lines.append("### Constitutional Amendments")
            for patch in const_patches:
                lines.append(f"- {patch.title}")
            lines.append("")

        return "\n".join(lines) or "No changes defined."

    def _generate_testing_notes(
        self,
        attack: AttackEvent,
        report: VulnerabilityReport,
    ) -> str:
        """Generate testing notes."""
        lines = [
            "### Pre-merge Testing",
            "",
            "1. Run the full test suite: `pytest tests/`",
            "2. Run security-specific tests: `pytest tests/test_smith.py tests/test_boundary.py`",
            "",
            "### Post-merge Verification",
            "",
            f"1. Verify the {attack.attack_type.name} attack pattern is blocked",
            "2. Check Smith metrics for proper detection",
            "3. Review boundary daemon event log",
            "",
            "### Regression Testing",
            "",
            "Ensure no false positives for legitimate requests matching these patterns.",
        ]

        return "\n".join(lines)

    def _generate_rollback_plan(self, plan: RemediationPlan) -> str:
        """Generate rollback plan."""
        lines = [
            "### Immediate Rollback",
            "",
            "If issues are detected after deployment:",
            "",
            "1. Revert the commit(s) containing these changes",
            "2. Re-run the test suite to verify rollback",
            "",
            "### File Backups",
            "",
        ]

        for patch in plan.patches:
            for f in patch.files:
                lines.append(f"- `{f.file_path}.bak` (auto-created on patch application)")

        lines.extend([
            "",
            "### Manual Rollback",
            "",
            "Pattern and constitutional changes can be reverted by removing",
            "the auto-generated entries marked with the attack ID.",
        ])

        return "\n".join(lines)

    def _determine_labels(
        self,
        attack: AttackEvent,
        report: VulnerabilityReport,
    ) -> List[str]:
        """Determine labels for the recommendation."""
        labels = []

        # Priority label
        if attack.severity in [AttackSeverity.CRITICAL, AttackSeverity.CATASTROPHIC]:
            labels.append("priority:critical")
        elif attack.severity == AttackSeverity.HIGH:
            labels.append("priority:high")

        # Type labels
        labels.append(f"attack:{attack.attack_type.name.lower()}")

        # Auto-generated label
        labels.append("auto-generated")

        # Security label
        labels.append("security")

        # Component labels
        for component in report.affected_components[:3]:
            labels.append(f"component:{component}")

        return labels


def create_recommendation_system(
    remediation_engine: Optional[RemediationEngine] = None,
    git_integration: Optional[GitIntegration] = None,
    on_recommendation: Optional[Callable[[FixRecommendation], None]] = None,
    auto_create_pr: bool = False,
    pr_draft_mode: bool = True,
) -> RecommendationSystem:
    """
    Factory function to create a recommendation system.

    Args:
        remediation_engine: Engine for generating patches
        git_integration: Git integration for auto-creating PRs
        on_recommendation: Callback when recommendation created
        auto_create_pr: Automatically create PRs when recommendations are approved
        pr_draft_mode: Create PRs as drafts (requires manual publish)

    Returns:
        Configured RecommendationSystem
    """
    return RecommendationSystem(
        remediation_engine=remediation_engine,
        git_integration=git_integration,
        on_recommendation=on_recommendation,
        auto_create_pr=auto_create_pr,
        pr_draft_mode=pr_draft_mode,
    )
