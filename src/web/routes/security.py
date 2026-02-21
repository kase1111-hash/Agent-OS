"""
Security API Routes

Provides endpoints for attack detection, security monitoring, and
fix recommendations from Agent Smith's attack detection system.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, BackgroundTasks
from pydantic import BaseModel, Field

from ..auth_helpers import require_admin_user

logger = logging.getLogger(__name__)
router = APIRouter()

# Try to import attack detection components
try:
    from src.agents.smith.attack_detection import (
        AttackDetector,
        AttackEvent,
        AttackSeverity,
        AttackType,
        RecommendationStatus,
        create_attack_detector,
    )
    ATTACK_DETECTION_AVAILABLE = True
except ImportError:
    ATTACK_DETECTION_AVAILABLE = False

# Try to import Smith agent
try:
    from src.agents.smith.agent import SmithAgent, create_smith
    SMITH_AVAILABLE = True
except ImportError:
    SMITH_AVAILABLE = False


# =============================================================================
# Models
# =============================================================================


class AttackSummary(BaseModel):
    """Summary of a detected attack."""

    attack_id: str
    attack_type: str
    severity: str
    status: str
    detected_at: datetime
    description: str
    confidence: float
    source: Optional[str] = None


class AttackDetail(BaseModel):
    """Detailed attack information."""

    attack_id: str
    attack_type: str
    severity: str
    status: str
    detected_at: datetime
    description: str
    confidence: float
    source: Optional[str] = None
    indicators_of_compromise: List[str] = Field(default_factory=list)
    affected_agents: List[str] = Field(default_factory=list)
    raw_event: Optional[Dict[str, Any]] = None
    pattern_matches: List[str] = Field(default_factory=list)


class RecommendationSummary(BaseModel):
    """Summary of a fix recommendation."""

    recommendation_id: str
    attack_id: str
    title: str
    priority: str
    status: str
    created_at: datetime
    patch_count: int


class RecommendationDetail(BaseModel):
    """Detailed fix recommendation."""

    recommendation_id: str
    attack_id: str
    report_id: str
    plan_id: str
    title: str
    description: str
    priority: str
    status: str
    created_at: datetime
    updated_at: Optional[datetime] = None
    patches: List[Dict[str, Any]] = Field(default_factory=list)
    reviewers: List[str] = Field(default_factory=list)
    comments: List[Dict[str, Any]] = Field(default_factory=list)
    approval_info: Optional[Dict[str, Any]] = None


class ApproveRecommendationRequest(BaseModel):
    """Request to approve a recommendation."""

    approver: str
    comments: str = ""


class RejectRecommendationRequest(BaseModel):
    """Request to reject a recommendation."""

    rejector: str
    reason: str


class AddCommentRequest(BaseModel):
    """Request to add a comment to a recommendation."""

    author: str
    content: str


class AttackDetectionStatus(BaseModel):
    """Attack detection system status."""

    enabled: bool
    available: bool
    pipeline_running: bool
    detector_stats: Optional[Dict[str, Any]] = None
    attacks_detected: int
    attacks_mitigated: int
    recommendations_generated: int
    auto_lockdowns_triggered: int


class PipelineControlRequest(BaseModel):
    """Request to control the attack detection pipeline."""

    action: str = Field(..., pattern="^(start|stop)$")


class MarkFalsePositiveRequest(BaseModel):
    """Request to mark an attack as false positive."""

    reason: str


# =============================================================================
# State Management
# =============================================================================


# Global Smith instance for API (initialized on first use)
_smith_instance: Optional[Any] = None


def get_smith() -> Any:
    """Get or create Smith agent instance."""
    global _smith_instance

    if _smith_instance is None:
        if not SMITH_AVAILABLE:
            raise HTTPException(
                status_code=503,
                detail="Smith agent not available"
            )
        try:
            _smith_instance = create_smith(config={
                "attack_detection_enabled": True,
                "attack_detection_config": {
                    "enable_boundary_events": True,
                    "enable_flow_monitoring": True,
                }
            })
        except Exception as e:
            logger.error(f"Failed to create Smith agent: {e}")
            raise HTTPException(
                status_code=503,
                detail="Failed to initialize security agent. Check server logs for details."
            )

    return _smith_instance


# =============================================================================
# Attack Endpoints
# =============================================================================


@router.get("/attacks", response_model=List[AttackSummary])
async def list_attacks(
    severity: Optional[str] = Query(None, description="Filter by severity"),
    attack_type: Optional[str] = Query(None, description="Filter by attack type"),
    status: Optional[str] = Query(None, description="Filter by status"),
    since: Optional[datetime] = Query(None, description="Only attacks after this time"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum results"),
    offset: int = Query(0, ge=0, description="Skip first N results"),
    admin_id: str = Depends(require_admin_user),
) -> List[AttackSummary]:
    """
    List detected attacks.

    Returns a paginated list of attacks detected by the attack detection system.
    """
    try:
        smith = get_smith()
        attacks = smith.get_detected_attacks(since=since)

        # Apply filters first, then paginate
        filtered = []
        for attack in attacks:
            if severity and attack.get("severity") != severity:
                continue
            if attack_type and attack.get("attack_type") != attack_type:
                continue
            if status and attack.get("status") != status:
                continue

            filtered.append(AttackSummary(
                attack_id=attack["attack_id"],
                attack_type=attack.get("attack_type", "UNKNOWN"),
                severity=attack.get("severity", "MEDIUM"),
                status=attack.get("status", "DETECTED"),
                detected_at=attack.get("detected_at", datetime.now()),
                description=attack.get("description", ""),
                confidence=attack.get("confidence", 0.5),
                source=attack.get("source"),
            ))

        # Apply pagination after filtering
        filtered = filtered[offset:offset + limit]

        return filtered

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error listing attacks: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/attacks/{attack_id}", response_model=AttackDetail)
async def get_attack(attack_id: str, admin_id: str = Depends(require_admin_user)) -> AttackDetail:
    """
    Get detailed information about a specific attack.
    """
    try:
        smith = get_smith()

        # Find the attack
        attacks = smith.get_detected_attacks(limit=1000)
        attack = next((a for a in attacks if a["attack_id"] == attack_id), None)

        if not attack:
            raise HTTPException(status_code=404, detail="Attack not found")

        return AttackDetail(
            attack_id=attack["attack_id"],
            attack_type=attack.get("attack_type", "UNKNOWN"),
            severity=attack.get("severity", "MEDIUM"),
            status=attack.get("status", "DETECTED"),
            detected_at=attack.get("detected_at", datetime.now()),
            description=attack.get("description", ""),
            confidence=attack.get("confidence", 0.5),
            source=attack.get("source"),
            indicators_of_compromise=attack.get("indicators_of_compromise", []),
            affected_agents=attack.get("affected_agents", []),
            raw_event=attack.get("raw_event"),
            pattern_matches=attack.get("pattern_matches", []),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting attack {attack_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/attacks/{attack_id}/false-positive")
async def mark_attack_false_positive(
    attack_id: str,
    request: MarkFalsePositiveRequest,
    admin_id: str = Depends(require_admin_user),
) -> Dict[str, Any]:
    """
    Mark an attack as a false positive.

    This updates the attack status and can be used to tune detection patterns.
    """
    try:
        smith = get_smith()

        if not smith._attack_detector:
            raise HTTPException(
                status_code=503,
                detail="Attack detector not enabled"
            )

        result = smith._attack_detector.mark_false_positive(
            attack_id,
            request.reason
        )

        if not result:
            raise HTTPException(status_code=404, detail="Attack not found")

        return {
            "status": "success",
            "attack_id": attack_id,
            "message": "Attack marked as false positive",
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error marking false positive: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Recommendation Endpoints
# =============================================================================


@router.get("/recommendations", response_model=List[RecommendationSummary])
async def list_recommendations(
    status: Optional[str] = Query(None, description="Filter by status"),
    priority: Optional[str] = Query(None, description="Filter by priority"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum results"),
    admin_id: str = Depends(require_admin_user),
) -> List[RecommendationSummary]:
    """
    List fix recommendations.

    Returns recommendations generated by the attack detection system,
    with optional filtering by status and priority.
    """
    try:
        smith = get_smith()

        if not smith._recommendation_system:
            return []

        # Get all recommendations
        recommendations = []

        if status:
            try:
                status_enum = RecommendationStatus[status.upper()]
                recs = smith._recommendation_system.list_recommendations(status=status_enum)
            except KeyError:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid status: {status}"
                )
        else:
            recs = smith._recommendation_system.list_recommendations()

        for rec in recs[:limit]:
            rec_dict = rec.to_dict() if hasattr(rec, 'to_dict') else rec

            if priority and rec_dict.get("priority") != priority.upper():
                continue

            recommendations.append(RecommendationSummary(
                recommendation_id=rec_dict["recommendation_id"],
                attack_id=rec_dict["attack_id"],
                title=rec_dict.get("title", ""),
                priority=rec_dict.get("priority", "MEDIUM"),
                status=rec_dict.get("status", "PENDING"),
                created_at=rec_dict.get("created_at", datetime.now()),
                patch_count=len(rec_dict.get("patches", [])),
            ))

        return recommendations

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error listing recommendations: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/recommendations/{recommendation_id}", response_model=RecommendationDetail)
async def get_recommendation(recommendation_id: str, admin_id: str = Depends(require_admin_user)) -> RecommendationDetail:
    """
    Get detailed information about a fix recommendation.

    Includes patches, reviewers, comments, and approval status.
    """
    try:
        smith = get_smith()

        if not smith._recommendation_system:
            raise HTTPException(
                status_code=503,
                detail="Recommendation system not enabled"
            )

        rec = smith._recommendation_system.get_recommendation(recommendation_id)

        if not rec:
            raise HTTPException(status_code=404, detail="Recommendation not found")

        rec_dict = rec.to_dict() if hasattr(rec, 'to_dict') else rec

        return RecommendationDetail(
            recommendation_id=rec_dict["recommendation_id"],
            attack_id=rec_dict["attack_id"],
            report_id=rec_dict.get("report_id", ""),
            plan_id=rec_dict.get("plan_id", ""),
            title=rec_dict.get("title", ""),
            description=rec_dict.get("description", ""),
            priority=rec_dict.get("priority", "MEDIUM"),
            status=rec_dict.get("status", "PENDING"),
            created_at=rec_dict.get("created_at", datetime.now()),
            updated_at=rec_dict.get("updated_at"),
            patches=rec_dict.get("patches", []),
            reviewers=rec_dict.get("reviewers", []),
            comments=rec_dict.get("comments", []),
            approval_info=rec_dict.get("approval_info"),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting recommendation {recommendation_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/recommendations/{recommendation_id}/markdown")
async def get_recommendation_markdown(recommendation_id: str, admin_id: str = Depends(require_admin_user)) -> Dict[str, str]:
    """
    Get recommendation formatted as markdown for human review.

    This is useful for generating PR descriptions or review documents.
    """
    try:
        smith = get_smith()

        if not smith._recommendation_system:
            raise HTTPException(
                status_code=503,
                detail="Recommendation system not enabled"
            )

        rec = smith._recommendation_system.get_recommendation(recommendation_id)

        if not rec:
            raise HTTPException(status_code=404, detail="Recommendation not found")

        markdown = rec.to_markdown()

        return {
            "recommendation_id": recommendation_id,
            "markdown": markdown,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting recommendation markdown: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/recommendations/{recommendation_id}/approve")
async def approve_recommendation(
    recommendation_id: str,
    request: ApproveRecommendationRequest,
    admin_id: str = Depends(require_admin_user),
) -> Dict[str, Any]:
    """
    Approve a fix recommendation.

    Approved recommendations can be applied to the codebase.
    """
    try:
        smith = get_smith()

        result = smith.approve_recommendation(
            recommendation_id,
            request.approver,
            request.comments,
        )

        if not result:
            raise HTTPException(
                status_code=400,
                detail="Failed to approve recommendation"
            )

        return {
            "status": "success",
            "recommendation_id": recommendation_id,
            "message": "Recommendation approved",
            "approver": request.approver,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error approving recommendation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/recommendations/{recommendation_id}/reject")
async def reject_recommendation(
    recommendation_id: str,
    request: RejectRecommendationRequest,
    admin_id: str = Depends(require_admin_user),
) -> Dict[str, Any]:
    """
    Reject a fix recommendation.

    Rejected recommendations are archived with the rejection reason.
    """
    try:
        smith = get_smith()

        if not smith._recommendation_system:
            raise HTTPException(
                status_code=503,
                detail="Recommendation system not enabled"
            )

        result = smith._recommendation_system.reject(
            recommendation_id,
            request.rejector,
            request.reason,
        )

        if not result:
            raise HTTPException(
                status_code=400,
                detail="Failed to reject recommendation"
            )

        return {
            "status": "success",
            "recommendation_id": recommendation_id,
            "message": "Recommendation rejected",
            "rejector": request.rejector,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error rejecting recommendation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/recommendations/{recommendation_id}/comments")
async def add_recommendation_comment(
    recommendation_id: str,
    request: AddCommentRequest,
    admin_id: str = Depends(require_admin_user),
) -> Dict[str, Any]:
    """
    Add a comment to a recommendation.

    Comments are used during the review process.
    """
    try:
        smith = get_smith()

        if not smith._recommendation_system:
            raise HTTPException(
                status_code=503,
                detail="Recommendation system not enabled"
            )

        comment_id = smith._recommendation_system.add_comment(
            recommendation_id,
            request.author,
            request.content,
        )

        if not comment_id:
            raise HTTPException(
                status_code=400,
                detail="Failed to add comment"
            )

        return {
            "status": "success",
            "recommendation_id": recommendation_id,
            "comment_id": comment_id,
            "message": "Comment added",
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error adding comment: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/recommendations/{recommendation_id}/assign")
async def assign_reviewers(
    recommendation_id: str,
    reviewers: List[str],
    admin_id: str = Depends(require_admin_user),
) -> Dict[str, Any]:
    """
    Assign reviewers to a recommendation.

    This moves the recommendation to UNDER_REVIEW status.
    """
    try:
        smith = get_smith()

        if not smith._recommendation_system:
            raise HTTPException(
                status_code=503,
                detail="Recommendation system not enabled"
            )

        smith._recommendation_system.assign_reviewers(
            recommendation_id,
            reviewers,
        )

        return {
            "status": "success",
            "recommendation_id": recommendation_id,
            "message": f"Assigned {len(reviewers)} reviewer(s)",
            "reviewers": reviewers,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error assigning reviewers: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# System Status Endpoints
# =============================================================================


@router.get("/status", response_model=AttackDetectionStatus)
async def get_attack_detection_status(admin_id: str = Depends(require_admin_user)) -> AttackDetectionStatus:
    """
    Get attack detection system status.

    Returns current status of the attack detection pipeline,
    detector statistics, and counts of detected attacks.
    """
    try:
        smith = get_smith()
        status = smith.get_attack_detection_status()

        return AttackDetectionStatus(
            enabled=status.get("enabled", False),
            available=status.get("available", ATTACK_DETECTION_AVAILABLE),
            pipeline_running=status.get("enabled", False),
            detector_stats=status.get("detector"),
            attacks_detected=status.get("attacks_detected", 0),
            attacks_mitigated=status.get("attacks_mitigated", 0),
            recommendations_generated=status.get("recommendations_generated", 0),
            auto_lockdowns_triggered=status.get("auto_lockdowns_triggered", 0),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/pipeline")
async def control_pipeline(
    request: PipelineControlRequest,
    background_tasks: BackgroundTasks,
    admin_id: str = Depends(require_admin_user),
) -> Dict[str, Any]:
    """
    Control the attack detection pipeline.

    Start or stop the attack detection system.
    """
    try:
        smith = get_smith()

        if request.action == "start":
            if not smith._attack_detector:
                raise HTTPException(
                    status_code=503,
                    detail="Attack detector not initialized"
                )
            smith._attack_detector.start()
            smith._attack_detection_enabled = True
            message = "Attack detection pipeline started"

        elif request.action == "stop":
            if smith._attack_detector:
                smith._attack_detector.stop()
            smith._attack_detection_enabled = False
            message = "Attack detection pipeline stopped"

        return {
            "status": "success",
            "action": request.action,
            "message": message,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error controlling pipeline: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/patterns")
async def list_attack_patterns(admin_id: str = Depends(require_admin_user)) -> Dict[str, Any]:
    """
    List available attack detection patterns.

    Returns the patterns used by the attack detector,
    including their status (enabled/disabled).
    """
    try:
        smith = get_smith()

        if not smith._attack_detector:
            raise HTTPException(
                status_code=503,
                detail="Attack detector not enabled"
            )

        patterns = smith._attack_detector.pattern_library.list_patterns()

        return {
            "total": len(patterns),
            "patterns": [
                {
                    "id": p.id,
                    "name": p.name,
                    "description": p.description,
                    "category": p.category.name if hasattr(p.category, 'name') else str(p.category),
                    "severity": p.severity,
                    "enabled": p.enabled,
                }
                for p in patterns
            ],
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error listing patterns: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/patterns/{pattern_id}/enable")
async def enable_pattern(pattern_id: str, admin_id: str = Depends(require_admin_user)) -> Dict[str, str]:
    """Enable an attack detection pattern."""
    try:
        smith = get_smith()

        if not smith._attack_detector:
            raise HTTPException(
                status_code=503,
                detail="Attack detector not enabled"
            )

        smith._attack_detector.pattern_library.enable_pattern(pattern_id)

        return {
            "status": "success",
            "pattern_id": pattern_id,
            "message": "Pattern enabled",
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error enabling pattern: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/patterns/{pattern_id}/disable")
async def disable_pattern(pattern_id: str, admin_id: str = Depends(require_admin_user)) -> Dict[str, str]:
    """Disable an attack detection pattern."""
    try:
        smith = get_smith()

        if not smith._attack_detector:
            raise HTTPException(
                status_code=503,
                detail="Attack detector not enabled"
            )

        smith._attack_detector.pattern_library.disable_pattern(pattern_id)

        return {
            "status": "success",
            "pattern_id": pattern_id,
            "message": "Pattern disabled",
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error disabling pattern: {e}")
        raise HTTPException(status_code=500, detail=str(e))
