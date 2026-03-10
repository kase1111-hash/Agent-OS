"""
Human Approval Endpoints for Agent OS

Provides a human-in-the-loop approval mechanism for high-privilege
agent actions that require explicit human authorization.

This addresses Finding #14 (Escalation callbacks lack human approval handler)
from the Agentic Security Audit v3.0.
"""

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from src.web.auth_helpers import require_admin_user

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/approvals", tags=["approvals"])


class ApprovalStatus(str, Enum):
    """Status of an approval request."""

    PENDING = "pending"
    APPROVED = "approved"
    DENIED = "denied"
    EXPIRED = "expired"


@dataclass
class ApprovalRequest:
    """A pending human approval request."""

    id: str
    action: str
    description: str
    source_agent: str
    severity: str
    context: Dict[str, Any] = field(default_factory=dict)
    status: ApprovalStatus = ApprovalStatus.PENDING
    created_at: datetime = field(default_factory=datetime.utcnow)
    resolved_at: Optional[datetime] = None
    resolved_by: Optional[str] = None


# In-memory approval queue (for MVP; would be backed by DB in production)
_approval_queue: Dict[str, ApprovalRequest] = {}


def submit_for_approval(
    action: str,
    description: str,
    source_agent: str,
    severity: str = "medium",
    context: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Submit an action for human approval.

    Called by the enforcement/escalation system when an action
    requires human authorization.

    Returns:
        The approval request ID.
    """
    request_id = str(uuid.uuid4())
    request = ApprovalRequest(
        id=request_id,
        action=action,
        description=description,
        source_agent=source_agent,
        severity=severity,
        context=context or {},
    )
    _approval_queue[request_id] = request
    logger.info(
        "Approval requested: id=%s action=%s agent=%s severity=%s",
        request_id,
        action,
        source_agent,
        severity,
    )
    return request_id


def check_approval_status(request_id: str) -> Optional[ApprovalStatus]:
    """Check if an approval request has been resolved."""
    request = _approval_queue.get(request_id)
    if not request:
        return None
    return request.status


# Pydantic models for API
class ApprovalSummary(BaseModel):
    id: str
    action: str
    description: str
    source_agent: str
    severity: str
    status: str
    created_at: str


class ApprovalDecision(BaseModel):
    reason: Optional[str] = None


@router.get("", response_model=List[ApprovalSummary])
async def list_pending_approvals(
    status: Optional[str] = "pending",
    admin_id: str = Depends(require_admin_user),
) -> List[ApprovalSummary]:
    """List approval requests. Defaults to pending only."""
    results = []
    for req in _approval_queue.values():
        if status and req.status.value != status:
            continue
        results.append(
            ApprovalSummary(
                id=req.id,
                action=req.action,
                description=req.description,
                source_agent=req.source_agent,
                severity=req.severity,
                status=req.status.value,
                created_at=req.created_at.isoformat(),
            )
        )
    return sorted(results, key=lambda r: r.created_at, reverse=True)


@router.post("/{request_id}/approve")
async def approve_request(
    request_id: str,
    decision: ApprovalDecision,
    admin_id: str = Depends(require_admin_user),
):
    """Approve a pending action."""
    req = _approval_queue.get(request_id)
    if not req:
        raise HTTPException(status_code=404, detail="Approval request not found")
    if req.status != ApprovalStatus.PENDING:
        raise HTTPException(status_code=409, detail=f"Request already {req.status.value}")

    req.status = ApprovalStatus.APPROVED
    req.resolved_at = datetime.utcnow()
    req.resolved_by = admin_id
    logger.info("Approval GRANTED: id=%s by=%s reason=%s", request_id, admin_id, decision.reason)
    return {"status": "approved", "id": request_id}


@router.post("/{request_id}/deny")
async def deny_request(
    request_id: str,
    decision: ApprovalDecision,
    admin_id: str = Depends(require_admin_user),
):
    """Deny a pending action."""
    req = _approval_queue.get(request_id)
    if not req:
        raise HTTPException(status_code=404, detail="Approval request not found")
    if req.status != ApprovalStatus.PENDING:
        raise HTTPException(status_code=409, detail=f"Request already {req.status.value}")

    req.status = ApprovalStatus.DENIED
    req.resolved_at = datetime.utcnow()
    req.resolved_by = admin_id
    logger.info("Approval DENIED: id=%s by=%s reason=%s", request_id, admin_id, decision.reason)
    return {"status": "denied", "id": request_id}
