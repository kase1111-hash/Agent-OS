"""
Tool Permission Layer

Controls which users and agents can use which tools.
Implements role-based access control with fine-grained permissions.
"""

import json
import logging
import secrets
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from .interface import ToolCategory, ToolRiskLevel

logger = logging.getLogger(__name__)


class PermissionLevel(Enum):
    """Permission levels for tool access."""

    NONE = auto()  # No access
    INVOKE = auto()  # Can invoke (with approval)
    USE = auto()  # Can use freely (approved tools)
    MANAGE = auto()  # Can enable/disable
    ADMIN = auto()  # Full control including registration


class GrantType(Enum):
    """Type of permission grant."""

    PERMANENT = auto()  # No expiration
    TEMPORARY = auto()  # Time-limited
    SESSION = auto()  # Valid for current session only
    ONE_TIME = auto()  # Valid for single use


@dataclass
class PermissionGrant:
    """A permission grant for a user/agent."""

    grant_id: str
    principal_id: str  # User or agent ID
    principal_type: str  # "user" or "agent"
    tool_id: Optional[str] = None  # Specific tool (None = all tools)
    tool_name: Optional[str] = None  # Specific tool by name
    category: Optional[ToolCategory] = None  # All tools in category
    permission: PermissionLevel = PermissionLevel.USE
    grant_type: GrantType = GrantType.PERMANENT
    granted_at: datetime = field(default_factory=datetime.now)
    granted_by: Optional[str] = None
    expires_at: Optional[datetime] = None
    uses_remaining: Optional[int] = None
    conditions: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def generate_id() -> str:
        return f"GRANT-{secrets.token_hex(8)}"

    @property
    def is_valid(self) -> bool:
        """Check if grant is still valid."""
        if self.grant_type == GrantType.TEMPORARY:
            if self.expires_at and datetime.now() > self.expires_at:
                return False

        if self.grant_type == GrantType.ONE_TIME:
            if self.uses_remaining is not None and self.uses_remaining <= 0:
                return False

        return True

    def use(self) -> bool:
        """Record a use of this grant. Returns False if grant exhausted."""
        if not self.is_valid:
            return False

        if self.uses_remaining is not None:
            self.uses_remaining -= 1
            if self.uses_remaining <= 0:
                return True  # This was the last use

        return True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "grant_id": self.grant_id,
            "principal_id": self.principal_id,
            "principal_type": self.principal_type,
            "tool_id": self.tool_id,
            "tool_name": self.tool_name,
            "category": self.category.value if self.category else None,
            "permission": self.permission.name,
            "grant_type": self.grant_type.name,
            "granted_at": self.granted_at.isoformat(),
            "granted_by": self.granted_by,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "uses_remaining": self.uses_remaining,
            "conditions": self.conditions,
            "metadata": self.metadata,
        }


@dataclass
class PermissionDenial:
    """Explicit denial of permission."""

    denial_id: str
    principal_id: str
    principal_type: str
    tool_id: Optional[str] = None
    tool_name: Optional[str] = None
    category: Optional[ToolCategory] = None
    reason: str = ""
    denied_at: datetime = field(default_factory=datetime.now)
    denied_by: Optional[str] = None
    expires_at: Optional[datetime] = None

    @staticmethod
    def generate_id() -> str:
        return f"DENY-{secrets.token_hex(8)}"

    @property
    def is_active(self) -> bool:
        if self.expires_at and datetime.now() > self.expires_at:
            return False
        return True


@dataclass
class RiskLimitPolicy:
    """Policy limiting tool use by risk level."""

    max_risk_level: ToolRiskLevel = ToolRiskLevel.MEDIUM
    require_confirmation_above: ToolRiskLevel = ToolRiskLevel.MEDIUM
    require_human_approval_above: ToolRiskLevel = ToolRiskLevel.HIGH


@dataclass
class PermissionCheckResult:
    """Result of a permission check."""

    allowed: bool
    permission_level: PermissionLevel = PermissionLevel.NONE
    requires_confirmation: bool = False
    requires_human_approval: bool = False
    grant: Optional[PermissionGrant] = None
    denial: Optional[PermissionDenial] = None
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "allowed": self.allowed,
            "permission_level": self.permission_level.name,
            "requires_confirmation": self.requires_confirmation,
            "requires_human_approval": self.requires_human_approval,
            "grant_id": self.grant.grant_id if self.grant else None,
            "denial_id": self.denial.denial_id if self.denial else None,
            "reason": self.reason,
        }


class PermissionManager:
    """
    Manages tool permissions for users and agents.

    Features:
    - Role-based access control
    - Fine-grained tool permissions
    - Temporary and one-time grants
    - Explicit denials
    - Risk-based policies
    """

    def __init__(
        self,
        storage_path: Optional[Path] = None,
        default_policy: Optional[RiskLimitPolicy] = None,
    ):
        """
        Initialize permission manager.

        Args:
            storage_path: Path for persistent storage
            default_policy: Default risk policy for all principals
        """
        self._grants: Dict[str, PermissionGrant] = {}
        self._denials: Dict[str, PermissionDenial] = {}
        self._principal_grants: Dict[str, Set[str]] = {}  # principal -> grant IDs
        self._principal_denials: Dict[str, Set[str]] = {}  # principal -> denial IDs
        self._policies: Dict[str, RiskLimitPolicy] = {}  # principal -> policy
        self._default_policy = default_policy or RiskLimitPolicy()
        self._storage_path = storage_path
        self._lock = threading.RLock()

        if storage_path:
            storage_path.mkdir(parents=True, exist_ok=True)
            self._load_state()

    def grant(
        self,
        principal_id: str,
        principal_type: str = "user",
        tool_id: Optional[str] = None,
        tool_name: Optional[str] = None,
        category: Optional[ToolCategory] = None,
        permission: PermissionLevel = PermissionLevel.USE,
        grant_type: GrantType = GrantType.PERMANENT,
        duration_hours: Optional[int] = None,
        uses: Optional[int] = None,
        granted_by: Optional[str] = None,
        conditions: Optional[Dict[str, Any]] = None,
    ) -> PermissionGrant:
        """
        Grant permission to a principal.

        Args:
            principal_id: User or agent ID
            principal_type: "user" or "agent"
            tool_id: Specific tool ID (optional)
            tool_name: Specific tool name (optional)
            category: Tool category (optional)
            permission: Permission level
            grant_type: Type of grant
            duration_hours: Duration for temporary grants
            uses: Number of uses for one-time grants
            granted_by: ID of granter
            conditions: Additional conditions

        Returns:
            PermissionGrant
        """
        with self._lock:
            expires_at = None
            if grant_type == GrantType.TEMPORARY and duration_hours:
                expires_at = datetime.now() + timedelta(hours=duration_hours)

            grant = PermissionGrant(
                grant_id=PermissionGrant.generate_id(),
                principal_id=principal_id,
                principal_type=principal_type,
                tool_id=tool_id,
                tool_name=tool_name,
                category=category,
                permission=permission,
                grant_type=grant_type,
                granted_by=granted_by,
                expires_at=expires_at,
                uses_remaining=uses if grant_type == GrantType.ONE_TIME else None,
                conditions=conditions or {},
            )

            self._grants[grant.grant_id] = grant

            if principal_id not in self._principal_grants:
                self._principal_grants[principal_id] = set()
            self._principal_grants[principal_id].add(grant.grant_id)

            self._save_state()

            logger.info(
                f"Granted {permission.name} to {principal_type}:{principal_id} "
                f"(tool={tool_name or tool_id or 'all'})"
            )

            return grant

    def deny(
        self,
        principal_id: str,
        principal_type: str = "user",
        tool_id: Optional[str] = None,
        tool_name: Optional[str] = None,
        category: Optional[ToolCategory] = None,
        reason: str = "",
        denied_by: Optional[str] = None,
        duration_hours: Optional[int] = None,
    ) -> PermissionDenial:
        """
        Explicitly deny permission to a principal.

        Args:
            principal_id: User or agent ID
            principal_type: "user" or "agent"
            tool_id: Specific tool ID (optional)
            tool_name: Specific tool name (optional)
            category: Tool category (optional)
            reason: Reason for denial
            denied_by: ID of denier
            duration_hours: Duration for temporary denial

        Returns:
            PermissionDenial
        """
        with self._lock:
            expires_at = None
            if duration_hours:
                expires_at = datetime.now() + timedelta(hours=duration_hours)

            denial = PermissionDenial(
                denial_id=PermissionDenial.generate_id(),
                principal_id=principal_id,
                principal_type=principal_type,
                tool_id=tool_id,
                tool_name=tool_name,
                category=category,
                reason=reason,
                denied_by=denied_by,
                expires_at=expires_at,
            )

            self._denials[denial.denial_id] = denial

            if principal_id not in self._principal_denials:
                self._principal_denials[principal_id] = set()
            self._principal_denials[principal_id].add(denial.denial_id)

            self._save_state()

            logger.info(
                f"Denied access to {principal_type}:{principal_id} "
                f"(tool={tool_name or tool_id or 'all'}): {reason}"
            )

            return denial

    def revoke_grant(self, grant_id: str) -> bool:
        """Revoke a permission grant."""
        with self._lock:
            grant = self._grants.get(grant_id)
            if not grant:
                return False

            del self._grants[grant_id]
            if grant.principal_id in self._principal_grants:
                self._principal_grants[grant.principal_id].discard(grant_id)

            self._save_state()
            logger.info(f"Revoked grant: {grant_id}")
            return True

    def revoke_denial(self, denial_id: str) -> bool:
        """Revoke a permission denial."""
        with self._lock:
            denial = self._denials.get(denial_id)
            if not denial:
                return False

            del self._denials[denial_id]
            if denial.principal_id in self._principal_denials:
                self._principal_denials[denial.principal_id].discard(denial_id)

            self._save_state()
            logger.info(f"Revoked denial: {denial_id}")
            return True

    def set_policy(
        self,
        principal_id: str,
        policy: RiskLimitPolicy,
    ) -> None:
        """Set risk limit policy for a principal."""
        with self._lock:
            self._policies[principal_id] = policy
            self._save_state()

    def get_policy(self, principal_id: str) -> RiskLimitPolicy:
        """Get risk limit policy for a principal."""
        with self._lock:
            return self._policies.get(principal_id, self._default_policy)

    def check(
        self,
        principal_id: str,
        principal_type: str,
        tool_id: str,
        tool_name: str,
        tool_category: ToolCategory,
        tool_risk: ToolRiskLevel,
    ) -> PermissionCheckResult:
        """
        Check if a principal has permission to use a tool.

        Args:
            principal_id: User or agent ID
            principal_type: "user" or "agent"
            tool_id: Tool ID
            tool_name: Tool name
            tool_category: Tool category
            tool_risk: Tool risk level

        Returns:
            PermissionCheckResult
        """
        with self._lock:
            # Check for explicit denials first (denials take precedence)
            denial = self._find_matching_denial(principal_id, tool_id, tool_name, tool_category)
            if denial:
                return PermissionCheckResult(
                    allowed=False,
                    denial=denial,
                    reason=denial.reason or "Access explicitly denied",
                )

            # Check risk policy
            policy = self.get_policy(principal_id)
            if tool_risk.value > policy.max_risk_level.value:
                return PermissionCheckResult(
                    allowed=False,
                    reason=f"Tool risk level ({tool_risk.name}) exceeds maximum allowed ({policy.max_risk_level.name})",
                )

            # Find matching grant
            grant = self._find_matching_grant(principal_id, tool_id, tool_name, tool_category)

            if not grant:
                return PermissionCheckResult(
                    allowed=False,
                    reason="No permission grant found",
                )

            if not grant.is_valid:
                return PermissionCheckResult(
                    allowed=False,
                    grant=grant,
                    reason="Permission grant has expired",
                )

            # Check if confirmation or approval needed
            requires_confirmation = tool_risk.value >= policy.require_confirmation_above.value
            requires_human_approval = tool_risk.value >= policy.require_human_approval_above.value

            return PermissionCheckResult(
                allowed=True,
                permission_level=grant.permission,
                requires_confirmation=requires_confirmation,
                requires_human_approval=requires_human_approval,
                grant=grant,
            )

    def record_use(self, grant_id: str) -> bool:
        """Record use of a grant (for one-time grants)."""
        with self._lock:
            grant = self._grants.get(grant_id)
            if not grant:
                return False

            result = grant.use()
            self._save_state()
            return result

    def get_grants_for_principal(self, principal_id: str) -> List[PermissionGrant]:
        """Get all grants for a principal."""
        with self._lock:
            grant_ids = self._principal_grants.get(principal_id, set())
            return [self._grants[gid] for gid in grant_ids if gid in self._grants]

    def get_denials_for_principal(self, principal_id: str) -> List[PermissionDenial]:
        """Get all denials for a principal."""
        with self._lock:
            denial_ids = self._principal_denials.get(principal_id, set())
            return [self._denials[did] for did in denial_ids if did in self._denials]

    def _find_matching_grant(
        self,
        principal_id: str,
        tool_id: str,
        tool_name: str,
        tool_category: ToolCategory,
    ) -> Optional[PermissionGrant]:
        """Find the best matching grant for a tool."""
        grant_ids = self._principal_grants.get(principal_id, set())

        best_grant = None
        best_specificity = -1

        for grant_id in grant_ids:
            grant = self._grants.get(grant_id)
            if not grant or not grant.is_valid:
                continue

            specificity = 0

            # Check specific tool match
            if grant.tool_id:
                if grant.tool_id == tool_id:
                    specificity = 3  # Most specific
                else:
                    continue  # Doesn't match

            elif grant.tool_name:
                if grant.tool_name == tool_name:
                    specificity = 2
                else:
                    continue

            elif grant.category:
                if grant.category == tool_category:
                    specificity = 1
                else:
                    continue

            # No specific filter = matches all tools
            # specificity stays 0

            if specificity > best_specificity:
                best_specificity = specificity
                best_grant = grant

        return best_grant

    def _find_matching_denial(
        self,
        principal_id: str,
        tool_id: str,
        tool_name: str,
        tool_category: ToolCategory,
    ) -> Optional[PermissionDenial]:
        """Find a matching denial for a tool."""
        denial_ids = self._principal_denials.get(principal_id, set())

        for denial_id in denial_ids:
            denial = self._denials.get(denial_id)
            if not denial or not denial.is_active:
                continue

            # Check if denial matches this tool
            if denial.tool_id:
                if denial.tool_id == tool_id:
                    return denial
            elif denial.tool_name:
                if denial.tool_name == tool_name:
                    return denial
            elif denial.category:
                if denial.category == tool_category:
                    return denial
            else:
                # No specific filter = denies all tools
                return denial

        return None

    def _save_state(self) -> None:
        """Save state to disk."""
        if not self._storage_path:
            return

        try:
            state_file = self._storage_path / "permissions_state.json"
            state = {
                "grants": {gid: g.to_dict() for gid, g in self._grants.items()},
                "denials": {
                    did: {
                        "denial_id": d.denial_id,
                        "principal_id": d.principal_id,
                        "principal_type": d.principal_type,
                        "tool_id": d.tool_id,
                        "tool_name": d.tool_name,
                        "category": d.category.value if d.category else None,
                        "reason": d.reason,
                        "denied_at": d.denied_at.isoformat(),
                        "denied_by": d.denied_by,
                        "expires_at": d.expires_at.isoformat() if d.expires_at else None,
                    }
                    for did, d in self._denials.items()
                },
                "saved_at": datetime.now().isoformat(),
            }

            with open(state_file, "w") as f:
                json.dump(state, f, indent=2)

        except Exception as e:
            logger.error(f"Failed to save permissions state: {e}")

    def _load_state(self) -> None:
        """Load state from disk."""
        if not self._storage_path:
            return

        state_file = self._storage_path / "permissions_state.json"
        if not state_file.exists():
            return

        try:
            with open(state_file, "r") as f:
                state = json.load(f)

            # Rebuild indices
            for grant_data in state.get("grants", {}).values():
                principal_id = grant_data["principal_id"]
                grant_id = grant_data["grant_id"]
                if principal_id not in self._principal_grants:
                    self._principal_grants[principal_id] = set()
                self._principal_grants[principal_id].add(grant_id)

            for denial_data in state.get("denials", {}).values():
                principal_id = denial_data["principal_id"]
                denial_id = denial_data["denial_id"]
                if principal_id not in self._principal_denials:
                    self._principal_denials[principal_id] = set()
                self._principal_denials[principal_id].add(denial_id)

            logger.info(
                f"Loaded permissions state: {len(state.get('grants', {}))} grants, "
                f"{len(state.get('denials', {}))} denials"
            )

        except Exception as e:
            logger.error(f"Failed to load permissions state: {e}")


def create_permission_manager(
    storage_path: Optional[Path] = None,
    default_policy: Optional[RiskLimitPolicy] = None,
) -> PermissionManager:
    """
    Create a permission manager.

    Args:
        storage_path: Path for persistent storage
        default_policy: Default risk policy

    Returns:
        PermissionManager instance
    """
    return PermissionManager(
        storage_path=storage_path,
        default_policy=default_policy,
    )
