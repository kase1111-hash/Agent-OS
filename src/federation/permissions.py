"""
Permission Negotiation

Provides permission management for federation including:
- Permission definitions
- Permission requests and grants
- Access control
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


# =============================================================================
# Permission Types
# =============================================================================


class PermissionLevel(str, Enum):
    """Permission access levels."""

    NONE = "none"
    READ = "read"
    WRITE = "write"
    EXECUTE = "execute"
    ADMIN = "admin"


class PermissionScope(str, Enum):
    """Permission scopes."""

    IDENTITY = "identity"  # Access to identity information
    MESSAGES = "messages"  # Send/receive messages
    DATA = "data"  # Access to shared data
    ACTIONS = "actions"  # Execute remote actions
    MEMORY = "memory"  # Access to agent memory
    TOOLS = "tools"  # Access to tools
    SYNC = "sync"  # Data synchronization
    ADMIN = "admin"  # Administrative operations


class PermissionStatus(str, Enum):
    """Permission request/grant status."""

    PENDING = "pending"
    APPROVED = "approved"
    DENIED = "denied"
    EXPIRED = "expired"
    REVOKED = "revoked"


# =============================================================================
# Permission Models
# =============================================================================


@dataclass
class Permission:
    """A single permission definition."""

    scope: PermissionScope
    level: PermissionLevel
    resource: str = "*"  # Specific resource or wildcard
    constraints: Dict[str, Any] = field(default_factory=dict)

    @property
    def key(self) -> str:
        """Get unique permission key."""
        return f"{self.scope.value}:{self.resource}:{self.level.value}"

    def matches(self, other: "Permission") -> bool:
        """Check if this permission matches or covers another."""
        if self.scope != other.scope:
            return False

        # Wildcard matches any resource
        if self.resource != "*" and self.resource != other.resource:
            return False

        # Check level hierarchy
        level_order = [
            PermissionLevel.NONE,
            PermissionLevel.READ,
            PermissionLevel.WRITE,
            PermissionLevel.EXECUTE,
            PermissionLevel.ADMIN,
        ]

        return level_order.index(self.level) >= level_order.index(other.level)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "scope": self.scope.value,
            "level": self.level.value,
            "resource": self.resource,
            "constraints": self.constraints,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Permission":
        """Create from dictionary."""
        return cls(
            scope=PermissionScope(data["scope"]),
            level=PermissionLevel(data["level"]),
            resource=data.get("resource", "*"),
            constraints=data.get("constraints", {}),
        )


@dataclass
class PermissionSet:
    """A set of permissions."""

    permissions: List[Permission] = field(default_factory=list)
    name: str = ""
    description: str = ""

    def add(self, permission: Permission) -> None:
        """Add a permission."""
        self.permissions.append(permission)

    def remove(self, permission: Permission) -> bool:
        """Remove a permission."""
        for i, p in enumerate(self.permissions):
            if p.key == permission.key:
                self.permissions.pop(i)
                return True
        return False

    def has_permission(self, required: Permission) -> bool:
        """Check if this set grants a required permission."""
        for p in self.permissions:
            if p.matches(required):
                return True
        return False

    def get_permissions_for_scope(
        self,
        scope: PermissionScope,
    ) -> List[Permission]:
        """Get all permissions for a scope."""
        return [p for p in self.permissions if p.scope == scope]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "permissions": [p.to_dict() for p in self.permissions],
            "name": self.name,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PermissionSet":
        """Create from dictionary."""
        return cls(
            permissions=[Permission.from_dict(p) for p in data.get("permissions", [])],
            name=data.get("name", ""),
            description=data.get("description", ""),
        )


@dataclass
class PermissionRequest:
    """A request for permissions from a peer."""

    request_id: str
    requester_id: str
    target_id: str
    permissions: PermissionSet
    reason: str = ""
    status: PermissionStatus = PermissionStatus.PENDING
    created_at: datetime = field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.expires_at:
            self.expires_at = self.created_at + timedelta(hours=24)

    @property
    def is_pending(self) -> bool:
        """Check if request is still pending."""
        return self.status == PermissionStatus.PENDING

    @property
    def is_expired(self) -> bool:
        """Check if request has expired."""
        if not self.expires_at:
            return False
        return datetime.utcnow() > self.expires_at

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "request_id": self.request_id,
            "requester_id": self.requester_id,
            "target_id": self.target_id,
            "permissions": self.permissions.to_dict(),
            "reason": self.reason,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PermissionRequest":
        """Create from dictionary."""
        return cls(
            request_id=data["request_id"],
            requester_id=data["requester_id"],
            target_id=data["target_id"],
            permissions=PermissionSet.from_dict(data["permissions"]),
            reason=data.get("reason", ""),
            status=PermissionStatus(data.get("status", "pending")),
            created_at=datetime.fromisoformat(
                data.get("created_at", datetime.utcnow().isoformat())
            ),
            expires_at=(
                datetime.fromisoformat(data["expires_at"]) if data.get("expires_at") else None
            ),
            metadata=data.get("metadata", {}),
        )


@dataclass
class PermissionGrant:
    """A grant of permissions to a peer."""

    grant_id: str
    granter_id: str
    grantee_id: str
    permissions: PermissionSet
    request_id: Optional[str] = None
    status: PermissionStatus = PermissionStatus.APPROVED
    created_at: datetime = field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None
    revoked_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.expires_at:
            self.expires_at = self.created_at + timedelta(days=30)

    @property
    def is_valid(self) -> bool:
        """Check if grant is still valid."""
        if self.status != PermissionStatus.APPROVED:
            return False
        if self.revoked_at:
            return False
        if self.expires_at and datetime.utcnow() > self.expires_at:
            return False
        return True

    @property
    def is_expired(self) -> bool:
        """Check if grant has expired."""
        if not self.expires_at:
            return False
        return datetime.utcnow() > self.expires_at

    def revoke(self) -> None:
        """Revoke this grant."""
        self.status = PermissionStatus.REVOKED
        self.revoked_at = datetime.utcnow()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "grant_id": self.grant_id,
            "granter_id": self.granter_id,
            "grantee_id": self.grantee_id,
            "permissions": self.permissions.to_dict(),
            "request_id": self.request_id,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "revoked_at": self.revoked_at.isoformat() if self.revoked_at else None,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PermissionGrant":
        """Create from dictionary."""
        return cls(
            grant_id=data["grant_id"],
            granter_id=data["granter_id"],
            grantee_id=data["grantee_id"],
            permissions=PermissionSet.from_dict(data["permissions"]),
            request_id=data.get("request_id"),
            status=PermissionStatus(data.get("status", "approved")),
            created_at=datetime.fromisoformat(
                data.get("created_at", datetime.utcnow().isoformat())
            ),
            expires_at=(
                datetime.fromisoformat(data["expires_at"]) if data.get("expires_at") else None
            ),
            revoked_at=(
                datetime.fromisoformat(data["revoked_at"]) if data.get("revoked_at") else None
            ),
            metadata=data.get("metadata", {}),
        )


# =============================================================================
# Permission Manager
# =============================================================================


class PermissionManager:
    """
    Manages permissions for federation.
    """

    def __init__(self, node_id: str):
        self.node_id = node_id

        # Permissions we've granted to others
        self._grants_given: Dict[str, List[PermissionGrant]] = {}

        # Permissions granted to us by others
        self._grants_received: Dict[str, List[PermissionGrant]] = {}

        # Pending requests (incoming)
        self._pending_requests: Dict[str, PermissionRequest] = {}

        # Our pending requests (outgoing)
        self._our_requests: Dict[str, PermissionRequest] = {}

        # Default permission sets
        self._default_permissions: PermissionSet = self._create_default_permissions()

        # Callbacks
        self._on_request_callbacks: List[Callable[[PermissionRequest], None]] = []
        self._on_grant_callbacks: List[Callable[[PermissionGrant], None]] = []

    def _create_default_permissions(self) -> PermissionSet:
        """Create default permissions for new peers."""
        return PermissionSet(
            name="default",
            description="Default permissions for new peers",
            permissions=[
                Permission(scope=PermissionScope.IDENTITY, level=PermissionLevel.READ),
                Permission(scope=PermissionScope.MESSAGES, level=PermissionLevel.WRITE),
            ],
        )

    def on_request(
        self,
        callback: Callable[[PermissionRequest], None],
    ) -> None:
        """Register callback for permission requests."""
        self._on_request_callbacks.append(callback)

    def on_grant(
        self,
        callback: Callable[[PermissionGrant], None],
    ) -> None:
        """Register callback for permission grants."""
        self._on_grant_callbacks.append(callback)

    # -------------------------------------------------------------------------
    # Request Management
    # -------------------------------------------------------------------------

    def create_request(
        self,
        target_id: str,
        permissions: List[Permission],
        reason: str = "",
    ) -> PermissionRequest:
        """
        Create a permission request to send to a peer.

        Args:
            target_id: Target node to request permissions from
            permissions: Permissions to request
            reason: Reason for request

        Returns:
            PermissionRequest
        """
        import uuid

        request = PermissionRequest(
            request_id=str(uuid.uuid4()),
            requester_id=self.node_id,
            target_id=target_id,
            permissions=PermissionSet(permissions=permissions),
            reason=reason,
        )

        self._our_requests[request.request_id] = request
        logger.info(f"Created permission request to {target_id}")

        return request

    def receive_request(
        self,
        request: PermissionRequest,
    ) -> None:
        """
        Receive a permission request from a peer.

        Args:
            request: Incoming permission request
        """
        self._pending_requests[request.request_id] = request
        logger.info(f"Received permission request from {request.requester_id}")

        # Notify callbacks
        for callback in self._on_request_callbacks:
            try:
                callback(request)
            except Exception as e:
                logger.error(f"Request callback error: {e}")

    def approve_request(
        self,
        request_id: str,
        approved_permissions: Optional[List[Permission]] = None,
        expires_in_days: int = 30,
    ) -> Optional[PermissionGrant]:
        """
        Approve a permission request.

        Args:
            request_id: Request to approve
            approved_permissions: Permissions to actually grant (may be subset)
            expires_in_days: How long the grant is valid

        Returns:
            PermissionGrant or None
        """
        request = self._pending_requests.get(request_id)
        if not request:
            logger.warning(f"Request not found: {request_id}")
            return None

        if not request.is_pending:
            logger.warning(f"Request not pending: {request_id}")
            return None

        import uuid

        # Use requested permissions if none specified
        if approved_permissions is None:
            approved_permissions = request.permissions.permissions

        grant = PermissionGrant(
            grant_id=str(uuid.uuid4()),
            granter_id=self.node_id,
            grantee_id=request.requester_id,
            permissions=PermissionSet(permissions=approved_permissions),
            request_id=request_id,
            expires_at=datetime.utcnow() + timedelta(days=expires_in_days),
        )

        # Update request status
        request.status = PermissionStatus.APPROVED

        # Store grant
        if request.requester_id not in self._grants_given:
            self._grants_given[request.requester_id] = []
        self._grants_given[request.requester_id].append(grant)

        # Remove from pending
        del self._pending_requests[request_id]

        logger.info(f"Approved permission request from {request.requester_id}")

        # Notify callbacks
        for callback in self._on_grant_callbacks:
            try:
                callback(grant)
            except Exception as e:
                logger.error(f"Grant callback error: {e}")

        return grant

    def deny_request(
        self,
        request_id: str,
        reason: str = "",
    ) -> bool:
        """
        Deny a permission request.

        Args:
            request_id: Request to deny
            reason: Reason for denial

        Returns:
            True if denied successfully
        """
        request = self._pending_requests.get(request_id)
        if not request:
            return False

        request.status = PermissionStatus.DENIED
        request.metadata["denial_reason"] = reason

        del self._pending_requests[request_id]

        logger.info(f"Denied permission request from {request.requester_id}")
        return True

    def receive_grant(
        self,
        grant: PermissionGrant,
    ) -> None:
        """
        Receive a permission grant from a peer.

        Args:
            grant: Incoming permission grant
        """
        if grant.granter_id not in self._grants_received:
            self._grants_received[grant.granter_id] = []
        self._grants_received[grant.granter_id].append(grant)

        # Update our request status if applicable
        if grant.request_id and grant.request_id in self._our_requests:
            self._our_requests[grant.request_id].status = PermissionStatus.APPROVED

        logger.info(f"Received permission grant from {grant.granter_id}")

    # -------------------------------------------------------------------------
    # Permission Checking
    # -------------------------------------------------------------------------

    def check_permission(
        self,
        peer_id: str,
        required: Permission,
    ) -> bool:
        """
        Check if a peer has a required permission.

        Args:
            peer_id: Peer to check
            required: Required permission

        Returns:
            True if peer has permission
        """
        # Get valid grants for this peer
        grants = self._grants_given.get(peer_id, [])

        for grant in grants:
            if not grant.is_valid:
                continue

            if grant.permissions.has_permission(required):
                return True

        return False

    def check_own_permission(
        self,
        peer_id: str,
        required: Permission,
    ) -> bool:
        """
        Check if we have a required permission from a peer.

        Args:
            peer_id: Peer who granted permission
            required: Required permission

        Returns:
            True if we have permission
        """
        grants = self._grants_received.get(peer_id, [])

        for grant in grants:
            if not grant.is_valid:
                continue

            if grant.permissions.has_permission(required):
                return True

        return False

    def get_permissions_for_peer(
        self,
        peer_id: str,
    ) -> PermissionSet:
        """
        Get all valid permissions for a peer.

        Args:
            peer_id: Peer to get permissions for

        Returns:
            PermissionSet with all valid permissions
        """
        all_permissions = []
        grants = self._grants_given.get(peer_id, [])

        for grant in grants:
            if grant.is_valid:
                all_permissions.extend(grant.permissions.permissions)

        return PermissionSet(permissions=all_permissions)

    def get_own_permissions_from_peer(
        self,
        peer_id: str,
    ) -> PermissionSet:
        """
        Get all permissions we have from a peer.

        Args:
            peer_id: Peer who granted permissions

        Returns:
            PermissionSet with all valid permissions
        """
        all_permissions = []
        grants = self._grants_received.get(peer_id, [])

        for grant in grants:
            if grant.is_valid:
                all_permissions.extend(grant.permissions.permissions)

        return PermissionSet(permissions=all_permissions)

    # -------------------------------------------------------------------------
    # Grant Management
    # -------------------------------------------------------------------------

    def revoke_grant(
        self,
        grant_id: str,
    ) -> bool:
        """
        Revoke a permission grant.

        Args:
            grant_id: Grant to revoke

        Returns:
            True if revoked successfully
        """
        for peer_id, grants in self._grants_given.items():
            for grant in grants:
                if grant.grant_id == grant_id:
                    grant.revoke()
                    logger.info(f"Revoked grant {grant_id} for peer {peer_id}")
                    return True

        return False

    def revoke_all_grants(
        self,
        peer_id: str,
    ) -> int:
        """
        Revoke all grants for a peer.

        Args:
            peer_id: Peer to revoke grants for

        Returns:
            Number of grants revoked
        """
        grants = self._grants_given.get(peer_id, [])
        count = 0

        for grant in grants:
            if grant.is_valid:
                grant.revoke()
                count += 1

        logger.info(f"Revoked {count} grants for peer {peer_id}")
        return count

    def grant_default_permissions(
        self,
        peer_id: str,
    ) -> PermissionGrant:
        """
        Grant default permissions to a new peer.

        Args:
            peer_id: Peer to grant to

        Returns:
            PermissionGrant
        """
        import uuid

        grant = PermissionGrant(
            grant_id=str(uuid.uuid4()),
            granter_id=self.node_id,
            grantee_id=peer_id,
            permissions=self._default_permissions,
        )

        if peer_id not in self._grants_given:
            self._grants_given[peer_id] = []
        self._grants_given[peer_id].append(grant)

        logger.info(f"Granted default permissions to peer {peer_id}")
        return grant

    # -------------------------------------------------------------------------
    # Queries
    # -------------------------------------------------------------------------

    def list_pending_requests(self) -> List[PermissionRequest]:
        """List all pending permission requests."""
        return [r for r in self._pending_requests.values() if r.is_pending and not r.is_expired]

    def list_grants_given(
        self,
        peer_id: Optional[str] = None,
        valid_only: bool = True,
    ) -> List[PermissionGrant]:
        """List grants we've given."""
        if peer_id:
            grants = self._grants_given.get(peer_id, [])
        else:
            grants = [g for grants in self._grants_given.values() for g in grants]

        if valid_only:
            grants = [g for g in grants if g.is_valid]

        return grants

    def list_grants_received(
        self,
        peer_id: Optional[str] = None,
        valid_only: bool = True,
    ) -> List[PermissionGrant]:
        """List grants we've received."""
        if peer_id:
            grants = self._grants_received.get(peer_id, [])
        else:
            grants = [g for grants in self._grants_received.values() for g in grants]

        if valid_only:
            grants = [g for g in grants if g.is_valid]

        return grants

    def cleanup_expired(self) -> Tuple[int, int]:
        """
        Clean up expired requests and grants.

        Returns:
            Tuple of (requests_cleaned, grants_cleaned)
        """
        requests_cleaned = 0
        grants_cleaned = 0

        # Clean expired requests
        expired_requests = [rid for rid, r in self._pending_requests.items() if r.is_expired]
        for rid in expired_requests:
            del self._pending_requests[rid]
            requests_cleaned += 1

        # Mark expired grants
        for grants in self._grants_given.values():
            for grant in grants:
                if grant.is_valid and grant.is_expired:
                    grant.status = PermissionStatus.EXPIRED
                    grants_cleaned += 1

        for grants in self._grants_received.values():
            for grant in grants:
                if grant.is_valid and grant.is_expired:
                    grant.status = PermissionStatus.EXPIRED
                    grants_cleaned += 1

        if requests_cleaned or grants_cleaned:
            logger.info(f"Cleaned up {requests_cleaned} requests, {grants_cleaned} grants")

        return requests_cleaned, grants_cleaned


# =============================================================================
# Factory Function
# =============================================================================


def create_permission_manager(node_id: str) -> PermissionManager:
    """
    Create a permission manager.

    Args:
        node_id: Local node ID

    Returns:
        PermissionManager instance
    """
    return PermissionManager(node_id)
