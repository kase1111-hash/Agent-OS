"""
Tool Registry System

Manages tool registration, discovery, and lifecycle.
All tools must be registered before use.
"""

import hashlib
import hmac as hmac_module
import json
import logging
import secrets
import threading
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Type

from .interface import (
    InvocationResult,
    ToolCategory,
    ToolInterface,
    ToolParameter,
    ToolResult,
    ToolRiskLevel,
    ToolSchema,
    ToolStatus,
)

logger = logging.getLogger(__name__)


@dataclass
class ToolRegistration:
    """A registered tool with metadata."""

    tool_id: str
    tool: ToolInterface
    schema: ToolSchema
    status: ToolStatus = ToolStatus.PENDING
    registered_at: datetime = field(default_factory=datetime.now)
    registered_by: Optional[str] = None
    approved_at: Optional[datetime] = None
    approved_by: Optional[str] = None
    disabled_at: Optional[datetime] = None
    disabled_reason: Optional[str] = None
    invocation_count: int = 0
    manifest: Optional[Any] = None  # V6-4: Optional ToolManifest
    last_invoked: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_available(self) -> bool:
        """Check if tool is available for use."""
        return self.status == ToolStatus.APPROVED and self.tool.enabled

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "tool_id": self.tool_id,
            "name": self.tool.name,
            "schema": self.schema.to_dict(),
            "status": self.status.name,
            "registered_at": self.registered_at.isoformat(),
            "registered_by": self.registered_by,
            "approved_at": self.approved_at.isoformat() if self.approved_at else None,
            "approved_by": self.approved_by,
            "disabled_at": self.disabled_at.isoformat() if self.disabled_at else None,
            "disabled_reason": self.disabled_reason,
            "invocation_count": self.invocation_count,
            "last_invoked": self.last_invoked.isoformat() if self.last_invoked else None,
            "metadata": self.metadata,
        }


@dataclass
class ToolQuery:
    """Query parameters for searching tools."""

    name_pattern: Optional[str] = None
    categories: Optional[Set[ToolCategory]] = None
    risk_levels: Optional[Set[ToolRiskLevel]] = None
    statuses: Optional[Set[ToolStatus]] = None
    tags: Optional[Set[str]] = None
    available_only: bool = False

    def matches(self, registration: ToolRegistration) -> bool:
        """Check if registration matches query."""
        schema = registration.schema

        if self.name_pattern:
            if self.name_pattern.lower() not in registration.tool.name.lower():
                return False

        if self.categories and schema.category not in self.categories:
            return False

        if self.risk_levels and schema.risk_level not in self.risk_levels:
            return False

        if self.statuses and registration.status not in self.statuses:
            return False

        if self.tags:
            if not self.tags.intersection(set(schema.tags)):
                return False

        if self.available_only and not registration.is_available:
            return False

        return True


class ToolRegistry:
    """
    Central registry for all tools.

    Manages:
    - Tool registration and approval
    - Tool discovery and lookup
    - Tool lifecycle (enable/disable/revoke)
    - Persistence of registry state
    """

    def __init__(
        self,
        storage_path: Optional[Path] = None,
        auto_approve_low_risk: bool = False,
    ):
        """
        Initialize registry.

        Args:
            storage_path: Path for persistent storage
            auto_approve_low_risk: Auto-approve LOW risk tools
        """
        self._tools: Dict[str, ToolRegistration] = {}
        self._by_name: Dict[str, str] = {}  # name -> tool_id
        self._lock = threading.RLock()
        self._storage_path = storage_path
        self._auto_approve_low_risk = auto_approve_low_risk
        self._event_handlers: Dict[str, List[Callable]] = {
            "registered": [],
            "approved": [],
            "disabled": [],
            "revoked": [],
            "invoked": [],
        }

        if storage_path:
            storage_path.mkdir(parents=True, exist_ok=True)
            self._load_state()

    def register(
        self,
        tool: ToolInterface,
        registered_by: Optional[str] = None,
        auto_approve: bool = False,
        manifest: Optional[Any] = None,
    ) -> ToolRegistration:
        """
        Register a new tool.

        Args:
            tool: Tool instance to register
            registered_by: ID of registering entity
            auto_approve: Force auto-approval
            manifest: V6-4 optional ToolManifest declaring tool permissions

        Returns:
            ToolRegistration
        """
        # V6-4: Verify manifest signature if provided
        if manifest is not None:
            if not manifest.verify_signature():
                raise ValueError(
                    f"Manifest signature verification failed for tool '{tool.name}'"
                )
            logger.info(
                f"Manifest verified for tool '{tool.name}' "
                f"(author={manifest.author}, version={manifest.version})"
            )

        with self._lock:
            # Check for duplicate name
            if tool.name in self._by_name:
                existing_id = self._by_name[tool.name]
                existing = self._tools.get(existing_id)
                if existing and existing.status not in (
                    ToolStatus.REVOKED,
                    ToolStatus.DEPRECATED,
                ):
                    raise ValueError(f"Tool '{tool.name}' is already registered")

            # Generate tool ID
            tool_id = f"TOOL-{secrets.token_hex(8)}"

            # Get schema
            schema = tool.get_schema()

            # Determine initial status
            status = ToolStatus.PENDING
            approved_at = None
            approved_by = None

            if auto_approve or (
                self._auto_approve_low_risk and schema.risk_level == ToolRiskLevel.LOW
            ):
                status = ToolStatus.APPROVED
                approved_at = datetime.now()
                approved_by = "auto"

            registration = ToolRegistration(
                tool_id=tool_id,
                tool=tool,
                schema=schema,
                status=status,
                registered_by=registered_by,
                approved_at=approved_at,
                approved_by=approved_by,
                manifest=manifest,
            )

            self._tools[tool_id] = registration
            self._by_name[tool.name] = tool_id

            self._emit_event("registered", registration)
            self._save_state()

            logger.info(
                f"Registered tool: {tool.name} (id={tool_id}, "
                f"risk={schema.risk_level.name}, status={status.name})"
            )

            return registration

    def approve(
        self,
        tool_id: str,
        approved_by: str,
    ) -> bool:
        """
        Approve a pending tool.

        Args:
            tool_id: Tool ID
            approved_by: ID of approver

        Returns:
            True if approved
        """
        with self._lock:
            registration = self._tools.get(tool_id)
            if not registration:
                logger.warning(f"Tool not found: {tool_id}")
                return False

            if registration.status != ToolStatus.PENDING:
                logger.warning(f"Tool {tool_id} is not pending (status={registration.status.name})")
                return False

            registration.status = ToolStatus.APPROVED
            registration.approved_at = datetime.now()
            registration.approved_by = approved_by

            self._emit_event("approved", registration)
            self._save_state()

            logger.info(f"Approved tool: {registration.tool.name} by {approved_by}")
            return True

    def disable(
        self,
        tool_id: str,
        reason: str,
        _disabled_by: Optional[str] = None,
    ) -> bool:
        """
        Disable a tool temporarily.

        Args:
            tool_id: Tool ID
            reason: Reason for disabling
            disabled_by: ID of disabler

        Returns:
            True if disabled
        """
        with self._lock:
            registration = self._tools.get(tool_id)
            if not registration:
                return False

            registration.status = ToolStatus.DISABLED
            registration.disabled_at = datetime.now()
            registration.disabled_reason = reason
            registration.tool.disable()

            self._emit_event("disabled", registration)
            self._save_state()

            logger.info(f"Disabled tool: {registration.tool.name} - {reason}")
            return True

    def enable(self, tool_id: str) -> bool:
        """
        Re-enable a disabled tool.

        Args:
            tool_id: Tool ID

        Returns:
            True if enabled
        """
        with self._lock:
            registration = self._tools.get(tool_id)
            if not registration:
                return False

            if registration.status != ToolStatus.DISABLED:
                return False

            registration.status = ToolStatus.APPROVED
            registration.disabled_at = None
            registration.disabled_reason = None
            registration.tool.enable()

            self._save_state()
            logger.info(f"Re-enabled tool: {registration.tool.name}")
            return True

    def revoke(
        self,
        tool_id: str,
        reason: str,
        revoked_by: Optional[str] = None,
    ) -> bool:
        """
        Permanently revoke a tool.

        Args:
            tool_id: Tool ID
            reason: Reason for revocation
            revoked_by: ID of revoker

        Returns:
            True if revoked
        """
        with self._lock:
            registration = self._tools.get(tool_id)
            if not registration:
                return False

            registration.status = ToolStatus.REVOKED
            registration.disabled_at = datetime.now()
            registration.disabled_reason = reason
            registration.tool.disable()

            self._emit_event("revoked", registration)
            self._save_state()

            logger.info(f"Revoked tool: {registration.tool.name} - {reason}")
            return True

    def get(self, tool_id: str) -> Optional[ToolRegistration]:
        """Get tool registration by ID."""
        with self._lock:
            return self._tools.get(tool_id)

    def get_by_name(self, name: str) -> Optional[ToolRegistration]:
        """Get tool registration by name."""
        with self._lock:
            tool_id = self._by_name.get(name)
            if tool_id:
                return self._tools.get(tool_id)
            return None

    def query(self, query: ToolQuery) -> List[ToolRegistration]:
        """
        Query tools.

        Args:
            query: Query parameters

        Returns:
            List of matching registrations
        """
        with self._lock:
            results = []
            for registration in self._tools.values():
                if query.matches(registration):
                    results.append(registration)
            return results

    def list_all(self) -> List[ToolRegistration]:
        """List all registered tools."""
        with self._lock:
            return list(self._tools.values())

    def list_available(self) -> List[ToolRegistration]:
        """List all available (approved and enabled) tools."""
        return self.query(ToolQuery(available_only=True))

    def list_pending(self) -> List[ToolRegistration]:
        """List all pending tools awaiting approval."""
        return self.query(ToolQuery(statuses={ToolStatus.PENDING}))

    def list_by_category(self, category: ToolCategory) -> List[ToolRegistration]:
        """List tools by category."""
        return self.query(ToolQuery(categories={category}))

    def list_by_risk(self, risk_level: ToolRiskLevel) -> List[ToolRegistration]:
        """List tools by risk level."""
        return self.query(ToolQuery(risk_levels={risk_level}))

    def record_invocation(self, tool_id: str) -> None:
        """Record a tool invocation."""
        with self._lock:
            registration = self._tools.get(tool_id)
            if registration:
                registration.invocation_count += 1
                registration.last_invoked = datetime.now()
                self._emit_event("invoked", registration)

    def on(self, event: str, handler: Callable) -> None:
        """Register an event handler."""
        if event in self._event_handlers:
            self._event_handlers[event].append(handler)

    def _emit_event(self, event: str, registration: ToolRegistration) -> None:
        """Emit an event to handlers."""
        for handler in self._event_handlers.get(event, []):
            try:
                handler(registration)
            except Exception as e:
                logger.warning(f"Event handler error: {e}")

    def _get_registry_hmac_key(self) -> bytes:
        """Derive an HMAC key for registry state integrity verification."""
        # Use a deterministic key derived from the storage path
        key_material = f"agent-os-registry-v1:{self._storage_path}".encode()
        return hashlib.sha256(key_material).digest()

    def _compute_state_hmac(self, state_json: str) -> str:
        """Compute HMAC-SHA256 of the serialized state."""
        key = self._get_registry_hmac_key()
        return hmac_module.new(key, state_json.encode(), hashlib.sha256).hexdigest()

    def _save_state(self) -> None:
        """Save registry state to disk with HMAC integrity verification."""
        if not self._storage_path:
            return

        try:
            state_file = self._storage_path / "registry_state.json"
            state = {
                "tools": {
                    tid: {
                        "name": reg.tool.name,
                        "status": reg.status.name,
                        "registered_at": reg.registered_at.isoformat(),
                        "registered_by": reg.registered_by,
                        "approved_at": reg.approved_at.isoformat() if reg.approved_at else None,
                        "approved_by": reg.approved_by,
                        "disabled_at": reg.disabled_at.isoformat() if reg.disabled_at else None,
                        "disabled_reason": reg.disabled_reason,
                        "invocation_count": reg.invocation_count,
                        "metadata": reg.metadata,
                    }
                    for tid, reg in self._tools.items()
                },
                "name_index": self._by_name,
                "saved_at": datetime.now().isoformat(),
            }

            state_json = json.dumps(state, sort_keys=True)
            state_hmac = self._compute_state_hmac(state_json)

            signed_state = {
                "state": state,
                "hmac": state_hmac,
            }

            with open(state_file, "w") as f:
                json.dump(signed_state, f, indent=2)

        except Exception as e:
            logger.error(f"Failed to save registry state: {e}")

    def _load_state(self) -> None:
        """Load registry state from disk with HMAC integrity verification."""
        if not self._storage_path:
            return

        state_file = self._storage_path / "registry_state.json"
        if not state_file.exists():
            return

        try:
            with open(state_file, "r") as f:
                raw = json.load(f)

            # Handle both signed (new) and unsigned (legacy) formats
            if "hmac" in raw and "state" in raw:
                state = raw["state"]
                stored_hmac = raw["hmac"]
                # Verify HMAC
                state_json = json.dumps(state, sort_keys=True)
                expected_hmac = self._compute_state_hmac(state_json)
                if not hmac_module.compare_digest(stored_hmac, expected_hmac):
                    logger.error(
                        "Registry state HMAC verification failed — possible tampering. "
                        "State file will not be loaded."
                    )
                    return
            else:
                # Legacy unsigned format — load but log warning
                state = raw
                logger.warning(
                    "Registry state file has no HMAC signature. "
                    "It will be re-signed on next save."
                )

            # Restore name index
            self._by_name = state.get("name_index", {})

            logger.info(f"Loaded registry state: {len(state.get('tools', {}))} tools")

        except Exception as e:
            logger.error(f"Failed to load registry state: {e}")

    def get_statistics(self) -> Dict[str, Any]:
        """Get registry statistics."""
        with self._lock:
            by_status = {}
            by_category = {}
            by_risk = {}
            total_invocations = 0

            for registration in self._tools.values():
                # By status
                status = registration.status.name
                by_status[status] = by_status.get(status, 0) + 1

                # By category
                category = registration.schema.category.value
                by_category[category] = by_category.get(category, 0) + 1

                # By risk
                risk = registration.schema.risk_level.name
                by_risk[risk] = by_risk.get(risk, 0) + 1

                total_invocations += registration.invocation_count

            return {
                "total_tools": len(self._tools),
                "by_status": by_status,
                "by_category": by_category,
                "by_risk_level": by_risk,
                "total_invocations": total_invocations,
            }


# Global registry instance
_global_registry: Optional[ToolRegistry] = None


def get_registry() -> ToolRegistry:
    """Get the global tool registry."""
    global _global_registry
    if _global_registry is None:
        _global_registry = ToolRegistry()
    return _global_registry


def set_registry(registry: ToolRegistry) -> None:
    """Set the global tool registry."""
    global _global_registry
    _global_registry = registry


def create_registry(
    storage_path: Optional[Path] = None,
    auto_approve_low_risk: bool = False,
) -> ToolRegistry:
    """
    Create a new tool registry.

    Args:
        storage_path: Path for persistent storage
        auto_approve_low_risk: Auto-approve LOW risk tools

    Returns:
        ToolRegistry instance
    """
    return ToolRegistry(
        storage_path=storage_path,
        auto_approve_low_risk=auto_approve_low_risk,
    )
