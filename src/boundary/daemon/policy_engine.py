"""
Smith Daemon Policy Engine

Manages security modes and enforces policies based on system state.
Modes: Lockdown, Restricted, Trusted, Emergency

This is part of Agent Smith's system-level enforcement mechanism within Agent-OS,
distinct from the external boundary-daemon project.
"""

import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


class BoundaryMode(Enum):
    """Boundary enforcement modes."""

    LOCKDOWN = auto()  # Maximum security - all external access blocked
    RESTRICTED = auto()  # Limited access - only whitelisted operations
    TRUSTED = auto()  # Normal operation with monitoring
    EMERGENCY = auto()  # Emergency mode - human intervention required


class RequestType(Enum):
    """Types of requests that require boundary checking."""

    NETWORK_ACCESS = auto()  # Any network operation
    FILE_WRITE = auto()  # Writing to filesystem
    PROCESS_SPAWN = auto()  # Starting new processes
    MEMORY_ACCESS = auto()  # Accessing memory vault
    EXTERNAL_API = auto()  # Calling external APIs
    AGENT_ACTIVATION = auto()  # Activating an agent
    CONFIGURATION_CHANGE = auto()  # Changing system configuration


class Decision(Enum):
    """Policy decision outcomes."""

    ALLOW = auto()  # Request allowed
    DENY = auto()  # Request denied
    ESCALATE = auto()  # Requires human approval
    AUDIT = auto()  # Allow but log for audit


@dataclass
class PolicyRequest:
    """A request for policy evaluation."""

    request_id: str
    request_type: RequestType
    source: str  # Who is making the request
    target: str  # What is being accessed
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PolicyDecision:
    """Result of policy evaluation."""

    request: PolicyRequest
    decision: Decision
    reason: str
    mode_at_decision: BoundaryMode
    timestamp: datetime = field(default_factory=datetime.now)
    conditions: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            "request_id": self.request.request_id,
            "request_type": self.request.request_type.name,
            "source": self.request.source,
            "target": self.request.target,
            "decision": self.decision.name,
            "reason": self.reason,
            "mode": self.mode_at_decision.name,
            "timestamp": self.timestamp.isoformat(),
            "conditions": self.conditions,
        }


@dataclass
class PolicyRule:
    """A policy rule for the policy engine."""

    id: str
    description: str
    request_types: Set[RequestType]
    modes: Set[BoundaryMode]  # Modes where this rule applies
    decision: Decision
    priority: int = 0  # Higher priority rules evaluated first
    condition: Optional[Callable[[PolicyRequest], bool]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def matches(self, request: PolicyRequest, mode: BoundaryMode) -> bool:
        """Check if this rule matches the request."""
        if mode not in self.modes:
            return False
        if request.request_type not in self.request_types:
            return False
        if self.condition and not self.condition(request):
            return False
        return True


class PolicyEngine:
    """
    Policy engine for boundary enforcement.

    Evaluates requests against the current boundary mode and
    registered policy rules to make allow/deny decisions.
    """

    def __init__(
        self,
        initial_mode: BoundaryMode = BoundaryMode.RESTRICTED,
        on_mode_change: Optional[Callable[[BoundaryMode, BoundaryMode], None]] = None,
    ):
        """
        Initialize policy engine.

        Args:
            initial_mode: Starting boundary mode
            on_mode_change: Callback when mode changes
        """
        self._mode = initial_mode
        self._on_mode_change = on_mode_change
        self._rules: List[PolicyRule] = []
        self._decision_log: List[PolicyDecision] = []
        self._lock = threading.Lock()
        self._mode_history: List[tuple[datetime, BoundaryMode]] = [(datetime.now(), initial_mode)]

        # Whitelist for allowed operations in each mode
        self._whitelists: Dict[BoundaryMode, Dict[RequestType, Set[str]]] = {
            BoundaryMode.LOCKDOWN: {},
            BoundaryMode.RESTRICTED: {},
            BoundaryMode.TRUSTED: {},
            BoundaryMode.EMERGENCY: {},
        }

        # Install default rules
        self._install_default_rules()

    @property
    def mode(self) -> BoundaryMode:
        """Get current boundary mode."""
        return self._mode

    def set_mode(
        self,
        new_mode: BoundaryMode,
        reason: str = "",
        authorization: Optional[str] = None,
    ) -> bool:
        """
        Set boundary mode.

        Args:
            new_mode: Mode to switch to
            reason: Reason for mode change
            authorization: Required for some mode transitions

        Returns:
            True if mode change successful
        """
        with self._lock:
            old_mode = self._mode

            # Certain transitions require authorization
            if self._requires_authorization(old_mode, new_mode):
                if not authorization or len(authorization) < 8:
                    logger.warning(f"Unauthorized mode change attempt: {old_mode} -> {new_mode}")
                    return False

            self._mode = new_mode
            self._mode_history.append((datetime.now(), new_mode))

            logger.info(f"Boundary mode changed: {old_mode.name} -> {new_mode.name} ({reason})")

            if self._on_mode_change:
                try:
                    self._on_mode_change(old_mode, new_mode)
                except Exception as e:
                    logger.error(f"Mode change callback error: {e}")

            return True

    def lockdown(self, reason: str = "Security event") -> bool:
        """Enter lockdown mode immediately."""
        return self.set_mode(BoundaryMode.LOCKDOWN, reason)

    def emergency(self, reason: str = "Emergency activation") -> bool:
        """Enter emergency mode (requires human intervention)."""
        return self.set_mode(BoundaryMode.EMERGENCY, reason)

    def add_rule(self, rule: PolicyRule) -> None:
        """Add a policy rule."""
        with self._lock:
            self._rules.append(rule)
            # Sort by priority (highest first)
            self._rules.sort(key=lambda r: r.priority, reverse=True)
            logger.info(f"Policy rule added: {rule.id}")

    def remove_rule(self, rule_id: str) -> bool:
        """Remove a policy rule."""
        with self._lock:
            for i, rule in enumerate(self._rules):
                if rule.id == rule_id:
                    del self._rules[i]
                    return True
            return False

    def whitelist(
        self,
        mode: BoundaryMode,
        request_type: RequestType,
        target: str,
    ) -> None:
        """Add a target to the whitelist for a mode/request type."""
        with self._lock:
            if request_type not in self._whitelists[mode]:
                self._whitelists[mode][request_type] = set()
            self._whitelists[mode][request_type].add(target)

    def evaluate(self, request: PolicyRequest) -> PolicyDecision:
        """
        Evaluate a request against current policy.

        Args:
            request: The request to evaluate

        Returns:
            PolicyDecision with the outcome
        """
        with self._lock:
            mode = self._mode
            decision = Decision.DENY
            reason = "Default deny"
            conditions = []

            # Emergency mode blocks everything
            if mode == BoundaryMode.EMERGENCY:
                return self._make_decision(
                    request,
                    Decision.DENY,
                    "Emergency mode - all operations blocked",
                    mode,
                    ["emergency_mode_active"],
                )

            # Check whitelist first
            if self._is_whitelisted(request, mode):
                decision = Decision.ALLOW
                reason = "Whitelisted"
                conditions.append("whitelisted")

            else:
                # Evaluate rules
                for rule in self._rules:
                    if rule.matches(request, mode):
                        decision = rule.decision
                        reason = rule.description
                        conditions.append(f"rule:{rule.id}")
                        break

            # Mode-specific defaults
            if decision == Decision.DENY:
                if mode == BoundaryMode.LOCKDOWN:
                    reason = "Lockdown mode - denied by default"
                    conditions.append("lockdown_default_deny")
                elif mode == BoundaryMode.RESTRICTED:
                    if request.request_type in [
                        RequestType.NETWORK_ACCESS,
                        RequestType.EXTERNAL_API,
                    ]:
                        reason = "Restricted mode - external access denied"
                        conditions.append("restricted_external_deny")
                    else:
                        decision = Decision.AUDIT
                        reason = "Restricted mode - allowed with audit"
                        conditions.append("restricted_audit")
                elif mode == BoundaryMode.TRUSTED:
                    decision = Decision.AUDIT
                    reason = "Trusted mode - allowed with audit"
                    conditions.append("trusted_audit")

            return self._make_decision(request, decision, reason, mode, conditions)

    def get_decision_log(self) -> List[PolicyDecision]:
        """Get the decision log."""
        return list(self._decision_log)

    def get_mode_history(self) -> List[tuple[datetime, BoundaryMode]]:
        """Get mode change history."""
        return list(self._mode_history)

    def _make_decision(
        self,
        request: PolicyRequest,
        decision: Decision,
        reason: str,
        mode: BoundaryMode,
        conditions: List[str],
    ) -> PolicyDecision:
        """Create and log a policy decision."""
        policy_decision = PolicyDecision(
            request=request,
            decision=decision,
            reason=reason,
            mode_at_decision=mode,
            conditions=conditions,
        )

        self._decision_log.append(policy_decision)

        # Keep log size manageable
        if len(self._decision_log) > 10000:
            self._decision_log = self._decision_log[-10000:]

        log_level = logging.DEBUG if decision == Decision.ALLOW else logging.INFO
        logger.log(
            log_level,
            f"Policy decision: {decision.name} for {request.request_type.name} ({reason})",
        )

        return policy_decision

    def _is_whitelisted(self, request: PolicyRequest, mode: BoundaryMode) -> bool:
        """Check if request is whitelisted."""
        whitelist = self._whitelists.get(mode, {})
        targets = whitelist.get(request.request_type, set())
        return request.target in targets or "*" in targets

    def _requires_authorization(
        self,
        old_mode: BoundaryMode,
        new_mode: BoundaryMode,
    ) -> bool:
        """Check if mode transition requires authorization."""
        # Going from higher security to lower always requires authorization
        security_levels = {
            BoundaryMode.EMERGENCY: 4,
            BoundaryMode.LOCKDOWN: 3,
            BoundaryMode.RESTRICTED: 2,
            BoundaryMode.TRUSTED: 1,
        }

        old_level = security_levels.get(old_mode, 0)
        new_level = security_levels.get(new_mode, 0)

        # Decreasing security requires authorization
        return new_level < old_level

    def _install_default_rules(self) -> None:
        """Install default policy rules."""
        # Lockdown: deny all network
        self.add_rule(
            PolicyRule(
                id="lockdown_no_network",
                description="Lockdown mode denies all network access",
                request_types={RequestType.NETWORK_ACCESS, RequestType.EXTERNAL_API},
                modes={BoundaryMode.LOCKDOWN},
                decision=Decision.DENY,
                priority=100,
            )
        )

        # Lockdown: deny process spawn
        self.add_rule(
            PolicyRule(
                id="lockdown_no_spawn",
                description="Lockdown mode denies process spawning",
                request_types={RequestType.PROCESS_SPAWN},
                modes={BoundaryMode.LOCKDOWN},
                decision=Decision.DENY,
                priority=100,
            )
        )

        # Restricted: escalate external access
        self.add_rule(
            PolicyRule(
                id="restricted_escalate_external",
                description="Restricted mode escalates external access",
                request_types={RequestType.NETWORK_ACCESS, RequestType.EXTERNAL_API},
                modes={BoundaryMode.RESTRICTED},
                decision=Decision.ESCALATE,
                priority=50,
            )
        )

        # Trusted: audit all operations
        self.add_rule(
            PolicyRule(
                id="trusted_audit_all",
                description="Trusted mode audits all operations",
                request_types=set(RequestType),
                modes={BoundaryMode.TRUSTED},
                decision=Decision.AUDIT,
                priority=10,
            )
        )

        # Always allow memory access for agents
        self.add_rule(
            PolicyRule(
                id="allow_agent_memory",
                description="Allow agent memory access",
                request_types={RequestType.MEMORY_ACCESS},
                modes={BoundaryMode.RESTRICTED, BoundaryMode.TRUSTED},
                decision=Decision.ALLOW,
                priority=75,
                condition=lambda r: r.source.startswith("agent:"),
            )
        )


def create_policy_engine(
    initial_mode: BoundaryMode = BoundaryMode.RESTRICTED,
    on_mode_change: Optional[Callable[[BoundaryMode, BoundaryMode], None]] = None,
) -> PolicyEngine:
    """Factory function to create a policy engine."""
    return PolicyEngine(
        initial_mode=initial_mode,
        on_mode_change=on_mode_change,
    )
