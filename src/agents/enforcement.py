"""
Agent OS Constitutional Boundary Enforcement

Integrates agents with the Constitutional Kernel for rule enforcement.
Provides middleware for validating requests against constitutional rules.
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Dict, List, Optional

from src.core.constitution import (
    ConstitutionalKernel,
    EnforcementResult,
    RequestContext,
)
from src.core.models import Rule
from src.messaging.models import (
    CheckStatus,
    ConstitutionalCheck,
    FlowRequest,
    FlowResponse,
    MessageStatus,
)

from .interface import AgentInterface, RequestValidationResult

logger = logging.getLogger(__name__)


@dataclass
class EnforcementConfig:
    """Configuration for constitutional enforcement."""

    strict_mode: bool = True  # Fail on any violation
    allow_conditional: bool = True  # Allow conditional approvals
    max_violations_before_block: int = 3  # Block after N violations in session
    escalation_timeout_seconds: int = 300  # Timeout for human escalation
    cache_enforcement_results: bool = True  # Cache enforcement decisions
    cache_ttl_seconds: int = 60  # Cache TTL


@dataclass
class ViolationRecord:
    """Record of a constitutional violation."""

    timestamp: datetime
    request_id: str
    agent_name: str
    violated_rules: List[Rule]
    request_content: str
    was_blocked: bool
    escalated: bool


class ConstitutionalEnforcer:
    """
    Enforces constitutional boundaries on agent requests.

    Acts as middleware between the agent interface and the constitutional kernel.
    """

    def __init__(
        self,
        kernel: ConstitutionalKernel,
        config: Optional[EnforcementConfig] = None,
    ):
        """
        Initialize enforcer.

        Args:
            kernel: Constitutional kernel instance
            config: Enforcement configuration
        """
        self.kernel = kernel
        self.config = config or EnforcementConfig()

        self._violations: List[ViolationRecord] = []
        self._enforcement_cache: Dict[str, EnforcementResult] = {}
        self._session_violations: Dict[str, int] = {}  # agent -> count
        self._escalation_callbacks: List[Callable[[ViolationRecord], None]] = []

    def validate_request(
        self,
        request: FlowRequest,
        agent: AgentInterface,
    ) -> RequestValidationResult:
        """
        Validate a request against constitutional rules.

        Args:
            request: Incoming request
            agent: Agent that will process the request

        Returns:
            RequestValidationResult with validation outcome
        """
        result = RequestValidationResult(is_valid=True)

        # Create request context for kernel
        context = RequestContext(
            request_id=str(request.request_id),
            source=request.source,
            destination=request.destination,
            intent=request.intent,
            content=request.content.prompt,
            requires_memory=request.content.metadata.requires_memory,
            metadata={
                "priority": request.content.metadata.priority.value,
                "tags": request.content.metadata.tags,
            },
        )

        # Enforce against kernel
        enforcement = self.kernel.enforce(context)

        # Process enforcement result
        result.applicable_rules = enforcement.applicable_rules

        if not enforcement.allowed:
            # Violation detected
            violation = ViolationRecord(
                timestamp=datetime.now(),
                request_id=str(request.request_id),
                agent_name=agent.name,
                violated_rules=enforcement.violated_rules,
                request_content=request.content.prompt[:500],
                was_blocked=True,
                escalated=enforcement.escalate_to_human,
            )
            self._record_violation(violation)

            # Check if we should block
            if self._should_block(agent.name):
                for rule in enforcement.violated_rules:
                    result.add_error(
                        f"Constitutional violation [{rule.section}]: {rule.content[:100]}"
                    )

            # Check if escalation is required
            if enforcement.escalate_to_human:
                result.requires_escalation = True
                result.escalation_reason = enforcement.reason

                # Notify escalation callbacks
                for callback in self._escalation_callbacks:
                    try:
                        callback(violation)
                    except Exception as e:
                        logger.error(f"Escalation callback error: {e}")

            # Add suggestions as warnings
            for suggestion in enforcement.suggestions:
                result.add_warning(suggestion)

        return result

    def approve_request(self, request: FlowRequest) -> FlowRequest:
        """
        Mark a request as constitutionally approved.

        Args:
            request: Request to approve

        Returns:
            Request with constitutional check updated
        """
        request.constitutional_check = ConstitutionalCheck(
            validated_by="smith",
            timestamp=datetime.now(),
            status=CheckStatus.APPROVED,
        )
        return request

    def deny_request(
        self,
        request: FlowRequest,
        reason: str,
        violated_rule_ids: Optional[List[str]] = None,
    ) -> FlowRequest:
        """
        Mark a request as constitutionally denied.

        Args:
            request: Request to deny
            reason: Denial reason
            violated_rule_ids: IDs of violated rules

        Returns:
            Request with constitutional check updated
        """
        request.constitutional_check = ConstitutionalCheck(
            validated_by="smith",
            timestamp=datetime.now(),
            status=CheckStatus.DENIED,
            reason=reason,
            rule_ids=violated_rule_ids or [],
        )
        return request

    def conditionally_approve(
        self,
        request: FlowRequest,
        constraints: List[str],
    ) -> FlowRequest:
        """
        Conditionally approve a request with constraints.

        Args:
            request: Request to approve
            constraints: Constraints that must be followed

        Returns:
            Request with constitutional check updated
        """
        request.constitutional_check = ConstitutionalCheck(
            validated_by="smith",
            timestamp=datetime.now(),
            status=CheckStatus.CONDITIONAL,
            constraints=constraints,
        )
        return request

    def register_escalation_callback(self, callback: Callable[[ViolationRecord], None]) -> None:
        """
        Register a callback for when escalation is required.

        Args:
            callback: Function to call with violation record
        """
        self._escalation_callbacks.append(callback)

    def get_violations(
        self,
        agent_name: Optional[str] = None,
        since: Optional[datetime] = None,
    ) -> List[ViolationRecord]:
        """
        Get violation records.

        Args:
            agent_name: Filter by agent (optional)
            since: Filter by time (optional)

        Returns:
            List of violation records
        """
        violations = self._violations

        if agent_name:
            violations = [v for v in violations if v.agent_name == agent_name]

        if since:
            violations = [v for v in violations if v.timestamp >= since]

        return violations

    def get_violation_count(self, agent_name: str) -> int:
        """Get session violation count for an agent."""
        return self._session_violations.get(agent_name, 0)

    def reset_violation_count(self, agent_name: str) -> None:
        """Reset session violation count for an agent."""
        self._session_violations[agent_name] = 0

    def clear_cache(self) -> None:
        """Clear enforcement cache."""
        self._enforcement_cache.clear()

    def _record_violation(self, violation: ViolationRecord) -> None:
        """Record a violation."""
        self._violations.append(violation)

        # Update session counter
        agent = violation.agent_name
        self._session_violations[agent] = self._session_violations.get(agent, 0) + 1

        logger.warning(
            f"Constitutional violation by {agent}: "
            f"{len(violation.violated_rules)} rules violated"
        )

    def _should_block(self, agent_name: str) -> bool:
        """Determine if agent should be blocked based on violation history."""
        if self.config.strict_mode:
            return True

        session_count = self._session_violations.get(agent_name, 0)
        return session_count >= self.config.max_violations_before_block


class EnforcementMiddleware:
    """
    Middleware that wraps agent request processing with constitutional enforcement.
    """

    def __init__(
        self,
        enforcer: ConstitutionalEnforcer,
        agent: AgentInterface,
    ):
        """
        Initialize middleware.

        Args:
            enforcer: Constitutional enforcer
            agent: Agent to wrap
        """
        self.enforcer = enforcer
        self.agent = agent

        # Load constitutional rules for this agent
        rules = self.enforcer.kernel.get_rules_for_agent(agent.name)
        agent.set_constitutional_rules(rules)

    def handle_request(self, request: FlowRequest) -> FlowResponse:
        """
        Handle a request with constitutional enforcement.

        Args:
            request: Incoming request

        Returns:
            FlowResponse
        """
        # Pre-validate with enforcer
        validation = self.enforcer.validate_request(request, self.agent)

        if not validation.is_valid:
            # Create refused response
            return request.create_response(
                source=self.agent.name,
                status=MessageStatus.REFUSED,
                output="Request refused due to constitutional constraints.",
                reasoning="; ".join(validation.errors),
            )

        if validation.requires_escalation:
            # Request requires human approval
            response = request.create_response(
                source=self.agent.name,
                status=MessageStatus.PARTIAL,
                output="This request requires human approval.",
                reasoning=validation.escalation_reason,
            )
            response.next_actions.append(
                {
                    "action": "escalate_to_human",
                    "reason": validation.escalation_reason,
                }
            )
            return response

        # Mark request as approved
        approved_request = self.enforcer.approve_request(request)

        # Process with agent
        return self.agent.handle_request(approved_request)


def create_enforced_agent(
    agent: AgentInterface,
    kernel: ConstitutionalKernel,
    config: Optional[EnforcementConfig] = None,
) -> EnforcementMiddleware:
    """
    Create an agent with constitutional enforcement middleware.

    Args:
        agent: Agent to enforce
        kernel: Constitutional kernel
        config: Enforcement configuration

    Returns:
        EnforcementMiddleware wrapping the agent
    """
    enforcer = ConstitutionalEnforcer(kernel, config)
    return EnforcementMiddleware(enforcer, agent)
