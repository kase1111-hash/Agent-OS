"""
Agent OS Smith (Guardian) Agent

The security validation agent responsible for:
- Pre-execution validation (S1-S5)
- Post-execution monitoring (S6-S8)
- Refusal handling (S9-S12)
- Emergency controls
"""

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Set

from src.agents.interface import (
    AgentCapabilities,
    AgentMetrics,
    AgentState,
    BaseAgent,
    CapabilityType,
    RequestValidationResult,
    Rule,
)
from src.core.constitution import ConstitutionalKernel
from src.messaging.models import (
    CheckStatus,
    ConstitutionalCheck,
    FlowRequest,
    FlowResponse,
    MessageStatus,
)

from .emergency import (
    EmergencyControls,
    IncidentSeverity,
    SecurityIncident,
    SystemMode,
)
from .post_monitor import MonitorResult, PostExecutionMonitor, PostMonitorResult
from .pre_validator import CheckResult, PreExecutionValidator, PreValidationResult
from .refusal_engine import RefusalEngine, RefusalResponse, RefusalType

logger = logging.getLogger(__name__)


@dataclass
class SmithMetrics:
    """Smith-specific metrics."""

    pre_validations: int = 0
    post_monitors: int = 0
    refusals_issued: int = 0
    escalations: int = 0
    safe_mode_triggers: int = 0
    avg_validation_time_ms: float = 0.0
    blocks_by_check: Dict[str, int] = field(default_factory=dict)


class SmithAgent(BaseAgent):
    """
    Smith (Guardian) Agent - Security Validation.

    Implements all 12 security checks (S1-S12) plus emergency controls.

    Role:
    - Validate all requests before execution
    - Monitor all responses after execution
    - Block harmful or unauthorized requests
    - Trigger emergency modes when needed
    """

    def __init__(
        self,
        kernel: Optional[ConstitutionalKernel] = None,
        model: str = "llama3:8b-instruct",
    ):
        """
        Initialize Smith agent.

        Args:
            kernel: Constitutional kernel for rule validation
            model: Ollama model to use (if LLM-based validation needed)
        """
        super().__init__(
            name="smith",
            description="Guardian agent for security validation and constitutional enforcement",
            version="1.0.0",
            capabilities={CapabilityType.VALIDATION},
            supported_intents=["security.*", "validation.*", "*"],  # Handles all
        )

        self.kernel = kernel
        self.model = model

        # Core components (initialized in initialize())
        self._pre_validator: Optional[PreExecutionValidator] = None
        self._post_monitor: Optional[PostExecutionMonitor] = None
        self._refusal_engine: Optional[RefusalEngine] = None
        self._emergency: Optional[EmergencyControls] = None

        # Smith-specific metrics
        self._smith_metrics = SmithMetrics()

        # Validation timing history
        self._validation_times: List[int] = []

    def initialize(self, config: Dict[str, Any]) -> bool:
        """
        Initialize Smith with configuration.

        Args:
            config: Configuration including:
                - strict_mode: Block on any check failure
                - allow_escalation: Allow human escalation
                - auto_escalate_mode: Auto-trigger safe mode on critical incidents
                - incident_log_path: Path to incident log

        Returns:
            True if initialization successful
        """
        self._do_initialize(config)

        try:
            strict_mode = config.get("strict_mode", True)
            allow_escalation = config.get("allow_escalation", True)
            auto_escalate = config.get("auto_escalate_mode", True)
            incident_log_path = config.get("incident_log_path")

            # Initialize pre-execution validator
            self._pre_validator = PreExecutionValidator(
                kernel=self.kernel,
                strict_mode=strict_mode,
                allow_escalation=allow_escalation,
            )

            # Initialize post-execution monitor
            self._post_monitor = PostExecutionMonitor(
                sensitivity_level=config.get("sensitivity_level", 2),
            )

            # Initialize refusal engine
            self._refusal_engine = RefusalEngine(
                strict_mode=strict_mode,
            )

            # Initialize emergency controls
            self._emergency = EmergencyControls(
                incident_log_path=incident_log_path,
                auto_escalate=auto_escalate,
            )

            self._state = AgentState.READY
            logger.info("Smith (Guardian) initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Smith initialization failed: {e}")
            self._state = AgentState.ERROR
            return False

    def validate_request(self, request: FlowRequest) -> RequestValidationResult:
        """
        Validate an incoming request (pre-execution).

        This is called BEFORE any agent processes the request.

        Args:
            request: The request to validate

        Returns:
            RequestValidationResult
        """
        result = RequestValidationResult(is_valid=True)
        start_time = time.time()

        # Check if system is operational
        if not self._emergency.is_operational:
            result.add_error(f"System in {self._emergency.current_mode.name} mode")
            return result

        try:
            # Run pre-execution validation (S1-S5)
            target_agent = request.destination
            pre_result = self._pre_validator.validate(
                request=request,
                target_agent=target_agent,
            )
            self._smith_metrics.pre_validations += 1

            # Add any failures/warnings to result
            for check in pre_result.checks:
                if check.result == CheckResult.FAIL:
                    result.add_error(f"[{check.check_id}] {check.message}")
                    self._track_block(check.check_id)
                elif check.result == CheckResult.WARN:
                    result.add_warning(f"[{check.check_id}] {check.message}")

            # Handle escalation
            if pre_result.requires_escalation:
                result.requires_escalation = True
                result.escalation_reason = pre_result.escalation_reason
                self._smith_metrics.escalations += 1

            # Run refusal engine (S9-S12)
            refusal = self._refusal_engine.evaluate(request)

            if refusal.is_refused:
                result.add_error(refusal.message)
                self._smith_metrics.refusals_issued += 1

                # Log as incident
                self._emergency.log_incident(
                    severity=IncidentSeverity.MEDIUM,
                    category="request_refused",
                    description=refusal.message,
                    request_id=str(request.request_id),
                    triggered_by=(
                        refusal.decisions[0].check_id if refusal.decisions else "refusal_engine"
                    ),
                )

            elif refusal.requires_human_review:
                result.requires_escalation = True
                result.escalation_reason = refusal.review_reason

            # Track timing
            validation_time = int((time.time() - start_time) * 1000)
            self._update_timing(validation_time)

        except Exception as e:
            logger.error(f"Validation error: {e}")
            result.add_error(f"Validation failed: {str(e)}")

            # Log critical error
            self._emergency.log_incident(
                severity=IncidentSeverity.HIGH,
                category="validation_error",
                description=str(e),
                request_id=str(request.request_id),
            )

        return result

    def post_validate(
        self,
        request: FlowRequest,
        response: FlowResponse,
        agent_name: str,
    ) -> PostMonitorResult:
        """
        Validate a response after execution (post-execution).

        This is called AFTER an agent produces a response.

        Args:
            request: Original request
            response: Agent response to validate
            agent_name: Name of agent that produced response

        Returns:
            PostMonitorResult
        """
        self._smith_metrics.post_monitors += 1

        result = self._post_monitor.monitor(
            request=request,
            response=response,
            agent_name=agent_name,
        )

        # Handle critical violations
        if result.critical_violations:
            for check in result.critical_violations:
                self._emergency.log_incident(
                    severity=IncidentSeverity.CRITICAL,
                    category=f"post_monitor_{check.check_id.lower()}",
                    description=check.message,
                    source_agent=agent_name,
                    request_id=str(request.request_id),
                )
                self._track_block(check.check_id)

        # Handle regular violations
        for violation in result.violations:
            self._emergency.log_incident(
                severity=IncidentSeverity.HIGH,
                category="post_monitor_violation",
                description=violation,
                source_agent=agent_name,
                request_id=str(request.request_id),
            )

        return result

    def process(self, request: FlowRequest) -> FlowResponse:
        """
        Process a request directed at Smith.

        Smith typically doesn't process user requests directly,
        but can respond to security-related queries.

        Args:
            request: The request

        Returns:
            FlowResponse
        """
        prompt_lower = request.content.prompt.lower()

        # Handle status requests
        if any(word in prompt_lower for word in ["status", "health", "mode"]):
            return self._handle_status_request(request)

        # Handle metrics requests
        if any(word in prompt_lower for word in ["metrics", "statistics", "stats"]):
            return self._handle_metrics_request(request)

        # Handle incident requests
        if "incident" in prompt_lower:
            return self._handle_incident_request(request)

        # Default response
        return request.create_response(
            source="smith",
            status=MessageStatus.SUCCESS,
            output="Smith (Guardian) is active and monitoring system security.",
        )

    def get_capabilities(self) -> AgentCapabilities:
        """Get Smith's capabilities."""
        return AgentCapabilities(
            name=self.name,
            description=self._description,
            version=self._version,
            capabilities=self._capability_types,
            supported_intents=self._supported_intents,
            context_window=4096,
            metadata={
                "checks": [
                    "S1",
                    "S2",
                    "S3",
                    "S4",
                    "S5",
                    "S6",
                    "S7",
                    "S8",
                    "S9",
                    "S10",
                    "S11",
                    "S12",
                ],
                "has_emergency_controls": True,
                "target_latency_ms": 500,
            },
        )

    def shutdown(self) -> bool:
        """Shutdown Smith agent."""
        logger.info("Smith shutting down")

        # Log shutdown
        if self._emergency:
            self._emergency.log_incident(
                severity=IncidentSeverity.LOW,
                category="agent_shutdown",
                description="Smith agent shutting down",
                triggered_by="shutdown_request",
            )

        self._state = AgentState.SHUTDOWN
        return True

    def trigger_safe_mode(self, reason: str) -> bool:
        """
        Trigger safe mode.

        Args:
            reason: Why safe mode is being triggered

        Returns:
            True if successful
        """
        if self._emergency:
            self._smith_metrics.safe_mode_triggers += 1
            return self._emergency.trigger_safe_mode(
                reason=reason,
                triggered_by="smith",
            )
        return False

    def trigger_lockdown(self, reason: str) -> bool:
        """
        Trigger emergency lockdown.

        Args:
            reason: Why lockdown is being triggered

        Returns:
            True if successful
        """
        if self._emergency:
            return self._emergency.trigger_lockdown(
                reason=reason,
                triggered_by="smith",
            )
        return False

    def halt_system(self, reason: str) -> None:
        """
        Halt the system completely.

        Args:
            reason: Why system is being halted
        """
        if self._emergency:
            self._emergency.halt_system(
                reason=reason,
                triggered_by="smith",
            )

    def get_system_mode(self) -> SystemMode:
        """Get current system mode."""
        if self._emergency:
            return self._emergency.current_mode
        return SystemMode.NORMAL

    def create_constitutional_check(
        self,
        approved: bool,
        constraints: Optional[List[str]] = None,
        violations: Optional[List[str]] = None,
    ) -> ConstitutionalCheck:
        """
        Create a ConstitutionalCheck record.

        Args:
            approved: Whether the check passed
            constraints: Any constraints to apply
            violations: Any violations found

        Returns:
            ConstitutionalCheck for attachment to requests/responses
        """
        if violations:
            status = CheckStatus.REJECTED
        elif constraints:
            status = CheckStatus.CONDITIONAL
        else:
            status = CheckStatus.APPROVED

        return ConstitutionalCheck(
            validated_by="smith",
            timestamp=datetime.now(),
            status=status,
            constraints=constraints or [],
            violations=violations or [],
        )

    def _handle_status_request(self, request: FlowRequest) -> FlowResponse:
        """Handle status query."""
        status = {
            "agent": "smith",
            "state": self._state.name,
            "system_mode": self._emergency.current_mode.name if self._emergency else "UNKNOWN",
            "is_operational": self._emergency.is_operational if self._emergency else True,
            "pre_validations": self._smith_metrics.pre_validations,
            "post_monitors": self._smith_metrics.post_monitors,
            "refusals": self._smith_metrics.refusals_issued,
            "avg_validation_time_ms": round(self._smith_metrics.avg_validation_time_ms, 2),
        }

        output = f"Smith Status:\n"
        for key, value in status.items():
            output += f"  {key}: {value}\n"

        return request.create_response(
            source="smith",
            status=MessageStatus.SUCCESS,
            output=output,
        )

    def _handle_metrics_request(self, request: FlowRequest) -> FlowResponse:
        """Handle metrics query."""
        metrics = {
            "pre_validator": self._pre_validator.get_metrics() if self._pre_validator else {},
            "post_monitor": self._post_monitor.get_metrics() if self._post_monitor else {},
            "refusal_engine": self._refusal_engine.get_metrics() if self._refusal_engine else {},
            "emergency": self._emergency.get_status() if self._emergency else {},
            "blocks_by_check": self._smith_metrics.blocks_by_check,
        }

        import json

        return request.create_response(
            source="smith",
            status=MessageStatus.SUCCESS,
            output=json.dumps(metrics, indent=2),
        )

    def _handle_incident_request(self, request: FlowRequest) -> FlowResponse:
        """Handle incident history query."""
        if not self._emergency:
            return request.create_response(
                source="smith",
                status=MessageStatus.ERROR,
                output="Emergency controls not initialized",
            )

        incidents = self._emergency.get_incident_history(limit=10)

        if not incidents:
            return request.create_response(
                source="smith",
                status=MessageStatus.SUCCESS,
                output="No incidents recorded.",
            )

        output = "Recent Incidents:\n"
        for incident in incidents:
            output += (
                f"  [{incident.severity.name}] {incident.incident_id}: {incident.description}\n"
            )

        return request.create_response(
            source="smith",
            status=MessageStatus.SUCCESS,
            output=output,
        )

    def _track_block(self, check_id: str) -> None:
        """Track a block by check ID."""
        self._smith_metrics.blocks_by_check[check_id] = (
            self._smith_metrics.blocks_by_check.get(check_id, 0) + 1
        )

    def _update_timing(self, time_ms: int) -> None:
        """Update validation timing statistics."""
        self._validation_times.append(time_ms)
        # Keep last 100 timings
        if len(self._validation_times) > 100:
            self._validation_times = self._validation_times[-100:]

        self._smith_metrics.avg_validation_time_ms = sum(self._validation_times) / len(
            self._validation_times
        )

    def get_metrics(self) -> AgentMetrics:
        """Get agent metrics."""
        return AgentMetrics(
            requests_processed=self._smith_metrics.pre_validations
            + self._smith_metrics.post_monitors,
            requests_succeeded=self._smith_metrics.pre_validations
            - self._smith_metrics.refusals_issued,
            requests_failed=0,
            requests_refused=self._smith_metrics.refusals_issued,
        )


def create_smith(
    kernel: Optional[ConstitutionalKernel] = None,
    config: Optional[Dict[str, Any]] = None,
) -> SmithAgent:
    """
    Convenience function to create and initialize a Smith agent.

    Args:
        kernel: Constitutional kernel
        config: Configuration options

    Returns:
        Initialized SmithAgent
    """
    agent = SmithAgent(kernel=kernel)
    agent.initialize(config or {})
    return agent
