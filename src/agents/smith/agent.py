"""
Agent OS Smith (Guardian) Agent

The security validation agent responsible for:
- Pre-execution validation (S1-S5)
- Post-execution monitoring (S6-S8)
- Refusal handling (S9-S12)
- Emergency controls
- Attack detection and auto-remediation (when enabled)
"""

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from src.agents.interface import (
    AgentCapabilities,
    AgentMetrics,
    AgentState,
    BaseAgent,
    CapabilityType,
    RequestValidationResult,
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
    SystemMode,
)
from .post_monitor import PostExecutionMonitor, PostMonitorResult
from .pre_validator import CheckResult, PreExecutionValidator
from .refusal_engine import RefusalEngine

# Attack detection imports (optional feature)
try:
    from .attack_detection import (
        AttackDetector,
        AttackEvent,
        AttackSeverity,
        AttackAnalyzer,
        RemediationEngine,
        RecommendationSystem,
        DetectorConfig,
        create_attack_detector,
        create_attack_analyzer,
        create_remediation_engine,
        create_recommendation_system,
    )
    ATTACK_DETECTION_AVAILABLE = True
except ImportError:
    ATTACK_DETECTION_AVAILABLE = False

# Advanced memory imports (optional feature)
try:
    from .advanced_memory import (
        AdvancedMemoryManager,
        MemoryConfig,
        IntelligenceEntry,
        IntelligenceType,
        ThreatCluster,
        ThreatLevel,
        SynthesizedPattern,
        AnomalyScore,
        IntelligenceSummary,
        BoundaryEvent,
        PolicyDecision,
        TripwireAlert,
        BoundaryMode,
        create_advanced_memory,
    )
    ADVANCED_MEMORY_AVAILABLE = True
except ImportError:
    ADVANCED_MEMORY_AVAILABLE = False

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

    # Attack detection metrics
    attacks_detected: int = 0
    attacks_mitigated: int = 0
    recommendations_generated: int = 0
    auto_lockdowns_triggered: int = 0

    # Advanced memory metrics
    memory_events_stored: int = 0
    memory_correlations: int = 0
    memory_patterns_detected: int = 0
    memory_anomalies: int = 0
    memory_threat_clusters: int = 0


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

        # Attack detection components (optional, initialized if enabled)
        self._attack_detector: Optional[Any] = None  # AttackDetector
        self._attack_analyzer: Optional[Any] = None  # AttackAnalyzer
        self._remediation_engine: Optional[Any] = None  # RemediationEngine
        self._recommendation_system: Optional[Any] = None  # RecommendationSystem
        self._attack_detection_enabled: bool = False

        # Advanced memory components (optional, initialized if enabled)
        self._advanced_memory: Optional[Any] = None  # AdvancedMemoryManager
        self._advanced_memory_enabled: bool = False

        # Callbacks for attack events
        self._on_attack_callbacks: List[Callable[[Any], None]] = []

        # Callbacks for memory events (threat clusters, anomalies, patterns)
        self._on_threat_callbacks: List[Callable[[Any], None]] = []
        self._on_anomaly_callbacks: List[Callable[[Any], None]] = []
        self._on_pattern_callbacks: List[Callable[[Any], None]] = []

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
                - attack_detection_enabled: Enable attack detection system
                - attack_detection_config: Attack detector configuration
                - advanced_memory_enabled: Enable advanced memory system
                - advanced_memory_config: Advanced memory configuration

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

            # Initialize attack detection if enabled
            attack_detection_enabled = config.get("attack_detection_enabled", False)
            if attack_detection_enabled and ATTACK_DETECTION_AVAILABLE:
                self._initialize_attack_detection(config.get("attack_detection_config", {}))

            # Initialize advanced memory if enabled
            advanced_memory_enabled = config.get("advanced_memory_enabled", False)
            if advanced_memory_enabled and ADVANCED_MEMORY_AVAILABLE:
                self._initialize_advanced_memory(config.get("advanced_memory_config", {}))

            self._state = AgentState.READY
            logger.info("Smith (Guardian) initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Smith initialization failed: {e}")
            self._state = AgentState.ERROR
            return False

    def _initialize_attack_detection(self, config: Dict[str, Any]) -> None:
        """
        Initialize the attack detection subsystem.

        Args:
            config: Attack detection configuration
        """
        if not ATTACK_DETECTION_AVAILABLE:
            logger.warning("Attack detection module not available")
            return

        try:
            # Build detector config
            detector_config = DetectorConfig(
                enable_siem=config.get("enable_siem", False),
                enable_boundary_events=config.get("enable_boundary_events", True),
                enable_flow_monitoring=config.get("enable_flow_monitoring", True),
                min_confidence=config.get("min_confidence", 0.3),
                auto_lockdown_on_critical=config.get("auto_lockdown_on_critical", True),
            )

            # Initialize attack detector
            self._attack_detector = create_attack_detector(
                config=detector_config,
                on_attack=self._on_attack_detected,
            )

            # Initialize analyzer
            codebase_root = config.get("codebase_root")
            self._attack_analyzer = create_attack_analyzer(
                codebase_root=Path(codebase_root) if codebase_root else None,
            )

            # Initialize remediation engine
            self._remediation_engine = create_remediation_engine(
                codebase_root=Path(codebase_root) if codebase_root else None,
                test_command=config.get("test_command", "pytest tests/"),
                allow_auto_apply=config.get("allow_auto_apply", False),
            )

            # Initialize recommendation system
            self._recommendation_system = create_recommendation_system(
                remediation_engine=self._remediation_engine,
                on_recommendation=self._on_recommendation_created,
            )

            # Start the detector
            self._attack_detector.start()
            self._attack_detection_enabled = True

            logger.info("Attack detection system initialized and started")

        except Exception as e:
            logger.error(f"Failed to initialize attack detection: {e}")
            self._attack_detection_enabled = False

    def _initialize_advanced_memory(self, config: Dict[str, Any]) -> None:
        """
        Initialize the advanced memory subsystem.

        Integrates with Boundary-SIEM and Boundary-Daemon for comprehensive
        security intelligence gathering, correlation, and synthesis.

        Args:
            config: Advanced memory configuration including:
                - storage_path: Path for persistent storage
                - boundary_endpoint: Boundary-Daemon API endpoint
                - boundary_api_key: Boundary-Daemon API key
                - hot_retention_days: Days to keep in hot storage (default: 7)
                - warm_retention_days: Days to keep in warm storage (default: 30)
                - cold_retention_days: Days to keep in cold storage (default: 365)
                - correlation_enabled: Enable threat correlation (default: True)
                - synthesis_enabled: Enable pattern synthesis (default: True)
                - baseline_enabled: Enable behavioral baseline (default: True)
        """
        if not ADVANCED_MEMORY_AVAILABLE:
            logger.warning("Advanced memory module not available")
            return

        try:
            # Build memory config
            memory_config = MemoryConfig(
                storage_path=config.get("storage_path"),
                boundary_endpoint=config.get("boundary_endpoint"),
                boundary_api_key=config.get("boundary_api_key"),
                hot_retention_days=config.get("hot_retention_days", 7),
                warm_retention_days=config.get("warm_retention_days", 30),
                cold_retention_days=config.get("cold_retention_days", 365),
                correlation_enabled=config.get("correlation_enabled", True),
                synthesis_enabled=config.get("synthesis_enabled", True),
                baseline_enabled=config.get("baseline_enabled", True),
                on_high_threat=self._on_high_threat_cluster,
                on_anomaly=self._on_memory_anomaly,
                on_pattern=self._on_memory_pattern,
            )

            # Create and initialize memory manager
            self._advanced_memory = create_advanced_memory(
                config=memory_config,
                auto_start=True,
            )
            self._advanced_memory_enabled = True

            logger.info("Advanced memory system initialized and started")

        except Exception as e:
            logger.error(f"Failed to initialize advanced memory: {e}")
            self._advanced_memory_enabled = False

    def _on_high_threat_cluster(self, cluster: Any) -> None:
        """Handle high-threat cluster detection from memory."""
        if not ADVANCED_MEMORY_AVAILABLE:
            return

        self._smith_metrics.memory_threat_clusters += 1
        logger.warning(
            f"High threat cluster detected: {cluster.cluster_id} - "
            f"{cluster.threat_level.name}"
        )

        # Notify external callbacks
        for callback in self._on_threat_callbacks:
            try:
                callback(cluster)
            except Exception as e:
                logger.error(f"Threat callback error: {e}")

        # Consider triggering safe mode for critical threats
        if cluster.threat_level.value >= 4:  # CRITICAL or higher
            self._smith_metrics.safe_mode_triggers += 1
            self.trigger_safe_mode(
                f"Critical threat cluster: {cluster.cluster_id}"
            )

    def _on_memory_anomaly(self, score: Any) -> None:
        """Handle anomaly detection from memory baseline."""
        if not ADVANCED_MEMORY_AVAILABLE:
            return

        self._smith_metrics.memory_anomalies += 1
        logger.warning(
            f"Behavioral anomaly detected: {score.score_id} - "
            f"score {score.overall_score:.2f}"
        )

        # Log as security incident
        if self._emergency and score.overall_score > 0.5:
            self._emergency.log_incident(
                severity=IncidentSeverity.MEDIUM if score.overall_score < 0.7 else IncidentSeverity.HIGH,
                category="behavioral_anomaly",
                description=score.description,
                triggered_by="advanced_memory",
            )

        # Notify external callbacks
        for callback in self._on_anomaly_callbacks:
            try:
                callback(score)
            except Exception as e:
                logger.error(f"Anomaly callback error: {e}")

    def _on_memory_pattern(self, pattern: Any) -> None:
        """Handle pattern detection from memory synthesizer."""
        if not ADVANCED_MEMORY_AVAILABLE:
            return

        self._smith_metrics.memory_patterns_detected += 1
        logger.info(
            f"Pattern detected: {pattern.pattern_id} - "
            f"{pattern.pattern_type.name} (severity {pattern.severity})"
        )

        # Notify external callbacks
        for callback in self._on_pattern_callbacks:
            try:
                callback(pattern)
            except Exception as e:
                logger.error(f"Pattern callback error: {e}")

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

            # Run attack detection on the request (non-blocking)
            if self._attack_detection_enabled and self._attack_detector:
                self._run_attack_detection(request)

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

    def _run_attack_detection(self, request: FlowRequest) -> None:
        """
        Run attack detection on a request.

        This is called during validation to check for attack patterns.
        Detection is performed but doesn't block the request directly -
        instead, attacks trigger callbacks and may cause lockdown.

        Args:
            request: The request to analyze
        """
        if not self._attack_detector:
            return

        try:
            # Convert request to event format for attack detector
            event_data = {
                "type": "agent_request",
                "request_id": str(request.request_id),
                "source": request.source,
                "destination": request.destination,
                "intent": request.intent,
                "content": request.content.prompt if request.content else "",
                "metadata": request.metadata,
                "timestamp": datetime.now().isoformat(),
            }

            # Process through attack detector
            self._attack_detector.process_flow_event(
                request=event_data,
                agent=request.destination,
            )

        except Exception as e:
            logger.warning(f"Attack detection error (non-fatal): {e}")

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

        # Handle attack-related requests
        if any(word in prompt_lower for word in ["attack", "threat", "security"]):
            return self._handle_attack_request(request)

        # Handle recommendation requests
        if any(word in prompt_lower for word in ["recommendation", "fix", "patch"]):
            return self._handle_recommendation_request(request)

        # Default response
        return request.create_response(
            source="smith",
            status=MessageStatus.SUCCESS,
            output="Smith (Guardian) is active and monitoring system security.",
        )

    def _handle_attack_request(self, request: FlowRequest) -> FlowResponse:
        """Handle attack-related queries."""
        if not self._attack_detection_enabled:
            return request.create_response(
                source="smith",
                status=MessageStatus.SUCCESS,
                output="Attack detection is not enabled.",
            )

        attacks = self.get_detected_attacks(limit=10)
        status = self.get_attack_detection_status()

        output = "Attack Detection Status:\n"
        output += f"  Enabled: {status['enabled']}\n"
        output += f"  Attacks Detected: {status['attacks_detected']}\n"
        output += f"  Recommendations Generated: {status['recommendations_generated']}\n"
        output += f"  Auto-Lockdowns: {status['auto_lockdowns_triggered']}\n\n"

        if attacks:
            output += "Recent Attacks:\n"
            for attack in attacks[:5]:
                output += f"  [{attack['severity']}] {attack['attack_id']}: {attack['attack_type']}\n"
        else:
            output += "No recent attacks detected.\n"

        return request.create_response(
            source="smith",
            status=MessageStatus.SUCCESS,
            output=output,
        )

    def _handle_recommendation_request(self, request: FlowRequest) -> FlowResponse:
        """Handle recommendation-related queries."""
        if not self._recommendation_system:
            return request.create_response(
                source="smith",
                status=MessageStatus.SUCCESS,
                output="Recommendation system is not enabled.",
            )

        pending = self.get_pending_recommendations()

        if not pending:
            return request.create_response(
                source="smith",
                status=MessageStatus.SUCCESS,
                output="No pending fix recommendations.",
            )

        output = f"Pending Fix Recommendations ({len(pending)}):\n\n"
        for rec in pending[:5]:
            output += f"  {rec['recommendation_id']}: {rec['title']}\n"
            output += f"    Priority: {rec['priority']}\n"
            output += f"    Patches: {len(rec.get('patches', []))}\n\n"

        return request.create_response(
            source="smith",
            status=MessageStatus.SUCCESS,
            output=output,
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

        # Stop attack detector if running
        if self._attack_detector and self._attack_detection_enabled:
            try:
                self._attack_detector.stop()
                logger.info("Attack detector stopped")
            except Exception as e:
                logger.warning(f"Error stopping attack detector: {e}")

        # Stop advanced memory if running
        if self._advanced_memory and self._advanced_memory_enabled:
            try:
                self._advanced_memory.stop()
                logger.info("Advanced memory stopped")
            except Exception as e:
                logger.warning(f"Error stopping advanced memory: {e}")

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

    # =========================================================================
    # Attack Detection Methods
    # =========================================================================

    def _on_attack_detected(self, attack: Any) -> None:
        """
        Callback when an attack is detected.

        This is called by the AttackDetector when a potential attack
        is identified. Triggers analysis, remediation, and potentially
        emergency responses.

        Args:
            attack: The detected AttackEvent
        """
        if not ATTACK_DETECTION_AVAILABLE:
            return

        self._smith_metrics.attacks_detected += 1

        logger.warning(
            f"Attack detected: {attack.attack_id} - "
            f"{attack.attack_type.name} ({attack.severity.name})"
        )

        # Log as security incident
        severity_mapping = {
            1: IncidentSeverity.LOW,  # AttackSeverity.LOW
            2: IncidentSeverity.MEDIUM,  # AttackSeverity.MEDIUM
            3: IncidentSeverity.HIGH,  # AttackSeverity.HIGH
            4: IncidentSeverity.CRITICAL,  # AttackSeverity.CRITICAL
            5: IncidentSeverity.CRITICAL,  # AttackSeverity.CATASTROPHIC
        }

        if self._emergency:
            self._emergency.log_incident(
                severity=severity_mapping.get(attack.severity.value, IncidentSeverity.MEDIUM),
                category=f"attack_detected_{attack.attack_type.name.lower()}",
                description=attack.description,
                triggered_by="attack_detector",
            )

        # Trigger emergency response for critical attacks
        if attack.severity.value >= 4:  # CRITICAL or CATASTROPHIC
            self._handle_critical_attack(attack)

        # Analyze and generate remediation
        if self._attack_analyzer and self._remediation_engine:
            self._analyze_and_remediate(attack)

        # Notify external callbacks
        for callback in self._on_attack_callbacks:
            try:
                callback(attack)
            except Exception as e:
                logger.error(f"Attack callback error: {e}")

    def _handle_critical_attack(self, attack: Any) -> None:
        """
        Handle a critical or catastrophic attack.

        Args:
            attack: The critical AttackEvent
        """
        logger.critical(f"CRITICAL ATTACK: {attack.attack_id}")

        self._smith_metrics.auto_lockdowns_triggered += 1

        # Trigger lockdown for catastrophic attacks
        if attack.severity.value >= 5:  # CATASTROPHIC
            self.trigger_lockdown(
                f"Catastrophic attack detected: {attack.attack_id}"
            )
        else:
            # Safe mode for critical attacks
            self.trigger_safe_mode(
                f"Critical attack detected: {attack.attack_id}"
            )

    def _analyze_and_remediate(self, attack: Any) -> None:
        """
        Analyze an attack and generate remediation recommendations.

        Args:
            attack: The AttackEvent to analyze
        """
        try:
            # Analyze the attack
            report = self._attack_analyzer.analyze(attack)

            if not report.findings:
                logger.info(f"No vulnerable code found for attack {attack.attack_id}")
                return

            # Generate remediation plan
            plan = self._remediation_engine.generate_remediation_plan(attack, report)

            if plan.patches:
                logger.info(
                    f"Generated {len(plan.patches)} patches for attack {attack.attack_id}"
                )

                # Create recommendation
                if self._recommendation_system:
                    recommendation = self._recommendation_system.create_recommendation(
                        attack, report, plan
                    )
                    self._smith_metrics.recommendations_generated += 1

                    logger.info(
                        f"Created recommendation {recommendation.recommendation_id} "
                        f"for attack {attack.attack_id}"
                    )

        except Exception as e:
            logger.error(f"Error analyzing/remediating attack {attack.attack_id}: {e}")

    def _on_recommendation_created(self, recommendation: Any) -> None:
        """
        Callback when a fix recommendation is created.

        Args:
            recommendation: The FixRecommendation
        """
        logger.info(
            f"New fix recommendation: {recommendation.recommendation_id} - "
            f"{recommendation.title}"
        )

    def register_attack_callback(self, callback: Callable[[Any], None]) -> None:
        """
        Register a callback for attack events.

        Args:
            callback: Function to call when attack detected
        """
        self._on_attack_callbacks.append(callback)

    def process_boundary_event(self, event: Dict[str, Any]) -> None:
        """
        Process an event from the boundary daemon.

        This allows the boundary daemon to send events to Smith
        for attack detection analysis.

        Args:
            event: Boundary daemon event data
        """
        if self._attack_detector and self._attack_detection_enabled:
            try:
                self._attack_detector.process_boundary_event(event)
            except Exception as e:
                logger.warning(f"Error processing boundary event: {e}")

    def process_tripwire_event(self, event: Dict[str, Any]) -> None:
        """
        Process a tripwire event from the boundary daemon.

        Args:
            event: Tripwire event data
        """
        if self._attack_detector and self._attack_detection_enabled:
            try:
                self._attack_detector.process_tripwire_event(event)
            except Exception as e:
                logger.warning(f"Error processing tripwire event: {e}")

    def get_detected_attacks(
        self,
        since: Optional[datetime] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        Get list of detected attacks.

        Args:
            since: Only return attacks after this time
            limit: Maximum number to return

        Returns:
            List of attack dictionaries
        """
        if not self._attack_detector or not self._attack_detection_enabled:
            return []

        try:
            attacks = self._attack_detector.get_attacks(since=since)
            return [a.to_dict() for a in attacks[:limit]]
        except Exception as e:
            logger.error(f"Error getting attacks: {e}")
            return []

    def get_pending_recommendations(self) -> List[Dict[str, Any]]:
        """
        Get pending fix recommendations awaiting review.

        Returns:
            List of recommendation dictionaries
        """
        if not self._recommendation_system:
            return []

        try:
            from .attack_detection import RecommendationStatus
            pending = self._recommendation_system.list_recommendations(
                status=RecommendationStatus.PENDING
            )
            return [r.to_dict() for r in pending]
        except Exception as e:
            logger.error(f"Error getting recommendations: {e}")
            return []

    def approve_recommendation(
        self,
        recommendation_id: str,
        approver: str,
        comments: str = "",
    ) -> bool:
        """
        Approve a fix recommendation.

        Args:
            recommendation_id: ID of recommendation to approve
            approver: Who is approving
            comments: Approval comments

        Returns:
            True if successful
        """
        if not self._recommendation_system:
            return False

        try:
            return self._recommendation_system.approve(
                recommendation_id,
                approver,
                comments,
            )
        except Exception as e:
            logger.error(f"Error approving recommendation: {e}")
            return False

    def get_attack_detection_status(self) -> Dict[str, Any]:
        """
        Get attack detection system status.

        Returns:
            Status dictionary
        """
        status = {
            "enabled": self._attack_detection_enabled,
            "available": ATTACK_DETECTION_AVAILABLE,
            "attacks_detected": self._smith_metrics.attacks_detected,
            "attacks_mitigated": self._smith_metrics.attacks_mitigated,
            "recommendations_generated": self._smith_metrics.recommendations_generated,
            "auto_lockdowns_triggered": self._smith_metrics.auto_lockdowns_triggered,
        }

        if self._attack_detector and self._attack_detection_enabled:
            try:
                detector_stats = self._attack_detector.get_stats()
                status["detector"] = detector_stats
            except Exception:
                status["detector"] = None

        return status

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

    # =========================================================================
    # Advanced Memory Methods
    # =========================================================================

    def ingest_siem_event(self, event_data: Dict[str, Any]) -> Optional[str]:
        """
        Ingest a security event from Boundary-SIEM.

        Args:
            event_data: SIEM event data dictionary

        Returns:
            Entry ID if successful, None otherwise
        """
        if not self._advanced_memory_enabled or not self._advanced_memory:
            logger.warning("Advanced memory not enabled, SIEM event not ingested")
            return None

        try:
            entry_id = self._advanced_memory.ingest_siem_event(event_data)
            self._smith_metrics.memory_events_stored += 1
            return entry_id
        except Exception as e:
            logger.error(f"Error ingesting SIEM event: {e}")
            return None

    def ingest_boundary_event(self, event_data: Dict[str, Any]) -> Optional[str]:
        """
        Ingest an event from Boundary-Daemon.

        Args:
            event_data: Boundary-Daemon event data

        Returns:
            Entry ID if successful, None otherwise
        """
        if not self._advanced_memory_enabled or not self._advanced_memory:
            return None

        if not ADVANCED_MEMORY_AVAILABLE:
            return None

        try:
            event = BoundaryEvent(
                event_id=event_data.get("id", ""),
                timestamp=datetime.fromisoformat(
                    event_data.get("timestamp", datetime.now().isoformat())
                ),
                event_type=event_data.get("type", "unknown"),
                source="boundary-daemon",
                current_mode=BoundaryMode[event_data.get("mode", "OPEN")],
                target_resource=event_data.get("target", ""),
                action_requested=event_data.get("action", ""),
                details=event_data.get("details", {}),
            )
            entry_id = self._advanced_memory.ingest_boundary_event(event)
            self._smith_metrics.memory_events_stored += 1
            return entry_id
        except Exception as e:
            logger.error(f"Error ingesting boundary event: {e}")
            return None

    def get_intelligence_summary(
        self,
        period: str = "last_24h",
    ) -> Optional[Dict[str, Any]]:
        """
        Generate an executive intelligence summary.

        Args:
            period: Time period ("last_24h", "last_7d", "last_30d")

        Returns:
            Summary dictionary or None if not available
        """
        if not self._advanced_memory_enabled or not self._advanced_memory:
            return None

        try:
            summary = self._advanced_memory.generate_summary(period=period)
            return summary.to_dict() if summary else None
        except Exception as e:
            logger.error(f"Error generating intelligence summary: {e}")
            return None

    def get_threat_clusters(
        self,
        min_level: str = "LOW",
    ) -> List[Dict[str, Any]]:
        """
        Get active threat clusters.

        Args:
            min_level: Minimum threat level ("LOW", "MEDIUM", "HIGH", "CRITICAL")

        Returns:
            List of threat cluster dictionaries
        """
        if not self._advanced_memory_enabled or not self._advanced_memory:
            return []

        if not ADVANCED_MEMORY_AVAILABLE:
            return []

        try:
            level = ThreatLevel[min_level]
            clusters = self._advanced_memory.get_threat_clusters(min_threat_level=level)
            return [c.to_dict() for c in clusters]
        except Exception as e:
            logger.error(f"Error getting threat clusters: {e}")
            return []

    def get_detected_patterns(
        self,
        min_severity: int = 1,
    ) -> List[Dict[str, Any]]:
        """
        Get detected security patterns.

        Args:
            min_severity: Minimum pattern severity (1-5)

        Returns:
            List of pattern dictionaries
        """
        if not self._advanced_memory_enabled or not self._advanced_memory:
            return []

        try:
            patterns = self._advanced_memory.get_patterns(min_severity=min_severity)
            return [p.to_dict() for p in patterns]
        except Exception as e:
            logger.error(f"Error getting patterns: {e}")
            return []

    def get_anomaly_score(self) -> Optional[Dict[str, Any]]:
        """
        Get current behavioral anomaly score.

        Returns:
            Anomaly score dictionary or None
        """
        if not self._advanced_memory_enabled or not self._advanced_memory:
            return None

        try:
            score = self._advanced_memory.get_anomaly_score()
            return score.to_dict() if score else None
        except Exception as e:
            logger.error(f"Error getting anomaly score: {e}")
            return None

    def get_memory_status(self) -> Dict[str, Any]:
        """
        Get advanced memory system status.

        Returns:
            Status dictionary
        """
        status = {
            "enabled": self._advanced_memory_enabled,
            "available": ADVANCED_MEMORY_AVAILABLE,
        }

        if self._advanced_memory and self._advanced_memory_enabled:
            try:
                mem_status = self._advanced_memory.get_status()
                status["memory"] = mem_status.to_dict()
                status["stats"] = self._advanced_memory.get_stats()
            except Exception as e:
                logger.error(f"Error getting memory status: {e}")
                status["error"] = str(e)

        return status

    def register_threat_callback(self, callback: Callable[[Any], None]) -> None:
        """Register a callback for threat cluster events."""
        self._on_threat_callbacks.append(callback)

    def register_anomaly_callback(self, callback: Callable[[Any], None]) -> None:
        """Register a callback for anomaly events."""
        self._on_anomaly_callbacks.append(callback)

    def register_pattern_callback(self, callback: Callable[[Any], None]) -> None:
        """Register a callback for pattern events."""
        self._on_pattern_callbacks.append(callback)

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
