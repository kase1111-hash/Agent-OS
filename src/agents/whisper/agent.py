"""
Agent OS Whisper (Orchestrator) Agent

The central routing agent that:
- Classifies user intent
- Routes requests to appropriate agents
- Coordinates multi-agent workflows
- Enforces constitutional boundaries via Smith
"""

import logging
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Set

from src.agents.interface import (
    AgentCapabilities,
    AgentInterface,
    AgentState,
    BaseAgent,
    CapabilityType,
    RequestValidationResult,
)
from src.messaging.models import (
    FlowRequest,
    FlowResponse,
    MessageStatus,
)

from .aggregator import ResponseAggregator
from .context import ContextMinimizer
from .flow import AgentResult, FlowController, FlowResult
from .intent import IntentCategory, IntentClassification, IntentClassifier
from .router import RoutingAuditor, RoutingDecision, RoutingEngine
from .smith import SmithIntegration, SmithValidation

logger = logging.getLogger(__name__)


@dataclass
class WhisperMetrics:
    """Metrics for Whisper orchestrator."""

    requests_processed: int = 0
    requests_routed: int = 0
    requests_denied: int = 0
    requests_escalated: int = 0
    average_routing_time_ms: float = 0.0
    intent_distribution: Dict[str, int] = None

    def __post_init__(self):
        if self.intent_distribution is None:
            self.intent_distribution = {}


class WhisperAgent(BaseAgent):
    """
    Whisper - The Orchestrator Agent.

    Central hub for request routing and multi-agent coordination.
    All user requests flow through Whisper for intent classification
    and routing to appropriate specialist agents.
    """

    def __init__(
        self,
        name: str = "whisper",
        model: Optional[str] = "mistral:7b",
    ):
        """
        Initialize Whisper agent.

        Args:
            name: Agent name
            model: LLM model for intent classification
        """
        super().__init__(
            name=name,
            description="Orchestrator agent for intent classification and request routing",
            version="1.0.0",
            capabilities={CapabilityType.ROUTING},
            supported_intents=["*"],  # Handles all intents
        )

        self.model = model

        # Core components (initialized in initialize())
        self._classifier: Optional[IntentClassifier] = None
        self._router: Optional[RoutingEngine] = None
        self._flow_controller: Optional[FlowController] = None
        self._context_minimizer: Optional[ContextMinimizer] = None
        self._smith: Optional[SmithIntegration] = None
        self._aggregator: Optional[ResponseAggregator] = None
        self._auditor: Optional[RoutingAuditor] = None

        # Agent registry (for routing)
        self._agent_invokers: Dict[str, Callable] = {}
        self._available_agents: Set[str] = set()

        # Whisper-specific metrics
        self._whisper_metrics = WhisperMetrics()

        # Classification cache to prevent double-classification security bypass
        # Maps request_id to IntentClassification to ensure consistent
        # classification between validate_request() and process()
        self._classification_cache: Dict[str, IntentClassification] = {}

    def initialize(self, config: Dict[str, Any]) -> bool:
        """
        Initialize Whisper with configuration.

        Args:
            config: Configuration including:
                - use_llm_classifier: Whether to use LLM for classification
                - available_agents: Set of available agent names
                - routing_table: Custom routing table
                - strict_mode: Strict Smith validation

        Returns:
            True if initialization successful
        """
        self._do_initialize(config)

        try:
            # Initialize intent classifier
            use_llm = config.get("use_llm_classifier", False)
            self._classifier = IntentClassifier(use_llm=use_llm)

            # Initialize routing engine
            available = config.get("available_agents", set())
            routing_table = config.get("routing_table")
            self._router = RoutingEngine(
                routing_table=routing_table,
                available_agents=available,
            )
            self._available_agents = available

            # Initialize flow controller
            max_workers = config.get("max_workers", 4)
            self._flow_controller = FlowController(max_workers=max_workers)

            # Initialize context minimizer
            context_budget = config.get("context_budget", 4096)
            self._context_minimizer = ContextMinimizer(default_budget=context_budget)

            # Initialize Smith integration
            strict_mode = config.get("strict_mode", True)
            self._smith = SmithIntegration(strict_mode=strict_mode)

            # Initialize response aggregator
            self._aggregator = ResponseAggregator()

            # Initialize auditor
            self._auditor = RoutingAuditor()

            self._state = AgentState.READY
            logger.info(f"Whisper initialized with {len(available)} available agents")
            return True

        except Exception as e:
            logger.error(f"Whisper initialization failed: {e}")
            self._state = AgentState.ERROR
            return False

    def validate_request(self, request: FlowRequest) -> RequestValidationResult:
        """
        Validate request with Smith integration.

        Args:
            request: Incoming request

        Returns:
            RequestValidationResult
        """
        result = RequestValidationResult(is_valid=True)

        # Classify intent first
        classification = self._classifier.classify(
            request.content.prompt,
            context=[msg for msg in request.content.context],
        )

        # SECURITY FIX: Cache classification to prevent double-classification bypass
        # This ensures the same classification is used in both validate_request and process
        request_id = str(request.request_id)
        self._classification_cache[request_id] = classification

        # Limit cache size to prevent memory leak
        if len(self._classification_cache) > 1000:
            # Remove oldest entries (first 100)
            keys_to_remove = list(self._classification_cache.keys())[:100]
            for key in keys_to_remove:
                self._classification_cache.pop(key, None)

        # Pre-validate with Smith
        smith_result = self._smith.pre_validate(request, classification)

        if not smith_result.approved:
            for violation in smith_result.violations:
                result.add_error(violation)

        if smith_result.requires_human_approval:
            result.requires_escalation = True
            result.escalation_reason = smith_result.approval_reason

        # Add warnings
        for warning in smith_result.warnings:
            result.add_warning(warning)

        return result

    def process(self, request: FlowRequest) -> FlowResponse:
        """
        Process a request through the orchestration pipeline.

        Pipeline:
        1. Get cached intent classification (from validate_request)
        2. Route to appropriate agent(s)
        3. Minimize context
        4. Execute flow
        5. Post-validate with Smith
        6. Aggregate responses

        Args:
            request: Validated request

        Returns:
            FlowResponse with aggregated result
        """
        start_time = time.time()
        self._whisper_metrics.requests_processed += 1

        # Step 1: Get cached classification to prevent security bypass
        # SECURITY: Reuse the same classification from validate_request to ensure
        # what was validated is what gets executed (prevents double-classification attack)
        request_id = str(request.request_id)
        classification = self._classification_cache.pop(request_id, None)

        if classification is None:
            # Fallback: classify if not cached (e.g., direct process call without validate)
            # Log this as it may indicate a security concern
            logger.warning(
                f"No cached classification for request {request_id}, "
                "performing new classification. This may indicate a security issue."
            )
            classification = self._classifier.classify(
                request.content.prompt,
                context=[msg for msg in request.content.context],
            )

        # Track intent distribution
        intent_key = classification.primary_intent.value
        self._whisper_metrics.intent_distribution[intent_key] = (
            self._whisper_metrics.intent_distribution.get(intent_key, 0) + 1
        )

        # Step 2: Route to agents
        routing = self._router.route(
            classification,
            request_metadata={"request_id": str(request.request_id)},
        )

        # Log routing decision
        self._auditor.log(str(request.request_id), routing)

        # Handle system.meta locally — but still validate with Smith first
        if classification.primary_intent == IntentCategory.SYSTEM_META:
            if routing.requires_smith:
                smith_check = self._smith.pre_validate(request, classification)
                if not smith_check.approved:
                    return self._smith.handle_denial(request, smith_check)
            return self._handle_meta_request(request, classification)

        # Step 3: Pre-validate with Smith (already done in validate_request)
        # Apply any constraints
        if routing.requires_smith:
            smith_check = self._smith.pre_validate(request, classification)
            if not smith_check.approved:
                return self._smith.handle_denial(request, smith_check)
            if smith_check.requires_human_approval:
                return self._smith.handle_escalation(request, smith_check)
            request = self._smith.apply_constraints(request, smith_check)

        # Step 4: Minimize context
        minimized = self._context_minimizer.minimize(
            request.content.context,
            request.content.prompt,
            budget=routing.context_budget,
            intent=classification.primary_intent.value,
        )

        # Step 5: Execute flow
        flow_result = self._flow_controller.execute(
            request_id=str(request.request_id),
            decision=routing,
            request=request,
            invoker=self._invoke_agent,
            context={"minimized_context": minimized},
        )

        self._whisper_metrics.requests_routed += 1

        # Step 6: Aggregate responses
        response = self._aggregator.aggregate(flow_result, request)

        # Step 7: Post-validate response with Smith
        if routing.requires_smith and self._smith:
            post_check = self._smith.post_validate(request, response)
            if not post_check.approved:
                logger.warning(
                    f"Smith post-validation denied response for request {request_id}: "
                    f"{post_check.denial_reason}"
                )
                return self._smith.handle_denial(request, post_check)

        # Update metrics
        elapsed_ms = int((time.time() - start_time) * 1000)
        self._update_routing_metrics(elapsed_ms)

        return response

    def get_capabilities(self) -> AgentCapabilities:
        """Get Whisper capabilities."""
        return AgentCapabilities(
            name=self.name,
            version=self._version,
            description=self._description,
            capabilities={CapabilityType.ROUTING},
            supported_intents=["*"],
            model=self.model,
            context_window=4096,
            requires_constitution=True,
            can_escalate=True,
            metadata={
                "available_agents": list(self._available_agents),
                "intent_categories": [c.value for c in IntentCategory],
            },
        )

    def shutdown(self) -> bool:
        """Shutdown Whisper and release resources."""
        if self._flow_controller:
            self._flow_controller.shutdown()
        return self._do_shutdown()

    # =========================================================================
    # Agent Management
    # =========================================================================

    def register_agent(
        self,
        agent_name: str,
        invoker: Callable[[str, Any, Dict[str, Any]], Any],
    ) -> None:
        """
        Register an agent for routing.

        Args:
            agent_name: Agent name
            invoker: Function to invoke the agent
        """
        self._agent_invokers[agent_name] = invoker
        self._available_agents.add(agent_name)
        self._router.set_agent_availability(agent_name, True)
        logger.debug(f"Registered agent: {agent_name}")

    def unregister_agent(self, agent_name: str) -> None:
        """
        Unregister an agent.

        Args:
            agent_name: Agent name
        """
        self._agent_invokers.pop(agent_name, None)
        self._available_agents.discard(agent_name)
        self._router.set_agent_availability(agent_name, False)
        logger.debug(f"Unregistered agent: {agent_name}")

    # =========================================================================
    # Internal Methods
    # =========================================================================

    def _invoke_agent(
        self,
        agent_name: str,
        request: Any,
        context: Dict[str, Any],
    ) -> Any:
        """Invoke a registered agent."""
        invoker = self._agent_invokers.get(agent_name)
        if not invoker:
            raise ValueError(f"Agent not registered: {agent_name}")

        return invoker(agent_name, request, context)

    def _handle_meta_request(
        self,
        request: FlowRequest,
        classification: IntentClassification,
    ) -> FlowResponse:
        """Handle system.meta requests locally."""
        prompt_lower = request.content.prompt.lower()

        # Status request
        if "status" in prompt_lower:
            output = self._get_status()
        # Help request
        elif "help" in prompt_lower:
            output = self._get_help()
        # Capabilities request
        elif "capabilit" in prompt_lower or "can you" in prompt_lower:
            output = self._get_capabilities_info()
        # Agent list request
        elif "agent" in prompt_lower or "list" in prompt_lower:
            output = self._get_agent_list()
        else:
            output = self._get_help()

        return request.create_response(
            source=self.name,
            status=MessageStatus.SUCCESS,
            output=output,
        )

    def _get_status(self) -> str:
        """Get system status."""
        return f"""**Agent OS Status**

- Orchestrator: {self.name} ({self._state.name})
- Available Agents: {len(self._available_agents)}
- Requests Processed: {self._whisper_metrics.requests_processed}
- Requests Routed: {self._whisper_metrics.requests_routed}
- Average Routing Time: {self._whisper_metrics.average_routing_time_ms:.1f}ms
"""

    def _get_help(self) -> str:
        """Get help information."""
        return """**Agent OS Help**

I am Whisper, the orchestrator agent. I route your requests to the appropriate specialist agents.

**Available Commands:**
- Ask questions → Routed to Sage (reasoning)
- Request creative content → Routed to Muse (creative)
- Technical/code requests → Routed to Sage or Quill
- Memory operations → Routed to Seshat
- "status" → Show system status
- "help" → Show this help

**Intent Categories:**
- query.factual - Factual information
- query.reasoning - Complex analysis
- content.creative - Creative writing
- content.technical - Code/technical content
- memory.recall - Retrieve memories
- memory.store - Save memories
- system.meta - System operations
"""

    def _get_capabilities_info(self) -> str:
        """Get capabilities information."""
        agents = "\n".join([f"- {a}" for a in sorted(self._available_agents)])
        return f"""**Agent OS Capabilities**

**Available Agents:**
{agents}

**Routing:**
All requests are classified by intent and routed to the most appropriate agent.
Security-sensitive requests are validated by Smith (Guardian) before processing.
"""

    def _get_agent_list(self) -> str:
        """Get list of available agents."""
        if not self._available_agents:
            return "No agents currently registered."

        agents = []
        for name in sorted(self._available_agents):
            intents = self._router.get_routes_for_agent(name)
            intent_str = ", ".join([i.value for i in intents]) or "none"
            agents.append(f"- **{name}**: {intent_str}")

        return "**Registered Agents:**\n" + "\n".join(agents)

    def _update_routing_metrics(self, elapsed_ms: int) -> None:
        """Update routing metrics."""
        count = self._whisper_metrics.requests_routed
        if count > 0:
            # Running average
            prev_avg = self._whisper_metrics.average_routing_time_ms
            self._whisper_metrics.average_routing_time_ms = (
                prev_avg * (count - 1) + elapsed_ms
            ) / count

    def get_whisper_metrics(self) -> WhisperMetrics:
        """Get Whisper-specific metrics."""
        return self._whisper_metrics

    def get_audit_log(
        self,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Get routing audit log."""
        if not self._auditor:
            return []

        entries = self._auditor.get_entries(limit=limit)
        return [
            {
                "timestamp": e.timestamp.isoformat(),
                "request_id": e.request_id,
                "intent": e.intent.value,
                "confidence": e.confidence,
                "routes": e.routes,
                "strategy": e.strategy.name,
            }
            for e in entries
        ]


def create_whisper(
    available_agents: Optional[Set[str]] = None,
    config: Optional[Dict[str, Any]] = None,
) -> WhisperAgent:
    """
    Create and initialize a Whisper agent.

    Args:
        available_agents: Set of available agent names
        config: Additional configuration

    Returns:
        Initialized WhisperAgent
    """
    whisper = WhisperAgent()

    full_config = {
        "available_agents": available_agents or set(),
        **(config or {}),
    }

    whisper.initialize(full_config)
    return whisper
