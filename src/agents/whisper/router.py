"""
Agent OS Routing Engine

Routes requests to appropriate agents based on intent classification.
Manages routing tables, confidence-based routing, fallback handling,
and load balancing across available agents.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set

from .intent import IntentCategory, IntentClassification
from .load_balancer import (
    AgentHealthStatus,
    AgentLoadTracker,
    HealthChecker,
    LoadBalancer,
    LoadBalancingStrategy,
    create_load_balancer,
)

logger = logging.getLogger(__name__)


class RoutingStrategy(Enum):
    """Routing strategy types."""

    SINGLE = auto()  # Route to single best agent
    PARALLEL = auto()  # Route to multiple agents in parallel
    SEQUENTIAL = auto()  # Route through agents in sequence
    FALLBACK = auto()  # Try agents in order until success


@dataclass
class AgentRoute:
    """A route to a specific agent."""

    agent_name: str
    priority: int = 0  # Higher = more preferred
    confidence_threshold: float = 0.5  # Minimum confidence to use this route
    requires_smith: bool = True  # Must pass through Smith first
    max_tokens: Optional[int] = None  # Token limit for this agent
    timeout_ms: int = 30000  # Timeout for this agent
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RoutingDecision:
    """Result of routing decision."""

    routes: List[AgentRoute]  # Ordered list of agents to route to
    strategy: RoutingStrategy  # How to execute routes
    intent: IntentClassification  # The classified intent
    requires_smith: bool = True  # Must validate with Smith
    context_budget: int = 4096  # Token budget for context
    reasoning: str = ""  # Why this routing was chosen
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def primary_agent(self) -> Optional[str]:
        """Get primary agent name."""
        return self.routes[0].agent_name if self.routes else None

    @property
    def is_multi_agent(self) -> bool:
        """Check if routing involves multiple agents."""
        return len(self.routes) > 1


# Default routing table mapping intents to agents
DEFAULT_ROUTING_TABLE: Dict[IntentCategory, List[AgentRoute]] = {
    IntentCategory.QUERY_FACTUAL: [
        AgentRoute(agent_name="sage", priority=10, confidence_threshold=0.5),
    ],
    IntentCategory.QUERY_REASONING: [
        AgentRoute(agent_name="sage", priority=10, confidence_threshold=0.5),
    ],
    IntentCategory.CONTENT_CREATIVE: [
        AgentRoute(agent_name="muse", priority=10, confidence_threshold=0.5),
        AgentRoute(agent_name="quill", priority=5, confidence_threshold=0.3),
    ],
    IntentCategory.CONTENT_TECHNICAL: [
        AgentRoute(agent_name="sage", priority=10, confidence_threshold=0.6),
        AgentRoute(agent_name="quill", priority=5, confidence_threshold=0.3),
    ],
    IntentCategory.MEMORY_RECALL: [
        AgentRoute(agent_name="seshat", priority=10, confidence_threshold=0.5),
    ],
    IntentCategory.MEMORY_STORE: [
        AgentRoute(agent_name="seshat", priority=10, confidence_threshold=0.6),
    ],
    IntentCategory.SYSTEM_META: [
        AgentRoute(agent_name="whisper", priority=10, confidence_threshold=0.5),
    ],
    IntentCategory.SECURITY_SENSITIVE: [
        AgentRoute(agent_name="smith", priority=10, confidence_threshold=0.0, requires_smith=False),
    ],
    IntentCategory.UNKNOWN: [
        AgentRoute(agent_name="sage", priority=5, confidence_threshold=0.0),
    ],
}


class RoutingEngine:
    """
    Routes requests to appropriate agents based on intent and confidence.

    Features:
    - Configurable routing tables
    - Confidence-based agent selection
    - Multi-agent routing strategies
    - Fallback handling
    - Load balancing with multiple strategies
    - Health-aware routing
    - Circuit breaker pattern
    """

    def __init__(
        self,
        routing_table: Optional[Dict[IntentCategory, List[AgentRoute]]] = None,
        available_agents: Optional[Set[str]] = None,
        default_agent: str = "sage",
        load_balancing_strategy: LoadBalancingStrategy = LoadBalancingStrategy.ADAPTIVE,
        enable_load_balancing: bool = True,
        enable_health_checks: bool = True,
        health_check_interval: float = 30.0,
    ):
        """
        Initialize routing engine.

        Args:
            routing_table: Custom routing table (uses default if not provided)
            available_agents: Set of currently available agent names
            default_agent: Fallback agent when no route matches
            load_balancing_strategy: Strategy for load balancing
            enable_load_balancing: Whether to enable load balancing
            enable_health_checks: Whether to enable periodic health checks
            health_check_interval: Interval between health checks in seconds
        """
        self.routing_table = routing_table or DEFAULT_ROUTING_TABLE.copy()
        self.available_agents = available_agents or set()
        self.default_agent = default_agent

        # Routing metrics
        self._routing_count = 0
        self._fallback_count = 0
        self._multi_agent_count = 0

        # Load balancing setup
        self._enable_load_balancing = enable_load_balancing
        self._load_tracker: Optional[AgentLoadTracker] = None
        self._load_balancer: Optional[LoadBalancer] = None
        self._health_checker: Optional[HealthChecker] = None

        if enable_load_balancing:
            self._load_tracker, self._load_balancer, self._health_checker = create_load_balancer(
                strategy=load_balancing_strategy,
                health_check_interval=health_check_interval,
            )
            # Register available agents for load tracking
            for agent_name in self.available_agents:
                self._load_tracker.register_agent(agent_name)

            # Start health checks if enabled
            if enable_health_checks:
                self._health_checker.start()

    def route(
        self,
        classification: IntentClassification,
        request_metadata: Optional[Dict[str, Any]] = None,
    ) -> RoutingDecision:
        """
        Determine routing for a classified request.

        Args:
            classification: Intent classification result
            request_metadata: Optional request metadata

        Returns:
            RoutingDecision with target agents and strategy
        """
        self._routing_count += 1

        intent = classification.primary_intent
        confidence = classification.confidence

        # Get routes for this intent
        routes = self._get_routes_for_intent(intent, confidence)

        # Filter by available agents if we have that information
        if self.available_agents:
            routes = [r for r in routes if r.agent_name in self.available_agents]

        # Handle no routes
        if not routes:
            self._fallback_count += 1
            routes = [AgentRoute(agent_name=self.default_agent, priority=0)]
            strategy = RoutingStrategy.FALLBACK
            reasoning = f"No routes found for {intent.value}, using fallback"
        else:
            strategy = self._determine_strategy(routes, classification)

            # Apply load balancing to select best agent when multiple candidates
            if self._enable_load_balancing and len(routes) > 1:
                routes = self._apply_load_balancing(routes, intent)
                reasoning = self._generate_reasoning(routes, intent, confidence, load_balanced=True)
            else:
                reasoning = self._generate_reasoning(routes, intent, confidence)

        # Track multi-agent routing
        if len(routes) > 1:
            self._multi_agent_count += 1

        # Determine Smith requirement
        requires_smith = any(r.requires_smith for r in routes)
        if classification.requires_smith_review:
            requires_smith = True

        # Build decision
        return RoutingDecision(
            routes=routes,
            strategy=strategy,
            intent=classification,
            requires_smith=requires_smith,
            context_budget=self._calculate_context_budget(routes),
            reasoning=reasoning,
        )

    def _apply_load_balancing(
        self,
        routes: List[AgentRoute],
        intent: IntentCategory,
    ) -> List[AgentRoute]:
        """
        Apply load balancing to reorder routes based on agent load.

        Args:
            routes: Candidate routes
            intent: The intent category for round-robin state

        Returns:
            Routes reordered based on load balancing
        """
        if not self._load_balancer or len(routes) <= 1:
            return routes

        # Get candidate agent names
        candidates = [r.agent_name for r in routes]

        # Use load balancer to select best agent
        selected_agent = self._load_balancer.select_agent(
            candidates,
            intent_key=intent.value,
        )

        if not selected_agent:
            return routes

        # Reorder routes to put selected agent first
        reordered = []
        for route in routes:
            if route.agent_name == selected_agent:
                reordered.insert(0, route)
            else:
                reordered.append(route)

        return reordered

    def _get_routes_for_intent(
        self,
        intent: IntentCategory,
        confidence: float,
    ) -> List[AgentRoute]:
        """Get applicable routes for an intent."""
        routes = self.routing_table.get(intent, [])

        # Filter by confidence threshold
        applicable = [r for r in routes if confidence >= r.confidence_threshold]

        # Sort by priority (highest first)
        applicable.sort(key=lambda r: r.priority, reverse=True)

        return applicable

    def _determine_strategy(
        self,
        routes: List[AgentRoute],
        classification: IntentClassification,
    ) -> RoutingStrategy:
        """Determine routing strategy based on routes and intent."""
        if len(routes) == 1:
            return RoutingStrategy.SINGLE

        intent = classification.primary_intent

        # Creative content often benefits from parallel processing
        if intent == IntentCategory.CONTENT_CREATIVE:
            return RoutingStrategy.PARALLEL

        # Technical content may need sequential refinement
        if intent == IntentCategory.CONTENT_TECHNICAL:
            return RoutingStrategy.SEQUENTIAL

        # Default to single agent (primary)
        return RoutingStrategy.SINGLE

    def _calculate_context_budget(self, routes: List[AgentRoute]) -> int:
        """Calculate total context budget across routes."""
        # If any route specifies max_tokens, use the minimum
        token_limits = [r.max_tokens for r in routes if r.max_tokens]
        if token_limits:
            return min(token_limits)

        # Default budget
        return 4096

    def _generate_reasoning(
        self,
        routes: List[AgentRoute],
        intent: IntentCategory,
        confidence: float,
        load_balanced: bool = False,
    ) -> str:
        """Generate human-readable routing reasoning."""
        agents = [r.agent_name for r in routes]
        base_reason = (
            f"Routing {intent.value} (confidence: {confidence:.2f}) " f"to {', '.join(agents)}"
        )
        if load_balanced:
            base_reason += " (load-balanced)"
        return base_reason

    def add_route(
        self,
        intent: IntentCategory,
        route: AgentRoute,
    ) -> None:
        """Add a route to the routing table."""
        if intent not in self.routing_table:
            self.routing_table[intent] = []
        self.routing_table[intent].append(route)
        # Re-sort by priority
        self.routing_table[intent].sort(key=lambda r: r.priority, reverse=True)

    def remove_route(
        self,
        intent: IntentCategory,
        agent_name: str,
    ) -> bool:
        """Remove a route from the routing table."""
        if intent not in self.routing_table:
            return False

        original_count = len(self.routing_table[intent])
        self.routing_table[intent] = [
            r for r in self.routing_table[intent] if r.agent_name != agent_name
        ]
        return len(self.routing_table[intent]) < original_count

    def set_agent_availability(
        self,
        agent_name: str,
        available: bool,
    ) -> None:
        """Update agent availability and load balancing registration."""
        if available:
            self.available_agents.add(agent_name)
            # Register for load balancing if enabled
            if self._enable_load_balancing and self._load_tracker:
                self._load_tracker.register_agent(agent_name)
        else:
            self.available_agents.discard(agent_name)
            # Unregister from load balancing
            if self._enable_load_balancing and self._load_tracker:
                self._load_tracker.unregister_agent(agent_name)

    def get_routes_for_agent(self, agent_name: str) -> List[IntentCategory]:
        """Get all intents routed to an agent."""
        intents = []
        for intent, routes in self.routing_table.items():
            if any(r.agent_name == agent_name for r in routes):
                intents.append(intent)
        return intents

    def get_metrics(self) -> Dict[str, Any]:
        """Get routing metrics including load balancing data."""
        metrics = {
            "total_routings": self._routing_count,
            "fallback_count": self._fallback_count,
            "fallback_rate": (
                self._fallback_count / self._routing_count if self._routing_count > 0 else 0.0
            ),
            "multi_agent_count": self._multi_agent_count,
            "multi_agent_rate": (
                self._multi_agent_count / self._routing_count if self._routing_count > 0 else 0.0
            ),
            "load_balancing_enabled": self._enable_load_balancing,
        }

        # Add load balancing metrics if enabled
        if self._enable_load_balancing and self._load_tracker:
            agent_loads = {}
            for agent_name, agent_metrics in self._load_tracker.get_all_metrics().items():
                agent_loads[agent_name] = agent_metrics.to_dict()
            metrics["agent_load_metrics"] = agent_loads

        return metrics

    # =========================================================================
    # Load Balancing Methods
    # =========================================================================

    def record_request_start(self, agent_name: str) -> None:
        """
        Record that a request has started for an agent.

        Call this when routing a request to an agent.

        Args:
            agent_name: The agent receiving the request
        """
        if self._enable_load_balancing and self._load_tracker:
            self._load_tracker.request_start(agent_name)

    def record_request_complete(
        self,
        agent_name: str,
        response_time_ms: float,
        success: bool,
    ) -> None:
        """
        Record that a request has completed for an agent.

        Call this when an agent finishes processing a request.

        Args:
            agent_name: The agent that processed the request
            response_time_ms: How long the request took in milliseconds
            success: Whether the request succeeded
        """
        if self._enable_load_balancing and self._load_tracker:
            self._load_tracker.request_complete(agent_name, response_time_ms, success)

    def register_agent_for_load_balancing(
        self,
        agent_name: str,
        weight: int = 100,
        max_concurrent: int = 10,
    ) -> None:
        """
        Register an agent for load tracking.

        Args:
            agent_name: Agent name
            weight: Weight for weighted round-robin (1-100)
            max_concurrent: Maximum concurrent requests
        """
        if self._enable_load_balancing and self._load_tracker:
            self._load_tracker.register_agent(agent_name, weight, max_concurrent)

    def unregister_agent_from_load_balancing(self, agent_name: str) -> None:
        """Remove an agent from load tracking."""
        if self._enable_load_balancing and self._load_tracker:
            self._load_tracker.unregister_agent(agent_name)

    def set_agent_weight(self, agent_name: str, weight: int) -> None:
        """
        Set weight for an agent in weighted round-robin.

        Args:
            agent_name: Agent name
            weight: Weight (1-100, higher means more requests)
        """
        if self._enable_load_balancing and self._load_tracker:
            self._load_tracker.set_weight(agent_name, weight)

    def set_agent_max_concurrent(self, agent_name: str, max_concurrent: int) -> None:
        """
        Set maximum concurrent requests for an agent.

        Args:
            agent_name: Agent name
            max_concurrent: Maximum concurrent requests
        """
        if self._enable_load_balancing and self._load_tracker:
            self._load_tracker.set_max_concurrent(agent_name, max_concurrent)

    def get_load_distribution(self) -> Dict[str, Dict[str, Any]]:
        """
        Get current load distribution across all agents.

        Returns:
            Dictionary mapping agent names to their load metrics
        """
        if not self._enable_load_balancing or not self._load_balancer:
            return {}

        candidates = list(self.available_agents) if self.available_agents else []
        return self._load_balancer.get_load_distribution(candidates)

    def get_agent_health_status(self, agent_name: str) -> Optional[str]:
        """
        Get health status for an agent.

        Args:
            agent_name: Agent name

        Returns:
            Health status string or None if not tracked
        """
        if not self._enable_load_balancing or not self._load_tracker:
            return None

        metrics = self._load_tracker.get_metrics(agent_name)
        if metrics:
            return metrics.health_status.value
        return None

    def set_load_balancing_strategy(self, strategy: LoadBalancingStrategy) -> None:
        """
        Change the load balancing strategy at runtime.

        Args:
            strategy: New load balancing strategy
        """
        if self._enable_load_balancing and self._load_balancer:
            self._load_balancer.strategy = strategy
            logger.info(f"Load balancing strategy changed to: {strategy.name}")

    def shutdown(self) -> None:
        """Shutdown the routing engine and cleanup resources."""
        if self._health_checker:
            self._health_checker.stop()
        logger.info("Routing engine shutdown complete")


@dataclass
class RoutingAuditEntry:
    """Audit log entry for routing decisions."""

    timestamp: datetime
    request_id: str
    intent: IntentCategory
    confidence: float
    routes: List[str]
    strategy: RoutingStrategy
    requires_smith: bool
    reasoning: str


class RoutingAuditor:
    """Maintains audit trail for routing decisions."""

    def __init__(self, max_entries: int = 1000):
        self.max_entries = max_entries
        self._entries: List[RoutingAuditEntry] = []

    def log(
        self,
        request_id: str,
        decision: RoutingDecision,
    ) -> None:
        """Log a routing decision."""
        entry = RoutingAuditEntry(
            timestamp=datetime.now(),
            request_id=request_id,
            intent=decision.intent.primary_intent,
            confidence=decision.intent.confidence,
            routes=[r.agent_name for r in decision.routes],
            strategy=decision.strategy,
            requires_smith=decision.requires_smith,
            reasoning=decision.reasoning,
        )
        self._entries.append(entry)

        # Trim if over limit
        if len(self._entries) > self.max_entries:
            self._entries = self._entries[-self.max_entries :]

    def get_entries(
        self,
        since: Optional[datetime] = None,
        intent: Optional[IntentCategory] = None,
        limit: int = 100,
    ) -> List[RoutingAuditEntry]:
        """Get audit entries with optional filtering."""
        entries = self._entries

        if since:
            entries = [e for e in entries if e.timestamp >= since]

        if intent:
            entries = [e for e in entries if e.intent == intent]

        return entries[-limit:]

    def clear(self) -> int:
        """Clear audit log. Returns number of entries cleared."""
        count = len(self._entries)
        self._entries = []
        return count
