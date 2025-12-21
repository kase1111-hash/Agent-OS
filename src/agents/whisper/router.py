"""
Agent OS Routing Engine

Routes requests to appropriate agents based on intent classification.
Manages routing tables, confidence-based routing, and fallback handling.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List, Dict, Any, Callable, Set
from enum import Enum, auto
import logging

from .intent import IntentCategory, IntentClassification


logger = logging.getLogger(__name__)


class RoutingStrategy(Enum):
    """Routing strategy types."""
    SINGLE = auto()      # Route to single best agent
    PARALLEL = auto()    # Route to multiple agents in parallel
    SEQUENTIAL = auto()  # Route through agents in sequence
    FALLBACK = auto()    # Try agents in order until success


@dataclass
class AgentRoute:
    """A route to a specific agent."""
    agent_name: str
    priority: int = 0              # Higher = more preferred
    confidence_threshold: float = 0.5  # Minimum confidence to use this route
    requires_smith: bool = True     # Must pass through Smith first
    max_tokens: Optional[int] = None  # Token limit for this agent
    timeout_ms: int = 30000        # Timeout for this agent
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RoutingDecision:
    """Result of routing decision."""
    routes: List[AgentRoute]           # Ordered list of agents to route to
    strategy: RoutingStrategy          # How to execute routes
    intent: IntentClassification       # The classified intent
    requires_smith: bool = True        # Must validate with Smith
    context_budget: int = 4096         # Token budget for context
    reasoning: str = ""                # Why this routing was chosen
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
    - Load balancing (future)
    """

    def __init__(
        self,
        routing_table: Optional[Dict[IntentCategory, List[AgentRoute]]] = None,
        available_agents: Optional[Set[str]] = None,
        default_agent: str = "sage",
    ):
        """
        Initialize routing engine.

        Args:
            routing_table: Custom routing table (uses default if not provided)
            available_agents: Set of currently available agent names
            default_agent: Fallback agent when no route matches
        """
        self.routing_table = routing_table or DEFAULT_ROUTING_TABLE.copy()
        self.available_agents = available_agents or set()
        self.default_agent = default_agent

        # Routing metrics
        self._routing_count = 0
        self._fallback_count = 0
        self._multi_agent_count = 0

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
    ) -> str:
        """Generate human-readable routing reasoning."""
        agents = [r.agent_name for r in routes]
        return (
            f"Routing {intent.value} (confidence: {confidence:.2f}) "
            f"to {', '.join(agents)}"
        )

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
            r for r in self.routing_table[intent]
            if r.agent_name != agent_name
        ]
        return len(self.routing_table[intent]) < original_count

    def set_agent_availability(
        self,
        agent_name: str,
        available: bool,
    ) -> None:
        """Update agent availability."""
        if available:
            self.available_agents.add(agent_name)
        else:
            self.available_agents.discard(agent_name)

    def get_routes_for_agent(self, agent_name: str) -> List[IntentCategory]:
        """Get all intents routed to an agent."""
        intents = []
        for intent, routes in self.routing_table.items():
            if any(r.agent_name == agent_name for r in routes):
                intents.append(intent)
        return intents

    def get_metrics(self) -> Dict[str, Any]:
        """Get routing metrics."""
        return {
            "total_routings": self._routing_count,
            "fallback_count": self._fallback_count,
            "fallback_rate": (
                self._fallback_count / self._routing_count
                if self._routing_count > 0
                else 0.0
            ),
            "multi_agent_count": self._multi_agent_count,
            "multi_agent_rate": (
                self._multi_agent_count / self._routing_count
                if self._routing_count > 0
                else 0.0
            ),
        }


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
            self._entries = self._entries[-self.max_entries:]

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
