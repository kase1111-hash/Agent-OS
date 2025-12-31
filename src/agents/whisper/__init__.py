"""
Agent OS Whisper (Orchestrator) Module

Whisper is the central orchestrator agent responsible for:
- Intent classification (8 categories)
- Request routing with confidence scoring
- Context minimization
- Multi-agent flow coordination
- Smith integration for constitutional validation
- Response aggregation
- Load balancing across agents
"""

from .agent import (
    WhisperAgent,
    WhisperMetrics,
    create_whisper,
)
from .aggregator import (
    AggregatedResponse,
    AggregationStrategy,
    ResponseAggregator,
    ResponseFormatter,
)
from .context import (
    ContextItem,
    ContextMinimizer,
    ContextRelevance,
    MinimizedContext,
    minimize_context,
)
from .flow import (
    AgentResult,
    AsyncFlowController,
    FlowController,
    FlowResult,
    FlowStatus,
)
from .intent import (
    INTENT_PATTERNS,
    IntentCategory,
    IntentClassification,
    IntentClassifier,
    classify_intent,
)
from .load_balancer import (
    AgentHealthStatus,
    AgentLoadMetrics,
    AgentLoadTracker,
    HealthChecker,
    LoadBalancer,
    LoadBalancingStrategy,
    create_load_balancer,
)
from .router import (
    DEFAULT_ROUTING_TABLE,
    AgentRoute,
    RoutingAuditEntry,
    RoutingAuditor,
    RoutingDecision,
    RoutingEngine,
    RoutingStrategy,
)
from .smith import (
    SmithCheckType,
    SmithIntegration,
    SmithValidation,
)

__all__ = [
    # Intent Classification
    "IntentCategory",
    "IntentClassification",
    "IntentClassifier",
    "INTENT_PATTERNS",
    "classify_intent",
    # Routing
    "RoutingEngine",
    "RoutingDecision",
    "RoutingStrategy",
    "AgentRoute",
    "RoutingAuditor",
    "RoutingAuditEntry",
    "DEFAULT_ROUTING_TABLE",
    # Load Balancing
    "LoadBalancer",
    "LoadBalancingStrategy",
    "AgentLoadTracker",
    "AgentLoadMetrics",
    "AgentHealthStatus",
    "HealthChecker",
    "create_load_balancer",
    # Context
    "ContextMinimizer",
    "MinimizedContext",
    "ContextItem",
    "ContextRelevance",
    "minimize_context",
    # Flow Control
    "FlowController",
    "AsyncFlowController",
    "FlowResult",
    "FlowStatus",
    "AgentResult",
    # Smith Integration
    "SmithIntegration",
    "SmithValidation",
    "SmithCheckType",
    # Aggregation
    "ResponseAggregator",
    "AggregatedResponse",
    "AggregationStrategy",
    "ResponseFormatter",
    # Whisper Agent
    "WhisperAgent",
    "WhisperMetrics",
    "create_whisper",
]
