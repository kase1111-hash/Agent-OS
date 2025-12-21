"""
Agent OS Whisper (Orchestrator) Module

Whisper is the central orchestrator agent responsible for:
- Intent classification (8 categories)
- Request routing with confidence scoring
- Context minimization
- Multi-agent flow coordination
- Smith integration for constitutional validation
- Response aggregation
"""

from .intent import (
    IntentCategory,
    IntentClassification,
    IntentClassifier,
    INTENT_PATTERNS,
    classify_intent,
)
from .router import (
    RoutingEngine,
    RoutingDecision,
    RoutingStrategy,
    AgentRoute,
    RoutingAuditor,
    RoutingAuditEntry,
    DEFAULT_ROUTING_TABLE,
)
from .context import (
    ContextMinimizer,
    MinimizedContext,
    ContextItem,
    ContextRelevance,
    minimize_context,
)
from .flow import (
    FlowController,
    AsyncFlowController,
    FlowResult,
    FlowStatus,
    AgentResult,
)
from .smith import (
    SmithIntegration,
    SmithValidation,
    SmithCheckType,
)
from .aggregator import (
    ResponseAggregator,
    AggregatedResponse,
    AggregationStrategy,
    ResponseFormatter,
)
from .agent import (
    WhisperAgent,
    WhisperMetrics,
    create_whisper,
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
