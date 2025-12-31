"""
Agent OS Value Ledger Integration Models

Data models for value tracking integration.
"""

import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Dict, List, Optional


class ValueDimension(Enum):
    """Dimensions of value being tracked."""

    EFFORT = "effort"  # Computational effort
    KNOWLEDGE = "knowledge"  # Knowledge contribution
    INTERACTION = "interaction"  # User interaction
    CREATIVE = "creative"  # Creative contribution
    REASONING = "reasoning"  # Reasoning contribution
    MEMORY = "memory"  # Memory operations


class IntentCategory(Enum):
    """Categories of intents for value mapping."""

    QUERY = "query"
    GENERATION = "generation"
    MEMORY = "memory"
    REASONING = "reasoning"
    CREATIVE = "creative"
    SYSTEM = "system"
    META = "meta"


@dataclass
class ValueEvent:
    """
    Event representing value to be recorded.

    IMPORTANT: Contains metadata only, never content.
    """

    event_id: str
    source: str  # Source agent
    dimension: ValueDimension
    intent_category: IntentCategory
    intent_type: str  # Specific intent (e.g., "memory.store")
    timestamp: datetime = field(default_factory=datetime.now)
    session_id: Optional[str] = None
    correlation_id: Optional[str] = None
    context_tokens: int = 0  # Token count (not content)
    output_tokens: int = 0
    processing_time_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "source": self.source,
            "dimension": self.dimension.value,
            "intent_category": self.intent_category.value,
            "intent_type": self.intent_type,
            "timestamp": self.timestamp.isoformat(),
            "session_id": self.session_id,
            "correlation_id": self.correlation_id,
            "context_tokens": self.context_tokens,
            "output_tokens": self.output_tokens,
            "processing_time_ms": self.processing_time_ms,
            "metadata": self.metadata,
        }

    def get_intent_hash(self) -> str:
        """Get a hash of the intent (not content)."""
        data = f"{self.intent_type}:{self.session_id}:{self.timestamp.isoformat()}"
        return hashlib.sha256(data.encode()).hexdigest()[:16]


@dataclass
class IntentValueMapping:
    """
    Mapping from intent types to value dimensions.

    Defines how different intents contribute to value.
    """

    intent_pattern: str  # Pattern to match (e.g., "memory.*")
    dimension: ValueDimension
    base_value: float  # Base value amount (0-1)
    token_multiplier: float = 0.0001  # Value per token
    time_multiplier: float = 0.00001  # Value per ms
    enabled: bool = True

    def matches(self, intent_type: str) -> bool:
        """Check if this mapping matches an intent."""
        if self.intent_pattern.endswith("*"):
            prefix = self.intent_pattern[:-1]
            return intent_type.startswith(prefix)
        return intent_type == self.intent_pattern

    def calculate_value(
        self,
        context_tokens: int = 0,
        output_tokens: int = 0,
        processing_time_ms: float = 0.0,
    ) -> float:
        """Calculate value for given metrics."""
        if not self.enabled:
            return 0.0

        value = self.base_value
        value += (context_tokens + output_tokens) * self.token_multiplier
        value += processing_time_ms * self.time_multiplier

        return min(max(value, 0.0), 1.0)  # Clamp to 0-1


# Default intent → value mappings
DEFAULT_MAPPINGS: List[IntentValueMapping] = [
    # Query intents
    IntentValueMapping(
        intent_pattern="query.factual",
        dimension=ValueDimension.KNOWLEDGE,
        base_value=0.05,
        token_multiplier=0.00005,
    ),
    IntentValueMapping(
        intent_pattern="query.reasoning",
        dimension=ValueDimension.REASONING,
        base_value=0.1,
        token_multiplier=0.0001,
    ),
    # Memory intents
    IntentValueMapping(
        intent_pattern="memory.store",
        dimension=ValueDimension.MEMORY,
        base_value=0.05,
        token_multiplier=0.00005,
    ),
    IntentValueMapping(
        intent_pattern="memory.retrieve",
        dimension=ValueDimension.MEMORY,
        base_value=0.02,
        token_multiplier=0.00002,
    ),
    IntentValueMapping(
        intent_pattern="memory.*",
        dimension=ValueDimension.MEMORY,
        base_value=0.03,
    ),
    # Creative intents
    IntentValueMapping(
        intent_pattern="content.creative",
        dimension=ValueDimension.CREATIVE,
        base_value=0.15,
        token_multiplier=0.0002,
    ),
    IntentValueMapping(
        intent_pattern="content.technical",
        dimension=ValueDimension.EFFORT,
        base_value=0.1,
        token_multiplier=0.00015,
    ),
    # Reasoning intents
    IntentValueMapping(
        intent_pattern="reasoning.*",
        dimension=ValueDimension.REASONING,
        base_value=0.12,
        token_multiplier=0.00015,
        time_multiplier=0.00002,
    ),
    # System intents (low value)
    IntentValueMapping(
        intent_pattern="system.*",
        dimension=ValueDimension.EFFORT,
        base_value=0.01,
    ),
    IntentValueMapping(
        intent_pattern="meta.*",
        dimension=ValueDimension.INTERACTION,
        base_value=0.02,
    ),
    # Default fallback
    IntentValueMapping(
        intent_pattern="*",
        dimension=ValueDimension.EFFORT,
        base_value=0.05,
        token_multiplier=0.00005,
    ),
]


def get_mapping_for_intent(
    intent_type: str,
    mappings: Optional[List[IntentValueMapping]] = None,
) -> Optional[IntentValueMapping]:
    """
    Get the value mapping for an intent type.

    Args:
        intent_type: Intent type to look up
        mappings: Custom mappings (uses defaults if None)

    Returns:
        Matching IntentValueMapping or None
    """
    mappings = mappings or DEFAULT_MAPPINGS

    for mapping in mappings:
        if mapping.matches(intent_type):
            return mapping

    return None
