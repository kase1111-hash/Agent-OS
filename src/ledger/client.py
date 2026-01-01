"""
Agent OS Value Ledger Client

Client for connecting to the value-ledger module.
Provides local or remote ledger access.
"""

import logging
import sys
import threading
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .models import (
    DEFAULT_MAPPINGS,
    IntentCategory,
    IntentValueMapping,
    ValueDimension,
    ValueEvent,
    get_mapping_for_intent,
)

logger = logging.getLogger(__name__)


@dataclass
class LedgerConfig:
    """Configuration for ledger client."""

    # Path to value-ledger module (for local mode)
    ledger_module_path: Optional[Path] = None
    # Path to SQLite database
    db_path: Optional[Path] = None
    # Use in-memory database (for testing)
    use_memory: bool = False
    # Custom intent → value mappings
    custom_mappings: List[IntentValueMapping] = field(default_factory=list)
    # Enable/disable value recording
    enabled: bool = True
    # Batch size for bulk operations
    batch_size: int = 100


@dataclass
class LedgerEntry:
    """Represents a ledger entry (mirrors value-ledger model)."""

    entry_id: str
    entry_type: str
    created_at: datetime
    dimension: str
    amount: float
    source: str
    intent_type: Optional[str]
    entry_hash: str
    sequence: int


@dataclass
class LedgerStats:
    """Statistics from the ledger."""

    total_entries: int
    total_value_by_dimension: Dict[str, float]
    entry_counts_by_type: Dict[str, int]
    latest_sequence: int


class LedgerClient:
    """
    Client for value-ledger integration.

    Supports both:
    1. Direct integration (value-ledger as Python module)
    2. Standalone mode (embedded implementation)
    """

    def __init__(self, config: Optional[LedgerConfig] = None):
        """
        Initialize ledger client.

        Args:
            config: Client configuration
        """
        self._config = config or LedgerConfig()
        self._initialized = False
        self._lock = threading.RLock()

        # Value-ledger module references
        self._store = None
        self._accrual_engine = None
        self._accrual_hook = None
        self._aggregation_engine = None

        # Mappings
        self._mappings = DEFAULT_MAPPINGS + self._config.custom_mappings

        # Metrics
        self._events_recorded = 0
        self._total_value = 0.0

    def initialize(self) -> bool:
        """
        Initialize the ledger client.

        Returns:
            True if initialization successful
        """
        if self._initialized:
            return True

        try:
            # Try to import value-ledger module
            if self._config.ledger_module_path:
                sys.path.insert(0, str(self._config.ledger_module_path.parent))

            try:
                from ledger import (
                    AccrualEngine,
                    AggregationEngine,
                    LedgerStore,
                    create_ledger_store,
                )

                # Create store
                if self._config.use_memory:
                    self._store = create_ledger_store(None)
                else:
                    db_path = self._config.db_path or Path("./data/ledger/entries.db")
                    self._store = create_ledger_store(db_path)

                # Create engines
                self._accrual_engine = AccrualEngine(self._store)
                self._accrual_hook = self._accrual_engine.create_hook()
                self._aggregation_engine = AggregationEngine(self._store)

                logger.info("Initialized with value-ledger module")

            except ImportError:
                # Use embedded implementation
                logger.warning("value-ledger module not found, using embedded implementation")
                self._init_embedded()

            self._initialized = True
            return True

        except Exception as e:
            logger.error(f"Failed to initialize ledger client: {e}")
            return False

    def record_event(self, event: ValueEvent) -> Optional[str]:
        """
        Record a value event to the ledger.

        Args:
            event: Value event to record

        Returns:
            Entry ID if recorded, None if disabled/failed
        """
        if not self._config.enabled:
            return None

        if not self._initialized:
            self.initialize()

        with self._lock:
            try:
                # Get mapping for intent
                mapping = get_mapping_for_intent(event.intent_type, self._mappings)
                if not mapping:
                    logger.debug(f"No mapping for intent: {event.intent_type}")
                    return None

                # Calculate value
                value = mapping.calculate_value(
                    context_tokens=event.context_tokens,
                    output_tokens=event.output_tokens,
                    processing_time_ms=event.processing_time_ms,
                )

                # Record via hook or embedded
                if self._accrual_hook:
                    entry = self._accrual_hook.record_intent(
                        source=event.source,
                        intent_type=event.intent_type,
                        context_tokens=event.context_tokens + event.output_tokens,
                        processing_time_ms=event.processing_time_ms,
                        session_id=event.session_id,
                        correlation_id=event.correlation_id,
                        **event.metadata,
                    )
                    if entry:
                        self._events_recorded += 1
                        self._total_value += value
                        return entry.entry_id
                else:
                    # Embedded recording
                    entry_id = self._record_embedded(event, value)
                    if entry_id:
                        self._events_recorded += 1
                        self._total_value += value
                    return entry_id

            except Exception as e:
                logger.error(f"Failed to record event: {e}")
                return None

        return None

    def record_intent(
        self,
        source: str,
        intent_type: str,
        context_tokens: int = 0,
        output_tokens: int = 0,
        processing_time_ms: float = 0.0,
        session_id: Optional[str] = None,
        **metadata,
    ) -> Optional[str]:
        """
        Convenience method to record an intent event.

        Args:
            source: Source agent
            intent_type: Intent type
            context_tokens: Input tokens
            output_tokens: Output tokens
            processing_time_ms: Processing time
            session_id: Session reference
            **metadata: Additional metadata

        Returns:
            Entry ID if recorded
        """
        import secrets

        # Determine category from intent
        category = self._categorize_intent(intent_type)

        # Get dimension from mapping
        mapping = get_mapping_for_intent(intent_type, self._mappings)
        dimension = mapping.dimension if mapping else ValueDimension.EFFORT

        event = ValueEvent(
            event_id=f"evt_{secrets.token_hex(8)}",
            source=source,
            dimension=dimension,
            intent_category=category,
            intent_type=intent_type,
            context_tokens=context_tokens,
            output_tokens=output_tokens,
            processing_time_ms=processing_time_ms,
            session_id=session_id,
            metadata=metadata,
        )

        return self.record_event(event)

    def get_total_value(
        self,
        dimension: Optional[ValueDimension] = None,
        source: Optional[str] = None,
    ) -> float:
        """
        Get total value recorded.

        Args:
            dimension: Filter by dimension
            source: Filter by source

        Returns:
            Total value amount
        """
        if self._aggregation_engine:
            try:
                from ledger import AggregationQuery
                from ledger import ValueDimension as LedgerDimension

                query = AggregationQuery(
                    dimension=LedgerDimension(dimension.value) if dimension else None,
                    source=source,
                )
                result = self._aggregation_engine.aggregate(query)
                return result.total_value
            except Exception as e:
                logger.debug(f"Aggregation engine query failed, using fallback: {e}")

        # Fallback to local tracking
        return self._total_value

    def get_stats(self) -> LedgerStats:
        """Get ledger statistics."""
        if self._store:
            try:
                stats = self._store.get_stats()
                return LedgerStats(
                    total_entries=stats.total_entries,
                    total_value_by_dimension=stats.total_value_by_dimension,
                    entry_counts_by_type=stats.entry_counts_by_type,
                    latest_sequence=stats.latest_sequence,
                )
            except Exception as e:
                logger.debug(f"Store stats query failed, using fallback: {e}")

        # Fallback
        return LedgerStats(
            total_entries=self._events_recorded,
            total_value_by_dimension={},
            entry_counts_by_type={},
            latest_sequence=0,
        )

    def verify_chain(self) -> Tuple[bool, List[str]]:
        """
        Verify the ledger chain integrity.

        Returns:
            Tuple of (is_valid, list of issues)
        """
        if self._store:
            try:
                return self._store.verify_chain()
            except Exception as e:
                return False, [str(e)]

        return True, []

    def get_metrics(self) -> Dict[str, Any]:
        """Get client metrics."""
        return {
            "initialized": self._initialized,
            "enabled": self._config.enabled,
            "events_recorded": self._events_recorded,
            "total_value": self._total_value,
            "mapping_count": len(self._mappings),
        }

    def shutdown(self) -> None:
        """Shutdown the client."""
        if self._store:
            try:
                self._store.shutdown()
            except Exception as e:
                logger.warning(f"Error during ledger store shutdown: {e}")

        self._initialized = False
        logger.info("Ledger client shutdown")

    def _init_embedded(self) -> None:
        """Initialize embedded implementation."""
        # Simple in-memory tracking for when value-ledger isn't available
        self._embedded_entries: List[Dict[str, Any]] = []
        logger.info("Initialized embedded ledger tracking")

    def _record_embedded(
        self,
        event: ValueEvent,
        value: float,
    ) -> Optional[str]:
        """Record event using embedded implementation."""
        import secrets

        entry_id = f"emb_{secrets.token_hex(16)}"

        self._embedded_entries.append(
            {
                "entry_id": entry_id,
                "event": event.to_dict(),
                "value": value,
                "timestamp": datetime.now().isoformat(),
            }
        )

        return entry_id

    def _categorize_intent(self, intent_type: str) -> IntentCategory:
        """Categorize an intent type."""
        lower = intent_type.lower()

        if lower.startswith("query"):
            return IntentCategory.QUERY
        elif lower.startswith("memory"):
            return IntentCategory.MEMORY
        elif lower.startswith("content") or lower.startswith("generat"):
            return IntentCategory.GENERATION
        elif lower.startswith("reason"):
            return IntentCategory.REASONING
        elif lower.startswith("creative"):
            return IntentCategory.CREATIVE
        elif lower.startswith("system"):
            return IntentCategory.SYSTEM
        elif lower.startswith("meta"):
            return IntentCategory.META
        else:
            return IntentCategory.QUERY


def create_ledger_client(
    db_path: Optional[Path] = None,
    use_memory: bool = False,
    enabled: bool = True,
) -> LedgerClient:
    """
    Create and initialize a ledger client.

    Args:
        db_path: Path to database
        use_memory: Use in-memory database
        enabled: Enable value recording

    Returns:
        Initialized LedgerClient
    """
    config = LedgerConfig(
        db_path=db_path,
        use_memory=use_memory,
        enabled=enabled,
    )

    client = LedgerClient(config)
    client.initialize()

    return client
