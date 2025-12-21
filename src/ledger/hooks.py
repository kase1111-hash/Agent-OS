"""
Agent OS Value Ledger Hooks

Hooks for integrating value tracking with agents and messaging.
"""

import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, Any, List, Callable

from .client import LedgerClient, LedgerConfig
from .models import (
    ValueEvent,
    ValueDimension,
    IntentCategory,
    IntentValueMapping,
)

logger = logging.getLogger(__name__)


class IntentValueHook:
    """
    Hook for recording intent → value automatically.

    Can be attached to the message bus or agent handlers
    to automatically record value for processed intents.
    """

    def __init__(
        self,
        client: LedgerClient,
        enabled: bool = True,
    ):
        """
        Initialize intent value hook.

        Args:
            client: Ledger client for recording
            enabled: Whether hook is active
        """
        self._client = client
        self._enabled = enabled
        self._lock = threading.RLock()

        # Tracking
        self._intents_processed = 0
        self._value_recorded = 0.0

    @property
    def enabled(self) -> bool:
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        self._enabled = value

    def on_intent_processed(
        self,
        source: str,
        intent: str,
        context_tokens: int = 0,
        output_tokens: int = 0,
        processing_time_ms: float = 0.0,
        session_id: Optional[str] = None,
        correlation_id: Optional[str] = None,
        success: bool = True,
        **metadata,
    ) -> Optional[str]:
        """
        Called when an intent is processed.

        Args:
            source: Source agent that processed the intent
            intent: Intent type that was processed
            context_tokens: Input context tokens
            output_tokens: Output tokens generated
            processing_time_ms: Processing duration
            session_id: Session reference
            correlation_id: Correlation reference
            success: Whether processing succeeded
            **metadata: Additional metadata

        Returns:
            Entry ID if recorded
        """
        if not self._enabled:
            return None

        with self._lock:
            self._intents_processed += 1

            # Only record successful processing
            if not success:
                return None

            entry_id = self._client.record_intent(
                source=source,
                intent_type=intent,
                context_tokens=context_tokens,
                output_tokens=output_tokens,
                processing_time_ms=processing_time_ms,
                session_id=session_id,
                correlation_id=correlation_id,
                **metadata,
            )

            return entry_id

    def on_request_complete(
        self,
        request: Any,
        response: Any,
        processing_time_ms: float,
    ) -> Optional[str]:
        """
        Called when a request/response cycle completes.

        Extracts relevant info from request/response objects.

        Args:
            request: FlowRequest object
            response: FlowResponse object
            processing_time_ms: Processing duration

        Returns:
            Entry ID if recorded
        """
        if not self._enabled:
            return None

        try:
            # Extract from FlowRequest/FlowResponse
            source = getattr(response, "source", "unknown")
            intent = getattr(request, "intent", "unknown")
            session_id = None
            correlation_id = None

            # Try to get tokens from response metadata
            context_tokens = 0
            output_tokens = 0

            if hasattr(response, "content") and hasattr(response.content, "metadata"):
                meta = response.content.metadata
                if hasattr(meta, "tokens_consumed"):
                    context_tokens = meta.tokens_consumed or 0

            # Check if successful
            success = True
            if hasattr(response, "is_success"):
                success = response.is_success()
            elif hasattr(response, "status"):
                success = str(response.status).lower() == "success"

            return self.on_intent_processed(
                source=source,
                intent=intent,
                context_tokens=context_tokens,
                output_tokens=output_tokens,
                processing_time_ms=processing_time_ms,
                session_id=session_id,
                correlation_id=correlation_id,
                success=success,
            )

        except Exception as e:
            logger.warning(f"Failed to record request completion: {e}")
            return None

    def get_metrics(self) -> Dict[str, Any]:
        """Get hook metrics."""
        return {
            "enabled": self._enabled,
            "intents_processed": self._intents_processed,
            "client_metrics": self._client.get_metrics(),
        }


@dataclass
class AgentValueStats:
    """Value statistics for an agent."""
    agent_name: str
    total_value: float
    intent_count: int
    value_by_dimension: Dict[str, float] = field(default_factory=dict)
    last_recorded: Optional[datetime] = None


class AgentValueTracker:
    """
    Tracks value per agent.

    Provides per-agent statistics and value breakdown.
    """

    def __init__(self, client: LedgerClient):
        """
        Initialize agent value tracker.

        Args:
            client: Ledger client for recording
        """
        self._client = client
        self._lock = threading.RLock()

        # Per-agent tracking
        self._agent_stats: Dict[str, AgentValueStats] = {}

    def record_agent_activity(
        self,
        agent_name: str,
        intent: str,
        dimension: ValueDimension,
        value: float,
        context_tokens: int = 0,
        processing_time_ms: float = 0.0,
        session_id: Optional[str] = None,
        **metadata,
    ) -> Optional[str]:
        """
        Record activity for a specific agent.

        Args:
            agent_name: Name of the agent
            intent: Intent processed
            dimension: Value dimension
            value: Value amount
            context_tokens: Token count
            processing_time_ms: Processing time
            session_id: Session reference
            **metadata: Additional metadata

        Returns:
            Entry ID if recorded
        """
        with self._lock:
            # Update agent stats
            if agent_name not in self._agent_stats:
                self._agent_stats[agent_name] = AgentValueStats(
                    agent_name=agent_name,
                    total_value=0.0,
                    intent_count=0,
                )

            stats = self._agent_stats[agent_name]
            stats.total_value += value
            stats.intent_count += 1
            stats.last_recorded = datetime.now()

            dim_key = dimension.value
            stats.value_by_dimension[dim_key] = (
                stats.value_by_dimension.get(dim_key, 0.0) + value
            )

            # Record to ledger
            return self._client.record_intent(
                source=agent_name,
                intent_type=intent,
                context_tokens=context_tokens,
                processing_time_ms=processing_time_ms,
                session_id=session_id,
                dimension=dimension.value,
                value=value,
                **metadata,
            )

    def get_agent_stats(self, agent_name: str) -> Optional[AgentValueStats]:
        """Get stats for a specific agent."""
        return self._agent_stats.get(agent_name)

    def get_all_agent_stats(self) -> Dict[str, AgentValueStats]:
        """Get stats for all agents."""
        return dict(self._agent_stats)

    def get_leaderboard(self, limit: int = 10) -> List[AgentValueStats]:
        """
        Get agents sorted by total value.

        Args:
            limit: Maximum number to return

        Returns:
            List of AgentValueStats sorted by value
        """
        sorted_agents = sorted(
            self._agent_stats.values(),
            key=lambda s: s.total_value,
            reverse=True,
        )
        return sorted_agents[:limit]

    def get_metrics(self) -> Dict[str, Any]:
        """Get tracker metrics."""
        return {
            "tracked_agents": len(self._agent_stats),
            "total_value_all_agents": sum(
                s.total_value for s in self._agent_stats.values()
            ),
            "total_intents_all_agents": sum(
                s.intent_count for s in self._agent_stats.values()
            ),
        }


class MessageBusHook:
    """
    Hook for integrating with the Agent OS message bus.

    Automatically records value for messages processed through the bus.
    """

    def __init__(
        self,
        client: LedgerClient,
        enabled: bool = True,
    ):
        """
        Initialize message bus hook.

        Args:
            client: Ledger client
            enabled: Whether hook is active
        """
        self._client = client
        self._enabled = enabled
        self._intent_hook = IntentValueHook(client, enabled)

    def on_message_sent(self, message: Any) -> None:
        """Called when a message is sent."""
        # Could track message send events if needed
        pass

    def on_message_received(self, message: Any) -> None:
        """Called when a message is received."""
        # Could track message receive events if needed
        pass

    def on_request_handled(
        self,
        request: Any,
        response: Any,
        handler: str,
        processing_time_ms: float,
    ) -> Optional[str]:
        """
        Called when a request is handled.

        Args:
            request: FlowRequest
            response: FlowResponse
            handler: Handler name (agent)
            processing_time_ms: Processing time

        Returns:
            Entry ID if recorded
        """
        return self._intent_hook.on_request_complete(
            request=request,
            response=response,
            processing_time_ms=processing_time_ms,
        )


def create_intent_hook(
    db_path: Optional[str] = None,
    use_memory: bool = True,
    enabled: bool = True,
) -> IntentValueHook:
    """
    Create an intent value hook.

    Args:
        db_path: Path to ledger database
        use_memory: Use in-memory storage
        enabled: Enable recording

    Returns:
        Configured IntentValueHook
    """
    from pathlib import Path

    config = LedgerConfig(
        db_path=Path(db_path) if db_path else None,
        use_memory=use_memory,
        enabled=enabled,
    )

    client = LedgerClient(config)
    client.initialize()

    return IntentValueHook(client, enabled)
