"""
Agent OS Value Ledger Integration

Provides integration with the external value-ledger module for
intent-based effort tracking.

Key Components:
- LedgerClient: Client for connecting to value-ledger
- IntentValueHook: Hook for recording intent → value
- AgentValueTracker: Per-agent value tracking
"""

from .client import (
    LedgerClient,
    LedgerConfig,
    create_ledger_client,
)

from .hooks import (
    IntentValueHook,
    AgentValueTracker,
    create_intent_hook,
)

from .models import (
    ValueEvent,
    ValueDimension,
    IntentValueMapping,
)

__all__ = [
    # Client
    "LedgerClient",
    "LedgerConfig",
    "create_ledger_client",
    # Hooks
    "IntentValueHook",
    "AgentValueTracker",
    "create_intent_hook",
    # Models
    "ValueEvent",
    "ValueDimension",
    "IntentValueMapping",
]
