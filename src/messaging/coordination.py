"""
Collective Action Detection Monitor.

Detects when multiple agents converge on the same resource or action
within a sliding time window. When the threshold is reached, the monitor
raises a CoordinationAlert and logs a convergence event for human review.

Phase 6.5 of the Agentic Security Audit remediation.
"""

import logging
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ConvergenceEvent:
    """Record of multiple agents converging on a single target."""

    action_type: str
    target: str
    agents: List[str]
    first_seen: float
    last_seen: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action_type": self.action_type,
            "target": self.target,
            "agents": self.agents,
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
            "agent_count": len(self.agents),
        }


class CoordinationAlert(Exception):
    """Raised when collective action threshold is reached."""

    def __init__(self, event: ConvergenceEvent):
        self.event = event
        super().__init__(
            f"Coordination alert: {len(event.agents)} agents converged on "
            f"{event.action_type}:{event.target} — agents: {event.agents}"
        )


class CoordinationMonitor:
    """
    Detects when multiple agents converge on the same resource or action.

    Thread-safe. Designed to be wired into the message bus publish path.
    """

    def __init__(self, threshold: int = 3, window_seconds: int = 60):
        """
        Args:
            threshold: Number of distinct agents acting on the same target
                       within the window before alerting.
            window_seconds: Sliding window size in seconds.
        """
        self._threshold = threshold
        self._window = window_seconds
        self._lock = threading.Lock()

        # Key = (action_type, target) → list of (timestamp, agent_name)
        self._actions: Dict[tuple, List[tuple]] = defaultdict(list)

        # Historical convergence events for human review
        self._convergence_log: List[ConvergenceEvent] = []

    def record_action(
        self,
        agent_name: str,
        action_type: str,
        target: str,
    ) -> Optional[ConvergenceEvent]:
        """
        Record an agent action.

        Args:
            agent_name: Identifier of the acting agent
            action_type: Category of action (e.g. "publish", "file_write")
            target: The target resource (e.g. channel name, file path)

        Returns:
            ConvergenceEvent if threshold was reached, None otherwise

        Raises:
            CoordinationAlert if threshold is reached
        """
        now = time.monotonic()
        key = (action_type, target)

        with self._lock:
            entries = self._actions[key]

            # Evict expired entries
            cutoff = now - self._window
            entries[:] = [(ts, agent) for ts, agent in entries if ts >= cutoff]

            # Add this action
            entries.append((now, agent_name))

            # Count distinct agents in the window
            distinct_agents = list({agent for _, agent in entries})

            if len(distinct_agents) >= self._threshold:
                event = ConvergenceEvent(
                    action_type=action_type,
                    target=target,
                    agents=distinct_agents,
                    first_seen=entries[0][0],
                    last_seen=now,
                )
                self._convergence_log.append(event)

                # Clear the window to avoid repeated alerts on the same burst
                entries.clear()

                logger.warning(
                    "Coordination alert: %d agents (%s) converged on %s:%s "
                    "within %ds window",
                    len(distinct_agents),
                    ", ".join(distinct_agents),
                    action_type,
                    target,
                    self._window,
                )

                raise CoordinationAlert(event)

        return None

    def get_convergence_report(self) -> List[ConvergenceEvent]:
        """Return recent convergence events for human review."""
        with self._lock:
            return list(self._convergence_log)

    def clear_history(self) -> None:
        """Clear convergence history."""
        with self._lock:
            self._convergence_log.clear()
            self._actions.clear()
