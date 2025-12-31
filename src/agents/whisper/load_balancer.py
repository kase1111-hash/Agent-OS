"""
Agent OS Load Balancer

Provides load balancing strategies for distributing requests across agents.
Tracks agent load metrics, health status, and enables intelligent routing.
"""

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


class LoadBalancingStrategy(Enum):
    """Available load balancing strategies."""

    ROUND_ROBIN = auto()  # Simple round-robin rotation
    WEIGHTED_ROUND_ROBIN = auto()  # Weighted by agent capacity
    LEAST_CONNECTIONS = auto()  # Route to agent with fewest active requests
    LEAST_RESPONSE_TIME = auto()  # Route to fastest responding agent
    ADAPTIVE = auto()  # Combined: considers load, response time, and errors


class AgentHealthStatus(Enum):
    """Agent health status levels."""

    HEALTHY = "healthy"  # Agent operating normally
    DEGRADED = "degraded"  # Agent having issues but functional
    UNHEALTHY = "unhealthy"  # Agent failing health checks
    UNKNOWN = "unknown"  # No health data available


@dataclass
class AgentLoadSnapshot:
    """Point-in-time snapshot of agent load."""

    timestamp: datetime
    active_requests: int
    response_time_ms: float
    error_count: int
    success_count: int


@dataclass
class AgentLoadMetrics:
    """Real-time load metrics for an agent."""

    agent_name: str

    # Current state
    active_requests: int = 0

    # Response time tracking (sliding window)
    response_times_ms: deque = field(default_factory=lambda: deque(maxlen=100))

    # Request counts (reset periodically)
    requests_total: int = 0
    requests_succeeded: int = 0
    requests_failed: int = 0

    # Health tracking
    last_health_check: Optional[datetime] = None
    health_status: AgentHealthStatus = AgentHealthStatus.UNKNOWN
    consecutive_failures: int = 0
    last_failure_time: Optional[datetime] = None

    # Configuration
    weight: int = 100  # Weight for weighted round-robin (1-100)
    max_concurrent: int = 10  # Max concurrent requests

    # Circuit breaker
    circuit_open: bool = False
    circuit_open_until: Optional[datetime] = None

    def __post_init__(self):
        if not isinstance(self.response_times_ms, deque):
            self.response_times_ms = deque(maxlen=100)

    @property
    def average_response_time_ms(self) -> float:
        """Calculate average response time from recent requests."""
        if not self.response_times_ms:
            return 0.0
        return sum(self.response_times_ms) / len(self.response_times_ms)

    @property
    def error_rate(self) -> float:
        """Calculate error rate."""
        if self.requests_total == 0:
            return 0.0
        return self.requests_failed / self.requests_total

    @property
    def load_factor(self) -> float:
        """Calculate load factor (0.0 = idle, 1.0 = at capacity)."""
        if self.max_concurrent <= 0:
            return 1.0
        return min(1.0, self.active_requests / self.max_concurrent)

    @property
    def is_available(self) -> bool:
        """Check if agent is available to receive requests."""
        # Check circuit breaker
        if self.circuit_open:
            if self.circuit_open_until and datetime.now() > self.circuit_open_until:
                self.circuit_open = False
            else:
                return False

        # Check capacity
        if self.active_requests >= self.max_concurrent:
            return False

        # Check health
        if self.health_status == AgentHealthStatus.UNHEALTHY:
            return False

        return True

    def record_request_start(self) -> None:
        """Record that a request has started."""
        self.active_requests += 1
        self.requests_total += 1

    def record_request_complete(self, response_time_ms: float, success: bool) -> None:
        """Record that a request has completed."""
        self.active_requests = max(0, self.active_requests - 1)
        self.response_times_ms.append(response_time_ms)

        if success:
            self.requests_succeeded += 1
            self.consecutive_failures = 0
        else:
            self.requests_failed += 1
            self.consecutive_failures += 1
            self.last_failure_time = datetime.now()

            # Check if we need to open circuit breaker
            if self.consecutive_failures >= 5:
                self._open_circuit_breaker()

    def _open_circuit_breaker(self) -> None:
        """Open circuit breaker to stop routing to failing agent."""
        self.circuit_open = True
        # Start with 10 second timeout, could implement exponential backoff
        self.circuit_open_until = datetime.now() + timedelta(seconds=10)
        logger.warning(
            f"Circuit breaker opened for agent {self.agent_name} "
            f"after {self.consecutive_failures} consecutive failures"
        )

    def update_health_status(self, status: AgentHealthStatus) -> None:
        """Update agent health status."""
        old_status = self.health_status
        self.health_status = status
        self.last_health_check = datetime.now()

        if old_status != status:
            logger.info(f"Agent {self.agent_name} health: {old_status.value} -> {status.value}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "agent_name": self.agent_name,
            "active_requests": self.active_requests,
            "average_response_time_ms": self.average_response_time_ms,
            "requests_total": self.requests_total,
            "requests_succeeded": self.requests_succeeded,
            "requests_failed": self.requests_failed,
            "error_rate": self.error_rate,
            "load_factor": self.load_factor,
            "health_status": self.health_status.value,
            "weight": self.weight,
            "max_concurrent": self.max_concurrent,
            "is_available": self.is_available,
            "circuit_open": self.circuit_open,
        }


class AgentLoadTracker:
    """
    Tracks load and performance metrics for all agents.

    Thread-safe tracking of:
    - Active request counts
    - Response times (sliding window)
    - Error rates
    - Health status
    """

    def __init__(self):
        self._metrics: Dict[str, AgentLoadMetrics] = {}
        self._lock = threading.RLock()
        self._snapshots: Dict[str, List[AgentLoadSnapshot]] = {}
        self._snapshot_interval = 60  # seconds
        self._max_snapshots = 60  # Keep 1 hour of data

    def register_agent(
        self,
        agent_name: str,
        weight: int = 100,
        max_concurrent: int = 10,
    ) -> None:
        """Register an agent for load tracking."""
        with self._lock:
            if agent_name not in self._metrics:
                self._metrics[agent_name] = AgentLoadMetrics(
                    agent_name=agent_name,
                    weight=weight,
                    max_concurrent=max_concurrent,
                )
                self._snapshots[agent_name] = []
                logger.debug(f"Registered agent {agent_name} for load tracking")

    def unregister_agent(self, agent_name: str) -> None:
        """Unregister an agent from load tracking."""
        with self._lock:
            self._metrics.pop(agent_name, None)
            self._snapshots.pop(agent_name, None)

    def get_metrics(self, agent_name: str) -> Optional[AgentLoadMetrics]:
        """Get metrics for an agent."""
        with self._lock:
            return self._metrics.get(agent_name)

    def get_all_metrics(self) -> Dict[str, AgentLoadMetrics]:
        """Get metrics for all agents."""
        with self._lock:
            return self._metrics.copy()

    def request_start(self, agent_name: str) -> None:
        """Record that a request has started for an agent."""
        with self._lock:
            if agent_name in self._metrics:
                self._metrics[agent_name].record_request_start()

    def request_complete(
        self,
        agent_name: str,
        response_time_ms: float,
        success: bool,
    ) -> None:
        """Record that a request has completed for an agent."""
        with self._lock:
            if agent_name in self._metrics:
                self._metrics[agent_name].record_request_complete(response_time_ms, success)

    def update_health(self, agent_name: str, status: AgentHealthStatus) -> None:
        """Update health status for an agent."""
        with self._lock:
            if agent_name in self._metrics:
                self._metrics[agent_name].update_health_status(status)

    def set_weight(self, agent_name: str, weight: int) -> None:
        """Set weight for an agent (1-100)."""
        with self._lock:
            if agent_name in self._metrics:
                self._metrics[agent_name].weight = max(1, min(100, weight))

    def set_max_concurrent(self, agent_name: str, max_concurrent: int) -> None:
        """Set max concurrent requests for an agent."""
        with self._lock:
            if agent_name in self._metrics:
                self._metrics[agent_name].max_concurrent = max(1, max_concurrent)

    def take_snapshot(self) -> None:
        """Take a snapshot of current metrics for all agents."""
        with self._lock:
            now = datetime.now()
            for agent_name, metrics in self._metrics.items():
                snapshot = AgentLoadSnapshot(
                    timestamp=now,
                    active_requests=metrics.active_requests,
                    response_time_ms=metrics.average_response_time_ms,
                    error_count=metrics.requests_failed,
                    success_count=metrics.requests_succeeded,
                )
                self._snapshots[agent_name].append(snapshot)

                # Trim old snapshots
                if len(self._snapshots[agent_name]) > self._max_snapshots:
                    self._snapshots[agent_name] = self._snapshots[agent_name][
                        -self._max_snapshots :
                    ]

    def get_snapshots(
        self,
        agent_name: str,
        since: Optional[datetime] = None,
    ) -> List[AgentLoadSnapshot]:
        """Get historical snapshots for an agent."""
        with self._lock:
            snapshots = self._snapshots.get(agent_name, [])
            if since:
                snapshots = [s for s in snapshots if s.timestamp >= since]
            return snapshots.copy()


class LoadBalancer:
    """
    Load balancer that selects agents based on configurable strategies.

    Features:
    - Multiple load balancing strategies
    - Health-aware routing
    - Circuit breaker pattern
    - Adaptive load distribution
    """

    def __init__(
        self,
        tracker: AgentLoadTracker,
        strategy: LoadBalancingStrategy = LoadBalancingStrategy.ADAPTIVE,
    ):
        """
        Initialize load balancer.

        Args:
            tracker: Agent load tracker for metrics
            strategy: Load balancing strategy to use
        """
        self.tracker = tracker
        self.strategy = strategy
        self._round_robin_index: Dict[str, int] = {}  # Per-intent round robin state
        self._lock = threading.RLock()

    def select_agent(
        self,
        candidates: List[str],
        intent_key: str = "default",
    ) -> Optional[str]:
        """
        Select the best agent from candidates based on load balancing strategy.

        Args:
            candidates: List of candidate agent names
            intent_key: Key for round-robin state (e.g., intent category)

        Returns:
            Selected agent name, or None if no agents available
        """
        if not candidates:
            return None

        # Filter to available agents
        available = self._get_available_agents(candidates)
        if not available:
            # If no agents available, try to find least loaded anyway
            logger.warning("No available agents, falling back to all candidates")
            available = candidates

        # Apply strategy
        if self.strategy == LoadBalancingStrategy.ROUND_ROBIN:
            return self._round_robin_select(available, intent_key)
        elif self.strategy == LoadBalancingStrategy.WEIGHTED_ROUND_ROBIN:
            return self._weighted_round_robin_select(available, intent_key)
        elif self.strategy == LoadBalancingStrategy.LEAST_CONNECTIONS:
            return self._least_connections_select(available)
        elif self.strategy == LoadBalancingStrategy.LEAST_RESPONSE_TIME:
            return self._least_response_time_select(available)
        elif self.strategy == LoadBalancingStrategy.ADAPTIVE:
            return self._adaptive_select(available)
        else:
            # Fallback to first available
            return available[0]

    def _get_available_agents(self, candidates: List[str]) -> List[str]:
        """Filter to available agents."""
        available = []
        for agent_name in candidates:
            metrics = self.tracker.get_metrics(agent_name)
            if metrics is None:
                # No metrics = assume available
                available.append(agent_name)
            elif metrics.is_available:
                available.append(agent_name)
        return available

    def _round_robin_select(
        self,
        candidates: List[str],
        intent_key: str,
    ) -> str:
        """Simple round-robin selection."""
        with self._lock:
            if intent_key not in self._round_robin_index:
                self._round_robin_index[intent_key] = 0

            index = self._round_robin_index[intent_key] % len(candidates)
            self._round_robin_index[intent_key] = index + 1

            return candidates[index]

    def _weighted_round_robin_select(
        self,
        candidates: List[str],
        intent_key: str,
    ) -> str:
        """Weighted round-robin selection based on agent weights."""
        # Build weighted list
        weighted_list = []
        for agent_name in candidates:
            metrics = self.tracker.get_metrics(agent_name)
            weight = metrics.weight if metrics else 100
            # Add agent multiple times based on weight (normalized to reasonable count)
            count = max(1, weight // 10)
            weighted_list.extend([agent_name] * count)

        if not weighted_list:
            weighted_list = candidates

        with self._lock:
            if intent_key not in self._round_robin_index:
                self._round_robin_index[intent_key] = 0

            index = self._round_robin_index[intent_key] % len(weighted_list)
            self._round_robin_index[intent_key] = index + 1

            return weighted_list[index]

    def _least_connections_select(self, candidates: List[str]) -> str:
        """Select agent with fewest active connections."""
        best_agent = candidates[0]
        best_load = float("inf")

        for agent_name in candidates:
            metrics = self.tracker.get_metrics(agent_name)
            load = metrics.active_requests if metrics else 0

            if load < best_load:
                best_load = load
                best_agent = agent_name

        return best_agent

    def _least_response_time_select(self, candidates: List[str]) -> str:
        """Select agent with lowest average response time."""
        best_agent = candidates[0]
        best_time = float("inf")

        for agent_name in candidates:
            metrics = self.tracker.get_metrics(agent_name)
            resp_time = metrics.average_response_time_ms if metrics else 0

            # Treat 0 as unknown/average
            if resp_time == 0:
                resp_time = 500  # Default assumption

            if resp_time < best_time:
                best_time = resp_time
                best_agent = agent_name

        return best_agent

    def _adaptive_select(self, candidates: List[str]) -> str:
        """
        Adaptive selection considering multiple factors.

        Score = weighted combination of:
        - Load factor (lower is better)
        - Response time (lower is better)
        - Error rate (lower is better)
        - Health status (healthy is best)
        """
        best_agent = candidates[0]
        best_score = float("inf")

        for agent_name in candidates:
            metrics = self.tracker.get_metrics(agent_name)

            if metrics is None:
                # No metrics = neutral score
                score = 50.0
            else:
                # Calculate component scores (0-100, lower is better)
                load_score = metrics.load_factor * 100

                # Normalize response time (assume 2000ms is "bad")
                resp_time = metrics.average_response_time_ms
                time_score = min(100, (resp_time / 2000) * 100) if resp_time > 0 else 50

                # Error rate directly maps to score
                error_score = metrics.error_rate * 100

                # Health status score
                health_scores = {
                    AgentHealthStatus.HEALTHY: 0,
                    AgentHealthStatus.DEGRADED: 30,
                    AgentHealthStatus.UNHEALTHY: 100,
                    AgentHealthStatus.UNKNOWN: 20,
                }
                health_score = health_scores[metrics.health_status]

                # Weighted combination
                score = (
                    load_score * 0.35  # 35% weight on current load
                    + time_score * 0.25  # 25% weight on response time
                    + error_score * 0.25  # 25% weight on error rate
                    + health_score * 0.15  # 15% weight on health status
                )

            if score < best_score:
                best_score = score
                best_agent = agent_name

        return best_agent

    def get_load_distribution(
        self,
        candidates: List[str],
    ) -> Dict[str, Dict[str, Any]]:
        """
        Get current load distribution across candidate agents.

        Useful for monitoring and debugging.
        """
        distribution = {}

        for agent_name in candidates:
            metrics = self.tracker.get_metrics(agent_name)
            if metrics:
                distribution[agent_name] = metrics.to_dict()
            else:
                distribution[agent_name] = {
                    "agent_name": agent_name,
                    "active_requests": 0,
                    "status": "untracked",
                }

        return distribution


class HealthChecker:
    """
    Performs health checks on agents.

    Supports:
    - Periodic health checks
    - Custom health check functions
    - Automatic status updates
    """

    def __init__(
        self,
        tracker: AgentLoadTracker,
        check_interval: float = 30.0,  # seconds
    ):
        """
        Initialize health checker.

        Args:
            tracker: Agent load tracker to update
            check_interval: Interval between health checks
        """
        self.tracker = tracker
        self.check_interval = check_interval
        self._health_check_fns: Dict[str, Callable[[], bool]] = {}
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def register_health_check(
        self,
        agent_name: str,
        check_fn: Callable[[], bool],
    ) -> None:
        """
        Register a health check function for an agent.

        Args:
            agent_name: Agent to check
            check_fn: Function that returns True if healthy
        """
        self._health_check_fns[agent_name] = check_fn

    def unregister_health_check(self, agent_name: str) -> None:
        """Unregister health check for an agent."""
        self._health_check_fns.pop(agent_name, None)

    def check_agent(self, agent_name: str) -> AgentHealthStatus:
        """
        Perform health check for a specific agent.

        Returns:
            Health status
        """
        metrics = self.tracker.get_metrics(agent_name)

        # Check custom health function if registered
        if agent_name in self._health_check_fns:
            try:
                is_healthy = self._health_check_fns[agent_name]()
                if not is_healthy:
                    status = AgentHealthStatus.UNHEALTHY
                    self.tracker.update_health(agent_name, status)
                    return status
            except Exception as e:
                logger.warning(f"Health check failed for {agent_name}: {e}")
                status = AgentHealthStatus.UNHEALTHY
                self.tracker.update_health(agent_name, status)
                return status

        # Infer health from metrics
        if metrics is None:
            return AgentHealthStatus.UNKNOWN

        # Unhealthy: circuit breaker open or high error rate
        if metrics.circuit_open or metrics.error_rate > 0.5:
            status = AgentHealthStatus.UNHEALTHY
        # Degraded: elevated errors or slow responses
        elif metrics.error_rate > 0.1 or metrics.average_response_time_ms > 5000:
            status = AgentHealthStatus.DEGRADED
        else:
            status = AgentHealthStatus.HEALTHY

        self.tracker.update_health(agent_name, status)
        return status

    def check_all_agents(self) -> Dict[str, AgentHealthStatus]:
        """Perform health checks on all registered agents."""
        results = {}
        all_metrics = self.tracker.get_all_metrics()

        for agent_name in all_metrics.keys():
            results[agent_name] = self.check_agent(agent_name)

        return results

    def start(self) -> None:
        """Start periodic health checking."""
        if self._running:
            return

        self._running = True
        self._thread = threading.Thread(target=self._health_check_loop, daemon=True)
        self._thread.start()
        logger.info("Health checker started")

    def stop(self) -> None:
        """Stop periodic health checking."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5.0)
            self._thread = None
        logger.info("Health checker stopped")

    def _health_check_loop(self) -> None:
        """Background health check loop."""
        while self._running:
            try:
                self.check_all_agents()
                self.tracker.take_snapshot()
            except Exception as e:
                logger.error(f"Health check loop error: {e}")

            time.sleep(self.check_interval)


# Convenience function to create a full load balancing setup
def create_load_balancer(
    strategy: LoadBalancingStrategy = LoadBalancingStrategy.ADAPTIVE,
    health_check_interval: float = 30.0,
) -> tuple[AgentLoadTracker, LoadBalancer, HealthChecker]:
    """
    Create a complete load balancing setup.

    Args:
        strategy: Load balancing strategy
        health_check_interval: Health check interval in seconds

    Returns:
        Tuple of (tracker, balancer, health_checker)
    """
    tracker = AgentLoadTracker()
    balancer = LoadBalancer(tracker, strategy)
    health_checker = HealthChecker(tracker, health_check_interval)

    return tracker, balancer, health_checker
