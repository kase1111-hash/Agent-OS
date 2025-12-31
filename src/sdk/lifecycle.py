"""
Agent Lifecycle Management

Provides tools for managing agent lifecycles including:
- Health checks
- Graceful shutdown
- Hot reloading
- State management
"""

import atexit
import logging
import signal
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set

from src.agents.interface import AgentState, BaseAgent

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Health status of an agent."""

    HEALTHY = auto()
    DEGRADED = auto()
    UNHEALTHY = auto()
    UNKNOWN = auto()


@dataclass
class HealthCheck:
    """Result of a health check."""

    status: HealthStatus
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    latency_ms: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status.name,
            "message": self.message,
            "details": self.details,
            "timestamp": self.timestamp.isoformat(),
            "latency_ms": self.latency_ms,
        }


@dataclass
class AgentStats:
    """Runtime statistics for an agent."""

    requests_processed: int = 0
    requests_succeeded: int = 0
    requests_failed: int = 0
    requests_refused: int = 0
    total_latency_ms: int = 0
    uptime_seconds: float = 0
    last_request_time: Optional[datetime] = None
    errors: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def average_latency_ms(self) -> float:
        if self.requests_processed == 0:
            return 0
        return self.total_latency_ms / self.requests_processed

    @property
    def success_rate(self) -> float:
        if self.requests_processed == 0:
            return 0
        return self.requests_succeeded / self.requests_processed

    def to_dict(self) -> Dict[str, Any]:
        return {
            "requests_processed": self.requests_processed,
            "requests_succeeded": self.requests_succeeded,
            "requests_failed": self.requests_failed,
            "requests_refused": self.requests_refused,
            "average_latency_ms": self.average_latency_ms,
            "success_rate": self.success_rate,
            "uptime_seconds": self.uptime_seconds,
            "last_request_time": (
                self.last_request_time.isoformat() if self.last_request_time else None
            ),
            "recent_errors": self.errors[-5:],
        }


class AgentLifecycleManager:
    """
    Manages the lifecycle of an agent.

    Provides:
    - Health monitoring
    - Graceful shutdown
    - Stats collection
    - Event callbacks
    """

    def __init__(self, agent: BaseAgent):
        """
        Initialize lifecycle manager.

        Args:
            agent: Agent to manage
        """
        self.agent = agent
        self.stats = AgentStats()
        self._started_at: Optional[datetime] = None
        self._health_checks: List[Callable[[], HealthCheck]] = []
        self._shutdown_hooks: List[Callable[[], None]] = []
        self._event_handlers: Dict[str, List[Callable]] = {
            "started": [],
            "stopped": [],
            "request_received": [],
            "request_completed": [],
            "error": [],
            "health_check": [],
        }
        self._shutdown_requested = False
        self._lock = threading.RLock()

    def start(self, config: Optional[Dict[str, Any]] = None) -> bool:
        """
        Start the agent.

        Args:
            config: Agent configuration

        Returns:
            True if started successfully
        """
        with self._lock:
            if self._started_at:
                logger.warning(f"Agent {self.agent.name} already started")
                return True

            config = config or {}

            try:
                result = self.agent.initialize(config)
                if result:
                    self._started_at = datetime.now()
                    self._emit_event("started")
                    logger.info(f"Agent {self.agent.name} started")
                    return True
                else:
                    logger.error(f"Agent {self.agent.name} failed to initialize")
                    return False

            except Exception as e:
                logger.error(f"Agent {self.agent.name} start error: {e}")
                return False

    def stop(self, timeout: float = 5.0) -> bool:
        """
        Stop the agent gracefully.

        Args:
            timeout: Maximum time to wait for shutdown

        Returns:
            True if stopped successfully
        """
        with self._lock:
            if not self._started_at:
                return True

            self._shutdown_requested = True

            try:
                # Run shutdown hooks
                for hook in self._shutdown_hooks:
                    try:
                        hook()
                    except Exception as e:
                        logger.warning(f"Shutdown hook error: {e}")

                # Shutdown agent
                result = self.agent.shutdown()

                self._started_at = None
                self._emit_event("stopped")
                logger.info(f"Agent {self.agent.name} stopped")

                return result

            except Exception as e:
                logger.error(f"Agent {self.agent.name} stop error: {e}")
                return False

    def is_running(self) -> bool:
        """Check if agent is running."""
        with self._lock:
            return self._started_at is not None and not self._shutdown_requested

    def get_uptime(self) -> timedelta:
        """Get agent uptime."""
        with self._lock:
            if not self._started_at:
                return timedelta(0)
            return datetime.now() - self._started_at

    def check_health(self) -> HealthCheck:
        """
        Perform health check.

        Returns:
            HealthCheck result
        """
        start = time.time()

        with self._lock:
            if not self._started_at:
                return HealthCheck(
                    status=HealthStatus.UNHEALTHY,
                    message="Agent not started",
                )

            if self.agent.state != AgentState.READY:
                return HealthCheck(
                    status=HealthStatus.UNHEALTHY,
                    message=f"Agent in state: {self.agent.state.name}",
                )

        # Run custom health checks
        issues = []
        for checker in self._health_checks:
            try:
                check = checker()
                if check.status != HealthStatus.HEALTHY:
                    issues.append(check)
            except Exception as e:
                issues.append(
                    HealthCheck(
                        status=HealthStatus.UNHEALTHY,
                        message=f"Health check error: {e}",
                    )
                )

        latency = int((time.time() - start) * 1000)

        if not issues:
            result = HealthCheck(
                status=HealthStatus.HEALTHY,
                message="Agent is healthy",
                latency_ms=latency,
            )
        elif any(i.status == HealthStatus.UNHEALTHY for i in issues):
            result = HealthCheck(
                status=HealthStatus.UNHEALTHY,
                message="Agent has health issues",
                details={"issues": [i.to_dict() for i in issues]},
                latency_ms=latency,
            )
        else:
            result = HealthCheck(
                status=HealthStatus.DEGRADED,
                message="Agent is degraded",
                details={"issues": [i.to_dict() for i in issues]},
                latency_ms=latency,
            )

        self._emit_event("health_check", result)
        return result

    def get_stats(self) -> AgentStats:
        """Get agent statistics."""
        with self._lock:
            if self._started_at:
                self.stats.uptime_seconds = (datetime.now() - self._started_at).total_seconds()
            return self.stats

    def add_health_check(
        self,
        checker: Callable[[], HealthCheck],
    ) -> "AgentLifecycleManager":
        """Add a custom health check."""
        self._health_checks.append(checker)
        return self

    def add_shutdown_hook(
        self,
        hook: Callable[[], None],
    ) -> "AgentLifecycleManager":
        """Add a shutdown hook."""
        self._shutdown_hooks.append(hook)
        return self

    def on(
        self,
        event: str,
        handler: Callable,
    ) -> "AgentLifecycleManager":
        """Register an event handler."""
        if event in self._event_handlers:
            self._event_handlers[event].append(handler)
        return self

    def record_request(self, success: bool, latency_ms: int) -> None:
        """Record a request completion."""
        with self._lock:
            self.stats.requests_processed += 1
            self.stats.total_latency_ms += latency_ms
            self.stats.last_request_time = datetime.now()

            if success:
                self.stats.requests_succeeded += 1
            else:
                self.stats.requests_failed += 1

    def record_error(self, error: Exception) -> None:
        """Record an error."""
        with self._lock:
            self.stats.errors.append(
                {
                    "error": str(error),
                    "type": type(error).__name__,
                    "timestamp": datetime.now().isoformat(),
                }
            )
            # Keep only last 100 errors
            if len(self.stats.errors) > 100:
                self.stats.errors = self.stats.errors[-100:]

        self._emit_event("error", error)

    def _emit_event(self, event: str, *args) -> None:
        """Emit an event to handlers."""
        for handler in self._event_handlers.get(event, []):
            try:
                handler(*args)
            except Exception as e:
                logger.warning(f"Event handler error: {e}")


class AgentPool:
    """
    Pool of agents for load balancing.

    Manages multiple agent instances for horizontal scaling.
    """

    def __init__(
        self,
        agent_factory: Callable[[], BaseAgent],
        pool_size: int = 3,
    ):
        """
        Initialize agent pool.

        Args:
            agent_factory: Factory function to create agents
            pool_size: Number of agents in pool
        """
        self.agent_factory = agent_factory
        self.pool_size = pool_size
        self._agents: List[AgentLifecycleManager] = []
        self._current_index = 0
        self._lock = threading.RLock()

    def start(self, config: Optional[Dict[str, Any]] = None) -> bool:
        """Start all agents in pool."""
        with self._lock:
            for i in range(self.pool_size):
                agent = self.agent_factory()
                manager = AgentLifecycleManager(agent)

                if manager.start(config):
                    self._agents.append(manager)
                else:
                    logger.error(f"Failed to start agent {i}")
                    return False

            logger.info(f"Started agent pool with {len(self._agents)} agents")
            return True

    def stop(self) -> bool:
        """Stop all agents in pool."""
        with self._lock:
            success = True
            for manager in self._agents:
                if not manager.stop():
                    success = False
            self._agents.clear()
            return success

    def get_agent(self) -> Optional[BaseAgent]:
        """Get next available agent (round-robin)."""
        with self._lock:
            if not self._agents:
                return None

            # Find healthy agent
            for _ in range(len(self._agents)):
                manager = self._agents[self._current_index]
                self._current_index = (self._current_index + 1) % len(self._agents)

                if manager.is_running():
                    health = manager.check_health()
                    if health.status != HealthStatus.UNHEALTHY:
                        return manager.agent

            return None

    def get_all_stats(self) -> List[Dict[str, Any]]:
        """Get stats for all agents in pool."""
        with self._lock:
            return [
                {
                    "name": m.agent.name,
                    "stats": m.get_stats().to_dict(),
                    "health": m.check_health().to_dict(),
                }
                for m in self._agents
            ]


_managers: Dict[str, AgentLifecycleManager] = {}
_shutdown_registered = False


def register_agent(
    agent: BaseAgent,
    auto_start: bool = False,
    config: Optional[Dict[str, Any]] = None,
) -> AgentLifecycleManager:
    """
    Register an agent for lifecycle management.

    Args:
        agent: Agent to register
        auto_start: Start agent immediately
        config: Agent configuration

    Returns:
        AgentLifecycleManager for the agent
    """
    global _shutdown_registered

    manager = AgentLifecycleManager(agent)
    _managers[agent.name] = manager

    # Register shutdown handler
    if not _shutdown_registered:
        atexit.register(_shutdown_all)
        signal.signal(signal.SIGTERM, _signal_handler)
        signal.signal(signal.SIGINT, _signal_handler)
        _shutdown_registered = True

    if auto_start:
        manager.start(config)

    return manager


def get_manager(agent_name: str) -> Optional[AgentLifecycleManager]:
    """Get manager for an agent."""
    return _managers.get(agent_name)


def shutdown_all() -> None:
    """Shutdown all registered agents."""
    _shutdown_all()


def _shutdown_all() -> None:
    """Internal shutdown function."""
    logger.info("Shutting down all agents...")
    for name, manager in _managers.items():
        try:
            manager.stop()
        except Exception as e:
            logger.error(f"Error shutting down {name}: {e}")
    _managers.clear()


def _signal_handler(signum, frame) -> None:
    """Handle shutdown signals."""
    logger.info(f"Received signal {signum}, initiating shutdown...")
    _shutdown_all()
