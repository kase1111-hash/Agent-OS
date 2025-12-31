"""
Health Check System

Provides a centralized health check aggregator for monitoring
the status of all Agent OS components.

Health Status Levels:
- HEALTHY: Component is fully operational
- DEGRADED: Component is operational but with reduced capacity
- UNHEALTHY: Component is not operational

Example:
    from src.observability.health import register_check, check_all

    # Register a health check
    @register_check("database")
    async def check_database():
        # Return True if healthy, False otherwise
        return await db.ping()

    # Or with more details
    @register_check("llm")
    async def check_llm():
        return {
            "healthy": True,
            "latency_ms": 45.2,
            "model": "llama3:8b",
        }

    # Check all components
    status = await check_all()
"""

import asyncio
import functools
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


class HealthStatus(str, Enum):
    """Health status levels."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class HealthCheckResult:
    """Result of a health check."""

    name: str
    status: HealthStatus
    message: Optional[str] = None
    latency_ms: Optional[float] = None
    details: Dict[str, Any] = field(default_factory=dict)
    checked_at: datetime = field(default_factory=datetime.utcnow)
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "status": self.status.value,
            "message": self.message,
            "latency_ms": self.latency_ms,
            "details": self.details,
            "checked_at": self.checked_at.isoformat(),
            "error": self.error,
        }


@dataclass
class HealthCheck:
    """A registered health check."""

    name: str
    check_fn: Callable[[], Any]
    timeout_seconds: float = 5.0
    critical: bool = True
    description: str = ""

    async def execute(self) -> HealthCheckResult:
        """Execute the health check."""
        start_time = time.perf_counter()

        try:
            # Handle both sync and async check functions
            if asyncio.iscoroutinefunction(self.check_fn):
                result = await asyncio.wait_for(
                    self.check_fn(),
                    timeout=self.timeout_seconds,
                )
            else:
                # Run sync function in thread pool
                loop = asyncio.get_event_loop()
                result = await asyncio.wait_for(
                    loop.run_in_executor(None, self.check_fn),
                    timeout=self.timeout_seconds,
                )

            latency_ms = (time.perf_counter() - start_time) * 1000

            # Parse result
            if isinstance(result, bool):
                status = HealthStatus.HEALTHY if result else HealthStatus.UNHEALTHY
                return HealthCheckResult(
                    name=self.name,
                    status=status,
                    latency_ms=latency_ms,
                )
            elif isinstance(result, dict):
                healthy = result.get("healthy", result.get("status") == "healthy")
                status = HealthStatus.HEALTHY if healthy else HealthStatus.UNHEALTHY
                if result.get("degraded"):
                    status = HealthStatus.DEGRADED
                return HealthCheckResult(
                    name=self.name,
                    status=status,
                    message=result.get("message"),
                    latency_ms=latency_ms,
                    details={k: v for k, v in result.items() if k not in ("healthy", "status", "message")},
                )
            elif isinstance(result, HealthCheckResult):
                result.latency_ms = latency_ms
                return result
            else:
                # Assume truthy = healthy
                status = HealthStatus.HEALTHY if result else HealthStatus.UNHEALTHY
                return HealthCheckResult(
                    name=self.name,
                    status=status,
                    latency_ms=latency_ms,
                )

        except asyncio.TimeoutError:
            latency_ms = (time.perf_counter() - start_time) * 1000
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.UNHEALTHY,
                latency_ms=latency_ms,
                error=f"Timeout after {self.timeout_seconds}s",
            )
        except Exception as e:
            latency_ms = (time.perf_counter() - start_time) * 1000
            logger.warning(f"Health check '{self.name}' failed: {e}")
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.UNHEALTHY,
                latency_ms=latency_ms,
                error=str(e),
            )


class HealthAggregator:
    """Aggregates health checks from all components."""

    def __init__(self):
        self._checks: Dict[str, HealthCheck] = {}
        self._last_results: Dict[str, HealthCheckResult] = {}

    def register(
        self,
        name: str,
        check_fn: Callable[[], Any],
        timeout_seconds: float = 5.0,
        critical: bool = True,
        description: str = "",
    ) -> None:
        """Register a health check."""
        self._checks[name] = HealthCheck(
            name=name,
            check_fn=check_fn,
            timeout_seconds=timeout_seconds,
            critical=critical,
            description=description,
        )
        logger.debug(f"Registered health check: {name}")

    def unregister(self, name: str) -> bool:
        """Unregister a health check."""
        if name in self._checks:
            del self._checks[name]
            self._last_results.pop(name, None)
            return True
        return False

    async def check(self, name: str) -> HealthCheckResult:
        """Run a specific health check."""
        if name not in self._checks:
            return HealthCheckResult(
                name=name,
                status=HealthStatus.UNKNOWN,
                error=f"Health check '{name}' not found",
            )

        result = await self._checks[name].execute()
        self._last_results[name] = result
        return result

    async def check_all(self, parallel: bool = True) -> Dict[str, Any]:
        """Run all health checks."""
        if not self._checks:
            return {
                "status": HealthStatus.HEALTHY.value,
                "checks": {},
                "checked_at": datetime.utcnow().isoformat(),
            }

        # Run checks
        if parallel:
            results = await asyncio.gather(
                *[check.execute() for check in self._checks.values()],
                return_exceptions=True,
            )
            for i, (name, _) in enumerate(self._checks.items()):
                if isinstance(results[i], Exception):
                    results[i] = HealthCheckResult(
                        name=name,
                        status=HealthStatus.UNHEALTHY,
                        error=str(results[i]),
                    )
                self._last_results[name] = results[i]
        else:
            results = []
            for name, check in self._checks.items():
                result = await check.execute()
                self._last_results[name] = result
                results.append(result)

        # Aggregate status
        overall_status = HealthStatus.HEALTHY
        for result in results:
            if isinstance(result, HealthCheckResult):
                check = self._checks.get(result.name)
                is_critical = check.critical if check else True

                if result.status == HealthStatus.UNHEALTHY and is_critical:
                    overall_status = HealthStatus.UNHEALTHY
                elif result.status == HealthStatus.DEGRADED and overall_status != HealthStatus.UNHEALTHY:
                    overall_status = HealthStatus.DEGRADED
                elif result.status == HealthStatus.UNHEALTHY and not is_critical:
                    if overall_status == HealthStatus.HEALTHY:
                        overall_status = HealthStatus.DEGRADED

        return {
            "status": overall_status.value,
            "checks": {
                r.name: r.to_dict() for r in results if isinstance(r, HealthCheckResult)
            },
            "checked_at": datetime.utcnow().isoformat(),
        }

    def get_last_results(self) -> Dict[str, HealthCheckResult]:
        """Get the last results of all health checks."""
        return dict(self._last_results)

    def list_checks(self) -> List[Dict[str, Any]]:
        """List all registered health checks."""
        return [
            {
                "name": check.name,
                "description": check.description,
                "timeout_seconds": check.timeout_seconds,
                "critical": check.critical,
            }
            for check in self._checks.values()
        ]


# Global aggregator instance
_aggregator: Optional[HealthAggregator] = None


def get_health_aggregator() -> HealthAggregator:
    """Get the global health aggregator."""
    global _aggregator
    if _aggregator is None:
        _aggregator = HealthAggregator()
        # Register default checks
        _register_default_checks(_aggregator)
    return _aggregator


def _register_default_checks(aggregator: HealthAggregator) -> None:
    """Register default health checks."""

    # System memory check
    def check_memory() -> Dict[str, Any]:
        try:
            import os

            # Try to get memory info (Linux)
            if os.path.exists("/proc/meminfo"):
                with open("/proc/meminfo") as f:
                    meminfo = {}
                    for line in f:
                        parts = line.split(":")
                        if len(parts) == 2:
                            key = parts[0].strip()
                            value = parts[1].strip().split()[0]
                            meminfo[key] = int(value)

                total = meminfo.get("MemTotal", 0)
                available = meminfo.get("MemAvailable", 0)
                if total > 0:
                    used_percent = ((total - available) / total) * 100
                    return {
                        "healthy": used_percent < 90,
                        "degraded": used_percent >= 80,
                        "used_percent": round(used_percent, 1),
                        "available_mb": round(available / 1024, 1),
                    }
            return {"healthy": True, "message": "Memory check not available on this platform"}
        except Exception as e:
            return {"healthy": True, "message": str(e)}

    aggregator.register(
        "system.memory",
        check_memory,
        timeout_seconds=2.0,
        critical=False,
        description="System memory usage",
    )

    # Disk space check
    def check_disk() -> Dict[str, Any]:
        try:
            import os
            import shutil

            total, used, free = shutil.disk_usage("/")
            used_percent = (used / total) * 100
            return {
                "healthy": used_percent < 90,
                "degraded": used_percent >= 80,
                "used_percent": round(used_percent, 1),
                "free_gb": round(free / (1024**3), 1),
            }
        except Exception as e:
            return {"healthy": True, "message": str(e)}

    aggregator.register(
        "system.disk",
        check_disk,
        timeout_seconds=2.0,
        critical=False,
        description="System disk usage",
    )


def register_check(
    name: str,
    timeout_seconds: float = 5.0,
    critical: bool = True,
    description: str = "",
) -> Callable[[Callable[[], Any]], Callable[[], Any]]:
    """Decorator to register a health check function."""

    def decorator(func: Callable[[], Any]) -> Callable[[], Any]:
        get_health_aggregator().register(
            name=name,
            check_fn=func,
            timeout_seconds=timeout_seconds,
            critical=critical,
            description=description or func.__doc__ or "",
        )
        return func

    return decorator


async def check_all() -> Dict[str, Any]:
    """Run all health checks."""
    return await get_health_aggregator().check_all()


async def check(name: str) -> HealthCheckResult:
    """Run a specific health check."""
    return await get_health_aggregator().check(name)
