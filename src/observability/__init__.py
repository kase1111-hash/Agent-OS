"""
Agent OS Observability Package

Provides metrics collection, distributed tracing, and health monitoring
for the Agent OS system.

Usage:
    from src.observability import metrics, tracing, health

    # Record a metric
    metrics.counter("requests_total", labels={"agent": "whisper"}).inc()

    # Create a trace span
    with tracing.span("process_request") as span:
        span.set_attribute("user_id", "123")
        # ... do work

    # Check system health
    status = health.check_all()
"""

from .metrics import (
    MetricsRegistry,
    Counter,
    Gauge,
    Histogram,
    get_metrics,
    counter,
    gauge,
    histogram,
)
from .tracing import (
    Span,
    Tracer,
    get_tracer,
    span,
    trace,
)
from .health import (
    HealthCheck,
    HealthStatus,
    HealthAggregator,
    get_health_aggregator,
    register_check,
    check_all,
)

__all__ = [
    # Metrics
    "MetricsRegistry",
    "Counter",
    "Gauge",
    "Histogram",
    "get_metrics",
    "counter",
    "gauge",
    "histogram",
    # Tracing
    "Span",
    "Tracer",
    "get_tracer",
    "span",
    "trace",
    # Health
    "HealthCheck",
    "HealthStatus",
    "HealthAggregator",
    "get_health_aggregator",
    "register_check",
    "check_all",
]
