"""
Observability API Routes

Provides endpoints for metrics, health checks, and tracing data.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Query, Response
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)
router = APIRouter()


# =============================================================================
# Response Models
# =============================================================================


class HealthCheckDetail(BaseModel):
    """Details of a single health check."""

    name: str = Field(description="Name of the health check")
    status: str = Field(description="Status: healthy, degraded, unhealthy, unknown")
    message: Optional[str] = Field(None, description="Status message")
    latency_ms: Optional[float] = Field(None, description="Check latency in milliseconds")
    details: Dict[str, Any] = Field(default_factory=dict, description="Additional details")
    checked_at: str = Field(description="When the check was performed")
    error: Optional[str] = Field(None, description="Error message if check failed")


class HealthResponse(BaseModel):
    """Aggregated health check response."""

    status: str = Field(description="Overall status: healthy, degraded, unhealthy")
    checks: Dict[str, HealthCheckDetail] = Field(description="Individual check results")
    checked_at: str = Field(description="When checks were performed")

    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "checks": {
                    "system.memory": {
                        "name": "system.memory",
                        "status": "healthy",
                        "latency_ms": 1.2,
                        "details": {"used_percent": 45.2, "available_mb": 8192},
                        "checked_at": "2025-01-01T12:00:00Z",
                    },
                    "system.disk": {
                        "name": "system.disk",
                        "status": "healthy",
                        "latency_ms": 0.8,
                        "details": {"used_percent": 62.1, "free_gb": 128.5},
                        "checked_at": "2025-01-01T12:00:00Z",
                    },
                },
                "checked_at": "2025-01-01T12:00:00Z",
            }
        }


class MetricValue(BaseModel):
    """A metric value with labels."""

    labels: Dict[str, str] = Field(description="Metric labels")
    value: float = Field(description="Metric value")


class MetricInfo(BaseModel):
    """Information about a metric."""

    type: str = Field(description="Metric type: counter, gauge, histogram")
    description: str = Field(description="Metric description")
    values: List[MetricValue] = Field(description="Current values")


class MetricsSummary(BaseModel):
    """Summary of all metrics."""

    metrics: Dict[str, MetricInfo] = Field(description="All metrics")
    collected_at: str = Field(description="When metrics were collected")


class SpanInfo(BaseModel):
    """Information about a trace span."""

    name: str
    trace_id: str
    span_id: str
    parent_span_id: Optional[str]
    start_time: str
    end_time: Optional[str]
    duration_ms: float
    status: str
    status_message: Optional[str]
    attributes: Dict[str, Any]
    events: List[Dict[str, Any]]


class TracesResponse(BaseModel):
    """Response containing trace spans."""

    spans: List[SpanInfo] = Field(description="Recent trace spans")
    count: int = Field(description="Number of spans returned")


# =============================================================================
# Endpoints
# =============================================================================


@router.get("/health", response_model=HealthResponse)
async def get_health() -> HealthResponse:
    """
    Get health status of all components.

    Runs all registered health checks and returns aggregated status.
    Overall status is:
    - healthy: All critical checks pass
    - degraded: Some non-critical checks fail or critical checks are slow
    - unhealthy: One or more critical checks fail
    """
    from src.observability.health import check_all

    result = await check_all()

    return HealthResponse(
        status=result["status"],
        checks={name: HealthCheckDetail(**data) for name, data in result.get("checks", {}).items()},
        checked_at=result["checked_at"],
    )


@router.get("/health/{check_name}")
async def get_health_check(check_name: str) -> HealthCheckDetail:
    """
    Get health status of a specific component.

    Args:
        check_name: Name of the health check (e.g., 'system.memory', 'database')
    """
    from src.observability.health import check

    result = await check(check_name)

    return HealthCheckDetail(
        name=result.name,
        status=result.status.value,
        message=result.message,
        latency_ms=result.latency_ms,
        details=result.details,
        checked_at=result.checked_at.isoformat(),
        error=result.error,
    )


@router.get("/health/checks/list")
async def list_health_checks() -> List[Dict[str, Any]]:
    """
    List all registered health checks.

    Returns information about available health checks without running them.
    """
    from src.observability.health import get_health_aggregator

    return get_health_aggregator().list_checks()


@router.get("/metrics")
async def get_metrics_prometheus() -> Response:
    """
    Get metrics in Prometheus text format.

    Returns metrics in a format compatible with Prometheus scraping.
    Content-Type: text/plain; version=0.0.4; charset=utf-8
    """
    from src.observability.metrics import get_metrics

    registry = get_metrics()
    content = registry.export_prometheus()

    return Response(
        content=content,
        media_type="text/plain; version=0.0.4; charset=utf-8",
    )


@router.get("/metrics/json", response_model=MetricsSummary)
async def get_metrics_json() -> MetricsSummary:
    """
    Get metrics in JSON format.

    Returns a structured JSON representation of all metrics.
    """
    from src.observability.metrics import get_metrics

    registry = get_metrics()
    summary = registry.get_summary()

    return MetricsSummary(
        metrics={
            name: MetricInfo(
                type=data["type"],
                description=data["description"],
                values=[MetricValue(**v) for v in data["values"]],
            )
            for name, data in summary.items()
        },
        collected_at=datetime.utcnow().isoformat(),
    )


@router.get("/traces", response_model=TracesResponse)
async def get_traces(
    limit: int = Query(default=50, le=500, description="Maximum number of spans to return"),
) -> TracesResponse:
    """
    Get recent trace spans.

    Returns the most recent completed trace spans for debugging
    and performance analysis.
    """
    from src.observability.tracing import get_tracer

    tracer = get_tracer()
    spans = tracer.get_recent_spans(limit=limit)

    return TracesResponse(
        spans=[SpanInfo(**s) for s in spans],
        count=len(spans),
    )


@router.delete("/traces")
async def clear_traces() -> Dict[str, str]:
    """
    Clear all stored trace spans.

    Use this to reset tracing data during development or testing.
    """
    from src.observability.tracing import get_tracer

    tracer = get_tracer()
    tracer.clear()

    return {"status": "cleared", "message": "All trace spans have been cleared"}


@router.get("/status")
async def get_observability_status() -> Dict[str, Any]:
    """
    Get overall observability system status.

    Returns information about what observability features are available
    and their current state.
    """
    from src.observability.health import get_health_aggregator
    from src.observability.metrics import PROMETHEUS_AVAILABLE, get_metrics
    from src.observability.tracing import OTEL_AVAILABLE, get_tracer

    registry = get_metrics()
    tracer = get_tracer()
    health = get_health_aggregator()

    metrics_summary = registry.get_summary()
    recent_spans = tracer.get_recent_spans(limit=10)

    return {
        "metrics": {
            "enabled": True,
            "prometheus_available": PROMETHEUS_AVAILABLE,
            "metric_count": len(metrics_summary),
            "metrics_endpoint": "/api/observability/metrics",
        },
        "tracing": {
            "enabled": True,
            "opentelemetry_available": OTEL_AVAILABLE,
            "recent_span_count": len(recent_spans),
            "traces_endpoint": "/api/observability/traces",
        },
        "health": {
            "enabled": True,
            "check_count": len(health.list_checks()),
            "health_endpoint": "/api/observability/health",
        },
        "collected_at": datetime.utcnow().isoformat(),
    }
