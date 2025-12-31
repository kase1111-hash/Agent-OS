"""
Observability Middleware for FastAPI

Provides automatic metrics collection and tracing for HTTP requests.

Example:
    from fastapi import FastAPI
    from src.observability.middleware import setup_observability

    app = FastAPI()
    setup_observability(app)
"""

import logging
import time
from typing import Callable, Optional

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
from starlette.types import ASGIApp

from .metrics import counter, gauge, histogram, get_metrics
from .tracing import get_tracer, SpanStatus

logger = logging.getLogger(__name__)


# Pre-defined metrics
REQUEST_COUNT = counter(
    "http_requests_total",
    "Total HTTP requests",
    labels=["method", "path", "status"],
)

REQUEST_LATENCY = histogram(
    "http_request_duration_seconds",
    "HTTP request latency in seconds",
    labels=["method", "path"],
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
)

REQUESTS_IN_PROGRESS = gauge(
    "http_requests_in_progress",
    "Number of HTTP requests currently being processed",
    labels=["method"],
)

REQUEST_SIZE = histogram(
    "http_request_size_bytes",
    "HTTP request size in bytes",
    labels=["method", "path"],
    buckets=(100, 1000, 10000, 100000, 1000000),
)

RESPONSE_SIZE = histogram(
    "http_response_size_bytes",
    "HTTP response size in bytes",
    labels=["method", "path"],
    buckets=(100, 1000, 10000, 100000, 1000000),
)


def _normalize_path(path: str) -> str:
    """Normalize path for metrics to avoid high cardinality."""
    # Replace common dynamic segments with placeholders
    import re

    # UUIDs
    path = re.sub(
        r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
        "{uuid}",
        path,
        flags=re.IGNORECASE,
    )
    # Numeric IDs
    path = re.sub(r"/\d+(/|$)", "/{id}\\1", path)
    # Common ID patterns
    path = re.sub(r"/(msg|user|session|conv|agent|memory|contract)_[a-zA-Z0-9]+", "/\\1_{id}", path)

    return path


class MetricsMiddleware(BaseHTTPMiddleware):
    """Middleware for collecting HTTP metrics."""

    def __init__(
        self,
        app: ASGIApp,
        exclude_paths: Optional[list[str]] = None,
    ):
        super().__init__(app)
        self.exclude_paths = exclude_paths or ["/health", "/metrics", "/favicon.ico"]

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Skip excluded paths
        path = request.url.path
        if any(path.startswith(exc) for exc in self.exclude_paths):
            return await call_next(request)

        method = request.method
        normalized_path = _normalize_path(path)

        # Track in-progress requests
        REQUESTS_IN_PROGRESS.inc(labels={"method": method})

        # Track request size
        content_length = request.headers.get("content-length")
        if content_length:
            REQUEST_SIZE.observe(int(content_length), labels={"method": method, "path": normalized_path})

        start_time = time.perf_counter()

        try:
            response = await call_next(request)
            status_code = response.status_code
        except Exception as e:
            status_code = 500
            raise
        finally:
            # Record metrics
            duration = time.perf_counter() - start_time

            REQUEST_COUNT.inc(
                labels={"method": method, "path": normalized_path, "status": str(status_code)}
            )
            REQUEST_LATENCY.observe(duration, labels={"method": method, "path": normalized_path})
            REQUESTS_IN_PROGRESS.dec(labels={"method": method})

        # Track response size
        response_size = response.headers.get("content-length")
        if response_size:
            RESPONSE_SIZE.observe(
                int(response_size), labels={"method": method, "path": normalized_path}
            )

        return response


class TracingMiddleware(BaseHTTPMiddleware):
    """Middleware for distributed tracing of HTTP requests."""

    def __init__(
        self,
        app: ASGIApp,
        exclude_paths: Optional[list[str]] = None,
    ):
        super().__init__(app)
        self.exclude_paths = exclude_paths or ["/health", "/metrics", "/favicon.ico"]
        self.tracer = get_tracer()

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Skip excluded paths
        path = request.url.path
        if any(path.startswith(exc) for exc in self.exclude_paths):
            return await call_next(request)

        method = request.method
        span_name = f"{method} {path}"

        with self.tracer.span(span_name) as span:
            # Set request attributes
            span.set_attribute("http.method", method)
            span.set_attribute("http.url", str(request.url))
            span.set_attribute("http.route", path)
            span.set_attribute("http.scheme", request.url.scheme)
            span.set_attribute("http.host", request.url.hostname or "")

            # Extract trace context from headers (if present)
            traceparent = request.headers.get("traceparent")
            if traceparent:
                span.set_attribute("http.traceparent", traceparent)

            # Client info
            client = request.client
            if client:
                span.set_attribute("http.client_ip", client.host)

            # User agent
            user_agent = request.headers.get("user-agent")
            if user_agent:
                span.set_attribute("http.user_agent", user_agent[:200])

            try:
                response = await call_next(request)

                # Set response attributes
                span.set_attribute("http.status_code", response.status_code)
                span.set_attribute("http.response_content_length",
                                 response.headers.get("content-length", "0"))

                if response.status_code >= 400:
                    span.set_status(
                        SpanStatus.ERROR,
                        f"HTTP {response.status_code}",
                    )

                return response

            except Exception as e:
                span.record_exception(e)
                span.set_attribute("http.status_code", 500)
                raise


def setup_observability(
    app: "FastAPI",  # type: ignore
    enable_metrics: bool = True,
    enable_tracing: bool = True,
    exclude_paths: Optional[list[str]] = None,
) -> None:
    """
    Set up observability middleware for a FastAPI app.

    Args:
        app: FastAPI application
        enable_metrics: Enable metrics collection
        enable_tracing: Enable distributed tracing
        exclude_paths: Paths to exclude from observability
    """
    exclude = exclude_paths or ["/health", "/metrics", "/favicon.ico", "/docs", "/redoc", "/openapi.json"]

    if enable_tracing:
        app.add_middleware(TracingMiddleware, exclude_paths=exclude)

    if enable_metrics:
        app.add_middleware(MetricsMiddleware, exclude_paths=exclude)

    logger.info(
        f"Observability middleware enabled: metrics={enable_metrics}, tracing={enable_tracing}"
    )


# Agent-specific metrics
AGENT_REQUESTS = counter(
    "agent_requests_total",
    "Total requests to agents",
    labels=["agent", "intent", "status"],
)

AGENT_LATENCY = histogram(
    "agent_request_duration_seconds",
    "Agent request processing time",
    labels=["agent"],
    buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0),
)

CONSTITUTION_VALIDATIONS = counter(
    "constitution_validations_total",
    "Total constitutional validation checks",
    labels=["result"],
)

MEMORY_OPERATIONS = counter(
    "memory_operations_total",
    "Total memory operations",
    labels=["operation", "memory_class"],
)

LLM_REQUESTS = counter(
    "llm_requests_total",
    "Total LLM inference requests",
    labels=["model", "status"],
)

LLM_LATENCY = histogram(
    "llm_request_duration_seconds",
    "LLM inference time",
    labels=["model"],
    buckets=(0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0),
)

LLM_TOKENS = counter(
    "llm_tokens_total",
    "Total tokens processed",
    labels=["model", "type"],  # type: input/output
)
