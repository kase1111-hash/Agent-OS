"""
Distributed Tracing System

Provides distributed tracing with span management for tracking
request flows across the Agent OS system.

Compatible with OpenTelemetry when available, falls back to
a lightweight built-in implementation.

Example:
    from src.observability.tracing import span, trace

    # Using context manager
    with span("process_request") as s:
        s.set_attribute("user_id", "123")
        s.add_event("starting processing")
        # ... do work

    # Using decorator
    @trace("my_function")
    def my_function():
        pass
"""

import functools
import logging
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, Generator, List, Optional, TypeVar

logger = logging.getLogger(__name__)

# Check for OpenTelemetry availability
try:
    from opentelemetry import trace as otel_trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor, ConsoleSpanExporter

    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False
    logger.debug("OpenTelemetry not installed, using built-in tracing")


class SpanStatus(str, Enum):
    """Status of a span."""

    UNSET = "unset"
    OK = "ok"
    ERROR = "error"


@dataclass
class SpanEvent:
    """An event within a span."""

    name: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    attributes: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SpanContext:
    """Context for trace propagation."""

    trace_id: str
    span_id: str
    parent_span_id: Optional[str] = None


class Span:
    """A trace span representing a unit of work."""

    def __init__(
        self,
        name: str,
        context: Optional[SpanContext] = None,
        parent: Optional["Span"] = None,
    ):
        self.name = name
        self.context = context or SpanContext(
            trace_id=str(uuid.uuid4()),
            span_id=str(uuid.uuid4())[:16],
            parent_span_id=parent.context.span_id if parent else None,
        )
        self.parent = parent
        self.start_time: datetime = datetime.utcnow()
        self.end_time: Optional[datetime] = None
        self.status: SpanStatus = SpanStatus.UNSET
        self.status_message: Optional[str] = None
        self.attributes: Dict[str, Any] = {}
        self.events: List[SpanEvent] = []
        self._otel_span: Optional[Any] = None

    def set_attribute(self, key: str, value: Any) -> "Span":
        """Set an attribute on the span."""
        self.attributes[key] = value
        if self._otel_span:
            self._otel_span.set_attribute(key, value)
        return self

    def set_attributes(self, attributes: Dict[str, Any]) -> "Span":
        """Set multiple attributes on the span."""
        for key, value in attributes.items():
            self.set_attribute(key, value)
        return self

    def add_event(self, name: str, attributes: Optional[Dict[str, Any]] = None) -> "Span":
        """Add an event to the span."""
        event = SpanEvent(name=name, attributes=attributes or {})
        self.events.append(event)
        if self._otel_span:
            self._otel_span.add_event(name, attributes or {})
        return self

    def set_status(self, status: SpanStatus, message: Optional[str] = None) -> "Span":
        """Set the span status."""
        self.status = status
        self.status_message = message
        if self._otel_span:
            from opentelemetry.trace import StatusCode

            code = StatusCode.OK if status == SpanStatus.OK else StatusCode.ERROR
            self._otel_span.set_status(code, message)
        return self

    def record_exception(self, exception: Exception) -> "Span":
        """Record an exception on the span."""
        self.set_status(SpanStatus.ERROR, str(exception))
        self.add_event(
            "exception",
            {
                "exception.type": type(exception).__name__,
                "exception.message": str(exception),
            },
        )
        if self._otel_span:
            self._otel_span.record_exception(exception)
        return self

    def end(self) -> None:
        """End the span."""
        self.end_time = datetime.utcnow()
        if self._otel_span:
            self._otel_span.end()

    @property
    def duration_ms(self) -> float:
        """Get span duration in milliseconds."""
        if self.end_time is None:
            end = datetime.utcnow()
        else:
            end = self.end_time
        return (end - self.start_time).total_seconds() * 1000

    def to_dict(self) -> Dict[str, Any]:
        """Convert span to dictionary."""
        return {
            "name": self.name,
            "trace_id": self.context.trace_id,
            "span_id": self.context.span_id,
            "parent_span_id": self.context.parent_span_id,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_ms": self.duration_ms,
            "status": self.status.value,
            "status_message": self.status_message,
            "attributes": self.attributes,
            "events": [
                {
                    "name": e.name,
                    "timestamp": e.timestamp.isoformat(),
                    "attributes": e.attributes,
                }
                for e in self.events
            ],
        }

    def __enter__(self) -> "Span":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if exc_val is not None:
            self.record_exception(exc_val)
        elif self.status == SpanStatus.UNSET:
            self.set_status(SpanStatus.OK)
        self.end()


class Tracer:
    """A tracer for creating spans."""

    def __init__(self, name: str = "agent-os"):
        self.name = name
        self._spans: List[Span] = []
        self._lock = threading.Lock()
        self._current_span: threading.local = threading.local()
        self._otel_tracer: Optional[Any] = None

        if OTEL_AVAILABLE:
            try:
                provider = TracerProvider()
                processor = SimpleSpanProcessor(ConsoleSpanExporter())
                provider.add_span_processor(processor)
                otel_trace.set_tracer_provider(provider)
                self._otel_tracer = otel_trace.get_tracer(name)
            except Exception as e:
                logger.warning(f"Failed to initialize OpenTelemetry tracer: {e}")

    def start_span(
        self,
        name: str,
        parent: Optional[Span] = None,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> Span:
        """Start a new span."""
        # Use current span as parent if not specified
        if parent is None:
            parent = getattr(self._current_span, "span", None)

        # Create span context
        if parent:
            context = SpanContext(
                trace_id=parent.context.trace_id,
                span_id=str(uuid.uuid4())[:16],
                parent_span_id=parent.context.span_id,
            )
        else:
            context = SpanContext(
                trace_id=str(uuid.uuid4()),
                span_id=str(uuid.uuid4())[:16],
            )

        span = Span(name, context, parent)

        if attributes:
            span.set_attributes(attributes)

        # Store span
        with self._lock:
            self._spans.append(span)

        # Use OpenTelemetry if available
        if self._otel_tracer:
            span._otel_span = self._otel_tracer.start_span(name)
            if attributes:
                for k, v in attributes.items():
                    span._otel_span.set_attribute(k, v)

        return span

    @contextmanager
    def span(
        self,
        name: str,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> Generator[Span, None, None]:
        """Context manager for creating a span."""
        span = self.start_span(name, attributes=attributes)
        old_span = getattr(self._current_span, "span", None)
        self._current_span.span = span
        try:
            yield span
        except Exception as e:
            span.record_exception(e)
            raise
        finally:
            if span.status == SpanStatus.UNSET:
                span.set_status(SpanStatus.OK)
            span.end()
            self._current_span.span = old_span

    def get_current_span(self) -> Optional[Span]:
        """Get the current active span."""
        return getattr(self._current_span, "span", None)

    def get_recent_spans(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent completed spans."""
        with self._lock:
            completed = [s for s in self._spans if s.end_time is not None]
            # Keep only recent spans
            if len(completed) > limit * 2:
                self._spans = [s for s in self._spans if s.end_time is None] + completed[-limit:]
            return [s.to_dict() for s in completed[-limit:]]

    def clear(self) -> None:
        """Clear all stored spans."""
        with self._lock:
            self._spans.clear()


# Type variable for decorated functions
F = TypeVar("F", bound=Callable[..., Any])


def trace(
    name: Optional[str] = None,
    attributes: Optional[Dict[str, Any]] = None,
) -> Callable[[F], F]:
    """Decorator for tracing a function."""

    def decorator(func: F) -> F:
        span_name = name or func.__name__

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            tracer = get_tracer()
            with tracer.span(span_name, attributes=attributes) as s:
                s.set_attribute("function", func.__name__)
                s.set_attribute("module", func.__module__)
                return func(*args, **kwargs)

        return wrapper  # type: ignore

    return decorator


# Global tracer instance
_tracer: Optional[Tracer] = None


def get_tracer(name: str = "agent-os") -> Tracer:
    """Get the global tracer instance."""
    global _tracer
    if _tracer is None:
        _tracer = Tracer(name)
    return _tracer


@contextmanager
def span(
    name: str,
    attributes: Optional[Dict[str, Any]] = None,
) -> Generator[Span, None, None]:
    """Create a span using the global tracer."""
    tracer = get_tracer()
    with tracer.span(name, attributes=attributes) as s:
        yield s
