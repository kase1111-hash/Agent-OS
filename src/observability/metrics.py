"""
Metrics Collection System

Provides a lightweight metrics collection framework compatible with
Prometheus format. Works standalone or with prometheus-client library.

Metric Types:
- Counter: Monotonically increasing value (requests, errors)
- Gauge: Value that can go up or down (active connections, temperature)
- Histogram: Distribution of values (request latency, response sizes)

Example:
    from src.observability.metrics import counter, gauge, histogram

    # Count requests
    requests = counter("http_requests_total", "Total HTTP requests")
    requests.labels(method="GET", path="/api/chat").inc()

    # Track active connections
    connections = gauge("active_connections", "Current connections")
    connections.set(42)

    # Record latencies
    latency = histogram("request_latency_seconds", "Request latency")
    latency.observe(0.125)
"""

import logging
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar

logger = logging.getLogger(__name__)

# Check for prometheus_client availability
try:
    import prometheus_client
    from prometheus_client import Counter as PromCounter
    from prometheus_client import Gauge as PromGauge
    from prometheus_client import Histogram as PromHistogram
    from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    logger.debug("prometheus_client not installed, using built-in metrics")


# Default histogram buckets (in seconds)
DEFAULT_BUCKETS = (
    0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0
)


@dataclass
class MetricValue:
    """A single metric value with labels."""

    value: float
    labels: Dict[str, str] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)


class Metric(ABC):
    """Base class for all metric types."""

    def __init__(
        self,
        name: str,
        description: str = "",
        labels: Optional[List[str]] = None,
    ):
        self.name = name
        self.description = description
        self.label_names = labels or []
        self._lock = threading.Lock()

    @abstractmethod
    def collect(self) -> List[MetricValue]:
        """Collect current metric values."""
        pass


class Counter(Metric):
    """A counter metric that only increases."""

    def __init__(
        self,
        name: str,
        description: str = "",
        labels: Optional[List[str]] = None,
    ):
        super().__init__(name, description, labels)
        self._values: Dict[Tuple[str, ...], float] = {}
        self._prom_counter = None

        if PROMETHEUS_AVAILABLE:
            self._prom_counter = PromCounter(name, description, labels or [])

    def labels(self, **kwargs: str) -> "CounterChild":
        """Return a child counter with specific label values."""
        return CounterChild(self, kwargs)

    def inc(self, value: float = 1.0, labels: Optional[Dict[str, str]] = None) -> None:
        """Increment the counter."""
        if value < 0:
            raise ValueError("Counter can only be incremented")

        label_values = tuple(sorted((labels or {}).items()))

        with self._lock:
            self._values[label_values] = self._values.get(label_values, 0) + value

        if self._prom_counter and labels:
            self._prom_counter.labels(**labels).inc(value)
        elif self._prom_counter:
            self._prom_counter.inc(value)

    def collect(self) -> List[MetricValue]:
        """Collect current counter values."""
        with self._lock:
            return [
                MetricValue(value=v, labels=dict(labels))
                for labels, v in self._values.items()
            ]


class CounterChild:
    """A counter with preset labels."""

    def __init__(self, parent: Counter, labels: Dict[str, str]):
        self._parent = parent
        self._labels = labels

    def inc(self, value: float = 1.0) -> None:
        """Increment this labeled counter."""
        self._parent.inc(value, self._labels)


class Gauge(Metric):
    """A gauge metric that can go up or down."""

    def __init__(
        self,
        name: str,
        description: str = "",
        labels: Optional[List[str]] = None,
    ):
        super().__init__(name, description, labels)
        self._values: Dict[Tuple[str, ...], float] = {}
        self._prom_gauge = None

        if PROMETHEUS_AVAILABLE:
            self._prom_gauge = PromGauge(name, description, labels or [])

    def labels(self, **kwargs: str) -> "GaugeChild":
        """Return a child gauge with specific label values."""
        return GaugeChild(self, kwargs)

    def set(self, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        """Set the gauge value."""
        label_values = tuple(sorted((labels or {}).items()))

        with self._lock:
            self._values[label_values] = value

        if self._prom_gauge and labels:
            self._prom_gauge.labels(**labels).set(value)
        elif self._prom_gauge:
            self._prom_gauge.set(value)

    def inc(self, value: float = 1.0, labels: Optional[Dict[str, str]] = None) -> None:
        """Increment the gauge."""
        label_values = tuple(sorted((labels or {}).items()))

        with self._lock:
            self._values[label_values] = self._values.get(label_values, 0) + value

        if self._prom_gauge and labels:
            self._prom_gauge.labels(**labels).inc(value)
        elif self._prom_gauge:
            self._prom_gauge.inc(value)

    def dec(self, value: float = 1.0, labels: Optional[Dict[str, str]] = None) -> None:
        """Decrement the gauge."""
        self.inc(-value, labels)

    def collect(self) -> List[MetricValue]:
        """Collect current gauge values."""
        with self._lock:
            return [
                MetricValue(value=v, labels=dict(labels))
                for labels, v in self._values.items()
            ]


class GaugeChild:
    """A gauge with preset labels."""

    def __init__(self, parent: Gauge, labels: Dict[str, str]):
        self._parent = parent
        self._labels = labels

    def set(self, value: float) -> None:
        """Set this labeled gauge."""
        self._parent.set(value, self._labels)

    def inc(self, value: float = 1.0) -> None:
        """Increment this labeled gauge."""
        self._parent.inc(value, self._labels)

    def dec(self, value: float = 1.0) -> None:
        """Decrement this labeled gauge."""
        self._parent.dec(value, self._labels)


@dataclass
class HistogramBucket:
    """A histogram bucket."""

    upper_bound: float
    count: int = 0


class Histogram(Metric):
    """A histogram metric for measuring distributions."""

    def __init__(
        self,
        name: str,
        description: str = "",
        labels: Optional[List[str]] = None,
        buckets: Tuple[float, ...] = DEFAULT_BUCKETS,
    ):
        super().__init__(name, description, labels)
        self._buckets = sorted(buckets) + [float("inf")]
        self._values: Dict[Tuple[str, ...], Dict[str, Any]] = {}
        self._prom_histogram = None

        if PROMETHEUS_AVAILABLE:
            self._prom_histogram = PromHistogram(name, description, labels or [], buckets=buckets)

    def labels(self, **kwargs: str) -> "HistogramChild":
        """Return a child histogram with specific label values."""
        return HistogramChild(self, kwargs)

    def _get_or_create_data(self, label_values: Tuple[str, ...]) -> Dict[str, Any]:
        """Get or create histogram data for labels."""
        if label_values not in self._values:
            self._values[label_values] = {
                "buckets": {b: 0 for b in self._buckets},
                "sum": 0.0,
                "count": 0,
            }
        return self._values[label_values]

    def observe(self, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        """Observe a value."""
        label_values = tuple(sorted((labels or {}).items()))

        with self._lock:
            data = self._get_or_create_data(label_values)
            data["sum"] += value
            data["count"] += 1
            for bucket in self._buckets:
                if value <= bucket:
                    data["buckets"][bucket] += 1

        if self._prom_histogram and labels:
            self._prom_histogram.labels(**labels).observe(value)
        elif self._prom_histogram:
            self._prom_histogram.observe(value)

    def time(self) -> "HistogramTimer":
        """Return a timer context manager."""
        return HistogramTimer(self)

    def collect(self) -> List[MetricValue]:
        """Collect current histogram values."""
        results = []
        with self._lock:
            for labels, data in self._values.items():
                label_dict = dict(labels)
                # Sum
                results.append(MetricValue(
                    value=data["sum"],
                    labels={**label_dict, "_type": "sum"},
                ))
                # Count
                results.append(MetricValue(
                    value=data["count"],
                    labels={**label_dict, "_type": "count"},
                ))
                # Buckets
                for bucket, count in data["buckets"].items():
                    results.append(MetricValue(
                        value=count,
                        labels={**label_dict, "_type": "bucket", "le": str(bucket)},
                    ))
        return results


class HistogramChild:
    """A histogram with preset labels."""

    def __init__(self, parent: Histogram, labels: Dict[str, str]):
        self._parent = parent
        self._labels = labels

    def observe(self, value: float) -> None:
        """Observe a value for this labeled histogram."""
        self._parent.observe(value, self._labels)

    def time(self) -> "HistogramTimer":
        """Return a timer context manager for this labeled histogram."""
        return HistogramTimer(self._parent, self._labels)


class HistogramTimer:
    """Context manager for timing operations."""

    def __init__(self, histogram: Histogram, labels: Optional[Dict[str, str]] = None):
        self._histogram = histogram
        self._labels = labels
        self._start: Optional[float] = None

    def __enter__(self) -> "HistogramTimer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args: Any) -> None:
        if self._start is not None:
            elapsed = time.perf_counter() - self._start
            self._histogram.observe(elapsed, self._labels)


class MetricsRegistry:
    """Registry for all metrics."""

    def __init__(self):
        self._metrics: Dict[str, Metric] = {}
        self._lock = threading.Lock()

    def counter(
        self,
        name: str,
        description: str = "",
        labels: Optional[List[str]] = None,
    ) -> Counter:
        """Get or create a counter metric."""
        with self._lock:
            if name not in self._metrics:
                self._metrics[name] = Counter(name, description, labels)
            return self._metrics[name]  # type: ignore

    def gauge(
        self,
        name: str,
        description: str = "",
        labels: Optional[List[str]] = None,
    ) -> Gauge:
        """Get or create a gauge metric."""
        with self._lock:
            if name not in self._metrics:
                self._metrics[name] = Gauge(name, description, labels)
            return self._metrics[name]  # type: ignore

    def histogram(
        self,
        name: str,
        description: str = "",
        labels: Optional[List[str]] = None,
        buckets: Tuple[float, ...] = DEFAULT_BUCKETS,
    ) -> Histogram:
        """Get or create a histogram metric."""
        with self._lock:
            if name not in self._metrics:
                self._metrics[name] = Histogram(name, description, labels, buckets)
            return self._metrics[name]  # type: ignore

    def collect_all(self) -> Dict[str, List[MetricValue]]:
        """Collect all metric values."""
        with self._lock:
            return {name: metric.collect() for name, metric in self._metrics.items()}

    def export_prometheus(self) -> str:
        """Export metrics in Prometheus text format."""
        if PROMETHEUS_AVAILABLE:
            return generate_latest().decode("utf-8")

        # Fallback to simple text format
        lines = []
        for name, values in self.collect_all().items():
            metric = self._metrics[name]
            lines.append(f"# HELP {name} {metric.description}")
            lines.append(f"# TYPE {name} {type(metric).__name__.lower()}")
            for mv in values:
                label_str = ",".join(f'{k}="{v}"' for k, v in mv.labels.items() if k != "_type")
                if label_str:
                    lines.append(f"{name}{{{label_str}}} {mv.value}")
                else:
                    lines.append(f"{name} {mv.value}")
        return "\n".join(lines)

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of all metrics as a dictionary."""
        summary: Dict[str, Any] = {}
        for name, values in self.collect_all().items():
            if not values:
                continue
            metric = self._metrics[name]
            metric_data: Dict[str, Any] = {
                "type": type(metric).__name__.lower(),
                "description": metric.description,
                "values": [],
            }
            for mv in values:
                metric_data["values"].append({
                    "labels": mv.labels,
                    "value": mv.value,
                })
            summary[name] = metric_data
        return summary


# Global registry instance
_registry: Optional[MetricsRegistry] = None


def get_metrics() -> MetricsRegistry:
    """Get the global metrics registry."""
    global _registry
    if _registry is None:
        _registry = MetricsRegistry()
    return _registry


def counter(
    name: str,
    description: str = "",
    labels: Optional[List[str]] = None,
) -> Counter:
    """Get or create a counter from the global registry."""
    return get_metrics().counter(name, description, labels)


def gauge(
    name: str,
    description: str = "",
    labels: Optional[List[str]] = None,
) -> Gauge:
    """Get or create a gauge from the global registry."""
    return get_metrics().gauge(name, description, labels)


def histogram(
    name: str,
    description: str = "",
    labels: Optional[List[str]] = None,
    buckets: Tuple[float, ...] = DEFAULT_BUCKETS,
) -> Histogram:
    """Get or create a histogram from the global registry."""
    return get_metrics().histogram(name, description, labels, buckets)
