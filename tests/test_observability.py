"""
Tests for src/observability/ modules.

Covers:
- Metrics (Counter, Gauge, Histogram)
- Health checks (HealthAggregator, HealthCheckResult)
"""

import asyncio
import pytest
import time
from unittest.mock import MagicMock, patch


class TestMetricTypes:
    """Tests for metric type implementations."""

    def test_counter_increment(self):
        """Test counter increment."""
        from src.observability.metrics import Counter

        counter = Counter("test_requests", "Test requests counter")
        counter.inc()
        counter.inc(5)

        values = counter.collect()
        assert len(values) == 1
        assert values[0].value == 6

    def test_counter_negative_raises(self):
        """Test that negative increment raises."""
        from src.observability.metrics import Counter

        counter = Counter("test_counter", "Test counter")
        with pytest.raises(ValueError):
            counter.inc(-1)

    def test_counter_with_labels(self):
        """Test counter with labels."""
        from src.observability.metrics import Counter

        counter = Counter("http_requests", "HTTP requests", labels=["method", "path"])
        counter.labels(method="GET", path="/api").inc()
        counter.labels(method="POST", path="/api").inc(3)

        values = counter.collect()
        assert len(values) == 2
        total = sum(v.value for v in values)
        assert total == 4

    def test_gauge_set(self):
        """Test gauge set."""
        from src.observability.metrics import Gauge

        gauge = Gauge("active_connections", "Active connections")
        gauge.set(42)

        values = gauge.collect()
        assert len(values) == 1
        assert values[0].value == 42

    def test_gauge_inc_dec(self):
        """Test gauge increment and decrement."""
        from src.observability.metrics import Gauge

        gauge = Gauge("temperature", "Temperature")
        gauge.set(20)
        gauge.inc(5)
        gauge.dec(3)

        values = gauge.collect()
        assert values[0].value == 22

    def test_gauge_with_labels(self):
        """Test gauge with labels."""
        from src.observability.metrics import Gauge

        gauge = Gauge("memory_usage", "Memory usage", labels=["host"])
        gauge.labels(host="server1").set(1024)
        gauge.labels(host="server2").set(2048)

        values = gauge.collect()
        assert len(values) == 2

    def test_histogram_observe(self):
        """Test histogram observe."""
        from src.observability.metrics import Histogram

        histogram = Histogram(
            "request_latency",
            "Request latency",
            buckets=(0.1, 0.5, 1.0),
        )
        histogram.observe(0.05)
        histogram.observe(0.3)
        histogram.observe(0.8)
        histogram.observe(1.5)

        values = histogram.collect()
        # Should have sum, count, and bucket values
        assert len(values) > 0

        # Check sum
        sum_values = [v for v in values if v.labels.get("_type") == "sum"]
        assert len(sum_values) == 1
        assert abs(sum_values[0].value - 2.65) < 0.01

        # Check count
        count_values = [v for v in values if v.labels.get("_type") == "count"]
        assert len(count_values) == 1
        assert count_values[0].value == 4

    def test_histogram_timer(self):
        """Test histogram timer context manager."""
        from src.observability.metrics import Histogram

        histogram = Histogram("operation_duration", "Operation duration")

        with histogram.time():
            time.sleep(0.01)

        values = histogram.collect()
        count_values = [v for v in values if v.labels.get("_type") == "count"]
        assert count_values[0].value == 1

    def test_histogram_with_labels(self):
        """Test histogram with labels."""
        from src.observability.metrics import Histogram

        histogram = Histogram(
            "db_query_time",
            "Database query time",
            labels=["query_type"],
        )
        histogram.labels(query_type="SELECT").observe(0.1)
        histogram.labels(query_type="INSERT").observe(0.2)

        values = histogram.collect()
        assert len(values) > 0


class TestMetricsRegistry:
    """Tests for MetricsRegistry."""

    def test_counter_registration(self):
        """Test counter registration."""
        from src.observability.metrics import MetricsRegistry

        registry = MetricsRegistry()
        counter1 = registry.counter("test_counter", "Test")
        counter2 = registry.counter("test_counter", "Test")

        assert counter1 is counter2  # Same instance

    def test_gauge_registration(self):
        """Test gauge registration."""
        from src.observability.metrics import MetricsRegistry

        registry = MetricsRegistry()
        gauge1 = registry.gauge("test_gauge", "Test")
        gauge2 = registry.gauge("test_gauge", "Test")

        assert gauge1 is gauge2  # Same instance

    def test_histogram_registration(self):
        """Test histogram registration."""
        from src.observability.metrics import MetricsRegistry

        registry = MetricsRegistry()
        hist1 = registry.histogram("test_histogram", "Test")
        hist2 = registry.histogram("test_histogram", "Test")

        assert hist1 is hist2  # Same instance

    def test_collect_all(self):
        """Test collecting all metrics."""
        from src.observability.metrics import MetricsRegistry

        registry = MetricsRegistry()
        registry.counter("requests", "Requests").inc()
        registry.gauge("connections", "Connections").set(5)

        all_metrics = registry.collect_all()
        assert "requests" in all_metrics
        assert "connections" in all_metrics

    def test_export_prometheus(self):
        """Test Prometheus format export."""
        from src.observability.metrics import MetricsRegistry

        registry = MetricsRegistry()
        registry.counter("http_requests_total", "Total HTTP requests").inc(10)

        output = registry.export_prometheus()
        assert "http_requests_total" in output
        assert "10" in output or "10.0" in output

    def test_get_summary(self):
        """Test metrics summary."""
        from src.observability.metrics import MetricsRegistry

        registry = MetricsRegistry()
        registry.counter("test_counter", "A test counter").inc()

        summary = registry.get_summary()
        assert "test_counter" in summary
        assert summary["test_counter"]["type"] == "counter"
        assert summary["test_counter"]["description"] == "A test counter"


class TestGlobalMetricsFunctions:
    """Tests for global metrics functions."""

    def test_counter_function(self):
        """Test global counter function."""
        from src.observability.metrics import counter, get_metrics

        # Reset global registry for test isolation
        import src.observability.metrics as metrics_module
        metrics_module._registry = None

        c = counter("global_test_counter", "Test")
        c.inc()

        registry = get_metrics()
        assert "global_test_counter" in registry.collect_all()

    def test_gauge_function(self):
        """Test global gauge function."""
        from src.observability.metrics import gauge, get_metrics

        import src.observability.metrics as metrics_module
        metrics_module._registry = None

        g = gauge("global_test_gauge", "Test")
        g.set(123)

        registry = get_metrics()
        assert "global_test_gauge" in registry.collect_all()

    def test_histogram_function(self):
        """Test global histogram function."""
        from src.observability.metrics import histogram, get_metrics

        import src.observability.metrics as metrics_module
        metrics_module._registry = None

        h = histogram("global_test_histogram", "Test")
        h.observe(0.5)

        registry = get_metrics()
        assert "global_test_histogram" in registry.collect_all()


class TestHealthStatus:
    """Tests for HealthStatus enum."""

    def test_health_status_values(self):
        """Test health status values."""
        from src.observability.health import HealthStatus

        assert HealthStatus.HEALTHY.value == "healthy"
        assert HealthStatus.DEGRADED.value == "degraded"
        assert HealthStatus.UNHEALTHY.value == "unhealthy"
        assert HealthStatus.UNKNOWN.value == "unknown"


class TestHealthCheckResult:
    """Tests for HealthCheckResult dataclass."""

    def test_result_creation(self):
        """Test health check result creation."""
        from src.observability.health import HealthCheckResult, HealthStatus

        result = HealthCheckResult(
            name="database",
            status=HealthStatus.HEALTHY,
            latency_ms=45.2,
        )

        assert result.name == "database"
        assert result.status == HealthStatus.HEALTHY
        assert result.latency_ms == 45.2

    def test_result_to_dict(self):
        """Test health check result serialization."""
        from src.observability.health import HealthCheckResult, HealthStatus

        result = HealthCheckResult(
            name="test",
            status=HealthStatus.HEALTHY,
            message="All good",
            details={"version": "1.0"},
        )

        data = result.to_dict()
        assert data["name"] == "test"
        assert data["status"] == "healthy"
        assert data["message"] == "All good"
        assert data["details"]["version"] == "1.0"


class TestHealthCheck:
    """Tests for HealthCheck class."""

    @pytest.mark.asyncio
    async def test_sync_check_function(self):
        """Test synchronous check function."""
        from src.observability.health import HealthCheck, HealthStatus

        def check_fn():
            return True

        check = HealthCheck(name="sync_test", check_fn=check_fn)
        result = await check.execute()

        assert result.status == HealthStatus.HEALTHY
        assert result.latency_ms is not None

    @pytest.mark.asyncio
    async def test_async_check_function(self):
        """Test asynchronous check function."""
        from src.observability.health import HealthCheck, HealthStatus

        async def check_fn():
            await asyncio.sleep(0.01)
            return {"healthy": True, "message": "OK"}

        check = HealthCheck(name="async_test", check_fn=check_fn)
        result = await check.execute()

        assert result.status == HealthStatus.HEALTHY
        assert result.message == "OK"

    @pytest.mark.asyncio
    async def test_check_timeout(self):
        """Test check timeout handling."""
        from src.observability.health import HealthCheck, HealthStatus

        async def slow_check():
            await asyncio.sleep(10)
            return True

        check = HealthCheck(name="slow_test", check_fn=slow_check, timeout_seconds=0.1)
        result = await check.execute()

        assert result.status == HealthStatus.UNHEALTHY
        assert "Timeout" in result.error

    @pytest.mark.asyncio
    async def test_check_exception(self):
        """Test check exception handling."""
        from src.observability.health import HealthCheck, HealthStatus

        def failing_check():
            raise RuntimeError("Database connection failed")

        check = HealthCheck(name="failing_test", check_fn=failing_check)
        result = await check.execute()

        assert result.status == HealthStatus.UNHEALTHY
        assert "Database connection failed" in result.error

    @pytest.mark.asyncio
    async def test_check_degraded_status(self):
        """Test degraded status from check."""
        from src.observability.health import HealthCheck, HealthStatus

        def degraded_check():
            return {"healthy": True, "degraded": True, "message": "High load"}

        check = HealthCheck(name="degraded_test", check_fn=degraded_check)
        result = await check.execute()

        assert result.status == HealthStatus.DEGRADED


class TestHealthAggregator:
    """Tests for HealthAggregator class."""

    @pytest.fixture
    def aggregator(self):
        """Create a fresh aggregator."""
        from src.observability.health import HealthAggregator

        return HealthAggregator()

    def test_register_check(self, aggregator):
        """Test registering a health check."""
        aggregator.register("test", lambda: True, description="Test check")

        checks = aggregator.list_checks()
        assert len(checks) == 1
        assert checks[0]["name"] == "test"

    def test_unregister_check(self, aggregator):
        """Test unregistering a health check."""
        aggregator.register("test", lambda: True)
        assert aggregator.unregister("test")
        assert not aggregator.unregister("test")  # Already removed

    @pytest.mark.asyncio
    async def test_check_single(self, aggregator):
        """Test running a single check."""
        from src.observability.health import HealthStatus

        aggregator.register("database", lambda: True)
        result = await aggregator.check("database")

        assert result.status == HealthStatus.HEALTHY

    @pytest.mark.asyncio
    async def test_check_nonexistent(self, aggregator):
        """Test checking non-existent health check."""
        from src.observability.health import HealthStatus

        result = await aggregator.check("nonexistent")
        assert result.status == HealthStatus.UNKNOWN
        assert "not found" in result.error

    @pytest.mark.asyncio
    async def test_check_all_healthy(self, aggregator):
        """Test checking all when all are healthy."""
        aggregator.register("db", lambda: True)
        aggregator.register("cache", lambda: True)

        result = await aggregator.check_all()
        assert result["status"] == "healthy"
        assert len(result["checks"]) == 2

    @pytest.mark.asyncio
    async def test_check_all_one_unhealthy_critical(self, aggregator):
        """Test overall status when critical check fails."""
        aggregator.register("db", lambda: False, critical=True)
        aggregator.register("cache", lambda: True)

        result = await aggregator.check_all()
        assert result["status"] == "unhealthy"

    @pytest.mark.asyncio
    async def test_check_all_one_unhealthy_noncritical(self, aggregator):
        """Test overall status when non-critical check fails."""
        aggregator.register("db", lambda: True, critical=True)
        aggregator.register("optional", lambda: False, critical=False)

        result = await aggregator.check_all()
        assert result["status"] == "degraded"

    @pytest.mark.asyncio
    async def test_check_all_empty(self, aggregator):
        """Test checking all with no registered checks."""
        result = await aggregator.check_all()
        assert result["status"] == "healthy"
        assert result["checks"] == {}

    @pytest.mark.asyncio
    async def test_check_all_sequential(self, aggregator):
        """Test sequential check execution."""
        aggregator.register("check1", lambda: True)
        aggregator.register("check2", lambda: True)

        result = await aggregator.check_all(parallel=False)
        assert result["status"] == "healthy"

    def test_get_last_results(self, aggregator):
        """Test getting last results."""
        assert aggregator.get_last_results() == {}


class TestGlobalHealthFunctions:
    """Tests for global health check functions."""

    @pytest.mark.asyncio
    async def test_register_check_decorator(self):
        """Test register_check decorator."""
        from src.observability.health import register_check, get_health_aggregator

        # Reset global aggregator
        import src.observability.health as health_module
        health_module._aggregator = None

        @register_check("decorator_test", description="Test via decorator")
        def my_check():
            return True

        aggregator = get_health_aggregator()
        checks = aggregator.list_checks()
        check_names = [c["name"] for c in checks]
        assert "decorator_test" in check_names

    @pytest.mark.asyncio
    async def test_check_all_function(self):
        """Test global check_all function."""
        from src.observability.health import check_all

        result = await check_all()
        assert "status" in result
        assert "checks" in result
        assert "checked_at" in result

    @pytest.mark.asyncio
    async def test_check_function(self):
        """Test global check function."""
        from src.observability.health import check, get_health_aggregator

        # Reset global aggregator
        import src.observability.health as health_module
        health_module._aggregator = None

        aggregator = get_health_aggregator()
        aggregator.register("test_check", lambda: {"healthy": True})

        result = await check("test_check")
        assert result.name == "test_check"


class TestDefaultHealthChecks:
    """Tests for default health checks."""

    @pytest.mark.asyncio
    async def test_memory_check(self):
        """Test system memory health check."""
        from src.observability.health import get_health_aggregator

        # Reset to get fresh default checks
        import src.observability.health as health_module
        health_module._aggregator = None

        aggregator = get_health_aggregator()
        result = await aggregator.check("system.memory")

        # Should return a result (may be healthy or have message about platform)
        assert result.name == "system.memory"

    @pytest.mark.asyncio
    async def test_disk_check(self):
        """Test system disk health check."""
        from src.observability.health import get_health_aggregator

        import src.observability.health as health_module
        health_module._aggregator = None

        aggregator = get_health_aggregator()
        result = await aggregator.check("system.disk")

        assert result.name == "system.disk"
