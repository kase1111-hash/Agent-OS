"""
Behavioral Baseline

Learns normal behavioral patterns to detect anomalies.
Provides statistical baselines for event volumes, sources, and patterns.

Features:
- Adaptive baseline learning
- Multi-dimensional anomaly scoring
- Source-specific profiling
- Temporal pattern learning
- Threshold auto-tuning
"""

import logging
import statistics
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Tuple

from .store import IntelligenceEntry, SecurityIntelligenceStore

logger = logging.getLogger(__name__)


class AnomalyType(Enum):
    """Types of detected anomalies."""

    VOLUME = auto()  # Volume outside normal range
    SEVERITY = auto()  # Unusual severity distribution
    SOURCE = auto()  # Unusual source activity
    CATEGORY = auto()  # Unusual category distribution
    TEMPORAL = auto()  # Unusual timing pattern
    COMPOSITE = auto()  # Multiple anomaly types


@dataclass
class BaselineMetrics:
    """Baseline metrics for a specific dimension."""

    dimension: str  # e.g., "volume", "source:boundary-siem"
    samples: int = 0
    last_updated: datetime = field(default_factory=datetime.now)

    # Statistical values
    mean: float = 0.0
    std_dev: float = 0.0
    min_value: float = float('inf')
    max_value: float = float('-inf')

    # Percentiles
    p25: float = 0.0
    p50: float = 0.0  # Median
    p75: float = 0.0
    p95: float = 0.0
    p99: float = 0.0

    # Thresholds (auto-tuned)
    low_threshold: float = 0.0
    high_threshold: float = float('inf')

    # Learning state
    is_stable: bool = False
    stability_score: float = 0.0

    def update(self, values: List[float]) -> None:
        """Update baseline with new values."""
        if not values:
            return

        # Combine with existing if we have samples
        all_values = sorted(values)
        n = len(all_values)

        self.samples += n
        self.last_updated = datetime.now()

        # Update statistics
        self.mean = statistics.mean(all_values)
        self.std_dev = statistics.stdev(all_values) if n > 1 else 0

        self.min_value = min(self.min_value, min(all_values))
        self.max_value = max(self.max_value, max(all_values))

        # Update percentiles
        self.p25 = all_values[int(n * 0.25)] if n > 4 else self.mean
        self.p50 = all_values[int(n * 0.50)] if n > 2 else self.mean
        self.p75 = all_values[int(n * 0.75)] if n > 4 else self.mean
        self.p95 = all_values[int(n * 0.95)] if n > 20 else self.max_value
        self.p99 = all_values[int(n * 0.99)] if n > 100 else self.max_value

        # Update thresholds (using IQR method)
        iqr = self.p75 - self.p25
        self.low_threshold = max(0, self.p25 - 1.5 * iqr)
        self.high_threshold = self.p75 + 1.5 * iqr

        # Check stability (low coefficient of variation)
        if self.mean > 0:
            cv = self.std_dev / self.mean
            self.stability_score = max(0, 1 - cv)
            self.is_stable = cv < 0.5 and self.samples >= 24

    def is_anomalous(self, value: float) -> Tuple[bool, float]:
        """
        Check if a value is anomalous.

        Returns:
            (is_anomalous, anomaly_score)
        """
        if not self.is_stable:
            return False, 0.0

        if value < self.low_threshold:
            deviation = (self.low_threshold - value) / max(self.std_dev, 0.1)
            return True, min(1.0, deviation / 3)

        if value > self.high_threshold:
            deviation = (value - self.high_threshold) / max(self.std_dev, 0.1)
            return True, min(1.0, deviation / 3)

        return False, 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "dimension": self.dimension,
            "samples": self.samples,
            "last_updated": self.last_updated.isoformat(),
            "mean": self.mean,
            "std_dev": self.std_dev,
            "min_value": self.min_value if self.min_value != float('inf') else None,
            "max_value": self.max_value if self.max_value != float('-inf') else None,
            "p25": self.p25,
            "p50": self.p50,
            "p75": self.p75,
            "p95": self.p95,
            "p99": self.p99,
            "low_threshold": self.low_threshold,
            "high_threshold": self.high_threshold,
            "is_stable": self.is_stable,
            "stability_score": self.stability_score,
        }


@dataclass
class AnomalyScore:
    """Anomaly detection result."""

    score_id: str
    timestamp: datetime
    overall_score: float  # 0.0 = normal, 1.0 = highly anomalous

    # Component scores
    volume_score: float = 0.0
    severity_score: float = 0.0
    source_score: float = 0.0
    category_score: float = 0.0
    temporal_score: float = 0.0

    # Details
    anomaly_types: List[AnomalyType] = field(default_factory=list)
    anomalous_dimensions: List[str] = field(default_factory=list)
    description: str = ""

    # Context
    observed_values: Dict[str, float] = field(default_factory=dict)
    baseline_values: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "score_id": self.score_id,
            "timestamp": self.timestamp.isoformat(),
            "overall_score": self.overall_score,
            "volume_score": self.volume_score,
            "severity_score": self.severity_score,
            "source_score": self.source_score,
            "category_score": self.category_score,
            "temporal_score": self.temporal_score,
            "anomaly_types": [t.name for t in self.anomaly_types],
            "anomalous_dimensions": self.anomalous_dimensions,
            "description": self.description,
            "observed_values": self.observed_values,
            "baseline_values": self.baseline_values,
        }


class BehavioralBaseline:
    """
    Behavioral baseline learning and anomaly detection.

    Learns normal patterns from security events and detects
    deviations that may indicate attacks or system issues.
    """

    # Minimum samples before baseline is considered stable
    MIN_SAMPLES_FOR_STABILITY = 24  # hours

    # How often to update baselines
    LEARNING_INTERVAL_HOURS = 1

    def __init__(
        self,
        store: SecurityIntelligenceStore,
        on_anomaly: Optional[Callable[[AnomalyScore], None]] = None,
        learning_enabled: bool = True,
    ):
        """
        Initialize behavioral baseline.

        Args:
            store: Intelligence store
            on_anomaly: Callback for detected anomalies
            learning_enabled: Whether to continuously learn
        """
        self.store = store
        self.on_anomaly = on_anomaly
        self.learning_enabled = learning_enabled

        # Baselines
        self._global_baselines: Dict[str, BaselineMetrics] = {}
        self._source_baselines: Dict[str, Dict[str, BaselineMetrics]] = {}
        self._temporal_baselines: Dict[int, BaselineMetrics] = {}  # Hour of day

        # Thread safety
        self._lock = threading.RLock()

        # Background learning
        self._running = False
        self._learning_thread: Optional[threading.Thread] = None

        # Counter
        self._score_counter = 0

        # Statistics
        self._stats = {
            "anomalies_detected": 0,
            "learning_cycles": 0,
            "baselines_stable": 0,
        }

    def start(self) -> None:
        """Start background learning."""
        if self._running or not self.learning_enabled:
            return

        self._running = True
        self._learning_thread = threading.Thread(
            target=self._learning_loop,
            daemon=True,
            name="BaselineLearning",
        )
        self._learning_thread.start()
        logger.info("Behavioral baseline learning started")

    def stop(self) -> None:
        """Stop background learning."""
        self._running = False
        if self._learning_thread:
            self._learning_thread.join(timeout=5.0)
            self._learning_thread = None
        logger.info("Behavioral baseline learning stopped")

    def learn(self, lookback_hours: int = 168) -> None:
        """
        Learn baselines from historical data.

        Args:
            lookback_hours: How far back to learn from (default: 1 week)
        """
        since = datetime.now() - timedelta(hours=lookback_hours)

        with self._lock:
            entries = self.store.query(since=since, limit=100000)

            if len(entries) < 24:
                logger.info("Insufficient data for baseline learning")
                return

            # Learn volume baseline (events per hour)
            self._learn_volume_baseline(entries, lookback_hours)

            # Learn severity baseline
            self._learn_severity_baseline(entries)

            # Learn source-specific baselines
            self._learn_source_baselines(entries)

            # Learn temporal baselines (by hour of day)
            self._learn_temporal_baselines(entries)

            # Learn category baselines
            self._learn_category_baselines(entries)

            # Update stability stats
            self._stats["baselines_stable"] = sum(
                1 for b in self._global_baselines.values() if b.is_stable
            )
            self._stats["learning_cycles"] += 1

            logger.info(
                f"Baseline learning complete: {self._stats['baselines_stable']} stable baselines"
            )

    def _learn_volume_baseline(
        self,
        entries: List[IntelligenceEntry],
        lookback_hours: int,
    ) -> None:
        """Learn volume baseline (events per hour)."""
        # Group by hour
        hourly_counts: Dict[str, int] = defaultdict(int)
        for entry in entries:
            hour_key = entry.timestamp.strftime("%Y-%m-%d %H:00")
            hourly_counts[hour_key] += 1

        if len(hourly_counts) < 6:
            return

        values = list(hourly_counts.values())

        if "volume" not in self._global_baselines:
            self._global_baselines["volume"] = BaselineMetrics(dimension="volume")

        self._global_baselines["volume"].update(values)

    def _learn_severity_baseline(self, entries: List[IntelligenceEntry]) -> None:
        """Learn severity distribution baseline."""
        severity_values = [e.severity for e in entries]

        if "severity" not in self._global_baselines:
            self._global_baselines["severity"] = BaselineMetrics(dimension="severity")

        self._global_baselines["severity"].update([float(s) for s in severity_values])

        # Also track high-severity ratio
        high_sev_ratio = sum(1 for s in severity_values if s >= 4) / len(severity_values)

        if "high_severity_ratio" not in self._global_baselines:
            self._global_baselines["high_severity_ratio"] = BaselineMetrics(
                dimension="high_severity_ratio"
            )

        self._global_baselines["high_severity_ratio"].update([high_sev_ratio])

    def _learn_source_baselines(self, entries: List[IntelligenceEntry]) -> None:
        """Learn source-specific baselines."""
        # Group by source and hour
        source_hourly: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

        for entry in entries:
            hour_key = entry.timestamp.strftime("%Y-%m-%d %H:00")
            source_hourly[entry.source][hour_key] += 1

        for source, hourly_counts in source_hourly.items():
            if len(hourly_counts) < 6:
                continue

            if source not in self._source_baselines:
                self._source_baselines[source] = {}

            if "volume" not in self._source_baselines[source]:
                self._source_baselines[source]["volume"] = BaselineMetrics(
                    dimension=f"source:{source}:volume"
                )

            self._source_baselines[source]["volume"].update(list(hourly_counts.values()))

    def _learn_temporal_baselines(self, entries: List[IntelligenceEntry]) -> None:
        """Learn hour-of-day baselines."""
        # Group by hour of day
        hour_of_day_counts: Dict[int, List[int]] = defaultdict(list)

        # First, count by day and hour
        daily_hourly: Dict[Tuple[str, int], int] = defaultdict(int)
        for entry in entries:
            day_key = entry.timestamp.strftime("%Y-%m-%d")
            hour = entry.timestamp.hour
            daily_hourly[(day_key, hour)] += 1

        # Group by hour
        for (day_key, hour), count in daily_hourly.items():
            hour_of_day_counts[hour].append(count)

        for hour, values in hour_of_day_counts.items():
            if len(values) < 3:
                continue

            if hour not in self._temporal_baselines:
                self._temporal_baselines[hour] = BaselineMetrics(
                    dimension=f"hour:{hour:02d}"
                )

            self._temporal_baselines[hour].update([float(v) for v in values])

    def _learn_category_baselines(self, entries: List[IntelligenceEntry]) -> None:
        """Learn category distribution baselines."""
        # Count categories per hour
        hourly_categories: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

        for entry in entries:
            hour_key = entry.timestamp.strftime("%Y-%m-%d %H:00")
            hourly_categories[hour_key][entry.category] += 1

        # Calculate category proportions
        all_categories = set()
        for cats in hourly_categories.values():
            all_categories.update(cats.keys())

        for category in all_categories:
            proportions = []
            for hour_cats in hourly_categories.values():
                total = sum(hour_cats.values())
                if total > 0:
                    proportions.append(hour_cats.get(category, 0) / total)

            if len(proportions) < 6:
                continue

            dim = f"category:{category}"
            if dim not in self._global_baselines:
                self._global_baselines[dim] = BaselineMetrics(dimension=dim)

            self._global_baselines[dim].update(proportions)

    def score(
        self,
        lookback_minutes: int = 60,
    ) -> AnomalyScore:
        """
        Calculate anomaly score for recent activity.

        Args:
            lookback_minutes: How far back to analyze

        Returns:
            AnomalyScore with component scores
        """
        now = datetime.now()
        since = now - timedelta(minutes=lookback_minutes)

        with self._lock:
            entries = self.store.query(since=since, limit=10000)

            # Initialize scores
            volume_score = 0.0
            severity_score = 0.0
            source_score = 0.0
            category_score = 0.0
            temporal_score = 0.0

            anomaly_types = []
            anomalous_dimensions = []
            observed_values = {}
            baseline_values = {}

            # Volume anomaly check
            if "volume" in self._global_baselines:
                baseline = self._global_baselines["volume"]
                hourly_rate = len(entries) * 60 / lookback_minutes
                observed_values["volume"] = hourly_rate
                baseline_values["volume"] = baseline.mean

                is_anom, score = baseline.is_anomalous(hourly_rate)
                if is_anom:
                    volume_score = score
                    anomaly_types.append(AnomalyType.VOLUME)
                    anomalous_dimensions.append("volume")

            # Severity anomaly check
            if entries and "severity" in self._global_baselines:
                baseline = self._global_baselines["severity"]
                avg_severity = sum(e.severity for e in entries) / len(entries)
                observed_values["avg_severity"] = avg_severity
                baseline_values["avg_severity"] = baseline.mean

                is_anom, score = baseline.is_anomalous(avg_severity)
                if is_anom:
                    severity_score = score
                    anomaly_types.append(AnomalyType.SEVERITY)
                    anomalous_dimensions.append("severity")

            # Source-specific anomaly check
            source_scores = []
            for source, baselines in self._source_baselines.items():
                if "volume" in baselines:
                    source_entries = [e for e in entries if e.source == source]
                    hourly_rate = len(source_entries) * 60 / lookback_minutes

                    is_anom, score = baselines["volume"].is_anomalous(hourly_rate)
                    if is_anom:
                        source_scores.append(score)
                        anomalous_dimensions.append(f"source:{source}")

            if source_scores:
                source_score = max(source_scores)
                anomaly_types.append(AnomalyType.SOURCE)

            # Temporal anomaly check
            current_hour = now.hour
            if current_hour in self._temporal_baselines:
                baseline = self._temporal_baselines[current_hour]
                hourly_rate = len(entries) * 60 / lookback_minutes
                observed_values[f"hour_{current_hour:02d}_volume"] = hourly_rate
                baseline_values[f"hour_{current_hour:02d}_volume"] = baseline.mean

                is_anom, score = baseline.is_anomalous(hourly_rate)
                if is_anom:
                    temporal_score = score
                    anomaly_types.append(AnomalyType.TEMPORAL)
                    anomalous_dimensions.append(f"hour:{current_hour:02d}")

            # Category anomaly check
            if entries:
                category_scores = []
                total_entries = len(entries)
                category_counts = defaultdict(int)
                for entry in entries:
                    category_counts[entry.category] += 1

                for category, count in category_counts.items():
                    dim = f"category:{category}"
                    if dim in self._global_baselines:
                        baseline = self._global_baselines[dim]
                        proportion = count / total_entries

                        is_anom, score = baseline.is_anomalous(proportion)
                        if is_anom:
                            category_scores.append(score)
                            anomalous_dimensions.append(dim)

                if category_scores:
                    category_score = max(category_scores)
                    anomaly_types.append(AnomalyType.CATEGORY)

            # Calculate overall score (weighted average)
            weights = {
                "volume": 0.25,
                "severity": 0.25,
                "source": 0.20,
                "category": 0.15,
                "temporal": 0.15,
            }

            overall_score = (
                volume_score * weights["volume"]
                + severity_score * weights["severity"]
                + source_score * weights["source"]
                + category_score * weights["category"]
                + temporal_score * weights["temporal"]
            )

            # Generate description
            if overall_score > 0.7:
                description = "CRITICAL: Multiple severe anomalies detected"
            elif overall_score > 0.5:
                description = "HIGH: Significant behavioral anomalies"
            elif overall_score > 0.3:
                description = "MEDIUM: Notable behavioral deviations"
            elif overall_score > 0.1:
                description = "LOW: Minor behavioral variations"
            else:
                description = "Normal: Activity within expected baselines"

            if anomalous_dimensions:
                description += f". Anomalous dimensions: {', '.join(anomalous_dimensions[:5])}"

            self._score_counter += 1

            result = AnomalyScore(
                score_id=f"anom_{now.strftime('%Y%m%d_%H%M%S')}_{self._score_counter:04d}",
                timestamp=now,
                overall_score=overall_score,
                volume_score=volume_score,
                severity_score=severity_score,
                source_score=source_score,
                category_score=category_score,
                temporal_score=temporal_score,
                anomaly_types=anomaly_types,
                anomalous_dimensions=anomalous_dimensions,
                description=description,
                observed_values=observed_values,
                baseline_values=baseline_values,
            )

            # Track anomalies
            if overall_score > 0.3:
                self._stats["anomalies_detected"] += 1
                if self.on_anomaly:
                    self.on_anomaly(result)

            return result

    def get_baseline(self, dimension: str) -> Optional[BaselineMetrics]:
        """Get a specific baseline."""
        return self._global_baselines.get(dimension)

    def get_all_baselines(self) -> Dict[str, BaselineMetrics]:
        """Get all global baselines."""
        return dict(self._global_baselines)

    def get_source_baseline(
        self,
        source: str,
        metric: str = "volume",
    ) -> Optional[BaselineMetrics]:
        """Get a source-specific baseline."""
        if source in self._source_baselines:
            return self._source_baselines[source].get(metric)
        return None

    def get_temporal_baseline(self, hour: int) -> Optional[BaselineMetrics]:
        """Get a temporal baseline for specific hour."""
        return self._temporal_baselines.get(hour)

    def is_ready(self) -> bool:
        """Check if baselines are ready for scoring."""
        return any(b.is_stable for b in self._global_baselines.values())

    def get_stats(self) -> Dict[str, Any]:
        """Get baseline statistics."""
        return {
            **self._stats,
            "global_baselines": len(self._global_baselines),
            "source_baselines": len(self._source_baselines),
            "temporal_baselines": len(self._temporal_baselines),
            "is_ready": self.is_ready(),
        }

    def _learning_loop(self) -> None:
        """Background learning loop."""
        while self._running:
            try:
                self.learn()
            except Exception as e:
                logger.error(f"Baseline learning error: {e}")

            # Sleep for learning interval
            for _ in range(self.LEARNING_INTERVAL_HOURS * 3600):
                if not self._running:
                    break
                time.sleep(1)


def create_behavioral_baseline(
    store: SecurityIntelligenceStore,
    on_anomaly: Optional[Callable[[AnomalyScore], None]] = None,
    auto_start: bool = True,
) -> BehavioralBaseline:
    """
    Factory function to create a behavioral baseline.

    Args:
        store: Intelligence store
        on_anomaly: Callback for anomalies
        auto_start: Start learning automatically

    Returns:
        Configured BehavioralBaseline
    """
    baseline = BehavioralBaseline(
        store=store,
        on_anomaly=on_anomaly,
    )

    if auto_start:
        baseline.start()

    return baseline
