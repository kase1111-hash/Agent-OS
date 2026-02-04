"""
Advanced Memory Manager

Unified manager for Agent Smith's advanced memory system.
Coordinates all components: store, correlator, synthesizer, baseline, and connectors.

Features:
- Unified configuration and lifecycle management
- Event routing and processing pipeline
- Cross-component integration
- Summary and report generation
- Health monitoring
"""

import logging
import threading
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from .store import (
    SecurityIntelligenceStore,
    IntelligenceEntry,
    IntelligenceType,
    RetentionPolicy,
    create_intelligence_store,
)
from .correlator import (
    ThreatCorrelator,
    CorrelationResult,
    ThreatCluster,
    ThreatLevel,
    create_threat_correlator,
)
from .synthesizer import (
    PatternSynthesizer,
    SynthesizedPattern,
    TrendAnalysis,
    IntelligenceSummary,
    create_pattern_synthesizer,
)
from .baseline import (
    BehavioralBaseline,
    AnomalyScore,
    create_behavioral_baseline,
)
from .boundary_connector import (
    BoundaryDaemonConnector,
    BoundaryEvent,
    BoundaryMode,
    create_boundary_connector,
)

logger = logging.getLogger(__name__)


@dataclass
class MemoryConfig:
    """Configuration for Advanced Memory System."""

    # Storage
    storage_path: Optional[str] = None  # Path for persistent storage
    hot_retention_days: int = 7
    warm_retention_days: int = 30
    cold_retention_days: int = 365

    # Boundary-Daemon integration
    boundary_endpoint: Optional[str] = None
    boundary_api_key: Optional[str] = None
    boundary_poll_interval: int = 30

    # SIEM integration (via existing siem_connector)
    siem_enabled: bool = False

    # Correlation
    correlation_enabled: bool = True
    correlation_lookback_minutes: int = 60

    # Pattern synthesis
    synthesis_enabled: bool = True
    synthesis_interval_hours: int = 1

    # Behavioral baseline
    baseline_enabled: bool = True
    baseline_learning_hours: int = 168  # 1 week

    # Callbacks
    on_high_threat: Optional[Callable[[ThreatCluster], None]] = None
    on_anomaly: Optional[Callable[[AnomalyScore], None]] = None
    on_pattern: Optional[Callable[[SynthesizedPattern], None]] = None


@dataclass
class MemoryStatus:
    """Status of the advanced memory system."""

    initialized: bool = False
    running: bool = False
    healthy: bool = False

    # Component status
    store_healthy: bool = False
    correlator_active: bool = False
    synthesizer_active: bool = False
    baseline_ready: bool = False
    boundary_connected: bool = False

    # Metrics
    total_entries: int = 0
    hot_entries: int = 0
    active_clusters: int = 0
    detected_patterns: int = 0
    anomaly_score: float = 0.0

    # Threat assessment
    current_threat_level: ThreatLevel = ThreatLevel.UNKNOWN
    critical_threats: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "initialized": self.initialized,
            "running": self.running,
            "healthy": self.healthy,
            "store_healthy": self.store_healthy,
            "correlator_active": self.correlator_active,
            "synthesizer_active": self.synthesizer_active,
            "baseline_ready": self.baseline_ready,
            "boundary_connected": self.boundary_connected,
            "total_entries": self.total_entries,
            "hot_entries": self.hot_entries,
            "active_clusters": self.active_clusters,
            "detected_patterns": self.detected_patterns,
            "anomaly_score": self.anomaly_score,
            "current_threat_level": self.current_threat_level.name,
            "critical_threats": self.critical_threats,
        }


class AdvancedMemoryManager:
    """
    Manages Agent Smith's advanced memory system.

    Integrates:
    - SecurityIntelligenceStore for tiered data storage
    - ThreatCorrelator for cross-source event correlation
    - PatternSynthesizer for trend analysis and insights
    - BehavioralBaseline for anomaly detection
    - BoundaryDaemonConnector for external integration

    This gives Smith the ability to:
    1. Remember security events with intelligent retention
    2. Correlate threats across Boundary-SIEM and Boundary-Daemon
    3. Detect patterns and generate actionable intelligence
    4. Learn normal behavior and detect anomalies
    5. Generate executive summaries for security posture
    """

    def __init__(self, config: MemoryConfig):
        """
        Initialize Advanced Memory Manager.

        Args:
            config: Memory configuration
        """
        self.config = config

        # Components (initialized in initialize())
        self._store: Optional[SecurityIntelligenceStore] = None
        self._correlator: Optional[ThreatCorrelator] = None
        self._synthesizer: Optional[PatternSynthesizer] = None
        self._baseline: Optional[BehavioralBaseline] = None
        self._boundary: Optional[BoundaryDaemonConnector] = None

        # State
        self._initialized = False
        self._running = False
        self._lock = threading.RLock()
        self._shutdown_event = threading.Event()

        # Synthesis thread
        self._synthesis_thread: Optional[threading.Thread] = None

        # Statistics
        self._stats = {
            "events_processed": 0,
            "correlations_found": 0,
            "patterns_detected": 0,
            "anomalies_detected": 0,
            "summaries_generated": 0,
        }

    def initialize(self) -> bool:
        """
        Initialize all memory components.

        Returns:
            True if initialization successful
        """
        if self._initialized:
            return True

        with self._lock:
            try:
                # Initialize store
                retention = RetentionPolicy(
                    hot_days=self.config.hot_retention_days,
                    warm_days=self.config.warm_retention_days,
                    cold_days=self.config.cold_retention_days,
                )

                self._store = create_intelligence_store(
                    storage_path=self.config.storage_path,
                    retention_policy=retention,
                    auto_start=False,  # We'll start it ourselves
                )
                logger.info("Intelligence store initialized")

                # Initialize correlator
                if self.config.correlation_enabled:
                    self._correlator = create_threat_correlator(
                        store=self._store,
                        on_correlation=self._on_correlation,
                    )
                    logger.info("Threat correlator initialized")

                # Initialize synthesizer
                if self.config.synthesis_enabled:
                    self._synthesizer = create_pattern_synthesizer(
                        store=self._store,
                        correlator=self._correlator,
                        on_pattern=self._on_pattern,
                    )
                    logger.info("Pattern synthesizer initialized")

                # Initialize baseline
                if self.config.baseline_enabled:
                    self._baseline = create_behavioral_baseline(
                        store=self._store,
                        on_anomaly=self._on_anomaly,
                        auto_start=False,
                    )
                    logger.info("Behavioral baseline initialized")

                # Initialize boundary connector (always init for local use)
                self._boundary = create_boundary_connector(
                    store=self._store,
                    endpoint=self.config.boundary_endpoint,
                    api_key=self.config.boundary_api_key,
                    auto_start=False,
                )
                logger.info("Boundary-Daemon connector initialized")

                self._initialized = True
                logger.info("Advanced Memory Manager initialized successfully")
                return True

            except Exception as e:
                logger.error(f"Failed to initialize Advanced Memory Manager: {e}")
                return False

    def start(self) -> bool:
        """
        Start all memory components.

        Returns:
            True if started successfully
        """
        if not self._initialized:
            if not self.initialize():
                return False

        if self._running:
            return True

        with self._lock:
            try:
                # Clear shutdown event for fresh start
                self._shutdown_event.clear()

                # Start store
                self._store.start()

                # Start baseline learning
                if self._baseline:
                    self._baseline.start()
                    # Initial learning
                    self._baseline.learn(lookback_hours=self.config.baseline_learning_hours)

                # Start boundary connector
                if self._boundary:
                    self._boundary.connect()
                    self._boundary.start(poll_interval=self.config.boundary_poll_interval)

                # Start synthesis thread
                if self.config.synthesis_enabled:
                    self._synthesis_thread = threading.Thread(
                        target=self._synthesis_loop,
                        daemon=True,
                        name="MemorySynthesis",
                    )
                    self._synthesis_thread.start()

                self._running = True
                logger.info("Advanced Memory Manager started")
                return True

            except Exception as e:
                logger.error(f"Failed to start Advanced Memory Manager: {e}")
                return False

    def stop(self) -> None:
        """Stop all memory components."""
        self._running = False
        self._shutdown_event.set()  # Signal shutdown to waiting threads

        with self._lock:
            # Stop synthesis thread
            if self._synthesis_thread:
                self._synthesis_thread.join(timeout=5.0)
                self._synthesis_thread = None

            # Stop boundary connector
            if self._boundary:
                self._boundary.stop()

            # Stop baseline
            if self._baseline:
                self._baseline.stop()

            # Stop store
            if self._store:
                self._store.stop()

        logger.info("Advanced Memory Manager stopped")

    def ingest(self, entry: IntelligenceEntry) -> str:
        """
        Ingest an intelligence entry.

        This is the main entry point for adding security events
        to the advanced memory system.

        Args:
            entry: Entry to ingest

        Returns:
            Entry ID
        """
        if not self._initialized:
            raise RuntimeError("Advanced Memory Manager not initialized")

        # Add to store
        entry_id = self._store.add(entry)
        self._stats["events_processed"] += 1

        # Run correlation
        if self._correlator:
            try:
                correlations = self._correlator.correlate(
                    entry,
                    lookback_minutes=self.config.correlation_lookback_minutes,
                )
                self._stats["correlations_found"] += len(correlations)
            except Exception as e:
                logger.warning(f"Correlation error: {e}")

        return entry_id

    def ingest_siem_event(self, event_data: Dict[str, Any]) -> str:
        """
        Ingest an event from Boundary-SIEM.

        Args:
            event_data: SIEM event data

        Returns:
            Entry ID
        """
        # Map SIEM severity
        severity_map = {
            "critical": 5,
            "high": 4,
            "medium": 3,
            "low": 2,
            "informational": 1,
            "info": 1,
        }
        severity = severity_map.get(
            str(event_data.get("severity", "")).lower(),
            2
        )

        entry = IntelligenceEntry(
            entry_id="",  # Will be generated
            entry_type=IntelligenceType.SIEM_EVENT,
            timestamp=datetime.fromisoformat(
                event_data.get("timestamp", datetime.now().isoformat()).replace("Z", "+00:00")
            ),
            source="boundary-siem",
            severity=severity,
            category=event_data.get("category", "security"),
            summary=event_data.get("message", event_data.get("description", "")),
            content=event_data,
            indicators=event_data.get("indicators", []),
            mitre_tactics=event_data.get("mitre_attack", {}).get("tactics", []),
            tags=set(event_data.get("tags", [])) | {"siem"},
        )

        return self.ingest(entry)

    def ingest_boundary_event(self, event: BoundaryEvent) -> str:
        """
        Ingest an event from Boundary-Daemon.

        Args:
            event: Boundary-Daemon event

        Returns:
            Entry ID
        """
        entry = event.to_intelligence_entry()
        return self.ingest(entry)

    def query(
        self,
        entry_types: Optional[List[IntelligenceType]] = None,
        sources: Optional[List[str]] = None,
        severity_min: Optional[int] = None,
        since: Optional[datetime] = None,
        limit: int = 100,
    ) -> List[IntelligenceEntry]:
        """
        Query intelligence entries.

        Args:
            entry_types: Filter by types
            sources: Filter by sources
            severity_min: Minimum severity
            since: Entries after this time
            limit: Maximum results

        Returns:
            List of matching entries
        """
        if not self._store:
            return []

        return self._store.query(
            entry_types=entry_types,
            sources=sources,
            severity_min=severity_min,
            since=since,
            limit=limit,
        )

    def get_threat_clusters(
        self,
        min_threat_level: ThreatLevel = ThreatLevel.LOW,
        since: Optional[datetime] = None,
    ) -> List[ThreatCluster]:
        """Get active threat clusters."""
        if not self._correlator:
            return []

        return self._correlator.get_active_clusters(
            min_threat_level=min_threat_level,
            since=since,
        )

    def get_patterns(
        self,
        min_severity: int = 1,
        since: Optional[datetime] = None,
    ) -> List[SynthesizedPattern]:
        """Get detected patterns."""
        if not self._synthesizer:
            return []

        return self._synthesizer.get_patterns(
            min_severity=min_severity,
            since=since,
        )

    def get_anomaly_score(self, lookback_minutes: int = 60) -> Optional[AnomalyScore]:
        """Get current anomaly score."""
        if not self._baseline:
            return None

        return self._baseline.score(lookback_minutes=lookback_minutes)

    def get_trend_analysis(
        self,
        period_hours: int = 24,
    ) -> Optional[TrendAnalysis]:
        """Get trend analysis."""
        if not self._synthesizer:
            return None

        return self._synthesizer.analyze_trends(period_hours=period_hours)

    def generate_summary(
        self,
        period: str = "last_24h",
    ) -> Optional[IntelligenceSummary]:
        """
        Generate an executive intelligence summary.

        Args:
            period: Time period ("last_24h", "last_7d", "last_30d")

        Returns:
            IntelligenceSummary
        """
        if not self._synthesizer:
            return None

        summary = self._synthesizer.generate_summary(period=period)
        self._stats["summaries_generated"] += 1
        return summary

    def get_status(self) -> MemoryStatus:
        """Get current memory system status."""
        status = MemoryStatus(
            initialized=self._initialized,
            running=self._running,
        )

        if self._store:
            store_stats = self._store.get_stats()
            status.store_healthy = True
            status.total_entries = store_stats.get("total_entries", 0)
            status.hot_entries = store_stats.get("hot_entries", 0)

        if self._correlator:
            corr_stats = self._correlator.get_stats()
            status.correlator_active = True
            status.active_clusters = corr_stats.get("active_clusters", 0)

            # Get threat level from clusters
            clusters = self._correlator.get_active_clusters()
            if clusters:
                status.current_threat_level = max(c.threat_level for c in clusters)
                status.critical_threats = sum(
                    1 for c in clusters if c.threat_level.value >= ThreatLevel.CRITICAL.value
                )

        if self._synthesizer:
            synth_stats = self._synthesizer.get_stats()
            status.synthesizer_active = True
            status.detected_patterns = synth_stats.get("patterns_stored", 0)

        if self._baseline:
            baseline_stats = self._baseline.get_stats()
            status.baseline_ready = baseline_stats.get("is_ready", False)
            if status.baseline_ready:
                score = self._baseline.score()
                status.anomaly_score = score.overall_score

        if self._boundary:
            boundary_stats = self._boundary.get_stats()
            status.boundary_connected = boundary_stats.get("connected", False)

        status.healthy = (
            status.store_healthy and
            (not self.config.correlation_enabled or status.correlator_active) and
            (not self.config.synthesis_enabled or status.synthesizer_active)
        )

        return status

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics."""
        stats = dict(self._stats)

        if self._store:
            stats["store"] = self._store.get_stats()

        if self._correlator:
            stats["correlator"] = self._correlator.get_stats()

        if self._synthesizer:
            stats["synthesizer"] = self._synthesizer.get_stats()

        if self._baseline:
            stats["baseline"] = self._baseline.get_stats()

        if self._boundary:
            stats["boundary"] = self._boundary.get_stats()

        return stats

    def get_boundary_mode(self) -> Optional[BoundaryMode]:
        """Get current boundary daemon mode."""
        if self._boundary:
            return self._boundary.get_mode()
        return None

    def set_boundary_mode(self, mode: BoundaryMode) -> None:
        """Set boundary daemon mode (for local tracking)."""
        if self._boundary:
            self._boundary.set_mode(mode)

    def _on_correlation(self, result: CorrelationResult) -> None:
        """Handle correlation detection."""
        logger.info(
            f"Correlation detected: {result.result_id} - "
            f"{result.correlation_type.name} ({result.threat_level.name})"
        )

        # Callback for high threats
        if result.threat_level.value >= ThreatLevel.HIGH.value:
            if self.config.on_high_threat and result.cluster_id:
                cluster = self._correlator.get_cluster(result.cluster_id)
                if cluster:
                    self.config.on_high_threat(cluster)

    def _on_pattern(self, pattern: SynthesizedPattern) -> None:
        """Handle pattern detection."""
        logger.info(
            f"Pattern detected: {pattern.pattern_id} - "
            f"{pattern.pattern_type.name} (severity {pattern.severity})"
        )
        self._stats["patterns_detected"] += 1

        if self.config.on_pattern:
            self.config.on_pattern(pattern)

    def _on_anomaly(self, score: AnomalyScore) -> None:
        """Handle anomaly detection."""
        logger.warning(
            f"Anomaly detected: {score.score_id} - "
            f"score {score.overall_score:.2f}"
        )
        self._stats["anomalies_detected"] += 1

        if self.config.on_anomaly:
            self.config.on_anomaly(score)

    def _synthesis_loop(self) -> None:
        """Background synthesis loop."""
        while self._running:
            try:
                # Run pattern detection
                if self._synthesizer:
                    self._synthesizer.detect_patterns()

            except Exception as e:
                logger.error(f"Synthesis error: {e}")

            # Wait for synthesis interval or shutdown signal
            # Event.wait() returns True if event was set (shutdown), False on timeout
            synthesis_seconds = self.config.synthesis_interval_hours * 3600
            if self._shutdown_event.wait(timeout=synthesis_seconds):
                break  # Shutdown requested


def create_advanced_memory(
    config: Optional[MemoryConfig] = None,
    storage_path: Optional[str] = None,
    auto_start: bool = True,
) -> AdvancedMemoryManager:
    """
    Factory function to create an Advanced Memory Manager.

    Args:
        config: Memory configuration (uses defaults if None)
        storage_path: Override storage path
        auto_start: Start automatically

    Returns:
        Configured AdvancedMemoryManager
    """
    if config is None:
        config = MemoryConfig()

    if storage_path:
        config.storage_path = storage_path

    manager = AdvancedMemoryManager(config)

    if auto_start:
        manager.initialize()
        manager.start()

    return manager
