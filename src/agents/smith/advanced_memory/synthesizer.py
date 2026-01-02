"""
Pattern Synthesizer

Synthesizes patterns and generates intelligence summaries from security data.
Provides trend analysis, pattern detection, and actionable intelligence reports.

Features:
- Temporal trend analysis
- Pattern detection and classification
- Intelligence summary generation
- Threat landscape assessment
- Predictive insights
"""

import logging
import statistics
import threading
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from .store import IntelligenceEntry, IntelligenceType, SecurityIntelligenceStore
from .correlator import ThreatCluster, ThreatCorrelator, ThreatLevel

logger = logging.getLogger(__name__)


class PatternType(Enum):
    """Types of detected patterns."""

    TEMPORAL = auto()  # Time-based pattern (recurring, periodic)
    VOLUMETRIC = auto()  # Volume anomaly (spike, sustained increase)
    BEHAVIORAL = auto()  # Behavior change pattern
    GEOGRAPHIC = auto()  # Source/destination patterns
    SEVERITY_SHIFT = auto()  # Severity escalation
    SOURCE_EMERGENCE = auto()  # New source activity
    INDICATOR_CLUSTERING = auto()  # Indicator convergence
    MITRE_PROGRESSION = auto()  # Attack progression


class TrendDirection(Enum):
    """Direction of a trend."""

    STABLE = auto()
    INCREASING = auto()
    DECREASING = auto()
    VOLATILE = auto()
    SPIKE = auto()


@dataclass
class SynthesizedPattern:
    """A detected pattern from intelligence synthesis."""

    pattern_id: str
    pattern_type: PatternType
    detected_at: datetime
    confidence: float  # 0.0 - 1.0
    severity: int  # 1-5

    # Pattern details
    description: str
    evidence: List[str]  # Entry IDs supporting this pattern
    affected_sources: Set[str] = field(default_factory=set)
    affected_categories: Set[str] = field(default_factory=set)

    # Temporal info
    first_occurrence: Optional[datetime] = None
    last_occurrence: Optional[datetime] = None
    frequency: Optional[float] = None  # Events per hour

    # Metadata
    tags: Set[str] = field(default_factory=set)
    recommendations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "pattern_id": self.pattern_id,
            "pattern_type": self.pattern_type.name,
            "detected_at": self.detected_at.isoformat(),
            "confidence": self.confidence,
            "severity": self.severity,
            "description": self.description,
            "evidence": self.evidence,
            "affected_sources": list(self.affected_sources),
            "affected_categories": list(self.affected_categories),
            "first_occurrence": self.first_occurrence.isoformat() if self.first_occurrence else None,
            "last_occurrence": self.last_occurrence.isoformat() if self.last_occurrence else None,
            "frequency": self.frequency,
            "tags": list(self.tags),
            "recommendations": self.recommendations,
        }


@dataclass
class TrendAnalysis:
    """Analysis of trends over time."""

    analysis_id: str
    analyzed_at: datetime
    period_start: datetime
    period_end: datetime

    # Volume trends
    total_events: int
    events_by_hour: List[int]
    volume_direction: TrendDirection
    volume_change_percent: float

    # Severity trends
    avg_severity: float
    severity_direction: TrendDirection
    max_severity_hour: Optional[datetime] = None

    # Source activity
    active_sources: Set[str] = field(default_factory=set)
    new_sources: Set[str] = field(default_factory=set)
    dormant_sources: Set[str] = field(default_factory=set)

    # Category breakdown
    category_distribution: Dict[str, int] = field(default_factory=dict)
    category_trends: Dict[str, TrendDirection] = field(default_factory=dict)

    # Top indicators
    top_indicators: List[Tuple[str, int]] = field(default_factory=list)

    # Key insights
    insights: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "analysis_id": self.analysis_id,
            "analyzed_at": self.analyzed_at.isoformat(),
            "period_start": self.period_start.isoformat(),
            "period_end": self.period_end.isoformat(),
            "total_events": self.total_events,
            "events_by_hour": self.events_by_hour,
            "volume_direction": self.volume_direction.name,
            "volume_change_percent": self.volume_change_percent,
            "avg_severity": self.avg_severity,
            "severity_direction": self.severity_direction.name,
            "active_sources": list(self.active_sources),
            "new_sources": list(self.new_sources),
            "dormant_sources": list(self.dormant_sources),
            "category_distribution": self.category_distribution,
            "category_trends": {k: v.name for k, v in self.category_trends.items()},
            "top_indicators": self.top_indicators,
            "insights": self.insights,
        }


@dataclass
class IntelligenceSummary:
    """Executive intelligence summary."""

    summary_id: str
    generated_at: datetime
    period: str  # e.g., "last_24h", "last_7d"

    # Threat landscape
    threat_level: ThreatLevel
    threat_level_rationale: str
    active_threats: int
    critical_threats: int

    # Key metrics
    total_events: int
    events_by_severity: Dict[int, int]
    top_sources: List[Tuple[str, int]]
    top_categories: List[Tuple[str, int]]

    # Patterns and trends
    detected_patterns: List[str]  # Pattern IDs
    trend_summary: str

    # Active clusters
    active_clusters: List[str]  # Cluster IDs
    cluster_summary: str

    # MITRE coverage
    mitre_tactics_observed: List[str]
    mitre_coverage_percent: float

    # Recommendations
    priority_actions: List[str]
    investigation_items: List[str]

    # Narrative
    executive_summary: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "summary_id": self.summary_id,
            "generated_at": self.generated_at.isoformat(),
            "period": self.period,
            "threat_level": self.threat_level.name,
            "threat_level_rationale": self.threat_level_rationale,
            "active_threats": self.active_threats,
            "critical_threats": self.critical_threats,
            "total_events": self.total_events,
            "events_by_severity": self.events_by_severity,
            "top_sources": self.top_sources,
            "top_categories": self.top_categories,
            "detected_patterns": self.detected_patterns,
            "trend_summary": self.trend_summary,
            "active_clusters": self.active_clusters,
            "cluster_summary": self.cluster_summary,
            "mitre_tactics_observed": self.mitre_tactics_observed,
            "mitre_coverage_percent": self.mitre_coverage_percent,
            "priority_actions": self.priority_actions,
            "investigation_items": self.investigation_items,
            "executive_summary": self.executive_summary,
        }


class PatternSynthesizer:
    """
    Synthesizes patterns and generates intelligence from security data.

    Analyzes data from Boundary-SIEM, Boundary-Daemon, and internal events
    to identify patterns, trends, and generate actionable intelligence.
    """

    def __init__(
        self,
        store: SecurityIntelligenceStore,
        correlator: Optional[ThreatCorrelator] = None,
        on_pattern: Optional[Callable[[SynthesizedPattern], None]] = None,
    ):
        """
        Initialize the synthesizer.

        Args:
            store: Intelligence store
            correlator: Threat correlator for cluster analysis
            on_pattern: Callback for detected patterns
        """
        self.store = store
        self.correlator = correlator
        self.on_pattern = on_pattern

        # Pattern history
        self._patterns: Dict[str, SynthesizedPattern] = {}
        self._baselines: Dict[str, Dict[str, float]] = {}  # source -> metric -> value

        # Thread safety
        self._lock = threading.RLock()

        # Counters
        self._pattern_counter = 0
        self._analysis_counter = 0
        self._summary_counter = 0

        # Statistics
        self._stats = {
            "patterns_detected": 0,
            "analyses_performed": 0,
            "summaries_generated": 0,
        }

    def detect_patterns(
        self,
        lookback_hours: int = 24,
        min_confidence: float = 0.5,
    ) -> List[SynthesizedPattern]:
        """
        Detect patterns in recent intelligence.

        Args:
            lookback_hours: How far back to analyze
            min_confidence: Minimum confidence for patterns

        Returns:
            List of detected patterns
        """
        patterns = []
        since = datetime.now() - timedelta(hours=lookback_hours)

        with self._lock:
            # Get all entries in period
            entries = self.store.query(since=since, limit=10000)

            if len(entries) < 10:
                return []

            # Detect various pattern types
            patterns.extend(self._detect_volumetric_patterns(entries, since))
            patterns.extend(self._detect_severity_patterns(entries, since))
            patterns.extend(self._detect_source_patterns(entries, since))
            patterns.extend(self._detect_indicator_patterns(entries))
            patterns.extend(self._detect_temporal_patterns(entries, lookback_hours))

            # Filter by confidence
            patterns = [p for p in patterns if p.confidence >= min_confidence]

            # Store patterns
            for pattern in patterns:
                self._patterns[pattern.pattern_id] = pattern
                self._stats["patterns_detected"] += 1

                if self.on_pattern:
                    self.on_pattern(pattern)

        return patterns

    def _detect_volumetric_patterns(
        self,
        entries: List[IntelligenceEntry],
        since: datetime,
    ) -> List[SynthesizedPattern]:
        """Detect volume-based patterns."""
        patterns = []

        # Group by hour
        hourly_counts: Dict[str, int] = defaultdict(int)
        for entry in entries:
            hour_key = entry.timestamp.strftime("%Y-%m-%d %H:00")
            hourly_counts[hour_key] += 1

        if len(hourly_counts) < 2:
            return []

        counts = list(hourly_counts.values())
        avg_count = statistics.mean(counts)
        stdev = statistics.stdev(counts) if len(counts) > 1 else 0

        # Detect spikes (>2 std dev above mean)
        for hour, count in hourly_counts.items():
            if stdev > 0 and (count - avg_count) / stdev > 2:
                self._pattern_counter += 1
                patterns.append(SynthesizedPattern(
                    pattern_id=f"pat_{datetime.now().strftime('%Y%m%d')}_{self._pattern_counter:04d}",
                    pattern_type=PatternType.VOLUMETRIC,
                    detected_at=datetime.now(),
                    confidence=min(1.0, (count - avg_count) / (stdev * 3)) if stdev > 0 else 0.5,
                    severity=3,
                    description=f"Volume spike detected at {hour}: {count} events (avg: {avg_count:.1f})",
                    evidence=[],
                    first_occurrence=datetime.fromisoformat(hour),
                    frequency=count,
                    tags={"volume-spike", "anomaly"},
                    recommendations=[
                        "Investigate cause of volume spike",
                        "Check for automated attack or misconfiguration",
                    ],
                ))

        return patterns

    def _detect_severity_patterns(
        self,
        entries: List[IntelligenceEntry],
        since: datetime,
    ) -> List[SynthesizedPattern]:
        """Detect severity-based patterns."""
        patterns = []

        # Group by hour and calculate average severity
        hourly_severity: Dict[str, List[int]] = defaultdict(list)
        for entry in entries:
            hour_key = entry.timestamp.strftime("%Y-%m-%d %H:00")
            hourly_severity[hour_key].append(entry.severity)

        if len(hourly_severity) < 3:
            return []

        # Calculate hourly averages
        hourly_avgs = {
            hour: statistics.mean(sevs)
            for hour, sevs in hourly_severity.items()
        }

        sorted_hours = sorted(hourly_avgs.keys())
        recent_hours = sorted_hours[-3:]
        older_hours = sorted_hours[:-3] if len(sorted_hours) > 3 else sorted_hours[:1]

        recent_avg = statistics.mean([hourly_avgs[h] for h in recent_hours])
        older_avg = statistics.mean([hourly_avgs[h] for h in older_hours]) if older_hours else recent_avg

        # Detect severity escalation
        if recent_avg > older_avg * 1.3:  # 30% increase
            self._pattern_counter += 1
            patterns.append(SynthesizedPattern(
                pattern_id=f"pat_{datetime.now().strftime('%Y%m%d')}_{self._pattern_counter:04d}",
                pattern_type=PatternType.SEVERITY_SHIFT,
                detected_at=datetime.now(),
                confidence=min(1.0, (recent_avg - older_avg) / older_avg) if older_avg > 0 else 0.5,
                severity=4,
                description=f"Severity escalation detected: {older_avg:.2f} -> {recent_avg:.2f}",
                evidence=[e.entry_id for e in entries if e.severity >= 4][:20],
                tags={"severity-escalation", "alert"},
                recommendations=[
                    "Review high-severity events for common patterns",
                    "Consider increasing monitoring sensitivity",
                    "Prepare incident response if trend continues",
                ],
            ))

        return patterns

    def _detect_source_patterns(
        self,
        entries: List[IntelligenceEntry],
        since: datetime,
    ) -> List[SynthesizedPattern]:
        """Detect source-based patterns."""
        patterns = []

        # Count by source
        source_counts = Counter(e.source for e in entries)

        # Identify dominant sources
        total = len(entries)
        for source, count in source_counts.most_common(5):
            if count / total > 0.5:  # >50% from one source
                self._pattern_counter += 1
                patterns.append(SynthesizedPattern(
                    pattern_id=f"pat_{datetime.now().strftime('%Y%m%d')}_{self._pattern_counter:04d}",
                    pattern_type=PatternType.SOURCE_EMERGENCE,
                    detected_at=datetime.now(),
                    confidence=count / total,
                    severity=2,
                    description=f"Dominant source detected: {source} ({count}/{total} = {count/total:.1%})",
                    evidence=[e.entry_id for e in entries if e.source == source][:20],
                    affected_sources={source},
                    tags={"dominant-source"},
                    recommendations=[
                        f"Investigate high activity from {source}",
                        "Check if this is expected behavior or anomaly",
                    ],
                ))

        return patterns

    def _detect_indicator_patterns(
        self,
        entries: List[IntelligenceEntry],
    ) -> List[SynthesizedPattern]:
        """Detect indicator clustering patterns."""
        patterns = []

        # Count indicators
        indicator_counts: Counter = Counter()
        indicator_entries: Dict[str, List[str]] = defaultdict(list)

        for entry in entries:
            for ind in entry.indicators:
                indicator_counts[ind] += 1
                indicator_entries[ind].append(entry.entry_id)

        # Find heavily repeated indicators
        for indicator, count in indicator_counts.most_common(10):
            if count >= 5:  # Appears in 5+ entries
                self._pattern_counter += 1
                patterns.append(SynthesizedPattern(
                    pattern_id=f"pat_{datetime.now().strftime('%Y%m%d')}_{self._pattern_counter:04d}",
                    pattern_type=PatternType.INDICATOR_CLUSTERING,
                    detected_at=datetime.now(),
                    confidence=min(1.0, count / 20),
                    severity=3,
                    description=f"Indicator clustering: {indicator} appears in {count} events",
                    evidence=indicator_entries[indicator][:20],
                    tags={"indicator-cluster", "ioc"},
                    recommendations=[
                        f"Investigate indicator: {indicator}",
                        "Check threat intelligence for known associations",
                        "Consider blocking if malicious",
                    ],
                ))

        return patterns

    def _detect_temporal_patterns(
        self,
        entries: List[IntelligenceEntry],
        lookback_hours: int,
    ) -> List[SynthesizedPattern]:
        """Detect temporal patterns (recurring events)."""
        patterns = []

        # Group by hour of day
        hour_of_day_counts: Dict[int, int] = defaultdict(int)
        for entry in entries:
            hour_of_day_counts[entry.timestamp.hour] += 1

        if len(hour_of_day_counts) < 2:
            return []

        avg_per_hour = len(entries) / 24
        for hour, count in hour_of_day_counts.items():
            if count > avg_per_hour * 2:  # 2x average
                self._pattern_counter += 1
                patterns.append(SynthesizedPattern(
                    pattern_id=f"pat_{datetime.now().strftime('%Y%m%d')}_{self._pattern_counter:04d}",
                    pattern_type=PatternType.TEMPORAL,
                    detected_at=datetime.now(),
                    confidence=min(1.0, count / (avg_per_hour * 3)),
                    severity=2,
                    description=f"Temporal pattern: High activity at {hour:02d}:00 ({count} events)",
                    evidence=[],
                    frequency=count / lookback_hours,
                    tags={"temporal-pattern", "recurring"},
                    recommendations=[
                        f"Investigate activity pattern at {hour:02d}:00",
                        "Check for scheduled tasks or automated processes",
                    ],
                ))

        return patterns

    def analyze_trends(
        self,
        period_hours: int = 24,
        compare_to_hours: Optional[int] = None,
    ) -> TrendAnalysis:
        """
        Analyze trends over a time period.

        Args:
            period_hours: Hours to analyze
            compare_to_hours: Previous period to compare (default: same length)

        Returns:
            TrendAnalysis results
        """
        compare_to_hours = compare_to_hours or period_hours
        now = datetime.now()
        period_start = now - timedelta(hours=period_hours)
        compare_start = period_start - timedelta(hours=compare_to_hours)

        with self._lock:
            # Get entries for both periods
            current_entries = self.store.query(since=period_start, limit=10000)
            compare_entries = self.store.query(
                since=compare_start,
                until=period_start,
                limit=10000,
            )

            # Calculate hourly distribution
            hourly_counts: Dict[int, int] = defaultdict(int)
            for entry in current_entries:
                hour_offset = int((now - entry.timestamp).total_seconds() / 3600)
                hourly_counts[hour_offset] += 1

            events_by_hour = [hourly_counts.get(h, 0) for h in range(period_hours)]

            # Volume analysis
            current_volume = len(current_entries)
            compare_volume = len(compare_entries)

            if compare_volume > 0:
                volume_change = (current_volume - compare_volume) / compare_volume * 100
            else:
                volume_change = 100.0 if current_volume > 0 else 0.0

            if abs(volume_change) < 10:
                volume_direction = TrendDirection.STABLE
            elif volume_change > 50:
                volume_direction = TrendDirection.SPIKE
            elif volume_change > 10:
                volume_direction = TrendDirection.INCREASING
            else:
                volume_direction = TrendDirection.DECREASING

            # Severity analysis
            current_severities = [e.severity for e in current_entries]
            compare_severities = [e.severity for e in compare_entries]

            avg_severity = statistics.mean(current_severities) if current_severities else 0
            compare_avg_severity = statistics.mean(compare_severities) if compare_severities else avg_severity

            if abs(avg_severity - compare_avg_severity) < 0.2:
                severity_direction = TrendDirection.STABLE
            elif avg_severity > compare_avg_severity:
                severity_direction = TrendDirection.INCREASING
            else:
                severity_direction = TrendDirection.DECREASING

            # Source analysis
            current_sources = {e.source for e in current_entries}
            compare_sources = {e.source for e in compare_entries}
            new_sources = current_sources - compare_sources
            dormant_sources = compare_sources - current_sources

            # Category analysis
            category_counts: Counter = Counter(e.category for e in current_entries)
            compare_category_counts: Counter = Counter(e.category for e in compare_entries)

            category_trends = {}
            for cat in category_counts:
                current = category_counts[cat]
                previous = compare_category_counts.get(cat, 0)
                if previous == 0:
                    category_trends[cat] = TrendDirection.SPIKE
                elif abs(current - previous) / max(previous, 1) < 0.1:
                    category_trends[cat] = TrendDirection.STABLE
                elif current > previous:
                    category_trends[cat] = TrendDirection.INCREASING
                else:
                    category_trends[cat] = TrendDirection.DECREASING

            # Top indicators
            indicator_counts: Counter = Counter()
            for entry in current_entries:
                indicator_counts.update(entry.indicators)
            top_indicators = indicator_counts.most_common(10)

            # Generate insights
            insights = []
            if volume_direction == TrendDirection.SPIKE:
                insights.append(f"Volume spike: {volume_change:.0f}% increase")
            if new_sources:
                insights.append(f"New sources active: {', '.join(list(new_sources)[:3])}")
            if severity_direction == TrendDirection.INCREASING:
                insights.append(f"Severity trending up: {compare_avg_severity:.2f} -> {avg_severity:.2f}")

            for cat, trend in category_trends.items():
                if trend == TrendDirection.SPIKE:
                    insights.append(f"Category spike: {cat}")

            self._analysis_counter += 1
            self._stats["analyses_performed"] += 1

            return TrendAnalysis(
                analysis_id=f"trend_{now.strftime('%Y%m%d_%H%M%S')}_{self._analysis_counter:04d}",
                analyzed_at=now,
                period_start=period_start,
                period_end=now,
                total_events=current_volume,
                events_by_hour=events_by_hour,
                volume_direction=volume_direction,
                volume_change_percent=volume_change,
                avg_severity=avg_severity,
                severity_direction=severity_direction,
                active_sources=current_sources,
                new_sources=new_sources,
                dormant_sources=dormant_sources,
                category_distribution=dict(category_counts),
                category_trends=category_trends,
                top_indicators=top_indicators,
                insights=insights,
            )

    def generate_summary(
        self,
        period: str = "last_24h",
    ) -> IntelligenceSummary:
        """
        Generate an executive intelligence summary.

        Args:
            period: Time period ("last_24h", "last_7d", "last_30d")

        Returns:
            IntelligenceSummary
        """
        now = datetime.now()

        # Parse period
        period_hours = {
            "last_24h": 24,
            "last_7d": 168,
            "last_30d": 720,
        }.get(period, 24)

        since = now - timedelta(hours=period_hours)

        with self._lock:
            # Get entries and patterns
            entries = self.store.query(since=since, limit=50000)
            patterns = [p for p in self._patterns.values() if p.detected_at >= since]

            # Get clusters if correlator available
            clusters = []
            if self.correlator:
                clusters = self.correlator.get_active_clusters(since=since)

            # Calculate metrics
            total_events = len(entries)
            severity_counts: Counter = Counter(e.severity for e in entries)
            source_counts: Counter = Counter(e.source for e in entries)
            category_counts: Counter = Counter(e.category for e in entries)

            # MITRE coverage
            all_tactics: Set[str] = set()
            for entry in entries:
                all_tactics.update(entry.mitre_tactics)

            mitre_total = 14  # Total MITRE tactics
            mitre_coverage = len(all_tactics) / mitre_total * 100

            # Determine threat level
            critical_count = sum(1 for c in clusters if c.threat_level.value >= ThreatLevel.CRITICAL.value)
            high_count = sum(1 for c in clusters if c.threat_level.value >= ThreatLevel.HIGH.value)
            high_sev_events = severity_counts.get(4, 0) + severity_counts.get(5, 0)

            if critical_count > 0:
                threat_level = ThreatLevel.CRITICAL
                rationale = f"{critical_count} critical threat cluster(s) active"
            elif high_count > 2:
                threat_level = ThreatLevel.HIGH
                rationale = f"{high_count} high-threat clusters detected"
            elif high_sev_events > total_events * 0.1:
                threat_level = ThreatLevel.HIGH
                rationale = f"High-severity events exceed 10% ({high_sev_events}/{total_events})"
            elif high_sev_events > 10:
                threat_level = ThreatLevel.MEDIUM
                rationale = f"{high_sev_events} high-severity events detected"
            else:
                threat_level = ThreatLevel.LOW
                rationale = "No significant threats detected"

            # Generate trend summary
            trend = self.analyze_trends(period_hours=min(period_hours, 168))
            trend_summary = f"Volume: {trend.volume_direction.name.lower()} ({trend.volume_change_percent:+.0f}%). "
            trend_summary += f"Severity: {trend.severity_direction.name.lower()}. "
            if trend.new_sources:
                trend_summary += f"New sources: {len(trend.new_sources)}."

            # Generate cluster summary
            if clusters:
                cluster_summary = f"{len(clusters)} active threat clusters. "
                cluster_summary += f"Critical: {critical_count}, High: {high_count - critical_count}. "
                if clusters[0].mitre_tactics:
                    cluster_summary += f"Top tactics: {', '.join(clusters[0].mitre_tactics[:3])}."
            else:
                cluster_summary = "No active threat clusters."

            # Generate priority actions
            priority_actions = []
            if critical_count > 0:
                priority_actions.append("IMMEDIATE: Address critical threat clusters")
            if high_sev_events > 20:
                priority_actions.append("Review high-severity events for common patterns")
            if trend.volume_direction == TrendDirection.SPIKE:
                priority_actions.append("Investigate volume spike")
            if "exfiltration" in all_tactics:
                priority_actions.append("Check for potential data exfiltration")
            if len(trend.new_sources) > 3:
                priority_actions.append(f"Investigate new activity sources ({len(trend.new_sources)} new)")

            # Generate investigation items
            investigation_items = []
            for pattern in patterns[:5]:
                investigation_items.append(f"[{pattern.pattern_type.name}] {pattern.description[:100]}")

            # Generate executive summary
            exec_summary = f"Over the {period.replace('_', ' ').replace('last', 'last ')}, "
            exec_summary += f"Agent Smith processed {total_events:,} security events. "
            exec_summary += f"Current threat level: {threat_level.name}. {rationale}. "
            exec_summary += f"Detected {len(patterns)} patterns and {len(clusters)} threat clusters. "
            if mitre_coverage > 50:
                exec_summary += f"Adversary techniques span {mitre_coverage:.0f}% of MITRE ATT&CK framework. "
            exec_summary += "Key sources: " + ", ".join([s for s, _ in source_counts.most_common(3)]) + "."

            self._summary_counter += 1
            self._stats["summaries_generated"] += 1

            return IntelligenceSummary(
                summary_id=f"summary_{now.strftime('%Y%m%d_%H%M%S')}_{self._summary_counter:04d}",
                generated_at=now,
                period=period,
                threat_level=threat_level,
                threat_level_rationale=rationale,
                active_threats=len(clusters),
                critical_threats=critical_count,
                total_events=total_events,
                events_by_severity=dict(severity_counts),
                top_sources=source_counts.most_common(10),
                top_categories=category_counts.most_common(10),
                detected_patterns=[p.pattern_id for p in patterns],
                trend_summary=trend_summary,
                active_clusters=[c.cluster_id for c in clusters],
                cluster_summary=cluster_summary,
                mitre_tactics_observed=list(all_tactics),
                mitre_coverage_percent=mitre_coverage,
                priority_actions=priority_actions[:5],
                investigation_items=investigation_items[:10],
                executive_summary=exec_summary,
            )

    def get_patterns(
        self,
        pattern_type: Optional[PatternType] = None,
        min_severity: int = 1,
        since: Optional[datetime] = None,
    ) -> List[SynthesizedPattern]:
        """Get detected patterns."""
        patterns = list(self._patterns.values())

        if pattern_type:
            patterns = [p for p in patterns if p.pattern_type == pattern_type]

        if min_severity > 1:
            patterns = [p for p in patterns if p.severity >= min_severity]

        if since:
            patterns = [p for p in patterns if p.detected_at >= since]

        return sorted(patterns, key=lambda p: (p.severity, p.detected_at), reverse=True)

    def get_stats(self) -> Dict[str, Any]:
        """Get synthesizer statistics."""
        return {
            **self._stats,
            "patterns_stored": len(self._patterns),
        }


def create_pattern_synthesizer(
    store: SecurityIntelligenceStore,
    correlator: Optional[ThreatCorrelator] = None,
    on_pattern: Optional[Callable[[SynthesizedPattern], None]] = None,
) -> PatternSynthesizer:
    """
    Factory function to create a pattern synthesizer.

    Args:
        store: Intelligence store
        correlator: Threat correlator
        on_pattern: Callback for patterns

    Returns:
        Configured PatternSynthesizer
    """
    return PatternSynthesizer(
        store=store,
        correlator=correlator,
        on_pattern=on_pattern,
    )
