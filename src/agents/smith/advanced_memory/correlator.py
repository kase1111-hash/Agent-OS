"""
Threat Correlator

Correlates security events across multiple sources to detect complex attack patterns.
Inspired by Boundary-SIEM's 103 detection rules and correlation capabilities.

Features:
- Cross-source event correlation
- Temporal sequence detection
- Indicator-based clustering
- MITRE ATT&CK kill chain tracking
- Configurable correlation rules
"""

import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set

from .store import IntelligenceEntry, IntelligenceType, SecurityIntelligenceStore

logger = logging.getLogger(__name__)


class CorrelationType(Enum):
    """Types of correlation."""

    TEMPORAL = auto()  # Events within time window
    INDICATOR = auto()  # Shared indicators (IP, hash, etc.)
    SOURCE = auto()  # Same source
    MITRE_CHAIN = auto()  # MITRE ATT&CK kill chain progression
    BEHAVIORAL = auto()  # Behavioral pattern match
    CUSTOM = auto()  # Custom rule-based


class ThreatLevel(Enum):
    """Threat assessment levels."""

    UNKNOWN = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4
    CONFIRMED = 5  # Confirmed active threat


@dataclass
class CorrelationRule:
    """A correlation rule definition."""

    rule_id: str
    name: str
    description: str
    correlation_type: CorrelationType
    enabled: bool = True

    # Matching criteria
    source_types: List[IntelligenceType] = field(default_factory=list)
    categories: List[str] = field(default_factory=list)
    min_severity: int = 1
    required_indicators: List[str] = field(default_factory=list)  # e.g., ["ip", "domain"]
    required_tactics: List[str] = field(default_factory=list)  # MITRE tactics

    # Temporal settings
    time_window_minutes: int = 60
    min_events: int = 2
    max_events: int = 1000

    # Scoring
    base_score: int = 50
    severity_weight: float = 1.5
    source_diversity_bonus: int = 10  # Bonus per unique source

    # Output
    output_threat_level: ThreatLevel = ThreatLevel.MEDIUM
    tags: Set[str] = field(default_factory=set)

    def matches(self, entry: IntelligenceEntry) -> bool:
        """Check if an entry matches this rule's criteria."""
        if self.source_types and entry.entry_type not in self.source_types:
            return False

        if self.categories and entry.category not in self.categories:
            return False

        if entry.severity < self.min_severity:
            return False

        if self.required_tactics:
            if not any(tac in entry.mitre_tactics for tac in self.required_tactics):
                return False

        return True


@dataclass
class ThreatCluster:
    """A cluster of correlated threats."""

    cluster_id: str
    created_at: datetime
    last_updated: datetime
    threat_level: ThreatLevel
    score: float

    # Entries in this cluster
    entry_ids: List[str] = field(default_factory=list)
    entry_count: int = 0

    # Aggregated data
    sources: Set[str] = field(default_factory=set)
    categories: Set[str] = field(default_factory=set)
    mitre_tactics: List[str] = field(default_factory=list)  # Ordered by first seen
    indicators: Set[str] = field(default_factory=set)

    # Timeline
    first_seen: Optional[datetime] = None
    last_seen: Optional[datetime] = None

    # Related rules
    matched_rules: List[str] = field(default_factory=list)
    tags: Set[str] = field(default_factory=set)

    # Description
    summary: str = ""

    def add_entry(self, entry: IntelligenceEntry) -> None:
        """Add an entry to this cluster."""
        if entry.entry_id not in self.entry_ids:
            self.entry_ids.append(entry.entry_id)
            self.entry_count += 1

        self.sources.add(entry.source)
        self.categories.add(entry.category)
        self.indicators.update(entry.indicators)
        self.tags.update(entry.tags)

        # Track MITRE progression
        for tactic in entry.mitre_tactics:
            if tactic not in self.mitre_tactics:
                self.mitre_tactics.append(tactic)

        # Update timeline
        if self.first_seen is None or entry.timestamp < self.first_seen:
            self.first_seen = entry.timestamp
        if self.last_seen is None or entry.timestamp > self.last_seen:
            self.last_seen = entry.timestamp

        self.last_updated = datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "cluster_id": self.cluster_id,
            "created_at": self.created_at.isoformat(),
            "last_updated": self.last_updated.isoformat(),
            "threat_level": self.threat_level.name,
            "score": self.score,
            "entry_count": self.entry_count,
            "sources": list(self.sources),
            "categories": list(self.categories),
            "mitre_tactics": self.mitre_tactics,
            "indicators": list(self.indicators),
            "first_seen": self.first_seen.isoformat() if self.first_seen else None,
            "last_seen": self.last_seen.isoformat() if self.last_seen else None,
            "matched_rules": self.matched_rules,
            "tags": list(self.tags),
            "summary": self.summary,
        }


@dataclass
class CorrelationResult:
    """Result of a correlation analysis."""

    result_id: str
    correlation_type: CorrelationType
    rule_id: Optional[str]
    timestamp: datetime
    threat_level: ThreatLevel
    confidence: float  # 0.0 - 1.0

    # Correlated entries
    entry_ids: List[str]
    entry_count: int

    # Analysis
    shared_indicators: List[str]
    mitre_chain: List[str]
    source_diversity: int

    # Output
    summary: str
    recommendations: List[str] = field(default_factory=list)
    cluster_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "result_id": self.result_id,
            "correlation_type": self.correlation_type.name,
            "rule_id": self.rule_id,
            "timestamp": self.timestamp.isoformat(),
            "threat_level": self.threat_level.name,
            "confidence": self.confidence,
            "entry_ids": self.entry_ids,
            "entry_count": self.entry_count,
            "shared_indicators": self.shared_indicators,
            "mitre_chain": self.mitre_chain,
            "source_diversity": self.source_diversity,
            "summary": self.summary,
            "recommendations": self.recommendations,
            "cluster_id": self.cluster_id,
        }


# Default MITRE ATT&CK kill chain for chain detection
MITRE_KILL_CHAIN = [
    "reconnaissance",
    "resource-development",
    "initial-access",
    "execution",
    "persistence",
    "privilege-escalation",
    "defense-evasion",
    "credential-access",
    "discovery",
    "lateral-movement",
    "collection",
    "command-and-control",
    "exfiltration",
    "impact",
]


class ThreatCorrelator:
    """
    Correlates security events to detect complex threats.

    Integrates with Boundary-SIEM and Boundary-Daemon events to provide
    comprehensive threat detection across the security perimeter.
    """

    def __init__(
        self,
        store: SecurityIntelligenceStore,
        on_correlation: Optional[Callable[[CorrelationResult], None]] = None,
        on_cluster_update: Optional[Callable[[ThreatCluster], None]] = None,
    ):
        """
        Initialize the correlator.

        Args:
            store: Intelligence store to query
            on_correlation: Callback for new correlations
            on_cluster_update: Callback for cluster updates
        """
        self.store = store
        self.on_correlation = on_correlation
        self.on_cluster_update = on_cluster_update

        # Correlation rules
        self._rules: Dict[str, CorrelationRule] = {}
        self._load_default_rules()

        # Active threat clusters
        self._clusters: Dict[str, ThreatCluster] = {}
        self._indicator_to_cluster: Dict[str, str] = {}  # Quick lookup

        # Thread safety
        self._lock = threading.RLock()

        # Statistics
        self._stats = {
            "correlations_found": 0,
            "clusters_created": 0,
            "clusters_merged": 0,
            "rules_triggered": 0,
        }

        # Result counter
        self._result_counter = 0

    def _load_default_rules(self) -> None:
        """Load default correlation rules."""
        default_rules = [
            CorrelationRule(
                rule_id="CORR-001",
                name="Multi-Source Attack Pattern",
                description="Events from multiple sources with shared indicators",
                correlation_type=CorrelationType.INDICATOR,
                min_severity=2,
                min_events=3,
                time_window_minutes=30,
                source_diversity_bonus=15,
                output_threat_level=ThreatLevel.HIGH,
                tags={"multi-source", "coordinated"},
            ),
            CorrelationRule(
                rule_id="CORR-002",
                name="Kill Chain Progression",
                description="MITRE ATT&CK kill chain progression detected",
                correlation_type=CorrelationType.MITRE_CHAIN,
                required_tactics=["initial-access", "execution"],
                min_events=3,
                time_window_minutes=120,
                base_score=70,
                output_threat_level=ThreatLevel.CRITICAL,
                tags={"kill-chain", "apt"},
            ),
            CorrelationRule(
                rule_id="CORR-003",
                name="Boundary Violation Cluster",
                description="Multiple boundary daemon violations in short time",
                correlation_type=CorrelationType.TEMPORAL,
                source_types=[IntelligenceType.BOUNDARY_EVENT, IntelligenceType.TRIPWIRE_ALERT],
                min_severity=3,
                min_events=5,
                time_window_minutes=15,
                output_threat_level=ThreatLevel.HIGH,
                tags={"boundary-violation", "policy-breach"},
            ),
            CorrelationRule(
                rule_id="CORR-004",
                name="Authentication Attack Sequence",
                description="Authentication failures followed by success or privilege escalation",
                correlation_type=CorrelationType.BEHAVIORAL,
                categories=["authentication", "authorization"],
                min_events=3,
                time_window_minutes=60,
                required_tactics=["credential-access", "privilege-escalation"],
                output_threat_level=ThreatLevel.HIGH,
                tags={"brute-force", "credential-attack"},
            ),
            CorrelationRule(
                rule_id="CORR-005",
                name="Data Exfiltration Pattern",
                description="Indicators of data exfiltration activity",
                correlation_type=CorrelationType.BEHAVIORAL,
                required_tactics=["collection", "exfiltration"],
                min_severity=3,
                min_events=2,
                time_window_minutes=60,
                base_score=80,
                output_threat_level=ThreatLevel.CRITICAL,
                tags={"data-exfiltration", "data-theft"},
            ),
            CorrelationRule(
                rule_id="CORR-006",
                name="SIEM-Daemon Correlation",
                description="Correlated events between Boundary-SIEM and Boundary-Daemon",
                correlation_type=CorrelationType.INDICATOR,
                source_types=[IntelligenceType.SIEM_EVENT, IntelligenceType.BOUNDARY_EVENT],
                min_events=2,
                time_window_minutes=30,
                source_diversity_bonus=20,
                output_threat_level=ThreatLevel.MEDIUM,
                tags={"cross-platform", "siem-daemon"},
            ),
            CorrelationRule(
                rule_id="CORR-007",
                name="Persistence Mechanism Detection",
                description="Multiple persistence indicators detected",
                correlation_type=CorrelationType.MITRE_CHAIN,
                required_tactics=["persistence"],
                categories=["file-modification", "registry", "scheduled-task"],
                min_events=2,
                time_window_minutes=60,
                output_threat_level=ThreatLevel.HIGH,
                tags={"persistence", "implant"},
            ),
        ]

        for rule in default_rules:
            self._rules[rule.rule_id] = rule

    def add_rule(self, rule: CorrelationRule) -> None:
        """Add or update a correlation rule."""
        with self._lock:
            self._rules[rule.rule_id] = rule
            logger.info(f"Added correlation rule: {rule.rule_id} - {rule.name}")

    def remove_rule(self, rule_id: str) -> bool:
        """Remove a correlation rule."""
        with self._lock:
            if rule_id in self._rules:
                del self._rules[rule_id]
                logger.info(f"Removed correlation rule: {rule_id}")
                return True
            return False

    def correlate(
        self,
        entry: IntelligenceEntry,
        lookback_minutes: int = 60,
    ) -> List[CorrelationResult]:
        """
        Correlate a new entry with existing intelligence.

        Args:
            entry: New entry to correlate
            lookback_minutes: How far back to look for correlations

        Returns:
            List of correlation results
        """
        results = []
        since = datetime.now() - timedelta(minutes=lookback_minutes)

        with self._lock:
            # Check each enabled rule
            for rule in self._rules.values():
                if not rule.enabled:
                    continue

                if not rule.matches(entry):
                    continue

                # Find matching entries based on correlation type
                if rule.correlation_type == CorrelationType.INDICATOR:
                    result = self._correlate_by_indicators(entry, rule, since)
                elif rule.correlation_type == CorrelationType.TEMPORAL:
                    result = self._correlate_by_time(entry, rule, since)
                elif rule.correlation_type == CorrelationType.MITRE_CHAIN:
                    result = self._correlate_by_mitre(entry, rule, since)
                elif rule.correlation_type == CorrelationType.BEHAVIORAL:
                    result = self._correlate_behavioral(entry, rule, since)
                else:
                    continue

                if result:
                    results.append(result)
                    self._stats["rules_triggered"] += 1

                    # Update or create cluster
                    self._update_clusters(result, entry)

                    # Notify callback
                    if self.on_correlation:
                        self.on_correlation(result)

            # Also check for indicator-based cluster membership
            self._check_cluster_membership(entry)

        return results

    def _correlate_by_indicators(
        self,
        entry: IntelligenceEntry,
        rule: CorrelationRule,
        since: datetime,
    ) -> Optional[CorrelationResult]:
        """Correlate by shared indicators."""
        if not entry.indicators:
            return None

        # Find entries with shared indicators (with pagination limit)
        related_entries = self.store.query(
            indicators=entry.indicators,
            since=since,
            severity_min=rule.min_severity,
            limit=rule.max_events,  # Prevent unbounded result sets
        )

        # Exclude self
        related_entries = [e for e in related_entries if e.entry_id != entry.entry_id]

        if len(related_entries) < rule.min_events - 1:
            return None

        # Find shared indicators
        shared = set(entry.indicators)
        for e in related_entries:
            shared &= set(e.indicators)

        if not shared:
            return None

        # Calculate score and confidence
        unique_sources = {e.source for e in related_entries} | {entry.source}
        source_diversity = len(unique_sources)

        score = rule.base_score
        score += len(related_entries) * 5
        score += source_diversity * rule.source_diversity_bonus
        score += sum(e.severity for e in related_entries) * rule.severity_weight

        confidence = min(1.0, (len(related_entries) / rule.max_events) + (source_diversity * 0.1))

        # Generate summary
        summary = (
            f"Correlated {len(related_entries) + 1} events with shared indicators: "
            f"{', '.join(list(shared)[:5])}. "
            f"Sources: {', '.join(unique_sources)}."
        )

        self._result_counter += 1
        return CorrelationResult(
            result_id=f"corr_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{self._result_counter:04d}",
            correlation_type=CorrelationType.INDICATOR,
            rule_id=rule.rule_id,
            timestamp=datetime.now(),
            threat_level=rule.output_threat_level,
            confidence=confidence,
            entry_ids=[entry.entry_id] + [e.entry_id for e in related_entries],
            entry_count=len(related_entries) + 1,
            shared_indicators=list(shared),
            mitre_chain=entry.mitre_tactics,
            source_diversity=source_diversity,
            summary=summary,
            recommendations=self._generate_recommendations(rule, related_entries),
        )

    def _correlate_by_time(
        self,
        entry: IntelligenceEntry,
        rule: CorrelationRule,
        since: datetime,
    ) -> Optional[CorrelationResult]:
        """Correlate by temporal proximity."""
        # Get entries in time window matching criteria (with pagination limit)
        related_entries = self.store.query(
            entry_types=rule.source_types if rule.source_types else None,
            categories=rule.categories if rule.categories else None,
            since=since,
            severity_min=rule.min_severity,
            limit=rule.max_events,  # Prevent unbounded result sets
        )

        related_entries = [e for e in related_entries if e.entry_id != entry.entry_id]

        if len(related_entries) < rule.min_events - 1:
            return None

        # Calculate temporal clustering
        time_diffs = [
            abs((e.timestamp - entry.timestamp).total_seconds())
            for e in related_entries
        ]
        avg_diff = sum(time_diffs) / len(time_diffs) if time_diffs else 0

        # Tighter clustering = higher confidence
        confidence = max(0.0, 1.0 - (avg_diff / (rule.time_window_minutes * 60)))

        unique_sources = {e.source for e in related_entries} | {entry.source}
        source_diversity = len(unique_sources)

        score = rule.base_score
        score += len(related_entries) * 3
        score += source_diversity * rule.source_diversity_bonus
        score += (1.0 - (avg_diff / (rule.time_window_minutes * 60))) * 20

        # Collect all indicators
        all_indicators = set(entry.indicators)
        for e in related_entries:
            all_indicators.update(e.indicators)

        summary = (
            f"Temporal cluster of {len(related_entries) + 1} events within "
            f"{rule.time_window_minutes} minutes. "
            f"Average time delta: {avg_diff:.1f}s."
        )

        self._result_counter += 1
        return CorrelationResult(
            result_id=f"corr_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{self._result_counter:04d}",
            correlation_type=CorrelationType.TEMPORAL,
            rule_id=rule.rule_id,
            timestamp=datetime.now(),
            threat_level=rule.output_threat_level,
            confidence=confidence,
            entry_ids=[entry.entry_id] + [e.entry_id for e in related_entries],
            entry_count=len(related_entries) + 1,
            shared_indicators=list(all_indicators)[:20],
            mitre_chain=[],
            source_diversity=source_diversity,
            summary=summary,
            recommendations=self._generate_recommendations(rule, related_entries),
        )

    def _correlate_by_mitre(
        self,
        entry: IntelligenceEntry,
        rule: CorrelationRule,
        since: datetime,
    ) -> Optional[CorrelationResult]:
        """Correlate by MITRE ATT&CK kill chain progression."""
        if not entry.mitre_tactics:
            return None

        # Get entries with any MITRE tactics (with pagination limit)
        related_entries = self.store.query(
            mitre_tactics=MITRE_KILL_CHAIN,  # Any kill chain tactic
            since=since,
            severity_min=rule.min_severity,
            limit=rule.max_events,  # Prevent unbounded result sets
        )

        related_entries = [e for e in related_entries if e.entry_id != entry.entry_id]

        if len(related_entries) < rule.min_events - 1:
            return None

        # Build observed kill chain
        all_tactics = set(entry.mitre_tactics)
        for e in related_entries:
            all_tactics.update(e.mitre_tactics)

        # Check for chain progression
        observed_chain = []
        for tactic in MITRE_KILL_CHAIN:
            if tactic in all_tactics:
                observed_chain.append(tactic)

        # Need at least 2 tactics in sequence
        if len(observed_chain) < 2:
            return None

        # Check if required tactics are present
        if rule.required_tactics:
            if not all(tac in all_tactics for tac in rule.required_tactics):
                return None

        # Calculate chain coverage
        chain_coverage = len(observed_chain) / len(MITRE_KILL_CHAIN)
        confidence = min(1.0, chain_coverage * 2)  # Full coverage = 100% confidence

        # Higher threat for longer chains or late-stage tactics
        late_stage_tactics = {"lateral-movement", "collection", "command-and-control", "exfiltration", "impact"}
        has_late_stage = bool(all_tactics & late_stage_tactics)

        threat_level = rule.output_threat_level
        if has_late_stage and threat_level.value < ThreatLevel.CRITICAL.value:
            threat_level = ThreatLevel.CRITICAL

        unique_sources = {e.source for e in related_entries} | {entry.source}

        summary = (
            f"MITRE ATT&CK kill chain progression detected: {' -> '.join(observed_chain)}. "
            f"Coverage: {chain_coverage:.0%}. "
            f"Late-stage tactics: {'Yes' if has_late_stage else 'No'}."
        )

        self._result_counter += 1
        return CorrelationResult(
            result_id=f"corr_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{self._result_counter:04d}",
            correlation_type=CorrelationType.MITRE_CHAIN,
            rule_id=rule.rule_id,
            timestamp=datetime.now(),
            threat_level=threat_level,
            confidence=confidence,
            entry_ids=[entry.entry_id] + [e.entry_id for e in related_entries],
            entry_count=len(related_entries) + 1,
            shared_indicators=list({ind for e in related_entries for ind in e.indicators} | set(entry.indicators))[:20],
            mitre_chain=observed_chain,
            source_diversity=len(unique_sources),
            summary=summary,
            recommendations=[
                f"Investigate kill chain progression starting with '{observed_chain[0]}'",
                "Review all associated indicators for IoC extraction",
                "Consider containment if late-stage tactics observed",
            ],
        )

    def _correlate_behavioral(
        self,
        entry: IntelligenceEntry,
        rule: CorrelationRule,
        since: datetime,
    ) -> Optional[CorrelationResult]:
        """Correlate by behavioral patterns."""
        # Get entries matching behavioral criteria (with pagination limit)
        related_entries = self.store.query(
            categories=rule.categories if rule.categories else None,
            mitre_tactics=rule.required_tactics if rule.required_tactics else None,
            since=since,
            severity_min=rule.min_severity,
            limit=rule.max_events,  # Prevent unbounded result sets
        )

        related_entries = [e for e in related_entries if e.entry_id != entry.entry_id]

        if len(related_entries) < rule.min_events - 1:
            return None

        # Check for behavioral pattern (e.g., auth failures -> success)
        unique_sources = {e.source for e in related_entries} | {entry.source}
        unique_categories = {e.category for e in related_entries} | {entry.category}

        # Simple scoring
        score = rule.base_score
        score += len(related_entries) * 5
        score += len(unique_categories) * 10  # More categories = more complex pattern

        confidence = min(1.0, len(related_entries) / rule.max_events)

        summary = (
            f"Behavioral pattern detected: {len(related_entries) + 1} events across "
            f"{len(unique_categories)} categories from {len(unique_sources)} sources."
        )

        self._result_counter += 1
        return CorrelationResult(
            result_id=f"corr_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{self._result_counter:04d}",
            correlation_type=CorrelationType.BEHAVIORAL,
            rule_id=rule.rule_id,
            timestamp=datetime.now(),
            threat_level=rule.output_threat_level,
            confidence=confidence,
            entry_ids=[entry.entry_id] + [e.entry_id for e in related_entries],
            entry_count=len(related_entries) + 1,
            shared_indicators=list({ind for e in related_entries for ind in e.indicators} | set(entry.indicators))[:20],
            mitre_chain=[],
            source_diversity=len(unique_sources),
            summary=summary,
            recommendations=self._generate_recommendations(rule, related_entries),
        )

    def _generate_recommendations(
        self,
        rule: CorrelationRule,
        entries: List[IntelligenceEntry],
    ) -> List[str]:
        """Generate recommendations based on correlation."""
        recommendations = []

        # Based on threat level
        if rule.output_threat_level >= ThreatLevel.CRITICAL:
            recommendations.append("IMMEDIATE: Initiate incident response procedures")
            recommendations.append("Consider isolating affected systems")
        elif rule.output_threat_level >= ThreatLevel.HIGH:
            recommendations.append("Escalate to security team for investigation")
            recommendations.append("Preserve evidence and logs")

        # Based on MITRE tactics
        all_tactics = set()
        for e in entries:
            all_tactics.update(e.mitre_tactics)

        if "exfiltration" in all_tactics:
            recommendations.append("Check for data loss and network exfiltration")
        if "persistence" in all_tactics:
            recommendations.append("Scan for persistence mechanisms and implants")
        if "credential-access" in all_tactics:
            recommendations.append("Review compromised accounts and rotate credentials")

        # Based on tags
        if "boundary-violation" in rule.tags:
            recommendations.append("Review boundary daemon policy decisions")
        if "multi-source" in rule.tags:
            recommendations.append("Investigate correlation across security tools")

        return recommendations[:5]  # Limit to 5 recommendations

    def _update_clusters(
        self,
        result: CorrelationResult,
        entry: IntelligenceEntry,
    ) -> None:
        """Update or create threat clusters based on correlation."""
        # Check if any indicators already belong to a cluster
        existing_cluster_id = None
        for indicator in entry.indicators:
            if indicator in self._indicator_to_cluster:
                existing_cluster_id = self._indicator_to_cluster[indicator]
                break

        if existing_cluster_id and existing_cluster_id in self._clusters:
            # Update existing cluster
            cluster = self._clusters[existing_cluster_id]
            cluster.add_entry(entry)
            if result.rule_id and result.rule_id not in cluster.matched_rules:
                cluster.matched_rules.append(result.rule_id)
            cluster.score = max(cluster.score, 50 + result.confidence * 50)
            cluster.threat_level = max(cluster.threat_level, result.threat_level, key=lambda x: x.value)

            # Update summary
            cluster.summary = (
                f"Cluster of {cluster.entry_count} correlated events from "
                f"{len(cluster.sources)} sources. "
                f"MITRE: {' -> '.join(cluster.mitre_tactics[:5])}."
            )

            result.cluster_id = cluster.cluster_id

            if self.on_cluster_update:
                self.on_cluster_update(cluster)

        else:
            # Create new cluster
            cluster_id = f"cluster_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{len(self._clusters):04d}"

            cluster = ThreatCluster(
                cluster_id=cluster_id,
                created_at=datetime.now(),
                last_updated=datetime.now(),
                threat_level=result.threat_level,
                score=50 + result.confidence * 50,
            )
            cluster.add_entry(entry)
            cluster.matched_rules.append(result.rule_id)
            cluster.summary = result.summary

            self._clusters[cluster_id] = cluster
            self._stats["clusters_created"] += 1

            # Index indicators
            for indicator in entry.indicators:
                self._indicator_to_cluster[indicator] = cluster_id

            result.cluster_id = cluster_id

            if self.on_cluster_update:
                self.on_cluster_update(cluster)

    def _check_cluster_membership(self, entry: IntelligenceEntry) -> None:
        """Check if entry should be added to existing clusters."""
        for indicator in entry.indicators:
            if indicator in self._indicator_to_cluster:
                cluster_id = self._indicator_to_cluster[indicator]
                if cluster_id in self._clusters:
                    cluster = self._clusters[cluster_id]
                    if entry.entry_id not in cluster.entry_ids:
                        cluster.add_entry(entry)
                        if self.on_cluster_update:
                            self.on_cluster_update(cluster)

    def get_cluster(self, cluster_id: str) -> Optional[ThreatCluster]:
        """Get a threat cluster by ID."""
        return self._clusters.get(cluster_id)

    def get_active_clusters(
        self,
        min_threat_level: ThreatLevel = ThreatLevel.LOW,
        since: Optional[datetime] = None,
    ) -> List[ThreatCluster]:
        """Get active threat clusters."""
        clusters = []
        for cluster in self._clusters.values():
            if cluster.threat_level.value >= min_threat_level.value:
                if since and cluster.last_updated < since:
                    continue
                clusters.append(cluster)

        return sorted(clusters, key=lambda c: (c.threat_level.value, c.score), reverse=True)

    def get_stats(self) -> Dict[str, Any]:
        """Get correlator statistics."""
        return {
            **self._stats,
            "active_rules": len([r for r in self._rules.values() if r.enabled]),
            "active_clusters": len(self._clusters),
            "indexed_indicators": len(self._indicator_to_cluster),
        }


def create_threat_correlator(
    store: SecurityIntelligenceStore,
    on_correlation: Optional[Callable[[CorrelationResult], None]] = None,
) -> ThreatCorrelator:
    """
    Factory function to create a threat correlator.

    Args:
        store: Intelligence store
        on_correlation: Callback for correlations

    Returns:
        Configured ThreatCorrelator
    """
    return ThreatCorrelator(
        store=store,
        on_correlation=on_correlation,
    )
