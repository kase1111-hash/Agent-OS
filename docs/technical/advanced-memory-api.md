# Agent Smith Advanced Memory - API Reference

Version: 1.0
Last Updated: January 2026

## Table of Contents

1. [AdvancedMemoryManager](#advancedmemorymanager)
2. [SecurityIntelligenceStore](#securityintelligencestore)
3. [ThreatCorrelator](#threatcorrelator)
4. [PatternSynthesizer](#patternsynthesizer)
5. [BehavioralBaseline](#behavioralbaseline)
6. [BoundaryDaemonConnector](#boundarydaemonconnector)
7. [Data Classes](#data-classes)
8. [Enumerations](#enumerations)
9. [Factory Functions](#factory-functions)

---

## AdvancedMemoryManager

The unified manager for the advanced memory system.

### Constructor

```python
AdvancedMemoryManager(config: MemoryConfig)
```

### Lifecycle Methods

#### `initialize() -> bool`
Initialize all memory components.

```python
manager = AdvancedMemoryManager(config)
success = manager.initialize()
```

#### `start() -> bool`
Start all memory components. Calls `initialize()` if not already initialized.

```python
manager.start()
```

#### `stop() -> None`
Stop all memory components gracefully.

```python
manager.stop()
```

### Ingestion Methods

#### `ingest(entry: IntelligenceEntry) -> str`
Ingest a raw intelligence entry.

```python
entry = IntelligenceEntry(
    entry_id="evt-001",
    entry_type=IntelligenceType.SIEM_EVENT,
    timestamp=datetime.now(),
    source="siem",
    severity=3,
    category="intrusion",
    summary="Suspicious activity",
)
entry_id = manager.ingest(entry)
```

#### `ingest_siem_event(event_data: Dict[str, Any]) -> str`
Ingest an event from Boundary-SIEM format.

```python
entry_id = manager.ingest_siem_event({
    "timestamp": "2025-01-02T10:00:00Z",
    "severity": "high",  # or 1-5
    "category": "intrusion",
    "message": "Attack detected",
    "indicators": ["192.168.1.100"],
    "mitre_attack": {"tactics": ["initial-access"]},
    "tags": ["network", "external"],
})
```

**Severity Mapping**:
| String | Integer |
|--------|---------|
| critical | 5 |
| high | 4 |
| medium | 3 |
| low | 2 |
| informational/info | 1 |

#### `ingest_boundary_event(event: BoundaryEvent) -> str`
Ingest an event from Boundary-Daemon.

```python
event = BoundaryEvent(
    event_id="be-001",
    timestamp=datetime.now(),
    event_type="policy_violation",
    source="boundary-daemon",
    current_mode=BoundaryMode.RESTRICTED,
    target_resource="/api/admin",
    action_requested="block",
    details={"ip": "10.0.0.1"},
)
entry_id = manager.ingest_boundary_event(event)
```

### Query Methods

#### `query(...) -> List[IntelligenceEntry]`
Query intelligence entries with filters.

```python
entries = manager.query(
    entry_types=[IntelligenceType.SIEM_EVENT],
    sources=["boundary-siem"],
    severity_min=3,
    since=datetime.now() - timedelta(hours=24),
    limit=100,
)
```

**Parameters**:
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `entry_types` | List[IntelligenceType] | None | Filter by entry types |
| `sources` | List[str] | None | Filter by sources |
| `severity_min` | int | None | Minimum severity (1-5) |
| `since` | datetime | None | Entries after this time |
| `limit` | int | 100 | Maximum results |

#### `get_threat_clusters(...) -> List[ThreatCluster]`
Get active threat clusters.

```python
clusters = manager.get_threat_clusters(
    min_threat_level=ThreatLevel.HIGH,
    since=datetime.now() - timedelta(hours=24),
)
```

#### `get_patterns(...) -> List[SynthesizedPattern]`
Get detected security patterns.

```python
patterns = manager.get_patterns(
    min_severity=3,
    since=datetime.now() - timedelta(hours=24),
)
```

#### `get_anomaly_score(lookback_minutes: int = 60) -> Optional[AnomalyScore]`
Get current behavioral anomaly score.

```python
score = manager.get_anomaly_score(lookback_minutes=60)
if score and score.overall_score > 0.7:
    print(f"High anomaly detected: {score.description}")
```

#### `get_trend_analysis(period_hours: int = 24) -> Optional[TrendAnalysis]`
Get trend analysis for a time period.

```python
trend = manager.get_trend_analysis(period_hours=24)
```

### Summary Methods

#### `generate_summary(period: str = "last_24h") -> Optional[IntelligenceSummary]`
Generate an executive intelligence summary.

```python
summary = manager.generate_summary(period="last_24h")
print(f"Total events: {summary.total_events}")
print(f"Threat level: {summary.highest_threat_level.name}")
print(f"Summary: {summary.executive_summary}")
```

**Period Options**: `"last_24h"`, `"last_7d"`, `"last_30d"`

### Status Methods

#### `get_status() -> MemoryStatus`
Get current system status.

```python
status = manager.get_status()
print(f"Initialized: {status.initialized}")
print(f"Running: {status.running}")
print(f"Healthy: {status.healthy}")
print(f"Total entries: {status.total_entries}")
print(f"Active clusters: {status.active_clusters}")
```

#### `get_stats() -> Dict[str, Any]`
Get comprehensive statistics.

```python
stats = manager.get_stats()
print(f"Events processed: {stats['events_processed']}")
print(f"Correlations found: {stats['correlations_found']}")
```

### Boundary Mode Methods

#### `get_boundary_mode() -> Optional[BoundaryMode]`
Get current Boundary-Daemon mode.

```python
mode = manager.get_boundary_mode()
if mode == BoundaryMode.LOCKDOWN:
    print("System is in lockdown!")
```

#### `set_boundary_mode(mode: BoundaryMode) -> None`
Set local boundary mode tracking.

```python
manager.set_boundary_mode(BoundaryMode.RESTRICTED)
```

---

## SecurityIntelligenceStore

Low-level storage interface. Usually accessed through `AdvancedMemoryManager`.

### Methods

#### `add(entry: IntelligenceEntry) -> str`
Add an entry to the store.

#### `get(entry_id: str) -> Optional[IntelligenceEntry]`
Get an entry by ID.

#### `query(...) -> List[IntelligenceEntry]`
Query entries with filters.

#### `count(...) -> int`
Count matching entries.

#### `get_stats() -> Dict[str, Any]`
Get storage statistics.

---

## ThreatCorrelator

Event correlation engine. Usually accessed through `AdvancedMemoryManager`.

### Methods

#### `correlate(entry: IntelligenceEntry, lookback_minutes: int = 60) -> List[CorrelationResult]`
Correlate an entry with existing entries.

#### `add_rule(rule: CorrelationRule) -> None`
Add a custom correlation rule.

#### `list_rules() -> List[CorrelationRule]`
List all correlation rules.

#### `get_active_clusters(...) -> List[ThreatCluster]`
Get active threat clusters.

#### `get_cluster(cluster_id: str) -> Optional[ThreatCluster]`
Get a specific cluster by ID.

#### `get_stats() -> Dict[str, Any]`
Get correlator statistics.

---

## PatternSynthesizer

Pattern detection and summary generation. Usually accessed through `AdvancedMemoryManager`.

### Methods

#### `detect_patterns() -> List[SynthesizedPattern]`
Detect patterns in recent events.

#### `analyze_trends(period_hours: int = 24) -> TrendAnalysis`
Analyze trends over a time period.

#### `generate_summary(period: str = "last_24h") -> IntelligenceSummary`
Generate an intelligence summary.

#### `get_patterns(...) -> List[SynthesizedPattern]`
Get stored patterns.

#### `get_stats() -> Dict[str, Any]`
Get synthesizer statistics.

---

## BehavioralBaseline

Anomaly detection through behavioral learning.

### Methods

#### `start() -> None`
Start the baseline system.

#### `stop() -> None`
Stop the baseline system.

#### `learn(lookback_hours: int = 168) -> bool`
Learn baseline from historical data.

#### `is_ready() -> bool`
Check if baseline has been learned.

#### `score(lookback_minutes: int = 60) -> AnomalyScore`
Calculate anomaly score for recent activity.

#### `get_stats() -> Dict[str, Any]`
Get baseline statistics.

---

## BoundaryDaemonConnector

Integration with Boundary-Daemon API.

### Methods

#### `connect() -> bool`
Connect to the Boundary-Daemon API.

#### `start(poll_interval: int = 30) -> None`
Start polling for events.

#### `stop() -> None`
Stop polling.

#### `get_mode() -> BoundaryMode`
Get current boundary mode.

#### `set_mode(mode: BoundaryMode) -> None`
Set local mode tracking.

#### `get_stats() -> Dict[str, Any]`
Get connector statistics.

---

## Data Classes

### IntelligenceEntry

```python
@dataclass
class IntelligenceEntry:
    entry_id: str
    entry_type: IntelligenceType
    timestamp: datetime
    source: str
    severity: int  # 1-5
    category: str
    summary: str

    content: Dict[str, Any] = field(default_factory=dict)
    indicators: List[str] = field(default_factory=list)
    mitre_tactics: List[str] = field(default_factory=list)
    related_entries: List[str] = field(default_factory=list)
    tags: Set[str] = field(default_factory=set)

    tier: RetentionTier = RetentionTier.HOT
    created_at: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]: ...
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "IntelligenceEntry": ...
```

### ThreatCluster

```python
@dataclass
class ThreatCluster:
    cluster_id: str
    created_at: datetime
    last_updated: datetime
    threat_level: ThreatLevel
    score: float

    entry_ids: List[str]
    entry_count: int

    sources: Set[str]
    categories: Set[str]
    mitre_tactics: List[str]
    indicators: Set[str]

    first_seen: Optional[datetime]
    last_seen: Optional[datetime]

    matched_rules: List[str]
    tags: Set[str]
    summary: str

    def to_dict(self) -> Dict[str, Any]: ...
```

### SynthesizedPattern

```python
@dataclass
class SynthesizedPattern:
    pattern_id: str
    pattern_type: PatternType
    detected_at: datetime
    severity: int  # 1-5

    description: str
    evidence: List[str]  # Entry IDs
    indicators: List[str]

    confidence: float  # 0.0 - 1.0
    recommendation: str

    def to_dict(self) -> Dict[str, Any]: ...
```

### AnomalyScore

```python
@dataclass
class AnomalyScore:
    score_id: str
    timestamp: datetime
    overall_score: float  # 0.0 (normal) to 1.0 (anomalous)

    dimensions: Dict[str, float]
    description: str
    contributing_factors: List[str]

    def to_dict(self) -> Dict[str, Any]: ...
```

### IntelligenceSummary

```python
@dataclass
class IntelligenceSummary:
    summary_id: str
    period: str
    generated_at: datetime

    total_events: int
    events_by_severity: Dict[int, int]
    events_by_source: Dict[str, int]
    events_by_category: Dict[str, int]

    active_threat_clusters: int
    highest_threat_level: ThreatLevel
    critical_indicators: List[str]

    detected_patterns: List[SynthesizedPattern]
    trend_direction: str

    executive_summary: str
    recommended_actions: List[str]

    def to_dict(self) -> Dict[str, Any]: ...
```

### MemoryConfig

```python
@dataclass
class MemoryConfig:
    storage_path: Optional[str] = None
    hot_retention_days: int = 7
    warm_retention_days: int = 30
    cold_retention_days: int = 365

    boundary_endpoint: Optional[str] = None
    boundary_api_key: Optional[str] = None
    boundary_poll_interval: int = 30

    correlation_enabled: bool = True
    correlation_lookback_minutes: int = 60

    synthesis_enabled: bool = True
    synthesis_interval_hours: int = 1

    baseline_enabled: bool = True
    baseline_learning_hours: int = 168

    on_high_threat: Optional[Callable[[ThreatCluster], None]] = None
    on_anomaly: Optional[Callable[[AnomalyScore], None]] = None
    on_pattern: Optional[Callable[[SynthesizedPattern], None]] = None
```

### MemoryStatus

```python
@dataclass
class MemoryStatus:
    initialized: bool = False
    running: bool = False
    healthy: bool = False

    store_healthy: bool = False
    correlator_active: bool = False
    synthesizer_active: bool = False
    baseline_ready: bool = False
    boundary_connected: bool = False

    total_entries: int = 0
    hot_entries: int = 0
    active_clusters: int = 0
    detected_patterns: int = 0
    anomaly_score: float = 0.0

    current_threat_level: ThreatLevel = ThreatLevel.UNKNOWN
    critical_threats: int = 0

    def to_dict(self) -> Dict[str, Any]: ...
```

### BoundaryEvent

```python
@dataclass
class BoundaryEvent:
    event_id: str
    timestamp: datetime
    event_type: str
    source: str
    current_mode: BoundaryMode
    target_resource: str
    action_requested: str
    details: Dict[str, Any]

    def to_intelligence_entry(self) -> IntelligenceEntry: ...
```

### PolicyDecision

```python
@dataclass
class PolicyDecision:
    decision_id: str
    timestamp: datetime
    policy_id: str
    decision: str  # "allow", "deny", "audit"
    reason: str
    target_resource: str
    details: Dict[str, Any]
```

### TripwireAlert

```python
@dataclass
class TripwireAlert:
    alert_id: str
    timestamp: datetime
    tripwire_id: str
    triggered_by: str
    alert_type: str  # "modification", "access", "deletion"
    severity: int
    details: Dict[str, Any]
```

### CorrelationRule

```python
@dataclass
class CorrelationRule:
    rule_id: str
    name: str
    description: str
    correlation_type: CorrelationType
    enabled: bool = True

    source_types: List[IntelligenceType] = field(default_factory=list)
    categories: List[str] = field(default_factory=list)
    min_severity: int = 1
    required_indicators: List[str] = field(default_factory=list)
    required_tactics: List[str] = field(default_factory=list)

    time_window_minutes: int = 60
    min_events: int = 2
    max_events: int = 1000

    base_score: int = 50
    severity_weight: float = 1.5
    source_diversity_bonus: int = 10

    output_threat_level: ThreatLevel = ThreatLevel.MEDIUM
    tags: Set[str] = field(default_factory=set)

    def matches(self, entry: IntelligenceEntry) -> bool: ...
```

---

## Enumerations

### IntelligenceType

```python
class IntelligenceType(Enum):
    # External events
    SIEM_EVENT = auto()
    BOUNDARY_EVENT = auto()
    TRIPWIRE_ALERT = auto()
    POLICY_DECISION = auto()

    # Internal events
    ATTACK_DETECTED = auto()
    INCIDENT = auto()
    VALIDATION_FAILURE = auto()
    REFUSAL = auto()

    # Synthesized
    PATTERN = auto()
    TREND = auto()
    CORRELATION = auto()
    ANOMALY = auto()

    # Reference
    THREAT_INTEL = auto()
    BASELINE = auto()
```

### RetentionTier

```python
class RetentionTier(Enum):
    HOT = "hot"       # 7 days, full detail
    WARM = "warm"     # 30 days, reduced
    COLD = "cold"     # 365 days, summary
    PERMANENT = "permanent"
```

### ThreatLevel

```python
class ThreatLevel(Enum):
    UNKNOWN = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4
    CONFIRMED = 5
```

### CorrelationType

```python
class CorrelationType(Enum):
    TEMPORAL = auto()
    INDICATOR = auto()
    SOURCE = auto()
    MITRE_CHAIN = auto()
    BEHAVIORAL = auto()
    CUSTOM = auto()
```

### PatternType

```python
class PatternType(Enum):
    VOLUMETRIC = auto()
    SEVERITY = auto()
    SOURCE = auto()
    INDICATOR = auto()
    TEMPORAL = auto()
```

### BoundaryMode

```python
class BoundaryMode(Enum):
    OPEN = "open"
    RESTRICTED = "restricted"
    TRUSTED = "trusted"
    AIRGAP = "airgap"
    COLDROOM = "coldroom"
    LOCKDOWN = "lockdown"
```

---

## Factory Functions

### create_advanced_memory

```python
def create_advanced_memory(
    config: Optional[MemoryConfig] = None,
    storage_path: Optional[str] = None,
    auto_start: bool = True,
) -> AdvancedMemoryManager:
    """Create and optionally start an Advanced Memory Manager."""
```

### create_intelligence_store

```python
def create_intelligence_store(
    storage_path: Optional[str] = None,
    retention_policy: Optional[RetentionPolicy] = None,
    auto_start: bool = True,
) -> SecurityIntelligenceStore:
    """Create a Security Intelligence Store."""
```

### create_threat_correlator

```python
def create_threat_correlator(
    store: SecurityIntelligenceStore,
    on_correlation: Optional[Callable] = None,
) -> ThreatCorrelator:
    """Create a Threat Correlator."""
```

### create_pattern_synthesizer

```python
def create_pattern_synthesizer(
    store: SecurityIntelligenceStore,
    correlator: Optional[ThreatCorrelator] = None,
    on_pattern: Optional[Callable] = None,
) -> PatternSynthesizer:
    """Create a Pattern Synthesizer."""
```

### create_behavioral_baseline

```python
def create_behavioral_baseline(
    store: SecurityIntelligenceStore,
    on_anomaly: Optional[Callable] = None,
    auto_start: bool = True,
) -> BehavioralBaseline:
    """Create a Behavioral Baseline."""
```

### create_boundary_connector

```python
def create_boundary_connector(
    store: SecurityIntelligenceStore,
    endpoint: Optional[str] = None,
    api_key: Optional[str] = None,
    auto_start: bool = False,
) -> BoundaryDaemonConnector:
    """Create a Boundary Daemon Connector."""
```

---

## Related Documentation

- [Technical Architecture](./advanced-memory.md)
- [Integration Guide](../integration/advanced-memory-integration.md)
