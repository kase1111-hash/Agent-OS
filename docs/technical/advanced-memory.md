# Agent Smith Advanced Memory System - Technical Architecture

Version: 1.0
Last Updated: January 2026
Maintainer: Agent OS Community

## Table of Contents

1. [Overview](#overview)
2. [Design Goals](#design-goals)
3. [System Architecture](#system-architecture)
4. [Component Details](#component-details)
5. [Data Flow](#data-flow)
6. [Storage Architecture](#storage-architecture)
7. [Correlation Engine](#correlation-engine)
8. [Security Considerations](#security-considerations)
9. [Performance Characteristics](#performance-characteristics)
10. [Configuration Reference](#configuration-reference)

---

## Overview

The Advanced Memory System extends Agent Smith with long-term security intelligence capabilities. It provides:

- **Tiered Storage**: Hot/warm/cold data retention modeled after enterprise SIEM systems
- **Threat Correlation**: Cross-source event correlation with MITRE ATT&CK mapping
- **Pattern Synthesis**: Automatic detection of attack patterns and trends
- **Behavioral Baseline**: Anomaly detection through adaptive learning
- **External Integration**: Connectors for Boundary-SIEM and Boundary-Daemon

This system enables Agent Smith to maintain institutional memory of security events, detect complex multi-stage attacks, and generate actionable intelligence summaries.

---

## Design Goals

### 1. Enterprise-Grade Retention
Modeled after Boundary-SIEM's ClickHouse storage architecture:
- Immediate access to recent events (hot tier)
- Cost-effective long-term storage (cold tier)
- Automatic tier migration based on age

### 2. Cross-Source Correlation
Inspired by Boundary-SIEM's 103 detection rules:
- Correlate events from multiple security sources
- Track attack progression through MITRE kill chain
- Cluster related threats for unified response

### 3. Intelligent Synthesis
Transform raw events into actionable intelligence:
- Detect volumetric, temporal, and severity patterns
- Generate executive summaries for security posture
- Identify trends and anomalies

### 4. Minimal Dependencies
Self-contained module using only:
- Python standard library
- SQLite for persistence (stdlib)
- `requests` for external API calls (optional)

---

## System Architecture

### High-Level View

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           Agent Smith                                    │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │                    Advanced Memory System                          │  │
│  │                                                                    │  │
│  │  ┌──────────────────────────────────────────────────────────────┐ │  │
│  │  │                 AdvancedMemoryManager                         │ │  │
│  │  │  • Unified lifecycle (init/start/stop)                       │ │  │
│  │  │  • Event routing and processing pipeline                     │ │  │
│  │  │  • Callback registration and dispatch                        │ │  │
│  │  │  • Status monitoring and statistics                          │ │  │
│  │  └──────────────────────────┬───────────────────────────────────┘ │  │
│  │                              │                                     │  │
│  │    ┌─────────┬───────────────┼───────────────┬─────────┐          │  │
│  │    ▼         ▼               ▼               ▼         ▼          │  │
│  │ ┌───────┐ ┌───────┐    ┌──────────┐    ┌────────┐ ┌─────────┐    │  │
│  │ │Store  │ │Correlat│    │Synthesiz│    │Baseline│ │Boundary │    │  │
│  │ │       │ │   or   │    │   er    │    │        │ │Connector│    │  │
│  │ │SQLite │ │7 Rules │    │Patterns │    │Anomaly │ │REST API │    │  │
│  │ │Tiered │ │MITRE   │    │Trends   │    │Learning│ │Polling  │    │  │
│  │ └───┬───┘ └───┬───┘    └────┬────┘    └───┬────┘ └────┬────┘    │  │
│  │     │         │             │             │           │          │  │
│  └─────┼─────────┼─────────────┼─────────────┼───────────┼──────────┘  │
│        │         │             │             │           │             │
└────────┼─────────┼─────────────┼─────────────┼───────────┼─────────────┘
         │         │             │             │           │
         ▼         ▼             ▼             ▼           ▼
    ┌─────────────────────────────────────────────────────────────────┐
    │                    Persistent Storage                            │
    │                      (SQLite DB)                                 │
    └─────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    ▼                               ▼
            ┌───────────────┐               ┌───────────────┐
            │Boundary-Daemon│               │ Boundary-SIEM │
            │  (Optional)   │               │  (Optional)   │
            └───────────────┘               └───────────────┘
```

### Component Relationships

```
                    ┌─────────────────────┐
                    │ MemoryConfig        │
                    │ • storage_path      │
                    │ • retention days    │
                    │ • feature flags     │
                    │ • callbacks         │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │AdvancedMemoryManager│◄──── Entry Point
                    └──────────┬──────────┘
                               │
         ┌─────────────────────┼─────────────────────┐
         │                     │                     │
         ▼                     ▼                     ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│SecurityIntelli- │  │ThreatCorrelator │  │PatternSynthesi- │
│genceStore       │  │                 │  │zer              │
│                 │  │ Uses store for  │  │                 │
│ Primary storage │  │ lookback queries│  │ Uses store +    │
│ for all entries │  │                 │  │ correlator      │
└────────┬────────┘  └────────┬────────┘  └────────┬────────┘
         │                    │                     │
         │           ┌────────┴────────┐           │
         │           ▼                 ▼           │
         │  ┌─────────────────┐  ┌─────────────────┐
         │  │BehavioralBaseli│  │BoundaryDaemon   │
         │  │ne              │  │Connector        │
         │  │                │  │                 │
         │  │Uses store for  │  │Polls external   │
         │  │baseline data   │  │API, stores      │
         │  └────────────────┘  │events in store  │
         │                      └─────────────────┘
         │
         ▼
    ┌─────────────────────────────────────┐
    │           SQLite Database            │
    │  ┌─────────┬─────────┬─────────┐    │
    │  │   HOT   │  WARM   │  COLD   │    │
    │  │ 7 days  │ 30 days │365 days │    │
    │  └─────────┴─────────┴─────────┘    │
    └─────────────────────────────────────┘
```

---

## Component Details

### SecurityIntelligenceStore

**Purpose**: Tiered storage for security intelligence entries.

**Key Classes**:
- `IntelligenceEntry`: Core data structure for all stored events
- `IntelligenceType`: Enum of 14 event types (SIEM_EVENT, BOUNDARY_EVENT, ATTACK_DETECTED, etc.)
- `RetentionTier`: HOT, WARM, COLD, PERMANENT
- `RetentionPolicy`: Configurable retention durations

**Features**:
- In-memory hot tier for immediate access
- SQLite persistence for warm/cold tiers
- Indexed queries on common fields (source, severity, category, timestamp)
- Automatic tier migration based on age
- Thread-safe operations with RLock

**Schema**:
```sql
CREATE TABLE intelligence (
    entry_id TEXT PRIMARY KEY,
    entry_type TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    source TEXT NOT NULL,
    severity INTEGER NOT NULL,
    category TEXT NOT NULL,
    summary TEXT,
    content TEXT,  -- JSON
    indicators TEXT,  -- JSON array
    mitre_tactics TEXT,  -- JSON array
    tags TEXT,  -- JSON array
    tier TEXT NOT NULL,
    created_at TEXT NOT NULL,
    expires_at TEXT
);

CREATE INDEX idx_timestamp ON intelligence(timestamp);
CREATE INDEX idx_source ON intelligence(source);
CREATE INDEX idx_severity ON intelligence(severity);
CREATE INDEX idx_category ON intelligence(category);
CREATE INDEX idx_tier ON intelligence(tier);
```

---

### ThreatCorrelator

**Purpose**: Cross-source event correlation and threat clustering.

**Key Classes**:
- `CorrelationRule`: Defines correlation criteria
- `CorrelationType`: TEMPORAL, INDICATOR, SOURCE, MITRE_CHAIN, BEHAVIORAL, CUSTOM
- `ThreatCluster`: Group of correlated events
- `ThreatLevel`: UNKNOWN, LOW, MEDIUM, HIGH, CRITICAL, CONFIRMED

**Default Rules**:

| Rule ID | Name | Type | Description |
|---------|------|------|-------------|
| `kill_chain_progression` | Kill Chain Progression | MITRE_CHAIN | Detects MITRE ATT&CK kill chain advancement |
| `multi_source_indicator` | Multi-Source Indicator | INDICATOR | Same IOC from multiple sources |
| `rapid_escalation` | Rapid Escalation | TEMPORAL | Quick severity escalation |
| `lateral_movement_pattern` | Lateral Movement | BEHAVIORAL | East-west traffic patterns |
| `exfiltration_sequence` | Exfiltration Sequence | MITRE_CHAIN | Collection → exfiltration progression |
| `persistence_establishment` | Persistence Establishment | MITRE_CHAIN | Persistence mechanism installation |
| `credential_abuse` | Credential Abuse Chain | BEHAVIORAL | Credential theft → lateral movement |

**MITRE Kill Chain Tracking**:
```python
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
```

---

### PatternSynthesizer

**Purpose**: Detect patterns and generate intelligence summaries.

**Pattern Types**:

| Type | Detection Method |
|------|------------------|
| VOLUMETRIC | Event count exceeds baseline by threshold |
| SEVERITY | Concentration of high-severity events |
| SOURCE | Unusual source distribution |
| INDICATOR | Repeated IOCs across events |
| TEMPORAL | Time-based clustering (off-hours, burst) |

**Intelligence Summary Structure**:
```python
@dataclass
class IntelligenceSummary:
    summary_id: str
    period: str  # "last_24h", "last_7d", "last_30d"
    generated_at: datetime

    # Statistics
    total_events: int
    events_by_severity: Dict[int, int]
    events_by_source: Dict[str, int]
    events_by_category: Dict[str, int]

    # Threats
    active_threat_clusters: int
    highest_threat_level: ThreatLevel
    critical_indicators: List[str]

    # Patterns
    detected_patterns: List[SynthesizedPattern]
    trend_direction: str  # "increasing", "decreasing", "stable"

    # Recommendations
    executive_summary: str
    recommended_actions: List[str]
```

---

### BehavioralBaseline

**Purpose**: Learn normal behavior and detect anomalies.

**Dimensions**:
- Volume: Event count per time window
- Severity: Average severity distribution
- Sources: Typical source diversity
- Categories: Normal category distribution
- Temporal: Expected time-of-day patterns

**Anomaly Scoring**:
```python
@dataclass
class AnomalyScore:
    score_id: str
    timestamp: datetime
    overall_score: float  # 0.0 (normal) to 1.0 (highly anomalous)

    dimensions: Dict[str, float]  # Per-dimension scores
    # Example: {"volume": 0.3, "severity": 0.7, "sources": 0.1}

    description: str
    contributing_factors: List[str]
```

**Learning Process**:
1. Collect events over learning period (default: 168 hours)
2. Compute statistical baselines per dimension
3. Store baseline snapshot in database
4. Compare current window against learned baseline
5. Score each dimension using z-score methodology

---

### BoundaryDaemonConnector

**Purpose**: Integration with external Boundary-Daemon API.

**Security Modes**:

| Mode | Description | Typical Use |
|------|-------------|-------------|
| OPEN | No restrictions | Development, trusted environments |
| RESTRICTED | Limited external access | Standard operation |
| TRUSTED | Enhanced logging | High-security environments |
| AIRGAP | No external network | Isolated deployments |
| COLDROOM | Minimal operation | Emergency response |
| LOCKDOWN | All operations blocked | Active incident |

**Polling Endpoints**:
- `GET /api/v1/health` - Health check
- `GET /api/v1/events` - Security events
- `GET /api/v1/policy/decisions` - Policy decisions
- `GET /api/v1/tripwires/alerts` - Tripwire violations

**Event Types Ingested**:
- Mode changes
- Policy violations
- Tripwire alerts
- Process sandbox events
- Audit log entries

---

## Data Flow

### Event Ingestion Flow

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│ Boundary-SIEM│     │Boundary-Daem.│     │ SmithAgent   │
│   (external) │     │  (external)  │     │  (internal)  │
└──────┬───────┘     └──────┬───────┘     └──────┬───────┘
       │                    │                    │
       │ SIEM Event         │ Daemon Event       │ Validation Failure
       ▼                    ▼                    ▼
┌─────────────────────────────────────────────────────────┐
│              AdvancedMemoryManager.ingest()              │
├─────────────────────────────────────────────────────────┤
│  1. Create IntelligenceEntry                            │
│  2. Add to SecurityIntelligenceStore                    │
│  3. Run ThreatCorrelator.correlate()                    │
│  4. Update BehavioralBaseline                           │
│  5. Trigger callbacks if thresholds met                 │
└─────────────────────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────────────┐
│                    Callbacks                             │
├─────────────────────────────────────────────────────────┤
│  • on_high_threat(cluster)  → Smith.trigger_safe_mode() │
│  • on_anomaly(score)        → Log security incident     │
│  • on_pattern(pattern)      → Alert notification        │
└─────────────────────────────────────────────────────────┘
```

### Query Flow

```
┌──────────────┐
│  API Call    │
│  (query,     │
│   clusters,  │
│   patterns)  │
└──────┬───────┘
       │
       ▼
┌─────────────────────────────────────────────────────────┐
│              AdvancedMemoryManager                       │
├─────────────────────────────────────────────────────────┤
│  Route to appropriate component:                        │
│  • query() → SecurityIntelligenceStore                  │
│  • get_threat_clusters() → ThreatCorrelator             │
│  • get_patterns() → PatternSynthesizer                  │
│  • get_anomaly_score() → BehavioralBaseline             │
│  • generate_summary() → PatternSynthesizer              │
└──────────────────────────┬──────────────────────────────┘
                           │
                           ▼
                    ┌──────────────┐
                    │   Response   │
                    └──────────────┘
```

---

## Storage Architecture

### Tier Migration

```
Event Ingested
      │
      ▼
┌─────────────┐
│  HOT TIER   │  In-memory + SQLite
│  (7 days)   │  Full content retention
└──────┬──────┘
       │ Age > 7 days
       ▼
┌─────────────┐
│  WARM TIER  │  SQLite only
│  (30 days)  │  Compacted content
└──────┬──────┘
       │ Age > 30 days
       ▼
┌─────────────┐
│  COLD TIER  │  SQLite only
│ (365 days)  │  Summary only
└──────┬──────┘
       │ Age > 365 days
       ▼
    [DELETED]
```

### Content Compaction

| Field | Hot | Warm | Cold |
|-------|-----|------|------|
| entry_id | Full | Full | Full |
| timestamp | Full | Full | Full |
| source | Full | Full | Full |
| severity | Full | Full | Full |
| category | Full | Full | Full |
| summary | Full | Full | Full |
| content | Full JSON | Essential fields only | Removed |
| indicators | Full list | Top 10 | Top 3 |
| mitre_tactics | Full list | Full list | Full list |
| related_entries | Full list | Removed | Removed |

---

## Security Considerations

### SQL Injection Prevention
All database queries use parameterized statements:
```python
# Safe - parameterized
cursor.execute(
    "SELECT * FROM intelligence WHERE source = ?",
    (source,)
)

# Never used - vulnerable
cursor.execute(f"SELECT * FROM intelligence WHERE source = '{source}'")
```

### API Authentication
Boundary-Daemon connector uses Bearer token authentication:
```python
headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
response = requests.get(endpoint, headers=headers, timeout=10)
```

### Thread Safety
All components use RLock for thread-safe operations:
```python
def add(self, entry: IntelligenceEntry) -> str:
    with self._lock:
        # Thread-safe operations
```

### No Data Exfiltration
- Only GET requests to configured endpoints
- No POST/PUT to external services
- No hardcoded external URLs

---

## Performance Characteristics

### Memory Usage
- Hot tier: ~1KB per entry (in-memory)
- SQLite: Disk-based, minimal memory footprint
- Correlation: O(n) per event where n = lookback window entries

### Query Performance
- Indexed queries: O(log n) with SQLite B-tree indexes
- Full scan: O(n) for complex queries
- Correlation lookback: Configurable window (default: 60 minutes)

### Recommended Limits
| Metric | Recommended | Maximum |
|--------|-------------|---------|
| Hot tier entries | 10,000 | 100,000 |
| Events per day | 50,000 | 500,000 |
| Concurrent queries | 10 | 100 |
| Correlation rules | 20 | 100 |

---

## Configuration Reference

### MemoryConfig Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `storage_path` | str | None | Path for SQLite database |
| `hot_retention_days` | int | 7 | Hot tier retention |
| `warm_retention_days` | int | 30 | Warm tier retention |
| `cold_retention_days` | int | 365 | Cold tier retention |
| `boundary_endpoint` | str | None | Boundary-Daemon API URL |
| `boundary_api_key` | str | None | Boundary-Daemon API key |
| `boundary_poll_interval` | int | 30 | Polling interval (seconds) |
| `correlation_enabled` | bool | True | Enable threat correlation |
| `correlation_lookback_minutes` | int | 60 | Correlation time window |
| `synthesis_enabled` | bool | True | Enable pattern synthesis |
| `synthesis_interval_hours` | int | 1 | Synthesis frequency |
| `baseline_enabled` | bool | True | Enable behavioral baseline |
| `baseline_learning_hours` | int | 168 | Initial learning period |
| `on_high_threat` | Callable | None | High threat callback |
| `on_anomaly` | Callable | None | Anomaly callback |
| `on_pattern` | Callable | None | Pattern callback |

### SmithAgent Configuration

```python
smith.initialize({
    "advanced_memory_enabled": True,
    "advanced_memory_config": {
        "storage_path": "/var/lib/agent-os/memory",
        "boundary_endpoint": "http://boundary:8080",
        "boundary_api_key": os.environ["BOUNDARY_API_KEY"],
        "hot_retention_days": 7,
        "warm_retention_days": 30,
        "cold_retention_days": 365,
        "correlation_enabled": True,
        "synthesis_enabled": True,
        "baseline_enabled": True,
    },
})
```

---

## Related Documentation

- [API Reference](./advanced-memory-api.md)
- [Integration Guide](../integration/advanced-memory-integration.md)
- [Boundary Tools Integration](../integration/boundary-tools.md)
- [Security Architecture](./architecture.md#security-architecture)
