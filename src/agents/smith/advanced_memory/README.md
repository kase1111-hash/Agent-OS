# Agent Smith Advanced Memory System

A sophisticated security intelligence storage and synthesis system that integrates with Boundary-SIEM and Boundary-Daemon.

## Quick Start

```python
from src.agents.smith.advanced_memory import (
    create_advanced_memory,
    MemoryConfig,
    BoundaryMode,
)

# Create with defaults
manager = create_advanced_memory(auto_start=True)

# Or with custom configuration
config = MemoryConfig(
    storage_path="/var/lib/agent-os/memory",
    boundary_endpoint="http://localhost:8080",
    boundary_api_key="your-api-key",
    hot_retention_days=7,
    warm_retention_days=30,
    cold_retention_days=365,
)
manager = create_advanced_memory(config=config, auto_start=True)
```

## Ingesting Events

```python
# From Boundary-SIEM
manager.ingest_siem_event({
    "timestamp": "2025-01-02T10:00:00Z",
    "severity": "high",
    "category": "intrusion",
    "message": "Suspicious activity detected",
    "indicators": ["192.168.1.100", "evil.com"],
    "mitre_attack": {"tactics": ["initial-access"]},
})

# From Boundary-Daemon
from src.agents.smith.advanced_memory import BoundaryEvent, BoundaryMode

manager.ingest_boundary_event(BoundaryEvent(
    event_id="be-001",
    timestamp=datetime.now(),
    event_type="policy_violation",
    source="boundary-daemon",
    current_mode=BoundaryMode.RESTRICTED,
    target_resource="/api/admin",
    action_requested="block",
    details={"user": "unknown"},
))
```

## Querying Intelligence

```python
# Get recent high-severity events
entries = manager.query(severity_min=4, limit=100)

# Get active threat clusters
clusters = manager.get_threat_clusters(min_threat_level=ThreatLevel.HIGH)

# Get detected patterns
patterns = manager.get_patterns(min_severity=3)

# Get anomaly score
score = manager.get_anomaly_score()

# Generate executive summary
summary = manager.generate_summary(period="last_24h")
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                 AdvancedMemoryManager                        │
├─────────────────────────────────────────────────────────────┤
│  Unified lifecycle management, event routing, callbacks     │
└──────────────────────────┬──────────────────────────────────┘
                           │
    ┌──────────┬───────────┼───────────┬──────────┐
    ▼          ▼           ▼           ▼          ▼
┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐
│ Store  │ │Correlat│ │Synthesi│ │Baseline│ │Boundary│
│        │ │   or   │ │   zer  │ │        │ │Connector
│Hot/Warm│ │ Rules  │ │Patterns│ │Anomaly │ │  REST  │
│ /Cold  │ │ MITRE  │ │ Trends │ │Learning│ │   API  │
└────────┘ └────────┘ └────────┘ └────────┘ └────────┘
```

## Components

| Component | Purpose | Key Features |
|-----------|---------|--------------|
| **SecurityIntelligenceStore** | Tiered data storage | Hot/warm/cold tiers, SQLite persistence, indexed queries |
| **ThreatCorrelator** | Event correlation | 7 default rules, MITRE kill chain tracking, threat clustering |
| **PatternSynthesizer** | Pattern detection | Volumetric, temporal, severity patterns; trend analysis |
| **BehavioralBaseline** | Anomaly detection | Adaptive learning, multi-dimensional scoring |
| **BoundaryDaemonConnector** | External integration | REST polling, 6 security modes, tripwire alerts |

## Retention Tiers

| Tier | Duration | Storage | Detail Level |
|------|----------|---------|--------------|
| **Hot** | 7 days | In-memory + SQLite | Full content |
| **Warm** | 30 days | SQLite | Reduced content |
| **Cold** | 365 days | SQLite | Summary only |

## Integration with SmithAgent

The advanced memory system is integrated into SmithAgent as an optional feature:

```python
from src.agents.smith import SmithAgent

smith = SmithAgent()
smith.initialize({
    "advanced_memory_enabled": True,
    "advanced_memory_config": {
        "storage_path": "/var/lib/agent-os/memory",
        "boundary_endpoint": "http://localhost:8080",
    },
})

# Use through SmithAgent API
smith.ingest_siem_event(event_data)
summary = smith.get_intelligence_summary()
clusters = smith.get_threat_clusters()
```

## Documentation

- [Technical Architecture](../../../docs/technical/advanced-memory.md)
- [API Reference](../../../docs/technical/advanced-memory-api.md)
- [Integration Guide](../../../docs/integration/advanced-memory-integration.md)

## External References

- [Boundary-Daemon](https://github.com/kase1111-hash/boundary-daemon-) - Trust policy & audit layer
- [Boundary-SIEM](https://github.com/kase1111-hash/Boundary-SIEM) - Security analytics platform
