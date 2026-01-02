# Advanced Memory Integration Guide

This guide covers integrating the Advanced Memory System with Agent Smith and external security tools.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Basic Integration](#basic-integration)
3. [SmithAgent Integration](#smithagent-integration)
4. [Boundary-SIEM Integration](#boundary-siem-integration)
5. [Boundary-Daemon Integration](#boundary-daemon-integration)
6. [Webhook & Callback Integration](#webhook--callback-integration)
7. [Custom Correlation Rules](#custom-correlation-rules)
8. [Monitoring & Alerting](#monitoring--alerting)
9. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### Required
- Python 3.9+
- Agent OS core modules

### Optional
- `requests` library (for Boundary-Daemon connector)
- Boundary-Daemon API endpoint
- Boundary-SIEM instance

---

## Basic Integration

### Standalone Usage

```python
from src.agents.smith.advanced_memory import (
    create_advanced_memory,
    MemoryConfig,
    IntelligenceEntry,
    IntelligenceType,
)
from datetime import datetime

# Create manager with defaults
manager = create_advanced_memory(auto_start=True)

# Ingest an event
entry = IntelligenceEntry(
    entry_id="custom-001",
    entry_type=IntelligenceType.SIEM_EVENT,
    timestamp=datetime.now(),
    source="my-security-tool",
    severity=3,
    category="detection",
    summary="Custom security event",
    content={"key": "value"},
    indicators=["192.168.1.100"],
)

manager.ingest(entry)

# Query events
entries = manager.query(severity_min=3, limit=10)

# Cleanup
manager.stop()
```

### With Custom Configuration

```python
config = MemoryConfig(
    # Storage
    storage_path="/var/lib/agent-os/memory/db",
    hot_retention_days=14,      # Keep hot data for 2 weeks
    warm_retention_days=60,     # Keep warm data for 2 months
    cold_retention_days=730,    # Keep cold data for 2 years

    # External integration
    boundary_endpoint="http://boundary-daemon:8080",
    boundary_api_key="your-secure-api-key",
    boundary_poll_interval=15,  # Poll every 15 seconds

    # Features
    correlation_enabled=True,
    correlation_lookback_minutes=120,  # 2-hour correlation window
    synthesis_enabled=True,
    synthesis_interval_hours=2,        # Synthesize every 2 hours
    baseline_enabled=True,
    baseline_learning_hours=336,       # 2-week learning period
)

manager = create_advanced_memory(config=config, auto_start=True)
```

---

## SmithAgent Integration

### Enable Advanced Memory

```python
from src.agents.smith import SmithAgent, create_smith

# Method 1: Direct initialization
smith = SmithAgent()
smith.initialize({
    "strict_mode": True,
    "allow_escalation": True,

    # Enable advanced memory
    "advanced_memory_enabled": True,
    "advanced_memory_config": {
        "storage_path": "/var/lib/agent-os/memory",
        "boundary_endpoint": "http://boundary:8080",
        "boundary_api_key": os.environ.get("BOUNDARY_API_KEY"),
    },
})

# Method 2: Factory function
smith = create_smith(config={
    "advanced_memory_enabled": True,
    "advanced_memory_config": {...},
})
```

### Using SmithAgent Memory APIs

```python
# Ingest events through SmithAgent
smith.ingest_siem_event({
    "timestamp": datetime.now().isoformat(),
    "severity": "high",
    "category": "intrusion",
    "message": "Attack detected",
})

smith.ingest_boundary_event({
    "id": "be-001",
    "type": "policy_violation",
    "mode": "RESTRICTED",
    "target": "/api/admin",
    "action": "block",
})

# Query through SmithAgent
summary = smith.get_intelligence_summary(period="last_24h")
clusters = smith.get_threat_clusters(min_level="HIGH")
patterns = smith.get_detected_patterns(min_severity=3)
score = smith.get_anomaly_score()
status = smith.get_memory_status()
```

### Register Callbacks

```python
def handle_threat(cluster):
    print(f"ALERT: Threat cluster {cluster.cluster_id}")
    print(f"  Level: {cluster.threat_level.name}")
    print(f"  Entries: {cluster.entry_count}")
    # Send to alerting system
    send_alert(cluster)

def handle_anomaly(score):
    if score.overall_score > 0.8:
        print(f"HIGH ANOMALY: {score.description}")
        trigger_investigation(score)

def handle_pattern(pattern):
    print(f"Pattern detected: {pattern.pattern_type.name}")

smith.register_threat_callback(handle_threat)
smith.register_anomaly_callback(handle_anomaly)
smith.register_pattern_callback(handle_pattern)
```

---

## Boundary-SIEM Integration

### Direct API Integration

If you have a Boundary-SIEM instance, you can forward events directly:

```python
import requests
from src.agents.smith.advanced_memory import create_advanced_memory

manager = create_advanced_memory(auto_start=True)

# Poll SIEM for events
def poll_siem(siem_url: str, api_key: str):
    headers = {"Authorization": f"Bearer {api_key}"}
    response = requests.get(
        f"{siem_url}/api/v1/events",
        headers=headers,
        params={"since": last_poll_time},
    )

    for event in response.json()["events"]:
        manager.ingest_siem_event(event)
```

### Webhook Receiver

Set up a webhook endpoint to receive events:

```python
from flask import Flask, request
from src.agents.smith.advanced_memory import create_advanced_memory

app = Flask(__name__)
manager = create_advanced_memory(auto_start=True)

@app.route("/webhook/siem", methods=["POST"])
def siem_webhook():
    event = request.json
    entry_id = manager.ingest_siem_event(event)
    return {"status": "ok", "entry_id": entry_id}

@app.route("/webhook/boundary", methods=["POST"])
def boundary_webhook():
    event = request.json
    # Convert to BoundaryEvent format
    entry_id = manager.ingest_boundary_event(
        BoundaryEvent.from_dict(event)
    )
    return {"status": "ok", "entry_id": entry_id}
```

### Event Format Mapping

Boundary-SIEM events are mapped to IntelligenceEntry:

| SIEM Field | IntelligenceEntry Field |
|------------|-------------------------|
| `timestamp` | `timestamp` |
| `severity` (string) | `severity` (1-5 mapped) |
| `category` | `category` |
| `message`/`description` | `summary` |
| `indicators` | `indicators` |
| `mitre_attack.tactics` | `mitre_tactics` |
| `tags` | `tags` (+ "siem") |
| *(entire event)* | `content` |

---

## Boundary-Daemon Integration

### Automatic Polling

The `BoundaryDaemonConnector` automatically polls for events:

```python
from src.agents.smith.advanced_memory import (
    create_advanced_memory,
    MemoryConfig,
)

config = MemoryConfig(
    boundary_endpoint="http://boundary-daemon:8080",
    boundary_api_key="your-api-key",
    boundary_poll_interval=30,  # seconds
)

manager = create_advanced_memory(config=config, auto_start=True)
# Connector will automatically poll and ingest events
```

### Polled Endpoints

| Endpoint | Event Type | Description |
|----------|------------|-------------|
| `/api/v1/events` | `BOUNDARY_EVENT` | General security events |
| `/api/v1/policy/decisions` | `POLICY_DECISION` | Allow/deny decisions |
| `/api/v1/tripwires/alerts` | `TRIPWIRE_ALERT` | File/config changes |

### Mode Synchronization

```python
from src.agents.smith.advanced_memory import BoundaryMode

# Get current mode
mode = manager.get_boundary_mode()
print(f"Current mode: {mode.name}")

# React to mode changes
if mode == BoundaryMode.LOCKDOWN:
    smith.trigger_safe_mode("Boundary in LOCKDOWN")
elif mode == BoundaryMode.AIRGAP:
    disable_external_network()
```

### Manual Event Ingestion

```python
from src.agents.smith.advanced_memory import (
    BoundaryEvent,
    BoundaryMode,
    PolicyDecision,
    TripwireAlert,
)
from datetime import datetime

# Policy decision
manager.ingest(PolicyDecision(
    decision_id="pd-001",
    timestamp=datetime.now(),
    policy_id="network-egress",
    decision="deny",
    reason="Unauthorized external connection",
    target_resource="https://evil.com",
    details={"user": "agent-123"},
).to_intelligence_entry())

# Tripwire alert
manager.ingest(TripwireAlert(
    alert_id="tw-001",
    timestamp=datetime.now(),
    tripwire_id="config-monitor",
    triggered_by="/etc/agent-os/constitution.yaml",
    alert_type="modification",
    severity=5,
    details={"old_hash": "abc", "new_hash": "xyz"},
).to_intelligence_entry())
```

---

## Webhook & Callback Integration

### Threat Escalation

Automatically escalate high-threat clusters:

```python
from src.agents.smith.advanced_memory import ThreatLevel

def on_high_threat(cluster):
    if cluster.threat_level == ThreatLevel.CRITICAL:
        # Trigger lockdown
        smith.trigger_lockdown(f"Critical threat: {cluster.cluster_id}")

        # Notify security team
        send_pagerduty_alert({
            "severity": "critical",
            "summary": f"Critical threat cluster detected",
            "details": cluster.to_dict(),
        })

    elif cluster.threat_level == ThreatLevel.HIGH:
        # Trigger safe mode
        smith.trigger_safe_mode(f"High threat: {cluster.cluster_id}")

        # Notify via Slack
        send_slack_message("#security-alerts", {
            "text": f"High threat cluster: {cluster.summary}",
            "attachments": format_cluster(cluster),
        })

config = MemoryConfig(
    on_high_threat=on_high_threat,
    ...
)
```

### Anomaly Response

React to behavioral anomalies:

```python
def on_anomaly(score):
    if score.overall_score > 0.9:
        # Critical anomaly - automatic response
        smith.trigger_safe_mode("Critical anomaly detected")
        start_forensic_capture()

    elif score.overall_score > 0.7:
        # High anomaly - alert and investigate
        create_investigation_ticket({
            "title": f"Anomaly detected: {score.description}",
            "priority": "high",
            "factors": score.contributing_factors,
        })

    elif score.overall_score > 0.5:
        # Medium anomaly - log for review
        logger.warning(f"Anomaly: {score.description}")

config = MemoryConfig(
    on_anomaly=on_anomaly,
    ...
)
```

### Pattern Alerting

Alert on specific pattern types:

```python
from src.agents.smith.advanced_memory import PatternType

def on_pattern(pattern):
    if pattern.pattern_type == PatternType.VOLUMETRIC:
        # Possible DDoS or scanning
        increase_rate_limits()

    elif pattern.pattern_type == PatternType.TEMPORAL:
        # Off-hours activity
        if pattern.severity >= 4:
            notify_on_call_team(pattern)

config = MemoryConfig(
    on_pattern=on_pattern,
    ...
)
```

---

## Custom Correlation Rules

### Creating Custom Rules

```python
from src.agents.smith.advanced_memory import (
    CorrelationRule,
    CorrelationType,
    IntelligenceType,
    ThreatLevel,
    create_advanced_memory,
)

# Custom rule: Detect API abuse
api_abuse_rule = CorrelationRule(
    rule_id="api_abuse_detection",
    name="API Abuse Detection",
    description="Detect repeated API failures from same source",
    correlation_type=CorrelationType.TEMPORAL,

    # Match criteria
    source_types=[IntelligenceType.SIEM_EVENT],
    categories=["api", "authentication"],
    min_severity=2,

    # Temporal settings
    time_window_minutes=15,
    min_events=10,  # 10 events in 15 minutes

    # Output
    output_threat_level=ThreatLevel.HIGH,
    tags={"api-abuse", "brute-force"},
)

manager = create_advanced_memory(auto_start=True)
manager._correlator.add_rule(api_abuse_rule)
```

### Rule Configuration Examples

**Lateral Movement Detection**:
```python
CorrelationRule(
    rule_id="lateral_movement",
    name="Lateral Movement Detection",
    correlation_type=CorrelationType.MITRE_CHAIN,
    required_tactics=["credential-access", "lateral-movement"],
    time_window_minutes=60,
    min_events=2,
    output_threat_level=ThreatLevel.HIGH,
)
```

**Multi-Source IOC**:
```python
CorrelationRule(
    rule_id="multi_source_ioc",
    name="Multi-Source IOC",
    correlation_type=CorrelationType.INDICATOR,
    source_diversity_bonus=20,  # Bonus per unique source
    min_events=3,
    output_threat_level=ThreatLevel.CRITICAL,
)
```

---

## Monitoring & Alerting

### Health Checks

```python
def check_memory_health():
    status = manager.get_status()

    health = {
        "healthy": status.healthy,
        "components": {
            "store": status.store_healthy,
            "correlator": status.correlator_active,
            "synthesizer": status.synthesizer_active,
            "baseline": status.baseline_ready,
            "boundary": status.boundary_connected,
        },
        "metrics": {
            "total_entries": status.total_entries,
            "hot_entries": status.hot_entries,
            "active_clusters": status.active_clusters,
            "anomaly_score": status.anomaly_score,
        },
    }

    return health
```

### Prometheus Metrics

```python
from prometheus_client import Gauge, Counter

# Define metrics
memory_entries = Gauge("smith_memory_entries_total", "Total entries")
memory_hot = Gauge("smith_memory_hot_entries", "Hot tier entries")
memory_clusters = Gauge("smith_memory_active_clusters", "Active clusters")
memory_anomaly = Gauge("smith_memory_anomaly_score", "Anomaly score")
events_ingested = Counter("smith_events_ingested_total", "Events ingested")

def update_prometheus_metrics():
    status = manager.get_status()
    stats = manager.get_stats()

    memory_entries.set(status.total_entries)
    memory_hot.set(status.hot_entries)
    memory_clusters.set(status.active_clusters)
    memory_anomaly.set(status.anomaly_score)
    events_ingested.inc(stats["events_processed"])
```

### Scheduled Reporting

```python
import schedule

def daily_security_report():
    summary = manager.generate_summary(period="last_24h")

    report = f"""
    Daily Security Report - {datetime.now().date()}

    Events: {summary.total_events}
    Threat Level: {summary.highest_threat_level.name}
    Active Clusters: {summary.active_threat_clusters}
    Trend: {summary.trend_direction}

    Summary: {summary.executive_summary}

    Recommended Actions:
    {format_list(summary.recommended_actions)}
    """

    send_email("security-team@company.com", "Daily Security Report", report)

schedule.every().day.at("08:00").do(daily_security_report)
```

---

## Troubleshooting

### Common Issues

**Memory not starting**:
```python
status = manager.get_status()
if not status.initialized:
    print("Check storage_path permissions")
if not status.running:
    print("Call manager.start() or use auto_start=True")
```

**Boundary connection failing**:
```python
# Check connectivity
import requests
try:
    resp = requests.get(
        f"{endpoint}/api/v1/health",
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=5,
    )
    print(f"Status: {resp.status_code}")
except Exception as e:
    print(f"Connection error: {e}")
```

**No correlations detected**:
```python
# Check if enough events exist
stats = manager.get_stats()
print(f"Events: {stats['store']['total_entries']}")
print(f"Correlations: {stats['correlations_found']}")
print(f"Rules: {stats['correlator']['rules_count']}")

# Verify time window
entries = manager.query(
    since=datetime.now() - timedelta(hours=1)
)
print(f"Events in last hour: {len(entries)}")
```

**Baseline not ready**:
```python
stats = manager._baseline.get_stats()
print(f"Ready: {stats['is_ready']}")
print(f"Learning hours: {stats['learning_hours']}")

# Force learning
manager._baseline.learn(lookback_hours=24)
```

### Debug Logging

```python
import logging

# Enable debug logging
logging.getLogger("src.agents.smith.advanced_memory").setLevel(logging.DEBUG)

# Specific components
logging.getLogger("src.agents.smith.advanced_memory.correlator").setLevel(logging.DEBUG)
logging.getLogger("src.agents.smith.advanced_memory.boundary_connector").setLevel(logging.DEBUG)
```

---

## Related Documentation

- [Technical Architecture](../technical/advanced-memory.md)
- [API Reference](../technical/advanced-memory-api.md)
- [Boundary Tools Integration](./boundary-tools.md)
- [Module README](../../src/agents/smith/advanced_memory/README.md)
