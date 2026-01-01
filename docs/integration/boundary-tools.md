# Boundary Tools Integration Guide

This guide covers the integration of external Boundary security tools with Agent OS. These tools are **optional** but recommended for enterprise and high-security deployments.

## Overview

Agent OS supports two complementary external security tools:

| Tool | Purpose | Repository |
|------|---------|------------|
| **Boundary Daemon** | Trust policy & audit layer | [kase1111-hash/boundary-daemon-](https://github.com/kase1111-hash/boundary-daemon-) |
| **Boundary SIEM** | Security analytics & compliance | [kase1111-hash/Boundary-SIEM](https://github.com/kase1111-hash/Boundary-SIEM) |

These tools work independently of Agent OS but provide enhanced security when integrated.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Agent OS                                     │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                    Agent Smith                               │   │
│  │  ┌─────────────────────────────────────────────────────┐    │   │
│  │  │              Attack Detection System                 │    │   │
│  │  │                                                      │    │   │
│  │  │  ┌──────────────┐  ┌──────────────┐  ┌───────────┐  │    │   │
│  │  │  │ Pattern      │  │ LLM          │  │ SIEM      │  │    │   │
│  │  │  │ Matching     │  │ Analyzer     │  │ Connector │  │    │   │
│  │  │  └──────────────┘  └──────────────┘  └─────┬─────┘  │    │   │
│  │  └────────────────────────────────────────────┼────────┘    │   │
│  └───────────────────────────────────────────────┼─────────────┘   │
│                                                   │                 │
└───────────────────────────────────────────────────┼─────────────────┘
                                                    │
        ┌───────────────────────────────────────────┼───────────────────┐
        │                                           │                   │
        ▼                                           ▼                   ▼
┌───────────────────┐                  ┌─────────────────────────────────┐
│  Boundary Daemon  │                  │        Boundary SIEM            │
│  (Policy Layer)   │─────────────────▶│     (Analytics Platform)        │
├───────────────────┤                  ├─────────────────────────────────┤
│ • 6 Security      │   Audit Events   │ • ClickHouse Storage            │
│   Modes           │                  │ • 103 Detection Rules           │
│ • Process         │   Log Shipping   │ • MITRE ATT&CK Mapping          │
│   Sandboxing      │                  │ • Compliance Reports            │
│ • Tripwires       │                  │ • Event Correlation             │
│ • Cryptographic   │                  │ • Threat Intelligence           │
│   Audit Logs      │                  │                                 │
└───────────────────┘                  └─────────────────────────────────┘
      Optional                                    Optional
```

## Boundary Daemon

### What is Boundary Daemon?

Boundary Daemon is the **Trust Policy & Audit Layer** for Agent OS. It provides system-level security enforcement that complements Agent Smith's request-level validation.

### Key Capabilities

| Capability | Description |
|------------|-------------|
| **Environment Sensing** | Monitors network, hardware, and process states continuously |
| **Mode Enforcement** | Implements 6 boundary modes with progressively restrictive controls |
| **Recall Gating** | Controls memory access based on classification levels (PUBLIC to CROWN_JEWEL) |
| **Execution Gating** | Restricts available tools, I/O operations, and AI models |
| **Tripwire Response** | Automatic lockdown on security violations |
| **Cryptographic Audit** | SHA-256 hash-chained, Ed25519-signed immutable logs |

### Boundary Modes

| Mode | Network | Memory Access | Tools | Use Case |
|------|---------|---------------|-------|----------|
| **OPEN** | Online | Classes 0-1 | All | Casual browsing |
| **RESTRICTED** | Online | Classes 0-2 | Most | Research work |
| **TRUSTED** | VPN only | Classes 0-3 | No USB | Serious work |
| **AIRGAP** | Offline | Classes 0-4 | No network tools | High-value IP |
| **COLDROOM** | Offline | Classes 0-5 | Display only | Crown jewels |
| **LOCKDOWN** | Blocked | None | None | Emergency |

### Installation

```bash
# Clone the repository
git clone https://github.com/kase1111-hash/boundary-daemon-.git
cd boundary-daemon-

# Review the installation script
less install.sh

# Install (requires root for system-level controls)
sudo ./install.sh

# Configure for Agent OS
sudo cp configs/agent-os.yaml /etc/boundary-daemon/config.d/

# Start the daemon
sudo systemctl enable boundary-daemon
sudo systemctl start boundary-daemon

# Verify operation
boundaryctl status
```

### Agent OS Integration

Add to your Agent OS attack detection configuration:

```yaml
# config/attack_detection.yaml
attack_detection:
  detector:
    enable_boundary_events: true

  boundary_daemon:
    enabled: true
    socket_path: /var/run/boundary-daemon/smith.sock

    # Events to monitor
    event_types:
      - tripwire_triggered
      - mode_transition
      - policy_violation
      - audit_event
      - sandbox_escape_attempt
      - memory_access_violation

    # Automatic responses
    auto_lockdown_on_critical: true
    lockdown_cooldown_seconds: 300

    # Event filtering
    min_severity: medium
    exclude_categories:
      - debug
      - info
```

### Event Types

| Event Type | Description | Severity |
|------------|-------------|----------|
| `tripwire_triggered` | Security violation detected | HIGH-CRITICAL |
| `mode_transition` | Security mode changed | INFO-MEDIUM |
| `policy_violation` | Policy rule violated | MEDIUM-HIGH |
| `audit_event` | Audit log entry | LOW-MEDIUM |
| `sandbox_escape_attempt` | Process tried to escape sandbox | CRITICAL |
| `memory_access_violation` | Unauthorized memory access | HIGH |

### CLI Tools

Boundary Daemon provides several command-line tools:

```bash
# Main daemon control
boundaryctl status              # Show current status
boundaryctl mode RESTRICTED     # Change security mode
boundaryctl tripwires list      # List active tripwires
boundaryctl tripwires test      # Test tripwire triggers

# Sandbox management
sandboxctl create --profile high-security app-name
sandboxctl list
sandboxctl destroy app-name

# Authentication management
authctl tokens list
authctl tokens revoke <token-id>

# Security scanning
security_scan --full /path/to/scan
```

## Boundary SIEM

### What is Boundary SIEM?

Boundary SIEM is an enterprise-grade security information and event management platform built with Go and ClickHouse, providing high-performance event processing and analytics.

### Key Capabilities

| Capability | Description |
|------------|-------------|
| **Event Ingestion** | CEF, JSON HTTP, RFC 5424 syslog support |
| **Ring Buffer** | 100K event queue with backpressure handling |
| **ClickHouse Storage** | Time-partitioned analytics database |
| **Detection Rules** | 103 rules with MITRE ATT&CK mapping |
| **Correlation Engine** | Time windows and sequence detection |
| **Compliance** | SOC 2, ISO 27001, NIST CSF, PCI DSS, GDPR |

### Tiered Retention

| Tier | Duration | Storage |
|------|----------|---------|
| **Hot** | 7 days | Fast SSD |
| **Warm** | 30 days | Standard storage |
| **Cold** | 365 days | Compressed archive |
| **Archive** | Indefinite | S3/Object storage |

### Installation

```bash
# Clone the repository
git clone https://github.com/kase1111-hash/Boundary-SIEM.git
cd Boundary-SIEM

# Start with Docker Compose (recommended)
docker-compose up -d

# Or install manually
go build -o boundary-siem ./cmd/server
./boundary-siem --config configs/config.yaml

# Verify operation
curl http://localhost:8080/api/v1/health
```

### Agent OS Integration

Add Boundary SIEM as a SIEM source:

```yaml
# config/attack_detection.yaml
attack_detection:
  siem:
    sources:
      - name: boundary-siem
        provider: boundary_siem
        endpoint: ${BOUNDARY_SIEM_URL}  # e.g., http://localhost:8080
        api_key: ${BOUNDARY_SIEM_API_KEY}
        poll_interval: 30
        batch_size: 100
        verify_ssl: true

        extra_params:
          # Filter for Agent OS-related events
          query_filter: "source:agent-os OR source:boundary-daemon OR source:smith-agent"
          severity_min: medium

          # Include correlated alerts
          include_alerts: true
          alert_status: open
```

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/events` | GET | List events with filtering |
| `/api/v1/events` | POST | Create new event |
| `/api/v1/events/{id}` | GET | Get event details |
| `/api/v1/alerts` | GET | List correlated alerts |
| `/api/v1/alerts/{id}` | GET | Get alert details |
| `/api/v1/alerts/{id}/acknowledge` | POST | Acknowledge alert |
| `/api/v1/alerts/{id}/close` | POST | Close alert |
| `/api/v1/search` | POST | Search events |
| `/api/v1/health` | GET | Health check |
| `/graphql` | POST | GraphQL queries |
| `/ws/alerts` | WebSocket | Real-time alert streaming |

### Environment Variables

```bash
# Boundary SIEM connection
export BOUNDARY_SIEM_URL="http://localhost:8080"
export BOUNDARY_SIEM_API_KEY="your-api-key-here"

# Optional: Custom CA for SSL verification
export BOUNDARY_SIEM_CA_CERT="/path/to/ca.crt"
```

### Sending Events to Boundary SIEM

Agent OS can send events to Boundary SIEM for centralized analysis:

```yaml
# config/attack_detection.yaml
attack_detection:
  notifications:
    channels:
      - name: boundary-siem-events
        type: webhook
        endpoint: ${BOUNDARY_SIEM_URL}/api/v1/events
        method: POST
        headers:
          Authorization: "Bearer ${BOUNDARY_SIEM_API_KEY}"
          Content-Type: application/json
        min_severity: low
        payload_template: |
          {
            "timestamp": "{{ .Timestamp }}",
            "source": "agent-os",
            "event_type": "{{ .Type }}",
            "severity": "{{ .Severity }}",
            "message": "{{ .Description }}",
            "metadata": {
              "attack_id": "{{ .AttackID }}",
              "pattern_id": "{{ .PatternID }}",
              "agent": "smith"
            }
          }
```

## Combined Deployment

### Full Security Stack

For maximum security, deploy all components together:

```yaml
# docker-compose.yaml (example)
version: '3.8'

services:
  agent-os:
    build: .
    ports:
      - "8000:8000"
    environment:
      - BOUNDARY_DAEMON_SOCKET=/var/run/boundary-daemon/smith.sock
      - BOUNDARY_SIEM_URL=http://boundary-siem:8080
      - BOUNDARY_SIEM_API_KEY=${BOUNDARY_SIEM_API_KEY}
    volumes:
      - /var/run/boundary-daemon:/var/run/boundary-daemon:ro
    depends_on:
      - boundary-siem

  boundary-siem:
    image: boundary-siem:latest
    ports:
      - "8080:8080"
    volumes:
      - siem-data:/data
    environment:
      - CLICKHOUSE_HOST=clickhouse
    depends_on:
      - clickhouse

  clickhouse:
    image: clickhouse/clickhouse-server:latest
    volumes:
      - clickhouse-data:/var/lib/clickhouse

  # Boundary Daemon runs on host (system-level)
  # Install separately: sudo ./install.sh

volumes:
  siem-data:
  clickhouse-data:
```

### Event Flow

1. **Agent OS Request** → Agent Smith validates
2. **Smith Detection** → Attack detected, event generated
3. **Boundary Daemon** → System-level policy check
4. **Boundary SIEM** → Event stored, correlated, analyzed
5. **Alert Generated** → Notification sent to Smith
6. **Remediation** → Patch generated, PR created

### Recommended Configuration

```yaml
# Full integration configuration
attack_detection:
  enabled: true
  severity_threshold: low

  detector:
    enable_boundary_events: true
    enable_siem_events: true
    auto_lockdown_on_critical: false
    detection_confidence_threshold: 0.7

  # External Boundary Daemon
  boundary_daemon:
    enabled: true
    socket_path: /var/run/boundary-daemon/smith.sock
    event_types:
      - tripwire_triggered
      - mode_transition
      - policy_violation
      - sandbox_escape_attempt
    auto_lockdown_on_critical: true

  # External Boundary SIEM
  siem:
    sources:
      - name: boundary-siem
        provider: boundary_siem
        endpoint: ${BOUNDARY_SIEM_URL}
        api_key: ${BOUNDARY_SIEM_API_KEY}
        poll_interval: 30
        extra_params:
          query_filter: "source:agent-os OR source:boundary-daemon"
          severity_min: low
          include_alerts: true

  # Multi-channel notifications
  notifications:
    channels:
      # Send to Boundary SIEM
      - name: boundary-siem-ingest
        type: webhook
        endpoint: ${BOUNDARY_SIEM_URL}/api/v1/events
        headers:
          Authorization: "Bearer ${BOUNDARY_SIEM_API_KEY}"
        min_severity: low

      # Also notify team
      - name: slack-security
        type: slack
        webhook_url: ${SLACK_WEBHOOK_URL}
        min_severity: high

  analyzer:
    enable_llm_analysis: true
    use_sage_agent: true
    mitre_mapping_enabled: true

  remediation:
    enabled: true
    auto_generate_patches: true
    require_approval: true
    test_patches_in_sandbox: true

  storage:
    backend: sqlite
    path: ./data/attack_detection.db
```

## Troubleshooting

### Boundary Daemon Issues

```bash
# Check daemon status
sudo systemctl status boundary-daemon
sudo journalctl -u boundary-daemon -f

# Verify socket exists
ls -la /var/run/boundary-daemon/smith.sock

# Test connection
boundaryctl status

# Check logs
sudo tail -f /var/log/boundary-daemon/daemon.log
```

### Boundary SIEM Issues

```bash
# Check container status
docker-compose ps
docker-compose logs boundary-siem

# Test API
curl http://localhost:8080/api/v1/health

# Check ClickHouse
docker-compose exec clickhouse clickhouse-client \
  --query "SELECT count() FROM security_events"
```

### Agent OS Integration Issues

```python
# Test SIEM connector
from src.agents.smith.attack_detection.siem_connector import (
    SIEMConnector, SIEMConfig, SIEMProvider
)

config = SIEMConfig(
    provider=SIEMProvider.BOUNDARY_SIEM,
    endpoint="http://localhost:8080",
    api_key="your-api-key",
)

connector = SIEMConnector()
connector.add_source("boundary-siem", config)
results = connector.connect_all()
print(f"Connection results: {results}")

events = connector.fetch_immediate(lookback_minutes=60)
print(f"Fetched {len(events)} events")
```

## Security Considerations

1. **API Keys**: Store in environment variables or secrets manager, never in config files
2. **TLS**: Always use HTTPS in production
3. **Network Isolation**: Run Boundary SIEM on internal network only
4. **Access Control**: Use Boundary SIEM's RBAC for API access
5. **Audit Logs**: Enable and monitor all audit logs
6. **Backup**: Regularly backup ClickHouse data and configs

## References

- [Boundary Daemon Repository](https://github.com/kase1111-hash/boundary-daemon-)
- [Boundary SIEM Repository](https://github.com/kase1111-hash/Boundary-SIEM)
- [Agent OS Security Policy](../governance/security.md)
- [Attack Detection System](../governance/security.md#attack-detection)
- [External Tools Integration](../governance/security.md#external-tools)

---

*Document Version: 1.0*
*Last Updated: January 2026*
*License: CC0 1.0 Universal (Public Domain)*
