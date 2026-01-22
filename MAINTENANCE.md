# Maintenance Guide

This guide covers operational maintenance for Agent-OS deployments.

## Regular Maintenance Tasks

### Daily Tasks

| Task | Command/Action | Purpose |
|------|----------------|---------|
| Health Check | `curl http://localhost:8080/health` | Verify system is running |
| Check Logs | Review `/api/system/logs` | Identify issues early |
| Monitor Memory | System monitoring tools | Prevent resource exhaustion |

### Weekly Tasks

| Task | Command/Action | Purpose |
|------|----------------|---------|
| Review Security Alerts | `/api/security/attacks` | Address security issues |
| Check Disk Usage | `df -h` / Disk Management | Prevent storage issues |
| Update Dependencies | `pip list --outdated` | Security updates |
| Backup Data | See Backup section | Data protection |

### Monthly Tasks

| Task | Command/Action | Purpose |
|------|----------------|---------|
| Full Backup | Complete data directory backup | Disaster recovery |
| Review Constitution | Manual review | Governance audit |
| Update Ollama Models | `ollama pull <model>` | Security/performance |
| Performance Review | Check metrics | Optimization |

## Monitoring

### Health Endpoints

```bash
# Basic health check
curl http://localhost:8080/health

# Detailed health with component status
curl http://localhost:8080/api/observability/health

# Specific component health
curl http://localhost:8080/api/observability/health/database
```

### Metrics

```bash
# Prometheus format
curl http://localhost:8080/api/observability/metrics

# JSON format
curl http://localhost:8080/api/observability/metrics/json

# System metrics
curl http://localhost:8080/api/system/metrics
```

### Key Metrics to Monitor

| Metric | Warning Threshold | Critical Threshold |
|--------|-------------------|-------------------|
| Memory Usage | > 80% | > 95% |
| CPU Usage | > 70% sustained | > 90% sustained |
| Disk Usage | > 80% | > 95% |
| Response Time | > 5 seconds | > 30 seconds |
| Error Rate | > 1% | > 5% |

### Setting Up Alerting

With Docker deployment, Prometheus and Grafana are included:

- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3000

Import the provided dashboards from `deploy/grafana/`.

## Backup and Recovery

### What to Backup

| Component | Location | Priority |
|-----------|----------|----------|
| Constitution | `./CONSTITUTION.md` | Critical |
| Memory Database | `./data/*.db` | High |
| Configuration | `./.env`, `./config/` | High |
| Attack Detection DB | `./data/attack_detection.db` | Medium |
| Conversations | Via `/api/chat/export` | Medium |
| Custom Agents | `./agents/` | Medium |

### Backup Procedures

**Manual Backup**:
```bash
# Create backup directory
mkdir -p backups/$(date +%Y%m%d)

# Backup data directory
cp -r data/ backups/$(date +%Y%m%d)/data/

# Backup configuration
cp .env backups/$(date +%Y%m%d)/
cp CONSTITUTION.md backups/$(date +%Y%m%d)/
cp -r agents/ backups/$(date +%Y%m%d)/agents/

# Export conversations (while running)
curl http://localhost:8080/api/chat/export > backups/$(date +%Y%m%d)/conversations.json

# Export memories
curl http://localhost:8080/api/memory/export > backups/$(date +%Y%m%d)/memories.json
```

**Automated Backup Script**:
```bash
#!/bin/bash
# backup.sh - Run via cron

BACKUP_DIR="/path/to/backups/$(date +%Y%m%d)"
AGENT_OS_DIR="/path/to/Agent-OS"

mkdir -p "$BACKUP_DIR"

# Stop for consistent backup (optional)
# systemctl stop agent-os

# Backup files
tar -czf "$BACKUP_DIR/agent-os-backup.tar.gz" \
    -C "$AGENT_OS_DIR" \
    data/ .env CONSTITUTION.md agents/

# Restart if stopped
# systemctl start agent-os

# Cleanup old backups (keep 30 days)
find /path/to/backups -type d -mtime +30 -exec rm -rf {} +
```

**Cron Schedule**:
```cron
# Daily backup at 2 AM
0 2 * * * /path/to/backup.sh
```

### Recovery Procedures

**Full Recovery**:
```bash
# Stop Agent-OS
systemctl stop agent-os

# Restore from backup
tar -xzf /path/to/backup/agent-os-backup.tar.gz -C /path/to/Agent-OS

# Start Agent-OS
systemctl start agent-os

# Verify
curl http://localhost:8080/health
```

**Selective Recovery**:
```bash
# Restore only memories
curl -X POST http://localhost:8080/api/chat/import \
    -H "Content-Type: application/json" \
    -d @backups/20240115/conversations.json
```

## Updates

### Updating Agent-OS

```bash
# 1. Backup current installation
./backup.sh

# 2. Pull latest changes
git fetch origin
git checkout main
git pull origin main

# 3. Update dependencies
pip install -r requirements.txt --upgrade

# 4. Run migrations if any
python -m src.migrations.run

# 5. Restart
systemctl restart agent-os

# 6. Verify
curl http://localhost:8080/health
```

### Updating Ollama Models

```bash
# List current models
ollama list

# Update a model
ollama pull mistral

# Remove old model version (optional)
ollama rm mistral:old-version
```

### Updating Dependencies

```bash
# Check for outdated packages
pip list --outdated

# Update specific package
pip install package-name --upgrade

# Update all (careful in production)
pip install -r requirements.txt --upgrade
```

## Log Management

### Log Locations

| Log Type | Location | Retention |
|----------|----------|-----------|
| Application | stdout/journald | System default |
| Agent Logs | `/api/agents/logs` | In-memory |
| Audit Logs | `data/audit.log` | Configure in settings |
| Security Logs | `data/attack_detection.db` | 90 days default |

### Log Rotation

For file-based logging, configure logrotate:

```
# /etc/logrotate.d/agent-os
/var/log/agent-os/*.log {
    daily
    rotate 30
    compress
    delaycompress
    missingok
    notifempty
    create 0640 www-data www-data
}
```

### Viewing Logs

```bash
# System logs (systemd)
journalctl -u agent-os -f

# Docker logs
docker compose logs -f agent-os

# API logs
curl http://localhost:8080/api/system/logs?limit=100
```

## Database Maintenance

### SQLite Optimization

```bash
# Vacuum database (reclaim space)
sqlite3 data/agent-os.db "VACUUM;"

# Analyze for query optimization
sqlite3 data/agent-os.db "ANALYZE;"

# Check integrity
sqlite3 data/agent-os.db "PRAGMA integrity_check;"
```

### Cleanup Old Data

```bash
# Clear old attack records (via API)
curl -X DELETE "http://localhost:8080/api/intent-log?older_than=90d"

# Clear old conversations
curl -X DELETE "http://localhost:8080/api/chat/conversations?archived=true&older_than=180d"
```

## Security Maintenance

### Security Checklist

- [ ] Review security alerts weekly
- [ ] Update dependencies monthly
- [ ] Rotate secrets/tokens quarterly
- [ ] Review user access quarterly
- [ ] Audit constitution changes
- [ ] Test backups monthly

### Reviewing Security Alerts

```bash
# List recent attacks
curl "http://localhost:8080/api/security/attacks?limit=50"

# Check for false positives
curl "http://localhost:8080/api/security/attacks?false_positive=false"

# Review recommendations
curl "http://localhost:8080/api/security/recommendations"
```

### Updating Security Patterns

```bash
# List current patterns
curl http://localhost:8080/api/security/patterns

# Enable/disable patterns as needed
curl -X POST http://localhost:8080/api/security/patterns/pattern-id/disable
```

## Performance Tuning

### Configuration Optimization

```bash
# .env optimizations

# Increase worker count (for multi-core)
WORKERS=4

# Adjust memory limits
MAX_MEMORY_MB=4096

# Configure connection pool
DB_POOL_SIZE=10
```

### Resource Monitoring

```bash
# Check system resources
htop

# Check Python memory usage
python -c "import psutil; print(psutil.Process().memory_info())"

# Monitor disk I/O
iotop
```

## Disaster Recovery

### Recovery Time Objectives

| Scenario | RTO Target | RPO Target |
|----------|------------|------------|
| Minor outage | 15 minutes | 0 (no data loss) |
| Hardware failure | 4 hours | 24 hours |
| Data corruption | 8 hours | 24 hours |
| Complete loss | 24 hours | Last backup |

### Recovery Procedures

**Scenario: Application Crash**
1. Check logs for error
2. Restart application
3. Monitor for recurrence

**Scenario: Database Corruption**
1. Stop application
2. Restore database from backup
3. Verify integrity
4. Restart application

**Scenario: Complete Server Loss**
1. Provision new server
2. Install dependencies
3. Restore from backup
4. Update DNS/networking
5. Verify operation

## Automation

### Systemd Service

```ini
# /etc/systemd/system/agent-os.service
[Unit]
Description=Agent-OS
After=network.target ollama.service

[Service]
Type=simple
User=agent-os
WorkingDirectory=/opt/agent-os
ExecStart=/opt/agent-os/venv/bin/python -m uvicorn src.web.app:get_app --factory --host 0.0.0.0 --port 8080
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

### Maintenance Scripts

Create scripts in a `maintenance/` directory:

- `backup.sh` - Automated backup
- `healthcheck.sh` - Health monitoring
- `cleanup.sh` - Log and data cleanup
- `update.sh` - Update procedure

---

**Last Updated**: January 2026

For issues during maintenance, see [TROUBLESHOOTING.md](./TROUBLESHOOTING.md).
