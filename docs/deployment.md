# Agent OS Production Deployment Guide

This guide covers deploying Agent OS in production environments using Docker.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Docker Deployment](#docker-deployment)
- [Monitoring Setup](#monitoring-setup)
- [Security Considerations](#security-considerations)
- [Scaling](#scaling)
- [Troubleshooting](#troubleshooting)

## Prerequisites

- Docker 20.10+ and Docker Compose v2
- At least 4GB RAM available
- 10GB disk space for data and logs
- Network access for pulling container images

## Quick Start

1. **Clone the repository:**
   ```bash
   git clone https://github.com/kase1111-hash/Agent-OS.git
   cd Agent-OS
   ```

2. **Create environment file:**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

3. **Start the services:**
   ```bash
   docker compose up -d
   ```

4. **Verify deployment:**
   ```bash
   curl http://localhost:8080/health
   ```

## Configuration

### Environment Variables

Create a `.env` file in the project root. See `.env.example` for all available options.

#### Core Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `AGENT_OS_WEB_PORT` | `8080` | HTTP port for the web interface |
| `AGENT_OS_WEB_HOST` | `0.0.0.0` | Bind address |
| `AGENT_OS_DATA_DIR` | `/app/data` | Data storage directory |
| `AGENT_OS_REQUIRE_AUTH` | `false` | Enable authentication |
| `AGENT_OS_API_KEY` | - | API key for authentication |

#### Rate Limiting

| Variable | Default | Description |
|----------|---------|-------------|
| `AGENT_OS_RATE_LIMIT_ENABLED` | `true` | Enable rate limiting |
| `AGENT_OS_RATE_LIMIT_PER_MINUTE` | `60` | Requests per minute |
| `AGENT_OS_RATE_LIMIT_PER_HOUR` | `1000` | Requests per hour |
| `AGENT_OS_RATE_LIMIT_STRATEGY` | `sliding_window` | Algorithm: `fixed_window`, `sliding_window`, `token_bucket` |
| `AGENT_OS_RATE_LIMIT_REDIS` | `true` | Use Redis for distributed rate limiting |
| `AGENT_OS_REDIS_URL` | `redis://redis:6379` | Redis connection URL |

#### Voice Features

| Variable | Default | Description |
|----------|---------|-------------|
| `AGENT_OS_STT_ENABLED` | `true` | Enable speech-to-text |
| `AGENT_OS_STT_ENGINE` | `auto` | STT engine: `auto`, `whisper`, `whisper_api` |
| `AGENT_OS_STT_MODEL` | `base` | Whisper model size |
| `AGENT_OS_TTS_ENABLED` | `true` | Enable text-to-speech |
| `AGENT_OS_TTS_ENGINE` | `auto` | TTS engine: `auto`, `piper`, `espeak` |

#### Monitoring

| Variable | Default | Description |
|----------|---------|-------------|
| `PROMETHEUS_PORT` | `9090` | Prometheus web UI port |
| `GRAFANA_PORT` | `3000` | Grafana web UI port |
| `GRAFANA_ADMIN_USER` | `admin` | Grafana admin username |
| `GRAFANA_ADMIN_PASSWORD` | `agentos` | Grafana admin password |

## Docker Deployment

### Production Build

Build the production image:

```bash
docker build --target production -t agentos:latest .
```

### Development Build

Build with hot reload for development:

```bash
docker build --target development -t agentos:dev .
```

### Running with Docker Compose

**Start all services:**
```bash
docker compose up -d
```

**View logs:**
```bash
docker compose logs -f agentos
```

**Stop services:**
```bash
docker compose down
```

**Stop and remove volumes:**
```bash
docker compose down -v
```

### Service Endpoints

After deployment, the following endpoints are available:

| Service | URL | Description |
|---------|-----|-------------|
| Agent OS | http://localhost:8080 | Main application |
| API Docs | http://localhost:8080/docs | OpenAPI documentation |
| Health Check | http://localhost:8080/health | Application health |
| Metrics | http://localhost:8080/api/observability/metrics | Prometheus metrics |
| Prometheus | http://localhost:9090 | Metrics dashboard |
| Grafana | http://localhost:3000 | Visualization dashboards |

## Monitoring Setup

### Prometheus

Prometheus is pre-configured to scrape metrics from Agent OS. Access the Prometheus UI at `http://localhost:9090`.

**Key metrics to monitor:**
- `agentos_http_requests_total` - Request count by endpoint
- `agentos_http_request_duration_seconds` - Request latency
- `agentos_active_connections` - WebSocket connections
- `agentos_agent_operations_total` - Agent operations count

### Grafana

Grafana is pre-configured with Prometheus as a datasource. Default credentials:
- Username: `admin`
- Password: `agentos` (change in production!)

Access Grafana at `http://localhost:3000`.

### Health Checks

The application exposes health endpoints:

```bash
# Quick health check
curl http://localhost:8080/health

# Detailed component health
curl http://localhost:8080/api/observability/health

# List available checks
curl http://localhost:8080/api/observability/health/checks/list
```

## Security Considerations

### Authentication

Enable authentication in production:

```bash
AGENT_OS_REQUIRE_AUTH=true
AGENT_OS_API_KEY=your-secure-api-key
```

### Network Security

1. **Use a reverse proxy** (nginx, Traefik, Caddy) for:
   - TLS termination
   - Additional rate limiting
   - Request filtering

2. **Firewall configuration:**
   - Only expose port 8080 (or reverse proxy port)
   - Keep Prometheus/Grafana internal or behind VPN

### Container Security

The production image runs as non-root user `agentos` (UID 1000).

### Secrets Management

For sensitive values, use Docker secrets or environment variable injection from a vault:

```yaml
# docker-compose.override.yml
services:
  agentos:
    secrets:
      - api_key
    environment:
      - AGENT_OS_API_KEY_FILE=/run/secrets/api_key

secrets:
  api_key:
    external: true
```

## Scaling

### Horizontal Scaling

For high availability, run multiple Agent OS instances behind a load balancer:

```yaml
# docker-compose.override.yml
services:
  agentos:
    deploy:
      replicas: 3
```

**Requirements for horizontal scaling:**
- Use Redis for rate limiting (`AGENT_OS_RATE_LIMIT_REDIS=true`)
- Use external session storage
- Configure sticky sessions for WebSocket connections

### Resource Limits

Default resource limits in docker-compose.yml:

| Service | CPU Limit | Memory Limit |
|---------|-----------|--------------|
| agentos | 2.0 cores | 2GB |
| redis | 0.5 cores | 512MB |
| prometheus | 0.5 cores | 512MB |
| grafana | 0.5 cores | 256MB |

Adjust based on your workload in `docker-compose.override.yml`.

## Troubleshooting

### Container Won't Start

Check logs:
```bash
docker compose logs agentos
```

Common issues:
- Port already in use: Change `AGENT_OS_WEB_PORT`
- Permission denied: Check volume mount permissions
- Missing dependencies: Rebuild image

### Health Check Failing

```bash
# Check application logs
docker compose logs --tail=100 agentos

# Check component health
curl http://localhost:8080/api/observability/health
```

### High Memory Usage

- Check for memory leaks in traces
- Reduce `--maxmemory` in Redis
- Adjust container memory limits

### Rate Limiting Issues

Check rate limit status:
```bash
curl -I http://localhost:8080/api/chat/conversations
# Look for X-RateLimit-* headers
```

If using Redis:
```bash
docker compose exec redis redis-cli KEYS "ratelimit:*"
```

### Resetting Data

To start fresh:
```bash
docker compose down -v
docker compose up -d
```

**Warning:** This removes all persistent data including user data and logs.
