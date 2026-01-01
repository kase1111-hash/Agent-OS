# Running & Compiling Agent-OS

This guide covers how to install, build, run, and test Agent-OS.

## Prerequisites

- **Python**: 3.10 or later
- **RAM**: 4GB minimum (development), 2GB+ (deployment)
- **Disk**: 10GB minimum
- **Docker** (optional): 20.10+ with Docker Compose v2

---

## Quick Start

### Option 1: Local Development

```bash
# Clone and enter directory
git clone https://github.com/kase1111-hash/Agent-OS.git
cd Agent-OS

# Install dependencies
pip install -r requirements.txt

# Run the application
python -m uvicorn src.web.app:get_app --factory --host 0.0.0.0 --port 8080
```

### Option 2: Docker (Recommended)

```bash
# Setup environment
cp .env.example .env

# Build and run
docker compose up -d

# Verify
curl http://localhost:8080/health
```

---

## Installation Options

### Core Installation

```bash
pip install -r requirements.txt
```

### Development Installation (Editable)

```bash
pip install -e .
```

### With Optional Dependencies

```bash
# Development tools (pytest, black, mypy, etc.)
pip install -e ".[dev]"

# Redis support (distributed rate limiting)
pip install -e ".[redis]"

# Observability (Prometheus, OpenTelemetry)
pip install -e ".[observability]"

# All optional dependencies
pip install -e ".[dev,redis,observability]"
```

---

## Building

### Build Python Package

```bash
# Build distribution packages
python -m build
```

### Build Docker Images

```bash
# Production image
docker build --target production -t agentos:latest .

# Development image (with hot reload)
docker build --target development -t agentos:dev .

# Using docker-compose
docker compose build
```

### Build Standalone Executable

```bash
# Install build dependencies
pip install -e ".[build]"

# Create standalone executable
pyinstaller --onefile src/web/app.py
```

---

## Running

### CLI Commands

The project provides three CLI entry points:

| Command | Description |
|---------|-------------|
| `agent-os` | Main web application |
| `agent-os-install` | Cross-platform installer |
| `agent-os-ceremony` | 8-phase initialization ceremony |

### Running the Web Application

```bash
# Direct execution
python -m uvicorn src.web.app:get_app --factory --host 0.0.0.0 --port 8080

# With auto-reload (development)
python -m uvicorn src.web.app:get_app --factory --reload

# Using installed command
agent-os
```

### Running with Docker Compose

```bash
# Start all services
docker compose up -d

# View logs
docker compose logs -f agentos

# Stop services
docker compose down

# Stop and remove volumes
docker compose down -v
```

### Accessing the Application

| Endpoint | URL |
|----------|-----|
| Main App | http://localhost:8080 |
| API Docs (Swagger) | http://localhost:8080/docs |
| Health Check | http://localhost:8080/health |
| Prometheus Metrics | http://localhost:8080/api/observability/metrics |
| Prometheus UI | http://localhost:9090 |
| Grafana | http://localhost:3000 (admin/agentos) |

---

## Testing

### Run All Tests

```bash
pytest tests/
```

### Run Specific Tests

```bash
# Single test file
pytest tests/test_kernel.py

# With verbose output
pytest -v tests/

# With coverage report
pytest --cov=src tests/
```

### Run End-to-End Simulation

```bash
python -m tests.e2e_simulation
```

### Run Benchmarks

```bash
pytest benchmarks/
```

---

## Configuration

### Environment Variables

Create a `.env` file (copy from `.env.example`):

```bash
# Core Settings
AGENT_OS_WEB_PORT=8080
AGENT_OS_WEB_HOST=0.0.0.0
AGENT_OS_DATA_DIR=/app/data
AGENT_OS_WEB_DEBUG=false

# Ollama AI Backend
OLLAMA_ENDPOINT=http://localhost:11434
OLLAMA_MODEL=mistral
OLLAMA_TIMEOUT=120

# Boundary Security
AGENT_OS_BOUNDARY_NETWORK_ALLOWED=true  # Allow network access to Ollama

# Authentication & Rate Limiting
AGENT_OS_REQUIRE_AUTH=false
AGENT_OS_RATE_LIMIT_ENABLED=true
AGENT_OS_RATE_LIMIT_PER_MINUTE=60
AGENT_OS_RATE_LIMIT_REDIS=true
AGENT_OS_REDIS_URL=redis://redis:6379

# Voice Features
AGENT_OS_STT_ENABLED=true
AGENT_OS_TTS_ENABLED=true

# Monitoring
PROMETHEUS_PORT=9090
GRAFANA_PORT=3000
GRAFANA_ADMIN_USER=admin
GRAFANA_ADMIN_PASSWORD=agentos
```

---

## Initialization Ceremony

Before production use, run the 8-phase Bring-Home Ceremony:

```bash
agent-os-ceremony
```

This initializes:
1. Cold boot verification
2. Boundary initialization
3. Constitution loading
4. Agent awakening
5. Memory consent
6. Federation setup

---

## Project Structure

```
Agent-OS/
├── src/                    # Source code (~50,000 lines)
│   ├── agents/             # 6 AI agents (whisper, smith, seshat, sage, quill, muse)
│   ├── kernel/             # Constitutional kernel
│   ├── memory/             # Encrypted memory vault
│   ├── messaging/          # Inter-agent communication
│   ├── boundary/           # Security enforcement
│   ├── web/                # FastAPI web interface
│   ├── ceremony/           # Bring-Home initialization
│   ├── contracts/          # Learning contracts
│   ├── tools/              # Tool integration
│   ├── mobile/             # Mobile backend API
│   ├── voice/              # Voice interface
│   ├── federation/         # Multi-node federation
│   ├── installer/          # Cross-platform installer
│   └── sdk/                # Agent SDK
├── tests/                  # 31 test modules
├── docs/                   # Documentation
├── agents/                 # Constitutional YAML files
├── Dockerfile              # Multi-stage Docker build
├── docker-compose.yml      # Full stack configuration
├── pyproject.toml          # Project configuration
└── requirements.txt        # Core dependencies
```

---

## Troubleshooting

### Port Already in Use

```bash
# Find process using port 8080
lsof -i :8080

# Kill process
kill -9 <PID>
```

### Redis Connection Failed

Ensure Redis is running or disable Redis rate limiting:

```bash
AGENT_OS_RATE_LIMIT_REDIS=false
```

### Missing Dependencies

```bash
pip install -r requirements.txt
pip install -e ".[dev]"
```

### Docker Build Fails

```bash
# Clean build
docker compose build --no-cache

# Check Docker logs
docker compose logs
```

---

## Windows Quick Start

For Windows users, we provide convenient batch scripts:

### First-Time Setup

1. Install [Python 3.10+](https://www.python.org/downloads/) (check "Add Python to PATH")
2. Install [Ollama](https://ollama.com/download)
3. Double-click `build.bat`

### Running

1. Double-click `start.bat`
2. Open http://localhost:8080 in your browser

See [START_HERE.md](../START_HERE.md) for detailed Windows instructions.

---

## Version Information

| Component | Version |
|-----------|---------|
| Agent-OS | 1.0 |
| Python | 3.10+ |
| FastAPI | 0.115+ |
| Pydantic | 2.10+ |
| Docker | 20.10+ (optional) |

---

*Last Updated: January 2026*
*License: CC0 1.0 Universal (Public Domain)*
