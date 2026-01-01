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

### Installation Issues

#### Python Version Errors

**Error:** `Python 3.10 or higher is required`

```bash
# Check your Python version
python3 --version

# Install correct version (Ubuntu/Debian)
sudo apt update && sudo apt install python3.11 python3.11-venv python3.11-pip

# Install correct version (macOS)
brew install python@3.11

# Install correct version (Windows)
# Download from https://python.org and reinstall
```

#### pip Installation Fails

**Error:** `Could not build wheels for...` or `Failed building wheel`

```bash
# Upgrade pip first
pip install --upgrade pip setuptools wheel

# Install build dependencies (Linux)
sudo apt install python3-dev build-essential libffi-dev

# Install build dependencies (macOS)
xcode-select --install

# Retry installation
pip install -r requirements.txt
```

#### Virtual Environment Issues

**Error:** `No module named venv` or `virtualenv not found`

```bash
# Install venv (Ubuntu/Debian)
sudo apt install python3-venv

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # Linux/macOS
# or: venv\Scripts\activate  # Windows
```

### Runtime Issues

#### Port Already in Use

**Error:** `Address already in use` or `Port 8080 is already in use`

```bash
# Find process using port (Linux/macOS)
lsof -i :8080
# or
netstat -tulpn | grep 8080

# Find process (Windows)
netstat -ano | findstr :8080

# Kill the process
kill -9 <PID>  # Linux/macOS
# or: taskkill /PID <PID> /F  # Windows

# Or use a different port
AGENT_OS_WEB_PORT=8081 python -m uvicorn src.web.app:get_app --factory --port 8081
```

#### Redis Connection Failed

**Error:** `Connection refused` or `Redis connection error`

```bash
# Option 1: Disable Redis (use in-memory rate limiting)
export AGENT_OS_RATE_LIMIT_REDIS=false

# Option 2: Start Redis (Docker)
docker run -d --name redis -p 6379:6379 redis:7-alpine

# Option 3: Start Redis (Ubuntu)
sudo apt install redis-server
sudo systemctl start redis

# Verify Redis connection
redis-cli ping  # Should respond: PONG
```

#### Ollama Not Detected

**Error:** `Ollama not running` or `Connection refused to localhost:11434`

```bash
# Check if Ollama is installed
which ollama

# Install Ollama (Linux)
curl -fsSL https://ollama.com/install.sh | sh

# Start Ollama
ollama serve

# Verify Ollama is running
curl http://localhost:11434/api/tags

# Pull a model if none available
ollama pull mistral
```

#### Module Not Found

**Error:** `ModuleNotFoundError: No module named 'src'`

```bash
# Ensure you're in the project directory
cd /path/to/Agent-OS

# Install in development mode
pip install -e .

# Or set PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

### Docker Issues

#### Docker Build Fails

**Error:** `Failed to build` or `COPY failed`

```bash
# Clean build (remove cache)
docker compose build --no-cache

# Check Docker daemon is running
docker info

# Clear Docker cache
docker system prune -a

# Check disk space
df -h
```

#### Container Won't Start

**Error:** `Container exited with code 1`

```bash
# Check container logs
docker compose logs agentos

# Check if ports are available
docker compose ps

# Verify environment file exists
ls -la .env

# Recreate containers
docker compose down && docker compose up -d
```

#### GRAFANA_ADMIN_PASSWORD Not Set

**Error:** `GRAFANA_ADMIN_PASSWORD must be set`

```bash
# Set the password in your environment
export GRAFANA_ADMIN_PASSWORD="your-secure-password-here"

# Or add to .env file
echo 'GRAFANA_ADMIN_PASSWORD=your-secure-password-here' >> .env

# Generate a secure password
openssl rand -base64 32
```

### Voice/Audio Issues

#### Microphone Not Detected

**Error:** `No microphone found` or `Audio device not available`

```bash
# Linux: Install audio dependencies
sudo apt install portaudio19-dev python3-pyaudio

# macOS: Install with Homebrew
brew install portaudio
pip install pyaudio

# Check available devices
python -c "import pyaudio; p = pyaudio.PyAudio(); print([p.get_device_info_by_index(i)['name'] for i in range(p.get_device_count())])"
```

#### TTS/STT Not Working

**Error:** `Whisper model not found` or `TTS engine error`

```bash
# Disable voice features if not needed
export AGENT_OS_STT_ENABLED=false
export AGENT_OS_TTS_ENABLED=false

# Or install voice dependencies
pip install openai-whisper piper-tts
```

### Performance Issues

#### High Memory Usage

```bash
# Use smaller AI models
ollama pull phi  # 2.7B parameters instead of 7B

# Increase swap space (Linux)
sudo fallocate -l 8G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile

# Monitor memory usage
htop
```

#### Slow Response Times

```bash
# Use GPU acceleration if available
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# Use quantized models
ollama pull mistral:q4_0  # 4-bit quantized
```

### Test Failures

#### Tests Skipped

**Message:** `X tests skipped due to missing dependencies`

```bash
# Install all optional dependencies
pip install -e ".[dev,redis,observability]"

# Install post-quantum crypto (optional)
pip install pqcrypto || echo "PQ crypto not available on this platform"

# Run tests with verbose output
pytest -v tests/
```

#### Async Test Issues

**Error:** `RuntimeError: no running event loop`

```bash
# Install pytest-asyncio
pip install pytest-asyncio

# Run with async mode
pytest --asyncio-mode=auto tests/
```

### Getting Help

If you're still having issues:

1. **Check the logs:**
   ```bash
   # Application logs
   cat logs/agentos.log

   # Docker logs
   docker compose logs -f
   ```

2. **Verify your environment:**
   ```bash
   # Run the hardware check script
   python scripts/check_requirements.py
   ```

3. **Search existing issues:**
   https://github.com/kase1111-hash/Agent-OS/issues

4. **Open a new issue** with:
   - Operating system and version
   - Python version (`python --version`)
   - Error message (full traceback)
   - Steps to reproduce

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
