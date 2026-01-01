# Agent-OS Examples

This directory contains practical examples and use cases for Agent-OS.

## Getting Started

Before running these examples, ensure you have Agent-OS installed and running. See [RUNNING_AND_COMPILING.md](../docs/RUNNING_AND_COMPILING.md) for installation instructions.

---

## Available Examples

### Basic Setup

- **[basic-homestead-setup.md](./basic-homestead-setup.md)** - Guide for setting up a basic Agent-OS homestead
  - Minimal local-first deployment
  - Step-by-step configuration
  - Verification steps

### Constitutional Examples

- **[constitution-examples.md](./constitution-examples.md)** - Example constitutions for different use cases
  - Family homestead constitution
  - Developer workspace constitution
  - Educational environment constitution

### Multi-Agent Workflows

- **[multi-agent-task.md](./multi-agent-task.md)** - Examples of coordinating multiple agents for complex tasks
  - Research and writing workflow
  - Creative collaboration
  - Memory-aware reasoning

---

## Quick Start Examples

### Start the Web Interface

```bash
# Install dependencies
pip install -r requirements.txt

# Start the application
python -m uvicorn src.web.app:get_app --factory --host 0.0.0.0 --port 8080
```

Visit http://localhost:8080 to access the web interface.

### Using Docker

```bash
# Start all services
docker compose up -d

# Access endpoints
# - Web UI: http://localhost:8080
# - API Docs: http://localhost:8080/docs
# - Prometheus: http://localhost:9090
# - Grafana: http://localhost:3000
```

### Run Tests

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest --cov=src tests/

# Run specific test file
pytest tests/test_kernel.py
```

### Run Benchmarks

```bash
pytest benchmarks/
```

---

## Example Structure

Each example in this directory includes:

| Section | Description |
|---------|-------------|
| **Context** | When and why to use this example |
| **Prerequisites** | What you need before starting |
| **Implementation** | Step-by-step instructions |
| **Verification** | How to confirm success |
| **Troubleshooting** | Common issues and solutions |

---

## Core Agents

Agent-OS uses six core agents that you'll interact with in these examples:

| Agent | Role | Use Case |
|-------|------|----------|
| **Whisper** | Orchestrator | Routes requests to appropriate agents |
| **Smith** | Guardian | Validates security and constitutional compliance |
| **Seshat** | Archivist | Manages memory and retrieval |
| **Sage** | Elder | Handles complex reasoning tasks |
| **Quill** | Refiner | Formats and refines documents |
| **Muse** | Creative | Generates creative content |

---

## API Endpoints

Common endpoints used in examples:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | System health check |
| `/api/chat` | POST | Send a chat message |
| `/api/agents` | GET | List available agents |
| `/api/constitution` | GET | View current constitution |
| `/api/memory` | GET | Query memory vault |
| `/docs` | GET | Swagger API documentation |

---

## Contributing Examples

We welcome community contributions! To add a new example:

1. Create a descriptive markdown file in this directory
2. Follow the structure outlined above
3. Include clear setup instructions
4. Provide expected results
5. Test your example thoroughly
6. Submit via pull request

See [CONTRIBUTING.md](../CONTRIBUTING.md) for detailed guidelines.

---

## Related Documentation

- [docs/README.md](../docs/README.md) - Complete documentation index
- [docs/FAQ.md](../docs/FAQ.md) - Frequently asked questions
- [docs/technical/architecture.md](../docs/technical/architecture.md) - System architecture
- [CONSTITUTION.md](../CONSTITUTION.md) - Constitutional governance

---

*Last Updated: January 2026*
*License: CC0 1.0 Universal (Public Domain)*
