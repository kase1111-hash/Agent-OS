# Agent-OS
Agent OS: A Constitutional Operating System for Local AI

**Public Declaration of Prior Art & Creative Commons Release**

- **Date:** December 15, 2025
- **Version:** 1.0
- **Author:** Kase Branham
- **Repository:** https://github.com/kase1111-hash/Agent-OS
- **License:** Creative Commons Zero v1.0 Universal (CC0 1.0) - Public Domain Dedication

📚 **[Complete Documentation Index →](./docs/README.md)**

---

## Declaration of Intent
I hereby declare the following architectural design, constitutional framework, and operational principles for what I call "Agent OS" to be prior art, released irrevocably into the public domain for the benefit of all humanity.
This document establishes:

The concept of constitutional governance for artificial intelligence agents
The architecture for local, family-owned AI operating systems
The principles of human sovereignty over machine intelligence
A complete specification that anyone may implement, modify, or distribute

No person or entity may claim exclusive ownership, patent rights, or proprietary control over these core concepts.
This is my contribution to the digital commons, inspired by the spirit of Linux, the principles of open source, and the belief that families should own and govern their own intelligence infrastructure.

Core Innovation: Natural Language Operating System Kernel
The Paradigm Shift
Traditional operating systems use compiled code (C, Assembly) as their kernel. This requires specialized knowledge to modify and makes governance inaccessible to ordinary users.
Agent OS uses natural language itself as the kernel, enabling:

Human-readable governance documents (constitutions)
Modification without programming expertise
Direct interpretation by large language models
Auditable, inspectable, and understandable system behavior

Key Architectural Principles

Constitutional Supremacy: A natural language constitution governs all agent behavior
Role-Based Agents: Specialized AI agents with explicit authority boundaries
Human Stewardship: Ultimate authority always resides with humans
Local-First Operation: Computation and data remain under user control
Memory Consent: No persistent storage without explicit permission
Orchestrated Flow: Central routing prevents unauthorized agent communication
Security as Doctrine: Refusal is a valid and required system response
Amendable Governance: Systems evolve through documented, auditable processes

## Inspiration & Differentiation

This project draws inspiration from Andrej Karpathy's visionary concept of an "LLM Operating System" (LLM OS)—where large language models form the core of a new computing paradigm, with agents as applications and natural language as the interface.

While many LLM OS explorations focus on performance, tool integration, or cloud-scale orchestration, Agent-OS takes a deliberately contrarian path:

- **Constitutional supremacy**: Governance through auditable natural-language constitutions, not code or prompts alone.
- **Human sovereignty first**: Explicit boundaries, refusal as a virtue, and default denial of external tools/network access.
- **Local homestead model**: A private, air-gapped, resilient digital residence—prioritizing privacy and self-sufficiency over capability.

We dedicate this work to the public domain (CC0) in the spirit of open, principled evolution of the LLM OS idea.

---

## Quick Start

### Windows Users
1. Install [Python 3.10+](https://www.python.org/downloads/) and [Ollama](https://ollama.com/download)
2. Double-click `build.bat` to set up the environment
3. Double-click `start.bat` to run Agent-OS
4. Open http://localhost:8080 in your browser

See [START_HERE.md](./START_HERE.md) for detailed Windows instructions.

### Linux/macOS Users
```bash
# Clone and install
git clone https://github.com/kase1111-hash/Agent-OS.git
cd Agent-OS
pip install -r requirements.txt

# Pull an Ollama model
ollama pull mistral

# Run the application
python -m uvicorn src.web.app:get_app --factory --host 0.0.0.0 --port 8080
```

### Docker Deployment
```bash
cp .env.example .env
docker compose up -d
```

Visit http://localhost:8080 to access the web interface.

---

## Project Structure

```
Agent-OS/
├── src/                    # Source code (~64,000 lines Python)
│   ├── agents/             # 6 core agents + LLM integrations
│   ├── core/               # Constitutional kernel
│   ├── kernel/             # Conversational kernel engine
│   ├── memory/             # Encrypted memory vault
│   ├── messaging/          # Inter-agent communication bus
│   ├── boundary/           # Security enforcement daemon
│   ├── contracts/          # Learning contracts & consent
│   ├── ceremony/           # 8-phase bring-home ceremony
│   ├── web/                # FastAPI web interface
│   ├── mobile/             # Mobile backend API
│   ├── voice/              # Voice interface (STT/TTS)
│   ├── multimodal/         # Vision, audio, video support
│   ├── federation/         # Multi-node federation
│   ├── tools/              # Tool integration framework
│   ├── installer/          # Cross-platform installer
│   ├── sdk/                # Agent development SDK
│   └── observability/      # Monitoring & metrics
├── agents/                 # Agent constitutional definitions
├── docs/                   # Comprehensive documentation
├── examples/               # Practical usage examples
├── tests/                  # Test suite (31 modules)
├── benchmarks/             # Performance benchmarks
└── deploy/                 # Deployment configurations
```

---

## Core Agents

| Agent | Role | Description |
|-------|------|-------------|
| **Whisper** | Orchestrator | Intent classification and request routing |
| **Smith** | Guardian | Security validation and constitutional enforcement |
| **Seshat** | Archivist | Memory management and RAG retrieval |
| **Sage** | Elder | Complex reasoning and synthesis |
| **Quill** | Refiner | Document formatting and writing assistance |
| **Muse** | Creative | Creative content and idea generation |

---

## Key Features

- **Constitutional Governance**: Human-readable rules govern all AI behavior
- **Local-First**: All computation stays on your hardware
- **Memory Consent**: No data persistence without explicit permission
- **Multi-Agent Architecture**: Specialized agents with clear authority boundaries
- **Encrypted Memory Vault**: Secure, consent-based data storage
- **Voice Interface**: Speech-to-text and text-to-speech integration
- **Web Interface**: Modern FastAPI-based UI with WebSocket support
- **Mobile Backend**: API support for iOS/Android applications
- **Federation**: Multi-node deployment with post-quantum cryptography
- **Agent SDK**: Framework for building custom agents

---

## Requirements

### Minimum
- Python 3.10+
- 8GB RAM
- 500GB storage

### Recommended
- 16GB+ RAM
- GPU with 16GB+ VRAM (for local LLM inference)
- SSD storage

### Dependencies
Core dependencies include FastAPI, Pydantic, PyYAML, Redis, and PyTorch (for image generation).

See [requirements.txt](./requirements.txt) for the complete list.

---

## Documentation

| Document | Description |
|----------|-------------|
| [docs/README.md](./docs/README.md) | Complete documentation index |
| [CONSTITUTION.md](./CONSTITUTION.md) | Supreme governing law |
| [ROADMAP.md](./ROADMAP.md) | Development timeline (2025-2028) |
| [docs/FAQ.md](./docs/FAQ.md) | Frequently asked questions |
| [docs/technical/architecture.md](./docs/technical/architecture.md) | System architecture details |
| [CONTRIBUTING.md](./CONTRIBUTING.md) | How to contribute |

---

## Development

### Running Tests
```bash
pytest tests/
```

### Running Benchmarks
```bash
pytest benchmarks/
```

### Code Formatting
```bash
black src/ tests/
isort src/ tests/
```

### Type Checking
```bash
mypy src/
```

---

## Endpoints

| Endpoint | URL | Description |
|----------|-----|-------------|
| Main App | http://localhost:8080 | Web interface |
| API Docs | http://localhost:8080/docs | Swagger documentation |
| Health Check | http://localhost:8080/health | System status |
| Prometheus | http://localhost:9090 | Metrics (Docker) |
| Grafana | http://localhost:3000 | Dashboards (Docker) |

---

## Current Status

**Phase**: 0 Complete, Phase 1 starting Q1 2026
**Version**: 1.0 (December 2025)
**Implementation**: ~90% complete

See [ROADMAP.md](./ROADMAP.md) for detailed development timeline.

---

## Contributing

We welcome contributions! Please see:
- [CONTRIBUTING.md](./CONTRIBUTING.md) - Contribution guidelines
- [CODE_OF_CONDUCT.md](./CODE_OF_CONDUCT.md) - Community standards
- [contrib/Contributing.md](./contrib/Contributing.md) - Detailed process

---

## License

This project is released under **CC0 1.0 Universal (Public Domain Dedication)**.

You are free to use, modify, and distribute this work without restriction.

---

## Links

- **Repository**: https://github.com/kase1111-hash/Agent-OS
- **Issues**: https://github.com/kase1111-hash/Agent-OS/issues
- **Documentation**: [docs/README.md](./docs/README.md)
