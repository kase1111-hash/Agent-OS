# Agent-OS

**Natural Language Operating System (NLOS) for AI Agents**

A constitutional AI OS that uses language-native runtime and prose-based operating system principles. Agent-OS is an agent orchestration platform enabling multi-agent coordination through natural language as runtime—the operating system for AI agents that puts human sovereignty first.

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

Core Innovation: Natural Language Operating System (NLOS) Kernel
The Paradigm Shift: Language-First Computing
Traditional operating systems use compiled code (C, Assembly) as their kernel. This requires specialized knowledge to modify and makes governance inaccessible to ordinary users.
Agent OS uses natural language itself as the kernel—an LLM-native OS architecture that enables language-first computing:

Human-readable governance documents (constitutions)
Modification without programming expertise
Direct interpretation by large language models
Auditable, inspectable, and understandable system behavior

Key Architectural Principles: Constitutional Governance OS

Constitutional Supremacy: A natural language constitution governs all agent behavior—constitutional AI design at its core
Role-Based Agents: Specialized AI agents with explicit authority boundaries for multi-agent operating system coordination
Human Stewardship: Ultimate authority always resides with humans—digital sovereignty for families
Local-First Operation: Computation and data remain under user control—self-hosted AI and owned AI infrastructure
Memory Consent: No persistent storage without explicit permission—data ownership guaranteed
Orchestrated Flow: Central routing prevents unauthorized agent communication—agent orchestration framework
Security as Doctrine: Refusal is a valid and required system response
Amendable Governance: Systems evolve through documented, auditable processes—intent preservation

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
├── tests/                  # Test suite (39 modules)
├── benchmarks/             # Performance benchmarks
└── deploy/                 # Deployment configurations
```

---

## What Problems Does Agent-OS Solve?

Agent-OS isn't just another AI chat interface. It's a complete agent coordination OS that addresses fundamental challenges in how we interact with and govern AI systems. If you've ever asked "how to coordinate multiple AI agents" or "what is a natural language OS"—this is the answer.

### 🛡️ The Control Problem
**Problem:** Most AI systems are black boxes where you hope the AI behaves correctly.
**Solution:** Constitutional governance with natural language rules that humans can read, modify, and audit. Every AI decision is logged and inspectable.

### 🔒 The Privacy Problem
**Problem:** Cloud AI services collect your data, train on your conversations, and share with third parties.
**Solution:** 100% local operation. No telemetry, no cloud dependency. Your data never leaves your hardware. AES-256-GCM encryption for anything stored.

### 🎯 The Authority Problem
**Problem:** AI assistants operate with unclear boundaries—what can they access? What can they do?
**Solution:** Six specialized agents with explicit, non-overlapping roles. Smith (Guardian) validates every action against constitutional rules before execution.

### 💾 The Memory Consent Problem
**Problem:** AI systems remember everything without asking, creating privacy and liability risks.
**Solution:** Consent-based memory with learning contracts. Nothing is stored without explicit permission. You can inspect and delete any data anytime.

### 🔗 The Integration Problem
**Problem:** AI assistants can't coordinate complex tasks across multiple domains.
**Solution:** Multi-agent orchestration where Whisper routes requests to specialized agents (reasoning, writing, creativity, memory) who collaborate through a secure message bus.

### 🏠 The Ownership Problem
**Problem:** Your AI assistant belongs to a corporation, not you.
**Solution:** Local-first, family-owned infrastructure with true digital sovereignty. Run it on a family AI server, federate with family members, own your private AI systems outright. This is self-hosted AI that respects data ownership.

---

## Daily Use Cases

| Use Case | How It Works |
|----------|--------------|
| **Private assistant** | Voice-activated, runs locally—unlike Alexa/Siri, your conversations stay private |
| **Family knowledge base** | Shared memories across federated nodes with consent-based access |
| **Document drafting** | Quill agent writes with constitutional tone limits you define |
| **Learning companion** | Sage explains concepts while Seshat remembers your progress |
| **Creative projects** | Muse brainstorms ideas, local image generation creates visuals |
| **Security monitoring** | Smith detects threats and can auto-generate patches |
| **Audit trail** | Every AI decision logged—useful for compliance and understanding |

---

## Core Agents

| Agent | Role | Description |
|-------|------|-------------|
| **Whisper** | Orchestrator | Intent classification and request routing |
| **Smith** | Guardian | Security validation, constitutional enforcement, and attack detection |
| **Seshat** | Archivist | Memory management and RAG retrieval |
| **Sage** | Elder | Complex reasoning and synthesis |
| **Quill** | Refiner | Document formatting and writing assistance |
| **Muse** | Creative | Creative content and idea generation |

---

## Key Features

- **Constitutional Governance**: Human-readable rules govern all AI behavior—prose-first development for AI systems
- **Local-First**: All computation stays on your hardware—true owned AI infrastructure
- **Memory Consent**: No data persistence without explicit permission—cognitive work value respected
- **Multi-Agent Architecture**: Specialized agents with clear authority boundaries—human-AI collaboration at its best
- **Encrypted Memory Vault**: Secure, consent-based data storage
- **Attack Detection & Auto-Remediation**: Real-time threat detection with LLM-powered analysis and automatic patch generation
- **SIEM Integration**: Connect to Splunk, Elasticsearch, Microsoft Sentinel, and Syslog
- **Voice Interface**: Speech-to-text and text-to-speech integration
- **Web Interface**: Modern FastAPI-based UI with WebSocket support
- **Mobile Backend**: API support for iOS/Android applications
- **Federation**: Multi-node deployment with post-quantum cryptography
- **Agent SDK**: Framework for building custom agents
- **Multi-Channel Notifications**: Alerts via Slack, Email, PagerDuty, Teams, and webhooks

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
| Security API | http://localhost:8080/api/security | Attack detection & recommendations |
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

---

## Connected Repositories

Agent-OS is part of a broader ecosystem of projects exploring the authenticity economy, human cognitive labor, and intent preservation.

### Agent-OS Ecosystem

Core modules that extend the natural language operating system:

| Repository | Description |
|------------|-------------|
| [synth-mind](https://github.com/kase1111-hash/synth-mind) | NLOS-based agent with six interconnected psychological modules for emergent continuity, empathy, and AI personality persistence |
| [boundary-daemon-](https://github.com/kase1111-hash/boundary-daemon-) | Mandatory trust enforcement layer defining cognition boundaries—AI trust enforcement and cognitive firewall |
| [memory-vault](https://github.com/kase1111-hash/memory-vault) | Secure, offline-capable, owner-sovereign storage for cognitive artifacts—private AI knowledge base |
| [value-ledger](https://github.com/kase1111-hash/value-ledger) | Economic accounting layer for cognitive work tracking ideas, effort, and novelty |
| [learning-contracts](https://github.com/kase1111-hash/learning-contracts) | Safety protocols for AI learning and data management—controlled AI learning boundaries |
| [Boundary-SIEM](https://github.com/kase1111-hash/Boundary-SIEM) | Security Information and Event Management for AI systems—agent security monitoring |

### NatLangChain Ecosystem

Related projects exploring natural language blockchain and intent-native protocols:

| Repository | Description |
|------------|-------------|
| [NatLangChain](https://github.com/kase1111-hash/NatLangChain) | Prose-first, intent-native blockchain protocol for recording human intent in natural language—semantic blockchain |
| [IntentLog](https://github.com/kase1111-hash/IntentLog) | Git for human reasoning—tracks "why" changes happen via prose commits and semantic version control |
| [RRA-Module](https://github.com/kase1111-hash/RRA-Module) | Revenant Repo Agent: Converts abandoned GitHub repositories into autonomous AI agents for licensing |
| [mediator-node](https://github.com/kase1111-hash/mediator-node) | LLM mediation layer for matching, negotiation, and closure proposals—AI negotiation node |
| [ILR-module](https://github.com/kase1111-hash/ILR-module) | IP & Licensing Reconciliation: Dispute resolution for intellectual property conflicts |
| [Finite-Intent-Executor](https://github.com/kase1111-hash/Finite-Intent-Executor) | Posthumous execution of predefined intent via Solidity smart contract—digital will executor |

### Game Development

Creative projects:

| Repository | Description |
|------------|-------------|
| [Shredsquatch](https://github.com/kase1111-hash/Shredsquatch) | 3D first-person snowboarding infinite runner (SkiFree homage) |
| [Midnight-pulse](https://github.com/kase1111-hash/Midnight-pulse) | Procedurally generated night drive—synthwave driving game |
| [Long-Home](https://github.com/kase1111-hash/Long-Home) | Atmospheric narrative indie game built with Godot |
