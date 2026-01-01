# Agent OS Alpha Release Notes

**Version:** 0.1.0-alpha
**Release Date:** January 2026
**Status:** Pre-release / Alpha

---

## Overview

This is the first alpha release of Agent OS, a language-native, constitutionally-governed AI operating system designed for local, family-controlled AI infrastructure.

**This is pre-release software.** It is intended for early adopters, developers, and researchers who want to explore the concepts and contribute to development.

---

## What's Included

### Core Components

| Component | Status | Description |
|-----------|--------|-------------|
| **Whisper Orchestrator** | Functional | Intent classification and request routing |
| **Agent Smith** | Functional | Constitutional validation and security enforcement |
| **Seshat Archivist** | Functional | Memory and RAG retrieval system |
| **Quill Refiner** | Functional | Document formatting and refinement |
| **Muse Creative** | Functional | Creative content generation |
| **Sage Reasoner** | Functional | Long-context reasoning and analysis |
| **Constitutional Parser** | Functional | YAML/Markdown constitution parsing |
| **Boundary System** | Functional | Security boundary enforcement |

### Security Features

| Feature | Status | Notes |
|---------|--------|-------|
| Constitutional Validation | Functional | All requests validated against constitution |
| Audit Logging | Functional | Comprehensive audit trail |
| Rate Limiting | Functional | Redis-backed rate limiting |
| Encryption at Rest | Functional | AES-256-GCM encryption |
| Attack Detection | Functional | Pattern-based detection with SIEM integration |
| Post-Quantum Crypto | Experimental | Dilithium/Kyber support (requires liboqs) |

### Integrations

| Integration | Status | Notes |
|-------------|--------|-------|
| Ollama | Functional | Local LLM backend |
| Voice (STT/TTS) | Functional | Whisper.cpp and Piper |
| Prometheus/Grafana | Functional | Metrics and monitoring |
| Boundary Daemon | Optional | External trust policy layer |
| Boundary SIEM | Optional | Enterprise SIEM integration |

---

## Known Limitations

### Functional Limitations

1. **Single-User Focus**: Multi-user/family features are not yet implemented
2. **No Mobile Apps**: Web interface only; iOS/Android apps planned for Phase 2
3. **Limited Tool Calling**: Agent tool execution is basic; sandboxing improvements planned
4. **No Federation**: Instance-to-instance communication not yet implemented

### Technical Limitations

1. **Hardware Requirements**: Minimum 16GB RAM, 50GB disk space recommended
2. **GPU Recommended**: RTX 3070 or better for optimal performance
3. **Linux Preferred**: Best tested on Ubuntu 22.04/24.04
4. **macOS Support**: Functional but less tested
5. **Windows Support**: WSL2 recommended; native support is experimental

### Security Limitations

1. **No HSM Support**: Hardware security module integration not implemented
2. **Certificate Upgrade**: Post-quantum certificate upgrade not implemented
3. **No Formal Audit**: Third-party security audit planned for Phase 2

---

## Hardware Requirements

### Minimum Requirements

| Component | Requirement |
|-----------|-------------|
| CPU | 8 cores (AMD Ryzen 5 / Intel i5 or better) |
| RAM | 16 GB |
| Storage | 50 GB SSD |
| GPU | Optional (CPU inference supported) |
| Network | Broadband for initial setup |

### Recommended Requirements

| Component | Requirement |
|-----------|-------------|
| CPU | 12+ cores (AMD Ryzen 7 / Intel i7 or better) |
| RAM | 32 GB |
| Storage | 100 GB NVMe SSD |
| GPU | NVIDIA RTX 4070 Ti (16GB VRAM) or better |
| Network | Gigabit Ethernet |

---

## Supported Platforms

| Platform | Status | Notes |
|----------|--------|-------|
| Ubuntu 22.04 LTS | Fully Tested | Primary development platform |
| Ubuntu 24.04 LTS | Tested | Recommended |
| Fedora 39/40 | Tested | Works well |
| Debian 12 | Tested | Stable |
| macOS 14+ (Sonoma) | Partially Tested | Apple Silicon supported |
| Windows 11 (WSL2) | Partially Tested | Use Ubuntu WSL |
| Windows 11 (Native) | Experimental | May have issues |

---

## Breaking Changes Expected Before 1.0

The following areas may have breaking changes in future releases:

1. **Configuration Format**: YAML configuration structure may change
2. **API Endpoints**: REST API routes and response formats
3. **Database Schema**: SQLite/ChromaDB schemas may be migrated
4. **Agent Prompts**: Agent prompt templates and constitution format
5. **Plugin API**: Third-party agent integration interface

We recommend:
- Not deploying to production environments
- Backing up data regularly
- Following the changelog for migration guides

---

## Installation

### Quick Start (Linux/macOS)

```bash
# Clone the repository
git clone https://github.com/kase1111-hash/Agent-OS.git
cd Agent-OS

# Run the build script
./build.sh

# Start Agent OS
./start.sh
```

### Quick Start (Windows)

1. Double-click `build.bat`
2. Double-click `start.bat`
3. Open http://localhost:8080

See [START_HERE.md](START_HERE.md) (Windows) or [START_HERE_LINUX.md](START_HERE_LINUX.md) (Linux/macOS) for detailed instructions.

---

## Configuration

### Required Environment Variables

```bash
# Security (MUST be set before deployment)
GRAFANA_ADMIN_PASSWORD=<strong-password>

# Authentication (enabled by default)
AGENT_OS_REQUIRE_AUTH=true
AGENT_OS_API_KEY=<your-api-key>
```

### Optional Environment Variables

```bash
# Server
AGENT_OS_WEB_PORT=8080
AGENT_OS_WEB_HOST=0.0.0.0

# LLM Backend
OLLAMA_HOST=http://localhost:11434

# Voice
AGENT_OS_STT_ENABLED=true
AGENT_OS_TTS_ENABLED=true
```

---

## Reporting Issues

### Bug Reports

Please report bugs at: https://github.com/kase1111-hash/Agent-OS/issues

Include:
- Operating system and version
- Python version
- Error messages and logs
- Steps to reproduce

### Security Issues

**Do not report security vulnerabilities publicly.**

See [SECURITY.md](docs/governance/security.md) for responsible disclosure instructions.

---

## Getting Help

- **Documentation**: [docs/README.md](docs/README.md)
- **FAQ**: [docs/FAQ.md](docs/FAQ.md)
- **Discussions**: GitHub Discussions (coming soon)
- **Discord**: Coming soon

---

## Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

Areas where we especially need help:
- Testing on different platforms
- Documentation improvements
- Bug reports and fixes
- Performance optimization
- Accessibility improvements

---

## Roadmap

| Phase | Timeline | Focus |
|-------|----------|-------|
| Phase 1 (Current) | Q1-Q2 2026 | Proof of Concept |
| Phase 2 | Q3-Q4 2026 | Usability & Refinement |
| Phase 3 | 2027 | Ecosystem Growth |
| Phase 4 | 2028+ | Maturity & Scale |

See [ROADMAP.md](ROADMAP.md) for detailed milestones.

---

## License

Agent OS is released under **CC0 1.0 Universal (Public Domain)**.

You are free to use, modify, and distribute this software without restriction.

---

## Acknowledgments

Thank you to all early adopters and contributors who are helping shape Agent OS. Together, we're building a future where families own and govern their own AI infrastructure.

---

*Last Updated: January 2026*
*Maintained By: Agent OS Community*
