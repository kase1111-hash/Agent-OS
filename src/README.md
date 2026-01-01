# Source Code

This directory contains the complete implementation of Agent-OS, a constitutional operating system for local AI.

## Implementation Status: ~90% Complete

**Codebase Statistics:**
- 155 Python source files
- ~50,000 lines of code
- 31 test modules

## Directory Structure

```
src/
├── core/                  # Core system components
│   ├── constitution.py    # Constitution parser and validator (~464 lines)
│   ├── parser.py          # Constitutional document parser
│   ├── validator.py       # Validation logic
│   └── models.py          # Core data models
│
├── agents/                # Agent implementations
│   ├── interface.py       # BaseAgent class with 5 mandatory methods (~592 lines)
│   ├── config.py          # Agent configuration management
│   ├── loader.py          # Agent configuration loader
│   ├── ollama.py          # Ollama LLM integration
│   ├── enforcement.py     # Constitutional enforcement
│   ├── isolation.py       # Agent sandboxing
│   ├── whisper/           # Orchestrator agent
│   │   ├── agent.py       # WhisperAgent implementation
│   │   ├── router.py      # Intent routing engine
│   │   ├── aggregator.py  # Response aggregation
│   │   └── flow.py        # Flow control
│   ├── smith/             # Guardian agent
│   │   ├── agent.py       # SmithAgent (S1-S12 checks)
│   │   ├── pre_validator.py
│   │   ├── post_monitor.py
│   │   ├── refusal_engine.py
│   │   ├── emergency.py   # Emergency controls
│   │   └── attack_detection/  # Attack detection & auto-remediation
│   │       ├── detector.py       # Attack event detection
│   │       ├── siem_connector.py # SIEM integration (Splunk, Elastic, Sentinel)
│   │       ├── patterns.py       # Attack pattern library
│   │       ├── analyzer.py       # Vulnerability analysis
│   │       ├── llm_analyzer.py   # LLM-powered attack analysis
│   │       ├── remediation.py    # Patch generation engine
│   │       ├── recommendation.py # Fix recommendation system
│   │       ├── storage.py        # SQLite/memory storage backends
│   │       ├── git_integration.py # PR automation for fixes
│   │       ├── notifications.py  # Multi-channel alerting
│   │       ├── config.py         # YAML configuration system
│   │       └── integration.py    # Pipeline orchestration
│   ├── seshat/            # Memory agent
│   │   ├── agent.py       # SeshatAgent (~710 lines)
│   │   ├── embeddings.py  # Embedding engine
│   │   ├── vectorstore.py # Vector storage
│   │   ├── retrieval.py   # RAG pipeline
│   │   └── consent_integration.py
│   ├── sage/              # Reasoning agent
│   │   ├── agent.py       # SageAgent
│   │   └── reasoning.py   # Chain-of-thought
│   ├── quill/             # Writer agent
│   │   ├── agent.py       # QuillAgent
│   │   └── formatting.py  # Document formatting
│   └── muse/              # Creative agent
│       ├── agent.py       # MuseAgent
│       └── creative.py    # Creative generation
│
├── kernel/                # Conversational kernel
│   ├── engine.py          # Core kernel engine (~700 lines)
│   ├── interpreter.py     # Natural language interpreter
│   ├── policy.py          # Policy management
│   ├── rules.py           # Rule registry
│   ├── monitor.py         # System monitoring
│   ├── ebpf.py            # eBPF filter integration
│   └── fuse.py            # FUSE wrapper
│
├── memory/                # Memory management system
│   ├── vault.py           # Memory Vault API (~684 lines)
│   ├── storage.py         # Encrypted blob storage
│   ├── keys.py            # Key management
│   ├── profiles.py        # Encryption profiles
│   ├── index.py           # Vault index
│   ├── consent.py         # Consent management
│   ├── deletion.py        # Secure deletion
│   └── genesis.py         # Genesis proofs
│
├── messaging/             # Inter-agent communication
│   ├── bus.py             # InMemoryMessageBus (~609 lines)
│   ├── redis_bus.py       # Redis backend
│   └── models.py          # FlowRequest/FlowResponse
│
├── boundary/              # Security enforcement
│   └── daemon/
│       ├── boundary_daemon.py  # Main daemon (~437 lines)
│       ├── policy_engine.py    # Policy evaluation
│       ├── enforcement.py      # Action enforcement
│       ├── tripwires.py        # Security triggers
│       ├── state_monitor.py    # System state monitoring
│       └── event_log.py        # Immutable logging
│
├── contracts/             # Learning contracts
│   ├── consent.py         # Consent engine (~445 lines)
│   ├── store.py           # Contract storage
│   ├── validator.py       # Contract validation
│   └── domains.py         # Domain restrictions
│
├── ceremony/              # Bring-home ceremony
│   ├── orchestrator.py    # 8-phase ceremony
│   ├── phases.py          # Phase implementations
│   └── state.py           # Ceremony state
│
├── tools/                 # Tool integration
│   ├── registry.py        # Tool registry
│   ├── executor.py        # Tool execution
│   ├── permissions.py     # Permission system
│   └── validation.py      # Tool validation
│
├── web/                   # Web interface
│   ├── app.py             # FastAPI application (~200 lines)
│   ├── config.py          # Web configuration
│   └── routes/            # API endpoints
│       ├── agents.py      # /api/agents
│       ├── chat.py        # /api/chat
│       ├── constitution.py # /api/constitution
│       ├── memory.py      # /api/memory
│       └── system.py      # /api/system
│
├── mobile/                # Mobile backend
│   ├── api.py             # Mobile API (~641 lines)
│   ├── client.py          # HTTP client
│   ├── auth.py            # Authentication
│   ├── notifications.py   # Push notifications
│   └── vpn.py             # VPN support
│
├── voice/                 # Voice interface
│   ├── assistant.py       # Voice assistant
│   ├── stt.py             # Speech-to-text
│   ├── tts.py             # Text-to-speech
│   └── wakeword.py        # Wake word detection
│
├── multimodal/            # Multimodal support
│   ├── agent.py           # Multimodal agent
│   ├── vision.py          # Vision processing
│   ├── audio.py           # Audio processing
│   └── video.py           # Video processing
│
├── federation/            # Multi-node federation
│   ├── node.py            # Federation node
│   ├── protocol.py        # Communication protocol
│   ├── crypto.py          # Cryptographic operations
│   ├── identity.py        # Identity management
│   └── permissions.py     # Permission negotiation
│
├── ledger/                # Value ledger
│   ├── client.py          # Ledger client
│   ├── hooks.py           # Intent hooks
│   └── models.py          # Ledger models
│
├── installer/             # Cross-platform installer
│   ├── base.py            # Base installer
│   ├── cli.py             # CLI interface
│   ├── docker.py          # Docker installation
│   ├── linux.py           # Linux installer
│   ├── macos.py           # macOS installer
│   └── windows.py         # Windows installer
│
└── sdk/                   # Agent SDK
    ├── builder.py         # Agent builder
    ├── decorators.py      # Common decorators
    ├── lifecycle.py       # Lifecycle management
    ├── templates/         # Agent templates
    │   ├── base.py
    │   ├── generation.py
    │   ├── reasoning.py
    │   └── tool_use.py
    └── testing/           # Testing framework
        ├── assertions.py
        ├── fixtures.py
        ├── mocks.py
        └── runner.py
```

## Key Components

### Core Infrastructure (Phase 0) - ✅ Complete

| Component | File | Status |
|-----------|------|--------|
| Constitutional Kernel | `core/constitution.py` | ✅ |
| Message Bus | `messaging/bus.py` | ✅ |
| Agent Interface | `agents/interface.py` | ✅ |

### Core Agents (Phase 1-2) - ✅ Complete

| Agent | Role | Status |
|-------|------|--------|
| Whisper | Orchestrator | ✅ |
| Smith | Guardian (S1-S12) + Attack Detection | ✅ |
| Seshat | Memory/RAG | ✅ |
| Sage | Reasoning | ✅ |
| Quill | Writing | ✅ |
| Muse | Creative | ✅ |

### Security & Trust (Phase 3) - ✅ Complete

| Component | Status |
|-----------|--------|
| Boundary Daemon | ✅ |
| Learning Contracts | ✅ |
| Bring-Home Ceremony | ✅ |

### Advanced Features - ✅ Complete

| Component | Status |
|-----------|--------|
| Web Interface | ✅ |
| Mobile Backend | ✅ |
| Voice Interface | ✅ |
| Multimodal | ✅ |
| Federation | ✅ |
| Tool Integration | ✅ |
| Agent SDK | ✅ |
| Cross-Platform Installer | ✅ |
| Attack Detection & Remediation | ✅ |

## Running Tests

```bash
# Run all tests
pytest tests/

# Run specific test module
pytest tests/test_kernel.py

# Run with coverage
pytest --cov=src tests/

# Run end-to-end simulation
python -m tests.e2e_simulation
```

## Development

See `CONTRIBUTING.md` in the root directory for contribution guidelines.

See `docs/technical/Specification.md` for the complete specification with implementation status.

## License

CC0 1.0 Universal (Public Domain)
