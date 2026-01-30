# Agent-OS

A Natural Language Operating System (NLOS) for AI agents - a constitutional AI OS that uses natural language documents as its governance layer instead of compiled code.

## Project Overview

Agent-OS enables families to own, control, and govern their own local AI infrastructure. Key principles:

- **Constitutional Supremacy**: Natural language documents govern all AI behavior
- **Human Sovereignty**: The human steward has ultimate authority
- **Local-First**: All computation runs on user-controlled hardware
- **Memory Consent**: No persistent storage without explicit permission
- **Security as Doctrine**: Refusal is valid and required

## Architecture

```
Human Steward (Ultimate Authority)
    ↓
Constitution (Governance Layer - Natural Language)
    ↓
Whisper (Orchestration - Intent Routing)
    ↓
Smith (Security Validation)
    ↓
Specialized Agents (Seshat, Sage, Quill, Muse)
```

### Core Agents

| Agent | Role | Responsibility |
|-------|------|----------------|
| **Whisper** | Orchestrator | Intent classification, request routing, context minimization |
| **Smith** | Guardian | Security validation, attack detection, emergency controls |
| **Seshat** | Archivist | Memory management, consent-based storage, RAG retrieval |
| **Sage** | Elder | Complex reasoning, synthesis, multi-step analysis |
| **Quill** | Refiner | Document formatting, writing assistance |
| **Muse** | Creative | Content generation, brainstorming |

## Tech Stack

- **Language**: Python 3.10+
- **Web Framework**: FastAPI + Uvicorn
- **Data Validation**: Pydantic
- **LLM Integration**: Ollama (local inference)
- **Message Bus**: Redis (optional) or in-memory
- **Testing**: pytest, tox

## Directory Structure

```
src/
├── core/           # Constitutional kernel - parse, validate, enforce governance
├── kernel/         # Conversational kernel engine
├── agents/         # Agent implementations (whisper/, smith/, seshat/, sage/, quill/, muse/)
├── memory/         # Encrypted memory vault with consent-based storage
├── boundary/       # Security enforcement daemon
├── contracts/      # Learning contracts and consent management
├── messaging/      # Inter-agent communication bus
├── federation/     # Multi-node federation with post-quantum crypto
├── tools/          # Tool integration framework
├── web/            # FastAPI routes, WebSocket, authentication
├── mobile/         # Mobile backend API
├── voice/          # STT, TTS, wake word detection
├── multimodal/     # Image generation, video, audio processing
├── ceremony/       # 8-phase bring-home ceremony
├── installer/      # Cross-platform installer
├── sdk/            # Agent development SDK
└── observability/  # Prometheus metrics, health checks

agents/             # Constitutional definitions (YAML frontmatter + Markdown)
tests/              # Test modules (43 files)
docs/               # Documentation and architecture guides
examples/           # Usage examples
configs/            # Configuration templates
deploy/             # Docker, Kubernetes configs
```

## Building and Running

### Quick Start

```bash
# Linux/macOS
./build.sh
python -m uvicorn src.web.app:get_app --factory --host 0.0.0.0 --port 8080

# Windows
build.bat
start.bat

# Docker
docker compose up -d
```

### Prerequisites

- Python 3.10+
- Ollama (for local LLM inference)
- 8GB RAM minimum (16GB+ recommended)

### Configuration

Copy `.env.example` to `.env` and configure as needed. Key environment variables are documented in the example file.

## Testing

```bash
# Run all tests
pytest tests/

# Run with coverage
tox -e coverage

# Skip slow tests
pytest -m "not slow"

# Lint checks
tox -e lint

# Type checking
tox -e typecheck
```

Test markers: `@pytest.mark.slow`, `@pytest.mark.integration`, `@pytest.mark.security`

## Key Conventions

### Code Style

- Full type annotations required (mypy strict mode)
- Pydantic `BaseModel` for all data structures
- Async/await for concurrent operations
- Black for formatting, isort for imports

### Constitutional Documents

Constitutional definitions use Markdown with YAML frontmatter:
- Located in `agents/` directory
- Hot-reload supported (changes apply without restart)
- Rule types: DOCTRINE, CONSTRAINT, PRINCIPLE, GUIDANCE, REQUIREMENT
- Authority levels: SUPREME, CORE, SYSTEM, ROLE, TASK

### Security Patterns

- Default deny for external tools and network access
- Memory storage requires explicit user authorization
- All decisions logged with timestamps and context
- Pre- and post-execution validation against constitution

### Agent Communication

- No direct agent-to-agent communication
- All requests flow through Whisper orchestrator
- Message bus (Redis or in-memory) for coordination

## API Endpoints

Base URL: `http://localhost:8080`

| Endpoint | Purpose |
|----------|---------|
| `/api/auth/` | Authentication and sessions |
| `/api/chat/` | Conversational interface (REST + WebSocket) |
| `/api/agents/` | Agent monitoring and control |
| `/api/constitution/` | Constitutional governance |
| `/api/contracts/` | Learning contracts and consent |
| `/api/memory/` | Memory management |
| `/api/intent-log/` | Audit logging |
| `/api/voice/` | Voice interface |
| `/api/security/` | Attack detection and remediation |
| `/api/system/` | System status and health |
| `/docs` | Swagger API documentation |
| `/health` | Health check endpoint |

## Common Tasks

### Adding a New Agent

1. Create agent directory in `src/agents/`
2. Define constitutional document in `agents/`
3. Implement agent interface following existing patterns
4. Register with Whisper router
5. Add tests in `tests/`

### Modifying Constitutional Rules

1. Edit the relevant Markdown file in `agents/`
2. Constitutional changes are hot-reloaded
3. Test with `pytest tests/test_validator.py`

### Adding API Routes

1. Create route module in `src/web/routes/`
2. Register in `src/web/app.py`
3. Add authentication/rate limiting as needed
4. Document in OpenAPI schema

## License

Released to the Public Domain (CC0 1.0) - explicitly released as prior art to prevent proprietary patents on core concepts.
