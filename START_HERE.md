# Agent-OS: Getting Started

A developer-focused guide to running Agent-OS and understanding its constitutional governance framework.

---

## Prerequisites

1. **Python 3.10+**
2. **Ollama** for local LLM inference -- [ollama.com](https://ollama.com/download)

---

## 1. Install

```bash
git clone https://github.com/kase1111-hash/Agent-OS.git
cd Agent-OS
pip install -e .
```

No GPU libraries are required. Agent-OS uses Ollama for all LLM inference.

---

## 2. Set Up Ollama

Pull a model for the agents to use:

```bash
ollama pull mistral        # General agent model
ollama pull llama3.2:3b    # Used by constitutional enforcement (lightweight)
ollama pull nomic-embed-text  # Used for semantic rule matching (embeddings)
```

Ollama must be running before starting Agent-OS:

```bash
ollama serve   # If not already running as a service
```

---

## 3. Run Agent-OS

```bash
python -m uvicorn src.web.app:get_app --factory --host 0.0.0.0 --port 8080
```

Then open http://localhost:8080 in your browser.

---

## 4. Write Your First Constitution

Constitutions are Markdown files with YAML frontmatter. They define the rules that govern agent behavior.

### The Supreme Constitution

The file `CONSTITUTION.md` at the project root is the supreme law. All agents obey it. It defines:

- Core principles (human sovereignty, transparency, consent)
- Universal prohibitions (no unauthorized external access, no data exfiltration)
- Universal mandates (memory consent, audit logging)
- Escalation rules (irreversible actions need human approval)

### Agent-Specific Constitutions

Each agent has its own constitution in `agents/<name>/constitution.md`:

```yaml
---
document_type: constitution
version: "1.0"
scope: "sage"
authority_level: "agent_specific"
---

# Sage Agent Constitution

## Mandate
Sage SHALL provide accurate, well-reasoned responses.

## Prohibited Actions
Sage MUST refuse to make ethical judgments on behalf of the user.
```

See [docs/constitutional-format-spec.md](docs/constitutional-format-spec.md) for the full format specification.

---

## 5. Understand the Request Lifecycle

Every user request follows this path:

```
User Request
    |
    v
[1] Whisper: Classify intent (factual, creative, memory, system)
    |
    v
[2] Smith: Pre-validate against constitutional rules
    |-- DENIED --> Return refusal with reason
    |-- ESCALATE --> Request human approval
    v
[3] Route to target agent (Sage, Quill, Seshat, Muse)
    |
    v
[4] Agent processes request and generates response
    |
    v
[5] Smith: Post-validate response (data leakage, anomalies)
    |-- DENIED --> Redact or refuse
    v
[6] Return response to user with audit trail
```

### Constitutional Enforcement (3-tier)

Smith validates requests using a 3-tier engine:

- **Tier 1 -- Structural checks:** Format validation, prompt injection detection, scope verification. No LLM needed.
- **Tier 2 -- Semantic matching:** Embedding similarity between request and constitutional rules via Ollama.
- **Tier 3 -- LLM judgment:** Full compliance evaluation by Ollama for ambiguous cases.

If Ollama is unavailable, enforcement falls back to keyword-based matching with conservative (deny) defaults.

---

## 6. Run the Tests

```bash
python -m pytest tests/ -v --ignore=tests/test_kernel.py \
  --ignore=tests/test_memory_vault.py \
  --ignore=tests/test_pq_keys.py \
  --ignore=tests/test_seshat.py
```

The ignored tests have dependencies on optional C libraries. All core governance tests run without them.

---

## 7. Project Structure

```
Agent-OS/
  CONSTITUTION.md           # Supreme constitutional law
  agents/                   # Agent constitutional definitions
    sage/constitution.md
    guardian/constitution.md
    muse/constitution.md
    ...
  src/
    core/                   # Constitutional kernel, parser, validator, enforcement
    agents/
      whisper/              # Orchestrator (intent classification, routing)
      smith/                # Guardian (security validation, emergency controls)
      sage/                 # Reasoning agent
      quill/                # Writing agent
      muse/                 # Creative agent
      seshat/               # Memory/archival agent
    messaging/              # Inter-agent message bus
    boundary/               # Security enforcement daemon
    web/                    # FastAPI web interface
  tests/                    # Test suite (30+ modules, 1000+ tests)
  docs/                     # Documentation
```

---

## Key Concepts

| Concept | Description |
|---------|-------------|
| **Supreme Constitution** | Top-level rules that no agent can override |
| **Agent Constitution** | Per-agent rules subordinate to the supreme |
| **Enforcement Engine** | 3-tier pipeline: structural, semantic, LLM judge |
| **Smith (Guardian)** | Security agent that validates every request and response |
| **Whisper (Orchestrator)** | Routes requests to the right agent |
| **Hot-Reload** | Constitution changes take effect without restart |
| **Consent-Based Memory** | No data persists without explicit user permission |

---

## Further Reading

- [Constitutional Format Specification](docs/constitutional-format-spec.md)
- [CONSTITUTION.md](CONSTITUTION.md) -- the supreme constitutional law
- [agents/sage/constitution.md](agents/sage/constitution.md) -- example agent constitution

---

## Getting Help

- Report issues: https://github.com/kase1111-hash/Agent-OS/issues
