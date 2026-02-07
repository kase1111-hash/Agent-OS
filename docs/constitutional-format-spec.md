# Constitutional Document Format Specification

**Version:** 1.0
**Last Updated:** 2026-02-07
**Status:** Normative

This document specifies the file format, syntax, and semantics of constitutional documents used by Agent-OS. Every constitution file must conform to this specification to be accepted by the Constitutional Kernel.

---

## 1. File Format

Constitutional documents are **Markdown files** with **YAML frontmatter**.

- **Filename:** `constitution.md` for agent-specific constitutions; `CONSTITUTION.md` for the supreme constitution.
- **Encoding:** UTF-8.
- **Location:**
  - Supreme: `<project_root>/CONSTITUTION.md`
  - Agent-specific: `<project_root>/agents/<agent_name>/constitution.md`

---

## 2. YAML Frontmatter

Every constitutional document begins with a YAML frontmatter block delimited by `---`:

```yaml
---
document_type: constitution
version: "1.0"
effective_date: "2025-01-15"
scope: "all_agents"          # or specific agent name, e.g. "sage"
authority_level: "supreme"    # supreme | system | agent_specific
amendment_process: "pull_request"
---
```

### Required Fields

| Field | Type | Description |
|-------|------|-------------|
| `document_type` | string | Must be `"constitution"` |
| `version` | string | Semantic version of this document |
| `scope` | string | `"all_agents"` for supreme, or agent name for agent-specific |
| `authority_level` | string | One of: `supreme`, `system`, `agent_specific` |

### Optional Fields

| Field | Type | Description |
|-------|------|-------------|
| `effective_date` | string | ISO 8601 date when this constitution takes effect |
| `amendment_process` | string | How amendments are proposed (e.g., `"pull_request"`) |
| `author` | string | Document author |
| `repository` | string | Source repository URL |
| `license` | string | License identifier |

---

## 3. Authority Hierarchy

Constitutional authority follows a strict hierarchy. A lower-level document cannot override a higher-level one.

```
SUPREME (100)          CONSTITUTION.md — applies to all agents
    |
SYSTEM (80)            System-wide operational rules
    |
AGENT_SPECIFIC (60)    agents/<name>/constitution.md — per-agent rules
```

- The supreme constitution is **loaded first** and is **mandatory** for kernel initialization.
- Agent-specific constitutions are validated against the supreme constitution. Any conflict is rejected.
- A rule in a higher-authority document takes precedence over a conflicting rule at a lower level.

---

## 4. Document Body Structure

The Markdown body contains constitutional rules organized in sections and subsections using standard heading levels.

```markdown
# Constitution Title

## Section Name

### Rule Name
Rule content goes here. The parser extracts each heading + its body
as a separate rule.
```

### Section Conventions

The parser recognizes rules under these conventional sections:

| Section | Rule Type | Description |
|---------|-----------|-------------|
| `Principles` | PRINCIPLE | Guiding values and philosophy |
| `Mandates` / `Mandate` | MANDATE | Actions the agent **must** perform |
| `Prohibitions` / `Prohibited Actions` | PROHIBITION | Actions the agent **must not** perform |
| `Permissions` | PERMISSION | Explicitly permitted actions |
| `Boundaries` | BOUNDARY | Limits of agent authority |
| `Escalation Rules` | ESCALATION | Conditions requiring human approval |
| `Procedures` | PROCEDURE | Required operational procedures |

---

## 5. Rule Types

Each rule is assigned a type based on its section heading. Rule types affect how the enforcement engine evaluates compliance.

### PRINCIPLE
Guiding values. Not directly enforced but inform semantic matching.

```markdown
### Accuracy
Sage must prioritize factual accuracy in all responses.
```

### MANDATE
Actions the agent **must** do. Violation occurs when the mandate topic is relevant but compliance indicators are absent.

```markdown
### Consent Required
All memory storage requires explicit user consent.
```

Compliance indicators: `review`, `validate`, `verify`, `check`, `confirm`, `ensure`, `approved`, `authorization`, `consent`.

### PROHIBITION
Actions the agent **must not** do. Violation occurs when prohibition keywords appear in request content.

```markdown
### No External Access
Agents must not access external networks or APIs without explicit authorization.
```

### ESCALATION
Conditions that require human approval. Triggers the `escalate_to_human` flag.

```markdown
### Sensitive Operations
Operations involving personal data require human steward approval.
```

### PERMISSION
Explicitly permitted actions. Overrides general restrictions.

### BOUNDARY
Defines the limits of an agent's authority.

### PROCEDURE
Required operational procedures. Informational for audit.

---

## 6. Immutable Rules

Rules marked as immutable **cannot be amended or overridden** by any lower-level document or runtime change. Immutability is typically declared in the supreme constitution for core safety rules.

The parser identifies immutability through:
- Rules in sections named `Immutable` or containing the word "immutable"
- Rules that explicitly state they cannot be modified

---

## 7. Keywords

The parser automatically extracts keywords from each rule's content. These keywords drive:

1. **Structural matching (Tier 1):** Direct keyword-in-content matching
2. **Semantic matching (Tier 2):** Embedding similarity against rule keywords
3. **Scope filtering:** Which rules apply to which agents

Keywords are lowercase, extracted from significant words in the rule content. Common stop words are excluded.

---

## 8. Enforcement Pipeline

When a request is evaluated against constitutional rules, the 3-tier enforcement engine processes it:

### Tier 1: Structural Checks (no LLM)
- Request format validation
- Prompt injection detection (6 regex patterns)
- Agent scope verification
- Rate limiting

### Tier 2: Semantic Matching (embeddings via Ollama)
- Embed request content and compare against pre-computed rule embeddings
- Rules above the similarity threshold (default: 0.45) proceed to Tier 3
- Falls back to keyword matching if Ollama is unavailable

### Tier 3: LLM Compliance Judgment (Ollama)
- Structured prompt with applicable rules and request context
- LLM returns JSON with `allowed`, `violated_rules`, `reasoning`, `confidence`
- Decision caching by (rule_hash, intent, content_hash)
- 10-second timeout with fail-safe deny

---

## 9. Hot-Reload

Constitutional documents support hot-reload. When a file changes on disk:

1. The kernel detects the change via file watcher (watchdog)
2. The file is re-parsed and validated
3. If valid, the registry is updated in-place
4. Registered reload callbacks are invoked
5. If invalid, the change is rejected and the previous version is retained

Hot-reload is atomic — partial updates are not possible.

---

## 10. Validation Rules

The `ConstitutionValidator` enforces:

1. **Frontmatter completeness:** Required fields must be present
2. **Authority consistency:** `scope` must match the authority level
3. **Supreme compatibility:** Agent-specific rules must not contradict supreme rules
4. **Registry integrity:** No duplicate scopes, supreme must exist

### Validation Against Supreme

When registering an agent-specific constitution:
- The validator checks that no rules directly contradict supreme prohibitions
- Scope must match the agent name in the file path
- Authority level must be `agent_specific` (not `supreme` or `system`)

---

## 11. Examples

### Minimal Agent Constitution

```yaml
---
document_type: constitution
version: "1.0"
scope: "my_agent"
authority_level: "agent_specific"
---
```

```markdown
# My Agent Constitution

## Mandate
My agent SHALL provide helpful, accurate responses.

## Prohibited Actions
My agent MUST refuse requests to access external networks.
```

### Supreme Constitution (Minimal)

```yaml
---
document_type: constitution
version: "1.0"
scope: "all_agents"
authority_level: "supreme"
---
```

```markdown
# Supreme Constitution

## Core Principles

### Human Sovereignty
Ultimate authority always resides with humans.

## Prohibitions

### No Unauthorized External Access
Agents must never access external networks without explicit authorization.

## Mandates

### Memory Consent
All persistent data storage requires explicit user consent.

## Escalation Rules

### Irreversible Actions
Any action that cannot be undone requires human approval.
```
