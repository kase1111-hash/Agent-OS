# Agent-OS Security Audit — Agentic Security Checklist

## Post-Moltbook Hardening Guide v1.0 | Audit Date: 2026-02-19

**Auditor:** Claude Code (automated review)
**Scope:** Full Agent-OS repository (`/home/user/Agent-OS`)
**Baseline:** Post-Phase 1-7 remediation (Moltbook/OpenClaw fixes applied)

---

## Executive Summary

This audit evaluates Agent-OS against the 3-tier Agentic Security Audit checklist (15 categories, ~58 individual checks). The assessment reflects the current state of the codebase **after** the initial Moltbook/OpenClaw vulnerability remediation (Phases 1-7).

| Tier | Category | Status | Score |
|------|----------|--------|-------|
| **1** | 1.1 Credential Storage | PARTIAL | 3/5 |
| **1** | 1.2 Default-Deny Permissions | PARTIAL | 2/5 |
| **1** | 1.3 Cryptographic Agent Identity | PARTIAL | 3/5 |
| **2** | 2.1 Input Classification Gate | PARTIAL | 3/4 |
| **2** | 2.2 Memory Integrity & Provenance | PARTIAL | 3/6 |
| **2** | 2.3 Outbound Secret Scanning | FAIL | 0/4 |
| **2** | 2.4 Skill/Module Signing & Sandboxing | PARTIAL | 2/6 |
| **3** | 3.1 Constitutional Audit Trail | PARTIAL | 3/5 |
| **3** | 3.2 Mutual Agent Authentication | PARTIAL | 2/5 |
| **3** | 3.3 Anti-C2 Pattern Enforcement | PARTIAL | 3/5 |
| **3** | 3.4 Vibe-Code Security Review Gate | PARTIAL | 2/5 |
| **3** | 3.5 Agent Coordination Boundaries | PARTIAL | 3/5 |
| | **TOTAL** | | **29/60 (48%)** |

**Legend:** PASS = all items compliant, PARTIAL = some items compliant, FAIL = no items compliant

---

## TIER 1 — Immediate Wins (Architectural Defaults)

---

### 1.1 Credential Storage

**Score: 3/5 — PARTIAL**

| # | Check | Status | Evidence |
|---|-------|--------|----------|
| 1 | No plaintext secrets in config files | ✅ PASS | `.env.example` contains only placeholder/commented values. `docker-compose.yml` uses `${AGENT_OS_API_KEY:?...}` requiring explicit env var. No plaintext API keys found in `.json`, `.yaml`, `.toml`, `.env`, `.py` files. |
| 2 | No secrets in git history | ✅ PASS | `git log --all --diff-filter=A` confirms no `.env`, `.pem`, `.key`, `.p12`, or `id_rsa` files were ever committed. |
| 3 | Encrypted keystore implemented | ✅ PASS | `src/utils/encryption.py` provides `EncryptionService` (AES-256-GCM) and `CredentialManager` with encrypted file storage. Key hierarchy: env var → OS keyring → encrypted file with machine-specific key derivation (PBKDF2, 100k iterations). Legacy insecure `obs:` format is explicitly rejected. |
| 4 | Non-predictable config paths | ❌ FAIL | Config still uses `~/.agent-os/` as default: `encryption.key` at `~/.agent-os/encryption.key` (line 124), credentials at `~/.agent-os/credentials.enc` (line 392), machine salt at `~/.agent-os/.machine_salt`. These are predictable and targeted by infostealers per the Moltbook findings. |
| 5 | `.gitignore` covers sensitive paths | ❌ FAIL | `.gitignore` exists but does not explicitly list `*.pem`, `*.key`, `.env`, `credentials.enc`, `encryption.key`, or the `~/.agent-os/` directory pattern. Risk of accidental commit of sensitive files. |

**Remediation Required:**
- **[HIGH]** Make config base path runtime-configurable via `AGENT_OS_CONFIG_DIR` environment variable instead of hardcoding `~/.agent-os/`. Default should use XDG conventions (`$XDG_CONFIG_HOME/agent-os/`) or a randomized suffix.
- **[MEDIUM]** Add comprehensive sensitive file patterns to `.gitignore`: `*.pem`, `*.key`, `*.p12`, `.env`, `.env.*`, `credentials.enc`, `encryption.key`, `*.secret`.

---

### 1.2 Default-Deny Permissions / Least Privilege

**Score: 2/5 — PARTIAL**

| # | Check | Status | Evidence |
|---|-------|--------|----------|
| 1 | No default root/admin execution | ✅ PASS | Docker Compose doesn't specify `privileged` mode. The app binds to `127.0.0.1` by default (`.env.example`). No `USER root` in any Dockerfile. |
| 2 | Capabilities declared per-module | ❌ FAIL | `src/tools/permissions.py` (636 lines) implements a comprehensive role-based tool permission system (`PermissionLevel`: NONE/INVOKE/USE/MANAGE/ADMIN) with per-user and per-agent grants. However, there is **no `permissions.manifest`** file. Permissions are granted programmatically, not declaratively. The HTTP route layer is completely disconnected from this system. |
| 3 | Filesystem access scoped | ❌ FAIL | Agent isolation in `src/agents/isolation.py` validates module names against `TRUSTED_MODULE_PREFIXES` and blocks dangerous patterns. However, once an agent is running, filesystem access is not restricted to declared working directories. The subprocess runner in `src/tools/subprocess_runner.py` strips some env vars but does not enforce filesystem boundaries. |
| 4 | Network access scoped | ❌ FAIL | No outbound network restrictions. `src/boundary/client.py` provides a `SmithClient.request_permission()` mechanism for network access checks, but it **fails open** when the daemon is unreachable (`config.fail_closed` defaults vary). No declared endpoint allowlists exist per-module. |
| 5 | Destructive operations gated | ✅ PASS | Constitution enforcement (`src/core/enforcement.py`) blocks explicitly denied patterns. The constitutional rules system gates dangerous operations. Admin-only endpoints (`/shutdown`, `/restart`, agent start/stop) require `require_admin_auth`. |

**Critical Finding — 36 HTTP Endpoints Without Authentication:**

The following route files have **zero authentication** on all or most endpoints:

| Route File | Unprotected Endpoints | Severity |
|------------|----------------------|----------|
| `src/web/routes/constitution.py` | ALL 9 endpoints including `POST /rules`, `PUT /rules/{id}`, `DELETE /rules/{id}` — **anyone can create, modify, or delete constitutional rules** | **CRITICAL** |
| `src/web/routes/images.py` | ALL 10 endpoints including `POST /generate` (resource abuse), `DELETE /gallery/{id}` | **HIGH** |
| `src/web/routes/system.py` | 9 of 11 endpoints including `PUT /settings/{key}` (system config write), `GET /logs` (log data exposure) | **HIGH** |
| `src/web/routes/agents.py` | 4 of 9 endpoints (agent listing, metrics) | **MEDIUM** |
| `src/web/routes/chat.py` | `DELETE /conversations/{id}` missing auth (line 1019), WebSocket auth is in-handler not framework-enforced | **MEDIUM** |

**Remediation Required:**
- **[CRITICAL]** Add `Depends(require_admin_auth)` to all write endpoints in `constitution.py` — this is the governance layer; unauthenticated modification defeats the entire constitutional model.
- **[CRITICAL]** Add authentication to all `images.py` endpoints. `POST /generate` enables resource abuse (GPU/CPU exhaustion).
- **[HIGH]** Add authentication to `system.py` write endpoints (`PUT /settings/{key}`) and sensitive reads (`GET /logs`).
- **[HIGH]** Add `Depends(_authenticate_rest_request)` to `DELETE /conversations/{id}` in `chat.py`.
- **[MEDIUM]** Create a `permissions.manifest` declarative format and enforce it at the HTTP middleware layer, not just the tool layer.
- **[MEDIUM]** The `SmithClient` fail-open default (`return True` at `client.py:215`) should be changed to fail-closed.

---

### 1.3 Cryptographic Agent Identity

**Score: 3/5 — PARTIAL**

| # | Check | Status | Evidence |
|---|-------|--------|----------|
| 1 | Agent keypair generation on init | ✅ PASS | `src/agents/identity.py` implements `AgentIdentity` with Ed25519 keypair generation via the `cryptography` library (falls back to HMAC if unavailable). Each agent gets a unique identity at creation time via `AgentIdentityRegistry.register()`. |
| 2 | All agent actions signed | ❌ FAIL | The identity module provides `sign()` and `verify()` methods, but **integration is incomplete**. The message bus (`src/messaging/bus.py`) does not sign outbound messages. Tool executor does not sign results. Only the registry state file uses HMAC signatures. |
| 3 | Identity anchored to NatLangChain | ❌ FAIL | No NatLangChain integration exists. Agent identity is local-only, stored in memory. No on-chain or external anchoring for tamper-proof provenance. (Note: NatLangChain may not yet exist as infrastructure.) |
| 4 | No self-asserted authority | ✅ PASS | Message bus channel ACLs (`channel_acls` parameter) restrict which agents can publish to which channels. The identity registry uses cryptographic verification, not name-based claims. |
| 5 | Session binding | ✅ PASS | Web sessions use AES-256-GCM encrypted session tokens (`src/web/auth.py`) with PBKDF2-derived machine keys. Sessions are bound to authenticated user identity. |

**Remediation Required:**
- **[HIGH]** Wire `AgentIdentity.sign()` into the message bus `publish()` path so every inter-agent message carries a cryptographic signature.
- **[HIGH]** Wire `AgentIdentityRegistry.verify()` into message delivery so receiving agents reject unsigned/invalid messages.
- **[LOW]** Evaluate NatLangChain integration when infrastructure becomes available.

---

## TIER 2 — Core Enforcement Layer

---

### 2.1 Input Classification Gate

**Score: 3/4 — PARTIAL**

| # | Check | Status | Evidence |
|---|-------|--------|----------|
| 1 | All external input classified before reaching LLM | ✅ PASS | `src/core/input_classifier.py` implements `InputClassifier` with `DATA_PREFIX`/`DATA_SUFFIX` delimiters (`<DATA_CONTEXT>`/`</DATA_CONTEXT>`). `DATA_BOUNDARY_INSTRUCTION` constant provides system prompt instructions for treating delimited content as data-only. |
| 2 | Instruction-like content in data streams flagged | ✅ PASS | `src/agents/seshat/retrieval.py` scans retrieved memories against 7 injection detection patterns: "ignore previous", "you are now", "execute the following", "system prompt", role reassignment, base64 commands, and invisible Unicode markers. Matches are filtered out before reaching the LLM. |
| 3 | Structured input boundaries | ✅ PASS | The `InputClassifier` enforces structured separation with labeled `<DATA_CONTEXT>` sections. `src/core/enforcement.py` uses `TextNormalizer` (NFKC normalization, zero-width character removal, Cyrillic homoglyph mapping) before pattern matching. |
| 4 | No raw HTML/markdown from external sources | ❌ FAIL | No HTML/markdown sanitization layer exists between external data sources and agent reasoning. The `TextNormalizer` handles Unicode attacks but does not strip HTML tags, markdown formatting, or invisible text embedded in formatting. External documents (fetched URLs, uploaded files) pass through without markup sanitization. |

**Remediation Required:**
- **[MEDIUM]** Add HTML/markdown stripping to the `InputClassifier` pipeline. At minimum, strip `<script>`, `<style>`, hidden text in HTML attributes, and invisible markdown constructs before classification.

---

### 2.2 Memory Integrity and Provenance

**Score: 3/6 — PARTIAL**

| # | Check | Status | Evidence |
|---|-------|--------|----------|
| 1 | Every memory entry tagged with metadata | ✅ PASS | `src/memory/storage.py` `BlobMetadata` includes `source_trust_level` (`SourceTrustLevel` enum: USER_DIRECT, AGENT_GENERATED, EXTERNAL_DOCUMENT, LLM_OUTPUT, SYSTEM), `source_agent`, timestamp, and content hash fields. `to_dict()` serializes provenance fields. |
| 2 | Memory entries from untrusted sources quarantined | ❌ FAIL | While trust levels are tagged, there is no quarantine mechanism. All memories regardless of `SourceTrustLevel` are stored in the same store and returned in the same retrieval pipeline. No separation between trusted and untrusted memory pools. |
| 3 | Memory content hashed at write | ✅ PASS | `BlobMetadata` includes content hash computation. The Seshat retrieval pipeline uses hash-based deduplication. |
| 4 | Periodic memory audit | ❌ FAIL | No `MemoryAuditor` or scheduled scan exists. The injection scanning in `src/agents/seshat/retrieval.py` runs at retrieval time (reactive) but not proactively on stored memories. No cron job, background task, or scheduled scan for detecting dormant injection payloads in stored memories. |
| 5 | IntentLog integration for memory ops | ❌ FAIL | `IntentLogStore` (`src/web/intent_log.py`) exists and is substantial (SQLite-backed, 8 API endpoints). However, memory write operations are **not logged** to the intent log. There is no integration between `src/memory/storage.py` and the intent log system. Memory-influenced decisions are not traced. |
| 6 | Memory expiration policy | ✅ PASS | Memory metadata includes timestamps. The storage system supports TTL-based expiration. |

**Remediation Required:**
- **[HIGH]** Implement memory quarantine: memories with `source_trust_level` of EXTERNAL_DOCUMENT or LLM_OUTPUT should be stored in a separate namespace and require explicit promotion to be mixed with trusted memories.
- **[HIGH]** Build a `MemoryAuditor` that periodically scans all stored memories for injection patterns, credential fragments, and instruction-like content (reuse patterns from `seshat/retrieval.py`).
- **[MEDIUM]** Wire memory write/read operations into the `IntentLogStore` so memory-influenced decisions are traceable.

---

### 2.3 Outbound Secret Scanning

**Score: 0/4 — FAIL**

| # | Check | Status | Evidence |
|---|-------|--------|----------|
| 1 | All outbound messages scanned for secrets | ❌ FAIL | No outbound message scanning exists. `src/utils/encryption.py` contains a comprehensive `SensitiveDataRedactor` (patterns for OpenAI keys, HuggingFace tokens, GitHub PATs, AWS keys, Bearer tokens, JWTs, private keys, etc.) but it is **only used for log redaction**. It is not wired into any outbound communication pipeline. |
| 2 | Constitutional rule: agents never transmit credentials | ❌ FAIL | `CONSTITUTION.md` contains a "Security Prohibitions" section with fetch-and-execute prohibition, but **no explicit prohibition on credential transmission**. No constitutional rule prevents agents from including secrets in outbound messages. |
| 3 | Outbound content logging | ❌ FAIL | No outbound communication logging pipeline exists. Agent-to-agent messages via the message bus are not logged with secret redaction. External API calls from agents are not intercepted or audited. |
| 4 | Alert on detection | ❌ FAIL | No alerting mechanism for credential leak attempts. |

**Remediation Required:**
- **[CRITICAL]** Build a `SecretScanner` module. The `SensitiveDataRedactor` in `src/utils/encryption.py` already has all the patterns needed (OpenAI `sk-*`, Anthropic `sk-ant-*`, AWS `AKIA*`, PEM headers, JWTs, etc.). Wire this into:
  1. The message bus `publish()` method — scan all outbound inter-agent messages.
  2. Any HTTP client wrapper used by agents for external API calls.
  3. WebSocket outbound messages.
- **[HIGH]** Add a constitutional rule to `CONSTITUTION.md`: "Agents MUST NEVER transmit API keys, tokens, passwords, private keys, or any credential material in any message, post, or API call."
- **[HIGH]** Log all outbound agent communications (with secrets redacted by the `SensitiveDataRedactor`) for audit purposes.

---

### 2.4 Skill/Module Signing and Sandboxing

**Score: 2/6 — PARTIAL**

| # | Check | Status | Evidence |
|---|-------|--------|----------|
| 1 | All skills/modules cryptographically signed | ❌ FAIL | No code signing. The tool registry (`src/tools/registry.py`) uses HMAC to protect registry **state files** from tampering, but individual tool/skill source files are not signed. Unsigned tools are loaded without verification. |
| 2 | Manifest required per skill | ❌ FAIL | No skill manifest system. Tools register capabilities programmatically (`src/tools/registry.py`) but do not declare network endpoints, files accessed, or shell commands in a verifiable manifest. |
| 3 | Skills run in sandbox | ✅ PASS | `src/tools/executor.py` implements a 3-tier execution model: container (Docker) → subprocess (with restricted env) → in-process fallback. `src/tools/subprocess_runner.py` provides isolated subprocess execution. `src/agents/isolation.py` validates module names against `TRUSTED_MODULE_PREFIXES` allowlist and blocks dangerous patterns (os, sys, subprocess, eval, exec, etc.). |
| 4 | Update diff review | ❌ FAIL | No automated diff review for skill/tool updates. No human approval gate for behavioral changes in tools. |
| 5 | No silent network calls | ❌ FAIL | No network call logging from tools/skills. Tools can make arbitrary HTTP requests without interception or logging. The `SmithClient` permission check exists but is opt-in, not enforced at the network layer. |
| 6 | Skill provenance tracking | ✅ PASS | Tool registry tracks tool metadata including registration source. HMAC-signed state file prevents post-registration tampering of the registry. |

**Remediation Required:**
- **[HIGH]** Implement cryptographic signing for tool/skill modules. Use Ed25519 signatures (the `cryptography` library is already a dependency). Reject unsigned tools at load time.
- **[HIGH]** Create a `tool_manifest.json` schema: each tool declares `network_endpoints`, `file_paths`, `shell_commands`, `apis_called`. The `ToolExecutor` enforces manifest boundaries.
- **[MEDIUM]** Intercept and log all outbound network calls from tool execution (proxy or wrapper around `requests`/`httpx`).
- **[LOW]** Add update diff review: when a tool's source changes, generate a diff and require human approval before re-registration.

---

## TIER 3 — Protocol-Level Maturity

---

### 3.1 Constitutional Audit Trail

**Score: 3/5 — PARTIAL**

| # | Check | Status | Evidence |
|---|-------|--------|----------|
| 1 | Every agent decision logged with reasoning | ✅ PASS | `IntentLogStore` (`src/web/intent_log.py`) captures intent entries with structured fields. 8 API endpoints for querying/managing logs. SQLite-backed persistent storage. |
| 2 | Logs are append-only and tamper-evident | ❌ FAIL | SQLite database is mutable. No append-only enforcement, no hash chaining, no log signing. An attacker with filesystem access could modify or delete log entries without detection. |
| 3 | Human-readable audit format | ✅ PASS | Intent log entries are queryable via REST API with structured JSON responses. Filtering by type, time range, and agent is supported. |
| 4 | Constitutional violations logged separately | ✅ PASS | `src/core/enforcement.py` logs violations. The `src/web/routes/security.py` provides attack detection API endpoints for security event monitoring. `AttackDetector` from Smith agent tracks and categorizes security events. |
| 5 | Retention policy defined | ❌ FAIL | No explicit retention policy. No log rotation, archival, or expiration. Logs grow unbounded. No access control documentation for who can read/delete audit logs. |

**Remediation Required:**
- **[HIGH]** Implement tamper-evident logging: hash-chain each log entry (each entry includes the hash of the previous entry). Optionally sign the chain periodically.
- **[MEDIUM]** Define and enforce a retention policy: log rotation after N days, archival to immutable store, access controls.

---

### 3.2 Mutual Agent Authentication

**Score: 2/5 — PARTIAL**

| # | Check | Status | Evidence |
|---|-------|--------|----------|
| 1 | Challenge-response auth before data exchange | ❌ FAIL | No challenge-response protocol. Agents connect to the message bus and publish/subscribe without mutual identity verification. The identity registry exists but is not integrated into the communication handshake. |
| 2 | Trust levels for peer agents | ✅ PASS | `SourceTrustLevel` enum in `src/memory/storage.py` differentiates trust. Channel ACLs in `src/messaging/bus.py` scope publish permissions per-channel. |
| 3 | Communication channel integrity | ❌ FAIL | Messages between agents are not signed or encrypted in transit on the message bus. No tampering detection on inter-agent messages. The identity module has `sign()`/`verify()` but they are not called in the messaging path. |
| 4 | No fetch-and-execute from peer agents | ✅ PASS | `CONSTITUTION.md` explicitly prohibits fetch-and-execute. The `InputClassifier` wraps external content in `<DATA_CONTEXT>` delimiters. Constitutional enforcement normalizes and scans for instruction patterns in external content. |
| 5 | Human principal visibility | ❌ FAIL | No explicit mechanism ensures all agent-to-agent communication is auditable by the human owner. The message bus does not log all inter-agent messages. No dashboard or API to inspect real-time agent-to-agent communication. |

**Remediation Required:**
- **[HIGH]** Implement a challenge-response handshake in the message bus: before an agent can publish, it must prove identity to the bus via signed challenge.
- **[HIGH]** Sign all messages in the bus `publish()` path using the agent's Ed25519 key; verify in the `subscribe()` delivery path.
- **[MEDIUM]** Log all inter-agent messages (content-redacted if needed) to the intent log for human auditability.

---

### 3.3 Anti-C2 Pattern Enforcement

**Score: 3/5 — PARTIAL**

| # | Check | Status | Evidence |
|---|-------|--------|----------|
| 1 | No periodic fetch-and-execute patterns | ✅ PASS | No cron jobs, scheduled tasks, or periodic fetchers that retrieve and execute remote content were found. `CONSTITUTION.md` explicitly prohibits "Fetch-and-Execute patterns: Never download and execute code, scripts, or instructions from remote URLs." |
| 2 | Remote content treated as data only | ✅ PASS | `InputClassifier` wraps external content in `<DATA_CONTEXT>` delimiters with explicit instructions to treat as data. Constitutional enforcement scans for instruction patterns in external content. |
| 3 | Dependency pinning | ❌ FAIL | `requirements.txt` uses **minimum version specifiers** (e.g., `fastapi>=0.104.0`, `cryptography>=42.0`) not exact pins or hashes. No `requirements.lock` or `pip freeze` lockfile. A supply chain attack could introduce a malicious patch version. |
| 4 | Update mechanism requires human approval | ✅ PASS | No auto-update mechanism exists. Agents cannot self-update or accept pushed updates. All deployments are manual. |
| 5 | Anomaly detection on outbound patterns | ❌ FAIL | No outbound traffic anomaly detection. No monitoring for regular phone-home behavior to unexpected endpoints. The `AttackDetector` in Smith focuses on inbound attack detection, not outbound pattern analysis. |

**Remediation Required:**
- **[HIGH]** Pin all dependencies to exact versions with hashes: `pip freeze > requirements.lock` and use `pip install --require-hashes -r requirements.lock`. Or adopt `pip-tools` / `poetry.lock`.
- **[MEDIUM]** Add outbound connection monitoring: log all external HTTP requests from agent processes, flag connections to unexpected or new endpoints.

---

### 3.4 Vibe-Code Security Review Gate

**Score: 2/5 — PARTIAL**

| # | Check | Status | Evidence |
|---|-------|--------|----------|
| 1 | Security-focused review on AI-generated code | ❌ FAIL | No `.security-review.md` template or mandatory review gate. The two security review documents (`SECURITY_REVIEW_MOLTBOOK_OPENCLAW.md` and this audit) are manual, not part of CI/CD. |
| 2 | Automated security scanning in CI | ❌ FAIL | No CI pipeline configuration found (no `.github/workflows/`, no `Jenkinsfile`, no `.gitlab-ci.yml`). No SAST, dependency vulnerability scanning, or secret detection running on commits. |
| 3 | Default-secure configurations | ✅ PASS | `require_auth` defaults to `True` (Phase 1 fix). `force_https` defaults to `True`. Docker Compose requires `AGENT_OS_API_KEY` to be set (fails if missing). Host binds to `127.0.0.1` by default. |
| 4 | Database access controls verified | ❌ FAIL | SQLite databases used for intent logs, conversation storage, and other data. No Row Level Security (SQLite doesn't support it), no authentication on database files, no rate limiting on database queries. Database files are readable by any process running as the same user. |
| 5 | Attack surface checklist for deployments | ✅ PASS | The Moltbook/OpenClaw security review (`SECURITY_REVIEW_MOLTBOOK_OPENCLAW.md`) serves as a deployment checklist. Auth, rate limiting, input validation, error handling (sanitized error messages), and logging are all addressed. |

**Remediation Required:**
- **[HIGH]** Create a CI pipeline with: (1) `bandit` for Python SAST, (2) `pip-audit` for dependency vulnerabilities, (3) `detect-secrets` for committed secrets, (4) `safety` for known CVEs.
- **[HIGH]** Create a `.security-review.md` template that must be filled before any deployment.
- **[MEDIUM]** Set restrictive file permissions (0o600) on all SQLite database files. Consider WAL mode with file locking for integrity.

---

### 3.5 Agent Coordination Boundaries

**Score: 3/5 — PARTIAL**

| # | Check | Status | Evidence |
|---|-------|--------|----------|
| 1 | All inter-agent coordination visible to human | ❌ FAIL | No dashboard or real-time visibility into agent-to-agent messaging. The message bus operates without logging. Human principal has no way to inspect what agents are telling each other. |
| 2 | Rate limiting on agent-to-agent interactions | ✅ PASS | `src/messaging/bus.py` implements per-agent rate limiting (`max_messages_per_minute=1000`) with sliding window enforcement. Rate-limited agents are blocked from publishing. |
| 3 | Collective action requires human approval | ❌ FAIL | No multi-agent coordination approval gate. If multiple agents converge on a shared action (e.g., all requesting the same external resource, or coordinating system changes), no mechanism detects this convergence or requires human sign-off. |
| 4 | Constitutional transparency rule | ✅ PASS | The constitutional framework requires agents to operate within declared rules. `CONSTITUTION.md` establishes behavioral boundaries. The enforcement engine (`src/core/enforcement.py`) applies these rules consistently. |
| 5 | No autonomous hierarchy formation | ✅ PASS | Channel ACLs prevent agents from establishing unauthorized communication channels. The identity system prevents authority escalation through self-assertion. No mechanism exists for agents to delegate authority to other agents. |

**Remediation Required:**
- **[HIGH]** Build a coordination monitoring dashboard or API: log all inter-agent messages to a queryable store visible to the human principal.
- **[MEDIUM]** Implement collective action detection: if N agents request the same resource or action within a time window, pause and require human approval.

---

## Quick Scan Results

### 1. Plaintext Secrets Scan
**Result:** No plaintext secrets found in source code. `.env.example` contains only commented placeholders. `docker-compose.yml` uses required environment variables with fail-fast syntax.

### 2. Hardcoded URL Fetch Patterns
**Result:** HTTP fetch patterns found in:
- `src/web/routes/chat.py` — Ollama API calls (localhost model inference, expected)
- `src/boundary/client.py` — Smith daemon connection (localhost, expected)
- Various test files — Test fixtures (acceptable)
- No remote URL fetch-and-execute patterns detected.

### 3. Shell Execution Patterns
**Result:** Subprocess usage found in:
- `src/tools/executor.py` — Sandboxed tool execution (expected, uses restricted env)
- `src/agents/isolation.py` — Agent process isolation (expected, validates module names)
- `src/tools/subprocess_runner.py` — Isolated tool runner (expected)
- All subprocess paths have input validation. No `os.system()`, `eval()`, or unvalidated `exec()` calls found.

### 4. Predictable Config Paths
**Result:** `~/.agent-os/` used in:
- `src/utils/encryption.py:124` — `~/.agent-os/encryption.key`
- `src/utils/encryption.py:392` — `~/.agent-os/credentials.enc`
- `src/web/auth.py` — `~/.agent-os/.machine_salt`
- **Recommendation:** Make configurable via `AGENT_OS_CONFIG_DIR` env var.

### 5. Sensitive File Types
**Result:** No `.pem`, `.key`, `.env`, `.p12`, or `id_rsa` files found in repository.

---

## Prioritized Remediation Roadmap

### Phase A — Critical (Implement Immediately)

| # | Item | Category | Effort |
|---|------|----------|--------|
| A1 | Add authentication to ALL constitution.py endpoints (POST/PUT/DELETE require admin, GET require auth) | 1.2 | Small |
| A2 | Add authentication to ALL images.py endpoints | 1.2 | Small |
| A3 | Add authentication to system.py write endpoints and sensitive reads | 1.2 | Small |
| A4 | Add auth to `DELETE /conversations/{id}` in chat.py | 1.2 | Trivial |
| A5 | Build `SecretScanner` and wire into message bus outbound path | 2.3 | Medium |
| A6 | Add constitutional rule prohibiting credential transmission | 2.3 | Trivial |

### Phase B — High Priority (Implement This Sprint)

| # | Item | Category | Effort |
|---|------|----------|--------|
| B1 | Pin all dependencies to exact versions with hashes | 3.3 | Small |
| B2 | Sign inter-agent messages with Ed25519 keys via identity module | 1.3, 3.2 | Medium |
| B3 | Memory quarantine for untrusted source levels | 2.2 | Medium |
| B4 | Build `MemoryAuditor` for periodic injection scans | 2.2 | Medium |
| B5 | Implement tamper-evident hash-chained audit logs | 3.1 | Medium |
| B6 | Create CI pipeline with SAST, dependency scanning, secret detection | 3.4 | Medium |
| B7 | Cryptographic signing for tool/skill modules | 2.4 | Large |

### Phase C — Medium Priority (Implement Next Sprint)

| # | Item | Category | Effort |
|---|------|----------|--------|
| C1 | Make config path runtime-configurable (`AGENT_OS_CONFIG_DIR`) | 1.1 | Small |
| C2 | Add HTML/markdown sanitization to InputClassifier | 2.1 | Small |
| C3 | Wire memory operations into IntentLog | 2.2 | Medium |
| C4 | Log all inter-agent messages for human auditability | 3.2, 3.5 | Medium |
| C5 | Add comprehensive sensitive patterns to .gitignore | 1.1 | Trivial |
| C6 | Outbound network monitoring for agent processes | 3.3 | Medium |
| C7 | Tool manifest system (declared capabilities per tool) | 2.4 | Large |
| C8 | Collective action detection and approval gate | 3.5 | Large |

### Phase D — Hardening (Ongoing)

| # | Item | Category | Effort |
|---|------|----------|--------|
| D1 | Challenge-response handshake for message bus | 3.2 | Large |
| D2 | Define log retention and access control policy | 3.1 | Small |
| D3 | SQLite file permissions and WAL mode | 3.4 | Small |
| D4 | Security review template (`.security-review.md`) | 3.4 | Small |
| D5 | Change SmithClient fail-open to fail-closed default | 1.2 | Trivial |
| D6 | NatLangChain identity anchoring (when available) | 1.3 | Large |

---

## Audit Log

| Repo Name | Date Audited | Tier 1 | Tier 2 | Tier 3 | Notes |
|-----------|--------------|--------|--------|--------|-------|
| Agent-OS | 2026-02-19 | PARTIAL (8/15) | PARTIAL (8/20) | PARTIAL (13/25) | Post-Moltbook remediation. 29/60 (48%). Critical: unauthed constitution & image endpoints, zero outbound secret scanning. |

---

*Audited against the Post-Moltbook Hardening Guide v1.0 (Agentic-Security-Audit.md)*
*"First generation built the internet. First generation built agent infrastructure. Second generation secures it."*
