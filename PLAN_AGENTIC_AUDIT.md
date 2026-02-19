# Implementation Plan: Agentic Security Audit Remediation

**Source:** `SECURITY_AUDIT_AGENTIC.md` — Post-Moltbook Hardening Guide v1.0
**Baseline Score:** 29/60 (48%)
**Target Score:** 52+/60 (87%+)

This plan addresses all 31 open findings across 12 categories (Tiers 1-3), organized into 6 phases ordered by criticality, dependency, and blast radius. Each phase lists exact files, functions, and patterns.

---

## Phase 1: Authentication Gap Closure (CRITICAL)

**Goal:** Close all unauthenticated write/delete endpoints. This is the single highest-impact fix — 36 endpoints currently lack auth.

**Rationale:** Unauthenticated constitution modification defeats the entire governance model. Unauthenticated image generation enables resource abuse. These are all small, mechanical changes reusing existing auth patterns.

### 1.1 Consolidate auth helpers

**File:** `src/web/auth_helpers.py` (new)

Currently three near-identical auth functions exist in `system.py`, `agents.py`, `chat.py`, and `memory.py`. Create one shared module:

- `require_authenticated_user(request) -> str` — extracts session token from cookie or `Authorization: Bearer` header, validates via `get_user_store().validate_session()`, returns `user_id`. Raises 401 on failure.
- `require_admin_user(request) -> str` — calls `require_authenticated_user()`, then checks `user.role == UserRole.ADMIN`. Raises 403 if not admin.

Update `system.py`, `agents.py`, `chat.py`, `memory.py`, `intent_log.py` to import from this shared module instead of defining their own.

### 1.2 Secure constitution.py endpoints

**File:** `src/web/routes/constitution.py`

| Endpoint | Auth Required |
|----------|--------------|
| `GET /overview` | `require_authenticated_user` |
| `GET /sections` | `require_authenticated_user` |
| `GET /rules` | `require_authenticated_user` |
| `GET /rules/{rule_id}` | `require_authenticated_user` |
| `POST /rules` | `require_admin_user` |
| `PUT /rules/{rule_id}` | `require_admin_user` |
| `DELETE /rules/{rule_id}` | `require_admin_user` |
| `POST /validate` | `require_authenticated_user` |
| `GET /search` | `require_authenticated_user` |

Add `from src.web.auth_helpers import require_authenticated_user, require_admin_user` and `Depends()` to each endpoint signature.

### 1.3 Secure images.py endpoints

**File:** `src/web/routes/images.py`

| Endpoint | Auth Required |
|----------|--------------|
| `GET /models` | `require_authenticated_user` |
| `GET /models/{model_id}` | `require_authenticated_user` |
| `POST /generate` | `require_authenticated_user` |
| `GET /generate/{job_id}` | `require_authenticated_user` |
| `GET /jobs` | `require_authenticated_user` |
| `GET /gallery` | `require_authenticated_user` |
| `DELETE /gallery/{image_id}` | `require_admin_user` |
| `GET /image/{image_id}` | `require_authenticated_user` |
| `GET /stats` | `require_authenticated_user` |
| `WebSocket /ws` | Auth check in handler (same pattern as chat.py WebSocket) |

### 1.4 Secure system.py sensitive endpoints

**File:** `src/web/routes/system.py`

| Endpoint | Auth Required |
|----------|--------------|
| `GET /settings` | `require_admin_user` |
| `GET /settings/{key}` | `require_admin_user` |
| `PUT /settings/{key}` | `require_admin_user` |
| `GET /logs` | `require_admin_user` |
| `GET /metrics` | `require_authenticated_user` |

Leave `GET /info`, `GET /status`, `GET /health`, `GET /version`, `GET /dreaming` public (operational monitoring).

### 1.5 Fix chat.py delete endpoint

**File:** `src/web/routes/chat.py`

Add `user_id: str = Depends(require_authenticated_user)` to `delete_conversation()` at line 1019 (currently the only chat mutation endpoint missing auth).

### 1.6 Change SmithClient fail-open to fail-closed

**File:** `src/boundary/client.py`

At line 215, change `return True  # Fail open (dangerous!)` to `return False` and update the comment. When the Smith daemon is unreachable, deny by default.

**Tests:** Run `pytest tests/test_web.py tests/test_routes*.py` (skip known pyo3 failures). Manually verify that unauthenticated requests to secured endpoints return 401/403.

---

## Phase 2: Outbound Secret Scanning (CRITICAL)

**Goal:** Prevent credential leakage through inter-agent messages and external communications. This is currently a 0/4 score — the only FAIL tier.

### 2.1 Build SecretScanner module

**File:** `src/security/secret_scanner.py` (new)

Wrap the existing `SensitiveDataRedactor` from `src/utils/encryption.py` into a scanner that **blocks** (not just redacts):

```python
class SecretScanner:
    def __init__(self, redactor: SensitiveDataRedactor = None):
        self.redactor = redactor or SensitiveDataRedactor()

    def scan(self, content: str) -> ScanResult:
        """Returns ScanResult with found_secrets: bool, patterns_matched: List[str], redacted_content: str"""

    def scan_and_block(self, content: str) -> str:
        """Raises SecretLeakageError if secrets detected. Returns content unchanged if clean."""
```

The scanner reuses all patterns from `SensitiveDataRedactor` plus adds:
- Anthropic keys: `sk-ant-api03-*`
- Generic high-entropy string detection (Shannon entropy > 4.5 for strings > 20 chars)

### 2.2 Wire into message bus

**File:** `src/messaging/bus.py`

In `InMemoryMessageBus.publish()`, after the rate limit and ACL checks (before message delivery), add:

```python
# Scan message content for secrets
scanner = get_secret_scanner()
content_str = str(message.content) if hasattr(message, 'content') else str(message)
scan_result = scanner.scan(content_str)
if scan_result.found_secrets:
    logger.critical(f"BLOCKED: Agent '{source}' attempted to send message containing secrets: {scan_result.patterns_matched}")
    # Log to intent log if available
    return False
```

### 2.3 Add constitutional rule

**File:** `CONSTITUTION.md`

Add to the "Security Prohibitions" section:

```
### Credential Transmission Prohibition
No agent shall transmit API keys, tokens, passwords, private keys, session secrets,
or any credential material in any message, post, API call, or inter-agent communication.
Credentials must only be accessed through the CredentialManager and never included in
message payloads.
```

### 2.4 Outbound communication logging

**File:** `src/messaging/bus.py`

Add a `_log_outbound_message()` method that writes a redacted copy of every published message to the audit log (already exists as `self._audit_log`). Ensure the redactor runs before logging so no secrets appear in logs.

**Tests:** Unit test with messages containing fake API keys (`sk-test1234...`), tokens, and PEM blocks. Verify they are blocked. Verify clean messages pass.

---

## Phase 3: Message Signing & Mutual Authentication (HIGH)

**Goal:** Wire the existing Ed25519 identity system into the message bus so all inter-agent messages are cryptographically signed and verified.

### 3.1 Sign messages on publish

**File:** `src/messaging/bus.py`

In `InMemoryMessageBus.publish()`, after secret scanning and before delivery:

```python
if self._identity_registry and hasattr(message, 'metadata'):
    source_agent = message.metadata.get('source_agent') or message.source
    if self._identity_registry.is_registered(source_agent):
        content_hash = hashlib.sha256(str(message.content).encode()).hexdigest()
        signature = self._identity_registry.sign_message(
            source_agent, message.request_id, source_agent, content_hash
        )
        message.metadata['_signature'] = signature.hex()
        message.metadata['_content_hash'] = content_hash
```

### 3.2 Verify messages on delivery

**File:** `src/messaging/bus.py`

In the subscriber delivery loop, before calling the handler:

```python
if self._identity_registry and hasattr(message, 'metadata'):
    sig_hex = message.metadata.get('_signature')
    content_hash = message.metadata.get('_content_hash')
    source_agent = message.metadata.get('source_agent') or message.source
    if sig_hex and content_hash:
        if not self._identity_registry.verify_message_payload(
            source_agent, message.request_id, source_agent, content_hash, bytes.fromhex(sig_hex)
        ):
            logger.warning(f"REJECTED: Message from '{source_agent}' failed signature verification")
            continue  # Skip delivery
```

### 3.3 Log inter-agent messages for human visibility

**File:** `src/messaging/bus.py`

Extend the existing `_audit_log` to capture all message metadata (channel, source, target, timestamp, signature status) in a format queryable by the human principal. Add a new method:

```python
def get_message_log(self, channel: str = None, agent: str = None, limit: int = 100) -> List[dict]:
    """Return recent inter-agent message log for human inspection."""
```

**Tests:** Unit test with two registered agents. Verify signed messages are delivered. Verify forged/unsigned messages are rejected. Verify the message log is populated.

---

## Phase 4: Memory Hardening (HIGH)

**Goal:** Implement memory quarantine, periodic auditing, and intent log integration.

### 4.1 Memory quarantine

**File:** `src/memory/storage.py`

Add a `QUARANTINE_TRUST_LEVELS` set:
```python
QUARANTINE_TRUST_LEVELS = {SourceTrustLevel.EXTERNAL_DOCUMENT, SourceTrustLevel.LLM_OUTPUT}
```

In `BlobStorage.store()`, if `metadata.source_trust_level in QUARANTINE_TRUST_LEVELS`:
- Prefix the `blob_id` with `quarantine/` namespace
- Set a `quarantined=True` flag in metadata
- Do NOT return quarantined blobs in standard `retrieve()` calls unless `include_quarantined=True` is passed

In `BlobStorage.retrieve()`, add `include_quarantined: bool = False` parameter. By default, skip blobs with `quarantined=True` metadata.

Add `promote_from_quarantine(blob_id) -> BlobMetadata` — moves a blob from quarantine to trusted storage after human review.

### 4.2 Build MemoryAuditor

**File:** `src/memory/auditor.py` (new)

```python
class MemoryAuditor:
    """Periodically scans stored memories for injection patterns, credential fragments, and anomalies."""

    def __init__(self, storage: BlobStorage, scanner: SecretScanner, injection_patterns: list):
        ...

    def audit_all(self) -> AuditReport:
        """Full scan of all stored memories. Returns report with findings."""

    def audit_recent(self, hours: int = 24) -> AuditReport:
        """Scan memories written in the last N hours."""

    def schedule(self, interval_hours: int = 6):
        """Start background thread running audit_recent on interval."""
```

Reuse injection patterns from `src/agents/seshat/retrieval.py` and secret patterns from `SensitiveDataRedactor`.

### 4.3 Wire memory operations into IntentLog

**File:** `src/memory/storage.py`

In `BlobStorage.store()`, `retrieve()`, and `delete()`, add optional `intent_log: IntentLogStore` parameter. When provided, log:
- `MEMORY_CREATE` on store with `source_trust_level`, `source_agent`, `blob_id`
- `MEMORY_SEARCH` on retrieve with `blob_id`, accessor identity
- `MEMORY_DELETE` on delete with `blob_id`, deleter identity

Use the existing `IntentType.MEMORY_CREATE`, `MEMORY_DELETE`, `MEMORY_SEARCH` enums from `intent_log.py`.

**Tests:** Unit test quarantine: store a blob with `EXTERNAL_DOCUMENT` trust, verify it's not returned by default retrieve, verify it's returned with `include_quarantined=True`. Test promotion. Test auditor finding an injected memory.

---

## Phase 5: Supply Chain & CI Hardening (HIGH)

**Goal:** Pin dependencies, harden CI pipeline, add security review gates.

### 5.1 Pin dependencies with hashes

**Action:** Generate a locked requirements file:

```bash
pip freeze > requirements.lock
# OR use pip-compile from pip-tools:
pip-compile --generate-hashes requirements.txt -o requirements.lock
```

**File:** `requirements.lock` (new) — exact versions with hashes for all production dependencies.

Update `Dockerfile` / CI to use `pip install --require-hashes -r requirements.lock`.

Keep `requirements.txt` as the human-readable source with minimum versions.

### 5.2 Add cryptography to pyproject.toml

**File:** `pyproject.toml`

Add `cryptography>=42.0` and `defusedxml>=0.7` to the `dependencies` list (currently missing — only in `requirements.txt`).

### 5.3 Harden CI security checks

**File:** `.github/workflows/security.yml`

Remove `continue-on-error: true` from critical checks:
- `bandit` (SAST) — make failures block merges
- `safety` / `pip-audit` (dependency vulnerabilities) — make failures block merges
- `gitleaks` (secret detection) — make failures block merges

Keep `continue-on-error: true` only for informational checks (license audit, container scan).

### 5.4 Add security review template

**File:** `.security-review.md` (new)

Template checklist to be filled before any deployment:
- [ ] Authentication on all endpoints verified
- [ ] No plaintext secrets in config or code
- [ ] Dependencies pinned and scanned
- [ ] Input validation on all user-facing inputs
- [ ] Error messages sanitized (no internal details)
- [ ] Rate limiting configured
- [ ] Logging covers security-relevant events
- [ ] HTTPS enforced in production

### 5.5 Update .gitignore

**File:** `.gitignore`

Add:
```
# Sensitive files
*.pem
*.key
*.p12
*.pfx
id_rsa
id_ed25519
credentials.enc
encryption.key
*.secret
.machine_salt
```

**Tests:** `pip install --require-hashes -r requirements.lock` succeeds. CI pipeline runs with security checks blocking.

---

## Phase 6: Structural Hardening (MEDIUM)

**Goal:** Address remaining medium-priority items across all tiers.

### 6.1 Configurable config path

**Files:** `src/utils/encryption.py`, `src/web/auth.py`

Replace all hardcoded `~/.agent-os/` paths with:
```python
def get_config_dir() -> Path:
    """Return the Agent-OS config directory, configurable via env var."""
    config_dir = os.environ.get("AGENT_OS_CONFIG_DIR")
    if config_dir:
        return Path(config_dir)
    # XDG fallback
    xdg_config = os.environ.get("XDG_CONFIG_HOME", os.path.expanduser("~/.config"))
    return Path(xdg_config) / "agent-os"
```

Add this to a shared module (e.g., `src/utils/paths.py`) and update:
- `src/utils/encryption.py:124` (`encryption.key` path)
- `src/utils/encryption.py:392` (`credentials.enc` path)
- `src/web/auth.py` (`.machine_salt` path)

### 6.2 HTML/Markdown sanitization in InputClassifier

**File:** `src/core/input_classifier.py`

Add a `sanitize_markup(text: str) -> str` method to `InputClassifier`:
- Strip `<script>`, `<style>`, `<iframe>`, `<object>`, `<embed>` tags and content
- Strip HTML event attributes (`onclick`, `onload`, etc.)
- Strip invisible text (CSS `display:none`, `visibility:hidden` in inline styles)
- Strip zero-width joiners and other Unicode tricks (extend existing `TextNormalizer`)
- Preserve readable text content

Call `sanitize_markup()` before `classify()` for all external data inputs.

### 6.3 Tamper-evident audit logs

**File:** `src/web/intent_log.py`

Add hash chaining to `IntentLogStore`:

In `log_intent()`:
1. After creating the entry, compute `entry_hash = SHA256(previous_hash + entry_id + user_id + intent_type + timestamp + details_json)`
2. Store `entry_hash` and `previous_hash` as additional columns in the SQLite table
3. Add `verify_chain(start_entry_id=None) -> ChainVerificationResult` method that walks the chain and reports any breaks

Add a migration to extend the `intent_log` table with `entry_hash TEXT` and `previous_hash TEXT` columns.

### 6.4 Tool/skill manifest system

**File:** `src/tools/manifest.py` (new)

```python
@dataclass
class ToolManifest:
    tool_name: str
    version: str
    author: str
    signature: str  # Ed25519 signature of manifest content
    permissions:
        network_endpoints: List[str]  # Allowed outbound URLs/patterns
        file_paths: List[str]         # Allowed filesystem paths (glob patterns)
        shell_commands: List[str]     # Allowed shell commands
        apis_called: List[str]        # External API identifiers
```

**File:** `src/tools/registry.py`

In `ToolRegistry.register()`, optionally accept a `ToolManifest`. If provided, verify its signature. Store manifest alongside tool registration.

**File:** `src/tools/executor.py`

In `_execute_subprocess()` and `_execute_in_process()`, check the tool's manifest before execution. If the tool attempts an undeclared network call or file access, block and log.

### 6.5 Collective action detection

**File:** `src/messaging/coordination.py` (new)

```python
class CoordinationMonitor:
    """Detects when multiple agents converge on the same resource or action."""

    def __init__(self, threshold: int = 3, window_seconds: int = 60):
        ...

    def record_action(self, agent_name: str, action_type: str, target: str):
        """Record an agent action. Raises CoordinationAlert if threshold reached."""

    def get_convergence_report(self) -> List[ConvergenceEvent]:
        """Return recent convergence events for human review."""
```

Wire into the message bus: when multiple agents publish to the same channel targeting the same resource within the time window, pause delivery and log a coordination alert.

### 6.6 SQLite file permissions

**Files:** `src/web/intent_log.py`, `src/web/dependencies.py`

After creating SQLite databases, set file permissions:
```python
os.chmod(db_path, 0o600)  # Owner read/write only
```

Enable WAL mode for better concurrent access:
```python
conn.execute("PRAGMA journal_mode=WAL")
```

---

## Dependency Graph

```
Phase 1 (Auth)          ──> No dependencies, implement first
Phase 2 (Secret Scan)   ──> No dependencies, implement in parallel with Phase 1
Phase 3 (Msg Signing)   ──> Depends on Phase 2 (scanner wired before signer in publish())
Phase 4 (Memory)        ──> Depends on Phase 2 (SecretScanner used by MemoryAuditor)
Phase 5 (Supply Chain)  ──> No dependencies, implement in parallel with Phases 1-2
Phase 6 (Structural)    ──> Depends on Phases 1-4 being stable
```

**Parallelizable:** Phases 1, 2, and 5 can be implemented simultaneously.

---

## Expected Score After Implementation

| Category | Before | After | Delta |
|----------|--------|-------|-------|
| 1.1 Credential Storage | 3/5 | 5/5 | +2 |
| 1.2 Default-Deny Permissions | 2/5 | 4/5 | +2 |
| 1.3 Cryptographic Agent Identity | 3/5 | 4/5 | +1 |
| 2.1 Input Classification Gate | 3/4 | 4/4 | +1 |
| 2.2 Memory Integrity | 3/6 | 6/6 | +3 |
| 2.3 Outbound Secret Scanning | 0/4 | 4/4 | +4 |
| 2.4 Skill/Module Signing | 2/6 | 4/6 | +2 |
| 3.1 Constitutional Audit Trail | 3/5 | 5/5 | +2 |
| 3.2 Mutual Agent Authentication | 2/5 | 4/5 | +2 |
| 3.3 Anti-C2 Pattern Enforcement | 3/5 | 4/5 | +1 |
| 3.4 Vibe-Code Security Review | 2/5 | 4/5 | +2 |
| 3.5 Agent Coordination Boundaries | 3/5 | 4/5 | +1 |
| **TOTAL** | **29/60** | **52/60** | **+23** |
| **Percentage** | **48%** | **87%** | |

---

## Files Modified (Summary)

| Phase | New Files | Modified Files |
|-------|-----------|---------------|
| 1 | `src/web/auth_helpers.py` | `constitution.py`, `images.py`, `system.py`, `chat.py`, `agents.py`, `memory.py`, `intent_log.py`, `client.py` |
| 2 | `src/security/secret_scanner.py` | `bus.py`, `CONSTITUTION.md` |
| 3 | — | `bus.py` |
| 4 | `src/memory/auditor.py` | `storage.py` |
| 5 | `requirements.lock`, `.security-review.md` | `pyproject.toml`, `security.yml`, `.gitignore` |
| 6 | `src/utils/paths.py`, `src/tools/manifest.py`, `src/messaging/coordination.py` | `encryption.py`, `auth.py`, `input_classifier.py`, `intent_log.py`, `registry.py`, `executor.py`, `dependencies.py` |

**Total: 8 new files, ~20 modified files across 6 phases.**
