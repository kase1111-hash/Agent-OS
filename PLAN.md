# Implementation Plan: Fix All Audit Findings (VIBE_CHECK_AUDIT_V3)

This plan addresses all 22 findings from `VIBE_CHECK_AUDIT_V3.md` (9 HIGH, 9 MEDIUM, 4 LOW).

## Already-Fixed Findings (4 of 22)

These findings from the audit are **already resolved** in current code:

| # | Finding | Evidence |
|---|---------|----------|
| 2 | Auth defaults to disabled | `require_auth: bool = True` at `config.py:48` |
| 5 | Inter-agent messages unsigned | `bus.py` has `_sign_message()` + `_verify_message_signature()` enforced in publish flow |
| 8 | Unauthenticated constitution endpoints | All POST/PUT/DELETE use `Depends(require_admin_user)` |
| 12 | .env.example risky defaults | Host is `127.0.0.1`, Grafana password is commented out with generation instructions |

**Remaining: 18 findings to fix (6 HIGH, 7 MEDIUM, 5 LOW)**

---

## Phase 1: HIGH Severity Fixes

### Fix 1 — Remove XOR Encryption Fallback
**Finding:** #1 (Session secret XOR fallback)
**File:** `src/web/auth.py` (lines 442-507)
**Changes:**
- Remove XOR fallback path in `_encrypt_secret()` (lines 456-460) — always use AES-256-GCM
- Remove legacy XOR format support in `_decrypt_secret()` (64-byte and 65-byte formats at lines 471-489)
- Make `cryptography` a hard import (remove try/except fallback)
- Add migration: if legacy-encrypted data detected, log error telling admin to re-encrypt
- Generate and store a random persistent key at `~/.agent-os/.session_key` (0o600 permissions) instead of deriving from hostname+username

### Fix 3 — Implement Scoped API Keys
**Finding:** #3 (API key grants unscoped ADMIN access)
**File:** `src/web/auth.py`
**Changes:**
- Add `ApiKeyScope` enum: `READ_CHAT`, `WRITE_CHAT`, `READ_MEMORY`, `WRITE_MEMORY`, `ADMIN`
- Add `api_keys` table: `(key_hash TEXT, scopes TEXT, created_at TEXT, expires_at TEXT, description TEXT)`
- Add `create_api_key()`, `revoke_api_key()`, `validate_api_key(token) -> (user_id, scopes)`
- Add `require_scope(scope)` FastAPI dependency
- Backward compat: single `AGENT_OS_API_KEY` env var treated as legacy admin key with deprecation warning in logs

### Fix 4 — Implement Prompt Sanitization Layer
**Finding:** #4 (No prompt sanitization layer)
**File:** `src/web/routes/chat.py`
**New file:** `src/core/input_sanitizer.py`
**Changes:**
- Create `PromptSanitizer` class:
  - `sanitize(text: str) -> str`: strips known injection patterns (role impersonation, system prompt overrides, delimiter escapes)
  - `should_block(detection_score: float) -> bool`: blocks when score > configurable threshold (default 0.8)
- Wire into chat endpoint: existing attack detection score → block if threshold exceeded → sanitize → send to LLM
- Add `prompt_injection_block_threshold: float = 0.8` to `WebConfig`
- Wrap external/retrieved content in `<DATA_CONTEXT>` delimiters before LLM inclusion

### Fix 6 — Add Smith Mutual Validation
**Finding:** #6 (Smith agent single point of failure)
**File:** `src/agents/whisper/router.py` (line 99)
**New file:** `src/agents/whisper/integrity.py`
**Changes:**
- Create `SmithIntegrityChecker`:
  - Validates Smith agent's code hash against known-good hash at startup
  - Periodic re-validation heartbeat
  - Triggers fail-closed LOCKDOWN if Smith unavailable or tampered
- Update router to call integrity checker before routing to Smith
- Fail-closed: if Smith unavailable, reject all security-sensitive requests (don't silently pass)

### Fix 7 — Implement S3 Instruction Integrity Validation
**Finding:** #7 (S3 instruction integrity missing)
**File:** `src/agents/smith/attack_detection/analyzer.py` (line 673)
**New file:** `src/agents/smith/instruction_integrity.py`
**Changes:**
- Create `InstructionIntegrityValidator`:
  - Ed25519 signatures for instruction files
  - Validates before loading
  - Rejects tampered instructions with clear error logging
- Wire into analyzer where S3 validation is referenced

### Fix 9 — Wire SensitiveDataRedactor to Outbound Messages
**Finding:** #9 (No outbound secret scanning)
**Files:** `src/utils/encryption.py`, `src/web/routes/chat.py`
**Changes:**
- Add `redact_outbound_message(message: str) -> str` wrapper in `encryption.py`
- Wire into chat HTTP response path: redact before returning LLM responses
- Wire into WebSocket send path: redact before `websocket.send_text()`
- Log redaction events to intent log for audit trail

---

## Phase 2: MEDIUM Severity Fixes

### Fix 10 — Encrypt Conversation Content at Rest
**Finding:** #10 (Plaintext conversation storage)
**File:** `src/web/conversation_store.py`
**Changes:**
- Add `_encrypt_content()` / `_decrypt_content()` using existing AES-256-GCM from `src/utils/encryption.py`
- Encrypt `content` field on write, decrypt on read
- Add `encrypted INTEGER DEFAULT 0` column to messages table
- Migration: background task encrypts existing plaintext messages
- Configurable via `AGENT_OS_ENCRYPT_CONVERSATIONS=true` (default: true)

### Fix 11 — Implement Automatic Key Rotation
**Finding:** #11 (No automatic key rotation policy)
**File:** `src/memory/keys.py`
**Changes:**
- Add `KeyRotationScheduler` class:
  - Configurable TTL (default 90 days) via `AGENT_OS_KEY_ROTATION_TTL_DAYS`
  - Grace period for old keys (default 7 days)
  - Background asyncio task checks key ages periodically
  - Logs warnings when keys approach rotation deadline
- Wire into `KeyManager` initialization
- Add monitoring metrics for key age

### Fix 13 — Validate Memory Accessor Identity
**Finding:** #13 (Memory accessor identity spoofable)
**File:** `src/memory/seshat/consent_integration.py` (lines 93-205)
**Changes:**
- In `verify_access()`, `verify_store()`, `verify_delete()`:
  - Validate accessor is non-empty string matching expected format
  - Cross-reference accessor against authenticated request context (session user_id)
  - Reject mismatched accessor/authenticated identity with clear error

### Fix 14 — Implement Human Approval Handler
**Finding:** #14 (Escalation callbacks lack human handler)
**File:** `src/boundary/daemon/enforcement.py`
**New file:** `src/web/routes/approvals.py`
**Changes:**
- Add approval queue: pending actions requiring human review
- Add REST endpoints: `GET /approvals`, `POST /approvals/{id}/approve`, `POST /approvals/{id}/deny`
- Wire escalation callback to push to approval queue instead of auto-resolving
- Require admin auth on all approval endpoints

### Fix 15 — Implement Memory Trust Segregation
**Finding:** #15 (Untrusted memories not quarantined)
**File:** `src/agents/seshat/agent.py`
**Changes:**
- Add `quarantined INTEGER DEFAULT 0` column to memory storage
- Untrusted sources (`EXTERNAL_DOCUMENT`, `LLM_OUTPUT`) quarantined by default
- Quarantined memories get reduced relevance scores in retrieval (0.5x weight)
- Quarantined memories excluded from default retrieval; include only with explicit `include_quarantined=True` flag

### Fix 16 — Reject Unsigned Tool Manifests
**Finding:** #16 (Unsigned manifests accepted)
**File:** `src/tools/manifest.py` (lines 85-90, 98-104)
**Changes:**
- Change `verify_signature()` to return `False` for unsigned manifests (currently returns `True`)
- Change ImportError catch to return `False` (fail closed, not open)
- Add `AGENT_OS_ALLOW_UNSIGNED_MANIFESTS=false` env var for dev override
- Log rejection events

### Fix 17 — Add Code Verification for Dynamic Agent Loading
**Finding:** #17 (Agent loading without code verification)
**File:** `src/agents/loader.py` (lines 364-370, 436-443)
**Changes:**
- Before `importlib.import_module()` or `spec_from_file_location()`:
  - Compute SHA-256 hash of source file
  - Verify against known-good hash or Ed25519 signature
  - Reject unsigned/tampered agent code
- Add `AGENT_OS_ALLOW_UNSIGNED_AGENTS=false` env var for dev override
- Add `.agent-signatures/` directory for storing agent code signatures

---

## Phase 3: LOW Severity Fixes

### Fix 18 — Default Docker Binding to 127.0.0.1
**Finding:** L5 Docker binding
**Files:** `Dockerfile`, `docker-compose.yml`
**Changes:**
- Change `ENV AGENT_OS_WEB_HOST=0.0.0.0` to `127.0.0.1` in Dockerfile
- Change docker-compose port binding to `127.0.0.1:8080:8080`
- Add comment: "Use 0.0.0.0 only behind a reverse proxy"

### Fix 19 — Remove unsafe-inline from CSP
**Finding:** L5 CSP unsafe-inline
**File:** `src/web/middleware.py` (lines 147-154)
**Changes:**
- Remove `'unsafe-inline'` from `style-src`
- Add nonce-based style loading if inline styles are needed
- Or use hash-based CSP for known inline styles

### Fix 20 — Add CSRF Token Support
**Finding:** L5 No CSRF tokens
**File:** `src/web/middleware.py`
**Changes:**
- Add CSRF token middleware for session-authenticated routes
- Generate token on session creation, store in session
- Validate `X-CSRF-Token` header on POST/PUT/DELETE with cookie auth
- Skip CSRF for Bearer token auth (not vulnerable to CSRF)

### Fix 21 — Fix HTTPException Information Leakage
**Finding:** L5 HTTPException leaks
**File:** `src/web/routes/security.py` (15 occurrences)
**Changes:**
- Replace all `raise HTTPException(status_code=500, detail=str(e))` with:
  ```python
  logger.error(f"Operation failed: {e}", exc_info=True)
  raise HTTPException(status_code=500, detail="Internal server error")
  ```

### Fix 22 — Generate Dependency Lock File
**Finding:** L4 No lock file
**Location:** Project root
**Changes:**
- Run `pip-compile requirements.txt -o requirements.lock`
- Commit the lock file
- Update CI to install from lock file

---

## Phase 4: Update Audit Report

### Update VIBE_CHECK_AUDIT_V3.md
- Add "Resolution Status" column to finding summary table
- Mark already-fixed findings (2, 5, 8, 12)
- Mark all newly-fixed findings with commit references
- Update layer verdicts: L2 → PASS, L3 → PASS, L4 → PASS
- Add v3.1 entry to version table

---

## Execution Order

```
Phase 1 (HIGH):  Fixes 1, 3, 4, 6, 7, 9     → 6 fixes
Phase 2 (MEDIUM): Fixes 10-17                 → 8 fixes  (7 findings + fix 17 already counted)
Phase 3 (LOW):   Fixes 18-22                  → 5 fixes
Phase 4:         Audit report update           → 1 document
```

## New Files Created

| File | Purpose |
|------|---------|
| `src/core/input_sanitizer.py` | Prompt sanitization + data/instruction separation |
| `src/agents/whisper/integrity.py` | Smith mutual validation / integrity checker |
| `src/agents/smith/instruction_integrity.py` | S3 instruction integrity validator |
| `src/web/routes/approvals.py` | Human-in-the-loop approval API |

## Testing Strategy

- Run existing test suite after each phase to verify no regressions
- Add targeted tests for each new security feature
- Run Bandit security scanner after all changes
- Validate all 18 fixes against the original audit findings

## Estimated Scope

- **Phase 1:** 6 fixes, ~3 new files, ~600 lines new/changed
- **Phase 2:** 8 fixes, ~1 new file, ~500 lines new/changed
- **Phase 3:** 5 fixes, ~100 lines changed
- **Phase 4:** 1 document update
- **Total:** ~4 new files, ~20 modified files, ~1,200 lines
