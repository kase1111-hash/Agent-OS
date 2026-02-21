# Vibe Check Remediation Plan

## Implementation plan for all findings from SECURITY_AUDIT_VIBE_CHECK.md

10 findings total: 3 HIGH, 4 MEDIUM, 3 LOW. Organized into 8 phases.

---

## Phase 1: Fix Import Bug Blocking Web Tests

**Finding:** MEDIUM — Test Failures in Web Module (67 Errors, 22 Failures)

**Root cause:** `src/memory/storage.py:63` uses `SourceTrustLevel` before it is
defined at line 66. This breaks the import chain:
`routes/__init__.py` → `routes/memory.py` → `seshat/agent.py` → `memory/consent.py`
→ `memory/index.py` → `memory/storage.py` → **NameError**.

All 67 test errors share this single root cause.

**Changes:**

| File | Change |
|------|--------|
| `src/memory/storage.py` | Move `class SourceTrustLevel(Enum)` (lines 66–73) **above** the `QUARANTINE_TRUST_LEVELS` constant (line 63). The constant references enum members that must already exist. |

**Verification:** `python -m pytest tests/test_web.py -x -q` — errors should drop from 67 to 0.

---

## Phase 2: Add Authentication to Unprotected Routes

**Finding:** HIGH — 36 Web Endpoints Missing Authentication Guards

**Analysis from route-by-route audit:**

| Route file | Unprotected endpoints | Fix |
|------------|----------------------|-----|
| `routes/security.py` | ALL 15 endpoints (attacks, recommendations, pipeline, patterns) | Add `Depends(require_admin_user)` — security operations are admin-only |
| `routes/system.py` | 5 endpoints: `/info`, `/status`, `/health`, `/version`, `/dreaming` | Add `Depends(require_authenticated_user)` to `/info`, `/status`, `/dreaming`. Keep `/health` and `/version` public (needed by monitoring). |
| `routes/agents.py` | 4 endpoints: `GET /`, `GET /{agent_name}`, `GET /{agent_name}/metrics`, `GET /stats/overview` | Add `Depends(require_authenticated_user)` — agent info should not be public |
| `routes/contracts.py` | 3 endpoints: `GET /types`, `GET /templates`, `GET /templates/{template_id}` | Add `Depends(require_authenticated_user)` — contract templates are not public |
| `routes/intent_log.py` | 1 endpoint: `GET /types` | Add `Depends(require_authenticated_user)` |

**Routes already properly authenticated (no changes needed):**
- `routes/constitution.py` — all 8 endpoints use auth
- `routes/images.py` — all REST endpoints use auth (WebSocket has its own auth)
- `routes/chat.py` — all endpoints use `_authenticate_rest_request` or `_authenticate_websocket`
- `routes/memory.py` — all endpoints use `get_current_user_id`
- `routes/auth.py` — properly uses optional/required auth as appropriate

**Changes per file:**

### `src/web/routes/security.py`

1. Add import: `from ..auth_helpers import require_admin_user`
2. Add `admin_id: str = Depends(require_admin_user)` parameter to every endpoint
3. Also add missing `Request`, `Cookie`, `Depends` to the fastapi import

### `src/web/routes/system.py`

1. Already imports `require_admin_user` — also import `require_authenticated_user`
2. Add `user_id: str = Depends(require_authenticated_user)` to `/info`, `/status`, `/dreaming`
3. Leave `/health` and `/version` unauthenticated (monitoring probes need them)

### `src/web/routes/agents.py`

1. Already imports `require_admin_user` — also import `require_authenticated_user`
2. Add `user_id: str = Depends(require_authenticated_user)` to the 4 read-only endpoints

### `src/web/routes/contracts.py`

1. Add import: `from ..auth_helpers import require_authenticated_user`
2. Add `user_id: str = Depends(require_authenticated_user)` to `/types`, `/templates`, `/templates/{template_id}`

### `src/web/routes/intent_log.py`

1. Add import: `from ..auth_helpers import require_authenticated_user`
2. Add `user_id: str = Depends(require_authenticated_user)` to `GET /types`

**Verification:** Add integration tests in `tests/test_web.py` that call each
previously-unprotected endpoint without auth and assert 401.

---

## Phase 3: Agent Permission Model — Default-Deny

**Finding:** HIGH — Default-ALLOW Agent Permission Model

**Changes:**

| File | Change |
|------|--------|
| `src/agents/config.py:209` | Change `can_escalate: bool = True` → `can_escalate: bool = False` |
| `src/agents/config.py:279` | Change `data.get("can_escalate", True)` → `data.get("can_escalate", False)` |
| `agents/*/config.yaml` | Any agent YAML that needs escalation must now explicitly set `can_escalate: true` (only Whisper orchestrator and Smith guardian should have this) |

**Verification:** Run `python -m pytest tests/test_agent_config.py tests/test_agent_interface.py -v`
and update any tests that assumed `can_escalate=True` was the default.

---

## Phase 4: Wire Agent Identity Signing Into Message Bus

**Finding:** HIGH — Agent Identity Signing Not Integrated Into Message Bus

**Current state:** `src/messaging/bus.py` already has `_sign_message()` and
`_verify_message_signature()` methods that call the identity registry. The
`AgentIdentityRegistry` in `src/agents/identity.py` is fully implemented with
Ed25519 signing.

**The gap:** The `InMemoryMessageBus` constructor accepts an optional
`identity_registry` parameter. If not provided, signing/verification is silently
skipped. The application startup in `src/web/app.py` and agent initialization
paths do not create or inject the registry.

**Changes:**

| File | Change |
|------|--------|
| `src/messaging/bus.py` | Make `identity_registry` a **required** parameter (not Optional). If not provided, raise `ValueError`. Remove silent skip when registry is None — signing must always occur. |
| `src/web/dependencies.py` | In the dependency injection container, create an `AgentIdentityRegistry` singleton and inject it when constructing the `MessageBus`. |
| `src/messaging/bus.py` | In `_verify_message_signature()`, change behavior on verification failure from logging a warning to **rejecting the message** (raise or return error). |

**Verification:** `python -m pytest tests/test_messaging_bus.py -v` — update
tests to always provide an identity registry. Add test that unsigned messages are
rejected.

---

## Phase 5: Harden Enforcement Engine

**Finding:** MEDIUM — LLM Judge Can Override Structural Denial

**Current flow in `src/core/enforcement.py`:**
1. Tier 1 structural check returns `definitive=False` → falls through
2. Tier 2 semantic matching finds no matches → **auto-allows at line 651–658**
3. Request never reaches Tier 3 LLM judge

**Changes:**

| File | Line | Change |
|------|------|--------|
| `src/core/enforcement.py` | ~651–658 | When Tier 2 returns no semantic matches, do NOT auto-allow. Instead, fall through to Tier 3 LLM judge (or the keyword fallback). The absence of matching rules does not mean the request is safe — it means the rules don't cover this case, which should trigger conservative review. |
| `src/core/enforcement.py` | StructuralChecker | Ensure all DENY_PATTERN matches return `definitive=True` (they already do — verify). |
| `src/core/enforcement.py` | LLMJudge.judge() ~line 453 | When `matched_rules` is empty, do NOT auto-allow. Instead, return `allowed=False` with `reason="No applicable rules found — conservative denial"` and set `escalate_to_human=True`. |

**Verification:** `python -m pytest tests/test_enforcement.py -v` — add test that
verifies an unknown request type (no matching rules) is denied, not allowed.

---

## Phase 6: Network Default + Dependency Pinning + CORS Env Var

**Finding:** MEDIUM — Network Access Enabled by Default
**Finding:** MEDIUM — Floating Dependency Versions
**Finding:** LOW — CORS Origins Configured for Localhost Only

**Changes:**

### Network default

| File | Change |
|------|--------|
| `.env.example:96` | Change `AGENT_OS_BOUNDARY_NETWORK_ALLOWED=true` → `AGENT_OS_BOUNDARY_NETWORK_ALLOWED=false` |

### Dependency pinning

| File | Change |
|------|--------|
| `pyproject.toml` | Add upper bounds to all dependencies. Examples: `PyYAML>=6.0.2,<7.0`, `fastapi>=0.115,<1.0`, `cryptography>=42.0,<44.0`, `pydantic>=2.10,<3.0`, `httpx>=0.28,<1.0`, `uvicorn[standard]>=0.32,<1.0`, `jinja2>=3.1,<4.0`, `websockets>=14.0,<15.0`, `defusedxml>=0.7,<1.0`, `python-multipart>=0.0.17,<1.0`, `watchdog>=5.0,<6.0` |
| `requirements.txt` | Apply matching upper bounds |

### CORS env var

| File | Change |
|------|--------|
| `src/web/config.py` | In `from_env()`, add parsing: `cors_origins` from `AGENT_OS_CORS_ORIGINS` env var (comma-separated). Default remains `["http://localhost:8080"]`. |
| `.env.example` | Add commented `AGENT_OS_CORS_ORIGINS=http://localhost:8080` with documentation |

**Verification:** `pip install -e .` still works. Run test suite to confirm no
breakage from pinned versions.

---

## Phase 7: CI/CD Hardening

**Finding:** LOW — CI Security Scans Use continue-on-error
**Finding:** LOW — GitHub Actions Not Pinned to SHA

**Changes:**

### Make security scans blocking

| File | Line | Change |
|------|------|--------|
| `.github/workflows/ci.yml:186` | `bandit -r src/ ... || true` | Remove `|| true` |
| `.github/workflows/ci.yml:189` | `safety check || true` + `continue-on-error: true` | Remove both |
| `.github/workflows/security.yml:128` | Trivy `exit-code: '0'` | Change to `exit-code: '1'` |
| `.github/workflows/security.yml:139` | Trivy `continue-on-error: true` | Remove |
| `.github/workflows/security.yml:166` | License check `continue-on-error: true` | Remove |

### Align action versions and pin to SHA

| File | Change |
|------|--------|
| `.github/workflows/security.yml` | Update `actions/checkout@v4` → `@v6`, `actions/setup-python@v5` → `@v6`, `actions/upload-artifact@v4` → `@v6` to match ci.yml |
| `.github/workflows/security.yml` | Pin `aquasecurity/trivy-action@master` to a tagged release (e.g., `@0.28.0` or specific SHA) |

**Note:** Full SHA pinning for all actions is ideal but could be done in a follow-up
since Dependabot already tracks action updates.

---

## Phase 8: File System Sandboxing Improvements

**Finding:** MEDIUM — Agent File System Access Not Sandboxed

**Current state:** `src/tools/executor.py` and `src/agents/isolation.py` already
have sandbox configurations with filesystem=read-only and network=disabled
defaults. The infrastructure exists.

**Changes:**

| File | Change |
|------|--------|
| `src/agents/config.py` | Add `allowed_paths: List[str] = field(default_factory=list)` to `AgentConfig`. Each agent declares which directories it may access. |
| `src/tools/executor.py` | Before subprocess execution, validate that any file paths in tool parameters fall within the agent's `allowed_paths`. Reject requests targeting paths outside the allowlist. |
| `agents/seshat/config.yaml` | Set `allowed_paths: ["/app/data/memory"]` — seshat only needs memory vault |
| `agents/sage/config.yaml` | Set `allowed_paths: []` — sage does not need file access |

**Verification:** Add test in `tests/test_tools.py` that a tool invocation with a
path outside `allowed_paths` is rejected.

---

## Execution Order

Phases are ordered by dependency and impact:

```
Phase 1 (import bug)     — Unblocks Phase 2 testing
Phase 2 (route auth)     — Fixes the highest-impact vulnerability
Phase 3 (deny-default)   — Reduces agent attack surface
Phase 4 (bus signing)    — Closes agent spoofing gap
Phase 5 (enforcement)    — Hardens constitutional enforcement
Phase 6 (deps/network)   — Supply chain and config hardening
Phase 7 (CI/CD)          — Pipeline hardening
Phase 8 (fs sandbox)     — Defense-in-depth for tool execution
```

Phases 1–5 address all HIGH and the most impactful MEDIUM findings.
Phases 6–8 address remaining MEDIUM and LOW findings.

---

## Out of Scope (Long-term / Requires Design Discussion)

These items from the audit are deferred for future work:

- **Linux namespace/seccomp isolation** for subprocess execution
- **Mutual TLS** for networked agent communication
- **Formal threat model** document for agent permission system
- **Per-agent network allowlists** (beyond default-deny)
- **External human security audit** engagement
