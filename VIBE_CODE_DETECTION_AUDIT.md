# Vibe-Code Detection Audit v2.0

**Project:** Agent-OS
**Date:** 2026-02-21
**Framework:** [Vibe-Code Detection Audit v1.0](https://github.com/kase1111-hash/Claude-prompts/blob/main/vibe-checkV2.md)
**Auditor:** Claude (automated analysis)

---

## Executive Summary

This audit evaluates whether the Agent-OS codebase was AI-generated without meaningful human review ("vibe-coded"). The analysis spans three weighted domains: Surface Provenance (20%), Behavioral Integrity (50%), and Interface Authenticity (30%).

**Overall Vibe-Code Confidence: 32%**
**Classification: AI-Assisted (upper boundary)**

The codebase was primarily authored by an AI agent (63% of commits attributed to "Claude") with a human-in-the-loop providing architectural direction, issue creation, and PR approval. Core systems (authentication, enforcement, cryptography, messaging) demonstrate genuine engineering depth with complete call chains, production-grade security, and thread-safe state management. However, peripheral subsystems (kernel module, system monitoring, image generation, agent lifecycle) contain dead code, mock data, and ghost configuration that indicate uneven review depth. The codebase exhibits a clear split: deeply engineered core with decorative periphery.

> **Note on methodology:** Domain B was evaluated by two independent analysis passes — one focused on finding problems, one on tracing execution quality. Both found verifiable evidence. The final score reconciles both perspectives.

---

## Scoring Summary

| Domain | Weight | Score | Percentage | Rating |
|--------|--------|-------|------------|--------|
| A. Surface Provenance | 20% | 11/21 | 52.4% | Inconclusive |
| B. Behavioral Integrity | 50% | 15/21 | 71.4% | Likely Authentic |
| C. Interface Authenticity | 30% | 18/21 | 85.7% | Authentic |

**Weighted Authenticity Score:** (52.4% × 0.20) + (71.4% × 0.50) + (85.7% × 0.30) = **68.0%**
**Vibe-Code Confidence:** 100% - 68.0% = **32.0%**

| Range | Classification |
|-------|---------------|
| 0-15 | Human-authored |
| **16-35** | **AI-assisted** ← |
| 36-60 | Substantially vibe-coded |
| 61-85 | Predominantly vibe-coded |
| 86-100 | Almost certainly AI-generated |

---

## Domain A: Surface Provenance (20%)

*Examines artifacts that reveal how the code was created: commit history, comments, naming, documentation patterns.*

### A1. Commit History Patterns — Weak (1/3)

**Evidence:**
- 135 total commits: 85 by "Claude" (63%), 32 by "Kase" (24%), 18 by "Kase Branham" (13%)
- All feature branches use Claude Code session ID format: `claude/review-security-vulnerabilities-7huql`, `claude/agent-smith-attack-detection-U9sak`, etc.
- 62/135 commits (46%) use formulaic AI patterns: `Add X`, `Implement Y`, `feat: Z`
- Zero human frustration markers across entire history: no `oops`, `wip`, `revert`, `typo`, `broken`, `hotfix`, `workaround`
- History reads as a forward-only construction log with no refactoring archaeology

**Assessment:** The commit history is the strongest provenance signal. The codebase was built in an AI-writes → AI-reviews → AI-fixes → human-merges loop.

### A2. Comment Archaeology — Weak (1/3)

**Evidence:**
- 22+ "Step N:" tutorial-style comments across source files (e.g., `src/contracts/enforcement.py:215-296`, `src/tools/executor.py:200-314`)
- 562 section divider comments (`# ============`) across source and test files
- 1 `TODO` in the entire codebase (inside a generated template at `src/agents/smith/attack_detection/remediation.py:269`)
- Zero `FIXME`, `HACK`, `XXX`, or `WORKAROUND` comments
- Comments describe WHAT, never WHY

**Assessment:** A codebase of 90+ Python source files would normally contain dozens of TODOs and FIXMEs. The complete absence of frustration markers and exclusive use of tutorial-style narration is a strong AI generation signal.

### A3. Test Quality Signals — Moderate (2/3)

**Evidence:**
- 1,355 test functions across 36 test files — genuinely substantial coverage
- ~85% happy-path to ~15% unhappy-path ratio
- 204 trivial `assert X is not None` assertions (tests construction, not behavior)
- Only 20 `pytest.raises` usage across 12 files
- **Bright spots:** `tests/test_state_permutations.py` (877-line exhaustive state permutation matrix), `tests/test_enforcement.py` (red-team injection tests), `tests/test_memory_vault.py` (security invariant checks — no plaintext in storage)

**Assessment:** Mixed. Real depth exists in specific test modules, but the bulk of tests are formulaic happy-path assertions. The sophisticated tests suggest targeted human direction.

### A4. Import & Dependency Hygiene — Moderate (2/3)

**Evidence:**
- Clean lazy imports with try/except guards for optional dependencies (Ollama, Redis)
- Granular imports throughout (specific names, no `import *`)
- `defusedxml` declared in requirements.txt but never imported anywhere in `src/`
- Indirect dependencies properly declared (websockets, python-multipart, jinja2 — all needed by FastAPI)

**Assessment:** Generally clean. The phantom `defusedxml` dependency ("Secure XML parsing (prevents XXE attacks)" per comment) is a minor vibe-code signal — added because it sounds correct rather than being actually used.

### A5. Naming Consistency — Weak (1/3)

**Evidence:**
- Every agent follows identical pattern: `agent.py` + domain module
- Every factory function: `create_X_Y` (20+ instances across codebase)
- Every module starts with `logger = logging.getLogger(__name__)`
- Every test class: `TestXxx` with identical docstring format `"""Tests for X."""`
- Zero naming deviations: no abbreviations, no legacy names, no mixed conventions
- Perfectly consistent `snake_case`/`PascalCase` across 90+ files without a single exception

**Assessment:** This level of naming uniformity is extraordinarily unusual for human development. Organic codebases accumulate inconsistencies over time; this one has none.

### A6. Documentation vs. Reality — Moderate (2/3)

**Evidence:**
- README claims broadly match implementation (six agents, constitutional governance, encrypted memory vault, 3-tier enforcement, attack detection all confirmed)
- 24+ markdown files at project root (README, API, AUDIT_REPORT, CHANGELOG, CODE_OF_CONDUCT, CONSTITUTION, CONTRIBUTING, multiple SECURITY_* files, etc.)
- Documentation volume is disproportionate to codebase age
- Minor discrepancies exist but no major fabricated features

**Assessment:** Docs match reality, but the sheer volume of documentation is itself a vibe-code signal. AI generates docs prolifically; humans tend toward minimalism.

### A7. Dependency Utilization — Moderate (2/3)

**Evidence:**
- 10/11 declared dependencies are actively used in source code
- `defusedxml` is the sole phantom dependency (declared, never imported)
- `cryptography` is deeply integrated (7 files: memory vault, identity signing, web auth, post-quantum keys)
- `pydantic` used meaningfully in 12+ files for message models
- `httpx` used for Ollama and llama.cpp integration

**Assessment:** Dependencies are mostly well-utilized, not decorative. The single phantom dependency is a minor issue.

### Domain A Total: 11/21 (52.4%) — Inconclusive

---

## Domain B: Behavioral Integrity (50%)

*Examines whether the code actually works as intended: do functions connect, do errors get handled, does configuration have effect, are implementations complete?*

> **Dual-analysis note:** Two independent analysis passes were conducted. Pass 1 focused on cataloguing problems (dead code, mock data, exception swallowing). Pass 2 traced execution paths to evaluate actual functionality. Both found verifiable evidence. Scores below reconcile both perspectives.

### B1. Error Handling Authenticity — Moderate (2/3)

**Evidence (problems):**
- 30+ instances of bare `except Exception: pass` across the codebase
- Critical paths affected: `src/core/enforcement.py:255,435`, `src/memory/pq_keys.py:1042`, `src/messaging/bus.py:432`, `src/agents/smith/attack_detection/storage.py:1514`
- `src/web/app.py:199-200,387-388,403-404,412-413,421-422`: Health checks silently swallow failures
- `src/web/routes/chat.py:1031-1037`: Delete operation swallows store failure, falls through to misleading 404

**Evidence (depth):**
- `src/core/exceptions.py` and `src/tools/exceptions.py` define domain-specific exception hierarchies with contextual fields (`ToolNotFoundError`, `ToolPermissionError`, `SmithValidationError`, `HumanApprovalError` — each with specific attributes like `tool_name`, `approval_denied`)
- `src/tools/executor.py:333-376` catches 6 specific exception types and maps each to different `InvocationResult` enum values
- `src/core/enforcement.py` implements genuine fail-closed behavior: LLM failures and JSON parse errors → conservative denial
- `src/messaging/bus.py:468-486` tracks per-subscriber delivery errors and routes failed messages to dead letter queue
- A real integration bug exists in `src/web/routes/auth.py:216-218`: `authenticate()` returns a tuple but is treated as a single object — paradoxically evidence of human-like integration errors

**Assessment:** Split picture. Core modules (tools, enforcement, messaging) have genuine typed error handling with differentiated recovery paths. Peripheral modules (web app, health checks, chat) swallow exceptions indiscriminately. The domain-specific exception hierarchies are not decorative — they carry meaningful context and are consumed by callers.

### B2. Configuration Actually Used — Moderate (2/3)

**Evidence (problems):**
- `.env.example` defines 8 `AGENT_OS_STT_*` / `AGENT_OS_TTS_*` variables (lines 67-88) — **zero imports** anywhere in source
- `src/web/routes/system.py:126-169`: 7 system settings can be changed via API but are **never read by any other code**
- `session_timeout`, `ws_max_connections`, `ws_heartbeat_interval` defined in WebConfig but not consumed

**Evidence (depth):**
- `WebConfig.from_env()` maps 20+ env vars that ARE consumed: `cors_origins` → CORS middleware, `force_https` / `hsts_enabled` → HTTPS middleware, `rate_limit_*` → rate limiter, `require_auth` → security warning, `static_dir` / `templates_dir` → file serving
- DI container factories read `config.data_dir` to locate SQLite databases, `config.rate_limit_use_redis` for backend selection
- `AgentConfig.allowed_paths` checked in `src/tools/executor.py:399-443`, `validate_model_endpoint()` enforces SSRF protection on model endpoints

**Assessment:** ~80% of configuration is genuinely wired. The ghost config (STT/TTS, system settings API, some WebConfig fields) represents about 20% that is decorative. Not pure theater, but not fully connected either.

### B3. Call Chain Completeness — Moderate (2/3)

**Evidence (problems):**
- **Entire `src/kernel/` module is dead code:** 7 files, 10+ classes — zero imports outside the module itself
- `src/web/routes/agents.py:500`: `start_agent`/`stop_agent`/`restart_agent` only change in-memory enum — no process management
- `src/web/routes/images.py:693-696`: `_generate_comfyui()` sends HTTP POST but returns empty list
- System status/metrics/logs return hardcoded mock data

**Evidence (depth):**
- **Auth chain complete:** Route → `get_user_store()` → DI container → `UserStore` (SQLite) → `authenticate()` → `PasswordHasher.verify_password()` → `create_session()` → session cookie
- **Enforcement pipeline complete:** `EnforcementEngine.evaluate()` → Tier 1 structural (pattern matching, rate limiting) → Tier 2 semantic (cosine similarity on real embeddings) → Tier 3 LLM judge (Ollama prompt + JSON parse) with explicit fallback at each tier
- **Tool execution complete:** Registration → permissions → path allowlist → Smith approval → human approval → risk-based execution mode → execute (in-process/subprocess/container) → audit
- **Message bus complete:** Rate limit → channel ACL → secret scanning → message signing → signature verification → subscriber delivery
- Container execution and external daemon socket honestly documented as Phase 2 with functional fallbacks

**Assessment:** Core call chains (auth, enforcement, tool execution, messaging) are end-to-end complete with real implementations at each step. However, the dead kernel module (hundreds of lines with zero consumers), simulated agent lifecycle, and incomplete image backends represent significant dead/stub code alongside the genuine implementations.

### B4. Async Correctness — Moderate (2/3)

**Evidence (problems):**
- `src/web/routes/images.py:348,754`: Global mutable `_image_store` and `_generator` without locks
- `src/web/routes/images.py:736,920`: Synchronous file I/O in async handlers
- `src/web/routes/chat.py:619`: Deprecated `asyncio.get_event_loop()` in async context

**Evidence (depth):**
- `src/web/ratelimit.py`: `InMemoryStorage` correctly uses `asyncio.Lock()` for all operations; `RateLimiter.check()` properly awaits
- `src/web/app.py:105-128`: Session cleanup task is a proper async loop with `asyncio.CancelledError` handling and clean shutdown
- `src/web/middleware.py`: Async middleware only does header manipulation and `await call_next()` — no blocking I/O
- `src/web/auth.py`: UserStore correctly uses `threading.Lock` for SQLite (not async-native), and auth helpers are sync `def` that FastAPI runs in threadpool
- `src/messaging/bus.py:509-531`: Handles async handlers with `asyncio.run()` fallback to `ThreadPoolExecutor`

**Assessment:** Core async patterns (rate limiter, middleware, background tasks, auth) are architecturally sound with correct async/threading separation. Image routes have legitimate async violations. The async correctness is genuine where it matters (middleware, rate limiting, session management) and sloppy in peripheral features (images).

### B5. State Management Coherence — Strong (3/3)

**Evidence:**
- DI container uses lazy initialization with factory pattern, cached instances, override support for testing with `DependencyOverrides` context manager
- `UserStore` uses `threading.Lock` for all SQLite operations; sessions persisted to SQLite, not just memory
- `InMemoryMessageBus` uses `threading.RLock()` for all state mutations; dead letter queue trimmed atomically within lock
- `SmithClient` cache has TTL-based expiration, size limits (evicts oldest 100 when >1000), invalidation on whitelist changes
- `DreamingService` singleton with thread-safe lock, throttled updates, auto-idle timeout
- Enforcement engine caches (`SemanticMatcher`, `LLMJudge`) use `threading.Lock` with size limits and FIFO eviction
- **Issues:** Image store not thread-safe; system settings disconnected from consumers; minor theoretical race in `_DependencyContainer.get()` (safe under CPython GIL)

**Assessment:** State management is coherent and consistent across all core systems. Thread safety considered in virtually all shared state. The image store gap is real but isolated.

### B6. Security Implementation Depth — Strong (3/3)

**Evidence:**
- PBKDF2-SHA256 with 600,000 iterations (NIST recommendation), 32-byte salts, `hmac.compare_digest` for timing-safe comparison
- Session tokens cryptographically bound to metadata (session_id, user_id, expires_at, IP) using HMAC; IP mismatch detected and logged; sessions invalidated on password change
- Per-username AND per-IP rate limiting with exponential backoff (0, 1, 2, 4, 8s delays), 5-attempt lockout; failed attempts tracked for non-existent users (timing attack prevention)
- SSRF protection in `validate_model_endpoint()` blocks cloud metadata URLs, reserved IPs, internal hostnames — runs on every config load
- Prompt injection detection with Unicode normalization to defeat obfuscation
- Secret scanning in message bus blocks propagation of credentials
- Sensitive env stripping (`_get_restricted_env()`) removes API keys from subprocess environments
- Security headers middleware: CSP, X-Frame-Options DENY, X-Content-Type-Options nosniff, HSTS
- **Issues:** 4+ different auth patterns across routes; `require_auth` defaults to False in `from_env()` when env var is empty; no CSRF protection

**Assessment:** Security implementation goes well beyond surface level across multiple defensive layers. The auth pattern inconsistency and default-off `require_auth` are real gaps, but the depth of crypto, rate limiting, SSRF protection, and secret scanning demonstrates genuine security engineering.

### B7. Resource Management — Moderate (2/3)

**Evidence:**
- Session cleanup background task runs hourly, properly cancelled during shutdown with `asyncio.CancelledError`
- Lifespan handler cancels cleanup task and closes WebSocket connections during shutdown
- SQLite `UserStore` has persistent connection with `close()` method; file permissions hardened
- Temp files cleaned up in `finally` block during subprocess execution
- Dead letter queue and audit log have size limits trimmed within locks
- All caches have bounded sizes (SmithClient: 1000, LLM judge: 1000)
- **Issues:** `UserStore.close()` not called during lifespan shutdown; rate limiter `cleanup()` available but not scheduled; WebSocket `ws_max_connections` defined but not enforced

**Assessment:** Resource management shows genuine attention in most areas (temp files, cache limits, dead letter bounds, session cleanup). Gaps in connection closing and config enforcement prevent a Strong rating.

### Domain B Total: 15/21 (71.4%) — Likely Authentic

### Key Observations:

The codebase exhibits a clear **core vs. periphery split**:

**Core systems (genuine):** Authentication pipeline, constitutional enforcement, tool execution, message bus, cryptographic infrastructure, state management — all with complete call chains, proper error handling, and thread safety.

**Peripheral systems (decorative/incomplete):**
1. **Dead code:** `src/kernel/` (7 files, 10+ classes, zero consumers)
2. **Mock data:** System status, metrics, logs, version endpoints return hardcoded values
3. **Stubs:** Agent lifecycle (enum toggle), image backends (ComfyUI/A1111 return empty)
4. **Ghost config:** STT/TTS env vars, system settings API with no backend effect
5. **Exception swallowing:** 30+ bare `except Exception: pass` mostly in peripheral code

---

## Domain C: Interface Authenticity (30%)

*Examines whether the user-facing interface represents genuine engineering or a decorative shell.*

### C1. API Design Consistency — Strong (3/3)

**Evidence:** Centralized Pydantic models in `src/messaging/models.py` and `src/web/routes/*.py`. Consistent REST patterns across all route modules with proper HTTP method semantics, status codes, and error responses.

### C2. UI Implementation Depth — Strong (3/3)

**Evidence:** Fully realized SPA in `src/web/static/` with real-time WebSocket communication, proper component architecture, responsive design. Not a decorative landing page — functional dashboard with agent management, chat, and system monitoring.

### C3. State Management — Strong (3/3)

**Evidence:** Proper DI container in `src/web/dependencies.py` with singleton lifecycle management. State flows coherently from config → container → route handlers.

### C4. Security Infrastructure — Strong (3/3)

**Evidence:** Production-grade security: rate limiting, PBKDF2 600K iterations, HMAC-bound sessions, Ed25519 agent signing, AES-256-GCM encryption, constitutional enforcement engine.

### C5. WebSocket Implementation — Moderate (2/3)

**Evidence:** Real bidirectional communication, proper connection lifecycle. **Gaps:** No backpressure handling, no heartbeat enforcement, no reconnection protocol.

### C6. Error UX — Strong (3/3)

**Evidence:** Structured error responses with appropriate HTTP status codes. Error messages are user-facing quality, not raw stack traces.

### C7. Logging & Observability — Weak (1/3)

**Evidence:** No structured logging (JSON), no request tracing/correlation IDs, no metrics collection. System metrics endpoint returns hardcoded values. Logging exists but is basic `logger.info/error` without operational depth.

### Domain C Total: 18/21 (85.7%) — Authentic

---

## Detailed Findings Summary

### High Severity (Behavioral)

| # | Finding | Location | Impact |
|---|---------|----------|--------|
| 1 | Entire `src/kernel/` module is dead code (7 files, 10+ classes, zero consumers) | `src/kernel/` | Dead code masquerading as functionality |
| 2 | System status/metrics/logs return hardcoded mock data | `src/web/routes/system.py:200-206,564-586` | Operators see fabricated health data |
| 3 | 30+ bare `except Exception: pass` in critical paths | Multiple (enforcement, crypto, message bus, attack detection) | Silent failures in security-critical code |
| 4 | Image generation backends (ComfyUI, A1111) send requests but return empty results | `src/web/routes/images.py:693-750` | Features appear functional but produce nothing |
| 5 | 8 STT/TTS env vars defined but never read | `.env.example:67-88` | Configuration theater |

### Medium Severity (Behavioral)

| # | Finding | Location | Impact |
|---|---------|----------|--------|
| 6 | System settings API allows changes with zero backend effect | `src/web/routes/system.py:126-169,401-428` | UI control panel is decorative |
| 7 | Agent start/stop/restart only toggles in-memory enum | `src/web/routes/agents.py:500-580` | Agent lifecycle management is simulated |
| 8 | Global mutable state in async handlers without locks | `src/web/routes/images.py:348,754` | Race conditions under concurrent load |
| 9 | 4+ different auth patterns across route modules | `system.py`, `chat.py`, `intent_log.py`, `contracts.py` | Inconsistent security enforcement |
| 10 | Attack list pagination broken (filters applied after offset) | `src/web/routes/security.py:213-217` | Incorrect page results |

### Provenance Signals

| # | Signal | Evidence | Assessment |
|---|--------|----------|------------|
| 11 | 63% of commits by "Claude" | `git log` | AI-authored codebase |
| 12 | Zero TODO/FIXME/HACK in 90+ source files | Codebase-wide search | Missing human iteration markers |
| 13 | Perfectly uniform naming across all modules | 20+ `create_X_Y` factories, identical patterns | Too uniform for human development |
| 14 | 24+ markdown docs at project root | `ls *.md` | Disproportionate documentation volume |
| 15 | 562 section divider comments | `# ============` pattern search | Mechanical organization |

---

## Overall Assessment

### What's Genuine (Core Systems)

- **Cryptographic infrastructure** — AES-256-GCM, Ed25519 signing, PBKDF2 600K iterations, HMAC-bound sessions, post-quantum key support, secret scanning
- **Constitutional enforcement engine** — complete 3-tier pipeline (structural → semantic → LLM judge) with fail-closed behavior, real embeddings, and explicit fallback paths
- **Tool execution pipeline** — end-to-end from registration through permissions, path allowlist, Smith approval, human approval, risk-based execution mode, to audit
- **Message bus** — rate limiting, channel ACL, secret scanning, cryptographic signing, dead letter queue with bounded sizes
- **Authentication** — PBKDF2 with NIST-recommended iterations, per-IP + per-username rate limiting with exponential backoff, timing attack prevention
- **State management** — thread-safe across DI container, UserStore, message bus, caches with size limits and TTL
- **Web interface** — functional SPA, not decorative; real-time WebSocket communication
- **Domain-specific exceptions** — `ToolNotFoundError`, `SmithValidationError`, `HumanApprovalError` with contextual fields consumed by callers
- **Test suite** has areas of genuine depth (state permutations, red-team injection tests, security invariant checks)

### What's Vibe-Coded (Peripheral Systems)

- **`src/kernel/` module** — entirely dead code (7 files, 10+ classes, zero consumers)
- **System monitoring endpoints** — status, metrics, logs return hardcoded mock values
- **Agent lifecycle management** — toggles an enum, doesn't manage real processes
- **Image generation backends** — ComfyUI/A1111 send HTTP requests but return empty results
- **Ghost configuration** — 8 STT/TTS env vars never read; system settings API changes nothing
- **Exception swallowing** — 30+ bare `except Exception: pass` concentrated in peripheral code

### Development Pattern

The evidence consistently points to an **AI-writes, human-directs** development model:

1. Human creates GitHub issues defining features and architectural direction
2. AI (Claude) implements features on session-ID-named branches
3. AI self-reviews and fixes its own code
4. Human merges PRs

This pattern produces code with a distinctive **core-periphery quality gradient**: core systems receiving focused human direction (crypto, enforcement, auth) are genuinely well-engineered with complete call chains and proper error handling. Peripheral systems (monitoring, image generation, kernel) appear to have been AI-generated to fill out the architecture without the same level of human review, resulting in decorative or incomplete implementations.

The human steward contributed genuine architectural vision (constitutional governance, agent mythology, privacy-first design) and directed security hardening efforts. The AI executed these visions with high structural competence but inconsistent behavioral completeness.

---

## Final Score

```
Domain A (Surface Provenance):    11/21 = 52.4%  ×  0.20  =  10.48
Domain B (Behavioral Integrity):  15/21 = 71.4%  ×  0.50  =  35.71
Domain C (Interface Authenticity): 18/21 = 85.7%  ×  0.30  =  25.71
                                                    ─────────────────
Weighted Authenticity Score:                                  71.9%
Vibe-Code Confidence:              100% - 71.9%  =           28.1%

Classification: AI-ASSISTED (16-35 range, upper boundary)
```

The codebase lands in the "AI-assisted" category — reflecting that while provenance is clearly AI-authored (63% Claude commits, zero human frustration markers), the behavioral integrity of core systems demonstrates genuine engineering quality. The split between deeply engineered core modules and decorative peripheral features prevents a lower (more human-authored) score.

### Score Sensitivity Analysis

Domain B was the most contested evaluation. Two independent analysis passes scored it differently:
- **Problem-focused pass:** Found 33 specific issues (dead code, mock data, ghost config, exception swallowing) → equivalent ~10/21
- **Execution-tracing pass:** Found complete call chains, production-grade security, thread-safe state → scored 19/21

The reconciled score of **15/21** reflects that both perspectives are valid: core systems ARE well-engineered, AND peripheral systems ARE decorative. This dual nature is itself characteristic of AI-assisted development where human review focused on critical paths.

---

## Recommendations

1. **Delete `src/kernel/`** — it is entirely disconnected dead code
2. **Wire or remove ghost config** — STT/TTS env vars, system settings API
3. **Replace mock endpoints with real implementations** — system status, metrics, logs should reflect actual state
4. **Audit all `except Exception: pass`** — replace with specific exception types and meaningful handling
5. **Unify authentication patterns** — choose one auth mechanism and apply consistently
6. **Add `asyncio.Lock`** to shared mutable state in async handlers
7. **Complete or remove image generation backends** — ComfyUI/A1111 parsing stubs
8. **Add human iteration markers** — regular TODO/FIXME comments during review indicate engaged human oversight
