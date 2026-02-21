# Vibe-Code Detection Audit v2.0

**Project:** Agent-OS
**Date:** 2026-02-21
**Framework:** [Vibe-Code Detection Audit v1.0](https://github.com/kase1111-hash/Claude-prompts/blob/main/vibe-checkV2.md)
**Auditor:** Claude (automated analysis)

---

## Executive Summary

This audit evaluates whether the Agent-OS codebase was AI-generated without meaningful human review ("vibe-coded"). The analysis spans three weighted domains: Surface Provenance (20%), Behavioral Integrity (50%), and Interface Authenticity (30%).

**Overall Vibe-Code Confidence: 40%**
**Classification: Substantially Vibe-Coded**

The codebase was primarily authored by an AI agent (63% of commits attributed to "Claude") with a human-in-the-loop providing architectural direction, issue creation, and PR approval. The interface layer and some core modules show genuine depth, but significant behavioral integrity issues — dead code modules, hardcoded mock data, ghost configuration, and pervasive exception swallowing — indicate insufficient human review of implementation details.

---

## Scoring Summary

| Domain | Weight | Score | Percentage | Rating |
|--------|--------|-------|------------|--------|
| A. Surface Provenance | 20% | 11/21 | 52.4% | Inconclusive |
| B. Behavioral Integrity | 50% | 10/21 | 47.6% | Inconclusive |
| C. Interface Authenticity | 30% | 18/21 | 85.7% | Authentic |

**Weighted Authenticity Score:** (52.4% × 0.20) + (47.6% × 0.50) + (85.7% × 0.30) = **60.0%**
**Vibe-Code Confidence:** 100% - 60.0% = **40.0%**

| Range | Classification |
|-------|---------------|
| 0-15 | Human-authored |
| 16-35 | AI-assisted |
| **36-60** | **Substantially vibe-coded** ← |
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

### B1. Error Handling Authenticity — Weak (1/3)

**Evidence:**
- 30+ instances of bare `except Exception: pass` across the codebase
- Critical paths affected: `src/core/enforcement.py:255,435`, `src/memory/pq_keys.py:1042`, `src/messaging/bus.py:432`, `src/agents/smith/attack_detection/storage.py:1514`
- `src/web/app.py:199-200,387-388,403-404,412-413,421-422`: Health checks silently swallow failures, forcing "down" status with no logging
- `src/web/dependencies.py:173-174`: SQLite permission hardening silently swallows `chmod` failures
- `src/web/routes/chat.py:1031-1037`: Delete operation swallows store failure, falls through to misleading 404

**Assessment:** Error handling is consistently decorative. Exceptions are caught to prevent crashes, not to enable recovery or diagnosis. This is a hallmark of AI-generated code where error handling is "added" as a pattern rather than designed for specific failure modes.

### B2. Configuration Actually Used — Weak (1/3)

**Evidence:**
- `.env.example` defines 8 `AGENT_OS_STT_*` / `AGENT_OS_TTS_*` variables (lines 67-88) — **zero imports** of these variables anywhere in source
- `src/web/routes/system.py:126-169`: 7 system settings (`chat.max_history`, `agents.default_timeout`, `logging.level`, `api.rate_limit`, etc.) can be changed via API but are **never read by any other code** — changing `logging.level` does not change the actual log level
- `AGENT_OS_BOUNDARY_NETWORK_ALLOWED` accessed via raw `os.getenv()` in `chat.py:313` bypassing the config system entirely

**Assessment:** Configuration theater. Multiple config surfaces exist (env vars, API settings) that look functional in the UI but have no backend effect.

### B3. Call Chain Completeness — Weak (1/3)

**Evidence:**
- **Entire `src/kernel/` module is dead code:** 7 files, 10+ classes (`ConversationalKernel`, `EbpfFilter`, `SeccompFilter`, `FuseWrapper`, `FileMonitor`, `PolicyCompiler`, `PolicyInterpreter`, `ContextMemory`, etc.) — zero imports outside the module itself
- `src/web/routes/agents.py:500`: `start_agent`/`stop_agent`/`restart_agent` only change an in-memory enum value — no actual process management
- `src/web/routes/images.py:693-696`: `_generate_comfyui()` sends HTTP POST but returns empty list — response parsing unimplemented
- `src/web/routes/images.py:698-750`: `_generate_a1111()` similarly incomplete

**Assessment:** Multiple major subsystems are either entirely disconnected (kernel) or stub implementations (agent lifecycle, image generation backends). These represent hundreds of lines of code that exist structurally but serve no functional purpose.

### B4. Async Correctness — Weak (1/3)

**Evidence:**
- `src/web/routes/images.py:348,754`: Global mutable `_image_store` and `_generator` accessed from concurrent async handlers without any lock — race condition risk
- `src/web/routes/images.py:736,920`: Synchronous `open()` calls in async handlers block the event loop
- `src/web/routes/chat.py:619`: Uses deprecated `asyncio.get_event_loop()` inside async function
- `src/web/dreaming.py:60`: Uses `threading.Lock` in code called from async handlers — can block event loop under contention

**Assessment:** Async/await is used structurally (functions are marked `async`) but the implementation violates async correctness in multiple ways. This is a classic AI generation pattern: correct syntax, incorrect concurrency semantics.

### B5. State Management Coherence — Moderate (2/3)

**Evidence:**
- DI container (`src/web/dependencies.py`) works correctly with proper singleton lifecycle
- Message bus state management is coherent (subscriptions, channels, dead letter queue)
- **Issues:** Image store not thread-safe, system settings stored but disconnected from consumers, `src/web/routes/system.py:186` hardcodes version `"1.0.0"` while importing `PKG_VERSION` (line 19)

**Assessment:** Core state management (DI, message bus, contract store) is genuine. Peripheral state (images, system settings, metrics) is disconnected or mock.

### B6. Security Implementation Depth — Moderate (2/3)

**Evidence:**
- **Genuine depth:** AES-256-GCM encryption in memory vault, Ed25519 agent identity signing, PBKDF2 with 600K iterations, HMAC-bound sessions
- **Issues:** 4+ different authentication patterns across routes (system.py uses `require_authenticated_user`, intent_log.py uses custom `get_current_user_id()`, chat.py uses `_authenticate_rest_request`, agents.py mixes both)
- `src/web/dependencies.py:173-174`: `_harden_sqlite_path()` silently swallows chmod failures
- No CSRF protection

**Assessment:** Core crypto is real and well-implemented. But the authentication layer has organic inconsistencies that suggest it was built incrementally by AI without a unified design pass.

### B7. Resource Management — Moderate (2/3)

**Evidence:**
- Redis bus has proper connection handling patterns
- Memory vault properly manages encryption contexts
- **Issues:** Pervasive `except Exception: pass` prevents proper resource cleanup on failure paths, no connection pooling evidence for HTTP clients, background tasks in image generation lack cleanup

**Assessment:** Resource management is adequate in core modules but undermined by exception swallowing in peripheral code.

### Domain B Total: 10/21 (47.6%) — Inconclusive

### Most Critical Clusters:

1. **Dead code modules** (B3): The entire `src/kernel/` package (7 files, 10+ classes) is never imported. Hundreds of lines with zero consumers.
2. **Fabricated system data** (B3, B5): System status, metrics, logs, and version endpoints return hardcoded mock values that never reflect actual state.
3. **Exception swallowing** (B1): 30+ instances of `except Exception: pass` in security-critical paths including enforcement, encryption, and attack detection.
4. **Ghost configuration** (B2): STT/TTS env vars, system settings API — functional-looking surfaces with no backend wiring.

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

### What's Genuine

- **Cryptographic infrastructure** is real and well-implemented (AES-256-GCM, Ed25519, PBKDF2 600K, post-quantum key support)
- **Constitutional enforcement engine** with legitimate 3-tier evaluation (structural → semantic → LLM judge)
- **Memory vault** with proper encryption, consent tracking, and quarantine levels
- **Learning contracts system** with domain checking, consent prompts, and enforcement hooks
- **Web interface** is a functional SPA, not a decorative shell
- **Test suite** has areas of genuine depth (state permutations, red-team injection tests)

### What's Vibe-Coded

- **`src/kernel/` module** — entirely dead code, never imported or used
- **System monitoring endpoints** — return hardcoded fake data
- **Agent lifecycle management** — toggles an enum, doesn't manage processes
- **Image generation backends** — send HTTP requests, return empty results
- **Configuration surfaces** — STT/TTS vars and settings API have no backend effect
- **Error handling** — 30+ silent exception swallows across security-critical paths

### Development Pattern

The evidence consistently points to an **AI-writes, human-approves** development model:

1. Human creates GitHub issues defining features
2. AI (Claude) implements features on session-ID-named branches
3. AI self-reviews and fixes its own code
4. Human merges PRs

This pattern produces code that is **structurally sophisticated** (the AI understands architecture) but **behaviorally incomplete** (the AI generates plausible implementations that aren't always connected or functional). The human review appears focused on high-level correctness rather than tracing execution paths to verify behavioral integrity.

---

## Final Score

```
Domain A (Surface Provenance):    11/21 = 52.4%  ×  0.20  =  10.48
Domain B (Behavioral Integrity):  10/21 = 47.6%  ×  0.50  =  23.80
Domain C (Interface Authenticity): 18/21 = 85.7%  ×  0.30  =  25.71
                                                    ─────────────────
Weighted Authenticity Score:                                  60.0%
Vibe-Code Confidence:              100% - 60.0%  =           40.0%

Classification: SUBSTANTIALLY VIBE-CODED (36-60 range)
```

The codebase sits at the lower boundary of "Substantially Vibe-Coded" — it has more genuine engineering than a typical vibe-coded project (especially in crypto, enforcement, and UI), but the behavioral integrity gaps (dead modules, mock data, ghost config, exception theater) push it firmly out of the "AI-assisted" category.

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
