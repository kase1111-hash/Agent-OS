# Agent-OS Cleanup Plan

**Based on:** Vibe-Code Detection Audit v2.0 findings
**Principle:** Only change things that are verifiably wrong. Do not change design decisions, intentional placeholders, or Phase 2 stubs.

---

## Category 1: Confirmed Bugs (code does not work as intended)

### 1.1 Auth route tuple unpacking bug
- **File:** `src/web/routes/auth.py:216-218`
- **Bug:** `authenticate()` returns `Tuple[Optional[User], Optional[str]]` but the caller assigns the entire tuple to `user` and checks `if not user`. A tuple is always truthy, so authentication failures are never caught.
- **Fix:** Unpack the tuple: `user, error_message = store.authenticate(...)` and use `error_message` in the 401 response.

### 1.2 `require_auth` default inconsistency
- **File:** `src/web/config.py:48,138`
- **Bug:** Dataclass field defaults to `True`, but `from_env()` evaluates `os.getenv("AGENT_OS_REQUIRE_AUTH", "").lower() in ("1", "true", "yes")` which is `False` when the env var is unset. Auth is silently OFF by default in real deployments.
- **Fix:** Change the `from_env()` default to `True` when env var is not set: `os.getenv("AGENT_OS_REQUIRE_AUTH", "true").lower() in ("1", "true", "yes")`.

### 1.3 Pagination bug in security route
- **File:** `src/web/routes/security.py:213-238`
- **Bug:** Offset is applied before filtering (`attacks[offset:]` then filter), so pages skip pre-filter records. Correct order: filter first, then offset+limit.
- **Fix:** Iterate all attacks with filters, then apply offset/limit to the filtered results.

### 1.4 Hardcoded version instead of PKG_VERSION
- **File:** `src/web/routes/system.py:186`
- **Bug:** `version="1.0.0"` is hardcoded while `PKG_VERSION` is imported (line 19) and equals `"0.1.0"`. The `SystemInfo` model even declares `PKG_VERSION` as its field default (line 60).
- **Fix:** Replace `version="1.0.0"` with `version=PKG_VERSION`.

### 1.5 Deprecated `asyncio.get_event_loop()` in async context
- **File:** `src/web/routes/chat.py:619`
- **Bug:** `asyncio.get_event_loop()` is deprecated in Python 3.10+ and unreliable inside `async def` functions.
- **Fix:** Replace with `asyncio.get_running_loop()`.

### 1.6 UserStore not closed during shutdown
- **File:** `src/web/app.py` (lifespan handler, ~line 162-200)
- **Bug:** The lifespan shutdown cancels the cleanup task and closes WebSocket connections but never calls `UserStore.close()`, leaving the SQLite connection unclosed.
- **Fix:** Add `user_store.close()` call in the shutdown sequence (retrieve via DI container).

---

## Category 2: Dead Code (confirmed zero external consumers)

### 2.1 Remove `src/kernel/` module
- **Files:** 8 files — `__init__.py`, `context.py`, `ebpf.py`, `engine.py`, `fuse.py`, `interpreter.py`, `monitor.py`, `policy.py`, `rules.py`
- **Evidence:** Zero imports from outside the module. Grep for `from kernel`, `import kernel`, and all class names (`ConversationalKernel`, `EbpfFilter`, `SeccompFilter`, `FuseWrapper`, `FileMonitor`, `PolicyCompiler`, `PolicyInterpreter`, `ContextMemory`) returns matches ONLY within `src/kernel/` itself. No test files import from it.
- **Action:** Delete the entire `src/kernel/` directory.

### 2.2 Remove `defusedxml` from requirements.txt
- **File:** `requirements.txt`
- **Evidence:** `defusedxml` is never imported anywhere in `src/`. Comment says "Secure XML parsing (prevents XXE attacks)" but no XML parsing exists.
- **Action:** Remove the `defusedxml` line from requirements.txt.

---

## Category 3: Ghost Configuration (defined but never consumed)

### 3.1 Remove STT/TTS env vars from .env.example
- **File:** `.env.example:67-88`
- **Evidence:** 8 `AGENT_OS_STT_*` / `AGENT_OS_TTS_*` variables defined. Zero imports of these variables anywhere in `src/`.
- **Action:** Remove the STT/TTS section from `.env.example` or add a clear `# NOT YET IMPLEMENTED` header.

### 3.2 Wire `session_timeout` or remove it
- **File:** `src/web/config.py:75` defines `session_timeout: int = 3600`
- **Evidence:** `src/web/routes/auth.py:154` hardcodes `duration_hours=24` for registration; line 222 hardcodes `24 * 30` or `24` for login. The config value is never read.
- **Action:** Replace the hardcoded `duration_hours` values in auth routes with `config.session_timeout` (converted from seconds to hours), OR remove the `session_timeout` field from WebConfig.

### 3.3 Wire or remove `ws_max_connections` / `ws_heartbeat_interval`
- **File:** `src/web/config.py:63-64`
- **Evidence:** Neither value is checked in the WebSocket ConnectionManager or anywhere else.
- **Action:** Enforce `ws_max_connections` in the ConnectionManager's `connect()` method, OR remove these fields from WebConfig.

### 3.4 System settings API stores but never reads
- **File:** `src/web/routes/system.py:126-169`
- **Evidence:** 7 settings (`chat.max_history`, `agents.default_timeout`, `logging.level`, etc.) can be PUT via API but no other code reads them. Changing `logging.level` does not change the actual log level.
- **Action:** Either wire the settings to their consumers (e.g., `logging.level` → `logging.setLevel()`), or add a clear comment/API response indicating these are placeholders. At minimum, do not silently accept changes that have no effect.

---

## Category 4: Concurrency Bugs

### 4.1 Global mutable state without locks in images.py
- **File:** `src/web/routes/images.py:348,754`
- **Bug:** `_image_store = ImageStore()` and `_generator = ImageGenerator()` are module-level globals accessed by concurrent async handlers. `ImageStore` has mutable dicts (`self.jobs`, `self.gallery`) with no synchronization. Zero `Lock`/`RLock`/`asyncio.Lock` in the entire file.
- **Fix:** Add `threading.Lock` to `ImageStore` methods that mutate state (matching the pattern used by `UserStore` in `auth.py`).

### 4.2 Synchronous file I/O in async handlers
- **File:** `src/web/routes/images.py:736,920`
- **Bug:** `open(filepath, "wb")` (line 736) and `open(filepath, "rb")` (line 920) block the event loop.
- **Fix:** Use `aiofiles` for writes, or wrap in `await asyncio.to_thread(...)`. For reads, `StreamingResponse` can use a sync file object — this is actually OK for FastAPI/Starlette (they handle it in a threadpool). The write at line 736 should be moved to a thread.

---

## Category 5: Exception Handling (only clearly wrong cases)

30 bare except/pass blocks were found. I will NOT change all of them — some are legitimate patterns (ImportError guards for optional deps, ProcessLookupError for dead processes, sqlite migration patterns). Only fix the ones that are clearly wrong:

### 5.1 Broad `except Exception: pass` in critical security paths
These swallow unexpected errors silently in security-critical code:

| File | Line | Context | Fix |
|------|------|---------|-----|
| `src/agents/smith/attack_detection/storage.py` | 1514 | Connection close in attack detection | Add `logger.debug()` |
| `src/boundary/daemon/tripwires.py` | 448 | Boundary enforcement check | Narrow to specific exceptions, add `logger.warning()` |
| `src/web/app.py` | 387 | Session validation silently fails | Add `logger.warning()` |
| `src/web/intent_log.py` | 192 | Reading hash from intent log | Narrow to `(KeyError, sqlite3.Error)`, add `logger.warning()` |

### 5.2 Leave alone (legitimate patterns)
- `ImportError: pass` for optional deps (diffusers, keyring, SecretScanner) — correct
- `ProcessLookupError: pass` for "already dead" processes — correct
- `sqlite3.OperationalError: pass` for "column already exists" migrations — correct with existing comments
- `ValueError: pass` in enum parsing loops — acceptable fallthrough pattern
- `UnicodeDecodeError: pass` in vault blob decoding — acceptable fallback to raw bytes
- `OSError: pass` for temp file cleanup in `finally` blocks — acceptable

---

## Category 6: NOT changing (uncertain or intentional)

These were flagged in the audit but I am NOT confident they are wrong:

- **Agent start/stop as enum toggle** — may be intentional Phase 1 design
- **ComfyUI/A1111 returning empty** — may be intentional stubs awaiting implementation
- **System metrics returning mock data** — may be intentional placeholder
- **System health hardcoded to "up"** — may be intentional until real health checks are built
- **4+ auth patterns across routes** — inconsistent but each works; unifying requires architectural decision
- **No CSRF protection** — requires architectural decision about token strategy
- **`threading.Lock` in `dreaming.py`** — used from sync code called by async; not clearly wrong

---

## Execution Order

1. **Category 1 (Bugs)** — highest priority, these cause incorrect behavior
2. **Category 4 (Concurrency)** — race conditions under load
3. **Category 2 (Dead code)** — cleanup, reduces confusion
4. **Category 5 (Exception handling)** — only the 4 clearly-wrong cases
5. **Category 3 (Ghost config)** — cleanup, reduces confusion

---

## Estimated Scope

- **Files modified:** ~12
- **Files deleted:** 8 (entire `src/kernel/` directory)
- **Lines added:** ~30
- **Lines removed:** ~3,000+ (mostly kernel module deletion)
- **Risk level:** Low-Medium (all changes are narrowly scoped bug fixes or dead code removal)
