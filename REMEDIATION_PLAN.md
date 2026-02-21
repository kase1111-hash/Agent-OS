# Remediation Plan — Vibe-Code Audit Findings

**Source:** `VIBE_CHECK_REPORT.md` remediation checklist (8 items)
**Date:** 2026-02-21
**Principle:** Minimal, targeted changes. No over-engineering. Fix what's broken, remove what's dead, wire what's disconnected.

---

## Item 1: Wire system health/metrics to real data

**Priority:** High
**Files:** `src/web/routes/system.py`, `src/web/app.py`

### Current State

| Endpoint | Status |
|----------|--------|
| `GET /health` (lines 233-381) | **Partially real.** Agents, memory, and constitution health checks attempt real imports and measure latency. API and WebSocket components are hardcoded to `"up"` with 1.0ms latency. |
| `GET /status` (lines 196-230) | **Hardcoded.** All 5 components (`api`, `websocket`, `agents`, `memory`, `constitution`) always return `"up"`. |
| `GET /metrics` (lines 562-588) | **Fully hardcoded.** Memory, CPU, request counts, WebSocket message counts are all fake numbers. |
| `GET /logs` (lines 433-474) | **Mock.** Returns generated startup messages. Nothing feeds real log entries into the endpoint. |

### Plan

#### 1a. Add a request counter middleware (~20 lines)

Add a lightweight counter to `src/web/app.py` that tracks total requests, successes, and failures. This feeds real data into `/metrics` and `/status`.

```
File: src/web/app.py

Add to AppState (around line 85):
  request_count: int = 0
  request_errors: int = 0

Add a simple middleware in create_app() (after rate limit middleware):
  @app.middleware("http")
  async def count_requests(request, call_next):
      _app_state.request_count += 1
      response = await call_next(request)
      if response.status_code >= 500:
          _app_state.request_errors += 1
      return response
```

#### 1b. Wire `/status` to real component checks (~15 lines)

Replace the hardcoded component dict in the `/status` endpoint (line 206) with the same real checks used in `/health`. The `/health` endpoint already imports `create_loader()`, `create_seshat_agent()`, `create_kernel()` — reuse those.

```
File: src/web/routes/system.py, lines 196-230

Replace hardcoded components dict with:
  - "api": "up" (always true — we're responding)
  - "websocket": "up" if len(_app_state.active_connections) >= 0 (check import works)
  - "agents": try create_loader(), "up" if success, "down" if ImportError
  - "memory": try create_seshat_agent(), "up" if success, "down" if ImportError
  - "constitution": try create_kernel(), "up" if success, "down" if ImportError
```

#### 1c. Wire `/metrics` to real counters (~25 lines)

Replace hardcoded values in the `/metrics` endpoint (lines 562-588):

| Metric | Current | Replace With |
|--------|---------|-------------|
| `requests.total` | `1542` | `_app_state.request_count` |
| `requests.failed` | `22` | `_app_state.request_errors` |
| `requests.success` | `1520` | `request_count - request_errors` |
| `requests.rate_per_minute` | `25.7` | `request_count / (uptime / 60)` |
| `websocket.active_connections` | `3` | `len(_app_state.active_connections)` |
| `memory.total_mb` | `1024` | Keep as-is (requires `psutil`, not worth adding a dependency) |
| `cpu.usage_percent` | `15.5` | Keep as-is (requires `psutil`) |

#### 1d. Wire `/logs` to real logging (~30 lines)

Add a `logging.Handler` subclass that captures log records into the in-memory `_logs` list (already defined at module level). Install it during app lifespan startup.

```
File: src/web/routes/system.py

class InMemoryLogHandler(logging.Handler):
    def emit(self, record):
        entry = LogEntry(
            timestamp=datetime.fromtimestamp(record.created),
            level=record.levelname,
            message=self.format(record),
            source=record.name,
        )
        _logs.append(entry)
        if len(_logs) > 1000:  # bound the list
            _logs.pop(0)

Install in create_app() lifespan:
    handler = InMemoryLogHandler()
    handler.setLevel(logging.INFO)
    logging.getLogger("src").addHandler(handler)
```

### Estimated Changes
- `src/web/app.py`: +15 lines (counter middleware, AppState fields)
- `src/web/routes/system.py`: +50 lines (handler class), ~30 lines modified (wire real data)

---

## Item 2: Agent lifecycle management

**Priority:** High
**Files:** `src/web/routes/agents.py`

### Current State

Lines 472-587: `start_agent`, `stop_agent`, `restart_agent` endpoints only call `store.update_status(agent_name, AgentStatus.ACTIVE/DISABLED)` — a single enum flag change. No actual agent process is started or stopped.

The `AgentStore` (lines 119-395) wraps the real `AgentRegistry` when available, falling back to mock data. Real agents ARE discoverable from the `agents/` directory via `create_loader()`.

### Plan

**Approach:** Document as Phase 2 rather than implement full process management. The agents in Agent-OS are not long-running daemons — they are request-response modules loaded on demand. "Starting" an agent means making it available for routing; "stopping" means removing it from the routing table. The current enum approach is actually correct for this architecture.

#### 2a. Add honest status responses (~10 lines)

Modify the start/stop/restart endpoints to return a response that reflects reality:

```
File: src/web/routes/agents.py

In start_agent (line 497):
  Change response to include:
    "note": "Agent marked as active for request routing. Agents are loaded on-demand, not as persistent processes."

In stop_agent (line 537):
  Change response to include:
    "note": "Agent marked as disabled. Incoming requests will not be routed to this agent."

In restart_agent (line 575):
  Change response to include:
    "note": "Agent status reset. Cached state cleared."
```

#### 2b. Add cache/state clearing to restart (~15 lines)

The restart endpoint should actually clear any cached state for the agent (if applicable). Check if the agent has a cached instance in the DI container or loader and invalidate it.

```
File: src/web/routes/agents.py, restart_agent()

After status update:
  # Clear any cached agent instances
  try:
      loader = create_loader()
      if hasattr(loader, 'clear_cache'):
          loader.clear_cache(agent_name)
  except ImportError:
      pass
```

### Estimated Changes
- `src/web/routes/agents.py`: ~25 lines modified

---

## Item 3: Image generation backends

**Priority:** High
**Files:** `src/web/routes/images.py`

### Current State

| Backend | Lines | Status |
|---------|-------|--------|
| `_generate_a1111()` | 707-758 | **Working.** Sends POST to A1111 API, parses base64 images from response, decodes and saves to disk. Uses `asyncio.to_thread()` for file I/O. |
| `_generate_comfyui()` | 664-705 | **Broken.** Sends POST with incomplete workflow (only KSampler node, missing checkpoint loader, CLIP encoder, VAE decoder). Returns empty list — comment says `# ... parse ComfyUI output` but no parsing code. |

### Plan

#### 3a. Complete ComfyUI response parsing (~40 lines)

ComfyUI uses a queue-based API. The current code sends a prompt but doesn't poll for results. A real implementation needs:

1. POST the workflow to `/prompt` (already done, line 700)
2. Poll `/history/{prompt_id}` for completion
3. Fetch generated images from `/view?filename=...`

```
File: src/web/routes/images.py, _generate_comfyui(), lines 700-705

Replace the empty parsing block with:
  prompt_id = response.json().get("prompt_id")
  if not prompt_id:
      return []

  # Poll for completion (timeout after 5 minutes)
  import time
  for _ in range(300):
      await asyncio.sleep(1)
      history_resp = await asyncio.to_thread(
          httpx.get, f"{job.api_url}/history/{prompt_id}", timeout=10
      )
      history = history_resp.json()
      if prompt_id in history:
          break
  else:
      return []

  # Extract images from output
  outputs = history[prompt_id].get("outputs", {})
  images = []
  store = get_image_store()
  for node_id, node_output in outputs.items():
      for img_info in node_output.get("images", []):
          filename = img_info["filename"]
          img_resp = await asyncio.to_thread(
              httpx.get,
              f"{job.api_url}/view?filename={filename}",
              timeout=30
          )
          image_id = str(uuid.uuid4())
          out_filename = f"{image_id}.png"
          filepath = store.output_dir / out_filename
          await asyncio.to_thread(filepath.write_bytes, img_resp.content)
          images.append({
              "id": image_id,
              "filename": out_filename,
              "width": job.width,
              "height": job.height,
          })
  return images
```

#### 3b. Complete ComfyUI workflow definition (~20 lines)

The workflow at lines 673-691 only defines a KSampler node. Add the minimum required nodes:

```
Add to workflow dict:
  "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": model_name}}
  "2": {"class_type": "CLIPTextEncode", "inputs": {"text": job.prompt, "clip": ["1", 1]}}
  "3": {"class_type": "CLIPTextEncode", "inputs": {"text": job.negative_prompt or "", "clip": ["1", 1]}}
  "4": {"class_type": "EmptyLatentImage", "inputs": {"width": job.width, "height": job.height, "batch_size": job.num_images}}
  "5": (existing KSampler, wire inputs to nodes 1-4)
  "6": {"class_type": "VAEDecode", "inputs": {"samples": ["5", 0], "vae": ["1", 2]}}
  "7": {"class_type": "SaveImage", "inputs": {"images": ["6", 0], "filename_prefix": "agent_os"}}
```

### Estimated Changes
- `src/web/routes/images.py`: ~60 lines added/modified

---

## Item 4: System settings API

**Priority:** Medium
**Files:** `src/web/routes/system.py`

### Current State

Lines 125-171: 7 settings stored in `_settings` dict. `GET /settings`, `GET /settings/{key}`, `PUT /settings/{key}` endpoints exist. No code reads these settings. Already documented with TODO comment (added during cleanup).

### Plan

Wire the two settings that have clear, safe consumers. Mark the rest as read-only informational.

#### 4a. Wire `logging.level` (~10 lines)

```
File: src/web/routes/system.py, update_setting() (around line 420)

After _settings[key].value = body.value:
  if key == "logging.level":
      level = getattr(logging, str(body.value).upper(), None)
      if level is not None:
          logging.getLogger("src").setLevel(level)
          logger.info(f"Logging level changed to {body.value}")
```

#### 4b. Wire `api.rate_limit` (~10 lines)

```
File: src/web/routes/system.py, update_setting()

  if key == "api.rate_limit":
      try:
          from ..dependencies import get_limiter
          limiter = get_limiter()
          if hasattr(limiter, 'update_limit'):
              limiter.update_limit(int(body.value))
      except Exception:
          pass  # Rate limiter may not support dynamic updates
```

#### 4c. Mark unwired settings as informational (~5 lines)

Add `"readonly": true` or `"wired": false` to the remaining 5 settings so the UI can indicate they are informational only. Add a note to the PUT response:

```
For unwired settings, return:
  {"status": "stored", "note": "This setting is stored but not yet wired to a runtime consumer."}
```

### Estimated Changes
- `src/web/routes/system.py`: ~25 lines added

---

## Item 5: Structured logging

**Priority:** Medium
**Files:** `src/web/app.py`, `src/web/middleware.py`

### Current State

All logging uses basic `logger.info/error/warning` with unstructured string messages. No JSON formatting. No request correlation IDs. No log aggregation configuration.

### Plan

#### 5a. Add JSON log formatter (~25 lines)

```
File: src/web/app.py

class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if hasattr(record, "request_id"):
            log_entry["request_id"] = record.request_id
        if record.exc_info and record.exc_info[0]:
            log_entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_entry)

Install in create_app() lifespan (only in production/non-debug mode):
    if not config.debug:
        handler = logging.StreamHandler()
        handler.setFormatter(JSONFormatter())
        logging.getLogger("src").addHandler(handler)
```

#### 5b. Add request correlation ID middleware (~15 lines)

```
File: src/web/middleware.py

Add to SecurityHeadersMiddleware.dispatch() (or as separate middleware):
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    # Store in request state for access by route handlers
    request.state.request_id = request_id
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response
```

### Estimated Changes
- `src/web/app.py`: ~25 lines (JSON formatter)
- `src/web/middleware.py`: ~15 lines (correlation ID)

---

## Item 6: WebSocket heartbeat and connection limits

**Priority:** Medium
**Files:** `src/web/routes/chat.py`

### Current State

`ConnectionManager` (lines 287-359) tracks connections in a dict. No heartbeat mechanism. No max connection limit. No message counting. No stale connection cleanup.

### Plan

#### 6a. Add connection limit check (~10 lines)

```
File: src/web/routes/chat.py, ConnectionManager.connect()

Add at top of connect():
    MAX_CONNECTIONS = 100
    if len(self.connections) >= MAX_CONNECTIONS:
        await websocket.close(code=1013, reason="Maximum connections reached")
        return None
```

#### 6b. Add heartbeat ping/pong (~30 lines)

Add a background task that pings all connections periodically and removes stale ones.

```
File: src/web/routes/chat.py

Add method to ConnectionManager:
    async def _heartbeat_loop(self):
        while True:
            await asyncio.sleep(30)  # 30-second interval
            stale = []
            for conn_id, conn in list(self.connections.items()):
                try:
                    await asyncio.wait_for(conn.websocket.send_json({"type": "ping"}), timeout=5)
                except Exception:
                    stale.append(conn_id)
            for conn_id in stale:
                await self.disconnect(conn_id)
                logger.info(f"Removed stale WebSocket connection: {conn_id}")

Start heartbeat in connect() if first connection:
    if len(self.connections) == 1:
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())

Cancel in disconnect() if last connection:
    if len(self.connections) == 0 and self._heartbeat_task:
        self._heartbeat_task.cancel()
```

### Estimated Changes
- `src/web/routes/chat.py`: ~40 lines added

---

## Item 7: Unify authentication patterns

**Priority:** Medium
**Files:** `src/web/routes/chat.py`, `src/web/auth_helpers.py`

### Current State (Revised)

The research reveals auth is **more unified than the audit suggested.** All REST routes use two centralized dependencies from `auth_helpers.py`:
- `require_authenticated_user` — returns user_id, used by most endpoints
- `require_admin_user` — returns user_id, verifies admin role

Both extract tokens from cookies or Bearer headers, validate via `UserStore.validate_session()`, and return appropriate 401/403 responses.

**The only deviation is WebSocket auth** in `chat.py`, which must handle authentication differently because FastAPI `Depends()` doesn't work the same way for WebSocket endpoints.

### Plan

#### 7a. Extract WebSocket auth helper (~15 lines)

The WebSocket handler in `chat.py` likely has inline auth logic. Extract it into a reusable function in `auth_helpers.py` that follows the same pattern:

```
File: src/web/auth_helpers.py

async def authenticate_websocket(websocket: WebSocket) -> Optional[str]:
    """Authenticate a WebSocket connection. Returns user_id or None."""
    token = websocket.cookies.get("session_token")
    if not token:
        # Check query parameter fallback for WebSocket clients
        token = websocket.query_params.get("token")
    if not token:
        return None
    try:
        store = get_user_store()
        user = store.validate_session(token)
        return user.user_id if user else None
    except Exception:
        return None
```

#### 7b. Use the helper in chat.py WebSocket handler (~5 lines)

Replace any inline auth logic in the WebSocket endpoint with:

```
user_id = await authenticate_websocket(websocket)
if not user_id:
    await websocket.close(code=4001, reason="Authentication required")
    return
```

### Estimated Changes
- `src/web/auth_helpers.py`: ~15 lines added
- `src/web/routes/chat.py`: ~5 lines modified

---

## Item 8: Documentation cleanup

**Priority:** Low
**Files:** `docs/Conversational-Kernel.md`

### Current State (Revised)

**No action needed.** Research confirms `docs/Conversational-Kernel.md` references `src/core/constitution.py` (the `ConstitutionalKernel` class), which exists and is actively used in:
- `src/web/routes/system.py:38` — imported for health checks
- `src/web/routes/constitution.py:23` — imported for rule management

The deleted `src/kernel/` module was a separate, unrelated module. The documentation is accurate.

### Plan

No changes required. Remove this item from the remediation checklist.

---

## Summary

| # | Item | Priority | Est. Lines | Risk |
|---|------|----------|-----------|------|
| 1 | Wire system health/metrics to real data | High | ~95 | Low — additive changes, no behavior change to existing code |
| 2 | Agent lifecycle documentation | High | ~25 | Low — response message changes only |
| 3 | Complete ComfyUI backend | High | ~60 | Medium — new network I/O code, needs error handling |
| 4 | Wire system settings | Medium | ~25 | Low — 2 settings wired, rest marked informational |
| 5 | Structured logging | Medium | ~40 | Low — additive, opt-in via non-debug mode |
| 6 | WebSocket heartbeat + limits | Medium | ~40 | Medium — affects connection lifecycle |
| 7 | Unify WebSocket auth | Medium | ~20 | Low — extraction refactor |
| 8 | Documentation cleanup | Low | 0 | None — no action needed |

**Total estimated: ~305 lines across 8 files**

### Execution Order

```
Phase 1 (Critical fixes — items that affect correctness):
  1. Item 1a: Request counter middleware
  2. Item 1b: Wire /status to real checks
  3. Item 1c: Wire /metrics to real counters
  4. Item 2: Agent lifecycle documentation
  5. Item 3: ComfyUI backend completion

Phase 2 (Quality improvements — items that improve production readiness):
  6. Item 1d: Wire /logs to real logging
  7. Item 4: Wire system settings
  8. Item 5a: JSON log formatter
  9. Item 5b: Request correlation IDs
  10. Item 6: WebSocket heartbeat + limits
  11. Item 7: Unify WebSocket auth
```
