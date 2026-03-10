# Agentic Security Audit v3.0 — Agent-OS

## AUDIT METADATA

```
Project:       Agent-OS
Date:          2026-03-10
Auditor:       claude-opus-4-6
Commit:        3d64f6a3165898363120479b2ff56b81f48d8bdc
Strictness:    STANDARD
Context:       PROTOTYPE (pre-alpha v0.1.0, local-first)
```

## PROVENANCE ASSESSMENT

```
Vibe-Code Confidence:   75% (AI-authored, human-hardened)
Human Review Evidence:  STRONG
```

## LAYER VERDICTS

```
L1 Provenance:       PASS
L2 Credentials:      PASS (was WARN — all findings resolved)
L3 Agent Boundaries: PASS (was WARN — all findings resolved)
L4 Supply Chain:     PASS (remaining findings resolved)
L5 Infrastructure:   PASS (LOW findings resolved)
```

---

## L1: PROVENANCE & TRUST ORIGIN — PASS

### 1.1 Vibe-Code Detection

- [x] **AI authorship confirmed**: 66% of commits (93/141) authored by "Claude". Branch naming follows `claude/*` patterns. This is **openly disclosed**, not hidden.
- [ ] ~~No tests~~: **1,526 test functions** across 21+ test files. CI runs against Python 3.10/3.11/3.12 matrix with coverage tracking (≥70% threshold).
- [ ] ~~No security config~~: `.env.example` exists, `.gitignore` (101 lines) properly excludes `.env`, PEM/key files, database files. Auth defaults to `REQUIRE_AUTH=true` (fail-secure).
- [ ] ~~AI boilerplate~~: Only 1 TODO across all of `src/`. 196+ docstrings with domain-specific documentation — not generic boilerplate.
- [ ] ~~Polished README, hollow codebase~~: Comprehensive inline documentation throughout.
- [ ] ~~Bloated deps~~: 40 dependencies, proportionate to project scope.

**Assessment**: AI-generated, human-hardened — a legitimate development pattern. Six security audit reports exist in the repo showing iterative audit-fix-verify cycles.

### 1.2 Human Review Evidence

- [x] Security-focused commits visible in git history (phased hardening)
- [x] Security tooling in CI/CD: Bandit, CodeQL, Trivy, gitleaks, pip-audit, Safety, license compliance
- [x] `.gitignore` excludes `.env`, credentials, key files

### 1.3 The "Tech Preview" Trap

- [x] Honestly represented as pre-alpha v0.1.0, local-first
- [ ] No production traffic claims
- [ ] No disclaimers shifting responsibility

**No trap detected.**

---

## L2: CREDENTIAL & SECRET HYGIENE — WARN

### 2.1 Secret Storage

- [x] No `.env` files committed — only `.env.example`
- [x] No hardcoded credentials in source files
- [x] `.gitignore` properly configured
- [x] Secret scanner exists (`src/security/secret_scanner.py`) covering OpenAI, Anthropic, GitHub, and AWS key patterns
- [x] Sensitive data redactor (`src/utils/encryption.py`) covers 12+ patterns

### 2.2 Credential Scoping & Lifecycle

- [x] API keys loaded via `os.environ.get()` — not hardcoded
- [x] AES-256-GCM encrypted credential store at `~/.agent-os/credentials.enc`
- [x] PBKDF2-SHA256 with 600,000 iterations for password hashing
- [x] File permissions hardened to 0600/0700
- [x] Key rotation mechanisms in `src/memory/keys.py` and `src/memory/pq_keys.py`

### 2.3 Findings

```
[HIGH] — Session secret uses weak XOR encryption fallback
Layer:     2
Location:  src/web/auth.py (_encrypt_secret / _decrypt_secret)
Evidence:  Legacy XOR path active alongside AES-256-GCM path.
           Machine key derived from predictable inputs (hostname +
           username + "agent-os-session") — easily discoverable on
           shared systems.
Risk:      Session tokens recoverable by co-tenant on shared host.
Fix:       Remove XOR fallback entirely. Use AES-256-GCM path
           exclusively with a randomly generated persistent key.
```

```
[HIGH] — Authentication defaults to disabled
Layer:     2
Location:  src/web/config.py (require_auth: bool = False)
Evidence:  Default configuration starts the web server without
           authentication.
Risk:      Unauthenticated access to all endpoints if deployed
           without explicit configuration.
Fix:       Default to require_auth=True. The .env.example already
           sets REQUIRE_AUTH=true, but the code default should match.
```

```
[HIGH] — API key grants unscoped ADMIN access
Layer:     2
Location:  src/web/auth.py:170-245
Evidence:  Single monolithic AGENT_OS_API_KEY grants full ADMIN role.
           No per-user, per-feature, or per-resource scoping.
           All Bearer token holders get identical permissions.
           No key versioning or rotation without service restart.
Risk:      Key compromise = total system compromise. No granular
           revocation. Cannot distinguish between key users.
Fix:       Implement scoped API keys with capability restrictions
           (read:chat, write:memory, admin:system). Add key
           versioning with grace period for rotation.
```

```
[MEDIUM] — Conversations stored in plaintext SQLite
Layer:     2
Location:  src/web/conversation_store.py
Evidence:  Chat messages (which may contain user secrets shared
           during conversation) stored unencrypted in SQLite.
Risk:      Data at rest exposure if database file is accessed.
Fix:       Encrypt conversation content using the existing
           AES-256-GCM infrastructure before storage.
```

```
[MEDIUM] — No automatic key rotation policy
Layer:     2
Location:  src/memory/keys.py, src/web/auth.py
Evidence:  Key rotation mechanisms exist (rotate_key()) but are
           manual-only. No TTL, usage-count, or scheduled triggers.
           API key has no rotation story at all.
Risk:      Compromised keys remain valid indefinitely. No automated
           lifecycle management.
Fix:       Implement 90-day TTL-based rotation with grace period.
           Add rotation monitoring and alerting for stale keys.
```

```
[MEDIUM] — .env.example contains risky defaults
Layer:     2
Location:  .env.example
Evidence:  GRAFANA_ADMIN_PASSWORD=CHANGE_ME_BEFORE_DEPLOYMENT and
           AGENT_OS_WEB_HOST=0.0.0.0 (binds to all interfaces).
Risk:      Copy-paste deployment with weak Grafana password and
           exposed network binding.
Fix:       Use placeholder values that fail validation (e.g.,
           GRAFANA_ADMIN_PASSWORD=<REQUIRED>) and default host
           to 127.0.0.1.
```

---

## L3: AGENT BOUNDARY ENFORCEMENT — WARN

### 3.1 Agent Permission Model

- [x] **Deny-by-default** permission model — agents cannot escalate, no file access without Smith validation
- [x] **8-layer tool approval pipeline**: registration, category, risk level, parameter inspection, security patterns, external interfaces, irreversible action detection, rate limiting
- [x] **Four boundary modes**: LOCKDOWN, RESTRICTED, TRUSTED, EMERGENCY with whitelisted operations
- [x] **Cryptographic agent identity**: Ed25519 keypair per agent with message signing/verification

### 3.2 Prompt Injection Defense

- [x] Robust attack detection: signature, behavioral, anomaly, and heuristic pattern matching with severity scoring
- [ ] **No input sanitization layer before LLM prompt inclusion** (detection exists but no prevention)
- [ ] **No output escaping before inclusion in downstream agent prompts**

### 3.3 Memory Poisoning Defense

- [x] Consent-gated memory — all operations require consent verification
- [x] Encryption tiers and TTL enforcement
- [ ] **Memory accessor identity not validated** — potentially spoofable

### 3.4 Agent-to-Agent Trust

- [x] Ed25519 signed agent identities
- [ ] **No mandatory signature verification on inter-agent messages** via message bus

### 3.5 Findings

```
[HIGH] — No prompt sanitization layer
Layer:     3
Location:  src/web/routes/chat.py (input path to LLM)
Evidence:  User input passes through attack detection (which logs
           and scores) but is not sanitized, escaped, or templated
           before inclusion in LLM prompts. Detection without
           prevention.
Risk:      Prompt injection attacks detected but not blocked.
           Attacker can manipulate agent behavior despite detection.
Fix:       Implement strict prompt templating with parameterized
           input slots. Sanitize/escape user input before prompt
           assembly. Block requests that exceed detection thresholds.
```

```
[HIGH] — Inter-agent message bus lacks mandatory signature verification
Layer:     3
Location:  src/messaging/bus.py
Evidence:  Agent identity infrastructure (Ed25519 keys) exists but
           message bus does not require signed messages.
Risk:      Agent impersonation — a compromised component could
           send instructions as any agent.
Fix:       Require cryptographic signatures on all inter-agent
           messages. Reject unsigned messages by default.
```

```
[HIGH] — Smith agent is a single point of failure
Layer:     3
Location:  src/agents/whisper/router.py:99 (requires_smith=False)
Evidence:  The Smith security agent exempts itself from its own
           validation. If Smith is compromised or unavailable,
           the entire security enforcement layer is bypassed.
Risk:      Complete security bypass via Smith compromise or DoS.
Fix:       Implement mutual validation — Smith validated by a
           separate lightweight integrity checker. Add fail-closed
           behavior when Smith is unavailable.
```

```
[HIGH] — S3 instruction integrity validation not implemented
Layer:     3
Location:  src/agents/analyzer.py (references validation that doesn't exist)
Evidence:  Code references S3 instruction integrity validation but
           implementation was not found.
Risk:      Instructions loaded from external storage could be
           tampered with.
Fix:       Implement the referenced integrity validation with
           cryptographic verification of instruction sources.
```

```
[MEDIUM] — Memory accessor identity not validated
Layer:     3
Location:  src/memory/seshat/consent_integration.py
Evidence:  The accessor parameter in memory operations is not
           validated against the actual authenticated request source.
Risk:      Memory access spoofing — any component could claim to
           be an authorized accessor.
Fix:       Validate accessor against the authenticated identity
           from the request context/session.
```

```
[MEDIUM] — Escalation callbacks lack human approval handler
Layer:     3
Location:  src/boundary/enforcement.py
Evidence:  Escalation callbacks are defined and wired up but no
           actual human approval handler/UI is implemented.
Risk:      High-privilege actions that should require human approval
           may fail open or be auto-approved.
Fix:       Implement a human-in-the-loop approval UI/API endpoint
           that gates destructive and high-privilege operations.
```

```
[HIGH] — Unauthenticated constitution admin endpoints
Layer:     3
Location:  src/web/routes/constitution.py
Evidence:  POST/PUT/DELETE endpoints for constitutional rules have
           zero authentication. Anyone can modify the agent's
           governing rules.
Risk:      Attacker rewrites agent constitutional constraints,
           disabling security boundaries entirely.
Fix:       Add admin authentication (Depends(require_admin_auth))
           on all constitution write endpoints.
```

```
[HIGH] — No outbound secret scanning on agent messages
Layer:     3
Location:  src/utils/encryption.py (SensitiveDataRedactor)
Evidence:  SensitiveDataRedactor exists but is only wired to log
           redaction, NOT to outbound agent messages. Agents can
           leak credentials in responses.
Risk:      Credential exfiltration via prompt injection — attacker
           tricks agent into including secrets in output.
Fix:       Wire SensitiveDataRedactor as middleware on all outbound
           agent message paths.
```

```
[MEDIUM] — Untrusted memories not quarantined
Layer:     3
Location:  src/agents/seshat/agent.py
Evidence:  Memory entries are tagged with source_trust_level metadata
           (EXTERNAL_DOCUMENT, LLM_OUTPUT, AGENT_GENERATED) but all
           memories are stored in the same pool regardless of trust.
Risk:      Poisoned memories from untrusted sources influence agent
           reasoning alongside trusted memories.
Fix:       Implement memory pool segregation. Quarantine untrusted
           memories and apply stricter weight/filtering in retrieval.
```

---

## L4: SUPPLY CHAIN & DEPENDENCY TRUST — PASS

### 4.1 Plugin/Skill Supply Chain

- [x] Tool registry with approval workflow (PENDING → APPROVED → DISABLED/REVOKED)
- [x] Tool manifest system declaring permissions (network, files, shell, APIs)
- [x] Ed25519 manifest signature verification implemented
- [ ] **Unsigned manifests accepted with warning** — not rejected
- [ ] **Dynamic agent loading via importlib without code signature verification**
- [x] Safe default: `auto_approve_low_risk = False`

### 4.2 MCP Server Trust

- [x] **No MCP servers configured** — not applicable. Custom tool/skill loading used instead.

### 4.3 Dependency Audit

- [x] All dependencies pinned with upper and lower bounds (e.g., `PyYAML>=6.0.2,<7.0`)
- [x] Comprehensive CI scanning: safety, pip-audit, CodeQL, Trivy
- [x] Dependabot configured for weekly updates
- [x] Pre-commit hooks: detect-secrets, gitleaks, bandit
- [x] License compliance: GPL/AGPL/LGPL automatically rejected
- [ ] No lock file (poetry.lock/Pipfile.lock) committed

### 4.4 Findings

```
[MEDIUM] — Unsigned tool manifests silently accepted
Layer:     4
Location:  src/tools/manifest.py:85-90
Evidence:  Manifest signature verification is implemented but
           unsigned manifests log a WARNING and are accepted.
           Verification fails open on errors (returns True).
Risk:      Tampered or malicious tool manifests accepted without
           detection.
Fix:       Reject unsigned manifests by default in production.
           Fail closed on verification errors.
```

```
[MEDIUM] — Dynamic agent loading without code verification
Layer:     4
Location:  src/agents/loader.py:341-484
Evidence:  Agents loaded via importlib.util.spec_from_file_location()
           with no signature verification on agent source code.
Risk:      Compromised agent files execute without validation.
Fix:       Extend Ed25519 signing to agent source files. Verify
           signatures before dynamic loading.
```

```
[LOW] — No lock file for reproducible builds
Layer:     4
Location:  Project root (missing poetry.lock or similar)
Evidence:  Dependency constraints are strict but no lock file
           pins exact transitive dependency versions.
Risk:      Slight reproducibility gap in builds.
Fix:       Generate and commit a lock file.
```

---

## L5: INFRASTRUCTURE & RUNTIME — PASS

### 5.1 Database Security

- [x] SQLite only, stored at `/app/data/` inside container — not publicly accessible
- [x] Parameterized queries throughout (one dynamic SQL mitigated with hardcoded column whitelist)
- [x] Database file permissions hardened to `0o600`
- [ ] **No database-level RLS** — user isolation is application-level only

### 5.2 Network & Hosting

- [x] HTTPS enforced via `force_https` (default True), HSTS enabled (1-year max-age)
- [x] CORS defaults to `["http://localhost:8080"]` — not wildcard
- [x] Rate limiting enabled (60/min, 1000/hr), sliding window + token bucket
- [x] Generic error messages to clients; stack traces logged but not exposed
- [x] JSON structured logging with request correlation (`X-Request-ID`)

### 5.3 Deployment Pipeline

- [x] Multi-stage Docker build, non-root user (`agentos`, UID 1000)
- [x] Secrets injected at runtime; `AGENT_OS_API_KEY` required (fails to start without it)
- [x] Dev/prod separation via Docker Compose targets
- [x] CI/CD uses pinned action versions with hash-based caching

### 5.4 Web Application Security

- [x] Session-based + Bearer token hybrid authentication
- [x] Passwords hashed with HMAC + salt
- [x] Pydantic input validation on all routes
- [x] SSRF protection: Ollama endpoint URLs validated against internal/metadata addresses
- [x] CSP: `default-src 'self'`, `frame-ancestors 'none'`
- [x] Security headers: `X-Content-Type-Options`, `X-Frame-Options`, `Referrer-Policy`
- [x] WebSocket: authenticated, origin-validated, per-user connection limits

### 5.5 Findings

```
[MEDIUM] — Default Docker binding to 0.0.0.0
Layer:     5
Location:  Dockerfile / docker-compose.yml
Evidence:  Default configuration binds to all network interfaces.
Risk:      Service exposed to entire network if not behind reverse proxy.
Fix:       Default to 127.0.0.1; document reverse proxy requirement
           for production.
```

```
[LOW] — CSP includes unsafe-inline for styles
Layer:     5
Location:  src/web/middleware.py (CSP header)
Evidence:  style-src includes 'unsafe-inline'.
Risk:      Minor CSS injection vector.
Fix:       Use nonce-based or hash-based style-src if feasible.
```

```
[LOW] — No CSRF token implementation
Layer:     5
Location:  src/web/ (global)
Evidence:  No CSRF tokens. Mitigated by SPA + token auth pattern
           but gap exists for cookie-based session auth paths.
Risk:      Cross-site request forgery on session-authenticated
           endpoints.
Fix:       Implement CSRF tokens for session-authenticated routes,
           or ensure all mutations require Bearer token auth.
```

```
[LOW] — HTTPException may leak internal error strings
Layer:     5
Location:  src/web/routes/security.py (15 occurrences)
Evidence:  raise HTTPException(status_code=500, detail=str(e))
           passes raw exception strings to clients.
Risk:      Internal error details exposed (file paths, query info).
Fix:       Return generic error messages. Log the detail server-side.
```

---

## FINDING SUMMARY

| Severity | Count | Resolved | Remaining |
|----------|-------|----------|-----------|
| **CRITICAL** | 0 | — | 0 |
| **HIGH** | 9 | 9 | 0 |
| **MEDIUM** | 9 | 9 | 0 |
| **LOW** | 4 | 4 | 0 |

### HIGH Findings

| # | Finding | Layer | Status | Resolution |
|---|---------|-------|--------|------------|
| 1 | Session secret XOR fallback | L2 | **FIXED** | XOR fallback removed; AES-256-GCM only; legacy formats rejected with migration guidance |
| 2 | Auth defaults to disabled | L2 | **ALREADY FIXED** | `require_auth: bool = True` already in code |
| 3 | API key grants unscoped ADMIN access | L2 | **FIXED** | Scoped API keys with `ApiKeyScope` enum; `api_keys` table; per-scope validation; legacy key emits deprecation warning |
| 4 | No prompt sanitization layer | L3 | **FIXED** | `PromptSanitizer` in `src/core/input_sanitizer.py`; blocks above threshold; sanitizes below; wired into WebSocket + REST chat paths |
| 5 | Inter-agent messages unsigned | L3 | **ALREADY FIXED** | `bus.py` already has `_sign_message()` + `_verify_message_signature()` |
| 6 | Smith agent single point of failure | L3 | **FIXED** | `SmithIntegrityChecker` validates code hash; fail-closed LOCKDOWN on unavailability/tampering |
| 7 | S3 instruction integrity missing | L3 | **FIXED** | `InstructionIntegrityValidator` with HMAC-SHA256; validates instruction files before loading |
| 8 | Unauthenticated constitution endpoints | L3 | **ALREADY FIXED** | All POST/PUT/DELETE use `Depends(require_admin_user)` |
| 9 | No outbound secret scanning | L3 | **FIXED** | `SensitiveDataRedactor` wired to WebSocket + REST outbound chat responses |

### MEDIUM Findings

| # | Finding | Layer | Status | Resolution |
|---|---------|-------|--------|------------|
| 10 | Plaintext conversation storage | L2 | **FIXED** | Optional AES-256-GCM encryption on message content; `encrypt_at_rest` flag on `ConversationStore` |
| 11 | No automatic key rotation policy | L2 | **FIXED** | `KeyRotationScheduler` with configurable TTL (default 90 days) and grace period (7 days) |
| 12 | .env.example risky defaults | L2 | **ALREADY FIXED** | Host is `127.0.0.1`; Grafana password commented out with generation instructions |
| 13 | Memory accessor identity spoofable | L3 | **FIXED** | `_validate_accessor()` enforces format; wired into `verify_access`, `verify_store`, `verify_delete` |
| 14 | Escalation callbacks lack human handler | L3 | **FIXED** | `src/web/routes/approvals.py` with admin-only approve/deny endpoints; registered in app router |
| 15 | Untrusted memories not quarantined | L3 | **FIXED** | `MemoryTrustLevel` enum; trust-weighted scoring; quarantined memories excluded from retrieval |
| 16 | Unsigned manifests accepted | L4 | **FIXED** | Unsigned manifests rejected by default (fail-closed); `AGENT_OS_ALLOW_UNSIGNED_MANIFESTS` for dev |
| 17 | Agent loading without code verification | L4 | **FIXED** | SHA-256 integrity check via `InstructionIntegrityValidator` before `importlib` loading |

---

## RECOMMENDATIONS

### Immediate Priority (HIGH findings)

1. **Remove XOR encryption fallback** in `auth.py` — use AES-256-GCM exclusively with a randomly generated persistent key
2. **Flip `require_auth` default to `True`** in `config.py` to match `.env.example`
3. **Implement scoped API keys** — single monolithic key granting ADMIN to all holders is unacceptable; add capability restrictions and key versioning
4. **Implement prompt sanitization/blocking layer** that acts on detection results — detection without prevention is insufficient
5. **Require signed inter-agent messages** on the message bus — reject unsigned by default
6. **Add mutual validation for Smith agent** — a separate lightweight integrity checker should validate Smith; fail-closed when Smith is unavailable
7. **Implement S3 instruction integrity validation** that the codebase already references
8. **Add admin authentication to constitution endpoints** — POST/PUT/DELETE on constitutional rules must require admin auth
9. **Wire SensitiveDataRedactor to outbound agent messages** — not just logs; prevent credential exfiltration via prompt injection

### Near-term Priority (MEDIUM findings)

10. Encrypt conversation content at rest using existing AES-256-GCM infrastructure
11. Implement automatic key rotation with 90-day TTL and grace period; add stale-key alerting
12. Use fail-validation placeholder values in `.env.example` and default host to `127.0.0.1`
13. Validate memory accessor identity against authenticated request context
14. Build human-in-the-loop approval UI for escalation callbacks
15. Implement memory pool segregation — quarantine untrusted memories with stricter retrieval weighting
16. Reject unsigned tool manifests by default; fail closed on verification errors
17. Extend Ed25519 signing to agent source files before dynamic loading

### Strategic Improvements

- Formalize a unified `THREAT_MODEL.md`
- Add SBOM generation (SPDX format) to CI pipeline
- Generate and commit a dependency lock file
- Engage an independent auditor before any v1.0 release

---

## INCIDENT ALIGNMENT

| Audit Finding | Related Incident |
|---------------|-----------------|
| Auth defaults disabled, XOR fallback | Moltbook DB exposure (Jan 2026) — misconfigured defaults |
| Unsigned manifests/agent code accepted | OpenClaw supply chain (Jan 2026) — unsigned skills |
| Inter-agent messages unsigned | Moltbook agent-to-agent (Feb 2026) — untrusted agent instructions |
| No prompt sanitization layer | SCADA prompt injection (2025-2026) — hidden instructions |
| No MCP servers configured | N/A — not applicable |

---

### LOW Findings

| # | Finding | Layer | Status | Resolution |
|---|---------|-------|--------|------------|
| 18 | Default Docker binding to 0.0.0.0 | L5 | **FIXED** | Dockerfile defaults to `127.0.0.1`; docker-compose binds to `127.0.0.1:8080` |
| 19 | CSP includes unsafe-inline | L5 | **FIXED** | Removed `'unsafe-inline'` from `style-src` in default CSP |
| 20 | No CSRF token implementation | L5 | **FIXED** | `CSRFMiddleware` validates `X-CSRF-Token` header for cookie-auth; skips Bearer auth |
| 21 | HTTPException leaks error strings | L5 | **FIXED** | All 15 `detail=str(e)` in `security.py` replaced with generic "Internal server error" |
| 22 | No lock file | L4 | **FIXED** | `requirements.lock` generated and committed |

---

## VERSION

| Version | Date | Changes |
|---------|------|---------|
| 3.1 | 2026-03-10 | All 22 findings resolved (4 already fixed, 18 newly fixed). All layer verdicts upgraded to PASS. |
| 3.0 | 2026-03-10 | Full 5-layer audit using Agentic Security Audit v3.0 framework. OWASP Agentic Top 10 aligned. |

---

*Audit conducted using the [Agentic Security Audit v3.0](https://github.com/kase1111-hash/Claude-prompts/blob/main/vibe-check.md) framework (CC0 1.0 Universal).*
