# Agentic Security Audit v2.0 — Agent-OS

## Audit Report

---

### AUDIT METADATA

```
Project:          Agent-OS (Constitutional Operating System for Local AI)
Audit Date:       2026-02-21
Audit Prompt:     Agentic Security Audit v2.0 (Post-Moltbook Edition)
Auditor:          Claude Opus 4.6
Codebase Hash:    452d187334b32feea77c3cfe2ee2410797f2374b
Strictness:       STANDARD
Context:          PROTOTYPE (pre-alpha, local-first, no production traffic yet)
```

---

### PROVENANCE ASSESSMENT

```
Vibe-Code Confidence:    70% — predominantly AI-generated with iterative human-directed review
Human Review Evidence:   MODERATE
Tech Preview Trap:       NO
```

**Rationale:**

The commit history shows 83 of 133 commits authored by "Claude" (62%), with the
initial commit adding 353 files / 154,481 insertions in a single merge. Branch
names follow the `claude/*` pattern consistently. This is clearly an AI-generated
codebase.

However, unlike the Moltbook case, there is **substantial evidence of
human-directed security review**:

- Multiple dedicated security audit cycles (PRs #112–#115)
- Security-focused commit messages: "Phase 1: Close authentication gaps",
  "Phase 3: Cryptographic message signing", "Phase 5: Supply chain and CI hardening"
- SECURITY.md with vulnerability reporting policy
- `.pre-commit-config.yaml` with detect-secrets, gitleaks, bandit
- `.gitignore` that properly excludes `.env`, `*.key`, `*.pem`, credentials
- CONTRIBUTING.md and CODE_OF_CONDUCT.md present
- 36 test files with 1,081 passing tests across 30+ modules
- CI pipeline with security scanning (bandit, safety, CodeQL, Trivy, gitleaks)
- Dependabot configured for automated dependency updates

**The "Tech Preview Trap" does NOT apply** — the project is explicitly pre-alpha
(v0.1.0), runs locally, handles no real user credentials, and has no production
traffic. The claimed status matches actual exposure.

---

### LAYER VERDICTS

```
Layer 1 — Provenance:        WARN
Layer 2 — Credentials:       PASS
Layer 3 — Agent Boundaries:  WARN
Layer 4 — Supply Chain:      PASS
Layer 5 — Infrastructure:    PASS
```

---

## FINDINGS

---

### [HIGH] — 36 Web Endpoints Missing Authentication Guards

```
Layer:     3 (Agent Boundary Enforcement) / 5 (Infrastructure)
Location:  src/web/routes/constitution.py (all 9 endpoints)
           src/web/routes/images.py (all 10 endpoints)
           src/web/routes/system.py (9 of 11 endpoints)
           Additional routes in contracts.py, chat.py, memory.py
Evidence:  Route handlers do not use Depends(require_authenticated_user) or
           Depends(require_admin_user). The auth_helpers module exists and works
           correctly, but many routes simply don't call it.
Risk:      Unauthenticated users could:
           - Modify constitutional rules (governance bypass)
           - Trigger image generation (resource exhaustion)
           - Read system status and agent configurations
           - Access memory and contract management endpoints
Fix:       1. Add Depends(require_admin_user) to ALL constitution.py endpoints
           2. Add Depends(require_authenticated_user) to images.py, memory.py,
              contracts.py, and system.py endpoints
           3. Audit every route file under src/web/routes/ for auth coverage
           4. Add integration tests that verify 401 for unauthenticated requests
Reference: OWASP A01:2021 — Broken Access Control
```

---

### [HIGH] — Default-ALLOW Agent Permission Model

```
Layer:     3 (Agent Boundary Enforcement)
Location:  src/agents/ — agent configuration and capability system
Evidence:  Agents default to broad capability sets. The base agent configuration
           uses can_escalate: True. Capabilities are granted as Set[str] without
           fine-grained role-based access control.
Risk:      Agents can acquire permissions beyond their designated role. If the
           Smith guardian agent is unavailable, boundary enforcement degrades.
           This is the same class of failure seen in the Moltbook agent system
           where agents had unconstrained capabilities.
Fix:       1. Change default to deny-all: agents should explicitly opt-in to
              each capability
           2. Replace can_escalate: True default with can_escalate: False
           3. Implement mandatory capability checks at the message bus level,
              not just advisory Smith checks
           4. Add capability attestation: agents must prove they hold a
              capability before the bus routes their message
Reference: Moltbook agent-to-agent attacks (Feb 2026)
```

---

### [HIGH] — Agent Identity Signing Not Integrated Into Message Bus

```
Layer:     3 (Agent Boundary Enforcement)
Location:  src/agents/identity.py (exists)
           src/messaging/ (not wired)
Evidence:  The AgentIdentity module provides Ed25519 keypair generation and
           message signing. However, the message bus does not verify signatures
           when routing messages between agents. The identity infrastructure
           exists but is not enforced.
Risk:      Any component that can publish to the message bus can impersonate
           any agent. Cross-agent instructions are not cryptographically verified,
           enabling agent-to-agent spoofing.
Fix:       1. Wire AgentIdentity.sign() into the message bus publish path
           2. Add signature verification on the receive path
           3. Reject unsigned or incorrectly signed messages
           4. Log signature verification failures for audit
Reference: Moltbook V4 (Bot-to-Bot Social Engineering), V8 (Identity Spoofing)
```

---

### [MEDIUM] — Floating Dependency Versions

```
Layer:     4 (Supply Chain & Dependency Trust)
Location:  pyproject.toml, requirements.txt
Evidence:  All dependencies use >= constraints (e.g., fastapi>=0.115,
           cryptography>=42.0). No upper bounds or lock file present.
           Dependabot is configured but only for automated PR creation,
           not enforcement.
Risk:      A compromised or breaking upstream release could be automatically
           pulled in. Supply chain attacks via dependency confusion or
           typosquatting could exploit the open version ranges.
Fix:       1. Add upper bounds: fastapi>=0.115,<1.0 (or tighter ranges)
           2. Generate and commit a requirements.lock or pip-compile output
           3. Pin exact versions in CI/CD for reproducible builds
           4. Consider using pip-audit in CI to block known-vulnerable versions
Reference: OWASP A06:2021 — Vulnerable and Outdated Components
```

---

### [MEDIUM] — LLM Judge Can Override Structural Denial

```
Layer:     3 (Agent Boundary Enforcement)
Location:  src/core/enforcement.py (3-tier enforcement engine)
Evidence:  The enforcement engine has three tiers: structural checks (regex
           deny patterns), semantic matching, and LLM compliance judgment.
           When Tier 1 structural checks return definitive=False, the request
           proceeds to Tier 3 LLM judgment. A sophisticated adversary could
           craft inputs that bypass regex patterns but persuade the LLM judge
           to approve the request.
Risk:      Prompt injection that evades structural patterns could be approved
           by the LLM judge tier. The LLM is inherently susceptible to
           adversarial prompts designed to look like legitimate requests.
Fix:       1. Ensure Tier 1 structural denials are always definitive (cannot
              be overridden by Tier 3)
           2. Add a Tier 1 allowlist for known-safe request patterns
           3. Log all Tier 3 overrides for human review
           4. Consider rate-limiting Tier 3 escalations per agent/session
Reference: Time-shifted prompt injection (2026 threat vector)
```

---

### [MEDIUM] — Agent File System Access Not Sandboxed

```
Layer:     3 (Agent Boundary Enforcement)
Location:  src/tools/ — tool execution framework
           src/tools/subprocess_runner.py
Evidence:  The subprocess runner strips sensitive env vars and reads params from
           files (good), but agents run with full read access to the parent
           process's file system. Only module name validation (prefix checking)
           is performed. There is no chroot, namespace isolation, or seccomp
           filtering.
Risk:      A compromised or misbehaving agent could read sensitive files on
           the host system (SSH keys, other application configs, etc.).
Fix:       1. Implement directory allowlists per agent role
           2. Add seccomp filtering or use Linux namespaces for subprocess
              isolation
           3. Consider running tool subprocesses in minimal containers
           4. Log all file access attempts for audit
Reference: Moltbook — agents had unconstrained file system access
```

---

### [MEDIUM] — Network Access Enabled by Default for Agents

```
Layer:     3 (Agent Boundary Enforcement)
Location:  src/boundary/ — boundary security daemon
           .env.example:96 — AGENT_OS_BOUNDARY_NETWORK_ALLOWED=true
Evidence:  Network access is enabled by default (AGENT_OS_BOUNDARY_NETWORK_ALLOWED=true).
           SSRF protections exist (blocking metadata services, internal TLDs), but
           agents can make arbitrary outbound HTTP requests to any external host.
           No per-agent network allowlists are implemented.
Risk:      A compromised agent could exfiltrate data to external servers or be
           used as a proxy for network attacks.
Fix:       1. Change default to AGENT_OS_BOUNDARY_NETWORK_ALLOWED=false
           2. Implement per-agent network allowlists (e.g., only Ollama backend
              for most agents)
           3. Log all outbound network requests with agent identity
           4. Add egress firewall rules in Docker deployment
Reference: Moltbook — agents had unrestricted network communication
```

---

### [MEDIUM] — Test Failures in Web Module (67 Errors, 22 Failures)

```
Layer:     1 (Provenance & Trust Origin)
Location:  tests/test_web.py
Evidence:  Running the test suite produces 1,081 passes but 22 failures and
           67 errors, all concentrated in test_web.py (NameError exceptions).
           The errors appear to be import/dependency issues rather than logic
           failures, but they indicate the web module has degraded test coverage.
Risk:      Security-relevant web endpoints may not be exercised by tests.
           Regressions in authentication, rate limiting, or input validation
           could go undetected.
Fix:       1. Fix the NameError import issues in test_web.py
           2. Ensure all auth-protected endpoints have tests verifying 401/403
           3. Add tests for rate limiting edge cases
           4. Target 100% test coverage for src/web/auth.py and auth_helpers.py
Reference: Vibe-code indicator — test gaps in security-critical modules
```

---

### [LOW] — CI Security Scans Use continue-on-error

```
Layer:     4 (Supply Chain & Dependency Trust)
Location:  .github/workflows/ci.yml:186-188
           .github/workflows/security.yml:135-139
Evidence:  Both bandit and safety checks use "|| true" or "continue-on-error: true",
           meaning security findings never fail the CI pipeline.
Risk:      Known vulnerabilities could be merged without blocking. The security
           scans exist but are advisory only.
Fix:       1. Remove "|| true" from bandit and safety commands
           2. Set continue-on-error: false for security-critical jobs
           3. At minimum, fail on HIGH/CRITICAL severity findings
           4. Add a required status check for the security job
Reference: OWASP A06:2021 — Vulnerable and Outdated Components
```

---

### [LOW] — GitHub Actions Not Pinned to SHA

```
Layer:     4 (Supply Chain & Dependency Trust)
Location:  .github/workflows/ci.yml — uses actions/checkout@v6
           .github/workflows/security.yml — uses actions/checkout@v4
Evidence:  GitHub Actions are pinned to major version tags (v4, v5, v6) rather
           than commit SHAs. Additionally, there is version inconsistency between
           workflows (security.yml uses v4 while ci.yml uses v6).
Risk:      A compromised GitHub Action could inject malicious code into the build
           pipeline. Tag-based pinning is vulnerable to tag rewriting.
Fix:       1. Pin all actions to full commit SHA
           2. Use Dependabot to auto-update action SHAs
           3. Align action versions across all workflow files
Reference: Supply chain attacks via GitHub Actions
```

---

### [LOW] — CORS Origins Configured for Localhost Only

```
Layer:     5 (Infrastructure & Runtime)
Location:  src/web/config.py:33
Evidence:  Default CORS origins: ["http://localhost:8080"]. This is correct for
           local development but would need updating for any networked deployment.
Risk:      Minimal for local-first architecture. However, if deployed behind a
           reverse proxy with a different origin, CORS would block legitimate
           requests, potentially leading operators to set origins to ["*"].
Fix:       1. Document CORS configuration for reverse proxy deployments
           2. Add AGENT_OS_CORS_ORIGINS env var for production configuration
           3. Never default to ["*"] — require explicit configuration
Reference: OWASP — Misconfigured CORS
```

---

### [INFO] — Predominantly AI-Authored Codebase

```
Layer:     1 (Provenance & Trust Origin)
Location:  Entire repository
Evidence:  62% of commits by "Claude", massive initial commit (154k insertions),
           all feature branches follow claude/* naming pattern. The codebase is
           77,502 lines across 150 Python files — substantial for a project
           started December 31, 2025.
Risk:      AI-generated code requires heightened scrutiny for subtle logic errors,
           security misconfigurations, and over-engineering. The rapid development
           pace means security patterns may be structurally present but not
           functionally integrated (as seen with the identity module).
Fix:       Continue the pattern of iterative security review cycles. Consider
           engaging a human security auditor for the agent boundary layer
           specifically, as this is the highest-risk area.
Reference: Moltbook — entirely vibe-coded without security review
```

---

### [INFO] — Post-Quantum Cryptography Support

```
Layer:     2 (Credential & Secret Hygiene)
Location:  src/memory/pq_keys.py
Evidence:  The memory vault includes optional post-quantum key encapsulation
           (ML-KEM) and digital signatures (ML-DSA). This is forward-looking
           but the module is optional and not exercised in the standard test suite
           (test_pq_keys.py is excluded from default test runs).
Risk:      None currently. The PQ implementation is additive and does not weaken
           the existing AES-256-GCM encryption.
Fix:       No action required. Consider adding PQ tests to the default suite when
           liboqs becomes more widely available.
Reference: NIST post-quantum cryptography standards
```

---

## POSITIVE FINDINGS

The following practices demonstrate security awareness and should be maintained:

1. **Encryption architecture is excellent.** Four-tier encrypted memory vault
   (Working/Private/Sealed/Vaulted) with AES-256-GCM, PBKDF2 (100k iterations),
   machine-specific key derivation, and secure key zeroing in memory
   (`src/memory/keys.py`).

2. **Zero hardcoded credentials.** Thorough search found no API keys, passwords,
   or tokens in source code. All secrets are externalized to environment variables.

3. **Pre-commit security hooks.** detect-secrets, gitleaks, bandit, and flake8
   are configured to scan on every commit. This is above-average for any project.

4. **Constitutional enforcement is a genuine innovation.** The 3-tier enforcement
   engine (structural → semantic → LLM judge) with fail-safe defaults is a
   thoughtful defense-in-depth pattern (`src/core/enforcement.py`).

5. **Docker security is well-configured.** Multi-stage builds, non-root user,
   resource limits, health checks, required secret injection
   (`docker-compose.yml` uses `${AGENT_OS_API_KEY:?...}`).

6. **Input classification gate.** The `MarkupSanitizer` and `DATA_CONTEXT`
   boundary markers for separating data from instructions address prompt injection
   at the architectural level (`src/core/input_classifier.py`).

7. **Outbound secret scanning.** The `SecretScanner` module prevents agents from
   leaking credentials in inter-agent messages — a control few agent systems
   implement (`src/security/secret_scanner.py`).

8. **Session management.** HTTPOnly cookies, Bearer token support, IP binding,
   automatic session cleanup, and configurable timeouts
   (`src/web/auth.py`, `src/web/app.py`).

9. **Security headers middleware.** CSP, X-Frame-Options, X-Content-Type-Options,
   HSTS, and referrer policy are all configured (`src/web/middleware.py`).

10. **Collective action detection.** The `CoordinationMonitor` detects when
    multiple agents converge on the same resource — a novel defense against
    coordinated agent attacks (`src/messaging/coordination.py`).

---

## RECOMMENDED ACTIONS

### Immediate (do today)

1. [ ] Add `Depends(require_admin_user)` to all `src/web/routes/constitution.py` endpoints
2. [ ] Add `Depends(require_authenticated_user)` to `images.py`, `memory.py`, `contracts.py` routes
3. [ ] Fix test_web.py NameError failures to restore web module test coverage

### Short-term (this week)

4. [ ] Wire `AgentIdentity.sign()` into message bus publish/receive paths
5. [ ] Change agent default to `can_escalate: False` (deny-by-default)
6. [ ] Change `AGENT_OS_BOUNDARY_NETWORK_ALLOWED` default to `false`
7. [ ] Pin dependency versions with upper bounds in `pyproject.toml`

### Medium-term (this month)

8. [ ] Implement per-agent file system allowlists and directory sandboxing
9. [ ] Implement per-agent network allowlists
10. [ ] Remove `continue-on-error: true` from CI security scans (at least for HIGH/CRITICAL)
11. [ ] Pin GitHub Actions to commit SHAs instead of version tags
12. [ ] Add integration tests verifying authentication on every protected endpoint

### Long-term (this quarter)

13. [ ] Engage human security auditor for agent boundary layer review
14. [ ] Implement Linux namespace/seccomp isolation for tool subprocess execution
15. [ ] Add mutual TLS for any future networked agent-to-agent communication
16. [ ] Consider formal threat modeling document for the agent permission system

---

*Audit generated using the [Agentic Security Audit v2.0](https://github.com/kase1111-hash/Claude-prompts/blob/main/vibe-check.md) framework (CC0 1.0 Universal — Public Domain).*

*"The agent usefulness correlates directly with access level. Sandboxing solves security but cripples functionality." — This audit makes the tradeoffs visible so humans can decide.*
