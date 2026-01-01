# Agent OS - Alpha Release TODO

This document tracks items that need to be addressed before the alpha release.

## Status Legend
- 🔴 **Critical** - Blocks alpha release
- 🟡 **High** - Should fix before alpha
- 🟢 **Medium** - Nice to have for alpha
- ⚪ **Defer** - Post-alpha

---

## Testing ⚠️ Needs Work

### 🔴 Critical: 59 Skipped Tests
Tests are being skipped due to missing dependencies or conditional checks. These need validation in CI.

**Affected areas:**
- `test_web.py` (~13 skips) - FastAPI availability checks
- `test_pq_keys.py` (~23 skips) - Post-quantum crypto not available
- `test_value_ledger.py` (5 skips) - Optional dependency
- `test_web.py` (2 skips) - No rules/configurations to test

**Action items:**
- [ ] Install optional dependencies in CI environment
- [ ] Add conditional skip documentation explaining why each is skipped
- [ ] Validate that skipped tests pass when dependencies available
- [ ] Create CI matrix to test with/without optional deps

### ✅ ~~High: Missing Test Modules~~ PARTIALLY FIXED
**Resolution:** Created key test files:
- ✅ `tests/test_utils.py` - Covers encryption, credentials, redaction
- ✅ `tests/test_observability.py` - Covers metrics, health checks
- [ ] Expand `tests/test_voice.py` - Limited coverage currently
- [ ] Expand `tests/test_core.py` - Constitution kernel partial coverage

### 🟡 High: Exception Handler Review
50+ `pass` statements in exception handlers across:
- [ ] `src/messaging/bus.py` (6 instances)
- [ ] `src/boundary/__init__.py` (2 instances)
- [ ] `src/ledger/` modules (3 instances)
- [ ] `src/boundary/daemon/` modules (2 instances)
- [ ] `src/federation/` modules (8 instances)
- [ ] `src/observability/` modules (1 instance)
- [ ] `src/installer/` modules (8 instances)

**Risk:** Silent failures in error paths. Review each and add proper error handling or logging.

---

## Security Config ⚠️ Needs Work

### ✅ ~~Critical: Hardcoded Grafana Password~~ FIXED
**File:** `docker-compose.yml:113`

**Resolution:** Removed default value. Now requires `GRAFANA_ADMIN_PASSWORD` to be set:
```yaml
GF_SECURITY_ADMIN_PASSWORD=${GRAFANA_ADMIN_PASSWORD:?GRAFANA_ADMIN_PASSWORD must be set}
```

Also updated `.env.example` with clear security warnings.

### ✅ ~~Critical: Auth Disabled by Default~~ FIXED
**File:** `.env.example:28`

**Resolution:** Changed default to `AGENT_OS_REQUIRE_AUTH=true` with clear documentation.

### ✅ ~~High: No Pre-commit Hooks~~ FIXED
**Resolution:** Created `.pre-commit-config.yaml` with:
- Secret detection (detect-secrets, gitleaks)
- Python linting (black, isort, flake8)
- Security scanning (bandit)
- YAML/JSON/TOML validation
- Markdown linting

**Setup:** `pip install pre-commit && pre-commit install`

### 🟡 High: API Key Not Enforced
**File:** `.env.example:31`
```
# AGENT_OS_API_KEY=your-secure-api-key-here
```

**Issue:** API key is commented out and not required.

**Fix:**
- [ ] Add startup validation for API key when auth enabled
- [ ] Document API key requirements
- [ ] Add API key generation utility

---

## Documentation ⚠️ Needs Work

### ✅ ~~Critical: Windows-Only Quickstart~~ FIXED
**Resolution:** Created comprehensive `START_HERE_LINUX.md` covering:
- Linux (Ubuntu/Debian, Fedora/RHEL) and macOS installation
- Shell scripts: `build.sh` and `start.sh`
- Systemd and launchd service configuration
- Troubleshooting section

Also updated `docs/README.md` quick reference table.

### ✅ ~~High: "Coming Soon" Items in Docs~~ FIXED
**Resolution:** Updated `docs/governance/security.md` to provide clear alpha-phase guidance:
- Removed vague "coming soon" references
- Added practical alternatives for security reporting
- Clarified that dedicated infrastructure is planned for Phase 2

### ✅ ~~High: Alpha Release Notes~~ FIXED
**Resolution:** Created `ALPHA_RELEASE_NOTES.md` with:
- Known limitations (functional, technical, security)
- Hardware requirements (minimum and recommended)
- Supported platforms table
- Breaking changes expected before 1.0
- Installation and configuration instructions

### ✅ ~~Medium: Installation Docs Improvements~~ FIXED
**Resolution:**
- ✅ Expanded troubleshooting section in `docs/RUNNING_AND_COMPILING.md` (20+ error scenarios)
- ✅ Created `scripts/check_requirements.py` for hardware/software verification
- ✅ Documented common errors with solutions
- [ ] Add video walkthrough links (placeholder for now) - deferred

---

## Code Completeness

### 🟡 High: Unimplemented TODOs
- [ ] `build/windows/build.py:279` - WiX MSI installer
- [ ] `src/boundary/client.py:137` - Remote socket connection
- [ ] `src/agents/smith/attack_detection/remediation.py:269` - Input validation

### ⚪ Defer: NotImplementedError (Post-Alpha)
These are Phase 2+ features:
- `federation/pq/hsm.py` - PKCS#11 HSM support (9 methods)
- `federation/pq/hybrid_certs.py` - Certificate upgrade
- `agents/seshat/embeddings.py` - Abstract methods (need concrete impl)
- `sdk/testing/fixtures.py` - SDK testing framework

---

## CI/CD

### ✅ ~~High: Missing CI Pipeline~~ FIXED
**Resolution:** CI/CD workflows already exist and enhanced:
- ✅ `.github/workflows/ci.yml` - Python 3.10/3.11/3.12 matrix, lint, test, build
- ✅ `.github/workflows/security.yml` - Enhanced with CodeQL, Trivy, license checks

### ✅ ~~Medium: Additional CI Improvements~~ FIXED
**Resolution:**
- ✅ Created `tox.ini` with py310/py311/py312, lint, typecheck, coverage, security envs
- ✅ Created `.coveragerc` with detailed coverage configuration
- [ ] Add performance benchmarks - deferred to Phase 2
- [ ] Add documentation build verification - deferred

---

## Tracking

| Category | Critical | High | Medium | Defer | Fixed |
|----------|----------|------|--------|-------|-------|
| Testing | 1 | ~~2~~ 1 | 0 | 0 | 1 |
| Security Config | ~~2~~ 0 | ~~2~~ 1 | 0 | 0 | 3 |
| Documentation | ~~1~~ 0 | ~~3~~ 0 | ~~1~~ 0 | 0 | 5 |
| Code Completeness | 0 | 1 | 0 | 1 | 0 |
| CI/CD | 0 | ~~1~~ 0 | ~~1~~ 0 | 0 | 2 |
| **Total** | **1** | **3** | **0** | **1** | **11** |

### Fixed This Session
- ✅ Hardcoded Grafana password (docker-compose.yml)
- ✅ Auth disabled by default (.env.example)
- ✅ No pre-commit hooks (.pre-commit-config.yaml)
- ✅ Windows-only quickstart (START_HERE_LINUX.md, build.sh, start.sh)
- ✅ Missing test modules (test_utils.py, test_observability.py)
- ✅ "Coming Soon" docs updated (security.md)
- ✅ Alpha release notes (ALPHA_RELEASE_NOTES.md)
- ✅ CI/CD pipeline verification
- ✅ Enhanced security workflow
- ✅ Troubleshooting docs (RUNNING_AND_COMPILING.md)
- ✅ Hardware check script (scripts/check_requirements.py)
- ✅ Tox configuration (tox.ini)
- ✅ Coverage configuration (.coveragerc)

---

*Last Updated: January 2026*
*Maintained By: Agent OS Team*
