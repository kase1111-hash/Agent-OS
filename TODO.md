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

### 🟡 High: Missing Test Modules
No dedicated test files for these modules:

- [ ] `tests/test_utils.py` - Cover `src/utils/` functions
- [ ] `tests/test_observability.py` - Cover `src/observability/` metrics/tracing
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

### 🟡 High: "Coming Soon" Items in Docs
Items marked "coming soon" in `docs/governance/security.md`:
- [ ] `security@agentos.org` email - Set up or remove reference
- [ ] PGP key for security reporting - Create and publish
- [ ] Discord security DMs - Set up or clarify timeline
- [ ] Signal/Matrix channels - Set up or clarify timeline

### 🟡 High: Alpha Release Notes
- [ ] Create `ALPHA_RELEASE_NOTES.md` documenting:
  - Known limitations
  - Features not yet implemented (HSM, certificate upgrade)
  - Hardware requirements
  - Supported platforms
  - Breaking changes expected before 1.0

### 🟢 Medium: Installation Docs Improvements
- [ ] Add troubleshooting section to installation docs
- [ ] Add hardware requirement verification script
- [ ] Document common error messages and solutions
- [ ] Add video walkthrough links (placeholder for now)

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

### 🟡 High: Missing CI Pipeline
- [ ] Create `.github/workflows/ci.yml` with:
  - Python 3.10, 3.11, 3.12 matrix
  - Lint checks (black, isort, flake8, mypy)
  - Unit tests with coverage
  - Integration tests
  - Docker build verification
- [ ] Create `.github/workflows/security.yml` with:
  - Dependency vulnerability scanning
  - Secret detection
  - SAST scanning

### 🟢 Medium: Additional CI Improvements
- [ ] Create `tox.ini` for multi-version testing
- [ ] Add test coverage reporting (codecov)
- [ ] Add performance benchmarks
- [ ] Add documentation build verification

---

## Tracking

| Category | Critical | High | Medium | Defer | Fixed |
|----------|----------|------|--------|-------|-------|
| Testing | 1 | 2 | 0 | 0 | 0 |
| Security Config | ~~2~~ 0 | ~~2~~ 1 | 0 | 0 | 3 |
| Documentation | ~~1~~ 0 | 3 | 1 | 0 | 1 |
| Code Completeness | 0 | 1 | 0 | 1 | 0 |
| CI/CD | 0 | 1 | 1 | 0 | 0 |
| **Total** | **1** | **8** | **2** | **1** | **4** |

### Fixed This Session
- ✅ Hardcoded Grafana password (docker-compose.yml)
- ✅ Auth disabled by default (.env.example)
- ✅ No pre-commit hooks (.pre-commit-config.yaml)
- ✅ Windows-only quickstart (START_HERE_LINUX.md, build.sh, start.sh)

---

*Last Updated: January 2026*
*Maintained By: Agent OS Team*
