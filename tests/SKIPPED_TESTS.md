# Skipped Tests Documentation

This document explains why certain tests are conditionally skipped in the Agent OS test suite.

## Overview

Some tests require optional dependencies or external services that may not be available in all environments. These tests use `pytest.mark.skipif` or `pytest.skip()` to gracefully skip when dependencies are missing.

**CI Configuration:** The CI pipeline includes two test jobs:
- `test`: Runs with core dependencies only (some tests will skip)
- `test-full`: Runs with all optional dependencies installed (validates skipped tests pass)

---

## Test Files with Conditional Skips

### test_web.py (~13 skips)

**Skip Condition:** `not FASTAPI_AVAILABLE`

**Reason:** Web API tests require FastAPI and the test client (httpx) to be installed.

**Dependencies:**
- `fastapi` - Web framework
- `httpx` - Async HTTP client for TestClient

**Install:**
```bash
pip install fastapi httpx
# or
pip install -e ".[dev]"
```

**Tests Affected:**
- `TestChatAPI` - Chat endpoint tests
- `TestIntentAPI` - Intent detection tests
- `TestWebApp` - Application initialization tests
- `TestAPIRoutes` - Route registration tests

---

### test_pq_keys.py (~23 skips)

**Skip Condition:** `not _check_liboqs_available()` (module-level)

**Reason:** Post-quantum cryptography tests require the liboqs library with Kyber768 KEM support.

**Dependencies:**
- `liboqs` - Open Quantum Safe library (system-level)
- `liboqs-python` - Python bindings for liboqs

**Install (Linux):**
```bash
# System dependencies
sudo apt-get install cmake ninja-build libssl-dev

# Build liboqs
git clone https://github.com/open-quantum-safe/liboqs.git
cd liboqs && mkdir build && cd build
cmake -GNinja -DBUILD_SHARED_LIBS=ON ..
ninja && sudo ninja install && sudo ldconfig

# Python bindings
pip install liboqs-python
```

**Tests Affected:**
- `TestPQKeyManagerBasics` - Basic key manager tests
- `TestKEMKeyGeneration` - Key encapsulation tests
- `TestSigningKeyGeneration` - Digital signature tests
- `TestKeyRetrieval` - Key storage/retrieval tests
- `TestKeyOperations` - Encapsulation/decapsulation tests
- `TestKeyLifecycle` - Key rotation/revocation tests
- `TestKeyListing` - Key enumeration tests
- `TestPersistence` - Key persistence tests
- `TestPQKeyMetadata` - Metadata serialization tests

---

### test_value_ledger.py (~5 skips)

**Skip Condition:** `pytest.skip("value-ledger module not available")` (in-test)

**Reason:** Value ledger integration tests require the external value-ledger module.

**Dependencies:**
- `value-ledger` - External ledger module (not in standard requirements)

**Tests Affected:**
- `TestValueLedgerCore.test_ledger_store_create`
- `TestValueLedgerCore.test_ledger_store_append`
- `TestValueLedgerCore.test_ledger_chain_verify`
- `TestMerkleProofs.test_merkle_tree_build`
- `TestMerkleProofs.test_merkle_proof_generate`
- `TestMerkleProofs.test_merkle_proof_verify`

**Note:** These tests require the value-ledger repository to be cloned alongside Agent-OS. They validate the integration but are not required for core functionality.

---

## Expected Skips in Standard CI

When running `pytest` with standard dependencies:

| Test File | Expected Skips | Reason |
|-----------|----------------|--------|
| test_web.py | ~13 | FastAPI not in core deps |
| test_pq_keys.py | ~23 | liboqs requires system install |
| test_value_ledger.py | ~5 | External module |
| **Total** | **~41** | |

---

## Running Tests with Full Dependencies

To run all tests without skips:

```bash
# 1. Install liboqs (see above)

# 2. Install all Python dependencies
pip install -e ".[dev,redis,observability]"
pip install httpx liboqs-python

# 3. Clone value-ledger (optional)
git clone https://github.com/kase1111-hash/value-ledger.git ../value-ledger

# 4. Run tests
pytest tests/ -v
```

---

## Adding New Conditional Tests

When adding tests that require optional dependencies:

1. **Use module-level skip for entire test files:**
   ```python
   import pytest

   try:
       import optional_dep
       AVAILABLE = True
   except ImportError:
       AVAILABLE = False

   pytestmark = pytest.mark.skipif(
       not AVAILABLE,
       reason="optional_dep required for these tests"
   )
   ```

2. **Use in-test skip for specific tests:**
   ```python
   def test_feature(self, dep_available):
       if not dep_available:
           pytest.skip("dependency not available")
       # test code
   ```

3. **Document the skip in this file.**

4. **Update CI `test-full` job if new system dependencies needed.**

---

*Last Updated: January 2026*
