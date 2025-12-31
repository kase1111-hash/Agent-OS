"""
Memory System Benchmarks

Performance benchmarks for the memory vault, storage, and retrieval systems.
Tests encrypted storage, consent management, and memory search.

Target: Memory operations should complete in <100ms for typical operations.
"""

import os
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime, timedelta

import pytest

from src.memory.vault import MemoryVault, MemoryEntry, MemoryClass
from src.memory.storage import MemoryStorage
from src.memory.consent import ConsentManager, ConsentRecord, ConsentType
from src.memory.keys import KeyManager
from src.memory.index import MemoryIndex


class TestMemoryVaultBenchmarks:
    """Benchmarks for the encrypted memory vault."""

    @pytest.mark.memory
    def test_vault_initialization(self, benchmark, temp_dir: Path) -> None:
        """Benchmark vault initialization with key generation.

        Target: <200ms for cold start with key generation.
        """
        vault_dir = temp_dir / "vault_init"

        def init_vault() -> MemoryVault:
            return MemoryVault(
                storage_path=vault_dir,
                master_key=os.urandom(32),
            )

        result = benchmark(init_vault)
        assert result is not None

    @pytest.mark.memory
    def test_vault_store_entry(self, benchmark, temp_dir: Path) -> None:
        """Benchmark storing a single memory entry.

        Target: <50ms per store operation (including encryption).
        """
        vault = MemoryVault(
            storage_path=temp_dir / "vault_store",
            master_key=os.urandom(32),
        )

        entry = MemoryEntry(
            id="test-entry-001",
            content="This is a test memory entry with some content",
            memory_class=MemoryClass.WORKING,
            user_id="user-123",
            created_at=datetime.utcnow(),
            metadata={"source": "benchmark", "tags": ["test", "performance"]},
        )

        result = benchmark(vault.store, entry)
        assert result is True or result is None

    @pytest.mark.memory
    def test_vault_store_many(self, benchmark, temp_dir: Path) -> None:
        """Benchmark storing many memory entries.

        Target: 100 entries in <2 seconds.
        """
        vault = MemoryVault(
            storage_path=temp_dir / "vault_many",
            master_key=os.urandom(32),
        )

        entries = [
            MemoryEntry(
                id=f"batch-entry-{i:04d}",
                content=f"Memory entry content number {i} with additional text for realistic size",
                memory_class=MemoryClass.WORKING if i % 3 != 0 else MemoryClass.LONG_TERM,
                user_id="user-123",
                created_at=datetime.utcnow(),
                metadata={"index": i, "batch": "benchmark"},
            )
            for i in range(100)
        ]

        def store_all() -> int:
            stored = 0
            for entry in entries:
                vault.store(entry)
                stored += 1
            return stored

        result = benchmark(store_all)
        assert result == 100

    @pytest.mark.memory
    def test_vault_retrieve(self, benchmark, temp_dir: Path) -> None:
        """Benchmark retrieving a memory entry.

        Target: <20ms per retrieval (including decryption).
        """
        vault = MemoryVault(
            storage_path=temp_dir / "vault_retrieve",
            master_key=os.urandom(32),
        )

        # Store entries first
        for i in range(50):
            entry = MemoryEntry(
                id=f"retrieve-{i:04d}",
                content=f"Retrievable content {i}",
                memory_class=MemoryClass.WORKING,
                user_id="user-123",
            )
            vault.store(entry)

        # Benchmark retrieval
        result = benchmark(vault.retrieve, "retrieve-0025")

        assert result is not None
        assert result.id == "retrieve-0025"


class TestMemoryStorageBenchmarks:
    """Benchmarks for raw memory storage operations."""

    @pytest.mark.memory
    def test_storage_write(self, benchmark, temp_dir: Path) -> None:
        """Benchmark raw storage write.

        Target: 1000 writes in <500ms.
        """
        storage = MemoryStorage(storage_path=temp_dir / "storage_write")

        def write_many() -> int:
            written = 0
            for i in range(1000):
                data = {
                    "id": f"data-{i:06d}",
                    "content": f"Storage content {i}",
                    "timestamp": datetime.utcnow().isoformat(),
                }
                storage.write(f"data-{i:06d}", data)
                written += 1
            return written

        result = benchmark(write_many)
        assert result == 1000

    @pytest.mark.memory
    def test_storage_read(self, benchmark, temp_dir: Path) -> None:
        """Benchmark raw storage read.

        Target: 1000 reads in <200ms.
        """
        storage = MemoryStorage(storage_path=temp_dir / "storage_read")

        # Write data first
        for i in range(1000):
            data = {"id": f"read-{i:06d}", "content": f"Content {i}"}
            storage.write(f"read-{i:06d}", data)

        def read_many() -> int:
            read_count = 0
            for i in range(1000):
                data = storage.read(f"read-{i:06d}")
                if data:
                    read_count += 1
            return read_count

        result = benchmark(read_many)
        assert result == 1000


class TestConsentBenchmarks:
    """Benchmarks for consent management."""

    @pytest.mark.memory
    def test_consent_check(self, benchmark, temp_dir: Path) -> None:
        """Benchmark consent verification.

        Target: <5ms per consent check.
        """
        manager = ConsentManager(storage_path=temp_dir / "consent")

        # Grant some consents
        for i in range(100):
            record = ConsentRecord(
                id=f"consent-{i:04d}",
                user_id="user-123",
                consent_type=ConsentType.LONG_TERM_MEMORY,
                granted=True,
                granted_at=datetime.utcnow(),
                scope=f"memory:category-{i % 10}",
            )
            manager.record_consent(record)

        # Benchmark checking
        result = benchmark(
            manager.check_consent,
            user_id="user-123",
            consent_type=ConsentType.LONG_TERM_MEMORY,
            scope="memory:category-5",
        )

        assert result is True

    @pytest.mark.memory
    def test_consent_bulk_check(self, benchmark, temp_dir: Path) -> None:
        """Benchmark checking many consent records.

        Target: 1000 checks in <500ms.
        """
        manager = ConsentManager(storage_path=temp_dir / "consent_bulk")

        # Grant consents
        for i in range(100):
            record = ConsentRecord(
                id=f"bulk-consent-{i:04d}",
                user_id="user-123",
                consent_type=ConsentType.LONG_TERM_MEMORY,
                granted=i % 5 != 0,  # 80% granted
                granted_at=datetime.utcnow(),
                scope=f"memory:scope-{i}",
            )
            manager.record_consent(record)

        def check_many() -> int:
            granted = 0
            for i in range(1000):
                if manager.check_consent(
                    user_id="user-123",
                    consent_type=ConsentType.LONG_TERM_MEMORY,
                    scope=f"memory:scope-{i % 100}",
                ):
                    granted += 1
            return granted

        result = benchmark(check_many)
        assert result >= 0


class TestMemoryIndexBenchmarks:
    """Benchmarks for memory indexing and search."""

    @pytest.mark.memory
    def test_index_add(self, benchmark, temp_dir: Path) -> None:
        """Benchmark adding entries to the index.

        Target: 1000 index additions in <1 second.
        """
        index = MemoryIndex(index_path=temp_dir / "index_add")

        entries = [
            {
                "id": f"idx-{i:06d}",
                "content": f"This is searchable content number {i} with keywords like python, agent, memory",
                "tags": ["test", f"batch-{i % 10}"],
                "timestamp": datetime.utcnow().isoformat(),
            }
            for i in range(1000)
        ]

        def index_all() -> int:
            indexed = 0
            for entry in entries:
                index.add(entry["id"], entry)
                indexed += 1
            return indexed

        result = benchmark(index_all)
        assert result == 1000

    @pytest.mark.memory
    def test_index_search(self, benchmark, temp_dir: Path) -> None:
        """Benchmark memory search.

        Target: <50ms per search query.
        """
        index = MemoryIndex(index_path=temp_dir / "index_search")

        # Populate index
        for i in range(500):
            entry = {
                "id": f"search-{i:04d}",
                "content": f"Document about {'python programming' if i % 5 == 0 else 'general topics'} number {i}",
                "tags": ["indexed"],
            }
            index.add(entry["id"], entry)

        # Benchmark search
        result = benchmark(index.search, "python programming")

        assert isinstance(result, list)

    @pytest.mark.memory
    def test_index_search_with_filters(self, benchmark, temp_dir: Path) -> None:
        """Benchmark filtered memory search.

        Target: <100ms per filtered search.
        """
        index = MemoryIndex(index_path=temp_dir / "index_filter")

        # Populate index
        for i in range(500):
            entry = {
                "id": f"filter-{i:04d}",
                "content": f"Filterable content {i}",
                "category": f"cat-{i % 10}",
                "priority": i % 5,
            }
            index.add(entry["id"], entry)

        filters = {"category": "cat-5", "priority": 2}

        result = benchmark(index.search, "content", filters=filters)

        assert isinstance(result, list)


class TestKeyManagementBenchmarks:
    """Benchmarks for cryptographic key operations."""

    @pytest.mark.memory
    def test_key_derivation(self, benchmark, temp_dir: Path) -> None:
        """Benchmark key derivation.

        Target: <100ms per key derivation (security vs performance tradeoff).
        """
        manager = KeyManager(key_path=temp_dir / "keys")
        master_key = os.urandom(32)

        result = benchmark(
            manager.derive_key,
            master_key,
            purpose="memory-encryption",
            context=b"benchmark-context",
        )

        assert result is not None
        assert len(result) == 32

    @pytest.mark.memory
    def test_key_rotation(self, benchmark, temp_dir: Path) -> None:
        """Benchmark key rotation.

        Target: <500ms for key rotation.
        """
        manager = KeyManager(key_path=temp_dir / "keys_rotation")
        manager.initialize(os.urandom(32))

        result = benchmark(manager.rotate_keys)

        assert result is True or result is None
