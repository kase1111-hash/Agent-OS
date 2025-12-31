"""
Benchmark Configuration and Fixtures

Shared fixtures and configuration for performance benchmarks.
"""

import tempfile
from pathlib import Path
from typing import Generator

import pytest


# Sample constitution content for benchmarks
SAMPLE_CONSTITUTION = """---
title: Test Constitution
version: "1.0.0"
authority: constitutional
immutable_sections: [1, 3]
---

# Article I: Core Principles

## Section 1.1: Human Sovereignty (immutable)

The human user shall always maintain ultimate authority over the AI system.
All AI actions must be traceable and auditable.

## Section 1.2: Transparency

The AI must not hide its reasoning or decision-making process.
Users may inspect any stored data about themselves.

# Article II: Agent Behavior

## Section 2.1: Mandate - Response Requirements

Agents shall respond to user queries within reasonable time limits.
Agents must acknowledge receipt of requests.

## Section 2.2: Prohibition - Harmful Actions

Agents shall not execute code that could harm the system.
Agents must not access files outside designated directories.
Agents shall not share user data without explicit consent.

## Section 2.3: Permission - Optional Capabilities

Agents may cache frequently accessed data for performance.
Agents may suggest related topics to users.

# Article III: Data Handling (immutable)

## Section 3.1: Privacy

User data shall remain on local storage unless explicitly shared.
Encryption must be used for sensitive data at rest.

## Section 3.2: Retention

Data older than the retention period may be automatically archived.
Users can request immediate deletion of their data.

# Article IV: Escalation

## Section 4.1: Human Override

When uncertainty exceeds threshold, escalate to human steward.
Critical decisions require human confirmation.
"""

LARGE_CONSTITUTION = SAMPLE_CONSTITUTION * 10  # For stress testing


@pytest.fixture
def sample_constitution() -> str:
    """Return a sample constitution string."""
    return SAMPLE_CONSTITUTION


@pytest.fixture
def large_constitution() -> str:
    """Return a large constitution for stress testing."""
    return LARGE_CONSTITUTION


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for benchmark tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def constitution_file(temp_dir: Path, sample_constitution: str) -> Path:
    """Create a temporary constitution file."""
    file_path = temp_dir / "CONSTITUTION.md"
    file_path.write_text(sample_constitution)
    return file_path


@pytest.fixture
def large_constitution_file(temp_dir: Path, large_constitution: str) -> Path:
    """Create a large temporary constitution file for stress testing."""
    file_path = temp_dir / "LARGE_CONSTITUTION.md"
    file_path.write_text(large_constitution)
    return file_path


# Benchmark groups for organization
def pytest_configure(config: pytest.Config) -> None:
    """Register custom markers for benchmark categories."""
    config.addinivalue_line("markers", "core: Core component benchmarks (parser, validator)")
    config.addinivalue_line("markers", "kernel: Kernel engine benchmarks")
    config.addinivalue_line("markers", "agents: Agent processing benchmarks")
    config.addinivalue_line("markers", "memory: Memory system benchmarks")
    config.addinivalue_line("markers", "integration: Integration benchmarks")
