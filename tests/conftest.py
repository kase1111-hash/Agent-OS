"""
Pytest Configuration and Shared Fixtures

This module provides centralized fixtures for testing Agent OS components.
It integrates with the dependency injection system for clean test isolation.
"""

import os
import tempfile
from pathlib import Path
from typing import Generator
from unittest.mock import MagicMock

import pytest


# =============================================================================
# Environment Setup
# =============================================================================


@pytest.fixture(autouse=True)
def clean_environment():
    """Ensure clean environment for each test."""
    # Store original environment
    original_env = os.environ.copy()

    yield

    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)


@pytest.fixture(autouse=True)
def reset_dependencies():
    """Reset all dependency injection state between tests."""
    yield

    # Reset after test
    try:
        from src.web.dependencies import reset_all_dependencies

        reset_all_dependencies()
    except ImportError:
        pass  # Dependencies module not available


# =============================================================================
# Temporary Directory Fixtures
# =============================================================================


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Provide a temporary directory that's cleaned up after the test."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def temp_db_path(temp_dir: Path) -> Path:
    """Provide a temporary database path."""
    return temp_dir / "test.db"


# =============================================================================
# Configuration Fixtures
# =============================================================================


@pytest.fixture
def mock_config():
    """Provide a mock WebConfig for testing."""
    from src.web.config import WebConfig, VoiceConfig

    return WebConfig(
        host="127.0.0.1",
        port=8080,
        debug=True,
        require_auth=False,
        rate_limit_enabled=False,
        voice=VoiceConfig(
            stt_enabled=False,
            tts_enabled=False,
        ),
    )


@pytest.fixture
def test_config(temp_dir: Path):
    """Provide a test WebConfig with temporary directories."""
    from src.web.config import WebConfig, VoiceConfig

    return WebConfig(
        host="127.0.0.1",
        port=8080,
        debug=True,
        require_auth=False,
        rate_limit_enabled=False,
        static_dir=temp_dir / "static",
        templates_dir=temp_dir / "templates",
        data_dir=temp_dir / "data",
        voice=VoiceConfig(
            stt_enabled=False,
            tts_enabled=False,
        ),
    )


# =============================================================================
# Dependency Override Fixtures
# =============================================================================


@pytest.fixture
def dependency_overrides():
    """
    Provide a context manager for overriding dependencies in tests.

    Usage:
        def test_example(dependency_overrides):
            with dependency_overrides as overrides:
                overrides.set_config(mock_config)
                # Test with mocked config
    """
    from src.web.dependencies import DependencyOverrides

    return DependencyOverrides()


@pytest.fixture
def override_config(test_config, dependency_overrides):
    """
    Automatically override config for the duration of a test.

    Usage:
        def test_example(override_config):
            # Config is already overridden with test_config
            config = get_config()
            assert config.debug is True
    """
    with dependency_overrides as overrides:
        overrides.set_config(test_config)
        yield test_config


# =============================================================================
# Mock Store Fixtures
# =============================================================================


@pytest.fixture
def mock_user_store():
    """Provide a mock UserStore."""
    store = MagicMock()
    store.initialize.return_value = True
    store.authenticate_user.return_value = None
    store.get_user.return_value = None
    return store


@pytest.fixture
def mock_conversation_store():
    """Provide a mock ConversationStore."""
    store = MagicMock()
    store.initialize.return_value = True
    store.get_conversation.return_value = None
    store.list_conversations.return_value = []
    return store


@pytest.fixture
def mock_rate_limiter():
    """Provide a mock RateLimiter that always allows requests."""
    limiter = MagicMock()
    limiter.check_limit.return_value = (True, None)
    limiter.is_allowed.return_value = True
    return limiter


# =============================================================================
# Real Store Fixtures (with temp directories)
# =============================================================================


@pytest.fixture
def user_store(temp_dir: Path):
    """Provide a real UserStore with temporary database."""
    from src.web.auth import UserStore

    db_path = temp_dir / "users.db"
    store = UserStore(db_path)
    store.initialize()
    yield store
    store.close()


@pytest.fixture
def conversation_store(temp_dir: Path):
    """Provide a real ConversationStore with temporary database."""
    from src.web.conversation_store import ConversationStore

    db_path = temp_dir / "conversations.db"
    store = ConversationStore(db_path)
    store.initialize()
    yield store
    store.close()


# =============================================================================
# FastAPI Test Client Fixtures
# =============================================================================


@pytest.fixture
def test_app(test_config, dependency_overrides):
    """
    Provide a FastAPI test application with mocked dependencies.

    Usage:
        def test_endpoint(test_app):
            from fastapi.testclient import TestClient
            client = TestClient(test_app)
            response = client.get("/api/health")
    """
    with dependency_overrides as overrides:
        overrides.set_config(test_config)

        from src.web.app import get_app

        app = get_app()
        yield app


# =============================================================================
# Async Fixtures
# =============================================================================


@pytest.fixture
def event_loop():
    """Create an instance of the default event loop for the test session."""
    import asyncio

    loop = asyncio.new_event_loop()
    yield loop
    loop.close()
