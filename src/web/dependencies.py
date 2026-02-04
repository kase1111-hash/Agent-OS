"""
FastAPI Dependency Injection Module

Provides centralized dependency injection for the Agent OS web interface.
This module enables:
- Clean separation of concerns
- Easy testing via dependency overrides
- Consistent instance management across routes

Usage in routes:
    from src.web.dependencies import get_config, get_rate_limiter, get_user_store

    @router.get("/example")
    async def example(
        config: WebConfig = Depends(get_config),
        limiter: RateLimiter = Depends(get_rate_limiter),
    ):
        ...

Usage in tests:
    from src.web.dependencies import DependencyOverrides

    with DependencyOverrides() as overrides:
        overrides.set_config(mock_config)
        overrides.set_rate_limiter(mock_limiter)
        # Run tests with mocked dependencies
"""

import logging
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Dict, Generator, Optional, Type, TypeVar

from fastapi import Request

logger = logging.getLogger(__name__)

# Type variable for generic dependency handling
T = TypeVar("T")


# =============================================================================
# Dependency Container
# =============================================================================


@dataclass
class _DependencyContainer:
    """
    Internal container for managing dependency instances.

    Provides lazy initialization and override capability for testing.
    """

    _instances: Dict[str, Any] = field(default_factory=dict)
    _overrides: Dict[str, Any] = field(default_factory=dict)
    _factories: Dict[str, Callable[[], Any]] = field(default_factory=dict)

    def register(self, name: str, factory: Callable[[], Any]) -> None:
        """Register a factory function for creating a dependency."""
        self._factories[name] = factory

    def get(self, name: str) -> Any:
        """
        Get a dependency instance.

        Order of resolution:
        1. Override (if set for testing)
        2. Cached instance
        3. Create new instance via factory
        """
        # Check overrides first (for testing)
        if name in self._overrides:
            return self._overrides[name]

        # Return cached instance if available
        if name in self._instances:
            return self._instances[name]

        # Create and cache new instance
        if name in self._factories:
            instance = self._factories[name]()
            self._instances[name] = instance
            return instance

        raise KeyError(f"Unknown dependency: {name}")

    def set_override(self, name: str, instance: Any) -> None:
        """Set an override for testing."""
        self._overrides[name] = instance

    def clear_override(self, name: str) -> None:
        """Clear a specific override."""
        self._overrides.pop(name, None)

    def clear_all_overrides(self) -> None:
        """Clear all overrides."""
        self._overrides.clear()

    def reset(self, name: Optional[str] = None) -> None:
        """
        Reset cached instances.

        Args:
            name: Specific dependency to reset, or None for all
        """
        if name:
            self._instances.pop(name, None)
        else:
            self._instances.clear()

    def reset_all(self) -> None:
        """Reset all instances and overrides."""
        self._instances.clear()
        self._overrides.clear()


# Global container instance
_container = _DependencyContainer()


# =============================================================================
# Dependency Registration
# =============================================================================


def _create_config():
    """Factory for WebConfig."""
    from src.web.config import WebConfig

    return WebConfig.from_env()


def _create_rate_limiter():
    """Factory for RateLimiter."""
    from src.web.ratelimit import RateLimiter, RateLimitStorage, create_rate_limiter

    config = _container.get("config")

    if config.rate_limit_use_redis:
        try:
            from src.web.ratelimit import RedisRateLimitStorage

            storage = RedisRateLimitStorage(config.rate_limit_redis_url)
            logger.info("Using Redis for rate limiting")
        except Exception as e:
            logger.warning(f"Failed to connect to Redis, using in-memory storage: {e}")
            storage = RateLimitStorage()
    else:
        storage = RateLimitStorage()

    return create_rate_limiter(
        requests_per_minute=config.rate_limit_requests_per_minute,
        requests_per_hour=config.rate_limit_requests_per_hour,
        strategy=config.rate_limit_strategy,
        storage=storage,
    )


def _create_user_store():
    """Factory for UserStore."""
    from src.web.auth import UserStore

    config = _container.get("config")
    db_path = config.data_dir / "users.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)

    store = UserStore(db_path)
    store.initialize()
    return store


def _create_conversation_store():
    """Factory for ConversationStore."""
    from src.web.conversation_store import ConversationStore

    config = _container.get("config")
    db_path = config.data_dir / "conversations.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)

    store = ConversationStore(db_path)
    store.initialize()
    return store


def _create_intent_log_store():
    """Factory for IntentLogStore."""
    from src.web.intent_log import IntentLogStore

    config = _container.get("config")
    db_path = config.data_dir / "intent_log.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)

    store = IntentLogStore(db_path)
    store.initialize()
    return store


# Register all factories
_container.register("config", _create_config)
_container.register("rate_limiter", _create_rate_limiter)
_container.register("user_store", _create_user_store)
_container.register("conversation_store", _create_conversation_store)
_container.register("intent_log_store", _create_intent_log_store)


# =============================================================================
# FastAPI Dependency Functions
# =============================================================================


def get_config():
    """
    FastAPI dependency for WebConfig.

    Usage:
        @router.get("/")
        async def handler(config: WebConfig = Depends(get_config)):
            ...
    """
    from src.web.config import WebConfig

    return _container.get("config")


def get_rate_limiter():
    """
    FastAPI dependency for RateLimiter.

    Usage:
        @router.get("/")
        async def handler(limiter: RateLimiter = Depends(get_rate_limiter)):
            ...
    """
    from src.web.ratelimit import RateLimiter

    return _container.get("rate_limiter")


def get_user_store():
    """
    FastAPI dependency for UserStore.

    Usage:
        @router.get("/")
        async def handler(store: UserStore = Depends(get_user_store)):
            ...
    """
    from src.web.auth import UserStore

    return _container.get("user_store")


def get_conversation_store():
    """
    FastAPI dependency for ConversationStore.

    Usage:
        @router.get("/")
        async def handler(store: ConversationStore = Depends(get_conversation_store)):
            ...
    """
    from src.web.conversation_store import ConversationStore

    return _container.get("conversation_store")


def get_intent_log_store():
    """
    FastAPI dependency for IntentLogStore.

    Usage:
        @router.get("/")
        async def handler(store: IntentLogStore = Depends(get_intent_log_store)):
            ...
    """
    from src.web.intent_log import IntentLogStore

    return _container.get("intent_log_store")


# =============================================================================
# Testing Utilities
# =============================================================================


class DependencyOverrides:
    """
    Context manager for temporarily overriding dependencies in tests.

    Usage:
        def test_example():
            mock_config = WebConfig(debug=True)
            mock_limiter = MockRateLimiter()

            with DependencyOverrides() as overrides:
                overrides.set_config(mock_config)
                overrides.set_rate_limiter(mock_limiter)

                # Dependencies now return mocked instances
                config = get_config()
                assert config.debug is True
    """

    def __init__(self):
        self._original_overrides: Dict[str, Any] = {}

    def __enter__(self) -> "DependencyOverrides":
        # Save current overrides
        self._original_overrides = dict(_container._overrides)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        # Restore original overrides
        _container._overrides.clear()
        _container._overrides.update(self._original_overrides)

    def set_config(self, config) -> "DependencyOverrides":
        """Override WebConfig."""
        _container.set_override("config", config)
        return self

    def set_rate_limiter(self, limiter) -> "DependencyOverrides":
        """Override RateLimiter."""
        _container.set_override("rate_limiter", limiter)
        return self

    def set_user_store(self, store) -> "DependencyOverrides":
        """Override UserStore."""
        _container.set_override("user_store", store)
        return self

    def set_conversation_store(self, store) -> "DependencyOverrides":
        """Override ConversationStore."""
        _container.set_override("conversation_store", store)
        return self

    def set_intent_log_store(self, store) -> "DependencyOverrides":
        """Override IntentLogStore."""
        _container.set_override("intent_log_store", store)
        return self

    def set(self, name: str, instance: Any) -> "DependencyOverrides":
        """Set any dependency override by name."""
        _container.set_override(name, instance)
        return self


def reset_dependencies(name: Optional[str] = None) -> None:
    """
    Reset cached dependency instances.

    Useful for tests that need fresh instances.

    Args:
        name: Specific dependency to reset, or None for all
    """
    _container.reset(name)


def reset_all_dependencies() -> None:
    """Reset all dependency instances and overrides."""
    _container.reset_all()


# =============================================================================
# Backward Compatibility Layer
# =============================================================================


def _get_config_compat():
    """Backward compatible get_config for modules that import from config.py."""
    return _container.get("config")


def _set_config_compat(config):
    """Backward compatible set_config for modules that import from config.py."""
    _container.set_override("config", config)


def _get_limiter_compat():
    """Backward compatible get_limiter for modules that import from ratelimit.py."""
    return _container.get("rate_limiter")


def _set_limiter_compat(limiter):
    """Backward compatible set_limiter for modules that import from ratelimit.py."""
    _container.set_override("rate_limiter", limiter)
