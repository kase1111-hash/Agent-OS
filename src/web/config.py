"""
Web Interface Configuration

Configuration settings for the Agent OS web interface.
"""

import logging
import os
import secrets
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)


class ConfigurationError(Exception):
    """Raised when configuration validation fails."""

    pass


def _parse_cors_origins(value: str) -> List[str]:
    """Parse CORS origins from comma-separated env var. Returns default if empty."""
    if not value or not value.strip():
        return ["http://localhost:8080"]
    return [origin.strip() for origin in value.split(",") if origin.strip()]


@dataclass
class WebConfig:
    """Configuration for the web interface."""

    # Server settings
    host: str = "127.0.0.1"
    port: int = 8080
    debug: bool = False

    # Security
    cors_origins: List[str] = field(default_factory=lambda: ["http://localhost:8080"])
    cors_methods: List[str] = field(
        default_factory=lambda: ["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"]
    )
    cors_headers: List[str] = field(
        default_factory=lambda: ["Authorization", "Content-Type", "Accept", "X-Requested-With"]
    )
    api_key: Optional[str] = None
    require_auth: bool = True  # SECURITY: Auth enabled by default; set to False only for local dev

    # HTTPS/TLS settings
    force_https: bool = True  # SECURITY: HTTPS enforced by default; set to False only for local dev
    hsts_enabled: bool = False  # HTTP Strict Transport Security
    hsts_max_age: int = 31536000  # 1 year in seconds
    hsts_include_subdomains: bool = True
    hsts_preload: bool = False

    # Paths
    static_dir: Path = field(default_factory=lambda: Path(__file__).parent / "static")
    templates_dir: Path = field(default_factory=lambda: Path(__file__).parent / "templates")
    data_dir: Path = field(default_factory=lambda: Path(__file__).parent / "data")

    # WebSocket settings
    ws_heartbeat_interval: int = 30  # seconds
    ws_max_connections: int = 100

    # Rate limiting
    rate_limit_enabled: bool = True
    rate_limit_requests_per_minute: int = 60
    rate_limit_requests_per_hour: int = 1000
    rate_limit_strategy: str = "sliding_window"  # fixed_window, sliding_window, token_bucket
    rate_limit_use_redis: bool = False
    rate_limit_redis_url: str = "redis://localhost:6379"

    # Session
    session_timeout: int = 3600  # seconds

    def validate(self) -> None:
        """
        Validate configuration settings.

        Raises:
            ConfigurationError: If configuration is invalid
        """
        # If auth is required, API key must be set
        if self.require_auth and not self.api_key:
            raise ConfigurationError(
                "AGENT_OS_API_KEY must be set when AGENT_OS_REQUIRE_AUTH=true. "
                "Generate a secure key with: python -c \"import secrets; print(secrets.token_urlsafe(32))\""
            )

        # Warn if API key is too short
        if self.api_key and len(self.api_key) < 16:
            logger.warning(
                "API key is shorter than recommended (16+ characters). "
                "Consider using a longer key for better security."
            )

        # Warn if auth is disabled in non-debug mode
        if not self.require_auth and not self.debug:
            logger.warning(
                "Authentication is disabled. Set AGENT_OS_REQUIRE_AUTH=true "
                "and AGENT_OS_API_KEY for production use."
            )

        # Warn if HTTPS is not enforced in non-debug mode with auth
        if self.require_auth and not self.force_https and not self.debug:
            logger.warning(
                "HTTPS enforcement is disabled while authentication is enabled. "
                "Set AGENT_OS_FORCE_HTTPS=true for production use to protect credentials."
            )

        # Warn if HSTS is enabled without HTTPS
        if self.hsts_enabled and not self.force_https:
            logger.warning(
                "HSTS is enabled but HTTPS enforcement is disabled. "
                "HSTS headers will only be effective over HTTPS connections."
            )

    @classmethod
    def from_env(cls, validate: bool = True) -> "WebConfig":
        """
        Create configuration from environment variables.

        Args:
            validate: If True, validate configuration after loading

        Returns:
            WebConfig instance

        Raises:
            ConfigurationError: If validation fails and validate=True
        """
        config = cls(
            host=os.getenv("AGENT_OS_WEB_HOST", "127.0.0.1"),
            port=int(os.getenv("AGENT_OS_WEB_PORT", "8080")),
            debug=os.getenv("AGENT_OS_WEB_DEBUG", "").lower() in ("1", "true", "yes"),
            api_key=os.getenv("AGENT_OS_API_KEY"),
            require_auth=os.getenv("AGENT_OS_REQUIRE_AUTH", "").lower() in ("1", "true", "yes"),
            # HTTPS/TLS settings
            force_https=os.getenv("AGENT_OS_FORCE_HTTPS", "").lower() in ("1", "true", "yes"),
            hsts_enabled=os.getenv("AGENT_OS_HSTS_ENABLED", "").lower() in ("1", "true", "yes"),
            hsts_max_age=int(os.getenv("AGENT_OS_HSTS_MAX_AGE", "31536000")),
            hsts_include_subdomains=os.getenv("AGENT_OS_HSTS_INCLUDE_SUBDOMAINS", "true").lower()
            in ("1", "true", "yes"),
            hsts_preload=os.getenv("AGENT_OS_HSTS_PRELOAD", "").lower() in ("1", "true", "yes"),
            # Rate limiting
            rate_limit_enabled=os.getenv("AGENT_OS_RATE_LIMIT_ENABLED", "true").lower()
            in ("1", "true", "yes"),
            rate_limit_requests_per_minute=int(os.getenv("AGENT_OS_RATE_LIMIT_PER_MINUTE", "60")),
            rate_limit_requests_per_hour=int(os.getenv("AGENT_OS_RATE_LIMIT_PER_HOUR", "1000")),
            rate_limit_strategy=os.getenv("AGENT_OS_RATE_LIMIT_STRATEGY", "sliding_window"),
            rate_limit_use_redis=os.getenv("AGENT_OS_RATE_LIMIT_REDIS", "").lower()
            in ("1", "true", "yes"),
            rate_limit_redis_url=os.getenv("AGENT_OS_REDIS_URL", "redis://localhost:6379"),
            cors_origins=_parse_cors_origins(os.getenv("AGENT_OS_CORS_ORIGINS", "")),
        )

        if validate:
            config.validate()

        return config


# =============================================================================
# Dependency Injection Integration
# =============================================================================

# Import from dependencies module for DI-based access
# These are the preferred methods for accessing configuration


def get_config() -> WebConfig:
    """
    Get the web configuration.

    This function integrates with the dependency injection system.
    For FastAPI routes, use Depends(get_config) instead.
    """
    from src.web.dependencies import get_config as _get_config_di

    return _get_config_di()


def set_config(config: WebConfig) -> None:
    """
    Set/override the web configuration.

    Primarily used for testing. In production, configuration
    is loaded from environment variables.
    """
    from src.web.dependencies import _container

    _container.set_override("config", config)


def reset_config() -> None:
    """
    Reset the configuration to reload from environment.

    Useful for tests that need fresh configuration.
    """
    from src.web.dependencies import reset_dependencies

    reset_dependencies("config")


def generate_api_key(length: int = 32) -> str:
    """
    Generate a secure API key.

    Args:
        length: Length of the key in bytes (default 32 = 43 characters)

    Returns:
        URL-safe base64 encoded API key

    Usage:
        python -c "from src.web.config import generate_api_key; print(generate_api_key())"
    """
    return secrets.token_urlsafe(length)
