"""
Web Interface Configuration

Configuration settings for the Agent OS web interface.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class WebConfig:
    """Configuration for the web interface."""

    # Server settings
    host: str = "127.0.0.1"
    port: int = 8080
    debug: bool = False

    # Security
    cors_origins: List[str] = field(default_factory=lambda: ["http://localhost:8080"])
    api_key: Optional[str] = None
    require_auth: bool = False

    # Paths
    static_dir: Path = field(default_factory=lambda: Path(__file__).parent / "static")
    templates_dir: Path = field(default_factory=lambda: Path(__file__).parent / "templates")

    # WebSocket settings
    ws_heartbeat_interval: int = 30  # seconds
    ws_max_connections: int = 100

    # Rate limiting
    rate_limit_requests: int = 100
    rate_limit_window: int = 60  # seconds

    # Session
    session_timeout: int = 3600  # seconds

    @classmethod
    def from_env(cls) -> "WebConfig":
        """Create configuration from environment variables."""
        return cls(
            host=os.getenv("AGENT_OS_WEB_HOST", "127.0.0.1"),
            port=int(os.getenv("AGENT_OS_WEB_PORT", "8080")),
            debug=os.getenv("AGENT_OS_WEB_DEBUG", "").lower() in ("1", "true", "yes"),
            api_key=os.getenv("AGENT_OS_API_KEY"),
            require_auth=os.getenv("AGENT_OS_REQUIRE_AUTH", "").lower() in ("1", "true", "yes"),
        )


# Global configuration instance
_config: Optional[WebConfig] = None


def get_config() -> WebConfig:
    """Get the global web configuration."""
    global _config
    if _config is None:
        _config = WebConfig.from_env()
    return _config


def set_config(config: WebConfig) -> None:
    """Set the global web configuration."""
    global _config
    _config = config
