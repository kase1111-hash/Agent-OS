"""
Web Interface Configuration

Configuration settings for the Agent OS web interface.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class VoiceConfig:
    """Configuration for voice (STT/TTS) features."""

    # STT settings
    stt_enabled: bool = True
    stt_engine: str = "auto"  # auto, whisper, whisper_api, mock
    stt_model: str = "base"  # tiny, base, small, medium, large
    stt_language: str = "en"

    # TTS settings
    tts_enabled: bool = True
    tts_engine: str = "auto"  # auto, piper, espeak, mock
    tts_voice: str = "en_US-lessac-medium"
    tts_speed: float = 1.0


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

    # Voice settings
    voice: VoiceConfig = field(default_factory=VoiceConfig)

    @classmethod
    def from_env(cls) -> "WebConfig":
        """Create configuration from environment variables."""
        voice_config = VoiceConfig(
            stt_enabled=os.getenv("AGENT_OS_STT_ENABLED", "true").lower() in ("1", "true", "yes"),
            stt_engine=os.getenv("AGENT_OS_STT_ENGINE", "auto"),
            stt_model=os.getenv("AGENT_OS_STT_MODEL", "base"),
            stt_language=os.getenv("AGENT_OS_STT_LANGUAGE", "en"),
            tts_enabled=os.getenv("AGENT_OS_TTS_ENABLED", "true").lower() in ("1", "true", "yes"),
            tts_engine=os.getenv("AGENT_OS_TTS_ENGINE", "auto"),
            tts_voice=os.getenv("AGENT_OS_TTS_VOICE", "en_US-lessac-medium"),
            tts_speed=float(os.getenv("AGENT_OS_TTS_SPEED", "1.0")),
        )

        return cls(
            host=os.getenv("AGENT_OS_WEB_HOST", "127.0.0.1"),
            port=int(os.getenv("AGENT_OS_WEB_PORT", "8080")),
            debug=os.getenv("AGENT_OS_WEB_DEBUG", "").lower() in ("1", "true", "yes"),
            api_key=os.getenv("AGENT_OS_API_KEY"),
            require_auth=os.getenv("AGENT_OS_REQUIRE_AUTH", "").lower() in ("1", "true", "yes"),
            voice=voice_config,
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
