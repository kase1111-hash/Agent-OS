"""
Agent OS Web Interface

Provides web-based UI for Agent OS including:
- Chat interface with WebSocket support
- Agent monitoring dashboard
- Visual constitutional editor
- Memory management UI

Stack: FastAPI + Jinja2 + vanilla JavaScript
"""

from .app import (
    create_app,
    get_app,
)
from .config import (
    WebConfig,
    get_config,
)

__all__ = [
    # Application
    "create_app",
    "get_app",
    # Configuration
    "WebConfig",
    "get_config",
]
