"""
FastAPI Application

Main application factory for the Agent OS web interface.
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, Optional

from .config import WebConfig, get_config

# API version
API_VERSION = "1.0.0"

# OpenAPI Tags with descriptions for documentation
OPENAPI_TAGS = [
    {
        "name": "Authentication",
        "description": "User authentication, registration, and session management. "
        "Supports both cookie-based sessions and Bearer token authentication.",
    },
    {
        "name": "Chat",
        "description": "Conversational AI interface. Send messages, manage conversations, "
        "and interact with AI agents through chat. Supports WebSocket for real-time streaming.",
    },
    {
        "name": "Agents",
        "description": "Monitor and manage AI agents. View agent status, metrics, logs, "
        "and control agent lifecycle (start, stop, restart).",
    },
    {
        "name": "Constitution",
        "description": "Constitutional governance management. View, search, and manage "
        "the rules that govern AI behavior. Core principles are immutable.",
    },
    {
        "name": "Contracts",
        "description": "Learning contracts and consent management. Create and manage "
        "user consent for data storage and AI learning capabilities.",
    },
    {
        "name": "Memory",
        "description": "User memory management with consent-based storage. "
        "View, search, export, and delete stored memories. Supports right to inspect and delete.",
    },
    {
        "name": "Images",
        "description": "AI image generation and gallery management. "
        "Generate images with various models and manage your image gallery.",
    },
    {
        "name": "Voice",
        "description": "Voice interface for speech-to-text and text-to-speech. "
        "Supports real-time voice streaming via WebSocket.",
    },
    {
        "name": "Intent Log",
        "description": "Immutable audit log of all AI decisions and actions. "
        "Provides transparency and accountability for AI behavior.",
    },
    {
        "name": "System",
        "description": "System status, health checks, and configuration. "
        "Monitor system resources and component health.",
    },
    {
        "name": "Observability",
        "description": "Metrics, tracing, and health monitoring endpoints. "
        "Prometheus-compatible metrics export at /api/observability/metrics.",
    },
    {
        "name": "Security",
        "description": "Attack detection, security monitoring, and remediation. "
        "View detected attacks, manage fix recommendations, and control the attack detection pipeline.",
    },
]

logger = logging.getLogger(__name__)


# Lazy imports to handle missing dependencies gracefully
def _check_dependencies() -> bool:
    """Check if required dependencies are installed."""
    try:
        import fastapi
        import uvicorn

        return True
    except ImportError:
        return False


# Application state
class AppState:
    """Application state container."""

    def __init__(self):
        self.config: Optional[WebConfig] = None
        self.agent_registry = None
        self.memory_store = None
        self.constitution_registry = None
        self.active_connections: Dict[str, Any] = {}


_app_state = AppState()
_app = None
_cleanup_task = None


async def _session_cleanup_task(interval_seconds: int = 3600):
    """
    Background task to periodically clean up expired sessions.

    Args:
        interval_seconds: How often to run cleanup (default: 1 hour)
    """
    while True:
        try:
            await asyncio.sleep(interval_seconds)

            # Clean up expired sessions
            from .auth import get_user_store

            store = get_user_store()
            cleaned = store.cleanup_expired_sessions()
            if cleaned > 0:
                logger.info(f"Session cleanup: removed {cleaned} expired sessions")

        except asyncio.CancelledError:
            logger.debug("Session cleanup task cancelled")
            break
        except Exception as e:
            logger.error(f"Session cleanup error: {e}")


def create_app(config: Optional[WebConfig] = None) -> Any:
    """
    Create the FastAPI application.

    Args:
        config: Optional configuration. Uses global config if not provided.

    Returns:
        FastAPI application instance

    Raises:
        ImportError: If FastAPI is not installed
    """
    if not _check_dependencies():
        raise ImportError(
            "FastAPI and uvicorn are required for the web interface. "
            "Install them with: pip install fastapi uvicorn[standard]"
        )

    from fastapi import FastAPI, Request
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.staticfiles import StaticFiles
    from fastapi.templating import Jinja2Templates

    global _app, _app_state

    if config is None:
        config = get_config()

    _app_state.config = config

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """Application lifespan handler."""
        global _cleanup_task

        logger.info("Starting Agent OS Web Interface...")

        # Start session cleanup background task
        _cleanup_task = asyncio.create_task(_session_cleanup_task())
        logger.info("Session cleanup task started (runs every hour)")

        yield

        logger.info("Shutting down Agent OS Web Interface...")

        # Stop session cleanup task
        if _cleanup_task:
            _cleanup_task.cancel()
            try:
                await _cleanup_task
            except asyncio.CancelledError:
                pass
            _cleanup_task = None

        # Clean up WebSocket connections
        for conn_id in list(_app_state.active_connections.keys()):
            try:
                ws = _app_state.active_connections.pop(conn_id)
                await ws.close()
            except Exception:
                pass

    app = FastAPI(
        title="Agent OS API",
        description="""
## Constitutional Operating System for Local AI

Agent OS provides a REST API for interacting with your personal AI operating system.
All AI behavior is governed by a transparent, natural language constitution.

### Key Features

- **Constitutional Governance**: AI behavior governed by readable, amendable rules
- **Consent-Based Memory**: Long-term memory requires explicit user approval
- **Multi-Agent Architecture**: Specialized agents (Whisper, Smith, Seshat, Sage, Quill, Muse)
- **Privacy First**: All data stays local on your machine

### Authentication

Most endpoints require authentication. Use one of these methods:

- **Session Cookie**: Set automatically after login via `/api/auth/login`
- **Bearer Token**: Pass `Authorization: Bearer <token>` header

### Rate Limits

Default rate limits (configurable via environment variables):
- **60 requests/minute** per client
- **1000 requests/hour** per client

Rate limit headers are included in responses:
- `X-RateLimit-Limit`: Maximum requests allowed
- `X-RateLimit-Remaining`: Requests remaining in window
- `X-RateLimit-Reset`: Unix timestamp when limit resets

When rate limited, responses return HTTP 429 with `Retry-After` header.

### WebSocket Endpoints

Real-time streaming is available via WebSocket:
- `/api/chat/ws` - Chat message streaming
- `/api/images/ws` - Image generation progress
- `/api/voice/ws` - Voice streaming
        """,
        version=API_VERSION,
        lifespan=lifespan,
        openapi_tags=OPENAPI_TAGS,
        license_info={
            "name": "MIT",
            "url": "https://opensource.org/licenses/MIT",
        },
        contact={
            "name": "Agent OS Community",
            "url": "https://github.com/kase1111-hash/Agent-OS",
        },
    )

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Static files
    if config.static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(config.static_dir)), name="static")

    # Templates
    templates = None
    if config.templates_dir.exists():
        templates = Jinja2Templates(directory=str(config.templates_dir))
        app.state.templates = templates

    # Store app state
    app.state.app_state = _app_state

    # Set up rate limiting
    if config.rate_limit_enabled:
        try:
            from .ratelimit import RateLimitMiddleware, create_limiter

            limiter = create_limiter(
                requests_per_minute=config.rate_limit_requests_per_minute,
                requests_per_hour=config.rate_limit_requests_per_hour,
                strategy=config.rate_limit_strategy,
                use_redis=config.rate_limit_use_redis,
                redis_url=config.rate_limit_redis_url,
            )
            app.add_middleware(
                RateLimitMiddleware,
                limiter=limiter,
                exclude_paths=["/health", "/metrics", "/docs", "/redoc", "/openapi.json"],
            )
            logger.info(
                f"Rate limiting enabled: {config.rate_limit_requests_per_minute}/min, "
                f"{config.rate_limit_requests_per_hour}/hour ({config.rate_limit_strategy})"
            )
        except Exception as e:
            logger.warning(f"Failed to enable rate limiting: {e}")

    # Set up observability (metrics and tracing middleware)
    try:
        from src.observability.middleware import setup_observability

        setup_observability(app, enable_metrics=True, enable_tracing=True)
        logger.info("Observability middleware enabled")
    except ImportError:
        logger.debug("Observability module not available, skipping middleware")

    # Include routers
    from .routes import (
        agents,
        auth,
        chat,
        constitution,
        contracts,
        images,
        intent_log,
        memory,
        security,
        system,
        voice,
    )

    app.include_router(auth.router, prefix="/api/auth", tags=["Authentication"])
    app.include_router(chat.router, prefix="/api/chat", tags=["Chat"])
    app.include_router(images.router, prefix="/api/images", tags=["Images"])
    app.include_router(agents.router, prefix="/api/agents", tags=["Agents"])
    app.include_router(constitution.router, prefix="/api/constitution", tags=["Constitution"])
    app.include_router(contracts.router, prefix="/api/contracts", tags=["Contracts"])
    app.include_router(intent_log.router, prefix="/api/intent-log", tags=["Intent Log"])
    app.include_router(memory.router, prefix="/api/memory", tags=["Memory"])
    app.include_router(security.router, prefix="/api/security", tags=["Security"])
    app.include_router(system.router, prefix="/api/system", tags=["System"])
    app.include_router(voice.router, prefix="/api/voice", tags=["Voice"])

    # Observability routes (metrics, health, tracing)
    try:
        from .routes import observability

        app.include_router(
            observability.router, prefix="/api/observability", tags=["Observability"]
        )
        logger.info("Observability routes enabled")
    except ImportError:
        logger.debug("Observability routes not available")

    # Root endpoint
    @app.get("/")
    async def root(request: Request):
        """Serve the main application page."""
        from fastapi.responses import HTMLResponse

        if templates:
            return templates.TemplateResponse(
                "index.html",
                {"request": request, "title": "Agent OS"},
            )
        else:
            # Fallback to static HTML
            index_path = config.static_dir / "index.html"
            if index_path.exists():
                return HTMLResponse(content=index_path.read_text())
            return {"message": "Agent OS Web Interface", "status": "running"}

    # Health check
    @app.get("/health")
    async def health_check():
        """Health check endpoint."""
        return {
            "status": "healthy",
            "version": "1.0.0",
            "components": {
                "api": "up",
                "websocket": "up",
                "voice": "up",
            },
        }

    _app = app
    return app


def get_app() -> Any:
    """Get the current FastAPI application instance."""
    global _app
    if _app is None:
        _app = create_app()
    return _app


def run_server(
    host: Optional[str] = None,
    port: Optional[int] = None,
    reload: bool = False,
) -> None:
    """
    Run the web server.

    Args:
        host: Host to bind to (default from config)
        port: Port to bind to (default from config)
        reload: Enable auto-reload for development
    """
    if not _check_dependencies():
        raise ImportError(
            "FastAPI and uvicorn are required. "
            "Install with: pip install fastapi uvicorn[standard]"
        )

    import uvicorn

    config = get_config()
    uvicorn.run(
        "src.web.app:get_app",
        factory=True,
        host=host or config.host,
        port=port or config.port,
        reload=reload,
    )


def main() -> None:
    """Main entry point for the Agent OS web interface."""
    run_server()


if __name__ == "__main__":
    main()
