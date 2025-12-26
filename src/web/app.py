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

    from fastapi import FastAPI
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
        title="Agent OS",
        description="Web interface for Agent OS - Your Personal AI Operating System",
        version="1.0.0",
        lifespan=lifespan,
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

    # Include routers
    from .routes import agents, auth, chat, constitution, contracts, intent_log, memory, system, voice

    app.include_router(auth.router, prefix="/api/auth", tags=["Authentication"])
    app.include_router(chat.router, prefix="/api/chat", tags=["Chat"])
    app.include_router(agents.router, prefix="/api/agents", tags=["Agents"])
    app.include_router(constitution.router, prefix="/api/constitution", tags=["Constitution"])
    app.include_router(contracts.router, prefix="/api/contracts", tags=["Contracts"])
    app.include_router(intent_log.router, prefix="/api/intent-log", tags=["Intent Log"])
    app.include_router(memory.router, prefix="/api/memory", tags=["Memory"])
    app.include_router(system.router, prefix="/api/system", tags=["System"])
    app.include_router(voice.router, prefix="/api/voice", tags=["Voice"])

    # Root endpoint
    @app.get("/")
    async def root():
        """Serve the main application page."""
        from fastapi import Request
        from fastapi.responses import HTMLResponse

        if templates:
            return templates.TemplateResponse(
                "index.html",
                {"request": Request, "title": "Agent OS"},
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
