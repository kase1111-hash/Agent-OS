"""
System API Routes

Provides endpoints for system status and administration.
"""

import logging
import os
import platform
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Cookie, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from src import __version__ as PKG_VERSION

from ..app import _app_state
from ..auth_helpers import require_admin_user, require_authenticated_user

# Backward compatibility alias
require_admin_auth = require_admin_user

logger = logging.getLogger(__name__)
router = APIRouter()

# Try to import real components for health checks
try:
    from src.agents.loader import create_loader

    AGENTS_AVAILABLE = True
except ImportError:
    AGENTS_AVAILABLE = False

try:
    from src.core.constitution import create_kernel

    CONSTITUTION_AVAILABLE = True
except ImportError:
    CONSTITUTION_AVAILABLE = False

try:
    from src.agents.seshat import create_seshat_agent

    MEMORY_AVAILABLE = True
except ImportError:
    MEMORY_AVAILABLE = False


# =============================================================================
# Models
# =============================================================================


class SystemInfo(BaseModel):
    """System information."""

    version: str = PKG_VERSION
    python_version: str
    platform: str
    hostname: str
    started_at: datetime


class SystemStatus(BaseModel):
    """System status."""

    status: str  # healthy, degraded, unhealthy
    uptime_seconds: float
    components: Dict[str, str]  # component name -> status
    warnings: List[str] = Field(default_factory=list)


class ComponentHealth(BaseModel):
    """Health status of a component."""

    name: str
    status: str  # up, down, degraded
    latency_ms: Optional[float] = None
    last_check: datetime
    details: Dict[str, Any] = Field(default_factory=dict)


class LogEntry(BaseModel):
    """A system log entry."""

    timestamp: datetime
    level: str
    logger: str
    message: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SettingValue(BaseModel):
    """A configuration setting."""

    key: str
    value: Any
    description: str = ""
    editable: bool = True
    data_type: str = "string"  # string, number, boolean, array, object
    wired: bool = False  # True if changing this value has a runtime effect


class UpdateSettingRequest(BaseModel):
    """Request to update a setting."""

    value: Any


# =============================================================================
# System State
# =============================================================================


_start_time = datetime.utcnow()


def _get_uptime_seconds() -> float:
    """Get system uptime in seconds."""
    return (datetime.utcnow() - _start_time).total_seconds()


# Settings store. Settings with wired=True have runtime effect when changed.
# Settings with wired=False are stored but informational only.
_settings: Dict[str, SettingValue] = {
    "chat.max_history": SettingValue(
        key="chat.max_history",
        value=100,
        description="Maximum messages to keep in conversation history",
        data_type="number",
    ),
    "agents.default_timeout": SettingValue(
        key="agents.default_timeout",
        value=30,
        description="Default timeout for agent requests (seconds)",
        data_type="number",
    ),
    "memory.auto_cleanup": SettingValue(
        key="memory.auto_cleanup",
        value=True,
        description="Automatically clean up expired memories",
        data_type="boolean",
    ),
    "constitution.strict_mode": SettingValue(
        key="constitution.strict_mode",
        value=True,
        description="Enforce strict constitutional checking",
        data_type="boolean",
    ),
    "logging.level": SettingValue(
        key="logging.level",
        value="INFO",
        description="Logging level (DEBUG, INFO, WARNING, ERROR)",
        data_type="string",
        wired=True,
    ),
    "api.rate_limit": SettingValue(
        key="api.rate_limit",
        value=100,
        description="Maximum API requests per minute",
        data_type="number",
    ),
    "websocket.heartbeat": SettingValue(
        key="websocket.heartbeat",
        value=30,
        description="WebSocket heartbeat interval (seconds)",
        data_type="number",
    ),
}

# In-memory log capture — wired to real Python logging
_logs: List[LogEntry] = []
_MAX_LOG_ENTRIES = 1000


class _InMemoryLogHandler(logging.Handler):
    """Captures log records into _logs for the /logs endpoint."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            entry = LogEntry(
                timestamp=datetime.utcfromtimestamp(record.created),
                level=record.levelname,
                logger=record.name,
                message=self.format(record),
            )
            _logs.append(entry)
            if len(_logs) > _MAX_LOG_ENTRIES:
                del _logs[: len(_logs) - _MAX_LOG_ENTRIES]
        except Exception:
            pass  # Never let logging crash the app


# Install the handler on the "src" logger so all application logs are captured.
_mem_handler = _InMemoryLogHandler()
_mem_handler.setLevel(logging.DEBUG)
_mem_handler.setFormatter(logging.Formatter("%(message)s"))
logging.getLogger("src").addHandler(_mem_handler)


# =============================================================================
# Endpoints
# =============================================================================


@router.get("/info", response_model=SystemInfo)
async def get_system_info(
    user_id: str = Depends(require_authenticated_user),
) -> SystemInfo:
    """Get system information."""
    return SystemInfo(
        version=PKG_VERSION,
        python_version=sys.version,
        platform=f"{platform.system()} {platform.release()}",
        hostname=platform.node(),
        started_at=_start_time,
    )


@router.get("/status", response_model=SystemStatus)
async def get_system_status(
    user_id: str = Depends(require_authenticated_user),
) -> SystemStatus:
    """Get system status."""
    # Check component health with real availability probes
    components = {"api": "up", "websocket": "up"}

    if AGENTS_AVAILABLE:
        try:
            create_loader(Path.cwd())
            components["agents"] = "up"
        except Exception:
            components["agents"] = "degraded"
    else:
        components["agents"] = "degraded"

    if MEMORY_AVAILABLE:
        try:
            create_seshat_agent(use_mock=True)
            components["memory"] = "up"
        except Exception:
            components["memory"] = "degraded"
    else:
        components["memory"] = "degraded"

    if CONSTITUTION_AVAILABLE:
        try:
            kernel = create_kernel(project_root=Path.cwd())
            result = kernel.initialize()
            components["constitution"] = "up" if result.is_valid else "degraded"
        except Exception:
            components["constitution"] = "degraded"
    else:
        components["constitution"] = "degraded"

    # Determine overall status
    down_count = sum(1 for s in components.values() if s == "down")
    degraded_count = sum(1 for s in components.values() if s == "degraded")

    if down_count > 0:
        status = "unhealthy"
    elif degraded_count > 0:
        status = "degraded"
    else:
        status = "healthy"

    warnings = []
    if degraded_count > 0:
        warnings.append(f"{degraded_count} component(s) degraded")

    return SystemStatus(
        status=status,
        uptime_seconds=_get_uptime_seconds(),
        components=components,
        warnings=warnings,
    )


@router.get("/health", response_model=List[ComponentHealth])
async def get_component_health() -> List[ComponentHealth]:
    """Get detailed health status of all components."""
    now = datetime.utcnow()
    components = []

    # API health (always up if we're responding)
    components.append(
        ComponentHealth(
            name="api",
            status="up",
            latency_ms=1.0,
            last_check=now,
            details={"version": PKG_VERSION, "framework": "FastAPI"},
        )
    )

    # WebSocket health
    components.append(
        ComponentHealth(
            name="websocket",
            status="up",
            latency_ms=1.0,
            last_check=now,
            details={"protocol": "WebSocket", "available": True},
        )
    )

    # Agents health - try real check
    if AGENTS_AVAILABLE:
        try:
            start = time.time()
            loader = create_loader(Path.cwd())
            agents = loader.registry.get_all()
            latency = (time.time() - start) * 1000
            active_count = sum(1 for a in agents if a.is_active)
            components.append(
                ComponentHealth(
                    name="agents",
                    status="up",
                    latency_ms=round(latency, 2),
                    last_check=now,
                    details={
                        "registered_agents": len(agents),
                        "active_agents": active_count,
                        "using_real_registry": True,
                    },
                )
            )
        except Exception as e:
            components.append(
                ComponentHealth(
                    name="agents",
                    status="degraded",
                    latency_ms=0,
                    last_check=now,
                    details={"error": str(e), "using_real_registry": False},
                )
            )
    else:
        components.append(
            ComponentHealth(
                name="agents",
                status="up",
                latency_ms=1.0,
                last_check=now,
                details={"using_real_registry": False, "mode": "mock"},
            )
        )

    # Memory health - try real check
    if MEMORY_AVAILABLE:
        try:
            start = time.time()
            seshat = create_seshat_agent(use_mock=True)
            latency = (time.time() - start) * 1000
            components.append(
                ComponentHealth(
                    name="memory",
                    status="up",
                    latency_ms=round(latency, 2),
                    last_check=now,
                    details={"using_seshat": True, "backend": "in-memory"},
                )
            )
        except Exception as e:
            components.append(
                ComponentHealth(
                    name="memory",
                    status="degraded",
                    latency_ms=0,
                    last_check=now,
                    details={"error": str(e), "using_seshat": False},
                )
            )
    else:
        components.append(
            ComponentHealth(
                name="memory",
                status="up",
                latency_ms=1.0,
                last_check=now,
                details={"using_seshat": False, "mode": "mock"},
            )
        )

    # Constitution health - try real check
    if CONSTITUTION_AVAILABLE:
        try:
            start = time.time()
            kernel = create_kernel(project_root=Path.cwd())
            result = kernel.initialize()
            latency = (time.time() - start) * 1000
            rules_count = len(kernel._registry.get_all_rules()) if result.is_valid else 0
            components.append(
                ComponentHealth(
                    name="constitution",
                    status="up" if result.is_valid else "degraded",
                    latency_ms=round(latency, 2),
                    last_check=now,
                    details={
                        "using_real_kernel": True,
                        "rules_loaded": rules_count,
                        "is_valid": result.is_valid,
                    },
                )
            )
        except Exception as e:
            components.append(
                ComponentHealth(
                    name="constitution",
                    status="degraded",
                    latency_ms=0,
                    last_check=now,
                    details={"error": str(e), "using_real_kernel": False},
                )
            )
    else:
        components.append(
            ComponentHealth(
                name="constitution",
                status="up",
                latency_ms=1.0,
                last_check=now,
                details={"using_real_kernel": False, "mode": "mock"},
            )
        )

    return components


@router.get("/settings", response_model=List[SettingValue])
async def list_settings(
    admin_id: str = Depends(require_admin_user),
) -> List[SettingValue]:
    """List all configuration settings (admin only)."""
    return list(_settings.values())


@router.get("/settings/{key}", response_model=SettingValue)
async def get_setting(
    key: str,
    admin_id: str = Depends(require_admin_user),
) -> SettingValue:
    """Get a specific setting (admin only)."""
    if key not in _settings:
        raise HTTPException(status_code=404, detail=f"Setting not found: {key}")
    return _settings[key]


@router.put("/settings/{key}", response_model=SettingValue)
async def update_setting(
    key: str,
    request: UpdateSettingRequest,
    admin_id: str = Depends(require_admin_user),
) -> SettingValue:
    """Update a setting (admin only)."""
    if key not in _settings:
        raise HTTPException(status_code=404, detail=f"Setting not found: {key}")

    setting = _settings[key]

    if not setting.editable:
        raise HTTPException(status_code=403, detail=f"Setting is not editable: {key}")

    # Validate type
    expected_type = setting.data_type
    value = request.value

    if expected_type == "number" and not isinstance(value, (int, float)):
        raise HTTPException(status_code=400, detail="Value must be a number")
    elif expected_type == "boolean" and not isinstance(value, bool):
        raise HTTPException(status_code=400, detail="Value must be a boolean")
    elif expected_type == "string" and not isinstance(value, str):
        raise HTTPException(status_code=400, detail="Value must be a string")

    setting.value = value

    # Apply wired settings to their runtime consumers
    if key == "logging.level":
        level = getattr(logging, str(value).upper(), None)
        if level is not None:
            logging.getLogger("src").setLevel(level)
            logger.info(f"Logging level changed to {value}")

    return setting


@router.get("/logs", response_model=List[LogEntry])
async def get_logs(
    level: Optional[str] = None,
    logger_name: Optional[str] = None,
    limit: int = 100,
    admin_id: str = Depends(require_admin_user),
) -> List[LogEntry]:
    """Get system logs (admin only)."""
    logs = _logs

    if level:
        logs = [l for l in logs if l.level == level.upper()]

    if logger_name:
        logs = [l for l in logs if l.logger == logger_name]

    return logs[-limit:]


@router.post("/shutdown")
async def shutdown_system(
    confirm: bool = False,
    admin_user: str = Depends(require_admin_auth),
) -> Dict[str, str]:
    """
    Request system shutdown (requires admin authentication).

    Requires confirmation.
    """
    if not confirm:
        raise HTTPException(
            status_code=400,
            detail="Confirmation required. Set confirm=true to proceed.",
        )

    # In production, this would initiate a graceful shutdown
    # For the mock API, we just return a message
    return {
        "status": "shutdown_requested",
        "message": "System shutdown has been requested. This is a mock response.",
    }


@router.post("/restart")
async def restart_system(
    confirm: bool = False,
    admin_user: str = Depends(require_admin_auth),
) -> Dict[str, str]:
    """
    Request system restart (requires admin authentication).

    Requires confirmation.
    """
    if not confirm:
        raise HTTPException(
            status_code=400,
            detail="Confirmation required. Set confirm=true to proceed.",
        )

    return {
        "status": "restart_requested",
        "message": "System restart has been requested. This is a mock response.",
    }


@router.get("/version")
async def get_version() -> Dict[str, str]:
    """Get version information."""
    return {
        "version": PKG_VERSION,
        "api_version": "v1",
        "build": "2026.01.01",
        "python": platform.python_version(),
    }


@router.get("/dreaming")
async def get_dreaming_status(
    user_id: str = Depends(require_authenticated_user),
) -> Dict[str, Any]:
    """
    Get the current "dreaming" status.

    Shows what the system is currently working on internally.
    Updates are throttled to every 5 seconds for performance.

    Returns:
        Current dreaming status with message, phase, and timing info
    """
    try:
        from src.web.dreaming import get_dreaming

        return get_dreaming().get_status()
    except ImportError:
        return {
            "message": "Idle",
            "phase": "idle",
            "operation": None,
            "updated_at": datetime.utcnow().isoformat(),
            "operations_count": 0,
            "throttle_interval": 5.0,
        }


@router.get("/metrics")
async def get_metrics(
    user_id: str = Depends(require_authenticated_user),
) -> Dict[str, Any]:
    """Get system metrics."""
    uptime = _get_uptime_seconds()
    total_requests = _app_state.request_count
    failed_requests = _app_state.request_errors
    rate_per_minute = (total_requests / (uptime / 60)) if uptime > 0 else 0.0

    return {
        "uptime_seconds": uptime,
        "memory": {
            "total_mb": 1024,  # Requires psutil — kept as placeholder
            "used_mb": 256,
            "available_mb": 768,
        },
        "cpu": {
            "usage_percent": 15.5,  # Requires psutil — kept as placeholder
            "cores": os.cpu_count() or 1,
        },
        "requests": {
            "total": total_requests,
            "success": total_requests - failed_requests,
            "failed": failed_requests,
            "rate_per_minute": round(rate_per_minute, 2),
        },
        "websocket": {
            "active_connections": len(_app_state.active_connections),
        },
    }
