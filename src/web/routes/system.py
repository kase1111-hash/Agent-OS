"""
System API Routes

Provides endpoints for system status and administration.
"""

import logging
import os
import platform
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)
router = APIRouter()


# =============================================================================
# Models
# =============================================================================


class SystemInfo(BaseModel):
    """System information."""

    version: str = "1.0.0"
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


# Mock settings store
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

# Mock logs
_logs: List[LogEntry] = []


# =============================================================================
# Endpoints
# =============================================================================


@router.get("/info", response_model=SystemInfo)
async def get_system_info() -> SystemInfo:
    """Get system information."""
    return SystemInfo(
        version="1.0.0",
        python_version=sys.version,
        platform=f"{platform.system()} {platform.release()}",
        hostname=platform.node(),
        started_at=_start_time,
    )


@router.get("/status", response_model=SystemStatus)
async def get_system_status() -> SystemStatus:
    """Get system status."""
    # Check component health
    components = {
        "api": "up",
        "websocket": "up",
        "agents": "up",
        "memory": "up",
        "constitution": "up",
    }

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

    components = [
        ComponentHealth(
            name="api",
            status="up",
            latency_ms=2.5,
            last_check=now,
            details={"version": "1.0.0", "requests_served": 1542},
        ),
        ComponentHealth(
            name="websocket",
            status="up",
            latency_ms=1.2,
            last_check=now,
            details={"active_connections": 3, "messages_processed": 856},
        ),
        ComponentHealth(
            name="agents",
            status="up",
            latency_ms=45.3,
            last_check=now,
            details={"active_agents": 4, "total_requests": 2500},
        ),
        ComponentHealth(
            name="memory",
            status="up",
            latency_ms=5.8,
            last_check=now,
            details={"entries_stored": 5, "size_bytes": 2048},
        ),
        ComponentHealth(
            name="constitution",
            status="up",
            latency_ms=3.1,
            last_check=now,
            details={"rules_loaded": 7, "validations_performed": 1542},
        ),
    ]

    return components


@router.get("/settings", response_model=List[SettingValue])
async def list_settings() -> List[SettingValue]:
    """List all configuration settings."""
    return list(_settings.values())


@router.get("/settings/{key}", response_model=SettingValue)
async def get_setting(key: str) -> SettingValue:
    """Get a specific setting."""
    if key not in _settings:
        raise HTTPException(status_code=404, detail=f"Setting not found: {key}")
    return _settings[key]


@router.put("/settings/{key}", response_model=SettingValue)
async def update_setting(key: str, request: UpdateSettingRequest) -> SettingValue:
    """Update a setting."""
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
    return setting


@router.get("/logs", response_model=List[LogEntry])
async def get_logs(
    level: Optional[str] = None,
    logger_name: Optional[str] = None,
    limit: int = 100,
) -> List[LogEntry]:
    """Get system logs."""
    global _logs

    # Generate some mock logs if empty
    if not _logs:
        _logs = [
            LogEntry(
                timestamp=datetime.utcnow(),
                level="INFO",
                logger="system",
                message="Agent OS Web Interface started",
            ),
            LogEntry(
                timestamp=datetime.utcnow(),
                level="INFO",
                logger="agents",
                message="All agents initialized successfully",
            ),
            LogEntry(
                timestamp=datetime.utcnow(),
                level="DEBUG",
                logger="websocket",
                message="WebSocket server listening on port 8080",
            ),
        ]

    logs = _logs

    if level:
        logs = [l for l in logs if l.level == level.upper()]

    if logger_name:
        logs = [l for l in logs if l.logger == logger_name]

    return logs[-limit:]


@router.post("/shutdown")
async def shutdown_system(confirm: bool = False) -> Dict[str, str]:
    """
    Request system shutdown.

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
async def restart_system(confirm: bool = False) -> Dict[str, str]:
    """
    Request system restart.

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
        "version": "1.0.0",
        "api_version": "v1",
        "build": "2024.12.21",
        "python": platform.python_version(),
    }


@router.get("/metrics")
async def get_metrics() -> Dict[str, Any]:
    """Get system metrics."""
    return {
        "uptime_seconds": _get_uptime_seconds(),
        "memory": {
            "total_mb": 1024,  # Mock value
            "used_mb": 256,
            "available_mb": 768,
        },
        "cpu": {
            "usage_percent": 15.5,  # Mock value
            "cores": os.cpu_count() or 1,
        },
        "requests": {
            "total": 1542,
            "success": 1520,
            "failed": 22,
            "rate_per_minute": 25.7,
        },
        "websocket": {
            "active_connections": 3,
            "total_messages": 856,
        },
    }
