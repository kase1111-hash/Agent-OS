"""
Shared API Models

Common response models used across multiple API endpoints.
Provides standardized error responses and operation results.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, Generic, List, Optional, TypeVar

from pydantic import BaseModel, Field


# =============================================================================
# Common Error Response Definitions for OpenAPI
# =============================================================================

# These can be used in route decorators like:
# @router.get("/resource", responses={**NOT_FOUND_RESPONSE, **UNAUTHORIZED_RESPONSE})

VALIDATION_ERROR_RESPONSE = {
    422: {
        "description": "Validation Error - Invalid request parameters",
        "content": {
            "application/json": {
                "example": {
                    "detail": [
                        {
                            "loc": ["body", "field_name"],
                            "msg": "field required",
                            "type": "value_error.missing",
                        }
                    ]
                }
            }
        },
    }
}

UNAUTHORIZED_RESPONSE = {
    401: {
        "description": "Unauthorized - Authentication required",
        "content": {
            "application/json": {
                "example": {"detail": "Not authenticated"}
            }
        },
    }
}

FORBIDDEN_RESPONSE = {
    403: {
        "description": "Forbidden - Insufficient permissions",
        "content": {
            "application/json": {
                "example": {"detail": "Not authorized to access this resource"}
            }
        },
    }
}

NOT_FOUND_RESPONSE = {
    404: {
        "description": "Not Found - Resource does not exist",
        "content": {
            "application/json": {
                "example": {"detail": "Resource not found"}
            }
        },
    }
}

CONFLICT_RESPONSE = {
    409: {
        "description": "Conflict - Resource already exists or state conflict",
        "content": {
            "application/json": {
                "example": {"detail": "Resource already exists"}
            }
        },
    }
}

INTERNAL_ERROR_RESPONSE = {
    500: {
        "description": "Internal Server Error",
        "content": {
            "application/json": {
                "example": {"detail": "An unexpected error occurred"}
            }
        },
    }
}


# =============================================================================
# Generic Types
# =============================================================================

T = TypeVar("T")


class PaginatedResponse(BaseModel, Generic[T]):
    """Paginated response wrapper."""

    items: List[T] = Field(description="List of items in this page")
    total: int = Field(description="Total number of items across all pages")
    page: int = Field(description="Current page number (1-indexed)")
    page_size: int = Field(description="Number of items per page")
    has_more: bool = Field(description="Whether there are more pages")


# =============================================================================
# Error Responses
# =============================================================================


class ErrorDetail(BaseModel):
    """Detailed error information."""

    code: str = Field(description="Machine-readable error code")
    message: str = Field(description="Human-readable error message")
    field: Optional[str] = Field(None, description="Field that caused the error, if applicable")


class ErrorResponse(BaseModel):
    """Standard error response format."""

    error: str = Field(description="Error type or category")
    detail: str = Field(description="Detailed error message")
    errors: Optional[List[ErrorDetail]] = Field(
        None, description="List of specific errors for validation failures"
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow, description="When the error occurred"
    )
    request_id: Optional[str] = Field(None, description="Request ID for debugging")

    class Config:
        json_schema_extra = {
            "example": {
                "error": "validation_error",
                "detail": "Invalid request parameters",
                "errors": [{"code": "required", "message": "Field is required", "field": "name"}],
                "timestamp": "2025-01-01T12:00:00Z",
                "request_id": "req_abc123",
            }
        }


# =============================================================================
# Operation Responses
# =============================================================================


class OperationStatus(str, Enum):
    """Status of an operation."""

    SUCCESS = "success"
    PENDING = "pending"
    FAILED = "failed"
    CANCELLED = "cancelled"


class OperationResponse(BaseModel):
    """Standard response for operations that modify state."""

    status: OperationStatus = Field(description="Operation result status")
    message: str = Field(description="Human-readable result message")
    resource_id: Optional[str] = Field(None, description="ID of the affected resource")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional operation metadata"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "status": "success",
                "message": "Resource created successfully",
                "resource_id": "res_abc123",
                "metadata": {"created_at": "2025-01-01T12:00:00Z"},
            }
        }


class DeleteResponse(BaseModel):
    """Response for delete operations."""

    deleted: bool = Field(description="Whether the resource was deleted")
    resource_id: str = Field(description="ID of the deleted resource")
    message: Optional[str] = Field(None, description="Additional information")

    class Config:
        json_schema_extra = {
            "example": {"deleted": True, "resource_id": "res_abc123", "message": "Resource deleted"}
        }


# =============================================================================
# Agent Control Responses
# =============================================================================


class AgentControlResponse(BaseModel):
    """Response for agent control operations (start, stop, restart)."""

    status: str = Field(description="Result status: started, stopped, restarted, already_active, already_stopped")
    agent: str = Field(description="Name of the agent")
    message: Optional[str] = Field(None, description="Additional information")

    class Config:
        json_schema_extra = {
            "example": {"status": "started", "agent": "whisper", "message": None}
        }


class AgentsOverviewResponse(BaseModel):
    """Overview statistics for all agents."""

    total_agents: int = Field(description="Total number of registered agents")
    status_distribution: Dict[str, int] = Field(
        description="Count of agents by status (active, idle, error, etc.)"
    )
    total_requests: int = Field(description="Total requests processed by all agents")
    total_success: int = Field(description="Total successful requests")
    total_failed: int = Field(description="Total failed requests")
    success_rate: float = Field(description="Overall success rate as percentage (0-100)")

    class Config:
        json_schema_extra = {
            "example": {
                "total_agents": 6,
                "status_distribution": {"active": 2, "idle": 4},
                "total_requests": 1500,
                "total_success": 1480,
                "total_failed": 20,
                "success_rate": 98.67,
            }
        }


# =============================================================================
# Chat/Model Responses
# =============================================================================


class ModelInfo(BaseModel):
    """Information about an available LLM model."""

    id: str = Field(description="Model identifier")
    name: str = Field(description="Display name")
    provider: str = Field(description="Model provider (ollama, llama_cpp, etc.)")
    size: Optional[str] = Field(None, description="Model size (e.g., 7B, 13B)")
    context_length: Optional[int] = Field(None, description="Maximum context length in tokens")
    is_current: bool = Field(False, description="Whether this is the currently active model")

    class Config:
        json_schema_extra = {
            "example": {
                "id": "llama3:8b",
                "name": "Llama 3 8B",
                "provider": "ollama",
                "size": "8B",
                "context_length": 8192,
                "is_current": True,
            }
        }


class ModelsListResponse(BaseModel):
    """List of available models."""

    models: List[ModelInfo] = Field(description="Available models")
    current: Optional[str] = Field(None, description="Currently active model ID")


class ModelSwitchResponse(BaseModel):
    """Response when switching models."""

    status: str = Field(description="Result: switched, already_active, failed")
    model: str = Field(description="Model ID")
    previous: Optional[str] = Field(None, description="Previously active model ID")
    message: Optional[str] = Field(None, description="Additional information")


# =============================================================================
# Export/Import Responses
# =============================================================================


class ExportResponse(BaseModel):
    """Response for data export operations."""

    format: str = Field(description="Export format (json, csv, markdown)")
    filename: str = Field(description="Suggested filename")
    data: Any = Field(description="Exported data")
    item_count: int = Field(description="Number of items exported")
    exported_at: datetime = Field(
        default_factory=datetime.utcnow, description="When the export was created"
    )


class ImportResponse(BaseModel):
    """Response for data import operations."""

    status: str = Field(description="Result: success, partial, failed")
    imported_count: int = Field(description="Number of items successfully imported")
    skipped_count: int = Field(0, description="Number of items skipped")
    error_count: int = Field(0, description="Number of items that failed to import")
    errors: List[str] = Field(default_factory=list, description="Error messages for failed items")


# =============================================================================
# Storage/Stats Responses
# =============================================================================


class StorageStatsResponse(BaseModel):
    """Storage statistics response."""

    total_items: int = Field(description="Total number of stored items")
    total_size_bytes: int = Field(description="Total storage size in bytes")
    total_size_human: str = Field(description="Human-readable storage size")
    by_type: Dict[str, int] = Field(
        default_factory=dict, description="Item count by type/category"
    )
    oldest_item: Optional[datetime] = Field(None, description="Timestamp of oldest item")
    newest_item: Optional[datetime] = Field(None, description="Timestamp of newest item")


# =============================================================================
# Image Gallery Responses
# =============================================================================


class ImageDeleteResponse(BaseModel):
    """Response for image deletion."""

    deleted: bool = Field(description="Whether the image was deleted")
    image_id: str = Field(description="ID of the deleted image")
    freed_bytes: Optional[int] = Field(None, description="Storage space freed in bytes")


class ImageStatsResponse(BaseModel):
    """Image generation and gallery statistics."""

    total_images: int = Field(description="Total images in gallery")
    total_generations: int = Field(description="Total generation jobs run")
    successful_generations: int = Field(description="Successful generation count")
    failed_generations: int = Field(description="Failed generation count")
    total_storage_bytes: int = Field(description="Total storage used in bytes")
    average_generation_time_ms: float = Field(description="Average generation time in milliseconds")
    models_used: Dict[str, int] = Field(
        default_factory=dict, description="Generation count by model"
    )


# =============================================================================
# Memory Export Response
# =============================================================================


class MemoryExportResponse(BaseModel):
    """Response for memory export operations."""

    format: str = Field(description="Export format (json, csv)")
    entries_count: int = Field(description="Number of memory entries exported")
    total_size_bytes: int = Field(description="Total size of exported data")
    exported_at: datetime = Field(
        default_factory=datetime.utcnow, description="When the export was created"
    )
    data: Any = Field(description="Exported memory data")


# =============================================================================
# Health Check Response
# =============================================================================


class ComponentHealth(BaseModel):
    """Health status of a system component."""

    status: str = Field(description="Component status: up, down, degraded")
    latency_ms: Optional[float] = Field(None, description="Response latency in milliseconds")
    message: Optional[str] = Field(None, description="Additional status information")


class HealthResponse(BaseModel):
    """System health check response."""

    status: str = Field(description="Overall status: healthy, degraded, unhealthy")
    version: str = Field(description="API version")
    components: Dict[str, ComponentHealth] = Field(
        description="Health status of individual components"
    )
    uptime_seconds: Optional[float] = Field(None, description="System uptime in seconds")

    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "version": "1.0.0",
                "components": {
                    "api": {"status": "up", "latency_ms": 1.2},
                    "database": {"status": "up", "latency_ms": 5.3},
                    "llm": {"status": "up", "latency_ms": 45.0},
                },
                "uptime_seconds": 86400.0,
            }
        }
