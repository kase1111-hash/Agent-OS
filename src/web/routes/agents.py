"""
Agent Monitoring API Routes

Provides endpoints for monitoring and managing agents.
"""

import logging
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Cookie, Depends, HTTPException, Query, Request
from pydantic import BaseModel, Field

from ..models import NOT_FOUND_RESPONSE, AgentControlResponse, AgentsOverviewResponse


def require_admin_auth(
    request: Request,
    session_token: Optional[str] = Cookie(None),
) -> str:
    """
    Dependency to require admin authentication for protected endpoints.

    Returns the user_id if authenticated and authorized.
    Raises HTTPException if not authenticated or not admin.
    """
    from ..auth import UserRole, get_user_store

    # Get token from cookie or Authorization header
    token = session_token
    if not token:
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            token = auth_header[7:]

    if not token:
        raise HTTPException(
            status_code=401,
            detail="Authentication required for admin operations"
        )

    store = get_user_store()
    user = store.validate_session(token)

    if not user:
        raise HTTPException(
            status_code=401,
            detail="Session expired or invalid"
        )

    # Check for admin role
    if user.role != UserRole.ADMIN:
        raise HTTPException(
            status_code=403,
            detail="Admin privileges required for this operation"
        )

    return user.user_id

logger = logging.getLogger(__name__)
router = APIRouter()

# Try to import real agent registry
try:
    from src.agents.interface import AgentState
    from src.agents.loader import AgentLoader, AgentRegistry, create_loader

    REAL_AGENTS_AVAILABLE = True
except ImportError:
    REAL_AGENTS_AVAILABLE = False
    logger.warning("Real agent registry not available, using mock data")


# =============================================================================
# Models
# =============================================================================


class AgentStatus(str, Enum):
    """Agent status values."""

    ACTIVE = "active"
    IDLE = "idle"
    PROCESSING = "processing"
    ERROR = "error"
    DISABLED = "disabled"
    UNINITIALIZED = "uninitialized"


class AgentMetrics(BaseModel):
    """Agent performance metrics."""

    requests_total: int = 0
    requests_success: int = 0
    requests_failed: int = 0
    average_response_time_ms: float = 0.0
    uptime_seconds: float = 0.0
    last_request_at: Optional[datetime] = None


class AgentCapability(BaseModel):
    """Agent capability information."""

    name: str
    description: str = ""


class AgentInfo(BaseModel):
    """Detailed agent information."""

    name: str
    description: str = ""
    status: AgentStatus = AgentStatus.UNINITIALIZED
    capabilities: List[AgentCapability] = Field(default_factory=list)
    supported_intents: List[str] = Field(default_factory=list)
    metrics: AgentMetrics = Field(default_factory=AgentMetrics)
    created_at: Optional[datetime] = None
    config: Dict[str, Any] = Field(default_factory=dict)


class AgentSummary(BaseModel):
    """Summary of an agent for listing."""

    name: str
    status: AgentStatus
    description: str = ""
    requests_total: int = 0


class AgentLogEntry(BaseModel):
    """A log entry from an agent."""

    timestamp: datetime
    level: str
    message: str
    agent_name: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class StartAgentRequest(BaseModel):
    """Request to start an agent."""

    config: Optional[Dict[str, Any]] = None


class UpdateAgentRequest(BaseModel):
    """Request to update agent configuration."""

    config: Dict[str, Any]


# =============================================================================
# Mock Data Store
# =============================================================================


class AgentStore:
    """
    Agent store that integrates with the real AgentRegistry.

    Falls back to mock data if real agents aren't available.
    """

    def __init__(self):
        self._registry: Optional[Any] = None
        self._loader: Optional[Any] = None
        self._mock_agents: Dict[str, AgentInfo] = {}
        self._logs: List[AgentLogEntry] = []
        self._use_real_agents = False

        # Try to initialize real agent registry
        if REAL_AGENTS_AVAILABLE:
            try:
                self._loader = create_loader(Path.cwd())
                self._registry = self._loader.registry
                # Try to discover agents from common locations
                agents_dir = Path.cwd() / "agents"
                if agents_dir.exists():
                    self._loader.discover_agents(agents_dir)
                # Only use real agents if any were actually discovered
                if self._registry.get_all():
                    self._use_real_agents = True
                    logger.info("Connected to real agent registry")
                else:
                    logger.info("No agents discovered, falling back to mock agents")
            except Exception as e:
                logger.warning(f"Failed to initialize real agent registry: {e}")
                self._use_real_agents = False

        # Initialize mock agents as fallback
        if not self._use_real_agents:
            self._init_mock_agents()

    def _init_mock_agents(self):
        """Initialize mock agents for demonstration."""
        self._mock_agents = {
            "whisper": AgentInfo(
                name="whisper",
                description="Intent classification and request routing",
                status=AgentStatus.ACTIVE,
                capabilities=[
                    AgentCapability(
                        name="intent_classification", description="Classify user intents"
                    ),
                    AgentCapability(name="routing", description="Route requests to agents"),
                ],
                supported_intents=["*"],
                metrics=AgentMetrics(
                    requests_total=1542,
                    requests_success=1520,
                    requests_failed=22,
                    average_response_time_ms=45.2,
                    uptime_seconds=86400,
                    last_request_at=datetime.utcnow(),
                ),
                created_at=datetime.utcnow(),
            ),
            "smith": AgentInfo(
                name="smith",
                description="Constitutional validation and safety guardian",
                status=AgentStatus.ACTIVE,
                capabilities=[
                    AgentCapability(
                        name="constitutional_check", description="Validate against constitution"
                    ),
                    AgentCapability(name="safety_filter", description="Filter unsafe content"),
                ],
                supported_intents=["*"],
                metrics=AgentMetrics(
                    requests_total=1542,
                    requests_success=1541,
                    requests_failed=1,
                    average_response_time_ms=12.8,
                    uptime_seconds=86400,
                    last_request_at=datetime.utcnow(),
                ),
                created_at=datetime.utcnow(),
            ),
            "seshat": AgentInfo(
                name="seshat",
                description="Memory management and semantic retrieval",
                status=AgentStatus.IDLE,
                capabilities=[
                    AgentCapability(
                        name="memory_storage", description="Store and retrieve memories"
                    ),
                    AgentCapability(
                        name="semantic_search", description="Search memories by meaning"
                    ),
                ],
                supported_intents=["memory.*", "recall.*"],
                metrics=AgentMetrics(
                    requests_total=423,
                    requests_success=420,
                    requests_failed=3,
                    average_response_time_ms=65.2,
                    uptime_seconds=86400,
                ),
                created_at=datetime.utcnow(),
            ),
            "muse": AgentInfo(
                name="muse",
                description="Creative content generation",
                status=AgentStatus.IDLE,
                capabilities=[
                    AgentCapability(
                        name="creative_writing", description="Generate creative content"
                    ),
                    AgentCapability(name="brainstorming", description="Help with ideas"),
                ],
                supported_intents=["creative.*", "content.*"],
                metrics=AgentMetrics(
                    requests_total=234,
                    requests_success=230,
                    requests_failed=4,
                    average_response_time_ms=1250.5,
                    uptime_seconds=86400,
                ),
                created_at=datetime.utcnow(),
            ),
            "sage": AgentInfo(
                name="sage",
                description="Deep reasoning and analysis",
                status=AgentStatus.IDLE,
                capabilities=[
                    AgentCapability(name="reasoning", description="Complex logical reasoning"),
                    AgentCapability(name="analysis", description="In-depth analysis"),
                ],
                supported_intents=["reason.*", "analyze.*"],
                metrics=AgentMetrics(
                    requests_total=156,
                    requests_success=152,
                    requests_failed=4,
                    average_response_time_ms=2100.8,
                    uptime_seconds=86400,
                ),
                created_at=datetime.utcnow(),
            ),
            "quill": AgentInfo(
                name="quill",
                description="Document formatting and output generation",
                status=AgentStatus.IDLE,
                capabilities=[
                    AgentCapability(name="formatting", description="Format documents"),
                    AgentCapability(name="export", description="Export to various formats"),
                ],
                supported_intents=["format.*", "export.*"],
                metrics=AgentMetrics(
                    requests_total=89,
                    requests_success=88,
                    requests_failed=1,
                    average_response_time_ms=320.4,
                    uptime_seconds=86400,
                ),
                created_at=datetime.utcnow(),
            ),
        }

    def _convert_real_agent(self, registered) -> AgentInfo:
        """Convert a RegisteredAgent to AgentInfo."""
        caps = registered.capabilities
        metrics = registered.instance.metrics

        # Map AgentState to AgentStatus
        state_map = {
            "UNINITIALIZED": AgentStatus.UNINITIALIZED,
            "READY": AgentStatus.IDLE,
            "PROCESSING": AgentStatus.PROCESSING,
            "ERROR": AgentStatus.ERROR,
            "STOPPED": AgentStatus.DISABLED,
        }
        status = state_map.get(registered.state.name, AgentStatus.IDLE)
        if registered.is_active and status == AgentStatus.IDLE:
            status = AgentStatus.ACTIVE

        return AgentInfo(
            name=registered.name,
            description=caps.description if caps else "",
            status=status,
            capabilities=[
                AgentCapability(name=c.value, description=c.value)
                for c in (caps.capabilities if caps else [])
            ],
            supported_intents=list(caps.supported_intents) if caps else [],
            metrics=AgentMetrics(
                requests_total=metrics.requests_processed,
                requests_success=metrics.requests_succeeded,
                requests_failed=metrics.requests_failed,
                average_response_time_ms=metrics.average_response_time_ms,
                uptime_seconds=metrics.uptime_seconds,
                last_request_at=metrics.last_request_time,
            ),
            created_at=registered.registered_at,
            config=(
                registered.config.model_dump() if hasattr(registered.config, "model_dump") else {}
            ),
        )

    def get_all(self) -> List[AgentInfo]:
        """Get all agents."""
        if self._use_real_agents and self._registry:
            try:
                registered_agents = self._registry.get_all()
                return [self._convert_real_agent(a) for a in registered_agents]
            except Exception as e:
                logger.error(f"Error getting real agents: {e}")
        return list(self._mock_agents.values())

    def get(self, name: str) -> Optional[AgentInfo]:
        """Get an agent by name."""
        if self._use_real_agents and self._registry:
            try:
                registered = self._registry.get(name)
                if registered:
                    return self._convert_real_agent(registered)
            except Exception as e:
                logger.error(f"Error getting agent {name}: {e}")
        return self._mock_agents.get(name)

    def update_status(self, name: str, status: AgentStatus) -> bool:
        """Update an agent's status."""
        if self._use_real_agents and self._registry:
            try:
                if status == AgentStatus.ACTIVE:
                    return self._registry.start_agent(name)
                elif status == AgentStatus.DISABLED:
                    return self._registry.stop_agent(name)
            except Exception as e:
                logger.error(f"Error updating agent status: {e}")

        # Fallback to mock
        if name in self._mock_agents:
            self._mock_agents[name].status = status
            return True
        return False

    def add_log(self, entry: AgentLogEntry) -> None:
        """Add a log entry."""
        self._logs.append(entry)
        # Keep only last 1000 entries
        if len(self._logs) > 1000:
            self._logs = self._logs[-1000:]

    def get_logs(
        self,
        agent_name: Optional[str] = None,
        level: Optional[str] = None,
        limit: int = 100,
    ) -> List[AgentLogEntry]:
        """Get log entries."""
        logs = self._logs

        if agent_name:
            logs = [l for l in logs if l.agent_name == agent_name]

        if level:
            logs = [l for l in logs if l.level == level]

        return logs[-limit:]

    @property
    def is_using_real_agents(self) -> bool:
        """Check if using real agent registry."""
        return self._use_real_agents


# Global store instance
_store: Optional[AgentStore] = None


def get_store() -> AgentStore:
    """Get the agent store."""
    global _store
    if _store is None:
        _store = AgentStore()
    return _store


# =============================================================================
# Endpoints
# =============================================================================


@router.get("/", response_model=List[AgentSummary])
async def list_agents(
    status: Optional[AgentStatus] = None,
) -> List[AgentSummary]:
    """
    List all registered agents.

    Optionally filter by status.
    """
    store = get_store()
    agents = store.get_all()
    logger.info(f"Listing agents: found {len(agents)} agents, using_real={store._use_real_agents}")

    if status:
        agents = [a for a in agents if a.status == status]

    return [
        AgentSummary(
            name=a.name,
            status=a.status,
            description=a.description,
            requests_total=a.metrics.requests_total,
        )
        for a in agents
    ]


@router.get("/{agent_name}", response_model=AgentInfo, responses={**NOT_FOUND_RESPONSE})
async def get_agent(agent_name: str) -> AgentInfo:
    """
    Get detailed information about a specific agent.

    Returns comprehensive agent information including capabilities,
    supported intents, metrics, and current configuration.

    Args:
        agent_name: The unique name of the agent (e.g., 'whisper', 'smith', 'seshat')

    Returns:
        AgentInfo: Detailed agent information

    Raises:
        404: Agent with the specified name not found
    """
    store = get_store()
    agent = store.get(agent_name)

    if not agent:
        raise HTTPException(status_code=404, detail=f"Agent not found: {agent_name}")

    return agent


@router.get("/{agent_name}/metrics", response_model=AgentMetrics)
async def get_agent_metrics(agent_name: str) -> AgentMetrics:
    """Get performance metrics for an agent."""
    store = get_store()
    agent = store.get(agent_name)

    if not agent:
        raise HTTPException(status_code=404, detail=f"Agent not found: {agent_name}")

    return agent.metrics


@router.post("/{agent_name}/start", response_model=AgentControlResponse)
async def start_agent(
    agent_name: str,
    start_request: Optional[StartAgentRequest] = None,
    admin_user: str = Depends(require_admin_auth),
) -> AgentControlResponse:
    """
    Start an agent (requires admin authentication).

    Activates a stopped or idle agent so it can process requests.

    Returns:
        AgentControlResponse with status 'started' or 'already_active'

    Raises:
        401: Not authenticated
        403: Not admin
        404: Agent not found
    """
    store = get_store()
    agent = store.get(agent_name)

    if not agent:
        raise HTTPException(status_code=404, detail=f"Agent not found: {agent_name}")

    if agent.status == AgentStatus.ACTIVE:
        return AgentControlResponse(status="already_active", agent=agent_name)

    store.update_status(agent_name, AgentStatus.ACTIVE)
    store.add_log(
        AgentLogEntry(
            timestamp=datetime.utcnow(),
            level="INFO",
            message="Agent started",
            agent_name=agent_name,
        )
    )

    return AgentControlResponse(status="started", agent=agent_name)


@router.post("/{agent_name}/stop", response_model=AgentControlResponse)
async def stop_agent(
    agent_name: str,
    admin_user: str = Depends(require_admin_auth),
) -> AgentControlResponse:
    """
    Stop an agent (requires admin authentication).

    Deactivates an agent so it will no longer process requests.

    Returns:
        AgentControlResponse with status 'stopped' or 'already_stopped'

    Raises:
        401: Not authenticated
        403: Not admin
        404: Agent not found
    """
    store = get_store()
    agent = store.get(agent_name)

    if not agent:
        raise HTTPException(status_code=404, detail=f"Agent not found: {agent_name}")

    if agent.status == AgentStatus.DISABLED:
        return AgentControlResponse(status="already_stopped", agent=agent_name)

    store.update_status(agent_name, AgentStatus.DISABLED)
    store.add_log(
        AgentLogEntry(
            timestamp=datetime.utcnow(),
            level="INFO",
            message="Agent stopped",
            agent_name=agent_name,
        )
    )

    return AgentControlResponse(status="stopped", agent=agent_name)


@router.post("/{agent_name}/restart", response_model=AgentControlResponse)
async def restart_agent(
    agent_name: str,
    admin_user: str = Depends(require_admin_auth),
) -> AgentControlResponse:
    """
    Restart an agent (requires admin authentication).

    Stops and immediately restarts an agent. Useful for applying configuration changes.

    Returns:
        AgentControlResponse with status 'restarted'

    Raises:
        401: Not authenticated
        403: Not admin
        404: Agent not found
    """
    store = get_store()
    agent = store.get(agent_name)

    if not agent:
        raise HTTPException(status_code=404, detail=f"Agent not found: {agent_name}")

    store.update_status(agent_name, AgentStatus.ACTIVE)
    store.add_log(
        AgentLogEntry(
            timestamp=datetime.utcnow(),
            level="INFO",
            message="Agent restarted",
            agent_name=agent_name,
        )
    )

    return AgentControlResponse(status="restarted", agent=agent_name)


@router.get("/{agent_name}/logs", response_model=List[AgentLogEntry])
async def get_agent_logs(
    agent_name: str,
    level: Optional[str] = None,
    limit: int = Query(default=100, le=1000),
    admin_user: str = Depends(require_admin_auth),
) -> List[AgentLogEntry]:
    """Get logs for an agent (requires admin authentication)."""
    store = get_store()
    agent = store.get(agent_name)

    if not agent:
        raise HTTPException(status_code=404, detail=f"Agent not found: {agent_name}")

    return store.get_logs(agent_name=agent_name, level=level, limit=limit)


@router.get("/logs/all", response_model=List[AgentLogEntry])
async def get_all_logs(
    level: Optional[str] = None,
    limit: int = Query(default=100, le=1000),
    admin_user: str = Depends(require_admin_auth),
) -> List[AgentLogEntry]:
    """Get logs from all agents (requires admin authentication)."""
    store = get_store()
    return store.get_logs(level=level, limit=limit)


@router.get("/stats/overview", response_model=AgentsOverviewResponse)
async def get_agents_overview() -> AgentsOverviewResponse:
    """
    Get overview statistics for all agents.

    Returns aggregate metrics across all registered agents including:
    - Total agent count and status distribution
    - Request counts (total, success, failed)
    - Overall success rate
    """
    store = get_store()
    agents = store.get_all()

    total_requests = sum(a.metrics.requests_total for a in agents)
    total_success = sum(a.metrics.requests_success for a in agents)
    total_failed = sum(a.metrics.requests_failed for a in agents)

    status_counts = {}
    for agent in agents:
        status = agent.status.value
        status_counts[status] = status_counts.get(status, 0) + 1

    return AgentsOverviewResponse(
        total_agents=len(agents),
        status_distribution=status_counts,
        total_requests=total_requests,
        total_success=total_success,
        total_failed=total_failed,
        success_rate=(total_success / total_requests * 100) if total_requests > 0 else 0,
    )
