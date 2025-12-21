"""
Agent Monitoring API Routes

Provides endpoints for monitoring and managing agents.
"""

import logging
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)
router = APIRouter()


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
    Mock agent store for the web interface.

    In production, this would integrate with the actual AgentRegistry.
    """

    def __init__(self):
        # Initialize with some mock agents
        self._agents: Dict[str, AgentInfo] = {
            "whisper": AgentInfo(
                name="whisper",
                description="Intent classification and request routing",
                status=AgentStatus.ACTIVE,
                capabilities=[
                    AgentCapability(name="intent_classification", description="Classify user intents"),
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
                    AgentCapability(name="constitutional_check", description="Validate against constitution"),
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
            "muse": AgentInfo(
                name="muse",
                description="Creative content generation",
                status=AgentStatus.IDLE,
                capabilities=[
                    AgentCapability(name="creative_writing", description="Generate creative content"),
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
            "oracle": AgentInfo(
                name="oracle",
                description="Knowledge retrieval and factual queries",
                status=AgentStatus.IDLE,
                capabilities=[
                    AgentCapability(name="knowledge_retrieval", description="Retrieve information"),
                    AgentCapability(name="fact_checking", description="Verify facts"),
                ],
                supported_intents=["query.*", "factual.*"],
                metrics=AgentMetrics(
                    requests_total=856,
                    requests_success=840,
                    requests_failed=16,
                    average_response_time_ms=890.3,
                    uptime_seconds=86400,
                ),
                created_at=datetime.utcnow(),
            ),
        }
        self._logs: List[AgentLogEntry] = []

    def get_all(self) -> List[AgentInfo]:
        """Get all agents."""
        return list(self._agents.values())

    def get(self, name: str) -> Optional[AgentInfo]:
        """Get an agent by name."""
        return self._agents.get(name)

    def update_status(self, name: str, status: AgentStatus) -> bool:
        """Update an agent's status."""
        if name in self._agents:
            self._agents[name].status = status
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


@router.get("/{agent_name}", response_model=AgentInfo)
async def get_agent(agent_name: str) -> AgentInfo:
    """Get detailed information about a specific agent."""
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


@router.post("/{agent_name}/start")
async def start_agent(
    agent_name: str,
    request: Optional[StartAgentRequest] = None,
) -> Dict[str, Any]:
    """Start an agent."""
    store = get_store()
    agent = store.get(agent_name)

    if not agent:
        raise HTTPException(status_code=404, detail=f"Agent not found: {agent_name}")

    if agent.status == AgentStatus.ACTIVE:
        return {"status": "already_active", "agent": agent_name}

    store.update_status(agent_name, AgentStatus.ACTIVE)
    store.add_log(
        AgentLogEntry(
            timestamp=datetime.utcnow(),
            level="INFO",
            message=f"Agent started",
            agent_name=agent_name,
        )
    )

    return {"status": "started", "agent": agent_name}


@router.post("/{agent_name}/stop")
async def stop_agent(agent_name: str) -> Dict[str, Any]:
    """Stop an agent."""
    store = get_store()
    agent = store.get(agent_name)

    if not agent:
        raise HTTPException(status_code=404, detail=f"Agent not found: {agent_name}")

    if agent.status == AgentStatus.DISABLED:
        return {"status": "already_stopped", "agent": agent_name}

    store.update_status(agent_name, AgentStatus.DISABLED)
    store.add_log(
        AgentLogEntry(
            timestamp=datetime.utcnow(),
            level="INFO",
            message=f"Agent stopped",
            agent_name=agent_name,
        )
    )

    return {"status": "stopped", "agent": agent_name}


@router.post("/{agent_name}/restart")
async def restart_agent(agent_name: str) -> Dict[str, Any]:
    """Restart an agent."""
    store = get_store()
    agent = store.get(agent_name)

    if not agent:
        raise HTTPException(status_code=404, detail=f"Agent not found: {agent_name}")

    store.update_status(agent_name, AgentStatus.ACTIVE)
    store.add_log(
        AgentLogEntry(
            timestamp=datetime.utcnow(),
            level="INFO",
            message=f"Agent restarted",
            agent_name=agent_name,
        )
    )

    return {"status": "restarted", "agent": agent_name}


@router.get("/{agent_name}/logs", response_model=List[AgentLogEntry])
async def get_agent_logs(
    agent_name: str,
    level: Optional[str] = None,
    limit: int = Query(default=100, le=1000),
) -> List[AgentLogEntry]:
    """Get logs for an agent."""
    store = get_store()
    agent = store.get(agent_name)

    if not agent:
        raise HTTPException(status_code=404, detail=f"Agent not found: {agent_name}")

    return store.get_logs(agent_name=agent_name, level=level, limit=limit)


@router.get("/logs/all", response_model=List[AgentLogEntry])
async def get_all_logs(
    level: Optional[str] = None,
    limit: int = Query(default=100, le=1000),
) -> List[AgentLogEntry]:
    """Get logs from all agents."""
    store = get_store()
    return store.get_logs(level=level, limit=limit)


@router.get("/stats/overview")
async def get_agents_overview() -> Dict[str, Any]:
    """Get overview statistics for all agents."""
    store = get_store()
    agents = store.get_all()

    total_requests = sum(a.metrics.requests_total for a in agents)
    total_success = sum(a.metrics.requests_success for a in agents)
    total_failed = sum(a.metrics.requests_failed for a in agents)

    status_counts = {}
    for agent in agents:
        status = agent.status.value
        status_counts[status] = status_counts.get(status, 0) + 1

    return {
        "total_agents": len(agents),
        "status_distribution": status_counts,
        "total_requests": total_requests,
        "total_success": total_success,
        "total_failed": total_failed,
        "success_rate": (total_success / total_requests * 100) if total_requests > 0 else 0,
    }
