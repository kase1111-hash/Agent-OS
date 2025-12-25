"""
Learning Contracts API Routes

Provides endpoints for managing learning consent contracts.
"""

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)
router = APIRouter()

# Try to import real contracts module
try:
    from src.contracts import (
        ContractStore,
        LearningContract,
        ContractType,
        ContractStatus,
        ContractScope,
        LearningScope,
        ContractQuery,
        create_contract_store,
        CONTRACT_TEMPLATES,
        ContractTemplate,
        get_template,
        list_templates,
        create_contract_from_template,
        ensure_default_contracts,
    )
    REAL_CONTRACTS_AVAILABLE = True
except ImportError as e:
    REAL_CONTRACTS_AVAILABLE = False
    logger.warning(f"Real contracts module not available: {e}")


# =============================================================================
# Models
# =============================================================================


class ContractTypeModel(BaseModel):
    """Contract type information."""

    name: str
    description: str
    allows_storage: bool = True
    allows_generalization: bool = False
    allows_cross_context: bool = False
    allows_long_term_patterns: bool = False


class ContractTemplateModel(BaseModel):
    """Contract template for quick creation."""

    id: str
    name: str
    description: str
    contract_type: str
    default_domains: List[str] = Field(default_factory=list)
    default_duration_days: Optional[int] = None
    recommended_for: str = ""


class ContractModel(BaseModel):
    """Learning contract model."""

    id: str
    user_id: str
    contract_type: str
    status: str
    domains: List[str] = Field(default_factory=list)
    created_at: datetime
    expires_at: Optional[datetime] = None
    description: str = ""
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ContractSummary(BaseModel):
    """Summary of a contract for listing."""

    id: str
    contract_type: str
    status: str
    domains: List[str] = Field(default_factory=list)
    created_at: datetime
    expires_at: Optional[datetime] = None


class CreateContractRequest(BaseModel):
    """Request to create a new contract."""

    user_id: str = "default"
    contract_type: str
    domains: List[str] = Field(default_factory=list)
    duration_days: Optional[int] = None
    description: str = ""
    metadata: Dict[str, Any] = Field(default_factory=dict)


class CreateFromTemplateRequest(BaseModel):
    """Request to create a contract from a template."""

    template_id: str
    user_id: str = "default"
    domains: Optional[List[str]] = None
    duration_days: Optional[int] = None


class UpdateContractRequest(BaseModel):
    """Request to update a contract."""

    domains: Optional[List[str]] = None
    description: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class ContractsStats(BaseModel):
    """Contracts statistics."""

    total_contracts: int = 0
    active_contracts: int = 0
    pending_contracts: int = 0
    expired_contracts: int = 0
    revoked_contracts: int = 0
    contracts_by_type: Dict[str, int] = Field(default_factory=dict)


# =============================================================================
# Mock Data Store
# =============================================================================


class ContractsStore:
    """
    Contracts store that integrates with the real ContractStore.

    Falls back to mock data if real contracts aren't available.
    """

    def __init__(self):
        self._real_store: Optional[Any] = None
        self._mock_contracts: Dict[str, ContractModel] = {}
        self._mock_templates: Dict[str, ContractTemplateModel] = {}
        self._use_real_contracts = False

        # Try to initialize real contract store
        if REAL_CONTRACTS_AVAILABLE:
            try:
                self._real_store = create_contract_store()
                # Ensure default Agent-OS contracts exist
                ensure_default_contracts(self._real_store, user_id="default")
                self._use_real_contracts = True
                logger.info("Connected to real contracts store with default contracts")
            except Exception as e:
                logger.warning(f"Failed to initialize real contracts store: {e}")
                self._use_real_contracts = False

        # Initialize mock data as fallback
        if not self._use_real_contracts:
            self._init_mock_data()

    def _init_mock_data(self):
        """Initialize mock contracts and templates matching real Agent-OS defaults."""
        now = datetime.utcnow()

        # Mock templates matching CONTRACT_TEMPLATES from store.py
        self._mock_templates = {
            "coding": ContractTemplateModel(
                id="coding",
                name="Coding Assistant",
                description="Learn from code patterns, style preferences, and project structures",
                contract_type="PROCEDURAL",
                default_domains=["coding", "programming", "development", "debugging"],
                default_duration_days=365,
                recommended_for="Developers wanting personalized coding assistance",
            ),
            "journaling": ContractTemplateModel(
                id="journaling",
                name="Personal Journal",
                description="Remember personal reflections with highest privacy",
                contract_type="EPISODIC",
                default_domains=["personal", "journal", "diary", "notes"],
                default_duration_days=None,
                recommended_for="Private journaling and self-reflection",
            ),
            "work_projects": ContractTemplateModel(
                id="work_projects",
                name="Work Projects",
                description="Learn project patterns while respecting confidentiality",
                contract_type="PROCEDURAL",
                default_domains=["work", "professional", "business"],
                default_duration_days=180,
                recommended_for="Professional project management",
            ),
            "gaming": ContractTemplateModel(
                id="gaming",
                name="Gaming Assistant",
                description="Remember game preferences and strategies",
                contract_type="EPISODIC",
                default_domains=["gaming", "games", "entertainment"],
                default_duration_days=1,
                recommended_for="Gamers wanting session-based memory",
            ),
            "study": ContractTemplateModel(
                id="study",
                name="Study Assistant",
                description="Learn study patterns and help with retention",
                contract_type="PROCEDURAL",
                default_domains=["education", "study", "learning", "courses"],
                default_duration_days=180,
                recommended_for="Students wanting learning assistance",
            ),
            "restricted": ContractTemplateModel(
                id="restricted",
                name="Restricted Domains",
                description="Block learning for sensitive domains (medical, financial, legal)",
                contract_type="PROHIBITED",
                default_domains=["medical", "financial", "legal", "credentials"],
                default_duration_days=None,
                recommended_for="GDPR/HIPAA compliant privacy protection",
            ),
            "strategy": ContractTemplateModel(
                id="strategy",
                name="Strategic Learning",
                description="High-trust long-term pattern learning across contexts",
                contract_type="STRATEGIC",
                default_domains=["general"],
                default_duration_days=365,
                recommended_for="Trusted long-term AI relationships",
            ),
        }

        # Mock contracts matching default Agent-OS contracts
        self._mock_contracts = {
            # 1. Code Assistance Contract
            "aos-code-001": ContractModel(
                id="aos-code-001",
                user_id="default",
                contract_type="PROCEDURAL",
                status="ACTIVE",
                domains=["coding", "programming", "development", "debugging", "refactoring"],
                created_at=now - timedelta(days=30),
                expires_at=now + timedelta(days=335),
                description="Enables Agent-OS to learn your coding patterns, style preferences, and project conventions for better code assistance.",
                metadata={"category": "technical", "trust_level": "medium", "created_by": "agent_os_defaults"},
            ),
            # 2. Memory Management Contract (Seshat)
            "aos-memory-002": ContractModel(
                id="aos-memory-002",
                user_id="default",
                contract_type="EPISODIC",
                status="ACTIVE",
                domains=["memory", "recall", "context", "conversation"],
                created_at=now - timedelta(days=30),
                expires_at=None,
                description="Allows the Seshat memory agent to store conversation context and recall relevant information without cross-context generalization.",
                metadata={"category": "memory", "agent": "seshat", "created_by": "agent_os_defaults"},
            ),
            # 3. Constitutional Compliance Contract (Smith)
            "aos-smith-003": ContractModel(
                id="aos-smith-003",
                user_id="default",
                contract_type="OBSERVATION",
                status="ACTIVE",
                domains=["constitution", "safety", "compliance", "ethics"],
                created_at=now - timedelta(days=30),
                expires_at=None,
                description="Allows the Smith constitutional agent to observe interactions for safety compliance without storing or learning from content.",
                metadata={"category": "safety", "agent": "smith", "observation_only": True},
            ),
            # 4. Security Prohibition Contract
            "aos-security-004": ContractModel(
                id="aos-security-004",
                user_id="default",
                contract_type="PROHIBITED",
                status="ACTIVE",
                domains=["credentials", "passwords", "api_keys", "secrets", "tokens", "private_keys"],
                created_at=now - timedelta(days=30),
                expires_at=None,
                description="Explicitly prohibits Agent-OS from learning or storing any credentials, API keys, passwords, or other security-sensitive data.",
                metadata={"category": "security", "immutable": True, "priority": "critical"},
            ),
            # 5. Intent Classification Contract (Whisper)
            "aos-whisper-005": ContractModel(
                id="aos-whisper-005",
                user_id="default",
                contract_type="PROCEDURAL",
                status="ACTIVE",
                domains=["intent", "routing", "classification", "commands"],
                created_at=now - timedelta(days=30),
                expires_at=now + timedelta(days=150),
                description="Enables the Whisper agent to learn user intent patterns for improved request classification and agent routing.",
                metadata={"category": "routing", "agent": "whisper", "created_by": "agent_os_defaults"},
            ),
            # 6. Personal Data Protection Contract
            "aos-privacy-006": ContractModel(
                id="aos-privacy-006",
                user_id="default",
                contract_type="PROHIBITED",
                status="ACTIVE",
                domains=["medical", "health", "financial", "banking", "legal", "biometric"],
                created_at=now - timedelta(days=30),
                expires_at=None,
                description="Prohibits Agent-OS from learning personal health, financial, legal, or biometric information to protect user privacy.",
                metadata={"category": "privacy", "gdpr_compliant": True, "hipaa_compliant": True},
            ),
            # 7. General Assistance Contract
            "aos-general-007": ContractModel(
                id="aos-general-007",
                user_id="default",
                contract_type="EPISODIC",
                status="ACTIVE",
                domains=["general", "chat", "assistance", "help", "questions"],
                created_at=now - timedelta(days=30),
                expires_at=now + timedelta(days=0),  # Expires today for demo
                description="Allows Agent-OS to remember conversation context for general assistance without cross-session learning.",
                metadata={"category": "general", "session_memory": True},
            ),
        }

    def _convert_real_contract(self, contract: Any) -> ContractModel:
        """Convert a real LearningContract to ContractModel."""
        return ContractModel(
            id=contract.id,
            user_id=contract.user_id,
            contract_type=contract.contract_type.name if hasattr(contract.contract_type, 'name') else str(contract.contract_type),
            status=contract.status.name if hasattr(contract.status, 'name') else str(contract.status),
            domains=list(contract.scope.domains) if hasattr(contract.scope, 'domains') else [],
            created_at=contract.created_at,
            expires_at=contract.expires_at,
            description=contract.description if hasattr(contract, 'description') else "",
            metadata=contract.metadata if hasattr(contract, 'metadata') else {},
        )

    def _convert_real_template(self, template: Any) -> ContractTemplateModel:
        """Convert a real ContractTemplate to ContractTemplateModel."""
        return ContractTemplateModel(
            id=template.id,
            name=template.name,
            description=template.description,
            contract_type=template.contract_type.name if hasattr(template.contract_type, 'name') else str(template.contract_type),
            default_domains=list(template.default_domains) if hasattr(template, 'default_domains') else [],
            default_duration_days=template.default_duration_days if hasattr(template, 'default_duration_days') else None,
            recommended_for=template.recommended_for if hasattr(template, 'recommended_for') else "",
        )

    def get_templates(self) -> List[ContractTemplateModel]:
        """Get all available templates."""
        if self._use_real_contracts:
            try:
                templates = list_templates()
                return [self._convert_real_template(t) for t in templates]
            except Exception as e:
                logger.error(f"Error getting templates: {e}")
        return list(self._mock_templates.values())

    def get_template(self, template_id: str) -> Optional[ContractTemplateModel]:
        """Get a specific template."""
        if self._use_real_contracts:
            try:
                template = get_template(template_id)
                if template:
                    return self._convert_real_template(template)
            except Exception as e:
                logger.error(f"Error getting template {template_id}: {e}")
        return self._mock_templates.get(template_id)

    def get_contracts(self, status: Optional[str] = None, user_id: str = "default") -> List[ContractModel]:
        """Get all contracts, optionally filtered by status."""
        if self._use_real_contracts and self._real_store:
            try:
                query = ContractQuery(user_id=user_id)
                if status:
                    query.status = ContractStatus[status]
                contracts = self._real_store.query(query)
                return [self._convert_real_contract(c) for c in contracts]
            except Exception as e:
                logger.error(f"Error getting contracts: {e}")

        contracts = list(self._mock_contracts.values())
        if status:
            contracts = [c for c in contracts if c.status == status]
        return contracts

    def get_contract(self, contract_id: str) -> Optional[ContractModel]:
        """Get a specific contract."""
        if self._use_real_contracts and self._real_store:
            try:
                contract = self._real_store.get(contract_id)
                if contract:
                    return self._convert_real_contract(contract)
            except Exception as e:
                logger.error(f"Error getting contract {contract_id}: {e}")
        return self._mock_contracts.get(contract_id)

    def create_contract(self, request: CreateContractRequest) -> ContractModel:
        """Create a new contract."""
        now = datetime.utcnow()
        expires_at = None
        if request.duration_days:
            expires_at = now + timedelta(days=request.duration_days)

        if self._use_real_contracts and self._real_store:
            try:
                contract = LearningContract(
                    user_id=request.user_id,
                    contract_type=ContractType[request.contract_type],
                    scope=ContractScope(domains=set(request.domains)),
                    expires_at=expires_at,
                    description=request.description,
                    metadata=request.metadata,
                )
                self._real_store.store(contract)
                return self._convert_real_contract(contract)
            except Exception as e:
                logger.error(f"Error creating contract: {e}")

        # Mock creation
        contract_id = f"contract-{len(self._mock_contracts) + 1:03d}"
        contract = ContractModel(
            id=contract_id,
            user_id=request.user_id,
            contract_type=request.contract_type,
            status="ACTIVE",
            domains=request.domains,
            created_at=now,
            expires_at=expires_at,
            description=request.description,
            metadata=request.metadata,
        )
        self._mock_contracts[contract_id] = contract
        return contract

    def create_from_template(self, request: CreateFromTemplateRequest) -> ContractModel:
        """Create a contract from a template."""
        template = self.get_template(request.template_id)
        if not template:
            raise ValueError(f"Template not found: {request.template_id}")

        domains = request.domains if request.domains else template.default_domains
        duration = request.duration_days if request.duration_days else template.default_duration_days

        create_request = CreateContractRequest(
            user_id=request.user_id,
            contract_type=template.contract_type,
            domains=domains,
            duration_days=duration,
            description=f"Created from template: {template.name}",
        )
        return self.create_contract(create_request)

    def revoke_contract(self, contract_id: str) -> bool:
        """Revoke a contract."""
        if self._use_real_contracts and self._real_store:
            try:
                return self._real_store.revoke(contract_id)
            except Exception as e:
                logger.error(f"Error revoking contract {contract_id}: {e}")

        if contract_id in self._mock_contracts:
            self._mock_contracts[contract_id].status = "REVOKED"
            return True
        return False

    def get_stats(self, user_id: str = "default") -> ContractsStats:
        """Get contract statistics."""
        contracts = self.get_contracts(user_id=user_id)

        stats = ContractsStats(
            total_contracts=len(contracts),
            active_contracts=sum(1 for c in contracts if c.status == "ACTIVE"),
            pending_contracts=sum(1 for c in contracts if c.status == "PENDING"),
            expired_contracts=sum(1 for c in contracts if c.status == "EXPIRED"),
            revoked_contracts=sum(1 for c in contracts if c.status == "REVOKED"),
        )

        for contract in contracts:
            ctype = contract.contract_type
            stats.contracts_by_type[ctype] = stats.contracts_by_type.get(ctype, 0) + 1

        return stats

    def get_contract_types(self) -> List[ContractTypeModel]:
        """Get available contract types."""
        types = [
            ContractTypeModel(
                name="OBSERVATION",
                description="Permits watching signals only - no storage or inference",
                allows_storage=False,
                allows_generalization=False,
                allows_cross_context=False,
                allows_long_term_patterns=False,
            ),
            ContractTypeModel(
                name="EPISODIC",
                description="Store specific instances only - no cross-context generalization",
                allows_storage=True,
                allows_generalization=False,
                allows_cross_context=False,
                allows_long_term_patterns=False,
            ),
            ContractTypeModel(
                name="PROCEDURAL",
                description="Derive reusable heuristics and patterns",
                allows_storage=True,
                allows_generalization=True,
                allows_cross_context=False,
                allows_long_term_patterns=False,
            ),
            ContractTypeModel(
                name="STRATEGIC",
                description="Long-term pattern inference across contexts",
                allows_storage=True,
                allows_generalization=True,
                allows_cross_context=True,
                allows_long_term_patterns=True,
            ),
            ContractTypeModel(
                name="PROHIBITED",
                description="Explicitly blocks all learning from this domain",
                allows_storage=False,
                allows_generalization=False,
                allows_cross_context=False,
                allows_long_term_patterns=False,
            ),
        ]
        return types

    @property
    def is_using_real_contracts(self) -> bool:
        """Check if using real contracts store."""
        return self._use_real_contracts


# Global store instance
_store: Optional[ContractsStore] = None


def get_store() -> ContractsStore:
    """Get the contracts store."""
    global _store
    if _store is None:
        _store = ContractsStore()
    return _store


# =============================================================================
# Endpoints
# =============================================================================


@router.get("/", response_model=List[ContractSummary])
async def list_contracts(
    status: Optional[str] = None,
    user_id: str = "default",
) -> List[ContractSummary]:
    """
    List all contracts.

    Optionally filter by status (ACTIVE, PENDING, EXPIRED, REVOKED).
    """
    store = get_store()
    contracts = store.get_contracts(status=status, user_id=user_id)

    return [
        ContractSummary(
            id=c.id,
            contract_type=c.contract_type,
            status=c.status,
            domains=c.domains,
            created_at=c.created_at,
            expires_at=c.expires_at,
        )
        for c in contracts
    ]


@router.get("/stats", response_model=ContractsStats)
async def get_contracts_stats(user_id: str = "default") -> ContractsStats:
    """Get contracts statistics."""
    store = get_store()
    return store.get_stats(user_id=user_id)


@router.get("/types", response_model=List[ContractTypeModel])
async def get_contract_types() -> List[ContractTypeModel]:
    """Get available contract types with their permissions."""
    store = get_store()
    return store.get_contract_types()


@router.get("/templates", response_model=List[ContractTemplateModel])
async def get_templates() -> List[ContractTemplateModel]:
    """Get all available contract templates."""
    store = get_store()
    return store.get_templates()


@router.get("/templates/{template_id}", response_model=ContractTemplateModel)
async def get_template_by_id(template_id: str) -> ContractTemplateModel:
    """Get a specific template by ID."""
    store = get_store()
    template = store.get_template(template_id)

    if not template:
        raise HTTPException(status_code=404, detail=f"Template not found: {template_id}")

    return template


@router.get("/{contract_id}", response_model=ContractModel)
async def get_contract(contract_id: str) -> ContractModel:
    """Get detailed information about a specific contract."""
    store = get_store()
    contract = store.get_contract(contract_id)

    if not contract:
        raise HTTPException(status_code=404, detail=f"Contract not found: {contract_id}")

    return contract


@router.post("/", response_model=ContractModel)
async def create_contract(request: CreateContractRequest) -> ContractModel:
    """Create a new learning contract."""
    store = get_store()

    # Validate contract type
    valid_types = [t.name for t in store.get_contract_types()]
    if request.contract_type not in valid_types:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid contract type. Valid types: {valid_types}"
        )

    return store.create_contract(request)


@router.post("/from-template", response_model=ContractModel)
async def create_contract_from_template(request: CreateFromTemplateRequest) -> ContractModel:
    """Create a contract from a template."""
    store = get_store()

    try:
        return store.create_from_template(request)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.post("/{contract_id}/revoke")
async def revoke_contract(contract_id: str) -> Dict[str, Any]:
    """Revoke an active contract."""
    store = get_store()
    contract = store.get_contract(contract_id)

    if not contract:
        raise HTTPException(status_code=404, detail=f"Contract not found: {contract_id}")

    if contract.status == "REVOKED":
        return {"status": "already_revoked", "contract_id": contract_id}

    success = store.revoke_contract(contract_id)

    if success:
        return {"status": "revoked", "contract_id": contract_id}
    else:
        raise HTTPException(status_code=500, detail="Failed to revoke contract")


@router.delete("/{contract_id}")
async def delete_contract(contract_id: str) -> Dict[str, Any]:
    """Delete a contract (same as revoke for safety)."""
    return await revoke_contract(contract_id)
