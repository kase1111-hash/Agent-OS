"""
Learning Contracts API Routes

Provides endpoints for managing learning consent contracts.
Each user has their own contracts - contracts are isolated by user_id.

Backed by the persistent ``src.contracts`` ContractStore, stored in
``<data_dir>/contracts.db``.
"""

import logging
import math
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Cookie, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from src.contracts import (
    ContractQuery,
    ContractScope,
    ContractStatus,
    ContractStore,
    ContractTemplate,
    ContractType,
    LearningContract,
    LearningScope,
    create_contract_store,
)
from src.contracts import get_template as get_contract_template
from src.contracts import list_templates as list_contract_templates

from ..auth_helpers import require_authenticated_user

logger = logging.getLogger(__name__)
router = APIRouter()


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
# Contracts Store Adapter
# =============================================================================


class TemplateNotFoundError(LookupError):
    """Raised when a contract template name is unknown."""


# Contract types offered through the web API (the learning-contracts spec types).
# The legacy ContractType members (FULL_CONSENT, LIMITED_CONSENT, ...) are not offered.
_CONTRACT_TYPE_DESCRIPTIONS: Dict[ContractType, str] = {
    ContractType.OBSERVATION: "Permits watching signals only - no storage or inference",
    ContractType.EPISODIC: "Store specific instances only - no cross-context generalization",
    ContractType.PROCEDURAL: "Derive reusable heuristics and patterns",
    ContractType.STRATEGIC: "Long-term pattern inference across contexts",
    ContractType.PROHIBITED: "Explicitly blocks all learning from this domain",
}

# Display names for the templates in CONTRACT_TEMPLATES (keyed by template name).
_TEMPLATE_DISPLAY: Dict[str, Dict[str, str]] = {
    "coding": {
        "name": "Coding Assistant",
        "recommended_for": "Developers wanting personalized coding assistance",
    },
    "gaming": {
        "name": "Gaming Assistant",
        "recommended_for": "Gamers wanting session-based memory",
    },
    "journaling": {
        "name": "Personal Journal",
        "recommended_for": "Private journaling and self-reflection",
    },
    "work_projects": {
        "name": "Work Projects",
        "recommended_for": "Professional project management",
    },
    "restricted": {
        "name": "Restricted Domains",
        "recommended_for": "GDPR/HIPAA compliant privacy protection",
    },
    "study": {
        "name": "Study Assistant",
        "recommended_for": "Students wanting learning assistance",
    },
    "strategy": {
        "name": "Strategic Learning",
        "recommended_for": "Trusted long-term AI relationships",
    },
}

# ContractStore clamps query limits to this value; used for "all of a user's contracts".
_MAX_QUERY_LIMIT = 10000

# Upper bound for duration_days (100 years); keeps expiry dates representable.
_MAX_DURATION_DAYS = 36500


def _clean_domains(domains: Optional[List[str]]) -> List[str]:
    """Strip whitespace, drop empty entries and duplicates (order preserved)."""
    cleaned: List[str] = []
    for domain in domains or []:
        domain = domain.strip()
        if domain and domain not in cleaned:
            cleaned.append(domain)
    return cleaned


def _duration_from_days(duration_days: Optional[int]) -> Optional[timedelta]:
    """Convert a request's duration in days to a timedelta (None/0 = no expiry)."""
    if not duration_days:
        return None
    if duration_days < 0 or duration_days > _MAX_DURATION_DAYS:
        raise ValueError(f"duration_days must be between 1 and {_MAX_DURATION_DAYS}")
    return timedelta(days=duration_days)


class ContractsStore:
    """
    Web adapter over the persistent ``src.contracts.ContractStore``.

    Every contract read and write is scoped to a user: a contract owned by
    another user is reported as not found.
    """

    def __init__(self, db_path: Optional[Path] = None):
        """
        Open (or create) the contracts database.

        Args:
            db_path: SQLite database file; None keeps contracts in memory only.
        """
        if db_path is not None:
            db_path = Path(db_path)
            db_path.parent.mkdir(parents=True, exist_ok=True)

        self._store: ContractStore = create_contract_store(db_path=db_path)

        if db_path is not None:
            from ..dependencies import _harden_sqlite_path

            _harden_sqlite_path(db_path)
        logger.info(f"Contracts store ready ({db_path or 'in-memory'})")

    def close(self) -> None:
        """Close the underlying database connection."""
        self._store.close()

    # -- conversion helpers ---------------------------------------------------

    @staticmethod
    def _to_model(contract: LearningContract) -> ContractModel:
        """Convert a LearningContract to the API model."""
        return ContractModel(
            id=contract.contract_id,
            user_id=contract.user_id,
            contract_type=contract.contract_type.name,
            status=contract.status.name,
            domains=sorted(contract.scope.domains),
            created_at=contract.created_at,
            expires_at=contract.expires_at,
            description=contract.description,
            metadata=contract.metadata,
        )

    @staticmethod
    def _template_to_model(template: ContractTemplate) -> ContractTemplateModel:
        """Convert a ContractTemplate to the API model (id = template name)."""
        display = _TEMPLATE_DISPLAY.get(template.name, {})
        duration_days = None
        if template.default_duration:
            duration_days = max(1, math.ceil(template.default_duration.total_seconds() / 86400))
        return ContractTemplateModel(
            id=template.name,
            name=display.get("name", template.name.replace("_", " ").title()),
            description=template.description,
            contract_type=template.contract_type.name,
            default_domains=sorted(template.scope.domains),
            default_duration_days=duration_days,
            recommended_for=display.get("recommended_for", ""),
        )

    @staticmethod
    def _parse_contract_type(name: str) -> ContractType:
        """Resolve an API contract type name (e.g. "EPISODIC")."""
        for contract_type in _CONTRACT_TYPE_DESCRIPTIONS:
            if contract_type.name == name.strip().upper():
                return contract_type
        valid = [t.name for t in _CONTRACT_TYPE_DESCRIPTIONS]
        raise ValueError(f"Invalid contract type. Valid types: {valid}")

    @staticmethod
    def _parse_status(name: str) -> ContractStatus:
        """Resolve an API status filter (e.g. "ACTIVE", case-insensitive)."""
        try:
            return ContractStatus[name.strip().upper()]
        except KeyError:
            valid = [s.name for s in ContractStatus]
            raise ValueError(f"Invalid status. Valid statuses: {valid}") from None

    def _refresh_expiry(self, contract: LearningContract) -> LearningContract:
        """Persist the ACTIVE -> EXPIRED transition once a contract's expiry has passed."""
        if (
            contract.status == ContractStatus.ACTIVE
            and contract.expires_at is not None
            and datetime.utcnow() >= contract.expires_at
        ):
            self._store.expire_contract(contract.contract_id)
            contract.status = ContractStatus.EXPIRED
        return contract

    def _get_owned(self, contract_id: str, user_id: str) -> Optional[LearningContract]:
        """Fetch a contract only if it belongs to user_id."""
        if not user_id:
            return None
        contract = self._store.get_contract(contract_id)
        if contract is None or contract.user_id != user_id:
            return None
        return self._refresh_expiry(contract)

    # -- templates and types --------------------------------------------------

    def get_templates(self) -> List[ContractTemplateModel]:
        """Get all available templates."""
        templates = (get_contract_template(name) for name in list_contract_templates())
        return [self._template_to_model(t) for t in templates if t is not None]

    def get_template(self, template_id: str) -> Optional[ContractTemplateModel]:
        """Get a specific template by id (its name, e.g. "coding")."""
        template = get_contract_template(template_id)
        return self._template_to_model(template) if template else None

    def get_contract_types(self) -> List[ContractTypeModel]:
        """Get the contract types that can be created, with their permissions."""
        return [
            ContractTypeModel(
                name=contract_type.name,
                description=description,
                allows_storage=contract_type.allows_storage(),
                allows_generalization=contract_type.allows_generalization(),
                allows_cross_context=contract_type.allows_cross_context(),
                allows_long_term_patterns=contract_type.allows_long_term_patterns(),
            )
            for contract_type, description in _CONTRACT_TYPE_DESCRIPTIONS.items()
        ]

    # -- contracts ------------------------------------------------------------

    def get_contracts(self, user_id: str, status: Optional[str] = None) -> List[ContractModel]:
        """
        Get a user's contracts, newest first, optionally filtered by status.

        Raises:
            ValueError: If status is not a ContractStatus name.
        """
        status_filter = self._parse_status(status) if status else None
        if not user_id:
            # An empty user_id would make ContractQuery match every user's contracts.
            return []

        query = ContractQuery(user_id=user_id, include_expired=True, limit=_MAX_QUERY_LIMIT)
        contracts = [self._refresh_expiry(c) for c in self._store.query_contracts(query)]
        if status_filter is not None:
            contracts = [c for c in contracts if c.status == status_filter]
        return [self._to_model(c) for c in contracts]

    def get_contract(self, contract_id: str, user_id: str) -> Optional[ContractModel]:
        """Get a contract owned by user_id (None if missing or owned by someone else)."""
        contract = self._get_owned(contract_id, user_id)
        return self._to_model(contract) if contract else None

    def create_contract(self, user_id: str, request: CreateContractRequest) -> ContractModel:
        """
        Create and activate a contract for user_id.

        An empty domain list creates a contract covering all domains.

        Raises:
            ValueError: If the contract type or duration is invalid.
        """
        if not user_id:
            raise ValueError("user_id is required")
        contract_type = self._parse_contract_type(request.contract_type)
        duration = _duration_from_days(request.duration_days)
        domains = _clean_domains(request.domains)
        scope = ContractScope(
            scope_type=LearningScope.DOMAIN_SPECIFIC if domains else LearningScope.ALL,
            domains=set(domains),
        )

        contract = self._store.create_contract(
            user_id=user_id,
            contract_type=contract_type,
            scope=scope,
            duration=duration,
            description=request.description,
            consent_method="explicit",
            metadata=dict(request.metadata),
            auto_activate=True,
        )
        return self._to_model(contract)

    def create_from_template(
        self, user_id: str, request: CreateFromTemplateRequest
    ) -> ContractModel:
        """
        Create and activate a contract for user_id from a template.

        Request domains, when given, replace the template's domains; the rest of
        the template scope (exclusions, content types, tasks) is kept.

        Raises:
            TemplateNotFoundError: If the template does not exist.
            ValueError: If the duration is invalid.
        """
        if not user_id:
            raise ValueError("user_id is required")
        template = get_contract_template(request.template_id)
        if template is None:
            raise TemplateNotFoundError(f"Template not found: {request.template_id}")

        duration = _duration_from_days(request.duration_days) or template.default_duration

        # Copy the scope so the shared template definition is never mutated.
        scope = ContractScope.from_dict(template.scope.to_dict())
        domains = _clean_domains(request.domains)
        if domains:
            scope.domains = set(domains)
            if scope.scope_type == LearningScope.ALL:
                scope.scope_type = LearningScope.DOMAIN_SPECIFIC

        contract = self._store.create_contract(
            user_id=user_id,
            contract_type=template.contract_type,
            scope=scope,
            duration=duration,
            description=template.description,
            consent_method="explicit",
            metadata={**template.metadata, "template": template.name},
            auto_activate=True,
        )
        return self._to_model(contract)

    def revoke_contract(self, contract_id: str, user_id: str, reason: str = "") -> bool:
        """Revoke a contract owned by user_id. Returns False if not found or not revocable."""
        if self._get_owned(contract_id, user_id) is None:
            return False
        return self._store.revoke_contract(contract_id, revoked_by=user_id, reason=reason)

    def get_stats(self, user_id: str) -> ContractsStats:
        """Get contract statistics for user_id."""
        contracts = self.get_contracts(user_id)

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


# Global store instance
_store: Optional[ContractsStore] = None


def get_store() -> ContractsStore:
    """Get the contracts store, persisted under the configured data directory."""
    global _store
    if _store is None:
        from ..config import get_config

        _store = ContractsStore(db_path=get_config().data_dir / "contracts.db")
    return _store


def reset_store() -> None:
    """Close and forget the contracts store (it is reopened from config on next use)."""
    global _store
    if _store is not None:
        _store.close()
        _store = None


# =============================================================================
# Authentication Helper
# =============================================================================


def get_current_user_id(request: Request, session_token: Optional[str] = None) -> str:
    """
    Get the current user ID from the session.

    Returns the authenticated user's ID.
    Raises HTTPException 401 if not authenticated.

    Note: This wraps require_authenticated_user for endpoints that call it
    directly instead of via Depends().
    """
    return require_authenticated_user(request, session_token)


# =============================================================================
# Endpoints
# =============================================================================


@router.get("/", response_model=List[ContractSummary])
async def list_contracts(
    request: Request,
    status: Optional[str] = None,
    session_token: Optional[str] = Cookie(None),
) -> List[ContractSummary]:
    """
    List all contracts for the current user.

    Optionally filter by status (ACTIVE, PENDING, EXPIRED, REVOKED).
    Each user only sees their own contracts.
    """
    user_id = get_current_user_id(request, session_token)
    store = get_store()
    try:
        contracts = store.get_contracts(user_id=user_id, status=status)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

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
async def get_contracts_stats(
    request: Request,
    session_token: Optional[str] = Cookie(None),
) -> ContractsStats:
    """Get contracts statistics for the current user."""
    user_id = get_current_user_id(request, session_token)
    store = get_store()
    return store.get_stats(user_id=user_id)


@router.get("/types", response_model=List[ContractTypeModel])
async def get_contract_types(
    user_id: str = Depends(require_authenticated_user),
) -> List[ContractTypeModel]:
    """Get available contract types with their permissions."""
    store = get_store()
    return store.get_contract_types()


@router.get("/templates", response_model=List[ContractTemplateModel])
async def get_templates(
    user_id: str = Depends(require_authenticated_user),
) -> List[ContractTemplateModel]:
    """Get all available contract templates."""
    store = get_store()
    return store.get_templates()


@router.get("/templates/{template_id}", response_model=ContractTemplateModel)
async def get_template_by_id(
    template_id: str, user_id: str = Depends(require_authenticated_user)
) -> ContractTemplateModel:
    """Get a specific template by ID."""
    store = get_store()
    template = store.get_template(template_id)

    if not template:
        raise HTTPException(status_code=404, detail=f"Template not found: {template_id}")

    return template


@router.get("/{contract_id}", response_model=ContractModel)
async def get_contract(
    contract_id: str,
    request: Request,
    session_token: Optional[str] = Cookie(None),
) -> ContractModel:
    """Get detailed information about a specific contract (must be owned by current user)."""
    user_id = get_current_user_id(request, session_token)
    store = get_store()
    # Contracts owned by other users are reported as not found.
    contract = store.get_contract(contract_id, user_id=user_id)

    if not contract:
        raise HTTPException(status_code=404, detail=f"Contract not found: {contract_id}")

    return contract


@router.post("/", response_model=ContractModel)
async def create_contract(
    request: Request,
    body: CreateContractRequest,
    session_token: Optional[str] = Cookie(None),
) -> ContractModel:
    """Create a new learning contract for the current user."""
    user_id = get_current_user_id(request, session_token)
    store = get_store()

    # The contract always belongs to the authenticated user, whatever the body says.
    body.user_id = user_id

    try:
        # Validates the contract type against the names listed by GET /types.
        contract = store.create_contract(user_id, body)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Log intent
    try:
        from ..intent_log import IntentType, log_user_intent

        log_user_intent(
            user_id=user_id,
            intent_type=IntentType.CONTRACT_CREATE,
            description=f"Created {body.contract_type} contract for domains: {body.domains}",
            details={
                "contract_id": contract.id,
                "contract_type": body.contract_type,
                "domains": body.domains,
            },
            related_entity_type="contract",
            related_entity_id=contract.id,
        )
    except Exception as e:
        logger.debug(f"Failed to log intent: {e}")

    return contract


@router.post("/from-template", response_model=ContractModel)
async def create_contract_from_template(
    request: Request,
    body: CreateFromTemplateRequest,
    session_token: Optional[str] = Cookie(None),
) -> ContractModel:
    """Create a contract from a template for the current user."""
    user_id = get_current_user_id(request, session_token)
    store = get_store()

    # The contract always belongs to the authenticated user, whatever the body says.
    body.user_id = user_id

    try:
        contract = store.create_from_template(user_id, body)
    except TemplateNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Log intent
    try:
        from ..intent_log import IntentType, log_user_intent

        log_user_intent(
            user_id=user_id,
            intent_type=IntentType.CONTRACT_CREATE,
            description=f"Created contract from template: {body.template_id}",
            details={"contract_id": contract.id, "template_id": body.template_id},
            related_entity_type="contract",
            related_entity_id=contract.id,
        )
    except Exception as e:
        logger.debug(f"Failed to log intent: {e}")

    return contract


@router.post("/{contract_id}/revoke")
async def revoke_contract(
    contract_id: str,
    request: Request,
    session_token: Optional[str] = Cookie(None),
) -> Dict[str, Any]:
    """Revoke an active contract (must be owned by current user)."""
    user_id = get_current_user_id(request, session_token)
    store = get_store()
    # Contracts owned by other users are reported as not found.
    contract = store.get_contract(contract_id, user_id=user_id)

    if not contract:
        raise HTTPException(status_code=404, detail=f"Contract not found: {contract_id}")

    if contract.status == "REVOKED":
        return {"status": "already_revoked", "contract_id": contract_id}

    if contract.status == "EXPIRED":
        # Expired contracts are terminal; there is nothing left to revoke.
        return {"status": "already_expired", "contract_id": contract_id}

    success = store.revoke_contract(
        contract_id, user_id=user_id, reason="Revoked by user via web interface"
    )

    if success:
        # Log intent
        try:
            from ..intent_log import IntentType, log_user_intent

            log_user_intent(
                user_id=user_id,
                intent_type=IntentType.CONTRACT_REVOKE,
                description=f"Revoked contract: {contract_id}",
                details={"contract_id": contract_id, "contract_type": contract.contract_type},
                related_entity_type="contract",
                related_entity_id=contract_id,
            )
        except Exception as e:
            logger.debug(f"Failed to log intent: {e}")

        return {"status": "revoked", "contract_id": contract_id}
    else:
        raise HTTPException(status_code=500, detail="Failed to revoke contract")


@router.delete("/{contract_id}")
async def delete_contract(
    contract_id: str,
    request: Request,
    session_token: Optional[str] = Cookie(None),
) -> Dict[str, Any]:
    """Delete a contract (same as revoke for safety, must be owned by current user)."""
    return await revoke_contract(contract_id, request, session_token)
