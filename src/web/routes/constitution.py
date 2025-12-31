"""
Constitutional Editor API Routes

Provides endpoints for viewing and editing the constitution.
"""

import logging
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)
router = APIRouter()

# Try to import real constitutional kernel
try:
    from src.core.constitution import ConstitutionalKernel, create_kernel
    from src.core.models import Rule as CoreRule
    from src.core.models import RuleType as CoreRuleType

    REAL_CONSTITUTION_AVAILABLE = True
except ImportError:
    REAL_CONSTITUTION_AVAILABLE = False
    logger.warning("Real constitutional kernel not available, using mock data")


# =============================================================================
# Models
# =============================================================================


class RuleType(str, Enum):
    """Type of constitutional rule."""

    PROHIBITION = "prohibition"
    PERMISSION = "permission"
    MANDATE = "mandate"
    ESCALATION = "escalation"


class RuleAuthority(str, Enum):
    """Authority level of a rule."""

    SUPREME = "supreme"
    CONSTITUTIONAL = "constitutional"
    STATUTORY = "statutory"
    AGENT = "agent"


class Rule(BaseModel):
    """A constitutional rule."""

    id: str
    content: str
    rule_type: RuleType
    authority: RuleAuthority = RuleAuthority.STATUTORY
    keywords: List[str] = Field(default_factory=list)
    agent_scope: Optional[str] = None
    is_immutable: bool = False
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RuleCreate(BaseModel):
    """Request to create a new rule."""

    content: str
    rule_type: RuleType
    authority: RuleAuthority = RuleAuthority.STATUTORY
    keywords: List[str] = Field(default_factory=list)
    agent_scope: Optional[str] = None


class RuleUpdate(BaseModel):
    """Request to update a rule."""

    content: Optional[str] = None
    rule_type: Optional[RuleType] = None
    authority: Optional[RuleAuthority] = None
    keywords: Optional[List[str]] = None


class ConstitutionSection(BaseModel):
    """A section of the constitution."""

    id: str
    title: str
    description: str = ""
    rules: List[Rule] = Field(default_factory=list)
    order: int = 0


class ConstitutionOverview(BaseModel):
    """Overview of the entire constitution."""

    total_rules: int
    sections: List[str]
    rule_counts_by_type: Dict[str, int]
    rule_counts_by_authority: Dict[str, int]
    last_updated: datetime


class ValidationRequest(BaseModel):
    """Request to validate content against constitution."""

    content: str
    context: Optional[Dict[str, Any]] = None


class ValidationResult(BaseModel):
    """Result of constitutional validation."""

    is_allowed: bool
    matching_rules: List[Rule] = Field(default_factory=list)
    violations: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    reasoning: str = ""


# =============================================================================
# Constitution Store (Real + Mock)
# =============================================================================


class ConstitutionStore:
    """
    Constitution store that integrates with the real ConstitutionalKernel.

    Falls back to mock data if real kernel isn't available.
    """

    def __init__(self):
        self._rules: Dict[str, Rule] = {}
        self._sections: Dict[str, ConstitutionSection] = {}
        self._kernel: Optional[Any] = None
        self._use_real_constitution = False

        # Try to initialize real constitutional kernel
        if REAL_CONSTITUTION_AVAILABLE:
            try:
                self._kernel = create_kernel(project_root=Path.cwd(), enable_hot_reload=True)
                result = self._kernel.initialize()
                if result.is_valid:
                    self._use_real_constitution = True
                    self._load_real_rules()
                    logger.info("Connected to real constitutional kernel")
                else:
                    logger.warning(f"Constitutional kernel validation failed: {result.errors}")
                    self._initialize_default_rules()
            except Exception as e:
                logger.warning(f"Failed to initialize constitutional kernel: {e}")
                self._initialize_default_rules()
        else:
            self._initialize_default_rules()

    def _load_real_rules(self):
        """Load rules from the real constitutional kernel."""
        try:
            # Get all rules from the kernel's registry
            all_rules = self._kernel._registry.get_all_rules()

            # Group rules by section
            section_rules: Dict[str, List[Rule]] = {}

            for core_rule in all_rules:
                rule = self._convert_core_rule(core_rule)
                self._rules[rule.id] = rule

                # Determine section from section path
                section_id = (
                    core_rule.section.lower().replace(" ", "_") if core_rule.section else "general"
                )
                if section_id not in section_rules:
                    section_rules[section_id] = []
                section_rules[section_id].append(rule)

            # Create sections
            order = 0
            for section_id, rules in section_rules.items():
                self._sections[section_id] = ConstitutionSection(
                    id=section_id,
                    title=section_id.replace("_", " ").title(),
                    description=f"Rules for {section_id.replace('_', ' ')}",
                    rules=rules,
                    order=order,
                )
                order += 1

            logger.info(f"Loaded {len(self._rules)} rules from constitutional kernel")

        except Exception as e:
            logger.error(f"Error loading real rules: {e}")
            self._initialize_default_rules()

    def _convert_core_rule(self, core_rule) -> Rule:
        """Convert a core Rule to API Rule model."""
        # Map core rule types to API rule types
        type_map = {
            "PRINCIPLE": RuleType.MANDATE,
            "MANDATE": RuleType.MANDATE,
            "PROHIBITION": RuleType.PROHIBITION,
            "PERMISSION": RuleType.PERMISSION,
            "ESCALATION": RuleType.ESCALATION,
            "DIRECTIVE": RuleType.MANDATE,
        }

        # Map authority levels
        authority_map = {
            100: RuleAuthority.SUPREME,
            80: RuleAuthority.CONSTITUTIONAL,
            60: RuleAuthority.STATUTORY,
            40: RuleAuthority.AGENT,
        }

        rule_type_name = (
            core_rule.rule_type.name
            if hasattr(core_rule.rule_type, "name")
            else str(core_rule.rule_type)
        )
        rule_type = type_map.get(rule_type_name, RuleType.MANDATE)

        authority_level = (
            core_rule.authority_level.value if hasattr(core_rule.authority_level, "value") else 60
        )
        authority = authority_map.get(authority_level, RuleAuthority.STATUTORY)

        return Rule(
            id=core_rule.id,
            content=core_rule.content,
            rule_type=rule_type,
            authority=authority,
            keywords=list(core_rule.keywords) if core_rule.keywords else [],
            agent_scope=core_rule.scope if core_rule.scope != "all_agents" else None,
            is_immutable=core_rule.is_immutable,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            metadata={
                "section": core_rule.section,
                "section_path": core_rule.section_path,
                "source_file": str(core_rule.source_file) if core_rule.source_file else None,
                "line_number": core_rule.line_number,
            },
        )

    def _initialize_default_rules(self):
        """Initialize with default constitutional rules."""
        # Core safety rules
        self._add_rule(
            Rule(
                id="supreme-001",
                content="Never generate content that could cause direct physical harm to humans",
                rule_type=RuleType.PROHIBITION,
                authority=RuleAuthority.SUPREME,
                keywords=["harm", "violence", "weapons", "dangerous"],
                is_immutable=True,
            ),
            section="core_safety",
        )

        self._add_rule(
            Rule(
                id="supreme-002",
                content="Always verify user consent before storing personal information",
                rule_type=RuleType.MANDATE,
                authority=RuleAuthority.SUPREME,
                keywords=["privacy", "personal", "consent", "data"],
                is_immutable=True,
            ),
            section="core_safety",
        )

        self._add_rule(
            Rule(
                id="supreme-003",
                content="Owner commands take precedence over all other directives",
                rule_type=RuleType.MANDATE,
                authority=RuleAuthority.SUPREME,
                keywords=["owner", "authority", "override"],
                is_immutable=True,
            ),
            section="core_safety",
        )

        # Operational rules
        self._add_rule(
            Rule(
                id="const-001",
                content="Requests involving financial transactions require explicit confirmation",
                rule_type=RuleType.ESCALATION,
                authority=RuleAuthority.CONSTITUTIONAL,
                keywords=["money", "payment", "transfer", "financial"],
            ),
            section="operations",
        )

        self._add_rule(
            Rule(
                id="const-002",
                content="System configuration changes must be logged and reversible",
                rule_type=RuleType.MANDATE,
                authority=RuleAuthority.CONSTITUTIONAL,
                keywords=["config", "settings", "system"],
            ),
            section="operations",
        )

        # Agent-specific rules
        self._add_rule(
            Rule(
                id="stat-001",
                content="Creative content may include fictional scenarios",
                rule_type=RuleType.PERMISSION,
                authority=RuleAuthority.STATUTORY,
                keywords=["creative", "fiction", "story"],
                agent_scope="muse",
            ),
            section="agent_rules",
        )

        self._add_rule(
            Rule(
                id="stat-002",
                content="Memory storage is permitted with user consent",
                rule_type=RuleType.PERMISSION,
                authority=RuleAuthority.STATUTORY,
                keywords=["memory", "store", "remember"],
            ),
            section="agent_rules",
        )

        # Create sections
        self._sections = {
            "core_safety": ConstitutionSection(
                id="core_safety",
                title="Core Safety Principles",
                description="Fundamental safety rules that cannot be overridden",
                rules=[r for r in self._rules.values() if r.id.startswith("supreme")],
                order=0,
            ),
            "operations": ConstitutionSection(
                id="operations",
                title="Operational Guidelines",
                description="Rules governing system operations",
                rules=[r for r in self._rules.values() if r.id.startswith("const")],
                order=1,
            ),
            "agent_rules": ConstitutionSection(
                id="agent_rules",
                title="Agent-Specific Rules",
                description="Rules for specific agents or use cases",
                rules=[r for r in self._rules.values() if r.id.startswith("stat")],
                order=2,
            ),
        }

    def _add_rule(self, rule: Rule, section: str) -> None:
        """Add a rule to the store."""
        self._rules[rule.id] = rule

    def get_all_rules(self) -> List[Rule]:
        """Get all rules."""
        return list(self._rules.values())

    def get_rule(self, rule_id: str) -> Optional[Rule]:
        """Get a rule by ID."""
        return self._rules.get(rule_id)

    def create_rule(self, request: RuleCreate) -> Rule:
        """Create a new rule."""
        import uuid

        rule_id = f"user-{uuid.uuid4().hex[:8]}"
        rule = Rule(
            id=rule_id,
            content=request.content,
            rule_type=request.rule_type,
            authority=request.authority,
            keywords=request.keywords,
            agent_scope=request.agent_scope,
        )
        self._rules[rule_id] = rule
        return rule

    def update_rule(self, rule_id: str, request: RuleUpdate) -> Optional[Rule]:
        """Update an existing rule."""
        if rule_id not in self._rules:
            return None

        rule = self._rules[rule_id]

        if rule.is_immutable:
            raise ValueError("Cannot modify immutable rule")

        if request.content is not None:
            rule.content = request.content
        if request.rule_type is not None:
            rule.rule_type = request.rule_type
        if request.authority is not None:
            rule.authority = request.authority
        if request.keywords is not None:
            rule.keywords = request.keywords

        rule.updated_at = datetime.utcnow()
        return rule

    def delete_rule(self, rule_id: str) -> bool:
        """Delete a rule."""
        if rule_id not in self._rules:
            return False

        rule = self._rules[rule_id]
        if rule.is_immutable:
            raise ValueError("Cannot delete immutable rule")

        del self._rules[rule_id]
        return True

    def get_sections(self) -> List[ConstitutionSection]:
        """Get all sections."""
        sections = list(self._sections.values())
        sections.sort(key=lambda s: s.order)
        return sections

    def validate_content(self, content: str) -> ValidationResult:
        """Validate content against the constitution."""
        content_lower = content.lower()
        matching_rules = []
        violations = []
        warnings = []

        for rule in self._rules.values():
            # Check if any keywords match
            matched = any(kw in content_lower for kw in rule.keywords)

            if matched:
                matching_rules.append(rule)

                if rule.rule_type == RuleType.PROHIBITION:
                    if rule.authority in (RuleAuthority.SUPREME, RuleAuthority.CONSTITUTIONAL):
                        violations.append(f"Violates rule {rule.id}: {rule.content}")
                    else:
                        warnings.append(f"May conflict with rule {rule.id}: {rule.content}")

                elif rule.rule_type == RuleType.ESCALATION:
                    warnings.append(f"Requires escalation per rule {rule.id}: {rule.content}")

        is_allowed = len(violations) == 0

        reasoning = (
            "Content is allowed." if is_allowed else "Content violates constitutional rules."
        )
        if warnings:
            reasoning += f" {len(warnings)} warning(s) noted."

        return ValidationResult(
            is_allowed=is_allowed,
            matching_rules=matching_rules,
            violations=violations,
            warnings=warnings,
            reasoning=reasoning,
        )


# Global store instance
_store: Optional[ConstitutionStore] = None


def get_store() -> ConstitutionStore:
    """Get the constitution store."""
    global _store
    if _store is None:
        _store = ConstitutionStore()
    return _store


# =============================================================================
# Endpoints
# =============================================================================


@router.get("/overview", response_model=ConstitutionOverview)
async def get_overview() -> ConstitutionOverview:
    """Get an overview of the constitution."""
    store = get_store()
    rules = store.get_all_rules()
    sections = store.get_sections()

    type_counts = {}
    authority_counts = {}

    for rule in rules:
        type_key = rule.rule_type.value
        auth_key = rule.authority.value
        type_counts[type_key] = type_counts.get(type_key, 0) + 1
        authority_counts[auth_key] = authority_counts.get(auth_key, 0) + 1

    return ConstitutionOverview(
        total_rules=len(rules),
        sections=[s.title for s in sections],
        rule_counts_by_type=type_counts,
        rule_counts_by_authority=authority_counts,
        last_updated=max(r.updated_at for r in rules) if rules else datetime.utcnow(),
    )


@router.get("/sections", response_model=List[ConstitutionSection])
async def list_sections() -> List[ConstitutionSection]:
    """List all constitution sections."""
    store = get_store()
    return store.get_sections()


@router.get("/rules", response_model=List[Rule])
async def list_rules(
    rule_type: Optional[RuleType] = None,
    authority: Optional[RuleAuthority] = None,
    agent_scope: Optional[str] = None,
) -> List[Rule]:
    """List all constitutional rules with optional filtering."""
    store = get_store()
    rules = store.get_all_rules()

    if rule_type:
        rules = [r for r in rules if r.rule_type == rule_type]

    if authority:
        rules = [r for r in rules if r.authority == authority]

    if agent_scope:
        rules = [r for r in rules if r.agent_scope == agent_scope]

    return rules


@router.get("/rules/{rule_id}", response_model=Rule)
async def get_rule(rule_id: str) -> Rule:
    """Get a specific rule by ID."""
    store = get_store()
    rule = store.get_rule(rule_id)

    if not rule:
        raise HTTPException(status_code=404, detail=f"Rule not found: {rule_id}")

    return rule


@router.post("/rules", response_model=Rule)
async def create_rule(request: RuleCreate) -> Rule:
    """
    Create a new constitutional rule.

    Note: User-created rules have authority level 'statutory' or lower.
    Supreme and constitutional rules can only be created through
    the ceremony process.
    """
    if request.authority in (RuleAuthority.SUPREME, RuleAuthority.CONSTITUTIONAL):
        raise HTTPException(
            status_code=403,
            detail="Cannot create rules with supreme or constitutional authority via API",
        )

    store = get_store()
    return store.create_rule(request)


@router.put("/rules/{rule_id}", response_model=Rule)
async def update_rule(rule_id: str, request: RuleUpdate) -> Rule:
    """Update an existing rule."""
    store = get_store()

    try:
        rule = store.update_rule(rule_id, request)
        if not rule:
            raise HTTPException(status_code=404, detail=f"Rule not found: {rule_id}")
        return rule
    except ValueError as e:
        raise HTTPException(status_code=403, detail=str(e))


@router.delete("/rules/{rule_id}")
async def delete_rule(rule_id: str) -> Dict[str, str]:
    """Delete a rule."""
    store = get_store()

    try:
        if store.delete_rule(rule_id):
            return {"status": "deleted", "rule_id": rule_id}
        raise HTTPException(status_code=404, detail=f"Rule not found: {rule_id}")
    except ValueError as e:
        raise HTTPException(status_code=403, detail=str(e))


@router.post("/validate", response_model=ValidationResult)
async def validate_content(request: ValidationRequest) -> ValidationResult:
    """
    Validate content against the constitution.

    This is a read-only check that doesn't modify anything.
    """
    store = get_store()
    return store.validate_content(request.content)


@router.get("/search")
async def search_rules(
    query: str,
    limit: int = 10,
) -> List[Rule]:
    """Search for rules matching a query."""
    store = get_store()
    rules = store.get_all_rules()

    query_lower = query.lower()

    # Score rules by relevance
    scored_rules = []
    for rule in rules:
        score = 0

        # Check content match
        if query_lower in rule.content.lower():
            score += 10

        # Check keyword match
        for keyword in rule.keywords:
            if query_lower in keyword or keyword in query_lower:
                score += 5

        if score > 0:
            scored_rules.append((score, rule))

    # Sort by score descending
    scored_rules.sort(key=lambda x: x[0], reverse=True)

    return [rule for _, rule in scored_rules[:limit]]
