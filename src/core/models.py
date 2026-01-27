"""
Agent OS Core Models

Data models for constitutional documents, rules, and validation results.
"""

import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, List, Optional, Set


class AuthorityLevel(Enum):
    """
    Authority levels in the constitutional hierarchy.
    Higher numeric value = higher authority.
    """

    SUPREME = 100  # Core constitution (CONSTITUTION.md)
    SYSTEM = 80  # System-wide instructions
    AGENT_SPECIFIC = 60  # Agent-specific constitutions
    ROLE = 40  # Role instructions
    TASK = 20  # Task-level prompts

    def __lt__(self, other: "AuthorityLevel") -> bool:
        return self.value < other.value

    def __le__(self, other: "AuthorityLevel") -> bool:
        return self.value <= other.value

    def __gt__(self, other: "AuthorityLevel") -> bool:
        return self.value > other.value

    def __ge__(self, other: "AuthorityLevel") -> bool:
        return self.value >= other.value


class RuleType(Enum):
    """Types of constitutional rules."""

    PRINCIPLE = auto()  # Core principles (e.g., "Human Sovereignty")
    MANDATE = auto()  # Required actions ("SHALL", "MUST")
    PROHIBITION = auto()  # Forbidden actions ("MUST NOT", "SHALL NOT")
    PERMISSION = auto()  # Allowed actions ("MAY", "CAN")
    BOUNDARY = auto()  # Authority boundaries
    PROCEDURE = auto()  # Process requirements
    ESCALATION = auto()  # Escalation requirements
    IMMUTABLE = auto()  # Cannot be amended


class ConflictType(Enum):
    """Types of rule conflicts that can be detected."""

    CONTRADICTION = auto()  # Rules directly contradict each other
    AUTHORITY_OVERLAP = auto()  # Multiple agents claim same authority
    PRECEDENCE_AMBIGUITY = auto()  # Unclear which rule takes precedence
    SCOPE_CONFLICT = auto()  # Overlapping scopes with different rules
    CIRCULAR_DEPENDENCY = auto()  # Rules reference each other circularly


@dataclass
class ConstitutionMetadata:
    """
    Metadata extracted from constitutional document YAML frontmatter.
    """

    document_type: str
    version: str
    effective_date: str
    scope: str
    authority_level: AuthorityLevel
    amendment_process: str
    author: Optional[str] = None
    repository: Optional[str] = None
    license: Optional[str] = None
    file_path: Optional[Path] = None
    raw_frontmatter: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_frontmatter(
        cls, frontmatter: Dict[str, Any], file_path: Optional[Path] = None
    ) -> "ConstitutionMetadata":
        """Create metadata from parsed YAML frontmatter."""
        # Map authority level string to enum
        authority_str = frontmatter.get("authority_level", "agent_specific").lower()
        authority_map = {
            "supreme": AuthorityLevel.SUPREME,
            "system": AuthorityLevel.SYSTEM,
            "agent_specific": AuthorityLevel.AGENT_SPECIFIC,
            "role": AuthorityLevel.ROLE,
            "task": AuthorityLevel.TASK,
        }
        authority_level = authority_map.get(authority_str, AuthorityLevel.AGENT_SPECIFIC)

        return cls(
            document_type=frontmatter.get("document_type", "unknown"),
            version=frontmatter.get("version", "0.0.0"),
            effective_date=frontmatter.get("effective_date", ""),
            scope=frontmatter.get("scope", "unknown"),
            authority_level=authority_level,
            amendment_process=frontmatter.get("amendment_process", "pull_request"),
            author=frontmatter.get("author"),
            repository=frontmatter.get("repository"),
            license=frontmatter.get("license"),
            file_path=file_path,
            raw_frontmatter=frontmatter,
        )


@dataclass
class Rule:
    """
    A single constitutional rule extracted from a document.
    """

    id: str  # Unique identifier (hash of content)
    content: str  # The rule text
    rule_type: RuleType  # Type of rule
    section: str  # Section header where rule was found
    section_path: List[str]  # Full path of headers (e.g., ["Core Principles", "Human Sovereignty"])
    authority_level: AuthorityLevel  # Inherited from document
    scope: str  # Inherited from document or overridden
    source_file: Optional[Path] = None
    line_number: Optional[int] = None
    is_immutable: bool = False  # Whether rule is marked as immutable
    keywords: Set[str] = field(default_factory=set)  # Extracted keywords
    references: List[str] = field(default_factory=list)  # References to other rules/sections

    def __post_init__(self):
        if not self.id:
            self.id = self._generate_id()

    def _generate_id(self) -> str:
        """Generate unique ID from rule content and context."""
        content = f"{self.section_path}:{self.content}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def conflicts_with(self, other: "Rule") -> bool:
        """Check if this rule potentially conflicts with another rule."""
        # Same scope and overlapping keywords might indicate conflict
        if self.scope == other.scope:
            if self.keywords & other.keywords:  # Overlapping keywords
                # Mandate vs Prohibition on same topic = conflict
                if (
                    self.rule_type == RuleType.MANDATE and other.rule_type == RuleType.PROHIBITION
                ) or (
                    self.rule_type == RuleType.PROHIBITION and other.rule_type == RuleType.MANDATE
                ):
                    return True
        return False


@dataclass
class RuleConflict:
    """Represents a detected conflict between rules."""

    conflict_type: ConflictType
    rule_a: Rule
    rule_b: Rule
    description: str
    severity: str  # "error", "warning", "info"
    resolution_hint: Optional[str] = None

    def __str__(self) -> str:
        return f"[{self.severity.upper()}] {self.conflict_type.name}: {self.description}"


@dataclass
class ValidationResult:
    """Result of validating a constitution or request against rules."""

    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    conflicts: List[RuleConflict] = field(default_factory=list)
    applicable_rules: List[Rule] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)

    def add_error(self, message: str) -> None:
        self.errors.append(message)
        self.is_valid = False

    def add_warning(self, message: str) -> None:
        self.warnings.append(message)

    def add_conflict(self, conflict: RuleConflict) -> None:
        self.conflicts.append(conflict)
        if conflict.severity == "error":
            self.is_valid = False


@dataclass
class Constitution:
    """
    A complete parsed constitutional document with all extracted rules.
    """

    metadata: ConstitutionMetadata
    rules: List[Rule]
    sections: Dict[str, str]  # Section name -> content
    raw_content: str
    file_hash: str
    loaded_at: datetime = field(default_factory=datetime.now)

    @property
    def rule_count(self) -> int:
        return len(self.rules)

    @property
    def section_count(self) -> int:
        return len(self.sections)

    def get_rules_by_type(self, rule_type: RuleType) -> List[Rule]:
        """Get all rules of a specific type."""
        return [r for r in self.rules if r.rule_type == rule_type]

    def get_rules_by_section(self, section: str) -> List[Rule]:
        """Get all rules from a specific section."""
        return [r for r in self.rules if section.lower() in r.section.lower()]

    def get_immutable_rules(self) -> List[Rule]:
        """Get all immutable rules."""
        return [r for r in self.rules if r.is_immutable]

    def get_rules_for_scope(self, scope: str) -> List[Rule]:
        """Get rules applicable to a specific scope (agent)."""
        return [r for r in self.rules if r.scope == "all_agents" or r.scope == scope]


@dataclass
class ConstitutionRegistry:
    """
    Registry of all loaded constitutions with hierarchy management.
    """

    constitutions: Dict[str, Constitution] = field(default_factory=dict)
    _supreme_scope: Optional[str] = field(default=None, repr=False)

    def register(self, constitution: Constitution) -> None:
        """
        Register a constitution in the registry.

        Warns if overwriting an existing constitution.
        Raises ValueError if trying to register multiple supreme constitutions.
        """
        import logging
        logger = logging.getLogger(__name__)

        key = constitution.metadata.scope

        # Check for duplicate scope (warn but allow overwrite)
        if key in self.constitutions:
            existing = self.constitutions[key]
            logger.warning(
                f"Overwriting existing constitution for scope '{key}' "
                f"(version {existing.metadata.version} -> {constitution.metadata.version})"
            )

        # Enforce single supreme constitution
        if constitution.metadata.authority_level == AuthorityLevel.SUPREME:
            if self._supreme_scope is not None and self._supreme_scope != key:
                raise ValueError(
                    f"Cannot register multiple supreme constitutions. "
                    f"Existing supreme: '{self._supreme_scope}', "
                    f"attempted: '{key}'"
                )
            self._supreme_scope = key

        self.constitutions[key] = constitution

    def get(self, scope: str) -> Optional[Constitution]:
        """Get constitution for a specific scope."""
        return self.constitutions.get(scope)

    def get_supreme(self) -> Optional[Constitution]:
        """
        Get the supreme constitution (highest authority).

        Returns the single supreme constitution, or None if not registered.
        """
        if self._supreme_scope:
            return self.constitutions.get(self._supreme_scope)

        # Fallback: search for supreme (for backwards compatibility)
        for const in self.constitutions.values():
            if const.metadata.authority_level == AuthorityLevel.SUPREME:
                self._supreme_scope = const.metadata.scope
                return const
        return None

    def get_all_rules(self) -> List[Rule]:
        """Get all rules from all constitutions, sorted by authority."""
        all_rules = []
        for const in self.constitutions.values():
            all_rules.extend(const.rules)
        return sorted(all_rules, key=lambda r: r.authority_level.value, reverse=True)

    def get_rules_for_agent(self, agent_scope: str) -> List[Rule]:
        """
        Get all applicable rules for an agent, including inherited rules.
        Returns rules sorted by authority level (highest first).
        """
        rules = []

        # Get supreme constitution rules (always apply)
        supreme = self.get_supreme()
        if supreme:
            rules.extend(supreme.rules)

        # Get agent-specific constitution rules
        agent_const = self.get(agent_scope)
        if agent_const and agent_const != supreme:
            rules.extend(agent_const.rules)

        # Sort by authority level (highest first)
        return sorted(rules, key=lambda r: r.authority_level.value, reverse=True)
