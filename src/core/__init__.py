"""
Agent OS Core - Constitutional Kernel and Core Components

This module contains the foundational components of Agent OS:
- Constitution parser and validator
- Rule extraction and precedence management
- Conflict detection
- Hot-reload capabilities
"""

from .constitution import ConstitutionalKernel
from .enforcement import EnforcementEngine, EnforcementDecision
from .models import (
    AuthorityLevel,
    ConflictType,
    Constitution,
    ConstitutionMetadata,
    Rule,
    RuleConflict,
    RuleType,
    ValidationResult,
)
from .parser import ConstitutionParser
from .validator import ConstitutionValidator

__all__ = [
    "ConstitutionMetadata",
    "Rule",
    "RuleType",
    "AuthorityLevel",
    "Constitution",
    "ValidationResult",
    "ConflictType",
    "RuleConflict",
    "ConstitutionParser",
    "ConstitutionalKernel",
    "ConstitutionValidator",
    "EnforcementEngine",
    "EnforcementDecision",
]
