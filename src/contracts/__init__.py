"""
Agent OS Learning Contracts Module

Consent engine preventing AI learning without explicit authorization.
Provides comprehensive contract management for user consent.

Components:
- ContractStore: Persistent storage for learning contracts
- ContractValidator: Validates contracts and learning requests
- ProhibitedDomainChecker: Enforces domain-based restrictions
- AbstractionGuard: Ensures proper data abstraction
- LearningContractsEngine: Main enforcement engine
- ContractsClient: Client for Agent-OS integration

Usage:
    from src.contracts import ContractsClient, create_contracts_client

    # Create client
    client = create_contracts_client()

    # Check if learning is allowed
    if client.can_learn(user_id="user123", domain="general"):
        # Learning is allowed
        result = client.check_learning(
            user_id="user123",
            content="Some content to learn",
            domain="general",
        )
        if result.allowed:
            # Proceed with learning
            pass
        elif result.requires_abstraction:
            # Use abstracted content
            abstracted = result.abstracted_content
"""

# Store
from .store import (
    ContractStore,
    LearningContract,
    ContractType,
    ContractStatus,
    ContractScope,
    LearningScope,
    ContractQuery,
    create_contract_store,
)

# Validator
from .validator import (
    ContractValidator,
    ValidationResult,
    ValidationCode,
    ValidationSeverity,
    ValidationIssue,
    LearningRequest,
    create_validator,
)

# Prohibited Domains
from .domains import (
    ProhibitedDomainChecker,
    ProhibitedDomain,
    DomainCheckResult,
    DomainCategory,
    ProhibitionLevel,
    create_domain_checker,
)

# Abstraction
from .abstraction import (
    AbstractionGuard,
    AbstractionRule,
    AbstractionResult,
    AbstractionLevel,
    AbstractionType,
    AbstractionPolicy,
    create_abstraction_guard,
)

# Enforcement
from .enforcement import (
    LearningContractsEngine,
    EnforcementResult,
    EnforcementDecision,
    EnforcementConfig,
    create_learning_contracts_engine,
)

# Consent
from .consent import (
    ConsentPrompt,
    ConsentRequest,
    ConsentDecision,
    ConsentResponse,
    ConsentMode,
    ConsentUI,
    create_consent_prompt,
)

# Client
from .client import (
    ContractsClient,
    ContractsClientConfig,
    create_contracts_client,
    get_default_client,
    can_learn,
    check_learning,
)


__all__ = [
    # Store
    "ContractStore",
    "LearningContract",
    "ContractType",
    "ContractStatus",
    "ContractScope",
    "LearningScope",
    "ContractQuery",
    "create_contract_store",
    # Validator
    "ContractValidator",
    "ValidationResult",
    "ValidationCode",
    "ValidationSeverity",
    "ValidationIssue",
    "LearningRequest",
    "create_validator",
    # Domains
    "ProhibitedDomainChecker",
    "ProhibitedDomain",
    "DomainCheckResult",
    "DomainCategory",
    "ProhibitionLevel",
    "create_domain_checker",
    # Abstraction
    "AbstractionGuard",
    "AbstractionRule",
    "AbstractionResult",
    "AbstractionLevel",
    "AbstractionType",
    "AbstractionPolicy",
    "create_abstraction_guard",
    # Enforcement
    "LearningContractsEngine",
    "EnforcementResult",
    "EnforcementDecision",
    "EnforcementConfig",
    "create_learning_contracts_engine",
    # Consent
    "ConsentPrompt",
    "ConsentRequest",
    "ConsentDecision",
    "ConsentResponse",
    "ConsentMode",
    "ConsentUI",
    "create_consent_prompt",
    # Client
    "ContractsClient",
    "ContractsClientConfig",
    "create_contracts_client",
    "get_default_client",
    "can_learn",
    "check_learning",
]
