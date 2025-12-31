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

# Abstraction
from .abstraction import (
    AbstractionGuard,
    AbstractionLevel,
    AbstractionPolicy,
    AbstractionResult,
    AbstractionRule,
    AbstractionType,
    create_abstraction_guard,
)

# Client
from .client import (
    ContractsClient,
    ContractsClientConfig,
    can_learn,
    check_learning,
    create_contracts_client,
    get_default_client,
)

# Consent
from .consent import (
    ConsentDecision,
    ConsentMode,
    ConsentPrompt,
    ConsentRequest,
    ConsentResponse,
    ConsentUI,
    create_consent_prompt,
)

# Prohibited Domains
from .domains import (
    DomainCategory,
    DomainCheckResult,
    ProhibitedDomain,
    ProhibitedDomainChecker,
    ProhibitionLevel,
    create_domain_checker,
)

# Enforcement
from .enforcement import (
    EnforcementConfig,
    EnforcementDecision,
    EnforcementResult,
    LearningContractsEngine,
    create_learning_contracts_engine,
)

# Store
from .store import (  # Templates (learning-contracts spec); Default Agent-OS contracts
    CONTRACT_TEMPLATES,
    ContractQuery,
    ContractScope,
    ContractStatus,
    ContractStore,
    ContractTemplate,
    ContractType,
    LearningContract,
    LearningScope,
    create_contract_from_template,
    create_contract_store,
    create_default_agent_os_contracts,
    ensure_default_contracts,
    get_template,
    list_templates,
)

# Validator
from .validator import (
    ContractValidator,
    LearningRequest,
    ValidationCode,
    ValidationIssue,
    ValidationResult,
    ValidationSeverity,
    create_validator,
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
    # Templates (learning-contracts spec)
    "ContractTemplate",
    "CONTRACT_TEMPLATES",
    "get_template",
    "list_templates",
    "create_contract_from_template",
    # Default Agent-OS contracts
    "create_default_agent_os_contracts",
    "ensure_default_contracts",
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
