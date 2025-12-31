"""
Learning Contracts Enforcement Engine

Central enforcement layer that coordinates contract validation,
domain checking, and abstraction requirements.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set

from .abstraction import (
    AbstractionGuard,
    AbstractionLevel,
    AbstractionResult,
    create_abstraction_guard,
)
from .domains import (
    DomainCheckResult,
    ProhibitedDomainChecker,
    ProhibitionLevel,
    create_domain_checker,
)
from .store import (
    ContractScope,
    ContractStatus,
    ContractStore,
    ContractType,
    LearningContract,
    LearningScope,
    create_contract_store,
)
from .validator import (
    ContractValidator,
    LearningRequest,
    ValidationCode,
    ValidationResult,
    create_validator,
)

logger = logging.getLogger(__name__)


class EnforcementDecision(Enum):
    """Result of enforcement decision."""

    ALLOW = auto()  # Learning allowed
    ALLOW_ABSTRACTED = auto()  # Learning allowed with abstraction
    DENY = auto()  # Learning denied
    DENY_DOMAIN = auto()  # Denied due to prohibited domain
    DENY_NO_CONTRACT = auto()  # Denied, no valid contract
    ESCALATE = auto()  # Requires human approval
    PROMPT = auto()  # Prompt user for consent


@dataclass
class EnforcementResult:
    """Result of an enforcement check."""

    decision: EnforcementDecision
    allowed: bool
    contract: Optional[LearningContract] = None
    requires_abstraction: bool = False
    abstraction_level: Optional[AbstractionLevel] = None
    abstracted_content: Optional[str] = None
    reason: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "decision": self.decision.name,
            "allowed": self.allowed,
            "contract_id": self.contract.contract_id if self.contract else None,
            "requires_abstraction": self.requires_abstraction,
            "abstraction_level": self.abstraction_level.name if self.abstraction_level else None,
            "reason": self.reason,
            "details": self.details,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class EnforcementConfig:
    """Configuration for the enforcement engine."""

    default_deny: bool = True
    require_explicit_consent: bool = True
    enable_abstraction: bool = True
    default_abstraction_level: AbstractionLevel = AbstractionLevel.MODERATE
    enable_domain_checking: bool = True
    strict_mode: bool = True  # Stricter validation
    auto_cleanup_expired: bool = True
    audit_all_decisions: bool = True


class LearningContractsEngine:
    """
    Main enforcement engine for learning contracts.

    Coordinates all components:
    - Contract store (persistence)
    - Contract validator (validation)
    - Domain checker (prohibited domains)
    - Abstraction guard (data abstraction)
    """

    def __init__(
        self,
        config: Optional[EnforcementConfig] = None,
        db_path: Optional[Path] = None,
    ):
        """
        Initialize the enforcement engine.

        Args:
            config: Engine configuration
            db_path: Path to contract database
        """
        self.config = config or EnforcementConfig()

        # Initialize components
        self._store = create_contract_store(
            db_path=db_path,
            default_deny=self.config.default_deny,
        )
        self._validator = create_validator(
            require_signature=self.config.strict_mode,
        )
        self._domain_checker = create_domain_checker()
        self._abstraction_guard = create_abstraction_guard()

        # Callbacks
        self._on_decision: Optional[Callable[[EnforcementResult], None]] = None
        self._consent_prompt: Optional[Callable[[str, Dict], bool]] = None

        # Statistics
        self._stats = {
            "total_checks": 0,
            "allowed": 0,
            "denied": 0,
            "abstracted": 0,
            "escalated": 0,
        }

    @property
    def store(self) -> ContractStore:
        """Get the contract store."""
        return self._store

    @property
    def validator(self) -> ContractValidator:
        """Get the contract validator."""
        return self._validator

    @property
    def domain_checker(self) -> ProhibitedDomainChecker:
        """Get the domain checker."""
        return self._domain_checker

    @property
    def abstraction_guard(self) -> AbstractionGuard:
        """Get the abstraction guard."""
        return self._abstraction_guard

    def check_learning(
        self,
        user_id: str,
        content: str,
        domain: str = "",
        task: str = "",
        agent: str = "",
        content_type: str = "",
        requires_raw_storage: bool = False,
    ) -> EnforcementResult:
        """
        Check if learning is allowed for the given content.

        This is the main entry point for learning checks.

        Args:
            user_id: User making the request
            content: Content to potentially learn
            domain: Learning domain
            task: Task type
            agent: Agent involved
            content_type: Type of content
            requires_raw_storage: If raw storage is needed

        Returns:
            EnforcementResult
        """
        self._stats["total_checks"] += 1

        # Create learning request
        request = LearningRequest(
            request_id=f"LR-{datetime.now().strftime('%Y%m%d%H%M%S')}",
            user_id=user_id,
            domain=domain,
            task=task,
            agent=agent,
            content_type=content_type,
            content=content,
            requires_raw_storage=requires_raw_storage,
        )

        # Step 1: Check prohibited domains
        if self.config.enable_domain_checking:
            domain_result = self._domain_checker.check(content)
            if domain_result.is_prohibited:
                result = EnforcementResult(
                    decision=EnforcementDecision.DENY_DOMAIN,
                    allowed=False,
                    reason="; ".join(domain_result.reasons),
                    details={
                        "prohibited_domains": [d.name for d in domain_result.matching_domains],
                        "can_override": domain_result.can_override,
                    },
                )
                self._record_decision(result)
                return result

        # Step 2: Check for valid contract
        contract = self._store.get_valid_contract(
            user_id=user_id,
            domain=domain,
            task=task,
            agent=agent,
        )

        if not contract:
            if self.config.default_deny:
                result = EnforcementResult(
                    decision=EnforcementDecision.DENY_NO_CONTRACT,
                    allowed=False,
                    reason="No valid learning contract found",
                )
            else:
                result = EnforcementResult(
                    decision=EnforcementDecision.PROMPT,
                    allowed=False,
                    reason="User consent required",
                )
            self._record_decision(result)
            return result

        # Step 3: Validate the contract and request
        validation = self._validator.validate_request(request, contract)
        if not validation.is_valid:
            result = EnforcementResult(
                decision=EnforcementDecision.DENY,
                allowed=False,
                contract=contract,
                reason=validation.issues[0].message if validation.issues else "Validation failed",
                details={"issues": [i.to_dict() for i in validation.issues]},
            )
            self._record_decision(result)
            return result

        # Step 4: Check abstraction requirements
        if contract.contract_type == ContractType.ABSTRACTION_ONLY:
            if self.config.enable_abstraction:
                abstraction = self._abstraction_guard.abstract(
                    content=content,
                    level=self.config.default_abstraction_level,
                    content_type=content_type,
                )
                result = EnforcementResult(
                    decision=EnforcementDecision.ALLOW_ABSTRACTED,
                    allowed=True,
                    contract=contract,
                    requires_abstraction=True,
                    abstraction_level=self.config.default_abstraction_level,
                    abstracted_content=abstraction.content,
                    reason="Learning allowed with abstraction",
                    details={"abstraction": abstraction.to_dict()},
                )
            else:
                result = EnforcementResult(
                    decision=EnforcementDecision.DENY,
                    allowed=False,
                    contract=contract,
                    reason="Contract requires abstraction but abstraction is disabled",
                )
            self._record_decision(result)
            return result

        # Step 5: Allow learning
        result = EnforcementResult(
            decision=EnforcementDecision.ALLOW,
            allowed=True,
            contract=contract,
            reason="Learning allowed",
        )
        self._record_decision(result)
        return result

    def create_contract(
        self,
        user_id: str,
        contract_type: ContractType,
        scope: ContractScope,
        duration: Optional[timedelta] = None,
        description: str = "",
        auto_activate: bool = False,
    ) -> LearningContract:
        """
        Create a new learning contract.

        Args:
            user_id: User creating contract
            contract_type: Type of contract
            scope: Scope of learning allowed
            duration: Optional duration
            description: Contract description
            auto_activate: Auto-activate contract

        Returns:
            Created LearningContract
        """
        return self._store.create_contract(
            user_id=user_id,
            contract_type=contract_type,
            scope=scope,
            duration=duration,
            description=description,
            auto_activate=auto_activate,
        )

    def create_default_deny_contract(
        self,
        user_id: str,
        duration: Optional[timedelta] = None,
    ) -> LearningContract:
        """
        Create a default no-learning contract.

        This explicitly denies all learning for a user.
        """
        scope = ContractScope(scope_type=LearningScope.ALL)
        return self.create_contract(
            user_id=user_id,
            contract_type=ContractType.NO_LEARNING,
            scope=scope,
            duration=duration,
            description="Default no-learning contract",
            auto_activate=True,
        )

    def create_full_consent_contract(
        self,
        user_id: str,
        duration: Optional[timedelta] = None,
        excluded_domains: Optional[Set[str]] = None,
    ) -> LearningContract:
        """
        Create a full consent contract.

        Allows all learning except for excluded domains.
        """
        scope = ContractScope(
            scope_type=LearningScope.ALL,
            excluded_domains=excluded_domains or set(),
        )
        return self.create_contract(
            user_id=user_id,
            contract_type=ContractType.FULL_CONSENT,
            scope=scope,
            duration=duration,
            description="Full learning consent",
            auto_activate=True,
        )

    def create_abstraction_only_contract(
        self,
        user_id: str,
        domains: Optional[Set[str]] = None,
        duration: Optional[timedelta] = None,
    ) -> LearningContract:
        """
        Create an abstraction-only contract.

        Only allows learning of abstracted/anonymized data.
        """
        scope = ContractScope(
            scope_type=LearningScope.DOMAIN_SPECIFIC if domains else LearningScope.ALL,
            domains=domains or set(),
        )
        return self.create_contract(
            user_id=user_id,
            contract_type=ContractType.ABSTRACTION_ONLY,
            scope=scope,
            duration=duration,
            description="Abstraction-only learning consent",
            auto_activate=True,
        )

    def revoke_all_contracts(
        self,
        user_id: str,
        reason: str = "User requested revocation",
    ) -> int:
        """
        Revoke all active contracts for a user.

        Returns count of revoked contracts.
        """
        contracts = self._store.get_active_contracts(user_id)
        count = 0
        for contract in contracts:
            if self._store.revoke_contract(contract.contract_id, user_id, reason):
                count += 1
        return count

    def get_user_contracts(
        self,
        user_id: str,
        active_only: bool = True,
    ) -> List[LearningContract]:
        """Get all contracts for a user."""
        if active_only:
            return self._store.get_active_contracts(user_id)
        from .store import ContractQuery

        return self._store.query_contracts(ContractQuery(user_id=user_id))

    def abstract_content(
        self,
        content: str,
        level: AbstractionLevel = AbstractionLevel.MODERATE,
    ) -> AbstractionResult:
        """Abstract content to specified level."""
        return self._abstraction_guard.abstract(content, level)

    def check_domain(self, content: str) -> DomainCheckResult:
        """Check content for prohibited domains."""
        return self._domain_checker.check(content)

    def add_prohibited_domain(
        self,
        name: str,
        keywords: Set[str],
        patterns: Optional[List[str]] = None,
    ) -> None:
        """Add a prohibited domain."""
        from .domains import DomainCategory, ProhibitedDomain, ProhibitionLevel

        domain = ProhibitedDomain(
            domain_id=f"custom_{name.lower().replace(' ', '_')}",
            name=name,
            category=DomainCategory.CUSTOM,
            level=ProhibitionLevel.DEFAULT,
            keywords=keywords,
            patterns=patterns or [],
        )
        self._domain_checker.add_domain(domain)

    def set_consent_prompt(
        self,
        callback: Callable[[str, Dict], bool],
    ) -> None:
        """Set callback for prompting user consent."""
        self._consent_prompt = callback

    def set_decision_callback(
        self,
        callback: Callable[[EnforcementResult], None],
    ) -> None:
        """Set callback for enforcement decisions."""
        self._on_decision = callback

    def get_statistics(self) -> Dict[str, Any]:
        """Get enforcement statistics."""
        stats = dict(self._stats)
        stats["store"] = self._store.get_statistics()
        return stats

    def cleanup(self) -> int:
        """Clean up expired contracts."""
        if self.config.auto_cleanup_expired:
            return self._store.cleanup_expired()
        return 0

    def shutdown(self) -> None:
        """Shutdown the engine."""
        self._store.close()

    def _record_decision(self, result: EnforcementResult) -> None:
        """Record a decision for statistics and callbacks."""
        if result.allowed:
            if result.requires_abstraction:
                self._stats["abstracted"] += 1
            else:
                self._stats["allowed"] += 1
        else:
            if result.decision == EnforcementDecision.ESCALATE:
                self._stats["escalated"] += 1
            else:
                self._stats["denied"] += 1

        if self._on_decision and self.config.audit_all_decisions:
            try:
                self._on_decision(result)
            except Exception as e:
                logger.error(f"Decision callback error: {e}")

    # =========================================================================
    # Four Enforcement Hooks (learning-contracts spec)
    # These are the mandatory check points for contract compliance.
    # =========================================================================

    def check_before_memory_creation(
        self,
        user_id: str,
        content: str,
        domain: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> EnforcementResult:
        """
        Hook 1: Check contract compliance before memory creation.

        This hook MUST be called before any memory/data storage operation.
        It validates that:
        - A valid contract exists for the user/domain
        - The contract allows storage (not OBSERVATION or PROHIBITED)
        - The domain is not prohibited

        Args:
            user_id: User creating the memory
            content: Content to be stored
            domain: Domain of the content
            metadata: Additional metadata

        Returns:
            EnforcementResult with decision
        """
        result = self.check_learning(
            user_id=user_id,
            content=content,
            domain=domain,
            requires_raw_storage=True,
        )

        # Additional check: contract must allow storage
        if result.allowed and result.contract:
            if not result.contract.allows_raw_storage():
                return EnforcementResult(
                    decision=EnforcementDecision.DENY,
                    allowed=False,
                    contract=result.contract,
                    reason="Contract does not allow memory storage (OBSERVATION type)",
                    details={"hook": "before_memory_creation"},
                )

        return result

    def check_before_abstraction(
        self,
        user_id: str,
        content: str,
        domain: str = "",
        target_level: Optional[AbstractionLevel] = None,
    ) -> EnforcementResult:
        """
        Hook 2: Check contract compliance before abstraction/generalization.

        This hook MUST be called before deriving patterns or generalizations.
        It validates that:
        - A valid contract exists
        - The contract allows generalization (PROCEDURAL, STRATEGIC, or higher)
        - For EPISODIC contracts, cross-context abstraction is blocked

        Args:
            user_id: User requesting abstraction
            content: Content to abstract
            domain: Domain of the content
            target_level: Desired abstraction level

        Returns:
            EnforcementResult with decision
        """
        result = self.check_learning(
            user_id=user_id,
            content=content,
            domain=domain,
        )

        if result.allowed and result.contract:
            if not result.contract.allows_generalization():
                return EnforcementResult(
                    decision=EnforcementDecision.DENY,
                    allowed=False,
                    contract=result.contract,
                    reason="Contract does not allow generalization/abstraction",
                    details={
                        "hook": "before_abstraction",
                        "contract_type": result.contract.contract_type.name,
                    },
                )

            # If abstraction is allowed, perform it
            if self.config.enable_abstraction and target_level:
                abstraction = self._abstraction_guard.abstract(
                    content=content,
                    level=target_level,
                )
                result.abstracted_content = abstraction.content
                result.abstraction_level = target_level
                result.requires_abstraction = True

        return result

    def check_before_recall(
        self,
        user_id: str,
        query: str,
        source_domain: str = "",
        target_context: str = "",
    ) -> EnforcementResult:
        """
        Hook 3: Check contract compliance before memory recall.

        This hook MUST be called before retrieving stored memories.
        It validates that:
        - A valid contract exists for the user
        - Cross-context recall is allowed if target differs from source
        - For EPISODIC contracts, only same-context recall is allowed

        Args:
            user_id: User requesting recall
            query: Recall query
            source_domain: Original domain of memories
            target_context: Context where recall is being used

        Returns:
            EnforcementResult with decision
        """
        contract = self._store.get_valid_contract(
            user_id=user_id,
            domain=source_domain,
        )

        if not contract:
            return EnforcementResult(
                decision=EnforcementDecision.DENY_NO_CONTRACT,
                allowed=False,
                reason="No valid contract for memory recall",
                details={"hook": "before_recall"},
            )

        # Check cross-context recall
        if source_domain and target_context and source_domain != target_context:
            if not contract.allows_cross_context():
                return EnforcementResult(
                    decision=EnforcementDecision.DENY,
                    allowed=False,
                    contract=contract,
                    reason="Contract does not allow cross-context recall",
                    details={
                        "hook": "before_recall",
                        "source_domain": source_domain,
                        "target_context": target_context,
                        "contract_type": contract.contract_type.name,
                    },
                )

        return EnforcementResult(
            decision=EnforcementDecision.ALLOW,
            allowed=True,
            contract=contract,
            reason="Recall allowed",
            details={"hook": "before_recall"},
        )

    def check_before_export(
        self,
        user_id: str,
        content: str,
        export_format: str = "",
        destination: str = "",
    ) -> EnforcementResult:
        """
        Hook 4: Check contract compliance before data export.

        This hook MUST be called before exporting any learned data.
        It validates that:
        - A valid contract exists
        - The contract allows export (explicit consent required)
        - Prohibited domains are never exported

        Args:
            user_id: User requesting export
            content: Content to export
            export_format: Format of export (json, csv, etc.)
            destination: Where data is being exported

        Returns:
            EnforcementResult with decision
        """
        # Always check domains first for export
        if self.config.enable_domain_checking:
            domain_result = self._domain_checker.check(content)
            if domain_result.is_prohibited:
                return EnforcementResult(
                    decision=EnforcementDecision.DENY_DOMAIN,
                    allowed=False,
                    reason="Cannot export content from prohibited domains",
                    details={
                        "hook": "before_export",
                        "prohibited_domains": [d.name for d in domain_result.matching_domains],
                    },
                )

        # Check for valid contract
        contract = self._store.get_valid_contract(user_id=user_id)

        if not contract:
            return EnforcementResult(
                decision=EnforcementDecision.DENY_NO_CONTRACT,
                allowed=False,
                reason="No valid contract for data export",
                details={"hook": "before_export"},
            )

        # OBSERVATION and PROHIBITED contracts never allow export
        if contract.contract_type in [ContractType.OBSERVATION, ContractType.PROHIBITED]:
            return EnforcementResult(
                decision=EnforcementDecision.DENY,
                allowed=False,
                contract=contract,
                reason="Contract type does not allow data export",
                details={
                    "hook": "before_export",
                    "contract_type": contract.contract_type.name,
                },
            )

        # Export requires explicit consent in metadata
        if contract.metadata.get("export_allowed") is not True:
            return EnforcementResult(
                decision=EnforcementDecision.ESCALATE,
                allowed=False,
                contract=contract,
                reason="Export requires explicit consent approval",
                details={
                    "hook": "before_export",
                    "export_format": export_format,
                    "destination": destination,
                },
            )

        return EnforcementResult(
            decision=EnforcementDecision.ALLOW,
            allowed=True,
            contract=contract,
            reason="Export allowed",
            details={"hook": "before_export"},
        )


def create_learning_contracts_engine(
    db_path: Optional[Path] = None,
    default_deny: bool = True,
    enable_abstraction: bool = True,
) -> LearningContractsEngine:
    """
    Factory function to create a learning contracts engine.

    Args:
        db_path: Path to database (None for in-memory)
        default_deny: Default to denying learning
        enable_abstraction: Enable abstraction support

    Returns:
        Configured LearningContractsEngine
    """
    config = EnforcementConfig(
        default_deny=default_deny,
        enable_abstraction=enable_abstraction,
    )
    return LearningContractsEngine(config=config, db_path=db_path)
