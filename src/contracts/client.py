"""
Learning Contracts Client - Agent-OS Integration

Client for Agent-OS components to interact with the Learning Contracts system.
Provides a simple interface for checking learning permissions.
"""

import logging
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set

from .store import (
    ContractStore,
    LearningContract,
    ContractType,
    ContractScope,
    LearningScope,
    ContractStatus,
)
from .enforcement import (
    LearningContractsEngine,
    EnforcementResult,
    EnforcementDecision,
    EnforcementConfig,
    create_learning_contracts_engine,
)
from .consent import (
    ConsentPrompt,
    ConsentRequest,
    ConsentDecision,
    ConsentResponse,
    ConsentMode,
)
from .abstraction import (
    AbstractionLevel,
    AbstractionResult,
)


logger = logging.getLogger(__name__)


@dataclass
class ContractsClientConfig:
    """Configuration for the contracts client."""
    db_path: Optional[Path] = None
    default_deny: bool = True
    enable_abstraction: bool = True
    default_abstraction_level: AbstractionLevel = AbstractionLevel.MODERATE
    consent_mode: ConsentMode = ConsentMode.CALLBACK
    cache_decisions: bool = True
    cache_ttl_minutes: float = 5.0


class ContractsClient:
    """
    Client for Learning Contracts System.

    Provides a simple interface for Agent-OS components to:
    - Check if learning is allowed
    - Request consent for learning
    - Create and manage contracts
    - Abstract content when needed
    """

    def __init__(self, config: Optional[ContractsClientConfig] = None):
        """
        Initialize contracts client.

        Args:
            config: Client configuration
        """
        self.config = config or ContractsClientConfig()

        # Initialize engine
        engine_config = EnforcementConfig(
            default_deny=self.config.default_deny,
            enable_abstraction=self.config.enable_abstraction,
            default_abstraction_level=self.config.default_abstraction_level,
        )
        self._engine = LearningContractsEngine(
            config=engine_config,
            db_path=self.config.db_path,
        )

        # Initialize consent prompt
        self._consent_prompt = ConsentPrompt(
            mode=self.config.consent_mode,
            cache_decisions=self.config.cache_decisions,
        )

        # Decision cache
        self._decision_cache: Dict[str, tuple] = {}

        # Statistics
        self._request_count = 0
        self._allowed_count = 0
        self._denied_count = 0

    @property
    def engine(self) -> LearningContractsEngine:
        """Get the underlying engine."""
        return self._engine

    def can_learn(
        self,
        user_id: str,
        domain: str = "",
        task: str = "",
        agent: str = "",
    ) -> bool:
        """
        Quick check if learning is allowed.

        Args:
            user_id: User ID
            domain: Learning domain
            task: Task type
            agent: Agent involved

        Returns:
            True if learning is allowed
        """
        result = self.check_learning(
            user_id=user_id,
            content="",  # Empty content for quick check
            domain=domain,
            task=task,
            agent=agent,
        )
        return result.allowed

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
        Check if learning is allowed for content.

        This is the main method for learning checks.

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
        self._request_count += 1

        # Check cache
        cache_key = f"{user_id}:{domain}:{task}:{agent}"
        cached = self._check_cache(cache_key)
        if cached and not content:  # Only use cache for quick checks
            return cached

        result = self._engine.check_learning(
            user_id=user_id,
            content=content,
            domain=domain,
            task=task,
            agent=agent,
            content_type=content_type,
            requires_raw_storage=requires_raw_storage,
        )

        # Update stats
        if result.allowed:
            self._allowed_count += 1
        else:
            self._denied_count += 1

        # Cache result
        self._cache_result(cache_key, result)

        return result

    def request_consent(
        self,
        user_id: str,
        domain: str,
        description: str,
        data_types: Optional[List[str]] = None,
        purpose: str = "",
    ) -> ConsentDecision:
        """
        Request consent from user for learning.

        Args:
            user_id: User to request from
            domain: Domain for learning
            description: Description of what will be learned
            data_types: Types of data involved
            purpose: Purpose of learning

        Returns:
            User's consent decision
        """
        request = ConsentRequest(
            request_id=f"CR-{self._request_count}",
            user_id=user_id,
            domain=domain,
            description=description,
            data_types=data_types or [],
            purpose=purpose,
        )

        return self._consent_prompt.prompt(request)

    def create_contract_from_consent(
        self,
        decision: ConsentDecision,
    ) -> Optional[LearningContract]:
        """
        Create a contract from a consent decision.

        Args:
            decision: User's consent decision

        Returns:
            Created contract or None if denied
        """
        if decision.response in [ConsentResponse.DENY, ConsentResponse.DENY_ALWAYS]:
            return None

        scope = decision.scope_restrictions or ContractScope(
            scope_type=LearningScope.ALL,
        )

        contract_type = decision.contract_type or ContractType.FULL_CONSENT

        return self._engine.create_contract(
            user_id=decision.user_id,
            contract_type=contract_type,
            scope=scope,
            duration=decision.duration,
            description=decision.reason or "Created from consent",
            auto_activate=True,
        )

    def create_session_contract(
        self,
        user_id: str,
        domains: Optional[Set[str]] = None,
    ) -> LearningContract:
        """
        Create a session-only contract.

        Args:
            user_id: User ID
            domains: Optional specific domains

        Returns:
            Created contract
        """
        scope = ContractScope(
            scope_type=LearningScope.DOMAIN_SPECIFIC if domains else LearningScope.ALL,
            domains=domains or set(),
        )

        return self._engine.create_contract(
            user_id=user_id,
            contract_type=ContractType.SESSION_ONLY,
            scope=scope,
            duration=timedelta(hours=24),  # Session contracts expire after 24h
            description="Session-only learning consent",
            auto_activate=True,
        )

    def abstract_content(
        self,
        content: str,
        level: Optional[AbstractionLevel] = None,
    ) -> AbstractionResult:
        """
        Abstract content for safe learning.

        Args:
            content: Content to abstract
            level: Abstraction level

        Returns:
            Abstraction result with abstracted content
        """
        return self._engine.abstract_content(
            content=content,
            level=level or self.config.default_abstraction_level,
        )

    def revoke_contract(
        self,
        contract_id: str,
        user_id: str,
        reason: str = "",
    ) -> bool:
        """
        Revoke a specific contract.

        Args:
            contract_id: Contract to revoke
            user_id: User revoking
            reason: Reason for revocation

        Returns:
            True if revoked
        """
        return self._engine.store.revoke_contract(
            contract_id=contract_id,
            revoked_by=user_id,
            reason=reason,
        )

    def revoke_all(
        self,
        user_id: str,
        reason: str = "User requested full revocation",
    ) -> int:
        """
        Revoke all contracts for a user.

        Args:
            user_id: User ID
            reason: Reason for revocation

        Returns:
            Number of contracts revoked
        """
        return self._engine.revoke_all_contracts(user_id, reason)

    def get_active_contracts(self, user_id: str) -> List[LearningContract]:
        """Get active contracts for a user."""
        return self._engine.get_user_contracts(user_id, active_only=True)

    def has_valid_contract(
        self,
        user_id: str,
        domain: str = "",
    ) -> bool:
        """Check if user has a valid contract for domain."""
        contract = self._engine.store.get_valid_contract(
            user_id=user_id,
            domain=domain,
        )
        return contract is not None

    def set_consent_callback(
        self,
        callback: Callable[[ConsentRequest], ConsentDecision],
    ) -> None:
        """Set callback for consent prompts."""
        self._consent_prompt.set_callback(callback)

    def set_domain_preference(
        self,
        user_id: str,
        domain: str,
        response: ConsentResponse,
    ) -> None:
        """Set a permanent preference for a domain."""
        self._consent_prompt.set_preference(user_id, domain, response)

    def check_prohibited_domain(self, content: str) -> bool:
        """
        Check if content contains prohibited domains.

        Returns True if content is prohibited.
        """
        result = self._engine.check_domain(content)
        return result.is_prohibited

    def add_prohibited_domain(
        self,
        name: str,
        keywords: Set[str],
    ) -> None:
        """Add a custom prohibited domain."""
        self._engine.add_prohibited_domain(name, keywords)

    def get_statistics(self) -> Dict[str, Any]:
        """Get client statistics."""
        return {
            "request_count": self._request_count,
            "allowed_count": self._allowed_count,
            "denied_count": self._denied_count,
            "allowed_ratio": self._allowed_count / self._request_count if self._request_count > 0 else 0,
            "engine_stats": self._engine.get_statistics(),
        }

    def shutdown(self) -> None:
        """Shutdown the client."""
        self._engine.shutdown()

    def _check_cache(self, cache_key: str) -> Optional[EnforcementResult]:
        """Check decision cache."""
        cached = self._decision_cache.get(cache_key)
        if cached:
            result, timestamp = cached
            from datetime import datetime
            if (datetime.now() - timestamp).total_seconds() < self.config.cache_ttl_minutes * 60:
                return result
            del self._decision_cache[cache_key]
        return None

    def _cache_result(self, cache_key: str, result: EnforcementResult) -> None:
        """Cache a decision result."""
        if self.config.cache_decisions:
            from datetime import datetime
            self._decision_cache[cache_key] = (result, datetime.now())

            # Limit cache size
            if len(self._decision_cache) > 1000:
                # Remove oldest entries
                keys = list(self._decision_cache.keys())
                for key in keys[:100]:
                    del self._decision_cache[key]


def create_contracts_client(
    db_path: Optional[Path] = None,
    default_deny: bool = True,
    **kwargs,
) -> ContractsClient:
    """
    Factory function to create a contracts client.

    Args:
        db_path: Path to database
        default_deny: Default to denying learning
        **kwargs: Additional configuration

    Returns:
        Configured ContractsClient
    """
    config = ContractsClientConfig(
        db_path=db_path,
        default_deny=default_deny,
        **kwargs,
    )
    return ContractsClient(config)


# Convenience functions for quick access

_default_client: Optional[ContractsClient] = None


def get_default_client() -> ContractsClient:
    """Get or create the default client."""
    global _default_client
    if _default_client is None:
        _default_client = create_contracts_client()
    return _default_client


def can_learn(
    user_id: str,
    domain: str = "",
    task: str = "",
    agent: str = "",
) -> bool:
    """Quick check if learning is allowed using default client."""
    return get_default_client().can_learn(user_id, domain, task, agent)


def check_learning(
    user_id: str,
    content: str,
    domain: str = "",
    **kwargs,
) -> EnforcementResult:
    """Check learning permission using default client."""
    return get_default_client().check_learning(user_id, content, domain, **kwargs)
