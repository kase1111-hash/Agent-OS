"""
Consent Prompt Interface

User interface components for obtaining learning consent.
Supports multiple modalities: CLI, callback, and programmatic.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set

from .store import (
    ContractScope,
    ContractType,
    LearningContract,
    LearningScope,
)

logger = logging.getLogger(__name__)


class ConsentMode(Enum):
    """Mode for obtaining consent."""

    CALLBACK = auto()  # Use callback function
    CLI = auto()  # Command-line interface
    AUTO_DENY = auto()  # Automatically deny
    AUTO_ALLOW = auto()  # Automatically allow (dangerous!)
    CACHED = auto()  # Use cached decision


class ConsentResponse(Enum):
    """User's response to consent prompt."""

    ALLOW = auto()
    ALLOW_ABSTRACTED = auto()
    ALLOW_SESSION = auto()
    DENY = auto()
    DENY_ALWAYS = auto()
    ASK_LATER = auto()
    NEED_INFO = auto()


@dataclass
class ConsentRequest:
    """A request for user consent."""

    request_id: str
    user_id: str
    domain: str
    description: str
    data_types: List[str] = field(default_factory=list)
    purpose: str = ""
    retention_period: Optional[str] = None
    sharing_info: str = ""
    abstraction_available: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ConsentDecision:
    """User's decision on a consent request."""

    request_id: str
    response: ConsentResponse
    user_id: str
    contract_type: Optional[ContractType] = None
    duration: Optional[timedelta] = None
    scope_restrictions: Optional[ContractScope] = None
    reason: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "request_id": self.request_id,
            "response": self.response.name,
            "user_id": self.user_id,
            "contract_type": self.contract_type.name if self.contract_type else None,
            "duration_seconds": self.duration.total_seconds() if self.duration else None,
            "reason": self.reason,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }


class ConsentPrompt:
    """
    Manages consent prompts and user decisions.

    Provides interface for:
    - Displaying consent requests to users
    - Collecting user decisions
    - Managing consent preferences
    """

    def __init__(
        self,
        mode: ConsentMode = ConsentMode.CALLBACK,
        callback: Optional[Callable[[ConsentRequest], ConsentDecision]] = None,
        cache_decisions: bool = True,
        cache_ttl_hours: int = 24,
    ):
        """
        Initialize consent prompt.

        Args:
            mode: Consent collection mode
            callback: Callback for CALLBACK mode
            cache_decisions: Whether to cache decisions
            cache_ttl_hours: How long to cache decisions
        """
        self.mode = mode
        self._callback = callback
        self.cache_decisions = cache_decisions
        self.cache_ttl = timedelta(hours=cache_ttl_hours)

        self._cache: Dict[str, tuple[ConsentDecision, datetime]] = {}
        self._preferences: Dict[str, Dict[str, ConsentResponse]] = {}  # user -> domain -> response

    def prompt(self, request: ConsentRequest) -> ConsentDecision:
        """
        Prompt user for consent.

        Args:
            request: Consent request

        Returns:
            User's consent decision
        """
        # Check cache
        if self.cache_decisions:
            cached = self._check_cache(request)
            if cached:
                return cached

        # Check preferences
        preference = self._check_preference(request)
        if preference:
            return preference

        # Get decision based on mode
        if self.mode == ConsentMode.CALLBACK:
            decision = self._prompt_callback(request)
        elif self.mode == ConsentMode.CLI:
            decision = self._prompt_cli(request)
        elif self.mode == ConsentMode.AUTO_DENY:
            decision = self._auto_deny(request)
        elif self.mode == ConsentMode.AUTO_ALLOW:
            decision = self._auto_allow(request)
        else:
            decision = self._auto_deny(request)

        # Cache decision
        if self.cache_decisions:
            self._cache_decision(request, decision)

        return decision

    def set_preference(
        self,
        user_id: str,
        domain: str,
        response: ConsentResponse,
    ) -> None:
        """Set a permanent preference for a domain."""
        if user_id not in self._preferences:
            self._preferences[user_id] = {}
        self._preferences[user_id][domain] = response

    def clear_preference(self, user_id: str, domain: str) -> bool:
        """Clear a preference."""
        if user_id in self._preferences and domain in self._preferences[user_id]:
            del self._preferences[user_id][domain]
            return True
        return False

    def clear_all_preferences(self, user_id: str) -> None:
        """Clear all preferences for a user."""
        self._preferences.pop(user_id, None)

    def get_preferences(self, user_id: str) -> Dict[str, ConsentResponse]:
        """Get all preferences for a user."""
        return dict(self._preferences.get(user_id, {}))

    def clear_cache(self) -> None:
        """Clear the decision cache."""
        self._cache.clear()

    def set_callback(
        self,
        callback: Callable[[ConsentRequest], ConsentDecision],
    ) -> None:
        """Set the consent callback."""
        self._callback = callback
        self.mode = ConsentMode.CALLBACK

    def create_decision(
        self,
        request: ConsentRequest,
        response: ConsentResponse,
        duration_days: Optional[int] = None,
        scope_domains: Optional[Set[str]] = None,
    ) -> ConsentDecision:
        """
        Helper to create a consent decision.

        Args:
            request: Original request
            response: User's response
            duration_days: Duration of consent
            scope_domains: Specific domains for consent

        Returns:
            ConsentDecision
        """
        # Map response to contract type
        contract_type = None
        if response == ConsentResponse.ALLOW:
            contract_type = ContractType.FULL_CONSENT
        elif response == ConsentResponse.ALLOW_ABSTRACTED:
            contract_type = ContractType.ABSTRACTION_ONLY
        elif response == ConsentResponse.ALLOW_SESSION:
            contract_type = ContractType.SESSION_ONLY
        elif response in [ConsentResponse.DENY, ConsentResponse.DENY_ALWAYS]:
            contract_type = ContractType.NO_LEARNING

        # Create scope
        scope = None
        if scope_domains:
            scope = ContractScope(
                scope_type=LearningScope.DOMAIN_SPECIFIC,
                domains=scope_domains,
            )

        return ConsentDecision(
            request_id=request.request_id,
            response=response,
            user_id=request.user_id,
            contract_type=contract_type,
            duration=timedelta(days=duration_days) if duration_days else None,
            scope_restrictions=scope,
        )

    def _check_cache(self, request: ConsentRequest) -> Optional[ConsentDecision]:
        """Check cache for existing decision."""
        cache_key = f"{request.user_id}:{request.domain}"
        cached = self._cache.get(cache_key)

        if cached:
            decision, timestamp = cached
            if datetime.now() - timestamp < self.cache_ttl:
                # Return cached decision with new request_id
                return ConsentDecision(
                    request_id=request.request_id,
                    response=decision.response,
                    user_id=request.user_id,
                    contract_type=decision.contract_type,
                    duration=decision.duration,
                    scope_restrictions=decision.scope_restrictions,
                    reason="Cached decision",
                    metadata={"cached": True},
                )

            # Cache expired
            del self._cache[cache_key]

        return None

    def _check_preference(self, request: ConsentRequest) -> Optional[ConsentDecision]:
        """Check if user has set a preference."""
        user_prefs = self._preferences.get(request.user_id, {})

        # Check specific domain
        if request.domain in user_prefs:
            response = user_prefs[request.domain]
            return self.create_decision(request, response)

        # Check wildcard
        if "*" in user_prefs:
            response = user_prefs["*"]
            return self.create_decision(request, response)

        return None

    def _cache_decision(
        self,
        request: ConsentRequest,
        decision: ConsentDecision,
    ) -> None:
        """Cache a decision."""
        cache_key = f"{request.user_id}:{request.domain}"
        self._cache[cache_key] = (decision, datetime.now())

    def _prompt_callback(self, request: ConsentRequest) -> ConsentDecision:
        """Get consent via callback."""
        if not self._callback:
            logger.warning("No consent callback set, defaulting to deny")
            return self._auto_deny(request)

        try:
            return self._callback(request)
        except Exception as e:
            logger.error(f"Consent callback error: {e}")
            return self._auto_deny(request)

    def _prompt_cli(self, request: ConsentRequest) -> ConsentDecision:
        """Get consent via CLI (placeholder for actual implementation)."""
        # This would typically involve interactive prompts
        # For now, we'll use a simple approach
        logger.info(f"CLI consent prompt for: {request.domain}")
        logger.info(f"Purpose: {request.purpose}")
        logger.info(f"Data types: {', '.join(request.data_types)}")

        # In a real implementation, this would wait for user input
        # For safety, we default to deny
        return self._auto_deny(request)

    def _auto_deny(self, request: ConsentRequest) -> ConsentDecision:
        """Automatically deny consent."""
        return ConsentDecision(
            request_id=request.request_id,
            response=ConsentResponse.DENY,
            user_id=request.user_id,
            contract_type=ContractType.NO_LEARNING,
            reason="Auto-denied (default policy)",
        )

    def _auto_allow(self, request: ConsentRequest) -> ConsentDecision:
        """Automatically allow consent (use with caution!)."""
        logger.warning(f"Auto-allowing consent for {request.domain} - USE WITH CAUTION")
        return ConsentDecision(
            request_id=request.request_id,
            response=ConsentResponse.ALLOW,
            user_id=request.user_id,
            contract_type=ContractType.FULL_CONSENT,
            reason="Auto-allowed (policy)",
        )


@dataclass
class ConsentUI:
    """
    UI elements for consent dialogs.

    Provides formatted text for various consent scenarios.
    """

    @staticmethod
    def format_consent_dialog(request: ConsentRequest) -> str:
        """Format a consent dialog for display."""
        lines = [
            "=" * 50,
            "LEARNING CONSENT REQUEST",
            "=" * 50,
            "",
            f"Domain: {request.domain}",
            f"Description: {request.description}",
            "",
        ]

        if request.purpose:
            lines.append(f"Purpose: {request.purpose}")

        if request.data_types:
            lines.append(f"Data Types: {', '.join(request.data_types)}")

        if request.retention_period:
            lines.append(f"Retention: {request.retention_period}")

        if request.sharing_info:
            lines.append(f"Sharing: {request.sharing_info}")

        lines.extend(
            [
                "",
                "Options:",
                "  [A] Allow - Permit learning in this domain",
                "  [B] Allow (Abstracted) - Learn patterns only, not raw data",
                "  [S] Allow (Session) - Allow for this session only",
                "  [D] Deny - Do not permit learning",
                "  [N] Deny (Always) - Never ask again for this domain",
                "",
                "=" * 50,
            ]
        )

        return "\n".join(lines)

    @staticmethod
    def format_privacy_notice() -> str:
        """Format standard privacy notice."""
        return """
PRIVACY NOTICE

When you consent to learning, the AI system may:
- Remember patterns from your interactions
- Use learned information to improve responses
- Store abstracted data in secure storage

You have the right to:
- Revoke consent at any time
- Request deletion of learned data
- View what has been learned

All learning is subject to prohibited domain restrictions.
Sensitive data (passwords, financial info) is never learned.
        """.strip()

    @staticmethod
    def format_abstraction_explanation() -> str:
        """Explain abstraction for users."""
        return """
ABSTRACTION-ONLY LEARNING

When you choose abstraction-only learning:
- Raw text/data is NOT stored
- Only patterns and statistics are retained
- Personal identifiers are removed
- Specific names, dates, numbers are generalized

Example:
  Original: "John Smith called on 555-123-4567"
  Abstracted: "[NAME] called on [PHONE]"

This provides a balance between useful learning
and privacy protection.
        """.strip()


def create_consent_prompt(
    mode: ConsentMode = ConsentMode.CALLBACK,
    callback: Optional[Callable[[ConsentRequest], ConsentDecision]] = None,
) -> ConsentPrompt:
    """
    Factory function to create a consent prompt.

    Args:
        mode: Consent collection mode
        callback: Callback for obtaining consent

    Returns:
        Configured ConsentPrompt
    """
    return ConsentPrompt(mode=mode, callback=callback)
