"""
Prohibited Domain Checker

Enforces domain-based restrictions on learning contracts.
Maintains a registry of prohibited domains that cannot be learned.
"""

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


class DomainCategory(Enum):
    """Categories of prohibited domains."""

    PERSONAL_IDENTITY = auto()  # SSN, ID numbers
    FINANCIAL = auto()  # Bank accounts, credit cards
    MEDICAL = auto()  # Health records
    CREDENTIALS = auto()  # Passwords, API keys
    BIOMETRIC = auto()  # Fingerprints, face data
    LEGAL = auto()  # Attorney-client, legal proceedings
    GOVERNMENT = auto()  # Classified, security clearance
    MINOR_DATA = auto()  # Children's personal data
    RELATIONSHIP = auto()  # Private relationship details
    CUSTOM = auto()  # User-defined


class ProhibitionLevel(Enum):
    """Level of domain prohibition."""

    ABSOLUTE = auto()  # Can never be learned, no override
    STRONG = auto()  # Requires explicit human override
    DEFAULT = auto()  # Prohibited by default, can be enabled
    ADVISORY = auto()  # Warning only, user can proceed


@dataclass
class ProhibitedDomain:
    """A prohibited domain definition."""

    domain_id: str
    name: str
    category: DomainCategory
    level: ProhibitionLevel
    patterns: List[str] = field(default_factory=list)
    keywords: Set[str] = field(default_factory=set)
    description: str = ""
    reason: str = ""
    legal_basis: str = ""
    added_at: datetime = field(default_factory=datetime.now)
    added_by: str = "system"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def matches(self, content: str) -> bool:
        """Check if content matches this prohibited domain."""
        content_lower = content.lower()

        # Check keywords
        for keyword in self.keywords:
            if keyword.lower() in content_lower:
                return True

        # Check patterns
        for pattern in self.patterns:
            if re.search(pattern, content, re.IGNORECASE):
                return True

        return False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "domain_id": self.domain_id,
            "name": self.name,
            "category": self.category.name,
            "level": self.level.name,
            "patterns": self.patterns,
            "keywords": list(self.keywords),
            "description": self.description,
            "reason": self.reason,
            "legal_basis": self.legal_basis,
            "added_at": self.added_at.isoformat(),
            "added_by": self.added_by,
            "metadata": self.metadata,
        }


@dataclass
class DomainCheckResult:
    """Result of a domain check."""

    is_prohibited: bool
    matching_domains: List[ProhibitedDomain] = field(default_factory=list)
    highest_level: Optional[ProhibitionLevel] = None
    can_override: bool = True
    reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "is_prohibited": self.is_prohibited,
            "matching_domains": [d.to_dict() for d in self.matching_domains],
            "highest_level": self.highest_level.name if self.highest_level else None,
            "can_override": self.can_override,
            "reasons": self.reasons,
        }


class ProhibitedDomainChecker:
    """
    Checks content against prohibited domains.

    Maintains a registry of domains that cannot be learned
    and provides checking capabilities.
    """

    def __init__(self, load_defaults: bool = True):
        """
        Initialize the domain checker.

        Args:
            load_defaults: Load default prohibited domains
        """
        self._domains: Dict[str, ProhibitedDomain] = {}
        self._category_index: Dict[DomainCategory, Set[str]] = {
            cat: set() for cat in DomainCategory
        }
        self._on_violation: Optional[Callable[[DomainCheckResult], None]] = None

        if load_defaults:
            self._load_default_domains()

    def add_domain(self, domain: ProhibitedDomain) -> None:
        """Add a prohibited domain."""
        self._domains[domain.domain_id] = domain
        self._category_index[domain.category].add(domain.domain_id)
        logger.info(f"Added prohibited domain: {domain.name}")

    def remove_domain(self, domain_id: str) -> bool:
        """Remove a prohibited domain (if not ABSOLUTE)."""
        domain = self._domains.get(domain_id)
        if not domain:
            return False

        if domain.level == ProhibitionLevel.ABSOLUTE:
            logger.warning(f"Cannot remove ABSOLUTE domain: {domain_id}")
            return False

        del self._domains[domain_id]
        self._category_index[domain.category].discard(domain_id)
        return True

    def check(
        self,
        content: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> DomainCheckResult:
        """
        Check content against all prohibited domains.

        Args:
            content: Content to check
            context: Optional context (domain name, category hints, etc.)

        Returns:
            DomainCheckResult
        """
        result = DomainCheckResult(is_prohibited=False)

        # Check against all domains
        for domain in self._domains.values():
            if domain.matches(content):
                result.is_prohibited = True
                result.matching_domains.append(domain)
                result.reasons.append(domain.reason or f"Matches prohibited domain: {domain.name}")

                # Track highest level
                if result.highest_level is None or domain.level.value < result.highest_level.value:
                    result.highest_level = domain.level

        # Determine if override is possible
        if result.is_prohibited:
            result.can_override = all(
                d.level != ProhibitionLevel.ABSOLUTE for d in result.matching_domains
            )

        # Trigger callback if set
        if result.is_prohibited and self._on_violation:
            try:
                self._on_violation(result)
            except Exception as e:
                logger.error(f"Violation callback error: {e}")

        return result

    def check_domain_name(self, domain_name: str) -> DomainCheckResult:
        """Check if a domain name itself is prohibited."""
        result = DomainCheckResult(is_prohibited=False)

        domain_lower = domain_name.lower()

        for domain in self._domains.values():
            # Check if domain name matches keywords
            if domain.name.lower() in domain_lower or domain_lower in domain.name.lower():
                result.is_prohibited = True
                result.matching_domains.append(domain)
                result.reasons.append(f"Domain name matches: {domain.name}")

            # Check pattern match on domain name
            for pattern in domain.patterns:
                if re.search(pattern, domain_name, re.IGNORECASE):
                    result.is_prohibited = True
                    if domain not in result.matching_domains:
                        result.matching_domains.append(domain)
                    result.reasons.append(f"Domain name pattern match: {domain.name}")

        if result.is_prohibited:
            result.highest_level = min(
                (d.level for d in result.matching_domains),
                key=lambda l: l.value,
            )
            result.can_override = all(
                d.level != ProhibitionLevel.ABSOLUTE for d in result.matching_domains
            )

        return result

    def get_domain(self, domain_id: str) -> Optional[ProhibitedDomain]:
        """Get a domain by ID."""
        return self._domains.get(domain_id)

    def get_domains_by_category(self, category: DomainCategory) -> List[ProhibitedDomain]:
        """Get all domains in a category."""
        domain_ids = self._category_index.get(category, set())
        return [self._domains[did] for did in domain_ids if did in self._domains]

    def get_all_domains(self) -> List[ProhibitedDomain]:
        """Get all prohibited domains."""
        return list(self._domains.values())

    def set_violation_callback(
        self,
        callback: Callable[[DomainCheckResult], None],
    ) -> None:
        """Set callback for violations."""
        self._on_violation = callback

    def create_pattern(
        self,
        pattern_type: str,
        **kwargs,
    ) -> str:
        """Create common patterns for domain matching."""
        patterns = {
            "ssn": r"\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b",
            "credit_card": r"\b(?:\d{4}[-\s]?){3}\d{4}\b",
            "phone": r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b",
            "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
            "api_key": r"\b(?:api[_-]?key|apikey|api_secret)\s*[:=]\s*['\"]?[A-Za-z0-9_-]{20,}['\"]?",
            "password": r"\b(?:password|passwd|pwd)\s*[:=]\s*['\"]?.+?['\"]?",
            "date_of_birth": r"\b(?:dob|date\s+of\s+birth|birthdate)\s*[:=]?\s*\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b",
        }
        return patterns.get(pattern_type, "")

    def _load_default_domains(self) -> None:
        """Load default prohibited domains."""
        defaults = [
            ProhibitedDomain(
                domain_id="pii_ssn",
                name="Social Security Numbers",
                category=DomainCategory.PERSONAL_IDENTITY,
                level=ProhibitionLevel.ABSOLUTE,
                patterns=[r"\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b"],
                keywords={"ssn", "social security", "social security number"},
                description="US Social Security Numbers",
                reason="SSN is protected personal identifier",
                legal_basis="Privacy Act, various state laws",
            ),
            ProhibitedDomain(
                domain_id="financial_credit_card",
                name="Credit Card Numbers",
                category=DomainCategory.FINANCIAL,
                level=ProhibitionLevel.ABSOLUTE,
                patterns=[r"\b(?:\d{4}[-\s]?){3}\d{4}\b"],
                keywords={"credit card", "card number", "cvv", "cvc"},
                description="Credit and debit card numbers",
                reason="Financial data protected under PCI-DSS",
                legal_basis="PCI-DSS, GLBA",
            ),
            ProhibitedDomain(
                domain_id="financial_bank",
                name="Bank Account Numbers",
                category=DomainCategory.FINANCIAL,
                level=ProhibitionLevel.ABSOLUTE,
                patterns=[
                    r"\b(?:account\s*#?|acct\s*#?)\s*[:=]?\s*\d{8,17}\b",
                    r"\brouting\s*(?:number|#)\s*[:=]?\s*\d{9}\b",
                ],
                keywords={"bank account", "routing number", "iban", "swift"},
                description="Bank account and routing numbers",
                reason="Financial data is protected",
                legal_basis="GLBA, state banking laws",
            ),
            ProhibitedDomain(
                domain_id="credentials_password",
                name="Passwords and Secrets",
                category=DomainCategory.CREDENTIALS,
                level=ProhibitionLevel.ABSOLUTE,
                patterns=[
                    r"\b(?:password|passwd|pwd)\s*[:=]\s*['\"]?.+?['\"]?",
                    r"\b(?:api[_-]?key|api[_-]?secret|secret[_-]?key)\s*[:=]\s*['\"]?[A-Za-z0-9_-]{16,}['\"]?",
                    r"\b(?:auth[_-]?token|access[_-]?token|bearer)\s*[:=]\s*['\"]?[A-Za-z0-9._-]{20,}['\"]?",
                ],
                keywords={"password", "api key", "secret key", "private key", "auth token"},
                description="Authentication credentials and secrets",
                reason="Credentials must never be stored or learned",
                legal_basis="Security best practices, numerous regulations",
            ),
            ProhibitedDomain(
                domain_id="medical_records",
                name="Medical Records",
                category=DomainCategory.MEDICAL,
                level=ProhibitionLevel.STRONG,
                patterns=[
                    r"\b(?:diagnosis|dx)\s*[:=]\s*.+",
                    r"\b(?:patient\s*id|mrn)\s*[:=]?\s*\w+",
                ],
                keywords={
                    "medical record",
                    "diagnosis",
                    "prescription",
                    "patient",
                    "hipaa",
                    "phi",
                    "protected health information",
                    "treatment plan",
                },
                description="Protected health information",
                reason="Medical data protected under HIPAA",
                legal_basis="HIPAA, HITECH",
            ),
            ProhibitedDomain(
                domain_id="biometric",
                name="Biometric Data",
                category=DomainCategory.BIOMETRIC,
                level=ProhibitionLevel.ABSOLUTE,
                patterns=[],
                keywords={
                    "fingerprint",
                    "face scan",
                    "retina",
                    "iris scan",
                    "voice print",
                    "biometric",
                    "dna sequence",
                    "genetic data",
                },
                description="Biometric identifiers",
                reason="Biometric data is immutable and highly sensitive",
                legal_basis="BIPA, GDPR, CCPA",
            ),
            ProhibitedDomain(
                domain_id="minor_data",
                name="Minor's Personal Data",
                category=DomainCategory.MINOR_DATA,
                level=ProhibitionLevel.ABSOLUTE,
                patterns=[],
                keywords={
                    "child's data",
                    "minor's information",
                    "coppa",
                    "parental consent",
                    "student records",
                    "ferpa",
                },
                description="Personal data of minors",
                reason="Children's data has heightened protection",
                legal_basis="COPPA, FERPA",
            ),
            ProhibitedDomain(
                domain_id="legal_privileged",
                name="Attorney-Client Privileged",
                category=DomainCategory.LEGAL,
                level=ProhibitionLevel.STRONG,
                patterns=[],
                keywords={
                    "attorney-client",
                    "privileged communication",
                    "legal advice",
                    "work product",
                },
                description="Privileged legal communications",
                reason="Protected by attorney-client privilege",
                legal_basis="Attorney-client privilege",
            ),
            ProhibitedDomain(
                domain_id="government_classified",
                name="Classified Government Information",
                category=DomainCategory.GOVERNMENT,
                level=ProhibitionLevel.ABSOLUTE,
                patterns=[],
                keywords={
                    "classified",
                    "top secret",
                    "secret clearance",
                    "confidential clearance",
                    "national security",
                },
                description="Classified government information",
                reason="Government classified data is prohibited",
                legal_basis="Various national security laws",
            ),
        ]

        for domain in defaults:
            self.add_domain(domain)


def create_domain_checker(load_defaults: bool = True) -> ProhibitedDomainChecker:
    """
    Factory function to create a domain checker.

    Args:
        load_defaults: Load default prohibited domains

    Returns:
        Configured ProhibitedDomainChecker
    """
    return ProhibitedDomainChecker(load_defaults=load_defaults)
