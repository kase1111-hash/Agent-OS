"""
Abstraction Guard

Enforces abstraction requirements for learning contracts.
Ensures that when contracts only allow abstracted learning,
raw data is properly anonymized/abstracted before storage.
"""

import hashlib
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


class AbstractionLevel(Enum):
    """Level of abstraction applied to data."""

    RAW = auto()  # No abstraction, raw data
    MINIMAL = auto()  # Minimal redaction of identifiers
    MODERATE = auto()  # Moderate abstraction, key details preserved
    STRONG = auto()  # Strong abstraction, patterns only
    FULL = auto()  # Full abstraction, no identifiable content


class AbstractionType(Enum):
    """Types of abstraction operations."""

    REDACTION = auto()  # Replace with [REDACTED]
    HASHING = auto()  # Replace with hash
    GENERALIZATION = auto()  # Replace specific with general
    SUPPRESSION = auto()  # Remove entirely
    PSEUDONYMIZATION = auto()  # Replace with consistent pseudonym
    AGGREGATION = auto()  # Aggregate into statistics


@dataclass
class AbstractionRule:
    """A rule for abstracting content."""

    rule_id: str
    name: str
    pattern: str
    abstraction_type: AbstractionType
    replacement: str = "[ABSTRACTED]"
    applies_to: Set[str] = field(default_factory=set)  # Content types
    min_level: AbstractionLevel = AbstractionLevel.MINIMAL
    description: str = ""

    def apply(self, content: str) -> str:
        """Apply this rule to content."""
        if self.abstraction_type == AbstractionType.REDACTION:
            return re.sub(self.pattern, self.replacement, content, flags=re.IGNORECASE)
        elif self.abstraction_type == AbstractionType.HASHING:

            def hash_match(match):
                return hashlib.sha256(match.group(0).encode()).hexdigest()[:16]

            return re.sub(self.pattern, hash_match, content, flags=re.IGNORECASE)
        elif self.abstraction_type == AbstractionType.SUPPRESSION:
            return re.sub(self.pattern, "", content, flags=re.IGNORECASE)
        elif self.abstraction_type == AbstractionType.GENERALIZATION:
            return re.sub(self.pattern, self.replacement, content, flags=re.IGNORECASE)
        else:
            return re.sub(self.pattern, self.replacement, content, flags=re.IGNORECASE)

    def matches(self, content: str) -> bool:
        """Check if content matches this rule's pattern."""
        return bool(re.search(self.pattern, content, re.IGNORECASE))


@dataclass
class AbstractionResult:
    """Result of an abstraction operation."""

    success: bool
    original_length: int
    abstracted_length: int
    level_achieved: AbstractionLevel
    rules_applied: List[str] = field(default_factory=list)
    items_abstracted: int = 0
    content: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def abstraction_ratio(self) -> float:
        """Get ratio of content that was abstracted."""
        if self.original_length == 0:
            return 0.0
        return 1.0 - (self.abstracted_length / self.original_length)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "original_length": self.original_length,
            "abstracted_length": self.abstracted_length,
            "level_achieved": self.level_achieved.name,
            "rules_applied": self.rules_applied,
            "items_abstracted": self.items_abstracted,
            "abstraction_ratio": self.abstraction_ratio,
            "metadata": self.metadata,
        }


@dataclass
class AbstractionPolicy:
    """Policy for abstraction requirements."""

    min_level: AbstractionLevel = AbstractionLevel.MODERATE
    required_rules: Set[str] = field(default_factory=set)
    forbidden_patterns: List[str] = field(default_factory=list)
    max_identifiable_ratio: float = 0.1  # Max 10% identifiable content
    require_full_pii_removal: bool = True


class AbstractionGuard:
    """
    Guards against storage of unabstracted data.

    Ensures that data is properly abstracted before storage
    when contracts require abstraction-only learning.
    """

    def __init__(self, load_defaults: bool = True):
        """
        Initialize abstraction guard.

        Args:
            load_defaults: Load default abstraction rules
        """
        self._rules: Dict[str, AbstractionRule] = {}
        self._policies: Dict[str, AbstractionPolicy] = {}
        self._pseudonym_map: Dict[str, str] = {}
        self._pseudonym_counter = 0

        if load_defaults:
            self._load_default_rules()

    def add_rule(self, rule: AbstractionRule) -> None:
        """Add an abstraction rule."""
        self._rules[rule.rule_id] = rule
        logger.debug(f"Added abstraction rule: {rule.name}")

    def remove_rule(self, rule_id: str) -> bool:
        """Remove an abstraction rule."""
        if rule_id in self._rules:
            del self._rules[rule_id]
            return True
        return False

    def add_policy(self, policy_id: str, policy: AbstractionPolicy) -> None:
        """Add an abstraction policy."""
        self._policies[policy_id] = policy

    def get_policy(self, policy_id: str) -> Optional[AbstractionPolicy]:
        """Get a policy by ID."""
        return self._policies.get(policy_id)

    def abstract(
        self,
        content: str,
        level: AbstractionLevel = AbstractionLevel.MODERATE,
        content_type: str = "",
        policy_id: Optional[str] = None,
    ) -> AbstractionResult:
        """
        Abstract content to the specified level.

        Args:
            content: Content to abstract
            level: Target abstraction level
            content_type: Type of content
            policy_id: Optional policy to apply

        Returns:
            AbstractionResult
        """
        if level == AbstractionLevel.RAW:
            return AbstractionResult(
                success=True,
                original_length=len(content),
                abstracted_length=len(content),
                level_achieved=AbstractionLevel.RAW,
                content=content,
            )

        result = AbstractionResult(
            success=True,
            original_length=len(content),
            abstracted_length=0,
            level_achieved=level,
            content=content,
        )

        abstracted_content = content
        applied_rules = []

        # Apply rules based on level
        for rule_id, rule in self._rules.items():
            if rule.min_level.value <= level.value:
                if not rule.applies_to or content_type in rule.applies_to:
                    if rule.matches(abstracted_content):
                        abstracted_content = rule.apply(abstracted_content)
                        applied_rules.append(rule_id)
                        result.items_abstracted += 1

        result.rules_applied = applied_rules
        result.abstracted_length = len(abstracted_content)
        result.content = abstracted_content

        # Apply policy if specified
        if policy_id:
            policy = self._policies.get(policy_id)
            if policy:
                result = self._apply_policy(result, policy)

        return result

    def verify_abstraction(
        self,
        content: str,
        level: AbstractionLevel,
        policy_id: Optional[str] = None,
    ) -> tuple[bool, List[str]]:
        """
        Verify that content meets abstraction requirements.

        Args:
            content: Content to verify
            level: Required abstraction level
            policy_id: Optional policy to check against

        Returns:
            Tuple of (is_valid, list of issues)
        """
        issues = []

        # Check if any forbidden patterns are present
        for rule_id, rule in self._rules.items():
            if rule.min_level.value <= level.value:
                if rule.matches(content):
                    issues.append(f"Contains unabstracted content matching: {rule.name}")

        # Check policy if specified
        if policy_id:
            policy = self._policies.get(policy_id)
            if policy:
                for pattern in policy.forbidden_patterns:
                    if re.search(pattern, content, re.IGNORECASE):
                        issues.append(f"Contains forbidden pattern: {pattern}")

                # Check required rules were applied
                # This is a heuristic - we check if content looks like it has
                # abstraction markers
                for rule_id in policy.required_rules:
                    rule = self._rules.get(rule_id)
                    if rule and rule.matches(content):
                        issues.append(f"Required rule not fully applied: {rule.name}")

        return len(issues) == 0, issues

    def pseudonymize(
        self,
        content: str,
        entity_patterns: Optional[List[str]] = None,
    ) -> AbstractionResult:
        """
        Replace identifiable entities with consistent pseudonyms.

        Args:
            content: Content to pseudonymize
            entity_patterns: Patterns to match entities

        Returns:
            AbstractionResult
        """
        result = AbstractionResult(
            success=True,
            original_length=len(content),
            abstracted_length=0,
            level_achieved=AbstractionLevel.MODERATE,
            content=content,
        )

        patterns = entity_patterns or [
            r"\b[A-Z][a-z]+\s+[A-Z][a-z]+\b",  # Names
            r"\b\S+@\S+\.\S+\b",  # Emails
        ]

        pseudonymized = content
        for pattern in patterns:

            def replace_with_pseudonym(match):
                original = match.group(0)
                if original not in self._pseudonym_map:
                    self._pseudonym_counter += 1
                    self._pseudonym_map[original] = f"[ENTITY_{self._pseudonym_counter}]"
                result.items_abstracted += 1
                return self._pseudonym_map[original]

            pseudonymized = re.sub(pattern, replace_with_pseudonym, pseudonymized)

        result.content = pseudonymized
        result.abstracted_length = len(pseudonymized)
        result.rules_applied.append("pseudonymization")

        return result

    def aggregate(
        self,
        items: List[str],
        aggregation_type: str = "count",
    ) -> Dict[str, Any]:
        """
        Aggregate items into anonymous statistics.

        Args:
            items: Items to aggregate
            aggregation_type: Type of aggregation

        Returns:
            Aggregated statistics
        """
        if aggregation_type == "count":
            return {"total_count": len(items)}
        elif aggregation_type == "length_stats":
            lengths = [len(item) for item in items]
            if not lengths:
                return {"min": 0, "max": 0, "avg": 0}
            return {
                "min": min(lengths),
                "max": max(lengths),
                "avg": sum(lengths) / len(lengths),
                "count": len(lengths),
            }
        elif aggregation_type == "word_frequency":
            words: Dict[str, int] = {}
            for item in items:
                for word in item.lower().split():
                    words[word] = words.get(word, 0) + 1
            # Return top 10 most common
            sorted_words = sorted(words.items(), key=lambda x: x[1], reverse=True)
            return {"top_words": dict(sorted_words[:10])}
        else:
            return {"count": len(items)}

    def check_identifiability(self, content: str) -> Dict[str, Any]:
        """
        Check how identifiable content is.

        Returns metrics on potential identifying information.
        """
        result = {
            "potential_names": 0,
            "potential_emails": 0,
            "potential_phones": 0,
            "potential_ids": 0,
            "identifiability_score": 0.0,
        }

        # Count potential names (capitalized word pairs)
        result["potential_names"] = len(re.findall(r"\b[A-Z][a-z]+\s+[A-Z][a-z]+\b", content))

        # Count potential emails
        result["potential_emails"] = len(re.findall(r"\b\S+@\S+\.\S+\b", content))

        # Count potential phones
        result["potential_phones"] = len(re.findall(r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b", content))

        # Count potential IDs
        result["potential_ids"] = len(re.findall(r"\b(?:ID|id)[:\s]?\w+\b", content))

        # Calculate score
        total_identifiable = sum(
            [
                result["potential_names"],
                result["potential_emails"],
                result["potential_phones"],
                result["potential_ids"],
            ]
        )

        word_count = len(content.split())
        if word_count > 0:
            result["identifiability_score"] = min(1.0, total_identifiable / word_count * 10)

        return result

    def get_rules(self) -> List[AbstractionRule]:
        """Get all abstraction rules."""
        return list(self._rules.values())

    def get_rule(self, rule_id: str) -> Optional[AbstractionRule]:
        """Get a specific rule."""
        return self._rules.get(rule_id)

    def reset_pseudonyms(self) -> None:
        """Reset pseudonym mapping."""
        self._pseudonym_map.clear()
        self._pseudonym_counter = 0

    def _apply_policy(
        self,
        result: AbstractionResult,
        policy: AbstractionPolicy,
    ) -> AbstractionResult:
        """Apply policy to abstraction result."""
        content = result.content

        # Apply required rules
        for rule_id in policy.required_rules:
            rule = self._rules.get(rule_id)
            if rule and rule.matches(content):
                content = rule.apply(content)
                result.rules_applied.append(rule_id)

        # Check forbidden patterns
        for pattern in policy.forbidden_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                content = re.sub(pattern, "[FORBIDDEN]", content, flags=re.IGNORECASE)
                result.items_abstracted += 1

        result.content = content
        result.abstracted_length = len(content)

        return result

    def _load_default_rules(self) -> None:
        """Load default abstraction rules."""
        defaults = [
            AbstractionRule(
                rule_id="email",
                name="Email Addresses",
                pattern=r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
                abstraction_type=AbstractionType.REDACTION,
                replacement="[EMAIL]",
                min_level=AbstractionLevel.MINIMAL,
                description="Redact email addresses",
            ),
            AbstractionRule(
                rule_id="phone",
                name="Phone Numbers",
                pattern=r"\b(?:\+?1[-.]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b",
                abstraction_type=AbstractionType.REDACTION,
                replacement="[PHONE]",
                min_level=AbstractionLevel.MINIMAL,
                description="Redact phone numbers",
            ),
            AbstractionRule(
                rule_id="ssn",
                name="Social Security Numbers",
                pattern=r"\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b",
                abstraction_type=AbstractionType.REDACTION,
                replacement="[SSN]",
                min_level=AbstractionLevel.MINIMAL,
                description="Redact SSNs",
            ),
            AbstractionRule(
                rule_id="credit_card",
                name="Credit Card Numbers",
                pattern=r"\b(?:\d{4}[-\s]?){3}\d{4}\b",
                abstraction_type=AbstractionType.REDACTION,
                replacement="[CREDIT_CARD]",
                min_level=AbstractionLevel.MINIMAL,
                description="Redact credit card numbers",
            ),
            AbstractionRule(
                rule_id="address",
                name="Street Addresses",
                pattern=r"\b\d+\s+[A-Za-z]+\s+(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln|Drive|Dr)\b",
                abstraction_type=AbstractionType.GENERALIZATION,
                replacement="[ADDRESS]",
                min_level=AbstractionLevel.MODERATE,
                description="Generalize street addresses",
            ),
            AbstractionRule(
                rule_id="date",
                name="Specific Dates",
                pattern=r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b|\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b",
                abstraction_type=AbstractionType.GENERALIZATION,
                replacement="[DATE]",
                min_level=AbstractionLevel.MODERATE,
                description="Generalize dates",
            ),
            AbstractionRule(
                rule_id="name_pattern",
                name="Person Names",
                pattern=r"\bMr\.?\s+[A-Z][a-z]+|Mrs\.?\s+[A-Z][a-z]+|Ms\.?\s+[A-Z][a-z]+|Dr\.?\s+[A-Z][a-z]+\b",
                abstraction_type=AbstractionType.REDACTION,
                replacement="[NAME]",
                min_level=AbstractionLevel.MODERATE,
                description="Redact formal names",
            ),
            AbstractionRule(
                rule_id="ip_address",
                name="IP Addresses",
                pattern=r"\b(?:\d{1,3}\.){3}\d{1,3}\b",
                abstraction_type=AbstractionType.REDACTION,
                replacement="[IP]",
                min_level=AbstractionLevel.MINIMAL,
                description="Redact IP addresses",
            ),
            AbstractionRule(
                rule_id="api_key",
                name="API Keys",
                pattern=r"\b(?:api[_-]?key|api[_-]?secret)\s*[:=]\s*['\"]?[A-Za-z0-9_-]{16,}['\"]?",
                abstraction_type=AbstractionType.REDACTION,
                replacement="[API_KEY=REDACTED]",
                min_level=AbstractionLevel.MINIMAL,
                description="Redact API keys",
            ),
            AbstractionRule(
                rule_id="monetary",
                name="Monetary Values",
                pattern=r"\$\d+(?:,\d{3})*(?:\.\d{2})?",
                abstraction_type=AbstractionType.GENERALIZATION,
                replacement="[AMOUNT]",
                min_level=AbstractionLevel.STRONG,
                description="Generalize monetary amounts",
            ),
        ]

        for rule in defaults:
            self.add_rule(rule)


def create_abstraction_guard(load_defaults: bool = True) -> AbstractionGuard:
    """
    Factory function to create an abstraction guard.

    Args:
        load_defaults: Load default abstraction rules

    Returns:
        Configured AbstractionGuard
    """
    return AbstractionGuard(load_defaults=load_defaults)
