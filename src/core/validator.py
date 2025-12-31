"""
Agent OS Constitution Validator

Validates constitutional documents for:
- Structural correctness
- Rule conflicts
- Precedence issues
- Authority boundary violations
"""

from itertools import combinations
from typing import Dict, List, Optional, Set, Tuple

from .models import (
    AuthorityLevel,
    ConflictType,
    Constitution,
    ConstitutionRegistry,
    Rule,
    RuleConflict,
    RuleType,
    ValidationResult,
)

# Keywords that indicate opposing concepts
OPPOSING_CONCEPTS = [
    (
        {"allow", "permit", "enable", "can", "may"},
        {"prohibit", "forbid", "prevent", "cannot", "block"},
    ),
    ({"store", "persist", "save", "remember"}, {"delete", "forget", "erase", "purge"}),
    ({"access", "read", "retrieve"}, {"deny", "block", "restrict"}),
    ({"approve", "accept", "confirm"}, {"reject", "refuse", "decline"}),
    ({"internal", "local"}, {"external", "remote", "cloud"}),
]


class ConstitutionValidator:
    """
    Validates constitutional documents and detects conflicts.

    Provides:
    - Structural validation
    - Conflict detection between rules
    - Precedence validation
    - Authority boundary checking
    """

    def __init__(self):
        self._conflict_cache: Dict[str, List[RuleConflict]] = {}

    def validate(self, constitution: Constitution) -> ValidationResult:
        """
        Perform full validation of a constitution.

        Args:
            constitution: Constitution to validate

        Returns:
            ValidationResult with all findings
        """
        result = ValidationResult(is_valid=True)

        # Validate metadata
        self._validate_metadata(constitution, result)

        # Validate structure
        self._validate_structure(constitution, result)

        # Detect internal conflicts
        conflicts = self.detect_conflicts(constitution.rules)
        for conflict in conflicts:
            result.add_conflict(conflict)

        # Validate rule coverage
        self._validate_coverage(constitution, result)

        return result

    def validate_against_supreme(
        self,
        constitution: Constitution,
        supreme: Constitution,
    ) -> ValidationResult:
        """
        Validate an agent constitution against the supreme constitution.

        Args:
            constitution: Agent-specific constitution
            supreme: Supreme constitution (CONSTITUTION.md)

        Returns:
            ValidationResult with findings
        """
        result = ValidationResult(is_valid=True)

        # Check authority level
        if constitution.metadata.authority_level >= supreme.metadata.authority_level:
            result.add_error(
                f"Agent constitution '{constitution.metadata.scope}' claims authority "
                f"level {constitution.metadata.authority_level.name} which is not less than "
                f"supreme authority level {supreme.metadata.authority_level.name}"
            )

        # Check for conflicts with supreme rules
        supreme_immutable = supreme.get_immutable_rules()

        for agent_rule in constitution.rules:
            for supreme_rule in supreme_immutable:
                if self._rules_conflict(agent_rule, supreme_rule):
                    result.add_conflict(
                        RuleConflict(
                            conflict_type=ConflictType.CONTRADICTION,
                            rule_a=agent_rule,
                            rule_b=supreme_rule,
                            description=f"Agent rule conflicts with immutable supreme rule",
                            severity="error",
                            resolution_hint="Agent rules cannot contradict immutable supreme rules",
                        )
                    )

        # Check for authority expansion
        for agent_rule in constitution.rules:
            if self._expands_authority(agent_rule, supreme):
                result.add_error(
                    f"Rule '{agent_rule.content[:50]}...' may expand authority "
                    f"beyond what is permitted by supreme constitution"
                )

        return result

    def validate_registry(self, registry: ConstitutionRegistry) -> ValidationResult:
        """
        Validate all constitutions in a registry for cross-document conflicts.

        Args:
            registry: Registry containing all constitutions

        Returns:
            ValidationResult with findings
        """
        result = ValidationResult(is_valid=True)

        # Must have a supreme constitution
        supreme = registry.get_supreme()
        if not supreme:
            result.add_error("No supreme constitution found in registry")
            return result

        # Validate each constitution against supreme
        for scope, constitution in registry.constitutions.items():
            if constitution.metadata.authority_level < AuthorityLevel.SUPREME:
                sub_result = self.validate_against_supreme(constitution, supreme)
                result.errors.extend(sub_result.errors)
                result.warnings.extend(sub_result.warnings)
                result.conflicts.extend(sub_result.conflicts)
                if not sub_result.is_valid:
                    result.is_valid = False

        # Check for cross-constitution conflicts
        all_rules = registry.get_all_rules()
        cross_conflicts = self.detect_conflicts(all_rules)
        for conflict in cross_conflicts:
            # Only report if different source files
            if conflict.rule_a.source_file != conflict.rule_b.source_file:
                result.add_conflict(conflict)

        return result

    def detect_conflicts(self, rules: List[Rule]) -> List[RuleConflict]:
        """
        Detect conflicts between a list of rules.

        Args:
            rules: List of rules to check

        Returns:
            List of detected conflicts
        """
        conflicts: List[RuleConflict] = []

        # Check each pair of rules
        for rule_a, rule_b in combinations(rules, 2):
            if self._rules_conflict(rule_a, rule_b):
                conflict = self._create_conflict(rule_a, rule_b)
                if conflict:
                    conflicts.append(conflict)

        return conflicts

    def _validate_metadata(self, constitution: Constitution, result: ValidationResult) -> None:
        """Validate constitution metadata."""
        meta = constitution.metadata

        # Check required fields
        if not meta.document_type:
            result.add_error("Missing document_type in frontmatter")

        if meta.document_type != "constitution":
            result.add_warning(f"document_type is '{meta.document_type}', expected 'constitution'")

        if not meta.version:
            result.add_warning("Missing version in frontmatter")

        if not meta.scope:
            result.add_error("Missing scope in frontmatter")

        if not meta.amendment_process:
            result.add_warning("Missing amendment_process in frontmatter")

    def _validate_structure(self, constitution: Constitution, result: ValidationResult) -> None:
        """Validate constitution structure."""
        # Check for minimum required content
        if constitution.rule_count == 0:
            result.add_warning("Constitution contains no extractable rules")

        if constitution.section_count == 0:
            result.add_warning("Constitution contains no sections")

        # Check for essential sections in supreme constitution
        if constitution.metadata.authority_level == AuthorityLevel.SUPREME:
            required_concepts = ["sovereignty", "authority", "memory", "security"]
            content_lower = constitution.raw_content.lower()

            for concept in required_concepts:
                if concept not in content_lower:
                    result.add_warning(
                        f"Supreme constitution may be missing '{concept}' provisions"
                    )

    def _validate_coverage(self, constitution: Constitution, result: ValidationResult) -> None:
        """Validate that constitution covers essential areas."""
        has_mandate = any(r.rule_type == RuleType.MANDATE for r in constitution.rules)
        has_prohibition = any(r.rule_type == RuleType.PROHIBITION for r in constitution.rules)

        if not has_mandate:
            result.add_warning("Constitution contains no mandate rules (MUST/SHALL)")

        if not has_prohibition:
            result.add_warning("Constitution contains no prohibition rules (MUST NOT)")

    def _rules_conflict(self, rule_a: Rule, rule_b: Rule) -> bool:
        """Check if two rules conflict."""
        # Same rule can't conflict with itself
        if rule_a.id == rule_b.id:
            return False

        # Rules at different authority levels don't conflict (higher wins)
        if rule_a.authority_level != rule_b.authority_level:
            return False

        # Check for opposing rule types on same topic
        if self._are_opposing_types(rule_a, rule_b):
            if self._share_topic(rule_a, rule_b):
                return True

        # Check for explicit contradiction indicators
        if self._have_opposing_concepts(rule_a.content, rule_b.content):
            if self._share_topic(rule_a, rule_b):
                return True

        return False

    def _are_opposing_types(self, rule_a: Rule, rule_b: Rule) -> bool:
        """Check if rule types are naturally opposing."""
        opposing_pairs = [
            (RuleType.MANDATE, RuleType.PROHIBITION),
            (RuleType.PERMISSION, RuleType.PROHIBITION),
        ]

        for type_1, type_2 in opposing_pairs:
            if (rule_a.rule_type == type_1 and rule_b.rule_type == type_2) or (
                rule_a.rule_type == type_2 and rule_b.rule_type == type_1
            ):
                return True

        return False

    def _share_topic(self, rule_a: Rule, rule_b: Rule) -> bool:
        """Check if two rules share a common topic (via keywords)."""
        # Need significant keyword overlap
        common = rule_a.keywords & rule_b.keywords
        min_keywords = min(len(rule_a.keywords), len(rule_b.keywords))

        if min_keywords == 0:
            return False

        # Require at least 40% overlap AND 3+ common meaningful keywords
        # This reduces false positives from incidental keyword matches
        overlap_ratio = len(common) / min_keywords
        return overlap_ratio >= 0.4 and len(common) >= 3

    def _have_opposing_concepts(self, text_a: str, text_b: str) -> bool:
        """Check if two texts contain opposing concepts."""
        text_a_lower = text_a.lower()
        text_b_lower = text_b.lower()

        for concept_a, concept_b in OPPOSING_CONCEPTS:
            a_has_first = any(c in text_a_lower for c in concept_a)
            b_has_second = any(c in text_b_lower for c in concept_b)

            a_has_second = any(c in text_a_lower for c in concept_b)
            b_has_first = any(c in text_b_lower for c in concept_a)

            # Cross-match: A has allow-type, B has deny-type (or vice versa)
            if (a_has_first and b_has_second) or (a_has_second and b_has_first):
                return True

        return False

    def _create_conflict(self, rule_a: Rule, rule_b: Rule) -> Optional[RuleConflict]:
        """Create a RuleConflict object for two conflicting rules."""
        # Determine conflict type
        if self._are_opposing_types(rule_a, rule_b):
            conflict_type = ConflictType.CONTRADICTION
            description = (
                f"Conflicting rule types: {rule_a.rule_type.name} vs {rule_b.rule_type.name} "
                f"on related topic"
            )
        elif rule_a.scope == rule_b.scope:
            conflict_type = ConflictType.SCOPE_CONFLICT
            description = f"Same scope '{rule_a.scope}' with potentially conflicting rules"
        else:
            conflict_type = ConflictType.PRECEDENCE_AMBIGUITY
            description = "Rules may conflict depending on context"

        # Determine severity
        if rule_a.is_immutable or rule_b.is_immutable:
            severity = "error"
            resolution_hint = "Immutable rules cannot be overridden"
        elif (
            rule_a.authority_level == AuthorityLevel.SUPREME
            or rule_b.authority_level == AuthorityLevel.SUPREME
        ):
            severity = "error"
            resolution_hint = "Supreme constitution rules take precedence"
        else:
            severity = "warning"
            resolution_hint = "Consider clarifying rule scope or precedence"

        return RuleConflict(
            conflict_type=conflict_type,
            rule_a=rule_a,
            rule_b=rule_b,
            description=description,
            severity=severity,
            resolution_hint=resolution_hint,
        )

    def _expands_authority(self, rule: Rule, supreme: Constitution) -> bool:
        """
        Check if a rule expands authority beyond supreme constitution limits.

        Note: Prohibitions that mention these keywords are OK - they're preventing
        expansion, not enabling it. Similarly, rules about human override are legitimate.
        """
        # Check if it's a prohibition (which is OK - prohibiting expansion is good)
        if rule.rule_type == RuleType.PROHIBITION:
            return False

        rule_lower = rule.content.lower()

        # Check if the rule is about REFUSING or BLOCKING expansion (which is OK)
        safety_context = [
            "refuse",
            "reject",
            "block",
            "deny",
            "not",
            "never",
            "prohibited",
            "forbidden",
            "veto",
            "prevent",
        ]
        if any(ctx in rule_lower for ctx in safety_context):
            return False

        # Check if "override" is in context of human override (which is legitimate)
        if "override" in rule_lower:
            if "human" in rule_lower or "steward" in rule_lower:
                return False  # Human override is legitimate

        # Look for authority expansion keywords in non-prohibition context
        expansion_indicators = [
            "self-modify",
            "self-create",
            "self-replicate",
        ]

        if any(indicator in rule_lower for indicator in expansion_indicators):
            return True

        return False


def validate_constitution(constitution: Constitution) -> ValidationResult:
    """
    Convenience function to validate a constitution.

    Args:
        constitution: Constitution to validate

    Returns:
        ValidationResult with findings
    """
    validator = ConstitutionValidator()
    return validator.validate(constitution)
