"""
Tests for Agent OS Core Models
"""

import pytest
from datetime import datetime
from pathlib import Path

from src.core.models import (
    AuthorityLevel,
    RuleType,
    ConflictType,
    ConstitutionMetadata,
    Rule,
    RuleConflict,
    ValidationResult,
    Constitution,
    ConstitutionRegistry,
)


class TestAuthorityLevel:
    """Tests for AuthorityLevel enum."""

    def test_hierarchy_ordering(self):
        """Authority levels should have correct ordering."""
        assert AuthorityLevel.SUPREME > AuthorityLevel.SYSTEM
        assert AuthorityLevel.SYSTEM > AuthorityLevel.AGENT_SPECIFIC
        assert AuthorityLevel.AGENT_SPECIFIC > AuthorityLevel.ROLE
        assert AuthorityLevel.ROLE > AuthorityLevel.TASK

    def test_supreme_is_highest(self):
        """Supreme should be the highest authority."""
        assert AuthorityLevel.SUPREME.value == 100
        for level in AuthorityLevel:
            if level != AuthorityLevel.SUPREME:
                assert AuthorityLevel.SUPREME > level


class TestConstitutionMetadata:
    """Tests for ConstitutionMetadata."""

    def test_from_frontmatter_basic(self):
        """Create metadata from basic frontmatter."""
        frontmatter = {
            "document_type": "constitution",
            "version": "1.0",
            "effective_date": "2024-12-15",
            "scope": "all_agents",
            "authority_level": "supreme",
            "amendment_process": "pull_request",
        }

        meta = ConstitutionMetadata.from_frontmatter(frontmatter)

        assert meta.document_type == "constitution"
        assert meta.version == "1.0"
        assert meta.scope == "all_agents"
        assert meta.authority_level == AuthorityLevel.SUPREME

    def test_from_frontmatter_agent_specific(self):
        """Create metadata for agent-specific constitution."""
        frontmatter = {
            "document_type": "constitution",
            "version": "1.0",
            "scope": "guardian",
            "authority_level": "agent_specific",
        }

        meta = ConstitutionMetadata.from_frontmatter(frontmatter)

        assert meta.scope == "guardian"
        assert meta.authority_level == AuthorityLevel.AGENT_SPECIFIC

    def test_from_frontmatter_defaults(self):
        """Missing fields should get defaults."""
        meta = ConstitutionMetadata.from_frontmatter({})

        assert meta.document_type == "unknown"
        assert meta.version == "0.0.0"
        assert meta.authority_level == AuthorityLevel.AGENT_SPECIFIC


class TestRule:
    """Tests for Rule model."""

    def test_rule_id_generation(self):
        """Rule ID should be generated from content."""
        rule = Rule(
            id="",
            content="Agents must not expand their own authority",
            rule_type=RuleType.PROHIBITION,
            section="Role Boundaries",
            section_path=["Core Principles", "Role Boundaries"],
            authority_level=AuthorityLevel.SUPREME,
            scope="all_agents",
        )

        assert rule.id != ""
        assert len(rule.id) == 16  # SHA256 truncated to 16 chars

    def test_rule_id_deterministic(self):
        """Same content should produce same ID."""
        rule1 = Rule(
            id="",
            content="Test rule content",
            rule_type=RuleType.MANDATE,
            section="Test",
            section_path=["Test"],
            authority_level=AuthorityLevel.SUPREME,
            scope="all_agents",
        )

        rule2 = Rule(
            id="",
            content="Test rule content",
            rule_type=RuleType.MANDATE,
            section="Test",
            section_path=["Test"],
            authority_level=AuthorityLevel.SUPREME,
            scope="all_agents",
        )

        assert rule1.id == rule2.id

    def test_conflicts_with_opposing_types(self):
        """Mandate and prohibition on same topic should conflict."""
        mandate = Rule(
            id="",
            content="Agents must store user preferences",
            rule_type=RuleType.MANDATE,
            section="Memory",
            section_path=["Memory"],
            authority_level=AuthorityLevel.SUPREME,
            scope="all_agents",
            keywords={"store", "user", "preferences"},
        )

        prohibition = Rule(
            id="",
            content="Agents must not store user data",
            rule_type=RuleType.PROHIBITION,
            section="Privacy",
            section_path=["Privacy"],
            authority_level=AuthorityLevel.SUPREME,
            scope="all_agents",
            keywords={"store", "user", "data"},
        )

        assert mandate.conflicts_with(prohibition)


class TestValidationResult:
    """Tests for ValidationResult."""

    def test_starts_valid(self):
        """New result should be valid by default."""
        result = ValidationResult(is_valid=True)
        assert result.is_valid

    def test_add_error_invalidates(self):
        """Adding error should invalidate result."""
        result = ValidationResult(is_valid=True)
        result.add_error("Test error")

        assert not result.is_valid
        assert "Test error" in result.errors

    def test_add_warning_keeps_valid(self):
        """Adding warning should not invalidate result."""
        result = ValidationResult(is_valid=True)
        result.add_warning("Test warning")

        assert result.is_valid
        assert "Test warning" in result.warnings

    def test_add_error_conflict_invalidates(self):
        """Adding error-severity conflict should invalidate."""
        result = ValidationResult(is_valid=True)

        rule = Rule(
            id="test",
            content="Test",
            rule_type=RuleType.MANDATE,
            section="Test",
            section_path=[],
            authority_level=AuthorityLevel.SUPREME,
            scope="all_agents",
        )

        conflict = RuleConflict(
            conflict_type=ConflictType.CONTRADICTION,
            rule_a=rule,
            rule_b=rule,
            description="Test conflict",
            severity="error",
        )

        result.add_conflict(conflict)
        assert not result.is_valid


class TestConstitutionRegistry:
    """Tests for ConstitutionRegistry."""

    def test_register_and_get(self):
        """Should register and retrieve constitutions."""
        registry = ConstitutionRegistry()

        meta = ConstitutionMetadata(
            document_type="constitution",
            version="1.0",
            effective_date="2024-12-15",
            scope="test_agent",
            authority_level=AuthorityLevel.AGENT_SPECIFIC,
            amendment_process="pull_request",
        )

        constitution = Constitution(
            metadata=meta,
            rules=[],
            sections={},
            raw_content="test",
            file_hash="abc123",
        )

        registry.register(constitution)

        assert registry.get("test_agent") == constitution

    def test_get_supreme(self):
        """Should return supreme constitution."""
        registry = ConstitutionRegistry()

        supreme_meta = ConstitutionMetadata(
            document_type="constitution",
            version="1.0",
            effective_date="2024-12-15",
            scope="all_agents",
            authority_level=AuthorityLevel.SUPREME,
            amendment_process="pull_request",
        )

        supreme = Constitution(
            metadata=supreme_meta,
            rules=[],
            sections={},
            raw_content="supreme",
            file_hash="supreme123",
        )

        registry.register(supreme)

        assert registry.get_supreme() == supreme

    def test_get_rules_for_agent(self):
        """Should return rules for agent including inherited."""
        registry = ConstitutionRegistry()

        # Supreme constitution with a rule
        supreme_rule = Rule(
            id="supreme1",
            content="All agents must respect human sovereignty",
            rule_type=RuleType.MANDATE,
            section="Sovereignty",
            section_path=["Sovereignty"],
            authority_level=AuthorityLevel.SUPREME,
            scope="all_agents",
        )

        supreme = Constitution(
            metadata=ConstitutionMetadata(
                document_type="constitution",
                version="1.0",
                effective_date="2024-12-15",
                scope="all_agents",
                authority_level=AuthorityLevel.SUPREME,
                amendment_process="pull_request",
            ),
            rules=[supreme_rule],
            sections={},
            raw_content="supreme",
            file_hash="supreme123",
        )

        # Agent-specific constitution
        agent_rule = Rule(
            id="agent1",
            content="Guardian must review all outputs",
            rule_type=RuleType.MANDATE,
            section="Mandate",
            section_path=["Mandate"],
            authority_level=AuthorityLevel.AGENT_SPECIFIC,
            scope="guardian",
        )

        agent = Constitution(
            metadata=ConstitutionMetadata(
                document_type="constitution",
                version="1.0",
                effective_date="2024-12-15",
                scope="guardian",
                authority_level=AuthorityLevel.AGENT_SPECIFIC,
                amendment_process="pull_request",
            ),
            rules=[agent_rule],
            sections={},
            raw_content="agent",
            file_hash="agent123",
        )

        registry.register(supreme)
        registry.register(agent)

        rules = registry.get_rules_for_agent("guardian")

        assert len(rules) == 2
        # Supreme rules should come first (higher authority)
        assert rules[0].authority_level == AuthorityLevel.SUPREME
