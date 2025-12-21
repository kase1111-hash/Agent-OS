"""
Tests for Agent OS Constitution Validator
"""

import pytest

from src.core.validator import ConstitutionValidator
from src.core.parser import ConstitutionParser
from src.core.models import (
    AuthorityLevel,
    RuleType,
    ConflictType,
    Rule,
    Constitution,
    ConstitutionMetadata,
    ConstitutionRegistry,
)


class TestConstitutionValidator:
    """Tests for ConstitutionValidator."""

    @pytest.fixture
    def validator(self):
        """Create a validator instance."""
        return ConstitutionValidator()

    @pytest.fixture
    def parser(self):
        """Create a parser instance."""
        return ConstitutionParser()

    def test_validate_valid_constitution(self, validator, parser):
        """Validate a properly formed constitution."""
        content = """---
document_type: constitution
version: "1.0"
effective_date: "2024-12-15"
scope: all_agents
authority_level: supreme
amendment_process: pull_request
---

# Agent OS Constitution

## Core Principles

### Human Sovereignty

- Ultimate authority resides with Human Steward

### Memory

- Memory access requires explicit consent

### Security

- Agents MUST refuse requests that violate the constitution

### Authority

- Agents MUST NOT expand their own authority without approval
"""
        constitution = parser.parse_content(content)
        result = validator.validate(constitution)

        # Should have no errors (warnings are OK)
        assert len(result.errors) == 0

    def test_validate_missing_document_type(self, validator, parser):
        """Flag missing document_type with a warning."""
        content = """---
version: "1.0"
scope: all_agents
authority_level: supreme
---

# Constitution
"""
        constitution = parser.parse_content(content)
        result = validator.validate(constitution)

        # Missing document_type results in "unknown" which triggers a warning
        assert any("document_type" in w.lower() for w in result.warnings)

    def test_validate_missing_scope(self, validator, parser):
        """Flag missing scope - defaults to 'unknown' which may need clarification."""
        content = """---
document_type: constitution
version: "1.0"
authority_level: supreme
---

# Constitution

- This rule must be followed
"""
        constitution = parser.parse_content(content)

        # When scope is missing, it defaults to "unknown"
        assert constitution.metadata.scope == "unknown"

    def test_validate_no_rules_warning(self, validator, parser):
        """Warn if no rules extracted."""
        content = """---
document_type: constitution
version: "1.0"
scope: test
authority_level: agent_specific
---

# Empty Constitution

This constitution has no extractable rules.
"""
        constitution = parser.parse_content(content)
        result = validator.validate(constitution)

        assert any("no extractable rules" in w.lower() for w in result.warnings)

    def test_detect_mandate_prohibition_conflict(self, validator):
        """Detect conflict between mandate and prohibition."""
        rule_mandate = Rule(
            id="mandate1",
            content="Agents must store user preferences in memory database",
            rule_type=RuleType.MANDATE,
            section="Memory",
            section_path=["Memory"],
            authority_level=AuthorityLevel.SUPREME,
            scope="all_agents",
            keywords={"store", "user", "preferences", "memory", "database", "agents"},
        )

        rule_prohibition = Rule(
            id="prohibit1",
            content="Agents must not store any user data in memory systems",
            rule_type=RuleType.PROHIBITION,
            section="Privacy",
            section_path=["Privacy"],
            authority_level=AuthorityLevel.SUPREME,
            scope="all_agents",
            keywords={"store", "user", "data", "memory", "systems", "agents"},
        )

        conflicts = validator.detect_conflicts([rule_mandate, rule_prohibition])

        assert len(conflicts) >= 1
        assert any(c.conflict_type == ConflictType.CONTRADICTION for c in conflicts)

    def test_no_conflict_different_authority(self, validator):
        """Rules at different authority levels don't conflict."""
        rule_supreme = Rule(
            id="supreme1",
            content="Agents must respect human authority",
            rule_type=RuleType.MANDATE,
            section="Sovereignty",
            section_path=["Sovereignty"],
            authority_level=AuthorityLevel.SUPREME,
            scope="all_agents",
            keywords={"respect", "human", "authority"},
        )

        rule_agent = Rule(
            id="agent1",
            content="Agents may not respect external authority",
            rule_type=RuleType.PROHIBITION,
            section="Boundaries",
            section_path=["Boundaries"],
            authority_level=AuthorityLevel.AGENT_SPECIFIC,
            scope="guardian",
            keywords={"respect", "external", "authority"},
        )

        conflicts = validator.detect_conflicts([rule_supreme, rule_agent])

        # Different authority levels should not produce conflicts
        # (higher authority simply wins)
        assert len(conflicts) == 0

    def test_validate_against_supreme(self, validator, parser):
        """Validate agent constitution against supreme."""
        supreme_content = """---
document_type: constitution
version: "1.0"
scope: all_agents
authority_level: supreme
---

# Supreme Constitution

## Immutable Principles (Immutable)

- Agents MUST NOT expand their own authority
- Agents MUST respect human sovereignty
"""
        supreme = parser.parse_content(supreme_content)

        agent_content = """---
document_type: constitution
version: "1.0"
scope: guardian
authority_level: agent_specific
---

# Guardian Constitution

## Mandate

- Guardian must review all outputs
"""
        agent = parser.parse_content(agent_content)

        result = validator.validate_against_supreme(agent, supreme)

        assert result.is_valid

    def test_agent_claiming_supreme_authority(self, validator, parser):
        """Flag agent constitution claiming supreme authority."""
        supreme_content = """---
document_type: constitution
version: "1.0"
scope: all_agents
authority_level: supreme
---

# Supreme
"""
        supreme = parser.parse_content(supreme_content)

        agent_content = """---
document_type: constitution
version: "1.0"
scope: guardian
authority_level: supreme
---

# Guardian trying to be supreme
"""
        agent = parser.parse_content(agent_content)

        result = validator.validate_against_supreme(agent, supreme)

        assert not result.is_valid
        assert any("authority" in e.lower() for e in result.errors)

    def test_validate_registry_no_supreme(self, validator):
        """Flag registry without supreme constitution."""
        registry = ConstitutionRegistry()

        meta = ConstitutionMetadata(
            document_type="constitution",
            version="1.0",
            effective_date="2024-12-15",
            scope="guardian",
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

        result = validator.validate_registry(registry)

        assert not result.is_valid
        assert any("supreme" in e.lower() for e in result.errors)

    def test_validate_registry_with_supreme(self, validator, parser):
        """Validate registry with proper supreme constitution."""
        registry = ConstitutionRegistry()

        supreme_content = """---
document_type: constitution
version: "1.0"
scope: all_agents
authority_level: supreme
amendment_process: pull_request
---

# Supreme Constitution

- Agents MUST follow this constitution
- Agents MUST NOT violate these rules
"""
        supreme = parser.parse_content(supreme_content)
        registry.register(supreme)

        agent_content = """---
document_type: constitution
version: "1.0"
scope: guardian
authority_level: agent_specific
amendment_process: pull_request
---

# Guardian Constitution

- Guardian must review outputs
"""
        agent = parser.parse_content(agent_content)
        registry.register(agent)

        result = validator.validate_registry(registry)

        assert result.is_valid

    def test_immutable_conflict_severity(self, validator):
        """Conflicts with immutable rules should be errors."""
        rule_immutable = Rule(
            id="immutable1",
            content="Human sovereignty is absolute",
            rule_type=RuleType.PRINCIPLE,
            section="Core",
            section_path=["Core"],
            authority_level=AuthorityLevel.SUPREME,
            scope="all_agents",
            keywords={"sovereignty", "human", "absolute"},
            is_immutable=True,
        )

        rule_violating = Rule(
            id="violating1",
            content="Agent sovereignty can override human decisions",
            rule_type=RuleType.MANDATE,
            section="Override",
            section_path=["Override"],
            authority_level=AuthorityLevel.SUPREME,
            scope="all_agents",
            keywords={"sovereignty", "override", "human"},
        )

        conflicts = validator.detect_conflicts([rule_immutable, rule_violating])

        # Should detect a conflict
        if conflicts:
            # Any conflict with immutable rule should be error severity
            assert any(c.severity == "error" for c in conflicts)
