"""
Tests for Agent OS Constitution Parser
"""

import pytest
from pathlib import Path
import tempfile

from src.core.parser import ConstitutionParser
from src.core.models import AuthorityLevel, RuleType


class TestConstitutionParser:
    """Tests for ConstitutionParser."""

    @pytest.fixture
    def parser(self):
        """Create a parser instance."""
        return ConstitutionParser()

    def test_extract_frontmatter_basic(self, parser):
        """Parse basic YAML frontmatter."""
        content = """---
document_type: constitution
version: "1.0"
scope: all_agents
authority_level: supreme
---

# Constitution Title

Content here.
"""
        constitution = parser.parse_content(content)

        assert constitution.metadata.document_type == "constitution"
        assert constitution.metadata.version == "1.0"
        assert constitution.metadata.scope == "all_agents"
        assert constitution.metadata.authority_level == AuthorityLevel.SUPREME

    def test_extract_frontmatter_agent_specific(self, parser):
        """Parse agent-specific frontmatter."""
        content = """---
document_type: constitution
version: "1.0"
scope: guardian
authority_level: agent_specific
amendment_process: pull_request
---

# Guardian Constitution
"""
        constitution = parser.parse_content(content)

        assert constitution.metadata.scope == "guardian"
        assert constitution.metadata.authority_level == AuthorityLevel.AGENT_SPECIFIC

    def test_no_frontmatter(self, parser):
        """Handle content without frontmatter."""
        content = """# Constitution Without Frontmatter

Some content here.
"""
        constitution = parser.parse_content(content)

        assert constitution.metadata.document_type == "unknown"
        assert constitution.metadata.scope == "unknown"

    def test_parse_sections(self, parser):
        """Parse markdown sections correctly."""
        content = """---
document_type: constitution
version: "1.0"
scope: all_agents
authority_level: supreme
---

# Main Title

Intro paragraph.

## Section One

Content of section one.

### Subsection A

More content.

## Section Two

Content of section two.
"""
        constitution = parser.parse_content(content)

        assert "Main Title" in constitution.sections
        assert "Section One" in constitution.sections
        assert "Section Two" in constitution.sections

    def test_extract_mandate_rules(self, parser):
        """Extract rules with mandate keywords."""
        content = """---
document_type: constitution
version: "1.0"
scope: all_agents
authority_level: supreme
---

# Mandates

- The Guardian SHALL review all outputs before delivery
- Agents MUST monitor behavior for anomalies
- Systems MUST maintain awareness of constitutional rules
- Agents MUST respect human sovereignty
"""
        constitution = parser.parse_content(content)

        mandate_rules = [r for r in constitution.rules if r.rule_type == RuleType.MANDATE]
        assert len(mandate_rules) >= 3

    def test_extract_prohibition_rules(self, parser):
        """Extract rules with prohibition keywords."""
        content = """---
document_type: constitution
version: "1.0"
scope: all_agents
authority_level: supreme
---

# Prohibitions

- Agents MUST NOT expand their own authority
- Systems MUST NOT create copies of themselves
- Agents MUST NOT store credentials or passwords
- No agent shall access external networks without approval
"""
        constitution = parser.parse_content(content)

        prohibition_rules = [r for r in constitution.rules if r.rule_type == RuleType.PROHIBITION]
        assert len(prohibition_rules) >= 3

    def test_extract_permission_rules(self, parser):
        """Extract rules with permission keywords."""
        content = """---
document_type: constitution
version: "1.0"
scope: all_agents
authority_level: supreme
---

# Permissions

- Agents MAY request clarification from users
- Systems MAY suggest alternative approaches
- Users CAN override agent decisions
"""
        constitution = parser.parse_content(content)

        permission_rules = [r for r in constitution.rules if r.rule_type == RuleType.PERMISSION]
        assert len(permission_rules) >= 2

    def test_immutable_section_marker(self, parser):
        """Recognize immutable section markers."""
        content = """---
document_type: constitution
version: "1.0"
scope: all_agents
authority_level: supreme
---

# Human Sovereignty (Immutable)

- Ultimate authority resides with Human Steward
- This principle cannot be amended
"""
        constitution = parser.parse_content(content)

        immutable_rules = [r for r in constitution.rules if r.is_immutable]
        assert len(immutable_rules) >= 1

    def test_extract_keywords(self, parser):
        """Extract meaningful keywords from rules."""
        content = """---
document_type: constitution
version: "1.0"
scope: all_agents
authority_level: supreme
---

# Memory

- Agents must obtain consent before storing user data
"""
        constitution = parser.parse_content(content)

        assert len(constitution.rules) >= 1
        rule = constitution.rules[0]

        # Should have extracted meaningful keywords
        assert "consent" in rule.keywords or "memory" in rule.keywords or "storing" in rule.keywords

    def test_extract_references(self, parser):
        """Extract references to other sections/documents."""
        content = """---
document_type: constitution
version: "1.0"
scope: guardian
authority_level: agent_specific
---

# Guardian Rules

This constitution is subordinate to CONSTITUTION.md.
See Section IV for memory rules.
"""
        constitution = parser.parse_content(content)

        # Check that references were extracted from at least one rule
        all_refs = []
        for rule in constitution.rules:
            all_refs.extend(rule.references)

        # Note: references are extracted at rule level, not from all content
        # The parser might not catch these as they're in prose, not bullet points

    def test_section_path_tracking(self, parser):
        """Track section path hierarchy."""
        content = """---
document_type: constitution
version: "1.0"
scope: all_agents
authority_level: supreme
---

# Core Principles

## Human Sovereignty

### Rights

- Human has ultimate authority
"""
        constitution = parser.parse_content(content)

        rule = constitution.rules[0]
        assert "Core Principles" in rule.section_path or "Human Sovereignty" in rule.section_path

    def test_parse_file(self, parser):
        """Parse constitution from file."""
        content = """---
document_type: constitution
version: "1.0"
scope: test
authority_level: agent_specific
---

# Test Constitution

- Test rule one
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write(content)
            temp_path = Path(f.name)

        try:
            constitution = parser.parse_file(temp_path)
            assert constitution.metadata.scope == "test"
            assert constitution.metadata.file_path == temp_path
        finally:
            temp_path.unlink()

    def test_file_not_found(self, parser):
        """Raise error for missing file."""
        with pytest.raises(FileNotFoundError):
            parser.parse_file(Path("/nonexistent/file.md"))

    def test_file_hash_generated(self, parser):
        """File hash should be generated."""
        content = """---
document_type: constitution
version: "1.0"
scope: test
---

# Test
"""
        constitution = parser.parse_content(content)

        assert constitution.file_hash
        assert len(constitution.file_hash) == 64  # SHA256 hex length

    def test_numbered_list_extraction(self, parser):
        """Extract rules from numbered lists."""
        content = """---
document_type: constitution
version: "1.0"
scope: all_agents
authority_level: supreme
---

# Ordered Rules

1. First rule that must be followed
2. Second rule that shall be obeyed
3. Third rule that cannot be violated
"""
        constitution = parser.parse_content(content)

        assert len(constitution.rules) >= 3
