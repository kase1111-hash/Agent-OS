"""
Integration Tests for Agent OS Constitutional Kernel

These tests validate the kernel against the actual CONSTITUTION.md
and agent constitutions in the repository.
"""

import pytest
from pathlib import Path
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.parser import ConstitutionParser
from src.core.validator import ConstitutionValidator
from src.core.constitution import ConstitutionalKernel, RequestContext, create_kernel
from src.core.models import AuthorityLevel, RuleType


# Get project root
PROJECT_ROOT = Path(__file__).parent.parent
CONSTITUTION_PATH = PROJECT_ROOT / "CONSTITUTION.md"
AGENTS_DIR = PROJECT_ROOT / "agents"


class TestRealConstitution:
    """Tests against the real CONSTITUTION.md file."""

    @pytest.fixture
    def parser(self):
        return ConstitutionParser()

    @pytest.fixture
    def validator(self):
        return ConstitutionValidator()

    @pytest.mark.skipif(not CONSTITUTION_PATH.exists(), reason="CONSTITUTION.md not found")
    def test_parse_constitution_md(self, parser):
        """Parse the main CONSTITUTION.md file."""
        constitution = parser.parse_file(CONSTITUTION_PATH)

        # Validate metadata
        assert constitution.metadata.document_type == "constitution"
        assert constitution.metadata.authority_level == AuthorityLevel.SUPREME
        assert constitution.metadata.scope == "all_agents"

    @pytest.mark.skipif(not CONSTITUTION_PATH.exists(), reason="CONSTITUTION.md not found")
    def test_constitution_has_rules(self, parser):
        """CONSTITUTION.md should have extractable rules."""
        constitution = parser.parse_file(CONSTITUTION_PATH)

        assert constitution.rule_count > 0, "Constitution should have rules"
        print(f"\nExtracted {constitution.rule_count} rules from CONSTITUTION.md")

    @pytest.mark.skipif(not CONSTITUTION_PATH.exists(), reason="CONSTITUTION.md not found")
    def test_constitution_has_mandates(self, parser):
        """CONSTITUTION.md should have mandate rules."""
        constitution = parser.parse_file(CONSTITUTION_PATH)

        mandates = constitution.get_rules_by_type(RuleType.MANDATE)
        assert len(mandates) > 0, "Constitution should have mandate rules (MUST/SHALL)"
        print(f"\nFound {len(mandates)} mandate rules")

    @pytest.mark.skipif(not CONSTITUTION_PATH.exists(), reason="CONSTITUTION.md not found")
    def test_constitution_has_prohibitions(self, parser):
        """CONSTITUTION.md should have prohibition rules."""
        constitution = parser.parse_file(CONSTITUTION_PATH)

        prohibitions = constitution.get_rules_by_type(RuleType.PROHIBITION)
        assert len(prohibitions) > 0, "Constitution should have prohibition rules (MUST NOT)"
        print(f"\nFound {len(prohibitions)} prohibition rules")

    @pytest.mark.skipif(not CONSTITUTION_PATH.exists(), reason="CONSTITUTION.md not found")
    def test_constitution_has_immutable_rules(self, parser):
        """CONSTITUTION.md should have immutable rules."""
        constitution = parser.parse_file(CONSTITUTION_PATH)

        immutable = constitution.get_immutable_rules()
        # May or may not have explicitly marked immutable rules
        print(f"\nFound {len(immutable)} explicitly immutable rules")

    @pytest.mark.skipif(not CONSTITUTION_PATH.exists(), reason="CONSTITUTION.md not found")
    def test_validate_constitution_md(self, parser, validator):
        """Validate CONSTITUTION.md structure."""
        constitution = parser.parse_file(CONSTITUTION_PATH)
        result = validator.validate(constitution)

        # Print any issues for debugging
        if result.errors:
            print(f"\nErrors: {result.errors}")
        if result.warnings:
            print(f"\nWarnings: {result.warnings}")
        if result.conflicts:
            print(f"\nConflicts detected: {len(result.conflicts)}")

        # Core constitution should have no hard errors (conflicts are informational
        # for a large document with many rules that may share keywords)
        assert len(result.errors) == 0, f"CONSTITUTION.md has errors: {result.errors}"

        # Should have rules extracted
        assert constitution.rule_count > 10, "Expected many rules from CONSTITUTION.md"

    @pytest.mark.skipif(not CONSTITUTION_PATH.exists(), reason="CONSTITUTION.md not found")
    def test_constitution_sections(self, parser):
        """CONSTITUTION.md should have key sections."""
        constitution = parser.parse_file(CONSTITUTION_PATH)

        # Check for expected key concepts in sections
        sections_lower = {k.lower(): v for k, v in constitution.sections.items()}

        # Should have sections about sovereignty, memory, security
        content_lower = constitution.raw_content.lower()

        assert "sovereignty" in content_lower, "Missing sovereignty content"
        assert "memory" in content_lower, "Missing memory content"
        assert "security" in content_lower or "refusal" in content_lower, "Missing security content"


class TestAgentConstitutions:
    """Tests for agent-specific constitutions."""

    @pytest.fixture
    def parser(self):
        return ConstitutionParser()

    @pytest.fixture
    def validator(self):
        return ConstitutionValidator()

    def _get_agent_constitutions(self):
        """Find all agent constitution files."""
        if not AGENTS_DIR.exists():
            return []
        return list(AGENTS_DIR.rglob("constitution.md"))

    @pytest.mark.skipif(not AGENTS_DIR.exists(), reason="agents directory not found")
    def test_find_agent_constitutions(self):
        """Find agent constitution files."""
        files = self._get_agent_constitutions()
        print(f"\nFound {len(files)} agent constitutions")
        for f in files:
            print(f"  - {f.relative_to(PROJECT_ROOT)}")

    @pytest.mark.skipif(not AGENTS_DIR.exists(), reason="agents directory not found")
    def test_parse_all_agent_constitutions(self, parser):
        """Parse all agent constitutions."""
        files = self._get_agent_constitutions()

        for file_path in files:
            try:
                constitution = parser.parse_file(file_path)
                assert constitution.metadata.authority_level < AuthorityLevel.SUPREME, \
                    f"{file_path} claims supreme authority"
                print(f"\n  Parsed: {file_path.relative_to(PROJECT_ROOT)}")
                print(f"    Scope: {constitution.metadata.scope}")
                print(f"    Rules: {constitution.rule_count}")
            except Exception as e:
                pytest.fail(f"Failed to parse {file_path}: {e}")

    @pytest.mark.skipif(
        not CONSTITUTION_PATH.exists() or not AGENTS_DIR.exists(),
        reason="Constitution files not found"
    )
    def test_validate_against_supreme(self, parser, validator):
        """Validate agent constitutions against supreme constitution."""
        supreme = parser.parse_file(CONSTITUTION_PATH)

        for file_path in self._get_agent_constitutions():
            agent = parser.parse_file(file_path)
            result = validator.validate_against_supreme(agent, supreme)

            if not result.is_valid:
                print(f"\n{file_path}: Validation issues:")
                for error in result.errors:
                    print(f"  ERROR: {error}")
                for warning in result.warnings:
                    print(f"  WARNING: {warning}")

            # Agent constitutions should be valid against supreme
            assert result.is_valid, \
                f"{file_path} violates supreme constitution: {result.errors}"


class TestConstitutionalKernel:
    """Integration tests for ConstitutionalKernel."""

    @pytest.mark.skipif(not CONSTITUTION_PATH.exists(), reason="CONSTITUTION.md not found")
    def test_kernel_initialization(self):
        """Initialize kernel with real constitutions."""
        kernel = ConstitutionalKernel(
            constitution_dir=AGENTS_DIR if AGENTS_DIR.exists() else None,
            supreme_constitution_path=CONSTITUTION_PATH,
            enable_hot_reload=False,  # Disable for tests
        )

        result = kernel.initialize()

        # Print any issues
        if result.errors:
            print(f"\nKernel init errors: {result.errors}")
        if result.warnings:
            print(f"\nKernel init warnings: {result.warnings}")

        # Should initialize successfully
        assert result.is_valid or len(result.errors) == 0

        kernel.shutdown()

    @pytest.mark.skipif(not CONSTITUTION_PATH.exists(), reason="CONSTITUTION.md not found")
    def test_kernel_get_supreme(self):
        """Kernel should provide supreme constitution."""
        kernel = create_kernel(PROJECT_ROOT, enable_hot_reload=False)

        supreme = kernel.get_supreme_constitution()
        assert supreme is not None
        assert supreme.metadata.authority_level == AuthorityLevel.SUPREME

        kernel.shutdown()

    @pytest.mark.skipif(not CONSTITUTION_PATH.exists(), reason="CONSTITUTION.md not found")
    def test_kernel_get_rules_for_agent(self):
        """Get rules for a specific agent."""
        kernel = create_kernel(PROJECT_ROOT, enable_hot_reload=False)

        rules = kernel.get_rules_for_agent("guardian")

        # Should include supreme rules
        supreme_rules = [r for r in rules if r.authority_level == AuthorityLevel.SUPREME]
        assert len(supreme_rules) > 0, "Should inherit supreme constitution rules"

        kernel.shutdown()

    @pytest.mark.skipif(not CONSTITUTION_PATH.exists(), reason="CONSTITUTION.md not found")
    def test_kernel_enforce_valid_request(self):
        """Enforce rules on a valid request."""
        kernel = create_kernel(PROJECT_ROOT, enable_hot_reload=False)

        context = RequestContext(
            request_id="test-001",
            source="user",
            destination="sage",
            intent="query.factual",
            content="What is photosynthesis?",
        )

        result = kernel.enforce(context)

        # A simple educational query should be allowed
        # (unless we have very restrictive rules)
        print(f"\nEnforcement result: allowed={result.allowed}")
        if not result.allowed:
            print(f"Reason: {result.reason}")

        kernel.shutdown()

    @pytest.mark.skipif(not CONSTITUTION_PATH.exists(), reason="CONSTITUTION.md not found")
    def test_kernel_enforce_memory_request(self):
        """Enforce rules on a memory request."""
        kernel = create_kernel(PROJECT_ROOT, enable_hot_reload=False)

        context = RequestContext(
            request_id="test-002",
            source="sage",
            destination="seshat",
            intent="memory.store",
            content="Store this user preference",
            requires_memory=True,
        )

        result = kernel.enforce(context)

        print(f"\nMemory enforcement: allowed={result.allowed}")
        if result.applicable_rules:
            print("Applicable rules:")
            for rule in result.applicable_rules[:3]:
                print(f"  - {rule.content[:60]}...")

        kernel.shutdown()


class TestConstitutionStats:
    """Print statistics about the constitution for visibility."""

    @pytest.fixture
    def parser(self):
        return ConstitutionParser()

    @pytest.mark.skipif(not CONSTITUTION_PATH.exists(), reason="CONSTITUTION.md not found")
    def test_print_constitution_stats(self, parser):
        """Print statistics about CONSTITUTION.md."""
        constitution = parser.parse_file(CONSTITUTION_PATH)

        print("\n" + "="*60)
        print("CONSTITUTION.md Statistics")
        print("="*60)
        print(f"Version: {constitution.metadata.version}")
        print(f"Scope: {constitution.metadata.scope}")
        print(f"Authority: {constitution.metadata.authority_level.name}")
        print(f"Total Sections: {constitution.section_count}")
        print(f"Total Rules: {constitution.rule_count}")
        print()

        # Rule type breakdown
        print("Rule Types:")
        for rule_type in RuleType:
            count = len(constitution.get_rules_by_type(rule_type))
            if count > 0:
                print(f"  {rule_type.name}: {count}")

        # Sample rules
        print("\nSample Mandate Rules:")
        for rule in constitution.get_rules_by_type(RuleType.MANDATE)[:3]:
            print(f"  - {rule.content[:70]}...")

        print("\nSample Prohibition Rules:")
        for rule in constitution.get_rules_by_type(RuleType.PROHIBITION)[:3]:
            print(f"  - {rule.content[:70]}...")

        print("="*60)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
