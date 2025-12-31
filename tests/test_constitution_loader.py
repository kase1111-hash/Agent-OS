"""
Tests for Constitution Loader

Verifies that:
- Constitution loader finds and loads CONSTITUTION.md
- Agent-specific constitutions are loaded correctly
- Agents only receive their own constitution (not other agents')
- System prompts are properly combined with constitutional context
"""

import pytest
from pathlib import Path

from src.agents.constitution_loader import (
    ConstitutionLoader,
    ConstitutionalContext,
    get_constitution_loader,
    load_constitutional_context,
    build_system_prompt_with_constitution,
)


# Find project root
PROJECT_ROOT = Path(__file__).parent.parent


class TestConstitutionLoader:
    """Tests for ConstitutionLoader class."""

    def test_init_with_project_root(self):
        """Test initialization with explicit project root."""
        loader = ConstitutionLoader(project_root=PROJECT_ROOT)
        assert loader.project_root == PROJECT_ROOT

    def test_auto_detect_project_root(self):
        """Test auto-detection of project root."""
        loader = ConstitutionLoader()
        assert loader.project_root.exists()
        # Should find CONSTITUTION.md
        assert (loader.project_root / "CONSTITUTION.md").exists()

    def test_load_supreme_constitution(self):
        """Test loading the supreme CONSTITUTION.md."""
        loader = ConstitutionLoader(project_root=PROJECT_ROOT)
        context = loader.load_for_agent("test_agent", include_supreme=True)

        assert context.has_supreme
        # Check for constitutional keywords (case-insensitive)
        constitution_lower = context.supreme_constitution.lower()
        assert "must" in constitution_lower or "shall" in constitution_lower

    def test_load_agent_specific_constitution(self):
        """Test loading an agent-specific constitution."""
        loader = ConstitutionLoader(project_root=PROJECT_ROOT)

        # Load for muse agent (has constitution.md)
        context = loader.load_for_agent("muse", include_supreme=True)

        assert context.agent_name == "muse"
        assert context.has_agent_specific
        assert "Muse" in context.agent_constitution

    def test_agent_only_gets_own_constitution(self):
        """Test that an agent only receives its own constitution."""
        loader = ConstitutionLoader(project_root=PROJECT_ROOT)

        # Load muse constitution
        muse_context = loader.load_for_agent("muse")

        # Load sage constitution
        sage_context = loader.load_for_agent("sage")

        # Muse should not have sage's specific rules
        assert "Sage" not in muse_context.agent_constitution
        assert "Muse" in muse_context.agent_constitution

        # Sage should not have muse's specific rules
        assert "Muse" not in sage_context.agent_constitution
        assert "Sage" in sage_context.agent_constitution

    def test_combined_prompt_structure(self):
        """Test the structure of the combined prompt."""
        loader = ConstitutionLoader(project_root=PROJECT_ROOT)
        context = loader.load_for_agent("muse", include_supreme=True)

        # Should have clear delineation
        assert "CONSTITUTIONAL GOVERNANCE" in context.combined_prompt
        assert "SUPREME CONSTITUTION" in context.combined_prompt
        assert "AGENT-SPECIFIC CONSTITUTION" in context.combined_prompt
        assert "END OF CONSTITUTIONAL GOVERNANCE" in context.combined_prompt

    def test_build_system_prompt_with_constitution(self):
        """Test building a complete system prompt."""
        loader = ConstitutionLoader(project_root=PROJECT_ROOT)

        base_prompt = "You are a test agent."
        full_prompt = loader.get_system_prompt_with_constitution(
            agent_name="muse",
            base_prompt=base_prompt,
            include_supreme=True,
        )

        # Should include constitutional context
        assert "CONSTITUTIONAL GOVERNANCE" in full_prompt
        # Should include base prompt
        assert "You are a test agent." in full_prompt
        # Constitutional context should come first
        assert full_prompt.index("CONSTITUTIONAL GOVERNANCE") < full_prompt.index("You are a test agent.")

    def test_caching(self):
        """Test that constitution loading is cached."""
        loader = ConstitutionLoader(project_root=PROJECT_ROOT)

        # First load
        context1 = loader.load_for_agent("muse")

        # Second load (should be cached)
        context2 = loader.load_for_agent("muse")

        # Should be the same object (cached)
        assert context1 is context2

    def test_force_reload(self):
        """Test force reload bypasses cache."""
        loader = ConstitutionLoader(project_root=PROJECT_ROOT)

        # First load
        context1 = loader.load_for_agent("muse")

        # Force reload
        context2 = loader.load_for_agent("muse", force_reload=True)

        # Should be different objects
        assert context1 is not context2

    def test_clear_cache(self):
        """Test cache clearing."""
        loader = ConstitutionLoader(project_root=PROJECT_ROOT)

        # Load something
        loader.load_for_agent("muse")
        assert len(loader._cache) > 0

        # Clear cache
        loader.clear_cache()
        assert len(loader._cache) == 0

    def test_nonexistent_agent(self):
        """Test loading for a nonexistent agent."""
        loader = ConstitutionLoader(project_root=PROJECT_ROOT)

        context = loader.load_for_agent("nonexistent_agent_xyz")

        # Should still have supreme constitution
        assert context.has_supreme
        # But no agent-specific
        assert not context.has_agent_specific

    def test_exclude_supreme(self):
        """Test loading without supreme constitution."""
        loader = ConstitutionLoader(project_root=PROJECT_ROOT)

        context = loader.load_for_agent("muse", include_supreme=False)

        # Should not have supreme
        assert not context.has_supreme
        # But should have agent-specific
        assert context.has_agent_specific


class TestConvenienceFunctions:
    """Tests for module-level convenience functions."""

    def test_get_constitution_loader(self):
        """Test getting the global loader."""
        loader1 = get_constitution_loader()
        loader2 = get_constitution_loader()

        # Should return the same instance
        assert loader1 is loader2

    def test_load_constitutional_context(self):
        """Test the convenience function for loading context."""
        context = load_constitutional_context("muse")

        assert isinstance(context, ConstitutionalContext)
        assert context.agent_name == "muse"

    def test_build_system_prompt_with_constitution_function(self):
        """Test the convenience function for building prompts."""
        prompt = build_system_prompt_with_constitution(
            agent_name="muse",
            base_prompt="Test prompt",
        )

        assert "CONSTITUTIONAL GOVERNANCE" in prompt
        assert "Test prompt" in prompt


class TestConstitutionalContext:
    """Tests for ConstitutionalContext dataclass."""

    def test_has_supreme_property(self):
        """Test has_supreme property."""
        context = ConstitutionalContext(
            agent_name="test",
            supreme_constitution="Some content",
            agent_constitution="",
            combined_prompt="",
        )
        assert context.has_supreme

        empty_context = ConstitutionalContext(
            agent_name="test",
            supreme_constitution="",
            agent_constitution="",
            combined_prompt="",
        )
        assert not empty_context.has_supreme

    def test_has_agent_specific_property(self):
        """Test has_agent_specific property."""
        context = ConstitutionalContext(
            agent_name="test",
            supreme_constitution="",
            agent_constitution="Agent rules",
            combined_prompt="",
        )
        assert context.has_agent_specific


class TestIntegrationWithAgents:
    """Integration tests with actual agents."""

    def test_muse_agent_constitution_loading(self):
        """Test that Muse agent can load its constitution."""
        # This is a simulated test - in real usage the agent would call
        # get_full_system_prompt() during initialization

        loader = get_constitution_loader()
        context = loader.load_for_agent("muse")

        # Verify muse-specific content is loaded
        assert "creative" in context.agent_constitution.lower()
        assert "Muse" in context.agent_constitution

    def test_all_agents_have_constitutions(self):
        """Verify all documented agents have constitution files."""
        agents_with_constitutions = [
            "muse", "sage", "quill", "guardian",
            "executive", "seshat", "planner", "researcher"
        ]

        loader = get_constitution_loader()

        for agent_name in agents_with_constitutions:
            context = loader.load_for_agent(agent_name)
            assert context.has_supreme, f"{agent_name} missing supreme constitution"
            assert context.has_agent_specific, f"{agent_name} missing agent-specific constitution"

    def test_no_cross_contamination(self):
        """Ensure no agent receives another agent's constitution content."""
        loader = get_constitution_loader()

        agents = ["muse", "sage", "quill", "guardian"]
        agent_contexts = {
            name: loader.load_for_agent(name, force_reload=True)
            for name in agents
        }

        for agent_name, context in agent_contexts.items():
            agent_constitution = context.agent_constitution.lower()

            for other_name in agents:
                if other_name != agent_name:
                    # The other agent's name shouldn't appear prominently
                    # in this agent's constitution
                    constitution_header = f"constitution of the {other_name} agent"
                    assert constitution_header not in agent_constitution, \
                        f"{agent_name}'s constitution contains {other_name}'s constitution"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
