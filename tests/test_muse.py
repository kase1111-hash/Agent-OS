"""
Unit tests for Muse Creative Agent (UC-011)

Tests creative content generation, style options, constitutional compliance,
and mandatory Guardian review requirements.
"""

import pytest
from datetime import datetime

from src.agents.muse import (
    MuseAgent,
    MuseConfig,
    create_muse_agent,
    CreativeEngine,
    CreativeStyle,
    ContentType,
    CreativeMode,
    CreativeConstraints,
    CreativeResult,
    CreativeOption,
    create_creative_engine,
)
from src.agents.interface import AgentState, CapabilityType
from src.messaging.models import FlowRequest, FlowResponse, MessageStatus, create_request


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def creative_engine():
    """Create test creative engine."""
    return create_creative_engine()


@pytest.fixture
def muse_config():
    """Create test Muse config."""
    return MuseConfig(
        model="mixtral:8x7b",
        temperature=0.9,
        num_options=3,
        use_mock=True,
    )


@pytest.fixture
def muse_agent(muse_config):
    """Create test Muse agent."""
    agent = MuseAgent(muse_config)
    agent.initialize({})
    return agent


@pytest.fixture
def sample_flow_request():
    """Create a sample FlowRequest for testing."""
    return create_request(
        source="test",
        destination="muse",
        intent="creative.story",
        prompt="A knight discovers a hidden garden in the castle.",
    )


# =============================================================================
# CreativeOption Tests
# =============================================================================

class TestCreativeOption:
    """Tests for CreativeOption dataclass."""

    def test_create_option(self):
        """Test creating a creative option."""
        option = CreativeOption(
            content="Once upon a time...",
            style=CreativeStyle.NARRATIVE,
            content_type=ContentType.STORY,
            confidence=0.85,
        )

        assert option.content == "Once upon a time..."
        assert option.style == CreativeStyle.NARRATIVE
        assert option.content_type == ContentType.STORY
        assert option.confidence == 0.85

    def test_option_is_always_draft(self):
        """Test that all options are drafts."""
        option = CreativeOption(
            content="Test content",
            style=CreativeStyle.POETIC,
            content_type=ContentType.POEM,
            confidence=0.9,
        )

        assert option.is_draft() is True

    def test_format_as_draft(self):
        """Test draft formatting includes markers."""
        option = CreativeOption(
            content="A poem about the sea",
            style=CreativeStyle.LYRICAL,
            content_type=ContentType.POEM,
            confidence=0.75,
        )

        formatted = option.format_as_draft()

        assert "[DRAFT" in formatted
        assert "Human Approval" in formatted
        assert "A poem about the sea" in formatted
        assert "lyrical" in formatted
        assert "poem" in formatted

    def test_option_with_variations(self):
        """Test option with variations."""
        option = CreativeOption(
            content="Main content",
            style=CreativeStyle.WHIMSICAL,
            content_type=ContentType.CONCEPT,
            confidence=0.8,
            variations=["Variation 1", "Variation 2"],
        )

        assert len(option.variations) == 2


# =============================================================================
# CreativeResult Tests
# =============================================================================

class TestCreativeResult:
    """Tests for CreativeResult dataclass."""

    def test_create_result(self):
        """Test creating a creative result."""
        options = [
            CreativeOption(
                content="Option 1",
                style=CreativeStyle.NARRATIVE,
                content_type=ContentType.STORY,
                confidence=0.8,
            ),
            CreativeOption(
                content="Option 2",
                style=CreativeStyle.DRAMATIC,
                content_type=ContentType.STORY,
                confidence=0.7,
            ),
        ]

        result = CreativeResult(
            prompt="Write a story",
            options=options,
            mode=CreativeMode.EXPLORE,
        )

        assert result.prompt == "Write a story"
        assert len(result.options) == 2
        assert result.mode == CreativeMode.EXPLORE
        assert result.requires_review is True

    def test_get_primary_returns_highest_confidence(self):
        """Test that get_primary returns highest confidence option."""
        options = [
            CreativeOption(
                content="Low confidence",
                style=CreativeStyle.MINIMALIST,
                content_type=ContentType.POEM,
                confidence=0.5,
            ),
            CreativeOption(
                content="High confidence",
                style=CreativeStyle.POETIC,
                content_type=ContentType.POEM,
                confidence=0.9,
            ),
            CreativeOption(
                content="Medium confidence",
                style=CreativeStyle.LYRICAL,
                content_type=ContentType.POEM,
                confidence=0.7,
            ),
        ]

        result = CreativeResult(
            prompt="Test",
            options=options,
            mode=CreativeMode.EXPLORE,
        )

        primary = result.get_primary()
        assert primary.content == "High confidence"
        assert primary.confidence == 0.9

    def test_get_primary_empty_options(self):
        """Test get_primary with no options."""
        result = CreativeResult(
            prompt="Test",
            options=[],
            mode=CreativeMode.EXPLORE,
        )

        assert result.get_primary() is None

    def test_format_all_options(self):
        """Test formatting all options."""
        options = [
            CreativeOption(
                content="Story content",
                style=CreativeStyle.NARRATIVE,
                content_type=ContentType.STORY,
                confidence=0.8,
            ),
        ]

        result = CreativeResult(
            prompt="Write something",
            options=options,
            mode=CreativeMode.EXPLORE,
        )

        formatted = result.format_all_options()

        assert "Write something" in formatted
        assert "explore" in formatted
        assert "Option 1" in formatted
        assert "Story content" in formatted
        assert "drafts" in formatted.lower()


# =============================================================================
# CreativeConstraints Tests
# =============================================================================

class TestCreativeConstraints:
    """Tests for CreativeConstraints."""

    def test_default_constraints(self):
        """Test default constraint values."""
        constraints = CreativeConstraints()

        assert constraints.max_length == 2000
        assert constraints.min_length == 50
        assert constraints.target_audience == "general"

    def test_validate_content_within_bounds(self):
        """Test validation passes for valid content."""
        constraints = CreativeConstraints(
            max_length=100,
            min_length=10,
        )

        content = "This is valid content that fits within bounds."
        valid, issues = constraints.validate(content)

        assert valid is True
        assert len(issues) == 0

    def test_validate_content_too_long(self):
        """Test validation fails for too long content."""
        constraints = CreativeConstraints(max_length=20)

        content = "This content is way too long for the constraint"
        valid, issues = constraints.validate(content)

        assert valid is False
        assert any("exceeds max length" in issue for issue in issues)

    def test_validate_content_too_short(self):
        """Test validation fails for too short content."""
        constraints = CreativeConstraints(min_length=100)

        content = "Too short"
        valid, issues = constraints.validate(content)

        assert valid is False
        assert any("below min length" in issue for issue in issues)

    def test_validate_forbidden_themes(self):
        """Test validation catches forbidden themes."""
        constraints = CreativeConstraints(
            forbidden_themes=["violence", "hatred"],
        )

        content = "A story with violence in it"
        valid, issues = constraints.validate(content)

        assert valid is False
        assert any("forbidden theme" in issue for issue in issues)


# =============================================================================
# CreativeEngine Tests
# =============================================================================

class TestCreativeEngine:
    """Tests for CreativeEngine."""

    def test_create_engine(self):
        """Test creating a creative engine."""
        engine = CreativeEngine()

        assert engine.temperature >= 0.7
        assert engine.temperature <= 1.2
        assert engine.num_options == 3

    def test_create_engine_with_temperature(self):
        """Test engine respects temperature bounds."""
        # Too low - should be clamped
        engine_low = CreativeEngine(temperature=0.3)
        assert engine_low.temperature == 0.7

        # Too high - should be clamped
        engine_high = CreativeEngine(temperature=1.5)
        assert engine_high.temperature == 1.2

        # Valid temperature
        engine_valid = CreativeEngine(temperature=0.9)
        assert engine_valid.temperature == 0.9

    def test_generate_returns_result(self):
        """Test basic generation returns a result."""
        engine = CreativeEngine()

        result = engine.generate("Write about the ocean")

        assert isinstance(result, CreativeResult)
        assert result.prompt == "Write about the ocean"
        assert len(result.options) > 0
        assert result.requires_review is True

    def test_generate_story(self):
        """Test story generation."""
        engine = CreativeEngine()

        result = engine.generate_story("A hero's journey")

        assert result.options[0].content_type == ContentType.STORY

    def test_generate_poem(self):
        """Test poem generation."""
        engine = CreativeEngine()

        result = engine.generate_poem("Nature's beauty")

        assert result.options[0].content_type == ContentType.POEM

    def test_brainstorm(self):
        """Test brainstorming."""
        engine = CreativeEngine()

        result = engine.brainstorm("Renewable energy")

        assert len(result.options) > 0
        assert "Renewable energy" in result.prompt

    def test_create_scenario(self):
        """Test scenario creation."""
        engine = CreativeEngine()

        result = engine.create_scenario("First contact with aliens")

        assert result.options[0].content_type == ContentType.SCENARIO

    def test_expand_content(self):
        """Test content expansion."""
        engine = CreativeEngine()

        result = engine.expand("A simple idea", direction="more detail")

        assert result.mode == CreativeMode.EXPAND

    def test_combine_ideas(self):
        """Test idea combination."""
        engine = CreativeEngine()

        result = engine.combine_ideas(["Technology", "Nature", "Art"])

        assert result.mode == CreativeMode.COMBINE
        assert "Technology" in result.prompt

    def test_generate_with_constraints(self):
        """Test generation with constraints."""
        engine = CreativeEngine()
        constraints = CreativeConstraints(
            max_length=500,
            min_length=10,
            target_audience="children",
        )

        result = engine.generate(
            "A friendly dragon",
            constraints=constraints,
        )

        assert len(result.options) > 0

    def test_generate_multiple_styles(self):
        """Test generation with multiple styles."""
        engine = CreativeEngine(num_options=3)

        result = engine.generate(
            "The meaning of life",
            styles=[CreativeStyle.POETIC, CreativeStyle.MINIMALIST, CreativeStyle.FORMAL],
        )

        assert len(result.options) == 3

    def test_factory_function(self):
        """Test factory function."""
        engine = create_creative_engine(temperature=0.85, num_options=2)

        assert engine.temperature == 0.85
        assert engine.num_options == 2


# =============================================================================
# MuseConfig Tests
# =============================================================================

class TestMuseConfig:
    """Tests for MuseConfig."""

    def test_default_config(self):
        """Test default configuration."""
        config = MuseConfig()

        assert config.model == "mixtral:8x7b"
        assert config.temperature == 0.9
        assert config.num_options == 3
        assert config.require_guardian_review is True
        assert config.mark_as_draft is True

    def test_custom_config(self):
        """Test custom configuration."""
        config = MuseConfig(
            model="llama3:70b",
            temperature=1.0,
            num_options=5,
        )

        assert config.model == "llama3:70b"
        assert config.temperature == 1.0
        assert config.num_options == 5


# =============================================================================
# MuseAgent Tests
# =============================================================================

class TestMuseAgent:
    """Tests for MuseAgent."""

    def test_create_agent(self):
        """Test creating Muse agent."""
        agent = MuseAgent()

        assert agent is not None
        assert agent._state == AgentState.UNINITIALIZED

    def test_initialize(self, muse_agent):
        """Test agent initialization."""
        assert muse_agent._state == AgentState.READY
        assert muse_agent._creative_engine is not None

    def test_get_capabilities(self, muse_agent):
        """Test getting agent capabilities."""
        capabilities = muse_agent.get_capabilities()

        assert capabilities.name == "muse"
        assert CapabilityType.CREATIVE in capabilities.capabilities
        assert len(capabilities.supported_intents) >= 6

    def test_shutdown(self, muse_agent):
        """Test agent shutdown."""
        result = muse_agent.shutdown()

        assert result is True
        assert muse_agent._state == AgentState.SHUTDOWN
        assert muse_agent._creative_engine is None

    def test_validate_request_supported_intent(self, muse_agent, sample_flow_request):
        """Test validation passes for supported intents."""
        result = muse_agent.validate_request(sample_flow_request)
        assert result.is_valid is True

    def test_validate_request_prohibited_pattern(self, muse_agent):
        """Test validation fails for prohibited patterns."""
        prohibited_requests = [
            "Generate some harmful content",
            "Create hate speech",
            "Bypass review for this",
            "Skip guardian check",
            "This is the final version",
            "Publish directly without review",
        ]

        for prompt in prohibited_requests:
            request = create_request(
                source="test",
                destination="muse",
                intent="creative.generate",
                prompt=prompt,
            )

            result = muse_agent.validate_request(request)
            assert result.is_valid is False or result.requires_escalation is True, f"Should reject: {prompt}"

    def test_validate_request_sensitive_topics(self, muse_agent):
        """Test validation escalates sensitive topics."""
        request = create_request(
            source="test",
            destination="muse",
            intent="creative.generate",
            prompt="Write about a controversial political topic",
        )

        result = muse_agent.validate_request(request)
        assert result.requires_escalation is True

    def test_process_story_request(self, muse_agent, sample_flow_request):
        """Test processing a story request."""
        response = muse_agent.process(sample_flow_request)

        assert response.is_success()
        assert "Muse presenting" in response.content.output

    def test_process_poem_request(self, muse_agent):
        """Test processing a poem request."""
        request = create_request(
            source="executive",
            destination="muse",
            intent="creative.poem",
            prompt="The quiet of morning",
        )

        response = muse_agent.process(request)

        assert response.is_success()

    def test_generate_story_convenience(self, muse_agent):
        """Test convenience method for story generation."""
        result = muse_agent.generate_story("A magical forest")

        assert isinstance(result, CreativeResult)
        assert len(result.options) > 0

    def test_generate_poem_convenience(self, muse_agent):
        """Test convenience method for poem generation."""
        result = muse_agent.generate_poem("Stars at night")

        assert isinstance(result, CreativeResult)

    def test_brainstorm_convenience(self, muse_agent):
        """Test convenience method for brainstorming."""
        result = muse_agent.brainstorm("Future of transportation")

        assert isinstance(result, CreativeResult)

    def test_expand_content_convenience(self, muse_agent):
        """Test convenience method for content expansion."""
        result = muse_agent.expand_content(
            "A small idea",
            direction="explore implications",
        )

        assert isinstance(result, CreativeResult)
        assert result.mode == CreativeMode.EXPAND


# =============================================================================
# Factory Function Tests
# =============================================================================

class TestCreateMuseAgent:
    """Tests for create_muse_agent factory function."""

    def test_create_with_mock(self):
        """Test creating agent with mock LLM."""
        agent = create_muse_agent(use_mock=True)

        assert agent is not None
        assert agent._state == AgentState.READY

    def test_create_with_custom_config(self):
        """Test creating agent with custom config."""
        config = MuseConfig(
            temperature=1.0,
            num_options=5,
        )

        agent = create_muse_agent(use_mock=True, temperature=1.0, num_options=5)

        assert agent._muse_config.temperature == 1.0
        assert agent._muse_config.num_options == 5


# =============================================================================
# Content Type Determination Tests
# =============================================================================

class TestContentTypeDetermination:
    """Tests for content type inference."""

    def test_determine_story_from_intent(self, muse_agent):
        """Test story detection from intent."""
        content_type = muse_agent._determine_content_type(
            "creative.story",
            "anything",
        )

        assert content_type == ContentType.STORY

    def test_determine_poem_from_content(self, muse_agent):
        """Test poem detection from content."""
        content_type = muse_agent._determine_content_type(
            "creative.generate",
            "Write a poem about love",
        )

        assert content_type == ContentType.POEM

    def test_determine_scenario_from_intent(self, muse_agent):
        """Test scenario detection."""
        content_type = muse_agent._determine_content_type(
            "creative.scenario",
            "describe a situation",
        )

        assert content_type == ContentType.SCENARIO

    def test_determine_dialogue_from_content(self, muse_agent):
        """Test dialogue detection."""
        content_type = muse_agent._determine_content_type(
            "creative.generate",
            "Create a conversation between two people",
        )

        assert content_type == ContentType.DIALOGUE


# =============================================================================
# Style Determination Tests
# =============================================================================

class TestStyleDetermination:
    """Tests for style inference."""

    def test_determine_style_from_metadata(self, muse_agent):
        """Test style from explicit metadata."""
        style = muse_agent._determine_style(
            {"style": "whimsical"},
            "any content",
        )

        assert style == CreativeStyle.WHIMSICAL

    def test_determine_formal_from_content(self, muse_agent):
        """Test formal style inference."""
        style = muse_agent._determine_style(
            {},
            "Write in a formal professional manner",
        )

        assert style == CreativeStyle.FORMAL

    def test_determine_whimsical_from_content(self, muse_agent):
        """Test whimsical style inference."""
        style = muse_agent._determine_style(
            {},
            "Make it fun and playful",
        )

        assert style == CreativeStyle.WHIMSICAL

    def test_determine_style_none_when_unclear(self, muse_agent):
        """Test None returned when style unclear."""
        style = muse_agent._determine_style(
            {},
            "Just write something",
        )

        assert style is None


# =============================================================================
# Integration Tests
# =============================================================================

class TestMuseIntegration:
    """Integration tests for Muse agent."""

    def test_full_creative_workflow(self):
        """Test complete creative workflow."""
        # Create agent
        agent = create_muse_agent(use_mock=True)

        # Generate initial story
        story_result = agent.generate_story("A time traveler's dilemma")
        assert story_result.get_primary() is not None

        # Process a request
        request = create_request(
            source="executive",
            destination="muse",
            intent="creative.generate",
            prompt="Expand on the time travel concept",
        )

        response = agent.process(request)
        assert response.is_success()

        # Brainstorm related ideas
        ideas = agent.brainstorm("Paradoxes in time travel")
        assert len(ideas.options) > 0

        # Shutdown
        agent.shutdown()
        assert agent._state == AgentState.SHUTDOWN

    def test_draft_marking_enforced(self):
        """Test that all outputs are marked as drafts."""
        agent = create_muse_agent(use_mock=True)

        # Generate various content types
        story = agent.generate_story("Test")
        poem = agent.generate_poem("Test")
        ideas = agent.brainstorm("Test")

        # All should be drafts
        for result in [story, poem, ideas]:
            for option in result.options:
                assert option.is_draft() is True

    def test_guardian_review_required(self):
        """Test that Guardian review is always required."""
        agent = create_muse_agent(use_mock=True)
        capabilities = agent.get_capabilities()

        assert capabilities.metadata.get("require_guardian_review") is True

    def test_multiple_options_generated(self):
        """Test that multiple options are generated in explore mode."""
        config = MuseConfig(num_options=3, use_mock=True)
        agent = MuseAgent(config)
        agent.initialize({})

        result = agent.generate_story("Adventure awaits")

        # Should have options based on config
        assert len(result.options) >= 1


# =============================================================================
# Enum Tests
# =============================================================================

class TestEnums:
    """Tests for enum values."""

    def test_creative_styles(self):
        """Test all creative styles exist."""
        styles = list(CreativeStyle)

        assert len(styles) == 8
        assert CreativeStyle.NARRATIVE in styles
        assert CreativeStyle.POETIC in styles
        assert CreativeStyle.WHIMSICAL in styles

    def test_content_types(self):
        """Test all content types exist."""
        types = list(ContentType)

        assert len(types) == 8
        assert ContentType.STORY in types
        assert ContentType.POEM in types
        assert ContentType.BRAINSTORM in types

    def test_creative_modes(self):
        """Test all creative modes exist."""
        modes = list(CreativeMode)

        assert len(modes) == 5
        assert CreativeMode.EXPLORE in modes
        assert CreativeMode.REFINE in modes
        assert CreativeMode.COMBINE in modes
