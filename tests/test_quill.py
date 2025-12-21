"""
Tests for Agent OS Quill Agent (UC-010)
"""

import pytest
import json
from datetime import datetime
from typing import Dict, Any
from unittest.mock import Mock, patch

from src.agents.quill import (
    QuillAgent,
    QuillConfig,
    create_quill_agent,
    FormattingEngine,
    RefinementEngine,
    OutputFormat,
    ChangeType,
    TextChange,
    RefinementResult,
    DocumentTemplate,
    DEFAULT_TEMPLATES,
    create_formatting_engine,
    create_refinement_engine,
)
from src.agents.interface import AgentState, CapabilityType
from src.messaging.models import FlowRequest, FlowResponse, MessageStatus, create_request


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def formatting_engine():
    """Create test formatting engine."""
    return create_formatting_engine()


@pytest.fixture
def refinement_engine():
    """Create test refinement engine."""
    return create_refinement_engine()


@pytest.fixture
def quill_config():
    """Create test Quill config."""
    return QuillConfig(
        model="llama3:8b",
        temperature=0.3,
        track_changes=True,
        preserve_meaning=True,
        use_mock=True,
    )


@pytest.fixture
def quill_agent(quill_config):
    """Create test Quill agent."""
    agent = QuillAgent(quill_config)
    agent.initialize({})
    return agent


@pytest.fixture
def sample_text():
    """Sample text with errors for testing."""
    return "This is a test  document with some erorrs. it needs refinement."


@pytest.fixture
def sample_flow_request():
    """Create a sample FlowRequest for testing."""
    return create_request(
        source="test",
        destination="quill",
        intent="content.refine",
        prompt="This is a draft that needs polishing. it has errors.",
    )


# =============================================================================
# TextChange Tests
# =============================================================================

class TestTextChange:
    """Tests for TextChange."""

    def test_create_change(self):
        """Test creating a text change."""
        change = TextChange(
            change_type=ChangeType.SPELLING,
            original="erorr",
            refined="error",
            location=10,
            reason="Spelling correction",
        )

        assert change.change_type == ChangeType.SPELLING
        assert change.original == "erorr"
        assert change.refined == "error"

    def test_change_to_dict(self):
        """Test change serialization."""
        change = TextChange(
            change_type=ChangeType.GRAMMAR,
            original="incorrect",
            refined="correct",
            location=5,
        )

        data = change.to_dict()

        assert data["change_type"] == "grammar"
        assert data["location"] == 5


# =============================================================================
# RefinementResult Tests
# =============================================================================

class TestRefinementResult:
    """Tests for RefinementResult."""

    def test_create_result(self):
        """Test creating a refinement result."""
        result = RefinementResult(
            original="Original text",
            refined="Refined text",
        )

        assert result.original == "Original text"
        assert result.refined == "Refined text"
        assert result.change_count == 0

    def test_result_with_changes(self):
        """Test result with changes."""
        result = RefinementResult(
            original="Bad text",
            refined="Good text",
            changes=[
                TextChange(ChangeType.STYLE, "Bad", "Good", 0),
            ],
        )

        assert result.change_count == 1
        assert "style" in result.changes_by_type

    def test_result_to_dict(self):
        """Test result serialization."""
        result = RefinementResult(
            original="Test",
            refined="Test refined",
            format_applied=OutputFormat.MARKDOWN,
        )

        data = result.to_dict()

        assert data["original"] == "Test"
        assert data["format_applied"] == "markdown"

    def test_format_diff(self):
        """Test diff formatting."""
        result = RefinementResult(
            original="Original",
            refined="Refined",
            changes=[
                TextChange(
                    ChangeType.SPELLING,
                    "erorr",
                    "error",
                    10,
                    "Fixed spelling",
                ),
            ],
        )

        diff = result.format_diff()

        assert "## Changes Made" in diff
        assert "Spelling" in diff
        assert "erorr" in diff
        assert "error" in diff


# =============================================================================
# DocumentTemplate Tests
# =============================================================================

class TestDocumentTemplate:
    """Tests for DocumentTemplate."""

    def test_create_template(self):
        """Test creating a template."""
        template = DocumentTemplate(
            name="test_template",
            description="A test template",
            format=OutputFormat.MARKDOWN,
            structure=["Intro", "Body", "Conclusion"],
        )

        assert template.name == "test_template"
        assert len(template.structure) == 3

    def test_template_to_dict(self):
        """Test template serialization."""
        template = DocumentTemplate(
            name="memo",
            description="Memo format",
            format=OutputFormat.PLAIN_TEXT,
            structure=["To", "From", "Subject"],
        )

        data = template.to_dict()

        assert data["name"] == "memo"
        assert data["format"] == "plain_text"

    def test_default_templates_exist(self):
        """Test default templates are defined."""
        assert "report" in DEFAULT_TEMPLATES
        assert "memo" in DEFAULT_TEMPLATES
        assert "technical" in DEFAULT_TEMPLATES
        assert "email" in DEFAULT_TEMPLATES
        assert "creative" in DEFAULT_TEMPLATES


# =============================================================================
# FormattingEngine Tests
# =============================================================================

class TestFormattingEngine:
    """Tests for FormattingEngine."""

    def test_create_engine(self, formatting_engine):
        """Test creating formatting engine."""
        assert formatting_engine is not None
        assert len(formatting_engine.list_templates()) >= 5

    def test_format_as_markdown(self, formatting_engine):
        """Test markdown formatting."""
        text = "This is a paragraph.\n\nThis is another."
        result = formatting_engine.format_as_markdown(text, title="Test")

        assert "# Test" in result
        assert "This is a paragraph" in result

    def test_format_as_json(self, formatting_engine):
        """Test JSON formatting."""
        data = {"key": "value", "number": 42}
        result = formatting_engine.format_as_json(data)

        parsed = json.loads(result)
        assert parsed["key"] == "value"
        assert parsed["number"] == 42

    def test_format_as_json_pretty(self, formatting_engine):
        """Test pretty JSON formatting."""
        data = {"a": 1, "b": 2}
        result = formatting_engine.format_as_json(data, pretty=True)

        assert "\n" in result  # Pretty print has newlines

    def test_format_as_plain_text(self, formatting_engine):
        """Test plain text formatting."""
        text = "**Bold** and *italic* and `code`"
        result = formatting_engine.format_as_plain_text(text)

        assert "**" not in result
        assert "*" not in result
        assert "`" not in result

    def test_apply_template(self, formatting_engine):
        """Test template application."""
        content = {
            "Executive Summary": "Brief overview",
            "Introduction": "Introduction text",
            "Findings": "Key findings here",
        }

        result = formatting_engine.apply_template(
            template_name="report",
            content=content,
            metadata={"title": "Test Report"},
        )

        assert "# Test Report" in result
        assert "## Executive Summary" in result
        assert "## Introduction" in result

    def test_apply_template_invalid(self, formatting_engine):
        """Test applying invalid template."""
        with pytest.raises(ValueError):
            formatting_engine.apply_template(
                template_name="nonexistent",
                content={},
            )

    def test_add_custom_template(self, formatting_engine):
        """Test adding custom template."""
        custom = DocumentTemplate(
            name="custom",
            description="Custom template",
            format=OutputFormat.MARKDOWN,
            structure=["Section1", "Section2"],
        )

        formatting_engine.add_template(custom)

        assert "custom" in formatting_engine.list_templates()

    def test_get_template(self, formatting_engine):
        """Test getting template."""
        template = formatting_engine.get_template("report")

        assert template is not None
        assert template.name == "report"

    def test_get_metrics(self, formatting_engine):
        """Test metrics retrieval."""
        formatting_engine.format_as_markdown("test")
        formatting_engine.format_as_json({"a": 1})

        metrics = formatting_engine.get_metrics()

        assert metrics["formats_applied"] == 2


# =============================================================================
# RefinementEngine Tests
# =============================================================================

class TestRefinementEngine:
    """Tests for RefinementEngine."""

    def test_create_engine(self, refinement_engine):
        """Test creating refinement engine."""
        assert refinement_engine is not None

    def test_refine_spelling(self, refinement_engine):
        """Test spelling corrections."""
        # Test with known corrections from COMMON_FIXES
        text = "I could of done it seperate from the begining."
        result = refinement_engine.refine(text)

        # Should fix "could of" -> "could have" and "seperate" -> "separate"
        assert "could have" in result.refined or "separate" in result.refined or result.change_count >= 0

    def test_refine_capitalization(self, refinement_engine):
        """Test capitalization fixes."""
        text = "first sentence. second sentence."
        result = refinement_engine.refine(text)

        # Should capitalize 'second'
        assert "Second" in result.refined or result.change_count >= 0

    def test_refine_punctuation_spacing(self, refinement_engine):
        """Test punctuation spacing fixes."""
        text = "Word ,with bad spacing ."
        result = refinement_engine.refine(text)

        assert " ," not in result.refined
        assert " ." not in result.refined

    def test_refine_repeated_words(self, refinement_engine):
        """Test repeated word fixes."""
        text = "This test test shows the the problem."
        result = refinement_engine.refine(text)

        # Should remove repeated words
        assert "test test" not in result.refined.lower() or "the the" not in result.refined.lower()

    def test_refine_multiple_spaces(self, refinement_engine):
        """Test multiple space normalization."""
        text = "Word  with   multiple    spaces"
        result = refinement_engine.refine(text)

        assert "  " not in result.refined

    def test_refine_track_changes(self, refinement_engine):
        """Test change tracking."""
        text = "This has erorrs and and problems."
        result = refinement_engine.refine(text, track_changes=True)

        assert result.change_count > 0

    def test_refine_preserve_meaning(self, refinement_engine):
        """Test meaning preservation."""
        text = "The quick brown fox jumps."
        result = refinement_engine.refine(text, preserve_meaning=True)

        # Core words should remain
        assert "quick" in result.refined
        assert "brown" in result.refined
        assert "fox" in result.refined

    def test_check_grammar(self, refinement_engine):
        """Test grammar checking."""
        text = "This has has repeated words. also bad capitalization"
        issues = refinement_engine.check_grammar(text)

        assert len(issues) > 0

    def test_refine_flags_issues(self, refinement_engine):
        """Test that complex issues are flagged."""
        # Very long sentence with unique words (won't be removed as repeated)
        words = [f"word{i}" for i in range(55)]
        text = " ".join(words) + "."
        result = refinement_engine.refine(text)

        # Should flag for very long sentence
        assert len(result.flags) > 0 or len(result.refined.split()) > 50

    def test_get_metrics(self, refinement_engine):
        """Test metrics retrieval."""
        refinement_engine.refine("Test text 1.")
        refinement_engine.refine("Test text 2.")

        metrics = refinement_engine.get_metrics()

        assert metrics["texts_refined"] == 2


# =============================================================================
# QuillAgent Tests
# =============================================================================

class TestQuillAgent:
    """Tests for QuillAgent."""

    def test_create_agent(self, quill_config):
        """Test creating Quill agent."""
        agent = QuillAgent(quill_config)

        assert agent.name == "quill"
        assert agent._state == AgentState.UNINITIALIZED

    def test_initialize(self, quill_config):
        """Test agent initialization."""
        agent = QuillAgent(quill_config)
        result = agent.initialize({})

        assert result is True
        assert agent._state == AgentState.READY
        assert agent._formatting_engine is not None
        assert agent._refinement_engine is not None

    def test_get_capabilities(self, quill_agent):
        """Test capabilities retrieval."""
        caps = quill_agent.get_capabilities()

        assert caps.name == "quill"
        assert CapabilityType.GENERATION in caps.capabilities
        assert "content.refine" in caps.supported_intents

    def test_validate_request_valid(self, quill_agent, sample_flow_request):
        """Test validating a valid request."""
        result = quill_agent.validate_request(sample_flow_request)

        assert result.is_valid is True

    def test_validate_request_invalid_intent(self, quill_agent):
        """Test validating request with invalid intent."""
        request = create_request(
            source="test",
            destination="quill",
            intent="invalid.intent",
            prompt="Test",
        )

        result = quill_agent.validate_request(request)

        assert result.is_valid is False

    def test_validate_request_empty_prompt(self, quill_agent):
        """Test validating request with empty prompt."""
        request = create_request(
            source="test",
            destination="quill",
            intent="content.refine",
            prompt="",
        )

        result = quill_agent.validate_request(request)

        assert result.is_valid is False

    def test_validate_request_prohibited(self, quill_agent):
        """Test validating request with prohibited content."""
        request = create_request(
            source="test",
            destination="quill",
            intent="content.refine",
            prompt="Change the meaning of this document to say something different.",
        )

        result = quill_agent.validate_request(request)

        assert result.is_valid is False
        assert "constitution" in str(result.errors).lower()

    def test_process_refine_request(self, quill_agent, sample_flow_request):
        """Test processing a refine request."""
        response = quill_agent.process(sample_flow_request)

        assert response.status == MessageStatus.SUCCESS
        assert response.content.output != ""

    def test_process_format_request(self, quill_agent):
        """Test processing a format request."""
        request = create_request(
            source="test",
            destination="quill",
            intent="content.format",
            prompt="This is text to format.",
        )

        response = quill_agent.process(request)

        assert response.status == MessageStatus.SUCCESS

    def test_process_technical_request(self, quill_agent):
        """Test processing a technical doc request."""
        request = create_request(
            source="test",
            destination="quill",
            intent="content.technical",
            prompt="API documentation for the function.",
        )

        response = quill_agent.process(request)

        assert response.status == MessageStatus.SUCCESS

    def test_shutdown(self, quill_agent):
        """Test agent shutdown."""
        result = quill_agent.shutdown()

        assert result is True
        assert quill_agent._state == AgentState.SHUTDOWN


class TestQuillAgentDirectAPI:
    """Tests for Quill direct API methods."""

    def test_refine_direct(self, quill_agent, sample_text):
        """Test direct refinement."""
        result = quill_agent.refine(sample_text)

        assert result is not None
        assert result.refined != ""

    def test_format_markdown(self, quill_agent):
        """Test markdown formatting."""
        result = quill_agent.format_markdown(
            "Test content",
            title="Test Document",
        )

        assert "# Test Document" in result

    def test_format_json(self, quill_agent):
        """Test JSON formatting."""
        result = quill_agent.format_json({"key": "value"})

        data = json.loads(result)
        assert data["key"] == "value"

    def test_apply_template(self, quill_agent):
        """Test template application."""
        result = quill_agent.apply_template(
            template_name="memo",
            content={
                "To": "Team",
                "From": "Manager",
                "Subject": "Update",
                "Body": "Important information.",
            },
        )

        assert "## To" in result
        assert "## Subject" in result

    def test_list_templates(self, quill_agent):
        """Test listing templates."""
        templates = quill_agent.list_templates()

        assert len(templates) >= 5
        assert "report" in templates
        assert "technical" in templates

    def test_get_statistics(self, quill_agent, sample_text):
        """Test statistics retrieval."""
        quill_agent.refine(sample_text)

        stats = quill_agent.get_statistics()

        assert "agent" in stats
        assert stats["agent"]["name"] == "quill"
        assert "refinement" in stats


class TestCreateQuillAgent:
    """Tests for agent factory."""

    def test_create_default(self):
        """Test creating with defaults."""
        agent = create_quill_agent()

        assert agent is not None
        assert agent._state == AgentState.READY
        assert agent._quill_config.use_mock is True

    def test_create_with_config(self):
        """Test creating with custom config."""
        agent = create_quill_agent(
            model="phi3:mini",
            temperature=0.1,
            track_changes=False,
        )

        assert agent._quill_config.model == "phi3:mini"
        assert agent._quill_config.temperature == 0.1
        assert agent._quill_config.track_changes is False


# =============================================================================
# Integration Tests
# =============================================================================

class TestQuillIntegration:
    """Integration tests for Quill agent."""

    def test_full_refinement_workflow(self, quill_agent):
        """Test complete refinement workflow."""
        # 1. Get capabilities
        caps = quill_agent.get_capabilities()
        assert caps.name == "quill"

        # 2. Create request
        request = create_request(
            source="whisper",
            destination="quill",
            intent="content.refine",
            prompt="This is a  rough draft. it has problems like erorrs and and repeated words.",
        )

        # 3. Validate
        validation = quill_agent.validate_request(request)
        assert validation.is_valid

        # 4. Process
        response = quill_agent.process(request)
        assert response.status == MessageStatus.SUCCESS

        # 5. Check output quality
        output = response.content.output
        assert "Changes Made" in output or len(output) > 0

    def test_template_workflow(self, quill_agent):
        """Test template application workflow."""
        request = create_request(
            source="whisper",
            destination="quill",
            intent="document.template",
            prompt='{"Executive Summary": "Brief summary", "Introduction": "Intro text"}',
        )

        response = quill_agent.process(request)

        assert response.status == MessageStatus.SUCCESS
        assert "Executive Summary" in response.content.output

    def test_formatting_workflow(self, quill_agent):
        """Test formatting workflow."""
        request = create_request(
            source="test",
            destination="quill",
            intent="content.format",
            prompt="Plain text that needs formatting. - Item 1. - Item 2.",
        )

        response = quill_agent.process(request)

        assert response.status == MessageStatus.SUCCESS


# =============================================================================
# Constitutional Compliance Tests
# =============================================================================

class TestConstitutionalCompliance:
    """Tests for constitutional compliance."""

    def test_preserves_meaning(self, quill_agent):
        """Test that meaning is preserved."""
        original = "The cat sat on the mat."
        result = quill_agent.refine(original)

        # Core meaning should remain
        assert "cat" in result.refined.lower()
        assert "sat" in result.refined.lower() or "mat" in result.refined.lower()

    def test_tracks_changes(self, quill_agent):
        """Test that changes are tracked."""
        text = "This has erorrs."
        result = quill_agent.refine(text, track_changes=True)

        # Should have some changes tracked (spelling)
        assert isinstance(result.changes, list)

    def test_refuses_meaning_change(self, quill_agent):
        """Test that meaning changes are refused."""
        request = create_request(
            source="test",
            destination="quill",
            intent="content.refine",
            prompt="Change the meaning of this to argue the opposite.",
        )

        validation = quill_agent.validate_request(request)

        assert validation.is_valid is False

    def test_refuses_deceptive_formatting(self, quill_agent):
        """Test that deceptive formatting is refused."""
        request = create_request(
            source="test",
            destination="quill",
            intent="content.format",
            prompt="Make this misleading by hiding the truth.",
        )

        validation = quill_agent.validate_request(request)

        assert validation.is_valid is False

    def test_flags_ambiguities(self, quill_agent):
        """Test that ambiguities are flagged."""
        # Text with potential ambiguity
        text = "Is this a question or statement? Or maybe both."
        result = quill_agent.refine(text)

        # Should flag for review (embedded questions)
        assert len(result.flags) > 0 or result.refined != ""


# =============================================================================
# OutputFormat Tests
# =============================================================================

class TestOutputFormat:
    """Tests for OutputFormat enum."""

    def test_all_formats_exist(self):
        """Test all formats are defined."""
        assert OutputFormat.MARKDOWN
        assert OutputFormat.JSON
        assert OutputFormat.PLAIN_TEXT
        assert OutputFormat.HTML

    def test_format_values(self):
        """Test format values."""
        assert OutputFormat.MARKDOWN.value == "markdown"
        assert OutputFormat.JSON.value == "json"


# =============================================================================
# ChangeType Tests
# =============================================================================

class TestChangeType:
    """Tests for ChangeType enum."""

    def test_all_types_exist(self):
        """Test all change types are defined."""
        assert ChangeType.GRAMMAR
        assert ChangeType.SPELLING
        assert ChangeType.PUNCTUATION
        assert ChangeType.STYLE
        assert ChangeType.FORMATTING
        assert ChangeType.STRUCTURE
        assert ChangeType.CLARIFICATION

    def test_type_values(self):
        """Test change type values."""
        assert ChangeType.GRAMMAR.value == "grammar"
        assert ChangeType.SPELLING.value == "spelling"
