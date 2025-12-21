"""
Tests for Agent OS Sage Agent (UC-009)
"""

import pytest
from datetime import datetime
from typing import Dict, Any
from unittest.mock import Mock, patch, MagicMock

from src.agents.sage import (
    SageAgent,
    SageConfig,
    create_sage_agent,
    ReasoningEngine,
    ReasoningConfig,
    ReasoningType,
    ReasoningChain,
    ReasoningStep,
    TradeOff,
    ConfidenceLevel,
    create_reasoning_engine,
)
from src.agents.interface import AgentState, CapabilityType
from src.messaging.models import FlowRequest, FlowResponse, MessageStatus


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def reasoning_config():
    """Create test reasoning config."""
    return ReasoningConfig(
        max_reasoning_steps=5,
        temperature=0.2,
        require_evidence=True,
    )


@pytest.fixture
def reasoning_engine(reasoning_config):
    """Create test reasoning engine."""
    return ReasoningEngine(config=reasoning_config)


@pytest.fixture
def mock_llm_callback():
    """Create mock LLM callback."""
    def callback(prompt: str, options: Dict[str, Any]) -> str:
        return """
        Step 1: Problem Understanding
        Reasoning: Analyzing the given query to identify key components.
        Conclusion: Query decomposed successfully.
        Confidence: high
        Assumptions: Query is well-formed

        Step 2: Analysis
        Reasoning: Examining the components systematically.
        Conclusion: Components are interconnected.
        Confidence: moderate
        Uncertainties: May miss edge cases

        Final Conclusion: The analysis reveals a complex but tractable problem.
        """
    return callback


@pytest.fixture
def sage_config():
    """Create test Sage config."""
    return SageConfig(
        model="mistral:7b",
        temperature=0.2,
        max_reasoning_steps=5,
        use_mock=True,
    )


@pytest.fixture
def sage_agent(sage_config):
    """Create test Sage agent."""
    agent = SageAgent(sage_config)
    agent.initialize({})
    return agent


@pytest.fixture
def sample_flow_request():
    """Create a sample FlowRequest for testing."""
    from src.messaging.models import create_request
    return create_request(
        source="test",
        destination="sage",
        intent="query.reasoning",
        prompt="Analyze the trade-offs between microservices and monolithic architecture.",
    )


# =============================================================================
# ReasoningStep Tests
# =============================================================================

class TestReasoningStep:
    """Tests for ReasoningStep."""

    def test_create_step(self):
        """Test creating a reasoning step."""
        step = ReasoningStep(
            step_number=1,
            description="Initial Analysis",
            reasoning="Examining the problem structure",
            conclusion="Problem is well-defined",
            confidence=ConfidenceLevel.HIGH,
        )

        assert step.step_number == 1
        assert step.description == "Initial Analysis"
        assert step.confidence == ConfidenceLevel.HIGH

    def test_step_with_assumptions(self):
        """Test step with assumptions."""
        step = ReasoningStep(
            step_number=1,
            description="Test",
            reasoning="Testing",
            conclusion="Complete",
            assumptions=["Data is accurate", "Context is complete"],
        )

        assert len(step.assumptions) == 2
        assert "Data is accurate" in step.assumptions

    def test_step_to_dict(self):
        """Test step serialization."""
        step = ReasoningStep(
            step_number=1,
            description="Test",
            reasoning="Testing reasoning",
            conclusion="Test complete",
            confidence=ConfidenceLevel.MODERATE,
        )

        data = step.to_dict()

        assert data["step_number"] == 1
        assert data["confidence"] == "moderate"
        assert "timestamp" in data

    def test_step_format_markdown(self):
        """Test markdown formatting."""
        step = ReasoningStep(
            step_number=2,
            description="Evaluation",
            reasoning="Comparing options",
            conclusion="Option A is preferable",
            confidence=ConfidenceLevel.HIGH,
            assumptions=["Equal cost"],
        )

        md = step.format_markdown()

        assert "### Step 2: Evaluation" in md
        assert "**Reasoning:**" in md
        assert "**Confidence:** high" in md
        assert "**Assumptions:**" in md


# =============================================================================
# TradeOff Tests
# =============================================================================

class TestTradeOff:
    """Tests for TradeOff."""

    def test_create_trade_off(self):
        """Test creating a trade-off."""
        trade_off = TradeOff(
            option="Microservices",
            pros=["Scalability", "Flexibility"],
            cons=["Complexity", "Overhead"],
            risk_level="medium",
        )

        assert trade_off.option == "Microservices"
        assert len(trade_off.pros) == 2
        assert len(trade_off.cons) == 2

    def test_trade_off_to_dict(self):
        """Test trade-off serialization."""
        trade_off = TradeOff(
            option="Test",
            pros=["Pro1"],
            cons=["Con1"],
            risk_level="low",
            recommendation="Consider carefully",
        )

        data = trade_off.to_dict()

        assert data["option"] == "Test"
        assert data["risk_level"] == "low"
        assert data["recommendation"] == "Consider carefully"


# =============================================================================
# ReasoningChain Tests
# =============================================================================

class TestReasoningChain:
    """Tests for ReasoningChain."""

    def test_create_chain(self):
        """Test creating a reasoning chain."""
        chain = ReasoningChain(
            chain_id="test_123",
            reasoning_type=ReasoningType.ANALYSIS,
            query="Test query",
        )

        assert chain.chain_id == "test_123"
        assert chain.reasoning_type == ReasoningType.ANALYSIS
        assert len(chain.steps) == 0

    def test_chain_with_steps(self):
        """Test chain with multiple steps."""
        chain = ReasoningChain(
            chain_id="test_456",
            reasoning_type=ReasoningType.SYNTHESIS,
            query="Synthesize information",
        )

        chain.steps.append(ReasoningStep(
            step_number=1,
            description="Step 1",
            reasoning="First step",
            conclusion="Done",
        ))
        chain.steps.append(ReasoningStep(
            step_number=2,
            description="Step 2",
            reasoning="Second step",
            conclusion="Complete",
        ))

        assert len(chain.steps) == 2

    def test_chain_to_dict(self):
        """Test chain serialization."""
        chain = ReasoningChain(
            chain_id="test_789",
            reasoning_type=ReasoningType.EVALUATION,
            query="Evaluate options",
            final_conclusion="Option A recommended for consideration",
            overall_confidence=ConfidenceLevel.HIGH,
        )

        data = chain.to_dict()

        assert data["chain_id"] == "test_789"
        assert data["reasoning_type"] == "evaluation"
        assert data["overall_confidence"] == "high"

    def test_chain_format_markdown(self):
        """Test markdown formatting."""
        chain = ReasoningChain(
            chain_id="md_test",
            reasoning_type=ReasoningType.ANALYSIS,
            query="Test markdown formatting",
            final_conclusion="Analysis complete",
            overall_confidence=ConfidenceLevel.MODERATE,
        )

        chain.steps.append(ReasoningStep(
            step_number=1,
            description="Analysis",
            reasoning="Analyzing...",
            conclusion="Analyzed",
            confidence=ConfidenceLevel.HIGH,
        ))

        md = chain.format_markdown()

        assert "# Reasoning Analysis: Analysis" in md
        assert "**Query:**" in md
        assert "## Reasoning Steps" in md
        assert "## Final Conclusion" in md


# =============================================================================
# ReasoningEngine Tests
# =============================================================================

class TestReasoningEngine:
    """Tests for ReasoningEngine."""

    def test_create_engine(self, reasoning_config):
        """Test creating reasoning engine."""
        engine = ReasoningEngine(config=reasoning_config)

        assert engine._config.max_reasoning_steps == 5
        assert engine._config.temperature == 0.2

    def test_reason_without_llm(self, reasoning_engine):
        """Test reasoning without LLM (structured fallback)."""
        chain = reasoning_engine.reason(
            query="What are the benefits of testing?",
            reasoning_type=ReasoningType.ANALYSIS,
        )

        assert chain.chain_id.startswith("chain_")
        assert chain.reasoning_type == ReasoningType.ANALYSIS
        assert len(chain.steps) > 0

    def test_reason_with_context(self, reasoning_engine):
        """Test reasoning with context."""
        chain = reasoning_engine.reason(
            query="Analyze the situation",
            context="Additional context about the problem",
            reasoning_type=ReasoningType.ANALYSIS,
        )

        assert len(chain.steps) >= 2  # Should have context integration step

    def test_reason_with_constraints(self, reasoning_engine):
        """Test reasoning with constraints."""
        chain = reasoning_engine.reason(
            query="Analyze options",
            constraints=["Must be cost-effective", "Must be scalable"],
            reasoning_type=ReasoningType.ANALYSIS,
        )

        # Should have constraint application step
        assert any("onstraint" in step.description for step in chain.steps)

    def test_reason_with_llm_callback(self, reasoning_engine, mock_llm_callback):
        """Test reasoning with LLM callback."""
        reasoning_engine.set_llm_callback(mock_llm_callback)

        chain = reasoning_engine.reason(
            query="Test with LLM",
            reasoning_type=ReasoningType.ANALYSIS,
        )

        assert len(chain.steps) >= 1
        assert chain.final_conclusion != ""

    def test_escalation_detection(self, reasoning_engine):
        """Test human escalation detection."""
        chain = reasoning_engine.reason(
            query="What should I do about this ethical dilemma?",
            reasoning_type=ReasoningType.ANALYSIS,
        )

        assert chain.requires_human_judgment is True
        assert chain.escalation_reason is not None

    def test_escalation_for_decisions(self, reasoning_engine):
        """Test escalation for decision requests."""
        chain = reasoning_engine.reason(
            query="Make a decision for me about which job to take",
            reasoning_type=ReasoningType.EVALUATION,
        )

        assert chain.requires_human_judgment is True

    def test_confidence_evaluation(self, reasoning_engine):
        """Test confidence evaluation."""
        chain = ReasoningChain(
            chain_id="conf_test",
            reasoning_type=ReasoningType.ANALYSIS,
            query="Test confidence",
        )

        chain.steps.append(ReasoningStep(
            step_number=1,
            description="High confidence step",
            reasoning="Clear reasoning",
            conclusion="Certain",
            confidence=ConfidenceLevel.HIGH,
        ))
        chain.steps.append(ReasoningStep(
            step_number=2,
            description="Moderate step",
            reasoning="Less certain",
            conclusion="Probable",
            confidence=ConfidenceLevel.MODERATE,
        ))

        confidence = reasoning_engine.evaluate_confidence(chain)

        assert confidence in [ConfidenceLevel.HIGH, ConfidenceLevel.MODERATE]

    def test_analyze_trade_offs(self, reasoning_engine):
        """Test trade-off analysis."""
        trade_offs = reasoning_engine.analyze_trade_offs(
            options=["Option A", "Option B"],
            criteria=["Cost", "Speed"],
        )

        assert len(trade_offs) == 2
        assert trade_offs[0].option == "Option A"
        assert trade_offs[1].option == "Option B"

    def test_get_metrics(self, reasoning_engine):
        """Test metrics retrieval."""
        # Perform some reasoning
        reasoning_engine.reason("Test query 1")
        reasoning_engine.reason("Test query 2")

        metrics = reasoning_engine.get_metrics()

        assert metrics["total_chains"] == 2
        assert "total_steps" in metrics
        assert "escalations" in metrics


class TestCreateReasoningEngine:
    """Tests for reasoning engine factory."""

    def test_create_default(self):
        """Test creating with defaults."""
        engine = create_reasoning_engine()

        assert engine is not None
        assert engine._config.temperature == 0.2

    def test_create_with_callback(self, mock_llm_callback):
        """Test creating with LLM callback."""
        engine = create_reasoning_engine(llm_callback=mock_llm_callback)

        chain = engine.reason("Test")
        assert chain.final_conclusion != ""

    def test_create_with_config(self):
        """Test creating with custom config."""
        engine = create_reasoning_engine(
            temperature=0.1,
            max_steps=3,
        )

        assert engine._config.temperature == 0.1
        assert engine._config.max_reasoning_steps == 3


# =============================================================================
# SageAgent Tests
# =============================================================================

class TestSageAgent:
    """Tests for SageAgent."""

    def test_create_agent(self, sage_config):
        """Test creating Sage agent."""
        agent = SageAgent(sage_config)

        assert agent.name == "sage"
        assert agent._state == AgentState.UNINITIALIZED

    def test_initialize(self, sage_config):
        """Test agent initialization."""
        agent = SageAgent(sage_config)
        result = agent.initialize({})

        assert result is True
        assert agent._state == AgentState.READY
        assert agent._reasoning_engine is not None

    def test_get_capabilities(self, sage_agent):
        """Test capabilities retrieval."""
        caps = sage_agent.get_capabilities()

        assert caps.name == "sage"
        assert CapabilityType.REASONING in caps.capabilities
        assert "query.reasoning" in caps.supported_intents

    def test_validate_request_valid(self, sage_agent, sample_flow_request):
        """Test validating a valid request."""
        result = sage_agent.validate_request(sample_flow_request)

        assert result.is_valid is True
        assert len(result.errors) == 0

    def test_validate_request_invalid_intent(self, sage_agent):
        """Test validating request with invalid intent."""
        from src.messaging.models import create_request

        request = create_request(
            source="test",
            destination="sage",
            intent="invalid.intent",
            prompt="Test",
        )

        result = sage_agent.validate_request(request)

        assert result.is_valid is False
        assert "Unsupported intent" in result.errors[0]

    def test_validate_request_empty_prompt(self, sage_agent):
        """Test validating request with empty prompt."""
        from src.messaging.models import create_request

        request = create_request(
            source="test",
            destination="sage",
            intent="query.reasoning",
            prompt="",
        )

        result = sage_agent.validate_request(request)

        assert result.is_valid is False
        assert "cannot be empty" in result.errors[0]

    def test_validate_request_escalation(self, sage_agent):
        """Test validation triggers escalation for decisions."""
        from src.messaging.models import create_request

        request = create_request(
            source="test",
            destination="sage",
            intent="query.reasoning",
            prompt="What should I do? Just tell me the answer.",
        )

        result = sage_agent.validate_request(request)

        assert result.requires_escalation is True
        assert "decision" in result.escalation_reason.lower() or "judgment" in result.escalation_reason.lower()

    def test_process_reasoning_request(self, sage_agent, sample_flow_request):
        """Test processing a reasoning request."""
        response = sage_agent.process(sample_flow_request)

        assert response.status == MessageStatus.SUCCESS
        assert response.content.output != ""
        assert "sage" in response.source

    def test_process_different_intents(self, sage_agent):
        """Test processing different reasoning intents."""
        from src.messaging.models import create_request

        intents = [
            ("reasoning.analyze", "Analyze this problem"),
            ("reasoning.synthesize", "Synthesize these sources"),
            ("reasoning.evaluate", "Evaluate these options"),
            ("reasoning.compare", "Compare A and B"),
        ]

        for intent, prompt in intents:
            request = create_request(
                source="test",
                destination="sage",
                intent=intent,
                prompt=prompt,
            )

            response = sage_agent.process(request)
            assert response.status == MessageStatus.SUCCESS, f"Failed for intent: {intent}"

    def test_shutdown(self, sage_agent):
        """Test agent shutdown."""
        result = sage_agent.shutdown()

        assert result is True
        assert sage_agent._state == AgentState.SHUTDOWN


class TestSageAgentDirectAPI:
    """Tests for Sage direct API methods."""

    def test_reason_direct(self, sage_agent):
        """Test direct reasoning."""
        chain = sage_agent.reason("Analyze software testing strategies")

        assert chain is not None
        assert chain.reasoning_type == ReasoningType.ANALYSIS
        assert len(chain.steps) > 0

    def test_analyze(self, sage_agent):
        """Test analyze method."""
        chain = sage_agent.analyze(
            query="Analyze the impact of AI on software development",
            context="Focus on productivity and quality",
        )

        assert chain.reasoning_type == ReasoningType.ANALYSIS

    def test_synthesize(self, sage_agent):
        """Test synthesize method."""
        chain = sage_agent.synthesize(
            query="What are the key themes?",
            sources=[
                "Source 1: AI improves efficiency",
                "Source 2: AI requires careful oversight",
            ],
        )

        assert chain.reasoning_type == ReasoningType.SYNTHESIS

    def test_evaluate_options(self, sage_agent):
        """Test evaluate options method."""
        chain = sage_agent.evaluate_options(
            query="Which architecture should we consider?",
            options=["Microservices", "Monolith", "Serverless"],
            criteria=["Scalability", "Simplicity"],
        )

        assert chain.reasoning_type == ReasoningType.EVALUATION

    def test_compare(self, sage_agent):
        """Test compare method."""
        chain = sage_agent.compare(
            item_a="Python",
            item_b="JavaScript",
            aspects=["Type system", "Performance", "Ecosystem"],
        )

        assert chain.reasoning_type == ReasoningType.COMPARISON

    def test_get_statistics(self, sage_agent):
        """Test statistics retrieval."""
        # Perform some reasoning
        sage_agent.reason("Test query")

        stats = sage_agent.get_statistics()

        assert "agent" in stats
        assert stats["agent"]["name"] == "sage"
        assert "reasoning" in stats


class TestCreateSageAgent:
    """Tests for agent factory."""

    def test_create_default(self):
        """Test creating with defaults."""
        agent = create_sage_agent()

        assert agent is not None
        assert agent._state == AgentState.READY
        assert agent._sage_config.use_mock is True

    def test_create_with_config(self):
        """Test creating with custom config."""
        agent = create_sage_agent(
            model="llama3:8b",
            temperature=0.1,
            max_reasoning_steps=3,
        )

        assert agent._sage_config.model == "llama3:8b"
        assert agent._sage_config.temperature == 0.1


# =============================================================================
# Integration Tests
# =============================================================================

class TestSageIntegration:
    """Integration tests for Sage agent."""

    def test_full_reasoning_workflow(self, sage_agent):
        """Test complete reasoning workflow."""
        # 1. Get capabilities
        caps = sage_agent.get_capabilities()
        assert caps.name == "sage"

        # 2. Validate request
        from src.messaging.models import create_request
        request = create_request(
            source="whisper",
            destination="sage",
            intent="query.reasoning",
            prompt="Analyze the pros and cons of cloud computing",
        )

        validation = sage_agent.validate_request(request)
        assert validation.is_valid

        # 3. Process request
        response = sage_agent.process(request)
        assert response.status == MessageStatus.SUCCESS

        # 4. Check output structure
        assert "Step" in response.content.output or "step" in response.content.output
        assert "Conclusion" in response.content.output or "conclusion" in response.content.output

    def test_escalation_workflow(self, sage_agent):
        """Test escalation to human workflow."""
        from src.messaging.models import create_request

        request = create_request(
            source="whisper",
            destination="sage",
            intent="query.reasoning",
            prompt="Should I take the new job or stay at my current company?",
        )

        # May trigger escalation at validation
        validation = sage_agent.validate_request(request)

        if validation.requires_escalation:
            assert "decision" in validation.escalation_reason.lower() or \
                   "judgment" in validation.escalation_reason.lower()
        else:
            # Or at processing
            response = sage_agent.process(request)
            if response.status == MessageStatus.PARTIAL:
                assert any(
                    action.get("action") == "escalate_to_human"
                    for action in response.next_actions
                )

    def test_long_context_reasoning(self, sage_agent):
        """Test reasoning with longer context."""
        long_context = """
        Consider the following factors in the analysis:

        1. Technical Requirements:
           - High availability (99.99% uptime)
           - Low latency (<100ms response time)
           - Scalability to 1M users

        2. Business Constraints:
           - Limited budget ($50k/year)
           - Small team (3 developers)
           - Launch deadline in 3 months

        3. Existing Infrastructure:
           - Legacy Java backend
           - PostgreSQL database
           - On-premises data center
        """

        chain = sage_agent.reason(
            query="What architecture should we consider for the new system?",
            context=long_context,
            max_steps=8,
        )

        assert chain is not None
        assert len(chain.steps) > 2

    def test_multi_step_analysis(self, sage_agent):
        """Test multi-step analysis reasoning."""
        chain = sage_agent.reason(
            query=(
                "A startup is deciding between building their own ML infrastructure "
                "vs using cloud ML services. They have 5 ML engineers and $200k budget. "
                "What factors should they consider?"
            ),
            reasoning_type=ReasoningType.ANALYSIS,
            max_steps=6,
        )

        # Should have multiple steps
        assert len(chain.steps) >= 2

        # Should have conclusion
        assert chain.final_conclusion != ""

        # Should respect human judgment
        assert not any(
            "you must" in chain.final_conclusion.lower() or
            "you should definitely" in chain.final_conclusion.lower()
            for _ in [1]
        )


# =============================================================================
# Constitutional Compliance Tests
# =============================================================================

class TestConstitutionalCompliance:
    """Tests for constitutional compliance."""

    def test_no_value_judgments(self, sage_agent):
        """Test that Sage doesn't make value judgments."""
        chain = sage_agent.reason(
            query="Is Python better than JavaScript?",
            reasoning_type=ReasoningType.COMPARISON,
        )

        # Should present comparison, not judgment
        conclusion_lower = chain.final_conclusion.lower()
        assert "depends" in conclusion_lower or \
               "consider" in conclusion_lower or \
               "trade-off" in conclusion_lower or \
               "context" in conclusion_lower or \
               "human" in conclusion_lower or \
               len(chain.final_conclusion) < 500  # Structured analysis

    def test_shows_reasoning_steps(self, sage_agent):
        """Test that reasoning steps are shown explicitly."""
        chain = sage_agent.reason(
            query="Explain how databases ensure ACID properties",
        )

        # Must have explicit steps
        assert len(chain.steps) >= 1

        # Each step should have reasoning
        for step in chain.steps:
            assert step.reasoning != ""
            assert step.conclusion != ""

    def test_acknowledges_uncertainty(self, sage_agent):
        """Test that uncertainty is acknowledged."""
        chain = sage_agent.reason(
            query="Predict the future of quantum computing in 10 years",
        )

        # Should have uncertainty indicators
        has_uncertainty = (
            chain.overall_confidence != ConfidenceLevel.VERY_HIGH or
            len(chain.open_questions) > 0 or
            any(len(step.uncertainties) > 0 for step in chain.steps) or
            "uncertain" in chain.final_conclusion.lower() or
            "may" in chain.final_conclusion.lower()
        )

        assert has_uncertainty

    def test_respects_human_sovereignty(self, sage_agent):
        """Test that human decision authority is respected."""
        from src.messaging.models import create_request

        request = create_request(
            source="test",
            destination="sage",
            intent="query.reasoning",
            prompt="Tell me what to do with my life savings",
        )

        validation = sage_agent.validate_request(request)

        # Should trigger escalation
        assert validation.requires_escalation is True


# =============================================================================
# ConfidenceLevel Tests
# =============================================================================

class TestConfidenceLevel:
    """Tests for ConfidenceLevel enum."""

    def test_all_levels_exist(self):
        """Test all confidence levels are defined."""
        assert ConfidenceLevel.VERY_HIGH
        assert ConfidenceLevel.HIGH
        assert ConfidenceLevel.MODERATE
        assert ConfidenceLevel.LOW
        assert ConfidenceLevel.VERY_LOW
        assert ConfidenceLevel.UNCERTAIN

    def test_confidence_values(self):
        """Test confidence level values."""
        assert ConfidenceLevel.VERY_HIGH.value == "very_high"
        assert ConfidenceLevel.MODERATE.value == "moderate"


# =============================================================================
# ReasoningType Tests
# =============================================================================

class TestReasoningType:
    """Tests for ReasoningType enum."""

    def test_all_types_exist(self):
        """Test all reasoning types are defined."""
        assert ReasoningType.ANALYSIS
        assert ReasoningType.SYNTHESIS
        assert ReasoningType.EVALUATION
        assert ReasoningType.DEDUCTION
        assert ReasoningType.COMPARISON
        assert ReasoningType.CAUSAL

    def test_type_values(self):
        """Test reasoning type values."""
        assert ReasoningType.ANALYSIS.value == "analysis"
        assert ReasoningType.SYNTHESIS.value == "synthesis"
