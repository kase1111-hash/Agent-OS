"""
Reasoning Agent Template

Provides a template for agents that perform complex reasoning and analysis.
Includes support for chain-of-thought, multi-step reasoning, and evidence tracking.
"""

import logging
from abc import abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Set

from src.agents.interface import CapabilityType
from src.messaging.models import FlowRequest, FlowResponse, MessageStatus

from .base import AgentConfig, AgentTemplate

logger = logging.getLogger(__name__)


@dataclass
class ReasoningStep:
    """A step in a reasoning chain."""

    step_number: int
    description: str
    input_data: Any
    output_data: Any
    confidence: float = 1.0
    evidence: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step": self.step_number,
            "description": self.description,
            "input": self.input_data,
            "output": self.output_data,
            "confidence": self.confidence,
            "evidence": self.evidence,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class ReasoningResult:
    """Result of a reasoning process."""

    conclusion: str
    confidence: float
    steps: List[ReasoningStep] = field(default_factory=list)
    evidence: List[str] = field(default_factory=list)
    alternatives: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "conclusion": self.conclusion,
            "confidence": self.confidence,
            "steps": [s.to_dict() for s in self.steps],
            "evidence": self.evidence,
            "alternatives": self.alternatives,
            "metadata": self.metadata,
        }


@dataclass
class ReasoningConfig(AgentConfig):
    """Configuration for reasoning agents."""

    min_confidence_threshold: float = 0.7
    max_reasoning_steps: int = 10
    require_evidence: bool = True
    track_alternatives: bool = True
    explain_reasoning: bool = True

    def __post_init__(self):
        self.capabilities = self.capabilities or set()
        self.capabilities.add(CapabilityType.REASONING)


class ReasoningAgentTemplate(AgentTemplate):
    """
    Template for reasoning agents.

    Provides:
    - Chain-of-thought reasoning support
    - Evidence tracking
    - Confidence scoring
    - Alternative hypothesis tracking
    - Reasoning explanation
    """

    def __init__(self, config: ReasoningConfig):
        """
        Initialize reasoning agent.

        Args:
            config: Reasoning agent configuration
        """
        super().__init__(config)
        self.reasoning_config = config
        self._reasoning_chain: List[ReasoningStep] = []

    def handle_request(self, request: FlowRequest) -> FlowResponse:
        """Process request through reasoning pipeline."""
        # Reset reasoning chain
        self._reasoning_chain = []

        try:
            # Extract question/problem
            problem = self.extract_problem(request)

            # Perform reasoning
            result = self.reason(problem, request)

            # Check confidence threshold
            if result.confidence < self.reasoning_config.min_confidence_threshold:
                if self.agent_config.can_escalate:
                    return self._create_escalation_response(request, result)

            # Format output
            output = self.format_result(result)

            return request.create_response(
                source=self.name,
                status=MessageStatus.SUCCESS,
                output=output,
                reasoning=(
                    self._format_reasoning_chain()
                    if self.reasoning_config.explain_reasoning
                    else None
                ),
            )

        except Exception as e:
            logger.error(f"Reasoning error: {e}")
            return request.create_response(
                source=self.name,
                status=MessageStatus.ERROR,
                output="",
                errors=[f"Reasoning failed: {str(e)}"],
            )

    def extract_problem(self, request: FlowRequest) -> str:
        """
        Extract the problem/question from the request.

        Override for custom extraction logic.

        Args:
            request: The request

        Returns:
            The problem statement
        """
        return request.content.prompt

    @abstractmethod
    def reason(self, problem: str, request: FlowRequest) -> ReasoningResult:
        """
        Perform reasoning on the problem.

        Override this method to implement reasoning logic.

        Args:
            problem: The problem statement
            request: Original request for context

        Returns:
            ReasoningResult with conclusion
        """
        pass

    def add_reasoning_step(
        self,
        description: str,
        input_data: Any,
        output_data: Any,
        confidence: float = 1.0,
        evidence: Optional[List[str]] = None,
    ) -> ReasoningStep:
        """
        Add a step to the reasoning chain.

        Args:
            description: What this step does
            input_data: Input to this step
            output_data: Output from this step
            confidence: Confidence in this step
            evidence: Evidence supporting this step

        Returns:
            The created step
        """
        step = ReasoningStep(
            step_number=len(self._reasoning_chain) + 1,
            description=description,
            input_data=input_data,
            output_data=output_data,
            confidence=confidence,
            evidence=evidence or [],
        )
        self._reasoning_chain.append(step)
        return step

    def format_result(self, result: ReasoningResult) -> str:
        """
        Format the reasoning result for output.

        Override for custom formatting.

        Args:
            result: The reasoning result

        Returns:
            Formatted output string
        """
        output = result.conclusion

        if result.evidence and self.reasoning_config.require_evidence:
            output += "\n\nEvidence:\n"
            for i, ev in enumerate(result.evidence, 1):
                output += f"{i}. {ev}\n"

        if result.alternatives and self.reasoning_config.track_alternatives:
            output += "\n\nAlternative possibilities:\n"
            for alt in result.alternatives:
                output += f"- {alt.get('conclusion', 'Unknown')}: {alt.get('confidence', 0):.0%}\n"

        return output

    def _format_reasoning_chain(self) -> str:
        """Format the reasoning chain for explanation."""
        if not self._reasoning_chain:
            return ""

        lines = ["Reasoning process:"]
        for step in self._reasoning_chain:
            lines.append(f"  Step {step.step_number}: {step.description}")
            if step.evidence:
                for ev in step.evidence:
                    lines.append(f"    - Evidence: {ev}")

        return "\n".join(lines)

    def _create_escalation_response(
        self,
        request: FlowRequest,
        result: ReasoningResult,
    ) -> FlowResponse:
        """Create an escalation response for low confidence."""
        response = request.create_response(
            source=self.name,
            status=MessageStatus.PARTIAL,
            output=f"Low confidence ({result.confidence:.0%}). Human review recommended.\n\n{result.conclusion}",
            reasoning=self._format_reasoning_chain(),
        )
        response.next_actions.append(
            {
                "action": "escalate_to_human",
                "reason": f"Confidence {result.confidence:.0%} below threshold {self.reasoning_config.min_confidence_threshold:.0%}",
            }
        )
        return response


def create_reasoning_agent(
    name: str,
    reason_fn: Callable[[str, FlowRequest], ReasoningResult],
    description: str = "",
    min_confidence: float = 0.7,
    **kwargs,
) -> ReasoningAgentTemplate:
    """
    Create a reasoning agent from a function.

    Example:
        def my_reasoner(problem, request):
            # Analyze the problem
            return ReasoningResult(
                conclusion="The answer is 42",
                confidence=0.95,
            )

        agent = create_reasoning_agent(
            name="analyzer",
            reason_fn=my_reasoner,
            description="Analyzes complex problems",
        )

    Args:
        name: Agent name
        reason_fn: Reasoning function
        description: Agent description
        min_confidence: Minimum confidence threshold
        **kwargs: Additional config options

    Returns:
        ReasoningAgentTemplate instance
    """
    config = ReasoningConfig(
        name=name,
        description=description,
        min_confidence_threshold=min_confidence,
        **kwargs,
    )

    class FunctionReasoningAgent(ReasoningAgentTemplate):
        def reason(self, problem: str, request: FlowRequest) -> ReasoningResult:
            return reason_fn(problem, request)

    return FunctionReasoningAgent(config)
