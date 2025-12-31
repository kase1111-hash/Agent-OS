"""
Agent OS Sage Reasoning Engine

Provides chain-of-thought reasoning with transparent intermediate steps.
Designed for complex multi-step analysis, synthesis, and trade-off evaluation.
"""

import logging
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class ReasoningType(Enum):
    """Types of reasoning tasks."""

    ANALYSIS = "analysis"  # Break down and analyze a problem
    SYNTHESIS = "synthesis"  # Combine information from multiple sources
    EVALUATION = "evaluation"  # Evaluate trade-offs and alternatives
    DEDUCTION = "deduction"  # Logical deduction from premises
    INDUCTION = "induction"  # Pattern recognition and generalization
    ABDUCTION = "abduction"  # Best explanation inference
    COMPARISON = "comparison"  # Compare and contrast options
    CAUSAL = "causal"  # Cause and effect analysis


class ConfidenceLevel(Enum):
    """Confidence levels for reasoning conclusions."""

    VERY_HIGH = "very_high"  # 90%+ confidence
    HIGH = "high"  # 75-90% confidence
    MODERATE = "moderate"  # 50-75% confidence
    LOW = "low"  # 25-50% confidence
    VERY_LOW = "very_low"  # <25% confidence
    UNCERTAIN = "uncertain"  # Cannot assess confidence


@dataclass
class ReasoningStep:
    """A single step in a reasoning chain."""

    step_number: int
    description: str
    reasoning: str
    conclusion: str
    assumptions: List[str] = field(default_factory=list)
    uncertainties: List[str] = field(default_factory=list)
    confidence: ConfidenceLevel = ConfidenceLevel.MODERATE
    evidence: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_number": self.step_number,
            "description": self.description,
            "reasoning": self.reasoning,
            "conclusion": self.conclusion,
            "assumptions": self.assumptions,
            "uncertainties": self.uncertainties,
            "confidence": self.confidence.value,
            "evidence": self.evidence,
            "timestamp": self.timestamp.isoformat(),
        }

    def format_markdown(self) -> str:
        """Format step as markdown."""
        lines = [
            f"### Step {self.step_number}: {self.description}",
            "",
            f"**Reasoning:** {self.reasoning}",
            "",
            f"**Conclusion:** {self.conclusion}",
            "",
            f"**Confidence:** {self.confidence.value}",
        ]

        if self.assumptions:
            lines.append("")
            lines.append("**Assumptions:**")
            for assumption in self.assumptions:
                lines.append(f"- {assumption}")

        if self.uncertainties:
            lines.append("")
            lines.append("**Uncertainties:**")
            for uncertainty in self.uncertainties:
                lines.append(f"- {uncertainty}")

        if self.evidence:
            lines.append("")
            lines.append("**Evidence:**")
            for item in self.evidence:
                lines.append(f"- {item}")

        return "\n".join(lines)


@dataclass
class TradeOff:
    """A trade-off consideration."""

    option: str
    pros: List[str] = field(default_factory=list)
    cons: List[str] = field(default_factory=list)
    risk_level: str = "medium"
    recommendation: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "option": self.option,
            "pros": self.pros,
            "cons": self.cons,
            "risk_level": self.risk_level,
            "recommendation": self.recommendation,
        }


@dataclass
class ReasoningChain:
    """Complete chain-of-thought reasoning result."""

    chain_id: str
    reasoning_type: ReasoningType
    query: str
    steps: List[ReasoningStep] = field(default_factory=list)
    final_conclusion: str = ""
    overall_confidence: ConfidenceLevel = ConfidenceLevel.MODERATE
    trade_offs: List[TradeOff] = field(default_factory=list)
    key_assumptions: List[str] = field(default_factory=list)
    open_questions: List[str] = field(default_factory=list)
    requires_human_judgment: bool = False
    escalation_reason: Optional[str] = None
    processing_time_ms: float = 0.0
    total_tokens: int = 0
    created_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chain_id": self.chain_id,
            "reasoning_type": self.reasoning_type.value,
            "query": self.query,
            "steps": [s.to_dict() for s in self.steps],
            "final_conclusion": self.final_conclusion,
            "overall_confidence": self.overall_confidence.value,
            "trade_offs": [t.to_dict() for t in self.trade_offs],
            "key_assumptions": self.key_assumptions,
            "open_questions": self.open_questions,
            "requires_human_judgment": self.requires_human_judgment,
            "escalation_reason": self.escalation_reason,
            "processing_time_ms": self.processing_time_ms,
            "total_tokens": self.total_tokens,
            "created_at": self.created_at.isoformat(),
        }

    def format_markdown(self) -> str:
        """Format chain as markdown report."""
        lines = [
            f"# Reasoning Analysis: {self.reasoning_type.value.title()}",
            "",
            f"**Query:** {self.query}",
            "",
            f"**Chain ID:** {self.chain_id}",
            f"**Overall Confidence:** {self.overall_confidence.value}",
            "",
            "---",
            "",
            "## Reasoning Steps",
            "",
        ]

        for step in self.steps:
            lines.append(step.format_markdown())
            lines.append("")
            lines.append("---")
            lines.append("")

        lines.append("## Final Conclusion")
        lines.append("")
        lines.append(self.final_conclusion)
        lines.append("")

        if self.key_assumptions:
            lines.append("## Key Assumptions")
            lines.append("")
            for assumption in self.key_assumptions:
                lines.append(f"- {assumption}")
            lines.append("")

        if self.trade_offs:
            lines.append("## Trade-Offs Analysis")
            lines.append("")
            for trade_off in self.trade_offs:
                lines.append(f"### {trade_off.option}")
                lines.append("")
                if trade_off.pros:
                    lines.append("**Pros:**")
                    for pro in trade_off.pros:
                        lines.append(f"- ✓ {pro}")
                if trade_off.cons:
                    lines.append("")
                    lines.append("**Cons:**")
                    for con in trade_off.cons:
                        lines.append(f"- ✗ {con}")
                lines.append("")
                lines.append(f"**Risk Level:** {trade_off.risk_level}")
                if trade_off.recommendation:
                    lines.append(f"**Recommendation:** {trade_off.recommendation}")
                lines.append("")

        if self.open_questions:
            lines.append("## Open Questions")
            lines.append("")
            for question in self.open_questions:
                lines.append(f"- {question}")
            lines.append("")

        if self.requires_human_judgment:
            lines.append("## ⚠️ Human Judgment Required")
            lines.append("")
            lines.append(self.escalation_reason or "This reasoning requires human decision-making.")
            lines.append("")

        lines.append("---")
        lines.append("")
        lines.append(f"*Processing time: {self.processing_time_ms:.1f}ms*")

        return "\n".join(lines)


@dataclass
class ReasoningConfig:
    """Configuration for the reasoning engine."""

    max_reasoning_steps: int = 10
    min_confidence_threshold: float = 0.25
    require_evidence: bool = True
    show_intermediate_steps: bool = True
    temperature: float = 0.2
    max_tokens_per_step: int = 1000
    escalation_keywords: List[str] = field(
        default_factory=lambda: [
            "ethical",
            "moral",
            "should",
            "value",
            "belief",
            "personal",
            "decision",
            "choose",
            "prefer",
            "recommend",
        ]
    )
    # Keywords that trigger human escalation
    prohibited_conclusions: List[str] = field(
        default_factory=lambda: [
            "you must",
            "you should definitely",
            "the only answer",
            "without question",
            "absolutely certain",
        ]
    )


class ReasoningEngine:
    """
    Chain-of-thought reasoning engine for complex analysis.

    Implements transparent multi-step reasoning with:
    - Explicit intermediate steps
    - Confidence tracking
    - Assumption documentation
    - Trade-off analysis
    - Human escalation detection
    """

    def __init__(
        self,
        config: Optional[ReasoningConfig] = None,
        llm_callback: Optional[Callable[[str, Dict[str, Any]], str]] = None,
    ):
        """
        Initialize reasoning engine.

        Args:
            config: Reasoning configuration
            llm_callback: Callback function to invoke LLM
                          (prompt, options) -> response
        """
        self._config = config or ReasoningConfig()
        self._llm_callback = llm_callback
        self._chain_count = 0

        # Metrics
        self._total_chains = 0
        self._total_steps = 0
        self._escalations = 0

    def set_llm_callback(self, callback: Callable[[str, Dict[str, Any]], str]) -> None:
        """Set the LLM callback for generation."""
        self._llm_callback = callback

    def reason(
        self,
        query: str,
        reasoning_type: ReasoningType = ReasoningType.ANALYSIS,
        context: Optional[str] = None,
        constraints: Optional[List[str]] = None,
        max_steps: Optional[int] = None,
    ) -> ReasoningChain:
        """
        Perform chain-of-thought reasoning on a query.

        Args:
            query: The question or problem to reason about
            reasoning_type: Type of reasoning to perform
            context: Additional context information
            constraints: Constraints to apply
            max_steps: Maximum reasoning steps (overrides config)

        Returns:
            ReasoningChain with complete reasoning
        """
        import secrets

        start_time = time.time()

        self._chain_count += 1
        chain_id = f"chain_{secrets.token_hex(8)}"
        max_steps = max_steps or self._config.max_reasoning_steps

        chain = ReasoningChain(
            chain_id=chain_id,
            reasoning_type=reasoning_type,
            query=query,
        )

        # Check for escalation triggers in query
        if self._requires_escalation(query):
            chain.requires_human_judgment = True
            chain.escalation_reason = self._get_escalation_reason(query)

        try:
            # Generate reasoning steps
            if self._llm_callback:
                chain = self._generate_reasoning_with_llm(
                    chain, query, reasoning_type, context, constraints, max_steps
                )
            else:
                # Fallback: structured analysis without LLM
                chain = self._generate_structured_reasoning(
                    chain, query, reasoning_type, context, constraints, max_steps
                )

            # Post-process chain
            chain = self._finalize_chain(chain)

        except Exception as e:
            logger.error(f"Reasoning error: {e}")
            chain.final_conclusion = f"Reasoning interrupted: {str(e)}"
            chain.overall_confidence = ConfidenceLevel.VERY_LOW

        # Calculate processing time
        chain.processing_time_ms = (time.time() - start_time) * 1000

        # Update metrics
        self._total_chains += 1
        self._total_steps += len(chain.steps)
        if chain.requires_human_judgment:
            self._escalations += 1

        return chain

    def analyze_trade_offs(
        self,
        options: List[str],
        criteria: Optional[List[str]] = None,
        context: Optional[str] = None,
    ) -> List[TradeOff]:
        """
        Analyze trade-offs between multiple options.

        Args:
            options: List of options to compare
            criteria: Evaluation criteria
            context: Additional context

        Returns:
            List of TradeOff analyses
        """
        trade_offs = []

        for option in options:
            trade_off = TradeOff(option=option)

            if self._llm_callback:
                # Use LLM for trade-off analysis
                prompt = self._build_trade_off_prompt(option, criteria, context)
                response = self._llm_callback(
                    prompt,
                    {
                        "temperature": self._config.temperature,
                        "max_tokens": 500,
                    },
                )
                trade_off = self._parse_trade_off_response(option, response)
            else:
                # Placeholder analysis
                trade_off.pros = [f"Potential benefit of {option}"]
                trade_off.cons = [f"Potential drawback of {option}"]
                trade_off.risk_level = "unknown"

            trade_offs.append(trade_off)

        return trade_offs

    def evaluate_confidence(self, chain: ReasoningChain) -> ConfidenceLevel:
        """
        Evaluate overall confidence for a reasoning chain.

        Args:
            chain: The reasoning chain

        Returns:
            Overall confidence level
        """
        if not chain.steps:
            return ConfidenceLevel.UNCERTAIN

        # Aggregate step confidences
        confidence_values = {
            ConfidenceLevel.VERY_HIGH: 0.95,
            ConfidenceLevel.HIGH: 0.8,
            ConfidenceLevel.MODERATE: 0.6,
            ConfidenceLevel.LOW: 0.35,
            ConfidenceLevel.VERY_LOW: 0.15,
            ConfidenceLevel.UNCERTAIN: 0.0,
        }

        total = sum(confidence_values[step.confidence] for step in chain.steps)
        avg = total / len(chain.steps)

        # Penalize for many assumptions
        assumption_count = sum(len(step.assumptions) for step in chain.steps)
        avg -= min(0.1, assumption_count * 0.02)

        # Penalize for uncertainties
        uncertainty_count = sum(len(step.uncertainties) for step in chain.steps)
        avg -= min(0.15, uncertainty_count * 0.03)

        # Map back to level
        if avg >= 0.85:
            return ConfidenceLevel.VERY_HIGH
        elif avg >= 0.7:
            return ConfidenceLevel.HIGH
        elif avg >= 0.5:
            return ConfidenceLevel.MODERATE
        elif avg >= 0.25:
            return ConfidenceLevel.LOW
        elif avg > 0:
            return ConfidenceLevel.VERY_LOW
        else:
            return ConfidenceLevel.UNCERTAIN

    def get_metrics(self) -> Dict[str, Any]:
        """Get reasoning engine metrics."""
        return {
            "total_chains": self._total_chains,
            "total_steps": self._total_steps,
            "average_steps_per_chain": (
                self._total_steps / self._total_chains if self._total_chains > 0 else 0
            ),
            "escalations": self._escalations,
            "escalation_rate": (
                self._escalations / self._total_chains if self._total_chains > 0 else 0
            ),
        }

    def _requires_escalation(self, query: str) -> bool:
        """Check if query requires human escalation."""
        query_lower = query.lower()

        for keyword in self._config.escalation_keywords:
            if keyword in query_lower:
                return True

        return False

    def _get_escalation_reason(self, query: str) -> str:
        """Get reason for escalation."""
        query_lower = query.lower()

        for keyword in self._config.escalation_keywords:
            if keyword in query_lower:
                if keyword in ["ethical", "moral", "value"]:
                    return "This reasoning involves ethical or moral considerations that require human judgment."
                elif keyword in ["should", "decision", "choose", "prefer"]:
                    return "This reasoning leads to a decision that should be made by a human."
                elif keyword == "personal":
                    return "This reasoning involves personal matters requiring human discretion."
                elif keyword == "recommend":
                    return "Recommendations require human evaluation of trade-offs."

        return "This reasoning may require human judgment to proceed."

    def _generate_reasoning_with_llm(
        self,
        chain: ReasoningChain,
        query: str,
        reasoning_type: ReasoningType,
        context: Optional[str],
        constraints: Optional[List[str]],
        max_steps: int,
    ) -> ReasoningChain:
        """Generate reasoning using LLM."""
        # Build system prompt for reasoning
        system_prompt = self._build_reasoning_system_prompt(reasoning_type)

        # Build initial prompt
        prompt = self._build_reasoning_prompt(
            query, reasoning_type, context, constraints, max_steps
        )

        # Get LLM response
        response = self._llm_callback(
            prompt,
            {
                "system": system_prompt,
                "temperature": self._config.temperature,
                "max_tokens": self._config.max_tokens_per_step * max_steps,
            },
        )

        # Parse response into steps
        chain = self._parse_reasoning_response(chain, response)

        return chain

    def _generate_structured_reasoning(
        self,
        chain: ReasoningChain,
        query: str,
        reasoning_type: ReasoningType,
        context: Optional[str],
        constraints: Optional[List[str]],
        max_steps: int,
    ) -> ReasoningChain:
        """Generate structured reasoning without LLM (fallback)."""
        # Step 1: Problem understanding
        step1 = ReasoningStep(
            step_number=1,
            description="Problem Understanding",
            reasoning=f"Analyzing the query: '{query[:100]}...'",
            conclusion="Query decomposed into analyzable components.",
            assumptions=["Query is well-formed", "Sufficient context available"],
            confidence=ConfidenceLevel.MODERATE,
        )
        chain.steps.append(step1)

        # Step 2: Context integration
        if context:
            step2 = ReasoningStep(
                step_number=2,
                description="Context Integration",
                reasoning=f"Incorporating provided context: '{context[:100]}...'",
                conclusion="Context integrated into analysis.",
                confidence=ConfidenceLevel.MODERATE,
            )
            chain.steps.append(step2)

        # Step 3: Apply constraints
        if constraints:
            step3 = ReasoningStep(
                step_number=len(chain.steps) + 1,
                description="Constraint Application",
                reasoning=f"Applying {len(constraints)} constraints to reasoning.",
                conclusion="Constraints incorporated.",
                assumptions=[f"Constraint valid: {c}" for c in constraints],
                confidence=ConfidenceLevel.HIGH,
            )
            chain.steps.append(step3)

        # Step 4: Preliminary conclusion
        step_final = ReasoningStep(
            step_number=len(chain.steps) + 1,
            description="Preliminary Analysis",
            reasoning="Based on available information, forming initial conclusions.",
            conclusion="Further analysis with LLM reasoning is recommended for complete results.",
            uncertainties=["Limited without LLM inference", "May miss complex relationships"],
            confidence=ConfidenceLevel.LOW,
        )
        chain.steps.append(step_final)

        chain.final_conclusion = (
            "Structured analysis complete. For comprehensive chain-of-thought "
            "reasoning with deeper insights, enable LLM inference."
        )
        chain.key_assumptions.append("Analysis limited to structural decomposition")
        chain.open_questions.append("What additional context would improve analysis?")

        return chain

    def _build_reasoning_system_prompt(self, reasoning_type: ReasoningType) -> str:
        """Build system prompt for reasoning type."""
        base = (
            "You are Sage, a reasoning agent performing rigorous chain-of-thought analysis. "
            "You MUST show your reasoning explicitly in numbered steps. "
            "Each step should include: description, reasoning, conclusion, and confidence level. "
            "Always acknowledge assumptions and uncertainties. "
            "Never make value judgments or final decisions - those belong to humans.\n\n"
        )

        type_instructions = {
            ReasoningType.ANALYSIS: (
                "Perform systematic analysis: break down the problem, examine components, "
                "identify relationships, and synthesize findings."
            ),
            ReasoningType.SYNTHESIS: (
                "Synthesize information from multiple sources: identify patterns, "
                "find connections, resolve conflicts, and build coherent understanding."
            ),
            ReasoningType.EVALUATION: (
                "Evaluate options and trade-offs: list pros and cons for each option, "
                "assess risks, compare against criteria, but do NOT make the final choice."
            ),
            ReasoningType.DEDUCTION: (
                "Apply deductive logic: state premises clearly, apply logical rules, "
                "derive conclusions step by step, verify validity."
            ),
            ReasoningType.COMPARISON: (
                "Compare and contrast options: identify similarities and differences, "
                "evaluate against criteria, present balanced analysis."
            ),
            ReasoningType.CAUSAL: (
                "Analyze cause and effect: identify potential causes, trace effects, "
                "consider confounding factors, assess causal strength."
            ),
        }

        return base + type_instructions.get(
            reasoning_type, type_instructions[ReasoningType.ANALYSIS]
        )

    def _build_reasoning_prompt(
        self,
        query: str,
        reasoning_type: ReasoningType,
        context: Optional[str],
        constraints: Optional[List[str]],
        max_steps: int,
    ) -> str:
        """Build the reasoning prompt."""
        prompt_parts = [
            f"Query: {query}\n",
            f"Reasoning Type: {reasoning_type.value}\n",
            f"Maximum Steps: {max_steps}\n",
        ]

        if context:
            prompt_parts.append(f"\nContext:\n{context}\n")

        if constraints:
            prompt_parts.append("\nConstraints:\n")
            for c in constraints:
                prompt_parts.append(f"- {c}\n")

        prompt_parts.append(
            "\nProvide your reasoning in clearly numbered steps. "
            "For each step, include:\n"
            "- Step description\n"
            "- Reasoning\n"
            "- Conclusion\n"
            "- Confidence (very_high/high/moderate/low/very_low)\n"
            "- Any assumptions made\n"
            "- Any uncertainties\n\n"
            "End with a final conclusion that synthesizes all steps. "
            "If this requires human judgment for value decisions, clearly state so."
        )

        return "".join(prompt_parts)

    def _build_trade_off_prompt(
        self,
        option: str,
        criteria: Optional[List[str]],
        context: Optional[str],
    ) -> str:
        """Build trade-off analysis prompt."""
        prompt = f"Analyze trade-offs for: {option}\n\n"

        if criteria:
            prompt += "Evaluation criteria:\n"
            for c in criteria:
                prompt += f"- {c}\n"
            prompt += "\n"

        if context:
            prompt += f"Context: {context}\n\n"

        prompt += (
            "Provide:\n"
            "1. List of pros (benefits)\n"
            "2. List of cons (drawbacks)\n"
            "3. Risk level (low/medium/high)\n"
            "4. Brief recommendation (what to consider, not what to choose)"
        )

        return prompt

    def _parse_reasoning_response(
        self,
        chain: ReasoningChain,
        response: str,
    ) -> ReasoningChain:
        """Parse LLM response into reasoning steps."""
        # Simple parsing - look for step patterns
        step_pattern = r"(?:Step|STEP)\s*(\d+)[:\.]?\s*(.*?)(?=(?:Step|STEP)\s*\d+|Final|FINAL|$)"
        matches = re.findall(step_pattern, response, re.DOTALL | re.IGNORECASE)

        for i, (num, content) in enumerate(matches):
            step = ReasoningStep(
                step_number=int(num) if num.isdigit() else i + 1,
                description=self._extract_field(content, "description", f"Step {num}"),
                reasoning=self._extract_field(content, "reasoning", content[:200]),
                conclusion=self._extract_field(content, "conclusion", "See reasoning above"),
                confidence=self._extract_confidence(content),
                assumptions=self._extract_list(content, "assumption"),
                uncertainties=self._extract_list(content, "uncertaint"),
            )
            chain.steps.append(step)

        # Extract final conclusion
        final_match = re.search(
            r"(?:Final\s*Conclusion|FINAL|Conclusion)[:\s]*(.*?)$",
            response,
            re.DOTALL | re.IGNORECASE,
        )
        if final_match:
            chain.final_conclusion = final_match.group(1).strip()
        elif chain.steps:
            chain.final_conclusion = chain.steps[-1].conclusion

        return chain

    def _extract_field(self, text: str, field_name: str, default: str) -> str:
        """Extract a named field from text."""
        pattern = rf"{field_name}[:\s]+(.*?)(?:\n\n|\n[A-Z]|$)"
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(1).strip()
        return default

    def _extract_confidence(self, text: str) -> ConfidenceLevel:
        """Extract confidence level from text."""
        text_lower = text.lower()

        if "very high" in text_lower or "very_high" in text_lower:
            return ConfidenceLevel.VERY_HIGH
        elif "very low" in text_lower or "very_low" in text_lower:
            return ConfidenceLevel.VERY_LOW
        elif "high" in text_lower:
            return ConfidenceLevel.HIGH
        elif "low" in text_lower:
            return ConfidenceLevel.LOW
        elif "moderate" in text_lower or "medium" in text_lower:
            return ConfidenceLevel.MODERATE
        else:
            return ConfidenceLevel.MODERATE

    def _extract_list(self, text: str, keyword: str) -> List[str]:
        """Extract list items related to a keyword."""
        items = []
        pattern = rf"{keyword}[s]?[:\s]*[-•]?\s*(.+?)(?:\n|$)"
        matches = re.findall(pattern, text, re.IGNORECASE)
        for match in matches:
            item = match.strip()
            if item and len(item) > 3:
                items.append(item)
        return items[:5]  # Limit to 5 items

    def _parse_trade_off_response(self, option: str, response: str) -> TradeOff:
        """Parse trade-off analysis from LLM response."""
        trade_off = TradeOff(option=option)

        # Extract pros
        pros_match = re.search(
            r"pros?[:\s]*(.*?)(?:cons?|risk|$)", response, re.IGNORECASE | re.DOTALL
        )
        if pros_match:
            pros_text = pros_match.group(1)
            trade_off.pros = [p.strip() for p in re.findall(r"[-•]\s*(.+)", pros_text)]

        # Extract cons
        cons_match = re.search(
            r"cons?[:\s]*(.*?)(?:risk|recommend|$)", response, re.IGNORECASE | re.DOTALL
        )
        if cons_match:
            cons_text = cons_match.group(1)
            trade_off.cons = [c.strip() for c in re.findall(r"[-•]\s*(.+)", cons_text)]

        # Extract risk level
        if "high risk" in response.lower():
            trade_off.risk_level = "high"
        elif "low risk" in response.lower():
            trade_off.risk_level = "low"
        else:
            trade_off.risk_level = "medium"

        # Extract recommendation
        rec_match = re.search(
            r"recommend[:\s]*(.*?)(?:\n\n|$)", response, re.IGNORECASE | re.DOTALL
        )
        if rec_match:
            trade_off.recommendation = rec_match.group(1).strip()

        return trade_off

    def _finalize_chain(self, chain: ReasoningChain) -> ReasoningChain:
        """Finalize and validate the reasoning chain."""
        # Evaluate overall confidence
        chain.overall_confidence = self.evaluate_confidence(chain)

        # Aggregate assumptions
        for step in chain.steps:
            for assumption in step.assumptions:
                if assumption not in chain.key_assumptions:
                    chain.key_assumptions.append(assumption)

        # Check for prohibited conclusions
        if chain.final_conclusion:
            for prohibited in self._config.prohibited_conclusions:
                if prohibited in chain.final_conclusion.lower():
                    chain.requires_human_judgment = True
                    chain.escalation_reason = (
                        "Conclusion contains language suggesting certainty that "
                        "exceeds what reasoning can establish."
                    )
                    # Soften the conclusion
                    chain.final_conclusion = chain.final_conclusion.replace(
                        "you must", "you might consider"
                    ).replace("the only answer", "one possible answer")

        return chain


def create_reasoning_engine(
    llm_callback: Optional[Callable[[str, Dict[str, Any]], str]] = None,
    temperature: float = 0.2,
    max_steps: int = 10,
) -> ReasoningEngine:
    """
    Create a reasoning engine.

    Args:
        llm_callback: Callback for LLM generation
        temperature: Reasoning temperature (lower = more focused)
        max_steps: Maximum reasoning steps

    Returns:
        Configured ReasoningEngine
    """
    config = ReasoningConfig(
        temperature=temperature,
        max_reasoning_steps=max_steps,
    )
    return ReasoningEngine(config=config, llm_callback=llm_callback)
