"""
Agent OS Sage Agent

The Reasoning Agent - provides deep chain-of-thought reasoning,
multi-step analysis, synthesis, and trade-off evaluation.

Named for its role as the wise counselor, Sage illuminates complex
problems without making decisions for humans.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.messaging.models import FlowRequest, FlowResponse, MessageStatus

from ..interface import (
    AgentCapabilities,
    AgentState,
    BaseAgent,
    CapabilityType,
    RequestValidationResult,
)
from .reasoning import (
    ReasoningChain,
    ReasoningEngine,
    ReasoningType,
    create_reasoning_engine,
)

# Optional Ollama integration
try:
    from ..ollama import OllamaClient, create_ollama_client

    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    OllamaClient = None

logger = logging.getLogger(__name__)


class SageConfig:
    """Configuration for Sage agent."""

    def __init__(
        self,
        model: str = "llama3:70b",
        fallback_model: str = "mistral:7b",
        temperature: float = 0.2,
        context_window: int = 32768,
        max_output_tokens: int = 4096,
        max_reasoning_steps: int = 10,
        require_explicit_steps: bool = True,
        ollama_endpoint: Optional[str] = None,
        use_mock: bool = False,
        prompt_template_path: Optional[Path] = None,
        constitution_path: Optional[Path] = None,
        **kwargs,
    ):
        self.model = model
        self.fallback_model = fallback_model
        self.temperature = temperature
        self.context_window = context_window
        self.max_output_tokens = max_output_tokens
        self.max_reasoning_steps = max_reasoning_steps
        self.require_explicit_steps = require_explicit_steps
        self.ollama_endpoint = ollama_endpoint
        self.use_mock = use_mock
        self.prompt_template_path = prompt_template_path
        self.constitution_path = constitution_path
        self.extra = kwargs

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SageConfig":
        """Create config from dictionary."""
        return cls(**data)


class SageAgent(BaseAgent):
    """
    Sage - The Reasoning Agent.

    Provides chain-of-thought reasoning for complex analysis:
    - Multi-step reasoning chains with explicit steps
    - Synthesis of information from multiple sources
    - Trade-off evaluation and comparison
    - Long-context reasoning
    - Constitutional compliance (never makes value judgments)

    Intents handled:
    - query.reasoning: Complex reasoning tasks
    - reasoning.analyze: Systematic analysis
    - reasoning.synthesize: Information synthesis
    - reasoning.evaluate: Trade-off evaluation
    - reasoning.compare: Option comparison
    """

    # Supported intents
    INTENTS = [
        "query.reasoning",
        "reasoning.analyze",
        "reasoning.synthesize",
        "reasoning.evaluate",
        "reasoning.compare",
        "reasoning.deduce",
        "reasoning.explain",
    ]

    # Intent to reasoning type mapping
    INTENT_MAP = {
        "query.reasoning": ReasoningType.ANALYSIS,
        "reasoning.analyze": ReasoningType.ANALYSIS,
        "reasoning.synthesize": ReasoningType.SYNTHESIS,
        "reasoning.evaluate": ReasoningType.EVALUATION,
        "reasoning.compare": ReasoningType.COMPARISON,
        "reasoning.deduce": ReasoningType.DEDUCTION,
        "reasoning.explain": ReasoningType.CAUSAL,
    }

    # Prohibited action patterns (from constitution)
    PROHIBITED_PATTERNS = [
        "make a decision for me",
        "tell me what to do",
        "you decide",
        "what should i",
        "give me the answer",
        "just tell me",
    ]

    def __init__(self, config: Optional[SageConfig] = None):
        """
        Initialize Sage agent.

        Args:
            config: Agent configuration
        """
        super().__init__(
            name="sage",
            description="Reasoning agent providing chain-of-thought analysis",
            version="0.1.0",
            capabilities={
                CapabilityType.REASONING,
            },
            supported_intents=self.INTENTS,
        )

        self._sage_config = config or SageConfig()
        self._reasoning_engine: Optional[ReasoningEngine] = None
        self._ollama_client: Optional[Any] = None
        self._model_loaded: bool = False
        self._system_prompt: str = ""

    def initialize(self, config: Dict[str, Any]) -> bool:
        """
        Initialize Sage with configuration.

        Args:
            config: Configuration dictionary

        Returns:
            True if initialization successful
        """
        self._do_initialize(config)

        try:
            # Merge configs
            merged_config = {**self._sage_config.__dict__, **config}
            self._sage_config = SageConfig.from_dict(merged_config)

            # Load system prompt
            self._load_system_prompt()

            # Initialize reasoning engine
            self._reasoning_engine = create_reasoning_engine(
                temperature=self._sage_config.temperature,
                max_steps=self._sage_config.max_reasoning_steps,
            )

            # Initialize Ollama client if available and not mocked
            if OLLAMA_AVAILABLE and not self._sage_config.use_mock:
                try:
                    self._ollama_client = create_ollama_client(
                        endpoint=self._sage_config.ollama_endpoint,
                        timeout=300.0,  # Long timeout for reasoning
                    )

                    # Check if model is available
                    if self._ollama_client.is_healthy():
                        if self._ollama_client.model_exists(self._sage_config.model):
                            self._model_loaded = True
                            logger.info(f"Using model: {self._sage_config.model}")
                        elif self._ollama_client.model_exists(self._sage_config.fallback_model):
                            self._sage_config.model = self._sage_config.fallback_model
                            self._model_loaded = True
                            logger.info(f"Using fallback model: {self._sage_config.fallback_model}")
                        else:
                            logger.warning("No suitable model found, using mock reasoning")

                    # Set LLM callback
                    if self._model_loaded:
                        self._reasoning_engine.set_llm_callback(self._llm_generate)

                except Exception as e:
                    logger.warning(f"Ollama not available: {e}, using mock reasoning")

            self._state = AgentState.READY
            logger.info("Sage agent initialized and ready")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize Sage: {e}")
            self._state = AgentState.ERROR
            return False

    def validate_request(self, request: FlowRequest) -> RequestValidationResult:
        """
        Validate incoming reasoning request.

        Args:
            request: Incoming FlowRequest

        Returns:
            RequestValidationResult
        """
        result = RequestValidationResult(is_valid=True)

        # Check intent is supported
        if request.intent not in self.INTENTS:
            result.add_error(f"Unsupported intent: {request.intent}")
            return result

        # Run parent validation (constitutional rules)
        parent_result = super().validate_request(request)
        if not parent_result.is_valid:
            return parent_result

        # Check for prohibited patterns (value judgments, decisions)
        prompt_lower = request.content.prompt.lower()
        for pattern in self.PROHIBITED_PATTERNS:
            if pattern in prompt_lower:
                result.requires_escalation = True
                result.escalation_reason = (
                    "This request asks for a decision or value judgment. "
                    "Sage provides analysis and reasoning, but decisions "
                    "must be made by humans. Would you like me to analyze "
                    "the options instead?"
                )
                break

        # Check for empty content
        if not request.content.prompt.strip():
            result.add_error("Reasoning query cannot be empty")

        return result

    def process(self, request: FlowRequest) -> FlowResponse:
        """
        Process a reasoning request.

        Args:
            request: Validated FlowRequest

        Returns:
            FlowResponse with reasoning result
        """
        intent = request.intent
        content = request.content.prompt
        metadata = request.content.metadata.model_dump() if request.content.metadata else {}

        try:
            # Get reasoning type from intent
            reasoning_type = self.INTENT_MAP.get(intent, ReasoningType.ANALYSIS)

            # Extract context if provided
            context = metadata.get("context")
            constraints = metadata.get("constraints", [])
            max_steps = metadata.get("max_steps", self._sage_config.max_reasoning_steps)

            # Perform reasoning
            chain = self._reasoning_engine.reason(
                query=content,
                reasoning_type=reasoning_type,
                context=context,
                constraints=constraints,
                max_steps=max_steps,
            )

            # Check if human judgment is required
            if chain.requires_human_judgment:
                return self._create_escalation_response(request, chain)

            # Format output
            output = chain.format_markdown()

            return request.create_response(
                source=self.name,
                status=MessageStatus.SUCCESS,
                output=output,
                reasoning=f"Chain-of-thought analysis: {len(chain.steps)} steps, "
                f"confidence: {chain.overall_confidence.value}",
            )

        except Exception as e:
            logger.exception(f"Error processing {intent}")
            return request.create_response(
                source=self.name,
                status=MessageStatus.ERROR,
                output="",
                errors=[str(e)],
            )

    def get_capabilities(self) -> AgentCapabilities:
        """Get Sage capabilities."""
        return AgentCapabilities(
            name=self.name,
            version=self._version,
            description=self._description,
            capabilities=self._capability_types,
            supported_intents=self.INTENTS,
            model=self._sage_config.model if self._sage_config else None,
            context_window=self._sage_config.context_window if self._sage_config else 32768,
            max_output_tokens=self._sage_config.max_output_tokens if self._sage_config else 4096,
            requires_constitution=True,
            requires_memory=False,
            can_escalate=True,
            metadata={
                "temperature": self._sage_config.temperature if self._sage_config else 0.2,
                "model_loaded": self._model_loaded,
                "ollama_available": OLLAMA_AVAILABLE,
            },
        )

    def shutdown(self) -> bool:
        """Shutdown Sage agent."""
        logger.info("Shutting down Sage agent")

        try:
            if self._ollama_client and hasattr(self._ollama_client, "close"):
                self._ollama_client.close()

            return self._do_shutdown()

        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
            return False

    # =========================================================================
    # Direct API Methods (for use without messaging)
    # =========================================================================

    def reason(
        self,
        query: str,
        reasoning_type: ReasoningType = ReasoningType.ANALYSIS,
        context: Optional[str] = None,
        constraints: Optional[List[str]] = None,
        max_steps: Optional[int] = None,
    ) -> ReasoningChain:
        """
        Perform reasoning directly.

        Args:
            query: The question or problem to reason about
            reasoning_type: Type of reasoning
            context: Additional context
            constraints: Constraints to apply
            max_steps: Maximum reasoning steps

        Returns:
            ReasoningChain with complete reasoning
        """
        if not self._reasoning_engine:
            raise RuntimeError("Sage not initialized")

        return self._reasoning_engine.reason(
            query=query,
            reasoning_type=reasoning_type,
            context=context,
            constraints=constraints,
            max_steps=max_steps,
        )

    def analyze(
        self,
        query: str,
        context: Optional[str] = None,
    ) -> ReasoningChain:
        """Perform analysis reasoning."""
        return self.reason(query, ReasoningType.ANALYSIS, context=context)

    def synthesize(
        self,
        query: str,
        sources: List[str],
    ) -> ReasoningChain:
        """Synthesize information from multiple sources."""
        context = "\n\n".join([f"Source {i+1}:\n{s}" for i, s in enumerate(sources)])
        return self.reason(query, ReasoningType.SYNTHESIS, context=context)

    def evaluate_options(
        self,
        query: str,
        options: List[str],
        criteria: Optional[List[str]] = None,
    ) -> ReasoningChain:
        """Evaluate trade-offs between options."""
        context = "Options to evaluate:\n" + "\n".join([f"- {o}" for o in options])
        if criteria:
            context += "\n\nCriteria:\n" + "\n".join([f"- {c}" for c in criteria])
        return self.reason(query, ReasoningType.EVALUATION, context=context)

    def compare(
        self,
        item_a: str,
        item_b: str,
        aspects: Optional[List[str]] = None,
    ) -> ReasoningChain:
        """Compare two items."""
        query = f"Compare and contrast: {item_a} vs {item_b}"
        context = None
        if aspects:
            context = "Aspects to compare:\n" + "\n".join([f"- {a}" for a in aspects])
        return self.reason(query, ReasoningType.COMPARISON, context=context)

    def get_statistics(self) -> Dict[str, Any]:
        """Get Sage statistics."""
        stats = {
            "agent": {
                "name": self.name,
                "state": self._state.name,
                "version": self._version,
            },
            "metrics": self._metrics.__dict__,
            "config": {
                "model": self._sage_config.model if self._sage_config else None,
                "temperature": self._sage_config.temperature if self._sage_config else None,
                "model_loaded": self._model_loaded,
            },
        }

        if self._reasoning_engine:
            stats["reasoning"] = self._reasoning_engine.get_metrics()

        return stats

    # =========================================================================
    # Private Methods
    # =========================================================================

    def _load_system_prompt(self) -> None:
        """Load system prompt with constitutional context.

        Loads:
        - Supreme CONSTITUTION.md (core governance rules)
        - Agent-specific constitution (agents/sage/constitution.md)
        - Base prompt (agents/sage/prompt.md or configured path)

        This ensures the LLM has full constitutional context in its prompt.
        """
        # Default fallback prompt
        fallback = (
            "You are Sage, a reasoning agent that performs rigorous chain-of-thought analysis. "
            "You analyze complex problems, showing your reasoning explicitly in numbered steps. "
            "You never make value judgments or decisions for humans - you illuminate options, "
            "trade-offs, and implications, but the human decides. "
            "Always acknowledge assumptions, uncertainties, and confidence levels."
        )

        # Check for configured custom prompt path first
        base_prompt = None
        if self._sage_config.prompt_template_path:
            try:
                base_prompt = self._sage_config.prompt_template_path.read_text()
            except Exception as e:
                logger.warning(f"Could not load configured prompt template: {e}")

        # If no custom prompt, use the standard method with constitutional context
        if base_prompt:
            # Custom prompt - still add constitutional context
            self._system_prompt = self.build_system_prompt_with_constitution(
                base_prompt=base_prompt,
                include_supreme=True,
            )
        else:
            # Use standard method which loads prompt.md + constitution
            self._system_prompt = self.get_full_system_prompt(
                include_constitution=True,
                include_supreme=True,
                fallback_prompt=fallback,
            )

        logger.info(
            f"Sage: Loaded system prompt with constitutional context ({len(self._system_prompt)} chars)"
        )

    def _llm_generate(self, prompt: str, options: Dict[str, Any]) -> str:
        """Generate response using Ollama LLM."""
        if not self._ollama_client or not self._model_loaded:
            return ""

        try:
            system = options.get("system", self._system_prompt)
            temperature = options.get("temperature", self._sage_config.temperature)
            max_tokens = options.get("max_tokens", self._sage_config.max_output_tokens)

            response = self._ollama_client.generate(
                model=self._sage_config.model,
                prompt=prompt,
                system=system,
                options={
                    "temperature": temperature,
                    "num_predict": max_tokens,
                    "num_ctx": self._sage_config.context_window,
                },
            )

            return response.content

        except Exception as e:
            logger.error(f"LLM generation error: {e}")
            return ""

    def _create_escalation_response(
        self,
        request: FlowRequest,
        chain: ReasoningChain,
    ) -> FlowResponse:
        """Create response for escalation to human."""
        # Include analysis but flag for human judgment
        output = chain.format_markdown()
        output += (
            "\n\n---\n\n"
            "## Human Judgment Required\n\n"
            f"{chain.escalation_reason}\n\n"
            "Sage provides analysis and illuminates options, but this situation "
            "requires human judgment to proceed. Please review the analysis above "
            "and make your decision."
        )

        response = request.create_response(
            source=self.name,
            status=MessageStatus.PARTIAL,
            output=output,
            reasoning=chain.escalation_reason,
        )

        response.next_actions.append(
            {
                "action": "escalate_to_human",
                "reason": chain.escalation_reason,
                "chain_id": chain.chain_id,
            }
        )

        return response


def create_sage_agent(
    model: str = "mistral:7b",
    use_mock: bool = True,
    **config_kwargs,
) -> SageAgent:
    """
    Create and initialize a Sage agent.

    Args:
        model: Model to use for reasoning
        use_mock: Use mock reasoning (no LLM)
        **config_kwargs: Additional configuration

    Returns:
        Initialized SageAgent
    """
    config = SageConfig(
        model=model,
        use_mock=use_mock,
        **config_kwargs,
    )

    agent = SageAgent(config)
    agent.initialize({})

    return agent
