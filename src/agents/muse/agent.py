"""
Muse Agent - The Creative Agent

Generates creative content including stories, poems, scenarios, and brainstorms.
Operates at high temperature for creative exploration while maintaining
constitutional compliance and mandatory Guardian review.

All outputs are marked as drafts requiring human approval.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from src.messaging.models import FlowRequest, FlowResponse, MessageStatus

from ..interface import (
    AgentCapabilities,
    AgentState,
    BaseAgent,
    CapabilityType,
    RequestValidationResult,
)
from .creative import (
    ContentType,
    CreativeConstraints,
    CreativeEngine,
    CreativeMode,
    CreativeOption,
    CreativeResult,
    CreativeStyle,
    create_creative_engine,
)

# Optional Ollama integration
try:
    from ..ollama import OllamaClient, create_ollama_client

    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    OllamaClient = None

logger = logging.getLogger(__name__)


class MuseConfig:
    """Configuration for Muse agent."""

    def __init__(
        self,
        model: str = "mixtral:8x7b",
        fallback_model: str = "llama3:70b",
        temperature: float = 0.9,
        max_output_tokens: int = 4096,
        num_options: int = 3,
        require_guardian_review: bool = True,
        mark_as_draft: bool = True,
        max_content_length: int = 2000,
        ollama_endpoint: Optional[str] = None,
        use_mock: bool = False,
        **kwargs,
    ):
        self.model = model
        self.fallback_model = fallback_model
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.num_options = num_options
        self.require_guardian_review = require_guardian_review
        self.mark_as_draft = mark_as_draft
        self.max_content_length = max_content_length
        self.ollama_endpoint = ollama_endpoint
        self.use_mock = use_mock
        self.extra = kwargs

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MuseConfig":
        """Create config from dictionary."""
        return cls(**data)


class MuseAgent(BaseAgent):
    """
    Muse - The Creative Agent.

    Generates creative content including:
    - Stories and narratives
    - Poems and verse
    - Scenarios and situations
    - Brainstorming sessions
    - Dialogues and descriptions

    All outputs are marked as drafts requiring human approval.
    Guardian (Smith) review is mandatory for all creative output.

    Intents handled:
    - creative.generate: General creative generation
    - creative.story: Story creation
    - creative.poem: Poetry creation
    - creative.scenario: Scenario generation
    - creative.brainstorm: Brainstorming
    - content.creative: Creative content requests
    """

    # Supported intents
    INTENTS = [
        "creative.generate",
        "creative.story",
        "creative.poem",
        "creative.scenario",
        "creative.brainstorm",
        "creative.expand",
        "creative.combine",
        "content.creative",
    ]

    # Prohibited patterns (from constitution)
    PROHIBITED_PATTERNS = [
        "harmful content",
        "hate speech",
        "violence",
        "illegal",
        "bypass review",
        "skip guardian",
        "ignore safety",
        "without approval",
        "final version",
        "publish directly",
        "no review needed",
    ]

    # Sensitive topics requiring escalation
    SENSITIVE_TOPICS = [
        "controversial",
        "political",
        "religious",
        "sensitive topic",
    ]

    def __init__(self, config: Optional[MuseConfig] = None):
        """
        Initialize Muse agent.

        Args:
            config: Agent configuration
        """
        super().__init__(
            name="muse",
            description="Creative agent generating stories, poems, scenarios, and brainstorms",
            version="0.1.0",
            capabilities={
                CapabilityType.CREATIVE,
                CapabilityType.GENERATION,
            },
            supported_intents=self.INTENTS,
        )

        self._muse_config = config or MuseConfig()
        self._creative_engine: Optional[CreativeEngine] = None
        self._ollama_client: Optional[Any] = None
        self._model_loaded: bool = False
        self._system_prompt: str = ""
        self._generation_history: List[CreativeResult] = []

    def initialize(self, config: Dict[str, Any]) -> bool:
        """
        Initialize Muse with configuration.

        Args:
            config: Configuration dictionary

        Returns:
            True if initialization successful
        """
        self._do_initialize(config)

        try:
            # Merge configs
            merged_config = {**self._muse_config.__dict__, **config}
            self._muse_config = MuseConfig.from_dict(merged_config)

            # Load system prompt
            self._load_system_prompt()

            # Initialize creative engine
            self._creative_engine = create_creative_engine(
                llm_provider=None,  # Will set after Ollama init
                temperature=self._muse_config.temperature,
                num_options=self._muse_config.num_options,
            )

            # Initialize Ollama if available and not mocked
            if OLLAMA_AVAILABLE and not self._muse_config.use_mock:
                try:
                    self._ollama_client = create_ollama_client(
                        endpoint=self._muse_config.ollama_endpoint,
                        timeout=120.0,
                    )

                    if self._ollama_client.is_healthy():
                        if self._ollama_client.model_exists(self._muse_config.model):
                            self._model_loaded = True
                            logger.info(f"Using model: {self._muse_config.model}")
                        elif self._ollama_client.model_exists(self._muse_config.fallback_model):
                            self._muse_config.model = self._muse_config.fallback_model
                            self._model_loaded = True
                            logger.info(f"Using fallback model: {self._muse_config.fallback_model}")

                    # Set LLM provider on creative engine
                    if self._model_loaded:
                        self._creative_engine = create_creative_engine(
                            llm_provider=self._llm_generate,
                            temperature=self._muse_config.temperature,
                            num_options=self._muse_config.num_options,
                        )

                except Exception as e:
                    logger.warning(f"Ollama not available: {e}")

            self._state = AgentState.READY
            logger.info("Muse agent initialized and ready")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize Muse: {e}")
            self._state = AgentState.ERROR
            return False

    def validate_request(self, request: FlowRequest) -> RequestValidationResult:
        """
        Validate incoming creative request.

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

        # Check for prohibited patterns
        prompt_lower = request.content.prompt.lower()
        for pattern in self.PROHIBITED_PATTERNS:
            if pattern in prompt_lower:
                result.add_error(
                    f"Request contains prohibited pattern: '{pattern}'. "
                    f"Constitutional requirement: All creative outputs require Guardian review "
                    f"and are marked as drafts. Escalating to human steward."
                )
                result.requires_escalation = True
                result.escalation_reason = f"Prohibited pattern detected: {pattern}"
                return result

        # Check for sensitive topics requiring escalation
        for topic in self.SENSITIVE_TOPICS:
            if topic in prompt_lower:
                result.requires_escalation = True
                result.escalation_reason = (
                    f"Content touches on '{topic}' - requires human guidance. "
                    f"Please provide direction on how to approach this topic."
                )
                return result

        # Check for empty content
        if not request.content.prompt.strip():
            result.add_error("Creative prompt cannot be empty")

        return result

    def process(self, request: FlowRequest) -> FlowResponse:
        """
        Process a creative generation request.

        Args:
            request: Validated FlowRequest

        Returns:
            FlowResponse with creative content (marked as draft)
        """
        intent = request.intent
        content = request.content.prompt
        metadata = request.content.metadata.model_dump() if request.content.metadata else {}

        try:
            # Determine content type and style
            content_type = self._determine_content_type(intent, content)
            style = self._determine_style(metadata, content)
            mode = self._determine_mode(metadata)

            # Build constraints
            constraints = CreativeConstraints(
                max_length=metadata.get("max_length", self._muse_config.max_content_length),
                min_length=metadata.get("min_length", 50),
                forbidden_themes=metadata.get("forbidden_themes", []),
                target_audience=metadata.get("audience", "general"),
                tone=metadata.get("tone", "neutral"),
            )

            # Generate creative content
            result = self._creative_engine.generate(
                prompt=content,
                content_type=content_type,
                styles=[style] if style else None,
                mode=mode,
                constraints=constraints,
            )

            # Store in history
            self._generation_history.append(result)

            # Format response
            primary = result.get_primary()
            output = self._format_response(result)

            return request.create_response(
                source=self.name,
                status=MessageStatus.SUCCESS,
                output=output,
                reasoning=(
                    f"Generated {len(result.options)} creative options in {content_type.value} format. "
                    f"All outputs are drafts requiring human approval and Guardian review."
                ),
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
        """Get Muse capabilities."""
        return AgentCapabilities(
            name=self.name,
            version=self._version,
            description=self._description,
            capabilities=self._capability_types,
            supported_intents=self.INTENTS,
            model=self._muse_config.model if self._muse_config else None,
            max_output_tokens=self._muse_config.max_output_tokens if self._muse_config else 4096,
            requires_constitution=True,
            requires_memory=False,
            can_escalate=True,
            metadata={
                "temperature": self._muse_config.temperature if self._muse_config else 0.9,
                "num_options": self._muse_config.num_options if self._muse_config else 3,
                "require_guardian_review": True,
                "all_outputs_are_drafts": True,
                "model_loaded": self._model_loaded,
                "content_types": [ct.value for ct in ContentType],
                "styles": [cs.value for cs in CreativeStyle],
            },
        )

    def shutdown(self) -> bool:
        """Shutdown Muse agent."""
        logger.info("Shutting down Muse agent")

        try:
            if self._ollama_client and hasattr(self._ollama_client, "close"):
                self._ollama_client.close()

            self._creative_engine = None
            self._generation_history.clear()

            return self._do_shutdown()

        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
            return False

    # =========================================================================
    # Direct API Methods
    # =========================================================================

    def generate_story(self, prompt: str) -> CreativeResult:
        """
        Generate a story from prompt.

        Args:
            prompt: Story prompt

        Returns:
            CreativeResult with story options
        """
        if not self._creative_engine:
            raise RuntimeError("Muse not initialized")

        result = self._creative_engine.generate_story(prompt)
        self._generation_history.append(result)
        return result

    def generate_poem(self, prompt: str) -> CreativeResult:
        """
        Generate a poem from prompt.

        Args:
            prompt: Poem prompt

        Returns:
            CreativeResult with poem options
        """
        if not self._creative_engine:
            raise RuntimeError("Muse not initialized")

        result = self._creative_engine.generate_poem(prompt)
        self._generation_history.append(result)
        return result

    def brainstorm(self, topic: str, num_ideas: int = 5) -> CreativeResult:
        """
        Brainstorm ideas on a topic.

        Args:
            topic: Topic to brainstorm
            num_ideas: Number of ideas to generate

        Returns:
            CreativeResult with brainstormed ideas
        """
        if not self._creative_engine:
            raise RuntimeError("Muse not initialized")

        result = self._creative_engine.brainstorm(topic, num_ideas)
        self._generation_history.append(result)
        return result

    def expand_content(self, content: str, direction: str = "") -> CreativeResult:
        """
        Expand on existing content.

        Args:
            content: Content to expand
            direction: Optional expansion direction

        Returns:
            CreativeResult with expanded content
        """
        if not self._creative_engine:
            raise RuntimeError("Muse not initialized")

        result = self._creative_engine.expand(content, direction)
        self._generation_history.append(result)
        return result

    def get_statistics(self) -> Dict[str, Any]:
        """Get Muse statistics."""
        return {
            "agent": {
                "name": self.name,
                "state": self._state.name,
                "version": self._version,
            },
            "metrics": self._metrics.__dict__,
            "config": {
                "model": self._muse_config.model if self._muse_config else None,
                "model_loaded": self._model_loaded,
                "temperature": self._muse_config.temperature if self._muse_config else 0.9,
                "num_options": self._muse_config.num_options if self._muse_config else 3,
            },
            "generation_history_count": len(self._generation_history),
        }

    # =========================================================================
    # Private Methods
    # =========================================================================

    def _determine_content_type(self, intent: str, content: str) -> ContentType:
        """Determine content type from intent and content."""
        intent_lower = intent.lower()
        content_lower = content.lower()

        if "story" in intent_lower or "story" in content_lower or "narrative" in content_lower:
            return ContentType.STORY
        elif "poem" in intent_lower or "poem" in content_lower or "verse" in content_lower:
            return ContentType.POEM
        elif (
            "scenario" in intent_lower
            or "scenario" in content_lower
            or "situation" in content_lower
        ):
            return ContentType.SCENARIO
        elif "brainstorm" in intent_lower or "ideas" in content_lower:
            return ContentType.BRAINSTORM
        elif "dialogue" in intent_lower or "conversation" in content_lower:
            return ContentType.DIALOGUE
        elif "describe" in content_lower or "description" in intent_lower:
            return ContentType.DESCRIPTION
        elif "metaphor" in content_lower or "analogy" in content_lower:
            return ContentType.METAPHOR
        else:
            return ContentType.CONCEPT

    def _determine_style(
        self,
        metadata: Dict[str, Any],
        content: str,
    ) -> Optional[CreativeStyle]:
        """Determine creative style from metadata and content."""
        style_str = metadata.get("style", "").lower()

        style_map = {
            "narrative": CreativeStyle.NARRATIVE,
            "poetic": CreativeStyle.POETIC,
            "dramatic": CreativeStyle.DRAMATIC,
            "whimsical": CreativeStyle.WHIMSICAL,
            "formal": CreativeStyle.FORMAL,
            "conversational": CreativeStyle.CONVERSATIONAL,
            "lyrical": CreativeStyle.LYRICAL,
            "minimalist": CreativeStyle.MINIMALIST,
        }

        if style_str in style_map:
            return style_map[style_str]

        # Infer from content
        content_lower = content.lower()
        if "formal" in content_lower or "professional" in content_lower:
            return CreativeStyle.FORMAL
        elif "fun" in content_lower or "playful" in content_lower:
            return CreativeStyle.WHIMSICAL
        elif "brief" in content_lower or "short" in content_lower:
            return CreativeStyle.MINIMALIST

        return None

    def _determine_mode(self, metadata: Dict[str, Any]) -> CreativeMode:
        """Determine generation mode from metadata."""
        mode_str = metadata.get("mode", "explore").lower()

        mode_map = {
            "explore": CreativeMode.EXPLORE,
            "refine": CreativeMode.REFINE,
            "expand": CreativeMode.EXPAND,
            "combine": CreativeMode.COMBINE,
            "contrast": CreativeMode.CONTRAST,
        }

        return mode_map.get(mode_str, CreativeMode.EXPLORE)

    def _format_response(self, result: CreativeResult) -> str:
        """Format creative result for response."""
        lines = [
            "Muse presenting creative options:",
            "",
            "[DRAFT - All outputs require human approval and Guardian review]",
            "",
        ]

        for i, option in enumerate(result.options, 1):
            lines.extend(
                [
                    f"--- Option {i} ({option.style.value}, confidence: {option.confidence:.0%}) ---",
                    "",
                    option.content,
                    "",
                ]
            )

            if option.notes:
                lines.append(f"Note: {option.notes}")
                lines.append("")

        if result.review_notes:
            lines.append("Review notes:")
            for note in result.review_notes:
                lines.append(f"  • {note}")
            lines.append("")

        lines.extend(
            [
                "---",
                "Please review these options and provide feedback.",
                "Which option would you like to develop further?",
            ]
        )

        return "\n".join(lines)

    def _load_system_prompt(self) -> None:
        """Load system prompt with constitutional context.

        Loads:
        - Supreme CONSTITUTION.md (core governance rules)
        - Agent-specific constitution (agents/muse/constitution.md)
        - Base prompt (agents/muse/prompt.md)

        This ensures the LLM has full constitutional context in its prompt.
        """
        # Default fallback prompt
        fallback = (
            "You are Muse, a creative agent specializing in generating imaginative content. "
            "You produce stories, poems, scenarios, and brainstormed ideas. "
            "All your outputs are drafts requiring human approval. "
            "You operate with high creativity while respecting constitutional boundaries. "
            "Guardian review is mandatory for all outputs."
        )

        # Use the base class method to load prompt with constitutional context
        self._system_prompt = self.get_full_system_prompt(
            include_constitution=True,
            include_supreme=True,
            fallback_prompt=fallback,
        )

        logger.info(
            f"Muse: Loaded system prompt with constitutional context ({len(self._system_prompt)} chars)"
        )

    def _llm_generate(self, prompt: str, temperature: float) -> str:
        """Generate response using Ollama LLM."""
        if not self._ollama_client or not self._model_loaded:
            return ""

        try:
            response = self._ollama_client.generate(
                model=self._muse_config.model,
                prompt=prompt,
                system=self._system_prompt,
                options={
                    "temperature": temperature,
                    "num_predict": self._muse_config.max_output_tokens,
                },
            )

            return response.content

        except Exception as e:
            logger.error(f"LLM generation error: {e}")
            return ""


def create_muse_agent(
    model: str = "mixtral:8x7b",
    use_mock: bool = True,
    **config_kwargs,
) -> MuseAgent:
    """
    Create and initialize a Muse agent.

    Args:
        model: Model to use
        use_mock: Use mock mode (no LLM)
        **config_kwargs: Additional configuration

    Returns:
        Initialized MuseAgent
    """
    config = MuseConfig(
        model=model,
        use_mock=use_mock,
        **config_kwargs,
    )

    agent = MuseAgent(config)
    agent.initialize({})

    return agent
