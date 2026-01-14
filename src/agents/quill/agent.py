"""
Agent OS Quill Agent

The Writer Agent - provides document refinement, formatting, and polishing.
Preserves authorial intent while improving clarity and presentation.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from src.messaging.models import FlowRequest, FlowResponse, MessageStatus

from ..interface import (
    AgentCapabilities,
    AgentState,
    BaseAgent,
    CapabilityType,
    RequestValidationResult,
)
from .formatting import (
    FormattingEngine,
    OutputFormat,
    RefinementEngine,
    RefinementResult,
    create_formatting_engine,
    create_refinement_engine,
)

# Optional Ollama integration
try:
    from ..ollama import OllamaClient, create_ollama_client

    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    OllamaClient = None

logger = logging.getLogger(__name__)


class QuillConfig:
    """Configuration for Quill agent."""

    def __init__(
        self,
        model: str = "llama3:8b",
        fallback_model: str = "phi3:mini",
        temperature: float = 0.3,
        max_output_tokens: int = 4096,
        track_changes: bool = True,
        preserve_meaning: bool = True,
        default_format: OutputFormat = OutputFormat.MARKDOWN,
        default_template: Optional[str] = None,
        ollama_endpoint: Optional[str] = None,
        use_mock: bool = False,
        **kwargs,
    ):
        self.model = model
        self.fallback_model = fallback_model
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.track_changes = track_changes
        self.preserve_meaning = preserve_meaning
        self.default_format = default_format
        self.default_template = default_template
        self.ollama_endpoint = ollama_endpoint
        self.use_mock = use_mock
        self.extra = kwargs

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "QuillConfig":
        """Create config from dictionary."""
        # Handle format enum conversion
        if "default_format" in data and isinstance(data["default_format"], str):
            data["default_format"] = OutputFormat(data["default_format"])
        return cls(**data)


class QuillAgent(BaseAgent):
    """
    Quill - The Writer Agent.

    Provides document refinement and formatting:
    - Grammar, spelling, punctuation fixes
    - Style improvements while preserving voice
    - Template-based formatting
    - Multiple output formats (Markdown, JSON, plain text)
    - Change tracking with annotations

    Intents handled:
    - content.refine: Refine and improve text
    - content.format: Format document
    - content.technical: Technical documentation
    - creation.text: Text generation/refinement
    """

    # Supported intents
    INTENTS = [
        "content.refine",
        "content.format",
        "content.technical",
        "creation.text",
        "document.format",
        "document.polish",
        "document.template",
    ]

    # Prohibited patterns (from constitution)
    PROHIBITED_PATTERNS = [
        "change the meaning",
        "add your opinion",
        "make it say",
        "rewrite to argue",
        "misleading",
        "deceptive",
    ]

    def __init__(self, config: Optional[QuillConfig] = None):
        """
        Initialize Quill agent.

        Args:
            config: Agent configuration
        """
        super().__init__(
            name="quill",
            description="Writer agent providing document refinement and formatting",
            version="0.1.0",
            capabilities={
                CapabilityType.GENERATION,
            },
            supported_intents=self.INTENTS,
        )

        self._quill_config = config or QuillConfig()
        self._formatting_engine: Optional[FormattingEngine] = None
        self._refinement_engine: Optional[RefinementEngine] = None
        self._ollama_client: Optional[Any] = None
        self._model_loaded: bool = False
        self._system_prompt: str = ""

    def initialize(self, config: Dict[str, Any]) -> bool:
        """
        Initialize Quill with configuration.

        Args:
            config: Configuration dictionary

        Returns:
            True if initialization successful
        """
        self._do_initialize(config)

        try:
            # Merge configs
            merged_config = {**self._quill_config.__dict__, **config}
            self._quill_config = QuillConfig.from_dict(merged_config)

            # Load system prompt
            self._load_system_prompt()

            # Initialize engines
            self._formatting_engine = create_formatting_engine()
            self._refinement_engine = create_refinement_engine()

            # Initialize Ollama if available and not mocked
            if OLLAMA_AVAILABLE and not self._quill_config.use_mock:
                try:
                    self._ollama_client = create_ollama_client(
                        endpoint=self._quill_config.ollama_endpoint,
                        timeout=120.0,
                    )

                    if self._ollama_client.is_healthy():
                        if self._ollama_client.model_exists(self._quill_config.model):
                            self._model_loaded = True
                            logger.info(f"Using model: {self._quill_config.model}")
                        elif self._ollama_client.model_exists(self._quill_config.fallback_model):
                            self._quill_config.model = self._quill_config.fallback_model
                            self._model_loaded = True
                            logger.info(
                                f"Using fallback model: {self._quill_config.fallback_model}"
                            )

                    # Set LLM callback
                    if self._model_loaded:
                        self._refinement_engine.set_llm_callback(self._llm_generate)

                except Exception as e:
                    logger.warning(f"Ollama not available: {e}")

            self._state = AgentState.READY
            logger.info("Quill agent initialized and ready")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize Quill: {e}")
            self._state = AgentState.ERROR
            return False

    def validate_request(self, request: FlowRequest) -> RequestValidationResult:
        """
        Validate incoming refinement request.

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
                    f"Request may violate Quill's constitution: "
                    f"Cannot {pattern}. Quill refines form, not meaning."
                )
                return result

        # Check for empty content
        if not request.content.prompt.strip():
            result.add_error("Content to refine cannot be empty")

        return result

    def process(self, request: FlowRequest) -> FlowResponse:
        """
        Process a refinement request.

        Args:
            request: Validated FlowRequest

        Returns:
            FlowResponse with refined content
        """
        intent = request.intent
        content = request.content.prompt
        metadata = request.content.metadata.model_dump() if request.content.metadata else {}

        try:
            if intent in ("content.refine", "creation.text", "document.polish"):
                return self._handle_refine(request, content, metadata)

            elif intent in ("content.format", "document.format"):
                return self._handle_format(request, content, metadata)

            elif intent == "content.technical":
                return self._handle_technical(request, content, metadata)

            elif intent == "document.template":
                return self._handle_template(request, content, metadata)

            else:
                # Default to refinement
                return self._handle_refine(request, content, metadata)

        except Exception as e:
            logger.exception(f"Error processing {intent}")
            return request.create_response(
                source=self.name,
                status=MessageStatus.ERROR,
                output="",
                errors=[str(e)],
            )

    def get_capabilities(self) -> AgentCapabilities:
        """Get Quill capabilities."""
        return AgentCapabilities(
            name=self.name,
            version=self._version,
            description=self._description,
            capabilities=self._capability_types,
            supported_intents=self.INTENTS,
            model=self._quill_config.model if self._quill_config else None,
            max_output_tokens=self._quill_config.max_output_tokens if self._quill_config else 4096,
            requires_constitution=True,
            requires_memory=False,
            can_escalate=True,
            metadata={
                "default_format": (
                    self._quill_config.default_format.value if self._quill_config else "markdown"
                ),
                "track_changes": self._quill_config.track_changes if self._quill_config else True,
                "templates_available": (
                    self._formatting_engine.list_templates() if self._formatting_engine else []
                ),
                "model_loaded": self._model_loaded,
            },
        )

    def shutdown(self) -> bool:
        """Shutdown Quill agent."""
        logger.info("Shutting down Quill agent")

        try:
            if self._ollama_client and hasattr(self._ollama_client, "close"):
                self._ollama_client.close()

            return self._do_shutdown()

        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
            return False

    # =========================================================================
    # Direct API Methods
    # =========================================================================

    def refine(
        self,
        text: str,
        preserve_meaning: bool = True,
        track_changes: bool = True,
        style_guide: Optional[str] = None,
    ) -> RefinementResult:
        """
        Refine text directly.

        Args:
            text: Text to refine
            preserve_meaning: Ensure meaning is preserved
            track_changes: Track all changes
            style_guide: Optional style guide

        Returns:
            RefinementResult
        """
        if not self._refinement_engine:
            raise RuntimeError("Quill not initialized")

        return self._refinement_engine.refine(
            text=text,
            preserve_meaning=preserve_meaning,
            track_changes=track_changes,
            style_guide=style_guide,
        )

    def format_markdown(
        self,
        text: str,
        title: Optional[str] = None,
        template: Optional[str] = None,
    ) -> str:
        """Format text as Markdown."""
        if not self._formatting_engine:
            raise RuntimeError("Quill not initialized")

        return self._formatting_engine.format_as_markdown(
            text=text,
            title=title,
            template=template,
        )

    def format_json(self, data: Dict[str, Any], pretty: bool = True) -> str:
        """Format data as JSON."""
        if not self._formatting_engine:
            raise RuntimeError("Quill not initialized")

        return self._formatting_engine.format_as_json(data=data, pretty=pretty)

    def apply_template(
        self,
        template_name: str,
        content: Dict[str, str],
        metadata: Optional[Dict[str, str]] = None,
    ) -> str:
        """Apply a document template."""
        if not self._formatting_engine:
            raise RuntimeError("Quill not initialized")

        return self._formatting_engine.apply_template(
            template_name=template_name,
            content=content,
            metadata=metadata,
        )

    def list_templates(self) -> List[str]:
        """List available templates."""
        if not self._formatting_engine:
            return []
        return self._formatting_engine.list_templates()

    def get_statistics(self) -> Dict[str, Any]:
        """Get Quill statistics."""
        stats = {
            "agent": {
                "name": self.name,
                "state": self._state.name,
                "version": self._version,
            },
            "metrics": self._metrics.__dict__,
            "config": {
                "model": self._quill_config.model if self._quill_config else None,
                "model_loaded": self._model_loaded,
                "default_format": (
                    self._quill_config.default_format.value if self._quill_config else None
                ),
            },
        }

        if self._formatting_engine:
            stats["formatting"] = self._formatting_engine.get_metrics()

        if self._refinement_engine:
            stats["refinement"] = self._refinement_engine.get_metrics()

        return stats

    # =========================================================================
    # Private Methods
    # =========================================================================

    def _handle_refine(
        self,
        request: FlowRequest,
        content: str,
        metadata: Dict[str, Any],
    ) -> FlowResponse:
        """Handle content refinement."""
        preserve_meaning = metadata.get("preserve_meaning", self._quill_config.preserve_meaning)
        track_changes = metadata.get("track_changes", self._quill_config.track_changes)
        style_guide = metadata.get("style_guide")

        result = self.refine(
            text=content,
            preserve_meaning=preserve_meaning,
            track_changes=track_changes,
            style_guide=style_guide,
        )

        # Build output with changes if tracking
        if track_changes and result.changes:
            output = (
                "# Quill Refinement Result\n\n"
                "## Refined Text\n\n"
                f"{result.refined}\n\n"
                "---\n\n"
                f"{result.format_diff()}"
            )
        else:
            output = result.refined

        # Add flags if any
        if result.flags:
            output += "\n\n---\n\n## Flags for Review\n\n"
            for flag in result.flags:
                output += f"- ⚠️ {flag}\n"

        return request.create_response(
            source=self.name,
            status=MessageStatus.SUCCESS,
            output=output,
            reasoning=f"Made {result.change_count} changes in {result.processing_time_ms:.1f}ms",
        )

    def _handle_format(
        self,
        request: FlowRequest,
        content: str,
        metadata: Dict[str, Any],
    ) -> FlowResponse:
        """Handle document formatting."""
        format_type = metadata.get("format", self._quill_config.default_format.value)
        title = metadata.get("title")
        template = metadata.get("template")

        if format_type == "markdown" or format_type == OutputFormat.MARKDOWN.value:
            output = self.format_markdown(content, title=title, template=template)
        elif format_type == "json" or format_type == OutputFormat.JSON.value:
            # Try to parse as dict for JSON formatting
            try:
                import json

                data = json.loads(content)
                output = self.format_json(data)
            except json.JSONDecodeError:
                # Format as wrapped JSON
                output = self.format_json({"content": content})
        elif format_type == "plain" or format_type == OutputFormat.PLAIN_TEXT.value:
            output = self._formatting_engine.format_as_plain_text(content)
        else:
            output = content

        return request.create_response(
            source=self.name,
            status=MessageStatus.SUCCESS,
            output=output,
            reasoning=f"Formatted as {format_type}",
        )

    def _handle_technical(
        self,
        request: FlowRequest,
        content: str,
        metadata: Dict[str, Any],
    ) -> FlowResponse:
        """Handle technical documentation formatting."""
        # First refine for clarity
        result = self.refine(
            text=content,
            preserve_meaning=True,
            track_changes=True,
            style_guide="technical documentation",
        )

        # Then apply technical template if sections provided
        sections = metadata.get("sections")
        if sections:
            output = self.apply_template(
                template_name="technical",
                content=sections,
                metadata={"title": metadata.get("title", "Technical Documentation")},
            )
        else:
            # Format as markdown with technical styling
            output = self._formatting_engine.format_as_markdown(
                result.refined,
                title=metadata.get("title"),
                template="technical",
            )

        return request.create_response(
            source=self.name,
            status=MessageStatus.SUCCESS,
            output=output,
            reasoning=f"Technical documentation formatted with {result.change_count} refinements",
        )

    def _handle_template(
        self,
        request: FlowRequest,
        content: str,
        metadata: Dict[str, Any],
    ) -> FlowResponse:
        """Handle template application."""
        template_name = metadata.get("template", "report")

        # Check if template exists
        if template_name not in self.list_templates():
            return request.create_response(
                source=self.name,
                status=MessageStatus.ERROR,
                output="",
                errors=[f"Unknown template: {template_name}. Available: {self.list_templates()}"],
            )

        # Parse content as sections if JSON
        sections = metadata.get("sections")
        if not sections:
            # Try to parse content as JSON sections
            try:
                import json

                sections = json.loads(content)
            except json.JSONDecodeError:
                # Use content as single body section
                sections = {"Body": content}

        output = self.apply_template(
            template_name=template_name,
            content=sections,
            metadata={
                "title": metadata.get("title", "Document"),
                "author": metadata.get("author"),
                "date": metadata.get("date", datetime.now().strftime("%Y-%m-%d")),
            },
        )

        return request.create_response(
            source=self.name,
            status=MessageStatus.SUCCESS,
            output=output,
            reasoning=f"Applied '{template_name}' template",
        )

    def _load_system_prompt(self) -> None:
        """Load system prompt with constitutional context.

        Loads:
        - Supreme CONSTITUTION.md (core governance rules)
        - Agent-specific constitution (agents/quill/constitution.md)
        - Base prompt (agents/quill/prompt.md)

        This ensures the LLM has full constitutional context in its prompt.
        """
        # Default fallback prompt
        fallback = (
            "You are Quill, a document refinement agent. "
            "You improve grammar, spelling, punctuation, and style while "
            "strictly preserving the author's meaning and voice. "
            "You never add new content or change the intent. "
            "You format documents cleanly and consistently."
        )

        # Use the base class method to load prompt with constitutional context
        self._system_prompt = self.get_full_system_prompt(
            include_constitution=True,
            include_supreme=True,
            fallback_prompt=fallback,
        )

        logger.info(
            f"Quill: Loaded system prompt with constitutional context ({len(self._system_prompt)} chars)"
        )

    def _llm_generate(self, prompt: str, options: Dict[str, Any]) -> str:
        """Generate response using Ollama LLM."""
        if not self._ollama_client or not self._model_loaded:
            return ""

        try:
            system = options.get("system", self._system_prompt)
            temperature = options.get("temperature", self._quill_config.temperature)
            max_tokens = options.get("max_tokens", self._quill_config.max_output_tokens)

            response = self._ollama_client.generate(
                model=self._quill_config.model,
                prompt=prompt,
                system=system,
                options={
                    "temperature": temperature,
                    "num_predict": max_tokens,
                },
            )

            return response.content

        except Exception as e:
            logger.error(f"LLM generation error: {e}")
            return ""


def create_quill_agent(
    model: str = "llama3:8b",
    use_mock: bool = True,
    **config_kwargs,
) -> QuillAgent:
    """
    Create and initialize a Quill agent.

    Args:
        model: Model to use
        use_mock: Use mock mode (no LLM)
        **config_kwargs: Additional configuration

    Returns:
        Initialized QuillAgent
    """
    config = QuillConfig(
        model=model,
        use_mock=use_mock,
        **config_kwargs,
    )

    agent = QuillAgent(config)
    agent.initialize({})

    return agent
