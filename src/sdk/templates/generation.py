"""
Generation Agent Template

Provides a template for agents that generate content (text, code, etc.).
Includes support for templates, formatting, and quality checks.
"""

import logging
from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set

from src.agents.interface import CapabilityType
from src.messaging.models import FlowRequest, FlowResponse, MessageStatus

from .base import AgentTemplate, AgentConfig


logger = logging.getLogger(__name__)


@dataclass
class GenerationResult:
    """Result of a generation process."""
    content: str
    content_type: str = "text"  # text, code, markdown, json, etc.
    quality_score: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    alternatives: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "content": self.content,
            "content_type": self.content_type,
            "quality_score": self.quality_score,
            "metadata": self.metadata,
            "alternatives": self.alternatives,
        }


@dataclass
class GenerationConfig(AgentConfig):
    """Configuration for generation agents."""
    content_type: str = "text"
    max_length: int = 4096
    min_quality_score: float = 0.5
    generate_alternatives: bool = False
    num_alternatives: int = 2
    apply_formatting: bool = True
    quality_checks: List[str] = field(default_factory=list)

    def __post_init__(self):
        self.capabilities = self.capabilities or set()
        self.capabilities.add(CapabilityType.GENERATION)


class GenerationAgentTemplate(AgentTemplate):
    """
    Template for content generation agents.

    Provides:
    - Content generation with quality checks
    - Template-based generation
    - Multiple alternative generation
    - Output formatting
    - Length and quality controls
    """

    def __init__(self, config: GenerationConfig):
        """
        Initialize generation agent.

        Args:
            config: Generation agent configuration
        """
        super().__init__(config)
        self.generation_config = config
        self._templates: Dict[str, str] = {}
        self._quality_checkers: List[Callable[[str], float]] = []
        self._formatters: List[Callable[[str], str]] = []

    def register_template(self, name: str, template: str) -> "GenerationAgentTemplate":
        """
        Register a generation template.

        Templates can use {variable} syntax for substitution.

        Args:
            name: Template name
            template: Template string

        Returns:
            Self for chaining
        """
        self._templates[name] = template
        return self

    def add_quality_checker(
        self,
        checker: Callable[[str], float],
    ) -> "GenerationAgentTemplate":
        """
        Add a quality checker function.

        Checkers should return a score between 0 and 1.

        Args:
            checker: Checker function

        Returns:
            Self for chaining
        """
        self._quality_checkers.append(checker)
        return self

    def add_formatter(
        self,
        formatter: Callable[[str], str],
    ) -> "GenerationAgentTemplate":
        """
        Add an output formatter.

        Args:
            formatter: Formatter function

        Returns:
            Self for chaining
        """
        self._formatters.append(formatter)
        return self

    def handle_request(self, request: FlowRequest) -> FlowResponse:
        """Process generation request."""
        try:
            # Extract generation parameters
            params = self.extract_params(request)

            # Generate content
            result = self.generate(params, request)

            # Check quality
            if result.quality_score < self.generation_config.min_quality_score:
                logger.warning(
                    f"Generated content quality ({result.quality_score:.2f}) "
                    f"below threshold ({self.generation_config.min_quality_score:.2f})"
                )

            # Apply formatting
            content = result.content
            if self.generation_config.apply_formatting:
                content = self._apply_formatters(content)

            # Truncate if needed
            if len(content) > self.generation_config.max_length:
                content = content[:self.generation_config.max_length]
                content += "\n\n[Content truncated]"

            return request.create_response(
                source=self.name,
                status=MessageStatus.SUCCESS,
                output=content,
            )

        except Exception as e:
            logger.error(f"Generation error: {e}")
            return request.create_response(
                source=self.name,
                status=MessageStatus.ERROR,
                output="",
                errors=[f"Generation failed: {str(e)}"],
            )

    def extract_params(self, request: FlowRequest) -> Dict[str, Any]:
        """
        Extract generation parameters from request.

        Override for custom extraction.

        Args:
            request: The request

        Returns:
            Parameters dict
        """
        return {
            "prompt": request.content.prompt,
            "context": request.content.context,
        }

    @abstractmethod
    def generate(
        self,
        params: Dict[str, Any],
        request: FlowRequest,
    ) -> GenerationResult:
        """
        Generate content based on parameters.

        Override this method to implement generation logic.

        Args:
            params: Generation parameters
            request: Original request for context

        Returns:
            GenerationResult with content
        """
        pass

    def apply_template(
        self,
        template_name: str,
        variables: Dict[str, Any],
    ) -> str:
        """
        Apply a registered template.

        Args:
            template_name: Name of template
            variables: Variables to substitute

        Returns:
            Filled template string
        """
        template = self._templates.get(template_name)
        if not template:
            raise ValueError(f"Unknown template: {template_name}")

        try:
            return template.format(**variables)
        except KeyError as e:
            raise ValueError(f"Missing template variable: {e}")

    def check_quality(self, content: str) -> float:
        """
        Check content quality using registered checkers.

        Args:
            content: Content to check

        Returns:
            Average quality score (0-1)
        """
        if not self._quality_checkers:
            return 1.0

        scores = []
        for checker in self._quality_checkers:
            try:
                score = checker(content)
                scores.append(max(0.0, min(1.0, score)))
            except Exception as e:
                logger.warning(f"Quality checker error: {e}")

        return sum(scores) / len(scores) if scores else 1.0

    def _apply_formatters(self, content: str) -> str:
        """Apply all registered formatters."""
        for formatter in self._formatters:
            try:
                content = formatter(content)
            except Exception as e:
                logger.warning(f"Formatter error: {e}")
        return content


def create_generation_agent(
    name: str,
    generate_fn: Callable[[Dict[str, Any], FlowRequest], GenerationResult],
    description: str = "",
    content_type: str = "text",
    **kwargs,
) -> GenerationAgentTemplate:
    """
    Create a generation agent from a function.

    Example:
        def my_generator(params, request):
            prompt = params["prompt"]
            return GenerationResult(
                content=f"Generated response to: {prompt}",
                content_type="text",
                quality_score=0.9,
            )

        agent = create_generation_agent(
            name="writer",
            generate_fn=my_generator,
            description="Generates text content",
        )

    Args:
        name: Agent name
        generate_fn: Generation function
        description: Agent description
        content_type: Type of content generated
        **kwargs: Additional config options

    Returns:
        GenerationAgentTemplate instance
    """
    config = GenerationConfig(
        name=name,
        description=description,
        content_type=content_type,
        **kwargs,
    )

    class FunctionGenerationAgent(GenerationAgentTemplate):
        def generate(
            self,
            params: Dict[str, Any],
            request: FlowRequest,
        ) -> GenerationResult:
            return generate_fn(params, request)

    return FunctionGenerationAgent(config)
