"""
Muse Creative Generation Engine

Provides creative content generation with multiple styles and formats.
Supports stories, poems, scenarios, brainstorming, and artistic expressions.
All outputs are marked as drafts requiring human approval.
"""

import random
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Callable


class CreativeStyle(Enum):
    """Creative writing styles."""

    NARRATIVE = "narrative"
    POETIC = "poetic"
    DRAMATIC = "dramatic"
    WHIMSICAL = "whimsical"
    FORMAL = "formal"
    CONVERSATIONAL = "conversational"
    LYRICAL = "lyrical"
    MINIMALIST = "minimalist"


class ContentType(Enum):
    """Types of creative content."""

    STORY = "story"
    POEM = "poem"
    SCENARIO = "scenario"
    BRAINSTORM = "brainstorm"
    DIALOGUE = "dialogue"
    DESCRIPTION = "description"
    CONCEPT = "concept"
    METAPHOR = "metaphor"


class CreativeMode(Enum):
    """Creative generation modes."""

    EXPLORE = "explore"  # Wide exploration, many options
    REFINE = "refine"  # Focused refinement of ideas
    EXPAND = "expand"  # Expand on existing content
    COMBINE = "combine"  # Combine multiple ideas
    CONTRAST = "contrast"  # Generate contrasting alternatives


@dataclass
class CreativeOption:
    """A single creative output option."""

    content: str
    style: CreativeStyle
    content_type: ContentType
    confidence: float  # 0.0 to 1.0
    notes: str = ""
    variations: list[str] = field(default_factory=list)

    def is_draft(self) -> bool:
        """All creative outputs are drafts requiring approval."""
        return True

    def format_as_draft(self) -> str:
        """Format with draft marker."""
        return f"[DRAFT - Requires Human Approval]\n\n{self.content}\n\n[Style: {self.style.value} | Type: {self.content_type.value}]"


@dataclass
class CreativeResult:
    """Result of creative generation with multiple options."""

    prompt: str
    options: list[CreativeOption]
    mode: CreativeMode
    generated_at: datetime = field(default_factory=datetime.now)
    requires_review: bool = True
    review_notes: list[str] = field(default_factory=list)

    def get_primary(self) -> CreativeOption | None:
        """Get the primary (highest confidence) option."""
        if not self.options:
            return None
        return max(self.options, key=lambda o: o.confidence)

    def format_all_options(self) -> str:
        """Format all options for presentation."""
        lines = [
            f"Creative Options for: {self.prompt}",
            f"Mode: {self.mode.value}",
            f"Generated: {self.generated_at.isoformat()}",
            "",
            "=" * 50,
        ]

        for i, option in enumerate(self.options, 1):
            lines.extend(
                [
                    f"\n--- Option {i} (Confidence: {option.confidence:.0%}) ---",
                    option.format_as_draft(),
                ]
            )
            if option.notes:
                lines.append(f"Notes: {option.notes}")
            if option.variations:
                lines.append(f"Variations: {len(option.variations)} available")

        lines.extend(
            [
                "",
                "=" * 50,
                "[All options are drafts requiring human approval]",
            ]
        )

        return "\n".join(lines)


@dataclass
class CreativeConstraints:
    """Constraints for creative generation."""

    max_length: int = 2000
    min_length: int = 50
    allowed_styles: list[CreativeStyle] = field(default_factory=list)
    forbidden_themes: list[str] = field(default_factory=list)
    target_audience: str = "general"
    tone: str = "neutral"

    def validate(self, content: str) -> tuple[bool, list[str]]:
        """Validate content against constraints."""
        issues = []

        if len(content) > self.max_length:
            issues.append(f"Content exceeds max length ({len(content)} > {self.max_length})")

        if len(content) < self.min_length:
            issues.append(f"Content below min length ({len(content)} < {self.min_length})")

        # Check for forbidden themes (simple substring check)
        content_lower = content.lower()
        for theme in self.forbidden_themes:
            if theme.lower() in content_lower:
                issues.append(f"Contains forbidden theme: {theme}")

        return len(issues) == 0, issues


class CreativeEngine:
    """
    Engine for creative content generation.

    Generates stories, poems, scenarios, and other creative content
    with multiple style options and mandatory draft marking.
    """

    # Default creative temperature range
    MIN_TEMPERATURE = 0.7
    MAX_TEMPERATURE = 1.2
    DEFAULT_TEMPERATURE = 0.9

    # Style prompts for different creative styles
    STYLE_PROMPTS = {
        CreativeStyle.NARRATIVE: "Tell a story with vivid details and engaging narrative flow",
        CreativeStyle.POETIC: "Express with rhythm, imagery, and emotional resonance",
        CreativeStyle.DRAMATIC: "Create tension, conflict, and emotional impact",
        CreativeStyle.WHIMSICAL: "Embrace playfulness, surprise, and imaginative wonder",
        CreativeStyle.FORMAL: "Maintain elegance, precision, and refined expression",
        CreativeStyle.CONVERSATIONAL: "Write naturally, as if speaking to a friend",
        CreativeStyle.LYRICAL: "Flow like music with rhythm and melodic language",
        CreativeStyle.MINIMALIST: "Say more with less, embracing simplicity",
    }

    # Content type templates
    CONTENT_TEMPLATES = {
        ContentType.STORY: {
            "structure": ["opening", "rising_action", "climax", "resolution"],
            "elements": ["characters", "setting", "conflict", "theme"],
        },
        ContentType.POEM: {
            "forms": ["free_verse", "haiku", "sonnet", "limerick", "prose_poem"],
            "devices": ["metaphor", "imagery", "rhythm", "alliteration"],
        },
        ContentType.SCENARIO: {
            "components": ["context", "participants", "challenge", "possibilities"],
            "perspectives": ["first_person", "third_person", "omniscient"],
        },
        ContentType.BRAINSTORM: {
            "methods": ["mind_map", "list", "connections", "opposites"],
            "expansion": ["what_if", "combine", "transform", "eliminate"],
        },
        ContentType.DIALOGUE: {
            "elements": ["characters", "subtext", "conflict", "revelation"],
            "styles": ["realistic", "stylized", "rapid_fire", "monologue"],
        },
        ContentType.DESCRIPTION: {
            "senses": ["visual", "auditory", "tactile", "olfactory", "gustatory"],
            "techniques": ["showing", "comparison", "personification"],
        },
        ContentType.CONCEPT: {
            "aspects": ["core_idea", "implications", "connections", "applications"],
            "formats": ["abstract", "concrete", "hybrid"],
        },
        ContentType.METAPHOR: {
            "types": ["simile", "metaphor", "analogy", "allegory"],
            "sources": ["nature", "technology", "human_experience", "abstract"],
        },
    }

    def __init__(
        self,
        llm_provider: Callable[[str, float], str] | None = None,
        temperature: float = DEFAULT_TEMPERATURE,
        num_options: int = 3,
    ):
        """
        Initialize creative engine.

        Args:
            llm_provider: Function that takes (prompt, temperature) and returns text
            temperature: Base temperature for generation (0.7-1.2)
            num_options: Number of creative options to generate
        """
        self.llm_provider = llm_provider
        self.temperature = max(self.MIN_TEMPERATURE, min(self.MAX_TEMPERATURE, temperature))
        self.num_options = num_options
        self._generation_count = 0

    def generate(
        self,
        prompt: str,
        content_type: ContentType = ContentType.STORY,
        styles: list[CreativeStyle] | None = None,
        mode: CreativeMode = CreativeMode.EXPLORE,
        constraints: CreativeConstraints | None = None,
    ) -> CreativeResult:
        """
        Generate creative content with multiple options.

        Args:
            prompt: The creative prompt or seed
            content_type: Type of content to generate
            styles: Specific styles to use (or random selection)
            mode: Generation mode
            constraints: Optional constraints on output

        Returns:
            CreativeResult with multiple draft options
        """
        self._generation_count += 1

        # Select styles if not provided
        if styles is None:
            styles = self._select_styles(content_type, self.num_options)

        # Apply default constraints if none provided
        if constraints is None:
            constraints = CreativeConstraints()

        options = []
        review_notes = []

        for style in styles[: self.num_options]:
            # Generate content for this style
            content, notes = self._generate_for_style(
                prompt, content_type, style, mode, constraints
            )

            # Validate against constraints
            valid, issues = constraints.validate(content)
            if not valid:
                review_notes.extend([f"[{style.value}] {issue}" for issue in issues])

            # Calculate confidence based on various factors
            confidence = self._calculate_confidence(content, style, valid)

            option = CreativeOption(
                content=content,
                style=style,
                content_type=content_type,
                confidence=confidence,
                notes=notes,
                variations=(
                    self._generate_variations(content, style)
                    if mode == CreativeMode.EXPLORE
                    else []
                ),
            )
            options.append(option)

        return CreativeResult(
            prompt=prompt,
            options=options,
            mode=mode,
            requires_review=True,  # Always requires review
            review_notes=review_notes,
        )

    def generate_story(
        self,
        prompt: str,
        style: CreativeStyle = CreativeStyle.NARRATIVE,
    ) -> CreativeResult:
        """Generate a story with the given prompt."""
        return self.generate(prompt, ContentType.STORY, [style])

    def generate_poem(
        self,
        prompt: str,
        style: CreativeStyle = CreativeStyle.POETIC,
    ) -> CreativeResult:
        """Generate a poem with the given prompt."""
        return self.generate(prompt, ContentType.POEM, [style])

    def brainstorm(
        self,
        topic: str,
        num_ideas: int = 5,
    ) -> CreativeResult:
        """Brainstorm ideas on a topic."""
        result = self.generate(
            topic,
            ContentType.BRAINSTORM,
            [CreativeStyle.CONVERSATIONAL],
            mode=CreativeMode.EXPLORE,
        )
        return result

    def create_scenario(
        self,
        context: str,
        style: CreativeStyle = CreativeStyle.DRAMATIC,
    ) -> CreativeResult:
        """Create a scenario or situation."""
        return self.generate(context, ContentType.SCENARIO, [style])

    def expand(
        self,
        existing_content: str,
        direction: str = "",
    ) -> CreativeResult:
        """Expand on existing creative content."""
        prompt = f"Expand on: {existing_content}"
        if direction:
            prompt += f"\nDirection: {direction}"

        return self.generate(
            prompt,
            ContentType.CONCEPT,
            mode=CreativeMode.EXPAND,
        )

    def combine_ideas(
        self,
        ideas: list[str],
    ) -> CreativeResult:
        """Combine multiple ideas into new creative content."""
        prompt = "Combine these ideas:\n" + "\n".join(f"- {idea}" for idea in ideas)

        return self.generate(
            prompt,
            ContentType.CONCEPT,
            mode=CreativeMode.COMBINE,
        )

    def _select_styles(
        self,
        content_type: ContentType,
        count: int,
    ) -> list[CreativeStyle]:
        """Select appropriate styles for content type."""
        # Style affinities for different content types
        affinities = {
            ContentType.STORY: [
                CreativeStyle.NARRATIVE,
                CreativeStyle.DRAMATIC,
                CreativeStyle.WHIMSICAL,
            ],
            ContentType.POEM: [
                CreativeStyle.POETIC,
                CreativeStyle.LYRICAL,
                CreativeStyle.MINIMALIST,
            ],
            ContentType.SCENARIO: [
                CreativeStyle.DRAMATIC,
                CreativeStyle.NARRATIVE,
                CreativeStyle.FORMAL,
            ],
            ContentType.BRAINSTORM: [CreativeStyle.CONVERSATIONAL, CreativeStyle.WHIMSICAL],
            ContentType.DIALOGUE: [CreativeStyle.CONVERSATIONAL, CreativeStyle.DRAMATIC],
            ContentType.DESCRIPTION: [
                CreativeStyle.POETIC,
                CreativeStyle.NARRATIVE,
                CreativeStyle.MINIMALIST,
            ],
            ContentType.CONCEPT: [CreativeStyle.FORMAL, CreativeStyle.CONVERSATIONAL],
            ContentType.METAPHOR: [CreativeStyle.POETIC, CreativeStyle.LYRICAL],
        }

        preferred = affinities.get(content_type, list(CreativeStyle))

        # Ensure we have enough styles
        all_styles = list(CreativeStyle)
        result = preferred[:count]

        while len(result) < count:
            remaining = [s for s in all_styles if s not in result]
            if remaining:
                result.append(random.choice(remaining))
            else:
                break

        return result

    def _generate_for_style(
        self,
        prompt: str,
        content_type: ContentType,
        style: CreativeStyle,
        mode: CreativeMode,
        constraints: CreativeConstraints,
    ) -> tuple[str, str]:
        """Generate content for a specific style."""
        # Build generation prompt
        style_guidance = self.STYLE_PROMPTS.get(style, "")

        full_prompt = f"""
Creative Prompt: {prompt}
Content Type: {content_type.value}
Style: {style.value} - {style_guidance}
Mode: {mode.value}
Audience: {constraints.target_audience}
Tone: {constraints.tone}
Length: {constraints.min_length}-{constraints.max_length} characters

Generate creative content following these guidelines.
"""

        # Use LLM if available
        if self.llm_provider:
            try:
                content = self.llm_provider(full_prompt, self.temperature)
                notes = "Generated via LLM"
            except Exception as e:
                content = self._mock_generate(prompt, content_type, style)
                notes = f"Fallback generation (LLM error: {e})"
        else:
            content = self._mock_generate(prompt, content_type, style)
            notes = "Mock generation (no LLM provider)"

        return content, notes

    def _mock_generate(
        self,
        prompt: str,
        content_type: ContentType,
        style: CreativeStyle,
    ) -> str:
        """Generate mock creative content for testing."""
        templates = {
            ContentType.STORY: [
                "Once upon a time, {prompt} began a journey that would change everything.",
                "In the heart of an ancient forest, {prompt} discovered a secret.",
                "The day {prompt} arrived was unlike any other in the village.",
            ],
            ContentType.POEM: [
                "In twilight's gentle embrace,\n{prompt} finds its place,\nWhispers of what might be,\nDancing wild and free.",
                "Words like scattered leaves,\n{prompt} weaves and weaves,\nMeaning in the space between,\nThe unseen and seen.",
            ],
            ContentType.SCENARIO: [
                "Imagine a world where {prompt} is the norm. How would society adapt?",
                "Consider this: {prompt} has just become reality. What happens next?",
            ],
            ContentType.BRAINSTORM: [
                "Ideas around {prompt}:\n• Explore the unexpected angles\n• Connect to related concepts\n• Question assumptions\n• Imagine alternatives",
            ],
            ContentType.DIALOGUE: [
                '"Have you considered {prompt}?" asked the first.\n"Perhaps," came the reply, "but what if we looked deeper?"',
            ],
            ContentType.DESCRIPTION: [
                "The essence of {prompt} unfolds like morning mist, revealing layers of meaning.",
            ],
            ContentType.CONCEPT: [
                "At its core, {prompt} represents a bridge between the familiar and the unknown.",
            ],
            ContentType.METAPHOR: [
                "{prompt} is like a river finding its path—persistent, adaptive, ever-flowing.",
            ],
        }

        options = templates.get(content_type, ["Exploring: {prompt}"])
        template = random.choice(options)

        content = template.format(prompt=prompt)

        # Add style-specific flourishes
        if style == CreativeStyle.WHIMSICAL:
            content = f"✨ {content} ✨"
        elif style == CreativeStyle.FORMAL:
            content = content.replace("!", ".")
        elif style == CreativeStyle.MINIMALIST:
            # Keep it brief
            sentences = content.split(". ")
            content = ". ".join(sentences[:2]) + "."

        return content

    def _calculate_confidence(
        self,
        content: str,
        style: CreativeStyle,
        passed_constraints: bool,
    ) -> float:
        """Calculate confidence score for generated content."""
        base_confidence = 0.7

        # Adjust for constraint compliance
        if not passed_constraints:
            base_confidence -= 0.2

        # Adjust for content length (too short or too long reduces confidence)
        length = len(content)
        if length < 100:
            base_confidence -= 0.1
        elif length > 1500:
            base_confidence -= 0.05

        # Add some variability
        variance = random.uniform(-0.1, 0.1)

        return max(0.1, min(1.0, base_confidence + variance))

    def _generate_variations(
        self,
        content: str,
        style: CreativeStyle,
    ) -> list[str]:
        """Generate quick variations of the content."""
        variations = []

        # Simple variation: different emphasis
        words = content.split()
        if len(words) > 10:
            # Rearrange opening
            mid = len(words) // 2
            variation1 = " ".join(
                words[mid : mid + 5] + ["—"] + words[:5] + words[5:mid] + words[mid + 5 :]
            )
            variations.append(variation1[:200] + "...")

        return variations[:2]  # Max 2 variations


def create_creative_engine(
    llm_provider: Callable[[str, float], str] | None = None,
    temperature: float = 0.9,
    num_options: int = 3,
) -> CreativeEngine:
    """Factory function to create a creative engine."""
    return CreativeEngine(
        llm_provider=llm_provider,
        temperature=temperature,
        num_options=num_options,
    )
