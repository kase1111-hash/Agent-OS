"""
Agent OS Quill Formatting Engine

Provides document formatting, templates, and structured output generation.
Supports Markdown, JSON, and plain text formats with change tracking.
"""

import json
import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class OutputFormat(Enum):
    """Supported output formats."""

    MARKDOWN = "markdown"
    JSON = "json"
    PLAIN_TEXT = "plain_text"
    HTML = "html"


class ChangeType(Enum):
    """Types of changes in text refinement."""

    GRAMMAR = "grammar"
    SPELLING = "spelling"
    PUNCTUATION = "punctuation"
    STYLE = "style"
    FORMATTING = "formatting"
    STRUCTURE = "structure"
    CLARIFICATION = "clarification"


@dataclass
class TextChange:
    """A tracked change in the document."""

    change_type: ChangeType
    original: str
    refined: str
    location: int  # Character position
    reason: str = ""
    accepted: Optional[bool] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "change_type": self.change_type.value,
            "original": self.original,
            "refined": self.refined,
            "location": self.location,
            "reason": self.reason,
            "accepted": self.accepted,
        }


@dataclass
class RefinementResult:
    """Result of text refinement."""

    original: str
    refined: str
    changes: List[TextChange] = field(default_factory=list)
    format_applied: OutputFormat = OutputFormat.PLAIN_TEXT
    template_used: Optional[str] = None
    flags: List[str] = field(default_factory=list)
    processing_time_ms: float = 0.0

    @property
    def change_count(self) -> int:
        return len(self.changes)

    @property
    def changes_by_type(self) -> Dict[str, int]:
        counts = {}
        for change in self.changes:
            key = change.change_type.value
            counts[key] = counts.get(key, 0) + 1
        return counts

    def to_dict(self) -> Dict[str, Any]:
        return {
            "original": self.original,
            "refined": self.refined,
            "changes": [c.to_dict() for c in self.changes],
            "change_count": self.change_count,
            "changes_by_type": self.changes_by_type,
            "format_applied": self.format_applied.value,
            "template_used": self.template_used,
            "flags": self.flags,
            "processing_time_ms": self.processing_time_ms,
        }

    def format_diff(self) -> str:
        """Format changes as a diff view."""
        lines = ["## Changes Made", ""]

        if not self.changes:
            lines.append("No changes were necessary.")
            return "\n".join(lines)

        for i, change in enumerate(self.changes, 1):
            lines.append(f"### Change {i}: {change.change_type.value.title()}")
            lines.append(f"- **Original:** `{change.original}`")
            lines.append(f"- **Refined:** `{change.refined}`")
            if change.reason:
                lines.append(f"- **Reason:** {change.reason}")
            lines.append("")

        return "\n".join(lines)

    def format_annotated(self) -> str:
        """Format with inline annotations."""
        if not self.changes:
            return self.refined

        # Sort changes by location (reverse for safe replacement)
        # TODO: Use sorted_changes for annotation placement
        _sorted_changes = sorted(self.changes, key=lambda c: c.location, reverse=True)

        annotated = self.refined
        # Note: This is a simplified version - full implementation would
        # track positions more carefully
        return annotated


@dataclass
class DocumentTemplate:
    """A document formatting template."""

    name: str
    description: str
    format: OutputFormat
    structure: List[str]  # Section names
    style_rules: Dict[str, Any] = field(default_factory=dict)
    placeholders: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "format": self.format.value,
            "structure": self.structure,
            "style_rules": self.style_rules,
            "placeholders": self.placeholders,
        }


# Default templates
DEFAULT_TEMPLATES = {
    "report": DocumentTemplate(
        name="report",
        description="Standard business report format",
        format=OutputFormat.MARKDOWN,
        structure=[
            "Executive Summary",
            "Introduction",
            "Findings",
            "Analysis",
            "Recommendations",
            "Conclusion",
        ],
        style_rules={
            "heading_style": "atx",
            "list_style": "dash",
            "emphasis": "bold_for_key_terms",
        },
        placeholders={
            "title": "Report Title",
            "author": "Author Name",
            "date": "{{date}}",
        },
    ),
    "memo": DocumentTemplate(
        name="memo",
        description="Internal memo format",
        format=OutputFormat.MARKDOWN,
        structure=["To", "From", "Date", "Subject", "Body"],
        style_rules={
            "heading_style": "bold",
            "concise": True,
        },
    ),
    "technical": DocumentTemplate(
        name="technical",
        description="Technical documentation format",
        format=OutputFormat.MARKDOWN,
        structure=[
            "Overview",
            "Requirements",
            "Implementation",
            "API Reference",
            "Examples",
            "Troubleshooting",
        ],
        style_rules={
            "heading_style": "atx",
            "code_blocks": "fenced",
            "list_style": "numbered_for_steps",
        },
    ),
    "email": DocumentTemplate(
        name="email",
        description="Professional email format",
        format=OutputFormat.PLAIN_TEXT,
        structure=["Greeting", "Opening", "Body", "Closing", "Signature"],
        style_rules={
            "tone": "professional",
            "length": "concise",
        },
    ),
    "creative": DocumentTemplate(
        name="creative",
        description="Creative writing format",
        format=OutputFormat.MARKDOWN,
        structure=["Title", "Content"],
        style_rules={
            "preserve_voice": True,
            "minimal_formatting": True,
        },
    ),
}


class FormattingEngine:
    """
    Document formatting engine.

    Provides:
    - Text formatting and structure
    - Template application
    - Multiple output formats
    - Consistent styling
    """

    def __init__(self, templates: Optional[Dict[str, DocumentTemplate]] = None):
        """
        Initialize formatting engine.

        Args:
            templates: Custom templates (merged with defaults)
        """
        self._templates = {**DEFAULT_TEMPLATES}
        if templates:
            self._templates.update(templates)

        # Metrics
        self._formats_applied = 0

    def format_as_markdown(
        self,
        text: str,
        template: Optional[str] = None,
        title: Optional[str] = None,
        sections: Optional[Dict[str, str]] = None,
    ) -> str:
        """
        Format text as Markdown.

        Args:
            text: Raw text to format
            template: Template name to apply
            title: Document title
            sections: Section name -> content mapping

        Returns:
            Formatted Markdown string
        """
        lines = []

        # Add title
        if title:
            lines.append(f"# {title}")
            lines.append("")

        # Apply template structure if provided
        if template and template in self._templates:
            tmpl = self._templates[template]
            if sections:
                for section_name in tmpl.structure:
                    if section_name in sections:
                        lines.append(f"## {section_name}")
                        lines.append("")
                        lines.append(sections[section_name])
                        lines.append("")
            else:
                # Just add text with basic formatting
                lines.append(self._format_markdown_text(text))
        else:
            lines.append(self._format_markdown_text(text))

        self._formats_applied += 1
        return "\n".join(lines)

    def format_as_json(
        self,
        data: Dict[str, Any],
        pretty: bool = True,
    ) -> str:
        """
        Format data as JSON.

        Args:
            data: Data to format
            pretty: Use indentation

        Returns:
            JSON string
        """
        self._formats_applied += 1
        if pretty:
            return json.dumps(data, indent=2, ensure_ascii=False)
        return json.dumps(data, ensure_ascii=False)

    def format_as_plain_text(
        self,
        text: str,
        line_width: int = 80,
        preserve_paragraphs: bool = True,
    ) -> str:
        """
        Format as clean plain text.

        Args:
            text: Text to format
            line_width: Maximum line width
            preserve_paragraphs: Keep paragraph breaks

        Returns:
            Formatted plain text
        """
        # Remove markdown formatting
        clean = self._strip_markdown(text)

        # Normalize whitespace
        if preserve_paragraphs:
            paragraphs = re.split(r"\n\s*\n", clean)
            formatted_paragraphs = []
            for p in paragraphs:
                wrapped = self._wrap_text(p.strip(), line_width)
                formatted_paragraphs.append(wrapped)
            clean = "\n\n".join(formatted_paragraphs)
        else:
            clean = " ".join(clean.split())
            clean = self._wrap_text(clean, line_width)

        self._formats_applied += 1
        return clean

    def apply_template(
        self,
        template_name: str,
        content: Dict[str, str],
        metadata: Optional[Dict[str, str]] = None,
    ) -> str:
        """
        Apply a template to content.

        Args:
            template_name: Name of template
            content: Section name -> content mapping
            metadata: Additional metadata (title, author, etc.)

        Returns:
            Formatted document
        """
        if template_name not in self._templates:
            raise ValueError(f"Unknown template: {template_name}")

        tmpl = self._templates[template_name]
        metadata = metadata or {}

        lines = []

        # Add metadata header
        if metadata.get("title"):
            lines.append(f"# {metadata['title']}")
            lines.append("")

        if metadata.get("author") or metadata.get("date"):
            if metadata.get("author"):
                lines.append(f"*Author: {metadata['author']}*")
            if metadata.get("date"):
                lines.append(f"*Date: {metadata['date']}*")
            lines.append("")
            lines.append("---")
            lines.append("")

        # Add sections in template order
        for section_name in tmpl.structure:
            if section_name in content:
                lines.append(f"## {section_name}")
                lines.append("")
                lines.append(content[section_name])
                lines.append("")

        # Add any extra sections not in template
        for section_name, section_content in content.items():
            if section_name not in tmpl.structure:
                lines.append(f"## {section_name}")
                lines.append("")
                lines.append(section_content)
                lines.append("")

        self._formats_applied += 1
        return "\n".join(lines)

    def get_template(self, name: str) -> Optional[DocumentTemplate]:
        """Get a template by name."""
        return self._templates.get(name)

    def list_templates(self) -> List[str]:
        """List available template names."""
        return list(self._templates.keys())

    def add_template(self, template: DocumentTemplate) -> None:
        """Add a custom template."""
        self._templates[template.name] = template

    def get_metrics(self) -> Dict[str, Any]:
        """Get formatting metrics."""
        return {
            "formats_applied": self._formats_applied,
            "templates_available": len(self._templates),
        }

    def _format_markdown_text(self, text: str) -> str:
        """Apply markdown formatting to text."""
        lines = text.split("\n")
        formatted = []

        for line in lines:
            line = line.rstrip()

            # Detect and format lists
            if re.match(r"^\s*[-*]\s+", line):
                formatted.append(line)
            elif re.match(r"^\s*\d+\.\s+", line):
                formatted.append(line)
            # Detect headers
            elif re.match(r"^#+\s+", line):
                formatted.append(line)
            else:
                formatted.append(line)

        return "\n".join(formatted)

    def _strip_markdown(self, text: str) -> str:
        """Remove markdown formatting."""
        # Remove headers
        text = re.sub(r"^#+\s+", "", text, flags=re.MULTILINE)
        # Remove bold/italic
        text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
        text = re.sub(r"\*(.+?)\*", r"\1", text)
        text = re.sub(r"__(.+?)__", r"\1", text)
        text = re.sub(r"_(.+?)_", r"\1", text)
        # Remove links
        text = re.sub(r"\[(.+?)\]\(.+?\)", r"\1", text)
        # Remove code
        text = re.sub(r"`(.+?)`", r"\1", text)
        return text

    def _wrap_text(self, text: str, width: int) -> str:
        """Wrap text to specified width."""
        words = text.split()
        lines = []
        current_line = []
        current_length = 0

        for word in words:
            if current_length + len(word) + 1 <= width:
                current_line.append(word)
                current_length += len(word) + 1
            else:
                if current_line:
                    lines.append(" ".join(current_line))
                current_line = [word]
                current_length = len(word)

        if current_line:
            lines.append(" ".join(current_line))

        return "\n".join(lines)


class RefinementEngine:
    """
    Text refinement engine.

    Provides:
    - Grammar and spelling corrections
    - Style improvements
    - Change tracking
    - Preservation of meaning
    """

    # Common corrections (simplified - real implementation would use NLP)
    COMMON_FIXES = {
        "its'": "its",
        "it's" + " own": "its own",
        "their" + " is": "there is",
        "your" + " welcome": "you're welcome",
        "could of": "could have",
        "would of": "would have",
        "should of": "should have",
        "alot": "a lot",
        "seperate": "separate",
        "occured": "occurred",
        "recieve": "receive",
        "beleive": "believe",
        "definately": "definitely",
        "occassion": "occasion",
        "accomodate": "accommodate",
        "untill": "until",
        "begining": "beginning",
    }

    # Patterns that indicate potential issues
    ISSUE_PATTERNS = [
        (r"\b(\w+)\s+\1\b", "Repeated word"),
        (r"[.!?]\s*[a-z]", "Missing capitalization after sentence"),
        (r"\s+[,.!?]", "Space before punctuation"),
        (r"[^\s]\s{2,}[^\s]", "Multiple spaces"),
    ]

    def __init__(
        self,
        llm_callback: Optional[Callable[[str, Dict[str, Any]], str]] = None,
    ):
        """
        Initialize refinement engine.

        Args:
            llm_callback: Callback for LLM-based refinement
        """
        self._llm_callback = llm_callback

        # Metrics
        self._texts_refined = 0
        self._total_changes = 0

    def set_llm_callback(self, callback: Callable[[str, Dict[str, Any]], str]) -> None:
        """Set the LLM callback."""
        self._llm_callback = callback

    def refine(
        self,
        text: str,
        preserve_meaning: bool = True,
        track_changes: bool = True,
        style_guide: Optional[str] = None,
    ) -> RefinementResult:
        """
        Refine text with grammar, spelling, and style improvements.

        Args:
            text: Text to refine
            preserve_meaning: Ensure meaning is preserved
            track_changes: Track all changes
            style_guide: Optional style guide to follow

        Returns:
            RefinementResult with refined text and changes
        """
        import time

        start_time = time.time()

        changes = []
        refined = text
        flags = []

        # Apply common fixes
        for wrong, correct in self.COMMON_FIXES.items():
            if wrong in refined.lower():
                # Find actual case
                pattern = re.compile(re.escape(wrong), re.IGNORECASE)
                for match in pattern.finditer(refined):
                    original = match.group()
                    # Preserve case
                    if original[0].isupper():
                        replacement = correct.capitalize()
                    else:
                        replacement = correct

                    if track_changes:
                        changes.append(
                            TextChange(
                                change_type=ChangeType.SPELLING,
                                original=original,
                                refined=replacement,
                                location=match.start(),
                                reason="Common spelling/grammar fix",
                            )
                        )

                refined = pattern.sub(correct, refined)

        # Fix punctuation spacing
        refined, punct_changes = self._fix_punctuation(refined, track_changes)
        changes.extend(punct_changes)

        # Fix capitalization
        refined, cap_changes = self._fix_capitalization(refined, track_changes)
        changes.extend(cap_changes)

        # Fix repeated words
        refined, repeat_changes = self._fix_repeated_words(refined, track_changes)
        changes.extend(repeat_changes)

        # Normalize whitespace
        refined, ws_changes = self._normalize_whitespace(refined, track_changes)
        changes.extend(ws_changes)

        # Check for issues that need human attention
        flags = self._check_for_flags(refined)

        # If LLM is available, use it for deeper refinement
        if self._llm_callback:
            refined, llm_changes = self._llm_refine(refined, preserve_meaning, style_guide)
            changes.extend(llm_changes)

        processing_time = (time.time() - start_time) * 1000

        self._texts_refined += 1
        self._total_changes += len(changes)

        return RefinementResult(
            original=text,
            refined=refined,
            changes=changes,
            flags=flags,
            processing_time_ms=processing_time,
        )

    def check_grammar(self, text: str) -> List[Tuple[str, str, int]]:
        """
        Check text for grammar issues.

        Args:
            text: Text to check

        Returns:
            List of (issue, suggestion, position)
        """
        issues = []

        for pattern, description in self.ISSUE_PATTERNS:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                issues.append(
                    (
                        description,
                        match.group(),
                        match.start(),
                    )
                )

        return issues

    def get_metrics(self) -> Dict[str, Any]:
        """Get refinement metrics."""
        return {
            "texts_refined": self._texts_refined,
            "total_changes": self._total_changes,
            "average_changes_per_text": (
                self._total_changes / self._texts_refined if self._texts_refined > 0 else 0
            ),
        }

    def _fix_punctuation(
        self,
        text: str,
        track: bool,
    ) -> Tuple[str, List[TextChange]]:
        """Fix punctuation spacing."""
        changes = []

        # Remove space before punctuation
        pattern = r"\s+([,.!?;:])"
        for match in re.finditer(pattern, text):
            if track:
                changes.append(
                    TextChange(
                        change_type=ChangeType.PUNCTUATION,
                        original=match.group(),
                        refined=match.group(1),
                        location=match.start(),
                        reason="Remove space before punctuation",
                    )
                )
        text = re.sub(pattern, r"\1", text)

        # Ensure space after punctuation (except at end)
        pattern = r"([,.!?;:])([A-Za-z])"
        for match in re.finditer(pattern, text):
            if track:
                changes.append(
                    TextChange(
                        change_type=ChangeType.PUNCTUATION,
                        original=match.group(),
                        refined=f"{match.group(1)} {match.group(2)}",
                        location=match.start(),
                        reason="Add space after punctuation",
                    )
                )
        text = re.sub(pattern, r"\1 \2", text)

        return text, changes

    def _fix_capitalization(
        self,
        text: str,
        track: bool,
    ) -> Tuple[str, List[TextChange]]:
        """Fix capitalization issues."""
        changes = []

        # Capitalize after sentence endings
        pattern = r"([.!?]\s+)([a-z])"

        def capitalize_match(m):
            return m.group(1) + m.group(2).upper()

        for match in re.finditer(pattern, text):
            if track:
                changes.append(
                    TextChange(
                        change_type=ChangeType.GRAMMAR,
                        original=match.group(),
                        refined=match.group(1) + match.group(2).upper(),
                        location=match.start(),
                        reason="Capitalize first letter after sentence",
                    )
                )

        text = re.sub(pattern, capitalize_match, text)

        return text, changes

    def _fix_repeated_words(
        self,
        text: str,
        track: bool,
    ) -> Tuple[str, List[TextChange]]:
        """Fix repeated words."""
        changes = []

        pattern = r"\b(\w+)\s+\1\b"
        for match in re.finditer(pattern, text, re.IGNORECASE):
            if track:
                changes.append(
                    TextChange(
                        change_type=ChangeType.GRAMMAR,
                        original=match.group(),
                        refined=match.group(1),
                        location=match.start(),
                        reason="Remove repeated word",
                    )
                )

        text = re.sub(pattern, r"\1", text, flags=re.IGNORECASE)

        return text, changes

    def _normalize_whitespace(
        self,
        text: str,
        track: bool,
    ) -> Tuple[str, List[TextChange]]:
        """Normalize whitespace."""
        changes = []

        # Multiple spaces to single
        pattern = r"[^\S\n]{2,}"
        for match in re.finditer(pattern, text):
            if track:
                changes.append(
                    TextChange(
                        change_type=ChangeType.FORMATTING,
                        original=match.group(),
                        refined=" ",
                        location=match.start(),
                        reason="Normalize multiple spaces",
                    )
                )
        text = re.sub(pattern, " ", text)

        # Trim trailing whitespace per line
        text = "\n".join(line.rstrip() for line in text.split("\n"))

        return text, changes

    def _check_for_flags(self, text: str) -> List[str]:
        """Check for issues that need human attention."""
        flags = []

        # Check for potential meaning ambiguity
        if "?" in text and not text.strip().endswith("?"):
            flags.append("Document contains embedded questions - verify intent")

        # Check for passive voice (simplified)
        passive_patterns = [
            r"\b(?:is|are|was|were|been|being)\s+\w+ed\b",
        ]
        for pattern in passive_patterns:
            if re.search(pattern, text):
                flags.append(
                    "Document contains passive voice - consider if active voice is clearer"
                )
                break

        # Check for very long sentences
        sentences = re.split(r"[.!?]+", text)
        for sentence in sentences:
            words = sentence.split()
            if len(words) > 50:
                flags.append("Document contains very long sentences - consider breaking up")
                break

        return flags

    def _llm_refine(
        self,
        text: str,
        preserve_meaning: bool,
        style_guide: Optional[str],
    ) -> Tuple[str, List[TextChange]]:
        """Use LLM for deeper refinement."""
        changes = []

        prompt = self._build_refinement_prompt(text, preserve_meaning, style_guide)

        response = self._llm_callback(
            prompt,
            {
                "temperature": 0.3,
                "max_tokens": len(text) * 2,
            },
        )

        if response and response != text:
            changes.append(
                TextChange(
                    change_type=ChangeType.STYLE,
                    original="(full text)",
                    refined="(see refined output)",
                    location=0,
                    reason="LLM-based style and clarity improvements",
                )
            )
            return response, changes

        return text, changes

    def _build_refinement_prompt(
        self,
        text: str,
        preserve_meaning: bool,
        style_guide: Optional[str],
    ) -> str:
        """Build prompt for LLM refinement."""
        prompt = (
            "You are Quill, a document refinement agent. "
            "Improve the following text for clarity, grammar, and style "
            "while strictly preserving the original meaning and voice.\n\n"
        )

        if preserve_meaning:
            prompt += "IMPORTANT: Do not change the meaning or add new content.\n\n"

        if style_guide:
            prompt += f"Style guide: {style_guide}\n\n"

        prompt += f"Text to refine:\n{text}\n\nRefined text:"

        return prompt


def create_formatting_engine() -> FormattingEngine:
    """Create a formatting engine with default templates."""
    return FormattingEngine()


def create_refinement_engine(
    llm_callback: Optional[Callable[[str, Dict[str, Any]], str]] = None,
) -> RefinementEngine:
    """Create a refinement engine."""
    return RefinementEngine(llm_callback=llm_callback)
