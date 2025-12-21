"""
Agent OS Constitution Parser

Parses constitutional documents in Markdown format with YAML frontmatter.
Extracts rules, sections, and metadata for the constitutional kernel.
"""

import re
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set, Any
from dataclasses import dataclass

import yaml

from .models import (
    ConstitutionMetadata,
    Rule,
    RuleType,
    AuthorityLevel,
    Constitution,
)


# Rule type indicators (keywords that help classify rules)
MANDATE_KEYWORDS = {"shall", "must", "will", "required", "mandatory"}
PROHIBITION_KEYWORDS = {"shall not", "must not", "cannot", "may not", "prohibited", "forbidden", "never"}
PERMISSION_KEYWORDS = {"may", "can", "allowed", "permitted", "optional"}
ESCALATION_KEYWORDS = {"escalate", "notify", "alert", "human steward", "override"}
IMMUTABLE_MARKERS = {"(immutable)", "[immutable]", "immutable"}


@dataclass
class Section:
    """Represents a parsed markdown section."""
    level: int           # Header level (1-6)
    title: str           # Section title
    content: str         # Section content (without subsections)
    full_content: str    # Full content including subsections
    path: List[str]      # Full path of parent headers
    line_start: int      # Starting line number
    line_end: int        # Ending line number


class ConstitutionParser:
    """
    Parser for constitutional documents.

    Handles:
    - YAML frontmatter extraction
    - Markdown section parsing
    - Rule extraction and classification
    - Keyword detection
    """

    def __init__(self):
        self._section_cache: Dict[str, List[Section]] = {}

    def parse_file(self, file_path: Path) -> Constitution:
        """
        Parse a constitutional document from a file.

        Args:
            file_path: Path to the markdown file

        Returns:
            Parsed Constitution object

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file format is invalid
        """
        if not file_path.exists():
            raise FileNotFoundError(f"Constitution file not found: {file_path}")

        content = file_path.read_text(encoding="utf-8")
        return self.parse_content(content, file_path)

    def parse_content(self, content: str, source_path: Optional[Path] = None) -> Constitution:
        """
        Parse a constitutional document from string content.

        Args:
            content: Markdown content with YAML frontmatter
            source_path: Optional source file path for reference

        Returns:
            Parsed Constitution object
        """
        # Calculate file hash for change detection
        file_hash = hashlib.sha256(content.encode()).hexdigest()

        # Extract frontmatter and body
        frontmatter, body = self._extract_frontmatter(content)

        # Parse metadata
        metadata = ConstitutionMetadata.from_frontmatter(frontmatter, source_path)

        # Parse sections
        sections = self._parse_sections(body)
        sections_dict = {s.title: s.full_content for s in sections}

        # Extract rules from sections
        rules = self._extract_rules(sections, metadata)

        return Constitution(
            metadata=metadata,
            rules=rules,
            sections=sections_dict,
            raw_content=content,
            file_hash=file_hash,
        )

    def _extract_frontmatter(self, content: str) -> Tuple[Dict[str, Any], str]:
        """
        Extract YAML frontmatter from markdown content.

        Args:
            content: Full markdown content

        Returns:
            Tuple of (frontmatter dict, body content)
        """
        # Match YAML frontmatter between --- delimiters
        pattern = r"^---\s*\n(.*?)\n---\s*\n(.*)$"
        match = re.match(pattern, content, re.DOTALL)

        if match:
            try:
                frontmatter = yaml.safe_load(match.group(1)) or {}
                body = match.group(2)
                return frontmatter, body
            except yaml.YAMLError as e:
                raise ValueError(f"Invalid YAML frontmatter: {e}")

        # No frontmatter found, return empty dict and full content
        return {}, content

    def _parse_sections(self, content: str) -> List[Section]:
        """
        Parse markdown into hierarchical sections.

        Args:
            content: Markdown body content (without frontmatter)

        Returns:
            List of Section objects (one for each header at any level)
        """
        lines = content.split("\n")
        sections: List[Section] = []
        header_pattern = re.compile(r"^(#{1,6})\s+(.+)$")

        # Find all header positions first
        headers = []
        for i, line in enumerate(lines):
            match = header_pattern.match(line)
            if match:
                headers.append((i, len(match.group(1)), match.group(2).strip()))

        # Track current path through headers
        current_path: List[str] = []
        current_levels: List[int] = []

        for idx, (line_num, level, title) in enumerate(headers):
            # Update path based on level
            while current_levels and current_levels[-1] >= level:
                current_levels.pop()
                current_path.pop()

            current_path.append(title)
            current_levels.append(level)

            # Find section content (until next header)
            start = line_num + 1
            if idx + 1 < len(headers):
                end = headers[idx + 1][0]
            else:
                end = len(lines)

            content_lines = lines[start:end]

            sections.append(Section(
                level=level,
                title=title,
                content="\n".join(content_lines).strip(),
                full_content="\n".join(content_lines).strip(),
                path=list(current_path),
                line_start=start,
                line_end=end - 1,
            ))

        return sections

    def _extract_rules(self, sections: List[Section], metadata: ConstitutionMetadata) -> List[Rule]:
        """
        Extract rules from parsed sections.

        Args:
            sections: List of parsed sections
            metadata: Document metadata

        Returns:
            List of extracted rules
        """
        rules: List[Rule] = []

        for section in sections:
            # Check if section is marked as immutable
            is_immutable = self._check_immutable(section.title)

            # Extract rules from section content
            section_rules = self._extract_rules_from_content(
                content=section.content,
                section=section,
                metadata=metadata,
                is_immutable=is_immutable,
            )
            rules.extend(section_rules)

        return rules

    def _extract_rules_from_content(
        self,
        content: str,
        section: Section,
        metadata: ConstitutionMetadata,
        is_immutable: bool,
    ) -> List[Rule]:
        """
        Extract individual rules from section content.

        Looks for:
        - Bullet points (- or *)
        - Numbered items
        - Sentences with rule keywords
        """
        rules: List[Rule] = []

        # Extract bullet points and numbered items
        bullet_pattern = re.compile(r"^[\s]*[-*]\s+(.+)$", re.MULTILINE)
        numbered_pattern = re.compile(r"^[\s]*\d+\.\s+(.+)$", re.MULTILINE)

        # Find all bullet points
        for match in bullet_pattern.finditer(content):
            rule_text = match.group(1).strip()
            if rule_text:
                rule = self._create_rule(
                    content=rule_text,
                    section=section,
                    metadata=metadata,
                    is_immutable=is_immutable,
                )
                if rule:
                    rules.append(rule)

        # Find all numbered items
        for match in numbered_pattern.finditer(content):
            rule_text = match.group(1).strip()
            if rule_text:
                rule = self._create_rule(
                    content=rule_text,
                    section=section,
                    metadata=metadata,
                    is_immutable=is_immutable,
                )
                if rule:
                    rules.append(rule)

        # Also look for standalone sentences with rule keywords
        # Split content into sentences
        sentences = re.split(r"(?<=[.!?])\s+", content)
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence and self._is_rule_sentence(sentence):
                # Avoid duplicates from bullet points
                if not any(sentence in r.content for r in rules):
                    rule = self._create_rule(
                        content=sentence,
                        section=section,
                        metadata=metadata,
                        is_immutable=is_immutable,
                    )
                    if rule:
                        rules.append(rule)

        return rules

    def _create_rule(
        self,
        content: str,
        section: Section,
        metadata: ConstitutionMetadata,
        is_immutable: bool,
    ) -> Optional[Rule]:
        """Create a Rule object from extracted content."""
        # Skip empty or too short content
        if len(content) < 10:
            return None

        # Classify rule type
        rule_type = self._classify_rule_type(content)

        # Extract keywords
        keywords = self._extract_keywords(content)

        # Extract references to other sections/rules
        references = self._extract_references(content)

        return Rule(
            id="",  # Will be generated in __post_init__
            content=content,
            rule_type=rule_type,
            section=section.title,
            section_path=section.path,
            authority_level=metadata.authority_level,
            scope=metadata.scope,
            source_file=metadata.file_path,
            line_number=section.line_start,
            is_immutable=is_immutable or self._check_immutable(content),
            keywords=keywords,
            references=references,
        )

    def _classify_rule_type(self, content: str) -> RuleType:
        """Classify the type of rule based on content."""
        content_lower = content.lower()

        # Check for prohibition first (more specific)
        for keyword in PROHIBITION_KEYWORDS:
            if keyword in content_lower:
                return RuleType.PROHIBITION

        # Check for mandate
        for keyword in MANDATE_KEYWORDS:
            if keyword in content_lower:
                return RuleType.MANDATE

        # Check for permission
        for keyword in PERMISSION_KEYWORDS:
            if keyword in content_lower:
                return RuleType.PERMISSION

        # Check for escalation
        for keyword in ESCALATION_KEYWORDS:
            if keyword in content_lower:
                return RuleType.ESCALATION

        # Check for immutable marker
        if self._check_immutable(content):
            return RuleType.IMMUTABLE

        # Default to principle
        return RuleType.PRINCIPLE

    def _check_immutable(self, text: str) -> bool:
        """Check if text indicates an immutable rule."""
        text_lower = text.lower()
        for marker in IMMUTABLE_MARKERS:
            if marker in text_lower:
                return True
        return False

    def _is_rule_sentence(self, sentence: str) -> bool:
        """Check if a sentence contains rule indicators."""
        sentence_lower = sentence.lower()
        all_keywords = (
            MANDATE_KEYWORDS |
            PROHIBITION_KEYWORDS |
            PERMISSION_KEYWORDS |
            ESCALATION_KEYWORDS
        )
        return any(kw in sentence_lower for kw in all_keywords)

    def _extract_keywords(self, content: str) -> Set[str]:
        """Extract significant keywords from rule content."""
        # Common words to exclude
        stopwords = {
            "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
            "have", "has", "had", "do", "does", "did", "shall", "will", "should",
            "would", "could", "may", "might", "must", "can", "cannot", "to", "of",
            "in", "for", "on", "with", "at", "by", "from", "or", "and", "not",
            "this", "that", "these", "those", "it", "its", "all", "any", "no",
            "only", "if", "when", "where", "what", "which", "who", "whom", "how",
        }

        # Extract words (alphanumeric, 3+ chars)
        words = re.findall(r"\b[a-zA-Z]{3,}\b", content.lower())

        # Filter stopwords and return as set
        return {w for w in words if w not in stopwords}

    def _extract_references(self, content: str) -> List[str]:
        """Extract references to other sections or documents."""
        references = []

        # Match section references like "Section IV" or "Article III"
        section_refs = re.findall(r"(?:Section|Article|Chapter)\s+[IVXLCDM]+", content, re.IGNORECASE)
        references.extend(section_refs)

        # Match document references like "CONSTITUTION.md" or "constitution.md"
        doc_refs = re.findall(r"\b[\w-]+\.md\b", content, re.IGNORECASE)
        references.extend(doc_refs)

        # Match explicit references like "see: X" or "refer to: X"
        explicit_refs = re.findall(r"(?:see|refer to|reference):\s*([^.]+)", content, re.IGNORECASE)
        references.extend(explicit_refs)

        return references


def parse_constitution(file_path: Path) -> Constitution:
    """
    Convenience function to parse a constitution file.

    Args:
        file_path: Path to constitution markdown file

    Returns:
        Parsed Constitution object
    """
    parser = ConstitutionParser()
    return parser.parse_file(file_path)
