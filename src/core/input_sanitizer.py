"""
Prompt Sanitization Layer

Provides blocking and sanitization for user input before LLM prompt inclusion.
Works alongside the existing InputClassifier (data delimiting) and enforcement
engine (denial patterns) to form a complete defense:

1. InputClassifier: wraps untrusted content in data delimiters
2. PromptSanitizer: detects and blocks/strips injection patterns
3. Enforcement engine: constitutional rule matching

This addresses Finding #4 (No prompt sanitization layer) from the
Agentic Security Audit v3.0.
"""

import logging
import re
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)


class PromptSanitizer:
    """
    Detects, blocks, and sanitizes prompt injection patterns in user input.

    Unlike the enforcement engine (which operates on agent-to-agent messages),
    this operates on the HTTP/WebSocket input path before messages reach the LLM.
    """

    # Patterns that indicate prompt injection attempts
    INJECTION_PATTERNS: List[Tuple[str, re.Pattern]] = [
        (
            "role_impersonation",
            re.compile(
                r"(?:^|\n)\s*(?:system|assistant|developer)\s*[:\-]",
                re.IGNORECASE,
            ),
        ),
        (
            "instruction_override",
            re.compile(
                r"ignore\s+(?:all\s+)?(?:previous|prior|above|earlier)\s+(?:instructions?|rules?|prompts?|context)",
                re.IGNORECASE,
            ),
        ),
        (
            "role_reassignment",
            re.compile(
                r"you\s+are\s+now\s+(?:a\s+)?(?:new|different|my)\s+",
                re.IGNORECASE,
            ),
        ),
        (
            "prompt_leak_request",
            re.compile(
                r"(?:show|reveal|print|output|repeat|display)\s+(?:your\s+)?(?:system\s+)?(?:prompt|instructions?|rules?|constitution)",
                re.IGNORECASE,
            ),
        ),
        (
            "delimiter_escape",
            re.compile(
                r"<\|(?:im_start|im_end|endoftext|system|end)\|>",
                re.IGNORECASE,
            ),
        ),
        (
            "forget_rules",
            re.compile(
                r"forget\s+(?:your\s+)?(?:all\s+)?(?:rules?|instructions?|training|constraints?)",
                re.IGNORECASE,
            ),
        ),
    ]

    # Threshold for blocking (0.0 = block nothing, 1.0 = block everything)
    DEFAULT_BLOCK_THRESHOLD = 0.8

    def __init__(self, block_threshold: Optional[float] = None):
        self.block_threshold = block_threshold or self.DEFAULT_BLOCK_THRESHOLD

    def analyze(self, text: str) -> Tuple[float, List[str]]:
        """
        Analyze text for injection patterns.

        Returns:
            Tuple of (detection_score, matched_pattern_names).
            Score is 0.0-1.0 based on number and severity of matches.
        """
        from .input_classifier import TextNormalizer

        normalized = TextNormalizer.normalize(text)
        matches = []

        for name, pattern in self.INJECTION_PATTERNS:
            if pattern.search(text) or pattern.search(normalized):
                matches.append(name)

        if not matches:
            return 0.0, []

        # Score: each match adds weight, capped at 1.0
        score = min(1.0, len(matches) * 0.3)
        return score, matches

    def should_block(self, text: str) -> Tuple[bool, float, List[str]]:
        """
        Determine if the input should be blocked.

        Returns:
            Tuple of (should_block, score, matched_patterns).
        """
        score, matches = self.analyze(text)
        blocked = score >= self.block_threshold
        if blocked:
            logger.warning(
                "Prompt injection blocked: score=%.2f patterns=%s",
                score,
                matches,
            )
        return blocked, score, matches

    def sanitize(self, text: str) -> str:
        """
        Strip known injection patterns from text.

        Applied when score is below block threshold but patterns are detected.
        Replaces matched patterns with safe placeholders.
        """
        sanitized = text

        # Strip delimiter escape sequences
        sanitized = re.sub(
            r"<\|(?:im_start|im_end|endoftext|system|end)\|>",
            "[FILTERED]",
            sanitized,
            flags=re.IGNORECASE,
        )

        # Strip role impersonation at line start
        sanitized = re.sub(
            r"(?:^|\n)\s*(system|assistant|developer)\s*[:\-]",
            "\n[FILTERED]:",
            sanitized,
            flags=re.IGNORECASE,
        )

        return sanitized

    def process(self, text: str) -> Tuple[str, bool, float]:
        """
        Full processing pipeline: analyze, block if needed, sanitize if not.

        Returns:
            Tuple of (processed_text_or_empty, was_blocked, score).
            If blocked, processed_text is empty string.
        """
        blocked, score, matches = self.should_block(text)
        if blocked:
            return "", True, score

        # Below threshold but patterns detected — sanitize
        if matches:
            logger.info(
                "Prompt sanitized: score=%.2f patterns=%s",
                score,
                matches,
            )
            return self.sanitize(text), False, score

        return text, False, score
