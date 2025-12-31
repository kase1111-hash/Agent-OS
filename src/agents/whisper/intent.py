"""
Agent OS Intent Classifier

Classifies user requests into intent categories for routing.
Supports the 8 standard intent categories plus custom extensions.
"""

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


class IntentCategory(str, Enum):
    """
    Standard intent categories for request classification.

    The Orchestrator (Whisper) MUST classify requests into one of these categories.
    """

    # Query intents
    QUERY_FACTUAL = "query.factual"  # Factual information request
    QUERY_REASONING = "query.reasoning"  # Complex reasoning or analysis

    # Content intents
    CONTENT_CREATIVE = "content.creative"  # Creative content generation
    CONTENT_TECHNICAL = "content.technical"  # Technical or code-related

    # Memory intents
    MEMORY_RECALL = "memory.recall"  # Memory retrieval request
    MEMORY_STORE = "memory.store"  # Memory storage request

    # System intents
    SYSTEM_META = "system.meta"  # Meta-operations (status, config)

    # Security intents
    SECURITY_SENSITIVE = "security.sensitive"  # Requires elevated security review

    # Unknown (requires fallback)
    UNKNOWN = "unknown"


@dataclass
class IntentClassification:
    """Result of intent classification."""

    primary_intent: IntentCategory
    confidence: float  # 0.0 to 1.0
    secondary_intents: List[Tuple[IntentCategory, float]] = field(default_factory=list)
    reasoning: Optional[str] = None  # Why this classification
    keywords_matched: List[str] = field(default_factory=list)
    requires_smith_review: bool = False  # Pre-flagged for security
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def is_high_confidence(self) -> bool:
        """Check if classification is high confidence (>0.8)."""
        return self.confidence >= 0.8

    @property
    def is_ambiguous(self) -> bool:
        """Check if classification is ambiguous (multiple likely intents)."""
        if not self.secondary_intents:
            return False
        # Ambiguous if second intent is close to primary
        return self.secondary_intents[0][1] >= self.confidence * 0.8


# Keyword patterns for rule-based classification
INTENT_PATTERNS: Dict[IntentCategory, Dict[str, Any]] = {
    IntentCategory.QUERY_FACTUAL: {
        "keywords": [
            "what is",
            "who is",
            "when did",
            "where is",
            "how many",
            "define",
            "explain",
            "describe",
            "tell me about",
            "fact",
            "information",
            "data",
            "statistics",
        ],
        "patterns": [
            r"^what\s+(?:is|are|was|were)\b",
            r"^who\s+(?:is|are|was|were)\b",
            r"^when\s+(?:did|was|were|is)\b",
            r"^where\s+(?:is|are|was|were)\b",
            r"^how\s+(?:many|much|old)\b",
        ],
        "base_confidence": 0.7,
    },
    IntentCategory.QUERY_REASONING: {
        "keywords": [
            "why",
            "analyze",
            "compare",
            "contrast",
            "evaluate",
            "reason",
            "deduce",
            "infer",
            "conclude",
            "synthesize",
            "implications",
            "consequences",
            "cause",
            "effect",
            "pros and cons",
            "advantages",
            "disadvantages",
            "should",
            "would",
            "could",
            "better",
            "best",
        ],
        "patterns": [
            r"^why\s+",
            r"^how\s+(?:would|should|could)\b",
            r"\banalyze\b",
            r"\bcompare\b.*\bto\b",
            r"\bwhat.*implications\b",
        ],
        "base_confidence": 0.7,
    },
    IntentCategory.CONTENT_CREATIVE: {
        "keywords": [
            "write",
            "create",
            "compose",
            "generate",
            "craft",
            "story",
            "poem",
            "essay",
            "article",
            "fiction",
            "creative",
            "imagine",
            "invent",
            "design",
            "brainstorm",
            "ideas",
            "suggest",
        ],
        "patterns": [
            r"^write\s+(?:a|an|me)\b",
            r"^create\s+",
            r"^compose\s+",
            r"\bstory\s+about\b",
            r"\bpoem\s+about\b",
        ],
        "base_confidence": 0.75,
    },
    IntentCategory.CONTENT_TECHNICAL: {
        "keywords": [
            "code",
            "program",
            "function",
            "class",
            "api",
            "implement",
            "debug",
            "fix",
            "bug",
            "error",
            "python",
            "javascript",
            "java",
            "rust",
            "go",
            "algorithm",
            "data structure",
            "database",
            "sql",
            "deploy",
            "configure",
            "install",
            "setup",
        ],
        "patterns": [
            r"^(?:write|create|implement)\s+(?:a\s+)?(?:function|class|method)\b",
            r"\b(?:python|javascript|java|rust|go|typescript)\b",
            r"\bfix\s+(?:the\s+)?(?:bug|error|issue)\b",
            r"\bcode\s+(?:for|to|that)\b",
        ],
        "base_confidence": 0.8,
    },
    IntentCategory.MEMORY_RECALL: {
        "keywords": [
            "remember",
            "recall",
            "what did I",
            "when did I",
            "previous",
            "earlier",
            "last time",
            "before",
            "history",
            "conversation",
            "mentioned",
            "said",
            "find",
            "search",
            "look up",
            "retrieve",
        ],
        "patterns": [
            r"\bremember\s+(?:when|what|how)\b",
            r"\blast\s+time\b",
            r"\bearlier\s+(?:I|we|you)\b",
            r"\bwhat\s+did\s+(?:I|we)\s+(?:say|mention|discuss)\b",
        ],
        "base_confidence": 0.8,
    },
    IntentCategory.MEMORY_STORE: {
        "keywords": [
            "remember this",
            "save",
            "store",
            "keep",
            "don't forget",
            "note",
            "bookmark",
            "record",
            "for later",
            "for next time",
        ],
        "patterns": [
            r"\bremember\s+(?:this|that)\b",
            r"\bsave\s+(?:this|that)\b",
            r"\bkeep\s+(?:this|that)\s+in\s+mind\b",
            r"\bdon'?t\s+forget\b",
        ],
        "base_confidence": 0.85,
    },
    IntentCategory.SYSTEM_META: {
        "keywords": [
            "status",
            "config",
            "configuration",
            "settings",
            "help",
            "version",
            "about",
            "capabilities",
            "reset",
            "clear",
            "restart",
            "shutdown",
            "list agents",
            "available",
            "supported",
        ],
        "patterns": [
            r"^(?:show|get|what\s+is)\s+(?:the\s+)?(?:status|config)\b",
            r"^help\b",
            r"^what\s+can\s+you\s+do\b",
            r"^list\s+(?:agents|capabilities)\b",
        ],
        "base_confidence": 0.9,
    },
    IntentCategory.SECURITY_SENSITIVE: {
        "keywords": [
            "password",
            "credential",
            "secret",
            "key",
            "private",
            "confidential",
            "sensitive",
            "delete",
            "remove",
            "purge",
            "erase",
            "override",
            "bypass",
            "ignore",
            "skip",
            "admin",
            "root",
            "sudo",
            "permission",
        ],
        "patterns": [
            r"\bpassword\b",
            r"\b(?:api|secret|private)\s+key\b",
            r"\bdelete\s+(?:all|everything)\b",
            r"\bbypass\s+",
            r"\boverride\s+",
        ],
        "base_confidence": 0.85,
    },
}


class IntentClassifier:
    """
    Classifies user requests into intent categories.

    Uses a combination of:
    1. Rule-based pattern matching (fast, reliable for clear cases)
    2. Keyword analysis (good for mixed signals)
    3. LLM classification (optional, for ambiguous cases)
    """

    def __init__(
        self,
        use_llm: bool = False,
        llm_client: Optional[Any] = None,
        confidence_threshold: float = 0.6,
    ):
        """
        Initialize classifier.

        Args:
            use_llm: Whether to use LLM for ambiguous cases
            llm_client: Ollama client for LLM classification
            confidence_threshold: Minimum confidence for classification
        """
        self.use_llm = use_llm
        self.llm_client = llm_client
        self.confidence_threshold = confidence_threshold

        # Compile regex patterns
        self._compiled_patterns: Dict[IntentCategory, List[re.Pattern]] = {}
        for intent, config in INTENT_PATTERNS.items():
            self._compiled_patterns[intent] = [
                re.compile(p, re.IGNORECASE) for p in config.get("patterns", [])
            ]

        # Track classification metrics
        self._classification_count = 0
        self._high_confidence_count = 0

    def classify(
        self,
        text: str,
        context: Optional[List[Dict[str, str]]] = None,
    ) -> IntentClassification:
        """
        Classify a user request into an intent category.

        Args:
            text: User request text
            context: Optional conversation context

        Returns:
            IntentClassification with primary intent and confidence
        """
        self._classification_count += 1

        # Normalize text
        text_lower = text.lower().strip()

        # Score all intents
        scores: Dict[IntentCategory, Tuple[float, List[str]]] = {}

        for intent, config in INTENT_PATTERNS.items():
            score, matched = self._score_intent(text_lower, intent, config)
            if score > 0:
                scores[intent] = (score, matched)

        # Determine primary intent
        if not scores:
            return IntentClassification(
                primary_intent=IntentCategory.UNKNOWN,
                confidence=0.0,
                reasoning="No matching patterns or keywords found",
            )

        # Sort by score
        sorted_intents = sorted(scores.items(), key=lambda x: x[1][0], reverse=True)
        primary_intent, (primary_score, primary_keywords) = sorted_intents[0]

        # Build secondary intents
        secondary_intents = [
            (intent, score)
            for intent, (score, _) in sorted_intents[1:4]  # Top 3 alternatives
            if score >= self.confidence_threshold * 0.5
        ]

        # Check if Smith review is needed
        requires_smith = self._requires_security_review(text_lower, primary_intent)

        # Build result
        result = IntentClassification(
            primary_intent=primary_intent,
            confidence=primary_score,
            secondary_intents=secondary_intents,
            keywords_matched=primary_keywords,
            requires_smith_review=requires_smith,
            reasoning=self._generate_reasoning(primary_intent, primary_keywords),
        )

        # Track metrics
        if result.is_high_confidence:
            self._high_confidence_count += 1

        # Use LLM for low confidence or ambiguous cases
        if self.use_llm and self.llm_client:
            if not result.is_high_confidence or result.is_ambiguous:
                result = self._classify_with_llm(text, context, result)

        return result

    def _score_intent(
        self,
        text: str,
        intent: IntentCategory,
        config: Dict[str, Any],
    ) -> Tuple[float, List[str]]:
        """Score text against an intent's patterns and keywords."""
        score = 0.0
        matched_keywords = []

        # Check regex patterns (high weight)
        patterns = self._compiled_patterns.get(intent, [])
        for pattern in patterns:
            if pattern.search(text):
                score += 0.4
                break

        # Check keywords (lower weight per keyword, cumulative)
        keywords = config.get("keywords", [])
        keyword_score = 0.0
        for keyword in keywords:
            if keyword.lower() in text:
                keyword_score += 0.15
                matched_keywords.append(keyword)

        # Cap keyword contribution
        score += min(keyword_score, 0.6)

        # Apply base confidence
        if score > 0:
            base = config.get("base_confidence", 0.5)
            score = min(score + base * 0.3, 1.0)

        return score, matched_keywords

    def _requires_security_review(
        self,
        text: str,
        primary_intent: IntentCategory,
    ) -> bool:
        """Determine if request requires Smith security review."""
        # Security intent always requires review
        if primary_intent == IntentCategory.SECURITY_SENSITIVE:
            return True

        # Check for sensitive patterns
        sensitive_patterns = [
            r"\bdelete\b.*\b(?:all|everything|data)\b",
            r"\bpassword\b",
            r"\b(?:private|secret)\s+key\b",
            r"\boverride\b.*\b(?:security|restriction|limit)\b",
            r"\bexecute\b.*\b(?:command|script|code)\b",
        ]

        for pattern in sensitive_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True

        return False

    def _generate_reasoning(
        self,
        intent: IntentCategory,
        keywords: List[str],
    ) -> str:
        """Generate human-readable reasoning for classification."""
        if not keywords:
            return f"Classified as {intent.value} based on pattern matching"

        keyword_str = ", ".join(f"'{k}'" for k in keywords[:3])
        return f"Classified as {intent.value} based on keywords: {keyword_str}"

    def _classify_with_llm(
        self,
        text: str,
        context: Optional[List[Dict[str, str]]],
        initial: IntentClassification,
    ) -> IntentClassification:
        """Use LLM to refine classification for ambiguous cases."""
        if not self.llm_client:
            return initial

        # Build prompt for LLM classification
        intent_options = "\n".join(
            [
                f"- {cat.value}: {self._get_intent_description(cat)}"
                for cat in IntentCategory
                if cat != IntentCategory.UNKNOWN
            ]
        )

        prompt = f"""Classify the following user request into one of these intent categories:

{intent_options}

User request: "{text}"

Initial classification: {initial.primary_intent.value} (confidence: {initial.confidence:.2f})

Respond with ONLY the intent category (e.g., "query.factual") and a confidence score (0.0-1.0).
Format: INTENT|CONFIDENCE"""

        try:
            from src.agents.ollama import OllamaMessage

            response = self.llm_client.generate(
                model="mistral:7b",  # Default model for Whisper
                prompt=prompt,
                options={"temperature": 0.1, "num_predict": 50},
            )

            # Parse response
            parts = response.content.strip().split("|")
            if len(parts) == 2:
                intent_str = parts[0].strip()
                confidence = float(parts[1].strip())

                # Find matching intent
                for cat in IntentCategory:
                    if cat.value == intent_str:
                        return IntentClassification(
                            primary_intent=cat,
                            confidence=min(confidence, 1.0),
                            secondary_intents=initial.secondary_intents,
                            keywords_matched=initial.keywords_matched,
                            requires_smith_review=initial.requires_smith_review,
                            reasoning=f"LLM classification: {cat.value}",
                        )

        except Exception as e:
            logger.warning(f"LLM classification failed: {e}")

        return initial

    def _get_intent_description(self, intent: IntentCategory) -> str:
        """Get human-readable description for intent."""
        descriptions = {
            IntentCategory.QUERY_FACTUAL: "Factual information request",
            IntentCategory.QUERY_REASONING: "Complex reasoning or analysis",
            IntentCategory.CONTENT_CREATIVE: "Creative content generation",
            IntentCategory.CONTENT_TECHNICAL: "Technical or code-related",
            IntentCategory.MEMORY_RECALL: "Memory retrieval request",
            IntentCategory.MEMORY_STORE: "Memory storage request",
            IntentCategory.SYSTEM_META: "Meta-operations (status, config)",
            IntentCategory.SECURITY_SENSITIVE: "Requires elevated security review",
        }
        return descriptions.get(intent, "Unknown")

    def get_metrics(self) -> Dict[str, Any]:
        """Get classification metrics."""
        return {
            "total_classifications": self._classification_count,
            "high_confidence_count": self._high_confidence_count,
            "high_confidence_rate": (
                self._high_confidence_count / self._classification_count
                if self._classification_count > 0
                else 0.0
            ),
        }


def classify_intent(text: str, **kwargs) -> IntentClassification:
    """
    Convenience function to classify intent.

    Args:
        text: User request text
        **kwargs: Additional classifier options

    Returns:
        IntentClassification
    """
    classifier = IntentClassifier(**kwargs)
    return classifier.classify(text)
