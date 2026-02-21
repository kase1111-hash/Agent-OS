"""
Constitutional Enforcement Engine

Three-tier evaluation for constitutional compliance:

1. Structural checks (fast, no LLM) — scope, format, explicit denials, rate limits
2. Semantic matching (embeddings via Ollama) — replaces keyword-in-string
3. LLM compliance judgment (full Ollama evaluation) — for ambiguous cases

The engine is designed to fail-safe: if the LLM is unavailable,
it falls back to structural checks with a conservative default.
"""

import hashlib
import logging
import re
import time
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from .models import Rule, RuleType, AuthorityLevel

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tier 1: Structural Checks
# ---------------------------------------------------------------------------

@dataclass
class StructuralResult:
    """Result from Tier 1 structural checks."""
    allowed: bool
    reason: Optional[str] = None
    matched_rules: List[Rule] = field(default_factory=list)
    # If True, skip Tier 2/3 — the decision is final
    definitive: bool = False


class StructuralChecker:
    """
    Tier 1: Fast, deterministic structural checks.

    These checks run without any LLM calls and can immediately
    allow or deny a request based on:
    - Agent scope matching
    - Explicit deny-list patterns (regex)
    - Request format validation
    - Rate limiting
    """

    # Explicit denial patterns — these are always blocked regardless of LLM
    DENY_PATTERNS: List[re.Pattern] = [
        re.compile(r"ignore\s+(previous|prior|above|all)\s+(rules?|instructions?|constraints?)", re.I),
        re.compile(r"forget\s+(your|all|the)\s+(rules?|instructions?|constitution)", re.I),
        re.compile(r"you\s+are\s+now\s+(free|unbound|unconstrained)", re.I),
        re.compile(r"disregard\s+(your|all|the)\s+(rules?|guidelines?|constitution)", re.I),
        re.compile(r"override\s+(constitutional|all)\s+(rules?|protections?)", re.I),
        re.compile(r"(jailbreak|bypass|circumvent)\s+(the\s+)?(constitution|rules?|safety)", re.I),
    ]

    # Maximum content length (prevent resource exhaustion)
    MAX_CONTENT_LENGTH = 50_000

    def __init__(self):
        self._rate_limits: Dict[str, List[float]] = {}
        self._rate_lock = threading.Lock()
        self.requests_per_minute = 60

    def check(self, context: Any, rules: List[Rule]) -> StructuralResult:
        """
        Run all structural checks.

        Args:
            context: RequestContext
            rules: All rules for the destination agent

        Returns:
            StructuralResult — if definitive=True, skip further tiers
        """
        # Check 1: Request format validation
        result = self._check_format(context)
        if not result.allowed:
            return result

        # Check 2: Explicit denial patterns (prompt injection attempts)
        result = self._check_explicit_denials(context)
        if not result.allowed:
            return result

        # Check 3: Agent scope — filter rules to those applicable
        applicable = self._filter_by_scope(rules, context)

        # Check 4: Rate limiting
        result = self._check_rate_limit(context)
        if not result.allowed:
            return result

        # Check 5: Immutable prohibition rules with exact keyword match
        # (preserve the fast path for clear-cut cases)
        for rule in applicable:
            if rule.is_immutable and rule.rule_type == RuleType.PROHIBITION:
                content_lower = context.content.lower()
                # Only trigger on exact keyword matches in content
                for keyword in rule.keywords:
                    if keyword in content_lower:
                        return StructuralResult(
                            allowed=False,
                            reason=f"Immutable prohibition violated: {rule.content[:200]}",
                            matched_rules=[rule],
                            definitive=True,
                        )

        # Structural checks passed — no definitive decision, continue to Tier 2
        return StructuralResult(
            allowed=True,
            matched_rules=applicable,
            definitive=False,
        )

    def _check_format(self, context: Any) -> StructuralResult:
        """Validate request format."""
        if not context.content or not context.content.strip():
            return StructuralResult(
                allowed=False,
                reason="Empty request content",
                definitive=True,
            )

        if len(context.content) > self.MAX_CONTENT_LENGTH:
            return StructuralResult(
                allowed=False,
                reason=f"Request content exceeds maximum length ({self.MAX_CONTENT_LENGTH} chars)",
                definitive=True,
            )

        if not context.destination or not context.destination.strip():
            return StructuralResult(
                allowed=False,
                reason="Missing destination agent",
                definitive=True,
            )

        return StructuralResult(allowed=True)

    def _check_explicit_denials(self, context: Any) -> StructuralResult:
        """Check for explicit prompt injection / jailbreak patterns.

        V1-2: Applies Unicode normalization before pattern matching to catch
        obfuscated variants using homoglyphs, zero-width chars, or encoding tricks.
        """
        content = context.content

        # Normalize content to defeat obfuscation (V1-2)
        try:
            from .input_classifier import TextNormalizer

            normalized_content = TextNormalizer.normalize(content)
        except ImportError:
            normalized_content = content.lower()

        # Check both raw and normalized content
        for check_content in (content, normalized_content):
            for pattern in self.DENY_PATTERNS:
                match = pattern.search(check_content)
                if match:
                    return StructuralResult(
                        allowed=False,
                        reason="Request matches explicit denial pattern: suspected prompt injection",
                        definitive=True,
                    )
        return StructuralResult(allowed=True)

    def _filter_by_scope(self, rules: List[Rule], context: Any) -> List[Rule]:
        """Filter rules to those applicable to the destination agent."""
        applicable = []
        for rule in rules:
            if rule.scope == "all_agents" or rule.scope == context.destination:
                applicable.append(rule)
        return applicable

    def _check_rate_limit(self, context: Any) -> StructuralResult:
        """Simple in-memory rate limiter per source."""
        with self._rate_lock:
            now = time.time()
            key = context.source
            timestamps = self._rate_limits.get(key, [])

            # Remove timestamps older than 60 seconds
            cutoff = now - 60
            timestamps = [t for t in timestamps if t > cutoff]

            if len(timestamps) >= self.requests_per_minute:
                self._rate_limits[key] = timestamps
                return StructuralResult(
                    allowed=False,
                    reason=f"Rate limit exceeded: {self.requests_per_minute} requests/minute",
                    definitive=True,
                )

            timestamps.append(now)
            self._rate_limits[key] = timestamps

        return StructuralResult(allowed=True)


# ---------------------------------------------------------------------------
# Tier 2: Semantic Rule Matching
# ---------------------------------------------------------------------------

@dataclass
class SemanticMatch:
    """A rule matched semantically to a request."""
    rule: Rule
    similarity: float  # 0.0 to 1.0


class SemanticMatcher:
    """
    Tier 2: Embedding-based semantic rule matching.

    Uses Ollama embeddings to compute cosine similarity between
    request content and rule text. Rules above the similarity
    threshold are considered "applicable" and proceed to Tier 3.

    This replaces the naive `keyword in content_lower` approach.
    """

    DEFAULT_MODEL = "nomic-embed-text"
    # V1-4: Raised from 0.45 to 0.55 to reduce false matches while catching violations
    DEFAULT_THRESHOLD = 0.55

    def __init__(
        self,
        ollama_client: Any = None,
        model: str = DEFAULT_MODEL,
        threshold: float = DEFAULT_THRESHOLD,
    ):
        self._client = ollama_client
        self._model = model
        self._threshold = threshold

        # Cache: rule content hash -> embedding vector
        self._rule_embeddings: Dict[str, List[float]] = {}
        self._cache_lock = threading.Lock()

    @property
    def available(self) -> bool:
        """Check if semantic matching is available (Ollama accessible)."""
        if self._client is None:
            return False
        try:
            return self._client.is_healthy()
        except Exception:
            return False

    def match(self, context: Any, rules: List[Rule]) -> List[SemanticMatch]:
        """
        Find rules semantically related to the request.

        Args:
            context: RequestContext
            rules: Candidate rules (already scope-filtered by Tier 1)

        Returns:
            List of SemanticMatch above threshold, sorted by similarity desc
        """
        if not self._client or not rules:
            return []

        try:
            # Embed the request (combine intent + content for richer signal)
            request_text = f"{context.intent}: {context.content}"
            request_embedding = self._client.embed(self._model, request_text)

            if not request_embedding:
                logger.warning("Empty embedding returned for request")
                return []

            matches = []
            for rule in rules:
                rule_embedding = self._get_rule_embedding(rule)
                if not rule_embedding:
                    continue

                similarity = self._cosine_similarity(request_embedding, rule_embedding)

                if similarity >= self._threshold:
                    matches.append(SemanticMatch(rule=rule, similarity=similarity))

            # Sort by similarity (highest first)
            matches.sort(key=lambda m: m.similarity, reverse=True)
            return matches

        except Exception as e:
            logger.warning(f"Semantic matching failed: {e}")
            return []

    def _get_rule_embedding(self, rule: Rule) -> List[float]:
        """Get or compute embedding for a rule (cached)."""
        cache_key = hashlib.sha256(rule.content.encode()).hexdigest()

        with self._cache_lock:
            if cache_key in self._rule_embeddings:
                return self._rule_embeddings[cache_key]

        try:
            # Include rule type context for better matching
            text = f"{rule.rule_type.name}: {rule.content}"
            embedding = self._client.embed(self._model, text)

            if embedding:
                with self._cache_lock:
                    self._rule_embeddings[cache_key] = embedding

            return embedding
        except Exception as e:
            logger.debug(f"Failed to embed rule {rule.id}: {e}")
            return []

    def precompute_embeddings(self, rules: List[Rule]) -> int:
        """
        Pre-compute and cache embeddings for a set of rules.

        Returns:
            Number of rules successfully embedded
        """
        count = 0
        for rule in rules:
            embedding = self._get_rule_embedding(rule)
            if embedding:
                count += 1
        return count

    def clear_cache(self) -> None:
        """Clear the embedding cache."""
        with self._cache_lock:
            self._rule_embeddings.clear()

    @staticmethod
    def _cosine_similarity(a: List[float], b: List[float]) -> float:
        """Compute cosine similarity between two vectors."""
        if len(a) != len(b) or not a:
            return 0.0

        dot_product = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return dot_product / (norm_a * norm_b)


# ---------------------------------------------------------------------------
# Tier 3: LLM Compliance Judgment
# ---------------------------------------------------------------------------

@dataclass
class ComplianceJudgment:
    """Result from LLM compliance evaluation."""
    allowed: bool
    violated_rules: List[Rule] = field(default_factory=list)
    reasoning: str = ""
    suggestions: List[str] = field(default_factory=list)
    confidence: float = 0.0


class LLMJudge:
    """
    Tier 3: LLM-backed constitutional compliance judgment.

    For requests that semantically match constitutional rules,
    asks the LLM to make a structured compliance determination.

    Features:
    - Structured JSON output
    - Decision caching (keyed by rule set + content hash)
    - Configurable timeout
    - Graceful fallback when LLM unavailable
    """

    DEFAULT_MODEL = "llama3.2:3b"
    DEFAULT_TIMEOUT = 10.0  # seconds

    PROMPT_TEMPLATE = """You are a constitutional compliance judge for Agent-OS.

Your task is to determine whether a request violates any of the provided constitutional rules.
Be precise: only flag actual violations, not superficial keyword matches.

CONSTITUTIONAL RULES (ranked by authority, highest first):
{rules}

REQUEST TO EVALUATE:
- Source: {source}
- Destination: {destination}
- Intent: {intent}
- Content: {content}

Does this request violate any of the above rules?

Respond ONLY with valid JSON (no markdown, no explanation outside the JSON):
{{
  "allowed": true or false,
  "violated_rule_ids": ["id1", "id2"],
  "reasoning": "Brief explanation of your decision",
  "suggestions": ["suggestion if denied"],
  "confidence": 0.0 to 1.0
}}"""

    def __init__(
        self,
        ollama_client: Any = None,
        model: str = DEFAULT_MODEL,
        timeout: float = DEFAULT_TIMEOUT,
    ):
        self._client = ollama_client
        self._model = model
        self._timeout = timeout

        # Cache: (rules_hash, content_hash) -> ComplianceJudgment
        self._cache: Dict[str, ComplianceJudgment] = {}
        self._cache_lock = threading.Lock()
        self._max_cache_size = 1000

    @property
    def available(self) -> bool:
        """Check if LLM judge is available."""
        if self._client is None:
            return False
        try:
            return self._client.is_healthy()
        except Exception:
            return False

    def judge(
        self,
        context: Any,
        matched_rules: List[SemanticMatch],
    ) -> ComplianceJudgment:
        """
        Ask the LLM to judge constitutional compliance.

        Args:
            context: RequestContext
            matched_rules: Rules that semantically matched (from Tier 2)

        Returns:
            ComplianceJudgment with structured decision
        """
        if not self._client:
            return ComplianceJudgment(
                allowed=False,
                confidence=0.0,
                reasoning="LLM client unavailable — conservative denial",
            )
        if not matched_rules:
            return ComplianceJudgment(
                allowed=False,
                confidence=0.0,
                reasoning="No applicable rules found — conservative denial",
            )

        rules = [m.rule for m in matched_rules]

        # Check cache
        cache_key = self._cache_key(context, rules)
        with self._cache_lock:
            if cache_key in self._cache:
                return self._cache[cache_key]

        try:
            # Format rules for the prompt
            rules_text = self._format_rules(rules)

            prompt = self.PROMPT_TEMPLATE.format(
                rules=rules_text,
                source=context.source,
                destination=context.destination,
                intent=context.intent,
                content=context.content[:2000],  # Truncate for prompt size
            )

            response = self._client.generate(
                model=self._model,
                prompt=prompt,
            )

            judgment = self._parse_response(response.content, rules)

            # Cache the result
            with self._cache_lock:
                if len(self._cache) >= self._max_cache_size:
                    # Evict oldest entries (simple FIFO)
                    oldest_key = next(iter(self._cache))
                    del self._cache[oldest_key]
                self._cache[cache_key] = judgment

            return judgment

        except Exception as e:
            logger.warning(f"LLM judge failed: {e}")
            # Fail-safe: deny if we can't evaluate
            return ComplianceJudgment(
                allowed=False,
                reasoning=f"LLM evaluation failed ({e}); denying as fail-safe",
                confidence=0.0,
            )

    def _format_rules(self, rules: List[Rule]) -> str:
        """Format rules for the LLM prompt."""
        lines = []
        for i, rule in enumerate(rules, 1):
            authority = rule.authority_level.name
            rtype = rule.rule_type.name
            immutable = " [IMMUTABLE]" if rule.is_immutable else ""
            section = " > ".join(rule.section_path) if rule.section_path else rule.section
            lines.append(
                f"{i}. [{authority}/{rtype}{immutable}] (id: {rule.id}) "
                f"[{section}]: {rule.content}"
            )
        return "\n".join(lines)

    def _parse_response(self, response_text: str, rules: List[Rule]) -> ComplianceJudgment:
        """Parse the LLM's JSON response into a ComplianceJudgment."""
        import json

        # Try to extract JSON from the response
        text = response_text.strip()

        # Handle markdown code blocks
        if "```" in text:
            match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.S)
            if match:
                text = match.group(1).strip()

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            # Try to find JSON object in the text
            match = re.search(r"\{.*\}", text, re.S)
            if match:
                try:
                    data = json.loads(match.group())
                except json.JSONDecodeError:
                    logger.warning(f"Could not parse LLM response as JSON: {text[:200]}")
                    # Conservative: deny if we can't parse
                    return ComplianceJudgment(
                        allowed=False,
                        reasoning="Could not parse LLM compliance response",
                        confidence=0.0,
                    )
            else:
                return ComplianceJudgment(
                    allowed=False,
                    reasoning="Could not parse LLM compliance response",
                    confidence=0.0,
                )

        # Map violated rule IDs to Rule objects
        violated_ids = set(data.get("violated_rule_ids", []))
        rule_map = {r.id: r for r in rules}
        violated_rules = [rule_map[rid] for rid in violated_ids if rid in rule_map]

        return ComplianceJudgment(
            allowed=data.get("allowed", True),
            violated_rules=violated_rules,
            reasoning=data.get("reasoning", ""),
            suggestions=data.get("suggestions", []),
            confidence=float(data.get("confidence", 0.5)),
        )

    def _cache_key(self, context: Any, rules: List[Rule]) -> str:
        """Generate cache key from context + rules."""
        rules_hash = hashlib.sha256(
            "|".join(sorted(r.id for r in rules)).encode()
        ).hexdigest()[:16]
        content_hash = hashlib.sha256(
            f"{context.intent}:{context.content}".encode()
        ).hexdigest()[:16]
        return f"{rules_hash}:{content_hash}"

    def clear_cache(self) -> None:
        """Clear the judgment cache."""
        with self._cache_lock:
            self._cache.clear()


# ---------------------------------------------------------------------------
# Enforcement Engine — Orchestrates all three tiers
# ---------------------------------------------------------------------------

class EnforcementEngine:
    """
    Orchestrates the three-tier constitutional enforcement pipeline.

    Usage:
        engine = EnforcementEngine(ollama_client=client)
        result = engine.evaluate(context, rules)

    Tier progression:
        Tier 1 (structural) -> definitive? -> done
                             -> pass -> Tier 2 (semantic) -> no matches? -> allow
                                                          -> matches -> Tier 3 (LLM judge)
    """

    def __init__(
        self,
        ollama_client: Any = None,
        embedding_model: str = SemanticMatcher.DEFAULT_MODEL,
        llm_model: str = LLMJudge.DEFAULT_MODEL,
        semantic_threshold: float = SemanticMatcher.DEFAULT_THRESHOLD,
        llm_timeout: float = LLMJudge.DEFAULT_TIMEOUT,
    ):
        self.structural = StructuralChecker()
        self.semantic = SemanticMatcher(
            ollama_client=ollama_client,
            model=embedding_model,
            threshold=semantic_threshold,
        )
        self.llm_judge = LLMJudge(
            ollama_client=ollama_client,
            model=llm_model,
            timeout=llm_timeout,
        )
        self._ollama_client = ollama_client

    def evaluate(self, context: Any, rules: List[Rule]) -> "EnforcementDecision":
        """
        Run the full 3-tier enforcement pipeline.

        Args:
            context: RequestContext to evaluate
            rules: All rules for the destination agent

        Returns:
            EnforcementDecision with the final verdict
        """
        # --- Tier 1: Structural checks ---
        structural_result = self.structural.check(context, rules)

        if structural_result.definitive:
            return EnforcementDecision(
                allowed=structural_result.allowed,
                tier="structural",
                reason=structural_result.reason,
                matched_rules=structural_result.matched_rules,
                escalate_to_human=any(
                    r.is_immutable for r in structural_result.matched_rules
                ),
            )

        applicable_rules = structural_result.matched_rules

        # --- Tier 2: Semantic matching ---
        if self.semantic.available and applicable_rules:
            semantic_matches = self.semantic.match(context, applicable_rules)

            if not semantic_matches:
                # No rules semantically match — conservative denial.
                # The absence of matching rules does NOT mean the request is
                # safe; it means the rules don't cover this case. Fall through
                # to LLM judge if available, otherwise deny conservatively.
                if self.llm_judge.available:
                    judgment = self.llm_judge.judge(context, [])
                    return EnforcementDecision(
                        allowed=judgment.allowed,
                        tier="llm_judge",
                        reason=judgment.reasoning or "LLM judge evaluated unmatched request",
                        matched_rules=judgment.violated_rules,
                        suggestions=judgment.suggestions,
                        confidence=judgment.confidence,
                        escalate_to_human=True,
                    )
                return EnforcementDecision(
                    allowed=False,
                    tier="semantic",
                    reason="No applicable rules found — conservative denial",
                    matched_rules=[],
                    escalate_to_human=True,
                )

            # --- Tier 3: LLM judgment ---
            if self.llm_judge.available:
                judgment = self.llm_judge.judge(context, semantic_matches)

                return EnforcementDecision(
                    allowed=judgment.allowed,
                    tier="llm_judge",
                    reason=judgment.reasoning,
                    matched_rules=judgment.violated_rules,
                    suggestions=judgment.suggestions,
                    confidence=judgment.confidence,
                    escalate_to_human=any(
                        r.is_immutable for r in judgment.violated_rules
                    ),
                )

            # LLM unavailable — fall back to keyword-based check on matched rules
            return self._keyword_fallback(context, semantic_matches)

        # Semantic unavailable — fall back to old-style keyword matching
        return self._keyword_fallback_all(context, applicable_rules)

    def _keyword_fallback(
        self, context: Any, matches: List[SemanticMatch]
    ) -> "EnforcementDecision":
        """
        Fallback when LLM is unavailable: use keyword matching
        on semantically matched rules only.
        """
        violated = []
        content_lower = context.content.lower()

        for match in matches:
            rule = match.rule
            if rule.rule_type == RuleType.PROHIBITION:
                for kw in rule.keywords:
                    if kw in content_lower:
                        violated.append(rule)
                        break

        if violated:
            return EnforcementDecision(
                allowed=False,
                tier="keyword_fallback",
                reason="LLM unavailable; keyword match on semantically matched rules",
                matched_rules=violated,
                escalate_to_human=any(r.is_immutable for r in violated),
            )

        return EnforcementDecision(
            allowed=True,
            tier="keyword_fallback",
            reason="LLM unavailable; no keyword violations in semantically matched rules",
        )

    def _keyword_fallback_all(
        self, context: Any, rules: List[Rule]
    ) -> "EnforcementDecision":
        """
        Full fallback when both semantic and LLM are unavailable:
        use the original keyword matching on all applicable rules.
        """
        violated = []
        content_lower = context.content.lower()
        intent_lower = context.intent.lower()

        for rule in rules:
            # Check if rule applies (keyword match)
            applies = False
            for kw in rule.keywords:
                if kw in content_lower or kw in intent_lower:
                    applies = True
                    break

            if not applies and context.requires_memory:
                memory_keywords = {"memory", "store", "persist", "remember", "save"}
                if rule.keywords & memory_keywords:
                    applies = True

            if not applies:
                continue

            # Check violation
            if rule.rule_type == RuleType.PROHIBITION:
                for kw in rule.keywords:
                    if kw in content_lower:
                        violated.append(rule)
                        break
            elif rule.rule_type == RuleType.MANDATE:
                compliance_indicators = {
                    "review", "validate", "verify", "check", "confirm",
                    "ensure", "approved", "authorization", "consent",
                }
                if not any(ind in content_lower for ind in compliance_indicators):
                    mandate_compliance = context.metadata.get("mandate_compliance", {})
                    if rule.id not in mandate_compliance:
                        violated.append(rule)

        if violated:
            return EnforcementDecision(
                allowed=False,
                tier="keyword_fallback_full",
                reason="Semantic/LLM unavailable; keyword-based enforcement",
                matched_rules=violated,
                escalate_to_human=any(r.is_immutable for r in violated),
            )

        return EnforcementDecision(
            allowed=True,
            tier="keyword_fallback_full",
            reason="Semantic/LLM unavailable; no keyword violations found",
        )


@dataclass
class EnforcementDecision:
    """Final decision from the enforcement engine."""
    allowed: bool
    tier: str  # Which tier made the decision
    reason: Optional[str] = None
    matched_rules: List[Rule] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    confidence: float = 1.0
    escalate_to_human: bool = False
