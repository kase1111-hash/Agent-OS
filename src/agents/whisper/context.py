"""
Agent OS Context Minimization

Minimizes context passed to agents while preserving essential information.
Implements the "need to know" principle for agent communication.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List, Dict, Any, Set
from enum import Enum, auto
import logging
import re


logger = logging.getLogger(__name__)


class ContextRelevance(Enum):
    """Relevance levels for context items."""
    ESSENTIAL = auto()    # Must be included
    RELEVANT = auto()     # Should be included if space allows
    OPTIONAL = auto()     # Include only if abundant space
    IRRELEVANT = auto()   # Should be excluded


@dataclass
class ContextItem:
    """A single item of context."""
    role: str              # "user", "assistant", "system"
    content: str
    timestamp: Optional[datetime] = None
    relevance: ContextRelevance = ContextRelevance.OPTIONAL
    token_estimate: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MinimizedContext:
    """Result of context minimization."""
    items: List[ContextItem]
    total_tokens: int
    items_removed: int
    items_truncated: int
    budget_used: float     # Percentage of budget used
    strategy_used: str


class ContextMinimizer:
    """
    Minimizes context for agent requests.

    Strategies:
    1. Relevance filtering - Remove irrelevant context
    2. Recency weighting - Prefer recent messages
    3. Token-based truncation - Trim to fit budget
    4. Summarization - Condense verbose context (optional)
    """

    # Average tokens per character (rough estimate)
    TOKENS_PER_CHAR = 0.25

    def __init__(
        self,
        default_budget: int = 4096,
        min_context_items: int = 2,
        summarize: bool = False,
    ):
        """
        Initialize minimizer.

        Args:
            default_budget: Default token budget
            min_context_items: Minimum context items to keep
            summarize: Whether to use summarization
        """
        self.default_budget = default_budget
        self.min_context_items = min_context_items
        self.summarize = summarize

    def minimize(
        self,
        context: List[Dict[str, Any]],
        prompt: str,
        budget: Optional[int] = None,
        intent: Optional[str] = None,
    ) -> MinimizedContext:
        """
        Minimize context to fit within token budget.

        Args:
            context: List of context messages
            prompt: The current prompt (for relevance scoring)
            budget: Token budget (uses default if not specified)
            intent: Optional intent for relevance scoring

        Returns:
            MinimizedContext with filtered/truncated items
        """
        budget = budget or self.default_budget

        # Reserve space for prompt (estimate)
        prompt_tokens = self._estimate_tokens(prompt)
        available_budget = max(budget - prompt_tokens - 100, 500)  # 100 token buffer

        # Convert to ContextItems and score relevance
        items = self._score_context(context, prompt, intent)

        # Apply minimization strategy
        minimized_items, stats = self._apply_strategy(items, available_budget)

        return MinimizedContext(
            items=minimized_items,
            total_tokens=stats["total_tokens"],
            items_removed=stats["items_removed"],
            items_truncated=stats["items_truncated"],
            budget_used=stats["total_tokens"] / budget if budget > 0 else 0,
            strategy_used=stats["strategy"],
        )

    def _score_context(
        self,
        context: List[Dict[str, Any]],
        prompt: str,
        intent: Optional[str],
    ) -> List[ContextItem]:
        """Score context items for relevance."""
        items = []
        prompt_lower = prompt.lower()
        prompt_words = set(re.findall(r'\w+', prompt_lower))

        for i, msg in enumerate(context):
            content = msg.get("content", "")
            role = msg.get("role", "unknown")

            # Estimate tokens
            tokens = self._estimate_tokens(content)

            # Score relevance
            relevance = self._calculate_relevance(
                content, prompt_words, role, i, len(context), intent
            )

            items.append(ContextItem(
                role=role,
                content=content,
                timestamp=msg.get("timestamp"),
                relevance=relevance,
                token_estimate=tokens,
                metadata=msg.get("metadata", {}),
            ))

        return items

    def _calculate_relevance(
        self,
        content: str,
        prompt_words: Set[str],
        role: str,
        position: int,
        total: int,
        intent: Optional[str],
    ) -> ContextRelevance:
        """Calculate relevance of a context item."""
        content_lower = content.lower()
        content_words = set(re.findall(r'\w+', content_lower))

        # Calculate word overlap
        overlap = len(prompt_words & content_words)
        overlap_ratio = overlap / max(len(prompt_words), 1)

        # Position weight (more recent = more relevant)
        recency_weight = (position + 1) / total

        # Role weight
        role_weight = {
            "user": 1.0,
            "assistant": 0.8,
            "system": 0.6,
        }.get(role, 0.5)

        # Combined score
        score = (overlap_ratio * 0.5 + recency_weight * 0.3 + role_weight * 0.2)

        # Most recent user message is always essential
        if position == total - 1 and role == "user":
            return ContextRelevance.ESSENTIAL

        # System messages at start are essential
        if position == 0 and role == "system":
            return ContextRelevance.ESSENTIAL

        # Score to relevance level
        if score >= 0.7:
            return ContextRelevance.ESSENTIAL
        elif score >= 0.4:
            return ContextRelevance.RELEVANT
        elif score >= 0.2:
            return ContextRelevance.OPTIONAL
        else:
            return ContextRelevance.IRRELEVANT

    def _apply_strategy(
        self,
        items: List[ContextItem],
        budget: int,
    ) -> tuple:
        """Apply minimization strategy to fit budget."""
        stats = {
            "total_tokens": 0,
            "items_removed": 0,
            "items_truncated": 0,
            "strategy": "relevance_filter",
        }

        # Strategy 1: Filter by relevance
        essential = [i for i in items if i.relevance == ContextRelevance.ESSENTIAL]
        relevant = [i for i in items if i.relevance == ContextRelevance.RELEVANT]
        optional = [i for i in items if i.relevance == ContextRelevance.OPTIONAL]

        result = []
        current_tokens = 0

        # Always include essential items
        for item in essential:
            result.append(item)
            current_tokens += item.token_estimate

        # Add relevant items if space
        for item in relevant:
            if current_tokens + item.token_estimate <= budget:
                result.append(item)
                current_tokens += item.token_estimate
            else:
                stats["items_removed"] += 1

        # Add optional items if still space
        for item in optional:
            if current_tokens + item.token_estimate <= budget:
                result.append(item)
                current_tokens += item.token_estimate
            else:
                stats["items_removed"] += 1

        # Count removed irrelevant items
        irrelevant_count = len([i for i in items if i.relevance == ContextRelevance.IRRELEVANT])
        stats["items_removed"] += irrelevant_count

        # Strategy 2: Truncate if still over budget
        if current_tokens > budget:
            result, truncated = self._truncate_items(result, budget)
            stats["items_truncated"] = truncated
            stats["strategy"] = "truncation"
            current_tokens = sum(i.token_estimate for i in result)

        # Ensure minimum items
        if len(result) < self.min_context_items and len(items) >= self.min_context_items:
            # Add most recent items
            recent = items[-self.min_context_items:]
            for item in recent:
                if item not in result:
                    result.append(item)
            stats["strategy"] = "minimum_guarantee"

        # Sort by original order (using timestamp or position)
        result.sort(key=lambda x: items.index(x) if x in items else 0)

        stats["total_tokens"] = sum(i.token_estimate for i in result)
        return result, stats

    def _truncate_items(
        self,
        items: List[ContextItem],
        budget: int,
    ) -> tuple:
        """Truncate items to fit budget."""
        truncated_count = 0
        current_tokens = sum(i.token_estimate for i in items)

        if current_tokens <= budget:
            return items, 0

        # Start with optional items, then relevant, keep essential
        for relevance_level in [ContextRelevance.OPTIONAL, ContextRelevance.RELEVANT]:
            for item in items:
                if item.relevance == relevance_level and current_tokens > budget:
                    # Calculate how much to keep
                    target_tokens = max(item.token_estimate // 2, 50)
                    if item.token_estimate > target_tokens:
                        # Truncate content
                        char_limit = int(target_tokens / self.TOKENS_PER_CHAR)
                        item.content = item.content[:char_limit] + "..."
                        saved = item.token_estimate - target_tokens
                        item.token_estimate = target_tokens
                        current_tokens -= saved
                        truncated_count += 1

        return items, truncated_count

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count for text."""
        return int(len(text) * self.TOKENS_PER_CHAR)

    def to_messages(
        self,
        minimized: MinimizedContext,
    ) -> List[Dict[str, Any]]:
        """Convert minimized context back to message format."""
        return [
            {
                "role": item.role,
                "content": item.content,
            }
            for item in minimized.items
        ]


def minimize_context(
    context: List[Dict[str, Any]],
    prompt: str,
    budget: int = 4096,
    **kwargs,
) -> List[Dict[str, Any]]:
    """
    Convenience function to minimize context.

    Args:
        context: Context messages
        prompt: Current prompt
        budget: Token budget
        **kwargs: Additional minimizer options

    Returns:
        Minimized context messages
    """
    minimizer = ContextMinimizer(**kwargs)
    result = minimizer.minimize(context, prompt, budget)
    return minimizer.to_messages(result)
