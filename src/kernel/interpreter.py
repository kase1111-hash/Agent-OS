"""Policy Interpreter for Conversational Kernel.

Parses natural language statements into structured intents and rules.
Handles clarification dialogs for ambiguous requests.
"""

import json
import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from .rules import Rule, RuleAction, RuleEffect, RuleScope

logger = logging.getLogger(__name__)


class IntentAction(str, Enum):
    """Types of user intents."""

    SET_RULE = "set_rule"
    MODIFY_RULE = "modify_rule"
    DELETE_RULE = "delete_rule"
    QUERY_RULES = "query_rules"
    CHECK_ACCESS = "check_access"
    EXPLAIN = "explain"
    SUGGEST = "suggest"
    UNDO = "undo"


@dataclass
class ParsedIntent:
    """A parsed user intent."""

    action: IntentAction
    target: Optional[str] = None
    effect: Optional[RuleEffect] = None
    rule_actions: List[RuleAction] = field(default_factory=list)
    scope: Optional[RuleScope] = None
    conditions: Dict[str, Any] = field(default_factory=dict)
    reason: Optional[str] = None
    confidence: float = 1.0
    ambiguities: List[str] = field(default_factory=list)
    raw_text: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_ambiguous(self) -> bool:
        """Check if intent has ambiguities needing clarification."""
        return len(self.ambiguities) > 0 or self.confidence < 0.8

    def to_rule(self, rule_id: Optional[str] = None) -> Rule:
        """Convert intent to a Rule object."""
        return Rule(
            rule_id=rule_id or "",
            scope=self.scope or RuleScope.FOLDER,
            target=self.target or "",
            effect=self.effect or RuleEffect.DENY,
            actions=self.rule_actions,
            reason=self.reason or self.raw_text,
            conditions=self.conditions,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "action": self.action.value,
            "target": self.target,
            "effect": self.effect.value if self.effect else None,
            "rule_actions": [a.value for a in self.rule_actions],
            "scope": self.scope.value if self.scope else None,
            "conditions": self.conditions,
            "reason": self.reason,
            "confidence": self.confidence,
            "ambiguities": self.ambiguities,
            "raw_text": self.raw_text,
        }


class IntentParser:
    """Parses natural language into structured intents.

    Uses pattern matching and keyword extraction for local parsing.
    Can integrate with LLM for complex queries.
    """

    def __init__(self, llm_client: Optional[Any] = None):
        """Initialize parser.

        Args:
            llm_client: Optional LLM client for advanced parsing
        """
        self.llm_client = llm_client

        # Intent action patterns
        self._action_patterns = [
            (r"\b(protect|secure|lock|restrict)\b", IntentAction.SET_RULE),
            (r"\b(allow|permit|enable|grant)\b", IntentAction.SET_RULE),
            (r"\b(deny|block|prevent|forbid|disallow)\b", IntentAction.SET_RULE),
            (r"\b(update|change|modify)\s+rule\b", IntentAction.MODIFY_RULE),
            (r"\b(remove|delete)\s+rule\b", IntentAction.DELETE_RULE),
            (r"\b(show|list|what|which)\s+rules?\b", IntentAction.QUERY_RULES),
            (r"\bcan\s+\w+\s+(access|read|write|modify)\b", IntentAction.CHECK_ACCESS),
            (r"\bexplain\b", IntentAction.EXPLAIN),
            (r"\bsuggest\b", IntentAction.SUGGEST),
            (r"\bundo\b", IntentAction.UNDO),
        ]

        # Effect patterns
        self._effect_patterns = [
            (r"\b(never|don't|do not|cannot|should not|must not)\b", RuleEffect.DENY),
            (r"\b(block|deny|prevent|forbid|restrict|protect|secure)\b", RuleEffect.DENY),
            (r"\b(allow|permit|enable|grant|can)\b", RuleEffect.ALLOW),
            (r"\b(log|audit|track|monitor)\b", RuleEffect.AUDIT),
            (r"\b(ask|confirm|prompt|verify)\b", RuleEffect.PROMPT),
        ]

        # Action patterns
        self._rule_action_patterns = [
            (r"\b(read|reading|view|viewing|access|accessing|open|opening|see)\b", RuleAction.READ),
            (r"\b(write|writing|edit|editing|modify|modifying|change|changing|update|updating)\b", RuleAction.WRITE),
            (r"\b(delete|deleting|remove|removing|erase|erasing)\b", RuleAction.DELETE),
            (r"\b(execute|executing|run|running|launch|launching)\b", RuleAction.EXECUTE),
            (r"\b(create|creating|make|making|new)\b", RuleAction.CREATE),
            (r"\b(overwrite|overwriting|replace|replacing)\b", RuleAction.OVERWRITE),
            (r"\b(rename|renaming|move|moving)\b", RuleAction.RENAME),
            (r"\b(copy|copying|duplicate|duplicating)\b", RuleAction.COPY),
            (r"\bAI\s+(read|reading|access|accessing)\b", RuleAction.AI_READ),
            (r"\bAI\s+(index|indexing|embed|embedding)\b", RuleAction.AI_INDEX),
            (r"\bnetwork|internet|connect\b", RuleAction.NETWORK),
        ]

        # Scope indicators
        self._scope_patterns = [
            (r"\bsystem[-\s]?wide\b", RuleScope.SYSTEM),
            (r"\bfor\s+user\s+(\w+)\b", RuleScope.USER),
            (r"\bfor\s+agent\s+(\w+)\b", RuleScope.AGENT),
            (r"\bthis\s+folder\b", RuleScope.FOLDER),
            (r"\bthis\s+file\b", RuleScope.FILE),
        ]

        # Path extraction patterns
        self._path_patterns = [
            r'["\']([^"\']+)["\']',  # Quoted paths
            r"((?:/[\w\-.]+)+)",  # Unix paths (full capture)
            r"(~/[\w\-./]+)",  # Home-relative paths
            r"\b([\w]+/[\w\-./]+)",  # Relative paths
        ]

    def parse(self, text: str, context: Optional[Dict[str, Any]] = None) -> ParsedIntent:
        """Parse natural language text into intent.

        Args:
            text: User input text
            context: Current context (cwd, user, etc.)

        Returns:
            ParsedIntent object
        """
        context = context or {}
        text_lower = text.lower()

        intent = ParsedIntent(
            action=IntentAction.SET_RULE,
            raw_text=text,
        )

        # Extract intent action
        intent.action = self._extract_action(text_lower)

        # Extract effect
        intent.effect = self._extract_effect(text_lower)

        # Extract rule actions
        intent.rule_actions = self._extract_rule_actions(text_lower)

        # Extract target path
        intent.target = self._extract_target(text, context)

        # Extract scope
        intent.scope = self._extract_scope(text_lower, intent.target)

        # Extract conditions
        intent.conditions = self._extract_conditions(text_lower)

        # Check for ambiguities
        intent.ambiguities = self._detect_ambiguities(intent)

        # Calculate confidence
        intent.confidence = self._calculate_confidence(intent)

        # Extract reason (use cleaned version of input)
        intent.reason = self._clean_reason(text)

        return intent

    def _extract_action(self, text: str) -> IntentAction:
        """Extract the intent action from text."""
        for pattern, action in self._action_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return action
        return IntentAction.SET_RULE

    def _extract_effect(self, text: str) -> Optional[RuleEffect]:
        """Extract the rule effect from text."""
        for pattern, effect in self._effect_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return effect
        return None

    def _extract_rule_actions(self, text: str) -> List[RuleAction]:
        """Extract rule actions from text."""
        actions = []
        for pattern, action in self._rule_action_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                if action not in actions:
                    actions.append(action)

        # Default to read/write if none found
        if not actions:
            if "access" in text:
                actions = [RuleAction.READ, RuleAction.WRITE]

        return actions

    def _extract_target(
        self, text: str, context: Dict[str, Any]
    ) -> Optional[str]:
        """Extract target path from text."""
        # Try each path pattern
        for pattern in self._path_patterns:
            match = re.search(pattern, text)
            if match:
                path = match.group(1) if match.lastindex else match.group(0)

                # Expand home directory
                if path.startswith("~"):
                    home = context.get("home", "/home/user")
                    path = path.replace("~", home, 1)

                # Make relative paths absolute
                if not path.startswith("/"):
                    cwd = context.get("cwd", "/")
                    path = str(Path(cwd) / path)

                return path

        # Check for "this folder" / "this file"
        if "this folder" in text.lower() or "this directory" in text.lower():
            return context.get("cwd", ".")

        if "this file" in text.lower():
            return context.get("current_file")

        return None

    def _extract_scope(self, text: str, target: Optional[str]) -> Optional[RuleScope]:
        """Extract scope from text and target."""
        for pattern, scope in self._scope_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return scope

        # Infer from target
        if target:
            path = Path(target)
            if path.suffix:  # Has extension, likely a file
                return RuleScope.FILE
            return RuleScope.FOLDER

        return RuleScope.FOLDER

    def _extract_conditions(self, text: str) -> Dict[str, Any]:
        """Extract conditions from text."""
        conditions = {}

        # Time-based conditions
        time_patterns = [
            (r"during\s+(business\s+hours|work\s+hours)", {"start": "09:00", "end": "17:00"}),
            (r"after\s+(\d{1,2}(?::\d{2})?)\s*(?:pm|PM)?", "after_time"),
            (r"before\s+(\d{1,2}(?::\d{2})?)\s*(?:am|AM)?", "before_time"),
            (r"only\s+on\s+(weekdays|weekends)", "days"),
        ]

        for pattern, value in time_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                if isinstance(value, dict):
                    conditions["time"] = value
                else:
                    conditions[value] = match.group(1)

        # Size-based conditions
        size_match = re.search(r"files?\s+(larger|smaller|over|under)\s+(\d+)\s*(kb|mb|gb)?", text, re.IGNORECASE)
        if size_match:
            size = int(size_match.group(2))
            unit = (size_match.group(3) or "kb").lower()
            multiplier = {"kb": 1024, "mb": 1024**2, "gb": 1024**3}
            conditions["size"] = {
                "op": "gt" if size_match.group(1) in ("larger", "over") else "lt",
                "value": size * multiplier.get(unit, 1),
            }

        # Type-based conditions
        type_match = re.search(r"(text|image|video|code|document)\s+files?", text, re.IGNORECASE)
        if type_match:
            conditions["file_type"] = type_match.group(1).lower()

        return conditions

    def _detect_ambiguities(self, intent: ParsedIntent) -> List[str]:
        """Detect ambiguities in parsed intent."""
        ambiguities = []

        # Missing target
        if not intent.target and intent.action == IntentAction.SET_RULE:
            ambiguities.append("Which folder or file should this rule apply to?")

        # Missing effect with conflicting keywords
        if not intent.effect:
            ambiguities.append(
                "Should this rule allow or deny the action?"
            )

        # Missing actions
        if not intent.rule_actions and intent.action == IntentAction.SET_RULE:
            ambiguities.append(
                "What actions should be controlled (read, write, delete, etc.)?"
            )

        # Scope ambiguity
        if intent.target and not intent.scope:
            ambiguities.append(
                "Should this apply to just this item, or recursively?"
            )

        return ambiguities

    def _calculate_confidence(self, intent: ParsedIntent) -> float:
        """Calculate confidence score for parsed intent."""
        confidence = 1.0

        # Reduce for missing fields
        if not intent.target:
            confidence -= 0.3
        if not intent.effect:
            confidence -= 0.2
        if not intent.rule_actions:
            confidence -= 0.2

        # Reduce for ambiguities
        confidence -= 0.1 * len(intent.ambiguities)

        return max(0.0, min(1.0, confidence))

    def _clean_reason(self, text: str) -> str:
        """Clean text to use as rule reason."""
        # Remove common filler words
        cleaned = re.sub(r"\b(please|i want to|can you|could you)\b", "", text, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned[:200] if len(cleaned) > 200 else cleaned

    async def parse_with_llm(
        self, text: str, context: Optional[Dict[str, Any]] = None
    ) -> ParsedIntent:
        """Parse using LLM for complex queries.

        Args:
            text: User input text
            context: Current context

        Returns:
            ParsedIntent object
        """
        if not self.llm_client:
            return self.parse(text, context)

        # Build prompt for LLM
        prompt = self._build_llm_prompt(text, context)

        try:
            response = await self.llm_client.complete(prompt)
            intent = self._parse_llm_response(response, text)
            return intent
        except Exception as e:
            logger.warning(f"LLM parsing failed, using pattern matching: {e}")
            return self.parse(text, context)

    def _build_llm_prompt(self, text: str, context: Optional[Dict[str, Any]]) -> str:
        """Build prompt for LLM parsing."""
        return f"""Parse this natural language request into a structured policy intent.

User request: "{text}"

Current context:
- Working directory: {context.get('cwd', '/') if context else '/'}
- User: {context.get('user', 'unknown') if context else 'unknown'}

Extract the following:
1. action: One of [set_rule, modify_rule, delete_rule, query_rules, check_access, explain]
2. target: The file or folder path this applies to
3. effect: One of [allow, deny, audit, prompt]
4. rule_actions: List of [read, write, delete, execute, create, modify, ai_read, ai_index]
5. scope: One of [system, user, folder, file, agent]
6. conditions: Any conditions like time restrictions or file types
7. reason: A clear explanation of the rule's purpose

Respond in JSON format."""

    def _parse_llm_response(self, response: str, original_text: str) -> ParsedIntent:
        """Parse LLM response into intent."""
        try:
            data = json.loads(response)

            return ParsedIntent(
                action=IntentAction(data.get("action", "set_rule")),
                target=data.get("target"),
                effect=RuleEffect(data["effect"]) if data.get("effect") else None,
                rule_actions=[RuleAction(a) for a in data.get("rule_actions", [])],
                scope=RuleScope(data["scope"]) if data.get("scope") else None,
                conditions=data.get("conditions", {}),
                reason=data.get("reason", original_text),
                confidence=0.9,  # LLM parsing assumed higher confidence
                raw_text=original_text,
            )
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.warning(f"Failed to parse LLM response: {e}")
            return self.parse(original_text)


class PolicyInterpreter:
    """High-level interpreter combining parsing and clarification.

    Handles the full flow from user input to validated rules,
    including clarification dialogs for ambiguous requests.
    """

    def __init__(
        self,
        parser: Optional[IntentParser] = None,
        clarification_handler: Optional[Callable[[str, List[str]], str]] = None,
    ):
        """Initialize interpreter.

        Args:
            parser: Intent parser instance
            clarification_handler: Function to get user clarification
        """
        self.parser = parser or IntentParser()
        self.clarification_handler = clarification_handler
        self._history: List[ParsedIntent] = []

    def interpret(
        self, text: str, context: Optional[Dict[str, Any]] = None
    ) -> Tuple[ParsedIntent, bool]:
        """Interpret user input, handling clarification if needed.

        Args:
            text: User input
            context: Current context

        Returns:
            Tuple of (parsed_intent, needs_clarification)
        """
        intent = self.parser.parse(text, context)
        self._history.append(intent)

        needs_clarification = intent.is_ambiguous()
        return intent, needs_clarification

    async def interpret_with_clarification(
        self, text: str, context: Optional[Dict[str, Any]] = None
    ) -> ParsedIntent:
        """Interpret with automatic clarification prompts.

        Args:
            text: User input
            context: Current context

        Returns:
            Fully resolved ParsedIntent
        """
        intent, needs_clarification = self.interpret(text, context)

        while needs_clarification and self.clarification_handler:
            # Get clarification
            question = self._format_clarification_question(intent)
            response = self.clarification_handler(question, intent.ambiguities)

            # Re-parse with additional context
            combined_text = f"{text}. {response}"
            intent = self.parser.parse(combined_text, context)
            needs_clarification = intent.is_ambiguous()

        return intent

    def _format_clarification_question(self, intent: ParsedIntent) -> str:
        """Format clarification question for user."""
        if len(intent.ambiguities) == 1:
            return intent.ambiguities[0]

        questions = "\n".join(f"- {a}" for a in intent.ambiguities)
        return f"I need some clarification:\n{questions}"

    def suggest_similar_rules(
        self, intent: ParsedIntent, existing_rules: List[Rule]
    ) -> List[Tuple[Rule, str]]:
        """Suggest similar existing rules.

        Args:
            intent: Current intent
            existing_rules: Existing rules to compare

        Returns:
            List of (rule, similarity_reason) tuples
        """
        suggestions = []

        for rule in existing_rules:
            # Check target similarity
            if intent.target and rule.target:
                try:
                    intent_path = Path(intent.target)
                    rule_path = Path(rule.target)

                    # Same parent directory
                    if intent_path.parent == rule_path.parent:
                        suggestions.append(
                            (rule, f"Similar to your rule for {rule.target}")
                        )
                        continue

                    # Ancestor relationship
                    try:
                        intent_path.relative_to(rule_path)
                        suggestions.append(
                            (rule, f"This path is under {rule.target} which has rules")
                        )
                        continue
                    except ValueError:
                        pass

                except Exception:
                    pass

            # Check action similarity
            if intent.rule_actions:
                common_actions = set(intent.rule_actions) & set(rule.actions)
                if len(common_actions) >= len(intent.rule_actions) / 2:
                    suggestions.append(
                        (rule, f"Has similar actions: {[a.value for a in common_actions]}")
                    )

        return suggestions[:5]  # Return top 5 suggestions

    def get_history(self, limit: int = 10) -> List[ParsedIntent]:
        """Get recent interpretation history."""
        return self._history[-limit:]
