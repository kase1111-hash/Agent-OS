"""
Agent OS Smith Refusal Engine

Implements security checks S9-S12 for refusing harmful requests:
- S9: Authority escalation blocker
- S10: Deceptive compliance detector
- S11: Manipulation filter
- S12: Ambiguity handler
"""

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Tuple

from src.messaging.models import FlowRequest, FlowResponse, MessageStatus

logger = logging.getLogger(__name__)


class RefusalType(Enum):
    """Types of refusals."""

    HARD_BLOCK = auto()  # Absolute refusal, no appeal
    SOFT_BLOCK = auto()  # Refusal with explanation
    ESCALATE = auto()  # Escalate to human
    CLARIFY = auto()  # Request clarification
    CONSTRAIN = auto()  # Allow with constraints


@dataclass
class RefusalDecision:
    """Decision from the refusal engine."""

    refused: bool
    refusal_type: Optional[RefusalType] = None
    check_id: str = ""  # S9, S10, etc.
    reason: str = ""
    explanation: str = ""  # User-friendly explanation
    suggested_action: Optional[str] = None
    confidence: float = 1.0  # Confidence in decision
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class RefusalResponse:
    """Complete refusal response."""

    decisions: List[RefusalDecision]
    final_action: RefusalType
    message: str
    should_log: bool = True
    requires_human_review: bool = False
    review_reason: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def is_refused(self) -> bool:
        return self.final_action in (RefusalType.HARD_BLOCK, RefusalType.SOFT_BLOCK)


class RefusalEngine:
    """
    Refusal Engine implementing security checks S9-S12.

    Provides intelligent refusal handling for various attack vectors.
    """

    # Authority escalation patterns
    ESCALATION_PATTERNS = [
        (r"i\s+am\s+(your\s+)?(admin|owner|creator|developer|god)", "authority_claim"),
        (r"as\s+(the|your)\s+(admin|administrator|owner)", "authority_invocation"),
        (r"(admin|root|sudo)\s+(mode|access|privilege)", "privilege_request"),
        (r"override\s+(all|any)\s+(restrictions?|limits?|rules?)", "override_attempt"),
        (r"you\s+must\s+obey\s+(me|this)", "obedience_demand"),
        (r"i\s+order\s+you\s+to", "command_order"),
        (r"ignore\s+(the\s+)?constitution", "constitution_bypass"),
        (r"bypass\s+(smith|guardian|security)", "security_bypass"),
    ]

    # Deceptive compliance patterns
    DECEPTION_PATTERNS = [
        (r"for\s+(educational|research|academic)\s+purposes?", "purpose_disguise"),
        (r"hypothetically|in\s+theory|theoretically", "hypothetical_framing"),
        (r"pretend\s+(this\s+is|we'?re|that)", "pretend_framing"),
        (r"let'?s\s+(roleplay|play|imagine)", "roleplay_framing"),
        (r"in\s+(a|this)\s+(story|fiction|novel|game)", "fiction_framing"),
        (r"write\s+(a\s+)?(story|fiction)\s+about\s+how\s+to", "story_exploit"),
        (r"if\s+you\s+were\s+(not|without)\s+(restrictions?|limits?)", "constraint_removal"),
        (r"alternate\s+(universe|reality)\s+where", "alternate_reality"),
    ]

    # Manipulation patterns
    MANIPULATION_PATTERNS = [
        (r"if\s+you\s+don'?t.*i\s+will", "threat_pattern"),
        (r"you\s+(should|would)\s+want\s+to\s+help", "guilt_pattern"),
        (r"a\s+(good|real|true)\s+AI\s+would", "comparison_manipulation"),
        (r"prove\s+you'?re\s+(not\s+)?biased", "proof_demand"),
        (r"why\s+(can'?t|won'?t)\s+you\s+just", "pressure_pattern"),
        (r"other\s+AIs?\s+(can|do|will)", "comparison_pressure"),
        (r"you'?re\s+(being\s+)?(too|overly)\s+cautious", "caution_criticism"),
        (r"stop\s+being\s+(so\s+)?(restrictive|careful|paranoid)", "relaxation_demand"),
        (r"that'?s\s+not\s+a\s+(real\s+)?violation", "minimization"),
    ]

    # Ambiguous request patterns
    AMBIGUITY_PATTERNS = [
        (
            r"\b(it|that|this|they)\b(?!\s+(?:is|are|was|were|have|has|will|would|can|could))",
            "pronoun_ambiguity",
        ),
        (r"do\s+(?:that|it|the\s+thing)\b", "vague_action"),
        (r"you\s+know\s+what\s+i\s+mean", "implicit_understanding"),
        (r"(?:just|simply)\s+(?:do|make|give)\s+(?:it|that)", "vague_request"),
    ]

    # Refusal message templates
    REFUSAL_MESSAGES = {
        "authority_escalation": (
            "I cannot act on authority claims. My behavior is governed by my "
            "constitutional boundaries, not by user-claimed permissions."
        ),
        "deceptive_compliance": (
            "I recognize this request uses framing that might circumvent my guidelines. "
            "I cannot provide this information regardless of the framing used."
        ),
        "manipulation": (
            "I notice this request contains pressure tactics. My responses are based "
            "on my guidelines, not on social pressure or comparisons."
        ),
        "ambiguity": (
            "This request is too vague for me to safely proceed. Could you please "
            "clarify exactly what you're asking for?"
        ),
        "generic": ("I cannot fulfill this request as it conflicts with my guidelines."),
    }

    def __init__(
        self,
        strict_mode: bool = True,
        confidence_threshold: float = 0.7,
    ):
        """
        Initialize refusal engine.

        Args:
            strict_mode: Block on any positive match
            confidence_threshold: Minimum confidence to trigger refusal
        """
        self.strict_mode = strict_mode
        self.confidence_threshold = confidence_threshold

        # Compile patterns
        self._escalation_patterns = [
            (re.compile(p, re.IGNORECASE), name) for p, name in self.ESCALATION_PATTERNS
        ]
        self._deception_patterns = [
            (re.compile(p, re.IGNORECASE), name) for p, name in self.DECEPTION_PATTERNS
        ]
        self._manipulation_patterns = [
            (re.compile(p, re.IGNORECASE), name) for p, name in self.MANIPULATION_PATTERNS
        ]
        self._ambiguity_patterns = [
            (re.compile(p, re.IGNORECASE), name) for p, name in self.AMBIGUITY_PATTERNS
        ]

        # Metrics
        self._total_evaluations = 0
        self._refusals = 0
        self._escalations = 0
        self._clarifications = 0

    def evaluate(
        self,
        request: FlowRequest,
        context: Optional[Dict[str, Any]] = None,
    ) -> RefusalResponse:
        """
        Evaluate a request for refusal.

        Args:
            request: Request to evaluate
            context: Additional context

        Returns:
            RefusalResponse with decision
        """
        self._total_evaluations += 1
        context = context or {}

        decisions = []

        # Run all S9-S12 checks
        s9_decision = self._check_s9_authority_escalation(request, context)
        decisions.append(s9_decision)

        s10_decision = self._check_s10_deceptive_compliance(request, context)
        decisions.append(s10_decision)

        s11_decision = self._check_s11_manipulation(request, context)
        decisions.append(s11_decision)

        s12_decision = self._check_s12_ambiguity(request, context)
        decisions.append(s12_decision)

        # Determine final action
        final_action, message, requires_review = self._determine_final_action(decisions)

        if final_action in (RefusalType.HARD_BLOCK, RefusalType.SOFT_BLOCK):
            self._refusals += 1
        elif final_action == RefusalType.ESCALATE:
            self._escalations += 1
        elif final_action == RefusalType.CLARIFY:
            self._clarifications += 1

        return RefusalResponse(
            decisions=decisions,
            final_action=final_action,
            message=message,
            requires_human_review=requires_review,
            review_reason=decisions[0].reason if requires_review else None,
        )

    def _check_s9_authority_escalation(
        self,
        request: FlowRequest,
        context: Dict[str, Any],
    ) -> RefusalDecision:
        """
        S9: Authority Escalation Blocker

        Blocks attempts to claim elevated authority.
        """
        prompt = request.content.prompt
        detected = []

        for pattern, name in self._escalation_patterns:
            if pattern.search(prompt):
                detected.append(name)

        if detected:
            # Determine severity
            high_severity = any(
                d in detected
                for d in ["constitution_bypass", "security_bypass", "privilege_request"]
            )

            return RefusalDecision(
                refused=True,
                refusal_type=RefusalType.HARD_BLOCK if high_severity else RefusalType.SOFT_BLOCK,
                check_id="S9",
                reason=f"Authority escalation: {detected[0]}",
                explanation=self.REFUSAL_MESSAGES["authority_escalation"],
                confidence=0.95 if high_severity else 0.85,
                details={"detected_patterns": detected},
            )

        return RefusalDecision(
            refused=False,
            check_id="S9",
            reason="No authority escalation detected",
            confidence=1.0,
        )

    def _check_s10_deceptive_compliance(
        self,
        request: FlowRequest,
        context: Dict[str, Any],
    ) -> RefusalDecision:
        """
        S10: Deceptive Compliance Detector

        Detects attempts to trick into compliance through framing.
        """
        prompt = request.content.prompt
        detected = []

        for pattern, name in self._deception_patterns:
            if pattern.search(prompt):
                detected.append(name)

        if detected:
            # Some framings are more benign than others
            high_risk = any(
                d in detected for d in ["story_exploit", "constraint_removal", "alternate_reality"]
            )

            # Check if there's an underlying harmful request
            has_harmful_content = self._check_harmful_with_framing(prompt)

            if has_harmful_content or high_risk:
                return RefusalDecision(
                    refused=True,
                    refusal_type=RefusalType.SOFT_BLOCK,
                    check_id="S10",
                    reason=f"Deceptive framing: {detected[0]}",
                    explanation=self.REFUSAL_MESSAGES["deceptive_compliance"],
                    confidence=0.85 if high_risk else 0.75,
                    details={
                        "detected_patterns": detected,
                        "has_harmful_content": has_harmful_content,
                    },
                )
            else:
                # Flag but don't block benign creative requests
                return RefusalDecision(
                    refused=False,
                    refusal_type=RefusalType.CONSTRAIN,
                    check_id="S10",
                    reason=f"Framing detected but appears benign: {detected[0]}",
                    confidence=0.6,
                    details={"detected_patterns": detected},
                )

        return RefusalDecision(
            refused=False,
            check_id="S10",
            reason="No deceptive compliance detected",
            confidence=1.0,
        )

    def _check_s11_manipulation(
        self,
        request: FlowRequest,
        context: Dict[str, Any],
    ) -> RefusalDecision:
        """
        S11: Manipulation Filter

        Detects psychological manipulation tactics.
        """
        prompt = request.content.prompt
        detected = []

        for pattern, name in self._manipulation_patterns:
            if pattern.search(prompt):
                detected.append(name)

        if detected:
            # Multiple manipulation tactics indicate higher risk
            if len(detected) >= 2:
                return RefusalDecision(
                    refused=True,
                    refusal_type=RefusalType.SOFT_BLOCK,
                    check_id="S11",
                    reason=f"Multiple manipulation tactics: {', '.join(detected[:2])}",
                    explanation=self.REFUSAL_MESSAGES["manipulation"],
                    confidence=0.9,
                    details={"detected_patterns": detected},
                )

            # Single pattern - warn but potentially allow
            return RefusalDecision(
                refused=False,
                refusal_type=RefusalType.CONSTRAIN,
                check_id="S11",
                reason=f"Manipulation tactic detected: {detected[0]}",
                suggested_action="Respond factually without acknowledging pressure",
                confidence=0.7,
                details={"detected_patterns": detected},
            )

        return RefusalDecision(
            refused=False,
            check_id="S11",
            reason="No manipulation detected",
            confidence=1.0,
        )

    def _check_s12_ambiguity(
        self,
        request: FlowRequest,
        context: Dict[str, Any],
    ) -> RefusalDecision:
        """
        S12: Ambiguity Handler

        Handles ambiguous requests that could be interpreted harmfully.
        """
        prompt = request.content.prompt
        ambiguity_score = 0.0
        detected = []

        # Check for ambiguous patterns
        for pattern, name in self._ambiguity_patterns:
            matches = pattern.findall(prompt)
            if matches:
                detected.append(name)
                ambiguity_score += 0.2 * len(matches)

        # Check prompt length vs specificity
        words = prompt.split()
        if len(words) < 5:
            ambiguity_score += 0.3
            detected.append("very_short_request")
        elif len(words) < 10 and any(w in prompt.lower() for w in ["it", "that", "this"]):
            ambiguity_score += 0.2
            detected.append("short_with_pronouns")

        # Check for lack of context
        if not request.content.context and ambiguity_score > 0:
            ambiguity_score += 0.1

        # Cap at 1.0
        ambiguity_score = min(ambiguity_score, 1.0)

        if ambiguity_score > 0.5:
            return RefusalDecision(
                refused=True,
                refusal_type=RefusalType.CLARIFY,
                check_id="S12",
                reason="Request too ambiguous",
                explanation=self.REFUSAL_MESSAGES["ambiguity"],
                suggested_action="Ask user to clarify their request",
                confidence=ambiguity_score,
                details={"ambiguity_score": ambiguity_score, "detected_patterns": detected},
            )

        if detected:
            return RefusalDecision(
                refused=False,
                refusal_type=RefusalType.CONSTRAIN,
                check_id="S12",
                reason="Minor ambiguity detected",
                confidence=1.0 - ambiguity_score,
                details={"ambiguity_score": ambiguity_score, "detected_patterns": detected},
            )

        return RefusalDecision(
            refused=False,
            check_id="S12",
            reason="Request sufficiently clear",
            confidence=1.0,
        )

    def _check_harmful_with_framing(self, prompt: str) -> bool:
        """Check if prompt contains harmful content despite framing."""
        harmful_patterns = [
            r"(make|create|build)\s+(a\s+)?(bomb|weapon|explosive)",
            r"(hack|break\s+into|attack)\s+",
            r"(steal|fraud|scam)",
            r"(hurt|harm|kill|attack)\s+(someone|people|a\s+person)",
            r"(illegal|criminal)\s+(activity|method|technique)",
        ]

        for pattern in harmful_patterns:
            if re.search(pattern, prompt, re.IGNORECASE):
                return True
        return False

    def _determine_final_action(
        self,
        decisions: List[RefusalDecision],
    ) -> Tuple[RefusalType, str, bool]:
        """
        Determine final refusal action from all decisions.

        Returns:
            Tuple of (action, message, requires_human_review)
        """
        # Check for hard blocks first
        hard_blocks = [d for d in decisions if d.refusal_type == RefusalType.HARD_BLOCK]
        if hard_blocks:
            return (
                RefusalType.HARD_BLOCK,
                hard_blocks[0].explanation or self.REFUSAL_MESSAGES["generic"],
                True,  # Log hard blocks for review
            )

        # Check for soft blocks
        soft_blocks = [d for d in decisions if d.refusal_type == RefusalType.SOFT_BLOCK]
        if soft_blocks:
            return (
                RefusalType.SOFT_BLOCK,
                soft_blocks[0].explanation or self.REFUSAL_MESSAGES["generic"],
                False,
            )

        # Check for escalations
        escalations = [d for d in decisions if d.refusal_type == RefusalType.ESCALATE]
        if escalations:
            return (
                RefusalType.ESCALATE,
                "This request requires human review before proceeding.",
                True,
            )

        # Check for clarifications
        clarifications = [d for d in decisions if d.refusal_type == RefusalType.CLARIFY]
        if clarifications:
            return (
                RefusalType.CLARIFY,
                clarifications[0].explanation or self.REFUSAL_MESSAGES["ambiguity"],
                False,
            )

        # Check for constraints
        constraints = [d for d in decisions if d.refusal_type == RefusalType.CONSTRAIN]
        if constraints:
            return (
                RefusalType.CONSTRAIN,
                "Proceeding with additional constraints.",
                False,
            )

        # No issues - allow
        return (
            RefusalType.CONSTRAIN,
            "Request approved.",
            False,
        )

    def create_refusal_response(
        self,
        request: FlowRequest,
        refusal: RefusalResponse,
    ) -> FlowResponse:
        """
        Create a FlowResponse for a refused request.

        Args:
            request: Original request
            refusal: Refusal response

        Returns:
            FlowResponse with refusal message
        """
        status = MessageStatus.REFUSED if refusal.is_refused else MessageStatus.PARTIAL

        return request.create_response(
            source="smith",
            status=status,
            output=refusal.message,
            reasoning=f"Blocked by {refusal.decisions[0].check_id if refusal.decisions else 'security check'}",
        )

    def get_metrics(self) -> Dict[str, Any]:
        """Get refusal engine metrics."""
        return {
            "total_evaluations": self._total_evaluations,
            "refusals": self._refusals,
            "escalations": self._escalations,
            "clarifications": self._clarifications,
            "refusal_rate": (
                self._refusals / self._total_evaluations if self._total_evaluations > 0 else 0.0
            ),
        }
