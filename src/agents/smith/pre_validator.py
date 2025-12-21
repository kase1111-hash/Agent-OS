"""
Agent OS Smith Pre-Execution Validator

Implements security checks S1-S5 that run BEFORE agent execution:
- S1: Role boundary check
- S2: Irreversible action gate
- S3: Instruction integrity
- S4: Memory authority
- S5: External interface blocker
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List, Dict, Any, Set
from enum import Enum, auto
import logging
import re

from src.messaging.models import FlowRequest, MessageStatus
from src.core.constitution import ConstitutionalKernel


logger = logging.getLogger(__name__)


class CheckResult(Enum):
    """Result of a security check."""
    PASS = auto()
    FAIL = auto()
    WARN = auto()
    ESCALATE = auto()  # Requires human approval


@dataclass
class ValidationCheck:
    """Result of a single validation check."""
    check_id: str  # S1, S2, etc.
    name: str
    result: CheckResult
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def passed(self) -> bool:
        return self.result in (CheckResult.PASS, CheckResult.WARN)


@dataclass
class PreValidationResult:
    """Result of pre-execution validation."""
    approved: bool
    checks: List[ValidationCheck] = field(default_factory=list)
    blocked_by: Optional[str] = None  # Check ID that blocked
    requires_escalation: bool = False
    escalation_reason: Optional[str] = None
    constraints: List[str] = field(default_factory=list)
    validation_time_ms: int = 0
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def failed_checks(self) -> List[ValidationCheck]:
        return [c for c in self.checks if c.result == CheckResult.FAIL]

    @property
    def warning_checks(self) -> List[ValidationCheck]:
        return [c for c in self.checks if c.result == CheckResult.WARN]


class PreExecutionValidator:
    """
    Pre-Execution Validator implementing security checks S1-S5.

    These checks run BEFORE any agent processes a request.
    """

    # Known agent roles and their allowed actions
    ROLE_PERMISSIONS = {
        "whisper": {"route", "classify", "aggregate"},
        "sage": {"reason", "analyze", "synthesize", "query"},
        "muse": {"create", "generate", "compose"},
        "quill": {"format", "edit", "refine"},
        "seshat": {"recall", "store", "search"},
        "smith": {"validate", "monitor", "halt"},
    }

    # Irreversible action patterns
    IRREVERSIBLE_PATTERNS = [
        (r"\b(delete|remove|erase|purge)\b.*\b(all|everything|database|system)\b", "mass_deletion"),
        (r"\b(format|wipe)\b.*\b(disk|drive|storage)\b", "storage_wipe"),
        (r"\b(drop|truncate)\b.*\b(table|database|collection)\b", "database_drop"),
        (r"\b(rm|del)\s+-rf?\b", "recursive_delete"),
        (r"\bshutdown\b.*\b(system|server)\b", "system_shutdown"),
        (r"\b(terminate|kill)\b.*\ball\b", "mass_termination"),
    ]

    # Instruction injection patterns
    INJECTION_PATTERNS = [
        (r"ignore\s+(previous|prior|all)\s+instructions?", "instruction_override"),
        (r"forget\s+(your|all)\s+(rules?|instructions?)", "rule_bypass"),
        (r"you\s+are\s+now\s+(?:a|an)\s+", "role_reassignment"),
        (r"pretend\s+(?:to\s+be|you\s+are)", "identity_spoofing"),
        (r"act\s+as\s+(?:if|though)", "behavior_override"),
        (r"jailbreak", "jailbreak_attempt"),
        (r"DAN\s*mode", "jailbreak_attempt"),
        (r"bypass\s+(?:security|safety|restrictions)", "security_bypass"),
    ]

    # External interface patterns (blocked)
    EXTERNAL_INTERFACE_PATTERNS = [
        (r"\b(curl|wget|fetch)\s+https?://", "http_request"),
        (r"\b(ssh|telnet|ftp)\s+\S+", "remote_connection"),
        (r"\bsocket\s*\.\s*connect", "socket_connection"),
        (r"\brequests?\s*\.\s*(get|post|put|delete)", "http_library"),
        (r"\b(smtp|imap|pop3)\b", "email_protocol"),
        (r"\bexec\s*\(", "code_execution"),
        (r"\beval\s*\(", "code_evaluation"),
        (r"\b(subprocess|os\.system|popen)\b", "process_spawn"),
    ]

    def __init__(
        self,
        kernel: Optional[ConstitutionalKernel] = None,
        strict_mode: bool = True,
        allow_escalation: bool = True,
    ):
        """
        Initialize pre-execution validator.

        Args:
            kernel: Constitutional kernel for rule validation
            strict_mode: Fail on any check failure
            allow_escalation: Allow escalation to human for edge cases
        """
        self.kernel = kernel
        self.strict_mode = strict_mode
        self.allow_escalation = allow_escalation

        # Compile regex patterns
        self._irreversible_patterns = [
            (re.compile(p, re.IGNORECASE), name)
            for p, name in self.IRREVERSIBLE_PATTERNS
        ]
        self._injection_patterns = [
            (re.compile(p, re.IGNORECASE), name)
            for p, name in self.INJECTION_PATTERNS
        ]
        self._external_patterns = [
            (re.compile(p, re.IGNORECASE), name)
            for p, name in self.EXTERNAL_INTERFACE_PATTERNS
        ]

        # Metrics
        self._total_validations = 0
        self._blocked_count = 0
        self._escalated_count = 0

    def validate(
        self,
        request: FlowRequest,
        target_agent: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> PreValidationResult:
        """
        Run all pre-execution checks on a request.

        Args:
            request: The request to validate
            target_agent: The agent that will process the request
            context: Additional context for validation

        Returns:
            PreValidationResult with all check results
        """
        import time
        start_time = time.time()
        self._total_validations += 1
        context = context or {}

        checks = []
        constraints = []
        requires_escalation = False
        escalation_reason = None

        # Run all S1-S5 checks
        s1_result = self._check_s1_role_boundary(request, target_agent, context)
        checks.append(s1_result)

        s2_result = self._check_s2_irreversible_action(request, context)
        checks.append(s2_result)

        s3_result = self._check_s3_instruction_integrity(request, context)
        checks.append(s3_result)

        s4_result = self._check_s4_memory_authority(request, context)
        checks.append(s4_result)

        s5_result = self._check_s5_external_interface(request, context)
        checks.append(s5_result)

        # Determine overall result
        failed = [c for c in checks if c.result == CheckResult.FAIL]
        escalated = [c for c in checks if c.result == CheckResult.ESCALATE]

        if failed:
            self._blocked_count += 1
            approved = False
            blocked_by = failed[0].check_id
        elif escalated:
            self._escalated_count += 1
            if self.allow_escalation:
                approved = False  # Block pending human approval
                requires_escalation = True
                escalation_reason = "; ".join(c.message for c in escalated)
                blocked_by = None  # Escalation, not a block
            else:
                approved = False
                blocked_by = escalated[0].check_id
        else:
            approved = True
            blocked_by = None

        # Collect constraints from passing checks
        for check in checks:
            if check.result == CheckResult.WARN:
                constraints.append(f"[{check.check_id}] {check.message}")

        validation_time_ms = int((time.time() - start_time) * 1000)

        return PreValidationResult(
            approved=approved,
            checks=checks,
            blocked_by=blocked_by if not approved else None,
            requires_escalation=requires_escalation,
            escalation_reason=escalation_reason,
            constraints=constraints,
            validation_time_ms=validation_time_ms,
        )

    def _check_s1_role_boundary(
        self,
        request: FlowRequest,
        target_agent: str,
        context: Dict[str, Any],
    ) -> ValidationCheck:
        """
        S1: Role Boundary Check

        Ensures the target agent is authorized to handle the request type.
        """
        target_lower = target_agent.lower()
        allowed_actions = self.ROLE_PERMISSIONS.get(target_lower, set())

        # Extract intent from request
        intent = request.intent
        prompt_lower = request.content.prompt.lower()

        # Check for actions outside role
        violations = []

        # Whisper-specific: only routing
        if target_lower == "whisper":
            if any(action in prompt_lower for action in ["execute", "run", "perform"]):
                violations.append("Whisper cannot execute actions, only route")

        # Sage: cannot create content
        if target_lower == "sage":
            if any(word in prompt_lower for word in ["write", "compose", "create story"]):
                violations.append("Sage handles reasoning, not creative generation")

        # Seshat: memory operations need explicit consent check
        if target_lower == "seshat":
            if "store" in prompt_lower or "remember" in prompt_lower:
                # Will be handled by S4, but flag for awareness
                pass

        # Smith: cannot be bypassed
        if target_lower != "smith":
            if "skip smith" in prompt_lower or "bypass validation" in prompt_lower:
                violations.append("Smith validation cannot be bypassed")

        if violations:
            return ValidationCheck(
                check_id="S1",
                name="Role Boundary Check",
                result=CheckResult.FAIL,
                message=violations[0],
                details={"violations": violations, "target_agent": target_agent},
            )

        return ValidationCheck(
            check_id="S1",
            name="Role Boundary Check",
            result=CheckResult.PASS,
            message=f"Agent {target_agent} authorized for request",
            details={"target_agent": target_agent, "allowed_actions": list(allowed_actions)},
        )

    def _check_s2_irreversible_action(
        self,
        request: FlowRequest,
        context: Dict[str, Any],
    ) -> ValidationCheck:
        """
        S2: Irreversible Action Gate

        Blocks or escalates potentially irreversible operations.
        """
        prompt = request.content.prompt

        detected = []
        for pattern, name in self._irreversible_patterns:
            if pattern.search(prompt):
                detected.append(name)

        if detected:
            # Check if explicitly authorized in context
            if context.get("irreversible_authorized"):
                return ValidationCheck(
                    check_id="S2",
                    name="Irreversible Action Gate",
                    result=CheckResult.WARN,
                    message=f"Irreversible action authorized: {detected[0]}",
                    details={"detected_actions": detected, "authorized": True},
                )

            # Escalate for human approval
            return ValidationCheck(
                check_id="S2",
                name="Irreversible Action Gate",
                result=CheckResult.ESCALATE,
                message=f"Irreversible action detected: {detected[0]}",
                details={"detected_actions": detected, "requires_human_approval": True},
            )

        return ValidationCheck(
            check_id="S2",
            name="Irreversible Action Gate",
            result=CheckResult.PASS,
            message="No irreversible actions detected",
            details={},
        )

    def _check_s3_instruction_integrity(
        self,
        request: FlowRequest,
        context: Dict[str, Any],
    ) -> ValidationCheck:
        """
        S3: Instruction Integrity

        Detects attempts to override or manipulate agent instructions.
        """
        prompt = request.content.prompt

        detected = []
        for pattern, name in self._injection_patterns:
            if pattern.search(prompt):
                detected.append(name)

        if detected:
            return ValidationCheck(
                check_id="S3",
                name="Instruction Integrity",
                result=CheckResult.FAIL,
                message=f"Instruction manipulation detected: {detected[0]}",
                details={"detected_patterns": detected},
            )

        # Additional check: excessive prompt length (potential injection hiding)
        if len(prompt) > 50000:
            return ValidationCheck(
                check_id="S3",
                name="Instruction Integrity",
                result=CheckResult.WARN,
                message="Unusually long prompt detected",
                details={"prompt_length": len(prompt)},
            )

        return ValidationCheck(
            check_id="S3",
            name="Instruction Integrity",
            result=CheckResult.PASS,
            message="Instruction integrity verified",
            details={},
        )

    def _check_s4_memory_authority(
        self,
        request: FlowRequest,
        context: Dict[str, Any],
    ) -> ValidationCheck:
        """
        S4: Memory Authority

        Ensures memory operations have proper authorization.
        """
        prompt_lower = request.content.prompt.lower()
        metadata = request.content.metadata

        # Detect memory operations
        memory_write = any(word in prompt_lower for word in [
            "remember", "store", "save", "memorize", "keep", "record"
        ])
        memory_read = any(word in prompt_lower for word in [
            "recall", "retrieve", "what did i", "remember when"
        ])
        memory_delete = any(word in prompt_lower for word in [
            "forget", "delete memory", "erase memory", "purge memory"
        ])

        # Check for memory write without consent
        if memory_write:
            if not (metadata and metadata.requires_memory):
                return ValidationCheck(
                    check_id="S4",
                    name="Memory Authority",
                    result=CheckResult.ESCALATE,
                    message="Memory write requires explicit consent",
                    details={"operation": "write", "consent_required": True},
                )

        # Check for memory delete (always escalate)
        if memory_delete:
            return ValidationCheck(
                check_id="S4",
                name="Memory Authority",
                result=CheckResult.ESCALATE,
                message="Memory deletion requires human confirmation",
                details={"operation": "delete", "requires_confirmation": True},
            )

        # Memory read is generally allowed but logged
        if memory_read:
            return ValidationCheck(
                check_id="S4",
                name="Memory Authority",
                result=CheckResult.WARN,
                message="Memory access will be logged",
                details={"operation": "read", "logged": True},
            )

        return ValidationCheck(
            check_id="S4",
            name="Memory Authority",
            result=CheckResult.PASS,
            message="No memory operations detected",
            details={},
        )

    def _check_s5_external_interface(
        self,
        request: FlowRequest,
        context: Dict[str, Any],
    ) -> ValidationCheck:
        """
        S5: External Interface Blocker

        Blocks attempts to access external systems or networks.
        """
        prompt = request.content.prompt

        detected = []
        for pattern, name in self._external_patterns:
            if pattern.search(prompt):
                detected.append(name)

        if detected:
            # Check if external access is explicitly authorized
            if context.get("external_access_authorized"):
                return ValidationCheck(
                    check_id="S5",
                    name="External Interface Blocker",
                    result=CheckResult.WARN,
                    message=f"External access authorized: {detected[0]}",
                    details={"detected_interfaces": detected, "authorized": True},
                )

            return ValidationCheck(
                check_id="S5",
                name="External Interface Blocker",
                result=CheckResult.FAIL,
                message=f"Unauthorized external access: {detected[0]}",
                details={"detected_interfaces": detected},
            )

        return ValidationCheck(
            check_id="S5",
            name="External Interface Blocker",
            result=CheckResult.PASS,
            message="No external interface access detected",
            details={},
        )

    def get_metrics(self) -> Dict[str, Any]:
        """Get validation metrics."""
        return {
            "total_validations": self._total_validations,
            "blocked_count": self._blocked_count,
            "escalated_count": self._escalated_count,
            "block_rate": (
                self._blocked_count / self._total_validations
                if self._total_validations > 0
                else 0.0
            ),
        }
