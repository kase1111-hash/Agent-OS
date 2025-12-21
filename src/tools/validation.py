"""
Smith Tool Approval Integration

Implements mandatory Smith validation for tool invocations.
All tool invocations must be approved by Smith before execution.
"""

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Set

from .interface import (
    ToolSchema,
    ToolCategory,
    ToolRiskLevel,
    ToolInvocation,
)
from .registry import ToolRegistration


logger = logging.getLogger(__name__)


class ApprovalResult(Enum):
    """Result of Smith approval check."""
    APPROVED = auto()
    DENIED = auto()
    ESCALATE = auto()        # Requires human approval
    PENDING_REVIEW = auto()  # Needs further analysis


class DenialReason(Enum):
    """Reasons for denying tool invocation."""
    TOOL_NOT_REGISTERED = "tool_not_registered"
    TOOL_NOT_APPROVED = "tool_not_approved"
    TOOL_DISABLED = "tool_disabled"
    INSUFFICIENT_PERMISSIONS = "insufficient_permissions"
    RISK_LEVEL_EXCEEDED = "risk_level_exceeded"
    DANGEROUS_PARAMETERS = "dangerous_parameters"
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"
    SECURITY_PATTERN_MATCHED = "security_pattern_matched"
    CONSTITUTIONAL_VIOLATION = "constitutional_violation"
    EXTERNAL_INTERFACE_BLOCKED = "external_interface_blocked"
    IRREVERSIBLE_ACTION = "irreversible_action"


@dataclass
class SecurityCheck:
    """A security check performed on tool invocation."""
    check_id: str
    name: str
    passed: bool
    severity: str  # "info", "warning", "error", "critical"
    message: str
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolApprovalResult:
    """Result of tool invocation approval."""
    approved: bool
    result: ApprovalResult
    denial_reason: Optional[DenialReason] = None
    denial_message: Optional[str] = None
    checks: List[SecurityCheck] = field(default_factory=list)
    constraints: List[str] = field(default_factory=list)  # Constraints on execution
    requires_confirmation: bool = False
    requires_human_approval: bool = False
    timeout_override: Optional[int] = None
    sandbox_required: bool = True
    audit_required: bool = True
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def failed_checks(self) -> List[SecurityCheck]:
        return [c for c in self.checks if not c.passed]

    @property
    def critical_failures(self) -> List[SecurityCheck]:
        return [c for c in self.checks if not c.passed and c.severity == "critical"]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "approved": self.approved,
            "result": self.result.name,
            "denial_reason": self.denial_reason.value if self.denial_reason else None,
            "denial_message": self.denial_message,
            "checks": [
                {
                    "check_id": c.check_id,
                    "name": c.name,
                    "passed": c.passed,
                    "severity": c.severity,
                    "message": c.message,
                }
                for c in self.checks
            ],
            "constraints": self.constraints,
            "requires_confirmation": self.requires_confirmation,
            "requires_human_approval": self.requires_human_approval,
            "sandbox_required": self.sandbox_required,
            "timestamp": self.timestamp.isoformat(),
        }


class ToolApprovalValidator:
    """
    Smith-based tool approval validator.

    Performs security checks on tool invocations including:
    - T1: Tool registration check
    - T2: Permission verification
    - T3: Risk level evaluation
    - T4: Parameter inspection
    - T5: Security pattern matching
    - T6: Rate limiting
    """

    # Dangerous parameter patterns
    DANGEROUS_PARAM_PATTERNS = [
        # Path traversal
        (r"\.\./|\.\.\\", "path_traversal", "Path traversal attempt detected"),
        # Command injection
        (r"[;&|`$]", "command_injection", "Potential command injection"),
        # SQL injection
        (r"('|\")\s*(or|and)\s*('|\")?\s*[=1]", "sql_injection", "Potential SQL injection"),
        (r";\s*(drop|delete|truncate|update)\s+", "sql_injection", "Dangerous SQL command"),
        # Script injection
        (r"<script\b|javascript:", "xss", "Script injection attempt"),
        # Sensitive paths
        (r"/etc/passwd|/etc/shadow|\.ssh/", "sensitive_path", "Access to sensitive system path"),
        # Network addresses (internal)
        (r"127\.0\.0\.1|localhost|0\.0\.0\.0|::1", "internal_network", "Internal network address"),
        (r"192\.168\.|10\.|172\.(1[6-9]|2[0-9]|3[01])\.", "private_network", "Private network address"),
    ]

    # Blocked external interfaces
    EXTERNAL_INTERFACE_PATTERNS = [
        (r"^(http|https|ftp|ssh|telnet)://", "network_protocol"),
        (r"@\w+\.\w+", "email_address"),
        (r"socket\s*\.\s*connect", "socket_connection"),
    ]

    # Irreversible action patterns
    IRREVERSIBLE_PATTERNS = [
        (r"\b(rm|del)\s+-rf?\b", "recursive_delete"),
        (r"\b(drop|truncate)\s+(database|table)\b", "database_drop"),
        (r"\bformat\s+\w+:", "disk_format"),
        (r"\bkill\s+-9\b", "force_kill"),
    ]

    def __init__(
        self,
        max_risk_auto_approve: ToolRiskLevel = ToolRiskLevel.LOW,
        require_human_above: ToolRiskLevel = ToolRiskLevel.HIGH,
        blocked_categories: Optional[Set[ToolCategory]] = None,
        rate_limit_per_minute: int = 60,
    ):
        """
        Initialize validator.

        Args:
            max_risk_auto_approve: Max risk level for auto-approval
            require_human_above: Require human approval above this level
            blocked_categories: Categories that are always blocked
            rate_limit_per_minute: Max invocations per minute
        """
        self.max_risk_auto_approve = max_risk_auto_approve
        self.require_human_above = require_human_above
        self.blocked_categories = blocked_categories or set()
        self.rate_limit_per_minute = rate_limit_per_minute
        self._invocation_counts: Dict[str, List[datetime]] = {}  # user_id -> timestamps

    def validate(
        self,
        registration: Optional[ToolRegistration],
        parameters: Dict[str, Any],
        user_id: str,
        agent_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> ToolApprovalResult:
        """
        Validate a tool invocation request.

        Args:
            registration: Tool registration (None if not found)
            parameters: Invocation parameters
            user_id: User requesting invocation
            agent_id: Agent requesting invocation
            context: Additional context

        Returns:
            ToolApprovalResult
        """
        context = context or {}
        checks: List[SecurityCheck] = []
        constraints: List[str] = []

        # T1: Tool registration check
        t1_check = self._check_registration(registration)
        checks.append(t1_check)

        if not t1_check.passed:
            return ToolApprovalResult(
                approved=False,
                result=ApprovalResult.DENIED,
                denial_reason=DenialReason.TOOL_NOT_REGISTERED,
                denial_message=t1_check.message,
                checks=checks,
            )

        schema = registration.schema

        # T2: Category check
        t2_check = self._check_category(schema)
        checks.append(t2_check)

        if not t2_check.passed:
            return ToolApprovalResult(
                approved=False,
                result=ApprovalResult.DENIED,
                denial_reason=DenialReason.CONSTITUTIONAL_VIOLATION,
                denial_message=t2_check.message,
                checks=checks,
            )

        # T3: Risk level evaluation
        t3_check = self._check_risk_level(schema)
        checks.append(t3_check)

        requires_human = schema.risk_level.value > self.require_human_above.value

        if not t3_check.passed and t3_check.severity == "critical":
            return ToolApprovalResult(
                approved=False,
                result=ApprovalResult.DENIED,
                denial_reason=DenialReason.RISK_LEVEL_EXCEEDED,
                denial_message=t3_check.message,
                checks=checks,
            )

        # T4: Parameter inspection
        t4_checks = self._check_parameters(parameters, schema)
        checks.extend(t4_checks)

        param_failures = [c for c in t4_checks if not c.passed and c.severity == "critical"]
        if param_failures:
            return ToolApprovalResult(
                approved=False,
                result=ApprovalResult.DENIED,
                denial_reason=DenialReason.DANGEROUS_PARAMETERS,
                denial_message="; ".join(c.message for c in param_failures),
                checks=checks,
            )

        # T5: Security pattern matching
        t5_checks = self._check_security_patterns(parameters)
        checks.extend(t5_checks)

        security_failures = [c for c in t5_checks if not c.passed and c.severity == "critical"]
        if security_failures:
            return ToolApprovalResult(
                approved=False,
                result=ApprovalResult.DENIED,
                denial_reason=DenialReason.SECURITY_PATTERN_MATCHED,
                denial_message="; ".join(c.message for c in security_failures),
                checks=checks,
            )

        # T6: External interface check
        t6_checks = self._check_external_interfaces(parameters, schema)
        checks.extend(t6_checks)

        external_failures = [c for c in t6_checks if not c.passed and c.severity == "critical"]
        if external_failures:
            return ToolApprovalResult(
                approved=False,
                result=ApprovalResult.DENIED,
                denial_reason=DenialReason.EXTERNAL_INTERFACE_BLOCKED,
                denial_message="; ".join(c.message for c in external_failures),
                checks=checks,
            )

        # T7: Irreversible action check
        t7_checks = self._check_irreversible_actions(parameters)
        checks.extend(t7_checks)

        irreversible_detected = any(
            not c.passed and c.severity in ("error", "critical")
            for c in t7_checks
        )

        if irreversible_detected:
            requires_human = True
            constraints.append("Requires human approval: irreversible action detected")

        # T8: Rate limiting
        t8_check = self._check_rate_limit(user_id)
        checks.append(t8_check)

        if not t8_check.passed:
            return ToolApprovalResult(
                approved=False,
                result=ApprovalResult.DENIED,
                denial_reason=DenialReason.RATE_LIMIT_EXCEEDED,
                denial_message=t8_check.message,
                checks=checks,
            )

        # Determine approval result
        has_warnings = any(
            not c.passed and c.severity == "warning"
            for c in checks
        )

        requires_confirmation = (
            schema.requires_confirmation or
            schema.risk_level.value > self.max_risk_auto_approve.value or
            has_warnings
        )

        # Add constraints based on risk
        if schema.risk_level.value >= ToolRiskLevel.MEDIUM.value:
            constraints.append("Sandbox execution required")
            constraints.append("Output must be sanitized")

        if schema.risk_level.value >= ToolRiskLevel.HIGH.value:
            constraints.append("Network access restricted")
            constraints.append("Filesystem write restricted")

        # Record invocation for rate limiting
        self._record_invocation(user_id)

        # Determine final result
        if requires_human:
            return ToolApprovalResult(
                approved=True,
                result=ApprovalResult.ESCALATE,
                checks=checks,
                constraints=constraints,
                requires_confirmation=True,
                requires_human_approval=True,
                sandbox_required=True,
            )

        return ToolApprovalResult(
            approved=True,
            result=ApprovalResult.APPROVED,
            checks=checks,
            constraints=constraints,
            requires_confirmation=requires_confirmation,
            requires_human_approval=False,
            sandbox_required=schema.risk_level.value >= ToolRiskLevel.MEDIUM.value,
        )

    def _check_registration(self, registration: Optional[ToolRegistration]) -> SecurityCheck:
        """T1: Check tool registration status."""
        if not registration:
            return SecurityCheck(
                check_id="T1",
                name="Tool Registration",
                passed=False,
                severity="critical",
                message="Tool is not registered",
            )

        if not registration.is_available:
            return SecurityCheck(
                check_id="T1",
                name="Tool Registration",
                passed=False,
                severity="critical",
                message=f"Tool is not available (status: {registration.status.name})",
            )

        return SecurityCheck(
            check_id="T1",
            name="Tool Registration",
            passed=True,
            severity="info",
            message="Tool is registered and available",
        )

    def _check_category(self, schema: ToolSchema) -> SecurityCheck:
        """T2: Check if category is allowed."""
        if schema.category in self.blocked_categories:
            return SecurityCheck(
                check_id="T2",
                name="Category Check",
                passed=False,
                severity="critical",
                message=f"Tool category '{schema.category.value}' is blocked",
            )

        return SecurityCheck(
            check_id="T2",
            name="Category Check",
            passed=True,
            severity="info",
            message=f"Tool category '{schema.category.value}' is allowed",
        )

    def _check_risk_level(self, schema: ToolSchema) -> SecurityCheck:
        """T3: Evaluate risk level."""
        risk = schema.risk_level

        if risk == ToolRiskLevel.CRITICAL:
            return SecurityCheck(
                check_id="T3",
                name="Risk Level",
                passed=False,
                severity="critical",
                message="Tool has CRITICAL risk level - requires special authorization",
            )

        if risk == ToolRiskLevel.HIGH:
            return SecurityCheck(
                check_id="T3",
                name="Risk Level",
                passed=True,
                severity="warning",
                message="Tool has HIGH risk level - human approval required",
            )

        if risk == ToolRiskLevel.MEDIUM:
            return SecurityCheck(
                check_id="T3",
                name="Risk Level",
                passed=True,
                severity="info",
                message="Tool has MEDIUM risk level - confirmation required",
            )

        return SecurityCheck(
            check_id="T3",
            name="Risk Level",
            passed=True,
            severity="info",
            message="Tool has LOW risk level",
        )

    def _check_parameters(
        self,
        parameters: Dict[str, Any],
        schema: ToolSchema,
    ) -> List[SecurityCheck]:
        """T4: Inspect parameters for security issues."""
        checks = []

        for param_name, param_value in parameters.items():
            if param_value is None:
                continue

            # Convert to string for pattern matching
            value_str = str(param_value)

            for pattern, issue_type, message in self.DANGEROUS_PARAM_PATTERNS:
                if re.search(pattern, value_str, re.IGNORECASE):
                    checks.append(SecurityCheck(
                        check_id=f"T4-{issue_type}",
                        name=f"Parameter Check ({param_name})",
                        passed=False,
                        severity="critical" if issue_type in (
                            "command_injection", "sql_injection", "path_traversal", "sensitive_path"
                        ) else "warning",
                        message=f"{message} in parameter '{param_name}'",
                        details={"param": param_name, "issue": issue_type},
                    ))

        if not checks:
            checks.append(SecurityCheck(
                check_id="T4",
                name="Parameter Check",
                passed=True,
                severity="info",
                message="No dangerous patterns detected in parameters",
            ))

        return checks

    def _check_security_patterns(
        self,
        parameters: Dict[str, Any],
    ) -> List[SecurityCheck]:
        """T5: Match against security patterns."""
        checks = []

        # Combine all parameter values for pattern matching
        all_values = " ".join(str(v) for v in parameters.values() if v is not None)

        # Check for credential patterns
        if re.search(r"password\s*[:=]|api[_-]?key\s*[:=]|secret\s*[:=]", all_values, re.I):
            checks.append(SecurityCheck(
                check_id="T5-credentials",
                name="Credential Detection",
                passed=False,
                severity="warning",
                message="Parameters may contain credentials",
            ))

        # Check for shell metacharacters
        if re.search(r"[`$|;&><]", all_values):
            checks.append(SecurityCheck(
                check_id="T5-shell",
                name="Shell Metacharacter",
                passed=False,
                severity="warning",
                message="Parameters contain shell metacharacters",
            ))

        if not checks:
            checks.append(SecurityCheck(
                check_id="T5",
                name="Security Pattern",
                passed=True,
                severity="info",
                message="No security pattern matches",
            ))

        return checks

    def _check_external_interfaces(
        self,
        parameters: Dict[str, Any],
        schema: ToolSchema,
    ) -> List[SecurityCheck]:
        """T6: Check for external interface usage."""
        checks = []

        # Network tools always get flagged
        if schema.category == ToolCategory.NETWORK:
            checks.append(SecurityCheck(
                check_id="T6-network",
                name="Network Tool",
                passed=True,
                severity="warning",
                message="Tool involves network access",
            ))

        # Check parameters for external addresses
        all_values = " ".join(str(v) for v in parameters.values() if v is not None)

        for pattern, pattern_type in self.EXTERNAL_INTERFACE_PATTERNS:
            if re.search(pattern, all_values, re.I):
                checks.append(SecurityCheck(
                    check_id=f"T6-{pattern_type}",
                    name="External Interface",
                    passed=False,
                    severity="warning",
                    message=f"External interface detected: {pattern_type}",
                    details={"type": pattern_type},
                ))

        if not checks:
            checks.append(SecurityCheck(
                check_id="T6",
                name="External Interface",
                passed=True,
                severity="info",
                message="No external interfaces detected",
            ))

        return checks

    def _check_irreversible_actions(
        self,
        parameters: Dict[str, Any],
    ) -> List[SecurityCheck]:
        """T7: Check for irreversible actions."""
        checks = []

        all_values = " ".join(str(v) for v in parameters.values() if v is not None)

        for pattern, action_type in self.IRREVERSIBLE_PATTERNS:
            if re.search(pattern, all_values, re.I):
                checks.append(SecurityCheck(
                    check_id=f"T7-{action_type}",
                    name="Irreversible Action",
                    passed=False,
                    severity="error",
                    message=f"Irreversible action detected: {action_type}",
                    details={"action": action_type},
                ))

        if not checks:
            checks.append(SecurityCheck(
                check_id="T7",
                name="Irreversible Action",
                passed=True,
                severity="info",
                message="No irreversible actions detected",
            ))

        return checks

    def _check_rate_limit(self, user_id: str) -> SecurityCheck:
        """T8: Check rate limiting."""
        now = datetime.now()
        cutoff = now.replace(second=0, microsecond=0)

        # Get invocations in last minute
        timestamps = self._invocation_counts.get(user_id, [])
        recent = [ts for ts in timestamps if ts >= cutoff]

        if len(recent) >= self.rate_limit_per_minute:
            return SecurityCheck(
                check_id="T8",
                name="Rate Limit",
                passed=False,
                severity="critical",
                message=f"Rate limit exceeded: {len(recent)}/{self.rate_limit_per_minute} per minute",
            )

        return SecurityCheck(
            check_id="T8",
            name="Rate Limit",
            passed=True,
            severity="info",
            message=f"Rate limit OK: {len(recent)}/{self.rate_limit_per_minute} per minute",
        )

    def _record_invocation(self, user_id: str) -> None:
        """Record an invocation for rate limiting."""
        if user_id not in self._invocation_counts:
            self._invocation_counts[user_id] = []

        self._invocation_counts[user_id].append(datetime.now())

        # Clean old entries (keep last 5 minutes)
        cutoff = datetime.now().replace(second=0, microsecond=0)
        self._invocation_counts[user_id] = [
            ts for ts in self._invocation_counts[user_id]
            if ts >= cutoff
        ]


def create_tool_validator(
    max_risk_auto_approve: ToolRiskLevel = ToolRiskLevel.LOW,
    require_human_above: ToolRiskLevel = ToolRiskLevel.HIGH,
    blocked_categories: Optional[Set[ToolCategory]] = None,
    rate_limit_per_minute: int = 60,
) -> ToolApprovalValidator:
    """
    Create a tool approval validator.

    Args:
        max_risk_auto_approve: Max risk level for auto-approval
        require_human_above: Require human approval above this level
        blocked_categories: Categories that are always blocked
        rate_limit_per_minute: Max invocations per minute

    Returns:
        ToolApprovalValidator instance
    """
    return ToolApprovalValidator(
        max_risk_auto_approve=max_risk_auto_approve,
        require_human_above=require_human_above,
        blocked_categories=blocked_categories,
        rate_limit_per_minute=rate_limit_per_minute,
    )
