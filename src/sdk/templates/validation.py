"""
Validation Agent Template

Provides a template for agents that validate and filter content.
Includes support for rule-based and pattern-based validation.
"""

import logging
import re
from abc import abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Pattern, Set, Union

from src.agents.interface import CapabilityType
from src.messaging.models import FlowRequest, FlowResponse, MessageStatus

from .base import AgentTemplate, AgentConfig


logger = logging.getLogger(__name__)


class ValidationSeverity(Enum):
    """Severity level of validation issues."""
    INFO = auto()
    WARNING = auto()
    ERROR = auto()
    CRITICAL = auto()


@dataclass
class ValidationIssue:
    """A validation issue found."""
    code: str
    message: str
    severity: ValidationSeverity
    location: Optional[str] = None
    suggestion: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "severity": self.severity.name,
            "location": self.location,
            "suggestion": self.suggestion,
            "metadata": self.metadata,
        }


@dataclass
class ValidationResult:
    """Result of a validation process."""
    is_valid: bool
    issues: List[ValidationIssue] = field(default_factory=list)
    sanitized_content: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def errors(self) -> List[ValidationIssue]:
        return [i for i in self.issues if i.severity in (
            ValidationSeverity.ERROR, ValidationSeverity.CRITICAL
        )]

    @property
    def warnings(self) -> List[ValidationIssue]:
        return [i for i in self.issues if i.severity == ValidationSeverity.WARNING]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "issues": [i.to_dict() for i in self.issues],
            "sanitized_content": self.sanitized_content,
            "metadata": self.metadata,
        }


@dataclass
class ValidationRule:
    """A validation rule."""
    code: str
    description: str
    pattern: Optional[Union[str, Pattern]] = None
    checker: Optional[Callable[[str], bool]] = None
    severity: ValidationSeverity = ValidationSeverity.ERROR
    suggestion: Optional[str] = None
    enabled: bool = True

    def check(self, content: str) -> Optional[ValidationIssue]:
        """
        Check content against this rule.

        Returns ValidationIssue if rule is violated, None otherwise.
        """
        if not self.enabled:
            return None

        violated = False

        if self.pattern:
            pattern = self.pattern if isinstance(self.pattern, Pattern) else re.compile(self.pattern)
            if pattern.search(content):
                violated = True

        if self.checker:
            try:
                if self.checker(content):
                    violated = True
            except Exception as e:
                logger.warning(f"Rule checker error for {self.code}: {e}")

        if violated:
            return ValidationIssue(
                code=self.code,
                message=self.description,
                severity=self.severity,
                suggestion=self.suggestion,
            )

        return None


@dataclass
class ValidationConfig(AgentConfig):
    """Configuration for validation agents."""
    fail_on_warning: bool = False
    fail_on_error: bool = True
    sanitize_content: bool = True
    max_issues: int = 100
    block_on_critical: bool = True

    def __post_init__(self):
        self.capabilities = self.capabilities or set()
        self.capabilities.add(CapabilityType.VALIDATION)


class ValidationAgentTemplate(AgentTemplate):
    """
    Template for validation agents.

    Provides:
    - Rule-based validation
    - Pattern matching
    - Content sanitization
    - Severity levels
    - Issue reporting
    """

    def __init__(self, config: ValidationConfig):
        """
        Initialize validation agent.

        Args:
            config: Validation agent configuration
        """
        super().__init__(config)
        self.validation_config = config
        self._rules: List[ValidationRule] = []
        self._sanitizers: List[Callable[[str], str]] = []

    def add_rule(self, rule: ValidationRule) -> "ValidationAgentTemplate":
        """
        Add a validation rule.

        Args:
            rule: Validation rule

        Returns:
            Self for chaining
        """
        self._rules.append(rule)
        return self

    def add_pattern_rule(
        self,
        code: str,
        pattern: str,
        description: str,
        severity: ValidationSeverity = ValidationSeverity.ERROR,
        suggestion: Optional[str] = None,
    ) -> "ValidationAgentTemplate":
        """
        Add a pattern-based validation rule.

        Args:
            code: Rule code
            pattern: Regex pattern (matches = violation)
            description: Description of the issue
            severity: Issue severity
            suggestion: Suggestion for fixing

        Returns:
            Self for chaining
        """
        rule = ValidationRule(
            code=code,
            description=description,
            pattern=pattern,
            severity=severity,
            suggestion=suggestion,
        )
        return self.add_rule(rule)

    def add_checker_rule(
        self,
        code: str,
        checker: Callable[[str], bool],
        description: str,
        severity: ValidationSeverity = ValidationSeverity.ERROR,
        suggestion: Optional[str] = None,
    ) -> "ValidationAgentTemplate":
        """
        Add a checker-based validation rule.

        Args:
            code: Rule code
            checker: Checker function (returns True = violation)
            description: Description of the issue
            severity: Issue severity
            suggestion: Suggestion for fixing

        Returns:
            Self for chaining
        """
        rule = ValidationRule(
            code=code,
            description=description,
            checker=checker,
            severity=severity,
            suggestion=suggestion,
        )
        return self.add_rule(rule)

    def add_sanitizer(
        self,
        sanitizer: Callable[[str], str],
    ) -> "ValidationAgentTemplate":
        """
        Add a content sanitizer.

        Args:
            sanitizer: Sanitizer function

        Returns:
            Self for chaining
        """
        self._sanitizers.append(sanitizer)
        return self

    def handle_request(self, request: FlowRequest) -> FlowResponse:
        """Process validation request."""
        try:
            content = request.content.prompt

            # Validate
            result = self.validate(content, request)

            # Check for blocking issues
            if result.errors and self.validation_config.block_on_critical:
                critical = [i for i in result.issues if i.severity == ValidationSeverity.CRITICAL]
                if critical:
                    return self._create_blocked_response(request, critical)

            # Determine overall status
            if not result.is_valid:
                status = MessageStatus.REFUSED
            elif result.warnings:
                status = MessageStatus.PARTIAL
            else:
                status = MessageStatus.SUCCESS

            # Format output
            output = self._format_result(result)

            return request.create_response(
                source=self.name,
                status=status,
                output=output,
            )

        except Exception as e:
            logger.error(f"Validation error: {e}")
            return request.create_response(
                source=self.name,
                status=MessageStatus.ERROR,
                output="",
                errors=[f"Validation failed: {str(e)}"],
            )

    def validate(self, content: str, request: FlowRequest) -> ValidationResult:
        """
        Validate content.

        Override to add custom validation logic.

        Args:
            content: Content to validate
            request: Original request for context

        Returns:
            ValidationResult
        """
        issues: List[ValidationIssue] = []

        # Run rules
        for rule in self._rules:
            issue = rule.check(content)
            if issue:
                issues.append(issue)
                if len(issues) >= self.validation_config.max_issues:
                    break

        # Run custom validation
        custom_issues = self.validate_content(content, request)
        issues.extend(custom_issues[:self.validation_config.max_issues - len(issues)])

        # Determine validity
        has_errors = any(
            i.severity in (ValidationSeverity.ERROR, ValidationSeverity.CRITICAL)
            for i in issues
        )
        has_warnings = any(i.severity == ValidationSeverity.WARNING for i in issues)

        is_valid = not has_errors
        if self.validation_config.fail_on_warning and has_warnings:
            is_valid = False

        # Sanitize if configured
        sanitized = None
        if self.validation_config.sanitize_content and is_valid:
            sanitized = self._sanitize(content)

        return ValidationResult(
            is_valid=is_valid,
            issues=issues,
            sanitized_content=sanitized,
        )

    def validate_content(
        self,
        content: str,
        request: FlowRequest,
    ) -> List[ValidationIssue]:
        """
        Perform custom validation.

        Override to add custom validation logic.

        Args:
            content: Content to validate
            request: Original request

        Returns:
            List of validation issues
        """
        return []

    def _sanitize(self, content: str) -> str:
        """Apply sanitizers to content."""
        for sanitizer in self._sanitizers:
            try:
                content = sanitizer(content)
            except Exception as e:
                logger.warning(f"Sanitizer error: {e}")
        return content

    def _format_result(self, result: ValidationResult) -> str:
        """Format validation result for output."""
        if result.is_valid and not result.issues:
            return "Validation passed."

        lines = []
        if result.is_valid:
            lines.append("Validation passed with warnings:")
        else:
            lines.append("Validation failed:")

        for issue in result.issues:
            prefix = {
                ValidationSeverity.INFO: "ℹ",
                ValidationSeverity.WARNING: "⚠",
                ValidationSeverity.ERROR: "✗",
                ValidationSeverity.CRITICAL: "🚫",
            }.get(issue.severity, "•")
            lines.append(f"  {prefix} [{issue.code}] {issue.message}")
            if issue.suggestion:
                lines.append(f"      Suggestion: {issue.suggestion}")

        return "\n".join(lines)

    def _create_blocked_response(
        self,
        request: FlowRequest,
        issues: List[ValidationIssue],
    ) -> FlowResponse:
        """Create a blocked response for critical issues."""
        messages = [f"[{i.code}] {i.message}" for i in issues]
        return request.create_response(
            source=self.name,
            status=MessageStatus.REFUSED,
            output="Request blocked due to critical validation issues.",
            errors=messages,
        )


def create_validation_agent(
    name: str,
    rules: Optional[List[ValidationRule]] = None,
    validate_fn: Optional[Callable[[str, FlowRequest], List[ValidationIssue]]] = None,
    description: str = "",
    **kwargs,
) -> ValidationAgentTemplate:
    """
    Create a validation agent.

    Example:
        agent = create_validation_agent(
            name="content_filter",
            description="Filters inappropriate content",
            rules=[
                ValidationRule(
                    code="PROFANITY",
                    description="Content contains profanity",
                    pattern=r"\\b(bad|words)\\b",
                    severity=ValidationSeverity.ERROR,
                ),
            ],
        )

    Args:
        name: Agent name
        rules: Validation rules
        validate_fn: Custom validation function
        description: Agent description
        **kwargs: Additional config options

    Returns:
        ValidationAgentTemplate instance
    """
    config = ValidationConfig(
        name=name,
        description=description,
        **kwargs,
    )

    class ConfiguredValidationAgent(ValidationAgentTemplate):
        def __init__(self):
            super().__init__(config)
            if rules:
                for rule in rules:
                    self.add_rule(rule)

        def validate_content(
            self,
            content: str,
            request: FlowRequest,
        ) -> List[ValidationIssue]:
            if validate_fn:
                return validate_fn(content, request)
            return []

    return ConfiguredValidationAgent()
