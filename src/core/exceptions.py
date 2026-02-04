"""
Agent OS Core Exceptions

Custom exceptions for constitutional kernel and core governance operations.
"""

from pathlib import Path
from typing import List, Optional


class ConstitutionalError(Exception):
    """Base exception for constitutional operations."""

    pass


class ConstitutionLoadError(ConstitutionalError):
    """Failed to load a constitution file."""

    def __init__(
        self,
        message: str,
        path: Optional[Path] = None,
        original_error: Optional[Exception] = None,
    ):
        self.path = path
        self.original_error = original_error
        super().__init__(message)


class SupremeConstitutionError(ConstitutionLoadError):
    """
    Critical error: Failed to load the supreme constitution.

    This is a fatal error that prevents the kernel from operating safely.
    """

    pass


class ConstitutionParseError(ConstitutionalError):
    """Failed to parse constitution document content."""

    def __init__(
        self,
        message: str,
        path: Optional[Path] = None,
        line_number: Optional[int] = None,
    ):
        self.path = path
        self.line_number = line_number
        super().__init__(message)


class ConstitutionValidationError(ConstitutionalError):
    """Constitution document failed validation."""

    def __init__(
        self,
        message: str,
        errors: Optional[List[str]] = None,
        warnings: Optional[List[str]] = None,
    ):
        self.errors = errors or []
        self.warnings = warnings or []
        super().__init__(message)


class RuleViolationError(ConstitutionalError):
    """A request violated one or more constitutional rules."""

    def __init__(
        self,
        message: str,
        violated_rules: Optional[List[str]] = None,
        is_immutable: bool = False,
        requires_escalation: bool = False,
    ):
        self.violated_rules = violated_rules or []
        self.is_immutable = is_immutable
        self.requires_escalation = requires_escalation
        super().__init__(message)


class KernelNotInitializedError(ConstitutionalError):
    """Operation attempted on an uninitialized constitutional kernel."""

    def __init__(self, operation: str = "enforce"):
        super().__init__(f"Cannot {operation}: Constitutional kernel not initialized")


class ReloadCallbackError(ConstitutionalError):
    """A reload callback raised an exception."""

    def __init__(
        self,
        message: str,
        callback_name: Optional[str] = None,
        original_error: Optional[Exception] = None,
    ):
        self.callback_name = callback_name
        self.original_error = original_error
        super().__init__(message)


# =============================================================================
# Security Exceptions
# =============================================================================


class SecurityError(Exception):
    """Base exception for security-related operations."""

    pass


class SSRFProtectionError(SecurityError):
    """
    Server-Side Request Forgery protection triggered.

    Raised when a URL points to localhost, private networks,
    or other restricted destinations.
    """

    def __init__(
        self,
        message: str = "SSRF protection: Request to restricted address blocked",
        url: Optional[str] = None,
    ):
        self.url = url
        super().__init__(message)


class PathValidationError(SecurityError):
    """
    Path validation/traversal protection triggered.

    Raised when a file path contains traversal sequences or
    attempts to access restricted locations.
    """

    def __init__(
        self,
        message: str = "Path validation failed",
        path: Optional[str] = None,
    ):
        self.path = path
        super().__init__(message)


class AuthError(SecurityError):
    """Authentication or authorization error."""

    def __init__(
        self,
        message: str,
        error_code: str = "AUTH_ERROR",
        is_recoverable: bool = True,
    ):
        super().__init__(message)
        self.error_code = error_code
        self.is_recoverable = is_recoverable
