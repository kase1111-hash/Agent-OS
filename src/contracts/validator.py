"""
Learning Contract Validator

Validates learning contracts and learning requests against contracts.
Ensures compliance with contract terms and prohibited domains.
"""

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Set

from .store import (
    LearningContract,
    ContractType,
    ContractStatus,
    ContractScope,
    LearningScope,
)


logger = logging.getLogger(__name__)


class ValidationSeverity(Enum):
    """Severity level of validation issues."""
    INFO = auto()
    WARNING = auto()
    ERROR = auto()
    CRITICAL = auto()


class ValidationCode(Enum):
    """Validation result codes."""
    # Success codes
    VALID = "valid"
    VALID_WITH_WARNINGS = "valid_with_warnings"

    # Contract issues
    CONTRACT_EXPIRED = "contract_expired"
    CONTRACT_REVOKED = "contract_revoked"
    CONTRACT_PENDING = "contract_pending"
    CONTRACT_SUSPENDED = "contract_suspended"
    CONTRACT_INVALID_SCOPE = "contract_invalid_scope"
    CONTRACT_MISSING_SIGNATURE = "contract_missing_signature"
    CONTRACT_SIGNATURE_MISMATCH = "contract_signature_mismatch"

    # Domain issues
    PROHIBITED_DOMAIN = "prohibited_domain"
    EXCLUDED_DOMAIN = "excluded_domain"
    DOMAIN_NOT_ALLOWED = "domain_not_allowed"

    # Content issues
    CONTENT_TYPE_NOT_ALLOWED = "content_type_not_allowed"
    RAW_STORAGE_NOT_ALLOWED = "raw_storage_not_allowed"
    ABSTRACTION_REQUIRED = "abstraction_required"

    # Authorization issues
    NO_CONTRACT = "no_contract"
    INSUFFICIENT_SCOPE = "insufficient_scope"
    USER_MISMATCH = "user_mismatch"
    AGENT_NOT_ALLOWED = "agent_not_allowed"


@dataclass
class ValidationIssue:
    """A single validation issue."""
    code: ValidationCode
    severity: ValidationSeverity
    message: str
    source_field: str = ""
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "code": self.code.value,
            "severity": self.severity.name,
            "message": self.message,
            "field": self.source_field,
            "details": self.details,
        }


@dataclass
class ValidationResult:
    """Result of a validation operation."""
    is_valid: bool
    code: ValidationCode
    issues: List[ValidationIssue] = field(default_factory=list)
    contract: Optional[LearningContract] = None
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_issue(
        self,
        code: ValidationCode,
        severity: ValidationSeverity,
        message: str,
        source_field: str = "",
        **details,
    ) -> None:
        """Add a validation issue."""
        self.issues.append(ValidationIssue(
            code=code,
            severity=severity,
            message=message,
            source_field=source_field,
            details=details,
        ))

        # Update validity based on severity
        if severity in [ValidationSeverity.ERROR, ValidationSeverity.CRITICAL]:
            self.is_valid = False

    @property
    def has_errors(self) -> bool:
        """Check if there are any errors."""
        return any(
            i.severity in [ValidationSeverity.ERROR, ValidationSeverity.CRITICAL]
            for i in self.issues
        )

    @property
    def has_warnings(self) -> bool:
        """Check if there are any warnings."""
        return any(i.severity == ValidationSeverity.WARNING for i in self.issues)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "is_valid": self.is_valid,
            "code": self.code.value,
            "issues": [i.to_dict() for i in self.issues],
            "contract_id": self.contract.contract_id if self.contract else None,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }


@dataclass
class LearningRequest:
    """A request to learn/store data."""
    request_id: str
    user_id: str
    domain: str = ""
    task: str = ""
    agent: str = ""
    content_type: str = ""
    content: str = ""
    requires_raw_storage: bool = False
    abstraction_allowed: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


class ContractValidator:
    """
    Validates learning contracts and learning requests.

    Performs comprehensive validation including:
    - Contract status and expiry
    - Scope matching
    - Prohibited domains
    - Content type restrictions
    - Abstraction requirements
    """

    def __init__(
        self,
        prohibited_domains: Optional[Set[str]] = None,
        prohibited_patterns: Optional[List[str]] = None,
        require_signature: bool = True,
    ):
        """
        Initialize validator.

        Args:
            prohibited_domains: Set of prohibited domains
            prohibited_patterns: Regex patterns for prohibited content
            require_signature: Require contract signatures
        """
        self.prohibited_domains = prohibited_domains or set()
        self.prohibited_patterns = [
            re.compile(p) for p in (prohibited_patterns or [])
        ]
        self.require_signature = require_signature

    def validate_contract(self, contract: LearningContract) -> ValidationResult:
        """
        Validate a contract itself.

        Args:
            contract: Contract to validate

        Returns:
            ValidationResult
        """
        result = ValidationResult(
            is_valid=True,
            code=ValidationCode.VALID,
            contract=contract,
        )

        # Check status
        self._validate_contract_status(contract, result)

        # Check expiry
        self._validate_contract_expiry(contract, result)

        # Check scope
        self._validate_contract_scope(contract, result)

        # Check signature
        if self.require_signature:
            self._validate_contract_signature(contract, result)

        # Check prohibited domains in scope
        self._validate_scope_domains(contract, result)

        # Update code based on issues
        if not result.is_valid:
            result.code = result.issues[0].code if result.issues else ValidationCode.CONTRACT_INVALID_SCOPE
        elif result.has_warnings:
            result.code = ValidationCode.VALID_WITH_WARNINGS

        return result

    def validate_request(
        self,
        request: LearningRequest,
        contract: Optional[LearningContract],
    ) -> ValidationResult:
        """
        Validate a learning request against a contract.

        Args:
            request: Learning request to validate
            contract: Contract to validate against (None if no contract)

        Returns:
            ValidationResult
        """
        result = ValidationResult(
            is_valid=True,
            code=ValidationCode.VALID,
            contract=contract,
        )

        # Check for contract
        if not contract:
            result.add_issue(
                ValidationCode.NO_CONTRACT,
                ValidationSeverity.ERROR,
                "No valid learning contract found",
            )
            result.code = ValidationCode.NO_CONTRACT
            return result

        # Validate the contract first
        contract_result = self.validate_contract(contract)
        if not contract_result.is_valid:
            result.issues.extend(contract_result.issues)
            result.is_valid = False
            result.code = contract_result.code
            return result

        # Check user match
        if request.user_id != contract.user_id:
            result.add_issue(
                ValidationCode.USER_MISMATCH,
                ValidationSeverity.ERROR,
                "Request user does not match contract user",
                user_id=request.user_id,
                contract_user=contract.user_id,
            )

        # Check prohibited domains
        if request.domain:
            self._validate_request_domain(request, result)

        # Check scope match
        if not contract.scope.matches(
            domain=request.domain,
            task=request.task,
            agent=request.agent,
            content_type=request.content_type,
        ):
            result.add_issue(
                ValidationCode.INSUFFICIENT_SCOPE,
                ValidationSeverity.ERROR,
                "Request not covered by contract scope",
                domain=request.domain,
                task=request.task,
                agent=request.agent,
            )

        # Check agent restrictions
        if request.agent and contract.scope.agents:
            if request.agent not in contract.scope.agents:
                result.add_issue(
                    ValidationCode.AGENT_NOT_ALLOWED,
                    ValidationSeverity.ERROR,
                    f"Agent '{request.agent}' not allowed by contract",
                    agent=request.agent,
                )

        # Check raw storage
        if request.requires_raw_storage:
            if not contract.allows_raw_storage():
                result.add_issue(
                    ValidationCode.RAW_STORAGE_NOT_ALLOWED,
                    ValidationSeverity.ERROR,
                    "Raw data storage not allowed by contract",
                    contract_type=contract.contract_type.name,
                )

        # Check abstraction requirements
        if contract.contract_type == ContractType.ABSTRACTION_ONLY:
            if request.requires_raw_storage:
                result.add_issue(
                    ValidationCode.ABSTRACTION_REQUIRED,
                    ValidationSeverity.ERROR,
                    "Contract only allows abstracted learning, not raw storage",
                )

        # Check no-learning contracts
        if contract.contract_type == ContractType.NO_LEARNING:
            result.add_issue(
                ValidationCode.DOMAIN_NOT_ALLOWED,
                ValidationSeverity.ERROR,
                "Learning explicitly denied by contract",
            )

        # Update code
        if not result.is_valid:
            result.code = result.issues[0].code if result.issues else ValidationCode.INSUFFICIENT_SCOPE
        elif result.has_warnings:
            result.code = ValidationCode.VALID_WITH_WARNINGS

        return result

    def is_domain_prohibited(self, domain: str) -> bool:
        """Check if a domain is prohibited."""
        if domain in self.prohibited_domains:
            return True

        domain_lower = domain.lower()
        for prohibited in self.prohibited_domains:
            if prohibited.lower() in domain_lower:
                return True

        return False

    def is_content_prohibited(self, content: str) -> bool:
        """Check if content matches prohibited patterns."""
        for pattern in self.prohibited_patterns:
            if pattern.search(content):
                return True
        return False

    def add_prohibited_domain(self, domain: str) -> None:
        """Add a prohibited domain."""
        self.prohibited_domains.add(domain)

    def remove_prohibited_domain(self, domain: str) -> bool:
        """Remove a prohibited domain."""
        if domain in self.prohibited_domains:
            self.prohibited_domains.remove(domain)
            return True
        return False

    def add_prohibited_pattern(self, pattern: str) -> None:
        """Add a prohibited content pattern."""
        self.prohibited_patterns.append(re.compile(pattern))

    def _validate_contract_status(
        self,
        contract: LearningContract,
        result: ValidationResult,
    ) -> None:
        """Validate contract status."""
        if contract.status == ContractStatus.PENDING:
            result.add_issue(
                ValidationCode.CONTRACT_PENDING,
                ValidationSeverity.ERROR,
                "Contract has not been activated",
            )
        elif contract.status == ContractStatus.REVOKED:
            result.add_issue(
                ValidationCode.CONTRACT_REVOKED,
                ValidationSeverity.ERROR,
                f"Contract was revoked: {contract.revocation_reason or 'No reason given'}",
            )
        elif contract.status == ContractStatus.EXPIRED:
            result.add_issue(
                ValidationCode.CONTRACT_EXPIRED,
                ValidationSeverity.ERROR,
                "Contract has expired",
            )
        elif contract.status == ContractStatus.SUSPENDED:
            result.add_issue(
                ValidationCode.CONTRACT_SUSPENDED,
                ValidationSeverity.ERROR,
                "Contract is currently suspended",
            )
        elif contract.status == ContractStatus.SUPERSEDED:
            result.add_issue(
                ValidationCode.CONTRACT_EXPIRED,
                ValidationSeverity.WARNING,
                "Contract has been superseded by a newer version",
            )

    def _validate_contract_expiry(
        self,
        contract: LearningContract,
        result: ValidationResult,
    ) -> None:
        """Validate contract expiry."""
        if contract.expires_at:
            if datetime.now() > contract.expires_at:
                result.add_issue(
                    ValidationCode.CONTRACT_EXPIRED,
                    ValidationSeverity.ERROR,
                    "Contract has expired",
                    expires_at=contract.expires_at.isoformat(),
                )
            elif (contract.expires_at - datetime.now()).days < 7:
                result.add_issue(
                    ValidationCode.CONTRACT_EXPIRED,
                    ValidationSeverity.INFO,
                    "Contract expires soon",
                    expires_at=contract.expires_at.isoformat(),
                )

    def _validate_contract_scope(
        self,
        contract: LearningContract,
        result: ValidationResult,
    ) -> None:
        """Validate contract scope."""
        scope = contract.scope

        # Check for empty scope where specificity is expected
        if scope.scope_type == LearningScope.DOMAIN_SPECIFIC and not scope.domains:
            result.add_issue(
                ValidationCode.CONTRACT_INVALID_SCOPE,
                ValidationSeverity.WARNING,
                "Domain-specific scope has no domains defined",
            )

        if scope.scope_type == LearningScope.AGENT_SPECIFIC and not scope.agents:
            result.add_issue(
                ValidationCode.CONTRACT_INVALID_SCOPE,
                ValidationSeverity.WARNING,
                "Agent-specific scope has no agents defined",
            )

    def _validate_contract_signature(
        self,
        contract: LearningContract,
        result: ValidationResult,
    ) -> None:
        """Validate contract signature."""
        if not contract.signature_hash:
            result.add_issue(
                ValidationCode.CONTRACT_MISSING_SIGNATURE,
                ValidationSeverity.WARNING,
                "Contract is missing signature hash",
            )
        else:
            expected = contract.compute_signature()
            if contract.signature_hash != expected:
                result.add_issue(
                    ValidationCode.CONTRACT_SIGNATURE_MISMATCH,
                    ValidationSeverity.ERROR,
                    "Contract signature does not match content",
                )

    def _validate_scope_domains(
        self,
        contract: LearningContract,
        result: ValidationResult,
    ) -> None:
        """Validate that scope doesn't include prohibited domains."""
        for domain in contract.scope.domains:
            if self.is_domain_prohibited(domain):
                result.add_issue(
                    ValidationCode.PROHIBITED_DOMAIN,
                    ValidationSeverity.ERROR,
                    f"Contract includes prohibited domain: {domain}",
                    domain=domain,
                )

    def _validate_request_domain(
        self,
        request: LearningRequest,
        result: ValidationResult,
    ) -> None:
        """Validate request domain is not prohibited."""
        if self.is_domain_prohibited(request.domain):
            result.add_issue(
                ValidationCode.PROHIBITED_DOMAIN,
                ValidationSeverity.ERROR,
                f"Domain is prohibited: {request.domain}",
                domain=request.domain,
            )

        # Check content for prohibited patterns
        if request.content and self.is_content_prohibited(request.content):
            result.add_issue(
                ValidationCode.PROHIBITED_DOMAIN,
                ValidationSeverity.ERROR,
                "Content contains prohibited patterns",
            )


def create_validator(
    prohibited_domains: Optional[Set[str]] = None,
    require_signature: bool = True,
) -> ContractValidator:
    """
    Factory function to create a contract validator.

    Args:
        prohibited_domains: Set of prohibited domains
        require_signature: Require contract signatures

    Returns:
        Configured ContractValidator
    """
    # Default prohibited domains
    default_prohibited = {
        "medical_records",
        "financial_data",
        "credentials",
        "passwords",
        "private_keys",
        "biometric",
        "social_security",
        "credit_card",
        "bank_account",
    }

    domains = prohibited_domains or default_prohibited

    return ContractValidator(
        prohibited_domains=domains,
        require_signature=require_signature,
    )
