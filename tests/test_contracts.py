"""
Tests for the Learning Contracts Module (UC-013).

Tests cover:
- Contract store (create, activate, revoke, expire)
- Contract validator
- Prohibited domain checker
- Abstraction guard
- Enforcement engine
- Contracts client
"""

import pytest
from datetime import datetime, timedelta
from pathlib import Path
import tempfile

from src.contracts import (
    # Store
    ContractStore,
    LearningContract,
    ContractType,
    ContractStatus,
    ContractScope,
    LearningScope,
    ContractQuery,
    create_contract_store,
    # Validator
    ContractValidator,
    ValidationResult,
    ValidationCode,
    LearningRequest,
    create_validator,
    # Domains
    ProhibitedDomainChecker,
    ProhibitedDomain,
    DomainCheckResult,
    DomainCategory,
    ProhibitionLevel,
    create_domain_checker,
    # Abstraction
    AbstractionGuard,
    AbstractionRule,
    AbstractionResult,
    AbstractionLevel,
    AbstractionType,
    create_abstraction_guard,
    # Enforcement
    LearningContractsEngine,
    EnforcementResult,
    EnforcementDecision,
    EnforcementConfig,
    create_learning_contracts_engine,
    # Consent
    ConsentPrompt,
    ConsentRequest,
    ConsentDecision,
    ConsentResponse,
    ConsentMode,
    create_consent_prompt,
    # Client
    ContractsClient,
    ContractsClientConfig,
    create_contracts_client,
)


# =============================================================================
# Contract Store Tests
# =============================================================================

class TestContractStore:
    """Tests for ContractStore."""

    def test_create_store(self):
        """Test creating a contract store."""
        store = create_contract_store()
        assert store is not None
        store.close()

    def test_create_store_with_db(self):
        """Test creating a store with database file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "contracts.db"
            store = create_contract_store(db_path=db_path)
            assert store is not None
            store.close()

    def test_create_contract(self):
        """Test creating a learning contract."""
        store = create_contract_store()

        scope = ContractScope(scope_type=LearningScope.ALL)
        contract = store.create_contract(
            user_id="user123",
            contract_type=ContractType.FULL_CONSENT,
            scope=scope,
            description="Test contract",
        )

        assert contract is not None
        assert contract.user_id == "user123"
        assert contract.contract_type == ContractType.FULL_CONSENT
        assert contract.status == ContractStatus.PENDING
        store.close()

    def test_activate_contract(self):
        """Test activating a contract."""
        store = create_contract_store()

        scope = ContractScope(scope_type=LearningScope.ALL)
        contract = store.create_contract(
            user_id="user123",
            contract_type=ContractType.FULL_CONSENT,
            scope=scope,
        )

        assert contract.status == ContractStatus.PENDING

        success = store.activate_contract(contract.contract_id, "user123")
        assert success

        updated = store.get_contract(contract.contract_id)
        assert updated.status == ContractStatus.ACTIVE
        store.close()

    def test_auto_activate_contract(self):
        """Test auto-activating a contract on creation."""
        store = create_contract_store()

        scope = ContractScope(scope_type=LearningScope.ALL)
        contract = store.create_contract(
            user_id="user123",
            contract_type=ContractType.FULL_CONSENT,
            scope=scope,
            auto_activate=True,
        )

        assert contract.status == ContractStatus.ACTIVE
        store.close()

    def test_revoke_contract(self):
        """Test revoking a contract."""
        store = create_contract_store()

        scope = ContractScope(scope_type=LearningScope.ALL)
        contract = store.create_contract(
            user_id="user123",
            contract_type=ContractType.FULL_CONSENT,
            scope=scope,
            auto_activate=True,
        )

        success = store.revoke_contract(
            contract.contract_id,
            "user123",
            "User requested revocation",
        )
        assert success

        updated = store.get_contract(contract.contract_id)
        assert updated.status == ContractStatus.REVOKED
        assert updated.revocation_reason == "User requested revocation"
        store.close()

    def test_expire_contract(self):
        """Test expiring a contract."""
        store = create_contract_store()

        scope = ContractScope(scope_type=LearningScope.ALL)
        contract = store.create_contract(
            user_id="user123",
            contract_type=ContractType.FULL_CONSENT,
            scope=scope,
            auto_activate=True,
        )

        success = store.expire_contract(contract.contract_id)
        assert success

        updated = store.get_contract(contract.contract_id)
        assert updated.status == ContractStatus.EXPIRED
        store.close()

    def test_suspend_resume_contract(self):
        """Test suspending and resuming a contract."""
        store = create_contract_store()

        scope = ContractScope(scope_type=LearningScope.ALL)
        contract = store.create_contract(
            user_id="user123",
            contract_type=ContractType.FULL_CONSENT,
            scope=scope,
            auto_activate=True,
        )

        # Suspend
        success = store.suspend_contract(contract.contract_id, "admin", "Temporary suspension")
        assert success

        updated = store.get_contract(contract.contract_id)
        assert updated.status == ContractStatus.SUSPENDED

        # Resume
        success = store.resume_contract(contract.contract_id, "admin")
        assert success

        updated = store.get_contract(contract.contract_id)
        assert updated.status == ContractStatus.ACTIVE
        store.close()

    def test_get_active_contracts(self):
        """Test getting active contracts for a user."""
        store = create_contract_store()

        scope = ContractScope(scope_type=LearningScope.ALL)

        # Create active contract
        contract1 = store.create_contract(
            user_id="user123",
            contract_type=ContractType.FULL_CONSENT,
            scope=scope,
            auto_activate=True,
        )

        # Create pending contract
        contract2 = store.create_contract(
            user_id="user123",
            contract_type=ContractType.LIMITED_CONSENT,
            scope=scope,
        )

        active = store.get_active_contracts("user123")
        assert len(active) == 1
        assert active[0].contract_id == contract1.contract_id
        store.close()

    def test_check_learning_allowed(self):
        """Test checking if learning is allowed."""
        store = create_contract_store()

        scope = ContractScope(
            scope_type=LearningScope.DOMAIN_SPECIFIC,
            domains={"general", "research"},
        )

        store.create_contract(
            user_id="user123",
            contract_type=ContractType.FULL_CONSENT,
            scope=scope,
            auto_activate=True,
        )

        # Allowed domain
        allowed, contract = store.check_learning_allowed("user123", domain="general")
        assert allowed
        assert contract is not None

        # Not in scope
        allowed, contract = store.check_learning_allowed("user123", domain="medical")
        assert not allowed
        store.close()

    def test_default_deny(self):
        """Test default deny when no contract exists."""
        store = create_contract_store(default_deny=True)

        allowed, contract = store.check_learning_allowed("user123")
        assert not allowed
        assert contract is None
        store.close()

    def test_contract_with_duration(self):
        """Test creating contract with duration."""
        store = create_contract_store()

        scope = ContractScope(scope_type=LearningScope.ALL)
        contract = store.create_contract(
            user_id="user123",
            contract_type=ContractType.TEMPORARY,
            scope=scope,
            duration=timedelta(hours=1),
            auto_activate=True,
        )

        assert contract.expires_at is not None
        assert contract.expires_at > datetime.now()
        store.close()

    def test_supersede_contract(self):
        """Test superseding a contract with a new one."""
        store = create_contract_store()

        scope = ContractScope(scope_type=LearningScope.ALL)
        old_contract = store.create_contract(
            user_id="user123",
            contract_type=ContractType.LIMITED_CONSENT,
            scope=scope,
            auto_activate=True,
        )

        new_scope = ContractScope(scope_type=LearningScope.ALL)
        new_contract = LearningContract(
            contract_id="LC-new",
            user_id="user123",
            contract_type=ContractType.FULL_CONSENT,
            scope=new_scope,
        )
        new_contract.signature_hash = new_contract.compute_signature()

        success = store.supersede_contract(old_contract.contract_id, new_contract, "user123")
        assert success

        old_updated = store.get_contract(old_contract.contract_id)
        assert old_updated.status == ContractStatus.SUPERSEDED

        new_retrieved = store.get_contract(new_contract.contract_id)
        assert new_retrieved.previous_contract_id == old_contract.contract_id
        assert new_retrieved.version == 2
        store.close()

    def test_contract_history(self):
        """Test getting contract version history."""
        store = create_contract_store()

        scope = ContractScope(scope_type=LearningScope.ALL)
        contract1 = store.create_contract(
            user_id="user123",
            contract_type=ContractType.LIMITED_CONSENT,
            scope=scope,
            auto_activate=True,
        )

        contract2 = LearningContract(
            contract_id="LC-v2",
            user_id="user123",
            contract_type=ContractType.FULL_CONSENT,
            scope=scope,
        )
        contract2.signature_hash = contract2.compute_signature()
        store.supersede_contract(contract1.contract_id, contract2, "user123")

        history = store.get_contract_history(contract2.contract_id)
        assert len(history) == 2
        assert history[0].contract_id == contract1.contract_id
        assert history[1].contract_id == contract2.contract_id
        store.close()


# =============================================================================
# Contract Validator Tests
# =============================================================================

class TestContractValidator:
    """Tests for ContractValidator."""

    def test_create_validator(self):
        """Test creating a validator."""
        validator = create_validator()
        assert validator is not None

    def test_validate_active_contract(self):
        """Test validating an active contract."""
        validator = create_validator(require_signature=False)

        scope = ContractScope(scope_type=LearningScope.ALL)
        contract = LearningContract(
            contract_id="LC-001",
            user_id="user123",
            contract_type=ContractType.FULL_CONSENT,
            scope=scope,
            status=ContractStatus.ACTIVE,
            activated_at=datetime.now(),
        )

        result = validator.validate_contract(contract)
        assert result.is_valid
        assert result.code == ValidationCode.VALID

    def test_validate_expired_contract(self):
        """Test validating an expired contract."""
        validator = create_validator(require_signature=False)

        scope = ContractScope(scope_type=LearningScope.ALL)
        contract = LearningContract(
            contract_id="LC-001",
            user_id="user123",
            contract_type=ContractType.FULL_CONSENT,
            scope=scope,
            status=ContractStatus.ACTIVE,
            expires_at=datetime.now() - timedelta(hours=1),
        )

        result = validator.validate_contract(contract)
        assert not result.is_valid
        assert result.code == ValidationCode.CONTRACT_EXPIRED

    def test_validate_revoked_contract(self):
        """Test validating a revoked contract."""
        validator = create_validator(require_signature=False)

        scope = ContractScope(scope_type=LearningScope.ALL)
        contract = LearningContract(
            contract_id="LC-001",
            user_id="user123",
            contract_type=ContractType.FULL_CONSENT,
            scope=scope,
            status=ContractStatus.REVOKED,
            revocation_reason="User requested",
        )

        result = validator.validate_contract(contract)
        assert not result.is_valid
        assert result.code == ValidationCode.CONTRACT_REVOKED

    def test_validate_request_with_valid_contract(self):
        """Test validating a request against a valid contract."""
        validator = create_validator(require_signature=False)

        scope = ContractScope(
            scope_type=LearningScope.DOMAIN_SPECIFIC,
            domains={"general"},
        )
        contract = LearningContract(
            contract_id="LC-001",
            user_id="user123",
            contract_type=ContractType.FULL_CONSENT,
            scope=scope,
            status=ContractStatus.ACTIVE,
        )

        request = LearningRequest(
            request_id="LR-001",
            user_id="user123",
            domain="general",
        )

        result = validator.validate_request(request, contract)
        assert result.is_valid

    def test_validate_request_outside_scope(self):
        """Test validating a request outside contract scope."""
        validator = create_validator(require_signature=False)

        scope = ContractScope(
            scope_type=LearningScope.DOMAIN_SPECIFIC,
            domains={"general"},
        )
        contract = LearningContract(
            contract_id="LC-001",
            user_id="user123",
            contract_type=ContractType.FULL_CONSENT,
            scope=scope,
            status=ContractStatus.ACTIVE,
        )

        request = LearningRequest(
            request_id="LR-001",
            user_id="user123",
            domain="medical",  # Not in scope
        )

        result = validator.validate_request(request, contract)
        assert not result.is_valid
        assert result.code == ValidationCode.INSUFFICIENT_SCOPE

    def test_validate_request_no_contract(self):
        """Test validating a request without a contract."""
        validator = create_validator()

        request = LearningRequest(
            request_id="LR-001",
            user_id="user123",
            domain="general",
        )

        result = validator.validate_request(request, None)
        assert not result.is_valid
        assert result.code == ValidationCode.NO_CONTRACT

    def test_check_prohibited_domain(self):
        """Test checking for prohibited domains."""
        validator = create_validator()

        # Add default prohibited domains
        assert validator.is_domain_prohibited("credentials")
        assert validator.is_domain_prohibited("passwords")
        assert not validator.is_domain_prohibited("general")

    def test_signature_validation(self):
        """Test contract signature validation."""
        validator = create_validator(require_signature=True)

        scope = ContractScope(scope_type=LearningScope.ALL)
        contract = LearningContract(
            contract_id="LC-001",
            user_id="user123",
            contract_type=ContractType.FULL_CONSENT,
            scope=scope,
            status=ContractStatus.ACTIVE,
        )
        contract.signature_hash = contract.compute_signature()

        result = validator.validate_contract(contract)
        assert result.is_valid

    def test_signature_mismatch(self):
        """Test detecting signature mismatch."""
        validator = create_validator(require_signature=True)

        scope = ContractScope(scope_type=LearningScope.ALL)
        contract = LearningContract(
            contract_id="LC-001",
            user_id="user123",
            contract_type=ContractType.FULL_CONSENT,
            scope=scope,
            status=ContractStatus.ACTIVE,
            signature_hash="invalid_hash",
        )

        result = validator.validate_contract(contract)
        assert not result.is_valid


# =============================================================================
# Prohibited Domain Checker Tests
# =============================================================================

class TestProhibitedDomainChecker:
    """Tests for ProhibitedDomainChecker."""

    def test_create_checker(self):
        """Test creating a domain checker."""
        checker = create_domain_checker()
        assert checker is not None
        assert len(checker.get_all_domains()) > 0

    def test_check_prohibited_content(self):
        """Test checking content for prohibited domains."""
        checker = create_domain_checker()

        # SSN pattern
        result = checker.check("My SSN is 123-45-6789")
        assert result.is_prohibited
        assert len(result.matching_domains) > 0

        # Clean content
        result = checker.check("Hello world")
        assert not result.is_prohibited

    def test_check_credit_card(self):
        """Test checking for credit card numbers."""
        checker = create_domain_checker()

        result = checker.check("Card: 1234-5678-9012-3456")
        assert result.is_prohibited
        assert any(d.name == "Credit Card Numbers" for d in result.matching_domains)

    def test_check_password_pattern(self):
        """Test checking for password patterns."""
        checker = create_domain_checker()

        result = checker.check("password = 'mysecretpass123'")
        assert result.is_prohibited

    def test_check_domain_name(self):
        """Test checking if a domain name is prohibited."""
        checker = create_domain_checker()

        # Add a custom domain for testing domain name checks
        custom = ProhibitedDomain(
            domain_id="test_domain",
            name="TestDomain",
            category=DomainCategory.CUSTOM,
            level=ProhibitionLevel.DEFAULT,
            keywords={"testdomain"},
        )
        checker.add_domain(custom)

        # Domain names containing the prohibited domain name
        result = checker.check_domain_name("my_testdomain_storage")
        assert result.is_prohibited

        result = checker.check_domain_name("general_topics")
        assert not result.is_prohibited

    def test_prohibition_levels(self):
        """Test different prohibition levels."""
        checker = create_domain_checker()

        # SSN should be ABSOLUTE
        result = checker.check("SSN: 123-45-6789")
        assert result.is_prohibited
        assert result.highest_level == ProhibitionLevel.ABSOLUTE
        assert not result.can_override

    def test_add_custom_domain(self):
        """Test adding a custom prohibited domain."""
        checker = create_domain_checker()

        custom = ProhibitedDomain(
            domain_id="custom_secret",
            name="Custom Secret Data",
            category=DomainCategory.CUSTOM,
            level=ProhibitionLevel.STRONG,
            keywords={"topsecret", "classified_custom"},
        )
        checker.add_domain(custom)

        result = checker.check("This is topsecret information")
        assert result.is_prohibited

    def test_remove_domain(self):
        """Test removing a prohibited domain."""
        checker = create_domain_checker()

        custom = ProhibitedDomain(
            domain_id="removable",
            name="Removable Domain",
            category=DomainCategory.CUSTOM,
            level=ProhibitionLevel.DEFAULT,
            keywords={"removable_keyword"},
        )
        checker.add_domain(custom)

        # Check it's there
        result = checker.check("removable_keyword in content")
        assert result.is_prohibited

        # Remove it
        success = checker.remove_domain("removable")
        assert success

        # Check it's gone
        result = checker.check("removable_keyword in content")
        assert not result.is_prohibited

    def test_cannot_remove_absolute(self):
        """Test that ABSOLUTE domains cannot be removed."""
        checker = create_domain_checker()

        # Try to remove SSN domain
        success = checker.remove_domain("pii_ssn")
        assert not success


# =============================================================================
# Abstraction Guard Tests
# =============================================================================

class TestAbstractionGuard:
    """Tests for AbstractionGuard."""

    def test_create_guard(self):
        """Test creating an abstraction guard."""
        guard = create_abstraction_guard()
        assert guard is not None
        assert len(guard.get_rules()) > 0

    def test_abstract_email(self):
        """Test abstracting email addresses."""
        guard = create_abstraction_guard()

        result = guard.abstract(
            "Contact john@example.com for more info",
            level=AbstractionLevel.MINIMAL,
        )

        assert result.success
        assert "[EMAIL]" in result.content
        assert "john@example.com" not in result.content

    def test_abstract_phone(self):
        """Test abstracting phone numbers."""
        guard = create_abstraction_guard()

        result = guard.abstract(
            "Call me at 555-123-4567",
            level=AbstractionLevel.MINIMAL,
        )

        assert result.success
        assert "[PHONE]" in result.content

    def test_abstract_ssn(self):
        """Test abstracting SSN."""
        guard = create_abstraction_guard()

        result = guard.abstract(
            "SSN: 123-45-6789",
            level=AbstractionLevel.MINIMAL,
        )

        assert result.success
        assert "[SSN]" in result.content

    def test_abstract_credit_card(self):
        """Test abstracting credit card numbers."""
        guard = create_abstraction_guard()

        result = guard.abstract(
            "Card: 1234-5678-9012-3456",
            level=AbstractionLevel.MINIMAL,
        )

        assert result.success
        assert "[CREDIT_CARD]" in result.content

    def test_raw_level_no_abstraction(self):
        """Test that RAW level doesn't abstract."""
        guard = create_abstraction_guard()

        original = "Email: test@test.com, Phone: 555-123-4567"
        result = guard.abstract(original, level=AbstractionLevel.RAW)

        assert result.success
        assert result.content == original

    def test_abstraction_ratio(self):
        """Test abstraction ratio calculation."""
        guard = create_abstraction_guard()

        result = guard.abstract(
            "Contact: test@test.com",
            level=AbstractionLevel.MINIMAL,
        )

        assert result.abstraction_ratio > 0

    def test_verify_abstraction(self):
        """Test verifying abstraction level."""
        guard = create_abstraction_guard()

        # Unabstracted content fails verification
        valid, issues = guard.verify_abstraction(
            "Email: test@test.com",
            AbstractionLevel.MINIMAL,
        )
        assert not valid
        assert len(issues) > 0

        # Abstracted content passes
        abstracted = guard.abstract("Email: test@test.com", AbstractionLevel.MINIMAL)
        valid, issues = guard.verify_abstraction(
            abstracted.content,
            AbstractionLevel.MINIMAL,
        )
        assert valid

    def test_pseudonymize(self):
        """Test pseudonymization."""
        guard = create_abstraction_guard()

        result = guard.pseudonymize("Contact John Smith at john@example.com")

        assert result.success
        assert "John Smith" not in result.content
        assert "john@example.com" not in result.content
        assert "[ENTITY_" in result.content

    def test_consistent_pseudonyms(self):
        """Test that pseudonyms are consistent."""
        guard = create_abstraction_guard()

        result1 = guard.pseudonymize("John Smith said hello")
        result2 = guard.pseudonymize("John Smith said goodbye")

        # Same name should get same pseudonym
        assert "[ENTITY_1]" in result1.content
        assert "[ENTITY_1]" in result2.content

    def test_aggregate(self):
        """Test data aggregation."""
        guard = create_abstraction_guard()

        items = ["short", "medium length", "a very long string here"]

        count = guard.aggregate(items, "count")
        assert count["total_count"] == 3

        stats = guard.aggregate(items, "length_stats")
        assert stats["count"] == 3
        assert stats["min"] == 5  # len("short")
        assert stats["max"] == 23  # len("a very long string here")

    def test_check_identifiability(self):
        """Test checking content identifiability."""
        guard = create_abstraction_guard()

        result = guard.check_identifiability(
            "John Smith (john@example.com) called 555-123-4567"
        )

        assert result["potential_names"] >= 1
        assert result["potential_emails"] == 1
        assert result["potential_phones"] == 1
        assert result["identifiability_score"] > 0

    def test_add_custom_rule(self):
        """Test adding a custom abstraction rule."""
        guard = create_abstraction_guard()

        rule = AbstractionRule(
            rule_id="custom_id",
            name="Custom ID",
            pattern=r"\bCUST-\d{6}\b",
            abstraction_type=AbstractionType.REDACTION,
            replacement="[CUSTOM_ID]",
            min_level=AbstractionLevel.MINIMAL,
        )
        guard.add_rule(rule)

        result = guard.abstract("Customer CUST-123456", level=AbstractionLevel.MINIMAL)
        assert "[CUSTOM_ID]" in result.content


# =============================================================================
# Enforcement Engine Tests
# =============================================================================

class TestLearningContractsEngine:
    """Tests for LearningContractsEngine."""

    def test_create_engine(self):
        """Test creating an enforcement engine."""
        engine = create_learning_contracts_engine()
        assert engine is not None
        engine.shutdown()

    def test_check_learning_no_contract(self):
        """Test checking learning without a contract."""
        engine = create_learning_contracts_engine(default_deny=True)

        result = engine.check_learning(
            user_id="user123",
            content="Some content",
            domain="general",
        )

        assert not result.allowed
        assert result.decision == EnforcementDecision.DENY_NO_CONTRACT
        engine.shutdown()

    def test_check_learning_with_contract(self):
        """Test checking learning with a valid contract."""
        engine = create_learning_contracts_engine()

        # Create contract
        scope = ContractScope(scope_type=LearningScope.ALL)
        engine.create_contract(
            user_id="user123",
            contract_type=ContractType.FULL_CONSENT,
            scope=scope,
            auto_activate=True,
        )

        result = engine.check_learning(
            user_id="user123",
            content="Some learning content",
        )

        assert result.allowed
        assert result.decision == EnforcementDecision.ALLOW
        engine.shutdown()

    def test_check_learning_prohibited_domain(self):
        """Test checking learning with prohibited domain content."""
        engine = create_learning_contracts_engine()

        # Create contract
        scope = ContractScope(scope_type=LearningScope.ALL)
        engine.create_contract(
            user_id="user123",
            contract_type=ContractType.FULL_CONSENT,
            scope=scope,
            auto_activate=True,
        )

        # Content with SSN
        result = engine.check_learning(
            user_id="user123",
            content="My SSN is 123-45-6789",
        )

        assert not result.allowed
        assert result.decision == EnforcementDecision.DENY_DOMAIN
        engine.shutdown()

    def test_check_learning_abstraction_only(self):
        """Test abstraction-only contract."""
        engine = create_learning_contracts_engine(enable_abstraction=True)

        # Create abstraction-only contract
        engine.create_abstraction_only_contract("user123")

        result = engine.check_learning(
            user_id="user123",
            content="Email: test@test.com",
        )

        assert result.allowed
        assert result.decision == EnforcementDecision.ALLOW_ABSTRACTED
        assert result.requires_abstraction
        assert result.abstracted_content is not None
        assert "[EMAIL]" in result.abstracted_content
        engine.shutdown()

    def test_create_default_deny_contract(self):
        """Test creating a default deny contract."""
        engine = create_learning_contracts_engine()

        contract = engine.create_default_deny_contract("user123")

        assert contract.contract_type == ContractType.NO_LEARNING

        result = engine.check_learning(
            user_id="user123",
            content="Any content",
        )

        assert not result.allowed
        engine.shutdown()

    def test_create_full_consent_contract(self):
        """Test creating a full consent contract."""
        engine = create_learning_contracts_engine()

        contract = engine.create_full_consent_contract(
            "user123",
            excluded_domains={"medical"},
        )

        assert contract.contract_type == ContractType.FULL_CONSENT
        assert "medical" in contract.scope.excluded_domains
        engine.shutdown()

    def test_revoke_all_contracts(self):
        """Test revoking all contracts for a user."""
        engine = create_learning_contracts_engine()

        scope = ContractScope(scope_type=LearningScope.ALL)
        engine.create_contract(
            user_id="user123",
            contract_type=ContractType.FULL_CONSENT,
            scope=scope,
            auto_activate=True,
        )
        engine.create_contract(
            user_id="user123",
            contract_type=ContractType.LIMITED_CONSENT,
            scope=scope,
            auto_activate=True,
        )

        count = engine.revoke_all_contracts("user123")
        assert count == 2

        contracts = engine.get_user_contracts("user123", active_only=True)
        assert len(contracts) == 0
        engine.shutdown()

    def test_add_prohibited_domain(self):
        """Test adding a custom prohibited domain."""
        engine = create_learning_contracts_engine()

        engine.add_prohibited_domain(
            name="Custom Secret",
            keywords={"customsecret"},
        )

        result = engine.check_domain("This contains customsecret info")
        assert result.is_prohibited
        engine.shutdown()

    def test_statistics(self):
        """Test getting engine statistics."""
        engine = create_learning_contracts_engine()

        engine.check_learning("user123", "content1")
        engine.check_learning("user123", "content2")

        stats = engine.get_statistics()
        assert stats["total_checks"] == 2
        engine.shutdown()


# =============================================================================
# Consent Prompt Tests
# =============================================================================

class TestConsentPrompt:
    """Tests for ConsentPrompt."""

    def test_create_prompt(self):
        """Test creating a consent prompt."""
        prompt = create_consent_prompt()
        assert prompt is not None

    def test_auto_deny_mode(self):
        """Test auto-deny mode."""
        prompt = create_consent_prompt(mode=ConsentMode.AUTO_DENY)

        request = ConsentRequest(
            request_id="CR-001",
            user_id="user123",
            domain="general",
            description="Test request",
        )

        decision = prompt.prompt(request)
        assert decision.response == ConsentResponse.DENY

    def test_callback_mode(self):
        """Test callback mode."""
        def consent_callback(request):
            return ConsentDecision(
                request_id=request.request_id,
                response=ConsentResponse.ALLOW,
                user_id=request.user_id,
                contract_type=ContractType.FULL_CONSENT,
            )

        prompt = create_consent_prompt(
            mode=ConsentMode.CALLBACK,
            callback=consent_callback,
        )

        request = ConsentRequest(
            request_id="CR-001",
            user_id="user123",
            domain="general",
            description="Test request",
        )

        decision = prompt.prompt(request)
        assert decision.response == ConsentResponse.ALLOW

    def test_preference_setting(self):
        """Test setting preferences."""
        prompt = create_consent_prompt(mode=ConsentMode.AUTO_DENY)

        # Set preference to allow
        prompt.set_preference("user123", "general", ConsentResponse.ALLOW)

        request = ConsentRequest(
            request_id="CR-001",
            user_id="user123",
            domain="general",
            description="Test request",
        )

        decision = prompt.prompt(request)
        assert decision.response == ConsentResponse.ALLOW

    def test_wildcard_preference(self):
        """Test wildcard preference."""
        prompt = create_consent_prompt(mode=ConsentMode.AUTO_DENY)

        prompt.set_preference("user123", "*", ConsentResponse.ALLOW)

        request = ConsentRequest(
            request_id="CR-001",
            user_id="user123",
            domain="any_domain",
            description="Test request",
        )

        decision = prompt.prompt(request)
        assert decision.response == ConsentResponse.ALLOW

    def test_decision_caching(self):
        """Test decision caching."""
        call_count = 0

        def consent_callback(request):
            nonlocal call_count
            call_count += 1
            return ConsentDecision(
                request_id=request.request_id,
                response=ConsentResponse.ALLOW,
                user_id=request.user_id,
            )

        prompt = create_consent_prompt(
            mode=ConsentMode.CALLBACK,
            callback=consent_callback,
        )

        request1 = ConsentRequest(
            request_id="CR-001",
            user_id="user123",
            domain="general",
            description="Test",
        )
        request2 = ConsentRequest(
            request_id="CR-002",
            user_id="user123",
            domain="general",
            description="Test",
        )

        prompt.prompt(request1)
        prompt.prompt(request2)

        # Second call should use cache
        assert call_count == 1

    def test_create_decision_helper(self):
        """Test creating a decision using helper."""
        prompt = create_consent_prompt()

        request = ConsentRequest(
            request_id="CR-001",
            user_id="user123",
            domain="general",
            description="Test",
        )

        decision = prompt.create_decision(
            request,
            ConsentResponse.ALLOW_ABSTRACTED,
            duration_days=30,
        )

        assert decision.response == ConsentResponse.ALLOW_ABSTRACTED
        assert decision.contract_type == ContractType.ABSTRACTION_ONLY
        assert decision.duration == timedelta(days=30)


# =============================================================================
# Contracts Client Tests
# =============================================================================

class TestContractsClient:
    """Tests for ContractsClient."""

    def test_create_client(self):
        """Test creating a contracts client."""
        client = create_contracts_client()
        assert client is not None
        client.shutdown()

    def test_can_learn_no_contract(self):
        """Test can_learn without contract."""
        client = create_contracts_client(default_deny=True)

        result = client.can_learn("user123", domain="general")
        assert not result
        client.shutdown()

    def test_can_learn_with_contract(self):
        """Test can_learn with valid contract."""
        client = create_contracts_client()

        # Create session contract
        client.create_session_contract("user123")

        result = client.can_learn("user123", domain="general")
        assert result
        client.shutdown()

    def test_check_learning(self):
        """Test full learning check."""
        client = create_contracts_client()

        client.create_session_contract("user123")

        result = client.check_learning(
            user_id="user123",
            content="Test content",
            domain="general",
        )

        assert result.allowed
        client.shutdown()

    def test_abstract_content(self):
        """Test content abstraction."""
        client = create_contracts_client()

        result = client.abstract_content("Email: test@test.com")

        assert "[EMAIL]" in result.content
        client.shutdown()

    def test_revoke_contract(self):
        """Test revoking a contract."""
        client = create_contracts_client()

        contract = client.create_session_contract("user123")

        success = client.revoke_contract(
            contract.contract_id,
            "user123",
            "Testing revocation",
        )

        assert success

        # Check it's gone
        contracts = client.get_active_contracts("user123")
        assert len(contracts) == 0
        client.shutdown()

    def test_revoke_all(self):
        """Test revoking all contracts."""
        client = create_contracts_client()

        client.create_session_contract("user123")
        client.create_session_contract("user123", domains={"research"})

        count = client.revoke_all("user123")
        assert count == 2
        client.shutdown()

    def test_check_prohibited_domain(self):
        """Test checking for prohibited domains."""
        client = create_contracts_client()

        prohibited = client.check_prohibited_domain("SSN: 123-45-6789")
        assert prohibited

        clean = client.check_prohibited_domain("Hello world")
        assert not clean
        client.shutdown()

    def test_has_valid_contract(self):
        """Test checking for valid contract."""
        client = create_contracts_client()

        assert not client.has_valid_contract("user123")

        client.create_session_contract("user123")

        assert client.has_valid_contract("user123")
        client.shutdown()

    def test_statistics(self):
        """Test getting statistics."""
        client = create_contracts_client()

        client.check_learning("user123", "content1")
        client.check_learning("user123", "content2")

        stats = client.get_statistics()
        assert stats["request_count"] == 2
        client.shutdown()

    def test_consent_callback(self):
        """Test setting consent callback."""
        client = create_contracts_client()

        decisions = []

        def callback(request):
            decision = ConsentDecision(
                request_id=request.request_id,
                response=ConsentResponse.ALLOW,
                user_id=request.user_id,
                contract_type=ContractType.SESSION_ONLY,
            )
            decisions.append(decision)
            return decision

        client.set_consent_callback(callback)

        decision = client.request_consent(
            user_id="user123",
            domain="general",
            description="Test consent",
        )

        assert len(decisions) == 1
        assert decision.response == ConsentResponse.ALLOW
        client.shutdown()

    def test_domain_preference(self):
        """Test setting domain preferences."""
        client = create_contracts_client()

        client.set_domain_preference(
            "user123",
            "general",
            ConsentResponse.ALLOW,
        )

        decision = client.request_consent(
            user_id="user123",
            domain="general",
            description="Test",
        )

        assert decision.response == ConsentResponse.ALLOW
        client.shutdown()

    def test_create_contract_from_consent(self):
        """Test creating contract from consent decision."""
        client = create_contracts_client()

        decision = ConsentDecision(
            request_id="CR-001",
            response=ConsentResponse.ALLOW,
            user_id="user123",
            contract_type=ContractType.FULL_CONSENT,
            duration=timedelta(days=30),
        )

        contract = client.create_contract_from_consent(decision)

        assert contract is not None
        assert contract.contract_type == ContractType.FULL_CONSENT
        client.shutdown()


# =============================================================================
# Integration Tests
# =============================================================================

class TestLearningContractsIntegration:
    """Integration tests for the complete system."""

    def test_full_workflow(self):
        """Test complete learning contracts workflow."""
        client = create_contracts_client()

        # 1. Check learning without contract - should fail
        result = client.check_learning(
            user_id="user123",
            content="Learn this data",
            domain="general",
        )
        assert not result.allowed

        # 2. Request consent
        decision = client.request_consent(
            user_id="user123",
            domain="general",
            description="Allow learning of general topics",
        )
        # Default is auto-deny
        assert decision.response == ConsentResponse.DENY

        # 3. Create contract manually
        contract = client.create_session_contract("user123")
        assert contract is not None

        # 4. Check learning again - should succeed
        result = client.check_learning(
            user_id="user123",
            content="Learn this data",
            domain="general",
        )
        assert result.allowed

        # 5. Check prohibited content - should fail
        result = client.check_learning(
            user_id="user123",
            content="SSN: 123-45-6789",
        )
        assert not result.allowed

        # 6. Revoke contract
        client.revoke_all("user123")

        # 7. Check again - should fail
        result = client.check_learning(
            user_id="user123",
            content="Learn this",
        )
        assert not result.allowed

        client.shutdown()

    def test_abstraction_workflow(self):
        """Test abstraction-based learning workflow."""
        client = create_contracts_client()

        # Create abstraction-only contract
        client.engine.create_abstraction_only_contract("user123")

        # Check learning with PII
        result = client.check_learning(
            user_id="user123",
            content="Contact john@example.com at 555-123-4567",
        )

        assert result.allowed
        assert result.requires_abstraction
        assert "[EMAIL]" in result.abstracted_content
        assert "[PHONE]" in result.abstracted_content

        client.shutdown()

    def test_domain_scoping(self):
        """Test domain-scoped contracts."""
        client = create_contracts_client()

        # Create domain-specific contract
        scope = ContractScope(
            scope_type=LearningScope.DOMAIN_SPECIFIC,
            domains={"research", "general"},
            excluded_domains={"medical"},
        )
        client.engine.create_contract(
            user_id="user123",
            contract_type=ContractType.FULL_CONSENT,
            scope=scope,
            auto_activate=True,
        )

        # Research domain - allowed
        result = client.check_learning(
            user_id="user123",
            content="Research data",
            domain="research",
        )
        assert result.allowed

        # Medical domain - not allowed (excluded)
        result = client.check_learning(
            user_id="user123",
            content="Medical data",
            domain="medical",
        )
        assert not result.allowed

        # Finance domain - not allowed (not in scope)
        result = client.check_learning(
            user_id="user123",
            content="Finance data",
            domain="finance",
        )
        assert not result.allowed

        client.shutdown()

    def test_persistence(self):
        """Test contract persistence."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "contracts.db"

            # Create client and contract
            client1 = create_contracts_client(db_path=db_path)
            client1.create_session_contract("user123")
            client1.shutdown()

            # Create new client with same database
            client2 = create_contracts_client(db_path=db_path)

            # Contract should still exist
            contracts = client2.get_active_contracts("user123")
            assert len(contracts) == 1

            client2.shutdown()
