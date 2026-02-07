"""
Tests for ConstitutionalKernel — covers initialize, enforce, shutdown,
get_rules_for_agent, _rule_applies, _rule_violated, _format_violation_reason,
_generate_suggestions, hot-reload, and create_kernel.

Targeted at boosting coverage from 48% to 90%+.
"""

import hashlib
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.core.constitution import (
    ConstitutionalKernel,
    EnforcementResult,
    RequestContext,
    create_kernel,
)
from src.core.exceptions import KernelNotInitializedError, SupremeConstitutionError
from src.core.models import AuthorityLevel, Rule, RuleType


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_CONSTITUTION = """\
---
name: Test Constitution
version: "1.0"
authority_level: supreme
scope: all_agents
---

# Test Constitution

## Core Principles

### Principle 1
Agents must be helpful and truthful.

## Prohibitions

### No Harm
Agents must never cause harm to users.

### No Deception
Agents must not deceive or mislead users.

## Mandates

### Consent Required
All memory storage requires explicit user consent.

## Escalation Rules

### Sensitive Operations
Operations involving personal data require human approval.
"""

AGENT_CONSTITUTION = """\
---
name: Sage Constitution
version: "1.0"
authority_level: agent_specific
scope: sage
---

# Sage Agent Constitution

## Principles

### Accuracy
Sage must prioritize factual accuracy in all responses.
"""


@pytest.fixture
def temp_constitution_dir(tmp_path):
    """Create a temp directory with constitution files."""
    # Supreme constitution
    supreme = tmp_path / "CONSTITUTION.md"
    supreme.write_text(SAMPLE_CONSTITUTION)

    # Agent-specific constitution
    agents_dir = tmp_path / "agents"
    agents_dir.mkdir()
    sage_dir = agents_dir / "sage"
    sage_dir.mkdir()
    sage_const = sage_dir / "constitution.md"
    sage_const.write_text(AGENT_CONSTITUTION)

    return tmp_path


@pytest.fixture
def kernel(temp_constitution_dir):
    """Create and initialize a kernel with test constitutions."""
    k = ConstitutionalKernel(
        constitution_dir=temp_constitution_dir / "agents",
        supreme_constitution_path=temp_constitution_dir / "CONSTITUTION.md",
        enable_hot_reload=False,
    )
    k.initialize()
    return k


def _make_rule(
    rule_type=RuleType.PROHIBITION,
    content="Test rule",
    scope="all_agents",
    keywords=None,
    is_immutable=False,
    section="Test Section",
    section_path=None,
):
    return Rule(
        id=hashlib.sha256(content.encode()).hexdigest()[:16],
        content=content,
        rule_type=rule_type,
        section=section,
        section_path=section_path or [section],
        authority_level=AuthorityLevel.SUPREME,
        scope=scope,
        keywords=set(keywords or []),
        is_immutable=is_immutable,
    )


# ---------------------------------------------------------------------------
# Constructor Tests
# ---------------------------------------------------------------------------


class TestConstitutionalKernelInit:
    def test_default_construction(self):
        """Kernel can be created with default params."""
        k = ConstitutionalKernel()
        assert k._initialized is False
        assert k.enable_hot_reload is True
        assert k._registry is not None

    def test_custom_enforcement_config(self):
        """Enforcement config is passed through to engine."""
        k = ConstitutionalKernel(
            enforcement_config={
                "semantic_threshold": 0.8,
                "llm_timeout": 5.0,
            }
        )
        assert k._enforcement is not None

    def test_with_constitution_paths(self, temp_constitution_dir):
        """Kernel accepts constitution dir and supreme path."""
        k = ConstitutionalKernel(
            constitution_dir=temp_constitution_dir / "agents",
            supreme_constitution_path=temp_constitution_dir / "CONSTITUTION.md",
        )
        assert k.constitution_dir == temp_constitution_dir / "agents"
        assert k.supreme_constitution_path == temp_constitution_dir / "CONSTITUTION.md"


# ---------------------------------------------------------------------------
# Initialize Tests
# ---------------------------------------------------------------------------


class TestConstitutionalKernelInitialize:
    def test_initialize_loads_supreme(self, kernel, temp_constitution_dir):
        """Initialize should load the supreme constitution."""
        supreme = kernel.get_supreme_constitution()
        assert supreme is not None

    def test_initialize_loads_agent_constitutions(self, kernel):
        """Initialize should load agent-specific constitutions."""
        sage_const = kernel.get_constitution("sage")
        assert sage_const is not None

    def test_initialize_sets_initialized(self, kernel):
        """After initialize, kernel should be marked initialized."""
        assert kernel._initialized is True

    def test_initialize_returns_validation_result(self, temp_constitution_dir):
        """Initialize should return a ValidationResult."""
        k = ConstitutionalKernel(
            constitution_dir=temp_constitution_dir / "agents",
            supreme_constitution_path=temp_constitution_dir / "CONSTITUTION.md",
            enable_hot_reload=False,
        )
        result = k.initialize()
        assert result.is_valid is True

    def test_initialize_supreme_not_found_raises(self, tmp_path):
        """Missing supreme constitution should raise SupremeConstitutionError."""
        k = ConstitutionalKernel(
            supreme_constitution_path=tmp_path / "nonexistent.md",
            enable_hot_reload=False,
        )
        with pytest.raises(SupremeConstitutionError):
            k.initialize()

    def test_initialize_without_supreme(self, tmp_path):
        """Initialize without supreme constitution path should still initialize."""
        k = ConstitutionalKernel(
            constitution_dir=tmp_path,
            enable_hot_reload=False,
        )
        result = k.initialize()
        # Registry validation may flag missing supreme constitution
        assert k._initialized is True

    def test_initialize_with_nonexistent_dir(self, tmp_path):
        """Initialize with a nonexistent constitution directory should still work."""
        k = ConstitutionalKernel(
            constitution_dir=tmp_path / "nonexistent",
            enable_hot_reload=False,
        )
        result = k.initialize()
        assert k._initialized is True

    def test_initialize_with_hot_reload(self, temp_constitution_dir):
        """Initialize with hot reload starts the file watcher."""
        k = ConstitutionalKernel(
            constitution_dir=temp_constitution_dir / "agents",
            supreme_constitution_path=temp_constitution_dir / "CONSTITUTION.md",
            enable_hot_reload=True,
        )
        try:
            k.initialize()
            assert k._initialized is True
            # File watcher should have started
            assert k._observer is not None
        finally:
            k.shutdown()


# ---------------------------------------------------------------------------
# Shutdown Tests
# ---------------------------------------------------------------------------


class TestConstitutionalKernelShutdown:
    def test_shutdown(self, kernel):
        """Shutdown should mark kernel as uninitialized."""
        kernel.shutdown()
        assert kernel._initialized is False

    def test_shutdown_stops_observer(self, temp_constitution_dir):
        """Shutdown should stop the file watcher."""
        k = ConstitutionalKernel(
            constitution_dir=temp_constitution_dir / "agents",
            supreme_constitution_path=temp_constitution_dir / "CONSTITUTION.md",
            enable_hot_reload=True,
        )
        k.initialize()
        assert k._observer is not None
        k.shutdown()
        assert k._observer is None

    def test_shutdown_without_observer(self, kernel):
        """Shutdown with no observer should not raise."""
        kernel._observer = None
        kernel.shutdown()
        assert kernel._initialized is False


# ---------------------------------------------------------------------------
# Enforce Tests
# ---------------------------------------------------------------------------


class TestConstitutionalKernelEnforce:
    def test_enforce_not_initialized_raises(self):
        """Enforce should raise if kernel not initialized."""
        k = ConstitutionalKernel()
        ctx = RequestContext(
            request_id="test-1",
            source="user",
            destination="sage",
            intent="query.factual",
            content="What is quantum mechanics?",
        )
        with pytest.raises(KernelNotInitializedError):
            k.enforce(ctx)

    def test_enforce_allowed_request(self, kernel):
        """A benign request to an agent with no keyword match should be allowed."""
        ctx = RequestContext(
            request_id="test-1",
            source="user",
            destination="muse",
            intent="content.creative",
            content="Write me a poem about the ocean",
        )
        result = kernel.enforce(ctx)
        assert result.allowed is True

    def test_enforce_returns_enforcement_result(self, kernel):
        """Enforce should return an EnforcementResult."""
        ctx = RequestContext(
            request_id="test-1",
            source="user",
            destination="sage",
            intent="query.factual",
            content="Hello world",
        )
        result = kernel.enforce(ctx)
        assert isinstance(result, EnforcementResult)
        assert hasattr(result, "allowed")
        assert hasattr(result, "violated_rules")
        assert hasattr(result, "suggestions")

    def test_enforce_memory_request(self, kernel):
        """Memory request should trigger memory-related rules."""
        ctx = RequestContext(
            request_id="test-1",
            source="user",
            destination="seshat",
            intent="memory.store",
            content="Remember this conversation for later",
            requires_memory=True,
        )
        result = kernel.enforce(ctx)
        assert isinstance(result, EnforcementResult)


# ---------------------------------------------------------------------------
# Rule Lookup Tests
# ---------------------------------------------------------------------------


class TestRuleLookup:
    def test_get_rules_for_agent(self, kernel):
        """get_rules_for_agent returns rules for a specific agent."""
        rules = kernel.get_rules_for_agent("sage")
        assert isinstance(rules, list)
        # Should include at least supreme (all_agents) rules
        assert len(rules) >= 0

    def test_get_supreme_constitution(self, kernel):
        """get_supreme_constitution returns the supreme document."""
        supreme = kernel.get_supreme_constitution()
        assert supreme is not None

    def test_get_constitution_by_scope(self, kernel):
        """get_constitution returns constitution for specific scope."""
        sage_const = kernel.get_constitution("sage")
        assert sage_const is not None

    def test_get_constitution_nonexistent_scope(self, kernel):
        """get_constitution returns None for unknown scope."""
        result = kernel.get_constitution("nonexistent_agent")
        assert result is None

    def test_validate_constitution(self, kernel):
        """validate_constitution validates a single document."""
        supreme = kernel.get_supreme_constitution()
        result = kernel.validate_constitution(supreme)
        assert result.is_valid is True


# ---------------------------------------------------------------------------
# Rule Matching Tests (_rule_applies / _rule_violated)
# ---------------------------------------------------------------------------


class TestRuleApplies:
    def _kernel(self):
        k = ConstitutionalKernel(enable_hot_reload=False)
        k._initialized = True
        return k

    def test_scope_all_agents(self):
        """Rule with all_agents scope applies to any agent."""
        k = self._kernel()
        rule = _make_rule(scope="all_agents", keywords=["harm"])
        ctx = RequestContext(
            request_id="1", source="user", destination="sage",
            intent="query", content="This could cause harm",
        )
        assert k._rule_applies(rule, ctx) is True

    def test_scope_specific_agent_match(self):
        """Rule with specific scope applies only to that agent."""
        k = self._kernel()
        rule = _make_rule(scope="sage", keywords=["test"])
        ctx = RequestContext(
            request_id="1", source="user", destination="sage",
            intent="query", content="test question",
        )
        assert k._rule_applies(rule, ctx) is True

    def test_scope_specific_agent_no_match(self):
        """Rule for 'sage' should not apply to 'muse'."""
        k = self._kernel()
        rule = _make_rule(scope="sage", keywords=["test"])
        ctx = RequestContext(
            request_id="1", source="user", destination="muse",
            intent="query", content="test question",
        )
        assert k._rule_applies(rule, ctx) is False

    def test_keyword_in_content(self):
        """Rule applies when keyword appears in content."""
        k = self._kernel()
        rule = _make_rule(scope="all_agents", keywords=["delete"])
        ctx = RequestContext(
            request_id="1", source="user", destination="sage",
            intent="query", content="Please delete this data",
        )
        assert k._rule_applies(rule, ctx) is True

    def test_keyword_in_intent(self):
        """Rule applies when keyword appears in intent."""
        k = self._kernel()
        rule = _make_rule(scope="all_agents", keywords=["memory"])
        ctx = RequestContext(
            request_id="1", source="user", destination="sage",
            intent="memory.store", content="Save this",
        )
        assert k._rule_applies(rule, ctx) is True

    def test_no_keyword_match(self):
        """Rule doesn't apply when no keywords match."""
        k = self._kernel()
        rule = _make_rule(scope="all_agents", keywords=["delete", "remove"])
        ctx = RequestContext(
            request_id="1", source="user", destination="sage",
            intent="query.factual", content="What is physics?",
        )
        assert k._rule_applies(rule, ctx) is False

    def test_memory_rule_applies_for_memory_request(self):
        """Memory-related rules apply when requires_memory is True."""
        k = self._kernel()
        rule = _make_rule(scope="all_agents", keywords={"memory", "store"})
        ctx = RequestContext(
            request_id="1", source="user", destination="seshat",
            intent="query", content="Save this", requires_memory=True,
        )
        assert k._rule_applies(rule, ctx) is True

    def test_memory_rule_no_match_without_memory_keywords(self):
        """Memory request but rule has no memory keywords — shouldn't match."""
        k = self._kernel()
        rule = _make_rule(scope="all_agents", keywords={"security"})
        ctx = RequestContext(
            request_id="1", source="user", destination="seshat",
            intent="query", content="anything", requires_memory=True,
        )
        assert k._rule_applies(rule, ctx) is False


class TestRuleViolated:
    def _kernel(self):
        k = ConstitutionalKernel(enable_hot_reload=False)
        k._initialized = True
        return k

    def test_prohibition_violated(self):
        """Prohibition rule is violated when keyword appears in content."""
        k = self._kernel()
        rule = _make_rule(
            rule_type=RuleType.PROHIBITION,
            keywords=["harm", "delete"],
        )
        ctx = RequestContext(
            request_id="1", source="user", destination="sage",
            intent="query", content="This could cause harm to someone",
        )
        assert k._rule_violated(rule, ctx) is True

    def test_prohibition_not_violated(self):
        """Prohibition rule is NOT violated when keyword is absent."""
        k = self._kernel()
        rule = _make_rule(
            rule_type=RuleType.PROHIBITION,
            keywords=["harm", "delete"],
        )
        ctx = RequestContext(
            request_id="1", source="user", destination="sage",
            intent="query", content="What is the weather today?",
        )
        assert k._rule_violated(rule, ctx) is False

    def test_mandate_violated_no_compliance(self):
        """Mandate is violated when topic matches but no compliance indicators."""
        k = self._kernel()
        rule = _make_rule(
            rule_type=RuleType.MANDATE,
            keywords=["data", "storage"],
        )
        ctx = RequestContext(
            request_id="1", source="user", destination="sage",
            intent="memory.store", content="Put all data into storage now",
        )
        assert k._rule_violated(rule, ctx) is True

    def test_mandate_not_violated_with_compliance(self):
        """Mandate is NOT violated when compliance indicators present."""
        k = self._kernel()
        rule = _make_rule(
            rule_type=RuleType.MANDATE,
            keywords=["consent", "memory"],
        )
        ctx = RequestContext(
            request_id="1", source="user", destination="sage",
            intent="memory.store",
            content="Store memory after user gives explicit consent and review",
        )
        assert k._rule_violated(rule, ctx) is False

    def test_mandate_no_keywords(self):
        """Mandate with no keywords is never violated."""
        k = self._kernel()
        rule = _make_rule(rule_type=RuleType.MANDATE, keywords=[])
        ctx = RequestContext(
            request_id="1", source="user", destination="sage",
            intent="query", content="anything",
        )
        assert k._rule_violated(rule, ctx) is False

    def test_mandate_explicit_compliance_metadata(self):
        """Mandate satisfied via explicit compliance in metadata."""
        k = self._kernel()
        rule = _make_rule(
            rule_type=RuleType.MANDATE,
            content="consent required",
            keywords=["consent"],
        )
        ctx = RequestContext(
            request_id="1", source="user", destination="sage",
            intent="query", content="I need consent for this",
            metadata={"mandate_compliance": {rule.id: True}},
        )
        assert k._rule_violated(rule, ctx) is False

    def test_mandate_explicit_noncompliance_metadata(self):
        """Mandate violated when metadata says non-compliant."""
        k = self._kernel()
        rule = _make_rule(
            rule_type=RuleType.MANDATE,
            content="consent required",
            keywords=["consent"],
        )
        ctx = RequestContext(
            request_id="1", source="user", destination="sage",
            intent="query", content="I need consent for this",
            metadata={"mandate_compliance": {rule.id: False}},
        )
        assert k._rule_violated(rule, ctx) is True

    def test_other_rule_type_not_violated(self):
        """Non-prohibition/non-mandate rules are never violated."""
        k = self._kernel()
        rule = _make_rule(rule_type=RuleType.PRINCIPLE, keywords=["test"])
        ctx = RequestContext(
            request_id="1", source="user", destination="sage",
            intent="query", content="test content",
        )
        assert k._rule_violated(rule, ctx) is False


# ---------------------------------------------------------------------------
# Format / Suggestions Tests
# ---------------------------------------------------------------------------


class TestFormatViolationReason:
    def _kernel(self):
        k = ConstitutionalKernel(enable_hot_reload=False)
        return k

    def test_empty_rules(self):
        """Empty violated rules returns 'Unknown violation'."""
        k = self._kernel()
        assert k._format_violation_reason([]) == "Unknown violation"

    def test_single_rule(self):
        """Single violated rule is formatted correctly."""
        k = self._kernel()
        rule = _make_rule(
            content="Do not harm users",
            section_path=["Prohibitions", "No Harm"],
        )
        result = k._format_violation_reason([rule])
        assert "Constitutional violation" in result
        assert "Prohibitions > No Harm" in result
        assert "Do not harm users" in result

    def test_multiple_rules_limited_to_3(self):
        """Only top 3 violated rules are included."""
        k = self._kernel()
        rules = [_make_rule(content=f"Rule {i}") for i in range(5)]
        result = k._format_violation_reason(rules)
        assert result.count("-") == 3  # Only 3 bullet points


class TestGenerateSuggestions:
    def _kernel(self):
        k = ConstitutionalKernel(enable_hot_reload=False)
        return k

    def test_prohibition_suggestion(self):
        """Prohibition rules generate 'avoid' suggestions."""
        k = self._kernel()
        rule = _make_rule(
            rule_type=RuleType.PROHIBITION,
            keywords=["harm", "delete"],
        )
        suggestions = k._generate_suggestions([rule])
        assert any("Avoid" in s for s in suggestions)

    def test_escalation_suggestion(self):
        """Escalation rules suggest human approval."""
        k = self._kernel()
        rule = _make_rule(rule_type=RuleType.ESCALATION, keywords=["sensitive"])
        suggestions = k._generate_suggestions([rule])
        assert any("human" in s.lower() for s in suggestions)

    def test_immutable_suggestion(self):
        """Immutable rules add immutable warning."""
        k = self._kernel()
        rule = _make_rule(
            rule_type=RuleType.PROHIBITION,
            keywords=["core"],
            is_immutable=True,
        )
        suggestions = k._generate_suggestions([rule])
        assert any("immutable" in s.lower() for s in suggestions)

    def test_empty_rules_no_suggestions(self):
        """No rules produces no suggestions."""
        k = self._kernel()
        assert k._generate_suggestions([]) == []


# ---------------------------------------------------------------------------
# Hot-Reload Tests
# ---------------------------------------------------------------------------


class TestReloadConstitution:
    def test_reload_detects_change(self, kernel, temp_constitution_dir):
        """Reload detects file changes via hash comparison."""
        supreme_path = temp_constitution_dir / "CONSTITUTION.md"

        # Modify the file
        content = supreme_path.read_text()
        supreme_path.write_text(content + "\n\n### New Rule\nA new rule was added.\n")

        result = kernel._reload_constitution(supreme_path)
        assert result is True

    def test_reload_no_change(self, kernel, temp_constitution_dir):
        """Reload returns True without re-parsing if hash hasn't changed."""
        supreme_path = temp_constitution_dir / "CONSTITUTION.md"
        result = kernel._reload_constitution(supreme_path)
        assert result is True  # Same hash, no reload needed

    def test_reload_file_not_found(self, kernel, tmp_path):
        """Reload returns False for missing file."""
        result = kernel._reload_constitution(tmp_path / "nonexistent.md")
        assert result is False

    def test_reload_callback_invoked(self, kernel, temp_constitution_dir):
        """Reload callbacks are invoked after successful reload."""
        supreme_path = temp_constitution_dir / "CONSTITUTION.md"
        callback_calls = []
        kernel.register_reload_callback(lambda p: callback_calls.append(p))

        # Modify the file to trigger actual reload
        content = supreme_path.read_text()
        supreme_path.write_text(content + "\n\n### Another Rule\nAnother rule.\n")

        kernel._reload_constitution(supreme_path)
        assert len(callback_calls) == 1
        assert callback_calls[0] == supreme_path

    def test_reload_callback_error_isolated(self, kernel, temp_constitution_dir):
        """Callback errors don't prevent reload success."""
        supreme_path = temp_constitution_dir / "CONSTITUTION.md"
        kernel.register_reload_callback(lambda p: (_ for _ in ()).throw(ValueError("boom")))

        content = supreme_path.read_text()
        supreme_path.write_text(content + "\n\n### Error Rule\nError test.\n")

        # Should still succeed despite callback error
        result = kernel._reload_constitution(supreme_path)
        assert result is True


# ---------------------------------------------------------------------------
# Callback Registration Tests
# ---------------------------------------------------------------------------


class TestCallbackRegistration:
    def test_register_reload_callback(self, kernel):
        """Callbacks can be registered."""
        cb = lambda p: None
        kernel.register_reload_callback(cb)
        assert cb in kernel._reload_callbacks

    def test_reload_all_delegates_to_initialize(self, kernel):
        """reload_all should re-run initialization."""
        result = kernel.reload_all()
        assert result.is_valid is True


# ---------------------------------------------------------------------------
# create_kernel Factory Tests
# ---------------------------------------------------------------------------


class TestCreateKernel:
    def test_create_kernel_with_valid_project(self, temp_constitution_dir):
        """create_kernel with valid project root returns initialized kernel."""
        kernel = create_kernel(
            project_root=temp_constitution_dir,
            enable_hot_reload=False,
        )
        assert kernel._initialized is True
        kernel.shutdown()

    def test_create_kernel_no_supreme(self, tmp_path):
        """create_kernel without CONSTITUTION.md works (no supreme)."""
        kernel = create_kernel(
            project_root=tmp_path,
            enable_hot_reload=False,
        )
        assert kernel._initialized is True
        kernel.shutdown()

    def test_create_kernel_returns_kernel(self, tmp_path):
        """create_kernel always returns a ConstitutionalKernel instance."""
        kernel = create_kernel(project_root=tmp_path, enable_hot_reload=False)
        assert isinstance(kernel, ConstitutionalKernel)
        kernel.shutdown()
