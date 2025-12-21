"""Tests for the Conversational Kernel module."""

import json
import os
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict

import pytest

from src.kernel import (
    # Rules
    Rule,
    RuleAction,
    RuleConflict,
    RuleEffect,
    RuleRegistry,
    RuleScope,
    RuleValidationError,
    # Policy
    AccessPolicy,
    FilePolicy,
    Policy,
    PolicyCompiler,
    PolicyType,
    SyscallPolicy,
    # Interpreter
    IntentAction,
    IntentParser,
    ParsedIntent,
    PolicyInterpreter,
    # FUSE
    FuseConfig,
    FuseMount,
    FuseOperation,
    FuseWrapper,
    # eBPF
    EbpfFilter,
    EbpfProgram,
    EbpfProgType,
    SeccompAction,
    SeccompFilter,
    SyscallFilter,
    # Monitor
    AuditEntry,
    AuditLog,
    EventType,
    FileMonitor,
    MonitorEvent,
    # Context
    AgentContext,
    ContextMemory,
    FolderContext,
    UserContext,
    # Engine
    ConversationalKernel,
    KernelConfig,
    KernelState,
)


# ============================================================================
# Rule Tests
# ============================================================================


class TestRuleEnums:
    """Test rule enumeration types."""

    def test_rule_effect_values(self):
        """Test RuleEffect enum values."""
        assert RuleEffect.ALLOW == "allow"
        assert RuleEffect.DENY == "deny"
        assert RuleEffect.AUDIT == "audit"
        assert RuleEffect.PROMPT == "prompt"

    def test_rule_action_values(self):
        """Test RuleAction enum values."""
        assert RuleAction.READ == "read"
        assert RuleAction.WRITE == "write"
        assert RuleAction.DELETE == "delete"
        assert RuleAction.EXECUTE == "execute"
        assert RuleAction.AI_READ == "ai_read"

    def test_rule_scope_values(self):
        """Test RuleScope enum values."""
        assert RuleScope.SYSTEM == "system"
        assert RuleScope.USER == "user"
        assert RuleScope.FOLDER == "folder"
        assert RuleScope.FILE == "file"
        assert RuleScope.AGENT == "agent"


class TestRule:
    """Test Rule dataclass."""

    def test_rule_creation(self):
        """Test creating a rule."""
        rule = Rule(
            rule_id="r_test",
            scope=RuleScope.FOLDER,
            target="/home/user/private",
            effect=RuleEffect.DENY,
            actions=[RuleAction.READ, RuleAction.WRITE],
            reason="Protect private folder",
        )

        assert rule.rule_id == "r_test"
        assert rule.scope == RuleScope.FOLDER
        assert rule.target == "/home/user/private"
        assert rule.effect == RuleEffect.DENY
        assert RuleAction.READ in rule.actions
        assert rule.enabled is True

    def test_rule_auto_id(self):
        """Test auto-generated rule ID."""
        rule = Rule(
            rule_id="",
            scope=RuleScope.FILE,
            target="/test",
            effect=RuleEffect.ALLOW,
            actions=[RuleAction.READ],
            reason="Test",
        )

        assert rule.rule_id.startswith("r_")
        assert len(rule.rule_id) > 2

    def test_rule_matches_action(self):
        """Test rule matching by action."""
        rule = Rule(
            rule_id="r_test",
            scope=RuleScope.FOLDER,
            target="/data",
            effect=RuleEffect.DENY,
            actions=[RuleAction.DELETE],
            reason="Prevent deletion",
        )

        assert rule.matches("/data/file.txt", RuleAction.DELETE) is True
        assert rule.matches("/data/file.txt", RuleAction.READ) is False

    def test_rule_matches_path(self):
        """Test rule matching by path."""
        rule = Rule(
            rule_id="r_test",
            scope=RuleScope.FOLDER,
            target="/data",
            effect=RuleEffect.DENY,
            actions=[RuleAction.WRITE],
            reason="Read-only data",
        )

        assert rule.matches("/data/subdir/file.txt", RuleAction.WRITE) is True
        assert rule.matches("/other/file.txt", RuleAction.WRITE) is False

    def test_rule_to_dict(self):
        """Test rule serialization."""
        rule = Rule(
            rule_id="r_test",
            scope=RuleScope.FILE,
            target="/test.txt",
            effect=RuleEffect.ALLOW,
            actions=[RuleAction.READ],
            reason="Allow reading",
        )

        data = rule.to_dict()
        assert data["rule_id"] == "r_test"
        assert data["scope"] == "file"
        assert data["effect"] == "allow"
        assert "read" in data["actions"]

    def test_rule_from_dict(self):
        """Test rule deserialization."""
        data = {
            "rule_id": "r_test",
            "scope": "folder",
            "target": "/data",
            "effect": "deny",
            "actions": ["read", "write"],
            "reason": "Test rule",
            "priority": 100,
            "enabled": True,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "created_by": "user",
            "version": 1,
            "parent_rule_id": None,
            "tags": [],
            "metadata": {},
            "conditions": {},
        }

        rule = Rule.from_dict(data)
        assert rule.rule_id == "r_test"
        assert rule.scope == RuleScope.FOLDER
        assert rule.effect == RuleEffect.DENY


class TestRuleRegistry:
    """Test RuleRegistry class."""

    def test_registry_creation(self):
        """Test creating a rule registry."""
        registry = RuleRegistry()
        assert registry is not None

    def test_add_rule(self):
        """Test adding a rule."""
        registry = RuleRegistry()
        rule = Rule(
            rule_id="r_test",
            scope=RuleScope.FOLDER,
            target="/data",
            effect=RuleEffect.DENY,
            actions=[RuleAction.DELETE],
            reason="Prevent deletion",
        )

        rule_id = registry.add_rule(rule)
        assert rule_id == "r_test"

        retrieved = registry.get_rule("r_test")
        assert retrieved is not None
        assert retrieved.target == "/data"

    def test_delete_rule(self):
        """Test deleting a rule."""
        registry = RuleRegistry()
        rule = Rule(
            rule_id="r_test",
            scope=RuleScope.FILE,
            target="/test.txt",
            effect=RuleEffect.ALLOW,
            actions=[RuleAction.READ],
            reason="Test",
        )

        registry.add_rule(rule)
        assert registry.delete_rule("r_test") is True
        assert registry.get_rule("r_test") is None

    def test_get_rules_for_target(self):
        """Test getting rules for a target."""
        registry = RuleRegistry()

        rule1 = Rule(
            rule_id="r1",
            scope=RuleScope.FOLDER,
            target="/data",
            effect=RuleEffect.DENY,
            actions=[RuleAction.WRITE],
            reason="Read-only",
        )
        rule2 = Rule(
            rule_id="r2",
            scope=RuleScope.FOLDER,
            target="/data/logs",
            effect=RuleEffect.ALLOW,
            actions=[RuleAction.WRITE],
            reason="Allow logs",
            priority=200,  # Higher priority to allow override
        )

        registry.add_rule(rule1)
        registry.add_rule(rule2, check_conflicts=False)

        rules = registry.get_rules_for_target("/data/logs/app.log")
        assert len(rules) == 2

    def test_evaluate_rule(self):
        """Test rule evaluation."""
        registry = RuleRegistry()

        rule = Rule(
            rule_id="r_test",
            scope=RuleScope.FOLDER,
            target="/private",
            effect=RuleEffect.DENY,
            actions=[RuleAction.READ, RuleAction.WRITE],
            reason="Private folder",
        )
        registry.add_rule(rule)

        effect, matched = registry.evaluate("/private/secret.txt", RuleAction.READ)
        assert effect == RuleEffect.DENY
        assert matched is not None
        assert matched.rule_id == "r_test"


# ============================================================================
# Policy Tests
# ============================================================================


class TestSyscallPolicy:
    """Test SyscallPolicy class."""

    def test_syscall_policy_creation(self):
        """Test creating a syscall policy."""
        from src.kernel.policy import SyscallAction

        policy = SyscallPolicy(
            policy_id="pol_1",
            syscalls=["open", "read", "write"],
            action=SyscallAction.ALLOW,
        )

        assert policy.policy_id == "pol_1"
        assert "open" in policy.syscalls
        assert policy.action == SyscallAction.ALLOW

    def test_syscall_policy_to_dict(self):
        """Test syscall policy serialization."""
        from src.kernel.policy import SyscallAction

        policy = SyscallPolicy(
            policy_id="pol_1",
            syscalls=["unlink", "rmdir"],
            action=SyscallAction.DENY,
            errno_value=13,
        )

        data = policy.to_dict()
        assert data["policy_id"] == "pol_1"
        assert "unlink" in data["syscalls"]


class TestFilePolicy:
    """Test FilePolicy class."""

    def test_file_policy_creation(self):
        """Test creating a file policy."""
        policy = FilePolicy(
            policy_id="fpol_1",
            path="/data",
            operations=["read", "write"],
            effect="deny",
            recursive=True,
        )

        assert policy.policy_id == "fpol_1"
        assert policy.path == "/data"
        assert policy.recursive is True

    def test_file_policy_to_fuse_config(self):
        """Test converting to FUSE config."""
        policy = FilePolicy(
            policy_id="fpol_1",
            path="/protected",
            operations=["write", "delete"],
            effect="deny",
            recursive=True,
        )

        config = policy.to_fuse_config()
        assert config["path"] == "/protected"
        assert config["recursive"] is True


class TestPolicyCompiler:
    """Test PolicyCompiler class."""

    def test_compiler_creation(self):
        """Test creating a policy compiler."""
        compiler = PolicyCompiler()
        assert compiler is not None

    def test_compile_file_rule(self):
        """Test compiling a file rule."""
        compiler = PolicyCompiler()

        rule = Rule(
            rule_id="r_test",
            scope=RuleScope.FOLDER,
            target="/data",
            effect=RuleEffect.DENY,
            actions=[RuleAction.WRITE],
            reason="Read-only",
        )

        policies = compiler.compile_rule(rule)
        assert len(policies) > 0

        file_policies = [p for p in policies if isinstance(p, FilePolicy)]
        assert len(file_policies) > 0

    def test_compile_syscall_rule(self):
        """Test compiling a syscall rule."""
        compiler = PolicyCompiler()

        rule = Rule(
            rule_id="r_test",
            scope=RuleScope.PROCESS,
            target="sandbox",
            effect=RuleEffect.DENY,
            actions=[RuleAction.NETWORK],
            reason="No network",
        )

        policies = compiler.compile_rule(rule)
        syscall_policies = [p for p in policies if isinstance(p, SyscallPolicy)]
        assert len(syscall_policies) > 0


# ============================================================================
# Interpreter Tests
# ============================================================================


class TestIntentParser:
    """Test IntentParser class."""

    def test_parser_creation(self):
        """Test creating an intent parser."""
        parser = IntentParser()
        assert parser is not None

    def test_parse_protect_folder(self):
        """Test parsing a protect folder request."""
        parser = IntentParser()
        intent = parser.parse(
            "Protect /home/user/private from all access",
            {"cwd": "/home/user"},
        )

        assert intent.action == IntentAction.SET_RULE
        assert intent.effect == RuleEffect.DENY
        assert "/home/user/private" in intent.target

    def test_parse_allow_read(self):
        """Test parsing an allow read request."""
        parser = IntentParser()
        intent = parser.parse(
            "Allow reading files in /public",
            {"cwd": "/"},
        )

        assert intent.action == IntentAction.SET_RULE
        assert intent.effect == RuleEffect.ALLOW
        assert RuleAction.READ in intent.rule_actions

    def test_parse_query_rules(self):
        """Test parsing a rule query."""
        parser = IntentParser()
        intent = parser.parse("What rules apply to /data?")

        assert intent.action == IntentAction.QUERY_RULES

    def test_parse_ai_restriction(self):
        """Test parsing AI restriction request."""
        parser = IntentParser()
        intent = parser.parse(
            "Never let AI read or index /secrets",
            {"cwd": "/"},
        )

        assert intent.effect == RuleEffect.DENY
        assert any(a in intent.rule_actions for a in [RuleAction.AI_READ, RuleAction.READ])


class TestParsedIntent:
    """Test ParsedIntent class."""

    def test_intent_creation(self):
        """Test creating a parsed intent."""
        intent = ParsedIntent(
            action=IntentAction.SET_RULE,
            target="/data",
            effect=RuleEffect.DENY,
            rule_actions=[RuleAction.DELETE],
        )

        assert intent.action == IntentAction.SET_RULE
        assert intent.target == "/data"

    def test_intent_is_ambiguous(self):
        """Test ambiguity detection."""
        intent = ParsedIntent(
            action=IntentAction.SET_RULE,
            ambiguities=["Which folder should this apply to?"],
        )

        assert intent.is_ambiguous() is True

        intent2 = ParsedIntent(
            action=IntentAction.SET_RULE,
            target="/data",
            effect=RuleEffect.DENY,
            rule_actions=[RuleAction.WRITE],
            confidence=0.95,
        )

        assert intent2.is_ambiguous() is False

    def test_intent_to_rule(self):
        """Test converting intent to rule."""
        intent = ParsedIntent(
            action=IntentAction.SET_RULE,
            target="/data",
            effect=RuleEffect.DENY,
            rule_actions=[RuleAction.WRITE],
            scope=RuleScope.FOLDER,
            reason="Test rule",
        )

        rule = intent.to_rule("r_test")
        assert rule.rule_id == "r_test"
        assert rule.target == "/data"
        assert rule.effect == RuleEffect.DENY


# ============================================================================
# FUSE Tests
# ============================================================================


class TestFuseConfig:
    """Test FuseConfig class."""

    def test_fuse_config_creation(self):
        """Test creating FUSE config."""
        config = FuseConfig(
            source_path=Path("/data"),
            mount_point=Path("/mnt/data"),
            read_only=True,
        )

        assert config.source_path == Path("/data")
        assert config.mount_point == Path("/mnt/data")
        assert config.read_only is True

    def test_fuse_options(self):
        """Test FUSE mount options."""
        config = FuseConfig(
            source_path=Path("/data"),
            mount_point=Path("/mnt/data"),
            read_only=True,
            allow_other=True,
        )

        options = config.to_fuse_options()
        assert "ro" in options
        assert "allow_other" in options


class TestFuseWrapper:
    """Test FuseWrapper class."""

    def test_wrapper_creation(self):
        """Test creating FUSE wrapper."""
        registry = RuleRegistry()
        wrapper = FuseWrapper(registry)
        assert wrapper is not None

    def test_check_access_allowed(self):
        """Test checking allowed access."""
        registry = RuleRegistry()
        wrapper = FuseWrapper(registry)

        # No rules = allowed
        allowed, rule = wrapper.check_access("/data/file.txt", FuseOperation.READ)
        assert allowed is True
        assert rule is None

    def test_check_access_denied(self):
        """Test checking denied access."""
        registry = RuleRegistry()

        rule = Rule(
            rule_id="r_test",
            scope=RuleScope.FOLDER,
            target="/private",
            effect=RuleEffect.DENY,
            actions=[RuleAction.READ],
            reason="Private",
        )
        registry.add_rule(rule)

        wrapper = FuseWrapper(registry)

        allowed, matched = wrapper.check_access("/private/secret.txt", FuseOperation.READ)
        assert allowed is False
        assert matched is not None


# ============================================================================
# eBPF Tests
# ============================================================================


class TestSeccompFilter:
    """Test SeccompFilter class."""

    def test_filter_creation(self):
        """Test creating seccomp filter."""
        filter_obj = SeccompFilter(
            filter_id="test_filter",
            default_action=SeccompAction.ALLOW,
        )

        assert filter_obj.filter_id == "test_filter"
        assert filter_obj.default_action == SeccompAction.ALLOW

    def test_add_syscall(self):
        """Test adding syscall rule."""
        filter_obj = SeccompFilter(filter_id="test")
        filter_obj.add_syscall("execve", SeccompAction.DENY)

        assert len(filter_obj.syscalls) == 1
        assert filter_obj.syscalls[0].syscall == "execve"

    def test_to_libseccomp_format(self):
        """Test libseccomp format export."""
        filter_obj = SeccompFilter(filter_id="test")
        filter_obj.add_syscall("ptrace", SeccompAction.DENY)

        config = filter_obj.to_libseccomp_format()
        assert "defaultAction" in config
        assert "syscalls" in config


class TestEbpfFilter:
    """Test EbpfFilter class."""

    def test_filter_creation(self):
        """Test creating eBPF filter."""
        ebpf = EbpfFilter()
        assert ebpf is not None

    def test_create_deny_filter(self):
        """Test creating a deny filter."""
        ebpf = EbpfFilter()
        filter_obj = ebpf.create_deny_filter(
            "no_network",
            ["socket", "connect", "bind"],
        )

        assert len(filter_obj.syscalls) == 3

    def test_create_sandbox_filter(self):
        """Test creating sandbox filter."""
        ebpf = EbpfFilter()
        filter_obj = ebpf.create_sandbox_filter("sandbox")

        assert len(filter_obj.syscalls) > 0


# ============================================================================
# Monitor Tests
# ============================================================================


class TestEventType:
    """Test EventType enum."""

    def test_event_type_flags(self):
        """Test event type flags."""
        events = EventType.CREATE | EventType.DELETE
        assert EventType.CREATE in events
        assert EventType.DELETE in events
        assert EventType.MODIFY not in events

    def test_all_events(self):
        """Test ALL_EVENTS combination."""
        assert EventType.CREATE in EventType.ALL_EVENTS
        assert EventType.DELETE in EventType.ALL_EVENTS
        assert EventType.MODIFY in EventType.ALL_EVENTS


class TestMonitorEvent:
    """Test MonitorEvent class."""

    def test_event_creation(self):
        """Test creating monitor event."""
        event = MonitorEvent(
            event_id="evt_1",
            event_type=EventType.CREATE,
            path="/data",
            filename="test.txt",
        )

        assert event.event_id == "evt_1"
        assert event.event_type == EventType.CREATE
        assert event.full_path == "/data/test.txt"

    def test_event_to_dict(self):
        """Test event serialization."""
        event = MonitorEvent(
            event_id="evt_1",
            event_type=EventType.MODIFY,
            path="/data",
            filename="file.txt",
        )

        data = event.to_dict()
        assert data["event_id"] == "evt_1"
        assert "full_path" in data


class TestAuditLog:
    """Test AuditLog class."""

    def test_audit_log_creation(self):
        """Test creating audit log."""
        log = AuditLog()
        assert log is not None

    def test_log_entry(self):
        """Test logging an entry."""
        log = AuditLog()

        entry = AuditEntry(
            entry_id="audit_1",
            timestamp=datetime.now(),
            event_type="access",
            path="/data/file.txt",
            action="denied",
            rule_id="r_test",
        )

        log.log(entry)

        entries = log.query(path="/data")
        assert len(entries) == 1
        assert entries[0].action == "denied"

    def test_query_with_filters(self):
        """Test querying with filters."""
        log = AuditLog()

        for i in range(5):
            entry = AuditEntry(
                entry_id=f"audit_{i}",
                timestamp=datetime.now(),
                event_type="access",
                path=f"/data/file{i}.txt",
                action="allowed" if i % 2 == 0 else "denied",
            )
            log.log(entry)

        denied = log.query(action="denied")
        assert len(denied) == 2


class TestFileMonitor:
    """Test FileMonitor class."""

    def test_monitor_creation(self):
        """Test creating file monitor."""
        monitor = FileMonitor()
        assert monitor is not None

    def test_add_watch(self):
        """Test adding a watch."""
        monitor = FileMonitor()

        with tempfile.TemporaryDirectory() as tmpdir:
            result = monitor.add_watch(tmpdir)
            assert result is True

            watches = monitor.list_watches()
            assert len(watches) == 1

    def test_simulate_event(self):
        """Test simulating an event."""
        monitor = FileMonitor()
        events_received = []

        def handler(event):
            events_received.append(event)

        monitor.on_event(handler)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "test.txt")
            event = monitor.simulate_event(filepath, EventType.CREATE)

            assert event is not None
            assert len(events_received) == 1


# ============================================================================
# Context Tests
# ============================================================================


class TestUserContext:
    """Test UserContext class."""

    def test_user_context_creation(self):
        """Test creating user context."""
        ctx = UserContext(
            user_id="user_1",
            username="testuser",
            home_directory="/home/testuser",
        )

        assert ctx.user_id == "user_1"
        assert ctx.username == "testuser"

    def test_user_context_to_dict(self):
        """Test user context serialization."""
        ctx = UserContext(
            user_id="user_1",
            username="testuser",
            groups=["users", "developers"],
        )

        data = ctx.to_dict()
        assert data["username"] == "testuser"
        assert "developers" in data["groups"]


class TestFolderContext:
    """Test FolderContext class."""

    def test_folder_context_creation(self):
        """Test creating folder context."""
        ctx = FolderContext(
            path="/data/sensitive",
            sensitivity="confidential",
            tags=["pii", "financial"],
        )

        assert ctx.path == "/data/sensitive"
        assert ctx.sensitivity == "confidential"

    def test_folder_context_to_dict(self):
        """Test folder context serialization."""
        ctx = FolderContext(
            path="/data",
            rule_ids=["r_1", "r_2"],
        )

        data = ctx.to_dict()
        assert "r_1" in data["rule_ids"]


class TestContextMemory:
    """Test ContextMemory class."""

    def test_context_memory_creation(self):
        """Test creating context memory."""
        mem = ContextMemory()
        assert mem is not None

    def test_set_get_user(self):
        """Test setting and getting user context."""
        mem = ContextMemory()

        user = UserContext(
            user_id="user_1",
            username="testuser",
        )
        mem.set_user(user)

        retrieved = mem.get_user("user_1")
        assert retrieved is not None
        assert retrieved.username == "testuser"

    def test_set_get_folder(self):
        """Test setting and getting folder context."""
        mem = ContextMemory()

        folder = FolderContext(
            path="/data",
            sensitivity="sensitive",
        )
        mem.set_folder(folder)

        retrieved = mem.get_folder("/data")
        assert retrieved is not None
        assert retrieved.sensitivity == "sensitive"

    def test_inherited_context(self):
        """Test getting inherited context."""
        mem = ContextMemory()

        parent = FolderContext(
            path="/data",
            sensitivity="sensitive",
        )
        mem.set_folder(parent)

        child = FolderContext(
            path="/data/subdir",
            sensitivity="normal",
        )
        mem.set_folder(child)

        inherited = mem.get_inherited_context("/data/subdir/file.txt")
        assert len(inherited) > 0
        assert any(f.path == "/data" for f in inherited)


# ============================================================================
# Engine Tests
# ============================================================================


class TestKernelConfig:
    """Test KernelConfig class."""

    def test_config_defaults(self):
        """Test default configuration."""
        config = KernelConfig()
        assert config.enable_fuse is True
        assert config.enable_monitoring is True
        assert config.default_effect == RuleEffect.ALLOW


class TestConversationalKernel:
    """Test ConversationalKernel class."""

    def test_kernel_creation(self):
        """Test creating the kernel."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = KernelConfig(data_dir=Path(tmpdir))
            kernel = ConversationalKernel(config)

            assert kernel is not None
            assert kernel.state == KernelState.READY
            kernel.shutdown()

    def test_process_protect_request(self):
        """Test processing a protect request."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = KernelConfig(data_dir=Path(tmpdir))
            kernel = ConversationalKernel(config)

            result = kernel.process_request(
                "Protect /home/user/private from all access",
                user_id="test_user",
                context={"cwd": "/home/user"},
            )

            # May need clarification or create rule
            assert result is not None
            kernel.shutdown()

    def test_process_query_request(self):
        """Test processing a query request."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = KernelConfig(data_dir=Path(tmpdir))
            kernel = ConversationalKernel(config)

            # Add a rule first
            rule = Rule(
                rule_id="r_test",
                scope=RuleScope.FOLDER,
                target="/data",
                effect=RuleEffect.DENY,
                actions=[RuleAction.WRITE],
                reason="Read-only",
            )
            kernel.rule_registry.add_rule(rule)

            result = kernel.process_request(
                "What rules apply to /data?",
                context={"cwd": "/"},
            )

            assert result["success"] is True
            assert "rules" in result or "intent" in result
            kernel.shutdown()

    def test_get_status(self):
        """Test getting kernel status."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = KernelConfig(data_dir=Path(tmpdir))
            kernel = ConversationalKernel(config)

            status = kernel.get_status()
            assert "state" in status
            assert "rules_count" in status
            kernel.shutdown()

    def test_export_configuration(self):
        """Test exporting configuration."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = KernelConfig(data_dir=Path(tmpdir))
            kernel = ConversationalKernel(config)

            export = kernel.export_configuration()
            assert "rules" in export
            assert "context" in export
            kernel.shutdown()


# ============================================================================
# Integration Tests
# ============================================================================


class TestIntegration:
    """Integration tests for the kernel module."""

    def test_full_rule_flow(self):
        """Test complete rule creation and enforcement flow."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = KernelConfig(data_dir=Path(tmpdir))
            kernel = ConversationalKernel(config)

            # Create a rule via natural language
            result = kernel.process_request(
                "Block write access to /data/readonly",
                user_id="admin",
                context={"cwd": "/"},
            )

            # Check if rule was created or needs clarification
            if result.get("needs_clarification"):
                # Handle clarification
                assert "clarification_questions" in result
            else:
                # Rule should be created
                assert result.get("success") or result.get("rule_id")

            kernel.shutdown()

    def test_policy_compilation_flow(self):
        """Test rule to policy compilation."""
        compiler = PolicyCompiler()
        registry = RuleRegistry()

        # Create multiple rules
        rules = [
            Rule(
                rule_id="r1",
                scope=RuleScope.FOLDER,
                target="/data",
                effect=RuleEffect.DENY,
                actions=[RuleAction.DELETE],
                reason="No deletion",
            ),
            Rule(
                rule_id="r2",
                scope=RuleScope.SYSTEM,
                target="*",
                effect=RuleEffect.DENY,
                actions=[RuleAction.NETWORK],
                reason="No network",
            ),
        ]

        for rule in rules:
            registry.add_rule(rule)

        # Compile all rules
        policies = compiler.compile_rules(registry.list_rules())

        assert PolicyType.FILE in policies or PolicyType.SYSCALL in policies

    def test_monitoring_flow(self):
        """Test monitoring and audit flow."""
        audit_log = AuditLog()
        monitor = FileMonitor(audit_log=audit_log)

        events = []

        def event_handler(event):
            events.append(event)

        monitor.on_event(event_handler)

        with tempfile.TemporaryDirectory() as tmpdir:
            monitor.add_watch(tmpdir)

            # Simulate events
            filepath = os.path.join(tmpdir, "test.txt")
            monitor.simulate_event(filepath, EventType.CREATE)
            monitor.simulate_event(filepath, EventType.MODIFY)

            assert len(events) == 2

            # Check audit log
            entries = audit_log.query()
            assert len(entries) == 2


# ============================================================================
# Module Exports Tests
# ============================================================================


class TestModuleExports:
    """Test module exports."""

    def test_rules_exports(self):
        """Test rules module exports."""
        from src.kernel import Rule, RuleAction, RuleEffect, RuleRegistry, RuleScope

        assert Rule is not None
        assert RuleAction is not None
        assert RuleEffect is not None
        assert RuleScope is not None
        assert RuleRegistry is not None

    def test_policy_exports(self):
        """Test policy module exports."""
        from src.kernel import AccessPolicy, FilePolicy, PolicyCompiler, SyscallPolicy

        assert FilePolicy is not None
        assert SyscallPolicy is not None
        assert AccessPolicy is not None
        assert PolicyCompiler is not None

    def test_interpreter_exports(self):
        """Test interpreter module exports."""
        from src.kernel import IntentAction, IntentParser, ParsedIntent, PolicyInterpreter

        assert IntentParser is not None
        assert ParsedIntent is not None
        assert IntentAction is not None
        assert PolicyInterpreter is not None

    def test_engine_exports(self):
        """Test engine module exports."""
        from src.kernel import ConversationalKernel, KernelConfig, KernelState

        assert ConversationalKernel is not None
        assert KernelConfig is not None
        assert KernelState is not None
