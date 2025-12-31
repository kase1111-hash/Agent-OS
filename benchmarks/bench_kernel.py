"""
Kernel Engine Benchmarks

Performance benchmarks for the conversational kernel engine.
Tests intent parsing, rule matching, and policy evaluation.

Target: Sub-2-second orchestration overhead as per roadmap.
"""

from pathlib import Path
from typing import List

import pytest

from src.kernel.engine import ConversationalKernel, KernelConfig
from src.kernel.interpreter import IntentParser, ParsedIntent
from src.kernel.rules import Rule, RuleAction, RuleEffect, RuleRegistry, RuleScope
from src.kernel.policy import Policy, PolicyCompiler, PolicyType


class TestIntentParserBenchmarks:
    """Benchmarks for intent parsing."""

    @pytest.mark.kernel
    def test_parse_simple_intent(self, benchmark, temp_dir: Path) -> None:
        """Benchmark parsing a simple user intent.

        Target: <10ms per intent parse.
        """
        parser = IntentParser()
        query = "Show me the files in my documents folder"

        result = benchmark(parser.parse, query)

        assert result is not None

    @pytest.mark.kernel
    def test_parse_complex_intent(self, benchmark, temp_dir: Path) -> None:
        """Benchmark parsing a complex multi-part intent.

        Target: <50ms for complex intents.
        """
        parser = IntentParser()
        query = (
            "I want to create a new project called 'MyApp', "
            "set up the directory structure with src, tests, and docs folders, "
            "initialize a git repository, and create a README with project description"
        )

        result = benchmark(parser.parse, query)

        assert result is not None

    @pytest.mark.kernel
    def test_parse_many_intents(self, benchmark) -> None:
        """Benchmark parsing many intents sequentially.

        Target: 100 parses in <1 second.
        """
        parser = IntentParser()
        queries = [
            "Open the settings panel",
            "Search for documents about Python",
            "Create a new file called test.py",
            "Delete the temporary folder",
            "Show system status",
            "Run the build script",
            "Check for updates",
            "Export data to CSV",
            "Import configuration from file",
            "Restart the service",
        ] * 10  # 100 queries

        def parse_all() -> int:
            results = []
            for query in queries:
                results.append(parser.parse(query))
            return len(results)

        result = benchmark(parse_all)
        assert result == 100


class TestRuleRegistryBenchmarks:
    """Benchmarks for rule registry operations."""

    @pytest.mark.kernel
    def test_rule_lookup(self, benchmark, temp_dir: Path) -> None:
        """Benchmark rule lookup by ID.

        Target: <1ms per lookup.
        """
        registry = RuleRegistry(db_path=temp_dir / "rules.db")

        # Add rules
        for i in range(100):
            rule = Rule(
                id=f"rule-{i:04d}",
                name=f"Test Rule {i}",
                description=f"Description for rule {i}",
                scope=RuleScope.GLOBAL,
                action=RuleAction.ALLOW,
                effect=RuleEffect.ALLOW,
                pattern=f"pattern-{i}",
            )
            registry.add(rule)

        # Benchmark lookup
        result = benchmark(registry.get, "rule-0050")

        assert result is not None
        assert result.id == "rule-0050"

    @pytest.mark.kernel
    def test_rule_match(self, benchmark, temp_dir: Path) -> None:
        """Benchmark rule pattern matching.

        Target: <5ms to match against 100 rules.
        """
        registry = RuleRegistry(db_path=temp_dir / "rules.db")

        # Add rules with various patterns
        patterns = [
            "file:read:*",
            "file:write:documents/*",
            "network:connect:*",
            "process:spawn:*",
            "memory:allocate:*",
        ]

        for i in range(100):
            rule = Rule(
                id=f"rule-{i:04d}",
                name=f"Test Rule {i}",
                description=f"Description for rule {i}",
                scope=RuleScope.GLOBAL,
                action=RuleAction.ALLOW,
                effect=RuleEffect.ALLOW,
                pattern=patterns[i % len(patterns)],
            )
            registry.add(rule)

        # Benchmark matching
        result = benchmark(registry.match, "file:read:config.yaml")

        assert isinstance(result, list)

    @pytest.mark.kernel
    def test_rule_bulk_add(self, benchmark, temp_dir: Path) -> None:
        """Benchmark adding many rules.

        Target: 1000 rules added in <500ms.
        """

        def add_rules() -> int:
            registry = RuleRegistry(db_path=temp_dir / "rules_bulk.db")
            for i in range(1000):
                rule = Rule(
                    id=f"bulk-rule-{i:04d}",
                    name=f"Bulk Rule {i}",
                    description=f"Bulk description {i}",
                    scope=RuleScope.GLOBAL,
                    action=RuleAction.ALLOW,
                    effect=RuleEffect.ALLOW,
                    pattern=f"bulk:pattern:{i}",
                )
                registry.add(rule)
            return len(registry.get_all())

        result = benchmark(add_rules)
        assert result == 1000


class TestPolicyBenchmarks:
    """Benchmarks for policy compilation and evaluation."""

    @pytest.mark.kernel
    def test_policy_compile(self, benchmark) -> None:
        """Benchmark policy compilation.

        Target: <10ms per policy compilation.
        """
        compiler = PolicyCompiler()
        rules = [
            Rule(
                id=f"policy-rule-{i}",
                name=f"Policy Rule {i}",
                description=f"Policy description {i}",
                scope=RuleScope.GLOBAL,
                action=RuleAction.ALLOW if i % 2 == 0 else RuleAction.DENY,
                effect=RuleEffect.ALLOW if i % 2 == 0 else RuleEffect.DENY,
                pattern=f"action:{i}:*",
            )
            for i in range(50)
        ]

        result = benchmark(compiler.compile, rules)

        assert result is not None

    @pytest.mark.kernel
    def test_policy_evaluate(self, benchmark) -> None:
        """Benchmark policy evaluation.

        Target: <5ms per evaluation.
        """
        compiler = PolicyCompiler()
        rules = [
            Rule(
                id=f"eval-rule-{i}",
                name=f"Eval Rule {i}",
                description=f"Eval description {i}",
                scope=RuleScope.GLOBAL,
                action=RuleAction.ALLOW,
                effect=RuleEffect.ALLOW,
                pattern=f"resource:{i % 10}:*",
            )
            for i in range(100)
        ]
        policy = compiler.compile(rules)

        # Create a request context
        context = {
            "resource": "resource:5:file.txt",
            "action": "read",
            "user": "test-user",
        }

        result = benchmark(policy.evaluate, context)

        assert result is not None


class TestKernelBenchmarks:
    """End-to-end kernel benchmarks."""

    @pytest.mark.kernel
    @pytest.mark.integration
    def test_kernel_initialization(self, benchmark, temp_dir: Path) -> None:
        """Benchmark kernel initialization.

        Target: <500ms for cold start.
        """
        config = KernelConfig(
            data_dir=temp_dir / "kernel",
            enable_fuse=False,  # Disable for benchmark
            enable_ebpf=False,
            enable_monitoring=False,
        )

        result = benchmark(ConversationalKernel, config)

        assert result is not None
        assert result.state.value == "ready"

    @pytest.mark.kernel
    @pytest.mark.integration
    def test_kernel_process_request(self, benchmark, temp_dir: Path) -> None:
        """Benchmark processing a request through the kernel.

        Target: <2 seconds total orchestration overhead.
        """
        config = KernelConfig(
            data_dir=temp_dir / "kernel",
            enable_fuse=False,
            enable_ebpf=False,
            enable_monitoring=False,
        )
        kernel = ConversationalKernel(config)

        request = {
            "intent": "read_file",
            "resource": "/home/user/document.txt",
            "user": "test-user",
        }

        result = benchmark(kernel.process_request, request)

        assert result is not None
