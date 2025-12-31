"""
Core Component Benchmarks

Performance benchmarks for constitution parsing, validation, and core models.
These benchmarks establish baseline performance for the constitutional layer.

Target: Constitution parsing should complete in <100ms for typical documents.
"""

from pathlib import Path

import pytest

from src.core.parser import ConstitutionParser
from src.core.validator import ConstitutionValidator
from src.core.models import Rule, RuleType, AuthorityLevel


class TestParserBenchmarks:
    """Benchmarks for ConstitutionParser."""

    @pytest.mark.core
    def test_parse_constitution_small(self, benchmark, sample_constitution: str) -> None:
        """Benchmark parsing a typical constitution document.

        Target: <50ms for a standard constitution (~2KB).
        """
        parser = ConstitutionParser()

        result = benchmark(parser.parse_content, sample_constitution)

        assert result is not None
        assert len(result.rules) > 0

    @pytest.mark.core
    def test_parse_constitution_large(self, benchmark, large_constitution: str) -> None:
        """Benchmark parsing a large constitution document.

        Target: <500ms for 10x standard size (~20KB).
        """
        parser = ConstitutionParser()

        result = benchmark(parser.parse_content, large_constitution)

        assert result is not None
        assert len(result.rules) > 0

    @pytest.mark.core
    def test_parse_constitution_from_file(
        self, benchmark, constitution_file: Path
    ) -> None:
        """Benchmark parsing constitution from file (includes I/O).

        Target: <100ms including file I/O.
        """
        parser = ConstitutionParser()

        result = benchmark(parser.parse_file, constitution_file)

        assert result is not None

    @pytest.mark.core
    def test_parse_many_constitutions(
        self, benchmark, sample_constitution: str
    ) -> None:
        """Benchmark parsing multiple constitutions sequentially.

        Tests parser reuse and caching efficiency.
        Target: 100 parses in <2 seconds.
        """
        parser = ConstitutionParser()

        def parse_many() -> int:
            count = 0
            for _ in range(100):
                result = parser.parse_content(sample_constitution)
                count += len(result.rules)
            return count

        result = benchmark(parse_many)
        assert result > 0


class TestValidatorBenchmarks:
    """Benchmarks for ConstitutionValidator."""

    @pytest.mark.core
    def test_validate_constitution(
        self, benchmark, sample_constitution: str
    ) -> None:
        """Benchmark constitution validation.

        Target: <20ms for validation of parsed constitution.
        """
        parser = ConstitutionParser()
        constitution = parser.parse_content(sample_constitution)
        validator = ConstitutionValidator()

        result = benchmark(validator.validate, constitution)

        assert result is not None
        assert result.is_valid

    @pytest.mark.core
    def test_validate_rule(self, benchmark) -> None:
        """Benchmark individual rule validation.

        Target: <1ms per rule validation.
        """
        validator = ConstitutionValidator()
        rule = Rule(
            id="test-rule-001",
            text="Agents shall respond within 2 seconds",
            rule_type=RuleType.MANDATE,
            authority=AuthorityLevel.CONSTITUTIONAL,
            section="2.1",
            keywords=["respond", "seconds"],
            is_immutable=False,
        )

        result = benchmark(validator.validate_rule, rule)

        assert result is not None

    @pytest.mark.core
    def test_validate_many_rules(self, benchmark) -> None:
        """Benchmark validating many rules sequentially.

        Target: 1000 rules in <500ms.
        """
        validator = ConstitutionValidator()
        rules = [
            Rule(
                id=f"test-rule-{i:04d}",
                text=f"Test rule number {i} with some content",
                rule_type=RuleType.MANDATE if i % 3 == 0 else RuleType.PROHIBITION,
                authority=AuthorityLevel.SYSTEM,
                section=f"{(i // 10) + 1}.{(i % 10) + 1}",
                keywords=["test", f"rule{i}"],
                is_immutable=i % 5 == 0,
            )
            for i in range(1000)
        ]

        def validate_all() -> int:
            valid_count = 0
            for rule in rules:
                result = validator.validate_rule(rule)
                if result.is_valid:
                    valid_count += 1
            return valid_count

        result = benchmark(validate_all)
        assert result == 1000


class TestModelBenchmarks:
    """Benchmarks for core data models."""

    @pytest.mark.core
    def test_rule_creation(self, benchmark) -> None:
        """Benchmark Rule model instantiation.

        Target: 10000 rules created in <100ms.
        """

        def create_rules() -> int:
            rules = []
            for i in range(10000):
                rule = Rule(
                    id=f"rule-{i}",
                    text=f"Rule text {i}",
                    rule_type=RuleType.MANDATE,
                    authority=AuthorityLevel.SYSTEM,
                    section="1.1",
                    keywords=["test"],
                    is_immutable=False,
                )
                rules.append(rule)
            return len(rules)

        result = benchmark(create_rules)
        assert result == 10000

    @pytest.mark.core
    def test_rule_serialization(self, benchmark) -> None:
        """Benchmark Rule serialization to dict.

        Target: 1000 serializations in <50ms.
        """
        rules = [
            Rule(
                id=f"rule-{i}",
                text=f"Rule text {i}",
                rule_type=RuleType.MANDATE,
                authority=AuthorityLevel.SYSTEM,
                section="1.1",
                keywords=["test", "benchmark"],
                is_immutable=False,
            )
            for i in range(1000)
        ]

        def serialize_all() -> int:
            dicts = []
            for rule in rules:
                dicts.append(rule.to_dict())
            return len(dicts)

        result = benchmark(serialize_all)
        assert result == 1000
