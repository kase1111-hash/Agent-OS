"""
Agent Test Runner

Provides test execution and reporting for agent tests.
"""

import inspect
import logging
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Type

from .fixtures import AgentTestCase


logger = logging.getLogger(__name__)


@dataclass
class TestResult:
    """Result of a single test."""
    name: str
    passed: bool
    duration_ms: int = 0
    error: Optional[str] = None
    traceback: Optional[str] = None
    output: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "passed": self.passed,
            "duration_ms": self.duration_ms,
            "error": self.error,
            "output": self.output,
        }


@dataclass
class TestSuiteResult:
    """Result of a test suite."""
    suite_name: str
    results: List[TestResult] = field(default_factory=list)
    total_duration_ms: int = 0
    started_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None

    @property
    def passed_count(self) -> int:
        return sum(1 for r in self.results if r.passed)

    @property
    def failed_count(self) -> int:
        return sum(1 for r in self.results if not r.passed)

    @property
    def total_count(self) -> int:
        return len(self.results)

    @property
    def all_passed(self) -> bool:
        return all(r.passed for r in self.results)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "suite_name": self.suite_name,
            "results": [r.to_dict() for r in self.results],
            "passed_count": self.passed_count,
            "failed_count": self.failed_count,
            "total_count": self.total_count,
            "total_duration_ms": self.total_duration_ms,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
        }


class TestSuite:
    """
    A collection of related tests.

    Example:
        suite = TestSuite("MyAgent Tests")
        suite.add_test("test_greeting", test_greeting)
        suite.add_test("test_farewell", test_farewell)
    """

    def __init__(self, name: str):
        self.name = name
        self._tests: List[tuple[str, Callable]] = []
        self._setup: Optional[Callable] = None
        self._teardown: Optional[Callable] = None

    def add_test(
        self,
        name: str,
        test_fn: Callable[[], None],
    ) -> "TestSuite":
        """Add a test function."""
        self._tests.append((name, test_fn))
        return self

    def set_setup(self, fn: Callable[[], None]) -> "TestSuite":
        """Set suite setup function."""
        self._setup = fn
        return self

    def set_teardown(self, fn: Callable[[], None]) -> "TestSuite":
        """Set suite teardown function."""
        self._teardown = fn
        return self

    def from_class(
        self,
        test_class: Type[AgentTestCase],
    ) -> "TestSuite":
        """
        Create suite from test class.

        All methods starting with 'test_' are added as tests.
        """
        instance = test_class()
        self._setup = instance.setup
        self._teardown = instance.teardown

        for name in dir(instance):
            if name.startswith("test_"):
                method = getattr(instance, name)
                if callable(method):
                    self._tests.append((name, method))

        return self

    def run(self, verbose: bool = True) -> TestSuiteResult:
        """Run all tests in the suite."""
        result = TestSuiteResult(suite_name=self.name)

        if verbose:
            print(f"\n{'='*60}")
            print(f"Running: {self.name}")
            print(f"{'='*60}")

        # Suite setup
        if self._setup:
            try:
                self._setup()
            except Exception as e:
                logger.error(f"Suite setup failed: {e}")

        suite_start = time.time()

        for test_name, test_fn in self._tests:
            test_result = self._run_single_test(test_name, test_fn, verbose)
            result.results.append(test_result)

        result.total_duration_ms = int((time.time() - suite_start) * 1000)
        result.completed_at = datetime.now()

        # Suite teardown
        if self._teardown:
            try:
                self._teardown()
            except Exception as e:
                logger.error(f"Suite teardown failed: {e}")

        if verbose:
            self._print_summary(result)

        return result

    def _run_single_test(
        self,
        name: str,
        test_fn: Callable,
        verbose: bool,
    ) -> TestResult:
        """Run a single test."""
        start = time.time()

        try:
            test_fn()
            duration = int((time.time() - start) * 1000)

            if verbose:
                print(f"  ✓ {name} ({duration}ms)")

            return TestResult(
                name=name,
                passed=True,
                duration_ms=duration,
            )

        except Exception as e:
            duration = int((time.time() - start) * 1000)

            if verbose:
                print(f"  ✗ {name} ({duration}ms)")
                print(f"    Error: {e}")

            return TestResult(
                name=name,
                passed=False,
                duration_ms=duration,
                error=str(e),
                traceback=traceback.format_exc(),
            )

    def _print_summary(self, result: TestSuiteResult) -> None:
        """Print test summary."""
        print(f"\n{'-'*60}")
        print(f"Results: {result.passed_count}/{result.total_count} passed")
        print(f"Duration: {result.total_duration_ms}ms")

        if result.failed_count > 0:
            print(f"\nFailed tests:")
            for r in result.results:
                if not r.passed:
                    print(f"  - {r.name}: {r.error}")


class AgentTestRunner:
    """
    Test runner for agent tests.

    Supports running multiple suites and generating reports.
    """

    def __init__(self):
        self._suites: List[TestSuite] = []
        self._results: List[TestSuiteResult] = []

    def add_suite(self, suite: TestSuite) -> "AgentTestRunner":
        """Add a test suite."""
        self._suites.append(suite)
        return self

    def add_test_class(
        self,
        test_class: Type[AgentTestCase],
        suite_name: Optional[str] = None,
    ) -> "AgentTestRunner":
        """Add a test class as a suite."""
        name = suite_name or test_class.__name__
        suite = TestSuite(name).from_class(test_class)
        self._suites.append(suite)
        return self

    def run(self, verbose: bool = True) -> List[TestSuiteResult]:
        """Run all test suites."""
        self._results = []

        for suite in self._suites:
            result = suite.run(verbose=verbose)
            self._results.append(result)

        if verbose:
            self._print_final_summary()

        return self._results

    def _print_final_summary(self) -> None:
        """Print final summary across all suites."""
        total_passed = sum(r.passed_count for r in self._results)
        total_failed = sum(r.failed_count for r in self._results)
        total_tests = total_passed + total_failed
        total_time = sum(r.total_duration_ms for r in self._results)

        print(f"\n{'='*60}")
        print("FINAL SUMMARY")
        print(f"{'='*60}")
        print(f"Suites: {len(self._results)}")
        print(f"Tests:  {total_passed}/{total_tests} passed")
        print(f"Time:   {total_time}ms")

        if total_failed > 0:
            print(f"\nFAILED ({total_failed} tests)")
        else:
            print("\nALL PASSED")

    def get_report(self) -> Dict[str, Any]:
        """Get test report as dictionary."""
        return {
            "suites": [r.to_dict() for r in self._results],
            "total_suites": len(self._results),
            "total_tests": sum(r.total_count for r in self._results),
            "total_passed": sum(r.passed_count for r in self._results),
            "total_failed": sum(r.failed_count for r in self._results),
            "all_passed": all(r.all_passed for r in self._results),
        }


def run_agent_tests(
    *test_classes: Type[AgentTestCase],
    verbose: bool = True,
) -> List[TestSuiteResult]:
    """
    Run agent test classes.

    Convenience function for running multiple test classes.

    Example:
        results = run_agent_tests(
            TestMyAgent,
            TestAnotherAgent,
            verbose=True,
        )

    Args:
        *test_classes: Test class types
        verbose: Print progress and results

    Returns:
        List of test suite results
    """
    runner = AgentTestRunner()

    for test_class in test_classes:
        runner.add_test_class(test_class)

    return runner.run(verbose=verbose)
