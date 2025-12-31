"""
Agent OS Flow Controller

Manages sequential and parallel execution of multi-agent workflows.
Coordinates agent invocations based on routing decisions.
"""

import asyncio
import logging
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Union

from .router import AgentRoute, RoutingDecision, RoutingStrategy

logger = logging.getLogger(__name__)


class FlowStatus(Enum):
    """Status of a flow execution."""

    PENDING = auto()
    RUNNING = auto()
    COMPLETED = auto()
    PARTIAL = auto()  # Some agents succeeded, some failed
    FAILED = auto()
    CANCELLED = auto()
    TIMEOUT = auto()


@dataclass
class AgentResult:
    """Result from a single agent invocation."""

    agent_name: str
    success: bool
    output: Any
    error: Optional[str] = None
    duration_ms: int = 0
    tokens_consumed: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FlowResult:
    """Result of complete flow execution."""

    request_id: str
    status: FlowStatus
    results: List[AgentResult]
    primary_output: Optional[Any] = None
    total_duration_ms: int = 0
    strategy_used: RoutingStrategy = RoutingStrategy.SINGLE
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def is_success(self) -> bool:
        return self.status == FlowStatus.COMPLETED

    @property
    def successful_results(self) -> List[AgentResult]:
        return [r for r in self.results if r.success]

    @property
    def failed_results(self) -> List[AgentResult]:
        return [r for r in self.results if not r.success]


# Type for agent invocation function
AgentInvoker = Callable[[str, Any, Dict[str, Any]], Any]


class FlowController:
    """
    Controls execution flow for multi-agent workflows.

    Supports:
    - Single agent execution
    - Parallel execution (multiple agents simultaneously)
    - Sequential execution (agents in order, output chaining)
    - Fallback execution (try agents until success)
    """

    def __init__(
        self,
        max_workers: int = 4,
        default_timeout_ms: int = 30000,
    ):
        """
        Initialize flow controller.

        Args:
            max_workers: Maximum parallel workers
            default_timeout_ms: Default timeout per agent
        """
        self.max_workers = max_workers
        self.default_timeout_ms = default_timeout_ms
        self._executor = ThreadPoolExecutor(max_workers=max_workers)

        # Metrics
        self._flow_count = 0
        self._parallel_count = 0
        self._sequential_count = 0
        self._fallback_count = 0

    def execute(
        self,
        request_id: str,
        decision: RoutingDecision,
        request: Any,
        invoker: AgentInvoker,
        context: Optional[Dict[str, Any]] = None,
    ) -> FlowResult:
        """
        Execute a flow based on routing decision.

        Args:
            request_id: Request identifier
            decision: Routing decision
            request: The request to process
            invoker: Function to invoke an agent
            context: Additional context

        Returns:
            FlowResult with all agent results
        """
        self._flow_count += 1
        start_time = time.time()
        context = context or {}

        strategy = decision.strategy
        routes = decision.routes

        if not routes:
            return FlowResult(
                request_id=request_id,
                status=FlowStatus.FAILED,
                results=[],
                total_duration_ms=0,
                strategy_used=strategy,
            )

        # Execute based on strategy
        if strategy == RoutingStrategy.SINGLE:
            results = self._execute_single(routes[0], request, invoker, context)
        elif strategy == RoutingStrategy.PARALLEL:
            self._parallel_count += 1
            results = self._execute_parallel(routes, request, invoker, context)
        elif strategy == RoutingStrategy.SEQUENTIAL:
            self._sequential_count += 1
            results = self._execute_sequential(routes, request, invoker, context)
        elif strategy == RoutingStrategy.FALLBACK:
            self._fallback_count += 1
            results = self._execute_fallback(routes, request, invoker, context)
        else:
            results = self._execute_single(routes[0], request, invoker, context)

        # Determine overall status
        status = self._determine_status(results)
        primary_output = self._get_primary_output(results, strategy)

        total_duration = int((time.time() - start_time) * 1000)

        return FlowResult(
            request_id=request_id,
            status=status,
            results=results,
            primary_output=primary_output,
            total_duration_ms=total_duration,
            strategy_used=strategy,
        )

    def _execute_single(
        self,
        route: AgentRoute,
        request: Any,
        invoker: AgentInvoker,
        context: Dict[str, Any],
    ) -> List[AgentResult]:
        """Execute single agent."""
        result = self._invoke_agent(route, request, invoker, context)
        return [result]

    def _execute_parallel(
        self,
        routes: List[AgentRoute],
        request: Any,
        invoker: AgentInvoker,
        context: Dict[str, Any],
    ) -> List[AgentResult]:
        """Execute multiple agents in parallel."""
        results = []
        futures: Dict[Future, AgentRoute] = {}

        for route in routes:
            future = self._executor.submit(self._invoke_agent, route, request, invoker, context)
            futures[future] = route

        # Collect results as they complete
        for future in as_completed(futures.keys()):
            route = futures[future]
            try:
                result = future.result(timeout=route.timeout_ms / 1000)
                results.append(result)
            except Exception as e:
                results.append(
                    AgentResult(
                        agent_name=route.agent_name,
                        success=False,
                        output=None,
                        error=str(e),
                    )
                )

        return results

    def _execute_sequential(
        self,
        routes: List[AgentRoute],
        request: Any,
        invoker: AgentInvoker,
        context: Dict[str, Any],
    ) -> List[AgentResult]:
        """Execute agents sequentially, chaining outputs."""
        results = []
        current_input = request

        for route in routes:
            result = self._invoke_agent(route, current_input, invoker, context)
            results.append(result)

            if not result.success:
                # Stop sequence on failure
                break

            # Chain output to next agent's input
            current_input = result.output

        return results

    def _execute_fallback(
        self,
        routes: List[AgentRoute],
        request: Any,
        invoker: AgentInvoker,
        context: Dict[str, Any],
    ) -> List[AgentResult]:
        """Execute agents in order until one succeeds."""
        results = []

        for route in routes:
            result = self._invoke_agent(route, request, invoker, context)
            results.append(result)

            if result.success:
                # Stop on first success
                break

        return results

    def _invoke_agent(
        self,
        route: AgentRoute,
        request: Any,
        invoker: AgentInvoker,
        context: Dict[str, Any],
    ) -> AgentResult:
        """Invoke a single agent with timeout."""
        start_time = time.time()
        timeout = route.timeout_ms / 1000

        try:
            # Add route metadata to context
            agent_context = {
                **context,
                "agent_name": route.agent_name,
                "max_tokens": route.max_tokens,
                "route_metadata": route.metadata,
            }

            # Execute with timeout
            output = invoker(route.agent_name, request, agent_context)

            duration_ms = int((time.time() - start_time) * 1000)

            return AgentResult(
                agent_name=route.agent_name,
                success=True,
                output=output,
                duration_ms=duration_ms,
            )

        except TimeoutError:
            return AgentResult(
                agent_name=route.agent_name,
                success=False,
                output=None,
                error=f"Agent timed out after {timeout}s",
                duration_ms=int(timeout * 1000),
            )

        except Exception as e:
            duration_ms = int((time.time() - start_time) * 1000)
            logger.error(f"Agent {route.agent_name} error: {e}")
            return AgentResult(
                agent_name=route.agent_name,
                success=False,
                output=None,
                error=str(e),
                duration_ms=duration_ms,
            )

    def _determine_status(self, results: List[AgentResult]) -> FlowStatus:
        """Determine overall flow status from results."""
        if not results:
            return FlowStatus.FAILED

        successes = sum(1 for r in results if r.success)
        failures = len(results) - successes

        if successes == len(results):
            return FlowStatus.COMPLETED
        elif successes > 0:
            return FlowStatus.PARTIAL
        else:
            return FlowStatus.FAILED

    def _get_primary_output(
        self,
        results: List[AgentResult],
        strategy: RoutingStrategy,
    ) -> Optional[Any]:
        """Get primary output based on strategy."""
        if not results:
            return None

        successful = [r for r in results if r.success]
        if not successful:
            return None

        if strategy == RoutingStrategy.SEQUENTIAL:
            # Last successful result is the output
            return successful[-1].output
        elif strategy == RoutingStrategy.FALLBACK:
            # First successful result
            return successful[0].output
        else:
            # First successful result (single or parallel)
            return successful[0].output

    def shutdown(self) -> None:
        """Shutdown executor."""
        self._executor.shutdown(wait=False)

    def get_metrics(self) -> Dict[str, Any]:
        """Get flow execution metrics."""
        return {
            "total_flows": self._flow_count,
            "parallel_flows": self._parallel_count,
            "sequential_flows": self._sequential_count,
            "fallback_flows": self._fallback_count,
        }


class AsyncFlowController:
    """
    Async version of FlowController for use with asyncio.
    """

    def __init__(
        self,
        default_timeout_ms: int = 30000,
    ):
        self.default_timeout_ms = default_timeout_ms

    async def execute(
        self,
        request_id: str,
        decision: RoutingDecision,
        request: Any,
        invoker: Callable[[str, Any, Dict[str, Any]], Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> FlowResult:
        """Async execute a flow."""
        context = context or {}
        routes = decision.routes
        strategy = decision.strategy

        if not routes:
            return FlowResult(
                request_id=request_id,
                status=FlowStatus.FAILED,
                results=[],
                strategy_used=strategy,
            )

        start_time = time.time()

        if strategy == RoutingStrategy.PARALLEL:
            results = await self._execute_parallel_async(routes, request, invoker, context)
        else:
            # For other strategies, run synchronously in executor
            loop = asyncio.get_event_loop()
            sync_controller = FlowController()
            result = await loop.run_in_executor(
                None, sync_controller.execute, request_id, decision, request, invoker, context
            )
            sync_controller.shutdown()
            return result

        status = self._determine_status(results)
        primary_output = results[0].output if results and results[0].success else None
        total_duration = int((time.time() - start_time) * 1000)

        return FlowResult(
            request_id=request_id,
            status=status,
            results=results,
            primary_output=primary_output,
            total_duration_ms=total_duration,
            strategy_used=strategy,
        )

    async def _execute_parallel_async(
        self,
        routes: List[AgentRoute],
        request: Any,
        invoker: Callable,
        context: Dict[str, Any],
    ) -> List[AgentResult]:
        """Execute agents in parallel using asyncio."""
        tasks = []

        for route in routes:
            task = asyncio.create_task(self._invoke_agent_async(route, request, invoker, context))
            tasks.append(task)

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Convert exceptions to AgentResults
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append(
                    AgentResult(
                        agent_name=routes[i].agent_name,
                        success=False,
                        output=None,
                        error=str(result),
                    )
                )
            else:
                processed_results.append(result)

        return processed_results

    async def _invoke_agent_async(
        self,
        route: AgentRoute,
        request: Any,
        invoker: Callable,
        context: Dict[str, Any],
    ) -> AgentResult:
        """Invoke agent asynchronously."""
        start_time = time.time()
        timeout = route.timeout_ms / 1000

        try:
            agent_context = {
                **context,
                "agent_name": route.agent_name,
                "max_tokens": route.max_tokens,
            }

            # Run invoker in executor
            loop = asyncio.get_event_loop()
            output = await asyncio.wait_for(
                loop.run_in_executor(None, invoker, route.agent_name, request, agent_context),
                timeout=timeout,
            )

            duration_ms = int((time.time() - start_time) * 1000)
            return AgentResult(
                agent_name=route.agent_name,
                success=True,
                output=output,
                duration_ms=duration_ms,
            )

        except asyncio.TimeoutError:
            return AgentResult(
                agent_name=route.agent_name,
                success=False,
                output=None,
                error=f"Agent timed out after {timeout}s",
            )

        except Exception as e:
            duration_ms = int((time.time() - start_time) * 1000)
            return AgentResult(
                agent_name=route.agent_name,
                success=False,
                output=None,
                error=str(e),
                duration_ms=duration_ms,
            )

    def _determine_status(self, results: List[AgentResult]) -> FlowStatus:
        """Determine overall flow status."""
        if not results:
            return FlowStatus.FAILED

        successes = sum(1 for r in results if r.success)
        if successes == len(results):
            return FlowStatus.COMPLETED
        elif successes > 0:
            return FlowStatus.PARTIAL
        return FlowStatus.FAILED
