"""
Tool-Using Agent Template

Provides a template for agents that can invoke external tools.
Integrates with the Tool Integration Framework (UC-015).
"""

import logging
from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set

from src.agents.interface import CapabilityType
from src.messaging.models import FlowRequest, FlowResponse, MessageStatus
from src.tools import (
    ToolsClient,
    ToolResult,
    ToolCategory,
    ToolRiskLevel,
    InvocationResult,
)

from .base import AgentTemplate, AgentConfig


logger = logging.getLogger(__name__)


@dataclass
class ToolCall:
    """A tool call to be executed."""
    tool_name: str
    parameters: Dict[str, Any]
    reason: str = ""


@dataclass
class ToolCallResult:
    """Result of a tool call."""
    tool_name: str
    success: bool
    output: Any = None
    error: Optional[str] = None
    execution_time_ms: int = 0


@dataclass
class ToolUseConfig(AgentConfig):
    """Configuration for tool-using agents."""
    allowed_tools: Optional[Set[str]] = None
    allowed_categories: Optional[Set[ToolCategory]] = None
    max_tool_calls: int = 10
    require_tool_reason: bool = True
    parallel_tools: bool = False
    stop_on_error: bool = True

    def __post_init__(self):
        self.capabilities = self.capabilities or set()
        self.capabilities.add(CapabilityType.TOOL_USE)


class ToolUseAgentTemplate(AgentTemplate):
    """
    Template for tool-using agents.

    Provides:
    - Tool discovery and invocation
    - Tool selection logic
    - Multi-step tool use
    - Error handling
    - Result aggregation
    """

    def __init__(
        self,
        config: ToolUseConfig,
        tools_client: Optional[ToolsClient] = None,
    ):
        """
        Initialize tool-using agent.

        Args:
            config: Tool use agent configuration
            tools_client: Tools client (uses default if not provided)
        """
        super().__init__(config)
        self.tool_config = config
        self._tools_client = tools_client
        self._tool_history: List[ToolCallResult] = []

    @property
    def tools_client(self) -> ToolsClient:
        """Get tools client, creating default if needed."""
        if self._tools_client is None:
            from src.tools import get_default_client
            self._tools_client = get_default_client()
        return self._tools_client

    def handle_request(self, request: FlowRequest) -> FlowResponse:
        """Process request using tools."""
        self._tool_history = []

        try:
            # Analyze request and determine tools needed
            tool_calls = self.plan_tool_use(request)

            if not tool_calls:
                # No tools needed, generate response directly
                return self.generate_response(request, [])

            # Validate tool access
            for call in tool_calls:
                if not self._can_use_tool(call.tool_name):
                    return request.create_response(
                        source=self.name,
                        status=MessageStatus.REFUSED,
                        output="",
                        errors=[f"Tool not allowed: {call.tool_name}"],
                    )

            # Execute tools
            if self.tool_config.parallel_tools:
                results = self._execute_parallel(tool_calls, request)
            else:
                results = self._execute_sequential(tool_calls, request)

            # Check for errors
            errors = [r for r in results if not r.success]
            if errors and self.tool_config.stop_on_error:
                return request.create_response(
                    source=self.name,
                    status=MessageStatus.ERROR,
                    output="Tool execution failed",
                    errors=[r.error for r in errors if r.error],
                )

            # Generate response using tool results
            return self.generate_response(request, results)

        except Exception as e:
            logger.error(f"Tool use error: {e}")
            return request.create_response(
                source=self.name,
                status=MessageStatus.ERROR,
                output="",
                errors=[f"Tool use failed: {str(e)}"],
            )

    @abstractmethod
    def plan_tool_use(self, request: FlowRequest) -> List[ToolCall]:
        """
        Plan which tools to use for this request.

        Override this method to implement tool selection logic.

        Args:
            request: The request

        Returns:
            List of tool calls to execute
        """
        pass

    @abstractmethod
    def generate_response(
        self,
        request: FlowRequest,
        tool_results: List[ToolCallResult],
    ) -> FlowResponse:
        """
        Generate final response using tool results.

        Override this method to implement response generation.

        Args:
            request: Original request
            tool_results: Results from tool executions

        Returns:
            FlowResponse
        """
        pass

    def invoke_tool(
        self,
        tool_name: str,
        parameters: Dict[str, Any],
        user_id: str,
    ) -> ToolCallResult:
        """
        Invoke a single tool.

        Args:
            tool_name: Name of tool
            parameters: Tool parameters
            user_id: User ID

        Returns:
            ToolCallResult
        """
        result = self.tools_client.invoke(
            tool_name=tool_name,
            parameters=parameters,
            user_id=user_id,
            agent_id=self.name,
        )

        call_result = ToolCallResult(
            tool_name=tool_name,
            success=result.is_success,
            output=result.output,
            error=result.error,
            execution_time_ms=result.execution_time_ms,
        )

        self._tool_history.append(call_result)
        return call_result

    def get_available_tools(self) -> List[str]:
        """Get list of available tool names."""
        registrations = self.tools_client.list_tools(available_only=True)

        tool_names = []
        for reg in registrations:
            # Check allowed filters
            if self.tool_config.allowed_tools:
                if reg.tool.name not in self.tool_config.allowed_tools:
                    continue

            if self.tool_config.allowed_categories:
                if reg.schema.category not in self.tool_config.allowed_categories:
                    continue

            tool_names.append(reg.tool.name)

        return tool_names

    def get_tool_schema(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """Get schema for a tool."""
        schema = self.tools_client.get_tool_schema(tool_name)
        if schema:
            return schema.to_dict()
        return None

    def _can_use_tool(self, tool_name: str) -> bool:
        """Check if agent can use a tool."""
        if self.tool_config.allowed_tools:
            if tool_name not in self.tool_config.allowed_tools:
                return False

        registration = self.tools_client.get_tool(tool_name)
        if not registration:
            return False

        if self.tool_config.allowed_categories:
            if registration.schema.category not in self.tool_config.allowed_categories:
                return False

        return True

    def _execute_sequential(
        self,
        tool_calls: List[ToolCall],
        request: FlowRequest,
    ) -> List[ToolCallResult]:
        """Execute tools sequentially."""
        results = []
        user_id = request.source

        for i, call in enumerate(tool_calls):
            if i >= self.tool_config.max_tool_calls:
                logger.warning(f"Max tool calls ({self.tool_config.max_tool_calls}) reached")
                break

            logger.info(f"Executing tool: {call.tool_name}")
            result = self.invoke_tool(call.tool_name, call.parameters, user_id)
            results.append(result)

            if not result.success and self.tool_config.stop_on_error:
                break

        return results

    def _execute_parallel(
        self,
        tool_calls: List[ToolCall],
        request: FlowRequest,
    ) -> List[ToolCallResult]:
        """Execute tools in parallel."""
        import concurrent.futures

        user_id = request.source
        results = []

        # Limit to max_tool_calls
        calls_to_execute = tool_calls[:self.tool_config.max_tool_calls]

        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = {
                executor.submit(
                    self.invoke_tool,
                    call.tool_name,
                    call.parameters,
                    user_id,
                ): call
                for call in calls_to_execute
            }

            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    call = futures[future]
                    results.append(ToolCallResult(
                        tool_name=call.tool_name,
                        success=False,
                        error=str(e),
                    ))

        return results


def create_tool_use_agent(
    name: str,
    plan_fn: Callable[[FlowRequest], List[ToolCall]],
    respond_fn: Callable[[FlowRequest, List[ToolCallResult]], FlowResponse],
    description: str = "",
    allowed_tools: Optional[Set[str]] = None,
    tools_client: Optional[ToolsClient] = None,
    **kwargs,
) -> ToolUseAgentTemplate:
    """
    Create a tool-using agent from functions.

    Example:
        def plan_tools(request):
            # Decide which tools to use
            return [
                ToolCall(
                    tool_name="calculator",
                    parameters={"expression": "2 + 2"},
                    reason="Need to calculate",
                ),
            ]

        def generate_response(request, results):
            output = "Results:\\n"
            for r in results:
                output += f"- {r.tool_name}: {r.output}\\n"
            return request.create_response(
                source="math_agent",
                status=MessageStatus.SUCCESS,
                output=output,
            )

        agent = create_tool_use_agent(
            name="math_agent",
            plan_fn=plan_tools,
            respond_fn=generate_response,
            description="Uses calculator tool",
        )

    Args:
        name: Agent name
        plan_fn: Tool planning function
        respond_fn: Response generation function
        description: Agent description
        allowed_tools: Set of allowed tool names
        tools_client: Tools client instance
        **kwargs: Additional config options

    Returns:
        ToolUseAgentTemplate instance
    """
    config = ToolUseConfig(
        name=name,
        description=description,
        allowed_tools=allowed_tools,
        **kwargs,
    )

    class FunctionToolUseAgent(ToolUseAgentTemplate):
        def plan_tool_use(self, request: FlowRequest) -> List[ToolCall]:
            return plan_fn(request)

        def generate_response(
            self,
            request: FlowRequest,
            tool_results: List[ToolCallResult],
        ) -> FlowResponse:
            return respond_fn(request, tool_results)

    return FunctionToolUseAgent(config, tools_client)
