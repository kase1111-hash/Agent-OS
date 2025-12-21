"""
Mock Objects for Agent Testing

Provides mock implementations of common dependencies.
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Set

from src.agents.interface import (
    BaseAgent,
    AgentCapabilities,
    CapabilityType,
    RequestValidationResult,
    AgentState,
)
from src.messaging.models import FlowRequest, FlowResponse, MessageStatus
from src.core.models import Rule, RuleType


class MockAgent(BaseAgent):
    """
    Mock agent for testing.

    Records all requests and returns configurable responses.
    """

    def __init__(
        self,
        name: str = "mock_agent",
        default_response: str = "Mock response",
    ):
        super().__init__(
            name=name,
            description="Mock agent for testing",
        )
        self.default_response = default_response
        self.received_requests: List[FlowRequest] = []
        self.responses: Dict[str, str] = {}  # intent -> response
        self._response_handler: Optional[Callable[[FlowRequest], str]] = None
        self._should_refuse = False
        self._should_error = False
        self._error_message = "Mock error"

    def set_response(self, intent: str, response: str) -> "MockAgent":
        """Set response for specific intent."""
        self.responses[intent] = response
        return self

    def set_handler(
        self,
        handler: Callable[[FlowRequest], str],
    ) -> "MockAgent":
        """Set custom response handler."""
        self._response_handler = handler
        return self

    def should_refuse(self, refuse: bool = True) -> "MockAgent":
        """Configure agent to refuse requests."""
        self._should_refuse = refuse
        return self

    def should_error(
        self,
        error: bool = True,
        message: str = "Mock error",
    ) -> "MockAgent":
        """Configure agent to return errors."""
        self._should_error = error
        self._error_message = message
        return self

    def process(self, request: FlowRequest) -> FlowResponse:
        """Process request and record it."""
        self.received_requests.append(request)

        if self._should_refuse:
            return request.create_response(
                source=self.name,
                status=MessageStatus.REFUSED,
                output="Request refused",
            )

        if self._should_error:
            return request.create_response(
                source=self.name,
                status=MessageStatus.ERROR,
                output="",
                errors=[self._error_message],
            )

        # Determine response
        if self._response_handler:
            output = self._response_handler(request)
        elif request.intent in self.responses:
            output = self.responses[request.intent]
        else:
            output = self.default_response

        return request.create_response(
            source=self.name,
            status=MessageStatus.SUCCESS,
            output=output,
        )

    def get_request(self, index: int = -1) -> Optional[FlowRequest]:
        """Get a recorded request by index."""
        if self.received_requests:
            return self.received_requests[index]
        return None

    def clear_requests(self) -> None:
        """Clear recorded requests."""
        self.received_requests = []


@dataclass
class MockMemoryEntry:
    """A mock memory entry."""
    id: str
    content: str
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


class MockMemoryStore:
    """
    Mock memory store for testing.

    Provides in-memory storage with basic search.
    """

    def __init__(self):
        self._entries: Dict[str, MockMemoryEntry] = {}
        self._recall_responses: Dict[str, List[MockMemoryEntry]] = {}

    def store(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Store content and return ID."""
        entry_id = f"mem-{uuid.uuid4().hex[:8]}"
        self._entries[entry_id] = MockMemoryEntry(
            id=entry_id,
            content=content,
            metadata=metadata or {},
        )
        return entry_id

    def recall(
        self,
        query: str,
        limit: int = 5,
    ) -> List[MockMemoryEntry]:
        """
        Recall memories matching query.

        If pre-configured responses exist for query, return those.
        Otherwise, do simple substring matching.
        """
        if query in self._recall_responses:
            return self._recall_responses[query][:limit]

        # Simple substring matching
        matches = []
        query_lower = query.lower()
        for entry in self._entries.values():
            if query_lower in entry.content.lower():
                matches.append(entry)
                if len(matches) >= limit:
                    break

        return matches

    def set_recall_response(
        self,
        query: str,
        entries: List[MockMemoryEntry],
    ) -> "MockMemoryStore":
        """Pre-configure response for a query."""
        self._recall_responses[query] = entries
        return self

    def get(self, entry_id: str) -> Optional[MockMemoryEntry]:
        """Get entry by ID."""
        return self._entries.get(entry_id)

    def delete(self, entry_id: str) -> bool:
        """Delete entry."""
        if entry_id in self._entries:
            del self._entries[entry_id]
            return True
        return False

    def clear(self) -> None:
        """Clear all entries."""
        self._entries.clear()
        self._recall_responses.clear()


class MockToolsClient:
    """
    Mock tools client for testing.

    Records tool invocations and returns configurable results.
    """

    def __init__(self):
        self._invocations: List[Dict[str, Any]] = []
        self._results: Dict[str, Any] = {}  # tool_name -> result
        self._available_tools: Set[str] = set()
        self._should_fail: Set[str] = set()

    def register_tool(self, name: str, result: Any = None) -> "MockToolsClient":
        """Register an available tool with optional result."""
        self._available_tools.add(name)
        if result is not None:
            self._results[name] = result
        return self

    def set_result(self, tool_name: str, result: Any) -> "MockToolsClient":
        """Set result for a tool."""
        self._results[tool_name] = result
        return self

    def should_fail(self, tool_name: str) -> "MockToolsClient":
        """Configure tool to fail."""
        self._should_fail.add(tool_name)
        return self

    def invoke(
        self,
        tool_name: str,
        parameters: Dict[str, Any],
        user_id: str,
        agent_id: Optional[str] = None,
        timeout: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Invoke a tool."""
        self._invocations.append({
            "tool_name": tool_name,
            "parameters": parameters,
            "user_id": user_id,
            "agent_id": agent_id,
            "timestamp": datetime.now(),
        })

        if tool_name not in self._available_tools:
            return {
                "success": False,
                "error": f"Tool not found: {tool_name}",
            }

        if tool_name in self._should_fail:
            return {
                "success": False,
                "error": f"Tool failed: {tool_name}",
            }

        return {
            "success": True,
            "output": self._results.get(tool_name, f"Result from {tool_name}"),
        }

    def list_tools(self, available_only: bool = True) -> List[str]:
        """List available tools."""
        return list(self._available_tools)

    def can_use_tool(self, user_id: str, tool_name: str) -> bool:
        """Check if tool can be used."""
        return tool_name in self._available_tools

    def get_invocations(
        self,
        tool_name: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Get recorded invocations."""
        if tool_name:
            return [i for i in self._invocations if i["tool_name"] == tool_name]
        return list(self._invocations)

    def clear_invocations(self) -> None:
        """Clear recorded invocations."""
        self._invocations.clear()


class MockConstitution:
    """
    Mock constitution for testing.

    Provides configurable rule checking.
    """

    def __init__(self):
        self._rules: List[Rule] = []
        self._blocked_patterns: Set[str] = set()
        self._escalation_patterns: Set[str] = set()

    def add_rule(
        self,
        content: str,
        rule_type: RuleType = RuleType.PERMISSION,
        keywords: Optional[List[str]] = None,
    ) -> "MockConstitution":
        """Add a rule."""
        rule = Rule(
            rule_id=f"rule-{len(self._rules) + 1}",
            content=content,
            rule_type=rule_type,
            keywords=keywords or [],
        )
        self._rules.append(rule)
        return self

    def block_pattern(self, pattern: str) -> "MockConstitution":
        """Add a pattern that should be blocked."""
        self._blocked_patterns.add(pattern.lower())
        return self

    def escalate_pattern(self, pattern: str) -> "MockConstitution":
        """Add a pattern that should trigger escalation."""
        self._escalation_patterns.add(pattern.lower())
        return self

    def check(self, content: str) -> Dict[str, Any]:
        """Check content against constitution."""
        content_lower = content.lower()

        # Check for blocked patterns
        for pattern in self._blocked_patterns:
            if pattern in content_lower:
                return {
                    "allowed": False,
                    "reason": f"Blocked by pattern: {pattern}",
                }

        # Check for escalation patterns
        for pattern in self._escalation_patterns:
            if pattern in content_lower:
                return {
                    "allowed": True,
                    "requires_escalation": True,
                    "reason": f"Escalation required: {pattern}",
                }

        return {
            "allowed": True,
            "requires_escalation": False,
        }

    def get_rules(self) -> List[Rule]:
        """Get all rules."""
        return list(self._rules)

    def get_applicable_rules(self, content: str) -> List[Rule]:
        """Get rules applicable to content."""
        content_lower = content.lower()
        applicable = []
        for rule in self._rules:
            for keyword in rule.keywords:
                if keyword in content_lower:
                    applicable.append(rule)
                    break
        return applicable
