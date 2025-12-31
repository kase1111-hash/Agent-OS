"""
Agent Development Templates

Provides base templates for creating various types of agents.
These templates simplify agent development by providing pre-configured
base classes with sensible defaults.
"""

from .base import (
    AgentTemplate,
    SimpleAgent,
    create_simple_agent,
)
from .generation import (
    GenerationAgentTemplate,
    create_generation_agent,
)
from .reasoning import (
    ReasoningAgentTemplate,
    create_reasoning_agent,
)
from .tool_use import (
    ToolUseAgentTemplate,
    create_tool_use_agent,
)
from .validation import (
    ValidationAgentTemplate,
    create_validation_agent,
)

__all__ = [
    # Base
    "AgentTemplate",
    "SimpleAgent",
    "create_simple_agent",
    # Reasoning
    "ReasoningAgentTemplate",
    "create_reasoning_agent",
    # Generation
    "GenerationAgentTemplate",
    "create_generation_agent",
    # Validation
    "ValidationAgentTemplate",
    "create_validation_agent",
    # Tool Use
    "ToolUseAgentTemplate",
    "create_tool_use_agent",
]
