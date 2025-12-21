"""
Agent OS Quill Agent

The Writer Agent - provides document refinement, formatting, and polishing.
Quill preserves authorial intent while improving clarity and presentation.

Key Components:
- QuillAgent: Main agent implementing BaseAgent interface
- FormattingEngine: Document formatting with templates
- RefinementEngine: Grammar, style, and change tracking

Usage:
    from src.agents.quill import create_quill_agent, QuillAgent

    # Create with mock (for testing)
    agent = create_quill_agent(use_mock=True)

    # Refine text
    result = agent.refine("This is a rough draft with some erorrs.")
    print(result.refined)
    print(result.format_diff())
"""

from .agent import (
    QuillAgent,
    QuillConfig,
    create_quill_agent,
)

from .formatting import (
    FormattingEngine,
    RefinementEngine,
    OutputFormat,
    ChangeType,
    TextChange,
    RefinementResult,
    DocumentTemplate,
    DEFAULT_TEMPLATES,
    create_formatting_engine,
    create_refinement_engine,
)

__all__ = [
    # Agent
    "QuillAgent",
    "QuillConfig",
    "create_quill_agent",
    # Formatting
    "FormattingEngine",
    "RefinementEngine",
    "OutputFormat",
    "ChangeType",
    "TextChange",
    "RefinementResult",
    "DocumentTemplate",
    "DEFAULT_TEMPLATES",
    "create_formatting_engine",
    "create_refinement_engine",
]
