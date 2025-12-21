"""
Agent OS Muse Agent

The Creative Agent - generates stories, poems, scenarios, and artistic content.
Muse operates at high temperature for creative exploration while respecting
constitutional boundaries. All outputs are drafts requiring human approval.

Key Components:
- MuseAgent: Main agent implementing BaseAgent interface
- CreativeEngine: Core creative generation with styles and modes
- Guardian integration: Mandatory Smith review for all outputs

Usage:
    from src.agents.muse import create_muse_agent, MuseAgent

    # Create with mock (for testing)
    agent = create_muse_agent(use_mock=True)

    # Generate a story
    result = agent.generate_story("A robot discovers music")
    print(result.get_primary().format_as_draft())

    # Brainstorm ideas
    ideas = agent.brainstorm("sustainable energy solutions")
    print(ideas.format_all_options())
"""

from .agent import (
    MuseAgent,
    MuseConfig,
    create_muse_agent,
)

from .creative import (
    CreativeEngine,
    CreativeStyle,
    ContentType,
    CreativeMode,
    CreativeConstraints,
    CreativeResult,
    CreativeOption,
    create_creative_engine,
)

__all__ = [
    # Agent
    "MuseAgent",
    "MuseConfig",
    "create_muse_agent",
    # Creative Engine
    "CreativeEngine",
    "CreativeStyle",
    "ContentType",
    "CreativeMode",
    "CreativeConstraints",
    "CreativeResult",
    "CreativeOption",
    "create_creative_engine",
]
