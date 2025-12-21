"""
Agent OS Sage Agent

The Reasoning Agent - provides deep chain-of-thought reasoning,
analysis, synthesis, and trade-off evaluation.

Sage illuminates complex problems through rigorous multi-step
reasoning while respecting human sovereignty in decision-making.

Key Components:
- SageAgent: Main agent implementing BaseAgent interface
- ReasoningEngine: Chain-of-thought reasoning system
- ReasoningChain: Complete reasoning result with steps

Usage:
    from src.agents.sage import create_sage_agent, SageAgent

    # Create with mock (for testing)
    agent = create_sage_agent(use_mock=True)

    # Perform reasoning
    chain = agent.reason("Analyze the trade-offs of remote work")
    print(chain.format_markdown())
"""

from .agent import (
    SageAgent,
    SageConfig,
    create_sage_agent,
)

from .reasoning import (
    ReasoningEngine,
    ReasoningConfig,
    ReasoningType,
    ReasoningChain,
    ReasoningStep,
    TradeOff,
    ConfidenceLevel,
    create_reasoning_engine,
)

__all__ = [
    # Agent
    "SageAgent",
    "SageConfig",
    "create_sage_agent",
    # Reasoning
    "ReasoningEngine",
    "ReasoningConfig",
    "ReasoningType",
    "ReasoningChain",
    "ReasoningStep",
    "TradeOff",
    "ConfidenceLevel",
    "create_reasoning_engine",
]
