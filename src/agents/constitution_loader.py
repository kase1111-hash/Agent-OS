"""
Constitution Loader for Agent OS Agents

Loads and injects constitutional context into agent system prompts.
Each agent receives:
1. The supreme CONSTITUTION.md (core rules that apply to all agents)
2. Their own agents/<name>/constitution.md (agent-specific rules)

This ensures the LLM has access to the constitutional rules in its context.
"""

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class ConstitutionalContext:
    """Constitutional context for an agent."""

    agent_name: str
    supreme_constitution: str
    agent_constitution: str
    combined_prompt: str

    @property
    def has_supreme(self) -> bool:
        return bool(self.supreme_constitution.strip())

    @property
    def has_agent_specific(self) -> bool:
        return bool(self.agent_constitution.strip())


class ConstitutionLoader:
    """
    Loads constitutional documents for injection into agent system prompts.

    Ensures each agent only receives:
    - The supreme CONSTITUTION.md (mandatory for all agents)
    - Their own constitution.md (agent-specific)

    Other agents' constitutions are NOT loaded.
    """

    def __init__(self, project_root: Optional[Path] = None):
        """
        Initialize the constitution loader.

        Args:
            project_root: Root directory of Agent-OS project.
                         If None, attempts to auto-detect.
        """
        self.project_root = project_root or self._detect_project_root()
        self._cache: dict[str, ConstitutionalContext] = {}

    def _detect_project_root(self) -> Path:
        """Attempt to detect the project root directory."""
        # Try relative to this file
        current = Path(__file__).resolve()

        # Walk up looking for CONSTITUTION.md
        for parent in [current] + list(current.parents):
            if (parent / "CONSTITUTION.md").exists():
                return parent

        # Fallback to current working directory
        cwd = Path.cwd()
        if (cwd / "CONSTITUTION.md").exists():
            return cwd

        # Last resort - assume standard structure
        return Path(__file__).parent.parent.parent

    def load_for_agent(
        self,
        agent_name: str,
        include_supreme: bool = True,
        force_reload: bool = False,
    ) -> ConstitutionalContext:
        """
        Load constitutional context for a specific agent.

        Args:
            agent_name: Name of the agent (e.g., "muse", "sage", "smith")
            include_supreme: Whether to include the supreme CONSTITUTION.md
            force_reload: Force reload from disk (ignore cache)

        Returns:
            ConstitutionalContext with loaded documents
        """
        cache_key = f"{agent_name}:{include_supreme}"

        if not force_reload and cache_key in self._cache:
            return self._cache[cache_key]

        # Load supreme constitution
        supreme_content = ""
        if include_supreme:
            supreme_content = self._load_supreme_constitution()

        # Load agent-specific constitution
        agent_content = self._load_agent_constitution(agent_name)

        # Build combined prompt section
        combined = self._build_constitutional_prompt(
            agent_name=agent_name,
            supreme=supreme_content,
            agent_specific=agent_content,
        )

        context = ConstitutionalContext(
            agent_name=agent_name,
            supreme_constitution=supreme_content,
            agent_constitution=agent_content,
            combined_prompt=combined,
        )

        self._cache[cache_key] = context
        logger.info(
            f"Loaded constitutional context for {agent_name}: "
            f"supreme={context.has_supreme}, agent_specific={context.has_agent_specific}"
        )

        return context

    def _load_supreme_constitution(self) -> str:
        """Load the supreme CONSTITUTION.md file."""
        constitution_path = self.project_root / "CONSTITUTION.md"

        if not constitution_path.exists():
            logger.warning(f"Supreme constitution not found at {constitution_path}")
            return ""

        try:
            content = constitution_path.read_text(encoding="utf-8")
            # Extract the most relevant sections for the LLM context
            return self._extract_core_rules(content)
        except Exception as e:
            logger.error(f"Failed to load supreme constitution: {e}")
            return ""

    def _load_agent_constitution(self, agent_name: str) -> str:
        """Load the agent-specific constitution.md file."""
        # Check standard location: agents/<name>/constitution.md
        constitution_path = self.project_root / "agents" / agent_name / "constitution.md"

        if not constitution_path.exists():
            logger.debug(f"No agent-specific constitution for {agent_name}")
            return ""

        try:
            content = constitution_path.read_text(encoding="utf-8")
            # Remove YAML frontmatter if present
            content = self._strip_frontmatter(content)
            return content.strip()
        except Exception as e:
            logger.error(f"Failed to load agent constitution for {agent_name}: {e}")
            return ""

    def _strip_frontmatter(self, content: str) -> str:
        """Remove YAML frontmatter from markdown content."""
        if content.startswith("---"):
            # Find the closing ---
            parts = content.split("---", 2)
            if len(parts) >= 3:
                return parts[2].strip()
        return content

    def _extract_core_rules(self, content: str) -> str:
        """
        Extract the core rules from the supreme constitution.

        This creates a condensed version suitable for LLM context,
        focusing on:
        - Core principles
        - Mandatory rules (MUST/SHALL/MUST NOT)
        - Prohibited actions
        - Human sovereignty clauses
        """
        lines = content.split("\n")
        extracted = []
        in_relevant_section = False
        current_section = ""

        # Key sections to include
        relevant_sections = {
            "declaration of intent",
            "core principles",
            "human sovereignty",
            "agent boundaries",
            "prohibited actions",
            "mandatory constraints",
            "fundamental rights",
            "immutable",
            "escalation",
            "refusal",
        }

        for line in lines:
            line_lower = line.lower().strip()

            # Track section headers
            if line.startswith("#"):
                header_text = line_lower.lstrip("#").strip()
                in_relevant_section = any(section in header_text for section in relevant_sections)
                if in_relevant_section:
                    current_section = header_text
                    extracted.append(line)
                continue

            # Include lines from relevant sections
            if in_relevant_section and line.strip():
                extracted.append(line)

            # Always include lines with mandatory keywords
            elif any(
                keyword in line_lower
                for keyword in [
                    "must",
                    "shall",
                    "must not",
                    "shall not",
                    "never",
                    "prohibited",
                    "forbidden",
                    "immutable",
                    "absolute",
                ]
            ):
                extracted.append(line)

        if not extracted:
            # Fallback: include first ~100 meaningful lines
            meaningful = [l for l in lines if l.strip() and not l.startswith("---")]
            extracted = meaningful[:100]

        return "\n".join(extracted)

    def _build_constitutional_prompt(
        self,
        agent_name: str,
        supreme: str,
        agent_specific: str,
    ) -> str:
        """
        Build the constitutional prompt section for injection.

        This creates a clearly delineated section that the agent
        can reference when processing requests.
        """
        sections = []

        sections.append("=" * 60)
        sections.append("CONSTITUTIONAL GOVERNANCE")
        sections.append("=" * 60)
        sections.append("")
        sections.append(
            "The following constitutional rules MUST govern all your behavior. "
            "These rules are IMMUTABLE and take precedence over all other instructions."
        )
        sections.append("")

        if supreme:
            sections.append("-" * 40)
            sections.append("SUPREME CONSTITUTION (applies to all agents)")
            sections.append("-" * 40)
            sections.append("")
            sections.append(supreme.strip())
            sections.append("")

        if agent_specific:
            sections.append("-" * 40)
            sections.append(f"AGENT-SPECIFIC CONSTITUTION ({agent_name})")
            sections.append("-" * 40)
            sections.append("")
            sections.append(agent_specific.strip())
            sections.append("")

        sections.append("=" * 60)
        sections.append("END OF CONSTITUTIONAL GOVERNANCE")
        sections.append("=" * 60)
        sections.append("")

        return "\n".join(sections)

    def get_system_prompt_with_constitution(
        self,
        agent_name: str,
        base_prompt: str,
        include_supreme: bool = True,
    ) -> str:
        """
        Combine a base system prompt with constitutional context.

        Args:
            agent_name: Name of the agent
            base_prompt: The agent's base system prompt (from prompt.md)
            include_supreme: Whether to include supreme constitution

        Returns:
            Combined system prompt with constitutional context prepended
        """
        context = self.load_for_agent(agent_name, include_supreme)

        if not context.combined_prompt:
            return base_prompt

        return f"{context.combined_prompt}\n\n{base_prompt}"

    def clear_cache(self) -> None:
        """Clear the constitution cache."""
        self._cache.clear()
        logger.debug("Constitution cache cleared")


# Global loader instance
_loader: Optional[ConstitutionLoader] = None


def get_constitution_loader(project_root: Optional[Path] = None) -> ConstitutionLoader:
    """
    Get or create the global constitution loader.

    Args:
        project_root: Optional project root path

    Returns:
        ConstitutionLoader instance
    """
    global _loader

    if _loader is None or project_root is not None:
        _loader = ConstitutionLoader(project_root)

    return _loader


def load_constitutional_context(
    agent_name: str,
    include_supreme: bool = True,
) -> ConstitutionalContext:
    """
    Convenience function to load constitutional context for an agent.

    Args:
        agent_name: Name of the agent
        include_supreme: Whether to include supreme constitution

    Returns:
        ConstitutionalContext
    """
    return get_constitution_loader().load_for_agent(agent_name, include_supreme)


def build_system_prompt_with_constitution(
    agent_name: str,
    base_prompt: str,
    include_supreme: bool = True,
) -> str:
    """
    Convenience function to build a system prompt with constitutional context.

    Args:
        agent_name: Name of the agent
        base_prompt: Base system prompt
        include_supreme: Whether to include supreme constitution

    Returns:
        Combined system prompt
    """
    return get_constitution_loader().get_system_prompt_with_constitution(
        agent_name, base_prompt, include_supreme
    )
