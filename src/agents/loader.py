"""
Agent OS Agent Loader and Registry

Provides dynamic agent loading, registration, and lifecycle management.
Supports loading agents from Python modules or configuration files.
"""

import importlib
import importlib.util
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any, Type, Callable
import logging
import threading

from .interface import AgentInterface, AgentState, AgentCapabilities
from .config import AgentConfig, ConfigLoader


logger = logging.getLogger(__name__)


@dataclass
class RegisteredAgent:
    """A registered agent with metadata."""
    name: str
    instance: AgentInterface
    config: AgentConfig
    registered_at: datetime = field(default_factory=datetime.now)
    loaded_from: Optional[str] = None  # Module path or config file
    is_active: bool = False

    @property
    def state(self) -> AgentState:
        return self.instance.state

    @property
    def capabilities(self) -> AgentCapabilities:
        return self.instance.get_capabilities()


class AgentRegistry:
    """
    Central registry for all agents in the system.

    Provides:
    - Agent registration and lookup
    - Lifecycle management (start/stop)
    - Health monitoring
    - Intent routing support
    """

    def __init__(self):
        """Initialize registry."""
        self._agents: Dict[str, RegisteredAgent] = {}
        self._lock = threading.RLock()
        self._lifecycle_callbacks: Dict[str, List[Callable]] = {
            "on_register": [],
            "on_unregister": [],
            "on_start": [],
            "on_stop": [],
        }

    def register(
        self,
        agent: AgentInterface,
        config: AgentConfig,
        loaded_from: Optional[str] = None,
    ) -> RegisteredAgent:
        """
        Register an agent with the registry.

        Args:
            agent: Agent instance
            config: Agent configuration
            loaded_from: Source of agent (for tracking)

        Returns:
            RegisteredAgent record

        Raises:
            ValueError: If agent already registered
        """
        with self._lock:
            if agent.name in self._agents:
                raise ValueError(f"Agent already registered: {agent.name}")

            record = RegisteredAgent(
                name=agent.name,
                instance=agent,
                config=config,
                loaded_from=loaded_from,
            )
            self._agents[agent.name] = record

            # Notify callbacks
            for callback in self._lifecycle_callbacks["on_register"]:
                try:
                    callback(record)
                except Exception as e:
                    logger.warning(f"Register callback error: {e}")

            logger.info(f"Registered agent: {agent.name}")
            return record

    def unregister(self, name: str) -> bool:
        """
        Unregister an agent.

        Args:
            name: Agent name

        Returns:
            True if unregistered, False if not found
        """
        with self._lock:
            if name not in self._agents:
                return False

            record = self._agents[name]

            # Stop if running
            if record.is_active:
                self.stop_agent(name)

            del self._agents[name]

            # Notify callbacks
            for callback in self._lifecycle_callbacks["on_unregister"]:
                try:
                    callback(record)
                except Exception as e:
                    logger.warning(f"Unregister callback error: {e}")

            logger.info(f"Unregistered agent: {name}")
            return True

    def get(self, name: str) -> Optional[RegisteredAgent]:
        """Get a registered agent by name."""
        with self._lock:
            return self._agents.get(name)

    def get_agent(self, name: str) -> Optional[AgentInterface]:
        """Get agent instance by name."""
        record = self.get(name)
        return record.instance if record else None

    def get_all(self) -> List[RegisteredAgent]:
        """Get all registered agents."""
        with self._lock:
            return list(self._agents.values())

    def get_active(self) -> List[RegisteredAgent]:
        """Get all active agents."""
        with self._lock:
            return [r for r in self._agents.values() if r.is_active]

    def get_by_capability(self, capability: str) -> List[RegisteredAgent]:
        """Get agents with a specific capability."""
        with self._lock:
            return [
                r for r in self._agents.values()
                if capability in {c.value for c in r.capabilities.capabilities}
            ]

    def get_by_intent(self, intent: str) -> List[RegisteredAgent]:
        """
        Get agents that handle a specific intent.

        Args:
            intent: Intent string (e.g., "query.factual")

        Returns:
            List of agents handling this intent
        """
        with self._lock:
            matches = []
            for record in self._agents.values():
                for pattern in record.capabilities.supported_intents:
                    if self._intent_matches(intent, pattern):
                        matches.append(record)
                        break
            return matches

    def start_agent(self, name: str) -> bool:
        """
        Start an agent (initialize and make ready).

        Args:
            name: Agent name

        Returns:
            True if started successfully
        """
        with self._lock:
            record = self._agents.get(name)
            if not record:
                logger.error(f"Agent not found: {name}")
                return False

            if record.is_active:
                logger.warning(f"Agent already active: {name}")
                return True

            # Initialize agent
            config_dict = record.config.to_dict()
            if record.instance.initialize(config_dict):
                record.is_active = True

                # Notify callbacks
                for callback in self._lifecycle_callbacks["on_start"]:
                    try:
                        callback(record)
                    except Exception as e:
                        logger.warning(f"Start callback error: {e}")

                logger.info(f"Started agent: {name}")
                return True
            else:
                logger.error(f"Failed to start agent: {name}")
                return False

    def stop_agent(self, name: str) -> bool:
        """
        Stop an agent (shutdown).

        Args:
            name: Agent name

        Returns:
            True if stopped successfully
        """
        with self._lock:
            record = self._agents.get(name)
            if not record:
                return False

            if not record.is_active:
                return True

            # Shutdown agent
            if record.instance.shutdown():
                record.is_active = False

                # Notify callbacks
                for callback in self._lifecycle_callbacks["on_stop"]:
                    try:
                        callback(record)
                    except Exception as e:
                        logger.warning(f"Stop callback error: {e}")

                logger.info(f"Stopped agent: {name}")
                return True
            else:
                logger.error(f"Failed to stop agent: {name}")
                return False

    def start_all(self) -> Dict[str, bool]:
        """Start all registered agents."""
        results = {}
        for name in list(self._agents.keys()):
            results[name] = self.start_agent(name)
        return results

    def stop_all(self) -> Dict[str, bool]:
        """Stop all active agents."""
        results = {}
        for name in list(self._agents.keys()):
            results[name] = self.stop_agent(name)
        return results

    def register_callback(
        self,
        event: str,
        callback: Callable[[RegisteredAgent], None]
    ) -> None:
        """
        Register a lifecycle callback.

        Args:
            event: Event name (on_register, on_unregister, on_start, on_stop)
            callback: Callback function
        """
        if event in self._lifecycle_callbacks:
            self._lifecycle_callbacks[event].append(callback)
        else:
            raise ValueError(f"Unknown event: {event}")

    def _intent_matches(self, intent: str, pattern: str) -> bool:
        """
        Check if intent matches a pattern.

        Supports wildcards: "query.*" matches "query.factual"
        """
        if pattern == "*":
            return True

        if pattern.endswith(".*"):
            prefix = pattern[:-2]
            return intent.startswith(prefix + ".") or intent == prefix

        return intent == pattern


class AgentLoader:
    """
    Dynamic agent loader supporting multiple loading strategies.

    Supports:
    - Loading from Python modules
    - Loading from configuration files
    - Auto-discovery from directories
    """

    def __init__(
        self,
        registry: Optional[AgentRegistry] = None,
        config_loader: Optional[ConfigLoader] = None,
    ):
        """
        Initialize loader.

        Args:
            registry: Agent registry (creates new if not provided)
            config_loader: Configuration loader
        """
        self.registry = registry or AgentRegistry()
        self.config_loader = config_loader or ConfigLoader()
        self._agent_classes: Dict[str, Type[AgentInterface]] = {}

    def register_class(
        self,
        name: str,
        agent_class: Type[AgentInterface]
    ) -> None:
        """
        Register an agent class for later instantiation.

        Args:
            name: Agent type name
            agent_class: Agent class
        """
        self._agent_classes[name] = agent_class
        logger.debug(f"Registered agent class: {name}")

    def load_from_module(
        self,
        module_path: str,
        class_name: str,
        config: AgentConfig,
    ) -> RegisteredAgent:
        """
        Load an agent from a Python module.

        Args:
            module_path: Python module path (e.g., "agents.sage")
            class_name: Agent class name
            config: Agent configuration

        Returns:
            RegisteredAgent record

        Raises:
            ImportError: If module cannot be loaded
            AttributeError: If class not found
        """
        # Import module
        module = importlib.import_module(module_path)

        # Get class
        agent_class = getattr(module, class_name)

        if not issubclass(agent_class, AgentInterface):
            raise TypeError(f"{class_name} is not an AgentInterface subclass")

        # Instantiate
        agent = agent_class(config.name)

        # Register
        return self.registry.register(
            agent,
            config,
            loaded_from=f"{module_path}.{class_name}",
        )

    def load_from_file(
        self,
        file_path: Path,
        class_name: str,
        config: AgentConfig,
    ) -> RegisteredAgent:
        """
        Load an agent from a Python file.

        Args:
            file_path: Path to Python file
            class_name: Agent class name
            config: Agent configuration

        Returns:
            RegisteredAgent record
        """
        # Load module from file
        spec = importlib.util.spec_from_file_location(
            f"agent_{config.name}",
            file_path
        )
        if not spec or not spec.loader:
            raise ImportError(f"Cannot load module from: {file_path}")

        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)

        # Get class
        agent_class = getattr(module, class_name)

        if not issubclass(agent_class, AgentInterface):
            raise TypeError(f"{class_name} is not an AgentInterface subclass")

        # Instantiate
        agent = agent_class(config.name)

        # Register
        return self.registry.register(
            agent,
            config,
            loaded_from=str(file_path),
        )

    def load_from_config(
        self,
        config_path: Path,
        agent_type: Optional[str] = None,
    ) -> RegisteredAgent:
        """
        Load an agent from a configuration file.

        Requires agent class to be pre-registered with register_class().

        Args:
            config_path: Path to configuration file
            agent_type: Agent type (class name) to use

        Returns:
            RegisteredAgent record
        """
        # Load config
        config = self.config_loader.load(config_path)

        # Determine agent type
        type_name = agent_type or config.custom.get("agent_type", "BaseAgent")

        # Get class
        if type_name not in self._agent_classes:
            raise ValueError(f"Unknown agent type: {type_name}")

        agent_class = self._agent_classes[type_name]

        # Instantiate
        agent = agent_class(config.name)

        # Register
        return self.registry.register(
            agent,
            config,
            loaded_from=str(config_path),
        )

    def discover_agents(
        self,
        agents_dir: Path,
        config_pattern: str = "config.yaml",
    ) -> List[RegisteredAgent]:
        """
        Discover and load agents from a directory.

        Looks for agent directories containing config files.

        Args:
            agents_dir: Directory to search
            config_pattern: Configuration file name pattern

        Returns:
            List of loaded agents
        """
        loaded = []

        if not agents_dir.exists():
            logger.warning(f"Agents directory not found: {agents_dir}")
            return loaded

        for agent_dir in agents_dir.iterdir():
            if not agent_dir.is_dir():
                continue

            config_path = agent_dir / config_pattern
            if not config_path.exists():
                continue

            try:
                # Check for agent.py
                agent_file = agent_dir / "agent.py"
                if agent_file.exists():
                    # Load config
                    config = self.config_loader.load(config_path)

                    # Load from file with class name from config
                    class_name = config.custom.get("class_name", "Agent")
                    record = self.load_from_file(agent_file, class_name, config)
                    loaded.append(record)
                else:
                    # Load from config only
                    record = self.load_from_config(config_path)
                    loaded.append(record)

                logger.info(f"Discovered agent: {agent_dir.name}")

            except Exception as e:
                logger.error(f"Failed to load agent from {agent_dir}: {e}")

        return loaded


def create_loader(base_path: Optional[Path] = None) -> AgentLoader:
    """
    Create a configured agent loader.

    Args:
        base_path: Base path for relative paths

    Returns:
        Configured AgentLoader
    """
    from .interface import BaseAgent

    loader = AgentLoader(
        config_loader=ConfigLoader(base_path or Path.cwd())
    )

    # Register built-in agent types
    loader.register_class("BaseAgent", BaseAgent)

    return loader
