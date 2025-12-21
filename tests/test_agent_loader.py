"""
Tests for Agent OS Agent Loader and Registry
"""

import pytest
import tempfile
from pathlib import Path

from src.agents.interface import BaseAgent, AgentState, CapabilityType
from src.agents.config import AgentConfig, ModelConfig
from src.agents.loader import (
    AgentRegistry,
    AgentLoader,
    RegisteredAgent,
    create_loader,
)


class TestAgentRegistry:
    """Tests for AgentRegistry."""

    @pytest.fixture
    def registry(self):
        """Create a registry."""
        return AgentRegistry()

    @pytest.fixture
    def sample_agent(self):
        """Create a sample agent."""
        return BaseAgent(
            name="sample",
            description="Sample agent",
            capabilities={CapabilityType.REASONING},
            supported_intents=["query.*"],
        )

    @pytest.fixture
    def sample_config(self):
        """Create sample config."""
        return AgentConfig(
            name="sample",
            version="1.0.0",
            description="Sample agent",
        )

    def test_register_agent(self, registry, sample_agent, sample_config):
        """Register an agent."""
        record = registry.register(sample_agent, sample_config)

        assert record is not None
        assert record.name == "sample"
        assert record.instance == sample_agent
        assert record.is_active is False

    def test_register_duplicate_fails(self, registry, sample_agent, sample_config):
        """Registering duplicate name fails."""
        registry.register(sample_agent, sample_config)

        with pytest.raises(ValueError):
            registry.register(sample_agent, sample_config)

    def test_get_agent(self, registry, sample_agent, sample_config):
        """Get registered agent by name."""
        registry.register(sample_agent, sample_config)

        record = registry.get("sample")

        assert record is not None
        assert record.name == "sample"

    def test_get_agent_not_found(self, registry):
        """Get non-existent agent returns None."""
        record = registry.get("nonexistent")
        assert record is None

    def test_unregister_agent(self, registry, sample_agent, sample_config):
        """Unregister an agent."""
        registry.register(sample_agent, sample_config)

        result = registry.unregister("sample")

        assert result is True
        assert registry.get("sample") is None

    def test_unregister_not_found(self, registry):
        """Unregistering non-existent agent returns False."""
        result = registry.unregister("nonexistent")
        assert result is False

    def test_get_all(self, registry, sample_config):
        """Get all registered agents."""
        agent1 = BaseAgent(name="agent1")
        agent2 = BaseAgent(name="agent2")
        config1 = AgentConfig(name="agent1")
        config2 = AgentConfig(name="agent2")

        registry.register(agent1, config1)
        registry.register(agent2, config2)

        all_agents = registry.get_all()

        assert len(all_agents) == 2
        assert any(r.name == "agent1" for r in all_agents)
        assert any(r.name == "agent2" for r in all_agents)

    def test_start_agent(self, registry, sample_agent, sample_config):
        """Start an agent."""
        registry.register(sample_agent, sample_config)

        result = registry.start_agent("sample")

        assert result is True
        record = registry.get("sample")
        assert record.is_active is True
        assert sample_agent.state == AgentState.READY

    def test_start_already_active(self, registry, sample_agent, sample_config):
        """Starting already active agent returns True."""
        registry.register(sample_agent, sample_config)
        registry.start_agent("sample")

        result = registry.start_agent("sample")

        assert result is True

    def test_stop_agent(self, registry, sample_agent, sample_config):
        """Stop an agent."""
        registry.register(sample_agent, sample_config)
        registry.start_agent("sample")

        result = registry.stop_agent("sample")

        assert result is True
        record = registry.get("sample")
        assert record.is_active is False

    def test_get_active(self, registry):
        """Get active agents only."""
        agent1 = BaseAgent(name="agent1")
        agent2 = BaseAgent(name="agent2")
        config1 = AgentConfig(name="agent1")
        config2 = AgentConfig(name="agent2")

        registry.register(agent1, config1)
        registry.register(agent2, config2)
        registry.start_agent("agent1")  # Only start one

        active = registry.get_active()

        assert len(active) == 1
        assert active[0].name == "agent1"

    def test_get_by_capability(self, registry):
        """Get agents by capability."""
        reasoning_agent = BaseAgent(
            name="sage",
            capabilities={CapabilityType.REASONING},
        )
        generation_agent = BaseAgent(
            name="quill",
            capabilities={CapabilityType.GENERATION},
        )

        registry.register(reasoning_agent, AgentConfig(name="sage"))
        registry.register(generation_agent, AgentConfig(name="quill"))

        reasoning_agents = registry.get_by_capability("reasoning")

        assert len(reasoning_agents) == 1
        assert reasoning_agents[0].name == "sage"

    def test_get_by_intent(self, registry):
        """Get agents by intent pattern."""
        query_agent = BaseAgent(
            name="sage",
            supported_intents=["query.*"],
        )
        code_agent = BaseAgent(
            name="coder",
            supported_intents=["code.*"],
        )

        registry.register(query_agent, AgentConfig(name="sage"))
        registry.register(code_agent, AgentConfig(name="coder"))

        matching = registry.get_by_intent("query.factual")

        assert len(matching) == 1
        assert matching[0].name == "sage"

    def test_intent_wildcard_matching(self, registry):
        """Intent matching supports wildcards."""
        all_agent = BaseAgent(
            name="whisper",
            supported_intents=["*"],  # Handles everything
        )

        registry.register(all_agent, AgentConfig(name="whisper"))

        matching = registry.get_by_intent("any.intent.here")

        assert len(matching) == 1

    def test_start_all(self, registry):
        """Start all agents."""
        agent1 = BaseAgent(name="agent1")
        agent2 = BaseAgent(name="agent2")

        registry.register(agent1, AgentConfig(name="agent1"))
        registry.register(agent2, AgentConfig(name="agent2"))

        results = registry.start_all()

        assert results["agent1"] is True
        assert results["agent2"] is True
        assert len(registry.get_active()) == 2

    def test_stop_all(self, registry):
        """Stop all agents."""
        agent1 = BaseAgent(name="agent1")
        agent2 = BaseAgent(name="agent2")

        registry.register(agent1, AgentConfig(name="agent1"))
        registry.register(agent2, AgentConfig(name="agent2"))
        registry.start_all()

        results = registry.stop_all()

        assert results["agent1"] is True
        assert results["agent2"] is True
        assert len(registry.get_active()) == 0


class TestRegistryCallbacks:
    """Tests for registry lifecycle callbacks."""

    @pytest.fixture
    def registry(self):
        return AgentRegistry()

    def test_on_register_callback(self, registry):
        """on_register callback is invoked."""
        registered = []

        registry.register_callback("on_register", lambda r: registered.append(r))

        agent = BaseAgent(name="test")
        registry.register(agent, AgentConfig(name="test"))

        assert len(registered) == 1
        assert registered[0].name == "test"

    def test_on_start_callback(self, registry):
        """on_start callback is invoked."""
        started = []

        registry.register_callback("on_start", lambda r: started.append(r))

        agent = BaseAgent(name="test")
        registry.register(agent, AgentConfig(name="test"))
        registry.start_agent("test")

        assert len(started) == 1

    def test_on_stop_callback(self, registry):
        """on_stop callback is invoked."""
        stopped = []

        registry.register_callback("on_stop", lambda r: stopped.append(r))

        agent = BaseAgent(name="test")
        registry.register(agent, AgentConfig(name="test"))
        registry.start_agent("test")
        registry.stop_agent("test")

        assert len(stopped) == 1


class TestAgentLoader:
    """Tests for AgentLoader."""

    @pytest.fixture
    def loader(self):
        """Create an agent loader."""
        return create_loader()

    def test_create_loader(self, loader):
        """Create an agent loader."""
        assert loader is not None
        assert loader.registry is not None

    def test_register_class(self, loader):
        """Register agent class."""

        class CustomAgent(BaseAgent):
            pass

        loader.register_class("CustomAgent", CustomAgent)

        # Class should be registered
        assert "CustomAgent" in loader._agent_classes

    def test_load_from_config_with_registered_class(self, loader):
        """Load agent using registered class."""
        loader.register_class("BaseAgent", BaseAgent)

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"
            config_path.write_text("""
name: test-agent
version: "1.0.0"
custom:
  agent_type: BaseAgent
""")

            record = loader.load_from_config(config_path, agent_type="BaseAgent")

            assert record.name == "test-agent"
            assert isinstance(record.instance, BaseAgent)

    def test_load_unknown_type_fails(self, loader):
        """Loading unknown agent type fails."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"
            config_path.write_text("""
name: test
""")

            with pytest.raises(ValueError):
                loader.load_from_config(config_path, agent_type="UnknownAgent")


class TestAgentDiscovery:
    """Tests for agent auto-discovery."""

    def test_discover_agents(self):
        """Discover agents from directory."""
        loader = create_loader()

        with tempfile.TemporaryDirectory() as tmpdir:
            agents_dir = Path(tmpdir) / "agents"
            agents_dir.mkdir()

            # Create agent directory with config
            agent_dir = agents_dir / "test-agent"
            agent_dir.mkdir()
            (agent_dir / "config.yaml").write_text("""
name: test-agent
version: "1.0.0"
custom:
  agent_type: BaseAgent
""")

            discovered = loader.discover_agents(agents_dir)

            assert len(discovered) == 1
            assert discovered[0].name == "test-agent"

    def test_discover_empty_directory(self):
        """Discover from empty directory."""
        loader = create_loader()

        with tempfile.TemporaryDirectory() as tmpdir:
            agents_dir = Path(tmpdir) / "agents"
            agents_dir.mkdir()

            discovered = loader.discover_agents(agents_dir)

            assert len(discovered) == 0

    def test_discover_nonexistent_directory(self):
        """Discover from non-existent directory."""
        loader = create_loader()

        with tempfile.TemporaryDirectory() as tmpdir:
            agents_dir = Path(tmpdir) / "nonexistent"

            discovered = loader.discover_agents(agents_dir)

            assert len(discovered) == 0
