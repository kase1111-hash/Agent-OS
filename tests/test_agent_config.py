"""
Tests for Agent OS Agent Configuration System
"""

import pytest
import os
import tempfile
from pathlib import Path

from src.agents.config import (
    AgentConfig,
    ModelConfig,
    ConstitutionBinding,
    ConfigLoader,
    create_default_config,
    generate_config_template,
)


class TestModelConfig:
    """Tests for ModelConfig."""

    def test_create_model_config(self):
        """Create model configuration."""
        config = ModelConfig(
            name="llama3:8b",
            endpoint="http://localhost:11434",
            temperature=0.7,
        )

        assert config.name == "llama3:8b"
        assert config.endpoint == "http://localhost:11434"
        assert config.temperature == 0.7
        assert config.context_window == 4096  # default

    def test_from_dict(self):
        """Create from dictionary."""
        data = {
            "name": "mistral:7b",
            "temperature": 0.5,
            "custom_param": "value",
        }

        config = ModelConfig.from_dict(data)

        assert config.name == "mistral:7b"
        assert config.temperature == 0.5
        assert config.extra_params["custom_param"] == "value"


class TestConstitutionBinding:
    """Tests for ConstitutionBinding."""

    def test_create_binding(self):
        """Create constitution binding."""
        binding = ConstitutionBinding(
            supreme_path=Path("CONSTITUTION.md"),
            agent_path=Path("agents/sage/constitution.md"),
        )

        assert binding.supreme_path == Path("CONSTITUTION.md")
        assert binding.agent_path == Path("agents/sage/constitution.md")

    def test_all_paths(self):
        """Get all constitution paths."""
        binding = ConstitutionBinding(
            supreme_path=Path("supreme.md"),
            agent_path=Path("agent.md"),
            role_paths=[Path("role1.md"), Path("role2.md")],
        )

        paths = binding.all_paths()

        assert len(paths) == 4
        assert Path("supreme.md") in paths
        assert Path("role1.md") in paths


class TestAgentConfig:
    """Tests for AgentConfig."""

    def test_create_config(self):
        """Create agent configuration."""
        config = AgentConfig(
            name="sage",
            version="1.0.0",
            description="Reasoning agent",
            capabilities={"reasoning", "generation"},
        )

        assert config.name == "sage"
        assert config.version == "1.0.0"
        assert "reasoning" in config.capabilities

    def test_from_dict(self):
        """Create from dictionary."""
        data = {
            "name": "whisper",
            "version": "0.1.0",
            "description": "Orchestrator",
            "capabilities": ["routing"],
            "requires_memory": True,
            "isolation_level": "process",
        }

        config = AgentConfig.from_dict(data)

        assert config.name == "whisper"
        assert config.requires_memory is True
        assert config.isolation_level == "process"

    def test_from_dict_with_model(self):
        """Create with model configuration."""
        data = {
            "name": "sage",
            "model": {
                "name": "llama3:70b",
                "temperature": 0.3,
            },
        }

        config = AgentConfig.from_dict(data)

        assert config.model is not None
        assert config.model.name == "llama3:70b"
        assert config.model.temperature == 0.3

    def test_from_dict_with_constitution(self):
        """Create with constitution binding."""
        data = {
            "name": "sage",
            "constitution": {
                "supreme": "CONSTITUTION.md",
                "agent": "agents/sage/constitution.md",
            },
        }

        config = AgentConfig.from_dict(data)

        assert config.constitution is not None
        assert config.constitution.supreme_path == Path("CONSTITUTION.md")

    def test_to_dict(self):
        """Convert to dictionary."""
        config = AgentConfig(
            name="test",
            version="1.0.0",
            description="Test",
            capabilities={"reasoning"},
        )

        d = config.to_dict()

        assert d["name"] == "test"
        assert d["version"] == "1.0.0"
        assert "reasoning" in d["capabilities"]


class TestConfigLoader:
    """Tests for ConfigLoader."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_load_yaml_config(self, temp_dir):
        """Load YAML configuration."""
        config_content = """
name: sage
version: 1.0.0
description: Reasoning agent
capabilities:
  - reasoning
  - generation
model:
  name: llama3:8b
  temperature: 0.3
"""
        config_path = temp_dir / "config.yaml"
        config_path.write_text(config_content)

        loader = ConfigLoader(temp_dir)
        config = loader.load(config_path)

        assert config.name == "sage"
        assert config.version == "1.0.0"
        assert config.model.name == "llama3:8b"

    def test_load_json_config(self, temp_dir):
        """Load JSON configuration."""
        config_content = '{"name": "whisper", "version": "1.0.0"}'
        config_path = temp_dir / "config.json"
        config_path.write_text(config_content)

        loader = ConfigLoader(temp_dir)
        config = loader.load(config_path)

        assert config.name == "whisper"

    def test_env_interpolation(self, temp_dir):
        """Environment variable interpolation."""
        os.environ["TEST_MODEL_NAME"] = "custom-model"

        config_content = """
name: test
model:
  name: ${TEST_MODEL_NAME}
"""
        config_path = temp_dir / "config.yaml"
        config_path.write_text(config_content)

        loader = ConfigLoader(temp_dir)
        config = loader.load(config_path)

        assert config.model.name == "custom-model"

        del os.environ["TEST_MODEL_NAME"]

    def test_env_interpolation_with_default(self, temp_dir):
        """Environment variable with default value."""
        config_content = """
name: test
model:
  name: ${UNDEFINED_VAR:-default-model}
"""
        config_path = temp_dir / "config.yaml"
        config_path.write_text(config_content)

        loader = ConfigLoader(temp_dir)
        config = loader.load(config_path)

        assert config.model.name == "default-model"

    def test_load_from_string(self):
        """Load configuration from string."""
        content = """
name: inline-agent
version: "0.1.0"
"""
        loader = ConfigLoader()
        config = loader.load_from_string(content, format="yaml")

        assert config.name == "inline-agent"

    def test_file_not_found(self, temp_dir):
        """File not found raises error."""
        loader = ConfigLoader(temp_dir)

        with pytest.raises(FileNotFoundError):
            loader.load("nonexistent.yaml")

    def test_unsupported_format(self, temp_dir):
        """Unsupported format raises error."""
        config_path = temp_dir / "config.xml"
        config_path.write_text("<config></config>")

        loader = ConfigLoader(temp_dir)

        with pytest.raises(ValueError):
            loader.load(config_path)


class TestConfigValidation:
    """Tests for configuration validation."""

    def test_validate_valid_config(self):
        """Valid configuration passes."""
        config = AgentConfig(
            name="sage",
            model=ModelConfig(name="llama3:8b", temperature=0.5),
        )

        loader = ConfigLoader()
        errors = loader.validate(config)

        assert len(errors) == 0

    def test_validate_missing_name(self):
        """Missing name fails validation."""
        config = AgentConfig(name="")

        loader = ConfigLoader()
        errors = loader.validate(config)

        assert len(errors) > 0
        assert any("name" in e.lower() for e in errors)

    def test_validate_invalid_temperature(self):
        """Invalid temperature fails validation."""
        config = AgentConfig(
            name="test",
            model=ModelConfig(name="test", temperature=5.0),  # Invalid
        )

        loader = ConfigLoader()
        errors = loader.validate(config)

        assert any("temperature" in e.lower() for e in errors)

    def test_validate_invalid_isolation_level(self):
        """Invalid isolation level fails validation."""
        config = AgentConfig(
            name="test",
            isolation_level="invalid",
        )

        loader = ConfigLoader()
        errors = loader.validate(config)

        assert any("isolation" in e.lower() for e in errors)


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_create_default_config(self):
        """Create default configuration."""
        config = create_default_config(
            name="sage",
            model_name="llama3:8b",
            description="Test agent",
        )

        assert config.name == "sage"
        assert config.model.name == "llama3:8b"
        assert config.description == "Test agent"

    def test_create_default_config_no_model(self):
        """Create default configuration without model."""
        config = create_default_config(name="minimal")

        assert config.name == "minimal"
        assert config.model is None

    def test_generate_config_template(self):
        """Generate configuration template."""
        template = generate_config_template(
            name="sage",
            description="Reasoning agent",
            model_name="llama3:8b",
        )

        assert "sage" in template
        assert "llama3:8b" in template
        assert "Reasoning agent" in template
