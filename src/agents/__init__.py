"""
Agent OS Agents Module

Provides the agent base interface and infrastructure for Agent OS:
- AgentInterface: Abstract base class for all agents
- Agent configuration and loading
- Constitutional boundary enforcement
- Ollama LLM integration
- Process isolation

All agents MUST implement the AgentInterface.
"""

from .interface import (
    AgentInterface,
    BaseAgent,
    AgentState,
    CapabilityType,
    AgentCapabilities,
    RequestValidationResult,
    AgentMetrics,
)
from .config import (
    AgentConfig,
    ModelConfig,
    ConstitutionBinding,
    ConfigLoader,
    create_default_config,
    generate_config_template,
)
from .loader import (
    AgentRegistry,
    AgentLoader,
    RegisteredAgent,
    create_loader,
)
from .enforcement import (
    ConstitutionalEnforcer,
    EnforcementMiddleware,
    EnforcementConfig,
    ViolationRecord,
    create_enforced_agent,
)
from .isolation import (
    IsolationLevel,
    ResourceLimits,
    ProcessIsolator,
    ThreadIsolator,
    ContainerIsolator,
    IsolatedProcessInfo,
    create_isolator,
)

# Ollama is optional (requires httpx)
try:
    from .ollama import (
        OllamaClient,
        OllamaMessage,
        OllamaResponse,
        OllamaModelInfo,
        OllamaModelManager,
        OllamaError,
        OllamaConnectionError,
        OllamaModelError,
        create_ollama_client,
    )
    OLLAMA_AVAILABLE = True
except ImportError:
    OllamaClient = None
    OllamaMessage = None
    OllamaResponse = None
    OllamaModelInfo = None
    OllamaModelManager = None
    OllamaError = None
    OllamaConnectionError = None
    OllamaModelError = None
    create_ollama_client = None
    OLLAMA_AVAILABLE = False


__all__ = [
    # Interface
    "AgentInterface",
    "BaseAgent",
    "AgentState",
    "CapabilityType",
    "AgentCapabilities",
    "RequestValidationResult",
    "AgentMetrics",
    # Config
    "AgentConfig",
    "ModelConfig",
    "ConstitutionBinding",
    "ConfigLoader",
    "create_default_config",
    "generate_config_template",
    # Loader
    "AgentRegistry",
    "AgentLoader",
    "RegisteredAgent",
    "create_loader",
    # Enforcement
    "ConstitutionalEnforcer",
    "EnforcementMiddleware",
    "EnforcementConfig",
    "ViolationRecord",
    "create_enforced_agent",
    # Isolation
    "IsolationLevel",
    "ResourceLimits",
    "ProcessIsolator",
    "ThreadIsolator",
    "ContainerIsolator",
    "IsolatedProcessInfo",
    "create_isolator",
    # Ollama
    "OllamaClient",
    "OllamaMessage",
    "OllamaResponse",
    "OllamaModelInfo",
    "OllamaModelManager",
    "OllamaError",
    "OllamaConnectionError",
    "OllamaModelError",
    "create_ollama_client",
    "OLLAMA_AVAILABLE",
]
