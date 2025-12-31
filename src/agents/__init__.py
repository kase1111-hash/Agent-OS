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

from .config import (
    AgentConfig,
    ConfigLoader,
    ConstitutionBinding,
    ModelConfig,
    create_default_config,
    generate_config_template,
)
from .constitution_loader import (
    ConstitutionalContext,
    ConstitutionLoader,
    build_system_prompt_with_constitution,
    get_constitution_loader,
    load_constitutional_context,
)
from .enforcement import (
    ConstitutionalEnforcer,
    EnforcementConfig,
    EnforcementMiddleware,
    ViolationRecord,
    create_enforced_agent,
)
from .interface import (
    AgentCapabilities,
    AgentInterface,
    AgentMetrics,
    AgentState,
    BaseAgent,
    CapabilityType,
    RequestValidationResult,
)
from .isolation import (
    ContainerIsolator,
    IsolatedProcessInfo,
    IsolationLevel,
    ProcessIsolator,
    ResourceLimits,
    ThreadIsolator,
    create_isolator,
)
from .loader import (
    AgentLoader,
    AgentRegistry,
    RegisteredAgent,
    create_loader,
)

# Ollama is optional (requires httpx)
try:
    from .ollama import (
        OllamaClient,
        OllamaConnectionError,
        OllamaError,
        OllamaMessage,
        OllamaModelError,
        OllamaModelInfo,
        OllamaModelManager,
        OllamaResponse,
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

# Llama.cpp is optional (requires llama-cpp-python or httpx)
try:
    from .llama_cpp import (
        LLAMA_CPP_PYTHON_AVAILABLE,
        LlamaCppClient,
        LlamaCppConnectionError,
        LlamaCppError,
        LlamaCppMessage,
        LlamaCppModelError,
        LlamaCppModelInfo,
        LlamaCppModelManager,
        LlamaCppResponse,
        create_llama_cpp_client,
    )

    LLAMA_CPP_AVAILABLE = True
except ImportError:
    LlamaCppClient = None
    LlamaCppMessage = None
    LlamaCppResponse = None
    LlamaCppModelInfo = None
    LlamaCppModelManager = None
    LlamaCppError = None
    LlamaCppConnectionError = None
    LlamaCppModelError = None
    create_llama_cpp_client = None
    LLAMA_CPP_PYTHON_AVAILABLE = False
    LLAMA_CPP_AVAILABLE = False


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
    # Constitution Loader
    "ConstitutionLoader",
    "ConstitutionalContext",
    "get_constitution_loader",
    "load_constitutional_context",
    "build_system_prompt_with_constitution",
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
    # Llama.cpp
    "LlamaCppClient",
    "LlamaCppMessage",
    "LlamaCppResponse",
    "LlamaCppModelInfo",
    "LlamaCppModelManager",
    "LlamaCppError",
    "LlamaCppConnectionError",
    "LlamaCppModelError",
    "create_llama_cpp_client",
    "LLAMA_CPP_AVAILABLE",
    "LLAMA_CPP_PYTHON_AVAILABLE",
]
