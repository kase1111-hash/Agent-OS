"""
Agent OS Seshat Memory Agent

Seshat is the Memory Agent providing RAG-based retrieval capabilities.
Named after the Egyptian goddess of wisdom, knowledge, and writing.

Main components:
- SeshatAgent: The agent implementation
- EmbeddingEngine: Text embedding with caching
- VectorStore: Similarity search backends
- RetrievalPipeline: RAG retrieval with consent
"""

from .agent import (
    SeshatAgent,
    SeshatConfig,
    create_seshat_agent,
)
from .consent_integration import (
    ConsentAwareConfig,
    ConsentAwareRetrievalPipeline,
    ConsentBridge,
    SeshatConsentScope,
    create_consent_aware_pipeline,
)
from .embeddings import (
    EmbeddingBatch,
    EmbeddingCache,
    EmbeddingEngine,
    EmbeddingModel,
    EmbeddingResult,
    MockEmbeddingModel,
    SentenceTransformerModel,
    create_embedding_engine,
)
from .retrieval import (
    ConsentVerifier,
    ContextType,
    HybridRetriever,
    MemoryEntry,
    RAGContext,
    RetrievalMode,
    RetrievalPipeline,
    RetrievalResult,
    create_retrieval_pipeline,
)
from .vectorstore import (
    ChromaDBVectorStore,
    InMemoryVectorStore,
    QdrantVectorStore,
    SearchQuery,
    SearchResult,
    VectorBackend,
    VectorDocument,
    VectorStoreBase,
    create_vector_store,
)

__all__ = [
    # Embeddings
    "EmbeddingResult",
    "EmbeddingBatch",
    "EmbeddingCache",
    "EmbeddingModel",
    "SentenceTransformerModel",
    "MockEmbeddingModel",
    "EmbeddingEngine",
    "create_embedding_engine",
    # Vector Store
    "VectorBackend",
    "VectorDocument",
    "SearchResult",
    "SearchQuery",
    "VectorStoreBase",
    "InMemoryVectorStore",
    "ChromaDBVectorStore",
    "QdrantVectorStore",
    "create_vector_store",
    # Retrieval
    "RetrievalMode",
    "ContextType",
    "MemoryEntry",
    "RetrievalResult",
    "RAGContext",
    "ConsentVerifier",
    "RetrievalPipeline",
    "HybridRetriever",
    "create_retrieval_pipeline",
    # Consent
    "SeshatConsentScope",
    "ConsentAwareConfig",
    "ConsentBridge",
    "ConsentAwareRetrievalPipeline",
    "create_consent_aware_pipeline",
    # Agent
    "SeshatConfig",
    "SeshatAgent",
    "create_seshat_agent",
]
