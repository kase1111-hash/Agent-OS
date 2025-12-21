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

from .embeddings import (
    EmbeddingResult,
    EmbeddingBatch,
    EmbeddingCache,
    EmbeddingModel,
    SentenceTransformerModel,
    MockEmbeddingModel,
    EmbeddingEngine,
    create_embedding_engine,
)

from .vectorstore import (
    VectorBackend,
    VectorDocument,
    SearchResult,
    SearchQuery,
    VectorStoreBase,
    InMemoryVectorStore,
    ChromaDBVectorStore,
    QdrantVectorStore,
    create_vector_store,
)

from .retrieval import (
    RetrievalMode,
    ContextType,
    MemoryEntry,
    RetrievalResult,
    RAGContext,
    ConsentVerifier,
    RetrievalPipeline,
    HybridRetriever,
    create_retrieval_pipeline,
)

from .consent_integration import (
    SeshatConsentScope,
    ConsentAwareConfig,
    ConsentBridge,
    ConsentAwareRetrievalPipeline,
    create_consent_aware_pipeline,
)

from .agent import (
    SeshatConfig,
    SeshatAgent,
    create_seshat_agent,
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
