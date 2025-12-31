"""
Agent OS Seshat Agent

The Memory Agent - provides RAG-based memory and retrieval capabilities.
Named after the Egyptian goddess of wisdom, knowledge, and writing.

Capabilities:
- Semantic memory storage and retrieval
- RAG context assembly for other agents
- Consent-aware memory operations
- Memory consolidation and summarization
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Set

from src.messaging.models import FlowRequest, FlowResponse, MessageStatus

from ..interface import (
    AgentCapabilities,
    AgentState,
    BaseAgent,
    CapabilityType,
    RequestValidationResult,
)
from .consent_integration import (
    ConsentAwareConfig,
    ConsentAwareRetrievalPipeline,
    ConsentBridge,
    SeshatConsentScope,
)
from .embeddings import (
    EmbeddingEngine,
    MockEmbeddingModel,
    create_embedding_engine,
)
from .retrieval import (
    ContextType,
    HybridRetriever,
    MemoryEntry,
    RAGContext,
    RetrievalPipeline,
    RetrievalResult,
)
from .vectorstore import (
    InMemoryVectorStore,
    VectorBackend,
    VectorStoreBase,
    create_vector_store,
)

logger = logging.getLogger(__name__)


class SeshatConfig:
    """Configuration for Seshat agent."""

    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        embedding_dimension: int = 384,
        vector_backend: VectorBackend = VectorBackend.MEMORY,
        collection_name: str = "seshat_memories",
        use_mock_embeddings: bool = False,
        cache_size: int = 10000,
        default_top_k: int = 10,
        min_similarity_score: float = 0.3,
        use_hybrid_retrieval: bool = False,
        consent_enabled: bool = False,
        consent_strict: bool = False,
        max_context_tokens: int = 2000,
        **kwargs,
    ):
        self.embedding_model = embedding_model
        self.embedding_dimension = embedding_dimension
        self.vector_backend = vector_backend
        self.collection_name = collection_name
        self.use_mock_embeddings = use_mock_embeddings
        self.cache_size = cache_size
        self.default_top_k = default_top_k
        self.min_similarity_score = min_similarity_score
        self.use_hybrid_retrieval = use_hybrid_retrieval
        self.consent_enabled = consent_enabled
        self.consent_strict = consent_strict
        self.max_context_tokens = max_context_tokens
        self.extra = kwargs

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SeshatConfig":
        """Create config from dictionary."""
        # Handle vector_backend conversion
        backend = data.get("vector_backend", VectorBackend.MEMORY)
        if isinstance(backend, str):
            backend = VectorBackend[backend.upper()]
        data["vector_backend"] = backend

        return cls(**data)


class SeshatAgent(BaseAgent):
    """
    Seshat - The Memory Agent.

    Provides RAG-based memory capabilities for the Agent OS:
    - Store memories with semantic embeddings
    - Retrieve relevant context for queries
    - Assemble RAG context for LLM prompts
    - Consent-aware memory operations

    Intents handled:
    - memory.store: Store new memory
    - memory.retrieve: Retrieve relevant memories
    - memory.rag: Get RAG context for a query
    - memory.delete: Delete memory (right to forget)
    - memory.stats: Get memory statistics
    """

    # Supported intents
    INTENTS = [
        "memory.store",
        "memory.retrieve",
        "memory.rag",
        "memory.delete",
        "memory.forget",
        "memory.stats",
        "memory.consolidate",
    ]

    def __init__(self, config: Optional[SeshatConfig] = None):
        """
        Initialize Seshat agent.

        Args:
            config: Agent configuration
        """
        super().__init__(
            name="seshat",
            description="Memory agent providing RAG-based retrieval",
            version="0.1.0",
            capabilities={
                CapabilityType.RETRIEVAL,
                CapabilityType.MEMORY,
            },
            supported_intents=self.INTENTS,
        )

        self._seshat_config = config or SeshatConfig()
        self._embedding_engine: Optional[EmbeddingEngine] = None
        self._vector_store: Optional[VectorStoreBase] = None
        self._retrieval_pipeline: Optional[RetrievalPipeline] = None
        self._hybrid_retriever: Optional[HybridRetriever] = None
        self._consent_pipeline: Optional[ConsentAwareRetrievalPipeline] = None

    def initialize(self, config: Dict[str, Any]) -> bool:
        """
        Initialize Seshat with configuration.

        Args:
            config: Configuration dictionary

        Returns:
            True if initialization successful
        """
        self._do_initialize(config)

        try:
            # Merge configs
            merged_config = {**self._seshat_config.__dict__, **config}
            self._seshat_config = SeshatConfig.from_dict(merged_config)

            # Initialize embedding engine
            self._embedding_engine = create_embedding_engine(
                model_name=self._seshat_config.embedding_model,
                cache_size=self._seshat_config.cache_size,
                use_mock=self._seshat_config.use_mock_embeddings,
            )
            logger.info(f"Initialized embedding engine: {self._embedding_engine.model_name}")

            # Initialize vector store
            self._vector_store = create_vector_store(
                backend=self._seshat_config.vector_backend,
                collection_name=self._seshat_config.collection_name,
                dimension=self._seshat_config.embedding_dimension,
                **self._seshat_config.extra,
            )
            logger.info(f"Initialized vector store: {self._seshat_config.vector_backend.name}")

            # Create retrieval pipeline
            self._retrieval_pipeline = RetrievalPipeline(
                embedding_engine=self._embedding_engine,
                vector_store=self._vector_store,
            )

            # Create hybrid retriever if enabled
            if self._seshat_config.use_hybrid_retrieval:
                self._hybrid_retriever = HybridRetriever(self._retrieval_pipeline)

            # Setup consent if enabled
            if self._seshat_config.consent_enabled:
                self._setup_consent()

            self._state = AgentState.READY
            logger.info("Seshat agent initialized and ready")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize Seshat: {e}")
            self._state = AgentState.ERROR
            return False

    def validate_request(self, request: FlowRequest) -> RequestValidationResult:
        """
        Validate incoming memory request.

        Args:
            request: Incoming FlowRequest

        Returns:
            RequestValidationResult
        """
        result = RequestValidationResult(is_valid=True)

        # Check intent is supported
        if request.intent not in self.INTENTS:
            result.add_error(f"Unsupported intent: {request.intent}")
            return result

        # Run parent validation (constitutional rules)
        parent_result = super().validate_request(request)
        if not parent_result.is_valid:
            return parent_result

        # Intent-specific validation
        intent = request.intent
        metadata = request.content.metadata.model_dump() if request.content.metadata else {}

        if intent == "memory.store":
            if not request.content.prompt.strip():
                result.add_error("Content to store cannot be empty")

        elif intent == "memory.retrieve" or intent == "memory.rag":
            if not request.content.prompt.strip():
                result.add_error("Query cannot be empty")

        elif intent == "memory.delete" or intent == "memory.forget":
            # Need either memory_id or consent_id
            if not metadata.get("memory_id") and not metadata.get("consent_id"):
                result.add_error("Either memory_id or consent_id required for deletion")

        return result

    def process(self, request: FlowRequest) -> FlowResponse:
        """
        Process a memory request.

        Args:
            request: Validated FlowRequest

        Returns:
            FlowResponse with result
        """
        intent = request.intent
        content = request.content.prompt
        metadata = request.content.metadata.model_dump() if request.content.metadata else {}

        try:
            if intent == "memory.store":
                return self._handle_store(request, content, metadata)

            elif intent == "memory.retrieve":
                return self._handle_retrieve(request, content, metadata)

            elif intent == "memory.rag":
                return self._handle_rag(request, content, metadata)

            elif intent in ("memory.delete", "memory.forget"):
                return self._handle_delete(request, metadata)

            elif intent == "memory.stats":
                return self._handle_stats(request)

            elif intent == "memory.consolidate":
                return self._handle_consolidate(request, metadata)

            else:
                return request.create_response(
                    source=self.name,
                    status=MessageStatus.ERROR,
                    output="",
                    errors=[f"Unknown intent: {intent}"],
                )

        except Exception as e:
            logger.exception(f"Error processing {intent}")
            return request.create_response(
                source=self.name,
                status=MessageStatus.ERROR,
                output="",
                errors=[str(e)],
            )

    def get_capabilities(self) -> AgentCapabilities:
        """Get Seshat capabilities."""
        return AgentCapabilities(
            name=self.name,
            version=self._version,
            description=self._description,
            capabilities=self._capability_types,
            supported_intents=self.INTENTS,
            model=self._seshat_config.embedding_model if self._seshat_config else None,
            requires_constitution=True,
            requires_memory=False,  # We ARE the memory system
            can_escalate=False,
            metadata={
                "vector_backend": (
                    self._seshat_config.vector_backend.name if self._seshat_config else None
                ),
                "consent_enabled": (
                    self._seshat_config.consent_enabled if self._seshat_config else False
                ),
            },
        )

    def shutdown(self) -> bool:
        """Shutdown Seshat agent."""
        logger.info("Shutting down Seshat agent")

        try:
            if self._retrieval_pipeline:
                self._retrieval_pipeline.shutdown()

            if self._embedding_engine:
                self._embedding_engine.shutdown()

            if self._vector_store:
                self._vector_store.shutdown()

            return self._do_shutdown()

        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
            return False

    # =========================================================================
    # Direct API Methods (for use without messaging)
    # =========================================================================

    def store_memory(
        self,
        content: str,
        context_type: ContextType = ContextType.KNOWLEDGE,
        source: str = "direct",
        consent_id: Optional[str] = None,
        importance: float = 0.5,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> MemoryEntry:
        """
        Store a memory directly.

        Args:
            content: Memory content
            context_type: Type of context
            source: Source of memory
            consent_id: Associated consent
            importance: Importance (0-1)
            metadata: Additional metadata

        Returns:
            Created MemoryEntry
        """
        if not self._retrieval_pipeline:
            raise RuntimeError("Seshat not initialized")

        return self._retrieval_pipeline.store_memory(
            content=content,
            context_type=context_type,
            source=source,
            consent_id=consent_id,
            importance=importance,
            metadata=metadata,
        )

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        min_score: Optional[float] = None,
        context_types: Optional[List[ContextType]] = None,
    ) -> RetrievalResult:
        """
        Retrieve memories directly.

        Args:
            query: Query text
            top_k: Number of results
            min_score: Minimum similarity
            context_types: Filter by types

        Returns:
            RetrievalResult
        """
        if not self._retrieval_pipeline:
            raise RuntimeError("Seshat not initialized")

        top_k = top_k or self._seshat_config.default_top_k
        min_score = min_score or self._seshat_config.min_similarity_score

        if self._hybrid_retriever:
            return self._hybrid_retriever.retrieve(
                query=query,
                top_k=top_k,
                min_score=min_score,
                context_types=context_types,
            )

        return self._retrieval_pipeline.retrieve(
            query=query,
            top_k=top_k,
            min_score=min_score,
            context_types=context_types,
        )

    def get_rag_context(
        self,
        query: str,
        system_context: Optional[str] = None,
        conversation_history: Optional[str] = None,
        max_tokens: Optional[int] = None,
        **retrieve_kwargs,
    ) -> RAGContext:
        """
        Get RAG context directly.

        Args:
            query: User query
            system_context: System instructions
            conversation_history: Recent conversation
            max_tokens: Max context tokens
            **retrieve_kwargs: Retrieval parameters

        Returns:
            RAGContext for LLM
        """
        if not self._retrieval_pipeline:
            raise RuntimeError("Seshat not initialized")

        max_tokens = max_tokens or self._seshat_config.max_context_tokens

        return self._retrieval_pipeline.retrieve_for_rag(
            query=query,
            system_context=system_context,
            conversation_history=conversation_history,
            max_context_tokens=max_tokens,
            **retrieve_kwargs,
        )

    def delete_memory(self, memory_id: str) -> bool:
        """Delete a specific memory."""
        if not self._retrieval_pipeline:
            raise RuntimeError("Seshat not initialized")

        return self._retrieval_pipeline.delete_memory(memory_id)

    def forget_by_consent(self, consent_id: str) -> int:
        """Delete all memories for a consent (right to forget)."""
        if not self._retrieval_pipeline:
            raise RuntimeError("Seshat not initialized")

        return self._retrieval_pipeline.delete_by_consent(consent_id)

    def get_statistics(self) -> Dict[str, Any]:
        """Get Seshat statistics."""
        stats = {
            "agent": {
                "name": self.name,
                "state": self._state.name,
                "version": self._version,
            },
            "metrics": self._metrics.__dict__,
        }

        if self._retrieval_pipeline:
            stats["pipeline"] = self._retrieval_pipeline.get_statistics()

        if self._embedding_engine:
            stats["embedding"] = self._embedding_engine.get_metrics()

        if self._vector_store:
            stats["vector_store"] = {
                "count": self._vector_store.count(),
            }

        return stats

    # =========================================================================
    # Private Handler Methods
    # =========================================================================

    def _handle_store(
        self,
        request: FlowRequest,
        content: str,
        metadata: Dict[str, Any],
    ) -> FlowResponse:
        """Handle memory.store intent."""
        # Parse context type
        context_type_str = metadata.get("context_type", "KNOWLEDGE")
        try:
            context_type = ContextType[context_type_str.upper()]
        except KeyError:
            context_type = ContextType.KNOWLEDGE

        memory = self.store_memory(
            content=content,
            context_type=context_type,
            source=metadata.get("source", str(request.request_id)),
            consent_id=metadata.get("consent_id"),
            importance=metadata.get("importance", 0.5),
            metadata={
                k: v
                for k, v in metadata.items()
                if k not in ["context_type", "source", "consent_id", "importance"]
            },
        )

        return request.create_response(
            source=self.name,
            status=MessageStatus.SUCCESS,
            output=f"Stored memory: {memory.memory_id}",
            reasoning=f"Context type: {context_type.name}",
        )

    def _handle_retrieve(
        self,
        request: FlowRequest,
        query: str,
        metadata: Dict[str, Any],
    ) -> FlowResponse:
        """Handle memory.retrieve intent."""
        # Parse context types filter
        context_types = None
        if "context_types" in metadata:
            context_types = [ContextType[t.upper()] for t in metadata["context_types"]]

        result = self.retrieve(
            query=query,
            top_k=metadata.get("top_k"),
            min_score=metadata.get("min_score"),
            context_types=context_types,
        )

        # Format output
        output_lines = [f"Found {len(result.memories)} relevant memories:"]
        for i, (memory, score) in enumerate(zip(result.memories, result.scores)):
            output_lines.append(
                f"[{i+1}] (score: {score:.3f}, type: {memory.context_type.name}) "
                f"{memory.content[:100]}..."
            )

        return request.create_response(
            source=self.name,
            status=MessageStatus.SUCCESS,
            output="\n".join(output_lines),
            reasoning=f"Retrieved in {result.retrieval_time_ms:.1f}ms",
        )

    def _handle_rag(
        self,
        request: FlowRequest,
        query: str,
        metadata: Dict[str, Any],
    ) -> FlowResponse:
        """Handle memory.rag intent."""
        context = self.get_rag_context(
            query=query,
            system_context=metadata.get("system_context"),
            conversation_history=metadata.get("conversation_history"),
            max_tokens=metadata.get("max_tokens"),
            top_k=metadata.get("top_k"),
            min_score=metadata.get("min_score"),
        )

        return request.create_response(
            source=self.name,
            status=MessageStatus.SUCCESS,
            output=context.retrieved_context,
            reasoning=f"Retrieved {context.total_memories} memories, ~{context.context_tokens} tokens",
        )

    def _handle_delete(
        self,
        request: FlowRequest,
        metadata: Dict[str, Any],
    ) -> FlowResponse:
        """Handle memory.delete/memory.forget intent."""
        if "memory_id" in metadata:
            success = self.delete_memory(metadata["memory_id"])
            if success:
                return request.create_response(
                    source=self.name,
                    status=MessageStatus.SUCCESS,
                    output=f"Deleted memory: {metadata['memory_id']}",
                )
            else:
                return request.create_response(
                    source=self.name,
                    status=MessageStatus.ERROR,
                    output="",
                    errors=[f"Memory not found: {metadata['memory_id']}"],
                )

        elif "consent_id" in metadata:
            count = self.forget_by_consent(metadata["consent_id"])
            return request.create_response(
                source=self.name,
                status=MessageStatus.SUCCESS,
                output=f"Deleted {count} memories for consent: {metadata['consent_id']}",
                reasoning="Right to forget executed",
            )

        return request.create_response(
            source=self.name,
            status=MessageStatus.ERROR,
            output="",
            errors=["Either memory_id or consent_id required"],
        )

    def _handle_stats(self, request: FlowRequest) -> FlowResponse:
        """Handle memory.stats intent."""
        stats = self.get_statistics()

        # Format output
        lines = [
            "Seshat Memory Statistics:",
            f"  State: {stats['agent']['state']}",
            f"  Memories: {stats.get('vector_store', {}).get('count', 0)}",
        ]

        if "pipeline" in stats:
            pipeline = stats["pipeline"]
            lines.append(f"  Total stores: {pipeline.get('total_stores', 0)}")
            lines.append(f"  Total retrievals: {pipeline.get('total_retrievals', 0)}")

        if "embedding" in stats:
            emb = stats["embedding"]
            lines.append(f"  Embeddings generated: {emb.get('total_embeddings', 0)}")
            lines.append(f"  Cache hit rate: {emb.get('cache_hit_rate', 0):.1%}")

        return request.create_response(
            source=self.name,
            status=MessageStatus.SUCCESS,
            output="\n".join(lines),
        )

    def _handle_consolidate(
        self,
        request: FlowRequest,
        metadata: Dict[str, Any],
    ) -> FlowResponse:
        """Handle memory.consolidate intent."""
        # Note: consolidation requires a summarization callback
        # For now, return info about what would be consolidated

        if not self._retrieval_pipeline:
            return request.create_response(
                source=self.name,
                status=MessageStatus.ERROR,
                output="",
                errors=["Pipeline not initialized"],
            )

        stats = self._retrieval_pipeline.get_statistics()
        return request.create_response(
            source=self.name,
            status=MessageStatus.PARTIAL,
            output=f"Consolidation would process {stats['total_memories']} memories. "
            f"Provide a summarization callback to execute.",
            reasoning="Summarization callback required",
        )

    def _setup_consent(self) -> None:
        """Setup consent-aware operations."""
        # This would integrate with Memory Vault's consent system
        # For now, we just log that consent is enabled
        logger.info("Consent-aware operations enabled")

        # In full implementation, would create ConsentAwareRetrievalPipeline
        # self._consent_pipeline = create_consent_aware_pipeline(...)


def create_seshat_agent(
    use_mock: bool = True,
    **config_kwargs,
) -> SeshatAgent:
    """
    Create and initialize a Seshat agent.

    Args:
        use_mock: Use mock embeddings for testing
        **config_kwargs: Configuration options

    Returns:
        Initialized SeshatAgent
    """
    config = SeshatConfig(
        use_mock_embeddings=use_mock,
        **config_kwargs,
    )

    agent = SeshatAgent(config)
    agent.initialize({})

    return agent
