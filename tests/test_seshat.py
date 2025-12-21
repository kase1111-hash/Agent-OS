"""
Tests for Agent OS Seshat (Memory Agent)
"""

import pytest
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from src.agents.seshat.embeddings import (
    EmbeddingResult,
    EmbeddingBatch,
    EmbeddingCache,
    EmbeddingModel,
    MockEmbeddingModel,
    EmbeddingEngine,
    create_embedding_engine,
)
from src.agents.seshat.vectorstore import (
    VectorBackend,
    VectorDocument,
    SearchResult,
    SearchQuery,
    InMemoryVectorStore,
    create_vector_store,
)
from src.agents.seshat.retrieval import (
    ContextType,
    MemoryEntry,
    RetrievalResult,
    RAGContext,
    ConsentVerifier,
    RetrievalPipeline,
    HybridRetriever,
)
from src.agents.seshat.agent import (
    SeshatConfig,
    SeshatAgent,
    create_seshat_agent,
)
from src.messaging.models import create_request, MessageStatus


# =============================================================================
# Embedding Tests
# =============================================================================

class TestEmbeddingCache:
    """Tests for embedding cache."""

    @pytest.fixture
    def cache(self):
        return EmbeddingCache(max_size=100)

    def test_cache_put_get(self, cache):
        """Test basic cache operations."""
        result = EmbeddingResult(
            embedding=np.array([1.0, 2.0, 3.0]),
            text="test",
            model="mock",
            dimension=3,
        )

        cache.put("hash123", result)
        cached = cache.get("hash123")

        assert cached is not None
        assert cached.text == "test"
        assert cached.cached is True

    def test_cache_miss(self, cache):
        """Test cache miss."""
        result = cache.get("nonexistent")
        assert result is None

    def test_cache_eviction(self):
        """Test LRU eviction."""
        cache = EmbeddingCache(max_size=3)

        for i in range(5):
            result = EmbeddingResult(
                embedding=np.array([float(i)]),
                text=f"text{i}",
                model="mock",
                dimension=1,
            )
            cache.put(f"hash{i}", result)

        # First two should be evicted
        assert cache.get("hash0") is None
        assert cache.get("hash1") is None
        assert cache.get("hash2") is not None
        assert cache.get("hash3") is not None
        assert cache.get("hash4") is not None

    def test_cache_hit_rate(self, cache):
        """Test hit rate calculation."""
        result = EmbeddingResult(
            embedding=np.array([1.0]),
            text="test",
            model="mock",
            dimension=1,
        )
        cache.put("hash1", result)

        # 2 hits, 1 miss
        cache.get("hash1")
        cache.get("hash1")
        cache.get("nonexistent")

        assert cache.hit_rate == pytest.approx(2/3, rel=0.01)


class TestMockEmbeddingModel:
    """Tests for mock embedding model."""

    @pytest.fixture
    def model(self):
        return MockEmbeddingModel(dimension=384)

    def test_embed_returns_correct_dimension(self, model):
        """Test embedding dimension."""
        embedding = model.embed("test text")

        assert embedding.shape == (384,)

    def test_embed_is_normalized(self, model):
        """Test embedding is normalized."""
        embedding = model.embed("test text")

        norm = np.linalg.norm(embedding)
        assert norm == pytest.approx(1.0, rel=0.01)

    def test_embed_is_deterministic(self, model):
        """Test same text gives same embedding."""
        embedding1 = model.embed("test text")
        embedding2 = model.embed("test text")

        np.testing.assert_array_almost_equal(embedding1, embedding2)

    def test_embed_different_text_different_embedding(self, model):
        """Test different text gives different embedding."""
        embedding1 = model.embed("hello")
        embedding2 = model.embed("world")

        assert not np.allclose(embedding1, embedding2)

    def test_embed_batch(self, model):
        """Test batch embedding."""
        texts = ["hello", "world", "test"]
        embeddings = model.embed_batch(texts)

        assert embeddings.shape == (3, 384)


class TestEmbeddingEngine:
    """Tests for embedding engine."""

    @pytest.fixture
    def engine(self):
        return EmbeddingEngine(use_mock=True, cache_size=100)

    def test_embed_single(self, engine):
        """Test single text embedding."""
        result = engine.embed("test text")

        assert isinstance(result, EmbeddingResult)
        assert result.text == "test text"
        assert result.embedding.shape == (384,)
        assert result.model == "mock"

    def test_embed_uses_cache(self, engine):
        """Test caching behavior."""
        # Use unique text to avoid cache from other tests
        unique_text = f"unique_cache_test_{id(engine)}"
        result1 = engine.embed(unique_text, use_cache=False)
        # Now get from cache
        result2 = engine.embed(unique_text, use_cache=True)

        # First should not be cached since we disabled it
        assert result1.cached is False
        # Second should be from cache but since first wasn't put in cache, it won't be either
        # Let's test differently
        engine.embed("cache_me")  # First embed, puts in cache
        result_cached = engine.embed("cache_me")  # Second embed, from cache
        assert result_cached.cached is True

    def test_embed_batch(self, engine):
        """Test batch embedding."""
        texts = ["one", "two", "three"]
        result = engine.embed_batch(texts)

        assert isinstance(result, EmbeddingBatch)
        assert len(result.embeddings) == 3

    def test_similarity(self, engine):
        """Test similarity calculation."""
        e1 = np.array([1.0, 0.0, 0.0])
        e2 = np.array([1.0, 0.0, 0.0])
        e3 = np.array([0.0, 1.0, 0.0])

        assert engine.similarity(e1, e2) == pytest.approx(1.0, rel=0.01)
        assert engine.similarity(e1, e3) == pytest.approx(0.0, rel=0.01)

    def test_similarity_text(self, engine):
        """Test text similarity."""
        # Same text should have similarity 1.0
        sim = engine.similarity_text("hello world", "hello world")
        assert sim == pytest.approx(1.0, rel=0.01)

    def test_get_metrics(self, engine):
        """Test metrics collection."""
        engine.embed("test1")
        engine.embed("test2")
        engine.embed("test1")  # Cache hit

        metrics = engine.get_metrics()

        assert metrics["model"] == "mock"
        assert metrics["total_embeddings"] >= 2


# =============================================================================
# Vector Store Tests
# =============================================================================

class TestInMemoryVectorStore:
    """Tests for in-memory vector store."""

    @pytest.fixture
    def store(self):
        store = InMemoryVectorStore("test", dimension=3)
        store.initialize()
        return store

    def test_add_and_get(self, store):
        """Test adding and retrieving documents."""
        doc = VectorDocument(
            doc_id="doc1",
            content="test content",
            embedding=np.array([1.0, 0.0, 0.0]),
            metadata={"key": "value"},
        )

        store.add(doc)
        retrieved = store.get("doc1")

        assert retrieved is not None
        assert retrieved.content == "test content"
        assert retrieved.metadata["key"] == "value"

    def test_search(self, store):
        """Test similarity search."""
        # Add documents
        store.add(VectorDocument(
            doc_id="doc1", content="first", embedding=np.array([1.0, 0.0, 0.0])
        ))
        store.add(VectorDocument(
            doc_id="doc2", content="second", embedding=np.array([0.0, 1.0, 0.0])
        ))
        store.add(VectorDocument(
            doc_id="doc3", content="third", embedding=np.array([0.9, 0.1, 0.0])
        ))

        # Search for similar to [1, 0, 0]
        query = SearchQuery(embedding=np.array([1.0, 0.0, 0.0]), top_k=2)
        results = store.search(query)

        assert len(results) == 2
        assert results[0].doc_id == "doc1"  # Most similar
        assert results[0].score > results[1].score

    def test_search_with_consent_filter(self, store):
        """Test search with consent filtering."""
        store.add(VectorDocument(
            doc_id="doc1", content="first",
            embedding=np.array([1.0, 0.0, 0.0]),
            consent_id="consent_a"
        ))
        store.add(VectorDocument(
            doc_id="doc2", content="second",
            embedding=np.array([1.0, 0.0, 0.0]),
            consent_id="consent_b"
        ))

        # Search with consent filter
        query = SearchQuery(
            embedding=np.array([1.0, 0.0, 0.0]),
            consent_ids=["consent_a"],
        )
        results = store.search(query)

        assert len(results) == 1
        assert results[0].doc_id == "doc1"

    def test_delete(self, store):
        """Test document deletion."""
        store.add(VectorDocument(
            doc_id="doc1", content="test", embedding=np.array([1.0, 0.0, 0.0])
        ))

        assert store.count() == 1
        store.delete("doc1")
        assert store.count() == 0

    def test_delete_by_consent(self, store):
        """Test deletion by consent ID."""
        store.add(VectorDocument(
            doc_id="doc1", content="one",
            embedding=np.array([1.0, 0.0, 0.0]),
            consent_id="consent_x"
        ))
        store.add(VectorDocument(
            doc_id="doc2", content="two",
            embedding=np.array([0.0, 1.0, 0.0]),
            consent_id="consent_x"
        ))
        store.add(VectorDocument(
            doc_id="doc3", content="three",
            embedding=np.array([0.0, 0.0, 1.0]),
            consent_id="consent_y"
        ))

        deleted = store.delete_by_consent("consent_x")

        assert deleted == 2
        assert store.count() == 1

    def test_clear(self, store):
        """Test clearing store."""
        store.add(VectorDocument(
            doc_id="doc1", content="test", embedding=np.array([1.0, 0.0, 0.0])
        ))
        store.add(VectorDocument(
            doc_id="doc2", content="test2", embedding=np.array([0.0, 1.0, 0.0])
        ))

        store.clear()
        assert store.count() == 0


class TestCreateVectorStore:
    """Tests for vector store factory."""

    def test_create_memory_store(self):
        """Test creating in-memory store."""
        store = create_vector_store(
            backend=VectorBackend.MEMORY,
            collection_name="test",
            dimension=384,
        )

        assert isinstance(store, InMemoryVectorStore)
        assert store._initialized


# =============================================================================
# Retrieval Pipeline Tests
# =============================================================================

class TestConsentVerifier:
    """Tests for consent verifier."""

    def test_no_consent_required(self):
        """Test access without consent requirement."""
        verifier = ConsentVerifier()
        assert verifier.verify(None, "accessor") is True

    def test_verify_with_callback(self):
        """Test verification with callback."""
        callback = Mock(return_value=True)
        verifier = ConsentVerifier(verify_callback=callback)

        result = verifier.verify("consent_123", "accessor")

        assert result is True
        callback.assert_called_once_with("consent_123", "accessor")

    def test_verify_denied(self):
        """Test verification denied."""
        callback = Mock(return_value=False)
        verifier = ConsentVerifier(verify_callback=callback)

        result = verifier.verify("consent_123", "accessor")
        assert result is False


class TestRetrievalPipeline:
    """Tests for retrieval pipeline."""

    @pytest.fixture
    def pipeline(self):
        engine = EmbeddingEngine(use_mock=True)
        store = InMemoryVectorStore("test", dimension=384)
        store.initialize()

        return RetrievalPipeline(
            embedding_engine=engine,
            vector_store=store,
        )

    def test_store_memory(self, pipeline):
        """Test storing memory."""
        memory = pipeline.store_memory(
            content="This is a test memory",
            context_type=ContextType.KNOWLEDGE,
            source="test",
        )

        assert memory.memory_id.startswith("mem_")
        assert memory.content == "This is a test memory"
        assert memory.context_type == ContextType.KNOWLEDGE

    def test_retrieve(self, pipeline):
        """Test memory retrieval."""
        # Store some memories
        pipeline.store_memory("Python is a programming language", ContextType.KNOWLEDGE, "test")
        pipeline.store_memory("JavaScript is used for web", ContextType.KNOWLEDGE, "test")
        pipeline.store_memory("Cats are cute animals", ContextType.KNOWLEDGE, "test")

        # Retrieve
        result = pipeline.retrieve("What programming languages exist?", top_k=2)

        assert isinstance(result, RetrievalResult)
        assert len(result.memories) <= 2
        assert result.query == "What programming languages exist?"

    def test_retrieve_with_context_type_filter(self, pipeline):
        """Test retrieval with context type filter."""
        pipeline.store_memory("Conversation memory", ContextType.CONVERSATION, "test")
        pipeline.store_memory("Knowledge memory", ContextType.KNOWLEDGE, "test")

        result = pipeline.retrieve(
            "memory",
            context_types=[ContextType.KNOWLEDGE],
        )

        for memory in result.memories:
            assert memory.context_type == ContextType.KNOWLEDGE

    def test_retrieve_for_rag(self, pipeline):
        """Test RAG context retrieval."""
        pipeline.store_memory("The sky is blue", ContextType.KNOWLEDGE, "test")

        context = pipeline.retrieve_for_rag(
            query="What color is the sky?",
            system_context="You are a helpful assistant.",
        )

        assert isinstance(context, RAGContext)
        assert context.query == "What color is the sky?"
        assert context.system_context is not None

    def test_delete_memory(self, pipeline):
        """Test memory deletion."""
        memory = pipeline.store_memory("To be deleted", ContextType.KNOWLEDGE, "test")

        assert pipeline.get_memory(memory.memory_id) is not None
        pipeline.delete_memory(memory.memory_id)
        assert pipeline.get_memory(memory.memory_id) is None

    def test_delete_by_consent(self, pipeline):
        """Test deletion by consent (right to forget)."""
        pipeline.store_memory("Memory 1", ContextType.KNOWLEDGE, "test", consent_id="user_1")
        pipeline.store_memory("Memory 2", ContextType.KNOWLEDGE, "test", consent_id="user_1")
        pipeline.store_memory("Memory 3", ContextType.KNOWLEDGE, "test", consent_id="user_2")

        count = pipeline.delete_by_consent("user_1")

        assert count == 2

    def test_get_statistics(self, pipeline):
        """Test statistics collection."""
        pipeline.store_memory("Test memory", ContextType.KNOWLEDGE, "test")
        pipeline.retrieve("test query")

        stats = pipeline.get_statistics()

        assert stats["total_memories"] >= 1
        assert stats["total_stores"] >= 1
        assert stats["total_retrievals"] >= 1


class TestHybridRetriever:
    """Tests for hybrid retriever."""

    @pytest.fixture
    def retriever(self):
        engine = EmbeddingEngine(use_mock=True)
        store = InMemoryVectorStore("test", dimension=384)
        store.initialize()

        pipeline = RetrievalPipeline(engine, store)
        return HybridRetriever(pipeline)

    def test_hybrid_retrieve(self, retriever):
        """Test hybrid retrieval."""
        retriever._pipeline.store_memory(
            "Python programming basics",
            ContextType.KNOWLEDGE,
            "test"
        )
        retriever._pipeline.store_memory(
            "Advanced Python techniques",
            ContextType.KNOWLEDGE,
            "test"
        )

        result = retriever.retrieve("Python programming", top_k=2)

        assert isinstance(result, RetrievalResult)


# =============================================================================
# Seshat Agent Tests
# =============================================================================

class TestSeshatConfig:
    """Tests for Seshat configuration."""

    def test_default_config(self):
        """Test default configuration."""
        config = SeshatConfig()

        assert config.embedding_model == "all-MiniLM-L6-v2"
        assert config.vector_backend == VectorBackend.MEMORY
        assert config.use_mock_embeddings is False

    def test_from_dict(self):
        """Test creating config from dictionary."""
        config = SeshatConfig.from_dict({
            "embedding_model": "custom-model",
            "vector_backend": "MEMORY",
            "use_mock_embeddings": True,
        })

        assert config.embedding_model == "custom-model"
        assert config.use_mock_embeddings is True


class TestSeshatAgent:
    """Tests for Seshat agent."""

    @pytest.fixture
    def agent(self):
        """Create initialized Seshat agent."""
        agent = SeshatAgent(SeshatConfig(use_mock_embeddings=True))
        agent.initialize({})
        return agent

    def test_initialize(self, agent):
        """Test agent initialization."""
        assert agent.is_ready
        assert agent.name == "seshat"

    def test_get_capabilities(self, agent):
        """Test getting capabilities."""
        caps = agent.get_capabilities()

        assert caps.name == "seshat"
        assert "memory.store" in caps.supported_intents
        assert "memory.retrieve" in caps.supported_intents

    def test_store_memory_direct(self, agent):
        """Test storing memory directly."""
        memory = agent.store_memory(
            content="Test memory content",
            context_type=ContextType.KNOWLEDGE,
        )

        assert memory.content == "Test memory content"

    def test_retrieve_direct(self, agent):
        """Test retrieving directly."""
        agent.store_memory("Python is great", ContextType.KNOWLEDGE)

        result = agent.retrieve("What is Python?")

        assert isinstance(result, RetrievalResult)

    def test_get_rag_context_direct(self, agent):
        """Test getting RAG context directly."""
        agent.store_memory("The capital of France is Paris", ContextType.KNOWLEDGE)

        context = agent.get_rag_context("What is the capital of France?")

        assert isinstance(context, RAGContext)

    def test_delete_memory_direct(self, agent):
        """Test deleting memory directly."""
        memory = agent.store_memory("To delete", ContextType.KNOWLEDGE)

        result = agent.delete_memory(memory.memory_id)
        assert result is True

    def test_get_statistics(self, agent):
        """Test getting statistics."""
        stats = agent.get_statistics()

        assert "agent" in stats
        assert stats["agent"]["name"] == "seshat"

    def test_shutdown(self, agent):
        """Test agent shutdown."""
        result = agent.shutdown()
        assert result is True


class TestSeshatAgentMessaging:
    """Tests for Seshat agent message handling."""

    @pytest.fixture
    def agent(self):
        """Create initialized Seshat agent."""
        agent = SeshatAgent(SeshatConfig(use_mock_embeddings=True))
        agent.initialize({})
        return agent

    def test_validate_store_request(self, agent):
        """Test validating store request."""
        request = create_request(
            source="user",
            destination="seshat",
            prompt="This is content to store",
            intent="memory.store",
        )

        result = agent.validate_request(request)
        assert result.is_valid

    def test_validate_empty_store_rejected(self, agent):
        """Test that empty store content is rejected."""
        request = create_request(
            source="user",
            destination="seshat",
            prompt="   ",
            intent="memory.store",
        )

        result = agent.validate_request(request)
        assert not result.is_valid

    def test_validate_unsupported_intent(self, agent):
        """Test unsupported intent is rejected."""
        request = create_request(
            source="user",
            destination="seshat",
            prompt="test",
            intent="unknown.intent",
        )

        result = agent.validate_request(request)
        assert not result.is_valid

    def test_handle_store_request(self, agent):
        """Test handling store request."""
        request = create_request(
            source="user",
            destination="seshat",
            prompt="Remember this important fact",
            intent="memory.store",
        )

        response = agent.handle_request(request)

        assert response.status == MessageStatus.SUCCESS
        assert "Stored memory:" in response.content.output

    def test_handle_retrieve_request(self, agent):
        """Test handling retrieve request."""
        # First store something
        agent.store_memory("Python is a programming language", ContextType.KNOWLEDGE)

        request = create_request(
            source="user",
            destination="seshat",
            prompt="What is Python?",
            intent="memory.retrieve",
        )

        response = agent.handle_request(request)

        assert response.status == MessageStatus.SUCCESS

    def test_handle_rag_request(self, agent):
        """Test handling RAG request."""
        agent.store_memory("The Earth is round", ContextType.KNOWLEDGE)

        request = create_request(
            source="user",
            destination="seshat",
            prompt="What shape is the Earth?",
            intent="memory.rag",
        )

        response = agent.handle_request(request)

        assert response.status == MessageStatus.SUCCESS

    def test_handle_stats_request(self, agent):
        """Test handling stats request."""
        request = create_request(
            source="user",
            destination="seshat",
            prompt="",
            intent="memory.stats",
        )

        response = agent.handle_request(request)

        assert response.status == MessageStatus.SUCCESS
        assert "Seshat Memory Statistics:" in response.content.output


class TestCreateSeshatAgent:
    """Tests for agent factory function."""

    def test_create_with_mock(self):
        """Test creating agent with mock embeddings."""
        agent = create_seshat_agent(use_mock=True)

        assert agent.is_ready
        assert agent._seshat_config.use_mock_embeddings is True

    def test_create_with_config(self):
        """Test creating agent with custom config."""
        agent = create_seshat_agent(
            use_mock=True,
            collection_name="custom_collection",
        )

        assert agent._seshat_config.collection_name == "custom_collection"


# =============================================================================
# Integration Tests
# =============================================================================

class TestSeshatIntegration:
    """Integration tests for Seshat."""

    @pytest.fixture
    def agent(self):
        """Create agent for integration tests."""
        return create_seshat_agent(use_mock=True)

    def test_full_memory_lifecycle(self, agent):
        """Test complete memory lifecycle."""
        # Store
        memory = agent.store_memory(
            content="Important project deadline is Friday",
            context_type=ContextType.EPISODIC,
            source="calendar",
            importance=0.9,
        )
        assert memory is not None

        # Retrieve with very low min_score to ensure we get results with mock embeddings
        # (mock embeddings generate random vectors, so similarity may be low)
        result = agent.retrieve(
            "Important project deadline is Friday",  # Use same text for high similarity
            min_score=0.0,  # Accept any score
        )
        # With mock embeddings matching text will have similarity ~1.0
        assert len(result.memories) > 0

        # Get RAG context - use the same text for deterministic mock embeddings
        context = agent.get_rag_context(
            "Important project deadline is Friday",  # Same text for similarity
            system_context="You are a helpful assistant.",
            min_score=0.0,
        )
        assert context.total_memories > 0

        # Delete
        success = agent.delete_memory(memory.memory_id)
        assert success is True

    def test_consent_based_deletion(self, agent):
        """Test right-to-forget functionality."""
        # Store with consent
        agent.store_memory("User preference 1", ContextType.KNOWLEDGE, "user", consent_id="user_consent")
        agent.store_memory("User preference 2", ContextType.KNOWLEDGE, "user", consent_id="user_consent")

        # Forget
        count = agent.forget_by_consent("user_consent")
        assert count == 2

    def test_multiple_context_types(self, agent):
        """Test storing different context types."""
        agent.store_memory("Chat message", ContextType.CONVERSATION, "chat")
        agent.store_memory("Fact about Python", ContextType.KNOWLEDGE, "docs")
        agent.store_memory("Meeting yesterday", ContextType.EPISODIC, "calendar")
        agent.store_memory("How to deploy", ContextType.PROCEDURAL, "docs")

        stats = agent.get_statistics()
        assert stats["pipeline"]["total_memories"] == 4
