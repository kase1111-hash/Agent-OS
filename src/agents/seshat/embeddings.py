"""
Agent OS Seshat Embedding Module

Provides text embedding capabilities using sentence transformers.
Supports multiple embedding models with caching.
"""

import hashlib
import logging
import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingResult:
    """Result of an embedding operation."""

    embedding: np.ndarray
    text: str
    model: str
    dimension: int
    cached: bool = False
    generation_time_ms: float = 0.0
    text_hash: str = ""

    def to_list(self) -> List[float]:
        """Convert embedding to list for serialization."""
        return self.embedding.tolist()


@dataclass
class EmbeddingBatch:
    """Batch of embedding results."""

    embeddings: List[EmbeddingResult]
    model: str
    total_time_ms: float = 0.0

    def to_numpy(self) -> np.ndarray:
        """Stack embeddings into numpy array."""
        return np.vstack([e.embedding for e in self.embeddings])


class EmbeddingCache:
    """
    LRU cache for embeddings.

    Caches computed embeddings to avoid redundant computation.
    Uses OrderedDict for O(1) LRU operations.
    """

    def __init__(self, max_size: int = 10000):
        """
        Initialize embedding cache.

        Args:
            max_size: Maximum number of embeddings to cache
        """
        # OrderedDict provides O(1) move_to_end and popitem operations
        self._cache: OrderedDict[str, EmbeddingResult] = OrderedDict()
        self._max_size = max_size
        self._lock = threading.RLock()
        self._hits = 0
        self._misses = 0

    def get(self, text_hash: str) -> Optional[EmbeddingResult]:
        """Get embedding from cache."""
        with self._lock:
            if text_hash in self._cache:
                self._hits += 1
                # Move to end of access order - O(1) operation
                self._cache.move_to_end(text_hash)
                result = self._cache[text_hash]
                result.cached = True
                return result
            self._misses += 1
            return None

    def put(self, text_hash: str, result: EmbeddingResult) -> None:
        """Add embedding to cache."""
        with self._lock:
            if text_hash in self._cache:
                # Update and move to end
                self._cache.move_to_end(text_hash)
                return

            # Evict oldest if at capacity - O(1) with popitem(last=False)
            while len(self._cache) >= self._max_size:
                self._cache.popitem(last=False)

            self._cache[text_hash] = result

    def clear(self) -> None:
        """Clear the cache."""
        with self._lock:
            self._cache.clear()

    @property
    def hit_rate(self) -> float:
        """Get cache hit rate."""
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    @property
    def size(self) -> int:
        """Get current cache size."""
        return len(self._cache)


class EmbeddingModel:
    """
    Abstract base for embedding models.

    Implementations can use sentence-transformers, OpenAI, etc.
    """

    def __init__(self, model_name: str, dimension: int):
        self.model_name = model_name
        self.dimension = dimension
        self._initialized = False

    def initialize(self) -> bool:
        """Initialize the model."""
        raise NotImplementedError

    def embed(self, text: str) -> np.ndarray:
        """Generate embedding for text."""
        raise NotImplementedError

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for multiple texts."""
        raise NotImplementedError

    def shutdown(self) -> None:
        """Shutdown the model."""
        pass


class SentenceTransformerModel(EmbeddingModel):
    """
    Embedding model using sentence-transformers library.

    Default model: all-MiniLM-L6-v2 (384 dimensions)
    """

    # Known model dimensions
    MODEL_DIMENSIONS = {
        "all-MiniLM-L6-v2": 384,
        "all-MiniLM-L12-v2": 384,
        "all-mpnet-base-v2": 768,
        "paraphrase-MiniLM-L6-v2": 384,
        "multi-qa-MiniLM-L6-cos-v1": 384,
    }

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: Optional[str] = None,
    ):
        """
        Initialize sentence transformer model.

        Args:
            model_name: Model name from sentence-transformers hub
            device: Device to use (cuda, cpu, or None for auto)
        """
        dimension = self.MODEL_DIMENSIONS.get(model_name, 384)
        super().__init__(model_name, dimension)
        self._device = device
        self._model = None

    def initialize(self) -> bool:
        """Initialize the sentence transformer model."""
        try:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(
                self.model_name,
                device=self._device,
            )
            self._initialized = True
            logger.info(f"Initialized embedding model: {self.model_name}")
            return True

        except ImportError:
            logger.warning(
                "sentence-transformers not available. "
                "Install with: pip install sentence-transformers"
            )
            return False
        except Exception as e:
            logger.error(f"Failed to initialize embedding model: {e}")
            return False

    def embed(self, text: str) -> np.ndarray:
        """Generate embedding for text."""
        if not self._initialized or self._model is None:
            raise RuntimeError("Model not initialized")

        embedding = self._model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return embedding

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for multiple texts."""
        if not self._initialized or self._model is None:
            raise RuntimeError("Model not initialized")

        embeddings = self._model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            batch_size=32,
            show_progress_bar=False,
        )
        return embeddings

    def shutdown(self) -> None:
        """Shutdown the model."""
        self._model = None
        self._initialized = False


class MockEmbeddingModel(EmbeddingModel):
    """
    Mock embedding model for testing.

    Generates deterministic random embeddings based on text hash.
    """

    def __init__(self, dimension: int = 384):
        super().__init__("mock", dimension)
        self._initialized = True

    def initialize(self) -> bool:
        return True

    def embed(self, text: str) -> np.ndarray:
        """Generate deterministic mock embedding."""
        # Use text hash as seed for reproducibility
        seed = int(hashlib.md5(text.encode(), usedforsecurity=False).hexdigest()[:8], 16)
        rng = np.random.RandomState(seed)
        embedding = rng.randn(self.dimension).astype(np.float32)
        # Normalize
        embedding = embedding / np.linalg.norm(embedding)
        return embedding

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """Generate mock embeddings for batch."""
        return np.vstack([self.embed(text) for text in texts])


class EmbeddingEngine:
    """
    Main embedding engine with caching and batch support.

    Provides a unified interface for generating embeddings.
    """

    def __init__(
        self,
        model: Optional[EmbeddingModel] = None,
        cache_size: int = 10000,
        use_mock: bool = False,
    ):
        """
        Initialize embedding engine.

        Args:
            model: Embedding model to use
            cache_size: Maximum cache size
            use_mock: Use mock model for testing
        """
        if use_mock:
            self._model = MockEmbeddingModel()
        elif model:
            self._model = model
        else:
            self._model = SentenceTransformerModel()

        self._cache = EmbeddingCache(max_size=cache_size)
        self._lock = threading.RLock()

        # Metrics
        self._total_embeddings = 0
        self._total_time_ms = 0.0

    def initialize(self) -> bool:
        """Initialize the embedding engine."""
        return self._model.initialize()

    def shutdown(self) -> None:
        """Shutdown the engine."""
        self._model.shutdown()
        self._cache.clear()

    def embed(self, text: str, use_cache: bool = True) -> EmbeddingResult:
        """
        Generate embedding for text.

        Args:
            text: Text to embed
            use_cache: Whether to use cache

        Returns:
            EmbeddingResult
        """
        import time

        start_time = time.time()

        # Generate hash for caching
        text_hash = hashlib.sha256(text.encode()).hexdigest()[:16]

        # Check cache
        if use_cache:
            cached = self._cache.get(text_hash)
            if cached:
                return cached

        # Generate embedding
        embedding = self._model.embed(text)
        generation_time = (time.time() - start_time) * 1000

        result = EmbeddingResult(
            embedding=embedding,
            text=text,
            model=self._model.model_name,
            dimension=self._model.dimension,
            cached=False,
            generation_time_ms=generation_time,
            text_hash=text_hash,
        )

        # Update cache
        if use_cache:
            self._cache.put(text_hash, result)

        # Update metrics
        with self._lock:
            self._total_embeddings += 1
            self._total_time_ms += generation_time

        return result

    def embed_batch(
        self,
        texts: List[str],
        use_cache: bool = True,
    ) -> EmbeddingBatch:
        """
        Generate embeddings for multiple texts.

        Args:
            texts: Texts to embed
            use_cache: Whether to use cache

        Returns:
            EmbeddingBatch
        """
        import time

        start_time = time.time()

        results = []
        uncached_texts = []
        uncached_indices = []

        # Check cache for each text
        for i, text in enumerate(texts):
            text_hash = hashlib.sha256(text.encode()).hexdigest()[:16]

            if use_cache:
                cached = self._cache.get(text_hash)
                if cached:
                    results.append((i, cached))
                    continue

            uncached_texts.append(text)
            uncached_indices.append(i)

        # Batch embed uncached texts
        if uncached_texts:
            embeddings = self._model.embed_batch(uncached_texts)

            for idx, (text, embedding) in enumerate(zip(uncached_texts, embeddings)):
                text_hash = hashlib.sha256(text.encode()).hexdigest()[:16]
                result = EmbeddingResult(
                    embedding=embedding,
                    text=text,
                    model=self._model.model_name,
                    dimension=self._model.dimension,
                    cached=False,
                    text_hash=text_hash,
                )
                results.append((uncached_indices[idx], result))

                if use_cache:
                    self._cache.put(text_hash, result)

        # Sort by original index
        results.sort(key=lambda x: x[0])
        ordered_results = [r for _, r in results]

        total_time = (time.time() - start_time) * 1000

        # Update metrics
        with self._lock:
            self._total_embeddings += len(texts)
            self._total_time_ms += total_time

        return EmbeddingBatch(
            embeddings=ordered_results,
            model=self._model.model_name,
            total_time_ms=total_time,
        )

    def similarity(
        self,
        embedding1: np.ndarray,
        embedding2: np.ndarray,
    ) -> float:
        """
        Calculate cosine similarity between embeddings.

        Args:
            embedding1: First embedding
            embedding2: Second embedding

        Returns:
            Cosine similarity (-1 to 1)
        """
        dot_product = np.dot(embedding1, embedding2)
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(dot_product / (norm1 * norm2))

    def similarity_text(self, text1: str, text2: str) -> float:
        """
        Calculate similarity between two texts.

        Args:
            text1: First text
            text2: Second text

        Returns:
            Cosine similarity
        """
        e1 = self.embed(text1).embedding
        e2 = self.embed(text2).embedding
        return self.similarity(e1, e2)

    @property
    def dimension(self) -> int:
        """Get embedding dimension."""
        return self._model.dimension

    @property
    def model_name(self) -> str:
        """Get model name."""
        return self._model.model_name

    def get_metrics(self) -> Dict[str, Any]:
        """Get engine metrics."""
        with self._lock:
            avg_time = (
                self._total_time_ms / self._total_embeddings if self._total_embeddings > 0 else 0.0
            )
            return {
                "model": self._model.model_name,
                "dimension": self._model.dimension,
                "total_embeddings": self._total_embeddings,
                "avg_time_ms": round(avg_time, 2),
                "cache_size": self._cache.size,
                "cache_hit_rate": round(self._cache.hit_rate, 3),
            }


def create_embedding_engine(
    model_name: str = "all-MiniLM-L6-v2",
    cache_size: int = 10000,
    use_mock: bool = False,
) -> EmbeddingEngine:
    """
    Create and initialize an embedding engine.

    Args:
        model_name: Model name to use
        cache_size: Cache size
        use_mock: Use mock model for testing

    Returns:
        Initialized EmbeddingEngine
    """
    if use_mock:
        engine = EmbeddingEngine(use_mock=True, cache_size=cache_size)
    else:
        model = SentenceTransformerModel(model_name)
        engine = EmbeddingEngine(model=model, cache_size=cache_size)

    engine.initialize()
    return engine
