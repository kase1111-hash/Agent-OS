"""
Agent OS Seshat Vector Store Module

Provides vector storage and similarity search abstraction.
Supports ChromaDB and Qdrant backends with a unified interface.
"""

import logging
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple, Union
from enum import Enum, auto
import numpy as np

from .embeddings import EmbeddingEngine, EmbeddingResult

logger = logging.getLogger(__name__)


class VectorBackend(Enum):
    """Supported vector store backends."""
    MEMORY = auto()  # In-memory for testing
    CHROMADB = auto()
    QDRANT = auto()


@dataclass
class VectorDocument:
    """Document stored in vector store."""
    doc_id: str
    content: str
    embedding: np.ndarray
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    consent_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "content": self.content,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "consent_id": self.consent_id,
        }


@dataclass
class SearchResult:
    """Result from vector similarity search."""
    doc_id: str
    content: str
    score: float  # Similarity score (higher is more similar)
    metadata: Dict[str, Any] = field(default_factory=dict)
    consent_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "content": self.content,
            "score": self.score,
            "metadata": self.metadata,
            "consent_id": self.consent_id,
        }


@dataclass
class SearchQuery:
    """Query for vector search."""
    text: Optional[str] = None
    embedding: Optional[np.ndarray] = None
    top_k: int = 10
    min_score: float = 0.0
    filters: Dict[str, Any] = field(default_factory=dict)
    consent_ids: Optional[List[str]] = None  # Filter by consent


class VectorStoreBase(ABC):
    """Abstract base class for vector stores."""

    def __init__(self, collection_name: str, dimension: int):
        self.collection_name = collection_name
        self.dimension = dimension
        self._initialized = False

    @abstractmethod
    def initialize(self) -> bool:
        """Initialize the vector store."""
        pass

    @abstractmethod
    def add(self, document: VectorDocument) -> bool:
        """Add a document to the store."""
        pass

    @abstractmethod
    def add_batch(self, documents: List[VectorDocument]) -> int:
        """Add multiple documents. Returns count added."""
        pass

    @abstractmethod
    def search(self, query: SearchQuery) -> List[SearchResult]:
        """Search for similar documents."""
        pass

    @abstractmethod
    def get(self, doc_id: str) -> Optional[VectorDocument]:
        """Get document by ID."""
        pass

    @abstractmethod
    def delete(self, doc_id: str) -> bool:
        """Delete document by ID."""
        pass

    @abstractmethod
    def delete_by_consent(self, consent_id: str) -> int:
        """Delete all documents with consent ID. Returns count deleted."""
        pass

    @abstractmethod
    def count(self) -> int:
        """Get total document count."""
        pass

    @abstractmethod
    def clear(self) -> None:
        """Clear all documents."""
        pass

    @abstractmethod
    def shutdown(self) -> None:
        """Shutdown the store."""
        pass


class InMemoryVectorStore(VectorStoreBase):
    """
    In-memory vector store for testing.

    Uses brute-force similarity search with numpy.
    """

    def __init__(self, collection_name: str, dimension: int = 384):
        super().__init__(collection_name, dimension)
        self._documents: Dict[str, VectorDocument] = {}
        self._embeddings: Dict[str, np.ndarray] = {}
        self._lock = threading.RLock()

    def initialize(self) -> bool:
        self._initialized = True
        logger.info(f"Initialized in-memory vector store: {self.collection_name}")
        return True

    def add(self, document: VectorDocument) -> bool:
        if document.embedding.shape[0] != self.dimension:
            raise ValueError(
                f"Embedding dimension mismatch: {document.embedding.shape[0]} vs {self.dimension}"
            )

        with self._lock:
            self._documents[document.doc_id] = document
            # Normalize embedding for cosine similarity
            norm = np.linalg.norm(document.embedding)
            if norm > 0:
                self._embeddings[document.doc_id] = document.embedding / norm
            else:
                self._embeddings[document.doc_id] = document.embedding

        return True

    def add_batch(self, documents: List[VectorDocument]) -> int:
        count = 0
        for doc in documents:
            if self.add(doc):
                count += 1
        return count

    def search(self, query: SearchQuery) -> List[SearchResult]:
        if query.embedding is None:
            raise ValueError("Query must have embedding")

        with self._lock:
            if not self._embeddings:
                return []

            # Normalize query embedding
            query_norm = np.linalg.norm(query.embedding)
            if query_norm > 0:
                query_vec = query.embedding / query_norm
            else:
                query_vec = query.embedding

            results = []
            for doc_id, doc_emb in self._embeddings.items():
                doc = self._documents[doc_id]

                # Apply consent filter
                if query.consent_ids is not None:
                    if doc.consent_id not in query.consent_ids:
                        continue

                # Apply metadata filters
                if query.filters:
                    match = True
                    for key, value in query.filters.items():
                        if doc.metadata.get(key) != value:
                            match = False
                            break
                    if not match:
                        continue

                # Calculate cosine similarity
                score = float(np.dot(query_vec, doc_emb))

                if score >= query.min_score:
                    results.append(SearchResult(
                        doc_id=doc_id,
                        content=doc.content,
                        score=score,
                        metadata=doc.metadata,
                        consent_id=doc.consent_id,
                    ))

            # Sort by score descending
            results.sort(key=lambda r: r.score, reverse=True)
            return results[:query.top_k]

    def get(self, doc_id: str) -> Optional[VectorDocument]:
        with self._lock:
            return self._documents.get(doc_id)

    def delete(self, doc_id: str) -> bool:
        with self._lock:
            if doc_id in self._documents:
                del self._documents[doc_id]
                del self._embeddings[doc_id]
                return True
            return False

    def delete_by_consent(self, consent_id: str) -> int:
        with self._lock:
            to_delete = [
                doc_id for doc_id, doc in self._documents.items()
                if doc.consent_id == consent_id
            ]
            for doc_id in to_delete:
                del self._documents[doc_id]
                del self._embeddings[doc_id]
            return len(to_delete)

    def count(self) -> int:
        with self._lock:
            return len(self._documents)

    def clear(self) -> None:
        with self._lock:
            self._documents.clear()
            self._embeddings.clear()

    def shutdown(self) -> None:
        self.clear()
        self._initialized = False


class ChromaDBVectorStore(VectorStoreBase):
    """
    ChromaDB vector store implementation.

    Requires: pip install chromadb
    """

    def __init__(
        self,
        collection_name: str,
        dimension: int = 384,
        persist_directory: Optional[str] = None,
    ):
        super().__init__(collection_name, dimension)
        self._persist_directory = persist_directory
        self._client = None
        self._collection = None

    def initialize(self) -> bool:
        try:
            import chromadb
            from chromadb.config import Settings

            if self._persist_directory:
                self._client = chromadb.PersistentClient(
                    path=self._persist_directory,
                    settings=Settings(anonymized_telemetry=False),
                )
            else:
                self._client = chromadb.Client(
                    Settings(anonymized_telemetry=False)
                )

            self._collection = self._client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
            self._initialized = True
            logger.info(f"Initialized ChromaDB store: {self.collection_name}")
            return True

        except ImportError:
            logger.warning(
                "ChromaDB not available. Install with: pip install chromadb"
            )
            return False
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
            return False

    def add(self, document: VectorDocument) -> bool:
        if not self._initialized or self._collection is None:
            raise RuntimeError("Store not initialized")

        try:
            metadata = document.metadata.copy()
            metadata["created_at"] = document.created_at.isoformat()
            if document.consent_id:
                metadata["consent_id"] = document.consent_id

            self._collection.add(
                ids=[document.doc_id],
                embeddings=[document.embedding.tolist()],
                documents=[document.content],
                metadatas=[metadata],
            )
            return True
        except Exception as e:
            logger.error(f"Failed to add document: {e}")
            return False

    def add_batch(self, documents: List[VectorDocument]) -> int:
        if not self._initialized or self._collection is None:
            raise RuntimeError("Store not initialized")

        try:
            ids = [doc.doc_id for doc in documents]
            embeddings = [doc.embedding.tolist() for doc in documents]
            contents = [doc.content for doc in documents]
            metadatas = []
            for doc in documents:
                meta = doc.metadata.copy()
                meta["created_at"] = doc.created_at.isoformat()
                if doc.consent_id:
                    meta["consent_id"] = doc.consent_id
                metadatas.append(meta)

            self._collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=contents,
                metadatas=metadatas,
            )
            return len(documents)
        except Exception as e:
            logger.error(f"Failed to add batch: {e}")
            return 0

    def search(self, query: SearchQuery) -> List[SearchResult]:
        if not self._initialized or self._collection is None:
            raise RuntimeError("Store not initialized")

        if query.embedding is None:
            raise ValueError("Query must have embedding")

        try:
            # Build where clause for filters
            where = None
            if query.filters or query.consent_ids:
                where_clauses = []
                for key, value in query.filters.items():
                    where_clauses.append({key: value})
                if query.consent_ids:
                    where_clauses.append(
                        {"consent_id": {"$in": query.consent_ids}}
                    )
                if len(where_clauses) == 1:
                    where = where_clauses[0]
                elif len(where_clauses) > 1:
                    where = {"$and": where_clauses}

            results = self._collection.query(
                query_embeddings=[query.embedding.tolist()],
                n_results=query.top_k,
                where=where,
            )

            search_results = []
            if results["ids"] and results["ids"][0]:
                for i, doc_id in enumerate(results["ids"][0]):
                    # ChromaDB returns distance, convert to similarity
                    distance = results["distances"][0][i] if results["distances"] else 0
                    score = 1 - distance  # Convert distance to similarity

                    if score >= query.min_score:
                        metadata = results["metadatas"][0][i] if results["metadatas"] else {}
                        content = results["documents"][0][i] if results["documents"] else ""

                        search_results.append(SearchResult(
                            doc_id=doc_id,
                            content=content,
                            score=score,
                            metadata={k: v for k, v in metadata.items()
                                     if k not in ["created_at", "consent_id"]},
                            consent_id=metadata.get("consent_id"),
                        ))

            return search_results

        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []

    def get(self, doc_id: str) -> Optional[VectorDocument]:
        if not self._initialized or self._collection is None:
            raise RuntimeError("Store not initialized")

        try:
            result = self._collection.get(
                ids=[doc_id],
                include=["embeddings", "documents", "metadatas"],
            )

            if result["ids"]:
                metadata = result["metadatas"][0] if result["metadatas"] else {}
                return VectorDocument(
                    doc_id=doc_id,
                    content=result["documents"][0] if result["documents"] else "",
                    embedding=np.array(result["embeddings"][0]) if result["embeddings"] else np.array([]),
                    metadata={k: v for k, v in metadata.items()
                             if k not in ["created_at", "consent_id"]},
                    consent_id=metadata.get("consent_id"),
                )
            return None

        except Exception as e:
            logger.error(f"Get failed: {e}")
            return None

    def delete(self, doc_id: str) -> bool:
        if not self._initialized or self._collection is None:
            raise RuntimeError("Store not initialized")

        try:
            self._collection.delete(ids=[doc_id])
            return True
        except Exception as e:
            logger.error(f"Delete failed: {e}")
            return False

    def delete_by_consent(self, consent_id: str) -> int:
        if not self._initialized or self._collection is None:
            raise RuntimeError("Store not initialized")

        try:
            # Get all documents with this consent_id
            result = self._collection.get(
                where={"consent_id": consent_id},
                include=["metadatas"],
            )

            if result["ids"]:
                self._collection.delete(ids=result["ids"])
                return len(result["ids"])
            return 0

        except Exception as e:
            logger.error(f"Delete by consent failed: {e}")
            return 0

    def count(self) -> int:
        if not self._initialized or self._collection is None:
            return 0
        return self._collection.count()

    def clear(self) -> None:
        if self._initialized and self._client:
            try:
                self._client.delete_collection(self.collection_name)
                self._collection = self._client.create_collection(
                    name=self.collection_name,
                    metadata={"hnsw:space": "cosine"},
                )
            except Exception as e:
                logger.error(f"Clear failed: {e}")

    def shutdown(self) -> None:
        self._collection = None
        self._client = None
        self._initialized = False


class QdrantVectorStore(VectorStoreBase):
    """
    Qdrant vector store implementation.

    Requires: pip install qdrant-client
    """

    def __init__(
        self,
        collection_name: str,
        dimension: int = 384,
        host: str = "localhost",
        port: int = 6333,
        path: Optional[str] = None,  # For local file-based storage
    ):
        super().__init__(collection_name, dimension)
        self._host = host
        self._port = port
        self._path = path
        self._client = None

    def initialize(self) -> bool:
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import (
                VectorParams,
                Distance,
            )

            if self._path:
                self._client = QdrantClient(path=self._path)
            else:
                self._client = QdrantClient(host=self._host, port=self._port)

            # Check if collection exists
            collections = self._client.get_collections().collections
            exists = any(c.name == self.collection_name for c in collections)

            if not exists:
                self._client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.dimension,
                        distance=Distance.COSINE,
                    ),
                )

            self._initialized = True
            logger.info(f"Initialized Qdrant store: {self.collection_name}")
            return True

        except ImportError:
            logger.warning(
                "Qdrant client not available. Install with: pip install qdrant-client"
            )
            return False
        except Exception as e:
            logger.error(f"Failed to initialize Qdrant: {e}")
            return False

    def add(self, document: VectorDocument) -> bool:
        if not self._initialized or self._client is None:
            raise RuntimeError("Store not initialized")

        try:
            from qdrant_client.models import PointStruct

            payload = document.metadata.copy()
            payload["content"] = document.content
            payload["created_at"] = document.created_at.isoformat()
            if document.consent_id:
                payload["consent_id"] = document.consent_id

            # Qdrant needs numeric IDs, use hash of doc_id
            point_id = abs(hash(document.doc_id)) % (10 ** 18)

            self._client.upsert(
                collection_name=self.collection_name,
                points=[
                    PointStruct(
                        id=point_id,
                        vector=document.embedding.tolist(),
                        payload={"doc_id": document.doc_id, **payload},
                    )
                ],
            )
            return True
        except Exception as e:
            logger.error(f"Failed to add document: {e}")
            return False

    def add_batch(self, documents: List[VectorDocument]) -> int:
        if not self._initialized or self._client is None:
            raise RuntimeError("Store not initialized")

        try:
            from qdrant_client.models import PointStruct

            points = []
            for doc in documents:
                payload = doc.metadata.copy()
                payload["content"] = doc.content
                payload["doc_id"] = doc.doc_id
                payload["created_at"] = doc.created_at.isoformat()
                if doc.consent_id:
                    payload["consent_id"] = doc.consent_id

                point_id = abs(hash(doc.doc_id)) % (10 ** 18)
                points.append(
                    PointStruct(
                        id=point_id,
                        vector=doc.embedding.tolist(),
                        payload=payload,
                    )
                )

            self._client.upsert(
                collection_name=self.collection_name,
                points=points,
            )
            return len(documents)
        except Exception as e:
            logger.error(f"Failed to add batch: {e}")
            return 0

    def search(self, query: SearchQuery) -> List[SearchResult]:
        if not self._initialized or self._client is None:
            raise RuntimeError("Store not initialized")

        if query.embedding is None:
            raise ValueError("Query must have embedding")

        try:
            from qdrant_client.models import Filter, FieldCondition, MatchValue

            # Build filter
            must_conditions = []
            for key, value in query.filters.items():
                must_conditions.append(
                    FieldCondition(key=key, match=MatchValue(value=value))
                )

            if query.consent_ids:
                # Match any of the consent_ids
                for consent_id in query.consent_ids:
                    must_conditions.append(
                        FieldCondition(
                            key="consent_id",
                            match=MatchValue(value=consent_id)
                        )
                    )

            query_filter = Filter(must=must_conditions) if must_conditions else None

            results = self._client.search(
                collection_name=self.collection_name,
                query_vector=query.embedding.tolist(),
                limit=query.top_k,
                query_filter=query_filter,
                score_threshold=query.min_score,
            )

            search_results = []
            for hit in results:
                payload = hit.payload or {}
                search_results.append(SearchResult(
                    doc_id=payload.get("doc_id", str(hit.id)),
                    content=payload.get("content", ""),
                    score=hit.score,
                    metadata={k: v for k, v in payload.items()
                             if k not in ["doc_id", "content", "created_at", "consent_id"]},
                    consent_id=payload.get("consent_id"),
                ))

            return search_results

        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []

    def get(self, doc_id: str) -> Optional[VectorDocument]:
        if not self._initialized or self._client is None:
            raise RuntimeError("Store not initialized")

        try:
            from qdrant_client.models import Filter, FieldCondition, MatchValue

            results = self._client.scroll(
                collection_name=self.collection_name,
                scroll_filter=Filter(
                    must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))]
                ),
                limit=1,
                with_vectors=True,
            )

            if results[0]:
                point = results[0][0]
                payload = point.payload or {}
                return VectorDocument(
                    doc_id=doc_id,
                    content=payload.get("content", ""),
                    embedding=np.array(point.vector),
                    metadata={k: v for k, v in payload.items()
                             if k not in ["doc_id", "content", "created_at", "consent_id"]},
                    consent_id=payload.get("consent_id"),
                )
            return None

        except Exception as e:
            logger.error(f"Get failed: {e}")
            return None

    def delete(self, doc_id: str) -> bool:
        if not self._initialized or self._client is None:
            raise RuntimeError("Store not initialized")

        try:
            from qdrant_client.models import Filter, FieldCondition, MatchValue

            self._client.delete(
                collection_name=self.collection_name,
                points_selector=Filter(
                    must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))]
                ),
            )
            return True
        except Exception as e:
            logger.error(f"Delete failed: {e}")
            return False

    def delete_by_consent(self, consent_id: str) -> int:
        if not self._initialized or self._client is None:
            raise RuntimeError("Store not initialized")

        try:
            from qdrant_client.models import Filter, FieldCondition, MatchValue

            # Get count before deletion
            results = self._client.scroll(
                collection_name=self.collection_name,
                scroll_filter=Filter(
                    must=[FieldCondition(key="consent_id", match=MatchValue(value=consent_id))]
                ),
                limit=10000,
            )
            count = len(results[0]) if results[0] else 0

            if count > 0:
                self._client.delete(
                    collection_name=self.collection_name,
                    points_selector=Filter(
                        must=[FieldCondition(key="consent_id", match=MatchValue(value=consent_id))]
                    ),
                )

            return count

        except Exception as e:
            logger.error(f"Delete by consent failed: {e}")
            return 0

    def count(self) -> int:
        if not self._initialized or self._client is None:
            return 0

        try:
            info = self._client.get_collection(self.collection_name)
            return info.points_count
        except Exception:
            return 0

    def clear(self) -> None:
        if self._initialized and self._client:
            try:
                from qdrant_client.models import VectorParams, Distance

                self._client.delete_collection(self.collection_name)
                self._client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.dimension,
                        distance=Distance.COSINE,
                    ),
                )
            except Exception as e:
                logger.error(f"Clear failed: {e}")

    def shutdown(self) -> None:
        if self._client:
            self._client.close()
        self._client = None
        self._initialized = False


def create_vector_store(
    backend: VectorBackend,
    collection_name: str,
    dimension: int = 384,
    **kwargs,
) -> VectorStoreBase:
    """
    Create a vector store instance.

    Args:
        backend: Vector store backend to use
        collection_name: Name of the collection
        dimension: Embedding dimension
        **kwargs: Backend-specific arguments

    Returns:
        Initialized vector store
    """
    if backend == VectorBackend.MEMORY:
        store = InMemoryVectorStore(collection_name, dimension)
    elif backend == VectorBackend.CHROMADB:
        store = ChromaDBVectorStore(
            collection_name,
            dimension,
            persist_directory=kwargs.get("persist_directory"),
        )
    elif backend == VectorBackend.QDRANT:
        store = QdrantVectorStore(
            collection_name,
            dimension,
            host=kwargs.get("host", "localhost"),
            port=kwargs.get("port", 6333),
            path=kwargs.get("path"),
        )
    else:
        raise ValueError(f"Unsupported backend: {backend}")

    store.initialize()
    return store
