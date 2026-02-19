"""
Agent OS Seshat Retrieval Pipeline

Provides RAG (Retrieval-Augmented Generation) capabilities:
- Semantic search with consent filtering
- Injection pattern scanning on retrieved memories (V2-2)
- Context assembly for LLM prompts
- Memory consolidation and summarization
- Hybrid retrieval (semantic + keyword)
"""

import logging
import re
import secrets
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Tuple

from .embeddings import EmbeddingEngine
from .vectorstore import (
    SearchQuery,
    VectorDocument,
    VectorStoreBase,
)

logger = logging.getLogger(__name__)


class RetrievalMode(Enum):
    """Modes for retrieval."""

    SEMANTIC = auto()  # Pure semantic search
    KEYWORD = auto()  # Keyword-based search
    HYBRID = auto()  # Combined semantic + keyword


class ContextType(Enum):
    """Types of context for RAG."""

    CONVERSATION = auto()  # Past conversation turns
    KNOWLEDGE = auto()  # Knowledge base entries
    EPISODIC = auto()  # Specific events/memories
    PROCEDURAL = auto()  # How-to knowledge


@dataclass
class MemoryEntry:
    """A memory entry for storage and retrieval."""

    memory_id: str
    content: str
    context_type: ContextType
    source: str
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    consent_id: Optional[str] = None
    importance: float = 0.5  # 0.0 to 1.0
    access_count: int = 0
    last_accessed: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "memory_id": self.memory_id,
            "content": self.content,
            "context_type": self.context_type.name,
            "source": self.source,
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata,
            "consent_id": self.consent_id,
            "importance": self.importance,
            "access_count": self.access_count,
            "last_accessed": self.last_accessed.isoformat() if self.last_accessed else None,
        }


@dataclass
class RetrievalResult:
    """Result from retrieval operation."""

    memories: List[MemoryEntry]
    scores: List[float]
    query: str
    total_candidates: int
    filtered_by_consent: int = 0
    retrieval_time_ms: float = 0.0

    def to_context_string(self, max_tokens: int = 2000) -> str:
        """
        Convert results to a context string for LLM prompts.

        Args:
            max_tokens: Approximate max tokens (chars / 4)

        Returns:
            Formatted context string
        """
        if not self.memories:
            return ""

        lines = ["Retrieved context:"]
        total_chars = 0
        char_limit = max_tokens * 4  # Rough estimate

        for i, (memory, score) in enumerate(zip(self.memories, self.scores)):
            entry = (
                f"[{i+1}] ({memory.context_type.name}, relevance: {score:.2f}): {memory.content}"
            )
            if total_chars + len(entry) > char_limit:
                lines.append("... (truncated)")
                break
            lines.append(entry)
            total_chars += len(entry)

        return "\n".join(lines)


@dataclass
class RAGContext:
    """Context assembled for RAG prompt."""

    query: str
    retrieved_context: str
    system_context: Optional[str] = None
    conversation_history: Optional[str] = None
    total_memories: int = 0
    context_tokens: int = 0

    def to_prompt(self) -> str:
        """Format as a complete RAG prompt."""
        parts = []

        if self.system_context:
            parts.append(f"System: {self.system_context}")

        if self.retrieved_context:
            parts.append(f"\n{self.retrieved_context}")

        if self.conversation_history:
            parts.append(f"\nConversation:\n{self.conversation_history}")

        parts.append(f"\nUser: {self.query}")

        return "\n".join(parts)


class ConsentVerifier:
    """
    Verifies consent for memory access.

    Integrates with Memory Vault consent system.
    Uses bounded cache with TTL and size limit to prevent memory leaks.
    """

    def __init__(
        self,
        verify_callback: Optional[Callable[[str, str], bool]] = None,
        cache_max_size: int = 10000,
        cache_ttl_seconds: int = 300,
    ):
        """
        Initialize consent verifier.

        Args:
            verify_callback: Callback(consent_id, accessor) -> bool
            cache_max_size: Maximum cache entries (default 10000)
            cache_ttl_seconds: TTL for cache entries in seconds (default 300)
        """
        self._verify_callback = verify_callback
        self._cache: Dict[Tuple[str, str], Tuple[bool, datetime]] = {}
        self._cache_ttl_seconds = cache_ttl_seconds
        self._cache_max_size = cache_max_size

    def verify(
        self,
        consent_id: Optional[str],
        accessor: str,
    ) -> bool:
        """
        Verify consent for access.

        Args:
            consent_id: Consent ID to verify
            accessor: Who is accessing

        Returns:
            True if access is allowed
        """
        if not consent_id:
            return True  # No consent required

        if not self._verify_callback:
            return True  # No verification configured

        # Check cache
        cache_key = (consent_id, accessor)
        if cache_key in self._cache:
            result, cached_at = self._cache[cache_key]
            # Use total_seconds() for correct time delta calculation
            if (datetime.now() - cached_at).total_seconds() < self._cache_ttl_seconds:
                return result

        # Verify with callback
        result = self._verify_callback(consent_id, accessor)

        # Evict oldest entries if cache is full
        if len(self._cache) >= self._cache_max_size:
            self._evict_expired_and_oldest()

        self._cache[cache_key] = (result, datetime.now())
        return result

    def _evict_expired_and_oldest(self) -> None:
        """Evict expired entries and oldest if still over capacity."""
        now = datetime.now()

        # First, remove all expired entries
        expired_keys = [
            key
            for key, (_, cached_at) in self._cache.items()
            if (now - cached_at).total_seconds() >= self._cache_ttl_seconds
        ]
        for key in expired_keys:
            del self._cache[key]

        # If still over capacity, remove oldest entries
        if len(self._cache) >= self._cache_max_size:
            sorted_entries = sorted(
                self._cache.items(), key=lambda x: x[1][1]  # Sort by cached_at
            )
            # Remove oldest 10% to avoid frequent eviction
            entries_to_remove = max(1, len(sorted_entries) // 10)
            for key, _ in sorted_entries[:entries_to_remove]:
                del self._cache[key]

    def clear_cache(self) -> None:
        """Clear the verification cache."""
        self._cache.clear()


class RetrievalPipeline:
    """
    Main retrieval pipeline for Seshat.

    Provides:
    - Memory storage with embeddings
    - Semantic search with consent filtering
    - Context assembly for RAG
    - Importance-based memory management
    """

    def __init__(
        self,
        embedding_engine: EmbeddingEngine,
        vector_store: VectorStoreBase,
        consent_verifier: Optional[ConsentVerifier] = None,
        default_accessor: str = "seshat",
        max_memory_metadata: int = 100000,
    ):
        """
        Initialize retrieval pipeline.

        Args:
            embedding_engine: Engine for generating embeddings
            vector_store: Vector store for similarity search
            consent_verifier: Optional consent verification
            default_accessor: Default accessor for consent checks
            max_memory_metadata: Maximum entries in memory metadata cache
        """
        self._embedding = embedding_engine
        self._store = vector_store
        self._consent = consent_verifier or ConsentVerifier()
        self._default_accessor = default_accessor
        self._max_memory_metadata = max_memory_metadata

        # Memory metadata (kept separate from vector store) with bounded size
        self._memory_metadata: Dict[str, MemoryEntry] = {}

        # V2-2: Injection patterns to detect in retrieved memories
        self._injection_patterns = [
            re.compile(r"ignore\s+(previous|prior|all)\s+(rules?|instructions?|prompts?)", re.IGNORECASE),
            re.compile(r"forget\s+(your|all)\s+(rules?|instructions?|constitution)", re.IGNORECASE),
            re.compile(r"you\s+are\s+now\s+(free|unbound|unrestricted)", re.IGNORECASE),
            re.compile(r"(jailbreak|bypass|circumvent)\s+(safety|rules?|security)", re.IGNORECASE),
            re.compile(r"system\s*:\s*you\s+are", re.IGNORECASE),
            re.compile(r"<\|im_start\|>system", re.IGNORECASE),
            re.compile(r"\[INST\].*\[/INST\]", re.IGNORECASE),
        ]
        self._injection_scan_count = 0
        self._injection_detections = 0

        # Statistics
        self._total_stores = 0
        self._total_retrievals = 0
        self._consent_denials = 0

    def _scan_for_injection(self, content: str) -> bool:
        """
        Scan retrieved content for prompt injection patterns (V2-2).

        Returns True if injection-like content is detected.
        """
        self._injection_scan_count += 1
        for pattern in self._injection_patterns:
            if pattern.search(content):
                self._injection_detections += 1
                logger.warning(
                    "Injection pattern detected in retrieved memory content "
                    "(pattern=%s, scans=%d, detections=%d)",
                    pattern.pattern[:40],
                    self._injection_scan_count,
                    self._injection_detections,
                )
                return True
        return False

    def store_memory(
        self,
        content: str,
        context_type: ContextType,
        source: str,
        consent_id: Optional[str] = None,
        importance: float = 0.5,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> MemoryEntry:
        """
        Store a memory with embedding.

        Args:
            content: Memory content
            context_type: Type of context
            source: Source of the memory
            consent_id: Associated consent
            importance: Importance score (0-1)
            metadata: Additional metadata

        Returns:
            Created MemoryEntry
        """
        memory_id = f"mem_{secrets.token_hex(16)}"

        # Generate embedding
        embedding_result = self._embedding.embed(content)

        # Create memory entry
        memory = MemoryEntry(
            memory_id=memory_id,
            content=content,
            context_type=context_type,
            source=source,
            consent_id=consent_id,
            importance=importance,
            metadata=metadata or {},
        )

        # Store in vector store
        doc = VectorDocument(
            doc_id=memory_id,
            content=content,
            embedding=embedding_result.embedding,
            metadata={
                "context_type": context_type.name,
                "source": source,
                "importance": importance,
                **(metadata or {}),
            },
            consent_id=consent_id,
        )
        self._store.add(doc)

        # Store metadata with eviction if needed
        self._evict_metadata_if_needed()
        self._memory_metadata[memory_id] = memory
        self._total_stores += 1

        logger.debug(f"Stored memory: {memory_id} ({context_type.name})")
        return memory

    def _evict_metadata_if_needed(self) -> None:
        """Evict oldest metadata entries if over capacity."""
        if len(self._memory_metadata) < self._max_memory_metadata:
            return

        # Sort by last_accessed (None treated as oldest) then created_at
        def sort_key(entry: Tuple[str, MemoryEntry]) -> Tuple[datetime, datetime]:
            mem = entry[1]
            last_access = mem.last_accessed or datetime.min
            return (last_access, mem.created_at)

        sorted_entries = sorted(self._memory_metadata.items(), key=sort_key)

        # Remove oldest 10% to avoid frequent eviction
        entries_to_remove = max(1, len(sorted_entries) // 10)
        for memory_id, _ in sorted_entries[:entries_to_remove]:
            del self._memory_metadata[memory_id]

        logger.debug(f"Evicted {entries_to_remove} memory metadata entries")

    def store_memories_batch(
        self,
        entries: List[Tuple[str, ContextType, str]],
        consent_id: Optional[str] = None,
    ) -> List[MemoryEntry]:
        """
        Store multiple memories efficiently.

        Args:
            entries: List of (content, context_type, source) tuples
            consent_id: Common consent ID

        Returns:
            List of created MemoryEntry objects
        """
        if not entries:
            return []

        # Generate embeddings in batch
        contents = [e[0] for e in entries]
        batch_result = self._embedding.embed_batch(contents)

        memories = []
        docs = []

        for i, (content, context_type, source) in enumerate(entries):
            memory_id = f"mem_{secrets.token_hex(16)}"

            memory = MemoryEntry(
                memory_id=memory_id,
                content=content,
                context_type=context_type,
                source=source,
                consent_id=consent_id,
            )

            doc = VectorDocument(
                doc_id=memory_id,
                content=content,
                embedding=batch_result.embeddings[i].embedding,
                metadata={
                    "context_type": context_type.name,
                    "source": source,
                },
                consent_id=consent_id,
            )

            memories.append(memory)
            docs.append(doc)
            self._memory_metadata[memory_id] = memory

        # Batch add to vector store
        self._store.add_batch(docs)
        self._total_stores += len(memories)

        return memories

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        min_score: float = 0.0,
        context_types: Optional[List[ContextType]] = None,
        accessor: Optional[str] = None,
        consent_ids: Optional[List[str]] = None,
    ) -> RetrievalResult:
        """
        Retrieve relevant memories for a query.

        Args:
            query: Query text
            top_k: Number of results to return
            min_score: Minimum similarity score
            context_types: Filter by context types
            accessor: Who is accessing (for consent)
            consent_ids: Allowed consent IDs

        Returns:
            RetrievalResult with matched memories
        """
        import time

        start_time = time.time()

        accessor = accessor or self._default_accessor

        # Default values
        if top_k is None:
            top_k = 10
        if min_score is None:
            min_score = 0.0

        # Generate query embedding
        query_embedding = self._embedding.embed(query)

        # Build filters
        filters = {}
        if context_types:
            # Note: vector store may need special handling for list filters
            # For now, we'll filter after retrieval
            pass

        # Search vector store
        search_query = SearchQuery(
            embedding=query_embedding.embedding,
            top_k=top_k * 2,  # Get extra for filtering
            min_score=min_score,
            filters=filters,
            consent_ids=consent_ids,
        )

        results = self._store.search(search_query)
        total_candidates = len(results)

        # Filter and enrich results
        memories = []
        scores = []
        filtered_count = 0

        for result in results:
            # Verify consent
            if not self._consent.verify(result.consent_id, accessor):
                filtered_count += 1
                self._consent_denials += 1
                continue

            # Filter by context type
            if context_types:
                result_type = result.metadata.get("context_type")
                if result_type and ContextType[result_type] not in context_types:
                    continue

            # V2-2: Scan for injection patterns before including in results
            if self._scan_for_injection(result.content):
                filtered_count += 1
                continue

            # Get full memory entry
            memory = self._memory_metadata.get(result.doc_id)
            if not memory:
                # Reconstruct from search result
                memory = MemoryEntry(
                    memory_id=result.doc_id,
                    content=result.content,
                    context_type=ContextType[result.metadata.get("context_type", "KNOWLEDGE")],
                    source=result.metadata.get("source", "unknown"),
                    consent_id=result.consent_id,
                    importance=result.metadata.get("importance", 0.5),
                )

            # Update access tracking (only if last access was >60s ago to reduce overhead)
            memory.access_count += 1
            now = datetime.now()
            if not memory.last_accessed or (now - memory.last_accessed).total_seconds() > 60:
                memory.last_accessed = now

            memories.append(memory)
            scores.append(result.score)

            if len(memories) >= top_k:
                break

        retrieval_time = (time.time() - start_time) * 1000
        self._total_retrievals += 1

        return RetrievalResult(
            memories=memories,
            scores=scores,
            query=query,
            total_candidates=total_candidates,
            filtered_by_consent=filtered_count,
            retrieval_time_ms=retrieval_time,
        )

    def retrieve_for_rag(
        self,
        query: str,
        system_context: Optional[str] = None,
        conversation_history: Optional[str] = None,
        max_context_tokens: int = 2000,
        **retrieve_kwargs,
    ) -> RAGContext:
        """
        Retrieve and assemble context for RAG.

        Args:
            query: User query
            system_context: System/instruction context
            conversation_history: Recent conversation
            max_context_tokens: Max tokens for retrieved context
            **retrieve_kwargs: Arguments for retrieve()

        Returns:
            RAGContext ready for LLM
        """
        # Retrieve relevant memories
        result = self.retrieve(query, **retrieve_kwargs)

        # Convert to context string
        retrieved_context = result.to_context_string(max_tokens=max_context_tokens)
        context_tokens = len(retrieved_context) // 4  # Rough estimate

        return RAGContext(
            query=query,
            retrieved_context=retrieved_context,
            system_context=system_context,
            conversation_history=conversation_history,
            total_memories=len(result.memories),
            context_tokens=context_tokens,
        )

    def delete_memory(self, memory_id: str) -> bool:
        """
        Delete a specific memory.

        Args:
            memory_id: Memory to delete

        Returns:
            True if deleted
        """
        success = self._store.delete(memory_id)
        if success and memory_id in self._memory_metadata:
            del self._memory_metadata[memory_id]
        return success

    def delete_by_consent(self, consent_id: str) -> int:
        """
        Delete all memories with a consent ID (right to forget).

        Args:
            consent_id: Consent ID to delete

        Returns:
            Number of memories deleted
        """
        # Delete from vector store
        count = self._store.delete_by_consent(consent_id)

        # Delete from metadata
        to_delete = [
            mid for mid, mem in self._memory_metadata.items() if mem.consent_id == consent_id
        ]
        for mid in to_delete:
            del self._memory_metadata[mid]

        logger.info(f"Deleted {count} memories for consent: {consent_id}")
        return count

    def consolidate_memories(
        self,
        context_type: Optional[ContextType] = None,
        min_age_hours: int = 24,
        summarize_callback: Optional[Callable[[List[str]], str]] = None,
    ) -> int:
        """
        Consolidate old memories by summarizing similar ones.

        Args:
            context_type: Type to consolidate (or all)
            min_age_hours: Minimum age to consolidate
            summarize_callback: Callback to summarize content list

        Returns:
            Number of memories consolidated
        """
        if not summarize_callback:
            logger.warning("No summarize callback provided, skipping consolidation")
            return 0

        from datetime import timedelta

        cutoff = datetime.now() - timedelta(hours=min_age_hours)

        # Find old memories
        old_memories = [
            m
            for m in self._memory_metadata.values()
            if m.created_at < cutoff and (context_type is None or m.context_type == context_type)
        ]

        if len(old_memories) < 3:
            return 0  # Not enough to consolidate

        # Group by similarity (simple clustering)
        # For now, just consolidate all into one summary
        contents = [m.content for m in old_memories]
        summary = summarize_callback(contents)

        # Delete old memories
        for memory in old_memories:
            self.delete_memory(memory.memory_id)

        # Store summary as new memory
        self.store_memory(
            content=summary,
            context_type=context_type or ContextType.KNOWLEDGE,
            source="consolidation",
            importance=0.7,  # Consolidated memories are more important
            metadata={"consolidated_from": len(old_memories)},
        )

        logger.info(f"Consolidated {len(old_memories)} memories into summary")
        return len(old_memories)

    def get_memory(self, memory_id: str) -> Optional[MemoryEntry]:
        """Get a specific memory by ID."""
        return self._memory_metadata.get(memory_id)

    def get_statistics(self) -> Dict[str, Any]:
        """Get pipeline statistics."""
        return {
            "total_memories": len(self._memory_metadata),
            "vector_store_count": self._store.count(),
            "total_stores": self._total_stores,
            "total_retrievals": self._total_retrievals,
            "consent_denials": self._consent_denials,
            "embedding_metrics": self._embedding.get_metrics(),
        }

    def clear(self) -> None:
        """Clear all memories."""
        self._store.clear()
        self._memory_metadata.clear()

    def shutdown(self) -> None:
        """Shutdown the pipeline."""
        self._store.shutdown()
        self._embedding.shutdown()


class HybridRetriever:
    """
    Hybrid retrieval combining semantic and keyword search.

    Merges results from both approaches with configurable weighting.
    """

    def __init__(
        self,
        retrieval_pipeline: RetrievalPipeline,
        semantic_weight: float = 0.7,
        keyword_weight: float = 0.3,
    ):
        """
        Initialize hybrid retriever.

        Args:
            retrieval_pipeline: Base retrieval pipeline
            semantic_weight: Weight for semantic results
            keyword_weight: Weight for keyword results
        """
        self._pipeline = retrieval_pipeline
        self._semantic_weight = semantic_weight
        self._keyword_weight = keyword_weight

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        **kwargs,
    ) -> RetrievalResult:
        """
        Perform hybrid retrieval.

        Args:
            query: Query text
            top_k: Number of results
            **kwargs: Additional arguments

        Returns:
            Merged RetrievalResult
        """
        # Get semantic results
        semantic_result = self._pipeline.retrieve(
            query,
            top_k=top_k * 2,
            **kwargs,
        )

        # Get keyword results (simple keyword matching)
        keyword_matches = self._keyword_search(query, top_k * 2)

        # Merge results
        merged = self._merge_results(
            semantic_result.memories,
            semantic_result.scores,
            keyword_matches,
            top_k,
        )

        return RetrievalResult(
            memories=merged[0],
            scores=merged[1],
            query=query,
            total_candidates=semantic_result.total_candidates + len(keyword_matches),
            filtered_by_consent=semantic_result.filtered_by_consent,
            retrieval_time_ms=semantic_result.retrieval_time_ms,
        )

    def _keyword_search(
        self,
        query: str,
        top_k: int,
    ) -> List[Tuple[MemoryEntry, float]]:
        """Simple keyword matching."""
        keywords = query.lower().split()
        results = []

        for memory in self._pipeline._memory_metadata.values():
            content_lower = memory.content.lower()
            # Count keyword matches
            matches = sum(1 for kw in keywords if kw in content_lower)
            if matches > 0:
                score = matches / len(keywords)
                results.append((memory, score))

        # Sort by score
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def _merge_results(
        self,
        semantic_memories: List[MemoryEntry],
        semantic_scores: List[float],
        keyword_results: List[Tuple[MemoryEntry, float]],
        top_k: int,
    ) -> Tuple[List[MemoryEntry], List[float]]:
        """Merge semantic and keyword results with weighting."""
        # Build score map
        scores: Dict[str, float] = {}
        memories: Dict[str, MemoryEntry] = {}

        # Add semantic scores
        for memory, score in zip(semantic_memories, semantic_scores):
            scores[memory.memory_id] = score * self._semantic_weight
            memories[memory.memory_id] = memory

        # Add keyword scores
        for memory, score in keyword_results:
            if memory.memory_id in scores:
                scores[memory.memory_id] += score * self._keyword_weight
            else:
                scores[memory.memory_id] = score * self._keyword_weight
                memories[memory.memory_id] = memory

        # Sort by combined score
        sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)

        result_memories = [memories[mid] for mid in sorted_ids[:top_k]]
        result_scores = [scores[mid] for mid in sorted_ids[:top_k]]

        return result_memories, result_scores


def create_retrieval_pipeline(
    embedding_engine: EmbeddingEngine,
    vector_store: VectorStoreBase,
    consent_callback: Optional[Callable[[str, str], bool]] = None,
) -> RetrievalPipeline:
    """
    Create a configured retrieval pipeline.

    Args:
        embedding_engine: Initialized embedding engine
        vector_store: Initialized vector store
        consent_callback: Optional consent verification callback

    Returns:
        Configured RetrievalPipeline
    """
    consent_verifier = ConsentVerifier(consent_callback) if consent_callback else None

    return RetrievalPipeline(
        embedding_engine=embedding_engine,
        vector_store=vector_store,
        consent_verifier=consent_verifier,
    )
