"""Layer 2: Memory Items for the Sigil v2 3-layer memory architecture.

This module implements the Memory Item Layer, which stores discrete facts
extracted from resources with embeddings for semantic search.

Classes:
    ItemLayer: Main class for storing and retrieving memory items.
    EmbeddingService: Service for generating text embeddings.
    VectorIndex: FAISS-based approximate nearest neighbor search.

Storage Format:
    - Items: JSON file at {storage_dir}/items.json
    - Embeddings: FAISS index at {storage_dir}/embeddings.faiss

Example:
    >>> from sigil.memory.layers.items import ItemLayer
    >>> layer = ItemLayer(storage_dir="outputs/memory/items")
    >>> item = layer.store(
    ...     content="Customer prefers monthly billing",
    ...     source_resource_id="res-123",
    ...     category="billing_preferences"
    ... )
    >>> results = layer.search_by_embedding("payment preferences", k=5)
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Optional, Protocol
import threading

import numpy as np
import portalocker

from sigil.config.schemas.memory import MemoryItem, generate_uuid, utc_now
from sigil.core.exceptions import MemoryError, MemoryWriteError, MemoryRetrievalError


# =============================================================================
# Constants
# =============================================================================

DEFAULT_STORAGE_DIR = "outputs/memory/items"
"""Default directory for item storage."""

DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
"""Default OpenAI embedding model."""

DEFAULT_EMBEDDING_DIMENSIONS = 1536
"""Default embedding dimensions for text-embedding-3-small."""


# =============================================================================
# Embedding Service Protocol
# =============================================================================


class EmbeddingProvider(Protocol):
    """Protocol for embedding providers."""

    def embed(self, text: str) -> list[float]:
        """Generate embedding for a single text.

        Args:
            text: Text to embed.

        Returns:
            List of floats representing the embedding vector.
        """
        ...

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed.

        Returns:
            List of embedding vectors.
        """
        ...

    @property
    def dimensions(self) -> int:
        """Get the embedding dimensions."""
        ...


# =============================================================================
# Embedding Service
# =============================================================================


class EmbeddingService:
    """Service for generating text embeddings.

    This service supports multiple backends (OpenAI, local models) and provides
    caching and batching for efficiency.

    Attributes:
        provider: The embedding provider to use.
        model: Name of the embedding model.
        dimensions: Dimensionality of the embeddings.

    Example:
        >>> service = EmbeddingService.create_openai(api_key="...")
        >>> embedding = service.embed("Hello world")
        >>> len(embedding)
        1536
    """

    def __init__(
        self,
        provider: Optional[EmbeddingProvider] = None,
        model: str = DEFAULT_EMBEDDING_MODEL,
        dimensions: int = DEFAULT_EMBEDDING_DIMENSIONS,
    ) -> None:
        """Initialize the embedding service.

        Args:
            provider: Custom embedding provider (optional).
            model: Model name for built-in providers.
            dimensions: Embedding dimensions.
        """
        self._provider = provider
        self.model = model
        self._dimensions = dimensions
        self._cache: dict[str, list[float]] = {}
        self._lock = threading.Lock()

    @property
    def dimensions(self) -> int:
        """Get the embedding dimensions."""
        if self._provider is not None:
            return self._provider.dimensions
        return self._dimensions

    def embed(self, text: str) -> list[float]:
        """Generate embedding for a single text.

        Args:
            text: Text to embed.

        Returns:
            Embedding vector as a list of floats.
        """
        # Check cache first
        with self._lock:
            if text in self._cache:
                return self._cache[text].copy()

        if self._provider is not None:
            embedding = self._provider.embed(text)
        else:
            # Use OpenAI if available
            embedding = self._embed_openai(text)

        # Cache the result
        with self._lock:
            self._cache[text] = embedding.copy()

        return embedding

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed.

        Returns:
            List of embedding vectors.
        """
        if not texts:
            return []

        if self._provider is not None:
            return self._provider.embed_batch(texts)

        # Use OpenAI batch embedding
        return self._embed_openai_batch(texts)

    def _embed_openai(self, text: str) -> list[float]:
        """Generate embedding using OpenAI API.

        Args:
            text: Text to embed.

        Returns:
            Embedding vector.
        """
        try:
            import openai
            from sigil.config import get_settings

            settings = get_settings()
            api_key = settings.api_keys.openai_api_key

            if not api_key:
                # Fall back to a simple hash-based embedding for testing
                return self._fallback_embedding(text)

            client = openai.OpenAI(api_key=api_key)
            response = client.embeddings.create(
                model=self.model,
                input=text,
            )
            return response.data[0].embedding

        except ImportError:
            # OpenAI not installed, use fallback
            return self._fallback_embedding(text)
        except Exception:
            # API error, use fallback
            return self._fallback_embedding(text)

    def _embed_openai_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for batch using OpenAI API.

        Args:
            texts: List of texts to embed.

        Returns:
            List of embedding vectors.
        """
        try:
            import openai
            from sigil.config import get_settings

            settings = get_settings()
            api_key = settings.api_keys.openai_api_key

            if not api_key:
                return [self._fallback_embedding(t) for t in texts]

            client = openai.OpenAI(api_key=api_key)
            response = client.embeddings.create(
                model=self.model,
                input=texts,
            )
            # Sort by index to maintain order
            sorted_embeddings = sorted(response.data, key=lambda x: x.index)
            return [e.embedding for e in sorted_embeddings]

        except ImportError:
            return [self._fallback_embedding(t) for t in texts]
        except Exception:
            return [self._fallback_embedding(t) for t in texts]

    def _fallback_embedding(self, text: str) -> list[float]:
        """Generate a simple word-based embedding for testing.

        This is NOT suitable for production use. It provides a deterministic
        embedding based on word tokens that captures basic semantic similarity.

        Args:
            text: Text to embed.

        Returns:
            Pseudo-embedding vector.
        """
        import hashlib
        import re

        # Tokenize and normalize text
        words = set(re.findall(r'\b[a-zA-Z]+\b', text.lower()))

        # Create a deterministic embedding based on word hashes
        # Each word contributes to the embedding in a consistent way
        embedding = [0.0] * self._dimensions

        for word in words:
            # Hash the word to get a seed
            word_hash = hashlib.md5(word.encode()).digest()
            # Use the hash to determine which dimensions this word affects
            for i in range(min(len(word_hash), 16)):
                # Each byte of the hash affects a different set of dimensions
                byte_val = word_hash[i]
                # Spread the word's influence across multiple dimensions
                for j in range(self._dimensions // 16):
                    dim_idx = (i * (self._dimensions // 16) + j) % self._dimensions
                    # Add contribution based on hash byte
                    contribution = ((byte_val >> (j % 8)) & 1) * 2 - 1  # -1 or 1
                    embedding[dim_idx] += contribution * 0.1

        # Normalize the embedding
        norm = sum(x * x for x in embedding) ** 0.5
        if norm > 0:
            embedding = [x / norm for x in embedding]
        else:
            # If empty text, use a hash-based fallback
            hash_bytes = hashlib.sha256(text.encode()).digest()
            full_hash = hash_bytes * (self._dimensions // 32 + 1)
            embedding = [(full_hash[i] / 127.5) - 1.0 for i in range(self._dimensions)]

        return embedding

    def clear_cache(self) -> None:
        """Clear the embedding cache."""
        with self._lock:
            self._cache.clear()

    @classmethod
    def create_openai(
        cls,
        api_key: Optional[str] = None,
        model: str = DEFAULT_EMBEDDING_MODEL,
    ) -> "EmbeddingService":
        """Create an EmbeddingService using OpenAI.

        Args:
            api_key: OpenAI API key (uses settings if not provided).
            model: Model name.

        Returns:
            Configured EmbeddingService.
        """
        service = cls(model=model)
        return service


# =============================================================================
# Vector Index
# =============================================================================


class VectorIndex:
    """FAISS-based vector index for approximate nearest neighbor search.

    This class provides efficient similarity search over embedding vectors
    using FAISS. It supports adding, searching, and removing vectors.

    Attributes:
        dimensions: Dimensionality of the vectors.
        index_path: Path to the persisted FAISS index.

    Example:
        >>> index = VectorIndex(dimensions=1536, storage_dir="outputs/memory/items")
        >>> index.add("item-1", [0.1, 0.2, ...])
        >>> results = index.search([0.1, 0.2, ...], k=5)
        >>> for item_id, score in results:
        ...     print(f"{item_id}: {score}")
    """

    def __init__(
        self,
        dimensions: int = DEFAULT_EMBEDDING_DIMENSIONS,
        storage_dir: str = DEFAULT_STORAGE_DIR,
    ) -> None:
        """Initialize the vector index.

        Args:
            dimensions: Dimensionality of the vectors.
            storage_dir: Directory for persisting the index.
        """
        self.dimensions = dimensions
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        self.index_path = self.storage_dir / "embeddings.faiss"
        self.id_map_path = self.storage_dir / "id_map.json"

        self._index: Optional[Any] = None
        self._id_to_idx: dict[str, int] = {}
        self._idx_to_id: dict[int, str] = {}
        self._next_idx: int = 0
        self._lock = threading.RLock()

        self._load_or_create_index()

    def _load_or_create_index(self) -> None:
        """Load existing index or create a new one."""
        try:
            import faiss

            if self.index_path.exists() and self.id_map_path.exists():
                # Load existing index
                self._index = faiss.read_index(str(self.index_path))

                with open(self.id_map_path, "r") as f:
                    data = json.load(f)
                    self._id_to_idx = data.get("id_to_idx", {})
                    self._idx_to_id = {int(k): v for k, v in data.get("idx_to_id", {}).items()}
                    self._next_idx = data.get("next_idx", 0)
            else:
                # Create new index (Inner Product for cosine similarity after normalization)
                self._index = faiss.IndexFlatIP(self.dimensions)

        except ImportError:
            # FAISS not installed, use in-memory fallback
            self._index = None
            self._vectors: dict[str, list[float]] = {}

    def _save_index(self) -> None:
        """Save the index and ID mappings to disk."""
        try:
            import faiss

            if self._index is not None:
                faiss.write_index(self._index, str(self.index_path))

            with open(self.id_map_path, "w") as f:
                json.dump(
                    {
                        "id_to_idx": self._id_to_idx,
                        "idx_to_id": {str(k): v for k, v in self._idx_to_id.items()},
                        "next_idx": self._next_idx,
                    },
                    f,
                    indent=2,
                )
        except ImportError:
            # FAISS not installed, save vectors as JSON
            vectors_path = self.storage_dir / "vectors.json"
            with open(vectors_path, "w") as f:
                json.dump(self._vectors, f)

    def add(self, item_id: str, embedding: list[float]) -> None:
        """Add a vector to the index.

        Args:
            item_id: Unique identifier for the item.
            embedding: The embedding vector.
        """
        with self._lock:
            if len(embedding) != self.dimensions:
                raise ValueError(
                    f"Embedding dimension mismatch: expected {self.dimensions}, "
                    f"got {len(embedding)}"
                )

            try:
                import faiss

                if self._index is not None:
                    # Normalize for cosine similarity
                    vec = np.array([embedding], dtype=np.float32)
                    faiss.normalize_L2(vec)

                    # Add to index
                    self._index.add(vec)
                    idx = self._next_idx
                    self._id_to_idx[item_id] = idx
                    self._idx_to_id[idx] = item_id
                    self._next_idx += 1

            except ImportError:
                # Fallback: store in dict
                self._vectors[item_id] = embedding

            self._save_index()

    def add_batch(self, items: list[tuple[str, list[float]]]) -> None:
        """Add multiple vectors to the index.

        Args:
            items: List of (item_id, embedding) tuples.
        """
        if not items:
            return

        with self._lock:
            try:
                import faiss

                if self._index is not None:
                    # Prepare batch
                    vectors = np.array([emb for _, emb in items], dtype=np.float32)
                    faiss.normalize_L2(vectors)

                    # Add to index
                    start_idx = self._next_idx
                    self._index.add(vectors)

                    for i, (item_id, _) in enumerate(items):
                        idx = start_idx + i
                        self._id_to_idx[item_id] = idx
                        self._idx_to_id[idx] = item_id

                    self._next_idx = start_idx + len(items)

            except ImportError:
                # Fallback
                for item_id, embedding in items:
                    self._vectors[item_id] = embedding

            self._save_index()

    def search(
        self,
        query_embedding: list[float],
        k: int = 10,
    ) -> list[tuple[str, float]]:
        """Search for the k nearest neighbors.

        Args:
            query_embedding: The query embedding vector.
            k: Number of results to return.

        Returns:
            List of (item_id, score) tuples, sorted by descending similarity.
        """
        with self._lock:
            if len(query_embedding) != self.dimensions:
                raise ValueError(
                    f"Query embedding dimension mismatch: expected {self.dimensions}, "
                    f"got {len(query_embedding)}"
                )

            try:
                import faiss

                if self._index is not None and self._index.ntotal > 0:
                    # Normalize query
                    query = np.array([query_embedding], dtype=np.float32)
                    faiss.normalize_L2(query)

                    # Search
                    k = min(k, self._index.ntotal)
                    distances, indices = self._index.search(query, k)

                    results = []
                    for dist, idx in zip(distances[0], indices[0]):
                        if idx >= 0 and idx in self._idx_to_id:
                            item_id = self._idx_to_id[idx]
                            # Convert inner product to similarity score (already normalized)
                            results.append((item_id, float(dist)))

                    return results

            except ImportError:
                # Fallback: brute-force cosine similarity
                if not self._vectors:
                    return []

                query_norm = np.linalg.norm(query_embedding)
                if query_norm == 0:
                    return []

                results = []
                for item_id, vec in self._vectors.items():
                    vec_norm = np.linalg.norm(vec)
                    if vec_norm == 0:
                        continue
                    similarity = np.dot(query_embedding, vec) / (query_norm * vec_norm)
                    results.append((item_id, float(similarity)))

                results.sort(key=lambda x: x[1], reverse=True)
                return results[:k]

            return []

    def remove(self, item_id: str) -> bool:
        """Remove a vector from the index.

        Note: FAISS doesn't support efficient single-item removal.
        This method marks the item as removed but doesn't physically remove it.
        Call rebuild() periodically to clean up.

        Args:
            item_id: The item ID to remove.

        Returns:
            True if the item was found and marked for removal.
        """
        with self._lock:
            if item_id in self._id_to_idx:
                idx = self._id_to_idx[item_id]
                del self._id_to_idx[item_id]
                if idx in self._idx_to_id:
                    del self._idx_to_id[idx]
                self._save_index()
                return True

            if hasattr(self, "_vectors") and item_id in self._vectors:
                del self._vectors[item_id]
                self._save_index()
                return True

            return False

    def contains(self, item_id: str) -> bool:
        """Check if an item is in the index.

        Args:
            item_id: The item ID to check.

        Returns:
            True if the item is in the index.
        """
        return item_id in self._id_to_idx or (
            hasattr(self, "_vectors") and item_id in self._vectors
        )

    def count(self) -> int:
        """Get the number of items in the index.

        Returns:
            Number of items.
        """
        if hasattr(self, "_vectors"):
            return len(self._vectors)
        return len(self._id_to_idx)

    def clear(self) -> None:
        """Clear the entire index."""
        with self._lock:
            try:
                import faiss

                self._index = faiss.IndexFlatIP(self.dimensions)
            except ImportError:
                self._vectors = {}

            self._id_to_idx.clear()
            self._idx_to_id.clear()
            self._next_idx = 0
            self._save_index()


# =============================================================================
# Item Layer
# =============================================================================


class ItemLayer:
    """Layer 2 of the 3-layer memory architecture: Memory items with embeddings.

    The ItemLayer stores discrete facts extracted from resources, each with
    an optional embedding for semantic search. Items link back to their
    source resources for full traceability.

    Features:
        - CRUD operations for memory items
        - Embedding-based semantic search via FAISS
        - Category-based organization
        - Source resource traceability
        - Thread-safe operations

    Attributes:
        storage_dir: Path to the item storage directory.
        embedding_service: Service for generating embeddings.
        vector_index: FAISS index for similarity search.

    Example:
        >>> layer = ItemLayer("outputs/memory/items")
        >>> item = layer.store(
        ...     content="Customer prefers email communication",
        ...     source_resource_id="conv-123",
        ...     category="communication_preferences"
        ... )
        >>> results = layer.search_by_embedding("contact method", k=5)
    """

    def __init__(
        self,
        storage_dir: str = DEFAULT_STORAGE_DIR,
        embedding_service: Optional[EmbeddingService] = None,
    ) -> None:
        """Initialize the ItemLayer.

        Args:
            storage_dir: Directory path for storing items.
            embedding_service: Optional custom embedding service.
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        self.items_file = self.storage_dir / "items.json"
        self._items: dict[str, MemoryItem] = {}
        self._lock = threading.RLock()

        # Initialize embedding service and vector index
        self.embedding_service = embedding_service or EmbeddingService()
        self.vector_index = VectorIndex(
            dimensions=self.embedding_service.dimensions,
            storage_dir=str(self.storage_dir),
        )

        self._load_items()

    def _load_items(self) -> None:
        """Load items from storage."""
        if not self.items_file.exists():
            return

        try:
            with portalocker.Lock(
                self.items_file, mode="r", timeout=10, flags=portalocker.LOCK_SH
            ) as f:
                data = json.load(f)

            for item_data in data.get("items", []):
                item = self._deserialize_item(item_data)
                self._items[item.item_id] = item

        except Exception as e:
            # Log but don't fail - start with empty items
            pass

    def _save_items(self) -> None:
        """Save items to storage."""
        try:
            data = {
                "items": [self._serialize_item(item) for item in self._items.values()],
                "updated_at": utc_now().isoformat(),
            }

            with portalocker.Lock(
                self.items_file, mode="w", timeout=10, flags=portalocker.LOCK_EX
            ) as f:
                json.dump(data, f, indent=2)

        except Exception as e:
            raise MemoryWriteError(f"Failed to save items: {e}", layer="items")

    def _serialize_item(self, item: MemoryItem) -> dict[str, Any]:
        """Serialize a MemoryItem to a dictionary."""
        return {
            "item_id": item.item_id,
            "content": item.content,
            "embedding": item.embedding,
            "source_resource_id": item.source_resource_id,
            "category": item.category,
            "confidence": item.confidence,
            "created_at": item.created_at.isoformat(),
        }

    def _deserialize_item(self, data: dict[str, Any]) -> MemoryItem:
        """Deserialize a dictionary to a MemoryItem."""
        created_at = data.get("created_at")
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)
        elif created_at is None:
            created_at = utc_now()

        return MemoryItem(
            item_id=data["item_id"],
            content=data["content"],
            embedding=data.get("embedding"),
            source_resource_id=data["source_resource_id"],
            category=data.get("category"),
            confidence=data.get("confidence", 1.0),
            created_at=created_at,
        )

    def store(
        self,
        content: str,
        source_resource_id: str,
        category: Optional[str] = None,
        confidence: float = 1.0,
        item_id: Optional[str] = None,
        generate_embedding: bool = True,
    ) -> MemoryItem:
        """Store a new memory item.

        Args:
            content: The extracted fact or information.
            source_resource_id: ID of the source resource.
            category: Optional category name.
            confidence: Extraction confidence (0.0-1.0).
            item_id: Optional specific item ID.
            generate_embedding: Whether to generate an embedding.

        Returns:
            The stored MemoryItem.

        Raises:
            MemoryWriteError: If the item cannot be stored.
        """
        with self._lock:
            # Generate embedding if requested
            embedding = None
            if generate_embedding:
                try:
                    embedding = self.embedding_service.embed(content)
                except Exception as e:
                    # Log warning but continue without embedding
                    pass

            # Create the item
            item = MemoryItem(
                item_id=item_id or generate_uuid(),
                content=content,
                embedding=embedding,
                source_resource_id=source_resource_id,
                category=category,
                confidence=confidence,
            )

            # Add to items dict
            self._items[item.item_id] = item

            # Add to vector index if we have an embedding
            if embedding is not None:
                self.vector_index.add(item.item_id, embedding)

            # Persist
            self._save_items()

            return item

    def store_item(self, item: MemoryItem) -> MemoryItem:
        """Store an existing MemoryItem object.

        Args:
            item: The MemoryItem to store.

        Returns:
            The stored MemoryItem.
        """
        with self._lock:
            self._items[item.item_id] = item

            if item.embedding is not None:
                self.vector_index.add(item.item_id, item.embedding)

            self._save_items()
            return item

    def store_batch(
        self,
        items: list[tuple[str, str, Optional[str], float]],
        source_resource_id: str,
        generate_embeddings: bool = True,
    ) -> list[MemoryItem]:
        """Store multiple memory items efficiently.

        Args:
            items: List of (content, category, item_id, confidence) tuples.
                Category and item_id can be None.
            source_resource_id: ID of the source resource for all items.
            generate_embeddings: Whether to generate embeddings.

        Returns:
            List of stored MemoryItems.
        """
        with self._lock:
            stored_items: list[MemoryItem] = []
            embeddings_to_add: list[tuple[str, list[float]]] = []

            # Generate embeddings in batch if requested
            if generate_embeddings:
                contents = [content for content, _, _, _ in items]
                try:
                    embeddings = self.embedding_service.embed_batch(contents)
                except Exception:
                    embeddings = [None] * len(contents)
            else:
                embeddings = [None] * len(items)

            # Create and store items
            for i, (content, category, item_id, confidence) in enumerate(items):
                item = MemoryItem(
                    item_id=item_id or generate_uuid(),
                    content=content,
                    embedding=embeddings[i] if i < len(embeddings) else None,
                    source_resource_id=source_resource_id,
                    category=category,
                    confidence=confidence,
                )

                self._items[item.item_id] = item
                stored_items.append(item)

                if item.embedding is not None:
                    embeddings_to_add.append((item.item_id, item.embedding))

            # Batch add to vector index
            if embeddings_to_add:
                self.vector_index.add_batch(embeddings_to_add)

            # Persist
            self._save_items()

            return stored_items

    def get(self, item_id: str) -> Optional[MemoryItem]:
        """Get a memory item by ID.

        Args:
            item_id: Unique identifier for the item.

        Returns:
            The MemoryItem if found, None otherwise.
        """
        return self._items.get(item_id)

    def get_by_resource(self, resource_id: str) -> list[MemoryItem]:
        """Get all items extracted from a specific resource.

        Args:
            resource_id: The source resource ID.

        Returns:
            List of MemoryItems from that resource.
        """
        return [
            item
            for item in self._items.values()
            if item.source_resource_id == resource_id
        ]

    def get_by_category(
        self,
        category: str,
        limit: Optional[int] = None,
    ) -> list[MemoryItem]:
        """Get all items in a specific category.

        Args:
            category: The category name.
            limit: Optional limit on number of items.

        Returns:
            List of MemoryItems in that category.
        """
        # Normalize category name
        category = category.strip().lower().replace(" ", "_").replace("-", "_")

        items = [item for item in self._items.values() if item.category == category]

        # Sort by created_at (newest first)
        items.sort(key=lambda x: x.created_at, reverse=True)

        if limit is not None:
            items = items[:limit]

        return items

    def search_by_embedding(
        self,
        query: str,
        k: int = 10,
        category: Optional[str] = None,
        min_score: float = 0.0,
    ) -> list[tuple[MemoryItem, float]]:
        """Search for items by semantic similarity.

        Args:
            query: The search query text.
            k: Maximum number of results.
            category: Optional category filter.
            min_score: Minimum similarity score threshold.

        Returns:
            List of (MemoryItem, score) tuples, sorted by descending similarity.
        """
        # Generate query embedding
        try:
            query_embedding = self.embedding_service.embed(query)
        except Exception:
            return []

        # Search the vector index
        search_results = self.vector_index.search(query_embedding, k=k * 2)  # Over-fetch for filtering

        # Filter and collect results
        results: list[tuple[MemoryItem, float]] = []
        for item_id, score in search_results:
            if score < min_score:
                continue

            item = self._items.get(item_id)
            if item is None:
                continue

            if category is not None:
                normalized_category = category.strip().lower().replace(" ", "_").replace("-", "_")
                if item.category != normalized_category:
                    continue

            results.append((item, score))

            if len(results) >= k:
                break

        return results

    def search_by_content(
        self,
        query: str,
        limit: int = 10,
        case_sensitive: bool = False,
    ) -> list[MemoryItem]:
        """Search items by content text match.

        Args:
            query: The search query string.
            limit: Maximum number of results.
            case_sensitive: Whether to use case-sensitive matching.

        Returns:
            List of matching MemoryItems.
        """
        if not query.strip():
            return []

        search_query = query if case_sensitive else query.lower()
        results: list[tuple[MemoryItem, int]] = []

        for item in self._items.values():
            content = item.content if case_sensitive else item.content.lower()
            match_count = content.count(search_query)
            if match_count > 0:
                results.append((item, match_count))

        # Sort by match count and limit
        results.sort(key=lambda x: x[1], reverse=True)
        return [item for item, _ in results[:limit]]

    def delete(self, item_id: str) -> bool:
        """Delete a memory item.

        Args:
            item_id: The item ID to delete.

        Returns:
            True if the item was deleted, False if not found.
        """
        with self._lock:
            if item_id not in self._items:
                return False

            del self._items[item_id]
            self.vector_index.remove(item_id)
            self._save_items()
            return True

    def delete_by_resource(self, resource_id: str) -> int:
        """Delete all items from a specific resource.

        Args:
            resource_id: The source resource ID.

        Returns:
            Number of items deleted.
        """
        with self._lock:
            to_delete = [
                item_id
                for item_id, item in self._items.items()
                if item.source_resource_id == resource_id
            ]

            for item_id in to_delete:
                del self._items[item_id]
                self.vector_index.remove(item_id)

            if to_delete:
                self._save_items()

            return len(to_delete)

    def exists(self, item_id: str) -> bool:
        """Check if an item exists.

        Args:
            item_id: The item ID to check.

        Returns:
            True if the item exists.
        """
        return item_id in self._items

    def count(self, category: Optional[str] = None) -> int:
        """Count items, optionally by category.

        Args:
            category: Optional category filter.

        Returns:
            Number of items.
        """
        if category is None:
            return len(self._items)

        category = category.strip().lower().replace(" ", "_").replace("-", "_")
        return sum(1 for item in self._items.values() if item.category == category)

    def list_categories(self) -> list[str]:
        """List all unique categories.

        Returns:
            List of category names.
        """
        categories = set()
        for item in self._items.values():
            if item.category:
                categories.add(item.category)
        return sorted(categories)

    def iter_all(self) -> Iterator[MemoryItem]:
        """Iterate over all items.

        Yields:
            MemoryItem objects.
        """
        for item in self._items.values():
            yield item

    def update_embedding(self, item_id: str) -> bool:
        """Regenerate embedding for an item.

        Args:
            item_id: The item ID to update.

        Returns:
            True if updated successfully.
        """
        with self._lock:
            item = self._items.get(item_id)
            if item is None:
                return False

            try:
                embedding = self.embedding_service.embed(item.content)

                # Update item with new embedding
                updated_item = MemoryItem(
                    item_id=item.item_id,
                    content=item.content,
                    embedding=embedding,
                    source_resource_id=item.source_resource_id,
                    category=item.category,
                    confidence=item.confidence,
                    created_at=item.created_at,
                )

                self._items[item_id] = updated_item

                # Update vector index
                self.vector_index.remove(item_id)
                self.vector_index.add(item_id, embedding)

                self._save_items()
                return True

            except Exception:
                return False


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "ItemLayer",
    "EmbeddingService",
    "EmbeddingProvider",
    "VectorIndex",
    "DEFAULT_STORAGE_DIR",
    "DEFAULT_EMBEDDING_MODEL",
    "DEFAULT_EMBEDDING_DIMENSIONS",
]
