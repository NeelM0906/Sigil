"""In-memory storage backend implementation.

This module provides in-memory implementations of the storage protocols.
These implementations are primarily intended for testing but can also be
used for ephemeral caching or when persistence is not required.

Classes:
    MemoryBackend: In-memory key-value storage.
    MemoryVectorBackend: In-memory vector storage with brute-force search.
    MemoryLocking: In-memory locking using asyncio locks.

Example:
    >>> from sigil.core.storage.memory import MemoryBackend, MemoryLocking
    >>>
    >>> backend = MemoryBackend()
    >>> await backend.write("key1", b"value1")
    >>> data = await backend.read("key1")
    >>> print(data)
    b'value1'
    >>>
    >>> # Clean up for tests
    >>> backend.clear()
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Optional

import numpy as np


# =============================================================================
# Memory Backend
# =============================================================================


class MemoryBackend:
    """In-memory key-value storage backend.

    This backend stores all data in a dictionary. Data is lost when the
    process ends. Thread-safe through asyncio locks.

    This implementation is useful for:
        - Unit testing without filesystem I/O
        - Temporary caching
        - Development and prototyping

    Attributes:
        None public. Internal storage is in _data dict.

    Example:
        >>> backend = MemoryBackend()
        >>> await backend.write("test", b"data")
        >>> assert await backend.exists("test")
        >>> backend.clear()  # Reset for next test
    """

    def __init__(self) -> None:
        """Initialize the in-memory backend."""
        self._data: dict[str, bytes] = {}
        self._lock = asyncio.Lock()

    async def read(self, key: str) -> Optional[bytes]:
        """Read data by key.

        Args:
            key: The unique key identifying the data.

        Returns:
            The data as bytes if found, None if the key does not exist.
        """
        async with self._lock:
            return self._data.get(key)

    async def write(self, key: str, data: bytes) -> None:
        """Write data to key.

        Args:
            key: The unique key identifying the data.
            data: The data to store as bytes.
        """
        async with self._lock:
            self._data[key] = data

    async def delete(self, key: str) -> bool:
        """Delete data by key.

        Args:
            key: The unique key identifying the data to delete.

        Returns:
            True if the key was found and deleted, False if the key
            did not exist.
        """
        async with self._lock:
            if key in self._data:
                del self._data[key]
                return True
            return False

    async def exists(self, key: str) -> bool:
        """Check if a key exists in storage.

        Args:
            key: The unique key to check.

        Returns:
            True if the key exists, False otherwise.
        """
        async with self._lock:
            return key in self._data

    async def list_keys(self, prefix: str = "") -> list[str]:
        """List keys with optional prefix filter.

        Args:
            prefix: Optional prefix to filter keys. If empty, lists all keys.

        Returns:
            List of key strings matching the prefix.
        """
        async with self._lock:
            if prefix:
                return sorted([k for k in self._data.keys() if k.startswith(prefix)])
            return sorted(list(self._data.keys()))

    def clear(self) -> None:
        """Clear all data from storage.

        This is a synchronous method for easy use in test fixtures.
        """
        self._data.clear()

    async def clear_async(self) -> int:
        """Clear all data from storage asynchronously.

        Returns:
            The number of keys deleted.
        """
        async with self._lock:
            count = len(self._data)
            self._data.clear()
            return count

    def size(self) -> int:
        """Get the number of keys in storage.

        Returns:
            The count of keys.
        """
        return len(self._data)


# =============================================================================
# Memory Vector Backend
# =============================================================================


class MemoryVectorBackend:
    """In-memory vector storage with brute-force similarity search.

    This backend stores vectors in memory and performs exact nearest
    neighbor search using cosine similarity. Not suitable for large-scale
    production use but ideal for testing.

    Features:
        - Exact (not approximate) nearest neighbor search
        - Cosine similarity metric
        - Metadata support
        - Thread-safe operations

    Attributes:
        dimensions: The dimensionality of stored vectors.

    Example:
        >>> backend = MemoryVectorBackend(dimensions=128)
        >>> vectors = np.random.rand(10, 128).astype(np.float32)
        >>> ids = [f"vec_{i}" for i in range(10)]
        >>> await backend.add_vectors(vectors, ids)
        >>> query = np.random.rand(128).astype(np.float32)
        >>> results = await backend.search(query, k=5)
    """

    def __init__(self, dimensions: int = 1536) -> None:
        """Initialize the in-memory vector backend.

        Args:
            dimensions: The dimensionality of vectors to store.
        """
        self.dimensions = dimensions
        self._vectors: dict[str, np.ndarray] = {}
        self._metadata: dict[str, dict[str, Any]] = {}
        self._lock = asyncio.Lock()

    async def add_vectors(
        self,
        vectors: np.ndarray,
        ids: list[str],
        metadata: Optional[list[dict[str, Any]]] = None,
    ) -> None:
        """Add vectors with IDs and optional metadata.

        Args:
            vectors: A 2D numpy array of shape (n_vectors, dimensions).
            ids: List of unique IDs corresponding to each vector.
            metadata: Optional list of metadata dicts for each vector.

        Raises:
            ValueError: If the number of vectors doesn't match the number of IDs.
            ValueError: If vector dimensions don't match the backend dimensions.
        """
        if len(vectors) != len(ids):
            raise ValueError(
                f"Number of vectors ({len(vectors)}) must match number of IDs ({len(ids)})"
            )

        if len(vectors) > 0 and vectors.shape[1] != self.dimensions:
            raise ValueError(
                f"Vector dimensions ({vectors.shape[1]}) don't match backend dimensions ({self.dimensions})"
            )

        if metadata is not None and len(metadata) != len(ids):
            raise ValueError(
                f"Number of metadata entries ({len(metadata)}) must match number of IDs ({len(ids)})"
            )

        async with self._lock:
            for i, (vec_id, vec) in enumerate(zip(ids, vectors)):
                # Normalize for cosine similarity
                norm = np.linalg.norm(vec)
                if norm > 0:
                    vec = vec / norm
                self._vectors[vec_id] = vec.astype(np.float32)

                if metadata is not None:
                    self._metadata[vec_id] = metadata[i]

    async def search(
        self,
        query_vector: np.ndarray,
        k: int = 10,
        filter_metadata: Optional[dict[str, Any]] = None,
    ) -> list[tuple[str, float]]:
        """Search for k nearest vectors.

        Uses cosine similarity (dot product of normalized vectors).

        Args:
            query_vector: A 1D numpy array of shape (dimensions,).
            k: Number of results to return.
            filter_metadata: Optional metadata filter (key-value equality).

        Returns:
            List of (id, score) tuples, sorted by descending similarity.
        """
        if len(query_vector) != self.dimensions:
            raise ValueError(
                f"Query dimensions ({len(query_vector)}) don't match backend dimensions ({self.dimensions})"
            )

        async with self._lock:
            if not self._vectors:
                return []

            # Normalize query
            query_norm = np.linalg.norm(query_vector)
            if query_norm > 0:
                query_vector = query_vector / query_norm

            results: list[tuple[str, float]] = []

            for vec_id, vec in self._vectors.items():
                # Apply metadata filter if provided
                if filter_metadata is not None:
                    vec_meta = self._metadata.get(vec_id, {})
                    if not all(vec_meta.get(k) == v for k, v in filter_metadata.items()):
                        continue

                # Compute cosine similarity (dot product of normalized vectors)
                similarity = float(np.dot(query_vector, vec))
                results.append((vec_id, similarity))

            # Sort by similarity descending
            results.sort(key=lambda x: x[1], reverse=True)
            return results[:k]

    async def delete_vectors(self, ids: list[str]) -> int:
        """Delete vectors by ID.

        Args:
            ids: List of vector IDs to delete.

        Returns:
            The number of vectors actually deleted.
        """
        async with self._lock:
            count = 0
            for vec_id in ids:
                if vec_id in self._vectors:
                    del self._vectors[vec_id]
                    self._metadata.pop(vec_id, None)
                    count += 1
            return count

    async def get_vector(self, id: str) -> Optional[np.ndarray]:
        """Get a single vector by ID.

        Args:
            id: The unique ID of the vector.

        Returns:
            The vector as a numpy array if found, None otherwise.
        """
        async with self._lock:
            return self._vectors.get(id)

    async def count(self) -> int:
        """Get the total number of vectors in storage.

        Returns:
            The count of vectors.
        """
        async with self._lock:
            return len(self._vectors)

    def clear(self) -> None:
        """Clear all vectors and metadata.

        This is a synchronous method for easy use in test fixtures.
        """
        self._vectors.clear()
        self._metadata.clear()


# =============================================================================
# Memory Locking
# =============================================================================


class MemoryLocking:
    """In-memory locking using asyncio locks.

    This backend provides locking using asyncio.Lock objects. Locks are
    only valid within the same process and are not distributed.

    This implementation is useful for:
        - Unit testing without filesystem I/O
        - Single-process applications
        - Development and prototyping

    Example:
        >>> locking = MemoryLocking()
        >>> async with locking.lock("my-resource"):
        ...     # Critical section
        ...     pass
    """

    def __init__(self) -> None:
        """Initialize the in-memory locking backend."""
        self._locks: dict[str, asyncio.Lock] = {}
        self._held: set[str] = set()
        self._global_lock = asyncio.Lock()

    async def _get_or_create_lock(self, key: str) -> asyncio.Lock:
        """Get or create a lock for a key.

        Args:
            key: The lock key.

        Returns:
            The asyncio.Lock for this key.
        """
        async with self._global_lock:
            if key not in self._locks:
                self._locks[key] = asyncio.Lock()
            return self._locks[key]

    @asynccontextmanager
    async def lock(self, key: str, timeout: float = 30.0) -> AsyncIterator[None]:
        """Acquire a lock on key.

        Args:
            key: The unique key identifying the resource to lock.
            timeout: Maximum time in seconds to wait for the lock.

        Yields:
            Nothing. The lock is held while in the context.

        Raises:
            asyncio.TimeoutError: If the lock cannot be acquired within the timeout.
        """
        lock = await self._get_or_create_lock(key)

        try:
            await asyncio.wait_for(lock.acquire(), timeout=timeout)
            async with self._global_lock:
                self._held.add(key)
            yield
        finally:
            async with self._global_lock:
                self._held.discard(key)
            try:
                lock.release()
            except RuntimeError:
                # Lock was not held (possible in edge cases)
                pass

    async def is_locked(self, key: str) -> bool:
        """Check if a key is currently locked.

        Args:
            key: The unique key to check.

        Returns:
            True if the key is currently locked, False otherwise.
        """
        async with self._global_lock:
            if key not in self._locks:
                return False
            return self._locks[key].locked()

    async def try_lock(self, key: str) -> bool:
        """Attempt to acquire a lock without blocking.

        Args:
            key: The unique key identifying the resource to lock.

        Returns:
            True if the lock was acquired, False if it's already held.
        """
        lock = await self._get_or_create_lock(key)

        if lock.locked():
            return False

        try:
            # Try to acquire without blocking
            acquired = lock.locked() is False
            if acquired:
                await lock.acquire()
                async with self._global_lock:
                    self._held.add(key)
            return acquired
        except Exception:
            return False

    async def release_lock(self, key: str) -> bool:
        """Release a lock.

        Args:
            key: The unique key identifying the resource to unlock.

        Returns:
            True if the lock was released, False if it wasn't held.
        """
        async with self._global_lock:
            if key not in self._locks:
                return False
            if key not in self._held:
                return False

            try:
                self._locks[key].release()
                self._held.discard(key)
                return True
            except RuntimeError:
                return False

    def clear(self) -> None:
        """Clear all locks.

        Warning: This does not release held locks gracefully.
        Only use in test cleanup.
        """
        self._locks.clear()
        self._held.clear()


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "MemoryBackend",
    "MemoryVectorBackend",
    "MemoryLocking",
]
