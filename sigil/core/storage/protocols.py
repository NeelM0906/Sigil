"""Storage backend protocols for Sigil memory system.

This module defines the abstract protocols (interfaces) for storage backends.
These protocols allow the memory layers to work with different storage
implementations (filesystem, in-memory, S3, PostgreSQL, etc.) without
changing the layer code.

Protocols:
    StorageBackend: Key-value storage operations (read, write, delete, list).
    VectorBackend: Vector storage and similarity search operations.
    LockingBackend: Distributed locking for concurrent access.

Example:
    >>> from sigil.core.storage.protocols import StorageBackend
    >>>
    >>> class MyCustomBackend:
    ...     async def read(self, key: str) -> Optional[bytes]:
    ...         ...
    ...     # ... implement other protocol methods
    >>>
    >>> def uses_storage(backend: StorageBackend):
    ...     data = await backend.read("my-key")
"""

from __future__ import annotations

from abc import abstractmethod
from contextlib import asynccontextmanager
from typing import Any, AsyncContextManager, AsyncIterator, Optional, Protocol, runtime_checkable

import numpy as np


# =============================================================================
# Storage Backend Protocol
# =============================================================================


@runtime_checkable
class StorageBackend(Protocol):
    """Protocol for key-value storage operations.

    This protocol defines the interface for basic storage operations that
    abstract away the underlying storage mechanism. Implementations can
    use local filesystem, cloud storage (S3, GCS), databases, or in-memory
    storage.

    All operations are async to support non-blocking I/O in async contexts.

    Methods:
        read: Read data by key.
        write: Write data to key.
        delete: Delete data by key.
        exists: Check if key exists.
        list_keys: List keys with optional prefix filter.

    Example:
        >>> async def store_data(backend: StorageBackend, key: str, data: bytes):
        ...     await backend.write(key, data)
        ...     exists = await backend.exists(key)
        ...     assert exists
    """

    async def read(self, key: str) -> Optional[bytes]:
        """Read data by key.

        Args:
            key: The unique key identifying the data.

        Returns:
            The data as bytes if found, None if the key does not exist.

        Raises:
            StorageError: If there's an error reading from storage.
        """
        ...

    async def write(self, key: str, data: bytes) -> None:
        """Write data to key.

        This operation is idempotent - writing the same data to the same
        key multiple times has the same effect as writing once.

        Args:
            key: The unique key identifying the data.
            data: The data to store as bytes.

        Raises:
            StorageError: If there's an error writing to storage.
        """
        ...

    async def delete(self, key: str) -> bool:
        """Delete data by key.

        Args:
            key: The unique key identifying the data to delete.

        Returns:
            True if the key was found and deleted, False if the key
            did not exist.

        Raises:
            StorageError: If there's an error deleting from storage.
        """
        ...

    async def exists(self, key: str) -> bool:
        """Check if a key exists in storage.

        Args:
            key: The unique key to check.

        Returns:
            True if the key exists, False otherwise.

        Raises:
            StorageError: If there's an error checking storage.
        """
        ...

    async def list_keys(self, prefix: str = "") -> list[str]:
        """List keys with optional prefix filter.

        Args:
            prefix: Optional prefix to filter keys. If empty, lists all keys.

        Returns:
            List of key strings matching the prefix.

        Raises:
            StorageError: If there's an error listing keys.
        """
        ...


# =============================================================================
# Vector Backend Protocol
# =============================================================================


@runtime_checkable
class VectorBackend(Protocol):
    """Protocol for vector storage and similarity search operations.

    This protocol defines the interface for storing embedding vectors and
    performing approximate nearest neighbor (ANN) search. Implementations
    can use FAISS, Annoy, ScaNN, Pinecone, or other vector databases.

    Vectors are identified by string IDs and can have associated metadata.

    Methods:
        add_vectors: Add vectors with IDs and optional metadata.
        search: Search for k nearest vectors to a query.
        delete_vectors: Delete vectors by ID.
        get_vector: Retrieve a single vector by ID.
        count: Get the total number of vectors.

    Example:
        >>> async def semantic_search(backend: VectorBackend, query: np.ndarray):
        ...     results = await backend.search(query, k=10)
        ...     for vec_id, score in results:
        ...         print(f"{vec_id}: {score}")
    """

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
            StorageError: If there's an error adding vectors.
        """
        ...

    async def search(
        self,
        query_vector: np.ndarray,
        k: int = 10,
        filter_metadata: Optional[dict[str, Any]] = None,
    ) -> list[tuple[str, float]]:
        """Search for k nearest vectors.

        Args:
            query_vector: A 1D numpy array of shape (dimensions,).
            k: Number of results to return.
            filter_metadata: Optional metadata filter (implementation-specific).

        Returns:
            List of (id, score) tuples, sorted by descending similarity.
            Score interpretation depends on the distance metric used.

        Raises:
            StorageError: If there's an error searching vectors.
        """
        ...

    async def delete_vectors(self, ids: list[str]) -> int:
        """Delete vectors by ID.

        Args:
            ids: List of vector IDs to delete.

        Returns:
            The number of vectors actually deleted.

        Raises:
            StorageError: If there's an error deleting vectors.
        """
        ...

    async def get_vector(self, id: str) -> Optional[np.ndarray]:
        """Get a single vector by ID.

        Args:
            id: The unique ID of the vector.

        Returns:
            The vector as a numpy array if found, None otherwise.

        Raises:
            StorageError: If there's an error retrieving the vector.
        """
        ...

    async def count(self) -> int:
        """Get the total number of vectors in storage.

        Returns:
            The count of vectors.
        """
        ...


# =============================================================================
# Locking Backend Protocol
# =============================================================================


@runtime_checkable
class LockingBackend(Protocol):
    """Protocol for distributed locking.

    This protocol defines the interface for acquiring and releasing locks
    on resources. Implementations can use file-based locks, Redis locks,
    database locks, or in-memory locks.

    Locks are identified by string keys and support timeout to prevent
    deadlocks.

    Methods:
        lock: Acquire a lock on a key (async context manager).
        is_locked: Check if a key is currently locked.
        try_lock: Attempt to acquire a lock without blocking.

    Example:
        >>> async def critical_section(locking: LockingBackend, key: str):
        ...     async with locking.lock(key, timeout=30.0):
        ...         # Only one process can be here at a time
        ...         await do_critical_work()
    """

    def lock(self, key: str, timeout: float = 30.0) -> AsyncContextManager[None]:
        """Acquire a lock on key.

        This method returns an async context manager that acquires the lock
        on entry and releases it on exit.

        Args:
            key: The unique key identifying the resource to lock.
            timeout: Maximum time in seconds to wait for the lock.

        Returns:
            An async context manager that holds the lock.

        Raises:
            LockTimeout: If the lock cannot be acquired within the timeout.
            StorageError: If there's an error with the locking mechanism.

        Example:
            >>> async with locking.lock("my-resource", timeout=10.0):
            ...     # Lock is held here
            ...     pass
            >>> # Lock is released here
        """
        ...

    async def is_locked(self, key: str) -> bool:
        """Check if a key is currently locked.

        Note: This is a point-in-time check. The lock status may change
        immediately after this method returns.

        Args:
            key: The unique key to check.

        Returns:
            True if the key is currently locked, False otherwise.
        """
        ...

    async def try_lock(self, key: str) -> bool:
        """Attempt to acquire a lock without blocking.

        This method tries to acquire the lock immediately and returns
        whether the acquisition was successful. If successful, the caller
        is responsible for releasing the lock.

        Args:
            key: The unique key identifying the resource to lock.

        Returns:
            True if the lock was acquired, False if it's already held.

        Note:
            If this returns True, you must call release_lock() when done.
        """
        ...

    async def release_lock(self, key: str) -> bool:
        """Release a lock.

        Args:
            key: The unique key identifying the resource to unlock.

        Returns:
            True if the lock was released, False if it wasn't held.
        """
        ...


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "StorageBackend",
    "VectorBackend",
    "LockingBackend",
]
