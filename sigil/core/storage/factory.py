"""Backend factory for creating storage instances from configuration.

This module provides a factory class for creating storage backend instances
based on configuration settings. It abstracts the instantiation details
and provides a clean interface for the memory layers.

Classes:
    BackendFactory: Factory for creating storage backend instances.

Example:
    >>> from sigil.core.storage.factory import BackendFactory
    >>> from pathlib import Path
    >>>
    >>> # Create backends from configuration
    >>> storage = BackendFactory.create_storage("filesystem", base_path=Path("./data"))
    >>> locking = BackendFactory.create_locking("filesystem", lock_dir=Path("./locks"))
    >>>
    >>> # Or use in-memory backends for testing
    >>> storage = BackendFactory.create_storage("memory")
    >>> locking = BackendFactory.create_locking("memory")
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Union

from sigil.core.storage.protocols import StorageBackend, VectorBackend, LockingBackend
from sigil.core.storage.filesystem import FilesystemBackend, FilesystemLocking
from sigil.core.storage.memory import MemoryBackend, MemoryVectorBackend, MemoryLocking


logger = logging.getLogger(__name__)


# =============================================================================
# Backend Factory
# =============================================================================


class BackendFactory:
    """Factory for creating storage backend instances.

    This factory provides static methods for creating various storage
    backend instances based on type strings and configuration parameters.
    It centralizes backend instantiation logic and validates inputs.

    Supported backend types:
        - "filesystem": Local filesystem storage (default)
        - "memory": In-memory storage (for testing)

    Future backends (not yet implemented):
        - "s3": Amazon S3 storage
        - "gcs": Google Cloud Storage
        - "postgres": PostgreSQL storage
        - "redis": Redis storage

    Example:
        >>> # From settings
        >>> from sigil.config import get_settings
        >>> settings = get_settings()
        >>> storage = BackendFactory.create_storage(
        ...     settings.memory.backend,
        ...     base_path=settings.paths.memory_dir
        ... )
    """

    # Registry of supported backend types
    STORAGE_BACKENDS = frozenset({"filesystem", "memory"})
    VECTOR_BACKENDS = frozenset({"memory", "faiss"})  # FAISS handled separately
    LOCKING_BACKENDS = frozenset({"filesystem", "memory"})

    @staticmethod
    def create_storage(
        backend_type: str = "filesystem",
        base_path: Optional[Union[str, Path]] = None,
        **kwargs,
    ) -> StorageBackend:
        """Create a storage backend instance.

        Args:
            backend_type: One of "filesystem", "memory".
            base_path: Base path for filesystem backend. Required if
                backend_type is "filesystem".
            **kwargs: Additional backend-specific options (reserved for future use).

        Returns:
            A StorageBackend instance.

        Raises:
            ValueError: If backend_type is unknown or required parameters are missing.

        Example:
            >>> # Filesystem backend
            >>> storage = BackendFactory.create_storage(
            ...     "filesystem",
            ...     base_path=Path("./data/storage")
            ... )
            >>>
            >>> # Memory backend (for testing)
            >>> storage = BackendFactory.create_storage("memory")
        """
        backend_type = backend_type.lower().strip()

        if backend_type not in BackendFactory.STORAGE_BACKENDS:
            raise ValueError(
                f"Unknown storage backend type: '{backend_type}'. "
                f"Supported types: {', '.join(sorted(BackendFactory.STORAGE_BACKENDS))}"
            )

        if backend_type == "filesystem":
            if base_path is None:
                raise ValueError(
                    "base_path is required for filesystem storage backend. "
                    "Example: BackendFactory.create_storage('filesystem', base_path=Path('./data'))"
                )
            path = Path(base_path) if isinstance(base_path, str) else base_path
            logger.debug(f"Creating FilesystemBackend at {path}")
            return FilesystemBackend(path)

        elif backend_type == "memory":
            logger.debug("Creating MemoryBackend")
            return MemoryBackend()

        # This should never be reached due to the check above
        raise ValueError(f"Unknown storage backend type: '{backend_type}'")

    @staticmethod
    def create_vector_backend(
        backend_type: str = "memory",
        dimensions: int = 1536,
        storage_dir: Optional[Union[str, Path]] = None,
        **kwargs,
    ) -> VectorBackend:
        """Create a vector storage backend instance.

        Args:
            backend_type: One of "memory", "faiss".
            dimensions: Dimensionality of vectors.
            storage_dir: Directory for FAISS index persistence.
            **kwargs: Additional backend-specific options.

        Returns:
            A VectorBackend instance.

        Raises:
            ValueError: If backend_type is unknown.

        Note:
            For "faiss" backend, the existing VectorIndex class from
            sigil.memory.layers.items is used. This factory method is
            primarily for the "memory" backend used in testing.

        Example:
            >>> # Memory backend (for testing)
            >>> vector_storage = BackendFactory.create_vector_backend(
            ...     "memory",
            ...     dimensions=1536
            ... )
        """
        backend_type = backend_type.lower().strip()

        if backend_type == "memory":
            logger.debug(f"Creating MemoryVectorBackend with {dimensions} dimensions")
            return MemoryVectorBackend(dimensions=dimensions)

        elif backend_type == "faiss":
            # FAISS backend uses the existing VectorIndex implementation
            # which is not a direct implementation of VectorBackend protocol
            # This is a placeholder for future integration
            raise ValueError(
                "FAISS backend should be created using sigil.memory.layers.items.VectorIndex "
                "directly. Use 'memory' for testing purposes."
            )

        raise ValueError(
            f"Unknown vector backend type: '{backend_type}'. "
            f"Supported types: {', '.join(sorted(BackendFactory.VECTOR_BACKENDS))}"
        )

    @staticmethod
    def create_locking(
        backend_type: str = "filesystem",
        lock_dir: Optional[Union[str, Path]] = None,
        **kwargs,
    ) -> LockingBackend:
        """Create a locking backend instance.

        Args:
            backend_type: One of "filesystem", "memory".
            lock_dir: Directory for lock files (required for filesystem backend).
            **kwargs: Additional backend-specific options (reserved for future use).

        Returns:
            A LockingBackend instance.

        Raises:
            ValueError: If backend_type is unknown or required parameters are missing.

        Example:
            >>> # Filesystem locking
            >>> locking = BackendFactory.create_locking(
            ...     "filesystem",
            ...     lock_dir=Path("./data/locks")
            ... )
            >>>
            >>> # Memory locking (for testing)
            >>> locking = BackendFactory.create_locking("memory")
        """
        backend_type = backend_type.lower().strip()

        if backend_type not in BackendFactory.LOCKING_BACKENDS:
            raise ValueError(
                f"Unknown locking backend type: '{backend_type}'. "
                f"Supported types: {', '.join(sorted(BackendFactory.LOCKING_BACKENDS))}"
            )

        if backend_type == "filesystem":
            if lock_dir is None:
                raise ValueError(
                    "lock_dir is required for filesystem locking backend. "
                    "Example: BackendFactory.create_locking('filesystem', lock_dir=Path('./locks'))"
                )
            path = Path(lock_dir) if isinstance(lock_dir, str) else lock_dir
            logger.debug(f"Creating FilesystemLocking at {path}")
            return FilesystemLocking(path)

        elif backend_type == "memory":
            logger.debug("Creating MemoryLocking")
            return MemoryLocking()

        # This should never be reached due to the check above
        raise ValueError(f"Unknown locking backend type: '{backend_type}'")

    @staticmethod
    def create_all_from_settings(
        backend_type: str = "filesystem",
        base_path: Optional[Union[str, Path]] = None,
        lock_dir: Optional[Union[str, Path]] = None,
    ) -> tuple[StorageBackend, LockingBackend]:
        """Create both storage and locking backends from settings.

        Convenience method for creating all backends at once with
        consistent configuration.

        Args:
            backend_type: Backend type for all components.
            base_path: Base path for storage (filesystem backend).
            lock_dir: Directory for locks (filesystem backend).
                If None, uses base_path/locks.

        Returns:
            Tuple of (StorageBackend, LockingBackend).

        Example:
            >>> storage, locking = BackendFactory.create_all_from_settings(
            ...     backend_type="filesystem",
            ...     base_path=Path("./data")
            ... )
        """
        if backend_type == "memory":
            return (
                BackendFactory.create_storage("memory"),
                BackendFactory.create_locking("memory"),
            )

        if base_path is None:
            raise ValueError("base_path is required for filesystem backends")

        base_path = Path(base_path) if isinstance(base_path, str) else base_path

        if lock_dir is None:
            lock_dir = base_path / "locks"

        return (
            BackendFactory.create_storage("filesystem", base_path=base_path),
            BackendFactory.create_locking("filesystem", lock_dir=lock_dir),
        )

    @staticmethod
    def get_supported_backends() -> dict[str, frozenset[str]]:
        """Get all supported backend types.

        Returns:
            Dictionary mapping backend category to supported types.

        Example:
            >>> backends = BackendFactory.get_supported_backends()
            >>> print(backends["storage"])
            frozenset({'filesystem', 'memory'})
        """
        return {
            "storage": BackendFactory.STORAGE_BACKENDS,
            "vector": BackendFactory.VECTOR_BACKENDS,
            "locking": BackendFactory.LOCKING_BACKENDS,
        }


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "BackendFactory",
]
