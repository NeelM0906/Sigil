"""Storage backend abstraction for Sigil memory system.

This module provides a unified interface for storage operations across
different backend implementations. The abstraction allows the memory
layers to work with filesystem, in-memory, or cloud storage without
code changes.

Architecture:
    The storage system uses a protocol-based design:

    1. Protocols (interfaces):
       - StorageBackend: Key-value storage operations
       - VectorBackend: Vector storage and similarity search
       - LockingBackend: Distributed locking

    2. Implementations:
       - FilesystemBackend / FilesystemLocking: Local filesystem
       - MemoryBackend / MemoryLocking: In-memory (for testing)

    3. Factory:
       - BackendFactory: Creates backend instances from configuration

Usage:
    >>> from sigil.core.storage import BackendFactory
    >>> from pathlib import Path
    >>>
    >>> # Create backends from configuration
    >>> storage = BackendFactory.create_storage(
    ...     "filesystem",
    ...     base_path=Path("./data/storage")
    ... )
    >>> locking = BackendFactory.create_locking(
    ...     "filesystem",
    ...     lock_dir=Path("./data/locks")
    ... )
    >>>
    >>> # Use the storage backend
    >>> await storage.write("my-key", b"my-data")
    >>> data = await storage.read("my-key")
    >>>
    >>> # Use locking for critical sections
    >>> async with locking.lock("my-resource"):
    ...     await do_critical_work()

Testing Usage:
    >>> from sigil.core.storage import BackendFactory
    >>>
    >>> # Use in-memory backends for fast, isolated tests
    >>> storage = BackendFactory.create_storage("memory")
    >>> locking = BackendFactory.create_locking("memory")
    >>>
    >>> # Tests can clear state easily
    >>> storage.clear()

Configuration:
    The backend type can be configured via settings:

    ```python
    from sigil.config import get_settings

    settings = get_settings()
    storage = BackendFactory.create_storage(
        settings.memory.backend,
        base_path=settings.paths.memory_dir
    )
    ```

Extending:
    To add a new storage backend:

    1. Create a class implementing StorageBackend protocol
    2. Register it in BackendFactory.STORAGE_BACKENDS
    3. Add creation logic in BackendFactory.create_storage()

    Example for S3 backend:
    ```python
    class S3Backend:
        async def read(self, key: str) -> Optional[bytes]:
            # S3 implementation
            ...
    ```
"""

from sigil.core.storage.protocols import (
    StorageBackend,
    VectorBackend,
    LockingBackend,
)
from sigil.core.storage.filesystem import (
    FilesystemBackend,
    FilesystemLocking,
    LockTimeout,
    StorageError,
)
from sigil.core.storage.memory import (
    MemoryBackend,
    MemoryVectorBackend,
    MemoryLocking,
)
from sigil.core.storage.factory import BackendFactory


__all__ = [
    # Protocols
    "StorageBackend",
    "VectorBackend",
    "LockingBackend",
    # Filesystem implementations
    "FilesystemBackend",
    "FilesystemLocking",
    # Memory implementations
    "MemoryBackend",
    "MemoryVectorBackend",
    "MemoryLocking",
    # Factory
    "BackendFactory",
    # Exceptions
    "LockTimeout",
    "StorageError",
]
