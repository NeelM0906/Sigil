"""Filesystem storage backend implementation.

This module provides filesystem-based implementations of the storage protocols.
It uses the local filesystem for persistent storage with file-based locking
for concurrent access safety.

Classes:
    FilesystemBackend: Key-value storage using local files.
    FilesystemLocking: File-based locking using portalocker.

Example:
    >>> from sigil.core.storage.filesystem import FilesystemBackend, FilesystemLocking
    >>> from pathlib import Path
    >>>
    >>> backend = FilesystemBackend(Path("./data"))
    >>> await backend.write("my-key", b"hello world")
    >>> data = await backend.read("my-key")
    >>> print(data)
    b'hello world'
"""

from __future__ import annotations

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncContextManager, AsyncIterator, Optional

import aiofiles
import portalocker

from sigil.core.exceptions import SigilMemoryError, SigilMemoryWriteError, SigilMemoryRetrievalError


logger = logging.getLogger(__name__)


# =============================================================================
# Exceptions
# =============================================================================


class LockTimeout(SigilMemoryError):
    """Raised when a lock cannot be acquired within the timeout."""

    def __init__(self, key: str, timeout: float):
        super().__init__(
            f"Failed to acquire lock for '{key}' within {timeout}s",
            layer="storage",
        )
        self.key = key
        self.timeout = timeout


class StorageError(SigilMemoryError):
    """Raised when a storage operation fails."""

    def __init__(self, message: str, operation: str, key: Optional[str] = None):
        super().__init__(message, layer="storage")
        self.operation = operation
        self.key = key


# =============================================================================
# Filesystem Backend
# =============================================================================


class FilesystemBackend:
    """Filesystem-based key-value storage backend.

    This backend stores data as files in a directory. Keys are mapped to
    file paths within the base directory. The implementation sanitizes
    keys to prevent directory traversal attacks.

    Features:
        - Atomic writes using temporary files
        - Automatic directory creation
        - Key sanitization for security
        - Support for nested key paths

    Attributes:
        base_path: The root directory for storage.

    Example:
        >>> backend = FilesystemBackend(Path("./storage"))
        >>> await backend.write("users/123", b'{"name": "John"}')
        >>> data = await backend.read("users/123")
    """

    def __init__(self, base_path: Path) -> None:
        """Initialize the filesystem backend.

        Args:
            base_path: The root directory for storage.
                Will be created if it doesn't exist.
        """
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

    def _key_to_path(self, key: str) -> Path:
        """Convert a key to a filesystem path.

        This method sanitizes the key to prevent directory traversal
        attacks while still supporting hierarchical keys using forward
        slashes.

        Args:
            key: The storage key.

        Returns:
            The sanitized file path.
        """
        # Replace dangerous patterns
        safe_key = key.replace("..", "_").replace("\\", "/")

        # Split on forward slash for hierarchical keys
        parts = [p for p in safe_key.split("/") if p and p != "."]

        # Sanitize each part
        sanitized_parts = []
        for part in parts:
            # Replace any remaining problematic characters
            safe_part = "".join(c if c.isalnum() or c in "._-" else "_" for c in part)
            if safe_part:
                sanitized_parts.append(safe_part)

        if not sanitized_parts:
            sanitized_parts = ["_default"]

        return self.base_path.joinpath(*sanitized_parts)

    async def read(self, key: str) -> Optional[bytes]:
        """Read data by key.

        Args:
            key: The unique key identifying the data.

        Returns:
            The data as bytes if found, None if the key does not exist.

        Raises:
            StorageError: If there's an error reading from storage.
        """
        path = self._key_to_path(key)

        if not path.exists():
            return None

        try:
            async with aiofiles.open(path, "rb") as f:
                return await f.read()
        except PermissionError as e:
            raise StorageError(
                f"Permission denied reading key '{key}': {e}",
                operation="read",
                key=key,
            )
        except Exception as e:
            raise StorageError(
                f"Error reading key '{key}': {e}",
                operation="read",
                key=key,
            )

    async def write(self, key: str, data: bytes) -> None:
        """Write data to key.

        Uses a temporary file and rename for atomic writes.

        Args:
            key: The unique key identifying the data.
            data: The data to store as bytes.

        Raises:
            StorageError: If there's an error writing to storage.
        """
        path = self._key_to_path(key)

        # Ensure parent directory exists
        path.parent.mkdir(parents=True, exist_ok=True)

        try:
            # Write to temporary file first for atomic operation
            temp_path = path.with_suffix(path.suffix + ".tmp")
            async with aiofiles.open(temp_path, "wb") as f:
                await f.write(data)

            # Atomic rename
            temp_path.rename(path)

        except PermissionError as e:
            raise StorageError(
                f"Permission denied writing key '{key}': {e}",
                operation="write",
                key=key,
            )
        except Exception as e:
            # Clean up temp file if it exists
            if temp_path.exists():
                try:
                    temp_path.unlink()
                except Exception:
                    pass
            raise StorageError(
                f"Error writing key '{key}': {e}",
                operation="write",
                key=key,
            )

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
        path = self._key_to_path(key)

        if not path.exists():
            return False

        try:
            path.unlink()
            return True
        except PermissionError as e:
            raise StorageError(
                f"Permission denied deleting key '{key}': {e}",
                operation="delete",
                key=key,
            )
        except Exception as e:
            raise StorageError(
                f"Error deleting key '{key}': {e}",
                operation="delete",
                key=key,
            )

    async def exists(self, key: str) -> bool:
        """Check if a key exists in storage.

        Args:
            key: The unique key to check.

        Returns:
            True if the key exists, False otherwise.
        """
        path = self._key_to_path(key)
        return path.exists() and path.is_file()

    async def list_keys(self, prefix: str = "") -> list[str]:
        """List keys with optional prefix filter.

        Args:
            prefix: Optional prefix to filter keys. If empty, lists all keys.

        Returns:
            List of key strings matching the prefix.

        Raises:
            StorageError: If there's an error listing keys.
        """
        try:
            keys = []

            if prefix:
                # Get the path for the prefix
                prefix_path = self._key_to_path(prefix)

                # If prefix is a directory, list files in it
                if prefix_path.is_dir():
                    for path in prefix_path.rglob("*"):
                        if path.is_file():
                            # Reconstruct key from path
                            relative = path.relative_to(self.base_path)
                            keys.append(str(relative))
                # If prefix is a partial filename, find matches
                else:
                    parent = prefix_path.parent
                    if parent.exists():
                        pattern = prefix_path.name + "*"
                        for path in parent.glob(pattern):
                            if path.is_file():
                                relative = path.relative_to(self.base_path)
                                keys.append(str(relative))
            else:
                # List all files
                for path in self.base_path.rglob("*"):
                    if path.is_file() and not path.name.endswith(".tmp"):
                        relative = path.relative_to(self.base_path)
                        keys.append(str(relative))

            return sorted(keys)

        except Exception as e:
            raise StorageError(
                f"Error listing keys with prefix '{prefix}': {e}",
                operation="list_keys",
                key=prefix,
            )

    async def clear(self) -> int:
        """Clear all data from storage.

        Returns:
            The number of keys deleted.
        """
        count = 0
        for path in self.base_path.rglob("*"):
            if path.is_file():
                try:
                    path.unlink()
                    count += 1
                except Exception:
                    pass
        return count


# =============================================================================
# Filesystem Locking
# =============================================================================


class FilesystemLocking:
    """File-based locking using portalocker.

    This backend provides distributed locking using lock files. It uses
    portalocker for cross-platform file locking support.

    Features:
        - Cross-platform support (Windows, Linux, macOS)
        - Automatic lock file cleanup
        - Timeout support
        - Reentrant locks within same process

    Attributes:
        lock_dir: The directory for lock files.

    Example:
        >>> locking = FilesystemLocking(Path("./locks"))
        >>> async with locking.lock("my-resource", timeout=10.0):
        ...     # Critical section
        ...     pass
    """

    def __init__(self, lock_dir: Path) -> None:
        """Initialize the filesystem locking backend.

        Args:
            lock_dir: Directory for storing lock files.
                Will be created if it doesn't exist.
        """
        self.lock_dir = Path(lock_dir)
        self.lock_dir.mkdir(parents=True, exist_ok=True)
        self._held_locks: dict[str, Any] = {}
        self._lock = asyncio.Lock()

    def _lock_path(self, key: str) -> Path:
        """Get the path for a lock file.

        Args:
            key: The lock key.

        Returns:
            Path to the lock file.
        """
        # Sanitize key for filename
        safe_key = key.replace("..", "_").replace("/", "_").replace("\\", "_")
        safe_key = "".join(c if c.isalnum() or c in "._-" else "_" for c in safe_key)
        return self.lock_dir / f"{safe_key}.lock"

    @asynccontextmanager
    async def lock(self, key: str, timeout: float = 30.0) -> AsyncIterator[None]:
        """Acquire a lock on key.

        Args:
            key: The unique key identifying the resource to lock.
            timeout: Maximum time in seconds to wait for the lock.

        Yields:
            Nothing. The lock is held while in the context.

        Raises:
            LockTimeout: If the lock cannot be acquired within the timeout.
        """
        lock_path = self._lock_path(key)

        # Ensure lock directory exists
        lock_path.parent.mkdir(parents=True, exist_ok=True)

        lock_obj = None
        try:
            # Run the blocking lock acquisition in a thread pool
            def acquire_lock():
                nonlocal lock_obj
                try:
                    # Use portalocker.Lock class which supports timeout
                    lock_obj = portalocker.Lock(
                        str(lock_path),
                        mode="w",
                        timeout=timeout,
                        flags=portalocker.LOCK_EX,
                    )
                    lock_obj.acquire()
                    return True
                except (portalocker.LockException, portalocker.AlreadyLocked):
                    if lock_obj is not None:
                        try:
                            lock_obj.release()
                        except Exception:
                            pass
                    return False

            loop = asyncio.get_event_loop()
            acquired = await loop.run_in_executor(None, acquire_lock)

            if not acquired:
                raise LockTimeout(key, timeout)

            async with self._lock:
                self._held_locks[key] = lock_obj

            yield

        finally:
            # Release the lock
            async with self._lock:
                if key in self._held_locks:
                    held_lock = self._held_locks.pop(key)
                    try:
                        held_lock.release()
                    except Exception:
                        pass
                elif lock_obj is not None:
                    try:
                        lock_obj.release()
                    except Exception:
                        pass

    async def is_locked(self, key: str) -> bool:
        """Check if a key is currently locked.

        Note: This is a point-in-time check. The lock status may change
        immediately after this method returns.

        Args:
            key: The unique key to check.

        Returns:
            True if the key is currently locked, False otherwise.
        """
        async with self._lock:
            return key in self._held_locks

    async def try_lock(self, key: str) -> bool:
        """Attempt to acquire a lock without blocking.

        Args:
            key: The unique key identifying the resource to lock.

        Returns:
            True if the lock was acquired, False if it's already held.
        """
        lock_path = self._lock_path(key)
        lock_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            # Use portalocker.Lock with fail_when_locked=True for non-blocking
            lock_obj = portalocker.Lock(
                str(lock_path),
                mode="w",
                timeout=0,
                fail_when_locked=True,
                flags=portalocker.LOCK_EX | portalocker.LOCK_NB,
            )
            try:
                lock_obj.acquire()
                async with self._lock:
                    self._held_locks[key] = lock_obj
                return True
            except (portalocker.LockException, portalocker.AlreadyLocked):
                try:
                    lock_obj.release()
                except Exception:
                    pass
                return False
        except Exception:
            return False

    async def release_lock(self, key: str) -> bool:
        """Release a lock.

        Args:
            key: The unique key identifying the resource to unlock.

        Returns:
            True if the lock was released, False if it wasn't held.
        """
        async with self._lock:
            if key not in self._held_locks:
                return False

            lock_obj = self._held_locks.pop(key)
            try:
                lock_obj.release()
            except Exception:
                pass
            return True


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "FilesystemBackend",
    "FilesystemLocking",
    "LockTimeout",
    "StorageError",
]
