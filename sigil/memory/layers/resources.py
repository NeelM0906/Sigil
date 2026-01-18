"""Layer 1: Resource storage for the Sigil v2 3-layer memory architecture.

This module implements the Resource Layer, which stores raw source data including
conversations, documents, configurations, and feedback. Resources provide full
traceability for memory items extracted from them.

Classes:
    ResourceLayer: Main class for storing and retrieving resources.
    ConversationResource: Specialized handling for conversation-type resources.

Storage Format:
    Resources are stored as JSON files in the configured storage directory:
    {storage_dir}/{resource_type}/{resource_id}.json

Example:
    >>> from sigil.memory.layers.resources import ResourceLayer
    >>> layer = ResourceLayer(storage_dir="outputs/memory/resources")
    >>> resource = layer.store(
    ...     resource_type="conversation",
    ...     content="User: Hello\\nAgent: Hi there!",
    ...     metadata={"session_id": "sess-123"}
    ... )
    >>> retrieved = layer.get(resource.resource_id)
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Optional

import portalocker

from sigil.config.schemas.memory import Resource, generate_uuid, utc_now
from sigil.core.exceptions import SigilMemoryError, SigilMemoryWriteError, SigilMemoryRetrievalError


# =============================================================================
# Constants
# =============================================================================

VALID_RESOURCE_TYPES = frozenset({"conversation", "document", "config", "feedback"})
"""Supported resource types in the memory system."""

DEFAULT_STORAGE_DIR = "outputs/memory/resources"
"""Default directory for resource storage."""


# =============================================================================
# ConversationResource
# =============================================================================


class ConversationResource:
    """Specialized handling for conversation-type resources.

    This class provides utilities for parsing, formatting, and manipulating
    conversation transcripts stored as resources. It handles multi-turn
    conversations with speaker identification.

    Attributes:
        resource: The underlying Resource object.
        turns: List of conversation turns (role, content) tuples.

    Example:
        >>> conv = ConversationResource.from_resource(resource)
        >>> for role, content in conv.turns:
        ...     print(f"{role}: {content}")
        >>> conv.add_turn("user", "What's the weather?")
        >>> conv.add_turn("assistant", "It's sunny today!")
    """

    def __init__(self, resource: Resource) -> None:
        """Initialize a ConversationResource from an existing Resource.

        Args:
            resource: The underlying Resource object (must be type 'conversation').

        Raises:
            ValueError: If the resource is not a conversation type.
        """
        if resource.resource_type != "conversation":
            raise ValueError(
                f"Expected conversation resource, got {resource.resource_type}"
            )
        self.resource = resource
        self._turns: list[tuple[str, str]] = []
        self._parse_content()

    def _parse_content(self) -> None:
        """Parse conversation content into turns."""
        self._turns = []
        content = self.resource.content

        # Parse format: "Role: Message" with newline separators
        # Support for multi-line messages (subsequent lines without "Role:" prefix)
        current_role: Optional[str] = None
        current_content: list[str] = []

        for line in content.split("\n"):
            # Check for role prefix (User:, Assistant:, System:, etc.)
            match = re.match(r"^(User|Assistant|System|Agent|Human):\s*(.*)$", line, re.IGNORECASE)
            if match:
                # Save previous turn if exists
                if current_role is not None:
                    self._turns.append((current_role, "\n".join(current_content).strip()))
                # Start new turn
                current_role = match.group(1).lower()
                current_content = [match.group(2)] if match.group(2) else []
            elif current_role is not None:
                # Continue previous turn
                current_content.append(line)

        # Don't forget the last turn
        if current_role is not None and current_content:
            self._turns.append((current_role, "\n".join(current_content).strip()))

    @property
    def turns(self) -> list[tuple[str, str]]:
        """Get the list of conversation turns.

        Returns:
            List of (role, content) tuples.
        """
        return self._turns.copy()

    @property
    def turn_count(self) -> int:
        """Get the number of turns in the conversation."""
        return len(self._turns)

    def add_turn(self, role: str, content: str) -> None:
        """Add a new turn to the conversation.

        Args:
            role: The speaker role (user, assistant, system).
            content: The message content.
        """
        role = role.lower()
        self._turns.append((role, content))
        self._update_resource_content()

    def _update_resource_content(self) -> None:
        """Update the underlying resource content from turns."""
        lines = []
        for role, content in self._turns:
            # Capitalize first letter of role
            formatted_role = role.capitalize()
            lines.append(f"{formatted_role}: {content}")

        # Create new resource with updated content
        self.resource = Resource(
            resource_id=self.resource.resource_id,
            resource_type=self.resource.resource_type,
            content="\n".join(lines),
            metadata=self.resource.metadata,
            created_at=self.resource.created_at,
        )

    def get_last_n_turns(self, n: int) -> list[tuple[str, str]]:
        """Get the last N turns from the conversation.

        Args:
            n: Number of turns to retrieve.

        Returns:
            List of the last N (role, content) tuples.
        """
        return self._turns[-n:] if n > 0 else []

    def get_turns_by_role(self, role: str) -> list[str]:
        """Get all messages from a specific role.

        Args:
            role: The role to filter by (user, assistant, etc.).

        Returns:
            List of message contents from the specified role.
        """
        role = role.lower()
        return [content for r, content in self._turns if r == role]

    @classmethod
    def from_resource(cls, resource: Resource) -> "ConversationResource":
        """Create a ConversationResource from an existing Resource.

        Args:
            resource: The Resource object to wrap.

        Returns:
            A new ConversationResource instance.
        """
        return cls(resource)

    @classmethod
    def create_new(
        cls,
        initial_content: str = "",
        metadata: Optional[dict[str, Any]] = None,
        resource_id: Optional[str] = None,
    ) -> "ConversationResource":
        """Create a new conversation resource.

        Args:
            initial_content: Optional initial conversation content.
            metadata: Optional metadata dictionary.
            resource_id: Optional specific resource ID (auto-generated if not provided).

        Returns:
            A new ConversationResource instance.
        """
        resource = Resource(
            resource_id=resource_id or generate_uuid(),
            resource_type="conversation",
            content=initial_content,
            metadata=metadata or {},
        )
        return cls(resource)


# =============================================================================
# ResourceLayer
# =============================================================================


class ResourceLayer:
    """Layer 1 of the 3-layer memory architecture: Raw resource storage.

    The ResourceLayer stores raw source data including conversations, documents,
    configurations, and feedback. Each resource is stored as a JSON file and
    provides full traceability for memory items extracted from it.

    Features:
        - CRUD operations for resources
        - Full-text search across resources
        - Type-based and temporal filtering
        - Thread-safe file operations with locking
        - Automatic directory management

    Attributes:
        storage_dir: Path to the resource storage directory.

    Example:
        >>> layer = ResourceLayer("outputs/memory/resources")
        >>> resource = layer.store(
        ...     resource_type="document",
        ...     content="Product manual content here...",
        ...     metadata={"title": "User Guide", "version": "1.0"}
        ... )
        >>> results = layer.search("product manual")
        >>> recent = layer.list_recent(limit=5)
    """

    def __init__(self, storage_dir: str = DEFAULT_STORAGE_DIR) -> None:
        """Initialize the ResourceLayer.

        Args:
            storage_dir: Directory path for storing resource files.
                Creates the directory if it does not exist.
        """
        self.storage_dir = Path(storage_dir)
        self._ensure_storage_dirs()

    def _ensure_storage_dirs(self) -> None:
        """Ensure storage directories exist for all resource types."""
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        for resource_type in VALID_RESOURCE_TYPES:
            (self.storage_dir / resource_type).mkdir(exist_ok=True)

    def _get_resource_path(self, resource_type: str, resource_id: str) -> Path:
        """Get the file path for a specific resource.

        Args:
            resource_type: Type of the resource.
            resource_id: Unique identifier for the resource.

        Returns:
            Path to the resource JSON file.
        """
        # Sanitize IDs to prevent path traversal
        safe_type = resource_type.replace("/", "_").replace("\\", "_").replace("..", "_")
        safe_id = resource_id.replace("/", "_").replace("\\", "_").replace("..", "_")
        return self.storage_dir / safe_type / f"{safe_id}.json"

    def _serialize_resource(self, resource: Resource) -> dict[str, Any]:
        """Serialize a Resource to a dictionary for JSON storage.

        Args:
            resource: The Resource to serialize.

        Returns:
            Dictionary representation of the resource.
        """
        return {
            "resource_id": resource.resource_id,
            "resource_type": resource.resource_type,
            "content": resource.content,
            "metadata": resource.metadata,
            "created_at": resource.created_at.isoformat(),
        }

    def _deserialize_resource(self, data: dict[str, Any]) -> Resource:
        """Deserialize a dictionary to a Resource.

        Args:
            data: Dictionary containing resource data.

        Returns:
            A Resource instance.
        """
        created_at = data.get("created_at")
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)
        elif created_at is None:
            created_at = utc_now()

        return Resource(
            resource_id=data["resource_id"],
            resource_type=data["resource_type"],
            content=data["content"],
            metadata=data.get("metadata", {}),
            created_at=created_at,
        )

    def store(
        self,
        resource_type: str,
        content: str,
        metadata: Optional[dict[str, Any]] = None,
        resource_id: Optional[str] = None,
    ) -> Resource:
        """Store a new resource.

        Args:
            resource_type: Type of resource (conversation, document, config, feedback).
            content: Raw content of the resource.
            metadata: Optional metadata dictionary.
            resource_id: Optional specific resource ID (auto-generated if not provided).

        Returns:
            The stored Resource object.

        Raises:
            MemoryWriteError: If the resource cannot be stored.
            ValueError: If the resource type is not valid.
        """
        # Validate resource type
        resource_type = resource_type.strip().lower()
        if resource_type not in VALID_RESOURCE_TYPES:
            raise ValueError(
                f"Invalid resource type '{resource_type}'. "
                f"Must be one of: {', '.join(sorted(VALID_RESOURCE_TYPES))}"
            )

        # Create the resource
        resource = Resource(
            resource_id=resource_id or generate_uuid(),
            resource_type=resource_type,
            content=content,
            metadata=metadata or {},
        )

        # Get the file path
        file_path = self._get_resource_path(resource.resource_type, resource.resource_id)

        # Write with file locking
        try:
            with portalocker.Lock(
                file_path, mode="w", timeout=10, flags=portalocker.LOCK_EX
            ) as f:
                json.dump(self._serialize_resource(resource), f, indent=2)
        except portalocker.LockException as e:
            raise SigilMemoryWriteError(
                f"Failed to acquire lock for resource {resource.resource_id}: {e}",
                layer="resources",
            )
        except Exception as e:
            raise SigilMemoryWriteError(
                f"Failed to store resource {resource.resource_id}: {e}",
                layer="resources",
            )

        return resource

    def store_resource(self, resource: Resource) -> Resource:
        """Store an existing Resource object.

        Args:
            resource: The Resource object to store.

        Returns:
            The stored Resource object.

        Raises:
            MemoryWriteError: If the resource cannot be stored.
        """
        return self.store(
            resource_type=resource.resource_type,
            content=resource.content,
            metadata=resource.metadata,
            resource_id=resource.resource_id,
        )

    def get(self, resource_id: str) -> Optional[Resource]:
        """Retrieve a resource by ID.

        This method searches across all resource types to find the resource.

        Args:
            resource_id: Unique identifier for the resource.

        Returns:
            The Resource if found, None otherwise.

        Raises:
            MemoryRetrievalError: If there's an error reading the resource.
        """
        # Search across all resource types
        for resource_type in VALID_RESOURCE_TYPES:
            file_path = self._get_resource_path(resource_type, resource_id)
            if file_path.exists():
                try:
                    with portalocker.Lock(
                        file_path, mode="r", timeout=10, flags=portalocker.LOCK_SH
                    ) as f:
                        data = json.load(f)
                    return self._deserialize_resource(data)
                except json.JSONDecodeError as e:
                    raise SigilMemoryRetrievalError(
                        f"Corrupted resource file for {resource_id}: {e}",
                        layer="resources",
                    )
                except portalocker.LockException as e:
                    raise SigilMemoryRetrievalError(
                        f"Failed to acquire lock for resource {resource_id}: {e}",
                        layer="resources",
                    )
                except Exception as e:
                    raise SigilMemoryRetrievalError(
                        f"Failed to read resource {resource_id}: {e}",
                        layer="resources",
                    )

        return None

    def get_by_type(self, resource_type: str, resource_id: str) -> Optional[Resource]:
        """Retrieve a resource by type and ID (more efficient than get()).

        Args:
            resource_type: Type of the resource.
            resource_id: Unique identifier for the resource.

        Returns:
            The Resource if found, None otherwise.
        """
        resource_type = resource_type.strip().lower()
        file_path = self._get_resource_path(resource_type, resource_id)

        if not file_path.exists():
            return None

        try:
            with portalocker.Lock(
                file_path, mode="r", timeout=10, flags=portalocker.LOCK_SH
            ) as f:
                data = json.load(f)
            return self._deserialize_resource(data)
        except Exception as e:
            raise SigilMemoryRetrievalError(
                f"Failed to read resource {resource_id}: {e}",
                layer="resources",
            )

    def list_by_type(
        self,
        resource_type: str,
        limit: Optional[int] = None,
        offset: int = 0,
    ) -> list[Resource]:
        """List resources of a specific type.

        Args:
            resource_type: Type of resources to list.
            limit: Maximum number of resources to return (None for all).
            offset: Number of resources to skip.

        Returns:
            List of Resource objects.

        Raises:
            ValueError: If the resource type is not valid.
        """
        resource_type = resource_type.strip().lower()
        if resource_type not in VALID_RESOURCE_TYPES:
            raise ValueError(
                f"Invalid resource type '{resource_type}'. "
                f"Must be one of: {', '.join(sorted(VALID_RESOURCE_TYPES))}"
            )

        type_dir = self.storage_dir / resource_type
        if not type_dir.exists():
            return []

        resources: list[Resource] = []
        json_files = sorted(type_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)

        # Apply offset
        json_files = json_files[offset:]

        # Apply limit
        if limit is not None:
            json_files = json_files[:limit]

        for file_path in json_files:
            try:
                with portalocker.Lock(
                    file_path, mode="r", timeout=5, flags=portalocker.LOCK_SH
                ) as f:
                    data = json.load(f)
                resources.append(self._deserialize_resource(data))
            except Exception:
                # Skip corrupted files
                continue

        return resources

    def list_recent(
        self,
        limit: int = 10,
        resource_type: Optional[str] = None,
    ) -> list[Resource]:
        """List the most recently created/modified resources.

        Args:
            limit: Maximum number of resources to return.
            resource_type: Optional filter by resource type.

        Returns:
            List of Resource objects, sorted by creation time (newest first).
        """
        resources: list[Resource] = []

        types_to_scan = (
            [resource_type.strip().lower()] if resource_type else VALID_RESOURCE_TYPES
        )

        for rtype in types_to_scan:
            if rtype not in VALID_RESOURCE_TYPES:
                continue
            type_dir = self.storage_dir / rtype
            if not type_dir.exists():
                continue

            for file_path in type_dir.glob("*.json"):
                try:
                    with portalocker.Lock(
                        file_path, mode="r", timeout=5, flags=portalocker.LOCK_SH
                    ) as f:
                        data = json.load(f)
                    resources.append(self._deserialize_resource(data))
                except Exception:
                    continue

        # Sort by created_at (newest first) and limit
        resources.sort(key=lambda r: r.created_at, reverse=True)
        return resources[:limit]

    def search(
        self,
        query: str,
        resource_type: Optional[str] = None,
        limit: int = 10,
        case_sensitive: bool = False,
    ) -> list[Resource]:
        """Full-text search across resources.

        Searches both content and metadata for the query string.

        Args:
            query: Search query string.
            resource_type: Optional filter by resource type.
            limit: Maximum number of results to return.
            case_sensitive: Whether the search is case-sensitive.

        Returns:
            List of matching Resource objects.
        """
        if not query.strip():
            return []

        search_query = query if case_sensitive else query.lower()
        results: list[tuple[Resource, int]] = []  # (resource, match_count)

        types_to_scan = (
            [resource_type.strip().lower()] if resource_type else VALID_RESOURCE_TYPES
        )

        for rtype in types_to_scan:
            if rtype not in VALID_RESOURCE_TYPES:
                continue
            type_dir = self.storage_dir / rtype
            if not type_dir.exists():
                continue

            for file_path in type_dir.glob("*.json"):
                try:
                    with portalocker.Lock(
                        file_path, mode="r", timeout=5, flags=portalocker.LOCK_SH
                    ) as f:
                        data = json.load(f)

                    resource = self._deserialize_resource(data)

                    # Search in content
                    content = resource.content if case_sensitive else resource.content.lower()
                    match_count = content.count(search_query)

                    # Search in metadata (stringify and search)
                    metadata_str = json.dumps(resource.metadata)
                    if not case_sensitive:
                        metadata_str = metadata_str.lower()
                    match_count += metadata_str.count(search_query)

                    if match_count > 0:
                        results.append((resource, match_count))

                except Exception:
                    continue

        # Sort by match count (descending) and limit
        results.sort(key=lambda x: x[1], reverse=True)
        return [r for r, _ in results[:limit]]

    def delete(self, resource_id: str) -> bool:
        """Delete a resource by ID.

        Args:
            resource_id: Unique identifier for the resource.

        Returns:
            True if the resource was deleted, False if not found.

        Raises:
            MemoryWriteError: If deletion fails.
        """
        # Search across all resource types
        for resource_type in VALID_RESOURCE_TYPES:
            file_path = self._get_resource_path(resource_type, resource_id)
            if file_path.exists():
                try:
                    file_path.unlink()
                    return True
                except Exception as e:
                    raise SigilMemoryWriteError(
                        f"Failed to delete resource {resource_id}: {e}",
                        layer="resources",
                    )

        return False

    def exists(self, resource_id: str) -> bool:
        """Check if a resource exists.

        Args:
            resource_id: Unique identifier for the resource.

        Returns:
            True if the resource exists, False otherwise.
        """
        for resource_type in VALID_RESOURCE_TYPES:
            file_path = self._get_resource_path(resource_type, resource_id)
            if file_path.exists():
                return True
        return False

    def count(self, resource_type: Optional[str] = None) -> int:
        """Count the number of resources.

        Args:
            resource_type: Optional filter by resource type.

        Returns:
            Number of resources.
        """
        count = 0
        types_to_scan = (
            [resource_type.strip().lower()] if resource_type else VALID_RESOURCE_TYPES
        )

        for rtype in types_to_scan:
            if rtype not in VALID_RESOURCE_TYPES:
                continue
            type_dir = self.storage_dir / rtype
            if type_dir.exists():
                count += len(list(type_dir.glob("*.json")))

        return count

    def iter_all(
        self,
        resource_type: Optional[str] = None,
    ) -> Iterator[Resource]:
        """Iterate over all resources (memory-efficient).

        Args:
            resource_type: Optional filter by resource type.

        Yields:
            Resource objects one at a time.
        """
        types_to_scan = (
            [resource_type.strip().lower()] if resource_type else VALID_RESOURCE_TYPES
        )

        for rtype in types_to_scan:
            if rtype not in VALID_RESOURCE_TYPES:
                continue
            type_dir = self.storage_dir / rtype
            if not type_dir.exists():
                continue

            for file_path in type_dir.glob("*.json"):
                try:
                    with portalocker.Lock(
                        file_path, mode="r", timeout=5, flags=portalocker.LOCK_SH
                    ) as f:
                        data = json.load(f)
                    yield self._deserialize_resource(data)
                except Exception:
                    continue

    def get_conversation(self, resource_id: str) -> Optional[ConversationResource]:
        """Get a resource as a ConversationResource.

        Args:
            resource_id: Unique identifier for the resource.

        Returns:
            A ConversationResource if found and is a conversation type, None otherwise.

        Raises:
            ValueError: If the resource exists but is not a conversation type.
        """
        resource = self.get(resource_id)
        if resource is None:
            return None
        if resource.resource_type != "conversation":
            raise ValueError(
                f"Resource {resource_id} is not a conversation "
                f"(type: {resource.resource_type})"
            )
        return ConversationResource.from_resource(resource)

    def store_conversation(
        self,
        content: str,
        metadata: Optional[dict[str, Any]] = None,
        resource_id: Optional[str] = None,
    ) -> ConversationResource:
        """Store a new conversation resource.

        Args:
            content: Conversation content (formatted as "Role: Message" lines).
            metadata: Optional metadata dictionary.
            resource_id: Optional specific resource ID.

        Returns:
            A ConversationResource wrapping the stored resource.
        """
        resource = self.store(
            resource_type="conversation",
            content=content,
            metadata=metadata,
            resource_id=resource_id,
        )
        return ConversationResource.from_resource(resource)


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "ResourceLayer",
    "ConversationResource",
    "VALID_RESOURCE_TYPES",
    "DEFAULT_STORAGE_DIR",
]
