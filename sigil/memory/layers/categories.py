"""Layer 3: Memory Categories for the Sigil v2 3-layer memory architecture.

This module implements the Category Layer, which aggregates memory items
into human-readable markdown categories. Categories provide a high-level
view of learned knowledge.

Classes:
    CategoryLayer: Main class for managing memory categories.

Storage Format:
    Categories are stored as markdown files:
    {storage_dir}/{category_name}.md

    Each markdown file contains:
    - YAML frontmatter with metadata
    - Aggregated knowledge content

Example:
    >>> from sigil.memory.layers.categories import CategoryLayer
    >>> layer = CategoryLayer(storage_dir="outputs/memory/categories")
    >>> category = layer.create(
    ...     name="lead_preferences",
    ...     description="Customer preferences and requirements"
    ... )
    >>> layer.add_items("lead_preferences", ["item-1", "item-2"])
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Iterator
import threading

import portalocker

from sigil.config.schemas.memory import MemoryCategory, generate_uuid, utc_now
from sigil.core.exceptions import MemoryError, MemoryWriteError, MemoryRetrievalError


# =============================================================================
# Constants
# =============================================================================

DEFAULT_STORAGE_DIR = "outputs/memory/categories"
"""Default directory for category storage."""


# =============================================================================
# Category Layer
# =============================================================================


class CategoryLayer:
    """Layer 3 of the 3-layer memory architecture: Aggregated knowledge categories.

    The CategoryLayer stores aggregated knowledge derived from memory items.
    Each category is stored as a markdown file with YAML frontmatter for
    metadata, making it readable by both humans and LLMs.

    Features:
        - CRUD operations for categories
        - Markdown file storage with frontmatter
        - Item tracking and traceability
        - Incremental content updates
        - Thread-safe operations

    Attributes:
        storage_dir: Path to the category storage directory.

    Example:
        >>> layer = CategoryLayer("outputs/memory/categories")
        >>> category = layer.create(
        ...     name="objection_patterns",
        ...     description="Common objections and effective responses"
        ... )
        >>> layer.update_content("objection_patterns", "## Pricing Objections\\n...")
    """

    def __init__(self, storage_dir: str = DEFAULT_STORAGE_DIR) -> None:
        """Initialize the CategoryLayer.

        Args:
            storage_dir: Directory path for storing category files.
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        self._categories: dict[str, MemoryCategory] = {}
        self._lock = threading.RLock()

        self._load_categories()

    def _get_category_path(self, name: str) -> Path:
        """Get the file path for a category.

        Args:
            name: Category name.

        Returns:
            Path to the category markdown file.
        """
        # Normalize and sanitize name
        safe_name = name.strip().lower().replace(" ", "_").replace("-", "_")
        safe_name = re.sub(r"[^a-z0-9_]", "", safe_name)
        return self.storage_dir / f"{safe_name}.md"

    def _load_categories(self) -> None:
        """Load all categories from storage."""
        for file_path in self.storage_dir.glob("*.md"):
            try:
                category = self._load_category_file(file_path)
                if category:
                    self._categories[category.name] = category
            except Exception:
                # Skip corrupted files
                continue

    def _load_category_file(self, file_path: Path) -> Optional[MemoryCategory]:
        """Load a category from a markdown file.

        Args:
            file_path: Path to the markdown file.

        Returns:
            MemoryCategory if successful, None otherwise.
        """
        try:
            with portalocker.Lock(
                file_path, mode="r", timeout=10, flags=portalocker.LOCK_SH
            ) as f:
                content = f.read()

            return self._parse_markdown(content)

        except Exception:
            return None

    def _parse_markdown(self, content: str) -> Optional[MemoryCategory]:
        """Parse markdown content into a MemoryCategory.

        Args:
            content: Markdown content with frontmatter.

        Returns:
            MemoryCategory if parsing succeeds, None otherwise.
        """
        # Parse YAML frontmatter (between --- markers)
        frontmatter_match = re.match(r"^---\n(.*?)\n---\n(.*)$", content, re.DOTALL)

        if not frontmatter_match:
            return None

        frontmatter_str = frontmatter_match.group(1)
        markdown_content = frontmatter_match.group(2).strip()

        # Simple YAML parsing (key: value format)
        metadata: dict[str, Any] = {}
        for line in frontmatter_str.split("\n"):
            if ":" in line:
                key, _, value = line.partition(":")
                key = key.strip()
                value = value.strip()

                # Handle special types
                if key == "item_ids":
                    # Parse as JSON array
                    try:
                        metadata[key] = json.loads(value) if value else []
                    except json.JSONDecodeError:
                        metadata[key] = []
                elif key == "updated_at":
                    try:
                        metadata[key] = datetime.fromisoformat(value)
                    except ValueError:
                        metadata[key] = utc_now()
                else:
                    # Remove quotes if present
                    if value.startswith('"') and value.endswith('"'):
                        value = value[1:-1]
                    elif value.startswith("'") and value.endswith("'"):
                        value = value[1:-1]
                    metadata[key] = value

        # Validate required fields
        if "category_id" not in metadata or "name" not in metadata:
            return None

        return MemoryCategory(
            category_id=metadata["category_id"],
            name=metadata["name"],
            description=metadata.get("description", ""),
            markdown_content=markdown_content,
            item_ids=metadata.get("item_ids", []),
            updated_at=metadata.get("updated_at", utc_now()),
        )

    def _serialize_to_markdown(self, category: MemoryCategory) -> str:
        """Serialize a MemoryCategory to markdown with frontmatter.

        Args:
            category: The category to serialize.

        Returns:
            Markdown string with YAML frontmatter.
        """
        # Build frontmatter
        frontmatter_lines = [
            "---",
            f'category_id: "{category.category_id}"',
            f'name: "{category.name}"',
            f'description: "{category.description}"',
            f"item_ids: {json.dumps(category.item_ids)}",
            f"updated_at: {category.updated_at.isoformat()}",
            "---",
        ]

        return "\n".join(frontmatter_lines) + "\n\n" + category.markdown_content

    def _save_category(self, category: MemoryCategory) -> None:
        """Save a category to its markdown file.

        Args:
            category: The category to save.
        """
        file_path = self._get_category_path(category.name)
        content = self._serialize_to_markdown(category)

        try:
            with portalocker.Lock(
                file_path, mode="w", timeout=10, flags=portalocker.LOCK_EX
            ) as f:
                f.write(content)
        except Exception as e:
            raise MemoryWriteError(
                f"Failed to save category {category.name}: {e}",
                layer="categories",
            )

    def create(
        self,
        name: str,
        description: str,
        markdown_content: str = "",
        category_id: Optional[str] = None,
    ) -> MemoryCategory:
        """Create a new category.

        Args:
            name: Category name (will be normalized).
            description: What this category contains.
            markdown_content: Initial markdown content.
            category_id: Optional specific category ID.

        Returns:
            The created MemoryCategory.

        Raises:
            ValueError: If a category with this name already exists.
        """
        with self._lock:
            # Normalize name
            normalized_name = name.strip().lower().replace(" ", "_").replace("-", "_")

            if normalized_name in self._categories:
                raise ValueError(f"Category '{normalized_name}' already exists")

            category = MemoryCategory(
                category_id=category_id or generate_uuid(),
                name=normalized_name,
                description=description,
                markdown_content=markdown_content,
            )

            self._categories[normalized_name] = category
            self._save_category(category)

            return category

    def get(self, category_id: str) -> Optional[MemoryCategory]:
        """Get a category by ID.

        Args:
            category_id: The category's unique ID.

        Returns:
            The MemoryCategory if found, None otherwise.
        """
        for category in self._categories.values():
            if category.category_id == category_id:
                return category
        return None

    def get_by_name(self, name: str) -> Optional[MemoryCategory]:
        """Get a category by name.

        Args:
            name: The category name.

        Returns:
            The MemoryCategory if found, None otherwise.
        """
        normalized_name = name.strip().lower().replace(" ", "_").replace("-", "_")
        return self._categories.get(normalized_name)

    def list_all(self) -> list[MemoryCategory]:
        """List all categories.

        Returns:
            List of all MemoryCategory objects.
        """
        return list(self._categories.values())

    def update_content(
        self,
        name: str,
        markdown_content: str,
        append: bool = False,
    ) -> Optional[MemoryCategory]:
        """Update the markdown content of a category.

        Args:
            name: The category name.
            markdown_content: New markdown content.
            append: If True, append to existing content instead of replacing.

        Returns:
            The updated MemoryCategory, or None if not found.
        """
        with self._lock:
            category = self.get_by_name(name)
            if category is None:
                return None

            if append:
                new_content = category.markdown_content + "\n\n" + markdown_content
            else:
                new_content = markdown_content

            # Create updated category
            updated = MemoryCategory(
                category_id=category.category_id,
                name=category.name,
                description=category.description,
                markdown_content=new_content.strip(),
                item_ids=category.item_ids,
                updated_at=utc_now(),
            )

            self._categories[category.name] = updated
            self._save_category(updated)

            return updated

    def add_items(
        self,
        name: str,
        item_ids: list[str],
    ) -> Optional[MemoryCategory]:
        """Add memory item IDs to a category.

        Args:
            name: The category name.
            item_ids: List of memory item IDs to add.

        Returns:
            The updated MemoryCategory, or None if not found.
        """
        with self._lock:
            category = self.get_by_name(name)
            if category is None:
                return None

            # Add new items (avoid duplicates)
            existing_ids = set(category.item_ids)
            new_ids = [i for i in item_ids if i not in existing_ids]

            if not new_ids:
                return category

            updated_ids = category.item_ids + new_ids

            # Create updated category
            updated = MemoryCategory(
                category_id=category.category_id,
                name=category.name,
                description=category.description,
                markdown_content=category.markdown_content,
                item_ids=updated_ids,
                updated_at=utc_now(),
            )

            self._categories[category.name] = updated
            self._save_category(updated)

            return updated

    def remove_items(
        self,
        name: str,
        item_ids: list[str],
    ) -> Optional[MemoryCategory]:
        """Remove memory item IDs from a category.

        Args:
            name: The category name.
            item_ids: List of memory item IDs to remove.

        Returns:
            The updated MemoryCategory, or None if not found.
        """
        with self._lock:
            category = self.get_by_name(name)
            if category is None:
                return None

            ids_to_remove = set(item_ids)
            updated_ids = [i for i in category.item_ids if i not in ids_to_remove]

            if len(updated_ids) == len(category.item_ids):
                return category  # No changes

            # Create updated category
            updated = MemoryCategory(
                category_id=category.category_id,
                name=category.name,
                description=category.description,
                markdown_content=category.markdown_content,
                item_ids=updated_ids,
                updated_at=utc_now(),
            )

            self._categories[category.name] = updated
            self._save_category(updated)

            return updated

    def delete(self, name: str) -> bool:
        """Delete a category.

        Args:
            name: The category name.

        Returns:
            True if the category was deleted, False if not found.
        """
        with self._lock:
            normalized_name = name.strip().lower().replace(" ", "_").replace("-", "_")

            if normalized_name not in self._categories:
                return False

            del self._categories[normalized_name]

            # Delete the file
            file_path = self._get_category_path(normalized_name)
            if file_path.exists():
                file_path.unlink()

            return True

    def exists(self, name: str) -> bool:
        """Check if a category exists.

        Args:
            name: The category name.

        Returns:
            True if the category exists.
        """
        normalized_name = name.strip().lower().replace(" ", "_").replace("-", "_")
        return normalized_name in self._categories

    def count(self) -> int:
        """Get the number of categories.

        Returns:
            Number of categories.
        """
        return len(self._categories)

    def get_content(self, name: str) -> Optional[str]:
        """Get just the markdown content of a category.

        Args:
            name: The category name.

        Returns:
            The markdown content if found, None otherwise.
        """
        category = self.get_by_name(name)
        return category.markdown_content if category else None

    def get_item_count(self, name: str) -> int:
        """Get the number of items in a category.

        Args:
            name: The category name.

        Returns:
            Number of items, or 0 if category not found.
        """
        category = self.get_by_name(name)
        return len(category.item_ids) if category else 0

    def iter_all(self) -> Iterator[MemoryCategory]:
        """Iterate over all categories.

        Yields:
            MemoryCategory objects.
        """
        for category in self._categories.values():
            yield category

    def get_categories_with_item(self, item_id: str) -> list[MemoryCategory]:
        """Get all categories that contain a specific item.

        Args:
            item_id: The memory item ID.

        Returns:
            List of categories containing the item.
        """
        return [
            category
            for category in self._categories.values()
            if item_id in category.item_ids
        ]

    def search_by_content(
        self,
        query: str,
        limit: int = 10,
        case_sensitive: bool = False,
    ) -> list[MemoryCategory]:
        """Search categories by content.

        Args:
            query: Search query string.
            limit: Maximum number of results.
            case_sensitive: Whether to use case-sensitive matching.

        Returns:
            List of matching categories.
        """
        if not query.strip():
            return []

        search_query = query if case_sensitive else query.lower()
        results: list[tuple[MemoryCategory, int]] = []

        for category in self._categories.values():
            content = category.markdown_content
            if not case_sensitive:
                content = content.lower()

            match_count = content.count(search_query)
            if match_count > 0:
                results.append((category, match_count))

        results.sort(key=lambda x: x[1], reverse=True)
        return [cat for cat, _ in results[:limit]]

    def create_or_update(
        self,
        name: str,
        description: str,
        markdown_content: str = "",
    ) -> MemoryCategory:
        """Create a category or update it if it exists.

        Args:
            name: Category name.
            description: Category description.
            markdown_content: Markdown content.

        Returns:
            The created or updated MemoryCategory.
        """
        with self._lock:
            normalized_name = name.strip().lower().replace(" ", "_").replace("-", "_")

            if normalized_name in self._categories:
                # Update existing
                return self.update_content(normalized_name, markdown_content) or self._categories[normalized_name]
            else:
                # Create new
                return self.create(name, description, markdown_content)


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "CategoryLayer",
    "DEFAULT_STORAGE_DIR",
]
