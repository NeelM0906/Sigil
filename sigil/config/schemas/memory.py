"""Memory system schemas for Sigil v2 3-layer architecture.

This module defines Pydantic models for the 3-layer memory system:
- Layer 1 (Resources): Raw source data - conversations, documents, configs
- Layer 2 (MemoryItems): Discrete facts with embeddings for RAG search
- Layer 3 (MemoryCategories): Aggregated knowledge in markdown format

Classes:
    Resource: Raw source data container (Layer 1)
    MemoryItem: Extracted fact with embedding (Layer 2)
    MemoryCategory: Aggregated knowledge category (Layer 3)
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator


def generate_uuid() -> str:
    """Generate a UUID4 hex string for use as an identifier.

    Returns:
        A 32-character hex string UUID.
    """
    return uuid.uuid4().hex


def utc_now() -> datetime:
    """Get the current UTC datetime.

    Returns:
        Current datetime with UTC timezone.
    """
    return datetime.now(timezone.utc)


class Resource(BaseModel):
    """Raw source data container (Layer 1).

    Resources represent the raw, unprocessed data that enters the memory
    system. This includes conversation transcripts, documents, configuration
    files, and any other source material that memory items are extracted from.

    The resource layer provides:
    - Full traceability back to original sources
    - Immutable storage of raw data
    - Type-specific metadata for different resource types

    Attributes:
        resource_id: Unique UUID identifier for this resource
        resource_type: Type of resource (conversation, document, config, etc.)
        content: Raw content of the resource
        metadata: Type-specific metadata (e.g., speaker for conversations)
        created_at: UTC timestamp when the resource was created

    Example:
        ```python
        resource = Resource(
            resource_type="conversation",
            content="User: What pricing plans do you offer?\\nAgent: We have...",
            metadata={"session_id": "abc123", "user_id": "user_456"}
        )
        ```
    """

    resource_id: str = Field(
        default_factory=generate_uuid,
        description="Unique UUID identifier for this resource",
    )
    resource_type: str = Field(
        ...,
        description="Type of resource: conversation, document, config, etc.",
        examples=["conversation", "document", "config"],
    )
    content: str = Field(
        ...,
        description="Raw content of the resource",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Type-specific metadata for this resource",
    )
    created_at: datetime = Field(
        default_factory=utc_now,
        description="UTC timestamp when the resource was created",
    )

    @field_validator("resource_type")
    @classmethod
    def validate_resource_type(cls, v: str) -> str:
        """Validate resource type is non-empty and lowercase.

        Args:
            v: Resource type string to validate.

        Returns:
            Validated and normalized resource type.

        Raises:
            ValueError: If resource type is empty.
        """
        v = v.strip().lower()
        if not v:
            raise ValueError("resource_type cannot be empty")
        return v


class MemoryItem(BaseModel):
    """Extracted fact with embedding (Layer 2).

    Memory items are discrete facts or pieces of information extracted
    from resources. Each item has an optional embedding for semantic
    search via RAG retrieval.

    The memory item layer provides:
    - Discrete, searchable facts
    - Vector embeddings for semantic retrieval
    - Traceability links back to source resources
    - Confidence scores for extraction quality
    - Category assignment for organization

    Attributes:
        item_id: Unique UUID identifier for this memory item
        content: Extracted fact or information
        embedding: Optional vector embedding for RAG search
        source_resource_id: ID of the resource this was extracted from
        category: Optional category this item belongs to
        confidence: Extraction confidence score (0.0 to 1.0)
        created_at: UTC timestamp when the item was created

    Example:
        ```python
        item = MemoryItem(
            content="Customer prefers monthly billing over annual",
            source_resource_id="abc123def456",
            category="billing_preferences",
            confidence=0.95
        )
        ```
    """

    item_id: str = Field(
        default_factory=generate_uuid,
        description="Unique UUID identifier for this memory item",
    )
    content: str = Field(
        ...,
        description="Extracted fact or piece of information",
    )
    embedding: Optional[list[float]] = Field(
        default=None,
        description="Vector embedding for RAG semantic search",
    )
    source_resource_id: str = Field(
        ...,
        description="UUID of the resource this item was extracted from",
    )
    category: Optional[str] = Field(
        default=None,
        description="Category this memory item belongs to",
    )
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Extraction confidence score (0.0 to 1.0)",
    )
    created_at: datetime = Field(
        default_factory=utc_now,
        description="UTC timestamp when the item was created",
    )

    @field_validator("confidence")
    @classmethod
    def validate_confidence(cls, v: float) -> float:
        """Validate confidence is within valid range.

        Args:
            v: Confidence value to validate.

        Returns:
            Validated confidence value.

        Raises:
            ValueError: If confidence is outside [0.0, 1.0] range.
        """
        if not 0.0 <= v <= 1.0:
            raise ValueError("confidence must be between 0.0 and 1.0")
        return v

    @field_validator("category", mode="before")
    @classmethod
    def normalize_category(cls, v: Optional[str]) -> Optional[str]:
        """Normalize category to lowercase snake_case.

        Args:
            v: Category string to normalize.

        Returns:
            Normalized category or None.
        """
        if v is None:
            return None
        return v.strip().lower().replace(" ", "_").replace("-", "_")

    @field_validator("embedding")
    @classmethod
    def validate_embedding(cls, v: Optional[list[float]]) -> Optional[list[float]]:
        """Validate embedding is a non-empty list of floats if provided.

        Args:
            v: Embedding vector to validate.

        Returns:
            Validated embedding or None.

        Raises:
            ValueError: If embedding is an empty list.
        """
        if v is not None and len(v) == 0:
            raise ValueError("embedding cannot be an empty list")
        return v


class MemoryCategory(BaseModel):
    """Aggregated knowledge category (Layer 3).

    Memory categories represent aggregated knowledge derived from multiple
    memory items. The content is stored in markdown format for both human
    readability and LLM consumption.

    The category layer provides:
    - Human and LLM-readable knowledge summaries
    - Aggregated insights from multiple memory items
    - Structured organization of learned information
    - Source traceability via item_ids

    Attributes:
        category_id: Unique UUID identifier for this category
        name: Category name (e.g., "lead_preferences")
        description: What this category contains
        markdown_content: Aggregated knowledge in markdown format
        item_ids: List of source memory item IDs
        updated_at: UTC timestamp of last update

    Example:
        ```python
        category = MemoryCategory(
            name="pricing_objections",
            description="Common pricing objections and effective responses",
            markdown_content="## Common Pricing Objections\\n\\n1. **Too expensive**...",
            item_ids=["item1", "item2", "item3"]
        )
        ```
    """

    category_id: str = Field(
        default_factory=generate_uuid,
        description="Unique UUID identifier for this category",
    )
    name: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Category name (e.g., 'lead_preferences')",
        examples=["lead_preferences", "objection_patterns", "pricing_responses"],
    )
    description: str = Field(
        ...,
        min_length=1,
        description="What this category contains",
    )
    markdown_content: str = Field(
        default="",
        description="Aggregated knowledge in markdown format",
    )
    item_ids: list[str] = Field(
        default_factory=list,
        description="List of source memory item UUIDs",
    )
    updated_at: datetime = Field(
        default_factory=utc_now,
        description="UTC timestamp of last update",
    )

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Validate and normalize category name.

        Args:
            v: Category name to validate.

        Returns:
            Normalized category name.

        Raises:
            ValueError: If name is empty after normalization.
        """
        v = v.strip().lower().replace(" ", "_").replace("-", "_")
        if not v:
            raise ValueError("name cannot be empty")
        return v

    def add_item(self, item_id: str) -> None:
        """Add a memory item ID to this category.

        Args:
            item_id: The memory item UUID to add.
        """
        if item_id not in self.item_ids:
            self.item_ids.append(item_id)
            self.updated_at = utc_now()

    def remove_item(self, item_id: str) -> bool:
        """Remove a memory item ID from this category.

        Args:
            item_id: The memory item UUID to remove.

        Returns:
            True if the item was removed, False if not found.
        """
        if item_id in self.item_ids:
            self.item_ids.remove(item_id)
            self.updated_at = utc_now()
            return True
        return False


__all__ = [
    "Resource",
    "MemoryItem",
    "MemoryCategory",
    "generate_uuid",
    "utc_now",
]
