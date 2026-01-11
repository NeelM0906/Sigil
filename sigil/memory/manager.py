"""Memory Manager for the Sigil v2 3-layer memory architecture.

This module implements the MemoryManager class, which orchestrates all three
memory layers and integrates with the event store for audit trails.

Classes:
    MemoryManager: Main orchestrator for the memory system.

Example:
    >>> from sigil.memory.manager import MemoryManager
    >>> manager = MemoryManager()
    >>> resource = await manager.store_resource(
    ...     resource_type="conversation",
    ...     content="User: Hello!\\nAgent: Hi there!",
    ...     session_id="sess-123"
    ... )
    >>> items = await manager.extract_and_store(resource.resource_id)
    >>> results = await manager.retrieve("greeting patterns", k=5)
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from sigil.config import get_settings
from sigil.config.schemas.memory import Resource, MemoryItem, MemoryCategory
from sigil.core.exceptions import MemoryError, MemoryWriteError, MemoryRetrievalError
from sigil.state.events import EventType, create_memory_extracted_event
from sigil.state.store import EventStore

from sigil.memory.layers.resources import ResourceLayer, ConversationResource
from sigil.memory.layers.items import ItemLayer, EmbeddingService
from sigil.memory.layers.categories import CategoryLayer
from sigil.memory.extraction import MemoryExtractor, ExtractedFact
from sigil.memory.consolidation import MemoryConsolidator, ConsolidationTrigger, ConsolidationResult
from sigil.memory.retrieval import (
    RetrievalMode,
    RAGRetriever,
    LLMRetriever,
    HybridRetriever,
    RetrievalContext,
)
from sigil.core.base import RetrievalResult


# =============================================================================
# Constants
# =============================================================================

DEFAULT_BASE_DIR = "outputs/memory"
"""Default base directory for all memory storage."""


# =============================================================================
# Memory Stats
# =============================================================================


@dataclass
class MemoryStats:
    """Statistics about the memory system.

    Attributes:
        resource_count: Total number of resources.
        item_count: Total number of memory items.
        category_count: Total number of categories.
        resources_by_type: Resource counts by type.
        items_by_category: Item counts by category.
    """

    resource_count: int
    item_count: int
    category_count: int
    resources_by_type: dict[str, int]
    items_by_category: dict[str, int]


# =============================================================================
# Category Consolidated Event
# =============================================================================


def create_category_consolidated_event(
    session_id: str,
    category_id: str,
    category_name: str,
    items_consolidated: int,
    is_incremental: bool,
    trigger: str,
    correlation_id: Optional[str] = None,
) -> Any:
    """Create a CategoryConsolidatedEvent.

    This is a custom event type for tracking category consolidation.

    Args:
        session_id: Session identifier.
        category_id: Unique category identifier.
        category_name: Name of the category.
        items_consolidated: Number of items consolidated.
        is_incremental: Whether this was an incremental update.
        trigger: What triggered the consolidation.
        correlation_id: Optional correlation ID.

    Returns:
        Event object.
    """
    from sigil.state.events import Event, _generate_event_id, _get_utc_now

    # Use MEMORY_EXTRACTED type with category-specific payload
    # In a full implementation, we'd add a new EventType for this
    payload = {
        "memory_id": category_id,
        "memory_type": "category_consolidated",
        "content": f"Category '{category_name}' consolidated",
        "source": trigger,
        "confidence": 1.0,
        "metadata": {
            "category_name": category_name,
            "items_consolidated": items_consolidated,
            "is_incremental": is_incremental,
        },
    }

    return Event(
        event_id=_generate_event_id(),
        event_type=EventType.MEMORY_EXTRACTED,
        timestamp=_get_utc_now(),
        session_id=session_id,
        correlation_id=correlation_id,
        payload=payload,
    )


# =============================================================================
# Memory Manager
# =============================================================================


class MemoryManager:
    """Orchestrator for the 3-layer memory system.

    The MemoryManager coordinates all three memory layers:
    - Layer 1 (Resources): Raw data storage
    - Layer 2 (Items): Extracted facts with embeddings
    - Layer 3 (Categories): Aggregated knowledge

    It also integrates with the event store for audit trails and
    provides a unified interface for memory operations.

    Features:
        - Store and retrieve resources
        - Extract and store memory items
        - Retrieve using RAG, LLM, or hybrid modes
        - Consolidate items into categories
        - Event emission for audit trails
        - Health checking and statistics

    Attributes:
        resource_layer: Layer 1 - Resource storage.
        item_layer: Layer 2 - Memory item storage.
        category_layer: Layer 3 - Category storage.
        event_store: Event store for audit trails.
        extractor: Memory extraction service.
        consolidator: Memory consolidation service.
        retriever: Hybrid retriever for queries.

    Example:
        >>> manager = MemoryManager()
        >>> resource = await manager.store_resource(
        ...     resource_type="conversation",
        ...     content="...",
        ...     session_id="sess-123"
        ... )
        >>> items = await manager.extract_and_store(resource.resource_id)
        >>> results = await manager.retrieve("query", k=5)
    """

    def __init__(
        self,
        base_dir: str = DEFAULT_BASE_DIR,
        event_store: Optional[EventStore] = None,
        embedding_service: Optional[EmbeddingService] = None,
    ) -> None:
        """Initialize the MemoryManager.

        Args:
            base_dir: Base directory for memory storage.
            event_store: Optional custom event store.
            embedding_service: Optional custom embedding service.
        """
        base_path = Path(base_dir)
        base_path.mkdir(parents=True, exist_ok=True)

        # Initialize layers
        self.resource_layer = ResourceLayer(str(base_path / "resources"))
        self.item_layer = ItemLayer(
            str(base_path / "items"),
            embedding_service=embedding_service,
        )
        self.category_layer = CategoryLayer(str(base_path / "categories"))

        # Initialize services
        self.extractor = MemoryExtractor()
        self.consolidator = MemoryConsolidator()

        # Initialize retriever
        self.retriever = HybridRetriever(
            item_layer=self.item_layer,
            category_layer=self.category_layer,
        )

        # Event store
        self._event_store = event_store or EventStore()

        # Track pending consolidation counts per category
        self._pending_consolidation: dict[str, int] = {}

    @property
    def is_healthy(self) -> bool:
        """Check if the memory system is healthy.

        Returns:
            True if all components are functioning.
        """
        try:
            # Basic health checks
            _ = self.resource_layer.count()
            _ = self.item_layer.count()
            _ = self.category_layer.count()
            return True
        except Exception:
            return False

    # =========================================================================
    # Resource Operations
    # =========================================================================

    async def store_resource(
        self,
        resource_type: str,
        content: str,
        metadata: Optional[dict[str, Any]] = None,
        session_id: Optional[str] = None,
        resource_id: Optional[str] = None,
    ) -> Resource:
        """Store a new resource.

        Args:
            resource_type: Type of resource (conversation, document, config, feedback).
            content: Raw content of the resource.
            metadata: Optional metadata dictionary.
            session_id: Optional session ID for event tracking.
            resource_id: Optional specific resource ID.

        Returns:
            The stored Resource.
        """
        # Add session_id to metadata if provided
        if session_id and metadata:
            metadata = {**metadata, "session_id": session_id}
        elif session_id:
            metadata = {"session_id": session_id}

        resource = self.resource_layer.store(
            resource_type=resource_type,
            content=content,
            metadata=metadata or {},
            resource_id=resource_id,
        )

        return resource

    async def get_resource(self, resource_id: str) -> Optional[Resource]:
        """Get a resource by ID.

        Args:
            resource_id: Unique resource identifier.

        Returns:
            The Resource if found, None otherwise.
        """
        return self.resource_layer.get(resource_id)

    async def store_conversation(
        self,
        content: str,
        session_id: str,
        metadata: Optional[dict[str, Any]] = None,
    ) -> ConversationResource:
        """Store a conversation resource.

        Args:
            content: Conversation content.
            session_id: Session identifier.
            metadata: Optional metadata.

        Returns:
            ConversationResource wrapper.
        """
        meta = metadata or {}
        meta["session_id"] = session_id

        return self.resource_layer.store_conversation(
            content=content,
            metadata=meta,
        )

    # =========================================================================
    # Extraction Operations
    # =========================================================================

    async def extract_and_store(
        self,
        resource_id: str,
        session_id: Optional[str] = None,
    ) -> list[MemoryItem]:
        """Extract facts from a resource and store them as memory items.

        Args:
            resource_id: ID of the resource to extract from.
            session_id: Optional session ID for event tracking.

        Returns:
            List of created MemoryItems.
        """
        # Get the resource
        resource = self.resource_layer.get(resource_id)
        if resource is None:
            raise MemoryRetrievalError(
                f"Resource not found: {resource_id}",
                layer="resources",
            )

        # Extract facts
        facts = await self.extractor.extract_from_resource(resource)

        if not facts:
            return []

        # Store items
        items: list[MemoryItem] = []
        for fact in facts:
            item = self.item_layer.store(
                content=fact.content,
                source_resource_id=resource_id,
                category=fact.category,
                confidence=fact.confidence,
            )
            items.append(item)

            # Track for consolidation
            if fact.category:
                self._pending_consolidation[fact.category] = (
                    self._pending_consolidation.get(fact.category, 0) + 1
                )

            # Emit event
            if session_id:
                event = create_memory_extracted_event(
                    session_id=session_id,
                    memory_id=item.item_id,
                    memory_type="item",
                    content=item.content,
                    source=resource_id,
                    confidence=item.confidence,
                    metadata={"category": item.category},
                )
                self._event_store.append(event)

        # Check for consolidation triggers
        await self._check_consolidation_triggers(session_id)

        return items

    async def remember(
        self,
        content: str,
        category: Optional[str] = None,
        session_id: Optional[str] = None,
        confidence: float = 1.0,
    ) -> MemoryItem:
        """Directly store a fact as a memory item.

        This bypasses the extraction process and stores the content directly.

        Args:
            content: The fact to remember.
            category: Optional category.
            session_id: Optional session ID.
            confidence: Confidence level.

        Returns:
            The stored MemoryItem.
        """
        # Create a minimal resource for traceability
        resource = self.resource_layer.store(
            resource_type="feedback",
            content=content,
            metadata={"source": "direct_remember", "session_id": session_id},
        )

        # Store the item
        item = self.item_layer.store(
            content=content,
            source_resource_id=resource.resource_id,
            category=category,
            confidence=confidence,
        )

        # Track for consolidation
        if category:
            self._pending_consolidation[category] = (
                self._pending_consolidation.get(category, 0) + 1
            )

        # Emit event
        if session_id:
            event = create_memory_extracted_event(
                session_id=session_id,
                memory_id=item.item_id,
                memory_type="direct",
                content=item.content,
                confidence=item.confidence,
                metadata={"category": item.category},
            )
            self._event_store.append(event)

        return item

    # =========================================================================
    # Retrieval Operations
    # =========================================================================

    async def retrieve(
        self,
        query: str,
        k: int = 10,
        category: Optional[str] = None,
        mode: RetrievalMode = RetrievalMode.HYBRID,
    ) -> list[MemoryItem]:
        """Retrieve relevant memory items.

        Args:
            query: Search query.
            k: Maximum number of results.
            category: Optional category filter.
            mode: Retrieval mode (RAG, LLM, or HYBRID).

        Returns:
            List of relevant MemoryItems.
        """
        return await self.retriever.retrieve(
            query=query,
            k=k,
            category=category,
            force_mode=mode if mode != RetrievalMode.HYBRID else None,
        )

    async def retrieve_with_context(
        self,
        query: str,
        k: int = 10,
        category: Optional[str] = None,
        mode: RetrievalMode = RetrievalMode.HYBRID,
    ) -> tuple[list[MemoryItem], RetrievalContext]:
        """Retrieve with full context information.

        Args:
            query: Search query.
            k: Maximum number of results.
            category: Optional category filter.
            mode: Retrieval mode.

        Returns:
            Tuple of (items, context).
        """
        result, context = await self.retriever.retrieve_with_context(
            query=query,
            k=k,
            category=category,
            force_mode=mode if mode != RetrievalMode.HYBRID else None,
        )
        return result.items, context

    async def recall(
        self,
        query: str,
        k: int = 10,
        category: Optional[str] = None,
        mode: str = "hybrid",
    ) -> list[dict[str, Any]]:
        """Retrieve memories as dictionaries (for tool use).

        Args:
            query: Search query.
            k: Maximum number of results.
            category: Optional category filter.
            mode: Retrieval mode string (rag, llm, hybrid).

        Returns:
            List of memory dictionaries.
        """
        retrieval_mode = RetrievalMode(mode.lower())
        items = await self.retrieve(query, k, category, retrieval_mode)

        return [
            {
                "item_id": item.item_id,
                "content": item.content,
                "category": item.category,
                "confidence": item.confidence,
                "source_resource_id": item.source_resource_id,
            }
            for item in items
        ]

    # =========================================================================
    # Category Operations
    # =========================================================================

    async def get_category_content(self, name: str) -> Optional[str]:
        """Get the markdown content of a category.

        Args:
            name: Category name.

        Returns:
            Markdown content if found, None otherwise.
        """
        category = self.category_layer.get_by_name(name)
        return category.markdown_content if category else None

    async def list_categories(self) -> list[dict[str, Any]]:
        """List all categories with metadata.

        Returns:
            List of category dictionaries.
        """
        categories = self.category_layer.list_all()
        return [
            {
                "category_id": cat.category_id,
                "name": cat.name,
                "description": cat.description,
                "item_count": len(cat.item_ids),
                "updated_at": cat.updated_at.isoformat(),
            }
            for cat in categories
        ]

    async def consolidate_category(
        self,
        name: str,
        session_id: Optional[str] = None,
        force_rebuild: bool = False,
    ) -> Optional[ConsolidationResult]:
        """Consolidate memory items into a category.

        Args:
            name: Category name.
            session_id: Optional session ID for event tracking.
            force_rebuild: If True, rebuild from all items.

        Returns:
            ConsolidationResult if successful, None if no items.
        """
        # Get or create category
        category = self.category_layer.get_by_name(name)
        existing_content = ""

        if category is None:
            # Create new category
            template = self.consolidator.get_template(name)
            description = template["description"] if template else f"Knowledge about {name}"
            category = self.category_layer.create(name=name, description=description)
        else:
            existing_content = category.markdown_content if not force_rebuild else ""

        # Get items for this category
        items = self.item_layer.get_by_category(name)

        if not items:
            return None

        # Consolidate
        result = await self.consolidator.consolidate(
            category_name=name,
            items=items,
            existing_content=existing_content,
            description=category.description,
            trigger=ConsolidationTrigger.MANUAL,
        )

        # Update category
        self.category_layer.update_content(name, result.markdown_content)
        self.category_layer.add_items(name, [item.item_id for item in items])

        # Reset pending count
        self._pending_consolidation[name] = 0

        # Emit event
        if session_id:
            event = create_category_consolidated_event(
                session_id=session_id,
                category_id=category.category_id,
                category_name=name,
                items_consolidated=result.items_consolidated,
                is_incremental=result.is_incremental,
                trigger=result.trigger.value,
            )
            self._event_store.append(event)

        return result

    async def _check_consolidation_triggers(
        self,
        session_id: Optional[str] = None,
    ) -> None:
        """Check if any categories need consolidation.

        Args:
            session_id: Optional session ID for event tracking.
        """
        for category, pending_count in self._pending_consolidation.items():
            if pending_count >= self.consolidator.item_threshold:
                await self.consolidate_category(category, session_id)

    # =========================================================================
    # Statistics and Health
    # =========================================================================

    def get_stats(self) -> MemoryStats:
        """Get statistics about the memory system.

        Returns:
            MemoryStats object with counts.
        """
        # Count resources by type
        resources_by_type: dict[str, int] = {}
        for rtype in ["conversation", "document", "config", "feedback"]:
            resources_by_type[rtype] = self.resource_layer.count(rtype)

        # Count items by category
        items_by_category: dict[str, int] = {}
        for category in self.item_layer.list_categories():
            items_by_category[category] = self.item_layer.count(category)

        return MemoryStats(
            resource_count=self.resource_layer.count(),
            item_count=self.item_layer.count(),
            category_count=self.category_layer.count(),
            resources_by_type=resources_by_type,
            items_by_category=items_by_category,
        )

    # =========================================================================
    # Cleanup Operations
    # =========================================================================

    async def delete_resource_and_items(self, resource_id: str) -> int:
        """Delete a resource and all items extracted from it.

        Args:
            resource_id: Resource ID to delete.

        Returns:
            Number of items deleted.
        """
        # Delete items first
        deleted_count = self.item_layer.delete_by_resource(resource_id)

        # Delete resource
        self.resource_layer.delete(resource_id)

        return deleted_count


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "MemoryManager",
    "MemoryStats",
    "DEFAULT_BASE_DIR",
]
