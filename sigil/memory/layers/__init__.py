"""Memory layer implementations for Sigil v2.

This module contains the concrete implementations of the 3-layer
memory architecture:

Layers:
    - ResourceLayer (Layer 1): Stores raw source data (conversations,
        documents, configs, feedback). Provides full traceability.

    - ItemLayer (Layer 2): Stores discrete facts extracted from resources.
        Includes embeddings for semantic search via FAISS.

    - CategoryLayer (Layer 3): Stores aggregated knowledge in markdown
        format. Human and LLM-readable summaries of learned information.

Example:
    >>> from sigil.memory.layers import ResourceLayer, ItemLayer, CategoryLayer
    >>>
    >>> resources = ResourceLayer("outputs/memory/resources")
    >>> items = ItemLayer("outputs/memory/items")
    >>> categories = CategoryLayer("outputs/memory/categories")
    >>>
    >>> # Store a conversation
    >>> resource = resources.store(
    ...     resource_type="conversation",
    ...     content="User: Hello!\\nAgent: Hi there!"
    ... )
"""

from sigil.memory.layers.resources import (
    ResourceLayer,
    ConversationResource,
    VALID_RESOURCE_TYPES,
    DEFAULT_STORAGE_DIR as RESOURCES_STORAGE_DIR,
)

from sigil.memory.layers.items import (
    ItemLayer,
    EmbeddingService,
    EmbeddingProvider,
    VectorIndex,
    DEFAULT_STORAGE_DIR as ITEMS_STORAGE_DIR,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_EMBEDDING_DIMENSIONS,
)

from sigil.memory.layers.categories import (
    CategoryLayer,
    DEFAULT_STORAGE_DIR as CATEGORIES_STORAGE_DIR,
)

__all__ = [
    # Layer 1: Resources
    "ResourceLayer",
    "ConversationResource",
    "VALID_RESOURCE_TYPES",
    "RESOURCES_STORAGE_DIR",
    # Layer 2: Items
    "ItemLayer",
    "EmbeddingService",
    "EmbeddingProvider",
    "VectorIndex",
    "ITEMS_STORAGE_DIR",
    "DEFAULT_EMBEDDING_MODEL",
    "DEFAULT_EMBEDDING_DIMENSIONS",
    # Layer 3: Categories
    "CategoryLayer",
    "CATEGORIES_STORAGE_DIR",
]
