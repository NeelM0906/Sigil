"""Memory module for Sigil v2 framework.

This module implements the 3-layer memory architecture:

Layer 1 - Resources:
    Raw source data storage for conversations, documents, configs, and feedback.
    Each resource provides traceability for extracted memory items.

Layer 2 - Memory Items:
    Discrete facts extracted from resources with embeddings for semantic search.
    Each item links back to its source resource.

Layer 3 - Memory Categories:
    Aggregated knowledge in markdown format, readable by both humans and LLMs.
    Categories consolidate multiple items into coherent summaries.

Retrieval Modes:
    - RAG: Fast embedding-based similarity search
    - LLM: Accurate reading-based selection
    - Hybrid: Starts with RAG, escalates to LLM if confidence is low

Key Components:
    - MemoryManager: Main orchestrator for all memory operations
    - ResourceLayer: Layer 1 storage
    - ItemLayer: Layer 2 storage with FAISS vector index
    - CategoryLayer: Layer 3 storage with markdown files
    - MemoryExtractor: LLM-based fact extraction
    - MemoryConsolidator: LLM-based category consolidation
    - Retrievers: RAG, LLM, and Hybrid retrieval implementations

Example:
    >>> from sigil.memory import MemoryManager
    >>>
    >>> manager = MemoryManager()
    >>>
    >>> # Store a conversation
    >>> resource = await manager.store_resource(
    ...     resource_type="conversation",
    ...     content="User: I prefer monthly billing\\nAgent: Noted!",
    ...     session_id="sess-123"
    ... )
    >>>
    >>> # Extract facts
    >>> items = await manager.extract_and_store(resource.resource_id)
    >>>
    >>> # Retrieve relevant memories
    >>> results = await manager.retrieve("billing preferences", k=5)
"""

# Layer implementations
from sigil.memory.layers import (
    # Layer 1: Resources
    ResourceLayer,
    ConversationResource,
    VALID_RESOURCE_TYPES,
    # Layer 2: Items
    ItemLayer,
    EmbeddingService,
    EmbeddingProvider,
    VectorIndex,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_EMBEDDING_DIMENSIONS,
    # Layer 3: Categories
    CategoryLayer,
)

# Extraction
from sigil.memory.extraction import (
    MemoryExtractor,
    ExtractedFact,
    ExtractionStrategy,
    ConversationExtractionStrategy,
    DocumentExtractionStrategy,
    ConfigExtractionStrategy,
    FeedbackExtractionStrategy,
)

# Consolidation
from sigil.memory.consolidation import (
    MemoryConsolidator,
    ConsolidationTrigger,
    ConsolidationResult,
    CATEGORY_TEMPLATES,
)

# Retrieval
from sigil.memory.retrieval import (
    RetrievalMode,
    ScoredItem,
    RetrievalContext,
    ConfidenceScorer,
    RAGRetriever,
    LLMRetriever,
    HybridRetriever,
)

# Manager
from sigil.memory.manager import (
    MemoryManager,
    MemoryStats,
    DEFAULT_BASE_DIR,
)

__all__ = [
    # Manager (main entry point)
    "MemoryManager",
    "MemoryStats",
    "DEFAULT_BASE_DIR",
    # Layer 1: Resources
    "ResourceLayer",
    "ConversationResource",
    "VALID_RESOURCE_TYPES",
    # Layer 2: Items
    "ItemLayer",
    "EmbeddingService",
    "EmbeddingProvider",
    "VectorIndex",
    "DEFAULT_EMBEDDING_MODEL",
    "DEFAULT_EMBEDDING_DIMENSIONS",
    # Layer 3: Categories
    "CategoryLayer",
    # Extraction
    "MemoryExtractor",
    "ExtractedFact",
    "ExtractionStrategy",
    "ConversationExtractionStrategy",
    "DocumentExtractionStrategy",
    "ConfigExtractionStrategy",
    "FeedbackExtractionStrategy",
    # Consolidation
    "MemoryConsolidator",
    "ConsolidationTrigger",
    "ConsolidationResult",
    "CATEGORY_TEMPLATES",
    # Retrieval
    "RetrievalMode",
    "ScoredItem",
    "RetrievalContext",
    "ConfidenceScorer",
    "RAGRetriever",
    "LLMRetriever",
    "HybridRetriever",
]
