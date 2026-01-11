# API Contract: Phase 4 Memory System

## Overview

This document defines the comprehensive API contract for Sigil v2's Phase 4 Memory System. The memory system implements a 3-layer architecture enabling persistent knowledge storage, semantic search via RAG, and human-readable knowledge aggregation.

**Key Capabilities:**
- **Layer 1 (Resources)**: Raw source data storage with full content preservation
- **Layer 2 (Items)**: Extracted discrete facts with vector embeddings for RAG
- **Layer 3 (Categories)**: Aggregated human-readable markdown knowledge
- **Retrieval**: Dual-mode retrieval (RAG for speed, LLM for accuracy, Hybrid for both)
- **Tools**: LangChain-compatible tools for agent memory interaction

**Dependencies:**
- Phase 3 schemas (`sigil/config/schemas/memory.py`)
- Phase 3 event store (`sigil/state/store.py`)
- Phase 3 base classes (`sigil/core/base.py`)

---

## Design Principles

1. **Traceability**: Every memory item links to its source resource
2. **Dual Retrieval**: Optimize for speed (RAG) or accuracy (LLM) based on query needs
3. **Composable Layers**: Each layer operates independently but composes into unified workflows
4. **Event-Sourced**: All mutations emit events for audit trail and replay
5. **Graceful Degradation**: System operates with reduced capability if components fail
6. **Agent-First**: Tools designed for LLM consumption with clear descriptions

---

## Module Structure

```
sigil/memory/
|-- __init__.py              # Module exports
|-- manager.py               # MemoryManager (orchestrates all layers)
|-- layers/
|   |-- __init__.py          # Layer exports
|   |-- resources.py         # Layer 1: ResourceLayer
|   |-- items.py             # Layer 2: ItemLayer
|   |-- categories.py        # Layer 3: CategoryLayer
|-- extraction.py            # MemoryExtractor (resource -> items)
|-- retrieval.py             # RAGRetriever, LLMRetriever, HybridRetriever
|-- consolidation.py         # MemoryConsolidator (items -> categories)
|-- embedding.py             # EmbeddingService (text -> vectors)
|-- vector_index.py          # VectorIndex (FAISS wrapper)
```

---

## 1. MemoryManager API Contract

The `MemoryManager` is the primary interface for all memory operations. It orchestrates the three layers and provides a unified API.

### Class Definition

```python
class MemoryManager:
    """Orchestrates the 3-layer memory system.

    The MemoryManager is the single entry point for all memory operations.
    It coordinates resource storage, item extraction, category consolidation,
    and retrieval across all layers.

    Attributes:
        resource_layer: Layer 1 - raw data storage
        item_layer: Layer 2 - extracted facts with embeddings
        category_layer: Layer 3 - aggregated markdown knowledge
        event_store: Event store for audit trail (optional)
    """
```

### Constructor

```python
def __init__(
    self,
    storage_dir: Path | str = "outputs/memory",
    embedding_service: EmbeddingService | None = None,
    event_store: EventStore | None = None,
    config: MemoryConfig | None = None,
) -> None:
```

**Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `storage_dir` | `Path | str` | No | `"outputs/memory"` | Base directory for memory storage |
| `embedding_service` | `EmbeddingService` | No | Auto-created | Service for generating embeddings |
| `event_store` | `EventStore` | No | `None` | Event store for audit trail |
| `config` | `MemoryConfig` | No | `MemoryConfig()` | Memory system configuration |

**Behavior:**
1. Creates storage directory structure if it does not exist
2. Initializes all three layers with shared configuration
3. Loads existing FAISS index if available
4. Validates embedding service connectivity

---

### Core Operations

#### 1.1 `store_resource`

Stores raw content as a resource in Layer 1.

```python
async def store_resource(
    self,
    content: str,
    resource_type: str,
    metadata: dict[str, Any] | None = None,
) -> Resource:
```

**Parameters:**

| Parameter | Type | Required | Constraints | Description |
|-----------|------|----------|-------------|-------------|
| `content` | `str` | Yes | 1 - 1,000,000 chars | Raw content to store |
| `resource_type` | `str` | Yes | Valid types below | Type classification |
| `metadata` | `dict` | No | JSON-serializable | Type-specific metadata |

**Valid Resource Types:**
- `conversation`: Chat transcripts with `messages` metadata
- `document`: Documents with `title`, `source` metadata
- `config`: Agent configs with `agent_name` metadata
- `feedback`: User feedback with `rating`, `context` metadata

**Return Value:**

Returns a `Resource` object with generated `resource_id` and timestamps.

```python
Resource(
    resource_id="res_550e8400-e29b-41d4-a716-446655440000",
    resource_type="conversation",
    content="User: Hi...\nAssistant: Hello...",
    metadata={"agent_name": "lead_qualifier", "messages": 12},
    created_at=datetime(2026, 1, 11, 10, 30, 0),
)
```

**Events Emitted:**
- `ResourceStoredEvent` with `resource_id`, `resource_type`, `content_length`

**Error Handling:**

| Error Type | Condition | Recovery |
|------------|-----------|----------|
| `ValueError` | Invalid `resource_type` | Use valid type from list above |
| `ValueError` | Content exceeds 1MB | Split content into multiple resources |
| `StorageError` | Disk write failed | Check disk space, retry |

**Example:**

```python
manager = MemoryManager()

resource = await manager.store_resource(
    content="User: What's your pricing?\nAssistant: Our plans start at $99/month...",
    resource_type="conversation",
    metadata={
        "agent_name": "sales_agent",
        "session_id": "sess_abc123",
        "messages": 2,
    },
)

print(f"Stored resource: {resource.resource_id}")
```

---

#### 1.2 `extract_and_store`

Extracts discrete facts from a resource and stores them as memory items.

```python
async def extract_and_store(
    self,
    resource_id: str,
    category_hint: str | None = None,
) -> list[MemoryItem]:
```

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `resource_id` | `str` | Yes | ID of resource to extract from |
| `category_hint` | `str` | No | Suggested category for extracted items |

**Return Value:**

Returns a list of `MemoryItem` objects, each representing an extracted fact.

```python
[
    MemoryItem(
        item_id="item_123",
        content="Customer pricing starts at $99/month for basic plan",
        embedding=[0.1, 0.2, ...],  # 1536 floats
        source_resource_id="res_550e8400...",
        category="product_knowledge",
        confidence=0.92,
        created_at=datetime(2026, 1, 11, 10, 31, 0),
    ),
    MemoryItem(
        item_id="item_124",
        content="User asked about pricing - indicates purchase intent",
        embedding=[0.3, 0.4, ...],
        source_resource_id="res_550e8400...",
        category="lead_preferences",
        confidence=0.85,
        created_at=datetime(2026, 1, 11, 10, 31, 0),
    ),
]
```

**Events Emitted:**
- `ItemsExtractedEvent` with `resource_id`, `item_count`, `categories`

**Error Handling:**

| Error Type | Condition | Recovery |
|------------|-----------|----------|
| `ResourceNotFoundError` | Resource ID not found | Verify resource exists |
| `ExtractionError` | LLM extraction failed | Retry with simpler prompt |
| `EmbeddingError` | Embedding generation failed | Retry with backoff |

**Example:**

```python
items = await manager.extract_and_store(
    resource_id="res_550e8400...",
    category_hint="product_knowledge",
)

print(f"Extracted {len(items)} facts from resource")
for item in items:
    print(f"  - {item.content[:50]}... (confidence: {item.confidence:.2f})")
```

---

#### 1.3 `retrieve`

Retrieves relevant memory items based on a query.

```python
async def retrieve(
    self,
    query: str,
    mode: RetrievalMode = RetrievalMode.HYBRID,
    k: int = 5,
    category_filter: str | None = None,
) -> RetrievalResult:
```

**Parameters:**

| Parameter | Type | Required | Default | Constraints | Description |
|-----------|------|----------|---------|-------------|-------------|
| `query` | `str` | Yes | - | 1 - 10,000 chars | Search query |
| `mode` | `RetrievalMode` | No | `HYBRID` | Valid enum | Retrieval strategy |
| `k` | `int` | No | `5` | 1 - 100 | Max items to return |
| `category_filter` | `str` | No | `None` | Valid category | Filter by category |

**Return Value:**

```python
@dataclass
class RetrievalResult:
    """Result from memory retrieval operation."""
    items: list[MemoryItem]          # Retrieved items, sorted by relevance
    mode_used: RetrievalMode         # Actual mode used (may differ if escalated)
    total_found: int                 # Total matching items before k limit
    search_time_ms: float            # Search duration in milliseconds
    confidence_scores: list[float]   # Per-item confidence/similarity scores
    escalated: bool = False          # True if hybrid escalated to LLM
```

**Events Emitted:**
- `RetrievalPerformedEvent` (optional, for debugging) with `query`, `mode`, `result_count`

**Error Handling:**

| Error Type | Condition | Recovery |
|------------|-----------|----------|
| `RetrievalError` | Search failed | Falls back to sequential search |
| `EmbeddingError` | Query embedding failed | Retry with backoff |
| `ValueError` | Invalid mode or k | Use valid parameters |

**Example:**

```python
result = await manager.retrieve(
    query="What pricing plans are available?",
    mode=RetrievalMode.HYBRID,
    k=5,
)

print(f"Found {result.total_found} items in {result.search_time_ms:.1f}ms")
print(f"Mode used: {result.mode_used.value}")
if result.escalated:
    print("(Escalated from RAG to LLM for accuracy)")

for item, score in zip(result.items, result.confidence_scores):
    print(f"  [{score:.2f}] {item.content[:60]}...")
```

---

#### 1.4 `get_category_content`

Retrieves the aggregated markdown content for a category.

```python
async def get_category_content(
    self,
    category_name: str,
) -> str:
```

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `category_name` | `str` | Yes | Name of category to retrieve |

**Return Value:**

Returns markdown string containing aggregated knowledge.

```markdown
# Product Knowledge

## Pricing Plans

- Basic plan starts at $99/month
- Professional plan at $299/month includes advanced features
- Enterprise pricing available on request

## Key Features

- Real-time analytics dashboard
- API access with 10,000 requests/month
- 24/7 email support

---
*Last updated: 2026-01-11 10:45:00*
*Sources: 5 memory items from 3 resources*
```

**Error Handling:**

| Error Type | Condition | Recovery |
|------------|-----------|----------|
| `CategoryNotFoundError` | Category does not exist | Create category first |

---

#### 1.5 `consolidate_category`

Triggers LLM-based consolidation of a category's items into markdown.

```python
async def consolidate_category(
    self,
    category_name: str,
    force: bool = False,
) -> str:
```

**Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `category_name` | `str` | Yes | - | Category to consolidate |
| `force` | `bool` | No | `False` | Force even if recently consolidated |

**Return Value:**

Returns the newly consolidated markdown content.

**Events Emitted:**
- `CategoryConsolidatedEvent` with `category_name`, `item_count`, `content_length`

**Error Handling:**

| Error Type | Condition | Recovery |
|------------|-----------|----------|
| `CategoryNotFoundError` | Category does not exist | Create category first |
| `ConsolidationError` | LLM consolidation failed | Keep previous version |

**Example:**

```python
content = await manager.consolidate_category("lead_preferences")
print(f"Consolidated category:\n{content}")
```

---

#### 1.6 `get_stats`

Returns statistics about the memory system.

```python
def get_stats(self) -> MemoryStats:
```

**Return Value:**

```python
@dataclass
class MemoryStats:
    """Statistics about the memory system state."""
    total_resources: int          # Layer 1 resource count
    total_items: int              # Layer 2 item count
    total_categories: int         # Layer 3 category count
    vector_index_size: int        # FAISS index entry count
    storage_bytes: int            # Total disk usage in bytes
    resources_by_type: dict[str, int]  # Count per resource type
    items_by_category: dict[str, int]  # Count per category
    last_extraction_at: datetime | None
    last_consolidation_at: datetime | None
```

**Example:**

```python
stats = manager.get_stats()
print(f"Memory Stats:")
print(f"  Resources: {stats.total_resources}")
print(f"  Items: {stats.total_items}")
print(f"  Categories: {stats.total_categories}")
print(f"  Index size: {stats.vector_index_size}")
print(f"  Storage: {stats.storage_bytes / 1024:.1f} KB")
```

---

#### 1.7 `is_healthy` (Property)

Checks if the memory system is operational.

```python
@property
def is_healthy(self) -> bool:
```

**Return Value:**

Returns `True` if all components are operational:
- Storage directory is writable
- FAISS index is loaded
- Embedding service is responsive

**Example:**

```python
if not manager.is_healthy:
    print("Memory system degraded - check logs")
```

---

### Additional Operations

#### 1.8 `create_category`

Creates a new memory category.

```python
async def create_category(
    self,
    name: str,
    description: str,
) -> MemoryCategory:
```

**Parameters:**

| Parameter | Type | Required | Constraints | Description |
|-----------|------|----------|-------------|-------------|
| `name` | `str` | Yes | snake_case, 1-64 chars | Unique category name |
| `description` | `str` | Yes | 10-500 chars | What this category contains |

**Return Value:**

```python
MemoryCategory(
    category_id="cat_abc123",
    name="lead_preferences",
    description="Captured preferences and requirements from leads",
    markdown_content="# Lead Preferences\n\n*No items yet.*",
    item_ids=[],
    updated_at=datetime(2026, 1, 11, 10, 30, 0),
)
```

**Events Emitted:**
- `CategoryCreatedEvent` with `category_id`, `name`

**Error Handling:**

| Error Type | Condition | Recovery |
|------------|-----------|----------|
| `ValueError` | Invalid name format | Use snake_case |
| `CategoryExistsError` | Name already exists | Use unique name |

---

#### 1.9 `list_categories`

Lists all memory categories.

```python
async def list_categories(self) -> list[MemoryCategory]:
```

**Return Value:**

Returns list of all `MemoryCategory` objects, sorted by name.

---

#### 1.10 `delete_resource`

Deletes a resource and its associated items.

```python
async def delete_resource(
    self,
    resource_id: str,
    cascade: bool = True,
) -> bool:
```

**Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `resource_id` | `str` | Yes | - | Resource to delete |
| `cascade` | `bool` | No | `True` | Also delete extracted items |

**Return Value:**

Returns `True` if deleted, `False` if not found.

**Behavior:**
1. If `cascade=True`, deletes all items with `source_resource_id` matching
2. Removes embeddings from FAISS index
3. Triggers category re-consolidation if items removed

---

## 2. Retrieval API Contract

### 2.1 RetrievalMode Enum

```python
class RetrievalMode(str, Enum):
    """Retrieval strategy selection."""
    RAG = "rag"         # Fast embedding-similarity search
    LLM = "llm"         # Slower but accurate LLM-based selection
    HYBRID = "hybrid"   # Start RAG, escalate to LLM if needed
```

### 2.2 RAGRetriever

Fast embedding-similarity search using FAISS.

```python
class RAGRetriever(BaseRetriever):
    """Retrieves memory items using embedding similarity search.

    Uses FAISS for approximate nearest neighbor search over item embeddings.
    Fast (< 100ms for 100K items) but may miss semantic nuance.

    Best for:
    - Simple factual queries
    - High-throughput scenarios
    - When speed > precision
    """
```

#### Constructor

```python
def __init__(
    self,
    item_layer: ItemLayer,
    embedding_service: EmbeddingService,
    similarity_threshold: float = 0.5,
) -> None:
```

#### Method: `retrieve`

```python
async def retrieve(
    self,
    query: str,
    k: int = 5,
    category_filter: str | None = None,
) -> RetrievalResult:
```

**Behavior:**
1. Generate embedding for query
2. Search FAISS index for top-k similar vectors
3. Filter by category if specified
4. Return items sorted by similarity score (descending)

**Return Value:**

```python
RetrievalResult(
    items=[...],
    mode_used=RetrievalMode.RAG,
    total_found=25,
    search_time_ms=12.5,
    confidence_scores=[0.92, 0.87, 0.81, 0.75, 0.68],
    escalated=False,
)
```

---

### 2.3 LLMRetriever

Accurate LLM-based item selection with reasoning.

```python
class LLMRetriever(BaseRetriever):
    """Retrieves memory items using LLM reasoning.

    Reads candidate items and uses LLM to select most relevant.
    Slower (1-3s) but handles complex semantic queries accurately.

    Best for:
    - Complex reasoning queries
    - When precision > speed
    - Ambiguous or multi-faceted queries
    """
```

#### Constructor

```python
def __init__(
    self,
    item_layer: ItemLayer,
    model: str = "anthropic:claude-sonnet-4-20250514",
    max_candidates: int = 50,
) -> None:
```

#### Method: `retrieve`

```python
async def retrieve(
    self,
    query: str,
    k: int = 5,
    category_filter: str | None = None,
) -> RetrievalResult:
```

**Behavior:**
1. Load candidate items (up to `max_candidates`)
2. Construct prompt with query and item contents
3. LLM selects top-k most relevant with confidence scores
4. Return selected items with LLM-assigned confidence

**Return Value:**

```python
RetrievalResult(
    items=[...],
    mode_used=RetrievalMode.LLM,
    total_found=50,
    search_time_ms=2100.0,
    confidence_scores=[0.95, 0.88, 0.82, 0.75, 0.70],
    escalated=False,
)
```

---

### 2.4 HybridRetriever

Combines RAG speed with LLM accuracy via escalation.

```python
class HybridRetriever(BaseRetriever):
    """Hybrid retrieval that starts with RAG and escalates to LLM.

    Uses RAG for initial retrieval, evaluates confidence, and escalates
    to LLM if results are ambiguous or low-confidence.

    Escalation criteria:
    - Top result similarity < escalation_threshold
    - High variance in similarity scores (ambiguous results)
    - Query contains complex reasoning indicators
    """
```

#### Constructor

```python
def __init__(
    self,
    rag_retriever: RAGRetriever,
    llm_retriever: LLMRetriever,
    escalation_threshold: float = 0.6,
    variance_threshold: float = 0.15,
) -> None:
```

**Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `rag_retriever` | `RAGRetriever` | Yes | - | RAG retriever instance |
| `llm_retriever` | `LLMRetriever` | Yes | - | LLM retriever instance |
| `escalation_threshold` | `float` | No | `0.6` | Min similarity to avoid escalation |
| `variance_threshold` | `float` | No | `0.15` | Max score variance before escalation |

#### Method: `retrieve`

```python
async def retrieve(
    self,
    query: str,
    k: int = 5,
    escalation_threshold: float | None = None,
    category_filter: str | None = None,
) -> RetrievalResult:
```

**Behavior:**
1. Execute RAG retrieval
2. Evaluate confidence using `ConfidenceScorer`
3. If confidence >= threshold, return RAG results
4. If confidence < threshold, escalate to LLM
5. Return results with escalation flag

**Example:**

```python
result = await hybrid_retriever.retrieve(
    query="What are the key objections we've seen from enterprise customers?",
    k=5,
    escalation_threshold=0.7,
)

if result.escalated:
    print(f"Escalated to LLM for better accuracy")
```

---

### 2.5 ConfidenceScorer

Evaluates confidence of RAG results to determine escalation.

```python
class ConfidenceScorer:
    """Scores confidence of retrieval results.

    Analyzes similarity score distribution to determine if results
    are confident (clear top matches) or ambiguous (need LLM).
    """

    def score(
        self,
        results: list[tuple[MemoryItem, float]],
        query: str,
    ) -> float:
        """
        Score confidence of retrieval results.

        Args:
            results: List of (item, similarity_score) tuples
            query: Original query for context

        Returns:
            Confidence score between 0.0 and 1.0.

        Factors considered:
        - Top result similarity (higher = more confident)
        - Score distribution variance (lower = more confident)
        - Gap between top result and rest (larger = more confident)
        """
```

---

## 3. Layer Interfaces

### 3.1 ResourceLayer (Layer 1)

Stores and retrieves raw source data.

```python
class ResourceLayer:
    """Layer 1: Raw resource storage.

    Stores complete source content (conversations, documents, configs)
    with metadata and full-text search capability.

    Storage: JSON files in outputs/memory/resources/
    Naming: {resource_type}/{resource_id}.json
    """
```

#### Methods

| Method | Signature | Description |
|--------|-----------|-------------|
| `store` | `async def store(content: str, resource_type: str, metadata: dict) -> Resource` | Create new resource |
| `get` | `async def get(resource_id: str) -> Resource | None` | Get by ID |
| `list_by_type` | `async def list_by_type(resource_type: str, limit: int = 100) -> list[Resource]` | List by type |
| `list_recent` | `async def list_recent(n: int = 10) -> list[Resource]` | Get most recent |
| `search` | `async def search(query: str, resource_type: str | None = None) -> list[Resource]` | Full-text search |
| `delete` | `async def delete(resource_id: str) -> bool` | Delete resource |

#### Constraints

| Constraint | Value | Description |
|------------|-------|-------------|
| Max content size | 1 MB | Split larger content |
| Supported types | conversation, document, config, feedback | Extensible via config |
| Retention policy | 90 days default | Configurable in MemoryConfig |

---

### 3.2 ItemLayer (Layer 2)

Manages extracted facts with vector embeddings.

```python
class ItemLayer:
    """Layer 2: Extracted memory items with embeddings.

    Stores discrete facts extracted from resources, each with:
    - Content text
    - Vector embedding for similarity search
    - Source resource link (traceability)
    - Category assignment
    - Confidence score

    Storage:
    - Items: JSON file outputs/memory/items/items.jsonl
    - Embeddings: FAISS index outputs/memory/items/index.faiss
    """
```

#### Methods

| Method | Signature | Description |
|--------|-----------|-------------|
| `store` | `async def store(item: MemoryItem) -> MemoryItem` | Store item with embedding |
| `store_batch` | `async def store_batch(items: list[MemoryItem]) -> list[MemoryItem]` | Batch store |
| `get` | `async def get(item_id: str) -> MemoryItem | None` | Get by ID |
| `get_by_resource` | `async def get_by_resource(resource_id: str) -> list[MemoryItem]` | Items from resource |
| `get_by_category` | `async def get_by_category(category: str) -> list[MemoryItem]` | Items in category |
| `search_by_embedding` | `async def search_by_embedding(embedding: list[float], k: int) -> list[tuple[MemoryItem, float]]` | Vector search |
| `delete` | `async def delete(item_id: str) -> bool` | Delete item and embedding |
| `rebuild_index` | `async def rebuild_index() -> int` | Rebuild FAISS from items |

#### Invariants

| Invariant | Description |
|-----------|-------------|
| Source link required | Every item must have valid `source_resource_id` |
| Confidence range | `confidence` must be between 0.0 and 1.0 |
| Embedding dimension | All embeddings must match configured dimension (default 1536) |
| Index sync | FAISS index must contain all and only current items |

---

### 3.3 CategoryLayer (Layer 3)

Manages aggregated human-readable knowledge.

```python
class CategoryLayer:
    """Layer 3: Aggregated markdown categories.

    Categories contain consolidated knowledge from multiple items,
    stored as human/LLM-readable markdown files.

    Storage: Markdown files in outputs/memory/categories/{name}.md
    Metadata: JSON sidecar files outputs/memory/categories/{name}.json
    """
```

#### Methods

| Method | Signature | Description |
|--------|-----------|-------------|
| `create` | `async def create(name: str, description: str) -> MemoryCategory` | Create category |
| `get` | `async def get(category_id: str) -> MemoryCategory | None` | Get by ID |
| `get_by_name` | `async def get_by_name(name: str) -> MemoryCategory | None` | Get by name |
| `list_all` | `async def list_all() -> list[MemoryCategory]` | List all categories |
| `update_content` | `async def update_content(category_id: str, content: str) -> MemoryCategory` | Update markdown |
| `add_items` | `async def add_items(category_id: str, item_ids: list[str]) -> MemoryCategory` | Link items |
| `delete` | `async def delete(category_id: str) -> bool` | Delete category |

#### Content Format

Categories store markdown with embedded metadata:

```markdown
# Category Name

## Section 1

- Fact from item_001 [^1]
- Fact from item_002 [^2]

## Section 2

...content...

---

## Sources

[^1]: Extracted from conversation with lead John (res_abc123)
[^2]: Extracted from product documentation (res_def456)

---
*Last consolidated: 2026-01-11 10:45:00*
*Items: 15 | Resources: 8*
```

---

## 4. Event Contract

The memory system emits events for audit trail and debugging.

### Event Types

```python
class MemoryEventType(str, Enum):
    """Memory system event types."""
    RESOURCE_STORED = "memory.resource_stored"
    ITEMS_EXTRACTED = "memory.items_extracted"
    CATEGORY_CREATED = "memory.category_created"
    CATEGORY_CONSOLIDATED = "memory.category_consolidated"
    RETRIEVAL_PERFORMED = "memory.retrieval_performed"  # Optional, debug only
    RESOURCE_DELETED = "memory.resource_deleted"
    ITEM_DELETED = "memory.item_deleted"
```

### Event Payloads

#### ResourceStoredEvent

```python
@dataclass
class ResourceStoredPayload:
    """Payload for RESOURCE_STORED event."""
    resource_id: str
    resource_type: str
    content_length: int
    metadata_keys: list[str]
```

#### ItemsExtractedEvent

```python
@dataclass
class ItemsExtractedPayload:
    """Payload for ITEMS_EXTRACTED event."""
    resource_id: str
    item_ids: list[str]
    item_count: int
    categories: list[str]
    extraction_time_ms: float
```

#### CategoryCreatedEvent

```python
@dataclass
class CategoryCreatedPayload:
    """Payload for CATEGORY_CREATED event."""
    category_id: str
    name: str
    description: str
```

#### CategoryConsolidatedEvent

```python
@dataclass
class CategoryConsolidatedPayload:
    """Payload for CATEGORY_CONSOLIDATED event."""
    category_id: str
    name: str
    item_count: int
    content_length: int
    consolidation_time_ms: float
```

#### RetrievalPerformedEvent (Optional)

```python
@dataclass
class RetrievalPerformedPayload:
    """Payload for RETRIEVAL_PERFORMED event. Debug only."""
    query_hash: str  # SHA256 of query (privacy)
    mode: str
    result_count: int
    search_time_ms: float
    escalated: bool
```

### Event Structure

All events follow the base structure from Phase 3:

```python
@dataclass
class MemoryEvent(Event):
    """Memory system event."""
    event_id: str                    # UUID
    event_type: MemoryEventType      # Type discriminator
    timestamp: datetime              # When event occurred
    session_id: str                  # Session context
    correlation_id: str | None       # Link related events
    payload: dict[str, Any]          # Type-specific payload
```

---

## 5. Tool API Contract

LangChain-compatible tools for agent memory interaction.

### 5.1 `recall` Tool

Search memory for relevant information.

```python
@tool
def recall(
    query: str,
    k: int = 5,
    mode: str = "hybrid",
    category: str | None = None,
) -> str:
    """Search memory for relevant information based on a query.

    Use this tool when you need to recall previously learned information,
    past conversations, or stored knowledge.

    Args:
        query: What to search for. Be specific for better results.
        k: Maximum number of results to return (1-20, default 5).
        mode: Search mode - "rag" (fast), "llm" (accurate), "hybrid" (balanced).
        category: Optional category to filter results (e.g., "lead_preferences").

    Returns:
        A formatted list of relevant memory items with:
        - Content of each item
        - Source resource type
        - Confidence score
        - Category

    Example:
        >>> recall("What pricing plans have we discussed?", k=3)
        "Found 3 relevant memories:
         1. [0.92] Basic plan is $99/month (from conversation, category: product_knowledge)
         2. [0.87] Enterprise pricing is custom (from document, category: product_knowledge)
         3. [0.81] User asked about volume discounts (from conversation, category: lead_preferences)"
    """
```

**Return Format:**

```
Found {n} relevant memories:

1. [{confidence}] {content}
   Source: {resource_type} | Category: {category}

2. [{confidence}] {content}
   Source: {resource_type} | Category: {category}

...
```

If no results: `"No relevant memories found for: {query}"`

---

### 5.2 `remember` Tool

Store a new fact or insight to memory.

```python
@tool
def remember(
    content: str,
    category: str | None = None,
    importance: str = "normal",
) -> str:
    """Store an important fact, insight, or observation to memory.

    Use this tool when you learn something valuable that should be
    remembered for future interactions.

    Args:
        content: The fact or insight to remember. Be specific and complete.
        category: Category to file under (e.g., "lead_preferences", "product_knowledge").
                 If not specified, will be auto-categorized.
        importance: Priority level - "low", "normal", "high".
                   High importance items are prioritized in retrieval.

    Returns:
        Confirmation with the stored item ID and assigned category.

    Example:
        >>> remember("Customer prefers email over phone communication",
        ...          category="lead_preferences", importance="high")
        "Remembered: item_abc123
         Category: lead_preferences
         Importance: high
         Content: Customer prefers email over phone communication"
    """
```

**Return Format:**

Success:
```
Remembered: {item_id}
Category: {category}
Importance: {importance}
Content: {content_preview}...
```

Error:
```
Failed to remember: {error_message}
```

---

### 5.3 `list_categories` Tool

List all available memory categories.

```python
@tool
def list_categories() -> str:
    """List all available memory categories.

    Use this tool to see what categories exist for organizing memories.

    Returns:
        A formatted list of categories with:
        - Name
        - Description
        - Number of items

    Example:
        >>> list_categories()
        "Memory Categories (4):

         1. lead_preferences
            Captured preferences and requirements from leads
            Items: 23

         2. product_knowledge
            Information about products and pricing
            Items: 45

         3. objection_patterns
            Common objections and successful responses
            Items: 12

         4. conversation_insights
            Meta-learnings from conversations
            Items: 8"
    """
```

---

### 5.4 `get_category` Tool

Get the full content of a memory category.

```python
@tool
def get_category(name: str) -> str:
    """Get the full aggregated content of a memory category.

    Use this tool when you need comprehensive knowledge about a topic
    that has been organized into a category.

    Args:
        name: Category name (e.g., "lead_preferences", "product_knowledge").

    Returns:
        The full markdown content of the category, including:
        - Organized sections
        - Key facts and insights
        - Source citations

    Example:
        >>> get_category("product_knowledge")
        "# Product Knowledge

         ## Pricing Plans

         - Basic: $99/month
         - Professional: $299/month
         - Enterprise: Custom pricing

         ## Key Features

         - Real-time analytics
         - API access
         - 24/7 support

         ---
         *Last updated: 2026-01-11*
         *Sources: 45 items from 12 resources*"
    """
```

**Error Return:**

```
Category '{name}' not found.

Available categories: lead_preferences, product_knowledge, objection_patterns
```

---

## 6. Data Schema Contracts

### 6.1 Resource Schema

```python
@dataclass
class Resource:
    """Layer 1: Raw source data resource.

    Attributes:
        resource_id: Unique identifier (format: res_{uuid})
        resource_type: Type classification (conversation, document, config, feedback)
        content: Full raw content
        metadata: Type-specific metadata
        created_at: Creation timestamp
    """
    resource_id: str
    resource_type: str
    content: str
    metadata: dict[str, Any]
    created_at: datetime

    def to_dict(self) -> dict[str, Any]:
        """Serialize for JSON storage."""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Resource":
        """Deserialize from JSON storage."""
```

### 6.2 MemoryItem Schema

```python
@dataclass
class MemoryItem:
    """Layer 2: Extracted fact with embedding.

    Attributes:
        item_id: Unique identifier (format: item_{uuid})
        content: Extracted fact text
        embedding: Vector embedding (1536 floats for OpenAI ada-002)
        source_resource_id: Link to source resource (traceability)
        category: Category assignment (optional)
        confidence: Extraction confidence 0.0-1.0
        created_at: Extraction timestamp
        importance: Priority level (low, normal, high)
    """
    item_id: str
    content: str
    embedding: list[float] | None
    source_resource_id: str
    category: str | None
    confidence: float
    created_at: datetime
    importance: str = "normal"
```

### 6.3 MemoryCategory Schema

```python
@dataclass
class MemoryCategory:
    """Layer 3: Aggregated knowledge category.

    Attributes:
        category_id: Unique identifier (format: cat_{uuid})
        name: Category name (snake_case)
        description: What this category contains
        markdown_content: Aggregated knowledge in markdown
        item_ids: Source memory items
        updated_at: Last consolidation timestamp
    """
    category_id: str
    name: str
    description: str
    markdown_content: str
    item_ids: list[str]
    updated_at: datetime
```

### 6.4 RetrievalResult Schema

```python
@dataclass
class RetrievalResult:
    """Result from memory retrieval operation.

    Attributes:
        items: Retrieved items sorted by relevance
        mode_used: Actual retrieval mode used
        total_found: Total matching items (before k limit)
        search_time_ms: Search duration in milliseconds
        confidence_scores: Per-item relevance scores
        escalated: True if hybrid escalated to LLM
    """
    items: list[MemoryItem]
    mode_used: RetrievalMode
    total_found: int
    search_time_ms: float
    confidence_scores: list[float]
    escalated: bool = False
```

### 6.5 MemoryStats Schema

```python
@dataclass
class MemoryStats:
    """Statistics about the memory system state.

    Attributes:
        total_resources: Layer 1 resource count
        total_items: Layer 2 item count
        total_categories: Layer 3 category count
        vector_index_size: FAISS index entry count
        storage_bytes: Total disk usage
        resources_by_type: Count per resource type
        items_by_category: Count per category
        last_extraction_at: Most recent extraction timestamp
        last_consolidation_at: Most recent consolidation timestamp
    """
    total_resources: int
    total_items: int
    total_categories: int
    vector_index_size: int
    storage_bytes: int
    resources_by_type: dict[str, int]
    items_by_category: dict[str, int]
    last_extraction_at: datetime | None
    last_consolidation_at: datetime | None
```

### 6.6 MemoryConfig Schema

```python
@dataclass
class MemoryConfig:
    """Configuration for the memory system.

    Attributes:
        storage_dir: Base directory for storage
        embedding_model: Model for embeddings
        embedding_dimension: Vector dimension
        extraction_model: Model for fact extraction
        consolidation_model: Model for category consolidation
        max_resource_size: Maximum resource content size
        default_k: Default retrieval count
        escalation_threshold: Hybrid escalation threshold
        consolidation_interval: Auto-consolidation interval (items)
        retention_days: Resource retention period
    """
    storage_dir: Path = Path("outputs/memory")
    embedding_model: str = "text-embedding-3-small"
    embedding_dimension: int = 1536
    extraction_model: str = "anthropic:claude-sonnet-4-20250514"
    consolidation_model: str = "anthropic:claude-sonnet-4-20250514"
    max_resource_size: int = 1_000_000  # 1MB
    default_k: int = 5
    escalation_threshold: float = 0.6
    consolidation_interval: int = 10  # Consolidate after 10 new items
    retention_days: int = 90
```

---

## 7. Error Contract

### Error Types

```python
class MemoryError(SigilError):
    """Base exception for memory system errors."""
    code = "MEM-000"
    name = "MemoryError"


class ResourceNotFoundError(MemoryError):
    """Resource with given ID does not exist."""
    code = "MEM-001"
    name = "ResourceNotFoundError"

    def __init__(self, resource_id: str):
        self.resource_id = resource_id
        super().__init__(
            message=f"Resource '{resource_id}' not found",
            context={"resource_id": resource_id},
            recovery_suggestions=[
                "Verify the resource ID is correct",
                "Use list_by_type() to see available resources",
            ],
        )


class CategoryNotFoundError(MemoryError):
    """Category with given name does not exist."""
    code = "MEM-002"
    name = "CategoryNotFoundError"

    def __init__(self, category_name: str, available: list[str] | None = None):
        self.category_name = category_name
        suggestions = ["Create the category with create_category()"]
        if available:
            suggestions.append(f"Available categories: {', '.join(available)}")
        super().__init__(
            message=f"Category '{category_name}' not found",
            context={"category_name": category_name, "available": available},
            recovery_suggestions=suggestions,
        )


class CategoryExistsError(MemoryError):
    """Category with given name already exists."""
    code = "MEM-003"
    name = "CategoryExistsError"

    def __init__(self, category_name: str):
        super().__init__(
            message=f"Category '{category_name}' already exists",
            context={"category_name": category_name},
            recovery_suggestions=[
                "Use a different name",
                "Use get_by_name() to access existing category",
            ],
        )


class EmbeddingError(MemoryError):
    """Failed to generate embeddings."""
    code = "MEM-004"
    name = "EmbeddingError"
    recovery_suggestions = [
        "Check embedding service API key",
        "Verify network connectivity",
        "Retry with exponential backoff",
    ]


class ExtractionError(MemoryError):
    """Failed to extract items from resource."""
    code = "MEM-005"
    name = "ExtractionError"
    recovery_suggestions = [
        "Check extraction model API key",
        "Try with smaller content",
        "Retry with simpler extraction prompt",
    ]


class ConsolidationError(MemoryError):
    """Failed to consolidate category."""
    code = "MEM-006"
    name = "ConsolidationError"
    recovery_suggestions = [
        "Previous category content is preserved",
        "Retry consolidation later",
        "Check consolidation model API key",
    ]


class RetrievalError(MemoryError):
    """Failed to retrieve from memory."""
    code = "MEM-007"
    name = "RetrievalError"
    recovery_suggestions = [
        "Falling back to sequential search",
        "Try with simpler query",
        "Check FAISS index health",
    ]


class VectorIndexError(MemoryError):
    """FAISS index is corrupted or unavailable."""
    code = "MEM-008"
    name = "VectorIndexError"
    recovery_suggestions = [
        "Rebuild index with rebuild_index()",
        "Check FAISS installation",
        "Verify index file integrity",
    ]


class StorageError(MemoryError):
    """Failed to read/write to storage."""
    code = "MEM-009"
    name = "StorageError"
    recovery_suggestions = [
        "Check disk space",
        "Verify directory permissions",
        "Check storage_dir path",
    ]
```

### Error Handling Matrix

| Operation | Error | Behavior | User Notification |
|-----------|-------|----------|-------------------|
| `store_resource` | `StorageError` | Raise exception | Yes - "Storage failed" |
| `extract_and_store` | `ExtractionError` | Log warning, return empty list | Yes - "Extraction failed" |
| `extract_and_store` | `EmbeddingError` | Retry 3x with backoff, then raise | Yes - "Embedding failed" |
| `retrieve` (RAG) | `VectorIndexError` | Fallback to sequential | No - transparent fallback |
| `retrieve` (LLM) | LLM error | Retry 2x, then raise | Yes - "Retrieval failed" |
| `consolidate_category` | `ConsolidationError` | Keep previous content | Yes - "Consolidation failed" |
| `get_category_content` | `CategoryNotFoundError` | Raise exception | Yes - "Category not found" |

### Retry Configuration

```python
@dataclass
class MemoryRetryConfig:
    """Retry configuration for memory operations."""
    embedding_max_retries: int = 3
    embedding_backoff_base: float = 1.0
    embedding_backoff_max: float = 30.0

    extraction_max_retries: int = 2
    extraction_backoff_base: float = 2.0

    consolidation_max_retries: int = 2
    consolidation_backoff_base: float = 2.0

    retrieval_max_retries: int = 2
    retrieval_backoff_base: float = 1.0
```

---

## 8. Integration Examples

### 8.1 Full Workflow: Resource to Retrieval

```python
from sigil.memory import MemoryManager, RetrievalMode

# Initialize
manager = MemoryManager(storage_dir="outputs/memory")

# 1. Store a conversation resource
resource = await manager.store_resource(
    content="""
    User: What's your pricing?
    Agent: We have three plans - Basic at $99/month, Pro at $299/month,
           and Enterprise with custom pricing.
    User: What's included in Pro?
    Agent: Pro includes unlimited users, advanced analytics, API access,
           and priority support.
    """,
    resource_type="conversation",
    metadata={"agent_name": "sales_agent", "lead_id": "lead_123"},
)
print(f"Stored: {resource.resource_id}")

# 2. Extract facts from conversation
items = await manager.extract_and_store(
    resource_id=resource.resource_id,
    category_hint="product_knowledge",
)
print(f"Extracted {len(items)} facts")

# 3. Retrieve relevant memories
result = await manager.retrieve(
    query="What pricing options are available?",
    mode=RetrievalMode.HYBRID,
    k=3,
)
print(f"Found {result.total_found} items in {result.search_time_ms:.1f}ms")
for item, score in zip(result.items, result.confidence_scores):
    print(f"  [{score:.2f}] {item.content}")

# 4. Consolidate category
await manager.consolidate_category("product_knowledge")
content = await manager.get_category_content("product_knowledge")
print(f"Category content:\n{content}")
```

### 8.2 Agent with Memory Tools

```python
from langchain_core.tools import tool
from sigil.memory import MemoryManager
from sigil.tools.builtin.memory_tools import recall, remember, list_categories, get_category

# Tools are automatically available to agents
agent_tools = [recall, remember, list_categories, get_category]

# Agent can use memory in its reasoning:
# "Let me recall what we discussed about pricing..."
# recall("pricing plans discussed")

# "I should remember this preference for future reference..."
# remember("Customer prefers annual billing over monthly", category="lead_preferences")
```

### 8.3 Category Pre-seeding

```python
# Create default ACTi categories
default_categories = [
    ("lead_preferences", "Captured preferences, requirements, and communication styles of leads"),
    ("product_knowledge", "Information about products, pricing, features, and comparisons"),
    ("objection_patterns", "Common objections encountered and successful response patterns"),
    ("conversation_insights", "Meta-learnings and patterns observed across conversations"),
    ("successful_approaches", "Documented approaches that led to positive outcomes"),
]

for name, description in default_categories:
    await manager.create_category(name, description)
```

---

## API Design Report

### Spec Files Created/Updated

- `/Users/zidane/Bland-Agents-Dataset/acti-agent-builder/docs/api-contract-memory.md` - Complete Phase 4 Memory System API contract (8 sections, 25+ interfaces)

### Core Design Decisions

1. **3-Layer Architecture**: Resources (raw) -> Items (facts + embeddings) -> Categories (aggregated markdown). This enables traceability from any knowledge back to source data while optimizing for different access patterns.

2. **Dual Retrieval Modes**: RAG for speed (< 100ms), LLM for accuracy (1-3s), Hybrid for balance. Automatic escalation when RAG confidence is low ensures quality without sacrificing performance for simple queries.

3. **Event-Sourced Mutations**: All write operations emit events to the Phase 3 event store. This provides full audit trail, enables state replay, and supports debugging retrieval behavior.

4. **Tool-First Design**: Memory tools (`recall`, `remember`, `list_categories`, `get_category`) designed for LLM consumption with comprehensive docstrings that become tool descriptions.

5. **Graceful Degradation**: Embedding failures retry with backoff, retrieval falls back to sequential search, consolidation failures preserve previous content. System remains operational even with partial component failures.

### Authentication & Security

- Method: None (local system, inherits host access controls)
- File permissions: Standard filesystem permissions apply
- No API keys stored in memory files (only content and metadata)
- Query content optionally hashed in debug events for privacy

### Open Questions

1. **Embedding Provider**: Default to OpenAI `text-embedding-3-small`, but should we support local embeddings (sentence-transformers) for offline operation?

2. **Consolidation Triggers**: Currently manual or after N items. Should we add time-based triggers (daily consolidation)?

3. **Multi-tenancy**: Current design is single-tenant. Should categories be namespaced for multi-agent scenarios?

4. **Compression**: Should old resources be compressed to save storage? What retention policy for embeddings?

### Implementation Guidance

**Phase 4.1 - Resources (Week 1):**
1. Implement `ResourceLayer` with CRUD operations
2. Add JSON storage with file locking
3. Implement full-text search (simple substring initially)
4. Write unit tests for resource operations

**Phase 4.2 - Items (Week 1-2):**
1. Implement `EmbeddingService` with OpenAI integration
2. Implement `VectorIndex` FAISS wrapper
3. Implement `ItemLayer` with embedding storage
4. Implement `MemoryExtractor` with LLM extraction
5. Write tests for extraction and search

**Phase 4.3 - Categories (Week 2):**
1. Implement `CategoryLayer` with markdown storage
2. Implement `MemoryConsolidator` with LLM aggregation
3. Create default ACTi category templates
4. Write tests for consolidation

**Phase 4.4 - Retrieval (Week 2-3):**
1. Implement `RAGRetriever` using VectorIndex
2. Implement `LLMRetriever` with candidate selection
3. Implement `HybridRetriever` with escalation logic
4. Implement `ConfidenceScorer`
5. Write tests for all retrieval modes

**Phase 4.5 - Manager (Week 3):**
1. Implement `MemoryManager` orchestration
2. Add event emission integration
3. Implement health checks and stats
4. Write integration tests

**Phase 4.6 - Tools (Week 3):**
1. Implement `recall`, `remember`, `list_categories`, `get_category` tools
2. Test tool integration with mock agents
3. Write tool documentation

### Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `faiss-cpu` | >= 1.7.4 | Vector similarity search |
| `openai` | >= 1.0.0 | Embedding generation (default) |
| `langchain-core` | >= 0.1.0 | Tool definitions |
| `anthropic` | >= 0.20.0 | LLM extraction/consolidation |

### File Locations

```
sigil/memory/
|-- __init__.py                    # Module exports
|-- manager.py                     # MemoryManager class
|-- layers/
|   |-- __init__.py
|   |-- resources.py               # ResourceLayer
|   |-- items.py                   # ItemLayer
|   |-- categories.py              # CategoryLayer
|-- extraction.py                  # MemoryExtractor
|-- retrieval.py                   # RAG/LLM/Hybrid retrievers
|-- consolidation.py               # MemoryConsolidator
|-- embedding.py                   # EmbeddingService
|-- vector_index.py                # VectorIndex (FAISS wrapper)

sigil/tools/builtin/
|-- __init__.py
|-- memory_tools.py                # recall, remember, list_categories, get_category

outputs/memory/                    # Runtime storage
|-- resources/
|   |-- conversation/
|   |-- document/
|   |-- config/
|   |-- feedback/
|-- items/
|   |-- items.jsonl
|   |-- index.faiss
|-- categories/
|   |-- lead_preferences.md
|   |-- product_knowledge.md
|   |-- ...
```

---

## Version History

- **1.0.0** (2026-01-11): Initial API contract
  - MemoryManager orchestration interface
  - 3-layer architecture (Resources, Items, Categories)
  - Dual retrieval system (RAG, LLM, Hybrid)
  - Event contract for audit trail
  - LangChain-compatible tools
  - Complete error taxonomy with recovery strategies
