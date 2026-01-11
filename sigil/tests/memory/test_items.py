"""Tests for Layer 2: ItemLayer.

Tests cover:
- CRUD operations
- Embedding generation
- Vector search
- Category filtering
- Traceability to resources
"""

import pytest
import tempfile
import shutil
import numpy as np
from pathlib import Path

from sigil.memory.layers.items import (
    ItemLayer,
    EmbeddingService,
    VectorIndex,
    DEFAULT_EMBEDDING_DIMENSIONS,
)
from sigil.config.schemas.memory import MemoryItem


@pytest.fixture
def temp_storage_dir():
    """Create a temporary storage directory for tests."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def item_layer(temp_storage_dir):
    """Create an ItemLayer instance with temporary storage."""
    return ItemLayer(storage_dir=temp_storage_dir)


@pytest.fixture
def embedding_service():
    """Create an EmbeddingService instance."""
    return EmbeddingService()


@pytest.fixture
def vector_index(temp_storage_dir):
    """Create a VectorIndex instance."""
    return VectorIndex(
        dimensions=DEFAULT_EMBEDDING_DIMENSIONS,
        storage_dir=temp_storage_dir,
    )


class TestItemLayerCRUD:
    """Tests for basic CRUD operations."""

    def test_store_item(self, item_layer):
        """Test storing a new memory item."""
        item = item_layer.store(
            content="Customer prefers monthly billing",
            source_resource_id="res-123",
            category="billing_preferences",
            confidence=0.9,
        )

        assert item is not None
        assert item.item_id is not None
        assert item.content == "Customer prefers monthly billing"
        assert item.source_resource_id == "res-123"
        assert item.category == "billing_preferences"
        assert item.confidence == 0.9

    def test_store_generates_embedding(self, item_layer):
        """Test that storing generates an embedding."""
        item = item_layer.store(
            content="Test content for embedding",
            source_resource_id="res-123",
            generate_embedding=True,
        )

        assert item.embedding is not None
        assert len(item.embedding) == DEFAULT_EMBEDDING_DIMENSIONS

    def test_store_without_embedding(self, item_layer):
        """Test storing without generating embedding."""
        item = item_layer.store(
            content="Test content",
            source_resource_id="res-123",
            generate_embedding=False,
        )

        assert item.embedding is None

    def test_get_item(self, item_layer):
        """Test retrieving an item by ID."""
        stored = item_layer.store(
            content="Test content",
            source_resource_id="res-123",
        )

        retrieved = item_layer.get(stored.item_id)

        assert retrieved is not None
        assert retrieved.item_id == stored.item_id
        assert retrieved.content == stored.content

    def test_get_nonexistent_item(self, item_layer):
        """Test retrieving a non-existent item returns None."""
        result = item_layer.get("nonexistent-id")
        assert result is None

    def test_delete_item(self, item_layer):
        """Test deleting an item."""
        stored = item_layer.store(
            content="Test",
            source_resource_id="res-123",
        )

        assert item_layer.exists(stored.item_id)
        deleted = item_layer.delete(stored.item_id)
        assert deleted is True
        assert not item_layer.exists(stored.item_id)

    def test_delete_nonexistent_item(self, item_layer):
        """Test deleting a non-existent item returns False."""
        result = item_layer.delete("nonexistent-id")
        assert result is False


class TestResourceTraceability:
    """Tests for traceability to source resources."""

    def test_get_by_resource(self, item_layer):
        """Test getting items by source resource ID."""
        resource_id = "res-123"

        item_layer.store(content="Fact 1", source_resource_id=resource_id)
        item_layer.store(content="Fact 2", source_resource_id=resource_id)
        item_layer.store(content="Other fact", source_resource_id="res-456")

        items = item_layer.get_by_resource(resource_id)

        assert len(items) == 2
        for item in items:
            assert item.source_resource_id == resource_id

    def test_delete_by_resource(self, item_layer):
        """Test deleting all items from a resource."""
        resource_id = "res-123"

        item_layer.store(content="Fact 1", source_resource_id=resource_id)
        item_layer.store(content="Fact 2", source_resource_id=resource_id)
        item_layer.store(content="Keep this", source_resource_id="res-456")

        deleted_count = item_layer.delete_by_resource(resource_id)

        assert deleted_count == 2
        assert item_layer.count() == 1


class TestCategoryOperations:
    """Tests for category-based operations."""

    def test_get_by_category(self, item_layer):
        """Test getting items by category."""
        item_layer.store(
            content="Pref 1",
            source_resource_id="res-1",
            category="lead_preferences",
        )
        item_layer.store(
            content="Pref 2",
            source_resource_id="res-2",
            category="lead_preferences",
        )
        item_layer.store(
            content="Objection 1",
            source_resource_id="res-3",
            category="objection_patterns",
        )

        prefs = item_layer.get_by_category("lead_preferences")
        objections = item_layer.get_by_category("objection_patterns")

        assert len(prefs) == 2
        assert len(objections) == 1

    def test_category_normalization(self, item_layer):
        """Test that categories are normalized."""
        item_layer.store(
            content="Test",
            source_resource_id="res-1",
            category="Lead Preferences",
        )

        items = item_layer.get_by_category("lead_preferences")
        assert len(items) == 1

    def test_list_categories(self, item_layer):
        """Test listing all categories."""
        item_layer.store(content="A", source_resource_id="r1", category="cat_a")
        item_layer.store(content="B", source_resource_id="r2", category="cat_b")
        item_layer.store(content="C", source_resource_id="r3", category="cat_a")

        categories = item_layer.list_categories()

        assert "cat_a" in categories
        assert "cat_b" in categories
        assert len(categories) == 2

    def test_count_by_category(self, item_layer):
        """Test counting items by category."""
        item_layer.store(content="A", source_resource_id="r1", category="cat_a")
        item_layer.store(content="B", source_resource_id="r2", category="cat_a")
        item_layer.store(content="C", source_resource_id="r3", category="cat_b")

        assert item_layer.count("cat_a") == 2
        assert item_layer.count("cat_b") == 1
        assert item_layer.count() == 3


class TestEmbeddingService:
    """Tests for the EmbeddingService class."""

    def test_embed_text(self, embedding_service):
        """Test generating embedding for text."""
        embedding = embedding_service.embed("Hello world")

        assert embedding is not None
        assert len(embedding) == embedding_service.dimensions
        assert all(isinstance(x, float) for x in embedding)

    def test_embed_batch(self, embedding_service):
        """Test generating embeddings for batch."""
        texts = ["Hello", "World", "Test"]
        embeddings = embedding_service.embed_batch(texts)

        assert len(embeddings) == 3
        for emb in embeddings:
            assert len(emb) == embedding_service.dimensions

    def test_embed_deterministic(self, embedding_service):
        """Test that embeddings are deterministic."""
        text = "Test content"
        emb1 = embedding_service.embed(text)
        emb2 = embedding_service.embed(text)

        # With caching, should be identical
        assert emb1 == emb2

    def test_cache_clearing(self, embedding_service):
        """Test that cache can be cleared."""
        embedding_service.embed("Test")
        embedding_service.clear_cache()
        # Should not raise - cache should be empty


class TestVectorIndex:
    """Tests for the VectorIndex class."""

    def test_add_and_search(self, vector_index):
        """Test adding vectors and searching."""
        # Create test vectors
        vec1 = [1.0] + [0.0] * (DEFAULT_EMBEDDING_DIMENSIONS - 1)
        vec2 = [0.0, 1.0] + [0.0] * (DEFAULT_EMBEDDING_DIMENSIONS - 2)

        vector_index.add("item-1", vec1)
        vector_index.add("item-2", vec2)

        # Search for similar to vec1
        results = vector_index.search(vec1, k=2)

        assert len(results) > 0
        # First result should be most similar to query
        assert results[0][0] == "item-1"

    def test_add_batch(self, vector_index):
        """Test batch adding vectors."""
        items = [
            ("item-1", [1.0] + [0.0] * (DEFAULT_EMBEDDING_DIMENSIONS - 1)),
            ("item-2", [0.0, 1.0] + [0.0] * (DEFAULT_EMBEDDING_DIMENSIONS - 2)),
            ("item-3", [0.0, 0.0, 1.0] + [0.0] * (DEFAULT_EMBEDDING_DIMENSIONS - 3)),
        ]

        vector_index.add_batch(items)

        assert vector_index.count() == 3

    def test_remove(self, vector_index):
        """Test removing vectors."""
        vec = [1.0] + [0.0] * (DEFAULT_EMBEDDING_DIMENSIONS - 1)
        vector_index.add("item-1", vec)

        assert vector_index.contains("item-1")
        removed = vector_index.remove("item-1")
        assert removed is True
        assert not vector_index.contains("item-1")

    def test_dimension_mismatch(self, vector_index):
        """Test that dimension mismatch raises error."""
        wrong_dim_vec = [1.0, 2.0, 3.0]  # Wrong dimensions

        with pytest.raises(ValueError) as exc_info:
            vector_index.add("item-1", wrong_dim_vec)
        assert "dimension mismatch" in str(exc_info.value).lower()

    def test_clear(self, vector_index):
        """Test clearing the index."""
        vec = [1.0] + [0.0] * (DEFAULT_EMBEDDING_DIMENSIONS - 1)
        vector_index.add("item-1", vec)
        vector_index.add("item-2", vec)

        vector_index.clear()

        assert vector_index.count() == 0


class TestSemanticSearch:
    """Tests for semantic search functionality."""

    def test_search_by_embedding(self, item_layer):
        """Test searching items by embedding similarity."""
        # Store items with embeddings
        item_layer.store(
            content="Customer prefers email communication",
            source_resource_id="res-1",
            category="communication",
        )
        item_layer.store(
            content="Lead wants phone calls only",
            source_resource_id="res-2",
            category="communication",
        )
        item_layer.store(
            content="Product pricing is $500/month",
            source_resource_id="res-3",
            category="pricing",
        )

        # Search for communication-related items
        results = item_layer.search_by_embedding("contact method", k=3)

        assert len(results) > 0
        # Results are (item, score) tuples
        for item, score in results:
            assert isinstance(item, MemoryItem)
            assert isinstance(score, float)

    def test_search_with_category_filter(self, item_layer):
        """Test searching with category filter."""
        item_layer.store(
            content="Prefers email",
            source_resource_id="res-1",
            category="communication",
        )
        item_layer.store(
            content="Budget is $1000",
            source_resource_id="res-2",
            category="pricing",
        )

        results = item_layer.search_by_embedding(
            "contact preferences",
            k=5,
            category="communication",
        )

        for item, score in results:
            assert item.category == "communication"

    def test_search_by_content(self, item_layer):
        """Test text-based search."""
        item_layer.store(
            content="Monthly billing preferred",
            source_resource_id="res-1",
        )
        item_layer.store(
            content="Annual billing discount",
            source_resource_id="res-2",
        )

        results = item_layer.search_by_content("billing")

        assert len(results) == 2


class TestBatchOperations:
    """Tests for batch operations."""

    def test_store_batch(self, item_layer):
        """Test storing multiple items in batch."""
        items = [
            ("Fact 1", "category_a", None, 0.9),
            ("Fact 2", "category_b", None, 0.8),
            ("Fact 3", None, None, 1.0),
        ]

        stored = item_layer.store_batch(
            items=items,
            source_resource_id="res-123",
        )

        assert len(stored) == 3
        assert all(item.source_resource_id == "res-123" for item in stored)

    def test_store_batch_generates_embeddings(self, item_layer):
        """Test that batch store generates embeddings."""
        items = [
            ("Content 1", None, None, 1.0),
            ("Content 2", None, None, 1.0),
        ]

        stored = item_layer.store_batch(
            items=items,
            source_resource_id="res-123",
            generate_embeddings=True,
        )

        for item in stored:
            assert item.embedding is not None


class TestPersistence:
    """Tests for persistence functionality."""

    def test_persistence_across_instances(self, temp_storage_dir):
        """Test that items persist across layer instances."""
        # Create first instance and store items
        layer1 = ItemLayer(storage_dir=temp_storage_dir)
        stored = layer1.store(
            content="Persistent fact",
            source_resource_id="res-123",
            category="test",
        )

        # Create second instance
        layer2 = ItemLayer(storage_dir=temp_storage_dir)
        retrieved = layer2.get(stored.item_id)

        assert retrieved is not None
        assert retrieved.content == "Persistent fact"

    def test_embedding_persistence(self, temp_storage_dir):
        """Test that embeddings persist across instances."""
        layer1 = ItemLayer(storage_dir=temp_storage_dir)
        stored = layer1.store(
            content="Content with embedding",
            source_resource_id="res-123",
            generate_embedding=True,
        )
        original_embedding = stored.embedding

        layer2 = ItemLayer(storage_dir=temp_storage_dir)
        retrieved = layer2.get(stored.item_id)

        assert retrieved.embedding == original_embedding
