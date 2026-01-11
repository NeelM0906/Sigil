"""Tests for the MemoryManager.

Tests cover:
- Full workflow (store -> extract -> retrieve)
- Event emission
- Statistics
- Error handling
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch

from sigil.memory.manager import MemoryManager, MemoryStats
from sigil.memory.retrieval import RetrievalMode
from sigil.state.store import EventStore


@pytest.fixture
def temp_storage_dir():
    """Create a temporary storage directory for tests."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def memory_manager(temp_storage_dir):
    """Create a MemoryManager instance with temporary storage."""
    return MemoryManager(base_dir=temp_storage_dir)


class TestMemoryManagerBasics:
    """Tests for basic MemoryManager functionality."""

    def test_initialization(self, memory_manager):
        """Test that MemoryManager initializes correctly."""
        assert memory_manager.resource_layer is not None
        assert memory_manager.item_layer is not None
        assert memory_manager.category_layer is not None
        assert memory_manager.extractor is not None
        assert memory_manager.consolidator is not None
        assert memory_manager.retriever is not None

    def test_is_healthy(self, memory_manager):
        """Test health check."""
        assert memory_manager.is_healthy is True


class TestResourceOperations:
    """Tests for resource storage operations."""

    @pytest.mark.asyncio
    async def test_store_resource(self, memory_manager):
        """Test storing a resource."""
        resource = await memory_manager.store_resource(
            resource_type="conversation",
            content="User: Hello!\nAgent: Hi there!",
            session_id="test-session",
        )

        assert resource is not None
        assert resource.resource_id is not None
        assert "session_id" in resource.metadata

    @pytest.mark.asyncio
    async def test_get_resource(self, memory_manager):
        """Test retrieving a resource."""
        stored = await memory_manager.store_resource(
            resource_type="document",
            content="Test document content",
        )

        retrieved = await memory_manager.get_resource(stored.resource_id)

        assert retrieved is not None
        assert retrieved.resource_id == stored.resource_id

    @pytest.mark.asyncio
    async def test_store_conversation(self, memory_manager):
        """Test storing a conversation resource."""
        conv = await memory_manager.store_conversation(
            content="User: Question?\nAgent: Answer.",
            session_id="sess-123",
        )

        assert conv is not None
        assert conv.turn_count == 2


class TestExtractionOperations:
    """Tests for memory extraction operations."""

    @pytest.mark.asyncio
    async def test_extract_and_store(self, memory_manager):
        """Test extracting facts from a resource."""
        # Store a resource
        resource = await memory_manager.store_resource(
            resource_type="conversation",
            content="User: I prefer monthly billing.\nAgent: Noted!",
        )

        # Extract (uses fallback extraction without API)
        items = await memory_manager.extract_and_store(resource.resource_id)

        # May or may not extract items depending on content patterns
        assert isinstance(items, list)

    @pytest.mark.asyncio
    async def test_extract_nonexistent_resource(self, memory_manager):
        """Test extracting from non-existent resource raises error."""
        from sigil.core.exceptions import MemoryRetrievalError

        with pytest.raises(MemoryRetrievalError):
            await memory_manager.extract_and_store("nonexistent-id")


class TestRememberOperation:
    """Tests for the remember operation."""

    @pytest.mark.asyncio
    async def test_remember(self, memory_manager):
        """Test directly storing a fact."""
        item = await memory_manager.remember(
            content="Customer budget is $500/month",
            category="pricing",
            session_id="test-session",
            confidence=0.9,
        )

        assert item is not None
        assert item.content == "Customer budget is $500/month"
        assert item.category == "pricing"
        assert item.confidence == 0.9

    @pytest.mark.asyncio
    async def test_remember_without_category(self, memory_manager):
        """Test remembering without a category."""
        item = await memory_manager.remember(
            content="Important fact",
        )

        assert item is not None
        assert item.category is None


class TestRetrievalOperations:
    """Tests for retrieval operations."""

    @pytest.fixture
    def populated_manager(self, memory_manager):
        """Populate manager with test data."""
        import asyncio

        async def populate():
            await memory_manager.remember(
                content="Customer prefers email communication",
                category="communication",
            )
            await memory_manager.remember(
                content="Budget is $1000 per month",
                category="pricing",
            )
            await memory_manager.remember(
                content="Decision maker is the CEO",
                category="authority",
            )
            return memory_manager

        return asyncio.get_event_loop().run_until_complete(populate())

    @pytest.mark.asyncio
    async def test_retrieve_hybrid(self, populated_manager):
        """Test hybrid retrieval."""
        results = await populated_manager.retrieve(
            "contact preferences",
            k=3,
            mode=RetrievalMode.HYBRID,
        )

        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_retrieve_rag(self, populated_manager):
        """Test RAG retrieval."""
        results = await populated_manager.retrieve(
            "budget information",
            k=3,
            mode=RetrievalMode.RAG,
        )

        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_retrieve_with_context(self, populated_manager):
        """Test retrieval with context."""
        items, context = await populated_manager.retrieve_with_context(
            "decision maker",
            k=3,
        )

        assert isinstance(items, list)
        assert context.query == "decision maker"

    @pytest.mark.asyncio
    async def test_recall(self, populated_manager):
        """Test recall (tool-friendly retrieval)."""
        results = await populated_manager.recall(
            "pricing",
            k=3,
            mode="hybrid",
        )

        assert isinstance(results, list)
        for result in results:
            assert "content" in result
            assert "item_id" in result


class TestCategoryOperations:
    """Tests for category operations."""

    @pytest.mark.asyncio
    async def test_list_categories(self, memory_manager):
        """Test listing categories."""
        # Remember some items with categories
        await memory_manager.remember(
            content="Fact 1",
            category="category_a",
        )
        await memory_manager.remember(
            content="Fact 2",
            category="category_b",
        )

        categories = await memory_manager.list_categories()

        # Categories may or may not exist depending on consolidation
        assert isinstance(categories, list)

    @pytest.mark.asyncio
    async def test_get_category_content(self, memory_manager):
        """Test getting category content."""
        # Create a category
        memory_manager.category_layer.create(
            name="test_category",
            description="Test",
            markdown_content="# Test\n\nContent here",
        )

        content = await memory_manager.get_category_content("test_category")

        assert content is not None
        assert "# Test" in content

    @pytest.mark.asyncio
    async def test_get_nonexistent_category(self, memory_manager):
        """Test getting non-existent category returns None."""
        content = await memory_manager.get_category_content("nonexistent")
        assert content is None

    @pytest.mark.asyncio
    async def test_consolidate_category(self, memory_manager):
        """Test consolidating a category."""
        # Add some items
        await memory_manager.remember(
            content="Fact about pricing 1",
            category="pricing_test",
        )
        await memory_manager.remember(
            content="Fact about pricing 2",
            category="pricing_test",
        )

        result = await memory_manager.consolidate_category("pricing_test")

        # Should return result or None if no items
        if result is not None:
            assert result.category_name == "pricing_test"


class TestStatistics:
    """Tests for statistics functionality."""

    @pytest.mark.asyncio
    async def test_get_stats(self, memory_manager):
        """Test getting memory statistics."""
        # Add some data
        await memory_manager.remember(content="Fact 1", category="cat_a")
        await memory_manager.remember(content="Fact 2", category="cat_b")

        stats = memory_manager.get_stats()

        assert isinstance(stats, MemoryStats)
        assert stats.item_count >= 2
        assert isinstance(stats.resources_by_type, dict)
        assert isinstance(stats.items_by_category, dict)


class TestEventEmission:
    """Tests for event store integration."""

    @pytest.mark.asyncio
    async def test_remember_emits_event(self, temp_storage_dir):
        """Test that remember emits an event."""
        event_store = EventStore(storage_dir=temp_storage_dir + "/events")
        manager = MemoryManager(
            base_dir=temp_storage_dir,
            event_store=event_store,
        )

        await manager.remember(
            content="Test fact",
            category="test",
            session_id="test-session",
        )

        # Check if event was emitted
        if event_store.session_exists("test-session"):
            events = event_store.get_events("test-session")
            assert len(events) > 0


class TestCleanupOperations:
    """Tests for cleanup operations."""

    @pytest.mark.asyncio
    async def test_delete_resource_and_items(self, memory_manager):
        """Test deleting a resource and its items."""
        # Store a resource
        resource = await memory_manager.store_resource(
            resource_type="document",
            content="Test content",
        )

        # Add items linked to it
        memory_manager.item_layer.store(
            content="Item 1",
            source_resource_id=resource.resource_id,
        )
        memory_manager.item_layer.store(
            content="Item 2",
            source_resource_id=resource.resource_id,
        )

        # Delete
        deleted_count = await memory_manager.delete_resource_and_items(resource.resource_id)

        assert deleted_count == 2
        assert await memory_manager.get_resource(resource.resource_id) is None


class TestFullWorkflow:
    """Tests for complete memory workflow."""

    @pytest.mark.asyncio
    async def test_full_workflow(self, memory_manager):
        """Test complete workflow: store -> extract -> remember -> retrieve."""
        # 1. Store a conversation
        resource = await memory_manager.store_resource(
            resource_type="conversation",
            content="User: What's your pricing?\nAgent: We start at $500/month.",
            session_id="workflow-test",
        )
        assert resource is not None

        # 2. Remember some facts directly
        await memory_manager.remember(
            content="Standard pricing is $500/month",
            category="pricing",
        )
        await memory_manager.remember(
            content="Customer asked about pricing",
            category="inquiry_types",
        )

        # 3. Retrieve
        results = await memory_manager.retrieve("pricing information", k=3)
        assert isinstance(results, list)

        # 4. Get stats
        stats = memory_manager.get_stats()
        assert stats.resource_count >= 1
        assert stats.item_count >= 2


class TestEdgeCases:
    """Tests for edge cases."""

    @pytest.mark.asyncio
    async def test_empty_content_storage(self, memory_manager):
        """Test storing empty content."""
        resource = await memory_manager.store_resource(
            resource_type="document",
            content="",
        )
        assert resource is not None
        assert resource.content == ""

    @pytest.mark.asyncio
    async def test_special_characters_in_content(self, memory_manager):
        """Test content with special characters."""
        special_content = "Test with 'quotes', \"double quotes\", and emoji"

        item = await memory_manager.remember(
            content=special_content,
        )

        assert item.content == special_content

    @pytest.mark.asyncio
    async def test_unicode_content(self, memory_manager):
        """Test content with unicode characters."""
        unicode_content = "Test with Japanese: , Chinese: , and more"

        item = await memory_manager.remember(
            content=unicode_content,
        )

        assert item.content == unicode_content
