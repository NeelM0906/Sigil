"""Tests for memory tools.

Tests cover:
- recall tool
- remember tool
- list_categories tool
- get_category tool
"""

import pytest
import json
import tempfile
import shutil
from unittest.mock import AsyncMock, MagicMock, patch

from sigil.memory.manager import MemoryManager
from sigil.tools.builtin.memory_tools import (
    create_recall_tool,
    create_remember_tool,
    create_list_categories_tool,
    create_get_category_tool,
    create_memory_tools,
    RecallInput,
    RememberInput,
    GetCategoryInput,
    LANGCHAIN_AVAILABLE,
)


@pytest.fixture
def temp_storage_dir():
    """Create a temporary storage directory for tests."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def memory_manager(temp_storage_dir):
    """Create a MemoryManager instance."""
    return MemoryManager(base_dir=temp_storage_dir)


@pytest.fixture
def populated_manager(memory_manager):
    """Populate manager with test data."""
    import asyncio

    async def populate():
        await memory_manager.remember(
            content="Customer prefers email",
            category="communication",
        )
        await memory_manager.remember(
            content="Budget is $500",
            category="pricing",
        )

        # Create a category with content
        memory_manager.category_layer.create(
            name="test_category",
            description="Test",
            markdown_content="# Test Category\n\nContent here.",
        )

        return memory_manager

    return asyncio.get_event_loop().run_until_complete(populate())


class TestRecallTool:
    """Tests for the recall tool."""

    @pytest.mark.asyncio
    async def test_recall_basic(self, populated_manager):
        """Test basic recall functionality."""
        recall = create_recall_tool(populated_manager)
        result = await recall("communication preferences", k=3)

        data = json.loads(result)
        assert data["status"] == "success"
        assert "memories" in data

    @pytest.mark.asyncio
    async def test_recall_with_category(self, populated_manager):
        """Test recall with category filter."""
        recall = create_recall_tool(populated_manager)
        result = await recall(
            "preferences",
            k=3,
            category="communication",
        )

        data = json.loads(result)
        assert data["status"] == "success"

    @pytest.mark.asyncio
    async def test_recall_no_results(self, memory_manager):
        """Test recall with no matching results."""
        recall = create_recall_tool(memory_manager)
        result = await recall("very specific nonexistent query", k=3)

        data = json.loads(result)
        assert data["status"] == "success"
        assert len(data["memories"]) == 0

    @pytest.mark.asyncio
    async def test_recall_different_modes(self, populated_manager):
        """Test recall with different retrieval modes."""
        recall = create_recall_tool(populated_manager)

        for mode in ["rag", "llm", "hybrid"]:
            result = await recall("test", k=3, mode=mode)
            data = json.loads(result)
            assert data["status"] == "success"


class TestRememberTool:
    """Tests for the remember tool."""

    @pytest.mark.asyncio
    async def test_remember_basic(self, memory_manager):
        """Test basic remember functionality."""
        remember = create_remember_tool(memory_manager)
        result = await remember("New important fact")

        data = json.loads(result)
        assert data["status"] == "success"
        assert "item_id" in data

    @pytest.mark.asyncio
    async def test_remember_with_category(self, memory_manager):
        """Test remember with category."""
        remember = create_remember_tool(memory_manager)
        result = await remember(
            "Price is $100",
            category="pricing",
        )

        data = json.loads(result)
        assert data["status"] == "success"

    @pytest.mark.asyncio
    async def test_remember_with_confidence(self, memory_manager):
        """Test remember with confidence score."""
        remember = create_remember_tool(memory_manager)
        result = await remember(
            "Uncertain fact",
            confidence=0.7,
        )

        data = json.loads(result)
        assert data["status"] == "success"

    @pytest.mark.asyncio
    async def test_remember_with_session(self, memory_manager):
        """Test remember with session ID."""
        remember = create_remember_tool(memory_manager, session_id="test-session")
        result = await remember("Session-tracked fact")

        data = json.loads(result)
        assert data["status"] == "success"


class TestListCategoriesTool:
    """Tests for the list_categories tool."""

    @pytest.mark.asyncio
    async def test_list_categories_empty(self, memory_manager):
        """Test listing with no categories."""
        list_cats = create_list_categories_tool(memory_manager)
        result = await list_cats()

        data = json.loads(result)
        assert data["status"] == "success"
        assert "categories" in data

    @pytest.mark.asyncio
    async def test_list_categories_with_data(self, populated_manager):
        """Test listing with categories."""
        list_cats = create_list_categories_tool(populated_manager)
        result = await list_cats()

        data = json.loads(result)
        assert data["status"] == "success"
        assert "categories" in data


class TestGetCategoryTool:
    """Tests for the get_category tool."""

    @pytest.mark.asyncio
    async def test_get_category_exists(self, populated_manager):
        """Test getting an existing category."""
        get_cat = create_get_category_tool(populated_manager)
        result = await get_cat("test_category")

        data = json.loads(result)
        assert data["status"] == "success"
        assert "content" in data
        assert "# Test Category" in data["content"]

    @pytest.mark.asyncio
    async def test_get_category_not_found(self, memory_manager):
        """Test getting a non-existent category."""
        get_cat = create_get_category_tool(memory_manager)
        result = await get_cat("nonexistent")

        data = json.loads(result)
        assert data["status"] == "not_found"


class TestCreateMemoryTools:
    """Tests for the create_memory_tools factory."""

    def test_create_tools_as_functions(self, memory_manager):
        """Test creating tools as plain functions."""
        tools = create_memory_tools(memory_manager, as_langchain=False)

        assert len(tools) == 4
        assert all(callable(t) for t in tools)

    @pytest.mark.skipif(not LANGCHAIN_AVAILABLE, reason="LangChain not installed")
    def test_create_tools_as_langchain(self, memory_manager):
        """Test creating tools as LangChain tools."""
        tools = create_memory_tools(memory_manager, as_langchain=True)

        assert len(tools) == 4

    def test_create_tools_with_session(self, memory_manager):
        """Test creating tools with session ID."""
        tools = create_memory_tools(
            memory_manager,
            session_id="test-session",
            as_langchain=False,
        )

        assert len(tools) == 4


class TestInputSchemas:
    """Tests for input schema validation."""

    def test_recall_input_valid(self):
        """Test valid RecallInput."""
        input_obj = RecallInput(
            query="test query",
            k=5,
            category="test",
            mode="hybrid",
        )

        assert input_obj.query == "test query"
        assert input_obj.k == 5

    def test_recall_input_defaults(self):
        """Test RecallInput defaults."""
        input_obj = RecallInput(query="test")

        assert input_obj.k == 5
        assert input_obj.category is None
        assert input_obj.mode == "hybrid"

    def test_recall_input_k_bounds(self):
        """Test RecallInput k bounds validation."""
        # Valid bounds
        RecallInput(query="test", k=1)
        RecallInput(query="test", k=20)

        # Invalid bounds should fail
        with pytest.raises(ValueError):
            RecallInput(query="test", k=0)
        with pytest.raises(ValueError):
            RecallInput(query="test", k=21)

    def test_remember_input_valid(self):
        """Test valid RememberInput."""
        input_obj = RememberInput(
            content="Test fact",
            category="test_category",
            confidence=0.9,
        )

        assert input_obj.content == "Test fact"
        assert input_obj.confidence == 0.9

    def test_remember_input_confidence_bounds(self):
        """Test RememberInput confidence bounds."""
        # Valid bounds
        RememberInput(content="test", confidence=0.0)
        RememberInput(content="test", confidence=1.0)

        # Invalid bounds should fail
        with pytest.raises(ValueError):
            RememberInput(content="test", confidence=-0.1)
        with pytest.raises(ValueError):
            RememberInput(content="test", confidence=1.1)

    def test_get_category_input_valid(self):
        """Test valid GetCategoryInput."""
        input_obj = GetCategoryInput(name="test_category")
        assert input_obj.name == "test_category"


@pytest.mark.skipif(not LANGCHAIN_AVAILABLE, reason="LangChain not installed")
class TestLangChainTools:
    """Tests for LangChain tool classes."""

    def test_recall_tool_class(self, memory_manager):
        """Test RecallTool class."""
        from sigil.tools.builtin.memory_tools import RecallTool

        tool = RecallTool(memory_manager=memory_manager)

        assert tool.name == "recall"
        assert "retrieve" in tool.description.lower()

    def test_remember_tool_class(self, memory_manager):
        """Test RememberTool class."""
        from sigil.tools.builtin.memory_tools import RememberTool

        tool = RememberTool(memory_manager=memory_manager)

        assert tool.name == "remember"
        assert "store" in tool.description.lower()

    def test_list_categories_tool_class(self, memory_manager):
        """Test ListCategoriesTool class."""
        from sigil.tools.builtin.memory_tools import ListCategoriesTool

        tool = ListCategoriesTool(memory_manager=memory_manager)

        assert tool.name == "list_categories"

    def test_get_category_tool_class(self, memory_manager):
        """Test GetCategoryTool class."""
        from sigil.tools.builtin.memory_tools import GetCategoryTool

        tool = GetCategoryTool(memory_manager=memory_manager)

        assert tool.name == "get_category"


class TestErrorHandling:
    """Tests for error handling in tools."""

    @pytest.mark.asyncio
    async def test_recall_handles_errors(self, memory_manager):
        """Test that recall handles errors gracefully."""
        recall = create_recall_tool(memory_manager)

        # Should not raise, but return error in JSON
        result = await recall("test", k=5)
        data = json.loads(result)
        # Either success or error status
        assert data["status"] in ["success", "error"]

    @pytest.mark.asyncio
    async def test_remember_handles_errors(self, memory_manager):
        """Test that remember handles errors gracefully."""
        remember = create_remember_tool(memory_manager)

        result = await remember("test")
        data = json.loads(result)
        assert data["status"] in ["success", "error"]
