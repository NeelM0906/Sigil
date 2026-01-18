"""Tests for the BuiltinToolExecutor class.

Tests cover:
- Tool name parsing
- Memory tool execution
- Planning tool execution
- Argument interpolation
- Error handling
"""

import pytest
import json
from unittest.mock import MagicMock, AsyncMock, patch

from sigil.planning.executors.builtin_executor import (
    BuiltinToolExecutor,
    BuiltinExecutorError,
    ToolNotFoundError,
    MissingDependencyError,
    ToolExecutionError,
)


class TestToolNameParsing:
    """Tests for tool name parsing."""

    def test_parse_valid_tool_name(self):
        """Test parsing a valid tool name."""
        category, operation = BuiltinToolExecutor.parse_tool_name("memory.recall")
        assert category == "memory"
        assert operation == "recall"

    def test_parse_planning_tool_name(self):
        """Test parsing planning tool name."""
        category, operation = BuiltinToolExecutor.parse_tool_name("planning.create_plan")
        assert category == "planning"
        assert operation == "create_plan"

    def test_parse_invalid_tool_name(self):
        """Test parsing invalid tool name."""
        with pytest.raises(ValueError) as exc_info:
            BuiltinToolExecutor.parse_tool_name("memory")
        assert "Invalid tool name format" in str(exc_info.value)

    def test_is_builtin_tool_memory(self):
        """Test is_builtin_tool for memory tools."""
        assert BuiltinToolExecutor.is_builtin_tool("memory.recall") is True
        assert BuiltinToolExecutor.is_builtin_tool("memory.store") is True

    def test_is_builtin_tool_planning(self):
        """Test is_builtin_tool for planning tools."""
        assert BuiltinToolExecutor.is_builtin_tool("planning.create_plan") is True
        assert BuiltinToolExecutor.is_builtin_tool("planning.get_status") is True

    def test_is_builtin_tool_external(self):
        """Test is_builtin_tool for external tools (Tavily, etc.)."""
        assert BuiltinToolExecutor.is_builtin_tool("websearch.search") is False
        assert BuiltinToolExecutor.is_builtin_tool("tavily_search") is False


class TestBuiltinToolExecutorInit:
    """Tests for BuiltinToolExecutor initialization."""

    def test_default_initialization(self):
        """Test default initialization."""
        executor = BuiltinToolExecutor()
        assert executor._memory_manager is None
        assert executor._planner is None

    def test_initialization_with_managers(self):
        """Test initialization with managers."""
        mock_memory = MagicMock()
        mock_planner = MagicMock()

        executor = BuiltinToolExecutor(
            memory_manager=mock_memory,
            planner=mock_planner
        )

        assert executor._memory_manager is mock_memory
        assert executor._planner is mock_planner

    def test_list_available_tools_none(self):
        """Test listing tools with no managers."""
        executor = BuiltinToolExecutor()
        tools = executor.list_available_tools()
        assert tools == []

    def test_list_available_tools_memory_only(self):
        """Test listing tools with memory manager only."""
        executor = BuiltinToolExecutor(memory_manager=MagicMock())
        tools = executor.list_available_tools()

        assert "memory.recall" in tools
        assert "memory.store" in tools
        assert "planning.create_plan" not in tools

    def test_list_available_tools_all(self):
        """Test listing tools with all managers."""
        executor = BuiltinToolExecutor(
            memory_manager=MagicMock(),
            planner=MagicMock()
        )
        tools = executor.list_available_tools()

        assert "memory.recall" in tools
        assert "memory.store" in tools
        assert "planning.create_plan" in tools


class TestArgumentInterpolation:
    """Tests for argument interpolation from context."""

    def test_no_interpolation_needed(self):
        """Test when no interpolation is needed."""
        executor = BuiltinToolExecutor()

        args = {"query": "test query", "k": 5}
        context = {}

        result = executor.interpolate_args(args, context)
        assert result == args

    def test_simple_interpolation(self):
        """Test simple template interpolation."""
        executor = BuiltinToolExecutor()

        # Create mock prior result
        mock_result = MagicMock()
        mock_result.output = "previous step output"

        args = {"query": "{{step_1.output}}"}
        context = {"prior_results": {"step_1": mock_result}}

        result = executor.interpolate_args(args, context)
        assert result["query"] == "previous step output"

    def test_partial_interpolation(self):
        """Test partial template interpolation."""
        executor = BuiltinToolExecutor()

        mock_result = MagicMock()
        mock_result.output = "AI"

        args = {"query": "Latest news about {{step_1.output}}"}
        context = {"prior_results": {"step_1": mock_result}}

        result = executor.interpolate_args(args, context)
        assert result["query"] == "Latest news about AI"

    def test_nested_interpolation(self):
        """Test nested attribute interpolation."""
        executor = BuiltinToolExecutor()

        mock_result = MagicMock()
        mock_result.output = {"email": "test@example.com", "name": "John"}

        args = {"to": "{{step_1.output.email}}"}
        context = {"prior_results": {"step_1": mock_result}}

        result = executor.interpolate_args(args, context)
        assert result["to"] == "test@example.com"

    def test_missing_prior_result(self):
        """Test interpolation with missing prior result."""
        executor = BuiltinToolExecutor()

        args = {"query": "{{nonexistent.output}}"}
        context = {"prior_results": {}}

        # Should return unchanged when reference not found
        result = executor.interpolate_args(args, context)
        assert result["query"] == "{{nonexistent.output}}"


class TestMemoryToolExecution:
    """Tests for memory tool execution."""

    @pytest.mark.asyncio
    async def test_execute_recall(self):
        """Test memory.recall execution."""
        mock_memory = MagicMock()
        mock_memory.recall = AsyncMock(return_value=[
            {"item_id": "1", "content": "Fact 1"},
            {"item_id": "2", "content": "Fact 2"},
        ])

        executor = BuiltinToolExecutor(memory_manager=mock_memory)

        result = await executor.execute(
            tool_name="memory.recall",
            tool_args={"query": "user preferences", "k": 5}
        )

        result_data = json.loads(result)
        assert result_data["status"] == "success"
        assert result_data["count"] == 2
        mock_memory.recall.assert_called_once_with(
            query="user preferences",
            k=5,
            category=None,
            mode="hybrid"
        )

    @pytest.mark.asyncio
    async def test_execute_retrieve_alias(self):
        """Test memory.retrieve (alias for recall) execution."""
        mock_memory = MagicMock()
        mock_memory.recall = AsyncMock(return_value=[])

        executor = BuiltinToolExecutor(memory_manager=mock_memory)

        result = await executor.execute(
            tool_name="memory.retrieve",
            tool_args={"query": "test"}
        )

        result_data = json.loads(result)
        assert result_data["status"] == "success"

    @pytest.mark.asyncio
    async def test_execute_store(self):
        """Test memory.store execution."""
        mock_memory = MagicMock()
        mock_item = MagicMock()
        mock_item.item_id = "item-123"
        mock_memory.remember = AsyncMock(return_value=mock_item)

        executor = BuiltinToolExecutor(memory_manager=mock_memory)

        result = await executor.execute(
            tool_name="memory.store",
            tool_args={"content": "Important fact to remember", "category": "preferences"}
        )

        result_data = json.loads(result)
        assert result_data["status"] == "success"
        assert result_data["item_id"] == "item-123"

    @pytest.mark.asyncio
    async def test_execute_remember_alias(self):
        """Test memory.remember (alias for store) execution."""
        mock_memory = MagicMock()
        mock_item = MagicMock()
        mock_item.item_id = "item-456"
        mock_memory.remember = AsyncMock(return_value=mock_item)

        executor = BuiltinToolExecutor(memory_manager=mock_memory)

        result = await executor.execute(
            tool_name="memory.remember",
            tool_args={"content": "Test fact"}
        )

        result_data = json.loads(result)
        assert result_data["status"] == "success"

    @pytest.mark.asyncio
    async def test_execute_list_categories(self):
        """Test memory.list_categories execution."""
        mock_memory = MagicMock()
        mock_memory.list_categories = AsyncMock(return_value=[
            {"name": "preferences", "item_count": 10},
            {"name": "history", "item_count": 5},
        ])

        executor = BuiltinToolExecutor(memory_manager=mock_memory)

        result = await executor.execute(
            tool_name="memory.list_categories",
            tool_args={}
        )

        result_data = json.loads(result)
        assert result_data["status"] == "success"
        assert result_data["count"] == 2

    @pytest.mark.asyncio
    async def test_execute_get_category(self):
        """Test memory.get_category execution."""
        mock_memory = MagicMock()
        mock_memory.get_category_content = AsyncMock(return_value="# Preferences\n- Item 1\n- Item 2")

        executor = BuiltinToolExecutor(memory_manager=mock_memory)

        result = await executor.execute(
            tool_name="memory.get_category",
            tool_args={"name": "preferences"}
        )

        result_data = json.loads(result)
        assert result_data["status"] == "success"
        assert "Preferences" in result_data["content"]

    @pytest.mark.asyncio
    async def test_execute_get_category_not_found(self):
        """Test memory.get_category with non-existent category."""
        mock_memory = MagicMock()
        mock_memory.get_category_content = AsyncMock(return_value=None)

        executor = BuiltinToolExecutor(memory_manager=mock_memory)

        result = await executor.execute(
            tool_name="memory.get_category",
            tool_args={"name": "nonexistent"}
        )

        result_data = json.loads(result)
        assert result_data["status"] == "not_found"


class TestPlanningToolExecution:
    """Tests for planning tool execution."""

    @pytest.mark.asyncio
    async def test_execute_create_plan(self):
        """Test planning.create_plan execution."""
        mock_planner = MagicMock()
        mock_plan = MagicMock()
        mock_plan.plan_id = "plan-123"
        mock_plan.goal = "Test goal"
        mock_plan.steps = [
            MagicMock(step_id="s1", description="Step 1"),
            MagicMock(step_id="s2", description="Step 2"),
        ]
        mock_planner.create_plan = AsyncMock(return_value=mock_plan)

        executor = BuiltinToolExecutor(planner=mock_planner)

        result = await executor.execute(
            tool_name="planning.create_plan",
            tool_args={"goal": "Test goal", "max_steps": 5}
        )

        result_data = json.loads(result)
        assert result_data["status"] == "success"
        assert result_data["plan_id"] == "plan-123"
        assert result_data["step_count"] == 2

    @pytest.mark.asyncio
    async def test_execute_create_plan_with_json_context(self):
        """Test planning.create_plan with JSON context string."""
        mock_planner = MagicMock()
        mock_plan = MagicMock()
        mock_plan.plan_id = "plan-456"
        mock_plan.goal = "Goal"
        mock_plan.steps = []
        mock_planner.create_plan = AsyncMock(return_value=mock_plan)

        executor = BuiltinToolExecutor(planner=mock_planner)

        result = await executor.execute(
            tool_name="planning.create_plan",
            tool_args={
                "goal": "Goal",
                "context": '{"industry": "tech"}'
            }
        )

        result_data = json.loads(result)
        assert result_data["status"] == "success"

        # Verify context was parsed
        call_args = mock_planner.create_plan.call_args
        assert call_args.kwargs["context"] == {"industry": "tech"}

    @pytest.mark.asyncio
    async def test_execute_get_status(self):
        """Test planning.get_status execution."""
        mock_planner = MagicMock()
        executor = BuiltinToolExecutor(planner=mock_planner)

        result = await executor.execute(
            tool_name="planning.get_status",
            tool_args={"plan_id": "plan-123"}
        )

        result_data = json.loads(result)
        assert result_data["status"] == "success"
        assert result_data["plan_id"] == "plan-123"


class TestErrorHandling:
    """Tests for error handling."""

    @pytest.mark.asyncio
    async def test_missing_memory_manager(self):
        """Test error when memory manager not provided."""
        executor = BuiltinToolExecutor()

        with pytest.raises(MissingDependencyError) as exc_info:
            await executor.execute("memory.recall", {"query": "test"})

        assert "memory_manager" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_missing_planner(self):
        """Test error when planner not provided."""
        executor = BuiltinToolExecutor()

        with pytest.raises(MissingDependencyError) as exc_info:
            await executor.execute("planning.create_plan", {"goal": "test"})

        assert "planner" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_unknown_tool(self):
        """Test error for unknown tool."""
        executor = BuiltinToolExecutor(
            memory_manager=MagicMock(),
            planner=MagicMock()
        )

        with pytest.raises(ToolNotFoundError) as exc_info:
            await executor.execute("other.tool", {})

        assert "not found" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_unknown_memory_operation(self):
        """Test error for unknown memory operation."""
        mock_memory = MagicMock()
        executor = BuiltinToolExecutor(memory_manager=mock_memory)

        with pytest.raises(ToolNotFoundError) as exc_info:
            await executor.execute("memory.unknown_operation", {})

        assert "memory.unknown_operation" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_tool_execution_error(self):
        """Test handling of tool execution errors."""
        mock_memory = MagicMock()
        mock_memory.recall = AsyncMock(side_effect=Exception("Database connection failed"))

        executor = BuiltinToolExecutor(memory_manager=mock_memory)

        with pytest.raises(ToolExecutionError) as exc_info:
            await executor.execute("memory.recall", {"query": "test"})

        assert "Database connection failed" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_invalid_tool_name_format(self):
        """Test error for invalid tool name format."""
        executor = BuiltinToolExecutor(memory_manager=MagicMock())

        with pytest.raises(BuiltinExecutorError) as exc_info:
            await executor.execute("invalid_name", {})

        assert "Invalid tool name format" in str(exc_info.value)


class TestContextHandling:
    """Tests for context handling during execution."""

    @pytest.mark.asyncio
    async def test_context_interpolation_in_recall(self):
        """Test context interpolation in memory.recall."""
        mock_memory = MagicMock()
        mock_memory.recall = AsyncMock(return_value=[])

        mock_result = MagicMock()
        mock_result.output = "AI trends"

        executor = BuiltinToolExecutor(memory_manager=mock_memory)

        await executor.execute(
            tool_name="memory.recall",
            tool_args={"query": "{{step_1.output}}"},
            context={"prior_results": {"step_1": mock_result}}
        )

        # Verify interpolated query was used
        mock_memory.recall.assert_called_once()
        call_args = mock_memory.recall.call_args
        assert call_args.kwargs["query"] == "AI trends"
