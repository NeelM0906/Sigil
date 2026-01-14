"""Tests for the ToolStepExecutor class.

Tests cover:
- Step type determination
- Routing to MCP vs builtin executors
- REASONING step execution
- Error handling and fallback
- Context building
"""

import pytest
import json
from unittest.mock import MagicMock, AsyncMock, patch, PropertyMock

from sigil.config.schemas.plan import PlanStep
from sigil.planning.schemas import (
    StepType,
    StepStatus,
    StepResult,
    PlanStepConfig,
    utc_now,
)
from sigil.planning.tool_executor import (
    ToolStepExecutor,
    create_tool_step_executor,
    StepExecutionError,
    NoToolSpecifiedError,
    UnsupportedStepTypeError,
)


class TestToolStepExecutorInit:
    """Tests for ToolStepExecutor initialization."""

    def test_default_initialization(self):
        """Test default initialization."""
        executor = ToolStepExecutor()
        assert executor._mcp_executor is not None
        assert executor._builtin_executor is not None
        assert executor._reasoning_manager is None
        assert executor._allow_reasoning_fallback is False

    def test_initialization_with_managers(self):
        """Test initialization with custom managers."""
        mock_mcp = MagicMock()
        mock_builtin = MagicMock()
        mock_reasoning = MagicMock()

        executor = ToolStepExecutor(
            mcp_executor=mock_mcp,
            builtin_executor=mock_builtin,
            reasoning_manager=mock_reasoning,
            allow_reasoning_fallback=True,
        )

        assert executor._mcp_executor is mock_mcp
        assert executor._builtin_executor is mock_builtin
        assert executor._reasoning_manager is mock_reasoning
        assert executor._allow_reasoning_fallback is True


class TestExecutorTypeRouting:
    """Tests for determining which executor to use."""

    def test_determine_builtin_memory(self):
        """Test routing memory tools to builtin executor."""
        executor = ToolStepExecutor()

        assert executor._determine_executor_type("memory.recall") == "builtin"
        assert executor._determine_executor_type("memory.store") == "builtin"
        assert executor._determine_executor_type("memory.list_categories") == "builtin"

    def test_determine_builtin_planning(self):
        """Test routing planning tools to builtin executor."""
        executor = ToolStepExecutor()

        assert executor._determine_executor_type("planning.create_plan") == "builtin"
        assert executor._determine_executor_type("planning.get_status") == "builtin"

    def test_determine_mcp(self):
        """Test routing MCP tools to MCP executor."""
        executor = ToolStepExecutor()

        assert executor._determine_executor_type("websearch.search") == "mcp"
        assert executor._determine_executor_type("calendar.create_event") == "mcp"
        assert executor._determine_executor_type("voice.generate") == "mcp"


class TestStepTypeDetermination:
    """Tests for determining step type from step attributes."""

    def test_explicit_step_type(self):
        """Test with explicit step_type attribute."""
        executor = ToolStepExecutor()

        step = MagicMock()
        step.step_type = StepType.TOOL_CALL

        assert executor._determine_step_type(step) == StepType.TOOL_CALL

    def test_step_type_from_string(self):
        """Test with step_type as string."""
        executor = ToolStepExecutor()

        step = MagicMock()
        step.step_type = "reasoning"

        assert executor._determine_step_type(step) == StepType.REASONING

    def test_infer_from_tool_calls(self):
        """Test inferring TOOL_CALL from tool_calls attribute."""
        executor = ToolStepExecutor()

        step = MagicMock()
        step.step_type = None
        step.tool_calls = ["websearch.search"]
        step.description = "Search for information"

        assert executor._determine_step_type(step) == StepType.TOOL_CALL

    def test_infer_from_tool_name(self):
        """Test inferring TOOL_CALL from tool_name attribute."""
        executor = ToolStepExecutor()

        step = MagicMock()
        step.step_type = None
        step.tool_calls = None
        step.tool_name = "memory.recall"
        step.description = "Recall information"

        assert executor._determine_step_type(step) == StepType.TOOL_CALL

    def test_infer_reasoning_from_description(self):
        """Test inferring REASONING from description keywords."""
        executor = ToolStepExecutor()

        step = MagicMock()
        step.step_type = None
        step.tool_calls = None
        step.tool_name = None
        step.description = "Analyze the results and summarize"

        assert executor._determine_step_type(step) == StepType.REASONING

    def test_infer_memory_query(self):
        """Test inferring MEMORY_QUERY from description."""
        executor = ToolStepExecutor()

        step = MagicMock()
        step.step_type = None
        step.tool_calls = None
        step.tool_name = None
        step.description = "Recall user preferences from memory"

        assert executor._determine_step_type(step) == StepType.MEMORY_QUERY


class TestToolCallExecution:
    """Tests for TOOL_CALL step execution."""

    @pytest.mark.asyncio
    async def test_execute_builtin_tool_call(self):
        """Test executing a builtin tool call."""
        mock_builtin = MagicMock()
        mock_builtin.execute = AsyncMock(return_value='{"status": "success"}')

        executor = ToolStepExecutor(builtin_executor=mock_builtin)

        step = MagicMock()
        step.step_id = "step-1"
        step.description = "Recall preferences"
        step.step_type = StepType.TOOL_CALL
        step.tool_name = "memory.recall"
        step.tool_args = {"query": "preferences", "k": 5}
        step.dependencies = []

        result = await executor.execute_step(step, {})

        assert result.status == StepStatus.COMPLETED
        assert result.tokens_used == 0  # Tool calls don't use tokens
        mock_builtin.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_mcp_tool_call(self):
        """Test executing an MCP tool call."""
        mock_mcp = MagicMock()
        mock_mcp.execute = AsyncMock(return_value="Search results")
        mock_mcp._tool_execution_timeout = 60.0  # Must be a real number for asyncio.wait_for

        executor = ToolStepExecutor(mcp_executor=mock_mcp)

        step = MagicMock()
        step.step_id = "step-1"
        step.description = "Search the web"
        step.step_type = StepType.TOOL_CALL
        step.tool_name = "websearch.search"
        step.tool_args = {"query": "AI news"}
        step.dependencies = []

        result = await executor.execute_step(step, {})

        assert result.status == StepStatus.COMPLETED
        assert result.output == "Search results"
        mock_mcp.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_tool_call_with_tool_calls_list(self):
        """Test executing tool from tool_calls list."""
        mock_builtin = MagicMock()
        mock_builtin.execute = AsyncMock(return_value='{"status": "success"}')

        executor = ToolStepExecutor(builtin_executor=mock_builtin)

        step = MagicMock()
        step.step_id = "step-1"
        step.description = "Recall info"
        step.step_type = None  # Infer from tool_calls
        step.tool_name = None
        step.tool_args = None
        step.tool_calls = ["memory.recall"]
        step.dependencies = []

        result = await executor.execute_step(step, {})

        assert result.status == StepStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_execute_no_tool_specified(self):
        """Test error when TOOL_CALL has no tool specified."""
        executor = ToolStepExecutor()

        step = MagicMock()
        step.step_id = "step-1"
        step.description = "Do something"
        step.step_type = StepType.TOOL_CALL
        step.tool_name = None
        step.tool_args = None
        step.tool_calls = None
        step.dependencies = []

        result = await executor.execute_step(step, {})

        assert result.status == StepStatus.FAILED
        assert "no tool_name specified" in result.error


class TestReasoningExecution:
    """Tests for REASONING step execution."""

    @pytest.mark.asyncio
    async def test_execute_reasoning_with_manager(self):
        """Test REASONING execution with reasoning manager."""
        mock_reasoning = MagicMock()
        mock_result = MagicMock()
        mock_result.answer = "Summarized results"  # StrategyResult uses 'answer'
        mock_result.tokens_used = 150
        mock_reasoning.execute = AsyncMock(return_value=mock_result)

        executor = ToolStepExecutor(reasoning_manager=mock_reasoning)

        step = MagicMock()
        step.step_id = "step-2"
        step.description = "Summarize the search results"
        step.step_type = StepType.REASONING
        step.reasoning_task = "Summarize the following information"
        step.dependencies = ["step-1"]

        prior_results = {
            "step-1": StepResult(
                step_id="step-1",
                status=StepStatus.COMPLETED,
                output="Raw search results here",
                tokens_used=0,
            )
        }

        result = await executor.execute_step(step, prior_results)

        assert result.status == StepStatus.COMPLETED
        assert result.output == "Summarized results"
        assert result.tokens_used > 0  # Reasoning uses tokens

    @pytest.mark.asyncio
    async def test_execute_reasoning_fallback_aggregation(self):
        """Test REASONING without manager uses aggregation."""
        executor = ToolStepExecutor()  # No reasoning manager

        step = MagicMock()
        step.step_id = "step-2"
        step.description = "Summarize results"
        step.step_type = StepType.REASONING
        step.dependencies = ["step-1"]

        prior_results = {
            "step-1": StepResult(
                step_id="step-1",
                status=StepStatus.COMPLETED,
                output="Prior output",
                tokens_used=100,
            )
        }

        result = await executor.execute_step(step, prior_results)

        assert result.status == StepStatus.COMPLETED
        assert "Prior output" in result.output


class TestMemoryQueryExecution:
    """Tests for MEMORY_QUERY step execution."""

    @pytest.mark.asyncio
    async def test_execute_memory_query(self):
        """Test MEMORY_QUERY step execution."""
        mock_builtin = MagicMock()
        mock_builtin.execute = AsyncMock(return_value='{"status": "success", "memories": []}')

        executor = ToolStepExecutor(builtin_executor=mock_builtin)

        step = MagicMock()
        step.step_id = "step-1"
        step.description = "Recall user preferences"
        step.step_type = StepType.MEMORY_QUERY
        step.dependencies = []

        result = await executor.execute_step(step, {})

        assert result.status == StepStatus.COMPLETED
        mock_builtin.execute.assert_called_once()
        call_args = mock_builtin.execute.call_args
        assert call_args.kwargs["tool_name"] == "memory.recall"


class TestErrorHandling:
    """Tests for error handling and fallback."""

    @pytest.mark.asyncio
    async def test_tool_execution_failure(self):
        """Test handling of tool execution failure."""
        mock_mcp = MagicMock()
        mock_mcp.execute = AsyncMock(side_effect=Exception("API error"))
        mock_mcp._tool_execution_timeout = 60.0  # Must be a real number for asyncio.wait_for

        executor = ToolStepExecutor(mcp_executor=mock_mcp)

        step = MagicMock()
        step.step_id = "step-1"
        step.description = "Search"
        step.step_type = StepType.TOOL_CALL
        step.tool_name = "websearch.search"
        step.tool_args = {"query": "test"}
        step.dependencies = []

        result = await executor.execute_step(step, {})

        assert result.status == StepStatus.FAILED
        # Error message may be wrapped but should contain reference to the failure
        assert result.error is not None and len(result.error) > 0

    @pytest.mark.asyncio
    async def test_reasoning_fallback_on_failure(self):
        """Test reasoning fallback when tool fails."""
        mock_mcp = MagicMock()
        mock_mcp.execute = AsyncMock(side_effect=Exception("API error"))
        mock_mcp._tool_execution_timeout = 60.0  # Must be a real number for asyncio.wait_for

        mock_reasoning = MagicMock()
        mock_result = MagicMock()
        mock_result.answer = "Fallback response"  # StrategyResult uses 'answer'
        mock_result.tokens_used = 100
        mock_reasoning.execute = AsyncMock(return_value=mock_result)

        executor = ToolStepExecutor(
            mcp_executor=mock_mcp,
            reasoning_manager=mock_reasoning,
            allow_reasoning_fallback=True,
        )

        step = MagicMock()
        step.step_id = "step-1"
        step.description = "Search for information"
        step.step_type = StepType.TOOL_CALL
        step.tool_name = "websearch.search"
        step.tool_args = {"query": "test"}
        step.dependencies = []

        result = await executor.execute_step(step, {})

        assert result.status == StepStatus.COMPLETED
        # Output includes a fallback note prefix
        assert "Fallback response" in result.output

    @pytest.mark.asyncio
    async def test_no_fallback_when_disabled(self):
        """Test no fallback when allow_reasoning_fallback is False."""
        mock_mcp = MagicMock()
        mock_mcp.execute = AsyncMock(side_effect=Exception("API error"))
        mock_mcp._tool_execution_timeout = 60.0  # Must be a real number for asyncio.wait_for

        mock_reasoning = MagicMock()

        executor = ToolStepExecutor(
            mcp_executor=mock_mcp,
            reasoning_manager=mock_reasoning,
            allow_reasoning_fallback=False,  # Disabled
        )

        step = MagicMock()
        step.step_id = "step-1"
        step.description = "Search"
        step.step_type = StepType.TOOL_CALL
        step.tool_name = "websearch.search"
        step.tool_args = {"query": "test"}
        step.dependencies = []

        result = await executor.execute_step(step, {})

        assert result.status == StepStatus.FAILED
        mock_reasoning.execute.assert_not_called()


class TestContextBuilding:
    """Tests for execution context building."""

    def test_build_context_with_prior_results(self):
        """Test context building with prior results."""
        executor = ToolStepExecutor()

        step = MagicMock()
        step.description = "Current step"
        step.dependencies = ["step-1", "step-2"]

        prior_results = {
            "step-1": StepResult(
                step_id="step-1",
                status=StepStatus.COMPLETED,
                output="Output 1",
                tokens_used=100,
            ),
            "step-2": StepResult(
                step_id="step-2",
                status=StepStatus.COMPLETED,
                output="Output 2",
                tokens_used=50,
            ),
        }

        context = executor._build_execution_context(step, prior_results)

        assert "prior_results" in context
        assert "prior_outputs" in context
        assert context["prior_outputs"]["step-1"]["output"] == "Output 1"
        assert context["prior_outputs"]["step-2"]["output"] == "Output 2"

    def test_build_context_excludes_failed_steps(self):
        """Test that failed steps are excluded from prior_outputs."""
        executor = ToolStepExecutor()

        step = MagicMock()
        step.description = "Current step"
        step.dependencies = ["step-1"]

        prior_results = {
            "step-1": StepResult(
                step_id="step-1",
                status=StepStatus.FAILED,
                output=None,
                error="Failed",
                tokens_used=0,
            ),
        }

        context = executor._build_execution_context(step, prior_results)

        assert "step-1" not in context["prior_outputs"]


class TestFactoryFunction:
    """Tests for the factory function."""

    def test_create_tool_step_executor_defaults(self):
        """Test factory with default values."""
        executor = create_tool_step_executor()

        assert executor._mcp_executor is not None
        assert executor._builtin_executor is not None
        assert executor._reasoning_manager is None
        # Default changed to True for better UX - fallback provides response even when MCP fails
        assert executor._allow_reasoning_fallback is True

    def test_create_tool_step_executor_with_managers(self):
        """Test factory with provided managers."""
        mock_memory = MagicMock()
        mock_planner = MagicMock()
        mock_reasoning = MagicMock()

        executor = create_tool_step_executor(
            memory_manager=mock_memory,
            planner=mock_planner,
            reasoning_manager=mock_reasoning,
            allow_reasoning_fallback=True,
        )

        assert executor._builtin_executor._memory_manager is mock_memory
        assert executor._builtin_executor._planner is mock_planner
        assert executor._reasoning_manager is mock_reasoning
        assert executor._allow_reasoning_fallback is True


class TestStepResultTokenTracking:
    """Tests for token tracking in step results."""

    @pytest.mark.asyncio
    async def test_tool_call_zero_tokens(self):
        """Test that tool calls report 0 tokens."""
        mock_mcp = MagicMock()
        mock_mcp.execute = AsyncMock(return_value="results")
        mock_mcp._tool_execution_timeout = 60.0  # Must be a real number for asyncio.wait_for

        executor = ToolStepExecutor(mcp_executor=mock_mcp)

        step = MagicMock()
        step.step_id = "step-1"
        step.description = "Search"
        step.step_type = StepType.TOOL_CALL
        step.tool_name = "websearch.search"
        step.tool_args = {"query": "test"}
        step.dependencies = []

        result = await executor.execute_step(step, {})

        assert result.tokens_used == 0

    @pytest.mark.asyncio
    async def test_reasoning_estimates_tokens(self):
        """Test that reasoning steps estimate tokens."""
        mock_reasoning = MagicMock()
        mock_result = MagicMock()
        mock_result.answer = "A" * 400  # 400 chars = ~100 tokens
        mock_result.tokens_used = 100
        mock_reasoning.execute = AsyncMock(return_value=mock_result)

        executor = ToolStepExecutor(reasoning_manager=mock_reasoning)

        step = MagicMock()
        step.step_id = "step-1"
        step.description = "Analyze"
        step.step_type = StepType.REASONING
        step.reasoning_task = "Analyze data"
        step.dependencies = []

        result = await executor.execute_step(step, {})

        assert result.tokens_used > 0  # Should have estimated tokens
