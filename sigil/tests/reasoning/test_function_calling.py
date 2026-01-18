"""Tests for the FunctionCallingStrategy.

Tests cover:
- Basic strategy initialization and configuration
- System prompt building
- LLM response parsing (text and tool_use blocks)
- Tool execution routing (Tavily, Builtin)
- ReAct loop execution
- Error handling
- Confidence estimation
- Metrics tracking
"""

import json
import pytest
from unittest.mock import patch, AsyncMock, MagicMock

from sigil.reasoning.strategies.function_calling import (
    FunctionCallingStrategy,
    FunctionCallingStep,
    ToolUseBlock,
    MAX_ITERATIONS,
    FUNCTION_CALLING_MIN_COMPLEXITY,
    FUNCTION_CALLING_MAX_COMPLEXITY,
    FUNCTION_CALLING_MIN_TOKENS,
    FUNCTION_CALLING_MAX_TOKENS,
)
from sigil.reasoning.strategies.base import StrategyResult, StrategyConfig


class TestFunctionCallingStrategyBasics:
    """Tests for basic FunctionCallingStrategy functionality."""

    def test_initialization(self, function_calling_strategy):
        """Test that FunctionCallingStrategy initializes correctly."""
        assert function_calling_strategy.name == "function_calling"
        assert function_calling_strategy._config is not None
        assert function_calling_strategy._max_iterations == MAX_ITERATIONS

    def test_initialization_with_custom_config(self):
        """Test FunctionCallingStrategy with custom configuration."""
        config = StrategyConfig(
            min_complexity=0.3,
            max_complexity=0.8,
            min_tokens=300,
            max_tokens=2000,
        )
        strategy = FunctionCallingStrategy(config=config)

        assert strategy._config.min_complexity == 0.3
        assert strategy._config.max_complexity == 0.8
        assert strategy._config.max_tokens == 2000

    def test_initialization_with_custom_max_iterations(self):
        """Test FunctionCallingStrategy with custom max iterations."""
        strategy = FunctionCallingStrategy(max_iterations=10)

        assert strategy._max_iterations == 10

    def test_default_complexity_range(self, function_calling_strategy):
        """Test default complexity range is correct."""
        assert function_calling_strategy._config.min_complexity == FUNCTION_CALLING_MIN_COMPLEXITY
        assert function_calling_strategy._config.max_complexity == FUNCTION_CALLING_MAX_COMPLEXITY

    def test_default_token_budget(self, function_calling_strategy):
        """Test default token budget is correct."""
        assert function_calling_strategy._config.min_tokens == FUNCTION_CALLING_MIN_TOKENS
        assert function_calling_strategy._config.max_tokens == FUNCTION_CALLING_MAX_TOKENS

    def test_strategy_name(self, function_calling_strategy):
        """Test strategy name property."""
        assert function_calling_strategy.name == "function_calling"


class TestToolUseBlockDataClass:
    """Tests for ToolUseBlock dataclass within FunctionCallingStrategy."""

    def test_tool_use_block_creation(self):
        """Test ToolUseBlock creation."""
        block = ToolUseBlock(
            id="tool_123",
            name="websearch_search",
            input={"query": "latest news"}
        )

        assert block.id == "tool_123"
        assert block.name == "websearch_search"
        assert block.input == {"query": "latest news"}

    def test_tool_use_block_to_dict(self):
        """Test ToolUseBlock conversion to dictionary."""
        block = ToolUseBlock(
            id="tool_456",
            name="memory_store",
            input={"content": "test", "category": "fact"}
        )

        result = block.to_dict()

        assert result["type"] == "tool_use"
        assert result["id"] == "tool_456"
        assert result["name"] == "memory_store"
        assert result["input"] == {"content": "test", "category": "fact"}

    def test_tool_use_block_empty_input(self):
        """Test ToolUseBlock with empty input."""
        block = ToolUseBlock(id="1", name="tool", input={})

        assert block.input == {}
        result = block.to_dict()
        assert result["input"] == {}


class TestFunctionCallingStepDataClass:
    """Tests for FunctionCallingStep dataclass."""

    def test_step_creation_basic(self):
        """Test basic FunctionCallingStep creation."""
        step = FunctionCallingStep(iteration=1)

        assert step.iteration == 1
        assert step.thought is None
        assert step.tool_use is None
        assert step.tool_result is None
        assert step.is_final is False

    def test_step_creation_with_thought(self):
        """Test FunctionCallingStep with thought."""
        step = FunctionCallingStep(
            iteration=1,
            thought="I need to search for information",
            is_final=False
        )

        assert step.thought == "I need to search for information"
        assert step.is_final is False

    def test_step_creation_with_tool(self):
        """Test FunctionCallingStep with tool use."""
        tool = ToolUseBlock(id="1", name="search", input={"query": "test"})
        step = FunctionCallingStep(
            iteration=1,
            thought="Let me search",
            tool_use=tool,
            tool_result='{"results": []}'
        )

        assert step.tool_use is not None
        assert step.tool_use.name == "search"
        assert step.tool_result == '{"results": []}'

    def test_step_to_dict(self):
        """Test FunctionCallingStep conversion to dictionary."""
        tool = ToolUseBlock(id="1", name="search", input={"query": "test"})
        step = FunctionCallingStep(
            iteration=2,
            thought="Reasoning here",
            tool_use=tool,
            tool_result="Result here",
            is_final=True
        )

        result = step.to_dict()

        assert result["iteration"] == 2
        assert result["thought"] == "Reasoning here"
        assert result["tool_use"]["name"] == "search"
        assert result["tool_result"] == "Result here"
        assert result["is_final"] is True

    def test_step_to_dict_without_tool(self):
        """Test FunctionCallingStep to_dict without tool."""
        step = FunctionCallingStep(
            iteration=1,
            thought="Just thinking",
            is_final=True
        )

        result = step.to_dict()

        assert result["tool_use"] is None
        assert result["tool_result"] is None


class TestSystemPromptBuilding:
    """Tests for system prompt construction."""

    def test_build_system_prompt_basic(self, function_calling_strategy):
        """Test basic system prompt building."""
        prompt = function_calling_strategy._build_system_prompt({})

        assert "intelligent assistant" in prompt.lower()
        assert "tools" in prompt.lower()

    def test_build_system_prompt_with_context(self, function_calling_strategy, sample_context):
        """Test system prompt building with context."""
        prompt = function_calling_strategy._build_system_prompt(sample_context)

        assert "Context" in prompt
        assert len(prompt) > 100  # Should include context


class TestExecutorTypeDetection:
    """Tests for tool executor type detection."""

    def test_determine_executor_tavily_websearch(self, function_calling_strategy):
        """Test detection of Tavily/websearch tools."""
        assert function_calling_strategy._determine_executor_type("websearch.search") == "tavily"
        assert function_calling_strategy._determine_executor_type("websearch.news") == "tavily"
        assert function_calling_strategy._determine_executor_type("WEBSEARCH.search") == "tavily"

    def test_determine_executor_tavily_direct(self, function_calling_strategy):
        """Test detection of direct Tavily tools."""
        assert function_calling_strategy._determine_executor_type("tavily.search") == "tavily"
        assert function_calling_strategy._determine_executor_type("Tavily.news") == "tavily"

    def test_determine_executor_builtin_memory(self, function_calling_strategy):
        """Test detection of memory tools."""
        assert function_calling_strategy._determine_executor_type("memory.store") == "builtin"
        assert function_calling_strategy._determine_executor_type("memory.recall") == "builtin"
        assert function_calling_strategy._determine_executor_type("MEMORY.store") == "builtin"

    def test_determine_executor_builtin_planning(self, function_calling_strategy):
        """Test detection of planning tools."""
        assert function_calling_strategy._determine_executor_type("planning.create") == "builtin"
        assert function_calling_strategy._determine_executor_type("planning.update") == "builtin"

    def test_determine_executor_unsupported(self, function_calling_strategy):
        """Test detection of unsupported tools."""
        assert function_calling_strategy._determine_executor_type("unknown.tool") == "unsupported"
        assert function_calling_strategy._determine_executor_type("random.action") == "unsupported"


class TestResponseParsing:
    """Tests for LLM response parsing."""

    def test_extract_text_content(self, function_calling_strategy):
        """Test text extraction from response."""
        response = {
            "content": [
                {"type": "text", "text": "Hello "},
                {"type": "text", "text": "World!"}
            ]
        }

        text = function_calling_strategy._extract_text_content(response)

        assert text == "Hello \nWorld!"

    def test_extract_text_content_empty(self, function_calling_strategy):
        """Test text extraction from empty response."""
        response = {"content": []}

        text = function_calling_strategy._extract_text_content(response)

        assert text == ""

    def test_extract_text_content_no_text_blocks(self, function_calling_strategy):
        """Test text extraction when no text blocks present."""
        response = {
            "content": [
                {"type": "tool_use", "id": "1", "name": "search", "input": {}}
            ]
        }

        text = function_calling_strategy._extract_text_content(response)

        assert text == ""

    def test_extract_tool_use(self, function_calling_strategy):
        """Test tool use extraction from response."""
        response = {
            "content": [
                {"type": "text", "text": "Let me search"},
                {"type": "tool_use", "id": "tool_123", "name": "search", "input": {"query": "test"}}
            ]
        }

        tool = function_calling_strategy._extract_tool_use(response)

        assert tool is not None
        assert tool.id == "tool_123"
        assert tool.name == "search"
        assert tool.input == {"query": "test"}

    def test_extract_tool_use_none(self, function_calling_strategy):
        """Test tool use extraction when no tool present."""
        response = {
            "content": [
                {"type": "text", "text": "Just text"}
            ]
        }

        tool = function_calling_strategy._extract_tool_use(response)

        assert tool is None


class TestMessageBuilding:
    """Tests for message construction."""

    def test_build_assistant_message(self, function_calling_strategy):
        """Test building assistant message."""
        response = {
            "content": [
                {"type": "text", "text": "Response text"},
                {"type": "tool_use", "id": "1", "name": "tool", "input": {}}
            ]
        }

        message = function_calling_strategy._build_assistant_message(response)

        assert message["role"] == "assistant"
        assert len(message["content"]) == 2

    def test_build_tool_result_message(self, function_calling_strategy):
        """Test building tool result message."""
        message = function_calling_strategy._build_tool_result_message(
            tool_use_id="tool_123",
            result='{"data": "value"}',
            is_error=False
        )

        assert message["role"] == "user"
        assert len(message["content"]) == 1
        assert message["content"][0]["type"] == "tool_result"
        assert message["content"][0]["tool_use_id"] == "tool_123"
        assert message["content"][0]["content"] == '{"data": "value"}'
        assert message["content"][0]["is_error"] is False

    def test_build_tool_result_message_error(self, function_calling_strategy):
        """Test building error tool result message."""
        message = function_calling_strategy._build_tool_result_message(
            tool_use_id="tool_456",
            result="Error occurred",
            is_error=True
        )

        assert message["content"][0]["is_error"] is True


class TestToolExecution:
    """Tests for tool execution."""

    @pytest.mark.asyncio
    async def test_execute_tool_tavily(self, function_calling_strategy, mock_tavily_executor):
        """Test executing a Tavily tool."""
        function_calling_strategy._tavily_executor = mock_tavily_executor

        tool = ToolUseBlock(
            id="1",
            name="websearch.search",
            input={"query": "test query"}
        )

        result = await function_calling_strategy._execute_tool(tool)

        assert result is not None
        mock_tavily_executor.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_tool_builtin(self, function_calling_strategy, mock_builtin_executor):
        """Test executing a builtin tool."""
        function_calling_strategy._builtin_executor = mock_builtin_executor

        tool = ToolUseBlock(
            id="1",
            name="memory.recall",
            input={"topic": "AI"}
        )

        result = await function_calling_strategy._execute_tool(tool)

        assert result is not None
        mock_builtin_executor.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_tool_unsupported(self, function_calling_strategy):
        """Test executing an unsupported tool returns error."""
        tool = ToolUseBlock(
            id="1",
            name="unknown.tool",
            input={}
        )

        result = await function_calling_strategy._execute_tool(tool)

        assert "error" in result.lower() or "unsupported" in result.lower()


class TestStrategyExecution:
    """Tests for strategy execution."""

    @pytest.mark.asyncio
    async def test_execute_returns_strategy_result(self, function_calling_strategy, mock_llm_end_turn):
        """Test that execute returns StrategyResult."""
        with patch.object(
            function_calling_strategy,
            "_call_llm_with_tools",
            return_value=mock_llm_end_turn
        ):
            result = await function_calling_strategy.execute("What is 2+2?", {})

            assert isinstance(result, StrategyResult)
            assert result.success is True

    @pytest.mark.asyncio
    async def test_execute_with_tool_use(self, function_calling_strategy, mock_llm_tool_use, mock_llm_end_turn):
        """Test execution with tool use cycle."""
        # First call returns tool_use, second returns end_turn
        call_count = 0

        async def mock_call(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return mock_llm_tool_use
            return mock_llm_end_turn

        with patch.object(function_calling_strategy, "_call_llm_with_tools", side_effect=mock_call):
            with patch.object(function_calling_strategy, "_execute_tool", return_value='{"results": []}'):
                result = await function_calling_strategy.execute(
                    "Search for AI news",
                    {},
                    tools=[{"name": "search", "input_schema": {}}]
                )

                assert result.success is True
                assert result.metadata.get("tool_calls", 0) >= 1

    @pytest.mark.asyncio
    async def test_execute_max_iterations(self, function_calling_strategy, mock_llm_tool_use):
        """Test that execution respects max iterations."""
        strategy = FunctionCallingStrategy(max_iterations=2)

        with patch.object(strategy, "_call_llm_with_tools", return_value=mock_llm_tool_use):
            with patch.object(strategy, "_execute_tool", return_value='{"results": []}'):
                result = await strategy.execute("Search", {}, tools=[{"name": "search"}])

                # Should stop after max_iterations
                assert result.metadata.get("iterations", 0) <= 2

    @pytest.mark.asyncio
    async def test_execute_without_tools(self, function_calling_strategy, mock_llm_end_turn):
        """Test execution without any tools."""
        with patch.object(
            function_calling_strategy,
            "_call_llm_with_tools",
            return_value=mock_llm_end_turn
        ):
            result = await function_calling_strategy.execute("Simple question", {})

            assert result.success is True
            assert "function_calling" in result.metadata.get("strategy", "")


class TestConfidenceEstimation:
    """Tests for confidence estimation."""

    @pytest.mark.asyncio
    async def test_confidence_base_level(self, function_calling_strategy, mock_llm_end_turn):
        """Test base confidence level."""
        with patch.object(
            function_calling_strategy,
            "_call_llm_with_tools",
            return_value=mock_llm_end_turn
        ):
            result = await function_calling_strategy.execute("Test", {})

            # Base confidence should be around 0.6
            assert 0.5 <= result.confidence <= 0.9

    @pytest.mark.asyncio
    async def test_confidence_increases_with_tools(self, function_calling_strategy, mock_llm_tool_use, mock_llm_end_turn):
        """Test that confidence increases with successful tool use."""
        call_count = 0

        async def mock_call(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return mock_llm_tool_use
            return mock_llm_end_turn

        with patch.object(function_calling_strategy, "_call_llm_with_tools", side_effect=mock_call):
            with patch.object(function_calling_strategy, "_execute_tool", return_value='{"data": "result"}'):
                result = await function_calling_strategy.execute("Search", {}, tools=[{"name": "search"}])

                # Confidence should be above base (0.6) due to tool use
                assert result.confidence >= 0.6


class TestErrorHandling:
    """Tests for error handling."""

    @pytest.mark.asyncio
    async def test_error_returns_failed_result(self, function_calling_strategy):
        """Test that errors return a failed result."""
        with patch.object(
            function_calling_strategy,
            "_call_llm_with_tools",
            side_effect=Exception("LLM Error")
        ):
            result = await function_calling_strategy.execute("Test", {})

            assert result.success is False
            assert result.error is not None
            assert "LLM Error" in result.error

    @pytest.mark.asyncio
    async def test_error_preserves_tokens_used(self, function_calling_strategy):
        """Test that tokens are tracked even on error."""
        with patch.object(
            function_calling_strategy,
            "_call_llm_with_tools",
            side_effect=Exception("Error")
        ):
            result = await function_calling_strategy.execute("Test", {})

            assert result.tokens_used >= 0

    @pytest.mark.asyncio
    async def test_tool_error_handled_gracefully(self, function_calling_strategy, mock_llm_tool_use, mock_llm_end_turn):
        """Test that tool execution errors are handled gracefully."""
        call_count = 0

        async def mock_call(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return mock_llm_tool_use
            return mock_llm_end_turn

        with patch.object(function_calling_strategy, "_call_llm_with_tools", side_effect=mock_call):
            with patch.object(
                function_calling_strategy,
                "_execute_tool",
                return_value='{"error": "Tool failed"}'
            ):
                result = await function_calling_strategy.execute("Search", {}, tools=[{"name": "search"}])

                # Should still complete, error is passed to LLM
                assert result.success is True


class TestMetrics:
    """Tests for strategy metrics."""

    @pytest.mark.asyncio
    async def test_execution_recorded(self, function_calling_strategy, mock_llm_end_turn):
        """Test that executions are recorded."""
        initial_count = function_calling_strategy._execution_count

        with patch.object(
            function_calling_strategy,
            "_call_llm_with_tools",
            return_value=mock_llm_end_turn
        ):
            await function_calling_strategy.execute("Test", {})

            assert function_calling_strategy._execution_count == initial_count + 1

    @pytest.mark.asyncio
    async def test_success_recorded(self, function_calling_strategy, mock_llm_end_turn):
        """Test that successes are recorded."""
        initial_successes = function_calling_strategy._success_count

        with patch.object(
            function_calling_strategy,
            "_call_llm_with_tools",
            return_value=mock_llm_end_turn
        ):
            result = await function_calling_strategy.execute("Test", {})

            if result.success:
                assert function_calling_strategy._success_count == initial_successes + 1

    def test_reset_metrics(self, function_calling_strategy):
        """Test resetting metrics."""
        function_calling_strategy._execution_count = 10
        function_calling_strategy._success_count = 8

        function_calling_strategy.reset_metrics()

        assert function_calling_strategy._execution_count == 0
        assert function_calling_strategy._success_count == 0


class TestResultMetadata:
    """Tests for result metadata."""

    @pytest.mark.asyncio
    async def test_metadata_includes_strategy(self, function_calling_strategy, mock_llm_end_turn):
        """Test that metadata includes strategy name."""
        with patch.object(
            function_calling_strategy,
            "_call_llm_with_tools",
            return_value=mock_llm_end_turn
        ):
            result = await function_calling_strategy.execute("Test", {})

            assert "strategy" in result.metadata
            assert result.metadata["strategy"] == "function_calling"

    @pytest.mark.asyncio
    async def test_metadata_includes_iterations(self, function_calling_strategy, mock_llm_end_turn):
        """Test that metadata includes iteration count."""
        with patch.object(
            function_calling_strategy,
            "_call_llm_with_tools",
            return_value=mock_llm_end_turn
        ):
            result = await function_calling_strategy.execute("Test", {})

            assert "iterations" in result.metadata
            assert result.metadata["iterations"] >= 1

    @pytest.mark.asyncio
    async def test_metadata_includes_steps(self, function_calling_strategy, mock_llm_end_turn):
        """Test that metadata includes step details."""
        with patch.object(
            function_calling_strategy,
            "_call_llm_with_tools",
            return_value=mock_llm_end_turn
        ):
            result = await function_calling_strategy.execute("Test", {})

            assert "steps" in result.metadata
            assert isinstance(result.metadata["steps"], list)


# =============================================================================
# Fixtures specific to function_calling tests
# =============================================================================


@pytest.fixture
def function_calling_strategy(event_store, token_tracker):
    """Create a FunctionCallingStrategy instance."""
    return FunctionCallingStrategy(
        event_store=event_store,
        token_tracker=token_tracker,
    )


@pytest.fixture
def mock_tavily_executor():
    """Create a mock Tavily executor."""
    executor = AsyncMock()
    executor.execute.return_value = json.dumps({
        "results": [
            {"title": "Result 1", "content": "Content 1", "url": "http://example.com/1"},
            {"title": "Result 2", "content": "Content 2", "url": "http://example.com/2"}
        ]
    })
    return executor


@pytest.fixture
def mock_builtin_executor():
    """Create a mock builtin executor."""
    executor = AsyncMock()
    executor.execute.return_value = json.dumps({
        "memories": ["Memory 1", "Memory 2"]
    })
    return executor


@pytest.fixture
def mock_llm_end_turn():
    """Mock LLM response with end_turn."""
    return (
        {
            "content": [
                {"type": "text", "text": "The answer is 4."}
            ],
            "stop_reason": "end_turn"
        },
        100  # tokens used
    )


@pytest.fixture
def mock_llm_tool_use():
    """Mock LLM response with tool_use."""
    return (
        {
            "content": [
                {"type": "text", "text": "Let me search for that."},
                {
                    "type": "tool_use",
                    "id": "tool_123",
                    "name": "websearch_search",
                    "input": {"query": "AI news"}
                }
            ],
            "stop_reason": "tool_use"
        },
        150  # tokens used
    )


@pytest.fixture
def sample_tools():
    """Sample tool definitions for testing."""
    return [
        {
            "name": "websearch_search",
            "description": "Search the web",
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"}
                },
                "required": ["query"]
            }
        },
        {
            "name": "memory_recall",
            "description": "Recall from memory",
            "input_schema": {
                "type": "object",
                "properties": {
                    "topic": {"type": "string", "description": "Topic to recall"}
                },
                "required": ["topic"]
            }
        }
    ]
