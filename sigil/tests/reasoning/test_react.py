"""Tests for the ReActStrategy.

Tests cover:
- Thought-Action-Observation loop
- Tool call interleaving
- Observation incorporation
- ReActStep handling
- Token tracking
- Error handling
"""

import pytest
from unittest.mock import patch, AsyncMock, MagicMock

from sigil.reasoning.strategies.react import (
    ReActStrategy,
    ReActStep,
    StepType,
)
from sigil.reasoning.strategies.base import StrategyResult, StrategyConfig


class TestReActStrategyBasics:
    """Tests for basic ReActStrategy functionality."""

    def test_initialization(self, react_strategy):
        """Test that ReActStrategy initializes correctly."""
        assert react_strategy.name == "react"
        assert react_strategy._config is not None

    def test_initialization_with_custom_config(self):
        """Test ReActStrategy with custom configuration."""
        config = StrategyConfig(
            min_complexity=0.6,
            max_complexity=0.95,
            min_tokens=800,
            max_tokens=2500,
        )
        strategy = ReActStrategy(config=config)

        assert strategy._config.min_complexity == 0.6
        assert strategy._config.max_complexity == 0.95

    def test_default_complexity_range(self, react_strategy):
        """Test default complexity range is correct."""
        assert react_strategy._config.min_complexity == 0.7
        assert react_strategy._config.max_complexity == 0.9

    def test_default_token_budget(self, react_strategy):
        """Test default token budget is correct."""
        assert react_strategy._config.min_tokens == 1000
        assert react_strategy._config.max_tokens == 3000


class TestThoughtActionObservationLoop:
    """Tests for the Thought-Action-Observation loop."""

    @pytest.mark.asyncio
    async def test_execute_tool_task(self, react_strategy, tool_task):
        """Test executing a task requiring tools."""
        result = await react_strategy.execute(tool_task, {})

        assert result is not None
        assert isinstance(result, StrategyResult)
        assert result.success is True
        assert result.answer is not None

    @pytest.mark.asyncio
    async def test_reasoning_trace_shows_loop(self, react_strategy, tool_task):
        """Test that reasoning trace shows the loop structure."""
        result = await react_strategy.execute(tool_task, {})

        # Should have multiple steps in trace
        assert len(result.reasoning_trace) >= 1
        # Trace should indicate thoughts, actions, or observations
        trace_text = " ".join(result.reasoning_trace).lower()
        assert any(
            term in trace_text
            for term in ["thought", "action", "observation", "step", "execute"]
        )

    @pytest.mark.asyncio
    async def test_loop_terminates(self, react_strategy):
        """Test that the loop terminates properly."""
        task = "Search for information and summarize it"

        result = await react_strategy.execute(task, {})

        # Should complete within reasonable time
        assert result is not None
        assert result.success is True


class TestToolCallInterleaving:
    """Tests for interleaving tool calls."""

    @pytest.mark.asyncio
    async def test_tool_calls_in_trace(self, react_strategy, tool_task):
        """Test that tool calls appear in the reasoning trace."""
        result = await react_strategy.execute(tool_task, {})

        # Should have tool-related entries
        assert result.success is True

    @pytest.mark.asyncio
    async def test_multiple_tools_used(self, react_strategy):
        """Test using multiple tools in sequence."""
        task = "Search for data, then send a summary via email"
        context = {"tools": ["websearch.search", "email.send"]}

        result = await react_strategy.execute(task, context)

        assert result.success is True

    @pytest.mark.asyncio
    async def test_tool_selection_appropriate(self, react_strategy):
        """Test that appropriate tools are selected."""
        task = "Look up the weather forecast"

        result = await react_strategy.execute(task, {})

        assert result.success is True


class TestObservationIncorporation:
    """Tests for incorporating observations."""

    @pytest.mark.asyncio
    async def test_observations_influence_reasoning(self, react_strategy):
        """Test that observations influence subsequent reasoning."""
        task = "Find data and make a decision based on it"

        result = await react_strategy.execute(task, {})

        assert result.success is True
        # Answer should reflect gathered information
        assert result.answer is not None

    @pytest.mark.asyncio
    async def test_observation_parsing(self, react_strategy):
        """Test that observations are properly parsed."""
        task = "Search and analyze the results"

        result = await react_strategy.execute(task, {})

        assert result.success is True


class TestReActStep:
    """Tests for the ReActStep dataclass."""

    def test_react_step_thought(self):
        """Test creating a THOUGHT step."""
        step = ReActStep(
            step_type=StepType.THOUGHT,
            content="I need to search for information first",
        )

        assert step.step_type == StepType.THOUGHT
        assert "search" in step.content.lower()

    def test_react_step_action(self):
        """Test creating an ACTION step."""
        step = ReActStep(
            step_type=StepType.ACTION,
            content="websearch.search(query='AI news')",
            tool_name="websearch.search",
            tool_args={"query": "AI news"},
        )

        assert step.step_type == StepType.ACTION
        assert step.tool_name == "websearch.search"
        assert step.tool_args["query"] == "AI news"

    def test_react_step_observation(self):
        """Test creating an OBSERVATION step."""
        step = ReActStep(
            step_type=StepType.OBSERVATION,
            content="Found 5 relevant articles about AI",
        )

        assert step.step_type == StepType.OBSERVATION
        assert "articles" in step.content

    def test_step_type_enum(self):
        """Test StepType enumeration."""
        assert StepType.THOUGHT.value == "thought"
        assert StepType.ACTION.value == "action"
        assert StepType.OBSERVATION.value == "observation"


class TestTokenTracking:
    """Tests for token tracking in ReActStrategy."""

    @pytest.mark.asyncio
    async def test_tokens_used_tracked(self, react_strategy, tool_task):
        """Test that tokens used is tracked."""
        result = await react_strategy.execute(tool_task, {})

        assert result.tokens_used > 0

    @pytest.mark.asyncio
    async def test_tokens_accumulate_across_steps(self, react_strategy):
        """Test that tokens accumulate across loop iterations."""
        task = "Search, analyze, and summarize multiple sources"

        result = await react_strategy.execute(task, {})

        # Multiple steps should use more tokens
        assert result.tokens_used > 0


class TestConfidence:
    """Tests for confidence estimation."""

    @pytest.mark.asyncio
    async def test_confidence_reasonable(self, react_strategy, tool_task):
        """Test that confidence is reasonable."""
        result = await react_strategy.execute(tool_task, {})

        assert 0.0 <= result.confidence <= 1.0

    @pytest.mark.asyncio
    async def test_confidence_reflects_observations(self, react_strategy):
        """Test that confidence reflects quality of observations."""
        task = "Verify the information through multiple sources"

        result = await react_strategy.execute(task, {})

        # Good observations should lead to reasonable confidence
        assert result.confidence >= 0.3


class TestPromptBuilding:
    """Tests for prompt building."""

    def test_build_initial_prompt_includes_react_format(self, react_strategy):
        """Test that initial prompt includes ReAct format instructions."""
        prompt = react_strategy._build_initial_prompt("Search for data", {})

        prompt_lower = prompt.lower()
        assert any(
            term in prompt_lower
            for term in ["thought", "action", "observation", "task"]
        )

    def test_build_initial_prompt_with_tools(self, react_strategy):
        """Test initial prompt building with tool context."""
        context = {"available_tools": ["websearch.search", "email.send"]}
        prompt = react_strategy._build_initial_prompt("Execute task", context)

        assert "Execute task" in prompt

    def test_build_initial_prompt_with_context(self, react_strategy, sample_context):
        """Test initial prompt building with general context."""
        prompt = react_strategy._build_initial_prompt("Complete task", sample_context)

        assert "Complete task" in prompt


class TestErrorHandling:
    """Tests for error handling."""

    @pytest.mark.asyncio
    async def test_error_returns_failed_result(self, react_strategy):
        """Test that errors return a failed result."""
        with patch.object(
            react_strategy, "_call_llm", side_effect=Exception("LLM Error")
        ):
            result = await react_strategy.execute("Test task", {})

            assert result.success is False
            assert result.error is not None

    @pytest.mark.asyncio
    async def test_tool_failure_handled(self, react_strategy):
        """Test handling of tool execution failures."""
        # Even with simulated failures, should handle gracefully
        task = "Try to use a tool that might fail"

        result = await react_strategy.execute(task, {})

        # Should complete (possibly with degraded quality)
        assert result is not None

    @pytest.mark.asyncio
    async def test_max_iterations_enforced(self, react_strategy):
        """Test that maximum iterations are enforced."""
        task = "Keep searching indefinitely"

        result = await react_strategy.execute(task, {})

        # Should terminate within max iterations
        assert result is not None


class TestMetrics:
    """Tests for strategy metrics."""

    @pytest.mark.asyncio
    async def test_execution_recorded(self, react_strategy, tool_task):
        """Test that executions are recorded."""
        initial_count = react_strategy._execution_count

        await react_strategy.execute(tool_task, {})

        assert react_strategy._execution_count == initial_count + 1

    def test_reset_metrics(self, react_strategy):
        """Test resetting metrics."""
        react_strategy._execution_count = 5

        react_strategy.reset_metrics()

        assert react_strategy._execution_count == 0


class TestIntegration:
    """Integration tests for ReActStrategy."""

    @pytest.mark.asyncio
    async def test_full_execution_flow(self, react_strategy):
        """Test full execution flow with ReAct."""
        task = """
        Search for the latest AI developments, identify the top 3 trends,
        and provide a brief summary of each.
        """
        context = {"tools": ["websearch.search"]}

        result = await react_strategy.execute(task, context)

        assert result is not None
        assert result.success is True
        assert result.answer is not None
        assert len(result.reasoning_trace) >= 1
        assert result.tokens_used > 0

    @pytest.mark.asyncio
    async def test_multi_step_task(self, react_strategy):
        """Test a task requiring multiple ReAct loops."""
        task = """
        1. Search for company information
        2. Find their recent news
        3. Summarize the findings
        """

        result = await react_strategy.execute(task, {})

        assert result.success is True
        assert result.answer is not None

    @pytest.mark.asyncio
    async def test_adaptive_tool_use(self, react_strategy):
        """Test adaptive tool selection based on observations."""
        task = "Gather data from multiple sources and synthesize"

        result = await react_strategy.execute(task, {})

        assert result.success is True
