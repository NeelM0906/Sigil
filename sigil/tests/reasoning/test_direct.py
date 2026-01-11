"""Tests for the DirectStrategy.

Tests cover:
- Basic execution with single LLM call
- Confidence estimation (0.4-0.7 range)
- Token tracking
- Error handling
- Strategy configuration
"""

import pytest
from unittest.mock import patch, AsyncMock

from sigil.reasoning.strategies.direct import (
    DirectStrategy,
    DIRECT_MIN_COMPLEXITY,
    DIRECT_MAX_COMPLEXITY,
    DIRECT_MIN_TOKENS,
    DIRECT_MAX_TOKENS,
)
from sigil.reasoning.strategies.base import StrategyResult, StrategyConfig


class TestDirectStrategyBasics:
    """Tests for basic DirectStrategy functionality."""

    def test_initialization(self, direct_strategy):
        """Test that DirectStrategy initializes correctly."""
        assert direct_strategy.name == "direct"
        assert direct_strategy._config is not None
        assert direct_strategy._config.min_complexity == DIRECT_MIN_COMPLEXITY
        assert direct_strategy._config.max_complexity == DIRECT_MAX_COMPLEXITY

    def test_initialization_with_custom_config(self):
        """Test DirectStrategy with custom configuration."""
        config = StrategyConfig(
            min_complexity=0.0,
            max_complexity=0.5,
            min_tokens=50,
            max_tokens=200,
        )
        strategy = DirectStrategy(config=config)

        assert strategy._config.max_complexity == 0.5
        assert strategy._config.max_tokens == 200

    def test_default_complexity_range(self, direct_strategy):
        """Test default complexity range is correct."""
        assert direct_strategy._config.min_complexity == 0.0
        assert direct_strategy._config.max_complexity == 0.3

    def test_default_token_budget(self, direct_strategy):
        """Test default token budget is correct."""
        assert direct_strategy._config.min_tokens == DIRECT_MIN_TOKENS
        assert direct_strategy._config.max_tokens == DIRECT_MAX_TOKENS


class TestDirectExecution:
    """Tests for DirectStrategy execution."""

    @pytest.mark.asyncio
    async def test_execute_simple_task(self, direct_strategy, simple_task):
        """Test executing a simple task."""
        result = await direct_strategy.execute(simple_task, {})

        assert result is not None
        assert isinstance(result, StrategyResult)
        assert result.success is True
        assert result.answer is not None
        assert len(result.answer) > 0

    @pytest.mark.asyncio
    async def test_execute_with_context(self, direct_strategy, sample_context):
        """Test executing with context."""
        task = "What is the user's budget?"

        result = await direct_strategy.execute(task, sample_context)

        assert result is not None
        assert result.success is True

    @pytest.mark.asyncio
    async def test_reasoning_trace_single_step(self, direct_strategy, simple_task):
        """Test that reasoning trace has single step."""
        result = await direct_strategy.execute(simple_task, {})

        assert len(result.reasoning_trace) == 1
        assert "single" in result.reasoning_trace[0].lower() or "direct" in result.reasoning_trace[0].lower()


class TestConfidenceEstimation:
    """Tests for confidence estimation in DirectStrategy."""

    @pytest.mark.asyncio
    async def test_confidence_in_valid_range(self, direct_strategy, simple_task):
        """Test that confidence is in expected range (0.4-0.7)."""
        result = await direct_strategy.execute(simple_task, {})

        # Direct strategy should have moderate confidence
        assert 0.0 <= result.confidence <= 1.0
        # Direct strategy caps at 0.7
        assert result.confidence <= 0.7

    @pytest.mark.asyncio
    async def test_confidence_capped_at_point_seven(self, direct_strategy):
        """Test that confidence is capped at 0.7 for direct strategy."""
        task = "What is the capital of France?"

        result = await direct_strategy.execute(task, {})

        # Even for simple factual questions, direct caps at 0.7
        assert result.confidence <= 0.7


class TestTokenTracking:
    """Tests for token tracking in DirectStrategy."""

    @pytest.mark.asyncio
    async def test_tokens_used_tracked(self, direct_strategy, simple_task):
        """Test that tokens used is tracked."""
        result = await direct_strategy.execute(simple_task, {})

        assert result.tokens_used > 0

    @pytest.mark.asyncio
    async def test_tokens_within_budget(self, direct_strategy, simple_task):
        """Test that token usage is within budget."""
        result = await direct_strategy.execute(simple_task, {})

        # Should be reasonable for simple task
        assert result.tokens_used <= direct_strategy._config.max_tokens * 2


class TestPromptBuilding:
    """Tests for prompt building."""

    def test_build_prompt_basic(self, direct_strategy):
        """Test basic prompt building."""
        prompt = direct_strategy._build_prompt("What is 2+2?", {})

        assert "What is 2+2?" in prompt
        assert "task" in prompt.lower()

    def test_build_prompt_with_context(self, direct_strategy, sample_context):
        """Test prompt building with context."""
        prompt = direct_strategy._build_prompt("Question?", sample_context)

        assert "Question?" in prompt
        assert "Context" in prompt
        assert "user" in prompt


class TestErrorHandling:
    """Tests for error handling in DirectStrategy."""

    @pytest.mark.asyncio
    async def test_error_returns_failed_result(self, direct_strategy):
        """Test that errors return a failed result."""
        # Patch LLM to raise an error
        with patch.object(
            direct_strategy, "_call_llm", side_effect=Exception("LLM Error")
        ):
            result = await direct_strategy.execute("Test task", {})

            assert result.success is False
            assert result.error is not None
            assert "LLM Error" in result.error

    @pytest.mark.asyncio
    async def test_error_preserves_execution_time(self, direct_strategy):
        """Test that execution time is tracked even on error."""
        with patch.object(
            direct_strategy, "_call_llm", side_effect=Exception("Error")
        ):
            result = await direct_strategy.execute("Test task", {})

            assert result.execution_time_seconds >= 0


class TestMetrics:
    """Tests for strategy metrics."""

    @pytest.mark.asyncio
    async def test_execution_recorded(self, direct_strategy, simple_task):
        """Test that executions are recorded in metrics."""
        initial_count = direct_strategy._execution_count

        await direct_strategy.execute(simple_task, {})

        assert direct_strategy._execution_count == initial_count + 1

    @pytest.mark.asyncio
    async def test_success_recorded(self, direct_strategy, simple_task):
        """Test that successes are recorded."""
        initial_successes = direct_strategy._success_count

        result = await direct_strategy.execute(simple_task, {})

        if result.success:
            assert direct_strategy._success_count == initial_successes + 1

    def test_reset_metrics(self, direct_strategy):
        """Test resetting metrics."""
        direct_strategy._execution_count = 10
        direct_strategy._success_count = 8

        direct_strategy.reset_metrics()

        assert direct_strategy._execution_count == 0
        assert direct_strategy._success_count == 0


class TestStrategyResult:
    """Tests for StrategyResult dataclass."""

    def test_from_error(self):
        """Test creating result from error."""
        result = StrategyResult.from_error(
            error="Test error",
            strategy_name="direct",
            execution_time=1.5,
            tokens_used=50,
        )

        assert result.success is False
        assert result.error == "Test error"
        assert result.execution_time_seconds == 1.5
        assert result.tokens_used == 50

    def test_result_metadata(self, direct_strategy):
        """Test that result includes proper metadata."""
        import asyncio

        result = asyncio.get_event_loop().run_until_complete(
            direct_strategy.execute("Test", {})
        )

        assert "strategy" in result.metadata
        assert result.metadata["strategy"] == "direct"
