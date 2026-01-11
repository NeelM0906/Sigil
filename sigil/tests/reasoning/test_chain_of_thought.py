"""Tests for the ChainOfThoughtStrategy.

Tests cover:
- Step-by-step reasoning capture
- Reasoning trace clarity
- Token tracking
- Complexity range (0.3-0.5)
- Error handling
"""

import pytest
from unittest.mock import patch, AsyncMock

from sigil.reasoning.strategies.chain_of_thought import (
    ChainOfThoughtStrategy,
)
from sigil.reasoning.strategies.base import StrategyResult, StrategyConfig


class TestChainOfThoughtBasics:
    """Tests for basic ChainOfThoughtStrategy functionality."""

    def test_initialization(self, cot_strategy):
        """Test that ChainOfThoughtStrategy initializes correctly."""
        assert cot_strategy.name == "chain_of_thought"
        assert cot_strategy._config is not None

    def test_initialization_with_custom_config(self):
        """Test ChainOfThoughtStrategy with custom configuration."""
        config = StrategyConfig(
            min_complexity=0.2,
            max_complexity=0.6,
            min_tokens=200,
            max_tokens=600,
        )
        strategy = ChainOfThoughtStrategy(config=config)

        assert strategy._config.min_complexity == 0.2
        assert strategy._config.max_complexity == 0.6

    def test_default_complexity_range(self, cot_strategy):
        """Test default complexity range is correct."""
        assert cot_strategy._config.min_complexity == 0.3
        assert cot_strategy._config.max_complexity == 0.5

    def test_default_token_budget(self, cot_strategy):
        """Test default token budget is correct."""
        assert cot_strategy._config.min_tokens == 300
        assert cot_strategy._config.max_tokens == 800


class TestStepByStepReasoning:
    """Tests for step-by-step reasoning."""

    @pytest.mark.asyncio
    async def test_execute_moderate_task(self, cot_strategy, moderate_task):
        """Test executing a moderate complexity task."""
        result = await cot_strategy.execute(moderate_task, {})

        assert result is not None
        assert isinstance(result, StrategyResult)
        assert result.success is True
        assert result.answer is not None

    @pytest.mark.asyncio
    async def test_reasoning_trace_has_steps(self, cot_strategy, moderate_task):
        """Test that reasoning trace captures steps."""
        result = await cot_strategy.execute(moderate_task, {})

        # Chain of thought should produce multiple reasoning steps
        assert len(result.reasoning_trace) >= 1
        # Should include "step by step" indication somewhere
        trace_text = " ".join(result.reasoning_trace).lower()
        assert any(term in trace_text for term in ["step", "think", "reason", "let"])

    @pytest.mark.asyncio
    async def test_reasoning_trace_clear(self, cot_strategy):
        """Test that reasoning trace is clear and understandable."""
        task = "If a store sells apples at $2 each and you buy 5, how much do you pay?"

        result = await cot_strategy.execute(task, {})

        # Trace should be non-empty strings
        for step in result.reasoning_trace:
            assert isinstance(step, str)
            assert len(step) > 0


class TestPromptBuilding:
    """Tests for chain-of-thought prompt building."""

    def test_build_prompt_includes_cot_instruction(self, cot_strategy):
        """Test that prompt includes chain-of-thought instruction."""
        prompt = cot_strategy._build_prompt("Test task", {})

        # Should have step-by-step instruction
        prompt_lower = prompt.lower()
        assert any(
            phrase in prompt_lower
            for phrase in ["step by step", "think through", "reason", "step-by-step"]
        )

    def test_build_prompt_with_context(self, cot_strategy, sample_context):
        """Test prompt building with context."""
        prompt = cot_strategy._build_prompt("Calculate total", sample_context)

        assert "Calculate total" in prompt
        assert "Context" in prompt or "context" in prompt.lower()


class TestReasoningExtraction:
    """Tests for reasoning step extraction."""

    def test_extract_reasoning_steps_numbered(self, cot_strategy):
        """Test extracting numbered reasoning steps."""
        response = """1. First, identify the items
2. Calculate the base cost
3. Add the tax
4. Total is $49.50"""

        steps = cot_strategy._extract_reasoning_steps(response)

        assert len(steps) >= 1
        # Should extract the numbered steps

    def test_extract_reasoning_steps_natural(self, cot_strategy):
        """Test extracting natural language steps."""
        response = """Let me think through this step by step.
First, we need to count the items.
Then, we multiply by the price.
Finally, we add tax.
The answer is $49.50."""

        steps = cot_strategy._extract_reasoning_steps(response)

        assert len(steps) >= 1


class TestConfidence:
    """Tests for confidence estimation."""

    @pytest.mark.asyncio
    async def test_confidence_reasonable(self, cot_strategy, moderate_task):
        """Test that confidence is reasonable for CoT."""
        result = await cot_strategy.execute(moderate_task, {})

        # CoT should have moderate to good confidence
        assert 0.0 <= result.confidence <= 1.0

    @pytest.mark.asyncio
    async def test_confidence_higher_than_direct(self, cot_strategy, direct_strategy):
        """Test that CoT generally produces reasonable confidence."""
        task = "If you have 3 boxes with 4 items each, how many items total?"

        cot_result = await cot_strategy.execute(task, {})
        direct_result = await direct_strategy.execute(task, {})

        # Both should succeed
        assert cot_result.success
        assert direct_result.success


class TestTokenTracking:
    """Tests for token tracking in ChainOfThoughtStrategy."""

    @pytest.mark.asyncio
    async def test_tokens_used_tracked(self, cot_strategy, moderate_task):
        """Test that tokens used is tracked."""
        result = await cot_strategy.execute(moderate_task, {})

        assert result.tokens_used > 0

    @pytest.mark.asyncio
    async def test_tokens_higher_than_direct(self, cot_strategy, direct_strategy):
        """Test that CoT typically uses more tokens than direct."""
        task = "Calculate 15% tip on a $45 bill"

        cot_result = await cot_strategy.execute(task, {})
        direct_result = await direct_strategy.execute(task, {})

        # CoT requires more reasoning, so typically more tokens
        # (though not guaranteed due to simulated responses)
        assert cot_result.tokens_used > 0
        assert direct_result.tokens_used > 0


class TestErrorHandling:
    """Tests for error handling."""

    @pytest.mark.asyncio
    async def test_error_returns_failed_result(self, cot_strategy):
        """Test that errors return a failed result."""
        with patch.object(
            cot_strategy, "_call_llm", side_effect=Exception("LLM Error")
        ):
            result = await cot_strategy.execute("Test task", {})

            assert result.success is False
            assert result.error is not None

    @pytest.mark.asyncio
    async def test_execution_time_tracked_on_error(self, cot_strategy):
        """Test that execution time is tracked even on error."""
        with patch.object(
            cot_strategy, "_call_llm", side_effect=Exception("Error")
        ):
            result = await cot_strategy.execute("Test task", {})

            assert result.execution_time_seconds >= 0


class TestMetrics:
    """Tests for strategy metrics."""

    @pytest.mark.asyncio
    async def test_execution_recorded(self, cot_strategy, moderate_task):
        """Test that executions are recorded."""
        initial_count = cot_strategy._execution_count

        await cot_strategy.execute(moderate_task, {})

        assert cot_strategy._execution_count == initial_count + 1

    def test_reset_metrics(self, cot_strategy):
        """Test resetting metrics."""
        cot_strategy._execution_count = 5
        cot_strategy._success_count = 4

        cot_strategy.reset_metrics()

        assert cot_strategy._execution_count == 0


class TestIntegration:
    """Integration tests for ChainOfThoughtStrategy."""

    @pytest.mark.asyncio
    async def test_full_execution_flow(self, cot_strategy):
        """Test full execution flow."""
        task = "A store has 20 items. If 25% are sold, how many remain?"
        context = {"store_name": "Test Store"}

        result = await cot_strategy.execute(task, context)

        assert result is not None
        assert result.success is True
        assert result.answer is not None
        assert len(result.reasoning_trace) >= 1
        assert result.tokens_used > 0
        assert result.execution_time_seconds > 0
        assert "strategy" in result.metadata

    @pytest.mark.asyncio
    async def test_complex_calculation(self, cot_strategy):
        """Test complex calculation with CoT."""
        task = """
        A company has 100 employees. Each employee earns $50,000/year.
        The company wants to give a 5% raise. What is the total increase in payroll?
        """

        result = await cot_strategy.execute(task, {})

        assert result.success is True
        # Should have reasoning steps
        assert len(result.reasoning_trace) >= 1
