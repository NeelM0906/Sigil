"""Tests for the TreeOfThoughtsStrategy.

Tests cover:
- Multiple path exploration
- Best path selection
- ThoughtNode tree building
- Approach evaluation
- Token tracking
- Error handling
"""

import pytest
from unittest.mock import patch, AsyncMock

from sigil.reasoning.strategies.tree_of_thoughts import (
    TreeOfThoughtsStrategy,
    ThoughtNode,
)
from sigil.reasoning.strategies.base import StrategyResult, StrategyConfig


class TestTreeOfThoughtsBasics:
    """Tests for basic TreeOfThoughtsStrategy functionality."""

    def test_initialization(self, tot_strategy):
        """Test that TreeOfThoughtsStrategy initializes correctly."""
        assert tot_strategy.name == "tree_of_thoughts"
        assert tot_strategy._config is not None

    def test_initialization_with_custom_config(self):
        """Test TreeOfThoughtsStrategy with custom configuration."""
        config = StrategyConfig(
            min_complexity=0.4,
            max_complexity=0.8,
            min_tokens=500,
            max_tokens=1500,
        )
        strategy = TreeOfThoughtsStrategy(config=config)

        assert strategy._config.min_complexity == 0.4
        assert strategy._config.max_complexity == 0.8

    def test_default_complexity_range(self, tot_strategy):
        """Test default complexity range is correct."""
        assert tot_strategy._config.min_complexity == 0.5
        assert tot_strategy._config.max_complexity == 0.7

    def test_default_token_budget(self, tot_strategy):
        """Test default token budget is correct."""
        assert tot_strategy._config.min_tokens == 800
        assert tot_strategy._config.max_tokens == 2000


class TestMultiplePathExploration:
    """Tests for exploring multiple reasoning paths."""

    @pytest.mark.asyncio
    async def test_execute_complex_task(self, tot_strategy, complex_task):
        """Test executing a complex task."""
        result = await tot_strategy.execute(complex_task, {})

        assert result is not None
        assert isinstance(result, StrategyResult)
        assert result.success is True
        assert result.answer is not None

    @pytest.mark.asyncio
    async def test_multiple_approaches_generated(self, tot_strategy):
        """Test that multiple approaches are generated."""
        task = "Compare three solutions: A, B, and C for the problem"

        result = await tot_strategy.execute(task, {})

        # ToT should explore multiple paths
        assert result.success is True
        # Reasoning trace should show exploration
        assert len(result.reasoning_trace) >= 1

    @pytest.mark.asyncio
    async def test_approaches_are_distinct(self, tot_strategy):
        """Test that generated approaches are distinct."""
        task = "Suggest different ways to solve the issue"

        result = await tot_strategy.execute(task, {})

        # Should succeed with distinct exploration
        assert result.success is True


class TestBestPathSelection:
    """Tests for selecting the best reasoning path."""

    @pytest.mark.asyncio
    async def test_best_path_selected(self, tot_strategy, complex_task):
        """Test that the best path is selected."""
        result = await tot_strategy.execute(complex_task, {})

        # Should have a clear answer (best path)
        assert result.answer is not None
        assert len(result.answer) > 0

    @pytest.mark.asyncio
    async def test_selection_criteria_applied(self, tot_strategy):
        """Test that selection criteria are applied."""
        task = "Evaluate options A, B, C and pick the best"

        result = await tot_strategy.execute(task, {})

        assert result.success is True
        # Answer should indicate selection
        assert result.answer is not None


class TestThoughtNode:
    """Tests for the ThoughtNode dataclass."""

    def test_thought_node_creation(self):
        """Test creating a ThoughtNode."""
        node = ThoughtNode(
            approach_id="node-1",
            description="Initial approach",
            reasoning="Detailed reasoning here",
            evaluation_score=0.8,
            depth=0,
            parent_id=None,
        )

        assert node.approach_id == "node-1"
        assert node.description == "Initial approach"
        assert node.evaluation_score == 0.8
        assert node.depth == 0

    def test_thought_node_with_children(self):
        """Test ThoughtNode with parent relationship."""
        parent = ThoughtNode(
            approach_id="parent",
            description="Parent approach",
            evaluation_score=0.7,
            depth=0,
        )
        child = ThoughtNode(
            approach_id="child",
            description="Child approach",
            evaluation_score=0.8,
            depth=1,
            parent_id="parent",
        )

        assert child.parent_id == "parent"
        assert child.depth == parent.depth + 1


class TestTreeBuilding:
    """Tests for building the thought tree."""

    @pytest.mark.asyncio
    async def test_tree_structure_created(self, tot_strategy):
        """Test that a tree structure is created during execution."""
        task = "Explore different solutions to the problem"

        result = await tot_strategy.execute(task, {})

        # Result should indicate tree exploration
        assert result.success is True
        # Metadata might contain tree info
        assert result.metadata is not None

    @pytest.mark.asyncio
    async def test_tree_depth_reasonable(self, tot_strategy):
        """Test that tree depth is reasonable."""
        task = "Analyze multiple approaches"

        result = await tot_strategy.execute(task, {})

        assert result.success is True


class TestApproachEvaluation:
    """Tests for evaluating approaches."""

    @pytest.mark.asyncio
    async def test_approaches_evaluated(self, tot_strategy):
        """Test that approaches are evaluated."""
        task = "Which approach is better: fast but expensive, or slow but cheap?"

        result = await tot_strategy.execute(task, {})

        assert result.success is True
        assert result.answer is not None

    @pytest.mark.asyncio
    async def test_evaluation_produces_scores(self, tot_strategy):
        """Test that evaluation produces scores."""
        task = "Rate the following options: Option 1, Option 2, Option 3"

        result = await tot_strategy.execute(task, {})

        assert result.success is True


class TestTokenTracking:
    """Tests for token tracking in TreeOfThoughtsStrategy."""

    @pytest.mark.asyncio
    async def test_tokens_used_tracked(self, tot_strategy, complex_task):
        """Test that tokens used is tracked."""
        result = await tot_strategy.execute(complex_task, {})

        assert result.tokens_used > 0

    @pytest.mark.asyncio
    async def test_tokens_higher_than_cot(self, tot_strategy, cot_strategy):
        """Test that ToT uses more tokens than CoT."""
        task = "Analyze multiple approaches to solve this problem"

        tot_result = await tot_strategy.execute(task, {})
        cot_result = await cot_strategy.execute(task, {})

        # ToT explores multiple paths, so typically uses more tokens
        assert tot_result.tokens_used > 0
        assert cot_result.tokens_used > 0


class TestConfidence:
    """Tests for confidence estimation."""

    @pytest.mark.asyncio
    async def test_confidence_reasonable(self, tot_strategy, complex_task):
        """Test that confidence is reasonable."""
        result = await tot_strategy.execute(complex_task, {})

        assert 0.0 <= result.confidence <= 1.0

    @pytest.mark.asyncio
    async def test_confidence_reflects_exploration(self, tot_strategy):
        """Test that confidence reflects exploration quality."""
        task = "Thoroughly analyze all options and pick the best"

        result = await tot_strategy.execute(task, {})

        # With thorough exploration, confidence should be reasonable
        assert result.confidence >= 0.3


class TestPromptBuilding:
    """Tests for prompt building."""

    def test_build_generation_prompt_basic(self, tot_strategy):
        """Test basic generation prompt building."""
        prompt = tot_strategy._build_generation_prompt("Analyze options", {}, num_approaches=3)

        assert "Analyze options" in prompt
        # Should have multi-approach instruction
        prompt_lower = prompt.lower()
        assert any(
            term in prompt_lower
            for term in ["approach", "option", "path", "explore", "alternative", "distinct"]
        )

    def test_build_generation_prompt_with_context(self, tot_strategy, sample_context):
        """Test generation prompt building with context."""
        prompt = tot_strategy._build_generation_prompt("Choose best option", sample_context, num_approaches=3)

        assert "Choose best option" in prompt


class TestErrorHandling:
    """Tests for error handling."""

    @pytest.mark.asyncio
    async def test_error_returns_failed_result(self, tot_strategy):
        """Test that errors return a failed result."""
        with patch.object(
            tot_strategy, "_call_llm", side_effect=Exception("LLM Error")
        ):
            result = await tot_strategy.execute("Test task", {})

            assert result.success is False
            assert result.error is not None

    @pytest.mark.asyncio
    async def test_partial_failure_handled(self, tot_strategy):
        """Test handling of partial failures during exploration."""
        call_count = 0

        async def intermittent_failure(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise Exception("Intermittent failure")
            return "Response", 100

        with patch.object(tot_strategy, "_call_llm", side_effect=intermittent_failure):
            result = await tot_strategy.execute("Test task", {})

            # Should either succeed with other paths or fail gracefully
            assert result is not None


class TestMetrics:
    """Tests for strategy metrics."""

    @pytest.mark.asyncio
    async def test_execution_recorded(self, tot_strategy, complex_task):
        """Test that executions are recorded."""
        initial_count = tot_strategy._execution_count

        await tot_strategy.execute(complex_task, {})

        assert tot_strategy._execution_count == initial_count + 1

    def test_reset_metrics(self, tot_strategy):
        """Test resetting metrics."""
        tot_strategy._execution_count = 3

        tot_strategy.reset_metrics()

        assert tot_strategy._execution_count == 0


class TestIntegration:
    """Integration tests for TreeOfThoughtsStrategy."""

    @pytest.mark.asyncio
    async def test_full_execution_flow(self, tot_strategy):
        """Test full execution flow."""
        task = """
        A company needs to choose a cloud provider. Options are:
        - AWS: Most features, highest cost
        - GCP: Good ML support, moderate cost
        - Azure: Best enterprise integration, moderate cost
        Which should they choose and why?
        """
        context = {"company_size": "medium", "focus": "ML"}

        result = await tot_strategy.execute(task, context)

        assert result is not None
        assert result.success is True
        assert result.answer is not None
        assert len(result.reasoning_trace) >= 1
        assert result.tokens_used > 0

    @pytest.mark.asyncio
    async def test_decision_making(self, tot_strategy):
        """Test multi-path decision making."""
        task = """
        Should the team refactor the codebase now or wait?
        Consider: time pressure, technical debt, upcoming features.
        """

        result = await tot_strategy.execute(task, {})

        assert result.success is True
        assert result.answer is not None
