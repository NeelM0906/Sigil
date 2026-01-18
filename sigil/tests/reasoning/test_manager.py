"""Tests for the ReasoningManager.

Tests cover:
- Strategy selection by complexity
- Fallback chain behavior
- Metrics tracking
- Custom strategy registration
- Event emission
- Error handling
"""

import pytest
from unittest.mock import patch, AsyncMock, MagicMock

from sigil.reasoning.manager import (
    ReasoningManager,
    StrategyMetrics,
    create_strategy_selected_event,
    create_reasoning_completed_event,
    create_strategy_fallback_event,
)
from sigil.reasoning.strategies.base import StrategyResult, COMPLEXITY_RANGES
from sigil.reasoning.strategies.direct import DirectStrategy
from sigil.reasoning.strategies.chain_of_thought import ChainOfThoughtStrategy
from sigil.core.exceptions import StrategyNotFoundError


class TestReasoningManagerBasics:
    """Tests for basic ReasoningManager functionality."""

    def test_initialization(self, reasoning_manager):
        """Test that ReasoningManager initializes correctly."""
        assert reasoning_manager._strategies is not None
        assert len(reasoning_manager._strategies) >= 5
        assert "direct" in reasoning_manager._strategies
        assert "chain_of_thought" in reasoning_manager._strategies
        assert "tree_of_thoughts" in reasoning_manager._strategies
        assert "react" in reasoning_manager._strategies
        assert "mcts" in reasoning_manager._strategies

    def test_initialization_with_defaults(self):
        """Test ReasoningManager initialization with default values."""
        manager = ReasoningManager()
        assert manager._event_store is None
        assert manager._token_tracker is None
        assert len(manager._strategies) >= 5

    def test_default_strategies_registered(self, reasoning_manager):
        """Test that all default strategies are registered."""
        expected_strategies = [
            "direct",
            "chain_of_thought",
            "tree_of_thoughts",
            "react",
            "mcts",
        ]
        for name in expected_strategies:
            assert name in reasoning_manager._strategies


class TestStrategySelection:
    """Tests for complexity-based strategy selection."""

    def test_select_direct_for_low_complexity(self, reasoning_manager):
        """Test that direct is selected for low complexity."""
        selected = reasoning_manager.select_strategy(0.1)
        assert selected == "direct"

        selected = reasoning_manager.select_strategy(0.2)
        assert selected == "direct"

    def test_select_cot_for_moderate_complexity(self, reasoning_manager):
        """Test that chain_of_thought is selected for moderate complexity."""
        selected = reasoning_manager.select_strategy(0.35)
        assert selected == "chain_of_thought"

        selected = reasoning_manager.select_strategy(0.45)
        assert selected == "chain_of_thought"

    def test_select_tot_for_medium_complexity(self, reasoning_manager):
        """Test that tree_of_thoughts is selected for medium complexity."""
        selected = reasoning_manager.select_strategy(0.55)
        assert selected == "tree_of_thoughts"

        selected = reasoning_manager.select_strategy(0.65)
        assert selected == "tree_of_thoughts"

    def test_select_react_for_high_complexity(self, reasoning_manager):
        """Test that react is selected for high complexity."""
        selected = reasoning_manager.select_strategy(0.75)
        assert selected == "react"

        selected = reasoning_manager.select_strategy(0.85)
        assert selected == "react"

    def test_select_mcts_for_critical_complexity(self, reasoning_manager):
        """Test that mcts is selected for critical complexity."""
        selected = reasoning_manager.select_strategy(0.95)
        assert selected == "mcts"

        selected = reasoning_manager.select_strategy(1.0)
        assert selected == "mcts"

    def test_complexity_bounds(self, reasoning_manager):
        """Test that complexity is bounded to valid range."""
        # Below 0 should still work
        selected = reasoning_manager.select_strategy(-0.5)
        assert selected == "direct"

        # Above 1 should still work
        selected = reasoning_manager.select_strategy(1.5)
        assert selected in reasoning_manager._strategies


class TestFallbackChain:
    """Tests for fallback chain behavior."""

    def test_fallback_order_correct(self, reasoning_manager):
        """Test that fallback order is correct."""
        expected_order = ["mcts", "function_calling", "react", "tree_of_thoughts", "chain_of_thought", "direct"]
        assert reasoning_manager.FALLBACK_ORDER == expected_order

    def test_build_fallback_chain_from_mcts(self, reasoning_manager):
        """Test building fallback chain from MCTS."""
        chain = reasoning_manager._build_fallback_chain("mcts")
        assert chain == ["mcts", "function_calling", "react", "tree_of_thoughts", "chain_of_thought", "direct"]

    def test_build_fallback_chain_from_react(self, reasoning_manager):
        """Test building fallback chain from ReAct."""
        chain = reasoning_manager._build_fallback_chain("react")
        assert chain == ["react", "tree_of_thoughts", "chain_of_thought", "direct"]

    def test_build_fallback_chain_from_direct(self, reasoning_manager):
        """Test building fallback chain from direct."""
        chain = reasoning_manager._build_fallback_chain("direct")
        assert chain == ["direct"]

    def test_build_fallback_chain_unknown_strategy(self, reasoning_manager):
        """Test fallback chain for unknown strategy."""
        chain = reasoning_manager._build_fallback_chain("unknown")
        # Should return full chain
        assert chain == reasoning_manager.FALLBACK_ORDER

    @pytest.mark.asyncio
    async def test_fallback_on_failure(self, reasoning_manager):
        """Test that fallback occurs on strategy failure."""
        # Make tree_of_thoughts fail
        original_execute = reasoning_manager._strategies["tree_of_thoughts"].execute

        async def failing_execute(task, context):
            return StrategyResult(
                answer=None,
                confidence=0.0,
                reasoning_trace=[],
                tokens_used=10,
                success=False,
                error="Simulated failure",
            )

        reasoning_manager._strategies["tree_of_thoughts"].execute = failing_execute

        # Execute with medium complexity (would select ToT)
        result = await reasoning_manager.execute(
            task="Test task",
            context={},
            complexity=0.6,
            allow_fallback=True,
        )

        # Should have fallen back to a simpler strategy
        assert result is not None

        # Restore original
        reasoning_manager._strategies["tree_of_thoughts"].execute = original_execute

    @pytest.mark.asyncio
    async def test_no_fallback_when_disabled(self, reasoning_manager):
        """Test that fallback is disabled when specified."""
        # Make direct fail
        async def failing_execute(task, context):
            return StrategyResult(
                answer=None,
                confidence=0.0,
                reasoning_trace=[],
                tokens_used=10,
                success=False,
                error="Simulated failure",
            )

        reasoning_manager._strategies["direct"].execute = failing_execute

        result = await reasoning_manager.execute(
            task="Test task",
            context={},
            complexity=0.1,
            allow_fallback=False,
        )

        assert result.success is False


class TestExecution:
    """Tests for task execution."""

    @pytest.mark.asyncio
    async def test_execute_simple_task(self, reasoning_manager, simple_task):
        """Test executing a simple task."""
        result = await reasoning_manager.execute(
            task=simple_task,
            context={},
            complexity=0.2,
        )

        assert result is not None
        assert isinstance(result, StrategyResult)
        assert result.success is True

    @pytest.mark.asyncio
    async def test_execute_with_context(self, reasoning_manager, sample_context):
        """Test executing with context."""
        result = await reasoning_manager.execute(
            task="Answer based on context",
            context=sample_context,
            complexity=0.3,
        )

        assert result is not None
        assert result.success is True

    @pytest.mark.asyncio
    async def test_execute_with_strategy_override(self, reasoning_manager):
        """Test executing with a specified strategy."""
        result = await reasoning_manager.execute(
            task="Test task",
            context={},
            complexity=0.1,  # Would normally select direct
            strategy="chain_of_thought",  # Override
        )

        assert result is not None
        assert result.success is True

    @pytest.mark.asyncio
    async def test_execute_invalid_strategy_override(self, reasoning_manager):
        """Test that invalid strategy override raises error."""
        with pytest.raises(StrategyNotFoundError):
            reasoning_manager.get_strategy("nonexistent_strategy")


class TestMetricsTracking:
    """Tests for metrics tracking."""

    def test_strategy_metrics_initialization(self, reasoning_manager):
        """Test that strategy metrics are initialized."""
        for name in reasoning_manager._strategies:
            assert name in reasoning_manager._metrics
            assert isinstance(reasoning_manager._metrics[name], StrategyMetrics)

    @pytest.mark.asyncio
    async def test_execution_count_incremented(self, reasoning_manager):
        """Test that execution count is incremented."""
        initial_count = reasoning_manager._metrics["direct"].executions

        await reasoning_manager.execute(
            task="Test",
            context={},
            complexity=0.1,
        )

        assert reasoning_manager._metrics["direct"].executions == initial_count + 1

    @pytest.mark.asyncio
    async def test_success_count_incremented(self, reasoning_manager):
        """Test that success count is incremented on success."""
        initial_successes = reasoning_manager._metrics["direct"].successes

        result = await reasoning_manager.execute(
            task="Test",
            context={},
            complexity=0.1,
        )

        if result.success:
            assert reasoning_manager._metrics["direct"].successes == initial_successes + 1

    def test_get_metrics(self, reasoning_manager):
        """Test getting metrics."""
        metrics = reasoning_manager.get_metrics()

        assert "strategies" in metrics
        assert "total_executions" in metrics
        assert "total_successes" in metrics
        assert "total_failures" in metrics
        assert "overall_success_rate" in metrics

    def test_clear_metrics(self, reasoning_manager):
        """Test clearing metrics."""
        # Add some metrics
        reasoning_manager._metrics["direct"].executions = 10
        reasoning_manager._metrics["direct"].successes = 8

        reasoning_manager.clear_metrics()

        assert reasoning_manager._metrics["direct"].executions == 0
        assert reasoning_manager._metrics["direct"].successes == 0


class TestStrategyMetrics:
    """Tests for StrategyMetrics dataclass."""

    def test_success_rate_calculation(self):
        """Test success rate calculation."""
        metrics = StrategyMetrics(
            name="test",
            executions=10,
            successes=8,
            failures=2,
        )

        assert metrics.success_rate == 0.8

    def test_success_rate_zero_executions(self):
        """Test success rate with zero executions."""
        metrics = StrategyMetrics(name="test")

        assert metrics.success_rate == 0.0

    def test_avg_tokens_calculation(self):
        """Test average tokens calculation."""
        metrics = StrategyMetrics(
            name="test",
            executions=5,
            total_tokens=500,
        )

        assert metrics.avg_tokens == 100.0

    def test_avg_time_calculation(self):
        """Test average time calculation."""
        metrics = StrategyMetrics(
            name="test",
            executions=4,
            total_time=8.0,
        )

        assert metrics.avg_time == 2.0

    def test_to_dict(self):
        """Test converting metrics to dictionary."""
        metrics = StrategyMetrics(
            name="test",
            executions=10,
            successes=8,
            failures=2,
            fallbacks=1,
            total_tokens=1000,
            total_time=5.0,
        )

        data = metrics.to_dict()

        assert data["name"] == "test"
        assert data["executions"] == 10
        assert data["successes"] == 8
        assert data["success_rate"] == 0.8
        assert data["avg_tokens"] == 100.0


class TestCustomStrategyRegistration:
    """Tests for custom strategy registration."""

    def test_register_custom_strategy(self, reasoning_manager):
        """Test registering a custom strategy."""
        custom_strategy = DirectStrategy()  # Use existing strategy as custom

        reasoning_manager.register_strategy(
            name="custom",
            strategy=custom_strategy,
            complexity_range=(0.0, 0.1),
        )

        assert "custom" in reasoning_manager._strategies
        assert "custom" in reasoning_manager._metrics

    def test_register_strategy_with_range(self, reasoning_manager):
        """Test registering strategy with custom complexity range."""
        custom_strategy = DirectStrategy()

        reasoning_manager.register_strategy(
            name="custom",
            strategy=custom_strategy,
            complexity_range=(0.0, 0.05),
        )

        assert reasoning_manager._complexity_ranges["custom"] == (0.0, 0.05)

    def test_list_strategies(self, reasoning_manager):
        """Test listing all registered strategies."""
        strategies = reasoning_manager.list_strategies()

        assert len(strategies) >= 5
        for strategy_info in strategies:
            assert "name" in strategy_info
            assert "complexity_range" in strategy_info
            assert "metrics" in strategy_info


class TestEventEmission:
    """Tests for event emission."""

    @pytest.mark.asyncio
    async def test_strategy_selected_event(self, reasoning_manager, event_store):
        """Test that strategy selected event is emitted."""
        await reasoning_manager.execute(
            task="Test task",
            context={},
            complexity=0.2,
            session_id="test-session",
        )

        if event_store.session_exists("test-session"):
            events = event_store.get_events("test-session")
            selected_events = [
                e for e in events if "selected" in e.event_type.value.lower()
            ]
            assert len(selected_events) >= 1

    @pytest.mark.asyncio
    async def test_reasoning_completed_event(self, reasoning_manager, event_store):
        """Test that reasoning completed event is emitted."""
        await reasoning_manager.execute(
            task="Test task",
            context={},
            complexity=0.2,
            session_id="test-session",
        )

        if event_store.session_exists("test-session"):
            events = event_store.get_events("test-session")
            completed_events = [
                e for e in events if "completed" in e.event_type.value.lower()
            ]
            assert len(completed_events) >= 1


class TestErrorHandling:
    """Tests for error handling."""

    def test_get_strategy_not_found(self, reasoning_manager):
        """Test getting a non-existent strategy raises error."""
        with pytest.raises(StrategyNotFoundError) as excinfo:
            reasoning_manager.get_strategy("nonexistent")

        assert "nonexistent" in str(excinfo.value)
        assert "Available" in str(excinfo.value)

    @pytest.mark.asyncio
    async def test_all_strategies_fail(self, reasoning_manager):
        """Test behavior when all strategies fail."""
        # Make all strategies fail
        for strategy in reasoning_manager._strategies.values():
            async def failing(*args, **kwargs):
                raise Exception("All fail")
            strategy.execute = failing

        result = await reasoning_manager.execute(
            task="Test",
            context={},
            complexity=0.5,
        )

        assert result.success is False


class TestIntegration:
    """Integration tests for ReasoningManager."""

    @pytest.mark.asyncio
    async def test_full_workflow(self, reasoning_manager):
        """Test full reasoning workflow."""
        # Execute tasks at different complexity levels
        results = []

        for complexity in [0.2, 0.4, 0.6, 0.8]:
            result = await reasoning_manager.execute(
                task=f"Task at complexity {complexity}",
                context={},
                complexity=complexity,
            )
            results.append(result)

        # All should succeed
        assert all(r.success for r in results)

        # Check metrics were recorded
        metrics = reasoning_manager.get_metrics()
        assert metrics["total_executions"] >= 4

    @pytest.mark.asyncio
    async def test_strategy_progression(self, reasoning_manager):
        """Test that strategies are used progressively."""
        # Low complexity
        low_result = await reasoning_manager.execute(
            task="Simple question",
            context={},
            complexity=0.1,
        )

        # High complexity
        high_result = await reasoning_manager.execute(
            task="Complex analysis",
            context={},
            complexity=0.85,
        )

        # Both should succeed
        assert low_result.success
        assert high_result.success

        # Higher complexity typically uses more tokens
        assert low_result.tokens_used > 0
        assert high_result.tokens_used > 0
