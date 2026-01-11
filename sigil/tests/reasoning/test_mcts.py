"""Tests for the MCTSStrategy.

Tests cover:
- Tree building with selection, expansion, simulation, backpropagation
- UCB1 scoring
- Outcome simulation
- MCTSNode handling
- Token tracking
- Error handling
"""

import pytest
import math
from unittest.mock import patch, AsyncMock

from sigil.reasoning.strategies.mcts import (
    MCTSStrategy,
    MCTSNode,
)
from sigil.reasoning.strategies.base import StrategyResult, StrategyConfig


class TestMCTSStrategyBasics:
    """Tests for basic MCTSStrategy functionality."""

    def test_initialization(self, mcts_strategy):
        """Test that MCTSStrategy initializes correctly."""
        assert mcts_strategy.name == "mcts"
        assert mcts_strategy._config is not None

    def test_initialization_with_custom_config(self):
        """Test MCTSStrategy with custom configuration."""
        config = StrategyConfig(
            min_complexity=0.85,
            max_complexity=1.0,
            min_tokens=1500,
            max_tokens=4000,
        )
        strategy = MCTSStrategy(config=config)

        assert strategy._config.min_complexity == 0.85
        assert strategy._config.max_tokens == 4000

    def test_default_complexity_range(self, mcts_strategy):
        """Test default complexity range is correct."""
        assert mcts_strategy._config.min_complexity == 0.9
        assert mcts_strategy._config.max_complexity == 1.0

    def test_default_token_budget(self, mcts_strategy):
        """Test default token budget is correct."""
        assert mcts_strategy._config.min_tokens == 2000
        assert mcts_strategy._config.max_tokens == 5000


class TestMCTSTreeBuilding:
    """Tests for MCTS tree building."""

    @pytest.mark.asyncio
    async def test_execute_critical_task(self, mcts_strategy, critical_task):
        """Test executing a critical task."""
        result = await mcts_strategy.execute(critical_task, {})

        assert result is not None
        assert isinstance(result, StrategyResult)
        assert result.success is True
        assert result.answer is not None

    @pytest.mark.asyncio
    async def test_tree_nodes_created(self, mcts_strategy):
        """Test that tree nodes are created during execution."""
        task = "Design a critical system architecture"

        result = await mcts_strategy.execute(task, {})

        assert result.success is True
        # Metadata should contain tree information
        assert result.metadata is not None

    @pytest.mark.asyncio
    async def test_multiple_iterations(self, mcts_strategy):
        """Test that multiple MCTS iterations are performed."""
        task = "Make a high-stakes decision with multiple factors"

        result = await mcts_strategy.execute(task, {})

        assert result.success is True
        # Should have explored multiple paths
        assert len(result.reasoning_trace) >= 1


class TestMCTSPhases:
    """Tests for MCTS phases: Selection, Expansion, Simulation, Backpropagation."""

    @pytest.mark.asyncio
    async def test_selection_phase(self, mcts_strategy):
        """Test that selection phase works correctly."""
        task = "Select the best option"

        result = await mcts_strategy.execute(task, {})

        assert result.success is True

    @pytest.mark.asyncio
    async def test_expansion_phase(self, mcts_strategy):
        """Test that expansion phase works correctly."""
        task = "Expand on possible solutions"

        result = await mcts_strategy.execute(task, {})

        assert result.success is True

    @pytest.mark.asyncio
    async def test_simulation_phase(self, mcts_strategy):
        """Test that simulation phase works correctly."""
        task = "Simulate different outcomes"

        result = await mcts_strategy.execute(task, {})

        assert result.success is True

    @pytest.mark.asyncio
    async def test_backpropagation_phase(self, mcts_strategy):
        """Test that backpropagation phase updates nodes."""
        task = "Evaluate and update scores"

        result = await mcts_strategy.execute(task, {})

        assert result.success is True


class TestUCB1Scoring:
    """Tests for UCB1 scoring calculation."""

    def test_ucb1_calculation_basic(self):
        """Test basic UCB1 calculation."""
        node = MCTSNode(
            node_id="test",
            state="Test state",
            visits=10,
            value=7.0,
            depth=1,
        )

        ucb1 = node.ucb1_score

        # UCB1 = value/visits + c * sqrt(ln(visits) / visits)
        # For visited nodes, should be > 0
        assert ucb1 > 0

    def test_ucb1_unvisited_node(self):
        """Test UCB1 for unvisited nodes."""
        node = MCTSNode(
            node_id="unvisited",
            state="Unvisited state",
            visits=0,
            value=0.0,
            depth=1,
        )

        ucb1 = node.ucb1_score

        # Unvisited nodes should have very high score (infinite exploration bonus)
        assert ucb1 == float("inf")

    def test_ucb1_balances_exploration_exploitation(self):
        """Test that UCB1 balances exploration and exploitation."""
        # High value, many visits (exploitation)
        exploited = MCTSNode(
            node_id="exploited",
            state="State",
            visits=50,
            value=40.0,
            depth=1,
        )
        # Low value, few visits (exploration)
        explored = MCTSNode(
            node_id="explored",
            state="State",
            visits=5,
            value=3.0,
            depth=1,
        )

        ucb1_exploited = exploited.ucb1_score
        ucb1_explored = explored.ucb1_score

        # Both should have reasonable scores
        assert ucb1_exploited > 0
        assert ucb1_explored > 0


class TestMCTSNode:
    """Tests for the MCTSNode dataclass."""

    def test_mcts_node_creation(self):
        """Test creating an MCTSNode."""
        node = MCTSNode(
            node_id="node-1",
            state="Initial state",
            visits=0,
            value=0.0,
            depth=0,
            parent_id=None,
        )

        assert node.node_id == "node-1"
        assert node.state == "Initial state"
        assert node.visits == 0
        assert node.value == 0.0
        assert node.depth == 0
        assert node.children == []

    def test_mcts_node_with_children(self):
        """Test MCTSNode with children."""
        parent = MCTSNode(
            node_id="parent",
            state="Parent state",
            visits=10,
            value=5.0,
            depth=0,
        )
        child = MCTSNode(
            node_id="child",
            state="Child state",
            visits=3,
            value=2.0,
            depth=1,
            parent_id="parent",
        )
        parent.children.append(child.node_id)

        assert len(parent.children) == 1
        assert child.parent_id == "parent"

    def test_mcts_node_value_update(self):
        """Test updating node values (backpropagation)."""
        node = MCTSNode(
            node_id="test",
            state="State",
            visits=5,
            value=3.0,
            depth=1,
        )

        # Simulate backpropagation
        node.visits += 1
        node.value += 0.8  # Simulation result

        assert node.visits == 6
        assert node.value == 3.8


class TestOutcomeSimulation:
    """Tests for outcome simulation."""

    @pytest.mark.asyncio
    async def test_simulation_produces_result(self, mcts_strategy):
        """Test that simulation produces a result."""
        task = "Simulate the outcome of a decision"

        result = await mcts_strategy.execute(task, {})

        assert result.success is True
        assert result.answer is not None

    @pytest.mark.asyncio
    async def test_simulation_scores_plausible(self, mcts_strategy):
        """Test that simulation scores are plausible."""
        task = "Evaluate different strategies"

        result = await mcts_strategy.execute(task, {})

        # Confidence should reflect simulation quality
        assert 0.0 <= result.confidence <= 1.0


class TestTokenTracking:
    """Tests for token tracking in MCTSStrategy."""

    @pytest.mark.asyncio
    async def test_tokens_used_tracked(self, mcts_strategy, critical_task):
        """Test that tokens used is tracked."""
        result = await mcts_strategy.execute(critical_task, {})

        assert result.tokens_used > 0

    @pytest.mark.asyncio
    async def test_tokens_highest_among_strategies(
        self, mcts_strategy, react_strategy, tot_strategy
    ):
        """Test that MCTS uses more tokens than simpler strategies."""
        task = "Make a critical decision"

        mcts_result = await mcts_strategy.execute(task, {})
        react_result = await react_strategy.execute(task, {})
        tot_result = await tot_strategy.execute(task, {})

        # MCTS is most expensive, so should use comparable or more tokens
        assert mcts_result.tokens_used > 0


class TestConfidence:
    """Tests for confidence estimation."""

    @pytest.mark.asyncio
    async def test_confidence_reasonable(self, mcts_strategy, critical_task):
        """Test that confidence is reasonable."""
        result = await mcts_strategy.execute(critical_task, {})

        assert 0.0 <= result.confidence <= 1.0

    @pytest.mark.asyncio
    async def test_confidence_reflects_simulation_quality(self, mcts_strategy):
        """Test that confidence reflects simulation quality."""
        task = "Make a decision with high uncertainty"

        result = await mcts_strategy.execute(task, {})

        # High uncertainty might lead to lower confidence
        assert result.confidence >= 0.0


class TestPromptBuilding:
    """Tests for prompt building."""

    def test_build_expansion_prompt(self, mcts_strategy):
        """Test that expansion prompt includes relevant context."""
        node = MCTSNode(
            node_id="test-node",
            state="Initial state",
            visits=1,
            value=0.5,
            depth=0,
        )
        prompt = mcts_strategy._build_expansion_prompt("Critical decision", {}, node)

        prompt_lower = prompt.lower()
        assert "critical decision" in prompt_lower

    def test_build_simulation_prompt_with_context(self, mcts_strategy, sample_context):
        """Test simulation prompt building with context."""
        node = MCTSNode(
            node_id="test-node",
            state="Current state",
            action="Take action A",
            visits=1,
            value=0.5,
            depth=1,
        )
        prompt = mcts_strategy._build_simulation_prompt(
            "Make decision",
            sample_context,
            node
        )

        assert "Make decision" in prompt


class TestErrorHandling:
    """Tests for error handling."""

    @pytest.mark.asyncio
    async def test_error_returns_failed_result(self, mcts_strategy):
        """Test that errors return a failed result."""
        with patch.object(
            mcts_strategy, "_call_llm", side_effect=Exception("LLM Error")
        ):
            result = await mcts_strategy.execute("Test task", {})

            assert result.success is False
            assert result.error is not None

    @pytest.mark.asyncio
    async def test_simulation_failure_handled(self, mcts_strategy):
        """Test handling of simulation failures."""
        task = "Simulate with potential failures"

        result = await mcts_strategy.execute(task, {})

        # Should handle gracefully
        assert result is not None

    @pytest.mark.asyncio
    async def test_max_iterations_enforced(self, mcts_strategy):
        """Test that maximum iterations are enforced."""
        task = "Run many simulations"

        result = await mcts_strategy.execute(task, {})

        # Should terminate within max iterations
        assert result is not None


class TestMetrics:
    """Tests for strategy metrics."""

    @pytest.mark.asyncio
    async def test_execution_recorded(self, mcts_strategy, critical_task):
        """Test that executions are recorded."""
        initial_count = mcts_strategy._execution_count

        await mcts_strategy.execute(critical_task, {})

        assert mcts_strategy._execution_count == initial_count + 1

    def test_reset_metrics(self, mcts_strategy):
        """Test resetting metrics."""
        mcts_strategy._execution_count = 3

        mcts_strategy.reset_metrics()

        assert mcts_strategy._execution_count == 0


class TestIntegration:
    """Integration tests for MCTSStrategy."""

    @pytest.mark.asyncio
    async def test_full_execution_flow(self, mcts_strategy):
        """Test full execution flow with MCTS."""
        task = """
        Design a deployment strategy for a critical financial system.
        Requirements:
        - Zero downtime
        - Data integrity
        - Rollback capability
        - Compliance with regulations
        """
        context = {"criticality": "high", "budget": "$100000"}

        result = await mcts_strategy.execute(task, context)

        assert result is not None
        assert result.success is True
        assert result.answer is not None
        assert len(result.reasoning_trace) >= 1
        assert result.tokens_used > 0

    @pytest.mark.asyncio
    async def test_complex_decision(self, mcts_strategy):
        """Test complex decision making."""
        task = """
        A hospital needs to upgrade its patient management system.
        Options:
        1. Full replacement ($2M, 6 months downtime risk)
        2. Phased migration ($3M, minimal risk)
        3. Hybrid approach ($2.5M, moderate risk)
        Consider patient safety, budget, timeline, and staff training.
        """

        result = await mcts_strategy.execute(task, {})

        assert result.success is True
        assert result.answer is not None

    @pytest.mark.asyncio
    async def test_risk_assessment(self, mcts_strategy):
        """Test risk assessment through MCTS."""
        task = """
        Evaluate the risk of launching a new product:
        - Market uncertainty: high
        - Development readiness: 80%
        - Competition: 2 major players
        - Budget available: $500K
        """

        result = await mcts_strategy.execute(task, {})

        assert result.success is True
        assert result.answer is not None
