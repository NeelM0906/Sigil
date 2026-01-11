"""Monte Carlo Tree Search reasoning strategy for Sigil v2.

This module implements the MCTSStrategy, which uses Monte Carlo Tree Search
for decision making on high-complexity tasks.

Classes:
    MCTSNode: A node in the MCTS tree.
    MCTSStrategy: MCTS-based reasoning for critical decisions.

Example:
    >>> from sigil.reasoning.strategies.mcts import MCTSStrategy
    >>>
    >>> strategy = MCTSStrategy()
    >>> result = await strategy.execute(
    ...     task="Design optimal pricing strategy",
    ...     context={"constraints": "maximize revenue while retaining customers"},
    ... )
"""

from __future__ import annotations

import asyncio
import math
import random
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from sigil.state.store import EventStore
from sigil.telemetry.tokens import TokenTracker

from sigil.reasoning.strategies.base import (
    BaseReasoningStrategy,
    StrategyResult,
    StrategyConfig,
    utc_now,
)


# =============================================================================
# Constants
# =============================================================================

MCTS_MIN_COMPLEXITY = 0.9
MCTS_MAX_COMPLEXITY = 1.0
MCTS_MIN_TOKENS = 2000
MCTS_MAX_TOKENS = 5000

DEFAULT_SIMULATIONS = 50
"""Default number of MCTS simulations."""

MAX_PARALLEL_SIMULATIONS = 10
"""Maximum parallel simulations."""

UCB1_EXPLORATION = 1.414
"""UCB1 exploration constant (sqrt(2))."""


# =============================================================================
# MCTS Node
# =============================================================================


@dataclass
class MCTSNode:
    """A node in the MCTS tree.

    Represents a state/decision in the search tree.

    Attributes:
        node_id: Unique node identifier.
        state: Description of this state/decision.
        parent_id: Parent node ID (None for root).
        children: Child node IDs.
        visits: Number of times this node was visited.
        value: Accumulated value from simulations.
        depth: Depth in the tree (0 for root).
        action: Action that led to this state.
        is_terminal: Whether this is a terminal state.
        simulation_results: Results from simulations at this node.
    """

    node_id: str
    state: str
    parent_id: Optional[str] = None
    children: list[str] = field(default_factory=list)
    visits: int = 0
    value: float = 0.0
    depth: int = 0
    action: Optional[str] = None
    is_terminal: bool = False
    simulation_results: list[float] = field(default_factory=list)

    @property
    def ucb1_score(self) -> float:
        """Calculate UCB1 score for node selection.

        UCB1 = value/visits + C * sqrt(ln(parent_visits) / visits)

        For unvisited nodes, returns infinity to ensure exploration.
        """
        if self.visits == 0:
            return float("inf")

        exploitation = self.value / self.visits
        exploration = UCB1_EXPLORATION * math.sqrt(
            math.log(max(self.visits, 1)) / self.visits
        )
        return exploitation + exploration

    @property
    def average_value(self) -> float:
        """Get average value from all visits."""
        if self.visits == 0:
            return 0.0
        return self.value / self.visits

    def to_dict(self) -> dict[str, Any]:
        """Convert node to dictionary."""
        return {
            "node_id": self.node_id,
            "state": self.state[:100] + "..." if len(self.state) > 100 else self.state,
            "parent_id": self.parent_id,
            "children_count": len(self.children),
            "visits": self.visits,
            "value": self.value,
            "average_value": self.average_value,
            "depth": self.depth,
            "action": self.action,
            "is_terminal": self.is_terminal,
        }


# =============================================================================
# MCTS Strategy
# =============================================================================


class MCTSStrategy(BaseReasoningStrategy):
    """Monte Carlo Tree Search reasoning strategy.

    MCTSStrategy applies MCTS to high-complexity decision problems.
    It builds a search tree, simulates outcomes, and uses UCB1
    scoring to balance exploration and exploitation.

    Characteristics:
        - Complexity range: 0.9-1.0
        - Token budget: 2000-5000
        - Reasoning trace: Tree structure with simulation counts
        - Confidence: 0.8-0.95 (highest)
        - Best for: Strategic decisions, optimization, planning

    The strategy follows the MCTS phases:
    1. Selection: Select promising node using UCB1
    2. Expansion: Expand node with new actions
    3. Simulation: Simulate outcome (rollout)
    4. Backpropagation: Update values up the tree

    Attributes:
        name: "mcts"
        num_simulations: Number of simulations to run.
        max_depth: Maximum tree depth.
        config: Strategy configuration with MCTS-specific defaults.

    Example:
        >>> strategy = MCTSStrategy(num_simulations=100)
        >>> result = await strategy.execute(
        ...     task="Optimize marketing channel allocation",
        ...     context={"budget": "$100k", "channels": ["paid", "organic", "social"]},
        ... )
    """

    def __init__(
        self,
        config: Optional[StrategyConfig] = None,
        event_store: Optional[EventStore] = None,
        token_tracker: Optional[TokenTracker] = None,
        num_simulations: int = DEFAULT_SIMULATIONS,
        max_depth: int = 5,
    ) -> None:
        """Initialize MCTSStrategy.

        Args:
            config: Optional strategy configuration.
            event_store: Optional event store for audit trails.
            token_tracker: Optional token tracker for budget.
            num_simulations: Number of MCTS simulations.
            max_depth: Maximum tree depth.
        """
        if config is None:
            config = StrategyConfig(
                min_complexity=MCTS_MIN_COMPLEXITY,
                max_complexity=MCTS_MAX_COMPLEXITY,
                min_tokens=MCTS_MIN_TOKENS,
                max_tokens=MCTS_MAX_TOKENS,
                timeout_seconds=300.0,
            )
        super().__init__(config, event_store, token_tracker)
        self._num_simulations = num_simulations
        self._max_depth = max_depth
        self._nodes: dict[str, MCTSNode] = {}
        self._node_counter = 0

    @property
    def name(self) -> str:
        """Return the strategy name."""
        return "mcts"

    def _generate_node_id(self) -> str:
        """Generate a unique node ID."""
        self._node_counter += 1
        return f"node_{self._node_counter}"

    def _reset_tree(self) -> None:
        """Reset the MCTS tree."""
        self._nodes.clear()
        self._node_counter = 0

    def _build_expansion_prompt(
        self,
        task: str,
        context: dict[str, Any],
        node: MCTSNode,
    ) -> str:
        """Build prompt for expanding a node with actions.

        Args:
            task: The original task.
            context: Context information.
            node: The node to expand.

        Returns:
            Prompt string.
        """
        context_str = ""
        if context:
            context_str = "\n\nContext:\n"
            for key, value in context.items():
                context_str += f"- {key}: {value}\n"

        parent_path = self._get_path_to_root(node)
        path_str = " -> ".join(parent_path) if parent_path else "root"

        return f"""You are analyzing a decision problem using tree search.

Task: {task}
{context_str}
Current state: {node.state}
Path from root: {path_str}

Generate 3-5 possible next actions or decisions from this state.
Each action should be distinct and lead to a different outcome.

Format your response as:
Action 1: [brief description]
Action 2: [brief description]
Action 3: [brief description]
... and so on.

Generate possible actions:"""

    def _build_simulation_prompt(
        self,
        task: str,
        context: dict[str, Any],
        node: MCTSNode,
    ) -> str:
        """Build prompt for simulating outcome from a node.

        Args:
            task: The original task.
            context: Context information.
            node: The node to simulate from.

        Returns:
            Prompt string.
        """
        context_str = ""
        if context:
            context_str = "\n\nContext:\n"
            for key, value in context.items():
                context_str += f"- {key}: {value}\n"

        parent_path = self._get_path_to_root(node)
        path_str = " -> ".join(parent_path) if parent_path else "root"

        return f"""You are simulating the outcome of a decision path.

Task: {task}
{context_str}
Decision path: {path_str}
Current decision: {node.state}

Simulate the outcome of following this decision path. Consider:
1. Likely results and consequences
2. Potential risks and benefits
3. Overall quality of this path

Rate the outcome on a scale of 0 to 100, where:
- 0-30: Poor outcome
- 31-60: Moderate outcome
- 61-80: Good outcome
- 81-100: Excellent outcome

Format your response as:
Score: [0-100]
Reasoning: [brief explanation]

Evaluate:"""

    def _build_solution_prompt(
        self,
        task: str,
        context: dict[str, Any],
        best_path: list[str],
    ) -> str:
        """Build prompt for generating final solution from best path.

        Args:
            task: The original task.
            context: Context information.
            best_path: The best path found.

        Returns:
            Prompt string.
        """
        context_str = ""
        if context:
            context_str = "\n\nContext:\n"
            for key, value in context.items():
                context_str += f"- {key}: {value}\n"

        path_str = "\n".join(f"  {i+1}. {step}" for i, step in enumerate(best_path))

        return f"""Based on tree search analysis, the optimal decision path is:

Task: {task}
{context_str}
Optimal path:
{path_str}

Provide a comprehensive solution following this path. Include:
1. Summary of the recommended approach
2. Key steps for implementation
3. Expected outcomes and benefits

Solution:"""

    def _get_path_to_root(self, node: MCTSNode) -> list[str]:
        """Get the path from root to this node.

        Args:
            node: The target node.

        Returns:
            List of state descriptions from root to node.
        """
        path = []
        current = node

        while current.parent_id is not None:
            if current.action:
                path.append(current.action)
            else:
                path.append(current.state)
            parent = self._nodes.get(current.parent_id)
            if parent is None:
                break
            current = parent

        path.reverse()
        return path

    async def _expand_node(
        self,
        task: str,
        context: dict[str, Any],
        node: MCTSNode,
    ) -> tuple[list[MCTSNode], int]:
        """Expand a node with new child actions.

        Args:
            task: The original task.
            context: Context information.
            node: The node to expand.

        Returns:
            Tuple of (list of new child nodes, tokens_used).
        """
        if node.depth >= self._max_depth:
            node.is_terminal = True
            return [], 0

        prompt = self._build_expansion_prompt(task, context, node)

        response, tokens_used = await self._call_llm(
            prompt=prompt,
            temperature=0.8,
            max_tokens=500,
        )

        # Parse actions from response
        import re

        action_pattern = r"[Aa]ction\s*\d*:?\s*(.+?)(?=[Aa]ction\s*\d*:|$)"
        actions = re.findall(action_pattern, response, re.DOTALL)

        if not actions:
            # Fallback: split by lines
            actions = [line.strip() for line in response.split("\n") if line.strip()]

        children = []
        for action in actions[:5]:  # Max 5 children
            action_text = action.strip()
            if len(action_text) < 5:
                continue

            child = MCTSNode(
                node_id=self._generate_node_id(),
                state=action_text,
                parent_id=node.node_id,
                depth=node.depth + 1,
                action=action_text,
            )
            self._nodes[child.node_id] = child
            node.children.append(child.node_id)
            children.append(child)

        return children, tokens_used

    async def _simulate(
        self,
        task: str,
        context: dict[str, Any],
        node: MCTSNode,
    ) -> tuple[float, int]:
        """Simulate outcome from a node.

        Args:
            task: The original task.
            context: Context information.
            node: The node to simulate from.

        Returns:
            Tuple of (value 0-1, tokens_used).
        """
        prompt = self._build_simulation_prompt(task, context, node)

        response, tokens_used = await self._call_llm(
            prompt=prompt,
            temperature=0.5,
            max_tokens=200,
        )

        # Parse score from response
        import re

        score_match = re.search(r"[Ss]core:?\s*(\d+)", response)
        if score_match:
            score = int(score_match.group(1))
            value = score / 100.0
        else:
            # Fallback: random score
            value = random.uniform(0.3, 0.7)

        node.simulation_results.append(value)
        return value, tokens_used

    def _backpropagate(self, node: MCTSNode, value: float) -> None:
        """Backpropagate simulation result up the tree.

        Args:
            node: Starting node.
            value: Value to propagate.
        """
        current = node
        while current is not None:
            current.visits += 1
            current.value += value
            if current.parent_id:
                current = self._nodes.get(current.parent_id)
            else:
                break

    def _select_best_child(self, node: MCTSNode) -> Optional[MCTSNode]:
        """Select best child using UCB1.

        Args:
            node: Parent node.

        Returns:
            Best child node or None.
        """
        if not node.children:
            return None

        best_score = -float("inf")
        best_child = None

        for child_id in node.children:
            child = self._nodes.get(child_id)
            if child is None:
                continue

            # Calculate UCB1 with parent's visits
            if child.visits == 0:
                score = float("inf")
            else:
                exploitation = child.value / child.visits
                exploration = UCB1_EXPLORATION * math.sqrt(
                    math.log(node.visits + 1) / child.visits
                )
                score = exploitation + exploration

            if score > best_score:
                best_score = score
                best_child = child

        return best_child

    def _get_best_path(self, root: MCTSNode) -> list[str]:
        """Get the best path from root based on average values.

        Args:
            root: Root node.

        Returns:
            List of actions in the best path.
        """
        path = []
        current = root

        while current.children:
            # Select child with highest average value
            best_child = None
            best_avg = -float("inf")

            for child_id in current.children:
                child = self._nodes.get(child_id)
                if child and child.visits > 0:
                    avg = child.value / child.visits
                    if avg > best_avg:
                        best_avg = avg
                        best_child = child

            if best_child is None:
                break

            if best_child.action:
                path.append(best_child.action)
            current = best_child

        return path

    async def execute(self, task: str, context: dict[str, Any]) -> StrategyResult:
        """Execute MCTS reasoning on a task.

        Builds a search tree through simulation and backpropagation,
        then selects the best path for the final answer.

        Args:
            task: The task/question to reason about.
            context: Context information for the task.

        Returns:
            StrategyResult with answer, confidence (0.8-0.95), and trace.

        Example:
            >>> result = await strategy.execute(
            ...     "Design product launch strategy",
            ...     {"timeline": "3 months"},
            ... )
        """
        started_at = utc_now()
        start_time = time.time()
        total_tokens = 0

        try:
            # Reset tree
            self._reset_tree()

            # Create root node
            root = MCTSNode(
                node_id=self._generate_node_id(),
                state=f"Root: {task}",
                depth=0,
            )
            self._nodes[root.node_id] = root

            reasoning_trace = [f"Starting MCTS with {self._num_simulations} simulations"]

            # Run MCTS iterations
            for sim_num in range(self._num_simulations):
                # Selection: traverse tree using UCB1
                current = root
                while current.children and not current.is_terminal:
                    next_node = self._select_best_child(current)
                    if next_node is None:
                        break
                    current = next_node

                # Expansion: if not terminal and not fully expanded
                if not current.is_terminal and not current.children:
                    new_children, expand_tokens = await self._expand_node(
                        task, context, current
                    )
                    total_tokens += expand_tokens

                    if new_children:
                        current = new_children[0]  # Select first child for simulation

                # Simulation: evaluate current node
                value, sim_tokens = await self._simulate(task, context, current)
                total_tokens += sim_tokens

                # Backpropagation: update values up the tree
                self._backpropagate(current, value)

                # Log progress periodically
                if (sim_num + 1) % 10 == 0:
                    reasoning_trace.append(
                        f"Completed {sim_num + 1}/{self._num_simulations} simulations"
                    )

            # Get best path
            best_path = self._get_best_path(root)
            reasoning_trace.append(f"Best path found with {len(best_path)} decisions")

            # Generate final solution
            solution_prompt = self._build_solution_prompt(task, context, best_path)
            solution, solution_tokens = await self._call_llm(
                prompt=solution_prompt,
                temperature=self._config.temperature,
                max_tokens=self._config.max_tokens // 2,
            )
            total_tokens += solution_tokens

            # Calculate confidence based on best path's visit count and value
            if best_path:
                # Find best child stats
                best_child_id = root.children[0] if root.children else None
                for child_id in root.children:
                    child = self._nodes.get(child_id)
                    if child and child.action == best_path[0]:
                        best_child_id = child_id
                        break

                if best_child_id:
                    best_child = self._nodes.get(best_child_id)
                    if best_child and best_child.visits > 0:
                        avg_value = best_child.value / best_child.visits
                        # Scale to 0.8-0.95 range
                        confidence = 0.8 + (avg_value * 0.15)
                    else:
                        confidence = 0.8
                else:
                    confidence = 0.8
            else:
                confidence = 0.75

            confidence = min(max(confidence, 0.7), 0.95)

            # Build tree summary for metadata
            tree_summary = {
                "total_nodes": len(self._nodes),
                "root_visits": root.visits,
                "max_depth_reached": max(n.depth for n in self._nodes.values()),
                "simulations_run": self._num_simulations,
            }

            # Add top-level decisions to trace
            for child_id in root.children:
                child = self._nodes.get(child_id)
                if child:
                    reasoning_trace.append(
                        f"  Option: {child.state[:50]}... "
                        f"(visits={child.visits}, avg={child.average_value:.2f})"
                    )

            reasoning_trace.append(f"Selected path: {' -> '.join(best_path[:3])}")

            execution_time = time.time() - start_time

            # Record metrics
            self._record_execution(True, total_tokens, execution_time)

            return StrategyResult(
                answer=solution,
                confidence=confidence,
                reasoning_trace=reasoning_trace,
                tokens_used=total_tokens,
                execution_time_seconds=execution_time,
                model=self._settings.llm.model,
                metadata={
                    "strategy": self.name,
                    "simulations": self._num_simulations,
                    "tree_summary": tree_summary,
                    "best_path": best_path,
                    "top_nodes": [
                        self._nodes[cid].to_dict()
                        for cid in root.children[:5]
                        if cid in self._nodes
                    ],
                },
                success=True,
                started_at=started_at,
                completed_at=utc_now(),
            )

        except Exception as e:
            execution_time = time.time() - start_time
            self._record_execution(False, total_tokens, execution_time)

            return StrategyResult.from_error(
                error=str(e),
                strategy_name=self.name,
                execution_time=execution_time,
                tokens_used=total_tokens,
            )


# =============================================================================
# Exports
# =============================================================================

__all__ = ["MCTSStrategy", "MCTSNode"]
