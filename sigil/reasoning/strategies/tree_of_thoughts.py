"""Tree of Thoughts reasoning strategy for Sigil v2.

This module implements the TreeOfThoughtsStrategy, which explores multiple
reasoning paths in parallel and selects the best approach.

Classes:
    ThoughtNode: A node in the thought tree.
    TreeOfThoughtsStrategy: Multi-path exploration for complex tasks.

Example:
    >>> from sigil.reasoning.strategies.tree_of_thoughts import TreeOfThoughtsStrategy
    >>>
    >>> strategy = TreeOfThoughtsStrategy()
    >>> result = await strategy.execute(
    ...     task="Design a marketing strategy for a new product",
    ...     context={"product": "AI writing assistant"},
    ... )
"""

from __future__ import annotations

import asyncio
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

TOT_MIN_COMPLEXITY = 0.5
TOT_MAX_COMPLEXITY = 0.7
TOT_MIN_TOKENS = 800
TOT_MAX_TOKENS = 2000

DEFAULT_NUM_BRANCHES = 3
"""Default number of thought branches to explore."""

MAX_BRANCHES = 5
"""Maximum number of parallel branches."""


# =============================================================================
# Thought Node
# =============================================================================


@dataclass
class ThoughtNode:
    """A node in the thought tree.

    Represents a single reasoning approach or thought branch.

    Attributes:
        approach_id: Unique identifier for this approach.
        description: Description of the approach.
        reasoning: The reasoning/thought process.
        evaluation_score: Score from evaluation (0.0-1.0).
        depth: Depth in the tree.
        parent_id: Parent node ID (if any).
        tokens_used: Tokens used for this node.
    """

    approach_id: str
    description: str
    reasoning: str = ""
    evaluation_score: float = 0.0
    depth: int = 0
    parent_id: Optional[str] = None
    tokens_used: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert node to dictionary."""
        return {
            "approach_id": self.approach_id,
            "description": self.description,
            "reasoning": self.reasoning[:200] + "..." if len(self.reasoning) > 200 else self.reasoning,
            "evaluation_score": self.evaluation_score,
            "depth": self.depth,
            "parent_id": self.parent_id,
            "tokens_used": self.tokens_used,
        }


# =============================================================================
# Tree of Thoughts Strategy
# =============================================================================


class TreeOfThoughtsStrategy(BaseReasoningStrategy):
    """Multi-path exploration reasoning strategy.

    TreeOfThoughtsStrategy generates multiple possible approaches to a task,
    evaluates each approach, and selects the best one for refinement.
    This enables exploration of the solution space and selection of
    the most promising path.

    Characteristics:
        - Complexity range: 0.5-0.7
        - Token budget: 800-2000
        - Reasoning trace: All paths explored with selection rationale
        - Confidence: 0.7-0.9
        - Best for: Creative tasks, design decisions, open-ended problems

    The strategy follows three phases:
    1. Generation: Generate 3-5 possible approaches
    2. Evaluation: Score each approach
    3. Refinement: Elaborate on the best approach

    Attributes:
        name: "tree_of_thoughts"
        num_branches: Number of branches to explore (default 3).
        config: Strategy configuration with ToT-specific defaults.

    Example:
        >>> strategy = TreeOfThoughtsStrategy(num_branches=4)
        >>> result = await strategy.execute(
        ...     task="Design a user onboarding flow",
        ...     context={"app_type": "fitness"},
        ... )
    """

    def __init__(
        self,
        config: Optional[StrategyConfig] = None,
        event_store: Optional[EventStore] = None,
        token_tracker: Optional[TokenTracker] = None,
        num_branches: int = DEFAULT_NUM_BRANCHES,
    ) -> None:
        """Initialize TreeOfThoughtsStrategy.

        Args:
            config: Optional strategy configuration.
            event_store: Optional event store for audit trails.
            token_tracker: Optional token tracker for budget.
            num_branches: Number of branches to explore (1-5).
        """
        if config is None:
            config = StrategyConfig(
                min_complexity=TOT_MIN_COMPLEXITY,
                max_complexity=TOT_MAX_COMPLEXITY,
                min_tokens=TOT_MIN_TOKENS,
                max_tokens=TOT_MAX_TOKENS,
                timeout_seconds=120.0,
            )
        super().__init__(config, event_store, token_tracker)
        self._num_branches = min(max(num_branches, 1), MAX_BRANCHES)

    @property
    def name(self) -> str:
        """Return the strategy name."""
        return "tree_of_thoughts"

    def _build_generation_prompt(
        self,
        task: str,
        context: dict[str, Any],
        num_approaches: int,
    ) -> str:
        """Build prompt for generating multiple approaches.

        Args:
            task: The task to execute.
            context: Context information.
            num_approaches: Number of approaches to generate.

        Returns:
            Prompt string.
        """
        context_str = ""
        if context:
            context_str = "\n\nContext:\n"
            for key, value in context.items():
                context_str += f"- {key}: {value}\n"

        return f"""You are a strategic thinker. Generate {num_approaches} different approaches to solve the following task.

Each approach should be distinct and explore a different angle or method.
{context_str}
Task: {task}

Generate exactly {num_approaches} approaches in the following format:

Approach 1: [Title]
Description: [Brief description of this approach]

Approach 2: [Title]
Description: [Brief description of this approach]

... and so on.

Generate {num_approaches} distinct approaches:"""

    def _build_evaluation_prompt(
        self,
        task: str,
        approaches: list[ThoughtNode],
    ) -> str:
        """Build prompt for evaluating approaches.

        Args:
            task: The original task.
            approaches: List of approach nodes to evaluate.

        Returns:
            Prompt string.
        """
        approaches_str = ""
        for i, approach in enumerate(approaches, 1):
            approaches_str += f"\nApproach {i}: {approach.description}\n"

        return f"""Evaluate the following approaches for solving this task:

Task: {task}

Approaches:{approaches_str}

For each approach, provide a score from 0 to 10 based on:
- Feasibility: How practical is this approach?
- Effectiveness: How likely is it to achieve the goal?
- Efficiency: How resource-efficient is it?

Format your response as:
Approach 1: [score]/10 - [brief justification]
Approach 2: [score]/10 - [brief justification]
... and so on.

Then state which approach is best and why.

Evaluation:"""

    def _build_refinement_prompt(
        self,
        task: str,
        best_approach: ThoughtNode,
        context: dict[str, Any],
    ) -> str:
        """Build prompt for refining the best approach.

        Args:
            task: The original task.
            best_approach: The selected best approach.
            context: Context information.

        Returns:
            Prompt string.
        """
        context_str = ""
        if context:
            context_str = "\n\nContext:\n"
            for key, value in context.items():
                context_str += f"- {key}: {value}\n"

        return f"""You selected the following approach as the best for the task:

Task: {task}

Selected Approach: {best_approach.description}
{context_str}
Now, elaborate on this approach in detail. Provide:
1. A step-by-step implementation plan
2. Key considerations and potential challenges
3. Success criteria

Detailed solution:"""

    async def _generate_approaches(
        self,
        task: str,
        context: dict[str, Any],
    ) -> tuple[list[ThoughtNode], int]:
        """Generate multiple approaches in parallel.

        Args:
            task: The task to generate approaches for.
            context: Context information.

        Returns:
            Tuple of (list of ThoughtNode, tokens_used).
        """
        prompt = self._build_generation_prompt(task, context, self._num_branches)

        # In production, this could be parallelized for independent generation
        response, tokens_used = await self._call_llm(
            prompt=prompt,
            temperature=0.8,  # Higher temperature for diversity
            max_tokens=self._config.max_tokens // 2,
        )

        # Parse approaches from response
        approaches = []
        import re

        pattern = r"[Aa]pproach\s*(\d+):?\s*([^\n]+)(?:\n[Dd]escription:?\s*([^\n]+))?"
        matches = re.findall(pattern, response)

        for i, match in enumerate(matches[: self._num_branches]):
            approach_num, title, description = match
            desc = f"{title.strip()}: {description.strip()}" if description else title.strip()
            approaches.append(
                ThoughtNode(
                    approach_id=f"approach_{i + 1}",
                    description=desc,
                    depth=0,
                )
            )

        # Ensure we have at least one approach
        if not approaches:
            # Fallback: create a single approach from the response
            approaches.append(
                ThoughtNode(
                    approach_id="approach_1",
                    description=response[:200].strip(),
                    depth=0,
                )
            )

        return approaches, tokens_used

    async def _evaluate_approaches(
        self,
        task: str,
        approaches: list[ThoughtNode],
    ) -> tuple[ThoughtNode, int]:
        """Evaluate approaches and select the best one.

        Args:
            task: The original task.
            approaches: List of approaches to evaluate.

        Returns:
            Tuple of (best ThoughtNode, tokens_used).
        """
        if len(approaches) == 1:
            approaches[0].evaluation_score = 0.8
            return approaches[0], 0

        prompt = self._build_evaluation_prompt(task, approaches)

        response, tokens_used = await self._call_llm(
            prompt=prompt,
            temperature=0.3,  # Lower temperature for consistent evaluation
            max_tokens=self._config.max_tokens // 4,
        )

        # Parse scores from response
        import re

        score_pattern = r"[Aa]pproach\s*(\d+):?\s*(\d+(?:\.\d+)?)\s*/\s*10"
        matches = re.findall(score_pattern, response)

        for match in matches:
            approach_num, score = match
            idx = int(approach_num) - 1
            if 0 <= idx < len(approaches):
                approaches[idx].evaluation_score = float(score) / 10.0

        # Select best approach
        best_approach = max(approaches, key=lambda a: a.evaluation_score)

        # Store evaluation reasoning
        best_approach.reasoning = response

        return best_approach, tokens_used

    async def _refine_approach(
        self,
        task: str,
        best_approach: ThoughtNode,
        context: dict[str, Any],
    ) -> tuple[str, int]:
        """Refine the best approach into a detailed solution.

        Args:
            task: The original task.
            best_approach: The selected best approach.
            context: Context information.

        Returns:
            Tuple of (refined solution, tokens_used).
        """
        prompt = self._build_refinement_prompt(task, best_approach, context)

        response, tokens_used = await self._call_llm(
            prompt=prompt,
            temperature=self._config.temperature,
            max_tokens=self._config.max_tokens // 2,
        )

        return response, tokens_used

    async def execute(self, task: str, context: dict[str, Any]) -> StrategyResult:
        """Execute tree-of-thoughts reasoning on a task.

        Generates multiple approaches, evaluates them, and refines
        the best approach into a detailed solution.

        Args:
            task: The task/question to reason about.
            context: Context information for the task.

        Returns:
            StrategyResult with answer, confidence (0.7-0.9), and trace.

        Example:
            >>> result = await strategy.execute(
            ...     "Design a notification system",
            ...     {"users": "10000"},
            ... )
        """
        started_at = utc_now()
        start_time = time.time()
        total_tokens = 0

        try:
            # Phase 1: Generate approaches
            approaches, gen_tokens = await self._generate_approaches(task, context)
            total_tokens += gen_tokens

            reasoning_trace = [
                f"Generated {len(approaches)} possible approaches"
            ]
            for approach in approaches:
                reasoning_trace.append(f"  - {approach.description}")

            # Phase 2: Evaluate approaches
            best_approach, eval_tokens = await self._evaluate_approaches(
                task, approaches
            )
            total_tokens += eval_tokens

            reasoning_trace.append(
                f"Evaluated approaches, selected: {best_approach.description} "
                f"(score: {best_approach.evaluation_score:.2f})"
            )

            # Phase 3: Refine best approach
            refined_solution, refine_tokens = await self._refine_approach(
                task, best_approach, context
            )
            total_tokens += refine_tokens

            reasoning_trace.append("Refined best approach into detailed solution")

            # Calculate confidence based on evaluation score
            confidence = 0.7 + (best_approach.evaluation_score * 0.2)
            confidence = min(confidence, 0.9)

            execution_time = time.time() - start_time

            # Record metrics
            self._record_execution(True, total_tokens, execution_time)

            return StrategyResult(
                answer=refined_solution,
                confidence=confidence,
                reasoning_trace=reasoning_trace,
                tokens_used=total_tokens,
                execution_time_seconds=execution_time,
                model=self._settings.llm.model,
                metadata={
                    "strategy": self.name,
                    "approaches_generated": len(approaches),
                    "best_approach": best_approach.description,
                    "best_score": best_approach.evaluation_score,
                    "all_approaches": [a.to_dict() for a in approaches],
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

__all__ = ["TreeOfThoughtsStrategy", "ThoughtNode"]
