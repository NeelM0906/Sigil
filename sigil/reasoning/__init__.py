"""Reasoning module for Sigil v2 framework.

This module implements Phase 5 hierarchical reasoning with multiple strategies:
- Strategy selection based on task complexity
- Strategy execution with fallback chains
- Performance tracking for strategy optimization

Key Components:
    - ReasoningManager: Orchestrates strategy selection and execution
    - Strategies: Direct, CoT, ToT, ReAct, MCTS
    - Prompts: Templates for each strategy

Strategy Selection by Complexity:
    - 0.0-0.3: DirectStrategy (single LLM call)
    - 0.3-0.5: ChainOfThoughtStrategy (step-by-step)
    - 0.5-0.7: TreeOfThoughtsStrategy (multi-path)
    - 0.7-0.9: ReActStrategy (tool integration)
    - 0.9-1.0: MCTSStrategy (tree search)

Fallback Chain:
    MCTS -> ReAct -> ToT -> CoT -> Direct

Example:
    >>> from sigil.reasoning import ReasoningManager
    >>>
    >>> manager = ReasoningManager()
    >>> result = await manager.execute(
    ...     task="Analyze market trends",
    ...     context={"industry": "SaaS"},
    ...     complexity=0.65,
    ... )
    >>> print(f"Answer: {result.answer}, Confidence: {result.confidence}")
"""

from sigil.reasoning.manager import (
    ReasoningManager,
    StrategyMetrics,
    create_strategy_selected_event,
    create_reasoning_completed_event,
    create_strategy_fallback_event,
)

from sigil.reasoning.strategies import (
    # Base classes
    BaseReasoningStrategy,
    StrategyResult,
    StrategyConfig,
    # Constants
    COMPLEXITY_RANGES,
    TOKEN_BUDGETS,
    # Strategy implementations
    DirectStrategy,
    ChainOfThoughtStrategy,
    TreeOfThoughtsStrategy,
    ReActStrategy,
    MCTSStrategy,
    # Supporting types
    ThoughtNode,
    ReActStep,
    MCTSNode,
)

from sigil.reasoning.prompts import (
    PromptTemplate,
    DirectPrompts,
    ChainOfThoughtPrompts,
    TreeOfThoughtsPrompts,
    ReActPrompts,
    MCTSPrompts,
)


__all__ = [
    # Main class
    "ReasoningManager",
    "StrategyMetrics",
    # Base classes
    "BaseReasoningStrategy",
    "StrategyResult",
    "StrategyConfig",
    # Constants
    "COMPLEXITY_RANGES",
    "TOKEN_BUDGETS",
    # Strategy implementations
    "DirectStrategy",
    "ChainOfThoughtStrategy",
    "TreeOfThoughtsStrategy",
    "ReActStrategy",
    "MCTSStrategy",
    # Supporting types
    "ThoughtNode",
    "ReActStep",
    "MCTSNode",
    # Prompts
    "PromptTemplate",
    "DirectPrompts",
    "ChainOfThoughtPrompts",
    "TreeOfThoughtsPrompts",
    "ReActPrompts",
    "MCTSPrompts",
    # Event creators
    "create_strategy_selected_event",
    "create_reasoning_completed_event",
    "create_strategy_fallback_event",
]
