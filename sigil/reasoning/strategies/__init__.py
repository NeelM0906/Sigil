"""Reasoning strategy implementations for Sigil v2.

This module contains concrete reasoning strategy implementations:

Strategies:
    - DirectStrategy: Single LLM call for simple tasks (0.0-0.3)
    - ChainOfThoughtStrategy: Step-by-step reasoning (0.3-0.5)
    - TreeOfThoughtsStrategy: Multi-path exploration (0.5-0.7)
    - ReActStrategy: Thought-Action-Observation loop (0.7-0.9)
    - MCTSStrategy: Monte Carlo Tree Search (0.9-1.0)

Each strategy is optimized for a specific complexity range and provides:
- Execution method with task and context
- Confidence estimation
- Token tracking
- Reasoning trace generation

Example:
    >>> from sigil.reasoning.strategies import DirectStrategy, ChainOfThoughtStrategy
    >>>
    >>> strategy = ChainOfThoughtStrategy()
    >>> result = await strategy.execute(
    ...     task="What is 15% of 80?",
    ...     context={},
    ... )
    >>> print(result.answer)
"""

from sigil.reasoning.strategies.base import (
    # Data classes
    StrategyResult,
    StrategyConfig,
    # Base class
    BaseReasoningStrategy,
    # Constants
    COMPLEXITY_RANGES,
    TOKEN_BUDGETS,
    # Utilities
    utc_now,
)

from sigil.reasoning.strategies.direct import DirectStrategy
from sigil.reasoning.strategies.chain_of_thought import ChainOfThoughtStrategy
from sigil.reasoning.strategies.tree_of_thoughts import TreeOfThoughtsStrategy, ThoughtNode
from sigil.reasoning.strategies.react import ReActStrategy, ReActStep, StepType
from sigil.reasoning.strategies.mcts import MCTSStrategy, MCTSNode
from sigil.reasoning.strategies.function_calling import (
    FunctionCallingStrategy,
    FunctionCallingStep,
    ToolUseBlock,
)
from sigil.reasoning.strategies.utils import (
    extract_tool_outputs,
    format_tool_results_section,
    format_tavily_results,
    build_tool_aware_context_string,
    DEFAULT_MAX_CHARS_PER_RESULT,
    TRUNCATION_MARKER,
)


__all__ = [
    # Base classes
    "BaseReasoningStrategy",
    "StrategyResult",
    "StrategyConfig",
    # Constants
    "COMPLEXITY_RANGES",
    "TOKEN_BUDGETS",
    # Utilities
    "utc_now",
    # Strategy implementations
    "DirectStrategy",
    "ChainOfThoughtStrategy",
    "TreeOfThoughtsStrategy",
    "ReActStrategy",
    "MCTSStrategy",
    "FunctionCallingStrategy",
    # Supporting types
    "ThoughtNode",
    "ReActStep",
    "StepType",
    "MCTSNode",
    "FunctionCallingStep",
    "ToolUseBlock",
    # Utility functions
    "extract_tool_outputs",
    "format_tool_results_section",
    "format_tavily_results",
    "build_tool_aware_context_string",
    "DEFAULT_MAX_CHARS_PER_RESULT",
    "TRUNCATION_MARKER",
]
