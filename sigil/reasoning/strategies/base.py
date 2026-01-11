"""Base classes for reasoning strategies in Sigil v2.

This module defines the abstract base class for reasoning strategies and
the StrategyResult dataclass for returning execution results.

Classes:
    StrategyResult: Result from strategy execution.
    BaseReasoningStrategy: Abstract base class for all strategies.
    StrategyConfig: Configuration for strategy execution.

Example:
    >>> class MyStrategy(BaseReasoningStrategy):
    ...     async def execute(self, task: str, context: dict) -> StrategyResult:
    ...         # Implementation here
    ...         pass
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

from sigil.config import get_settings
from sigil.state.store import EventStore
from sigil.telemetry.tokens import TokenTracker


# =============================================================================
# Constants
# =============================================================================

# Strategy complexity ranges
COMPLEXITY_RANGES = {
    "direct": (0.0, 0.3),
    "chain_of_thought": (0.3, 0.5),
    "tree_of_thoughts": (0.5, 0.7),
    "react": (0.7, 0.9),
    "mcts": (0.9, 1.0),
}

# Token budgets by strategy
TOKEN_BUDGETS = {
    "direct": (100, 300),
    "chain_of_thought": (300, 800),
    "tree_of_thoughts": (800, 2000),
    "react": (1000, 3000),
    "mcts": (2000, 5000),
}


def utc_now() -> datetime:
    """Get the current UTC datetime."""
    return datetime.now(timezone.utc)


# =============================================================================
# Strategy Result
# =============================================================================


@dataclass
class StrategyResult:
    """Result from strategy execution.

    Contains the answer, confidence, reasoning trace, and execution metrics.

    Attributes:
        answer: The final answer/result from the strategy.
        confidence: Confidence score (0.0-1.0).
        reasoning_trace: List of reasoning steps/thoughts.
        tokens_used: Total tokens consumed.
        execution_time_seconds: Total execution time.
        model: The LLM model used.
        metadata: Additional strategy-specific metadata.
        success: Whether the strategy succeeded.
        error: Error message if failed.
        started_at: Execution start time.
        completed_at: Execution completion time.

    Example:
        >>> result = StrategyResult(
        ...     answer="The answer is 42",
        ...     confidence=0.85,
        ...     reasoning_trace=["Step 1: Analyzed input", "Step 2: Computed result"],
        ...     tokens_used=150,
        ...     execution_time_seconds=1.5,
        ...     model="claude-3-sonnet",
        ... )
    """

    answer: str
    confidence: float
    reasoning_trace: list[str]
    tokens_used: int
    execution_time_seconds: float
    model: str
    metadata: dict[str, Any] = field(default_factory=dict)
    success: bool = True
    error: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    def __post_init__(self) -> None:
        """Validate result values after initialization."""
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence must be 0.0-1.0, got {self.confidence}")

    @classmethod
    def from_error(
        cls,
        error: str,
        strategy_name: str,
        execution_time: float = 0.0,
        tokens_used: int = 0,
    ) -> "StrategyResult":
        """Create a failed result from an error.

        Args:
            error: Error message.
            strategy_name: Name of the strategy that failed.
            execution_time: Time spent before failure.
            tokens_used: Tokens used before failure.

        Returns:
            StrategyResult with success=False.
        """
        return cls(
            answer="",
            confidence=0.0,
            reasoning_trace=[f"Strategy {strategy_name} failed: {error}"],
            tokens_used=tokens_used,
            execution_time_seconds=execution_time,
            model="unknown",
            success=False,
            error=error,
            completed_at=utc_now(),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert result to dictionary for serialization."""
        return {
            "answer": self.answer,
            "confidence": self.confidence,
            "reasoning_trace": self.reasoning_trace,
            "tokens_used": self.tokens_used,
            "execution_time_seconds": self.execution_time_seconds,
            "model": self.model,
            "metadata": self.metadata,
            "success": self.success,
            "error": self.error,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
        }


# =============================================================================
# Strategy Configuration
# =============================================================================


@dataclass
class StrategyConfig:
    """Configuration for strategy execution.

    Attributes:
        min_complexity: Minimum complexity for this strategy.
        max_complexity: Maximum complexity for this strategy.
        min_tokens: Minimum token budget.
        max_tokens: Maximum token budget.
        max_retries: Maximum retry attempts.
        timeout_seconds: Execution timeout.
        temperature: LLM temperature setting.
        model_override: Optional model override.
    """

    min_complexity: float = 0.0
    max_complexity: float = 1.0
    min_tokens: int = 100
    max_tokens: int = 1000
    max_retries: int = 3
    timeout_seconds: float = 60.0
    temperature: float = 0.7
    model_override: Optional[str] = None

    def is_complexity_in_range(self, complexity: float) -> bool:
        """Check if complexity is within this strategy's range."""
        return self.min_complexity <= complexity <= self.max_complexity


# =============================================================================
# Base Reasoning Strategy
# =============================================================================


class BaseReasoningStrategy(ABC):
    """Abstract base class for reasoning strategies.

    All reasoning strategies must inherit from this class and implement
    the execute() method. The base class provides common functionality
    for token tracking, event emission, and configuration.

    Attributes:
        name: Strategy name (e.g., "direct", "chain_of_thought").
        config: Strategy configuration.
        event_store: Optional event store for audit trails.
        token_tracker: Optional token tracker for budget.

    Example:
        >>> class MyStrategy(BaseReasoningStrategy):
        ...     @property
        ...     def name(self) -> str:
        ...         return "my_strategy"
        ...
        ...     async def execute(self, task: str, context: dict) -> StrategyResult:
        ...         # Implementation
        ...         pass
    """

    def __init__(
        self,
        config: Optional[StrategyConfig] = None,
        event_store: Optional[EventStore] = None,
        token_tracker: Optional[TokenTracker] = None,
    ) -> None:
        """Initialize the strategy.

        Args:
            config: Optional strategy configuration.
            event_store: Optional event store for audit trails.
            token_tracker: Optional token tracker for budget.
        """
        self._config = config or self._default_config()
        self._event_store = event_store
        self._token_tracker = token_tracker
        self._settings = get_settings()
        self._execution_count = 0
        self._success_count = 0
        self._total_tokens = 0
        self._total_time = 0.0

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the strategy name."""
        pass

    @property
    def config(self) -> StrategyConfig:
        """Return the strategy configuration."""
        return self._config

    def _default_config(self) -> StrategyConfig:
        """Return default configuration for this strategy."""
        # Get ranges from constants based on strategy name
        complexity_range = COMPLEXITY_RANGES.get(self.name, (0.0, 1.0))
        token_range = TOKEN_BUDGETS.get(self.name, (100, 1000))

        return StrategyConfig(
            min_complexity=complexity_range[0],
            max_complexity=complexity_range[1],
            min_tokens=token_range[0],
            max_tokens=token_range[1],
        )

    @abstractmethod
    async def execute(self, task: str, context: dict[str, Any]) -> StrategyResult:
        """Execute the reasoning strategy on a task.

        Args:
            task: The task/question to reason about.
            context: Context information for the task.

        Returns:
            StrategyResult with answer, confidence, and trace.

        Raises:
            ReasoningError: If execution fails.
        """
        pass

    async def _call_llm(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> tuple[str, int]:
        """Call the LLM with a prompt.

        This is a placeholder implementation. In production, this would
        use the actual LLM API.

        Args:
            prompt: The prompt to send.
            system_prompt: Optional system prompt.
            temperature: Optional temperature override.
            max_tokens: Optional max tokens override.

        Returns:
            Tuple of (response_text, tokens_used).
        """
        # Placeholder implementation
        # In production, this would call the actual LLM API
        response = f"[LLM Response to: {prompt[:100]}...]"
        tokens_used = len(prompt) // 4 + len(response) // 4

        # Track tokens
        if self._token_tracker:
            self._token_tracker.record_usage(
                input_tokens=len(prompt) // 4,
                output_tokens=len(response) // 4,
            )

        return response, tokens_used

    def _estimate_confidence(self, response: str, trace: list[str]) -> float:
        """Estimate confidence from response characteristics.

        Args:
            response: The LLM response.
            trace: Reasoning trace.

        Returns:
            Confidence score (0.0-1.0).
        """
        # Base confidence varies by strategy
        base_confidence_map = {
            "direct": 0.55,
            "chain_of_thought": 0.70,
            "tree_of_thoughts": 0.80,
            "react": 0.70,
            "mcts": 0.85,
        }
        base = base_confidence_map.get(self.name, 0.5)

        # Adjust based on response characteristics
        adjustments = 0.0

        # Longer, more detailed responses suggest higher confidence
        if len(response) > 200:
            adjustments += 0.05
        if len(response) > 500:
            adjustments += 0.05

        # More reasoning steps suggest higher confidence
        if len(trace) > 3:
            adjustments += 0.05
        if len(trace) > 5:
            adjustments += 0.05

        # Uncertainty markers reduce confidence
        uncertainty_markers = ["uncertain", "unclear", "might", "possibly", "maybe"]
        for marker in uncertainty_markers:
            if marker in response.lower():
                adjustments -= 0.05

        return min(max(base + adjustments, 0.1), 0.95)

    def get_metrics(self) -> dict[str, Any]:
        """Get strategy execution metrics.

        Returns:
            Dictionary with execution statistics.
        """
        return {
            "name": self.name,
            "execution_count": self._execution_count,
            "success_count": self._success_count,
            "success_rate": (
                self._success_count / self._execution_count
                if self._execution_count > 0
                else 0.0
            ),
            "total_tokens": self._total_tokens,
            "total_time_seconds": self._total_time,
            "avg_tokens_per_execution": (
                self._total_tokens / self._execution_count
                if self._execution_count > 0
                else 0
            ),
            "avg_time_per_execution": (
                self._total_time / self._execution_count
                if self._execution_count > 0
                else 0.0
            ),
        }

    def reset_metrics(self) -> None:
        """Reset execution metrics."""
        self._execution_count = 0
        self._success_count = 0
        self._total_tokens = 0
        self._total_time = 0.0

    def _record_execution(
        self,
        success: bool,
        tokens_used: int,
        execution_time: float,
    ) -> None:
        """Record execution statistics.

        Args:
            success: Whether execution succeeded.
            tokens_used: Tokens consumed.
            execution_time: Execution time in seconds.
        """
        self._execution_count += 1
        if success:
            self._success_count += 1
        self._total_tokens += tokens_used
        self._total_time += execution_time

    def is_suitable_for_complexity(self, complexity: float) -> bool:
        """Check if this strategy is suitable for the given complexity.

        Args:
            complexity: Complexity score (0.0-1.0).

        Returns:
            True if complexity is in this strategy's range.
        """
        return self._config.is_complexity_in_range(complexity)


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Data classes
    "StrategyResult",
    "StrategyConfig",
    # Base class
    "BaseReasoningStrategy",
    # Constants
    "COMPLEXITY_RANGES",
    "TOKEN_BUDGETS",
    # Utilities
    "utc_now",
]
