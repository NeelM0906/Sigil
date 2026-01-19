"""Reasoning Manager for Sigil v2 Phase 5 Planning & Reasoning.

This module implements the ReasoningManager, which orchestrates strategy
selection and execution based on task complexity.

Classes:
    ReasoningManager: Orchestrates reasoning strategy selection and execution.

Example:
    >>> from sigil.reasoning.manager import ReasoningManager
    >>>
    >>> manager = ReasoningManager()
    >>> result = await manager.execute(
    ...     task="Calculate quarterly revenue growth",
    ...     context={"Q1": 100000, "Q2": 120000},
    ...     complexity=0.4,
    ... )
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

from sigil.config import get_settings
from sigil.core.exceptions import ReasoningError, StrategyNotFoundError
from sigil.state.events import Event, EventType, _generate_event_id, _get_utc_now
from sigil.state.store import EventStore
from sigil.telemetry.tokens import TokenTracker

from sigil.reasoning.strategies.base import (
    BaseReasoningStrategy,
    StrategyResult,
    StrategyConfig,
    COMPLEXITY_RANGES,
)
from sigil.reasoning.strategies.direct import DirectStrategy
from sigil.reasoning.strategies.chain_of_thought import ChainOfThoughtStrategy
from sigil.reasoning.strategies.tree_of_thoughts import TreeOfThoughtsStrategy
from sigil.reasoning.strategies.react import ReActStrategy
from sigil.reasoning.strategies.mcts import MCTSStrategy


# =============================================================================
# Module Logger
# =============================================================================

logger = logging.getLogger(__name__)


# =============================================================================
# Event Creators
# =============================================================================


def create_strategy_selected_event(
    session_id: str,
    strategy_name: str,
    complexity: float,
    reason: str,
    correlation_id: Optional[str] = None,
) -> Event:
    """Create a StrategySelectedEvent."""
    payload = {
        "strategy_name": strategy_name,
        "complexity": complexity,
        "reason": reason,
    }
    return Event(
        event_id=_generate_event_id(),
        event_type=EventType.STRATEGY_SELECTED,
        timestamp=_get_utc_now(),
        session_id=session_id,
        correlation_id=correlation_id,
        payload=payload,
    )


def create_reasoning_completed_event(
    session_id: str,
    strategy_name: str,
    success: bool,
    confidence: float,
    tokens_used: int,
    execution_time: float,
    error: Optional[str] = None,
    correlation_id: Optional[str] = None,
) -> Event:
    """Create a ReasoningCompletedEvent."""
    payload = {
        "strategy_name": strategy_name,
        "success": success,
        "confidence": confidence,
        "tokens_used": tokens_used,
        "execution_time_seconds": execution_time,
        "error": error,
    }
    return Event(
        event_id=_generate_event_id(),
        event_type=EventType.REASONING_COMPLETED,
        timestamp=_get_utc_now(),
        session_id=session_id,
        correlation_id=correlation_id,
        payload=payload,
    )


def create_strategy_fallback_event(
    session_id: str,
    from_strategy: str,
    to_strategy: str,
    reason: str,
    correlation_id: Optional[str] = None,
) -> Event:
    """Create a StrategyFallbackEvent."""
    payload = {
        "from_strategy": from_strategy,
        "to_strategy": to_strategy,
        "reason": reason,
    }
    return Event(
        event_id=_generate_event_id(),
        event_type=EventType.STRATEGY_FALLBACK,
        timestamp=_get_utc_now(),
        session_id=session_id,
        correlation_id=correlation_id,
        payload=payload,
    )


# =============================================================================
# Strategy Metrics
# =============================================================================


@dataclass
class StrategyMetrics:
    """Metrics for a single strategy.

    Attributes:
        name: Strategy name.
        executions: Total execution count.
        successes: Successful execution count.
        failures: Failed execution count.
        fallbacks: Times this strategy triggered fallback.
        total_tokens: Total tokens consumed.
        total_time: Total execution time.
    """

    name: str
    executions: int = 0
    successes: int = 0
    failures: int = 0
    fallbacks: int = 0
    total_tokens: int = 0
    total_time: float = 0.0

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.executions == 0:
            return 0.0
        return self.successes / self.executions

    @property
    def avg_tokens(self) -> float:
        """Calculate average tokens per execution."""
        if self.executions == 0:
            return 0.0
        return self.total_tokens / self.executions

    @property
    def avg_time(self) -> float:
        """Calculate average execution time."""
        if self.executions == 0:
            return 0.0
        return self.total_time / self.executions

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "executions": self.executions,
            "successes": self.successes,
            "failures": self.failures,
            "fallbacks": self.fallbacks,
            "success_rate": self.success_rate,
            "total_tokens": self.total_tokens,
            "avg_tokens": self.avg_tokens,
            "total_time": self.total_time,
            "avg_time": self.avg_time,
        }


# =============================================================================
# Reasoning Manager
# =============================================================================


class ReasoningManager:
    """Orchestrates reasoning strategy selection and execution.

    The ReasoningManager registers all reasoning strategies, selects the
    appropriate strategy based on task complexity, and handles fallback
    chains when strategies fail.

    Features:
        - Auto-selection based on complexity
        - Manual strategy override
        - Fallback chain: MCTS -> ReAct -> ToT -> CoT -> Direct
        - Per-strategy metrics tracking
        - Event emission for audit trails

    Strategy Selection:
        - 0.0-0.3: Direct
        - 0.3-0.5: Chain of Thought
        - 0.5-0.7: Tree of Thoughts
        - 0.7-0.9: ReAct
        - 0.9-1.0: MCTS

    Attributes:
        strategies: Dictionary of registered strategies.
        event_store: Event store for audit trails.
        token_tracker: Token tracker for budget management.

    Example:
        >>> manager = ReasoningManager()
        >>> result = await manager.execute(
        ...     task="Design a marketing campaign",
        ...     context={"budget": "$50k"},
        ...     complexity=0.6,
        ... )
    """

    # Strategy names in fallback order (most complex to least)
    FALLBACK_ORDER = ["mcts", "react", "tree_of_thoughts", "chain_of_thought", "direct"]

    def __init__(
        self,
        event_store: Optional[EventStore] = None,
        token_tracker: Optional[TokenTracker] = None,
        complexity_ranges: Optional[dict[str, tuple[float, float]]] = None,
    ) -> None:
        """Initialize the ReasoningManager.

        Args:
            event_store: Optional custom event store.
            token_tracker: Optional token tracker for budget.
            complexity_ranges: Optional custom complexity ranges.
        """
        self._event_store = event_store
        self._token_tracker = token_tracker
        self._settings = get_settings()

        # Use provided complexity ranges or defaults
        self._complexity_ranges = complexity_ranges or dict(COMPLEXITY_RANGES)

        # Initialize strategies
        self._strategies: dict[str, BaseReasoningStrategy] = {}
        self._register_default_strategies()

        # Initialize metrics
        self._metrics: dict[str, StrategyMetrics] = {
            name: StrategyMetrics(name=name) for name in self._strategies
        }

    def _register_default_strategies(self) -> None:
        """Register all default strategies."""
        self._strategies["direct"] = DirectStrategy(
            event_store=self._event_store,
            token_tracker=self._token_tracker,
        )
        self._strategies["chain_of_thought"] = ChainOfThoughtStrategy(
            event_store=self._event_store,
            token_tracker=self._token_tracker,
        )
        self._strategies["tree_of_thoughts"] = TreeOfThoughtsStrategy(
            event_store=self._event_store,
            token_tracker=self._token_tracker,
        )
        self._strategies["react"] = ReActStrategy(
            event_store=self._event_store,
            token_tracker=self._token_tracker,
        )
        self._strategies["mcts"] = MCTSStrategy(
            event_store=self._event_store,
            token_tracker=self._token_tracker,
        )

    def register_strategy(
        self,
        name: str,
        strategy: BaseReasoningStrategy,
        complexity_range: Optional[tuple[float, float]] = None,
    ) -> None:
        """Register a custom strategy.

        Args:
            name: Strategy name.
            strategy: The strategy instance.
            complexity_range: Optional (min, max) complexity range.
        """
        self._strategies[name] = strategy
        self._metrics[name] = StrategyMetrics(name=name)

        if complexity_range:
            self._complexity_ranges[name] = complexity_range

        logger.info(f"Registered strategy: {name}")

    def select_strategy(self, complexity: float) -> str:
        """Select the appropriate strategy for a complexity level.

        Args:
            complexity: Task complexity (0.0-1.0).

        Returns:
            Name of the selected strategy.

        Example:
            >>> manager.select_strategy(0.4)
            'chain_of_thought'
        """
        complexity = max(0.0, min(1.0, complexity))

        # Find strategy whose range contains the complexity
        for name, (min_c, max_c) in self._complexity_ranges.items():
            if min_c <= complexity <= max_c:
                return name

        # Fallback to direct for edge cases
        return "direct"

    def get_strategy(self, name: str) -> BaseReasoningStrategy:
        """Get a strategy by name.

        Args:
            name: Strategy name.

        Returns:
            The strategy instance.

        Raises:
            StrategyNotFoundError: If strategy not found.
        """
        if name not in self._strategies:
            raise StrategyNotFoundError(
                f"Strategy not found: {name}. "
                f"Available: {list(self._strategies.keys())}",
                strategy=name,
            )
        return self._strategies[name]

    async def execute(
        self,
        task: str,
        context: dict[str, Any],
        complexity: float,
        strategy: Optional[str] = None,
        session_id: Optional[str] = None,
        allow_fallback: bool = True,
    ) -> StrategyResult:
        """Execute reasoning on a task.

        Selects or uses the provided strategy, executes reasoning,
        and falls back to simpler strategies on failure.

        Args:
            task: The task to reason about.
            context: Context information.
            complexity: Task complexity (0.0-1.0).
            strategy: Optional strategy name override.
            session_id: Optional session ID for events.
            allow_fallback: Whether to allow fallback on failure.

        Returns:
            StrategyResult from successful strategy.

        Example:
            >>> result = await manager.execute(
            ...     task="Analyze market trends",
            ...     context={},
            ...     complexity=0.65,
            ... )
        """
        # Select strategy
        if strategy:
            selected_strategy = strategy
            selection_reason = "user_specified"
        else:
            selected_strategy = self.select_strategy(complexity)
            selection_reason = f"auto_selected_for_complexity_{complexity:.2f}"

        # Emit strategy selected event
        if self._event_store and session_id:
            event = create_strategy_selected_event(
                session_id=session_id,
                strategy_name=selected_strategy,
                complexity=complexity,
                reason=selection_reason,
            )
            self._event_store.append(event)

        logger.info(
            f"Selected strategy '{selected_strategy}' for complexity {complexity:.2f}"
        )

        # Build fallback chain starting from selected strategy
        fallback_chain = self._build_fallback_chain(selected_strategy)

        # Execute with fallback
        last_error: Optional[str] = None
        for strategy_name in fallback_chain:
            try:
                strategy_instance = self._strategies[strategy_name]

                # Execute strategy
                start_time = time.time()
                result = await strategy_instance.execute(task, context)
                execution_time = time.time() - start_time

                # Update metrics
                metrics = self._metrics[strategy_name]
                metrics.executions += 1
                metrics.total_tokens += result.tokens_used
                metrics.total_time += execution_time

                if result.success:
                    metrics.successes += 1

                    # Emit reasoning completed event
                    if self._event_store and session_id:
                        event = create_reasoning_completed_event(
                            session_id=session_id,
                            strategy_name=strategy_name,
                            success=True,
                            confidence=result.confidence,
                            tokens_used=result.tokens_used,
                            execution_time=execution_time,
                        )
                        self._event_store.append(event)

                    return result
                else:
                    metrics.failures += 1
                    last_error = result.error

                    if not allow_fallback:
                        return result

                    # Try fallback
                    metrics.fallbacks += 1
                    logger.warning(
                        f"Strategy '{strategy_name}' failed: {result.error}. "
                        f"Attempting fallback..."
                    )

                    # Emit fallback event
                    if self._event_store and session_id:
                        next_idx = fallback_chain.index(strategy_name) + 1
                        if next_idx < len(fallback_chain):
                            next_strategy = fallback_chain[next_idx]
                            event = create_strategy_fallback_event(
                                session_id=session_id,
                                from_strategy=strategy_name,
                                to_strategy=next_strategy,
                                reason=result.error or "unknown",
                            )
                            self._event_store.append(event)

            except Exception as e:
                last_error = str(e)
                logger.warning(
                    f"Strategy '{strategy_name}' raised exception: {e}. "
                    f"Attempting fallback..."
                )

                # Update metrics
                metrics = self._metrics[strategy_name]
                metrics.executions += 1
                metrics.failures += 1
                metrics.fallbacks += 1

                if not allow_fallback:
                    return StrategyResult.from_error(
                        error=str(e),
                        strategy_name=strategy_name,
                    )

        # All strategies failed - should not happen as direct never fails
        logger.error("All strategies failed, including direct fallback")

        if self._event_store and session_id:
            event = create_reasoning_completed_event(
                session_id=session_id,
                strategy_name="all",
                success=False,
                confidence=0.0,
                tokens_used=0,
                execution_time=0.0,
                error=last_error,
            )
            self._event_store.append(event)

        return StrategyResult.from_error(
            error=last_error or "All strategies failed",
            strategy_name="fallback_chain",
        )

    def _build_fallback_chain(self, starting_strategy: str) -> list[str]:
        """Build the fallback chain starting from a strategy.

        Args:
            starting_strategy: The initial strategy.

        Returns:
            List of strategy names in fallback order.
        """
        if starting_strategy not in self.FALLBACK_ORDER:
            # Unknown strategy, start from the beginning
            return self.FALLBACK_ORDER.copy()

        start_idx = self.FALLBACK_ORDER.index(starting_strategy)
        return self.FALLBACK_ORDER[start_idx:]

    def get_metrics(self) -> dict[str, Any]:
        """Get execution metrics for all strategies.

        Returns:
            Dictionary with per-strategy metrics.
        """
        return {
            "strategies": {
                name: metrics.to_dict() for name, metrics in self._metrics.items()
            },
            "total_executions": sum(m.executions for m in self._metrics.values()),
            "total_successes": sum(m.successes for m in self._metrics.values()),
            "total_failures": sum(m.failures for m in self._metrics.values()),
            "total_fallbacks": sum(m.fallbacks for m in self._metrics.values()),
            "overall_success_rate": (
                sum(m.successes for m in self._metrics.values())
                / max(sum(m.executions for m in self._metrics.values()), 1)
            ),
        }

    def clear_metrics(self) -> None:
        """Clear all execution metrics."""
        for metrics in self._metrics.values():
            metrics.executions = 0
            metrics.successes = 0
            metrics.failures = 0
            metrics.fallbacks = 0
            metrics.total_tokens = 0
            metrics.total_time = 0.0

        # Also clear strategy-level metrics
        for strategy in self._strategies.values():
            strategy.reset_metrics()

        logger.info("Cleared all reasoning metrics")

    def list_strategies(self) -> list[dict[str, Any]]:
        """List all registered strategies with their complexity ranges.

        Returns:
            List of strategy information dictionaries.
        """
        strategies = []
        for name, strategy in self._strategies.items():
            complexity_range = self._complexity_ranges.get(name, (0.0, 1.0))
            strategies.append(
                {
                    "name": name,
                    "complexity_range": complexity_range,
                    "metrics": self._metrics[name].to_dict(),
                }
            )
        return strategies


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "ReasoningManager",
    "StrategyMetrics",
    "create_strategy_selected_event",
    "create_reasoning_completed_event",
    "create_strategy_fallback_event",
]
