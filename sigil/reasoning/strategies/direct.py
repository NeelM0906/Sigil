"""Direct reasoning strategy for Sigil v2.

This module implements the DirectStrategy, which uses a single LLM call
for simple, low-complexity tasks.

Classes:
    DirectStrategy: Single-call reasoning for simple tasks.

Example:
    >>> from sigil.reasoning.strategies.direct import DirectStrategy
    >>>
    >>> strategy = DirectStrategy()
    >>> result = await strategy.execute(
    ...     task="What is the capital of France?",
    ...     context={},
    ... )
    >>> print(result.answer)
    'Paris'
"""

from __future__ import annotations

import time
from typing import Any, Optional

from sigil.state.store import EventStore
from sigil.telemetry.tokens import TokenTracker

from sigil.reasoning.strategies.base import (
    BaseReasoningStrategy,
    StrategyResult,
    StrategyConfig,
    utc_now,
)
from sigil.reasoning.strategies.utils import build_tool_aware_context_string


# =============================================================================
# Constants
# =============================================================================

DIRECT_MIN_COMPLEXITY = 0.0
DIRECT_MAX_COMPLEXITY = 0.3
DIRECT_MIN_TOKENS = 100
DIRECT_MAX_TOKENS = 300


# =============================================================================
# Direct Strategy
# =============================================================================


class DirectStrategy(BaseReasoningStrategy):
    """Single-call reasoning strategy for simple tasks.

    DirectStrategy is designed for low-complexity tasks (0.0-0.3) that can
    be answered with a single LLM call. It provides fast response times
    with minimal token usage.

    Characteristics:
        - Complexity range: 0.0-0.3
        - Token budget: 100-300
        - Reasoning trace: Single step
        - Confidence: 0.4-0.7 (estimated from response)
        - Best for: Simple questions, factual queries, short tasks

    Attributes:
        name: "direct"
        config: Strategy configuration with direct-specific defaults.

    Example:
        >>> strategy = DirectStrategy()
        >>> result = await strategy.execute(
        ...     task="Summarize in one sentence: AI is transforming industries.",
        ...     context={},
        ... )
    """

    def __init__(
        self,
        config: Optional[StrategyConfig] = None,
        event_store: Optional[EventStore] = None,
        token_tracker: Optional[TokenTracker] = None,
    ) -> None:
        """Initialize DirectStrategy.

        Args:
            config: Optional strategy configuration.
            event_store: Optional event store for audit trails.
            token_tracker: Optional token tracker for budget.
        """
        # Set default config if not provided
        if config is None:
            config = StrategyConfig(
                min_complexity=DIRECT_MIN_COMPLEXITY,
                max_complexity=DIRECT_MAX_COMPLEXITY,
                min_tokens=DIRECT_MIN_TOKENS,
                max_tokens=DIRECT_MAX_TOKENS,
                timeout_seconds=30.0,
            )
        super().__init__(config, event_store, token_tracker)

    @property
    def name(self) -> str:
        """Return the strategy name."""
        return "direct"

    def _build_prompt(self, task: str, context: dict[str, Any]) -> str:
        """Build the prompt for direct execution.

        Args:
            task: The task to execute.
            context: Context information.

        Returns:
            Prompt string.
        """
        context_str = ""
        if context:
            context_str = build_tool_aware_context_string(context)
            if context_str:
                context_str = "\n\nContext:\n" + context_str

        return f"""Answer the following task directly and concisely.
{context_str}
Task: {task}

Provide a clear, direct answer:"""

    async def execute(self, task: str, context: dict[str, Any]) -> StrategyResult:
        """Execute direct reasoning on a task.

        Makes a single LLM call to generate an answer. Suitable for
        simple, factual questions that don't require complex reasoning.

        Args:
            task: The task/question to answer.
            context: Context information for the task.

        Returns:
            StrategyResult with answer, confidence (0.4-0.7), and trace.

        Example:
            >>> result = await strategy.execute("What is 2+2?", {})
            >>> result.answer
            '4'
        """
        started_at = utc_now()
        start_time = time.time()
        tokens_used = 0

        try:
            # Build prompt
            prompt = self._build_prompt(task, context)

            # Call LLM
            response, call_tokens = await self._call_llm(
                prompt=prompt,
                temperature=self._config.temperature,
                max_tokens=self._config.max_tokens,
            )
            tokens_used += call_tokens

            # Create reasoning trace
            reasoning_trace = ["Single LLM call for direct answer"]

            # Estimate confidence
            confidence = self._estimate_confidence(response, reasoning_trace)
            # Direct strategy caps confidence lower
            confidence = min(confidence, 0.7)

            execution_time = time.time() - start_time

            # Record metrics
            self._record_execution(True, tokens_used, execution_time)

            return StrategyResult(
                answer=response,
                confidence=confidence,
                reasoning_trace=reasoning_trace,
                tokens_used=tokens_used,
                execution_time_seconds=execution_time,
                model=self._settings.llm.model,
                metadata={
                    "strategy": self.name,
                    "prompt_length": len(prompt),
                },
                success=True,
                started_at=started_at,
                completed_at=utc_now(),
            )

        except Exception as e:
            execution_time = time.time() - start_time
            self._record_execution(False, tokens_used, execution_time)

            return StrategyResult.from_error(
                error=str(e),
                strategy_name=self.name,
                execution_time=execution_time,
                tokens_used=tokens_used,
            )


# =============================================================================
# Exports
# =============================================================================

__all__ = ["DirectStrategy"]
