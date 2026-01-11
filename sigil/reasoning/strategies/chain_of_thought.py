"""Chain of Thought reasoning strategy for Sigil v2.

This module implements the ChainOfThoughtStrategy, which prompts the LLM
to reason step-by-step before providing an answer.

Classes:
    ChainOfThoughtStrategy: Step-by-step reasoning for moderate tasks.

Example:
    >>> from sigil.reasoning.strategies.chain_of_thought import ChainOfThoughtStrategy
    >>>
    >>> strategy = ChainOfThoughtStrategy()
    >>> result = await strategy.execute(
    ...     task="How many days are there in 5 years?",
    ...     context={},
    ... )
"""

from __future__ import annotations

import re
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


# =============================================================================
# Constants
# =============================================================================

COT_MIN_COMPLEXITY = 0.3
COT_MAX_COMPLEXITY = 0.5
COT_MIN_TOKENS = 300
COT_MAX_TOKENS = 800


# =============================================================================
# Chain of Thought Strategy
# =============================================================================


class ChainOfThoughtStrategy(BaseReasoningStrategy):
    """Step-by-step reasoning strategy.

    ChainOfThoughtStrategy prompts the LLM to think step-by-step,
    making the reasoning process explicit. This improves accuracy
    for tasks requiring logical deduction or multi-step thinking.

    Characteristics:
        - Complexity range: 0.3-0.5
        - Token budget: 300-800
        - Reasoning trace: Extracted step-by-step thinking
        - Confidence: 0.6-0.8
        - Best for: Math problems, logical reasoning, analysis

    The strategy uses the "Let's think step by step" prompt pattern
    to encourage explicit reasoning.

    Attributes:
        name: "chain_of_thought"
        config: Strategy configuration with CoT-specific defaults.

    Example:
        >>> strategy = ChainOfThoughtStrategy()
        >>> result = await strategy.execute(
        ...     task="If a train travels 60 mph for 2.5 hours, how far does it go?",
        ...     context={},
        ... )
    """

    def __init__(
        self,
        config: Optional[StrategyConfig] = None,
        event_store: Optional[EventStore] = None,
        token_tracker: Optional[TokenTracker] = None,
    ) -> None:
        """Initialize ChainOfThoughtStrategy.

        Args:
            config: Optional strategy configuration.
            event_store: Optional event store for audit trails.
            token_tracker: Optional token tracker for budget.
        """
        if config is None:
            config = StrategyConfig(
                min_complexity=COT_MIN_COMPLEXITY,
                max_complexity=COT_MAX_COMPLEXITY,
                min_tokens=COT_MIN_TOKENS,
                max_tokens=COT_MAX_TOKENS,
                timeout_seconds=60.0,
            )
        super().__init__(config, event_store, token_tracker)

    @property
    def name(self) -> str:
        """Return the strategy name."""
        return "chain_of_thought"

    def _build_prompt(self, task: str, context: dict[str, Any]) -> str:
        """Build the chain-of-thought prompt.

        Args:
            task: The task to execute.
            context: Context information.

        Returns:
            Prompt string with step-by-step instruction.
        """
        context_str = ""
        if context:
            context_str = "\n\nContext:\n"
            for key, value in context.items():
                context_str += f"- {key}: {value}\n"

        return f"""You are a careful analytical thinker. Solve the following task by thinking through it step by step.
{context_str}
Task: {task}

Let's think step by step:

1."""

    def _extract_reasoning_steps(self, response: str) -> list[str]:
        """Extract reasoning steps from the LLM response.

        Parses the response to identify individual reasoning steps,
        looking for numbered lists, bullet points, or paragraph breaks.

        Args:
            response: The LLM response text.

        Returns:
            List of reasoning step strings.
        """
        steps = []

        # Try to extract numbered steps (1., 2., etc.)
        numbered_pattern = r"(?:^|\n)\s*(\d+)[.):]\s*(.+?)(?=(?:\n\s*\d+[.):])|\n\n|$)"
        numbered_matches = re.findall(numbered_pattern, response, re.DOTALL)

        if numbered_matches:
            for _, step_text in numbered_matches:
                cleaned = step_text.strip()
                if cleaned and len(cleaned) > 10:
                    steps.append(cleaned)

        # If no numbered steps found, try bullet points
        if not steps:
            bullet_pattern = r"(?:^|\n)\s*[-*]\s*(.+?)(?=(?:\n\s*[-*])|\n\n|$)"
            bullet_matches = re.findall(bullet_pattern, response, re.DOTALL)
            for match in bullet_matches:
                cleaned = match.strip()
                if cleaned and len(cleaned) > 10:
                    steps.append(cleaned)

        # If still no steps, split by double newlines or "Step" markers
        if not steps:
            step_pattern = r"[Ss]tep\s*\d*:?\s*(.+?)(?=[Ss]tep\s*\d*|$)"
            step_matches = re.findall(step_pattern, response, re.DOTALL)
            for match in step_matches:
                cleaned = match.strip()
                if cleaned and len(cleaned) > 10:
                    steps.append(cleaned)

        # Fallback: split response into paragraphs
        if not steps:
            paragraphs = response.split("\n\n")
            for para in paragraphs:
                cleaned = para.strip()
                if cleaned and len(cleaned) > 20:
                    steps.append(cleaned)

        # Ensure we have at least one step
        if not steps:
            steps = [response.strip()[:200] + "..." if len(response) > 200 else response.strip()]

        return steps

    def _extract_final_answer(self, response: str) -> str:
        """Extract the final answer from the response.

        Looks for explicit answer markers or returns the last
        substantive content.

        Args:
            response: The LLM response text.

        Returns:
            The final answer string.
        """
        # Look for explicit answer markers
        answer_patterns = [
            r"[Ff]inal [Aa]nswer:?\s*(.+?)$",
            r"[Tt]herefore,?\s*(.+?)$",
            r"[Ss]o,?\s*the answer is:?\s*(.+?)$",
            r"[Ii]n conclusion,?\s*(.+?)$",
            r"[Tt]he answer is:?\s*(.+?)$",
        ]

        for pattern in answer_patterns:
            match = re.search(pattern, response, re.DOTALL | re.MULTILINE)
            if match:
                return match.group(1).strip()

        # Fallback: return last non-empty paragraph
        paragraphs = [p.strip() for p in response.split("\n\n") if p.strip()]
        if paragraphs:
            return paragraphs[-1]

        return response.strip()

    async def execute(self, task: str, context: dict[str, Any]) -> StrategyResult:
        """Execute chain-of-thought reasoning on a task.

        Prompts the LLM to think step-by-step, then extracts the
        reasoning trace and final answer.

        Args:
            task: The task/question to reason about.
            context: Context information for the task.

        Returns:
            StrategyResult with answer, confidence (0.6-0.8), and trace.

        Example:
            >>> result = await strategy.execute(
            ...     "What is 15% of 80?",
            ...     {},
            ... )
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

            # Extract reasoning steps
            reasoning_trace = self._extract_reasoning_steps(response)

            # Extract final answer
            answer = self._extract_final_answer(response)

            # Estimate confidence based on reasoning quality
            confidence = self._estimate_confidence(response, reasoning_trace)
            # CoT typically achieves 0.6-0.8 confidence
            confidence = max(0.6, min(confidence, 0.8))

            execution_time = time.time() - start_time

            # Record metrics
            self._record_execution(True, tokens_used, execution_time)

            return StrategyResult(
                answer=answer,
                confidence=confidence,
                reasoning_trace=reasoning_trace,
                tokens_used=tokens_used,
                execution_time_seconds=execution_time,
                model=self._settings.llm.model,
                metadata={
                    "strategy": self.name,
                    "step_count": len(reasoning_trace),
                    "full_response_length": len(response),
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

__all__ = ["ChainOfThoughtStrategy"]
