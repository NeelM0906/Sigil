"""ReAct reasoning strategy for Sigil v2.

This module implements the ReActStrategy, which interleaves reasoning
with tool calls in a Thought-Action-Observation loop.

Classes:
    ReActStep: A single step in the ReAct loop.
    ReActStrategy: Thought-Action-Observation reasoning for tool-heavy tasks.

Example:
    >>> from sigil.reasoning.strategies.react import ReActStrategy
    >>>
    >>> strategy = ReActStrategy()
    >>> result = await strategy.execute(
    ...     task="Find the current weather in Tokyo and summarize",
    ...     context={"tools": ["websearch.search"]},
    ... )
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional, Awaitable

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

REACT_MIN_COMPLEXITY = 0.7
REACT_MAX_COMPLEXITY = 0.9
REACT_MIN_TOKENS = 1000
REACT_MAX_TOKENS = 3000

MAX_REACT_ITERATIONS = 10
"""Maximum iterations in the ReAct loop."""


# =============================================================================
# ReAct Step
# =============================================================================


class StepType(str, Enum):
    """Type of ReAct step."""

    THOUGHT = "thought"
    ACTION = "action"
    OBSERVATION = "observation"


@dataclass
class ReActStep:
    """A single step in the ReAct loop.

    Attributes:
        step_type: Type of step (thought, action, observation).
        content: Content of the step.
        tool_name: Tool name (for action steps).
        tool_args: Tool arguments (for action steps).
        iteration: Iteration number.
    """

    step_type: StepType
    content: str
    tool_name: Optional[str] = None
    tool_args: Optional[dict[str, Any]] = None
    iteration: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert step to dictionary."""
        return {
            "step_type": self.step_type.value,
            "content": self.content,
            "tool_name": self.tool_name,
            "tool_args": self.tool_args,
            "iteration": self.iteration,
        }


# =============================================================================
# Tool Executor Type
# =============================================================================

ToolExecutorFn = Callable[[str, dict[str, Any]], Awaitable[str]]


# =============================================================================
# ReAct Strategy
# =============================================================================


class ReActStrategy(BaseReasoningStrategy):
    """Thought-Action-Observation reasoning strategy.

    ReActStrategy implements the ReAct pattern, which interleaves
    reasoning (thoughts) with actions (tool calls) and observations
    (results). This enables dynamic, tool-heavy problem solving.

    Characteristics:
        - Complexity range: 0.7-0.9
        - Token budget: 1000-3000
        - Reasoning trace: Full Thought-Action-Observation sequence
        - Confidence: Variable (0.5-0.9)
        - Best for: Multi-step tasks, tool-heavy problems, research

    The strategy follows a loop:
    1. Thought: Reason about what to do next
    2. Action: Execute a tool/action
    3. Observation: Process the result
    4. Repeat until done or max iterations

    Attributes:
        name: "react"
        max_iterations: Maximum loop iterations.
        tool_executor: Optional function to execute tools.
        config: Strategy configuration with ReAct-specific defaults.

    Example:
        >>> strategy = ReActStrategy(max_iterations=5)
        >>> result = await strategy.execute(
        ...     task="Research recent AI developments",
        ...     context={"tools": ["websearch.search"]},
        ... )
    """

    def __init__(
        self,
        config: Optional[StrategyConfig] = None,
        event_store: Optional[EventStore] = None,
        token_tracker: Optional[TokenTracker] = None,
        max_iterations: int = MAX_REACT_ITERATIONS,
        tool_executor: Optional[ToolExecutorFn] = None,
    ) -> None:
        """Initialize ReActStrategy.

        Args:
            config: Optional strategy configuration.
            event_store: Optional event store for audit trails.
            token_tracker: Optional token tracker for budget.
            max_iterations: Maximum loop iterations.
            tool_executor: Optional function to execute tools.
        """
        if config is None:
            config = StrategyConfig(
                min_complexity=REACT_MIN_COMPLEXITY,
                max_complexity=REACT_MAX_COMPLEXITY,
                min_tokens=REACT_MIN_TOKENS,
                max_tokens=REACT_MAX_TOKENS,
                timeout_seconds=180.0,
            )
        super().__init__(config, event_store, token_tracker)
        self._max_iterations = max_iterations
        self._tool_executor = tool_executor or self._default_tool_executor

    @property
    def name(self) -> str:
        """Return the strategy name."""
        return "react"

    async def _default_tool_executor(
        self,
        tool_name: str,
        tool_args: dict[str, Any],
    ) -> str:
        """Default tool executor (simulated).

        Args:
            tool_name: Name of the tool to execute.
            tool_args: Arguments for the tool.

        Returns:
            Simulated tool result.
        """
        # In production, this would call actual tools
        return f"[Tool '{tool_name}' executed with args: {tool_args}. Result: Simulated response]"

    def _build_initial_prompt(
        self,
        task: str,
        context: dict[str, Any],
    ) -> str:
        """Build the initial ReAct prompt.

        Args:
            task: The task to execute.
            context: Context information.

        Returns:
            Prompt string.
        """
        tools = context.get("tools", [])
        tools_str = ", ".join(tools) if tools else "No specific tools available"

        context_str = ""
        if context:
            context_str = build_tool_aware_context_string(context)
            if context_str:
                context_str = "\n\nContext:\n" + context_str

        return f"""You are a problem-solving agent using the ReAct framework.
You have access to the following tools: {tools_str}

Solve the task by interleaving Thought, Action, and Observation steps.
{context_str}
Task: {task}

Instructions:
1. First, think about what you need to do (Thought)
2. Then, take an action using a tool (Action: tool_name[args])
3. You will receive an observation (Observation: result)
4. Continue until you can provide a final answer

When you have the final answer, respond with:
Final Answer: [your answer]

Begin:

Thought 1:"""

    def _build_continuation_prompt(
        self,
        steps: list[ReActStep],
        iteration: int,
    ) -> str:
        """Build continuation prompt with history.

        Args:
            steps: Previous steps in the loop.
            iteration: Current iteration number.

        Returns:
            Prompt string.
        """
        history = ""
        for step in steps:
            if step.step_type == StepType.THOUGHT:
                history += f"Thought {step.iteration}: {step.content}\n"
            elif step.step_type == StepType.ACTION:
                history += f"Action {step.iteration}: {step.tool_name}[{step.tool_args}]\n"
            elif step.step_type == StepType.OBSERVATION:
                history += f"Observation {step.iteration}: {step.content}\n"

        return f"{history}\nThought {iteration}:"

    def _parse_response(self, response: str, iteration: int) -> list[ReActStep]:
        """Parse LLM response into ReAct steps.

        Args:
            response: The LLM response.
            iteration: Current iteration number.

        Returns:
            List of parsed ReActSteps.
        """
        steps = []

        # Check for final answer
        final_match = re.search(r"[Ff]inal [Aa]nswer:?\s*(.+?)$", response, re.DOTALL)
        if final_match:
            # This is a thought leading to final answer
            thought_content = response[: final_match.start()].strip()
            if thought_content:
                steps.append(
                    ReActStep(
                        step_type=StepType.THOUGHT,
                        content=thought_content,
                        iteration=iteration,
                    )
                )
            steps.append(
                ReActStep(
                    step_type=StepType.THOUGHT,
                    content=f"Final Answer: {final_match.group(1).strip()}",
                    iteration=iteration,
                )
            )
            return steps

        # Parse thought
        thought_match = re.match(r"(.+?)(?=[Aa]ction|$)", response, re.DOTALL)
        if thought_match:
            thought_content = thought_match.group(1).strip()
            if thought_content:
                steps.append(
                    ReActStep(
                        step_type=StepType.THOUGHT,
                        content=thought_content,
                        iteration=iteration,
                    )
                )

        # Parse action
        action_match = re.search(
            r"[Aa]ction\s*\d*:?\s*(\w+)\s*\[([^\]]*)\]", response
        )
        if action_match:
            tool_name = action_match.group(1)
            args_str = action_match.group(2)

            # Try to parse args
            try:
                # Simple key=value parsing
                tool_args = {}
                if args_str:
                    for part in args_str.split(","):
                        if "=" in part:
                            key, value = part.split("=", 1)
                            tool_args[key.strip()] = value.strip().strip("'\"")
                        else:
                            tool_args["query"] = part.strip().strip("'\"")
            except Exception:
                tool_args = {"raw": args_str}

            steps.append(
                ReActStep(
                    step_type=StepType.ACTION,
                    content=f"{tool_name}[{args_str}]",
                    tool_name=tool_name,
                    tool_args=tool_args,
                    iteration=iteration,
                )
            )

        # If no steps parsed, treat whole response as thought
        if not steps:
            steps.append(
                ReActStep(
                    step_type=StepType.THOUGHT,
                    content=response.strip(),
                    iteration=iteration,
                )
            )

        return steps

    def _is_final_answer(self, steps: list[ReActStep]) -> bool:
        """Check if the steps contain a final answer.

        Args:
            steps: List of ReAct steps.

        Returns:
            True if a final answer was found.
        """
        for step in steps:
            if step.step_type == StepType.THOUGHT and "Final Answer:" in step.content:
                return True
        return False

    def _extract_final_answer(self, steps: list[ReActStep]) -> str:
        """Extract the final answer from steps.

        Args:
            steps: List of ReAct steps.

        Returns:
            The final answer string.
        """
        for step in reversed(steps):
            if step.step_type == StepType.THOUGHT and "Final Answer:" in step.content:
                match = re.search(r"[Ff]inal [Aa]nswer:?\s*(.+?)$", step.content, re.DOTALL)
                if match:
                    return match.group(1).strip()

        # Fallback: return last thought
        for step in reversed(steps):
            if step.step_type == StepType.THOUGHT:
                return step.content

        return "No answer found"

    async def execute(self, task: str, context: dict[str, Any]) -> StrategyResult:
        """Execute ReAct reasoning on a task.

        Runs the Thought-Action-Observation loop until a final answer
        is reached or max iterations are hit.

        Args:
            task: The task/question to reason about.
            context: Context information (should include "tools" list).

        Returns:
            StrategyResult with answer, confidence (0.5-0.9), and trace.

        Example:
            >>> result = await strategy.execute(
            ...     "Find current stock price of AAPL",
            ...     {"tools": ["websearch.search"]},
            ... )
        """
        started_at = utc_now()
        start_time = time.time()
        total_tokens = 0
        all_steps: list[ReActStep] = []

        try:
            # Start with initial prompt
            prompt = self._build_initial_prompt(task, context)
            iteration = 1

            while iteration <= self._max_iterations:
                # Get next thought/action from LLM
                response, call_tokens = await self._call_llm(
                    prompt=prompt,
                    temperature=self._config.temperature,
                    max_tokens=self._config.max_tokens // self._max_iterations,
                )
                total_tokens += call_tokens

                # Parse response into steps
                new_steps = self._parse_response(response, iteration)
                all_steps.extend(new_steps)

                # Check for final answer
                if self._is_final_answer(new_steps):
                    break

                # Execute any actions
                for step in new_steps:
                    if step.step_type == StepType.ACTION and step.tool_name:
                        # Execute tool
                        observation = await self._tool_executor(
                            step.tool_name,
                            step.tool_args or {},
                        )

                        # Add observation
                        obs_step = ReActStep(
                            step_type=StepType.OBSERVATION,
                            content=observation,
                            iteration=iteration,
                        )
                        all_steps.append(obs_step)

                # Build continuation prompt
                iteration += 1
                prompt = self._build_continuation_prompt(all_steps, iteration)

            # Extract final answer
            answer = self._extract_final_answer(all_steps)

            # Build reasoning trace
            reasoning_trace = []
            for step in all_steps:
                prefix = step.step_type.value.capitalize()
                reasoning_trace.append(f"{prefix} {step.iteration}: {step.content}")

            # Calculate confidence based on iteration count and observations
            action_count = sum(1 for s in all_steps if s.step_type == StepType.ACTION)
            obs_count = sum(1 for s in all_steps if s.step_type == StepType.OBSERVATION)

            # More observations generally mean more grounded answer
            base_confidence = 0.6
            if obs_count > 0:
                base_confidence += min(obs_count * 0.1, 0.3)
            if iteration < self._max_iterations:
                base_confidence += 0.05  # Bonus for finishing early

            confidence = min(base_confidence, 0.9)

            execution_time = time.time() - start_time

            # Record metrics
            self._record_execution(True, total_tokens, execution_time)

            return StrategyResult(
                answer=answer,
                confidence=confidence,
                reasoning_trace=reasoning_trace,
                tokens_used=total_tokens,
                execution_time_seconds=execution_time,
                model=self._settings.llm.model,
                metadata={
                    "strategy": self.name,
                    "iterations": iteration,
                    "action_count": action_count,
                    "observation_count": obs_count,
                    "all_steps": [s.to_dict() for s in all_steps],
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

__all__ = ["ReActStrategy", "ReActStep", "StepType"]
