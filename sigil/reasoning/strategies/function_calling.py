"""LLM-Native Function Calling strategy for Sigil v2.

This module implements the FunctionCallingStrategy, which uses LLM-native
function calling (tool_use) in a ReAct (Reason+Act) loop pattern.

Unlike the text-based ReActStrategy that parses Thought/Action/Observation
from text, this strategy uses Claude's native tool_use capability for
more reliable tool invocation and result handling.

Classes:
    ToolUseBlock: Represents a tool_use response from the LLM.
    FunctionCallingStep: A single step in the ReAct loop.
    FunctionCallingStrategy: Native function calling with ReAct loop.

Example:
    >>> from sigil.reasoning.strategies.function_calling import FunctionCallingStrategy
    >>>
    >>> strategy = FunctionCallingStrategy()
    >>> tools = [
    ...     {
    ...         "name": "websearch.search",
    ...         "description": "Search the web",
    ...         "input_schema": {"type": "object", "properties": {"query": {"type": "string"}}}
    ...     }
    ... ]
    >>> result = await strategy.execute(
    ...     task="Find the latest news about AI",
    ...     context={},
    ...     tools=tools,
    ... )
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Optional, TYPE_CHECKING

from sigil.config import get_settings
from sigil.state.store import EventStore
from sigil.telemetry.tokens import TokenTracker

from sigil.reasoning.strategies.base import (
    BaseReasoningStrategy,
    StrategyResult,
    StrategyConfig,
    utc_now,
)
from sigil.reasoning.strategies.utils import build_tool_aware_context_string
from sigil.planning.executors.tavily_executor import TavilyExecutor, TavilyExecutorError
from sigil.planning.executors.builtin_executor import BuiltinToolExecutor, BuiltinExecutorError

if TYPE_CHECKING:
    pass


# =============================================================================
# Module Logger
# =============================================================================

logger = logging.getLogger(__name__)


# =============================================================================
# Constants
# =============================================================================

FUNCTION_CALLING_MIN_COMPLEXITY = 0.4
FUNCTION_CALLING_MAX_COMPLEXITY = 0.9
FUNCTION_CALLING_MIN_TOKENS = 500
FUNCTION_CALLING_MAX_TOKENS = 4000

MAX_ITERATIONS = 5
"""Maximum iterations in the ReAct loop."""


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class ToolUseBlock:
    """Represents a tool_use response from the LLM.

    Attributes:
        id: Unique identifier for this tool use (required for tool_result).
        name: Name of the tool to call.
        input: Tool arguments/parameters.
    """

    id: str
    name: str
    input: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for API messages."""
        return {
            "type": "tool_use",
            "id": self.id,
            "name": self.name,
            "input": self.input,
        }


@dataclass
class FunctionCallingStep:
    """A single step in the function calling ReAct loop.

    Attributes:
        iteration: Iteration number (1-based).
        thought: LLM reasoning text (from text blocks).
        tool_use: Tool call if any.
        tool_result: Result from tool execution.
        is_final: Whether the LLM indicated end_turn.
    """

    iteration: int
    thought: Optional[str] = None
    tool_use: Optional[ToolUseBlock] = None
    tool_result: Optional[str] = None
    is_final: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert step to dictionary."""
        return {
            "iteration": self.iteration,
            "thought": self.thought,
            "tool_use": self.tool_use.to_dict() if self.tool_use else None,
            "tool_result": self.tool_result,
            "is_final": self.is_final,
        }


# =============================================================================
# Function Calling Strategy
# =============================================================================


class FunctionCallingStrategy(BaseReasoningStrategy):
    """LLM-native function calling strategy with ReAct loop.

    FunctionCallingStrategy uses Claude's native tool_use capability to
    implement a Reason+Act loop. The LLM decides when to call tools and
    processes tool results to generate a final answer.

    Characteristics:
        - Complexity range: 0.4-0.9
        - Token budget: 500-4000
        - Max iterations: 5 (configurable)
        - Uses native tool_use (not text parsing)
        - Best for: Tool-heavy tasks, search, data retrieval

    The strategy flow:
    1. Send task + tools to LLM
    2. LLM responds with text (reasoning) and/or tool_use
    3. If tool_use: execute tool, send result back
    4. Repeat until LLM returns end_turn or max iterations

    Attributes:
        name: "function_calling"
        max_iterations: Maximum loop iterations.
        tavily_executor: Executor for websearch tools.
        builtin_executor: Executor for memory/planning tools.

    Example:
        >>> strategy = FunctionCallingStrategy(max_iterations=5)
        >>> tools = [{"name": "websearch.search", ...}]
        >>> result = await strategy.execute(
        ...     task="Find latest AI developments",
        ...     context={},
        ...     tools=tools,
        ... )
    """

    def __init__(
        self,
        config: Optional[StrategyConfig] = None,
        event_store: Optional[EventStore] = None,
        token_tracker: Optional[TokenTracker] = None,
        max_iterations: int = MAX_ITERATIONS,
        tavily_executor: Optional[TavilyExecutor] = None,
        builtin_executor: Optional[BuiltinToolExecutor] = None,
    ) -> None:
        """Initialize FunctionCallingStrategy.

        Args:
            config: Optional strategy configuration.
            event_store: Optional event store for audit trails.
            token_tracker: Optional token tracker for budget.
            max_iterations: Maximum loop iterations (default 5).
            tavily_executor: Optional Tavily executor for websearch.
            builtin_executor: Optional builtin executor for memory/planning.
        """
        if config is None:
            config = StrategyConfig(
                min_complexity=FUNCTION_CALLING_MIN_COMPLEXITY,
                max_complexity=FUNCTION_CALLING_MAX_COMPLEXITY,
                min_tokens=FUNCTION_CALLING_MIN_TOKENS,
                max_tokens=FUNCTION_CALLING_MAX_TOKENS,
                timeout_seconds=120.0,
            )
        super().__init__(config, event_store, token_tracker)
        self._max_iterations = max_iterations
        self._tavily_executor = tavily_executor or TavilyExecutor()
        self._builtin_executor = builtin_executor or BuiltinToolExecutor()

    @property
    def name(self) -> str:
        """Return the strategy name."""
        return "function_calling"

    def _build_system_prompt(self, context: dict[str, Any]) -> str:
        """Build the system prompt for function calling.

        Args:
            context: Context information.

        Returns:
            System prompt string.
        """
        context_str = ""
        if context:
            context_str = build_tool_aware_context_string(context)
            if context_str:
                context_str = "\n\nContext:\n" + context_str

        return f"""You are an intelligent assistant that can use tools to help answer questions and complete tasks.

When you need information that you don't have or need to perform an action, use the available tools.
Think step by step about what you need to do, then use tools as needed.
After getting tool results, analyze them and either use more tools or provide your final answer.

When you have enough information to answer the user's question, provide a clear and comprehensive response.
{context_str}"""

    async def _call_llm_with_tools(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> tuple[dict[str, Any], int]:
        """Call the LLM with tools using the Anthropic API.

        Args:
            messages: Conversation messages.
            tools: Tool definitions in Anthropic format.
            system_prompt: Optional system prompt.
            temperature: Optional temperature.
            max_tokens: Optional max tokens.

        Returns:
            Tuple of (response dict, tokens_used).

        Raises:
            ValueError: If API key is not configured.
        """
        try:
            import anthropic
        except ImportError:
            logger.error("Anthropic library not installed")
            raise ValueError("Anthropic library not installed")

        api_key = self._settings.api_keys.anthropic_api_key
        if not api_key:
            logger.error("ANTHROPIC_API_KEY not configured")
            raise ValueError("ANTHROPIC_API_KEY environment variable is required")

        model = self._settings.llm.model
        if model.startswith("anthropic:"):
            model = model[len("anthropic:"):]

        actual_temperature = temperature if temperature is not None else self._config.temperature
        actual_max_tokens = max_tokens if max_tokens is not None else self._config.max_tokens

        try:
            client = anthropic.Anthropic(api_key=api_key)

            request_params = {
                "model": model,
                "max_tokens": actual_max_tokens,
                "temperature": actual_temperature,
                "messages": messages,
            }

            if system_prompt:
                request_params["system"] = system_prompt

            if tools:
                request_params["tools"] = tools

            logger.debug(f"[FunctionCalling] Calling LLM with {len(messages)} messages, {len(tools)} tools")

            response = client.messages.create(**request_params)

            input_tokens = response.usage.input_tokens
            output_tokens = response.usage.output_tokens
            tokens_used = input_tokens + output_tokens

            if self._token_tracker:
                self._token_tracker.record_usage(
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                )

            response_dict = {
                "id": response.id,
                "type": response.type,
                "role": response.role,
                "content": [self._content_block_to_dict(block) for block in response.content],
                "model": response.model,
                "stop_reason": response.stop_reason,
                "stop_sequence": response.stop_sequence,
                "usage": {
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                },
            }

            logger.debug(
                f"[FunctionCalling] LLM response: stop_reason={response.stop_reason}, "
                f"tokens={tokens_used}"
            )

            return response_dict, tokens_used

        except anthropic.APIConnectionError as e:
            logger.error(f"Failed to connect to Anthropic API: {e}")
            raise ValueError(f"Failed to connect to Anthropic API: {e}")
        except anthropic.RateLimitError as e:
            logger.error(f"Rate limit exceeded: {e}")
            raise ValueError(f"Rate limit exceeded: {e}")
        except anthropic.APIStatusError as e:
            logger.error(f"Anthropic API error: {e.status_code} - {e.message}")
            raise ValueError(f"Anthropic API error: {e.status_code} - {e.message}")

    def _content_block_to_dict(self, block: Any) -> dict[str, Any]:
        """Convert a content block to dictionary.

        Args:
            block: Anthropic content block.

        Returns:
            Dictionary representation.
        """
        if hasattr(block, "type"):
            if block.type == "text":
                return {"type": "text", "text": block.text}
            elif block.type == "tool_use":
                return {
                    "type": "tool_use",
                    "id": block.id,
                    "name": block.name,
                    "input": block.input,
                }
        return {"type": "unknown", "raw": str(block)}

    def _extract_text_content(self, response: dict[str, Any]) -> str:
        """Extract text content from response.

        Args:
            response: LLM response dictionary.

        Returns:
            Combined text from all text blocks.
        """
        texts = []
        for block in response.get("content", []):
            if block.get("type") == "text":
                text = block.get("text", "")
                if text:
                    texts.append(text)
        return "\n".join(texts)

    def _extract_tool_use(self, response: dict[str, Any]) -> Optional[ToolUseBlock]:
        """Extract tool_use block from response.

        Args:
            response: LLM response dictionary.

        Returns:
            ToolUseBlock if found, None otherwise.
        """
        for block in response.get("content", []):
            if block.get("type") == "tool_use":
                return ToolUseBlock(
                    id=block.get("id", ""),
                    name=block.get("name", ""),
                    input=block.get("input", {}),
                )
        return None

    def _build_assistant_message(self, response: dict[str, Any]) -> dict[str, Any]:
        """Build assistant message from response for conversation history.

        Args:
            response: LLM response dictionary.

        Returns:
            Assistant message dictionary.
        """
        return {
            "role": "assistant",
            "content": response.get("content", []),
        }

    def _build_tool_result_message(
        self,
        tool_use_id: str,
        result: str,
        is_error: bool = False,
    ) -> dict[str, Any]:
        """Build tool_result message.

        Args:
            tool_use_id: ID of the tool_use block.
            result: Tool execution result.
            is_error: Whether the result is an error.

        Returns:
            User message with tool_result.
        """
        return {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": tool_use_id,
                    "content": result,
                    "is_error": is_error,
                }
            ],
        }

    def _determine_executor_type(self, tool_name: str) -> str:
        """Determine which executor should handle this tool.

        Args:
            tool_name: Full tool name.

        Returns:
            Executor type: "tavily", "builtin", or "unsupported".
        """
        tool_lower = tool_name.lower()

        # Check for websearch/tavily tools
        if tool_lower.startswith("websearch.") or tool_lower.startswith("tavily"):
            return "tavily"

        # Check for builtin tools
        if tool_lower.startswith("memory.") or tool_lower.startswith("planning."):
            return "builtin"

        # Also check using the executor's static methods
        if TavilyExecutor.is_tavily_tool(tool_name):
            return "tavily"

        if BuiltinToolExecutor.is_builtin_tool(tool_name):
            return "builtin"

        return "unsupported"

    async def _execute_tool(self, tool_use: ToolUseBlock) -> str:
        """Execute a tool and return the result.

        Routes the tool to the appropriate executor based on the tool name.

        Args:
            tool_use: The tool use block from LLM response.

        Returns:
            Tool execution result as string.
        """
        tool_name = tool_use.name
        tool_args = tool_use.input

        logger.info(f"[FunctionCalling] Executing tool: {tool_name}")
        logger.debug(f"[FunctionCalling] Tool args: {tool_args}")

        executor_type = self._determine_executor_type(tool_name)

        try:
            if executor_type == "tavily":
                result = await self._tavily_executor.execute(
                    tool_name=tool_name,
                    tool_args=tool_args,
                )
                logger.info(f"[FunctionCalling] Tavily tool completed: {tool_name}")
                return result

            elif executor_type == "builtin":
                result = await self._builtin_executor.execute(
                    tool_name=tool_name,
                    tool_args=tool_args,
                )
                logger.info(f"[FunctionCalling] Builtin tool completed: {tool_name}")
                return result

            else:
                error_msg = f"Unsupported tool: {tool_name}. Only websearch.*, memory.*, and planning.* tools are supported."
                logger.warning(f"[FunctionCalling] {error_msg}")
                return json.dumps({"error": error_msg})

        except TavilyExecutorError as e:
            error_msg = f"Tavily tool error: {e}"
            logger.error(f"[FunctionCalling] {error_msg}")
            return json.dumps({"error": error_msg})

        except BuiltinExecutorError as e:
            error_msg = f"Builtin tool error: {e}"
            logger.error(f"[FunctionCalling] {error_msg}")
            return json.dumps({"error": error_msg})

        except Exception as e:
            error_msg = f"Tool execution failed: {type(e).__name__}: {e}"
            logger.error(f"[FunctionCalling] {error_msg}", exc_info=True)
            return json.dumps({"error": error_msg})

    async def execute(
        self,
        task: str,
        context: dict[str, Any],
        tools: Optional[list[dict[str, Any]]] = None,
    ) -> StrategyResult:
        """Execute function calling reasoning on a task.

        Runs a ReAct loop using native function calling until the LLM
        returns end_turn or max iterations are reached.

        Args:
            task: The task/question to reason about.
            context: Context information for the task.
            tools: Optional list of tool definitions in Anthropic format.
                   If not provided, no tools will be available.

        Returns:
            StrategyResult with answer, confidence, and trace.

        Example:
            >>> tools = [{"name": "websearch.search", "description": "...", "input_schema": {...}}]
            >>> result = await strategy.execute("Find AI news", {}, tools=tools)
        """
        started_at = utc_now()
        start_time = time.time()
        total_tokens = 0
        all_steps: list[FunctionCallingStep] = []

        # Use empty list if no tools provided
        tools = tools or []

        try:
            # Build initial messages
            system_prompt = self._build_system_prompt(context)
            messages = [{"role": "user", "content": task}]

            iteration = 0
            final_text = ""

            while iteration < self._max_iterations:
                iteration += 1
                step = FunctionCallingStep(iteration=iteration)

                # Call LLM with tools
                response, call_tokens = await self._call_llm_with_tools(
                    messages=messages,
                    tools=tools,
                    system_prompt=system_prompt,
                    temperature=self._config.temperature,
                    max_tokens=self._config.max_tokens // self._max_iterations,
                )
                total_tokens += call_tokens

                # Extract text (thought/reasoning)
                text_content = self._extract_text_content(response)
                if text_content:
                    step.thought = text_content
                    final_text = text_content

                # Check stop reason
                stop_reason = response.get("stop_reason")
                logger.debug(f"[FunctionCalling] Iteration {iteration}: stop_reason={stop_reason}")

                if stop_reason == "end_turn":
                    step.is_final = True
                    all_steps.append(step)
                    logger.info(f"[FunctionCalling] LLM returned end_turn at iteration {iteration}")
                    break

                # Extract and execute tool use
                tool_use = self._extract_tool_use(response)
                if tool_use:
                    step.tool_use = tool_use
                    logger.info(f"[FunctionCalling] Tool use requested: {tool_use.name}")

                    # Execute the tool
                    tool_result = await self._execute_tool(tool_use)
                    step.tool_result = tool_result

                    # Build conversation for next iteration
                    messages.append(self._build_assistant_message(response))

                    # Check if result is an error
                    is_error = False
                    try:
                        result_data = json.loads(tool_result)
                        if isinstance(result_data, dict) and "error" in result_data:
                            is_error = True
                    except json.JSONDecodeError:
                        pass

                    messages.append(self._build_tool_result_message(
                        tool_use_id=tool_use.id,
                        result=tool_result,
                        is_error=is_error,
                    ))
                else:
                    # No tool use and not end_turn - unusual, but continue
                    step.is_final = True
                    all_steps.append(step)
                    logger.info(f"[FunctionCalling] No tool use at iteration {iteration}, treating as final")
                    break

                all_steps.append(step)

            # Build reasoning trace
            reasoning_trace = []
            for step in all_steps:
                prefix = f"Iteration {step.iteration}"
                if step.thought:
                    reasoning_trace.append(f"{prefix} - Thought: {step.thought[:200]}...")
                if step.tool_use:
                    reasoning_trace.append(
                        f"{prefix} - Tool: {step.tool_use.name}({json.dumps(step.tool_use.input)[:100]})"
                    )
                if step.tool_result:
                    # Truncate tool result for trace
                    result_preview = step.tool_result[:200] + "..." if len(step.tool_result) > 200 else step.tool_result
                    reasoning_trace.append(f"{prefix} - Result: {result_preview}")

            # Calculate confidence based on iterations and tool usage
            tool_count = sum(1 for s in all_steps if s.tool_use is not None)
            base_confidence = 0.6

            if tool_count > 0:
                base_confidence += min(tool_count * 0.08, 0.25)
            if iteration < self._max_iterations:
                base_confidence += 0.05

            confidence = min(base_confidence, 0.9)

            execution_time = time.time() - start_time

            # Record metrics
            self._record_execution(True, total_tokens, execution_time)

            return StrategyResult(
                answer=final_text,
                confidence=confidence,
                reasoning_trace=reasoning_trace,
                tokens_used=total_tokens,
                execution_time_seconds=execution_time,
                model=self._settings.llm.model,
                metadata={
                    "strategy": self.name,
                    "iterations": iteration,
                    "tool_calls": tool_count,
                    "max_iterations": self._max_iterations,
                    "steps": [s.to_dict() for s in all_steps],
                },
                success=True,
                started_at=started_at,
                completed_at=utc_now(),
            )

        except Exception as e:
            execution_time = time.time() - start_time
            self._record_execution(False, total_tokens, execution_time)

            logger.error(f"[FunctionCalling] Strategy failed: {e}", exc_info=True)

            return StrategyResult.from_error(
                error=str(e),
                strategy_name=self.name,
                execution_time=execution_time,
                tokens_used=total_tokens,
            )


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "FunctionCallingStrategy",
    "FunctionCallingStep",
    "ToolUseBlock",
    # Constants
    "MAX_ITERATIONS",
    "FUNCTION_CALLING_MIN_COMPLEXITY",
    "FUNCTION_CALLING_MAX_COMPLEXITY",
]
