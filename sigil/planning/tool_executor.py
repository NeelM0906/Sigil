"""Tool Step Executor for Sigil v2 plan execution.

This module implements the bridge between PlanExecutor and tool executors,
routing plan steps to appropriate executors based on step type and tool name.

The ToolStepExecutor:
- Routes TOOL_CALL steps to MCPToolExecutor or BuiltinToolExecutor
- Routes REASONING steps to the reasoning manager for response generation
- Returns proper StepResult objects with correct token counts
- Handles errors gracefully with fallback to LLM reasoning
- Fails fast with clear error messages instead of hanging

Example:
    >>> executor = ToolStepExecutor(
    ...     mcp_executor=MCPToolExecutor(),
    ...     builtin_executor=BuiltinToolExecutor(memory_manager=mm),
    ...     reasoning_manager=reasoning_manager,
    ...     allow_reasoning_fallback=True,  # Enable fallback on MCP failure
    ... )
    >>> result = await executor.execute_step(plan_step, prior_results)
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any, Optional, TYPE_CHECKING

from sigil.planning.schemas import (
    StepResult,
    StepStatus,
    StepType,
    utc_now,
)
from sigil.planning.executors.mcp_executor import (
    MCPToolExecutor,
    MCPConnectionError,
    MCPTimeoutError,
    MCPNotAvailableError,
    MCPExecutorError,
)
from sigil.planning.executors.builtin_executor import BuiltinToolExecutor
from sigil.core.exceptions import SigilError
from sigil.reasoning.strategies.utils import format_tool_results_section

if TYPE_CHECKING:
    from sigil.config.schemas.plan import PlanStep
    from sigil.reasoning.manager import ReasoningManager


# =============================================================================
# Module Logger
# =============================================================================

logger = logging.getLogger(__name__)


# =============================================================================
# Exceptions
# =============================================================================


class StepExecutionError(SigilError):
    """Error during step execution."""

    def __init__(
        self,
        message: str,
        step_id: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(message, code="STEP_EXECUTION_ERROR", **kwargs)
        self.step_id = step_id


class NoToolSpecifiedError(StepExecutionError):
    """Raised when a TOOL_CALL step has no tool specified."""

    def __init__(self, step_id: str, **kwargs: Any) -> None:
        super().__init__(
            f"Step '{step_id}' is a TOOL_CALL but has no tool_name specified",
            step_id=step_id,
            **kwargs,
        )


class UnsupportedStepTypeError(StepExecutionError):
    """Raised when step type is not supported."""

    def __init__(
        self,
        step_id: str,
        step_type: str,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            f"Step '{step_id}' has unsupported step type: {step_type}",
            step_id=step_id,
            **kwargs,
        )
        self.step_type = step_type


# =============================================================================
# Tool Step Executor
# =============================================================================


class ToolStepExecutor:
    """Routes plan steps to appropriate tool executors.

    The ToolStepExecutor serves as the bridge between the PlanExecutor and
    the various tool executors (MCP, builtin) and reasoning manager.

    For TOOL_CALL steps:
    - If tool_name starts with "memory." or "planning." -> BuiltinToolExecutor
    - Otherwise -> MCPToolExecutor

    For REASONING steps:
    - Route to ReasoningManager with context from prior results
    - Used for response generation and aggregation tasks

    Attributes:
        mcp_executor: Executor for MCP tools.
        builtin_executor: Executor for builtin tools.
        reasoning_manager: Optional manager for reasoning steps.
        allow_reasoning_fallback: Whether to fallback to reasoning on tool failure.

    Example:
        >>> executor = ToolStepExecutor(
        ...     mcp_executor=MCPToolExecutor(),
        ...     builtin_executor=BuiltinToolExecutor(memory_manager=mm),
        ... )
        >>> result = await executor.execute_step(step, prior_results)
    """

    def __init__(
        self,
        mcp_executor: Optional[MCPToolExecutor] = None,
        builtin_executor: Optional[BuiltinToolExecutor] = None,
        reasoning_manager: Optional["ReasoningManager"] = None,
        allow_reasoning_fallback: bool = False,
    ) -> None:
        """Initialize the tool step executor.

        Args:
            mcp_executor: Executor for MCP tools.
            builtin_executor: Executor for builtin tools.
            reasoning_manager: Optional manager for reasoning steps.
            allow_reasoning_fallback: Whether to fallback to reasoning on failure.
        """
        # Initialize executors with appropriate timeouts
        # Connection timeout: 5 seconds (quick fail if server is down)
        # Tool execution timeout: 30 seconds (per tool call, accounts for API latency)
        # Wrapper timeout: 5 + 10 = 15 seconds for initial calculation
        # Note: Actual tool execution can take up to 30 seconds due to external API latency
        self._mcp_executor = mcp_executor or MCPToolExecutor(
            connection_timeout=5.0,
            tool_execution_timeout=30.0,
        )
        self._builtin_executor = builtin_executor or BuiltinToolExecutor()
        self._reasoning_manager = reasoning_manager
        self._allow_reasoning_fallback = allow_reasoning_fallback

    def _determine_executor_type(self, tool_name: str) -> str:
        """Determine which executor should handle this tool.

        Args:
            tool_name: Full tool name (e.g., "websearch.search").

        Returns:
            "builtin" or "mcp".
        """
        if BuiltinToolExecutor.is_builtin_tool(tool_name):
            return "builtin"
        return "mcp"

    def _build_execution_context(
        self,
        step: "PlanStep",
        prior_results: dict[str, StepResult],
    ) -> dict[str, Any]:
        """Build context for tool execution from prior results.

        Args:
            step: Current step being executed.
            prior_results: Results from prior steps.

        Returns:
            Context dictionary.
        """
        # Extract outputs from prior results
        prior_outputs = {}
        for step_id, result in prior_results.items():
            if result.status == StepStatus.COMPLETED and result.output:
                prior_outputs[step_id] = {
                    "output": result.output,
                    "tokens_used": result.tokens_used,
                }

        return {
            "prior_results": prior_results,
            "prior_outputs": prior_outputs,
            "step_description": step.description,
            "step_dependencies": getattr(step, "dependencies", []),
        }

    async def execute_step(
        self,
        step: "PlanStep",
        prior_results: dict[str, StepResult],
    ) -> StepResult:
        """Execute a plan step by routing to the appropriate executor.

        This is the main entry point matching the StepExecutorFn signature
        expected by PlanExecutor.

        Args:
            step: The step to execute.
            prior_results: Results from prior steps (keyed by step_id).

        Returns:
            StepResult with execution outcome.

        Raises:
            NoToolSpecifiedError: If TOOL_CALL step has no tool_name.
            UnsupportedStepTypeError: If step type is not supported.
            StepExecutionError: If execution fails.
        """
        started_at = utc_now()
        start_time = time.time()

        logger.debug(f"Executing step {step.step_id}: {step.description[:50]}...")

        try:
            # Determine step type - check for tool_calls first (for compatibility)
            # In the current schema, tool_calls is used for listing tools
            # We need to check if this is a TOOL_CALL type step
            step_type = self._determine_step_type(step)

            # Track tokens - tool calls use 0 tokens, reasoning returns actual count
            tokens_used = 0
            result: str

            if step_type == StepType.TOOL_CALL:
                result = await self._execute_tool_call(step, prior_results)
                tokens_used = 0  # Tool calls don't use LLM tokens
            elif step_type == StepType.REASONING:
                result, tokens_used = await self._execute_reasoning(step, prior_results)
            elif step_type == StepType.MEMORY_QUERY:
                result = await self._execute_memory_query(step, prior_results)
                tokens_used = 0  # Memory queries don't use LLM tokens
            else:
                # For unknown types, try to execute based on available info
                if self._has_tool_info(step):
                    result = await self._execute_tool_call(step, prior_results)
                    tokens_used = 0
                elif self._allow_reasoning_fallback and self._reasoning_manager:
                    result, tokens_used = await self._execute_reasoning(step, prior_results)
                else:
                    raise UnsupportedStepTypeError(step.step_id, str(step_type))

            duration_ms = (time.time() - start_time) * 1000

            logger.debug(
                f"Step {step.step_id} completed: type={step_type.value}, "
                f"tokens={tokens_used}, duration={duration_ms:.1f}ms"
            )

            return StepResult(
                step_id=step.step_id,
                status=StepStatus.COMPLETED,
                output=result,
                tokens_used=tokens_used,
                duration_ms=duration_ms,
                started_at=started_at,
                completed_at=utc_now(),
            )

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            logger.error(f"Step {step.step_id} failed: {e}")

            # Try reasoning fallback if enabled
            if self._allow_reasoning_fallback and self._reasoning_manager:
                try:
                    logger.info(f"Attempting reasoning fallback for step {step.step_id}")
                    fallback_result, fallback_tokens = await self._execute_reasoning_fallback(
                        step, prior_results, str(e)
                    )
                    return StepResult(
                        step_id=step.step_id,
                        status=StepStatus.COMPLETED,
                        output=fallback_result,
                        tokens_used=fallback_tokens,
                        duration_ms=(time.time() - start_time) * 1000,
                        started_at=started_at,
                        completed_at=utc_now(),
                    )
                except Exception as fallback_error:
                    logger.error(f"Reasoning fallback also failed: {fallback_error}")

            return StepResult(
                step_id=step.step_id,
                status=StepStatus.FAILED,
                error=str(e),
                duration_ms=duration_ms,
                started_at=started_at,
                completed_at=utc_now(),
            )

    def _determine_step_type(self, step: "PlanStep") -> StepType:
        """Determine the step type from step attributes.

        The PlanStep schema uses tool_calls list for available tools,
        but PlanStepConfig has step_type. We need to handle both.

        Args:
            step: The plan step.

        Returns:
            StepType enum value.
        """
        # Check if step has explicit step_type (PlanStepConfig)
        if hasattr(step, "step_type") and step.step_type:
            if isinstance(step.step_type, StepType):
                return step.step_type
            try:
                return StepType(step.step_type)
            except (ValueError, TypeError):
                pass

        # Infer from tool_calls (PlanStep from plan.py)
        if hasattr(step, "tool_calls") and step.tool_calls:
            return StepType.TOOL_CALL

        # Infer from tool_name (PlanStepConfig)
        if hasattr(step, "tool_name") and step.tool_name:
            return StepType.TOOL_CALL

        # Default to REASONING if description suggests it
        description = step.description.lower()
        if any(kw in description for kw in ["analyze", "summarize", "aggregate", "format", "generate response"]):
            return StepType.REASONING

        # Check for memory-related keywords
        if any(kw in description for kw in ["recall", "remember", "retrieve memory", "memory"]):
            return StepType.MEMORY_QUERY

        # Default to REASONING for ambiguous cases
        return StepType.REASONING

    def _has_tool_info(self, step: "PlanStep") -> bool:
        """Check if step has tool execution information.

        Args:
            step: The plan step.

        Returns:
            True if step has tool_name or tool_calls.
        """
        if hasattr(step, "tool_name") and step.tool_name:
            return True
        if hasattr(step, "tool_calls") and step.tool_calls:
            return True
        return False

    def _get_tool_name(self, step: "PlanStep") -> Optional[str]:
        """Extract tool name from step.

        Args:
            step: The plan step.

        Returns:
            Tool name if available, None otherwise.
        """
        if hasattr(step, "tool_name") and step.tool_name:
            return step.tool_name
        if hasattr(step, "tool_calls") and step.tool_calls:
            return step.tool_calls[0]  # Use first tool
        return None

    def _get_tool_args(self, step: "PlanStep") -> dict[str, Any]:
        """Extract tool arguments from step.

        Args:
            step: The plan step.

        Returns:
            Tool arguments dictionary.
        """
        # Check for tool_args (standard attribute)
        if hasattr(step, "tool_args") and step.tool_args:
            return step.tool_args
        # Check for _tool_args (set by planner's _enrich_steps_with_tools)
        if hasattr(step, "_tool_args") and step._tool_args:
            return step._tool_args
        # Check for _parsed_tool_args (set during plan parsing)
        if hasattr(step, "_parsed_tool_args") and step._parsed_tool_args:
            return step._parsed_tool_args
        return {}

    async def _execute_tool_call(
        self,
        step: "PlanStep",
        prior_results: dict[str, StepResult],
    ) -> str:
        """Execute a TOOL_CALL step.

        For MCP tools, includes timeout protection and detailed error messages.
        On MCP failure, can fall back to reasoning if enabled.

        Args:
            step: The step to execute.
            prior_results: Results from prior steps.

        Returns:
            Tool output as string.

        Raises:
            NoToolSpecifiedError: If no tool name is specified.
            StepExecutionError: If tool execution fails.
        """
        tool_name = self._get_tool_name(step)
        if not tool_name:
            raise NoToolSpecifiedError(step.step_id)

        tool_args = self._get_tool_args(step)
        context = self._build_execution_context(step, prior_results)

        executor_type = self._determine_executor_type(tool_name)

        logger.info(f"[ToolExecutor] Executing tool '{tool_name}' via {executor_type}")

        if executor_type == "builtin":
            return await self._builtin_executor.execute(
                tool_name=tool_name,
                tool_args=tool_args,
                context=context,
            )
        else:
            # MCP tool execution with timeout and detailed error handling
            return await self._execute_mcp_tool_with_fallback(
                tool_name=tool_name,
                tool_args=tool_args,
                step=step,
                prior_results=prior_results,
            )

    async def _execute_mcp_tool_with_fallback(
        self,
        tool_name: str,
        tool_args: dict[str, Any],
        step: "PlanStep",
        prior_results: dict[str, StepResult],
    ) -> str:
        """Execute an MCP tool with timeout protection and optional reasoning fallback.

        This method:
        1. Wraps MCP execution in a timeout to prevent indefinite hangs
        2. Logs detailed error information for debugging
        3. Falls back to LLM reasoning if enabled and tool execution fails

        Args:
            tool_name: The MCP tool to execute.
            tool_args: Arguments for the tool.
            step: The plan step being executed.
            prior_results: Results from prior steps.

        Returns:
            Tool output or fallback reasoning result.

        Raises:
            StepExecutionError: If execution fails and no fallback is available.
        """
        # Use the MCP executor's tool execution timeout with buffer for network delays
        # This gives the tool the full time it needs to execute before timing out
        tool_timeout = self._mcp_executor._tool_execution_timeout

        try:
            logger.debug(
                f"[ToolExecutor] Calling MCP tool '{tool_name}' with args: {tool_args}"
            )

            # Wrap MCP execution in a timeout
            result = await asyncio.wait_for(
                self._mcp_executor.execute(
                    tool_name=tool_name,
                    tool_args=tool_args,
                ),
                timeout=tool_timeout,
            )

            logger.info(f"[ToolExecutor] MCP tool '{tool_name}' completed successfully")
            return result

        except asyncio.TimeoutError:
            error_msg = (
                f"MCP tool '{tool_name}' timed out after {tool_timeout}s. "
                f"The MCP server may be unresponsive or hanging."
            )
            logger.error(f"[ToolExecutor] {error_msg}")
            return await self._handle_mcp_failure(
                tool_name=tool_name,
                tool_args=tool_args,
                error=error_msg,
                error_type="timeout",
                step=step,
                prior_results=prior_results,
            )

        except MCPTimeoutError as e:
            logger.error(f"[ToolExecutor] MCP timeout: {e}")
            return await self._handle_mcp_failure(
                tool_name=tool_name,
                tool_args=tool_args,
                error=str(e),
                error_type="timeout",
                step=step,
                prior_results=prior_results,
            )

        except MCPNotAvailableError as e:
            logger.error(f"[ToolExecutor] MCP not available: {e}")
            return await self._handle_mcp_failure(
                tool_name=tool_name,
                tool_args=tool_args,
                error=str(e),
                error_type="not_available",
                step=step,
                prior_results=prior_results,
            )

        except MCPConnectionError as e:
            logger.error(f"[ToolExecutor] MCP connection error: {e}")
            return await self._handle_mcp_failure(
                tool_name=tool_name,
                tool_args=tool_args,
                error=str(e),
                error_type=e.error_type,
                step=step,
                prior_results=prior_results,
            )

        except MCPExecutorError as e:
            logger.error(f"[ToolExecutor] MCP executor error: {e}")
            return await self._handle_mcp_failure(
                tool_name=tool_name,
                tool_args=tool_args,
                error=str(e),
                error_type="execution",
                step=step,
                prior_results=prior_results,
            )

        except Exception as e:
            logger.error(
                f"[ToolExecutor] Unexpected error executing '{tool_name}': "
                f"{type(e).__name__}: {e}",
                exc_info=True,
            )
            return await self._handle_mcp_failure(
                tool_name=tool_name,
                tool_args=tool_args,
                error=str(e),
                error_type="unknown",
                step=step,
                prior_results=prior_results,
            )

    async def _handle_mcp_failure(
        self,
        tool_name: str,
        tool_args: dict[str, Any],
        error: str,
        error_type: str,
        step: "PlanStep",
        prior_results: dict[str, StepResult],
    ) -> str:
        """Handle MCP tool execution failure with optional fallback to reasoning.

        Args:
            tool_name: The failed tool name.
            tool_args: The tool arguments that were passed.
            error: The error message.
            error_type: Type of error (timeout, not_available, connection, etc).
            step: The plan step being executed.
            prior_results: Results from prior steps.

        Returns:
            Fallback reasoning result or error message.

        Raises:
            StepExecutionError: If fallback is disabled or fails.
        """
        # Create a user-friendly error message
        if error_type == "timeout":
            user_message = (
                f"The '{tool_name}' tool is taking too long to respond. "
                f"The external service may be down or overloaded."
            )
        elif error_type == "not_available":
            user_message = (
                f"The '{tool_name}' tool is not available. "
                f"This may be due to missing configuration or credentials."
            )
        elif error_type == "connection_refused":
            user_message = (
                f"Could not connect to the '{tool_name}' service. "
                f"The server may not be running."
            )
        else:
            user_message = f"The '{tool_name}' tool encountered an error: {error}"

        logger.warning(
            f"[ToolExecutor] MCP failure handled. Type: {error_type}. "
            f"Fallback enabled: {self._allow_reasoning_fallback}"
        )

        # Try reasoning fallback if enabled
        if self._allow_reasoning_fallback and self._reasoning_manager:
            logger.info(
                f"[ToolExecutor] Attempting reasoning fallback for failed tool "
                f"'{tool_name}'"
            )
            try:
                context = self._build_execution_context(step, prior_results)
                context["failed_tool"] = tool_name
                context["failed_tool_args"] = tool_args
                context["error"] = error
                context["error_type"] = error_type

                task = (
                    f"The tool '{tool_name}' could not be executed. "
                    f"Error: {error}. "
                    f"Please provide an alternative response based on the task: "
                    f"'{step.description}'. Use your knowledge to help the user "
                    f"as best as possible without the external tool."
                )

                result = await self._reasoning_manager.execute(
                    task=task,
                    context=context,
                    complexity=0.5,
                )

                # Extract answer from result
                if hasattr(result, "answer"):
                    fallback_response = result.answer
                elif hasattr(result, "output"):
                    fallback_response = result.output
                else:
                    fallback_response = str(result)

                logger.info(
                    f"[ToolExecutor] Reasoning fallback successful for '{tool_name}'"
                )

                # Prepend a note about fallback
                return (
                    f"[Note: External tool unavailable, using reasoning fallback]\n\n"
                    f"{fallback_response}"
                )

            except Exception as fallback_error:
                logger.error(
                    f"[ToolExecutor] Reasoning fallback also failed: {fallback_error}"
                )
                # Fall through to raise error

        # No fallback available - raise error with clear message
        raise StepExecutionError(
            f"{user_message}\n\nDetails: {error}",
            step_id=step.step_id,
        )

    async def _execute_reasoning(
        self,
        step: "PlanStep",
        prior_results: dict[str, StepResult],
    ) -> tuple[str, int]:
        """Execute a REASONING step using the reasoning manager.

        Args:
            step: The step to execute.
            prior_results: Results from prior steps.

        Returns:
            Tuple of (reasoning output as string, tokens used).
        """
        if self._reasoning_manager is None:
            # Without reasoning manager, return a simple aggregation
            output = self._aggregate_prior_outputs(step, prior_results)
            return (output, 0)

        # Get reasoning task from step
        base_task = getattr(step, "reasoning_task", None) or step.description
        context = self._build_execution_context(step, prior_results)

        # Extract and format tool outputs for explicit inclusion in task
        prior_outputs = context.get("prior_outputs", {})
        if prior_outputs:
            tool_results_section = format_tool_results_section(prior_outputs)
            if tool_results_section:
                reasoning_task = f"""Based on the following tool results, {base_task}:

## Tool Results
{tool_results_section}

Generate a comprehensive, well-structured response for the user."""
            else:
                reasoning_task = base_task
        else:
            reasoning_task = base_task

        # Execute through reasoning manager
        try:
            result = await self._reasoning_manager.execute(
                task=reasoning_task,
                context=context,
                complexity=0.3,  # Low complexity for response generation
            )
            # Extract tokens used from result
            tokens_used = getattr(result, "tokens_used", 0)

            # StrategyResult has 'answer' attribute, not 'output'
            if hasattr(result, "answer"):
                return (result.answer, tokens_used)
            elif hasattr(result, "output"):
                return (result.output, tokens_used)
            else:
                return (str(result), tokens_used)
        except Exception as e:
            logger.error(f"Reasoning manager failed: {e}")
            # Fallback to simple aggregation
            output = self._aggregate_prior_outputs(step, prior_results)
            return (output, 0)

    async def _execute_memory_query(
        self,
        step: "PlanStep",
        prior_results: dict[str, StepResult],
    ) -> str:
        """Execute a MEMORY_QUERY step.

        Args:
            step: The step to execute.
            prior_results: Results from prior steps.

        Returns:
            Memory query result as string.
        """
        # Extract query from step description
        query = step.description
        context = self._build_execution_context(step, prior_results)

        # Execute as memory.recall
        return await self._builtin_executor.execute(
            tool_name="memory.recall",
            tool_args={"query": query, "k": 5},
            context=context,
        )

    async def _execute_reasoning_fallback(
        self,
        step: "PlanStep",
        prior_results: dict[str, StepResult],
        error: str,
    ) -> tuple[str, int]:
        """Execute reasoning fallback when tool execution fails.

        Args:
            step: The step that failed.
            prior_results: Results from prior steps.
            error: The error message from the failed execution.

        Returns:
            Tuple of (reasoning fallback output, tokens used).
        """
        if self._reasoning_manager is None:
            return (f"Tool execution failed: {error}", 0)

        context = self._build_execution_context(step, prior_results)
        context["error"] = error
        context["fallback"] = True

        task = f"The tool execution for '{step.description}' failed with error: {error}. Please provide an alternative response or explanation."

        result = await self._reasoning_manager.execute(
            task=task,
            context=context,
            complexity=0.4,
        )
        # Extract tokens used from result
        tokens_used = getattr(result, "tokens_used", 0)

        # StrategyResult has 'answer' attribute, not 'output'
        if hasattr(result, "answer"):
            return (result.answer, tokens_used)
        elif hasattr(result, "output"):
            return (result.output, tokens_used)
        else:
            return (str(result), tokens_used)

    def _aggregate_prior_outputs(
        self,
        step: "PlanStep",
        prior_results: dict[str, StepResult],
    ) -> str:
        """Aggregate outputs from prior steps into a simple response.

        Used as fallback when no reasoning manager is available.

        Args:
            step: Current step.
            prior_results: Results from prior steps.

        Returns:
            Aggregated output string.
        """
        outputs = []
        for step_id, result in prior_results.items():
            if result.status == StepStatus.COMPLETED and result.output:
                outputs.append(f"[{step_id}]: {result.output}")

        if outputs:
            return f"Step '{step.description}' completed. Prior results:\n" + "\n".join(outputs)
        return f"Step '{step.description}' completed."

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count for a text string.

        Args:
            text: Text to estimate tokens for.

        Returns:
            Estimated token count.
        """
        # Rough estimate: ~4 characters per token
        return len(str(text)) // 4 + 50


# =============================================================================
# Factory Function
# =============================================================================


def create_tool_step_executor(
    memory_manager: Optional[Any] = None,
    planner: Optional[Any] = None,
    reasoning_manager: Optional[Any] = None,
    allow_reasoning_fallback: bool = True,
    mcp_connection_timeout: float = 15.0,
    mcp_tool_execution_timeout: float = 30.0,
) -> ToolStepExecutor:
    """Factory function to create a configured ToolStepExecutor.

    Args:
        memory_manager: MemoryManager for builtin memory tools.
        planner: Planner for builtin planning tools.
        reasoning_manager: ReasoningManager for reasoning steps.
        allow_reasoning_fallback: Whether to fallback to reasoning on failure.
            Defaults to True to ensure users get responses even when MCP fails.
        mcp_connection_timeout: Timeout for MCP server connections in seconds.
            Defaults to 15 seconds to fail fast instead of hanging.
        mcp_tool_execution_timeout: Timeout for individual MCP tool executions in seconds.
            Defaults to 30 seconds to account for external API latency.

    Returns:
        Configured ToolStepExecutor.

    Note:
        The reasoning fallback is enabled by default. When MCP tools fail
        (due to timeout, missing credentials, or server unavailability),
        the executor will attempt to use LLM reasoning to provide a response
        based on the task description and available context.
    """
    mcp_executor = MCPToolExecutor(
        connection_timeout=mcp_connection_timeout,
        tool_execution_timeout=mcp_tool_execution_timeout,
    )
    builtin_executor = BuiltinToolExecutor(
        memory_manager=memory_manager,
        planner=planner,
    )

    return ToolStepExecutor(
        mcp_executor=mcp_executor,
        builtin_executor=builtin_executor,
        reasoning_manager=reasoning_manager,
        allow_reasoning_fallback=allow_reasoning_fallback,
    )


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "ToolStepExecutor",
    "create_tool_step_executor",
    "StepExecutionError",
    "NoToolSpecifiedError",
    "UnsupportedStepTypeError",
]
