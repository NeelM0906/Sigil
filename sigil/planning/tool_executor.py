"""Tool Step Executor for Sigil v2 plan execution.

This module implements the bridge between PlanExecutor and tool executors,
routing plan steps to appropriate executors based on step type and tool name.

The ToolStepExecutor:
- Routes TOOL_CALL steps to MCPToolExecutor or BuiltinToolExecutor
- Routes REASONING steps to the reasoning manager for response generation
- Routes workflow_knowledge tools to WorkflowKnowledgeTool
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
    the various tool executors (MCP, builtin, workflow_knowledge) and reasoning manager.

    For TOOL_CALL steps:
    - If tool_name starts with "memory." or "planning." -> BuiltinToolExecutor
    - If tool_name starts with "workflow_knowledge." -> WorkflowKnowledgeTool
    - Otherwise -> MCPToolExecutor

    For REASONING steps:
    - Route to ReasoningManager with context from prior results
    - Used for response generation and aggregation tasks

    Attributes:
        mcp_executor: Executor for MCP tools.
        builtin_executor: Executor for builtin tools.
        reasoning_manager: Optional manager for reasoning steps.
        allow_reasoning_fallback: Whether to fallback to reasoning on tool failure.
        _workflow_knowledge: Lazy-loaded workflow knowledge tool.

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
        
        # NEW: Lazy-loaded workflow knowledge tool
        self._workflow_knowledge = None

    def _get_workflow_knowledge_tool(self):
        """
        Lazy initialization of workflow knowledge tool.
        Only loads when first needed to avoid startup delays.
        """
        if self._workflow_knowledge is None:
            logger.info("🔍 Loading workflow knowledge tool...")
            try:
                from sigil.tools.workflow_knowledge import create_workflow_knowledge_tool
                self._workflow_knowledge = create_workflow_knowledge_tool(
                    index_calls=True  # Set to False if you don't want call data
                )
                logger.info("✅ Workflow knowledge tool loaded successfully")
            except Exception as e:
                logger.warning(f"⚠️  Could not load workflow knowledge tool: {e}")
                self._workflow_knowledge = None
        
        return self._workflow_knowledge

    def _determine_executor_type(self, tool_name: str) -> str:
        """Determine which executor should handle this tool.

        Args:
            tool_name: Full tool name (e.g., "websearch.search").

        Returns:
            "builtin", "workflow_knowledge", or "mcp".
        """
        # NEW: Check for workflow knowledge tools
        if tool_name.startswith("workflow_knowledge."):
            return "workflow_knowledge"
        
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

        logger.info(f"Executing step: {step.step_id} ({step.step_type.value})")

        try:
            # Route based on step type
            if step.step_type == StepType.TOOL_CALL:
                if not step.tool_name:
                    raise NoToolSpecifiedError(step.step_id)
                
                # NEW: Check if this is a workflow knowledge tool
                if step.tool_name.startswith("workflow_knowledge."):
                    output, tokens_used = await self._execute_workflow_knowledge(
                        step, prior_results
                    )
                else:
                    output, tokens_used = await self._execute_tool_call(step, prior_results)

            elif step.step_type == StepType.REASONING:
                output, tokens_used = await self._execute_reasoning(step, prior_results)

            elif step.step_type == StepType.MEMORY_QUERY:
                output = await self._execute_memory_query(step, prior_results)
                tokens_used = self._estimate_tokens(output)

            else:
                raise UnsupportedStepTypeError(step.step_id, step.step_type.value)

            # Calculate execution time
            execution_time = time.time() - start_time

            # Return successful result
            return StepResult(
                step_id=step.step_id,
                step_type=step.step_type,
                status=StepStatus.COMPLETED,
                output=output,
                tokens_used=tokens_used,
                started_at=started_at,
                completed_at=utc_now(),
                execution_time=execution_time,
            )

        except (NoToolSpecifiedError, UnsupportedStepTypeError):
            # Re-raise these specific errors
            raise

        except Exception as e:
            # Log error and create failed result
            logger.error(f"Step {step.step_id} failed: {e}")
            execution_time = time.time() - start_time

            return StepResult(
                step_id=step.step_id,
                step_type=step.step_type,
                status=StepStatus.FAILED,
                error=str(e),
                started_at=started_at,
                completed_at=utc_now(),
                execution_time=execution_time,
            )

    async def _execute_tool_call(
        self,
        step: "PlanStep",
        prior_results: dict[str, StepResult],
    ) -> tuple[str, int]:
        """Execute a TOOL_CALL step by routing to appropriate executor.

        Args:
            step: The step to execute.
            prior_results: Results from prior steps.

        Returns:
            Tuple of (tool output as string, tokens used).

        Raises:
            StepExecutionError: If tool execution fails.
        """
        tool_name = step.tool_name
        tool_args = getattr(step, "tool_args", {}) or {}
        context = self._build_execution_context(step, prior_results)

        # Determine which executor to use
        executor_type = self._determine_executor_type(tool_name)

        try:
            if executor_type == "builtin":
                logger.debug(f"Routing {tool_name} to BuiltinToolExecutor")
                output = await self._builtin_executor.execute(
                    tool_name=tool_name,
                    tool_args=tool_args,
                    context=context,
                )
                tokens_used = self._estimate_tokens(output)
                return (output, tokens_used)

            else:  # mcp
                logger.debug(f"Routing {tool_name} to MCPToolExecutor")
                try:
                    output = await self._mcp_executor.execute(
                        tool_name=tool_name,
                        tool_args=tool_args,
                    )
                    tokens_used = self._estimate_tokens(output)
                    return (output, tokens_used)

                except (MCPConnectionError, MCPTimeoutError, MCPNotAvailableError) as e:
                    # MCP-specific errors - these are often recoverable
                    if self._allow_reasoning_fallback:
                        logger.warning(
                            f"MCP tool '{tool_name}' failed with {type(e).__name__}: {e}. "
                            f"Attempting reasoning fallback..."
                        )
                        return await self._execute_reasoning_fallback(step, prior_results, str(e))
                    else:
                        # Re-raise with user-friendly message
                        self._raise_user_friendly_error(step, e)

        except Exception as e:
            # Generic errors
            if self._allow_reasoning_fallback:
                logger.warning(f"Tool execution failed: {e}. Attempting reasoning fallback...")
                return await self._execute_reasoning_fallback(step, prior_results, str(e))
            else:
                raise StepExecutionError(
                    f"Tool execution failed: {e}",
                    step_id=step.step_id,
                )

    async def _execute_workflow_knowledge(
        self,
        step: "PlanStep",
        prior_results: dict[str, StepResult],
    ) -> tuple[str, int]:
        """
        Execute a workflow_knowledge tool call.
        
        Supports:
        - workflow_knowledge.search: Search for relevant workflows
        - workflow_knowledge.ask: Ask a question
        - workflow_knowledge.examples: Get examples
        - workflow_knowledge.compare: Compare approaches
        
        Args:
            step: The step to execute.
            prior_results: Results from prior steps.
        
        Returns:
            Tuple of (output string, tokens used).
        """
        tool = self._get_workflow_knowledge_tool()
        
        if tool is None:
            return (
                "Workflow knowledge tool is not available. "
                "Please ensure ACTi Router is properly configured.",
                0
            )
        
        tool_name = step.tool_name
        tool_args = getattr(step, "tool_args", {}) or {}
        
        try:
            if tool_name == "workflow_knowledge.search":
                # Search for relevant workflows
                query = tool_args.get("query", "")
                top_k = tool_args.get("top_k", 5)
                data_type = tool_args.get("type", "all")
                
                logger.info(f"Searching workflow knowledge: '{query}'")
                results = tool.search(query, top_k, data_type)
                
                # Format results
                output = self._format_knowledge_results(results)
                tokens_used = self._estimate_tokens(output)
                
                return (output, tokens_used)
            
            elif tool_name == "workflow_knowledge.ask":
                # Ask a question
                question = tool_args.get("question", "")
                max_results = tool_args.get("max_results", 3)
                include_calls = tool_args.get("include_calls", False)
                
                logger.info(f"Answering workflow knowledge question: '{question}'")
                answer = tool.ask(question, max_results, include_calls)
                tokens_used = self._estimate_tokens(answer)
                
                return (answer, tokens_used)
            
            elif tool_name == "workflow_knowledge.examples":
                # Get examples
                topic = tool_args.get("topic", "")
                top_k = tool_args.get("top_k", 3)
                from_calls = tool_args.get("from_calls", False)
                
                logger.info(f"Getting examples for: '{topic}'")
                examples = tool.get_examples(topic, top_k, from_calls)
                tokens_used = self._estimate_tokens(examples)
                
                return (examples, tokens_used)
            
            elif tool_name == "workflow_knowledge.compare":
                # Compare approaches
                topic = tool_args.get("topic", "")
                num_approaches = tool_args.get("num", 3)
                
                logger.info(f"Comparing approaches for: '{topic}'")
                comparison = tool.compare_approaches(topic, num_approaches)
                tokens_used = self._estimate_tokens(comparison)
                
                return (comparison, tokens_used)
            
            else:
                raise ValueError(f"Unknown workflow knowledge tool: {tool_name}")
        
        except Exception as e:
            logger.error(f"Workflow knowledge tool failed: {e}")
            return (
                f"Error executing workflow knowledge tool: {str(e)}",
                0
            )

    def _format_knowledge_results(self, results) -> str:
        """
        Format knowledge results for LLM consumption.
        
        Args:
            results: List of KnowledgeResult objects.
        
        Returns:
            Formatted string.
        """
        if not results:
            return "No relevant workflow knowledge found."
        
        formatted = "## Workflow Knowledge Results\n\n"
        
        for i, result in enumerate(results, 1):
            formatted += f"### {i}. {result.source_name} ({result.source_type})\n"
            formatted += f"**Relevance:** {result.relevance_score:.0%}\n\n"
            formatted += f"{result.content}\n\n"
            formatted += "---\n\n"
        
        return formatted

    def _raise_user_friendly_error(
        self,
        step: "PlanStep",
        error: MCPExecutorError,
    ) -> None:
        """Raise a user-friendly error for MCP failures.

        Args:
            step: The step that failed.
            error: The MCP error that occurred.

        Raises:
            StepExecutionError: With user-friendly message.
        """
        if isinstance(error, MCPConnectionError):
            user_message = (
                f"Unable to connect to the {step.tool_name} service. "
                "Please check that the MCP server is running and configured correctly."
            )
        elif isinstance(error, MCPTimeoutError):
            user_message = (
                f"The {step.tool_name} service timed out. "
                "The request may be too complex or the service may be unavailable."
            )
        elif isinstance(error, MCPNotAvailableError):
            user_message = (
                f"The {step.tool_name} service is not available. "
                "Please check your configuration and ensure all required services are set up."
            )
        else:
            user_message = f"The {step.tool_name} service encountered an error."

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
