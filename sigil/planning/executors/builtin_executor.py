"""Builtin Tool Executor for Sigil v2 plan execution.

This module implements direct execution of builtin Sigil tools (memory, planning)
without LLM reasoning, enabling efficient tool calls during plan execution.

The BuiltinToolExecutor:
- Routes memory.* tools to MemoryManager methods
- Routes planning.* tools to Planner methods
- Supports argument interpolation from prior step results
- Returns JSON-formatted outputs

Example:
    >>> executor = BuiltinToolExecutor(memory_manager=mm, planner=planner)
    >>> result = await executor.execute(
    ...     tool_name="memory.recall",
    ...     tool_args={"query": "user preferences", "k": 5}
    ... )
"""

from __future__ import annotations

import json
import logging
from typing import Any, Optional, TYPE_CHECKING

from sigil.core.exceptions import SigilError

if TYPE_CHECKING:
    from sigil.memory.manager import MemoryManager
    from sigil.planning.planner import Planner


# =============================================================================
# Module Logger
# =============================================================================

logger = logging.getLogger(__name__)


# =============================================================================
# Exceptions
# =============================================================================


class BuiltinExecutorError(SigilError):
    """Base exception for builtin executor errors."""

    def __init__(
        self,
        message: str,
        tool_name: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(message, code="BUILTIN_EXECUTOR_ERROR", **kwargs)
        self.tool_name = tool_name


class ToolNotFoundError(BuiltinExecutorError):
    """Raised when requested builtin tool is not found."""

    def __init__(
        self,
        tool_name: str,
        available_tools: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> None:
        available = ", ".join(available_tools[:10]) if available_tools else "none"
        super().__init__(
            f"Builtin tool '{tool_name}' not found. Available tools: {available}",
            tool_name=tool_name,
            **kwargs,
        )
        self.available_tools = available_tools or []


class MissingDependencyError(BuiltinExecutorError):
    """Raised when a required dependency (manager, planner) is not provided."""

    def __init__(
        self,
        tool_name: str,
        dependency: str,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            f"Tool '{tool_name}' requires {dependency} but it was not provided",
            tool_name=tool_name,
            **kwargs,
        )
        self.dependency = dependency


class ToolExecutionError(BuiltinExecutorError):
    """Raised when builtin tool execution fails."""

    def __init__(
        self,
        tool_name: str,
        original_error: Exception,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            f"Execution of builtin tool '{tool_name}' failed: {original_error}",
            tool_name=tool_name,
            **kwargs,
        )
        self.original_error = original_error


# =============================================================================
# Builtin Tool Executor
# =============================================================================


class BuiltinToolExecutor:
    """Executes builtin Sigil tools directly without LLM reasoning.

    This executor handles tools from the memory and planning subsystems:
    - memory.recall: Retrieve memories by query
    - memory.store (memory.remember): Store a new fact
    - memory.list_categories: List available categories
    - memory.get_category: Get category content
    - planning.create_plan: Create a new plan
    - planning.get_status: Get plan status

    Attributes:
        memory_manager: Optional MemoryManager for memory tools.
        planner: Optional Planner for planning tools.

    Example:
        >>> executor = BuiltinToolExecutor(
        ...     memory_manager=memory_manager,
        ...     planner=planner
        ... )
        >>> result = await executor.execute(
        ...     "memory.recall",
        ...     {"query": "user preferences"}
        ... )
    """

    # List of supported tools
    SUPPORTED_TOOLS = [
        "memory.recall",
        "memory.retrieve",
        "memory.store",
        "memory.remember",
        "memory.list_categories",
        "memory.get_category",
        "planning.create_plan",
        "planning.get_status",
    ]

    def __init__(
        self,
        memory_manager: Optional["MemoryManager"] = None,
        planner: Optional["Planner"] = None,
    ) -> None:
        """Initialize the builtin tool executor.

        Args:
            memory_manager: Optional MemoryManager for memory tools.
            planner: Optional Planner for planning tools.
        """
        self._memory_manager = memory_manager
        self._planner = planner

    @staticmethod
    def parse_tool_name(tool_name: str) -> tuple[str, str]:
        """Parse a tool name into category and operation.

        Args:
            tool_name: Full tool name (e.g., "memory.recall").

        Returns:
            Tuple of (category, operation).

        Raises:
            ValueError: If tool name format is invalid.
        """
        parts = tool_name.split(".", 1)
        if len(parts) != 2:
            raise ValueError(
                f"Invalid tool name format: '{tool_name}'. "
                f"Expected format: 'category.operation'"
            )
        return parts[0], parts[1]

    @staticmethod
    def is_builtin_tool(tool_name: str) -> bool:
        """Check if a tool name refers to a builtin tool.

        Args:
            tool_name: Tool name to check.

        Returns:
            True if this is a builtin tool.
        """
        try:
            category, _ = BuiltinToolExecutor.parse_tool_name(tool_name)
            return category in ("memory", "planning")
        except ValueError:
            return False

    def interpolate_args(
        self,
        tool_args: dict[str, Any],
        context: dict[str, Any],
    ) -> dict[str, Any]:
        """Interpolate argument values from context.

        Supports template syntax like "{{step_1.output}}" to reference
        prior step results.

        Args:
            tool_args: Original tool arguments.
            context: Context with prior_results and other data.

        Returns:
            Interpolated arguments.
        """
        prior_results = context.get("prior_results", {})

        def interpolate_value(value: Any) -> Any:
            if not isinstance(value, str):
                return value

            # Check for template pattern {{...}}
            if "{{" in value and "}}" in value:
                import re
                pattern = r"\{\{([^}]+)\}\}"
                matches = re.findall(pattern, value)

                for match in matches:
                    # Parse reference like "step_1.output" or "step_1.output.data"
                    parts = match.strip().split(".")
                    if len(parts) >= 2:
                        step_id = parts[0]
                        attr_path = parts[1:]

                        # Get value from prior results
                        if step_id in prior_results:
                            result = prior_results[step_id]
                            resolved = result.output if hasattr(result, "output") else result

                            # Navigate attribute path
                            for attr in attr_path:
                                if isinstance(resolved, dict):
                                    resolved = resolved.get(attr, resolved)
                                elif hasattr(resolved, attr):
                                    resolved = getattr(resolved, attr)

                            # Replace in value
                            placeholder = "{{" + match + "}}"
                            if value == placeholder:
                                # Full replacement
                                return resolved
                            else:
                                # Partial replacement
                                value = value.replace(placeholder, str(resolved))

            return value

        return {k: interpolate_value(v) for k, v in tool_args.items()}

    async def execute(
        self,
        tool_name: str,
        tool_args: dict[str, Any],
        context: Optional[dict[str, Any]] = None,
    ) -> str:
        """Execute a builtin tool with the given arguments.

        Args:
            tool_name: Full tool name (e.g., "memory.recall").
            tool_args: Arguments to pass to the tool.
            context: Optional context with prior_results for interpolation.

        Returns:
            Tool execution result as a JSON string.

        Raises:
            ValueError: If tool name format is invalid.
            ToolNotFoundError: If tool is not found.
            MissingDependencyError: If required manager is not provided.
            ToolExecutionError: If tool execution fails.
        """
        # Parse tool name
        try:
            category, operation = self.parse_tool_name(tool_name)
        except ValueError as e:
            raise BuiltinExecutorError(str(e), tool_name=tool_name)

        # Interpolate arguments if context provided
        if context:
            tool_args = self.interpolate_args(tool_args, context)

        logger.debug(f"Executing builtin tool: {tool_name} with args: {tool_args}")

        # Route to appropriate handler
        if category == "memory":
            return await self._execute_memory_tool(operation, tool_args)
        elif category == "planning":
            return await self._execute_planning_tool(operation, tool_args)
        else:
            raise ToolNotFoundError(
                tool_name=tool_name,
                available_tools=self.SUPPORTED_TOOLS,
            )

    async def _execute_memory_tool(
        self,
        operation: str,
        tool_args: dict[str, Any],
    ) -> str:
        """Execute a memory tool.

        Args:
            operation: Tool operation (recall, store, etc.).
            tool_args: Tool arguments.

        Returns:
            JSON-formatted result.
        """
        if self._memory_manager is None:
            raise MissingDependencyError(
                tool_name=f"memory.{operation}",
                dependency="memory_manager",
            )

        try:
            if operation in ("recall", "retrieve"):
                query = tool_args.get("query", "")
                k = tool_args.get("k", 5)
                category = tool_args.get("category")
                mode = tool_args.get("mode", "hybrid")

                results = await self._memory_manager.recall(
                    query=query,
                    k=k,
                    category=category,
                    mode=mode,
                )

                return json.dumps({
                    "status": "success",
                    "memories": results,
                    "count": len(results),
                })

            elif operation in ("store", "remember"):
                content = tool_args.get("content", "")
                category = tool_args.get("category")
                confidence = tool_args.get("confidence", 1.0)
                session_id = tool_args.get("session_id")

                item = await self._memory_manager.remember(
                    content=content,
                    category=category,
                    session_id=session_id,
                    confidence=confidence,
                )

                return json.dumps({
                    "status": "success",
                    "item_id": item.item_id,
                    "message": f"Stored: {content[:50]}..." if len(content) > 50 else f"Stored: {content}",
                })

            elif operation == "list_categories":
                categories = await self._memory_manager.list_categories()

                return json.dumps({
                    "status": "success",
                    "categories": categories,
                    "count": len(categories),
                })

            elif operation == "get_category":
                name = tool_args.get("name", "")
                content = await self._memory_manager.get_category_content(name)

                if content is None:
                    return json.dumps({
                        "status": "not_found",
                        "message": f"Category '{name}' not found.",
                    })

                return json.dumps({
                    "status": "success",
                    "name": name,
                    "content": content,
                })

            else:
                raise ToolNotFoundError(
                    tool_name=f"memory.{operation}",
                    available_tools=[t for t in self.SUPPORTED_TOOLS if t.startswith("memory.")],
                )

        except ToolNotFoundError:
            raise
        except Exception as e:
            logger.error(f"Memory tool '{operation}' failed: {e}")
            raise ToolExecutionError(f"memory.{operation}", e)

    async def _execute_planning_tool(
        self,
        operation: str,
        tool_args: dict[str, Any],
    ) -> str:
        """Execute a planning tool.

        Args:
            operation: Tool operation (create_plan, get_status, etc.).
            tool_args: Tool arguments.

        Returns:
            JSON-formatted result.
        """
        if self._planner is None:
            raise MissingDependencyError(
                tool_name=f"planning.{operation}",
                dependency="planner",
            )

        try:
            if operation == "create_plan":
                goal = tool_args.get("goal", "")
                context = tool_args.get("context", {})
                tools = tool_args.get("tools", [])
                max_steps = tool_args.get("max_steps", 10)
                session_id = tool_args.get("session_id")

                # Parse context if it's a string
                if isinstance(context, str):
                    try:
                        context = json.loads(context)
                    except json.JSONDecodeError:
                        context = {"raw_context": context}

                plan = await self._planner.create_plan(
                    goal=goal,
                    context=context,
                    tools=tools,
                    max_steps=max_steps,
                    session_id=session_id,
                )

                step_summaries = [
                    {"step_id": step.step_id, "description": step.description[:100]}
                    for step in plan.steps
                ]

                return json.dumps({
                    "status": "success",
                    "plan_id": plan.plan_id,
                    "goal": plan.goal,
                    "step_count": len(plan.steps),
                    "steps": step_summaries,
                })

            elif operation == "get_status":
                # This would need access to executor status
                # For now, return a placeholder
                plan_id = tool_args.get("plan_id", "")

                return json.dumps({
                    "status": "success",
                    "plan_id": plan_id,
                    "message": "Status tracking requires executor integration",
                })

            else:
                raise ToolNotFoundError(
                    tool_name=f"planning.{operation}",
                    available_tools=[t for t in self.SUPPORTED_TOOLS if t.startswith("planning.")],
                )

        except ToolNotFoundError:
            raise
        except Exception as e:
            logger.error(f"Planning tool '{operation}' failed: {e}")
            raise ToolExecutionError(f"planning.{operation}", e)

    def list_available_tools(self) -> list[str]:
        """List available builtin tools based on configured dependencies.

        Returns:
            List of available tool names.
        """
        available = []

        if self._memory_manager is not None:
            available.extend([
                "memory.recall",
                "memory.retrieve",
                "memory.store",
                "memory.remember",
                "memory.list_categories",
                "memory.get_category",
            ])

        if self._planner is not None:
            available.extend([
                "planning.create_plan",
                "planning.get_status",
            ])

        return available


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "BuiltinToolExecutor",
    "BuiltinExecutorError",
    "ToolNotFoundError",
    "MissingDependencyError",
    "ToolExecutionError",
]
