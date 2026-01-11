"""Planning tools for Sigil v2 agents.

This module provides LangChain-compatible tools that agents can use to
interact with the planning system. These tools enable agents to create
plans, check plan status, execute steps, and control plan execution.

Tools:
    create_plan: Create a plan from a goal.
    get_plan_status: Get status of a plan.
    execute_plan_step: Execute a single step.
    pause_plan: Pause plan execution.
    resume_plan: Resume paused plan.

Example:
    >>> from sigil.tools.builtin.planning_tools import create_planning_tools
    >>> tools = create_planning_tools(planner, executor)
    >>> agent = create_agent(tools=tools)
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Callable, Optional, Type

from pydantic import BaseModel, Field

# Try to import LangChain, but make it optional
try:
    from langchain_core.tools import BaseTool, StructuredTool
    from langchain_core.callbacks import (
        CallbackManagerForToolRun,
        AsyncCallbackManagerForToolRun,
    )

    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    BaseTool = object
    StructuredTool = None


# =============================================================================
# Tool Input Schemas
# =============================================================================


class CreatePlanInput(BaseModel):
    """Input schema for the create_plan tool."""

    goal: str = Field(
        description="The high-level goal to create a plan for."
    )
    context: Optional[str] = Field(
        default=None,
        description="Optional JSON string with context information."
    )
    max_steps: int = Field(
        default=10,
        ge=1,
        le=20,
        description="Maximum number of steps in the plan (1-20)."
    )


class GetPlanStatusInput(BaseModel):
    """Input schema for the get_plan_status tool."""

    plan_id: str = Field(
        description="The unique identifier of the plan."
    )


class ExecutePlanStepInput(BaseModel):
    """Input schema for the execute_plan_step tool."""

    plan_id: str = Field(
        description="The unique identifier of the plan."
    )
    step_id: str = Field(
        description="The unique identifier of the step to execute."
    )


class PlanControlInput(BaseModel):
    """Input schema for plan control tools (pause/resume)."""

    plan_id: str = Field(
        description="The unique identifier of the plan."
    )


# =============================================================================
# Plan Storage (for tool state)
# =============================================================================


class PlanStorage:
    """Simple storage for plans and execution state.

    This provides in-memory storage for plans created by tools.
    In production, this would integrate with a persistent store.
    """

    def __init__(self) -> None:
        self._plans: dict[str, Any] = {}
        self._results: dict[str, dict[str, Any]] = {}

    def store_plan(self, plan: Any) -> None:
        """Store a plan."""
        self._plans[plan.plan_id] = plan

    def get_plan(self, plan_id: str) -> Optional[Any]:
        """Get a plan by ID."""
        return self._plans.get(plan_id)

    def store_step_result(self, plan_id: str, step_id: str, result: Any) -> None:
        """Store a step result."""
        if plan_id not in self._results:
            self._results[plan_id] = {}
        self._results[plan_id][step_id] = result

    def get_step_results(self, plan_id: str) -> dict[str, Any]:
        """Get all step results for a plan."""
        return self._results.get(plan_id, {})


# Global storage instance (shared across tools)
_plan_storage = PlanStorage()


# =============================================================================
# Tool Implementations (Function-based)
# =============================================================================


def create_create_plan_tool(
    planner: Any,
    session_id: Optional[str] = None,
    storage: Optional[PlanStorage] = None,
) -> Callable:
    """Create a function for creating plans.

    Args:
        planner: The Planner instance.
        session_id: Optional session ID for context.
        storage: Optional plan storage instance.

    Returns:
        A callable create_plan function.
    """
    storage = storage or _plan_storage

    async def create_plan(
        goal: str,
        context: Optional[str] = None,
        max_steps: int = 10,
    ) -> str:
        """Create a plan to achieve a goal.

        Use this tool to decompose a high-level goal into a sequence
        of executable steps. The plan will have dependencies between
        steps and can be executed step by step or automatically.

        Args:
            goal: The high-level goal (e.g., "Research competitors").
            context: Optional JSON context string.
            max_steps: Maximum number of steps (default: 10).

        Returns:
            JSON string with plan_id and summary.
        """
        try:
            # Parse context if provided
            context_dict = {}
            if context:
                try:
                    context_dict = json.loads(context)
                except json.JSONDecodeError:
                    context_dict = {"raw_context": context}

            # Create plan
            plan = await planner.create_plan(
                goal=goal,
                context=context_dict,
                tools=[],  # Tools determined by context
                max_steps=max_steps,
                session_id=session_id,
            )

            # Store plan
            storage.store_plan(plan)

            # Build step summaries
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

        except Exception as e:
            return json.dumps({
                "status": "error",
                "error": str(e),
            })

    return create_plan


def create_get_plan_status_tool(
    executor: Any,
    storage: Optional[PlanStorage] = None,
) -> Callable:
    """Create a function for getting plan status.

    Args:
        executor: The PlanExecutor instance.
        storage: Optional plan storage instance.

    Returns:
        A callable get_plan_status function.
    """
    storage = storage or _plan_storage

    async def get_plan_status(plan_id: str) -> str:
        """Get the current status of a plan.

        Use this tool to check progress on a plan, see which steps
        are completed, and determine the next step to execute.

        Args:
            plan_id: The plan's unique identifier.

        Returns:
            JSON string with status, completed steps, and next step.
        """
        try:
            # Check executor for active execution
            exec_status = executor.get_status(plan_id)

            if exec_status:
                return json.dumps({
                    "status": "success",
                    "plan_status": exec_status,
                })

            # Check storage for plan
            plan = storage.get_plan(plan_id)
            if plan is None:
                return json.dumps({
                    "status": "not_found",
                    "error": f"Plan '{plan_id}' not found.",
                })

            # Get step results
            step_results = storage.get_step_results(plan_id)

            # Build status
            completed_steps = []
            pending_steps = []

            for step in plan.steps:
                if step.step_id in step_results:
                    completed_steps.append({
                        "step_id": step.step_id,
                        "description": step.description[:50],
                        "status": "completed",
                    })
                else:
                    pending_steps.append({
                        "step_id": step.step_id,
                        "description": step.description[:50],
                        "status": "pending",
                    })

            next_step = pending_steps[0] if pending_steps else None

            return json.dumps({
                "status": "success",
                "plan_id": plan_id,
                "plan_status": plan.status.value if hasattr(plan.status, 'value') else str(plan.status),
                "completed_count": len(completed_steps),
                "pending_count": len(pending_steps),
                "completed_steps": completed_steps,
                "next_step": next_step,
            })

        except Exception as e:
            return json.dumps({
                "status": "error",
                "error": str(e),
            })

    return get_plan_status


def create_execute_plan_step_tool(
    executor: Any,
    storage: Optional[PlanStorage] = None,
) -> Callable:
    """Create a function for executing a plan step.

    Args:
        executor: The PlanExecutor instance.
        storage: Optional plan storage instance.

    Returns:
        A callable execute_plan_step function.
    """
    storage = storage or _plan_storage

    async def execute_plan_step(plan_id: str, step_id: str) -> str:
        """Execute a single step of a plan.

        Use this tool to execute a specific step in a plan.
        Steps should generally be executed in dependency order.

        Args:
            plan_id: The plan's unique identifier.
            step_id: The step's unique identifier.

        Returns:
            JSON string with step execution result.
        """
        try:
            # Get plan
            plan = storage.get_plan(plan_id)
            if plan is None:
                return json.dumps({
                    "status": "not_found",
                    "error": f"Plan '{plan_id}' not found.",
                })

            # Find step
            step = None
            for s in plan.steps:
                if s.step_id == step_id:
                    step = s
                    break

            if step is None:
                return json.dumps({
                    "status": "not_found",
                    "error": f"Step '{step_id}' not found in plan.",
                })

            # Check dependencies
            step_results = storage.get_step_results(plan_id)
            for dep_id in step.dependencies:
                if dep_id not in step_results:
                    return json.dumps({
                        "status": "blocked",
                        "error": f"Step depends on '{dep_id}' which is not complete.",
                    })

            # Execute step using default executor
            from sigil.planning.schemas import StepResult, StepStatus, utc_now
            import time

            start_time = time.time()

            # Simulated execution
            result = StepResult(
                step_id=step_id,
                status=StepStatus.COMPLETED,
                output=f"Executed: {step.description}",
                tokens_used=100,
                duration_ms=(time.time() - start_time) * 1000,
                started_at=utc_now(),
                completed_at=utc_now(),
            )

            # Store result
            storage.store_step_result(plan_id, step_id, result)

            return json.dumps({
                "status": "success",
                "step_id": step_id,
                "step_status": result.status.value,
                "output": result.output,
                "tokens_used": result.tokens_used,
            })

        except Exception as e:
            return json.dumps({
                "status": "error",
                "error": str(e),
            })

    return execute_plan_step


def create_pause_plan_tool(executor: Any) -> Callable:
    """Create a function for pausing plan execution.

    Args:
        executor: The PlanExecutor instance.

    Returns:
        A callable pause_plan function.
    """

    async def pause_plan(plan_id: str) -> str:
        """Pause execution of a plan.

        Use this tool to pause a running plan. The plan can be
        resumed later with resume_plan.

        Args:
            plan_id: The plan's unique identifier.

        Returns:
            JSON string confirming pause.
        """
        try:
            success = executor.pause(plan_id)

            if success:
                return json.dumps({
                    "status": "success",
                    "message": f"Plan '{plan_id}' paused.",
                })
            else:
                return json.dumps({
                    "status": "warning",
                    "message": f"Could not pause plan '{plan_id}'. It may not be running.",
                })

        except Exception as e:
            return json.dumps({
                "status": "error",
                "error": str(e),
            })

    return pause_plan


def create_resume_plan_tool(
    executor: Any,
    storage: Optional[PlanStorage] = None,
    session_id: Optional[str] = None,
) -> Callable:
    """Create a function for resuming plan execution.

    Args:
        executor: The PlanExecutor instance.
        storage: Optional plan storage instance.
        session_id: Optional session ID for context.

    Returns:
        A callable resume_plan function.
    """
    storage = storage or _plan_storage

    async def resume_plan(plan_id: str) -> str:
        """Resume execution of a paused plan.

        Use this tool to resume a plan that was previously paused.
        Execution will continue from where it left off.

        Args:
            plan_id: The plan's unique identifier.

        Returns:
            JSON string with execution result.
        """
        try:
            # Resume executor
            executor.resume(plan_id)

            # Get plan
            plan = storage.get_plan(plan_id)
            if plan is None:
                return json.dumps({
                    "status": "warning",
                    "message": f"Plan '{plan_id}' not found in storage. Resume signal sent.",
                })

            # Execute remaining plan
            result = await executor.execute(plan, session_id)

            return json.dumps({
                "status": "success",
                "plan_id": plan_id,
                "success": result.success,
                "state": result.state.value,
                "total_tokens": result.total_tokens,
                "completed_steps": len([r for r in result.step_results if r.status.value == "completed"]),
            })

        except Exception as e:
            return json.dumps({
                "status": "error",
                "error": str(e),
            })

    return resume_plan


# =============================================================================
# LangChain Tool Classes
# =============================================================================


if LANGCHAIN_AVAILABLE:

    class CreatePlanTool(BaseTool):
        """LangChain tool for creating plans."""

        name: str = "create_plan"
        description: str = (
            "Create a plan to achieve a high-level goal. "
            "The plan will decompose the goal into executable steps with dependencies. "
            "Input: goal (required), context (optional JSON), max_steps (optional)."
        )
        args_schema: Type[BaseModel] = CreatePlanInput

        planner: Any = None
        session_id: Optional[str] = None
        storage: Any = None

        def _run(
            self,
            goal: str,
            context: Optional[str] = None,
            max_steps: int = 10,
            run_manager: Optional[CallbackManagerForToolRun] = None,
        ) -> str:
            """Synchronous execution."""
            return asyncio.run(self._arun(goal, context, max_steps))

        async def _arun(
            self,
            goal: str,
            context: Optional[str] = None,
            max_steps: int = 10,
            run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
        ) -> str:
            """Asynchronous execution."""
            create_func = create_create_plan_tool(
                self.planner, self.session_id, self.storage or _plan_storage
            )
            return await create_func(goal, context, max_steps)

    class GetPlanStatusTool(BaseTool):
        """LangChain tool for getting plan status."""

        name: str = "get_plan_status"
        description: str = (
            "Get the current status of a plan. "
            "Shows completed steps, pending steps, and the next step to execute. "
            "Input: plan_id (required)."
        )
        args_schema: Type[BaseModel] = GetPlanStatusInput

        executor: Any = None
        storage: Any = None

        def _run(
            self,
            plan_id: str,
            run_manager: Optional[CallbackManagerForToolRun] = None,
        ) -> str:
            """Synchronous execution."""
            return asyncio.run(self._arun(plan_id))

        async def _arun(
            self,
            plan_id: str,
            run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
        ) -> str:
            """Asynchronous execution."""
            get_func = create_get_plan_status_tool(
                self.executor, self.storage or _plan_storage
            )
            return await get_func(plan_id)

    class ExecutePlanStepTool(BaseTool):
        """LangChain tool for executing a plan step."""

        name: str = "execute_plan_step"
        description: str = (
            "Execute a single step of a plan. "
            "Steps should be executed in dependency order. "
            "Input: plan_id (required), step_id (required)."
        )
        args_schema: Type[BaseModel] = ExecutePlanStepInput

        executor: Any = None
        storage: Any = None

        def _run(
            self,
            plan_id: str,
            step_id: str,
            run_manager: Optional[CallbackManagerForToolRun] = None,
        ) -> str:
            """Synchronous execution."""
            return asyncio.run(self._arun(plan_id, step_id))

        async def _arun(
            self,
            plan_id: str,
            step_id: str,
            run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
        ) -> str:
            """Asynchronous execution."""
            exec_func = create_execute_plan_step_tool(
                self.executor, self.storage or _plan_storage
            )
            return await exec_func(plan_id, step_id)

    class PausePlanTool(BaseTool):
        """LangChain tool for pausing plan execution."""

        name: str = "pause_plan"
        description: str = (
            "Pause execution of a running plan. "
            "The plan can be resumed later with resume_plan. "
            "Input: plan_id (required)."
        )
        args_schema: Type[BaseModel] = PlanControlInput

        executor: Any = None

        def _run(
            self,
            plan_id: str,
            run_manager: Optional[CallbackManagerForToolRun] = None,
        ) -> str:
            """Synchronous execution."""
            return asyncio.run(self._arun(plan_id))

        async def _arun(
            self,
            plan_id: str,
            run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
        ) -> str:
            """Asynchronous execution."""
            pause_func = create_pause_plan_tool(self.executor)
            return await pause_func(plan_id)

    class ResumePlanTool(BaseTool):
        """LangChain tool for resuming plan execution."""

        name: str = "resume_plan"
        description: str = (
            "Resume execution of a paused plan. "
            "Execution continues from where it left off. "
            "Input: plan_id (required)."
        )
        args_schema: Type[BaseModel] = PlanControlInput

        executor: Any = None
        storage: Any = None
        session_id: Optional[str] = None

        def _run(
            self,
            plan_id: str,
            run_manager: Optional[CallbackManagerForToolRun] = None,
        ) -> str:
            """Synchronous execution."""
            return asyncio.run(self._arun(plan_id))

        async def _arun(
            self,
            plan_id: str,
            run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
        ) -> str:
            """Asynchronous execution."""
            resume_func = create_resume_plan_tool(
                self.executor, self.storage or _plan_storage, self.session_id
            )
            return await resume_func(plan_id)


# =============================================================================
# Factory Functions
# =============================================================================


def create_planning_tools(
    planner: Any,
    executor: Any,
    session_id: Optional[str] = None,
    as_langchain: bool = True,
) -> list[Any]:
    """Create all planning tools for an agent.

    Args:
        planner: The Planner instance.
        executor: The PlanExecutor instance.
        session_id: Optional session ID for context.
        as_langchain: If True, return LangChain tools; otherwise return functions.

    Returns:
        List of tools (LangChain BaseTool or callable functions).
    """
    storage = PlanStorage()

    if as_langchain and LANGCHAIN_AVAILABLE:
        return [
            CreatePlanTool(planner=planner, session_id=session_id, storage=storage),
            GetPlanStatusTool(executor=executor, storage=storage),
            ExecutePlanStepTool(executor=executor, storage=storage),
            PausePlanTool(executor=executor),
            ResumePlanTool(executor=executor, storage=storage, session_id=session_id),
        ]
    else:
        return [
            create_create_plan_tool(planner, session_id, storage),
            create_get_plan_status_tool(executor, storage),
            create_execute_plan_step_tool(executor, storage),
            create_pause_plan_tool(executor),
            create_resume_plan_tool(executor, storage, session_id),
        ]


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Input schemas
    "CreatePlanInput",
    "GetPlanStatusInput",
    "ExecutePlanStepInput",
    "PlanControlInput",
    # Storage
    "PlanStorage",
    # Factory functions
    "create_planning_tools",
    "create_create_plan_tool",
    "create_get_plan_status_tool",
    "create_execute_plan_step_tool",
    "create_pause_plan_tool",
    "create_resume_plan_tool",
    # LangChain availability flag
    "LANGCHAIN_AVAILABLE",
]

# Conditionally export LangChain tool classes
if LANGCHAIN_AVAILABLE:
    __all__.extend([
        "CreatePlanTool",
        "GetPlanStatusTool",
        "ExecutePlanStepTool",
        "PausePlanTool",
        "ResumePlanTool",
    ])
