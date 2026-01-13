"""Planning module for Sigil v2 framework.

This module implements Phase 5 task decomposition and execution planning:
- Goal decomposition into executable steps
- Dependency graph construction (DAG validation)
- Parallel execution coordination
- Plan monitoring and adaptation
- Pause/resume/abort capabilities

Key Components:
    - Planner: Creates execution plans from goals
    - PlanExecutor: Executes plans with monitoring
    - Schemas: Plan, PlanStep, PlanResult, and related types

Example:
    >>> from sigil.planning import Planner, PlanExecutor
    >>>
    >>> planner = Planner()
    >>> plan = await planner.create_plan(
    ...     goal="Research competitors and create summary",
    ...     context={"industry": "SaaS"},
    ...     tools=["websearch.search"],
    ...     max_steps=10,
    ... )
    >>>
    >>> executor = PlanExecutor()
    >>> result = await executor.execute(plan)
    >>> print(f"Success: {result.success}, Tokens: {result.total_tokens}")
"""

from sigil.planning.schemas import (
    # Constants
    DEFAULT_MAX_STEPS,
    DEFAULT_STEP_TIMEOUT,
    DEFAULT_MAX_CONCURRENT,
    DEFAULT_MAX_RETRIES,
    # Utilities
    generate_uuid,
    utc_now,
    # Enumerations
    StepType,
    StepStatus,
    ExecutionState,
    # Configuration
    PlanStepConfig,
    PlanMetadata,
    PlanConstraints,
    # Results
    StepResult,
    ExecutionCheckpoint,
    PlanResult,
)

from sigil.planning.planner import (
    Planner,
    PlanningError,
    DAGValidationError,
    create_plan_created_event,
)

from sigil.planning.executor import (
    PlanExecutor,
    ExecutionContext,
    create_step_started_event,
    create_step_completed_event,
    create_plan_completed_event,
)

from sigil.planning.tool_executor import (
    ToolStepExecutor,
    create_tool_step_executor,
    StepExecutionError,
    NoToolSpecifiedError,
    UnsupportedStepTypeError,
)

from sigil.planning.executors import (
    MCPToolExecutor,
    BuiltinToolExecutor,
)


__all__ = [
    # Main classes
    "Planner",
    "PlanExecutor",
    # Exceptions
    "PlanningError",
    "DAGValidationError",
    # Schemas - Constants
    "DEFAULT_MAX_STEPS",
    "DEFAULT_STEP_TIMEOUT",
    "DEFAULT_MAX_CONCURRENT",
    "DEFAULT_MAX_RETRIES",
    # Schemas - Utilities
    "generate_uuid",
    "utc_now",
    # Schemas - Enumerations
    "StepType",
    "StepStatus",
    "ExecutionState",
    # Schemas - Configuration
    "PlanStepConfig",
    "PlanMetadata",
    "PlanConstraints",
    # Schemas - Results
    "StepResult",
    "ExecutionCheckpoint",
    "PlanResult",
    # Execution context
    "ExecutionContext",
    # Event creators
    "create_plan_created_event",
    "create_step_started_event",
    "create_step_completed_event",
    "create_plan_completed_event",
    # Tool executors
    "ToolStepExecutor",
    "create_tool_step_executor",
    "MCPToolExecutor",
    "BuiltinToolExecutor",
    # Tool executor exceptions
    "StepExecutionError",
    "NoToolSpecifiedError",
    "UnsupportedStepTypeError",
]
