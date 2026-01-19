"""Built-in tools for Sigil v2 framework.

This module contains built-in tool implementations:

Memory Tools:
    - RecallTool: Retrieve memories by query
    - RememberTool: Store facts to memory
    - ListCategoriesTool: List available categories
    - GetCategoryTool: Get category content

Planning Tools:
    - CreatePlanTool: Create plan from goal
    - GetPlanStatusTool: Get plan execution status
    - ExecutePlanStepTool: Execute a plan step
    - PausePlanTool: Pause plan execution
    - ResumePlanTool: Resume plan execution

Planned Tools:
    - FileReadTool: Read file contents
    - FileWriteTool: Write file contents
    - WebFetchTool: Fetch web page content
    - WebSearchTool: Search the web
    - ShellTool: Execute shell commands
    - PythonTool: Execute Python code
    - JSONTool: Parse and manipulate JSON
    - TextTool: Text processing utilities

Example:
    >>> from sigil.tools.builtin import create_memory_tools, create_planning_tools
    >>> from sigil.memory import MemoryManager
    >>> from sigil.planning import Planner, PlanExecutor
    >>>
    >>> # Memory tools
    >>> memory_manager = MemoryManager()
    >>> memory_tools = create_memory_tools(memory_manager, session_id="sess-123")
    >>>
    >>> # Planning tools
    >>> planner = Planner()
    >>> executor = PlanExecutor()
    >>> planning_tools = create_planning_tools(planner, executor)
"""

from sigil.tools.builtin.memory_tools import (
    # Input schemas
    RecallInput,
    RememberInput,
    GetCategoryInput,
    # Factory functions
    create_memory_tools,
    create_recall_tool,
    create_remember_tool,
    create_list_categories_tool,
    create_get_category_tool,
    create_recall_structured_tool,
    create_remember_structured_tool,
    # Availability flag
    LANGCHAIN_AVAILABLE,
)

from sigil.tools.builtin.planning_tools import (
    # Input schemas
    CreatePlanInput,
    GetPlanStatusInput,
    ExecutePlanStepInput,
    PlanControlInput,
    # Storage
    PlanStorage,
    # Factory functions
    create_planning_tools,
    create_create_plan_tool,
    create_get_plan_status_tool,
    create_execute_plan_step_tool,
    create_pause_plan_tool,
    create_resume_plan_tool,
)

__all__ = [
    # Memory - Input schemas
    "RecallInput",
    "RememberInput",
    "GetCategoryInput",
    # Memory - Factory functions
    "create_memory_tools",
    "create_recall_tool",
    "create_remember_tool",
    "create_list_categories_tool",
    "create_get_category_tool",
    "create_recall_structured_tool",
    "create_remember_structured_tool",
    # Planning - Input schemas
    "CreatePlanInput",
    "GetPlanStatusInput",
    "ExecutePlanStepInput",
    "PlanControlInput",
    # Planning - Storage
    "PlanStorage",
    # Planning - Factory functions
    "create_planning_tools",
    "create_create_plan_tool",
    "create_get_plan_status_tool",
    "create_execute_plan_step_tool",
    "create_pause_plan_tool",
    "create_resume_plan_tool",
    # Availability flag
    "LANGCHAIN_AVAILABLE",
]

# Conditionally export LangChain tool classes
if LANGCHAIN_AVAILABLE:
    from sigil.tools.builtin.memory_tools import (
        RecallTool,
        RememberTool,
        ListCategoriesTool,
        GetCategoryTool,
    )
    from sigil.tools.builtin.planning_tools import (
        CreatePlanTool,
        GetPlanStatusTool,
        ExecutePlanStepTool,
        PausePlanTool,
        ResumePlanTool,
    )

    __all__.extend([
        # Memory tools
        "RecallTool",
        "RememberTool",
        "ListCategoriesTool",
        "GetCategoryTool",
        # Planning tools
        "CreatePlanTool",
        "GetPlanStatusTool",
        "ExecutePlanStepTool",
        "PausePlanTool",
        "ResumePlanTool",
    ])
