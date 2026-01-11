"""Built-in tools for Sigil v2 framework.

This module contains built-in tool implementations:

Memory Tools (implemented):
    - RecallTool: Retrieve memories by query
    - RememberTool: Store facts to memory
    - ListCategoriesTool: List available categories
    - GetCategoryTool: Get category content

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
    >>> from sigil.tools.builtin import create_memory_tools
    >>> from sigil.memory import MemoryManager
    >>>
    >>> manager = MemoryManager()
    >>> tools = create_memory_tools(manager, session_id="sess-123")

TODO: Implement FileReadTool with path validation
TODO: Implement FileWriteTool with sandboxing
TODO: Implement WebFetchTool with rate limiting
TODO: Implement WebSearchTool with provider abstraction
TODO: Implement ShellTool with security constraints
TODO: Implement PythonTool with isolation
TODO: Implement JSONTool
TODO: Implement TextTool
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

__all__ = [
    # Input schemas
    "RecallInput",
    "RememberInput",
    "GetCategoryInput",
    # Factory functions
    "create_memory_tools",
    "create_recall_tool",
    "create_remember_tool",
    "create_list_categories_tool",
    "create_get_category_tool",
    "create_recall_structured_tool",
    "create_remember_structured_tool",
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

    __all__.extend([
        "RecallTool",
        "RememberTool",
        "ListCategoriesTool",
        "GetCategoryTool",
    ])
