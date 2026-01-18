"""Tools module for Sigil v2 framework.

This module implements tool integration and execution:
- Web search via Tavily API
- Built-in memory and planning tools
- Tool registry and discovery
- Tool execution sandboxing
- Tool schema conversion utilities

Key Components:
    - TavilyExecutor: Executes web search via Tavily
    - BuiltinToolExecutor: Executes memory and planning tools
    - ToolRegistry: Registry of available tools
    - convert_to_claude_tool_schema: Convert Sigil tool defs to Claude format
    - get_all_tool_schemas: Get all tool schemas in Claude format
"""
from sigil.tools.schemas import (
    convert_to_claude_tool_schema,
    get_all_tool_schemas,
    claude_name_to_sigil_name,
)

__all__ = [
    "convert_to_claude_tool_schema",
    "get_all_tool_schemas",
    "claude_name_to_sigil_name",
]
