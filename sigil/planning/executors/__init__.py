"""Tool executors for Sigil v2 plan execution.

This package contains executors that handle direct tool calls without
LLM reasoning, enabling efficient plan execution with minimal token usage.

Executors:
    - MCPToolExecutor: Executes MCP tools (websearch, voice, calendar, etc.)
    - BuiltinToolExecutor: Executes builtin tools (memory, planning)
    - TavilyExecutor: Direct Tavily API executor (used by MCPToolExecutor for websearch)

Note:
    MCPToolExecutor automatically routes websearch tools to TavilyExecutor
    for fast, reliable execution. This bypasses langchain-mcp-adapters which
    has connection management issues causing 30+ second timeouts.

Usage:
    >>> from sigil.planning.executors import MCPToolExecutor, BuiltinToolExecutor
    >>>
    >>> mcp_executor = MCPToolExecutor()
    >>> result = await mcp_executor.execute("websearch.search", {"query": "AI news"})
    >>>
    >>> builtin_executor = BuiltinToolExecutor(memory_manager=mm)
    >>> result = await builtin_executor.execute("memory.recall", {"query": "preferences"})
"""

from sigil.planning.executors.mcp_executor import MCPToolExecutor
from sigil.planning.executors.builtin_executor import BuiltinToolExecutor
from sigil.planning.executors.tavily_executor import TavilyExecutor


__all__ = [
    "MCPToolExecutor",
    "BuiltinToolExecutor",
    "TavilyExecutor",
]
