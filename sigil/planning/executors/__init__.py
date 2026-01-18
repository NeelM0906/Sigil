"""Tool executors for Sigil v2 plan execution.

This package contains executors that handle direct tool calls without
LLM reasoning, enabling efficient plan execution with minimal token usage.

Executors:
    - TavilyExecutor: Executes web search tools via Tavily API
    - BuiltinToolExecutor: Executes builtin tools (memory, planning)

Usage:
    >>> from sigil.planning.executors import TavilyExecutor, BuiltinToolExecutor
    >>>
    >>> tavily_executor = TavilyExecutor()
    >>> result = await tavily_executor.execute("websearch.search", {"query": "AI news"})
    >>>
    >>> builtin_executor = BuiltinToolExecutor(memory_manager=mm)
    >>> result = await builtin_executor.execute("memory.recall", {"query": "preferences"})
"""

from sigil.planning.executors.builtin_executor import BuiltinToolExecutor
from sigil.planning.executors.tavily_executor import TavilyExecutor


__all__ = [
    "BuiltinToolExecutor",
    "TavilyExecutor",
]
