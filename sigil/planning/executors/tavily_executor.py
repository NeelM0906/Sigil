"""Direct Tavily API Executor for Sigil v2 plan execution.

This module provides direct access to Tavily's search API without going through
the MCP protocol. This is significantly faster and more reliable than using
langchain-mcp-adapters with mcp-remote, which has connection management issues.

The TavilyExecutor:
- Uses the native tavily-python SDK for direct API calls
- Supports search, extract, and other Tavily operations
- Maintains a persistent async HTTP client for efficiency
- Returns results in a consistent format

Why direct API instead of MCP?
- langchain-mcp-adapters spawns a new subprocess for EACH get_tools() call
- mcp-remote connections are not cached, causing repeated 5-10 second delays
- Direct API calls complete in <1 second vs 30+ second timeouts with MCP

Example:
    >>> executor = TavilyExecutor()
    >>> result = await executor.execute(
    ...     tool_name="tavily_search",
    ...     tool_args={"query": "AI news 2026", "max_results": 5}
    ... )
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Optional

from sigil.core.exceptions import SigilError


# =============================================================================
# Module Logger
# =============================================================================

logger = logging.getLogger(__name__)


# =============================================================================
# Exceptions
# =============================================================================


class TavilyExecutorError(SigilError):
    """Base exception for Tavily executor errors."""

    def __init__(
        self,
        message: str,
        tool_name: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(message, code="TAVILY_EXECUTOR_ERROR", **kwargs)
        self.tool_name = tool_name


class TavilyNotConfiguredError(TavilyExecutorError):
    """Raised when Tavily API key is not configured."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(
            "Tavily API key not configured. Set TAVILY_API_KEY environment variable.",
            **kwargs,
        )


class TavilyToolNotFoundError(TavilyExecutorError):
    """Raised when requested Tavily tool is not found."""

    def __init__(
        self,
        tool_name: str,
        available_tools: list[str],
        **kwargs: Any,
    ) -> None:
        super().__init__(
            f"Tavily tool '{tool_name}' not found. "
            f"Available tools: {', '.join(available_tools)}",
            tool_name=tool_name,
            **kwargs,
        )
        self.available_tools = available_tools


# =============================================================================
# Tavily Executor
# =============================================================================


class TavilyExecutor:
    """Executes Tavily API calls directly without MCP overhead.

    This executor provides fast, reliable access to Tavily's search capabilities
    by using the native Python SDK instead of going through mcp-remote.

    Supported tools:
    - tavily_search / search: Web search with real-time results
    - tavily_extract / extract: Extract content from URLs
    - tavily_qna / qna: Question-answering with search

    Attributes:
        api_key: Tavily API key (from env or explicit)

    Example:
        >>> executor = TavilyExecutor()
        >>> result = await executor.execute(
        ...     "tavily_search",
        ...     {"query": "latest news", "max_results": 5}
        ... )
    """

    # Mapping of tool names to handler methods
    SUPPORTED_TOOLS = {
        "tavily_search": "search",
        "search": "search",
        "tavily_extract": "extract",
        "extract": "extract",
        "tavily_qna": "qna",
        "qna": "qna",
    }

    def __init__(self, api_key: Optional[str] = None) -> None:
        """Initialize the Tavily executor.

        Args:
            api_key: Tavily API key. If not provided, reads from TAVILY_API_KEY env var.
        """
        self._api_key = api_key or os.getenv("TAVILY_API_KEY")
        self._client: Optional[Any] = None

    def _ensure_client(self) -> Any:
        """Ensure Tavily client is initialized.

        Returns:
            AsyncTavilyClient instance.

        Raises:
            TavilyNotConfiguredError: If API key is not set.
            TavilyExecutorError: If tavily-python is not installed.
        """
        if self._client is not None:
            return self._client

        if not self._api_key:
            raise TavilyNotConfiguredError()

        try:
            from tavily import AsyncTavilyClient
            self._client = AsyncTavilyClient(api_key=self._api_key)
            logger.debug("Tavily client initialized")
            return self._client
        except ImportError as e:
            raise TavilyExecutorError(
                f"tavily-python package not installed: {e}. "
                f"Install with: pip install tavily-python"
            )

    @staticmethod
    def is_tavily_tool(tool_name: str) -> bool:
        """Check if a tool name is a Tavily tool.

        Args:
            tool_name: Tool name to check (e.g., "websearch.search", "tavily_search")

        Returns:
            True if this is a Tavily tool.
        """
        # Handle "websearch.X" format
        if tool_name.startswith("websearch."):
            operation = tool_name.split(".", 1)[1]
            return operation in TavilyExecutor.SUPPORTED_TOOLS

        # Handle direct tool names
        return tool_name in TavilyExecutor.SUPPORTED_TOOLS

    @staticmethod
    def normalize_tool_name(tool_name: str) -> str:
        """Normalize tool name to standard format.

        Args:
            tool_name: Tool name in various formats.

        Returns:
            Normalized tool operation name.
        """
        # Handle "websearch.X" format
        if tool_name.startswith("websearch."):
            return tool_name.split(".", 1)[1]
        return tool_name

    async def execute(
        self,
        tool_name: str,
        tool_args: dict[str, Any],
    ) -> str:
        """Execute a Tavily tool with given arguments.

        Args:
            tool_name: Tool name (e.g., "websearch.search", "tavily_search", "search")
            tool_args: Arguments for the tool.

        Returns:
            Tool result as JSON string.

        Raises:
            TavilyNotConfiguredError: If API key is not set.
            TavilyToolNotFoundError: If tool is not supported.
            TavilyExecutorError: If execution fails.
        """
        normalized_name = self.normalize_tool_name(tool_name)
        handler_name = self.SUPPORTED_TOOLS.get(normalized_name)

        if handler_name is None:
            raise TavilyToolNotFoundError(
                tool_name,
                list(self.SUPPORTED_TOOLS.keys()),
            )

        logger.info(f"[Tavily] Executing {tool_name} -> {handler_name}")
        logger.debug(f"[Tavily] Args: {tool_args}")

        # Get handler method
        handler = getattr(self, f"_execute_{handler_name}", None)
        if handler is None:
            raise TavilyExecutorError(
                f"Handler for '{handler_name}' not implemented",
                tool_name=tool_name,
            )

        try:
            result = await handler(tool_args)
            logger.info(f"[Tavily] {tool_name} completed successfully")
            return result
        except TavilyExecutorError:
            raise
        except Exception as e:
            logger.error(f"[Tavily] {tool_name} failed: {e}")
            raise TavilyExecutorError(
                f"Tavily API call failed: {e}",
                tool_name=tool_name,
            )

    async def _execute_search(self, args: dict[str, Any]) -> str:
        """Execute Tavily search.

        Args:
            args: Search arguments including:
                - query (required): Search query
                - max_results (optional): Max results (default 5)
                - search_depth (optional): "basic" or "advanced"
                - include_answer (optional): Include AI answer
                - include_raw_content (optional): Include raw HTML

        Returns:
            JSON string with search results.
        """
        client = self._ensure_client()

        query = args.get("query")
        if not query:
            raise TavilyExecutorError("'query' argument is required for search")

        # Build search kwargs
        search_kwargs = {
            "query": query,
            "max_results": args.get("max_results", 5),
        }

        # Optional parameters
        if "search_depth" in args:
            search_kwargs["search_depth"] = args["search_depth"]
        if "include_answer" in args:
            search_kwargs["include_answer"] = args["include_answer"]
        if "include_raw_content" in args:
            search_kwargs["include_raw_content"] = args["include_raw_content"]
        if "include_domains" in args:
            search_kwargs["include_domains"] = args["include_domains"]
        if "exclude_domains" in args:
            search_kwargs["exclude_domains"] = args["exclude_domains"]

        logger.debug(f"[Tavily] Search kwargs: {search_kwargs}")

        result = await client.search(**search_kwargs)
        return json.dumps(result, indent=2)

    async def _execute_extract(self, args: dict[str, Any]) -> str:
        """Execute Tavily content extraction.

        Args:
            args: Extract arguments including:
                - urls (required): List of URLs to extract from

        Returns:
            JSON string with extracted content.
        """
        client = self._ensure_client()

        urls = args.get("urls")
        if not urls:
            raise TavilyExecutorError("'urls' argument is required for extract")

        if isinstance(urls, str):
            urls = [urls]

        result = await client.extract(urls=urls)
        return json.dumps(result, indent=2)

    async def _execute_qna(self, args: dict[str, Any]) -> str:
        """Execute Tavily Q&A search.

        Args:
            args: QnA arguments including:
                - query (required): Question to answer

        Returns:
            JSON string with answer.
        """
        client = self._ensure_client()

        query = args.get("query")
        if not query:
            raise TavilyExecutorError("'query' argument is required for qna")

        result = await client.qna_search(query=query)
        return json.dumps({"answer": result}, indent=2)

    def list_tools(self) -> list[str]:
        """List available Tavily tools.

        Returns:
            List of tool names.
        """
        return list(set(self.SUPPORTED_TOOLS.values()))

    def is_configured(self) -> bool:
        """Check if Tavily is properly configured.

        Returns:
            True if API key is available.
        """
        return bool(self._api_key or os.getenv("TAVILY_API_KEY"))


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "TavilyExecutor",
    "TavilyExecutorError",
    "TavilyNotConfiguredError",
    "TavilyToolNotFoundError",
]
