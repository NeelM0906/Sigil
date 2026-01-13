"""MCP Tool Executor for Sigil v2 plan execution.

This module implements direct execution of MCP tools without LLM reasoning,
enabling efficient tool calls during plan execution with zero token usage.

The MCPToolExecutor:
- Parses tool names to extract categories (e.g., "websearch.search" -> "websearch")
- Routes websearch tools to TavilyExecutor for fast, reliable execution
- Manages MCP client connections with caching for other tool categories
- Executes tools directly via LangChain's BaseTool interface
- Returns raw tool output as strings
- Handles connection failures gracefully with detailed error messages

Note on websearch/Tavily:
    The langchain-mcp-adapters library spawns a new subprocess for EACH get_tools()
    call when using mcp-remote. This causes 30+ second delays and timeouts.
    For websearch, we bypass MCP entirely and use Tavily's native Python SDK,
    which provides sub-second response times.

Example:
    >>> executor = MCPToolExecutor()
    >>> result = await executor.execute(
    ...     tool_name="websearch.search",
    ...     tool_args={"query": "AI news 2026", "max_results": 5}
    ... )
    >>> print(result)  # Raw search results
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, Optional, TYPE_CHECKING

from sigil.core.exceptions import SigilError
from sigil.planning.executors.tavily_executor import (
    TavilyExecutor,
    TavilyExecutorError,
)

if TYPE_CHECKING:
    from langchain_core.tools import BaseTool
    from langchain_mcp_adapters.client import MultiServerMCPClient


# =============================================================================
# Module Logger
# =============================================================================

logger = logging.getLogger(__name__)


# =============================================================================
# Exceptions
# =============================================================================


class MCPExecutorError(SigilError):
    """Base exception for MCP executor errors."""

    def __init__(
        self,
        message: str,
        tool_name: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(message, code="MCP_EXECUTOR_ERROR", **kwargs)
        self.tool_name = tool_name


class ToolNotFoundError(MCPExecutorError):
    """Raised when requested tool is not found in MCP server."""

    def __init__(
        self,
        tool_name: str,
        available_tools: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> None:
        available = ", ".join(available_tools[:10]) if available_tools else "none"
        super().__init__(
            f"Tool '{tool_name}' not found. Available tools: {available}",
            tool_name=tool_name,
            **kwargs,
        )
        self.available_tools = available_tools or []


class MCPConnectionError(MCPExecutorError):
    """Raised when connection to MCP server fails."""

    def __init__(
        self,
        category: str,
        original_error: Exception,
        error_type: str = "connection",
        **kwargs: Any,
    ) -> None:
        super().__init__(
            f"Failed to connect to MCP server for category '{category}': {original_error}",
            **kwargs,
        )
        self.category = category
        self.original_error = original_error
        self.error_type = error_type  # "timeout", "connection_refused", "not_found", "credentials"


class MCPTimeoutError(MCPExecutorError):
    """Raised when MCP server connection times out."""

    def __init__(
        self,
        category: str,
        timeout: float,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            f"Connection to MCP server '{category}' timed out after {timeout}s. "
            f"The server may not be running or may be unresponsive.",
            **kwargs,
        )
        self.category = category
        self.timeout = timeout


class MCPNotAvailableError(MCPExecutorError):
    """Raised when MCP server is not available (credentials missing or server not configured)."""

    def __init__(
        self,
        category: str,
        reason: str,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            f"MCP server '{category}' is not available: {reason}",
            **kwargs,
        )
        self.category = category
        self.reason = reason


# Alias for backward compatibility
ConnectionError = MCPConnectionError


class ToolExecutionError(MCPExecutorError):
    """Raised when tool execution fails."""

    def __init__(
        self,
        tool_name: str,
        original_error: Exception,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            f"Execution of tool '{tool_name}' failed: {original_error}",
            tool_name=tool_name,
            **kwargs,
        )
        self.original_error = original_error


# =============================================================================
# MCP Tool Executor
# =============================================================================


class MCPToolExecutor:
    """Executes MCP tools directly without LLM reasoning.

    This executor handles tools from MCP servers (websearch, voice, calendar,
    communication, crm) by:
    1. Parsing the tool name to extract the category
    2. For websearch: Routes to TavilyExecutor for fast, reliable execution
    3. For other categories: Connects to MCP server with connection caching
    4. Finding the specific tool in the server's tool list
    5. Executing the tool with provided arguments
    6. Returning the raw result

    Note:
        Websearch tools are routed to TavilyExecutor (direct API) instead of
        MCP because langchain-mcp-adapters has severe connection management
        issues that cause 30+ second timeouts.

    Attributes:
        connection_timeout: Timeout for MCP server connections in seconds.

    Example:
        >>> executor = MCPToolExecutor()
        >>> result = await executor.execute(
        ...     "websearch.search",
        ...     {"query": "latest AI news"}
        ... )
    """

    def __init__(
        self,
        connection_timeout: float = 30.0,
        tool_execution_timeout: float = 60.0,
    ) -> None:
        """Initialize the MCP tool executor.

        Args:
            connection_timeout: Timeout for MCP server connections in seconds.
            tool_execution_timeout: Timeout for individual tool executions in seconds.
        """
        self._connection_timeout = connection_timeout
        self._tool_execution_timeout = tool_execution_timeout
        self._client_cache: dict[str, "MultiServerMCPClient"] = {}
        self._tool_cache: dict[str, dict[str, "BaseTool"]] = {}
        self._lock = asyncio.Lock()
        # Use TavilyExecutor for websearch tools (bypasses problematic MCP)
        self._tavily_executor = TavilyExecutor()

    @staticmethod
    def parse_tool_name(tool_name: str) -> tuple[str, str]:
        """Parse a tool name into category and operation.

        Tool names follow the format "category.operation" where:
        - category: The MCP server category (e.g., "websearch", "calendar")
        - operation: The specific tool operation (e.g., "search", "create_event")

        Args:
            tool_name: Full tool name (e.g., "websearch.search").

        Returns:
            Tuple of (category, operation).

        Raises:
            ValueError: If tool name format is invalid.

        Example:
            >>> MCPToolExecutor.parse_tool_name("websearch.search")
            ('websearch', 'search')
            >>> MCPToolExecutor.parse_tool_name("calendar.create_event")
            ('calendar', 'create_event')
        """
        parts = tool_name.split(".", 1)
        if len(parts) != 2:
            raise ValueError(
                f"Invalid tool name format: '{tool_name}'. "
                f"Expected format: 'category.operation'"
            )
        return parts[0], parts[1]

    async def _get_client(self, category: str) -> "MultiServerMCPClient":
        """Get or create an MCP client for a category.

        Uses connection caching to avoid reconnecting per tool call.
        Implements timeout protection and detailed error classification.

        Args:
            category: MCP server category (e.g., "websearch").

        Returns:
            Connected MultiServerMCPClient.

        Raises:
            MCPTimeoutError: If connection times out.
            MCPNotAvailableError: If credentials are missing or server not configured.
            MCPConnectionError: If connection fails for other reasons.
        """
        # Check cache first (outside lock for fast path)
        if category in self._client_cache:
            return self._client_cache[category]

        async with self._lock:
            # Double-check after acquiring lock
            if category in self._client_cache:
                return self._client_cache[category]

            logger.info(
                f"[MCP] Attempting to connect to '{category}' server "
                f"(timeout: {self._connection_timeout}s)"
            )

            try:
                # Import here to avoid import errors if not installed
                from src.mcp_integration import (
                    connect_mcp_servers,
                    REQUIRED_ENV_VARS,
                    _ensure_mcp_servers,
                )

                # Pre-flight check: verify category is known
                servers = _ensure_mcp_servers()
                if category not in servers:
                    valid_categories = ", ".join(servers.keys())
                    logger.error(
                        f"[MCP] Unknown category '{category}'. "
                        f"Valid categories: {valid_categories}"
                    )
                    raise MCPNotAvailableError(
                        category,
                        f"Unknown MCP category. Valid options: {valid_categories}",
                    )

                # Pre-flight check: verify credentials are present
                required_vars = REQUIRED_ENV_VARS.get(category, [])
                missing_vars = [var for var in required_vars if not os.getenv(var)]
                if missing_vars:
                    logger.error(
                        f"[MCP] Missing credentials for '{category}': {missing_vars}. "
                        f"Set these environment variables to use this tool."
                    )
                    raise MCPNotAvailableError(
                        category,
                        f"Missing environment variables: {', '.join(missing_vars)}",
                    )

                logger.debug(f"[MCP] Credentials verified for '{category}'")

                # Wrap the entire connection process in a timeout
                # This protects against hangs in MultiServerMCPClient initialization
                try:
                    result = await asyncio.wait_for(
                        connect_mcp_servers(
                            [category],
                            timeout=self._connection_timeout,
                        ),
                        timeout=self._connection_timeout + 5,  # Extra buffer for wrapper
                    )
                except asyncio.TimeoutError:
                    logger.error(
                        f"[MCP] Connection to '{category}' timed out after "
                        f"{self._connection_timeout}s. The MCP server process may be "
                        f"hanging or the server is unresponsive."
                    )
                    raise MCPTimeoutError(category, self._connection_timeout)

                if not result.client:
                    error_msg = result.errors.get(category, "Unknown connection error")
                    logger.error(
                        f"[MCP] Failed to connect to '{category}': {error_msg}"
                    )

                    # Classify the error type
                    error_lower = error_msg.lower()
                    if "timed out" in error_lower:
                        raise MCPTimeoutError(category, self._connection_timeout)
                    elif "credential" in error_lower or "missing" in error_lower:
                        raise MCPNotAvailableError(category, error_msg)
                    elif "refused" in error_lower or "econnrefused" in error_lower:
                        raise MCPConnectionError(
                            category,
                            Exception(error_msg),
                            error_type="connection_refused",
                        )
                    else:
                        raise MCPConnectionError(
                            category,
                            Exception(error_msg),
                            error_type="connection",
                        )

                # Cache the client
                self._client_cache[category] = result.client
                logger.info(
                    f"[MCP] Successfully connected to '{category}' server. "
                    f"Tools available: {result.connected_tools}"
                )

                return result.client

            except ImportError as e:
                logger.error(
                    f"[MCP] MCP integration module not available: {e}. "
                    f"Install langchain-mcp-adapters to use MCP tools."
                )
                raise MCPNotAvailableError(
                    category,
                    f"MCP integration module not installed: {e}",
                )
            except (MCPTimeoutError, MCPNotAvailableError, MCPConnectionError):
                # Re-raise our custom exceptions as-is
                raise
            except Exception as e:
                logger.error(
                    f"[MCP] Unexpected error connecting to '{category}': {type(e).__name__}: {e}",
                    exc_info=True,
                )
                raise MCPConnectionError(category, e, error_type="unknown")

    async def _get_tools(self, category: str) -> dict[str, "BaseTool"]:
        """Get tools from an MCP server, with caching.

        Args:
            category: MCP server category.

        Returns:
            Dictionary mapping tool names to BaseTool instances.
        """
        if category in self._tool_cache:
            return self._tool_cache[category]

        async with self._lock:
            if category in self._tool_cache:
                return self._tool_cache[category]

            client = await self._get_client(category)

            try:
                from src.mcp_integration import get_mcp_tools

                tools = await get_mcp_tools(client)

                # Build tool lookup by name
                tool_dict = {tool.name: tool for tool in tools}
                self._tool_cache[category] = tool_dict

                logger.debug(
                    f"Loaded {len(tool_dict)} tools for category '{category}': "
                    f"{list(tool_dict.keys())}"
                )

                return tool_dict

            except Exception as e:
                logger.error(f"Failed to get tools for category '{category}': {e}")
                raise ConnectionError(category, e)

    async def execute(
        self,
        tool_name: str,
        tool_args: dict[str, Any],
    ) -> str:
        """Execute an MCP tool with the given arguments.

        Args:
            tool_name: Full tool name (e.g., "websearch.search").
            tool_args: Arguments to pass to the tool.

        Returns:
            Tool execution result as a string.

        Raises:
            ValueError: If tool name format is invalid.
            ConnectionError: If MCP server connection fails.
            ToolNotFoundError: If tool is not found in server.
            ToolExecutionError: If tool execution fails.

        Example:
            >>> result = await executor.execute(
            ...     "websearch.search",
            ...     {"query": "AI news", "max_results": 5}
            ... )
        """
        # Parse tool name
        try:
            category, operation = self.parse_tool_name(tool_name)
        except ValueError as e:
            raise MCPExecutorError(str(e), tool_name=tool_name)

        logger.debug(
            f"Executing MCP tool: {tool_name} with args: {tool_args}"
        )

        # Route websearch tools to TavilyExecutor for fast, reliable execution
        # This bypasses the problematic langchain-mcp-adapters which spawns
        # new subprocesses for each get_tools() call
        if category == "websearch" and TavilyExecutor.is_tavily_tool(tool_name):
            logger.info(
                f"[MCP] Routing '{tool_name}' to TavilyExecutor (bypassing MCP)"
            )
            try:
                return await self._tavily_executor.execute(tool_name, tool_args)
            except TavilyExecutorError as e:
                logger.error(f"[Tavily] Execution failed: {e}")
                raise ToolExecutionError(tool_name, e)

        # For non-websearch tools, use the MCP client
        # Get tools for this category
        tools = await self._get_tools(category)

        # Find the specific tool
        # Try exact match first, then try with category prefix stripped
        tool = tools.get(operation) or tools.get(tool_name)

        if tool is None:
            # Try partial matching for tools that might have different naming
            for name, t in tools.items():
                if operation in name or name.endswith(operation):
                    tool = t
                    break

        if tool is None:
            raise ToolNotFoundError(
                tool_name=tool_name,
                available_tools=list(tools.keys()),
            )

        # Execute the tool with timeout
        try:
            # Wrap tool execution in timeout (tool_execution_timeout)
            # Default is 60 seconds per tool call
            timeout = getattr(self, '_tool_execution_timeout', 60.0)

            logger.debug(f"Executing tool '{tool_name}' with timeout: {timeout}s")
            result = await asyncio.wait_for(
                tool.ainvoke(tool_args),
                timeout=timeout
            )

            # Ensure result is a string
            if not isinstance(result, str):
                result = str(result)

            logger.debug(f"Tool '{tool_name}' executed successfully")
            return result

        except asyncio.TimeoutError:
            error_msg = f"Tool execution timed out after {timeout}s"
            logger.error(f"[MCP] Tool '{tool_name}' {error_msg}")
            raise ToolExecutionError(tool_name, TimeoutError(error_msg))
        except Exception as e:
            logger.error(f"Tool '{tool_name}' execution failed: {e}")
            raise ToolExecutionError(tool_name, e)

    async def list_available_tools(self, category: str) -> list[str]:
        """List available tools for a category.

        Args:
            category: MCP server category.

        Returns:
            List of tool names.
        """
        try:
            tools = await self._get_tools(category)
            return list(tools.keys())
        except Exception:
            return []

    async def is_category_available(self, category: str) -> bool:
        """Check if an MCP category is available.

        Args:
            category: MCP server category.

        Returns:
            True if category is available and connectable.
        """
        try:
            await self._get_client(category)
            return True
        except Exception:
            return False

    def clear_cache(self) -> None:
        """Clear all cached connections and tools.

        This is useful for testing or when connections need to be refreshed.
        """
        self._client_cache.clear()
        self._tool_cache.clear()
        logger.debug("Cleared MCP executor cache")


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "MCPToolExecutor",
    "MCPExecutorError",
    "ToolNotFoundError",
    "MCPConnectionError",
    "MCPTimeoutError",
    "MCPNotAvailableError",
    "ConnectionError",  # Backward compatibility alias
    "ToolExecutionError",
]
