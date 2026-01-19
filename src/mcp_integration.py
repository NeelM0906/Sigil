"""MCP Integration module for ACTi Agent Builder.

This module provides the bridge between agent configurations created by the builder
and real MCP (Model Context Protocol) tool servers. It enables agents to execute
actual tool operations via standardized MCP connections.

The module provides:
    - Connection management to MCP servers via langchain-mcp-adapters
    - Tool registry mapping agent tool categories to MCP server configurations
    - Agent instantiation with live MCP tools attached
    - Validation and error handling for tool availability

Usage:
    from src.mcp_integration import create_agent_with_tools
    from src.schemas import AgentConfig

    config = AgentConfig(
        name="scheduler",
        description="Schedules appointments",
        system_prompt="You are a scheduling assistant...",
        tools=["calendar", "communication"],
    )

    agent = await create_agent_with_tools(config)
    result = await agent.ainvoke({"messages": [...]})

Exports:
    create_agent_with_tools: Main entry point for creating agents with MCP tools
    connect_mcp_servers: Connect to specified MCP servers
    get_mcp_tools: Retrieve LangChain tools from MCP client
    validate_agent_tools: Validate tool availability before connection
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import TYPE_CHECKING, Literal, NamedTuple

from deepagents import create_deep_agent

from .prompts import DEFAULT_MODEL
from .schemas import AgentConfig, MCP_TOOL_CATEGORIES

if TYPE_CHECKING:
    from langchain_core.tools import BaseTool
    from langchain_mcp_adapters.client import MultiServerMCPClient
    from langgraph.graph.state import CompiledStateGraph


# Configure module logger
logger = logging.getLogger(__name__)


# =============================================================================
# Exceptions
# =============================================================================


class MCPIntegrationError(Exception):
    """Base exception for MCP integration errors."""

    pass


class MCPConfigurationError(MCPIntegrationError):
    """Raised when MCP server configuration is invalid."""

    def __init__(self, tool_name: str, message: str) -> None:
        self.tool_name = tool_name
        super().__init__(f"Configuration error for '{tool_name}': {message}")


class MCPCredentialError(MCPIntegrationError):
    """Raised when required credentials/environment variables are missing."""

    def __init__(self, tool_name: str, missing_vars: list[str]) -> None:
        self.tool_name = tool_name
        self.missing_vars = missing_vars
        vars_str = ", ".join(missing_vars)
        super().__init__(
            f"Missing credentials for '{tool_name}': {vars_str}. "
            f"Set these environment variables to use this tool."
        )


class MCPConnectionError(MCPIntegrationError):
    """Raised when connection to MCP server fails."""

    def __init__(self, tool_name: str, original_error: Exception) -> None:
        self.tool_name = tool_name
        self.original_error = original_error
        super().__init__(
            f"Failed to connect to MCP server for '{tool_name}': {original_error}"
        )


class MCPTimeoutError(MCPIntegrationError):
    """Raised when MCP server connection times out."""

    def __init__(self, tool_name: str, timeout: float) -> None:
        self.tool_name = tool_name
        self.timeout = timeout
        super().__init__(
            f"Connection to MCP server '{tool_name}' timed out after {timeout}s"
        )


# =============================================================================
# Type Definitions
# =============================================================================


class MCPServerConfig(dict):
    """Configuration for an MCP server connection.

    Attributes:
        command: Executable command (for stdio transport)
        args: Command arguments
        transport: Connection transport type (stdio, http, sse)
        env: Environment variables for the server process
        url: Server URL (for http/sse transport)
        headers: HTTP headers (for http/sse transport)
    """

    command: str
    args: list[str]
    transport: Literal["stdio", "http", "sse"]
    env: dict[str, str | None]
    url: str
    headers: dict[str, str]


class MCPConnectionResult(NamedTuple):
    """Result of MCP server connection attempt.

    Attributes:
        client: Connected MultiServerMCPClient, or None if all connections failed
        connected_tools: List of successfully connected tool categories
        failed_tools: List of tool categories that failed to connect
        errors: Mapping of tool category to error message for failures
    """

    client: MultiServerMCPClient | None
    connected_tools: list[str]
    failed_tools: list[str]
    errors: dict[str, str]


class ToolValidationResult(NamedTuple):
    """Result of tool availability validation.

    Attributes:
        valid: True if all requested tools are available
        available: List of tools that are properly configured
        missing: List of tools that are missing or misconfigured
        messages: Human-readable messages describing tool status
    """

    valid: bool
    available: list[str]
    missing: list[str]
    messages: list[str]


# =============================================================================
# MCP Server Registry
# =============================================================================


# Required environment variables for each tool category
REQUIRED_ENV_VARS: dict[str, list[str]] = {
    "voice": ["ELEVENLABS_API_KEY"],
    "websearch": ["TAVILY_API_KEY"],
    "calendar": ["GOOGLE_OAUTH_CREDENTIALS"],
    "communication": ["TWILIO_ACCOUNT_SID", "TWILIO_API_KEY", "TWILIO_API_SECRET"],
    "crm": ["HUBSPOT_API_KEY"],
}


def _get_mcp_servers() -> dict[str, MCPServerConfig]:
    """Build MCP server configurations from environment variables.

    Returns:
        Dictionary mapping tool category names to their MCP server configurations.
        Environment variables are resolved at call time.
    """
    return {
        "voice": MCPServerConfig(
            command="uvx",
            args=["elevenlabs-mcp"],
            transport="stdio",
            env={"ELEVENLABS_API_KEY": os.getenv("ELEVENLABS_API_KEY")},
        ),
        "websearch": MCPServerConfig(
            command="npx",
            args=["-y", "tavily-mcp@latest"],
            transport="stdio",
            env={"TAVILY_API_KEY": os.getenv("TAVILY_API_KEY")},
        ),
        "calendar": MCPServerConfig(
            command="npx",
            args=["@anthropic-ai/google-calendar-mcp"],
            transport="stdio",
            env={
                "GOOGLE_OAUTH_CREDENTIALS": os.getenv("GOOGLE_OAUTH_CREDENTIALS"),
            },
        ),
        "communication": MCPServerConfig(
            command="npx",
            args=["-y", "@twilio-alpha/mcp"],
            transport="stdio",
            env={
                "TWILIO_ACCOUNT_SID": os.getenv("TWILIO_ACCOUNT_SID"),
                "TWILIO_API_KEY": os.getenv("TWILIO_API_KEY"),
                "TWILIO_API_SECRET": os.getenv("TWILIO_API_SECRET"),
            },
        ),
        "crm": MCPServerConfig(
            command="npx",
            args=["-y", "@hubspot/mcp-server"],
            transport="stdio",
            env={"HUBSPOT_API_KEY": os.getenv("HUBSPOT_API_KEY")},
        ),
    }


# Lazy-loaded server registry
MCP_SERVERS: dict[str, MCPServerConfig] = {}


def _ensure_mcp_servers() -> dict[str, MCPServerConfig]:
    """Ensure MCP_SERVERS is populated (lazy initialization)."""
    global MCP_SERVERS
    if not MCP_SERVERS:
        MCP_SERVERS = _get_mcp_servers()
    return MCP_SERVERS


# =============================================================================
# Helper Functions
# =============================================================================


def _check_credentials(tool_name: str) -> list[str]:
    """Check if required credentials exist for a tool.

    Args:
        tool_name: The tool category name to check.

    Returns:
        List of missing environment variable names. Empty if all present.
    """
    required = REQUIRED_ENV_VARS.get(tool_name, [])
    missing = [var for var in required if not os.getenv(var)]
    return missing


def get_server_config(tool_name: str) -> MCPServerConfig | None:
    """Retrieve the MCP server configuration for a tool category.

    Args:
        tool_name: The tool category name (e.g., 'voice', 'websearch').

    Returns:
        MCPServerConfig if tool exists in registry, None otherwise.

    Example:
        >>> config = get_server_config("voice")
        >>> if config:
        ...     print(f"Voice server: {config['command']}")
    """
    servers = _ensure_mcp_servers()
    return servers.get(tool_name)


def list_configured_tools() -> list[str]:
    """List all tool categories configured in the MCP server registry.

    Returns:
        List of tool category names available in MCP_SERVERS.

    Example:
        >>> tools = list_configured_tools()
        >>> print(tools)
        ['voice', 'websearch', 'calendar', 'communication', 'crm']
    """
    servers = _ensure_mcp_servers()
    return list(servers.keys())


# =============================================================================
# Core Functions
# =============================================================================


async def validate_agent_tools(agent_config: AgentConfig) -> ToolValidationResult:
    """Validate that an agent's requested tools are available and properly configured.

    This performs pre-flight checks without actually connecting to MCP servers.
    Use this to verify tool availability before attempting connections.

    Args:
        agent_config: The agent configuration to validate.

    Returns:
        ToolValidationResult containing validation status and details.

    Example:
        >>> config = AgentConfig(
        ...     name="test",
        ...     description="Test agent",
        ...     system_prompt="You are a test...",
        ...     tools=["calendar", "invalid"],
        ... )
        >>> result = await validate_agent_tools(config)
        >>> if not result.valid:
        ...     print(f"Missing: {result.missing}")
    """
    servers = _ensure_mcp_servers()
    available: list[str] = []
    missing: list[str] = []
    messages: list[str] = []

    for tool_name in agent_config.tools:
        # Check if tool exists in registry
        if tool_name not in servers:
            missing.append(tool_name)
            valid_tools = ", ".join(servers.keys())
            messages.append(
                f"{tool_name}: Unknown tool category. Valid options: {valid_tools}"
            )
            continue

        # Check for required credentials
        missing_vars = _check_credentials(tool_name)
        if missing_vars:
            missing.append(tool_name)
            vars_str = ", ".join(missing_vars)
            messages.append(
                f"{tool_name}: Missing environment variable(s): {vars_str}"
            )
            continue

        # Tool is available
        available.append(tool_name)

    return ToolValidationResult(
        valid=len(missing) == 0,
        available=available,
        missing=missing,
        messages=messages,
    )


async def connect_mcp_servers(
    tool_names: list[str],
    *,
    timeout: float = 30.0,
) -> MCPConnectionResult:
    """Connect to specified MCP servers and return a client with tools.

    Args:
        tool_names: List of tool category names to connect.
        timeout: Connection timeout per server in seconds.

    Returns:
        MCPConnectionResult with connection status and client.

    Example:
        >>> result = await connect_mcp_servers(["websearch", "calendar"])
        >>> if result.client:
        ...     tools = await result.client.get_tools()
        ...     print(f"Connected with {len(tools)} tools")
    """
    # Import here to avoid import errors if langchain-mcp-adapters not installed
    try:
        from langchain_mcp_adapters.client import MultiServerMCPClient
    except ImportError as e:
        logger.error("langchain-mcp-adapters not installed: %s", e)
        return MCPConnectionResult(
            client=None,
            connected_tools=[],
            failed_tools=tool_names,
            errors={
                tool: "langchain-mcp-adapters package not installed"
                for tool in tool_names
            },
        )

    servers = _ensure_mcp_servers()
    connected: list[str] = []
    failed: list[str] = []
    errors: dict[str, str] = {}

    # Build configuration for requested tools
    server_configs: dict[str, dict] = {}

    for tool_name in tool_names:
        # Check tool exists
        if tool_name not in servers:
            failed.append(tool_name)
            valid_tools = ", ".join(servers.keys())
            errors[tool_name] = f"Unknown tool category. Valid options: {valid_tools}"
            continue

        # Check credentials
        missing_vars = _check_credentials(tool_name)
        if missing_vars:
            failed.append(tool_name)
            vars_str = ", ".join(missing_vars)
            errors[tool_name] = f"Missing credentials: {vars_str}"
            continue

        # Add to connection config
        config = servers[tool_name]
        server_configs[tool_name] = dict(config)
        connected.append(tool_name)

    # If no tools to connect, return early
    if not server_configs:
        return MCPConnectionResult(
            client=None,
            connected_tools=[],
            failed_tools=failed,
            errors=errors,
        )

    # Attempt connection
    try:
        client = MultiServerMCPClient(server_configs)

        # Verify connection by getting tools (with timeout)
        try:
            await asyncio.wait_for(
                client.get_tools(),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            # Connection timed out
            for tool in connected:
                failed.append(tool)
                errors[tool] = f"Connection timed out after {timeout}s"
            return MCPConnectionResult(
                client=None,
                connected_tools=[],
                failed_tools=failed,
                errors=errors,
            )

        logger.info("Connected to MCP servers: %s", ", ".join(connected))
        return MCPConnectionResult(
            client=client,
            connected_tools=connected,
            failed_tools=failed,
            errors=errors,
        )

    except Exception as e:
        logger.error("Failed to connect to MCP servers: %s", e)
        # Mark all as failed
        for tool in connected:
            failed.append(tool)
            errors[tool] = str(e)
        return MCPConnectionResult(
            client=None,
            connected_tools=[],
            failed_tools=failed,
            errors=errors,
        )


async def get_mcp_tools(client: MultiServerMCPClient) -> list[BaseTool]:
    """Retrieve LangChain-compatible tools from an MCP client.

    Args:
        client: A connected MultiServerMCPClient instance.

    Returns:
        List of BaseTool objects compatible with deepagents/LangChain.

    Example:
        >>> result = await connect_mcp_servers(["websearch"])
        >>> if result.client:
        ...     tools = await get_mcp_tools(result.client)
        ...     for tool in tools:
        ...         print(f"{tool.name}: {tool.description}")
    """
    tools = await client.get_tools()
    logger.debug("Retrieved %d tools from MCP client", len(tools))
    return tools


async def create_agent_with_tools(
    agent_config: AgentConfig,
    *,
    skip_unavailable: bool = False,
    timeout: float = 30.0,
    max_turns: int = 10,
) -> CompiledStateGraph:
    """Create a deepagent instance with real MCP tools attached.

    This is the main entry point for instantiating executable agents from
    configurations created by the builder. It connects to required MCP servers,
    retrieves tools, and creates a deepagent ready for invocation.

    Args:
        agent_config: Agent configuration from the builder.
        skip_unavailable: If True, create agent with available tools only.
            If False (default), raise an error if any requested tools
            are unavailable.
        timeout: Connection timeout per MCP server in seconds.
        max_turns: Maximum number of tool call rounds (default 10).
            This limits how many times the agent can call tools before
            being forced to stop. Each "turn" may involve multiple internal
            LangGraph steps, so the recursion_limit is set to max_turns * 5.
            This prevents infinite loops where agents keep calling tools
            without providing a final response.

    Returns:
        A CompiledStateGraph (deepagent) configured with:
        - System prompt from agent_config.system_prompt
        - Model from agent_config.model
        - MCP tools from connected servers

    Raises:
        MCPConfigurationError: If requested tool category is unknown.
        MCPCredentialError: If required credentials are missing
            (only when skip_unavailable=False).
        MCPConnectionError: If MCP server connection fails
            (only when skip_unavailable=False).
        MCPTimeoutError: If connection exceeds timeout
            (only when skip_unavailable=False).

    Example:
        >>> config = AgentConfig(
        ...     name="scheduler",
        ...     description="Schedules appointments",
        ...     system_prompt="You are a scheduling assistant...",
        ...     tools=["calendar", "communication"],
        ... )
        >>> agent = await create_agent_with_tools(config)
        >>> result = await agent.ainvoke({
        ...     "messages": [{"role": "user", "content": "Schedule a meeting"}]
        ... })
    """
    # Validate tools first
    validation = await validate_agent_tools(agent_config)

    if not validation.valid and not skip_unavailable:
        # Determine the specific error type
        servers = _ensure_mcp_servers()
        for tool in validation.missing:
            if tool not in servers:
                raise MCPConfigurationError(
                    tool,
                    f"Unknown tool category. Valid options: {', '.join(servers.keys())}",
                )
            missing_vars = _check_credentials(tool)
            if missing_vars:
                raise MCPCredentialError(tool, missing_vars)

    # Determine which tools to connect
    tools_to_connect = (
        validation.available if skip_unavailable else agent_config.tools
    )

    # Handle case where no tools are requested
    mcp_tools: list[BaseTool] = []

    if tools_to_connect:
        # Connect to MCP servers
        result = await connect_mcp_servers(tools_to_connect, timeout=timeout)

        if result.failed_tools and not skip_unavailable:
            # Get the first failure for error message
            first_failed = result.failed_tools[0]
            error_msg = result.errors.get(first_failed, "Unknown error")

            if "timed out" in error_msg.lower():
                raise MCPTimeoutError(first_failed, timeout)
            else:
                raise MCPConnectionError(first_failed, Exception(error_msg))

        # Log any failures when skipping
        if result.failed_tools and skip_unavailable:
            for tool, error in result.errors.items():
                logger.warning("Tool '%s' unavailable: %s", tool, error)

        # Get tools from client
        if result.client:
            mcp_tools = await get_mcp_tools(result.client)
            logger.info(
                "Creating agent '%s' with %d MCP tools",
                agent_config.name,
                len(mcp_tools),
            )

    # Create the deepagent with MCP tools
    agent = create_deep_agent(
        model=agent_config.model or DEFAULT_MODEL,
        tools=mcp_tools,
        system_prompt=agent_config.system_prompt,
    )

    # Apply recursion limit configuration to prevent infinite loops
    # Each "turn" can involve multiple internal LangGraph steps, so we multiply by 5
    recursion_limit = max_turns * 5
    logger.debug(
        "Applying recursion_limit=%d (max_turns=%d) to agent '%s'",
        recursion_limit,
        max_turns,
        agent_config.name,
    )

    return agent.with_config({
        "recursion_limit": recursion_limit,
    })


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    # Constants
    "MCP_SERVERS",
    "REQUIRED_ENV_VARS",
    # Main functions
    "create_agent_with_tools",
    "connect_mcp_servers",
    "get_mcp_tools",
    "validate_agent_tools",
    # Helper functions
    "get_server_config",
    "list_configured_tools",
    # Exceptions
    "MCPIntegrationError",
    "MCPConfigurationError",
    "MCPCredentialError",
    "MCPConnectionError",
    "MCPTimeoutError",
    # Types
    "MCPServerConfig",
    "MCPConnectionResult",
    "ToolValidationResult",
]
