"""ACTi Agent Builder - A meta-agent framework for creating executable AI agents with real tool capabilities."""

import logging

__version__ = "0.1.0"

# Configure default logging for the package
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# Set default level for the package's loggers
logging.getLogger("src").setLevel(logging.INFO)

from .builder import create_builder
from .tool_registry import (
    MCP_SERVERS,
    get_mcp_servers,
    get_server_config,
    get_available_tools,
    get_configured_tools,
    validate_tool_availability,
    get_missing_credentials,
    get_tools_status,
    format_tool_status_report,
)
from .mcp_integration import (
    create_agent_with_tools,
    connect_mcp_servers,
    get_mcp_tools,
    validate_agent_tools,
    list_configured_tools,
    MCPIntegrationError,
    MCPConfigurationError,
    MCPCredentialError,
    MCPConnectionError,
    MCPTimeoutError,
)
from .tools import (
    BUILDER_TOOLS,
    execute_created_agent,
    execute_created_agent_async,
)

# CLI entry point (lazy import to avoid circular dependencies)
def run_cli():
    """Run the interactive CLI."""
    from .cli import main
    main()

__all__ = [
    "create_builder",
    "run_cli",
    "__version__",
    # Tool registry exports
    "MCP_SERVERS",
    "get_mcp_servers",
    "get_server_config",
    "get_available_tools",
    "get_configured_tools",
    "validate_tool_availability",
    "get_missing_credentials",
    "get_tools_status",
    "format_tool_status_report",
    # MCP integration exports
    "create_agent_with_tools",
    "connect_mcp_servers",
    "get_mcp_tools",
    "validate_agent_tools",
    "list_configured_tools",
    "MCPIntegrationError",
    "MCPConfigurationError",
    "MCPCredentialError",
    "MCPConnectionError",
    "MCPTimeoutError",
    # Builder tools exports
    "BUILDER_TOOLS",
    "execute_created_agent",
    "execute_created_agent_async",
]
