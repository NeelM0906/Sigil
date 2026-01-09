"""MCP Tool Registry for ACTi Agent Builder.

This module provides the registry of official MCP (Model Context Protocol) servers
that agents can use for real tool capabilities. Each MCP server provides a set of
tools that agents can invoke to perform actual tasks like voice synthesis, web
search, calendar management, and communication.

The registry maps tool category names to their MCP server configurations,
including the command to run, arguments, and required environment variables.

Official MCP Servers Used:
    - elevenlabs-mcp (Python/uvx): Voice synthesis and TTS
    - tavily-mcp (npm): Web search and research
    - @cocal/google-calendar-mcp (npm): Google Calendar integration
    - @twilio-alpha/mcp (npm): SMS and voice calls

Usage:
    >>> from src.tool_registry import get_server_config, validate_tool_availability
    >>> if validate_tool_availability("voice"):
    ...     config = get_server_config("voice")
    ...     # Use config to start MCP server
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

from dotenv import load_dotenv

from .schemas import MCP_TOOL_CATEGORIES

# Load environment variables from .env file
load_dotenv()

# Configure module logger
logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Environment Variable Requirements
# -----------------------------------------------------------------------------

# Mapping of tool categories to their required environment variables
TOOL_ENV_REQUIREMENTS: dict[str, list[str]] = {
    "voice": [
        "ELEVENLABS_API_KEY",
    ],
    "websearch": [
        "TAVILY_API_KEY",
    ],
    "calendar": [
        "GOOGLE_CALENDAR_CREDENTIALS",
    ],
    "communication": [
        "TWILIO_ACCOUNT_SID",
        "TWILIO_AUTH_TOKEN",
    ],
    "crm": [
        "HUBSPOT_API_KEY",
    ],
}

# Optional environment variables that enhance functionality but aren't required
TOOL_ENV_OPTIONAL: dict[str, list[str]] = {
    "voice": [
        "ELEVENLABS_MCP_BASE_PATH",
        "ELEVENLABS_MCP_OUTPUT_MODE",
    ],
    "websearch": [],
    "calendar": [
        "ENABLED_TOOLS",
    ],
    "communication": [],
    "crm": [],
}


# -----------------------------------------------------------------------------
# MCP Server Configurations
# -----------------------------------------------------------------------------

def _build_mcp_servers() -> dict[str, dict[str, Any]]:
    """Build the MCP server configuration dictionary.

    This function constructs the MCP_SERVERS dict dynamically to ensure
    environment variables are read at runtime rather than import time.
    This allows for proper testing and environment switching.

    Returns:
        Dictionary mapping tool category names to MCP server configurations.
        Each configuration contains:
            - command: The executable to run (uvx, npx, etc.)
            - args: Command-line arguments for the server
            - env: Environment variables to pass to the server
            - description: Human-readable description of the tool
    """
    # Build Twilio args dynamically based on available credentials
    twilio_account_sid = os.getenv("TWILIO_ACCOUNT_SID", "")
    twilio_auth_token = os.getenv("TWILIO_AUTH_TOKEN", "")

    return {
        "voice": {
            "command": "uvx",
            "args": ["elevenlabs-mcp"],
            "env": {
                "ELEVENLABS_API_KEY": os.getenv("ELEVENLABS_API_KEY", ""),
                "ELEVENLABS_MCP_BASE_PATH": os.getenv(
                    "ELEVENLABS_MCP_BASE_PATH", "~/Desktop"
                ),
                "ELEVENLABS_MCP_OUTPUT_MODE": os.getenv(
                    "ELEVENLABS_MCP_OUTPUT_MODE", "files"
                ),
            },
            "description": "ElevenLabs voice synthesis and text-to-speech",
            "capabilities": [
                "text_to_speech",
                "voice_cloning",
                "voice_generation",
                "audio_transcription",
            ],
        },
        "websearch": {
            "command": "npx",
            "args": ["-y", "tavily-mcp@latest"],
            "env": {
                "TAVILY_API_KEY": os.getenv("TAVILY_API_KEY", ""),
            },
            "description": "Tavily web search and research",
            "capabilities": [
                "web_search",
                "web_extract",
                "website_mapping",
                "website_crawling",
            ],
        },
        "calendar": {
            "command": "npx",
            "args": ["@cocal/google-calendar-mcp"],
            "env": {
                "GOOGLE_OAUTH_CREDENTIALS": os.getenv(
                    "GOOGLE_CALENDAR_CREDENTIALS", ""
                ),
            },
            "description": "Google Calendar integration for scheduling",
            "capabilities": [
                "list_calendars",
                "list_events",
                "create_event",
                "update_event",
                "delete_event",
                "search_events",
                "get_freebusy",
            ],
        },
        "communication": {
            "command": "npx",
            "args": [
                "-y",
                "@twilio-alpha/mcp",
                f"{twilio_account_sid}:{twilio_auth_token}",
            ],
            "env": {
                "TWILIO_ACCOUNT_SID": twilio_account_sid,
                "TWILIO_AUTH_TOKEN": twilio_auth_token,
            },
            "description": "Twilio SMS and voice communication",
            "capabilities": [
                "send_sms",
                "make_call",
                "receive_messages",
                "manage_conversations",
            ],
        },
        "crm": {
            "command": "npx",
            "args": ["-y", "@hubspot/mcp-server"],
            "env": {
                "HUBSPOT_API_KEY": os.getenv("HUBSPOT_API_KEY", ""),
            },
            "description": "HubSpot CRM for contact and deal management",
            "capabilities": [
                "manage_contacts",
                "manage_deals",
                "track_interactions",
                "pipeline_management",
            ],
        },
    }


def get_mcp_servers() -> dict[str, dict[str, Any]]:
    """Get the current MCP server configurations.

    This function returns freshly-built configurations to ensure
    environment variables are current.

    Returns:
        Dictionary of MCP server configurations keyed by tool category.

    Example:
        >>> servers = get_mcp_servers()
        >>> servers["voice"]["command"]
        'uvx'
    """
    return _build_mcp_servers()


# For backward compatibility and direct access
MCP_SERVERS = _build_mcp_servers()


# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------

def get_server_config(tool_name: str) -> Optional[dict[str, Any]]:
    """Get the MCP server configuration for a specific tool category.

    Retrieves the complete configuration needed to start an MCP server
    for the specified tool category. Returns None if the tool is not found.

    Args:
        tool_name: The name of the tool category (e.g., "voice", "websearch").
            Must be one of the keys in MCP_TOOL_CATEGORIES.

    Returns:
        The MCP server configuration dictionary containing:
            - command: Executable to run
            - args: Command-line arguments
            - env: Environment variables
            - description: Tool description
            - capabilities: List of capabilities
        Returns None if the tool is not found.

    Example:
        >>> config = get_server_config("voice")
        >>> config["command"]
        'uvx'
        >>> config["args"]
        ['elevenlabs-mcp']

        >>> config = get_server_config("invalid_tool")
        >>> config is None
        True
    """
    servers = get_mcp_servers()
    config = servers.get(tool_name)

    if config is None:
        logger.warning(f"Unknown tool category: {tool_name}")
        return None

    logger.debug(f"Retrieved config for tool: {tool_name}")
    return config


def get_available_tools() -> list[str]:
    """Get a list of all available MCP tool category names.

    Returns the names of all tool categories that have MCP server
    configurations defined in the registry.

    Returns:
        Sorted list of available tool category names.

    Example:
        >>> tools = get_available_tools()
        >>> "voice" in tools
        True
        >>> "websearch" in tools
        True
    """
    servers = get_mcp_servers()
    return sorted(servers.keys())


def validate_tool_availability(tool_name: str) -> bool:
    """Check if a tool category has all required environment variables set.

    Validates that the necessary API keys and credentials are configured
    for the specified tool to function properly.

    Args:
        tool_name: The name of the tool category to validate.

    Returns:
        True if all required environment variables are set and non-empty,
        False otherwise.

    Example:
        >>> # Assuming ELEVENLABS_API_KEY is set in environment
        >>> validate_tool_availability("voice")
        True

        >>> # Assuming TAVILY_API_KEY is not set
        >>> validate_tool_availability("websearch")
        False

        >>> validate_tool_availability("unknown_tool")
        False
    """
    if tool_name not in TOOL_ENV_REQUIREMENTS:
        logger.warning(f"Unknown tool category for validation: {tool_name}")
        return False

    required_vars = TOOL_ENV_REQUIREMENTS[tool_name]
    missing = get_missing_credentials(tool_name)

    if missing:
        logger.debug(
            f"Tool '{tool_name}' missing credentials: {', '.join(missing)}"
        )
        return False

    logger.debug(f"Tool '{tool_name}' has all required credentials")
    return True


def get_missing_credentials(tool_name: str) -> list[str]:
    """Get the list of missing environment variables for a tool category.

    Checks which required environment variables are not set or empty
    for the specified tool category.

    Args:
        tool_name: The name of the tool category to check.

    Returns:
        List of environment variable names that are missing or empty.
        Returns an empty list if all required variables are set.
        Returns a list with a single "UNKNOWN_TOOL" entry if the tool
        category is not recognized.

    Example:
        >>> # Assuming ELEVENLABS_API_KEY is set
        >>> get_missing_credentials("voice")
        []

        >>> # Assuming TAVILY_API_KEY is not set
        >>> get_missing_credentials("websearch")
        ['TAVILY_API_KEY']

        >>> get_missing_credentials("unknown_tool")
        ['UNKNOWN_TOOL']
    """
    if tool_name not in TOOL_ENV_REQUIREMENTS:
        logger.warning(f"Unknown tool category: {tool_name}")
        return ["UNKNOWN_TOOL"]

    required_vars = TOOL_ENV_REQUIREMENTS[tool_name]
    missing = []

    for var in required_vars:
        value = os.getenv(var, "")
        if not value or value.strip() == "":
            missing.append(var)

    return missing


def get_tool_capabilities(tool_name: str) -> list[str]:
    """Get the list of capabilities provided by a tool category.

    Each MCP server provides specific capabilities that agents can use.
    This function returns the list of capabilities for a given tool.

    Args:
        tool_name: The name of the tool category.

    Returns:
        List of capability names for the tool, or empty list if unknown.

    Example:
        >>> caps = get_tool_capabilities("voice")
        >>> "text_to_speech" in caps
        True
    """
    config = get_server_config(tool_name)
    if config is None:
        return []
    return config.get("capabilities", [])


def get_tools_status() -> dict[str, dict[str, Any]]:
    """Get the availability status of all tool categories.

    Provides a comprehensive status report for all registered tools,
    including whether they are available and what credentials are missing.

    Returns:
        Dictionary mapping tool names to their status information:
            - available: Whether the tool is ready to use
            - missing_credentials: List of missing environment variables
            - description: Tool description
            - capabilities: List of tool capabilities

    Example:
        >>> status = get_tools_status()
        >>> status["voice"]["available"]
        True
        >>> status["voice"]["missing_credentials"]
        []
    """
    servers = get_mcp_servers()
    status = {}

    for tool_name in servers:
        config = servers[tool_name]
        missing = get_missing_credentials(tool_name)

        status[tool_name] = {
            "available": len(missing) == 0,
            "missing_credentials": missing,
            "description": config.get("description", ""),
            "capabilities": config.get("capabilities", []),
        }

    return status


def get_configured_tools() -> list[str]:
    """Get a list of tool categories that are fully configured and ready to use.

    Filters the available tools to return only those that have all
    required environment variables properly set.

    Returns:
        Sorted list of tool category names that are ready to use.

    Example:
        >>> # Assuming only ELEVENLABS_API_KEY is set
        >>> configured = get_configured_tools()
        >>> "voice" in configured
        True
    """
    available = get_available_tools()
    return sorted([
        tool for tool in available
        if validate_tool_availability(tool)
    ])


def format_tool_status_report() -> str:
    """Generate a human-readable status report for all MCP tools.

    Creates a formatted string showing the configuration status of
    all registered MCP tool categories, useful for debugging and
    user feedback.

    Returns:
        Multi-line formatted string with tool status information.

    Example:
        >>> report = format_tool_status_report()
        >>> print(report)
        MCP Tool Status Report
        ======================
        ...
    """
    status = get_tools_status()
    lines = [
        "MCP Tool Status Report",
        "=" * 50,
        "",
    ]

    for tool_name in sorted(status.keys()):
        info = status[tool_name]
        availability = "READY" if info["available"] else "NOT CONFIGURED"
        lines.append(f"  {tool_name}: {availability}")
        lines.append(f"    Description: {info['description']}")

        if info["missing_credentials"]:
            lines.append(f"    Missing: {', '.join(info['missing_credentials'])}")

        if info["capabilities"]:
            caps = ", ".join(info["capabilities"][:3])
            if len(info["capabilities"]) > 3:
                caps += f" (+{len(info['capabilities']) - 3} more)"
            lines.append(f"    Capabilities: {caps}")

        lines.append("")

    # Summary
    configured_count = len(get_configured_tools())
    total_count = len(status)
    lines.append("=" * 50)
    lines.append(f"Summary: {configured_count}/{total_count} tools configured")

    return "\n".join(lines)


# -----------------------------------------------------------------------------
# Validation Functions
# -----------------------------------------------------------------------------

def validate_tools_list(tools: list[str]) -> tuple[bool, list[str]]:
    """Validate a list of tool names against the registry.

    Checks that all provided tool names are valid MCP tool categories.
    This is useful for validating agent configurations.

    Args:
        tools: List of tool category names to validate.

    Returns:
        Tuple of (is_valid, invalid_tools):
            - is_valid: True if all tools are valid
            - invalid_tools: List of tool names that are not recognized

    Example:
        >>> valid, invalid = validate_tools_list(["voice", "websearch"])
        >>> valid
        True
        >>> invalid
        []

        >>> valid, invalid = validate_tools_list(["voice", "unknown"])
        >>> valid
        False
        >>> invalid
        ['unknown']
    """
    available = set(get_available_tools())
    invalid = [t for t in tools if t not in available]
    return len(invalid) == 0, invalid


def get_tools_for_stratum(stratum: str) -> list[str]:
    """Get recommended tools for an ACTi stratum.

    Returns the suggested MCP tools based on the agent's stratum
    classification in the ACTi methodology.

    Args:
        stratum: The ACTi stratum (RTI, RAI, ZACS, EEI, IGE).

    Returns:
        List of recommended tool category names for the stratum.

    Example:
        >>> tools = get_tools_for_stratum("ZACS")
        >>> "calendar" in tools
        True
        >>> "communication" in tools
        True
    """
    stratum_tools = {
        "RTI": ["websearch", "crm"],
        "RAI": ["communication", "crm"],
        "ZACS": ["calendar", "communication", "voice"],
        "EEI": ["websearch", "crm"],
        "IGE": ["voice", "websearch", "calendar", "communication", "crm"],
    }

    stratum_upper = stratum.upper()
    if stratum_upper not in stratum_tools:
        logger.warning(f"Unknown stratum: {stratum}")
        return []

    return stratum_tools[stratum_upper]


# -----------------------------------------------------------------------------
# Module Exports
# -----------------------------------------------------------------------------

__all__ = [
    # Main registry
    "MCP_SERVERS",
    "get_mcp_servers",
    # Configuration retrieval
    "get_server_config",
    "get_available_tools",
    "get_configured_tools",
    "get_tool_capabilities",
    # Validation
    "validate_tool_availability",
    "get_missing_credentials",
    "validate_tools_list",
    # Status reporting
    "get_tools_status",
    "format_tool_status_report",
    # ACTi integration
    "get_tools_for_stratum",
    # Constants
    "TOOL_ENV_REQUIREMENTS",
    "TOOL_ENV_OPTIONAL",
]
