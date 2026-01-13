"""Health check utilities for MCP servers in Sigil CLI.

This module provides utilities to check the availability of MCP servers
and display their status to the user.
"""

import asyncio
import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)


# =============================================================================
# MCP Status Checking
# =============================================================================


async def check_mcp_server_availability() -> Dict[str, Dict[str, Any]]:
    """Check which MCP servers are available and configured.

    Returns:
        Dictionary mapping server names to their status:
        {
            "tavily": {
                "available": True/False,
                "reason": "OK" or error message
            },
            ...
        }
    """
    from sigil.interfaces.cli.mcp_config import get_mcp_registry

    registry = get_mcp_registry()
    results = {}

    # Check each registered server
    for server_name in registry.list_servers():
        server = registry.get_server(server_name)
        if server:
            results[server_name] = {
                "available": True,
                "reason": "Configured and ready",
            }
        else:
            results[server_name] = {
                "available": False,
                "reason": "Not configured",
            }

    # Check for unconfigured but known servers
    known_servers = {
        "tavily": ("Web Search (Tavily)", ["TAVILY_API_KEY"]),
        "elevenlabs": ("Voice Communication (ElevenLabs)", ["ELEVENLABS_API_KEY"]),
        "google-calendar": ("Calendar Management", ["GOOGLE_OAUTH_CREDENTIALS"]),
        "twilio": ("Communications", ["TWILIO_ACCOUNT_SID", "TWILIO_API_KEY", "TWILIO_API_SECRET"]),
        "hubspot": ("CRM Integration", ["HUBSPOT_API_KEY"]),
    }

    for server_name, (display_name, required_vars) in known_servers.items():
        if server_name not in results:
            import os
            missing_vars = [var for var in required_vars if not os.getenv(var)]
            if missing_vars:
                results[server_name] = {
                    "available": False,
                    "reason": f"Missing: {', '.join(missing_vars)}",
                }

    return results


def format_mcp_status(status: Dict[str, Dict[str, Any]]) -> str:
    """Format MCP server status for CLI display.

    Args:
        status: Dictionary from check_mcp_server_availability()

    Returns:
        Formatted string for CLI output
    """
    # Display name mapping
    display_names = {
        "tavily": "Web Search (Tavily)",
        "elevenlabs": "Voice Communication (ElevenLabs)",
        "google-calendar": "Calendar Management",
        "twilio": "Communications (Twilio)",
        "hubspot": "CRM Integration (HubSpot)",
    }

    lines = []
    lines.append("MCP Servers:")

    available_count = sum(1 for s in status.values() if s["available"])
    total_count = len(status)
    lines.append(f"  Status: {available_count}/{total_count} available")
    lines.append("")

    for server_name, info in sorted(status.items()):
        display_name = display_names.get(server_name, server_name.title())
        status_icon = "✓" if info["available"] else "✗"
        reason = info.get("reason", "Unknown")
        lines.append(f"  {status_icon} {display_name}: {reason}")

    return "\n".join(lines)


async def display_mcp_status(echo_func=None) -> Dict[str, Dict[str, Any]]:
    """Check and display MCP server status.

    Args:
        echo_func: Optional function to display output (e.g., click.echo)

    Returns:
        Status dictionary
    """
    if echo_func is None:
        echo_func = print

    echo_func("\nMCP Server Configuration:")
    status = await check_mcp_server_availability()

    formatted = format_mcp_status(status)
    echo_func(formatted)

    return status
