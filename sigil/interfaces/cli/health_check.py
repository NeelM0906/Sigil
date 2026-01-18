"""Health check utilities for Sigil CLI.

This module provides utilities to check the availability of external services
and display their status to the user.
"""

import logging
import os
from typing import Any, Dict

logger = logging.getLogger(__name__)


# =============================================================================
# Tool Status Checking
# =============================================================================


async def check_tool_availability() -> Dict[str, Dict[str, Any]]:
    """Check which external tools are available and configured.

    Returns:
        Dictionary mapping tool names to their status:
        {
            "tavily": {
                "available": True/False,
                "reason": "OK" or error message
            },
            ...
        }
    """
    results = {}

    # Check Tavily (websearch)
    tavily_key = os.getenv("TAVILY_API_KEY")
    if tavily_key:
        results["tavily"] = {
            "available": True,
            "reason": "Configured and ready",
        }
    else:
        results["tavily"] = {
            "available": False,
            "reason": "Missing: TAVILY_API_KEY",
        }

    return results


def format_tool_status(status: Dict[str, Dict[str, Any]]) -> str:
    """Format tool status for CLI display.

    Args:
        status: Dictionary from check_tool_availability()

    Returns:
        Formatted string for CLI output
    """
    # Display name mapping
    display_names = {
        "tavily": "Web Search (Tavily)",
    }

    lines = []
    lines.append("External Tools:")

    available_count = sum(1 for s in status.values() if s["available"])
    total_count = len(status)
    lines.append(f"  Status: {available_count}/{total_count} available")
    lines.append("")

    for tool_name, info in sorted(status.items()):
        display_name = display_names.get(tool_name, tool_name.title())
        status_icon = "[OK]" if info["available"] else "[--]"
        reason = info.get("reason", "Unknown")
        lines.append(f"  {status_icon} {display_name}: {reason}")

    return "\n".join(lines)


async def display_tool_status(echo_func=None) -> Dict[str, Dict[str, Any]]:
    """Check and display tool status.

    Args:
        echo_func: Optional function to display output (e.g., click.echo)

    Returns:
        Status dictionary
    """
    if echo_func is None:
        echo_func = print

    echo_func("\nExternal Tool Configuration:")
    status = await check_tool_availability()

    formatted = format_tool_status(status)
    echo_func(formatted)

    return status


# Backward compatibility aliases
check_mcp_server_availability = check_tool_availability
format_mcp_status = format_tool_status
display_mcp_status = display_tool_status
