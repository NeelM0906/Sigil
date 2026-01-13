"""MCP Server Configuration for Sigil Interactive CLI.

This module provides configuration and management of MCP (Model Context Protocol)
servers specifically for the interactive CLI environment.

Supported MCP Servers:
    - tavily-remote: Tavily web search via MCP remote
    - elevenlabs-mcp: ElevenLabs voice/audio via uvx
    - google-calendar-mcp: Google Calendar integration
    - twilio-mcp: Twilio communications
    - hubspot-mcp: HubSpot CRM integration
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


# =============================================================================
# MCP Server Configuration
# =============================================================================


class MCPServerConfig:
    """Configuration for an MCP server."""

    def __init__(
        self,
        name: str,
        command: str,
        args: list[str],
        env: Optional[dict[str, str]] = None,
        transport: str = "stdio",
    ) -> None:
        """Initialize MCP server configuration.

        Args:
            name: Server name (e.g., 'tavily')
            command: Command to execute (e.g., 'npx')
            args: Command arguments
            env: Environment variables for the server process
            transport: Transport type (stdio, http, sse)
        """
        self.name = name
        self.command = command
        self.args = args
        self.env = env or {}
        self.transport = transport

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary for MCP client."""
        return {
            "command": self.command,
            "args": self.args,
            "transport": self.transport,
            "env": self.env,
        }


# =============================================================================
# MCP Server Registry
# =============================================================================


class MCPRegistry:
    """Registry of available MCP servers with their configurations."""

    def __init__(self) -> None:
        """Initialize the MCP registry."""
        self._servers: Dict[str, MCPServerConfig] = {}
        self._api_keys: Dict[str, str] = {}
        self._register_servers()

    def _register_servers(self) -> None:
        """Register all available MCP servers."""
        # Load environment variables
        self._load_env_vars()

        # Tavily web search via mcp-remote
        tavily_key = self._api_keys.get("TAVILY_API_KEY", "")
        if tavily_key:
            self._servers["tavily"] = MCPServerConfig(
                name="tavily",
                command="npx",
                args=["-y", "mcp-remote", f"https://mcp.tavily.com/mcp/?tavilyApiKey={tavily_key}"],
                transport="stdio",
            )
            logger.debug(f"Registered Tavily MCP server")

        # ElevenLabs voice
        elevenlabs_key = self._api_keys.get("ELEVENLABS_API_KEY", "")
        if elevenlabs_key:
            self._servers["elevenlabs"] = MCPServerConfig(
                name="elevenlabs",
                command="uvx",
                args=["elevenlabs-mcp"],
                env={"ELEVENLABS_API_KEY": elevenlabs_key},
                transport="stdio",
            )
            logger.debug(f"Registered ElevenLabs MCP server")

        # Google Calendar
        google_creds = self._api_keys.get("GOOGLE_OAUTH_CREDENTIALS", "")
        if google_creds:
            self._servers["google-calendar"] = MCPServerConfig(
                name="google-calendar",
                command="npx",
                args=["@anthropic-ai/google-calendar-mcp"],
                env={"GOOGLE_OAUTH_CREDENTIALS": google_creds},
                transport="stdio",
            )
            logger.debug(f"Registered Google Calendar MCP server")

        # Twilio communications
        twilio_sid = self._api_keys.get("TWILIO_ACCOUNT_SID", "")
        twilio_key = self._api_keys.get("TWILIO_API_KEY", "")
        twilio_secret = self._api_keys.get("TWILIO_API_SECRET", "")
        if twilio_sid and twilio_key and twilio_secret:
            self._servers["twilio"] = MCPServerConfig(
                name="twilio",
                command="npx",
                args=["-y", "@twilio-alpha/mcp"],
                env={
                    "TWILIO_ACCOUNT_SID": twilio_sid,
                    "TWILIO_API_KEY": twilio_key,
                    "TWILIO_API_SECRET": twilio_secret,
                },
                transport="stdio",
            )
            logger.debug(f"Registered Twilio MCP server")

        # HubSpot CRM
        hubspot_key = self._api_keys.get("HUBSPOT_API_KEY", "")
        if hubspot_key:
            self._servers["hubspot"] = MCPServerConfig(
                name="hubspot",
                command="npx",
                args=["-y", "@hubspot/mcp-server"],
                env={"HUBSPOT_API_KEY": hubspot_key},
                transport="stdio",
            )
            logger.debug(f"Registered HubSpot MCP server")

    def _load_env_vars(self) -> None:
        """Load environment variables from .env if present."""
        # Try to load .env file
        env_file = Path(".env")
        if env_file.exists():
            from dotenv import load_dotenv
            load_dotenv(env_file, override=False)

        # Load API keys from environment
        self._api_keys = {
            "TAVILY_API_KEY": os.getenv("TAVILY_API_KEY", ""),
            "ELEVENLABS_API_KEY": os.getenv("ELEVENLABS_API_KEY", ""),
            "GOOGLE_OAUTH_CREDENTIALS": os.getenv("GOOGLE_OAUTH_CREDENTIALS", ""),
            "TWILIO_ACCOUNT_SID": os.getenv("TWILIO_ACCOUNT_SID", ""),
            "TWILIO_API_KEY": os.getenv("TWILIO_API_KEY", ""),
            "TWILIO_API_SECRET": os.getenv("TWILIO_API_SECRET", ""),
            "HUBSPOT_API_KEY": os.getenv("HUBSPOT_API_KEY", ""),
        }

    def get_server(self, name: str) -> Optional[MCPServerConfig]:
        """Get MCP server configuration by name.

        Args:
            name: Server name (e.g., 'tavily', 'elevenlabs')

        Returns:
            MCPServerConfig if server is registered and configured, None otherwise
        """
        return self._servers.get(name)

    def list_servers(self) -> list[str]:
        """List all registered MCP servers.

        Returns:
            List of server names
        """
        return list(self._servers.keys())

    def get_available_servers(self) -> Dict[str, MCPServerConfig]:
        """Get all available MCP servers.

        Returns:
            Dictionary mapping server names to configurations
        """
        return dict(self._servers)

    def is_server_available(self, name: str) -> bool:
        """Check if a server is available.

        Args:
            name: Server name

        Returns:
            True if server is registered and configured
        """
        return name in self._servers


# =============================================================================
# Global Registry Instance
# =============================================================================

_registry: Optional[MCPRegistry] = None


def get_mcp_registry() -> MCPRegistry:
    """Get the global MCP registry.

    Returns:
        Global MCPRegistry instance
    """
    global _registry
    if _registry is None:
        _registry = MCPRegistry()
    return _registry


def reset_mcp_registry() -> None:
    """Reset the MCP registry (useful for testing).

    This clears the cached registry so a fresh one will be created
    on the next call to get_mcp_registry().
    """
    global _registry
    _registry = None


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "MCPServerConfig",
    "MCPRegistry",
    "get_mcp_registry",
    "reset_mcp_registry",
]
