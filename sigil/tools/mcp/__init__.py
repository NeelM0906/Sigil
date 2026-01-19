"""MCP (Model Context Protocol) integrations for Sigil v2.

This module implements MCP client functionality:
- MCP server connection management
- Tool discovery from MCP servers
- Tool invocation via MCP protocol
- Resource access via MCP

Key Components:
    - MCPClient: Connects to MCP servers
    - MCPToolProvider: Provides tools from MCP servers
    - MCPResourceProvider: Provides resources from MCP
    - MCPServerManager: Manages multiple MCP connections

Supported MCP Features:
    - Tools: Execute server-provided tools
    - Resources: Access server-provided resources
    - Prompts: Use server-provided prompts
    - Sampling: Server-side LLM sampling

TODO: Implement MCPClient with stdio transport
TODO: Implement MCPToolProvider
TODO: Implement MCPResourceProvider
TODO: Implement MCPServerManager for multi-server support
"""

__all__ = []  # Will export: MCPClient, MCPToolProvider, MCPServerManager
