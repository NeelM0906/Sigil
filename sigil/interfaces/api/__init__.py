"""FastAPI interface for Sigil v2 framework.

This module provides the REST API and WebSocket interfaces.

Key Components:
    - app: FastAPI application instance
    - create_app: Factory function for app creation
    - run_server: CLI entry point for server

Endpoints:
    - POST /agents - Create agent
    - GET /agents - List agents
    - GET /agents/{name} - Get agent
    - POST /agents/{name}/run - Run agent
    - DELETE /agents/{name} - Delete agent
    - POST /memory/search - Search memory
    - POST /memory/store - Store memory
    - GET /tools - List tools
    - GET /health - Health check
    - /ws/agents/{name}/run - WebSocket streaming

Example:
    >>> from sigil.interfaces.api import app, run_server
    >>>
    >>> # Run with uvicorn directly
    >>> # uvicorn sigil.interfaces.api:app --reload
    >>>
    >>> # Or use the run_server function
    >>> run_server(host="0.0.0.0", port=8000)
"""

from sigil.interfaces.api.server import (
    app,
    create_app,
    run_server,
    AppState,
    get_state,
    # Request models
    AgentCreateRequest,
    AgentRunRequest,
    MemorySearchRequest,
    MemoryStoreRequest,
    # Response models
    AgentResponse,
    AgentRunResponse,
    MemoryItem,
    ToolInfo,
    HealthResponse,
    ErrorResponse,
)

__all__ = [
    # Application
    "app",
    "create_app",
    "run_server",
    # State
    "AppState",
    "get_state",
    # Request models
    "AgentCreateRequest",
    "AgentRunRequest",
    "MemorySearchRequest",
    "MemoryStoreRequest",
    # Response models
    "AgentResponse",
    "AgentRunResponse",
    "MemoryItem",
    "ToolInfo",
    "HealthResponse",
    "ErrorResponse",
]
