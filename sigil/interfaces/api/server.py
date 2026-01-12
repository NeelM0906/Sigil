"""FastAPI Server for Sigil v2 REST API.

This module implements the FastAPI server with REST endpoints and
WebSocket support for the Sigil framework.

Core Endpoints:
    - POST /agents - Create a new agent
    - GET /agents - List all agents
    - GET /agents/{name} - Get agent details
    - POST /agents/{name}/run - Run agent with input
    - DELETE /agents/{name} - Delete agent

    - POST /memory/search - Search memory
    - POST /memory/store - Store memory item

    - GET /tools - List available tools
    - GET /health - Health check

WebSocket:
    - /ws/agents/{name}/run - Stream agent execution

Features:
    - API key authentication
    - CORS support
    - Request validation
    - Error handling
    - Streaming responses
    - Backpressure handling

Example:
    >>> from sigil.interfaces.api.server import create_app
    >>>
    >>> app = create_app()
    >>> # Run with uvicorn
    >>> # uvicorn sigil.interfaces.api.server:app --reload
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, AsyncGenerator, Optional

from fastapi import (
    FastAPI,
    HTTPException,
    Depends,
    Request,
    WebSocket,
    WebSocketDisconnect,
    status,
    Query,
    Header,
    BackgroundTasks,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from sigil.config import get_settings
from sigil.config.settings import SigilSettings
from sigil.orchestrator import (
    SigilOrchestrator,
    OrchestratorRequest,
    OrchestratorResponse,
    OrchestratorStatus,
)


# =============================================================================
# Module Logger
# =============================================================================

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

API_VERSION = "v1"
API_TITLE = "Sigil Agent Framework API"
API_DESCRIPTION = """
Sigil is a self-improving agent framework that creates executable AI agents
with real tool capabilities, persistent memory, hierarchical reasoning,
and contract-based verification.

## Features

- **Agent Management**: Create, run, and manage AI agents
- **Memory System**: 3-layer memory with RAG and LLM retrieval
- **Planning**: Automatic task decomposition
- **Contracts**: Output verification guarantees
- **Evolution**: Self-improvement capabilities

## Authentication

All endpoints require an API key passed in the `X-API-Key` header.
"""

# WebSocket settings
WS_QUEUE_MAX_SIZE = 100
WS_HEARTBEAT_INTERVAL = 30


# =============================================================================
# Request/Response Models
# =============================================================================


class AgentCreateRequest(BaseModel):
    """Request to create a new agent."""
    name: str = Field(..., min_length=1, max_length=100, description="Agent name")
    description: str = Field(..., description="Agent description")
    system_prompt: str = Field(..., description="System prompt for the agent")
    tools: list[str] = Field(default_factory=list, description="List of tool names")
    contract_name: Optional[str] = Field(None, description="Contract to enforce")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class AgentRunRequest(BaseModel):
    """Request to run an agent."""
    message: str = Field(..., min_length=1, description="Message to process")
    context: dict[str, Any] = Field(default_factory=dict, description="Additional context")
    session_id: Optional[str] = Field(None, description="Session ID for continuity")
    stream: bool = Field(False, description="Whether to stream response")


class AgentResponse(BaseModel):
    """Response containing agent information."""
    name: str
    description: str
    system_prompt: str
    tools: list[str]
    contract_name: Optional[str]
    metadata: dict[str, Any]
    created_at: str
    updated_at: str


class AgentRunResponse(BaseModel):
    """Response from running an agent."""
    request_id: str
    status: str
    output: dict[str, Any]
    tokens_used: int
    execution_time_ms: float
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class MemorySearchRequest(BaseModel):
    """Request to search memory."""
    query: str = Field(..., min_length=1, description="Search query")
    k: int = Field(10, ge=1, le=100, description="Number of results")
    category: Optional[str] = Field(None, description="Filter by category")


class MemoryStoreRequest(BaseModel):
    """Request to store a memory item."""
    content: str = Field(..., min_length=1, description="Content to store")
    category: str = Field("general", description="Memory category")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Metadata")


class MemoryItem(BaseModel):
    """A memory item."""
    item_id: str
    content: str
    category: str
    created_at: str
    metadata: dict[str, Any]


class ToolInfo(BaseModel):
    """Information about a tool."""
    name: str
    description: str
    parameters: dict[str, Any]
    enabled: bool


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str
    features: list[str]
    uptime_seconds: float


class ErrorResponse(BaseModel):
    """Error response."""
    error: str
    code: str
    detail: Optional[str] = None


# =============================================================================
# Application State
# =============================================================================


@dataclass
class AppState:
    """Application state."""
    orchestrator: SigilOrchestrator
    settings: SigilSettings
    start_time: datetime
    agents: dict[str, dict[str, Any]]
    active_connections: list[WebSocket]


# Global state (initialized on startup)
_app_state: Optional[AppState] = None


def get_state() -> AppState:
    """Get application state."""
    if _app_state is None:
        raise RuntimeError("Application not initialized")
    return _app_state


# =============================================================================
# Authentication
# =============================================================================


async def verify_api_key(
    x_api_key: Optional[str] = Header(None),
) -> str:
    """Verify API key from header.

    Args:
        x_api_key: API key from header

    Returns:
        The API key if valid

    Raises:
        HTTPException: If API key is invalid
    """
    if not x_api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API key",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    # Get expected API key from settings
    settings = get_settings()
    expected_key = settings.api_keys.anthropic_api_key

    # In production, you'd validate against stored keys
    # For now, we accept any non-empty key
    if len(x_api_key) < 10:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
        )

    return x_api_key


# =============================================================================
# Lifespan Management
# =============================================================================


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Manage application lifecycle.

    Initializes resources on startup and cleans up on shutdown.
    """
    global _app_state

    logger.info("Starting Sigil API server...")

    # Initialize settings and orchestrator
    settings = get_settings()
    orchestrator = SigilOrchestrator(settings=settings)

    # Initialize state
    _app_state = AppState(
        orchestrator=orchestrator,
        settings=settings,
        start_time=datetime.now(timezone.utc),
        agents={},  # In-memory agent storage (use DB in production)
        active_connections=[],
    )

    logger.info(
        f"Sigil API initialized with features: {settings.get_active_features()}"
    )

    yield

    # Cleanup
    logger.info("Shutting down Sigil API server...")

    # Close active WebSocket connections
    for ws in _app_state.active_connections:
        try:
            await ws.close()
        except Exception:
            pass

    _app_state = None


# =============================================================================
# Create Application
# =============================================================================


def create_app() -> FastAPI:
    """Create and configure the FastAPI application.

    Returns:
        Configured FastAPI application
    """
    app = FastAPI(
        title=API_TITLE,
        description=API_DESCRIPTION,
        version=API_VERSION,
        lifespan=lifespan,
    )

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Register routes
    register_routes(app)

    return app


def register_routes(app: FastAPI) -> None:
    """Register all API routes.

    Args:
        app: FastAPI application
    """

    # =========================================================================
    # Health & Info
    # =========================================================================

    @app.get(
        "/health",
        response_model=HealthResponse,
        tags=["System"],
        summary="Health check",
    )
    async def health_check() -> HealthResponse:
        """Check API health status."""
        state = get_state()
        uptime = (datetime.now(timezone.utc) - state.start_time).total_seconds()

        return HealthResponse(
            status="healthy",
            version=API_VERSION,
            features=state.settings.get_active_features(),
            uptime_seconds=uptime,
        )

    @app.get(
        "/",
        tags=["System"],
        summary="API root",
    )
    async def root() -> dict[str, str]:
        """API root endpoint."""
        return {
            "name": "Sigil Agent Framework API",
            "version": API_VERSION,
            "docs": "/docs",
        }

    # =========================================================================
    # Agent Endpoints
    # =========================================================================

    @app.post(
        "/agents",
        response_model=AgentResponse,
        status_code=status.HTTP_201_CREATED,
        tags=["Agents"],
        summary="Create agent",
        dependencies=[Depends(verify_api_key)],
    )
    async def create_agent(
        request: AgentCreateRequest,
    ) -> AgentResponse:
        """Create a new agent."""
        state = get_state()

        if request.name in state.agents:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Agent '{request.name}' already exists",
            )

        now = datetime.now(timezone.utc).isoformat()
        agent_data = {
            "name": request.name,
            "description": request.description,
            "system_prompt": request.system_prompt,
            "tools": request.tools,
            "contract_name": request.contract_name,
            "metadata": request.metadata,
            "created_at": now,
            "updated_at": now,
        }

        state.agents[request.name] = agent_data

        logger.info(f"Created agent: {request.name}")

        return AgentResponse(**agent_data)

    @app.get(
        "/agents",
        response_model=list[AgentResponse],
        tags=["Agents"],
        summary="List agents",
        dependencies=[Depends(verify_api_key)],
    )
    async def list_agents(
        limit: int = Query(100, ge=1, le=1000),
        offset: int = Query(0, ge=0),
    ) -> list[AgentResponse]:
        """List all agents."""
        state = get_state()
        agents = list(state.agents.values())
        return [AgentResponse(**a) for a in agents[offset : offset + limit]]

    @app.get(
        "/agents/{name}",
        response_model=AgentResponse,
        tags=["Agents"],
        summary="Get agent",
        dependencies=[Depends(verify_api_key)],
    )
    async def get_agent(name: str) -> AgentResponse:
        """Get agent by name."""
        state = get_state()

        if name not in state.agents:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Agent '{name}' not found",
            )

        return AgentResponse(**state.agents[name])

    @app.delete(
        "/agents/{name}",
        status_code=status.HTTP_204_NO_CONTENT,
        tags=["Agents"],
        summary="Delete agent",
        dependencies=[Depends(verify_api_key)],
        response_model=None,
    )
    async def delete_agent(name: str):
        """Delete an agent."""
        state = get_state()

        if name not in state.agents:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Agent '{name}' not found",
            )

        del state.agents[name]
        logger.info(f"Deleted agent: {name}")

    @app.post(
        "/agents/{name}/run",
        response_model=AgentRunResponse,
        tags=["Agents"],
        summary="Run agent",
        dependencies=[Depends(verify_api_key)],
    )
    async def run_agent(
        name: str,
        request: AgentRunRequest,
    ) -> AgentRunResponse:
        """Run an agent with the given input."""
        state = get_state()

        if name not in state.agents:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Agent '{name}' not found",
            )

        agent = state.agents[name]

        # Create orchestrator request
        session_id = request.session_id or str(uuid.uuid4())
        orch_request = OrchestratorRequest(
            message=request.message,
            session_id=session_id,
            agent_name=name,
            context={
                **request.context,
                "system_prompt": agent["system_prompt"],
            },
            contract_name=agent.get("contract_name"),
        )

        # Process request
        response = await state.orchestrator.process(orch_request)

        return AgentRunResponse(
            request_id=response.request_id,
            status=response.status.value,
            output=response.output,
            tokens_used=response.tokens_used,
            execution_time_ms=response.execution_time_ms,
            errors=response.errors,
            warnings=response.warnings,
        )

    # =========================================================================
    # Memory Endpoints
    # =========================================================================

    @app.post(
        "/memory/search",
        response_model=list[MemoryItem],
        tags=["Memory"],
        summary="Search memory",
        dependencies=[Depends(verify_api_key)],
    )
    async def search_memory(
        request: MemorySearchRequest,
    ) -> list[MemoryItem]:
        """Search memory for relevant items."""
        state = get_state()

        # Check if memory is enabled
        if not state.settings.use_memory:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Memory system not enabled",
            )

        # For now, return empty list (memory manager integration pending)
        # In production, this would call state.orchestrator._memory_manager
        return []

    @app.post(
        "/memory/store",
        response_model=MemoryItem,
        status_code=status.HTTP_201_CREATED,
        tags=["Memory"],
        summary="Store memory",
        dependencies=[Depends(verify_api_key)],
    )
    async def store_memory(
        request: MemoryStoreRequest,
    ) -> MemoryItem:
        """Store a new memory item."""
        state = get_state()

        if not state.settings.use_memory:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Memory system not enabled",
            )

        # Create memory item
        item = MemoryItem(
            item_id=str(uuid.uuid4()),
            content=request.content,
            category=request.category,
            created_at=datetime.now(timezone.utc).isoformat(),
            metadata=request.metadata,
        )

        # In production, this would store via memory manager
        return item

    # =========================================================================
    # Tools Endpoints
    # =========================================================================

    @app.get(
        "/tools",
        response_model=list[ToolInfo],
        tags=["Tools"],
        summary="List tools",
        dependencies=[Depends(verify_api_key)],
    )
    async def list_tools() -> list[ToolInfo]:
        """List available tools."""
        # Return static list of available tools
        # In production, this would query the tool registry
        return [
            ToolInfo(
                name="websearch",
                description="Search the web for information",
                parameters={"query": {"type": "string"}},
                enabled=True,
            ),
            ToolInfo(
                name="voice",
                description="Text-to-speech synthesis",
                parameters={"text": {"type": "string"}},
                enabled=True,
            ),
            ToolInfo(
                name="calendar",
                description="Calendar management",
                parameters={"action": {"type": "string"}},
                enabled=False,
            ),
        ]

    # =========================================================================
    # WebSocket Endpoints
    # =========================================================================

    @app.websocket("/ws/agents/{name}/run")
    async def websocket_run_agent(
        websocket: WebSocket,
        name: str,
    ) -> None:
        """WebSocket endpoint for streaming agent execution.

        Handles:
        - Connection management
        - Message streaming
        - Heartbeat/keepalive
        - Backpressure via bounded queue
        """
        state = get_state()

        # Verify agent exists
        if name not in state.agents:
            await websocket.close(code=4004, reason=f"Agent '{name}' not found")
            return

        await websocket.accept()
        state.active_connections.append(websocket)

        logger.info(f"WebSocket connected for agent: {name}")

        # Create bounded queue for backpressure
        message_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(
            maxsize=WS_QUEUE_MAX_SIZE
        )

        async def send_messages() -> None:
            """Send messages from queue to client."""
            while True:
                try:
                    message = await message_queue.get()
                    await websocket.send_json(message)
                    message_queue.task_done()
                except Exception as e:
                    logger.warning(f"WebSocket send error: {e}")
                    break

        # Start sender task
        sender_task = asyncio.create_task(send_messages())

        try:
            while True:
                # Receive message from client
                data = await websocket.receive_json()

                # Validate message
                if "message" not in data:
                    await message_queue.put({
                        "type": "error",
                        "error": "Missing 'message' field",
                    })
                    continue

                agent = state.agents[name]
                session_id = data.get("session_id", str(uuid.uuid4()))

                # Send acknowledgment
                await message_queue.put({
                    "type": "ack",
                    "session_id": session_id,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                })

                # Create request
                orch_request = OrchestratorRequest(
                    message=data["message"],
                    session_id=session_id,
                    agent_name=name,
                    context={
                        **data.get("context", {}),
                        "system_prompt": agent["system_prompt"],
                    },
                    contract_name=agent.get("contract_name"),
                )

                # Process request
                # Send progress updates
                await message_queue.put({
                    "type": "progress",
                    "status": "processing",
                    "message": "Processing request...",
                })

                response = await state.orchestrator.process(orch_request)

                # Send result
                await message_queue.put({
                    "type": "result",
                    "request_id": response.request_id,
                    "status": response.status.value,
                    "output": response.output,
                    "tokens_used": response.tokens_used,
                    "execution_time_ms": response.execution_time_ms,
                    "errors": response.errors,
                    "warnings": response.warnings,
                })

        except WebSocketDisconnect:
            logger.info(f"WebSocket disconnected for agent: {name}")
        except Exception as e:
            logger.exception(f"WebSocket error: {e}")
            try:
                await websocket.send_json({
                    "type": "error",
                    "error": str(e),
                })
            except Exception:
                pass
        finally:
            # Cleanup
            sender_task.cancel()
            if websocket in state.active_connections:
                state.active_connections.remove(websocket)

    # =========================================================================
    # Metrics Endpoints
    # =========================================================================

    @app.get(
        "/metrics",
        tags=["System"],
        summary="Get metrics",
        dependencies=[Depends(verify_api_key)],
    )
    async def get_metrics() -> dict[str, Any]:
        """Get system metrics."""
        state = get_state()

        return {
            "orchestrator": state.orchestrator.get_metrics(),
            "agents_count": len(state.agents),
            "active_connections": len(state.active_connections),
            "uptime_seconds": (
                datetime.now(timezone.utc) - state.start_time
            ).total_seconds(),
        }

    # =========================================================================
    # Error Handlers
    # =========================================================================

    @app.exception_handler(HTTPException)
    async def http_exception_handler(
        request: Request,
        exc: HTTPException,
    ) -> JSONResponse:
        """Handle HTTP exceptions."""
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": exc.detail,
                "code": f"HTTP_{exc.status_code}",
            },
        )

    @app.exception_handler(Exception)
    async def general_exception_handler(
        request: Request,
        exc: Exception,
    ) -> JSONResponse:
        """Handle unexpected exceptions."""
        logger.exception(f"Unexpected error: {exc}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": "Internal server error",
                "code": "INTERNAL_ERROR",
            },
        )


# =============================================================================
# Application Instance
# =============================================================================

# Create the application instance
app = create_app()


# =============================================================================
# CLI Entry Point
# =============================================================================


def run_server(
    host: str = "0.0.0.0",
    port: int = 8000,
    reload: bool = False,
) -> None:
    """Run the API server.

    Args:
        host: Host to bind to
        port: Port to listen on
        reload: Enable auto-reload for development
    """
    import uvicorn

    uvicorn.run(
        "sigil.interfaces.api.server:app",
        host=host,
        port=port,
        reload=reload,
    )


if __name__ == "__main__":
    run_server(reload=True)


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Application
    "app",
    "create_app",
    # Request/Response models
    "AgentCreateRequest",
    "AgentRunRequest",
    "AgentResponse",
    "AgentRunResponse",
    "MemorySearchRequest",
    "MemoryStoreRequest",
    "MemoryItem",
    "ToolInfo",
    "HealthResponse",
    "ErrorResponse",
    # State
    "AppState",
    "get_state",
    # Utilities
    "run_server",
]
