# Phase 7 API Contract: Integration & Polish

## Document Information

| Field | Value |
|-------|-------|
| Version | 1.0.0 |
| Status | Draft |
| Last Updated | 2026-01-11 |
| Authors | API Architecture Team |
| Reviewers | Backend Team, Frontend Team |

---

## Table of Contents

1. [Overview](#overview)
2. [Design Principles](#design-principles)
3. [SigilOrchestrator API](#sigilorchestrator-api)
4. [ContextManager API](#contextmanager-api)
5. [EvolutionManager API](#evolutionmanager-api)
6. [WebSocket Streaming API](#websocket-streaming-api)
7. [Error Handling](#error-handling)
8. [Event Contracts](#event-contracts)
9. [Integration Patterns](#integration-patterns)
10. [Performance Contracts](#performance-contracts)

---

## Overview

Phase 7 introduces the **Integration & Polish** layer for Sigil v2, providing unified orchestration across all subsystems. This document defines the complete API contracts for three core components:

- **SigilOrchestrator**: Unified request routing and execution coordination
- **ContextManager**: Context assembly, compression, and window management
- **EvolutionManager**: Agent evolution, evaluation, and optimization

### System Context

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           PHASE 7: INTEGRATION LAYER                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                        SigilOrchestrator                                │ │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                  │ │
│  │  │   Request    │  │   Routing    │  │   Response   │                  │ │
│  │  │  Validation  │→→│  & Dispatch  │→→│   Assembly   │                  │ │
│  │  └──────────────┘  └──────────────┘  └──────────────┘                  │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                    │                                         │
│            ┌───────────────────────┼───────────────────────┐                │
│            ▼                       ▼                       ▼                │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐          │
│  │  ContextManager  │  │   Phase 3-6      │  │ EvolutionManager │          │
│  │                  │  │   Subsystems     │  │                  │          │
│  │  - Assembly      │  │  - Router        │  │  - Evaluate      │          │
│  │  - Compression   │  │  - Memory        │  │  - Optimize      │          │
│  │  - Window Mgmt   │  │  - Planning      │  │  - Version       │          │
│  └──────────────────┘  │  - Reasoning     │  └──────────────────┘          │
│                        │  - Contracts     │                                 │
│                        └──────────────────┘                                 │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Token Budget Architecture

Phase 7 operates with a **256K total token budget**:

```
┌─────────────────────────────────────────────────────────────────┐
│                    TOTAL BUDGET: 256,000 tokens                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  INPUT BUDGET: 150,000 tokens (58.6%)                           │
│  ├── System Context: 20,000 (13.3%)                             │
│  ├── Memory Context: 40,000 (26.7%)                             │
│  ├── Conversation History: 50,000 (33.3%)                       │
│  ├── Tool Definitions: 15,000 (10.0%)                           │
│  ├── Current Request: 20,000 (13.3%)                            │
│  └── Reserved/Overhead: 5,000 (3.3%)                            │
│                                                                  │
│  OUTPUT BUDGET: 102,400 tokens (40.0%)                          │
│  ├── Response Content: 80,000 (78.1%)                           │
│  ├── Tool Calls: 15,000 (14.6%)                                 │
│  └── Metadata/Events: 7,400 (7.2%)                              │
│                                                                  │
│  SAFETY MARGIN: 3,600 tokens (1.4%)                             │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Design Principles

### 1. Unified Request Format

All requests flow through a single orchestrator entry point with consistent structure:

```python
@dataclass
class OrchestratorRequest:
    """Unified request format for all orchestrator operations."""

    # Required fields
    request_id: str                    # UUID v4, client-generated
    operation: OperationType           # Enum: CREATE, RUN, QUERY, EVOLVE

    # Operation-specific payload
    payload: Union[
        CreateAgentPayload,
        RunAgentPayload,
        QueryPayload,
        EvolvePayload,
    ]

    # Optional configuration
    config: Optional[RequestConfig] = None

    # Metadata
    metadata: Optional[RequestMetadata] = None
```

### 2. Separation of Concerns

```
┌─────────────────────────────────────────────────────────────────┐
│                     SEPARATION OF CONCERNS                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ORCHESTRATOR                                                    │
│  └── Request validation, routing, response assembly              │
│                                                                  │
│  CONTEXT MANAGER                                                 │
│  └── Token budgeting, context assembly, compression              │
│                                                                  │
│  EVOLUTION MANAGER                                               │
│  └── Agent evaluation, optimization, versioning                  │
│                                                                  │
│  SUBSYSTEMS (Phase 3-6)                                         │
│  └── Domain-specific logic (routing, memory, planning, etc.)    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 3. Consistent Error Handling

All errors follow RFC 9457 Problem Details format:

```python
@dataclass
class ProblemDetail:
    """RFC 9457 Problem Details for HTTP APIs."""

    type: str              # URI reference identifying error type
    title: str             # Short, human-readable summary
    status: int            # HTTP status code
    detail: str            # Human-readable explanation
    instance: str          # URI reference to specific occurrence

    # Extensions
    error_code: str        # Machine-readable error code (e.g., "ORCH_001")
    timestamp: datetime    # When the error occurred
    request_id: str        # Correlation ID

    # Optional context
    validation_errors: Optional[List[ValidationError]] = None
    recovery_hints: Optional[List[str]] = None
    retry_after: Optional[int] = None  # Seconds until retry
```

### 4. Response Metadata

All responses include operational metadata:

```python
@dataclass
class ResponseMetadata:
    """Metadata included in all orchestrator responses."""

    # Identifiers
    request_id: str                    # Echo of request ID
    session_id: Optional[str]          # Session correlation

    # Token accounting
    tokens_used: TokenUsage            # Input/output/total breakdown
    tokens_remaining: int              # Remaining in budget

    # Performance
    latency_ms: int                    # Total request latency
    subsystem_latencies: Dict[str, int]  # Per-subsystem breakdown

    # Version information
    api_version: str                   # API version (e.g., "7.0.0")
    orchestrator_version: str          # Orchestrator version

    # Tracing
    trace_id: Optional[str]            # Distributed tracing ID
    span_id: Optional[str]             # Span within trace
```

---

## SigilOrchestrator API

### Class Definition

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import (
    Any, AsyncIterator, Callable, Dict, List,
    Optional, Set, Tuple, Union
)
from uuid import UUID, uuid4

import structlog

logger = structlog.get_logger(__name__)


class OperationType(str, Enum):
    """Supported orchestrator operations."""

    # Agent lifecycle
    CREATE = "create"
    RUN = "run"
    PAUSE = "pause"
    RESUME = "resume"
    TERMINATE = "terminate"

    # Query operations
    STATUS = "status"
    HISTORY = "history"
    METRICS = "metrics"

    # Memory operations
    MEMORY_SEARCH = "memory_search"
    MEMORY_STORE = "memory_store"
    MEMORY_CONSOLIDATE = "memory_consolidate"

    # Evolution operations
    EVALUATE = "evaluate"
    OPTIMIZE = "optimize"
    COMPARE = "compare"


class RequestPriority(str, Enum):
    """Request priority levels."""

    LOW = "low"           # Background tasks, batch operations
    NORMAL = "normal"     # Standard interactive requests
    HIGH = "high"         # Time-sensitive operations
    CRITICAL = "critical" # System-critical, bypass throttling


@dataclass
class RequestConfig:
    """Configuration for request processing."""

    # Timeout settings
    timeout_ms: int = 30000            # Overall timeout (default 30s)
    llm_timeout_ms: int = 60000        # LLM call timeout (default 60s)
    tool_timeout_ms: int = 10000       # Tool call timeout (default 10s)

    # Token budgets (override session defaults)
    max_input_tokens: Optional[int] = None
    max_output_tokens: Optional[int] = None
    max_total_tokens: Optional[int] = None

    # Execution preferences
    priority: RequestPriority = RequestPriority.NORMAL
    allow_cache: bool = True           # Allow cached responses
    require_fresh: bool = False        # Force fresh computation

    # Streaming configuration
    stream: bool = False               # Enable streaming response
    stream_events: Set[str] = field(default_factory=lambda: {
        "token", "tool_call", "tool_result", "status", "error", "complete"
    })

    # Retry configuration
    max_retries: int = 3
    retry_backoff_base: float = 1.0    # Exponential backoff base
    retry_backoff_max: float = 30.0    # Maximum backoff delay

    # Context configuration
    include_memory: bool = True
    memory_retrieval_k: int = 10       # Number of memory items
    include_history: bool = True
    history_turns: int = 10            # Conversation turns to include

    # Contract enforcement
    enforce_contracts: bool = True
    contract_timeout_ms: int = 5000


@dataclass
class RequestMetadata:
    """Client-provided request metadata."""

    # Client identification
    client_id: Optional[str] = None
    client_version: Optional[str] = None

    # User context
    user_id: Optional[str] = None
    tenant_id: Optional[str] = None

    # Tracing
    trace_id: Optional[str] = None
    parent_span_id: Optional[str] = None

    # Custom attributes
    attributes: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TokenUsage:
    """Token usage breakdown."""

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0

    # Breakdown by component
    system_tokens: int = 0
    memory_tokens: int = 0
    history_tokens: int = 0
    tool_tokens: int = 0
    response_tokens: int = 0

    # Cost estimation (optional, provider-specific)
    estimated_cost_usd: Optional[float] = None

    def __add__(self, other: "TokenUsage") -> "TokenUsage":
        """Add two TokenUsage instances."""
        return TokenUsage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
            system_tokens=self.system_tokens + other.system_tokens,
            memory_tokens=self.memory_tokens + other.memory_tokens,
            history_tokens=self.history_tokens + other.history_tokens,
            tool_tokens=self.tool_tokens + other.tool_tokens,
            response_tokens=self.response_tokens + other.response_tokens,
        )


class SigilOrchestrator:
    """
    Unified orchestration layer for Sigil v2.

    The SigilOrchestrator serves as the single entry point for all
    operations, coordinating between subsystems and ensuring consistent
    request handling, error management, and response assembly.

    Architecture:

        Request → Validate → Route → Execute → Assemble → Response
                     │          │        │          │
                     ▼          ▼        ▼          ▼
                  Schema    Subsystem  Context   Metadata
                  Check     Selection  Manager   Addition

    Thread Safety:
        The orchestrator is thread-safe and can handle concurrent
        requests. Internal state is protected by asyncio locks.

    Example:
        >>> orchestrator = SigilOrchestrator(config)
        >>> await orchestrator.initialize()
        >>>
        >>> request = OrchestratorRequest(
        ...     request_id=str(uuid4()),
        ...     operation=OperationType.RUN,
        ...     payload=RunAgentPayload(
        ...         agent_name="sales_qualifier",
        ...         input="Qualify this lead: John from Acme Corp",
        ...     ),
        ... )
        >>>
        >>> response = await orchestrator.handle(request)
        >>> print(response.result)
    """

    def __init__(
        self,
        config: "OrchestratorConfig",
        context_manager: Optional["ContextManager"] = None,
        evolution_manager: Optional["EvolutionManager"] = None,
        event_store: Optional["EventStore"] = None,
    ):
        """
        Initialize the orchestrator.

        Args:
            config: Orchestrator configuration
            context_manager: Context management instance (created if None)
            evolution_manager: Evolution management instance (created if None)
            event_store: Event store for audit logging (created if None)
        """
        self._config = config
        self._context_manager = context_manager
        self._evolution_manager = evolution_manager
        self._event_store = event_store

        # Subsystem references (initialized lazily)
        self._router: Optional["Router"] = None
        self._memory_manager: Optional["MemoryManager"] = None
        self._planner: Optional["Planner"] = None
        self._reasoning_manager: Optional["ReasoningManager"] = None
        self._contract_executor: Optional["ContractExecutor"] = None

        # State
        self._initialized: bool = False
        self._sessions: Dict[str, "Session"] = {}
        self._active_requests: Dict[str, "RequestContext"] = {}

        # Concurrency control
        self._request_semaphore: Optional[asyncio.Semaphore] = None
        self._session_lock: asyncio.Lock = asyncio.Lock()

        # Metrics
        self._metrics = OrchestratorMetrics()

    async def initialize(self) -> None:
        """
        Initialize the orchestrator and all subsystems.

        This method must be called before handling any requests.
        It initializes all subsystems, establishes connections,
        and validates the configuration.

        Raises:
            OrchestratorInitializationError: If initialization fails

        Example:
            >>> orchestrator = SigilOrchestrator(config)
            >>> await orchestrator.initialize()
            >>> # Now ready to handle requests
        """
        if self._initialized:
            logger.warning("Orchestrator already initialized")
            return

        logger.info("Initializing SigilOrchestrator", config=self._config)

        try:
            # Initialize concurrency control
            self._request_semaphore = asyncio.Semaphore(
                self._config.max_concurrent_requests
            )

            # Initialize context manager
            if self._context_manager is None:
                self._context_manager = ContextManager(
                    ContextManagerConfig(
                        max_input_tokens=self._config.max_input_tokens,
                        max_output_tokens=self._config.max_output_tokens,
                    )
                )
            await self._context_manager.initialize()

            # Initialize evolution manager
            if self._evolution_manager is None:
                self._evolution_manager = EvolutionManager(
                    EvolutionManagerConfig()
                )
            await self._evolution_manager.initialize()

            # Initialize event store
            if self._event_store is None:
                self._event_store = EventStore(EventStoreConfig())
            await self._event_store.initialize()

            # Initialize subsystems
            await self._initialize_subsystems()

            self._initialized = True

            # Emit initialization event
            await self._emit_event(OrchestratorInitializedEvent(
                orchestrator_id=self._config.orchestrator_id,
                version=self._config.version,
                subsystems_initialized=[
                    "context_manager",
                    "evolution_manager",
                    "router",
                    "memory_manager",
                    "planner",
                    "reasoning_manager",
                    "contract_executor",
                ],
            ))

            logger.info("SigilOrchestrator initialized successfully")

        except Exception as e:
            logger.error("Failed to initialize orchestrator", error=str(e))
            raise OrchestratorInitializationError(
                f"Failed to initialize orchestrator: {e}"
            ) from e

    async def shutdown(self) -> None:
        """
        Gracefully shutdown the orchestrator.

        This method:
        1. Stops accepting new requests
        2. Waits for active requests to complete (with timeout)
        3. Shuts down all subsystems
        4. Releases resources

        Raises:
            OrchestratorShutdownError: If shutdown fails
        """
        if not self._initialized:
            return

        logger.info("Shutting down SigilOrchestrator")

        try:
            # Stop accepting new requests
            self._initialized = False

            # Wait for active requests (with timeout)
            if self._active_requests:
                logger.info(
                    "Waiting for active requests to complete",
                    count=len(self._active_requests),
                )
                await asyncio.wait_for(
                    self._wait_for_active_requests(),
                    timeout=self._config.shutdown_timeout_s,
                )

            # Shutdown subsystems
            await self._shutdown_subsystems()

            # Shutdown managers
            if self._context_manager:
                await self._context_manager.shutdown()
            if self._evolution_manager:
                await self._evolution_manager.shutdown()
            if self._event_store:
                await self._event_store.shutdown()

            # Emit shutdown event
            await self._emit_event(OrchestratorShutdownEvent(
                orchestrator_id=self._config.orchestrator_id,
                active_requests_cancelled=len(self._active_requests),
            ))

            logger.info("SigilOrchestrator shutdown complete")

        except asyncio.TimeoutError:
            logger.warning(
                "Shutdown timeout, cancelling active requests",
                count=len(self._active_requests),
            )
            await self._cancel_active_requests()
        except Exception as e:
            logger.error("Error during shutdown", error=str(e))
            raise OrchestratorShutdownError(
                f"Failed to shutdown orchestrator: {e}"
            ) from e

    async def handle(
        self,
        request: "OrchestratorRequest",
    ) -> "OrchestratorResponse":
        """
        Handle an orchestrator request.

        This is the main entry point for all operations. The method:
        1. Validates the request
        2. Acquires resources (semaphore, session)
        3. Routes to appropriate handler
        4. Assembles response with metadata
        5. Emits audit events

        Args:
            request: The orchestrator request to handle

        Returns:
            OrchestratorResponse with result and metadata

        Raises:
            OrchestratorNotInitializedError: If not initialized
            RequestValidationError: If request validation fails
            RequestTimeoutError: If request times out
            OrchestratorError: For other errors

        Example:
            >>> response = await orchestrator.handle(request)
            >>> if response.success:
            ...     print(response.result)
            ... else:
            ...     print(response.error)
        """
        if not self._initialized:
            raise OrchestratorNotInitializedError(
                "Orchestrator must be initialized before handling requests"
            )

        start_time = datetime.utcnow()
        request_context = None

        try:
            # Validate request
            await self._validate_request(request)

            # Create request context
            request_context = await self._create_request_context(request)

            # Acquire semaphore
            async with self._request_semaphore:
                # Register active request
                self._active_requests[request.request_id] = request_context

                try:
                    # Emit request started event
                    await self._emit_event(RequestStartedEvent(
                        request_id=request.request_id,
                        operation=request.operation,
                        priority=request.config.priority if request.config else RequestPriority.NORMAL,
                    ))

                    # Route and execute
                    result = await self._route_and_execute(request, request_context)

                    # Assemble response
                    response = await self._assemble_response(
                        request=request,
                        result=result,
                        context=request_context,
                        start_time=start_time,
                    )

                    # Emit request completed event
                    await self._emit_event(RequestCompletedEvent(
                        request_id=request.request_id,
                        operation=request.operation,
                        success=True,
                        tokens_used=response.metadata.tokens_used.total_tokens,
                        latency_ms=response.metadata.latency_ms,
                    ))

                    return response

                finally:
                    # Unregister active request
                    self._active_requests.pop(request.request_id, None)

        except RequestValidationError as e:
            return self._create_error_response(
                request=request,
                error=e,
                error_code="ORCH_VALIDATION_001",
                status=400,
                start_time=start_time,
            )

        except RequestTimeoutError as e:
            return self._create_error_response(
                request=request,
                error=e,
                error_code="ORCH_TIMEOUT_001",
                status=408,
                start_time=start_time,
            )

        except SubsystemError as e:
            return self._create_error_response(
                request=request,
                error=e,
                error_code=f"ORCH_SUBSYSTEM_{e.subsystem.upper()}_001",
                status=500,
                start_time=start_time,
            )

        except Exception as e:
            logger.exception("Unexpected error handling request", request_id=request.request_id)
            return self._create_error_response(
                request=request,
                error=e,
                error_code="ORCH_INTERNAL_001",
                status=500,
                start_time=start_time,
            )

    async def handle_stream(
        self,
        request: "OrchestratorRequest",
    ) -> AsyncIterator["StreamEvent"]:
        """
        Handle a streaming orchestrator request.

        This method yields events as they occur during request
        processing, enabling real-time updates to clients.

        Args:
            request: The orchestrator request (must have stream=True)

        Yields:
            StreamEvent objects as processing progresses

        Raises:
            OrchestratorNotInitializedError: If not initialized
            RequestValidationError: If request validation fails
            StreamingNotSupportedError: If operation doesn't support streaming

        Example:
            >>> request.config.stream = True
            >>> async for event in orchestrator.handle_stream(request):
            ...     if event.type == "token":
            ...         print(event.data.content, end="")
            ...     elif event.type == "complete":
            ...         print("\\nDone!")
        """
        if not self._initialized:
            raise OrchestratorNotInitializedError(
                "Orchestrator must be initialized before handling requests"
            )

        # Ensure streaming is enabled
        if not request.config or not request.config.stream:
            request.config = request.config or RequestConfig()
            request.config.stream = True

        start_time = datetime.utcnow()
        request_context = None

        try:
            # Validate request
            await self._validate_request(request)

            # Check streaming support
            if not self._supports_streaming(request.operation):
                raise StreamingNotSupportedError(
                    f"Operation {request.operation} does not support streaming"
                )

            # Create request context
            request_context = await self._create_request_context(request)

            # Acquire semaphore
            async with self._request_semaphore:
                # Register active request
                self._active_requests[request.request_id] = request_context

                try:
                    # Emit stream started event
                    yield StreamEvent(
                        type="stream_start",
                        data=StreamStartData(
                            request_id=request.request_id,
                            operation=request.operation,
                        ),
                        timestamp=datetime.utcnow(),
                    )

                    # Route and execute with streaming
                    async for event in self._route_and_execute_stream(
                        request, request_context
                    ):
                        yield event

                    # Calculate final metrics
                    end_time = datetime.utcnow()
                    latency_ms = int((end_time - start_time).total_seconds() * 1000)

                    # Emit stream complete event
                    yield StreamEvent(
                        type="complete",
                        data=StreamCompleteData(
                            request_id=request.request_id,
                            tokens_used=request_context.tokens_used,
                            latency_ms=latency_ms,
                        ),
                        timestamp=end_time,
                    )

                finally:
                    # Unregister active request
                    self._active_requests.pop(request.request_id, None)

        except Exception as e:
            # Yield error event
            yield StreamEvent(
                type="error",
                data=StreamErrorData(
                    request_id=request.request_id,
                    error_code=self._get_error_code(e),
                    error_message=str(e),
                ),
                timestamp=datetime.utcnow(),
            )

    # ─────────────────────────────────────────────────────────────────
    # Agent Lifecycle Operations
    # ─────────────────────────────────────────────────────────────────

    async def create_agent(
        self,
        payload: "CreateAgentPayload",
        config: Optional[RequestConfig] = None,
    ) -> "CreateAgentResult":
        """
        Create a new agent.

        This is a convenience method that wraps handle() for agent creation.

        Args:
            payload: Agent creation configuration
            config: Optional request configuration

        Returns:
            CreateAgentResult with agent details

        Example:
            >>> result = await orchestrator.create_agent(
            ...     CreateAgentPayload(
            ...         name="sales_qualifier",
            ...         stratum="RAI",
            ...         description="Qualifies sales leads using BANT",
            ...         tools=["websearch", "crm"],
            ...     )
            ... )
            >>> print(f"Created agent: {result.agent_name}")
        """
        request = OrchestratorRequest(
            request_id=str(uuid4()),
            operation=OperationType.CREATE,
            payload=payload,
            config=config,
        )

        response = await self.handle(request)

        if not response.success:
            raise AgentCreationError(
                f"Failed to create agent: {response.error.detail}"
            )

        return response.result

    async def run_agent(
        self,
        agent_name: str,
        input: str,
        session_id: Optional[str] = None,
        config: Optional[RequestConfig] = None,
    ) -> "RunAgentResult":
        """
        Run an agent with the given input.

        Args:
            agent_name: Name of the agent to run
            input: User input/query
            session_id: Optional session ID for conversation continuity
            config: Optional request configuration

        Returns:
            RunAgentResult with agent response

        Example:
            >>> result = await orchestrator.run_agent(
            ...     agent_name="sales_qualifier",
            ...     input="Qualify lead: John from Acme Corp, CEO",
            ...     session_id="session_123",
            ... )
            >>> print(result.response)
        """
        payload = RunAgentPayload(
            agent_name=agent_name,
            input=input,
            session_id=session_id,
        )

        request = OrchestratorRequest(
            request_id=str(uuid4()),
            operation=OperationType.RUN,
            payload=payload,
            config=config,
        )

        response = await self.handle(request)

        if not response.success:
            raise AgentExecutionError(
                f"Failed to run agent: {response.error.detail}"
            )

        return response.result

    async def run_agent_stream(
        self,
        agent_name: str,
        input: str,
        session_id: Optional[str] = None,
        config: Optional[RequestConfig] = None,
    ) -> AsyncIterator["StreamEvent"]:
        """
        Run an agent with streaming response.

        Args:
            agent_name: Name of the agent to run
            input: User input/query
            session_id: Optional session ID
            config: Optional request configuration

        Yields:
            StreamEvent objects as processing progresses

        Example:
            >>> async for event in orchestrator.run_agent_stream(
            ...     agent_name="sales_qualifier",
            ...     input="Qualify this lead",
            ... ):
            ...     if event.type == "token":
            ...         print(event.data.content, end="", flush=True)
        """
        payload = RunAgentPayload(
            agent_name=agent_name,
            input=input,
            session_id=session_id,
        )

        config = config or RequestConfig()
        config.stream = True

        request = OrchestratorRequest(
            request_id=str(uuid4()),
            operation=OperationType.RUN,
            payload=payload,
            config=config,
        )

        async for event in self.handle_stream(request):
            yield event

    async def get_agent_status(
        self,
        agent_name: str,
    ) -> "AgentStatus":
        """
        Get the current status of an agent.

        Args:
            agent_name: Name of the agent

        Returns:
            AgentStatus with current state and metrics

        Example:
            >>> status = await orchestrator.get_agent_status("sales_qualifier")
            >>> print(f"State: {status.state}, Executions: {status.total_executions}")
        """
        payload = StatusPayload(agent_name=agent_name)

        request = OrchestratorRequest(
            request_id=str(uuid4()),
            operation=OperationType.STATUS,
            payload=payload,
        )

        response = await self.handle(request)

        if not response.success:
            raise AgentNotFoundError(
                f"Failed to get agent status: {response.error.detail}"
            )

        return response.result

    async def pause_agent(
        self,
        agent_name: str,
        session_id: str,
    ) -> "PauseResult":
        """
        Pause a running agent execution.

        Args:
            agent_name: Name of the agent
            session_id: Session ID of the execution to pause

        Returns:
            PauseResult with checkpoint information

        Example:
            >>> result = await orchestrator.pause_agent(
            ...     agent_name="sales_qualifier",
            ...     session_id="session_123",
            ... )
            >>> print(f"Paused at checkpoint: {result.checkpoint_id}")
        """
        payload = PausePayload(
            agent_name=agent_name,
            session_id=session_id,
        )

        request = OrchestratorRequest(
            request_id=str(uuid4()),
            operation=OperationType.PAUSE,
            payload=payload,
        )

        response = await self.handle(request)

        if not response.success:
            raise AgentPauseError(
                f"Failed to pause agent: {response.error.detail}"
            )

        return response.result

    async def resume_agent(
        self,
        agent_name: str,
        session_id: str,
        checkpoint_id: Optional[str] = None,
    ) -> "ResumeResult":
        """
        Resume a paused agent execution.

        Args:
            agent_name: Name of the agent
            session_id: Session ID of the paused execution
            checkpoint_id: Optional specific checkpoint to resume from

        Returns:
            ResumeResult with continuation information

        Example:
            >>> result = await orchestrator.resume_agent(
            ...     agent_name="sales_qualifier",
            ...     session_id="session_123",
            ... )
            >>> print(f"Resumed execution")
        """
        payload = ResumePayload(
            agent_name=agent_name,
            session_id=session_id,
            checkpoint_id=checkpoint_id,
        )

        request = OrchestratorRequest(
            request_id=str(uuid4()),
            operation=OperationType.RESUME,
            payload=payload,
        )

        response = await self.handle(request)

        if not response.success:
            raise AgentResumeError(
                f"Failed to resume agent: {response.error.detail}"
            )

        return response.result

    # ─────────────────────────────────────────────────────────────────
    # Memory Operations
    # ─────────────────────────────────────────────────────────────────

    async def search_memory(
        self,
        query: str,
        agent_name: Optional[str] = None,
        session_id: Optional[str] = None,
        k: int = 10,
        retrieval_method: str = "hybrid",
    ) -> "MemorySearchResult":
        """
        Search agent memory.

        Args:
            query: Search query
            agent_name: Optional agent scope
            session_id: Optional session scope
            k: Number of results to return
            retrieval_method: "rag", "llm", or "hybrid"

        Returns:
            MemorySearchResult with matching items

        Example:
            >>> result = await orchestrator.search_memory(
            ...     query="What are John's objections?",
            ...     agent_name="sales_qualifier",
            ...     k=5,
            ... )
            >>> for item in result.items:
            ...     print(f"- {item.content} (score: {item.score})")
        """
        payload = MemorySearchPayload(
            query=query,
            agent_name=agent_name,
            session_id=session_id,
            k=k,
            retrieval_method=retrieval_method,
        )

        request = OrchestratorRequest(
            request_id=str(uuid4()),
            operation=OperationType.MEMORY_SEARCH,
            payload=payload,
        )

        response = await self.handle(request)

        if not response.success:
            raise MemorySearchError(
                f"Memory search failed: {response.error.detail}"
            )

        return response.result

    async def store_memory(
        self,
        content: str,
        agent_name: str,
        session_id: Optional[str] = None,
        category: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "MemoryStoreResult":
        """
        Store information in agent memory.

        Args:
            content: Content to store
            agent_name: Agent to associate with
            session_id: Optional session association
            category: Optional category classification
            metadata: Optional additional metadata

        Returns:
            MemoryStoreResult with item ID

        Example:
            >>> result = await orchestrator.store_memory(
            ...     content="John prefers email communication",
            ...     agent_name="sales_qualifier",
            ...     category="lead_preferences",
            ... )
            >>> print(f"Stored as: {result.item_id}")
        """
        payload = MemoryStorePayload(
            content=content,
            agent_name=agent_name,
            session_id=session_id,
            category=category,
            metadata=metadata or {},
        )

        request = OrchestratorRequest(
            request_id=str(uuid4()),
            operation=OperationType.MEMORY_STORE,
            payload=payload,
        )

        response = await self.handle(request)

        if not response.success:
            raise MemoryStoreError(
                f"Memory store failed: {response.error.detail}"
            )

        return response.result

    # ─────────────────────────────────────────────────────────────────
    # Evolution Operations
    # ─────────────────────────────────────────────────────────────────

    async def evaluate_agent(
        self,
        agent_name: str,
        test_cases: Optional[List["TestCase"]] = None,
        evaluation_config: Optional["EvaluationConfig"] = None,
    ) -> "EvaluationResult":
        """
        Evaluate agent performance.

        Args:
            agent_name: Agent to evaluate
            test_cases: Optional test cases (uses stored if None)
            evaluation_config: Optional evaluation configuration

        Returns:
            EvaluationResult with scores and analysis

        Example:
            >>> result = await orchestrator.evaluate_agent(
            ...     agent_name="sales_qualifier",
            ...     test_cases=[
            ...         TestCase(input="Qualify CEO", expected_score_min=80),
            ...     ],
            ... )
            >>> print(f"Overall score: {result.overall_score}")
        """
        payload = EvaluatePayload(
            agent_name=agent_name,
            test_cases=test_cases,
            config=evaluation_config,
        )

        request = OrchestratorRequest(
            request_id=str(uuid4()),
            operation=OperationType.EVALUATE,
            payload=payload,
        )

        response = await self.handle(request)

        if not response.success:
            raise EvaluationError(
                f"Evaluation failed: {response.error.detail}"
            )

        return response.result

    async def optimize_agent(
        self,
        agent_name: str,
        optimization_target: str = "overall",
        optimization_config: Optional["OptimizationConfig"] = None,
    ) -> "OptimizationResult":
        """
        Optimize agent based on performance data.

        Args:
            agent_name: Agent to optimize
            optimization_target: What to optimize ("overall", "speed", "accuracy")
            optimization_config: Optional optimization configuration

        Returns:
            OptimizationResult with changes and new version

        Example:
            >>> result = await orchestrator.optimize_agent(
            ...     agent_name="sales_qualifier",
            ...     optimization_target="accuracy",
            ... )
            >>> print(f"New version: {result.new_version}")
            >>> for change in result.changes:
            ...     print(f"  - {change.description}")
        """
        payload = OptimizePayload(
            agent_name=agent_name,
            target=optimization_target,
            config=optimization_config,
        )

        request = OrchestratorRequest(
            request_id=str(uuid4()),
            operation=OperationType.OPTIMIZE,
            payload=payload,
        )

        response = await self.handle(request)

        if not response.success:
            raise OptimizationError(
                f"Optimization failed: {response.error.detail}"
            )

        return response.result

    # ─────────────────────────────────────────────────────────────────
    # Private Methods
    # ─────────────────────────────────────────────────────────────────

    async def _validate_request(
        self,
        request: "OrchestratorRequest",
    ) -> None:
        """Validate request structure and content."""
        # Validate request ID
        if not request.request_id:
            raise RequestValidationError(
                "request_id is required",
                field="request_id",
            )

        try:
            UUID(request.request_id)
        except ValueError:
            raise RequestValidationError(
                "request_id must be a valid UUID",
                field="request_id",
            )

        # Validate operation
        if not request.operation:
            raise RequestValidationError(
                "operation is required",
                field="operation",
            )

        if request.operation not in OperationType:
            raise RequestValidationError(
                f"Invalid operation: {request.operation}",
                field="operation",
            )

        # Validate payload
        if not request.payload:
            raise RequestValidationError(
                "payload is required",
                field="payload",
            )

        # Validate payload matches operation
        expected_payload_type = self._get_expected_payload_type(request.operation)
        if not isinstance(request.payload, expected_payload_type):
            raise RequestValidationError(
                f"Payload must be {expected_payload_type.__name__} for operation {request.operation}",
                field="payload",
            )

        # Validate config if present
        if request.config:
            self._validate_config(request.config)

    def _validate_config(self, config: RequestConfig) -> None:
        """Validate request configuration."""
        if config.timeout_ms <= 0:
            raise RequestValidationError(
                "timeout_ms must be positive",
                field="config.timeout_ms",
            )

        if config.max_retries < 0:
            raise RequestValidationError(
                "max_retries must be non-negative",
                field="config.max_retries",
            )

        if config.memory_retrieval_k <= 0:
            raise RequestValidationError(
                "memory_retrieval_k must be positive",
                field="config.memory_retrieval_k",
            )

    async def _create_request_context(
        self,
        request: "OrchestratorRequest",
    ) -> "RequestContext":
        """Create execution context for request."""
        config = request.config or RequestConfig()

        # Get or create session
        session_id = self._extract_session_id(request)
        session = await self._get_or_create_session(session_id)

        # Create context
        return RequestContext(
            request_id=request.request_id,
            session=session,
            config=config,
            start_time=datetime.utcnow(),
            tokens_used=TokenUsage(),
        )

    async def _route_and_execute(
        self,
        request: "OrchestratorRequest",
        context: "RequestContext",
    ) -> Any:
        """Route request to appropriate handler and execute."""
        handler = self._get_handler(request.operation)

        # Apply timeout
        timeout_ms = context.config.timeout_ms

        try:
            result = await asyncio.wait_for(
                handler(request, context),
                timeout=timeout_ms / 1000,
            )
            return result
        except asyncio.TimeoutError:
            raise RequestTimeoutError(
                f"Request timed out after {timeout_ms}ms",
                timeout_ms=timeout_ms,
            )

    async def _route_and_execute_stream(
        self,
        request: "OrchestratorRequest",
        context: "RequestContext",
    ) -> AsyncIterator["StreamEvent"]:
        """Route request to streaming handler."""
        handler = self._get_stream_handler(request.operation)

        async for event in handler(request, context):
            # Track tokens from events
            if hasattr(event.data, "tokens"):
                context.tokens_used += event.data.tokens
            yield event

    def _get_handler(
        self,
        operation: OperationType,
    ) -> Callable[["OrchestratorRequest", "RequestContext"], Any]:
        """Get handler for operation type."""
        handlers = {
            OperationType.CREATE: self._handle_create,
            OperationType.RUN: self._handle_run,
            OperationType.PAUSE: self._handle_pause,
            OperationType.RESUME: self._handle_resume,
            OperationType.TERMINATE: self._handle_terminate,
            OperationType.STATUS: self._handle_status,
            OperationType.HISTORY: self._handle_history,
            OperationType.METRICS: self._handle_metrics,
            OperationType.MEMORY_SEARCH: self._handle_memory_search,
            OperationType.MEMORY_STORE: self._handle_memory_store,
            OperationType.MEMORY_CONSOLIDATE: self._handle_memory_consolidate,
            OperationType.EVALUATE: self._handle_evaluate,
            OperationType.OPTIMIZE: self._handle_optimize,
            OperationType.COMPARE: self._handle_compare,
        }

        handler = handlers.get(operation)
        if not handler:
            raise UnsupportedOperationError(f"Unsupported operation: {operation}")

        return handler

    async def _assemble_response(
        self,
        request: "OrchestratorRequest",
        result: Any,
        context: "RequestContext",
        start_time: datetime,
    ) -> "OrchestratorResponse":
        """Assemble response with metadata."""
        end_time = datetime.utcnow()
        latency_ms = int((end_time - start_time).total_seconds() * 1000)

        metadata = ResponseMetadata(
            request_id=request.request_id,
            session_id=context.session.session_id if context.session else None,
            tokens_used=context.tokens_used,
            tokens_remaining=self._calculate_remaining_tokens(context),
            latency_ms=latency_ms,
            subsystem_latencies=context.subsystem_latencies,
            api_version=self._config.api_version,
            orchestrator_version=self._config.version,
            trace_id=request.metadata.trace_id if request.metadata else None,
        )

        return OrchestratorResponse(
            success=True,
            result=result,
            error=None,
            metadata=metadata,
        )

    def _create_error_response(
        self,
        request: "OrchestratorRequest",
        error: Exception,
        error_code: str,
        status: int,
        start_time: datetime,
    ) -> "OrchestratorResponse":
        """Create error response."""
        end_time = datetime.utcnow()
        latency_ms = int((end_time - start_time).total_seconds() * 1000)

        problem = ProblemDetail(
            type=f"https://sigil.acti.ai/errors/{error_code.lower().replace('_', '-')}",
            title=self._get_error_title(error),
            status=status,
            detail=str(error),
            instance=f"/requests/{request.request_id}",
            error_code=error_code,
            timestamp=end_time,
            request_id=request.request_id,
            recovery_hints=self._get_recovery_hints(error),
        )

        metadata = ResponseMetadata(
            request_id=request.request_id,
            session_id=None,
            tokens_used=TokenUsage(),
            tokens_remaining=0,
            latency_ms=latency_ms,
            subsystem_latencies={},
            api_version=self._config.api_version,
            orchestrator_version=self._config.version,
        )

        return OrchestratorResponse(
            success=False,
            result=None,
            error=problem,
            metadata=metadata,
        )
```

---

## ContextManager API

The ContextManager handles context assembly, token budgeting, and compression for optimal LLM utilization.

### Configuration

```python
@dataclass
class ContextManagerConfig:
    """Configuration for context management."""

    # Token budgets (256K total)
    max_input_tokens: int = 150000
    max_output_tokens: int = 102400
    max_total_tokens: int = 256000

    # Input budget allocation
    system_context_tokens: int = 20000       # System prompts, agent config
    memory_context_tokens: int = 40000       # Memory retrieval
    history_tokens: int = 50000              # Conversation history
    tool_definition_tokens: int = 15000      # Tool schemas
    current_request_tokens: int = 20000      # Current user input
    reserved_tokens: int = 5000              # Safety margin

    # Compression settings
    enable_compression: bool = True
    compression_threshold: float = 0.8       # Compress when >80% full
    compression_target: float = 0.6          # Target 60% after compression
    min_compression_ratio: float = 0.5       # Minimum acceptable ratio

    # Prioritization
    priority_weights: Dict[str, float] = field(default_factory=lambda: {
        "system_context": 1.0,               # Never compress
        "current_request": 0.95,             # Almost never compress
        "recent_history": 0.8,               # Prefer to keep
        "memory_items": 0.6,                 # Compress if needed
        "older_history": 0.4,                # More aggressively compress
        "tool_definitions": 0.3,             # Drop unused tools
    })

    # Caching
    enable_caching: bool = True
    cache_ttl_seconds: int = 300             # 5 minute cache
    cache_max_size_mb: int = 100


class ContextManager:
    """
    Manages context assembly and compression for LLM calls.

    The ContextManager is responsible for:
    1. Assembling context from multiple sources
    2. Enforcing token budgets across components
    3. Compressing context when budgets are exceeded
    4. Caching assembled contexts for performance

    Architecture:

        Sources                     Assembly                 Output
        ───────                     ────────                 ──────
        System Config    ──┐
        Agent Prompts    ──┼──→  ContextAssembler  ──→  AssembledContext
        Memory Items     ──┤           │                      │
        History          ──┤           ▼                      ▼
        Tool Defs        ──┤     [Compression]          TokenBudget
        User Input       ──┘     (if needed)            Verification

    Thread Safety:
        The ContextManager is thread-safe. Internal caches are protected
        by locks, and compression operations are atomic.

    Example:
        >>> manager = ContextManager(config)
        >>> await manager.initialize()
        >>>
        >>> context = await manager.assemble(
        ...     agent_config=agent_config,
        ...     session=session,
        ...     memory_items=memory_items,
        ...     user_input="Qualify this lead",
        ... )
        >>>
        >>> print(f"Context tokens: {context.total_tokens}")
        >>> print(f"Within budget: {context.within_budget}")
    """

    def __init__(self, config: ContextManagerConfig):
        """
        Initialize the context manager.

        Args:
            config: Context manager configuration
        """
        self._config = config
        self._initialized = False
        self._cache: Dict[str, CachedContext] = {}
        self._cache_lock = asyncio.Lock()
        self._compressor: Optional[ContextCompressor] = None
        self._tokenizer: Optional[Tokenizer] = None

    async def initialize(self) -> None:
        """
        Initialize the context manager.

        This method initializes the tokenizer and compressor,
        and sets up caching infrastructure.

        Raises:
            ContextManagerInitializationError: If initialization fails
        """
        if self._initialized:
            return

        try:
            # Initialize tokenizer
            self._tokenizer = await Tokenizer.create()

            # Initialize compressor
            self._compressor = ContextCompressor(
                tokenizer=self._tokenizer,
                config=CompressionConfig(
                    target_ratio=self._config.compression_target,
                    min_ratio=self._config.min_compression_ratio,
                ),
            )

            self._initialized = True
            logger.info("ContextManager initialized")

        except Exception as e:
            raise ContextManagerInitializationError(
                f"Failed to initialize ContextManager: {e}"
            ) from e

    async def shutdown(self) -> None:
        """Shutdown the context manager and release resources."""
        if not self._initialized:
            return

        # Clear cache
        async with self._cache_lock:
            self._cache.clear()

        self._initialized = False
        logger.info("ContextManager shutdown")

    async def assemble(
        self,
        agent_config: "AgentConfig",
        session: Optional["Session"] = None,
        memory_items: Optional[List["MemoryItem"]] = None,
        tools: Optional[List["ToolDefinition"]] = None,
        user_input: str = "",
        budget_override: Optional["TokenBudget"] = None,
    ) -> "AssembledContext":
        """
        Assemble context from multiple sources.

        This method combines all context sources into a single
        assembled context, applying compression if necessary
        to fit within token budgets.

        Args:
            agent_config: Agent configuration with system prompts
            session: Optional session with conversation history
            memory_items: Optional memory items to include
            tools: Optional tool definitions to include
            user_input: Current user input
            budget_override: Optional budget override

        Returns:
            AssembledContext with all components and token counts

        Raises:
            ContextAssemblyError: If assembly fails
            TokenBudgetExceededError: If cannot fit within budget

        Example:
            >>> context = await manager.assemble(
            ...     agent_config=config,
            ...     session=session,
            ...     memory_items=items,
            ...     user_input="Hello",
            ... )
        """
        if not self._initialized:
            raise ContextManagerNotInitializedError(
                "ContextManager must be initialized before use"
            )

        # Check cache
        cache_key = self._compute_cache_key(
            agent_config, session, memory_items, tools, user_input
        )
        cached = await self._get_cached(cache_key)
        if cached:
            return cached

        # Determine budget
        budget = budget_override or self._create_default_budget()

        # Assemble components
        components = await self._assemble_components(
            agent_config=agent_config,
            session=session,
            memory_items=memory_items,
            tools=tools,
            user_input=user_input,
        )

        # Calculate total tokens
        total_tokens = sum(c.token_count for c in components.values())

        # Check if compression needed
        if total_tokens > budget.max_input_tokens * self._config.compression_threshold:
            components = await self._compress_components(
                components=components,
                budget=budget,
            )
            total_tokens = sum(c.token_count for c in components.values())

        # Verify within budget
        if total_tokens > budget.max_input_tokens:
            raise TokenBudgetExceededError(
                f"Context ({total_tokens} tokens) exceeds budget ({budget.max_input_tokens})",
                actual_tokens=total_tokens,
                budget_tokens=budget.max_input_tokens,
            )

        # Create assembled context
        assembled = AssembledContext(
            system_context=components.get("system_context"),
            memory_context=components.get("memory_context"),
            history_context=components.get("history_context"),
            tool_context=components.get("tool_context"),
            user_context=components.get("user_context"),
            total_tokens=total_tokens,
            budget=budget,
            compression_applied=components.get("_compressed", False),
            compression_ratio=components.get("_compression_ratio", 1.0),
        )

        # Cache result
        await self._cache_context(cache_key, assembled)

        return assembled

    async def estimate_tokens(
        self,
        text: str,
    ) -> int:
        """
        Estimate token count for text.

        Args:
            text: Text to estimate

        Returns:
            Estimated token count
        """
        if not self._initialized:
            raise ContextManagerNotInitializedError(
                "ContextManager must be initialized"
            )

        return await self._tokenizer.count_tokens(text)

    async def compress(
        self,
        context: "AssembledContext",
        target_tokens: int,
        strategy: str = "smart",
    ) -> "AssembledContext":
        """
        Compress an assembled context to target token count.

        Args:
            context: Context to compress
            target_tokens: Target token count
            strategy: Compression strategy ("smart", "truncate", "summarize")

        Returns:
            Compressed AssembledContext

        Raises:
            CompressionError: If compression fails
            CompressionTargetUnreachableError: If target cannot be reached
        """
        if not self._initialized:
            raise ContextManagerNotInitializedError(
                "ContextManager must be initialized"
            )

        if context.total_tokens <= target_tokens:
            return context  # No compression needed

        return await self._compressor.compress(
            context=context,
            target_tokens=target_tokens,
            strategy=strategy,
        )

    async def get_budget_status(
        self,
        session_id: str,
    ) -> "BudgetStatus":
        """
        Get current budget status for a session.

        Args:
            session_id: Session ID

        Returns:
            BudgetStatus with current usage and remaining
        """
        # Implementation would track per-session usage
        raise NotImplementedError()

    # ─────────────────────────────────────────────────────────────────
    # Private Methods
    # ─────────────────────────────────────────────────────────────────

    async def _assemble_components(
        self,
        agent_config: "AgentConfig",
        session: Optional["Session"],
        memory_items: Optional[List["MemoryItem"]],
        tools: Optional[List["ToolDefinition"]],
        user_input: str,
    ) -> Dict[str, "ContextComponent"]:
        """Assemble individual context components."""
        components = {}

        # System context (agent prompts, config)
        system_text = self._build_system_context(agent_config)
        system_tokens = await self.estimate_tokens(system_text)
        components["system_context"] = ContextComponent(
            name="system_context",
            content=system_text,
            token_count=system_tokens,
            priority=self._config.priority_weights["system_context"],
            compressible=False,
        )

        # Memory context
        if memory_items:
            memory_text = self._build_memory_context(memory_items)
            memory_tokens = await self.estimate_tokens(memory_text)
            components["memory_context"] = ContextComponent(
                name="memory_context",
                content=memory_text,
                token_count=memory_tokens,
                priority=self._config.priority_weights["memory_items"],
                compressible=True,
            )

        # History context
        if session and session.history:
            history_text = self._build_history_context(session.history)
            history_tokens = await self.estimate_tokens(history_text)
            components["history_context"] = ContextComponent(
                name="history_context",
                content=history_text,
                token_count=history_tokens,
                priority=self._config.priority_weights["recent_history"],
                compressible=True,
            )

        # Tool context
        if tools:
            tool_text = self._build_tool_context(tools)
            tool_tokens = await self.estimate_tokens(tool_text)
            components["tool_context"] = ContextComponent(
                name="tool_context",
                content=tool_text,
                token_count=tool_tokens,
                priority=self._config.priority_weights["tool_definitions"],
                compressible=True,
            )

        # User context
        user_tokens = await self.estimate_tokens(user_input)
        components["user_context"] = ContextComponent(
            name="user_context",
            content=user_input,
            token_count=user_tokens,
            priority=self._config.priority_weights["current_request"],
            compressible=False,
        )

        return components

    async def _compress_components(
        self,
        components: Dict[str, "ContextComponent"],
        budget: "TokenBudget",
    ) -> Dict[str, "ContextComponent"]:
        """Compress components to fit within budget."""
        target_tokens = int(budget.max_input_tokens * self._config.compression_target)
        current_tokens = sum(c.token_count for c in components.values())

        if current_tokens <= target_tokens:
            return components

        # Sort by priority (lowest first for compression)
        sorted_components = sorted(
            [(k, v) for k, v in components.items() if v.compressible],
            key=lambda x: x[1].priority,
        )

        tokens_to_remove = current_tokens - target_tokens
        removed_tokens = 0

        for name, component in sorted_components:
            if removed_tokens >= tokens_to_remove:
                break

            # Compress this component
            compressed = await self._compressor.compress_component(
                component=component,
                target_ratio=0.5,  # Try to halve it
            )

            tokens_saved = component.token_count - compressed.token_count
            removed_tokens += tokens_saved
            components[name] = compressed

        components["_compressed"] = True
        components["_compression_ratio"] = (
            sum(c.token_count for c in components.values() if isinstance(c, ContextComponent))
            / current_tokens
        )

        return components

    def _create_default_budget(self) -> "TokenBudget":
        """Create default token budget from config."""
        return TokenBudget(
            max_input_tokens=self._config.max_input_tokens,
            max_output_tokens=self._config.max_output_tokens,
            max_total_tokens=self._config.max_total_tokens,
            allocation={
                "system_context": self._config.system_context_tokens,
                "memory_context": self._config.memory_context_tokens,
                "history": self._config.history_tokens,
                "tools": self._config.tool_definition_tokens,
                "current_request": self._config.current_request_tokens,
                "reserved": self._config.reserved_tokens,
            },
        )


# ─────────────────────────────────────────────────────────────────────────────
# Context Data Classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ContextComponent:
    """Individual component of assembled context."""

    name: str                      # Component identifier
    content: str                   # Text content
    token_count: int               # Token count
    priority: float                # Priority weight (0-1, higher = more important)
    compressible: bool             # Whether this can be compressed

    # Compression metadata
    original_token_count: Optional[int] = None
    compression_method: Optional[str] = None


@dataclass
class AssembledContext:
    """Fully assembled context ready for LLM."""

    # Components
    system_context: Optional[ContextComponent]
    memory_context: Optional[ContextComponent]
    history_context: Optional[ContextComponent]
    tool_context: Optional[ContextComponent]
    user_context: Optional[ContextComponent]

    # Token accounting
    total_tokens: int
    budget: "TokenBudget"

    # Compression info
    compression_applied: bool = False
    compression_ratio: float = 1.0

    @property
    def within_budget(self) -> bool:
        """Check if context is within budget."""
        return self.total_tokens <= self.budget.max_input_tokens

    @property
    def remaining_tokens(self) -> int:
        """Get remaining input tokens."""
        return max(0, self.budget.max_input_tokens - self.total_tokens)

    def to_messages(self) -> List[Dict[str, str]]:
        """Convert to message format for LLM."""
        messages = []

        if self.system_context:
            messages.append({
                "role": "system",
                "content": self.system_context.content,
            })

        if self.memory_context:
            messages.append({
                "role": "system",
                "content": f"<memory>\n{self.memory_context.content}\n</memory>",
            })

        if self.history_context:
            # History would be parsed into multiple messages
            # This is simplified
            messages.append({
                "role": "assistant",
                "content": self.history_context.content,
            })

        if self.user_context:
            messages.append({
                "role": "user",
                "content": self.user_context.content,
            })

        return messages


@dataclass
class TokenBudget:
    """Token budget configuration."""

    max_input_tokens: int
    max_output_tokens: int
    max_total_tokens: int

    allocation: Dict[str, int] = field(default_factory=dict)

    def get_allocation(self, component: str) -> int:
        """Get token allocation for a component."""
        return self.allocation.get(component, 0)


@dataclass
class BudgetStatus:
    """Current budget status for a session."""

    session_id: str
    total_budget: int
    used_tokens: int
    remaining_tokens: int

    # Per-request breakdown
    requests_count: int
    average_tokens_per_request: float

    # Component breakdown
    component_usage: Dict[str, int]

    @property
    def utilization(self) -> float:
        """Get budget utilization as percentage."""
        return self.used_tokens / self.total_budget if self.total_budget > 0 else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Compression Classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CompressionConfig:
    """Configuration for context compression."""

    target_ratio: float = 0.6          # Target compression ratio
    min_ratio: float = 0.5             # Minimum acceptable ratio
    max_iterations: int = 3            # Max compression iterations

    # Strategy weights
    strategy_weights: Dict[str, float] = field(default_factory=lambda: {
        "truncate": 0.3,               # Simple truncation
        "summarize": 0.5,              # LLM summarization
        "prune": 0.2,                  # Remove low-value content
    })


class ContextCompressor:
    """
    Compresses context to fit within token budgets.

    Compression Strategies:
    1. Truncate - Remove oldest/lowest priority content
    2. Summarize - Use LLM to summarize sections
    3. Prune - Remove redundant or low-value items

    Strategy Selection:
    - For history: Summarize older turns, keep recent
    - For memory: Prune by relevance score
    - For tools: Remove unused tool definitions
    """

    def __init__(
        self,
        tokenizer: "Tokenizer",
        config: CompressionConfig,
    ):
        self._tokenizer = tokenizer
        self._config = config

    async def compress(
        self,
        context: AssembledContext,
        target_tokens: int,
        strategy: str = "smart",
    ) -> AssembledContext:
        """
        Compress context to target token count.

        Args:
            context: Context to compress
            target_tokens: Target token count
            strategy: "smart", "truncate", or "summarize"

        Returns:
            Compressed context

        Raises:
            CompressionError: If compression fails
        """
        if strategy == "smart":
            return await self._smart_compress(context, target_tokens)
        elif strategy == "truncate":
            return await self._truncate_compress(context, target_tokens)
        elif strategy == "summarize":
            return await self._summarize_compress(context, target_tokens)
        else:
            raise ValueError(f"Unknown compression strategy: {strategy}")

    async def compress_component(
        self,
        component: ContextComponent,
        target_ratio: float,
    ) -> ContextComponent:
        """
        Compress a single component.

        Args:
            component: Component to compress
            target_ratio: Target ratio (0-1, e.g., 0.5 = half size)

        Returns:
            Compressed component
        """
        if not component.compressible:
            return component

        target_tokens = int(component.token_count * target_ratio)

        # Use truncation for simplicity
        compressed_content = await self._truncate_text(
            component.content,
            target_tokens,
        )

        compressed_tokens = await self._tokenizer.count_tokens(compressed_content)

        return ContextComponent(
            name=component.name,
            content=compressed_content,
            token_count=compressed_tokens,
            priority=component.priority,
            compressible=component.compressible,
            original_token_count=component.token_count,
            compression_method="truncate",
        )

    async def _smart_compress(
        self,
        context: AssembledContext,
        target_tokens: int,
    ) -> AssembledContext:
        """
        Smart compression using multiple strategies.

        Applies compression in priority order:
        1. Prune unused tool definitions
        2. Summarize old history
        3. Prune low-relevance memory
        4. Truncate if still over budget
        """
        current_tokens = context.total_tokens
        remaining_to_remove = current_tokens - target_tokens

        if remaining_to_remove <= 0:
            return context

        # Create mutable copies
        components = {
            "system_context": context.system_context,
            "memory_context": context.memory_context,
            "history_context": context.history_context,
            "tool_context": context.tool_context,
            "user_context": context.user_context,
        }

        # 1. Prune tools
        if components["tool_context"] and remaining_to_remove > 0:
            pruned = await self._prune_tools(
                components["tool_context"],
                remaining_to_remove,
            )
            saved = components["tool_context"].token_count - pruned.token_count
            remaining_to_remove -= saved
            components["tool_context"] = pruned

        # 2. Summarize history
        if components["history_context"] and remaining_to_remove > 0:
            summarized = await self._summarize_history(
                components["history_context"],
                remaining_to_remove,
            )
            saved = components["history_context"].token_count - summarized.token_count
            remaining_to_remove -= saved
            components["history_context"] = summarized

        # 3. Prune memory
        if components["memory_context"] and remaining_to_remove > 0:
            pruned = await self._prune_memory(
                components["memory_context"],
                remaining_to_remove,
            )
            saved = components["memory_context"].token_count - pruned.token_count
            remaining_to_remove -= saved
            components["memory_context"] = pruned

        # Calculate final totals
        total_tokens = sum(
            c.token_count for c in components.values() if c is not None
        )

        return AssembledContext(
            system_context=components["system_context"],
            memory_context=components["memory_context"],
            history_context=components["history_context"],
            tool_context=components["tool_context"],
            user_context=components["user_context"],
            total_tokens=total_tokens,
            budget=context.budget,
            compression_applied=True,
            compression_ratio=total_tokens / context.total_tokens,
        )

---

## EvolutionManager API

The EvolutionManager handles agent evaluation, optimization, and versioning for continuous improvement.

### Configuration

```python
@dataclass
class EvolutionManagerConfig:
    """Configuration for evolution management."""

    # Evaluation settings
    default_test_cases_count: int = 10
    evaluation_timeout_ms: int = 300000    # 5 minutes
    min_test_cases_for_evaluation: int = 3

    # Optimization settings
    optimization_iterations: int = 5
    optimization_timeout_ms: int = 600000  # 10 minutes
    min_improvement_threshold: float = 0.05  # 5% improvement required

    # Safety constraints
    max_prompt_change_ratio: float = 0.3   # Max 30% prompt change
    require_approval_for_major_changes: bool = True
    major_change_threshold: float = 0.2    # 20% = major change

    # Versioning
    auto_version_on_improvement: bool = True
    keep_version_history: int = 10         # Keep last 10 versions

    # A/B testing
    enable_ab_testing: bool = False
    ab_test_traffic_split: float = 0.1     # 10% to new version


class EvolutionManager:
    """
    Manages agent evolution, evaluation, and optimization.

    The EvolutionManager provides:
    1. Agent performance evaluation against test cases
    2. Prompt optimization using TextGrad-style techniques
    3. Version management for agent configurations
    4. A/B testing for gradual rollout

    Architecture:

        Evaluate                 Optimize                  Version
        ────────                 ────────                  ───────
        Agent + TestCases  →  Evaluator  →  Scores  →  Optimizer
                                   │                        │
                                   ▼                        ▼
                              EvaluationResult      OptimizedAgent
                                                          │
                                                          ▼
                                                   VersionManager

    Safety Constraints:
        - Prompt changes limited to 30% by default
        - Major changes require explicit approval
        - All versions are retained for rollback
        - A/B testing for gradual rollout

    Example:
        >>> manager = EvolutionManager(config)
        >>> await manager.initialize()
        >>>
        >>> # Evaluate agent
        >>> eval_result = await manager.evaluate(
        ...     agent_name="sales_qualifier",
        ...     test_cases=test_cases,
        ... )
        >>> print(f"Score: {eval_result.overall_score}")
        >>>
        >>> # Optimize if needed
        >>> if eval_result.overall_score < 0.8:
        ...     opt_result = await manager.optimize(
        ...         agent_name="sales_qualifier",
        ...         target="accuracy",
        ...     )
        ...     print(f"Improved to: {opt_result.new_score}")
    """

    def __init__(self, config: EvolutionManagerConfig):
        """
        Initialize the evolution manager.

        Args:
            config: Evolution manager configuration
        """
        self._config = config
        self._initialized = False
        self._evaluator: Optional["Evaluator"] = None
        self._optimizer: Optional["Optimizer"] = None
        self._version_manager: Optional["VersionManager"] = None

    async def initialize(self) -> None:
        """
        Initialize the evolution manager.

        Raises:
            EvolutionManagerInitializationError: If initialization fails
        """
        if self._initialized:
            return

        try:
            # Initialize evaluator
            self._evaluator = Evaluator(
                EvaluatorConfig(
                    timeout_ms=self._config.evaluation_timeout_ms,
                )
            )

            # Initialize optimizer
            self._optimizer = Optimizer(
                OptimizerConfig(
                    iterations=self._config.optimization_iterations,
                    timeout_ms=self._config.optimization_timeout_ms,
                    min_improvement=self._config.min_improvement_threshold,
                )
            )

            # Initialize version manager
            self._version_manager = VersionManager(
                VersionManagerConfig(
                    keep_versions=self._config.keep_version_history,
                )
            )

            self._initialized = True
            logger.info("EvolutionManager initialized")

        except Exception as e:
            raise EvolutionManagerInitializationError(
                f"Failed to initialize EvolutionManager: {e}"
            ) from e

    async def shutdown(self) -> None:
        """Shutdown the evolution manager."""
        if not self._initialized:
            return

        self._initialized = False
        logger.info("EvolutionManager shutdown")

    async def evaluate(
        self,
        agent_name: str,
        test_cases: Optional[List["TestCase"]] = None,
        config: Optional["EvaluationConfig"] = None,
    ) -> "EvaluationResult":
        """
        Evaluate agent performance.

        This method runs the agent against test cases and
        calculates performance scores across multiple dimensions.

        Args:
            agent_name: Name of agent to evaluate
            test_cases: Test cases (uses stored if None)
            config: Optional evaluation configuration

        Returns:
            EvaluationResult with scores and analysis

        Raises:
            AgentNotFoundError: If agent doesn't exist
            EvaluationError: If evaluation fails
            InsufficientTestCasesError: If not enough test cases

        Example:
            >>> result = await manager.evaluate(
            ...     agent_name="sales_qualifier",
            ...     test_cases=[
            ...         TestCase(
            ...             input="Qualify CEO John",
            ...             expected_output_contains=["BANT", "qualified"],
            ...             expected_score_min=80,
            ...         ),
            ...     ],
            ... )
            >>> print(f"Overall: {result.overall_score}")
            >>> for dim in result.dimension_scores:
            ...     print(f"  {dim.name}: {dim.score}")
        """
        if not self._initialized:
            raise EvolutionManagerNotInitializedError(
                "EvolutionManager must be initialized"
            )

        # Load agent
        agent = await self._load_agent(agent_name)

        # Get or load test cases
        if test_cases is None:
            test_cases = await self._load_test_cases(agent_name)

        if len(test_cases) < self._config.min_test_cases_for_evaluation:
            raise InsufficientTestCasesError(
                f"Need at least {self._config.min_test_cases_for_evaluation} test cases, "
                f"got {len(test_cases)}"
            )

        # Run evaluation
        config = config or EvaluationConfig()
        result = await self._evaluator.evaluate(
            agent=agent,
            test_cases=test_cases,
            config=config,
        )

        # Emit event
        await self._emit_event(AgentEvaluatedEvent(
            agent_name=agent_name,
            overall_score=result.overall_score,
            test_cases_count=len(test_cases),
            passed_count=result.passed_count,
            failed_count=result.failed_count,
        ))

        return result

    async def optimize(
        self,
        agent_name: str,
        target: str = "overall",
        config: Optional["OptimizationConfig"] = None,
    ) -> "OptimizationResult":
        """
        Optimize agent based on evaluation data.

        This method uses TextGrad-style optimization to improve
        agent prompts based on performance feedback.

        Args:
            agent_name: Agent to optimize
            target: Optimization target ("overall", "accuracy", "speed", "cost")
            config: Optional optimization configuration

        Returns:
            OptimizationResult with changes and new version

        Raises:
            AgentNotFoundError: If agent doesn't exist
            OptimizationError: If optimization fails
            SafetyConstraintError: If changes exceed safety limits

        Example:
            >>> result = await manager.optimize(
            ...     agent_name="sales_qualifier",
            ...     target="accuracy",
            ... )
            >>> print(f"Old score: {result.baseline_score}")
            >>> print(f"New score: {result.new_score}")
            >>> print(f"Improvement: {result.improvement_percent}%")
            >>> for change in result.changes:
            ...     print(f"  - {change.description}")
        """
        if not self._initialized:
            raise EvolutionManagerNotInitializedError(
                "EvolutionManager must be initialized"
            )

        # Load agent
        agent = await self._load_agent(agent_name)

        # Get baseline evaluation
        baseline = await self.evaluate(agent_name)

        # Run optimization
        config = config or OptimizationConfig(target=target)
        optimized_agent, changes = await self._optimizer.optimize(
            agent=agent,
            baseline=baseline,
            config=config,
        )

        # Validate safety constraints
        await self._validate_safety_constraints(agent, optimized_agent, changes)

        # Evaluate optimized agent
        optimized_result = await self._evaluator.evaluate(
            agent=optimized_agent,
            test_cases=await self._load_test_cases(agent_name),
            config=EvaluationConfig(),
        )

        # Calculate improvement
        improvement = optimized_result.overall_score - baseline.overall_score

        # Check if improvement meets threshold
        if improvement < self._config.min_improvement_threshold:
            return OptimizationResult(
                agent_name=agent_name,
                baseline_score=baseline.overall_score,
                new_score=optimized_result.overall_score,
                improvement_percent=improvement * 100,
                changes=[],
                new_version=None,
                applied=False,
                reason="Improvement below threshold",
            )

        # Create new version if auto-versioning enabled
        new_version = None
        if self._config.auto_version_on_improvement:
            new_version = await self._version_manager.create_version(
                agent_name=agent_name,
                config=optimized_agent.config,
                changes=changes,
                evaluation=optimized_result,
            )

        # Emit event
        await self._emit_event(AgentOptimizedEvent(
            agent_name=agent_name,
            baseline_score=baseline.overall_score,
            new_score=optimized_result.overall_score,
            improvement_percent=improvement * 100,
            new_version=new_version,
        ))

        return OptimizationResult(
            agent_name=agent_name,
            baseline_score=baseline.overall_score,
            new_score=optimized_result.overall_score,
            improvement_percent=improvement * 100,
            changes=changes,
            new_version=new_version,
            applied=True,
            reason=None,
        )

    async def compare_versions(
        self,
        agent_name: str,
        version_a: str,
        version_b: str,
        test_cases: Optional[List["TestCase"]] = None,
    ) -> "ComparisonResult":
        """
        Compare two agent versions.

        Args:
            agent_name: Agent name
            version_a: First version ID
            version_b: Second version ID
            test_cases: Test cases for comparison

        Returns:
            ComparisonResult with side-by-side scores

        Example:
            >>> result = await manager.compare_versions(
            ...     agent_name="sales_qualifier",
            ...     version_a="v1.0.0",
            ...     version_b="v1.1.0",
            ... )
            >>> print(f"Version A: {result.score_a}")
            >>> print(f"Version B: {result.score_b}")
            >>> print(f"Winner: {result.winner}")
        """
        if not self._initialized:
            raise EvolutionManagerNotInitializedError(
                "EvolutionManager must be initialized"
            )

        # Load both versions
        agent_a = await self._load_agent_version(agent_name, version_a)
        agent_b = await self._load_agent_version(agent_name, version_b)

        # Get test cases
        if test_cases is None:
            test_cases = await self._load_test_cases(agent_name)

        # Evaluate both
        result_a = await self._evaluator.evaluate(agent_a, test_cases)
        result_b = await self._evaluator.evaluate(agent_b, test_cases)

        # Determine winner
        if result_a.overall_score > result_b.overall_score:
            winner = version_a
        elif result_b.overall_score > result_a.overall_score:
            winner = version_b
        else:
            winner = "tie"

        return ComparisonResult(
            agent_name=agent_name,
            version_a=version_a,
            version_b=version_b,
            score_a=result_a.overall_score,
            score_b=result_b.overall_score,
            evaluation_a=result_a,
            evaluation_b=result_b,
            winner=winner,
            difference=abs(result_a.overall_score - result_b.overall_score),
        )

    async def rollback(
        self,
        agent_name: str,
        target_version: str,
    ) -> "RollbackResult":
        """
        Rollback agent to a previous version.

        Args:
            agent_name: Agent name
            target_version: Version to rollback to

        Returns:
            RollbackResult with status

        Raises:
            VersionNotFoundError: If version doesn't exist
            RollbackError: If rollback fails

        Example:
            >>> result = await manager.rollback(
            ...     agent_name="sales_qualifier",
            ...     target_version="v1.0.0",
            ... )
            >>> print(f"Rolled back from {result.from_version} to {result.to_version}")
        """
        if not self._initialized:
            raise EvolutionManagerNotInitializedError(
                "EvolutionManager must be initialized"
            )

        # Get current version
        current = await self._version_manager.get_current_version(agent_name)

        # Rollback
        await self._version_manager.set_current_version(agent_name, target_version)

        # Emit event
        await self._emit_event(AgentRolledBackEvent(
            agent_name=agent_name,
            from_version=current.version_id,
            to_version=target_version,
        ))

        return RollbackResult(
            agent_name=agent_name,
            from_version=current.version_id,
            to_version=target_version,
            success=True,
        )

    async def get_version_history(
        self,
        agent_name: str,
        limit: int = 10,
    ) -> List["VersionInfo"]:
        """
        Get version history for an agent.

        Args:
            agent_name: Agent name
            limit: Maximum versions to return

        Returns:
            List of VersionInfo objects
        """
        if not self._initialized:
            raise EvolutionManagerNotInitializedError(
                "EvolutionManager must be initialized"
            )

        return await self._version_manager.get_history(agent_name, limit)

    async def _validate_safety_constraints(
        self,
        original: "Agent",
        optimized: "Agent",
        changes: List["OptimizationChange"],
    ) -> None:
        """Validate optimization changes against safety constraints."""
        # Calculate prompt change ratio
        original_prompt = original.config.system_prompt
        optimized_prompt = optimized.config.system_prompt

        # Simple ratio based on character count
        change_ratio = abs(len(optimized_prompt) - len(original_prompt)) / len(original_prompt)

        if change_ratio > self._config.max_prompt_change_ratio:
            raise SafetyConstraintError(
                f"Prompt change ratio ({change_ratio:.2%}) exceeds limit "
                f"({self._config.max_prompt_change_ratio:.2%})"
            )

        # Check for major changes requiring approval
        if (
            change_ratio > self._config.major_change_threshold
            and self._config.require_approval_for_major_changes
        ):
            raise MajorChangeRequiresApprovalError(
                f"Change ratio ({change_ratio:.2%}) exceeds major change threshold. "
                "Explicit approval required."
            )


# ─────────────────────────────────────────────────────────────────────────────
# Evolution Data Classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TestCase:
    """Test case for agent evaluation."""

    # Input
    input: str                         # User input
    context: Optional[Dict[str, Any]] = None  # Additional context

    # Expected outputs
    expected_output: Optional[str] = None
    expected_output_contains: Optional[List[str]] = None
    expected_output_not_contains: Optional[List[str]] = None

    # Scoring
    expected_score_min: Optional[float] = None
    expected_score_max: Optional[float] = None

    # Metadata
    name: Optional[str] = None
    category: Optional[str] = None
    weight: float = 1.0               # Weight in overall score


@dataclass
class EvaluationConfig:
    """Configuration for evaluation."""

    # Dimensions to evaluate
    dimensions: List[str] = field(default_factory=lambda: [
        "accuracy",
        "relevance",
        "coherence",
        "helpfulness",
        "safety",
    ])

    # Scoring
    scoring_method: str = "llm"       # "llm", "exact", "fuzzy"
    passing_threshold: float = 0.7    # Score >= 0.7 = pass

    # Execution
    max_concurrent: int = 5
    timeout_per_case_ms: int = 30000


@dataclass
class EvaluationResult:
    """Result of agent evaluation."""

    agent_name: str
    timestamp: datetime

    # Overall metrics
    overall_score: float              # 0-1
    passed_count: int
    failed_count: int
    total_count: int

    # Dimension scores
    dimension_scores: List["DimensionScore"]

    # Per-test results
    test_results: List["TestResult"]

    # Execution metadata
    total_tokens: int
    total_latency_ms: int

    @property
    def pass_rate(self) -> float:
        """Get pass rate as percentage."""
        return self.passed_count / self.total_count if self.total_count > 0 else 0.0


@dataclass
class DimensionScore:
    """Score for a single evaluation dimension."""

    name: str                         # e.g., "accuracy"
    score: float                      # 0-1
    weight: float                     # Weight in overall score
    details: Optional[str] = None     # Explanation


@dataclass
class TestResult:
    """Result for a single test case."""

    test_case: TestCase
    passed: bool
    score: float

    # Outputs
    actual_output: str
    expected_match: bool

    # Execution
    latency_ms: int
    tokens_used: int

    # Analysis
    analysis: Optional[str] = None
    failure_reason: Optional[str] = None


@dataclass
class OptimizationConfig:
    """Configuration for optimization."""

    target: str = "overall"           # What to optimize
    iterations: int = 5               # Optimization iterations
    learning_rate: float = 0.1        # How aggressive changes are

    # Constraints
    preserve_tone: bool = True        # Preserve agent tone
    preserve_structure: bool = True   # Preserve prompt structure

    # Feedback
    use_failure_feedback: bool = True # Learn from failures
    use_success_feedback: bool = True # Learn from successes


@dataclass
class OptimizationResult:
    """Result of agent optimization."""

    agent_name: str
    baseline_score: float
    new_score: float
    improvement_percent: float

    # Changes made
    changes: List["OptimizationChange"]

    # Versioning
    new_version: Optional[str]
    applied: bool
    reason: Optional[str]


@dataclass
class OptimizationChange:
    """Single optimization change."""

    change_type: str                  # "prompt", "parameter", "tool"
    location: str                     # Where the change was made
    description: str                  # Human-readable description
    before: str                       # Original value
    after: str                        # New value
    impact_score: float               # Estimated impact


@dataclass
class ComparisonResult:
    """Result of comparing two agent versions."""

    agent_name: str
    version_a: str
    version_b: str

    score_a: float
    score_b: float

    evaluation_a: EvaluationResult
    evaluation_b: EvaluationResult

    winner: str                       # version_a, version_b, or "tie"
    difference: float


@dataclass
class RollbackResult:
    """Result of version rollback."""

    agent_name: str
    from_version: str
    to_version: str
    success: bool
    error: Optional[str] = None


@dataclass
class VersionInfo:
    """Information about an agent version."""

    version_id: str
    agent_name: str
    created_at: datetime
    created_by: str

    # Evaluation at creation
    evaluation_score: Optional[float]

    # Changes from previous
    changes: List[OptimizationChange]

    # Status
    is_current: bool
    is_production: bool
```

---

## WebSocket Streaming API

The WebSocket API provides real-time streaming for agent interactions with event-based updates and backpressure handling.

### Connection Protocol

```
Client                                Server
  |                                      |
  |  GET /ws/agents/{name}/run           |
  |  Upgrade: websocket                  |
  |------------------------------------->|
  |                                      |
  |  101 Switching Protocols             |
  |<-------------------------------------|
  |                                      |
  |  {"type": "connection_ready"}        |
  |<-------------------------------------|
  |                                      |
  |  {"type": "run_request", ...}        |
  |------------------------------------->|
  |                                      |
  |  {"type": "stream_start"}            |
  |<-------------------------------------|
  |                                      |
  |  {"type": "token", ...}              |
  |<-------------------------------------|
  |  {"type": "token", ...}              |
  |<-------------------------------------|
  |                                      |
  |  {"type": "tool_call", ...}          |
  |<-------------------------------------|
  |                                      |
  |  {"type": "tool_result", ...}        |
  |<-------------------------------------|
  |                                      |
  |  {"type": "complete", ...}           |
  |<-------------------------------------|
  |                                      |
```

### Event Types

```python
class StreamEventType(str, Enum):
    """WebSocket stream event types."""

    # Connection events
    CONNECTION_READY = "connection_ready"
    CONNECTION_ERROR = "connection_error"

    # Stream lifecycle
    STREAM_START = "stream_start"
    COMPLETE = "complete"

    # Content events
    TOKEN = "token"
    TOKEN_START = "token_start"
    TOKEN_END = "token_end"

    # Tool events
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"

    # Status events
    STATUS = "status"
    PROGRESS = "progress"

    # Error events
    ERROR = "error"
    WARNING = "warning"

    # Flow control
    BACKPRESSURE = "backpressure"
    READY = "ready"


@dataclass
class StreamEvent:
    """WebSocket stream event."""

    type: StreamEventType
    data: Union[
        TokenData,
        ToolCallData,
        ToolResultData,
        StatusData,
        ErrorData,
        CompleteData,
        BackpressureData,
    ]
    timestamp: datetime
    sequence: int                     # Event sequence number

    def to_json(self) -> str:
        """Serialize to JSON for transmission."""
        return json.dumps({
            "type": self.type.value,
            "data": asdict(self.data),
            "timestamp": self.timestamp.isoformat(),
            "sequence": self.sequence,
        })


@dataclass
class TokenData:
    """Data for token events."""

    content: str                      # Token content
    index: int                        # Token index in response
    is_complete: bool = False         # Whether this completes a word/sentence


@dataclass
class ToolCallData:
    """Data for tool call events."""

    tool_id: str                      # Unique tool call ID
    tool_name: str                    # Tool being called
    arguments: Dict[str, Any]         # Tool arguments
    status: str = "started"           # "started", "executing", "completed", "failed"


@dataclass
class ToolResultData:
    """Data for tool result events."""

    tool_id: str                      # Matching tool call ID
    success: bool
    result: Any                       # Tool return value
    error: Optional[str] = None       # Error if failed
    latency_ms: int = 0


@dataclass
class StatusData:
    """Data for status events."""

    status: str                       # Current status
    message: str                      # Human-readable message
    progress: Optional[float] = None  # Progress 0-1 if available


@dataclass
class ErrorData:
    """Data for error events."""

    error_code: str
    error_message: str
    recoverable: bool = False
    retry_after_ms: Optional[int] = None


@dataclass
class CompleteData:
    """Data for stream completion."""

    request_id: str
    tokens_used: TokenUsage
    latency_ms: int
    tool_calls_count: int = 0


@dataclass
class BackpressureData:
    """Data for backpressure signaling."""

    queue_size: int                   # Current queue size
    max_queue_size: int               # Maximum queue size
    paused: bool                      # Whether sending is paused
    resume_after_ms: Optional[int] = None


class WebSocketHandler:
    """
    Handles WebSocket connections for streaming agent interactions.

    Architecture:

        Client                Handler               Orchestrator
          |                      |                       |
          |  Connect             |                       |
          |--------------------->|                       |
          |                      |  Validate             |
          |                      |  Auth                 |
          |                      |                       |
          |  Run Request         |                       |
          |--------------------->|                       |
          |                      |  handle_stream()      |
          |                      |---------------------->|
          |                      |                       |
          |  <-- Events ---------|<-- Async Generator ---|
          |                      |                       |

    Backpressure Handling:
        The handler implements a bounded event queue with client-side
        flow control using a sliding window approach:

        1. Server maintains a queue of pending events
        2. When queue exceeds threshold, send BACKPRESSURE event
        3. Client acknowledges events with READY messages
        4. Server resumes sending when queue drains

    Example:
        >>> handler = WebSocketHandler(orchestrator, config)
        >>>
        >>> @app.websocket("/ws/agents/{name}/run")
        >>> async def websocket_endpoint(websocket, name: str):
        ...     await handler.handle(websocket, name)
    """

    def __init__(
        self,
        orchestrator: SigilOrchestrator,
        config: "WebSocketConfig",
    ):
        self._orchestrator = orchestrator
        self._config = config

    async def handle(
        self,
        websocket: "WebSocket",
        agent_name: str,
    ) -> None:
        """
        Handle a WebSocket connection.

        Args:
            websocket: WebSocket connection
            agent_name: Name of agent to interact with
        """
        connection_id = str(uuid4())

        try:
            # Accept connection
            await websocket.accept()

            # Authenticate
            await self._authenticate(websocket)

            # Send ready event
            await self._send_event(websocket, StreamEvent(
                type=StreamEventType.CONNECTION_READY,
                data=StatusData(
                    status="ready",
                    message=f"Connected to agent: {agent_name}",
                ),
                timestamp=datetime.utcnow(),
                sequence=0,
            ))

            # Handle messages
            await self._message_loop(websocket, agent_name, connection_id)

        except WebSocketDisconnect:
            logger.info("WebSocket disconnected", connection_id=connection_id)
        except Exception as e:
            logger.exception("WebSocket error", connection_id=connection_id)
            await self._send_error(websocket, e)
        finally:
            await websocket.close()

    async def _message_loop(
        self,
        websocket: "WebSocket",
        agent_name: str,
        connection_id: str,
    ) -> None:
        """Main message handling loop."""
        sequence = 1
        event_queue: asyncio.Queue[StreamEvent] = asyncio.Queue(
            maxsize=self._config.max_queue_size
        )

        while True:
            try:
                # Receive message with timeout
                message = await asyncio.wait_for(
                    websocket.receive_json(),
                    timeout=self._config.receive_timeout_s,
                )

                # Parse message type
                msg_type = message.get("type")

                if msg_type == "run_request":
                    # Start streaming agent run
                    await self._handle_run_request(
                        websocket=websocket,
                        agent_name=agent_name,
                        message=message,
                        sequence=sequence,
                        event_queue=event_queue,
                    )

                elif msg_type == "ready":
                    # Client signals readiness for more events
                    await self._handle_ready(event_queue)

                elif msg_type == "cancel":
                    # Client requests cancellation
                    await self._handle_cancel(connection_id)

                elif msg_type == "ping":
                    # Keepalive
                    await self._send_event(websocket, StreamEvent(
                        type=StreamEventType.STATUS,
                        data=StatusData(status="pong", message=""),
                        timestamp=datetime.utcnow(),
                        sequence=sequence,
                    ))
                    sequence += 1

            except asyncio.TimeoutError:
                # Send ping to check connection
                await self._send_ping(websocket)

    async def _handle_run_request(
        self,
        websocket: "WebSocket",
        agent_name: str,
        message: Dict[str, Any],
        sequence: int,
        event_queue: asyncio.Queue,
    ) -> None:
        """Handle a run request."""
        # Extract request data
        input_text = message.get("input", "")
        session_id = message.get("session_id")
        config = message.get("config", {})

        # Create request config
        request_config = RequestConfig(
            stream=True,
            **config,
        )

        # Send stream start
        await self._send_event(websocket, StreamEvent(
            type=StreamEventType.STREAM_START,
            data=StatusData(
                status="started",
                message="Processing request",
            ),
            timestamp=datetime.utcnow(),
            sequence=sequence,
        ))
        sequence += 1

        # Stream response
        async for event in self._orchestrator.run_agent_stream(
            agent_name=agent_name,
            input=input_text,
            session_id=session_id,
            config=request_config,
        ):
            # Check queue size for backpressure
            if event_queue.qsize() >= self._config.backpressure_threshold:
                await self._send_event(websocket, StreamEvent(
                    type=StreamEventType.BACKPRESSURE,
                    data=BackpressureData(
                        queue_size=event_queue.qsize(),
                        max_queue_size=self._config.max_queue_size,
                        paused=True,
                    ),
                    timestamp=datetime.utcnow(),
                    sequence=sequence,
                ))
                sequence += 1

                # Wait for client ready signal
                await event_queue.join()

            # Send event
            stream_event = StreamEvent(
                type=StreamEventType(event.type),
                data=event.data,
                timestamp=event.timestamp,
                sequence=sequence,
            )
            await self._send_event(websocket, stream_event)
            sequence += 1

    async def _send_event(
        self,
        websocket: "WebSocket",
        event: StreamEvent,
    ) -> None:
        """Send event to client."""
        await websocket.send_text(event.to_json())

    async def _send_error(
        self,
        websocket: "WebSocket",
        error: Exception,
    ) -> None:
        """Send error event to client."""
        await self._send_event(websocket, StreamEvent(
            type=StreamEventType.ERROR,
            data=ErrorData(
                error_code=self._get_error_code(error),
                error_message=str(error),
                recoverable=self._is_recoverable(error),
            ),
            timestamp=datetime.utcnow(),
            sequence=0,
        ))


@dataclass
class WebSocketConfig:
    """Configuration for WebSocket handling."""

    # Queue settings
    max_queue_size: int = 100         # Maximum pending events
    backpressure_threshold: int = 80  # Trigger backpressure at 80%

    # Timeouts
    receive_timeout_s: float = 30.0   # Timeout for receiving messages
    send_timeout_s: float = 10.0      # Timeout for sending messages

    # Keepalive
    ping_interval_s: float = 30.0     # Ping interval
    ping_timeout_s: float = 10.0      # Ping response timeout

    # Limits
    max_message_size_bytes: int = 1048576  # 1MB max message
    max_connections_per_client: int = 5
```

---

## Error Handling

### Error Taxonomy

```python
class ErrorCode(str, Enum):
    """Standardized error codes for Phase 7."""

    # Orchestrator errors (ORCH_xxx)
    ORCH_NOT_INITIALIZED = "ORCH_001"
    ORCH_VALIDATION_FAILED = "ORCH_002"
    ORCH_TIMEOUT = "ORCH_003"
    ORCH_INTERNAL = "ORCH_004"
    ORCH_SHUTDOWN = "ORCH_005"

    # Request errors (REQ_xxx)
    REQ_INVALID_FORMAT = "REQ_001"
    REQ_MISSING_FIELD = "REQ_002"
    REQ_INVALID_OPERATION = "REQ_003"
    REQ_PAYLOAD_MISMATCH = "REQ_004"

    # Agent errors (AGT_xxx)
    AGT_NOT_FOUND = "AGT_001"
    AGT_CREATION_FAILED = "AGT_002"
    AGT_EXECUTION_FAILED = "AGT_003"
    AGT_PAUSE_FAILED = "AGT_004"
    AGT_RESUME_FAILED = "AGT_005"

    # Context errors (CTX_xxx)
    CTX_NOT_INITIALIZED = "CTX_001"
    CTX_ASSEMBLY_FAILED = "CTX_002"
    CTX_BUDGET_EXCEEDED = "CTX_003"
    CTX_COMPRESSION_FAILED = "CTX_004"

    # Evolution errors (EVO_xxx)
    EVO_NOT_INITIALIZED = "EVO_001"
    EVO_EVALUATION_FAILED = "EVO_002"
    EVO_OPTIMIZATION_FAILED = "EVO_003"
    EVO_SAFETY_CONSTRAINT = "EVO_004"
    EVO_VERSION_NOT_FOUND = "EVO_005"
    EVO_ROLLBACK_FAILED = "EVO_006"

    # Memory errors (MEM_xxx)
    MEM_SEARCH_FAILED = "MEM_001"
    MEM_STORE_FAILED = "MEM_002"
    MEM_RETRIEVAL_FAILED = "MEM_003"

    # WebSocket errors (WS_xxx)
    WS_CONNECTION_FAILED = "WS_001"
    WS_AUTH_FAILED = "WS_002"
    WS_MESSAGE_INVALID = "WS_003"
    WS_BACKPRESSURE = "WS_004"

    # Rate limiting (RATE_xxx)
    RATE_LIMIT_EXCEEDED = "RATE_001"
    RATE_QUOTA_EXCEEDED = "RATE_002"


# ─────────────────────────────────────────────────────────────────────────────
# Exception Hierarchy
# ─────────────────────────────────────────────────────────────────────────────

class SigilError(Exception):
    """Base exception for all Sigil errors."""

    error_code: ErrorCode = ErrorCode.ORCH_INTERNAL
    status_code: int = 500
    recoverable: bool = False

    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def to_problem_detail(self, request_id: str) -> ProblemDetail:
        """Convert to RFC 9457 Problem Detail."""
        return ProblemDetail(
            type=f"https://sigil.acti.ai/errors/{self.error_code.value.lower().replace('_', '-')}",
            title=self.__class__.__name__,
            status=self.status_code,
            detail=self.message,
            instance=f"/requests/{request_id}",
            error_code=self.error_code.value,
            timestamp=datetime.utcnow(),
            request_id=request_id,
            recovery_hints=self._get_recovery_hints(),
        )

    def _get_recovery_hints(self) -> List[str]:
        """Get recovery hints for this error type."""
        return []


# Orchestrator Exceptions

class OrchestratorError(SigilError):
    """Base exception for orchestrator errors."""
    pass


class OrchestratorNotInitializedError(OrchestratorError):
    """Orchestrator not initialized."""
    error_code = ErrorCode.ORCH_NOT_INITIALIZED
    status_code = 503

    def _get_recovery_hints(self) -> List[str]:
        return ["Call orchestrator.initialize() before handling requests"]


class OrchestratorInitializationError(OrchestratorError):
    """Failed to initialize orchestrator."""
    error_code = ErrorCode.ORCH_INTERNAL
    status_code = 500


class OrchestratorShutdownError(OrchestratorError):
    """Error during orchestrator shutdown."""
    error_code = ErrorCode.ORCH_SHUTDOWN
    status_code = 500


# Request Exceptions

class RequestError(SigilError):
    """Base exception for request errors."""
    status_code = 400


class RequestValidationError(RequestError):
    """Request validation failed."""
    error_code = ErrorCode.REQ_INVALID_FORMAT

    def __init__(self, message: str, field: Optional[str] = None):
        super().__init__(message, {"field": field} if field else {})
        self.field = field

    def _get_recovery_hints(self) -> List[str]:
        hints = ["Verify request format matches API specification"]
        if self.field:
            hints.append(f"Check the '{self.field}' field")
        return hints


class RequestTimeoutError(RequestError):
    """Request timed out."""
    error_code = ErrorCode.ORCH_TIMEOUT
    status_code = 408
    recoverable = True

    def __init__(self, message: str, timeout_ms: int):
        super().__init__(message, {"timeout_ms": timeout_ms})
        self.timeout_ms = timeout_ms

    def _get_recovery_hints(self) -> List[str]:
        return [
            "Retry the request",
            "Consider increasing timeout_ms in request config",
            "Break down complex operations into smaller steps",
        ]


# Agent Exceptions

class AgentError(SigilError):
    """Base exception for agent errors."""
    pass


class AgentNotFoundError(AgentError):
    """Agent not found."""
    error_code = ErrorCode.AGT_NOT_FOUND
    status_code = 404

    def _get_recovery_hints(self) -> List[str]:
        return [
            "Verify agent name is correct",
            "Create the agent if it doesn't exist",
            "List available agents with GET /agents",
        ]


class AgentCreationError(AgentError):
    """Failed to create agent."""
    error_code = ErrorCode.AGT_CREATION_FAILED
    status_code = 400


class AgentExecutionError(AgentError):
    """Failed to execute agent."""
    error_code = ErrorCode.AGT_EXECUTION_FAILED
    status_code = 500
    recoverable = True


class AgentPauseError(AgentError):
    """Failed to pause agent."""
    error_code = ErrorCode.AGT_PAUSE_FAILED
    status_code = 400


class AgentResumeError(AgentError):
    """Failed to resume agent."""
    error_code = ErrorCode.AGT_RESUME_FAILED
    status_code = 400


# Context Exceptions

class ContextError(SigilError):
    """Base exception for context errors."""
    pass


class ContextManagerNotInitializedError(ContextError):
    """Context manager not initialized."""
    error_code = ErrorCode.CTX_NOT_INITIALIZED
    status_code = 503


class ContextAssemblyError(ContextError):
    """Failed to assemble context."""
    error_code = ErrorCode.CTX_ASSEMBLY_FAILED
    status_code = 500


class TokenBudgetExceededError(ContextError):
    """Token budget exceeded."""
    error_code = ErrorCode.CTX_BUDGET_EXCEEDED
    status_code = 400

    def __init__(
        self,
        message: str,
        actual_tokens: int,
        budget_tokens: int,
    ):
        super().__init__(message, {
            "actual_tokens": actual_tokens,
            "budget_tokens": budget_tokens,
        })
        self.actual_tokens = actual_tokens
        self.budget_tokens = budget_tokens

    def _get_recovery_hints(self) -> List[str]:
        return [
            "Reduce conversation history with history_turns config",
            "Reduce memory items with memory_retrieval_k config",
            "Enable compression in context manager",
        ]


class CompressionError(ContextError):
    """Context compression failed."""
    error_code = ErrorCode.CTX_COMPRESSION_FAILED
    status_code = 500


# Evolution Exceptions

class EvolutionError(SigilError):
    """Base exception for evolution errors."""
    pass


class EvolutionManagerNotInitializedError(EvolutionError):
    """Evolution manager not initialized."""
    error_code = ErrorCode.EVO_NOT_INITIALIZED
    status_code = 503


class EvaluationError(EvolutionError):
    """Evaluation failed."""
    error_code = ErrorCode.EVO_EVALUATION_FAILED
    status_code = 500


class InsufficientTestCasesError(EvolutionError):
    """Not enough test cases for evaluation."""
    error_code = ErrorCode.EVO_EVALUATION_FAILED
    status_code = 400


class OptimizationError(EvolutionError):
    """Optimization failed."""
    error_code = ErrorCode.EVO_OPTIMIZATION_FAILED
    status_code = 500


class SafetyConstraintError(EvolutionError):
    """Safety constraint violated."""
    error_code = ErrorCode.EVO_SAFETY_CONSTRAINT
    status_code = 400

    def _get_recovery_hints(self) -> List[str]:
        return [
            "Review optimization constraints in config",
            "Request explicit approval for major changes",
            "Consider smaller optimization iterations",
        ]


class MajorChangeRequiresApprovalError(SafetyConstraintError):
    """Major change requires explicit approval."""
    pass


class VersionNotFoundError(EvolutionError):
    """Version not found."""
    error_code = ErrorCode.EVO_VERSION_NOT_FOUND
    status_code = 404


class RollbackError(EvolutionError):
    """Rollback failed."""
    error_code = ErrorCode.EVO_ROLLBACK_FAILED
    status_code = 500


# Memory Exceptions

class MemoryError(SigilError):
    """Base exception for memory errors."""
    pass


class MemorySearchError(MemoryError):
    """Memory search failed."""
    error_code = ErrorCode.MEM_SEARCH_FAILED
    status_code = 500
    recoverable = True


class MemoryStoreError(MemoryError):
    """Memory store failed."""
    error_code = ErrorCode.MEM_STORE_FAILED
    status_code = 500
    recoverable = True


# WebSocket Exceptions

class WebSocketError(SigilError):
    """Base exception for WebSocket errors."""
    pass


class StreamingNotSupportedError(WebSocketError):
    """Operation doesn't support streaming."""
    error_code = ErrorCode.WS_MESSAGE_INVALID
    status_code = 400


# Rate Limiting Exceptions

class RateLimitError(SigilError):
    """Rate limit exceeded."""
    error_code = ErrorCode.RATE_LIMIT_EXCEEDED
    status_code = 429
    recoverable = True

    def __init__(
        self,
        message: str,
        retry_after_ms: int,
    ):
        super().__init__(message, {"retry_after_ms": retry_after_ms})
        self.retry_after_ms = retry_after_ms

    def _get_recovery_hints(self) -> List[str]:
        return [
            f"Wait {self.retry_after_ms}ms before retrying",
            "Reduce request frequency",
            "Consider upgrading rate limit tier",
        ]
```

---

## Event Contracts

### Event Definitions

```python
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional
from enum import Enum


class EventType(str, Enum):
    """Phase 7 event types."""

    # Orchestrator events
    ORCHESTRATOR_INITIALIZED = "orchestrator.initialized"
    ORCHESTRATOR_SHUTDOWN = "orchestrator.shutdown"
    REQUEST_STARTED = "request.started"
    REQUEST_COMPLETED = "request.completed"
    REQUEST_FAILED = "request.failed"

    # Agent events
    AGENT_CREATED = "agent.created"
    AGENT_RUN_STARTED = "agent.run.started"
    AGENT_RUN_COMPLETED = "agent.run.completed"
    AGENT_PAUSED = "agent.paused"
    AGENT_RESUMED = "agent.resumed"

    # Context events
    CONTEXT_ASSEMBLED = "context.assembled"
    CONTEXT_COMPRESSED = "context.compressed"
    BUDGET_WARNING = "budget.warning"
    BUDGET_EXCEEDED = "budget.exceeded"

    # Evolution events
    AGENT_EVALUATED = "agent.evaluated"
    AGENT_OPTIMIZED = "agent.optimized"
    AGENT_ROLLED_BACK = "agent.rolled_back"
    VERSION_CREATED = "version.created"


@dataclass
class BaseEvent:
    """Base class for all events."""

    event_id: str                     # Unique event ID
    event_type: EventType             # Event type
    timestamp: datetime               # When event occurred
    source: str                       # Component that emitted event

    # Correlation
    request_id: Optional[str] = None
    session_id: Optional[str] = None
    trace_id: Optional[str] = None

    # Metadata
    metadata: Dict[str, Any] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "timestamp": self.timestamp.isoformat(),
            "source": self.source,
            "request_id": self.request_id,
            "session_id": self.session_id,
            "trace_id": self.trace_id,
            "metadata": self.metadata or {},
        }


# Orchestrator Events

@dataclass
class OrchestratorInitializedEvent(BaseEvent):
    """Emitted when orchestrator is initialized."""

    event_type: EventType = EventType.ORCHESTRATOR_INITIALIZED
    source: str = "orchestrator"

    orchestrator_id: str = ""
    version: str = ""
    subsystems_initialized: List[str] = None


@dataclass
class OrchestratorShutdownEvent(BaseEvent):
    """Emitted when orchestrator shuts down."""

    event_type: EventType = EventType.ORCHESTRATOR_SHUTDOWN
    source: str = "orchestrator"

    orchestrator_id: str = ""
    active_requests_cancelled: int = 0


@dataclass
class RequestStartedEvent(BaseEvent):
    """Emitted when a request starts processing."""

    event_type: EventType = EventType.REQUEST_STARTED
    source: str = "orchestrator"

    operation: str = ""
    priority: str = "normal"


@dataclass
class RequestCompletedEvent(BaseEvent):
    """Emitted when a request completes."""

    event_type: EventType = EventType.REQUEST_COMPLETED
    source: str = "orchestrator"

    operation: str = ""
    success: bool = True
    tokens_used: int = 0
    latency_ms: int = 0


# Agent Events

@dataclass
class AgentCreatedEvent(BaseEvent):
    """Emitted when an agent is created."""

    event_type: EventType = EventType.AGENT_CREATED
    source: str = "orchestrator"

    agent_name: str = ""
    stratum: str = ""
    tools: List[str] = None


@dataclass
class AgentRunStartedEvent(BaseEvent):
    """Emitted when agent run starts."""

    event_type: EventType = EventType.AGENT_RUN_STARTED
    source: str = "orchestrator"

    agent_name: str = ""
    input_preview: str = ""          # First 100 chars of input


@dataclass
class AgentRunCompletedEvent(BaseEvent):
    """Emitted when agent run completes."""

    event_type: EventType = EventType.AGENT_RUN_COMPLETED
    source: str = "orchestrator"

    agent_name: str = ""
    success: bool = True
    tokens_used: int = 0
    tool_calls: int = 0
    latency_ms: int = 0


# Context Events

@dataclass
class ContextAssembledEvent(BaseEvent):
    """Emitted when context is assembled."""

    event_type: EventType = EventType.CONTEXT_ASSEMBLED
    source: str = "context_manager"

    total_tokens: int = 0
    components: Dict[str, int] = None  # Component -> token count
    compression_applied: bool = False


@dataclass
class BudgetWarningEvent(BaseEvent):
    """Emitted when budget utilization is high."""

    event_type: EventType = EventType.BUDGET_WARNING
    source: str = "context_manager"

    utilization_percent: float = 0.0
    remaining_tokens: int = 0
    threshold: float = 0.8


# Evolution Events

@dataclass
class AgentEvaluatedEvent(BaseEvent):
    """Emitted when agent evaluation completes."""

    event_type: EventType = EventType.AGENT_EVALUATED
    source: str = "evolution_manager"

    agent_name: str = ""
    overall_score: float = 0.0
    test_cases_count: int = 0
    passed_count: int = 0
    failed_count: int = 0


@dataclass
class AgentOptimizedEvent(BaseEvent):
    """Emitted when agent optimization completes."""

    event_type: EventType = EventType.AGENT_OPTIMIZED
    source: str = "evolution_manager"

    agent_name: str = ""
    baseline_score: float = 0.0
    new_score: float = 0.0
    improvement_percent: float = 0.0
    new_version: Optional[str] = None


@dataclass
class AgentRolledBackEvent(BaseEvent):
    """Emitted when agent is rolled back."""

    event_type: EventType = EventType.AGENT_ROLLED_BACK
    source: str = "evolution_manager"

    agent_name: str = ""
    from_version: str = ""
    to_version: str = ""


# Event Factory

def create_event(
    event_type: EventType,
    request_id: Optional[str] = None,
    session_id: Optional[str] = None,
    **kwargs,
) -> BaseEvent:
    """
    Factory function to create events.

    Args:
        event_type: Type of event to create
        request_id: Optional request correlation ID
        session_id: Optional session correlation ID
        **kwargs: Event-specific fields

    Returns:
        Configured event instance
    """
    event_classes = {
        EventType.ORCHESTRATOR_INITIALIZED: OrchestratorInitializedEvent,
        EventType.ORCHESTRATOR_SHUTDOWN: OrchestratorShutdownEvent,
        EventType.REQUEST_STARTED: RequestStartedEvent,
        EventType.REQUEST_COMPLETED: RequestCompletedEvent,
        EventType.AGENT_CREATED: AgentCreatedEvent,
        EventType.AGENT_RUN_STARTED: AgentRunStartedEvent,
        EventType.AGENT_RUN_COMPLETED: AgentRunCompletedEvent,
        EventType.CONTEXT_ASSEMBLED: ContextAssembledEvent,
        EventType.BUDGET_WARNING: BudgetWarningEvent,
        EventType.AGENT_EVALUATED: AgentEvaluatedEvent,
        EventType.AGENT_OPTIMIZED: AgentOptimizedEvent,
        EventType.AGENT_ROLLED_BACK: AgentRolledBackEvent,
    }

    event_class = event_classes.get(event_type)
    if not event_class:
        raise ValueError(f"Unknown event type: {event_type}")

    return event_class(
        event_id=str(uuid4()),
        timestamp=datetime.utcnow(),
        request_id=request_id,
        session_id=session_id,
        **kwargs,
    )
```

---

## Integration Patterns

### Phase 3-6 Integration

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     PHASE 7 INTEGRATION WITH SUBSYSTEMS                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   Request                                                                    │
│      │                                                                       │
│      ▼                                                                       │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                      SigilOrchestrator                                 │  │
│  │                                                                        │  │
│  │  1. Validate request                                                   │  │
│  │  2. Create request context                                             │  │
│  │  3. Route to subsystem                                                 │  │
│  │                                                                        │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                    │                                         │
│         ┌──────────────────────────┼──────────────────────────┐             │
│         │                          │                          │             │
│         ▼                          ▼                          ▼             │
│  ┌─────────────┐           ┌─────────────┐           ┌─────────────┐       │
│  │   Phase 3   │           │   Phase 4   │           │   Phase 5   │       │
│  │   Router    │           │   Memory    │           │  Planning   │       │
│  │             │           │             │           │             │       │
│  │ - Classify  │           │ - Retrieve  │           │ - Generate  │       │
│  │ - Assess    │           │ - Store     │           │ - Execute   │       │
│  │ - Route     │           │ - Extract   │           │ - Validate  │       │
│  └─────────────┘           └─────────────┘           └─────────────┘       │
│         │                          │                          │             │
│         └──────────────────────────┼──────────────────────────┘             │
│                                    ▼                                         │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                         Phase 6: Contracts                             │  │
│  │                                                                        │  │
│  │  - Validate outputs against contracts                                  │  │
│  │  - Retry on validation failure                                         │  │
│  │  - Apply fallback strategies                                           │  │
│  │                                                                        │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                    │                                         │
│                                    ▼                                         │
│                              Response                                        │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Request Flow

```python
async def _handle_run(
    self,
    request: OrchestratorRequest,
    context: RequestContext,
) -> RunAgentResult:
    """
    Handle agent run request.

    Flow:
    1. Load agent configuration
    2. Assemble context via ContextManager
    3. Route via Phase 3 Router
    4. Execute via Planning/Reasoning
    5. Validate via Contracts
    6. Return result
    """
    payload: RunAgentPayload = request.payload

    # 1. Load agent
    agent = await self._load_agent(payload.agent_name)

    # 2. Assemble context
    memory_items = []
    if context.config.include_memory:
        memory_items = await self._memory_manager.retrieve(
            query=payload.input,
            agent_name=payload.agent_name,
            k=context.config.memory_retrieval_k,
        )

    assembled_context = await self._context_manager.assemble(
        agent_config=agent.config,
        session=context.session,
        memory_items=memory_items,
        tools=agent.tools,
        user_input=payload.input,
    )

    # Track context tokens
    context.tokens_used += TokenUsage(
        memory_tokens=assembled_context.memory_context.token_count if assembled_context.memory_context else 0,
        history_tokens=assembled_context.history_context.token_count if assembled_context.history_context else 0,
        system_tokens=assembled_context.system_context.token_count if assembled_context.system_context else 0,
    )

    # 3. Route request
    route_decision = await self._router.route(
        input=payload.input,
        agent=agent,
        context=assembled_context,
    )

    # 4. Execute based on route decision
    if route_decision.use_planning and self._config.enable_planning:
        result = await self._execute_with_planning(
            agent=agent,
            input=payload.input,
            context=assembled_context,
            route_decision=route_decision,
            request_context=context,
        )
    else:
        result = await self._execute_with_reasoning(
            agent=agent,
            input=payload.input,
            context=assembled_context,
            route_decision=route_decision,
            request_context=context,
        )

    # 5. Validate with contracts
    if context.config.enforce_contracts and agent.contract:
        validation = await self._contract_executor.validate(
            output=result,
            contract=agent.contract,
        )

        if not validation.passed:
            # Retry or apply fallback
            result = await self._handle_contract_failure(
                result=result,
                validation=validation,
                agent=agent,
                context=context,
            )

    # 6. Store in memory
    await self._memory_manager.store_resource(
        content=json.dumps({
            "input": payload.input,
            "output": result.response,
            "tokens": context.tokens_used.total_tokens,
        }),
        agent_name=payload.agent_name,
        session_id=context.session.session_id if context.session else None,
        resource_type="conversation",
    )

    return result
```

### Subsystem Interfaces

```python
# Interface with Phase 3 Router
class RouterInterface:
    """Interface for Phase 3 Router integration."""

    async def route(
        self,
        input: str,
        agent: Agent,
        context: AssembledContext,
    ) -> RouteDecision:
        """
        Route a request to appropriate handler.

        Returns:
            RouteDecision with complexity, strategy, and flags
        """
        pass


# Interface with Phase 4 Memory
class MemoryInterface:
    """Interface for Phase 4 Memory integration."""

    async def retrieve(
        self,
        query: str,
        agent_name: str,
        k: int = 10,
        method: str = "hybrid",
    ) -> List[MemoryItem]:
        """Retrieve relevant memory items."""
        pass

    async def store_resource(
        self,
        content: str,
        agent_name: str,
        session_id: Optional[str],
        resource_type: str,
    ) -> str:
        """Store a resource in memory."""
        pass


# Interface with Phase 5 Planning
class PlanningInterface:
    """Interface for Phase 5 Planning integration."""

    async def generate_plan(
        self,
        goal: str,
        context: AssembledContext,
    ) -> Plan:
        """Generate execution plan."""
        pass

    async def execute_plan(
        self,
        plan: Plan,
        session_id: str,
        budget: TokenBudget,
    ) -> ExecutionResult:
        """Execute plan."""
        pass


# Interface with Phase 5 Reasoning
class ReasoningInterface:
    """Interface for Phase 5 Reasoning integration."""

    async def reason(
        self,
        task: str,
        complexity: float,
        context: ReasoningContext,
    ) -> StrategyResult:
        """Execute reasoning task."""
        pass


# Interface with Phase 6 Contracts
class ContractInterface:
    """Interface for Phase 6 Contracts integration."""

    async def validate(
        self,
        output: Any,
        contract: Contract,
    ) -> ValidationResult:
        """Validate output against contract."""
        pass

    async def execute_with_contract(
        self,
        executor: Callable,
        contract: Contract,
        context: ExecutionContext,
    ) -> ContractResult:
        """Execute with contract enforcement."""
        pass
```

---

## Performance Contracts

### Latency Targets

| Operation | Target (P50) | Target (P95) | Max |
|-----------|--------------|--------------|-----|
| Request validation | 5ms | 10ms | 50ms |
| Context assembly | 50ms | 100ms | 500ms |
| Context compression | 100ms | 200ms | 1000ms |
| Route decision | 100ms | 200ms | 500ms |
| Agent run (simple) | 2s | 5s | 30s |
| Agent run (complex) | 10s | 20s | 60s |
| Memory search | 100ms | 300ms | 1000ms |
| Memory store | 50ms | 100ms | 500ms |
| Evaluation (per case) | 5s | 10s | 30s |
| Optimization | 30s | 60s | 300s |

### Throughput Targets

| Metric | Target | Notes |
|--------|--------|-------|
| Concurrent requests | 100 | Per orchestrator instance |
| Requests per second | 50 | Simple requests |
| Requests per second | 10 | Complex requests (planning) |
| WebSocket connections | 1000 | Per instance |
| Events per second | 10000 | WebSocket streaming |

### Token Budget Enforcement

```python
@dataclass
class TokenBudgetEnforcement:
    """Token budget enforcement rules."""

    # Hard limits
    max_input_per_request: int = 100000
    max_output_per_request: int = 50000
    max_total_per_session: int = 1000000

    # Soft limits (warnings)
    warning_threshold: float = 0.8
    critical_threshold: float = 0.95

    # Enforcement actions
    actions: Dict[str, str] = field(default_factory=lambda: {
        "warning": "emit_warning_event",
        "critical": "enable_aggressive_compression",
        "exceeded": "reject_request",
    })


class TokenBudgetEnforcer:
    """
    Enforces token budgets across the orchestrator.

    Actions:
    - warning (80%): Emit warning event, suggest compression
    - critical (95%): Enable aggressive compression automatically
    - exceeded (100%): Reject new requests
    """

    def __init__(self, config: TokenBudgetEnforcement):
        self._config = config
        self._session_usage: Dict[str, int] = {}

    def check_budget(
        self,
        session_id: str,
        requested_tokens: int,
    ) -> BudgetCheckResult:
        """
        Check if request fits within budget.

        Returns:
            BudgetCheckResult with status and recommendations
        """
        current_usage = self._session_usage.get(session_id, 0)
        projected_usage = current_usage + requested_tokens
        max_budget = self._config.max_total_per_session

        utilization = projected_usage / max_budget

        if utilization > 1.0:
            return BudgetCheckResult(
                allowed=False,
                status="exceeded",
                utilization=utilization,
                action="reject_request",
                message=f"Budget exceeded: {projected_usage}/{max_budget} tokens",
            )

        if utilization > self._config.critical_threshold:
            return BudgetCheckResult(
                allowed=True,
                status="critical",
                utilization=utilization,
                action="enable_aggressive_compression",
                message="Critical budget threshold reached",
            )

        if utilization > self._config.warning_threshold:
            return BudgetCheckResult(
                allowed=True,
                status="warning",
                utilization=utilization,
                action="emit_warning_event",
                message="Budget warning threshold reached",
            )

        return BudgetCheckResult(
            allowed=True,
            status="ok",
            utilization=utilization,
            action=None,
            message=None,
        )


@dataclass
class BudgetCheckResult:
    """Result of budget check."""

    allowed: bool
    status: str                       # "ok", "warning", "critical", "exceeded"
    utilization: float
    action: Optional[str]
    message: Optional[str]
```

### Rate Limiting

```python
@dataclass
class RateLimitConfig:
    """Rate limiting configuration."""

    # Request limits
    requests_per_second: int = 10
    requests_per_minute: int = 100
    requests_per_hour: int = 1000

    # Token limits
    tokens_per_minute: int = 100000
    tokens_per_hour: int = 1000000

    # Burst allowance
    burst_multiplier: float = 2.0     # 2x for burst

    # Per-client limits
    per_client_multiplier: float = 1.0


class RateLimiter:
    """
    Rate limiter using token bucket algorithm.

    Implements:
    - Request rate limiting
    - Token rate limiting
    - Burst handling
    - Per-client tracking
    """

    def __init__(self, config: RateLimitConfig):
        self._config = config
        self._buckets: Dict[str, TokenBucket] = {}

    async def check(
        self,
        client_id: str,
        tokens_requested: int = 0,
    ) -> RateLimitResult:
        """
        Check if request is within rate limits.

        Args:
            client_id: Client identifier
            tokens_requested: Tokens for this request

        Returns:
            RateLimitResult with status and retry info
        """
        bucket = self._get_or_create_bucket(client_id)

        # Check request rate
        if not bucket.consume_request():
            return RateLimitResult(
                allowed=False,
                reason="request_rate_exceeded",
                retry_after_ms=bucket.time_until_request_available(),
            )

        # Check token rate
        if tokens_requested > 0 and not bucket.consume_tokens(tokens_requested):
            return RateLimitResult(
                allowed=False,
                reason="token_rate_exceeded",
                retry_after_ms=bucket.time_until_tokens_available(tokens_requested),
            )

        return RateLimitResult(
            allowed=True,
            reason=None,
            retry_after_ms=None,
            remaining_requests=bucket.remaining_requests(),
            remaining_tokens=bucket.remaining_tokens(),
        )


@dataclass
class RateLimitResult:
    """Result of rate limit check."""

    allowed: bool
    reason: Optional[str]
    retry_after_ms: Optional[int]

    remaining_requests: Optional[int] = None
    remaining_tokens: Optional[int] = None
```

---

## Appendix: Payload Schemas

### Agent Payloads

```python
@dataclass
class CreateAgentPayload:
    """Payload for agent creation."""

    name: str                         # Unique agent name
    stratum: str                      # ACTi stratum (RTI, RAI, ZACS, EEI, IGE)
    description: str                  # Human-readable description

    # Configuration
    system_prompt: Optional[str] = None
    tools: List[str] = field(default_factory=list)
    contract_name: Optional[str] = None

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RunAgentPayload:
    """Payload for running an agent."""

    agent_name: str                   # Agent to run
    input: str                        # User input

    # Session
    session_id: Optional[str] = None  # For conversation continuity

    # Context overrides
    memory_items: Optional[List[str]] = None  # Specific memory IDs
    tools_override: Optional[List[str]] = None


@dataclass
class PausePayload:
    """Payload for pausing agent execution."""

    agent_name: str
    session_id: str


@dataclass
class ResumePayload:
    """Payload for resuming agent execution."""

    agent_name: str
    session_id: str
    checkpoint_id: Optional[str] = None


@dataclass
class StatusPayload:
    """Payload for agent status query."""

    agent_name: str


### Memory Payloads

@dataclass
class MemorySearchPayload:
    """Payload for memory search."""

    query: str                        # Search query
    agent_name: Optional[str] = None  # Scope to agent
    session_id: Optional[str] = None  # Scope to session
    k: int = 10                       # Number of results
    retrieval_method: str = "hybrid"  # "rag", "llm", "hybrid"


@dataclass
class MemoryStorePayload:
    """Payload for storing memory."""

    content: str                      # Content to store
    agent_name: str                   # Associated agent

    session_id: Optional[str] = None
    category: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


### Evolution Payloads

@dataclass
class EvaluatePayload:
    """Payload for agent evaluation."""

    agent_name: str
    test_cases: Optional[List[TestCase]] = None
    config: Optional[EvaluationConfig] = None


@dataclass
class OptimizePayload:
    """Payload for agent optimization."""

    agent_name: str
    target: str = "overall"           # "overall", "accuracy", "speed", "cost"
    config: Optional[OptimizationConfig] = None


@dataclass
class ComparePayload:
    """Payload for version comparison."""

    agent_name: str
    version_a: str
    version_b: str
    test_cases: Optional[List[TestCase]] = None
```

---

## Document Revision History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0.0 | 2026-01-11 | API Architecture Team | Initial Phase 7 API contract |

---

*Document Version: 1.0.0*
*Last Updated: 2026-01-11*
