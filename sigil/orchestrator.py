"""SigilOrchestrator - Central coordinator for all Sigil v2 subsystems.

This module implements the SigilOrchestrator, which coordinates the execution
pipeline through all subsystems: Router -> Planning -> Memory -> Reasoning -> Contracts.

The orchestrator provides a unified request/response format and handles the
complete lifecycle of agent execution with proper error handling and telemetry.

Key Components:
    - OrchestratorRequest: Unified input format for all requests
    - OrchestratorResponse: Unified output format with metrics
    - SigilOrchestrator: Main coordinator class

Pipeline Flow:
    1. Route: Classify intent and assess complexity
    2. Plan: Generate execution plan if complex
    3. Context: Assemble context from memory, plan, and history
    4. Execute: Run through reasoning strategy
    5. Validate: Verify output against contract (if applicable)
    6. Return: Unified response with metrics

Example:
    >>> from sigil.orchestrator import SigilOrchestrator, OrchestratorRequest
    >>>
    >>> orchestrator = SigilOrchestrator()
    >>> request = OrchestratorRequest(
    ...     message="Create an agent that can search the web",
    ...     session_id="sess-123",
    ...     user_id="user-456",
    ... )
    >>> response = await orchestrator.process(request)
    >>> print(response.output)
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional, Callable, Awaitable

from sigil.config import get_settings
from sigil.config.settings import SigilSettings
from sigil.core.exceptions import (
    SigilError,
    RoutingError,
    ReasoningError,
    ContractValidationError,
    PlanExecutionError,
)
from sigil.routing.router import Router, RouteDecision, Intent
from sigil.memory.manager import MemoryManager
from sigil.planning.planner import Planner
from sigil.planning.executor import PlanExecutor
from sigil.planning.tool_executor import create_tool_step_executor
from sigil.reasoning.manager import ReasoningManager
from sigil.contracts.executor import ContractExecutor, ContractResult
from sigil.contracts.schema import Contract
from sigil.contracts.templates.acti import get_contract_for_intent
from sigil.state.events import Event, EventType, _generate_event_id, _get_utc_now
from sigil.state.store import EventStore
from sigil.telemetry.tokens import TokenTracker, TokenBudget

# Tool registry for getting available tools
try:
    from src.tool_registry import get_configured_tools
except ImportError:
    # Fallback if tool_registry not available
    def get_configured_tools() -> list[str]:
        return []


# =============================================================================
# Module Logger
# =============================================================================

logger = logging.getLogger(__name__)


# =============================================================================
# Constants
# =============================================================================

# Default token budget for 256K model (150K input, 102.4K output)
DEFAULT_INPUT_BUDGET = 150_000
DEFAULT_OUTPUT_BUDGET = 102_400
DEFAULT_TOTAL_BUDGET = 256_000


# =============================================================================
# Orchestrator Status
# =============================================================================


class OrchestratorStatus(str, Enum):
    """Status of an orchestrator request.

    Attributes:
        SUCCESS: Request completed successfully
        PARTIAL: Request completed with some failures
        FAILED: Request failed completely
        TIMEOUT: Request exceeded time limit
        CANCELLED: Request was cancelled
    """
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


# =============================================================================
# Orchestrator Request
# =============================================================================


@dataclass
class OrchestratorRequest:
    """Unified input format for orchestrator requests.

    Attributes:
        message: The user message to process
        session_id: Unique session identifier
        user_id: Optional user identifier for personalization
        agent_name: Optional specific agent to use
        context: Optional additional context dictionary
        contract_name: Optional specific contract to enforce
        force_strategy: Optional reasoning strategy override
        max_tokens: Optional token budget override
        timeout_seconds: Optional timeout override
        correlation_id: Optional correlation ID for tracing
        metadata: Optional additional metadata

    Example:
        >>> request = OrchestratorRequest(
        ...     message="Qualify lead John from Acme Corp",
        ...     session_id="sess-123",
        ...     user_id="user-456",
        ...     contract_name="lead_qualification",
        ... )
    """
    message: str
    session_id: str
    user_id: Optional[str] = None
    agent_name: Optional[str] = None
    context: dict[str, Any] = field(default_factory=dict)
    contract_name: Optional[str] = None
    force_strategy: Optional[str] = None
    max_tokens: Optional[int] = None
    timeout_seconds: Optional[float] = None
    correlation_id: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Generate correlation ID if not provided."""
        if not self.correlation_id:
            self.correlation_id = str(uuid.uuid4())


@dataclass
class OrchestratorResponse:
    """Unified output format for orchestrator responses.

    Attributes:
        request_id: Unique identifier for this request
        status: Overall status of the request
        output: The main output from processing
        route_decision: The routing decision made
        plan_id: Optional plan ID if planning was used
        contract_result: Optional contract validation result
        tokens_used: Total tokens consumed
        execution_time_ms: Total execution time in milliseconds
        errors: List of any errors encountered
        warnings: List of any warnings
        metadata: Additional response metadata

    Example:
        >>> response.status
        <OrchestratorStatus.SUCCESS: 'success'>
        >>> response.output
        {'result': 'Lead qualified with score 85'}
    """
    request_id: str
    status: OrchestratorStatus
    output: dict[str, Any]
    route_decision: Optional[RouteDecision] = None
    plan_id: Optional[str] = None
    contract_result: Optional[ContractResult] = None
    tokens_used: int = 0
    execution_time_ms: float = 0.0
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def success(self) -> bool:
        """Check if the request was successful."""
        return self.status == OrchestratorStatus.SUCCESS

    @property
    def has_errors(self) -> bool:
        """Check if there were any errors."""
        return len(self.errors) > 0

    def to_dict(self) -> dict[str, Any]:
        """Convert response to dictionary."""
        return {
            "request_id": self.request_id,
            "status": self.status.value,
            "output": self.output,
            "route_decision": self.route_decision.to_dict() if self.route_decision else None,
            "plan_id": self.plan_id,
            "contract_result": self.contract_result.to_dict() if self.contract_result else None,
            "tokens_used": self.tokens_used,
            "execution_time_ms": self.execution_time_ms,
            "errors": self.errors,
            "warnings": self.warnings,
            "metadata": self.metadata,
        }


# =============================================================================
# Event Creators
# =============================================================================


def create_orchestrator_started_event(
    session_id: str,
    request_id: str,
    message_preview: str,
    correlation_id: Optional[str] = None,
) -> Event:
    """Create an OrchestratorStartedEvent."""
    payload = {
        "request_id": request_id,
        "message_preview": message_preview[:100],
        "started_at": _get_utc_now().isoformat(),
    }
    return Event(
        event_id=_generate_event_id(),
        event_type=EventType.MESSAGE_ADDED,  # Using MESSAGE_ADDED for now
        timestamp=_get_utc_now(),
        session_id=session_id,
        correlation_id=correlation_id,
        payload=payload,
    )


def create_orchestrator_completed_event(
    session_id: str,
    request_id: str,
    status: OrchestratorStatus,
    tokens_used: int,
    execution_time_ms: float,
    error_count: int,
    correlation_id: Optional[str] = None,
) -> Event:
    """Create an OrchestratorCompletedEvent."""
    payload = {
        "request_id": request_id,
        "status": status.value,
        "tokens_used": tokens_used,
        "execution_time_ms": execution_time_ms,
        "error_count": error_count,
        "completed_at": _get_utc_now().isoformat(),
    }
    return Event(
        event_id=_generate_event_id(),
        event_type=EventType.PLAN_COMPLETED,  # Reusing for completion tracking
        timestamp=_get_utc_now(),
        session_id=session_id,
        correlation_id=correlation_id,
        payload=payload,
    )


# =============================================================================
# Pipeline Step Protocol
# =============================================================================


class PipelineContext:
    """Context passed through the orchestrator pipeline.

    Attributes:
        request: The original request
        route_decision: Routing decision
        plan: Generated plan (if any)
        assembled_context: Assembled context from memory
        reasoning_result: Result from reasoning
        contract_result: Contract validation result
        tokens_used: Running token count
        errors: Accumulated errors
        warnings: Accumulated warnings
    """

    def __init__(self, request: OrchestratorRequest) -> None:
        self.request = request
        self.route_decision: Optional[RouteDecision] = None
        self.plan: Optional[Any] = None
        self.plan_id: Optional[str] = None
        self.assembled_context: dict[str, Any] = {}
        self.reasoning_result: Optional[Any] = None
        self.contract_result: Optional[ContractResult] = None
        self.output: dict[str, Any] = {}
        self.tokens_used: int = 0
        self.errors: list[str] = []
        self.warnings: list[str] = []
        self.start_time: float = time.perf_counter()

    def add_tokens(self, count: int) -> None:
        """Add to token count."""
        self.tokens_used += count

    def add_error(self, error: str) -> None:
        """Add an error message."""
        self.errors.append(error)
        logger.error(f"Pipeline error: {error}")

    def add_warning(self, warning: str) -> None:
        """Add a warning message."""
        self.warnings.append(warning)
        logger.warning(f"Pipeline warning: {warning}")

    @property
    def elapsed_ms(self) -> float:
        """Get elapsed time in milliseconds."""
        return (time.perf_counter() - self.start_time) * 1000


# =============================================================================
# SigilOrchestrator
# =============================================================================


class SigilOrchestrator:
    """Central coordinator for all Sigil v2 subsystems.

    The orchestrator manages the complete lifecycle of request processing:
    1. Route: Classify intent and determine complexity
    2. Plan: Generate execution plan if needed
    3. Context: Assemble context from memory/history
    4. Execute: Run through appropriate reasoning strategy
    5. Validate: Check output against contract
    6. Return: Unified response with metrics

    Features:
        - Unified request/response format
        - Automatic subsystem integration
        - Token budget management
        - Error recovery and fallbacks
        - Event emission for audit trails
        - Metrics collection

    Attributes:
        settings: Framework settings
        router: Intent-based router
        memory_manager: Memory system manager
        planner: Task decomposition planner
        plan_executor: Plan execution engine
        reasoning_manager: Reasoning strategy manager
        contract_executor: Contract enforcement engine
        event_store: Event store for audit trails
        token_tracker: Token usage tracker

    Example:
        >>> orchestrator = SigilOrchestrator()
        >>> request = OrchestratorRequest(
        ...     message="Create a sales agent",
        ...     session_id="sess-123",
        ... )
        >>> response = await orchestrator.process(request)
    """

    def __init__(
        self,
        settings: Optional[SigilSettings] = None,
        router: Optional[Router] = None,
        memory_manager: Optional[MemoryManager] = None,
        planner: Optional[Planner] = None,
        plan_executor: Optional[PlanExecutor] = None,
        reasoning_manager: Optional[ReasoningManager] = None,
        contract_executor: Optional[ContractExecutor] = None,
        event_store: Optional[EventStore] = None,
        token_tracker: Optional[TokenTracker] = None,
    ) -> None:
        """Initialize the orchestrator.

        Args:
            settings: Optional custom settings
            router: Optional custom router
            memory_manager: Optional custom memory manager
            planner: Optional custom planner
            plan_executor: Optional custom plan executor
            reasoning_manager: Optional custom reasoning manager
            contract_executor: Optional custom contract executor
            event_store: Optional custom event store
            token_tracker: Optional custom token tracker
        """
        self._settings = settings or get_settings()
        self._event_store = event_store or EventStore()
        self._token_tracker = token_tracker or TokenTracker()

        # Initialize subsystems based on feature flags
        self._router = router or Router(self._settings)

        if self._settings.use_memory:
            self._memory_manager = memory_manager or MemoryManager(
                event_store=self._event_store,
            )
        else:
            self._memory_manager = memory_manager

        if self._settings.use_planning:
            self._planner = planner or Planner(event_store=self._event_store)
            self._plan_executor = plan_executor or PlanExecutor(
                event_store=self._event_store,
                token_tracker=self._token_tracker,
            )
        else:
            self._planner = planner
            self._plan_executor = plan_executor

        self._reasoning_manager = reasoning_manager or ReasoningManager(
            event_store=self._event_store,
            token_tracker=self._token_tracker,
        )

        if self._settings.use_contracts:
            self._contract_executor = contract_executor or ContractExecutor(
                event_store=self._event_store,
            )
        else:
            self._contract_executor = contract_executor

        # Request metrics
        self._total_requests: int = 0
        self._successful_requests: int = 0
        self._failed_requests: int = 0
        self._total_tokens: int = 0
        self._total_time_ms: float = 0.0

        logger.info(
            f"SigilOrchestrator initialized with features: "
            f"{self._settings.get_active_features()}"
        )

    async def process(
        self,
        request: OrchestratorRequest,
    ) -> OrchestratorResponse:
        """Process a request through the orchestration pipeline.

        This is the main entry point for all orchestrated execution.

        Args:
            request: The orchestrator request

        Returns:
            OrchestratorResponse with results and metrics

        Example:
            >>> response = await orchestrator.process(request)
            >>> if response.success:
            ...     print(response.output)
        """
        request_id = str(uuid.uuid4())
        ctx = PipelineContext(request)

        # Emit started event
        self._event_store.append(
            create_orchestrator_started_event(
                session_id=request.session_id,
                request_id=request_id,
                message_preview=request.message,
                correlation_id=request.correlation_id,
            )
        )

        self._total_requests += 1

        try:
            # Execute pipeline steps
            await self._step_route(ctx)
            await self._step_plan(ctx)
            await self._step_assemble_context(ctx)
            await self._step_execute(ctx)
            await self._step_validate(ctx)

            # Determine final status
            if ctx.errors:
                status = OrchestratorStatus.PARTIAL if ctx.output else OrchestratorStatus.FAILED
            else:
                status = OrchestratorStatus.SUCCESS

            if status == OrchestratorStatus.SUCCESS:
                self._successful_requests += 1
            else:
                self._failed_requests += 1

        except asyncio.TimeoutError:
            ctx.add_error("Request timed out")
            status = OrchestratorStatus.TIMEOUT
            self._failed_requests += 1

        except asyncio.CancelledError:
            ctx.add_error("Request was cancelled")
            status = OrchestratorStatus.CANCELLED
            self._failed_requests += 1

        except SigilError as e:
            ctx.add_error(str(e))
            status = OrchestratorStatus.FAILED
            self._failed_requests += 1

        except Exception as e:
            ctx.add_error(f"Unexpected error: {str(e)}")
            status = OrchestratorStatus.FAILED
            self._failed_requests += 1
            logger.exception("Unexpected error in orchestrator")

        # Update metrics
        self._total_tokens += ctx.tokens_used
        self._total_time_ms += ctx.elapsed_ms

        # Build response
        response = OrchestratorResponse(
            request_id=request_id,
            status=status,
            output=ctx.output,
            route_decision=ctx.route_decision,
            plan_id=ctx.plan_id,
            contract_result=ctx.contract_result,
            tokens_used=ctx.tokens_used,
            execution_time_ms=ctx.elapsed_ms,
            errors=ctx.errors,
            warnings=ctx.warnings,
            metadata={
                "correlation_id": request.correlation_id,
                "session_id": request.session_id,
                "user_id": request.user_id,
            },
        )

        # Emit completed event
        self._event_store.append(
            create_orchestrator_completed_event(
                session_id=request.session_id,
                request_id=request_id,
                status=status,
                tokens_used=ctx.tokens_used,
                execution_time_ms=ctx.elapsed_ms,
                error_count=len(ctx.errors),
                correlation_id=request.correlation_id,
            )
        )

        return response

    async def _step_route(self, ctx: PipelineContext) -> None:
        """Step 1: Route the request to determine intent and complexity.

        Args:
            ctx: Pipeline context
        """
        if not self._settings.use_routing:
            # Create default route decision
            ctx.route_decision = RouteDecision(
                intent=Intent.GENERAL_CHAT,
                confidence=1.0,
                complexity=0.5,
                handler_name="default",
                use_planning=self._settings.use_planning,
                use_memory=self._settings.use_memory,
                use_contracts=self._settings.use_contracts,
            )
            return

        try:
            ctx.route_decision = self._router.route(ctx.request.message)
            logger.debug(
                f"Routed request: intent={ctx.route_decision.intent.value}, "
                f"complexity={ctx.route_decision.complexity:.2f}"
            )
        except Exception as e:
            ctx.add_warning(f"Routing failed, using defaults: {e}")
            ctx.route_decision = RouteDecision(
                intent=Intent.GENERAL_CHAT,
                confidence=0.5,
                complexity=0.5,
                handler_name="default",
            )

    async def _step_plan(self, ctx: PipelineContext) -> None:
        """Step 2: Generate execution plan if complexity warrants it.

        Args:
            ctx: Pipeline context
        """
        if not ctx.route_decision:
            return

        if not ctx.route_decision.use_planning or not self._planner:
            return

        # Note: Complexity threshold and keyword detection are handled in router._should_use_planning()
        # If use_planning is True, we trust the router's decision (including keyword-based triggers)

        try:
            # Get available tools to make planner tool-aware
            available_tools = get_configured_tools()
            logger.debug(f"Planning with {len(available_tools)} available tools: {available_tools}")

            plan = await self._planner.create_plan(
                goal=ctx.request.message,
                context=ctx.request.context,
                tools=available_tools,  # Pass available tools to planner
            )
            ctx.plan = plan
            ctx.plan_id = plan.plan_id
            logger.debug(f"Created plan with {len(plan.steps)} steps")
        except Exception as e:
            ctx.add_warning(f"Planning failed, proceeding without plan: {e}")

    async def _step_assemble_context(self, ctx: PipelineContext) -> None:
        """Step 3: Assemble context from memory, plan, and history.

        Args:
            ctx: Pipeline context
        """
        assembled: dict[str, Any] = {
            "message": ctx.request.message,
            "user_context": ctx.request.context,
        }

        # Add routing info
        if ctx.route_decision:
            assembled["route"] = {
                "intent": ctx.route_decision.intent.value,
                "complexity": ctx.route_decision.complexity,
                "handler": ctx.route_decision.handler_name,
            }

        # Add plan info
        if ctx.plan:
            assembled["plan"] = {
                "plan_id": ctx.plan.plan_id,
                "goal": ctx.plan.goal,
                "step_count": len(ctx.plan.steps),
                "steps": [
                    {"description": s.description, "status": s.status.value}
                    for s in ctx.plan.steps[:5]  # Limit for context size
                ],
            }

        # Fetch from memory if enabled
        if (
            ctx.route_decision
            and ctx.route_decision.use_memory
            and self._memory_manager
        ):
            try:
                memories = await self._memory_manager.retrieve(
                    query=ctx.request.message,
                    k=10,
                )
                if memories:
                    assembled["relevant_memories"] = [
                        {"content": m.content, "category": m.category}
                        for m in memories[:5]
                    ]
                    logger.debug(f"Retrieved {len(memories)} relevant memories")
            except Exception as e:
                ctx.add_warning(f"Memory retrieval failed: {e}")

        ctx.assembled_context = assembled

    async def _step_execute(self, ctx: PipelineContext) -> None:
        """Step 4: Execute plan steps with tools or fall back to reasoning.

        This method executes plans using the ToolStepExecutor which routes:
        - TOOL_CALL steps to MCP or builtin tool executors
        - REASONING steps to the reasoning manager for response generation

        If no plan exists or tool execution fails, falls back to direct
        reasoning using the reasoning manager.

        Args:
            ctx: Pipeline context
        """
        complexity = ctx.route_decision.complexity if ctx.route_decision else 0.5
        strategy = ctx.request.force_strategy

        # If we have a plan, execute it with tool-aware executor
        if ctx.plan and self._plan_executor:
            try:
                logger.info(
                    f"Starting plan execution for plan {ctx.plan_id} "
                    f"with {len(ctx.plan.steps)} steps"
                )

                # Create tool-aware step executor
                # This routes TOOL_CALL steps to tool executors and
                # REASONING steps to the reasoning manager
                tool_step_executor = create_tool_step_executor(
                    memory_manager=self._memory_manager,
                    planner=self._planner,
                    reasoning_manager=self._reasoning_manager,
                    allow_reasoning_fallback=False,  # Pure Approach 1 - no fallback per step
                    mcp_connection_timeout=5.0,  # Quick timeout: fail fast if server is down
                    mcp_tool_execution_timeout=30.0,  # Allow 30s for tool execution (accounts for API latency)
                )

                # Create executor with custom step executor
                executor = PlanExecutor(
                    event_store=self._event_store,
                    token_tracker=self._token_tracker,
                    step_executor=tool_step_executor.execute_step,
                )

                # Execute plan with tool calls
                result = await executor.execute(
                    plan=ctx.plan,
                    session_id=ctx.request.session_id,
                )

                ctx.tokens_used += result.total_tokens

                if result.success:
                    logger.info(
                        f"Plan {ctx.plan_id} executed successfully "
                        f"with {len(result.step_results)} steps, "
                        f"tokens={result.total_tokens}"
                    )
                    ctx.output = {
                        "result": result.final_output or "Plan executed successfully",
                        "plan_id": ctx.plan_id,
                        "steps_executed": len(result.step_results),
                        "step_results": [
                            {
                                "step_id": sr.step_id,
                                "status": sr.status.value,
                                "tokens_used": sr.tokens_used,
                            }
                            for sr in result.step_results
                        ],
                        "total_tokens": result.total_tokens,
                        "total_duration_ms": result.total_duration_ms,
                    }
                else:
                    # Some steps failed but plan continued
                    logger.warning(
                        f"Plan {ctx.plan_id} execution had failures: {result.errors}"
                    )
                    ctx.add_error(f"Plan execution failed: {result.errors}")
                    ctx.output = {
                        "result": "Plan execution failed",
                        "plan_id": ctx.plan_id,
                        "errors": result.errors,
                        "partial_output": result.final_output,
                        "steps_executed": len(result.step_results),
                    }
                return

            except PlanExecutionError as e:
                logger.warning(
                    f"Plan execution error for {ctx.plan_id}: {e}. "
                    f"Falling back to reasoning."
                )
                ctx.add_warning(f"Plan execution failed, falling back to reasoning: {e}")
                # Fall through to reasoning manager below

            except Exception as e:
                logger.warning(
                    f"Unexpected error during plan execution for {ctx.plan_id}: {e}. "
                    f"Falling back to reasoning."
                )
                ctx.add_warning(f"Tool execution failed, falling back to reasoning: {e}")
                # Fall through to reasoning manager below

        # Fall back to reasoning manager if no plan or tool execution failed
        logger.debug(
            f"Using reasoning manager directly "
            f"(complexity={complexity:.2f}, strategy={strategy or 'auto'})"
        )

        try:
            result = await self._reasoning_manager.execute(
                task=ctx.request.message,
                context=ctx.assembled_context,
                complexity=complexity,
                strategy=strategy,
                session_id=ctx.request.session_id,
            )

            ctx.tokens_used += result.tokens_used
            ctx.reasoning_result = result

            if result.success:
                ctx.output = {
                    "result": result.answer,
                    "reasoning_trace": result.reasoning_trace[:5],  # Limit trace
                    "model": result.model,
                    "confidence": result.confidence,
                }
            else:
                ctx.add_error(f"Reasoning failed: {result.error}")
                ctx.output = {"error": result.error}

        except ReasoningError as e:
            ctx.add_error(f"Reasoning error: {e}")
            ctx.output = {"error": str(e)}

    async def _step_validate(self, ctx: PipelineContext) -> None:
        """Step 5: Validate output against contract.

        Args:
            ctx: Pipeline context
        """
        if not ctx.route_decision or not ctx.route_decision.use_contracts:
            return

        if not self._contract_executor:
            return

        # Determine contract to use
        contract_name = ctx.request.contract_name
        if not contract_name and ctx.route_decision:
            contract_name = get_contract_for_intent(ctx.route_decision.intent)

        if not contract_name:
            return

        try:
            # Get the contract
            from sigil.contracts.templates.acti import get_contract
            contract = get_contract(contract_name)

            if not contract:
                ctx.add_warning(f"Contract not found: {contract_name}")
                return

            # Create a simple agent wrapper for validation
            class OutputWrapper:
                def __init__(self, output: dict[str, Any]):
                    self._output = output

                async def run(self, task: str, context: Optional[Any] = None) -> dict[str, Any]:
                    return self._output

            wrapper = OutputWrapper(ctx.output)

            result = await self._contract_executor.execute_with_contract(
                agent=wrapper,
                task=ctx.request.message,
                contract=contract,
                context=ctx.assembled_context,
            )

            ctx.contract_result = result
            ctx.tokens_used += result.tokens_used

            if not result.is_valid:
                ctx.add_warning(
                    f"Contract validation failed: "
                    f"{result.validation_result.get_error_summary()}"
                )

        except Exception as e:
            ctx.add_warning(f"Contract validation error: {e}")

    def get_metrics(self) -> dict[str, Any]:
        """Get orchestrator metrics.

        Returns:
            Dictionary with orchestrator metrics
        """
        return {
            "total_requests": self._total_requests,
            "successful_requests": self._successful_requests,
            "failed_requests": self._failed_requests,
            "success_rate": (
                self._successful_requests / max(self._total_requests, 1)
            ),
            "total_tokens": self._total_tokens,
            "total_time_ms": self._total_time_ms,
            "avg_tokens_per_request": (
                self._total_tokens / max(self._total_requests, 1)
            ),
            "avg_time_per_request_ms": (
                self._total_time_ms / max(self._total_requests, 1)
            ),
            "features_enabled": self._settings.get_active_features(),
        }

    def reset_metrics(self) -> None:
        """Reset orchestrator metrics."""
        self._total_requests = 0
        self._successful_requests = 0
        self._failed_requests = 0
        self._total_tokens = 0
        self._total_time_ms = 0.0
        logger.info("Orchestrator metrics reset")

    @property
    def is_healthy(self) -> bool:
        """Check if all subsystems are healthy."""
        try:
            # Basic health checks
            if self._memory_manager and not self._memory_manager.is_healthy:
                return False
            return True
        except Exception:
            return False


# =============================================================================
# Convenience Functions
# =============================================================================


async def process_message(
    message: str,
    session_id: str,
    user_id: Optional[str] = None,
    **kwargs: Any,
) -> OrchestratorResponse:
    """Convenience function to process a message.

    Creates an orchestrator and processes a single request.

    Args:
        message: The message to process
        session_id: Session identifier
        user_id: Optional user identifier
        **kwargs: Additional request parameters

    Returns:
        OrchestratorResponse with results

    Example:
        >>> response = await process_message(
        ...     "Create a sales agent",
        ...     session_id="sess-123",
        ... )
    """
    orchestrator = SigilOrchestrator()
    request = OrchestratorRequest(
        message=message,
        session_id=session_id,
        user_id=user_id,
        **kwargs,
    )
    return await orchestrator.process(request)


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Enums
    "OrchestratorStatus",
    # Request/Response
    "OrchestratorRequest",
    "OrchestratorResponse",
    # Main class
    "SigilOrchestrator",
    "PipelineContext",
    # Convenience functions
    "process_message",
    # Constants
    "DEFAULT_INPUT_BUDGET",
    "DEFAULT_OUTPUT_BUDGET",
    "DEFAULT_TOTAL_BUDGET",
]
