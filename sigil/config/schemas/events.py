"""Event and message schemas for Sigil v2.

This module defines Pydantic models for the event-driven architecture
that enables communication between agents and system components.

Event System Overview:
    The event system provides:
    - Asynchronous communication between components
    - Event sourcing for state reconstruction
    - Audit trail for debugging and analysis
    - Integration points for external systems

Event Categories:
    - Agent Events: Lifecycle and execution events
    - Memory Events: Memory read/write operations
    - Tool Events: Tool invocation and results
    - Contract Events: Verification outcomes
    - Evolution Events: Optimization iterations

Classes:
    Event: Base event schema
        - id: Unique event identifier
        - type: Event type string
        - timestamp: When the event occurred
        - source: Component that emitted the event
        - payload: Event-specific data

    AgentEvent: Agent lifecycle events
    MemoryEvent: Memory operation events
    ToolEvent: Tool invocation events
    ContractEvent: Contract verification events
    EvolutionEvent: Self-improvement events

TODO:
    - Implement Event base class with serialization
    - Implement specialized event types
    - Add event validation
    - Add event routing support
    - Add event persistence
    - Add event replay for debugging
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional


class EventType(Enum):
    """Enumeration of event types.

    TODO: Add comprehensive event type coverage
    """

    # Agent events
    AGENT_STARTED = "agent.started"
    AGENT_COMPLETED = "agent.completed"
    AGENT_FAILED = "agent.failed"
    AGENT_THINKING = "agent.thinking"
    AGENT_ACTING = "agent.acting"

    # Memory events
    MEMORY_READ = "memory.read"
    MEMORY_WRITE = "memory.write"
    MEMORY_DELETE = "memory.delete"
    MEMORY_SEARCH = "memory.search"

    # Tool events
    TOOL_INVOKED = "tool.invoked"
    TOOL_COMPLETED = "tool.completed"
    TOOL_FAILED = "tool.failed"

    # Contract events
    CONTRACT_STARTED = "contract.started"
    CONTRACT_PASSED = "contract.passed"
    CONTRACT_FAILED = "contract.failed"
    CONTRACT_RETRY = "contract.retry"

    # Evolution events
    EVOLUTION_STARTED = "evolution.started"
    EVOLUTION_ITERATION = "evolution.iteration"
    EVOLUTION_COMPLETED = "evolution.completed"

    # System events
    SYSTEM_ERROR = "system.error"
    SYSTEM_WARNING = "system.warning"

    # Planning events (Phase 5)
    PLAN_CREATED = "plan.created"
    PLAN_EXECUTION_STARTED = "plan.execution_started"
    PLAN_STEP_STARTED = "plan.step_started"
    PLAN_STEP_COMPLETED = "plan.step_completed"
    PLAN_STEP_FAILED = "plan.step_failed"
    PLAN_COMPLETED = "plan.completed"
    PLAN_PAUSED = "plan.paused"
    PLAN_RESUMED = "plan.resumed"
    PLAN_ABORTED = "plan.aborted"

    # Reasoning events (Phase 5)
    REASONING_TASK_STARTED = "reasoning.task_started"
    REASONING_STRATEGY_SELECTED = "reasoning.strategy_selected"
    REASONING_COMPLETED = "reasoning.completed"
    REASONING_FALLBACK = "reasoning.fallback"


@dataclass
class Event:
    """Base event schema for all system events.

    Events provide a standardized way to communicate state changes
    and actions throughout the system. They enable event sourcing,
    debugging, and integration with external systems.

    Attributes:
        id: Unique event identifier (UUID)
        type: Event type from EventType enum
        timestamp: When the event occurred (UTC)
        source: Component/agent that emitted the event
        correlation_id: ID linking related events
        payload: Event-specific data
        metadata: Additional event metadata

    TODO:
        - Convert to Pydantic BaseModel
        - Add auto-generated id and timestamp
        - Add event validation
        - Add serialization methods
    """

    id: str = ""
    type: EventType = EventType.SYSTEM_ERROR
    timestamp: Optional[datetime] = None
    source: str = ""
    correlation_id: Optional[str] = None
    payload: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentEvent(Event):
    """Event for agent lifecycle changes.

    Attributes:
        agent_id: ID of the agent
        agent_name: Name of the agent type
        task: Current task (if applicable)
        result: Execution result (if completed)

    TODO: Add agent-specific payload validation
    """

    agent_id: str = ""
    agent_name: str = ""
    task: Optional[str] = None
    result: Optional[Any] = None


@dataclass
class MemoryEvent(Event):
    """Event for memory operations.

    Attributes:
        layer: Memory layer (episodic, semantic, procedural)
        operation: Operation type (read, write, delete)
        item_id: ID of the memory item
        item_count: Number of items affected

    TODO: Add memory-specific payload validation
    """

    layer: str = ""
    operation: str = ""
    item_id: Optional[str] = None
    item_count: int = 0


@dataclass
class ToolEvent(Event):
    """Event for tool invocations.

    Attributes:
        tool_name: Name of the tool
        tool_params: Parameters passed to the tool
        result: Tool execution result
        duration_ms: Execution time in milliseconds

    TODO: Add tool-specific payload validation
    """

    tool_name: str = ""
    tool_params: dict[str, Any] = field(default_factory=dict)
    result: Optional[Any] = None
    duration_ms: float = 0.0


@dataclass
class ContractEvent(Event):
    """Event for contract verification.

    Attributes:
        contract_id: ID of the contract
        contract_name: Name of the contract
        passed: Whether verification passed
        validation_errors: List of validation errors

    TODO: Add contract-specific payload validation
    """

    contract_id: str = ""
    contract_name: str = ""
    passed: bool = False
    validation_errors: list[str] = field(default_factory=list)


@dataclass
class EvolutionEvent(Event):
    """Event for self-evolution operations.

    Attributes:
        generation: Evolution generation number
        iteration: Iteration within generation
        loss: Current loss value
        improvement: Improvement from previous iteration

    TODO: Add evolution-specific payload validation
    """

    generation: int = 0
    iteration: int = 0
    loss: float = 0.0
    improvement: float = 0.0


# =============================================================================
# Phase 5 Planning Events
# =============================================================================


@dataclass
class PlanCreatedEvent(Event):
    """Event emitted when a new plan is created.

    Attributes:
        plan_id: Unique plan identifier.
        goal: The goal that generated the plan.
        step_count: Number of steps in the plan.
        complexity: Plan complexity score (0.0-1.0).
        estimated_tokens: Estimated token cost for execution.
    """

    plan_id: str = ""
    goal: str = ""
    step_count: int = 0
    complexity: float = 0.0
    estimated_tokens: int = 0

    def __post_init__(self) -> None:
        self.type = EventType.PLAN_CREATED


@dataclass
class PlanExecutionStartedEvent(Event):
    """Event emitted when plan execution begins.

    Attributes:
        plan_id: Plan being executed.
        token_budget: Token budget for execution.
        session_id: Session ID for tracking.
    """

    plan_id: str = ""
    token_budget: int = 0
    session_id: str = ""

    def __post_init__(self) -> None:
        self.type = EventType.PLAN_EXECUTION_STARTED


@dataclass
class PlanStepStartedEvent(Event):
    """Event emitted when a plan step starts executing.

    Attributes:
        plan_id: Parent plan ID.
        step_id: Step being executed.
        step_type: Type of step (tool_call, reasoning, etc.).
        description: Human-readable step description.
    """

    plan_id: str = ""
    step_id: str = ""
    step_type: str = ""
    description: str = ""

    def __post_init__(self) -> None:
        self.type = EventType.PLAN_STEP_STARTED


@dataclass
class PlanStepCompletedEvent(Event):
    """Event emitted when a plan step completes successfully.

    Attributes:
        plan_id: Parent plan ID.
        step_id: Completed step ID.
        success: Whether step succeeded.
        output_preview: First 200 chars of output.
        tokens_used: Tokens consumed by this step.
        duration_ms: Execution time in milliseconds.
    """

    plan_id: str = ""
    step_id: str = ""
    success: bool = True
    output_preview: str = ""
    tokens_used: int = 0
    duration_ms: float = 0.0

    def __post_init__(self) -> None:
        self.type = EventType.PLAN_STEP_COMPLETED


@dataclass
class PlanStepFailedEvent(Event):
    """Event emitted when a plan step fails.

    Attributes:
        plan_id: Parent plan ID.
        step_id: Failed step ID.
        error_type: Type of error.
        error_message: Error description.
        retry_count: Number of retries attempted.
        will_retry: Whether another retry will be attempted.
    """

    plan_id: str = ""
    step_id: str = ""
    error_type: str = ""
    error_message: str = ""
    retry_count: int = 0
    will_retry: bool = False

    def __post_init__(self) -> None:
        self.type = EventType.PLAN_STEP_FAILED


@dataclass
class PlanCompletedEvent(Event):
    """Event emitted when plan execution finishes.

    Attributes:
        plan_id: Completed plan ID.
        success: Whether all steps succeeded.
        total_tokens: Total tokens consumed.
        total_duration_ms: Total execution time.
        steps_completed: Number of completed steps.
        steps_failed: Number of failed steps.
    """

    plan_id: str = ""
    success: bool = True
    total_tokens: int = 0
    total_duration_ms: float = 0.0
    steps_completed: int = 0
    steps_failed: int = 0

    def __post_init__(self) -> None:
        self.type = EventType.PLAN_COMPLETED


@dataclass
class PlanPausedEvent(Event):
    """Event emitted when plan execution is paused.

    Attributes:
        plan_id: Paused plan ID.
        checkpoint_id: Checkpoint ID for resumption.
        last_completed_step: Last step that finished.
    """

    plan_id: str = ""
    checkpoint_id: str = ""
    last_completed_step: str = ""

    def __post_init__(self) -> None:
        self.type = EventType.PLAN_PAUSED


@dataclass
class PlanResumedEvent(Event):
    """Event emitted when plan execution resumes.

    Attributes:
        plan_id: Resumed plan ID.
        checkpoint_id: Checkpoint used for resumption.
        resuming_from: Step execution resumes from.
    """

    plan_id: str = ""
    checkpoint_id: str = ""
    resuming_from: str = ""

    def __post_init__(self) -> None:
        self.type = EventType.PLAN_RESUMED


@dataclass
class PlanAbortedEvent(Event):
    """Event emitted when plan execution is aborted.

    Attributes:
        plan_id: Aborted plan ID.
        reason: Why the plan was aborted.
        steps_completed: Steps that finished before abort.
    """

    plan_id: str = ""
    reason: str = ""
    steps_completed: int = 0

    def __post_init__(self) -> None:
        self.type = EventType.PLAN_ABORTED


# =============================================================================
# Phase 5 Reasoning Events
# =============================================================================


@dataclass
class ReasoningTaskStartedEvent(Event):
    """Event emitted when a reasoning task begins.

    Attributes:
        task_hash: SHA256 hash of task for privacy.
        complexity: Assessed complexity score.
        context_facts_count: Number of context facts provided.
        session_id: Session ID for tracking.
    """

    task_hash: str = ""
    complexity: float = 0.0
    context_facts_count: int = 0
    session_id: str = ""

    def __post_init__(self) -> None:
        self.type = EventType.REASONING_TASK_STARTED


@dataclass
class StrategySelectedEvent(Event):
    """Event emitted when a reasoning strategy is selected.

    Attributes:
        strategy_name: Name of selected strategy.
        complexity: Complexity score used for selection.
        was_override: True if strategy was explicitly requested.
        reason: Why this strategy was selected.
    """

    strategy_name: str = ""
    complexity: float = 0.0
    was_override: bool = False
    reason: str = ""

    def __post_init__(self) -> None:
        self.type = EventType.REASONING_STRATEGY_SELECTED


@dataclass
class ReasoningCompletedEvent(Event):
    """Event emitted when reasoning completes.

    Attributes:
        strategy_name: Strategy that executed.
        confidence: Result confidence score.
        tokens_used: Tokens consumed.
        duration_ms: Execution time.
        trace_length: Number of reasoning steps.
    """

    strategy_name: str = ""
    confidence: float = 0.0
    tokens_used: int = 0
    duration_ms: float = 0.0
    trace_length: int = 0

    def __post_init__(self) -> None:
        self.type = EventType.REASONING_COMPLETED


@dataclass
class StrategyFallbackEvent(Event):
    """Event emitted when strategy falls back to simpler one.

    Attributes:
        original_strategy: Strategy that was attempted.
        fallback_strategy: Strategy falling back to.
        reason: Why fallback occurred.
        error: Error message from original strategy.
    """

    original_strategy: str = ""
    fallback_strategy: str = ""
    reason: str = ""
    error: str = ""

    def __post_init__(self) -> None:
        self.type = EventType.REASONING_FALLBACK


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Enums
    "EventType",
    # Base Event
    "Event",
    # Original Events
    "AgentEvent",
    "MemoryEvent",
    "ToolEvent",
    "ContractEvent",
    "EvolutionEvent",
    # Phase 5 Planning Events
    "PlanCreatedEvent",
    "PlanExecutionStartedEvent",
    "PlanStepStartedEvent",
    "PlanStepCompletedEvent",
    "PlanStepFailedEvent",
    "PlanCompletedEvent",
    "PlanPausedEvent",
    "PlanResumedEvent",
    "PlanAbortedEvent",
    # Phase 5 Reasoning Events
    "ReasoningTaskStartedEvent",
    "StrategySelectedEvent",
    "ReasoningCompletedEvent",
    "StrategyFallbackEvent",
]
