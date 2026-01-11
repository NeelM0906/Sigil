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
