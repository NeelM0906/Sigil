"""Event definitions for Sigil v2 event-sourced state management.

This module defines the core Event dataclass and specialized event types
for tracking all state changes in the agent framework. Events are immutable
and provide a complete audit trail for session reconstruction.

Event Types:
    - SessionStartedEvent: New session begins
    - SessionEndedEvent: Session terminates
    - MessageAddedEvent: User or assistant message
    - ToolCalledEvent: Tool invocation with args and result
    - PlanCreatedEvent: New plan generated
    - PlanStepCompletedEvent: Plan step finished
    - MemoryExtractedEvent: Memory item extracted
    - ContractValidatedEvent: Contract check passed/failed
    - ErrorOccurredEvent: Error during execution

Design Philosophy:
    - Immutable events (frozen=True)
    - UTC timestamps
    - UUID4 for event identification
    - JSON serialization with datetime support
    - Type-safe event types via enum

Example:
    >>> from sigil.state.events import create_session_started_event
    >>> event = create_session_started_event("session-123")
    >>> event.event_type
    <EventType.SESSION_STARTED: 'session.started'>
    >>> event.to_dict()
    {'event_id': '...', 'event_type': 'session.started', ...}
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional
import json
import uuid


class EventType(Enum):
    """Enumeration of all event types in the system.

    Event types are organized by category:
    - Session events: Lifecycle of agent sessions
    - Message events: Conversation messages
    - Tool events: Tool invocations
    - Plan events: Planning operations
    - Memory events: Memory extraction
    - Reasoning events: Strategy selection and execution
    - Contract events: Verification outcomes
    - Error events: System errors
    """

    # Session events
    SESSION_STARTED = "session.started"
    SESSION_ENDED = "session.ended"

    # Message events
    MESSAGE_ADDED = "message.added"

    # Tool events
    TOOL_CALLED = "tool.called"

    # Plan events
    PLAN_CREATED = "plan.created"
    PLAN_COMPLETED = "plan.completed"
    PLAN_STEP_COMPLETED = "plan.step_completed"
    STEP_STARTED = "step.started"
    STEP_COMPLETED = "step.completed"

    # Memory events
    MEMORY_EXTRACTED = "memory.extracted"

    # Reasoning events (Phase 5)
    STRATEGY_SELECTED = "reasoning.strategy_selected"
    REASONING_COMPLETED = "reasoning.completed"
    STRATEGY_FALLBACK = "reasoning.fallback"

    # Contract events
    CONTRACT_VALIDATED = "contract.validated"

    # Error events
    ERROR_OCCURRED = "error.occurred"


def _generate_event_id() -> str:
    """Generate a unique event ID using UUID4.

    Returns:
        A string representation of a UUID4.
    """
    return str(uuid.uuid4())


def _get_utc_now() -> datetime:
    """Get the current UTC timestamp.

    Returns:
        Current datetime in UTC timezone.
    """
    return datetime.now(timezone.utc)


def _datetime_to_iso(dt: datetime) -> str:
    """Convert datetime to ISO format string.

    Args:
        dt: Datetime object to convert.

    Returns:
        ISO format string representation.
    """
    return dt.isoformat()


def _datetime_from_iso(iso_str: str) -> datetime:
    """Parse ISO format string to datetime.

    Args:
        iso_str: ISO format datetime string.

    Returns:
        Parsed datetime object.
    """
    return datetime.fromisoformat(iso_str)


@dataclass(frozen=True)
class Event:
    """Base event dataclass for all system events.

    Events are immutable records of state changes. They provide a complete
    audit trail and enable session state reconstruction through event replay.

    Attributes:
        event_id: Unique identifier for this event (UUID4).
        event_type: Type discriminator from EventType enum.
        timestamp: When the event occurred (UTC).
        session_id: Which session this event belongs to.
        correlation_id: Optional ID for linking related events.
        payload: Event-specific data dictionary.

    Example:
        >>> event = Event(
        ...     event_id="abc-123",
        ...     event_type=EventType.SESSION_STARTED,
        ...     timestamp=datetime.now(timezone.utc),
        ...     session_id="session-456",
        ...     payload={"agent_name": "TestAgent"}
        ... )
    """

    event_id: str
    event_type: EventType
    timestamp: datetime
    session_id: str
    correlation_id: Optional[str] = None
    payload: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize event to dictionary.

        Converts the event to a dictionary suitable for JSON serialization.
        Datetime objects are converted to ISO format strings.

        Returns:
            Dictionary representation of the event.
        """
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "timestamp": _datetime_to_iso(self.timestamp),
            "session_id": self.session_id,
            "correlation_id": self.correlation_id,
            "payload": self.payload,
        }

    def to_json(self) -> str:
        """Serialize event to JSON string.

        Returns:
            JSON string representation of the event.
        """
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Event":
        """Create an Event from a dictionary.

        Args:
            data: Dictionary with event data.

        Returns:
            New Event instance.
        """
        return cls(
            event_id=data["event_id"],
            event_type=EventType(data["event_type"]),
            timestamp=_datetime_from_iso(data["timestamp"]),
            session_id=data["session_id"],
            correlation_id=data.get("correlation_id"),
            payload=data.get("payload", {}),
        )

    @classmethod
    def from_json(cls, json_str: str) -> "Event":
        """Create an Event from a JSON string.

        Args:
            json_str: JSON string with event data.

        Returns:
            New Event instance.
        """
        return cls.from_dict(json.loads(json_str))


# =============================================================================
# Factory Functions for Creating Specific Event Types
# =============================================================================


def create_session_started_event(
    session_id: str,
    agent_name: Optional[str] = None,
    agent_config: Optional[dict[str, Any]] = None,
    correlation_id: Optional[str] = None,
) -> Event:
    """Create a SessionStartedEvent when a new session begins.

    Args:
        session_id: Unique identifier for the session.
        agent_name: Optional name of the agent starting the session.
        agent_config: Optional configuration for the agent.
        correlation_id: Optional ID for linking related events.

    Returns:
        Event with SESSION_STARTED type.

    Example:
        >>> event = create_session_started_event(
        ...     session_id="sess-123",
        ...     agent_name="LeadQualifier"
        ... )
    """
    payload: dict[str, Any] = {}
    if agent_name:
        payload["agent_name"] = agent_name
    if agent_config:
        payload["agent_config"] = agent_config

    return Event(
        event_id=_generate_event_id(),
        event_type=EventType.SESSION_STARTED,
        timestamp=_get_utc_now(),
        session_id=session_id,
        correlation_id=correlation_id,
        payload=payload,
    )


def create_session_ended_event(
    session_id: str,
    reason: str = "completed",
    summary: Optional[dict[str, Any]] = None,
    correlation_id: Optional[str] = None,
) -> Event:
    """Create a SessionEndedEvent when a session terminates.

    Args:
        session_id: Unique identifier for the session.
        reason: Why the session ended (e.g., "completed", "error", "timeout").
        summary: Optional summary data for the session.
        correlation_id: Optional ID for linking related events.

    Returns:
        Event with SESSION_ENDED type.
    """
    payload: dict[str, Any] = {"reason": reason}
    if summary:
        payload["summary"] = summary

    return Event(
        event_id=_generate_event_id(),
        event_type=EventType.SESSION_ENDED,
        timestamp=_get_utc_now(),
        session_id=session_id,
        correlation_id=correlation_id,
        payload=payload,
    )


def create_message_added_event(
    session_id: str,
    role: str,
    content: str,
    message_id: Optional[str] = None,
    metadata: Optional[dict[str, Any]] = None,
    correlation_id: Optional[str] = None,
) -> Event:
    """Create a MessageAddedEvent for user or assistant messages.

    Args:
        session_id: Unique identifier for the session.
        role: Message role (e.g., "user", "assistant", "system").
        content: Message content text.
        message_id: Optional unique ID for the message.
        metadata: Optional metadata for the message.
        correlation_id: Optional ID for linking related events.

    Returns:
        Event with MESSAGE_ADDED type.

    Example:
        >>> event = create_message_added_event(
        ...     session_id="sess-123",
        ...     role="user",
        ...     content="Hello, how can you help me?"
        ... )
    """
    payload: dict[str, Any] = {
        "role": role,
        "content": content,
    }
    if message_id:
        payload["message_id"] = message_id
    else:
        payload["message_id"] = _generate_event_id()
    if metadata:
        payload["metadata"] = metadata

    return Event(
        event_id=_generate_event_id(),
        event_type=EventType.MESSAGE_ADDED,
        timestamp=_get_utc_now(),
        session_id=session_id,
        correlation_id=correlation_id,
        payload=payload,
    )


def create_tool_called_event(
    session_id: str,
    tool_name: str,
    tool_args: dict[str, Any],
    tool_result: Optional[Any] = None,
    success: bool = True,
    error_message: Optional[str] = None,
    duration_ms: Optional[float] = None,
    correlation_id: Optional[str] = None,
) -> Event:
    """Create a ToolCalledEvent for tool invocations.

    Args:
        session_id: Unique identifier for the session.
        tool_name: Name of the tool that was called.
        tool_args: Arguments passed to the tool.
        tool_result: Result returned by the tool.
        success: Whether the tool call succeeded.
        error_message: Error message if the call failed.
        duration_ms: Execution time in milliseconds.
        correlation_id: Optional ID for linking related events.

    Returns:
        Event with TOOL_CALLED type.

    Example:
        >>> event = create_tool_called_event(
        ...     session_id="sess-123",
        ...     tool_name="web_search",
        ...     tool_args={"query": "weather today"},
        ...     tool_result={"temperature": "72F"}
        ... )
    """
    payload: dict[str, Any] = {
        "tool_name": tool_name,
        "tool_args": tool_args,
        "success": success,
    }
    if tool_result is not None:
        payload["tool_result"] = tool_result
    if error_message:
        payload["error_message"] = error_message
    if duration_ms is not None:
        payload["duration_ms"] = duration_ms

    return Event(
        event_id=_generate_event_id(),
        event_type=EventType.TOOL_CALLED,
        timestamp=_get_utc_now(),
        session_id=session_id,
        correlation_id=correlation_id,
        payload=payload,
    )


def create_plan_created_event(
    session_id: str,
    plan_id: str,
    goal: str,
    steps: list[dict[str, Any]],
    metadata: Optional[dict[str, Any]] = None,
    correlation_id: Optional[str] = None,
) -> Event:
    """Create a PlanCreatedEvent when a new plan is generated.

    Args:
        session_id: Unique identifier for the session.
        plan_id: Unique identifier for the plan.
        goal: The goal the plan aims to achieve.
        steps: List of plan steps with their details.
        metadata: Optional metadata for the plan.
        correlation_id: Optional ID for linking related events.

    Returns:
        Event with PLAN_CREATED type.

    Example:
        >>> event = create_plan_created_event(
        ...     session_id="sess-123",
        ...     plan_id="plan-456",
        ...     goal="Qualify lead John from Acme Corp",
        ...     steps=[
        ...         {"step_id": "1", "description": "Retrieve existing info"},
        ...         {"step_id": "2", "description": "Identify BANT gaps"}
        ...     ]
        ... )
    """
    payload: dict[str, Any] = {
        "plan_id": plan_id,
        "goal": goal,
        "steps": steps,
        "step_count": len(steps),
    }
    if metadata:
        payload["metadata"] = metadata

    return Event(
        event_id=_generate_event_id(),
        event_type=EventType.PLAN_CREATED,
        timestamp=_get_utc_now(),
        session_id=session_id,
        correlation_id=correlation_id,
        payload=payload,
    )


def create_plan_step_completed_event(
    session_id: str,
    plan_id: str,
    step_id: str,
    step_index: int,
    success: bool = True,
    result: Optional[Any] = None,
    error_message: Optional[str] = None,
    duration_ms: Optional[float] = None,
    correlation_id: Optional[str] = None,
) -> Event:
    """Create a PlanStepCompletedEvent when a plan step finishes.

    Args:
        session_id: Unique identifier for the session.
        plan_id: Unique identifier for the plan.
        step_id: Unique identifier for the step.
        step_index: Zero-based index of the step in the plan.
        success: Whether the step completed successfully.
        result: Result produced by the step.
        error_message: Error message if the step failed.
        duration_ms: Execution time in milliseconds.
        correlation_id: Optional ID for linking related events.

    Returns:
        Event with PLAN_STEP_COMPLETED type.
    """
    payload: dict[str, Any] = {
        "plan_id": plan_id,
        "step_id": step_id,
        "step_index": step_index,
        "success": success,
    }
    if result is not None:
        payload["result"] = result
    if error_message:
        payload["error_message"] = error_message
    if duration_ms is not None:
        payload["duration_ms"] = duration_ms

    return Event(
        event_id=_generate_event_id(),
        event_type=EventType.PLAN_STEP_COMPLETED,
        timestamp=_get_utc_now(),
        session_id=session_id,
        correlation_id=correlation_id,
        payload=payload,
    )


def create_memory_extracted_event(
    session_id: str,
    memory_id: str,
    memory_type: str,
    content: str,
    source: Optional[str] = None,
    confidence: Optional[float] = None,
    metadata: Optional[dict[str, Any]] = None,
    correlation_id: Optional[str] = None,
) -> Event:
    """Create a MemoryExtractedEvent when a memory item is extracted.

    Args:
        session_id: Unique identifier for the session.
        memory_id: Unique identifier for the memory item.
        memory_type: Type of memory (e.g., "fact", "preference", "insight").
        content: The extracted memory content.
        source: Source of the memory (e.g., message ID, document ID).
        confidence: Confidence score for the extraction (0.0-1.0).
        metadata: Optional metadata for the memory item.
        correlation_id: Optional ID for linking related events.

    Returns:
        Event with MEMORY_EXTRACTED type.
    """
    payload: dict[str, Any] = {
        "memory_id": memory_id,
        "memory_type": memory_type,
        "content": content,
    }
    if source:
        payload["source"] = source
    if confidence is not None:
        payload["confidence"] = confidence
    if metadata:
        payload["metadata"] = metadata

    return Event(
        event_id=_generate_event_id(),
        event_type=EventType.MEMORY_EXTRACTED,
        timestamp=_get_utc_now(),
        session_id=session_id,
        correlation_id=correlation_id,
        payload=payload,
    )


def create_contract_validated_event(
    session_id: str,
    contract_id: str,
    contract_name: str,
    passed: bool,
    deliverables_checked: list[str],
    validation_errors: Optional[list[str]] = None,
    retry_count: int = 0,
    correlation_id: Optional[str] = None,
) -> Event:
    """Create a ContractValidatedEvent for contract verification results.

    Args:
        session_id: Unique identifier for the session.
        contract_id: Unique identifier for the contract.
        contract_name: Name of the contract being validated.
        passed: Whether the contract validation passed.
        deliverables_checked: List of deliverables that were validated.
        validation_errors: List of validation error messages if failed.
        retry_count: Number of retries attempted before this result.
        correlation_id: Optional ID for linking related events.

    Returns:
        Event with CONTRACT_VALIDATED type.
    """
    payload: dict[str, Any] = {
        "contract_id": contract_id,
        "contract_name": contract_name,
        "passed": passed,
        "deliverables_checked": deliverables_checked,
        "retry_count": retry_count,
    }
    if validation_errors:
        payload["validation_errors"] = validation_errors

    return Event(
        event_id=_generate_event_id(),
        event_type=EventType.CONTRACT_VALIDATED,
        timestamp=_get_utc_now(),
        session_id=session_id,
        correlation_id=correlation_id,
        payload=payload,
    )


def create_error_occurred_event(
    session_id: str,
    error_type: str,
    error_message: str,
    error_code: Optional[str] = None,
    stack_trace: Optional[str] = None,
    recoverable: bool = False,
    context: Optional[dict[str, Any]] = None,
    correlation_id: Optional[str] = None,
) -> Event:
    """Create an ErrorOccurredEvent when an error happens during execution.

    Args:
        session_id: Unique identifier for the session.
        error_type: Type/class of the error (e.g., "ValueError", "APIError").
        error_message: Human-readable error message.
        error_code: Optional error code for categorization.
        stack_trace: Optional stack trace for debugging.
        recoverable: Whether the error is recoverable.
        context: Optional context about what was happening when error occurred.
        correlation_id: Optional ID for linking related events.

    Returns:
        Event with ERROR_OCCURRED type.
    """
    payload: dict[str, Any] = {
        "error_type": error_type,
        "error_message": error_message,
        "recoverable": recoverable,
    }
    if error_code:
        payload["error_code"] = error_code
    if stack_trace:
        payload["stack_trace"] = stack_trace
    if context:
        payload["context"] = context

    return Event(
        event_id=_generate_event_id(),
        event_type=EventType.ERROR_OCCURRED,
        timestamp=_get_utc_now(),
        session_id=session_id,
        correlation_id=correlation_id,
        payload=payload,
    )


# =============================================================================
# Event List Utilities
# =============================================================================


def events_to_json(events: list[Event]) -> str:
    """Serialize a list of events to JSON.

    Args:
        events: List of Event objects.

    Returns:
        JSON string containing all events.
    """
    return json.dumps([event.to_dict() for event in events], indent=2)


def events_from_json(json_str: str) -> list[Event]:
    """Deserialize a list of events from JSON.

    Args:
        json_str: JSON string containing event data.

    Returns:
        List of Event objects.
    """
    data = json.loads(json_str)
    return [Event.from_dict(item) for item in data]


def filter_events_by_type(events: list[Event], event_type: EventType) -> list[Event]:
    """Filter events by their type.

    Args:
        events: List of events to filter.
        event_type: Type to filter by.

    Returns:
        List of events matching the specified type.
    """
    return [event for event in events if event.event_type == event_type]


def filter_events_by_session(events: list[Event], session_id: str) -> list[Event]:
    """Filter events by session ID.

    Args:
        events: List of events to filter.
        session_id: Session ID to filter by.

    Returns:
        List of events belonging to the specified session.
    """
    return [event for event in events if event.session_id == session_id]


def filter_events_in_range(
    events: list[Event], start: datetime, end: datetime
) -> list[Event]:
    """Filter events within a time range.

    Args:
        events: List of events to filter.
        start: Start of the time range (inclusive).
        end: End of the time range (inclusive).

    Returns:
        List of events within the specified time range.
    """
    return [event for event in events if start <= event.timestamp <= end]
