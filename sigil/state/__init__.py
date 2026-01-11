"""State module for Sigil v2 framework.

This module implements event-sourced state management for agents:
- Event definitions and factory functions
- Append-only event store with JSON persistence
- Session state rebuilt from event replay
- Session lifecycle management

Key Components:
    - Event: Base dataclass for all events (immutable)
    - EventType: Enum of all event types
    - EventStore: Append-only event persistence
    - Session: Current state derived from events
    - SessionManager: Factory for session lifecycle

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
    - Event sourcing: State is derived from event replay
    - Immutable events: Events cannot be modified once created
    - Full audit trail: Every state change is recorded
    - JSON persistence: Human-readable, debuggable storage

Example:
    >>> from sigil.state import EventStore, Session, SessionManager, EventType
    >>> from sigil.state.events import create_session_started_event, create_message_added_event
    >>>
    >>> # Create a session via manager
    >>> manager = SessionManager(storage_dir="./outputs/sessions")
    >>> session = manager.create_session(agent_name="LeadQualifier")
    >>>
    >>> # Add events to the session
    >>> from sigil.state.events import create_message_added_event
    >>> event = create_message_added_event(
    ...     session_id=session.session_id,
    ...     role="user",
    ...     content="Hello!"
    ... )
    >>> session.apply_event(event)
    >>>
    >>> # Access derived state
    >>> session.messages
    [{'role': 'user', 'content': 'Hello!', ...}]
"""

from sigil.state.events import (
    Event,
    EventType,
    # Factory functions
    create_session_started_event,
    create_session_ended_event,
    create_message_added_event,
    create_tool_called_event,
    create_plan_created_event,
    create_plan_step_completed_event,
    create_memory_extracted_event,
    create_contract_validated_event,
    create_error_occurred_event,
    # Utility functions
    events_to_json,
    events_from_json,
    filter_events_by_type,
    filter_events_by_session,
    filter_events_in_range,
)

from sigil.state.store import (
    EventStore,
    EventStoreError,
    SessionNotFoundError,
)

from sigil.state.session import (
    Session,
    SessionManager,
    SessionSummary,
    SessionError,
    SessionEndedError,
)

__all__ = [
    # Core classes
    "Event",
    "EventType",
    "EventStore",
    "Session",
    "SessionManager",
    "SessionSummary",
    # Exceptions
    "EventStoreError",
    "SessionNotFoundError",
    "SessionError",
    "SessionEndedError",
    # Event factory functions
    "create_session_started_event",
    "create_session_ended_event",
    "create_message_added_event",
    "create_tool_called_event",
    "create_plan_created_event",
    "create_plan_step_completed_event",
    "create_memory_extracted_event",
    "create_contract_validated_event",
    "create_error_occurred_event",
    # Event utility functions
    "events_to_json",
    "events_from_json",
    "filter_events_by_type",
    "filter_events_by_session",
    "filter_events_in_range",
]
