"""Session management for Sigil v2 event-sourced state.

This module implements Session and SessionManager classes for managing
agent session state through event sourcing. State is derived by replaying
events rather than storing mutable state directly.

Key Concepts:
    - Session: Current state derived from replaying events
    - SessionManager: Factory for creating and loading sessions
    - Event Sourcing: State is the result of applying events sequentially

Example:
    >>> from sigil.state.session import SessionManager
    >>> from sigil.state.events import create_message_added_event
    >>>
    >>> manager = SessionManager(storage_dir="/tmp/sessions")
    >>> session = manager.create_session()
    >>>
    >>> # Add a message event
    >>> event = create_message_added_event(
    ...     session_id=session.session_id,
    ...     role="user",
    ...     content="Hello!"
    ... )
    >>> session.apply_event(event)
    >>>
    >>> # Access derived state
    >>> len(session.messages)
    1
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

from sigil.state.events import (
    Event,
    EventType,
    create_session_started_event,
    create_session_ended_event,
    _datetime_to_iso,
)
from sigil.state.store import EventStore, SessionNotFoundError
from sigil.telemetry.tokens import TokenTracker


class SessionError(Exception):
    """Base exception for session errors."""

    pass


class SessionEndedError(SessionError):
    """Raised when trying to modify an ended session."""

    pass


@dataclass
class SessionSummary:
    """Summary information about a session.

    Provides a lightweight view of session metadata without loading
    all events.

    Attributes:
        session_id: Unique identifier for the session.
        created_at: When the session was created.
        updated_at: When the session was last updated.
        event_count: Number of events in the session.
        status: Current session status (active, archived, error).
        agent_name: Optional name of the agent for this session.
    """

    session_id: str
    created_at: str
    updated_at: str
    event_count: int
    status: str = "active"
    agent_name: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "session_id": self.session_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "event_count": self.event_count,
            "status": self.status,
            "agent_name": self.agent_name,
        }


class Session:
    """Session state derived from event replay.

    A Session represents the current state of an agent conversation,
    computed by replaying all events for that session. State is not
    stored directly; it is always derived from events.

    Attributes:
        session_id: Unique identifier for the session.
        created_at: When the session was created.
        status: Current session status (active, ended, error).

    Properties (derived from events):
        messages: Conversation history
        tool_calls: Tools invoked during session
        current_plan: Active plan (if any)
        memory_items: Extracted memories
        token_usage: Accumulated token usage

    Example:
        >>> session = Session(session_id="sess-123")
        >>> event = create_message_added_event("sess-123", "user", "Hello!")
        >>> session.apply_event(event)
        >>> session.messages
        [{'role': 'user', 'content': 'Hello!', 'message_id': '...'}]
    """

    def __init__(
        self,
        session_id: str,
        created_at: Optional[datetime] = None,
        agent_name: Optional[str] = None,
        agent_config: Optional[dict[str, Any]] = None,
    ) -> None:
        """Initialize a new session.

        Args:
            session_id: Unique identifier for the session.
            created_at: When the session was created. Defaults to now.
            agent_name: Optional name of the agent for this session.
            agent_config: Optional configuration for the agent.
        """
        self.session_id = session_id
        self.created_at = created_at or datetime.now(timezone.utc)
        self.agent_name = agent_name
        self.agent_config = agent_config or {}
        self.status = "active"

        # Internal state derived from events
        self._messages: list[dict[str, Any]] = []
        self._tool_calls: list[dict[str, Any]] = []
        self._current_plan: Optional[dict[str, Any]] = None
        self._completed_plan_steps: list[dict[str, Any]] = []
        self._memory_items: list[dict[str, Any]] = []
        self._contract_validations: list[dict[str, Any]] = []
        self._errors: list[dict[str, Any]] = []
        self._token_usage = TokenTracker()

        # Event tracking
        self._events: list[Event] = []
        self._event_store: Optional[EventStore] = None

    @property
    def messages(self) -> list[dict[str, Any]]:
        """Get conversation messages in chronological order.

        Returns:
            List of message dictionaries with role, content, and metadata.
        """
        return self._messages.copy()

    @property
    def tool_calls(self) -> list[dict[str, Any]]:
        """Get tool invocations during this session.

        Returns:
            List of tool call dictionaries with name, args, result, etc.
        """
        return self._tool_calls.copy()

    @property
    def current_plan(self) -> Optional[dict[str, Any]]:
        """Get the current active plan, if any.

        Returns:
            Dictionary with plan details or None if no active plan.
        """
        return self._current_plan.copy() if self._current_plan else None

    @property
    def memory_items(self) -> list[dict[str, Any]]:
        """Get memory items extracted during this session.

        Returns:
            List of memory item dictionaries.
        """
        return self._memory_items.copy()

    @property
    def token_usage(self) -> TokenTracker:
        """Get accumulated token usage for this session.

        Returns:
            TokenTracker with cumulative usage.
        """
        return self._token_usage

    @property
    def errors(self) -> list[dict[str, Any]]:
        """Get errors that occurred during this session.

        Returns:
            List of error dictionaries.
        """
        return self._errors.copy()

    @property
    def contract_validations(self) -> list[dict[str, Any]]:
        """Get contract validation results.

        Returns:
            List of contract validation dictionaries.
        """
        return self._contract_validations.copy()

    @property
    def events(self) -> list[Event]:
        """Get all events for this session.

        Returns:
            List of Event objects in chronological order.
        """
        return self._events.copy()

    def apply_event(self, event: Event) -> None:
        """Apply an event to update session state.

        This is the core event sourcing mechanism. Each event type
        updates the appropriate aspect of session state.

        Args:
            event: The event to apply.

        Raises:
            SessionEndedError: If the session has already ended.
            ValueError: If the event doesn't belong to this session.
        """
        if event.session_id != self.session_id:
            raise ValueError(
                f"Event session_id {event.session_id} doesn't match "
                f"session {self.session_id}"
            )

        if self.status == "ended" and event.event_type != EventType.SESSION_ENDED:
            raise SessionEndedError("Cannot apply events to an ended session")

        # Store the event
        self._events.append(event)

        # Persist to event store if attached
        if self._event_store:
            self._event_store.append(event)

        # Apply state changes based on event type
        self._apply_event_to_state(event)

    def _apply_event_to_state(self, event: Event) -> None:
        """Apply event-specific state changes.

        Args:
            event: The event to apply.
        """
        event_type = event.event_type
        payload = event.payload

        if event_type == EventType.SESSION_STARTED:
            if payload.get("agent_name"):
                self.agent_name = payload["agent_name"]
            if payload.get("agent_config"):
                self.agent_config = payload["agent_config"]

        elif event_type == EventType.SESSION_ENDED:
            self.status = "ended"

        elif event_type == EventType.MESSAGE_ADDED:
            self._messages.append({
                "role": payload["role"],
                "content": payload["content"],
                "message_id": payload.get("message_id"),
                "timestamp": event.timestamp.isoformat(),
                "metadata": payload.get("metadata", {}),
            })

        elif event_type == EventType.TOOL_CALLED:
            self._tool_calls.append({
                "tool_name": payload["tool_name"],
                "tool_args": payload["tool_args"],
                "tool_result": payload.get("tool_result"),
                "success": payload.get("success", True),
                "error_message": payload.get("error_message"),
                "duration_ms": payload.get("duration_ms"),
                "timestamp": event.timestamp.isoformat(),
            })

        elif event_type == EventType.PLAN_CREATED:
            self._current_plan = {
                "plan_id": payload["plan_id"],
                "goal": payload["goal"],
                "steps": payload["steps"],
                "step_count": payload["step_count"],
                "completed_steps": 0,
                "created_at": event.timestamp.isoformat(),
                "metadata": payload.get("metadata", {}),
            }
            self._completed_plan_steps = []

        elif event_type == EventType.PLAN_STEP_COMPLETED:
            step_data = {
                "plan_id": payload["plan_id"],
                "step_id": payload["step_id"],
                "step_index": payload["step_index"],
                "success": payload.get("success", True),
                "result": payload.get("result"),
                "error_message": payload.get("error_message"),
                "duration_ms": payload.get("duration_ms"),
                "timestamp": event.timestamp.isoformat(),
            }
            self._completed_plan_steps.append(step_data)

            # Update current plan's completed count
            if self._current_plan and self._current_plan["plan_id"] == payload["plan_id"]:
                self._current_plan["completed_steps"] = len(self._completed_plan_steps)

        elif event_type == EventType.MEMORY_EXTRACTED:
            self._memory_items.append({
                "memory_id": payload["memory_id"],
                "memory_type": payload["memory_type"],
                "content": payload["content"],
                "source": payload.get("source"),
                "confidence": payload.get("confidence"),
                "metadata": payload.get("metadata", {}),
                "timestamp": event.timestamp.isoformat(),
            })

        elif event_type == EventType.CONTRACT_VALIDATED:
            self._contract_validations.append({
                "contract_id": payload["contract_id"],
                "contract_name": payload["contract_name"],
                "passed": payload["passed"],
                "deliverables_checked": payload["deliverables_checked"],
                "validation_errors": payload.get("validation_errors", []),
                "retry_count": payload.get("retry_count", 0),
                "timestamp": event.timestamp.isoformat(),
            })

        elif event_type == EventType.ERROR_OCCURRED:
            self._errors.append({
                "error_type": payload["error_type"],
                "error_message": payload["error_message"],
                "error_code": payload.get("error_code"),
                "stack_trace": payload.get("stack_trace"),
                "recoverable": payload.get("recoverable", False),
                "context": payload.get("context", {}),
                "timestamp": event.timestamp.isoformat(),
            })

    def rebuild_from_events(self, events: list[Event]) -> None:
        """Rebuild session state from a list of events.

        Clears current state and replays all events to reconstruct
        the session state. Used when loading a session from storage.

        Args:
            events: List of events to replay, in chronological order.
        """
        # Reset state
        self._messages = []
        self._tool_calls = []
        self._current_plan = None
        self._completed_plan_steps = []
        self._memory_items = []
        self._contract_validations = []
        self._errors = []
        self._token_usage = TokenTracker()
        self._events = []
        self.status = "active"

        # Replay events
        for event in sorted(events, key=lambda e: e.timestamp):
            # Don't persist when rebuilding - events are already stored
            self._events.append(event)
            self._apply_event_to_state(event)

    def get_context(self) -> dict[str, Any]:
        """Get current session context snapshot.

        Returns a dictionary containing all current state, suitable
        for passing to agents or serializing.

        Returns:
            Dictionary with all session state.
        """
        return {
            "session_id": self.session_id,
            "created_at": self.created_at.isoformat(),
            "status": self.status,
            "agent_name": self.agent_name,
            "agent_config": self.agent_config,
            "messages": self.messages,
            "message_count": len(self._messages),
            "tool_calls": self.tool_calls,
            "tool_call_count": len(self._tool_calls),
            "current_plan": self.current_plan,
            "has_active_plan": self._current_plan is not None,
            "memory_items": self.memory_items,
            "memory_item_count": len(self._memory_items),
            "contract_validations": self.contract_validations,
            "errors": self.errors,
            "error_count": len(self._errors),
            "token_usage": self._token_usage.to_dict(),
            "event_count": len(self._events),
        }

    def get_conversation_history(self) -> list[dict[str, str]]:
        """Get messages formatted for LLM consumption.

        Returns:
            List of message dicts with 'role' and 'content' keys.
        """
        return [
            {"role": msg["role"], "content": msg["content"]}
            for msg in self._messages
        ]

    def attach_store(self, store: EventStore) -> None:
        """Attach an event store for automatic persistence.

        When attached, all applied events will be automatically
        persisted to the store.

        Args:
            store: The EventStore to attach.
        """
        self._event_store = store

    def detach_store(self) -> None:
        """Detach the event store."""
        self._event_store = None


class SessionManager:
    """Factory and manager for Session instances.

    Provides methods to create new sessions, load existing sessions,
    and manage session lifecycle. Sessions are persisted through
    the EventStore.

    Attributes:
        storage_dir: Directory for session storage.

    Example:
        >>> manager = SessionManager(storage_dir="./outputs/sessions")
        >>> session = manager.create_session(agent_name="LeadQualifier")
        >>> session.session_id
        'sess-abc123...'
        >>>
        >>> # Later, load the session
        >>> loaded = manager.load_session(session.session_id)
    """

    def __init__(self, storage_dir: str = "outputs/sessions") -> None:
        """Initialize the session manager.

        Args:
            storage_dir: Directory for session storage.
        """
        self.storage_dir = storage_dir
        self._store = EventStore(storage_dir=storage_dir)

    def _generate_session_id(self) -> str:
        """Generate a unique session ID.

        Returns:
            A unique session identifier.
        """
        return f"sess-{uuid.uuid4().hex[:12]}"

    def create_session(
        self,
        session_id: Optional[str] = None,
        agent_name: Optional[str] = None,
        agent_config: Optional[dict[str, Any]] = None,
    ) -> Session:
        """Create a new session.

        Creates a new Session, emits a SessionStartedEvent, and
        persists it to the store.

        Args:
            session_id: Optional custom session ID. Generated if not provided.
            agent_name: Optional name of the agent for this session.
            agent_config: Optional configuration for the agent.

        Returns:
            A new Session instance.
        """
        if session_id is None:
            session_id = self._generate_session_id()

        # Check if session already exists
        if self._store.session_exists(session_id):
            raise SessionError(f"Session {session_id} already exists")

        # Create session
        session = Session(
            session_id=session_id,
            agent_name=agent_name,
            agent_config=agent_config,
        )

        # Attach store for automatic persistence
        session.attach_store(self._store)

        # Emit and persist start event
        start_event = create_session_started_event(
            session_id=session_id,
            agent_name=agent_name,
            agent_config=agent_config,
        )
        session.apply_event(start_event)

        return session

    def load_session(self, session_id: str) -> Session:
        """Load an existing session.

        Retrieves all events for the session and rebuilds state
        through event replay.

        Args:
            session_id: Unique identifier for the session.

        Returns:
            Session with state rebuilt from events.

        Raises:
            SessionNotFoundError: If the session does not exist.
        """
        # Get all events for the session
        events = self._store.get_events(session_id)

        if not events:
            raise SessionNotFoundError(f"Session {session_id} has no events")

        # Extract session info from first event
        first_event = events[0]
        agent_name = first_event.payload.get("agent_name")
        agent_config = first_event.payload.get("agent_config")

        # Create session and rebuild from events
        session = Session(
            session_id=session_id,
            created_at=first_event.timestamp,
            agent_name=agent_name,
            agent_config=agent_config,
        )
        session.attach_store(self._store)
        session.rebuild_from_events(events)

        return session

    def list_sessions(self) -> list[SessionSummary]:
        """List all sessions.

        Returns:
            List of SessionSummary objects for all sessions.
        """
        summaries = []
        for session_id in self._store.get_session_ids():
            try:
                metadata = self._store.get_session_metadata(session_id)

                # Determine status from events
                events = self._store.get_events(session_id)
                status = "active"
                agent_name = None

                for event in events:
                    if event.event_type == EventType.SESSION_STARTED:
                        agent_name = event.payload.get("agent_name")
                    elif event.event_type == EventType.SESSION_ENDED:
                        status = "ended"

                summaries.append(
                    SessionSummary(
                        session_id=metadata["session_id"],
                        created_at=metadata["created_at"],
                        updated_at=metadata["updated_at"],
                        event_count=metadata["event_count"],
                        status=status,
                        agent_name=agent_name,
                    )
                )
            except Exception:
                continue  # Skip corrupted sessions

        return summaries

    def archive_session(self, session_id: str) -> None:
        """Archive a session by marking it as complete.

        Emits a SessionEndedEvent with reason "archived".

        Args:
            session_id: Unique identifier for the session.

        Raises:
            SessionNotFoundError: If the session does not exist.
        """
        # Load the session to verify it exists
        session = self.load_session(session_id)

        if session.status == "ended":
            return  # Already ended

        # Emit end event
        end_event = create_session_ended_event(
            session_id=session_id,
            reason="archived",
            summary=session.get_context(),
        )
        session.apply_event(end_event)

    def delete_session(self, session_id: str) -> None:
        """Delete a session permanently.

        Warning: This operation is irreversible.

        Args:
            session_id: Unique identifier for the session.

        Raises:
            SessionNotFoundError: If the session does not exist.
        """
        self._store.delete_session(session_id)

    def get_session_summary(self, session_id: str) -> SessionSummary:
        """Get summary information about a specific session.

        Args:
            session_id: Unique identifier for the session.

        Returns:
            SessionSummary for the session.

        Raises:
            SessionNotFoundError: If the session does not exist.
        """
        metadata = self._store.get_session_metadata(session_id)
        events = self._store.get_events(session_id)

        status = "active"
        agent_name = None

        for event in events:
            if event.event_type == EventType.SESSION_STARTED:
                agent_name = event.payload.get("agent_name")
            elif event.event_type == EventType.SESSION_ENDED:
                status = "ended"

        return SessionSummary(
            session_id=metadata["session_id"],
            created_at=metadata["created_at"],
            updated_at=metadata["updated_at"],
            event_count=metadata["event_count"],
            status=status,
            agent_name=agent_name,
        )

    def session_exists(self, session_id: str) -> bool:
        """Check if a session exists.

        Args:
            session_id: Unique identifier for the session.

        Returns:
            True if the session exists, False otherwise.
        """
        return self._store.session_exists(session_id)
