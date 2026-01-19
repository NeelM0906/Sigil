"""Tests for Sigil v2 event-sourced state management.

This module tests:
- Event creation and serialization
- Append-only store behavior
- Session rebuild from events
- Concurrent access safety
- Event querying by type/time

Test Categories:
    - TestEvent: Event dataclass and factory functions
    - TestEventStore: EventStore persistence and queries
    - TestSession: Session state management
    - TestSessionManager: Session lifecycle
    - TestConcurrentAccess: Thread safety tests
"""

import json
import os
import tempfile
import threading
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import pytest

from sigil.state import (
    Event,
    EventType,
    EventStore,
    EventStoreError,
    Session,
    SessionManager,
    SessionSummary,
    SessionNotFoundError,
    SessionError,
    SessionEndedError,
    create_session_started_event,
    create_session_ended_event,
    create_message_added_event,
    create_tool_called_event,
    create_plan_created_event,
    create_plan_step_completed_event,
    create_memory_extracted_event,
    create_contract_validated_event,
    create_error_occurred_event,
    events_to_json,
    events_from_json,
    filter_events_by_type,
    filter_events_by_session,
    filter_events_in_range,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def temp_storage_dir():
    """Create a temporary directory for test storage."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def event_store(temp_storage_dir):
    """Create an EventStore with temporary storage."""
    return EventStore(storage_dir=temp_storage_dir)


@pytest.fixture
def session_manager(temp_storage_dir):
    """Create a SessionManager with temporary storage."""
    return SessionManager(storage_dir=temp_storage_dir)


@pytest.fixture
def sample_session_id():
    """Generate a sample session ID."""
    return f"test-session-{uuid.uuid4().hex[:8]}"


# =============================================================================
# TestEvent: Event Creation and Serialization
# =============================================================================


class TestEvent:
    """Tests for Event dataclass and factory functions."""

    def test_event_creation(self, sample_session_id):
        """Test creating an Event with all fields."""
        now = datetime.now(timezone.utc)
        event = Event(
            event_id="test-event-123",
            event_type=EventType.SESSION_STARTED,
            timestamp=now,
            session_id=sample_session_id,
            correlation_id="corr-456",
            payload={"key": "value"},
        )

        assert event.event_id == "test-event-123"
        assert event.event_type == EventType.SESSION_STARTED
        assert event.timestamp == now
        assert event.session_id == sample_session_id
        assert event.correlation_id == "corr-456"
        assert event.payload == {"key": "value"}

    def test_event_immutability(self, sample_session_id):
        """Test that Event instances are frozen (immutable)."""
        event = create_session_started_event(sample_session_id)

        with pytest.raises(AttributeError):
            event.event_id = "new-id"  # type: ignore

        with pytest.raises(AttributeError):
            event.payload = {"new": "payload"}  # type: ignore

    def test_event_to_dict(self, sample_session_id):
        """Test serializing Event to dictionary."""
        event = create_session_started_event(
            session_id=sample_session_id,
            agent_name="TestAgent",
        )
        data = event.to_dict()

        assert data["event_id"] == event.event_id
        assert data["event_type"] == "session.started"
        assert "timestamp" in data
        assert data["session_id"] == sample_session_id
        assert data["payload"]["agent_name"] == "TestAgent"

    def test_event_to_json(self, sample_session_id):
        """Test serializing Event to JSON string."""
        event = create_session_started_event(sample_session_id)
        json_str = event.to_json()

        # Should be valid JSON
        data = json.loads(json_str)
        assert data["event_id"] == event.event_id

    def test_event_from_dict(self, sample_session_id):
        """Test deserializing Event from dictionary."""
        original = create_session_started_event(sample_session_id, agent_name="Test")
        data = original.to_dict()
        restored = Event.from_dict(data)

        assert restored.event_id == original.event_id
        assert restored.event_type == original.event_type
        assert restored.session_id == original.session_id
        assert restored.payload == original.payload

    def test_event_from_json(self, sample_session_id):
        """Test deserializing Event from JSON string."""
        original = create_session_started_event(sample_session_id)
        json_str = original.to_json()
        restored = Event.from_json(json_str)

        assert restored.event_id == original.event_id
        assert restored.event_type == original.event_type

    def test_event_roundtrip(self, sample_session_id):
        """Test full serialization/deserialization roundtrip."""
        original = create_tool_called_event(
            session_id=sample_session_id,
            tool_name="web_search",
            tool_args={"query": "test"},
            tool_result={"results": [1, 2, 3]},
            duration_ms=150.5,
        )

        # Dict roundtrip
        restored = Event.from_dict(original.to_dict())
        assert restored.event_id == original.event_id
        assert restored.payload["tool_name"] == "web_search"
        assert restored.payload["duration_ms"] == 150.5

        # JSON roundtrip
        restored_json = Event.from_json(original.to_json())
        assert restored_json.event_id == original.event_id


class TestEventFactories:
    """Tests for event factory functions."""

    def test_create_session_started_event(self, sample_session_id):
        """Test SessionStartedEvent factory."""
        event = create_session_started_event(
            session_id=sample_session_id,
            agent_name="LeadQualifier",
            agent_config={"model": "claude-opus-4-5-20251101"},
        )

        assert event.event_type == EventType.SESSION_STARTED
        assert event.session_id == sample_session_id
        assert event.payload["agent_name"] == "LeadQualifier"
        assert event.payload["agent_config"]["model"] == "claude-opus-4-5-20251101"

    def test_create_session_ended_event(self, sample_session_id):
        """Test SessionEndedEvent factory."""
        event = create_session_ended_event(
            session_id=sample_session_id,
            reason="completed",
            summary={"messages": 5, "tool_calls": 3},
        )

        assert event.event_type == EventType.SESSION_ENDED
        assert event.payload["reason"] == "completed"
        assert event.payload["summary"]["messages"] == 5

    def test_create_message_added_event(self, sample_session_id):
        """Test MessageAddedEvent factory."""
        event = create_message_added_event(
            session_id=sample_session_id,
            role="user",
            content="Hello, world!",
            metadata={"source": "cli"},
        )

        assert event.event_type == EventType.MESSAGE_ADDED
        assert event.payload["role"] == "user"
        assert event.payload["content"] == "Hello, world!"
        assert "message_id" in event.payload
        assert event.payload["metadata"]["source"] == "cli"

    def test_create_tool_called_event_success(self, sample_session_id):
        """Test ToolCalledEvent factory for successful call."""
        event = create_tool_called_event(
            session_id=sample_session_id,
            tool_name="web_search",
            tool_args={"query": "weather"},
            tool_result={"temp": "72F"},
            success=True,
            duration_ms=250.0,
        )

        assert event.event_type == EventType.TOOL_CALLED
        assert event.payload["tool_name"] == "web_search"
        assert event.payload["success"] is True
        assert event.payload["tool_result"]["temp"] == "72F"

    def test_create_tool_called_event_failure(self, sample_session_id):
        """Test ToolCalledEvent factory for failed call."""
        event = create_tool_called_event(
            session_id=sample_session_id,
            tool_name="api_call",
            tool_args={"endpoint": "/users"},
            success=False,
            error_message="Connection timeout",
        )

        assert event.payload["success"] is False
        assert event.payload["error_message"] == "Connection timeout"

    def test_create_plan_created_event(self, sample_session_id):
        """Test PlanCreatedEvent factory."""
        steps = [
            {"step_id": "1", "description": "Research"},
            {"step_id": "2", "description": "Analyze"},
            {"step_id": "3", "description": "Report"},
        ]
        event = create_plan_created_event(
            session_id=sample_session_id,
            plan_id="plan-123",
            goal="Complete research task",
            steps=steps,
        )

        assert event.event_type == EventType.PLAN_CREATED
        assert event.payload["plan_id"] == "plan-123"
        assert event.payload["step_count"] == 3
        assert len(event.payload["steps"]) == 3

    def test_create_plan_step_completed_event(self, sample_session_id):
        """Test PlanStepCompletedEvent factory."""
        event = create_plan_step_completed_event(
            session_id=sample_session_id,
            plan_id="plan-123",
            step_id="step-1",
            step_index=0,
            success=True,
            result={"data": "found"},
            duration_ms=1500.0,
        )

        assert event.event_type == EventType.PLAN_STEP_COMPLETED
        assert event.payload["step_index"] == 0
        assert event.payload["success"] is True

    def test_create_memory_extracted_event(self, sample_session_id):
        """Test MemoryExtractedEvent factory."""
        event = create_memory_extracted_event(
            session_id=sample_session_id,
            memory_id="mem-123",
            memory_type="fact",
            content="User prefers morning meetings",
            source="message-456",
            confidence=0.95,
        )

        assert event.event_type == EventType.MEMORY_EXTRACTED
        assert event.payload["memory_type"] == "fact"
        assert event.payload["confidence"] == 0.95

    def test_create_contract_validated_event_pass(self, sample_session_id):
        """Test ContractValidatedEvent factory for passing validation."""
        event = create_contract_validated_event(
            session_id=sample_session_id,
            contract_id="contract-123",
            contract_name="lead_qualification",
            passed=True,
            deliverables_checked=["score", "bant_assessment", "recommendation"],
        )

        assert event.event_type == EventType.CONTRACT_VALIDATED
        assert event.payload["passed"] is True
        assert len(event.payload["deliverables_checked"]) == 3

    def test_create_contract_validated_event_fail(self, sample_session_id):
        """Test ContractValidatedEvent factory for failing validation."""
        event = create_contract_validated_event(
            session_id=sample_session_id,
            contract_id="contract-123",
            contract_name="lead_qualification",
            passed=False,
            deliverables_checked=["score"],
            validation_errors=["Missing bant_assessment", "Invalid score range"],
            retry_count=1,
        )

        assert event.payload["passed"] is False
        assert len(event.payload["validation_errors"]) == 2
        assert event.payload["retry_count"] == 1

    def test_create_error_occurred_event(self, sample_session_id):
        """Test ErrorOccurredEvent factory."""
        event = create_error_occurred_event(
            session_id=sample_session_id,
            error_type="APIError",
            error_message="Rate limit exceeded",
            error_code="429",
            recoverable=True,
            context={"endpoint": "/v1/chat"},
        )

        assert event.event_type == EventType.ERROR_OCCURRED
        assert event.payload["error_type"] == "APIError"
        assert event.payload["recoverable"] is True


class TestEventUtilities:
    """Tests for event utility functions."""

    def test_events_to_json(self, sample_session_id):
        """Test serializing event list to JSON."""
        events = [
            create_session_started_event(sample_session_id),
            create_message_added_event(sample_session_id, "user", "Hello"),
        ]

        json_str = events_to_json(events)
        data = json.loads(json_str)

        assert len(data) == 2
        assert data[0]["event_type"] == "session.started"
        assert data[1]["event_type"] == "message.added"

    def test_events_from_json(self, sample_session_id):
        """Test deserializing event list from JSON."""
        original_events = [
            create_session_started_event(sample_session_id),
            create_message_added_event(sample_session_id, "user", "Hello"),
        ]

        json_str = events_to_json(original_events)
        restored = events_from_json(json_str)

        assert len(restored) == 2
        assert restored[0].event_type == EventType.SESSION_STARTED
        assert restored[1].event_type == EventType.MESSAGE_ADDED

    def test_filter_events_by_type(self, sample_session_id):
        """Test filtering events by type."""
        events = [
            create_session_started_event(sample_session_id),
            create_message_added_event(sample_session_id, "user", "Hello"),
            create_message_added_event(sample_session_id, "assistant", "Hi"),
            create_tool_called_event(sample_session_id, "search", {}),
        ]

        messages = filter_events_by_type(events, EventType.MESSAGE_ADDED)
        assert len(messages) == 2

        tools = filter_events_by_type(events, EventType.TOOL_CALLED)
        assert len(tools) == 1

    def test_filter_events_by_session(self):
        """Test filtering events by session ID."""
        events = [
            create_session_started_event("session-1"),
            create_message_added_event("session-1", "user", "Hello"),
            create_session_started_event("session-2"),
            create_message_added_event("session-2", "user", "Hi"),
        ]

        session1_events = filter_events_by_session(events, "session-1")
        assert len(session1_events) == 2
        assert all(e.session_id == "session-1" for e in session1_events)

    def test_filter_events_in_range(self, sample_session_id):
        """Test filtering events by time range."""
        base_time = datetime.now(timezone.utc)

        # Create events at different times
        with patch("sigil.state.events._get_utc_now") as mock_now:
            mock_now.return_value = base_time
            event1 = create_session_started_event(sample_session_id)

            mock_now.return_value = base_time + timedelta(hours=1)
            event2 = create_message_added_event(sample_session_id, "user", "Hello")

            mock_now.return_value = base_time + timedelta(hours=2)
            event3 = create_message_added_event(sample_session_id, "assistant", "Hi")

        events = [event1, event2, event3]

        # Filter to middle hour
        start = base_time + timedelta(minutes=30)
        end = base_time + timedelta(hours=1, minutes=30)
        filtered = filter_events_in_range(events, start, end)

        assert len(filtered) == 1
        assert filtered[0].event_type == EventType.MESSAGE_ADDED


# =============================================================================
# TestEventStore: Store Persistence and Queries
# =============================================================================


class TestEventStore:
    """Tests for EventStore persistence and queries."""

    def test_store_initialization(self, temp_storage_dir):
        """Test EventStore creates storage directory."""
        store = EventStore(storage_dir=temp_storage_dir)
        assert Path(temp_storage_dir).exists()

    def test_append_event(self, event_store, sample_session_id):
        """Test appending an event to the store."""
        event = create_session_started_event(sample_session_id)
        event_store.append(event)

        # Verify the event was stored
        events = event_store.get_events(sample_session_id)
        assert len(events) == 1
        assert events[0].event_id == event.event_id

    def test_append_multiple_events(self, event_store, sample_session_id):
        """Test appending multiple events."""
        events_to_add = [
            create_session_started_event(sample_session_id),
            create_message_added_event(sample_session_id, "user", "Hello"),
            create_message_added_event(sample_session_id, "assistant", "Hi there"),
        ]

        for event in events_to_add:
            event_store.append(event)

        stored = event_store.get_events(sample_session_id)
        assert len(stored) == 3

    def test_append_batch(self, event_store, sample_session_id):
        """Test appending a batch of events atomically."""
        events = [
            create_session_started_event(sample_session_id),
            create_message_added_event(sample_session_id, "user", "Hello"),
        ]

        event_store.append_batch(events)

        stored = event_store.get_events(sample_session_id)
        assert len(stored) == 2

    def test_append_batch_different_sessions_fails(self, event_store):
        """Test that batch append fails for different sessions."""
        events = [
            create_session_started_event("session-1"),
            create_session_started_event("session-2"),
        ]

        with pytest.raises(ValueError, match="same session"):
            event_store.append_batch(events)

    def test_get_events_nonexistent_session(self, event_store):
        """Test getting events for nonexistent session raises error."""
        with pytest.raises(SessionNotFoundError):
            event_store.get_events("nonexistent-session")

    def test_session_exists(self, event_store, sample_session_id):
        """Test checking if a session exists."""
        assert not event_store.session_exists(sample_session_id)

        event = create_session_started_event(sample_session_id)
        event_store.append(event)

        assert event_store.session_exists(sample_session_id)

    def test_get_events_by_type(self, event_store, sample_session_id):
        """Test querying events by type."""
        event_store.append(create_session_started_event(sample_session_id))
        event_store.append(create_message_added_event(sample_session_id, "user", "Hi"))
        event_store.append(create_message_added_event(sample_session_id, "assistant", "Hello"))
        event_store.append(create_tool_called_event(sample_session_id, "search", {}))

        messages = event_store.get_events_by_type(
            EventType.MESSAGE_ADDED, session_id=sample_session_id
        )
        assert len(messages) == 2

    def test_get_events_by_type_all_sessions(self, event_store):
        """Test querying events by type across all sessions."""
        session1 = "session-1"
        session2 = "session-2"

        event_store.append(create_session_started_event(session1))
        event_store.append(create_message_added_event(session1, "user", "Hi"))
        event_store.append(create_session_started_event(session2))
        event_store.append(create_message_added_event(session2, "user", "Hello"))

        all_messages = event_store.get_events_by_type(EventType.MESSAGE_ADDED)
        assert len(all_messages) == 2

    def test_get_events_in_range(self, event_store, sample_session_id):
        """Test querying events in a time range."""
        base_time = datetime.now(timezone.utc)

        # Create events at different times using patches
        with patch("sigil.state.events._get_utc_now") as mock_now:
            mock_now.return_value = base_time
            event_store.append(create_session_started_event(sample_session_id))

            mock_now.return_value = base_time + timedelta(hours=1)
            event_store.append(create_message_added_event(sample_session_id, "user", "Hi"))

            mock_now.return_value = base_time + timedelta(hours=2)
            event_store.append(create_message_added_event(sample_session_id, "assistant", "Hello"))

        # Query middle range
        start = base_time + timedelta(minutes=30)
        end = base_time + timedelta(hours=1, minutes=30)
        events = event_store.get_events_in_range(start, end, session_id=sample_session_id)

        assert len(events) == 1
        assert events[0].event_type == EventType.MESSAGE_ADDED

    def test_get_latest_events(self, event_store, sample_session_id):
        """Test getting the N most recent events."""
        for i in range(5):
            event_store.append(
                create_message_added_event(sample_session_id, "user", f"Message {i}")
            )
            time.sleep(0.01)  # Ensure distinct timestamps

        latest = event_store.get_latest_events(3, session_id=sample_session_id)
        assert len(latest) == 3
        # Should be newest first
        assert "Message 4" in latest[0].payload["content"]

    def test_get_session_ids(self, event_store):
        """Test listing all session IDs."""
        sessions = ["session-a", "session-b", "session-c"]
        for session_id in sessions:
            event_store.append(create_session_started_event(session_id))

        stored_ids = event_store.get_session_ids()
        assert set(stored_ids) == set(sessions)

    def test_get_session_metadata(self, event_store, sample_session_id):
        """Test getting session metadata without loading events."""
        event_store.append(create_session_started_event(sample_session_id))
        event_store.append(create_message_added_event(sample_session_id, "user", "Hi"))

        metadata = event_store.get_session_metadata(sample_session_id)

        assert metadata["session_id"] == sample_session_id
        assert metadata["event_count"] == 2
        assert "created_at" in metadata
        assert "updated_at" in metadata

    def test_delete_session(self, event_store, sample_session_id):
        """Test deleting a session."""
        event_store.append(create_session_started_event(sample_session_id))
        assert event_store.session_exists(sample_session_id)

        event_store.delete_session(sample_session_id)
        assert not event_store.session_exists(sample_session_id)

    def test_delete_nonexistent_session(self, event_store):
        """Test deleting a nonexistent session raises error."""
        with pytest.raises(SessionNotFoundError):
            event_store.delete_session("nonexistent")

    def test_get_event_count(self, event_store, sample_session_id):
        """Test getting total event count."""
        event_store.append(create_session_started_event(sample_session_id))
        event_store.append(create_message_added_event(sample_session_id, "user", "Hi"))

        assert event_store.get_event_count(sample_session_id) == 2

    def test_path_traversal_prevention(self, event_store):
        """Test that path traversal attempts are sanitized."""
        malicious_id = "../../../etc/passwd"
        event = create_session_started_event(malicious_id)
        event_store.append(event)

        # Should work with sanitized ID
        assert event_store.session_exists("______etc_passwd")


# =============================================================================
# TestSession: Session State Management
# =============================================================================


class TestSession:
    """Tests for Session state management."""

    def test_session_creation(self, sample_session_id):
        """Test creating a new session."""
        session = Session(
            session_id=sample_session_id,
            agent_name="TestAgent",
        )

        assert session.session_id == sample_session_id
        assert session.agent_name == "TestAgent"
        assert session.status == "active"
        assert len(session.messages) == 0

    def test_apply_message_event(self, sample_session_id):
        """Test applying a message event updates state."""
        session = Session(session_id=sample_session_id)

        event = create_message_added_event(
            session_id=sample_session_id,
            role="user",
            content="Hello, world!",
        )
        session.apply_event(event)

        assert len(session.messages) == 1
        assert session.messages[0]["role"] == "user"
        assert session.messages[0]["content"] == "Hello, world!"

    def test_apply_tool_called_event(self, sample_session_id):
        """Test applying a tool called event updates state."""
        session = Session(session_id=sample_session_id)

        event = create_tool_called_event(
            session_id=sample_session_id,
            tool_name="web_search",
            tool_args={"query": "test"},
            tool_result={"results": []},
        )
        session.apply_event(event)

        assert len(session.tool_calls) == 1
        assert session.tool_calls[0]["tool_name"] == "web_search"

    def test_apply_plan_events(self, sample_session_id):
        """Test applying plan events updates state."""
        session = Session(session_id=sample_session_id)

        # Create plan
        plan_event = create_plan_created_event(
            session_id=sample_session_id,
            plan_id="plan-123",
            goal="Test goal",
            steps=[{"step_id": "1", "description": "Step 1"}],
        )
        session.apply_event(plan_event)

        assert session.current_plan is not None
        assert session.current_plan["plan_id"] == "plan-123"
        assert session.current_plan["completed_steps"] == 0

        # Complete step
        step_event = create_plan_step_completed_event(
            session_id=sample_session_id,
            plan_id="plan-123",
            step_id="1",
            step_index=0,
            success=True,
        )
        session.apply_event(step_event)

        assert session.current_plan["completed_steps"] == 1

    def test_apply_memory_extracted_event(self, sample_session_id):
        """Test applying memory extracted event updates state."""
        session = Session(session_id=sample_session_id)

        event = create_memory_extracted_event(
            session_id=sample_session_id,
            memory_id="mem-123",
            memory_type="fact",
            content="Important fact",
        )
        session.apply_event(event)

        assert len(session.memory_items) == 1
        assert session.memory_items[0]["content"] == "Important fact"

    def test_apply_contract_validated_event(self, sample_session_id):
        """Test applying contract validation event updates state."""
        session = Session(session_id=sample_session_id)

        event = create_contract_validated_event(
            session_id=sample_session_id,
            contract_id="contract-123",
            contract_name="test_contract",
            passed=True,
            deliverables_checked=["item1", "item2"],
        )
        session.apply_event(event)

        assert len(session.contract_validations) == 1
        assert session.contract_validations[0]["passed"] is True

    def test_apply_error_event(self, sample_session_id):
        """Test applying error event updates state."""
        session = Session(session_id=sample_session_id)

        event = create_error_occurred_event(
            session_id=sample_session_id,
            error_type="TestError",
            error_message="Something went wrong",
        )
        session.apply_event(event)

        assert len(session.errors) == 1
        assert session.errors[0]["error_type"] == "TestError"

    def test_apply_session_ended_event(self, sample_session_id):
        """Test applying session ended event updates status."""
        session = Session(session_id=sample_session_id)

        event = create_session_ended_event(
            session_id=sample_session_id,
            reason="completed",
        )
        session.apply_event(event)

        assert session.status == "ended"

    def test_apply_event_wrong_session_fails(self, sample_session_id):
        """Test that applying event from wrong session fails."""
        session = Session(session_id=sample_session_id)

        event = create_message_added_event(
            session_id="different-session",
            role="user",
            content="Hello",
        )

        with pytest.raises(ValueError, match="doesn't match"):
            session.apply_event(event)

    def test_apply_event_to_ended_session_fails(self, sample_session_id):
        """Test that applying events to ended session fails."""
        session = Session(session_id=sample_session_id)

        # End the session
        end_event = create_session_ended_event(sample_session_id)
        session.apply_event(end_event)

        # Try to add message
        msg_event = create_message_added_event(sample_session_id, "user", "Hi")

        with pytest.raises(SessionEndedError):
            session.apply_event(msg_event)

    def test_rebuild_from_events(self, sample_session_id):
        """Test rebuilding session state from events."""
        # Create events
        events = [
            create_session_started_event(sample_session_id, agent_name="TestAgent"),
            create_message_added_event(sample_session_id, "user", "Hello"),
            create_message_added_event(sample_session_id, "assistant", "Hi there"),
            create_tool_called_event(sample_session_id, "search", {"q": "test"}),
        ]

        # Create session and rebuild
        session = Session(session_id=sample_session_id)
        session.rebuild_from_events(events)

        assert session.agent_name == "TestAgent"
        assert len(session.messages) == 2
        assert len(session.tool_calls) == 1

    def test_get_context(self, sample_session_id):
        """Test getting session context snapshot."""
        session = Session(session_id=sample_session_id, agent_name="TestAgent")

        event = create_message_added_event(sample_session_id, "user", "Hello")
        session.apply_event(event)

        context = session.get_context()

        assert context["session_id"] == sample_session_id
        assert context["agent_name"] == "TestAgent"
        assert context["message_count"] == 1
        assert context["status"] == "active"

    def test_get_conversation_history(self, sample_session_id):
        """Test getting conversation history for LLM."""
        session = Session(session_id=sample_session_id)

        session.apply_event(create_message_added_event(sample_session_id, "user", "Hello"))
        session.apply_event(create_message_added_event(sample_session_id, "assistant", "Hi"))

        history = session.get_conversation_history()

        assert len(history) == 2
        assert history[0] == {"role": "user", "content": "Hello"}
        assert history[1] == {"role": "assistant", "content": "Hi"}

    def test_events_property(self, sample_session_id):
        """Test that events property returns a copy."""
        session = Session(session_id=sample_session_id)

        event = create_message_added_event(sample_session_id, "user", "Hi")
        session.apply_event(event)

        events = session.events
        assert len(events) == 1

        # Modifying returned list should not affect session
        events.clear()
        assert len(session.events) == 1


# =============================================================================
# TestSessionManager: Session Lifecycle
# =============================================================================


class TestSessionManager:
    """Tests for SessionManager lifecycle management."""

    def test_create_session(self, session_manager):
        """Test creating a new session."""
        session = session_manager.create_session(agent_name="TestAgent")

        assert session.session_id.startswith("sess-")
        assert session.agent_name == "TestAgent"
        assert session.status == "active"
        # Should have SESSION_STARTED event
        assert len(session.events) == 1
        assert session.events[0].event_type == EventType.SESSION_STARTED

    def test_create_session_custom_id(self, session_manager):
        """Test creating a session with custom ID."""
        session = session_manager.create_session(session_id="custom-session-123")
        assert session.session_id == "custom-session-123"

    def test_create_duplicate_session_fails(self, session_manager):
        """Test that creating a duplicate session fails."""
        session_manager.create_session(session_id="my-session")

        with pytest.raises(SessionError, match="already exists"):
            session_manager.create_session(session_id="my-session")

    def test_load_session(self, session_manager):
        """Test loading an existing session."""
        # Create session and add events
        original = session_manager.create_session(agent_name="TestAgent")
        original.apply_event(
            create_message_added_event(original.session_id, "user", "Hello")
        )

        # Load the session
        loaded = session_manager.load_session(original.session_id)

        assert loaded.session_id == original.session_id
        assert loaded.agent_name == "TestAgent"
        assert len(loaded.messages) == 1
        assert loaded.messages[0]["content"] == "Hello"

    def test_load_nonexistent_session_fails(self, session_manager):
        """Test that loading nonexistent session fails."""
        with pytest.raises(SessionNotFoundError):
            session_manager.load_session("nonexistent-session")

    def test_list_sessions(self, session_manager):
        """Test listing all sessions."""
        # Create multiple sessions
        session_manager.create_session(agent_name="Agent1")
        session_manager.create_session(agent_name="Agent2")
        session_manager.create_session(agent_name="Agent3")

        sessions = session_manager.list_sessions()

        assert len(sessions) == 3
        assert all(isinstance(s, SessionSummary) for s in sessions)
        agent_names = {s.agent_name for s in sessions}
        assert agent_names == {"Agent1", "Agent2", "Agent3"}

    def test_archive_session(self, session_manager):
        """Test archiving a session."""
        session = session_manager.create_session()
        session_manager.archive_session(session.session_id)

        # Load and verify status
        loaded = session_manager.load_session(session.session_id)
        assert loaded.status == "ended"

    def test_archive_already_ended_session(self, session_manager):
        """Test archiving an already ended session is no-op."""
        session = session_manager.create_session()
        session_manager.archive_session(session.session_id)
        # Should not raise
        session_manager.archive_session(session.session_id)

    def test_delete_session(self, session_manager):
        """Test deleting a session."""
        session = session_manager.create_session()
        session_id = session.session_id

        session_manager.delete_session(session_id)

        assert not session_manager.session_exists(session_id)

    def test_get_session_summary(self, session_manager):
        """Test getting session summary."""
        session = session_manager.create_session(agent_name="TestAgent")
        session.apply_event(
            create_message_added_event(session.session_id, "user", "Hello")
        )

        summary = session_manager.get_session_summary(session.session_id)

        assert summary.session_id == session.session_id
        assert summary.agent_name == "TestAgent"
        assert summary.event_count == 2  # start + message
        assert summary.status == "active"

    def test_session_exists(self, session_manager):
        """Test checking if session exists."""
        assert not session_manager.session_exists("nonexistent")

        session = session_manager.create_session()
        assert session_manager.session_exists(session.session_id)


# =============================================================================
# TestConcurrentAccess: Thread Safety Tests
# =============================================================================


class TestConcurrentAccess:
    """Tests for concurrent access safety."""

    def test_concurrent_append_same_session(self, temp_storage_dir):
        """Test concurrent appends to the same session are safe."""
        store = EventStore(storage_dir=temp_storage_dir)
        session_id = "concurrent-test-session"

        # Create initial event
        store.append(create_session_started_event(session_id))

        num_threads = 10
        events_per_thread = 5
        errors = []

        def append_events(thread_id: int):
            try:
                for i in range(events_per_thread):
                    event = create_message_added_event(
                        session_id=session_id,
                        role="user",
                        content=f"Thread {thread_id}, Message {i}",
                    )
                    store.append(event)
            except Exception as e:
                errors.append(str(e))

        threads = [
            threading.Thread(target=append_events, args=(i,))
            for i in range(num_threads)
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Verify no errors occurred
        assert len(errors) == 0, f"Errors: {errors}"

        # Verify all events were stored
        events = store.get_events(session_id)
        # 1 start event + (num_threads * events_per_thread) message events
        expected_count = 1 + (num_threads * events_per_thread)
        assert len(events) == expected_count

    def test_concurrent_read_write(self, temp_storage_dir):
        """Test concurrent reads and writes are safe."""
        store = EventStore(storage_dir=temp_storage_dir)
        session_id = "read-write-test"

        # Create initial events
        store.append(create_session_started_event(session_id))

        num_writers = 5
        num_readers = 5
        events_per_writer = 10
        read_results = []
        errors = []

        def writer(thread_id: int):
            try:
                for i in range(events_per_writer):
                    event = create_message_added_event(
                        session_id=session_id,
                        role="user",
                        content=f"Writer {thread_id}, Message {i}",
                    )
                    store.append(event)
                    time.sleep(0.001)  # Small delay
            except Exception as e:
                errors.append(f"Writer {thread_id}: {e}")

        def reader(thread_id: int):
            try:
                for _ in range(events_per_writer):
                    events = store.get_events(session_id)
                    read_results.append(len(events))
                    time.sleep(0.001)
            except Exception as e:
                errors.append(f"Reader {thread_id}: {e}")

        threads = []
        threads.extend(threading.Thread(target=writer, args=(i,)) for i in range(num_writers))
        threads.extend(threading.Thread(target=reader, args=(i,)) for i in range(num_readers))

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Errors: {errors}"
        # All reads should have gotten some events
        assert all(count > 0 for count in read_results)

    def test_concurrent_different_sessions(self, temp_storage_dir):
        """Test concurrent access to different sessions is safe."""
        store = EventStore(storage_dir=temp_storage_dir)
        num_sessions = 10
        events_per_session = 5
        errors = []

        def work_on_session(session_id: str):
            try:
                store.append(create_session_started_event(session_id))
                for i in range(events_per_session):
                    event = create_message_added_event(
                        session_id=session_id,
                        role="user",
                        content=f"Message {i}",
                    )
                    store.append(event)
            except Exception as e:
                errors.append(f"{session_id}: {e}")

        threads = [
            threading.Thread(target=work_on_session, args=(f"session-{i}",))
            for i in range(num_sessions)
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Errors: {errors}"

        # Verify each session has correct event count
        for i in range(num_sessions):
            events = store.get_events(f"session-{i}")
            assert len(events) == 1 + events_per_session


# =============================================================================
# TestIntegration: End-to-End Tests
# =============================================================================


class TestIntegration:
    """End-to-end integration tests."""

    def test_full_session_workflow(self, session_manager):
        """Test a complete session workflow."""
        # Create session
        session = session_manager.create_session(agent_name="IntegrationTestAgent")

        # Simulate conversation
        session.apply_event(
            create_message_added_event(session.session_id, "user", "What's the weather?")
        )

        session.apply_event(
            create_tool_called_event(
                session_id=session.session_id,
                tool_name="weather_api",
                tool_args={"location": "San Francisco"},
                tool_result={"temp": "65F", "conditions": "sunny"},
            )
        )

        session.apply_event(
            create_message_added_event(
                session.session_id,
                "assistant",
                "The weather in San Francisco is 65F and sunny.",
            )
        )

        # Verify state
        assert len(session.messages) == 2
        assert len(session.tool_calls) == 1
        assert session.tool_calls[0]["tool_result"]["temp"] == "65F"

        # Archive session
        session_manager.archive_session(session.session_id)

        # Reload and verify
        loaded = session_manager.load_session(session.session_id)
        assert loaded.status == "ended"
        assert len(loaded.messages) == 2

    def test_session_persistence_across_managers(self, temp_storage_dir):
        """Test that sessions persist across manager instances."""
        # Create session with first manager
        manager1 = SessionManager(storage_dir=temp_storage_dir)
        session = manager1.create_session(agent_name="PersistenceTest")
        session_id = session.session_id

        session.apply_event(
            create_message_added_event(session_id, "user", "Remember this!")
        )

        # Load with new manager instance
        manager2 = SessionManager(storage_dir=temp_storage_dir)
        loaded = manager2.load_session(session_id)

        assert loaded.agent_name == "PersistenceTest"
        assert len(loaded.messages) == 1
        assert loaded.messages[0]["content"] == "Remember this!"

    def test_json_file_human_readable(self, temp_storage_dir):
        """Test that stored JSON files are human-readable."""
        manager = SessionManager(storage_dir=temp_storage_dir)
        session = manager.create_session(
            session_id="readable-test",
            agent_name="TestAgent",
        )

        session.apply_event(
            create_message_added_event(session.session_id, "user", "Hello!")
        )

        # Read the raw JSON file
        file_path = Path(temp_storage_dir) / "readable-test.json"
        with open(file_path) as f:
            content = f.read()
            data = json.loads(content)

        # Verify structure
        assert data["session_id"] == "readable-test"
        assert len(data["events"]) == 2
        assert data["events"][0]["event_type"] == "session.started"
        assert data["events"][1]["event_type"] == "message.added"
        assert data["events"][1]["payload"]["content"] == "Hello!"
