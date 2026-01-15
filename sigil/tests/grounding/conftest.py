"""Pytest fixtures for grounding tests."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from sigil.grounding.grounder import Grounder
from sigil.grounding.schemas import GroundingRequest
from sigil.state.store import EventStore


@pytest.fixture
def event_store():
    """Create a test event store."""
    return EventStore()


@pytest.fixture
def mock_memory_manager():
    """Create a mock memory manager."""
    manager = AsyncMock()
    manager.retrieve = AsyncMock(return_value=[])
    return manager


@pytest.fixture
def grounder(event_store):
    """Create a grounder instance for testing."""
    return Grounder(event_store=event_store)


@pytest.fixture
def grounder_with_memory(event_store, mock_memory_manager):
    """Create a grounder with memory manager."""
    return Grounder(
        memory_manager=mock_memory_manager,
        event_store=event_store,
    )


@pytest.fixture
def grounder_strict(event_store):
    """Create a strict mode grounder."""
    return Grounder(
        event_store=event_store,
        strict_mode=True,
    )


@pytest.fixture
def simple_request():
    """Simple request with search intent."""
    return GroundingRequest(
        message="Search for latest AI news",
        session_id="test-session",
    )


@pytest.fixture
def ambiguous_request():
    """Request with ambiguous entity reference."""
    return GroundingRequest(
        message="Schedule a meeting with the team",
        session_id="test-session",
    )


@pytest.fixture
def complete_request():
    """Request with all information provided."""
    return GroundingRequest(
        message="Schedule a meeting with John Smith tomorrow at 3pm",
        session_id="test-session",
        context={"contact": "John Smith", "time": "3pm", "date": "tomorrow"},
    )


@pytest.fixture
def request_with_context():
    """Request with context information."""
    return GroundingRequest(
        message="Send an email to the client",
        session_id="test-session",
        context={
            "contact": "jane.doe@example.com",
            "to": "jane.doe@example.com",
        },
    )


@pytest.fixture
def request_with_memory():
    """Request with memory results."""
    return GroundingRequest(
        message="What do we know about the project?",
        session_id="test-session",
        memory_results=[
            {"content": "Project started in Q1 2026", "category": "projects"},
            {"content": "Budget is $50,000", "category": "finance"},
        ],
    )
