"""Tests for Layer 1: ResourceLayer.

Tests cover:
- CRUD operations (create, read, update, delete)
- Full-text search
- Conversation resource handling
- Concurrent access safety
- Edge cases and error handling
"""

import pytest
import tempfile
import shutil
import threading
import time
from pathlib import Path
from datetime import datetime, timezone

from sigil.memory.layers.resources import (
    ResourceLayer,
    ConversationResource,
    VALID_RESOURCE_TYPES,
)
from sigil.config.schemas.memory import Resource


@pytest.fixture
def temp_storage_dir():
    """Create a temporary storage directory for tests."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def resource_layer(temp_storage_dir):
    """Create a ResourceLayer instance with temporary storage."""
    return ResourceLayer(storage_dir=temp_storage_dir)


class TestResourceLayerCRUD:
    """Tests for basic CRUD operations."""

    def test_store_resource(self, resource_layer):
        """Test storing a new resource."""
        resource = resource_layer.store(
            resource_type="conversation",
            content="User: Hello!\nAgent: Hi there!",
            metadata={"session_id": "test-123"},
        )

        assert resource is not None
        assert resource.resource_id is not None
        assert resource.resource_type == "conversation"
        assert "Hello" in resource.content
        assert resource.metadata["session_id"] == "test-123"

    def test_store_with_custom_id(self, resource_layer):
        """Test storing a resource with a custom ID."""
        custom_id = "my-custom-id-123"
        resource = resource_layer.store(
            resource_type="document",
            content="Some document content",
            resource_id=custom_id,
        )

        assert resource.resource_id == custom_id

    def test_get_resource(self, resource_layer):
        """Test retrieving a resource by ID."""
        stored = resource_layer.store(
            resource_type="config",
            content="key=value",
        )

        retrieved = resource_layer.get(stored.resource_id)

        assert retrieved is not None
        assert retrieved.resource_id == stored.resource_id
        assert retrieved.content == stored.content

    def test_get_nonexistent_resource(self, resource_layer):
        """Test retrieving a non-existent resource returns None."""
        result = resource_layer.get("nonexistent-id")
        assert result is None

    def test_delete_resource(self, resource_layer):
        """Test deleting a resource."""
        stored = resource_layer.store(
            resource_type="feedback",
            content="Great product!",
        )

        assert resource_layer.exists(stored.resource_id)
        deleted = resource_layer.delete(stored.resource_id)
        assert deleted is True
        assert not resource_layer.exists(stored.resource_id)

    def test_delete_nonexistent_resource(self, resource_layer):
        """Test deleting a non-existent resource returns False."""
        result = resource_layer.delete("nonexistent-id")
        assert result is False

    def test_exists(self, resource_layer):
        """Test checking if a resource exists."""
        stored = resource_layer.store(
            resource_type="document",
            content="Content",
        )

        assert resource_layer.exists(stored.resource_id)
        assert not resource_layer.exists("nonexistent")


class TestResourceTypes:
    """Tests for resource type handling."""

    def test_valid_resource_types(self, resource_layer):
        """Test storing resources of all valid types."""
        for rtype in VALID_RESOURCE_TYPES:
            resource = resource_layer.store(
                resource_type=rtype,
                content=f"Content for {rtype}",
            )
            assert resource.resource_type == rtype

    def test_invalid_resource_type(self, resource_layer):
        """Test that invalid resource types raise ValueError."""
        with pytest.raises(ValueError) as exc_info:
            resource_layer.store(
                resource_type="invalid_type",
                content="Content",
            )
        assert "Invalid resource type" in str(exc_info.value)

    def test_resource_type_normalization(self, resource_layer):
        """Test that resource types are normalized to lowercase."""
        resource = resource_layer.store(
            resource_type="CONVERSATION",
            content="Test",
        )
        assert resource.resource_type == "conversation"


class TestListAndSearch:
    """Tests for listing and searching resources."""

    def test_list_by_type(self, resource_layer):
        """Test listing resources by type."""
        # Store resources of different types
        resource_layer.store(resource_type="conversation", content="Conv 1")
        resource_layer.store(resource_type="conversation", content="Conv 2")
        resource_layer.store(resource_type="document", content="Doc 1")

        convs = resource_layer.list_by_type("conversation")
        docs = resource_layer.list_by_type("document")

        assert len(convs) == 2
        assert len(docs) == 1

    def test_list_by_type_with_limit(self, resource_layer):
        """Test listing with limit parameter."""
        for i in range(5):
            resource_layer.store(resource_type="document", content=f"Doc {i}")

        limited = resource_layer.list_by_type("document", limit=3)
        assert len(limited) == 3

    def test_list_recent(self, resource_layer):
        """Test listing most recent resources."""
        for i in range(5):
            resource_layer.store(resource_type="feedback", content=f"Feedback {i}")
            time.sleep(0.01)  # Small delay to ensure ordering

        recent = resource_layer.list_recent(limit=3)
        assert len(recent) == 3

    def test_search(self, resource_layer):
        """Test full-text search."""
        resource_layer.store(
            resource_type="document",
            content="Product pricing information",
        )
        resource_layer.store(
            resource_type="document",
            content="User guide documentation",
        )
        resource_layer.store(
            resource_type="document",
            content="Pricing tiers and plans",
        )

        results = resource_layer.search("pricing")
        assert len(results) == 2

    def test_search_case_insensitive(self, resource_layer):
        """Test case-insensitive search."""
        resource_layer.store(
            resource_type="document",
            content="IMPORTANT Information",
        )

        results = resource_layer.search("important", case_sensitive=False)
        assert len(results) == 1

    def test_search_in_metadata(self, resource_layer):
        """Test search in metadata."""
        resource_layer.store(
            resource_type="document",
            content="Generic content",
            metadata={"tags": ["pricing", "sales"]},
        )

        results = resource_layer.search("pricing")
        assert len(results) == 1

    def test_count(self, resource_layer):
        """Test counting resources."""
        resource_layer.store(resource_type="conversation", content="Conv")
        resource_layer.store(resource_type="document", content="Doc 1")
        resource_layer.store(resource_type="document", content="Doc 2")

        assert resource_layer.count() == 3
        assert resource_layer.count("conversation") == 1
        assert resource_layer.count("document") == 2


class TestConversationResource:
    """Tests for ConversationResource class."""

    def test_create_conversation(self, resource_layer):
        """Test creating a conversation resource."""
        conv = resource_layer.store_conversation(
            content="User: Hello!\nAgent: Hi there!",
            metadata={"session_id": "test"},
        )

        assert conv is not None
        assert conv.turn_count == 2

    def test_parse_turns(self):
        """Test parsing conversation turns."""
        resource = Resource(
            resource_type="conversation",
            content="User: Question?\nAssistant: Answer.\nUser: Thanks!",
        )
        conv = ConversationResource.from_resource(resource)

        assert conv.turn_count == 3
        turns = conv.turns
        assert turns[0] == ("user", "Question?")
        assert turns[1] == ("assistant", "Answer.")
        assert turns[2] == ("user", "Thanks!")

    def test_add_turn(self):
        """Test adding turns to a conversation."""
        conv = ConversationResource.create_new()
        conv.add_turn("user", "Hello!")
        conv.add_turn("assistant", "Hi there!")

        assert conv.turn_count == 2
        assert conv.turns[0] == ("user", "Hello!")

    def test_get_turns_by_role(self):
        """Test filtering turns by role."""
        resource = Resource(
            resource_type="conversation",
            content="User: Q1\nAssistant: A1\nUser: Q2\nAssistant: A2",
        )
        conv = ConversationResource.from_resource(resource)

        user_messages = conv.get_turns_by_role("user")
        assert len(user_messages) == 2
        assert "Q1" in user_messages[0]

    def test_get_last_n_turns(self):
        """Test getting last N turns."""
        resource = Resource(
            resource_type="conversation",
            content="User: 1\nAssistant: 2\nUser: 3\nAssistant: 4",
        )
        conv = ConversationResource.from_resource(resource)

        last_2 = conv.get_last_n_turns(2)
        assert len(last_2) == 2
        assert last_2[0][1] == "3"

    def test_invalid_resource_type(self):
        """Test that non-conversation resources raise ValueError."""
        resource = Resource(
            resource_type="document",
            content="Not a conversation",
        )

        with pytest.raises(ValueError):
            ConversationResource.from_resource(resource)


class TestConcurrentAccess:
    """Tests for thread-safe concurrent access."""

    def test_concurrent_writes(self, resource_layer):
        """Test concurrent write operations."""
        errors = []

        def write_resources(thread_id):
            try:
                for i in range(10):
                    resource_layer.store(
                        resource_type="document",
                        content=f"Thread {thread_id} - Resource {i}",
                    )
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=write_resources, args=(i,))
            for i in range(5)
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert resource_layer.count("document") == 50

    def test_concurrent_read_write(self, resource_layer):
        """Test concurrent read and write operations."""
        # Pre-populate
        ids = []
        for i in range(20):
            r = resource_layer.store(
                resource_type="document",
                content=f"Resource {i}",
            )
            ids.append(r.resource_id)

        errors = []

        def read_resources():
            try:
                for rid in ids:
                    resource_layer.get(rid)
            except Exception as e:
                errors.append(e)

        def write_resources():
            try:
                for i in range(10):
                    resource_layer.store(
                        resource_type="feedback",
                        content=f"New {i}",
                    )
            except Exception as e:
                errors.append(e)

        threads = []
        for _ in range(3):
            threads.append(threading.Thread(target=read_resources))
            threads.append(threading.Thread(target=write_resources))

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0


class TestIterator:
    """Tests for the iterator functionality."""

    def test_iter_all(self, resource_layer):
        """Test iterating over all resources."""
        resource_layer.store(resource_type="conversation", content="Conv")
        resource_layer.store(resource_type="document", content="Doc")

        resources = list(resource_layer.iter_all())
        assert len(resources) == 2

    def test_iter_by_type(self, resource_layer):
        """Test iterating with type filter."""
        resource_layer.store(resource_type="conversation", content="Conv")
        resource_layer.store(resource_type="document", content="Doc")

        convs = list(resource_layer.iter_all(resource_type="conversation"))
        assert len(convs) == 1
        assert convs[0].resource_type == "conversation"
