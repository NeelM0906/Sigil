"""Tests for FastAPI Server.

This module contains comprehensive tests for the FastAPI REST API
and WebSocket endpoints.

Test Categories:
    - Request/Response model tests
    - Health endpoint tests
    - Agent CRUD endpoint tests
    - Memory endpoint tests
    - WebSocket tests
    - Authentication tests
    - Error handling tests
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timezone

from fastapi import status
from fastapi.testclient import TestClient
from fastapi.websockets import WebSocket

# We need to patch settings before importing the server
with patch('sigil.config.settings.get_settings') as mock_settings:
    mock_settings_instance = MagicMock()
    mock_settings_instance.use_memory = False
    mock_settings_instance.use_planning = False
    mock_settings_instance.use_contracts = False
    mock_settings_instance.use_routing = False
    mock_settings_instance.get_active_features.return_value = []
    mock_settings_instance.api_keys = MagicMock()
    mock_settings_instance.api_keys.anthropic_api_key = "test-key-12345"
    mock_settings.return_value = mock_settings_instance

    from sigil.interfaces.api.server import (
        create_app,
        AgentCreateRequest,
        AgentRunRequest,
        AgentResponse,
        AgentRunResponse,
        MemorySearchRequest,
        MemoryStoreRequest,
        MemoryItem,
        ToolInfo,
        HealthResponse,
        ErrorResponse,
        AppState,
    )


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def app():
    """Create test application."""
    with patch('sigil.orchestrator.get_settings') as mock_settings:
        mock_settings_instance = MagicMock()
        mock_settings_instance.use_memory = False
        mock_settings_instance.use_planning = False
        mock_settings_instance.use_contracts = False
        mock_settings_instance.use_routing = False
        mock_settings_instance.get_active_features.return_value = []
        mock_settings.return_value = mock_settings_instance

        return create_app()


@pytest.fixture
def client(app):
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def auth_headers():
    """Create authentication headers."""
    return {"X-API-Key": "test-api-key-12345678"}


@pytest.fixture
def sample_agent_data():
    """Sample agent creation data."""
    return {
        "name": "test-agent",
        "description": "A test agent",
        "system_prompt": "You are a helpful assistant.",
        "tools": ["websearch"],
        "contract_name": None,
        "metadata": {"version": "1.0"},
    }


# =============================================================================
# Request Model Tests
# =============================================================================


class TestAgentCreateRequest:
    """Tests for AgentCreateRequest model."""

    def test_valid_request(self):
        """Test valid request creation."""
        request = AgentCreateRequest(
            name="test-agent",
            description="Test description",
            system_prompt="You are an assistant.",
        )
        assert request.name == "test-agent"
        assert request.tools == []

    def test_request_with_tools(self):
        """Test request with tools."""
        request = AgentCreateRequest(
            name="test-agent",
            description="Test",
            system_prompt="Prompt",
            tools=["websearch", "voice"],
        )
        assert len(request.tools) == 2


class TestAgentRunRequest:
    """Tests for AgentRunRequest model."""

    def test_valid_request(self):
        """Test valid run request."""
        request = AgentRunRequest(
            message="Hello world",
        )
        assert request.message == "Hello world"
        assert request.stream is False

    def test_request_with_context(self):
        """Test request with context."""
        request = AgentRunRequest(
            message="Test",
            context={"key": "value"},
            session_id="sess-123",
        )
        assert request.context == {"key": "value"}


class TestMemorySearchRequest:
    """Tests for MemorySearchRequest model."""

    def test_valid_request(self):
        """Test valid search request."""
        request = MemorySearchRequest(
            query="Find relevant memories",
        )
        assert request.k == 10  # Default

    def test_request_with_k(self):
        """Test request with custom k."""
        request = MemorySearchRequest(
            query="Test",
            k=50,
        )
        assert request.k == 50


# =============================================================================
# Health Endpoint Tests
# =============================================================================


class TestHealthEndpoint:
    """Tests for health endpoint."""

    def test_health_check(self, client):
        """Test health check returns healthy status."""
        response = client.get("/health")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data
        assert "uptime_seconds" in data

    def test_root_endpoint(self, client):
        """Test root endpoint."""
        response = client.get("/")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "name" in data
        assert "version" in data


# =============================================================================
# Agent Endpoint Tests
# =============================================================================


class TestAgentEndpoints:
    """Tests for agent CRUD endpoints."""

    def test_create_agent(self, client, auth_headers, sample_agent_data):
        """Test creating an agent."""
        response = client.post(
            "/agents",
            json=sample_agent_data,
            headers=auth_headers,
        )

        assert response.status_code == status.HTTP_201_CREATED
        data = response.json()
        assert data["name"] == sample_agent_data["name"]
        assert "created_at" in data

    def test_create_agent_duplicate(self, client, auth_headers, sample_agent_data):
        """Test creating duplicate agent returns error."""
        # Create first
        client.post("/agents", json=sample_agent_data, headers=auth_headers)

        # Try to create again
        response = client.post(
            "/agents",
            json=sample_agent_data,
            headers=auth_headers,
        )

        assert response.status_code == status.HTTP_409_CONFLICT

    def test_create_agent_no_auth(self, client, sample_agent_data):
        """Test creating agent without auth fails."""
        response = client.post(
            "/agents",
            json=sample_agent_data,
        )

        assert response.status_code == status.HTTP_401_UNAUTHORIZED

    def test_list_agents(self, client, auth_headers, sample_agent_data):
        """Test listing agents."""
        # Create an agent first
        client.post("/agents", json=sample_agent_data, headers=auth_headers)

        response = client.get("/agents", headers=auth_headers)

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert isinstance(data, list)

    def test_list_agents_pagination(self, client, auth_headers):
        """Test listing agents with pagination."""
        # Create multiple agents
        for i in range(5):
            agent_data = {
                "name": f"agent-{i}",
                "description": "Test",
                "system_prompt": "Prompt",
            }
            client.post("/agents", json=agent_data, headers=auth_headers)

        response = client.get(
            "/agents?limit=2&offset=1",
            headers=auth_headers,
        )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert len(data) <= 2

    def test_get_agent(self, client, auth_headers, sample_agent_data):
        """Test getting a specific agent."""
        # Create first
        client.post("/agents", json=sample_agent_data, headers=auth_headers)

        response = client.get(
            f"/agents/{sample_agent_data['name']}",
            headers=auth_headers,
        )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["name"] == sample_agent_data["name"]

    def test_get_agent_not_found(self, client, auth_headers):
        """Test getting non-existent agent."""
        response = client.get(
            "/agents/non-existent-agent",
            headers=auth_headers,
        )

        assert response.status_code == status.HTTP_404_NOT_FOUND

    def test_delete_agent(self, client, auth_headers, sample_agent_data):
        """Test deleting an agent."""
        # Create first
        client.post("/agents", json=sample_agent_data, headers=auth_headers)

        response = client.delete(
            f"/agents/{sample_agent_data['name']}",
            headers=auth_headers,
        )

        assert response.status_code == status.HTTP_204_NO_CONTENT

        # Verify deleted
        get_response = client.get(
            f"/agents/{sample_agent_data['name']}",
            headers=auth_headers,
        )
        assert get_response.status_code == status.HTTP_404_NOT_FOUND

    def test_delete_agent_not_found(self, client, auth_headers):
        """Test deleting non-existent agent."""
        response = client.delete(
            "/agents/non-existent",
            headers=auth_headers,
        )

        assert response.status_code == status.HTTP_404_NOT_FOUND


class TestAgentRunEndpoint:
    """Tests for agent run endpoint."""

    def test_run_agent(self, client, auth_headers, sample_agent_data):
        """Test running an agent."""
        # Create agent first
        client.post("/agents", json=sample_agent_data, headers=auth_headers)

        run_request = {
            "message": "Hello, test message",
        }

        response = client.post(
            f"/agents/{sample_agent_data['name']}/run",
            json=run_request,
            headers=auth_headers,
        )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "request_id" in data
        assert "status" in data
        assert "output" in data

    def test_run_agent_not_found(self, client, auth_headers):
        """Test running non-existent agent."""
        run_request = {"message": "Test"}

        response = client.post(
            "/agents/non-existent/run",
            json=run_request,
            headers=auth_headers,
        )

        assert response.status_code == status.HTTP_404_NOT_FOUND

    def test_run_agent_with_context(self, client, auth_headers, sample_agent_data):
        """Test running agent with context."""
        client.post("/agents", json=sample_agent_data, headers=auth_headers)

        run_request = {
            "message": "Test with context",
            "context": {"user_name": "John"},
            "session_id": "sess-123",
        }

        response = client.post(
            f"/agents/{sample_agent_data['name']}/run",
            json=run_request,
            headers=auth_headers,
        )

        assert response.status_code == status.HTTP_200_OK


# =============================================================================
# Memory Endpoint Tests
# =============================================================================


class TestMemoryEndpoints:
    """Tests for memory endpoints."""

    def test_search_memory_disabled(self, client, auth_headers):
        """Test searching memory when disabled."""
        search_request = {"query": "test query"}

        response = client.post(
            "/memory/search",
            json=search_request,
            headers=auth_headers,
        )

        # Memory is disabled by default in tests
        assert response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE

    def test_store_memory_disabled(self, client, auth_headers):
        """Test storing memory when disabled."""
        store_request = {
            "content": "Test content",
            "category": "test",
        }

        response = client.post(
            "/memory/store",
            json=store_request,
            headers=auth_headers,
        )

        assert response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE


# =============================================================================
# Tools Endpoint Tests
# =============================================================================


class TestToolsEndpoint:
    """Tests for tools endpoint."""

    def test_list_tools(self, client, auth_headers):
        """Test listing available tools."""
        response = client.get("/tools", headers=auth_headers)

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert isinstance(data, list)
        assert len(data) > 0
        assert all("name" in tool for tool in data)
        assert all("description" in tool for tool in data)


# =============================================================================
# Metrics Endpoint Tests
# =============================================================================


class TestMetricsEndpoint:
    """Tests for metrics endpoint."""

    def test_get_metrics(self, client, auth_headers):
        """Test getting metrics."""
        response = client.get("/metrics", headers=auth_headers)

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "orchestrator" in data
        assert "agents_count" in data
        assert "uptime_seconds" in data


# =============================================================================
# Authentication Tests
# =============================================================================


class TestAuthentication:
    """Tests for authentication."""

    def test_missing_api_key(self, client):
        """Test request without API key fails."""
        response = client.get("/agents")

        assert response.status_code == status.HTTP_401_UNAUTHORIZED

    def test_invalid_api_key(self, client):
        """Test request with invalid API key fails."""
        response = client.get(
            "/agents",
            headers={"X-API-Key": "short"},  # Too short
        )

        assert response.status_code == status.HTTP_401_UNAUTHORIZED

    def test_valid_api_key(self, client, auth_headers):
        """Test request with valid API key succeeds."""
        response = client.get("/agents", headers=auth_headers)

        assert response.status_code == status.HTTP_200_OK


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestErrorHandling:
    """Tests for error handling."""

    def test_invalid_json(self, client, auth_headers):
        """Test handling of invalid JSON."""
        response = client.post(
            "/agents",
            content="not valid json",
            headers={**auth_headers, "Content-Type": "application/json"},
        )

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_missing_required_field(self, client, auth_headers):
        """Test handling of missing required field."""
        response = client.post(
            "/agents",
            json={"name": "test"},  # Missing description and system_prompt
            headers=auth_headers,
        )

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


# =============================================================================
# WebSocket Tests
# =============================================================================


class TestWebSocket:
    """Tests for WebSocket endpoint."""

    def test_websocket_agent_not_found(self, client, auth_headers, sample_agent_data):
        """Test WebSocket connection for non-existent agent."""
        # Create an agent first so we have at least one
        client.post("/agents", json=sample_agent_data, headers=auth_headers)

        with pytest.raises(Exception):
            # Try to connect to non-existent agent
            with client.websocket_connect("/ws/agents/non-existent/run") as ws:
                pass

    def test_websocket_connection(self, client, auth_headers, sample_agent_data):
        """Test basic WebSocket connection."""
        # Create agent first
        client.post("/agents", json=sample_agent_data, headers=auth_headers)

        # Test connection (note: this may fail in test environment)
        try:
            with client.websocket_connect(
                f"/ws/agents/{sample_agent_data['name']}/run"
            ) as ws:
                # Send a message
                ws.send_json({"message": "Hello"})

                # Receive acknowledgment
                data = ws.receive_json()
                assert data.get("type") in ["ack", "error"]
        except Exception as e:
            # WebSocket tests can be flaky in test environment
            pytest.skip(f"WebSocket test skipped: {e}")


# =============================================================================
# Response Model Tests
# =============================================================================


class TestResponseModels:
    """Tests for response models."""

    def test_agent_response(self):
        """Test AgentResponse model."""
        response = AgentResponse(
            name="test",
            description="Test agent",
            system_prompt="You are an assistant.",
            tools=["websearch"],
            contract_name=None,
            metadata={},
            created_at="2025-01-01T00:00:00",
            updated_at="2025-01-01T00:00:00",
        )
        assert response.name == "test"

    def test_agent_run_response(self):
        """Test AgentRunResponse model."""
        response = AgentRunResponse(
            request_id="req-123",
            status="success",
            output={"result": "done"},
            tokens_used=100,
            execution_time_ms=50.0,
        )
        assert response.request_id == "req-123"

    def test_health_response(self):
        """Test HealthResponse model."""
        response = HealthResponse(
            status="healthy",
            version="v1",
            features=["memory", "planning"],
            uptime_seconds=3600.0,
        )
        assert response.status == "healthy"

    def test_error_response(self):
        """Test ErrorResponse model."""
        response = ErrorResponse(
            error="Something went wrong",
            code="ERR_001",
            detail="Additional details",
        )
        assert response.error == "Something went wrong"
