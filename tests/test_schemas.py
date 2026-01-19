"""Tests for ACTi Agent Builder schemas.

This module tests the Pydantic schemas defined in src/schemas.py:
- Stratum enum values and validation
- AgentConfig validation (valid and invalid cases)
- MCP_TOOL_CATEGORIES constant
- CreateAgentRequest and RunAgentRequest schemas
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.schemas import (
    AgentConfig,
    AgentResponse,
    CreateAgentRequest,
    CreateAgentResponse,
    MCP_TOOL_CATEGORIES,
    RunAgentRequest,
    Stratum,
    ToolInfo,
    ToolListResponse,
)


# -----------------------------------------------------------------------------
# Stratum Enum Tests
# -----------------------------------------------------------------------------

class TestStratumEnum:
    """Tests for the Stratum enum class."""

    def test_stratum_has_five_values(self):
        """Verify Stratum enum has exactly five strata."""
        assert len(Stratum) == 5

    def test_stratum_rti_value(self):
        """Verify RTI stratum has correct value."""
        assert Stratum.RTI.value == "RTI"

    def test_stratum_rai_value(self):
        """Verify RAI stratum has correct value."""
        assert Stratum.RAI.value == "RAI"

    def test_stratum_zacs_value(self):
        """Verify ZACS stratum has correct value."""
        assert Stratum.ZACS.value == "ZACS"

    def test_stratum_eei_value(self):
        """Verify EEI stratum has correct value."""
        assert Stratum.EEI.value == "EEI"

    def test_stratum_ige_value(self):
        """Verify IGE stratum has correct value."""
        assert Stratum.IGE.value == "IGE"

    def test_stratum_is_string_enum(self, all_strata: list[Stratum]):
        """Verify all Stratum values are strings."""
        for stratum in all_strata:
            assert isinstance(stratum.value, str)

    def test_stratum_from_string(self):
        """Verify Stratum can be created from string value."""
        assert Stratum("RTI") == Stratum.RTI
        assert Stratum("RAI") == Stratum.RAI
        assert Stratum("ZACS") == Stratum.ZACS

    def test_stratum_invalid_value_raises(self):
        """Verify invalid stratum values raise ValueError."""
        with pytest.raises(ValueError):
            Stratum("INVALID")


# -----------------------------------------------------------------------------
# MCP_TOOL_CATEGORIES Tests
# -----------------------------------------------------------------------------

class TestMCPToolCategories:
    """Tests for the MCP_TOOL_CATEGORIES constant."""

    def test_mcp_tool_categories_is_dict(self):
        """Verify MCP_TOOL_CATEGORIES is a dictionary."""
        assert isinstance(MCP_TOOL_CATEGORIES, dict)

    def test_mcp_tool_categories_has_five_categories(self):
        """Verify MCP_TOOL_CATEGORIES has exactly five tool categories."""
        assert len(MCP_TOOL_CATEGORIES) == 5

    def test_mcp_tool_categories_contains_voice(self):
        """Verify voice tool category exists."""
        assert "voice" in MCP_TOOL_CATEGORIES
        assert "ElevenLabs" in MCP_TOOL_CATEGORIES["voice"]

    def test_mcp_tool_categories_contains_websearch(self):
        """Verify websearch tool category exists."""
        assert "websearch" in MCP_TOOL_CATEGORIES
        assert "Tavily" in MCP_TOOL_CATEGORIES["websearch"]

    def test_mcp_tool_categories_contains_calendar(self):
        """Verify calendar tool category exists."""
        assert "calendar" in MCP_TOOL_CATEGORIES
        assert "Google Calendar" in MCP_TOOL_CATEGORIES["calendar"]

    def test_mcp_tool_categories_contains_communication(self):
        """Verify communication tool category exists."""
        assert "communication" in MCP_TOOL_CATEGORIES
        assert "Twilio" in MCP_TOOL_CATEGORIES["communication"]

    def test_mcp_tool_categories_contains_crm(self):
        """Verify crm tool category exists."""
        assert "crm" in MCP_TOOL_CATEGORIES
        assert "HubSpot" in MCP_TOOL_CATEGORIES["crm"]

    def test_mcp_tool_categories_all_values_are_strings(self):
        """Verify all tool category descriptions are strings."""
        for key, value in MCP_TOOL_CATEGORIES.items():
            assert isinstance(key, str)
            assert isinstance(value, str)


# -----------------------------------------------------------------------------
# AgentConfig Validation Tests - Valid Cases
# -----------------------------------------------------------------------------

class TestAgentConfigValid:
    """Tests for valid AgentConfig creation."""

    def test_create_with_all_fields(self, valid_agent_config: AgentConfig):
        """Verify AgentConfig can be created with all fields."""
        assert valid_agent_config.name == "test_agent"
        assert valid_agent_config.stratum == Stratum.RTI
        assert valid_agent_config.tools == ["websearch", "crm"]

    def test_create_with_minimal_fields(self, valid_agent_config_minimal: AgentConfig):
        """Verify AgentConfig can be created with only required fields."""
        assert valid_agent_config_minimal.name == "minimal_agent"
        assert valid_agent_config_minimal.tools == []
        assert valid_agent_config_minimal.stratum is None

    def test_create_from_dict(self, valid_agent_config_dict: dict):
        """Verify AgentConfig can be created from dictionary."""
        config = AgentConfig(**valid_agent_config_dict)
        assert config.name == "dict_agent"
        assert config.stratum == Stratum.ZACS

    def test_name_with_numbers(self):
        """Verify names with numbers (not leading) are valid."""
        config = AgentConfig(
            name="agent_v2",
            description="Test agent with version number in name",
            system_prompt="A valid system prompt that meets the minimum length requirement.",
        )
        assert config.name == "agent_v2"

    def test_name_with_underscores(self):
        """Verify names with underscores are valid."""
        config = AgentConfig(
            name="my_test_agent_name",
            description="Test agent with multiple underscores",
            system_prompt="A valid system prompt that meets the minimum length requirement.",
        )
        assert config.name == "my_test_agent_name"

    def test_name_single_letter(self):
        """Verify single letter names are valid."""
        config = AgentConfig(
            name="a",
            description="Test agent with single letter name",
            system_prompt="A valid system prompt that meets the minimum length requirement.",
        )
        assert config.name == "a"

    def test_name_max_length(self):
        """Verify names at max length (64) are valid."""
        config = AgentConfig(
            name="a" * 64,
            description="Test agent with max length name",
            system_prompt="A valid system prompt that meets the minimum length requirement.",
        )
        assert len(config.name) == 64

    def test_description_min_length(self):
        """Verify descriptions at min length (10) are valid."""
        config = AgentConfig(
            name="test_agent",
            description="Exactly 10",  # exactly 10 chars
            system_prompt="A valid system prompt that meets the minimum length requirement.",
        )
        assert len(config.description) == 10

    def test_description_max_length(self):
        """Verify descriptions at max length (500) are valid."""
        config = AgentConfig(
            name="test_agent",
            description="x" * 500,
            system_prompt="A valid system prompt that meets the minimum length requirement.",
        )
        assert len(config.description) == 500

    def test_system_prompt_min_length(self):
        """Verify system prompts at min length (50) are valid."""
        prompt = "x" * 50
        config = AgentConfig(
            name="test_agent",
            description="Valid test description",
            system_prompt=prompt,
        )
        assert len(config.system_prompt) == 50

    def test_all_valid_tools(self):
        """Verify all valid tool combinations work."""
        all_tools = list(MCP_TOOL_CATEGORIES.keys())
        config = AgentConfig(
            name="test_agent",
            description="Test agent with all tools",
            system_prompt="A valid system prompt that meets the minimum length requirement.",
            tools=all_tools,
        )
        assert set(config.tools) == set(all_tools)

    def test_empty_tools_list(self):
        """Verify empty tools list is valid."""
        config = AgentConfig(
            name="test_agent",
            description="Test agent with no tools",
            system_prompt="A valid system prompt that meets the minimum length requirement.",
            tools=[],
        )
        assert config.tools == []

    def test_default_model(self):
        """Verify default model is set correctly."""
        config = AgentConfig(
            name="test_agent",
            description="Test agent checking default model",
            system_prompt="A valid system prompt that meets the minimum length requirement.",
        )
        assert config.model == "anthropic:claude-opus-4-5-20251101"

    def test_custom_model(self):
        """Verify custom model can be set."""
        config = AgentConfig(
            name="test_agent",
            description="Test agent with custom model",
            system_prompt="A valid system prompt that meets the minimum length requirement.",
            model="openai:gpt-4",
        )
        assert config.model == "openai:gpt-4"

    def test_stratum_optional(self):
        """Verify stratum is optional (None by default)."""
        config = AgentConfig(
            name="test_agent",
            description="Test agent without stratum",
            system_prompt="A valid system prompt that meets the minimum length requirement.",
        )
        assert config.stratum is None


# -----------------------------------------------------------------------------
# AgentConfig Validation Tests - Invalid Cases
# -----------------------------------------------------------------------------

class TestAgentConfigInvalid:
    """Tests for invalid AgentConfig rejection."""

    def test_name_uppercase_rejected(self, invalid_config_bad_name_uppercase: dict):
        """Verify uppercase names are rejected."""
        with pytest.raises(ValidationError) as exc_info:
            AgentConfig(**invalid_config_bad_name_uppercase)
        assert "snake_case" in str(exc_info.value).lower()

    def test_name_starts_with_number_rejected(
        self, invalid_config_bad_name_starts_with_number: dict
    ):
        """Verify names starting with numbers are rejected."""
        with pytest.raises(ValidationError) as exc_info:
            AgentConfig(**invalid_config_bad_name_starts_with_number)
        assert "snake_case" in str(exc_info.value).lower()

    def test_name_special_chars_rejected(
        self, invalid_config_bad_name_special_chars: dict
    ):
        """Verify names with special characters (dashes) are rejected."""
        with pytest.raises(ValidationError) as exc_info:
            AgentConfig(**invalid_config_bad_name_special_chars)
        assert "snake_case" in str(exc_info.value).lower()

    def test_name_too_long_rejected(self, invalid_config_bad_name_too_long: dict):
        """Verify names exceeding max length are rejected."""
        with pytest.raises(ValidationError):
            AgentConfig(**invalid_config_bad_name_too_long)

    def test_name_empty_rejected(self):
        """Verify empty names are rejected."""
        with pytest.raises(ValidationError):
            AgentConfig(
                name="",
                description="Test agent with empty name",
                system_prompt="A valid system prompt that meets the minimum length requirement.",
            )

    def test_description_too_short_rejected(
        self, invalid_config_description_too_short: dict
    ):
        """Verify descriptions below minimum length are rejected."""
        with pytest.raises(ValidationError):
            AgentConfig(**invalid_config_description_too_short)

    def test_description_too_long_rejected(
        self, invalid_config_description_too_long: dict
    ):
        """Verify descriptions exceeding max length are rejected."""
        with pytest.raises(ValidationError):
            AgentConfig(**invalid_config_description_too_long)

    def test_system_prompt_too_short_rejected(
        self, invalid_config_system_prompt_too_short: dict
    ):
        """Verify system prompts below minimum length are rejected."""
        with pytest.raises(ValidationError):
            AgentConfig(**invalid_config_system_prompt_too_short)

    def test_invalid_tool_rejected(self, invalid_config_invalid_tool: dict):
        """Verify unrecognized tools are rejected."""
        with pytest.raises(ValidationError) as exc_info:
            AgentConfig(**invalid_config_invalid_tool)
        assert "fake_tool" in str(exc_info.value)

    def test_invalid_stratum_rejected(self, invalid_config_invalid_stratum: dict):
        """Verify invalid stratum values are rejected."""
        with pytest.raises(ValidationError):
            AgentConfig(**invalid_config_invalid_stratum)

    def test_missing_name_rejected(self):
        """Verify missing name field is rejected."""
        with pytest.raises(ValidationError):
            AgentConfig(
                description="Test agent missing name",
                system_prompt="A valid system prompt that meets the minimum length requirement.",
            )

    def test_missing_description_rejected(self):
        """Verify missing description field is rejected."""
        with pytest.raises(ValidationError):
            AgentConfig(
                name="test_agent",
                system_prompt="A valid system prompt that meets the minimum length requirement.",
            )

    def test_missing_system_prompt_rejected(self):
        """Verify missing system_prompt field is rejected."""
        with pytest.raises(ValidationError):
            AgentConfig(
                name="test_agent",
                description="Test agent missing system prompt",
            )

    def test_name_with_spaces_rejected(self):
        """Verify names with spaces are rejected."""
        with pytest.raises(ValidationError):
            AgentConfig(
                name="bad name",
                description="Test agent with spaces in name",
                system_prompt="A valid system prompt that meets the minimum length requirement.",
            )


# -----------------------------------------------------------------------------
# CreateAgentRequest Tests
# -----------------------------------------------------------------------------

class TestCreateAgentRequest:
    """Tests for CreateAgentRequest schema."""

    def test_valid_request_minimal(self):
        """Verify CreateAgentRequest with only required fields."""
        request = CreateAgentRequest(
            prompt="Create an agent that qualifies leads for B2B sales"
        )
        assert len(request.prompt) >= 10
        assert request.preferred_stratum is None

    def test_valid_request_with_stratum(self):
        """Verify CreateAgentRequest with preferred stratum."""
        request = CreateAgentRequest(
            prompt="Create a scheduling agent for appointments",
            preferred_stratum=Stratum.ZACS,
        )
        assert request.preferred_stratum == Stratum.ZACS

    def test_prompt_min_length(self):
        """Verify prompt minimum length is enforced."""
        with pytest.raises(ValidationError):
            CreateAgentRequest(prompt="Too short")

    def test_prompt_max_length(self):
        """Verify prompt maximum length is enforced."""
        with pytest.raises(ValidationError):
            CreateAgentRequest(prompt="x" * 2001)

    def test_prompt_at_max_length(self):
        """Verify prompt at max length (2000) is valid."""
        request = CreateAgentRequest(prompt="x" * 2000)
        assert len(request.prompt) == 2000


# -----------------------------------------------------------------------------
# RunAgentRequest Tests
# -----------------------------------------------------------------------------

class TestRunAgentRequest:
    """Tests for RunAgentRequest schema."""

    def test_valid_request_minimal(self):
        """Verify RunAgentRequest with only message."""
        request = RunAgentRequest(message="Hello, agent!")
        assert request.message == "Hello, agent!"
        assert request.context is None

    def test_valid_request_with_context(self):
        """Verify RunAgentRequest with context dictionary."""
        request = RunAgentRequest(
            message="Schedule a meeting",
            context={"user_id": "12345", "timezone": "America/New_York"},
        )
        assert request.context["user_id"] == "12345"

    def test_message_min_length(self):
        """Verify message minimum length (1) is enforced."""
        with pytest.raises(ValidationError):
            RunAgentRequest(message="")

    def test_message_max_length(self):
        """Verify message maximum length is enforced."""
        with pytest.raises(ValidationError):
            RunAgentRequest(message="x" * 10001)

    def test_message_single_char(self):
        """Verify single character message is valid."""
        request = RunAgentRequest(message="?")
        assert request.message == "?"


# -----------------------------------------------------------------------------
# AgentResponse Tests
# -----------------------------------------------------------------------------

class TestAgentResponse:
    """Tests for AgentResponse schema."""

    def test_valid_response_minimal(self):
        """Verify AgentResponse with only message."""
        response = AgentResponse(message="Here is my response")
        assert response.message == "Here is my response"
        assert response.tool_calls == []
        assert response.metadata is None

    def test_valid_response_with_tool_calls(self):
        """Verify AgentResponse with tool calls (ToolCall schema - fix #8)."""
        response = AgentResponse(
            message="I searched the web for you",
            tool_calls=[
                {
                    "tool_name": "websearch",
                    "arguments": {"query": "test search"},
                    "result": "Found 10 results",
                }
            ],
        )
        assert len(response.tool_calls) == 1
        assert response.tool_calls[0].tool_name == "websearch"
        assert response.tool_calls[0].arguments == {"query": "test search"}
        assert response.tool_calls[0].result == "Found 10 results"

    def test_valid_response_with_metadata(self):
        """Verify AgentResponse with metadata."""
        response = AgentResponse(
            message="Response with metadata",
            metadata={"execution_time_ms": 1500},
        )
        assert response.metadata["execution_time_ms"] == 1500


# -----------------------------------------------------------------------------
# CreateAgentResponse Tests
# -----------------------------------------------------------------------------

class TestCreateAgentResponse:
    """Tests for CreateAgentResponse schema."""

    def test_valid_response(self, valid_agent_config: AgentConfig):
        """Verify CreateAgentResponse with agent config."""
        response = CreateAgentResponse(
            agent_config=valid_agent_config,
            file_path="/path/to/agent.json",
        )
        assert response.agent_config.name == "test_agent"
        assert response.file_path == "/path/to/agent.json"

    def test_valid_response_without_file_path(self, valid_agent_config: AgentConfig):
        """Verify CreateAgentResponse without file path."""
        response = CreateAgentResponse(agent_config=valid_agent_config)
        assert response.file_path is None


# -----------------------------------------------------------------------------
# ToolInfo Tests
# -----------------------------------------------------------------------------

class TestToolInfo:
    """Tests for ToolInfo schema."""

    def test_valid_tool_info(self):
        """Verify ToolInfo creation."""
        tool = ToolInfo(
            name="websearch",
            description="Web research and fact-finding",
            capabilities=["Search the web", "Find current information"],
        )
        assert tool.name == "websearch"
        assert len(tool.capabilities) == 2

    def test_tool_info_empty_capabilities(self):
        """Verify ToolInfo with empty capabilities."""
        tool = ToolInfo(
            name="test_tool",
            description="A test tool",
        )
        assert tool.capabilities == []


# -----------------------------------------------------------------------------
# ToolListResponse Tests
# -----------------------------------------------------------------------------

class TestToolListResponse:
    """Tests for ToolListResponse schema."""

    def test_valid_response(self):
        """Verify ToolListResponse creation."""
        tools = [
            ToolInfo(name="voice", description="Voice synthesis"),
            ToolInfo(name="websearch", description="Web search"),
        ]
        response = ToolListResponse(tools=tools)
        assert len(response.tools) == 2

    def test_empty_tools_list(self):
        """Verify ToolListResponse with empty tools."""
        response = ToolListResponse(tools=[])
        assert response.tools == []


# -----------------------------------------------------------------------------
# Serialization Tests
# -----------------------------------------------------------------------------

class TestSerialization:
    """Tests for model serialization and deserialization."""

    def test_agent_config_to_dict(self, valid_agent_config: AgentConfig):
        """Verify AgentConfig can be serialized to dict."""
        data = valid_agent_config.model_dump()
        assert isinstance(data, dict)
        assert data["name"] == "test_agent"
        assert data["stratum"] == "RTI"

    def test_agent_config_to_json(self, valid_agent_config: AgentConfig):
        """Verify AgentConfig can be serialized to JSON."""
        json_str = valid_agent_config.model_dump_json()
        assert isinstance(json_str, str)
        assert "test_agent" in json_str

    def test_agent_config_round_trip(self, valid_agent_config: AgentConfig):
        """Verify AgentConfig survives serialization round trip."""
        data = valid_agent_config.model_dump()
        restored = AgentConfig(**data)
        assert restored.name == valid_agent_config.name
        assert restored.stratum == valid_agent_config.stratum
        assert restored.tools == valid_agent_config.tools
