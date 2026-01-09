"""Tests for ACTi Agent Builder tools.

This module tests the LangChain tools defined in src/tools.py:
- create_agent_config: Creates and persists agent configurations
- list_available_tools: Returns available MCP tool categories
- get_agent_config: Retrieves saved agent configurations
- list_created_agents: Lists all created agents
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from src.schemas import MCP_TOOL_CATEGORIES, Stratum
from src.tools import (
    BUILDER_TOOLS,
    OUTPUT_DIR,
    create_agent_config,
    get_agent_config,
    list_available_tools,
    list_created_agents,
)


# -----------------------------------------------------------------------------
# create_agent_config Tests
# -----------------------------------------------------------------------------

class TestCreateAgentConfig:
    """Tests for the create_agent_config tool."""

    def test_success_with_all_parameters(self, patch_output_dir: Path):
        """Verify create_agent_config succeeds with all parameters."""
        result = create_agent_config.invoke({
            "name": "test_complete_agent",
            "description": "A complete test agent with all parameters specified",
            "system_prompt": (
                "You are a complete test agent. Your role is to validate that "
                "the create_agent_config tool works correctly with all parameters."
            ),
            "tools": ["websearch", "crm"],
            "model": "anthropic:claude-opus-4-5-20251101",
            "stratum": "RTI",
        })

        assert "SUCCESS" in result
        assert "test_complete_agent" in result
        assert "websearch" in result or "crm" in result

        # Verify file was created
        output_file = patch_output_dir / "test_complete_agent.json"
        assert output_file.exists()

    def test_success_with_minimal_parameters(self, patch_output_dir: Path):
        """Verify create_agent_config succeeds with only required parameters."""
        result = create_agent_config.invoke({
            "name": "minimal_agent",
            "description": "A minimal agent with only required fields",
            "system_prompt": (
                "You are a minimal test agent created with only the required "
                "fields to validate the tool handles defaults correctly."
            ),
        })

        assert "SUCCESS" in result
        assert "minimal_agent" in result

        # Verify file was created with defaults
        output_file = patch_output_dir / "minimal_agent.json"
        assert output_file.exists()

        content = json.loads(output_file.read_text())
        assert content["tools"] == []
        assert content["model"] == "anthropic:claude-opus-4-5-20251101"

    def test_creates_valid_json_file(self, patch_output_dir: Path):
        """Verify create_agent_config creates properly formatted JSON."""
        create_agent_config.invoke({
            "name": "json_test_agent",
            "description": "Agent to test JSON file creation and format",
            "system_prompt": (
                "You are a test agent that validates JSON file creation. "
                "Your configuration should be saved as valid, readable JSON."
            ),
            "tools": ["calendar"],
            "stratum": "ZACS",
        })

        output_file = patch_output_dir / "json_test_agent.json"
        content = json.loads(output_file.read_text())

        assert content["name"] == "json_test_agent"
        assert content["stratum"] == "ZACS"
        assert content["tools"] == ["calendar"]
        assert "_metadata" in content
        assert "created_at" in content["_metadata"]
        assert "version" in content["_metadata"]
        assert content["_metadata"]["builder"] == "acti-agent-builder"

    def test_validation_error_invalid_name(self, patch_output_dir: Path):
        """Verify create_agent_config returns error for invalid name."""
        result = create_agent_config.invoke({
            "name": "InvalidName",  # uppercase not allowed
            "description": "Agent with invalid uppercase name",
            "system_prompt": (
                "This agent should not be created due to invalid name format."
            ),
        })

        assert "ERROR" in result
        assert "snake_case" in result.lower()

    def test_validation_error_name_starts_with_number(self, patch_output_dir: Path):
        """Verify create_agent_config returns error for name starting with number."""
        result = create_agent_config.invoke({
            "name": "1_bad_name",
            "description": "Agent with name starting with number",
            "system_prompt": (
                "This agent should not be created due to invalid name format."
            ),
        })

        assert "ERROR" in result

    def test_validation_error_invalid_tools(self, patch_output_dir: Path):
        """Verify create_agent_config returns error for invalid tools."""
        result = create_agent_config.invoke({
            "name": "bad_tools_agent",
            "description": "Agent with invalid tool names in configuration",
            "system_prompt": (
                "This agent should not be created due to invalid tool names."
            ),
            "tools": ["fake_tool", "nonexistent"],
        })

        assert "ERROR" in result
        assert "fake_tool" in result

    def test_validation_error_description_too_short(self, patch_output_dir: Path):
        """Verify create_agent_config returns error for short description."""
        result = create_agent_config.invoke({
            "name": "short_desc",
            "description": "Short",  # less than 10 chars
            "system_prompt": (
                "This agent should not be created due to short description."
            ),
        })

        assert "ERROR" in result

    def test_validation_error_system_prompt_too_short(self, patch_output_dir: Path):
        """Verify create_agent_config returns error for short system prompt."""
        result = create_agent_config.invoke({
            "name": "short_prompt",
            "description": "Agent with system prompt that is too short",
            "system_prompt": "Too short",  # less than 50 chars
        })

        assert "ERROR" in result

    def test_validation_error_invalid_stratum(self, patch_output_dir: Path):
        """Verify create_agent_config returns error for invalid stratum."""
        result = create_agent_config.invoke({
            "name": "bad_stratum",
            "description": "Agent with invalid stratum classification",
            "system_prompt": (
                "This agent should not be created due to invalid stratum value."
            ),
            "stratum": "INVALID_STRATUM",
        })

        assert "ERROR" in result
        assert "stratum" in result.lower()

    def test_stratum_case_insensitive(self, patch_output_dir: Path):
        """Verify stratum accepts lowercase input."""
        result = create_agent_config.invoke({
            "name": "lowercase_stratum",
            "description": "Agent with lowercase stratum input value",
            "system_prompt": (
                "This agent tests that stratum values are case-insensitive."
            ),
            "stratum": "rti",  # lowercase
        })

        assert "SUCCESS" in result

        output_file = patch_output_dir / "lowercase_stratum.json"
        content = json.loads(output_file.read_text())
        assert content["stratum"] == "RTI"

    def test_overwrites_existing_file(self, patch_output_dir: Path):
        """Verify create_agent_config overwrites existing file with same name."""
        # Create first version
        create_agent_config.invoke({
            "name": "overwrite_test",
            "description": "First version of the overwrite test agent",
            "system_prompt": (
                "This is the first version of the agent configuration."
            ),
        })

        # Create second version with same name
        create_agent_config.invoke({
            "name": "overwrite_test",
            "description": "Second version of the overwrite test agent",
            "system_prompt": (
                "This is the second version that should overwrite the first."
            ),
            "tools": ["voice"],
        })

        output_file = patch_output_dir / "overwrite_test.json"
        content = json.loads(output_file.read_text())

        assert "Second version" in content["description"]
        assert content["tools"] == ["voice"]


# -----------------------------------------------------------------------------
# list_available_tools Tests
# -----------------------------------------------------------------------------

class TestListAvailableTools:
    """Tests for the list_available_tools tool."""

    def test_returns_all_tool_categories(self):
        """Verify list_available_tools returns all MCP tool categories."""
        result = list_available_tools.invoke({})

        for tool_name in MCP_TOOL_CATEGORIES.keys():
            assert tool_name in result

    def test_includes_voice_category(self):
        """Verify voice tool category is included with details."""
        result = list_available_tools.invoke({})

        assert "voice" in result
        assert "ElevenLabs" in result

    def test_includes_websearch_category(self):
        """Verify websearch tool category is included with details."""
        result = list_available_tools.invoke({})

        assert "websearch" in result
        assert "Tavily" in result

    def test_includes_calendar_category(self):
        """Verify calendar tool category is included with details."""
        result = list_available_tools.invoke({})

        assert "calendar" in result
        assert "Google Calendar" in result

    def test_includes_communication_category(self):
        """Verify communication tool category is included with details."""
        result = list_available_tools.invoke({})

        assert "communication" in result
        assert "Twilio" in result

    def test_includes_crm_category(self):
        """Verify crm tool category is included with details."""
        result = list_available_tools.invoke({})

        assert "crm" in result
        assert "HubSpot" in result

    def test_includes_stratum_recommendations(self):
        """Verify stratum recommendations are included."""
        result = list_available_tools.invoke({})

        assert "RTI" in result
        assert "RAI" in result
        assert "ZACS" in result
        assert "EEI" in result
        assert "IGE" in result

    def test_includes_capabilities(self):
        """Verify tool capabilities are described."""
        result = list_available_tools.invoke({})

        assert "Capabilities" in result

    def test_includes_use_cases(self):
        """Verify use cases are included."""
        result = list_available_tools.invoke({})

        assert "Use cases" in result or "use cases" in result.lower()

    def test_format_is_readable(self):
        """Verify output is well-formatted and readable."""
        result = list_available_tools.invoke({})

        # Should have section separators
        assert "=" in result
        # Should have structure
        assert len(result) > 500  # Substantial output expected


# -----------------------------------------------------------------------------
# get_agent_config Tests
# -----------------------------------------------------------------------------

class TestGetAgentConfig:
    """Tests for the get_agent_config tool."""

    def test_retrieves_existing_config(self, populated_agents_dir: Path):
        """Verify get_agent_config retrieves existing configuration."""
        with patch("src.tools.OUTPUT_DIR", populated_agents_dir):
            result = get_agent_config.invoke({"name": "rti_researcher"})

        assert "rti_researcher" in result
        assert "websearch" in result
        assert "RTI" in result

    def test_returns_full_json(self, populated_agents_dir: Path):
        """Verify get_agent_config returns complete JSON configuration."""
        with patch("src.tools.OUTPUT_DIR", populated_agents_dir):
            result = get_agent_config.invoke({"name": "zacs_scheduler"})

        assert "system_prompt" in result
        assert "description" in result
        assert "tools" in result

    def test_includes_quick_summary(self, populated_agents_dir: Path):
        """Verify get_agent_config includes quick summary section."""
        with patch("src.tools.OUTPUT_DIR", populated_agents_dir):
            result = get_agent_config.invoke({"name": "rai_qualifier"})

        assert "Quick Summary" in result
        assert "Tools:" in result
        assert "Stratum:" in result

    def test_handles_missing_config(self, temp_agents_dir: Path):
        """Verify get_agent_config handles missing configuration gracefully."""
        with patch("src.tools.OUTPUT_DIR", temp_agents_dir):
            result = get_agent_config.invoke({"name": "nonexistent_agent"})

        assert "ERROR" in result
        assert "not found" in result.lower()

    def test_lists_available_agents_on_error(self, populated_agents_dir: Path):
        """Verify get_agent_config lists available agents when not found."""
        with patch("src.tools.OUTPUT_DIR", populated_agents_dir):
            result = get_agent_config.invoke({"name": "nonexistent"})

        assert "Available agents" in result
        # Should list the agents that do exist
        assert "rti_researcher" in result or "rai_qualifier" in result

    def test_handles_empty_directory(self, temp_agents_dir: Path):
        """Verify get_agent_config handles empty directory gracefully."""
        with patch("src.tools.OUTPUT_DIR", temp_agents_dir):
            result = get_agent_config.invoke({"name": "any_agent"})

        assert "ERROR" in result
        assert "create_agent_config" in result.lower()


# -----------------------------------------------------------------------------
# list_created_agents Tests
# -----------------------------------------------------------------------------

class TestListCreatedAgents:
    """Tests for the list_created_agents tool."""

    def test_empty_state(self, temp_agents_dir: Path):
        """Verify list_created_agents handles empty directory."""
        with patch("src.tools.OUTPUT_DIR", temp_agents_dir):
            result = list_created_agents.invoke({})

        assert "No agents" in result
        assert "create_agent_config" in result.lower()

    def test_lists_all_agents(self, populated_agents_dir: Path):
        """Verify list_created_agents returns all created agents."""
        with patch("src.tools.OUTPUT_DIR", populated_agents_dir):
            result = list_created_agents.invoke({})

        assert "rti_researcher" in result
        assert "rai_qualifier" in result
        assert "zacs_scheduler" in result

    def test_grouped_by_stratum(self, populated_agents_dir: Path):
        """Verify list_created_agents groups agents by stratum."""
        with patch("src.tools.OUTPUT_DIR", populated_agents_dir):
            result = list_created_agents.invoke({})

        # Should have stratum headers
        assert "[RTI]" in result
        assert "[RAI]" in result
        assert "[ZACS]" in result

    def test_includes_descriptions(self, populated_agents_dir: Path):
        """Verify list_created_agents includes agent descriptions."""
        with patch("src.tools.OUTPUT_DIR", populated_agents_dir):
            result = list_created_agents.invoke({})

        assert "Description:" in result

    def test_includes_tools_info(self, populated_agents_dir: Path):
        """Verify list_created_agents includes tools information."""
        with patch("src.tools.OUTPUT_DIR", populated_agents_dir):
            result = list_created_agents.invoke({})

        assert "Tools:" in result

    def test_shows_total_count(self, populated_agents_dir: Path):
        """Verify list_created_agents shows total agent count."""
        with patch("src.tools.OUTPUT_DIR", populated_agents_dir):
            result = list_created_agents.invoke({})

        assert "Total: 3 agent(s)" in result

    def test_provides_help_text(self, populated_agents_dir: Path):
        """Verify list_created_agents provides helpful next steps."""
        with patch("src.tools.OUTPUT_DIR", populated_agents_dir):
            result = list_created_agents.invoke({})

        assert "get_agent_config" in result.lower()

    def test_truncates_long_descriptions(self, temp_agents_dir: Path):
        """Verify list_created_agents truncates very long descriptions."""
        # Create agent with very long description
        long_desc = "x" * 100
        agent_data = {
            "name": "long_desc_agent",
            "description": long_desc,
            "system_prompt": "Test prompt for long description agent.",
            "tools": [],
            "stratum": "RTI",
            "_metadata": {"created_at": "2025-01-09T10:00:00Z"},
        }
        (temp_agents_dir / "long_desc_agent.json").write_text(
            json.dumps(agent_data), encoding="utf-8"
        )

        with patch("src.tools.OUTPUT_DIR", temp_agents_dir):
            result = list_created_agents.invoke({})

        # Description should be truncated (original was 100 chars)
        assert "..." in result


# -----------------------------------------------------------------------------
# BUILDER_TOOLS Export Tests
# -----------------------------------------------------------------------------

class TestBuilderToolsExport:
    """Tests for the BUILDER_TOOLS export list."""

    def test_exports_four_tools(self):
        """Verify BUILDER_TOOLS exports exactly four tools."""
        assert len(BUILDER_TOOLS) == 4

    def test_contains_create_agent_config(self):
        """Verify create_agent_config is in BUILDER_TOOLS."""
        tool_names = [t.name for t in BUILDER_TOOLS]
        assert "create_agent_config" in tool_names

    def test_contains_list_available_tools(self):
        """Verify list_available_tools is in BUILDER_TOOLS."""
        tool_names = [t.name for t in BUILDER_TOOLS]
        assert "list_available_tools" in tool_names

    def test_contains_get_agent_config(self):
        """Verify get_agent_config is in BUILDER_TOOLS."""
        tool_names = [t.name for t in BUILDER_TOOLS]
        assert "get_agent_config" in tool_names

    def test_contains_list_created_agents(self):
        """Verify list_created_agents is in BUILDER_TOOLS."""
        tool_names = [t.name for t in BUILDER_TOOLS]
        assert "list_created_agents" in tool_names

    def test_tools_have_descriptions(self):
        """Verify all tools have descriptions."""
        for tool in BUILDER_TOOLS:
            assert tool.description is not None
            assert len(tool.description) > 50  # Should have substantial docs


# -----------------------------------------------------------------------------
# OUTPUT_DIR Tests
# -----------------------------------------------------------------------------

class TestOutputDir:
    """Tests for the OUTPUT_DIR configuration."""

    def test_output_dir_is_path(self):
        """Verify OUTPUT_DIR is a Path object."""
        assert isinstance(OUTPUT_DIR, Path)

    def test_output_dir_has_agents_suffix(self):
        """Verify OUTPUT_DIR ends with 'agents'."""
        assert OUTPUT_DIR.name == "agents"

    def test_output_dir_in_outputs_folder(self):
        """Verify OUTPUT_DIR is inside 'outputs' folder."""
        assert OUTPUT_DIR.parent.name == "outputs"
