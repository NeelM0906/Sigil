"""Tests for ACTi Agent Builder tools.

This module tests the LangChain tools defined in src/tools.py:
- create_agent_config: Creates and persists agent configurations
- list_available_tools: Returns available MCP tool categories
- get_agent_config: Retrieves saved agent configurations
- list_created_agents: Lists all created agents
- execute_created_agent: Executes a task with a created agent using MCP tools
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.schemas import MCP_TOOL_CATEGORIES, Stratum, AgentMetadata
from src.tools import (
    BUILDER_TOOLS,
    OUTPUT_DIR,
    create_agent_config,
    execute_created_agent,
    get_agent_config,
    list_available_tools,
    list_created_agents,
    update_agent_config,
    delete_agent_config,
    clone_agent_config,
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

    def test_exports_eight_tools(self):
        """Verify BUILDER_TOOLS exports exactly eight tools."""
        assert len(BUILDER_TOOLS) == 8

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

    def test_contains_execute_created_agent(self):
        """Verify execute_created_agent is in BUILDER_TOOLS."""
        tool_names = [t.name for t in BUILDER_TOOLS]
        assert "execute_created_agent" in tool_names

    def test_contains_update_agent_config(self):
        """Verify update_agent_config is in BUILDER_TOOLS."""
        tool_names = [t.name for t in BUILDER_TOOLS]
        assert "update_agent_config" in tool_names

    def test_contains_delete_agent_config(self):
        """Verify delete_agent_config is in BUILDER_TOOLS."""
        tool_names = [t.name for t in BUILDER_TOOLS]
        assert "delete_agent_config" in tool_names

    def test_contains_clone_agent_config(self):
        """Verify clone_agent_config is in BUILDER_TOOLS."""
        tool_names = [t.name for t in BUILDER_TOOLS]
        assert "clone_agent_config" in tool_names

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


# -----------------------------------------------------------------------------
# execute_created_agent Tests
# -----------------------------------------------------------------------------

class TestExecuteCreatedAgent:
    """Tests for the execute_created_agent tool."""

    def test_error_agent_not_found(self, patch_output_dir: Path):
        """Verify error is returned when agent doesn't exist."""
        result = execute_created_agent.invoke({
            "agent_name": "nonexistent_agent",
            "task": "Test task",
        })

        assert "ERROR" in result
        assert "nonexistent_agent" in result
        assert "not found" in result.lower()

    def test_error_agent_not_found_lists_available(self, populated_agents_dir: Path):
        """Verify available agents are listed when target agent doesn't exist."""
        # Use populated_agents_dir which has pre-populated agents
        with patch("src.tools.OUTPUT_DIR", populated_agents_dir):
            result = execute_created_agent.invoke({
                "agent_name": "nonexistent_agent",
                "task": "Test task",
            })

            assert "ERROR" in result
            # Should list one of the pre-populated agents
            assert "rti_researcher" in result or "rai_qualifier" in result or "zacs_scheduler" in result

    def test_error_path_traversal_rejected(self, patch_output_dir: Path):
        """Verify path traversal attacks are rejected."""
        result = execute_created_agent.invoke({
            "agent_name": "../../../etc/passwd",
            "task": "Test task",
        })

        assert "ERROR" in result
        assert "path traversal" in result.lower()

    def test_error_timeout_too_short(self, populated_agents_dir: Path):
        """Verify error when timeout is below minimum."""
        with patch("src.tools.OUTPUT_DIR", populated_agents_dir):
            result = execute_created_agent.invoke({
                "agent_name": "rti_researcher",
                "task": "Test task",
                "timeout": 2,
            })

            # Error is returned as plain text (not JSON) for validation errors
            assert "ERROR" in result
            assert "at least 5 seconds" in result

    def test_error_timeout_too_long(self, populated_agents_dir: Path):
        """Verify error when timeout exceeds maximum."""
        with patch("src.tools.OUTPUT_DIR", populated_agents_dir):
            result = execute_created_agent.invoke({
                "agent_name": "rti_researcher",
                "task": "Test task",
                "timeout": 700,
            })

            # Error is returned as plain text (not JSON) for validation errors
            assert "ERROR" in result
            assert "exceed 600 seconds" in result

    def test_loads_agent_config_successfully(self, populated_agents_dir: Path):
        """Verify agent config is loaded correctly before execution."""
        import json
        with patch("src.tools.OUTPUT_DIR", populated_agents_dir):
            # Mock the async execution to avoid actual MCP connection
            with patch("src.tools.asyncio.run") as mock_asyncio_run:
                mock_asyncio_run.return_value = {
                    "success": True,
                    "agent_name": "rti_researcher",
                    "task": "Test task",
                    "response": "Mocked response",
                    "tools_used": [],
                    "execution_time_seconds": 1.0,
                    "errors": [],
                    "warnings": [],
                }

                result = execute_created_agent.invoke({
                    "agent_name": "rti_researcher",
                    "task": "Test task",
                })

                # Result is now JSON (Fix 8b)
                parsed = json.loads(result)
                assert parsed["status"] == "success"
                assert parsed["agent_name"] == "rti_researcher"
                assert "response" in parsed

    def test_returns_structured_output(self, populated_agents_dir: Path):
        """Verify the output contains all expected sections (Fix 8b - JSON output)."""
        import json
        with patch("src.tools.OUTPUT_DIR", populated_agents_dir):
            # Mock execution
            with patch("src.tools.asyncio.run") as mock_asyncio_run:
                mock_asyncio_run.return_value = {
                    "success": True,
                    "agent_name": "rti_researcher",
                    "task": "Test the structured output",
                    "response": "This is the agent response",
                    "tools_used": [
                        {"tool_name": "test_tool", "arguments": {"arg": "value"}, "result": "result"}
                    ],
                    "execution_time_seconds": 2.5,
                    "errors": [],
                    "warnings": [],
                }

                result = execute_created_agent.invoke({
                    "agent_name": "rti_researcher",
                    "task": "Test the structured output",
                })

                # Result is now JSON (Fix 8b)
                parsed = json.loads(result)
                assert "status" in parsed
                assert "response" in parsed
                assert "agent_name" in parsed
                assert "execution_time_ms" in parsed
                assert "reason" in parsed
                assert parsed["status"] == "success"
                # Tools should be mentioned in the response
                assert "test_tool" in parsed["response"]

    def test_handles_execution_errors_gracefully(self, populated_agents_dir: Path):
        """Verify errors during execution are reported properly (Fix 8b - JSON output)."""
        import json
        with patch("src.tools.OUTPUT_DIR", populated_agents_dir):
            # Mock execution with error
            with patch("src.tools.asyncio.run") as mock_asyncio_run:
                mock_asyncio_run.return_value = {
                    "success": False,
                    "agent_name": "rti_researcher",
                    "task": "Test error handling",
                    "response": "",
                    "tools_used": [],
                    "execution_time_seconds": 0.5,
                    "errors": ["Connection failed", "Tool not available"],
                    "warnings": [],
                }

                result = execute_created_agent.invoke({
                    "agent_name": "rti_researcher",
                    "task": "Test error handling",
                })

                # Result is now JSON (Fix 8b)
                parsed = json.loads(result)
                assert parsed["status"] == "error"
                # Errors should be in the reason field
                assert "Connection failed" in parsed["reason"]
                assert "Tool not available" in parsed["reason"]

    def test_tool_has_correct_signature(self):
        """Verify execute_created_agent has the expected parameter signature."""
        tool = execute_created_agent

        # Check the tool has the expected parameters
        schema = tool.args_schema.schema()
        properties = schema.get("properties", {})

        assert "agent_name" in properties
        assert "task" in properties
        assert "timeout" in properties

        # Check required parameters
        required = schema.get("required", [])
        assert "agent_name" in required
        assert "task" in required
        # timeout should have a default value
        assert "timeout" not in required or properties["timeout"].get("default") is not None

    def test_default_timeout_is_60(self, populated_agents_dir: Path):
        """Verify the default timeout is 60 seconds."""
        import json
        with patch("src.tools.OUTPUT_DIR", populated_agents_dir):
            # Mock execution and capture the timeout used
            with patch("src.tools.asyncio.run") as mock_asyncio_run:
                mock_asyncio_run.return_value = {
                    "success": True,
                    "agent_name": "rti_researcher",
                    "task": "Test",
                    "response": "Response",
                    "tools_used": [],
                    "execution_time_seconds": 1.0,
                    "errors": [],
                    "warnings": [],
                }

                result = execute_created_agent.invoke({
                    "agent_name": "rti_researcher",
                    "task": "Test without explicit timeout",
                })

                # Verify asyncio.run was called
                assert mock_asyncio_run.called
                # Result should be valid JSON (Fix 8b)
                parsed = json.loads(result)
                assert parsed["status"] == "success"

    def test_error_max_turns_too_low(self, populated_agents_dir: Path):
        """Verify error when max_turns is below minimum."""
        with patch("src.tools.OUTPUT_DIR", populated_agents_dir):
            result = execute_created_agent.invoke({
                "agent_name": "rti_researcher",
                "task": "Test task",
                "max_turns": 0,
            })

            assert "ERROR" in result
            assert "at least 1" in result

    def test_error_max_turns_too_high(self, populated_agents_dir: Path):
        """Verify error when max_turns exceeds maximum."""
        with patch("src.tools.OUTPUT_DIR", populated_agents_dir):
            result = execute_created_agent.invoke({
                "agent_name": "rti_researcher",
                "task": "Test task",
                "max_turns": 51,
            })

            assert "ERROR" in result
            assert "exceed 50" in result

    def test_max_turns_parameter_in_signature(self):
        """Verify execute_created_agent has max_turns parameter."""
        tool = execute_created_agent
        schema = tool.args_schema.schema()
        properties = schema.get("properties", {})

        assert "max_turns" in properties
        # max_turns should have a default value and not be required
        required = schema.get("required", [])
        assert "max_turns" not in required or properties["max_turns"].get("default") is not None

    def test_default_max_turns_is_10(self, populated_agents_dir: Path):
        """Verify the default max_turns is 10."""
        import json
        with patch("src.tools.OUTPUT_DIR", populated_agents_dir):
            with patch("src.tools.asyncio.run") as mock_asyncio_run:
                mock_asyncio_run.return_value = {
                    "success": True,
                    "agent_name": "rti_researcher",
                    "task": "Test",
                    "response": "Response",
                    "tools_used": [],
                    "execution_time_seconds": 1.0,
                    "errors": [],
                    "warnings": [],
                }

                result = execute_created_agent.invoke({
                    "agent_name": "rti_researcher",
                    "task": "Test without explicit max_turns",
                })

                # Verify asyncio.run was called with max_turns=10 (default)
                assert mock_asyncio_run.called
                # Result should be valid JSON (Fix 8b)
                parsed = json.loads(result)
                assert parsed["status"] == "success"


# -----------------------------------------------------------------------------
# update_agent_config Tests
# -----------------------------------------------------------------------------

class TestUpdateAgentConfig:
    """Tests for the update_agent_config tool."""

    def test_error_agent_not_found(self, patch_output_dir: Path):
        """Verify error when agent doesn't exist."""
        result = update_agent_config.invoke({
            "agent_name": "nonexistent_agent",
            "description": "New description for test",
        })

        assert "ERROR" in result
        assert "not found" in result.lower()

    def test_error_path_traversal_rejected(self, patch_output_dir: Path):
        """Verify path traversal attacks are rejected."""
        result = update_agent_config.invoke({
            "agent_name": "../../../etc/passwd",
            "description": "New description",
        })

        assert "ERROR" in result
        assert "path traversal" in result.lower()

    def test_error_no_updates_specified(self, populated_agents_dir: Path):
        """Verify error when no updates are provided."""
        with patch("src.tools.OUTPUT_DIR", populated_agents_dir):
            result = update_agent_config.invoke({
                "agent_name": "rti_researcher",
            })

            assert "No changes specified" in result

    def test_updates_description(self, populated_agents_dir: Path):
        """Verify description can be updated."""
        with patch("src.tools.OUTPUT_DIR", populated_agents_dir):
            result = update_agent_config.invoke({
                "agent_name": "rti_researcher",
                "description": "Updated description for RTI researcher agent testing",
            })

            assert "SUCCESS" in result
            assert "description" in result

    def test_updates_tools(self, populated_agents_dir: Path):
        """Verify tools can be updated."""
        with patch("src.tools.OUTPUT_DIR", populated_agents_dir):
            result = update_agent_config.invoke({
                "agent_name": "rti_researcher",
                "tools": ["websearch", "calendar"],
            })

            assert "SUCCESS" in result
            assert "tools" in result

    def test_validates_invalid_tools(self, populated_agents_dir: Path):
        """Verify invalid tools are rejected."""
        with patch("src.tools.OUTPUT_DIR", populated_agents_dir):
            result = update_agent_config.invoke({
                "agent_name": "rti_researcher",
                "tools": ["invalid_tool"],
            })

            assert "ERROR" in result
            assert "Invalid tool" in result

    def test_updates_stratum(self, populated_agents_dir: Path):
        """Verify stratum can be updated."""
        with patch("src.tools.OUTPUT_DIR", populated_agents_dir):
            result = update_agent_config.invoke({
                "agent_name": "rti_researcher",
                "stratum": "EEI",
            })

            assert "SUCCESS" in result
            assert "stratum" in result

    def test_validates_invalid_stratum(self, populated_agents_dir: Path):
        """Verify invalid stratum is rejected."""
        with patch("src.tools.OUTPUT_DIR", populated_agents_dir):
            result = update_agent_config.invoke({
                "agent_name": "rti_researcher",
                "stratum": "INVALID",
            })

            assert "ERROR" in result
            assert "Invalid stratum" in result

    def test_updates_tags(self, populated_agents_dir: Path):
        """Verify tags can be updated."""
        with patch("src.tools.OUTPUT_DIR", populated_agents_dir):
            result = update_agent_config.invoke({
                "agent_name": "rti_researcher",
                "tags": ["research", "testing"],
            })

            assert "SUCCESS" in result
            assert "tags" in result

    def test_increments_version(self, populated_agents_dir: Path):
        """Verify version is incremented on update."""
        with patch("src.tools.OUTPUT_DIR", populated_agents_dir):
            result = update_agent_config.invoke({
                "agent_name": "rti_researcher",
                "description": "Updated for version increment test agent",
            })

            assert "v1 -> v2" in result or "v" in result


# -----------------------------------------------------------------------------
# delete_agent_config Tests
# -----------------------------------------------------------------------------

class TestDeleteAgentConfig:
    """Tests for the delete_agent_config tool."""

    def test_error_agent_not_found(self, patch_output_dir: Path):
        """Verify error when agent doesn't exist."""
        result = delete_agent_config.invoke({
            "agent_name": "nonexistent_agent",
            "confirm": True,
        })

        assert "ERROR" in result
        assert "not found" in result.lower()

    def test_error_path_traversal_rejected(self, patch_output_dir: Path):
        """Verify path traversal attacks are rejected."""
        result = delete_agent_config.invoke({
            "agent_name": "../../../etc/passwd",
            "confirm": True,
        })

        assert "ERROR" in result
        assert "path traversal" in result.lower()

    def test_requires_confirmation(self, populated_agents_dir: Path):
        """Verify deletion requires confirmation."""
        with patch("src.tools.OUTPUT_DIR", populated_agents_dir):
            result = delete_agent_config.invoke({
                "agent_name": "rti_researcher",
                "confirm": False,
            })

            assert "WARNING" in result
            assert "confirm=True" in result

            # Agent should still exist
            assert (populated_agents_dir / "rti_researcher.json").exists()

    def test_deletes_with_confirmation(self, populated_agents_dir: Path):
        """Verify agent is deleted when confirmed."""
        with patch("src.tools.OUTPUT_DIR", populated_agents_dir):
            # Verify agent exists first
            assert (populated_agents_dir / "rti_researcher.json").exists()

            result = delete_agent_config.invoke({
                "agent_name": "rti_researcher",
                "confirm": True,
            })

            assert "SUCCESS" in result
            assert "permanently deleted" in result

            # Agent should be gone
            assert not (populated_agents_dir / "rti_researcher.json").exists()


# -----------------------------------------------------------------------------
# clone_agent_config Tests
# -----------------------------------------------------------------------------

class TestCloneAgentConfig:
    """Tests for the clone_agent_config tool."""

    def test_error_source_not_found(self, patch_output_dir: Path):
        """Verify error when source agent doesn't exist."""
        result = clone_agent_config.invoke({
            "source_name": "nonexistent_agent",
            "new_name": "cloned_agent",
        })

        assert "ERROR" in result
        assert "not found" in result.lower()

    def test_error_path_traversal_rejected_source(self, patch_output_dir: Path):
        """Verify path traversal attacks are rejected for source."""
        result = clone_agent_config.invoke({
            "source_name": "../../../etc/passwd",
            "new_name": "cloned_agent",
        })

        assert "ERROR" in result
        assert "path traversal" in result.lower()

    def test_error_invalid_new_name(self, populated_agents_dir: Path):
        """Verify invalid new name is rejected."""
        with patch("src.tools.OUTPUT_DIR", populated_agents_dir):
            result = clone_agent_config.invoke({
                "source_name": "rti_researcher",
                "new_name": "Invalid-Name",
            })

            assert "ERROR" in result
            assert "Invalid name" in result

    def test_error_new_name_already_exists(self, populated_agents_dir: Path):
        """Verify error when new name already exists."""
        with patch("src.tools.OUTPUT_DIR", populated_agents_dir):
            result = clone_agent_config.invoke({
                "source_name": "rti_researcher",
                "new_name": "rai_qualifier",  # Already exists
            })

            assert "ERROR" in result
            assert "already exists" in result

    def test_clones_successfully(self, populated_agents_dir: Path):
        """Verify agent is cloned successfully."""
        with patch("src.tools.OUTPUT_DIR", populated_agents_dir):
            result = clone_agent_config.invoke({
                "source_name": "rti_researcher",
                "new_name": "cloned_researcher",
            })

            assert "SUCCESS" in result
            assert "Cloned" in result

            # New agent should exist
            assert (populated_agents_dir / "cloned_researcher.json").exists()

            # Original should still exist
            assert (populated_agents_dir / "rti_researcher.json").exists()

    def test_clone_has_fresh_metadata(self, populated_agents_dir: Path):
        """Verify cloned agent has fresh metadata."""
        with patch("src.tools.OUTPUT_DIR", populated_agents_dir):
            clone_agent_config.invoke({
                "source_name": "rti_researcher",
                "new_name": "cloned_fresh",
            })

            # Read the cloned config
            config_path = populated_agents_dir / "cloned_fresh.json"
            with open(config_path) as f:
                config_data = json.load(f)

            assert config_data["metadata"]["version"] == 1
            assert config_data["metadata"]["execution_count"] == 0
            assert config_data["name"] == "cloned_fresh"


# -----------------------------------------------------------------------------
# AgentMetadata Tests
# -----------------------------------------------------------------------------

class TestAgentMetadata:
    """Tests for the AgentMetadata schema."""

    def test_default_values(self):
        """Verify default values are set correctly."""
        metadata = AgentMetadata()

        assert metadata.version == 1
        assert metadata.execution_count == 0
        assert metadata.tags == []
        assert metadata.last_executed is None
        assert metadata.created_at is not None
        assert metadata.updated_at is not None

    def test_tags_are_cleaned(self):
        """Verify tags are lowercased and deduplicated."""
        metadata = AgentMetadata(tags=["Test", "TEST", "  spaces  ", "valid"])

        assert "test" in metadata.tags
        assert "spaces" in metadata.tags
        assert "valid" in metadata.tags
        # Should be deduplicated
        assert metadata.tags.count("test") == 1
