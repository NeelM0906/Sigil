"""Shared pytest fixtures for ACTi Agent Builder tests.

This module provides common fixtures used across all test modules:
- Valid and invalid AgentConfig fixtures
- Temporary output directory management
- Mock builder for tests that should not make API calls
- Setup/teardown for test isolation
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Generator
from unittest.mock import MagicMock, patch

import pytest

from src.schemas import AgentConfig, Stratum
from src.tools import reset_tool_history


# -----------------------------------------------------------------------------
# Test Isolation Fixtures (Fix 8a)
# -----------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def reset_tool_call_history():
    """Reset tool call history before each test.

    This fixture runs automatically before every test to ensure that the
    retry loop detection (Fix 8a) doesn't trigger false positives when
    execute_created_agent is called multiple times across different tests.
    """
    reset_tool_history()
    yield
    # Also reset after the test to clean up
    reset_tool_history()


# -----------------------------------------------------------------------------
# Valid Configuration Fixtures
# -----------------------------------------------------------------------------

@pytest.fixture
def valid_agent_config() -> AgentConfig:
    """Provide a valid AgentConfig instance for testing.

    Returns:
        A fully valid AgentConfig with all required and optional fields populated.
    """
    return AgentConfig(
        name="test_agent",
        description="A test agent for unit testing purposes",
        system_prompt="You are a test agent designed to validate the ACTi Agent Builder. "
                      "Your role is to respond to test prompts and verify system functionality.",
        tools=["websearch", "crm"],
        model="anthropic:claude-opus-4-5-20251101",
        stratum=Stratum.RTI,
    )


@pytest.fixture
def valid_agent_config_minimal() -> AgentConfig:
    """Provide a minimal valid AgentConfig with only required fields.

    Returns:
        An AgentConfig with only name, description, and system_prompt.
    """
    return AgentConfig(
        name="minimal_agent",
        description="A minimal test agent with only required fields",
        system_prompt="You are a minimal test agent. Your purpose is to demonstrate "
                      "that agents can be created with just the required fields.",
    )


@pytest.fixture
def valid_agent_config_dict() -> dict:
    """Provide valid agent configuration as a dictionary.

    Returns:
        Dictionary with valid agent configuration data.
    """
    return {
        "name": "dict_agent",
        "description": "Agent created from dictionary for testing",
        "system_prompt": "You are an agent created from a dictionary configuration. "
                         "This tests dictionary-to-model conversion.",
        "tools": ["calendar", "communication"],
        "model": "anthropic:claude-opus-4-5-20251101",
        "stratum": "ZACS",
    }


# -----------------------------------------------------------------------------
# Invalid Configuration Fixtures
# -----------------------------------------------------------------------------

@pytest.fixture
def invalid_config_bad_name_uppercase() -> dict:
    """Configuration with uppercase characters in name (invalid)."""
    return {
        "name": "BadName",
        "description": "A test agent with invalid uppercase name",
        "system_prompt": "This agent has an invalid name that contains uppercase letters.",
    }


@pytest.fixture
def invalid_config_bad_name_starts_with_number() -> dict:
    """Configuration with name starting with a number (invalid)."""
    return {
        "name": "1_invalid_agent",
        "description": "A test agent whose name starts with a number",
        "system_prompt": "This agent has an invalid name that starts with a number.",
    }


@pytest.fixture
def invalid_config_bad_name_special_chars() -> dict:
    """Configuration with special characters in name (invalid)."""
    return {
        "name": "bad-name-with-dashes",
        "description": "A test agent with dashes in name",
        "system_prompt": "This agent has an invalid name that contains dashes.",
    }


@pytest.fixture
def invalid_config_bad_name_too_long() -> dict:
    """Configuration with name exceeding max length (invalid)."""
    return {
        "name": "a" * 65,  # max is 64
        "description": "A test agent with name exceeding max length",
        "system_prompt": "This agent has a name that is too long to be valid.",
    }


@pytest.fixture
def invalid_config_description_too_short() -> dict:
    """Configuration with description below minimum length (invalid)."""
    return {
        "name": "short_desc",
        "description": "Too short",  # min is 10
        "system_prompt": "This agent has a description that is too short to be valid.",
    }


@pytest.fixture
def invalid_config_description_too_long() -> dict:
    """Configuration with description exceeding max length (invalid)."""
    return {
        "name": "long_desc",
        "description": "x" * 501,  # max is 500
        "system_prompt": "This agent has a description that is too long to be valid.",
    }


@pytest.fixture
def invalid_config_system_prompt_too_short() -> dict:
    """Configuration with system_prompt below minimum length (invalid)."""
    return {
        "name": "short_prompt",
        "description": "A test agent with system prompt too short",
        "system_prompt": "Too short.",  # min is 50
    }


@pytest.fixture
def invalid_config_invalid_tool() -> dict:
    """Configuration with unrecognized tool name (invalid)."""
    return {
        "name": "bad_tools",
        "description": "A test agent with invalid tool names",
        "system_prompt": "This agent has an invalid tool name that is not recognized.",
        "tools": ["fake_tool", "another_fake"],
    }


@pytest.fixture
def invalid_config_invalid_stratum() -> dict:
    """Configuration with invalid stratum value (invalid)."""
    return {
        "name": "bad_stratum",
        "description": "A test agent with invalid stratum value",
        "system_prompt": "This agent has an invalid stratum that is not one of the valid options.",
        "stratum": "INVALID_STRATUM",
    }


# -----------------------------------------------------------------------------
# Temporary Directory Fixtures
# -----------------------------------------------------------------------------

@pytest.fixture
def temp_output_dir() -> Generator[Path, None, None]:
    """Provide a temporary directory for test outputs.

    This fixture creates a temporary directory before each test and
    cleans it up afterward, ensuring test isolation.

    Yields:
        Path to the temporary output directory.
    """
    with tempfile.TemporaryDirectory(prefix="acti_test_") as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def temp_agents_dir(temp_output_dir: Path) -> Path:
    """Provide a temporary agents subdirectory.

    Args:
        temp_output_dir: Parent temporary directory.

    Returns:
        Path to the temporary agents directory.
    """
    agents_dir = temp_output_dir / "agents"
    agents_dir.mkdir(parents=True, exist_ok=True)
    return agents_dir


@pytest.fixture
def populated_agents_dir(temp_agents_dir: Path, valid_agent_config: AgentConfig) -> Path:
    """Provide a temporary directory pre-populated with agent configs.

    Creates several agent configuration files for testing retrieval functions.

    Args:
        temp_agents_dir: Temporary agents directory.
        valid_agent_config: Valid agent config fixture.

    Returns:
        Path to the populated agents directory.
    """
    # Create test agents for different strata
    test_agents = [
        {
            "name": "rti_researcher",
            "description": "Research agent for fact verification and data gathering",
            "system_prompt": "You are a research agent specializing in fact verification "
                             "and comprehensive data gathering for decision support.",
            "tools": ["websearch", "crm"],
            "model": "anthropic:claude-opus-4-5-20251101",
            "stratum": "RTI",
            "_metadata": {"created_at": "2025-01-09T10:00:00Z", "version": "1.0.0"},
        },
        {
            "name": "rai_qualifier",
            "description": "Lead qualification agent for assessing prospect readiness",
            "system_prompt": "You are a lead qualification specialist who builds rapport "
                             "and assesses budget, authority, need, and timeline.",
            "tools": ["communication", "crm"],
            "model": "anthropic:claude-opus-4-5-20251101",
            "stratum": "RAI",
            "_metadata": {"created_at": "2025-01-09T11:00:00Z", "version": "1.0.0"},
        },
        {
            "name": "zacs_scheduler",
            "description": "Appointment scheduling agent for sales team coordination",
            "system_prompt": "You are an appointment scheduler who manages calendar "
                             "availability and coordinates meeting times efficiently.",
            "tools": ["calendar", "communication"],
            "model": "anthropic:claude-opus-4-5-20251101",
            "stratum": "ZACS",
            "_metadata": {"created_at": "2025-01-09T12:00:00Z", "version": "1.0.0"},
        },
    ]

    for agent_data in test_agents:
        file_path = temp_agents_dir / f"{agent_data['name']}.json"
        file_path.write_text(json.dumps(agent_data, indent=2), encoding="utf-8")

    return temp_agents_dir


# -----------------------------------------------------------------------------
# Mock Builder Fixtures
# -----------------------------------------------------------------------------

@pytest.fixture
def mock_builder() -> MagicMock:
    """Provide a mock builder agent for testing without API calls.

    Returns:
        A MagicMock configured to simulate builder agent behavior.
    """
    mock = MagicMock()
    mock.invoke.return_value = {
        "messages": [
            MagicMock(type="ai", content="Agent created successfully."),
        ]
    }
    return mock


@pytest.fixture
def mock_anthropic_api():
    """Mock Anthropic API calls to avoid real API usage in tests.

    This fixture patches the environment to ensure no real API calls are made
    during testing.
    """
    with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key-for-testing"}):
        yield


# -----------------------------------------------------------------------------
# Tool Output Directory Patch Fixture
# -----------------------------------------------------------------------------

@pytest.fixture
def patch_output_dir(temp_agents_dir: Path):
    """Patch the OUTPUT_DIR in tools module to use temporary directory.

    This ensures that create_agent_config writes to a temporary directory
    during tests, preventing pollution of the real output directory.

    Args:
        temp_agents_dir: Temporary agents directory.

    Yields:
        The patched temporary directory path.
    """
    with patch("src.tools.OUTPUT_DIR", temp_agents_dir):
        yield temp_agents_dir


# -----------------------------------------------------------------------------
# Environment Setup Fixtures
# -----------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def setup_test_environment(monkeypatch):
    """Automatically set up test environment for all tests.

    This fixture runs before every test to ensure consistent environment:
    - Prevents accidental use of real API keys in most tests
    - Sets predictable environment state
    """
    # Only set a dummy key if one is not already set (allows integration tests)
    if not os.environ.get("ANTHROPIC_API_KEY"):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-api-key-not-real")


# -----------------------------------------------------------------------------
# Stratum Fixtures
# -----------------------------------------------------------------------------

@pytest.fixture
def all_strata() -> list[Stratum]:
    """Provide list of all valid Stratum enum values.

    Returns:
        List containing all Stratum enum members.
    """
    return list(Stratum)


@pytest.fixture
def stratum_tool_mapping() -> dict[Stratum, list[str]]:
    """Provide recommended tool mappings for each stratum.

    Returns:
        Dictionary mapping each stratum to its recommended tools.
    """
    return {
        Stratum.RTI: ["websearch", "crm"],
        Stratum.RAI: ["communication", "crm"],
        Stratum.ZACS: ["calendar", "communication", "voice"],
        Stratum.EEI: ["websearch", "crm"],
        Stratum.IGE: ["voice", "websearch", "calendar", "communication", "crm"],
    }
