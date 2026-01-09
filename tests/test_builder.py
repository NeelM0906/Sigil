"""Tests for ACTi Agent Builder main builder module.

This module tests the builder functionality defined in src/builder.py:
- create_builder() function
- Builder configuration with FilesystemBackend
- Subagent configuration
- Model parameter handling
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.builder import (
    BLAND_DATASET_DIR,
    DATASET_ROOT,
    N8N_WORKFLOWS_DIR,
    PROJECT_ROOT,
    create_builder,
)
from src.prompts import DEFAULT_MODEL


# -----------------------------------------------------------------------------
# Directory Configuration Tests
# -----------------------------------------------------------------------------

class TestDirectoryConfiguration:
    """Tests for directory path configuration."""

    def test_project_root_is_path(self):
        """Verify PROJECT_ROOT is a Path object."""
        assert isinstance(PROJECT_ROOT, Path)

    def test_project_root_is_absolute(self):
        """Verify PROJECT_ROOT is an absolute path."""
        assert PROJECT_ROOT.is_absolute()

    def test_project_root_exists(self):
        """Verify PROJECT_ROOT points to existing directory."""
        assert PROJECT_ROOT.exists()
        assert PROJECT_ROOT.is_dir()

    def test_project_root_is_acti_builder(self):
        """Verify PROJECT_ROOT is the acti-agent-builder directory."""
        assert PROJECT_ROOT.name == "acti-agent-builder"

    def test_dataset_root_is_path(self):
        """Verify DATASET_ROOT is a Path object."""
        assert isinstance(DATASET_ROOT, Path)

    def test_dataset_root_is_parent_of_project(self):
        """Verify DATASET_ROOT is parent of PROJECT_ROOT."""
        assert DATASET_ROOT == PROJECT_ROOT.parent

    def test_bland_dataset_dir_is_path(self):
        """Verify BLAND_DATASET_DIR is a Path object."""
        assert isinstance(BLAND_DATASET_DIR, Path)

    def test_bland_dataset_dir_points_to_bland_dataset(self):
        """Verify BLAND_DATASET_DIR has correct name."""
        assert BLAND_DATASET_DIR.name == "bland_dataset"

    def test_n8n_workflows_dir_is_path(self):
        """Verify N8N_WORKFLOWS_DIR is a Path object."""
        assert isinstance(N8N_WORKFLOWS_DIR, Path)

    def test_n8n_workflows_dir_points_to_n8n(self):
        """Verify N8N_WORKFLOWS_DIR has correct name."""
        assert N8N_WORKFLOWS_DIR.name == "n8n_workflows"


# -----------------------------------------------------------------------------
# create_builder() Tests - Basic Functionality
# -----------------------------------------------------------------------------

class TestCreateBuilderBasic:
    """Tests for basic create_builder() functionality."""

    @pytest.fixture
    def mock_deep_agent(self):
        """Mock the create_deep_agent function to avoid API calls."""
        with patch("src.builder.create_deep_agent") as mock:
            mock_agent = MagicMock()
            mock_agent.invoke = MagicMock(return_value={"messages": []})
            mock_agent.ainvoke = MagicMock(return_value={"messages": []})
            mock.return_value = mock_agent
            yield mock

    def test_returns_agent(self, mock_deep_agent):
        """Verify create_builder returns an agent object."""
        builder = create_builder()
        assert builder is not None

    def test_calls_create_deep_agent(self, mock_deep_agent):
        """Verify create_builder calls create_deep_agent."""
        create_builder()
        mock_deep_agent.assert_called_once()

    def test_default_model_is_used(self, mock_deep_agent):
        """Verify default model is passed to create_deep_agent."""
        create_builder()
        call_kwargs = mock_deep_agent.call_args.kwargs
        assert call_kwargs["model"] == DEFAULT_MODEL

    def test_custom_model_is_used(self, mock_deep_agent):
        """Verify custom model is passed to create_deep_agent."""
        custom_model = "openai:gpt-4"
        create_builder(model=custom_model)
        call_kwargs = mock_deep_agent.call_args.kwargs
        assert call_kwargs["model"] == custom_model

    def test_builder_tools_are_passed(self, mock_deep_agent):
        """Verify BUILDER_TOOLS are passed to create_deep_agent."""
        from src.tools import BUILDER_TOOLS

        create_builder()
        call_kwargs = mock_deep_agent.call_args.kwargs
        assert call_kwargs["tools"] == BUILDER_TOOLS

    def test_system_prompt_is_passed(self, mock_deep_agent):
        """Verify BUILDER_SYSTEM_PROMPT is passed to create_deep_agent."""
        from src.prompts import BUILDER_SYSTEM_PROMPT

        create_builder()
        call_kwargs = mock_deep_agent.call_args.kwargs
        assert call_kwargs["system_prompt"] == BUILDER_SYSTEM_PROMPT


# -----------------------------------------------------------------------------
# create_builder() Tests - FilesystemBackend Configuration
# -----------------------------------------------------------------------------

class TestCreateBuilderFilesystemBackend:
    """Tests for FilesystemBackend configuration in create_builder()."""

    @pytest.fixture
    def mock_deep_agent(self):
        """Mock create_deep_agent for testing."""
        with patch("src.builder.create_deep_agent") as mock:
            mock.return_value = MagicMock()
            yield mock

    @pytest.fixture
    def mock_filesystem_backend(self):
        """Mock FilesystemBackend for testing."""
        with patch("src.builder.FilesystemBackend") as mock:
            yield mock

    def test_creates_filesystem_backend(self, mock_deep_agent, mock_filesystem_backend):
        """Verify FilesystemBackend is created."""
        create_builder()
        mock_filesystem_backend.assert_called_once()

    def test_default_root_dir_is_dataset_root(
        self, mock_deep_agent, mock_filesystem_backend
    ):
        """Verify default root_dir is DATASET_ROOT."""
        create_builder()
        call_kwargs = mock_filesystem_backend.call_args.kwargs
        assert call_kwargs["root_dir"] == str(DATASET_ROOT)

    def test_custom_root_dir_is_used(
        self, mock_deep_agent, mock_filesystem_backend, temp_output_dir
    ):
        """Verify custom root_dir is passed correctly."""
        create_builder(root_dir=temp_output_dir)
        call_kwargs = mock_filesystem_backend.call_args.kwargs
        assert call_kwargs["root_dir"] == str(temp_output_dir.resolve())

    def test_virtual_mode_default_true(
        self, mock_deep_agent, mock_filesystem_backend
    ):
        """Verify virtual_mode defaults to True (security fix #2)."""
        create_builder()
        call_kwargs = mock_filesystem_backend.call_args.kwargs
        assert call_kwargs["virtual_mode"] is True

    def test_virtual_mode_can_be_enabled(
        self, mock_deep_agent, mock_filesystem_backend
    ):
        """Verify virtual_mode can be set to True."""
        create_builder(virtual_mode=True)
        call_kwargs = mock_filesystem_backend.call_args.kwargs
        assert call_kwargs["virtual_mode"] is True

    def test_backend_passed_to_agent(
        self, mock_deep_agent, mock_filesystem_backend
    ):
        """Verify backend is passed to create_deep_agent."""
        mock_backend_instance = MagicMock()
        mock_filesystem_backend.return_value = mock_backend_instance

        create_builder()
        call_kwargs = mock_deep_agent.call_args.kwargs
        assert call_kwargs["backend"] == mock_backend_instance


# -----------------------------------------------------------------------------
# create_builder() Tests - Subagent Configuration
# -----------------------------------------------------------------------------

class TestCreateBuilderSubagents:
    """Tests for subagent configuration in create_builder()."""

    @pytest.fixture
    def mock_deep_agent(self):
        """Mock create_deep_agent for testing."""
        with patch("src.builder.create_deep_agent") as mock:
            mock.return_value = MagicMock()
            yield mock

    def test_subagents_are_passed(self, mock_deep_agent):
        """Verify subagents are passed to create_deep_agent."""
        create_builder()
        call_kwargs = mock_deep_agent.call_args.kwargs
        assert "subagents" in call_kwargs
        assert isinstance(call_kwargs["subagents"], list)

    def test_two_subagents_configured(self, mock_deep_agent):
        """Verify exactly two subagents are configured."""
        create_builder()
        call_kwargs = mock_deep_agent.call_args.kwargs
        assert len(call_kwargs["subagents"]) == 2

    def test_prompt_engineer_subagent_present(self, mock_deep_agent):
        """Verify prompt-engineer subagent is configured."""
        create_builder()
        call_kwargs = mock_deep_agent.call_args.kwargs
        subagent_names = [s["name"] for s in call_kwargs["subagents"]]
        assert "prompt-engineer" in subagent_names

    def test_pattern_analyzer_subagent_present(self, mock_deep_agent):
        """Verify pattern-analyzer subagent is configured."""
        create_builder()
        call_kwargs = mock_deep_agent.call_args.kwargs
        subagent_names = [s["name"] for s in call_kwargs["subagents"]]
        assert "pattern-analyzer" in subagent_names

    def test_subagents_have_descriptions(self, mock_deep_agent):
        """Verify all subagents have descriptions."""
        create_builder()
        call_kwargs = mock_deep_agent.call_args.kwargs
        for subagent in call_kwargs["subagents"]:
            assert "description" in subagent
            assert len(subagent["description"]) > 10

    def test_subagents_have_system_prompts(self, mock_deep_agent):
        """Verify all subagents have system prompts."""
        create_builder()
        call_kwargs = mock_deep_agent.call_args.kwargs
        for subagent in call_kwargs["subagents"]:
            assert "system_prompt" in subagent
            assert len(subagent["system_prompt"]) > 100

    def test_subagents_have_empty_tools(self, mock_deep_agent):
        """Verify subagents have empty tools list (use built-in tools)."""
        create_builder()
        call_kwargs = mock_deep_agent.call_args.kwargs
        for subagent in call_kwargs["subagents"]:
            assert subagent["tools"] == []

    def test_subagents_inherit_model(self, mock_deep_agent):
        """Verify subagents use the same model as builder."""
        custom_model = "test:model"
        create_builder(model=custom_model)
        call_kwargs = mock_deep_agent.call_args.kwargs
        for subagent in call_kwargs["subagents"]:
            assert subagent["model"] == custom_model


# -----------------------------------------------------------------------------
# create_builder() Tests - Agent Interface
# -----------------------------------------------------------------------------

class TestBuilderAgentInterface:
    """Tests for the returned builder agent interface."""

    @pytest.fixture
    def mock_deep_agent(self):
        """Mock create_deep_agent for testing."""
        with patch("src.builder.create_deep_agent") as mock:
            mock_agent = MagicMock()
            mock_agent.invoke = MagicMock()
            mock_agent.ainvoke = MagicMock()
            mock.return_value = mock_agent
            yield mock

    def test_builder_has_invoke_method(self, mock_deep_agent):
        """Verify builder has invoke method."""
        builder = create_builder()
        assert hasattr(builder, "invoke")
        assert callable(builder.invoke)

    def test_builder_has_ainvoke_method(self, mock_deep_agent):
        """Verify builder has ainvoke method."""
        builder = create_builder()
        assert hasattr(builder, "ainvoke")
        assert callable(builder.ainvoke)


# -----------------------------------------------------------------------------
# Integration Tests (Require API Key - Marked for Skip)
# -----------------------------------------------------------------------------

class TestBuilderIntegration:
    """Integration tests that create real builder (require API key).

    These tests verify the actual builder creation and invocation.
    They are skipped if ANTHROPIC_API_KEY is not set.
    """

    @pytest.fixture
    def has_api_key(self):
        """Check if API key is available."""
        key = os.environ.get("ANTHROPIC_API_KEY", "")
        # Skip if it's the test placeholder key
        return key and not key.startswith("test-")

    @pytest.mark.skipif(
        not os.environ.get("ANTHROPIC_API_KEY") or
        os.environ.get("ANTHROPIC_API_KEY", "").startswith("test-"),
        reason="Requires real ANTHROPIC_API_KEY"
    )
    def test_create_builder_returns_compiled_state_graph(self):
        """Verify create_builder returns CompiledStateGraph type.

        This test verifies the actual builder creation without API calls.
        """
        from langgraph.graph.state import CompiledStateGraph

        builder = create_builder()
        assert isinstance(builder, CompiledStateGraph)

    @pytest.mark.skipif(
        not os.environ.get("ANTHROPIC_API_KEY") or
        os.environ.get("ANTHROPIC_API_KEY", "").startswith("test-"),
        reason="Requires real ANTHROPIC_API_KEY"
    )
    def test_builder_accepts_message_input(self):
        """Verify builder can accept message input format.

        This test only verifies the builder is created correctly,
        not that it makes successful API calls.
        """
        from langgraph.graph.state import CompiledStateGraph

        builder = create_builder()

        # Verify it's the right type that would accept messages
        assert isinstance(builder, CompiledStateGraph)


# -----------------------------------------------------------------------------
# Environment Configuration Tests
# -----------------------------------------------------------------------------

class TestEnvironmentConfiguration:
    """Tests for environment variable handling."""

    def test_dotenv_file_location(self):
        """Verify .env file path calculation is correct."""
        from src.builder import _env_path

        expected_path = PROJECT_ROOT / ".env"
        assert _env_path == expected_path


# -----------------------------------------------------------------------------
# Reference Directories Tests
# -----------------------------------------------------------------------------

class TestReferenceDirectories:
    """Tests for reference pattern directory configuration."""

    def test_bland_dataset_relative_to_dataset_root(self):
        """Verify BLAND_DATASET_DIR is relative to DATASET_ROOT."""
        assert BLAND_DATASET_DIR.parent == DATASET_ROOT

    def test_n8n_workflows_relative_to_dataset_root(self):
        """Verify N8N_WORKFLOWS_DIR is relative to DATASET_ROOT."""
        assert N8N_WORKFLOWS_DIR.parent == DATASET_ROOT

    def test_reference_directories_can_be_accessed(self):
        """Verify reference directories are accessible paths.

        Note: Directories may not exist in all test environments.
        """
        # Just verify they are valid Path objects
        assert BLAND_DATASET_DIR.is_absolute()
        assert N8N_WORKFLOWS_DIR.is_absolute()
