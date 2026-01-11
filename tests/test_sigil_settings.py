"""Tests for Sigil v2 settings module.

Task 3.2.5: Write tests for settings

Test Coverage:
- Default values work without env vars (except required keys)
- Feature flag combinations
- Invalid settings raise clear errors
- Environment variable override precedence
- Singleton pattern behavior
- API key validation
- Log level and environment normalization

Note: These tests DO NOT require ANTHROPIC_API_KEY because the settings
use lazy validation - keys are only required when explicitly accessed.
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from sigil.config.settings import (
    APIKeySettings,
    DEFAULT_MAX_TOKENS,
    DEFAULT_MODEL,
    DEFAULT_TEMPERATURE,
    EvolutionSettings,
    LLMSettings,
    MCPSettings,
    MemorySettings,
    PathSettings,
    SigilSettings,
    TelemetrySettings,
    clear_settings_cache,
    get_settings,
    reload_settings,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(autouse=True)
def clean_settings_cache():
    """Clear settings cache before and after each test."""
    clear_settings_cache()
    yield
    clear_settings_cache()


@pytest.fixture
def clean_env(monkeypatch, tmp_path):
    """Provide a clean environment for testing by removing SIGIL_ env vars.

    Also patches the .env file to prevent pydantic-settings from loading
    API keys from the project's .env file.
    """
    # Store original env vars
    sigil_keys = [k for k in os.environ.keys() if k.startswith("SIGIL_")]
    api_keys = ["ANTHROPIC_API_KEY", "ELEVENLABS_API_KEY", "TAVILY_API_KEY",
                "TWILIO_ACCOUNT_SID", "TWILIO_AUTH_TOKEN", "OPENAI_API_KEY"]

    for key in sigil_keys + api_keys:
        monkeypatch.delenv(key, raising=False)

    # Create an empty .env file in tmp_path and change to that directory
    # This prevents pydantic-settings from finding the real .env file
    empty_env = tmp_path / ".env"
    empty_env.write_text("")

    # Change to temp directory so pydantic-settings doesn't find real .env
    original_cwd = os.getcwd()
    os.chdir(tmp_path)

    yield

    # Restore original directory
    os.chdir(original_cwd)


# =============================================================================
# Test: Default Values (3.2.5 - Test default values work without env vars)
# =============================================================================


class TestDefaultValues:
    """Test that default values work correctly without environment variables."""

    def test_core_defaults(self, clean_env):
        """Test core settings have correct defaults."""
        settings = SigilSettings()
        assert settings.debug is False
        assert settings.log_level == "INFO"
        assert settings.environment == "development"

    def test_llm_defaults(self, clean_env):
        """Test LLM settings have correct defaults."""
        settings = SigilSettings()
        assert settings.llm.model == DEFAULT_MODEL
        assert settings.llm.max_tokens == DEFAULT_MAX_TOKENS
        assert settings.llm.temperature == DEFAULT_TEMPERATURE
        assert settings.llm.top_p == 1.0
        assert settings.llm.timeout == 120

    def test_memory_defaults(self, clean_env):
        """Test memory settings have correct defaults."""
        settings = SigilSettings()
        assert settings.memory.backend == "sqlite"
        assert settings.memory.embedding_model == "text-embedding-3-small"
        assert settings.memory.cache_ttl == 3600
        assert settings.memory.max_items_per_query == 10
        assert settings.memory.consolidation_threshold == 100

    def test_mcp_defaults(self, clean_env):
        """Test MCP settings have correct defaults."""
        settings = SigilSettings()
        assert settings.mcp.servers == []
        assert settings.mcp.timeout == 30
        assert settings.mcp.max_retries == 3
        assert settings.mcp.retry_delay == 1.0

    def test_path_defaults(self, clean_env):
        """Test path settings have correct defaults."""
        settings = SigilSettings()
        assert settings.paths.agents_dir == Path("outputs/agents")
        assert settings.paths.memory_dir == Path("data/memory")
        assert settings.paths.output_dir == Path("outputs")

    def test_telemetry_defaults(self, clean_env):
        """Test telemetry settings have correct defaults."""
        settings = SigilSettings()
        assert settings.telemetry.enabled is True
        assert settings.telemetry.endpoint is None
        assert settings.telemetry.sample_rate == 1.0

    def test_evolution_defaults(self, clean_env):
        """Test evolution settings have correct defaults."""
        settings = SigilSettings()
        assert settings.evolution.learning_rate == 0.01
        assert settings.evolution.max_iterations == 10
        assert settings.evolution.evaluation_samples == 5


# =============================================================================
# Test: Feature Flags (3.2.5 - Test feature flag combinations)
# =============================================================================


class TestFeatureFlags:
    """Test feature flag behavior and combinations."""

    def test_all_flags_default_to_false(self, clean_env):
        """Test that all feature flags default to False."""
        settings = SigilSettings()
        assert settings.use_memory is False
        assert settings.use_planning is False
        assert settings.use_contracts is False
        assert settings.use_evolution is False
        assert settings.use_routing is False

    def test_get_active_features_empty_by_default(self, clean_env):
        """Test get_active_features returns empty list by default."""
        settings = SigilSettings()
        assert settings.get_active_features() == []

    def test_enable_single_feature(self, clean_env):
        """Test enabling a single feature flag."""
        settings = SigilSettings(use_memory=True)
        assert settings.use_memory is True
        assert settings.get_active_features() == ["memory"]

    def test_enable_multiple_features(self, clean_env):
        """Test enabling multiple feature flags."""
        settings = SigilSettings(
            use_memory=True,
            use_planning=True,
            use_contracts=True,
        )
        active = settings.get_active_features()
        assert "memory" in active
        assert "planning" in active
        assert "contracts" in active
        assert len(active) == 3

    def test_enable_all_features(self, clean_env):
        """Test enabling all feature flags."""
        settings = SigilSettings(
            use_memory=True,
            use_planning=True,
            use_contracts=True,
            use_evolution=True,
            use_routing=True,
        )
        assert len(settings.get_active_features()) == 5

    def test_is_feature_enabled_valid(self, clean_env):
        """Test is_feature_enabled with valid feature names."""
        settings = SigilSettings(use_memory=True)
        assert settings.is_feature_enabled("memory") is True
        assert settings.is_feature_enabled("planning") is False

    def test_is_feature_enabled_invalid_raises(self, clean_env):
        """Test is_feature_enabled raises for unknown feature."""
        settings = SigilSettings()
        with pytest.raises(ValueError, match="Unknown feature 'invalid'"):
            settings.is_feature_enabled("invalid")

    def test_feature_flag_from_env(self, clean_env):
        """Test that feature flags can be set via environment."""
        os.environ["SIGIL_USE_MEMORY"] = "true"
        settings = SigilSettings()
        assert settings.use_memory is True


# =============================================================================
# Test: Invalid Settings (3.2.5 - Test invalid settings raise clear errors)
# =============================================================================


class TestInvalidSettings:
    """Test that invalid settings raise clear, informative errors."""

    def test_invalid_log_level_raises(self, clean_env):
        """Test that invalid log level raises ValueError."""
        with pytest.raises(ValueError, match="Invalid log level"):
            SigilSettings(log_level="INVALID")

    def test_invalid_environment_raises(self, clean_env):
        """Test that invalid environment raises ValueError."""
        with pytest.raises(ValueError, match="Invalid environment"):
            SigilSettings(environment="invalid_env")

    def test_temperature_out_of_range_high(self, clean_env):
        """Test that temperature > 2.0 raises error."""
        with pytest.raises(ValueError):
            LLMSettings(temperature=2.5)

    def test_temperature_out_of_range_low(self, clean_env):
        """Test that temperature < 0 raises error."""
        with pytest.raises(ValueError):
            LLMSettings(temperature=-0.1)

    def test_max_tokens_negative(self, clean_env):
        """Test that negative max_tokens raises error."""
        with pytest.raises(ValueError):
            LLMSettings(max_tokens=-100)

    def test_max_tokens_too_high(self, clean_env):
        """Test that max_tokens > 200000 raises error."""
        with pytest.raises(ValueError):
            LLMSettings(max_tokens=300000)

    def test_top_p_out_of_range(self, clean_env):
        """Test that top_p out of [0, 1] raises error."""
        with pytest.raises(ValueError):
            LLMSettings(top_p=1.5)

    def test_sample_rate_out_of_range(self, clean_env):
        """Test that sample_rate out of [0, 1] raises error."""
        with pytest.raises(ValueError):
            TelemetrySettings(sample_rate=1.5)

    def test_learning_rate_out_of_range(self, clean_env):
        """Test that learning_rate out of [0, 1] raises error."""
        with pytest.raises(ValueError):
            EvolutionSettings(learning_rate=2.0)


# =============================================================================
# Test: Environment Override Precedence (3.2.5)
# =============================================================================


class TestEnvironmentOverride:
    """Test environment variable override precedence."""

    def test_env_overrides_default(self, clean_env):
        """Test that environment variables override defaults."""
        os.environ["SIGIL_DEBUG"] = "true"
        os.environ["SIGIL_LOG_LEVEL"] = "DEBUG"
        settings = SigilSettings()
        assert settings.debug is True
        assert settings.log_level == "DEBUG"

    def test_nested_env_override(self, clean_env):
        """Test nested settings override via SIGIL_LLM__MAX_TOKENS."""
        os.environ["SIGIL_LLM__MAX_TOKENS"] = "8192"
        settings = SigilSettings()
        assert settings.llm.max_tokens == 8192

    def test_constructor_overrides_env(self, clean_env):
        """Test that constructor arguments override environment."""
        os.environ["SIGIL_DEBUG"] = "true"
        settings = SigilSettings(debug=False)
        assert settings.debug is False

    def test_case_insensitive_env_vars(self, clean_env):
        """Test that environment variable lookup is case-insensitive."""
        os.environ["SIGIL_DEBUG"] = "true"  # uppercase
        settings = SigilSettings()
        assert settings.debug is True

    def test_log_level_normalized_from_env(self, clean_env):
        """Test that log level is normalized from environment."""
        os.environ["SIGIL_LOG_LEVEL"] = "debug"  # lowercase
        settings = SigilSettings()
        assert settings.log_level == "DEBUG"  # normalized to uppercase

    def test_environment_normalized_from_env(self, clean_env):
        """Test that environment is normalized from environment."""
        os.environ["SIGIL_ENVIRONMENT"] = "PRODUCTION"  # uppercase
        settings = SigilSettings()
        assert settings.environment == "production"  # normalized to lowercase


# =============================================================================
# Test: Singleton Pattern (3.2.4)
# =============================================================================


class TestSingletonPattern:
    """Test singleton pattern behavior."""

    def test_get_settings_returns_same_instance(self, clean_env):
        """Test that get_settings returns the same cached instance."""
        s1 = get_settings()
        s2 = get_settings()
        assert s1 is s2

    def test_reload_settings_creates_new_instance(self, clean_env):
        """Test that reload_settings creates a fresh instance."""
        s1 = get_settings()
        s2 = reload_settings()
        assert s1 is not s2

    def test_reload_settings_picks_up_env_changes(self, clean_env):
        """Test that reload_settings picks up environment changes."""
        s1 = get_settings()
        assert s1.debug is False

        os.environ["SIGIL_DEBUG"] = "true"
        s2 = reload_settings()
        assert s2.debug is True

    def test_clear_settings_cache_clears_instance(self, clean_env):
        """Test that clear_settings_cache clears the cached instance."""
        s1 = get_settings()
        clear_settings_cache()
        s2 = get_settings()
        assert s1 is not s2


# =============================================================================
# Test: API Key Settings (3.2.3)
# =============================================================================


class TestAPIKeySettings:
    """Test API key settings behavior."""

    def test_api_keys_optional_by_default(self, clean_env):
        """Test that API keys are optional at instantiation."""
        settings = SigilSettings()
        assert settings.api_keys.anthropic_api_key is None
        assert settings.api_keys.elevenlabs_api_key is None
        assert settings.api_keys.tavily_api_key is None

    def test_require_anthropic_key_raises_when_missing(self, clean_env):
        """Test require_anthropic_key raises when key is not set."""
        api_keys = APIKeySettings()
        with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
            api_keys.require_anthropic_key()

    def test_require_anthropic_key_returns_key_when_set(self, clean_env):
        """Test require_anthropic_key returns key when set."""
        os.environ["ANTHROPIC_API_KEY"] = "sk-test-key"
        api_keys = APIKeySettings()
        assert api_keys.require_anthropic_key() == "sk-test-key"

    def test_require_elevenlabs_key_raises_when_missing(self, clean_env):
        """Test require_elevenlabs_key raises when key is not set."""
        api_keys = APIKeySettings()
        with pytest.raises(ValueError, match="ELEVENLABS_API_KEY"):
            api_keys.require_elevenlabs_key()

    def test_require_tavily_key_raises_when_missing(self, clean_env):
        """Test require_tavily_key raises when key is not set."""
        api_keys = APIKeySettings()
        with pytest.raises(ValueError, match="TAVILY_API_KEY"):
            api_keys.require_tavily_key()

    def test_require_twilio_credentials_raises_when_missing(self, clean_env):
        """Test require_twilio_credentials raises when credentials not set."""
        api_keys = APIKeySettings()
        with pytest.raises(ValueError, match="TWILIO"):
            api_keys.require_twilio_credentials()

    def test_require_twilio_credentials_returns_tuple(self, clean_env):
        """Test require_twilio_credentials returns tuple when set."""
        os.environ["TWILIO_ACCOUNT_SID"] = "AC123"
        os.environ["TWILIO_AUTH_TOKEN"] = "token456"
        api_keys = APIKeySettings()
        sid, token = api_keys.require_twilio_credentials()
        assert sid == "AC123"
        assert token == "token456"

    def test_api_keys_load_from_env(self, clean_env):
        """Test that API keys load from environment variables."""
        os.environ["ANTHROPIC_API_KEY"] = "sk-test"
        os.environ["TAVILY_API_KEY"] = "tvly-test"
        settings = SigilSettings()
        assert settings.api_keys.anthropic_api_key == "sk-test"
        assert settings.api_keys.tavily_api_key == "tvly-test"


# =============================================================================
# Test: Path Settings
# =============================================================================


class TestPathSettings:
    """Test path settings behavior."""

    def test_paths_are_path_objects(self, clean_env):
        """Test that path settings are Path objects."""
        settings = SigilSettings()
        assert isinstance(settings.paths.agents_dir, Path)
        assert isinstance(settings.paths.memory_dir, Path)
        assert isinstance(settings.paths.output_dir, Path)

    def test_string_paths_converted_to_path(self, clean_env):
        """Test that string paths are converted to Path objects."""
        paths = PathSettings(agents_dir="/custom/agents")
        assert isinstance(paths.agents_dir, Path)
        assert paths.agents_dir == Path("/custom/agents")


# =============================================================================
# Test: Serialization
# =============================================================================


class TestSerialization:
    """Test settings serialization."""

    def test_to_dict_returns_dict(self, clean_env):
        """Test that to_dict returns a dictionary."""
        settings = SigilSettings()
        data = settings.to_dict()
        assert isinstance(data, dict)

    def test_to_dict_masks_api_keys(self, clean_env):
        """Test that to_dict masks API keys for safe logging."""
        os.environ["ANTHROPIC_API_KEY"] = "sk-secret-key"
        settings = SigilSettings()
        data = settings.to_dict()
        assert data["api_keys"]["anthropic_api_key"] == "***MASKED***"

    def test_to_dict_contains_all_sections(self, clean_env):
        """Test that to_dict contains all settings sections."""
        settings = SigilSettings()
        data = settings.to_dict()
        assert "debug" in data
        assert "log_level" in data
        assert "llm" in data
        assert "memory" in data
        assert "mcp" in data
        assert "paths" in data
        assert "telemetry" in data
        assert "evolution" in data
        assert "api_keys" in data


# =============================================================================
# Test: Nested Settings Classes
# =============================================================================


class TestNestedSettingsClasses:
    """Test nested settings classes can be instantiated directly."""

    def test_llm_settings_standalone(self):
        """Test LLMSettings can be instantiated standalone."""
        llm = LLMSettings(model="openai:gpt-4", max_tokens=8000)
        assert llm.model == "openai:gpt-4"
        assert llm.max_tokens == 8000

    def test_memory_settings_standalone(self):
        """Test MemorySettings can be instantiated standalone."""
        memory = MemorySettings(backend="postgres", cache_ttl=7200)
        assert memory.backend == "postgres"
        assert memory.cache_ttl == 7200

    def test_mcp_settings_standalone(self):
        """Test MCPSettings can be instantiated standalone."""
        mcp = MCPSettings(timeout=60, max_retries=5)
        assert mcp.timeout == 60
        assert mcp.max_retries == 5

    def test_telemetry_settings_standalone(self):
        """Test TelemetrySettings can be instantiated standalone."""
        telemetry = TelemetrySettings(enabled=False, sample_rate=0.5)
        assert telemetry.enabled is False
        assert telemetry.sample_rate == 0.5

    def test_evolution_settings_standalone(self):
        """Test EvolutionSettings can be instantiated standalone."""
        evolution = EvolutionSettings(learning_rate=0.05, max_iterations=20)
        assert evolution.learning_rate == 0.05
        assert evolution.max_iterations == 20


# =============================================================================
# Test: Constants
# =============================================================================


class TestConstants:
    """Test module constants."""

    def test_default_model_constant(self):
        """Test DEFAULT_MODEL constant is correctly set."""
        assert DEFAULT_MODEL == "anthropic:claude-opus-4-5-20251101"

    def test_default_max_tokens_constant(self):
        """Test DEFAULT_MAX_TOKENS constant is correctly set."""
        assert DEFAULT_MAX_TOKENS == 4096

    def test_default_temperature_constant(self):
        """Test DEFAULT_TEMPERATURE constant is correctly set."""
        assert DEFAULT_TEMPERATURE == 0.7
