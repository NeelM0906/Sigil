"""Pydantic settings for the Sigil v2 framework.

This module defines the main SigilSettings class that loads configuration from
environment variables and .env files. It uses pydantic-settings for automatic
environment variable parsing and validation.

Settings Categories:
    - Core: Framework-level settings (debug mode, log level, etc.)
    - LLM: Language model provider settings (model, temperature, max tokens)
    - Memory: Memory system configuration (paths, embedding model)
    - Context: Context management and truncation settings
    - External Tools: External tool settings (Tavily, etc.)
    - Paths: Directory paths for agents, memory, and output
    - Feature Flags: Toggle for v2 subsystems (memory, planning, contracts, etc.)
    - API Keys: External service credentials

Environment Variables:
    SIGIL_DEBUG: Enable debug mode (default: false)
    SIGIL_LOG_LEVEL: Logging level (default: INFO)
    SIGIL_USE_MEMORY: Enable 3-layer memory system (default: true)
    SIGIL_USE_PLANNING: Enable task decomposition (default: true)
    SIGIL_USE_CONTRACTS: Enable output verification (default: true)
    SIGIL_USE_EVOLUTION: Enable self-improvement (default: false)
    SIGIL_USE_ROUTING: Enable intent-based routing (default: true)
    ANTHROPIC_API_KEY: API key for Anthropic Claude
    ELEVENLABS_API_KEY: API key for ElevenLabs voice
    TAVILY_API_KEY: API key for Tavily web search
    TWILIO_ACCOUNT_SID: Twilio account SID
    TWILIO_AUTH_TOKEN: Twilio auth token

    Context Management Variables (use SIGIL_CONTEXT__ prefix):
    SIGIL_CONTEXT__MAX_CHARS_PER_TOOL_RESULT: Max chars per tool result (default: 4000)
    SIGIL_CONTEXT__MAX_TAVILY_RESULTS: Max Tavily results to include (default: 20)
    SIGIL_CONTEXT__MAX_CONTEXT_VALUE_LENGTH: Max length for context values (default: 200)
    SIGIL_CONTEXT__MAX_MEMORY_ITEMS_IN_CONTEXT: Max memory items (default: 5)
    SIGIL_CONTEXT__MAX_PLAN_STEPS_IN_CONTEXT: Max plan steps (default: 5)
    SIGIL_CONTEXT__MAX_TOOL_OUTPUTS: Max tool outputs (default: 10)
    SIGIL_CONTEXT__OFFLOAD_THRESHOLD_CHARS: Char threshold for offloading (default: 80000)
    SIGIL_CONTEXT__OFFLOAD_STORAGE_PATH: Path for offloaded results (default: outputs/tool_results)
    SIGIL_CONTEXT__ENABLE_AUTO_OFFLOAD: Enable auto-offloading (default: true)
    SIGIL_CONTEXT__SUMMARIZATION_THRESHOLD_TOKENS: Token threshold for summarization (default: 120000)
    SIGIL_CONTEXT__PRESERVE_RECENT_MESSAGES: Recent messages to preserve (default: 6)
    SIGIL_CONTEXT__ENABLE_AUTO_SUMMARIZATION: Enable auto summarization (default: true)
    SIGIL_CONTEXT__ENABLE_PRE_SEND_VALIDATION: Validate context before LLM calls (default: true)
    SIGIL_CONTEXT__MAX_CONTEXT_TOKENS: Max tokens in context (default: 150000)
    SIGIL_CONTEXT__CHARS_PER_TOKEN_ESTIMATE: Chars per token estimate (default: 4)
    SIGIL_CONTEXT__VALIDATION_BUFFER_TOKENS: Buffer tokens for safety (default: 5000)

Usage:
    from sigil.config.settings import get_settings

    settings = get_settings()
    print(settings.debug)
    print(settings.llm.model)
    print(settings.paths.agents_dir)
    print(settings.context.max_chars_per_tool_result)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


# =============================================================================
# Default Constants
# =============================================================================

DEFAULT_MODEL = "anthropic:claude-opus-4-5-20251101"
"""Default model identifier matching src/prompts.py."""

DEFAULT_MAX_TOKENS = 4096
"""Default maximum tokens for LLM responses."""

DEFAULT_TEMPERATURE = 0.7
"""Default temperature for LLM sampling."""


# =============================================================================
# Nested Settings Models
# =============================================================================


class LLMSettings(BaseModel):
    """Settings for Language Model configuration.

    These settings control which model is used and how it generates responses.

    Attributes:
        model: Model identifier in format 'provider:model-name'.
        max_tokens: Maximum tokens for completion responses.
        temperature: Sampling temperature (0.0-2.0).
        top_p: Nucleus sampling parameter (0.0-1.0).
        timeout: Request timeout in seconds.
    """

    model: str = Field(
        default=DEFAULT_MODEL,
        description="Model identifier (e.g., 'anthropic:claude-opus-4-5-20251101')"
    )
    max_tokens: int = Field(
        default=DEFAULT_MAX_TOKENS,
        ge=1,
        le=200000,
        description="Maximum tokens for completion"
    )
    temperature: float = Field(
        default=DEFAULT_TEMPERATURE,
        ge=0.0,
        le=2.0,
        description="Sampling temperature"
    )
    top_p: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Nucleus sampling parameter"
    )
    timeout: int = Field(
        default=120,
        ge=1,
        description="Request timeout in seconds"
    )


class MemorySettings(BaseModel):
    """Settings for the 3-layer memory system.

    Attributes:
        backend: Storage backend type ('filesystem', 'memory').
        vector_backend: Vector storage backend ('faiss', 'memory').
        locking_backend: Locking backend type ('filesystem', 'memory').
        embedding_model: Model for generating embeddings.
        cache_ttl: Time-to-live for cached items in seconds.
        max_items_per_query: Maximum items returned from RAG retrieval.
        consolidation_threshold: Number of items before triggering consolidation.

    Storage Backend Types:
        - 'filesystem': Local filesystem storage using JSON files (default).
            Requires paths.memory_dir to be set.
        - 'memory': In-memory storage for testing. Data is lost on restart.

    Vector Backend Types:
        - 'faiss': FAISS-based vector index for similarity search (default).
            Provides efficient approximate nearest neighbor search.
        - 'memory': In-memory brute-force search for testing.

    Locking Backend Types:
        - 'filesystem': File-based locking using portalocker (default).
            Works across processes on the same machine.
        - 'memory': In-memory locking for testing.
            Only works within the same process.
    """

    backend: str = Field(
        default="filesystem",
        description="Storage backend type: 'filesystem' or 'memory'"
    )
    vector_backend: str = Field(
        default="faiss",
        description="Vector storage backend: 'faiss' or 'memory'"
    )
    locking_backend: str = Field(
        default="filesystem",
        description="Locking backend type: 'filesystem' or 'memory'"
    )
    embedding_model: str = Field(
        default="text-embedding-3-small",
        description="Model for generating embeddings"
    )
    cache_ttl: int = Field(
        default=3600,
        ge=0,
        description="Cache TTL in seconds"
    )
    max_items_per_query: int = Field(
        default=10,
        ge=1,
        description="Max items from RAG retrieval"
    )
    consolidation_threshold: int = Field(
        default=100,
        ge=1,
        description="Items before consolidation triggers"
    )

    @field_validator("backend")
    @classmethod
    def validate_backend(cls, v: str) -> str:
        """Validate storage backend type."""
        valid_backends = {"filesystem", "memory"}
        normalized = v.lower().strip()
        if normalized not in valid_backends:
            raise ValueError(
                f"Invalid storage backend '{v}'. Must be one of: {', '.join(sorted(valid_backends))}"
            )
        return normalized

    @field_validator("vector_backend")
    @classmethod
    def validate_vector_backend(cls, v: str) -> str:
        """Validate vector backend type."""
        valid_backends = {"faiss", "memory"}
        normalized = v.lower().strip()
        if normalized not in valid_backends:
            raise ValueError(
                f"Invalid vector backend '{v}'. Must be one of: {', '.join(sorted(valid_backends))}"
            )
        return normalized

    @field_validator("locking_backend")
    @classmethod
    def validate_locking_backend(cls, v: str) -> str:
        """Validate locking backend type."""
        valid_backends = {"filesystem", "memory"}
        normalized = v.lower().strip()
        if normalized not in valid_backends:
            raise ValueError(
                f"Invalid locking backend '{v}'. Must be one of: {', '.join(sorted(valid_backends))}"
            )
        return normalized


class ContextSettings(BaseModel):
    """Settings for context management and truncation.

    These settings control how context is assembled, truncated, and validated
    before being sent to the LLM.

    Attributes:
        max_chars_per_tool_result: Maximum characters per tool result before truncation.
        max_tavily_results: Maximum Tavily search results to include.
        max_context_value_length: Maximum length for context dictionary values.
        max_memory_items_in_context: Maximum memory items included in context.
        max_plan_steps_in_context: Maximum plan steps included in context.
        max_tool_outputs: Maximum tool outputs to include.
        offload_threshold_chars: Character threshold for auto-offloading large results.
        offload_storage_path: Directory path for storing offloaded results.
        enable_auto_offload: Whether to enable automatic offloading of large results.
        summarization_threshold_tokens: Token threshold for triggering summarization.
        preserve_recent_messages: Number of recent messages to preserve during summarization.
        enable_auto_summarization: Whether to enable automatic conversation summarization.
        enable_pre_send_validation: Whether to validate context size before LLM calls.
        max_context_tokens: Maximum tokens allowed in assembled context.
        chars_per_token_estimate: Estimated characters per token for size calculations.
        validation_buffer_tokens: Buffer tokens reserved for safety margin.
    """

    # Truncation limits (current hardcoded values as defaults)
    max_chars_per_tool_result: int = Field(
        default=4000,
        ge=500,
        le=100000,
        description="Maximum characters per tool result"
    )
    max_tavily_results: int = Field(
        default=20,
        ge=5,
        le=100,
        description="Maximum Tavily search results"
    )
    max_context_value_length: int = Field(
        default=200,
        ge=50,
        le=2000,
        description="Maximum length for context values"
    )
    max_memory_items_in_context: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum memory items in context"
    )
    max_plan_steps_in_context: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum plan steps in context"
    )
    max_tool_outputs: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Maximum tool outputs to include"
    )

    # Offloading settings
    offload_threshold_chars: int = Field(
        default=80000,
        ge=10000,
        le=500000,
        description="Character threshold for auto-offloading (approx 20K tokens)"
    )
    offload_storage_path: str = Field(
        default="outputs/tool_results",
        description="Directory for storing offloaded results"
    )
    enable_auto_offload: bool = Field(
        default=True,
        description="Enable automatic offloading of large results"
    )

    # Summarization settings
    summarization_threshold_tokens: int = Field(
        default=120000,
        ge=50000,
        le=200000,
        description="Token threshold for triggering summarization"
    )
    preserve_recent_messages: int = Field(
        default=6,
        ge=2,
        le=20,
        description="Recent messages to preserve during summarization"
    )
    enable_auto_summarization: bool = Field(
        default=True,
        description="Enable automatic conversation summarization"
    )

    # Pre-send validation settings
    enable_pre_send_validation: bool = Field(
        default=True,
        description="Validate context size before LLM calls"
    )
    max_context_tokens: int = Field(
        default=150000,
        ge=10000,
        le=500000,
        description="Maximum tokens in assembled context"
    )
    chars_per_token_estimate: int = Field(
        default=4,
        ge=2,
        le=6,
        description="Characters per token for estimation"
    )
    validation_buffer_tokens: int = Field(
        default=5000,
        ge=1000,
        le=20000,
        description="Buffer tokens reserved for safety margin"
    )

    def estimate_tokens(self, text: str) -> int:
        """Estimate token count for a given text.

        Args:
            text: The text to estimate tokens for.

        Returns:
            Estimated token count based on chars_per_token_estimate.
        """
        return len(text) // self.chars_per_token_estimate

    def get_effective_max_tokens(self) -> int:
        """Get the effective maximum tokens accounting for buffer.

        Returns:
            Maximum tokens minus the validation buffer.
        """
        return self.max_context_tokens - self.validation_buffer_tokens


class ExternalToolSettings(BaseModel):
    """Settings for external tool integrations (e.g., Tavily web search).

    Attributes:
        timeout: Default timeout for external tool calls in seconds.
        max_retries: Maximum retry attempts for failed calls.
        retry_delay: Delay between retries in seconds.
    """

    timeout: int = Field(
        default=30,
        ge=1,
        description="Default timeout for external tool calls"
    )
    max_retries: int = Field(
        default=3,
        ge=0,
        description="Maximum retry attempts"
    )
    retry_delay: float = Field(
        default=1.0,
        ge=0.0,
        description="Delay between retries in seconds"
    )


# Backward compatibility alias
MCPSettings = ExternalToolSettings


class PathSettings(BaseModel):
    """Settings for directory paths used by Sigil.

    Paths are resolved relative to the project root if not absolute.

    Attributes:
        agents_dir: Directory for agent configurations.
        memory_dir: Directory for memory storage.
        output_dir: Directory for generated outputs.
    """

    agents_dir: Path = Field(
        default=Path("outputs/agents"),
        description="Directory for agent configurations"
    )
    memory_dir: Path = Field(
        default=Path("data/memory"),
        description="Directory for memory storage"
    )
    output_dir: Path = Field(
        default=Path("outputs"),
        description="Directory for generated outputs"
    )

    @field_validator("agents_dir", "memory_dir", "output_dir", mode="before")
    @classmethod
    def convert_to_path(cls, v: Any) -> Path:
        """Convert string paths to Path objects."""
        if isinstance(v, str):
            return Path(v)
        return v


class TelemetrySettings(BaseModel):
    """Settings for observability and metrics.

    Attributes:
        enabled: Whether telemetry is enabled.
        endpoint: Optional endpoint for sending metrics.
        sample_rate: Sampling rate for traces (0.0-1.0).
    """

    enabled: bool = Field(
        default=True,
        description="Whether telemetry is enabled"
    )
    endpoint: Optional[str] = Field(
        default=None,
        description="Endpoint for sending metrics"
    )
    sample_rate: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Sampling rate for traces"
    )


class EvolutionSettings(BaseModel):
    """Settings for self-improvement and optimization.

    Attributes:
        learning_rate: Learning rate for prompt optimization.
        max_iterations: Maximum optimization iterations.
        evaluation_samples: Number of samples for evaluation.
    """

    learning_rate: float = Field(
        default=0.01,
        ge=0.0,
        le=1.0,
        description="Learning rate for optimization"
    )
    max_iterations: int = Field(
        default=10,
        ge=1,
        description="Maximum optimization iterations"
    )
    evaluation_samples: int = Field(
        default=5,
        ge=1,
        description="Number of samples for evaluation"
    )


class ContractSettings(BaseModel):
    """Settings for contract-based verification (Phase 6).

    Controls contract execution behavior including default strategies,
    retry limits, and validation settings.

    Attributes:
        enabled: Whether contract verification is enabled.
        default_failure_strategy: Default failure strategy ('retry', 'fallback', 'fail').
        default_max_retries: Default maximum retry attempts.
        strict_validation: If True, treat warnings as errors.
        min_tokens_for_retry: Minimum tokens required for retry attempts.
        min_partial_coverage: Minimum field coverage for partial results.
        emit_events: Whether to emit events during execution.
    """

    enabled: bool = Field(
        default=True,
        description="Whether contract verification is enabled"
    )
    default_failure_strategy: str = Field(
        default="retry",
        description="Default failure strategy ('retry', 'fallback', 'fail')"
    )
    default_max_retries: int = Field(
        default=2,
        ge=0,
        le=10,
        description="Default maximum retry attempts"
    )
    strict_validation: bool = Field(
        default=False,
        description="If True, treat warnings as errors"
    )
    min_tokens_for_retry: int = Field(
        default=500,
        ge=100,
        description="Minimum tokens required for retry attempts"
    )
    min_partial_coverage: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Minimum field coverage for partial results"
    )
    emit_events: bool = Field(
        default=True,
        description="Whether to emit events during execution"
    )


class APIKeySettings(BaseSettings):
    """Settings for external API keys.

    These are loaded WITHOUT a prefix since they use standard env var names.
    The Anthropic key is validated as required when LLM features are used.

    Attributes:
        anthropic_api_key: API key for Anthropic Claude (required for LLM).
        elevenlabs_api_key: API key for ElevenLabs voice synthesis.
        tavily_api_key: API key for Tavily web search.
        twilio_account_sid: Twilio account SID for communication.
        twilio_auth_token: Twilio auth token for communication.
        openai_api_key: API key for OpenAI (optional alternative).
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    anthropic_api_key: Optional[str] = Field(
        default=None,
        description="Anthropic API key"
    )
    elevenlabs_api_key: Optional[str] = Field(
        default=None,
        description="ElevenLabs API key for voice"
    )
    tavily_api_key: Optional[str] = Field(
        default=None,
        description="Tavily API key for web search"
    )
    twilio_account_sid: Optional[str] = Field(
        default=None,
        description="Twilio account SID"
    )
    twilio_auth_token: Optional[str] = Field(
        default=None,
        description="Twilio auth token"
    )
    openai_api_key: Optional[str] = Field(
        default=None,
        description="OpenAI API key"
    )

    def require_anthropic_key(self) -> str:
        """Get the Anthropic API key, raising an error if not set.

        Returns:
            The Anthropic API key.

        Raises:
            ValueError: If the Anthropic API key is not configured.
        """
        if not self.anthropic_api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY environment variable is required for LLM features. "
                "Please set it in your environment or .env file."
            )
        return self.anthropic_api_key

    def require_elevenlabs_key(self) -> str:
        """Get the ElevenLabs API key, raising an error if not set.

        Returns:
            The ElevenLabs API key.

        Raises:
            ValueError: If the ElevenLabs API key is not configured.
        """
        if not self.elevenlabs_api_key:
            raise ValueError(
                "ELEVENLABS_API_KEY environment variable is required for voice features. "
                "Please set it in your environment or .env file."
            )
        return self.elevenlabs_api_key

    def require_tavily_key(self) -> str:
        """Get the Tavily API key, raising an error if not set.

        Returns:
            The Tavily API key.

        Raises:
            ValueError: If the Tavily API key is not configured.
        """
        if not self.tavily_api_key:
            raise ValueError(
                "TAVILY_API_KEY environment variable is required for web search features. "
                "Please set it in your environment or .env file."
            )
        return self.tavily_api_key

    def require_twilio_credentials(self) -> tuple[str, str]:
        """Get Twilio credentials, raising an error if not set.

        Returns:
            Tuple of (account_sid, auth_token).

        Raises:
            ValueError: If Twilio credentials are not configured.
        """
        if not self.twilio_account_sid or not self.twilio_auth_token:
            raise ValueError(
                "TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN environment variables are "
                "required for communication features. Please set them in your "
                "environment or .env file."
            )
        return self.twilio_account_sid, self.twilio_auth_token


# =============================================================================
# Main Settings Class
# =============================================================================


class SigilSettings(BaseSettings):
    """Main settings class for Sigil framework configuration.

    This class uses pydantic-settings to load configuration from environment
    variables and .env files. Settings are organized into nested groups for
    clarity.

    Environment variables use the SIGIL_ prefix (except for API keys which
    use standard names like ANTHROPIC_API_KEY).

    Example usage:
        ```python
        from sigil.config.settings import get_settings

        settings = get_settings()
        print(settings.debug)
        print(settings.llm.model)
        print(settings.api_keys.anthropic_api_key)
        ```

    Attributes:
        debug: Enable debug mode for verbose logging.
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        environment: Deployment environment (development, staging, production).
        use_memory: Enable the 3-layer memory system.
        use_planning: Enable task decomposition and planning.
        use_contracts: Enable output verification with contracts.
        use_evolution: Enable self-improvement and optimization.
        use_routing: Enable intent-based request routing.
        llm: Language model configuration.
        memory: Memory system configuration.
        context: Context management configuration.
        external_tools: External tool configuration (Tavily, etc.).
        paths: Directory path configuration.
        telemetry: Observability configuration.
        evolution: Self-improvement configuration.
        api_keys: External API credentials.
    """

    model_config = SettingsConfigDict(
        env_prefix="SIGIL_",
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        extra="ignore",
        case_sensitive=False,
    )

    # Core settings
    debug: bool = Field(
        default=False,
        description="Enable debug mode"
    )
    log_level: str = Field(
        default="INFO",
        description="Logging level"
    )
    environment: str = Field(
        default="development",
        description="Deployment environment"
    )

    # Feature flags (enabled by default, except experimental features)
    use_memory: bool = Field(
        default=True,
        description="Enable 3-layer memory system"
    )
    use_planning: bool = Field(
        default=True,
        description="Enable task decomposition"
    )
    use_contracts: bool = Field(
        default=True,
        description="Enable output verification"
    )
    use_evolution: bool = Field(
        default=False,
        description="Enable self-improvement (experimental)"
    )
    use_routing: bool = Field(
        default=True,
        description="Enable intent-based routing"
    )

    # Nested configuration groups
    llm: LLMSettings = Field(
        default_factory=LLMSettings,
        description="LLM configuration"
    )
    memory: MemorySettings = Field(
        default_factory=MemorySettings,
        description="Memory system configuration"
    )
    context: ContextSettings = Field(
        default_factory=ContextSettings,
        description="Context management configuration"
    )
    external_tools: ExternalToolSettings = Field(
        default_factory=ExternalToolSettings,
        description="External tool configuration (Tavily, etc.)"
    )
    # Backward compatibility alias
    mcp: ExternalToolSettings = Field(
        default_factory=ExternalToolSettings,
        description="Deprecated: Use external_tools instead"
    )
    paths: PathSettings = Field(
        default_factory=PathSettings,
        description="Path configuration"
    )
    telemetry: TelemetrySettings = Field(
        default_factory=TelemetrySettings,
        description="Telemetry configuration"
    )
    evolution: EvolutionSettings = Field(
        default_factory=EvolutionSettings,
        description="Evolution configuration"
    )
    contracts: ContractSettings = Field(
        default_factory=ContractSettings,
        description="Contract verification configuration"
    )

    # API keys (loaded separately without prefix)
    api_keys: APIKeySettings = Field(
        default_factory=APIKeySettings,
        description="API key configuration"
    )

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Validate and normalize log level."""
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        normalized = v.upper()
        if normalized not in valid_levels:
            raise ValueError(
                f"Invalid log level '{v}'. Must be one of: {', '.join(valid_levels)}"
            )
        return normalized

    @field_validator("environment")
    @classmethod
    def validate_environment(cls, v: str) -> str:
        """Validate and normalize environment."""
        valid_envs = {"development", "staging", "production", "test"}
        normalized = v.lower()
        if normalized not in valid_envs:
            raise ValueError(
                f"Invalid environment '{v}'. Must be one of: {', '.join(valid_envs)}"
            )
        return normalized

    @model_validator(mode="after")
    def ensure_paths_exist_if_needed(self) -> "SigilSettings":
        """Create directories if they don't exist (only in non-test environments)."""
        if self.environment != "test":
            # We don't create directories on validation - that's a side effect
            # that should happen when the directories are actually needed
            pass
        return self

    def ensure_directories(self) -> None:
        """Create configured directories if they don't exist.

        Call this method explicitly when you need to ensure directories exist.
        """
        self.paths.agents_dir.mkdir(parents=True, exist_ok=True)
        self.paths.memory_dir.mkdir(parents=True, exist_ok=True)
        self.paths.output_dir.mkdir(parents=True, exist_ok=True)

    def is_feature_enabled(self, feature: str) -> bool:
        """Check if a feature flag is enabled.

        Args:
            feature: Feature name (memory, planning, contracts, evolution, routing).

        Returns:
            True if the feature is enabled, False otherwise.

        Raises:
            ValueError: If the feature name is not recognized.
        """
        feature_map = {
            "memory": self.use_memory,
            "planning": self.use_planning,
            "contracts": self.use_contracts,
            "evolution": self.use_evolution,
            "routing": self.use_routing,
        }
        if feature not in feature_map:
            valid_features = ", ".join(feature_map.keys())
            raise ValueError(
                f"Unknown feature '{feature}'. Valid features: {valid_features}"
            )
        return feature_map[feature]

    def get_active_features(self) -> list[str]:
        """Get a list of all enabled feature flags.

        Returns:
            List of enabled feature names.
        """
        features = []
        if self.use_memory:
            features.append("memory")
        if self.use_planning:
            features.append("planning")
        if self.use_contracts:
            features.append("contracts")
        if self.use_evolution:
            features.append("evolution")
        if self.use_routing:
            features.append("routing")
        return features

    def to_dict(self) -> dict[str, Any]:
        """Export settings as a dictionary.

        Sensitive values (API keys) are masked for safe logging.

        Returns:
            Dictionary representation of settings with masked secrets.
        """
        data = self.model_dump()
        # Mask API keys for safe logging
        if "api_keys" in data:
            for key in data["api_keys"]:
                if data["api_keys"][key]:
                    data["api_keys"][key] = "***MASKED***"
        return data


# =============================================================================
# Singleton Pattern
# =============================================================================

# Global settings instance cache
_settings_instance: Optional[SigilSettings] = None


def get_settings() -> SigilSettings:
    """Get the cached settings instance.

    This function implements a singleton pattern for settings to avoid
    repeated .env file parsing and validation. The settings are created
    once and cached for subsequent calls.

    Returns:
        The cached SigilSettings instance.

    Example:
        ```python
        settings = get_settings()
        print(settings.llm.model)
        ```
    """
    global _settings_instance
    if _settings_instance is None:
        _settings_instance = SigilSettings()
    return _settings_instance


def reload_settings() -> SigilSettings:
    """Reload settings from environment, clearing the cache.

    This function is primarily useful for testing scenarios where you
    need to reload settings after modifying environment variables.

    Returns:
        A fresh SigilSettings instance.

    Example:
        ```python
        import os
        os.environ["SIGIL_DEBUG"] = "true"
        settings = reload_settings()
        assert settings.debug is True
        ```
    """
    global _settings_instance
    _settings_instance = SigilSettings()
    return _settings_instance


def clear_settings_cache() -> None:
    """Clear the settings cache without creating a new instance.

    This is useful in testing when you want to ensure the next call
    to get_settings() creates a fresh instance.
    """
    global _settings_instance
    _settings_instance = None


# =============================================================================
# Convenience Exports
# =============================================================================

# Re-export nested settings classes for direct import
__all__ = [
    # Main settings class
    "SigilSettings",
    # Nested settings classes
    "LLMSettings",
    "MemorySettings",
    "ContextSettings",
    "ExternalToolSettings",
    "MCPSettings",  # Backward compatibility alias
    "PathSettings",
    "TelemetrySettings",
    "EvolutionSettings",
    "ContractSettings",
    "APIKeySettings",
    # Singleton functions
    "get_settings",
    "reload_settings",
    "clear_settings_cache",
    # Constants
    "DEFAULT_MODEL",
    "DEFAULT_MAX_TOKENS",
    "DEFAULT_TEMPERATURE",
]
