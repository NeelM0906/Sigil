"""Configuration module for Sigil v2 framework.

This module provides centralized configuration management using Pydantic
for validation and type safety. It includes:
- Environment-based settings with .env support
- Schema definitions for all configurable components
- Validation and serialization utilities

Usage:
    from sigil.config import get_settings, SigilSettings

    # Get the singleton settings instance
    settings = get_settings()
    print(settings.debug)
    print(settings.llm.model)

    # Check feature flags
    if settings.use_memory:
        print("Memory system enabled")

    # Access API keys (lazy validation)
    api_key = settings.api_keys.require_anthropic_key()
"""

from sigil.config.settings import (
    # Main settings class
    SigilSettings,
    # Nested settings classes
    LLMSettings,
    MemorySettings,
    ContextSettings,
    ExternalToolSettings,
    MCPSettings,  # Backward compatibility alias
    PathSettings,
    TelemetrySettings,
    EvolutionSettings,
    APIKeySettings,
    # Singleton functions
    get_settings,
    reload_settings,
    clear_settings_cache,
    # Constants
    DEFAULT_MODEL,
    DEFAULT_MAX_TOKENS,
    DEFAULT_TEMPERATURE,
)

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
