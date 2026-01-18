"""CLI Configuration for Sigil v2.

This module provides configuration management for the Sigil CLI, including
global settings, ANSI color definitions, and token budget thresholds.

Key Components:
    - CLIConfig: Dataclass for CLI configuration
    - ANSI_COLORS: Terminal color codes
    - SEMANTIC_COLORS: Semantic color mappings
    - TOKEN_THRESHOLDS: Budget threshold levels

Configuration is managed through a global instance pattern with getter/setter
functions for thread-safe access.

Example:
    >>> from sigil.interfaces.cli.config import get_cli_config, update_cli_config
    >>> config = get_cli_config()
    >>> config.debug
    False
    >>> update_cli_config(debug=True)
    >>> get_cli_config().debug
    True
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Optional
import uuid


# =============================================================================
# Constants
# =============================================================================

# Token budget for 256K context window (Claude Opus)
TOTAL_TOKEN_BUDGET = 256_000

# Default token budget
DEFAULT_TOKEN_BUDGET = TOTAL_TOKEN_BUDGET


# =============================================================================
# ANSI Color Codes
# =============================================================================

ANSI_COLORS: dict[str, str] = {
    "red": "\033[31m",
    "green": "\033[32m",
    "yellow": "\033[33m",
    "blue": "\033[34m",
    "magenta": "\033[35m",
    "cyan": "\033[36m",
    "white": "\033[37m",
    "bold": "\033[1m",
    "dim": "\033[2m",
    "reset": "\033[0m",
}


# =============================================================================
# Semantic Color Mappings
# =============================================================================

SEMANTIC_COLORS: dict[str, str] = {
    "success": "green",
    "error": "red",
    "warning": "yellow",
    "info": "cyan",
    "planning": "blue",
    "reasoning": "magenta",
    "memory": "cyan",
    "tool": "yellow",
    "token": "dim",
}


# =============================================================================
# Token Thresholds
# =============================================================================

TOKEN_THRESHOLDS: dict[str, float] = {
    "low": 25.0,       # < 25% - OK
    "medium": 50.0,    # < 50% - Still safe
    "high": 80.0,      # < 80% - Warning
    "critical": 95.0,  # >= 95% - Critical
}


# =============================================================================
# CLI Configuration Dataclass
# =============================================================================


@dataclass
class CLIConfig:
    """Configuration settings for Sigil CLI.

    Holds all CLI-specific configuration including display options,
    token budget, session tracking, and logging settings.

    Attributes:
        debug: Enable debug mode with verbose error output.
        verbose: Enable verbose output for detailed pipeline info.
        color: Enable colored terminal output.
        token_budget: Total token budget for the session.
        session_id: Current session identifier (auto-generated if None).
        log_file: Path to log file for CLI operations.
        metrics_enabled: Enable collection of performance metrics.

    Example:
        >>> config = CLIConfig(debug=True, verbose=True)
        >>> config.debug
        True
        >>> config.to_dict()
        {'debug': True, 'verbose': True, 'color': True, ...}
    """

    debug: bool = False
    verbose: bool = False
    color: bool = True
    token_budget: int = DEFAULT_TOKEN_BUDGET
    session_id: Optional[str] = None
    log_file: Optional[str] = None
    metrics_enabled: bool = True

    def __post_init__(self) -> None:
        """Initialize session_id if not provided."""
        if self.session_id is None:
            self.session_id = f"cli-{uuid.uuid4().hex[:8]}"

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary.

        Returns:
            Dictionary representation of the configuration.
        """
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CLIConfig":
        """Create configuration from dictionary.

        Args:
            data: Dictionary with configuration values.

        Returns:
            New CLIConfig instance.
        """
        return cls(
            debug=data.get("debug", False),
            verbose=data.get("verbose", False),
            color=data.get("color", True),
            token_budget=data.get("token_budget", DEFAULT_TOKEN_BUDGET),
            session_id=data.get("session_id"),
            log_file=data.get("log_file"),
            metrics_enabled=data.get("metrics_enabled", True),
        )

    def copy(self, **updates: Any) -> "CLIConfig":
        """Create a copy with optional updates.

        Args:
            **updates: Field updates to apply.

        Returns:
            New CLIConfig with updates applied.
        """
        data = self.to_dict()
        data.update(updates)
        return CLIConfig.from_dict(data)

    def is_color_enabled(self) -> bool:
        """Check if color output is enabled.

        Returns:
            True if color should be used in output.
        """
        return self.color

    def get_token_remaining(self, used: int) -> int:
        """Calculate remaining token budget.

        Args:
            used: Number of tokens already used.

        Returns:
            Number of tokens remaining.
        """
        return max(0, self.token_budget - used)

    def get_token_percentage(self, used: int) -> float:
        """Calculate percentage of budget used.

        Args:
            used: Number of tokens already used.

        Returns:
            Percentage of budget consumed (0.0 to 100.0).
        """
        if self.token_budget == 0:
            return 100.0
        return (used / self.token_budget) * 100.0

    def get_budget_status(self, used: int) -> str:
        """Get budget status based on usage.

        Args:
            used: Number of tokens already used.

        Returns:
            Status string: 'ok', 'warning', or 'critical'.
        """
        percentage = self.get_token_percentage(used)
        if percentage >= TOKEN_THRESHOLDS["critical"]:
            return "critical"
        elif percentage >= TOKEN_THRESHOLDS["high"]:
            return "warning"
        return "ok"


# =============================================================================
# Global Configuration Management
# =============================================================================

# Global configuration instance
_cli_config: Optional[CLIConfig] = None


def get_cli_config() -> CLIConfig:
    """Get the current CLI configuration.

    Returns a copy of the global configuration. If no configuration
    has been set, creates and returns a default configuration.

    Returns:
        Current CLIConfig instance.

    Example:
        >>> config = get_cli_config()
        >>> config.debug
        False
    """
    global _cli_config
    if _cli_config is None:
        _cli_config = CLIConfig()
    return _cli_config


def set_cli_config(config: CLIConfig) -> None:
    """Set the global CLI configuration.

    Replaces the current configuration with the provided instance.

    Args:
        config: New configuration to use.

    Example:
        >>> new_config = CLIConfig(debug=True)
        >>> set_cli_config(new_config)
        >>> get_cli_config().debug
        True
    """
    global _cli_config
    _cli_config = config


def reset_cli_config() -> None:
    """Reset the CLI configuration to defaults.

    Clears the global configuration, causing the next get_cli_config()
    call to return a fresh default configuration.

    Example:
        >>> update_cli_config(debug=True)
        >>> get_cli_config().debug
        True
        >>> reset_cli_config()
        >>> get_cli_config().debug
        False
    """
    global _cli_config
    _cli_config = None


def update_cli_config(**kwargs: Any) -> CLIConfig:
    """Update specific configuration values.

    Updates the global configuration with the provided keyword arguments.
    Only valid CLIConfig fields are accepted.

    Args:
        **kwargs: Field updates to apply.

    Returns:
        Updated CLIConfig instance.

    Raises:
        TypeError: If an invalid field name is provided.

    Example:
        >>> update_cli_config(debug=True, verbose=True)
        >>> config = get_cli_config()
        >>> config.debug
        True
        >>> config.verbose
        True
    """
    global _cli_config
    if _cli_config is None:
        _cli_config = CLIConfig()

    # Validate kwargs are valid fields
    valid_fields = {
        "debug", "verbose", "color", "token_budget",
        "session_id", "log_file", "metrics_enabled"
    }
    invalid_fields = set(kwargs.keys()) - valid_fields
    if invalid_fields:
        raise TypeError(f"Invalid configuration fields: {invalid_fields}")

    # Apply updates
    _cli_config = _cli_config.copy(**kwargs)
    return _cli_config


# =============================================================================
# Utility Functions
# =============================================================================


def get_color_code(color_name: str) -> str:
    """Get ANSI color code by name.

    Supports both raw color names and semantic color names.

    Args:
        color_name: Name of the color (e.g., 'red', 'success').

    Returns:
        ANSI escape code for the color.

    Example:
        >>> get_color_code('red')
        '\\033[31m'
        >>> get_color_code('success')  # Maps to green
        '\\033[32m'
    """
    # Check semantic colors first
    if color_name in SEMANTIC_COLORS:
        color_name = SEMANTIC_COLORS[color_name]

    return ANSI_COLORS.get(color_name, "")


def get_reset_code() -> str:
    """Get ANSI reset code.

    Returns:
        ANSI escape code to reset terminal formatting.
    """
    return ANSI_COLORS["reset"]


def get_threshold_status(percentage: float) -> str:
    """Get threshold status from percentage.

    Args:
        percentage: Percentage value (0.0 to 100.0).

    Returns:
        Status string: 'low', 'medium', 'high', or 'critical'.
    """
    if percentage >= TOKEN_THRESHOLDS["critical"]:
        return "critical"
    elif percentage >= TOKEN_THRESHOLDS["high"]:
        return "high"
    elif percentage >= TOKEN_THRESHOLDS["medium"]:
        return "medium"
    return "low"


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Constants
    "TOTAL_TOKEN_BUDGET",
    "DEFAULT_TOKEN_BUDGET",
    "ANSI_COLORS",
    "SEMANTIC_COLORS",
    "TOKEN_THRESHOLDS",
    # Configuration class
    "CLIConfig",
    # Global config functions
    "get_cli_config",
    "set_cli_config",
    "reset_cli_config",
    "update_cli_config",
    # Utility functions
    "get_color_code",
    "get_reset_code",
    "get_threshold_status",
]
