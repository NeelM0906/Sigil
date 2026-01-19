"""Monitoring utilities for Sigil v2 CLI.

This module provides token display formatting, structured logging, and
real-time monitoring capabilities for the Sigil CLI.

Key Components:
    - TokenDisplay: Formats token usage information for CLI output
    - SigilLogFormatter: Custom log formatter with token counts
    - PipelineTokenTracker: Tracks tokens per pipeline step
    - setup_execution_logging: Configures file logging for execution

Log Format:
    [TIMESTAMP] [LEVEL] [COMPONENT] [TOKENS: X] Message

Example:
    [2026-01-11 10:15:32] [INFO] [ROUTING] [50 tokens] Intent: REASONING, Complexity: 0.65
"""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Optional


# =============================================================================
# Constants
# =============================================================================

# Default log file path
LOG_FILE_PATH = Path("outputs/sigil-execution.log")

# Max log file size (10MB)
MAX_LOG_SIZE = 10 * 1024 * 1024

# Number of backup files to keep
BACKUP_COUNT = 5

# Token budget for 256K model
TOTAL_TOKEN_BUDGET = 256_000


# =============================================================================
# ANSI Colors
# =============================================================================

COLORS = {
    "reset": "\033[0m",
    "bold": "\033[1m",
    "dim": "\033[2m",
    "red": "\033[31m",
    "green": "\033[32m",
    "yellow": "\033[33m",
    "blue": "\033[34m",
    "magenta": "\033[35m",
    "cyan": "\033[36m",
    "white": "\033[37m",
}


def colorize(text: str, color: str, force: bool = False) -> str:
    """Apply ANSI color to text.

    Args:
        text: Text to colorize.
        color: Color name from COLORS dict.
        force: If True, always apply color even if not a TTY.

    Returns:
        Colorized text string.
    """
    if not force and not sys.stdout.isatty():
        return text
    return f"{COLORS.get(color, '')}{text}{COLORS['reset']}"


# =============================================================================
# Pipeline Token Tracker
# =============================================================================


@dataclass
class PipelineTokenTracker:
    """Tracks token usage across pipeline steps.

    This class maintains running totals of tokens used at each step
    of the orchestration pipeline for detailed reporting.

    Attributes:
        routing_tokens: Tokens used for routing/classification.
        memory_tokens: Tokens used for memory retrieval.
        planning_tokens: Tokens used for plan generation.
        reasoning_tokens: Tokens used for reasoning execution.
        validation_tokens: Tokens used for contract validation.

    Example:
        >>> tracker = PipelineTokenTracker()
        >>> tracker.record_routing(50)
        >>> tracker.record_reasoning(450)
        >>> tracker.total
        500
    """

    routing_tokens: int = 0
    memory_tokens: int = 0
    planning_tokens: int = 0
    reasoning_tokens: int = 0
    validation_tokens: int = 0

    @property
    def total(self) -> int:
        """Get total tokens across all steps."""
        return (
            self.routing_tokens
            + self.memory_tokens
            + self.planning_tokens
            + self.reasoning_tokens
            + self.validation_tokens
        )

    def record_routing(self, tokens: int) -> None:
        """Record routing step tokens."""
        self.routing_tokens += tokens

    def record_memory(self, tokens: int) -> None:
        """Record memory retrieval tokens."""
        self.memory_tokens += tokens

    def record_planning(self, tokens: int) -> None:
        """Record planning step tokens."""
        self.planning_tokens += tokens

    def record_reasoning(self, tokens: int) -> None:
        """Record reasoning step tokens."""
        self.reasoning_tokens += tokens

    def record_validation(self, tokens: int) -> None:
        """Record validation step tokens."""
        self.validation_tokens += tokens

    def reset(self) -> None:
        """Reset all counters to zero."""
        self.routing_tokens = 0
        self.memory_tokens = 0
        self.planning_tokens = 0
        self.reasoning_tokens = 0
        self.validation_tokens = 0

    def to_dict(self) -> dict[str, int]:
        """Convert to dictionary."""
        return {
            "routing": self.routing_tokens,
            "memory": self.memory_tokens,
            "planning": self.planning_tokens,
            "reasoning": self.reasoning_tokens,
            "validation": self.validation_tokens,
            "total": self.total,
        }


# =============================================================================
# Token Display
# =============================================================================


class TokenDisplay:
    """Formats token usage information for CLI output.

    Provides methods to format token counts, percentages, and
    pipeline summaries in a human-readable format.

    Attributes:
        budget: Total token budget (default 256K).

    Example:
        >>> display = TokenDisplay(256_000)
        >>> display.format_tokens(1234)
        '1,234'
        >>> display.format_percentage(1234)
        '0.48%'
    """

    def __init__(self, budget: int = TOTAL_TOKEN_BUDGET) -> None:
        """Initialize with token budget.

        Args:
            budget: Total token budget for percentage calculations.
        """
        self.budget = budget

    def format_tokens(self, tokens: int) -> str:
        """Format token count with thousands separator.

        Args:
            tokens: Number of tokens.

        Returns:
            Formatted string like '1,234'.
        """
        return f"{tokens:,}"

    def format_percentage(self, tokens: int) -> str:
        """Format tokens as percentage of budget.

        Args:
            tokens: Number of tokens.

        Returns:
            Percentage string like '0.48%'.
        """
        percentage = (tokens / self.budget) * 100
        return f"{percentage:.2f}%"

    def format_budget_usage(self, tokens: int) -> str:
        """Format tokens with budget context.

        Args:
            tokens: Number of tokens used.

        Returns:
            String like '1,234 / 256,000 (0.48%)'.
        """
        return f"{self.format_tokens(tokens)} / {self.format_tokens(self.budget)} ({self.format_percentage(tokens)})"

    def format_step(
        self,
        step_name: str,
        tokens: int,
        width: int = 15,
    ) -> str:
        """Format a single pipeline step.

        Args:
            step_name: Name of the step (e.g., 'Routing').
            tokens: Tokens used in this step.
            width: Column width for alignment.

        Returns:
            Formatted line like '  Routing:        50 tokens'.
        """
        name_padded = f"{step_name}:".ljust(width)
        if tokens > 0:
            return f"  {name_padded} {self.format_tokens(tokens)} tokens"
        else:
            return f"  {name_padded} 0 tokens"

    def format_pipeline_summary(self, tracker: PipelineTokenTracker) -> str:
        """Format complete pipeline token summary.

        Args:
            tracker: PipelineTokenTracker with usage data.

        Returns:
            Multi-line formatted summary.
        """
        lines = []
        lines.append(self.format_step("Routing", tracker.routing_tokens))
        lines.append(self.format_step("Memory", tracker.memory_tokens))
        lines.append(self.format_step("Planning", tracker.planning_tokens))
        lines.append(self.format_step("Reasoning", tracker.reasoning_tokens))
        lines.append(self.format_step("Validation", tracker.validation_tokens))
        return "\n".join(lines)

    def get_budget_status(self, tokens: int) -> str:
        """Get status indicator based on budget usage.

        Args:
            tokens: Tokens used.

        Returns:
            Status string: 'OK', 'WARNING', or 'CRITICAL'.
        """
        percentage = (tokens / self.budget) * 100
        if percentage < 50:
            return "OK"
        elif percentage < 80:
            return "WARNING"
        else:
            return "CRITICAL"

    def format_info_line(self, tokens: int) -> str:
        """Format standard info line for logging.

        Args:
            tokens: Total tokens used.

        Returns:
            Formatted string like '[INFO] Total tokens used: 1,234 / 256,000 (0.48%)'.
        """
        return f"[INFO] Total tokens used: {self.format_budget_usage(tokens)}"


# =============================================================================
# Custom Log Formatter
# =============================================================================


class SigilLogFormatter(logging.Formatter):
    """Custom log formatter with token count support.

    Formats log records with timestamp, level, component, and token count.

    Format:
        [TIMESTAMP] [LEVEL] [COMPONENT] [TOKENS: X] Message

    Example:
        [2026-01-11 10:15:32] [INFO] [ROUTING] [50 tokens] Intent: REASONING
    """

    # Format string for standard output
    STANDARD_FORMAT = "[%(asctime)s] [%(levelname)s] [%(component)s] [%(tokens)s tokens] %(message)s"

    # Date format
    DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

    def __init__(self, include_tokens: bool = True) -> None:
        """Initialize the formatter.

        Args:
            include_tokens: Whether to include token counts in output.
        """
        super().__init__(
            fmt=self.STANDARD_FORMAT,
            datefmt=self.DATE_FORMAT,
        )
        self.include_tokens = include_tokens

    def format(self, record: logging.LogRecord) -> str:
        """Format the log record.

        Adds default values for component and tokens if not present.

        Args:
            record: The log record to format.

        Returns:
            Formatted log string.
        """
        # Add defaults for custom fields if not present
        if not hasattr(record, "component"):
            record.component = record.name.split(".")[-1].upper()
        if not hasattr(record, "tokens"):
            record.tokens = 0

        return super().format(record)


class SigilLogAdapter(logging.LoggerAdapter):
    """Logger adapter that adds component and token context.

    Allows logging with component and token information easily.

    Example:
        >>> logger = SigilLogAdapter(logging.getLogger(__name__), {"component": "ROUTING"})
        >>> logger.info("Intent classified", extra={"tokens": 50})
    """

    def process(
        self,
        msg: str,
        kwargs: dict[str, Any],
    ) -> tuple[str, dict[str, Any]]:
        """Process the logging call to add extra fields.

        Args:
            msg: The log message.
            kwargs: Additional keyword arguments.

        Returns:
            Tuple of (message, kwargs) with added context.
        """
        extra = kwargs.get("extra", {})
        extra.update(self.extra)
        kwargs["extra"] = extra
        return msg, kwargs


# =============================================================================
# Logging Setup
# =============================================================================

# Global execution logger instance
_execution_logger: Optional[logging.Logger] = None


def get_log_file_path() -> Path:
    """Get the path to the execution log file.

    Returns:
        Path object for the log file.
    """
    return LOG_FILE_PATH


def setup_execution_logging(
    log_file: Optional[Path] = None,
    level: int = logging.DEBUG,
) -> logging.Logger:
    """Set up execution logging with file rotation.

    Creates a logger that writes to the execution log file with
    rotation and custom formatting.

    Args:
        log_file: Optional custom log file path.
        level: Logging level (default DEBUG).

    Returns:
        Configured logger instance.
    """
    global _execution_logger

    if _execution_logger is not None:
        return _execution_logger

    log_path = log_file or LOG_FILE_PATH

    # Ensure parent directory exists
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # Create logger
    logger = logging.getLogger("sigil.execution")
    logger.setLevel(level)

    # Remove existing handlers
    logger.handlers.clear()

    # Create rotating file handler
    file_handler = RotatingFileHandler(
        filename=log_path,
        maxBytes=MAX_LOG_SIZE,
        backupCount=BACKUP_COUNT,
        encoding="utf-8",
    )
    file_handler.setLevel(level)

    # Set custom formatter
    formatter = SigilLogFormatter()
    file_handler.setFormatter(formatter)

    # Add handler
    logger.addHandler(file_handler)

    # Prevent propagation to root logger
    logger.propagate = False

    _execution_logger = logger
    return logger


def get_execution_logger() -> logging.Logger:
    """Get the execution logger, creating it if necessary.

    Returns:
        The execution logger instance.
    """
    global _execution_logger
    if _execution_logger is None:
        return setup_execution_logging()
    return _execution_logger


def log_pipeline_step(
    component: str,
    message: str,
    tokens: int,
    level: int = logging.INFO,
) -> None:
    """Log a pipeline step with component and token info.

    Convenience function for logging pipeline execution steps.

    Args:
        component: Component name (e.g., 'ROUTING', 'REASONING').
        message: Log message.
        tokens: Tokens used in this step.
        level: Logging level (default INFO).

    Example:
        >>> log_pipeline_step("ROUTING", "Intent: REASONING, Complexity: 0.65", 50)
    """
    logger = get_execution_logger()
    logger.log(
        level,
        message,
        extra={
            "component": component,
            "tokens": tokens,
        },
    )


def log_token_summary(
    tracker: PipelineTokenTracker,
    budget: int = TOTAL_TOKEN_BUDGET,
) -> None:
    """Log a complete token usage summary.

    Args:
        tracker: PipelineTokenTracker with usage data.
        budget: Total token budget for percentage calculation.
    """
    logger = get_execution_logger()
    display = TokenDisplay(budget)

    percentage = (tracker.total / budget) * 100

    logger.info(
        f"Token Summary - Routing: {tracker.routing_tokens}, "
        f"Memory: {tracker.memory_tokens}, "
        f"Planning: {tracker.planning_tokens}, "
        f"Reasoning: {tracker.reasoning_tokens}, "
        f"Validation: {tracker.validation_tokens}, "
        f"Total: {tracker.total:,} / {budget:,} ({percentage:.2f}%)",
        extra={
            "component": "SUMMARY",
            "tokens": tracker.total,
        },
    )


# =============================================================================
# Real-time Token Counter
# =============================================================================


class RealTimeTokenCounter:
    """Real-time token counter for live display.

    Tracks token accumulation across pipeline steps and provides
    formatted output for live monitoring.

    Attributes:
        budget: Total token budget.
        tracker: Internal PipelineTokenTracker.
        start_time: Execution start time.

    Example:
        >>> counter = RealTimeTokenCounter()
        >>> counter.add_routing(50)
        >>> counter.add_reasoning(450)
        >>> counter.get_status_line()
        'Tokens: 500 / 256,000 (0.20%) | Routing: 50 | Reasoning: 450'
    """

    def __init__(self, budget: int = TOTAL_TOKEN_BUDGET) -> None:
        """Initialize the counter.

        Args:
            budget: Total token budget.
        """
        self.budget = budget
        self.tracker = PipelineTokenTracker()
        self.start_time = datetime.now(timezone.utc)
        self._display = TokenDisplay(budget)

    def add_routing(self, tokens: int) -> None:
        """Add routing tokens."""
        self.tracker.record_routing(tokens)

    def add_memory(self, tokens: int) -> None:
        """Add memory tokens."""
        self.tracker.record_memory(tokens)

    def add_planning(self, tokens: int) -> None:
        """Add planning tokens."""
        self.tracker.record_planning(tokens)

    def add_reasoning(self, tokens: int) -> None:
        """Add reasoning tokens."""
        self.tracker.record_reasoning(tokens)

    def add_validation(self, tokens: int) -> None:
        """Add validation tokens."""
        self.tracker.record_validation(tokens)

    @property
    def total(self) -> int:
        """Get total tokens."""
        return self.tracker.total

    @property
    def percentage(self) -> float:
        """Get percentage of budget used."""
        return (self.total / self.budget) * 100

    def get_status_line(self) -> str:
        """Get a single-line status string.

        Returns:
            Status line like 'Tokens: 500 / 256,000 (0.20%) | Routing: 50 | ...'
        """
        parts = [f"Tokens: {self._display.format_budget_usage(self.total)}"]

        if self.tracker.routing_tokens > 0:
            parts.append(f"Routing: {self.tracker.routing_tokens}")
        if self.tracker.memory_tokens > 0:
            parts.append(f"Memory: {self.tracker.memory_tokens}")
        if self.tracker.planning_tokens > 0:
            parts.append(f"Planning: {self.tracker.planning_tokens}")
        if self.tracker.reasoning_tokens > 0:
            parts.append(f"Reasoning: {self.tracker.reasoning_tokens}")
        if self.tracker.validation_tokens > 0:
            parts.append(f"Validation: {self.tracker.validation_tokens}")

        return " | ".join(parts)

    def get_elapsed_time(self) -> float:
        """Get elapsed time in seconds since start."""
        return (datetime.now(timezone.utc) - self.start_time).total_seconds()

    def reset(self) -> None:
        """Reset the counter."""
        self.tracker.reset()
        self.start_time = datetime.now(timezone.utc)


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Constants
    "LOG_FILE_PATH",
    "MAX_LOG_SIZE",
    "BACKUP_COUNT",
    "TOTAL_TOKEN_BUDGET",
    # Color utilities
    "colorize",
    "COLORS",
    # Token tracking
    "PipelineTokenTracker",
    "TokenDisplay",
    "RealTimeTokenCounter",
    # Logging
    "SigilLogFormatter",
    "SigilLogAdapter",
    "setup_execution_logging",
    "get_execution_logger",
    "get_log_file_path",
    "log_pipeline_step",
    "log_token_summary",
]
