"""Structured Logging for Sigil v2 CLI.

This module provides CLI-specific structured logging in JSON Lines format,
extending the base SigilLogFormatter with CLI-specific fields for command
execution tracking, token usage, and session management.

Key Components:
    - CliLogFormatter: JSON Lines log formatter for machine parsing
    - setup_cli_logging: Configure CLI logging with file handlers
    - CliLogger: Wrapper for logging CLI-specific events

Log Format (JSON Lines):
    Each log entry is a valid JSON object on a single line containing:
    - timestamp: ISO 8601 format (UTC)
    - level: INFO, ERROR, WARNING, DEBUG
    - command: Name of the CLI command
    - status: start, complete, error, timeout
    - session_id: Current session identifier
    - user_input: Original user input (for start events)
    - tokens_used: Tokens consumed by this command
    - tokens_remaining: Budget remaining after this command
    - percentage: Percentage of budget used
    - duration_ms: Command execution time
    - error: Error message (for error events)
    - error_type: Exception type name (for error events)

Example:
    {"timestamp": "2026-01-12T10:30:45.123Z", "level": "INFO", "command": "orchestrate", "status": "start", "session_id": "sess-abc123", "user_input": "analyze Acme Corp"}
    {"timestamp": "2026-01-12T10:30:48.456Z", "level": "INFO", "command": "orchestrate", "status": "complete", "session_id": "sess-abc123", "tokens_used": 523, "tokens_remaining": 255477, "percentage": 0.20, "duration_ms": 3333, "confidence": 0.87}
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Optional

from sigil.interfaces.cli.monitoring import SigilLogFormatter


# =============================================================================
# Constants
# =============================================================================

# Log file path for CLI execution logs
CLI_LOG_FILE_PATH = Path("outputs/cli-execution.log")

# Max log file size (10MB)
CLI_LOG_MAX_BYTES = 10 * 1024 * 1024

# Number of backup files to keep
CLI_LOG_BACKUP_COUNT = 5

# Default token budget (256K context window)
CLI_TOKEN_BUDGET = 256_000

# Log levels
LOG_LEVELS = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}


# =============================================================================
# CLI Log Entry Data Class
# =============================================================================


@dataclass
class CliLogEntry:
    """Structured log entry for CLI operations.

    Represents a single log entry in JSON Lines format with all
    CLI-specific fields for command tracking and token usage.

    Attributes:
        timestamp: When the event occurred (ISO 8601 UTC).
        level: Log level (INFO, ERROR, WARNING, DEBUG).
        command: CLI command name.
        status: Command status (start, complete, error, timeout).
        session_id: Current session identifier.
        user_input: Original user input (start events).
        output: Command output (complete events).
        tokens_used: Tokens consumed by this command.
        tokens_remaining: Budget remaining.
        percentage: Percentage of budget used.
        duration_ms: Execution time in milliseconds.
        confidence: Confidence score (if applicable).
        error: Error message (error events).
        error_type: Exception type name (error events).
        metadata: Additional context data.

    Example:
        >>> entry = CliLogEntry(
        ...     timestamp=datetime.now(timezone.utc),
        ...     level="INFO",
        ...     command="orchestrate",
        ...     status="start",
        ...     session_id="sess-abc123",
        ...     user_input="analyze Acme Corp",
        ... )
        >>> entry.to_json()
        '{"timestamp": "2026-01-12T10:30:45.123Z", ...}'
    """

    timestamp: datetime
    level: str
    command: str
    status: str
    session_id: str
    user_input: Optional[str] = None
    output: Optional[str] = None
    tokens_used: Optional[int] = None
    tokens_remaining: Optional[int] = None
    percentage: Optional[float] = None
    duration_ms: Optional[int] = None
    confidence: Optional[float] = None
    error: Optional[str] = None
    error_type: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary with only non-None values.

        Returns:
            Dictionary with log entry data (None values excluded).
        """
        result = {
            "timestamp": self.timestamp.isoformat(),
            "level": self.level,
            "command": self.command,
            "status": self.status,
            "session_id": self.session_id,
        }

        # Add optional fields if present
        if self.user_input is not None:
            result["user_input"] = self.user_input
        if self.output is not None:
            result["output"] = self.output
        if self.tokens_used is not None:
            result["tokens_used"] = self.tokens_used
        if self.tokens_remaining is not None:
            result["tokens_remaining"] = self.tokens_remaining
        if self.percentage is not None:
            result["percentage"] = round(self.percentage, 2)
        if self.duration_ms is not None:
            result["duration_ms"] = self.duration_ms
        if self.confidence is not None:
            result["confidence"] = round(self.confidence, 2)
        if self.error is not None:
            result["error"] = self.error
        if self.error_type is not None:
            result["error_type"] = self.error_type
        if self.metadata:
            result["metadata"] = self.metadata

        return result

    def to_json(self) -> str:
        """Serialize to JSON string (single line).

        Returns:
            JSON string representation.
        """
        return json.dumps(self.to_dict(), separators=(",", ":"))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CliLogEntry":
        """Create from dictionary.

        Args:
            data: Dictionary with log entry data.

        Returns:
            New CliLogEntry instance.
        """
        timestamp = data.get("timestamp")
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        elif timestamp is None:
            timestamp = datetime.now(timezone.utc)

        return cls(
            timestamp=timestamp,
            level=data.get("level", "INFO"),
            command=data.get("command", "unknown"),
            status=data.get("status", "unknown"),
            session_id=data.get("session_id", "unknown"),
            user_input=data.get("user_input"),
            output=data.get("output"),
            tokens_used=data.get("tokens_used"),
            tokens_remaining=data.get("tokens_remaining"),
            percentage=data.get("percentage"),
            duration_ms=data.get("duration_ms"),
            confidence=data.get("confidence"),
            error=data.get("error"),
            error_type=data.get("error_type"),
            metadata=data.get("metadata", {}),
        )

    @classmethod
    def from_json(cls, json_str: str) -> "CliLogEntry":
        """Create from JSON string.

        Args:
            json_str: JSON string with log entry data.

        Returns:
            New CliLogEntry instance.
        """
        return cls.from_dict(json.loads(json_str))


# =============================================================================
# CLI Log Formatter (JSON Lines)
# =============================================================================


class CliLogFormatter(logging.Formatter):
    """JSON Lines formatter for CLI execution logs.

    Extends logging.Formatter to produce JSON Lines output (one valid
    JSON object per line) with CLI-specific fields for command tracking
    and token usage.

    Extracts CLI-specific fields from the log record's `extra` dict:
    - command: CLI command name
    - status: start, complete, error, timeout
    - session_id: Session identifier
    - user_input: User's original input
    - tokens_used: Tokens consumed
    - tokens_remaining: Budget remaining
    - percentage: Budget percentage used
    - duration_ms: Execution time
    - confidence: Confidence score
    - error_details: Error information

    Example:
        >>> handler = logging.FileHandler("cli-execution.log")
        >>> handler.setFormatter(CliLogFormatter())
        >>> logger.addHandler(handler)
        >>> logger.info("Command start", extra={
        ...     "command": "orchestrate",
        ...     "status": "start",
        ...     "session_id": "sess-abc123",
        ...     "user_input": "analyze Acme Corp",
        ... })
    """

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON Lines entry.

        Args:
            record: The log record to format.

        Returns:
            JSON string (single line) with log entry.
        """
        # Build entry from record
        entry_data = {
            "timestamp": datetime.fromtimestamp(
                record.created, tz=timezone.utc
            ).isoformat(),
            "level": record.levelname,
            "command": getattr(record, "command", "unknown"),
            "status": getattr(record, "status", "info"),
            "session_id": getattr(record, "session_id", "unknown"),
        }

        # Add optional fields from record extras
        optional_fields = [
            "user_input",
            "output",
            "tokens_used",
            "tokens_remaining",
            "percentage",
            "duration_ms",
            "confidence",
            "error",
            "error_type",
        ]

        for field_name in optional_fields:
            value = getattr(record, field_name, None)
            if value is not None:
                # Round floats to 2 decimal places
                if isinstance(value, float):
                    value = round(value, 2)
                entry_data[field_name] = value

        # Add metadata if present
        metadata = getattr(record, "metadata", None)
        if metadata:
            entry_data["metadata"] = metadata

        # Add message if it contains additional info (not just status)
        if record.msg and record.msg not in ("Command START", "Command COMPLETE", "Command ERROR"):
            # Only include meaningful messages
            msg = record.getMessage()
            if msg and not msg.startswith("Command"):
                entry_data["message"] = msg

        return json.dumps(entry_data, separators=(",", ":"))


# =============================================================================
# CLI Logger Class
# =============================================================================


class CliLogger:
    """High-level logger for CLI command tracking.

    Provides convenient methods for logging command lifecycle events
    with automatic token tracking and timing.

    Attributes:
        logger: Underlying Python logger instance.
        session_id: Current session ID.
        total_tokens_used: Running total of tokens consumed.
        token_budget: Maximum token budget.

    Example:
        >>> cli_logger = CliLogger(session_id="sess-abc123")
        >>> cli_logger.log_command_start("orchestrate", "analyze Acme Corp")
        >>> # ... command execution ...
        >>> cli_logger.log_command_complete("orchestrate", tokens_used=523, duration_ms=3333)
    """

    def __init__(
        self,
        session_id: str = "unknown",
        token_budget: int = CLI_TOKEN_BUDGET,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        """Initialize CLI logger.

        Args:
            session_id: Current session identifier.
            token_budget: Maximum token budget.
            logger: Existing logger to use (creates new if None).
        """
        self.session_id = session_id
        self.token_budget = token_budget
        self.total_tokens_used = 0
        self._command_start_times: dict[str, float] = {}

        # Use provided logger or get/create CLI logger
        if logger:
            self.logger = logger
        else:
            self.logger = get_cli_logger()

    def set_session_id(self, session_id: str) -> None:
        """Update the session ID.

        Args:
            session_id: New session identifier.
        """
        self.session_id = session_id

    def log_command_start(
        self,
        command: str,
        user_input: str,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        """Log command start event.

        Args:
            command: Command name (e.g., "orchestrate").
            user_input: User's original input.
            metadata: Additional context data.
        """
        # Record start time
        self._command_start_times[command] = time.perf_counter()

        self.logger.info(
            "Command START",
            extra={
                "command": command,
                "status": "start",
                "session_id": self.session_id,
                "user_input": user_input,
                "metadata": metadata or {},
            },
        )

    def log_command_complete(
        self,
        command: str,
        tokens_used: int,
        confidence: Optional[float] = None,
        output: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        """Log command completion event.

        Args:
            command: Command name.
            tokens_used: Tokens consumed by this command.
            confidence: Confidence score (if applicable).
            output: Command output (truncated if long).
            metadata: Additional context data.
        """
        # Calculate duration
        start_time = self._command_start_times.pop(command, None)
        duration_ms = None
        if start_time:
            duration_ms = int((time.perf_counter() - start_time) * 1000)

        # Update running total
        self.total_tokens_used += tokens_used
        tokens_remaining = self.token_budget - self.total_tokens_used
        percentage = (self.total_tokens_used / self.token_budget) * 100

        # Truncate long output
        if output and len(output) > 500:
            output = output[:497] + "..."

        self.logger.info(
            "Command COMPLETE",
            extra={
                "command": command,
                "status": "complete",
                "session_id": self.session_id,
                "tokens_used": tokens_used,
                "tokens_remaining": max(0, tokens_remaining),
                "percentage": percentage,
                "duration_ms": duration_ms,
                "confidence": confidence,
                "output": output,
                "metadata": metadata or {},
            },
        )

    def log_command_error(
        self,
        command: str,
        error: Exception,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        """Log command error event.

        Args:
            command: Command name.
            error: The exception that occurred.
            metadata: Additional context data.
        """
        # Calculate duration if start time was recorded
        start_time = self._command_start_times.pop(command, None)
        duration_ms = None
        if start_time:
            duration_ms = int((time.perf_counter() - start_time) * 1000)

        self.logger.error(
            "Command ERROR",
            extra={
                "command": command,
                "status": "error",
                "session_id": self.session_id,
                "error": str(error),
                "error_type": type(error).__name__,
                "duration_ms": duration_ms,
                "metadata": metadata or {},
            },
        )

    def log_command_timeout(
        self,
        command: str,
        timeout_seconds: float,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        """Log command timeout event.

        Args:
            command: Command name.
            timeout_seconds: Timeout duration.
            metadata: Additional context data.
        """
        # Calculate duration
        start_time = self._command_start_times.pop(command, None)
        duration_ms = None
        if start_time:
            duration_ms = int((time.perf_counter() - start_time) * 1000)

        self.logger.warning(
            "Command TIMEOUT",
            extra={
                "command": command,
                "status": "timeout",
                "session_id": self.session_id,
                "duration_ms": duration_ms,
                "error": f"Command timed out after {timeout_seconds}s",
                "error_type": "TimeoutError",
                "metadata": metadata or {},
            },
        )

    def get_token_summary(self) -> dict[str, Any]:
        """Get current token usage summary.

        Returns:
            Dictionary with token usage statistics.
        """
        return {
            "total_used": self.total_tokens_used,
            "budget": self.token_budget,
            "remaining": max(0, self.token_budget - self.total_tokens_used),
            "percentage": (self.total_tokens_used / self.token_budget) * 100,
        }

    def reset_tokens(self) -> None:
        """Reset the token counter (for new sessions)."""
        self.total_tokens_used = 0


# =============================================================================
# Logging Setup Functions
# =============================================================================

# Global CLI logger instance
_cli_logger: Optional[logging.Logger] = None
_cli_logger_wrapper: Optional[CliLogger] = None


def setup_cli_logging(
    log_file: Optional[Path] = None,
    level: int = logging.INFO,
    max_bytes: int = CLI_LOG_MAX_BYTES,
    backup_count: int = CLI_LOG_BACKUP_COUNT,
) -> logging.Logger:
    """Set up CLI logging with JSON Lines format.

    Creates a logger with:
    - RotatingFileHandler with JSON Lines formatter
    - Size-based rotation (10MB default)
    - 5 backup files

    Args:
        log_file: Path to log file (defaults to CLI_LOG_FILE_PATH).
        level: Logging level (default INFO).
        max_bytes: Max size before rotation (default 10MB).
        backup_count: Number of backup files (default 5).

    Returns:
        Configured logger instance.
    """
    global _cli_logger

    if _cli_logger is not None:
        return _cli_logger

    log_path = log_file or CLI_LOG_FILE_PATH

    # Ensure parent directory exists
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # Create logger
    logger = logging.getLogger("sigil.cli.execution")
    logger.setLevel(level)

    # Remove existing handlers
    logger.handlers.clear()

    # Create rotating file handler
    file_handler = RotatingFileHandler(
        filename=log_path,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    file_handler.setLevel(level)

    # Set JSON Lines formatter
    formatter = CliLogFormatter()
    file_handler.setFormatter(formatter)

    # Add handler
    logger.addHandler(file_handler)

    # Prevent propagation to root logger
    logger.propagate = False

    _cli_logger = logger
    return logger


def get_cli_logger() -> logging.Logger:
    """Get the CLI logger, creating it if necessary.

    Returns:
        The CLI execution logger instance.
    """
    global _cli_logger
    if _cli_logger is None:
        return setup_cli_logging()
    return _cli_logger


def get_cli_logger_wrapper(session_id: str = "unknown") -> CliLogger:
    """Get the CLI logger wrapper for convenient logging.

    Args:
        session_id: Current session identifier.

    Returns:
        CliLogger wrapper instance.
    """
    global _cli_logger_wrapper
    if _cli_logger_wrapper is None:
        _cli_logger_wrapper = CliLogger(session_id=session_id)
    else:
        _cli_logger_wrapper.set_session_id(session_id)
    return _cli_logger_wrapper


def reset_cli_logging() -> None:
    """Reset CLI logging (for testing)."""
    global _cli_logger, _cli_logger_wrapper
    if _cli_logger:
        _cli_logger.handlers.clear()
    _cli_logger = None
    _cli_logger_wrapper = None


def get_cli_log_file_path() -> Path:
    """Get the path to the CLI execution log file.

    Returns:
        Path to the log file.
    """
    return CLI_LOG_FILE_PATH


# =============================================================================
# Command Logging Decorator
# =============================================================================


def log_command(command_name: str):
    """Decorator for automatic command logging.

    Wraps a command function to automatically log start/complete/error
    events with timing and token tracking.

    Args:
        command_name: Name of the command for logging.

    Returns:
        Decorator function.

    Example:
        >>> @log_command("orchestrate")
        ... async def orchestrate(task: str) -> dict:
        ...     result = await do_orchestration(task)
        ...     return {"result": result, "tokens_used": 523}
    """
    import functools

    def decorator(func):
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Get session ID from context if available
            session_id = kwargs.get("session_id", "unknown")
            user_input = kwargs.get("task") or kwargs.get("query") or str(args[0]) if args else ""

            cli_logger = get_cli_logger_wrapper(session_id)
            cli_logger.log_command_start(command_name, user_input)

            try:
                result = await func(*args, **kwargs)

                # Extract token info from result
                tokens_used = 0
                confidence = None
                output = None

                if isinstance(result, dict):
                    tokens_used = result.get("tokens_used", 0)
                    confidence = result.get("confidence")
                    output = result.get("result") or result.get("output")

                cli_logger.log_command_complete(
                    command_name,
                    tokens_used=tokens_used,
                    confidence=confidence,
                    output=str(output) if output else None,
                )
                return result

            except Exception as e:
                cli_logger.log_command_error(command_name, e)
                raise

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            session_id = kwargs.get("session_id", "unknown")
            user_input = kwargs.get("task") or kwargs.get("query") or str(args[0]) if args else ""

            cli_logger = get_cli_logger_wrapper(session_id)
            cli_logger.log_command_start(command_name, user_input)

            try:
                result = func(*args, **kwargs)

                tokens_used = 0
                confidence = None
                output = None

                if isinstance(result, dict):
                    tokens_used = result.get("tokens_used", 0)
                    confidence = result.get("confidence")
                    output = result.get("result") or result.get("output")

                cli_logger.log_command_complete(
                    command_name,
                    tokens_used=tokens_used,
                    confidence=confidence,
                    output=str(output) if output else None,
                )
                return result

            except Exception as e:
                cli_logger.log_command_error(command_name, e)
                raise

        # Return appropriate wrapper based on function type
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Constants
    "CLI_LOG_FILE_PATH",
    "CLI_LOG_MAX_BYTES",
    "CLI_LOG_BACKUP_COUNT",
    "CLI_TOKEN_BUDGET",
    # Classes
    "CliLogEntry",
    "CliLogFormatter",
    "CliLogger",
    # Functions
    "setup_cli_logging",
    "get_cli_logger",
    "get_cli_logger_wrapper",
    "reset_cli_logging",
    "get_cli_log_file_path",
    "log_command",
]
