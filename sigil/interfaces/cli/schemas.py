"""API Schemas for Sigil v2 CLI.

This module defines Pydantic models for CLI logging entries, token usage
statistics, and session tracking. These schemas serve as the API contracts
for CLI monitoring and logging.

Key Models:
    - CliLogEntry: Structured log entry for CLI operations
    - TokenUsageStats: Token consumption for a single command
    - SessionStats: Session-level statistics
    - MonitorEntry: Parsed entry for real-time monitor display

Example:
    >>> from sigil.interfaces.cli.schemas import CliLogEntry, TokenUsageStats
    >>> entry = CliLogEntry(
    ...     timestamp=datetime.now(timezone.utc),
    ...     level="INFO",
    ...     command="orchestrate",
    ...     status="complete",
    ...     session_id="sess-abc123",
    ...     tokens_used=523,
    ... )
    >>> entry.model_dump_json()
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


# =============================================================================
# Enums
# =============================================================================


class LogLevel(str, Enum):
    """Log level enumeration."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class CommandStatus(str, Enum):
    """Command execution status."""
    START = "start"
    COMPLETE = "complete"
    ERROR = "error"
    TIMEOUT = "timeout"


class BudgetStatus(str, Enum):
    """Token budget status levels."""
    OK = "OK"           # < 50%
    WARNING = "WARNING"  # 50-80%
    CRITICAL = "CRITICAL"  # > 80%


# =============================================================================
# CLI Log Entry Schema
# =============================================================================


class CliLogEntry(BaseModel):
    """Structured log entry for CLI operations.

    This is the primary schema for CLI execution logs in JSON Lines format.
    Each entry represents a single log event with optional fields based on
    the event type (start, complete, error, timeout).

    Attributes:
        timestamp: When the event occurred (ISO 8601 UTC).
        level: Log level (INFO, ERROR, WARNING, DEBUG).
        command: CLI command name (e.g., "orchestrate", "memory query").
        status: Command status (start, complete, error, timeout).
        session_id: Current session identifier.
        user_input: Original user input (for start events).
        output: Command output (for complete events, truncated).
        tokens_used: Tokens consumed by this command.
        tokens_remaining: Budget remaining after this command.
        percentage: Percentage of total budget used.
        duration_ms: Command execution time in milliseconds.
        confidence: Confidence score (if applicable).
        error: Error message (for error events).
        error_type: Exception type name (for error events).
        metadata: Additional context data.

    Example:
        >>> entry = CliLogEntry(
        ...     timestamp=datetime.now(timezone.utc),
        ...     level="INFO",
        ...     command="orchestrate",
        ...     status="complete",
        ...     session_id="sess-abc123",
        ...     tokens_used=523,
        ...     tokens_remaining=255477,
        ...     percentage=0.20,
        ...     duration_ms=3333,
        ... )
    """

    timestamp: datetime = Field(
        description="When the event occurred (ISO 8601 UTC)",
    )
    level: str = Field(
        description="Log level (INFO, ERROR, WARNING, DEBUG)",
    )
    command: str = Field(
        description="CLI command name",
    )
    status: str = Field(
        description="Command status (start, complete, error, timeout)",
    )
    session_id: str = Field(
        description="Current session identifier",
    )
    user_input: Optional[str] = Field(
        default=None,
        description="Original user input (start events)",
    )
    output: Optional[str] = Field(
        default=None,
        description="Command output (complete events)",
    )
    tokens_used: Optional[int] = Field(
        default=None,
        ge=0,
        description="Tokens consumed by this command",
    )
    tokens_remaining: Optional[int] = Field(
        default=None,
        ge=0,
        description="Budget remaining after this command",
    )
    percentage: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=100.0,
        description="Percentage of total budget used",
    )
    duration_ms: Optional[int] = Field(
        default=None,
        ge=0,
        description="Command execution time in milliseconds",
    )
    confidence: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Confidence score (0.0 to 1.0)",
    )
    error: Optional[str] = Field(
        default=None,
        description="Error message (error events)",
    )
    error_type: Optional[str] = Field(
        default=None,
        description="Exception type name (error events)",
    )
    metadata: Optional[dict[str, Any]] = Field(
        default=None,
        description="Additional context data",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "timestamp": "2026-01-12T10:30:45.123Z",
                    "level": "INFO",
                    "command": "orchestrate",
                    "status": "start",
                    "session_id": "sess-abc123",
                    "user_input": "analyze Acme Corp",
                },
                {
                    "timestamp": "2026-01-12T10:30:48.456Z",
                    "level": "INFO",
                    "command": "orchestrate",
                    "status": "complete",
                    "session_id": "sess-abc123",
                    "tokens_used": 523,
                    "tokens_remaining": 255477,
                    "percentage": 0.20,
                    "duration_ms": 3333,
                    "confidence": 0.87,
                },
            ]
        }
    }

    @field_validator("timestamp", mode="before")
    @classmethod
    def parse_timestamp(cls, v):
        """Parse timestamp from string if needed."""
        if isinstance(v, str):
            # Handle Z suffix
            v = v.replace("Z", "+00:00")
            return datetime.fromisoformat(v)
        return v

    def to_json_line(self) -> str:
        """Serialize to JSON Lines format (single line).

        Returns:
            JSON string without extra whitespace.
        """
        return self.model_dump_json(exclude_none=True)

    @classmethod
    def from_json_line(cls, json_line: str) -> "CliLogEntry":
        """Parse from JSON Lines format.

        Args:
            json_line: Single line JSON string.

        Returns:
            Parsed CliLogEntry instance.
        """
        return cls.model_validate_json(json_line)


# =============================================================================
# Token Usage Stats Schema
# =============================================================================


class TokenUsageStats(BaseModel):
    """Token consumption statistics for a single command.

    Tracks token usage with timing information for analytics and
    budget management.

    Attributes:
        command: Command that consumed the tokens.
        tokens_used: Tokens consumed by this command.
        total_used: Cumulative tokens used in session.
        percentage: Percentage of budget consumed.
        timestamp: When the command completed.
        duration_ms: Command execution time.

    Example:
        >>> stats = TokenUsageStats(
        ...     command="orchestrate",
        ...     tokens_used=523,
        ...     total_used=1046,
        ...     percentage=0.41,
        ...     timestamp=datetime.now(timezone.utc),
        ...     duration_ms=3333,
        ... )
    """

    command: str = Field(
        description="Command that consumed the tokens",
    )
    tokens_used: int = Field(
        ge=0,
        description="Tokens consumed by this command",
    )
    total_used: int = Field(
        ge=0,
        description="Cumulative tokens used in session",
    )
    percentage: float = Field(
        ge=0.0,
        le=100.0,
        description="Percentage of budget consumed",
    )
    timestamp: datetime = Field(
        description="When the command completed",
    )
    duration_ms: int = Field(
        ge=0,
        description="Command execution time in milliseconds",
    )

    @field_validator("timestamp", mode="before")
    @classmethod
    def parse_timestamp(cls, v):
        """Parse timestamp from string if needed."""
        if isinstance(v, str):
            v = v.replace("Z", "+00:00")
            return datetime.fromisoformat(v)
        return v

    def get_budget_status(self) -> BudgetStatus:
        """Get budget status based on percentage used.

        Returns:
            BudgetStatus enum value.
        """
        if self.percentage < 50.0:
            return BudgetStatus.OK
        elif self.percentage < 80.0:
            return BudgetStatus.WARNING
        else:
            return BudgetStatus.CRITICAL


# =============================================================================
# Session Stats Schema
# =============================================================================


class SessionStats(BaseModel):
    """Session-level statistics.

    Provides aggregate information about a CLI session including
    command counts, token usage, and activity status.

    Attributes:
        session_id: Session identifier.
        commands_executed: Number of commands run.
        total_tokens_used: Total tokens consumed.
        percentage_used: Percentage of budget used.
        active: Whether the session is currently active.
        created_at: When the session started.
        last_command_at: When the last command was executed.

    Example:
        >>> stats = SessionStats(
        ...     session_id="sess-abc123",
        ...     commands_executed=5,
        ...     total_tokens_used=2500,
        ...     percentage_used=0.98,
        ...     active=True,
        ...     created_at=datetime.now(timezone.utc),
        ... )
    """

    session_id: str = Field(
        description="Session identifier",
    )
    commands_executed: int = Field(
        ge=0,
        description="Number of commands run",
    )
    total_tokens_used: int = Field(
        ge=0,
        description="Total tokens consumed",
    )
    percentage_used: float = Field(
        ge=0.0,
        le=100.0,
        description="Percentage of budget used",
    )
    active: bool = Field(
        description="Whether the session is currently active",
    )
    created_at: Optional[datetime] = Field(
        default=None,
        description="When the session started",
    )
    last_command_at: Optional[datetime] = Field(
        default=None,
        description="When the last command was executed",
    )

    @field_validator("created_at", "last_command_at", mode="before")
    @classmethod
    def parse_timestamps(cls, v):
        """Parse timestamps from string if needed."""
        if isinstance(v, str):
            v = v.replace("Z", "+00:00")
            return datetime.fromisoformat(v)
        return v

    def get_budget_status(self) -> BudgetStatus:
        """Get budget status based on percentage used.

        Returns:
            BudgetStatus enum value.
        """
        if self.percentage_used < 50.0:
            return BudgetStatus.OK
        elif self.percentage_used < 80.0:
            return BudgetStatus.WARNING
        else:
            return BudgetStatus.CRITICAL


# =============================================================================
# Monitor Entry Schema
# =============================================================================


class MonitorEntry(BaseModel):
    """Parsed entry for real-time monitor display.

    Simplified schema for the CLI monitor that focuses on display-relevant
    fields with formatting helpers.

    Attributes:
        time: Display time (HH:MM:SS format).
        command: Command name.
        status: Command status.
        tokens_used: Tokens for this command (or None for start).
        total_budget: Total budget with percentage.
        duration: Duration string (or None for start).
        is_error: Whether this is an error entry.

    Example:
        >>> entry = MonitorEntry(
        ...     time="10:30:48",
        ...     command="orchestrate",
        ...     status="OK",
        ...     tokens_used="523 tokens",
        ...     total_budget="256,000 (0.20%)",
        ...     duration="3.3s",
        ... )
    """

    time: str = Field(
        description="Display time (HH:MM:SS)",
    )
    command: str = Field(
        description="Command name",
    )
    status: str = Field(
        description="Command status (START, OK, ERROR, TIMEOUT)",
    )
    tokens_used: Optional[str] = Field(
        default=None,
        description="Tokens for this command",
    )
    total_budget: Optional[str] = Field(
        default=None,
        description="Total budget with percentage",
    )
    duration: Optional[str] = Field(
        default=None,
        description="Duration string",
    )
    is_error: bool = Field(
        default=False,
        description="Whether this is an error entry",
    )

    @classmethod
    def from_log_entry(
        cls,
        entry: CliLogEntry,
        total_budget: int = 256_000,
        running_total: int = 0,
    ) -> "MonitorEntry":
        """Create MonitorEntry from CliLogEntry.

        Args:
            entry: The log entry to convert.
            total_budget: Total token budget.
            running_total: Running total of tokens used.

        Returns:
            MonitorEntry for display.
        """
        # Format time
        time_str = entry.timestamp.strftime("%H:%M:%S")

        # Format status
        status_map = {
            "start": "START",
            "complete": "OK",
            "error": "ERROR",
            "timeout": "TIMEOUT",
        }
        status = status_map.get(entry.status, entry.status.upper())

        # Format tokens
        tokens_str = None
        if entry.tokens_used is not None:
            tokens_str = f"{entry.tokens_used} tokens"

        # Format budget
        budget_str = None
        if entry.percentage is not None:
            budget_str = f"{total_budget:,} ({entry.percentage:.2f}%)"

        # Format duration
        duration_str = None
        if entry.duration_ms is not None:
            if entry.duration_ms >= 1000:
                duration_str = f"{entry.duration_ms / 1000:.1f}s"
            else:
                duration_str = f"{entry.duration_ms}ms"

        return cls(
            time=time_str,
            command=entry.command,
            status=status,
            tokens_used=tokens_str,
            total_budget=budget_str,
            duration=duration_str,
            is_error=entry.status in ("error", "timeout"),
        )


# =============================================================================
# Log Filter Schema
# =============================================================================


class LogFilter(BaseModel):
    """Filter criteria for log queries.

    Used to filter log entries in the monitor and for log analysis.

    Attributes:
        command: Filter by command name.
        session_id: Filter by session ID.
        status: Filter by status.
        level: Filter by log level.
        start_time: Filter entries after this time.
        end_time: Filter entries before this time.
        min_tokens: Filter entries with at least this many tokens.
        max_tokens: Filter entries with at most this many tokens.
        has_error: Filter to only error entries.
    """

    command: Optional[str] = Field(
        default=None,
        description="Filter by command name",
    )
    session_id: Optional[str] = Field(
        default=None,
        description="Filter by session ID",
    )
    status: Optional[str] = Field(
        default=None,
        description="Filter by status",
    )
    level: Optional[str] = Field(
        default=None,
        description="Filter by log level",
    )
    start_time: Optional[datetime] = Field(
        default=None,
        description="Filter entries after this time",
    )
    end_time: Optional[datetime] = Field(
        default=None,
        description="Filter entries before this time",
    )
    min_tokens: Optional[int] = Field(
        default=None,
        ge=0,
        description="Filter entries with at least this many tokens",
    )
    max_tokens: Optional[int] = Field(
        default=None,
        ge=0,
        description="Filter entries with at most this many tokens",
    )
    has_error: Optional[bool] = Field(
        default=None,
        description="Filter to only error entries",
    )

    def matches(self, entry: CliLogEntry) -> bool:
        """Check if an entry matches this filter.

        Args:
            entry: The log entry to check.

        Returns:
            True if the entry matches all filter criteria.
        """
        if self.command and entry.command != self.command:
            return False
        if self.session_id and entry.session_id != self.session_id:
            return False
        if self.status and entry.status != self.status:
            return False
        if self.level and entry.level != self.level:
            return False
        if self.start_time and entry.timestamp < self.start_time:
            return False
        if self.end_time and entry.timestamp > self.end_time:
            return False
        if self.min_tokens is not None and entry.tokens_used is not None:
            if entry.tokens_used < self.min_tokens:
                return False
        if self.max_tokens is not None and entry.tokens_used is not None:
            if entry.tokens_used > self.max_tokens:
                return False
        if self.has_error is not None:
            is_error = entry.status in ("error", "timeout")
            if self.has_error != is_error:
                return False
        return True


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Enums
    "LogLevel",
    "CommandStatus",
    "BudgetStatus",
    # Models
    "CliLogEntry",
    "TokenUsageStats",
    "SessionStats",
    "MonitorEntry",
    "LogFilter",
]
