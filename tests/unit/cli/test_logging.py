"""Unit tests for CLI logging module.

Tests:
- CliLogFormatter produces valid JSON Lines
- All required fields present in log entries
- Token calculations are accurate
- Error logging captures exception details
- CliLogger tracks token totals correctly
- Log entry serialization/deserialization
"""

from __future__ import annotations

import json
import logging
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from sigil.interfaces.cli.logging import (
    CLI_LOG_FILE_PATH,
    CLI_TOKEN_BUDGET,
    CliLogEntry,
    CliLogFormatter,
    CliLogger,
    get_cli_logger,
    get_cli_logger_wrapper,
    reset_cli_logging,
    setup_cli_logging,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def temp_log_file(tmp_path):
    """Create a temporary log file."""
    log_file = tmp_path / "test-cli-execution.log"
    return log_file


@pytest.fixture
def cli_logger(temp_log_file):
    """Create a CliLogger with a temporary file."""
    reset_cli_logging()
    logger = setup_cli_logging(log_file=temp_log_file)
    return CliLogger(session_id="test-session-123", logger=logger)


@pytest.fixture(autouse=True)
def cleanup_logging():
    """Reset logging after each test."""
    yield
    reset_cli_logging()


# =============================================================================
# CliLogEntry Tests
# =============================================================================


class TestCliLogEntry:
    """Tests for CliLogEntry dataclass."""

    def test_create_basic_entry(self):
        """Test creating a basic log entry."""
        entry = CliLogEntry(
            timestamp=datetime.now(timezone.utc),
            level="INFO",
            command="orchestrate",
            status="start",
            session_id="sess-abc123",
            user_input="analyze Acme Corp",
        )

        assert entry.level == "INFO"
        assert entry.command == "orchestrate"
        assert entry.status == "start"
        assert entry.session_id == "sess-abc123"
        assert entry.user_input == "analyze Acme Corp"

    def test_to_dict_excludes_none(self):
        """Test that to_dict excludes None values."""
        entry = CliLogEntry(
            timestamp=datetime.now(timezone.utc),
            level="INFO",
            command="orchestrate",
            status="start",
            session_id="sess-abc123",
        )

        data = entry.to_dict()

        # Required fields present
        assert "timestamp" in data
        assert "level" in data
        assert "command" in data
        assert "status" in data
        assert "session_id" in data

        # Optional fields excluded when None
        assert "user_input" not in data
        assert "tokens_used" not in data
        assert "error" not in data

    def test_to_dict_includes_optional_fields(self):
        """Test that to_dict includes optional fields when set."""
        entry = CliLogEntry(
            timestamp=datetime.now(timezone.utc),
            level="INFO",
            command="orchestrate",
            status="complete",
            session_id="sess-abc123",
            tokens_used=523,
            tokens_remaining=255477,
            percentage=0.204,
            duration_ms=3333,
            confidence=0.87,
        )

        data = entry.to_dict()

        assert data["tokens_used"] == 523
        assert data["tokens_remaining"] == 255477
        assert data["percentage"] == 0.20  # Rounded
        assert data["duration_ms"] == 3333
        assert data["confidence"] == 0.87

    def test_to_json_single_line(self):
        """Test that to_json produces single-line JSON."""
        entry = CliLogEntry(
            timestamp=datetime(2026, 1, 12, 10, 30, 45, tzinfo=timezone.utc),
            level="INFO",
            command="orchestrate",
            status="start",
            session_id="sess-abc123",
        )

        json_str = entry.to_json()

        # Should be single line
        assert "\n" not in json_str

        # Should be valid JSON
        parsed = json.loads(json_str)
        assert parsed["command"] == "orchestrate"

    def test_from_dict(self):
        """Test creating entry from dictionary."""
        data = {
            "timestamp": "2026-01-12T10:30:45+00:00",
            "level": "INFO",
            "command": "orchestrate",
            "status": "complete",
            "session_id": "sess-abc123",
            "tokens_used": 523,
        }

        entry = CliLogEntry.from_dict(data)

        assert entry.level == "INFO"
        assert entry.command == "orchestrate"
        assert entry.tokens_used == 523
        assert entry.timestamp.year == 2026

    def test_from_json(self):
        """Test creating entry from JSON string."""
        json_str = '{"timestamp":"2026-01-12T10:30:45+00:00","level":"INFO","command":"orchestrate","status":"start","session_id":"sess-abc123"}'

        entry = CliLogEntry.from_json(json_str)

        assert entry.command == "orchestrate"
        assert entry.status == "start"

    def test_roundtrip(self):
        """Test serialization and deserialization roundtrip."""
        original = CliLogEntry(
            timestamp=datetime(2026, 1, 12, 10, 30, 45, tzinfo=timezone.utc),
            level="INFO",
            command="orchestrate",
            status="complete",
            session_id="sess-abc123",
            tokens_used=523,
            percentage=0.2,
        )

        json_str = original.to_json()
        restored = CliLogEntry.from_json(json_str)

        assert restored.command == original.command
        assert restored.status == original.status
        assert restored.tokens_used == original.tokens_used


# =============================================================================
# CliLogFormatter Tests
# =============================================================================


class TestCliLogFormatter:
    """Tests for CliLogFormatter class."""

    def test_format_produces_valid_json(self):
        """Test that format() produces valid JSON."""
        formatter = CliLogFormatter()

        # Create a log record
        record = logging.LogRecord(
            name="sigil.cli.execution",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Command START",
            args=(),
            exc_info=None,
        )
        record.command = "orchestrate"
        record.status = "start"
        record.session_id = "sess-abc123"
        record.user_input = "analyze Acme Corp"

        result = formatter.format(record)

        # Should be valid JSON
        parsed = json.loads(result)
        assert parsed["command"] == "orchestrate"
        assert parsed["status"] == "start"
        assert parsed["session_id"] == "sess-abc123"
        assert parsed["user_input"] == "analyze Acme Corp"

    def test_format_single_line(self):
        """Test that format() produces single-line output."""
        formatter = CliLogFormatter()

        record = logging.LogRecord(
            name="sigil.cli.execution",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Command COMPLETE",
            args=(),
            exc_info=None,
        )
        record.command = "orchestrate"
        record.status = "complete"
        record.session_id = "sess-abc123"
        record.tokens_used = 523

        result = formatter.format(record)

        # Should be single line (no newlines)
        assert "\n" not in result

    def test_format_includes_timestamp(self):
        """Test that format() includes ISO 8601 timestamp."""
        formatter = CliLogFormatter()

        record = logging.LogRecord(
            name="sigil.cli.execution",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Command START",
            args=(),
            exc_info=None,
        )
        record.command = "orchestrate"
        record.status = "start"
        record.session_id = "sess-abc123"

        result = formatter.format(record)
        parsed = json.loads(result)

        # Should have ISO 8601 timestamp
        assert "timestamp" in parsed
        # Should be parseable
        datetime.fromisoformat(parsed["timestamp"].replace("Z", "+00:00"))

    def test_format_with_tokens(self):
        """Test that format() includes token information."""
        formatter = CliLogFormatter()

        record = logging.LogRecord(
            name="sigil.cli.execution",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Command COMPLETE",
            args=(),
            exc_info=None,
        )
        record.command = "orchestrate"
        record.status = "complete"
        record.session_id = "sess-abc123"
        record.tokens_used = 523
        record.tokens_remaining = 255477
        record.percentage = 0.204
        record.duration_ms = 3333

        result = formatter.format(record)
        parsed = json.loads(result)

        assert parsed["tokens_used"] == 523
        assert parsed["tokens_remaining"] == 255477
        assert parsed["percentage"] == 0.20  # Rounded
        assert parsed["duration_ms"] == 3333

    def test_format_with_error(self):
        """Test that format() includes error information."""
        formatter = CliLogFormatter()

        record = logging.LogRecord(
            name="sigil.cli.execution",
            level=logging.ERROR,
            pathname="test.py",
            lineno=1,
            msg="Command ERROR",
            args=(),
            exc_info=None,
        )
        record.command = "orchestrate"
        record.status = "error"
        record.session_id = "sess-abc123"
        record.error = "Connection timeout"
        record.error_type = "TimeoutError"

        result = formatter.format(record)
        parsed = json.loads(result)

        assert parsed["level"] == "ERROR"
        assert parsed["status"] == "error"
        assert parsed["error"] == "Connection timeout"
        assert parsed["error_type"] == "TimeoutError"


# =============================================================================
# CliLogger Tests
# =============================================================================


class TestCliLogger:
    """Tests for CliLogger class."""

    def test_log_command_start(self, cli_logger, temp_log_file):
        """Test logging command start event."""
        cli_logger.log_command_start("orchestrate", "analyze Acme Corp")

        # Give time for file write
        time.sleep(0.1)

        content = temp_log_file.read_text()
        assert content  # Not empty

        lines = content.strip().split("\n")
        assert len(lines) >= 1

        entry = json.loads(lines[-1])
        assert entry["command"] == "orchestrate"
        assert entry["status"] == "start"
        assert entry["user_input"] == "analyze Acme Corp"

    def test_log_command_complete(self, cli_logger, temp_log_file):
        """Test logging command complete event."""
        # Start then complete
        cli_logger.log_command_start("orchestrate", "analyze Acme Corp")
        time.sleep(0.1)  # Small delay for timing
        cli_logger.log_command_complete(
            "orchestrate",
            tokens_used=523,
            confidence=0.87,
        )

        # Give time for file write
        time.sleep(0.1)

        content = temp_log_file.read_text()
        lines = content.strip().split("\n")

        # Should have start and complete entries
        assert len(lines) >= 2

        complete_entry = json.loads(lines[-1])
        assert complete_entry["command"] == "orchestrate"
        assert complete_entry["status"] == "complete"
        assert complete_entry["tokens_used"] == 523
        assert complete_entry["confidence"] == 0.87
        assert "duration_ms" in complete_entry

    def test_log_command_error(self, cli_logger, temp_log_file):
        """Test logging command error event."""
        cli_logger.log_command_start("orchestrate", "analyze Acme Corp")
        cli_logger.log_command_error(
            "orchestrate",
            ValueError("Invalid input"),
        )

        time.sleep(0.1)

        content = temp_log_file.read_text()
        lines = content.strip().split("\n")

        error_entry = json.loads(lines[-1])
        assert error_entry["status"] == "error"
        assert error_entry["error"] == "Invalid input"
        assert error_entry["error_type"] == "ValueError"

    def test_token_tracking(self, cli_logger):
        """Test running token total tracking."""
        assert cli_logger.total_tokens_used == 0

        cli_logger.log_command_start("cmd1", "test")
        cli_logger.log_command_complete("cmd1", tokens_used=100)

        assert cli_logger.total_tokens_used == 100

        cli_logger.log_command_start("cmd2", "test")
        cli_logger.log_command_complete("cmd2", tokens_used=200)

        assert cli_logger.total_tokens_used == 300

    def test_get_token_summary(self, cli_logger):
        """Test token usage summary."""
        cli_logger.log_command_start("cmd1", "test")
        cli_logger.log_command_complete("cmd1", tokens_used=1000)

        summary = cli_logger.get_token_summary()

        assert summary["total_used"] == 1000
        assert summary["budget"] == CLI_TOKEN_BUDGET
        assert summary["remaining"] == CLI_TOKEN_BUDGET - 1000
        assert summary["percentage"] == (1000 / CLI_TOKEN_BUDGET) * 100

    def test_reset_tokens(self, cli_logger):
        """Test token counter reset."""
        cli_logger.log_command_start("cmd1", "test")
        cli_logger.log_command_complete("cmd1", tokens_used=500)

        assert cli_logger.total_tokens_used == 500

        cli_logger.reset_tokens()

        assert cli_logger.total_tokens_used == 0

    def test_duration_calculation(self, cli_logger, temp_log_file):
        """Test that duration is calculated correctly."""
        cli_logger.log_command_start("orchestrate", "test")
        time.sleep(0.2)  # 200ms delay
        cli_logger.log_command_complete("orchestrate", tokens_used=100)

        time.sleep(0.1)

        content = temp_log_file.read_text()
        lines = content.strip().split("\n")

        complete_entry = json.loads(lines[-1])
        duration = complete_entry.get("duration_ms", 0)

        # Should be at least 200ms (with some tolerance)
        assert duration >= 150  # Allow for timing variance

    def test_session_id_update(self, cli_logger):
        """Test updating session ID."""
        assert cli_logger.session_id == "test-session-123"

        cli_logger.set_session_id("new-session-456")

        assert cli_logger.session_id == "new-session-456"


# =============================================================================
# Setup Function Tests
# =============================================================================


class TestSetupFunctions:
    """Tests for logging setup functions."""

    def test_setup_cli_logging(self, temp_log_file):
        """Test setting up CLI logging."""
        logger = setup_cli_logging(log_file=temp_log_file)

        assert logger is not None
        assert logger.name == "sigil.cli.execution"
        assert len(logger.handlers) >= 1

    def test_get_cli_logger(self, temp_log_file):
        """Test getting the CLI logger."""
        setup_cli_logging(log_file=temp_log_file)
        logger = get_cli_logger()

        assert logger is not None
        assert logger.name == "sigil.cli.execution"

    def test_get_cli_logger_wrapper(self, temp_log_file):
        """Test getting the CLI logger wrapper."""
        setup_cli_logging(log_file=temp_log_file)
        wrapper = get_cli_logger_wrapper(session_id="test-session")

        assert wrapper is not None
        assert wrapper.session_id == "test-session"

    def test_reset_cli_logging(self, temp_log_file):
        """Test resetting CLI logging."""
        setup_cli_logging(log_file=temp_log_file)
        reset_cli_logging()

        # After reset, should create new logger
        logger = get_cli_logger()
        assert logger is not None


# =============================================================================
# Integration Tests
# =============================================================================


class TestLoggingIntegration:
    """Integration tests for CLI logging."""

    def test_multiple_commands(self, cli_logger, temp_log_file):
        """Test logging multiple commands in sequence."""
        commands = [
            ("orchestrate", "analyze company", 500),
            ("memory query", "preferences", 100),
            ("contracts test", "lead_qualification", 200),
        ]

        for cmd, input_text, tokens in commands:
            cli_logger.log_command_start(cmd, input_text)
            cli_logger.log_command_complete(cmd, tokens_used=tokens)

        time.sleep(0.1)

        content = temp_log_file.read_text()
        lines = content.strip().split("\n")

        # Should have 6 entries (2 per command)
        assert len(lines) == 6

        # Total tokens should be accumulated
        assert cli_logger.total_tokens_used == 800

    def test_error_recovery(self, cli_logger, temp_log_file):
        """Test logging errors and continuing."""
        cli_logger.log_command_start("cmd1", "test")
        cli_logger.log_command_error("cmd1", RuntimeError("Test error"))

        # Should be able to log new commands after error
        cli_logger.log_command_start("cmd2", "test2")
        cli_logger.log_command_complete("cmd2", tokens_used=100)

        time.sleep(0.1)

        content = temp_log_file.read_text()
        lines = content.strip().split("\n")

        # Should have all 4 entries
        assert len(lines) == 4

    def test_json_lines_format(self, cli_logger, temp_log_file):
        """Test that log file is valid JSON Lines format."""
        # Log several commands
        for i in range(5):
            cli_logger.log_command_start(f"cmd{i}", f"input{i}")
            cli_logger.log_command_complete(f"cmd{i}", tokens_used=i * 10)

        time.sleep(0.1)

        content = temp_log_file.read_text()
        lines = content.strip().split("\n")

        # Each line should be valid JSON
        for line in lines:
            parsed = json.loads(line)  # Should not raise
            assert "timestamp" in parsed
            assert "level" in parsed
            assert "command" in parsed


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error conditions."""

    def test_long_output_truncation(self, cli_logger, temp_log_file):
        """Test that long output is truncated."""
        cli_logger.log_command_start("test", "input")
        cli_logger.log_command_complete(
            "test",
            tokens_used=100,
            output="x" * 1000,  # Long output
        )

        time.sleep(0.1)

        content = temp_log_file.read_text()
        lines = content.strip().split("\n")

        complete_entry = json.loads(lines[-1])
        output = complete_entry.get("output", "")

        # Should be truncated to ~500 chars
        assert len(output) <= 510  # 500 + "..."

    def test_missing_start_time(self, cli_logger, temp_log_file):
        """Test completing command without start."""
        # Complete without start
        cli_logger.log_command_complete("orphan_cmd", tokens_used=100)

        time.sleep(0.1)

        content = temp_log_file.read_text()
        lines = content.strip().split("\n")

        entry = json.loads(lines[-1])
        # Should still log but without duration
        assert entry["command"] == "orphan_cmd"

    def test_special_characters_in_input(self, cli_logger, temp_log_file):
        """Test handling special characters in user input."""
        special_input = 'Test with "quotes" and\nnewlines and unicode: emoji \u2764'

        cli_logger.log_command_start("test", special_input)

        time.sleep(0.1)

        content = temp_log_file.read_text()
        lines = content.strip().split("\n")

        # Should still produce valid JSON
        entry = json.loads(lines[-1])
        assert "Test with" in entry["user_input"]

    def test_zero_tokens(self, cli_logger, temp_log_file):
        """Test logging command with zero tokens."""
        cli_logger.log_command_start("test", "input")
        cli_logger.log_command_complete("test", tokens_used=0)

        time.sleep(0.1)

        content = temp_log_file.read_text()
        lines = content.strip().split("\n")

        entry = json.loads(lines[-1])
        assert entry["tokens_used"] == 0
