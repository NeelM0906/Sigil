"""Integration tests for CLI execution monitor.

Tests:
- Real-time monitoring reads logs correctly
- Token totals accumulate correctly
- Color formatting works
- Command filtering works
- File rotation handling
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.cli_monitor import (
    LogEntry,
    TokenTracker,
    MonitorDisplay,
    CliMonitor,
    Colors,
    colorize,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def temp_log_file(tmp_path):
    """Create a temporary log file."""
    log_file = tmp_path / "cli-execution.log"
    log_file.touch()
    return log_file


@pytest.fixture
def sample_log_entries():
    """Generate sample JSON Lines log entries."""
    base_time = datetime(2026, 1, 12, 10, 30, 45, tzinfo=timezone.utc)

    entries = [
        {
            "timestamp": base_time.isoformat(),
            "level": "INFO",
            "command": "orchestrate",
            "status": "start",
            "session_id": "sess-abc123",
            "user_input": "analyze Acme Corp",
        },
        {
            "timestamp": (base_time.replace(second=48)).isoformat(),
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
        {
            "timestamp": (base_time.replace(second=50)).isoformat(),
            "level": "INFO",
            "command": "memory query",
            "status": "start",
            "session_id": "sess-abc123",
            "user_input": "preferences",
        },
        {
            "timestamp": (base_time.replace(second=51)).isoformat(),
            "level": "INFO",
            "command": "memory query",
            "status": "complete",
            "session_id": "sess-abc123",
            "tokens_used": 45,
            "tokens_remaining": 255432,
            "percentage": 0.22,
            "duration_ms": 1200,
        },
    ]
    return entries


@pytest.fixture
def populated_log_file(temp_log_file, sample_log_entries):
    """Create a log file with sample entries."""
    with open(temp_log_file, "w") as f:
        for entry in sample_log_entries:
            f.write(json.dumps(entry) + "\n")
    return temp_log_file


# =============================================================================
# LogEntry Tests
# =============================================================================


class TestLogEntry:
    """Tests for LogEntry parsing."""

    def test_parse_start_entry(self, sample_log_entries):
        """Test parsing a start entry."""
        entry = LogEntry.from_json_line(json.dumps(sample_log_entries[0]))

        assert entry is not None
        assert entry.command == "orchestrate"
        assert entry.status == "start"
        assert entry.user_input == "analyze Acme Corp"
        assert entry.tokens_used is None

    def test_parse_complete_entry(self, sample_log_entries):
        """Test parsing a complete entry."""
        entry = LogEntry.from_json_line(json.dumps(sample_log_entries[1]))

        assert entry is not None
        assert entry.command == "orchestrate"
        assert entry.status == "complete"
        assert entry.tokens_used == 523
        assert entry.percentage == 0.20
        assert entry.duration_ms == 3333
        assert entry.confidence == 0.87

    def test_parse_invalid_json(self):
        """Test handling invalid JSON."""
        entry = LogEntry.from_json_line("not valid json")

        assert entry is None

    def test_format_time(self, sample_log_entries):
        """Test time formatting."""
        entry = LogEntry.from_json_line(json.dumps(sample_log_entries[0]))

        time_str = entry.format_time()

        assert time_str == "10:30:45"

    def test_format_status_start(self, sample_log_entries):
        """Test status formatting for start."""
        entry = LogEntry.from_json_line(json.dumps(sample_log_entries[0]))

        status = entry.format_status()

        # Should contain START
        assert "START" in status

    def test_format_status_complete(self, sample_log_entries):
        """Test status formatting for complete."""
        entry = LogEntry.from_json_line(json.dumps(sample_log_entries[1]))

        status = entry.format_status()

        # Should contain OK
        assert "OK" in status

    def test_format_tokens(self, sample_log_entries):
        """Test token formatting."""
        entry = LogEntry.from_json_line(json.dumps(sample_log_entries[1]))

        tokens = entry.format_tokens()

        assert "523" in tokens
        assert "tokens" in tokens

    def test_format_tokens_none(self, sample_log_entries):
        """Test token formatting when None."""
        entry = LogEntry.from_json_line(json.dumps(sample_log_entries[0]))

        tokens = entry.format_tokens()

        assert "-" in tokens

    def test_format_duration(self, sample_log_entries):
        """Test duration formatting."""
        entry = LogEntry.from_json_line(json.dumps(sample_log_entries[1]))

        duration = entry.format_duration()

        assert "3.3s" in duration

    def test_format_duration_ms(self):
        """Test duration formatting for short durations."""
        entry = LogEntry(
            timestamp=datetime.now(timezone.utc),
            level="INFO",
            command="test",
            status="complete",
            session_id="sess-123",
            duration_ms=500,
        )

        duration = entry.format_duration()

        assert "500ms" in duration

    def test_is_error(self):
        """Test error detection."""
        error_entry = LogEntry(
            timestamp=datetime.now(timezone.utc),
            level="ERROR",
            command="test",
            status="error",
            session_id="sess-123",
            error="Connection failed",
            error_type="ConnectionError",
        )

        assert error_entry.is_error()

        normal_entry = LogEntry(
            timestamp=datetime.now(timezone.utc),
            level="INFO",
            command="test",
            status="complete",
            session_id="sess-123",
        )

        assert not normal_entry.is_error()


# =============================================================================
# TokenTracker Tests
# =============================================================================


class TestTokenTracker:
    """Tests for TokenTracker class."""

    def test_initial_state(self):
        """Test initial token tracker state."""
        tracker = TokenTracker()

        assert tracker.total_used == 0
        assert tracker.error_count == 0
        assert tracker.get_percentage() == 0.0

    def test_update_from_entry(self, sample_log_entries):
        """Test updating tracker from log entry."""
        tracker = TokenTracker()
        entry = LogEntry.from_json_line(json.dumps(sample_log_entries[1]))

        tracker.update(entry)

        assert tracker.total_used == 523
        assert tracker.command_counts["orchestrate"] == 1

    def test_accumulate_tokens(self, sample_log_entries):
        """Test token accumulation across entries."""
        tracker = TokenTracker()

        for entry_data in sample_log_entries:
            entry = LogEntry.from_json_line(json.dumps(entry_data))
            if entry:
                tracker.update(entry)

        # Should accumulate tokens from complete events
        assert tracker.total_used == 523 + 45  # orchestrate + memory query

    def test_get_percentage(self):
        """Test percentage calculation."""
        tracker = TokenTracker(budget=256_000)
        tracker.total_used = 25600

        percentage = tracker.get_percentage()

        assert percentage == 10.0

    def test_error_counting(self):
        """Test error counting."""
        tracker = TokenTracker()

        error_entry = LogEntry(
            timestamp=datetime.now(timezone.utc),
            level="ERROR",
            command="test",
            status="error",
            session_id="sess-123",
        )

        tracker.update(error_entry)

        assert tracker.error_count == 1

    def test_reset(self):
        """Test tracker reset."""
        tracker = TokenTracker()
        tracker.total_used = 1000
        tracker.error_count = 5

        tracker.reset()

        assert tracker.total_used == 0
        assert tracker.error_count == 0

    def test_format_summary(self):
        """Test summary formatting."""
        tracker = TokenTracker(budget=256_000)
        tracker.total_used = 1000

        summary = tracker.format_summary()

        assert "1,000" in summary
        assert "256,000" in summary


# =============================================================================
# MonitorDisplay Tests
# =============================================================================


class TestMonitorDisplay:
    """Tests for MonitorDisplay class."""

    def test_print_header(self, capsys):
        """Test header printing."""
        display = MonitorDisplay()

        # Capture output
        display.print_header(session_id="test-session")

        captured = capsys.readouterr()
        assert "CLI Monitor" in captured.out
        assert "test-session" in captured.out

    def test_print_table_header(self, capsys):
        """Test table header printing."""
        display = MonitorDisplay()

        display.print_table_header()

        captured = capsys.readouterr()
        assert "Time" in captured.out
        assert "Command" in captured.out
        assert "Status" in captured.out

    def test_print_entry(self, capsys, sample_log_entries):
        """Test entry printing."""
        display = MonitorDisplay()
        entry = LogEntry.from_json_line(json.dumps(sample_log_entries[1]))

        display.print_entry(entry)

        captured = capsys.readouterr()
        assert "orchestrate" in captured.out
        assert "OK" in captured.out or "complete" in captured.out

    def test_print_separator(self, capsys):
        """Test separator printing."""
        display = MonitorDisplay()

        display.print_separator()

        captured = capsys.readouterr()
        assert "-" in captured.out


# =============================================================================
# CliMonitor Tests
# =============================================================================


class TestCliMonitor:
    """Tests for CliMonitor class."""

    def test_load_initial_entries(self, populated_log_file, sample_log_entries):
        """Test loading initial entries from file."""
        monitor = CliMonitor(
            log_file=populated_log_file,
            initial_lines=20,
            follow=False,
        )

        monitor._load_initial_entries()

        # Should have loaded entries
        assert len(monitor.entries) == len(sample_log_entries)

    def test_token_tracking(self, populated_log_file):
        """Test token tracking during load."""
        monitor = CliMonitor(
            log_file=populated_log_file,
            initial_lines=20,
            follow=False,
        )

        monitor._load_initial_entries()

        # Should have accumulated tokens
        assert monitor.tracker.total_used == 523 + 45

    def test_command_filter(self, populated_log_file):
        """Test command filtering."""
        monitor = CliMonitor(
            log_file=populated_log_file,
            initial_lines=20,
            command_filter="orchestrate",
            follow=False,
        )

        monitor._load_initial_entries()

        # Should only have orchestrate entries
        for entry in monitor.entries:
            assert entry.command == "orchestrate"

    def test_session_detection(self, populated_log_file):
        """Test session ID detection."""
        monitor = CliMonitor(
            log_file=populated_log_file,
            initial_lines=20,
            follow=False,
        )

        monitor._load_initial_entries()

        assert monitor.current_session == "sess-abc123"

    def test_process_new_lines(self, temp_log_file, sample_log_entries):
        """Test processing new lines added to file."""
        # Write initial entries
        with open(temp_log_file, "w") as f:
            f.write(json.dumps(sample_log_entries[0]) + "\n")

        monitor = CliMonitor(
            log_file=temp_log_file,
            initial_lines=20,
            follow=False,
        )

        monitor._load_initial_entries()
        initial_count = len(monitor.entries)

        # Add more entries
        with open(temp_log_file, "a") as f:
            f.write(json.dumps(sample_log_entries[1]) + "\n")

        new_entries = monitor._process_new_lines()

        # Should detect new entry
        assert len(new_entries) == 1
        assert new_entries[0].status == "complete"

    def test_file_not_exists(self, tmp_path):
        """Test handling missing log file."""
        missing_file = tmp_path / "missing.log"

        monitor = CliMonitor(
            log_file=missing_file,
            initial_lines=20,
            follow=False,
        )

        # Should not raise
        monitor._load_initial_entries()

        assert len(monitor.entries) == 0

    def test_file_truncation_handling(self, temp_log_file, sample_log_entries):
        """Test handling file truncation."""
        # Write entries
        with open(temp_log_file, "w") as f:
            for entry in sample_log_entries:
                f.write(json.dumps(entry) + "\n")

        monitor = CliMonitor(
            log_file=temp_log_file,
            initial_lines=20,
            follow=False,
        )

        monitor._load_initial_entries()
        assert monitor.tracker.total_used > 0

        # Truncate file
        with open(temp_log_file, "w") as f:
            f.write("")

        monitor._process_new_lines()

        # Should detect truncation and reset
        assert monitor._last_position == 0


# =============================================================================
# Color Tests
# =============================================================================


class TestColors:
    """Tests for color formatting."""

    def test_colorize_with_tty(self, monkeypatch):
        """Test colorize with TTY."""
        monkeypatch.setattr(sys.stdout, "isatty", lambda: True)
        monkeypatch.delenv("NO_COLOR", raising=False)

        result = colorize("test", Colors.GREEN)

        assert Colors.GREEN in result
        assert "test" in result
        assert Colors.RESET in result

    def test_colorize_without_tty(self, monkeypatch):
        """Test colorize without TTY."""
        monkeypatch.setattr(sys.stdout, "isatty", lambda: False)

        result = colorize("test", Colors.GREEN)

        # Should return plain text
        assert result == "test"

    def test_colorize_with_no_color_env(self, monkeypatch):
        """Test colorize with NO_COLOR environment variable."""
        monkeypatch.setattr(sys.stdout, "isatty", lambda: True)
        monkeypatch.setenv("NO_COLOR", "1")

        result = colorize("test", Colors.GREEN)

        # Should return plain text
        assert result == "test"


# =============================================================================
# End-to-End Tests
# =============================================================================


class TestEndToEnd:
    """End-to-end integration tests."""

    def test_full_monitoring_flow(self, populated_log_file, capsys):
        """Test complete monitoring flow without follow mode."""
        monitor = CliMonitor(
            log_file=populated_log_file,
            initial_lines=20,
            follow=False,
        )

        monitor.run()

        captured = capsys.readouterr()

        # Should show header
        assert "CLI Monitor" in captured.out

        # Should show entries
        assert "orchestrate" in captured.out

        # Should show token summary
        assert "Running Total" in captured.out

    def test_filtered_monitoring(self, populated_log_file, capsys):
        """Test monitoring with filter."""
        monitor = CliMonitor(
            log_file=populated_log_file,
            initial_lines=20,
            command_filter="memory query",
            follow=False,
        )

        monitor.run()

        captured = capsys.readouterr()

        # Should only show filtered command
        assert "memory query" in captured.out

        # Should not show filtered out commands
        # (orchestrate entries should be filtered)

    def test_empty_file_handling(self, temp_log_file, capsys):
        """Test handling empty log file."""
        monitor = CliMonitor(
            log_file=temp_log_file,
            initial_lines=20,
            follow=False,
        )

        monitor.run()

        captured = capsys.readouterr()

        # Should show header
        assert "CLI Monitor" in captured.out

        # Should show zero tokens
        assert "0" in captured.out


# =============================================================================
# Schema Integration Tests
# =============================================================================


class TestSchemaIntegration:
    """Tests for schema integration with monitor."""

    def test_pydantic_schema_compatibility(self, sample_log_entries):
        """Test that log entries are compatible with Pydantic schemas."""
        from sigil.interfaces.cli.schemas import CliLogEntry as PydanticEntry

        for entry_data in sample_log_entries:
            # Should be able to create Pydantic model
            pydantic_entry = PydanticEntry(**entry_data)

            assert pydantic_entry.command == entry_data["command"]
            assert pydantic_entry.status == entry_data["status"]

    def test_monitor_entry_from_log_entry(self, sample_log_entries):
        """Test MonitorEntry creation from CliLogEntry."""
        from sigil.interfaces.cli.schemas import CliLogEntry as PydanticEntry, MonitorEntry

        pydantic_entry = PydanticEntry(**sample_log_entries[1])
        monitor_entry = MonitorEntry.from_log_entry(pydantic_entry)

        assert monitor_entry.command == "orchestrate"
        assert monitor_entry.status == "OK"
        assert "523" in (monitor_entry.tokens_used or "")

    def test_log_filter(self, sample_log_entries):
        """Test LogFilter schema."""
        from sigil.interfaces.cli.schemas import CliLogEntry as PydanticEntry, LogFilter

        filter_obj = LogFilter(command="orchestrate")

        # Test matching
        orchestrate_entry = PydanticEntry(**sample_log_entries[0])
        memory_entry = PydanticEntry(**sample_log_entries[2])

        assert filter_obj.matches(orchestrate_entry)
        assert not filter_obj.matches(memory_entry)
