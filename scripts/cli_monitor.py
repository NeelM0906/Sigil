#!/usr/bin/env python3
"""Sigil v2 CLI Execution Monitor - JSON Lines Log Viewer.

This script provides real-time monitoring of CLI execution logs in JSON Lines
format. It parses structured log entries and displays them in a formatted
table with color-coded status and running token totals.

Features:
    - Tails outputs/cli-execution.log in real-time (100ms updates)
    - Parses JSON Lines format for structured display
    - Shows running token totals with budget percentage
    - Color-coded status: green (complete), yellow (running), red (error)
    - Budget percentage colors: green (<50%), yellow (50-80%), red (>80%)
    - Filter by command name
    - Shows last 20 log entries (configurable)

Usage:
    python scripts/cli_monitor.py                    # Start monitoring
    python scripts/cli_monitor.py --follow           # Follow mode (default)
    python scripts/cli_monitor.py --filter orchestrate  # Show only orchestrate commands
    python scripts/cli_monitor.py --lines 50         # Show last 50 lines initially

Example Output:
    === CLI Monitor (Session: sess-abc123) ===
    Time       Command        Status   Tokens Used  Total Budget   Duration
    ---------------------------------------------------------------------------
    10:30:45   orchestrate    START    -            -              -
    10:30:48   orchestrate    OK       523 tokens   256,000 (0.20%) 3.3s
    10:30:50   /memory query  START    -            -              -
    10:30:51   /memory query  OK       45 tokens    256,000 (0.20%) 1.2s
    ---------------------------------------------------------------------------
    Running Total: 568 / 256,000 tokens (0.22%)
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# =============================================================================
# Constants
# =============================================================================

# Default log file path (JSON Lines format)
CLI_LOG_FILE = PROJECT_ROOT / "outputs" / "cli-execution.log"

# Total token budget (256K context window)
TOTAL_TOKEN_BUDGET = 256_000

# Poll interval in seconds
POLL_INTERVAL = 0.1  # 100ms

# Default number of entries to display
DEFAULT_DISPLAY_LINES = 20

# Column widths for table display
COLUMN_WIDTHS = {
    "time": 10,
    "command": 18,
    "status": 8,
    "tokens": 14,
    "budget": 18,
    "duration": 8,
}


# =============================================================================
# ANSI Colors
# =============================================================================


class Colors:
    """ANSI color codes for terminal output."""

    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"

    # Standard colors
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"

    # Bright colors
    BRIGHT_RED = "\033[91m"
    BRIGHT_GREEN = "\033[92m"
    BRIGHT_YELLOW = "\033[93m"
    BRIGHT_CYAN = "\033[96m"

    @classmethod
    def enabled(cls) -> bool:
        """Check if colors should be enabled."""
        return sys.stdout.isatty() and not os.environ.get("NO_COLOR")


def colorize(text: str, *colors: str) -> str:
    """Apply ANSI colors to text.

    Args:
        text: Text to colorize.
        *colors: Color codes to apply.

    Returns:
        Colorized text if TTY, plain text otherwise.
    """
    if not Colors.enabled():
        return text
    return "".join(colors) + text + Colors.RESET


# =============================================================================
# Log Entry Data Class
# =============================================================================


@dataclass
class LogEntry:
    """Parsed log entry from JSON Lines format.

    Attributes:
        timestamp: When the event occurred.
        level: Log level (INFO, ERROR, etc.).
        command: CLI command name.
        status: Command status (start, complete, error, timeout).
        session_id: Session identifier.
        user_input: User's original input (start events).
        tokens_used: Tokens consumed (complete events).
        tokens_remaining: Budget remaining.
        percentage: Budget percentage used.
        duration_ms: Execution time.
        confidence: Confidence score.
        error: Error message (error events).
        error_type: Exception type (error events).
    """

    timestamp: datetime
    level: str
    command: str
    status: str
    session_id: str
    user_input: Optional[str] = None
    tokens_used: Optional[int] = None
    tokens_remaining: Optional[int] = None
    percentage: Optional[float] = None
    duration_ms: Optional[int] = None
    confidence: Optional[float] = None
    error: Optional[str] = None
    error_type: Optional[str] = None

    @classmethod
    def from_json_line(cls, line: str) -> Optional["LogEntry"]:
        """Parse a JSON line into LogEntry.

        Args:
            line: Single JSON line.

        Returns:
            LogEntry if parsing succeeds, None otherwise.
        """
        try:
            data = json.loads(line)

            # Parse timestamp
            ts_str = data.get("timestamp", "")
            if ts_str:
                # Handle Z suffix
                ts_str = ts_str.replace("Z", "+00:00")
                timestamp = datetime.fromisoformat(ts_str)
            else:
                timestamp = datetime.now(timezone.utc)

            return cls(
                timestamp=timestamp,
                level=data.get("level", "INFO"),
                command=data.get("command", "unknown"),
                status=data.get("status", "unknown"),
                session_id=data.get("session_id", "unknown"),
                user_input=data.get("user_input"),
                tokens_used=data.get("tokens_used"),
                tokens_remaining=data.get("tokens_remaining"),
                percentage=data.get("percentage"),
                duration_ms=data.get("duration_ms"),
                confidence=data.get("confidence"),
                error=data.get("error"),
                error_type=data.get("error_type"),
            )
        except (json.JSONDecodeError, KeyError, ValueError):
            return None

    def format_time(self) -> str:
        """Format timestamp as HH:MM:SS."""
        return self.timestamp.strftime("%H:%M:%S")

    def format_status(self) -> str:
        """Format status with color."""
        status_map = {
            "start": ("START", Colors.YELLOW),
            "complete": ("OK", Colors.GREEN),
            "error": ("ERROR", Colors.RED, Colors.BOLD),
            "timeout": ("TIMEOUT", Colors.RED),
        }
        status_text, *colors = status_map.get(self.status, (self.status.upper(), Colors.WHITE))
        return colorize(status_text, *colors)

    def format_tokens(self) -> str:
        """Format tokens used."""
        if self.tokens_used is None:
            return colorize("-", Colors.DIM)
        return f"{self.tokens_used} tokens"

    def format_budget(self) -> str:
        """Format total budget with percentage."""
        if self.percentage is None:
            return colorize("-", Colors.DIM)

        # Color based on percentage
        if self.percentage < 50:
            color = Colors.GREEN
        elif self.percentage < 80:
            color = Colors.YELLOW
        else:
            color = Colors.RED

        return colorize(f"{TOTAL_TOKEN_BUDGET:,} ({self.percentage:.2f}%)", color)

    def format_duration(self) -> str:
        """Format duration."""
        if self.duration_ms is None:
            return colorize("-", Colors.DIM)

        if self.duration_ms >= 1000:
            return f"{self.duration_ms / 1000:.1f}s"
        return f"{self.duration_ms}ms"

    def is_error(self) -> bool:
        """Check if this is an error entry."""
        return self.status in ("error", "timeout")


# =============================================================================
# Token Tracker
# =============================================================================


class TokenTracker:
    """Tracks running token totals from log entries.

    Maintains cumulative token usage across all commands and provides
    formatted output for the status bar.
    """

    def __init__(self, budget: int = TOTAL_TOKEN_BUDGET):
        self.budget = budget
        self.total_used = 0
        self.command_counts: dict[str, int] = {}
        self.error_count = 0

    def update(self, entry: LogEntry) -> None:
        """Update counters from a log entry.

        Args:
            entry: LogEntry to process.
        """
        if entry.status == "complete" and entry.tokens_used is not None:
            self.total_used += entry.tokens_used
            self.command_counts[entry.command] = self.command_counts.get(entry.command, 0) + 1
        elif entry.is_error():
            self.error_count += 1

    def get_percentage(self) -> float:
        """Get percentage of budget used."""
        if self.budget <= 0:
            return 0.0
        return (self.total_used / self.budget) * 100

    def format_summary(self) -> str:
        """Get formatted summary line."""
        percentage = self.get_percentage()

        # Color based on percentage
        if percentage < 50:
            color = Colors.GREEN
        elif percentage < 80:
            color = Colors.YELLOW
        else:
            color = Colors.RED

        total_str = colorize(f"Running Total: {self.total_used:,} / {self.budget:,} tokens", Colors.BOLD)
        pct_str = colorize(f"({percentage:.2f}%)", color)

        parts = [total_str, pct_str]

        if self.error_count > 0:
            parts.append(colorize(f"| Errors: {self.error_count}", Colors.RED))

        return " ".join(parts)

    def reset(self) -> None:
        """Reset all counters."""
        self.total_used = 0
        self.command_counts.clear()
        self.error_count = 0


# =============================================================================
# Monitor Display
# =============================================================================


class MonitorDisplay:
    """Handles formatting and display of monitor output.

    Provides methods for rendering the table header, rows, and status bar
    in a consistent format.
    """

    def __init__(self, widths: dict[str, int] = None):
        self.widths = widths or COLUMN_WIDTHS

    def print_header(self, session_id: str = "unknown") -> None:
        """Print the monitor header."""
        print()
        print(colorize("=" * 75, Colors.CYAN))
        print(colorize(f"  CLI Monitor (Session: {session_id})", Colors.BOLD, Colors.CYAN))
        print(colorize("=" * 75, Colors.CYAN))
        print()
        print(colorize("Press Ctrl+C to stop", Colors.DIM))
        print()

    def print_table_header(self) -> None:
        """Print the table column headers."""
        headers = [
            "Time".ljust(self.widths["time"]),
            "Command".ljust(self.widths["command"]),
            "Status".ljust(self.widths["status"]),
            "Tokens Used".ljust(self.widths["tokens"]),
            "Total Budget".ljust(self.widths["budget"]),
            "Duration".ljust(self.widths["duration"]),
        ]
        header_line = " ".join(headers)
        print(colorize(header_line, Colors.BOLD))
        print(colorize("-" * 75, Colors.DIM))

    def print_entry(self, entry: LogEntry) -> None:
        """Print a single log entry as a table row."""
        row = [
            entry.format_time().ljust(self.widths["time"]),
            entry.command[:self.widths["command"]].ljust(self.widths["command"]),
            entry.format_status().ljust(self.widths["status"] + 10),  # Extra for color codes
            entry.format_tokens().ljust(self.widths["tokens"]),
            entry.format_budget().ljust(self.widths["budget"] + 10),  # Extra for color codes
            entry.format_duration().ljust(self.widths["duration"]),
        ]

        # Add error info for error entries
        if entry.is_error() and entry.error:
            row_str = " ".join(row)
            error_info = colorize(f" [{entry.error_type}: {entry.error[:40]}...]", Colors.RED, Colors.DIM)
            print(row_str + error_info)
        else:
            print(" ".join(row))

    def print_separator(self) -> None:
        """Print a separator line."""
        print(colorize("-" * 75, Colors.DIM))

    def print_summary(self, tracker: TokenTracker) -> None:
        """Print the running total summary."""
        self.print_separator()
        print(tracker.format_summary())

    def clear_line(self) -> None:
        """Clear the current line (for status updates)."""
        sys.stdout.write(f"\r{' ' * 80}\r")
        sys.stdout.flush()


# =============================================================================
# CLI Monitor
# =============================================================================


class CliMonitor:
    """Real-time CLI execution log monitor.

    Watches the JSON Lines log file and displays entries in a formatted
    table with running token totals.
    """

    def __init__(
        self,
        log_file: Path,
        initial_lines: int = DEFAULT_DISPLAY_LINES,
        command_filter: Optional[str] = None,
        follow: bool = True,
    ):
        """Initialize the monitor.

        Args:
            log_file: Path to the JSON Lines log file.
            initial_lines: Number of lines to show initially.
            command_filter: Filter to specific command (e.g., "orchestrate").
            follow: Whether to follow the file for updates.
        """
        self.log_file = log_file
        self.initial_lines = initial_lines
        self.command_filter = command_filter
        self.follow = follow

        self.tracker = TokenTracker()
        self.display = MonitorDisplay()
        self.entries: list[LogEntry] = []
        self.current_session: Optional[str] = None

        self._running = True
        self._last_position = 0

        # Set up signal handlers
        signal.signal(signal.SIGINT, self._handle_interrupt)
        signal.signal(signal.SIGTERM, self._handle_interrupt)

    def _handle_interrupt(self, signum: int, frame) -> None:
        """Handle interrupt signals."""
        self._running = False
        print()
        print(colorize("Monitor stopped.", Colors.DIM))
        sys.exit(0)

    def _load_initial_entries(self) -> None:
        """Load initial entries from the log file."""
        if not self.log_file.exists():
            print(colorize(f"Log file not found: {self.log_file}", Colors.YELLOW))
            print(colorize("Waiting for logs...", Colors.DIM))
            print()
            return

        try:
            with open(self.log_file, "r") as f:
                lines = f.readlines()
        except IOError as e:
            print(colorize(f"Error reading log file: {e}", Colors.RED))
            return

        # Parse all lines, keeping last N
        all_entries = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            entry = LogEntry.from_json_line(line)
            if entry:
                # Apply filter
                if self.command_filter and entry.command != self.command_filter:
                    continue
                all_entries.append(entry)
                self.tracker.update(entry)
                if entry.session_id != "unknown":
                    self.current_session = entry.session_id

        # Keep last N entries for display
        self.entries = all_entries[-self.initial_lines:] if all_entries else []

        # Update file position
        self._last_position = self.log_file.stat().st_size

    def _process_new_lines(self) -> list[LogEntry]:
        """Process new lines added to the log file.

        Returns:
            List of new entries parsed.
        """
        if not self.log_file.exists():
            return []

        try:
            current_size = self.log_file.stat().st_size
        except IOError:
            return []

        # File was truncated
        if current_size < self._last_position:
            self._last_position = 0
            self.tracker.reset()
            self.entries.clear()
            print()
            print(colorize("--- Log file reset ---", Colors.YELLOW))
            print()

        if current_size <= self._last_position:
            return []

        new_entries = []
        try:
            with open(self.log_file, "r") as f:
                f.seek(self._last_position)
                new_content = f.read()

            for line in new_content.splitlines():
                line = line.strip()
                if not line:
                    continue
                entry = LogEntry.from_json_line(line)
                if entry:
                    # Apply filter
                    if self.command_filter and entry.command != self.command_filter:
                        continue
                    new_entries.append(entry)
                    self.tracker.update(entry)
                    if entry.session_id != "unknown":
                        self.current_session = entry.session_id

            self._last_position = current_size

        except IOError:
            pass

        return new_entries

    def _display_entries(self, entries: list[LogEntry]) -> None:
        """Display a list of entries."""
        for entry in entries:
            self.display.print_entry(entry)

    def run(self) -> None:
        """Run the monitor."""
        # Print header
        self.display.print_header(self.current_session or "waiting...")

        # Load initial entries
        self._load_initial_entries()

        # Print table header and initial entries
        self.display.print_table_header()
        self._display_entries(self.entries)

        if not self.follow:
            self.display.print_summary(self.tracker)
            return

        # Follow mode - watch for new entries
        print()  # Space before status updates

        while self._running:
            try:
                new_entries = self._process_new_lines()

                if new_entries:
                    # Clear status line and print new entries
                    self.display.clear_line()
                    self._display_entries(new_entries)
                    self.entries.extend(new_entries)

                    # Trim entries list to max size
                    if len(self.entries) > self.initial_lines * 2:
                        self.entries = self.entries[-self.initial_lines:]

                # Print running summary
                summary = self.tracker.format_summary()
                sys.stdout.write(f"\r{summary}")
                sys.stdout.flush()

                time.sleep(POLL_INTERVAL)

            except Exception as e:
                print()
                print(colorize(f"Error: {e}", Colors.RED))
                time.sleep(1)


# =============================================================================
# Main Entry Point
# =============================================================================


def main() -> None:
    """Main entry point for the CLI monitor."""
    parser = argparse.ArgumentParser(
        description="Sigil v2 CLI Execution Monitor - JSON Lines Log Viewer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/cli_monitor.py                    Start monitoring (follow mode)
  python scripts/cli_monitor.py --no-follow        Show current entries and exit
  python scripts/cli_monitor.py --filter orchestrate  Show only orchestrate commands
  python scripts/cli_monitor.py --lines 50         Show last 50 lines initially

Typical usage:
  Terminal 1: python scripts/cli_monitor.py
  Terminal 2: sigil orchestrate "Analyze Acme Corp"
        """,
    )

    parser.add_argument(
        "--lines", "-n",
        type=int,
        default=DEFAULT_DISPLAY_LINES,
        help=f"Number of initial lines to show (default: {DEFAULT_DISPLAY_LINES})",
    )

    parser.add_argument(
        "--follow", "-f",
        action="store_true",
        default=True,
        help="Follow the log file for updates (default: True)",
    )

    parser.add_argument(
        "--no-follow",
        action="store_true",
        help="Don't follow, just show current entries and exit",
    )

    parser.add_argument(
        "--filter", "-F",
        type=str,
        default=None,
        dest="command_filter",
        help="Filter to show only specific command (e.g., 'orchestrate')",
    )

    parser.add_argument(
        "--log-file",
        type=Path,
        default=CLI_LOG_FILE,
        help=f"Path to log file (default: {CLI_LOG_FILE})",
    )

    parser.add_argument(
        "--clear", "-c",
        action="store_true",
        help="Clear the log file before starting",
    )

    args = parser.parse_args()

    # Handle --no-follow
    follow = args.follow and not args.no_follow

    # Ensure outputs directory exists
    args.log_file.parent.mkdir(parents=True, exist_ok=True)

    # Clear log file if requested
    if args.clear and args.log_file.exists():
        args.log_file.unlink()
        print(colorize(f"Cleared log file: {args.log_file}", Colors.YELLOW))

    # Create and run monitor
    monitor = CliMonitor(
        log_file=args.log_file,
        initial_lines=args.lines,
        command_filter=args.command_filter,
        follow=follow,
    )

    try:
        monitor.run()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
