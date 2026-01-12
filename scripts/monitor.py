#!/usr/bin/env python3
"""Sigil v2 Execution Monitor - Standalone Log Viewer.

This script launches a real-time log viewer in a separate process that
shows live execution logs and running token counters as they happen.

Features:
    - Tails outputs/sigil-execution.log in real-time
    - Shows live token counter that accumulates across operations
    - Color-coded output by log level and component
    - Can run in a separate terminal while CLI is executing
    - Non-blocking, can be killed without affecting main CLI

Usage:
    python scripts/monitor.py              # Start monitoring
    python scripts/monitor.py --clear      # Clear log file and start
    python scripts/monitor.py --lines 50   # Show last 50 lines initially

Example Setup:
    Terminal 1: python scripts/monitor.py
    Terminal 2: python -m sigil.interfaces.cli.app orchestrate --task "..."

The monitor will show logs in Terminal 1 as they are written by Terminal 2.
"""

from __future__ import annotations

import argparse
import os
import re
import signal
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# =============================================================================
# Constants
# =============================================================================

LOG_FILE = PROJECT_ROOT / "outputs" / "sigil-execution.log"
TOTAL_TOKEN_BUDGET = 256_000
POLL_INTERVAL = 0.1  # seconds


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
    BRIGHT_BLUE = "\033[94m"
    BRIGHT_MAGENTA = "\033[95m"
    BRIGHT_CYAN = "\033[96m"

    # Background colors
    BG_RED = "\033[41m"
    BG_GREEN = "\033[42m"
    BG_YELLOW = "\033[43m"
    BG_BLUE = "\033[44m"

    @classmethod
    def enabled(cls) -> bool:
        """Check if colors should be enabled."""
        return sys.stdout.isatty()


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
# Token Counter
# =============================================================================


class TokenCounter:
    """Running token counter for the monitoring display.

    Parses token counts from log lines and maintains running totals.
    """

    # Regex to extract token count from log line
    TOKEN_PATTERN = re.compile(r"\[(\d+) tokens\]")
    COMPONENT_PATTERN = re.compile(r"\[(ROUTING|MEMORY|PLANNING|REASONING|VALIDATION|COMPLETE|ORCHESTRATOR|SUMMARY)\]")

    def __init__(self):
        self.routing = 0
        self.memory = 0
        self.planning = 0
        self.reasoning = 0
        self.validation = 0
        self.total = 0
        self.session_count = 0
        self._last_component: Optional[str] = None

    def parse_line(self, line: str) -> None:
        """Parse a log line and update counters.

        Args:
            line: Log line to parse.
        """
        # Extract token count
        token_match = self.TOKEN_PATTERN.search(line)
        tokens = int(token_match.group(1)) if token_match else 0

        # Extract component
        component_match = self.COMPONENT_PATTERN.search(line)
        if not component_match:
            return

        component = component_match.group(1)
        self._last_component = component

        # Update appropriate counter
        if component == "ROUTING":
            self.routing += tokens
        elif component == "MEMORY":
            self.memory += tokens
        elif component == "PLANNING":
            self.planning += tokens
        elif component == "REASONING":
            self.reasoning += tokens
        elif component == "VALIDATION":
            self.validation += tokens
        elif component == "COMPLETE":
            self.session_count += 1

        # Update total (excluding COMPLETE which has cumulative total)
        if component not in ("COMPLETE", "SUMMARY"):
            self.total += tokens

    def get_status_bar(self) -> str:
        """Get formatted status bar string.

        Returns:
            Status bar showing running token counts.
        """
        percentage = (self.total / TOTAL_TOKEN_BUDGET) * 100 if TOTAL_TOKEN_BUDGET > 0 else 0

        parts = [
            colorize(f"TOKENS: {self.total:,}", Colors.BOLD, Colors.CYAN),
            f"/ {TOTAL_TOKEN_BUDGET:,}",
            colorize(f"({percentage:.2f}%)", Colors.YELLOW if percentage < 50 else Colors.RED if percentage > 80 else Colors.GREEN),
        ]

        # Add component breakdown
        breakdown = []
        if self.routing > 0:
            breakdown.append(f"R:{self.routing}")
        if self.memory > 0:
            breakdown.append(f"M:{self.memory}")
        if self.planning > 0:
            breakdown.append(f"P:{self.planning}")
        if self.reasoning > 0:
            breakdown.append(f"Re:{self.reasoning}")
        if self.validation > 0:
            breakdown.append(f"V:{self.validation}")

        if breakdown:
            parts.append(colorize(" | " + " ".join(breakdown), Colors.DIM))

        if self.session_count > 0:
            parts.append(colorize(f" | Sessions: {self.session_count}", Colors.DIM))

        return " ".join(parts)

    def reset(self) -> None:
        """Reset all counters."""
        self.routing = 0
        self.memory = 0
        self.planning = 0
        self.reasoning = 0
        self.validation = 0
        self.total = 0
        self.session_count = 0


# =============================================================================
# Log Line Formatting
# =============================================================================


def format_log_line(line: str) -> str:
    """Format a log line with colors based on level and component.

    Args:
        line: Raw log line.

    Returns:
        Formatted log line with colors.
    """
    if not line.strip():
        return ""

    # Color by log level
    if "[DEBUG]" in line:
        return colorize(line, Colors.DIM)
    elif "[WARNING]" in line:
        return colorize(line, Colors.YELLOW)
    elif "[ERROR]" in line:
        return colorize(line, Colors.RED, Colors.BOLD)
    elif "[CRITICAL]" in line:
        return colorize(line, Colors.BG_RED, Colors.WHITE, Colors.BOLD)

    # Color by component for INFO level
    if "[INFO]" in line:
        if "[ROUTING]" in line:
            line = line.replace("[ROUTING]", colorize("[ROUTING]", Colors.BLUE))
        elif "[MEMORY]" in line:
            line = line.replace("[MEMORY]", colorize("[MEMORY]", Colors.MAGENTA))
        elif "[PLANNING]" in line:
            line = line.replace("[PLANNING]", colorize("[PLANNING]", Colors.CYAN))
        elif "[REASONING]" in line:
            line = line.replace("[REASONING]", colorize("[REASONING]", Colors.GREEN))
        elif "[VALIDATION]" in line:
            line = line.replace("[VALIDATION]", colorize("[VALIDATION]", Colors.YELLOW))
        elif "[COMPLETE]" in line:
            line = line.replace("[COMPLETE]", colorize("[COMPLETE]", Colors.BRIGHT_GREEN, Colors.BOLD))
        elif "[ORCHESTRATOR]" in line:
            line = line.replace("[ORCHESTRATOR]", colorize("[ORCHESTRATOR]", Colors.BRIGHT_CYAN))

    return line


# =============================================================================
# Monitor Class
# =============================================================================


class LogMonitor:
    """Real-time log file monitor.

    Watches the log file for changes and displays new content with
    color formatting and a running token counter.
    """

    def __init__(
        self,
        log_file: Path,
        initial_lines: int = 20,
        show_status_bar: bool = True,
    ):
        """Initialize the monitor.

        Args:
            log_file: Path to the log file to monitor.
            initial_lines: Number of lines to show initially.
            show_status_bar: Whether to show the token status bar.
        """
        self.log_file = log_file
        self.initial_lines = initial_lines
        self.show_status_bar = show_status_bar
        self.token_counter = TokenCounter()
        self._running = True
        self._last_position = 0

        # Set up signal handler for graceful shutdown
        signal.signal(signal.SIGINT, self._handle_interrupt)
        signal.signal(signal.SIGTERM, self._handle_interrupt)

    def _handle_interrupt(self, signum: int, frame) -> None:
        """Handle interrupt signals."""
        self._running = False
        print()
        print(colorize("Monitor stopped.", Colors.DIM))
        sys.exit(0)

    def _print_header(self) -> None:
        """Print the monitor header."""
        print()
        print(colorize("=" * 70, Colors.CYAN))
        print(colorize("  SIGIL v2 EXECUTION MONITOR", Colors.BOLD, Colors.CYAN))
        print(colorize("=" * 70, Colors.CYAN))
        print()
        print(f"Watching: {colorize(str(self.log_file), Colors.DIM)}")
        print(colorize("Press Ctrl+C to stop", Colors.DIM))
        print()
        print(colorize("-" * 70, Colors.DIM))
        print()

    def _print_status_bar(self) -> None:
        """Print the token status bar (overwrites current line)."""
        if not self.show_status_bar:
            return

        status = self.token_counter.get_status_bar()
        # Move to start of line and clear
        sys.stdout.write(f"\r{' ' * 100}\r")
        sys.stdout.write(status)
        sys.stdout.flush()

    def _show_initial_lines(self) -> None:
        """Show initial lines from the log file."""
        if not self.log_file.exists():
            print(colorize("Log file does not exist yet. Waiting for logs...", Colors.YELLOW))
            print()
            return

        with open(self.log_file, "r") as f:
            lines = f.readlines()

        if not lines:
            print(colorize("Log file is empty. Waiting for logs...", Colors.YELLOW))
            print()
            return

        # Show last N lines
        start_idx = max(0, len(lines) - self.initial_lines)

        for line in lines[start_idx:]:
            line = line.rstrip()
            if line:
                # Parse for token counter
                self.token_counter.parse_line(line)
                # Format and print
                print(format_log_line(line))

        # Update last position
        self._last_position = self.log_file.stat().st_size
        print()

    def _process_new_content(self) -> None:
        """Process new content added to the log file."""
        if not self.log_file.exists():
            return

        current_size = self.log_file.stat().st_size

        if current_size < self._last_position:
            # File was truncated, reset
            self._last_position = 0
            self.token_counter.reset()
            print()
            print(colorize("--- Log file reset ---", Colors.YELLOW))
            print()

        if current_size > self._last_position:
            with open(self.log_file, "r") as f:
                f.seek(self._last_position)
                new_content = f.read()

            for line in new_content.splitlines():
                if line.strip():
                    # Parse for token counter
                    self.token_counter.parse_line(line)
                    # Clear status bar line before printing
                    if self.show_status_bar:
                        sys.stdout.write(f"\r{' ' * 100}\r")
                    # Format and print
                    print(format_log_line(line))

            self._last_position = current_size

    def run(self) -> None:
        """Run the monitor loop."""
        self._print_header()
        self._show_initial_lines()

        while self._running:
            try:
                self._process_new_content()
                self._print_status_bar()
                time.sleep(POLL_INTERVAL)
            except Exception as e:
                print()
                print(colorize(f"Error: {e}", Colors.RED))
                time.sleep(1)


# =============================================================================
# Main Entry Point
# =============================================================================


def main() -> None:
    """Main entry point for the monitor script."""
    parser = argparse.ArgumentParser(
        description="Sigil v2 Execution Monitor - Real-time log viewer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/monitor.py              Start monitoring
  python scripts/monitor.py --clear      Clear logs and start fresh
  python scripts/monitor.py --lines 50   Show last 50 lines initially
  python scripts/monitor.py --no-status  Disable token status bar

Typical usage:
  Terminal 1: python scripts/monitor.py
  Terminal 2: python -m sigil.interfaces.cli.app orchestrate --task "..."
        """,
    )

    parser.add_argument(
        "--lines", "-n",
        type=int,
        default=20,
        help="Number of initial lines to show (default: 20)",
    )

    parser.add_argument(
        "--clear", "-c",
        action="store_true",
        help="Clear the log file before starting",
    )

    parser.add_argument(
        "--no-status",
        action="store_true",
        help="Disable the token status bar",
    )

    parser.add_argument(
        "--log-file",
        type=Path,
        default=LOG_FILE,
        help=f"Path to log file (default: {LOG_FILE})",
    )

    args = parser.parse_args()

    # Ensure outputs directory exists
    args.log_file.parent.mkdir(parents=True, exist_ok=True)

    # Clear log file if requested
    if args.clear and args.log_file.exists():
        args.log_file.unlink()
        print(colorize(f"Cleared log file: {args.log_file}", Colors.YELLOW))

    # Create and run monitor
    monitor = LogMonitor(
        log_file=args.log_file,
        initial_lines=args.lines,
        show_status_bar=not args.no_status,
    )

    try:
        monitor.run()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
