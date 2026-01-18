"""Output Formatting for Sigil v2 CLI.

This module provides output formatting utilities for the Sigil CLI, including
colored terminal output, token usage meters, and result formatting.

Key Components:
    - ColoredOutput: Color-coded terminal output
    - TokenMeter: Visual token budget meter
    - ResultFormatter: Consistent result formatting

All formatting respects TTY detection and configuration settings for
color output.

Example:
    >>> from sigil.interfaces.cli.formatter import ColoredOutput, TokenMeter
    >>> output = ColoredOutput()
    >>> print(output.format_success("Operation completed"))
    >>> meter = TokenMeter(budget=256_000)
    >>> meter.add_tokens(1000)
    >>> print(meter.format_meter())
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import Any, List, Optional

from sigil.interfaces.cli.config import (
    ANSI_COLORS,
    SEMANTIC_COLORS,
    TOKEN_THRESHOLDS,
    DEFAULT_TOKEN_BUDGET,
    get_cli_config,
)


# =============================================================================
# Colorize Function (Module-level)
# =============================================================================


def colorize(text: str, color: str, force: bool = False) -> str:
    """Apply ANSI color to text.

    Checks if stdout is a TTY and respects configuration settings
    for color output.

    Args:
        text: Text to colorize.
        color: Color name (e.g., 'red', 'green', 'success').
        force: If True, apply color even if not a TTY.

    Returns:
        Colorized text string (or plain text if colors disabled).

    Example:
        >>> colorize("Error!", "red")
        '\\033[31mError!\\033[0m'  # If TTY
        >>> colorize("Error!", "red")
        'Error!'  # If not TTY
    """
    # Check if we should apply colors
    config = get_cli_config()
    if not force and (not sys.stdout.isatty() or not config.color):
        return text

    # Resolve semantic colors
    actual_color = SEMANTIC_COLORS.get(color, color)

    # Get the ANSI code
    color_code = ANSI_COLORS.get(actual_color, "")
    reset_code = ANSI_COLORS.get("reset", "")

    if not color_code:
        return text

    return f"{color_code}{text}{reset_code}"


# =============================================================================
# ColoredOutput Class
# =============================================================================


class ColoredOutput:
    """Color-coded terminal output helper.

    Provides methods for formatting text with semantic colors and
    creating consistent output styles throughout the CLI.

    Attributes:
        enabled: Whether color output is enabled.

    Example:
        >>> output = ColoredOutput()
        >>> print(output.format_success("Done!"))
        >>> print(output.format_error("Failed!"))
        >>> print(output.format_header("RESULTS"))
    """

    def __init__(self, enabled: Optional[bool] = None) -> None:
        """Initialize colored output.

        Args:
            enabled: Override for color enabled setting. If None,
                uses configuration and TTY detection.
        """
        if enabled is None:
            config = get_cli_config()
            self.enabled = config.color and sys.stdout.isatty()
        else:
            self.enabled = enabled

    def colorize(self, text: str, color: str) -> str:
        """Apply color to text.

        Args:
            text: Text to colorize.
            color: Color name (raw or semantic).

        Returns:
            Colorized text if enabled, otherwise plain text.
        """
        if not self.enabled:
            return text

        # Resolve semantic color
        actual_color = SEMANTIC_COLORS.get(color, color)
        color_code = ANSI_COLORS.get(actual_color, "")
        reset_code = ANSI_COLORS.get("reset", "")

        if not color_code:
            return text

        return f"{color_code}{text}{reset_code}"

    def format_success(self, message: str) -> str:
        """Format a success message.

        Args:
            message: Success message text.

        Returns:
            Green-colored success message with checkmark.
        """
        return self.colorize(f"[OK] {message}", "success")

    def format_error(self, message: str) -> str:
        """Format an error message.

        Args:
            message: Error message text.

        Returns:
            Red-colored error message with X marker.
        """
        return self.colorize(f"[ERROR] {message}", "error")

    def format_warning(self, message: str) -> str:
        """Format a warning message.

        Args:
            message: Warning message text.

        Returns:
            Yellow-colored warning message.
        """
        return self.colorize(f"[WARNING] {message}", "warning")

    def format_info(self, message: str) -> str:
        """Format an info message.

        Args:
            message: Info message text.

        Returns:
            Cyan-colored info message.
        """
        return self.colorize(f"[INFO] {message}", "info")

    def format_header(self, title: str, width: int = 70) -> str:
        """Format a section header.

        Creates a visually distinct header with optional width.

        Args:
            title: Header title text.
            width: Total width of the header line.

        Returns:
            Formatted header string.

        Example:
            >>> output.format_header("RESULTS")
            '====== RESULTS ======'
        """
        padding = (width - len(title) - 2) // 2
        line = "=" * padding
        header = f"{line} {title} {line}"
        # Adjust for odd widths
        if len(header) < width:
            header += "="
        return self.colorize(header, "bold")

    def format_section(self, title: str, content: str) -> str:
        """Format a section with title and content.

        Args:
            title: Section title.
            content: Section content (can be multi-line).

        Returns:
            Formatted section string.
        """
        title_line = self.colorize(f"{title}:", "bold")
        return f"{title_line}\n{content}"

    def format_key_value(self, key: str, value: Any, indent: int = 2) -> str:
        """Format a key-value pair.

        Args:
            key: The key/label.
            value: The value to display.
            indent: Number of spaces to indent.

        Returns:
            Formatted key-value string.
        """
        prefix = " " * indent
        key_colored = self.colorize(f"{key}:", "bold")
        return f"{prefix}{key_colored} {value}"

    def format_list(self, items: List[str], indent: int = 2) -> str:
        """Format a list of items.

        Args:
            items: List of items to format.
            indent: Number of spaces to indent.

        Returns:
            Formatted list as multi-line string.
        """
        prefix = " " * indent
        lines = [f"{prefix}- {item}" for item in items]
        return "\n".join(lines)

    def dim(self, text: str) -> str:
        """Apply dim formatting to text.

        Args:
            text: Text to dim.

        Returns:
            Dimmed text.
        """
        return self.colorize(text, "dim")

    def bold(self, text: str) -> str:
        """Apply bold formatting to text.

        Args:
            text: Text to bold.

        Returns:
            Bold text.
        """
        return self.colorize(text, "bold")


# =============================================================================
# TokenMeter Class
# =============================================================================


@dataclass
class TokenMeter:
    """Visual token budget meter.

    Tracks token usage and provides visual progress bar representation
    for CLI output.

    Attributes:
        budget: Total token budget.
        used: Tokens consumed.

    Example:
        >>> meter = TokenMeter(budget=256_000)
        >>> meter.add_tokens(50_000)
        >>> print(meter.format_meter())
        [========----------] 50,000 / 256,000 (19.53%)
    """

    budget: int = DEFAULT_TOKEN_BUDGET
    used: int = field(default=0, init=False)

    def add_tokens(self, amount: int) -> None:
        """Add tokens to usage.

        Args:
            amount: Number of tokens to add.
        """
        self.used += amount

    def get_usage(self) -> int:
        """Get total tokens used.

        Returns:
            Number of tokens consumed.
        """
        return self.used

    def get_remaining(self) -> int:
        """Get remaining tokens.

        Returns:
            Number of tokens remaining in budget.
        """
        return max(0, self.budget - self.used)

    def get_percentage(self) -> float:
        """Get percentage of budget used.

        Returns:
            Percentage consumed (0.0 to 100.0).
        """
        if self.budget == 0:
            return 100.0
        return (self.used / self.budget) * 100.0

    def is_critical(self) -> bool:
        """Check if critical threshold exceeded.

        Returns:
            True if usage is at or above critical threshold.
        """
        return self.get_percentage() >= TOKEN_THRESHOLDS["critical"]

    def is_warning(self) -> bool:
        """Check if warning threshold exceeded.

        Returns:
            True if usage is at or above warning (high) threshold.
        """
        return self.get_percentage() >= TOKEN_THRESHOLDS["high"]

    def get_status(self) -> str:
        """Get current budget status.

        Returns:
            Status string: 'ok', 'warning', or 'critical'.
        """
        percentage = self.get_percentage()
        if percentage >= TOKEN_THRESHOLDS["critical"]:
            return "critical"
        elif percentage >= TOKEN_THRESHOLDS["high"]:
            return "warning"
        return "ok"

    def format_meter(self, width: int = 20) -> str:
        """Return visual progress meter.

        Creates an ASCII progress bar with usage statistics.

        Args:
            width: Width of the progress bar in characters.

        Returns:
            Formatted meter string.

        Example:
            >>> meter = TokenMeter(budget=100_000)
            >>> meter.add_tokens(25_000)
            >>> meter.format_meter()
            '[=====---------------] 25,000 / 100,000 (25.00%)'
        """
        percentage = self.get_percentage()
        filled = int((percentage / 100) * width)
        filled = min(filled, width)  # Cap at width
        empty = width - filled

        # Choose bar character based on status
        if self.is_critical():
            fill_char = "!"
        elif self.is_warning():
            fill_char = "#"
        else:
            fill_char = "="

        bar = f"[{fill_char * filled}{'-' * empty}]"
        stats = f"{self.used:,} / {self.budget:,} ({percentage:.2f}%)"

        return f"{bar} {stats}"

    def format_compact(self) -> str:
        """Format compact meter for inline display.

        Returns:
            Compact meter string.
        """
        percentage = self.get_percentage()
        status = self.get_status().upper()
        return f"Tokens: {self.used:,}/{self.budget:,} ({percentage:.1f}%) [{status}]"

    def reset(self) -> None:
        """Reset usage counter to zero."""
        self.used = 0


# =============================================================================
# ResultFormatter Class
# =============================================================================


class ResultFormatter:
    """Consistent result formatting for CLI output.

    Provides methods for formatting various types of results from
    the Sigil pipeline in a consistent, readable format.

    Example:
        >>> formatter = ResultFormatter()
        >>> print(formatter.format_result({"answer": "42", "confidence": 0.95}))
        >>> print(formatter.format_error_result("Connection failed", "network"))
    """

    def __init__(self, colored: Optional[ColoredOutput] = None) -> None:
        """Initialize result formatter.

        Args:
            colored: ColoredOutput instance. Creates default if None.
        """
        self.colored = colored or ColoredOutput()

    def format_result(self, result: Any) -> str:
        """Format any result type.

        Intelligently formats different result types:
        - dict: Key-value pairs
        - list: Bulleted list
        - str: Direct output
        - Other: String conversion

        Args:
            result: Result to format.

        Returns:
            Formatted result string.
        """
        if result is None:
            return self.colored.dim("(no result)")

        if isinstance(result, dict):
            return self._format_dict_result(result)
        elif isinstance(result, list):
            return self._format_list_result(result)
        elif isinstance(result, str):
            return result
        else:
            return str(result)

    def _format_dict_result(self, result: dict[str, Any]) -> str:
        """Format dictionary result.

        Args:
            result: Dictionary to format.

        Returns:
            Formatted dictionary string.
        """
        lines = []
        for key, value in result.items():
            # Skip internal keys
            if key.startswith("_"):
                continue

            # Format value based on type
            if isinstance(value, dict):
                formatted_value = self._format_nested_dict(value, indent=4)
                lines.append(f"  {self.colored.bold(key)}:")
                lines.append(formatted_value)
            elif isinstance(value, list):
                lines.append(f"  {self.colored.bold(key)}:")
                for item in value[:5]:  # Limit to 5 items
                    lines.append(f"    - {item}")
                if len(value) > 5:
                    lines.append(f"    ... and {len(value) - 5} more")
            else:
                lines.append(f"  {self.colored.bold(key)}: {value}")

        return "\n".join(lines)

    def _format_nested_dict(self, d: dict[str, Any], indent: int = 2) -> str:
        """Format nested dictionary.

        Args:
            d: Dictionary to format.
            indent: Indentation level.

        Returns:
            Formatted nested dictionary string.
        """
        prefix = " " * indent
        lines = []
        for key, value in d.items():
            if isinstance(value, (dict, list)):
                lines.append(f"{prefix}{key}: {type(value).__name__}")
            else:
                lines.append(f"{prefix}{key}: {value}")
        return "\n".join(lines)

    def _format_list_result(self, result: list) -> str:
        """Format list result.

        Args:
            result: List to format.

        Returns:
            Formatted list string.
        """
        if not result:
            return self.colored.dim("(empty list)")

        lines = []
        for i, item in enumerate(result[:10], 1):  # Limit to 10 items
            if isinstance(item, dict):
                # Format dict items compactly
                summary = ", ".join(f"{k}={v}" for k, v in list(item.items())[:3])
                lines.append(f"  {i}. {summary}")
            else:
                lines.append(f"  {i}. {item}")

        if len(result) > 10:
            lines.append(f"  ... and {len(result) - 10} more items")

        return "\n".join(lines)

    def format_step_result(
        self,
        step_id: str,
        status: str,
        output: str,
    ) -> str:
        """Format step result.

        Args:
            step_id: Step identifier.
            status: Step status (completed, failed, skipped).
            output: Step output text.

        Returns:
            Formatted step result string.
        """
        # Choose status color
        status_colors = {
            "completed": "success",
            "success": "success",
            "failed": "error",
            "error": "error",
            "skipped": "warning",
            "pending": "dim",
        }
        color = status_colors.get(status.lower(), "dim")

        status_formatted = self.colored.colorize(f"[{status.upper()}]", color)
        step_formatted = self.colored.bold(step_id)

        lines = [
            f"{status_formatted} {step_formatted}",
        ]

        if output:
            # Indent and truncate output
            truncated = output[:200] + "..." if len(output) > 200 else output
            for line in truncated.split("\n"):
                lines.append(f"    {line}")

        return "\n".join(lines)

    def format_plan_result(
        self,
        plan_id: str,
        steps: List[dict],
        output: str,
    ) -> str:
        """Format plan result.

        Args:
            plan_id: Plan identifier.
            steps: List of step dictionaries with status.
            output: Final plan output.

        Returns:
            Formatted plan result string.
        """
        lines = [
            self.colored.format_header(f"PLAN: {plan_id}", width=50),
            "",
        ]

        # Format step summary
        completed = sum(1 for s in steps if s.get("status") == "completed")
        total = len(steps)
        lines.append(f"Steps: {completed}/{total} completed")
        lines.append("")

        # List steps with status
        for i, step in enumerate(steps, 1):
            status = step.get("status", "pending")
            name = step.get("name", step.get("step_id", f"Step {i}"))
            status_symbol = {
                "completed": "[+]",
                "failed": "[X]",
                "skipped": "[-]",
                "pending": "[ ]",
            }.get(status, "[ ]")
            status_color = {
                "completed": "success",
                "failed": "error",
                "skipped": "warning",
                "pending": "dim",
            }.get(status, "dim")

            step_line = f"  {self.colored.colorize(status_symbol, status_color)} {name}"
            lines.append(step_line)

        # Add output if present
        if output:
            lines.append("")
            lines.append(self.colored.bold("Output:"))
            truncated = output[:500] + "..." if len(output) > 500 else output
            lines.append(f"  {truncated}")

        return "\n".join(lines)

    def format_error_result(
        self,
        error: str,
        context: str = "",
    ) -> str:
        """Format error result.

        Args:
            error: Error message.
            context: Optional context information.

        Returns:
            Formatted error string.
        """
        lines = [
            self.colored.format_error(error),
        ]

        if context:
            lines.append(self.colored.dim(f"  Context: {context}"))

        return "\n".join(lines)

    def format_token_summary(
        self,
        tokens_used: int,
        budget: int = DEFAULT_TOKEN_BUDGET,
    ) -> str:
        """Format token usage summary.

        Args:
            tokens_used: Tokens consumed.
            budget: Total budget.

        Returns:
            Formatted token summary string.
        """
        percentage = (tokens_used / budget) * 100 if budget > 0 else 100.0

        # Determine status color
        if percentage >= TOKEN_THRESHOLDS["critical"]:
            color = "error"
            status = "CRITICAL"
        elif percentage >= TOKEN_THRESHOLDS["high"]:
            color = "warning"
            status = "WARNING"
        else:
            color = "success"
            status = "OK"

        meter = TokenMeter(budget=budget)
        meter.add_tokens(tokens_used)

        lines = [
            self.colored.bold("Token Usage:"),
            f"  {meter.format_meter()}",
            f"  Status: {self.colored.colorize(status, color)}",
        ]

        return "\n".join(lines)


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Module-level function
    "colorize",
    # Classes
    "ColoredOutput",
    "TokenMeter",
    "ResultFormatter",
]
