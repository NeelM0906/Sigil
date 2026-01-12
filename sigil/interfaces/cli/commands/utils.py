"""Utility functions for CLI command handlers.

This module provides common utilities used across all command handlers:
- Session management
- Result display
- Error formatting
- Argument validation
- Async command wrapper
- Structured logging with token tracking

Example:
    >>> from sigil.interfaces.cli.commands.utils import get_active_session, display_result
    >>> session = get_active_session()
    >>> result = some_operation()
    >>> display_result(result)

Logging Example:
    >>> from sigil.interfaces.cli.commands.utils import logged_command
    >>> @logged_command("orchestrate")
    ... async def orchestrate(task: str) -> dict:
    ...     result = await do_orchestration(task)
    ...     return {"result": result, "tokens_used": 523}
"""

from __future__ import annotations

import asyncio
import functools
import sys
import time
from typing import Any, Callable, Optional, TypeVar, ParamSpec

import click

from sigil.interfaces.cli.session import (
    InteractiveSession,
    get_active_session as _get_active_session,
)
from sigil.interfaces.cli.formatter import ColoredOutput, TokenMeter, ResultFormatter
from sigil.interfaces.cli.logging import (
    CliLogger,
    get_cli_logger_wrapper,
    setup_cli_logging,
)


# =============================================================================
# Type Variables
# =============================================================================

P = ParamSpec("P")
T = TypeVar("T")


# =============================================================================
# Session Management
# =============================================================================


def get_active_session() -> InteractiveSession:
    """Get or create the active CLI session.

    This delegates to the session module's get_active_session function.

    Returns:
        The active InteractiveSession instance.
    """
    return _get_active_session()


def ensure_session() -> InteractiveSession:
    """Ensure a session is active, starting one if needed.

    Returns:
        The active InteractiveSession instance.
    """
    session = get_active_session()
    if not session.is_active:
        session.start_new()
    return session


# =============================================================================
# Output Display
# =============================================================================


def get_output() -> ColoredOutput:
    """Get a ColoredOutput instance.

    Returns:
        ColoredOutput configured with current settings.
    """
    return ColoredOutput()


def display_result(result: dict[str, Any], verbose: bool = False) -> None:
    """Display a formatted result from an operation.

    Args:
        result: The result dictionary to display.
        verbose: Whether to show detailed output.
    """
    formatter = ResultFormatter()
    formatter.format_orchestration_result(result)


def display_success(message: str) -> None:
    """Display a success message.

    Args:
        message: The success message.
    """
    out = get_output()
    out.success(message)


def display_error(message: str, details: Optional[str] = None) -> None:
    """Display an error message.

    Args:
        message: The error message.
        details: Optional additional details.
    """
    out = get_output()
    out.error(message)
    if details:
        out.plain(f"  Details: {details}")


def display_warning(message: str) -> None:
    """Display a warning message.

    Args:
        message: The warning message.
    """
    out = get_output()
    out.warning(message)


def display_info(message: str) -> None:
    """Display an info message.

    Args:
        message: The info message.
    """
    out = get_output()
    out.info(message)


def display_tokens(used: int, budget: int = 256_000) -> None:
    """Display token usage.

    Args:
        used: Number of tokens used.
        budget: Total token budget.
    """
    meter = TokenMeter(budget=budget)
    meter.display_with_bar(used)


def display_table(
    headers: list[str],
    rows: list[list[Any]],
    widths: Optional[list[int]] = None,
) -> None:
    """Display a formatted table.

    Args:
        headers: Column headers.
        rows: Table rows.
        widths: Optional column widths.
    """
    out = get_output()

    # Calculate widths if not provided
    if widths is None:
        widths = []
        for i, header in enumerate(headers):
            max_width = len(str(header))
            for row in rows:
                if i < len(row):
                    max_width = max(max_width, len(str(row[i])))
            widths.append(max_width)

    # Print header
    out.table_row(headers, widths)
    out.separator(char="-", width=sum(widths) + (len(widths) - 1) * 3)

    # Print rows
    for row in rows:
        out.table_row([str(c) for c in row], widths)


def display_list(items: list[str], title: Optional[str] = None) -> None:
    """Display a list of items.

    Args:
        items: List of items to display.
        title: Optional title for the list.
    """
    out = get_output()
    if title:
        out.section(title)
    for item in items:
        out.list_item(item)


# =============================================================================
# Error Formatting
# =============================================================================


def format_error(error: Exception) -> str:
    """Format an exception into a user-friendly message.

    Args:
        error: The exception to format.

    Returns:
        Formatted error message string.
    """
    error_type = type(error).__name__
    error_msg = str(error)

    # Add suggestions based on error type
    suggestions = {
        "TokenBudgetExceeded": "Try simplifying your query or use /budget reset",
        "ContractViolation": "The output did not meet contract requirements. Try /contracts test <name>",
        "MemoryError": "Memory operation failed. Try /memory stats to check health",
        "RoutingError": "Could not route request. Use /help for command syntax",
        "SessionNotFoundError": "Session not found. Use /sessions list to see available sessions",
        "ValidationError": "Invalid input. Check argument format and try again",
    }

    suggestion = suggestions.get(error_type, "")
    message = f"{error_type}: {error_msg}"
    if suggestion:
        message += f"\n  Suggestion: {suggestion}"

    return message


# =============================================================================
# Argument Validation
# =============================================================================


def validate_command_args(
    args: dict[str, Any],
    required: Optional[list[str]] = None,
    types: Optional[dict[str, type]] = None,
) -> tuple[bool, Optional[str]]:
    """Validate command arguments.

    Args:
        args: Dictionary of argument name -> value.
        required: List of required argument names.
        types: Dictionary of argument name -> expected type.

    Returns:
        Tuple of (is_valid, error_message).
    """
    required = required or []
    types = types or {}

    # Check required arguments
    for arg_name in required:
        if arg_name not in args or args[arg_name] is None:
            return False, f"Missing required argument: {arg_name}"

    # Check types
    for arg_name, expected_type in types.items():
        if arg_name in args and args[arg_name] is not None:
            if not isinstance(args[arg_name], expected_type):
                return False, (
                    f"Invalid type for {arg_name}: "
                    f"expected {expected_type.__name__}, "
                    f"got {type(args[arg_name]).__name__}"
                )

    return True, None


# =============================================================================
# Async Command Wrapper
# =============================================================================


def async_command(f: Callable[P, T]) -> Callable[P, T]:
    """Decorator to run async functions in Click commands.

    This wrapper handles running async functions within Click's synchronous
    command framework.

    Args:
        f: The async function to wrap.

    Returns:
        Wrapped function that runs async code.

    Example:
        >>> @click.command()
        >>> @async_command
        >>> async def my_command():
        ...     result = await some_async_operation()
        ...     click.echo(result)
    """
    @functools.wraps(f)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        return asyncio.run(f(*args, **kwargs))
    return wrapper


# =============================================================================
# Context Helpers
# =============================================================================


def get_ctx_debug(ctx: click.Context) -> bool:
    """Check if debug mode is enabled in Click context.

    Args:
        ctx: Click context.

    Returns:
        True if debug mode is enabled.
    """
    return ctx.obj.get("debug", False) if ctx.obj else False


def get_ctx_verbose(ctx: click.Context) -> bool:
    """Check if verbose mode is enabled in Click context.

    Args:
        ctx: Click context.

    Returns:
        True if verbose mode is enabled.
    """
    return ctx.obj.get("verbose", False) if ctx.obj else False


# =============================================================================
# Progress Display
# =============================================================================


def show_progress(message: str) -> None:
    """Show a progress indicator.

    Args:
        message: Progress message.
    """
    click.echo(f"  -> {message}...", nl=False)


def complete_progress(success: bool = True) -> None:
    """Complete a progress indicator.

    Args:
        success: Whether the operation succeeded.
    """
    if success:
        click.echo(" done")
    else:
        click.echo(" failed")


# =============================================================================
# Confirmation
# =============================================================================


def confirm_action(message: str, default: bool = False) -> bool:
    """Ask for user confirmation.

    Args:
        message: Confirmation message.
        default: Default value if user just presses Enter.

    Returns:
        True if user confirmed, False otherwise.
    """
    return click.confirm(message, default=default)


# =============================================================================
# Command Logging
# =============================================================================


def get_command_logger() -> CliLogger:
    """Get the CLI logger for command tracking.

    Automatically sets the session ID from the active session.

    Returns:
        CliLogger instance with current session ID.
    """
    session = get_active_session()
    session_id = session.session_id or "unknown"
    return get_cli_logger_wrapper(session_id)


def logged_command(command_name: str):
    """Decorator for automatic command logging with token tracking.

    Wraps a command function to automatically log START/COMPLETE/ERROR
    events with timing and token tracking. The decorated function should
    return a dict with at least 'tokens_used' key for proper tracking.

    Args:
        command_name: Name of the command for logging (e.g., "orchestrate").

    Returns:
        Decorator function.

    Example:
        >>> @click.command()
        >>> @logged_command("orchestrate")
        >>> @async_command
        >>> async def orchestrate(ctx, task):
        ...     result = await do_orchestration(task)
        ...     return {
        ...         "result": result,
        ...         "tokens_used": 523,
        ...         "confidence": 0.87,
        ...     }

    The decorator:
    1. Logs START event before execution with user input
    2. Executes the command function
    3. Logs COMPLETE event with tokens_used, duration_ms, and confidence
    4. On exception, logs ERROR event with error details and re-raises
    """
    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        @functools.wraps(func)
        async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            # Initialize logging
            setup_cli_logging()
            cli_logger = get_command_logger()

            # Extract user input from common argument patterns
            user_input = _extract_user_input(args, kwargs)

            # Log command start
            cli_logger.log_command_start(command_name, user_input)

            try:
                # Execute the command
                result = await func(*args, **kwargs)

                # Extract token info from result
                tokens_used = 0
                confidence = None
                output = None

                if isinstance(result, dict):
                    tokens_used = result.get("tokens_used", 0)
                    confidence = result.get("confidence")
                    output = result.get("result") or result.get("output")
                    # Also check for nested token info
                    if tokens_used == 0 and "tokens" in result:
                        token_info = result["tokens"]
                        if isinstance(token_info, dict):
                            tokens_used = token_info.get("total", 0)

                # Log command completion
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
        def sync_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            # Initialize logging
            setup_cli_logging()
            cli_logger = get_command_logger()

            # Extract user input
            user_input = _extract_user_input(args, kwargs)

            # Log command start
            cli_logger.log_command_start(command_name, user_input)

            try:
                # Execute the command
                result = func(*args, **kwargs)

                # Extract token info
                tokens_used = 0
                confidence = None
                output = None

                if isinstance(result, dict):
                    tokens_used = result.get("tokens_used", 0)
                    confidence = result.get("confidence")
                    output = result.get("result") or result.get("output")
                    if tokens_used == 0 and "tokens" in result:
                        token_info = result["tokens"]
                        if isinstance(token_info, dict):
                            tokens_used = token_info.get("total", 0)

                # Log completion
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

        # Return appropriate wrapper
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator


def _extract_user_input(args: tuple, kwargs: dict) -> str:
    """Extract user input from command arguments.

    Looks for common argument names used across commands.

    Args:
        args: Positional arguments.
        kwargs: Keyword arguments.

    Returns:
        User input string, or empty if not found.
    """
    # Common argument names for user input
    input_keys = ["task", "query", "message", "content", "text", "input", "goal"]

    # Check kwargs first
    for key in input_keys:
        if key in kwargs and kwargs[key]:
            return str(kwargs[key])

    # Check first positional arg (often ctx, skip it)
    for arg in args:
        # Skip Click context objects
        if isinstance(arg, click.Context):
            continue
        # Return first string-like argument
        if isinstance(arg, str) and arg:
            return arg

    return ""


def log_command_manually(
    command_name: str,
    user_input: str,
    tokens_used: int = 0,
    confidence: Optional[float] = None,
    output: Optional[str] = None,
    error: Optional[Exception] = None,
) -> None:
    """Manually log a command execution.

    Use this for cases where the decorator doesn't fit, such as
    commands that don't return a dict or have complex execution patterns.

    Args:
        command_name: Name of the command.
        user_input: User's input.
        tokens_used: Tokens consumed.
        confidence: Confidence score (optional).
        output: Command output (optional).
        error: Exception if command failed (optional).

    Example:
        >>> log_command_manually(
        ...     command_name="memory query",
        ...     user_input="customer preferences",
        ...     tokens_used=45,
        ...     output="Found 5 relevant memories",
        ... )
    """
    setup_cli_logging()
    cli_logger = get_command_logger()

    cli_logger.log_command_start(command_name, user_input)

    if error:
        cli_logger.log_command_error(command_name, error)
    else:
        cli_logger.log_command_complete(
            command_name,
            tokens_used=tokens_used,
            confidence=confidence,
            output=output,
        )


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Session
    "get_active_session",
    "ensure_session",
    # Output
    "get_output",
    "display_result",
    "display_success",
    "display_error",
    "display_warning",
    "display_info",
    "display_tokens",
    "display_table",
    "display_list",
    # Error formatting
    "format_error",
    # Validation
    "validate_command_args",
    # Async
    "async_command",
    # Context
    "get_ctx_debug",
    "get_ctx_verbose",
    # Progress
    "show_progress",
    "complete_progress",
    # Confirmation
    "confirm_action",
    # Logging
    "get_command_logger",
    "logged_command",
    "log_command_manually",
]
