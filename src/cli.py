"""Interactive CLI for ACTi Agent Builder.

This module provides a chat-like command-line interface for creating, managing,
and interacting with AI agents. Users can create agents via natural language,
load existing agents, and chat with them using real MCP tools.

Usage:
    python -m src.cli

Commands:
    (no prefix)          - Chat with active agent or builder
    /create <prompt>     - Create a new agent from natural language
    /list                - List all saved agents
    /load <name>         - Load and activate an existing agent
    /run <name> <task>   - Run a specific agent with a task
    /modify <name> [instr] - Modify agent (natural language or interactive)
    /rollback <name>     - Restore agent from backup
    /delete <name>       - Delete an agent
    /status              - Show current agent and MCP tool status
    /tools               - List available MCP tools
    /builder             - Switch back to builder mode
    /stream [mode]       - Configure streaming output (on/off/verbose)
    /debug [N|clear|path] - Show recent errors and debug info
    /help                - Show available commands
    /exit                - Exit the CLI

Streaming Output:
    The CLI supports streaming output for real-time token display:
    - /stream on       - Enable streaming with tool indicators (default)
    - /stream off      - Disable streaming (batch mode)
    - /stream verbose  - Show detailed tool inputs/outputs

    A spinner displays while waiting for the first token. Tool calls
    are indicated in-line during streaming. Non-TTY output automatically
    uses batch mode.

Modification Flow:
    The /modify command supports two modes:

    1. Natural Language Mode:
       /modify my_agent make it friendlier and add websearch tool
       - Builder interprets the instruction
       - Shows diff of proposed changes
       - Confirms before applying

    2. Interactive Mode:
       /modify my_agent
       - Enters modification prompt
       - Supports commands: set, add tool, remove tool
       - Natural language descriptions also work
       - Type 'done' to apply, 'cancel' to abort

    Backups are automatically created before modifications.
    Use /rollback <name> to restore from backup.

Error Handling:
    The CLI provides comprehensive error handling with:
    - Categorized error codes (MCP_CONNECTION_FAILED, AGENT_NOT_FOUND, etc.)
    - Recovery suggestions for each error type
    - Error logging to ~/.acti/logs/error.log
    - Retry logic for transient failures (network, timeouts)
    - Graceful degradation when MCP tools are unavailable

    Use /debug to view recent errors and their suggestions.
    Error logs include timestamps, tracebacks, and context information.
"""

from __future__ import annotations

import asyncio
import atexit
import json
import logging
import os
import readline
import signal
import sys
import traceback
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Coroutine, Optional, TypeVar

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from .tools import extract_text_from_content, validate_and_patch_messages, reset_tool_history

if TYPE_CHECKING:
    from langgraph.graph.state import CompiledStateGraph

# Type variable for retry function
T = TypeVar("T")


# =============================================================================
# Streaming Configuration
# =============================================================================

class StreamingMode(str, Enum):
    """Streaming output modes."""

    ON = "on"       # Enable streaming (default for TTY)
    OFF = "off"     # Disable streaming (batch mode)
    VERBOSE = "verbose"  # Show all tool details


@dataclass
class StreamingConfig:
    """Configuration for streaming output.

    Attributes:
        enabled: Whether streaming is enabled (default True for TTY)
        show_tool_calls: Show tool invocation indicators
        show_thinking: Show model thinking/reasoning (if available)
        spinner: Show spinner while waiting for first token
        mode: Streaming mode (on, off, verbose)
    """

    enabled: bool = True
    show_tool_calls: bool = True
    show_thinking: bool = False
    spinner: bool = True
    mode: StreamingMode = StreamingMode.ON

    @classmethod
    def for_tty(cls) -> "StreamingConfig":
        """Create config optimized for TTY (interactive terminal)."""
        return cls(
            enabled=True,
            show_tool_calls=True,
            show_thinking=False,
            spinner=True,
            mode=StreamingMode.ON,
        )

    @classmethod
    def for_non_tty(cls) -> "StreamingConfig":
        """Create config for non-TTY (piped output, scripts)."""
        return cls(
            enabled=False,
            show_tool_calls=False,
            show_thinking=False,
            spinner=False,
            mode=StreamingMode.OFF,
        )

    def set_mode(self, mode: StreamingMode) -> None:
        """Set streaming mode and adjust settings accordingly."""
        self.mode = mode
        if mode == StreamingMode.OFF:
            self.enabled = False
            self.spinner = False
        elif mode == StreamingMode.ON:
            self.enabled = True
            self.show_tool_calls = True
            self.spinner = True
        elif mode == StreamingMode.VERBOSE:
            self.enabled = True
            self.show_tool_calls = True
            self.show_thinking = True
            self.spinner = True

# Configure module logger
logger = logging.getLogger(__name__)


# =============================================================================
# ANSI Color Codes for Terminal Output
# =============================================================================

class Colors:
    """ANSI color codes for terminal output."""

    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"

    # Foreground colors
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"

    # Bright foreground colors
    BRIGHT_RED = "\033[91m"
    BRIGHT_GREEN = "\033[92m"
    BRIGHT_YELLOW = "\033[93m"
    BRIGHT_BLUE = "\033[94m"
    BRIGHT_MAGENTA = "\033[95m"
    BRIGHT_CYAN = "\033[96m"

    @classmethod
    def disable(cls) -> None:
        """Disable colors (for non-TTY output)."""
        cls.RESET = ""
        cls.BOLD = ""
        cls.DIM = ""
        cls.RED = ""
        cls.GREEN = ""
        cls.YELLOW = ""
        cls.BLUE = ""
        cls.MAGENTA = ""
        cls.CYAN = ""
        cls.WHITE = ""
        cls.BRIGHT_RED = ""
        cls.BRIGHT_GREEN = ""
        cls.BRIGHT_YELLOW = ""
        cls.BRIGHT_BLUE = ""
        cls.BRIGHT_MAGENTA = ""
        cls.BRIGHT_CYAN = ""


def colorize(text: str, color: str) -> str:
    """Apply ANSI color to text."""
    return f"{color}{text}{Colors.RESET}"


def success(text: str) -> str:
    """Format text as success (green)."""
    return colorize(text, Colors.GREEN)


def error(text: str) -> str:
    """Format text as error (red)."""
    return colorize(text, Colors.RED)


def warning(text: str) -> str:
    """Format text as warning (yellow)."""
    return colorize(text, Colors.YELLOW)


def info(text: str) -> str:
    """Format text as info (cyan)."""
    return colorize(text, Colors.CYAN)


def highlight(text: str) -> str:
    """Format text as highlighted (bold)."""
    return colorize(text, Colors.BOLD)


def dim(text: str) -> str:
    """Format text as dimmed."""
    return colorize(text, Colors.DIM)


# =============================================================================
# Error Handling & Recovery
# =============================================================================

# Error codes and their user-friendly suggestions for recovery
ERROR_SUGGESTIONS: dict[str, str] = {
    # MCP-related errors
    "MCP_CONNECTION_FAILED": "Check your internet connection and verify the MCP server is running",
    "MCP_TIMEOUT": "The server is taking too long to respond. Try again or check server status",
    "MCP_CREDENTIAL_MISSING": "Check your API keys in .env file. Use /status to see which are missing",
    "MCP_CONFIGURATION_ERROR": "Invalid MCP configuration. Check CLAUDE.md for correct setup",
    # Agent-related errors
    "AGENT_NOT_FOUND": "Use /list to see available agents, then /load <name> to load one",
    "AGENT_CREATION_FAILED": "Check your prompt and try again. Use /help for examples",
    "AGENT_EXECUTION_FAILED": "The agent encountered an error. Try rephrasing your request",
    "AGENT_LOAD_FAILED": "Failed to load agent. Check if the config file is valid JSON",
    # Tool-related errors
    "INVALID_TOOL": "Use /tools to see available MCP tools and their status",
    "TOOL_NOT_CONFIGURED": "Required credentials missing. Check .env file and /status",
    "TOOL_EXECUTION_FAILED": "Tool execution failed. Check tool configuration and try again",
    # File/IO errors
    "FILE_NOT_FOUND": "The specified file does not exist. Check the path and try again",
    "FILE_PERMISSION_ERROR": "Permission denied. Check file permissions",
    "FILE_CORRUPTED": "Config file appears corrupted. Try /rollback if a backup exists",
    # Session errors
    "SESSION_SAVE_FAILED": "Failed to save session. Check disk space and permissions",
    "SESSION_RESTORE_FAILED": "Failed to restore session. Starting fresh",
    # General errors
    "TIMEOUT": "Operation timed out. Try again or increase timeout with /stream verbose",
    "NETWORK_ERROR": "Network error occurred. Check your internet connection",
    "API_KEY_INVALID": "API key is invalid or expired. Update your .env file",
    "RATE_LIMIT": "Rate limit exceeded. Wait a moment and try again",
    "UNKNOWN_ERROR": "An unexpected error occurred. Check logs with /debug",
}


# Logs directory
LOGS_DIR = Path.home() / ".acti" / "logs"


class ErrorLogger:
    """Handles logging errors to file for debugging and recovery.

    Logs are written to ~/.acti/logs/error.log with timestamps, error types,
    messages, and full tracebacks for debugging purposes.

    Attributes:
        log_file: Path to the error log file
        max_entries: Maximum number of entries to keep in memory for /debug
    """

    def __init__(self, max_entries: int = 50) -> None:
        """Initialize the error logger.

        Args:
            max_entries: Maximum recent errors to keep in memory for display.
        """
        self._ensure_log_dir()
        self.log_file = LOGS_DIR / "error.log"
        self.max_entries = max_entries
        self._recent_errors: list[dict] = []

    def _ensure_log_dir(self) -> None:
        """Ensure the logs directory exists."""
        LOGS_DIR.mkdir(parents=True, exist_ok=True)

    def log_error(
        self,
        error: Exception,
        error_code: str = "UNKNOWN_ERROR",
        context: Optional[dict] = None,
    ) -> None:
        """Log an error to file and memory.

        Args:
            error: The exception that occurred.
            error_code: A categorized error code from ERROR_SUGGESTIONS.
            context: Optional dictionary with additional context information.
        """
        timestamp = datetime.now().isoformat()
        error_type = type(error).__name__
        error_message = str(error)
        tb = traceback.format_exc()

        # Build log entry
        entry = {
            "timestamp": timestamp,
            "error_code": error_code,
            "error_type": error_type,
            "message": error_message,
            "traceback": tb,
            "context": context or {},
        }

        # Add to recent errors (keep last N)
        self._recent_errors.append(entry)
        if len(self._recent_errors) > self.max_entries:
            self._recent_errors = self._recent_errors[-self.max_entries:]

        # Write to log file
        try:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(f"\n{'=' * 60}\n")
                f.write(f"Timestamp: {timestamp}\n")
                f.write(f"Error Code: {error_code}\n")
                f.write(f"Error Type: {error_type}\n")
                f.write(f"Message: {error_message}\n")
                if context:
                    f.write(f"Context: {json.dumps(context, indent=2)}\n")
                f.write(f"Traceback:\n{tb}\n")
        except OSError as e:
            # If we can't write to log, at least keep in memory
            logger.warning(f"Could not write to error log: {e}")

    def get_recent_errors(self, count: int = 10) -> list[dict]:
        """Get the most recent errors.

        Args:
            count: Number of recent errors to return.

        Returns:
            List of error dictionaries, most recent first.
        """
        return list(reversed(self._recent_errors[-count:]))

    def clear_log(self) -> bool:
        """Clear the error log file.

        Returns:
            True if cleared successfully, False otherwise.
        """
        try:
            if self.log_file.exists():
                self.log_file.unlink()
            self._recent_errors = []
            return True
        except OSError:
            return False

    def get_log_path(self) -> Path:
        """Get the path to the error log file.

        Returns:
            Path to the error log file.
        """
        return self.log_file


# Global error logger instance
error_logger = ErrorLogger()


def print_error(
    message: str,
    error_code: Optional[str] = None,
    suggestion: Optional[str] = None,
) -> None:
    """Display a formatted error message to the user.

    Args:
        message: The error message to display.
        error_code: Optional error code for categorization.
        suggestion: Optional suggestion for recovery. If not provided and
            error_code is given, will look up suggestion from ERROR_SUGGESTIONS.
    """
    print(f"{Colors.RED}Error:{Colors.RESET} {message}")

    if error_code:
        print(f"{Colors.DIM}Code: {error_code}{Colors.RESET}")

    # Look up suggestion if not provided
    if suggestion is None and error_code:
        suggestion = ERROR_SUGGESTIONS.get(error_code)

    if suggestion:
        print(f"{Colors.YELLOW}Suggestion:{Colors.RESET} {suggestion}")


def print_warning(message: str) -> None:
    """Display a formatted warning message to the user.

    Args:
        message: The warning message to display.
    """
    print(f"{Colors.YELLOW}Warning:{Colors.RESET} {message}")


def get_error_code_for_exception(error: Exception) -> str:
    """Determine the appropriate error code for an exception.

    Args:
        error: The exception to categorize.

    Returns:
        An error code string from ERROR_SUGGESTIONS keys.
    """
    # Import MCP exceptions here to avoid circular imports
    try:
        from .mcp_integration import (
            MCPConfigurationError,
            MCPConnectionError,
            MCPCredentialError,
            MCPIntegrationError,
            MCPTimeoutError,
        )

        if isinstance(error, MCPConnectionError):
            return "MCP_CONNECTION_FAILED"
        elif isinstance(error, MCPTimeoutError):
            return "MCP_TIMEOUT"
        elif isinstance(error, MCPCredentialError):
            return "MCP_CREDENTIAL_MISSING"
        elif isinstance(error, MCPConfigurationError):
            return "MCP_CONFIGURATION_ERROR"
        elif isinstance(error, MCPIntegrationError):
            return "TOOL_EXECUTION_FAILED"
    except ImportError:
        pass

    # Standard exceptions
    if isinstance(error, FileNotFoundError):
        return "FILE_NOT_FOUND"
    elif isinstance(error, PermissionError):
        return "FILE_PERMISSION_ERROR"
    elif isinstance(error, json.JSONDecodeError):
        return "FILE_CORRUPTED"
    elif isinstance(error, TimeoutError):
        return "TIMEOUT"
    elif isinstance(error, asyncio.TimeoutError):
        return "TIMEOUT"
    elif isinstance(error, ConnectionError):
        return "NETWORK_ERROR"

    return "UNKNOWN_ERROR"


async def with_retry(
    coro_func: Callable[[], Coroutine[Any, Any, T]],
    max_retries: int = 3,
    delay: float = 1.0,
    retry_on: Optional[tuple[type[Exception], ...]] = None,
) -> T:
    """Retry a coroutine on transient failures with exponential backoff.

    Args:
        coro_func: A callable that returns a coroutine to execute.
        max_retries: Maximum number of retry attempts.
        delay: Base delay between retries in seconds (multiplied by attempt number).
        retry_on: Tuple of exception types to retry on. If None, retries on
            common transient errors (MCPConnectionError, TimeoutError, ConnectionError).

    Returns:
        The result of the successful coroutine execution.

    Raises:
        The last exception if all retries fail.

    Example:
        >>> result = await with_retry(
        ...     lambda: some_async_operation(),
        ...     max_retries=3,
        ...     delay=1.0
        ... )
    """
    # Import MCP exceptions for retry logic
    try:
        from .mcp_integration import MCPConnectionError, MCPTimeoutError
        default_retry_on = (MCPConnectionError, MCPTimeoutError, TimeoutError, ConnectionError)
    except ImportError:
        default_retry_on = (TimeoutError, ConnectionError)

    retry_exceptions = retry_on or default_retry_on
    last_error: Optional[Exception] = None

    for attempt in range(max_retries):
        try:
            return await coro_func()
        except retry_exceptions as e:
            last_error = e
            if attempt == max_retries - 1:
                # Last attempt failed, raise the error
                raise

            # Log and wait before retry
            wait_time = delay * (attempt + 1)
            print_warning(
                f"Attempt {attempt + 1}/{max_retries} failed: {e}. "
                f"Retrying in {wait_time:.1f}s..."
            )
            await asyncio.sleep(wait_time)

    # Should not reach here, but satisfy type checker
    if last_error:
        raise last_error
    raise RuntimeError("Retry loop completed without success or error")


def handle_error(
    error: Exception,
    context: Optional[str] = None,
    log: bool = True,
) -> None:
    """Handle an error with proper logging and user feedback.

    This is the central error handling function that:
    - Determines the appropriate error code
    - Logs the error if requested
    - Displays a user-friendly message with recovery suggestions

    Args:
        error: The exception to handle.
        context: Optional context string describing what was being attempted.
        log: Whether to log the error to file.
    """
    error_code = get_error_code_for_exception(error)
    error_message = str(error)

    # Build context dict for logging
    context_dict = {"operation": context} if context else {}

    # Log the error
    if log:
        error_logger.log_error(error, error_code, context_dict)
        logger.exception(f"Error during {context or 'operation'}")

    # Display to user
    if context:
        print_error(f"{context}: {error_message}", error_code)
    else:
        print_error(error_message, error_code)


# =============================================================================
# Session State Management
# =============================================================================

# Base directory for ACTi session data
ACTI_DIR = Path.home() / ".acti"
SESSIONS_DIR = ACTI_DIR / "sessions"
HISTORY_DIR = ACTI_DIR / "history"

# Maximum messages to keep per agent history
MAX_HISTORY_MESSAGES = 100

# Maximum messages per conversation before forced reset (Fix 9)
# This prevents infinite loops from consuming too much memory/context
MAX_MESSAGES_PER_CONVERSATION = 100


class SessionMode(str, Enum):
    """Current interaction mode for the session."""

    BUILDER = "builder"  # Interacting with the builder agent
    AGENT = "agent"      # Interacting with a loaded agent


def ensure_acti_directories() -> None:
    """Ensure all ACTi directories exist."""
    ACTI_DIR.mkdir(parents=True, exist_ok=True)
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)


def _serialize_message(msg: Any) -> dict:
    """Serialize a LangChain message to a dictionary.

    Args:
        msg: A LangChain message object (HumanMessage, AIMessage, etc.)

    Returns:
        Dictionary representation of the message.
    """
    if hasattr(msg, "type"):
        serialized = {
            "type": msg.type,
            "content": msg.content,
        }
        # Preserve tool_calls if present (for AIMessage)
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            serialized["tool_calls"] = msg.tool_calls
        # Preserve tool_call_id if present (for ToolMessage)
        if hasattr(msg, "tool_call_id"):
            serialized["tool_call_id"] = msg.tool_call_id
        # Preserve name if present (for ToolMessage)
        if hasattr(msg, "name"):
            serialized["name"] = msg.name
        return serialized
    elif isinstance(msg, dict):
        return msg
    else:
        return {"type": "unknown", "content": str(msg)}


def _deserialize_message(data: dict) -> Any:
    """Deserialize a dictionary to a LangChain message.

    Args:
        data: Dictionary representation of a message.

    Returns:
        A LangChain message object.
    """
    msg_type = data.get("type", "human")
    content = data.get("content", "")

    if msg_type == "human":
        return HumanMessage(content=content)
    elif msg_type == "ai":
        # Restore tool_calls if present
        tool_calls = data.get("tool_calls", [])
        return AIMessage(content=content, tool_calls=tool_calls if tool_calls else None)
    elif msg_type == "tool":
        # Restore ToolMessage with tool_call_id and name
        tool_call_id = data.get("tool_call_id", "")
        name = data.get("name", "tool")
        return ToolMessage(content=content, tool_call_id=tool_call_id, name=name)
    else:
        return HumanMessage(content=content)


@dataclass
class SessionMetadata:
    """Metadata for a session.

    Attributes:
        session_id: Unique identifier for the session
        created_at: Timestamp when session was created
        last_used: Timestamp when session was last active
        agents_used: List of agent names used in this session
        total_messages: Total message count across all interactions
    """

    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    last_used: str = field(default_factory=lambda: datetime.now().isoformat())
    agents_used: list = field(default_factory=list)
    total_messages: int = 0

    def update_last_used(self) -> None:
        """Update the last_used timestamp to now."""
        self.last_used = datetime.now().isoformat()

    def record_agent_used(self, agent_name: str) -> None:
        """Record that an agent was used in this session."""
        if agent_name and agent_name not in self.agents_used:
            self.agents_used.append(agent_name)

    def increment_messages(self, count: int = 1) -> None:
        """Increment the total message count."""
        self.total_messages += count

    def to_dict(self) -> dict:
        """Convert metadata to dictionary for serialization."""
        return {
            "session_id": self.session_id,
            "created_at": self.created_at,
            "last_used": self.last_used,
            "agents_used": self.agents_used,
            "total_messages": self.total_messages,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SessionMetadata":
        """Create SessionMetadata from dictionary."""
        return cls(
            session_id=data.get("session_id", str(uuid.uuid4())),
            created_at=data.get("created_at", datetime.now().isoformat()),
            last_used=data.get("last_used", datetime.now().isoformat()),
            agents_used=data.get("agents_used", []),
            total_messages=data.get("total_messages", 0),
        )


class SessionManager:
    """Manages session persistence and per-agent conversation history.

    This class handles:
    - Saving/restoring session state to ~/.acti/sessions/current.json
    - Per-agent conversation history in ~/.acti/history/{agent_name}.json
    - Session metadata tracking
    """

    def __init__(self) -> None:
        """Initialize the session manager."""
        ensure_acti_directories()
        self._current_session_path = SESSIONS_DIR / "current.json"

    def save_session(self, session: "Session") -> bool:
        """Save the current session state to disk.

        Args:
            session: The Session object to save.

        Returns:
            True if saved successfully, False otherwise.
        """
        try:
            # Update metadata
            session.metadata.update_last_used()

            # Build session state
            session_data = {
                "mode": session.mode.value,
                "active_agent_name": session.active_agent_name,
                "preferences": session.preferences,
                "metadata": session.metadata.to_dict(),
            }

            # Save session state
            with open(self._current_session_path, "w", encoding="utf-8") as f:
                json.dump(session_data, f, indent=2, ensure_ascii=False)

            # Save current agent's history if there is one
            if session.active_agent_name:
                self.save_agent_history(session.active_agent_name, session.agent_messages)

            # Save builder messages
            self.save_agent_history("__builder__", session.builder_messages)

            logger.debug(f"Session saved: {session.metadata.session_id}")
            return True

        except (OSError, TypeError) as e:
            logger.error(f"Failed to save session: {e}")
            return False

    def load_session(self) -> Optional[dict]:
        """Load the session state from disk.

        Returns:
            Dictionary with session state, or None if no saved session.
        """
        if not self._current_session_path.exists():
            return None

        try:
            with open(self._current_session_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.error(f"Failed to load session: {e}")
            return None

    def save_agent_history(self, agent_name: str, messages: list) -> bool:
        """Save conversation history for a specific agent.

        Args:
            agent_name: The agent's name (used as filename).
            messages: List of messages to save.

        Returns:
            True if saved successfully, False otherwise.
        """
        if not agent_name:
            return False

        history_path = HISTORY_DIR / f"{agent_name}.json"

        try:
            # Serialize messages
            serialized = [_serialize_message(msg) for msg in messages]

            # Limit to last MAX_HISTORY_MESSAGES
            if len(serialized) > MAX_HISTORY_MESSAGES:
                serialized = serialized[-MAX_HISTORY_MESSAGES:]

            # Add metadata
            history_data = {
                "agent_name": agent_name,
                "updated_at": datetime.now().isoformat(),
                "message_count": len(serialized),
                "messages": serialized,
            }

            with open(history_path, "w", encoding="utf-8") as f:
                json.dump(history_data, f, indent=2, ensure_ascii=False)

            return True

        except (OSError, TypeError) as e:
            logger.error(f"Failed to save history for {agent_name}: {e}")
            return False

    def load_agent_history(self, agent_name: str) -> list:
        """Load conversation history for a specific agent.

        Args:
            agent_name: The agent's name.

        Returns:
            List of deserialized messages, or empty list if none found.
        """
        if not agent_name:
            return []

        history_path = HISTORY_DIR / f"{agent_name}.json"

        if not history_path.exists():
            return []

        try:
            with open(history_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            messages = data.get("messages", [])
            return [_deserialize_message(msg) for msg in messages]

        except (json.JSONDecodeError, OSError) as e:
            logger.error(f"Failed to load history for {agent_name}: {e}")
            return []

    def clear_agent_history(self, agent_name: str) -> bool:
        """Clear conversation history for a specific agent.

        Args:
            agent_name: The agent's name.

        Returns:
            True if cleared successfully, False otherwise.
        """
        if not agent_name:
            return False

        history_path = HISTORY_DIR / f"{agent_name}.json"

        if history_path.exists():
            try:
                history_path.unlink()
                return True
            except OSError as e:
                logger.error(f"Failed to clear history for {agent_name}: {e}")
                return False

        return True

    def get_session_summary(self, session: "Session") -> str:
        """Generate a human-readable session summary.

        Args:
            session: The current session.

        Returns:
            Formatted string with session information.
        """
        meta = session.metadata

        # Parse timestamps for display
        try:
            created = datetime.fromisoformat(meta.created_at)
            created_str = created.strftime("%Y-%m-%d %H:%M")
        except ValueError:
            created_str = meta.created_at

        try:
            last_used = datetime.fromisoformat(meta.last_used)
            last_used_str = last_used.strftime("%Y-%m-%d %H:%M")
        except ValueError:
            last_used_str = meta.last_used

        agents_str = ", ".join(meta.agents_used) if meta.agents_used else "None"

        return (
            f"Session ID: {meta.session_id[:8]}...\n"
            f"Created: {created_str}\n"
            f"Last Used: {last_used_str}\n"
            f"Agents Used: {agents_str}\n"
            f"Total Messages: {meta.total_messages}"
        )


@dataclass
class Session:
    """Manages the state of a CLI session.

    Attributes:
        mode: Current interaction mode (builder or agent)
        builder: The builder agent instance
        active_agent: Currently loaded agent (if any)
        active_agent_name: Name of the currently loaded agent
        builder_messages: Conversation history with builder
        agent_messages: Conversation history with active agent
        metadata: Session metadata (timestamps, usage stats)
        preferences: User preferences dictionary
        manager: SessionManager instance for persistence
        streaming: Streaming output configuration
    """

    mode: SessionMode = SessionMode.BUILDER
    builder: Optional[CompiledStateGraph] = None
    active_agent: Optional[CompiledStateGraph] = None
    active_agent_name: Optional[str] = None
    builder_messages: list = field(default_factory=list)
    agent_messages: list = field(default_factory=list)
    metadata: SessionMetadata = field(default_factory=SessionMetadata)
    preferences: dict = field(default_factory=dict)
    manager: Optional[SessionManager] = field(default=None, repr=False)
    streaming: StreamingConfig = field(default_factory=StreamingConfig)

    def __post_init__(self) -> None:
        """Initialize session manager after dataclass init."""
        if self.manager is None:
            self.manager = SessionManager()

    @property
    def prompt(self) -> str:
        """Get the appropriate prompt string for current mode."""
        if self.mode == SessionMode.BUILDER:
            return f"{Colors.BRIGHT_BLUE}builder{Colors.RESET}> "
        else:
            agent_name = self.active_agent_name or "agent"
            return f"{Colors.BRIGHT_GREEN}{agent_name}{Colors.RESET}> "

    def switch_to_builder(self) -> None:
        """Switch to builder mode, saving current agent history."""
        # Save current agent's history before switching
        if self.active_agent_name and self.manager:
            self.manager.save_agent_history(self.active_agent_name, self.agent_messages)

        self.mode = SessionMode.BUILDER

    def switch_to_agent(self, agent: CompiledStateGraph, name: str) -> None:
        """Switch to agent mode with a specific agent.

        Saves current agent's history and loads new agent's history if available.

        Args:
            agent: The compiled agent graph.
            name: The agent's name.
        """
        # Save current agent's history before switching
        if self.active_agent_name and self.manager:
            self.manager.save_agent_history(self.active_agent_name, self.agent_messages)

        self.active_agent = agent
        self.active_agent_name = name
        self.mode = SessionMode.AGENT

        # Load existing history for the new agent
        if self.manager:
            self.agent_messages = self.manager.load_agent_history(name)
        else:
            self.agent_messages = []

        # Record agent usage in metadata
        self.metadata.record_agent_used(name)

    def clear_agent(self) -> None:
        """Clear the active agent and switch to builder mode."""
        # Save history before clearing
        if self.active_agent_name and self.manager:
            self.manager.save_agent_history(self.active_agent_name, self.agent_messages)

        self.active_agent = None
        self.active_agent_name = None
        self.agent_messages = []
        self.mode = SessionMode.BUILDER

    def save(self) -> bool:
        """Save the session to disk.

        Returns:
            True if saved successfully, False otherwise.
        """
        if self.manager:
            return self.manager.save_session(self)
        return False

    def record_message(self) -> None:
        """Record that a message was sent (for metadata tracking)."""
        self.metadata.increment_messages()
        self.metadata.update_last_used()

    @classmethod
    def restore(cls) -> "Session":
        """Restore a session from disk or create a new one.

        Returns:
            Restored Session or new Session if no saved state.
        """
        manager = SessionManager()
        session_data = manager.load_session()

        if session_data:
            # Restore from saved state
            mode_str = session_data.get("mode", "builder")
            mode = SessionMode(mode_str) if mode_str in [m.value for m in SessionMode] else SessionMode.BUILDER

            metadata_dict = session_data.get("metadata", {})
            metadata = SessionMetadata.from_dict(metadata_dict)

            preferences = session_data.get("preferences", {})
            active_agent_name = session_data.get("active_agent_name")

            session = cls(
                mode=mode,
                active_agent_name=active_agent_name,
                metadata=metadata,
                preferences=preferences,
                manager=manager,
            )

            # Load builder history
            session.builder_messages = manager.load_agent_history("__builder__")

            # Load agent history if there was an active agent
            if active_agent_name:
                session.agent_messages = manager.load_agent_history(active_agent_name)

            return session
        else:
            # Create new session
            return cls(manager=manager)


# =============================================================================
# Agent Configuration Storage
# =============================================================================

# Output directory for agent configurations
OUTPUT_DIR = Path(__file__).parent.parent / "outputs" / "agents"


def ensure_output_dir() -> Path:
    """Ensure the output directory exists."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR


def get_agent_config_path(name: str) -> Path:
    """Get the path to an agent's configuration file."""
    return OUTPUT_DIR / f"{name}.json"


def load_agent_config_dict(name: str) -> Optional[dict]:
    """Load an agent configuration as a dictionary.

    Args:
        name: The agent name (without .json extension).

    Returns:
        The agent configuration dictionary, or None if not found.
    """
    config_path = get_agent_config_path(name)
    if not config_path.exists():
        return None

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.error(f"Failed to load agent config {name}: {e}")
        return None


def save_agent_config_dict(name: str, config: dict) -> bool:
    """Save an agent configuration dictionary to disk.

    Args:
        name: The agent name.
        config: The configuration dictionary.

    Returns:
        True if saved successfully, False otherwise.
    """
    ensure_output_dir()
    config_path = get_agent_config_path(name)

    try:
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        return True
    except (OSError, TypeError) as e:
        logger.error(f"Failed to save agent config {name}: {e}")
        return False


def delete_agent_config_file(name: str) -> bool:
    """Delete an agent configuration file.

    Args:
        name: The agent name.

    Returns:
        True if deleted successfully, False otherwise.
    """
    config_path = get_agent_config_path(name)
    if not config_path.exists():
        return False

    try:
        config_path.unlink()
        return True
    except OSError as e:
        logger.error(f"Failed to delete agent config {name}: {e}")
        return False


def list_agent_configs() -> list[dict]:
    """List all saved agent configurations.

    Returns:
        List of agent configurations with basic info.
    """
    ensure_output_dir()
    agents = []

    for config_file in OUTPUT_DIR.glob("*.json"):
        try:
            with open(config_file, "r", encoding="utf-8") as f:
                config = json.load(f)
                agents.append({
                    "name": config.get("name", config_file.stem),
                    "description": config.get("description", "No description"),
                    "tools": config.get("tools", []),
                    "stratum": config.get("stratum"),
                    "file": str(config_file),
                })
        except (json.JSONDecodeError, OSError):
            # Include even if we can't parse it
            agents.append({
                "name": config_file.stem,
                "description": "(Error reading config)",
                "tools": [],
                "stratum": None,
                "file": str(config_file),
            })

    return sorted(agents, key=lambda x: x["name"])


# =============================================================================
# Command Handlers
# =============================================================================

async def cmd_create(session: Session, prompt: str) -> None:
    """Handle /create command - create a new agent from natural language.

    Args:
        session: The current CLI session.
        prompt: The natural language description of the agent to create.
    """
    if not prompt:
        print_error(
            "Missing agent description",
            "AGENT_CREATION_FAILED",
            "Usage: /create <natural language description>\n"
            "Example: /create an agent that qualifies leads for a B2B SaaS company"
        )
        return

    if session.builder is None:
        print_error(
            "Builder not initialized",
            "AGENT_CREATION_FAILED",
            "Please restart the CLI to reinitialize the builder"
        )
        return

    print(info("Creating agent..."))
    print()

    # Add the creation request to builder messages
    session.builder_messages.append(HumanMessage(content=prompt))

    try:
        # Invoke the builder with retry for transient failures
        async def invoke_builder():
            return session.builder.invoke({"messages": session.builder_messages})

        result = await with_retry(invoke_builder, max_retries=2, delay=1.0)

        # Extract AI response
        ai_messages = [
            msg for msg in result.get("messages", [])
            if hasattr(msg, "type") and msg.type == "ai"
        ]

        if ai_messages:
            response = extract_text_from_content(ai_messages[-1].content)
            print(f"{Colors.BRIGHT_MAGENTA}Builder:{Colors.RESET} {response}")
            # Update messages with full conversation
            session.builder_messages = result.get("messages", session.builder_messages)
        else:
            print_warning("No response received from builder. Try rephrasing your request.")

    except Exception as e:
        handle_error(e, context="Creating agent")
        # Remove the failed message from history
        if session.builder_messages and session.builder_messages[-1].content == prompt:
            session.builder_messages.pop()

    print()


async def cmd_list(session: Session) -> None:
    """Handle /list command - list all saved agents."""
    agents = list_agent_configs()

    if not agents:
        print(warning("No agents have been created yet."))
        print(dim("Use /create <description> to create your first agent."))
        return

    print(highlight(f"Saved Agents ({len(agents)})"))
    print("=" * 50)

    # Group by stratum
    by_stratum: dict[str, list] = {}
    for agent in agents:
        stratum = agent.get("stratum") or "Unclassified"
        if stratum not in by_stratum:
            by_stratum[stratum] = []
        by_stratum[stratum].append(agent)

    for stratum in ["RTI", "RAI", "ZACS", "EEI", "IGE", "Unclassified"]:
        if stratum in by_stratum:
            print(f"\n{Colors.BRIGHT_CYAN}[{stratum}]{Colors.RESET}")
            for agent in by_stratum[stratum]:
                name = agent["name"]
                desc = agent["description"]
                if len(desc) > 50:
                    desc = desc[:47] + "..."
                tools = ", ".join(agent["tools"]) if agent["tools"] else "none"
                print(f"  {Colors.GREEN}{name}{Colors.RESET}")
                print(f"    {dim(desc)}")
                print(f"    Tools: {tools}")

    print()
    print(dim("Use /load <name> to activate an agent, or /run <name> <task> to execute."))


async def cmd_load(session: Session, name: str) -> None:
    """Handle /load command - load and activate an existing agent.

    Saves current agent's history before switching and loads existing
    history for the new agent if available.

    Args:
        session: The current CLI session.
        name: The name of the agent to load.
    """
    if not name:
        print_error(
            "Missing agent name",
            "AGENT_NOT_FOUND",
            "Usage: /load <agent_name>\nUse /list to see available agents."
        )
        return

    # Load configuration
    try:
        config_dict = load_agent_config_dict(name)
    except json.JSONDecodeError as e:
        handle_error(e, context=f"Parsing agent config for '{name}'")
        print(dim("The config file may be corrupted. Try /rollback if a backup exists."))
        return
    except Exception as e:
        handle_error(e, context=f"Loading agent config for '{name}'")
        return

    if config_dict is None:
        print_error(f"Agent '{name}' not found", "AGENT_NOT_FOUND")
        return

    # Save current agent's history before switching
    if session.active_agent_name and session.manager:
        try:
            session.manager.save_agent_history(session.active_agent_name, session.agent_messages)
            print(dim(f"Saved history for '{session.active_agent_name}'."))
        except Exception as e:
            print_warning(f"Could not save history for '{session.active_agent_name}': {e}")

    print(info(f"Loading agent '{name}'..."))

    try:
        # Import here to avoid circular imports
        from .mcp_integration import (
            MCPConnectionError,
            MCPCredentialError,
            MCPTimeoutError,
            create_agent_with_tools,
        )
        from .schemas import AgentConfig

        # Convert dict to AgentConfig
        # Remove metadata fields that aren't part of AgentConfig
        config_copy = {k: v for k, v in config_dict.items() if not k.startswith("_")}
        config = AgentConfig(**config_copy)

        # Create agent with MCP tools (with graceful degradation)
        # Use retry for transient connection failures
        async def create_agent():
            return await create_agent_with_tools(config, skip_unavailable=True)

        try:
            agent = await with_retry(create_agent, max_retries=2, delay=1.5)
        except (MCPConnectionError, MCPTimeoutError) as e:
            # Graceful degradation: warn but continue without MCP tools
            print_warning(f"Some MCP tools unavailable: {e}")
            print(dim("Agent will run with reduced capabilities."))
            agent = await create_agent_with_tools(config, skip_unavailable=True)

        # Switch session to agent mode (this handles history loading)
        session.switch_to_agent(agent, name)

        print(success(f"Agent '{name}' loaded and activated!"))
        print()
        print(f"Description: {config.description}")
        print(f"Tools: {', '.join(config.tools) if config.tools else 'none'}")
        print(f"Stratum: {config.stratum or 'Not classified'}")

        # Show history status
        history_count = len(session.agent_messages)
        if history_count > 0:
            print(f"History: {history_count} messages restored")
        print()
        print(dim("You can now chat with this agent. Use /builder to switch back."))

    except MCPCredentialError as e:
        handle_error(e, context=f"Loading agent '{name}'")
        print(dim("Configure missing credentials in .env and try again."))
    except Exception as e:
        handle_error(e, context=f"Loading agent '{name}'")


async def cmd_run(session: Session, args: str) -> None:
    """Handle /run command - run a specific agent with a task.

    Args:
        session: The current CLI session.
        args: The agent name and task (space-separated).
    """
    parts = args.split(maxsplit=1)
    if len(parts) < 2:
        print_error(
            "Missing agent name or task",
            "AGENT_EXECUTION_FAILED",
            "Usage: /run <agent_name> <task>\n"
            "Example: /run lead_qualifier Qualify this lead: John from Acme Corp"
        )
        return

    name, task = parts

    # Load agent configuration
    try:
        config_dict = load_agent_config_dict(name)
    except json.JSONDecodeError as e:
        handle_error(e, context=f"Parsing agent config for '{name}'")
        return
    except Exception as e:
        handle_error(e, context=f"Loading agent config for '{name}'")
        return

    if config_dict is None:
        print_error(f"Agent '{name}' not found", "AGENT_NOT_FOUND")
        return

    print(info(f"Running agent '{name}'..."))

    try:
        from .mcp_integration import (
            MCPConnectionError,
            MCPCredentialError,
            MCPTimeoutError,
            create_agent_with_tools,
        )
        from .schemas import AgentConfig

        config_copy = {k: v for k, v in config_dict.items() if not k.startswith("_")}
        config = AgentConfig(**config_copy)

        # Create agent with graceful degradation
        async def create_agent():
            return await create_agent_with_tools(config, skip_unavailable=True)

        try:
            agent = await with_retry(create_agent, max_retries=2, delay=1.5)
        except (MCPConnectionError, MCPTimeoutError) as e:
            print_warning(f"Some MCP tools unavailable: {e}")
            print(dim("Running with reduced capabilities."))
            agent = await create_agent_with_tools(config, skip_unavailable=True)

        # Run the task with retry for transient failures
        async def run_task():
            return agent.invoke({
                "messages": [HumanMessage(content=task)]
            })

        result = await with_retry(run_task, max_retries=2, delay=1.0)

        # Extract response
        ai_messages = [
            msg for msg in result.get("messages", [])
            if hasattr(msg, "type") and msg.type == "ai"
        ]

        if ai_messages:
            response = extract_text_from_content(ai_messages[-1].content)
            print()
            print(f"{Colors.BRIGHT_GREEN}{name}:{Colors.RESET} {response}")
        else:
            print_warning("No response from agent. The task may have failed silently.")

    except MCPCredentialError as e:
        handle_error(e, context=f"Running agent '{name}'")
        print(dim("Configure missing credentials in .env and try again."))
    except Exception as e:
        handle_error(e, context=f"Running agent '{name}'")

    print()


def _get_backup_path(name: str) -> Path:
    """Get the backup file path for an agent.

    Args:
        name: The agent name.

    Returns:
        Path to the backup file.
    """
    return OUTPUT_DIR / f"{name}.backup.json"


def _backup_agent_config(name: str) -> bool:
    """Create a backup of an agent's configuration.

    Args:
        name: The agent name to backup.

    Returns:
        True if backup was successful, False otherwise.
    """
    config_path = get_agent_config_path(name)
    backup_path = _get_backup_path(name)

    if not config_path.exists():
        return False

    try:
        # Read current config
        with open(config_path, "r", encoding="utf-8") as f:
            config_data = json.load(f)

        # Add backup metadata
        config_data["_backup_metadata"] = {
            "backed_up_at": datetime.now().isoformat(),
            "original_path": str(config_path),
        }

        # Write backup
        with open(backup_path, "w", encoding="utf-8") as f:
            json.dump(config_data, f, indent=2, ensure_ascii=False)

        return True

    except (json.JSONDecodeError, OSError) as e:
        logger.error(f"Failed to backup agent config {name}: {e}")
        return False


def _show_config_diff(old_config: dict, new_config: dict, name: str) -> None:
    """Display a colored diff between old and new configurations.

    Args:
        old_config: The original configuration dictionary.
        new_config: The new configuration dictionary.
        name: The agent name.
    """
    print()
    print(highlight(f"Changes to agent: {name}"))
    print("=" * 60)
    print()

    # Fields to compare (excluding metadata)
    compare_fields = ["description", "system_prompt", "tools", "stratum", "model"]

    changes_found = False

    for field in compare_fields:
        old_value = old_config.get(field)
        new_value = new_config.get(field)

        if old_value != new_value:
            changes_found = True
            print(f"{Colors.BOLD}{field}:{Colors.RESET}")

            if field == "tools":
                # Special handling for tools list
                old_tools = set(old_value or [])
                new_tools = set(new_value or [])
                removed = old_tools - new_tools
                added = new_tools - old_tools
                unchanged = old_tools & new_tools

                if removed:
                    for tool in removed:
                        print(f"  {Colors.RED}- {tool}{Colors.RESET}")
                if unchanged:
                    for tool in unchanged:
                        print(f"  {Colors.DIM}  {tool}{Colors.RESET}")
                if added:
                    for tool in added:
                        print(f"  {Colors.GREEN}+ {tool}{Colors.RESET}")

            elif field == "system_prompt":
                # Truncate long prompts for display
                old_display = (old_value or "")[:100]
                new_display = (new_value or "")[:100]
                if len(old_value or "") > 100:
                    old_display += "..."
                if len(new_value or "") > 100:
                    new_display += "..."

                print(f"  {Colors.RED}- {old_display}{Colors.RESET}")
                print(f"  {Colors.GREEN}+ {new_display}{Colors.RESET}")

            else:
                # Simple field comparison
                print(f"  {Colors.RED}- {old_value}{Colors.RESET}")
                print(f"  {Colors.GREEN}+ {new_value}{Colors.RESET}")

            print()

    # Show version bump
    old_version = old_config.get("metadata", {}).get("version", 1) if isinstance(old_config.get("metadata"), dict) else 1
    new_version = old_version + 1
    print(f"{Colors.CYAN}Version: {old_version} -> {new_version}{Colors.RESET}")
    print()

    if not changes_found:
        print(dim("No changes detected."))
        print()


def _show_config_summary(config_dict: dict, name: str) -> None:
    """Display a summary of an agent's current configuration.

    Args:
        config_dict: The configuration dictionary.
        name: The agent name.
    """
    print()
    print(highlight(f"Current configuration: {name}"))
    print("=" * 60)
    print()
    print(f"  {Colors.CYAN}Description:{Colors.RESET}")
    print(f"    {config_dict.get('description', 'N/A')}")
    print()
    print(f"  {Colors.CYAN}Tools:{Colors.RESET}")
    tools = config_dict.get('tools', [])
    if tools:
        for tool in tools:
            print(f"    - {tool}")
    else:
        print(f"    {dim('none')}")
    print()
    print(f"  {Colors.CYAN}Stratum:{Colors.RESET} {config_dict.get('stratum', 'Not set')}")
    print(f"  {Colors.CYAN}Model:{Colors.RESET} {config_dict.get('model', 'default')}")

    # Show metadata if available
    metadata = config_dict.get("metadata", {})
    if isinstance(metadata, dict) and metadata:
        print()
        print(f"  {Colors.CYAN}Metadata:{Colors.RESET}")
        if metadata.get("version"):
            print(f"    Version: {metadata.get('version')}")
        if metadata.get("execution_count"):
            print(f"    Executions: {metadata.get('execution_count')}")
        if metadata.get("tags"):
            print(f"    Tags: {', '.join(metadata.get('tags', []))}")

    print()


async def _interpret_modification_with_builder(
    session: Session,
    agent_name: str,
    current_config: dict,
    instruction: str
) -> Optional[dict]:
    """Use the builder to interpret natural language modification instructions.

    Args:
        session: The current CLI session.
        agent_name: Name of the agent being modified.
        current_config: The current agent configuration.
        instruction: Natural language modification instruction.

    Returns:
        Dictionary with proposed changes, or None if interpretation failed.
    """
    if session.builder is None:
        return None

    # Create a focused prompt for the builder
    interpret_prompt = f"""I need to modify an existing agent. Please interpret the following modification request and return ONLY the fields that should change.

Current agent: {agent_name}
Current description: {current_config.get('description', 'N/A')}
Current tools: {', '.join(current_config.get('tools', []))}
Current stratum: {current_config.get('stratum', 'Not set')}

Modification request: "{instruction}"

Based on this request, what specific changes should be made? Please respond with a JSON object containing ONLY the fields that should change. Valid fields are:
- "description": new description string
- "tools": new list of tool names (valid: voice, websearch, calendar, communication, crm)
- "stratum": new stratum (valid: RTI, RAI, ZACS, EEI, IGE)
- "system_prompt": new system prompt (only if behavior change is requested)

Respond with ONLY the JSON object, no other text."""

    try:
        # Use the builder to interpret
        result = session.builder.invoke({
            "messages": [HumanMessage(content=interpret_prompt)]
        })

        # Extract response
        ai_messages = [
            msg for msg in result.get("messages", [])
            if hasattr(msg, "type") and msg.type == "ai"
        ]

        if ai_messages:
            response = extract_text_from_content(ai_messages[-1].content)

            # Try to extract JSON from response
            import re
            json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
            if json_match:
                changes = json.loads(json_match.group())
                return changes

    except Exception as e:
        logger.warning(f"Failed to interpret modification: {e}")

    return None


async def _interactive_modify_mode(session: Session, name: str, config_dict: dict) -> None:
    """Enter interactive modification mode for an agent.

    Args:
        session: The current CLI session.
        name: The agent name.
        config_dict: The current configuration dictionary.
    """
    print()
    print(f"{Colors.BRIGHT_CYAN}Entering modification mode for '{name}'{Colors.RESET}")
    print(dim("Describe changes in natural language, or use direct commands."))
    print(dim("Type 'done' to finish, 'cancel' to abort, 'show' to see current config."))
    print()

    # Keep track of accumulated changes
    pending_changes: dict = {}
    original_config = config_dict.copy()

    while True:
        try:
            user_input = input(f"{Colors.YELLOW}modify>{Colors.RESET} ").strip()
        except (KeyboardInterrupt, EOFError):
            print()
            print(warning("Modification cancelled."))
            return

        if not user_input:
            continue

        lower_input = user_input.lower()

        if lower_input == "cancel":
            print(warning("Modification cancelled."))
            return

        elif lower_input == "done":
            if not pending_changes:
                print(warning("No changes to apply."))
                return

            # Show final diff and confirm
            merged_config = {**config_dict, **pending_changes}
            _show_config_diff(original_config, merged_config, name)

            confirm = input(f"Apply these changes? [{Colors.GREEN}y{Colors.RESET}/{Colors.RED}N{Colors.RESET}]: ").strip().lower()
            if confirm == "y":
                # Create backup
                if _backup_agent_config(name):
                    print(dim(f"Backup saved to {_get_backup_path(name)}"))

                # Apply changes
                config_dict.update(pending_changes)

                # Increment version
                if "metadata" not in config_dict or not isinstance(config_dict.get("metadata"), dict):
                    config_dict["metadata"] = {}
                config_dict["metadata"]["version"] = config_dict["metadata"].get("version", 1) + 1
                config_dict["metadata"]["updated_at"] = datetime.now().isoformat()

                if save_agent_config_dict(name, config_dict):
                    print(success(f"Agent '{name}' updated successfully!"))
                else:
                    print(error("Failed to save configuration."))
            else:
                print(warning("Changes discarded."))
            return

        elif lower_input == "show":
            merged_config = {**config_dict, **pending_changes}
            _show_config_summary(merged_config, name)
            if pending_changes:
                print(f"{Colors.YELLOW}Pending changes: {list(pending_changes.keys())}{Colors.RESET}")
            continue

        elif lower_input.startswith("set "):
            # Direct field setting: set description New description here
            parts = user_input[4:].split(maxsplit=1)
            if len(parts) >= 2:
                field, value = parts
                field = field.lower()

                if field == "description":
                    pending_changes["description"] = value
                    print(success(f"Queued: description = '{value[:50]}...'"))
                elif field == "stratum":
                    value = value.upper()
                    if value in ["RTI", "RAI", "ZACS", "EEI", "IGE"]:
                        pending_changes["stratum"] = value
                        print(success(f"Queued: stratum = {value}"))
                    else:
                        print(error("Invalid stratum. Valid: RTI, RAI, ZACS, EEI, IGE"))
                elif field == "tools":
                    tools = [t.strip() for t in value.split(",")]
                    valid_tools = ["voice", "websearch", "calendar", "communication", "crm"]
                    invalid = [t for t in tools if t and t not in valid_tools]
                    if invalid:
                        print(error(f"Invalid tools: {invalid}. Valid: {valid_tools}"))
                    else:
                        pending_changes["tools"] = [t for t in tools if t]
                        print(success(f"Queued: tools = {pending_changes['tools']}"))
                else:
                    print(error(f"Unknown field: {field}. Valid: description, stratum, tools"))
            else:
                print(error("Usage: set <field> <value>"))
            continue

        elif lower_input.startswith("add tool "):
            tool = user_input[9:].strip().lower()
            valid_tools = ["voice", "websearch", "calendar", "communication", "crm"]
            if tool in valid_tools:
                current_tools = pending_changes.get("tools", config_dict.get("tools", []))[:]
                if tool not in current_tools:
                    current_tools.append(tool)
                    pending_changes["tools"] = current_tools
                    print(success(f"Queued: add tool '{tool}'"))
                else:
                    print(warning(f"Tool '{tool}' already present."))
            else:
                print(error(f"Invalid tool: {tool}. Valid: {valid_tools}"))
            continue

        elif lower_input.startswith("remove tool "):
            tool = user_input[12:].strip().lower()
            current_tools = pending_changes.get("tools", config_dict.get("tools", []))[:]
            if tool in current_tools:
                current_tools.remove(tool)
                pending_changes["tools"] = current_tools
                print(success(f"Queued: remove tool '{tool}'"))
            else:
                print(warning(f"Tool '{tool}' not present."))
            continue

        else:
            # Try to interpret as natural language
            print(dim("Interpreting..."))
            changes = await _interpret_modification_with_builder(
                session, name, {**config_dict, **pending_changes}, user_input
            )

            if changes:
                # Show what the builder interpreted
                print()
                print(f"{Colors.CYAN}Interpreted changes:{Colors.RESET}")
                for key, value in changes.items():
                    if key in ["description", "system_prompt", "tools", "stratum"]:
                        display_value = value
                        if isinstance(value, str) and len(value) > 50:
                            display_value = value[:50] + "..."
                        print(f"  {key}: {display_value}")
                        pending_changes[key] = value
                print()

                confirm = input(f"Accept these changes? [{Colors.GREEN}y{Colors.RESET}/{Colors.RED}N{Colors.RESET}]: ").strip().lower()
                if confirm != "y":
                    # Remove the just-added changes
                    for key in changes.keys():
                        pending_changes.pop(key, None)
                    print(dim("Changes rejected."))
            else:
                print(warning("Could not interpret that request. Try using 'set <field> <value>' or be more specific."))


async def cmd_modify(session: Session, args: str) -> None:
    """Handle /modify command - modify an existing agent's config.

    Supports two modes:
    1. Natural language: /modify my_agent make it friendlier and add websearch
    2. Interactive: /modify my_agent (enters modification mode)

    Args:
        session: The current CLI session.
        args: The agent name and optional modification instruction.
    """
    if not args:
        print_error(
            "Missing agent name",
            "AGENT_NOT_FOUND",
            "Usage: /modify <agent_name> [modification instruction]\n"
            "Example: /modify my_agent make it friendlier and add websearch\n"
            "         /modify my_agent  (enters interactive mode)"
        )
        return

    # Parse agent name and optional instruction
    parts = args.split(maxsplit=1)
    name = parts[0]
    instruction = parts[1] if len(parts) > 1 else None

    # Load current config
    try:
        config_dict = load_agent_config_dict(name)
    except json.JSONDecodeError as e:
        handle_error(e, context=f"Parsing agent config for '{name}'")
        print(dim("The config file may be corrupted. Try /rollback if a backup exists."))
        return
    except Exception as e:
        handle_error(e, context=f"Loading agent config for '{name}'")
        return

    if config_dict is None:
        print_error(f"Agent '{name}' not found", "AGENT_NOT_FOUND")
        return

    # Show current config summary
    _show_config_summary(config_dict, name)

    if instruction:
        # Natural language modification mode
        print(info("Interpreting modification request..."))
        print()

        changes = await _interpret_modification_with_builder(session, name, config_dict, instruction)

        if not changes:
            print(warning("Could not interpret the modification request."))
            print(dim("Try being more specific, or use interactive mode: /modify " + name))
            return

        # Build proposed config
        proposed_config = config_dict.copy()
        for key, value in changes.items():
            if key in ["description", "system_prompt", "tools", "stratum"]:
                proposed_config[key] = value

        # Show diff
        _show_config_diff(config_dict, proposed_config, name)

        # Confirm
        try:
            confirm = input(f"Apply these changes? [{Colors.GREEN}y{Colors.RESET}/{Colors.RED}N{Colors.RESET}]: ").strip().lower()
            if confirm == "y":
                # Create backup
                if _backup_agent_config(name):
                    print(dim(f"Backup saved to {_get_backup_path(name)}"))

                # Apply changes
                config_dict.update(changes)

                # Increment version
                if "metadata" not in config_dict or not isinstance(config_dict.get("metadata"), dict):
                    config_dict["metadata"] = {}
                config_dict["metadata"]["version"] = config_dict["metadata"].get("version", 1) + 1
                config_dict["metadata"]["updated_at"] = datetime.now().isoformat()

                if save_agent_config_dict(name, config_dict):
                    print()
                    print(success(f"Agent '{name}' updated successfully!"))
                else:
                    print(error("Failed to save configuration."))
            else:
                print(warning("Modification cancelled."))

        except (KeyboardInterrupt, EOFError):
            print()
            print(warning("Modification cancelled."))

    else:
        # Interactive modification mode
        await _interactive_modify_mode(session, name, config_dict)


async def cmd_rollback(session: Session, name: str) -> None:
    """Handle /rollback command - restore an agent from backup.

    Args:
        session: The current CLI session.
        name: The name of the agent to rollback.
    """
    if not name:
        print_error(
            "Missing agent name",
            "AGENT_NOT_FOUND",
            "Usage: /rollback <agent_name>"
        )
        return

    backup_path = _get_backup_path(name)
    config_path = get_agent_config_path(name)

    if not backup_path.exists():
        print_error(
            f"No backup found for agent '{name}'",
            "FILE_NOT_FOUND",
            "Backups are created automatically when using /modify."
        )
        return

    try:
        # Load backup
        with open(backup_path, "r", encoding="utf-8") as f:
            backup_data = json.load(f)

        # Show backup info
        backup_metadata = backup_data.pop("_backup_metadata", {})
        backed_up_at = backup_metadata.get("backed_up_at", "Unknown")

        print(highlight(f"Rollback agent: {name}"))
        print("=" * 50)
        print()
        print(f"Backup created: {backed_up_at}")
        print()

        # Show what will be restored
        _show_config_summary(backup_data, name)

        # Load current config for comparison
        try:
            current_config = load_agent_config_dict(name)
        except Exception:
            current_config = None

        if current_config:
            print(f"{Colors.YELLOW}This will revert the following changes:{Colors.RESET}")
            _show_config_diff(current_config, backup_data, name)

        # Confirm
        confirm = input(f"Restore from backup? [{Colors.GREEN}y{Colors.RESET}/{Colors.RED}N{Colors.RESET}]: ").strip().lower()
        if confirm == "y":
            # Save backup data as current config
            if save_agent_config_dict(name, backup_data):
                print(success(f"Agent '{name}' restored from backup!"))

                # Remove backup after successful restore
                try:
                    backup_path.unlink()
                    print(dim("Backup file removed."))
                except OSError as e:
                    print_warning(f"Could not remove backup file: {e}")
            else:
                print_error(
                    "Failed to restore from backup",
                    "FILE_PERMISSION_ERROR",
                    "Check file permissions and try again."
                )
        else:
            print(dim("Rollback cancelled."))

    except json.JSONDecodeError as e:
        handle_error(e, context=f"Reading backup for '{name}'")
        print(dim("The backup file may be corrupted."))
    except PermissionError as e:
        handle_error(e, context=f"Accessing backup for '{name}'")
    except OSError as e:
        handle_error(e, context=f"Reading backup for '{name}'")


async def cmd_delete(session: Session, name: str) -> None:
    """Handle /delete command - delete an agent.

    Args:
        session: The current CLI session.
        name: The name of the agent to delete.
    """
    if not name:
        print_error(
            "Missing agent name",
            "AGENT_NOT_FOUND",
            "Usage: /delete <agent_name>"
        )
        return

    try:
        config_dict = load_agent_config_dict(name)
    except Exception as e:
        handle_error(e, context=f"Loading agent config for '{name}'")
        return

    if config_dict is None:
        print_error(f"Agent '{name}' not found", "AGENT_NOT_FOUND")
        return

    print_warning(f"Are you sure you want to delete agent '{name}'?")
    print(f"  Description: {config_dict.get('description', 'N/A')}")

    try:
        confirm = input("Type 'yes' to confirm: ").strip().lower()
        if confirm == "yes":
            try:
                if delete_agent_config_file(name):
                    print(success(f"Agent '{name}' deleted."))
                    # Clear if this was the active agent
                    if session.active_agent_name == name:
                        session.clear_agent()
                        print(dim("Switched back to builder mode."))

                    # Also try to remove conversation history
                    if session.manager:
                        try:
                            session.manager.clear_agent_history(name)
                        except Exception:
                            pass  # History cleanup is optional
                else:
                    print_error(
                        f"Failed to delete agent '{name}'",
                        "FILE_PERMISSION_ERROR",
                        "Check file permissions and try again."
                    )
            except PermissionError as e:
                handle_error(e, context=f"Deleting agent '{name}'")
            except OSError as e:
                handle_error(e, context=f"Deleting agent '{name}'")
        else:
            print(dim("Deletion cancelled."))
    except (KeyboardInterrupt, EOFError):
        print()
        print(dim("Deletion cancelled."))


async def cmd_status(session: Session) -> None:
    """Handle /status command - show current agent and MCP tool status."""
    from .tool_registry import format_tool_status_report, get_configured_tools

    print(highlight("Session Status"))
    print("=" * 50)
    print()

    # Session metadata
    print(f"{Colors.BRIGHT_CYAN}Session Info{Colors.RESET}")
    print("-" * 30)
    if session.manager:
        print(session.manager.get_session_summary(session))
    else:
        print(f"Session ID: {session.metadata.session_id[:8]}...")
        print(f"Total Messages: {session.metadata.total_messages}")

    print()

    # Current mode and agent
    print(f"{Colors.BRIGHT_CYAN}Current State{Colors.RESET}")
    print("-" * 30)
    print(f"Mode: {Colors.BRIGHT_CYAN}{session.mode.value}{Colors.RESET}")

    # Active agent
    if session.active_agent_name:
        print(f"Active Agent: {Colors.GREEN}{session.active_agent_name}{Colors.RESET}")
    else:
        print(f"Active Agent: {dim('None (using builder)')}")

    # Message counts
    print(f"Builder History: {len(session.builder_messages)} messages")
    print(f"Agent History: {len(session.agent_messages)} messages")

    # Streaming status
    streaming_mode = session.streaming.mode.value
    streaming_status = "enabled" if session.streaming.enabled else "disabled"
    print(f"Streaming: {streaming_mode} ({streaming_status})")

    # Session persistence info
    print()
    print(f"{Colors.BRIGHT_CYAN}Persistence{Colors.RESET}")
    print("-" * 30)
    print(f"Session File: {dim(str(SESSIONS_DIR / 'current.json'))}")
    print(f"History Dir: {dim(str(HISTORY_DIR))}")

    print()

    # MCP Tools Status
    print(highlight("MCP Tools"))
    print("-" * 50)
    print(format_tool_status_report())


async def cmd_tools(session: Session) -> None:
    """Handle /tools command - list available MCP tools."""
    from .tool_registry import get_tools_status

    print(highlight("Available MCP Tools"))
    print("=" * 50)
    print()

    status = get_tools_status()

    for tool_name in sorted(status.keys()):
        info_dict = status[tool_name]
        is_available = info_dict["available"]

        # Color based on availability
        if is_available:
            status_str = success("READY")
        else:
            status_str = error("NOT CONFIGURED")

        print(f"  {Colors.BOLD}{tool_name}{Colors.RESET}: {status_str}")
        print(f"    {dim(info_dict['description'])}")

        if info_dict["capabilities"]:
            caps = ", ".join(info_dict["capabilities"][:4])
            print(f"    Capabilities: {caps}")

        if not is_available and info_dict["missing_credentials"]:
            missing = ", ".join(info_dict["missing_credentials"])
            print(f"    {warning('Missing: ' + missing)}")

        print()


async def cmd_builder(session: Session) -> None:
    """Handle /builder command - switch back to builder mode."""
    session.switch_to_builder()
    print(success("Switched to builder mode."))
    print(dim("You can now create new agents or chat with the builder."))


async def cmd_debug(session: Session, args: str) -> None:
    """Handle /debug command - show recent errors and debug information.

    Args:
        session: The current CLI session.
        args: Optional arguments (count, clear, path).
    """
    args = args.strip().lower()

    if args == "clear":
        # Clear error log
        if error_logger.clear_log():
            print(success("Error log cleared."))
        else:
            print_error("Failed to clear error log", "FILE_PERMISSION_ERROR")
        return

    if args == "path":
        # Show log file path
        log_path = error_logger.get_log_path()
        print(f"Error log file: {log_path}")
        if log_path.exists():
            size = log_path.stat().st_size
            print(f"Size: {size} bytes")
        else:
            print(dim("Log file does not exist yet (no errors logged)."))
        return

    # Parse count argument
    count = 10  # default
    if args:
        try:
            count = int(args)
            count = max(1, min(count, 50))  # Clamp between 1 and 50
        except ValueError:
            print_error(
                f"Invalid argument: {args}",
                suggestion="Usage: /debug [count|clear|path]"
            )
            return

    # Show recent errors
    recent_errors = error_logger.get_recent_errors(count)

    print(highlight("Recent Errors"))
    print("=" * 60)
    print()

    if not recent_errors:
        print(dim("No recent errors recorded."))
        print()
        print(dim("Error logs are stored at: " + str(error_logger.get_log_path())))
        return

    for i, err in enumerate(recent_errors, 1):
        timestamp = err.get("timestamp", "Unknown")
        error_code = err.get("error_code", "UNKNOWN")
        error_type = err.get("error_type", "Exception")
        message = err.get("message", "No message")
        context = err.get("context", {})

        # Parse timestamp for display
        try:
            dt = datetime.fromisoformat(timestamp)
            time_str = dt.strftime("%Y-%m-%d %H:%M:%S")
        except ValueError:
            time_str = timestamp

        print(f"{Colors.BRIGHT_RED}[{i}]{Colors.RESET} {error_type}")
        print(f"    {Colors.DIM}Time:{Colors.RESET} {time_str}")
        print(f"    {Colors.DIM}Code:{Colors.RESET} {error_code}")
        print(f"    {Colors.DIM}Message:{Colors.RESET} {message[:100]}{'...' if len(message) > 100 else ''}")

        if context.get("operation"):
            print(f"    {Colors.DIM}Context:{Colors.RESET} {context['operation']}")

        # Show suggestion if available
        suggestion = ERROR_SUGGESTIONS.get(error_code)
        if suggestion:
            print(f"    {Colors.YELLOW}Suggestion:{Colors.RESET} {suggestion}")

        print()

    print("-" * 60)
    print(f"Showing {len(recent_errors)} of {len(error_logger._recent_errors)} recent errors")
    print(dim(f"Full logs: {error_logger.get_log_path()}"))
    print()
    print(dim("Commands: /debug [N] - show N errors, /debug clear - clear log, /debug path - show log path"))


def cmd_help() -> None:
    """Handle /help command - show available commands."""
    print(highlight("ACTi Agent Builder - Commands"))
    print("=" * 50)
    print()

    commands = [
        ("(no prefix)", "Chat with active agent or builder"),
        ("/create <prompt>", "Create a new agent from natural language"),
        ("/list", "List all saved agents"),
        ("/load <name>", "Load and activate an existing agent"),
        ("/run <name> <task>", "Run a specific agent with a task"),
        ("/modify <name> [instr]", "Modify an agent (with instruction or interactive)"),
        ("/rollback <name>", "Restore agent from backup"),
        ("/delete <name>", "Delete an agent"),
        ("/status", "Show current agent and MCP tool status"),
        ("/tools", "List available MCP tools"),
        ("/builder", "Switch back to builder mode"),
        ("/stream [mode]", "Configure streaming (on/off/verbose)"),
        ("/debug [N|clear|path]", "Show recent errors and debug info"),
        ("/help", "Show this help message"),
        ("/exit", "Exit the CLI"),
    ]

    for cmd, desc in commands:
        print(f"  {Colors.CYAN}{cmd:<20}{Colors.RESET} {desc}")

    print()
    print(dim("Tip: Use Tab for command completion, Up/Down for history."))


# =============================================================================
# Streaming Output Helpers
# =============================================================================

# Spinner characters for progress indication
SPINNER_CHARS = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]


class SpinnerTask:
    """Async spinner task that displays while waiting for first token.

    Usage:
        spinner = SpinnerTask("Thinking")
        spinner.start()
        # ... do async work ...
        spinner.stop()
    """

    def __init__(self, message: str = "Thinking") -> None:
        self._message = message
        self._task: Optional[asyncio.Task] = None
        self._running = False

    async def _spin(self) -> None:
        """Internal spinning loop."""
        idx = 0
        while self._running:
            char = SPINNER_CHARS[idx % len(SPINNER_CHARS)]
            # Use carriage return to overwrite spinner
            sys.stdout.write(f"\r{Colors.DIM}{char} {self._message}...{Colors.RESET}")
            sys.stdout.flush()
            idx += 1
            await asyncio.sleep(0.1)

    def start(self) -> None:
        """Start the spinner in the background."""
        if not sys.stdout.isatty():
            return  # Skip spinner for non-TTY
        self._running = True
        self._task = asyncio.create_task(self._spin())

    def stop(self) -> None:
        """Stop the spinner and clear the line."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                # Suppress the CancelledError
                pass
            except asyncio.CancelledError:
                pass
            self._task = None
        # Clear the spinner line
        if sys.stdout.isatty():
            sys.stdout.write("\r" + " " * 40 + "\r")
            sys.stdout.flush()


async def _stream_agent_response(
    agent: "CompiledStateGraph",
    messages: list,
    config: StreamingConfig,
    agent_label: str,
    label_color: str,
) -> tuple[str, list]:
    """Stream agent response with real-time token output.

    Args:
        agent: The agent (builder or loaded agent) to invoke.
        messages: The message history to send.
        config: Streaming configuration.
        agent_label: Label to display (e.g., "Builder", agent name).
        label_color: ANSI color code for the label.

    Returns:
        Tuple of (full response text, updated messages list).
    """
    # Print the agent label
    print()
    print(f"{label_color}{agent_label}:{Colors.RESET} ", end="", flush=True)

    # Start spinner if enabled
    spinner = None
    if config.spinner:
        spinner = SpinnerTask("Thinking")
        spinner.start()

    full_response = ""
    first_token = True
    updated_messages = messages.copy()
    active_tools: set = set()

    # Validate and patch messages before sending to API
    # This ensures no dangling tool_use blocks without corresponding tool_results
    patched_messages = validate_and_patch_messages(messages)

    try:
        async for event in agent.astream_events(
            {"messages": patched_messages},
            version="v2",
            config={"recursion_limit": 100},  # Fix: Propagate recursion limit to streaming API
        ):
            event_type = event.get("event", "")
            event_name = event.get("name", "")
            event_data = event.get("data", {})

            # Handle chat model streaming
            if event_type == "on_chat_model_stream":
                # Stop spinner on first token
                if first_token and spinner:
                    spinner.stop()
                    # Move cursor back to overwrite the label line continuation
                    first_token = False

                chunk = event_data.get("chunk")
                if chunk and hasattr(chunk, "content"):
                    content = chunk.content
                    if content:
                        # Normalize Anthropic content blocks to string
                        content = extract_text_from_content(content)
                        if content:
                            print(content, end="", flush=True)
                            full_response += content

            # Handle tool start events
            elif event_type == "on_tool_start" and config.show_tool_calls:
                tool_name = event_name
                if tool_name and tool_name not in active_tools:
                    active_tools.add(tool_name)
                    # Stop spinner if still running
                    if spinner and first_token:
                        spinner.stop()
                        first_token = False
                    if config.mode == StreamingMode.VERBOSE:
                        tool_input = event_data.get("input", {})
                        print(f"\n{Colors.DIM}[Calling {tool_name}: {tool_input}]{Colors.RESET}", end="", flush=True)
                    else:
                        print(f"\n{Colors.DIM}[Calling {tool_name}...]{Colors.RESET}", end="", flush=True)

            # Handle tool end events
            elif event_type == "on_tool_end" and config.show_tool_calls:
                tool_name = event_name
                if tool_name in active_tools:
                    active_tools.discard(tool_name)
                    if config.mode == StreamingMode.VERBOSE:
                        output = event_data.get("output", "")
                        # Truncate long outputs
                        if isinstance(output, str) and len(output) > 200:
                            output = output[:200] + "..."
                        print(f"\n{Colors.DIM}[{tool_name} returned: {output}]{Colors.RESET}", end="", flush=True)
                    else:
                        print(f"\n{Colors.DIM}[{tool_name} complete]{Colors.RESET}", end="", flush=True)

            # Handle chain/graph end to get final messages
            elif event_type == "on_chain_end" and event_name == "LangGraph":
                output = event_data.get("output", {})
                if isinstance(output, dict) and "messages" in output:
                    updated_messages = output["messages"]

    except Exception as e:
        if spinner:
            spinner.stop()
        raise e
    finally:
        if spinner:
            spinner.stop()

    # Ensure newline at end
    print()

    # If streaming didn't capture messages, fall back to final state
    if not full_response and updated_messages == messages:
        # Fall back: try to extract from final messages
        ai_messages = [
            msg for msg in updated_messages
            if hasattr(msg, "type") and msg.type == "ai"
        ]
        if ai_messages:
            full_response = extract_text_from_content(ai_messages[-1].content)

    return full_response, updated_messages


async def _batch_agent_response(
    agent: "CompiledStateGraph",
    messages: list,
    agent_label: str,
    label_color: str,
) -> tuple[str, list]:
    """Get agent response in batch mode (no streaming).

    Args:
        agent: The agent to invoke.
        messages: The message history.
        agent_label: Label to display.
        label_color: ANSI color for label.

    Returns:
        Tuple of (response text, updated messages).
    """
    # Validate and patch messages before sending to API
    # This ensures no dangling tool_use blocks without corresponding tool_results
    patched_messages = validate_and_patch_messages(messages)

    result = await agent.ainvoke({"messages": patched_messages})

    ai_messages = [
        msg for msg in result.get("messages", [])
        if hasattr(msg, "type") and msg.type == "ai"
    ]

    response = ""
    if ai_messages:
        response = extract_text_from_content(ai_messages[-1].content)
        print()
        print(f"{label_color}{agent_label}:{Colors.RESET} {response}")
    else:
        print(warning("[No response]"))

    return response, result.get("messages", messages)


# =============================================================================
# Stream Command Handler
# =============================================================================

async def cmd_stream(session: Session, args: str) -> None:
    """Handle /stream command - configure streaming output.

    Args:
        session: The current CLI session.
        args: The streaming mode argument (on, off, verbose).
    """
    args = args.strip().lower()

    if not args:
        # Show current status
        mode = session.streaming.mode.value
        enabled = "enabled" if session.streaming.enabled else "disabled"
        print(f"Streaming: {highlight(mode)} ({enabled})")
        print()
        print(dim("Usage: /stream <on|off|verbose>"))
        print(dim("  on      - Enable streaming with tool indicators (default)"))
        print(dim("  off     - Disable streaming (batch mode)"))
        print(dim("  verbose - Show all tool details"))
        return

    if args == "on":
        session.streaming.set_mode(StreamingMode.ON)
        print(success("Streaming enabled."))
    elif args == "off":
        session.streaming.set_mode(StreamingMode.OFF)
        print(success("Streaming disabled (batch mode)."))
    elif args == "verbose":
        session.streaming.set_mode(StreamingMode.VERBOSE)
        print(success("Streaming enabled with verbose tool output."))
    else:
        print(error(f"Unknown mode: {args}"))
        print(dim("Valid options: on, off, verbose"))


# =============================================================================
# Main Chat Loop
# =============================================================================

async def handle_chat_message(session: Session, message: str) -> None:
    """Handle a regular chat message (no command prefix).

    Uses streaming output when enabled, falling back to batch mode otherwise.
    Displays a spinner while waiting for the first token in streaming mode.

    Args:
        session: The current CLI session.
        message: The user's message.
    """
    if session.mode == SessionMode.BUILDER:
        # Chat with builder
        if session.builder is None:
            print_error(
                "Builder not initialized",
                "AGENT_EXECUTION_FAILED",
                "Restart the CLI to reinitialize the builder."
            )
            return

        # Fix 9: Message count safeguard to prevent infinite loops
        if len(session.builder_messages) > MAX_MESSAGES_PER_CONVERSATION:
            logger.warning(
                f"Conversation exceeded {MAX_MESSAGES_PER_CONVERSATION} messages, "
                "forcing reset to prevent infinite loop"
            )
            print()
            print(warning(f"Conversation exceeded message limit ({MAX_MESSAGES_PER_CONVERSATION})."))
            print("This usually means the builder is in a loop.")
            print("Starting fresh conversation...")
            print()
            session.builder_messages = []
            reset_tool_history()  # Also reset tool call tracking

        session.builder_messages.append(HumanMessage(content=message))

        try:
            if session.streaming.enabled:
                # Use streaming mode with retry for transient failures
                async def stream_builder():
                    return await _stream_agent_response(
                        agent=session.builder,
                        messages=session.builder_messages,
                        config=session.streaming,
                        agent_label="Builder",
                        label_color=Colors.BRIGHT_MAGENTA,
                    )

                response, updated_messages = await with_retry(
                    stream_builder, max_retries=2, delay=1.0
                )
                if response:
                    session.builder_messages = updated_messages
                else:
                    print_warning("No response from builder. Try rephrasing your message.")
            else:
                # Use batch mode with retry
                async def batch_builder():
                    return await _batch_agent_response(
                        agent=session.builder,
                        messages=session.builder_messages,
                        agent_label="Builder",
                        label_color=Colors.BRIGHT_MAGENTA,
                    )

                response, updated_messages = await with_retry(
                    batch_builder, max_retries=2, delay=1.0
                )
                if response:
                    session.builder_messages = updated_messages
                else:
                    print_warning("No response from builder. Try rephrasing your message.")

        except Exception as e:
            handle_error(e, context="Chatting with builder")
            # Remove the failed message from history
            if session.builder_messages and session.builder_messages[-1].content == message:
                session.builder_messages.pop()

    else:
        # Chat with active agent
        if session.active_agent is None:
            print_error(
                "No agent loaded",
                "AGENT_NOT_FOUND",
                "Use /load <name> to load an agent, or /list to see available agents."
            )
            return

        # Fix 9: Message count safeguard for agent conversations
        if len(session.agent_messages) > MAX_MESSAGES_PER_CONVERSATION:
            logger.warning(
                f"Agent conversation exceeded {MAX_MESSAGES_PER_CONVERSATION} messages, "
                "forcing reset to prevent infinite loop"
            )
            print()
            print(warning(f"Conversation exceeded message limit ({MAX_MESSAGES_PER_CONVERSATION})."))
            print("This usually means the agent is in a loop.")
            print("Starting fresh conversation with this agent...")
            print()
            session.agent_messages = []
            reset_tool_history()  # Also reset tool call tracking

        session.agent_messages.append(HumanMessage(content=message))
        agent_name = session.active_agent_name or "Agent"

        try:
            if session.streaming.enabled:
                # Use streaming mode with retry
                async def stream_agent():
                    return await _stream_agent_response(
                        agent=session.active_agent,
                        messages=session.agent_messages,
                        config=session.streaming,
                        agent_label=agent_name,
                        label_color=Colors.BRIGHT_GREEN,
                    )

                response, updated_messages = await with_retry(
                    stream_agent, max_retries=2, delay=1.0
                )
                if response:
                    session.agent_messages = updated_messages
                else:
                    print_warning("No response from agent. Try rephrasing your message.")
            else:
                # Use batch mode with retry
                async def batch_agent():
                    return await _batch_agent_response(
                        agent=session.active_agent,
                        messages=session.agent_messages,
                        agent_label=agent_name,
                        label_color=Colors.BRIGHT_GREEN,
                    )

                response, updated_messages = await with_retry(
                    batch_agent, max_retries=2, delay=1.0
                )
                if response:
                    session.agent_messages = updated_messages
                else:
                    print_warning("No response from agent. Try rephrasing your message.")

        except Exception as e:
            handle_error(e, context=f"Chatting with {agent_name}")
            # Remove the failed message from history
            if session.agent_messages and session.agent_messages[-1].content == message:
                session.agent_messages.pop()

    print()


async def handle_command(session: Session, user_input: str) -> bool:
    """Handle a command (starts with /).

    Args:
        session: The current CLI session.
        user_input: The full user input including the command.

    Returns:
        True if the CLI should continue, False if it should exit.
    """
    # Parse command and arguments
    parts = user_input[1:].split(maxsplit=1)
    command = parts[0].lower() if parts else ""
    args = parts[1] if len(parts) > 1 else ""

    if command in ("exit", "quit", "q"):
        return False

    elif command == "create":
        await cmd_create(session, args)

    elif command == "list":
        await cmd_list(session)

    elif command == "load":
        await cmd_load(session, args)

    elif command == "run":
        await cmd_run(session, args)

    elif command == "modify":
        await cmd_modify(session, args)

    elif command == "rollback":
        await cmd_rollback(session, args)

    elif command == "delete":
        await cmd_delete(session, args)

    elif command == "status":
        await cmd_status(session)

    elif command == "tools":
        await cmd_tools(session)

    elif command == "builder":
        await cmd_builder(session)

    elif command == "stream":
        await cmd_stream(session, args)

    elif command == "debug":
        await cmd_debug(session, args)

    elif command == "help":
        cmd_help()

    else:
        print_error(
            f"Unknown command: /{command}",
            suggestion="Type /help for available commands."
        )

    return True


# =============================================================================
# Readline Configuration
# =============================================================================

COMMANDS = [
    "/create", "/list", "/load", "/run", "/modify", "/rollback", "/delete",
    "/status", "/tools", "/builder", "/stream", "/debug", "/help", "/exit"
]


def setup_readline() -> None:
    """Configure readline for history and tab completion."""
    # Set up history file
    history_file = Path.home() / ".acti_builder_history"

    try:
        readline.read_history_file(str(history_file))
        readline.set_history_length(1000)
    except FileNotFoundError:
        pass

    # Save history on exit
    atexit.register(readline.write_history_file, str(history_file))

    # Set up tab completion
    def completer(text: str, state: int) -> Optional[str]:
        """Tab completion function."""
        # Complete commands
        if text.startswith("/"):
            matches = [c for c in COMMANDS if c.startswith(text)]
        else:
            # Complete agent names for certain commands
            matches = []

        if state < len(matches):
            return matches[state]
        return None

    readline.set_completer(completer)
    readline.parse_and_bind("tab: complete")


# =============================================================================
# Main Entry Point
# =============================================================================

def _format_time_ago(iso_timestamp: str) -> str:
    """Format a timestamp as a human-readable 'time ago' string.

    Args:
        iso_timestamp: ISO format timestamp string.

    Returns:
        Human-readable string like '2 hours ago' or 'yesterday'.
    """
    try:
        then = datetime.fromisoformat(iso_timestamp)
        now = datetime.now()
        delta = now - then

        seconds = delta.total_seconds()
        if seconds < 60:
            return "just now"
        elif seconds < 3600:
            minutes = int(seconds / 60)
            return f"{minutes} minute{'s' if minutes != 1 else ''} ago"
        elif seconds < 86400:
            hours = int(seconds / 3600)
            return f"{hours} hour{'s' if hours != 1 else ''} ago"
        elif seconds < 172800:
            return "yesterday"
        else:
            days = int(seconds / 86400)
            return f"{days} days ago"
    except ValueError:
        return "unknown"


async def main_async() -> None:
    """Main async entry point for the CLI."""
    # Load environment variables
    env_path = Path(__file__).parent.parent / ".env"
    load_dotenv(env_path)

    # Detect TTY mode
    is_tty = sys.stdout.isatty()

    # Disable colors if not a TTY
    if not is_tty:
        Colors.disable()

    # Set up readline
    setup_readline()

    # Ensure ACTi directories exist
    ensure_acti_directories()

    # Print banner
    print()
    print(f"{Colors.BRIGHT_CYAN}{'=' * 60}{Colors.RESET}")
    print(f"{Colors.BRIGHT_CYAN}  ACTi Agent Builder - Interactive CLI{Colors.RESET}")
    print(f"{Colors.BRIGHT_CYAN}{'=' * 60}{Colors.RESET}")
    print()
    print("Create, manage, and interact with AI agents using real MCP tools.")
    print("Type /help for available commands, or just start chatting!")
    print()

    # Verify API key
    if not os.getenv("ANTHROPIC_API_KEY"):
        print_error(
            "ANTHROPIC_API_KEY environment variable not set",
            "API_KEY_INVALID",
            "Set ANTHROPIC_API_KEY in your .env file or environment."
        )
        return

    # Restore or create session
    try:
        session = Session.restore()
        is_restored = session.metadata.total_messages > 0
    except Exception as e:
        print_warning(f"Could not restore session: {e}")
        print(dim("Starting fresh session."))
        session = Session()
        is_restored = False

    # Reset tool call history for fresh session (Fix 8a/9)
    reset_tool_history()

    # Configure streaming based on TTY mode
    if is_tty:
        session.streaming = StreamingConfig.for_tty()
    else:
        session.streaming = StreamingConfig.for_non_tty()
        print(dim("Non-TTY detected: streaming disabled, using batch mode."))

    # Show welcome back message if restoring
    if is_restored:
        time_ago = _format_time_ago(session.metadata.last_used)
        print(f"{Colors.BRIGHT_YELLOW}Welcome back!{Colors.RESET}")
        print(f"Last session: {time_ago}")
        if session.active_agent_name:
            print(f"Previously active agent: {Colors.GREEN}{session.active_agent_name}{Colors.RESET}")
        if session.metadata.agents_used:
            agents_list = ", ".join(session.metadata.agents_used[:5])
            if len(session.metadata.agents_used) > 5:
                agents_list += f" (+{len(session.metadata.agents_used) - 5} more)"
            print(f"Agents used: {agents_list}")
        print(f"Total messages: {session.metadata.total_messages}")
        print()

    # Create builder agent
    print(info("Initializing builder agent..."))
    try:
        from .builder import create_builder
        # Fix 10: Apply strict recursion limit for interactive CLI to prevent infinite loops
        # max_turns=10 means the builder can make at most 10 rounds of tool calls
        # This translates to recursion_limit=50 (10 * 5) in the underlying graph
        session.builder = create_builder(max_turns=10)
        print(success("Builder ready (max_turns=10)."))
    except ImportError as e:
        print_error(
            f"Failed to import builder module: {e}",
            "AGENT_CREATION_FAILED",
            "Check that all dependencies are installed (pip install -e .)"
        )
        error_logger.log_error(e, "AGENT_CREATION_FAILED", {"phase": "initialization"})
        return
    except Exception as e:
        handle_error(e, context="Initializing builder agent")
        return

    # If there was a previously active agent, offer to reload it
    if is_restored and session.active_agent_name:
        print()
        print(dim(f"Tip: Use '/load {session.active_agent_name}' to reload your previous agent."))
        # Reset to builder mode since the agent object wasn't persisted
        session.mode = SessionMode.BUILDER
        session.active_agent = None

    # Register session save on exit
    def save_on_exit() -> None:
        """Save session state when exiting."""
        session.save()
        logger.debug("Session saved on exit")

    atexit.register(save_on_exit)

    print()
    if is_restored:
        print(dim("Session restored. You are in builder mode. Use /help for commands."))
    else:
        print(dim("You are now in builder mode. Describe an agent to create, or use /help."))
    print()

    # Main loop
    running = True
    while running:
        try:
            user_input = input(session.prompt).strip()
        except (KeyboardInterrupt, EOFError):
            print()
            print(dim("Saving session..."))
            session.save()
            print(dim("Goodbye!"))
            break

        if not user_input:
            continue

        if user_input.startswith("/"):
            running = await handle_command(session, user_input)
            # Save session after commands
            if not running:
                session.save()
        else:
            await handle_chat_message(session, user_input)
            # Record message and periodically save
            session.record_message()


def main() -> None:
    """Synchronous entry point for the CLI."""
    # Set up signal handlers
    def signal_handler(sig, frame):
        print()
        print(dim("Shutting down..."))
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Run the async main
    try:
        asyncio.run(main_async())
    except KeyboardInterrupt:
        print()
        print(dim("Goodbye!"))


if __name__ == "__main__":
    main()
