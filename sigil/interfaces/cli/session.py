"""Interactive Session Management for Sigil v2 CLI.

This module provides interactive session management for the Sigil CLI,
including message history tracking, session persistence, and REPL state.

Key Components:
    - InteractiveSession: Session state and history management
    - Message: Individual message representation
    - Session utilities: Create, load, list, delete sessions

Sessions are persisted as JSON files in the session storage directory,
allowing for session recovery and history viewing.

Example:
    >>> from sigil.interfaces.cli.session import InteractiveSession, create_session
    >>> session = create_session()
    >>> await session.add_message("Hello, Sigil!", source="user")
    >>> history = await session.get_history()
    >>> len(history)
    1
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from sigil.interfaces.cli.config import CLIConfig, get_cli_config


# =============================================================================
# Constants
# =============================================================================

# Default session storage directory
DEFAULT_SESSION_DIR = Path("outputs/cli-sessions")


# =============================================================================
# Message Dataclass
# =============================================================================


@dataclass
class Message:
    """Individual message in session history.

    Represents a single message exchanged during an interactive session,
    with metadata for tracking and display.

    Attributes:
        content: Message text content.
        source: Message source ('user', 'assistant', 'system').
        timestamp: When the message was created.
        message_id: Unique identifier for the message.
        metadata: Additional message metadata.

    Example:
        >>> msg = Message(content="Hello!", source="user")
        >>> msg.to_dict()
        {'content': 'Hello!', 'source': 'user', 'timestamp': '...', ...}
    """

    content: str
    source: str = "user"
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    message_id: str = field(default_factory=lambda: f"msg-{uuid.uuid4().hex[:8]}")
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary.

        Returns:
            Dictionary representation of the message.
        """
        return {
            "content": self.content,
            "source": self.source,
            "timestamp": self.timestamp.isoformat(),
            "message_id": self.message_id,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Message":
        """Create message from dictionary.

        Args:
            data: Dictionary with message data.

        Returns:
            New Message instance.
        """
        timestamp = data.get("timestamp")
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        elif timestamp is None:
            timestamp = datetime.now(timezone.utc)

        return cls(
            content=data.get("content", ""),
            source=data.get("source", "user"),
            timestamp=timestamp,
            message_id=data.get("message_id", f"msg-{uuid.uuid4().hex[:8]}"),
            metadata=data.get("metadata", {}),
        )


# =============================================================================
# InteractiveSession Class
# =============================================================================


class InteractiveSession:
    """Interactive CLI session with history management.

    Manages session state, message history, and persistence for
    interactive CLI sessions. Supports async operations for
    compatibility with the async CLI infrastructure.

    Attributes:
        session_id: Unique session identifier.
        config: CLI configuration for this session.
        created_at: When the session was created.
        updated_at: When the session was last updated.

    Example:
        >>> session = InteractiveSession(session_id="test-123")
        >>> await session.add_message("Hello!", source="user")
        >>> await session.add_message("Hi there!", source="assistant")
        >>> history = await session.get_history()
        >>> len(history)
        2
    """

    def __init__(
        self,
        session_id: str,
        config: Optional[CLIConfig] = None,
        storage_dir: Optional[Path] = None,
    ) -> None:
        """Initialize interactive session.

        Args:
            session_id: Unique session identifier.
            config: CLI configuration. Uses global if None.
            storage_dir: Directory for session storage.
        """
        self.session_id = session_id
        self.config = config or get_cli_config()
        self.storage_dir = storage_dir or DEFAULT_SESSION_DIR

        # Session state
        self.created_at = datetime.now(timezone.utc)
        self.updated_at = self.created_at
        self._messages: List[Message] = []
        self._active = True
        self._metadata: Dict[str, Any] = {}

        # Token tracking
        self._tokens_used = 0

        # Ensure storage directory exists
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    @property
    def message_count(self) -> int:
        """Get number of messages in session.

        Returns:
            Number of messages.
        """
        return len(self._messages)

    @property
    def tokens_used(self) -> int:
        """Get total tokens used in session.

        Returns:
            Number of tokens consumed.
        """
        return self._tokens_used

    async def add_message(
        self,
        message: str,
        source: str = "user",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Message:
        """Add message to session history.

        Args:
            message: Message content.
            source: Message source ('user', 'assistant', 'system').
            metadata: Optional message metadata.

        Returns:
            Created Message instance.
        """
        msg = Message(
            content=message,
            source=source,
            metadata=metadata or {},
        )
        self._messages.append(msg)
        self.updated_at = datetime.now(timezone.utc)

        # Auto-save if persistence is enabled
        if self.config.metrics_enabled:
            await self.save()

        return msg

    async def get_history(self) -> List[Dict[str, Any]]:
        """Get session message history.

        Returns:
            List of message dictionaries in chronological order.
        """
        return [msg.to_dict() for msg in self._messages]

    async def clear_history(self) -> None:
        """Clear all messages from session history."""
        self._messages = []
        self.updated_at = datetime.now(timezone.utc)
        await self.save()

    async def save(self) -> None:
        """Save session to storage.

        Persists the session state to a JSON file in the storage directory.
        """
        file_path = self._get_session_file_path()

        data = {
            "session_id": self.session_id,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "active": self._active,
            "tokens_used": self._tokens_used,
            "metadata": self._metadata,
            "messages": [msg.to_dict() for msg in self._messages],
            "config": {
                "debug": self.config.debug,
                "verbose": self.config.verbose,
                "token_budget": self.config.token_budget,
            },
        }

        # Write atomically using temp file
        temp_path = file_path.with_suffix(".tmp")
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        temp_path.rename(file_path)

    async def load(self, session_id: str) -> None:
        """Load existing session from storage.

        Args:
            session_id: Session ID to load.

        Raises:
            FileNotFoundError: If session file doesn't exist.
            json.JSONDecodeError: If session file is corrupted.
        """
        self.session_id = session_id
        file_path = self._get_session_file_path()

        if not file_path.exists():
            raise FileNotFoundError(f"Session not found: {session_id}")

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Restore state
        self.created_at = datetime.fromisoformat(
            data["created_at"].replace("Z", "+00:00")
        )
        self.updated_at = datetime.fromisoformat(
            data["updated_at"].replace("Z", "+00:00")
        )
        self._active = data.get("active", True)
        self._tokens_used = data.get("tokens_used", 0)
        self._metadata = data.get("metadata", {})

        # Restore messages
        self._messages = [
            Message.from_dict(msg_data)
            for msg_data in data.get("messages", [])
        ]

    def get_metadata(self) -> Dict[str, Any]:
        """Get session metadata.

        Returns:
            Dictionary with session metadata including:
            - id: Session ID
            - created_at: Creation timestamp
            - updated_at: Last update timestamp
            - message_count: Number of messages
            - tokens_used: Tokens consumed
            - active: Whether session is active
        """
        return {
            "id": self.session_id,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "message_count": self.message_count,
            "tokens_used": self._tokens_used,
            "active": self._active,
        }

    def is_active(self) -> bool:
        """Check if session is active.

        Returns:
            True if session is active and accepting input.
        """
        return self._active

    def deactivate(self) -> None:
        """Mark session as inactive."""
        self._active = False
        self.updated_at = datetime.now(timezone.utc)

    def activate(self) -> None:
        """Mark session as active."""
        self._active = True
        self.updated_at = datetime.now(timezone.utc)

    def add_tokens(self, amount: int) -> None:
        """Add to token usage count.

        Args:
            amount: Number of tokens to add.
        """
        self._tokens_used += amount
        self.updated_at = datetime.now(timezone.utc)

    def set_metadata(self, key: str, value: Any) -> None:
        """Set metadata value.

        Args:
            key: Metadata key.
            value: Metadata value.
        """
        self._metadata[key] = value
        self.updated_at = datetime.now(timezone.utc)

    def get_metadata_value(self, key: str, default: Any = None) -> Any:
        """Get metadata value.

        Args:
            key: Metadata key.
            default: Default value if key not found.

        Returns:
            Metadata value or default.
        """
        return self._metadata.get(key, default)

    def get_conversation_history(self) -> List[Dict[str, str]]:
        """Get messages formatted for LLM consumption.

        Returns:
            List of message dicts with 'role' and 'content' keys.
        """
        role_map = {
            "user": "user",
            "assistant": "assistant",
            "system": "system",
        }
        return [
            {
                "role": role_map.get(msg.source, "user"),
                "content": msg.content,
            }
            for msg in self._messages
        ]

    def _get_session_file_path(self) -> Path:
        """Get file path for this session.

        Returns:
            Path to session JSON file.
        """
        # Sanitize session_id for filesystem
        safe_id = (
            self.session_id
            .replace("/", "_")
            .replace("\\", "_")
            .replace("..", "_")
        )
        return self.storage_dir / f"{safe_id}.json"


# =============================================================================
# Global Session Management
# =============================================================================

# Global active session
_active_session: Optional[InteractiveSession] = None


def create_session(
    session_id: Optional[str] = None,
    config: Optional[CLIConfig] = None,
    storage_dir: Optional[Path] = None,
) -> InteractiveSession:
    """Create a new interactive session.

    Creates a new session and optionally sets it as the active session.

    Args:
        session_id: Optional custom session ID. Auto-generated if None.
        config: Optional CLI configuration. Uses global if None.
        storage_dir: Optional storage directory.

    Returns:
        New InteractiveSession instance.

    Example:
        >>> session = create_session()
        >>> session.session_id
        'cli-abc123...'
    """
    global _active_session

    if session_id is None:
        session_id = f"cli-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"

    session = InteractiveSession(
        session_id=session_id,
        config=config,
        storage_dir=storage_dir,
    )

    # Set as active session
    _active_session = session

    return session


def get_active_session() -> InteractiveSession:
    """Get the currently active session.

    Returns the active session, creating a new one if none exists.

    Returns:
        Currently active InteractiveSession.

    Example:
        >>> session = get_active_session()
        >>> session.is_active()
        True
    """
    global _active_session

    if _active_session is None:
        _active_session = create_session()

    return _active_session


def reset_active_session() -> None:
    """Reset the active session.

    Clears the global active session reference.
    """
    global _active_session
    _active_session = None


def set_active_session(session: InteractiveSession) -> None:
    """Set the active session.

    Args:
        session: Session to set as active.
    """
    global _active_session
    _active_session = session


async def load_session(
    session_id: str,
    storage_dir: Optional[Path] = None,
) -> InteractiveSession:
    """Load an existing session.

    Args:
        session_id: Session ID to load.
        storage_dir: Optional storage directory override.

    Returns:
        Loaded InteractiveSession.

    Raises:
        FileNotFoundError: If session doesn't exist.
    """
    session = InteractiveSession(
        session_id=session_id,
        storage_dir=storage_dir,
    )
    await session.load(session_id)
    return session


def list_sessions(storage_dir: Optional[Path] = None) -> List[Dict[str, Any]]:
    """List all saved sessions.

    Args:
        storage_dir: Optional storage directory override.

    Returns:
        List of session metadata dictionaries.
    """
    directory = storage_dir or DEFAULT_SESSION_DIR

    if not directory.exists():
        return []

    sessions = []
    for file_path in directory.glob("*.json"):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            sessions.append({
                "session_id": data.get("session_id", file_path.stem),
                "created_at": data.get("created_at"),
                "updated_at": data.get("updated_at"),
                "message_count": len(data.get("messages", [])),
                "active": data.get("active", True),
                "tokens_used": data.get("tokens_used", 0),
            })
        except (json.JSONDecodeError, OSError):
            # Skip corrupted files
            continue

    # Sort by updated_at descending (most recent first)
    sessions.sort(
        key=lambda s: s.get("updated_at", ""),
        reverse=True,
    )

    return sessions


def delete_session(
    session_id: str,
    storage_dir: Optional[Path] = None,
) -> bool:
    """Delete a session.

    Args:
        session_id: Session ID to delete.
        storage_dir: Optional storage directory override.

    Returns:
        True if session was deleted, False if not found.
    """
    directory = storage_dir or DEFAULT_SESSION_DIR

    # Sanitize session_id
    safe_id = (
        session_id
        .replace("/", "_")
        .replace("\\", "_")
        .replace("..", "_")
    )
    file_path = directory / f"{safe_id}.json"

    if file_path.exists():
        file_path.unlink()
        return True

    return False


def get_session_path(
    session_id: str,
    storage_dir: Optional[Path] = None,
) -> Path:
    """Get the file path for a session.

    Args:
        session_id: Session ID.
        storage_dir: Optional storage directory override.

    Returns:
        Path to session file.
    """
    directory = storage_dir or DEFAULT_SESSION_DIR
    safe_id = (
        session_id
        .replace("/", "_")
        .replace("\\", "_")
        .replace("..", "_")
    )
    return directory / f"{safe_id}.json"


def session_exists(
    session_id: str,
    storage_dir: Optional[Path] = None,
) -> bool:
    """Check if a session exists.

    Args:
        session_id: Session ID to check.
        storage_dir: Optional storage directory override.

    Returns:
        True if session exists.
    """
    return get_session_path(session_id, storage_dir).exists()


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Constants
    "DEFAULT_SESSION_DIR",
    # Classes
    "Message",
    "InteractiveSession",
    # Global session functions
    "create_session",
    "get_active_session",
    "reset_active_session",
    "set_active_session",
    "load_session",
    "list_sessions",
    "delete_session",
    "get_session_path",
    "session_exists",
]
