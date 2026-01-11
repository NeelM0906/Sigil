"""Event store for Sigil v2 event-sourced state management.

This module implements an append-only event store for persisting events.
Events are stored as JSON files, one per session, for easy debugging and
portability.

Key Features:
    - Append-only semantics (events are immutable)
    - JSON file storage (human-readable, debuggable)
    - File locking for concurrent access safety
    - Query methods for event retrieval

Storage Format:
    Each session's events are stored in a JSON file:
    {storage_dir}/{session_id}.json

    File contents:
    {
        "session_id": "...",
        "created_at": "...",
        "updated_at": "...",
        "events": [...]
    }

Example:
    >>> from sigil.state.store import EventStore
    >>> from sigil.state.events import create_session_started_event
    >>>
    >>> store = EventStore(storage_dir="/tmp/sessions")
    >>> event = create_session_started_event("session-123")
    >>> store.append(event)
    >>> events = store.get_events("session-123")
    >>> len(events)
    1
"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import portalocker

from sigil.state.events import (
    Event,
    EventType,
    _datetime_to_iso,
    _datetime_from_iso,
)


class EventStoreError(Exception):
    """Base exception for event store errors."""

    pass


class SessionNotFoundError(EventStoreError):
    """Raised when a session is not found in the store."""

    pass


class EventStore:
    """Append-only event store with JSON file persistence.

    Events are stored in JSON files, one per session. File locking ensures
    safe concurrent access. Events cannot be modified or deleted once
    appended (append-only semantics).

    Attributes:
        storage_dir: Directory where session event files are stored.

    Example:
        >>> store = EventStore(storage_dir="./outputs/sessions")
        >>> event = create_session_started_event("sess-123")
        >>> store.append(event)
        >>> events = store.get_events("sess-123")
    """

    def __init__(self, storage_dir: str = "outputs/sessions") -> None:
        """Initialize the event store.

        Args:
            storage_dir: Directory for storing session event files.
                Creates the directory if it does not exist.
        """
        self.storage_dir = Path(storage_dir)
        self._ensure_storage_dir()

    def _ensure_storage_dir(self) -> None:
        """Ensure the storage directory exists."""
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    def _get_session_file_path(self, session_id: str) -> Path:
        """Get the file path for a session's events.

        Args:
            session_id: Unique identifier for the session.

        Returns:
            Path to the session's JSON file.
        """
        # Sanitize session_id to prevent path traversal
        safe_id = session_id.replace("/", "_").replace("\\", "_").replace("..", "_")
        return self.storage_dir / f"{safe_id}.json"

    def _read_session_file(self, session_id: str) -> dict:
        """Read a session's event file with locking.

        Args:
            session_id: Unique identifier for the session.

        Returns:
            Dictionary containing session data and events.

        Raises:
            SessionNotFoundError: If the session file does not exist.
        """
        file_path = self._get_session_file_path(session_id)

        if not file_path.exists():
            raise SessionNotFoundError(f"Session not found: {session_id}")

        with portalocker.Lock(
            file_path, mode="r", timeout=10, flags=portalocker.LOCK_SH
        ) as f:
            return json.load(f)

    def _write_session_file(self, session_id: str, data: dict) -> None:
        """Write a session's event file with locking.

        Args:
            session_id: Unique identifier for the session.
            data: Dictionary containing session data and events.
        """
        file_path = self._get_session_file_path(session_id)

        with portalocker.Lock(
            file_path, mode="w", timeout=10, flags=portalocker.LOCK_EX
        ) as f:
            json.dump(data, f, indent=2)

    def _create_session_file(self, session_id: str) -> dict:
        """Create a new session file.

        Args:
            session_id: Unique identifier for the session.

        Returns:
            Dictionary containing the initial session data structure.
        """
        now = _datetime_to_iso(datetime.now(timezone.utc))
        data = {
            "session_id": session_id,
            "created_at": now,
            "updated_at": now,
            "events": [],
        }
        self._write_session_file(session_id, data)
        return data

    def session_exists(self, session_id: str) -> bool:
        """Check if a session exists in the store.

        Args:
            session_id: Unique identifier for the session.

        Returns:
            True if the session exists, False otherwise.
        """
        file_path = self._get_session_file_path(session_id)
        return file_path.exists()

    def append(self, event: Event) -> None:
        """Append an event to the store.

        Events are immutable and cannot be modified once appended.
        This method is thread-safe using file locking.

        Args:
            event: The event to append.

        Raises:
            EventStoreError: If the event cannot be appended.
        """
        session_id = event.session_id
        file_path = self._get_session_file_path(session_id)
        lock_path = file_path.with_suffix(".lock")

        # Use a separate lock file for atomic operations
        try:
            with portalocker.Lock(
                lock_path, mode="w", timeout=10, flags=portalocker.LOCK_EX
            ):
                # Create the file if it doesn't exist (inside the lock)
                if not file_path.exists():
                    now = _datetime_to_iso(datetime.now(timezone.utc))
                    data = {
                        "session_id": session_id,
                        "created_at": now,
                        "updated_at": now,
                        "events": [],
                    }
                else:
                    # Read existing data
                    with open(file_path, "r") as f:
                        data = json.load(f)

                # Append the event
                data["events"].append(event.to_dict())
                data["updated_at"] = _datetime_to_iso(datetime.now(timezone.utc))

                # Write atomically using temp file
                temp_path = file_path.with_suffix(".tmp")
                with open(temp_path, "w") as f:
                    json.dump(data, f, indent=2)

                # Rename is atomic on POSIX systems
                temp_path.rename(file_path)

        except portalocker.LockException as e:
            raise EventStoreError(f"Failed to acquire lock for session {session_id}: {e}")
        except json.JSONDecodeError as e:
            raise EventStoreError(f"Corrupted session file for {session_id}: {e}")
        except Exception as e:
            raise EventStoreError(f"Failed to append event: {e}")

    def append_batch(self, events: list[Event]) -> None:
        """Append multiple events to the store atomically.

        All events must belong to the same session. This is more efficient
        than appending events one at a time.

        Args:
            events: List of events to append.

        Raises:
            ValueError: If events belong to different sessions.
            EventStoreError: If the events cannot be appended.
        """
        if not events:
            return

        # Verify all events are for the same session
        session_id = events[0].session_id
        if not all(e.session_id == session_id for e in events):
            raise ValueError("All events must belong to the same session")

        file_path = self._get_session_file_path(session_id)
        lock_path = file_path.with_suffix(".lock")

        try:
            with portalocker.Lock(
                lock_path, mode="w", timeout=10, flags=portalocker.LOCK_EX
            ):
                # Create the file if it doesn't exist (inside the lock)
                if not file_path.exists():
                    now = _datetime_to_iso(datetime.now(timezone.utc))
                    data = {
                        "session_id": session_id,
                        "created_at": now,
                        "updated_at": now,
                        "events": [],
                    }
                else:
                    # Read existing data
                    with open(file_path, "r") as f:
                        data = json.load(f)

                # Append all events
                for event in events:
                    data["events"].append(event.to_dict())
                data["updated_at"] = _datetime_to_iso(datetime.now(timezone.utc))

                # Write atomically using temp file
                temp_path = file_path.with_suffix(".tmp")
                with open(temp_path, "w") as f:
                    json.dump(data, f, indent=2)

                # Rename is atomic on POSIX systems
                temp_path.rename(file_path)

        except portalocker.LockException as e:
            raise EventStoreError(f"Failed to acquire lock for session {session_id}: {e}")
        except Exception as e:
            raise EventStoreError(f"Failed to append events: {e}")

    def get_events(self, session_id: str) -> list[Event]:
        """Get all events for a session.

        Args:
            session_id: Unique identifier for the session.

        Returns:
            List of events for the session, in chronological order.

        Raises:
            SessionNotFoundError: If the session does not exist.
        """
        data = self._read_session_file(session_id)
        return [Event.from_dict(e) for e in data["events"]]

    def get_events_by_type(
        self, event_type: EventType, session_id: Optional[str] = None
    ) -> list[Event]:
        """Get events filtered by type.

        Args:
            event_type: Type of events to retrieve.
            session_id: Optional session ID to filter by. If None, searches
                all sessions.

        Returns:
            List of events matching the specified type.
        """
        if session_id:
            events = self.get_events(session_id)
            return [e for e in events if e.event_type == event_type]

        # Search all sessions
        all_events = []
        for file_path in self.storage_dir.glob("*.json"):
            try:
                session_id = file_path.stem
                events = self.get_events(session_id)
                all_events.extend([e for e in events if e.event_type == event_type])
            except EventStoreError:
                continue  # Skip corrupted files

        # Sort by timestamp
        all_events.sort(key=lambda e: e.timestamp)
        return all_events

    def get_events_in_range(
        self,
        start: datetime,
        end: datetime,
        session_id: Optional[str] = None,
    ) -> list[Event]:
        """Get events within a time range.

        Args:
            start: Start of the time range (inclusive).
            end: End of the time range (inclusive).
            session_id: Optional session ID to filter by. If None, searches
                all sessions.

        Returns:
            List of events within the specified time range.
        """
        if session_id:
            events = self.get_events(session_id)
            return [e for e in events if start <= e.timestamp <= end]

        # Search all sessions
        all_events = []
        for file_path in self.storage_dir.glob("*.json"):
            try:
                sess_id = file_path.stem
                events = self.get_events(sess_id)
                all_events.extend([e for e in events if start <= e.timestamp <= end])
            except EventStoreError:
                continue  # Skip corrupted files

        # Sort by timestamp
        all_events.sort(key=lambda e: e.timestamp)
        return all_events

    def get_latest_events(
        self, n: int, session_id: Optional[str] = None
    ) -> list[Event]:
        """Get the N most recent events.

        Args:
            n: Number of events to retrieve.
            session_id: Optional session ID to filter by. If None, retrieves
                from all sessions.

        Returns:
            List of the N most recent events, ordered newest first.
        """
        if session_id:
            events = self.get_events(session_id)
            # Sort by timestamp descending, take n
            events.sort(key=lambda e: e.timestamp, reverse=True)
            return events[:n]

        # Search all sessions
        all_events = []
        for file_path in self.storage_dir.glob("*.json"):
            try:
                sess_id = file_path.stem
                events = self.get_events(sess_id)
                all_events.extend(events)
            except EventStoreError:
                continue  # Skip corrupted files

        # Sort by timestamp descending, take n
        all_events.sort(key=lambda e: e.timestamp, reverse=True)
        return all_events[:n]

    def get_session_ids(self) -> list[str]:
        """Get all session IDs in the store.

        Returns:
            List of session IDs.
        """
        return [f.stem for f in self.storage_dir.glob("*.json")]

    def get_session_metadata(self, session_id: str) -> dict:
        """Get metadata about a session without loading all events.

        Args:
            session_id: Unique identifier for the session.

        Returns:
            Dictionary with session metadata including:
            - session_id: The session ID
            - created_at: When the session was created
            - updated_at: When the session was last updated
            - event_count: Number of events in the session

        Raises:
            SessionNotFoundError: If the session does not exist.
        """
        data = self._read_session_file(session_id)
        return {
            "session_id": data["session_id"],
            "created_at": data["created_at"],
            "updated_at": data["updated_at"],
            "event_count": len(data["events"]),
        }

    def delete_session(self, session_id: str) -> None:
        """Delete a session and all its events.

        Note: This operation is irreversible. Use with caution.

        Args:
            session_id: Unique identifier for the session to delete.

        Raises:
            SessionNotFoundError: If the session does not exist.
        """
        file_path = self._get_session_file_path(session_id)

        if not file_path.exists():
            raise SessionNotFoundError(f"Session not found: {session_id}")

        try:
            file_path.unlink()
        except Exception as e:
            raise EventStoreError(f"Failed to delete session {session_id}: {e}")

    def get_event_count(self, session_id: Optional[str] = None) -> int:
        """Get the total number of events.

        Args:
            session_id: Optional session ID. If None, counts all events.

        Returns:
            Total number of events.
        """
        if session_id:
            metadata = self.get_session_metadata(session_id)
            return metadata["event_count"]

        total = 0
        for file_path in self.storage_dir.glob("*.json"):
            try:
                sess_id = file_path.stem
                metadata = self.get_session_metadata(sess_id)
                total += metadata["event_count"]
            except EventStoreError:
                continue

        return total
