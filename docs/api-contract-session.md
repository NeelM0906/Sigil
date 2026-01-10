# API Contract: Session Management (Task 2.5.3)

## Overview

This document defines the API contract for session management in the ACTi Agent Builder CLI. Session management provides:

- **Session State Persistence**: Save and restore CLI sessions across restarts
- **Conversation History Management**: Maintain separate conversation histories per agent
- **Session Metadata Tracking**: Track timestamps, agent switches, and usage metrics
- **Multi-Agent Context Switching**: Preserve context when switching between agents
- **Session Configuration**: User preferences and defaults

Sessions persist to `~/.acti/sessions/` and enable users to resume work exactly where they left off.

---

## Design Principles

1. **Privacy-First**: Conversation histories are stored locally; sensitive data is never transmitted
2. **Graceful Degradation**: CLI works without persistence; persistence enhances but is not required
3. **Atomic Operations**: Session saves are atomic to prevent corruption
4. **Minimal Footprint**: Store only what's necessary for resumption
5. **Agent Isolation**: Each agent maintains its own conversation history
6. **Extensibility**: Schema supports future features (metrics, analytics, team sharing)

---

## Storage Layout

```
~/.acti/
  config.yaml                    # Global CLI configuration
  sessions/
    current.json                 # Pointer to most recent session
    {session_id}.json            # Session state files
    history/
      builder.jsonl              # Builder conversation history (append-only)
      {agent_name}.jsonl         # Per-agent conversation histories
  agents/
    {agent_name}/
      history.jsonl              # Alternative: co-located with agent
```

### File Permissions

All session files are created with restricted permissions:
- Files: `0600` (owner read/write only)
- Directories: `0700` (owner full access only)

---

## Type Definitions

### `SessionMode`

Current interaction mode for the session.

```python
from enum import Enum

class SessionMode(str, Enum):
    """Current interaction mode for the CLI session."""
    BUILDER = "builder"      # Creating/managing agents
    AGENT = "agent"          # Chatting with an active agent
    MODIFY = "modify"        # Modifying an agent configuration
```

### `SessionConfig`

User preferences and session behavior configuration.

```python
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

@dataclass
class SessionConfig:
    """Configuration for session behavior and persistence."""

    # Persistence settings
    auto_save: bool = True                       # Auto-save session on changes
    save_interval_seconds: int = 30              # Auto-save interval (0 = on every change)
    session_dir: Path = field(
        default_factory=lambda: Path.home() / ".acti" / "sessions"
    )

    # History settings
    history_enabled: bool = True                 # Store conversation histories
    max_history_messages: int = 1000             # Max messages per agent history
    history_retention_days: int = 30             # Auto-cleanup after N days (0 = forever)

    # Resume settings
    auto_resume: bool = True                     # Resume last session on CLI start
    restore_active_agent: bool = True            # Restore previously active agent

    # Display settings
    show_session_info: bool = True               # Show session ID in status
    prompt_on_unsaved: bool = True               # Prompt before exit with unsaved changes

    # Metrics settings
    track_metrics: bool = True                   # Track usage metrics
```

### `SessionMetadata`

Metadata about the session lifecycle and usage.

```python
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List

@dataclass
class SessionMetadata:
    """Metadata tracking session lifecycle and usage."""

    # Identity
    session_id: str                              # Unique session identifier (UUID)
    created_at: datetime                         # When session was created
    last_saved_at: Optional[datetime] = None     # When session was last saved
    last_activity_at: datetime = field(
        default_factory=datetime.now
    )                                            # Last user interaction

    # Lifecycle
    cli_version: str = "1.0.0"                   # CLI version that created session
    resumed_count: int = 0                       # Times this session was resumed

    # Agent tracking
    agents_used: List[str] = field(
        default_factory=list
    )                                            # Agent names used in this session
    agent_switches: int = 0                      # Number of agent context switches
    current_agent_loaded_at: Optional[datetime] = None  # When current agent was loaded
```

### `SessionMetrics`

Usage metrics for analytics and optimization.

```python
from dataclasses import dataclass, field
from typing import Dict

@dataclass
class SessionMetrics:
    """Usage metrics for the session."""

    # Message counts
    builder_messages_sent: int = 0               # Messages sent to builder
    builder_messages_received: int = 0           # Responses from builder
    agent_messages_sent: int = 0                 # Messages sent to agents
    agent_messages_received: int = 0             # Responses from agents

    # Agent metrics
    agents_created: int = 0                      # Agents created in session
    agents_loaded: int = 0                       # Times agents were loaded
    agents_modified: int = 0                     # Agents modified in session
    agents_deleted: int = 0                      # Agents deleted in session

    # Tool metrics
    tool_calls_total: int = 0                    # Total MCP tool invocations
    tool_calls_by_tool: Dict[str, int] = field(
        default_factory=dict
    )                                            # Tool calls per tool category

    # Timing metrics
    total_active_seconds: float = 0.0            # Total active time
    avg_response_time_ms: float = 0.0            # Average agent response time

    # Error tracking
    errors_count: int = 0                        # Total errors encountered
    mcp_connection_failures: int = 0             # MCP connection failures
```

### `ConversationMessage`

A single message in a conversation history.

```python
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, List, Any, Literal

@dataclass
class ConversationMessage:
    """A single message in conversation history."""

    # Identity
    id: str                                      # Unique message ID (UUID)
    timestamp: datetime                          # When message was sent/received

    # Content
    role: Literal["human", "assistant", "system", "tool"]
    content: str                                 # Message content

    # Metadata
    agent_name: Optional[str] = None             # Which agent (None for builder)
    tool_calls: Optional[List[dict]] = None      # Tool calls made (if assistant)
    tool_results: Optional[List[dict]] = None    # Tool results (if tool message)
    model: Optional[str] = None                  # Model used (if assistant)
    tokens_used: Optional[int] = None            # Token count (if tracked)

    # UI hints
    is_streaming: bool = False                   # Was this a streaming response
    duration_ms: Optional[int] = None            # Response generation time
```

### `ConversationHistory`

Complete conversation history for a context (builder or agent).

```python
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional

@dataclass
class ConversationHistory:
    """Conversation history for a single context."""

    # Identity
    context_name: str                            # "builder" or agent name
    context_type: Literal["builder", "agent"]

    # History
    messages: List[ConversationMessage] = field(
        default_factory=list
    )

    # Metadata
    created_at: datetime = field(
        default_factory=datetime.now
    )
    last_message_at: Optional[datetime] = None
    total_messages: int = 0
    total_tokens: int = 0                        # If tracking tokens
```

### `AgentContext`

Context preserved when switching away from an agent.

```python
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List

@dataclass
class AgentContext:
    """Preserved context for an agent between switches."""

    # Identity
    agent_name: str                              # Agent name
    agent_config_path: str                       # Path to agent config file

    # State at switch-away
    last_active_at: datetime                     # When user switched away
    message_count: int                           # Messages in conversation
    last_message_preview: Optional[str] = None   # Preview of last message

    # History reference
    history_file: str                            # Path to history file

    # Runtime state (not persisted, rebuilt on load)
    # Note: The actual agent graph is NOT persisted; it's recreated on load
```

### `Session`

Complete session state combining all components.

```python
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, TYPE_CHECKING

if TYPE_CHECKING:
    from langgraph.graph.state import CompiledStateGraph

@dataclass
class Session:
    """Complete session state for the CLI."""

    # Core state
    session_id: str                              # Unique identifier
    mode: SessionMode = SessionMode.BUILDER      # Current interaction mode

    # Active state (runtime only, not persisted)
    builder: Optional["CompiledStateGraph"] = None
    active_agent: Optional["CompiledStateGraph"] = None
    active_agent_name: Optional[str] = None

    # Persisted state
    config: SessionConfig = field(
        default_factory=SessionConfig
    )
    metadata: SessionMetadata = field(
        default_factory=lambda: SessionMetadata(
            session_id="",
            created_at=datetime.now(),
        )
    )
    metrics: SessionMetrics = field(
        default_factory=SessionMetrics
    )

    # Context tracking
    agent_contexts: Dict[str, AgentContext] = field(
        default_factory=dict
    )                                            # Preserved agent contexts

    # Dirty tracking
    _dirty: bool = field(default=False, repr=False)  # Has unsaved changes

    def mark_dirty(self) -> None:
        """Mark session as having unsaved changes."""
        self._dirty = True
        self.metadata.last_activity_at = datetime.now()

    def mark_clean(self) -> None:
        """Mark session as saved."""
        self._dirty = False
        self.metadata.last_saved_at = datetime.now()

    @property
    def has_unsaved_changes(self) -> bool:
        """Check if session has unsaved changes."""
        return self._dirty
```

---

## Persistence Schema

### Session File Format (`{session_id}.json`)

```json
{
  "schema_version": "1.0.0",
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "mode": "agent",
  "active_agent_name": "lead_qualifier",

  "metadata": {
    "created_at": "2026-01-09T14:00:00Z",
    "last_saved_at": "2026-01-09T16:30:00Z",
    "last_activity_at": "2026-01-09T16:28:45Z",
    "cli_version": "1.0.0",
    "resumed_count": 3,
    "agents_used": ["lead_qualifier", "appointment_scheduler"],
    "agent_switches": 5,
    "current_agent_loaded_at": "2026-01-09T15:45:00Z"
  },

  "config": {
    "auto_save": true,
    "save_interval_seconds": 30,
    "history_enabled": true,
    "max_history_messages": 1000,
    "history_retention_days": 30,
    "auto_resume": true,
    "restore_active_agent": true,
    "track_metrics": true
  },

  "metrics": {
    "builder_messages_sent": 12,
    "builder_messages_received": 12,
    "agent_messages_sent": 45,
    "agent_messages_received": 45,
    "agents_created": 2,
    "agents_loaded": 5,
    "tool_calls_total": 23,
    "tool_calls_by_tool": {
      "websearch": 15,
      "communication": 8
    },
    "total_active_seconds": 3600.0,
    "errors_count": 1
  },

  "agent_contexts": {
    "lead_qualifier": {
      "agent_name": "lead_qualifier",
      "agent_config_path": "outputs/agents/lead_qualifier.json",
      "last_active_at": "2026-01-09T16:28:45Z",
      "message_count": 28,
      "last_message_preview": "Based on the lead's responses, I would classify...",
      "history_file": "~/.acti/sessions/history/lead_qualifier.jsonl"
    },
    "appointment_scheduler": {
      "agent_name": "appointment_scheduler",
      "agent_config_path": "outputs/agents/appointment_scheduler.json",
      "last_active_at": "2026-01-09T15:30:00Z",
      "message_count": 17,
      "last_message_preview": "I've scheduled the meeting for tomorrow at 2pm...",
      "history_file": "~/.acti/sessions/history/appointment_scheduler.jsonl"
    }
  }
}
```

### Current Session Pointer (`current.json`)

```json
{
  "schema_version": "1.0.0",
  "current_session_id": "550e8400-e29b-41d4-a716-446655440000",
  "last_updated": "2026-01-09T16:30:00Z"
}
```

### Conversation History Format (`{context}.jsonl`)

Append-only JSONL format for efficient streaming writes:

```jsonl
{"id":"msg-001","timestamp":"2026-01-09T14:00:01Z","role":"human","content":"Create a lead qualification agent","agent_name":null}
{"id":"msg-002","timestamp":"2026-01-09T14:00:15Z","role":"assistant","content":"I'll create a lead qualification agent...","agent_name":null,"tool_calls":[{"name":"list_available_tools","args":{}}],"model":"anthropic:claude-opus-4-5-20251101","duration_ms":14200}
{"id":"msg-003","timestamp":"2026-01-09T14:00:16Z","role":"tool","content":"Available tools: voice, websearch...","tool_results":[{"name":"list_available_tools","result":"..."}]}
{"id":"msg-004","timestamp":"2026-01-09T14:00:45Z","role":"assistant","content":"SUCCESS: Agent 'lead_qualifier' created!","agent_name":null,"model":"anthropic:claude-opus-4-5-20251101","duration_ms":29000}
```

---

## Function Specifications

### Session Lifecycle Functions

#### 1. `create_session`

Creates a new session with optional configuration.

```python
def create_session(
    config: SessionConfig | None = None,
    session_id: str | None = None,
) -> Session:
```

**Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `config` | `SessionConfig` | No | `SessionConfig()` | Session configuration |
| `session_id` | `str` | No | Auto-generated UUID | Custom session ID |

**Return Value:**

Returns a new `Session` instance with:
- Unique session ID (generated or provided)
- Default or provided configuration
- Empty metrics and agent contexts
- Metadata initialized with current timestamp

**Behavior:**

1. Generates UUID if `session_id` not provided
2. Creates session directory if it doesn't exist
3. Initializes empty conversation histories
4. Does NOT save to disk (call `save_session` explicitly)

**Example:**

```python
from src.session import create_session, SessionConfig

# Create with defaults
session = create_session()

# Create with custom config
config = SessionConfig(
    auto_save=True,
    save_interval_seconds=60,
    max_history_messages=500,
)
session = create_session(config=config)
```

---

#### 2. `save_session`

Persists session state to disk.

```python
async def save_session(
    session: Session,
    *,
    force: bool = False,
) -> SaveResult:
```

**Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `session` | `Session` | Yes | - | Session to save |
| `force` | `bool` | No | `False` | Save even if no changes |

**Return Value:**

```python
@dataclass
class SaveResult:
    success: bool
    session_file: Path
    bytes_written: int
    error: Optional[str] = None
```

**Behavior:**

1. Skips save if `session.has_unsaved_changes` is False (unless `force=True`)
2. Writes session JSON atomically (write to temp, then rename)
3. Updates `current.json` to point to this session
4. Marks session as clean
5. Returns result with file path and bytes written

**Error Handling:**

| Error Condition | Behavior |
|-----------------|----------|
| Disk full | Returns `SaveResult(success=False, error="...")` |
| Permission denied | Returns `SaveResult(success=False, error="...")` |
| Invalid session state | Raises `SessionError` |

**Example:**

```python
from src.session import save_session

result = await save_session(session)
if result.success:
    print(f"Saved to: {result.session_file}")
else:
    print(f"Save failed: {result.error}")

# Force save even without changes
await save_session(session, force=True)
```

---

#### 3. `load_session`

Loads a session from disk.

```python
async def load_session(
    session_id: str | None = None,
    *,
    load_history: bool = True,
) -> Session | None:
```

**Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `session_id` | `str` | No | Current session | Session ID to load |
| `load_history` | `bool` | No | `True` | Load conversation histories |

**Return Value:**

Returns `Session` if found and valid, `None` otherwise.

**Behavior:**

1. If `session_id` is None, reads `current.json` for most recent session
2. Loads session JSON file
3. Validates schema version compatibility
4. Optionally loads conversation histories from JSONL files
5. Increments `metadata.resumed_count`
6. Returns session (does NOT restore active agent - call `restore_session` for that)

**Example:**

```python
from src.session import load_session

# Load most recent session
session = await load_session()
if session:
    print(f"Resumed session {session.session_id}")
else:
    print("No session to resume, creating new")
    session = create_session()

# Load specific session
session = await load_session("550e8400-e29b-41d4-a716-446655440000")
```

---

#### 4. `restore_session`

Fully restores a session including active agent.

```python
async def restore_session(
    session: Session,
) -> RestoreResult:
```

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `session` | `Session` | Yes | Session to restore |

**Return Value:**

```python
@dataclass
class RestoreResult:
    success: bool
    builder_restored: bool
    agent_restored: bool
    agent_name: Optional[str] = None
    errors: List[str] = field(default_factory=list)
```

**Behavior:**

1. Creates builder agent via `create_builder()`
2. If session had an active agent and `config.restore_active_agent=True`:
   - Loads agent config from disk
   - Creates agent with MCP tools via `create_agent_with_tools()`
   - Restores conversation history
   - Sets session mode to AGENT
3. Returns result indicating what was restored

**Example:**

```python
from src.session import load_session, restore_session

session = await load_session()
if session:
    result = await restore_session(session)
    if result.agent_restored:
        print(f"Restored agent: {result.agent_name}")
```

---

#### 5. `delete_session`

Deletes a session and its data.

```python
async def delete_session(
    session_id: str,
    *,
    delete_history: bool = True,
) -> bool:
```

**Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `session_id` | `str` | Yes | - | Session to delete |
| `delete_history` | `bool` | No | `True` | Also delete conversation histories |

**Return Value:**

Returns `True` if session was deleted, `False` if not found.

**Example:**

```python
from src.session import delete_session

# Delete session and history
deleted = await delete_session("550e8400-...")

# Delete session but keep history
deleted = await delete_session("550e8400-...", delete_history=False)
```

---

### Conversation History Functions

#### 6. `append_message`

Appends a message to conversation history.

```python
async def append_message(
    session: Session,
    message: ConversationMessage,
    context: str | None = None,
) -> None:
```

**Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `session` | `Session` | Yes | - | Active session |
| `message` | `ConversationMessage` | Yes | - | Message to append |
| `context` | `str` | No | Current context | "builder" or agent name |

**Behavior:**

1. Determines context from session mode if not provided
2. Appends message to in-memory history
3. If `config.history_enabled`, appends to JSONL file
4. Updates metrics counters
5. Marks session dirty

**Example:**

```python
from src.session import append_message, ConversationMessage
from datetime import datetime
import uuid

message = ConversationMessage(
    id=str(uuid.uuid4()),
    timestamp=datetime.now(),
    role="human",
    content="Create a research agent",
)

await append_message(session, message)
```

---

#### 7. `get_history`

Retrieves conversation history for a context.

```python
async def get_history(
    session: Session,
    context: str | None = None,
    *,
    limit: int | None = None,
    since: datetime | None = None,
) -> List[ConversationMessage]:
```

**Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `session` | `Session` | Yes | - | Active session |
| `context` | `str` | No | Current context | "builder" or agent name |
| `limit` | `int` | No | All | Max messages to return |
| `since` | `datetime` | No | All | Only messages after this time |

**Return Value:**

Returns list of `ConversationMessage` objects, ordered oldest to newest.

**Example:**

```python
from src.session import get_history
from datetime import datetime, timedelta

# Get all builder history
history = await get_history(session, "builder")

# Get last 10 messages from current agent
history = await get_history(session, limit=10)

# Get messages from last hour
since = datetime.now() - timedelta(hours=1)
history = await get_history(session, since=since)
```

---

#### 8. `clear_history`

Clears conversation history for a context.

```python
async def clear_history(
    session: Session,
    context: str | None = None,
    *,
    before: datetime | None = None,
) -> int:
```

**Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `session` | `Session` | Yes | - | Active session |
| `context` | `str` | No | Current context | "builder" or agent name |
| `before` | `datetime` | No | Clear all | Only clear messages before this time |

**Return Value:**

Returns number of messages cleared.

**Example:**

```python
from src.session import clear_history
from datetime import datetime, timedelta

# Clear all history for current context
cleared = await clear_history(session)

# Clear builder history
cleared = await clear_history(session, "builder")

# Clear history older than 7 days
cutoff = datetime.now() - timedelta(days=7)
cleared = await clear_history(session, before=cutoff)
```

---

#### 9. `export_history`

Exports conversation history to various formats.

```python
async def export_history(
    session: Session,
    context: str | None = None,
    *,
    format: Literal["json", "jsonl", "markdown", "txt"] = "json",
    output_path: Path | None = None,
) -> str | Path:
```

**Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `session` | `Session` | Yes | - | Active session |
| `context` | `str` | No | Current context | "builder" or agent name |
| `format` | `str` | No | `"json"` | Export format |
| `output_path` | `Path` | No | Return as string | File path to write |

**Return Value:**

Returns exported content as string, or writes to file and returns path.

**Example:**

```python
from src.session import export_history
from pathlib import Path

# Export as JSON string
json_str = await export_history(session, format="json")

# Export as Markdown to file
path = await export_history(
    session,
    context="lead_qualifier",
    format="markdown",
    output_path=Path("conversation.md"),
)
```

---

### Agent Context Functions

#### 10. `switch_to_agent`

Switches session context to a specific agent.

```python
async def switch_to_agent(
    session: Session,
    agent_name: str,
    *,
    preserve_builder: bool = True,
) -> SwitchResult:
```

**Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `session` | `Session` | Yes | - | Active session |
| `agent_name` | `str` | Yes | - | Agent to switch to |
| `preserve_builder` | `bool` | No | `True` | Keep builder history |

**Return Value:**

```python
@dataclass
class SwitchResult:
    success: bool
    previous_context: Optional[str]
    new_context: str
    agent_loaded: bool
    history_restored: bool
    error: Optional[str] = None
```

**Behavior:**

1. Preserves current agent context (if any) to `agent_contexts`
2. Loads target agent configuration
3. Creates agent with MCP tools
4. Restores target agent's conversation history
5. Updates session mode and active agent
6. Updates metrics (agent_switches, agents_loaded)

**Example:**

```python
from src.session import switch_to_agent

result = await switch_to_agent(session, "lead_qualifier")
if result.success:
    print(f"Switched from {result.previous_context} to {result.new_context}")
    if result.history_restored:
        print("Conversation history restored")
```

---

#### 11. `switch_to_builder`

Switches session back to builder mode.

```python
async def switch_to_builder(
    session: Session,
    *,
    preserve_agent: bool = True,
) -> SwitchResult:
```

**Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `session` | `Session` | Yes | - | Active session |
| `preserve_agent` | `bool` | No | `True` | Preserve agent context for later |

**Behavior:**

1. Preserves current agent context to `agent_contexts`
2. Sets session mode to BUILDER
3. Clears active agent (keeps builder)
4. Restores builder conversation history

**Example:**

```python
from src.session import switch_to_builder

result = await switch_to_builder(session)
print(f"Switched from {result.previous_context} to builder")
```

---

#### 12. `get_agent_contexts`

Lists all preserved agent contexts.

```python
def get_agent_contexts(
    session: Session,
) -> List[AgentContext]:
```

**Return Value:**

Returns list of `AgentContext` objects for all agents used in session.

**Example:**

```python
from src.session import get_agent_contexts

contexts = get_agent_contexts(session)
for ctx in contexts:
    print(f"{ctx.agent_name}: {ctx.message_count} messages, last active {ctx.last_active_at}")
```

---

### Session Query Functions

#### 13. `list_sessions`

Lists all available sessions.

```python
async def list_sessions(
    *,
    limit: int = 10,
    include_metadata: bool = True,
) -> List[SessionSummary]:
```

**Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `limit` | `int` | No | `10` | Max sessions to return |
| `include_metadata` | `bool` | No | `True` | Include full metadata |

**Return Value:**

```python
@dataclass
class SessionSummary:
    session_id: str
    created_at: datetime
    last_activity_at: datetime
    mode: SessionMode
    active_agent_name: Optional[str]
    agents_used: List[str]
    is_current: bool                 # Is this the current session?
```

**Example:**

```python
from src.session import list_sessions

sessions = await list_sessions(limit=5)
for s in sessions:
    current = "[CURRENT]" if s.is_current else ""
    print(f"{s.session_id[:8]}... {s.last_activity_at} {current}")
```

---

#### 14. `get_session_stats`

Returns aggregated statistics across sessions.

```python
async def get_session_stats() -> SessionStats:
```

**Return Value:**

```python
@dataclass
class SessionStats:
    total_sessions: int
    total_messages: int
    total_agents_created: int
    total_tool_calls: int
    most_used_agents: List[tuple[str, int]]  # (name, usage_count)
    most_used_tools: List[tuple[str, int]]   # (name, call_count)
    total_active_time: timedelta
    storage_used_bytes: int
```

**Example:**

```python
from src.session import get_session_stats

stats = await get_session_stats()
print(f"Total sessions: {stats.total_sessions}")
print(f"Total messages: {stats.total_messages}")
print(f"Storage used: {stats.storage_used_bytes / 1024:.1f} KB")
```

---

### Auto-Save Functions

#### 15. `start_auto_save`

Starts background auto-save task.

```python
async def start_auto_save(
    session: Session,
) -> asyncio.Task:
```

**Behavior:**

1. Creates background task that saves session at configured interval
2. Only saves if session has unsaved changes
3. Handles errors gracefully (logs, doesn't crash)
4. Returns task handle for cancellation

**Example:**

```python
from src.session import start_auto_save

auto_save_task = await start_auto_save(session)

# Later, to stop auto-save:
auto_save_task.cancel()
```

---

#### 16. `stop_auto_save`

Stops background auto-save task.

```python
async def stop_auto_save(
    task: asyncio.Task,
    *,
    save_before_stop: bool = True,
) -> None:
```

**Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `task` | `asyncio.Task` | Yes | - | Auto-save task handle |
| `save_before_stop` | `bool` | No | `True` | Final save before stopping |

---

## CLI Integration

### New Commands

| Command | Description |
|---------|-------------|
| `/session` | Show current session info |
| `/session list` | List recent sessions |
| `/session new` | Start a new session |
| `/session resume [id]` | Resume a specific session |
| `/session export [format]` | Export session/history |
| `/history` | Show conversation history |
| `/history clear` | Clear current history |
| `/history export` | Export current history |

### Session Display in Status

```
You: /status

Session Status
============================================================

Session: 550e8400 (active 2h 15m)
  Created: 2026-01-09 14:00:00
  Last saved: 2 minutes ago
  Unsaved changes: No

Mode: AGENT
Active Agent: lead_qualifier
  Loaded: 45 minutes ago
  Messages: 28

Previous Contexts:
  appointment_scheduler (17 messages, inactive 1h)

Metrics:
  Messages sent: 57 (12 builder, 45 agent)
  Agents used: 2
  Tool calls: 23 (websearch: 15, communication: 8)

============================================================
```

### Startup Flow

```
CLI Start
    |
    v
+-------------------+
| Check for session |
| config.auto_resume|
+--------+----------+
         |
    [auto_resume?]
         |
   Yes   |   No
    +----+----+
    |         |
    v         v
+-------+ +--------+
| Load  | | Create |
| last  | | new    |
| session| | session|
+---+---+ +---+----+
    |         |
    v         v
+-------------------+
| restore_session() |
| - Create builder  |
| - Restore agent?  |
+--------+----------+
         |
         v
+-------------------+
| start_auto_save() |
+--------+----------+
         |
         v
     [Main Loop]
         |
    (on exit)
         |
         v
+-------------------+
| stop_auto_save()  |
| save_session()    |
+-------------------+
```

---

## Exception Classes

```python
class SessionError(Exception):
    """Base exception for session management errors."""
    pass

class SessionNotFoundError(SessionError):
    """Session file not found."""
    def __init__(self, session_id: str):
        self.session_id = session_id
        super().__init__(f"Session '{session_id}' not found")

class SessionCorruptError(SessionError):
    """Session file is corrupted or invalid."""
    def __init__(self, session_id: str, reason: str):
        self.session_id = session_id
        self.reason = reason
        super().__init__(f"Session '{session_id}' is corrupted: {reason}")

class SessionVersionError(SessionError):
    """Session schema version incompatible."""
    def __init__(self, session_id: str, version: str, supported: str):
        self.session_id = session_id
        self.version = version
        self.supported = supported
        super().__init__(
            f"Session '{session_id}' has incompatible version {version} "
            f"(supported: {supported})"
        )

class HistoryError(SessionError):
    """Error with conversation history."""
    pass

class HistoryNotFoundError(HistoryError):
    """History file not found."""
    def __init__(self, context: str):
        self.context = context
        super().__init__(f"History not found for context '{context}'")
```

---

## Module Exports

```python
__all__ = [
    # Type definitions
    "SessionMode",
    "SessionConfig",
    "SessionMetadata",
    "SessionMetrics",
    "ConversationMessage",
    "ConversationHistory",
    "AgentContext",
    "Session",

    # Result types
    "SaveResult",
    "RestoreResult",
    "SwitchResult",
    "SessionSummary",
    "SessionStats",

    # Session lifecycle
    "create_session",
    "save_session",
    "load_session",
    "restore_session",
    "delete_session",

    # Conversation history
    "append_message",
    "get_history",
    "clear_history",
    "export_history",

    # Agent context
    "switch_to_agent",
    "switch_to_builder",
    "get_agent_contexts",

    # Session queries
    "list_sessions",
    "get_session_stats",

    # Auto-save
    "start_auto_save",
    "stop_auto_save",

    # Exceptions
    "SessionError",
    "SessionNotFoundError",
    "SessionCorruptError",
    "SessionVersionError",
    "HistoryError",
    "HistoryNotFoundError",
]
```

---

## Testing Patterns

### Unit Test: Session Creation

```python
import pytest
from src.session import create_session, SessionConfig

def test_create_session_with_defaults():
    """Session created with default configuration."""
    session = create_session()

    assert session.session_id is not None
    assert session.mode == SessionMode.BUILDER
    assert session.config.auto_save is True
    assert session.has_unsaved_changes is False

def test_create_session_with_custom_config():
    """Session respects custom configuration."""
    config = SessionConfig(
        auto_save=False,
        max_history_messages=100,
    )
    session = create_session(config=config)

    assert session.config.auto_save is False
    assert session.config.max_history_messages == 100
```

### Unit Test: Session Persistence

```python
import pytest
from pathlib import Path
from src.session import create_session, save_session, load_session

@pytest.mark.asyncio
async def test_session_save_and_load(tmp_path):
    """Session can be saved and loaded."""
    config = SessionConfig(session_dir=tmp_path / "sessions")
    session = create_session(config=config)
    session.mark_dirty()

    # Save
    result = await save_session(session)
    assert result.success
    assert result.session_file.exists()

    # Load
    loaded = await load_session(session.session_id)
    assert loaded is not None
    assert loaded.session_id == session.session_id

@pytest.mark.asyncio
async def test_session_skip_save_when_clean(tmp_path):
    """Save skipped when session has no changes."""
    config = SessionConfig(session_dir=tmp_path / "sessions")
    session = create_session(config=config)

    result = await save_session(session)
    assert result.success is True  # First save always happens

    result = await save_session(session)
    # Skipped because no changes
```

### Unit Test: Conversation History

```python
import pytest
from datetime import datetime
import uuid
from src.session import (
    create_session,
    append_message,
    get_history,
    clear_history,
    ConversationMessage,
)

@pytest.mark.asyncio
async def test_append_and_get_history():
    """Messages can be appended and retrieved."""
    session = create_session()

    msg = ConversationMessage(
        id=str(uuid.uuid4()),
        timestamp=datetime.now(),
        role="human",
        content="Hello",
    )

    await append_message(session, msg, "builder")

    history = await get_history(session, "builder")
    assert len(history) == 1
    assert history[0].content == "Hello"

@pytest.mark.asyncio
async def test_history_limit():
    """History respects limit parameter."""
    session = create_session()

    for i in range(10):
        msg = ConversationMessage(
            id=str(uuid.uuid4()),
            timestamp=datetime.now(),
            role="human",
            content=f"Message {i}",
        )
        await append_message(session, msg, "builder")

    history = await get_history(session, "builder", limit=5)
    assert len(history) == 5
```

### Unit Test: Agent Context Switching

```python
import pytest
from unittest.mock import AsyncMock, patch
from src.session import create_session, switch_to_agent, switch_to_builder

@pytest.mark.asyncio
async def test_switch_to_agent():
    """Can switch to agent and preserve context."""
    session = create_session()

    with patch("src.session.create_agent_with_tools", new_callable=AsyncMock):
        result = await switch_to_agent(session, "test_agent")

    assert result.success
    assert session.mode == SessionMode.AGENT
    assert session.active_agent_name == "test_agent"

@pytest.mark.asyncio
async def test_switch_preserves_context():
    """Switching preserves previous context."""
    session = create_session()

    # Switch to first agent
    with patch("src.session.create_agent_with_tools", new_callable=AsyncMock):
        await switch_to_agent(session, "agent_a")

    # Add some messages
    for i in range(5):
        msg = ConversationMessage(
            id=str(uuid.uuid4()),
            timestamp=datetime.now(),
            role="human",
            content=f"Message {i}",
        )
        await append_message(session, msg)

    # Switch to second agent
    with patch("src.session.create_agent_with_tools", new_callable=AsyncMock):
        await switch_to_agent(session, "agent_b")

    # Verify first agent context preserved
    assert "agent_a" in session.agent_contexts
    assert session.agent_contexts["agent_a"].message_count == 5
```

### Integration Test: Full Session Lifecycle

```python
import pytest
from src.session import (
    create_session,
    save_session,
    load_session,
    restore_session,
    append_message,
    switch_to_agent,
    ConversationMessage,
    SessionConfig,
)

@pytest.mark.asyncio
@pytest.mark.integration
async def test_full_session_lifecycle(tmp_path):
    """Integration test for complete session lifecycle."""
    config = SessionConfig(session_dir=tmp_path / "sessions")

    # 1. Create session
    session = create_session(config=config)
    original_id = session.session_id

    # 2. Add some builder messages
    for content in ["Create an agent", "I'll create that for you"]:
        msg = ConversationMessage(
            id=str(uuid.uuid4()),
            timestamp=datetime.now(),
            role="human" if "Create" in content else "assistant",
            content=content,
        )
        await append_message(session, msg, "builder")

    # 3. Switch to agent
    with patch("src.session.create_agent_with_tools", new_callable=AsyncMock):
        await switch_to_agent(session, "test_agent")

    # 4. Save session
    result = await save_session(session)
    assert result.success

    # 5. Load session (simulating restart)
    loaded = await load_session(original_id)
    assert loaded is not None
    assert loaded.session_id == original_id

    # 6. Restore session
    with patch("src.session.create_builder") as mock_builder:
        with patch("src.session.create_agent_with_tools", new_callable=AsyncMock):
            restore_result = await restore_session(loaded)

    assert restore_result.builder_restored
    assert restore_result.agent_restored
    assert restore_result.agent_name == "test_agent"
```

---

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `ACTI_SESSION_DIR` | No | `~/.acti/sessions` | Session storage directory |
| `ACTI_AUTO_SAVE` | No | `true` | Enable auto-save |
| `ACTI_HISTORY_ENABLED` | No | `true` | Enable conversation history |
| `ACTI_AUTO_RESUME` | No | `true` | Auto-resume last session |

---

## Configuration File (`~/.acti/config.yaml`)

```yaml
session:
  auto_save: true
  save_interval_seconds: 30
  session_dir: ~/.acti/sessions

  history:
    enabled: true
    max_messages: 1000
    retention_days: 30

  resume:
    auto_resume: true
    restore_active_agent: true

  metrics:
    track_metrics: true

  display:
    show_session_info: true
    prompt_on_unsaved: true
```

---

## Migration Notes

### From Current Session Class

The existing `Session` dataclass in `cli.py` will be replaced by the new `Session` class. Migration path:

1. **Mode**: `SessionMode.BUILDER/AGENT` already matches
2. **builder_messages/agent_messages**: Migrate to `ConversationHistory` objects
3. **active_agent/active_agent_name**: Keep as-is (runtime-only)
4. **New additions**: config, metadata, metrics, agent_contexts

### Backward Compatibility

- CLI continues to work without any session files
- First run creates `~/.acti/sessions/` directory
- Old CLI history (if any) is NOT migrated automatically

---

## Security Considerations

1. **File Permissions**: All session files created with `0600` permissions
2. **No Credentials**: Session files never contain API keys or credentials
3. **Local Storage**: All data stored locally; no network transmission
4. **Cleanup**: `history_retention_days` prevents unbounded growth
5. **Sensitive Content**: Users should be aware conversation content is stored

---

## Performance Considerations

1. **JSONL for History**: Append-only format enables efficient streaming writes
2. **Lazy Loading**: History loaded on-demand, not at session start
3. **Incremental Saves**: Only save when dirty, not on every change
4. **Background Auto-Save**: Non-blocking saves via asyncio
5. **Message Limits**: `max_history_messages` prevents memory issues

---

## Version History

- **1.0.0** (2026-01-09): Initial API contract
  - Session lifecycle functions (create, save, load, restore, delete)
  - Conversation history management
  - Multi-agent context switching
  - Auto-save functionality
  - Session metrics and statistics
  - CLI integration patterns

---

## API Design Report

### Spec Files Created/Updated
- `docs/api-contract-session.md` - Complete session management API (16 functions, 12 types)

### Core Design Decisions

1. **Separation of State and Behavior**: Session state is persisted; runtime objects (agents) are recreated on load. This avoids serialization complexity while maintaining full resume capability.

2. **JSONL for History**: Using append-only JSONL files for conversation history enables efficient writes without rewriting entire files, and allows streaming reads for large histories.

3. **Agent Context Preservation**: When switching agents, context is preserved to `agent_contexts` map, enabling seamless return to previous conversations.

4. **Atomic Saves**: Session files written atomically (temp file + rename) to prevent corruption on crashes or power loss.

5. **Privacy-First Persistence**: All data stored locally with restricted file permissions; no cloud sync or telemetry.

### Authentication & Security
- Method: None (local tool, no network auth)
- File permissions: `0600` for files, `0700` for directories
- No credentials stored in session files
- Conversation content stored locally only

### Open Questions

1. **History Encryption**: Should we offer optional encryption for conversation histories?
2. **Session Sharing**: Should we support exporting/importing sessions for team collaboration?
3. **Cloud Sync**: Should we provide optional cloud backup of sessions?
4. **Compression**: Should we compress old history files to save space?

### Implementation Guidance

1. **Phase 1**: Implement core session persistence (`create_session`, `save_session`, `load_session`)
2. **Phase 2**: Add conversation history management (`append_message`, `get_history`)
3. **Phase 3**: Implement agent context switching with history preservation
4. **Phase 4**: Add auto-save background task
5. **Phase 5**: Add session CLI commands (`/session`, `/history`)

Dependencies:
- No new external packages required
- Uses stdlib `json`, `pathlib`, `asyncio`, `uuid`, `datetime`
- Integrates with existing `create_builder()` and `create_agent_with_tools()`
