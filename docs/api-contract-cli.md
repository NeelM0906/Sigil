# API Contract: src/cli.py

## Overview

This document defines the API contract for `src/cli.py`, the interactive CLI module for the ACTi Agent Builder (Sigil). This module provides a chat-like interface where users can create agents via natural language, automatically instantiate them with MCP tools, and interact with active agents in real-time.

The CLI serves as the primary user interface for Phase 2.5, bridging the builder agent (Phase 1) with MCP tool execution (Phase 2).

---

## Design Principles

1. **Mode-Based Interaction**: Clear distinction between builder mode (creating agents) and agent mode (chatting with active agents)
2. **Seamless Transitions**: Automatic mode switching when agents are created or loaded
3. **Real-Time Feedback**: Streaming output for long-running operations with progress indicators
4. **Graceful Error Handling**: Clear error messages with actionable recovery suggestions
5. **Session Persistence**: State management across commands with optional persistence to disk
6. **Unix Philosophy**: Commands follow predictable patterns; output is parseable when needed

---

## Module Dependencies

```python
from src.builder import create_builder
from src.mcp_integration import (
    create_agent_with_tools,
    validate_agent_tools,
    MCPIntegrationError,
)
from src.schemas import AgentConfig
from src.tools import get_agent_config, list_created_agents, create_agent_config
```

### External Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `prompt_toolkit` | >=3.0.0 | Rich terminal input with history, completion |
| `rich` | >=13.0.0 | Colored output, progress indicators, panels |
| `asyncio` | stdlib | Async execution for MCP operations |

### Optional Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `readline` | stdlib | Fallback input handling if prompt_toolkit unavailable |

---

## Type Definitions

### `CLIMode`

Enum representing the current interaction mode.

```python
from enum import Enum, auto

class CLIMode(Enum):
    """Current interaction mode for the CLI."""
    BUILDER = auto()   # Interacting with the builder agent
    AGENT = auto()     # Interacting with an active agent
    MODIFY = auto()    # Modifying an existing agent's configuration
```

### `ModificationMode`

Sub-modes within MODIFY mode for different modification workflows.

```python
class ModificationMode(str, Enum):
    """Sub-modes within MODIFY mode."""
    INTERACTIVE = "interactive"   # Conversational modification with multiple turns
    DIRECT = "direct"             # Single command with inline changes
    BATCH = "batch"               # Multiple agents at once
```

### `SessionState`

Dataclass containing the current session state.

```python
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Any
from langgraph.graph.state import CompiledStateGraph

@dataclass
class SessionState:
    """Persistent session state for the CLI."""

    # Current interaction mode
    mode: CLIMode = CLIMode.BUILDER

    # Active agent (if in AGENT mode)
    active_agent: Optional[CompiledStateGraph] = None
    active_agent_name: Optional[str] = None
    active_agent_config: Optional[AgentConfig] = None

    # Builder agent (always available)
    builder: Optional[CompiledStateGraph] = None

    # Agent being modified (if in MODIFY mode)
    modify_target: Optional[str] = None
    modify_mode: Optional[ModificationMode] = None
    modify_original_config: Optional[dict] = None      # Original config before any changes
    modify_pending_changes: list[FieldChange] = field(default_factory=list)  # Staged changes

    # Conversation history per context
    builder_history: list = field(default_factory=list)
    agent_history: list = field(default_factory=list)

    # Session metadata
    session_id: str = ""
    started_at: datetime = field(default_factory=datetime.now)
    last_activity: datetime = field(default_factory=datetime.now)

    def enter_modify_mode(
        self,
        agent_name: str,
        original_config: dict,
        mode: ModificationMode = ModificationMode.INTERACTIVE,
    ) -> None:
        """Enter modification mode for an agent."""
        self.mode = CLIMode.MODIFY
        self.modify_target = agent_name
        self.modify_mode = mode
        self.modify_original_config = original_config.copy()
        self.modify_pending_changes = []

    def exit_modify_mode(self) -> None:
        """Exit modification mode and clear state."""
        self.mode = CLIMode.BUILDER
        self.modify_target = None
        self.modify_mode = None
        self.modify_original_config = None
        self.modify_pending_changes = []

    def add_pending_change(self, change: "FieldChange") -> None:
        """Add a change to the pending changes list."""
        self.modify_pending_changes.append(change)

    def clear_pending_changes(self) -> None:
        """Clear all pending changes without exiting modify mode."""
        self.modify_pending_changes = []

    def has_pending_changes(self) -> bool:
        """Check if there are uncommitted changes."""
        return len(self.modify_pending_changes) > 0
```

### `CommandResult`

Result of executing a CLI command.

```python
from dataclasses import dataclass
from typing import Optional, Any

@dataclass
class CommandResult:
    """Result of a CLI command execution."""

    success: bool
    message: str
    data: Optional[Any] = None

    # Mode transition (if command changes mode)
    new_mode: Optional[CLIMode] = None

    # For streaming output
    is_streaming: bool = False
```

### `CLIConfig`

Configuration options for the CLI.

```python
from dataclasses import dataclass
from pathlib import Path

@dataclass
class CLIConfig:
    """Configuration for CLI behavior."""

    # Display settings
    show_tool_calls: bool = True        # Show MCP tool invocations
    stream_output: bool = True          # Stream responses in real-time
    color_output: bool = True           # Enable colored terminal output

    # Behavior settings
    auto_instantiate: bool = True       # Auto-instantiate agents after creation
    auto_switch_to_agent: bool = True   # Switch to agent mode after creation
    confirm_delete: bool = True         # Require confirmation for delete

    # Persistence settings
    save_session: bool = False          # Persist session state to disk
    session_dir: Path = Path("~/.acti/sessions").expanduser()

    # Timeouts
    mcp_timeout: float = 30.0           # MCP connection timeout (seconds)
    response_timeout: float = 120.0     # Agent response timeout (seconds)
```

---

## Command Specifications

### Command Registry

All commands follow the pattern `/<command> [args...]`. The command parser extracts the command name and arguments.

```python
COMMANDS: dict[str, CommandSpec] = {
    "create": CommandSpec(
        name="create",
        usage="/create <prompt>",
        description="Create a new agent from natural language description",
        handler=handle_create,
        min_args=1,
        variadic=True,  # Accepts multi-word prompts
    ),
    "list": CommandSpec(
        name="list",
        usage="/list",
        description="List all saved agents",
        handler=handle_list,
        min_args=0,
    ),
    "load": CommandSpec(
        name="load",
        usage="/load <name>",
        description="Load and activate an existing agent",
        handler=handle_load,
        min_args=1,
    ),
    "run": CommandSpec(
        name="run",
        usage="/run <name> <task>",
        description="Run a specific agent with a task (one-shot execution)",
        handler=handle_run,
        min_args=2,
        variadic=True,
    ),
    "modify": CommandSpec(
        name="modify",
        usage="/modify <name> [changes] | /modify --all [changes] | /modify --filter <filter> [changes]",
        description="Modify agent(s) via natural language or interactive mode",
        handler=handle_modify,
        min_args=1,
        variadic=True,
        flags=["--all", "--filter", "--agents", "--add-tag", "--remove-tag", "--yes"],
    ),
    "rollback": CommandSpec(
        name="rollback",
        usage="/rollback <name> [version] [--yes]",
        description="Restore an agent to its previous version from backup",
        handler=handle_rollback,
        min_args=1,
        flags=["--yes"],
    ),
    "history": CommandSpec(
        name="history",
        usage="/history <name>",
        description="Show modification history for an agent",
        handler=handle_history,
        min_args=1,
    ),
    "delete": CommandSpec(
        name="delete",
        usage="/delete <name>",
        description="Delete an agent configuration",
        handler=handle_delete,
        min_args=1,
    ),
    "status": CommandSpec(
        name="status",
        usage="/status",
        description="Show current session status and active agent info",
        handler=handle_status,
        min_args=0,
    ),
    "tools": CommandSpec(
        name="tools",
        usage="/tools",
        description="List available MCP tools and their status",
        handler=handle_tools,
        min_args=0,
    ),
    "builder": CommandSpec(
        name="builder",
        usage="/builder",
        description="Switch to builder mode (deactivate current agent)",
        handler=handle_builder,
        min_args=0,
    ),
    "help": CommandSpec(
        name="help",
        usage="/help [command]",
        description="Show help for all commands or a specific command",
        handler=handle_help,
        min_args=0,
    ),
    "exit": CommandSpec(
        name="exit",
        usage="/exit",
        description="Exit the CLI",
        handler=handle_exit,
        min_args=0,
        aliases=["quit", "q"],
    ),
}
```

---

## Command Handler Specifications

### 1. `/create <prompt>`

Creates a new agent from a natural language description.

#### Signature

```python
async def handle_create(
    session: SessionState,
    args: list[str],
    config: CLIConfig,
) -> CommandResult:
```

#### Behavior

1. Concatenates args into a single prompt string
2. Invokes the builder agent with the prompt
3. Builder creates an AgentConfig and saves to `outputs/agents/{name}.json`
4. If `auto_instantiate=True`: Instantiates the agent with MCP tools
5. If `auto_switch_to_agent=True`: Switches session to AGENT mode
6. Returns success with agent summary

#### Flow Diagram

```
User: /create research agent that finds market trends

    +------------------+
    |  Parse Command   |
    +--------+---------+
             |
             v
    +------------------+
    |  Invoke Builder  |
    |  (with prompt)   |
    +--------+---------+
             |
             v
    +------------------+
    | Builder creates  |
    | AgentConfig      |
    | (uses tools)     |
    +--------+---------+
             |
             v
    +------------------+
    | Save to disk:    |
    | outputs/agents/  |
    | market_trends_   |
    | researcher.json  |
    +--------+---------+
             |
    [auto_instantiate?]
             |
     Yes     |     No
      +------+------+
      |             |
      v             v
+------------+ +------------+
| Connect to | | Return     |
| MCP servers| | config     |
| (websearch)| | info only  |
+-----+------+ +-----+------+
      |              |
      v              |
+------------+       |
| Create     |       |
| deepagent  |       |
+-----+------+       |
      |              |
      v              |
[auto_switch?]       |
      |              |
  Yes |  No          |
   +--+--+           |
   |     |           |
   v     v           v
+-----+ +-----+ +-------+
|AGENT| |BUILD| |SUCCESS|
|mode | |mode | |message|
+-----+ +-----+ +-------+
```

#### Example Output

```
You: /create research agent that finds market trends and writes reports

Builder: I'll create a market research agent for you.

[Analyzing request...]
[Classifying stratum: RTI (Reality & Truth Intelligence)]
[Selecting tools: websearch]
[Generating system prompt...]
[Creating configuration...]

SUCCESS: Agent 'market_trends_researcher' created!

  Name: market_trends_researcher
  Description: Researches market trends and compiles analytical reports
  Tools: websearch
  Stratum: RTI
  Model: anthropic:claude-opus-4-5-20251101

  Saved to: outputs/agents/market_trends_researcher.json

[Instantiating agent with MCP tools...]
[Connected to: websearch (Tavily)]

Agent 'market_trends_researcher' is now active.
What would you like it to research?
```

#### Error Handling

| Error Condition | Response |
|-----------------|----------|
| Empty prompt | "ERROR: Please provide a description. Usage: /create <prompt>" |
| Builder fails | "ERROR: Failed to create agent: {error}. Try rephrasing your request." |
| Save fails | "ERROR: Failed to save configuration: {error}" |
| MCP connection fails | "WARNING: Agent created but MCP tools unavailable: {error}. Agent saved but not instantiated." |

---

### 2. `/list`

Lists all saved agent configurations.

#### Signature

```python
async def handle_list(
    session: SessionState,
    args: list[str],
    config: CLIConfig,
) -> CommandResult:
```

#### Behavior

1. Reads all JSON files from `outputs/agents/`
2. Parses each configuration and extracts summary info
3. Groups agents by stratum for organization
4. Marks the currently active agent (if any)
5. Returns formatted list

#### Example Output

```
You: /list

Saved Agents
============================================================

[RTI] Reality & Truth Intelligence (2 agents)
------------------------------------------------------------
  market_trends_researcher          [ACTIVE]
    Tools: websearch
    Created: 2025-01-09T14:30:22Z

  fact_checker
    Tools: websearch
    Created: 2025-01-09T10:15:00Z

[ZACS] Zone Action & Conversion Systems (1 agent)
------------------------------------------------------------
  appointment_scheduler
    Tools: calendar, communication
    Created: 2025-01-08T16:45:30Z

============================================================
Total: 3 agent(s)

Commands:
  /load <name>   - Activate an agent
  /run <name> <task> - Run agent with task
  /delete <name> - Delete an agent
```

---

### 3. `/load <name>`

Loads an existing agent and activates it for conversation.

#### Signature

```python
async def handle_load(
    session: SessionState,
    args: list[str],
    config: CLIConfig,
) -> CommandResult:
```

#### Behavior

1. Validates agent name exists in `outputs/agents/`
2. Loads AgentConfig from JSON file
3. Validates MCP tools are available
4. Instantiates agent with `create_agent_with_tools()`
5. Updates session state to AGENT mode
6. Clears previous agent history (optional)
7. Returns success with agent info

#### Example Output

```
You: /load appointment_scheduler

Loading agent 'appointment_scheduler'...
[Validating tools: calendar, communication]
[Connecting to MCP servers...]
[Connected: calendar (Google Calendar), communication (Twilio)]

Agent 'appointment_scheduler' activated.
Mode: AGENT
Tools: calendar, communication

You can now chat with this agent. Type /builder to switch back.
```

#### Error Handling

| Error Condition | Response |
|-----------------|----------|
| Agent not found | "ERROR: Agent 'xyz' not found. Use /list to see available agents." |
| Missing credentials | "ERROR: Cannot load agent - missing credentials for 'calendar'. Set GOOGLE_OAUTH_CREDENTIALS." |
| MCP connection fails | "ERROR: Failed to connect to MCP server for 'calendar': {error}" |

---

### 4. `/run <name> <task>`

Executes a one-shot task with a specific agent without switching modes.

#### Signature

```python
async def handle_run(
    session: SessionState,
    args: list[str],
    config: CLIConfig,
) -> CommandResult:
```

#### Behavior

1. First arg is agent name, remaining args form the task
2. Loads and instantiates the agent (temporary)
3. Executes the task with streaming output
4. Returns result without changing session mode
5. Does NOT persist as active agent

#### Example Output

```
You: /run market_trends_researcher Find the top 5 AI trends this week

[Loading agent: market_trends_researcher]
[Executing task...]

Agent Response:
------------------------------------------------------------
Based on my research, here are the top 5 AI trends this week:

1. **Multimodal AI Assistants** - Major tech companies announced...
2. **AI Agents in Production** - Enterprise adoption of...
3. **Edge AI Deployment** - New hardware optimizations...
4. **Retrieval-Augmented Generation** - RAG techniques...
5. **AI Safety Regulations** - EU AI Act implementation...

[Tool calls: tavily-search (3 calls)]
------------------------------------------------------------

(Mode unchanged: still in BUILDER mode)
```

---

### 5. `/modify <name>`

Enters modification mode for an existing agent.

#### Signature

```python
async def handle_modify(
    session: SessionState,
    args: list[str],
    config: CLIConfig,
) -> CommandResult:
```

#### Behavior

1. Loads the agent configuration
2. Displays current configuration summary
3. Switches to MODIFY mode
4. User describes changes in natural language
5. Builder generates updated configuration
6. User confirms changes
7. Saves updated configuration (optionally with version history)

#### Modification Flow

```
You: /modify appointment_scheduler

Entering modification mode for 'appointment_scheduler'
------------------------------------------------------------

Current Configuration:
  Name: appointment_scheduler
  Description: Schedules appointments via calendar
  Tools: calendar, communication
  Stratum: ZACS

Describe the changes you want to make (natural language):

You: Add voice capabilities so it can make confirmation calls

Builder: I'll update the agent to include voice capabilities.

Proposed Changes:
  - Added tool: voice (ElevenLabs)
  - Updated description: "Schedules appointments and makes voice confirmation calls"
  - Enhanced system prompt with voice interaction guidelines

Apply these changes? [y/N]: y

SUCCESS: Agent 'appointment_scheduler' updated!
  Version: 2 (previous version backed up)

Use /load appointment_scheduler to activate the updated agent.
```

---

## Enhanced Agent Modification Flow (Task 2.5.5)

This section specifies the enhanced modification capabilities for the CLI, providing natural language modification, interactive editing, diff previews, rollback support, and batch operations.

### Overview

The modification flow transforms the basic field-by-field editing into an intelligent, conversational system where the builder agent interprets user intent and applies changes with full transparency.

### Modification Modes

```python
class ModificationMode(str, Enum):
    """Sub-modes within MODIFY mode."""
    INTERACTIVE = "interactive"   # Conversational modification
    DIRECT = "direct"             # Single command with changes
    BATCH = "batch"               # Multiple agents at once
```

---

### 1. Natural Language Modification

Users can describe changes conversationally, and the builder interprets and applies them.

#### Command Syntax

```bash
# Direct modification with natural language
/modify <agent_name> <natural language changes>

# Examples
/modify my_agent make it more friendly and add websearch
/modify lead_qualifier increase timeout and make responses shorter
/modify scheduler add voice capabilities for call confirmations
```

#### Signature

```python
async def handle_modify_with_description(
    session: SessionState,
    agent_name: str,
    modification_prompt: str,
    config: CLIConfig,
) -> CommandResult:
```

#### Behavior

1. Load current agent configuration
2. Send modification prompt to builder agent with current config context
3. Builder interprets intent and generates proposed changes
4. Generate diff preview showing before/after
5. Prompt user for confirmation
6. On confirm: Apply changes, increment version, create backup
7. Return success with summary of changes

#### Flow Diagram

```
User: /modify my_agent make it friendlier and add websearch

    +------------------------+
    | Parse command          |
    | Extract: agent_name,   |
    | modification_prompt    |
    +-----------+------------+
                |
                v
    +------------------------+
    | Load current config    |
    | from outputs/agents/   |
    | {agent_name}.json      |
    +-----------+------------+
                |
                v
    +------------------------+
    | Invoke builder with:   |
    | - Current config       |
    | - Modification prompt  |
    | - Available tools      |
    +-----------+------------+
                |
                v
    +------------------------+
    | Builder generates      |
    | ModificationPlan:      |
    | - Field changes        |
    | - Reasoning            |
    | - New config           |
    +-----------+------------+
                |
                v
    +------------------------+
    | Generate diff preview  |
    | - Colorized changes    |
    | - Version increment    |
    | - Impact summary       |
    +-----------+------------+
                |
                v
    +------------------------+
    | Display to user        |
    | Prompt: Apply? [y/N]   |
    +-----------+------------+
                |
        +-------+-------+
        |               |
        v               v
   [y/yes]          [n/no]
        |               |
        v               v
    +--------+     +---------+
    | Create |     | Discard |
    | backup |     | changes |
    +----+---+     +----+----+
         |              |
         v              v
    +--------+     +---------+
    | Apply  |     | Return  |
    | changes|     | to mode |
    +----+---+     +---------+
         |
         v
    +--------+
    | Return |
    | success|
    +--------+
```

#### Example Output

```
You: /modify my_agent make it more friendly and add websearch

Loading agent 'my_agent'...

Builder: I'll update the agent to be friendlier and add web search capabilities.

Proposed Changes:
============================================================

  Description:
    - "Handles customer inquiries efficiently"
    + "Handles customer inquiries with a warm, friendly approach"

  Tools:
    - ["communication"]
    + ["communication", "websearch"]

  System Prompt (excerpt):
    - "You are a customer service agent. Be professional..."
    + "You are a warm and friendly customer service agent. Always greet
    +  customers by name when possible and maintain an approachable tone..."

  Stratum: RAI (unchanged)
  Model: anthropic:claude-opus-4-5-20251101 (unchanged)

============================================================
Version: 2 -> 3
Backup will be saved to: outputs/agents/my_agent.backup.json

Apply these changes? [y/N]: y

SUCCESS: Agent 'my_agent' updated!
  Version: 3
  Changes: description, tools, system_prompt
  Backup: outputs/agents/my_agent.backup.json

Use /load my_agent to activate the updated agent.
```

---

### 2. Interactive Modification Mode

Entering modification mode without a description starts an interactive session.

#### Command Syntax

```bash
# Enter interactive modification mode
/modify <agent_name>
```

#### Signature

```python
async def handle_modify_interactive(
    session: SessionState,
    agent_name: str,
    config: CLIConfig,
) -> CommandResult:
```

#### Behavior

1. Load agent configuration
2. Display current configuration summary
3. Switch session to MODIFY mode with `modify_target` set
4. User describes changes conversationally (multiple turns allowed)
5. Builder suggests modifications after each message
6. User can approve, reject, or refine suggestions
7. `/done` exits modification mode and prompts for final save
8. `/cancel` exits without saving any changes

#### Interactive Commands (within MODIFY mode)

| Command | Description |
|---------|-------------|
| `/show` | Display current config (with pending changes highlighted) |
| `/diff` | Show diff between original and pending changes |
| `/reset` | Discard all pending changes |
| `/apply` | Apply pending changes and continue editing |
| `/done` | Finalize and save all changes, exit MODIFY mode |
| `/cancel` | Discard all changes and exit MODIFY mode |

#### Example Session

```
You: /modify appointment_scheduler

Entering interactive modification mode for 'appointment_scheduler'
============================================================

Current Configuration:
  Name: appointment_scheduler
  Description: Schedules appointments via calendar
  Tools: calendar, communication
  Stratum: ZACS
  Version: 2

Describe the changes you want (or use /show, /diff, /done, /cancel):

[Modify: appointment_scheduler] > Add voice so it can make confirmation calls

Builder: I'll add voice capabilities for phone confirmations.

Suggested change #1:
  + Add tool: voice (ElevenLabs)
  + Update system prompt to include voice confirmation instructions

Accept this change? [y/n/refine]: y

Change accepted. Pending changes: 1

[Modify: appointment_scheduler] > Also make the confirmation messages more concise

Builder: I'll update the system prompt to emphasize brevity in confirmations.

Suggested change #2:
  ~ Modify system_prompt: Add instruction for concise confirmation messages
    "When confirming appointments, be brief and clear. State only:
     appointment time, location, and any preparation needed."

Accept this change? [y/n/refine]: y

Change accepted. Pending changes: 2

[Modify: appointment_scheduler] > /diff

Pending Changes Preview:
============================================================

  Tools:
    - ["calendar", "communication"]
    + ["calendar", "communication", "voice"]

  System Prompt:
    [+15 lines added for voice confirmation workflow]
    [+3 lines added for concise messaging guidelines]

============================================================
2 changes pending | Original version: 2 | New version: 3

[Modify: appointment_scheduler] > /done

Save all changes?
  - 2 modifications will be applied
  - Version: 2 -> 3
  - Backup: outputs/agents/appointment_scheduler.backup.json

Confirm? [y/N]: y

SUCCESS: Agent 'appointment_scheduler' updated!
  Applied: 2 changes
  Version: 3
  Backup saved.

(Exited modification mode)
```

---

### 3. Modification Preview & Diff

All modifications show a clear before/after diff with syntax highlighting.

#### Diff Format Specification

```python
@dataclass
class ModificationDiff:
    """Represents a diff between original and modified config."""

    field: str                      # Field being modified
    change_type: Literal["add", "remove", "modify"]
    old_value: Any                  # None for additions
    new_value: Any                  # None for removals
    context_lines: int = 3          # Lines of context for large fields

class DiffRenderer:
    """Renders diffs in various formats."""

    def render_terminal(self, diff: ModificationDiff) -> str:
        """Render diff with ANSI colors for terminal."""

    def render_plain(self, diff: ModificationDiff) -> str:
        """Render diff without colors (for logging)."""

    def render_json(self, diff: ModificationDiff) -> dict:
        """Render diff as JSON (for API responses)."""
```

#### Color Scheme for Diffs

```python
DIFF_COLORS = {
    "addition": Colors.GREEN,      # + lines
    "removal": Colors.RED,         # - lines
    "modification": Colors.YELLOW, # ~ lines
    "context": Colors.DIM,         # unchanged context
    "field_name": Colors.CYAN,     # field being changed
    "version": Colors.MAGENTA,     # version numbers
}
```

#### Diff Output Example

```
Modification Preview: appointment_scheduler
============================================================

  Field: description
  --------------------------------------------------------
  - Schedules appointments via calendar
  + Schedules appointments via calendar with voice confirmations

  Field: tools
  --------------------------------------------------------
  - ["calendar", "communication"]
  + ["calendar", "communication", "voice"]

  Field: system_prompt
  --------------------------------------------------------
  @@ -5,3 +5,12 @@
     You are an appointment scheduling assistant.
     Your role is to help users book appointments efficiently.
  +
  +  ## Voice Confirmation Guidelines
  +  When confirming appointments by phone:
  +  - Introduce yourself briefly
  +  - State the appointment details clearly
  +  - Ask for verbal confirmation
  +  - Thank them and end the call promptly
  +
  +  Keep all voice messages under 30 seconds.

============================================================
Version: 2 -> 3
Changes: 3 fields modified
```

---

### 4. Rollback Support

Automatic backup creation and rollback capability for recovering previous versions.

#### Backup Strategy

```python
BACKUP_LOCATION = "outputs/agents/{name}.backup.json"
BACKUP_HISTORY_DIR = "outputs/agents/.history/{name}/"  # Optional multi-version

class BackupManager:
    """Manages agent configuration backups."""

    def create_backup(self, agent_name: str) -> Path:
        """Create backup before modification."""

    def restore_backup(self, agent_name: str) -> bool:
        """Restore from most recent backup."""

    def list_backups(self, agent_name: str) -> list[BackupInfo]:
        """List available backups for an agent."""

    def cleanup_old_backups(self, agent_name: str, keep: int = 5):
        """Remove old backups, keeping the most recent N."""
```

#### Backup File Format

```json
{
  "_backup_metadata": {
    "backup_created_at": "2025-01-09T16:30:00Z",
    "backed_up_version": 2,
    "reason": "pre_modification",
    "modified_by": "cli_modify"
  },
  "name": "appointment_scheduler",
  "description": "Schedules appointments via calendar",
  "system_prompt": "...",
  "tools": ["calendar", "communication"],
  "model": "anthropic:claude-opus-4-5-20251101",
  "stratum": "ZACS",
  "metadata": {
    "created_at": "2025-01-08T10:00:00Z",
    "updated_at": "2025-01-09T14:00:00Z",
    "version": 2,
    "execution_count": 15,
    "tags": ["scheduling", "production"]
  }
}
```

#### `/rollback` Command

```bash
# Restore previous version
/rollback <agent_name>

# Rollback with confirmation bypass (for scripts)
/rollback <agent_name> --yes
```

#### Signature

```python
async def handle_rollback(
    session: SessionState,
    agent_name: str,
    config: CLIConfig,
    force: bool = False,
) -> CommandResult:
```

#### Behavior

1. Check if backup exists for agent
2. Load backup and current config
3. Display diff showing what will be restored
4. Prompt for confirmation (unless `force=True`)
5. Swap current config with backup
6. Create new backup of the version being replaced
7. Return success with restored version info

#### Example Output

```
You: /rollback appointment_scheduler

Rollback Preview: appointment_scheduler
============================================================

Current Version: 3 (modified 2 hours ago)
Backup Version: 2 (backed up 2 hours ago)

Changes that will be UNDONE:
  - Remove tool: voice
  - Revert system_prompt to previous version
  - Revert description to previous version

============================================================
WARNING: This will replace version 3 with version 2.
A backup of version 3 will be created before rollback.

Proceed with rollback? [y/N]: y

SUCCESS: Agent 'appointment_scheduler' rolled back!
  Restored version: 2
  Previous version (3) backed up to: .history/appointment_scheduler/v3_20250109_163000.json

Use /load appointment_scheduler to activate the restored agent.
```

---

### 5. Batch Modifications

Apply the same modification to multiple agents at once.

#### Command Syntax

```bash
# Modify all agents
/modify --all <modification_description>

# Modify agents matching a filter
/modify --filter stratum=ZACS <modification>
/modify --filter tag=production <modification>
/modify --filter tools=calendar <modification>

# Modify specific list of agents
/modify --agents agent1,agent2,agent3 <modification>

# Add tag to all agents
/modify --all --add-tag production
/modify --filter stratum=RTI --add-tag research

# Remove tag from agents
/modify --all --remove-tag deprecated
```

#### Signature

```python
async def handle_batch_modify(
    session: SessionState,
    targets: BatchTarget,
    modification: BatchModification,
    config: CLIConfig,
) -> CommandResult:

@dataclass
class BatchTarget:
    """Specifies which agents to modify."""
    all_agents: bool = False
    agent_names: list[str] = field(default_factory=list)
    filters: dict[str, str] = field(default_factory=dict)  # field=value

@dataclass
class BatchModification:
    """Specifies what modification to apply."""
    prompt: Optional[str] = None          # Natural language modification
    add_tags: list[str] = field(default_factory=list)
    remove_tags: list[str] = field(default_factory=list)
    add_tools: list[str] = field(default_factory=list)
    remove_tools: list[str] = field(default_factory=list)
    set_stratum: Optional[Stratum] = None
```

#### Behavior

1. Resolve target agents based on filters
2. Preview all affected agents
3. Generate modification plan for each agent
4. Show aggregate diff preview
5. Prompt for confirmation
6. Apply changes to each agent (with individual backups)
7. Return summary with success/failure counts

#### Example: Batch Tag Addition

```
You: /modify --all --add-tag production

Batch Modification Preview
============================================================

Targets: ALL agents (5 found)

Modification:
  + Add tag: "production"

Affected Agents:
  1. lead_qualifier (RAI) - tags: ["sales"]
     + Will add: production

  2. appointment_scheduler (ZACS) - tags: []
     + Will add: production

  3. market_researcher (RTI) - tags: ["research"]
     + Will add: production

  4. fact_checker (RTI) - tags: ["research"]
     + Will add: production

  5. compliance_auditor (IGE) - tags: ["compliance"]
     + Will add: production

============================================================
5 agents will be modified.
Backups will be created for all agents.

Proceed? [y/N]: y

Applying modifications...
  [1/5] lead_qualifier: SUCCESS (v2 -> v3)
  [2/5] appointment_scheduler: SUCCESS (v3 -> v4)
  [3/5] market_researcher: SUCCESS (v1 -> v2)
  [4/5] fact_checker: SUCCESS (v2 -> v3)
  [5/5] compliance_auditor: SUCCESS (v1 -> v2)

============================================================
Batch Modification Complete
  Success: 5
  Failed: 0
  Backups created: 5
```

#### Example: Batch Tool Addition with Filter

```
You: /modify --filter stratum=ZACS add voice capabilities

Batch Modification Preview
============================================================

Targets: Agents where stratum=ZACS (2 found)

Modification (via builder):
  + Add tool: voice
  + Update system_prompt with voice interaction guidelines

Affected Agents:
  1. appointment_scheduler
     Current tools: ["calendar", "communication"]
     + Add: voice
     + System prompt changes: +12 lines

  2. follow_up_agent
     Current tools: ["communication"]
     + Add: voice
     + System prompt changes: +12 lines

============================================================
2 agents will be modified.

Proceed? [y/N]: y

[Processing with builder...]

Batch Modification Complete
  Success: 2
  Failed: 0
```

---

### 6. Modification Schemas

#### `ModificationRequest`

```python
class ModificationRequest(BaseModel):
    """Request to modify an agent configuration."""

    agent_name: str = Field(
        ...,
        description="Name of the agent to modify",
    )
    modification_prompt: Optional[str] = Field(
        default=None,
        description="Natural language description of desired changes",
    )
    direct_changes: Optional[DirectChanges] = Field(
        default=None,
        description="Explicit field changes (alternative to prompt)",
    )

class DirectChanges(BaseModel):
    """Explicit changes to apply without builder interpretation."""

    description: Optional[str] = None
    system_prompt: Optional[str] = None
    tools: Optional[list[str]] = None
    stratum: Optional[Stratum] = None
    add_tags: list[str] = Field(default_factory=list)
    remove_tags: list[str] = Field(default_factory=list)
```

#### `ModificationPlan`

```python
class ModificationPlan(BaseModel):
    """Plan generated by builder for a modification."""

    agent_name: str
    changes: list[FieldChange]
    reasoning: str = Field(
        ...,
        description="Builder's explanation of why these changes were made",
    )
    new_config: AgentConfig
    original_config: AgentConfig
    version_increment: tuple[int, int]  # (old, new)

class FieldChange(BaseModel):
    """A single field change in a modification plan."""

    field: str
    change_type: Literal["add", "remove", "modify"]
    old_value: Any
    new_value: Any
    reason: Optional[str] = None
```

#### `ModificationResult`

```python
class ModificationResult(BaseModel):
    """Result of a modification operation."""

    success: bool
    agent_name: str
    changes_applied: list[str]  # List of field names changed
    old_version: int
    new_version: int
    backup_path: Optional[str] = None
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
```

---

### 7. Builder Tool: `modify_agent_natural`

New tool for the builder to perform natural language modifications.

```python
@tool
def modify_agent_natural(
    agent_name: str,
    modification_prompt: str,
    preview_only: bool = True,
) -> str:
    """Modify an agent configuration using natural language description.

    Use this tool to interpret a user's modification request and generate
    the appropriate changes to an agent's configuration. The builder will
    analyze the current configuration, understand the user's intent, and
    produce a modification plan.

    Args:
        agent_name: Name of the agent to modify (must exist).
        modification_prompt: Natural language description of desired changes.
            Examples:
            - "make it more friendly and conversational"
            - "add web search capabilities"
            - "reduce response length and make it more concise"
            - "add voice for phone confirmations"
        preview_only: If True, returns the modification plan without applying.
            If False, applies the changes immediately (use with caution).

    Returns:
        On preview_only=True: Detailed modification plan with diff preview.
        On preview_only=False: Confirmation of applied changes.
        On failure: Error message with guidance.

    Example:
        >>> modify_agent_natural(
        ...     agent_name="lead_qualifier",
        ...     modification_prompt="add websearch and make it ask more qualifying questions",
        ...     preview_only=True
        ... )
        "Modification Plan: lead_qualifier
        ========================================

        Reasoning:
        The user wants to enhance the agent's research capabilities and
        improve its qualification process...

        Proposed Changes:
        1. Add tool: websearch
        2. Update system_prompt: Add 5 additional qualifying questions...

        Preview diff:
        ..."
    """
```

---

### 8. Integration with Existing Commands

The enhanced modification flow integrates with existing CLI commands:

#### Updated Command Registry

```python
COMMANDS = {
    # ... existing commands ...

    "modify": CommandSpec(
        name="modify",
        usage="/modify <name> [changes] | /modify --all [changes] | /modify --filter <filter> [changes]",
        description="Modify agent(s) via natural language or interactive mode",
        handler=handle_modify,
        min_args=1,
        variadic=True,
        flags=["--all", "--filter", "--agents", "--add-tag", "--remove-tag", "--yes"],
    ),
    "rollback": CommandSpec(
        name="rollback",
        usage="/rollback <name> [--yes]",
        description="Restore an agent to its previous version from backup",
        handler=handle_rollback,
        min_args=1,
        flags=["--yes"],
    ),
    "history": CommandSpec(
        name="history",
        usage="/history <name>",
        description="Show modification history for an agent",
        handler=handle_history,
        min_args=1,
    ),
}
```

#### `/history` Command

```bash
/history <agent_name>
```

Shows the modification history including all backed-up versions.

```
You: /history appointment_scheduler

Modification History: appointment_scheduler
============================================================

Current: v4 (2025-01-09T16:30:00Z)
  Changes: Added voice tool, updated system prompt

v3 (2025-01-09T14:00:00Z) [BACKUP AVAILABLE]
  Changes: Added concise messaging guidelines

v2 (2025-01-08T10:00:00Z) [BACKUP AVAILABLE]
  Changes: Initial configuration

v1 (2025-01-08T09:00:00Z)
  Changes: Created by builder

============================================================
Backups stored in: outputs/agents/.history/appointment_scheduler/

Commands:
  /rollback appointment_scheduler     - Restore to v3
  /rollback appointment_scheduler v2  - Restore to specific version
```

---

### 9. Error Handling for Modifications

```python
class ModificationError(CLIError):
    """Base exception for modification errors."""
    pass

class NoChangesDetectedError(ModificationError):
    """Builder could not determine any changes from the prompt."""
    def __init__(self, prompt: str):
        super().__init__(
            f"Could not determine changes from: '{prompt}'\n\n"
            f"Try being more specific, e.g.:\n"
            f"  - 'add websearch tool'\n"
            f"  - 'make the tone more friendly'\n"
            f"  - 'increase the number of qualifying questions'"
        )

class BackupNotFoundError(ModificationError):
    """No backup exists for rollback."""
    def __init__(self, agent_name: str):
        super().__init__(
            f"No backup found for agent '{agent_name}'.\n\n"
            f"Backups are created automatically when modifications are made."
        )

class BatchModificationError(ModificationError):
    """Error during batch modification."""
    def __init__(self, successes: int, failures: list[tuple[str, str]]):
        failed_names = [f"  - {name}: {error}" for name, error in failures]
        super().__init__(
            f"Batch modification partially failed.\n\n"
            f"Successes: {successes}\n"
            f"Failures: {len(failures)}\n"
            + "\n".join(failed_names)
        )
```

---

### 10. Testing Patterns for Modification Flow

```python
@pytest.mark.asyncio
async def test_modify_natural_language():
    """Test natural language modification."""
    session = create_test_session()

    # Create test agent first
    await handle_create(session, ["test agent for modification"], CLIConfig())

    # Modify with natural language
    result = await handle_modify(
        session,
        ["test_agent", "add", "websearch", "and", "make", "it", "friendlier"],
        CLIConfig(),
    )

    assert result.success
    assert "websearch" in result.data["new_config"]["tools"]
    assert result.data["backup_path"] is not None

@pytest.mark.asyncio
async def test_modify_interactive_mode():
    """Test interactive modification mode."""
    session = create_test_session()

    # Enter modify mode
    result = await handle_modify(session, ["test_agent"], CLIConfig())

    assert session.mode == CLIMode.MODIFY
    assert session.modify_target == "test_agent"

@pytest.mark.asyncio
async def test_rollback():
    """Test rollback functionality."""
    session = create_test_session()

    # Create and modify agent
    await handle_create(session, ["test agent"], CLIConfig())
    await handle_modify(session, ["test_agent", "add websearch"], CLIConfig())

    # Rollback
    result = await handle_rollback(session, ["test_agent"], CLIConfig(), force=True)

    assert result.success
    config = load_agent_config("test_agent")
    assert "websearch" not in config["tools"]

@pytest.mark.asyncio
async def test_batch_modify_add_tag():
    """Test batch tag addition."""
    session = create_test_session()

    # Create multiple agents
    for i in range(3):
        await handle_create(session, [f"test agent {i}"], CLIConfig())

    # Batch add tag
    result = await handle_batch_modify(
        session,
        BatchTarget(all_agents=True),
        BatchModification(add_tags=["production"]),
        CLIConfig(),
    )

    assert result.success
    assert result.data["success_count"] == 3
```

---

### 6. `/delete <name>`

Deletes an agent configuration from disk.

#### Signature

```python
async def handle_delete(
    session: SessionState,
    args: list[str],
    config: CLIConfig,
) -> CommandResult:
```

#### Behavior

1. Validates agent exists
2. If `confirm_delete=True`: Prompts for confirmation
3. Removes JSON file from `outputs/agents/`
4. If deleted agent was active: Switches to BUILDER mode
5. Returns success confirmation

#### Example Output

```
You: /delete old_test_agent

Delete agent 'old_test_agent'? This cannot be undone. [y/N]: y

SUCCESS: Agent 'old_test_agent' deleted.
```

---

### 7. `/status`

Shows current session status and active agent information.

#### Signature

```python
async def handle_status(
    session: SessionState,
    args: list[str],
    config: CLIConfig,
) -> CommandResult:
```

#### Example Output

```
You: /status

Session Status
============================================================

Mode: AGENT
Active Agent: market_trends_researcher
  Description: Researches market trends and compiles reports
  Tools: websearch (connected)
  Stratum: RTI
  Loaded at: 2025-01-09T15:30:00Z

Session Info:
  Session ID: abc123
  Started: 2025-01-09T14:00:00Z
  Duration: 1h 30m
  Messages: 12 (builder), 8 (agent)

MCP Tool Status:
  websearch: CONNECTED (Tavily)
  voice: AVAILABLE (credentials set)
  calendar: UNAVAILABLE (missing GOOGLE_OAUTH_CREDENTIALS)
  communication: AVAILABLE (credentials set)
  crm: UNAVAILABLE (missing HUBSPOT_API_KEY)

============================================================
```

---

### 8. `/tools`

Lists available MCP tools with status and capabilities.

#### Signature

```python
async def handle_tools(
    session: SessionState,
    args: list[str],
    config: CLIConfig,
) -> CommandResult:
```

#### Example Output

```
You: /tools

MCP Tool Registry
============================================================

voice (ElevenLabs)
  Status: AVAILABLE
  Credentials: ELEVENLABS_API_KEY [SET]
  Capabilities:
    - Text-to-speech synthesis
    - Voice cloning
    - Real-time audio generation
  Best for: ZACS, RAI strata

websearch (Tavily)
  Status: CONNECTED [active]
  Credentials: TAVILY_API_KEY [SET]
  Capabilities:
    - Web search and research
    - Content extraction
    - Site crawling
  Best for: RTI, EEI strata

calendar (Google Calendar)
  Status: UNAVAILABLE
  Credentials: GOOGLE_OAUTH_CREDENTIALS [MISSING]
  Setup: Run `npx @anthropic-ai/google-calendar-mcp` to configure

communication (Twilio)
  Status: AVAILABLE
  Credentials: TWILIO_ACCOUNT_SID [SET]
               TWILIO_API_KEY [SET]
               TWILIO_API_SECRET [SET]
  Capabilities:
    - Send SMS messages
    - Make phone calls
    - Manage conversations
  Best for: RAI, ZACS strata

crm (HubSpot)
  Status: UNAVAILABLE
  Credentials: HUBSPOT_API_KEY [MISSING]

============================================================
Connected: 1 | Available: 2 | Unavailable: 2
```

---

### 9. `/builder`

Switches to builder mode, deactivating any active agent.

#### Signature

```python
async def handle_builder(
    session: SessionState,
    args: list[str],
    config: CLIConfig,
) -> CommandResult:
```

#### Example Output

```
You: /builder

Switched to BUILDER mode.
Agent 'market_trends_researcher' deactivated (session preserved).

You can now create new agents or modify existing ones.
Type /load <name> to reactivate an agent.
```

---

### 10. `/help [command]`

Shows help information for all commands or a specific command.

#### Signature

```python
async def handle_help(
    session: SessionState,
    args: list[str],
    config: CLIConfig,
) -> CommandResult:
```

#### Example Output (General)

```
You: /help

ACTi Agent Builder CLI - Help
============================================================

AGENT MANAGEMENT
  /create <prompt>     Create a new agent from natural language
  /list                List all saved agents
  /load <name>         Load and activate an agent
  /run <name> <task>   Execute a one-shot task with an agent
  /modify <name>       Modify an existing agent
  /delete <name>       Delete an agent

SESSION CONTROL
  /status              Show current session status
  /tools               List MCP tools and status
  /builder             Switch to builder mode

GENERAL
  /help [command]      Show this help or command-specific help
  /exit                Exit the CLI (aliases: /quit, /q)

DEFAULT BEHAVIOR
  When in BUILDER mode: Messages create new agents
  When in AGENT mode: Messages are sent to active agent

============================================================
For command-specific help: /help <command>
```

#### Example Output (Specific)

```
You: /help create

/create <prompt>
============================================================

Create a new agent from a natural language description.

USAGE
  /create <description of the agent you want>

EXAMPLES
  /create a lead qualification agent for B2B SaaS sales
  /create research assistant that finds and summarizes articles
  /create appointment scheduler with voice confirmation

BEHAVIOR
  1. Builder agent interprets your request
  2. Classifies the appropriate ACTi stratum
  3. Selects relevant MCP tools
  4. Generates a tailored system prompt
  5. Creates and saves the AgentConfig
  6. (Auto) Instantiates agent with MCP tools
  7. (Auto) Switches to AGENT mode

OPTIONS (via CLI config)
  auto_instantiate: true   - Automatically connect MCP tools
  auto_switch_to_agent: true - Switch to agent mode after creation

============================================================
```

---

### 11. `/exit`

Exits the CLI gracefully.

#### Signature

```python
async def handle_exit(
    session: SessionState,
    args: list[str],
    config: CLIConfig,
) -> CommandResult:
```

#### Behavior

1. If `save_session=True`: Persists session state to disk
2. Closes any active MCP connections
3. Displays farewell message
4. Exits with code 0

---

## Core Functions

### 1. `create_cli`

Factory function that creates a configured CLI instance.

#### Signature

```python
def create_cli(
    config: CLIConfig | None = None,
) -> CLI:
```

#### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `config` | `CLIConfig` | No | `CLIConfig()` | Configuration options for CLI behavior |

#### Return Value

Returns a `CLI` instance ready to run.

#### Example

```python
from src.cli import create_cli, CLIConfig

# Create with defaults
cli = create_cli()

# Create with custom config
config = CLIConfig(
    show_tool_calls=True,
    stream_output=True,
    auto_instantiate=True,
    mcp_timeout=60.0,
)
cli = create_cli(config)

# Run the CLI
cli.run()
```

---

### 2. `run_cli`

Main entry point that creates and runs the CLI.

#### Signature

```python
def run_cli(
    config: CLIConfig | None = None,
) -> int:
```

#### Return Value

Returns exit code: 0 for normal exit, non-zero for errors.

#### Usage

```python
import sys
from src.cli import run_cli

if __name__ == "__main__":
    sys.exit(run_cli())
```

---

### 3. `parse_command`

Parses user input to extract command and arguments.

#### Signature

```python
def parse_command(
    user_input: str,
) -> tuple[str | None, list[str]]:
```

#### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `user_input` | `str` | Raw user input string |

#### Return Value

Returns tuple of `(command_name, args)`. If input is not a command, returns `(None, [])`.

#### Behavior

```python
# Command inputs
parse_command("/create lead qualifier")  # ("create", ["lead", "qualifier"])
parse_command("/list")                   # ("list", [])
parse_command("/load my_agent")          # ("load", ["my_agent"])

# Non-command inputs (chat messages)
parse_command("Hello, agent!")           # (None, [])
parse_command("  /create  ")             # ("create", [])
```

---

### 4. `handle_chat_message`

Processes a non-command chat message based on current mode.

#### Signature

```python
async def handle_chat_message(
    session: SessionState,
    message: str,
    config: CLIConfig,
) -> AsyncIterator[str]:
```

#### Behavior by Mode

| Mode | Behavior |
|------|----------|
| BUILDER | Sends message to builder agent; may create new agent |
| AGENT | Sends message to active agent; agent responds with tool calls |
| MODIFY | Interprets message as modification request |

#### Streaming Output

```python
async for chunk in handle_chat_message(session, "Find AI trends"):
    print(chunk, end="", flush=True)
```

---

### 5. `stream_agent_response`

Streams an agent's response with optional tool call display.

#### Signature

```python
async def stream_agent_response(
    agent: CompiledStateGraph,
    messages: list,
    show_tool_calls: bool = True,
) -> AsyncIterator[StreamChunk]:
```

#### StreamChunk Type

```python
from dataclasses import dataclass
from typing import Literal

@dataclass
class StreamChunk:
    """A chunk of streamed output."""
    type: Literal["text", "tool_call", "tool_result", "error"]
    content: str
    metadata: dict | None = None
```

#### Example Usage

```python
async for chunk in stream_agent_response(agent, messages, show_tool_calls=True):
    if chunk.type == "text":
        print(chunk.content, end="", flush=True)
    elif chunk.type == "tool_call":
        print(f"\n[Calling: {chunk.content}]")
    elif chunk.type == "tool_result":
        print(f"\n[Result received]")
```

---

## Session Management

### Session Lifecycle

```
CLI Start
    |
    v
+-------------------+
| Initialize        |
| - Create builder  |
| - Load config     |
| - Check env vars  |
+--------+----------+
         |
         v
+-------------------+
| BUILDER Mode      |<-----------------------+
| (default)         |                        |
+--------+----------+                        |
         |                                   |
         | /create or /load                  | /builder
         v                                   |
+-------------------+                        |
| AGENT Mode        |                        |
| - Active agent    |------------------------+
| - MCP connected   |
+--------+----------+
         |
         | /modify
         v
+-------------------+
| MODIFY Mode       |
| - Edit config     |
| - Confirm changes |
+--------+----------+
         |
         | Save or cancel
         v
    (back to previous mode)
```

### Session Persistence Format

When `save_session=True`, session state is persisted to JSON:

```json
{
  "session_id": "abc123",
  "started_at": "2025-01-09T14:00:00Z",
  "last_activity": "2025-01-09T15:30:00Z",
  "mode": "AGENT",
  "active_agent_name": "market_trends_researcher",
  "builder_history_length": 12,
  "agent_history_length": 8,
  "config": {
    "show_tool_calls": true,
    "stream_output": true,
    "auto_instantiate": true
  }
}
```

Note: Full conversation history is NOT persisted by default (privacy/size concerns). Only metadata is saved.

---

## Output Formatting

### ANSI Color Scheme

```python
from rich.console import Console
from rich.theme import Theme

CLI_THEME = Theme({
    "prompt.builder": "bold cyan",
    "prompt.agent": "bold green",
    "prompt.modify": "bold yellow",
    "command": "bold blue",
    "success": "bold green",
    "error": "bold red",
    "warning": "bold yellow",
    "info": "dim",
    "tool_call": "italic magenta",
    "agent_name": "bold",
    "stratum.RTI": "blue",
    "stratum.RAI": "green",
    "stratum.ZACS": "yellow",
    "stratum.EEI": "cyan",
    "stratum.IGE": "magenta",
})
```

### Prompt Format

```python
def get_prompt(session: SessionState) -> str:
    """Generate the appropriate prompt based on session state."""
    if session.mode == CLIMode.BUILDER:
        return "[Builder] > "
    elif session.mode == CLIMode.AGENT:
        return f"[{session.active_agent_name}] > "
    elif session.mode == CLIMode.MODIFY:
        return f"[Modify: {session.modify_target}] > "
```

---

## Error Handling

### Error Categories

```python
class CLIError(Exception):
    """Base exception for CLI errors."""
    pass

class CommandNotFoundError(CLIError):
    """Unknown command."""
    def __init__(self, command: str):
        self.command = command
        super().__init__(f"Unknown command: /{command}. Type /help for available commands.")

class InvalidArgumentsError(CLIError):
    """Invalid or missing command arguments."""
    def __init__(self, command: str, message: str, usage: str):
        self.command = command
        self.usage = usage
        super().__init__(f"{message}\nUsage: {usage}")

class AgentNotFoundError(CLIError):
    """Agent configuration not found."""
    def __init__(self, name: str, available: list[str]):
        self.name = name
        self.available = available
        msg = f"Agent '{name}' not found."
        if available:
            msg += f" Available agents: {', '.join(available)}"
        super().__init__(msg)

class AgentInstantiationError(CLIError):
    """Failed to instantiate agent with MCP tools."""
    def __init__(self, name: str, original_error: Exception):
        self.name = name
        self.original_error = original_error
        super().__init__(f"Failed to instantiate agent '{name}': {original_error}")
```

### Recovery Suggestions

Every error includes actionable guidance:

```python
def format_error_with_recovery(error: CLIError) -> str:
    """Format error message with recovery suggestions."""

    if isinstance(error, CommandNotFoundError):
        return f"""
ERROR: {error}

Did you mean one of these?
{find_similar_commands(error.command)}

Type /help to see all available commands.
"""

    elif isinstance(error, AgentNotFoundError):
        return f"""
ERROR: {error}

To see all agents: /list
To create a new agent: /create <description>
"""

    elif isinstance(error, AgentInstantiationError):
        return f"""
ERROR: {error}

Possible causes:
  - Missing API credentials (check /tools for status)
  - MCP server not responding (check network)
  - Invalid agent configuration

Try:
  /tools                    - Check tool availability
  /status                   - Check session status
  /modify {error.name}      - Fix agent configuration
"""
```

---

## Integration Patterns

### Main Entry Point

```python
# src/cli.py

import asyncio
import sys
from pathlib import Path

def main() -> int:
    """Main entry point for the CLI."""
    try:
        return asyncio.run(_async_main())
    except KeyboardInterrupt:
        print("\nGoodbye!")
        return 0
    except Exception as e:
        print(f"Fatal error: {e}", file=sys.stderr)
        return 1

async def _async_main() -> int:
    """Async main function."""
    cli = create_cli()
    return await cli.run_async()

if __name__ == "__main__":
    sys.exit(main())
```

### Module Entry Point

```bash
# Run CLI as module
python -m src.cli

# Or via package entry point (pyproject.toml)
acti-builder
```

### pyproject.toml Configuration

```toml
[project.scripts]
acti-builder = "src.cli:main"
```

---

## Testing Patterns

### Unit Test: Command Parsing

```python
import pytest
from src.cli import parse_command

def test_parse_create_command():
    cmd, args = parse_command("/create lead qualifier agent")
    assert cmd == "create"
    assert args == ["lead", "qualifier", "agent"]

def test_parse_command_no_args():
    cmd, args = parse_command("/list")
    assert cmd == "list"
    assert args == []

def test_parse_non_command():
    cmd, args = parse_command("Hello agent!")
    assert cmd is None
    assert args == []

def test_parse_command_with_extra_spaces():
    cmd, args = parse_command("  /load   my_agent  ")
    assert cmd == "load"
    assert args == ["my_agent"]
```

### Unit Test: Session State

```python
import pytest
from src.cli import SessionState, CLIMode

def test_session_default_mode():
    session = SessionState()
    assert session.mode == CLIMode.BUILDER
    assert session.active_agent is None

def test_session_mode_transition():
    session = SessionState()
    session.mode = CLIMode.AGENT
    session.active_agent_name = "test_agent"

    assert session.mode == CLIMode.AGENT
    assert session.active_agent_name == "test_agent"
```

### Integration Test: Create Command

```python
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

@pytest.mark.asyncio
async def test_handle_create_success():
    """Test /create command creates and instantiates agent."""
    session = SessionState()
    session.builder = MagicMock()
    session.builder.ainvoke = AsyncMock(return_value={
        "messages": [MagicMock(content="SUCCESS: Agent 'test_agent' created!")]
    })

    config = CLIConfig(auto_instantiate=False)  # Skip MCP for unit test

    with patch("src.cli.create_agent_with_tools"):
        result = await handle_create(
            session,
            ["test", "agent", "description"],
            config,
        )

    assert result.success
    assert "test_agent" in result.message
```

### Integration Test: Full CLI Flow

```python
import pytest
from src.cli import create_cli, CLIConfig

@pytest.mark.asyncio
@pytest.mark.integration
async def test_cli_create_and_chat_flow():
    """Integration test for create -> chat flow."""
    config = CLIConfig(
        auto_instantiate=True,
        auto_switch_to_agent=True,
    )
    cli = create_cli(config)

    # Simulate /create command
    result = await cli.process_input("/create simple test agent")
    assert result.success
    assert cli.session.mode == CLIMode.AGENT

    # Simulate chat message
    result = await cli.process_input("Hello, agent!")
    assert result.success
```

---

## Module Exports

```python
__all__ = [
    # Main entry points
    "main",
    "run_cli",
    "create_cli",

    # Core classes
    "CLI",
    "CLIConfig",
    "SessionState",
    "CLIMode",
    "ModificationMode",
    "CommandResult",

    # Modification types
    "ModificationDiff",
    "ModificationPlan",
    "ModificationResult",
    "FieldChange",
    "BatchTarget",
    "BatchModification",
    "DirectChanges",
    "ModificationRequest",

    # Backup management
    "BackupManager",
    "BackupInfo",

    # Diff rendering
    "DiffRenderer",

    # Command handlers (for extension)
    "handle_create",
    "handle_list",
    "handle_load",
    "handle_run",
    "handle_modify",
    "handle_modify_interactive",
    "handle_modify_with_description",
    "handle_batch_modify",
    "handle_rollback",
    "handle_history",
    "handle_delete",
    "handle_status",
    "handle_tools",
    "handle_builder",
    "handle_help",
    "handle_exit",

    # Utilities
    "parse_command",
    "stream_agent_response",
    "handle_chat_message",

    # Exceptions
    "CLIError",
    "CommandNotFoundError",
    "InvalidArgumentsError",
    "AgentNotFoundError",
    "AgentInstantiationError",
    "ModificationError",
    "NoChangesDetectedError",
    "BackupNotFoundError",
    "BatchModificationError",
]
```

---

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `ANTHROPIC_API_KEY` | Yes | API key for Claude models |
| `ACTI_CLI_CONFIG` | No | Path to CLI config file (JSON/YAML) |
| `ACTI_SESSION_DIR` | No | Directory for session persistence |
| `NO_COLOR` | No | Disable colored output if set |

---

## Configuration File (Optional)

CLI can load configuration from `~/.acti/config.yaml`:

```yaml
# ~/.acti/config.yaml

cli:
  show_tool_calls: true
  stream_output: true
  color_output: true
  auto_instantiate: true
  auto_switch_to_agent: true
  confirm_delete: true

  timeouts:
    mcp: 30.0
    response: 120.0

  persistence:
    save_session: false
    session_dir: ~/.acti/sessions

builder:
  model: anthropic:claude-opus-4-5-20251101
  include_subagents: true

mcp:
  default_timeout: 30.0
  retry_attempts: 3
```

---

## Streaming Output (Task 2.5.6)

This section specifies the streaming output system for the CLI, providing token-by-token response streaming, tool call indicators, progress feedback, and multiple output modes.

### Overview

The streaming output system transforms the blocking output model into a real-time streaming architecture where:
- Agent responses appear token-by-token as they are generated
- Tool calls are displayed with live status indicators
- Progress feedback keeps users informed during long operations
- Multiple output modes support different use cases (interactive, scripted, verbose, quiet)

### Design Principles

1. **Progressive Disclosure**: Show information as it becomes available
2. **Non-Blocking UX**: Never leave users waiting without feedback
3. **Mode-Appropriate Output**: Adapt formatting to context (TTY vs pipe)
4. **Graceful Degradation**: Fall back to batch mode when streaming fails
5. **Structured Protocol**: All stream events follow a consistent schema

---

### 1. StreamChunk Protocol

All streaming output follows the `StreamChunk` protocol, enabling consistent handling across text, tool calls, and status updates.

#### StreamChunk Type Definitions

```python
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Literal, Optional


class StreamChunkType(str, Enum):
    """Types of chunks that can appear in a stream."""

    TEXT = "text"                      # Token/text content from agent
    THINKING = "thinking"              # Agent reasoning/thinking (if exposed)
    TOOL_CALL_START = "tool_call_start"  # Tool invocation beginning
    TOOL_CALL_END = "tool_call_end"      # Tool invocation completed
    TOOL_RESULT = "tool_result"        # Result from tool execution
    STATUS = "status"                  # Status message (progress, info)
    ERROR = "error"                    # Error occurred
    HEARTBEAT = "heartbeat"            # Keep-alive signal
    METADATA = "metadata"              # Stream metadata (timing, stats)


class ToolCallStatus(str, Enum):
    """Status of a tool call in progress."""

    PENDING = "pending"        # Tool call queued
    CALLING = "calling"        # Tool is being invoked
    SUCCESS = "success"        # Tool completed successfully
    ERROR = "error"            # Tool failed
    TIMEOUT = "timeout"        # Tool call timed out
    CANCELLED = "cancelled"    # Tool call was cancelled


@dataclass
class ToolCallInfo:
    """Information about a tool call."""

    tool_id: str                       # Unique identifier for this call
    tool_name: str                     # Name of the tool being called
    status: ToolCallStatus             # Current status
    arguments: dict[str, Any] = field(default_factory=dict)  # Tool arguments
    result: Optional[str] = None       # Result (when complete)
    error: Optional[str] = None        # Error message (if failed)
    started_at: Optional[str] = None   # ISO timestamp when started
    completed_at: Optional[str] = None # ISO timestamp when completed
    duration_ms: Optional[int] = None  # Duration in milliseconds


@dataclass
class StreamChunk:
    """A single chunk of streamed output.

    This is the atomic unit of the streaming protocol. All output
    from the streaming system is delivered as StreamChunks.
    """

    type: StreamChunkType              # What kind of chunk this is
    content: str = ""                  # Primary content (text, message, etc.)
    metadata: dict[str, Any] = field(default_factory=dict)  # Additional data

    # Optional type-specific data
    tool_call: Optional[ToolCallInfo] = None  # For tool_call_* types
    timestamp: str = field(
        default_factory=lambda: datetime.now().isoformat()
    )
    sequence: int = 0                  # Sequence number in stream

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        result = {
            "type": self.type.value,
            "content": self.content,
            "timestamp": self.timestamp,
            "sequence": self.sequence,
        }
        if self.metadata:
            result["metadata"] = self.metadata
        if self.tool_call:
            result["tool_call"] = {
                "tool_id": self.tool_call.tool_id,
                "tool_name": self.tool_call.tool_name,
                "status": self.tool_call.status.value,
                "arguments": self.tool_call.arguments,
                "result": self.tool_call.result,
                "error": self.tool_call.error,
                "duration_ms": self.tool_call.duration_ms,
            }
        return result
```

#### StreamChunk Examples

```python
# Text token from agent response
StreamChunk(
    type=StreamChunkType.TEXT,
    content="Based on my research, ",
    sequence=1,
)

# Agent thinking/reasoning (if model supports it)
StreamChunk(
    type=StreamChunkType.THINKING,
    content="I should search for recent market trends...",
    sequence=2,
    metadata={"thinking_type": "planning"},
)

# Tool call starting
StreamChunk(
    type=StreamChunkType.TOOL_CALL_START,
    content="Searching web for market trends",
    tool_call=ToolCallInfo(
        tool_id="tc_001",
        tool_name="tavily-search",
        status=ToolCallStatus.CALLING,
        arguments={"query": "AI market trends 2025"},
        started_at="2025-01-09T15:30:00Z",
    ),
    sequence=3,
)

# Tool call completed
StreamChunk(
    type=StreamChunkType.TOOL_CALL_END,
    content="Search completed",
    tool_call=ToolCallInfo(
        tool_id="tc_001",
        tool_name="tavily-search",
        status=ToolCallStatus.SUCCESS,
        arguments={"query": "AI market trends 2025"},
        result="Found 5 relevant articles...",
        completed_at="2025-01-09T15:30:02Z",
        duration_ms=2150,
    ),
    sequence=4,
)

# Status message
StreamChunk(
    type=StreamChunkType.STATUS,
    content="Processing results...",
    sequence=5,
    metadata={"progress": 0.75},
)

# Error chunk
StreamChunk(
    type=StreamChunkType.ERROR,
    content="Tool call timed out",
    tool_call=ToolCallInfo(
        tool_id="tc_002",
        tool_name="calendar-check",
        status=ToolCallStatus.TIMEOUT,
        error="Connection timed out after 30s",
    ),
    sequence=6,
)
```

---

### 2. Output Modes

The CLI supports four output modes that control how streaming content is displayed.

#### OutputMode Enum

```python
class OutputMode(str, Enum):
    """Output display modes for the CLI."""

    STREAM = "stream"    # Token-by-token streaming (default for TTY)
    BATCH = "batch"      # Wait for complete response (default for pipes)
    VERBOSE = "verbose"  # Streaming + expanded tool calls + timing
    QUIET = "quiet"      # Minimal output, final response only


@dataclass
class OutputConfig:
    """Configuration for output behavior."""

    mode: OutputMode = OutputMode.STREAM

    # Display options
    show_thinking: bool = False        # Show agent thinking/reasoning
    show_tool_calls: bool = True       # Show tool call indicators
    show_tool_results: bool = False    # Show full tool results (collapsed by default)
    show_timing: bool = False          # Show elapsed time
    show_progress: bool = True         # Show progress indicators

    # Formatting options
    color_output: bool = True          # Enable ANSI colors
    spinner_style: str = "dots"        # Spinner animation style
    max_tool_result_lines: int = 5     # Max lines for tool results (0 = unlimited)

    # Behavior options
    stream_buffer_size: int = 1        # Tokens to buffer before display (1 = immediate)
    heartbeat_interval: float = 5.0    # Seconds between heartbeats for long operations
    tool_timeout: float = 30.0         # Tool call timeout in seconds

    @classmethod
    def for_tty(cls) -> "OutputConfig":
        """Create config optimized for interactive terminal."""
        return cls(
            mode=OutputMode.STREAM,
            show_thinking=False,
            show_tool_calls=True,
            show_tool_results=False,
            show_timing=False,
            show_progress=True,
            color_output=True,
        )

    @classmethod
    def for_pipe(cls) -> "OutputConfig":
        """Create config optimized for piped/scripted usage."""
        return cls(
            mode=OutputMode.BATCH,
            show_thinking=False,
            show_tool_calls=False,
            show_tool_results=False,
            show_timing=False,
            show_progress=False,
            color_output=False,
        )

    @classmethod
    def verbose(cls) -> "OutputConfig":
        """Create config for verbose debugging output."""
        return cls(
            mode=OutputMode.VERBOSE,
            show_thinking=True,
            show_tool_calls=True,
            show_tool_results=True,
            show_timing=True,
            show_progress=True,
            color_output=True,
            max_tool_result_lines=0,  # Show all
        )

    @classmethod
    def quiet(cls) -> "OutputConfig":
        """Create config for minimal output."""
        return cls(
            mode=OutputMode.QUIET,
            show_thinking=False,
            show_tool_calls=False,
            show_tool_results=False,
            show_timing=False,
            show_progress=False,
            color_output=False,
        )
```

#### Mode Selection Logic

```python
def detect_output_mode() -> OutputMode:
    """Auto-detect appropriate output mode based on environment."""
    import os
    import sys

    # Check for explicit override
    mode_env = os.getenv("ACTI_OUTPUT_MODE", "").lower()
    if mode_env in ("stream", "batch", "verbose", "quiet"):
        return OutputMode(mode_env)

    # Check for verbose flag
    if "--verbose" in sys.argv or "-v" in sys.argv:
        return OutputMode.VERBOSE

    # Check for quiet flag
    if "--quiet" in sys.argv or "-q" in sys.argv:
        return OutputMode.QUIET

    # Auto-detect based on TTY
    if sys.stdout.isatty():
        return OutputMode.STREAM
    else:
        return OutputMode.BATCH


def get_output_config() -> OutputConfig:
    """Get output configuration based on detected mode and environment."""
    mode = detect_output_mode()

    if mode == OutputMode.STREAM:
        return OutputConfig.for_tty()
    elif mode == OutputMode.BATCH:
        return OutputConfig.for_pipe()
    elif mode == OutputMode.VERBOSE:
        return OutputConfig.verbose()
    elif mode == OutputMode.QUIET:
        return OutputConfig.quiet()
```

#### Mode Behaviors

| Mode | Token Streaming | Tool Calls | Progress | Timing | Colors | Use Case |
|------|-----------------|------------|----------|--------|--------|----------|
| STREAM | Yes | Inline indicators | Spinner | No | Yes | Interactive TTY |
| BATCH | No (wait) | Hidden | No | No | No | Pipes, scripts |
| VERBOSE | Yes | Expanded + results | Detailed | Yes | Yes | Debugging |
| QUIET | No (wait) | Hidden | No | No | No | Automation |

---

### 3. Token Streaming

Token-by-token streaming displays agent responses as they are generated.

#### StreamingRenderer Interface

```python
from abc import ABC, abstractmethod
from typing import AsyncIterator


class StreamingRenderer(ABC):
    """Abstract base for streaming output renderers."""

    def __init__(self, config: OutputConfig):
        self.config = config
        self._sequence = 0

    @abstractmethod
    async def render_chunk(self, chunk: StreamChunk) -> None:
        """Render a single chunk to output."""
        pass

    @abstractmethod
    async def start_stream(self) -> None:
        """Initialize streaming output."""
        pass

    @abstractmethod
    async def end_stream(self, metadata: dict) -> None:
        """Finalize streaming output."""
        pass


class TerminalRenderer(StreamingRenderer):
    """Renders streaming output to a terminal with colors and formatting."""

    def __init__(self, config: OutputConfig):
        super().__init__(config)
        self._current_tool_calls: dict[str, ToolCallInfo] = {}
        self._spinner_task: Optional[asyncio.Task] = None
        self._start_time: Optional[float] = None

    async def render_chunk(self, chunk: StreamChunk) -> None:
        """Render a chunk to the terminal."""

        if chunk.type == StreamChunkType.TEXT:
            await self._render_text(chunk.content)

        elif chunk.type == StreamChunkType.THINKING:
            if self.config.show_thinking:
                await self._render_thinking(chunk.content)

        elif chunk.type == StreamChunkType.TOOL_CALL_START:
            if self.config.show_tool_calls:
                await self._render_tool_start(chunk.tool_call)

        elif chunk.type == StreamChunkType.TOOL_CALL_END:
            if self.config.show_tool_calls:
                await self._render_tool_end(chunk.tool_call)

        elif chunk.type == StreamChunkType.TOOL_RESULT:
            if self.config.show_tool_results:
                await self._render_tool_result(chunk.tool_call, chunk.content)

        elif chunk.type == StreamChunkType.STATUS:
            if self.config.show_progress:
                await self._render_status(chunk.content, chunk.metadata)

        elif chunk.type == StreamChunkType.ERROR:
            await self._render_error(chunk.content, chunk.tool_call)

    async def _render_text(self, content: str) -> None:
        """Render text content inline."""
        sys.stdout.write(content)
        sys.stdout.flush()

    async def _render_thinking(self, content: str) -> None:
        """Render thinking/reasoning in dimmed style."""
        if self.config.color_output:
            sys.stdout.write(f"{Colors.DIM}[Thinking: {content}]{Colors.RESET}\n")
        else:
            sys.stdout.write(f"[Thinking: {content}]\n")
        sys.stdout.flush()

    async def _render_tool_start(self, tool_call: ToolCallInfo) -> None:
        """Render tool call start indicator."""
        self._current_tool_calls[tool_call.tool_id] = tool_call

        if self.config.color_output:
            indicator = f"\n{Colors.CYAN}[{tool_call.tool_name}]{Colors.RESET} "
            status = f"{Colors.YELLOW}calling...{Colors.RESET}"
        else:
            indicator = f"\n[{tool_call.tool_name}] "
            status = "calling..."

        sys.stdout.write(indicator + status)
        sys.stdout.flush()

        # Start spinner if configured
        if self.config.show_progress:
            await self._start_spinner(tool_call.tool_id)

    async def _render_tool_end(self, tool_call: ToolCallInfo) -> None:
        """Render tool call completion."""
        await self._stop_spinner()

        # Clear the "calling..." line and show result
        sys.stdout.write("\r" + " " * 60 + "\r")  # Clear line

        if tool_call.status == ToolCallStatus.SUCCESS:
            if self.config.color_output:
                status = f"{Colors.GREEN}done{Colors.RESET}"
            else:
                status = "done"
            duration = ""
            if self.config.show_timing and tool_call.duration_ms:
                duration = f" ({tool_call.duration_ms}ms)"
            sys.stdout.write(f"[{tool_call.tool_name}] {status}{duration}\n")

        elif tool_call.status == ToolCallStatus.ERROR:
            if self.config.color_output:
                status = f"{Colors.RED}error{Colors.RESET}"
            else:
                status = "error"
            sys.stdout.write(f"[{tool_call.tool_name}] {status}: {tool_call.error}\n")

        sys.stdout.flush()
        self._current_tool_calls.pop(tool_call.tool_id, None)

    async def _render_tool_result(
        self, tool_call: ToolCallInfo, result: str
    ) -> None:
        """Render tool result (when expanded)."""
        lines = result.split("\n")
        max_lines = self.config.max_tool_result_lines

        if max_lines > 0 and len(lines) > max_lines:
            lines = lines[:max_lines]
            lines.append(f"... ({len(lines) - max_lines} more lines)")

        if self.config.color_output:
            sys.stdout.write(f"{Colors.DIM}")

        for line in lines:
            sys.stdout.write(f"  | {line}\n")

        if self.config.color_output:
            sys.stdout.write(f"{Colors.RESET}")

        sys.stdout.flush()

    async def _render_status(self, message: str, metadata: dict) -> None:
        """Render status/progress message."""
        progress = metadata.get("progress")
        if progress is not None:
            progress_bar = self._create_progress_bar(progress)
            sys.stdout.write(f"\r{progress_bar} {message}")
        else:
            sys.stdout.write(f"\r{message}")
        sys.stdout.flush()

    async def _render_error(
        self, message: str, tool_call: Optional[ToolCallInfo]
    ) -> None:
        """Render error message."""
        await self._stop_spinner()

        if self.config.color_output:
            prefix = f"{Colors.RED}ERROR:{Colors.RESET}"
        else:
            prefix = "ERROR:"

        if tool_call:
            sys.stdout.write(f"\n{prefix} [{tool_call.tool_name}] {message}\n")
        else:
            sys.stdout.write(f"\n{prefix} {message}\n")
        sys.stdout.flush()

    def _create_progress_bar(self, progress: float, width: int = 20) -> str:
        """Create a text-based progress bar."""
        filled = int(width * progress)
        empty = width - filled
        if self.config.color_output:
            return f"{Colors.GREEN}[{'=' * filled}{' ' * empty}]{Colors.RESET}"
        else:
            return f"[{'=' * filled}{' ' * empty}]"

    async def _start_spinner(self, tool_id: str) -> None:
        """Start an animated spinner."""
        # Implementation uses asyncio task with spinner frames
        pass  # Detailed implementation omitted for brevity

    async def _stop_spinner(self) -> None:
        """Stop the animated spinner."""
        if self._spinner_task:
            self._spinner_task.cancel()
            self._spinner_task = None
```

#### Example: STREAM Mode Output

```
You: Find the top 3 AI trends this week

[tavily-search] calling...
[tavily-search] done (1250ms)

Based on my research, here are the top 3 AI trends this week:

1. **Multimodal AI Assistants** - Major tech companies announced new
   capabilities combining vision, audio, and text understanding...

2. **AI Agents in Production** - Enterprise adoption of autonomous
   AI agents has accelerated, with companies deploying agents for...

3. **Edge AI Deployment** - New hardware optimizations enable running
   larger models on mobile devices and edge servers...
```

#### Example: VERBOSE Mode Output

```
You: Find the top 3 AI trends this week

[Thinking: I should search for recent AI news and trends to provide current information]

[tavily-search] calling...
  | Query: "AI trends January 2025"
  | Max results: 5
[tavily-search] done (1250ms)
  | Found 5 results:
  | 1. "Multimodal AI Takes Center Stage" - TechCrunch
  | 2. "Enterprise AI Agents Go Mainstream" - VentureBeat
  | 3. "Edge AI Revolution: 2025 Predictions" - MIT Tech Review
  | 4. "AI Safety: New Frameworks Emerge" - Wired
  | 5. "RAG and Vector Databases Mature" - InfoWorld

Based on my research, here are the top 3 AI trends this week:

1. **Multimodal AI Assistants** - Major tech companies announced new
   capabilities combining vision, audio, and text understanding...

2. **AI Agents in Production** - Enterprise adoption of autonomous
   AI agents has accelerated, with companies deploying agents for...

3. **Edge AI Deployment** - New hardware optimizations enable running
   larger models on mobile devices and edge servers...

---
Elapsed: 3.2s | Tokens: 847 | Tool calls: 1
```

---

### 4. Tool Call Indicators

Tool calls are displayed with live status indicators showing the tool name, status, and results.

#### Tool Call Display States

```python
TOOL_CALL_INDICATORS = {
    ToolCallStatus.PENDING: {
        "symbol": "...",
        "color": Colors.DIM,
        "message": "queued",
    },
    ToolCallStatus.CALLING: {
        "symbol": ">>>",
        "color": Colors.YELLOW,
        "message": "calling...",
        "animated": True,
    },
    ToolCallStatus.SUCCESS: {
        "symbol": "OK",
        "color": Colors.GREEN,
        "message": "done",
    },
    ToolCallStatus.ERROR: {
        "symbol": "ERR",
        "color": Colors.RED,
        "message": "failed",
    },
    ToolCallStatus.TIMEOUT: {
        "symbol": "TMO",
        "color": Colors.RED,
        "message": "timed out",
    },
    ToolCallStatus.CANCELLED: {
        "symbol": "---",
        "color": Colors.DIM,
        "message": "cancelled",
    },
}
```

#### Inline vs Expanded Tool Calls

The `show_tool_results` config controls whether tool results are shown inline or collapsed.

**Collapsed (default, STREAM mode):**
```
[tavily-search] done (1250ms)
```

**Expanded (VERBOSE mode):**
```
[tavily-search] done (1250ms)
  | Query: "AI trends January 2025"
  | Results:
  |   1. "Multimodal AI Takes Center Stage"
  |   2. "Enterprise AI Agents Go Mainstream"
  |   3. "Edge AI Revolution: 2025 Predictions"
```

#### Concurrent Tool Calls

When multiple tools are called concurrently, they are displayed with distinct identifiers:

```
[tavily-search#1] calling...
[calendar-check#2] calling...
[tavily-search#1] done (1250ms)
[calendar-check#2] done (800ms)
```

---

### 5. Progress Feedback

Progress feedback keeps users informed during long operations.

#### Progress Indicators

```python
@dataclass
class ProgressIndicator:
    """Configuration for a progress indicator."""

    type: Literal["spinner", "bar", "text"]
    message: str = ""
    progress: Optional[float] = None  # 0.0 to 1.0 for determinate
    started_at: float = field(default_factory=time.time)

    @property
    def elapsed_seconds(self) -> float:
        return time.time() - self.started_at

    @property
    def elapsed_display(self) -> str:
        """Format elapsed time for display."""
        elapsed = self.elapsed_seconds
        if elapsed < 60:
            return f"{elapsed:.1f}s"
        elif elapsed < 3600:
            minutes = int(elapsed // 60)
            seconds = int(elapsed % 60)
            return f"{minutes}m {seconds}s"
        else:
            hours = int(elapsed // 3600)
            minutes = int((elapsed % 3600) // 60)
            return f"{hours}h {minutes}m"


class SpinnerStyle(str, Enum):
    """Available spinner animation styles."""

    DOTS = "dots"           # ... -> .. -> . -> .. -> ...
    BRAILLE = "braille"     # Braille pattern animation
    LINE = "line"           # | / - \ animation
    ARROW = "arrow"         # > -> >> -> >>> animation
    PULSE = "pulse"         # * -> ** -> *** animation


SPINNER_FRAMES = {
    SpinnerStyle.DOTS: ["   ", ".  ", ".. ", "...", ".. ", ".  "],
    SpinnerStyle.BRAILLE: ["", "", "", "", "", "", "", ""],
    SpinnerStyle.LINE: ["|", "/", "-", "\\"],
    SpinnerStyle.ARROW: [">  ", ">> ", ">>>", ">> "],
    SpinnerStyle.PULSE: ["*  ", "** ", "***", "** "],
}
```

#### Progress Display Examples

**Spinner (indeterminate operations):**
```
[tavily-search] calling... |
[tavily-search] calling... /
[tavily-search] calling... -
[tavily-search] calling... \
```

**Progress Bar (determinate operations):**
```
Processing results... [========          ] 40%
Processing results... [================  ] 80%
Processing results... [==================] 100%
```

**Elapsed Time (long operations):**
```
[calendar-check] calling... (5.2s elapsed)
```

#### Heartbeat Messages

For operations exceeding the heartbeat interval, status messages are emitted:

```python
async def emit_heartbeat(
    stream: AsyncIterator[StreamChunk],
    interval: float = 5.0,
) -> AsyncIterator[StreamChunk]:
    """Wrap a stream to emit heartbeat messages during gaps."""
    last_emit = time.time()

    async for chunk in stream:
        yield chunk
        last_emit = time.time()

        # Check if we need a heartbeat
        elapsed = time.time() - last_emit
        if elapsed > interval:
            yield StreamChunk(
                type=StreamChunkType.HEARTBEAT,
                content=f"Still processing... ({elapsed:.1f}s)",
            )
            last_emit = time.time()
```

---

### 6. Core Streaming Functions

#### `stream_agent_response`

Main function for streaming agent responses.

```python
async def stream_agent_response(
    agent: CompiledStateGraph,
    messages: list,
    config: OutputConfig,
) -> AsyncIterator[StreamChunk]:
    """Stream an agent's response with tool call handling.

    This is the primary interface for streaming agent output. It handles:
    - Token-by-token text streaming
    - Tool call start/end events
    - Error handling and recovery
    - Heartbeat emission for long operations

    Args:
        agent: The compiled agent graph (from deepagents).
        messages: Conversation messages to send to the agent.
        config: Output configuration controlling display behavior.

    Yields:
        StreamChunk objects representing each piece of output.

    Example:
        >>> async for chunk in stream_agent_response(agent, messages, config):
        ...     if chunk.type == StreamChunkType.TEXT:
        ...         print(chunk.content, end="", flush=True)
        ...     elif chunk.type == StreamChunkType.TOOL_CALL_START:
        ...         print(f"\\n[{chunk.tool_call.tool_name}] calling...")
    """
    sequence = 0
    tool_call_stack: dict[str, ToolCallInfo] = {}
    start_time = time.time()

    try:
        # Use LangGraph's astream_events for granular streaming
        async for event in agent.astream_events(
            {"messages": messages},
            version="v2",
        ):
            event_type = event.get("event")
            event_data = event.get("data", {})

            # Handle different event types
            if event_type == "on_chat_model_stream":
                # Token from LLM response
                content = event_data.get("chunk", {}).get("content", "")
                if content:
                    yield StreamChunk(
                        type=StreamChunkType.TEXT,
                        content=content,
                        sequence=sequence,
                    )
                    sequence += 1

            elif event_type == "on_tool_start":
                # Tool invocation starting
                tool_name = event.get("name", "unknown")
                tool_id = event.get("run_id", str(uuid.uuid4()))
                tool_input = event_data.get("input", {})

                tool_info = ToolCallInfo(
                    tool_id=tool_id,
                    tool_name=tool_name,
                    status=ToolCallStatus.CALLING,
                    arguments=tool_input,
                    started_at=datetime.now().isoformat(),
                )
                tool_call_stack[tool_id] = tool_info

                yield StreamChunk(
                    type=StreamChunkType.TOOL_CALL_START,
                    content=f"Calling {tool_name}",
                    tool_call=tool_info,
                    sequence=sequence,
                )
                sequence += 1

            elif event_type == "on_tool_end":
                # Tool invocation completed
                tool_id = event.get("run_id")
                tool_output = event_data.get("output", "")

                if tool_id in tool_call_stack:
                    tool_info = tool_call_stack[tool_id]
                    tool_info.status = ToolCallStatus.SUCCESS
                    tool_info.result = str(tool_output)
                    tool_info.completed_at = datetime.now().isoformat()

                    # Calculate duration
                    if tool_info.started_at:
                        start = datetime.fromisoformat(tool_info.started_at)
                        end = datetime.fromisoformat(tool_info.completed_at)
                        tool_info.duration_ms = int(
                            (end - start).total_seconds() * 1000
                        )

                    yield StreamChunk(
                        type=StreamChunkType.TOOL_CALL_END,
                        content=f"{tool_info.tool_name} completed",
                        tool_call=tool_info,
                        sequence=sequence,
                    )
                    sequence += 1

                    # Optionally yield tool result
                    if config.show_tool_results:
                        yield StreamChunk(
                            type=StreamChunkType.TOOL_RESULT,
                            content=str(tool_output),
                            tool_call=tool_info,
                            sequence=sequence,
                        )
                        sequence += 1

            elif event_type == "on_tool_error":
                # Tool invocation failed
                tool_id = event.get("run_id")
                error_msg = str(event_data.get("error", "Unknown error"))

                if tool_id in tool_call_stack:
                    tool_info = tool_call_stack[tool_id]
                    tool_info.status = ToolCallStatus.ERROR
                    tool_info.error = error_msg

                    yield StreamChunk(
                        type=StreamChunkType.ERROR,
                        content=error_msg,
                        tool_call=tool_info,
                        sequence=sequence,
                    )
                    sequence += 1

    except asyncio.TimeoutError:
        yield StreamChunk(
            type=StreamChunkType.ERROR,
            content="Response timed out",
            sequence=sequence,
        )

    except Exception as e:
        yield StreamChunk(
            type=StreamChunkType.ERROR,
            content=f"Streaming error: {str(e)}",
            sequence=sequence,
        )

    finally:
        # Emit final metadata
        elapsed_ms = int((time.time() - start_time) * 1000)
        yield StreamChunk(
            type=StreamChunkType.METADATA,
            content="Stream complete",
            metadata={
                "elapsed_ms": elapsed_ms,
                "total_chunks": sequence,
                "tool_calls": len(tool_call_stack),
            },
            sequence=sequence + 1,
        )
```

#### `render_stream`

High-level function that renders a stream using the appropriate renderer.

```python
async def render_stream(
    stream: AsyncIterator[StreamChunk],
    config: OutputConfig,
) -> StreamResult:
    """Render a stream of chunks to output.

    This is the main entry point for displaying streaming output.
    It selects the appropriate renderer based on config and handles
    all chunk types.

    Args:
        stream: Async iterator of StreamChunks.
        config: Output configuration.

    Returns:
        StreamResult with final content and metadata.

    Example:
        >>> stream = stream_agent_response(agent, messages, config)
        >>> result = await render_stream(stream, config)
        >>> print(f"Response: {result.content}")
        >>> print(f"Tool calls: {len(result.tool_calls)}")
    """
    renderer = TerminalRenderer(config)
    await renderer.start_stream()

    result = StreamResult()

    async for chunk in stream:
        await renderer.render_chunk(chunk)

        # Accumulate content for result
        if chunk.type == StreamChunkType.TEXT:
            result.content += chunk.content
        elif chunk.type == StreamChunkType.TOOL_CALL_END and chunk.tool_call:
            result.tool_calls.append(chunk.tool_call)
        elif chunk.type == StreamChunkType.METADATA:
            result.metadata = chunk.metadata

    await renderer.end_stream(result.metadata)
    return result


@dataclass
class StreamResult:
    """Result of a completed stream."""

    content: str = ""
    tool_calls: list[ToolCallInfo] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
```

#### `batch_response`

Fallback function for batch (non-streaming) mode.

```python
async def batch_response(
    agent: CompiledStateGraph,
    messages: list,
    config: OutputConfig,
) -> StreamResult:
    """Get agent response in batch mode (non-streaming).

    Used when OutputMode.BATCH or OutputMode.QUIET is selected,
    or as a fallback when streaming fails.

    Args:
        agent: The compiled agent graph.
        messages: Conversation messages.
        config: Output configuration.

    Returns:
        StreamResult with complete response.
    """
    start_time = time.time()
    tool_calls = []

    try:
        result = await agent.ainvoke({"messages": messages})

        # Extract final AI message
        ai_messages = [
            msg for msg in result.get("messages", [])
            if hasattr(msg, "type") and msg.type == "ai"
        ]

        content = ai_messages[-1].content if ai_messages else ""

        # Extract tool calls from message history
        for msg in result.get("messages", []):
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc in msg.tool_calls:
                    tool_calls.append(ToolCallInfo(
                        tool_id=tc.get("id", ""),
                        tool_name=tc.get("name", ""),
                        status=ToolCallStatus.SUCCESS,
                        arguments=tc.get("args", {}),
                    ))

        elapsed_ms = int((time.time() - start_time) * 1000)

        return StreamResult(
            content=content,
            tool_calls=tool_calls,
            metadata={"elapsed_ms": elapsed_ms},
        )

    except Exception as e:
        return StreamResult(
            error=str(e),
            metadata={"elapsed_ms": int((time.time() - start_time) * 1000)},
        )
```

---

### 7. CLI Integration

Integration points with the existing CLI architecture.

#### Updated `handle_chat_message`

```python
async def handle_chat_message(
    session: SessionState,
    message: str,
    output_config: OutputConfig,
) -> None:
    """Process a chat message with streaming output.

    Args:
        session: Current session state.
        message: User's message.
        output_config: Output configuration for rendering.
    """
    # Determine which agent to use
    if session.mode == SessionMode.BUILDER:
        agent = session.builder
        messages = session.builder_messages
        agent_name = "Builder"
    else:
        agent = session.active_agent
        messages = session.agent_messages
        agent_name = session.active_agent_name or "Agent"

    if agent is None:
        print(error("No agent available."))
        return

    # Add user message to history
    messages.append(HumanMessage(content=message))

    # Display agent name prefix
    if output_config.color_output:
        print(f"\n{Colors.BRIGHT_MAGENTA}{agent_name}:{Colors.RESET} ", end="")
    else:
        print(f"\n{agent_name}: ", end="")

    # Stream or batch based on mode
    if output_config.mode in (OutputMode.STREAM, OutputMode.VERBOSE):
        stream = stream_agent_response(agent, messages, output_config)
        result = await render_stream(stream, output_config)
    else:
        result = await batch_response(agent, messages, output_config)
        print(result.content)

    # Update message history with response
    if result.content:
        messages.append(AIMessage(content=result.content))

    # Show timing in verbose mode
    if output_config.show_timing and result.metadata.get("elapsed_ms"):
        elapsed = result.metadata["elapsed_ms"]
        print(f"\n{Colors.DIM}---{Colors.RESET}")
        print(f"{Colors.DIM}Elapsed: {elapsed}ms{Colors.RESET}")

    print()  # Final newline
```

#### CLI Configuration Extension

```python
# Extended CLIConfig with streaming options
@dataclass
class CLIConfig:
    """Configuration for CLI behavior."""

    # ... existing fields ...

    # Streaming configuration
    output_mode: OutputMode = field(default_factory=detect_output_mode)
    show_thinking: bool = False
    show_tool_calls: bool = True
    show_tool_results: bool = False
    show_timing: bool = False
    spinner_style: SpinnerStyle = SpinnerStyle.DOTS
    heartbeat_interval: float = 5.0

    def get_output_config(self) -> OutputConfig:
        """Build OutputConfig from CLI settings."""
        return OutputConfig(
            mode=self.output_mode,
            show_thinking=self.show_thinking,
            show_tool_calls=self.show_tool_calls,
            show_tool_results=self.show_tool_results,
            show_timing=self.show_timing,
            color_output=self.color_output,
            spinner_style=self.spinner_style.value,
            heartbeat_interval=self.heartbeat_interval,
            tool_timeout=self.mcp_timeout,
        )
```

#### Command-Line Flags

```bash
# Output mode flags
acti-builder --stream      # Force stream mode (default for TTY)
acti-builder --batch       # Force batch mode (default for pipes)
acti-builder --verbose     # Enable verbose output
acti-builder --quiet       # Minimal output

# Display flags
acti-builder --show-thinking     # Show agent reasoning
acti-builder --show-tool-results # Expand tool results
acti-builder --show-timing       # Show elapsed time
acti-builder --no-color          # Disable colors

# Environment variables
ACTI_OUTPUT_MODE=verbose acti-builder
ACTI_SHOW_TIMING=1 acti-builder
NO_COLOR=1 acti-builder
```

---

### 8. Error Handling in Streams

Graceful error handling throughout the streaming pipeline.

```python
class StreamingError(Exception):
    """Base exception for streaming errors."""
    pass


class StreamTimeoutError(StreamingError):
    """Stream timed out waiting for response."""

    def __init__(self, timeout: float):
        self.timeout = timeout
        super().__init__(f"Stream timed out after {timeout}s")


class StreamInterruptedError(StreamingError):
    """Stream was interrupted (e.g., user cancelled)."""
    pass


class ToolStreamingError(StreamingError):
    """Error during tool call streaming."""

    def __init__(self, tool_name: str, error: str):
        self.tool_name = tool_name
        self.error = error
        super().__init__(f"Tool '{tool_name}' streaming error: {error}")


async def safe_stream(
    stream: AsyncIterator[StreamChunk],
    timeout: float = 120.0,
) -> AsyncIterator[StreamChunk]:
    """Wrap a stream with timeout and error handling.

    Args:
        stream: The stream to wrap.
        timeout: Maximum time to wait for chunks (seconds).

    Yields:
        StreamChunks from the wrapped stream.

    Raises:
        StreamTimeoutError: If no chunk received within timeout.
    """
    try:
        async for chunk in asyncio.wait_for(
            stream.__anext__(),
            timeout=timeout,
        ):
            yield chunk
    except StopAsyncIteration:
        return
    except asyncio.TimeoutError:
        yield StreamChunk(
            type=StreamChunkType.ERROR,
            content=f"Stream timed out after {timeout}s",
        )
    except asyncio.CancelledError:
        yield StreamChunk(
            type=StreamChunkType.ERROR,
            content="Stream cancelled",
        )
        raise
```

---

### 9. Testing Patterns

```python
import pytest
from unittest.mock import AsyncMock, MagicMock


@pytest.mark.asyncio
async def test_stream_chunk_creation():
    """Test StreamChunk creation and serialization."""
    chunk = StreamChunk(
        type=StreamChunkType.TEXT,
        content="Hello",
        sequence=1,
    )
    assert chunk.type == StreamChunkType.TEXT
    assert chunk.content == "Hello"

    # Test serialization
    data = chunk.to_dict()
    assert data["type"] == "text"
    assert data["content"] == "Hello"


@pytest.mark.asyncio
async def test_stream_with_tool_calls():
    """Test streaming with tool call events."""
    async def mock_stream():
        yield StreamChunk(
            type=StreamChunkType.TEXT,
            content="Let me search for that. ",
            sequence=0,
        )
        yield StreamChunk(
            type=StreamChunkType.TOOL_CALL_START,
            content="Starting search",
            tool_call=ToolCallInfo(
                tool_id="tc_001",
                tool_name="tavily-search",
                status=ToolCallStatus.CALLING,
                arguments={"query": "AI trends"},
            ),
            sequence=1,
        )
        yield StreamChunk(
            type=StreamChunkType.TOOL_CALL_END,
            content="Search complete",
            tool_call=ToolCallInfo(
                tool_id="tc_001",
                tool_name="tavily-search",
                status=ToolCallStatus.SUCCESS,
                result="Found 5 results",
                duration_ms=1250,
            ),
            sequence=2,
        )
        yield StreamChunk(
            type=StreamChunkType.TEXT,
            content="Based on my research...",
            sequence=3,
        )

    config = OutputConfig.for_tty()
    result = await render_stream(mock_stream(), config)

    assert "Let me search for that" in result.content
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].tool_name == "tavily-search"


@pytest.mark.asyncio
async def test_output_mode_detection():
    """Test automatic output mode detection."""
    import sys
    from io import StringIO

    # Mock TTY
    original_stdout = sys.stdout
    sys.stdout = MagicMock()
    sys.stdout.isatty = MagicMock(return_value=True)

    mode = detect_output_mode()
    assert mode == OutputMode.STREAM

    # Mock non-TTY (pipe)
    sys.stdout.isatty = MagicMock(return_value=False)
    mode = detect_output_mode()
    assert mode == OutputMode.BATCH

    sys.stdout = original_stdout


@pytest.mark.asyncio
async def test_batch_fallback():
    """Test batch mode fallback when streaming fails."""
    agent = MagicMock()
    agent.ainvoke = AsyncMock(return_value={
        "messages": [
            MagicMock(type="ai", content="Response from batch mode"),
        ]
    })

    config = OutputConfig.for_pipe()
    result = await batch_response(agent, [], config)

    assert result.content == "Response from batch mode"
    assert result.error is None


@pytest.mark.asyncio
async def test_stream_error_handling():
    """Test error handling in streams."""
    async def failing_stream():
        yield StreamChunk(type=StreamChunkType.TEXT, content="Starting...")
        raise RuntimeError("Stream failed")

    config = OutputConfig.for_tty()

    with pytest.raises(RuntimeError):
        async for chunk in failing_stream():
            pass
```

---

### 10. Module Exports (Streaming)

```python
# Added to __all__ in src/cli.py

__all__ = [
    # ... existing exports ...

    # Streaming types
    "StreamChunk",
    "StreamChunkType",
    "StreamResult",
    "ToolCallInfo",
    "ToolCallStatus",

    # Output configuration
    "OutputMode",
    "OutputConfig",
    "SpinnerStyle",

    # Streaming functions
    "stream_agent_response",
    "render_stream",
    "batch_response",
    "safe_stream",

    # Renderers
    "StreamingRenderer",
    "TerminalRenderer",

    # Progress indicators
    "ProgressIndicator",

    # Errors
    "StreamingError",
    "StreamTimeoutError",
    "StreamInterruptedError",
    "ToolStreamingError",
]
```

---

## Version History

- **1.2.0** (2025-01-09): Streaming Output System (Task 2.5.6)
  - StreamChunk protocol with 8 chunk types (TEXT, THINKING, TOOL_CALL_START, TOOL_CALL_END, TOOL_RESULT, STATUS, ERROR, HEARTBEAT, METADATA)
  - ToolCallInfo dataclass tracking tool invocations with timing and status
  - Four output modes: STREAM (token-by-token), BATCH (wait for complete), VERBOSE (expanded), QUIET (minimal)
  - OutputConfig with 15+ configurable options for display behavior
  - TerminalRenderer for rich terminal output with colors and spinners
  - Progress indicators with spinner animations and progress bars
  - `stream_agent_response()` function using LangGraph's astream_events
  - `render_stream()` high-level rendering function
  - `batch_response()` fallback for non-streaming contexts
  - Heartbeat emission for long-running operations
  - Command-line flags: --stream, --batch, --verbose, --quiet, --show-thinking, --show-tool-results, --show-timing, --no-color
  - Environment variables: ACTI_OUTPUT_MODE, ACTI_SHOW_TIMING, NO_COLOR
  - Error handling: StreamingError, StreamTimeoutError, StreamInterruptedError, ToolStreamingError
  - Comprehensive testing patterns for streaming components

- **1.1.0** (2025-01-09): Enhanced Agent Modification Flow (Task 2.5.5)
  - Natural language modification via `/modify <name> <description>`
  - Interactive modification mode with `/show`, `/diff`, `/done`, `/cancel` sub-commands
  - Colorized diff previews for all modifications
  - Automatic backup creation before modifications (`outputs/agents/{name}.backup.json`)
  - `/rollback <name>` command for restoring previous versions
  - `/history <name>` command for viewing modification history
  - Batch modifications with `--all`, `--filter`, `--agents` flags
  - Tag management with `--add-tag` and `--remove-tag` flags
  - New `modify_agent_natural` builder tool
  - New schemas: `ModificationRequest`, `ModificationPlan`, `ModificationResult`, `BatchTarget`, `BatchModification`
  - Error handling: `ModificationError`, `NoChangesDetectedError`, `BackupNotFoundError`, `BatchModificationError`

- **1.0.0** (2025-01-09): Initial API contract
  - Command specifications for all 11 commands
  - Session state management with mode transitions
  - Streaming output support
  - Error handling with recovery suggestions
  - Integration patterns with builder and MCP modules
  - Testing patterns for unit and integration tests

---

## API Design Report

### Spec Files Created/Updated
- `docs/api-contract-cli.md` - Complete CLI API specification (14 commands, 8 core functions, enhanced modification flow, streaming output system)

### Core Design Decisions
1. **Mode-based architecture**: Clear separation between BUILDER, AGENT, and MODIFY modes allows predictable behavior and clean state management
2. **Async-first design**: All handlers are async to support MCP operations and streaming without blocking
3. **Command + Chat hybrid**: Slash commands for explicit actions, plain text for natural conversation with current mode context
4. **Graceful degradation**: Auto-instantiation failures do not prevent agent creation; agents can be saved without MCP tools
5. **Natural language modifications**: The builder agent interprets modification requests, eliminating manual field-by-field editing
6. **Automatic backup strategy**: Every modification creates a backup, enabling safe rollback without data loss
7. **Batch operations**: Support for applying changes across multiple agents based on filters (stratum, tags, tools)
8. **Diff-first confirmation**: All modifications show clear before/after diffs before applying changes
9. **StreamChunk protocol**: Unified streaming protocol with 8 chunk types enables consistent handling of text, tool calls, and status
10. **Adaptive output modes**: Automatic TTY detection with manual override via flags/environment variables
11. **Progressive disclosure**: Token streaming with tool call indicators provides immediate feedback without overwhelming users

### Authentication & Security
- Method: Environment variables for API keys (follows existing project pattern)
- No additional CLI authentication required (local tool)
- File operations restricted to `outputs/agents/` directory
- Backup files stored alongside agent configs in `outputs/agents/.history/`
- Session data stored in user home directory with restricted permissions

### Open Questions
1. Should conversation history be persisted between sessions? (Currently: metadata only)
2. ~~Should `/modify` support rollback to previous versions?~~ **RESOLVED: Yes, via `/rollback` command**
3. Should we add `/export` and `/import` commands for agent portability?
4. ~~Should tool call output be collapsed/expandable in streaming mode?~~ **RESOLVED: Yes, via `show_tool_results` config**
5. Should batch modifications support dry-run mode without confirmation prompts?
6. Should `/history` support pagination for agents with many versions?
7. Should we add a `/replay` command to replay previous interactions with streaming?
8. Should VERBOSE mode show token count in real-time?

### Implementation Guidance
1. Start with `prompt_toolkit` for rich input handling; fall back to `readline` if unavailable
2. Use `rich` library for consistent formatting and progress indicators
3. Implement command handlers as independent async functions for testability
4. Create comprehensive fixtures for mocking builder and MCP integration in tests
5. Add `--debug` flag for verbose logging during development
6. Implement `BackupManager` class for centralized backup/restore logic
7. Use unified `DiffRenderer` for consistent diff output across terminal, plain text, and JSON
8. Parse modification flags (`--all`, `--filter`, etc.) before natural language to separate intent from target
9. Add `modify_agent_natural` to `BUILDER_TOOLS` list in `src/tools.py`
10. Store backup metadata in `_backup_metadata` field to track backup provenance
11. Implement `StreamingRenderer` as abstract base class for future renderer variants (web, file, etc.)
12. Use LangGraph's `astream_events(version="v2")` for granular streaming event access
13. Wrap streams with `safe_stream()` for consistent timeout and error handling
14. Auto-detect output mode via `sys.stdout.isatty()` with `ACTI_OUTPUT_MODE` override
15. Buffer spinner updates to prevent terminal flicker (16ms minimum between frames)
