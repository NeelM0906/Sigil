# API Contract: src/tools.py

## Overview

This document defines the API contract for `src/tools.py`, the LangChain tools module for the ACTi Agent Builder. These tools are used by the deepagents-based meta-agent to create and manage agent configurations.

---

## Design Principles

1. **String Returns**: All tools return strings (LLM-readable results)
2. **Thorough Docstrings**: Docstrings become tool descriptions for the LLM - be comprehensive
3. **Type Annotations**: Define input schemas via Python type hints
4. **Actionable Errors**: Error messages guide the LLM toward fixes
5. **Consistent Naming**: snake_case for all identifiers

---

## Tool Specifications

### 1. `create_agent_config`

Creates and persists an agent configuration to disk.

#### Signature

```python
@tool
def create_agent_config(
    name: str,
    description: str,
    system_prompt: str,
    tools: Optional[list[str]] = None,
    model: str = "anthropic:claude-opus-4-5-20251101",
    stratum: Optional[str] = None,
    tags: Optional[list[str]] = None,
) -> str:
```

#### Parameters

| Parameter | Type | Required | Constraints | Description |
|-----------|------|----------|-------------|-------------|
| `name` | `str` | Yes | snake_case, 1-64 chars, starts with letter | Unique identifier for the agent |
| `description` | `str` | Yes | 10-500 characters | What the agent does |
| `system_prompt` | `str` | Yes | Minimum 50 characters | Complete system prompt |
| `tools` | `list[str]` | No | Valid: voice, websearch, calendar, communication, crm | MCP tool categories |
| `model` | `str` | No | Default: anthropic:claude-opus-4-5-20251101 | LLM model identifier |
| `stratum` | `str` | No | Valid: RTI, RAI, ZACS, EEI, IGE | ACTi stratum classification |
| `tags` | `list[str]` | No | Max 10, each 2-32 chars, lowercase | Categorization tags |

#### Return Values

**Success:**
```
SUCCESS: Agent configuration created and saved!

Agent Summary:
  Name: lead_qualifier
  Description: Qualifies inbound leads by assessing BANT criteria
  Tools: communication, crm
  Stratum: RAI
  Model: anthropic:claude-opus-4-5-20251101

Saved to: /path/to/outputs/agents/lead_qualifier.json

Next steps:
  - Use get_agent_config('lead_qualifier') to review the full configuration
  - The agent can be instantiated with MCP tools using the FastAPI backend
```

**Validation Error:**
```
ERROR: Agent configuration validation failed.

Validation errors:
  - name: name must be snake_case: lowercase letters, numbers, and underscores only, starting with a letter

Requirements:
  - name: snake_case (lowercase, numbers, underscores, starts with letter)
  - description: 10-500 characters
  - system_prompt: minimum 50 characters
  - tools: valid options are [voice, websearch, calendar, communication, crm]

Please fix the issues and try again.
```

**Invalid Stratum Error:**
```
ERROR: Invalid stratum 'INVALID'.

Valid stratum options:
  - RTI: Reality & Truth Intelligence (data gathering)
  - RAI: Readiness & Agreement Intelligence (qualification)
  - ZACS: Zone Action & Conversion Systems (scheduling)
  - EEI: Economic & Ecosystem Intelligence (analytics)
  - IGE: Integrity & Governance Engine (compliance)

Please use one of: RTI, RAI, ZACS, EEI, IGE
```

#### Output File Format

Saved to: `outputs/agents/{name}.json`

```json
{
  "name": "lead_qualifier",
  "description": "Qualifies inbound leads by assessing BANT criteria",
  "system_prompt": "You are a professional lead qualification specialist...",
  "tools": ["communication", "crm"],
  "model": "anthropic:claude-opus-4-5-20251101",
  "stratum": "RAI",
  "tags": ["sales", "qualification", "b2b"],
  "_metadata": {
    "created_at": "2025-01-09T14:30:22.123456+00:00",
    "updated_at": "2025-01-09T14:30:22.123456+00:00",
    "last_executed_at": null,
    "version": 1,
    "builder": "acti-agent-builder",
    "execution_count": 0,
    "success_count": 0,
    "error_count": 0
  }
}
```

See [Enhanced Metadata Schema](#enhanced-metadata-schema) for details on all metadata fields.

---

### 2. `list_available_tools`

Returns available MCP tool categories with descriptions and guidance.

#### Signature

```python
@tool
def list_available_tools() -> str:
```

#### Parameters

None.

#### Return Value

```
Available MCP Tool Categories
==================================================

  voice
    Text-to-speech, voice synthesis (ElevenLabs)
    Capabilities:
      - Generate spoken audio from text
      - Clone and customize voices
      - Real-time voice synthesis for calls
    Best for strata: ZACS, RAI
    Use cases: Customer-facing agents, accessibility, audio content

  websearch
    Web research, fact-finding (Tavily)
    Capabilities:
      - Search the web for current information
      - Research topics and gather facts
      - Find documentation and references
    Best for strata: RTI, EEI
    Use cases: Research agents, fact-checkers, information gatherers

  calendar
    Scheduling, availability checks (Google Calendar)
    Capabilities:
      - Check availability and schedule meetings
      - Create, update, and delete calendar events
      - Send calendar invitations
    Best for strata: ZACS
    Use cases: Scheduling agents, appointment setters, coordinators

  communication
    SMS, calls (Twilio)
    Capabilities:
      - Send SMS messages
      - Make and receive phone calls
      - Manage communication workflows
    Best for strata: RAI, ZACS
    Use cases: Outreach agents, follow-up agents, notification systems

  crm
    Contact management, deals (HubSpot)
    Capabilities:
      - Access and update contact records
      - Manage deals and pipelines
      - Track customer interactions
    Best for strata: RTI, RAI, EEI
    Use cases: Sales agents, customer service, data management

==================================================
ACTi Stratum Reference & Tool Recommendations
==================================================

  RTI (Reality & Truth Intelligence)
    Purpose: Data gathering, fact verification, research
    Recommended tools: websearch, crm
    Example agents: Research assistant, fact-checker, data gatherer

  RAI (Readiness & Agreement Intelligence)
    Purpose: Lead qualification, rapport building, discovery
    Recommended tools: communication, crm
    Example agents: Lead qualifier, discovery agent, SDR assistant

  ZACS (Zone Action & Conversion Systems)
    Purpose: Scheduling, conversions, follow-ups, closings
    Recommended tools: calendar, communication, voice
    Example agents: Appointment scheduler, closer, follow-up agent

  EEI (Economic & Ecosystem Intelligence)
    Purpose: Analytics, market research, optimization
    Recommended tools: websearch, crm
    Example agents: Market analyst, pipeline optimizer, trend tracker

  IGE (Integrity & Governance Engine)
    Purpose: Compliance, quality control, auditing
    Recommended tools: all (for comprehensive auditing)
    Example agents: Compliance checker, QA agent, audit assistant
```

---

### 3. `get_agent_config`

Retrieves a previously saved agent configuration by name.

#### Signature

```python
@tool
def get_agent_config(name: str) -> str:
```

#### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `name` | `str` | Yes | Agent name (snake_case, without .json extension) |

#### Return Values

**Success:**
```
Agent Configuration: lead_qualifier
==================================================

Quick Summary:
  Tools: communication, crm
  Stratum: RAI
  Model: anthropic:claude-opus-4-5-20251101

Full Configuration:
{
  "name": "lead_qualifier",
  "description": "Qualifies inbound leads by assessing BANT criteria",
  "system_prompt": "You are a professional lead qualification specialist...",
  "tools": ["communication", "crm"],
  "model": "anthropic:claude-opus-4-5-20251101",
  "stratum": "RAI",
  "_metadata": {
    "created_at": "2025-01-09T14:30:22.123456+00:00",
    "version": "1.0.0",
    "builder": "acti-agent-builder"
  }
}
```

**Not Found (with suggestions):**
```
ERROR: Agent 'invalid_name' not found.

Available agents (3):
  - appointment_scheduler
  - lead_qualifier
  - research_assistant

Use one of the names above, or create a new agent with create_agent_config().
```

**Not Found (no agents exist):**
```
ERROR: Agent 'invalid_name' not found.

No agents have been created yet. Use create_agent_config() to create one.
```

---

### 4. `list_created_agents`

Lists all agent configurations that have been created and saved.

#### Signature

```python
@tool
def list_created_agents() -> str:
```

#### Parameters

None.

#### Return Values

**With agents (grouped by stratum):**
```
Created Agents
==================================================

[RTI] (1 agent(s))
----------------------------------------
  research_assistant
    Description: Researches topics and gathers facts from the web
    Tools: websearch
    Created: 2025-01-09T12:00:00+00:00

[RAI] (1 agent(s))
----------------------------------------
  lead_qualifier
    Description: Qualifies inbound leads by assessing BANT criteria
    Tools: communication, crm
    Created: 2025-01-09T14:30:22+00:00

[ZACS] (1 agent(s))
----------------------------------------
  appointment_scheduler
    Description: Schedules appointments and manages calendar availability
    Tools: calendar, communication
    Created: 2025-01-09T15:45:10+00:00

==================================================
Total: 3 agent(s)

Use get_agent_config('<name>') to see full configuration details.
```

**No agents:**
```
No agents have been created yet.

To create your first agent:
  1. Call list_available_tools() to see available tool options
  2. Call create_agent_config() with the agent's details

Example:
  create_agent_config(
    name='my_first_agent',
    description='A helpful agent that...',
    system_prompt='You are...',
    tools=['websearch'],
    stratum='RTI'
  )
```

---

## Utility Functions (Internal)

These functions support the tools but are not exposed to the LLM.

### `_ensure_output_dir() -> Path`
Ensures `outputs/agents/` directory exists, creates if needed.

### `_format_validation_error(error: ValidationError) -> str`
Converts Pydantic ValidationError to human-readable multi-line string.

### `_generate_filename(agent_name: str, include_timestamp: bool = False) -> str`
Generates filename for agent config. Optionally includes timestamp.

### `_serialize_config_with_metadata(config: AgentConfig) -> dict`
Serializes AgentConfig with `_metadata` block (created_at, version, builder).

---

## Module Exports

```python
# All tools available to the builder agent
BUILDER_TOOLS = [
    # Core CRUD
    create_agent_config,
    get_agent_config,
    update_agent_config,      # NEW: Task 2.5.4
    delete_agent_config,      # NEW: Task 2.5.4
    clone_agent_config,       # NEW: Task 2.5.4

    # Discovery
    list_available_tools,
    list_created_agents,
    search_agents,            # NEW: Task 2.5.4

    # Execution
    execute_created_agent,
    get_agent_history,        # NEW: Task 2.5.4
]

# Async variants for FastAPI integration
ASYNC_TOOLS = {
    # Core CRUD
    "create_agent_config": create_agent_config_async,
    "get_agent_config": get_agent_config_async,
    "update_agent_config": update_agent_config_async,
    "delete_agent_config": delete_agent_config_async,
    "clone_agent_config": clone_agent_config_async,

    # Listing and Search
    "list_created_agents": list_created_agents_async,
    "search_agents": search_agents_async,

    # Execution
    "execute_created_agent": execute_created_agent_async,
    "get_agent_history": get_agent_history_async,
}
```

---

## Error Handling Strategy

1. **Validation Errors**: Caught from Pydantic, formatted with field-by-field details
2. **File System Errors**: Caught as OSError, include original message
3. **JSON Parsing Errors**: Caught for corrupted config files
4. **Stratum Errors**: Pre-validated before creating AgentConfig
5. **No Generic Exceptions**: Each error type handled specifically for clear messages

---

## Integration with deepagents

```python
from deepagents import create_deep_agent
from src.tools import BUILDER_TOOLS
from src.prompts import BUILDER_SYSTEM_PROMPT

agent = create_deep_agent(
    model="anthropic:claude-opus-4-5-20251101",
    tools=BUILDER_TOOLS,
    system_prompt=BUILDER_SYSTEM_PROMPT,
)

# Use the agent
result = await agent.ainvoke({
    "messages": [{"role": "user", "content": "Create a lead qualification agent"}]
})
```

---

## File Structure

```
outputs/
  agents/
    lead_qualifier.json
    lead_qualifier_history.json
    appointment_scheduler.json
    appointment_scheduler_history.json
    research_assistant.json
    research_assistant_history.json
```

---

## Enhanced Metadata Schema

### Agent Metadata (Task 2.5.4)

The `_metadata` block in agent configurations has been enhanced to track lifecycle and execution statistics.

#### Schema Definition

```python
class AgentMetadata(BaseModel):
    """Enhanced metadata for agent configurations."""

    # Timestamps
    created_at: datetime = Field(
        ...,
        description="ISO 8601 timestamp when agent was first created"
    )
    updated_at: datetime = Field(
        ...,
        description="ISO 8601 timestamp when agent was last modified"
    )
    last_executed_at: Optional[datetime] = Field(
        default=None,
        description="ISO 8601 timestamp of most recent execution"
    )

    # Versioning
    version: int = Field(
        default=1,
        ge=1,
        description="Monotonically increasing version number, incremented on each update"
    )
    builder: str = Field(
        default="acti-agent-builder",
        description="Identifier of the system that created/modified this agent"
    )

    # Execution Statistics
    execution_count: int = Field(
        default=0,
        ge=0,
        description="Total number of times this agent has been executed"
    )
    success_count: int = Field(
        default=0,
        ge=0,
        description="Number of successful executions"
    )
    error_count: int = Field(
        default=0,
        ge=0,
        description="Number of failed executions"
    )
```

#### JSON Representation

```json
{
  "_metadata": {
    "created_at": "2025-01-09T14:30:22.123456+00:00",
    "updated_at": "2025-01-10T09:15:00.000000+00:00",
    "last_executed_at": "2025-01-10T09:20:45.123456+00:00",
    "version": 3,
    "builder": "acti-agent-builder",
    "execution_count": 47,
    "success_count": 45,
    "error_count": 2
  }
}
```

#### Metadata Update Rules

| Operation | `updated_at` | `version` | `execution_count` | `success_count` | `error_count` |
|-----------|--------------|-----------|-------------------|-----------------|---------------|
| `create_agent_config` | Set to now | Set to 1 | 0 | 0 | 0 |
| `update_agent_config` | Set to now | Increment +1 | Preserved | Preserved | Preserved |
| `clone_agent_config` | Set to now | Set to 1 | 0 | 0 | 0 |
| `execute_created_agent` (success) | Preserved | Preserved | +1 | +1 | Preserved |
| `execute_created_agent` (failure) | Preserved | Preserved | +1 | Preserved | +1 |

---

## Agent Tags

Agents can be categorized using tags for better organization and searchability.

#### Tag Constraints

| Constraint | Value |
|------------|-------|
| Format | lowercase alphanumeric with hyphens |
| Pattern | `^[a-z][a-z0-9-]*$` |
| Min length | 2 characters |
| Max length | 32 characters |
| Max tags per agent | 10 |

#### Reserved Tags

The following tags have special meaning in the system:

| Tag | Meaning |
|-----|---------|
| `deprecated` | Agent is no longer recommended for use |
| `experimental` | Agent is in testing phase |
| `production` | Agent is approved for production use |
| `template` | Agent serves as a template for cloning |

#### Example

```python
create_agent_config(
    name="lead_qualifier_v2",
    description="Enhanced lead qualification with MEDDIC framework",
    system_prompt="...",
    tools=["communication", "crm"],
    stratum="RAI",
    tags=["sales", "qualification", "meddic", "production"]
)
```

---

## Execution History

Each agent maintains an execution history in a separate file: `outputs/agents/{name}_history.json`

### History File Schema

```python
class ExecutionRecord(BaseModel):
    """Record of a single agent execution."""

    execution_id: str = Field(
        ...,
        description="UUID for this execution"
    )
    timestamp: datetime = Field(
        ...,
        description="When the execution started"
    )
    task: str = Field(
        ...,
        max_length=500,
        description="Task that was executed (truncated if longer)"
    )
    result_summary: str = Field(
        ...,
        max_length=200,
        description="Brief summary of the result"
    )
    success: bool = Field(
        ...,
        description="Whether execution completed successfully"
    )
    duration_seconds: float = Field(
        ...,
        description="Total execution time in seconds"
    )
    tool_calls_count: int = Field(
        default=0,
        description="Number of tool calls made"
    )
    error_type: Optional[str] = Field(
        default=None,
        description="Type of error if execution failed"
    )


class ExecutionHistory(BaseModel):
    """Execution history for an agent."""

    agent_name: str = Field(
        ...,
        description="Name of the agent this history belongs to"
    )
    executions: list[ExecutionRecord] = Field(
        default_factory=list,
        max_length=50,
        description="List of execution records (max 50, FIFO)"
    )
    total_executions: int = Field(
        default=0,
        description="Total lifetime executions (may exceed 50)"
    )
```

### History File Format

```json
{
  "agent_name": "lead_qualifier",
  "executions": [
    {
      "execution_id": "550e8400-e29b-41d4-a716-446655440001",
      "timestamp": "2025-01-10T09:20:45.123456+00:00",
      "task": "Qualify this lead: John from Acme Corp, $50k budget",
      "result_summary": "Lead qualified as HIGH priority - meets BANT criteria",
      "success": true,
      "duration_seconds": 3.2,
      "tool_calls_count": 2,
      "error_type": null
    },
    {
      "execution_id": "550e8400-e29b-41d4-a716-446655440000",
      "timestamp": "2025-01-10T08:15:30.000000+00:00",
      "task": "Qualify lead: Jane from StartupXYZ",
      "result_summary": "Lead qualified as MEDIUM priority - needs budget confirmation",
      "success": true,
      "duration_seconds": 2.8,
      "tool_calls_count": 1,
      "error_type": null
    }
  ],
  "total_executions": 47
}
```

### History Management

- **Maximum Records**: 50 executions stored per agent (FIFO - oldest removed first)
- **Automatic Cleanup**: When 51st execution is added, oldest record is removed
- **Separate File**: History is stored separately from config to avoid bloating the main config
- **Lazy Creation**: History file is created on first execution, not on agent creation

---

### 5. `execute_created_agent`

Loads a saved agent configuration, instantiates it with real MCP tools, and executes a task. This enables the builder to CREATE an agent and then IMMEDIATELY TEST it in one flow.

#### Signature

```python
@tool
def execute_created_agent(
    agent_name: str,
    task: str,
    timeout: Optional[float] = None,
    skip_unavailable_tools: bool = True,
) -> str:
```

#### Parameters

| Parameter | Type | Required | Default | Constraints | Description |
|-----------|------|----------|---------|-------------|-------------|
| `agent_name` | `str` | Yes | - | Must match existing agent in `outputs/agents/` | Name of the agent to load and execute |
| `task` | `str` | Yes | - | 1-10000 characters | The task or message to send to the agent |
| `timeout` | `float` | No | `60.0` | 1.0-300.0 seconds | Maximum execution time for the task |
| `skip_unavailable_tools` | `bool` | No | `True` | - | If True, continue with available tools only; if False, fail if any tool is unavailable |

#### Return Values

**Success (with tool calls):**
```
EXECUTION SUCCESS: Agent 'lead_qualifier' completed task.

Task: Qualify this lead: John from Acme Corp, $50k budget, Q2 timeline

Agent Response:
================================================================================
Based on my analysis of this lead, here's the qualification assessment:

**Lead: John from Acme Corp**

BANT Analysis:
- Budget: $50,000 - QUALIFIED (meets minimum threshold)
- Authority: Unknown - NEEDS DISCOVERY (recommend asking about decision-makers)
- Need: Not specified - NEEDS DISCOVERY (schedule discovery call)
- Timeline: Q2 - QUALIFIED (actionable timeframe)

**Recommendation:** This is a promising lead. Schedule a discovery call to assess
authority and understand specific needs. Priority: HIGH.
================================================================================

Execution Details:
  Duration: 3.2s
  Tool Calls: 2
    1. crm_lookup_contact(company="Acme Corp") -> Found: John Smith, VP Sales
    2. crm_get_deal_history(contact_id="12345") -> No previous deals

MCP Tools Used: crm (HubSpot)
```

**Success (no tool calls):**
```
EXECUTION SUCCESS: Agent 'research_assistant' completed task.

Task: Summarize the key points of machine learning

Agent Response:
================================================================================
Machine learning is a subset of artificial intelligence that enables systems
to learn and improve from experience without being explicitly programmed...

Key Points:
1. Supervised Learning: Training with labeled data
2. Unsupervised Learning: Finding patterns in unlabeled data
3. Reinforcement Learning: Learning through trial and reward
================================================================================

Execution Details:
  Duration: 1.8s
  Tool Calls: 0 (agent used internal knowledge)

MCP Tools Available: websearch (not needed for this task)
```

**Agent Not Found:**
```
ERROR: Agent 'invalid_agent' not found.

Available agents (3):
  - appointment_scheduler
  - lead_qualifier
  - research_assistant

Use one of the names above, or create a new agent with create_agent_config().
```

**MCP Connection Failure (skip_unavailable_tools=True):**
```
WARNING: Some MCP tools unavailable, proceeding with available tools.

Unavailable tools:
  - calendar: Missing credentials (GOOGLE_OAUTH_CREDENTIALS not set)

EXECUTION SUCCESS: Agent 'appointment_scheduler' completed task (degraded mode).

Task: Check my availability for next week

Agent Response:
================================================================================
I apologize, but I cannot access your calendar at the moment because the
calendar integration is not currently available. However, I can help you
prepare for scheduling once the calendar is connected.

Would you like me to:
1. Draft a list of your typical availability preferences?
2. Prepare meeting templates for different appointment types?
================================================================================

Execution Details:
  Duration: 2.1s
  Tool Calls: 0 (calendar unavailable)
  Tools Requested: calendar, communication
  Tools Available: communication
  Tools Unavailable: calendar

Note: Agent ran in degraded mode. Set GOOGLE_OAUTH_CREDENTIALS to enable full functionality.
```

**MCP Connection Failure (skip_unavailable_tools=False):**
```
ERROR: Cannot execute agent 'appointment_scheduler' - required tools unavailable.

Missing tools:
  - calendar: Missing credentials (GOOGLE_OAUTH_CREDENTIALS not set)

To fix this:
  1. Set the required environment variable: GOOGLE_OAUTH_CREDENTIALS
  2. Or use skip_unavailable_tools=True to run with available tools only

Use list_available_tools() to check current tool status.
```

**Timeout Error:**
```
ERROR: Agent execution timed out after 60.0 seconds.

Task: Research comprehensive market analysis for all tech sectors

The agent did not complete within the specified timeout. This may happen for:
  - Complex tasks requiring multiple tool calls
  - Slow network connections to MCP servers
  - Large data processing operations

Suggestions:
  1. Increase timeout: execute_created_agent(..., timeout=120.0)
  2. Simplify the task into smaller sub-tasks
  3. Check MCP server health with /tools command
```

**Execution Error (agent error):**
```
ERROR: Agent 'lead_qualifier' encountered an error during execution.

Task: Qualify this lead: [corrupted input]

Error: Invalid input format - could not parse lead information.

Partial execution details:
  Duration before error: 1.2s
  Tool Calls Attempted: 1
    1. crm_lookup_contact(company=null) -> ERROR: company parameter required

The agent configuration may need adjustment, or the task input was malformed.
Use get_agent_config('lead_qualifier') to review the agent's configuration.
```

#### Behavior Flow

```
execute_created_agent("lead_qualifier", "Qualify John from Acme")
    |
    v
+-------------------+
| 1. Load Config    |
| outputs/agents/   |
| lead_qualifier.   |
| json              |
+--------+----------+
         |
         v
+-------------------+
| 2. Validate       |
| - Config exists   |
| - Tools valid     |
+--------+----------+
         |
         v
+-------------------+
| 3. Check MCP      |
| - Credentials     |
| - Server status   |
+--------+----------+
         |
    [All tools OK?]
         |
    Yes  |  No
    +----+----+
    |         |
    v         v
+-------+ +-------------------+
|Continue| |skip_unavailable?|
+-------+ +--------+----------+
    |          |         |
    |      Yes |     No  |
    |          |         |
    |          v         v
    |     +-------+ +-------+
    |     |Degrade| |ERROR  |
    |     |mode   | |return |
    |     +---+---+ +-------+
    |         |
    +----+----+
         |
         v
+-------------------+
| 4. Connect MCP    |
| - Start servers   |
| - Get tools       |
+--------+----------+
         |
         v
+-------------------+
| 5. Instantiate    |
| create_agent_     |
| with_tools()      |
+--------+----------+
         |
         v
+-------------------+
| 6. Execute Task   |
| agent.invoke()    |
| with timeout      |
+--------+----------+
         |
    [Completed?]
         |
    Yes  |  Timeout  |  Error
    +----+----+------+----+
    |         |           |
    v         v           v
+-------+ +--------+ +--------+
|Format | |Timeout | |Error   |
|success| |error   | |message |
|output | |message | |+ debug |
+-------+ +--------+ +--------+
```

#### Integration with Existing Tools

This tool complements the existing builder tools to enable a complete create-test cycle:

```
1. list_available_tools()     -> Discover what tools are available
2. create_agent_config(...)   -> Create the agent configuration
3. execute_created_agent(...) -> Test the agent immediately
4. get_agent_config(...)      -> Review if modifications needed
```

**Example Builder Conversation:**

```
User: Create a research agent and test it with a sample query

Builder: I'll create a research agent for you and test it.

[Calling list_available_tools()]
[Calling create_agent_config(
    name="web_researcher",
    description="Researches topics using web search and provides summaries",
    system_prompt="You are a research assistant...",
    tools=["websearch"],
    stratum="RTI"
)]

SUCCESS: Agent 'web_researcher' created!

Now let me test it with a sample query...

[Calling execute_created_agent(
    agent_name="web_researcher",
    task="What are the latest developments in quantum computing?"
)]

EXECUTION SUCCESS: Agent 'web_researcher' completed task.

Agent Response:
================================================================================
Based on my research, here are the latest developments in quantum computing:

1. **IBM's 1000+ Qubit Processor** - IBM announced...
2. **Google's Error Correction Breakthrough** - Researchers at Google...
3. **Practical Applications Emerging** - Industries are beginning to...
================================================================================

Execution Details:
  Duration: 4.5s
  Tool Calls: 3
    1. tavily_search("quantum computing 2025 developments")
    2. tavily_search("IBM quantum processor latest")
    3. tavily_search("Google quantum error correction")

The agent is working correctly! You can now use it via:
  - CLI: /load web_researcher
  - API: POST /agents/web_researcher/run
```

#### Response Schema (Structured)

For programmatic consumption, the tool internally uses this structure before formatting:

```python
@dataclass
class ExecutionResult:
    """Result of agent execution."""
    success: bool
    agent_name: str
    task: str
    response: Optional[str] = None
    duration_seconds: float = 0.0
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    tools_used: list[str] = field(default_factory=list)
    tools_unavailable: list[str] = field(default_factory=list)
    degraded_mode: bool = False
    error: Optional[str] = None
    error_type: Optional[str] = None  # "not_found", "mcp_error", "timeout", "execution_error"

@dataclass
class ToolCallRecord:
    """Record of a single tool invocation."""
    tool_name: str
    arguments: dict[str, Any]
    result: Optional[str] = None
    error: Optional[str] = None
    duration_ms: float = 0.0
```

#### Async Variant

For FastAPI integration, an async version is also provided:

```python
async def execute_created_agent_async(
    agent_name: str,
    task: str,
    timeout: Optional[float] = None,
    skip_unavailable_tools: bool = True,
) -> str:
    """Async variant of execute_created_agent for FastAPI integration.

    Args:
        agent_name: Name of the agent to execute.
        task: Task or message to send to the agent.
        timeout: Maximum execution time in seconds.
        skip_unavailable_tools: Continue with available tools if some are missing.

    Returns:
        Formatted execution result string.
    """
```

#### Error Categories

| Error Type | Cause | Recovery |
|------------|-------|----------|
| `AgentNotFound` | Agent config file doesn't exist | Use `list_created_agents()` to see available agents |
| `MCPCredentialError` | Missing API keys for tools | Set environment variables or use `skip_unavailable_tools=True` |
| `MCPConnectionError` | Cannot connect to MCP server | Check network, verify MCP server is running |
| `MCPTimeoutError` | MCP connection took too long | Increase timeout or check server status |
| `ExecutionTimeout` | Agent took too long to respond | Increase timeout or simplify task |
| `ExecutionError` | Agent encountered runtime error | Review agent config, check task input |

#### Security Considerations

1. **Path Traversal Prevention**: Agent names are validated against the `outputs/agents/` directory; attempts to use `../` or absolute paths are rejected
2. **Timeout Enforcement**: Hard limit of 300 seconds prevents runaway executions
3. **Input Sanitization**: Task input is passed directly to the agent; agent's system prompt should include appropriate guardrails
4. **Credential Isolation**: MCP credentials are only loaded for tools specified in the agent config

#### Performance Characteristics

| Metric | Typical Value | Notes |
|--------|---------------|-------|
| Config Load | <10ms | File I/O from local disk |
| MCP Connection | 1-5s per server | First connection; subsequent calls reuse connection |
| Tool Validation | <50ms | Environment variable checks |
| Agent Instantiation | 100-500ms | deepagents setup with tools |
| Task Execution | 1-60s | Depends on task complexity and tool calls |

---

### 6. `update_agent_config`

Modifies an existing agent configuration. Supports partial updates - only specified fields are changed.

#### Signature

```python
@tool
def update_agent_config(
    name: str,
    description: Optional[str] = None,
    system_prompt: Optional[str] = None,
    tools: Optional[list[str]] = None,
    model: Optional[str] = None,
    stratum: Optional[str] = None,
    tags: Optional[list[str]] = None,
) -> str:
```

#### Parameters

| Parameter | Type | Required | Constraints | Description |
|-----------|------|----------|-------------|-------------|
| `name` | `str` | Yes | Must exist in `outputs/agents/` | Agent to update (cannot change name) |
| `description` | `str` | No | 10-500 characters | New description |
| `system_prompt` | `str` | No | Minimum 50 characters | New system prompt |
| `tools` | `list[str]` | No | Valid: voice, websearch, calendar, communication, crm | Replace tool list |
| `model` | `str` | No | provider:model format | New model identifier |
| `stratum` | `str` | No | Valid: RTI, RAI, ZACS, EEI, IGE | New stratum classification |
| `tags` | `list[str]` | No | Max 10, each 2-32 chars, lowercase | Replace tags list |

#### Return Values

**Success:**
```
SUCCESS: Agent 'lead_qualifier' updated (v1 -> v2).

Changes applied:
  - description: "Qualifies inbound leads..." -> "Enhanced lead qualification with MEDDIC..."
  - tools: ["communication", "crm"] -> ["communication", "crm", "websearch"]
  - tags: ["sales"] -> ["sales", "meddic", "production"]

Agent Summary:
  Name: lead_qualifier
  Description: Enhanced lead qualification with MEDDIC framework
  Tools: communication, crm, websearch
  Stratum: RAI
  Version: 2
  Last Updated: 2025-01-10T09:15:00+00:00

Saved to: /path/to/outputs/agents/lead_qualifier.json
```

**Agent Not Found:**
```
ERROR: Agent 'invalid_name' not found.

Available agents (3):
  - appointment_scheduler
  - lead_qualifier
  - research_assistant

Use one of the names above, or create a new agent with create_agent_config().
```

**No Changes Specified:**
```
ERROR: No changes specified for agent 'lead_qualifier'.

At least one of the following parameters must be provided:
  - description
  - system_prompt
  - tools
  - model
  - stratum
  - tags

Example:
  update_agent_config('lead_qualifier', description='New description')
```

**Validation Error:**
```
ERROR: Update validation failed for agent 'lead_qualifier'.

Validation errors:
  - description: String should have at least 10 characters

The agent was NOT modified. Please fix the issues and try again.
```

#### Behavior

1. Load existing agent configuration
2. Validate all provided fields against schema constraints
3. Apply only the fields that were explicitly provided (partial update)
4. Increment version number
5. Update `updated_at` timestamp
6. Preserve `created_at`, execution statistics, and history
7. Save updated configuration with file locking

#### Partial Update Semantics

- **Omitted fields**: Remain unchanged (use `None` or don't pass)
- **Empty list `[]`**: Clears the field (e.g., `tools=[]` removes all tools)
- **Explicit value**: Replaces the current value

```python
# Only update description, keep everything else
update_agent_config("lead_qualifier", description="New description")

# Clear all tags
update_agent_config("lead_qualifier", tags=[])

# Update multiple fields
update_agent_config(
    "lead_qualifier",
    description="New description",
    tools=["websearch", "crm"],
    tags=["research", "production"]
)
```

---

### 7. `delete_agent_config`

Permanently deletes an agent configuration and optionally its execution history.

#### Signature

```python
@tool
def delete_agent_config(
    name: str,
    confirm: bool = False,
    delete_history: bool = True,
) -> str:
```

#### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `name` | `str` | Yes | - | Name of the agent to delete |
| `confirm` | `bool` | No | `False` | Must be `True` to actually delete |
| `delete_history` | `bool` | No | `True` | Also delete execution history file |

#### Return Values

**Dry Run (confirm=False):**
```
WARNING: This will permanently delete agent 'lead_qualifier'.

Agent Details:
  Description: Qualifies inbound leads by assessing BANT criteria
  Tools: communication, crm
  Stratum: RAI
  Version: 3
  Created: 2025-01-09T14:30:22+00:00
  Executions: 47 total (45 success, 2 errors)

Files to be deleted:
  - /path/to/outputs/agents/lead_qualifier.json
  - /path/to/outputs/agents/lead_qualifier_history.json

To confirm deletion, call:
  delete_agent_config('lead_qualifier', confirm=True)
```

**Success (confirm=True):**
```
SUCCESS: Agent 'lead_qualifier' has been permanently deleted.

Deleted files:
  - /path/to/outputs/agents/lead_qualifier.json
  - /path/to/outputs/agents/lead_qualifier_history.json

This action cannot be undone. To recreate the agent, use create_agent_config().
```

**Success (keep history):**
```
SUCCESS: Agent 'lead_qualifier' configuration deleted.

Deleted files:
  - /path/to/outputs/agents/lead_qualifier.json

Preserved:
  - /path/to/outputs/agents/lead_qualifier_history.json (execution history retained)

Note: History file retained for audit purposes. Delete manually if not needed.
```

**Agent Not Found:**
```
ERROR: Agent 'invalid_name' not found.

Available agents (3):
  - appointment_scheduler
  - lead_qualifier
  - research_assistant

No files were deleted.
```

#### Safety Design

1. **Two-step deletion**: Requires explicit `confirm=True` to prevent accidental deletion
2. **Dry run by default**: Shows what would be deleted without taking action
3. **History preservation option**: Can keep execution history for audit trails
4. **File locking**: Ensures no concurrent operations during deletion

---

### 8. `clone_agent_config`

Creates a copy of an existing agent with a new name. Useful for creating variations or templates.

#### Signature

```python
@tool
def clone_agent_config(
    source_name: str,
    new_name: str,
    description: Optional[str] = None,
    tags: Optional[list[str]] = None,
) -> str:
```

#### Parameters

| Parameter | Type | Required | Constraints | Description |
|-----------|------|----------|-------------|-------------|
| `source_name` | `str` | Yes | Must exist | Name of the agent to clone |
| `new_name` | `str` | Yes | snake_case, unique, 1-64 chars | Name for the new agent |
| `description` | `str` | No | 10-500 characters | Override description (defaults to source + " (clone)") |
| `tags` | `list[str]` | No | Max 10, each 2-32 chars | Override tags (defaults to source tags) |

#### Return Values

**Success:**
```
SUCCESS: Agent 'lead_qualifier' cloned to 'lead_qualifier_v2'.

Clone Summary:
  Source: lead_qualifier (v3)
  Target: lead_qualifier_v2 (v1)

New Agent Details:
  Name: lead_qualifier_v2
  Description: Qualifies inbound leads by assessing BANT criteria (clone)
  Tools: communication, crm
  Stratum: RAI
  Tags: sales, qualification

Metadata reset:
  - version: 1 (new)
  - created_at: 2025-01-10T10:30:00+00:00 (new)
  - execution_count: 0 (reset)
  - success_count: 0 (reset)
  - error_count: 0 (reset)

Saved to: /path/to/outputs/agents/lead_qualifier_v2.json

Next steps:
  - Use update_agent_config('lead_qualifier_v2', ...) to customize
  - Use execute_created_agent('lead_qualifier_v2', ...) to test
```

**Source Not Found:**
```
ERROR: Source agent 'invalid_name' not found.

Available agents (3):
  - appointment_scheduler
  - lead_qualifier
  - research_assistant

Use one of the names above as the source_name parameter.
```

**Target Already Exists:**
```
ERROR: Cannot clone to 'lead_qualifier' - agent already exists.

Options:
  1. Choose a different name: clone_agent_config('source', 'new_unique_name')
  2. Delete existing agent first: delete_agent_config('lead_qualifier', confirm=True)
  3. Update existing agent: update_agent_config('lead_qualifier', ...)
```

**Invalid Name:**
```
ERROR: Invalid new_name 'Lead-Qualifier-V2'.

Name must be snake_case:
  - Only lowercase letters, numbers, and underscores
  - Must start with a letter
  - Maximum 64 characters

Example valid names: lead_qualifier_v2, research_agent_new, test_clone_1
```

#### Clone Behavior

| Field | Behavior |
|-------|----------|
| `name` | Set to `new_name` |
| `description` | Use override or source + " (clone)" |
| `system_prompt` | Copied from source |
| `tools` | Copied from source |
| `model` | Copied from source |
| `stratum` | Copied from source |
| `tags` | Use override or copy from source |
| `_metadata.created_at` | Set to now |
| `_metadata.updated_at` | Set to now |
| `_metadata.version` | Reset to 1 |
| `_metadata.execution_count` | Reset to 0 |
| `_metadata.success_count` | Reset to 0 |
| `_metadata.error_count` | Reset to 0 |
| History file | NOT copied (new agent starts fresh) |

---

### 9. `search_agents`

Searches and filters agent configurations based on various criteria.

#### Signature

```python
@tool
def search_agents(
    query: Optional[str] = None,
    stratum: Optional[str] = None,
    tools: Optional[list[str]] = None,
    tags: Optional[list[str]] = None,
    created_after: Optional[str] = None,
    created_before: Optional[str] = None,
    min_executions: Optional[int] = None,
    sort_by: str = "name",
    limit: int = 20,
) -> str:
```

#### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `query` | `str` | No | - | Search in name and description (case-insensitive) |
| `stratum` | `str` | No | - | Filter by stratum (RTI, RAI, ZACS, EEI, IGE) |
| `tools` | `list[str]` | No | - | Filter by tools (agents must have ALL specified tools) |
| `tags` | `list[str]` | No | - | Filter by tags (agents must have ALL specified tags) |
| `created_after` | `str` | No | - | ISO 8601 date, filter agents created after this date |
| `created_before` | `str` | No | - | ISO 8601 date, filter agents created before this date |
| `min_executions` | `int` | No | - | Minimum execution count |
| `sort_by` | `str` | No | `"name"` | Sort field: name, created_at, updated_at, execution_count |
| `limit` | `int` | No | `20` | Maximum results (1-100) |

#### Return Values

**Results Found:**
```
Search Results
==================================================

Query: "lead"
Filters: stratum=RAI, tools=[communication]
Sort: execution_count (descending)
Found: 2 agent(s)

----------------------------------------
1. lead_qualifier
   Description: Qualifies inbound leads by assessing BANT criteria
   Stratum: RAI
   Tools: communication, crm
   Tags: sales, qualification, production
   Executions: 47 (95.7% success rate)
   Created: 2025-01-09T14:30:22+00:00
   Last Updated: 2025-01-10T09:15:00+00:00

2. lead_researcher
   Description: Researches lead information before outreach
   Stratum: RAI
   Tools: communication, websearch
   Tags: sales, research
   Executions: 12 (100% success rate)
   Created: 2025-01-08T10:00:00+00:00
   Last Updated: 2025-01-08T10:00:00+00:00

==================================================
Use get_agent_config('<name>') for full details.
```

**No Results:**
```
Search Results
==================================================

Query: "nonexistent"
Filters: stratum=IGE
Found: 0 agent(s)

No agents match the search criteria.

Suggestions:
  - Broaden your search by removing filters
  - Use list_created_agents() to see all agents
  - Create a new agent with create_agent_config()

Current agents by stratum:
  - RTI: 2 agent(s)
  - RAI: 3 agent(s)
  - ZACS: 1 agent(s)
  - EEI: 0 agent(s)
  - IGE: 0 agent(s)
```

**Invalid Filter:**
```
ERROR: Invalid search parameter.

Validation errors:
  - stratum: 'INVALID' is not a valid stratum. Use: RTI, RAI, ZACS, EEI, IGE
  - tools: 'invalid_tool' is not a valid tool. Use: voice, websearch, calendar, communication, crm
  - created_after: Invalid date format. Use ISO 8601: YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS

Please fix the issues and try again.
```

#### Search Semantics

| Filter | Logic |
|--------|-------|
| `query` | Substring match (case-insensitive) in name OR description |
| `stratum` | Exact match |
| `tools` | Agent must have ALL specified tools (AND) |
| `tags` | Agent must have ALL specified tags (AND) |
| `created_after` | `created_at >= created_after` |
| `created_before` | `created_at <= created_before` |
| `min_executions` | `execution_count >= min_executions` |
| Multiple filters | Combined with AND |

#### Sort Options

| Value | Description |
|-------|-------------|
| `name` | Alphabetical by name (ascending) |
| `created_at` | Newest first (descending) |
| `updated_at` | Most recently updated first (descending) |
| `execution_count` | Most executed first (descending) |

#### Example Queries

```python
# Find all RAI agents with CRM integration
search_agents(stratum="RAI", tools=["crm"])

# Find recently created agents
search_agents(created_after="2025-01-01", sort_by="created_at")

# Find high-usage production agents
search_agents(tags=["production"], min_executions=50, sort_by="execution_count")

# Free text search
search_agents(query="scheduler appointment")
```

---

### 10. `get_agent_history`

Retrieves the execution history for a specific agent.

#### Signature

```python
@tool
def get_agent_history(
    name: str,
    limit: int = 10,
    include_errors_only: bool = False,
) -> str:
```

#### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `name` | `str` | Yes | - | Agent name |
| `limit` | `int` | No | `10` | Number of recent executions to return (1-50) |
| `include_errors_only` | `bool` | No | `False` | Only show failed executions |

#### Return Values

**Success:**
```
Execution History: lead_qualifier
==================================================

Statistics:
  Total Executions: 47
  Success Rate: 95.7% (45/47)
  Average Duration: 3.1s
  Last Executed: 2025-01-10T09:20:45+00:00

Recent Executions (showing 10 of 47):
----------------------------------------

1. [SUCCESS] 2025-01-10T09:20:45Z (3.2s)
   Task: Qualify this lead: John from Acme Corp, $50k budget
   Result: Lead qualified as HIGH priority - meets BANT criteria
   Tools: 2 calls

2. [SUCCESS] 2025-01-10T08:15:30Z (2.8s)
   Task: Qualify lead: Jane from StartupXYZ
   Result: Lead qualified as MEDIUM priority - needs budget confirmation
   Tools: 1 call

3. [ERROR] 2025-01-09T16:45:00Z (1.2s)
   Task: Qualify this lead: [corrupted data]
   Error: ExecutionError - Invalid input format
   Tools: 0 calls

...

==================================================
Full history stored in: outputs/agents/lead_qualifier_history.json
```

**No History:**
```
Execution History: lead_qualifier
==================================================

No execution history found.

This agent has not been executed yet. To test it:
  execute_created_agent('lead_qualifier', 'Your test task here')
```

**Agent Not Found:**
```
ERROR: Agent 'invalid_name' not found.

Available agents (3):
  - appointment_scheduler
  - lead_qualifier
  - research_assistant

Use one of the names above.
```

---

## Async Tool Variants

All tools have async variants for FastAPI integration:

```python
ASYNC_TOOLS = {
    # Core CRUD
    "create_agent_config": create_agent_config_async,
    "get_agent_config": get_agent_config_async,
    "update_agent_config": update_agent_config_async,
    "delete_agent_config": delete_agent_config_async,
    "clone_agent_config": clone_agent_config_async,

    # Listing and Search
    "list_created_agents": list_created_agents_async,
    "search_agents": search_agents_async,

    # Execution
    "execute_created_agent": execute_created_agent_async,
    "get_agent_history": get_agent_history_async,
}
```

---

## Version History

- **2.0.0** (2025-01-10): Enhanced Agent Persistence (Task 2.5.4)
  - **Enhanced Metadata Schema**
    - Added `updated_at`, `last_executed_at` timestamps
    - Changed `version` from semver string to integer (incremented on update)
    - Added `execution_count`, `success_count`, `error_count` statistics
  - **New Agent Fields**
    - Added `tags` field for categorization (max 10, lowercase)
    - Reserved tags: deprecated, experimental, production, template
  - **New Tools**
    - `update_agent_config`: Partial updates with version increment
    - `delete_agent_config`: Safe deletion with confirmation
    - `clone_agent_config`: Create copies with fresh metadata
    - `search_agents`: Search/filter by multiple criteria
    - `get_agent_history`: Retrieve execution history
  - **Execution History**
    - Separate history file per agent (`{name}_history.json`)
    - Max 50 records with FIFO management
    - Tracks task, result summary, duration, tool calls
  - **Breaking Changes**
    - `_metadata.version` changed from string "1.0.0" to integer 1
    - Existing agents without new metadata fields will be migrated on first access

- **1.1.0** (2025-01-09): Added execute_created_agent tool
  - New tool for immediate agent testing after creation
  - Supports timeout configuration
  - Graceful degradation when tools are unavailable
  - Detailed execution reporting with tool call traces
  - Async variant for FastAPI integration

- **1.0.0** (2025-01-09): Initial API contract
  - `create_agent_config`: Create and save agent configurations
  - `list_available_tools`: List MCP tool categories
  - `get_agent_config`: Retrieve saved configurations
  - `list_created_agents`: List all saved agents

---

## Migration Notes (v1.x to v2.0)

### Automatic Migration

When a v1.x agent config is loaded by any tool, it will be automatically migrated:

```python
def _migrate_v1_metadata(metadata: dict) -> dict:
    """Migrate v1.x metadata to v2.0 format."""
    if isinstance(metadata.get("version"), str):
        # Convert "1.0.0" to 1
        metadata["version"] = 1

    # Add missing fields with defaults
    metadata.setdefault("updated_at", metadata["created_at"])
    metadata.setdefault("last_executed_at", None)
    metadata.setdefault("execution_count", 0)
    metadata.setdefault("success_count", 0)
    metadata.setdefault("error_count", 0)

    return metadata
```

### Manual Migration Script

For bulk migration without loading each agent:

```bash
python -m src.tools migrate_agents
```

### Backwards Compatibility

- All v1.x tools continue to work with v2.0 agents
- v2.0 tools gracefully handle v1.x agents (auto-migrate on access)
- History files are created lazily (only on first execution)
- Tags default to empty list if not present
