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
    model: str = "anthropic:claude-sonnet-4-20250514",
    stratum: Optional[str] = None,
) -> str:
```

#### Parameters

| Parameter | Type | Required | Constraints | Description |
|-----------|------|----------|-------------|-------------|
| `name` | `str` | Yes | snake_case, 1-64 chars, starts with letter | Unique identifier for the agent |
| `description` | `str` | Yes | 10-500 characters | What the agent does |
| `system_prompt` | `str` | Yes | Minimum 50 characters | Complete system prompt |
| `tools` | `list[str]` | No | Valid: voice, websearch, calendar, communication, crm | MCP tool categories |
| `model` | `str` | No | Default: anthropic:claude-sonnet-4-20250514 | LLM model identifier |
| `stratum` | `str` | No | Valid: RTI, RAI, ZACS, EEI, IGE | ACTi stratum classification |

#### Return Values

**Success:**
```
SUCCESS: Agent configuration created and saved!

Agent Summary:
  Name: lead_qualifier
  Description: Qualifies inbound leads by assessing BANT criteria
  Tools: communication, crm
  Stratum: RAI
  Model: anthropic:claude-sonnet-4-20250514

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
  "model": "anthropic:claude-sonnet-4-20250514",
  "stratum": "RAI",
  "_metadata": {
    "created_at": "2025-01-09T14:30:22.123456+00:00",
    "version": "1.0.0",
    "builder": "acti-agent-builder"
  }
}
```

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
  Model: anthropic:claude-sonnet-4-20250514

Full Configuration:
{
  "name": "lead_qualifier",
  "description": "Qualifies inbound leads by assessing BANT criteria",
  "system_prompt": "You are a professional lead qualification specialist...",
  "tools": ["communication", "crm"],
  "model": "anthropic:claude-sonnet-4-20250514",
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
    create_agent_config,
    list_available_tools,
    get_agent_config,
    list_created_agents,
]
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
    model="anthropic:claude-opus-4-20250514",
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
    appointment_scheduler.json
    research_assistant.json
```

---

## Version History

- **1.0.0** (2025-01-09): Initial API contract
  - `create_agent_config`: Create and save agent configurations
  - `list_available_tools`: List MCP tool categories
  - `get_agent_config`: Retrieve saved configurations
  - `list_created_agents`: List all saved agents
