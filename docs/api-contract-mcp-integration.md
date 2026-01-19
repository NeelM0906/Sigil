# API Contract: src/mcp_integration.py

## Overview

This document defines the API contract for `src/mcp_integration.py`, the MCP (Model Context Protocol) integration module for the ACTi Agent Builder. This module bridges agent configurations created by the builder with real MCP tool servers, enabling agents to execute actual tool operations.

The module provides:
- Connection management to MCP servers via `langchain-mcp-adapters`
- Tool registry mapping agent tool categories to MCP server configurations
- Agent instantiation with live MCP tools attached
- Validation and error handling for tool availability

---

## Design Principles

1. **Async-First**: All MCP operations are asynchronous; the API reflects this
2. **Fail-Safe Defaults**: Missing credentials or unavailable servers produce clear errors, not crashes
3. **Registry-Based Configuration**: MCP server configs centralized in `MCP_SERVERS` registry
4. **Separation of Concerns**: Tool connection separate from agent creation
5. **Graceful Degradation**: Agents can be created with partial tool sets when some servers unavailable

---

## Module Dependencies

```python
from langchain_mcp_adapters.client import MultiServerMCPClient
from deepagents import create_deep_agent

from .schemas import AgentConfig
from .prompts import DEFAULT_MODEL
```

### External Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `langchain-mcp-adapters` | >=0.1.0 | MCP client for tool server connections |
| `deepagents` | >=0.3.5 | Agent orchestration framework |

### Environment Variables

| Variable | Required For | Description |
|----------|--------------|-------------|
| `ELEVENLABS_API_KEY` | voice | ElevenLabs TTS API key |
| `TAVILY_API_KEY` | websearch | Tavily search API key |
| `GOOGLE_OAUTH_CREDENTIALS` | calendar | Path to Google OAuth credentials JSON |
| `TWILIO_ACCOUNT_SID` | communication | Twilio account SID |
| `TWILIO_API_KEY` | communication | Twilio API key |
| `TWILIO_API_SECRET` | communication | Twilio API secret |
| `HUBSPOT_API_KEY` | crm | HubSpot API key |

---

## Constants

### `MCP_SERVERS`

Registry mapping tool category names to MCP server configurations.

```python
MCP_SERVERS: dict[str, MCPServerConfig] = {
    "voice": {
        "command": "uvx",
        "args": ["elevenlabs-mcp"],
        "transport": "stdio",
        "env": {"ELEVENLABS_API_KEY": os.getenv("ELEVENLABS_API_KEY")},
    },
    "websearch": {
        "command": "npx",
        "args": ["-y", "tavily-mcp@latest"],
        "transport": "stdio",
        "env": {"TAVILY_API_KEY": os.getenv("TAVILY_API_KEY")},
    },
    "calendar": {
        "command": "npx",
        "args": ["@anthropic-ai/google-calendar-mcp"],
        "transport": "stdio",
        "env": {"GOOGLE_OAUTH_CREDENTIALS": os.getenv("GOOGLE_OAUTH_CREDENTIALS")},
    },
    "communication": {
        "command": "npx",
        "args": ["-y", "@twilio-alpha/mcp"],
        "transport": "stdio",
        "env": {
            "TWILIO_ACCOUNT_SID": os.getenv("TWILIO_ACCOUNT_SID"),
            "TWILIO_API_KEY": os.getenv("TWILIO_API_KEY"),
            "TWILIO_API_SECRET": os.getenv("TWILIO_API_SECRET"),
        },
    },
    "crm": {
        "command": "npx",
        "args": ["-y", "@hubspot/mcp-server"],
        "transport": "stdio",
        "env": {"HUBSPOT_API_KEY": os.getenv("HUBSPOT_API_KEY")},
    },
}
```

---

## Type Definitions

### `MCPServerConfig`

Configuration for a single MCP server connection.

```python
class MCPServerConfig(TypedDict, total=False):
    """Configuration for an MCP server connection."""
    command: str                    # Executable command (stdio transport)
    args: list[str]                 # Command arguments
    transport: Literal["stdio", "http", "sse"]  # Connection transport type
    env: dict[str, str | None]      # Environment variables for the server
    url: str                        # Server URL (http/sse transport)
    headers: dict[str, str]         # HTTP headers (http/sse transport)
```

### `MCPConnectionResult`

Result of attempting to connect to MCP servers.

```python
class MCPConnectionResult(NamedTuple):
    """Result of MCP server connection attempt."""
    client: MultiServerMCPClient | None  # Connected client, or None if all failed
    connected_tools: list[str]           # Successfully connected tool categories
    failed_tools: list[str]              # Failed tool categories
    errors: dict[str, str]               # Tool category -> error message
```

### `ToolValidationResult`

Result of validating tool availability.

```python
class ToolValidationResult(NamedTuple):
    """Result of tool availability validation."""
    valid: bool                     # True if all tools are available
    available: list[str]            # Tools that are available
    missing: list[str]              # Tools that are missing or misconfigured
    messages: list[str]             # Human-readable status messages
```

---

## Function Specifications

### 1. `create_agent_with_tools`

Main entry point: Creates a deepagent instance with real MCP tools attached.

#### Signature

```python
async def create_agent_with_tools(
    agent_config: AgentConfig,
    *,
    skip_unavailable: bool = False,
    timeout: float = 30.0,
) -> CompiledStateGraph:
```

#### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `agent_config` | `AgentConfig` | Yes | - | Agent configuration from builder |
| `skip_unavailable` | `bool` | No | `False` | If True, create agent with available tools only; if False, raise on missing tools |
| `timeout` | `float` | No | `30.0` | Connection timeout per MCP server in seconds |

#### Return Value

Returns a `CompiledStateGraph` (deepagents agent) configured with:
- System prompt from `agent_config.system_prompt`
- Model from `agent_config.model`
- MCP tools from connected servers

#### Behavior

1. Validates that requested tools exist in `MCP_SERVERS` registry
2. Checks that required environment variables are set for each tool
3. Connects to MCP servers for requested tools
4. Retrieves LangChain-compatible tools from MCP client
5. Creates deepagent with system prompt, model, and tools

#### Error Handling

| Error Type | Condition | Behavior |
|------------|-----------|----------|
| `MCPConfigurationError` | Unknown tool category requested | Raised immediately |
| `MCPCredentialError` | Required env var missing | Raised if `skip_unavailable=False` |
| `MCPConnectionError` | Server connection failed | Raised if `skip_unavailable=False`, logged if True |
| `MCPTimeoutError` | Connection exceeded timeout | Raised if `skip_unavailable=False` |

#### Usage Examples

**Basic Usage:**

```python
import asyncio
from src.mcp_integration import create_agent_with_tools
from src.schemas import AgentConfig

async def main():
    config = AgentConfig(
        name="appointment_scheduler",
        description="Schedules appointments via calendar",
        system_prompt="You are an appointment scheduling assistant...",
        tools=["calendar", "communication"],
        model="anthropic:claude-opus-4-5-20251101",
        stratum="ZACS",
    )

    agent = await create_agent_with_tools(config)

    result = await agent.ainvoke({
        "messages": [{"role": "user", "content": "Schedule a meeting for tomorrow at 2pm"}]
    })

    return result["messages"][-1].content

response = asyncio.run(main())
```

**With Graceful Degradation:**

```python
# Create agent even if some tools unavailable
agent = await create_agent_with_tools(
    config,
    skip_unavailable=True,  # Won't raise if calendar MCP server unavailable
)
```

**With Custom Timeout:**

```python
agent = await create_agent_with_tools(
    config,
    timeout=60.0,  # Allow more time for slow connections
)
```

---

### 2. `connect_mcp_servers`

Connects to specified MCP servers and returns a client with tools.

#### Signature

```python
async def connect_mcp_servers(
    tool_names: list[str],
    *,
    timeout: float = 30.0,
) -> MCPConnectionResult:
```

#### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `tool_names` | `list[str]` | Yes | - | Tool category names to connect |
| `timeout` | `float` | No | `30.0` | Connection timeout per server |

#### Return Value

Returns `MCPConnectionResult` containing:
- `client`: Connected `MultiServerMCPClient` or None if all failed
- `connected_tools`: List of successfully connected tool categories
- `failed_tools`: List of tool categories that failed to connect
- `errors`: Dict mapping failed tool to error message

#### Behavior

1. Filters `tool_names` to only those in `MCP_SERVERS` registry
2. Validates credentials for each requested tool
3. Builds server configuration dict for `MultiServerMCPClient`
4. Attempts connection with timeout
5. Returns result with success/failure details

#### Usage Example

```python
from src.mcp_integration import connect_mcp_servers

result = await connect_mcp_servers(["calendar", "websearch", "voice"])

if result.client:
    print(f"Connected to: {result.connected_tools}")
    tools = await result.client.get_tools()
else:
    print(f"All connections failed: {result.errors}")

if result.failed_tools:
    print(f"Failed tools: {result.failed_tools}")
    for tool, error in result.errors.items():
        print(f"  {tool}: {error}")
```

---

### 3. `get_mcp_tools`

Retrieves LangChain-compatible tools from an MCP client.

#### Signature

```python
async def get_mcp_tools(
    client: MultiServerMCPClient,
) -> list[BaseTool]:
```

#### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `client` | `MultiServerMCPClient` | Yes | Connected MCP client |

#### Return Value

Returns a list of `BaseTool` objects from `langchain_core.tools` that can be passed directly to `create_deep_agent()`.

#### Usage Example

```python
from src.mcp_integration import connect_mcp_servers, get_mcp_tools

result = await connect_mcp_servers(["websearch"])
if result.client:
    tools = await get_mcp_tools(result.client)
    print(f"Retrieved {len(tools)} tools")
    for tool in tools:
        print(f"  - {tool.name}: {tool.description}")
```

---

### 4. `validate_agent_tools`

Validates that an agent's requested tools are available and properly configured.

#### Signature

```python
async def validate_agent_tools(
    agent_config: AgentConfig,
) -> ToolValidationResult:
```

#### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `agent_config` | `AgentConfig` | Yes | Agent configuration to validate |

#### Return Value

Returns `ToolValidationResult` containing:
- `valid`: True if all requested tools are available
- `available`: List of tools that are properly configured
- `missing`: List of tools that are missing or misconfigured
- `messages`: Human-readable messages describing tool status

#### Behavior

Checks for each tool in `agent_config.tools`:
1. Tool category exists in `MCP_SERVERS` registry
2. Required environment variables are set and non-empty
3. Server configuration is complete

Does NOT attempt actual connections (use `connect_mcp_servers` for that).

#### Usage Example

```python
from src.mcp_integration import validate_agent_tools
from src.schemas import AgentConfig

config = AgentConfig(
    name="test_agent",
    description="Test agent for validation",
    system_prompt="You are a test agent...",
    tools=["calendar", "websearch", "invalid_tool"],
)

result = await validate_agent_tools(config)

if result.valid:
    print("All tools available!")
else:
    print(f"Available: {result.available}")
    print(f"Missing: {result.missing}")
    for msg in result.messages:
        print(f"  - {msg}")
```

**Example Output:**

```
Available: ['websearch']
Missing: ['calendar', 'invalid_tool']
  - calendar: Missing GOOGLE_OAUTH_CREDENTIALS environment variable
  - invalid_tool: Unknown tool category. Valid options: voice, websearch, calendar, communication, crm
```

---

### 5. `get_server_config`

Retrieves the MCP server configuration for a tool category.

#### Signature

```python
def get_server_config(
    tool_name: str,
) -> MCPServerConfig | None:
```

#### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `tool_name` | `str` | Yes | Tool category name |

#### Return Value

Returns `MCPServerConfig` if tool exists in registry, `None` otherwise.

#### Usage Example

```python
from src.mcp_integration import get_server_config

config = get_server_config("voice")
if config:
    print(f"Voice server: {config['command']} {' '.join(config['args'])}")
else:
    print("Voice tool not configured")
```

---

### 6. `list_configured_tools`

Lists all tool categories configured in the MCP server registry.

#### Signature

```python
def list_configured_tools() -> list[str]:
```

#### Return Value

Returns list of tool category names available in `MCP_SERVERS`.

#### Usage Example

```python
from src.mcp_integration import list_configured_tools

tools = list_configured_tools()
print(f"Configured tools: {', '.join(tools)}")
# Output: Configured tools: voice, websearch, calendar, communication, crm
```

---

## Exception Classes

### `MCPIntegrationError`

Base exception for all MCP integration errors.

```python
class MCPIntegrationError(Exception):
    """Base exception for MCP integration errors."""
    pass
```

### `MCPConfigurationError`

Raised when MCP server configuration is invalid.

```python
class MCPConfigurationError(MCPIntegrationError):
    """Raised when MCP server configuration is invalid."""

    def __init__(self, tool_name: str, message: str):
        self.tool_name = tool_name
        super().__init__(f"Configuration error for '{tool_name}': {message}")
```

### `MCPCredentialError`

Raised when required credentials are missing.

```python
class MCPCredentialError(MCPIntegrationError):
    """Raised when required credentials/environment variables are missing."""

    def __init__(self, tool_name: str, missing_vars: list[str]):
        self.tool_name = tool_name
        self.missing_vars = missing_vars
        vars_str = ", ".join(missing_vars)
        super().__init__(
            f"Missing credentials for '{tool_name}': {vars_str}. "
            f"Set these environment variables to use this tool."
        )
```

### `MCPConnectionError`

Raised when connection to MCP server fails.

```python
class MCPConnectionError(MCPIntegrationError):
    """Raised when connection to MCP server fails."""

    def __init__(self, tool_name: str, original_error: Exception):
        self.tool_name = tool_name
        self.original_error = original_error
        super().__init__(
            f"Failed to connect to MCP server for '{tool_name}': {original_error}"
        )
```

### `MCPTimeoutError`

Raised when MCP server connection times out.

```python
class MCPTimeoutError(MCPIntegrationError):
    """Raised when MCP server connection times out."""

    def __init__(self, tool_name: str, timeout: float):
        self.tool_name = tool_name
        self.timeout = timeout
        super().__init__(
            f"Connection to MCP server '{tool_name}' timed out after {timeout}s"
        )
```

---

## Integration Patterns

### FastAPI Endpoint: Run Agent

```python
from fastapi import FastAPI, HTTPException
from src.mcp_integration import create_agent_with_tools, MCPIntegrationError
from src.schemas import AgentConfig, RunAgentRequest, AgentResponse
from langchain_core.messages import HumanMessage
import json

app = FastAPI(title="ACTi Agent Runner API")

@app.post("/agents/{agent_name}/run", response_model=AgentResponse)
async def run_agent(agent_name: str, request: RunAgentRequest):
    """Execute an agent with real MCP tools."""

    # Load agent config from disk
    config_path = Path(f"outputs/agents/{agent_name}.json")
    if not config_path.exists():
        raise HTTPException(status_code=404, detail=f"Agent '{agent_name}' not found")

    config_data = json.loads(config_path.read_text())
    agent_config = AgentConfig(**config_data)

    try:
        # Create agent with MCP tools
        agent = await create_agent_with_tools(agent_config)

        # Run the agent
        result = await agent.ainvoke({
            "messages": [HumanMessage(content=request.message)]
        })

        return AgentResponse(
            message=result["messages"][-1].content,
            tool_calls=extract_tool_calls(result),
        )

    except MCPIntegrationError as e:
        raise HTTPException(status_code=503, detail=str(e))
```

### FastAPI Endpoint: Validate Agent Tools

```python
from src.mcp_integration import validate_agent_tools

@app.post("/agents/{agent_name}/validate")
async def validate_agent(agent_name: str):
    """Validate that an agent's tools are available."""

    config_path = Path(f"outputs/agents/{agent_name}.json")
    if not config_path.exists():
        raise HTTPException(status_code=404, detail=f"Agent '{agent_name}' not found")

    config_data = json.loads(config_path.read_text())
    agent_config = AgentConfig(**config_data)

    result = await validate_agent_tools(agent_config)

    return {
        "valid": result.valid,
        "available_tools": result.available,
        "missing_tools": result.missing,
        "messages": result.messages,
    }
```

### CLI: Test MCP Connection

```python
import asyncio
from src.mcp_integration import connect_mcp_servers, get_mcp_tools

async def test_mcp_connection(tool_names: list[str]):
    """Test MCP server connections for specified tools."""

    print(f"Testing connection to: {', '.join(tool_names)}")

    result = await connect_mcp_servers(tool_names)

    if result.connected_tools:
        print(f"\nConnected successfully to: {', '.join(result.connected_tools)}")

        if result.client:
            tools = await get_mcp_tools(result.client)
            print(f"\nRetrieved {len(tools)} tools:")
            for tool in tools:
                print(f"  - {tool.name}")

    if result.failed_tools:
        print(f"\nFailed to connect: {', '.join(result.failed_tools)}")
        for tool, error in result.errors.items():
            print(f"  {tool}: {error}")

if __name__ == "__main__":
    asyncio.run(test_mcp_connection(["websearch", "calendar"]))
```

---

## Module Exports

```python
__all__ = [
    # Constants
    "MCP_SERVERS",

    # Main functions
    "create_agent_with_tools",
    "connect_mcp_servers",
    "get_mcp_tools",
    "validate_agent_tools",

    # Helper functions
    "get_server_config",
    "list_configured_tools",

    # Exceptions
    "MCPIntegrationError",
    "MCPConfigurationError",
    "MCPCredentialError",
    "MCPConnectionError",
    "MCPTimeoutError",

    # Types
    "MCPServerConfig",
    "MCPConnectionResult",
    "ToolValidationResult",
]
```

---

## Testing Patterns

### Unit Test: Validate Tools

```python
import pytest
from unittest.mock import patch
from src.mcp_integration import validate_agent_tools
from src.schemas import AgentConfig

@pytest.mark.asyncio
async def test_validate_tools_all_available():
    """All tools pass validation when credentials present."""
    config = AgentConfig(
        name="test_agent",
        description="Test agent",
        system_prompt="You are a test agent...",
        tools=["websearch"],
    )

    with patch.dict("os.environ", {"TAVILY_API_KEY": "test-key"}):
        result = await validate_agent_tools(config)

    assert result.valid
    assert "websearch" in result.available
    assert not result.missing

@pytest.mark.asyncio
async def test_validate_tools_missing_credentials():
    """Validation fails when credentials missing."""
    config = AgentConfig(
        name="test_agent",
        description="Test agent",
        system_prompt="You are a test agent...",
        tools=["calendar"],
    )

    with patch.dict("os.environ", {}, clear=True):
        result = await validate_agent_tools(config)

    assert not result.valid
    assert "calendar" in result.missing
    assert any("GOOGLE_OAUTH_CREDENTIALS" in msg for msg in result.messages)
```

### Integration Test: Connect MCP Server

```python
import pytest
from src.mcp_integration import connect_mcp_servers

@pytest.mark.asyncio
@pytest.mark.integration
async def test_connect_websearch_server():
    """Integration test for websearch MCP server connection."""
    result = await connect_mcp_servers(["websearch"], timeout=10.0)

    # May fail if no API key, but should not raise
    if result.client:
        assert "websearch" in result.connected_tools
    else:
        assert "websearch" in result.failed_tools
```

### Mock MCP Client

```python
import pytest
from unittest.mock import AsyncMock, MagicMock

@pytest.fixture
def mock_mcp_client():
    """Create a mock MCP client for testing."""
    client = MagicMock()

    # Mock get_tools to return fake tools
    mock_tool = MagicMock()
    mock_tool.name = "test_tool"
    mock_tool.description = "A test tool"

    client.get_tools = AsyncMock(return_value=[mock_tool])

    return client
```

---

## Error Handling Strategy

### Credential Validation

Before attempting MCP connections, validate that required environment variables exist:

```python
REQUIRED_ENV_VARS = {
    "voice": ["ELEVENLABS_API_KEY"],
    "websearch": ["TAVILY_API_KEY"],
    "calendar": ["GOOGLE_OAUTH_CREDENTIALS"],
    "communication": ["TWILIO_ACCOUNT_SID", "TWILIO_API_KEY", "TWILIO_API_SECRET"],
    "crm": ["HUBSPOT_API_KEY"],
}

def _check_credentials(tool_name: str) -> list[str]:
    """Return list of missing environment variables for a tool."""
    required = REQUIRED_ENV_VARS.get(tool_name, [])
    missing = [var for var in required if not os.getenv(var)]
    return missing
```

### Connection Retry Strategy

For transient failures, implement exponential backoff:

```python
async def _connect_with_retry(
    config: dict,
    max_retries: int = 3,
    base_delay: float = 1.0,
) -> MultiServerMCPClient:
    """Attempt MCP connection with exponential backoff."""
    last_error = None

    for attempt in range(max_retries):
        try:
            client = MultiServerMCPClient(config)
            await client.get_tools()  # Verify connection works
            return client
        except Exception as e:
            last_error = e
            if attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt)
                await asyncio.sleep(delay)

    raise MCPConnectionError("unknown", last_error)
```

### Graceful Degradation

When `skip_unavailable=True`, log failures but continue:

```python
import logging

logger = logging.getLogger(__name__)

async def create_agent_with_tools(config: AgentConfig, skip_unavailable: bool = False):
    result = await connect_mcp_servers(config.tools)

    if result.failed_tools:
        if skip_unavailable:
            for tool, error in result.errors.items():
                logger.warning(f"Tool '{tool}' unavailable: {error}")
        else:
            raise MCPConnectionError(
                result.failed_tools[0],
                Exception(result.errors[result.failed_tools[0]])
            )

    # Continue with available tools...
```

---

## Configuration File Structure

For future extensibility, MCP servers can also be configured via YAML:

```yaml
# mcp_servers.yaml
servers:
  voice:
    command: uvx
    args: [elevenlabs-mcp]
    transport: stdio
    env:
      ELEVENLABS_API_KEY: ${ELEVENLABS_API_KEY}

  websearch:
    command: npx
    args: [-y, tavily-mcp@latest]
    transport: stdio
    env:
      TAVILY_API_KEY: ${TAVILY_API_KEY}

  # HTTP-based server example
  custom_api:
    url: https://api.example.com/mcp
    transport: http
    headers:
      Authorization: Bearer ${CUSTOM_API_KEY}
```

---

## Performance Considerations

1. **Connection Pooling**: `MultiServerMCPClient` manages server connections; reuse the client when possible
2. **Lazy Loading**: Don't connect to MCP servers until tools are actually needed
3. **Timeout Management**: Set reasonable timeouts (30s default) to prevent hanging
4. **Parallel Connections**: When connecting to multiple servers, connect in parallel when possible

---

## Version History

- **1.0.0** (2025-01-09): Initial API contract
  - `create_agent_with_tools`: Main entry point for agent instantiation
  - `connect_mcp_servers`: Low-level server connection
  - `get_mcp_tools`: Tool retrieval from MCP client
  - `validate_agent_tools`: Pre-flight validation
  - Exception hierarchy for error handling
  - FastAPI integration patterns
