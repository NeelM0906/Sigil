# API Contract: src/builder.py

## Overview

This document defines the API contract for `src/builder.py`, the main module of the ACTi Agent Builder. This module creates and configures the meta-agent (the "builder") that designs and generates other AI agents. The builder uses the deepagents framework to orchestrate its capabilities, including subagent delegation and LangChain tool integration.

---

## Design Principles

1. **Single Entry Point**: One primary function (`create_builder()`) returns a fully configured agent
2. **Immutable Configuration**: Builder configuration is defined at creation time, not mutated
3. **Subagent Delegation**: Complex tasks (prompt engineering, pattern analysis) delegated to specialized subagents
4. **Framework Agnostic Return**: Returns a deepagents agent that can be invoked via standard interfaces
5. **Environment Isolation**: All configuration loaded from environment variables or explicit parameters

---

## Module Dependencies

```python
from deepagents import create_deep_agent
from src.tools import BUILDER_TOOLS
from src.prompts import (
    DEFAULT_MODEL,
    BUILDER_SYSTEM_PROMPT,
    PROMPT_ENGINEER_SYSTEM_PROMPT,
    PATTERN_ANALYZER_SYSTEM_PROMPT,
)
```

### External Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `deepagents` | >=0.3.5 | Agent orchestration framework |
| `langchain` | >=1.2.0 | Tool infrastructure |
| `langchain-anthropic` | >=1.3.0 | Anthropic model provider |

---

## Function Specifications

### 1. `create_builder`

Creates and returns a fully configured meta-agent capable of designing other agents.

#### Signature

```python
def create_builder(
    model: str = DEFAULT_MODEL,
    include_subagents: bool = True,
    custom_tools: list | None = None,
) -> DeepAgent:
```

#### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `model` | `str` | No | `"anthropic:claude-opus-4-5-20251101"` | Model identifier for the builder agent |
| `include_subagents` | `bool` | No | `True` | Whether to attach subagents (prompt-engineer, pattern-analyzer) |
| `custom_tools` | `list` | No | `None` | Additional tools to merge with BUILDER_TOOLS |

#### Return Value

Returns a `DeepAgent` instance from the deepagents framework, configured with:
- System prompt: `BUILDER_SYSTEM_PROMPT`
- Tools: `BUILDER_TOOLS` (plus any `custom_tools`)
- Subagents: prompt-engineer, pattern-analyzer, general-purpose (if `include_subagents=True`)

#### Return Type

```python
DeepAgent  # from deepagents package
```

The returned agent supports these invocation methods:
- `agent.invoke(input_dict)` - Synchronous invocation
- `await agent.ainvoke(input_dict)` - Asynchronous invocation
- `agent.stream(input_dict)` - Streaming response (sync)
- `await agent.astream(input_dict)` - Streaming response (async)

#### Usage Examples

**Basic Usage:**

```python
from src.builder import create_builder
from langchain_core.messages import HumanMessage

# Create the builder agent
builder = create_builder()

# Invoke synchronously
result = builder.invoke({
    "messages": [HumanMessage(content="Create a lead qualification agent for B2B SaaS")]
})

# Access the response
response = result["messages"][-1].content
print(response)
```

**Async Usage:**

```python
import asyncio
from src.builder import create_builder
from langchain_core.messages import HumanMessage

async def main():
    builder = create_builder()

    result = await builder.ainvoke({
        "messages": [HumanMessage(content="Create an appointment scheduling agent")]
    })

    return result["messages"][-1].content

response = asyncio.run(main())
```

**Custom Configuration:**

```python
from src.builder import create_builder

# Create builder without subagents (lighter weight)
builder = create_builder(include_subagents=False)

# Create builder with custom model
builder = create_builder(model="anthropic:claude-sonnet-4-20250514")

# Create builder with additional tools
from langchain_core.tools import tool

@tool
def custom_tool(query: str) -> str:
    """Custom tool for specialized functionality."""
    return f"Custom result for: {query}"

builder = create_builder(custom_tools=[custom_tool])
```

#### Error Handling

| Error Type | Condition | Resolution |
|------------|-----------|------------|
| `ImportError` | deepagents not installed | Install: `pip install deepagents>=0.3.5` |
| `ValueError` | Invalid model identifier | Use format: `provider:model-name` |
| `EnvironmentError` | Missing ANTHROPIC_API_KEY | Set environment variable or use .env file |

---

### 2. `_create_subagent_config` (Internal)

Creates the configuration dictionary for a subagent.

#### Signature

```python
def _create_subagent_config(
    name: str,
    description: str,
    system_prompt: str,
    tools: list | None = None,
) -> dict:
```

#### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `name` | `str` | Yes | Unique identifier for the subagent |
| `description` | `str` | Yes | What the subagent does (shown to parent agent) |
| `system_prompt` | `str` | Yes | Complete system prompt for the subagent |
| `tools` | `list` | No | Tools available to the subagent (default: empty) |

#### Return Value

```python
{
    "name": "prompt-engineer",
    "description": "Crafts detailed, effective system prompts for agents",
    "system_prompt": "You are an expert system prompt engineer...",
    "tools": [],
}
```

---

### 3. `_get_default_subagents` (Internal)

Returns the default subagent configurations for the builder.

#### Signature

```python
def _get_default_subagents() -> list[dict]:
```

#### Return Value

Returns a list of three subagent configurations:

```python
[
    {
        "name": "prompt-engineer",
        "description": "Crafts detailed, effective system prompts for agents",
        "system_prompt": PROMPT_ENGINEER_SYSTEM_PROMPT,
        "tools": [],
    },
    {
        "name": "pattern-analyzer",
        "description": "Analyzes Bland/N8N configs to extract reusable patterns",
        "system_prompt": PATTERN_ANALYZER_SYSTEM_PROMPT,
        "tools": [],  # Uses built-in read_file, glob, grep
    },
    {
        "name": "general-purpose",
        "description": "Handles complex multi-step tasks that would clutter context",
        "system_prompt": "You are a helpful assistant...",
        "tools": [],  # Uses built-in filesystem tools
    },
]
```

---

## Configuration Options

### Model Configuration

The builder supports any model identifier in the `provider:model-name` format:

| Provider | Model ID | Notes |
|----------|----------|-------|
| Anthropic | `anthropic:claude-opus-4-5-20251101` | Default, best for complex reasoning |
| Anthropic | `anthropic:claude-sonnet-4-20250514` | Faster, lower cost |
| OpenAI | `openai:gpt-4-turbo` | Alternative provider |

### Subagent Configuration

The builder includes three specialized subagents by default:

| Subagent | Purpose | Invocation |
|----------|---------|------------|
| `prompt-engineer` | Crafts system prompts for new agents | `task(name="prompt-engineer", task="...")` |
| `pattern-analyzer` | Analyzes reference configs for patterns | `task(name="pattern-analyzer", task="...")` |
| `general-purpose` | Handles complex multi-step tasks | `task(name="general-purpose", task="...")` |

### Tools Configuration

The builder has access to `BUILDER_TOOLS` from `src/tools.py`:

| Tool | Purpose |
|------|---------|
| `create_agent_config` | Creates and saves agent configurations |
| `list_available_tools` | Lists available MCP tool categories |
| `get_agent_config` | Retrieves saved configurations |
| `list_created_agents` | Lists all created agents |

---

## Integration with deepagents

### Agent Creation Pattern

```python
from deepagents import create_deep_agent

agent = create_deep_agent(
    model="anthropic:claude-opus-4-5-20251101",
    tools=BUILDER_TOOLS,
    system_prompt=BUILDER_SYSTEM_PROMPT,
    subagents=[
        {
            "name": "prompt-engineer",
            "description": "Crafts detailed, effective system prompts for agents",
            "system_prompt": PROMPT_ENGINEER_SYSTEM_PROMPT,
            "tools": [],
        },
        {
            "name": "pattern-analyzer",
            "description": "Analyzes Bland/N8N configs to extract reusable patterns",
            "system_prompt": PATTERN_ANALYZER_SYSTEM_PROMPT,
            "tools": [],
        },
    ],
)
```

### Input/Output Contract

**Input Format:**

```python
{
    "messages": [
        HumanMessage(content="Create a lead qualification agent for B2B SaaS"),
    ]
}
```

**Output Format:**

```python
{
    "messages": [
        HumanMessage(content="Create a lead qualification agent for B2B SaaS"),
        AIMessage(content="I'll create a lead qualification agent for you..."),
        ToolMessage(name="list_available_tools", content="Available MCP Tool Categories..."),
        AIMessage(content="Based on the available tools..."),
        ToolMessage(name="create_agent_config", content="SUCCESS: Agent configuration created..."),
        AIMessage(content="I've created the lead_qualifier agent with the following configuration..."),
    ]
}
```

### Streaming Pattern

```python
async def stream_builder_response(prompt: str):
    builder = create_builder()

    async for chunk in builder.astream({
        "messages": [HumanMessage(content=prompt)]
    }):
        if "messages" in chunk:
            yield chunk["messages"][-1].content
```

---

## FastAPI Integration (Phase 3)

### Endpoint: Create Agent

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_core.messages import HumanMessage
from src.builder import create_builder
from src.schemas import CreateAgentRequest, CreateAgentResponse

app = FastAPI(title="ACTi Agent Builder API")

# Singleton builder instance (or create per-request)
_builder = None

def get_builder():
    global _builder
    if _builder is None:
        _builder = create_builder()
    return _builder

@app.post("/agents/create", response_model=CreateAgentResponse)
async def create_agent(request: CreateAgentRequest):
    """Create a new agent configuration via natural language prompt."""
    builder = get_builder()

    try:
        result = await builder.ainvoke({
            "messages": [HumanMessage(content=request.prompt)]
        })

        # Extract the final response
        final_message = result["messages"][-1].content

        # Parse agent config from tool output (implementation detail)
        agent_config = extract_created_config(result)

        return CreateAgentResponse(
            agent_config=agent_config,
            file_path=f"outputs/agents/{agent_config.name}.json",
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

### Endpoint: Stream Agent Creation

```python
from fastapi.responses import StreamingResponse
import json

@app.post("/agents/create/stream")
async def create_agent_stream(request: CreateAgentRequest):
    """Stream the agent creation process."""
    builder = get_builder()

    async def generate():
        async for chunk in builder.astream({
            "messages": [HumanMessage(content=request.prompt)]
        }):
            if "messages" in chunk:
                msg = chunk["messages"][-1]
                yield json.dumps({
                    "type": msg.type,
                    "content": msg.content,
                }) + "\n"

    return StreamingResponse(
        generate(),
        media_type="application/x-ndjson",
    )
```

### Endpoint: List Tools

```python
@app.get("/tools")
async def list_tools():
    """List available MCP tool categories."""
    builder = get_builder()

    result = await builder.ainvoke({
        "messages": [HumanMessage(content="List available tools")]
    })

    # Or directly return from schema
    from src.schemas import MCP_TOOL_CATEGORIES
    return {"tools": list(MCP_TOOL_CATEGORIES.keys())}
```

### Endpoint: Get Agent Configuration

```python
@app.get("/agents/{agent_name}")
async def get_agent(agent_name: str):
    """Retrieve a saved agent configuration."""
    from pathlib import Path
    import json

    config_path = Path("outputs/agents") / f"{agent_name}.json"

    if not config_path.exists():
        raise HTTPException(status_code=404, detail=f"Agent '{agent_name}' not found")

    config = json.loads(config_path.read_text())
    return {"agent_config": config}
```

### Endpoint: Run Created Agent (Phase 2)

```python
from src.schemas import RunAgentRequest, AgentResponse

@app.post("/agents/{agent_name}/run", response_model=AgentResponse)
async def run_agent(agent_name: str, request: RunAgentRequest):
    """Execute a created agent with real MCP tools."""
    # Load agent config
    agent_config = await get_agent(agent_name)

    # Instantiate with MCP tools (Phase 2 implementation)
    agent = await create_agent_with_tools(agent_config["agent_config"])

    # Run the agent
    result = await agent.ainvoke({
        "messages": [HumanMessage(content=request.message)]
    })

    return AgentResponse(
        message=result["messages"][-1].content,
        tool_calls=extract_tool_calls(result),
    )
```

---

## Module Exports

```python
# Public API
__all__ = [
    "create_builder",
    "DEFAULT_MODEL",
]

# Re-exports for convenience
from src.prompts import DEFAULT_MODEL
```

---

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `ANTHROPIC_API_KEY` | Yes | API key for Anthropic models |
| `OPENAI_API_KEY` | No | API key for OpenAI models (if using) |

---

## Error Handling Strategy

1. **Missing API Keys**: Caught at agent invocation, raises `EnvironmentError` with clear message
2. **Invalid Model**: Validated during `create_deep_agent` call, raises `ValueError`
3. **Tool Errors**: Handled within individual tools, return error strings to the LLM
4. **Subagent Failures**: Caught by deepagents framework, surfaced in response messages
5. **Network Errors**: Caught during API calls, raised as appropriate HTTP exceptions in FastAPI context

---

## Testing Patterns

### Unit Test: Builder Creation

```python
import pytest
from src.builder import create_builder

def test_create_builder_returns_agent():
    """Builder returns a valid deepagents agent."""
    builder = create_builder()
    assert hasattr(builder, "invoke")
    assert hasattr(builder, "ainvoke")

def test_create_builder_without_subagents():
    """Builder can be created without subagents."""
    builder = create_builder(include_subagents=False)
    assert builder is not None

def test_create_builder_custom_model():
    """Builder accepts custom model parameter."""
    builder = create_builder(model="anthropic:claude-sonnet-4-20250514")
    assert builder is not None
```

### Integration Test: Agent Creation Flow

```python
import pytest
from src.builder import create_builder
from langchain_core.messages import HumanMessage

@pytest.mark.asyncio
async def test_builder_creates_agent():
    """Builder can create a complete agent configuration."""
    builder = create_builder()

    result = await builder.ainvoke({
        "messages": [HumanMessage(content="Create a simple research agent")]
    })

    # Verify response contains success message
    final_message = result["messages"][-1].content
    assert "SUCCESS" in final_message or "created" in final_message.lower()
```

### Fixture: Mock Builder

```python
import pytest
from unittest.mock import MagicMock, AsyncMock

@pytest.fixture
def mock_builder():
    """Create a mock builder for testing."""
    builder = MagicMock()
    builder.ainvoke = AsyncMock(return_value={
        "messages": [
            MagicMock(content="Agent created successfully")
        ]
    })
    return builder
```

---

## File Structure

```
src/
  __init__.py
  builder.py          # This module
  tools.py            # BUILDER_TOOLS
  prompts.py          # System prompts
  schemas.py          # Pydantic models
```

---

## Implementation Notes

### Recommended Implementation

```python
"""Main agent builder module for ACTi Agent Builder.

This module provides the create_builder() function that returns a fully
configured deepagents agent capable of designing and creating other AI agents.
"""

from __future__ import annotations

from deepagents import create_deep_agent

from .tools import BUILDER_TOOLS
from .prompts import (
    DEFAULT_MODEL,
    BUILDER_SYSTEM_PROMPT,
    PROMPT_ENGINEER_SYSTEM_PROMPT,
    PATTERN_ANALYZER_SYSTEM_PROMPT,
)


def _create_subagent_config(
    name: str,
    description: str,
    system_prompt: str,
    tools: list | None = None,
) -> dict:
    """Create configuration dictionary for a subagent."""
    return {
        "name": name,
        "description": description,
        "system_prompt": system_prompt,
        "tools": tools or [],
    }


def _get_default_subagents() -> list[dict]:
    """Return default subagent configurations for the builder."""
    return [
        _create_subagent_config(
            name="prompt-engineer",
            description="Crafts detailed, effective system prompts for agents",
            system_prompt=PROMPT_ENGINEER_SYSTEM_PROMPT,
        ),
        _create_subagent_config(
            name="pattern-analyzer",
            description="Analyzes Bland/N8N configs to extract reusable patterns",
            system_prompt=PATTERN_ANALYZER_SYSTEM_PROMPT,
        ),
        _create_subagent_config(
            name="general-purpose",
            description="Handles complex multi-step tasks that would clutter context",
            system_prompt="You are a helpful assistant that completes tasks efficiently.",
        ),
    ]


def create_builder(
    model: str = DEFAULT_MODEL,
    include_subagents: bool = True,
    custom_tools: list | None = None,
):
    """Create and return the ACTi Agent Builder meta-agent.

    Args:
        model: Model identifier for the builder agent.
            Default: "anthropic:claude-opus-4-5-20251101"
        include_subagents: Whether to attach subagents for specialized tasks.
            Default: True
        custom_tools: Additional tools to merge with BUILDER_TOOLS.
            Default: None

    Returns:
        A configured DeepAgent instance ready for invocation.

    Example:
        >>> builder = create_builder()
        >>> result = await builder.ainvoke({
        ...     "messages": [HumanMessage(content="Create a lead qualifier agent")]
        ... })
    """
    # Merge tools
    tools = list(BUILDER_TOOLS)
    if custom_tools:
        tools.extend(custom_tools)

    # Build subagents list
    subagents = _get_default_subagents() if include_subagents else []

    # Create and return the agent
    return create_deep_agent(
        model=model,
        tools=tools,
        system_prompt=BUILDER_SYSTEM_PROMPT,
        subagents=subagents,
    )


# Module exports
__all__ = [
    "create_builder",
    "DEFAULT_MODEL",
]
```

---

## Version History

- **1.0.0** (2025-01-09): Initial API contract
  - `create_builder`: Main entry point for creating the meta-agent
  - Subagent support: prompt-engineer, pattern-analyzer, general-purpose
  - FastAPI integration patterns
  - Testing patterns
