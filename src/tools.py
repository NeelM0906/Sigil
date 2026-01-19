"""LangChain tools for the ACTi Agent Builder.

This module provides the tools used by the deepagents-based meta-agent builder
to create and manage agent configurations. Tools are decorated with @tool from
langchain_core.tools, making their docstrings and type annotations available
to the LLM as tool descriptions.

Design Principles:
    - All tools return strings (tool results are displayed to the LLM)
    - Docstrings serve as tool descriptions - be thorough and include examples
    - Type annotations define the tool's input schema for the LLM
    - Error messages should be actionable and guide the LLM to fix issues

Tools Provided:
    create_agent_config: Creates and persists agent configurations to disk
    list_available_tools: Returns available MCP tool categories with guidance
    get_agent_config: Retrieves a saved agent configuration by name
    list_created_agents: Lists all previously created agent configurations
    execute_created_agent: Executes a task with a created agent using real MCP tools
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import aiofiles
import portalocker
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from pydantic import ValidationError

from .prompts import DEFAULT_MODEL
from .schemas import AgentConfig, AgentMetadata, MCP_TOOL_CATEGORIES, Stratum

# Configure module logger
logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Retry Loop Detection (Fix 8a)
# -----------------------------------------------------------------------------

# Module-level tracking of tool calls to detect retry loops
_tool_call_history: dict[str, int] = {}


def _record_tool_call(tool_name: str) -> None:
    """Record that a tool was called.

    Increments the call count for the specified tool.
    Used to detect retry loops where the same tool is called repeatedly.

    Args:
        tool_name: The name of the tool being called.
    """
    global _tool_call_history
    _tool_call_history[tool_name] = _tool_call_history.get(tool_name, 0) + 1
    logger.debug(f"Tool call recorded: {tool_name} (count: {_tool_call_history[tool_name]})")


def _detect_retry_loop(tool_name: str, max_consecutive: int = 3) -> bool:
    """Check if a tool has been called too many times in a row.

    Returns True if the same tool has been called more than max_consecutive times,
    indicating a likely retry loop that should be interrupted.

    Args:
        tool_name: The name of the tool to check.
        max_consecutive: Maximum allowed consecutive calls before detecting a loop.
            Default is 3, meaning the 4th consecutive call triggers loop detection.

    Returns:
        True if a retry loop is detected, False otherwise.
    """
    count = _tool_call_history.get(tool_name, 0)
    is_loop = count > max_consecutive
    if is_loop:
        logger.warning(
            f"Retry loop detected for {tool_name}: called {count} times "
            f"(max allowed: {max_consecutive})"
        )
    return is_loop


def reset_tool_history() -> None:
    """Reset the tool call history.

    Call this at the start of a new conversation or task to clear
    the retry loop detection state. This prevents false positives
    when the same tool is legitimately called across different tasks.
    """
    global _tool_call_history
    _tool_call_history = {}
    logger.debug("Tool call history reset")


def get_tool_call_count(tool_name: str) -> int:
    """Get the current call count for a specific tool.

    Args:
        tool_name: The name of the tool to check.

    Returns:
        The number of times the tool has been called since the last reset.
    """
    return _tool_call_history.get(tool_name, 0)


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

# Output directory for generated agent configurations
OUTPUT_DIR = Path(__file__).parent.parent / "outputs" / "agents"


# -----------------------------------------------------------------------------
# Utility Functions
# -----------------------------------------------------------------------------

def _ensure_output_dir() -> Path:
    """Ensure the output directory exists.

    Returns:
        Path: The output directory path.

    Raises:
        OSError: If the directory cannot be created.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR


def _format_validation_error(error: ValidationError) -> str:
    """Format a Pydantic validation error into a human-readable string.

    Args:
        error: The ValidationError from Pydantic model validation.

    Returns:
        Formatted multi-line string with each error on its own line.
    """
    messages = []
    for err in error.errors():
        loc = " -> ".join(str(x) for x in err["loc"])
        messages.append(f"  - {loc}: {err['msg']}")
    return "\n".join(messages)


def extract_text_from_content(content: str | list | Any) -> str:
    """Extract text from LangChain message content.

    Handles both string content and Anthropic-style content blocks. This is
    necessary because Anthropic's Messages API returns content as a list of
    content blocks (supporting tool_use, extended thinking, citations, etc.),
    while OpenAI returns simple strings.

    Args:
        content: Message content - can be string or list of content blocks.
            Content blocks are dicts with 'type' and type-specific fields.

    Returns:
        Extracted text as a single string. Multiple text blocks are joined
        with newlines.

    Examples:
        >>> extract_text_from_content("Hello world")
        'Hello world'
        >>> extract_text_from_content([{"type": "text", "text": "Hello"}])
        'Hello'
        >>> extract_text_from_content([
        ...     {"type": "tool_use", "id": "123", "name": "search"},
        ...     {"type": "text", "text": "Results here"}
        ... ])
        'Results here'
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    text_parts.append(block.get("text", ""))
            elif isinstance(block, str):
                text_parts.append(block)
        return "\n".join(text_parts)
    return str(content)


def validate_and_patch_messages(messages: list) -> list:
    """Validate and patch message structure before sending to Anthropic API.

    This function ensures that every tool_use block is followed by a tool_result.
    If dangling tool_use blocks are found (from partial execution), they are either
    removed or have synthetic ToolMessage results added.

    Anthropic's Messages API requires strict structure:
        AIMessage (with tool_use) → must be followed by → ToolMessage (with tool_result)

    Args:
        messages: List of LangChain message objects.

    Returns:
        Patched list of messages with valid structure.
    """
    if not messages:
        return messages

    patched = []
    i = 0
    while i < len(messages):
        msg = messages[i]

        # Check if this is an AIMessage with tool_use blocks
        if hasattr(msg, "type") and msg.type == "ai" and hasattr(msg, "tool_calls") and msg.tool_calls:
            patched.append(msg)

            # Look for corresponding ToolMessages in the next position
            tool_call_ids = {tc.get("id") if isinstance(tc, dict) else tc.id for tc in msg.tool_calls}
            next_is_tool_result = False

            if i + 1 < len(messages):
                next_msg = messages[i + 1]
                if hasattr(next_msg, "type") and next_msg.type == "tool":
                    next_is_tool_result = True

            # If no tool_result follows, add synthetic ones
            if not next_is_tool_result:
                for tool_call in msg.tool_calls:
                    tool_id = tool_call.get("id") if isinstance(tool_call, dict) else tool_call.id
                    # Add a synthetic tool_result to maintain message structure
                    synthetic_result = ToolMessage(
                        tool_call_id=tool_id,
                        content="[Tool execution was interrupted or incomplete]",
                        name=tool_call.get("name") if isinstance(tool_call, dict) else tool_call.name,
                    )
                    patched.append(synthetic_result)
        else:
            patched.append(msg)

        i += 1

    return patched


def _generate_filename(agent_name: str, include_timestamp: bool = False) -> str:
    """Generate a filename for an agent configuration.

    Args:
        agent_name: The snake_case name of the agent.
        include_timestamp: If True, append a timestamp to prevent overwrites.

    Returns:
        Filename string (without path).
    """
    if include_timestamp:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        return f"{agent_name}_{timestamp}.json"
    return f"{agent_name}.json"


def _serialize_config_with_metadata(config: AgentConfig, existing_metadata: Optional[AgentMetadata] = None) -> dict:
    """Serialize an AgentConfig with additional metadata.

    Args:
        config: The AgentConfig to serialize.
        existing_metadata: Optional existing metadata to preserve (for updates).

    Returns:
        Dictionary with config data and metadata fields.
    """
    now = datetime.now(timezone.utc).isoformat()

    # Create or update metadata
    if existing_metadata:
        # Update existing metadata
        metadata = AgentMetadata(
            created_at=existing_metadata.created_at,
            updated_at=now,
            last_executed=existing_metadata.last_executed,
            version=existing_metadata.version + 1,
            execution_count=existing_metadata.execution_count,
            tags=existing_metadata.tags,
        )
    elif config.metadata:
        # Use config's metadata but update timestamps
        metadata = AgentMetadata(
            created_at=config.metadata.created_at,
            updated_at=now,
            last_executed=config.metadata.last_executed,
            version=config.metadata.version,
            execution_count=config.metadata.execution_count,
            tags=config.metadata.tags,
        )
    else:
        # Create new metadata
        metadata = AgentMetadata(
            created_at=now,
            updated_at=now,
        )

    # Serialize config and include metadata
    data = config.model_dump(mode="json", exclude={"metadata"})
    data["metadata"] = metadata.model_dump(mode="json")

    # Also keep legacy _metadata for backward compatibility
    data["_metadata"] = {
        "created_at": metadata.created_at,
        "version": "1.0.0",
        "builder": "acti-agent-builder",
    }
    return data


def _load_agent_metadata(config_data: dict) -> Optional[AgentMetadata]:
    """Load metadata from a config dict, supporting both old and new formats.

    Args:
        config_data: The loaded config dictionary.

    Returns:
        AgentMetadata if available, None otherwise.
    """
    # Try new format first
    if "metadata" in config_data and config_data["metadata"]:
        try:
            return AgentMetadata(**config_data["metadata"])
        except (ValidationError, TypeError):
            pass

    # Fall back to legacy _metadata format
    if "_metadata" in config_data:
        legacy = config_data["_metadata"]
        return AgentMetadata(
            created_at=legacy.get("created_at", datetime.now(timezone.utc).isoformat()),
            updated_at=legacy.get("created_at", datetime.now(timezone.utc).isoformat()),
            version=1,
            execution_count=0,
            tags=[],
        )

    return None


def _update_execution_metadata(config_path: Path) -> None:
    """Update execution metadata after running an agent.

    Args:
        config_path: Path to the agent config file.
    """
    try:
        content = _read_config_safely(config_path)
        config_data = json.loads(content)

        now = datetime.now(timezone.utc).isoformat()

        # Update metadata
        if "metadata" in config_data and config_data["metadata"]:
            config_data["metadata"]["last_executed"] = now
            config_data["metadata"]["execution_count"] = config_data["metadata"].get("execution_count", 0) + 1
            config_data["metadata"]["updated_at"] = now
        else:
            # Create metadata if missing
            config_data["metadata"] = {
                "created_at": config_data.get("_metadata", {}).get("created_at", now),
                "updated_at": now,
                "last_executed": now,
                "version": 1,
                "execution_count": 1,
                "tags": [],
            }

        # Write back
        content = json.dumps(config_data, indent=2, ensure_ascii=False)
        _write_config_safely(config_path, content)

    except (json.JSONDecodeError, OSError, portalocker.LockException) as e:
        logger.warning(f"Failed to update execution metadata for {config_path}: {e}")


# -----------------------------------------------------------------------------
# File Locking for Concurrent Access (Fix #4)
# -----------------------------------------------------------------------------

def _write_config_safely(path: Path, content: str) -> None:
    """Write content to a file with exclusive locking for concurrent access safety.

    Args:
        path: The file path to write to.
        content: The content to write.

    Raises:
        OSError: If the file cannot be written.
        portalocker.LockException: If the file cannot be locked.
    """
    with open(path, 'w', encoding='utf-8') as f:
        portalocker.lock(f, portalocker.LOCK_EX)
        try:
            f.write(content)
        finally:
            portalocker.unlock(f)


def _read_config_safely(path: Path) -> str:
    """Read content from a file with shared locking for concurrent access safety.

    Args:
        path: The file path to read from.

    Returns:
        The file content as a string.

    Raises:
        OSError: If the file cannot be read.
        portalocker.LockException: If the file cannot be locked.
    """
    with open(path, 'r', encoding='utf-8') as f:
        portalocker.lock(f, portalocker.LOCK_SH)
        try:
            return f.read()
        finally:
            portalocker.unlock(f)


# -----------------------------------------------------------------------------
# Async Support for Tools (Fix #3)
# -----------------------------------------------------------------------------

# Thread pool executor for running sync tools asynchronously
_executor = ThreadPoolExecutor(max_workers=4)


async def _read_config_async(path: Path) -> str:
    """Read a configuration file asynchronously.

    Args:
        path: The file path to read from.

    Returns:
        The file content as a string.
    """
    async with aiofiles.open(path, 'r', encoding='utf-8') as f:
        return await f.read()


async def _write_config_async(path: Path, content: str) -> None:
    """Write content to a configuration file asynchronously.

    Args:
        path: The file path to write to.
        content: The content to write.
    """
    async with aiofiles.open(path, 'w', encoding='utf-8') as f:
        await f.write(content)


async def get_agent_config_async(name: str) -> str:
    """Async variant of get_agent_config for FastAPI integration.

    Args:
        name: The name of the agent to retrieve.

    Returns:
        The agent configuration or error message.
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(_executor, get_agent_config.func, name)


async def create_agent_config_async(
    name: str,
    description: str,
    system_prompt: str,
    tools: Optional[list[str]] = None,
    model: str = DEFAULT_MODEL,
    stratum: Optional[str] = None,
) -> str:
    """Async variant of create_agent_config for FastAPI integration.

    Args:
        name: Unique identifier for the agent.
        description: What the agent does.
        system_prompt: Complete system prompt.
        tools: List of MCP tool category names.
        model: Model identifier.
        stratum: ACTi stratum classification.

    Returns:
        Success confirmation or error message.
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        _executor,
        lambda: create_agent_config.func(name, description, system_prompt, tools, model, stratum)
    )


async def list_created_agents_async() -> str:
    """Async variant of list_created_agents for FastAPI integration.

    Returns:
        Formatted list of created agents.
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(_executor, list_created_agents.func)


@tool
def create_agent_config(
    name: str,
    description: str,
    system_prompt: str,
    tools: Optional[list[str]] = None,
    model: str = DEFAULT_MODEL,
    stratum: Optional[str] = None,
) -> str:
    """Create and save a new agent configuration to disk.

    Use this tool to finalize an agent design by creating a complete configuration
    that specifies the agent's name, purpose, behavior, and tool capabilities.
    The configuration is validated against the AgentConfig schema and saved as
    JSON for later instantiation with real MCP tools.

    IMPORTANT: Call list_available_tools() first to see valid tool options.

    Args:
        name: Unique identifier for the agent. MUST be snake_case format:
            - Only lowercase letters, numbers, and underscores
            - Must start with a letter
            - Maximum 64 characters
            Examples: "lead_qualifier", "appointment_scheduler", "research_assistant"

        description: Clear, concise statement of what the agent does.
            - Minimum 10 characters, maximum 500 characters
            - Should explain the agent's primary purpose and use case
            - Be specific about what problem the agent solves
            Example: "Qualifies inbound leads by assessing budget, authority, need, and timeline"

        system_prompt: Complete system prompt defining agent behavior.
            - Minimum 50 characters (thorough prompts lead to better agents)
            - Should include: role definition, personality traits, constraints,
              instructions for using tools, and example interactions
            - This is the core instruction set that shapes how the agent operates

        tools: List of MCP tool category names the agent should have access to.
            Valid options: "voice", "websearch", "calendar", "communication", "crm"
            - "voice": Text-to-speech, voice synthesis (ElevenLabs)
            - "websearch": Web research, fact-finding (Tavily)
            - "calendar": Scheduling, availability checks (Google Calendar)
            - "communication": SMS, calls (Twilio)
            - "crm": Contact management, deals (HubSpot)
            Defaults to empty list if not specified.

        model: Model identifier for the agent's LLM backbone.
            Default: "anthropic:claude-opus-4-5-20251101"
            Can be any supported model string.

        stratum: Optional ACTi stratum classification for the agent. Valid values:
            - "RTI": Reality & Truth Intelligence
                     Use for: data gathering, fact verification, research
                     Recommended tools: websearch, crm
            - "RAI": Readiness & Agreement Intelligence
                     Use for: lead qualification, rapport building, discovery
                     Recommended tools: communication, crm
            - "ZACS": Zone Action & Conversion Systems
                     Use for: scheduling, conversions, follow-ups, closings
                     Recommended tools: calendar, communication, voice
            - "EEI": Economic & Ecosystem Intelligence
                     Use for: analytics, market research, optimization
                     Recommended tools: websearch, crm
            - "IGE": Integrity & Governance Engine
                     Use for: compliance, quality control, auditing
                     Recommended tools: all (for comprehensive auditing)

    Returns:
        On success: Confirmation message with file path and agent summary.
        On failure: Error message describing what went wrong and how to fix it.

    Example:
        >>> create_agent_config(
        ...     name="appointment_scheduler",
        ...     description="Schedules appointments and manages calendar availability for sales teams",
        ...     system_prompt="You are a professional appointment scheduling assistant. Your role is to help sales teams manage their calendars effectively...",
        ...     tools=["calendar", "communication"],
        ...     stratum="ZACS"
        ... )
        "Successfully created agent configuration!

        Agent: appointment_scheduler
        Description: Schedules appointments and manages calendar availability for sales teams
        Tools: calendar, communication, stratum: ZACS
        Model: anthropic:claude-opus-4-5-20251101

        Saved to: /path/to/outputs/agents/appointment_scheduler.json"
    """
    # Default tools to empty list if None
    if tools is None:
        tools = []

    # Convert stratum string to enum if provided
    stratum_enum: Optional[Stratum] = None
    if stratum is not None:
        try:
            stratum_enum = Stratum(stratum.upper())
        except ValueError:
            valid_strata = ", ".join(s.value for s in Stratum)
            return (
                f"ERROR: Invalid stratum '{stratum}'.\n\n"
                f"Valid stratum options:\n"
                f"  - RTI: Reality & Truth Intelligence (data gathering)\n"
                f"  - RAI: Readiness & Agreement Intelligence (qualification)\n"
                f"  - ZACS: Zone Action & Conversion Systems (scheduling)\n"
                f"  - EEI: Economic & Ecosystem Intelligence (analytics)\n"
                f"  - IGE: Integrity & Governance Engine (compliance)\n\n"
                f"Please use one of: {valid_strata}"
            )

    # Validate and create the AgentConfig
    try:
        config = AgentConfig(
            name=name,
            description=description,
            system_prompt=system_prompt,
            tools=tools,
            model=model,
            stratum=stratum_enum,
        )
    except ValidationError as e:
        error_details = _format_validation_error(e)
        valid_tools = ", ".join(MCP_TOOL_CATEGORIES.keys())
        return (
            f"ERROR: Agent configuration validation failed.\n\n"
            f"Validation errors:\n{error_details}\n\n"
            f"Requirements:\n"
            f"  - name: snake_case (lowercase, numbers, underscores, starts with letter)\n"
            f"  - description: 10-500 characters\n"
            f"  - system_prompt: minimum 50 characters\n"
            f"  - tools: valid options are [{valid_tools}]\n\n"
            f"Please fix the issues and try again."
        )

    # Ensure output directory exists
    try:
        _ensure_output_dir()
    except OSError as e:
        return f"ERROR: Failed to create output directory: {e}"

    # Save configuration to JSON file with metadata
    # Validate path to prevent directory traversal attacks
    output_path = (OUTPUT_DIR / _generate_filename(config.name)).resolve()
    if not str(output_path).startswith(str(OUTPUT_DIR.resolve())):
        return "ERROR: Invalid agent name - path traversal detected"
    try:
        serialized = _serialize_config_with_metadata(config)
        content = json.dumps(serialized, indent=2, ensure_ascii=False)
        _write_config_safely(output_path, content)
        logger.info(f"Created agent configuration: {config.name} at {output_path}")
    except (OSError, TypeError, portalocker.LockException) as e:
        logger.error(f"Failed to save configuration file {output_path}: {e}")
        return f"ERROR: Failed to save configuration file: {e}"

    # Build success message with configuration summary
    tools_summary = ", ".join(config.tools) if config.tools else "none"
    stratum_info = config.stratum.value if config.stratum else "Not classified"

    return (
        f"SUCCESS: Agent configuration created and saved!\n\n"
        f"Agent Summary:\n"
        f"  Name: {config.name}\n"
        f"  Description: {config.description}\n"
        f"  Tools: {tools_summary}\n"
        f"  Stratum: {stratum_info}\n"
        f"  Model: {config.model}\n\n"
        f"Saved to: {output_path.absolute()}\n\n"
        f"Next steps:\n"
        f"  - Use get_agent_config('{config.name}') to review the full configuration\n"
        f"  - The agent can be instantiated with MCP tools using the FastAPI backend"
    )


@tool
def list_available_tools() -> str:
    """List all available MCP tool categories and their capabilities for agent configuration.

    Use this tool to discover what tools can be assigned to agents. Each tool
    category represents a set of real MCP (Model Context Protocol) integrations
    that give agents the ability to perform actual tasks like making calls,
    searching the web, or scheduling meetings.

    ALWAYS call this tool before creating an agent to understand available options.

    Returns:
        A comprehensive list including:
        - All available tool categories with descriptions
        - Specific capabilities provided by each tool
        - Recommended tool combinations for each ACTi stratum
        - Guidance on when to use each tool

    When to call this tool:
        - Before creating any new agent (to see valid tool options)
        - When a user asks what capabilities agents can have
        - To help decide which tools match an agent's purpose
        - When unsure which stratum an agent should target

    Example:
        >>> list_available_tools()
        "Available MCP Tool Categories
        ========================================

          voice
            Text-to-speech, voice synthesis (ElevenLabs)
            Capabilities:
              - Generate spoken audio from text
              - Clone and customize voices
            Best for: ZACS, RAI strata
        ..."
    """
    # Extended tool information with capabilities and recommendations
    tool_details = {
        "voice": {
            "capabilities": [
                "Generate spoken audio from text",
                "Clone and customize voices",
                "Real-time voice synthesis for calls",
            ],
            "best_for": ["ZACS", "RAI"],
            "use_cases": "Customer-facing agents, accessibility, audio content",
        },
        "websearch": {
            "capabilities": [
                "Search the web for current information",
                "Research topics and gather facts",
                "Find documentation and references",
            ],
            "best_for": ["RTI", "EEI"],
            "use_cases": "Research agents, fact-checkers, information gatherers",
        },
        "calendar": {
            "capabilities": [
                "Check availability and schedule meetings",
                "Create, update, and delete calendar events",
                "Send calendar invitations",
            ],
            "best_for": ["ZACS"],
            "use_cases": "Scheduling agents, appointment setters, coordinators",
        },
        "communication": {
            "capabilities": [
                "Send SMS messages",
                "Make and receive phone calls",
                "Manage communication workflows",
            ],
            "best_for": ["RAI", "ZACS"],
            "use_cases": "Outreach agents, follow-up agents, notification systems",
        },
        "crm": {
            "capabilities": [
                "Access and update contact records",
                "Manage deals and pipelines",
                "Track customer interactions",
            ],
            "best_for": ["RTI", "RAI", "EEI"],
            "use_cases": "Sales agents, customer service, data management",
        },
    }

    lines = [
        "Available MCP Tool Categories",
        "=" * 50,
        "",
    ]

    # List each tool with detailed information
    for tool_name, base_description in MCP_TOOL_CATEGORIES.items():
        details = tool_details.get(tool_name, {})
        capabilities = details.get("capabilities", [])
        best_for = details.get("best_for", [])
        use_cases = details.get("use_cases", "General purpose")

        lines.append(f"  {tool_name}")
        lines.append(f"    {base_description}")

        if capabilities:
            lines.append("    Capabilities:")
            for cap in capabilities:
                lines.append(f"      - {cap}")

        if best_for:
            lines.append(f"    Best for strata: {', '.join(best_for)}")

        lines.append(f"    Use cases: {use_cases}")
        lines.append("")

    # Add stratum reference guide
    lines.extend([
        "=" * 50,
        "ACTi Stratum Reference & Tool Recommendations",
        "=" * 50,
        "",
        "  RTI (Reality & Truth Intelligence)",
        "    Purpose: Data gathering, fact verification, research",
        "    Recommended tools: websearch, crm",
        "    Example agents: Research assistant, fact-checker, data gatherer",
        "",
        "  RAI (Readiness & Agreement Intelligence)",
        "    Purpose: Lead qualification, rapport building, discovery",
        "    Recommended tools: communication, crm",
        "    Example agents: Lead qualifier, discovery agent, SDR assistant",
        "",
        "  ZACS (Zone Action & Conversion Systems)",
        "    Purpose: Scheduling, conversions, follow-ups, closings",
        "    Recommended tools: calendar, communication, voice",
        "    Example agents: Appointment scheduler, closer, follow-up agent",
        "",
        "  EEI (Economic & Ecosystem Intelligence)",
        "    Purpose: Analytics, market research, optimization",
        "    Recommended tools: websearch, crm",
        "    Example agents: Market analyst, pipeline optimizer, trend tracker",
        "",
        "  IGE (Integrity & Governance Engine)",
        "    Purpose: Compliance, quality control, auditing",
        "    Recommended tools: all (for comprehensive auditing)",
        "    Example agents: Compliance checker, QA agent, audit assistant",
        "",
    ])

    return "\n".join(lines)


@tool
def get_agent_config(name: str) -> str:
    """Retrieve a previously saved agent configuration by name.

    Use this tool to load and review an existing agent configuration.
    This is useful for:
    - Reviewing the full system prompt of an agent
    - Checking what tools an agent has access to
    - Verifying an agent was created correctly
    - Getting the configuration to modify or recreate an agent

    Args:
        name: The name of the agent to retrieve (snake_case, without .json extension).
            Example: "lead_qualifier", "appointment_scheduler"

    Returns:
        On success: The full agent configuration formatted as JSON, including:
            - name, description, system_prompt
            - tools, model, stratum
            - metadata (created_at, version)
        On failure: Error message with list of available agents (if any exist).

    Example:
        >>> get_agent_config("lead_qualifier")
        "Agent Configuration: lead_qualifier
        ========================================

        {
          \"name\": \"lead_qualifier\",
          \"description\": \"Qualifies inbound leads...\",
          ...
        }"
    """
    # Validate path to prevent directory traversal attacks
    config_path = (OUTPUT_DIR / f"{name}.json").resolve()
    if not str(config_path).startswith(str(OUTPUT_DIR.resolve())):
        return "ERROR: Invalid agent name - path traversal detected"

    if not config_path.exists():
        # List available agents to help the user
        try:
            _ensure_output_dir()
            available = sorted(OUTPUT_DIR.glob("*.json"))
            if available:
                agent_names = [p.stem for p in available]
                return (
                    f"ERROR: Agent '{name}' not found.\n\n"
                    f"Available agents ({len(agent_names)}):\n"
                    + "\n".join(f"  - {n}" for n in agent_names)
                    + "\n\nUse one of the names above, or create a new agent with create_agent_config()."
                )
        except OSError:
            pass
        return (
            f"ERROR: Agent '{name}' not found.\n\n"
            f"No agents have been created yet. Use create_agent_config() to create one."
        )

    try:
        content = _read_config_safely(config_path)
        config_data = json.loads(content)

        # Format the output with a clear header
        formatted_json = json.dumps(config_data, indent=2, ensure_ascii=False)

        # Extract key info for summary
        tools = config_data.get("tools", [])
        stratum = config_data.get("stratum", "Not classified")
        tools_summary = ", ".join(tools) if tools else "none"

        logger.debug(f"Retrieved agent configuration: {name}")
        return (
            f"Agent Configuration: {name}\n"
            f"{'=' * 50}\n\n"
            f"Quick Summary:\n"
            f"  Tools: {tools_summary}\n"
            f"  Stratum: {stratum}\n"
            f"  Model: {config_data.get('model', 'default')}\n\n"
            f"Full Configuration:\n"
            f"{formatted_json}"
        )

    except json.JSONDecodeError as e:
        logger.warning(f"Corrupted JSON in agent config {config_path}: {e}")
        return f"ERROR: Failed to parse agent config file (corrupted JSON): {e}"

    except (OSError, portalocker.LockException) as e:
        logger.error(f"Failed to read agent config file {config_path}: {e}")
        return f"ERROR: Failed to read agent config file: {e}"


@tool
def list_created_agents() -> str:
    """List all agent configurations that have been created and saved.

    Use this tool to see what agents already exist in the system before
    creating new ones. This helps avoid duplicate agents and provides
    an overview of existing capabilities.

    Returns:
        A formatted summary of all saved agent configurations including:
        - Agent name and description
        - Assigned tools and stratum classification
        - Creation timestamp (if available)
        - Total count of agents

        If no agents exist, returns guidance on how to create one.

    When to use this tool:
        - At the start of a conversation to understand existing agents
        - Before creating a new agent to check for duplicates
        - When a user asks what agents are available
        - To get an overview of the agent ecosystem

    Example:
        >>> list_created_agents()
        "Created Agents
        ==================================================

          lead_qualifier
            Description: Qualifies inbound leads by assessing BANT criteria
            Tools: communication, crm
            Stratum: RAI
            Created: 2025-01-09T14:30:22Z

          appointment_scheduler
            Description: Schedules appointments and manages calendar availability
            Tools: calendar, communication
            Stratum: ZACS
            Created: 2025-01-09T15:45:10Z

        Total: 2 agent(s)"
    """
    try:
        _ensure_output_dir()
    except OSError as e:
        return f"ERROR: Failed to access output directory: {e}"

    config_files = sorted(OUTPUT_DIR.glob("*.json"))

    if not config_files:
        return (
            "No agents have been created yet.\n\n"
            "To create your first agent:\n"
            "  1. Call list_available_tools() to see available tool options\n"
            "  2. Call create_agent_config() with the agent's details\n\n"
            "Example:\n"
            "  create_agent_config(\n"
            "    name='my_first_agent',\n"
            "    description='A helpful agent that...',\n"
            "    system_prompt='You are...',\n"
            "    tools=['websearch'],\n"
            "    stratum='RTI'\n"
            "  )"
        )

    lines = [
        "Created Agents",
        "=" * 50,
        "",
    ]

    # Group agents by stratum for better organization
    agents_by_stratum: dict[str, list[dict]] = {}

    for config_path in config_files:
        try:
            content = config_path.read_text(encoding="utf-8")
            config_data = json.loads(content)

            name = config_data.get("name", config_path.stem)
            description = config_data.get("description", "No description")
            tools = config_data.get("tools", [])
            stratum = config_data.get("stratum", "Unclassified")
            metadata = config_data.get("_metadata", {})
            created_at = metadata.get("created_at", "Unknown")

            # Truncate description if too long
            if len(description) > 70:
                description = description[:67] + "..."

            agent_info = {
                "name": name,
                "description": description,
                "tools": tools,
                "stratum": stratum,
                "created_at": created_at,
            }

            stratum_key = stratum or "Unclassified"
            if stratum_key not in agents_by_stratum:
                agents_by_stratum[stratum_key] = []
            agents_by_stratum[stratum_key].append(agent_info)

        except (json.JSONDecodeError, OSError):
            lines.append(f"  {config_path.stem}")
            lines.append("    (Error reading configuration)")
            lines.append("")

    # Output agents grouped by stratum
    for stratum_name in ["RTI", "RAI", "ZACS", "EEI", "IGE", "Unclassified"]:
        if stratum_name in agents_by_stratum:
            agents = agents_by_stratum[stratum_name]
            lines.append(f"[{stratum_name}] ({len(agents)} agent(s))")
            lines.append("-" * 40)

            for agent in agents:
                lines.append(f"  {agent['name']}")
                lines.append(f"    Description: {agent['description']}")
                tools_str = ", ".join(agent["tools"]) if agent["tools"] else "none"
                lines.append(f"    Tools: {tools_str}")
                if agent["created_at"] != "Unknown":
                    # Format the timestamp more readably
                    lines.append(f"    Created: {agent['created_at']}")
                lines.append("")

    lines.append("=" * 50)
    lines.append(f"Total: {len(config_files)} agent(s)")
    lines.append("")
    lines.append("Use get_agent_config('<name>') to see full configuration details.")

    return "\n".join(lines)


# -----------------------------------------------------------------------------
# Agent Execution Tool (Phase 2.5)
# -----------------------------------------------------------------------------


def _extract_tool_calls_from_messages(messages: list[Any]) -> list[dict[str, Any]]:
    """Extract tool call information from agent response messages.

    Args:
        messages: List of LangChain message objects from agent response.

    Returns:
        List of dictionaries containing tool call details.
    """
    tool_calls = []
    for msg in messages:
        # Check for tool calls in AI messages
        if hasattr(msg, 'tool_calls') and msg.tool_calls:
            for tc in msg.tool_calls:
                tool_calls.append({
                    "tool_name": tc.get("name", "unknown"),
                    "arguments": tc.get("args", {}),
                    "id": tc.get("id", ""),
                })
        # Check for tool results in ToolMessages
        if isinstance(msg, ToolMessage):
            # Find the matching tool call and add result
            for tc in tool_calls:
                if hasattr(msg, 'tool_call_id') and tc.get("id") == msg.tool_call_id:
                    content_str = extract_text_from_content(msg.content)
                    tc["result"] = content_str[:500] if len(content_str) > 500 else content_str
                    break
    return tool_calls


async def _execute_agent_async(
    agent_config: AgentConfig,
    task: str,
    timeout: float,
    max_turns: int = 10,
) -> dict[str, Any]:
    """Execute an agent with a task asynchronously.

    Args:
        agent_config: The agent configuration to instantiate.
        task: The task/message to send to the agent.
        timeout: Maximum execution time in seconds.
        max_turns: Maximum number of tool call rounds (default 10).
            This limits how many times the agent can call tools before
            being forced to stop. Helps prevent infinite loops.

    Returns:
        Dictionary with execution results including response, tools used, and timing.
    """
    from .mcp_integration import (
        create_agent_with_tools,
        MCPIntegrationError,
    )

    start_time = time.time()
    result: dict[str, Any] = {
        "success": False,
        "agent_name": agent_config.name,
        "task": task,
        "response": "",
        "tools_used": [],
        "execution_time_seconds": 0.0,
        "errors": [],
        "warnings": [],
    }

    try:
        # Create the agent with MCP tools (skip unavailable tools gracefully)
        agent = await asyncio.wait_for(
            create_agent_with_tools(
                agent_config,
                skip_unavailable=True,
                timeout=timeout / 2,
                max_turns=max_turns,
            ),
            timeout=timeout / 2,
        )

        # Invoke the agent with the task
        response = await asyncio.wait_for(
            agent.ainvoke({
                "messages": [HumanMessage(content=task)]
            }),
            timeout=timeout / 2,
        )

        # Extract the final response message
        messages = response.get("messages", [])
        if messages:
            final_message = messages[-1]
            if hasattr(final_message, 'content'):
                result["response"] = extract_text_from_content(final_message.content)
            else:
                result["response"] = str(final_message)

            # Extract tool calls from the conversation
            result["tools_used"] = _extract_tool_calls_from_messages(messages)

        result["success"] = True

    except asyncio.TimeoutError:
        result["errors"].append(f"Execution timed out after {timeout} seconds")
    except MCPIntegrationError as e:
        result["errors"].append(f"MCP integration error: {str(e)}")
    except Exception as e:
        logger.error(f"Agent execution failed: {e}", exc_info=True)
        result["errors"].append(f"Execution error: {str(e)}")

    result["execution_time_seconds"] = round(time.time() - start_time, 2)
    return result


@tool
def execute_created_agent(agent_name: str, task: str, timeout: int = 60, max_turns: int = 10) -> str:
    """Execute a task with a created agent using real MCP tools.

    Use this tool to TEST agents immediately after creating them. This allows
    you to verify that an agent works correctly by giving it a real task to
    execute with its configured MCP tools.

    The agent will be instantiated with all available MCP tools from its
    configuration and then invoked with the provided task message.

    IMPORTANT: This executes the agent with REAL tool integrations. Any actions
    the agent takes (sending messages, making calls, scheduling events) will
    have real-world effects.

    Args:
        agent_name: Name of the agent to execute (must exist in outputs/agents/).
            Use list_created_agents() to see available agents.
            Example: "lead_qualifier", "appointment_scheduler"

        task: The task or message to send to the agent.
            This should be a realistic prompt that exercises the agent's capabilities.
            Example: "Qualify this lead: John Smith, CEO of Acme Corp, interested in
            our enterprise plan, budget around $50k"

        timeout: Maximum time in seconds for execution (default 60).
            Increase for complex tasks or slow tool integrations.
            Range: 10-300 seconds recommended.

        max_turns: Maximum number of tool call rounds (default 10).
            This limits how many times the agent can call tools before
            being forced to stop. Helps prevent infinite loops where
            agents keep calling tools without providing a final answer.
            Each "turn" may involve multiple internal steps; the underlying
            recursion_limit is set to max_turns * 5.

    Returns:
        A structured result containing:
        - Agent's response to the task
        - List of tools used during execution (with arguments and results)
        - Execution time in seconds
        - Any errors or warnings encountered

        If the agent is not found or execution fails, returns an error message
        with guidance on how to resolve the issue.

    Example:
        >>> # First create an agent
        >>> create_agent_config(
        ...     name="web_researcher",
        ...     description="Researches topics using web search",
        ...     system_prompt="You are a research assistant...",
        ...     tools=["websearch"],
        ...     stratum="RTI"
        ... )
        "SUCCESS: Agent configuration created..."

        >>> # Then test it immediately
        >>> execute_created_agent(
        ...     agent_name="web_researcher",
        ...     task="Find the current CEO of Microsoft and their background"
        ... )
        "Execution Results: web_researcher
        ========================================

        Task: Find the current CEO of Microsoft and their background

        Status: SUCCESS

        Response:
        Based on my research, the current CEO of Microsoft is Satya Nadella...

        Tools Used:
          1. tavily_search
             Arguments: {\"query\": \"Microsoft CEO current\"}
             Result: Found information about Satya Nadella...

        Execution Time: 3.45 seconds"

    Notes:
        - Agents with missing MCP credentials will execute with available tools only
        - Warnings about unavailable tools will be included in the output
        - For long-running tasks, increase the timeout parameter
        - Use this to validate agent behavior before deploying to production
    """
    # Record this tool call for retry loop detection (Fix 8a)
    _record_tool_call("execute_created_agent")

    # Check for retry loop before proceeding (Fix 8a)
    if _detect_retry_loop("execute_created_agent", max_consecutive=3):
        call_count = get_tool_call_count("execute_created_agent")
        logger.error(
            f"Retry loop detected: execute_created_agent called {call_count} times consecutively"
        )
        return json.dumps({
            "status": "loop_detected",
            "response": "",
            "agent_name": agent_name,
            "execution_time_ms": 0,
            "reason": (
                f"Retry loop detected: execute_created_agent has been called {call_count} times "
                "consecutively. This indicates the builder is stuck in a retry loop. "
                "STOP calling this tool and report the issue to the user. "
                "Ask them to simplify their request or try a different approach."
            )
        }, indent=2)

    # Validate path to prevent directory traversal attacks
    config_path = (OUTPUT_DIR / f"{agent_name}.json").resolve()
    if not str(config_path).startswith(str(OUTPUT_DIR.resolve())):
        return "ERROR: Invalid agent name - path traversal detected"

    # Check if agent exists
    if not config_path.exists():
        # List available agents to help the user
        try:
            _ensure_output_dir()
            available = sorted(OUTPUT_DIR.glob("*.json"))
            if available:
                agent_names = [p.stem for p in available]
                return (
                    f"ERROR: Agent '{agent_name}' not found.\n\n"
                    f"Available agents ({len(agent_names)}):\n"
                    + "\n".join(f"  - {n}" for n in agent_names)
                    + "\n\nUse one of the names above, or create a new agent with create_agent_config()."
                )
        except OSError:
            pass
        return (
            f"ERROR: Agent '{agent_name}' not found.\n\n"
            f"No agents have been created yet. Use create_agent_config() to create one first."
        )

    # Load the agent configuration
    try:
        content = _read_config_safely(config_path)
        config_data = json.loads(content)

        # Remove metadata before parsing as AgentConfig
        config_data.pop("_metadata", None)

        agent_config = AgentConfig(**config_data)

    except json.JSONDecodeError as e:
        logger.warning(f"Corrupted JSON in agent config {config_path}: {e}")
        return f"ERROR: Failed to parse agent config file (corrupted JSON): {e}"

    except ValidationError as e:
        error_details = _format_validation_error(e)
        return (
            f"ERROR: Agent configuration validation failed.\n\n"
            f"Validation errors:\n{error_details}\n\n"
            f"The saved agent config may be corrupted or incompatible."
        )

    except (OSError, portalocker.LockException) as e:
        logger.error(f"Failed to read agent config file {config_path}: {e}")
        return f"ERROR: Failed to read agent config file: {e}"

    # Validate timeout
    if timeout < 5:
        return "ERROR: Timeout must be at least 5 seconds."
    if timeout > 600:
        return "ERROR: Timeout cannot exceed 600 seconds (10 minutes)."

    # Execute the agent
    logger.info(f"Executing agent '{agent_name}' with task: {task[:100]}...")

    # Validate max_turns
    if max_turns < 1:
        return "ERROR: max_turns must be at least 1."
    if max_turns > 50:
        return "ERROR: max_turns cannot exceed 50."

    try:
        result = asyncio.run(_execute_agent_async(agent_config, task, float(timeout), max_turns))
    except RuntimeError as e:
        # Handle case where event loop is already running
        if "cannot be called from a running event loop" in str(e):
            import nest_asyncio
            nest_asyncio.apply()
            result = asyncio.run(_execute_agent_async(agent_config, task, float(timeout), max_turns))
        else:
            raise

    # Update execution metadata on success
    if result["success"]:
        _update_execution_metadata(config_path)

    # Build structured JSON response (Fix 8b)
    # This makes success/error status unambiguous so the builder knows when NOT to retry
    execution_time_ms = int(result["execution_time_seconds"] * 1000)

    # Determine status and reason
    if result["success"]:
        status = "success"
        reason = "Agent completed task successfully"
    elif result["errors"]:
        # Check for timeout specifically
        has_timeout = any("timed out" in err.lower() for err in result["errors"])
        status = "timeout" if has_timeout else "error"
        reason = "; ".join(result["errors"])
    else:
        status = "error"
        reason = "Unknown error occurred"

    # Build the response text with human-readable summary
    response_parts = []
    if result["response"]:
        response_parts.append(result["response"])

    if result["tools_used"]:
        tools_summary = [f"[Tools used: {', '.join(t.get('tool_name', 'unknown') for t in result['tools_used'])}]"]
        response_parts.append("\n".join(tools_summary))

    if result["warnings"]:
        response_parts.append(f"Warnings: {'; '.join(result['warnings'])}")

    response_text = "\n\n".join(response_parts) if response_parts else ""

    # Create structured output
    structured_response = {
        "status": status,
        "response": response_text,
        "agent_name": agent_name,
        "execution_time_ms": execution_time_ms,
        "reason": reason,
    }

    # Log the structured response for debugging
    logger.info(
        f"execute_created_agent completed: status={status}, agent={agent_name}, "
        f"time={execution_time_ms}ms"
    )

    return json.dumps(structured_response, indent=2)


# -----------------------------------------------------------------------------
# Agent CRUD Tools (Phase 2.5.4)
# -----------------------------------------------------------------------------


@tool
def update_agent_config(
    agent_name: str,
    description: Optional[str] = None,
    system_prompt: Optional[str] = None,
    tools: Optional[list[str]] = None,
    stratum: Optional[str] = None,
    tags: Optional[list[str]] = None,
) -> str:
    """Update an existing agent's configuration.

    Use this tool to modify an agent that has already been created. Only the
    fields you provide will be updated; other fields remain unchanged. The
    version number is automatically incremented on each update.

    Args:
        agent_name: Name of the agent to update (must exist in outputs/agents/).
            Use list_created_agents() to see available agents.
            Example: "lead_qualifier", "appointment_scheduler"

        description: New description for the agent (10-500 characters).
            If not provided, the existing description is preserved.

        system_prompt: New system prompt for the agent (minimum 50 characters).
            If not provided, the existing system prompt is preserved.

        tools: New list of MCP tool category names.
            Valid options: "voice", "websearch", "calendar", "communication", "crm"
            If not provided, the existing tools are preserved.

        stratum: New ACTi stratum classification.
            Valid values: "RTI", "RAI", "ZACS", "EEI", "IGE"
            If not provided, the existing stratum is preserved.

        tags: New list of tags for categorization.
            Tags are automatically lowercased and deduplicated.
            If not provided, existing tags are preserved.

    Returns:
        On success: Confirmation message with updated fields and new version.
        On failure: Error message describing what went wrong.

    Example:
        >>> update_agent_config(
        ...     agent_name="lead_qualifier",
        ...     description="Updated: Qualifies leads with enhanced BANT criteria",
        ...     tags=["sales", "qualification"]
        ... )
        "SUCCESS: Agent 'lead_qualifier' updated (v1 -> v2)
        ..."
    """
    # Validate path to prevent directory traversal attacks
    config_path = (OUTPUT_DIR / f"{agent_name}.json").resolve()
    if not str(config_path).startswith(str(OUTPUT_DIR.resolve())):
        return "ERROR: Invalid agent name - path traversal detected"

    if not config_path.exists():
        # List available agents to help the user
        try:
            _ensure_output_dir()
            available = sorted(OUTPUT_DIR.glob("*.json"))
            if available:
                agent_names = [p.stem for p in available]
                return (
                    f"ERROR: Agent '{agent_name}' not found.\n\n"
                    f"Available agents ({len(agent_names)}):\n"
                    + "\n".join(f"  - {n}" for n in agent_names)
                    + "\n\nUse one of the names above, or create a new agent with create_agent_config()."
                )
        except OSError:
            pass
        return (
            f"ERROR: Agent '{agent_name}' not found.\n\n"
            f"No agents have been created yet. Use create_agent_config() to create one."
        )

    # Load existing configuration
    try:
        content = _read_config_safely(config_path)
        config_data = json.loads(content)
    except json.JSONDecodeError as e:
        return f"ERROR: Failed to parse agent config file (corrupted JSON): {e}"
    except (OSError, portalocker.LockException) as e:
        return f"ERROR: Failed to read agent config file: {e}"

    # Load existing metadata
    existing_metadata = _load_agent_metadata(config_data)
    old_version = existing_metadata.version if existing_metadata else 1

    # Track what fields are being updated
    updates = []

    # Update description if provided
    if description is not None:
        if len(description) < 10 or len(description) > 500:
            return "ERROR: description must be 10-500 characters"
        config_data["description"] = description
        updates.append("description")

    # Update system_prompt if provided
    if system_prompt is not None:
        if len(system_prompt) < 50:
            return "ERROR: system_prompt must be at least 50 characters"
        config_data["system_prompt"] = system_prompt
        updates.append("system_prompt")

    # Update tools if provided
    if tools is not None:
        invalid_tools = [t for t in tools if t not in MCP_TOOL_CATEGORIES]
        if invalid_tools:
            valid_tools = ", ".join(MCP_TOOL_CATEGORIES.keys())
            return (
                f"ERROR: Invalid tool(s): {invalid_tools}.\n"
                f"Valid options: {valid_tools}"
            )
        config_data["tools"] = tools
        updates.append("tools")

    # Update stratum if provided
    if stratum is not None:
        try:
            stratum_enum = Stratum(stratum.upper())
            config_data["stratum"] = stratum_enum.value
            updates.append("stratum")
        except ValueError:
            valid_strata = ", ".join(s.value for s in Stratum)
            return (
                f"ERROR: Invalid stratum '{stratum}'.\n"
                f"Valid options: {valid_strata}"
            )

    # Update tags if provided
    if tags is not None:
        # Validate and clean tags
        clean_tags = []
        for tag in tags:
            tag = tag.strip().lower()
            if tag and len(tag) <= 50:
                clean_tags.append(tag)
        clean_tags = list(set(clean_tags))  # Remove duplicates

        # Update in metadata
        if "metadata" not in config_data or config_data["metadata"] is None:
            config_data["metadata"] = {}
        config_data["metadata"]["tags"] = clean_tags
        updates.append("tags")

    if not updates:
        return (
            f"No changes specified for agent '{agent_name}'.\n\n"
            f"Provide at least one of: description, system_prompt, tools, stratum, tags"
        )

    # Update metadata version and timestamp
    now = datetime.now(timezone.utc).isoformat()
    if "metadata" not in config_data or config_data["metadata"] is None:
        config_data["metadata"] = {
            "created_at": config_data.get("_metadata", {}).get("created_at", now),
            "updated_at": now,
            "last_executed": None,
            "version": old_version + 1,
            "execution_count": 0,
            "tags": [],
        }
    else:
        config_data["metadata"]["updated_at"] = now
        config_data["metadata"]["version"] = old_version + 1

    # Write back
    try:
        content = json.dumps(config_data, indent=2, ensure_ascii=False)
        _write_config_safely(config_path, content)
        logger.info(f"Updated agent configuration: {agent_name} (v{old_version} -> v{old_version + 1})")
    except (OSError, portalocker.LockException) as e:
        return f"ERROR: Failed to save updated configuration: {e}"

    new_version = old_version + 1
    updates_str = ", ".join(updates)

    return (
        f"SUCCESS: Agent '{agent_name}' updated (v{old_version} -> v{new_version})\n\n"
        f"Updated fields: {updates_str}\n\n"
        f"Use get_agent_config('{agent_name}') to review the full configuration."
    )


@tool
def delete_agent_config(agent_name: str, confirm: bool = False) -> str:
    """Delete an agent configuration permanently.

    Use this tool to remove an agent that is no longer needed. This action
    is IRREVERSIBLE - the agent configuration will be permanently deleted.

    IMPORTANT: You must set confirm=True to actually delete the agent.
    This is a safety measure to prevent accidental deletions.

    Args:
        agent_name: Name of the agent to delete (must exist in outputs/agents/).
            Use list_created_agents() to see available agents.

        confirm: Safety confirmation flag. Must be True to proceed with deletion.
            Example: delete_agent_config("old_agent", confirm=True)

    Returns:
        On success (confirm=True): Confirmation that agent was deleted.
        On confirm=False: Warning message asking for confirmation.
        On failure: Error message describing what went wrong.

    Example:
        >>> delete_agent_config("test_agent")
        "WARNING: This will permanently delete agent 'test_agent'.
        To confirm, call: delete_agent_config('test_agent', confirm=True)"

        >>> delete_agent_config("test_agent", confirm=True)
        "SUCCESS: Agent 'test_agent' has been permanently deleted."
    """
    # Validate path to prevent directory traversal attacks
    config_path = (OUTPUT_DIR / f"{agent_name}.json").resolve()
    if not str(config_path).startswith(str(OUTPUT_DIR.resolve())):
        return "ERROR: Invalid agent name - path traversal detected"

    if not config_path.exists():
        # List available agents to help the user
        try:
            _ensure_output_dir()
            available = sorted(OUTPUT_DIR.glob("*.json"))
            if available:
                agent_names = [p.stem for p in available]
                return (
                    f"ERROR: Agent '{agent_name}' not found.\n\n"
                    f"Available agents ({len(agent_names)}):\n"
                    + "\n".join(f"  - {n}" for n in agent_names)
                )
        except OSError:
            pass
        return f"ERROR: Agent '{agent_name}' not found."

    # Require confirmation
    if not confirm:
        # Load metadata to show what's being deleted
        try:
            content = _read_config_safely(config_path)
            config_data = json.loads(content)
            description = config_data.get("description", "No description")
            metadata = _load_agent_metadata(config_data)
            exec_count = metadata.execution_count if metadata else 0
            version = metadata.version if metadata else 1

            return (
                f"WARNING: This will permanently delete agent '{agent_name}'.\n\n"
                f"Agent details:\n"
                f"  Description: {description[:100]}...\n"
                f"  Version: {version}\n"
                f"  Execution count: {exec_count}\n\n"
                f"To confirm deletion, call:\n"
                f"  delete_agent_config('{agent_name}', confirm=True)"
            )
        except (json.JSONDecodeError, OSError):
            return (
                f"WARNING: This will permanently delete agent '{agent_name}'.\n\n"
                f"To confirm deletion, call:\n"
                f"  delete_agent_config('{agent_name}', confirm=True)"
            )

    # Perform deletion
    try:
        config_path.unlink()
        logger.info(f"Deleted agent configuration: {agent_name}")
        return (
            f"SUCCESS: Agent '{agent_name}' has been permanently deleted.\n\n"
            f"Use list_created_agents() to see remaining agents."
        )
    except OSError as e:
        logger.error(f"Failed to delete agent config {config_path}: {e}")
        return f"ERROR: Failed to delete agent: {e}"


@tool
def clone_agent_config(source_name: str, new_name: str) -> str:
    """Create a copy of an existing agent with a new name.

    Use this tool to create a new agent based on an existing one. This is useful
    for:
    - Creating variations of a successful agent
    - Testing modifications without affecting the original
    - Creating specialized versions for different use cases

    The cloned agent will have:
    - All configuration from the source agent
    - A new name (as specified)
    - Fresh metadata (version=1, execution_count=0, new timestamps)

    Args:
        source_name: Name of the agent to clone (must exist).
            Use list_created_agents() to see available agents.

        new_name: Name for the new cloned agent.
            Must be snake_case (lowercase, numbers, underscores, starts with letter).
            Must not already exist.

    Returns:
        On success: Confirmation message with the new agent's file path.
        On failure: Error message describing what went wrong.

    Example:
        >>> clone_agent_config("lead_qualifier", "lead_qualifier_v2")
        "SUCCESS: Cloned 'lead_qualifier' to 'lead_qualifier_v2'

        The new agent has:
          - Fresh metadata (version 1, execution count 0)
          - Same configuration as the original

        Saved to: /path/to/outputs/agents/lead_qualifier_v2.json

        Use update_agent_config('lead_qualifier_v2', ...) to customize."
    """
    import re

    # Validate source path
    source_path = (OUTPUT_DIR / f"{source_name}.json").resolve()
    if not str(source_path).startswith(str(OUTPUT_DIR.resolve())):
        return "ERROR: Invalid source agent name - path traversal detected"

    # Validate new name format
    if not re.match(r"^[a-z][a-z0-9_]*$", new_name):
        return (
            f"ERROR: Invalid name '{new_name}'.\n\n"
            f"Name must be snake_case: lowercase letters, numbers, "
            f"and underscores only, starting with a letter.\n"
            f"Examples: 'lead_qualifier_v2', 'new_appointment_scheduler'"
        )

    if len(new_name) > 64:
        return "ERROR: Name must be 64 characters or less."

    # Validate new path
    new_path = (OUTPUT_DIR / f"{new_name}.json").resolve()
    if not str(new_path).startswith(str(OUTPUT_DIR.resolve())):
        return "ERROR: Invalid new agent name - path traversal detected"

    # Check source exists
    if not source_path.exists():
        try:
            _ensure_output_dir()
            available = sorted(OUTPUT_DIR.glob("*.json"))
            if available:
                agent_names = [p.stem for p in available]
                return (
                    f"ERROR: Source agent '{source_name}' not found.\n\n"
                    f"Available agents ({len(agent_names)}):\n"
                    + "\n".join(f"  - {n}" for n in agent_names)
                )
        except OSError:
            pass
        return f"ERROR: Source agent '{source_name}' not found."

    # Check new name doesn't exist
    if new_path.exists():
        return (
            f"ERROR: Agent '{new_name}' already exists.\n\n"
            f"Choose a different name or delete the existing agent first:\n"
            f"  delete_agent_config('{new_name}', confirm=True)"
        )

    # Load source configuration
    try:
        content = _read_config_safely(source_path)
        config_data = json.loads(content)
    except json.JSONDecodeError as e:
        return f"ERROR: Failed to parse source agent config (corrupted JSON): {e}"
    except (OSError, portalocker.LockException) as e:
        return f"ERROR: Failed to read source agent config: {e}"

    # Update for new agent
    config_data["name"] = new_name

    # Create fresh metadata
    now = datetime.now(timezone.utc).isoformat()
    config_data["metadata"] = {
        "created_at": now,
        "updated_at": now,
        "last_executed": None,
        "version": 1,
        "execution_count": 0,
        "tags": config_data.get("metadata", {}).get("tags", []) if isinstance(config_data.get("metadata"), dict) else [],
    }

    # Update legacy _metadata
    config_data["_metadata"] = {
        "created_at": now,
        "version": "1.0.0",
        "builder": "acti-agent-builder",
    }

    # Save new agent
    try:
        _ensure_output_dir()
        content = json.dumps(config_data, indent=2, ensure_ascii=False)
        _write_config_safely(new_path, content)
        logger.info(f"Cloned agent '{source_name}' to '{new_name}'")
    except (OSError, portalocker.LockException) as e:
        return f"ERROR: Failed to save cloned agent: {e}"

    return (
        f"SUCCESS: Cloned '{source_name}' to '{new_name}'\n\n"
        f"The new agent has:\n"
        f"  - Fresh metadata (version 1, execution count 0)\n"
        f"  - Same configuration as the original\n\n"
        f"Saved to: {new_path.absolute()}\n\n"
        f"Use update_agent_config('{new_name}', ...) to customize."
    )


async def execute_created_agent_async(agent_name: str, task: str, timeout: int = 60, max_turns: int = 10) -> str:
    """Async variant of execute_created_agent for FastAPI integration.

    Args:
        agent_name: Name of the agent to execute.
        task: The task/message to send to the agent.
        timeout: Maximum execution time in seconds.
        max_turns: Maximum number of tool call rounds (default 10).

    Returns:
        Structured result string with execution details.
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        _executor,
        lambda: execute_created_agent.func(agent_name, task, timeout, max_turns)
    )


# Export all tools for easy importing
BUILDER_TOOLS = [
    create_agent_config,
    list_available_tools,
    get_agent_config,
    list_created_agents,
    execute_created_agent,
    update_agent_config,
    delete_agent_config,
    clone_agent_config,
]

# Async variants for FastAPI integration
ASYNC_TOOLS = {
    "get_agent_config": get_agent_config_async,
    "create_agent_config": create_agent_config_async,
    "list_created_agents": list_created_agents_async,
    "execute_created_agent": execute_created_agent_async,
}

__all__ = [
    "BUILDER_TOOLS",
    "ASYNC_TOOLS",
    "create_agent_config",
    "list_available_tools",
    "get_agent_config",
    "list_created_agents",
    "execute_created_agent",
    "update_agent_config",
    "delete_agent_config",
    "clone_agent_config",
    "get_agent_config_async",
    "create_agent_config_async",
    "list_created_agents_async",
    "execute_created_agent_async",
    # Retry loop detection (Fix 8a)
    "reset_tool_history",
    "get_tool_call_count",
]
