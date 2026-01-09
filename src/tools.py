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
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from langchain_core.tools import tool
from pydantic import ValidationError

from .schemas import AgentConfig, MCP_TOOL_CATEGORIES, Stratum


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


def _serialize_config_with_metadata(config: AgentConfig) -> dict:
    """Serialize an AgentConfig with additional metadata.

    Args:
        config: The AgentConfig to serialize.

    Returns:
        Dictionary with config data and metadata fields.
    """
    data = config.model_dump(mode="json")
    data["_metadata"] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "version": "1.0.0",
        "builder": "acti-agent-builder",
    }
    return data


@tool
def create_agent_config(
    name: str,
    description: str,
    system_prompt: str,
    tools: Optional[list[str]] = None,
    model: str = "anthropic:claude-opus-4-5-20251101",
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
    output_path = OUTPUT_DIR / _generate_filename(config.name)
    try:
        serialized = _serialize_config_with_metadata(config)
        output_path.write_text(
            json.dumps(serialized, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    except (OSError, TypeError) as e:
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
    config_path = OUTPUT_DIR / f"{name}.json"

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
        content = config_path.read_text(encoding="utf-8")
        config_data = json.loads(content)

        # Format the output with a clear header
        formatted_json = json.dumps(config_data, indent=2, ensure_ascii=False)

        # Extract key info for summary
        tools = config_data.get("tools", [])
        stratum = config_data.get("stratum", "Not classified")
        tools_summary = ", ".join(tools) if tools else "none"

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
        return f"ERROR: Failed to parse agent config file (corrupted JSON): {e}"

    except OSError as e:
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


# Export all tools for easy importing
BUILDER_TOOLS = [
    create_agent_config,
    list_available_tools,
    get_agent_config,
    list_created_agents,
]
