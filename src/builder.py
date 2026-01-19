"""ACTi Agent Builder - Main builder module.

This module provides the main entry point for creating the ACTi Agent Builder,
a meta-agent that designs and creates executable AI agents with real MCP tool
capabilities using the deepagents framework.

The builder agent can:
    - Create agent configurations using BUILDER_TOOLS
    - Delegate to prompt-engineer subagent for crafting system prompts
    - Delegate to pattern-analyzer subagent for examining reference configs
    - Read files from the filesystem (Bland/N8N patterns)

Usage:
    from src.builder import create_builder

    # Create the builder agent
    builder = create_builder()

    # Invoke with a message
    result = builder.invoke({
        "messages": [{"role": "user", "content": "Create an appointment scheduling agent"}]
    })

Example CLI usage:
    python -m src.builder

Exports:
    create_builder: Main function that returns a configured deepagents agent
"""

from __future__ import annotations

import logging
import os
import signal
import sys
from pathlib import Path

from dotenv import load_dotenv

from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend
from langgraph.graph.state import CompiledStateGraph

from .prompts import (
    BUILDER_SYSTEM_PROMPT,
    DEFAULT_MODEL,
    PATTERN_ANALYZER_SYSTEM_PROMPT,
    PROMPT_ENGINEER_SYSTEM_PROMPT,
)
from .tools import BUILDER_TOOLS, extract_text_from_content

# Configure module logger
logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

# Load environment variables from .env file
_env_path = Path(__file__).parent.parent / ".env"
load_dotenv(_env_path)

# Project root directory (acti-agent-builder/)
PROJECT_ROOT = Path(__file__).parent.parent.resolve()

# Reference patterns directories (relative to Bland-Agents-Dataset/)
DATASET_ROOT = PROJECT_ROOT.parent
BLAND_DATASET_DIR = DATASET_ROOT / "bland_dataset"
N8N_WORKFLOWS_DIR = DATASET_ROOT / "n8n_workflows"


# -----------------------------------------------------------------------------
# Builder Factory
# -----------------------------------------------------------------------------

def create_builder(
    model: str = DEFAULT_MODEL,
    root_dir: str | Path | None = None,
    virtual_mode: bool = True,  # Sandbox filesystem paths under root_dir for production safety
    max_turns: int = 15,  # Maximum tool call rounds before forced stop
) -> CompiledStateGraph:
    """Create and configure the ACTi Agent Builder.

    Creates a deepagents-based meta-agent configured with:
        - BUILDER_TOOLS for creating agent configurations
        - FilesystemBackend for real disk access to reference patterns
        - Subagents for prompt engineering and pattern analysis

    Args:
        model: Model identifier string for the builder agent.
            Default: "anthropic:claude-opus-4-5-20251101"
        root_dir: Root directory for filesystem access. If None, defaults to
            the Bland-Agents-Dataset directory (parent of acti-agent-builder).
            This gives the agent access to bland_dataset/ and n8n_workflows/.
        virtual_mode: If True (default), sandbox all filesystem paths under root_dir
            for production safety. If False, allow access to absolute paths.
            Set to False only for development/testing with trusted inputs.
        max_turns: Maximum number of tool call rounds before forcing termination.
            This prevents infinite agent loops. Each "turn" may involve multiple
            internal steps; the underlying recursion_limit is set to max_turns * 5.
            Valid range: 1-50. Default: 15.

    Returns:
        A configured deepagents agent instance ready for invocation.

    Raises:
        ValueError: If max_turns is outside the valid range (1-50).

    Example:
        >>> builder = create_builder()
        >>> result = builder.invoke({
        ...     "messages": [{"role": "user", "content": "Create a lead qualification agent"}]
        ... })
        >>> print(result["messages"][-1].content)
    """
    # Validate max_turns parameter
    if not 1 <= max_turns <= 50:
        raise ValueError(
            f"max_turns must be between 1 and 50, got {max_turns}. "
            "Use lower values (5-10) for interactive CLI, higher (15-30) for complex tasks."
        )

    # Calculate recursion_limit from max_turns
    # Each "turn" may involve multiple internal graph steps (LLM call, tool execution, etc.)
    recursion_limit = max_turns * 5
    logger.info(
        f"Creating builder agent with max_turns={max_turns} (recursion_limit={recursion_limit})"
    )

    # Determine root directory for filesystem access
    if root_dir is None:
        # Default to Bland-Agents-Dataset directory for access to reference patterns
        root_dir = DATASET_ROOT
    root_dir = Path(root_dir).resolve()

    # Configure FilesystemBackend for real disk access
    # This allows the agent to read reference patterns from:
    #   - bland_dataset/ (48 JSON files with conversation flow patterns)
    #   - n8n_workflows/ (110+ JSON files with task orchestration patterns)
    backend = FilesystemBackend(
        root_dir=str(root_dir),
        virtual_mode=virtual_mode,
    )

    # Define subagents for specialized tasks
    subagents = [
        {
            "name": "prompt-engineer",
            "description": (
                "Crafts detailed, effective system prompts for agents. "
                "Use when creating the system_prompt for a new agent. "
                "Returns ONLY the prompt text, ready to use directly."
            ),
            "system_prompt": PROMPT_ENGINEER_SYSTEM_PROMPT,
            "tools": [],  # Uses built-in filesystem tools only
            "model": model,
        },
        {
            "name": "pattern-analyzer",
            "description": (
                "Analyzes Bland AI and N8N workflow configurations to extract "
                "reusable design patterns. Use when designing complex agents that "
                "benefit from reference analysis. Returns pattern insights and "
                "stratum classification recommendations."
            ),
            "system_prompt": PATTERN_ANALYZER_SYSTEM_PROMPT,
            "tools": [],  # Uses built-in read_file, glob, grep
            "model": model,
        },
    ]

    # Create the builder agent with all components wired up
    builder = create_deep_agent(
        model=model,
        tools=BUILDER_TOOLS,
        system_prompt=BUILDER_SYSTEM_PROMPT,
        backend=backend,
        subagents=subagents,
    )

    # Apply hard recursion limit to prevent infinite loops
    # This is a structural fix - prompts alone cannot prevent runaway behavior
    builder = builder.with_config({"recursion_limit": recursion_limit})

    logger.debug(
        f"Builder agent created successfully with recursion_limit={recursion_limit}"
    )

    return builder


# -----------------------------------------------------------------------------
# CLI Interface
# -----------------------------------------------------------------------------

def _run_cli() -> None:
    """Run interactive CLI for testing the builder agent.

    Provides a simple REPL interface for interacting with the builder.
    Type 'quit' or 'exit' to end the session.
    """
    from langchain_core.messages import HumanMessage

    # Set up signal handlers for graceful shutdown
    def signal_handler(sig, frame):
        print("\nShutting down gracefully...")
        logger.info("CLI shutdown via signal %s", sig)
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    print("=" * 60)
    print("ACTi Agent Builder - Interactive CLI")
    print("=" * 60)
    print()
    print("This meta-agent creates executable AI agents with real MCP tools.")
    print("Type 'quit' or 'exit' to end the session.")
    print()

    # Verify API key is available
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY environment variable not set.")
        print("Please set it in your .env file or environment.")
        return

    print("Creating builder agent...")
    builder = create_builder()
    print("Builder ready.")
    print()

    # Track conversation history for multi-turn interactions
    messages = []

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit"):
            print("Goodbye!")
            break

        # Add user message to history
        messages.append(HumanMessage(content=user_input))

        print("\nBuilder: ", end="", flush=True)

        try:
            result = builder.invoke({"messages": messages})

            # Extract AI response messages
            ai_messages = [
                msg for msg in result.get("messages", [])
                if hasattr(msg, "type") and msg.type == "ai"
            ]

            if ai_messages:
                response = extract_text_from_content(ai_messages[-1].content)
                print(response)
                # Update messages with full conversation
                messages = result.get("messages", messages)
            else:
                print("[No response received]")

        except Exception as e:
            print(f"[Error: {e}]")

        print()


if __name__ == "__main__":
    _run_cli()
