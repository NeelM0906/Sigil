"""Convert Sigil tool definitions to Claude API format."""
from typing import Any


def convert_to_claude_tool_schema(tool_name: str, tool_def: dict) -> dict:
    """Convert Sigil tool definition to Claude API format.

    Args:
        tool_name: Tool name like "websearch.search"
        tool_def: Sigil format tool definition with description and arguments

    Returns:
        Claude API tool schema
    """
    properties = {}
    required = []

    for arg_name, arg_def in tool_def.get("arguments", {}).items():
        properties[arg_name] = {
            "type": arg_def.get("type", "string"),
            "description": arg_def.get("description", ""),
        }
        if arg_def.get("required", False):
            required.append(arg_name)

    return {
        "name": tool_name.replace(".", "_"),  # Claude requires alphanumeric
        "description": tool_def.get("description", ""),
        "input_schema": {
            "type": "object",
            "properties": properties,
            "required": required,
        }
    }


def get_all_tool_schemas(available_tools: list[str], tool_registry: dict) -> list[dict]:
    """Get Claude-formatted schemas for all available tools.

    Args:
        available_tools: List of tool names to include
        tool_registry: Registry mapping tool names to definitions

    Returns:
        List of Claude API tool schemas
    """
    schemas = []
    for tool_name in available_tools:
        if tool_name in tool_registry:
            schemas.append(convert_to_claude_tool_schema(tool_name, tool_registry[tool_name]))
    return schemas


def claude_name_to_sigil_name(claude_name: str) -> str:
    """Convert Claude tool name back to Sigil format.

    Args:
        claude_name: Name like "websearch_search"

    Returns:
        Sigil name like "websearch.search"
    """
    return claude_name.replace("_", ".", 1)  # Only replace first underscore
