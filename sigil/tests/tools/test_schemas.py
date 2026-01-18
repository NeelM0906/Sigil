"""Tests for tool schema conversion.

Tests cover:
- Converting Sigil tool definitions to Claude API format
- Handling required and optional arguments
- Name conversion between Sigil and Claude formats
- Batch schema conversion
- Edge cases and error handling
"""

import pytest
from sigil.tools.schemas import (
    convert_to_claude_tool_schema,
    get_all_tool_schemas,
    claude_name_to_sigil_name,
)


class TestToolSchemaConversion:
    """Test suite for tool schema conversion."""

    def test_convert_simple_tool(self):
        """Test converting a simple tool definition."""
        sigil_tool = {
            "description": "Search the web",
            "arguments": {
                "query": {"type": "string", "required": True, "description": "Search query"}
            }
        }
        result = convert_to_claude_tool_schema("websearch.search", sigil_tool)

        assert result["name"] == "websearch_search"
        assert result["description"] == "Search the web"
        assert result["input_schema"]["type"] == "object"
        assert result["input_schema"]["required"] == ["query"]
        assert "query" in result["input_schema"]["properties"]
        assert result["input_schema"]["properties"]["query"]["type"] == "string"
        assert result["input_schema"]["properties"]["query"]["description"] == "Search query"

    def test_convert_tool_with_optional_args(self):
        """Test tool with optional arguments."""
        sigil_tool = {
            "description": "Store memory",
            "arguments": {
                "content": {"type": "string", "required": True, "description": "Content to store"},
                "category": {"type": "string", "required": False, "description": "Memory category"}
            }
        }
        result = convert_to_claude_tool_schema("memory.store", sigil_tool)

        assert result["name"] == "memory_store"
        assert result["input_schema"]["required"] == ["content"]
        assert "category" in result["input_schema"]["properties"]
        assert "content" in result["input_schema"]["properties"]
        # category should NOT be in required since required=False
        assert "category" not in result["input_schema"]["required"]

    def test_convert_tool_with_multiple_required_args(self):
        """Test tool with multiple required arguments."""
        sigil_tool = {
            "description": "Send email",
            "arguments": {
                "to": {"type": "string", "required": True, "description": "Recipient"},
                "subject": {"type": "string", "required": True, "description": "Subject line"},
                "body": {"type": "string", "required": True, "description": "Email body"},
                "cc": {"type": "string", "required": False, "description": "CC recipients"}
            }
        }
        result = convert_to_claude_tool_schema("email.send", sigil_tool)

        assert result["name"] == "email_send"
        assert set(result["input_schema"]["required"]) == {"to", "subject", "body"}
        assert len(result["input_schema"]["properties"]) == 4

    def test_convert_tool_with_no_arguments(self):
        """Test tool with no arguments."""
        sigil_tool = {
            "description": "Get current time",
            "arguments": {}
        }
        result = convert_to_claude_tool_schema("util.time", sigil_tool)

        assert result["name"] == "util_time"
        assert result["description"] == "Get current time"
        assert result["input_schema"]["properties"] == {}
        assert result["input_schema"]["required"] == []

    def test_convert_tool_missing_arguments_key(self):
        """Test tool definition missing arguments key."""
        sigil_tool = {
            "description": "Simple tool"
        }
        result = convert_to_claude_tool_schema("simple.tool", sigil_tool)

        assert result["name"] == "simple_tool"
        assert result["input_schema"]["properties"] == {}
        assert result["input_schema"]["required"] == []

    def test_convert_tool_with_various_types(self):
        """Test tool with various argument types."""
        sigil_tool = {
            "description": "Complex tool",
            "arguments": {
                "text": {"type": "string", "required": True, "description": "Text input"},
                "count": {"type": "integer", "required": True, "description": "Count"},
                "enabled": {"type": "boolean", "required": False, "description": "Flag"},
                "value": {"type": "number", "required": False, "description": "Numeric value"}
            }
        }
        result = convert_to_claude_tool_schema("complex.tool", sigil_tool)

        props = result["input_schema"]["properties"]
        assert props["text"]["type"] == "string"
        assert props["count"]["type"] == "integer"
        assert props["enabled"]["type"] == "boolean"
        assert props["value"]["type"] == "number"

    def test_convert_tool_defaults_type_to_string(self):
        """Test that missing type defaults to string."""
        sigil_tool = {
            "description": "Tool with no type",
            "arguments": {
                "param": {"required": True, "description": "A parameter"}
            }
        }
        result = convert_to_claude_tool_schema("test.tool", sigil_tool)

        assert result["input_schema"]["properties"]["param"]["type"] == "string"

    def test_convert_tool_defaults_description_to_empty(self):
        """Test that missing description defaults to empty string."""
        sigil_tool = {
            "description": "Tool",
            "arguments": {
                "param": {"type": "string", "required": True}
            }
        }
        result = convert_to_claude_tool_schema("test.tool", sigil_tool)

        assert result["input_schema"]["properties"]["param"]["description"] == ""

    def test_convert_tool_empty_description(self):
        """Test tool with missing description defaults to empty string."""
        sigil_tool = {
            "arguments": {
                "query": {"type": "string", "required": True}
            }
        }
        result = convert_to_claude_tool_schema("test.tool", sigil_tool)

        assert result["description"] == ""


class TestClaudeNameToSigilName:
    """Test suite for Claude to Sigil name conversion."""

    def test_claude_name_to_sigil_name_simple(self):
        """Test simple name conversion back to Sigil format."""
        assert claude_name_to_sigil_name("websearch_search") == "websearch.search"
        assert claude_name_to_sigil_name("memory_store") == "memory.store"
        assert claude_name_to_sigil_name("email_send") == "email.send"

    def test_claude_name_to_sigil_name_with_multiple_underscores(self):
        """Test name conversion with multiple underscores (only first is replaced)."""
        # The function uses replace("_", ".", 1) - only first underscore
        assert claude_name_to_sigil_name("tool_name_method") == "tool.name_method"
        assert claude_name_to_sigil_name("my_complex_tool_action") == "my.complex_tool_action"

    def test_claude_name_to_sigil_name_no_underscore(self):
        """Test name with no underscore stays unchanged."""
        assert claude_name_to_sigil_name("simpletool") == "simpletool"

    def test_claude_name_to_sigil_name_empty_string(self):
        """Test empty string handling."""
        assert claude_name_to_sigil_name("") == ""

    def test_claude_name_to_sigil_name_single_underscore(self):
        """Test single underscore at various positions."""
        assert claude_name_to_sigil_name("_start") == ".start"
        assert claude_name_to_sigil_name("end_") == "end."


class TestGetAllToolSchemas:
    """Test suite for batch schema conversion."""

    def test_get_all_tool_schemas_basic(self):
        """Test batch schema conversion."""
        registry = {
            "websearch.search": {
                "description": "Search the web",
                "arguments": {
                    "query": {"type": "string", "required": True}
                }
            },
            "memory.recall": {
                "description": "Recall memories",
                "arguments": {
                    "topic": {"type": "string", "required": True}
                }
            }
        }
        schemas = get_all_tool_schemas(["websearch.search", "memory.recall"], registry)

        assert len(schemas) == 2
        assert schemas[0]["name"] == "websearch_search"
        assert schemas[1]["name"] == "memory_recall"

    def test_get_all_tool_schemas_filters_unavailable(self):
        """Test that unavailable tools are filtered out."""
        registry = {
            "websearch.search": {"description": "Search", "arguments": {}},
        }
        schemas = get_all_tool_schemas(
            ["websearch.search", "memory.recall", "nonexistent.tool"],
            registry
        )

        assert len(schemas) == 1
        assert schemas[0]["name"] == "websearch_search"

    def test_get_all_tool_schemas_empty_list(self):
        """Test with empty tool list."""
        registry = {
            "websearch.search": {"description": "Search", "arguments": {}}
        }
        schemas = get_all_tool_schemas([], registry)

        assert len(schemas) == 0

    def test_get_all_tool_schemas_empty_registry(self):
        """Test with empty registry."""
        schemas = get_all_tool_schemas(["websearch.search"], {})

        assert len(schemas) == 0

    def test_get_all_tool_schemas_preserves_order(self):
        """Test that order of tools is preserved."""
        registry = {
            "a.tool": {"description": "Tool A", "arguments": {}},
            "b.tool": {"description": "Tool B", "arguments": {}},
            "c.tool": {"description": "Tool C", "arguments": {}},
        }
        schemas = get_all_tool_schemas(["c.tool", "a.tool", "b.tool"], registry)

        assert len(schemas) == 3
        assert schemas[0]["name"] == "c_tool"
        assert schemas[1]["name"] == "a_tool"
        assert schemas[2]["name"] == "b_tool"

    def test_get_all_tool_schemas_with_complex_tools(self):
        """Test batch conversion with complex tool definitions."""
        registry = {
            "search.web": {
                "description": "Web search",
                "arguments": {
                    "query": {"type": "string", "required": True, "description": "Search query"},
                    "max_results": {"type": "integer", "required": False, "description": "Max results"},
                    "include_images": {"type": "boolean", "required": False, "description": "Include images"}
                }
            },
            "email.compose": {
                "description": "Compose email",
                "arguments": {
                    "to": {"type": "string", "required": True},
                    "subject": {"type": "string", "required": True},
                    "body": {"type": "string", "required": True},
                    "attachments": {"type": "array", "required": False}
                }
            }
        }

        schemas = get_all_tool_schemas(["search.web", "email.compose"], registry)

        assert len(schemas) == 2

        # Check search.web schema
        search_schema = schemas[0]
        assert search_schema["name"] == "search_web"
        assert "query" in search_schema["input_schema"]["required"]
        assert "max_results" not in search_schema["input_schema"]["required"]

        # Check email.compose schema
        email_schema = schemas[1]
        assert email_schema["name"] == "email_compose"
        assert set(email_schema["input_schema"]["required"]) == {"to", "subject", "body"}


class TestSchemaRoundTrip:
    """Test schema conversion round-trip integrity."""

    def test_name_roundtrip_simple(self):
        """Test that simple names can round-trip correctly."""
        original_name = "websearch.search"
        sigil_tool = {"description": "Test", "arguments": {}}

        claude_schema = convert_to_claude_tool_schema(original_name, sigil_tool)
        recovered_name = claude_name_to_sigil_name(claude_schema["name"])

        assert recovered_name == original_name

    def test_name_roundtrip_various_tools(self):
        """Test round-trip for various tool names."""
        tool_names = [
            "memory.store",
            "planning.create",
            "tavily.search",
            "email.send",
        ]

        for name in tool_names:
            sigil_tool = {"description": "Test", "arguments": {}}
            claude_schema = convert_to_claude_tool_schema(name, sigil_tool)
            recovered = claude_name_to_sigil_name(claude_schema["name"])
            assert recovered == name, f"Round-trip failed for {name}"


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_tool_name_with_dots_in_namespace(self):
        """Test tool names with dots only in expected places."""
        # Standard namespace.method format
        sigil_tool = {"description": "Test", "arguments": {}}
        result = convert_to_claude_tool_schema("namespace.method", sigil_tool)
        assert result["name"] == "namespace_method"

    def test_argument_without_required_field(self):
        """Test argument that doesn't specify required field (defaults to False)."""
        sigil_tool = {
            "description": "Test",
            "arguments": {
                "optional_param": {"type": "string", "description": "Optional"}
            }
        }
        result = convert_to_claude_tool_schema("test.tool", sigil_tool)

        # Should not be in required list since required was not specified (defaults to False)
        assert "optional_param" not in result["input_schema"]["required"]

    def test_preserves_input_schema_structure(self):
        """Test that input_schema has correct structure."""
        sigil_tool = {
            "description": "Test tool",
            "arguments": {
                "param1": {"type": "string", "required": True}
            }
        }
        result = convert_to_claude_tool_schema("test.tool", sigil_tool)

        assert "type" in result["input_schema"]
        assert result["input_schema"]["type"] == "object"
        assert "properties" in result["input_schema"]
        assert "required" in result["input_schema"]
