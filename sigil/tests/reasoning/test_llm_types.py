"""Tests for LLM response types.

Tests cover:
- ToolUseBlock creation and properties
- LLMResponse creation and properties
- has_tool_calls property behavior
- Edge cases and boundary conditions
"""

import pytest
from sigil.reasoning.llm_types import ToolUseBlock, LLMResponse


class TestToolUseBlock:
    """Test suite for ToolUseBlock dataclass."""

    def test_tool_use_block_creation(self):
        """Test basic ToolUseBlock creation."""
        block = ToolUseBlock(
            id="tool_123",
            name="websearch_search",
            input={"query": "latest news"}
        )

        assert block.id == "tool_123"
        assert block.name == "websearch_search"
        assert block.input == {"query": "latest news"}

    def test_tool_use_block_empty_input(self):
        """Test ToolUseBlock with empty input dict."""
        block = ToolUseBlock(
            id="tool_456",
            name="time_now",
            input={}
        )

        assert block.id == "tool_456"
        assert block.name == "time_now"
        assert block.input == {}

    def test_tool_use_block_complex_input(self):
        """Test ToolUseBlock with complex input."""
        block = ToolUseBlock(
            id="tool_789",
            name="email_send",
            input={
                "to": "user@example.com",
                "subject": "Test Email",
                "body": "Hello, World!",
                "attachments": ["file1.pdf", "file2.txt"],
                "priority": 1
            }
        )

        assert block.input["to"] == "user@example.com"
        assert len(block.input["attachments"]) == 2
        assert block.input["priority"] == 1

    def test_tool_use_block_nested_input(self):
        """Test ToolUseBlock with nested input structures."""
        block = ToolUseBlock(
            id="tool_nested",
            name="complex_tool",
            input={
                "config": {
                    "nested": {
                        "deeply": "value"
                    }
                },
                "options": [{"key": "value1"}, {"key": "value2"}]
            }
        )

        assert block.input["config"]["nested"]["deeply"] == "value"
        assert len(block.input["options"]) == 2

    def test_tool_use_block_equality(self):
        """Test ToolUseBlock equality comparison."""
        block1 = ToolUseBlock(id="1", name="test", input={"a": 1})
        block2 = ToolUseBlock(id="1", name="test", input={"a": 1})
        block3 = ToolUseBlock(id="2", name="test", input={"a": 1})

        assert block1 == block2
        assert block1 != block3

    def test_tool_use_block_input_modification(self):
        """Test that input dict can be modified after creation."""
        block = ToolUseBlock(id="1", name="test", input={"original": True})
        block.input["new_key"] = "new_value"

        assert "new_key" in block.input
        assert block.input["new_key"] == "new_value"


class TestLLMResponse:
    """Test suite for LLMResponse dataclass."""

    def test_llm_response_no_tools(self):
        """Test LLMResponse without tool calls."""
        response = LLMResponse(
            text="Hello world",
            stop_reason="end_turn",
            tokens_used=10
        )

        assert response.text == "Hello world"
        assert response.stop_reason == "end_turn"
        assert response.tokens_used == 10
        assert response.tool_uses == []
        assert response.has_tool_calls is False

    def test_llm_response_with_single_tool(self):
        """Test LLMResponse with a single tool call."""
        tool = ToolUseBlock(id="1", name="search", input={"query": "test"})
        response = LLMResponse(
            text="Let me search for that.",
            tool_uses=[tool],
            stop_reason="tool_use",
            tokens_used=20
        )

        assert response.has_tool_calls is True
        assert len(response.tool_uses) == 1
        assert response.tool_uses[0].name == "search"
        assert response.stop_reason == "tool_use"

    def test_llm_response_with_multiple_tools(self):
        """Test LLMResponse with multiple tool calls."""
        tool1 = ToolUseBlock(id="1", name="search", input={"query": "AI news"})
        tool2 = ToolUseBlock(id="2", name="memory_recall", input={"topic": "AI"})
        response = LLMResponse(
            text="I'll search and check memory.",
            tool_uses=[tool1, tool2],
            stop_reason="tool_use",
            tokens_used=30
        )

        assert response.has_tool_calls is True
        assert len(response.tool_uses) == 2
        assert response.tool_uses[0].name == "search"
        assert response.tool_uses[1].name == "memory_recall"

    def test_llm_response_default_values(self):
        """Test LLMResponse default values."""
        response = LLMResponse(text="Test")

        assert response.text == "Test"
        assert response.tool_uses == []
        assert response.stop_reason == "end_turn"
        assert response.tokens_used == 0

    def test_llm_response_empty_text(self):
        """Test LLMResponse with empty text."""
        tool = ToolUseBlock(id="1", name="tool", input={})
        response = LLMResponse(
            text="",
            tool_uses=[tool],
            stop_reason="tool_use",
            tokens_used=5
        )

        assert response.text == ""
        assert response.has_tool_calls is True


class TestHasToolCallsProperty:
    """Test suite for the has_tool_calls property logic."""

    def test_has_tool_calls_requires_both_conditions(self):
        """Test that has_tool_calls requires both tool_uses and stop_reason='tool_use'."""
        # Has tools but stop_reason is end_turn - should be False
        tool = ToolUseBlock(id="1", name="test", input={})
        response = LLMResponse(
            text="",
            tool_uses=[tool],
            stop_reason="end_turn",  # Not "tool_use"
            tokens_used=10
        )
        assert response.has_tool_calls is False

    def test_has_tool_calls_empty_list_tool_use_stop(self):
        """Test has_tool_calls with empty tool list but tool_use stop reason."""
        response = LLMResponse(
            text="",
            tool_uses=[],
            stop_reason="tool_use",
            tokens_used=10
        )
        assert response.has_tool_calls is False

    def test_has_tool_calls_true_when_both_present(self):
        """Test has_tool_calls is True when both conditions met."""
        tool = ToolUseBlock(id="1", name="test", input={})
        response = LLMResponse(
            text="",
            tool_uses=[tool],
            stop_reason="tool_use",
            tokens_used=10
        )
        assert response.has_tool_calls is True

    def test_has_tool_calls_with_max_tokens_stop(self):
        """Test has_tool_calls with max_tokens stop reason."""
        tool = ToolUseBlock(id="1", name="test", input={})
        response = LLMResponse(
            text="Partial response...",
            tool_uses=[tool],
            stop_reason="max_tokens",
            tokens_used=1000
        )
        assert response.has_tool_calls is False


class TestLLMResponseEquality:
    """Test LLMResponse equality and comparison."""

    def test_llm_response_equality(self):
        """Test LLMResponse equality comparison."""
        response1 = LLMResponse(text="Test", stop_reason="end_turn", tokens_used=10)
        response2 = LLMResponse(text="Test", stop_reason="end_turn", tokens_used=10)
        response3 = LLMResponse(text="Different", stop_reason="end_turn", tokens_used=10)

        assert response1 == response2
        assert response1 != response3

    def test_llm_response_equality_with_tools(self):
        """Test LLMResponse equality with tool uses."""
        tool = ToolUseBlock(id="1", name="test", input={})
        response1 = LLMResponse(text="", tool_uses=[tool], stop_reason="tool_use")
        response2 = LLMResponse(text="", tool_uses=[tool], stop_reason="tool_use")

        assert response1 == response2


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_tool_use_block_special_characters_in_name(self):
        """Test ToolUseBlock with special characters in name."""
        block = ToolUseBlock(
            id="tool-123",
            name="websearch_search",  # Underscores are valid
            input={"query": "test query"}
        )
        assert block.name == "websearch_search"

    def test_tool_use_block_unicode_in_input(self):
        """Test ToolUseBlock with unicode in input."""
        block = ToolUseBlock(
            id="tool_unicode",
            name="search",
            input={"query": "recherche en francais et japonais: 日本語"}
        )
        assert "日本語" in block.input["query"]

    def test_llm_response_large_tokens(self):
        """Test LLMResponse with large token count."""
        response = LLMResponse(
            text="A very long response...",
            tokens_used=100000
        )
        assert response.tokens_used == 100000

    def test_llm_response_zero_tokens(self):
        """Test LLMResponse with zero tokens."""
        response = LLMResponse(
            text="",
            tokens_used=0
        )
        assert response.tokens_used == 0
        assert response.has_tool_calls is False

    def test_tool_use_block_none_values_in_input(self):
        """Test ToolUseBlock with None values in input."""
        block = ToolUseBlock(
            id="tool_none",
            name="test",
            input={"param1": None, "param2": "value"}
        )
        assert block.input["param1"] is None
        assert block.input["param2"] == "value"

    def test_llm_response_multiline_text(self):
        """Test LLMResponse with multiline text."""
        text = """This is line 1.
This is line 2.
This is line 3."""
        response = LLMResponse(text=text, tokens_used=20)

        assert "\n" in response.text
        assert response.text.count("\n") == 2
