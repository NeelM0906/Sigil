"""Type definitions for LLM responses with function calling."""
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolUseBlock:
    """Represents a tool use request from the LLM."""
    id: str
    name: str
    input: dict[str, Any]


@dataclass
class LLMResponse:
    """Structured response from LLM, may include tool calls."""
    text: str
    tool_uses: list[ToolUseBlock] = field(default_factory=list)
    stop_reason: str = "end_turn"  # "end_turn" | "tool_use"
    tokens_used: int = 0

    @property
    def has_tool_calls(self) -> bool:
        """Check if response includes tool calls."""
        return len(self.tool_uses) > 0 and self.stop_reason == "tool_use"
