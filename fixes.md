# ACTi Agent Builder - Fixes Tracker

Issues identified during Phase 1 senior code review. Address before production deployment.

---

## Critical

### 1. Path Traversal Risk in `get_agent_config`
**File:** `src/tools.py` (lines 429-510)
**Status:** [ ] Open

**Problem:**
```python
def get_agent_config(name: str) -> str:
    config_path = OUTPUT_DIR / f"{name}.json"
```
User input is used directly in path construction without validation. Malicious input like `../../../etc/passwd` could read files outside the intended directory.

**Fix:**
```python
def get_agent_config(name: str) -> str:
    config_path = (OUTPUT_DIR / f"{name}.json").resolve()
    if not str(config_path).startswith(str(OUTPUT_DIR.resolve())):
        return "ERROR: Invalid agent name - path traversal detected"
    # ... rest of function
```

**Also apply to:** `create_agent_config` output path construction (line 250)

---

## High

### 2. Unsafe `virtual_mode=False` Default
**File:** `src/builder.py` (lines 71-74)
**Status:** [ ] Open

**Problem:**
```python
def create_builder(
    model: str = DEFAULT_MODEL,
    root_dir: str | Path | None = None,
    virtual_mode: bool = False,  # Allows unrestricted filesystem access
) -> object:
```
With `virtual_mode=False`, the agent has unrestricted filesystem access beyond the intended directories.

**Fix:**
Change default to `True` for production safety:
```python
virtual_mode: bool = True,  # Sandbox filesystem paths under root_dir
```
Document when `False` is appropriate (development/testing only).

---

### 3. No Async Support for Tools
**File:** `src/tools.py`
**Status:** [ ] Open

**Problem:**
All tools use synchronous file I/O (`read_text()`, `write_text()`). FastAPI (Phase 3) requires async operations to avoid blocking the event loop.

**Fix Options:**
1. Create async variants using `aiofiles`:
```python
import aiofiles

async def _read_config_async(path: Path) -> str:
    async with aiofiles.open(path, 'r', encoding='utf-8') as f:
        return await f.read()
```

2. Or use thread pool executor:
```python
from concurrent.futures import ThreadPoolExecutor
import asyncio

executor = ThreadPoolExecutor(max_workers=4)

async def get_agent_config_async(name: str) -> str:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(executor, get_agent_config, name)
```

**Dependencies to add:** `aiofiles` (if using option 1)

---

### 4. No File Locking for Concurrent Access
**File:** `src/tools.py` (lines 250-256)
**Status:** [ ] Open

**Problem:**
```python
output_path.write_text(
    json.dumps(serialized, indent=2, ensure_ascii=False),
    encoding="utf-8",
)
```
Multiple concurrent requests could corrupt agent configuration files.

**Fix:**
```python
import fcntl  # Unix only
# or
import portalocker  # Cross-platform

def _write_config_safely(path: Path, content: str) -> None:
    with open(path, 'w', encoding='utf-8') as f:
        portalocker.lock(f, portalocker.LOCK_EX)
        f.write(content)
        portalocker.unlock(f)
```

**Dependencies to add:** `portalocker` (for cross-platform support)

---

### 14. Anthropic Content Blocks Not Normalized in `execute_created_agent`
**File:** `src/tools.py` (lines ~981, 1171)
**Status:** [x] Fixed
**Identified:** Phase 2.5 CLI Testing (2026-01-10)

**Problem:**
When executing a created agent that uses MCP tools (e.g., websearch), the response crashes with:
```
AttributeError: 'list' object has no attribute 'split'
```

**Root Cause:**
Anthropic's Messages API returns `content` as a **list of content blocks**, not a string:
```json
{
  "content": [
    {"type": "tool_use", "id": "...", "name": "tavily-search", "input": {...}},
    {"type": "text", "text": "# Report content here..."}
  ]
}
```

The code at line ~981 assigns this directly:
```python
if hasattr(final_message, 'content'):
    result["response"] = final_message.content  # Can be str OR list!
```

Then at line 1171, it assumes string:
```python
response_lines = result["response"].split("\n")  # CRASH if list!
```

**Why This Happens:**
| Provider | Content Format |
|----------|----------------|
| Anthropic | Always returns list of content blocks |
| OpenAI | Returns simple string |
| LangChain ChatAnthropic | Preserves Anthropic's list format |

This is by design - Anthropic uses content blocks to support `tool_use`, extended thinking, citations, etc.

**Fix (Recommended - Utility Function + Normalize at Assignment):**

1. Create utility function in `src/tools.py`:
```python
def extract_text_from_content(content: str | list | Any) -> str:
    """Extract text from LangChain message content.

    Handles both string content and Anthropic-style content blocks.

    Args:
        content: Message content - can be string or list of content blocks

    Returns:
        Extracted text as a single string
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
```

2. Use at assignment (line ~981):
```python
if hasattr(final_message, 'content'):
    result["response"] = extract_text_from_content(final_message.content)
else:
    result["response"] = str(final_message)
```

**Alternative Approaches:**
- **Approach 2:** Use LangChain's `.content_blocks` property (newer API)
- **Approach 3:** Defensive check at line 1171 (not recommended - doesn't fix root cause)

**Also Apply To:**
- `src/cli.py` line ~2410 in `_stream_agent_response()` (partially fixed for streaming, needs review)
- Any other location extracting `.content` from LangChain messages

**References:**
- [Anthropic Messages API - Content Blocks](https://docs.anthropic.com/en/api/messages)
- [LangChain ChatAnthropic Integration](https://python.langchain.com/docs/integrations/chat/anthropic)
- [GitHub Issue #6238 - Content format inconsistency](https://github.com/langchain-ai/langchainjs/issues/6238)

---

## Medium

### 5. Return Type `object` Instead of Proper Type
**File:** `src/builder.py` (line 75)
**Status:** [ ] Open

**Problem:**
```python
def create_builder(...) -> object:  # Vague return type
```

**Fix:**
```python
from langgraph.graph.state import CompiledStateGraph

def create_builder(...) -> CompiledStateGraph:
```

---

### 6. `DEFAULT_MODEL` Hardcoded in Multiple Places
**Files:**
- `src/schemas.py` (line 76)
- `src/tools.py` (line 115)
- `src/prompts.py` (line 58) - canonical source

**Status:** [ ] Open

**Problem:**
Model string `"anthropic:claude-opus-4-5-20251101"` appears in 3 files.

**Fix:**
Import from `prompts.py` in other modules:
```python
# In schemas.py
from .prompts import DEFAULT_MODEL

class AgentConfig(BaseModel):
    model: str = Field(default=DEFAULT_MODEL, ...)

# In tools.py
from .prompts import DEFAULT_MODEL

@tool
def create_agent_config(
    ...
    model: str = DEFAULT_MODEL,
) -> str:
```

**Note:** May cause circular import - check import order.

---

### 7. No Validator for Model Field Format
**File:** `src/schemas.py` (lines 75-78)
**Status:** [ ] Open

**Problem:**
The `model` field accepts any string. Typos like `"anthropic:claude-opus"` (missing version) are silently accepted.

**Fix:**
```python
@field_validator("model")
@classmethod
def validate_model_format(cls, v: str) -> str:
    """Validate model follows provider:model-name pattern."""
    if ":" not in v:
        raise ValueError("Model must follow 'provider:model-name' format (e.g., 'anthropic:claude-opus-4-5-20251101')")
    provider, model_name = v.split(":", 1)
    if not provider or not model_name:
        raise ValueError("Model must have non-empty provider and model name")
    return v
```

---

### 8. `tool_calls` Uses Untyped `dict`
**File:** `src/schemas.py` (lines 162-165)
**Status:** [ ] Open

**Problem:**
```python
tool_calls: list[dict] = Field(...)  # Loses type safety
```

**Fix:**
```python
from typing import Any

class ToolCall(BaseModel):
    """Record of a tool invocation during agent execution."""
    tool_name: str = Field(..., description="Name of the tool that was called")
    arguments: dict[str, Any] = Field(default_factory=dict, description="Arguments passed to the tool")
    result: Optional[str] = Field(default=None, description="Result returned by the tool")

class AgentResponse(BaseModel):
    tool_calls: list[ToolCall] = Field(default_factory=list, description="List of tool calls made during execution")
```

---

### 9. Project Name Mismatch
**Files:**
- `pyproject.toml` (line 2): `name = "sigil"`
- `src/__init__.py` (line 1): `"""Sigil - A meta-agent framework..."""`
- Everywhere else: "ACTi Agent Builder"

**Status:** [ ] Open

**Fix:**
Decide on canonical name and update all references. Options:
1. Keep "Sigil" as package name, "ACTi Agent Builder" as product name
2. Rename package to `acti-agent-builder` everywhere

If option 2:
```toml
# pyproject.toml
[project]
name = "acti-agent-builder"
```

```python
# src/__init__.py
"""ACTi Agent Builder - A meta-agent that creates executable AI agents..."""
```

---

## Low

### 10. Missing `__all__` Export in schemas.py
**File:** `src/schemas.py`
**Status:** [ ] Open

**Problem:**
Module exports multiple classes but lacks `__all__` declaration (unlike `prompts.py`).

**Fix:**
Add at end of file:
```python
__all__ = [
    "Stratum",
    "MCP_TOOL_CATEGORIES",
    "AgentConfig",
    "CreateAgentRequest",
    "RunAgentRequest",
    "AgentResponse",
    "CreateAgentResponse",
    "ToolInfo",
    "ToolListResponse",
]
```

---

### 11. Test File Makes Real API Calls Without Skip Decorator
**File:** `tests/test_hello_world.py`
**Status:** [ ] Open

**Problem:**
Tests make real API calls without proper skip conditions for CI environments.

**Fix:**
```python
import os
import pytest

@pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY") or
    os.environ.get("SKIP_INTEGRATION_TESTS", "").lower() == "true",
    reason="Requires real ANTHROPIC_API_KEY and integration tests enabled"
)
def test_hello_world():
    ...
```

Also fix the pytest warning about return values:
```python
def test_hello_world():
    # ... existing code
    assert "messages" in result  # Use assert, don't return True
```

---

### 12. CLI Missing Signal Handlers
**File:** `src/builder.py` (lines 161-230)
**Status:** [ ] Open

**Problem:**
CLI catches `KeyboardInterrupt` but could benefit from proper signal handlers for graceful shutdown.

**Fix:**
```python
import signal
import sys

def _run_cli() -> None:
    def signal_handler(sig, frame):
        print("\nShutting down gracefully...")
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # ... rest of function
```

---

### 13. Missing Logging Infrastructure
**File:** All modules
**Status:** [ ] Open

**Problem:**
No logging module configured. Errors are silently handled or returned as strings.

**Fix:**
Add logging setup:
```python
# src/__init__.py or src/logging_config.py
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# In each module
logger = logging.getLogger(__name__)

# Usage
logger.warning(f"Failed to read agent config {config_path}: {e}")
```

---

## Dependencies to Add

For the fixes above, add to `pyproject.toml`:
```toml
dependencies = [
    # ... existing
    "aiofiles>=23.0.0",      # Async file I/O (fix #3)
    "portalocker>=2.0.0",    # Cross-platform file locking (fix #4)
]
```

---

## Progress Tracking

| # | Severity | Issue | Status |
|---|----------|-------|--------|
| 1 | Critical | Path traversal in get_agent_config | [x] Fixed |
| 2 | High | Unsafe virtual_mode default | [x] Fixed |
| 3 | High | No async support | [x] Fixed |
| 4 | High | No file locking | [x] Fixed |
| 5 | Medium | Return type object | [x] Fixed |
| 6 | Medium | DEFAULT_MODEL duplication | [x] Fixed |
| 7 | Medium | No model format validator | [x] Fixed |
| 8 | Medium | Untyped tool_calls | [x] Fixed |
| 9 | Medium | Project name mismatch | [x] Fixed |
| 10 | Low | Missing __all__ in schemas | [x] Fixed |
| 11 | Low | Test skip decorator | [x] Fixed |
| 12 | Low | CLI signal handlers | [x] Fixed |
| 13 | Low | Missing logging | [x] Fixed |
| 14 | High | Anthropic content blocks not normalized | [x] Fixed |
| 15 | Critical | GraphRecursionError - recursion_limit not propagating | [ ] Open |
| 16 | Critical | Agent non-termination / infinite tool loop | [ ] Open |

**Total: 16 issues (3 critical, 4 high, 5 medium, 4 low) - 14 Fixed, 2 Open**

**Note:** Issues 15-16 are documented in detail in `breaks.md`

---

*Last updated: 2026-01-10*
*Phase 1: Senior Code Review*
*Phase 2.5: CLI Testing*
