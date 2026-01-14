"""Utility functions for formatting tool results in LLM prompts.

This module provides utilities for extracting and formatting tool outputs
(like Tavily web search results) from the context dictionary into readable
sections for LLM prompts.

Problem being solved:
    Tool outputs are collected in the context dictionary under `prior_outputs`
    with a nested structure like:

    {
        "prior_outputs": {
            "step-1": {"output": "<JSON search results>", "tokens_used": 0}
        }
    }

    Without proper formatting, this gets dumped as raw dict string instead of
    readable content that the LLM can effectively use for reasoning.

Functions:
    extract_tool_outputs: Extract prior_outputs from context.
    format_tool_results_section: Format tool outputs for LLM prompts.
    format_tavily_results: Parse and format Tavily JSON output.
    build_tool_aware_context_string: Build context string with tool results.

Example:
    >>> from sigil.reasoning.strategies.utils import build_tool_aware_context_string
    >>>
    >>> context = {
    ...     "prior_outputs": {
    ...         "step-1": {"output": '{"results": [...]}', "tokens_used": 100}
    ...     },
    ...     "user_id": "test-user",
    ... }
    >>> formatted = build_tool_aware_context_string(context)
    >>> # Returns string with "## Tool Results" section
"""

from __future__ import annotations

import json
import logging
from typing import Any


# =============================================================================
# Module Logger
# =============================================================================

logger = logging.getLogger(__name__)


# =============================================================================
# Constants
# =============================================================================

DEFAULT_MAX_CHARS_PER_RESULT = 4000
"""Default maximum characters per tool result before truncation."""

DEFAULT_MAX_OUTPUTS = 10
"""Default maximum number of tool outputs to format."""

MAX_TAVILY_RESULTS = 20
"""Maximum number of Tavily search results to include."""

MAX_CONTEXT_VALUE_LENGTH = 200
"""Maximum length for non-tool context values before truncation."""

TRUNCATION_MARKER = "\n[truncated]"
"""Marker appended when content is truncated."""


# =============================================================================
# Tool Output Extraction
# =============================================================================


def extract_tool_outputs(context: dict[str, Any]) -> dict[str, str]:
    """Extract prior_outputs from context into a flattened dict.

    Extracts the `prior_outputs` key from the context dictionary and
    returns a simplified mapping of step_id to output string.

    Args:
        context: Context dictionary that may contain `prior_outputs`.

    Returns:
        Dictionary mapping step_id to output string.
        Returns empty dict if `prior_outputs` is missing or invalid.

    Example:
        >>> context = {
        ...     "prior_outputs": {
        ...         "step-1": {"output": "search results", "tokens_used": 0},
        ...         "step-2": {"output": "more results", "tokens_used": 50},
        ...     }
        ... }
        >>> outputs = extract_tool_outputs(context)
        >>> outputs
        {'step-1': 'search results', 'step-2': 'more results'}
    """
    if not context:
        logger.debug("extract_tool_outputs: Empty context provided")
        return {}

    prior_outputs = context.get("prior_outputs")
    if not prior_outputs:
        logger.debug("extract_tool_outputs: No prior_outputs in context")
        return {}

    if not isinstance(prior_outputs, dict):
        logger.warning(
            f"extract_tool_outputs: prior_outputs is not a dict, "
            f"got {type(prior_outputs).__name__}"
        )
        return {}

    result: dict[str, str] = {}
    for step_id, step_data in prior_outputs.items():
        if isinstance(step_data, dict):
            output = step_data.get("output", "")
            if output:
                result[step_id] = str(output)
        elif isinstance(step_data, str):
            # Handle case where step_data is directly a string
            result[step_id] = step_data
        else:
            logger.debug(
                f"extract_tool_outputs: Skipping step {step_id}, "
                f"unexpected type {type(step_data).__name__}"
            )

    logger.debug(f"extract_tool_outputs: Extracted {len(result)} outputs")
    return result


# =============================================================================
# Tavily Result Formatting
# =============================================================================


def format_tavily_results(raw_output: str) -> str:
    """Parse and format Tavily JSON output into readable text.

    Parses Tavily search results JSON and formats them as a readable list
    with titles, snippets, and URLs. Handles both single results and
    multiple results in the standard Tavily response format.

    Args:
        raw_output: Raw JSON string from Tavily search/extract/qna.

    Returns:
        Formatted string with titles, snippets, and URLs.
        Returns original string if parsing fails.

    Example:
        >>> raw = '{"results": [{"title": "AI News", "url": "...", "content": "..."}]}'
        >>> formatted = format_tavily_results(raw)
        >>> # Returns:
        >>> # 1. AI News
        >>> #    ...
        >>> #    Source: ...
    """
    try:
        data = json.loads(raw_output)
    except json.JSONDecodeError as e:
        logger.debug(f"format_tavily_results: Failed to parse JSON: {e}")
        return raw_output

    lines: list[str] = []

    # Handle QnA response format {"answer": "..."}
    if "answer" in data and isinstance(data["answer"], str):
        answer = data["answer"]
        lines.append("Answer from search:")
        lines.append(answer)
        return "\n".join(lines)

    # Handle search results format {"results": [...], "query": "...", ...}
    results = data.get("results", [])
    if not results:
        # Maybe it's a direct list of results
        if isinstance(data, list):
            results = data
        else:
            logger.debug("format_tavily_results: No results found in response")
            return raw_output

    # Add query context if available
    query = data.get("query")
    if query:
        lines.append(f"Search query: {query}")
        lines.append("")

    # Add AI answer if available (from include_answer=True)
    ai_answer = data.get("answer")
    if ai_answer:
        lines.append("AI-generated answer:")
        lines.append(ai_answer)
        lines.append("")
        lines.append("Sources:")
        lines.append("")

    # Format each result (limit to MAX_TAVILY_RESULTS to prevent memory issues)
    for i, result in enumerate(results[:MAX_TAVILY_RESULTS], 1):
        if not isinstance(result, dict):
            continue

        title = result.get("title", "Untitled")
        url = result.get("url", "")
        content = result.get("content", result.get("snippet", ""))

        lines.append(f"{i}. {title}")
        if content:
            # Clean up content - remove excessive whitespace
            content = " ".join(content.split())
            lines.append(f"   {content}")
        if url:
            lines.append(f"   Source: {url}")
        lines.append("")

    return "\n".join(lines).strip()


# =============================================================================
# Tool Type Detection
# =============================================================================


def _detect_tool_type(output: str) -> str:
    """Detect the tool type from output structure.

    Analyzes the JSON structure of an output to determine what tool
    produced it. Currently supports detection of:
    - Tavily search results
    - Tavily QnA results
    - Generic JSON
    - Plain text

    Args:
        output: The tool output string.

    Returns:
        Tool type identifier: "tavily_search", "tavily_qna", "json", or "text".
    """
    try:
        data = json.loads(output)
    except json.JSONDecodeError:
        return "text"

    if not isinstance(data, dict):
        return "json"

    # Detect Tavily QnA format
    if "answer" in data and isinstance(data.get("answer"), str):
        # Could be QnA or search with include_answer
        if "results" not in data:
            return "tavily_qna"

    # Detect Tavily search format
    if "results" in data:
        results = data.get("results", [])
        if results and isinstance(results, list):
            first = results[0] if results else {}
            if isinstance(first, dict) and any(
                k in first for k in ["title", "url", "content", "snippet"]
            ):
                return "tavily_search"

    # Detect Tavily extract format
    if "raw_content" in data or "extracted_content" in data:
        return "tavily_extract"

    return "json"


# =============================================================================
# Tool Results Section Formatting
# =============================================================================


def format_tool_results_section(
    prior_outputs: dict[str, Any],
    max_chars_per_result: int = DEFAULT_MAX_CHARS_PER_RESULT,
    max_outputs: int = DEFAULT_MAX_OUTPUTS,
) -> str:
    """Format tool outputs into a readable section for LLM prompts.

    Takes the prior_outputs dictionary and formats each output based on
    its detected type. Tavily results get special formatting, while other
    outputs are formatted as readable JSON or text.

    Args:
        prior_outputs: Dictionary from context["prior_outputs"] with structure:
            {"step-id": {"output": "...", "tokens_used": N}, ...}
        max_chars_per_result: Maximum characters per result before truncation.
            Defaults to 4000. Must be positive.
        max_outputs: Maximum number of tool outputs to format.
            Defaults to 10. Prevents token limit issues with many steps.

    Returns:
        Formatted string ready for LLM prompt inclusion.
        Returns empty string if no outputs are provided.

    Note:
        Invalid structures in prior_outputs are silently skipped with
        debug-level logging.

    Example:
        >>> outputs = {
        ...     "step-1": {"output": '{"results": [...]}', "tokens_used": 100}
        ... }
        >>> section = format_tool_results_section(outputs)
        >>> # Returns formatted "### step-1\\n..." section
    """
    if not prior_outputs:
        logger.debug("format_tool_results_section: No outputs to format")
        return ""

    # Validate max_chars_per_result
    if max_chars_per_result < len(TRUNCATION_MARKER) + 1:
        logger.warning(
            f"format_tool_results_section: Invalid max_chars_per_result={max_chars_per_result}, "
            f"using default {DEFAULT_MAX_CHARS_PER_RESULT}"
        )
        max_chars_per_result = DEFAULT_MAX_CHARS_PER_RESULT

    sections: list[str] = []
    output_count = 0

    for step_id, step_data in prior_outputs.items():
        # Limit number of outputs to prevent token issues
        if output_count >= max_outputs:
            logger.debug(
                f"format_tool_results_section: Reached max_outputs limit ({max_outputs}), "
                f"skipping remaining outputs"
            )
            break
        # Extract output string
        if isinstance(step_data, dict):
            output = step_data.get("output", "")
        elif isinstance(step_data, str):
            output = step_data
        else:
            logger.debug(
                f"format_tool_results_section: Skipping {step_id}, "
                f"unexpected type {type(step_data).__name__}"
            )
            continue

        if not output:
            continue

        # Detect tool type and format accordingly
        tool_type = _detect_tool_type(output)
        logger.debug(f"format_tool_results_section: {step_id} detected as {tool_type}")

        if tool_type in ("tavily_search", "tavily_qna", "tavily_extract"):
            formatted = format_tavily_results(output)
        elif tool_type == "json":
            # Format JSON nicely
            try:
                data = json.loads(output)
                formatted = json.dumps(data, indent=2)
            except json.JSONDecodeError:
                formatted = output
        else:
            # Plain text
            formatted = output

        # Truncate if necessary
        if len(formatted) > max_chars_per_result:
            formatted = formatted[: max_chars_per_result - len(TRUNCATION_MARKER)]
            formatted += TRUNCATION_MARKER
            logger.debug(
                f"format_tool_results_section: Truncated {step_id} "
                f"to {max_chars_per_result} chars"
            )

        sections.append(f"### {step_id}\n{formatted}")
        output_count += 1

    if not sections:
        return ""

    result = "\n\n".join(sections)
    logger.debug(
        f"format_tool_results_section: Formatted {len(sections)} sections, "
        f"total {len(result)} chars"
    )
    return result


# =============================================================================
# Tool-Aware Context String Builder
# =============================================================================


def build_tool_aware_context_string(context: dict[str, Any]) -> str:
    """Build a context string that properly formats tool results.

    Creates a formatted context string for LLM prompts where:
    - Tool results from `prior_outputs` get a special "## Tool Results" section
    - Other context items are formatted as key-value pairs
    - Empty or missing values are handled gracefully

    Args:
        context: Context dictionary that may contain:
            - prior_outputs: Tool execution results
            - Any other context key-value pairs

    Returns:
        Formatted context string ready for LLM prompt inclusion.
        Returns empty string if context is empty.

    Example:
        >>> context = {
        ...     "prior_outputs": {"step-1": {"output": "...", "tokens_used": 0}},
        ...     "user_id": "test-user",
        ...     "session_id": "abc123",
        ... }
        >>> formatted = build_tool_aware_context_string(context)
        >>> # Returns:
        >>> # - user_id: test-user
        >>> # - session_id: abc123
        >>> #
        >>> # ## Tool Results
        >>> # ### step-1
        >>> # ...
    """
    if not context:
        logger.debug("build_tool_aware_context_string: Empty context")
        return ""

    lines: list[str] = []

    # Extract prior_outputs for special handling
    prior_outputs = context.get("prior_outputs", {})

    # Format other context items
    other_items: list[str] = []
    for key, value in context.items():
        if key == "prior_outputs":
            continue

        # Skip None or empty values
        if value is None:
            continue

        # Format the value
        if isinstance(value, (dict, list)):
            # Compact JSON for complex values
            try:
                value_str = json.dumps(value, separators=(",", ":"))
                # Truncate very long values
                if len(value_str) > MAX_CONTEXT_VALUE_LENGTH:
                    value_str = value_str[:MAX_CONTEXT_VALUE_LENGTH - 3] + "..."
                    logger.debug(f"build_tool_aware_context_string: Truncated {key} to {MAX_CONTEXT_VALUE_LENGTH} chars")
            except (TypeError, ValueError):
                value_str = str(value)
        else:
            value_str = str(value)

        if value_str:
            other_items.append(f"- {key}: {value_str}")

    if other_items:
        lines.extend(other_items)

    # Format tool results section
    if prior_outputs:
        tool_section = format_tool_results_section(prior_outputs)
        if tool_section:
            if lines:
                lines.append("")  # Blank line separator
            lines.append("## Tool Results")
            lines.append("")
            lines.append(tool_section)

    result = "\n".join(lines)
    logger.debug(
        f"build_tool_aware_context_string: Generated {len(result)} chars "
        f"with {len(other_items)} context items and "
        f"{len(prior_outputs)} tool outputs"
    )
    return result


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "extract_tool_outputs",
    "format_tool_results_section",
    "format_tavily_results",
    "build_tool_aware_context_string",
    # Constants
    "DEFAULT_MAX_CHARS_PER_RESULT",
    "DEFAULT_MAX_OUTPUTS",
    "MAX_TAVILY_RESULTS",
    "MAX_CONTEXT_VALUE_LENGTH",
    "TRUNCATION_MARKER",
]
