"""Auto-offloading middleware for large tool results.

This middleware detects large tool outputs and saves them to files,
replacing them with references that the LLM can use to retrieve
the full data when needed.

Integration Point:
    The middleware hooks into the post_step phase for the "execute" step,
    checking prior_outputs in the assembled context and offloading any
    results that exceed the configured threshold.

Usage:
    >>> from sigil.core.middlewares import OffloadingMiddleware
    >>> from sigil.core.middleware import MiddlewareChain
    >>>
    >>> chain = MiddlewareChain()
    >>> chain.add(OffloadingMiddleware(threshold_chars=80000))

Example with settings:
    >>> middleware = OffloadingMiddleware.from_settings(settings)
    >>> chain.add(middleware)
"""

from __future__ import annotations

import json
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Coroutine, Optional, TYPE_CHECKING
import logging

from sigil.core.middleware import BaseMiddleware

if TYPE_CHECKING:
    from sigil.config.settings import SigilSettings

logger = logging.getLogger(__name__)


class ToolResultStore:
    """Storage for offloaded tool results.

    Handles saving, loading, and summarizing large tool results
    that have been offloaded from the context.

    The store maintains an index file that tracks all stored results,
    allowing for efficient lookup and cleanup operations.

    Attributes:
        storage_path: Directory path for storing offloaded results.

    Example:
        >>> store = ToolResultStore("outputs/tool_results")
        >>> storage_id = store.save("step-1", large_content, {"tool": "tavily"})
        >>> content = store.load(storage_id)
        >>> summary = store.summarize(content, max_items=3)
    """

    def __init__(self, storage_path: str = "outputs/tool_results") -> None:
        """Initialize the tool result store.

        Args:
            storage_path: Directory path for storing offloaded results.
                Will be created if it doesn't exist.
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self._index: dict[str, dict[str, Any]] = {}
        self._load_index()

    def _load_index(self) -> None:
        """Load the index of stored results from disk."""
        index_path = self.storage_path / "index.json"
        if index_path.exists():
            try:
                with open(index_path, "r", encoding="utf-8") as f:
                    self._index = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load offload index: {e}")
                self._index = {}

    def _save_index(self) -> None:
        """Save the index of stored results to disk."""
        index_path = self.storage_path / "index.json"
        try:
            with open(index_path, "w", encoding="utf-8") as f:
                json.dump(self._index, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save offload index: {e}")

    def _generate_id(self, step_id: str, content: str) -> str:
        """Generate unique ID for offloaded content.

        Args:
            step_id: Original step identifier.
            content: Content being stored (used for hash).

        Returns:
            A unique storage ID combining step_id, timestamp, and content hash.
        """
        content_hash = hashlib.sha256(content.encode()).hexdigest()[:8]
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        return f"{step_id}_{timestamp}_{content_hash}"

    def save(
        self,
        step_id: str,
        content: str,
        metadata: Optional[dict[str, Any]] = None,
    ) -> str:
        """Save content to file and return the storage ID.

        Stores the content as a JSON file with associated metadata,
        and updates the index for tracking.

        Args:
            step_id: Original step identifier.
            content: Full content to save.
            metadata: Optional metadata about the content.

        Returns:
            Storage ID that can be used to retrieve the content.

        Raises:
            IOError: If the file cannot be written.
        """
        storage_id = self._generate_id(step_id, content)
        file_path = self.storage_path / f"{storage_id}.json"

        stored_data = {
            "storage_id": storage_id,
            "step_id": step_id,
            "content": content,
            "metadata": metadata or {},
            "stored_at": datetime.now(timezone.utc).isoformat(),
            "size_chars": len(content),
            "size_tokens_estimate": len(content) // 4,
        }

        try:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(stored_data, f, indent=2, ensure_ascii=False)

            # Update index
            self._index[storage_id] = {
                "step_id": step_id,
                "file_path": str(file_path),
                "stored_at": stored_data["stored_at"],
                "size_chars": len(content),
            }
            self._save_index()

            logger.info(f"Offloaded {len(content):,} chars to {storage_id}")
            return storage_id

        except Exception as e:
            logger.error(f"Failed to save offloaded content: {e}")
            raise

    def load(self, storage_id: str) -> Optional[str]:
        """Load content by storage ID.

        Args:
            storage_id: The storage ID returned from save().

        Returns:
            The stored content, or None if not found.
        """
        file_path = self.storage_path / f"{storage_id}.json"

        if not file_path.exists():
            logger.warning(f"Offloaded content not found: {storage_id}")
            return None

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data.get("content")
        except Exception as e:
            logger.error(f"Failed to load offloaded content: {e}")
            return None

    def summarize(self, content: str, max_items: int = 3) -> str:
        """Generate a summary of the content for context.

        Attempts to parse the content as JSON and extract key information.
        For Tavily-style results, extracts titles. For lists, shows first
        few items. For plain text, returns first few lines.

        Args:
            content: Full content to summarize.
            max_items: Maximum items to include in summary.

        Returns:
            A brief summary of key findings.
        """
        # Try to parse as JSON (e.g., Tavily results)
        try:
            data = json.loads(content)

            # Handle Tavily-style results
            if isinstance(data, dict) and "results" in data:
                results = data["results"][:max_items]
                summary_lines = []
                for i, result in enumerate(results, 1):
                    title = result.get("title", "Untitled")
                    summary_lines.append(f"{i}. {title}")

                total = len(data["results"])
                suffix = f" (+ {total - max_items} more)" if total > max_items else ""
                return "Key findings:\n" + "\n".join(summary_lines) + suffix

            # Handle list results
            elif isinstance(data, list):
                items = data[:max_items]
                summary_lines = []
                for i, item in enumerate(items, 1):
                    item_str = str(item)[:100]
                    if len(str(item)) > 100:
                        item_str += "..."
                    summary_lines.append(f"{i}. {item_str}")

                total = len(data)
                suffix = f" (+ {total - max_items} more)" if total > max_items else ""
                return "Items:\n" + "\n".join(summary_lines) + suffix

            # Generic dict - show keys
            else:
                keys = list(data.keys())[:max_items]
                total = len(data.keys())
                suffix = f" (+ {total - max_items} more)" if total > max_items else ""
                return f"Contains keys: {', '.join(keys)}{suffix}"

        except json.JSONDecodeError:
            # Plain text - return first few lines
            lines = content.split("\n")[:max_items]
            result = "\n".join(lines)
            if len(content.split("\n")) > max_items:
                result += "\n..."
            return result

    def list_stored(self, step_id: Optional[str] = None) -> list[dict[str, Any]]:
        """List stored results, optionally filtered by step.

        Args:
            step_id: Optional step ID to filter by.

        Returns:
            List of stored result metadata, sorted by most recent first.
        """
        results = []
        for storage_id, metadata in self._index.items():
            if step_id is None or metadata.get("step_id") == step_id:
                results.append({
                    "storage_id": storage_id,
                    **metadata
                })
        return sorted(results, key=lambda x: x.get("stored_at", ""), reverse=True)

    def delete(self, storage_id: str) -> bool:
        """Delete a stored result.

        Removes both the stored file and the index entry.

        Args:
            storage_id: The storage ID to delete.

        Returns:
            True if deleted, False if not found.
        """
        file_path = self.storage_path / f"{storage_id}.json"

        if file_path.exists():
            try:
                file_path.unlink()
            except Exception as e:
                logger.error(f"Failed to delete file {file_path}: {e}")
                return False

            self._index.pop(storage_id, None)
            self._save_index()
            logger.debug(f"Deleted offloaded result: {storage_id}")
            return True

        return False

    def cleanup_old(self, max_age_hours: int = 24) -> int:
        """Remove results older than max_age_hours.

        Useful for periodic cleanup to prevent unbounded disk usage.

        Args:
            max_age_hours: Maximum age in hours.

        Returns:
            Number of results deleted.
        """
        deleted = 0
        cutoff = datetime.now(timezone.utc).timestamp() - (max_age_hours * 3600)

        for storage_id, metadata in list(self._index.items()):
            stored_at = metadata.get("stored_at", "")
            try:
                stored_time = datetime.fromisoformat(stored_at.replace("Z", "+00:00"))
                if stored_time.timestamp() < cutoff:
                    if self.delete(storage_id):
                        deleted += 1
            except Exception:
                continue

        if deleted > 0:
            logger.info(f"Cleaned up {deleted} offloaded results older than {max_age_hours}h")

        return deleted

    def get_total_size(self) -> int:
        """Get total size of all stored results in characters.

        Returns:
            Total characters stored across all results.
        """
        return sum(m.get("size_chars", 0) for m in self._index.values())


class OffloadingMiddleware(BaseMiddleware):
    """Middleware that auto-offloads large tool results.

    Monitors tool execution results and automatically offloads
    results that exceed the threshold to files, replacing them
    with references that include a brief summary.

    The middleware operates at the post_step hook for the "execute" step,
    which is where tool results have been collected in prior_outputs.
    Large results are saved to disk and replaced with reference messages
    that the LLM can use to understand what data is available.

    Attributes:
        threshold_chars: Character threshold for offloading.
        store: ToolResultStore instance for storage operations.
        enabled: Whether offloading is enabled.
        include_summary: Whether to include a summary in the reference.
        max_summary_items: Maximum items to include in summaries.

    Example:
        >>> middleware = OffloadingMiddleware(
        ...     threshold_chars=80000,
        ...     storage_path="outputs/tool_results",
        ...     include_summary=True,
        ... )
        >>> chain.add(middleware)
    """

    def __init__(
        self,
        threshold_chars: int = 80000,
        storage_path: str = "outputs/tool_results",
        enabled: bool = True,
        include_summary: bool = True,
        max_summary_items: int = 3,
    ) -> None:
        """Initialize the offloading middleware.

        Args:
            threshold_chars: Character threshold for offloading (default: 80000).
                Results larger than this will be offloaded to disk.
            storage_path: Directory for storing offloaded results.
            enabled: Whether offloading is enabled (default: True).
            include_summary: Whether to include a summary in the reference
                message (default: True).
            max_summary_items: Maximum items to include in summaries (default: 3).
        """
        self.threshold_chars = threshold_chars
        self.store = ToolResultStore(storage_path)
        self.enabled = enabled
        self.include_summary = include_summary
        self.max_summary_items = max_summary_items
        self._offloaded_this_request: list[str] = []

    @property
    def name(self) -> str:
        """Get the middleware name."""
        return "OffloadingMiddleware"

    @classmethod
    def from_settings(cls, settings: "SigilSettings") -> "OffloadingMiddleware":
        """Create middleware from SigilSettings.

        Extracts context-related settings from the settings object
        to configure the middleware. Falls back to defaults if
        context settings are not available.

        Args:
            settings: SigilSettings instance.

        Returns:
            Configured OffloadingMiddleware instance.
        """
        context_settings = getattr(settings, "context", None)

        if context_settings:
            return cls(
                threshold_chars=getattr(
                    context_settings, "offload_threshold_chars", 80000
                ),
                storage_path=getattr(
                    context_settings, "offload_storage_path", "outputs/tool_results"
                ),
                enabled=getattr(
                    context_settings, "enable_auto_offload", True
                ),
            )

        return cls()

    async def post_step(self, step_name: str, ctx: Any, result: Any) -> Any:
        """Check and offload large results after execute step.

        This hook is called after each pipeline step. It only performs
        offloading for the "execute" step, which is when tool results
        have been collected in prior_outputs.

        Large results (exceeding threshold_chars) are saved to disk
        and replaced with reference messages that include a summary.

        Args:
            step_name: Name of the step that executed.
            ctx: Pipeline context object.
            result: Result from the step.

        Returns:
            The result unchanged (context is modified in place).
        """
        if not self.enabled:
            return result

        # Only process after execute step
        if step_name != "execute":
            return result

        # Reset tracking for this request
        self._offloaded_this_request = []

        # Check prior_outputs in assembled context
        assembled_context = getattr(ctx, "assembled_context", None)
        if not assembled_context:
            return result

        prior_outputs = assembled_context.get("prior_outputs", {})

        for step_id, output_data in prior_outputs.items():
            # Handle both dict format and raw string
            if isinstance(output_data, dict):
                output = output_data.get("output", "")
            else:
                output = str(output_data)
                output_data = {"output": output}
                prior_outputs[step_id] = output_data

            if not output or len(output) <= self.threshold_chars:
                continue

            # Offload large result
            try:
                storage_id = self.store.save(
                    step_id=step_id,
                    content=output,
                    metadata={"original_step": step_id}
                )
            except Exception as e:
                logger.error(f"Failed to offload step '{step_id}': {e}")
                continue

            # Create reference message
            summary = ""
            if self.include_summary:
                summary = self.store.summarize(output, self.max_summary_items)

            reference_msg = (
                f"[OFFLOADED: Full results ({len(output):,} chars) "
                f"saved to storage_id='{storage_id}']\n"
                f"Use read_tool_result('{storage_id}') to retrieve full data.\n"
            )
            if summary:
                reference_msg += f"\n{summary}"

            # Replace output with reference
            output_data["output"] = reference_msg
            output_data["offloaded"] = True
            output_data["storage_id"] = storage_id
            output_data["original_size"] = len(output)

            self._offloaded_this_request.append(storage_id)

            logger.info(
                f"Offloaded step '{step_id}' output: {len(output):,} chars -> "
                f"storage_id='{storage_id}'"
            )

        return result

    def get_offloaded_this_request(self) -> list[str]:
        """Get list of storage IDs offloaded in current request.

        Returns:
            List of storage IDs created during the current request.
        """
        return self._offloaded_this_request.copy()

    def retrieve(self, storage_id: str) -> Optional[str]:
        """Retrieve offloaded content by storage ID.

        Convenience method for retrieving content without directly
        accessing the store.

        Args:
            storage_id: The storage ID.

        Returns:
            The stored content, or None if not found.
        """
        return self.store.load(storage_id)

    def cleanup(self, max_age_hours: int = 24) -> int:
        """Clean up old offloaded results.

        Delegates to the store's cleanup_old method.

        Args:
            max_age_hours: Maximum age in hours.

        Returns:
            Number of results deleted.
        """
        return self.store.cleanup_old(max_age_hours)

    def reset(self) -> None:
        """Reset the middleware state.

        Clears the list of offloaded results for this request.
        """
        self._offloaded_this_request = []


def create_read_tool_result_tool(
    store: ToolResultStore,
) -> dict[str, Any]:
    """Create a tool definition for reading offloaded results.

    This function creates a tool that can be added to the available
    tools for the LLM, allowing it to retrieve full content from
    offloaded results when needed.

    Args:
        store: ToolResultStore instance to read from.

    Returns:
        Tool definition dict with name, description, parameters, and handler.

    Example:
        >>> store = ToolResultStore("outputs/tool_results")
        >>> tool = create_read_tool_result_tool(store)
        >>> # Register tool with your tool registry
        >>> registry.add_tool(tool["name"], tool["handler"])
    """
    async def read_tool_result(storage_id: str) -> str:
        """Retrieve full content from an offloaded tool result.

        Args:
            storage_id: The storage ID from the offload reference.

        Returns:
            Full content of the tool result, or error message if not found.
        """
        content = store.load(storage_id)
        if content is None:
            return f"Error: No stored result found for storage_id='{storage_id}'"
        return content

    return {
        "name": "read_tool_result",
        "description": (
            "Retrieve full content from an offloaded tool result. "
            "Use when you see '[OFFLOADED:...]' in tool outputs and need "
            "the complete data for detailed analysis."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "storage_id": {
                    "type": "string",
                    "description": "The storage ID from the offload reference"
                }
            },
            "required": ["storage_id"]
        },
        "handler": read_tool_result
    }


# Type alias for the handler function
ReadToolResultHandler = Callable[[str], Coroutine[Any, Any, str]]


__all__ = [
    "ToolResultStore",
    "OffloadingMiddleware",
    "create_read_tool_result_tool",
    "ReadToolResultHandler",
]
