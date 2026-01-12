"""ContextManager - Context window management for Sigil v2.

This module implements the ContextManager, which assembles optimized context
windows for agent invocations by fetching from multiple sources in parallel
and managing token budgets.

Key Components:
    - ContextSource: Enumeration of context sources
    - ContextAssemblyResult: Result of parallel context assembly
    - ContextManager: Main context management class

Priority Order (highest to lowest):
    1. System prompt (required)
    2. Active task/step information
    3. Working memory (current scratchpad)
    4. Relevant long-term memories
    5. Recent conversation history
    6. Tool results from current turn

Example:
    >>> from sigil.context.manager import ContextManager
    >>>
    >>> manager = ContextManager(max_tokens=128000)
    >>> result = await manager.assemble(
    ...     task="Qualify lead John from Acme",
    ...     session_id="sess-123",
    ...     user_id="user-456",
    ... )
    >>> print(f"Total tokens: {result.total_tokens}")
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional, Callable, Awaitable, TypeVar

from sigil.config import get_settings
from sigil.config.settings import SigilSettings
from sigil.core.exceptions import SigilError
from sigil.telemetry.tokens import TokenBudget, TokenTracker


# =============================================================================
# Module Logger
# =============================================================================

logger = logging.getLogger(__name__)


# =============================================================================
# Constants
# =============================================================================

# Default token allocations (256K model: 150K input, 102.4K output)
DEFAULT_MAX_CONTEXT_TOKENS = 150_000
DEFAULT_RESERVED_FOR_RESPONSE = 4_000

# Token budget allocation percentages
SYSTEM_PROMPT_BUDGET_RATIO = 0.15  # 15% for system prompt
TASK_BUDGET_RATIO = 0.05  # 5% for task info
WORKING_MEMORY_BUDGET_RATIO = 0.10  # 10% for working memory
LONG_TERM_MEMORY_BUDGET_RATIO = 0.25  # 25% for long-term memories
CONVERSATION_HISTORY_BUDGET_RATIO = 0.35  # 35% for conversation history
TOOL_RESULTS_BUDGET_RATIO = 0.10  # 10% for tool results

# Simple token estimation (chars / 4 for English text)
CHARS_PER_TOKEN = 4


# =============================================================================
# Context Source Enum
# =============================================================================


class ContextSource(str, Enum):
    """Sources of context for assembly.

    Attributes:
        SYSTEM_PROMPT: The agent's system prompt
        TASK: Current task information
        WORKING_MEMORY: Session scratchpad/working memory
        LONG_TERM_MEMORY: Relevant memories from memory system
        CONVERSATION_HISTORY: Recent conversation messages
        TOOL_RESULTS: Results from tool executions
        PLAN: Current execution plan
        USER_CONTEXT: User-provided context
    """
    SYSTEM_PROMPT = "system_prompt"
    TASK = "task"
    WORKING_MEMORY = "working_memory"
    LONG_TERM_MEMORY = "long_term_memory"
    CONVERSATION_HISTORY = "conversation_history"
    TOOL_RESULTS = "tool_results"
    PLAN = "plan"
    USER_CONTEXT = "user_context"


# =============================================================================
# Context Item
# =============================================================================


@dataclass
class ContextItem:
    """A single item of context.

    Attributes:
        source: Where this context came from
        content: The actual content string or dict
        tokens: Estimated token count
        priority: Priority score (higher = more important)
        metadata: Additional metadata
    """
    source: ContextSource
    content: Any
    tokens: int
    priority: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_string(self) -> str:
        """Convert content to string representation."""
        if isinstance(self.content, str):
            return self.content
        elif isinstance(self.content, dict):
            import json
            return json.dumps(self.content, indent=2)
        elif isinstance(self.content, list):
            return "\n".join(str(item) for item in self.content)
        else:
            return str(self.content)


# =============================================================================
# Context Assembly Result
# =============================================================================


@dataclass
class ContextAssemblyResult:
    """Result of context assembly.

    Attributes:
        items: List of context items assembled
        total_tokens: Total tokens in assembled context
        budget_used: Percentage of budget used
        truncated_sources: Sources that were truncated
        failed_sources: Sources that failed to fetch
        assembly_time_ms: Time taken to assemble in milliseconds
        warnings: Any warnings during assembly
    """
    items: list[ContextItem] = field(default_factory=list)
    total_tokens: int = 0
    budget_used: float = 0.0
    truncated_sources: list[ContextSource] = field(default_factory=list)
    failed_sources: list[ContextSource] = field(default_factory=list)
    assembly_time_ms: float = 0.0
    warnings: list[str] = field(default_factory=list)

    def get_by_source(self, source: ContextSource) -> Optional[ContextItem]:
        """Get context item by source."""
        for item in self.items:
            if item.source == source:
                return item
        return None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "items": [
                {
                    "source": item.source.value,
                    "tokens": item.tokens,
                    "priority": item.priority,
                }
                for item in self.items
            ],
            "total_tokens": self.total_tokens,
            "budget_used": self.budget_used,
            "truncated_sources": [s.value for s in self.truncated_sources],
            "failed_sources": [s.value for s in self.failed_sources],
            "assembly_time_ms": self.assembly_time_ms,
            "warnings": self.warnings,
        }

    def to_context_string(self) -> str:
        """Build the complete context string."""
        parts = []
        for item in self.items:
            if item.content:
                parts.append(f"=== {item.source.value.upper()} ===")
                parts.append(item.to_string())
                parts.append("")
        return "\n".join(parts)

    def to_messages_format(self) -> list[dict[str, str]]:
        """Convert to OpenAI-style messages format."""
        messages = []

        # System prompt goes first
        system_item = self.get_by_source(ContextSource.SYSTEM_PROMPT)
        if system_item:
            messages.append({
                "role": "system",
                "content": system_item.to_string(),
            })

        # Conversation history
        history_item = self.get_by_source(ContextSource.CONVERSATION_HISTORY)
        if history_item and isinstance(history_item.content, list):
            for msg in history_item.content:
                if isinstance(msg, dict) and "role" in msg and "content" in msg:
                    messages.append(msg)

        # Current task as user message
        task_item = self.get_by_source(ContextSource.TASK)
        if task_item:
            # Build context for the task
            context_parts = []

            # Add memory context
            memory_item = self.get_by_source(ContextSource.LONG_TERM_MEMORY)
            if memory_item:
                context_parts.append("Relevant context from memory:")
                context_parts.append(memory_item.to_string())

            # Add user context
            user_ctx_item = self.get_by_source(ContextSource.USER_CONTEXT)
            if user_ctx_item:
                context_parts.append("Additional context:")
                context_parts.append(user_ctx_item.to_string())

            # Add task
            context_parts.append("Current task:")
            context_parts.append(task_item.to_string())

            messages.append({
                "role": "user",
                "content": "\n\n".join(context_parts),
            })

        return messages


# =============================================================================
# Token Budget Tracker
# =============================================================================


@dataclass
class ContextBudget:
    """Tracks token budget allocation for context assembly.

    Attributes:
        max_tokens: Maximum total tokens available
        allocated: Tokens allocated per source
        used: Tokens actually used per source
    """
    max_tokens: int
    allocated: dict[ContextSource, int] = field(default_factory=dict)
    used: dict[ContextSource, int] = field(default_factory=dict)

    @property
    def total_allocated(self) -> int:
        """Get total allocated tokens."""
        return sum(self.allocated.values())

    @property
    def total_used(self) -> int:
        """Get total used tokens."""
        return sum(self.used.values())

    @property
    def remaining(self) -> int:
        """Get remaining tokens."""
        return self.max_tokens - self.total_used

    def allocate(self, source: ContextSource, tokens: int) -> None:
        """Allocate tokens to a source."""
        self.allocated[source] = tokens
        self.used[source] = 0

    def use(self, source: ContextSource, tokens: int) -> int:
        """Use tokens from a source, returns actual tokens used."""
        max_for_source = self.allocated.get(source, 0)
        actual = min(tokens, max_for_source, self.remaining)
        self.used[source] = actual
        return actual

    def get_remaining_for(self, source: ContextSource) -> int:
        """Get remaining tokens for a specific source."""
        allocated = self.allocated.get(source, 0)
        used = self.used.get(source, 0)
        return min(allocated - used, self.remaining)


# =============================================================================
# Context Provider Protocol
# =============================================================================

T = TypeVar("T")


class ContextProvider:
    """Base class for context providers.

    Subclasses implement fetch() to retrieve context from specific sources.
    """

    def __init__(self, source: ContextSource) -> None:
        self.source = source

    async def fetch(
        self,
        query: str,
        budget_tokens: int,
        **kwargs: Any,
    ) -> Optional[ContextItem]:
        """Fetch context from this source.

        Args:
            query: Query string for context retrieval
            budget_tokens: Maximum tokens to return
            **kwargs: Additional parameters

        Returns:
            ContextItem or None if no context available
        """
        raise NotImplementedError


# =============================================================================
# Built-in Context Providers
# =============================================================================


class MemoryContextProvider(ContextProvider):
    """Provides context from memory system."""

    def __init__(self, memory_manager: Any) -> None:
        super().__init__(ContextSource.LONG_TERM_MEMORY)
        self._memory_manager = memory_manager

    async def fetch(
        self,
        query: str,
        budget_tokens: int,
        **kwargs: Any,
    ) -> Optional[ContextItem]:
        """Fetch relevant memories."""
        if not self._memory_manager:
            return None

        try:
            # Estimate how many items we can fit
            avg_item_tokens = 100  # Rough estimate
            max_items = max(1, budget_tokens // avg_item_tokens)

            memories = await self._memory_manager.retrieve(
                query=query,
                k=max_items,
            )

            if not memories:
                return None

            # Format memories
            formatted = []
            total_tokens = 0
            for mem in memories:
                mem_text = f"- {mem.content}"
                if mem.category:
                    mem_text += f" [{mem.category}]"
                mem_tokens = len(mem_text) // CHARS_PER_TOKEN
                if total_tokens + mem_tokens > budget_tokens:
                    break
                formatted.append(mem_text)
                total_tokens += mem_tokens

            if not formatted:
                return None

            return ContextItem(
                source=self.source,
                content="\n".join(formatted),
                tokens=total_tokens,
                priority=0.8,
                metadata={"item_count": len(formatted)},
            )

        except Exception as e:
            logger.warning(f"Memory context fetch failed: {e}")
            return None


class ConversationContextProvider(ContextProvider):
    """Provides context from conversation history."""

    def __init__(self, get_history: Optional[Callable[[str], list[dict]]] = None) -> None:
        super().__init__(ContextSource.CONVERSATION_HISTORY)
        self._get_history = get_history

    async def fetch(
        self,
        query: str,
        budget_tokens: int,
        session_id: Optional[str] = None,
        conversation: Optional[list[dict]] = None,
        **kwargs: Any,
    ) -> Optional[ContextItem]:
        """Fetch conversation history."""
        # Use provided conversation or fetch from callback
        history = conversation
        if history is None and self._get_history and session_id:
            try:
                history = self._get_history(session_id)
            except Exception as e:
                logger.warning(f"Failed to fetch conversation history: {e}")
                return None

        if not history:
            return None

        # Select messages within budget, prioritizing recent
        selected = []
        total_tokens = 0

        for msg in reversed(history):
            content = msg.get("content", "")
            msg_tokens = len(content) // CHARS_PER_TOKEN + 10  # +10 for role/formatting
            if total_tokens + msg_tokens > budget_tokens:
                break
            selected.insert(0, msg)
            total_tokens += msg_tokens

        if not selected:
            return None

        return ContextItem(
            source=self.source,
            content=selected,
            tokens=total_tokens,
            priority=0.7,
            metadata={"message_count": len(selected)},
        )


# =============================================================================
# Context Manager
# =============================================================================


class ContextManager:
    """Assembles optimized context windows for agent invocations.

    The ContextManager fetches context from multiple sources in parallel,
    manages token budgets, and assembles the final context window.

    Features:
        - Parallel context assembly from multiple sources
        - Token budget tracking and enforcement
        - Priority-based context selection
        - Compression when context exceeds budget
        - Error handling for individual source failures

    Attributes:
        max_tokens: Maximum tokens for context window
        reserved_for_response: Tokens reserved for response
        settings: Framework settings

    Example:
        >>> manager = ContextManager(max_tokens=128000)
        >>> result = await manager.assemble(
        ...     task="Analyze market trends",
        ...     session_id="sess-123",
        ... )
    """

    def __init__(
        self,
        max_tokens: int = DEFAULT_MAX_CONTEXT_TOKENS,
        reserved_for_response: int = DEFAULT_RESERVED_FOR_RESPONSE,
        memory_manager: Optional[Any] = None,
        settings: Optional[SigilSettings] = None,
        compressor: Optional[Any] = None,
    ) -> None:
        """Initialize the ContextManager.

        Args:
            max_tokens: Maximum context window tokens
            reserved_for_response: Tokens to reserve for response
            memory_manager: Optional memory manager instance
            settings: Optional settings instance
            compressor: Optional context compressor instance
        """
        self._max_tokens = max_tokens
        self._reserved_for_response = reserved_for_response
        self._settings = settings or get_settings()
        self._compressor = compressor

        # Initialize providers
        self._providers: dict[ContextSource, ContextProvider] = {}

        if memory_manager:
            self._providers[ContextSource.LONG_TERM_MEMORY] = MemoryContextProvider(
                memory_manager
            )

        self._providers[ContextSource.CONVERSATION_HISTORY] = ConversationContextProvider()

        # Metrics
        self._total_assemblies: int = 0
        self._total_tokens_assembled: int = 0
        self._total_assembly_time_ms: float = 0.0

    @property
    def available_tokens(self) -> int:
        """Get available tokens (max - reserved)."""
        return self._max_tokens - self._reserved_for_response

    def register_provider(
        self,
        source: ContextSource,
        provider: ContextProvider,
    ) -> None:
        """Register a context provider.

        Args:
            source: The context source
            provider: The provider instance
        """
        self._providers[source] = provider
        logger.debug(f"Registered provider for {source.value}")

    async def assemble(
        self,
        task: str,
        session_id: str,
        user_id: Optional[str] = None,
        system_prompt: Optional[str] = None,
        conversation: Optional[list[dict]] = None,
        user_context: Optional[dict[str, Any]] = None,
        plan: Optional[Any] = None,
        tool_results: Optional[list[dict]] = None,
        token_budget: Optional[int] = None,
    ) -> ContextAssemblyResult:
        """Assemble context from all sources.

        Fetches context from multiple sources in parallel, manages
        token budgets, and assembles the final context.

        Args:
            task: The current task
            session_id: Session identifier
            user_id: Optional user identifier
            system_prompt: Optional system prompt override
            conversation: Optional conversation history
            user_context: Optional user-provided context
            plan: Optional execution plan
            tool_results: Optional tool execution results
            token_budget: Optional token budget override

        Returns:
            ContextAssemblyResult with assembled context

        Example:
            >>> result = await manager.assemble(
            ...     task="Research competitor pricing",
            ...     session_id="sess-123",
            ...     system_prompt="You are a research analyst.",
            ... )
        """
        start_time = time.perf_counter()
        budget_tokens = token_budget or self.available_tokens

        # Initialize budget
        budget = self._allocate_budget(budget_tokens)

        # Build result
        result = ContextAssemblyResult()

        # Step 1: Add system prompt (always first, synchronous)
        if system_prompt:
            tokens = len(system_prompt) // CHARS_PER_TOKEN
            actual_tokens = budget.use(ContextSource.SYSTEM_PROMPT, tokens)
            result.items.append(ContextItem(
                source=ContextSource.SYSTEM_PROMPT,
                content=system_prompt,
                tokens=actual_tokens,
                priority=1.0,
            ))
            result.total_tokens += actual_tokens

        # Step 2: Add task (always included)
        task_tokens = len(task) // CHARS_PER_TOKEN
        actual_task_tokens = budget.use(ContextSource.TASK, task_tokens)
        result.items.append(ContextItem(
            source=ContextSource.TASK,
            content=task,
            tokens=actual_task_tokens,
            priority=0.95,
        ))
        result.total_tokens += actual_task_tokens

        # Step 3: Add user context if provided
        if user_context:
            import json
            ctx_str = json.dumps(user_context, indent=2)
            ctx_tokens = len(ctx_str) // CHARS_PER_TOKEN
            actual_ctx_tokens = budget.use(ContextSource.USER_CONTEXT, ctx_tokens)
            result.items.append(ContextItem(
                source=ContextSource.USER_CONTEXT,
                content=user_context,
                tokens=actual_ctx_tokens,
                priority=0.85,
            ))
            result.total_tokens += actual_ctx_tokens

        # Step 4: Add plan if provided
        if plan:
            plan_str = str(plan)
            plan_tokens = len(plan_str) // CHARS_PER_TOKEN
            actual_plan_tokens = budget.use(ContextSource.PLAN, plan_tokens)
            result.items.append(ContextItem(
                source=ContextSource.PLAN,
                content=plan,
                tokens=actual_plan_tokens,
                priority=0.9,
            ))
            result.total_tokens += actual_plan_tokens

        # Step 5: Fetch from async providers in parallel
        async_tasks = []

        # Memory provider
        if ContextSource.LONG_TERM_MEMORY in self._providers:
            async_tasks.append(
                self._fetch_with_timeout(
                    self._providers[ContextSource.LONG_TERM_MEMORY],
                    task,
                    budget.get_remaining_for(ContextSource.LONG_TERM_MEMORY),
                    session_id=session_id,
                )
            )

        # Conversation provider
        if ContextSource.CONVERSATION_HISTORY in self._providers:
            async_tasks.append(
                self._fetch_with_timeout(
                    self._providers[ContextSource.CONVERSATION_HISTORY],
                    task,
                    budget.get_remaining_for(ContextSource.CONVERSATION_HISTORY),
                    session_id=session_id,
                    conversation=conversation,
                )
            )

        # Execute parallel fetches
        if async_tasks:
            fetch_results = await asyncio.gather(*async_tasks, return_exceptions=True)

            for fetch_result in fetch_results:
                if isinstance(fetch_result, Exception):
                    logger.warning(f"Context fetch failed: {fetch_result}")
                    continue

                if fetch_result is not None:
                    # Use tokens from budget
                    actual_tokens = budget.use(
                        fetch_result.source,
                        fetch_result.tokens,
                    )
                    fetch_result.tokens = actual_tokens
                    result.items.append(fetch_result)
                    result.total_tokens += actual_tokens

        # Step 6: Add tool results if provided
        if tool_results:
            import json
            tool_str = json.dumps(tool_results, indent=2)
            tool_tokens = len(tool_str) // CHARS_PER_TOKEN
            actual_tool_tokens = budget.use(ContextSource.TOOL_RESULTS, tool_tokens)
            result.items.append(ContextItem(
                source=ContextSource.TOOL_RESULTS,
                content=tool_results,
                tokens=actual_tool_tokens,
                priority=0.75,
            ))
            result.total_tokens += actual_tool_tokens

        # Step 7: Apply compression if over budget
        if result.total_tokens > budget_tokens and self._compressor:
            try:
                result = await self._compress_context(result, budget_tokens)
            except Exception as e:
                result.warnings.append(f"Compression failed: {e}")

        # Calculate final metrics
        result.budget_used = result.total_tokens / budget_tokens if budget_tokens > 0 else 0
        result.assembly_time_ms = (time.perf_counter() - start_time) * 1000

        # Update metrics
        self._total_assemblies += 1
        self._total_tokens_assembled += result.total_tokens
        self._total_assembly_time_ms += result.assembly_time_ms

        logger.debug(
            f"Assembled context: {result.total_tokens} tokens, "
            f"{result.budget_used:.1%} budget, "
            f"{result.assembly_time_ms:.1f}ms"
        )

        return result

    def _allocate_budget(self, total_tokens: int) -> ContextBudget:
        """Allocate token budget to sources.

        Args:
            total_tokens: Total tokens available

        Returns:
            ContextBudget with allocations
        """
        budget = ContextBudget(max_tokens=total_tokens)

        # Allocate by ratio
        budget.allocate(
            ContextSource.SYSTEM_PROMPT,
            int(total_tokens * SYSTEM_PROMPT_BUDGET_RATIO),
        )
        budget.allocate(
            ContextSource.TASK,
            int(total_tokens * TASK_BUDGET_RATIO),
        )
        budget.allocate(
            ContextSource.WORKING_MEMORY,
            int(total_tokens * WORKING_MEMORY_BUDGET_RATIO),
        )
        budget.allocate(
            ContextSource.LONG_TERM_MEMORY,
            int(total_tokens * LONG_TERM_MEMORY_BUDGET_RATIO),
        )
        budget.allocate(
            ContextSource.CONVERSATION_HISTORY,
            int(total_tokens * CONVERSATION_HISTORY_BUDGET_RATIO),
        )
        budget.allocate(
            ContextSource.TOOL_RESULTS,
            int(total_tokens * TOOL_RESULTS_BUDGET_RATIO),
        )
        budget.allocate(
            ContextSource.PLAN,
            int(total_tokens * 0.05),  # 5% for plan
        )
        budget.allocate(
            ContextSource.USER_CONTEXT,
            int(total_tokens * 0.05),  # 5% for user context
        )

        return budget

    async def _fetch_with_timeout(
        self,
        provider: ContextProvider,
        query: str,
        budget_tokens: int,
        timeout: float = 5.0,
        **kwargs: Any,
    ) -> Optional[ContextItem]:
        """Fetch from provider with timeout.

        Args:
            provider: The context provider
            query: Query string
            budget_tokens: Token budget
            timeout: Timeout in seconds
            **kwargs: Additional arguments

        Returns:
            ContextItem or None
        """
        try:
            return await asyncio.wait_for(
                provider.fetch(query, budget_tokens, **kwargs),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            logger.warning(f"Provider {provider.source.value} timed out")
            return None
        except Exception as e:
            logger.warning(f"Provider {provider.source.value} failed: {e}")
            return None

    async def _compress_context(
        self,
        result: ContextAssemblyResult,
        target_tokens: int,
    ) -> ContextAssemblyResult:
        """Compress context to fit within budget.

        Args:
            result: Current assembly result
            target_tokens: Target token count

        Returns:
            Compressed result
        """
        if not self._compressor:
            return result

        # Use compressor to reduce context
        overflow_ratio = result.total_tokens / target_tokens

        for item in result.items:
            # Skip system prompt and task (critical)
            if item.source in (ContextSource.SYSTEM_PROMPT, ContextSource.TASK):
                continue

            # Compress item
            compressed = await self._compressor.compress_item(
                item,
                target_tokens=int(item.tokens / overflow_ratio),
            )

            if compressed:
                result.truncated_sources.append(item.source)
                item.content = compressed.content
                item.tokens = compressed.tokens

        # Recalculate total
        result.total_tokens = sum(item.tokens for item in result.items)

        return result

    def estimate_tokens(self, text: str) -> int:
        """Estimate token count for text.

        Args:
            text: Text to estimate

        Returns:
            Estimated token count
        """
        return len(text) // CHARS_PER_TOKEN

    def get_metrics(self) -> dict[str, Any]:
        """Get context manager metrics.

        Returns:
            Dictionary with metrics
        """
        return {
            "total_assemblies": self._total_assemblies,
            "total_tokens_assembled": self._total_tokens_assembled,
            "total_assembly_time_ms": self._total_assembly_time_ms,
            "avg_tokens_per_assembly": (
                self._total_tokens_assembled / max(self._total_assemblies, 1)
            ),
            "avg_assembly_time_ms": (
                self._total_assembly_time_ms / max(self._total_assemblies, 1)
            ),
            "max_tokens": self._max_tokens,
            "reserved_for_response": self._reserved_for_response,
            "registered_providers": list(self._providers.keys()),
        }

    def reset_metrics(self) -> None:
        """Reset metrics."""
        self._total_assemblies = 0
        self._total_tokens_assembled = 0
        self._total_assembly_time_ms = 0.0


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Enums
    "ContextSource",
    # Data classes
    "ContextItem",
    "ContextAssemblyResult",
    "ContextBudget",
    # Provider base
    "ContextProvider",
    "MemoryContextProvider",
    "ConversationContextProvider",
    # Main class
    "ContextManager",
    # Constants
    "DEFAULT_MAX_CONTEXT_TOKENS",
    "DEFAULT_RESERVED_FOR_RESPONSE",
    "CHARS_PER_TOKEN",
]
