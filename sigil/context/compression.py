"""ContextCompressor - Context compression strategies for Sigil v2.

This module implements compression strategies for reducing context size
when it exceeds the token budget.

Compression Strategies:
    - TRUNCATE: Remove oldest messages/content
    - SUMMARIZE: Use LLM to summarize content
    - PRIORITIZE: Keep only highest priority items

Strategy Selection:
    - overflow_ratio < 1.2: TRUNCATE (minimal overflow)
    - overflow_ratio < 1.5 and complexity < 0.5: TRUNCATE
    - overflow_ratio < 2.0: SUMMARIZE
    - overflow_ratio >= 2.0: PRIORITIZE

Example:
    >>> from sigil.context.compression import ContextCompressor, CompressionStrategy
    >>>
    >>> compressor = ContextCompressor()
    >>> result = await compressor.compress(
    ...     items=context_items,
    ...     target_tokens=10000,
    ...     strategy=CompressionStrategy.SUMMARIZE,
    ... )
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Any, Optional, Callable, Awaitable

from sigil.config import get_settings
from sigil.config.settings import SigilSettings


# =============================================================================
# Module Logger
# =============================================================================

logger = logging.getLogger(__name__)


# =============================================================================
# Constants
# =============================================================================

CHARS_PER_TOKEN = 4  # Simple estimation

# Cache settings
COMPRESSION_CACHE_TTL_SECONDS = 3600  # 1 hour
COMPRESSION_CACHE_MAX_SIZE = 1000


# =============================================================================
# Compression Strategy Enum
# =============================================================================


class CompressionStrategy(str, Enum):
    """Compression strategy to apply.

    Attributes:
        TRUNCATE: Remove oldest content to fit budget
        SUMMARIZE: Use LLM to create summary of content
        PRIORITIZE: Keep only highest priority items
        HYBRID: Combine strategies based on content type
    """
    TRUNCATE = "truncate"
    SUMMARIZE = "summarize"
    PRIORITIZE = "prioritize"
    HYBRID = "hybrid"


# =============================================================================
# Compression Result
# =============================================================================


@dataclass
class CompressionResult:
    """Result of a compression operation.

    Attributes:
        original_tokens: Token count before compression
        compressed_tokens: Token count after compression
        strategy_used: Strategy that was applied
        content: Compressed content
        compression_ratio: Ratio of compression (original/compressed)
        time_ms: Time taken for compression
        cache_hit: Whether result was from cache
        metadata: Additional metadata
    """
    original_tokens: int
    compressed_tokens: int
    strategy_used: CompressionStrategy
    content: Any
    compression_ratio: float = 1.0
    time_ms: float = 0.0
    cache_hit: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Calculate compression ratio."""
        if self.compressed_tokens > 0:
            self.compression_ratio = self.original_tokens / self.compressed_tokens
        else:
            self.compression_ratio = float("inf")


# =============================================================================
# Cache Entry
# =============================================================================


@dataclass
class CacheEntry:
    """Cache entry for compression results.

    Attributes:
        key: Cache key (hash of original content)
        result: Compressed result
        created_at: When the entry was created
        ttl_seconds: Time-to-live in seconds
    """
    key: str
    result: CompressionResult
    created_at: datetime
    ttl_seconds: int = COMPRESSION_CACHE_TTL_SECONDS

    @property
    def is_expired(self) -> bool:
        """Check if cache entry has expired."""
        expiry = self.created_at + timedelta(seconds=self.ttl_seconds)
        return datetime.now(timezone.utc) > expiry


# =============================================================================
# Compression Strategies
# =============================================================================


class TruncateStrategy:
    """Truncation-based compression strategy.

    Removes content from the end (or beginning for conversation history)
    to fit within the target token budget.
    """

    def compress(
        self,
        content: Any,
        original_tokens: int,
        target_tokens: int,
        keep_start: bool = True,
    ) -> CompressionResult:
        """Truncate content to fit target tokens.

        Args:
            content: Content to compress
            original_tokens: Original token count
            target_tokens: Target token count
            keep_start: If True, keep start and truncate end

        Returns:
            CompressionResult with truncated content
        """
        start_time = time.perf_counter()

        if original_tokens <= target_tokens:
            return CompressionResult(
                original_tokens=original_tokens,
                compressed_tokens=original_tokens,
                strategy_used=CompressionStrategy.TRUNCATE,
                content=content,
            )

        if isinstance(content, str):
            # Truncate string
            target_chars = target_tokens * CHARS_PER_TOKEN
            if keep_start:
                truncated = content[:target_chars]
                if len(content) > target_chars:
                    truncated = truncated[:-3] + "..."
            else:
                truncated = content[-target_chars:]
                if len(content) > target_chars:
                    truncated = "..." + truncated[3:]

            compressed_tokens = len(truncated) // CHARS_PER_TOKEN

        elif isinstance(content, list):
            # Truncate list (keep most recent for conversations)
            ratio = target_tokens / original_tokens
            keep_count = max(1, int(len(content) * ratio))

            if keep_start:
                truncated = content[:keep_count]
            else:
                truncated = content[-keep_count:]

            # Estimate tokens for truncated list
            compressed_tokens = sum(
                len(str(item)) // CHARS_PER_TOKEN for item in truncated
            )

        elif isinstance(content, dict):
            # Truncate dict values
            truncated = {}
            remaining_tokens = target_tokens
            for key, value in content.items():
                value_str = str(value)
                value_tokens = len(value_str) // CHARS_PER_TOKEN
                if remaining_tokens <= 0:
                    break
                if value_tokens <= remaining_tokens:
                    truncated[key] = value
                    remaining_tokens -= value_tokens
                else:
                    # Truncate this value
                    truncated[key] = value_str[:remaining_tokens * CHARS_PER_TOKEN]
                    remaining_tokens = 0

            compressed_tokens = sum(
                len(str(v)) // CHARS_PER_TOKEN for v in truncated.values()
            )

        else:
            # Unknown type, convert to string and truncate
            content_str = str(content)
            target_chars = target_tokens * CHARS_PER_TOKEN
            truncated = content_str[:target_chars]
            compressed_tokens = len(truncated) // CHARS_PER_TOKEN

        time_ms = (time.perf_counter() - start_time) * 1000

        return CompressionResult(
            original_tokens=original_tokens,
            compressed_tokens=compressed_tokens,
            strategy_used=CompressionStrategy.TRUNCATE,
            content=truncated,
            time_ms=time_ms,
        )


class SummarizeStrategy:
    """LLM-based summarization strategy.

    Uses an LLM to create a summary of the content that fits
    within the target token budget.
    """

    def __init__(
        self,
        llm_call: Optional[Callable[[str], Awaitable[str]]] = None,
    ) -> None:
        """Initialize summarize strategy.

        Args:
            llm_call: Async function to call LLM for summarization
        """
        self._llm_call = llm_call

    async def compress(
        self,
        content: Any,
        original_tokens: int,
        target_tokens: int,
    ) -> CompressionResult:
        """Summarize content to fit target tokens.

        Args:
            content: Content to compress
            original_tokens: Original token count
            target_tokens: Target token count

        Returns:
            CompressionResult with summarized content
        """
        start_time = time.perf_counter()

        if original_tokens <= target_tokens:
            return CompressionResult(
                original_tokens=original_tokens,
                compressed_tokens=original_tokens,
                strategy_used=CompressionStrategy.SUMMARIZE,
                content=content,
            )

        # Convert content to string if needed
        if isinstance(content, str):
            content_str = content
        elif isinstance(content, list):
            content_str = "\n".join(str(item) for item in content)
        elif isinstance(content, dict):
            content_str = json.dumps(content, indent=2)
        else:
            content_str = str(content)

        # Build summarization prompt
        target_words = target_tokens * 3 // 4  # Rough estimate
        prompt = f"""Summarize the following content in approximately {target_words} words.
Preserve the key information and main points.

Content:
{content_str}

Summary:"""

        # Call LLM if available
        if self._llm_call:
            try:
                summary = await self._llm_call(prompt)
                compressed_tokens = len(summary) // CHARS_PER_TOKEN
            except Exception as e:
                logger.warning(f"LLM summarization failed: {e}")
                # Fall back to truncation
                return TruncateStrategy().compress(
                    content, original_tokens, target_tokens
                )
        else:
            # Without LLM, fall back to smart truncation with marker
            summary = self._extractive_summary(content_str, target_tokens)
            compressed_tokens = len(summary) // CHARS_PER_TOKEN

        time_ms = (time.perf_counter() - start_time) * 1000

        return CompressionResult(
            original_tokens=original_tokens,
            compressed_tokens=compressed_tokens,
            strategy_used=CompressionStrategy.SUMMARIZE,
            content=summary,
            time_ms=time_ms,
        )

    def _extractive_summary(self, text: str, target_tokens: int) -> str:
        """Create extractive summary without LLM.

        Simple heuristic: keep first and last portions with marker.

        Args:
            text: Text to summarize
            target_tokens: Target token count

        Returns:
            Extractive summary
        """
        target_chars = target_tokens * CHARS_PER_TOKEN
        if len(text) <= target_chars:
            return text

        # Keep 40% from start, 40% from end, 20% for marker
        start_chars = int(target_chars * 0.4)
        end_chars = int(target_chars * 0.4)
        marker = "\n\n[... content truncated ...]\n\n"

        start_part = text[:start_chars]
        end_part = text[-end_chars:]

        return f"{start_part}{marker}{end_part}"


class PrioritizeStrategy:
    """Priority-based compression strategy.

    Keeps only the highest priority items to fit within
    the target token budget.
    """

    def compress(
        self,
        items: list[Any],
        original_tokens: int,
        target_tokens: int,
        get_priority: Optional[Callable[[Any], float]] = None,
        get_tokens: Optional[Callable[[Any], int]] = None,
    ) -> CompressionResult:
        """Keep highest priority items within budget.

        Args:
            items: List of items to prioritize
            original_tokens: Original total token count
            target_tokens: Target token count
            get_priority: Function to get priority of item (default: 1.0)
            get_tokens: Function to get token count of item

        Returns:
            CompressionResult with prioritized items
        """
        start_time = time.perf_counter()

        if original_tokens <= target_tokens:
            return CompressionResult(
                original_tokens=original_tokens,
                compressed_tokens=original_tokens,
                strategy_used=CompressionStrategy.PRIORITIZE,
                content=items,
            )

        # Default functions
        if get_priority is None:
            def get_priority(item: Any) -> float:
                if hasattr(item, "priority"):
                    return item.priority
                return 1.0

        if get_tokens is None:
            def get_tokens(item: Any) -> int:
                if hasattr(item, "tokens"):
                    return item.tokens
                return len(str(item)) // CHARS_PER_TOKEN

        # Sort by priority (highest first)
        sorted_items = sorted(items, key=get_priority, reverse=True)

        # Select items within budget
        selected = []
        remaining_tokens = target_tokens

        for item in sorted_items:
            item_tokens = get_tokens(item)
            if item_tokens <= remaining_tokens:
                selected.append(item)
                remaining_tokens -= item_tokens

        compressed_tokens = target_tokens - remaining_tokens
        time_ms = (time.perf_counter() - start_time) * 1000

        return CompressionResult(
            original_tokens=original_tokens,
            compressed_tokens=compressed_tokens,
            strategy_used=CompressionStrategy.PRIORITIZE,
            content=selected,
            time_ms=time_ms,
            metadata={"items_kept": len(selected), "items_total": len(items)},
        )


# =============================================================================
# Context Compressor
# =============================================================================


class ContextCompressor:
    """Context compression manager.

    Manages compression strategy selection and caching to reduce
    context size when it exceeds the token budget.

    Features:
        - Multiple compression strategies
        - Automatic strategy selection
        - Compression result caching
        - Metrics tracking

    Example:
        >>> compressor = ContextCompressor()
        >>> result = await compressor.compress(
        ...     content="Long text content...",
        ...     original_tokens=5000,
        ...     target_tokens=2000,
        ... )
    """

    def __init__(
        self,
        llm_call: Optional[Callable[[str], Awaitable[str]]] = None,
        cache_enabled: bool = True,
        cache_ttl: int = COMPRESSION_CACHE_TTL_SECONDS,
        settings: Optional[SigilSettings] = None,
    ) -> None:
        """Initialize the compressor.

        Args:
            llm_call: Optional async function for LLM summarization
            cache_enabled: Whether to cache compression results
            cache_ttl: Cache time-to-live in seconds
            settings: Optional settings instance
        """
        self._settings = settings or get_settings()

        # Initialize strategies
        self._truncate = TruncateStrategy()
        self._summarize = SummarizeStrategy(llm_call=llm_call)
        self._prioritize = PrioritizeStrategy()

        # Cache
        self._cache_enabled = cache_enabled
        self._cache_ttl = cache_ttl
        self._cache: dict[str, CacheEntry] = {}

        # Metrics
        self._total_compressions: int = 0
        self._cache_hits: int = 0
        self._total_tokens_saved: int = 0
        self._strategy_usage: dict[CompressionStrategy, int] = {
            s: 0 for s in CompressionStrategy
        }

    def select_strategy(
        self,
        overflow_ratio: float,
        complexity: float = 0.5,
        content_type: str = "text",
    ) -> CompressionStrategy:
        """Select compression strategy based on context.

        Strategy selection logic:
        - overflow_ratio < 1.2: TRUNCATE (minimal overflow)
        - overflow_ratio < 1.5 and complexity < 0.5: TRUNCATE
        - overflow_ratio < 2.0: SUMMARIZE
        - overflow_ratio >= 2.0: PRIORITIZE

        Args:
            overflow_ratio: Ratio of current tokens to target
            complexity: Task complexity (0.0-1.0)
            content_type: Type of content ("text", "list", "dict")

        Returns:
            Selected CompressionStrategy
        """
        if overflow_ratio <= 1.0:
            return CompressionStrategy.TRUNCATE

        if overflow_ratio < 1.2:
            return CompressionStrategy.TRUNCATE

        if overflow_ratio < 1.5 and complexity < 0.5:
            return CompressionStrategy.TRUNCATE

        if overflow_ratio < 2.0:
            return CompressionStrategy.SUMMARIZE

        if content_type == "list":
            return CompressionStrategy.PRIORITIZE

        return CompressionStrategy.HYBRID

    async def compress(
        self,
        content: Any,
        original_tokens: int,
        target_tokens: int,
        strategy: Optional[CompressionStrategy] = None,
        complexity: float = 0.5,
        use_cache: bool = True,
    ) -> CompressionResult:
        """Compress content to fit within target tokens.

        Args:
            content: Content to compress
            original_tokens: Original token count
            target_tokens: Target token count
            strategy: Optional strategy override
            complexity: Task complexity for strategy selection
            use_cache: Whether to use cache

        Returns:
            CompressionResult with compressed content
        """
        # Check if compression needed
        if original_tokens <= target_tokens:
            return CompressionResult(
                original_tokens=original_tokens,
                compressed_tokens=original_tokens,
                strategy_used=CompressionStrategy.TRUNCATE,
                content=content,
            )

        # Check cache
        cache_key = None
        if self._cache_enabled and use_cache:
            cache_key = self._compute_cache_key(content, target_tokens)
            cached = self._get_from_cache(cache_key)
            if cached:
                self._cache_hits += 1
                cached.cache_hit = True
                return cached

        # Select strategy if not provided
        if strategy is None:
            overflow_ratio = original_tokens / target_tokens
            content_type = "list" if isinstance(content, list) else "text"
            strategy = self.select_strategy(overflow_ratio, complexity, content_type)

        # Apply strategy
        result: CompressionResult

        if strategy == CompressionStrategy.TRUNCATE:
            result = self._truncate.compress(
                content, original_tokens, target_tokens
            )

        elif strategy == CompressionStrategy.SUMMARIZE:
            result = await self._summarize.compress(
                content, original_tokens, target_tokens
            )

        elif strategy == CompressionStrategy.PRIORITIZE:
            if isinstance(content, list):
                result = self._prioritize.compress(
                    content, original_tokens, target_tokens
                )
            else:
                # Fall back to truncate for non-lists
                result = self._truncate.compress(
                    content, original_tokens, target_tokens
                )

        elif strategy == CompressionStrategy.HYBRID:
            result = await self._hybrid_compress(
                content, original_tokens, target_tokens
            )

        else:
            result = self._truncate.compress(
                content, original_tokens, target_tokens
            )

        # Update metrics
        self._total_compressions += 1
        self._total_tokens_saved += original_tokens - result.compressed_tokens
        self._strategy_usage[result.strategy_used] += 1

        # Store in cache
        if self._cache_enabled and cache_key:
            self._store_in_cache(cache_key, result)

        return result

    async def compress_item(
        self,
        item: Any,
        target_tokens: int,
    ) -> Optional[Any]:
        """Compress a single context item.

        Convenience method for compressing individual items.

        Args:
            item: Item with content and tokens attributes
            target_tokens: Target token count

        Returns:
            Compressed item or None if compression fails
        """
        if hasattr(item, "tokens") and hasattr(item, "content"):
            result = await self.compress(
                content=item.content,
                original_tokens=item.tokens,
                target_tokens=target_tokens,
            )

            # Create a copy with compressed content
            from copy import copy
            compressed = copy(item)
            compressed.content = result.content
            compressed.tokens = result.compressed_tokens
            return compressed

        return None

    async def _hybrid_compress(
        self,
        content: Any,
        original_tokens: int,
        target_tokens: int,
    ) -> CompressionResult:
        """Apply hybrid compression strategy.

        Combines prioritization and summarization.

        Args:
            content: Content to compress
            original_tokens: Original token count
            target_tokens: Target token count

        Returns:
            CompressionResult
        """
        # First, try prioritization if content is list
        if isinstance(content, list):
            priority_result = self._prioritize.compress(
                content, original_tokens, target_tokens
            )

            # If still over budget, summarize the remaining
            if priority_result.compressed_tokens > target_tokens:
                return await self._summarize.compress(
                    priority_result.content,
                    priority_result.compressed_tokens,
                    target_tokens,
                )
            return priority_result

        # For non-lists, use summarization
        return await self._summarize.compress(
            content, original_tokens, target_tokens
        )

    def _compute_cache_key(
        self,
        content: Any,
        target_tokens: int,
    ) -> str:
        """Compute cache key for content.

        Args:
            content: Content to hash
            target_tokens: Target token count

        Returns:
            Cache key string
        """
        content_str = json.dumps(content, sort_keys=True, default=str)
        hash_input = f"{content_str}:{target_tokens}"
        return hashlib.sha256(hash_input.encode()).hexdigest()[:32]

    def _get_from_cache(self, key: str) -> Optional[CompressionResult]:
        """Get compression result from cache.

        Args:
            key: Cache key

        Returns:
            Cached result or None
        """
        entry = self._cache.get(key)
        if entry and not entry.is_expired:
            return entry.result
        elif entry:
            # Remove expired entry
            del self._cache[key]
        return None

    def _store_in_cache(
        self,
        key: str,
        result: CompressionResult,
    ) -> None:
        """Store compression result in cache.

        Args:
            key: Cache key
            result: Result to cache
        """
        # Enforce cache size limit
        if len(self._cache) >= COMPRESSION_CACHE_MAX_SIZE:
            # Remove oldest entries
            sorted_entries = sorted(
                self._cache.items(),
                key=lambda x: x[1].created_at,
            )
            for old_key, _ in sorted_entries[: len(sorted_entries) // 2]:
                del self._cache[old_key]

        self._cache[key] = CacheEntry(
            key=key,
            result=result,
            created_at=datetime.now(timezone.utc),
            ttl_seconds=self._cache_ttl,
        )

    def clear_cache(self) -> None:
        """Clear the compression cache."""
        self._cache.clear()
        logger.info("Compression cache cleared")

    def get_metrics(self) -> dict[str, Any]:
        """Get compressor metrics.

        Returns:
            Dictionary with metrics
        """
        return {
            "total_compressions": self._total_compressions,
            "cache_hits": self._cache_hits,
            "cache_hit_rate": (
                self._cache_hits / max(self._total_compressions, 1)
            ),
            "total_tokens_saved": self._total_tokens_saved,
            "avg_tokens_saved": (
                self._total_tokens_saved / max(self._total_compressions, 1)
            ),
            "strategy_usage": {
                s.value: count for s, count in self._strategy_usage.items()
            },
            "cache_size": len(self._cache),
            "cache_enabled": self._cache_enabled,
        }

    def reset_metrics(self) -> None:
        """Reset compressor metrics."""
        self._total_compressions = 0
        self._cache_hits = 0
        self._total_tokens_saved = 0
        self._strategy_usage = {s: 0 for s in CompressionStrategy}


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Enums
    "CompressionStrategy",
    # Data classes
    "CompressionResult",
    "CacheEntry",
    # Strategy classes
    "TruncateStrategy",
    "SummarizeStrategy",
    "PrioritizeStrategy",
    # Main class
    "ContextCompressor",
    # Constants
    "COMPRESSION_CACHE_TTL_SECONDS",
    "COMPRESSION_CACHE_MAX_SIZE",
]
