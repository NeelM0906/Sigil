"""Retrieval system for the Sigil v2 3-layer memory architecture.

This module implements the dual retrieval system with RAG (fast, embedding-based)
and LLM (accurate, reading-based) modes, plus a hybrid mode that starts with
RAG and escalates to LLM if confidence is low.

Classes:
    RetrievalMode: Enum for retrieval modes.
    RAGRetriever: Fast embedding-based retrieval.
    LLMRetriever: Accurate LLM-based retrieval.
    HybridRetriever: Combines RAG and LLM with confidence-based escalation.
    ConfidenceScorer: Scores RAG results based on similarity distribution.

Example:
    >>> from sigil.memory.retrieval import HybridRetriever, RetrievalMode
    >>> retriever = HybridRetriever(item_layer=items, category_layer=categories)
    >>> results = await retriever.retrieve("customer preferences", k=5)
"""

from __future__ import annotations

import json
import re
import statistics
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from sigil.config import get_settings
from sigil.config.schemas.memory import MemoryItem, MemoryCategory
from sigil.core.base import BaseRetriever, RetrievalResult


# =============================================================================
# Enums and Types
# =============================================================================


class RetrievalMode(Enum):
    """Retrieval mode selection."""

    RAG = "rag"  # Fast, embedding-based
    LLM = "llm"  # Accurate, reading-based
    HYBRID = "hybrid"  # Start RAG, escalate to LLM if needed


@dataclass
class ScoredItem:
    """A memory item with a relevance score.

    Attributes:
        item: The memory item.
        score: Relevance score (0.0 to 1.0).
        source: Where this item came from (rag, llm, category).
    """

    item: MemoryItem
    score: float
    source: str = "rag"


@dataclass
class RetrievalContext:
    """Context for a retrieval operation.

    Attributes:
        query: The original query.
        mode: The retrieval mode used.
        items_considered: Number of items considered.
        confidence: Overall confidence in the results.
        escalated: Whether retrieval was escalated from RAG to LLM.
    """

    query: str
    mode: RetrievalMode
    items_considered: int = 0
    confidence: float = 1.0
    escalated: bool = False


# =============================================================================
# Confidence Scorer
# =============================================================================


class ConfidenceScorer:
    """Scores RAG results based on similarity distribution.

    This class analyzes the distribution of similarity scores from RAG
    retrieval to determine confidence in the results. Low confidence
    triggers escalation to LLM in hybrid mode.

    Scoring factors:
    - Top score magnitude (higher = more confident)
    - Score gap between top results and rest
    - Score variance (lower = more ambiguous)
    - Number of highly relevant results

    Example:
        >>> scorer = ConfidenceScorer(threshold=0.7)
        >>> confidence = scorer.score([(item1, 0.9), (item2, 0.5), (item3, 0.3)])
        >>> if confidence < scorer.threshold:
        ...     # Escalate to LLM
    """

    def __init__(
        self,
        threshold: float = 0.7,
        min_top_score: float = 0.5,
        min_gap: float = 0.15,
    ) -> None:
        """Initialize the ConfidenceScorer.

        Args:
            threshold: Confidence threshold for escalation.
            min_top_score: Minimum top score for high confidence.
            min_gap: Minimum gap between top and average for high confidence.
        """
        self.threshold = threshold
        self.min_top_score = min_top_score
        self.min_gap = min_gap

    def score(self, results: list[tuple[MemoryItem, float]]) -> float:
        """Calculate confidence score for retrieval results.

        Args:
            results: List of (item, score) tuples from RAG retrieval.

        Returns:
            Confidence score (0.0 to 1.0).
        """
        if not results:
            return 0.0

        scores = [score for _, score in results]

        # Factor 1: Top score magnitude (0-0.4)
        top_score = scores[0]
        top_factor = min(top_score / self.min_top_score, 1.0) * 0.4

        # Factor 2: Gap between top and rest (0-0.3)
        if len(scores) > 1:
            avg_rest = statistics.mean(scores[1:])
            gap = top_score - avg_rest
            gap_factor = min(gap / self.min_gap, 1.0) * 0.3
        else:
            gap_factor = 0.3  # Only one result, assume good gap

        # Factor 3: Score variance (lower is better, 0-0.2)
        if len(scores) > 1:
            variance = statistics.variance(scores)
            # Lower variance = higher confidence (results are more uniform)
            # But we want high top score with lower rest scores
            # So actually higher variance is better (clear winner)
            variance_factor = min(variance * 5, 1.0) * 0.2
        else:
            variance_factor = 0.2

        # Factor 4: Number of good results (0-0.1)
        good_threshold = top_score * 0.8
        good_count = sum(1 for s in scores if s >= good_threshold)
        count_factor = min(good_count / 3, 1.0) * 0.1

        confidence = top_factor + gap_factor + variance_factor + count_factor

        return min(max(confidence, 0.0), 1.0)

    def should_escalate(self, results: list[tuple[MemoryItem, float]]) -> bool:
        """Determine if retrieval should escalate to LLM.

        Args:
            results: RAG retrieval results.

        Returns:
            True if confidence is below threshold.
        """
        confidence = self.score(results)
        return confidence < self.threshold


# =============================================================================
# RAG Retriever
# =============================================================================


class RAGRetriever(BaseRetriever):
    """Fast embedding-based retrieval using vector similarity.

    This retriever uses FAISS to find semantically similar memory items
    based on embedding vectors. It's fast but may miss nuanced matches.

    Attributes:
        item_layer: The memory item layer.
        category_layer: Optional category layer for context.

    Example:
        >>> retriever = RAGRetriever(item_layer=items)
        >>> results = await retriever.retrieve("billing preferences", k=5)
    """

    def __init__(
        self,
        item_layer: Any,  # ItemLayer - avoid circular import
        category_layer: Optional[Any] = None,  # CategoryLayer
    ) -> None:
        """Initialize the RAGRetriever.

        Args:
            item_layer: The memory item layer.
            category_layer: Optional category layer.
        """
        self._item_layer = item_layer
        self._category_layer = category_layer

    @property
    def retrieval_type(self) -> str:
        """Get the retrieval type."""
        return "rag"

    async def retrieve(
        self,
        query: str,
        k: int = 10,
        category: Optional[str] = None,
        min_score: float = 0.0,
    ) -> list[MemoryItem]:
        """Retrieve relevant items using embedding similarity.

        Args:
            query: The search query.
            k: Maximum number of results.
            category: Optional category filter.
            min_score: Minimum similarity score.

        Returns:
            List of relevant MemoryItems.
        """
        results = self._item_layer.search_by_embedding(
            query=query,
            k=k,
            category=category,
            min_score=min_score,
        )

        return [item for item, _ in results]

    async def retrieve_with_scores(
        self,
        query: str,
        k: int = 10,
        category: Optional[str] = None,
        min_score: float = 0.0,
    ) -> RetrievalResult:
        """Retrieve items with relevance scores.

        Args:
            query: The search query.
            k: Maximum number of results.
            category: Optional category filter.
            min_score: Minimum similarity score.

        Returns:
            RetrievalResult with items and scores.
        """
        import time

        start_time = time.time()

        results = self._item_layer.search_by_embedding(
            query=query,
            k=k,
            category=category,
            min_score=min_score,
        )

        items = [item for item, _ in results]
        scores = [score for _, score in results]

        query_time = (time.time() - start_time) * 1000

        return RetrievalResult(
            items=items,
            scores=scores,
            total_found=len(items),
            query_time_ms=query_time,
        )

    async def retrieve_scored(
        self,
        query: str,
        k: int = 10,
        category: Optional[str] = None,
        min_score: float = 0.0,
    ) -> list[tuple[MemoryItem, float]]:
        """Retrieve items as (item, score) tuples.

        Args:
            query: The search query.
            k: Maximum number of results.
            category: Optional category filter.
            min_score: Minimum similarity score.

        Returns:
            List of (MemoryItem, score) tuples.
        """
        return self._item_layer.search_by_embedding(
            query=query,
            k=k,
            category=category,
            min_score=min_score,
        )


# =============================================================================
# LLM Retriever
# =============================================================================


class LLMRetriever(BaseRetriever):
    """Accurate LLM-based retrieval through reading and reasoning.

    This retriever uses an LLM to read through items and select the most
    relevant ones. It's more accurate for complex queries but slower.

    The LLM considers:
    - Semantic relevance to the query
    - Context and implications
    - Nuanced meaning and intent

    Example:
        >>> retriever = LLMRetriever(item_layer=items)
        >>> results = await retriever.retrieve(
        ...     "What concerns has the customer raised about pricing?",
        ...     k=5
        ... )
    """

    SELECTION_PROMPT = """Given the following memory items, select the {k} most relevant items for the query.

Query: {query}

Available items:
{items_content}

Instructions:
1. Read each item carefully
2. Consider semantic relevance, not just keyword matching
3. Think about what information would help answer the query
4. Select the most relevant items

Respond with a JSON array of item indices (0-based) in order of relevance, plus a relevance score (0.0-1.0).
Example: [{"index": 2, "score": 0.95}, {"index": 0, "score": 0.8}]

Respond ONLY with the JSON array."""

    def __init__(
        self,
        item_layer: Any,  # ItemLayer
        category_layer: Optional[Any] = None,  # CategoryLayer
        max_items_to_consider: int = 50,
    ) -> None:
        """Initialize the LLMRetriever.

        Args:
            item_layer: The memory item layer.
            category_layer: Optional category layer.
            max_items_to_consider: Maximum items to send to LLM.
        """
        self._item_layer = item_layer
        self._category_layer = category_layer
        self._max_items = max_items_to_consider

    @property
    def retrieval_type(self) -> str:
        """Get the retrieval type."""
        return "llm"

    async def retrieve(
        self,
        query: str,
        k: int = 10,
        category: Optional[str] = None,
    ) -> list[MemoryItem]:
        """Retrieve relevant items using LLM selection.

        Args:
            query: The search query.
            k: Maximum number of results.
            category: Optional category filter.

        Returns:
            List of relevant MemoryItems.
        """
        result = await self.retrieve_with_scores(query, k, category)
        return result.items

    async def retrieve_with_scores(
        self,
        query: str,
        k: int = 10,
        category: Optional[str] = None,
    ) -> RetrievalResult:
        """Retrieve items with LLM-assigned relevance scores.

        Args:
            query: The search query.
            k: Maximum number of results.
            category: Optional category filter.

        Returns:
            RetrievalResult with items and scores.
        """
        import time

        start_time = time.time()

        # Get candidate items (use RAG pre-filter for efficiency)
        if category:
            candidates = self._item_layer.get_by_category(
                category, limit=self._max_items
            )
        else:
            # Pre-filter with RAG to get candidates
            rag_results = self._item_layer.search_by_embedding(
                query, k=self._max_items, min_score=0.1
            )
            candidates = [item for item, _ in rag_results]

        if not candidates:
            return RetrievalResult(items=[], total_found=0, query_time_ms=0)

        # Format items for the prompt
        items_content = self._format_items(candidates)

        prompt = self.SELECTION_PROMPT.format(
            k=min(k, len(candidates)),
            query=query,
            items_content=items_content,
        )

        try:
            response = await self._call_llm(prompt)
            selections = self._parse_response(response)
        except Exception:
            # Fall back to returning candidates in order
            selections = [(i, 0.5) for i in range(min(k, len(candidates)))]

        # Build results
        items = []
        scores = []
        for idx, score in selections[:k]:
            if 0 <= idx < len(candidates):
                items.append(candidates[idx])
                scores.append(score)

        query_time = (time.time() - start_time) * 1000

        return RetrievalResult(
            items=items,
            scores=scores,
            total_found=len(items),
            query_time_ms=query_time,
        )

    def _format_items(self, items: list[MemoryItem]) -> str:
        """Format items for the LLM prompt.

        Args:
            items: List of items.

        Returns:
            Formatted string.
        """
        lines = []
        for i, item in enumerate(items):
            category = f" [{item.category}]" if item.category else ""
            lines.append(f"{i}. {item.content}{category}")
        return "\n".join(lines)

    def _parse_response(self, response: str) -> list[tuple[int, float]]:
        """Parse LLM response into selections.

        Args:
            response: LLM response text.

        Returns:
            List of (index, score) tuples.
        """
        try:
            json_match = re.search(r"\[[\s\S]*\]", response)
            if json_match:
                data = json.loads(json_match.group())
            else:
                data = json.loads(response)

            selections = []
            for item in data:
                if isinstance(item, dict) and "index" in item:
                    idx = int(item["index"])
                    score = float(item.get("score", 0.8))
                    selections.append((idx, score))

            return selections

        except (json.JSONDecodeError, KeyError, ValueError):
            return []

    async def _call_llm(self, prompt: str) -> str:
        """Call the LLM for selection.

        Args:
            prompt: The selection prompt.

        Returns:
            LLM response.
        """
        try:
            import anthropic

            settings = get_settings()
            api_key = settings.api_keys.anthropic_api_key

            if not api_key:
                raise ValueError("No API key available")

            client = anthropic.Anthropic(api_key=api_key)
            message = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=1000,
                messages=[{"role": "user", "content": prompt}],
            )

            return message.content[0].text

        except ImportError:
            raise ValueError("Anthropic library not installed")


# =============================================================================
# Hybrid Retriever
# =============================================================================


class HybridRetriever(BaseRetriever):
    """Hybrid retrieval combining RAG and LLM with confidence-based escalation.

    This retriever starts with fast RAG retrieval, then escalates to LLM
    if the confidence in RAG results is below a threshold. It provides
    the best balance of speed and accuracy.

    Escalation triggers:
    - Low top similarity score
    - Small gap between top and other scores
    - Ambiguous score distribution

    Example:
        >>> retriever = HybridRetriever(
        ...     item_layer=items,
        ...     confidence_threshold=0.7
        ... )
        >>> results = await retriever.retrieve("complex query", k=5)
    """

    def __init__(
        self,
        item_layer: Any,  # ItemLayer
        category_layer: Optional[Any] = None,  # CategoryLayer
        confidence_threshold: float = 0.7,
        always_escalate_for_categories: Optional[list[str]] = None,
    ) -> None:
        """Initialize the HybridRetriever.

        Args:
            item_layer: The memory item layer.
            category_layer: Optional category layer.
            confidence_threshold: Threshold for escalation to LLM.
            always_escalate_for_categories: Categories that always use LLM.
        """
        self._item_layer = item_layer
        self._category_layer = category_layer

        self._rag = RAGRetriever(item_layer, category_layer)
        self._llm = LLMRetriever(item_layer, category_layer)
        self._scorer = ConfidenceScorer(threshold=confidence_threshold)

        self._always_escalate = set(always_escalate_for_categories or [])

    @property
    def retrieval_type(self) -> str:
        """Get the retrieval type."""
        return "hybrid"

    async def retrieve(
        self,
        query: str,
        k: int = 10,
        category: Optional[str] = None,
        force_mode: Optional[RetrievalMode] = None,
    ) -> list[MemoryItem]:
        """Retrieve relevant items using hybrid strategy.

        Args:
            query: The search query.
            k: Maximum number of results.
            category: Optional category filter.
            force_mode: Force a specific retrieval mode.

        Returns:
            List of relevant MemoryItems.
        """
        result, _ = await self._retrieve_with_context(query, k, category, force_mode)
        return result.items

    async def retrieve_with_scores(
        self,
        query: str,
        k: int = 10,
        category: Optional[str] = None,
        force_mode: Optional[RetrievalMode] = None,
    ) -> RetrievalResult:
        """Retrieve items with scores and metadata.

        Args:
            query: The search query.
            k: Maximum number of results.
            category: Optional category filter.
            force_mode: Force a specific retrieval mode.

        Returns:
            RetrievalResult with items and scores.
        """
        result, _ = await self._retrieve_with_context(query, k, category, force_mode)
        return result

    async def retrieve_with_context(
        self,
        query: str,
        k: int = 10,
        category: Optional[str] = None,
        force_mode: Optional[RetrievalMode] = None,
    ) -> tuple[RetrievalResult, RetrievalContext]:
        """Retrieve items with full context information.

        Args:
            query: The search query.
            k: Maximum number of results.
            category: Optional category filter.
            force_mode: Force a specific retrieval mode.

        Returns:
            Tuple of (RetrievalResult, RetrievalContext).
        """
        return await self._retrieve_with_context(query, k, category, force_mode)

    async def _retrieve_with_context(
        self,
        query: str,
        k: int,
        category: Optional[str],
        force_mode: Optional[RetrievalMode],
    ) -> tuple[RetrievalResult, RetrievalContext]:
        """Internal retrieval with context.

        Args:
            query: The search query.
            k: Maximum number of results.
            category: Optional category filter.
            force_mode: Force a specific retrieval mode.

        Returns:
            Tuple of (RetrievalResult, RetrievalContext).
        """
        # Determine mode
        if force_mode == RetrievalMode.LLM:
            return await self._llm_retrieve(query, k, category)

        if force_mode == RetrievalMode.RAG:
            return await self._rag_retrieve(query, k, category)

        # Check if category always escalates
        if category and category in self._always_escalate:
            return await self._llm_retrieve(query, k, category)

        # Start with RAG
        rag_results = await self._rag.retrieve_scored(query, k=k * 2, category=category)

        # Check confidence
        if self._scorer.should_escalate(rag_results):
            # Escalate to LLM
            result, context = await self._llm_retrieve(query, k, category)
            context.escalated = True
            return result, context

        # RAG results are confident enough
        items = [item for item, _ in rag_results[:k]]
        scores = [score for _, score in rag_results[:k]]

        result = RetrievalResult(
            items=items,
            scores=scores,
            total_found=len(items),
        )

        context = RetrievalContext(
            query=query,
            mode=RetrievalMode.RAG,
            items_considered=len(rag_results),
            confidence=self._scorer.score(rag_results),
            escalated=False,
        )

        return result, context

    async def _rag_retrieve(
        self,
        query: str,
        k: int,
        category: Optional[str],
    ) -> tuple[RetrievalResult, RetrievalContext]:
        """Perform RAG-only retrieval.

        Args:
            query: The search query.
            k: Maximum number of results.
            category: Optional category filter.

        Returns:
            Tuple of (RetrievalResult, RetrievalContext).
        """
        result = await self._rag.retrieve_with_scores(query, k, category)

        context = RetrievalContext(
            query=query,
            mode=RetrievalMode.RAG,
            items_considered=result.total_found,
            confidence=1.0,  # Forced mode, so confidence is assumed
        )

        return result, context

    async def _llm_retrieve(
        self,
        query: str,
        k: int,
        category: Optional[str],
    ) -> tuple[RetrievalResult, RetrievalContext]:
        """Perform LLM retrieval.

        Args:
            query: The search query.
            k: Maximum number of results.
            category: Optional category filter.

        Returns:
            Tuple of (RetrievalResult, RetrievalContext).
        """
        result = await self._llm.retrieve_with_scores(query, k, category)

        context = RetrievalContext(
            query=query,
            mode=RetrievalMode.LLM,
            items_considered=result.total_found,
            confidence=1.0,  # LLM is assumed accurate
        )

        return result, context


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "RetrievalMode",
    "ScoredItem",
    "RetrievalContext",
    "ConfidenceScorer",
    "RAGRetriever",
    "LLMRetriever",
    "HybridRetriever",
]
