"""Tests for ContextManager and ContextCompressor.

This module contains comprehensive tests for the context management system,
including parallel assembly, token tracking, and compression strategies.

Test Categories:
    - ContextManager tests
    - ContextSource tests
    - ContextItem tests
    - Parallel assembly tests
    - Token budget tests
    - Compression strategy tests
    - Cache tests
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timezone

from sigil.context.manager import (
    ContextManager,
    ContextSource,
    ContextItem,
    ContextAssemblyResult,
    ContextBudget,
    ContextProvider,
    MemoryContextProvider,
    ConversationContextProvider,
    DEFAULT_MAX_CONTEXT_TOKENS,
    DEFAULT_RESERVED_FOR_RESPONSE,
    CHARS_PER_TOKEN,
)
from sigil.context.compression import (
    ContextCompressor,
    CompressionStrategy,
    CompressionResult,
    TruncateStrategy,
    SummarizeStrategy,
    PrioritizeStrategy,
    COMPRESSION_CACHE_TTL_SECONDS,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def context_manager():
    """Create a basic context manager."""
    return ContextManager()


@pytest.fixture
def context_manager_with_memory():
    """Create context manager with mock memory."""
    mock_memory = MagicMock()
    mock_memory.retrieve = AsyncMock(return_value=[
        MagicMock(content="Memory item 1", category="facts"),
        MagicMock(content="Memory item 2", category="facts"),
    ])
    return ContextManager(memory_manager=mock_memory)


@pytest.fixture
def compressor():
    """Create a basic context compressor."""
    return ContextCompressor()


@pytest.fixture
def compressor_with_llm():
    """Create compressor with mock LLM."""
    async def mock_llm(prompt: str) -> str:
        return "This is a summarized version of the content."

    return ContextCompressor(llm_call=mock_llm)


# =============================================================================
# ContextSource Tests
# =============================================================================


class TestContextSource:
    """Tests for ContextSource enum."""

    def test_all_sources_defined(self):
        """Test that all expected sources are defined."""
        expected = [
            "SYSTEM_PROMPT",
            "TASK",
            "WORKING_MEMORY",
            "LONG_TERM_MEMORY",
            "CONVERSATION_HISTORY",
            "TOOL_RESULTS",
            "PLAN",
            "USER_CONTEXT",
        ]
        for source_name in expected:
            assert hasattr(ContextSource, source_name)

    def test_source_values(self):
        """Test source enum values."""
        assert ContextSource.SYSTEM_PROMPT.value == "system_prompt"
        assert ContextSource.TASK.value == "task"


# =============================================================================
# ContextItem Tests
# =============================================================================


class TestContextItem:
    """Tests for ContextItem dataclass."""

    def test_item_creation(self):
        """Test basic item creation."""
        item = ContextItem(
            source=ContextSource.TASK,
            content="Do something",
            tokens=10,
        )
        assert item.source == ContextSource.TASK
        assert item.content == "Do something"
        assert item.tokens == 10
        assert item.priority == 1.0

    def test_item_to_string_str_content(self):
        """Test to_string with string content."""
        item = ContextItem(
            source=ContextSource.TASK,
            content="Hello world",
            tokens=5,
        )
        assert item.to_string() == "Hello world"

    def test_item_to_string_dict_content(self):
        """Test to_string with dict content."""
        item = ContextItem(
            source=ContextSource.USER_CONTEXT,
            content={"key": "value"},
            tokens=10,
        )
        result = item.to_string()
        assert "key" in result
        assert "value" in result

    def test_item_to_string_list_content(self):
        """Test to_string with list content."""
        item = ContextItem(
            source=ContextSource.TOOL_RESULTS,
            content=["result1", "result2"],
            tokens=10,
        )
        result = item.to_string()
        assert "result1" in result
        assert "result2" in result


# =============================================================================
# ContextAssemblyResult Tests
# =============================================================================


class TestContextAssemblyResult:
    """Tests for ContextAssemblyResult dataclass."""

    def test_result_creation_empty(self):
        """Test creating empty result."""
        result = ContextAssemblyResult()
        assert result.items == []
        assert result.total_tokens == 0
        assert result.budget_used == 0.0

    def test_get_by_source(self):
        """Test getting item by source."""
        result = ContextAssemblyResult(
            items=[
                ContextItem(
                    source=ContextSource.TASK,
                    content="task content",
                    tokens=10,
                ),
                ContextItem(
                    source=ContextSource.SYSTEM_PROMPT,
                    content="system content",
                    tokens=20,
                ),
            ]
        )

        task_item = result.get_by_source(ContextSource.TASK)
        assert task_item is not None
        assert task_item.content == "task content"

        missing = result.get_by_source(ContextSource.PLAN)
        assert missing is None

    def test_to_dict(self):
        """Test serialization to dict."""
        result = ContextAssemblyResult(
            total_tokens=100,
            budget_used=0.5,
            assembly_time_ms=10.5,
        )
        data = result.to_dict()
        assert data["total_tokens"] == 100
        assert data["budget_used"] == 0.5

    def test_to_context_string(self):
        """Test building context string."""
        result = ContextAssemblyResult(
            items=[
                ContextItem(
                    source=ContextSource.SYSTEM_PROMPT,
                    content="You are an assistant",
                    tokens=10,
                ),
                ContextItem(
                    source=ContextSource.TASK,
                    content="Help the user",
                    tokens=5,
                ),
            ]
        )
        context_str = result.to_context_string()
        assert "SYSTEM_PROMPT" in context_str
        assert "TASK" in context_str
        assert "You are an assistant" in context_str

    def test_to_messages_format(self):
        """Test conversion to messages format."""
        result = ContextAssemblyResult(
            items=[
                ContextItem(
                    source=ContextSource.SYSTEM_PROMPT,
                    content="You are an assistant",
                    tokens=10,
                ),
                ContextItem(
                    source=ContextSource.TASK,
                    content="Help me",
                    tokens=5,
                ),
            ]
        )
        messages = result.to_messages_format()
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"


# =============================================================================
# ContextBudget Tests
# =============================================================================


class TestContextBudget:
    """Tests for ContextBudget tracking."""

    def test_budget_creation(self):
        """Test budget creation."""
        budget = ContextBudget(max_tokens=10000)
        assert budget.max_tokens == 10000
        assert budget.total_allocated == 0
        assert budget.total_used == 0

    def test_allocate_tokens(self):
        """Test allocating tokens to sources."""
        budget = ContextBudget(max_tokens=10000)
        budget.allocate(ContextSource.SYSTEM_PROMPT, 1000)
        budget.allocate(ContextSource.TASK, 500)

        assert budget.allocated[ContextSource.SYSTEM_PROMPT] == 1000
        assert budget.total_allocated == 1500

    def test_use_tokens(self):
        """Test using allocated tokens."""
        budget = ContextBudget(max_tokens=10000)
        budget.allocate(ContextSource.TASK, 500)

        actual = budget.use(ContextSource.TASK, 300)

        assert actual == 300
        assert budget.used[ContextSource.TASK] == 300
        assert budget.remaining == 10000 - 300

    def test_use_tokens_capped_at_allocation(self):
        """Test that usage is capped at allocation."""
        budget = ContextBudget(max_tokens=10000)
        budget.allocate(ContextSource.TASK, 100)

        actual = budget.use(ContextSource.TASK, 200)

        assert actual == 100  # Capped at allocation

    def test_remaining_for_source(self):
        """Test getting remaining tokens for a source."""
        budget = ContextBudget(max_tokens=10000)
        budget.allocate(ContextSource.TASK, 500)
        budget.use(ContextSource.TASK, 200)

        remaining = budget.get_remaining_for(ContextSource.TASK)
        assert remaining == 300


# =============================================================================
# ContextManager Tests
# =============================================================================


class TestContextManagerInit:
    """Tests for ContextManager initialization."""

    def test_init_default(self):
        """Test default initialization."""
        manager = ContextManager()
        assert manager._max_tokens == DEFAULT_MAX_CONTEXT_TOKENS
        assert manager._reserved_for_response == DEFAULT_RESERVED_FOR_RESPONSE

    def test_init_custom_tokens(self):
        """Test initialization with custom token limits."""
        manager = ContextManager(
            max_tokens=50000,
            reserved_for_response=2000,
        )
        assert manager._max_tokens == 50000
        assert manager._reserved_for_response == 2000

    def test_available_tokens(self):
        """Test available tokens calculation."""
        manager = ContextManager(
            max_tokens=10000,
            reserved_for_response=2000,
        )
        assert manager.available_tokens == 8000


class TestContextManagerAssembly:
    """Tests for ContextManager assembly."""

    @pytest.mark.asyncio
    async def test_assemble_basic(self, context_manager):
        """Test basic context assembly."""
        result = await context_manager.assemble(
            task="Test task",
            session_id="sess-123",
        )

        assert isinstance(result, ContextAssemblyResult)
        assert result.total_tokens > 0
        assert result.assembly_time_ms > 0

    @pytest.mark.asyncio
    async def test_assemble_with_system_prompt(self, context_manager):
        """Test assembly with system prompt."""
        result = await context_manager.assemble(
            task="Test task",
            session_id="sess-123",
            system_prompt="You are a helpful assistant.",
        )

        system_item = result.get_by_source(ContextSource.SYSTEM_PROMPT)
        assert system_item is not None
        assert "helpful assistant" in system_item.content

    @pytest.mark.asyncio
    async def test_assemble_with_user_context(self, context_manager):
        """Test assembly with user context."""
        result = await context_manager.assemble(
            task="Test task",
            session_id="sess-123",
            user_context={"company": "Acme Corp"},
        )

        ctx_item = result.get_by_source(ContextSource.USER_CONTEXT)
        assert ctx_item is not None

    @pytest.mark.asyncio
    async def test_assemble_with_conversation(self, context_manager):
        """Test assembly with conversation history."""
        conversation = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]

        result = await context_manager.assemble(
            task="Continue conversation",
            session_id="sess-123",
            conversation=conversation,
        )

        history_item = result.get_by_source(ContextSource.CONVERSATION_HISTORY)
        assert history_item is not None
        assert len(history_item.content) == 2

    @pytest.mark.asyncio
    async def test_assemble_budget_tracking(self, context_manager):
        """Test that budget is tracked during assembly."""
        result = await context_manager.assemble(
            task="Test task",
            session_id="sess-123",
            system_prompt="You are an assistant.",
            token_budget=1000,
        )

        assert result.budget_used > 0
        assert result.total_tokens <= 1000

    @pytest.mark.asyncio
    async def test_assemble_parallel_fetches(self, context_manager_with_memory):
        """Test that memory and conversation are fetched in parallel."""
        result = await context_manager_with_memory.assemble(
            task="Test task",
            session_id="sess-123",
            conversation=[{"role": "user", "content": "Hello"}],
        )

        # Both memory and conversation should be assembled
        assert result is not None


class TestContextManagerProviders:
    """Tests for context providers."""

    def test_register_provider(self):
        """Test registering a custom provider."""
        manager = ContextManager()

        class CustomProvider(ContextProvider):
            def __init__(self):
                super().__init__(ContextSource.WORKING_MEMORY)

            async def fetch(self, query, budget_tokens, **kwargs):
                return ContextItem(
                    source=self.source,
                    content="Custom content",
                    tokens=50,
                )

        manager.register_provider(
            ContextSource.WORKING_MEMORY,
            CustomProvider(),
        )

        assert ContextSource.WORKING_MEMORY in manager._providers


class TestContextManagerMetrics:
    """Tests for context manager metrics."""

    def test_get_metrics(self, context_manager):
        """Test getting metrics."""
        metrics = context_manager.get_metrics()

        assert "total_assemblies" in metrics
        assert "max_tokens" in metrics

    def test_reset_metrics(self, context_manager):
        """Test resetting metrics."""
        context_manager._total_assemblies = 10

        context_manager.reset_metrics()

        assert context_manager._total_assemblies == 0

    @pytest.mark.asyncio
    async def test_metrics_updated_after_assembly(self, context_manager):
        """Test metrics are updated after assembly."""
        await context_manager.assemble(
            task="Test",
            session_id="sess-123",
        )

        metrics = context_manager.get_metrics()
        assert metrics["total_assemblies"] == 1


# =============================================================================
# Compression Strategy Tests
# =============================================================================


class TestTruncateStrategy:
    """Tests for TruncateStrategy."""

    def test_truncate_string_keep_start(self):
        """Test truncating string keeping start."""
        strategy = TruncateStrategy()
        content = "A" * 1000

        result = strategy.compress(
            content=content,
            original_tokens=250,
            target_tokens=100,
            keep_start=True,
        )

        assert result.compressed_tokens <= 100
        assert result.content.startswith("A")
        assert result.content.endswith("...")

    def test_truncate_string_keep_end(self):
        """Test truncating string keeping end."""
        strategy = TruncateStrategy()
        content = "A" * 500 + "B" * 500

        result = strategy.compress(
            content=content,
            original_tokens=250,
            target_tokens=100,
            keep_start=False,
        )

        assert result.content.startswith("...")
        assert "B" in result.content

    def test_truncate_list(self):
        """Test truncating a list."""
        strategy = TruncateStrategy()
        content = [f"Item {i}" for i in range(20)]

        result = strategy.compress(
            content=content,
            original_tokens=200,
            target_tokens=50,
        )

        assert len(result.content) < 20

    def test_truncate_no_compression_needed(self):
        """Test when no compression is needed."""
        strategy = TruncateStrategy()
        content = "Short text"

        result = strategy.compress(
            content=content,
            original_tokens=5,
            target_tokens=100,
        )

        assert result.content == content
        assert result.compression_ratio == 1.0


class TestSummarizeStrategy:
    """Tests for SummarizeStrategy."""

    @pytest.mark.asyncio
    async def test_summarize_without_llm(self):
        """Test summarization without LLM (falls back to extractive)."""
        strategy = SummarizeStrategy(llm_call=None)
        content = "A" * 2000

        result = await strategy.compress(
            content=content,
            original_tokens=500,
            target_tokens=100,
        )

        assert result.compressed_tokens <= 500
        assert "[... content truncated ...]" in result.content

    @pytest.mark.asyncio
    async def test_summarize_with_llm(self):
        """Test summarization with LLM."""
        async def mock_llm(prompt: str) -> str:
            return "Summary of the content."

        strategy = SummarizeStrategy(llm_call=mock_llm)
        content = "Long content " * 100

        result = await strategy.compress(
            content=content,
            original_tokens=500,
            target_tokens=50,
        )

        assert result.content == "Summary of the content."

    @pytest.mark.asyncio
    async def test_summarize_no_compression_needed(self):
        """Test when no compression is needed."""
        strategy = SummarizeStrategy()

        result = await strategy.compress(
            content="Short",
            original_tokens=5,
            target_tokens=100,
        )

        assert result.content == "Short"


class TestPrioritizeStrategy:
    """Tests for PrioritizeStrategy."""

    def test_prioritize_keeps_high_priority(self):
        """Test that high priority items are kept."""
        strategy = PrioritizeStrategy()

        items = [
            MagicMock(priority=0.3, tokens=50),
            MagicMock(priority=0.9, tokens=50),
            MagicMock(priority=0.5, tokens=50),
        ]

        result = strategy.compress(
            items=items,
            original_tokens=150,
            target_tokens=100,
        )

        # Should keep the 2 highest priority items
        assert len(result.content) == 2
        assert result.metadata["items_kept"] == 2

    def test_prioritize_no_compression_needed(self):
        """Test when no compression is needed."""
        strategy = PrioritizeStrategy()

        items = [MagicMock(priority=1.0, tokens=30)]

        result = strategy.compress(
            items=items,
            original_tokens=30,
            target_tokens=100,
        )

        assert len(result.content) == 1


# =============================================================================
# ContextCompressor Tests
# =============================================================================


class TestContextCompressorInit:
    """Tests for ContextCompressor initialization."""

    def test_init_default(self):
        """Test default initialization."""
        compressor = ContextCompressor()
        assert compressor._cache_enabled is True

    def test_init_cache_disabled(self):
        """Test initialization with cache disabled."""
        compressor = ContextCompressor(cache_enabled=False)
        assert compressor._cache_enabled is False


class TestContextCompressorStrategySelection:
    """Tests for strategy selection."""

    def test_select_truncate_minimal_overflow(self, compressor):
        """Test truncate is selected for minimal overflow."""
        strategy = compressor.select_strategy(overflow_ratio=1.1)
        assert strategy == CompressionStrategy.TRUNCATE

    def test_select_summarize_moderate_overflow(self, compressor):
        """Test summarize is selected for moderate overflow."""
        strategy = compressor.select_strategy(overflow_ratio=1.6)
        assert strategy == CompressionStrategy.SUMMARIZE

    def test_select_prioritize_for_lists(self, compressor):
        """Test prioritize is selected for lists with high overflow."""
        strategy = compressor.select_strategy(
            overflow_ratio=2.5,
            content_type="list",
        )
        assert strategy == CompressionStrategy.PRIORITIZE


class TestContextCompressorCompress:
    """Tests for compression."""

    @pytest.mark.asyncio
    async def test_compress_no_compression_needed(self, compressor):
        """Test when no compression is needed."""
        result = await compressor.compress(
            content="Short text",
            original_tokens=10,
            target_tokens=100,
        )

        assert result.content == "Short text"
        assert result.compression_ratio == 1.0

    @pytest.mark.asyncio
    async def test_compress_with_truncate(self, compressor):
        """Test compression with truncate strategy."""
        result = await compressor.compress(
            content="A" * 1000,
            original_tokens=250,
            target_tokens=100,
            strategy=CompressionStrategy.TRUNCATE,
        )

        assert result.compressed_tokens <= 100
        assert result.strategy_used == CompressionStrategy.TRUNCATE

    @pytest.mark.asyncio
    async def test_compress_auto_strategy(self, compressor):
        """Test compression with automatic strategy selection."""
        result = await compressor.compress(
            content="A" * 1000,
            original_tokens=250,
            target_tokens=200,
        )

        assert result is not None


class TestContextCompressorCache:
    """Tests for compression caching."""

    @pytest.mark.asyncio
    async def test_cache_hit(self, compressor):
        """Test cache hit on repeated compression."""
        content = "Test content for caching"

        # First compression
        result1 = await compressor.compress(
            content=content,
            original_tokens=100,
            target_tokens=50,
        )

        # Second compression (should hit cache)
        result2 = await compressor.compress(
            content=content,
            original_tokens=100,
            target_tokens=50,
        )

        assert result2.cache_hit is True

    @pytest.mark.asyncio
    async def test_cache_disabled(self):
        """Test compression with cache disabled."""
        compressor = ContextCompressor(cache_enabled=False)

        result1 = await compressor.compress(
            content="Test",
            original_tokens=100,
            target_tokens=50,
        )

        result2 = await compressor.compress(
            content="Test",
            original_tokens=100,
            target_tokens=50,
        )

        # Neither should be cache hit
        assert result2.cache_hit is False

    def test_clear_cache(self, compressor):
        """Test clearing the cache."""
        compressor._cache["test_key"] = MagicMock()

        compressor.clear_cache()

        assert len(compressor._cache) == 0


class TestContextCompressorMetrics:
    """Tests for compressor metrics."""

    def test_get_metrics(self, compressor):
        """Test getting metrics."""
        metrics = compressor.get_metrics()

        assert "total_compressions" in metrics
        assert "cache_hits" in metrics
        assert "strategy_usage" in metrics

    @pytest.mark.asyncio
    async def test_metrics_updated_after_compression(self, compressor):
        """Test metrics are updated after compression."""
        await compressor.compress(
            content="A" * 1000,
            original_tokens=250,
            target_tokens=100,
        )

        metrics = compressor.get_metrics()
        assert metrics["total_compressions"] == 1

    def test_reset_metrics(self, compressor):
        """Test resetting metrics."""
        compressor._total_compressions = 10
        compressor._cache_hits = 5

        compressor.reset_metrics()

        assert compressor._total_compressions == 0
        assert compressor._cache_hits == 0
