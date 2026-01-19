"""Shared fixtures for reasoning module tests."""

import asyncio
import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock, patch

from sigil.reasoning.strategies.base import (
    BaseReasoningStrategy,
    StrategyResult,
    StrategyConfig,
    COMPLEXITY_RANGES,
    TOKEN_BUDGETS,
)
from sigil.reasoning.strategies.direct import DirectStrategy
from sigil.reasoning.strategies.chain_of_thought import ChainOfThoughtStrategy
from sigil.reasoning.strategies.tree_of_thoughts import TreeOfThoughtsStrategy
from sigil.reasoning.strategies.react import ReActStrategy
from sigil.reasoning.strategies.mcts import MCTSStrategy
from sigil.reasoning.manager import ReasoningManager, StrategyMetrics
from sigil.state.store import EventStore
from sigil.telemetry.tokens import TokenTracker


@pytest.fixture
def temp_storage_dir():
    """Create a temporary storage directory for tests."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def event_store(temp_storage_dir):
    """Create an EventStore for tests."""
    return EventStore(storage_dir=temp_storage_dir + "/events")


@pytest.fixture
def token_tracker():
    """Create a TokenTracker for tests."""
    return TokenTracker()


@pytest.fixture
def direct_strategy(event_store, token_tracker):
    """Create a DirectStrategy instance."""
    return DirectStrategy(
        event_store=event_store,
        token_tracker=token_tracker,
    )


@pytest.fixture
def cot_strategy(event_store, token_tracker):
    """Create a ChainOfThoughtStrategy instance."""
    return ChainOfThoughtStrategy(
        event_store=event_store,
        token_tracker=token_tracker,
    )


@pytest.fixture
def tot_strategy(event_store, token_tracker):
    """Create a TreeOfThoughtsStrategy instance."""
    return TreeOfThoughtsStrategy(
        event_store=event_store,
        token_tracker=token_tracker,
    )


@pytest.fixture
def react_strategy(event_store, token_tracker):
    """Create a ReActStrategy instance."""
    return ReActStrategy(
        event_store=event_store,
        token_tracker=token_tracker,
    )


@pytest.fixture
def mcts_strategy(event_store, token_tracker):
    """Create an MCTSStrategy instance."""
    return MCTSStrategy(
        event_store=event_store,
        token_tracker=token_tracker,
    )


@pytest.fixture
def reasoning_manager(event_store, token_tracker):
    """Create a ReasoningManager instance."""
    return ReasoningManager(
        event_store=event_store,
        token_tracker=token_tracker,
    )


@pytest.fixture
def simple_task():
    """Return a simple task suitable for direct strategy."""
    return "What is 2 + 2?"


@pytest.fixture
def moderate_task():
    """Return a moderate task suitable for chain-of-thought."""
    return "Calculate the total cost: 3 items at $15 each, with 10% tax"


@pytest.fixture
def complex_task():
    """Return a complex task suitable for tree-of-thoughts."""
    return "Analyze the pros and cons of three different marketing strategies and recommend the best one"


@pytest.fixture
def tool_task():
    """Return a task requiring tool use for ReAct strategy."""
    return "Search for the latest news about AI and summarize the top 3 stories"


@pytest.fixture
def critical_task():
    """Return a critical task suitable for MCTS strategy."""
    return "Design a production deployment strategy for a critical system with zero downtime requirements"


@pytest.fixture
def sample_context():
    """Return sample context for tasks."""
    return {
        "user": "test_user",
        "session_id": "test_session_123",
        "industry": "Technology",
        "budget": "$10000",
    }


@pytest.fixture
def mock_llm_response():
    """Create a mock LLM response function."""
    async def mock_call(prompt, temperature=0.7, max_tokens=300):
        # Simulate LLM response based on prompt content
        if "2 + 2" in prompt:
            return "4", 50
        elif "cost" in prompt.lower():
            return "Total cost: $49.50 (3 items at $15 = $45, plus 10% tax = $4.50)", 100
        elif "marketing" in prompt.lower():
            return "Based on analysis, Strategy B is recommended due to higher ROI potential.", 200
        else:
            return "Generated response for the task.", 75
    return mock_call


@pytest.fixture
def mock_tool_registry():
    """Create a mock tool registry."""
    async def mock_execute(tool_name: str, **kwargs):
        if tool_name == "websearch.search":
            return {"results": ["Result 1", "Result 2", "Result 3"]}
        elif tool_name == "memory.recall":
            return {"memories": ["Previous fact 1", "Previous fact 2"]}
        else:
            return {"status": "success"}

    registry = MagicMock()
    registry.execute = AsyncMock(side_effect=mock_execute)
    return registry
