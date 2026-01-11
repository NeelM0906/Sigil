"""Base classes for the Sigil v2 framework.

This module defines the abstract base classes that form the foundation of the
Sigil agent framework. All agents, strategies, and retrievers inherit from
these base classes to ensure consistent interfaces and behavior.

Classes:
    BaseAgent: Abstract base class for all agent implementations.
        - Defines the core agent lifecycle (run, get_tools)
        - Manages memory, reasoning, and tool integrations
        - Provides hooks for self-evolution and optimization

    BaseStrategy: Abstract base class for reasoning strategies.
        - Defines the interface for strategy execution
        - Supports hierarchical strategy composition
        - Enables dynamic strategy selection based on complexity

    BaseRetriever: Abstract base class for memory retrieval.
        - Defines the interface for querying memory layers
        - Supports semantic, temporal, and categorical retrieval
        - Enables cross-layer memory fusion

Example:
    >>> from sigil.core.base import BaseAgent, BaseStrategy
    >>>
    >>> class MyAgent(BaseAgent):
    ...     @property
    ...     def name(self) -> str:
    ...         return "my-agent"
    ...
    ...     async def run(self, message: str) -> dict:
    ...         return {"response": f"Processed: {message}"}
    ...
    ...     def get_tools(self) -> list:
    ...         return []
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Generic, Optional, TypeVar


# =============================================================================
# Type Variables
# =============================================================================

ConfigT = TypeVar("ConfigT")
"""Type variable for agent configuration."""

ResponseT = TypeVar("ResponseT")
"""Type variable for agent response."""


# =============================================================================
# Result Dataclasses
# =============================================================================


@dataclass
class StrategyResult:
    """Result from executing a reasoning strategy.

    Attributes:
        success: Whether the strategy execution succeeded.
        output: The output from the strategy execution.
        reasoning_trace: List of reasoning steps taken.
        tokens_used: Number of tokens consumed.
        metadata: Additional strategy-specific metadata.
    """

    success: bool
    output: Any
    reasoning_trace: list[str]
    tokens_used: int = 0
    metadata: dict[str, Any] | None = None


@dataclass
class RetrievalResult:
    """Result from a memory retrieval operation.

    Attributes:
        items: List of retrieved memory items.
        scores: Relevance scores for each item (optional).
        total_found: Total number of items matching the query.
        query_time_ms: Time taken to execute the query.
    """

    items: list[Any]
    scores: list[float] | None = None
    total_found: int = 0
    query_time_ms: float = 0.0


# =============================================================================
# BaseAgent
# =============================================================================


class BaseAgent(ABC, Generic[ConfigT, ResponseT]):
    """Abstract base class for all Sigil agents.

    All agent implementations must inherit from this class and implement
    the required abstract methods. The base class provides common
    functionality for memory management, tool integration, and lifecycle
    management.

    Type Parameters:
        ConfigT: The configuration type for this agent.
        ResponseT: The response type produced by this agent.

    Abstract Methods:
        run: Execute the agent on a user message.
        get_tools: Return the list of available tools.

    Required Properties:
        name: The unique name identifier for this agent.
        config: The agent's configuration object.

    Example:
        >>> class MyAgent(BaseAgent[MyConfig, MyResponse]):
        ...     def __init__(self, config: MyConfig):
        ...         self._config = config
        ...
        ...     @property
        ...     def name(self) -> str:
        ...         return "my-agent"
        ...
        ...     @property
        ...     def config(self) -> MyConfig:
        ...         return self._config
        ...
        ...     async def run(self, message: str) -> MyResponse:
        ...         # Implementation
        ...         pass
        ...
        ...     def get_tools(self) -> list:
        ...         return []
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Get the agent's unique name identifier.

        Returns:
            The name of this agent.
        """
        ...

    @property
    @abstractmethod
    def config(self) -> ConfigT:
        """Get the agent's configuration.

        Returns:
            The configuration object for this agent.
        """
        ...

    @abstractmethod
    async def run(self, message: str) -> ResponseT:
        """Execute the agent on a user message.

        This is the main entry point for agent execution. The agent should
        process the message, use any necessary tools, and return a response.

        Args:
            message: The user message to process.

        Returns:
            The agent's response to the message.

        Raises:
            AgentExecutionError: If execution fails.
        """
        ...

    @abstractmethod
    def get_tools(self) -> list[Any]:
        """Get the list of tools available to this agent.

        Returns:
            A list of tool objects that the agent can use.
        """
        ...

    def get_description(self) -> str:
        """Get a description of this agent's capabilities.

        Override this method to provide a custom description.

        Returns:
            A description string.
        """
        return f"Agent: {self.name}"


# =============================================================================
# BaseStrategy
# =============================================================================


class BaseStrategy(ABC):
    """Abstract base class for reasoning strategies.

    Strategies encapsulate different reasoning approaches that agents can
    use. The framework supports hierarchical strategy composition and
    dynamic strategy selection based on task complexity.

    Supported strategy types:
        - Direct: Simple single-step execution (complexity < 0.3)
        - Chain-of-Thought: Step-by-step reasoning (complexity 0.3-0.5)
        - Tree-of-Thoughts: Multiple path exploration (complexity 0.5-0.7)
        - ReAct: Reasoning and Acting interleaved (complexity 0.7-0.9)
        - MCTS: Monte Carlo Tree Search for critical decisions (complexity > 0.9)

    Abstract Methods:
        execute: Execute the strategy on a task.

    Required Properties:
        name: The strategy's unique name identifier.
        complexity_range: The complexity range this strategy handles.

    Example:
        >>> class DirectStrategy(BaseStrategy):
        ...     @property
        ...     def name(self) -> str:
        ...         return "direct"
        ...
        ...     @property
        ...     def complexity_range(self) -> tuple[float, float]:
        ...         return (0.0, 0.3)
        ...
        ...     async def execute(
        ...         self, task: str, context: dict
        ...     ) -> StrategyResult:
        ...         # Simple direct execution
        ...         return StrategyResult(
        ...             success=True,
        ...             output="Result",
        ...             reasoning_trace=["Executed directly"]
        ...         )
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Get the strategy's unique name identifier.

        Returns:
            The name of this strategy.
        """
        ...

    @property
    @abstractmethod
    def complexity_range(self) -> tuple[float, float]:
        """Get the complexity range this strategy handles.

        The complexity range is a tuple of (min, max) values between 0.0 and 1.0.
        Strategies are selected based on task complexity falling within their range.

        Returns:
            A tuple of (min_complexity, max_complexity).
        """
        ...

    @abstractmethod
    async def execute(self, task: str, context: dict[str, Any]) -> StrategyResult:
        """Execute the strategy on a task.

        Args:
            task: The task to execute.
            context: Additional context for execution, including:
                - memory: Available memory items
                - tools: Available tools
                - history: Previous conversation history
                - constraints: Any constraints on execution

        Returns:
            A StrategyResult containing the outcome.

        Raises:
            ReasoningError: If strategy execution fails.
        """
        ...

    def is_applicable(self, complexity: float) -> bool:
        """Check if this strategy is applicable for a given complexity.

        Args:
            complexity: Task complexity score (0.0 to 1.0).

        Returns:
            True if the strategy can handle this complexity level.
        """
        min_c, max_c = self.complexity_range
        return min_c <= complexity <= max_c


# =============================================================================
# BaseRetriever
# =============================================================================


class BaseRetriever(ABC):
    """Abstract base class for memory retrieval.

    Retrievers are responsible for querying the memory system and
    returning relevant information. They support multiple retrieval
    modes and can fuse results from different memory layers.

    Retrieval types:
        - "rag": Embedding-based semantic similarity search
        - "llm": LLM-based reading and reasoning
        - "hybrid": Combination of RAG and LLM approaches

    Abstract Methods:
        retrieve: Query memory and return relevant items.

    Required Properties:
        retrieval_type: The type of retrieval this retriever performs.

    Example:
        >>> class RAGRetriever(BaseRetriever):
        ...     @property
        ...     def retrieval_type(self) -> str:
        ...         return "rag"
        ...
        ...     async def retrieve(
        ...         self, query: str, k: int = 10
        ...     ) -> list:
        ...         # Semantic search implementation
        ...         return []
    """

    @property
    @abstractmethod
    def retrieval_type(self) -> str:
        """Get the retrieval type for this retriever.

        Returns:
            One of: "rag", "llm", "hybrid"
        """
        ...

    @abstractmethod
    async def retrieve(self, query: str, k: int = 10) -> list[Any]:
        """Retrieve relevant items from memory.

        Args:
            query: The retrieval query.
            k: Maximum number of items to retrieve (default: 10).

        Returns:
            List of relevant memory items, sorted by relevance.

        Raises:
            MemoryRetrievalError: If retrieval fails.
        """
        ...

    async def retrieve_with_scores(
        self, query: str, k: int = 10
    ) -> RetrievalResult:
        """Retrieve items with relevance scores.

        Override this method to provide scored results. The default
        implementation calls retrieve() and returns no scores.

        Args:
            query: The retrieval query.
            k: Maximum number of items to retrieve.

        Returns:
            A RetrievalResult containing items and optional scores.
        """
        items = await self.retrieve(query, k)
        return RetrievalResult(items=items, total_found=len(items))


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Type variables
    "ConfigT",
    "ResponseT",
    # Result dataclasses
    "StrategyResult",
    "RetrievalResult",
    # Base classes
    "BaseAgent",
    "BaseStrategy",
    "BaseRetriever",
]
