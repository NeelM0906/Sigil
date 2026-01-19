"""Core module for Sigil v2 framework.

This module contains the fundamental building blocks of the Sigil framework:
- Base classes: BaseAgent, BaseStrategy, BaseRetriever
- Result types: StrategyResult, RetrievalResult
- Custom exceptions for error handling

Usage:
    from sigil.core import BaseAgent, BaseStrategy, SigilError

    class MyAgent(BaseAgent):
        ...

    try:
        agent.run("message")
    except SigilError as e:
        print(f"Error: {e.code} - {e.message}")
"""

from sigil.core.base import (
    # Type variables
    ConfigT,
    ResponseT,
    # Result types
    StrategyResult,
    RetrievalResult,
    # Base classes
    BaseAgent,
    BaseStrategy,
    BaseRetriever,
)

from sigil.core.exceptions import (
    # Base exception
    SigilError,
    # Configuration
    ConfigurationError,
    # Agent errors
    AgentError,
    AgentInitializationError,
    AgentExecutionError,
    AgentTimeoutError,
    # Memory errors
    MemoryError,
    MemoryWriteError,
    MemoryRetrievalError,
    # Reasoning errors
    ReasoningError,
    StrategyNotFoundError,
    # Planning errors
    PlanExecutionError,
    # Routing errors
    RoutingError,
    # Contract errors
    ContractError,
    ContractValidationError,
    ContractViolation,
    # Tool errors
    ToolError,
    ToolNotFoundError,
    ToolExecutionError,
    MCPConnectionError,
    # Evolution errors
    EvolutionError,
    OptimizationError,
    PromptMutationError,
    # Token budget
    TokenBudgetExceeded,
    TokenBudgetWarning,
)

__all__ = [
    # Type variables
    "ConfigT",
    "ResponseT",
    # Result types
    "StrategyResult",
    "RetrievalResult",
    # Base classes
    "BaseAgent",
    "BaseStrategy",
    "BaseRetriever",
    # Base exception
    "SigilError",
    # Configuration
    "ConfigurationError",
    # Agent errors
    "AgentError",
    "AgentInitializationError",
    "AgentExecutionError",
    "AgentTimeoutError",
    # Memory errors
    "MemoryError",
    "MemoryWriteError",
    "MemoryRetrievalError",
    # Reasoning errors
    "ReasoningError",
    "StrategyNotFoundError",
    # Planning errors
    "PlanExecutionError",
    # Routing errors
    "RoutingError",
    # Contract errors
    "ContractError",
    "ContractValidationError",
    "ContractViolation",
    # Tool errors
    "ToolError",
    "ToolNotFoundError",
    "ToolExecutionError",
    "MCPConnectionError",
    # Evolution errors
    "EvolutionError",
    "OptimizationError",
    "PromptMutationError",
    # Token budget
    "TokenBudgetExceeded",
    "TokenBudgetWarning",
]
