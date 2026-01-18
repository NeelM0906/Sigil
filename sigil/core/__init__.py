"""Core module for Sigil v2 framework.

This module contains the fundamental building blocks of the Sigil framework:
- Base classes: BaseAgent, BaseStrategy, BaseRetriever
- Result types: StrategyResult, RetrievalResult
- Custom exceptions for error handling
- Middleware architecture for pipeline customization

Usage:
    from sigil.core import BaseAgent, BaseStrategy, SigilError

    class MyAgent(BaseAgent):
        ...

    try:
        agent.run("message")
    except SigilError as e:
        print(f"Error: {e.code} - {e.message}")

Middleware Example:
    from sigil.core import BaseMiddleware, MiddlewareChain

    class MyMiddleware(BaseMiddleware):
        async def pre_step(self, step_name, ctx):
            print(f"Starting: {step_name}")
            return ctx

    chain = MiddlewareChain()
    chain.add(MyMiddleware())
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

from sigil.core.middleware import (
    # Enums
    StepName,
    # Protocol and base class
    SigilMiddleware,
    BaseMiddleware,
    # Chain
    MiddlewareChain,
    # Steps
    StepHandler,
    StepDefinition,
    StepRegistry,
    # Config
    MiddlewareConfig,
    # Runner
    PipelineRunner,
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
    # Memory errors (new names)
    SigilMemoryError,
    SigilMemoryWriteError,
    SigilMemoryRetrievalError,
    # Memory errors (backwards compatibility aliases)
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
    ServiceConnectionError,
    MCPConnectionError,  # Backward compatibility alias
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
    # Middleware
    "StepName",
    "SigilMiddleware",
    "BaseMiddleware",
    "MiddlewareChain",
    "StepHandler",
    "StepDefinition",
    "StepRegistry",
    "MiddlewareConfig",
    "PipelineRunner",
    # Base exception
    "SigilError",
    # Configuration
    "ConfigurationError",
    # Agent errors
    "AgentError",
    "AgentInitializationError",
    "AgentExecutionError",
    "AgentTimeoutError",
    # Memory errors (new names)
    "SigilMemoryError",
    "SigilMemoryWriteError",
    "SigilMemoryRetrievalError",
    # Memory errors (backwards compatibility aliases)
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
    "ServiceConnectionError",
    "MCPConnectionError",  # Backward compatibility alias
    # Evolution errors
    "EvolutionError",
    "OptimizationError",
    "PromptMutationError",
    # Token budget
    "TokenBudgetExceeded",
    "TokenBudgetWarning",
]
