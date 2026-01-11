"""Custom exceptions for the Sigil v2 framework.

This module defines a hierarchy of exceptions used throughout the Sigil
framework. All exceptions inherit from SigilError, enabling catch-all
exception handling while still allowing specific exception types.

Exception Hierarchy:
    SigilError (base)
    ├── ConfigurationError: Invalid configuration or settings
    ├── AgentError: Agent execution failures
    │   ├── AgentInitializationError: Failed to initialize agent
    │   ├── AgentExecutionError: Failed during task execution
    │   └── AgentTimeoutError: Agent execution exceeded time limit
    ├── MemoryError: Memory system failures
    │   ├── MemoryWriteError: Failed to write to memory
    │   ├── MemoryRetrievalError: Failed to retrieve from memory
    │   └── MemoryLayerError: Memory layer-specific errors
    ├── ReasoningError: Reasoning and strategy failures
    │   ├── StrategyNotFoundError: Requested strategy not available
    │   └── ReasoningTimeoutError: Reasoning exceeded time limit
    ├── ContractError: Contract verification failures
    │   ├── ContractValidationError: Output failed contract validation
    │   └── ContractTimeoutError: Contract execution exceeded limit
    ├── ToolError: Tool execution failures
    │   ├── ToolNotFoundError: Requested tool not available
    │   ├── ToolExecutionError: Tool execution failed
    │   └── MCPConnectionError: Failed to connect to MCP server
    └── EvolutionError: Self-evolution failures
        ├── OptimizationError: TextGrad optimization failed
        └── PromptMutationError: Prompt evolution failed

TODO:
    - Implement full exception hierarchy
    - Add context information to each exception type
    - Add error codes for programmatic handling
    - Add recovery suggestions where applicable
    - Integrate with telemetry for error tracking
"""

from typing import Any, Optional


class SigilError(Exception):
    """Base exception for all Sigil framework errors.

    All custom exceptions in the Sigil framework inherit from this class.
    This enables unified exception handling while preserving the ability
    to catch specific exception types.

    Attributes:
        message: Human-readable error description
        code: Machine-readable error code for programmatic handling
        context: Additional context information about the error
        recoverable: Whether the error is potentially recoverable

    TODO:
        - Add error code registry
        - Add telemetry integration
        - Add structured logging support
    """

    def __init__(
        self,
        message: str,
        code: Optional[str] = None,
        context: Optional[dict[str, Any]] = None,
        recoverable: bool = False,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.code = code or "SIGIL_ERROR"
        self.context = context or {}
        self.recoverable = recoverable

    def __str__(self) -> str:
        return f"[{self.code}] {self.message}"


class ConfigurationError(SigilError):
    """Raised when configuration is invalid or missing.

    TODO: Add config validation details
    """

    def __init__(self, message: str, config_key: Optional[str] = None, **kwargs: Any) -> None:
        super().__init__(message, code="CONFIG_ERROR", **kwargs)
        self.config_key = config_key


class AgentError(SigilError):
    """Base exception for agent-related errors.

    TODO: Add agent identification to context
    """

    def __init__(self, message: str, agent_id: Optional[str] = None, **kwargs: Any) -> None:
        super().__init__(message, code="AGENT_ERROR", **kwargs)
        self.agent_id = agent_id


class AgentInitializationError(AgentError):
    """Raised when an agent fails to initialize.

    TODO: Add initialization step that failed
    """
    pass


class AgentExecutionError(AgentError):
    """Raised when an agent fails during task execution.

    TODO: Add task context and partial results
    """
    pass


class AgentTimeoutError(AgentError):
    """Raised when agent execution exceeds time limit.

    TODO: Add timeout duration and progress info
    """
    pass


class MemoryError(SigilError):
    """Base exception for memory system errors.

    TODO: Add memory layer identification
    """

    def __init__(self, message: str, layer: Optional[str] = None, **kwargs: Any) -> None:
        super().__init__(message, code="MEMORY_ERROR", **kwargs)
        self.layer = layer


class MemoryWriteError(MemoryError):
    """Raised when writing to memory fails.

    TODO: Add item details that failed to write
    """
    pass


class MemoryRetrievalError(MemoryError):
    """Raised when retrieving from memory fails.

    TODO: Add query details
    """
    pass


class ReasoningError(SigilError):
    """Base exception for reasoning and strategy errors.

    TODO: Add strategy context
    """

    def __init__(self, message: str, strategy: Optional[str] = None, **kwargs: Any) -> None:
        super().__init__(message, code="REASONING_ERROR", **kwargs)
        self.strategy = strategy


class StrategyNotFoundError(ReasoningError):
    """Raised when a requested strategy is not available.

    TODO: Add available strategies list
    """
    pass


class PlanExecutionError(SigilError):
    """Raised when plan step execution fails.

    This exception is raised when a step in a plan cannot be completed.
    It includes information about the plan, the failing step, and the cause.

    Attributes:
        plan_id: The ID of the plan being executed.
        step_index: The index of the step that failed.
        step_name: The name/description of the failing step.
    """

    def __init__(
        self,
        message: str,
        plan_id: Optional[str] = None,
        step_index: Optional[int] = None,
        step_name: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(message, code="PLAN_EXECUTION_ERROR", **kwargs)
        self.plan_id = plan_id
        self.step_index = step_index
        self.step_name = step_name


class RoutingError(SigilError):
    """Raised when request routing fails.

    This exception is raised when the router cannot determine how to
    handle a request, either because no handler matches or multiple
    handlers conflict.

    Attributes:
        intent: The detected intent (if any).
        available_handlers: List of available handler names.
    """

    def __init__(
        self,
        message: str,
        intent: Optional[str] = None,
        available_handlers: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(message, code="ROUTING_ERROR", **kwargs)
        self.intent = intent
        self.available_handlers = available_handlers or []


class ContractError(SigilError):
    """Base exception for contract verification errors.

    Attributes:
        contract_id: The ID of the contract that failed.
        deliverable: The deliverable that failed validation (if applicable).
    """

    def __init__(
        self,
        message: str,
        contract_id: Optional[str] = None,
        deliverable: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(message, code="CONTRACT_ERROR", **kwargs)
        self.contract_id = contract_id
        self.deliverable = deliverable


class ContractValidationError(ContractError):
    """Raised when output fails contract validation.

    Also known as ContractViolation in some contexts.
    """
    pass


# Alias for backwards compatibility
ContractViolation = ContractValidationError


class ToolError(SigilError):
    """Base exception for tool execution errors.

    TODO: Add tool identification and parameters
    """

    def __init__(self, message: str, tool_name: Optional[str] = None, **kwargs: Any) -> None:
        super().__init__(message, code="TOOL_ERROR", **kwargs)
        self.tool_name = tool_name


class ToolNotFoundError(ToolError):
    """Raised when a requested tool is not available.

    TODO: Add available tools list
    """
    pass


class ToolExecutionError(ToolError):
    """Raised when tool execution fails.

    TODO: Add execution context and partial results
    """
    pass


class MCPConnectionError(ToolError):
    """Raised when connection to MCP server fails.

    TODO: Add server details and connection parameters
    """
    pass


class EvolutionError(SigilError):
    """Base exception for self-evolution errors.

    TODO: Add evolution context and metrics
    """

    def __init__(self, message: str, optimization_step: Optional[int] = None, **kwargs: Any) -> None:
        super().__init__(message, code="EVOLUTION_ERROR", **kwargs)
        self.optimization_step = optimization_step


class OptimizationError(EvolutionError):
    """Raised when TextGrad optimization fails.

    TODO: Add gradient information and loss values
    """
    pass


class PromptMutationError(EvolutionError):
    """Raised when prompt evolution fails.

    TODO: Add mutation details and original prompt
    """
    pass


# =============================================================================
# Token Budget Exceptions
# =============================================================================


class TokenBudgetExceeded(SigilError):
    """Raised when token usage exceeds the allocated budget.

    This exception is raised when an operation would cause the token
    usage to exceed the configured budget limits. It includes detailed
    information about current usage and budget limits.

    Attributes:
        current_input_tokens: Current cumulative input tokens used
        current_output_tokens: Current cumulative output tokens used
        current_total_tokens: Current total tokens (input + output)
        max_input_tokens: Maximum allowed input tokens
        max_output_tokens: Maximum allowed output tokens
        max_total_tokens: Maximum allowed total tokens
        utilization: Current budget utilization (0.0 to 1.0+)
        exceeded_limit: Which limit was exceeded ('input', 'output', or 'total')
    """

    def __init__(
        self,
        message: str,
        current_input_tokens: int = 0,
        current_output_tokens: int = 0,
        max_input_tokens: int = 0,
        max_output_tokens: int = 0,
        max_total_tokens: int = 0,
        exceeded_limit: str = "total",
        **kwargs: Any,
    ) -> None:
        context = {
            "current_input_tokens": current_input_tokens,
            "current_output_tokens": current_output_tokens,
            "current_total_tokens": current_input_tokens + current_output_tokens,
            "max_input_tokens": max_input_tokens,
            "max_output_tokens": max_output_tokens,
            "max_total_tokens": max_total_tokens,
            "exceeded_limit": exceeded_limit,
        }

        # Calculate utilization
        if exceeded_limit == "input" and max_input_tokens > 0:
            utilization = current_input_tokens / max_input_tokens
        elif exceeded_limit == "output" and max_output_tokens > 0:
            utilization = current_output_tokens / max_output_tokens
        elif max_total_tokens > 0:
            utilization = (current_input_tokens + current_output_tokens) / max_total_tokens
        else:
            utilization = 0.0

        context["utilization"] = utilization

        super().__init__(
            message,
            code="TOKEN_BUDGET_EXCEEDED",
            context=context,
            recoverable=False,
            **kwargs,
        )

        self.current_input_tokens = current_input_tokens
        self.current_output_tokens = current_output_tokens
        self.current_total_tokens = current_input_tokens + current_output_tokens
        self.max_input_tokens = max_input_tokens
        self.max_output_tokens = max_output_tokens
        self.max_total_tokens = max_total_tokens
        self.utilization = utilization
        self.exceeded_limit = exceeded_limit

    def __str__(self) -> str:
        return (
            f"[{self.code}] {self.message} "
            f"(used {self.current_total_tokens}/{self.max_total_tokens} tokens, "
            f"{self.utilization:.1%} utilization, exceeded: {self.exceeded_limit})"
        )


class TokenBudgetWarning(UserWarning):
    """Warning issued when token usage approaches budget limits.

    This is a warning class (not an exception) that can be issued
    when token usage crosses the configured warning threshold.
    It allows the application to continue while alerting about
    approaching limits.

    Attributes:
        current_input_tokens: Current cumulative input tokens used
        current_output_tokens: Current cumulative output tokens used
        current_total_tokens: Current total tokens (input + output)
        max_total_tokens: Maximum allowed total tokens
        utilization: Current budget utilization (0.0 to 1.0)
        threshold: Warning threshold that was crossed

    Example:
        >>> import warnings
        >>> from sigil.core.exceptions import TokenBudgetWarning
        >>> warnings.warn(
        ...     TokenBudgetWarning(
        ...         "Approaching token budget limit",
        ...         current_input_tokens=3500,
        ...         current_output_tokens=1500,
        ...         max_total_tokens=6000,
        ...         utilization=0.83,
        ...         threshold=0.8
        ...     )
        ... )
    """

    def __init__(
        self,
        message: str,
        current_input_tokens: int = 0,
        current_output_tokens: int = 0,
        max_total_tokens: int = 0,
        utilization: float = 0.0,
        threshold: float = 0.8,
    ) -> None:
        self.message = message
        self.current_input_tokens = current_input_tokens
        self.current_output_tokens = current_output_tokens
        self.current_total_tokens = current_input_tokens + current_output_tokens
        self.max_total_tokens = max_total_tokens
        self.utilization = utilization
        self.threshold = threshold
        super().__init__(self._format_message())

    def _format_message(self) -> str:
        """Format the warning message with context."""
        return (
            f"{self.message} "
            f"(used {self.current_total_tokens}/{self.max_total_tokens} tokens, "
            f"{self.utilization:.1%} utilization, threshold: {self.threshold:.0%})"
        )
