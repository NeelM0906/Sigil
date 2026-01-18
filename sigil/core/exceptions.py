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
    ├── SigilMemoryError: Memory system failures
    │   ├── SigilMemoryWriteError: Failed to write to memory
    │   ├── SigilMemoryRetrievalError: Failed to retrieve from memory
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
    │   └── ServiceConnectionError: Failed to connect to external service
    └── EvolutionError: Self-evolution failures
        ├── OptimizationError: TextGrad optimization failed
        └── PromptMutationError: Prompt evolution failed

Features:
    - Complete exception hierarchy with specific error types
    - Context information included in each exception type
    - Error codes for programmatic handling
    - Optional telemetry integration via emit_event method
    - Structured logging support via to_log_dict method
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

    Methods:
        emit_event: Optional hook for telemetry integration
        to_log_dict: Returns structured dict for logging
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

    def emit_event(self, telemetry_client: Optional[Any] = None) -> None:
        """Optional hook for telemetry integration.

        Override this method or pass a telemetry client to emit error events
        to your observability platform.

        Args:
            telemetry_client: Optional client with an emit() method
        """
        if telemetry_client is not None and hasattr(telemetry_client, "emit"):
            telemetry_client.emit(self.to_log_dict())

    def to_log_dict(self) -> dict[str, Any]:
        """Return structured dict for logging.

        Returns:
            Dictionary with error details suitable for structured logging
        """
        return {
            "error_type": self.__class__.__name__,
            "error_code": self.code,
            "message": self.message,
            "recoverable": self.recoverable,
            "context": self.context,
        }


class ConfigurationError(SigilError):
    """Raised when configuration is invalid or missing.

    Attributes:
        config_key: The configuration key that caused the error
        validation_details: Details about why validation failed
    """

    def __init__(
        self,
        message: str,
        config_key: Optional[str] = None,
        validation_details: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        context = kwargs.pop("context", {}) or {}
        if config_key:
            context["config_key"] = config_key
        if validation_details:
            context["validation_details"] = validation_details
        super().__init__(message, code="CONFIG_ERROR", context=context, **kwargs)
        self.config_key = config_key
        self.validation_details = validation_details


class AgentError(SigilError):
    """Base exception for agent-related errors.

    Attributes:
        agent_id: Identifier of the agent that encountered the error
    """

    def __init__(self, message: str, agent_id: Optional[str] = None, **kwargs: Any) -> None:
        context = kwargs.pop("context", {}) or {}
        if agent_id:
            context["agent_id"] = agent_id
        super().__init__(message, code="AGENT_ERROR", context=context, **kwargs)
        self.agent_id = agent_id


class AgentInitializationError(AgentError):
    """Raised when an agent fails to initialize.

    Attributes:
        initialization_step: The step during initialization that failed
    """

    def __init__(
        self,
        message: str,
        agent_id: Optional[str] = None,
        initialization_step: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        context = kwargs.pop("context", {}) or {}
        if initialization_step:
            context["initialization_step"] = initialization_step
        super().__init__(message, agent_id=agent_id, context=context, **kwargs)
        self.initialization_step = initialization_step


class AgentExecutionError(AgentError):
    """Raised when an agent fails during task execution.

    Attributes:
        task_context: Context information about the task being executed
        partial_results: Any partial results obtained before failure
    """

    def __init__(
        self,
        message: str,
        agent_id: Optional[str] = None,
        task_context: Optional[dict[str, Any]] = None,
        partial_results: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        context = kwargs.pop("context", {}) or {}
        if task_context:
            context["task_context"] = task_context
        if partial_results is not None:
            context["partial_results"] = partial_results
        super().__init__(message, agent_id=agent_id, context=context, **kwargs)
        self.task_context = task_context
        self.partial_results = partial_results


class AgentTimeoutError(AgentError):
    """Raised when agent execution exceeds time limit.

    Attributes:
        timeout_seconds: The timeout duration that was exceeded
        progress: Description of progress made before timeout
    """

    def __init__(
        self,
        message: str,
        agent_id: Optional[str] = None,
        timeout_seconds: Optional[float] = None,
        progress: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        context = kwargs.pop("context", {}) or {}
        if timeout_seconds is not None:
            context["timeout_seconds"] = timeout_seconds
        if progress:
            context["progress"] = progress
        super().__init__(message, agent_id=agent_id, context=context, **kwargs)
        self.timeout_seconds = timeout_seconds
        self.progress = progress


class SigilMemoryError(SigilError):
    """Base exception for memory system errors.

    Attributes:
        layer: The memory layer that encountered the error (e.g., 'short_term', 'semantic')
    """

    def __init__(self, message: str, layer: Optional[str] = None, **kwargs: Any) -> None:
        context = kwargs.pop("context", {}) or {}
        if layer:
            context["layer"] = layer
        super().__init__(message, code="MEMORY_ERROR", context=context, **kwargs)
        self.layer = layer


class SigilMemoryWriteError(SigilMemoryError):
    """Raised when writing to memory fails.

    Attributes:
        item_details: Details about the item that failed to write
    """

    def __init__(
        self,
        message: str,
        layer: Optional[str] = None,
        item_details: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        context = kwargs.pop("context", {}) or {}
        if item_details:
            context["item_details"] = item_details
        super().__init__(message, layer=layer, context=context, **kwargs)
        self.item_details = item_details


class SigilMemoryRetrievalError(SigilMemoryError):
    """Raised when retrieving from memory fails.

    Attributes:
        query_details: Details about the query that failed
    """

    def __init__(
        self,
        message: str,
        layer: Optional[str] = None,
        query_details: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        context = kwargs.pop("context", {}) or {}
        if query_details:
            context["query_details"] = query_details
        super().__init__(message, layer=layer, context=context, **kwargs)
        self.query_details = query_details


# Aliases for backwards compatibility
MemoryError = SigilMemoryError
MemoryWriteError = SigilMemoryWriteError
MemoryRetrievalError = SigilMemoryRetrievalError


class ReasoningError(SigilError):
    """Base exception for reasoning and strategy errors.

    Attributes:
        strategy: The reasoning strategy that encountered the error
    """

    def __init__(self, message: str, strategy: Optional[str] = None, **kwargs: Any) -> None:
        context = kwargs.pop("context", {}) or {}
        if strategy:
            context["strategy"] = strategy
        super().__init__(message, code="REASONING_ERROR", context=context, **kwargs)
        self.strategy = strategy


class StrategyNotFoundError(ReasoningError):
    """Raised when a requested strategy is not available.

    Attributes:
        available_strategies: List of strategies that are available
    """

    def __init__(
        self,
        message: str,
        strategy: Optional[str] = None,
        available_strategies: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> None:
        context = kwargs.pop("context", {}) or {}
        if available_strategies:
            context["available_strategies"] = available_strategies
        super().__init__(message, strategy=strategy, context=context, **kwargs)
        self.available_strategies = available_strategies or []


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

    Attributes:
        tool_name: Name of the tool that encountered the error
        tool_parameters: Parameters passed to the tool
    """

    def __init__(
        self,
        message: str,
        tool_name: Optional[str] = None,
        tool_parameters: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        context = kwargs.pop("context", {}) or {}
        if tool_name:
            context["tool_name"] = tool_name
        if tool_parameters:
            context["tool_parameters"] = tool_parameters
        super().__init__(message, code="TOOL_ERROR", context=context, **kwargs)
        self.tool_name = tool_name
        self.tool_parameters = tool_parameters


class ToolNotFoundError(ToolError):
    """Raised when a requested tool is not available.

    Attributes:
        available_tools: List of tools that are available
    """

    def __init__(
        self,
        message: str,
        tool_name: Optional[str] = None,
        available_tools: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> None:
        context = kwargs.pop("context", {}) or {}
        if available_tools:
            context["available_tools"] = available_tools
        super().__init__(message, tool_name=tool_name, context=context, **kwargs)
        self.available_tools = available_tools or []


class ToolExecutionError(ToolError):
    """Raised when tool execution fails.

    Attributes:
        execution_context: Context information about the execution
        partial_results: Any partial results obtained before failure
    """

    def __init__(
        self,
        message: str,
        tool_name: Optional[str] = None,
        execution_context: Optional[dict[str, Any]] = None,
        partial_results: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        context = kwargs.pop("context", {}) or {}
        if execution_context:
            context["execution_context"] = execution_context
        if partial_results is not None:
            context["partial_results"] = partial_results
        super().__init__(message, tool_name=tool_name, context=context, **kwargs)
        self.execution_context = execution_context
        self.partial_results = partial_results


class ServiceConnectionError(ToolError):
    """Raised when connection to an external service fails.

    This includes web search services, APIs, etc.
    """
    pass


# Backward compatibility alias
MCPConnectionError = ServiceConnectionError


class EvolutionError(SigilError):
    """Base exception for self-evolution errors.

    Attributes:
        optimization_step: The step number in the optimization process
        evolution_context: Context information about the evolution process
        metrics: Metrics collected during evolution
    """

    def __init__(
        self,
        message: str,
        optimization_step: Optional[int] = None,
        evolution_context: Optional[dict[str, Any]] = None,
        metrics: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        context = kwargs.pop("context", {}) or {}
        if optimization_step is not None:
            context["optimization_step"] = optimization_step
        if evolution_context:
            context["evolution_context"] = evolution_context
        if metrics:
            context["metrics"] = metrics
        super().__init__(message, code="EVOLUTION_ERROR", context=context, **kwargs)
        self.optimization_step = optimization_step
        self.evolution_context = evolution_context
        self.metrics = metrics


class OptimizationError(EvolutionError):
    """Raised when TextGrad optimization fails.

    Attributes:
        gradient_info: Information about the gradients computed
        loss_values: Loss values at the time of failure
    """

    def __init__(
        self,
        message: str,
        optimization_step: Optional[int] = None,
        gradient_info: Optional[dict[str, Any]] = None,
        loss_values: Optional[list[float]] = None,
        **kwargs: Any,
    ) -> None:
        context = kwargs.pop("context", {}) or {}
        if gradient_info:
            context["gradient_info"] = gradient_info
        if loss_values:
            context["loss_values"] = loss_values
        super().__init__(message, optimization_step=optimization_step, context=context, **kwargs)
        self.gradient_info = gradient_info
        self.loss_values = loss_values


class PromptMutationError(EvolutionError):
    """Raised when prompt evolution fails.

    Attributes:
        mutation_details: Details about the attempted mutation
        original_prompt: The original prompt before mutation
    """

    def __init__(
        self,
        message: str,
        optimization_step: Optional[int] = None,
        mutation_details: Optional[dict[str, Any]] = None,
        original_prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        context = kwargs.pop("context", {}) or {}
        if mutation_details:
            context["mutation_details"] = mutation_details
        if original_prompt:
            context["original_prompt"] = original_prompt
        super().__init__(message, optimization_step=optimization_step, context=context, **kwargs)
        self.mutation_details = mutation_details
        self.original_prompt = original_prompt


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
