"""Middleware architecture for Sigil orchestrator pipeline.

This module provides a flexible middleware system that allows:
- Pre/post hooks around each pipeline step
- Per-request middleware configuration
- Step reordering and skipping
- Error recovery middleware

The middleware architecture maintains backward compatibility - the default
behavior with no middleware is identical to the previous hardcoded pipeline.

Key Components:
    StepName: Enum of standard pipeline step names
    SigilMiddleware: Protocol defining the middleware interface
    BaseMiddleware: Base class for middleware implementations
    MiddlewareChain: Ordered chain of middleware to execute
    StepDefinition: Definition of a pipeline step
    StepRegistry: Registry for pipeline steps with ordering support
    MiddlewareConfig: Per-request middleware configuration
    PipelineRunner: Executes the pipeline with middleware support

Example:
    >>> from sigil.core.middleware import (
    ...     MiddlewareChain,
    ...     BaseMiddleware,
    ...     StepRegistry,
    ...     StepName,
    ... )
    >>>
    >>> # Create a logging middleware
    >>> class MyLoggingMiddleware(BaseMiddleware):
    ...     async def pre_step(self, step_name, ctx):
    ...         print(f"Starting: {step_name}")
    ...         return ctx
    ...
    >>> # Create a chain with the middleware
    >>> chain = MiddlewareChain()
    >>> chain.add(MyLoggingMiddleware())
"""

from __future__ import annotations

from abc import ABC
from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Coroutine,
    Optional,
    Protocol,
    runtime_checkable,
)
from enum import Enum
import logging
import time

if TYPE_CHECKING:
    from sigil.orchestrator import PipelineContext

logger = logging.getLogger(__name__)


# =============================================================================
# Step Names Enum
# =============================================================================


class StepName(str, Enum):
    """Standard pipeline step names.

    These are the default steps in the Sigil orchestrator pipeline.
    Custom steps can be added using the StepRegistry.

    Attributes:
        ROUTE: Route the request to determine intent and complexity
        GROUND: Gather information and identify gaps
        PLAN: Generate execution plan if needed
        ASSEMBLE: Assemble context from grounding/memory/history
        EXECUTE: Execute plan steps or fall back to reasoning
        VALIDATE: Validate output against contract
    """

    ROUTE = "route"
    GROUND = "ground"
    PLAN = "plan"
    ASSEMBLE = "assemble"
    EXECUTE = "execute"
    VALIDATE = "validate"


# =============================================================================
# Middleware Protocol and Base Class
# =============================================================================


@runtime_checkable
class SigilMiddleware(Protocol):
    """Protocol for pipeline middleware.

    Middleware can intercept pipeline steps at three points:
    - pre_step: Called before a step executes, can modify context
    - post_step: Called after a step executes, can modify result
    - on_error: Called when a step raises an exception

    Example:
        >>> class MyMiddleware:
        ...     async def pre_step(self, step_name: str, ctx: Any) -> Any:
        ...         print(f"Before {step_name}")
        ...         return ctx
        ...
        ...     async def post_step(
        ...         self, step_name: str, ctx: Any, result: Any
        ...     ) -> Any:
        ...         print(f"After {step_name}")
        ...         return result
        ...
        ...     async def on_error(
        ...         self, step_name: str, ctx: Any, error: Exception
        ...     ) -> Optional[Any]:
        ...         print(f"Error in {step_name}: {error}")
        ...         raise error
    """

    @property
    def name(self) -> str:
        """Get the middleware name for identification."""
        ...

    async def pre_step(
        self,
        step_name: str,
        ctx: Any,
    ) -> Any:
        """Called before a step executes.

        Args:
            step_name: Name of the step about to execute
            ctx: Pipeline context

        Returns:
            Modified context (or original if unchanged)
        """
        ...

    async def post_step(
        self,
        step_name: str,
        ctx: Any,
        result: Any,
    ) -> Any:
        """Called after a step executes.

        Args:
            step_name: Name of the step that executed
            ctx: Pipeline context
            result: Result from the step (usually None for pipeline steps)

        Returns:
            Modified result (or original if unchanged)
        """
        ...

    async def on_error(
        self,
        step_name: str,
        ctx: Any,
        error: Exception,
    ) -> Optional[Any]:
        """Called when a step raises an exception.

        Args:
            step_name: Name of the step that failed
            ctx: Pipeline context
            error: The exception that was raised

        Returns:
            Recovery value if middleware can recover, or re-raise the error

        Raises:
            Exception: Re-raise the error if cannot recover
        """
        ...


class BaseMiddleware(ABC):
    """Base class for middleware implementations.

    Provides default implementations for all middleware hooks that simply
    pass through without modification. Subclasses can override specific
    hooks as needed.

    Example:
        >>> class TimingMiddleware(BaseMiddleware):
        ...     def __init__(self):
        ...         self._timers = {}
        ...
        ...     async def pre_step(self, step_name: str, ctx: Any) -> Any:
        ...         self._timers[step_name] = time.time()
        ...         return ctx
        ...
        ...     async def post_step(
        ...         self, step_name: str, ctx: Any, result: Any
        ...     ) -> Any:
        ...         elapsed = time.time() - self._timers.get(step_name, time.time())
        ...         print(f"{step_name} took {elapsed:.3f}s")
        ...         return result
    """

    @property
    def name(self) -> str:
        """Get the middleware name.

        Returns:
            The class name by default
        """
        return self.__class__.__name__

    async def pre_step(self, step_name: str, ctx: Any) -> Any:
        """Default pre_step passes through unchanged.

        Args:
            step_name: Name of the step about to execute
            ctx: Pipeline context

        Returns:
            The context unchanged
        """
        return ctx

    async def post_step(self, step_name: str, ctx: Any, result: Any) -> Any:
        """Default post_step passes through unchanged.

        Args:
            step_name: Name of the step that executed
            ctx: Pipeline context
            result: Result from the step

        Returns:
            The result unchanged
        """
        return result

    async def on_error(
        self, step_name: str, ctx: Any, error: Exception
    ) -> Optional[Any]:
        """Default on_error re-raises the exception.

        Args:
            step_name: Name of the step that failed
            ctx: Pipeline context
            error: The exception that was raised

        Raises:
            Exception: Always re-raises the error
        """
        raise error


# =============================================================================
# Middleware Chain
# =============================================================================


@dataclass
class MiddlewareChain:
    """Ordered chain of middleware to execute.

    Middleware are executed in order for pre_step hooks, and in reverse
    order for post_step and on_error hooks (similar to onion model).

    Example:
        >>> chain = MiddlewareChain()
        >>> chain.add(LoggingMiddleware())
        >>> chain.add(MetricsMiddleware())
        >>>
        >>> # Pre-step: Logging -> Metrics
        >>> # Post-step: Metrics -> Logging
        >>> # On-error: Metrics -> Logging

    Attributes:
        middlewares: List of middleware in execution order
    """

    middlewares: list[SigilMiddleware] = field(default_factory=list)

    def add(self, middleware: SigilMiddleware) -> "MiddlewareChain":
        """Add a middleware to the end of the chain.

        Args:
            middleware: Middleware to add

        Returns:
            Self for method chaining
        """
        self.middlewares.append(middleware)
        return self

    def insert(self, index: int, middleware: SigilMiddleware) -> "MiddlewareChain":
        """Insert a middleware at a specific position.

        Args:
            index: Position to insert at
            middleware: Middleware to insert

        Returns:
            Self for method chaining
        """
        self.middlewares.insert(index, middleware)
        return self

    def remove(self, middleware_name: str) -> bool:
        """Remove a middleware by name.

        Args:
            middleware_name: Name of the middleware to remove

        Returns:
            True if middleware was found and removed
        """
        for i, mw in enumerate(self.middlewares):
            if mw.name == middleware_name:
                self.middlewares.pop(i)
                return True
        return False

    def clear(self) -> None:
        """Remove all middleware from the chain."""
        self.middlewares.clear()

    def __len__(self) -> int:
        """Return the number of middleware in the chain."""
        return len(self.middlewares)

    def __iter__(self):
        """Iterate over middleware in the chain."""
        return iter(self.middlewares)

    async def run_pre_step(self, step_name: str, ctx: Any) -> Any:
        """Run pre_step hooks for all middleware.

        Middleware are executed in order (first to last).

        Args:
            step_name: Name of the step about to execute
            ctx: Pipeline context

        Returns:
            Modified context after all middleware
        """
        for mw in self.middlewares:
            try:
                ctx = await mw.pre_step(step_name, ctx)
            except Exception as e:
                logger.warning(
                    f"Middleware {mw.name} pre_step failed for {step_name}: {e}"
                )
                # Continue with other middleware
        return ctx

    async def run_post_step(self, step_name: str, ctx: Any, result: Any) -> Any:
        """Run post_step hooks for all middleware.

        Middleware are executed in reverse order (last to first).

        Args:
            step_name: Name of the step that executed
            ctx: Pipeline context
            result: Result from the step

        Returns:
            Modified result after all middleware
        """
        for mw in reversed(self.middlewares):
            try:
                result = await mw.post_step(step_name, ctx, result)
            except Exception as e:
                logger.warning(
                    f"Middleware {mw.name} post_step failed for {step_name}: {e}"
                )
                # Continue with other middleware
        return result

    async def run_on_error(
        self, step_name: str, ctx: Any, error: Exception
    ) -> Optional[Any]:
        """Run on_error hooks for all middleware.

        Middleware are executed in reverse order. If a middleware returns
        a non-None value, that value is used as recovery and remaining
        middleware are skipped.

        Args:
            step_name: Name of the step that failed
            ctx: Pipeline context
            error: The exception that was raised

        Returns:
            Recovery value if any middleware can recover

        Raises:
            Exception: The original error if no middleware can recover
        """
        for mw in reversed(self.middlewares):
            try:
                recovery = await mw.on_error(step_name, ctx, error)
                if recovery is not None:
                    logger.info(
                        f"Middleware {mw.name} recovered from error in {step_name}"
                    )
                    return recovery
            except Exception:
                # Middleware re-raised or threw new exception, continue
                continue
        # No middleware recovered, re-raise original error
        raise error


# =============================================================================
# Step Definition and Registry
# =============================================================================


# Type alias for step handlers
StepHandler = Callable[["PipelineContext"], Coroutine[Any, Any, None]]


@dataclass
class StepDefinition:
    """Definition of a pipeline step.

    Attributes:
        name: Unique name for the step
        handler: Async function that executes the step
        enabled: Whether the step is enabled (default True)
        skip_on_error: Whether to skip this step if a previous step errored
        required: Whether this step is required (cannot be disabled)
    """

    name: str
    handler: StepHandler
    enabled: bool = True
    skip_on_error: bool = False
    required: bool = False


class StepRegistry:
    """Registry for pipeline steps with ordering support.

    The registry maintains an ordered list of steps and provides methods
    for adding, removing, enabling/disabling, and reordering steps.

    Example:
        >>> registry = StepRegistry()
        >>> registry.register("route", route_handler)
        >>> registry.register("plan", plan_handler, after="route")
        >>> registry.register("execute", execute_handler, after="plan")
        >>>
        >>> # Disable a step
        >>> registry.disable("plan")
        >>>
        >>> # Get enabled steps in order
        >>> for step in registry.get_enabled_steps():
        ...     await step.handler(ctx)
    """

    def __init__(self) -> None:
        """Initialize an empty step registry."""
        self._steps: dict[str, StepDefinition] = {}
        self._order: list[str] = []

    def register(
        self,
        name: str,
        handler: StepHandler,
        after: Optional[str] = None,
        before: Optional[str] = None,
        enabled: bool = True,
        skip_on_error: bool = False,
        required: bool = False,
    ) -> None:
        """Register a new step.

        Args:
            name: Unique name for the step
            handler: Async function that executes the step
            after: Name of step to insert after (optional)
            before: Name of step to insert before (optional)
            enabled: Whether the step is enabled
            skip_on_error: Whether to skip if previous step errored
            required: Whether this step cannot be disabled
        """
        self._steps[name] = StepDefinition(
            name=name,
            handler=handler,
            enabled=enabled,
            skip_on_error=skip_on_error,
            required=required,
        )

        if name not in self._order:
            if after and after in self._order:
                idx = self._order.index(after) + 1
                self._order.insert(idx, name)
            elif before and before in self._order:
                idx = self._order.index(before)
                self._order.insert(idx, name)
            else:
                self._order.append(name)

        logger.debug(f"Registered step '{name}' in pipeline")

    def unregister(self, name: str) -> bool:
        """Remove a step from the registry.

        Args:
            name: Name of the step to remove

        Returns:
            True if step was found and removed
        """
        if name in self._steps:
            if self._steps[name].required:
                raise ValueError(f"Cannot unregister required step '{name}'")
            del self._steps[name]
            self._order.remove(name)
            return True
        return False

    def enable(self, name: str) -> None:
        """Enable a step.

        Args:
            name: Name of the step to enable
        """
        if name in self._steps:
            self._steps[name].enabled = True
            logger.debug(f"Enabled step '{name}'")

    def disable(self, name: str) -> None:
        """Disable a step.

        Args:
            name: Name of the step to disable

        Raises:
            ValueError: If the step is required and cannot be disabled
        """
        if name in self._steps:
            if self._steps[name].required:
                raise ValueError(f"Cannot disable required step '{name}'")
            self._steps[name].enabled = False
            logger.debug(f"Disabled step '{name}'")

    def is_enabled(self, name: str) -> bool:
        """Check if a step is enabled.

        Args:
            name: Name of the step

        Returns:
            True if step exists and is enabled
        """
        return name in self._steps and self._steps[name].enabled

    def get_step(self, name: str) -> Optional[StepDefinition]:
        """Get a step by name.

        Args:
            name: Name of the step

        Returns:
            The StepDefinition or None if not found
        """
        return self._steps.get(name)

    def get_enabled_steps(self) -> list[StepDefinition]:
        """Get all enabled steps in order.

        Returns:
            List of enabled StepDefinitions in execution order
        """
        return [
            self._steps[name]
            for name in self._order
            if name in self._steps and self._steps[name].enabled
        ]

    def get_all_steps(self) -> list[StepDefinition]:
        """Get all steps in order (enabled and disabled).

        Returns:
            List of all StepDefinitions in execution order
        """
        return [self._steps[name] for name in self._order if name in self._steps]

    def reorder(self, order: list[str]) -> None:
        """Reorder steps to match the given order.

        Steps not in the new order list are appended at the end.

        Args:
            order: List of step names in desired order
        """
        valid = [n for n in order if n in self._steps]
        remaining = [n for n in self._order if n not in valid]
        self._order = valid + remaining
        logger.debug(f"Reordered steps: {self._order}")

    def move_after(self, name: str, after: str) -> None:
        """Move a step to be after another step.

        Args:
            name: Name of step to move
            after: Name of step to move after
        """
        if name in self._order and after in self._order:
            self._order.remove(name)
            idx = self._order.index(after) + 1
            self._order.insert(idx, name)

    def move_before(self, name: str, before: str) -> None:
        """Move a step to be before another step.

        Args:
            name: Name of step to move
            before: Name of step to move before
        """
        if name in self._order and before in self._order:
            self._order.remove(name)
            idx = self._order.index(before)
            self._order.insert(idx, name)

    @property
    def step_names(self) -> list[str]:
        """Get list of all step names in order.

        Returns:
            List of step names
        """
        return list(self._order)

    def __len__(self) -> int:
        """Return the number of registered steps."""
        return len(self._steps)

    def __contains__(self, name: str) -> bool:
        """Check if a step is registered."""
        return name in self._steps


# =============================================================================
# Per-Request Middleware Configuration
# =============================================================================


@dataclass
class MiddlewareConfig:
    """Per-request middleware configuration.

    Allows customizing middleware behavior for individual requests
    without modifying the global orchestrator configuration.

    Example:
        >>> config = MiddlewareConfig()
        >>> config.disabled_steps.add("validate")  # Skip validation
        >>> config.chain.add(DebugMiddleware())  # Add debug logging
        >>>
        >>> request = OrchestratorRequest(
        ...     message="test",
        ...     session_id="test",
        ...     context={"middleware_config": config}
        ... )

    Attributes:
        chain: Optional middleware chain for this request
        disabled_steps: Set of step names to skip for this request
        step_overrides: Dict mapping step names to override handlers
        metadata: Additional configuration metadata
    """

    chain: Optional[MiddlewareChain] = None
    disabled_steps: set[str] = field(default_factory=set)
    step_overrides: dict[str, StepHandler] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def disable_step(self, name: str) -> "MiddlewareConfig":
        """Disable a step for this request.

        Args:
            name: Name of step to disable

        Returns:
            Self for method chaining
        """
        self.disabled_steps.add(name)
        return self

    def enable_step(self, name: str) -> "MiddlewareConfig":
        """Re-enable a previously disabled step.

        Args:
            name: Name of step to enable

        Returns:
            Self for method chaining
        """
        self.disabled_steps.discard(name)
        return self

    def override_step(self, name: str, handler: StepHandler) -> "MiddlewareConfig":
        """Override a step handler for this request.

        Args:
            name: Name of step to override
            handler: New handler function

        Returns:
            Self for method chaining
        """
        self.step_overrides[name] = handler
        return self

    def add_middleware(self, middleware: SigilMiddleware) -> "MiddlewareConfig":
        """Add a middleware to this request's chain.

        Creates the chain if it doesn't exist.

        Args:
            middleware: Middleware to add

        Returns:
            Self for method chaining
        """
        if self.chain is None:
            self.chain = MiddlewareChain()
        self.chain.add(middleware)
        return self


# =============================================================================
# Pipeline Runner
# =============================================================================


class PipelineRunner:
    """Executes the pipeline with middleware support.

    The PipelineRunner coordinates the execution of steps and middleware,
    handling:
    - Pre/post step hooks
    - Error handling and recovery
    - Step skipping and overrides
    - Per-request configuration

    Example:
        >>> runner = PipelineRunner(
        ...     registry=step_registry,
        ...     middleware_chain=chain,
        ... )
        >>> await runner.run(ctx)
    """

    def __init__(
        self,
        registry: StepRegistry,
        middleware_chain: Optional[MiddlewareChain] = None,
    ) -> None:
        """Initialize the pipeline runner.

        Args:
            registry: Registry of pipeline steps
            middleware_chain: Optional default middleware chain
        """
        self._registry = registry
        self._middleware_chain = middleware_chain or MiddlewareChain()

    @property
    def registry(self) -> StepRegistry:
        """Get the step registry."""
        return self._registry

    @property
    def middleware_chain(self) -> MiddlewareChain:
        """Get the middleware chain."""
        return self._middleware_chain

    def add_middleware(self, middleware: SigilMiddleware) -> None:
        """Add a middleware to the default chain.

        Args:
            middleware: Middleware to add
        """
        self._middleware_chain.add(middleware)

    async def run(
        self,
        ctx: "PipelineContext",
        config: Optional[MiddlewareConfig] = None,
    ) -> None:
        """Execute the pipeline with middleware.

        Args:
            ctx: Pipeline context
            config: Optional per-request configuration
        """
        # Determine which chain to use
        chain = self._middleware_chain
        if config and config.chain:
            # Merge request-specific middleware with default chain
            merged_chain = MiddlewareChain()
            merged_chain.middlewares = list(self._middleware_chain.middlewares)
            for mw in config.chain.middlewares:
                merged_chain.add(mw)
            chain = merged_chain

        # Get steps to execute
        steps = self._registry.get_enabled_steps()

        previous_error = False
        for step in steps:
            # Check if step is disabled for this request
            if config and step.name in config.disabled_steps:
                logger.debug(f"Skipping disabled step '{step.name}'")
                continue

            # Check if we should skip due to previous error
            if previous_error and step.skip_on_error:
                logger.debug(f"Skipping step '{step.name}' due to previous error")
                continue

            # Get handler (may be overridden per-request)
            handler = step.handler
            if config and step.name in config.step_overrides:
                handler = config.step_overrides[step.name]

            try:
                # Pre-step middleware
                ctx = await chain.run_pre_step(step.name, ctx)

                # Execute step
                await handler(ctx)

                # Post-step middleware
                await chain.run_post_step(step.name, ctx, None)

            except Exception as e:
                previous_error = True
                logger.error(f"Error in step '{step.name}': {e}")

                # Try error recovery via middleware
                try:
                    recovery = await chain.run_on_error(step.name, ctx, e)
                    if recovery is not None:
                        logger.info(f"Recovered from error in '{step.name}'")
                        continue
                except Exception:
                    # Middleware couldn't recover, re-raise
                    raise


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Enums
    "StepName",
    # Protocol and base class
    "SigilMiddleware",
    "BaseMiddleware",
    # Chain
    "MiddlewareChain",
    # Steps
    "StepHandler",
    "StepDefinition",
    "StepRegistry",
    # Config
    "MiddlewareConfig",
    # Runner
    "PipelineRunner",
]
