"""Logging middleware for Sigil orchestrator pipeline.

This middleware logs step entry, exit, and timing information.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Optional

from sigil.core.middleware import BaseMiddleware

logger = logging.getLogger(__name__)


class LoggingMiddleware(BaseMiddleware):
    """Logs step entry, exit, and timing information.

    This middleware provides detailed logging of pipeline execution,
    including:
    - Step start with context information
    - Step completion with elapsed time
    - Error details on failure

    Example:
        >>> middleware = LoggingMiddleware(log_level=logging.INFO)
        >>> chain.add(middleware)

    Attributes:
        log_level: The logging level to use (default: DEBUG)
        include_context: Whether to log context details (default: False)
    """

    def __init__(
        self,
        log_level: int = logging.DEBUG,
        include_context: bool = False,
        logger_name: Optional[str] = None,
    ) -> None:
        """Initialize the logging middleware.

        Args:
            log_level: Logging level to use
            include_context: Whether to log context details
            logger_name: Optional custom logger name
        """
        self.log_level = log_level
        self.include_context = include_context
        self._logger = (
            logging.getLogger(logger_name) if logger_name else logger
        )
        self._timers: dict[str, float] = {}

    @property
    def name(self) -> str:
        """Get the middleware name."""
        return "LoggingMiddleware"

    async def pre_step(self, step_name: str, ctx: Any) -> Any:
        """Log step start.

        Args:
            step_name: Name of the step
            ctx: Pipeline context

        Returns:
            The context unchanged
        """
        self._timers[step_name] = time.perf_counter()

        message = f"[Pipeline] Starting step: {step_name}"
        if self.include_context and hasattr(ctx, "request"):
            message += f" (session={ctx.request.session_id})"

        self._logger.log(self.log_level, message)
        return ctx

    async def post_step(self, step_name: str, ctx: Any, result: Any) -> Any:
        """Log step completion with timing.

        Args:
            step_name: Name of the step
            ctx: Pipeline context
            result: Step result

        Returns:
            The result unchanged
        """
        elapsed = time.perf_counter() - self._timers.get(
            step_name, time.perf_counter()
        )
        elapsed_ms = elapsed * 1000

        message = f"[Pipeline] Completed step: {step_name} ({elapsed_ms:.2f}ms)"

        # Add context-specific info
        if step_name == "route" and hasattr(ctx, "route_decision") and ctx.route_decision:
            message += f" - intent={ctx.route_decision.intent.value}"
        elif step_name == "plan" and hasattr(ctx, "plan") and ctx.plan:
            message += f" - steps={len(ctx.plan.steps)}"

        self._logger.log(self.log_level, message)
        return result

    async def on_error(
        self, step_name: str, ctx: Any, error: Exception
    ) -> Optional[Any]:
        """Log step error.

        Args:
            step_name: Name of the step
            ctx: Pipeline context
            error: The exception

        Raises:
            Exception: Always re-raises the error
        """
        elapsed = time.perf_counter() - self._timers.get(
            step_name, time.perf_counter()
        )
        elapsed_ms = elapsed * 1000

        self._logger.error(
            f"[Pipeline] Error in step {step_name} ({elapsed_ms:.2f}ms): "
            f"{type(error).__name__}: {error}"
        )
        raise error
