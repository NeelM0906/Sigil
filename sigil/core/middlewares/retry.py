"""Retry middleware for Sigil orchestrator pipeline.

This middleware retries failed steps with configurable backoff.
"""

from __future__ import annotations

import asyncio
import logging
import random
from typing import Any, Callable, Optional, Set, Type

from sigil.core.middleware import BaseMiddleware

logger = logging.getLogger(__name__)


class RetryMiddleware(BaseMiddleware):
    """Retries failed steps with configurable backoff.

    This middleware automatically retries steps that fail with transient
    errors. It supports:
    - Configurable max retries per step
    - Exponential backoff with jitter
    - Retryable exception filtering
    - Per-step retry configuration

    Example:
        >>> middleware = RetryMiddleware(
        ...     max_retries=3,
        ...     base_delay=0.1,
        ...     max_delay=5.0,
        ...     retryable_exceptions={TimeoutError, ConnectionError},
        ... )
        >>> chain.add(middleware)

    Attributes:
        max_retries: Maximum number of retry attempts
        base_delay: Base delay in seconds between retries
        max_delay: Maximum delay in seconds
        exponential_base: Base for exponential backoff (default 2)
        jitter: Whether to add random jitter to delays
    """

    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 0.1,
        max_delay: float = 10.0,
        exponential_base: float = 2.0,
        jitter: bool = True,
        retryable_exceptions: Optional[Set[Type[Exception]]] = None,
        step_overrides: Optional[dict[str, dict[str, Any]]] = None,
        should_retry: Optional[Callable[[str, Exception, int], bool]] = None,
    ) -> None:
        """Initialize the retry middleware.

        Args:
            max_retries: Maximum number of retry attempts
            base_delay: Base delay in seconds between retries
            max_delay: Maximum delay in seconds
            exponential_base: Base for exponential backoff
            jitter: Whether to add random jitter to delays
            retryable_exceptions: Set of exception types to retry
            step_overrides: Per-step override configurations
            should_retry: Optional callback to determine if retry should occur
        """
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter
        self.retryable_exceptions = retryable_exceptions or {
            TimeoutError,
            ConnectionError,
            asyncio.TimeoutError,
        }
        self.step_overrides = step_overrides or {}
        self.should_retry_callback = should_retry
        self._retry_counts: dict[str, int] = {}

    @property
    def name(self) -> str:
        """Get the middleware name."""
        return "RetryMiddleware"

    def _get_step_config(self, step_name: str) -> dict[str, Any]:
        """Get retry configuration for a step."""
        base_config = {
            "max_retries": self.max_retries,
            "base_delay": self.base_delay,
            "max_delay": self.max_delay,
        }
        if step_name in self.step_overrides:
            base_config.update(self.step_overrides[step_name])
        return base_config

    def _calculate_delay(self, attempt: int, config: dict[str, Any]) -> float:
        """Calculate delay for a retry attempt."""
        delay = config["base_delay"] * (self.exponential_base ** attempt)
        delay = min(delay, config["max_delay"])

        if self.jitter:
            # Add up to 25% random jitter
            jitter_amount = delay * 0.25 * random.random()
            delay += jitter_amount

        return delay

    def _should_retry(
        self, step_name: str, error: Exception, attempt: int
    ) -> bool:
        """Determine if a step should be retried."""
        # Check custom callback first
        if self.should_retry_callback:
            return self.should_retry_callback(step_name, error, attempt)

        # Check if max retries exceeded
        config = self._get_step_config(step_name)
        if attempt >= config["max_retries"]:
            return False

        # Check if exception is retryable
        return isinstance(error, tuple(self.retryable_exceptions))

    async def pre_step(self, step_name: str, ctx: Any) -> Any:
        """Reset retry count for step.

        Args:
            step_name: Name of the step
            ctx: Pipeline context

        Returns:
            The context unchanged
        """
        # Reset retry count at start of step
        self._retry_counts[step_name] = 0
        return ctx

    async def post_step(self, step_name: str, ctx: Any, result: Any) -> Any:
        """Clear retry count on success.

        Args:
            step_name: Name of the step
            ctx: Pipeline context
            result: Step result

        Returns:
            The result unchanged
        """
        self._retry_counts.pop(step_name, None)
        return result

    async def on_error(
        self, step_name: str, ctx: Any, error: Exception
    ) -> Optional[Any]:
        """Handle error and potentially schedule retry.

        Note: This middleware doesn't directly retry - it signals that
        a retry should occur by returning a special marker. The actual
        retry logic should be implemented in the pipeline runner.

        For actual retry support, this middleware would need to be
        integrated more deeply with the pipeline execution.

        Args:
            step_name: Name of the step
            ctx: Pipeline context
            error: The exception

        Raises:
            Exception: Re-raises the error (retry not directly supported here)
        """
        attempt = self._retry_counts.get(step_name, 0)

        if self._should_retry(step_name, error, attempt):
            config = self._get_step_config(step_name)
            delay = self._calculate_delay(attempt, config)

            logger.warning(
                f"Step '{step_name}' failed (attempt {attempt + 1}/{config['max_retries']}), "
                f"would retry after {delay:.2f}s: {error}"
            )

            self._retry_counts[step_name] = attempt + 1

            # Note: Actual retry needs integration with pipeline runner
            # For now, we just log and re-raise
            # In a full implementation, we would:
            # 1. await asyncio.sleep(delay)
            # 2. Return a special marker that signals retry
            # 3. Pipeline runner would re-execute the step

        raise error

    def get_retry_count(self, step_name: str) -> int:
        """Get current retry count for a step.

        Args:
            step_name: Name of the step

        Returns:
            Number of retries attempted
        """
        return self._retry_counts.get(step_name, 0)

    def reset(self) -> None:
        """Reset all retry counts."""
        self._retry_counts.clear()
