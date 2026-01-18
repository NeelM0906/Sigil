"""Metrics middleware for Sigil orchestrator pipeline.

This middleware collects timing and success metrics for each step.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Optional

from sigil.core.middleware import BaseMiddleware


@dataclass
class StepMetrics:
    """Metrics for a single step.

    Attributes:
        calls: Number of times the step was called
        successes: Number of successful completions
        errors: Number of errors
        total_time_ms: Total execution time in milliseconds
        min_time_ms: Minimum execution time
        max_time_ms: Maximum execution time
    """

    calls: int = 0
    successes: int = 0
    errors: int = 0
    total_time_ms: float = 0.0
    min_time_ms: float = float("inf")
    max_time_ms: float = 0.0

    @property
    def avg_time_ms(self) -> float:
        """Average execution time in milliseconds."""
        if self.calls == 0:
            return 0.0
        return self.total_time_ms / self.calls

    @property
    def error_rate(self) -> float:
        """Error rate as a fraction."""
        if self.calls == 0:
            return 0.0
        return self.errors / self.calls

    @property
    def success_rate(self) -> float:
        """Success rate as a fraction."""
        if self.calls == 0:
            return 0.0
        return self.successes / self.calls

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "calls": self.calls,
            "successes": self.successes,
            "errors": self.errors,
            "error_rate": self.error_rate,
            "success_rate": self.success_rate,
            "total_time_ms": self.total_time_ms,
            "avg_time_ms": self.avg_time_ms,
            "min_time_ms": self.min_time_ms if self.min_time_ms != float("inf") else 0.0,
            "max_time_ms": self.max_time_ms,
        }


class MetricsMiddleware(BaseMiddleware):
    """Collects timing and success metrics for each step.

    This middleware tracks:
    - Number of calls, successes, and errors per step
    - Execution time statistics (total, average, min, max)
    - Overall pipeline metrics

    Example:
        >>> middleware = MetricsMiddleware()
        >>> chain.add(middleware)
        >>>
        >>> # After some requests...
        >>> summary = middleware.get_summary()
        >>> print(summary["route"]["avg_time_ms"])

    The metrics can be exported to monitoring systems or logged
    periodically.
    """

    def __init__(self) -> None:
        """Initialize the metrics middleware."""
        self._metrics: dict[str, StepMetrics] = {}
        self._timers: dict[str, float] = {}
        self._total_requests: int = 0
        self._total_time_ms: float = 0.0

    @property
    def name(self) -> str:
        """Get the middleware name."""
        return "MetricsMiddleware"

    def _ensure_step_metrics(self, step_name: str) -> StepMetrics:
        """Ensure metrics exist for a step."""
        if step_name not in self._metrics:
            self._metrics[step_name] = StepMetrics()
        return self._metrics[step_name]

    async def pre_step(self, step_name: str, ctx: Any) -> Any:
        """Start timing for a step.

        Args:
            step_name: Name of the step
            ctx: Pipeline context

        Returns:
            The context unchanged
        """
        self._timers[step_name] = time.perf_counter()
        metrics = self._ensure_step_metrics(step_name)
        metrics.calls += 1
        return ctx

    async def post_step(self, step_name: str, ctx: Any, result: Any) -> Any:
        """Record successful completion and timing.

        Args:
            step_name: Name of the step
            ctx: Pipeline context
            result: Step result

        Returns:
            The result unchanged
        """
        elapsed_ms = (
            time.perf_counter() - self._timers.get(step_name, time.perf_counter())
        ) * 1000

        metrics = self._ensure_step_metrics(step_name)
        metrics.successes += 1
        metrics.total_time_ms += elapsed_ms
        metrics.min_time_ms = min(metrics.min_time_ms, elapsed_ms)
        metrics.max_time_ms = max(metrics.max_time_ms, elapsed_ms)

        return result

    async def on_error(
        self, step_name: str, ctx: Any, error: Exception
    ) -> Optional[Any]:
        """Record error.

        Args:
            step_name: Name of the step
            ctx: Pipeline context
            error: The exception

        Raises:
            Exception: Always re-raises the error
        """
        elapsed_ms = (
            time.perf_counter() - self._timers.get(step_name, time.perf_counter())
        ) * 1000

        metrics = self._ensure_step_metrics(step_name)
        metrics.errors += 1
        metrics.total_time_ms += elapsed_ms

        raise error

    def get_step_metrics(self, step_name: str) -> Optional[StepMetrics]:
        """Get metrics for a specific step.

        Args:
            step_name: Name of the step

        Returns:
            StepMetrics or None if not found
        """
        return self._metrics.get(step_name)

    def get_summary(self) -> dict[str, dict[str, Any]]:
        """Get summary of all step metrics.

        Returns:
            Dictionary mapping step names to metric dictionaries
        """
        return {name: metrics.to_dict() for name, metrics in self._metrics.items()}

    def get_total_calls(self) -> int:
        """Get total number of step calls across all steps."""
        return sum(m.calls for m in self._metrics.values())

    def get_total_errors(self) -> int:
        """Get total number of errors across all steps."""
        return sum(m.errors for m in self._metrics.values())

    def reset(self) -> None:
        """Reset all metrics."""
        self._metrics.clear()
        self._timers.clear()
        self._total_requests = 0
        self._total_time_ms = 0.0
