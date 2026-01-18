"""Built-in middleware implementations for Sigil orchestrator.

This module provides common middleware implementations that can be used
with the Sigil orchestrator pipeline:

- LoggingMiddleware: Logs step entry, exit, and timing
- MetricsMiddleware: Collects timing and success metrics
- TracingMiddleware: Adds correlation IDs and distributed tracing
- RetryMiddleware: Retries failed steps with configurable backoff
- CircuitBreakerMiddleware: Prevents cascade failures
- CachingMiddleware: Caches step results for repeated requests
- ValidationMiddleware: Validates step inputs and outputs

Example:
    >>> from sigil.core.middlewares import LoggingMiddleware, MetricsMiddleware
    >>> from sigil.core.middleware import MiddlewareChain
    >>>
    >>> chain = MiddlewareChain()
    >>> chain.add(LoggingMiddleware(log_level=logging.INFO))
    >>> chain.add(MetricsMiddleware())
"""

from sigil.core.middlewares.logging import LoggingMiddleware
from sigil.core.middlewares.metrics import MetricsMiddleware
from sigil.core.middlewares.tracing import TracingMiddleware
from sigil.core.middlewares.retry import RetryMiddleware
from sigil.core.middlewares.circuit_breaker import CircuitBreakerMiddleware

__all__ = [
    "LoggingMiddleware",
    "MetricsMiddleware",
    "TracingMiddleware",
    "RetryMiddleware",
    "CircuitBreakerMiddleware",
]
