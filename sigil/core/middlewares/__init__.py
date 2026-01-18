"""Built-in middleware implementations for Sigil orchestrator.

This module provides common middleware implementations that can be used
with the Sigil orchestrator pipeline:

- LoggingMiddleware: Logs step entry, exit, and timing
- MetricsMiddleware: Collects timing and success metrics
- TracingMiddleware: Adds correlation IDs and distributed tracing
- RetryMiddleware: Retries failed steps with configurable backoff
- CircuitBreakerMiddleware: Prevents cascade failures
- ContextValidationMiddleware: Validates context size before LLM calls
- OffloadingMiddleware: Auto-offloads large tool results to files
- SummarizationMiddleware: Auto-summarizes long conversation history

Example:
    >>> from sigil.core.middlewares import LoggingMiddleware, MetricsMiddleware
    >>> from sigil.core.middleware import MiddlewareChain
    >>>
    >>> chain = MiddlewareChain()
    >>> chain.add(LoggingMiddleware(log_level=logging.INFO))
    >>> chain.add(MetricsMiddleware())

Example with offloading:
    >>> from sigil.core.middlewares import OffloadingMiddleware
    >>> chain.add(OffloadingMiddleware(threshold_chars=80000))

Example with summarization:
    >>> from sigil.core.middlewares import SummarizationMiddleware
    >>> chain.add(SummarizationMiddleware(threshold_tokens=120000))
"""

from sigil.core.middlewares.logging import LoggingMiddleware
from sigil.core.middlewares.metrics import MetricsMiddleware
from sigil.core.middlewares.tracing import TracingMiddleware
from sigil.core.middlewares.retry import RetryMiddleware
from sigil.core.middlewares.circuit_breaker import CircuitBreakerMiddleware
from sigil.core.middlewares.validation import ContextValidationMiddleware
from sigil.core.middlewares.offloading import (
    OffloadingMiddleware,
    ToolResultStore,
    create_read_tool_result_tool,
)
from sigil.core.middlewares.summarization import (
    SummarizationMiddleware,
    ConversationSummarizer,
    SummarizationResult,
    Message,
)

__all__ = [
    "LoggingMiddleware",
    "MetricsMiddleware",
    "TracingMiddleware",
    "RetryMiddleware",
    "CircuitBreakerMiddleware",
    "ContextValidationMiddleware",
    "OffloadingMiddleware",
    "ToolResultStore",
    "create_read_tool_result_tool",
    "SummarizationMiddleware",
    "ConversationSummarizer",
    "SummarizationResult",
    "Message",
]
