"""Telemetry module for Sigil v2 framework.

This module implements observability and monitoring:
- Structured logging with context
- Metrics collection and export
- Distributed tracing
- Performance profiling
- Token budget tracking and management

Key Components:
    - TokenBudget: Immutable configuration for token limits
    - TokenTracker: Tracks cumulative token usage across calls
    - TokenMetrics: Per-call token usage with timestamps
    - TelemetryManager: Coordinates all telemetry (TODO)
    - Logger: Structured logging with context (TODO)
    - MetricsCollector: Collects and exports metrics (TODO)
    - Tracer: Distributed tracing support (TODO)

Supported Backends:
    - OpenTelemetry
    - Prometheus
    - DataDog
    - CloudWatch

Example:
    >>> from sigil.telemetry import TokenBudget, TokenTracker, TokenMetrics
    >>> budget = TokenBudget(
    ...     max_input_tokens=4000,
    ...     max_output_tokens=2000,
    ...     max_total_tokens=10000
    ... )
    >>> tracker = TokenTracker()
    >>> tracker.record_usage(1000, 500)
    >>> tracker.is_within_budget(budget)
    True
"""

from sigil.telemetry.tokens import (
    TokenBudget,
    TokenTracker,
    TokenMetrics,
    TokenCallRecord,
)

__all__ = [
    "TokenBudget",
    "TokenTracker",
    "TokenMetrics",
    "TokenCallRecord",
]

# TODO: Export TelemetryManager once implemented
# TODO: Export logging and metrics utilities once implemented
