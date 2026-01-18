"""Tracing middleware for Sigil orchestrator pipeline.

This middleware adds correlation IDs and distributed tracing support.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

from sigil.core.middleware import BaseMiddleware


@dataclass
class Span:
    """Represents a tracing span for a step.

    Attributes:
        span_id: Unique identifier for this span
        trace_id: Correlation ID for the entire request
        step_name: Name of the step
        start_time: Start timestamp
        end_time: End timestamp (None if not completed)
        status: "success", "error", or "pending"
        error: Error message if failed
        metadata: Additional span metadata
    """

    span_id: str
    trace_id: str
    step_name: str
    start_time: float
    end_time: Optional[float] = None
    status: str = "pending"
    error: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def duration_ms(self) -> Optional[float]:
        """Duration in milliseconds if completed."""
        if self.end_time is None:
            return None
        return (self.end_time - self.start_time) * 1000

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for export."""
        return {
            "span_id": self.span_id,
            "trace_id": self.trace_id,
            "step_name": self.step_name,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_ms": self.duration_ms,
            "status": self.status,
            "error": self.error,
            "metadata": self.metadata,
        }


class TracingMiddleware(BaseMiddleware):
    """Adds correlation IDs and distributed tracing support.

    This middleware creates spans for each step execution, allowing
    you to trace request flow through the pipeline. Spans can be
    exported to tracing systems like Jaeger, Zipkin, or OpenTelemetry.

    Example:
        >>> middleware = TracingMiddleware()
        >>> chain.add(middleware)
        >>>
        >>> # After processing a request...
        >>> trace = middleware.get_trace(trace_id)
        >>> for span in trace:
        ...     print(f"{span.step_name}: {span.duration_ms}ms")

    The middleware automatically extracts or generates trace IDs from
    the request context.
    """

    def __init__(
        self,
        trace_id_header: str = "x-trace-id",
        export_callback: Optional[callable] = None,
    ) -> None:
        """Initialize the tracing middleware.

        Args:
            trace_id_header: Header name to extract trace ID from
            export_callback: Optional callback to export completed spans
        """
        self.trace_id_header = trace_id_header
        self.export_callback = export_callback
        self._traces: dict[str, list[Span]] = {}
        self._active_spans: dict[str, Span] = {}

    @property
    def name(self) -> str:
        """Get the middleware name."""
        return "TracingMiddleware"

    def _get_trace_id(self, ctx: Any) -> str:
        """Extract or generate trace ID from context."""
        # Try to get from request correlation_id
        if hasattr(ctx, "request") and hasattr(ctx.request, "correlation_id"):
            return ctx.request.correlation_id or str(uuid.uuid4())

        # Try to get from metadata
        if hasattr(ctx, "request") and hasattr(ctx.request, "metadata"):
            trace_id = ctx.request.metadata.get(self.trace_id_header)
            if trace_id:
                return trace_id

        # Generate new trace ID
        return str(uuid.uuid4())

    def _create_span(self, trace_id: str, step_name: str, ctx: Any) -> Span:
        """Create a new span for a step."""
        span = Span(
            span_id=str(uuid.uuid4()),
            trace_id=trace_id,
            step_name=step_name,
            start_time=time.time(),
        )

        # Add context metadata
        if hasattr(ctx, "request"):
            span.metadata["session_id"] = ctx.request.session_id
            if ctx.request.user_id:
                span.metadata["user_id"] = ctx.request.user_id

        return span

    async def pre_step(self, step_name: str, ctx: Any) -> Any:
        """Create and start a span for the step.

        Args:
            step_name: Name of the step
            ctx: Pipeline context

        Returns:
            The context unchanged
        """
        trace_id = self._get_trace_id(ctx)
        span = self._create_span(trace_id, step_name, ctx)

        # Store active span
        span_key = f"{trace_id}:{step_name}"
        self._active_spans[span_key] = span

        # Initialize trace list if needed
        if trace_id not in self._traces:
            self._traces[trace_id] = []
        self._traces[trace_id].append(span)

        return ctx

    async def post_step(self, step_name: str, ctx: Any, result: Any) -> Any:
        """Complete the span with success status.

        Args:
            step_name: Name of the step
            ctx: Pipeline context
            result: Step result

        Returns:
            The result unchanged
        """
        trace_id = self._get_trace_id(ctx)
        span_key = f"{trace_id}:{step_name}"

        if span_key in self._active_spans:
            span = self._active_spans.pop(span_key)
            span.end_time = time.time()
            span.status = "success"

            # Add step-specific metadata
            if step_name == "route" and hasattr(ctx, "route_decision") and ctx.route_decision:
                span.metadata["intent"] = ctx.route_decision.intent.value
                span.metadata["complexity"] = ctx.route_decision.complexity
            elif step_name == "plan" and hasattr(ctx, "plan") and ctx.plan:
                span.metadata["plan_id"] = ctx.plan.plan_id
                span.metadata["step_count"] = len(ctx.plan.steps)

            # Export if callback provided
            if self.export_callback:
                self.export_callback(span)

        return result

    async def on_error(
        self, step_name: str, ctx: Any, error: Exception
    ) -> Optional[Any]:
        """Complete the span with error status.

        Args:
            step_name: Name of the step
            ctx: Pipeline context
            error: The exception

        Raises:
            Exception: Always re-raises the error
        """
        trace_id = self._get_trace_id(ctx)
        span_key = f"{trace_id}:{step_name}"

        if span_key in self._active_spans:
            span = self._active_spans.pop(span_key)
            span.end_time = time.time()
            span.status = "error"
            span.error = f"{type(error).__name__}: {error}"

            # Export if callback provided
            if self.export_callback:
                self.export_callback(span)

        raise error

    def get_trace(self, trace_id: str) -> list[Span]:
        """Get all spans for a trace ID.

        Args:
            trace_id: The trace ID

        Returns:
            List of spans for this trace
        """
        return self._traces.get(trace_id, [])

    def get_all_traces(self) -> dict[str, list[Span]]:
        """Get all traces.

        Returns:
            Dictionary mapping trace IDs to span lists
        """
        return dict(self._traces)

    def clear_trace(self, trace_id: str) -> None:
        """Clear a specific trace.

        Args:
            trace_id: The trace ID to clear
        """
        self._traces.pop(trace_id, None)

    def clear_all(self) -> None:
        """Clear all traces and active spans."""
        self._traces.clear()
        self._active_spans.clear()
