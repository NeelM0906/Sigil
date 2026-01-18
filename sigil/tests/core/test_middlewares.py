"""Tests for built-in middleware implementations.

This module tests:
- LoggingMiddleware
- MetricsMiddleware
- TracingMiddleware
- RetryMiddleware
- CircuitBreakerMiddleware
"""

import pytest
import asyncio
import logging
import time
from unittest.mock import MagicMock, patch

from sigil.core.middlewares import (
    LoggingMiddleware,
    MetricsMiddleware,
    TracingMiddleware,
    RetryMiddleware,
    CircuitBreakerMiddleware,
)
from sigil.core.middlewares.circuit_breaker import CircuitState, CircuitOpenError


# =============================================================================
# LoggingMiddleware Tests
# =============================================================================


class TestLoggingMiddleware:
    """Tests for LoggingMiddleware."""

    @pytest.mark.asyncio
    async def test_logs_step_start(self, caplog):
        """Test that pre_step logs step start."""
        middleware = LoggingMiddleware(log_level=logging.INFO)

        with caplog.at_level(logging.INFO):
            await middleware.pre_step("test_step", {})

        assert "Starting step: test_step" in caplog.text

    @pytest.mark.asyncio
    async def test_logs_step_completion(self, caplog):
        """Test that post_step logs step completion with timing."""
        middleware = LoggingMiddleware(log_level=logging.INFO)

        # Start timer
        await middleware.pre_step("test_step", {})
        # Allow some time to pass
        await asyncio.sleep(0.01)
        # End timer
        with caplog.at_level(logging.INFO):
            await middleware.post_step("test_step", {}, None)

        assert "Completed step: test_step" in caplog.text
        assert "ms" in caplog.text

    @pytest.mark.asyncio
    async def test_logs_step_error(self, caplog):
        """Test that on_error logs the error."""
        middleware = LoggingMiddleware(log_level=logging.INFO)

        await middleware.pre_step("test_step", {})

        with caplog.at_level(logging.ERROR):
            with pytest.raises(ValueError):
                await middleware.on_error("test_step", {}, ValueError("test error"))

        assert "Error in step test_step" in caplog.text
        assert "test error" in caplog.text

    @pytest.mark.asyncio
    async def test_context_logging(self, caplog):
        """Test that context info is logged when enabled."""
        middleware = LoggingMiddleware(
            log_level=logging.INFO, include_context=True
        )

        ctx = MagicMock()
        ctx.request.session_id = "test-session"

        with caplog.at_level(logging.INFO):
            await middleware.pre_step("test_step", ctx)

        assert "session=test-session" in caplog.text


# =============================================================================
# MetricsMiddleware Tests
# =============================================================================


class TestMetricsMiddleware:
    """Tests for MetricsMiddleware."""

    @pytest.mark.asyncio
    async def test_tracks_call_count(self):
        """Test that calls are counted."""
        middleware = MetricsMiddleware()

        await middleware.pre_step("test_step", {})
        await middleware.post_step("test_step", {}, None)

        metrics = middleware.get_step_metrics("test_step")
        assert metrics.calls == 1
        assert metrics.successes == 1

    @pytest.mark.asyncio
    async def test_tracks_errors(self):
        """Test that errors are counted."""
        middleware = MetricsMiddleware()

        await middleware.pre_step("test_step", {})

        with pytest.raises(ValueError):
            await middleware.on_error("test_step", {}, ValueError("test"))

        metrics = middleware.get_step_metrics("test_step")
        assert metrics.calls == 1
        assert metrics.errors == 1

    @pytest.mark.asyncio
    async def test_tracks_timing(self):
        """Test that timing is tracked."""
        middleware = MetricsMiddleware()

        await middleware.pre_step("test_step", {})
        await asyncio.sleep(0.01)  # 10ms
        await middleware.post_step("test_step", {}, None)

        metrics = middleware.get_step_metrics("test_step")
        assert metrics.total_time_ms >= 10
        assert metrics.min_time_ms >= 10
        assert metrics.max_time_ms >= 10
        assert metrics.avg_time_ms >= 10

    @pytest.mark.asyncio
    async def test_summary(self):
        """Test getting summary of all metrics."""
        middleware = MetricsMiddleware()

        await middleware.pre_step("step1", {})
        await middleware.post_step("step1", {}, None)
        await middleware.pre_step("step2", {})
        await middleware.post_step("step2", {}, None)

        summary = middleware.get_summary()

        assert "step1" in summary
        assert "step2" in summary
        assert summary["step1"]["calls"] == 1
        assert summary["step2"]["calls"] == 1

    @pytest.mark.asyncio
    async def test_reset_metrics(self):
        """Test resetting all metrics."""
        middleware = MetricsMiddleware()

        await middleware.pre_step("test_step", {})
        await middleware.post_step("test_step", {}, None)

        middleware.reset()

        assert middleware.get_step_metrics("test_step") is None
        assert middleware.get_total_calls() == 0


# =============================================================================
# TracingMiddleware Tests
# =============================================================================


class TestTracingMiddleware:
    """Tests for TracingMiddleware."""

    @pytest.mark.asyncio
    async def test_creates_span(self):
        """Test that spans are created for steps."""
        middleware = TracingMiddleware()

        ctx = MagicMock()
        ctx.request.correlation_id = "test-trace-id"
        ctx.request.session_id = "test-session"
        ctx.request.user_id = None

        await middleware.pre_step("test_step", ctx)
        await middleware.post_step("test_step", ctx, None)

        trace = middleware.get_trace("test-trace-id")
        assert len(trace) == 1

        span = trace[0]
        assert span.step_name == "test_step"
        assert span.status == "success"
        assert span.duration_ms is not None

    @pytest.mark.asyncio
    async def test_span_error_status(self):
        """Test that error status is recorded."""
        middleware = TracingMiddleware()

        ctx = MagicMock()
        ctx.request.correlation_id = "test-trace-id"
        ctx.request.session_id = "test-session"
        ctx.request.user_id = None

        await middleware.pre_step("test_step", ctx)

        with pytest.raises(ValueError):
            await middleware.on_error("test_step", ctx, ValueError("test"))

        trace = middleware.get_trace("test-trace-id")
        span = trace[0]
        assert span.status == "error"
        assert "test" in span.error

    @pytest.mark.asyncio
    async def test_export_callback(self):
        """Test that export callback is called."""
        exported_spans = []

        def export_callback(span):
            exported_spans.append(span)

        middleware = TracingMiddleware(export_callback=export_callback)

        ctx = MagicMock()
        ctx.request.correlation_id = "test-trace-id"
        ctx.request.session_id = "test-session"
        ctx.request.user_id = None

        await middleware.pre_step("test_step", ctx)
        await middleware.post_step("test_step", ctx, None)

        assert len(exported_spans) == 1
        assert exported_spans[0].step_name == "test_step"

    @pytest.mark.asyncio
    async def test_multiple_spans(self):
        """Test multiple spans in a trace."""
        middleware = TracingMiddleware()

        ctx = MagicMock()
        ctx.request.correlation_id = "test-trace-id"
        ctx.request.session_id = "test-session"
        ctx.request.user_id = None

        await middleware.pre_step("step1", ctx)
        await middleware.post_step("step1", ctx, None)
        await middleware.pre_step("step2", ctx)
        await middleware.post_step("step2", ctx, None)

        trace = middleware.get_trace("test-trace-id")
        assert len(trace) == 2

    def test_clear_trace(self):
        """Test clearing a specific trace."""
        middleware = TracingMiddleware()
        middleware._traces["test-id"] = []

        middleware.clear_trace("test-id")

        assert "test-id" not in middleware._traces

    def test_clear_all(self):
        """Test clearing all traces."""
        middleware = TracingMiddleware()
        middleware._traces["id1"] = []
        middleware._traces["id2"] = []

        middleware.clear_all()

        assert len(middleware._traces) == 0


# =============================================================================
# RetryMiddleware Tests
# =============================================================================


class TestRetryMiddleware:
    """Tests for RetryMiddleware."""

    @pytest.mark.asyncio
    async def test_tracks_retry_count(self):
        """Test that retry count is tracked."""
        middleware = RetryMiddleware(
            max_retries=3, retryable_exceptions={ValueError}
        )

        await middleware.pre_step("test_step", {})

        # First error
        with pytest.raises(ValueError):
            await middleware.on_error("test_step", {}, ValueError("test"))

        assert middleware.get_retry_count("test_step") == 1

    @pytest.mark.asyncio
    async def test_resets_on_success(self):
        """Test that retry count resets on success."""
        middleware = RetryMiddleware()

        await middleware.pre_step("test_step", {})
        await middleware.post_step("test_step", {}, None)

        assert middleware.get_retry_count("test_step") == 0

    @pytest.mark.asyncio
    async def test_should_retry_logic(self):
        """Test should_retry determines if retry is needed."""
        middleware = RetryMiddleware(
            max_retries=2, retryable_exceptions={TimeoutError}
        )

        # Should retry TimeoutError
        assert middleware._should_retry("step", TimeoutError(), 0)
        assert middleware._should_retry("step", TimeoutError(), 1)
        assert not middleware._should_retry("step", TimeoutError(), 2)  # Max reached

        # Should not retry ValueError (not in retryable list)
        assert not middleware._should_retry("step", ValueError(), 0)

    def test_delay_calculation(self):
        """Test exponential backoff delay calculation."""
        middleware = RetryMiddleware(
            base_delay=0.1, max_delay=10.0, exponential_base=2.0, jitter=False
        )

        config = middleware._get_step_config("test")

        # Without jitter, delays should be exact
        assert middleware._calculate_delay(0, config) == 0.1  # 0.1 * 2^0
        assert middleware._calculate_delay(1, config) == 0.2  # 0.1 * 2^1
        assert middleware._calculate_delay(2, config) == 0.4  # 0.1 * 2^2

    def test_max_delay_cap(self):
        """Test that delay is capped at max_delay."""
        middleware = RetryMiddleware(
            base_delay=1.0, max_delay=5.0, exponential_base=2.0, jitter=False
        )

        config = middleware._get_step_config("test")

        # 1.0 * 2^10 = 1024, should be capped at 5.0
        delay = middleware._calculate_delay(10, config)
        assert delay == 5.0

    def test_step_overrides(self):
        """Test per-step configuration overrides."""
        middleware = RetryMiddleware(
            max_retries=3,
            step_overrides={"critical_step": {"max_retries": 5}},
        )

        default_config = middleware._get_step_config("normal_step")
        assert default_config["max_retries"] == 3

        override_config = middleware._get_step_config("critical_step")
        assert override_config["max_retries"] == 5


# =============================================================================
# CircuitBreakerMiddleware Tests
# =============================================================================


class TestCircuitBreakerMiddleware:
    """Tests for CircuitBreakerMiddleware."""

    @pytest.mark.asyncio
    async def test_starts_closed(self):
        """Test that circuit starts in closed state."""
        middleware = CircuitBreakerMiddleware()

        ctx = {}
        await middleware.pre_step("test_step", ctx)

        assert middleware.get_circuit_state("test_step") == CircuitState.CLOSED

    @pytest.mark.asyncio
    async def test_opens_after_failures(self):
        """Test that circuit opens after failure threshold."""
        middleware = CircuitBreakerMiddleware(failure_threshold=3)

        ctx = {}

        for _ in range(3):
            await middleware.pre_step("test_step", ctx)
            with pytest.raises(ValueError):
                await middleware.on_error("test_step", ctx, ValueError("fail"))

        assert middleware.get_circuit_state("test_step") == CircuitState.OPEN

    @pytest.mark.asyncio
    async def test_blocks_when_open(self):
        """Test that requests are blocked when circuit is open."""
        middleware = CircuitBreakerMiddleware(failure_threshold=1)

        ctx = {}

        # Trigger circuit open
        await middleware.pre_step("test_step", ctx)
        with pytest.raises(ValueError):
            await middleware.on_error("test_step", ctx, ValueError("fail"))

        # Next request should be blocked
        with pytest.raises(CircuitOpenError):
            await middleware.pre_step("test_step", ctx)

    @pytest.mark.asyncio
    async def test_transitions_to_half_open(self):
        """Test that circuit transitions to half-open after timeout."""
        middleware = CircuitBreakerMiddleware(
            failure_threshold=1, recovery_timeout=0.01  # 10ms
        )

        ctx = {}

        # Trigger circuit open
        await middleware.pre_step("test_step", ctx)
        with pytest.raises(ValueError):
            await middleware.on_error("test_step", ctx, ValueError("fail"))

        assert middleware.get_circuit_state("test_step") == CircuitState.OPEN

        # Wait for recovery timeout
        await asyncio.sleep(0.02)

        # Should transition to half-open
        await middleware.pre_step("test_step", ctx)
        assert middleware.get_circuit_state("test_step") == CircuitState.HALF_OPEN

    @pytest.mark.asyncio
    async def test_closes_after_successful_recovery(self):
        """Test that circuit closes after successful calls in half-open."""
        middleware = CircuitBreakerMiddleware(
            failure_threshold=1, recovery_timeout=0.01, half_open_max_calls=2
        )

        ctx = {}

        # Trigger circuit open
        await middleware.pre_step("test_step", ctx)
        with pytest.raises(ValueError):
            await middleware.on_error("test_step", ctx, ValueError("fail"))

        # Wait for recovery timeout
        await asyncio.sleep(0.02)

        # Successful calls in half-open
        for _ in range(2):
            await middleware.pre_step("test_step", ctx)
            await middleware.post_step("test_step", ctx, None)

        assert middleware.get_circuit_state("test_step") == CircuitState.CLOSED

    @pytest.mark.asyncio
    async def test_reopens_on_half_open_failure(self):
        """Test that circuit reopens on failure in half-open state."""
        middleware = CircuitBreakerMiddleware(
            failure_threshold=1, recovery_timeout=0.01
        )

        ctx = {}

        # Trigger circuit open
        await middleware.pre_step("test_step", ctx)
        with pytest.raises(ValueError):
            await middleware.on_error("test_step", ctx, ValueError("fail"))

        # Wait for recovery timeout
        await asyncio.sleep(0.02)

        # Fail in half-open
        await middleware.pre_step("test_step", ctx)
        with pytest.raises(ValueError):
            await middleware.on_error("test_step", ctx, ValueError("fail again"))

        assert middleware.get_circuit_state("test_step") == CircuitState.OPEN

    @pytest.mark.asyncio
    async def test_excluded_steps(self):
        """Test that excluded steps bypass circuit breaker."""
        middleware = CircuitBreakerMiddleware(
            failure_threshold=1, excluded_steps={"excluded_step"}
        )

        ctx = {}

        # Fail the excluded step multiple times
        for _ in range(5):
            await middleware.pre_step("excluded_step", ctx)
            with pytest.raises(ValueError):
                await middleware.on_error("excluded_step", ctx, ValueError("fail"))

        # Should not have a circuit for excluded step
        # (or it should remain closed)
        # Actually it records but doesn't block
        await middleware.pre_step("excluded_step", ctx)  # Should not raise

    def test_reset_circuit(self):
        """Test manually resetting a circuit."""
        middleware = CircuitBreakerMiddleware()
        middleware._circuits["test_step"] = middleware._get_circuit("test_step")
        middleware._circuits["test_step"].state = CircuitState.OPEN

        middleware.reset_circuit("test_step")

        assert middleware.get_circuit_state("test_step") == CircuitState.CLOSED

    def test_reset_all_circuits(self):
        """Test resetting all circuits."""
        middleware = CircuitBreakerMiddleware()
        middleware._circuits["step1"] = middleware._get_circuit("step1")
        middleware._circuits["step1"].state = CircuitState.OPEN
        middleware._circuits["step2"] = middleware._get_circuit("step2")
        middleware._circuits["step2"].state = CircuitState.HALF_OPEN

        middleware.reset_all()

        assert middleware.get_circuit_state("step1") == CircuitState.CLOSED
        assert middleware.get_circuit_state("step2") == CircuitState.CLOSED

    def test_get_all_circuits(self):
        """Test getting state of all circuits."""
        middleware = CircuitBreakerMiddleware()
        middleware._get_circuit("step1")
        middleware._get_circuit("step2")

        all_circuits = middleware.get_all_circuits()

        assert "step1" in all_circuits
        assert "step2" in all_circuits
        assert all_circuits["step1"]["state"] == "closed"
