"""Tests for Sigil v2 Token Budget System.

Task 3.3: Token Budget System Implementation

Test Coverage:
- TokenBudget creation and validation
- TokenBudget immutability (frozen dataclass)
- TokenBudget serialization/deserialization
- TokenTracker usage recording
- TokenTracker budget checking
- TokenTracker utilization calculations
- TokenMetrics per-call recording
- TokenMetrics aggregation by model
- TokenMetrics summary statistics
- TokenBudgetExceeded exception
- TokenBudgetWarning warning class

Design Philosophy: Tokens not USD - all tests verify provider-agnostic behavior.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import FrozenInstanceError
from datetime import datetime, timezone
from unittest.mock import patch

import pytest

from sigil.telemetry import TokenBudget, TokenTracker, TokenMetrics, TokenCallRecord
from sigil.core.exceptions import TokenBudgetExceeded, TokenBudgetWarning


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def default_budget() -> TokenBudget:
    """Provide a standard token budget for testing."""
    return TokenBudget(
        max_input_tokens=4000,
        max_output_tokens=2000,
        max_total_tokens=10000,
        warn_threshold=0.8,
    )


@pytest.fixture
def small_budget() -> TokenBudget:
    """Provide a small token budget for edge case testing."""
    return TokenBudget(
        max_input_tokens=100,
        max_output_tokens=50,
        max_total_tokens=200,
        warn_threshold=0.9,
    )


@pytest.fixture
def fresh_tracker() -> TokenTracker:
    """Provide a fresh TokenTracker instance."""
    return TokenTracker()


@pytest.fixture
def used_tracker() -> TokenTracker:
    """Provide a TokenTracker with some recorded usage."""
    tracker = TokenTracker()
    tracker.record_usage(1000, 500)
    tracker.record_usage(2000, 1000)
    return tracker


@pytest.fixture
def fresh_metrics() -> TokenMetrics:
    """Provide a fresh TokenMetrics instance."""
    return TokenMetrics()


@pytest.fixture
def populated_metrics() -> TokenMetrics:
    """Provide TokenMetrics with multiple recorded calls."""
    metrics = TokenMetrics()
    metrics.add_call("anthropic:claude-opus-4-5-20251101", 1000, 500)
    metrics.add_call("openai:gpt-4", 800, 400)
    metrics.add_call("anthropic:claude-opus-4-5-20251101", 1200, 600)
    metrics.add_call("openai:gpt-4-turbo", 500, 250)
    return metrics


# =============================================================================
# Test: TokenBudget Creation and Validation
# =============================================================================


class TestTokenBudgetCreation:
    """Test TokenBudget creation and parameter validation."""

    def test_create_valid_budget(self):
        """Test creating a valid TokenBudget with all parameters."""
        budget = TokenBudget(
            max_input_tokens=4000,
            max_output_tokens=2000,
            max_total_tokens=10000,
            warn_threshold=0.8,
        )
        assert budget.max_input_tokens == 4000
        assert budget.max_output_tokens == 2000
        assert budget.max_total_tokens == 10000
        assert budget.warn_threshold == 0.8

    def test_create_budget_with_default_threshold(self):
        """Test creating TokenBudget with default warn_threshold."""
        budget = TokenBudget(
            max_input_tokens=4000,
            max_output_tokens=2000,
            max_total_tokens=10000,
        )
        assert budget.warn_threshold == 0.8

    def test_create_budget_minimum_valid_values(self):
        """Test creating TokenBudget with minimum valid values."""
        budget = TokenBudget(
            max_input_tokens=1,
            max_output_tokens=1,
            max_total_tokens=2,
            warn_threshold=0.01,
        )
        assert budget.max_input_tokens == 1
        assert budget.max_output_tokens == 1
        assert budget.max_total_tokens == 2

    def test_reject_negative_input_tokens(self):
        """Test that negative max_input_tokens raises ValueError."""
        with pytest.raises(ValueError, match="max_input_tokens must be positive"):
            TokenBudget(
                max_input_tokens=-1,
                max_output_tokens=100,
                max_total_tokens=1000,
            )

    def test_reject_zero_input_tokens(self):
        """Test that zero max_input_tokens raises ValueError."""
        with pytest.raises(ValueError, match="max_input_tokens must be positive"):
            TokenBudget(
                max_input_tokens=0,
                max_output_tokens=100,
                max_total_tokens=1000,
            )

    def test_reject_negative_output_tokens(self):
        """Test that negative max_output_tokens raises ValueError."""
        with pytest.raises(ValueError, match="max_output_tokens must be positive"):
            TokenBudget(
                max_input_tokens=100,
                max_output_tokens=-1,
                max_total_tokens=1000,
            )

    def test_reject_zero_output_tokens(self):
        """Test that zero max_output_tokens raises ValueError."""
        with pytest.raises(ValueError, match="max_output_tokens must be positive"):
            TokenBudget(
                max_input_tokens=100,
                max_output_tokens=0,
                max_total_tokens=1000,
            )

    def test_reject_negative_total_tokens(self):
        """Test that negative max_total_tokens raises ValueError."""
        with pytest.raises(ValueError, match="max_total_tokens must be positive"):
            TokenBudget(
                max_input_tokens=100,
                max_output_tokens=100,
                max_total_tokens=-1,
            )

    def test_reject_zero_total_tokens(self):
        """Test that zero max_total_tokens raises ValueError."""
        with pytest.raises(ValueError, match="max_total_tokens must be positive"):
            TokenBudget(
                max_input_tokens=100,
                max_output_tokens=100,
                max_total_tokens=0,
            )

    def test_reject_invalid_warn_threshold_zero(self):
        """Test that warn_threshold of 0 raises ValueError."""
        with pytest.raises(ValueError, match="warn_threshold must be between"):
            TokenBudget(
                max_input_tokens=100,
                max_output_tokens=100,
                max_total_tokens=1000,
                warn_threshold=0.0,
            )

    def test_reject_invalid_warn_threshold_negative(self):
        """Test that negative warn_threshold raises ValueError."""
        with pytest.raises(ValueError, match="warn_threshold must be between"):
            TokenBudget(
                max_input_tokens=100,
                max_output_tokens=100,
                max_total_tokens=1000,
                warn_threshold=-0.5,
            )

    def test_reject_invalid_warn_threshold_over_one(self):
        """Test that warn_threshold > 1.0 raises ValueError."""
        with pytest.raises(ValueError, match="warn_threshold must be between"):
            TokenBudget(
                max_input_tokens=100,
                max_output_tokens=100,
                max_total_tokens=1000,
                warn_threshold=1.5,
            )

    def test_allow_warn_threshold_exactly_one(self):
        """Test that warn_threshold of exactly 1.0 is allowed."""
        budget = TokenBudget(
            max_input_tokens=100,
            max_output_tokens=100,
            max_total_tokens=1000,
            warn_threshold=1.0,
        )
        assert budget.warn_threshold == 1.0

    def test_reject_sum_exceeds_total(self):
        """Test that input + output > total raises ValueError."""
        with pytest.raises(
            ValueError, match="max_input_tokens \\+ max_output_tokens cannot exceed"
        ):
            TokenBudget(
                max_input_tokens=5000,
                max_output_tokens=6000,
                max_total_tokens=10000,
            )

    def test_allow_sum_equals_total(self):
        """Test that input + output == total is allowed."""
        budget = TokenBudget(
            max_input_tokens=5000,
            max_output_tokens=5000,
            max_total_tokens=10000,
        )
        assert budget.max_input_tokens + budget.max_output_tokens == budget.max_total_tokens


# =============================================================================
# Test: TokenBudget Immutability
# =============================================================================


class TestTokenBudgetImmutability:
    """Test that TokenBudget is immutable (frozen dataclass)."""

    def test_cannot_modify_max_input_tokens(self, default_budget):
        """Test that max_input_tokens cannot be modified."""
        with pytest.raises(FrozenInstanceError):
            default_budget.max_input_tokens = 5000

    def test_cannot_modify_max_output_tokens(self, default_budget):
        """Test that max_output_tokens cannot be modified."""
        with pytest.raises(FrozenInstanceError):
            default_budget.max_output_tokens = 3000

    def test_cannot_modify_max_total_tokens(self, default_budget):
        """Test that max_total_tokens cannot be modified."""
        with pytest.raises(FrozenInstanceError):
            default_budget.max_total_tokens = 15000

    def test_cannot_modify_warn_threshold(self, default_budget):
        """Test that warn_threshold cannot be modified."""
        with pytest.raises(FrozenInstanceError):
            default_budget.warn_threshold = 0.9


# =============================================================================
# Test: TokenBudget Methods
# =============================================================================


class TestTokenBudgetMethods:
    """Test TokenBudget utility methods."""

    def test_validate_request_within_limits(self, default_budget):
        """Test validate_request returns True for valid request."""
        assert default_budget.validate_request(3000, 1500) is True

    def test_validate_request_at_limits(self, default_budget):
        """Test validate_request returns True at exact limits."""
        assert default_budget.validate_request(4000, 2000) is True

    def test_validate_request_exceeds_input(self, default_budget):
        """Test validate_request returns False when input exceeds limit."""
        assert default_budget.validate_request(5000, 1000) is False

    def test_validate_request_exceeds_output(self, default_budget):
        """Test validate_request returns False when output exceeds limit."""
        assert default_budget.validate_request(3000, 3000) is False

    def test_get_warning_thresholds(self, default_budget):
        """Test get_warning_thresholds returns correct values."""
        thresholds = default_budget.get_warning_thresholds()
        assert thresholds["input_tokens"] == 3200  # 4000 * 0.8
        assert thresholds["output_tokens"] == 1600  # 2000 * 0.8
        assert thresholds["total_tokens"] == 8000  # 10000 * 0.8


# =============================================================================
# Test: TokenBudget Serialization
# =============================================================================


class TestTokenBudgetSerialization:
    """Test TokenBudget serialization and deserialization."""

    def test_to_dict(self, default_budget):
        """Test serialization to dictionary."""
        data = default_budget.to_dict()
        assert data == {
            "max_input_tokens": 4000,
            "max_output_tokens": 2000,
            "max_total_tokens": 10000,
            "warn_threshold": 0.8,
        }

    def test_from_dict(self):
        """Test deserialization from dictionary."""
        data = {
            "max_input_tokens": 4000,
            "max_output_tokens": 2000,
            "max_total_tokens": 10000,
            "warn_threshold": 0.8,
        }
        budget = TokenBudget.from_dict(data)
        assert budget.max_input_tokens == 4000
        assert budget.max_output_tokens == 2000
        assert budget.max_total_tokens == 10000
        assert budget.warn_threshold == 0.8

    def test_from_dict_with_default_threshold(self):
        """Test from_dict uses default warn_threshold if missing."""
        data = {
            "max_input_tokens": 4000,
            "max_output_tokens": 2000,
            "max_total_tokens": 10000,
        }
        budget = TokenBudget.from_dict(data)
        assert budget.warn_threshold == 0.8

    def test_to_json(self, default_budget):
        """Test serialization to JSON string."""
        json_str = default_budget.to_json()
        data = json.loads(json_str)
        assert data["max_input_tokens"] == 4000
        assert data["max_output_tokens"] == 2000
        assert data["max_total_tokens"] == 10000
        assert data["warn_threshold"] == 0.8

    def test_from_json(self):
        """Test deserialization from JSON string."""
        json_str = '{"max_input_tokens": 4000, "max_output_tokens": 2000, "max_total_tokens": 10000, "warn_threshold": 0.8}'
        budget = TokenBudget.from_json(json_str)
        assert budget.max_input_tokens == 4000
        assert budget.max_output_tokens == 2000
        assert budget.max_total_tokens == 10000
        assert budget.warn_threshold == 0.8

    def test_roundtrip_serialization(self, default_budget):
        """Test that serialization roundtrip preserves data."""
        json_str = default_budget.to_json()
        restored = TokenBudget.from_json(json_str)
        assert restored == default_budget


# =============================================================================
# Test: TokenTracker Basic Operations
# =============================================================================


class TestTokenTrackerBasics:
    """Test TokenTracker basic operations."""

    def test_initial_state(self, fresh_tracker):
        """Test that a fresh tracker has zero usage."""
        assert fresh_tracker.input_tokens == 0
        assert fresh_tracker.output_tokens == 0
        assert fresh_tracker.total_tokens == 0
        assert fresh_tracker.call_count == 0

    def test_record_single_usage(self, fresh_tracker):
        """Test recording a single usage."""
        fresh_tracker.record_usage(1000, 500)
        assert fresh_tracker.input_tokens == 1000
        assert fresh_tracker.output_tokens == 500
        assert fresh_tracker.total_tokens == 1500
        assert fresh_tracker.call_count == 1

    def test_record_multiple_usage(self, fresh_tracker):
        """Test recording multiple usages accumulates correctly."""
        fresh_tracker.record_usage(1000, 500)
        fresh_tracker.record_usage(2000, 1000)
        fresh_tracker.record_usage(500, 250)
        assert fresh_tracker.input_tokens == 3500
        assert fresh_tracker.output_tokens == 1750
        assert fresh_tracker.total_tokens == 5250
        assert fresh_tracker.call_count == 3

    def test_record_zero_usage(self, fresh_tracker):
        """Test recording zero tokens is allowed."""
        fresh_tracker.record_usage(0, 0)
        assert fresh_tracker.input_tokens == 0
        assert fresh_tracker.output_tokens == 0
        assert fresh_tracker.call_count == 1

    def test_reject_negative_input_tokens(self, fresh_tracker):
        """Test that negative input tokens raises ValueError."""
        with pytest.raises(ValueError, match="Token counts cannot be negative"):
            fresh_tracker.record_usage(-100, 50)

    def test_reject_negative_output_tokens(self, fresh_tracker):
        """Test that negative output tokens raises ValueError."""
        with pytest.raises(ValueError, match="Token counts cannot be negative"):
            fresh_tracker.record_usage(100, -50)

    def test_reset(self, used_tracker):
        """Test that reset clears all counters."""
        used_tracker.reset()
        assert used_tracker.input_tokens == 0
        assert used_tracker.output_tokens == 0
        assert used_tracker.total_tokens == 0
        assert used_tracker.call_count == 0


# =============================================================================
# Test: TokenTracker Budget Checking
# =============================================================================


class TestTokenTrackerBudgetChecking:
    """Test TokenTracker budget checking methods."""

    def test_is_within_budget_fresh_tracker(self, fresh_tracker, default_budget):
        """Test that fresh tracker is within budget."""
        assert fresh_tracker.is_within_budget(default_budget) is True

    def test_is_within_budget_partial_usage(self, default_budget):
        """Test tracker is within budget with partial usage."""
        tracker = TokenTracker()
        tracker.record_usage(2000, 1000)
        assert tracker.is_within_budget(default_budget) is True

    def test_is_within_budget_at_limit(self, default_budget):
        """Test tracker is within budget at exact limits."""
        tracker = TokenTracker()
        tracker.record_usage(4000, 2000)
        # Total is 6000, which is under 10000
        assert tracker.is_within_budget(default_budget) is True

    def test_is_within_budget_exceeds_input(self, default_budget):
        """Test tracker exceeds budget when input limit crossed."""
        tracker = TokenTracker()
        tracker.record_usage(4500, 1000)
        assert tracker.is_within_budget(default_budget) is False

    def test_is_within_budget_exceeds_output(self, default_budget):
        """Test tracker exceeds budget when output limit crossed."""
        tracker = TokenTracker()
        tracker.record_usage(1000, 2500)
        assert tracker.is_within_budget(default_budget) is False

    def test_is_within_budget_exceeds_total(self, default_budget):
        """Test tracker exceeds budget when total limit crossed."""
        tracker = TokenTracker()
        # Make multiple calls that exceed total while staying under individual limits
        tracker.record_usage(3000, 1500)  # 4500 total
        tracker.record_usage(3000, 1500)  # 9000 total
        tracker.record_usage(1000, 500)   # 10500 total - exceeds
        assert tracker.is_within_budget(default_budget) is False


class TestTokenTrackerUtilization:
    """Test TokenTracker utilization calculation methods."""

    def test_get_utilization_zero_usage(self, fresh_tracker, default_budget):
        """Test utilization is 0% for fresh tracker."""
        assert fresh_tracker.get_utilization(default_budget) == 0.0

    def test_get_utilization_partial_usage(self, default_budget):
        """Test utilization calculation with partial usage."""
        tracker = TokenTracker()
        tracker.record_usage(1000, 500)  # 1500 total
        # Input: 1000/4000 = 25%, Output: 500/2000 = 25%, Total: 1500/10000 = 15%
        # Max is 25%
        assert tracker.get_utilization(default_budget) == 0.25

    def test_get_utilization_at_threshold(self, default_budget):
        """Test utilization calculation at warning threshold."""
        tracker = TokenTracker()
        tracker.record_usage(3200, 1600)  # At 80% of individual limits
        util = tracker.get_utilization(default_budget)
        assert util == pytest.approx(0.8, rel=0.01)

    def test_get_utilization_over_budget(self, default_budget):
        """Test utilization can exceed 1.0 when over budget."""
        tracker = TokenTracker()
        tracker.record_usage(5000, 2500)  # Over limits
        util = tracker.get_utilization(default_budget)
        assert util > 1.0

    def test_should_warn_below_threshold(self, default_budget):
        """Test should_warn returns False below threshold."""
        tracker = TokenTracker()
        tracker.record_usage(1000, 500)  # 25% utilization
        assert tracker.should_warn(default_budget) is False

    def test_should_warn_at_threshold(self, default_budget):
        """Test should_warn returns True at threshold."""
        tracker = TokenTracker()
        tracker.record_usage(3200, 1600)  # 80% utilization
        assert tracker.should_warn(default_budget) is True

    def test_should_warn_above_threshold(self, default_budget):
        """Test should_warn returns True above threshold."""
        tracker = TokenTracker()
        tracker.record_usage(3600, 1800)  # 90% utilization
        assert tracker.should_warn(default_budget) is True


class TestTokenTrackerRemainingBudget:
    """Test TokenTracker remaining budget calculation."""

    def test_get_remaining_fresh_tracker(self, fresh_tracker, default_budget):
        """Test remaining budget for fresh tracker equals original."""
        remaining = fresh_tracker.get_remaining(default_budget)
        assert remaining.max_input_tokens == default_budget.max_input_tokens
        assert remaining.max_output_tokens == default_budget.max_output_tokens
        assert remaining.max_total_tokens == default_budget.max_total_tokens

    def test_get_remaining_partial_usage(self, default_budget):
        """Test remaining budget after partial usage."""
        tracker = TokenTracker()
        tracker.record_usage(1000, 500)
        remaining = tracker.get_remaining(default_budget)
        assert remaining.max_input_tokens == 3000
        assert remaining.max_output_tokens == 1500
        assert remaining.max_total_tokens == 8500

    def test_get_remaining_preserves_threshold(self, default_budget):
        """Test that remaining budget preserves warn_threshold."""
        tracker = TokenTracker()
        tracker.record_usage(1000, 500)
        remaining = tracker.get_remaining(default_budget)
        assert remaining.warn_threshold == default_budget.warn_threshold

    def test_to_dict_includes_all_fields(self, used_tracker):
        """Test to_dict includes all tracking fields."""
        data = used_tracker.to_dict()
        assert "input_tokens" in data
        assert "output_tokens" in data
        assert "total_tokens" in data
        assert "call_count" in data
        assert data["input_tokens"] == 3000
        assert data["output_tokens"] == 1500
        assert data["total_tokens"] == 4500
        assert data["call_count"] == 2


# =============================================================================
# Test: TokenMetrics Basic Operations
# =============================================================================


class TestTokenMetricsBasics:
    """Test TokenMetrics basic operations."""

    def test_initial_state(self, fresh_metrics):
        """Test that fresh metrics has no records."""
        assert len(fresh_metrics.records) == 0
        assert fresh_metrics.total_input_tokens == 0
        assert fresh_metrics.total_output_tokens == 0
        assert fresh_metrics.total_tokens == 0

    def test_add_single_call(self, fresh_metrics):
        """Test adding a single call record."""
        fresh_metrics.add_call("anthropic:claude-opus-4-5-20251101", 1000, 500)
        assert len(fresh_metrics.records) == 1
        assert fresh_metrics.total_input_tokens == 1000
        assert fresh_metrics.total_output_tokens == 500
        assert fresh_metrics.total_tokens == 1500

    def test_add_multiple_calls(self, fresh_metrics):
        """Test adding multiple call records."""
        fresh_metrics.add_call("anthropic:claude-opus-4-5-20251101", 1000, 500)
        fresh_metrics.add_call("openai:gpt-4", 800, 400)
        assert len(fresh_metrics.records) == 2
        assert fresh_metrics.total_input_tokens == 1800
        assert fresh_metrics.total_output_tokens == 900
        assert fresh_metrics.total_tokens == 2700

    def test_add_call_with_custom_timestamp(self, fresh_metrics):
        """Test adding call with custom timestamp."""
        custom_time = datetime(2024, 1, 15, 10, 30, 0, tzinfo=timezone.utc)
        fresh_metrics.add_call("anthropic:claude-opus-4-5-20251101", 1000, 500, timestamp=custom_time)
        assert fresh_metrics.records[0].timestamp == custom_time

    def test_reject_negative_input_tokens(self, fresh_metrics):
        """Test that negative input tokens raises ValueError."""
        with pytest.raises(ValueError, match="Token counts cannot be negative"):
            fresh_metrics.add_call("anthropic:claude-opus-4-5-20251101", -100, 50)

    def test_reject_negative_output_tokens(self, fresh_metrics):
        """Test that negative output tokens raises ValueError."""
        with pytest.raises(ValueError, match="Token counts cannot be negative"):
            fresh_metrics.add_call("anthropic:claude-opus-4-5-20251101", 100, -50)

    def test_clear(self, populated_metrics):
        """Test that clear removes all records."""
        populated_metrics.clear()
        assert len(populated_metrics.records) == 0
        assert populated_metrics.total_tokens == 0


# =============================================================================
# Test: TokenMetrics Aggregation
# =============================================================================


class TestTokenMetricsAggregation:
    """Test TokenMetrics aggregation methods."""

    def test_get_average_tokens_empty(self, fresh_metrics):
        """Test average tokens for empty metrics."""
        averages = fresh_metrics.get_average_tokens_per_call()
        assert averages["avg_input_tokens"] == 0.0
        assert averages["avg_output_tokens"] == 0.0
        assert averages["avg_total_tokens"] == 0.0

    def test_get_average_tokens(self, populated_metrics):
        """Test average tokens calculation."""
        averages = populated_metrics.get_average_tokens_per_call()
        # 4 calls: (1000+800+1200+500)/4 = 875 avg input
        assert averages["avg_input_tokens"] == 875.0
        # (500+400+600+250)/4 = 437.5 avg output
        assert averages["avg_output_tokens"] == 437.5
        # Total: 875 + 437.5 = 1312.5
        assert averages["avg_total_tokens"] == 1312.5

    def test_get_usage_by_model(self, populated_metrics):
        """Test usage breakdown by model."""
        by_model = populated_metrics.get_usage_by_model()

        # Claude calls: 1000+1200 input, 500+600 output
        assert "anthropic:claude-opus-4-5-20251101" in by_model
        assert by_model["anthropic:claude-opus-4-5-20251101"]["input_tokens"] == 2200
        assert by_model["anthropic:claude-opus-4-5-20251101"]["output_tokens"] == 1100
        assert by_model["anthropic:claude-opus-4-5-20251101"]["call_count"] == 2

        # GPT-4 calls: 800 input, 400 output
        assert "openai:gpt-4" in by_model
        assert by_model["openai:gpt-4"]["input_tokens"] == 800
        assert by_model["openai:gpt-4"]["output_tokens"] == 400
        assert by_model["openai:gpt-4"]["call_count"] == 1

        # GPT-4-turbo: 500 input, 250 output
        assert "openai:gpt-4-turbo" in by_model
        assert by_model["openai:gpt-4-turbo"]["input_tokens"] == 500
        assert by_model["openai:gpt-4-turbo"]["output_tokens"] == 250
        assert by_model["openai:gpt-4-turbo"]["call_count"] == 1


class TestTokenMetricsSummary:
    """Test TokenMetrics summary generation."""

    def test_get_summary_empty(self, fresh_metrics):
        """Test summary for empty metrics."""
        summary = fresh_metrics.get_summary()
        assert summary["total_calls"] == 0
        assert summary["total_input_tokens"] == 0
        assert summary["total_output_tokens"] == 0
        assert summary["total_tokens"] == 0
        assert summary["time_range"] is None

    def test_get_summary_populated(self, populated_metrics):
        """Test summary for populated metrics."""
        summary = populated_metrics.get_summary()
        assert summary["total_calls"] == 4
        assert summary["total_input_tokens"] == 3500
        assert summary["total_output_tokens"] == 1750
        assert summary["total_tokens"] == 5250
        assert "averages" in summary
        assert "by_model" in summary
        assert "time_range" in summary
        assert summary["time_range"] is not None

    def test_summary_time_range(self, fresh_metrics):
        """Test that time_range reflects actual call times."""
        time1 = datetime(2024, 1, 1, 10, 0, 0, tzinfo=timezone.utc)
        time2 = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        time3 = datetime(2024, 1, 1, 11, 0, 0, tzinfo=timezone.utc)

        fresh_metrics.add_call("model-a", 100, 50, timestamp=time1)
        fresh_metrics.add_call("model-b", 100, 50, timestamp=time2)
        fresh_metrics.add_call("model-c", 100, 50, timestamp=time3)

        summary = fresh_metrics.get_summary()
        assert summary["time_range"]["first_call"] == time1.isoformat()
        assert summary["time_range"]["last_call"] == time2.isoformat()


# =============================================================================
# Test: TokenMetrics Export
# =============================================================================


class TestTokenMetricsExport:
    """Test TokenMetrics export functionality."""

    def test_export_metrics_empty(self, fresh_metrics):
        """Test exporting empty metrics."""
        exported = fresh_metrics.export_metrics()
        assert exported == []

    def test_export_metrics_populated(self, populated_metrics):
        """Test exporting populated metrics."""
        exported = populated_metrics.export_metrics()
        assert len(exported) == 4

        # Check structure of first record
        first = exported[0]
        assert "model" in first
        assert "input_tokens" in first
        assert "output_tokens" in first
        assert "total_tokens" in first
        assert "timestamp" in first

    def test_export_metrics_json_serializable(self, populated_metrics):
        """Test that exported metrics are JSON serializable."""
        exported = populated_metrics.export_metrics()
        # Should not raise
        json_str = json.dumps(exported)
        assert json_str is not None

    def test_to_json(self, populated_metrics):
        """Test to_json produces valid JSON."""
        json_str = populated_metrics.to_json()
        data = json.loads(json_str)
        assert len(data) == 4

    def test_from_records(self):
        """Test creating TokenMetrics from exported records."""
        records = [
            {
                "model": "anthropic:claude-opus-4-5-20251101",
                "input_tokens": 1000,
                "output_tokens": 500,
                "timestamp": "2024-01-15T10:30:00+00:00",
            },
            {
                "model": "openai:gpt-4",
                "input_tokens": 800,
                "output_tokens": 400,
                "timestamp": "2024-01-15T11:00:00+00:00",
            },
        ]
        metrics = TokenMetrics.from_records(records)
        assert len(metrics.records) == 2
        assert metrics.total_input_tokens == 1800
        assert metrics.total_output_tokens == 900


# =============================================================================
# Test: TokenCallRecord
# =============================================================================


class TestTokenCallRecord:
    """Test TokenCallRecord dataclass."""

    def test_create_record(self):
        """Test creating a call record."""
        record = TokenCallRecord(
            model="anthropic:claude-opus-4-5-20251101",
            input_tokens=1000,
            output_tokens=500,
        )
        assert record.model == "anthropic:claude-opus-4-5-20251101"
        assert record.input_tokens == 1000
        assert record.output_tokens == 500
        assert record.total_tokens == 1500
        assert record.timestamp is not None

    def test_record_with_custom_timestamp(self):
        """Test creating record with custom timestamp."""
        custom_time = datetime(2024, 1, 15, 10, 30, 0, tzinfo=timezone.utc)
        record = TokenCallRecord(
            model="openai:gpt-4",
            input_tokens=800,
            output_tokens=400,
            timestamp=custom_time,
        )
        assert record.timestamp == custom_time

    def test_record_to_dict(self):
        """Test serializing record to dictionary."""
        record = TokenCallRecord(
            model="anthropic:claude-opus-4-5-20251101",
            input_tokens=1000,
            output_tokens=500,
        )
        data = record.to_dict()
        assert data["model"] == "anthropic:claude-opus-4-5-20251101"
        assert data["input_tokens"] == 1000
        assert data["output_tokens"] == 500
        assert data["total_tokens"] == 1500
        assert "timestamp" in data


# =============================================================================
# Test: TokenBudgetExceeded Exception
# =============================================================================


class TestTokenBudgetExceededException:
    """Test TokenBudgetExceeded exception class."""

    def test_create_exception(self):
        """Test creating TokenBudgetExceeded exception."""
        exc = TokenBudgetExceeded(
            "Token budget exceeded",
            current_input_tokens=5000,
            current_output_tokens=2500,
            max_input_tokens=4000,
            max_output_tokens=2000,
            max_total_tokens=10000,
            exceeded_limit="input",
        )
        assert exc.current_input_tokens == 5000
        assert exc.current_output_tokens == 2500
        assert exc.current_total_tokens == 7500
        assert exc.max_input_tokens == 4000
        assert exc.max_output_tokens == 2000
        assert exc.max_total_tokens == 10000
        assert exc.exceeded_limit == "input"

    def test_exception_code(self):
        """Test that exception has correct code."""
        exc = TokenBudgetExceeded("Test", exceeded_limit="total")
        assert exc.code == "TOKEN_BUDGET_EXCEEDED"

    def test_exception_utilization_input(self):
        """Test utilization calculation for input exceeded."""
        exc = TokenBudgetExceeded(
            "Test",
            current_input_tokens=5000,
            current_output_tokens=1000,
            max_input_tokens=4000,
            max_output_tokens=2000,
            max_total_tokens=10000,
            exceeded_limit="input",
        )
        assert exc.utilization == pytest.approx(1.25, rel=0.01)

    def test_exception_utilization_output(self):
        """Test utilization calculation for output exceeded."""
        exc = TokenBudgetExceeded(
            "Test",
            current_input_tokens=3000,
            current_output_tokens=3000,
            max_input_tokens=4000,
            max_output_tokens=2000,
            max_total_tokens=10000,
            exceeded_limit="output",
        )
        assert exc.utilization == pytest.approx(1.5, rel=0.01)

    def test_exception_utilization_total(self):
        """Test utilization calculation for total exceeded."""
        exc = TokenBudgetExceeded(
            "Test",
            current_input_tokens=6000,
            current_output_tokens=6000,
            max_input_tokens=8000,
            max_output_tokens=8000,
            max_total_tokens=10000,
            exceeded_limit="total",
        )
        assert exc.utilization == pytest.approx(1.2, rel=0.01)

    def test_exception_str(self):
        """Test exception string representation."""
        exc = TokenBudgetExceeded(
            "Token budget exceeded",
            current_input_tokens=5000,
            current_output_tokens=2500,
            max_input_tokens=4000,
            max_output_tokens=2000,
            max_total_tokens=10000,
            exceeded_limit="input",
        )
        exc_str = str(exc)
        assert "TOKEN_BUDGET_EXCEEDED" in exc_str
        assert "7500/10000" in exc_str
        assert "exceeded: input" in exc_str

    def test_exception_context(self):
        """Test that exception has context dictionary."""
        exc = TokenBudgetExceeded(
            "Test",
            current_input_tokens=5000,
            current_output_tokens=2500,
            max_input_tokens=4000,
            max_output_tokens=2000,
            max_total_tokens=10000,
            exceeded_limit="input",
        )
        assert "current_input_tokens" in exc.context
        assert "utilization" in exc.context

    def test_exception_is_not_recoverable(self):
        """Test that exception is marked as not recoverable."""
        exc = TokenBudgetExceeded("Test", exceeded_limit="total")
        assert exc.recoverable is False


# =============================================================================
# Test: TokenBudgetWarning
# =============================================================================


class TestTokenBudgetWarning:
    """Test TokenBudgetWarning warning class."""

    def test_create_warning(self):
        """Test creating TokenBudgetWarning."""
        warning = TokenBudgetWarning(
            "Approaching token budget limit",
            current_input_tokens=3500,
            current_output_tokens=1500,
            max_total_tokens=6000,
            utilization=0.83,
            threshold=0.8,
        )
        assert warning.current_input_tokens == 3500
        assert warning.current_output_tokens == 1500
        assert warning.current_total_tokens == 5000
        assert warning.max_total_tokens == 6000
        assert warning.utilization == 0.83
        assert warning.threshold == 0.8

    def test_warning_message_format(self):
        """Test warning message includes key information."""
        warning = TokenBudgetWarning(
            "Approaching limit",
            current_input_tokens=3500,
            current_output_tokens=1500,
            max_total_tokens=6000,
            utilization=0.83,
            threshold=0.8,
        )
        msg = str(warning)
        assert "5000/6000" in msg
        assert "83" in msg  # utilization percentage
        assert "80%" in msg  # threshold

    def test_warning_can_be_issued(self):
        """Test that warning can be issued via warnings module."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            warning = TokenBudgetWarning(
                "Test warning",
                current_input_tokens=100,
                current_output_tokens=50,
                max_total_tokens=200,
                utilization=0.75,
                threshold=0.7,
            )
            warnings.warn(warning)
            assert len(w) == 1
            assert issubclass(w[0].category, UserWarning)


# =============================================================================
# Test: Integration Scenarios
# =============================================================================


class TestIntegrationScenarios:
    """Test realistic integration scenarios."""

    def test_full_session_tracking(self, default_budget):
        """Test tracking a complete session with multiple calls."""
        tracker = TokenTracker()
        metrics = TokenMetrics()

        # Simulate multiple API calls
        calls = [
            ("anthropic:claude-opus-4-5-20251101", 1000, 500),
            ("anthropic:claude-opus-4-5-20251101", 1500, 800),
            ("openai:gpt-4", 800, 400),
            ("anthropic:claude-opus-4-5-20251101", 500, 200),
        ]

        for model, input_tok, output_tok in calls:
            tracker.record_usage(input_tok, output_tok)
            metrics.add_call(model, input_tok, output_tok)

        # Verify tracker
        assert tracker.total_tokens == 5700
        assert tracker.call_count == 4
        assert tracker.is_within_budget(default_budget)
        assert tracker.get_utilization(default_budget) < 1.0

        # Verify metrics
        summary = metrics.get_summary()
        assert summary["total_calls"] == 4
        assert summary["total_tokens"] == 5700
        assert len(summary["by_model"]) == 2

    def test_budget_exceeded_workflow(self, small_budget):
        """Test workflow when budget is exceeded."""
        tracker = TokenTracker()

        # First call - within budget
        tracker.record_usage(50, 25)
        assert tracker.is_within_budget(small_budget)

        # Second call - still within budget
        tracker.record_usage(40, 20)
        assert tracker.is_within_budget(small_budget)

        # Third call - exceeds budget
        tracker.record_usage(30, 15)
        assert not tracker.is_within_budget(small_budget)

        # Can detect which limit was exceeded
        util = tracker.get_utilization(small_budget)
        assert util > 1.0

    def test_warning_threshold_workflow(self, default_budget):
        """Test warning threshold detection workflow."""
        tracker = TokenTracker()

        # Build up usage gradually
        tracker.record_usage(2000, 1000)
        assert not tracker.should_warn(default_budget)

        tracker.record_usage(1000, 500)
        # Now at 75% - still below 80% threshold
        assert not tracker.should_warn(default_budget)

        tracker.record_usage(300, 150)
        # Now above 80% threshold
        assert tracker.should_warn(default_budget)

    def test_remaining_budget_for_subtasks(self, default_budget):
        """Test calculating remaining budget for subtasks."""
        tracker = TokenTracker()

        # Main task uses some budget
        tracker.record_usage(2000, 1000)

        # Calculate remaining for subtask
        remaining = tracker.get_remaining(default_budget)

        # Verify remaining allows subtask execution
        assert remaining.max_input_tokens > 0
        assert remaining.max_output_tokens > 0
        assert remaining.max_total_tokens > 0

        # Subtask tracker can use remaining budget
        subtask_tracker = TokenTracker()
        subtask_tracker.record_usage(1000, 500)
        assert subtask_tracker.is_within_budget(remaining)
