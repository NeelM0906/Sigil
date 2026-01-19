"""Token budget system for Sigil v2 framework.

This module implements provider-agnostic token tracking and budgeting.
All costs are tracked in tokens (not USD) to remain provider-agnostic.

Key Components:
    - TokenBudget: Immutable configuration defining token limits
    - TokenTracker: Tracks cumulative token usage across calls
    - TokenMetrics: Records per-call token usage with timestamps

Design Philosophy:
    - Tokens, not USD: Provider-agnostic budgeting
    - Immutable budgets: Configuration is frozen after creation
    - Detailed metrics: Per-call tracking with model attribution
    - Warning thresholds: Alert before exceeding limits

Example:
    >>> budget = TokenBudget(
    ...     max_input_tokens=4000,
    ...     max_output_tokens=2000,
    ...     max_total_tokens=10000
    ... )
    >>> tracker = TokenTracker()
    >>> tracker.record_usage(1000, 500)
    >>> tracker.is_within_budget(budget)
    True
    >>> tracker.get_utilization(budget)
    0.15
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Optional
import json


@dataclass(frozen=True)
class TokenBudget:
    """Immutable token budget configuration.

    Defines the maximum token limits for a session or request.
    The budget is frozen (immutable) after creation to ensure
    consistent enforcement throughout execution.

    Attributes:
        max_input_tokens: Maximum input tokens per request
        max_output_tokens: Maximum output tokens per request
        max_total_tokens: Maximum total tokens per session
        warn_threshold: Percentage (0.0-1.0) to trigger warning

    Example:
        >>> budget = TokenBudget(
        ...     max_input_tokens=4000,
        ...     max_output_tokens=2000,
        ...     max_total_tokens=10000,
        ...     warn_threshold=0.8
        ... )
        >>> budget.max_total_tokens
        10000
    """

    max_input_tokens: int
    max_output_tokens: int
    max_total_tokens: int
    warn_threshold: float = 0.8

    def __post_init__(self) -> None:
        """Validate budget parameters after initialization."""
        if self.max_input_tokens <= 0:
            raise ValueError("max_input_tokens must be positive")
        if self.max_output_tokens <= 0:
            raise ValueError("max_output_tokens must be positive")
        if self.max_total_tokens <= 0:
            raise ValueError("max_total_tokens must be positive")
        if not 0.0 < self.warn_threshold <= 1.0:
            raise ValueError("warn_threshold must be between 0.0 and 1.0")
        if self.max_input_tokens + self.max_output_tokens > self.max_total_tokens:
            raise ValueError(
                "max_input_tokens + max_output_tokens cannot exceed max_total_tokens"
            )

    def validate_request(self, input_tokens: int, output_tokens: int) -> bool:
        """Check if a single request is within per-request limits.

        Args:
            input_tokens: Number of input tokens for the request
            output_tokens: Number of output tokens for the request

        Returns:
            True if the request is within limits, False otherwise
        """
        return (
            input_tokens <= self.max_input_tokens
            and output_tokens <= self.max_output_tokens
        )

    def get_warning_thresholds(self) -> dict[str, int]:
        """Get the token counts at which warnings should trigger.

        Returns:
            Dictionary with warning thresholds for each limit type
        """
        return {
            "input_tokens": int(self.max_input_tokens * self.warn_threshold),
            "output_tokens": int(self.max_output_tokens * self.warn_threshold),
            "total_tokens": int(self.max_total_tokens * self.warn_threshold),
        }

    def to_dict(self) -> dict:
        """Serialize budget to dictionary.

        Returns:
            Dictionary representation of the budget
        """
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "TokenBudget":
        """Create TokenBudget from dictionary.

        Args:
            data: Dictionary with budget parameters

        Returns:
            New TokenBudget instance
        """
        return cls(
            max_input_tokens=data["max_input_tokens"],
            max_output_tokens=data["max_output_tokens"],
            max_total_tokens=data["max_total_tokens"],
            warn_threshold=data.get("warn_threshold", 0.8),
        )

    def to_json(self) -> str:
        """Serialize budget to JSON string.

        Returns:
            JSON string representation of the budget
        """
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, json_str: str) -> "TokenBudget":
        """Create TokenBudget from JSON string.

        Args:
            json_str: JSON string with budget parameters

        Returns:
            New TokenBudget instance
        """
        return cls.from_dict(json.loads(json_str))


@dataclass
class TokenTracker:
    """Tracks cumulative token usage across multiple calls.

    Maintains running totals of input and output tokens consumed
    during a session. Provides methods to check budget compliance
    and calculate utilization metrics.

    Attributes:
        input_tokens: Cumulative input tokens consumed
        output_tokens: Cumulative output tokens consumed
        call_count: Number of API calls recorded

    Example:
        >>> tracker = TokenTracker()
        >>> tracker.record_usage(1000, 500)
        >>> tracker.record_usage(800, 400)
        >>> tracker.total_tokens
        2700
        >>> tracker.call_count
        2
    """

    input_tokens: int = 0
    output_tokens: int = 0
    call_count: int = 0

    @property
    def total_tokens(self) -> int:
        """Get total tokens consumed (input + output).

        Returns:
            Sum of input and output tokens
        """
        return self.input_tokens + self.output_tokens

    def record_usage(self, input_tokens: int, output_tokens: int) -> None:
        """Record token usage from a single API call.

        Args:
            input_tokens: Number of input tokens used
            output_tokens: Number of output tokens used

        Raises:
            ValueError: If token counts are negative
        """
        if input_tokens < 0 or output_tokens < 0:
            raise ValueError("Token counts cannot be negative")

        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        self.call_count += 1

    def get_remaining(self, budget: TokenBudget) -> TokenBudget:
        """Calculate remaining budget based on current usage.

        Creates a new TokenBudget with limits reduced by current usage.
        Useful for passing remaining budget to sub-tasks.

        Args:
            budget: The original budget to compare against

        Returns:
            New TokenBudget with remaining limits

        Note:
            If usage exceeds budget, remaining values will be 0.
        """
        remaining_input = max(0, budget.max_input_tokens - self.input_tokens)
        remaining_output = max(0, budget.max_output_tokens - self.output_tokens)
        remaining_total = max(0, budget.max_total_tokens - self.total_tokens)

        # Ensure remaining input + output does not exceed remaining total
        if remaining_input + remaining_output > remaining_total:
            # Scale down proportionally
            scale = remaining_total / (remaining_input + remaining_output) if (remaining_input + remaining_output) > 0 else 0
            remaining_input = int(remaining_input * scale)
            remaining_output = int(remaining_output * scale)

        # Handle edge case where values become 0 (avoid validation error)
        remaining_input = max(1, remaining_input) if remaining_total > 0 else 1
        remaining_output = max(1, remaining_output) if remaining_total > 0 else 1
        remaining_total = max(remaining_input + remaining_output, remaining_total) if remaining_total > 0 else remaining_input + remaining_output

        return TokenBudget(
            max_input_tokens=remaining_input,
            max_output_tokens=remaining_output,
            max_total_tokens=remaining_total,
            warn_threshold=budget.warn_threshold,
        )

    def is_within_budget(self, budget: TokenBudget) -> bool:
        """Check if current usage is within budget limits.

        Args:
            budget: The budget to check against

        Returns:
            True if all usage is within limits, False otherwise
        """
        return (
            self.input_tokens <= budget.max_input_tokens
            and self.output_tokens <= budget.max_output_tokens
            and self.total_tokens <= budget.max_total_tokens
        )

    def get_utilization(self, budget: TokenBudget) -> float:
        """Calculate overall budget utilization as a percentage.

        Uses the highest utilization among input, output, and total
        to give a conservative estimate.

        Args:
            budget: The budget to calculate utilization against

        Returns:
            Utilization as a float between 0.0 and 1.0+
            (can exceed 1.0 if over budget)
        """
        input_util = self.input_tokens / budget.max_input_tokens
        output_util = self.output_tokens / budget.max_output_tokens
        total_util = self.total_tokens / budget.max_total_tokens

        return max(input_util, output_util, total_util)

    def should_warn(self, budget: TokenBudget) -> bool:
        """Check if usage has reached the warning threshold.

        Args:
            budget: The budget to check against

        Returns:
            True if utilization exceeds warn_threshold
        """
        return self.get_utilization(budget) >= budget.warn_threshold

    def reset(self) -> None:
        """Reset all tracking counters to zero."""
        self.input_tokens = 0
        self.output_tokens = 0
        self.call_count = 0

    def to_dict(self) -> dict:
        """Serialize tracker state to dictionary.

        Returns:
            Dictionary with current usage stats
        """
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "call_count": self.call_count,
        }


@dataclass
class TokenCallRecord:
    """Record of a single API call's token usage.

    Attributes:
        model: Model identifier (e.g., 'anthropic:claude-opus-4-5-20251101')
        input_tokens: Input tokens for this call
        output_tokens: Output tokens for this call
        timestamp: When the call was made (UTC)
    """

    model: str
    input_tokens: int
    output_tokens: int
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def total_tokens(self) -> int:
        """Get total tokens for this call."""
        return self.input_tokens + self.output_tokens

    def to_dict(self) -> dict:
        """Serialize record to dictionary.

        Returns:
            Dictionary representation of the record
        """
        return {
            "model": self.model,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class TokenMetrics:
    """Detailed token usage metrics with per-call tracking.

    Stores individual call records for detailed analysis, including
    timestamps and model attribution. Supports aggregation by model
    and summary statistics.

    Attributes:
        records: List of individual call records

    Example:
        >>> metrics = TokenMetrics()
        >>> metrics.add_call('anthropic:claude-opus-4-5-20251101', 1000, 500)
        >>> metrics.add_call('openai:gpt-4', 800, 400)
        >>> summary = metrics.get_summary()
        >>> summary['total_calls']
        2
    """

    records: list[TokenCallRecord] = field(default_factory=list)

    def add_call(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        timestamp: Optional[datetime] = None,
    ) -> None:
        """Record a single API call.

        Args:
            model: Model identifier (e.g., 'anthropic:claude-opus-4-5-20251101')
            input_tokens: Number of input tokens used
            output_tokens: Number of output tokens used
            timestamp: Optional timestamp (defaults to now UTC)

        Raises:
            ValueError: If token counts are negative
        """
        if input_tokens < 0 or output_tokens < 0:
            raise ValueError("Token counts cannot be negative")

        record = TokenCallRecord(
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            timestamp=timestamp or datetime.now(timezone.utc),
        )
        self.records.append(record)

    @property
    def total_input_tokens(self) -> int:
        """Get total input tokens across all calls."""
        return sum(r.input_tokens for r in self.records)

    @property
    def total_output_tokens(self) -> int:
        """Get total output tokens across all calls."""
        return sum(r.output_tokens for r in self.records)

    @property
    def total_tokens(self) -> int:
        """Get total tokens across all calls."""
        return self.total_input_tokens + self.total_output_tokens

    def get_average_tokens_per_call(self) -> dict[str, float]:
        """Calculate average tokens per call.

        Returns:
            Dictionary with average input, output, and total tokens
        """
        if not self.records:
            return {
                "avg_input_tokens": 0.0,
                "avg_output_tokens": 0.0,
                "avg_total_tokens": 0.0,
            }

        count = len(self.records)
        return {
            "avg_input_tokens": self.total_input_tokens / count,
            "avg_output_tokens": self.total_output_tokens / count,
            "avg_total_tokens": self.total_tokens / count,
        }

    def get_usage_by_model(self) -> dict[str, dict[str, int]]:
        """Calculate token usage grouped by model.

        Returns:
            Dictionary mapping model names to usage stats
        """
        usage: dict[str, dict[str, int]] = {}

        for record in self.records:
            if record.model not in usage:
                usage[record.model] = {
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0,
                    "call_count": 0,
                }

            usage[record.model]["input_tokens"] += record.input_tokens
            usage[record.model]["output_tokens"] += record.output_tokens
            usage[record.model]["total_tokens"] += record.total_tokens
            usage[record.model]["call_count"] += 1

        return usage

    def get_summary(self) -> dict:
        """Get comprehensive summary statistics.

        Returns:
            Dictionary with all summary metrics including:
            - total_calls: Number of API calls
            - total_input_tokens: Sum of all input tokens
            - total_output_tokens: Sum of all output tokens
            - total_tokens: Grand total of all tokens
            - averages: Average tokens per call
            - by_model: Usage breakdown by model
            - time_range: First and last call timestamps
        """
        summary = {
            "total_calls": len(self.records),
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_tokens,
            "averages": self.get_average_tokens_per_call(),
            "by_model": self.get_usage_by_model(),
        }

        if self.records:
            sorted_records = sorted(self.records, key=lambda r: r.timestamp)
            summary["time_range"] = {
                "first_call": sorted_records[0].timestamp.isoformat(),
                "last_call": sorted_records[-1].timestamp.isoformat(),
            }
        else:
            summary["time_range"] = None

        return summary

    def export_metrics(self) -> list[dict]:
        """Export all metrics records as a list of dictionaries.

        Returns:
            List of dictionaries, one per call record
        """
        return [record.to_dict() for record in self.records]

    def to_json(self) -> str:
        """Serialize all metrics to JSON string.

        Returns:
            JSON string with all records
        """
        return json.dumps(self.export_metrics(), indent=2)

    @classmethod
    def from_records(cls, records: list[dict]) -> "TokenMetrics":
        """Create TokenMetrics from list of record dictionaries.

        Args:
            records: List of dictionaries with record data

        Returns:
            New TokenMetrics instance with loaded records
        """
        metrics = cls()
        for record_data in records:
            timestamp = None
            if "timestamp" in record_data:
                timestamp = datetime.fromisoformat(record_data["timestamp"])

            metrics.add_call(
                model=record_data["model"],
                input_tokens=record_data["input_tokens"],
                output_tokens=record_data["output_tokens"],
                timestamp=timestamp,
            )
        return metrics

    def clear(self) -> None:
        """Clear all recorded metrics."""
        self.records.clear()
