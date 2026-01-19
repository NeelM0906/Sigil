# Logging Contracts for Sigil v2

## Overview

This document defines the comprehensive logging contract for the Sigil v2 CLI, providing structured logging schemas, token tracking specifications, and filtering capabilities. All logging in Sigil v2 follows these contracts to ensure consistent observability, debugging, and cost management.

**Version:** 1.0.0
**Last Updated:** 2026-01-11
**Status:** Authoritative

---

## Table of Contents

1. [Design Principles](#design-principles)
2. [Core Schemas](#core-schemas)
   - [ExecutionLogEntry](#executionlogentry)
   - [TokenUsage](#tokenusage)
   - [ComponentTokenUsage](#componenttokenusage)
   - [BudgetStatus](#budgetstatus)
   - [LogFilter](#logfilter)
3. [Log Levels](#log-levels)
4. [Component Identifiers](#component-identifiers)
5. [Structured Metadata Schemas](#structured-metadata-schemas)
6. [Error Logging Patterns](#error-logging-patterns)
7. [Token Tracking Contract](#token-tracking-contract)
8. [Streaming Protocol](#streaming-protocol)
9. [Query and Filter Specification](#query-and-filter-specification)
10. [Complete Examples](#complete-examples)
11. [Implementation Notes](#implementation-notes)

---

## Design Principles

### 1. Structured Over Unstructured

All log entries follow a strict schema. No free-form text logging is permitted in production components. This enables:
- Machine-parseable logs
- Consistent filtering and querying
- Reliable aggregation and analytics

### 2. Tokens as the Universal Currency

All operations track token consumption. The 256K context window budget is the primary constraint, and every component reports its token usage.

### 3. Traceable and Auditable

Every log entry links to:
- A session (via `session_id`)
- A correlation chain (via `correlation_id`)
- A specific operation (via `operation_id`)

### 4. Component-Aware

Logs are tagged by component, enabling:
- Component-level filtering
- Per-component token budgets
- Component health monitoring

### 5. Real-Time Streaming

Logs are designed for real-time consumption by monitoring scripts. The protocol supports:
- WebSocket streaming
- Buffered batch delivery
- Heartbeat mechanisms

---

## Core Schemas

### ExecutionLogEntry

The `ExecutionLogEntry` is the atomic unit of logging in Sigil v2. Every logged event produces exactly one entry conforming to this schema.

```python
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional
import uuid


class LogLevel(str, Enum):
    """Log severity levels following syslog conventions."""
    TRACE = "trace"      # Finest granularity, typically disabled
    DEBUG = "debug"      # Detailed debugging information
    INFO = "info"        # Normal operational messages
    WARNING = "warning"  # Potential issues, non-blocking
    ERROR = "error"      # Errors that may affect operation
    CRITICAL = "critical"  # Critical failures requiring attention


class ComponentId(str, Enum):
    """Component identifiers for log attribution."""
    ORCHESTRATOR = "orchestrator"
    ROUTER = "router"
    PLANNER = "planner"
    REASONING = "reasoning"
    MEMORY = "memory"
    CONTRACTS = "contracts"
    EVOLUTION = "evolution"
    CONTEXT = "context"
    TOOLS = "tools"
    CLI = "cli"
    API = "api"
    SYSTEM = "system"


@dataclass
class ExecutionLogEntry:
    """
    A single log entry in the Sigil v2 logging system.

    This is the canonical format for all logged events. Every component
    MUST produce logs conforming to this schema.

    Attributes:
        id: Unique identifier for this log entry (UUID v4).
        timestamp: ISO 8601 timestamp when the event occurred (UTC).
        level: Severity level of the log entry.
        component: Which component produced this log.
        operation: Name of the operation being performed.
        message: Human-readable description of the event.
        tokens_used: Token consumption for this specific operation.
        session_id: Session identifier for grouping related logs.
        correlation_id: Links related events across components.
        operation_id: Unique identifier for the current operation.
        parent_operation_id: Links to parent operation for nesting.
        duration_ms: Operation duration in milliseconds (if completed).
        metadata: Component-specific structured data.
        error_info: Error details if level is ERROR or CRITICAL.
        budget_snapshot: Token budget status at log time.

    Example:
        >>> entry = ExecutionLogEntry(
        ...     level=LogLevel.INFO,
        ...     component=ComponentId.REASONING,
        ...     operation="tree_of_thoughts",
        ...     message="Reasoning completed with 3 branches explored",
        ...     tokens_used=TokenUsage(input_tokens=1200, output_tokens=450),
        ...     session_id="sess_abc123",
        ...     metadata={
        ...         "strategy": "tree_of_thoughts",
        ...         "branches_explored": 3,
        ...         "confidence": 0.86,
        ...     }
        ... )
    """

    # Required fields
    level: LogLevel
    component: ComponentId
    operation: str
    message: str

    # Token tracking
    tokens_used: Optional["TokenUsage"] = None

    # Correlation and tracing
    session_id: str = ""
    correlation_id: str = ""
    operation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    parent_operation_id: Optional[str] = None

    # Timing
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    duration_ms: Optional[float] = None

    # Extensible metadata
    metadata: dict[str, Any] = field(default_factory=dict)

    # Error handling
    error_info: Optional["ErrorInfo"] = None

    # Budget tracking
    budget_snapshot: Optional["BudgetStatus"] = None

    # Auto-generated
    id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary for JSON encoding."""
        result = {
            "id": self.id,
            "timestamp": self.timestamp.isoformat(),
            "level": self.level.value,
            "component": self.component.value,
            "operation": self.operation,
            "message": self.message,
            "session_id": self.session_id,
            "correlation_id": self.correlation_id,
            "operation_id": self.operation_id,
        }

        if self.parent_operation_id:
            result["parent_operation_id"] = self.parent_operation_id

        if self.duration_ms is not None:
            result["duration_ms"] = self.duration_ms

        if self.tokens_used:
            result["tokens_used"] = self.tokens_used.to_dict()

        if self.metadata:
            result["metadata"] = self.metadata

        if self.error_info:
            result["error_info"] = self.error_info.to_dict()

        if self.budget_snapshot:
            result["budget_snapshot"] = self.budget_snapshot.to_dict()

        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExecutionLogEntry":
        """Deserialize from dictionary."""
        tokens_used = None
        if "tokens_used" in data:
            tokens_used = TokenUsage.from_dict(data["tokens_used"])

        error_info = None
        if "error_info" in data:
            error_info = ErrorInfo.from_dict(data["error_info"])

        budget_snapshot = None
        if "budget_snapshot" in data:
            budget_snapshot = BudgetStatus.from_dict(data["budget_snapshot"])

        return cls(
            id=data.get("id", str(uuid.uuid4())),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            level=LogLevel(data["level"]),
            component=ComponentId(data["component"]),
            operation=data["operation"],
            message=data["message"],
            session_id=data.get("session_id", ""),
            correlation_id=data.get("correlation_id", ""),
            operation_id=data.get("operation_id", ""),
            parent_operation_id=data.get("parent_operation_id"),
            duration_ms=data.get("duration_ms"),
            tokens_used=tokens_used,
            metadata=data.get("metadata", {}),
            error_info=error_info,
            budget_snapshot=budget_snapshot,
        )
```

#### JSON Schema

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://sigil.acti.ai/schemas/execution-log-entry.json",
  "title": "ExecutionLogEntry",
  "description": "A single log entry in the Sigil v2 logging system",
  "type": "object",
  "required": ["id", "timestamp", "level", "component", "operation", "message"],
  "properties": {
    "id": {
      "type": "string",
      "format": "uuid",
      "description": "Unique identifier for this log entry"
    },
    "timestamp": {
      "type": "string",
      "format": "date-time",
      "description": "ISO 8601 timestamp when the event occurred (UTC)"
    },
    "level": {
      "type": "string",
      "enum": ["trace", "debug", "info", "warning", "error", "critical"],
      "description": "Severity level of the log entry"
    },
    "component": {
      "type": "string",
      "enum": [
        "orchestrator", "router", "planner", "reasoning", "memory",
        "contracts", "evolution", "context", "tools", "cli", "api", "system"
      ],
      "description": "Which component produced this log"
    },
    "operation": {
      "type": "string",
      "minLength": 1,
      "maxLength": 128,
      "pattern": "^[a-z][a-z0-9_]*$",
      "description": "Name of the operation being performed (snake_case)"
    },
    "message": {
      "type": "string",
      "minLength": 1,
      "maxLength": 2048,
      "description": "Human-readable description of the event"
    },
    "tokens_used": {
      "$ref": "#/$defs/TokenUsage",
      "description": "Token consumption for this specific operation"
    },
    "session_id": {
      "type": "string",
      "description": "Session identifier for grouping related logs"
    },
    "correlation_id": {
      "type": "string",
      "description": "Links related events across components"
    },
    "operation_id": {
      "type": "string",
      "format": "uuid",
      "description": "Unique identifier for the current operation"
    },
    "parent_operation_id": {
      "type": "string",
      "format": "uuid",
      "description": "Links to parent operation for nesting"
    },
    "duration_ms": {
      "type": "number",
      "minimum": 0,
      "description": "Operation duration in milliseconds"
    },
    "metadata": {
      "type": "object",
      "additionalProperties": true,
      "description": "Component-specific structured data"
    },
    "error_info": {
      "$ref": "#/$defs/ErrorInfo",
      "description": "Error details if level is ERROR or CRITICAL"
    },
    "budget_snapshot": {
      "$ref": "#/$defs/BudgetStatus",
      "description": "Token budget status at log time"
    }
  },
  "$defs": {
    "TokenUsage": {
      "type": "object",
      "required": ["input_tokens", "output_tokens", "total_tokens"],
      "properties": {
        "input_tokens": {"type": "integer", "minimum": 0},
        "output_tokens": {"type": "integer", "minimum": 0},
        "total_tokens": {"type": "integer", "minimum": 0},
        "model": {"type": "string"},
        "cache_read_tokens": {"type": "integer", "minimum": 0},
        "cache_write_tokens": {"type": "integer", "minimum": 0}
      }
    },
    "ErrorInfo": {
      "type": "object",
      "required": ["error_type", "error_message"],
      "properties": {
        "error_type": {"type": "string"},
        "error_message": {"type": "string"},
        "error_code": {"type": "string"},
        "stack_trace": {"type": "string"},
        "recoverable": {"type": "boolean"},
        "retry_after_ms": {"type": "integer", "minimum": 0}
      }
    },
    "BudgetStatus": {
      "type": "object",
      "required": ["total_budget", "used", "remaining", "percentage_used"],
      "properties": {
        "total_budget": {"type": "integer", "minimum": 0},
        "used": {"type": "integer", "minimum": 0},
        "remaining": {"type": "integer", "minimum": 0},
        "percentage_used": {"type": "number", "minimum": 0, "maximum": 100}
      }
    }
  }
}
```

---

### TokenUsage

Tracks token consumption for a single operation.

```python
@dataclass
class TokenUsage:
    """
    Token usage for a single operation.

    All token tracking in Sigil uses this schema. Components MUST report
    token usage after every LLM call.

    Attributes:
        input_tokens: Tokens in the prompt/input.
        output_tokens: Tokens in the response/output.
        total_tokens: Sum of input and output tokens.
        model: Model identifier (e.g., 'anthropic:claude-opus-4-5-20251101').
        cache_read_tokens: Tokens read from prompt cache.
        cache_write_tokens: Tokens written to prompt cache.
        reasoning_tokens: Tokens used for chain-of-thought (if applicable).

    Example:
        >>> usage = TokenUsage(
        ...     input_tokens=1500,
        ...     output_tokens=800,
        ...     model="anthropic:claude-opus-4-5-20251101",
        ...     cache_read_tokens=500,
        ... )
        >>> usage.total_tokens
        2300
    """

    input_tokens: int
    output_tokens: int
    model: str = ""
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    reasoning_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        """Calculate total tokens consumed."""
        return self.input_tokens + self.output_tokens

    @property
    def effective_input_tokens(self) -> int:
        """Input tokens minus cache hits (for cost calculation)."""
        return max(0, self.input_tokens - self.cache_read_tokens)

    def __add__(self, other: "TokenUsage") -> "TokenUsage":
        """Combine two TokenUsage objects."""
        return TokenUsage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            model=self.model or other.model,
            cache_read_tokens=self.cache_read_tokens + other.cache_read_tokens,
            cache_write_tokens=self.cache_write_tokens + other.cache_write_tokens,
            reasoning_tokens=self.reasoning_tokens + other.reasoning_tokens,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "model": self.model,
            "cache_read_tokens": self.cache_read_tokens,
            "cache_write_tokens": self.cache_write_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "effective_input_tokens": self.effective_input_tokens,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TokenUsage":
        """Deserialize from dictionary."""
        return cls(
            input_tokens=data["input_tokens"],
            output_tokens=data["output_tokens"],
            model=data.get("model", ""),
            cache_read_tokens=data.get("cache_read_tokens", 0),
            cache_write_tokens=data.get("cache_write_tokens", 0),
            reasoning_tokens=data.get("reasoning_tokens", 0),
        )

    @classmethod
    def zero(cls) -> "TokenUsage":
        """Create a zero-usage instance."""
        return cls(input_tokens=0, output_tokens=0)

    @classmethod
    def from_anthropic_response(cls, response: dict, model: str) -> "TokenUsage":
        """Create from Anthropic API response usage field."""
        usage = response.get("usage", {})
        return cls(
            input_tokens=usage.get("input_tokens", 0),
            output_tokens=usage.get("output_tokens", 0),
            model=model,
            cache_read_tokens=usage.get("cache_read_input_tokens", 0),
            cache_write_tokens=usage.get("cache_creation_input_tokens", 0),
        )
```

#### Validation Rules

1. `input_tokens` MUST be non-negative
2. `output_tokens` MUST be non-negative
3. `total_tokens` MUST equal `input_tokens + output_tokens`
4. `cache_read_tokens` MUST NOT exceed `input_tokens`
5. `model` SHOULD follow pattern `provider:model-id`

---

### ComponentTokenUsage

Aggregates token usage by component across a session or execution.

```python
@dataclass
class ComponentTokenUsage:
    """
    Aggregated token usage for a component.

    Used for component-level budget tracking and reporting.

    Attributes:
        component: Component identifier.
        operation_count: Number of operations performed.
        usage: Cumulative token usage.
        first_operation: Timestamp of first operation.
        last_operation: Timestamp of last operation.
        peak_single_operation: Highest token count in a single operation.
    """

    component: ComponentId
    operation_count: int = 0
    usage: TokenUsage = field(default_factory=TokenUsage.zero)
    first_operation: Optional[datetime] = None
    last_operation: Optional[datetime] = None
    peak_single_operation: int = 0

    def record(self, usage: TokenUsage) -> None:
        """Record token usage from an operation."""
        self.operation_count += 1
        self.usage = self.usage + usage

        now = datetime.now(timezone.utc)
        if self.first_operation is None:
            self.first_operation = now
        self.last_operation = now

        if usage.total_tokens > self.peak_single_operation:
            self.peak_single_operation = usage.total_tokens

    @property
    def average_tokens_per_operation(self) -> float:
        """Calculate average tokens per operation."""
        if self.operation_count == 0:
            return 0.0
        return self.usage.total_tokens / self.operation_count

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "component": self.component.value,
            "operation_count": self.operation_count,
            "usage": self.usage.to_dict(),
            "first_operation": self.first_operation.isoformat() if self.first_operation else None,
            "last_operation": self.last_operation.isoformat() if self.last_operation else None,
            "peak_single_operation": self.peak_single_operation,
            "average_tokens_per_operation": self.average_tokens_per_operation,
        }
```

---

### BudgetStatus

Tracks budget consumption against the 256K limit.

```python
@dataclass
class BudgetStatus:
    """
    Token budget status snapshot.

    Provides real-time visibility into budget consumption relative
    to the 256K context window limit.

    Attributes:
        total_budget: Maximum tokens available (typically 256,000).
        used: Tokens consumed so far.
        remaining: Tokens still available.
        percentage_used: Consumption as a percentage (0-100).
        by_component: Breakdown by component.
        warning_threshold_reached: True if > 80% consumed.
        critical_threshold_reached: True if > 95% consumed.
    """

    total_budget: int
    used: int
    by_component: dict[str, int] = field(default_factory=dict)

    # Warning thresholds
    warning_threshold: float = 0.80
    critical_threshold: float = 0.95

    @property
    def remaining(self) -> int:
        """Calculate remaining tokens."""
        return max(0, self.total_budget - self.used)

    @property
    def percentage_used(self) -> float:
        """Calculate percentage of budget consumed."""
        if self.total_budget == 0:
            return 0.0
        return (self.used / self.total_budget) * 100

    @property
    def warning_threshold_reached(self) -> bool:
        """Check if warning threshold exceeded."""
        return (self.used / self.total_budget) >= self.warning_threshold

    @property
    def critical_threshold_reached(self) -> bool:
        """Check if critical threshold exceeded."""
        return (self.used / self.total_budget) >= self.critical_threshold

    def can_afford(self, tokens: int) -> bool:
        """Check if budget can afford the specified tokens."""
        return self.remaining >= tokens

    def consume(self, tokens: int, component: str) -> None:
        """Consume tokens from budget."""
        self.used += tokens
        if component not in self.by_component:
            self.by_component[component] = 0
        self.by_component[component] += tokens

    def get_component_percentage(self, component: str) -> float:
        """Get percentage of total budget used by component."""
        if self.total_budget == 0:
            return 0.0
        return (self.by_component.get(component, 0) / self.total_budget) * 100

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "total_budget": self.total_budget,
            "used": self.used,
            "remaining": self.remaining,
            "percentage_used": round(self.percentage_used, 2),
            "by_component": self.by_component,
            "warning_threshold_reached": self.warning_threshold_reached,
            "critical_threshold_reached": self.critical_threshold_reached,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BudgetStatus":
        """Deserialize from dictionary."""
        return cls(
            total_budget=data["total_budget"],
            used=data["used"],
            by_component=data.get("by_component", {}),
        )

    @classmethod
    def default(cls) -> "BudgetStatus":
        """Create default budget status with 256K limit."""
        return cls(total_budget=256_000, used=0)
```

---

### LogFilter

Specifies filtering criteria for log queries.

```python
@dataclass
class LogFilter:
    """
    Filter specification for querying logs.

    All filter fields are optional. Multiple filters are combined with AND logic.

    Attributes:
        session_id: Filter by specific session.
        correlation_id: Filter by correlation chain.
        components: Filter by component(s).
        levels: Filter by log level(s).
        operations: Filter by operation name(s).
        start_time: Filter logs after this time (inclusive).
        end_time: Filter logs before this time (inclusive).
        min_tokens: Filter logs with token usage >= this value.
        max_tokens: Filter logs with token usage <= this value.
        has_error: Filter logs with/without errors.
        metadata_query: Filter by metadata fields (key=value).
        search_text: Full-text search in message field.
        limit: Maximum number of results.
        offset: Number of results to skip (for pagination).
        sort_by: Field to sort by.
        sort_order: Sort direction (asc or desc).

    Example:
        >>> filter = LogFilter(
        ...     components=[ComponentId.REASONING, ComponentId.PLANNING],
        ...     levels=[LogLevel.WARNING, LogLevel.ERROR],
        ...     min_tokens=1000,
        ...     start_time=datetime(2026, 1, 11, 0, 0, 0),
        ...     sort_by="tokens_used",
        ...     sort_order="desc",
        ...     limit=100,
        ... )
    """

    # Identity filters
    session_id: Optional[str] = None
    correlation_id: Optional[str] = None
    operation_id: Optional[str] = None

    # Categorical filters
    components: Optional[list[ComponentId]] = None
    levels: Optional[list[LogLevel]] = None
    operations: Optional[list[str]] = None

    # Time filters
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None

    # Token filters
    min_tokens: Optional[int] = None
    max_tokens: Optional[int] = None

    # Error filters
    has_error: Optional[bool] = None
    error_types: Optional[list[str]] = None

    # Metadata filters
    metadata_query: Optional[dict[str, Any]] = None

    # Text search
    search_text: Optional[str] = None

    # Pagination
    limit: int = 100
    offset: int = 0

    # Sorting
    sort_by: str = "timestamp"
    sort_order: str = "desc"  # "asc" or "desc"

    def validate(self) -> list[str]:
        """Validate filter parameters, return list of errors."""
        errors = []

        if self.limit < 1 or self.limit > 10000:
            errors.append("limit must be between 1 and 10000")

        if self.offset < 0:
            errors.append("offset must be non-negative")

        if self.sort_order not in ("asc", "desc"):
            errors.append("sort_order must be 'asc' or 'desc'")

        valid_sort_fields = [
            "timestamp", "level", "component", "operation",
            "tokens_used", "duration_ms"
        ]
        if self.sort_by not in valid_sort_fields:
            errors.append(f"sort_by must be one of: {valid_sort_fields}")

        if self.min_tokens is not None and self.max_tokens is not None:
            if self.min_tokens > self.max_tokens:
                errors.append("min_tokens cannot exceed max_tokens")

        if self.start_time is not None and self.end_time is not None:
            if self.start_time > self.end_time:
                errors.append("start_time cannot be after end_time")

        return errors

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary for API requests."""
        result = {}

        if self.session_id:
            result["session_id"] = self.session_id
        if self.correlation_id:
            result["correlation_id"] = self.correlation_id
        if self.operation_id:
            result["operation_id"] = self.operation_id

        if self.components:
            result["components"] = [c.value for c in self.components]
        if self.levels:
            result["levels"] = [l.value for l in self.levels]
        if self.operations:
            result["operations"] = self.operations

        if self.start_time:
            result["start_time"] = self.start_time.isoformat()
        if self.end_time:
            result["end_time"] = self.end_time.isoformat()

        if self.min_tokens is not None:
            result["min_tokens"] = self.min_tokens
        if self.max_tokens is not None:
            result["max_tokens"] = self.max_tokens

        if self.has_error is not None:
            result["has_error"] = self.has_error
        if self.error_types:
            result["error_types"] = self.error_types

        if self.metadata_query:
            result["metadata_query"] = self.metadata_query
        if self.search_text:
            result["search_text"] = self.search_text

        result["limit"] = self.limit
        result["offset"] = self.offset
        result["sort_by"] = self.sort_by
        result["sort_order"] = self.sort_order

        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LogFilter":
        """Deserialize from dictionary."""
        components = None
        if "components" in data:
            components = [ComponentId(c) for c in data["components"]]

        levels = None
        if "levels" in data:
            levels = [LogLevel(l) for l in data["levels"]]

        start_time = None
        if "start_time" in data:
            start_time = datetime.fromisoformat(data["start_time"])

        end_time = None
        if "end_time" in data:
            end_time = datetime.fromisoformat(data["end_time"])

        return cls(
            session_id=data.get("session_id"),
            correlation_id=data.get("correlation_id"),
            operation_id=data.get("operation_id"),
            components=components,
            levels=levels,
            operations=data.get("operations"),
            start_time=start_time,
            end_time=end_time,
            min_tokens=data.get("min_tokens"),
            max_tokens=data.get("max_tokens"),
            has_error=data.get("has_error"),
            error_types=data.get("error_types"),
            metadata_query=data.get("metadata_query"),
            search_text=data.get("search_text"),
            limit=data.get("limit", 100),
            offset=data.get("offset", 0),
            sort_by=data.get("sort_by", "timestamp"),
            sort_order=data.get("sort_order", "desc"),
        )
```

#### JSON Schema for LogFilter

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://sigil.acti.ai/schemas/log-filter.json",
  "title": "LogFilter",
  "description": "Filter specification for querying logs",
  "type": "object",
  "properties": {
    "session_id": {
      "type": "string",
      "description": "Filter by specific session"
    },
    "correlation_id": {
      "type": "string",
      "description": "Filter by correlation chain"
    },
    "operation_id": {
      "type": "string",
      "format": "uuid",
      "description": "Filter by specific operation"
    },
    "components": {
      "type": "array",
      "items": {
        "type": "string",
        "enum": [
          "orchestrator", "router", "planner", "reasoning", "memory",
          "contracts", "evolution", "context", "tools", "cli", "api", "system"
        ]
      },
      "description": "Filter by component(s)"
    },
    "levels": {
      "type": "array",
      "items": {
        "type": "string",
        "enum": ["trace", "debug", "info", "warning", "error", "critical"]
      },
      "description": "Filter by log level(s)"
    },
    "operations": {
      "type": "array",
      "items": {"type": "string"},
      "description": "Filter by operation name(s)"
    },
    "start_time": {
      "type": "string",
      "format": "date-time",
      "description": "Filter logs after this time (inclusive)"
    },
    "end_time": {
      "type": "string",
      "format": "date-time",
      "description": "Filter logs before this time (inclusive)"
    },
    "min_tokens": {
      "type": "integer",
      "minimum": 0,
      "description": "Filter logs with token usage >= this value"
    },
    "max_tokens": {
      "type": "integer",
      "minimum": 0,
      "description": "Filter logs with token usage <= this value"
    },
    "has_error": {
      "type": "boolean",
      "description": "Filter logs with/without errors"
    },
    "error_types": {
      "type": "array",
      "items": {"type": "string"},
      "description": "Filter by specific error types"
    },
    "metadata_query": {
      "type": "object",
      "additionalProperties": true,
      "description": "Filter by metadata fields (key=value)"
    },
    "search_text": {
      "type": "string",
      "description": "Full-text search in message field"
    },
    "limit": {
      "type": "integer",
      "minimum": 1,
      "maximum": 10000,
      "default": 100,
      "description": "Maximum number of results"
    },
    "offset": {
      "type": "integer",
      "minimum": 0,
      "default": 0,
      "description": "Number of results to skip"
    },
    "sort_by": {
      "type": "string",
      "enum": ["timestamp", "level", "component", "operation", "tokens_used", "duration_ms"],
      "default": "timestamp",
      "description": "Field to sort by"
    },
    "sort_order": {
      "type": "string",
      "enum": ["asc", "desc"],
      "default": "desc",
      "description": "Sort direction"
    }
  }
}
```

---

### ErrorInfo

Structured error information for error logs.

```python
@dataclass
class ErrorInfo:
    """
    Structured error information.

    Attached to log entries with level ERROR or CRITICAL.

    Attributes:
        error_type: Exception class name or error category.
        error_message: Human-readable error description.
        error_code: Machine-readable error code (optional).
        stack_trace: Full stack trace (optional, redacted in production).
        recoverable: Whether the error can be recovered from.
        retry_after_ms: Suggested retry delay in milliseconds.
        context: Additional context about the error.
    """

    error_type: str
    error_message: str
    error_code: Optional[str] = None
    stack_trace: Optional[str] = None
    recoverable: bool = False
    retry_after_ms: Optional[int] = None
    context: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        result = {
            "error_type": self.error_type,
            "error_message": self.error_message,
            "recoverable": self.recoverable,
        }

        if self.error_code:
            result["error_code"] = self.error_code
        if self.stack_trace:
            result["stack_trace"] = self.stack_trace
        if self.retry_after_ms is not None:
            result["retry_after_ms"] = self.retry_after_ms
        if self.context:
            result["context"] = self.context

        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ErrorInfo":
        """Deserialize from dictionary."""
        return cls(
            error_type=data["error_type"],
            error_message=data["error_message"],
            error_code=data.get("error_code"),
            stack_trace=data.get("stack_trace"),
            recoverable=data.get("recoverable", False),
            retry_after_ms=data.get("retry_after_ms"),
            context=data.get("context", {}),
        )

    @classmethod
    def from_exception(cls, exc: Exception, recoverable: bool = False) -> "ErrorInfo":
        """Create ErrorInfo from an exception."""
        import traceback
        return cls(
            error_type=type(exc).__name__,
            error_message=str(exc),
            stack_trace=traceback.format_exc(),
            recoverable=recoverable,
        )
```

---

## Log Levels

Sigil v2 uses syslog-style severity levels. Choose the appropriate level based on the guidance below.

### Level Definitions

| Level | Value | Description | When to Use |
|-------|-------|-------------|-------------|
| TRACE | 0 | Finest granularity | Detailed debugging, rarely enabled |
| DEBUG | 1 | Debugging information | Development and troubleshooting |
| INFO | 2 | Normal operations | Significant state changes, milestones |
| WARNING | 3 | Potential issues | Non-blocking problems, degraded performance |
| ERROR | 4 | Errors | Operation failures that affect results |
| CRITICAL | 5 | Critical failures | System failures requiring immediate attention |

### Level Selection Guide

```python
# TRACE - Very detailed, typically disabled
logger.trace("Entering function with args", extra={
    "component": "memory",
    "operation": "rag_search",
    "metadata": {"query_embedding": [...], "k": 10}
})

# DEBUG - Detailed debugging
logger.debug("Retrieved memory items", extra={
    "component": "memory",
    "operation": "rag_search",
    "metadata": {"items_found": 15, "search_time_ms": 45}
})

# INFO - Normal operations (default for completed operations)
logger.info("Memory search completed", extra={
    "component": "memory",
    "operation": "rag_search",
    "tokens_used": TokenUsage(input_tokens=200, output_tokens=0),
    "metadata": {"items_returned": 5, "relevance_threshold": 0.75}
})

# WARNING - Potential issues
logger.warning("Memory search returned fewer items than requested", extra={
    "component": "memory",
    "operation": "rag_search",
    "metadata": {"requested": 10, "returned": 3}
})

# ERROR - Operation failures
logger.error("Memory search failed", extra={
    "component": "memory",
    "operation": "rag_search",
    "error_info": ErrorInfo(
        error_type="VectorStoreError",
        error_message="Connection to FAISS index failed",
        recoverable=True,
        retry_after_ms=1000,
    )
})

# CRITICAL - System failures
logger.critical("Memory subsystem unavailable", extra={
    "component": "memory",
    "operation": "initialization",
    "error_info": ErrorInfo(
        error_type="SystemInitializationError",
        error_message="Failed to load memory indices on startup",
        recoverable=False,
    )
})
```

---

## Component Identifiers

Each component in Sigil v2 has a unique identifier for log attribution.

### Component Registry

| Component ID | Module Path | Description |
|-------------|-------------|-------------|
| `orchestrator` | `sigil.orchestrator` | Top-level request orchestration |
| `router` | `sigil.routing` | Intent classification and routing |
| `planner` | `sigil.planning` | Task decomposition and planning |
| `reasoning` | `sigil.reasoning` | Strategy selection and execution |
| `memory` | `sigil.memory` | Memory retrieval and storage |
| `contracts` | `sigil.contracts` | Output validation and enforcement |
| `evolution` | `sigil.evolution` | Prompt optimization and learning |
| `context` | `sigil.context` | Context window management |
| `tools` | `sigil.tools` | Tool invocation and MCP integration |
| `cli` | `sigil.interfaces.cli` | Command-line interface |
| `api` | `sigil.interfaces.api` | REST/WebSocket API |
| `system` | `sigil.core` | Core infrastructure and utilities |

### Component Hierarchy

```
orchestrator
├── router
│   └── (intent classification LLM call)
├── planner
│   └── (plan generation LLM call)
├── reasoning
│   ├── direct
│   ├── chain_of_thought
│   ├── tree_of_thoughts
│   ├── react
│   └── mcts
├── memory
│   ├── resources
│   ├── items
│   ├── categories
│   └── retrieval
├── contracts
│   ├── validator
│   └── retry
├── context
│   └── compression
└── tools
    ├── mcp
    └── builtin
```

---

## Structured Metadata Schemas

Each component defines specific metadata schemas for its log entries.

### Orchestrator Metadata

```python
@dataclass
class OrchestratorMetadata:
    """Metadata for orchestrator logs."""
    request_id: str
    user_id: Optional[str] = None
    agent_name: Optional[str] = None
    execution_mode: str = "sync"  # sync, async, stream
    timeout_ms: Optional[int] = None
    retry_attempt: int = 0
    max_retries: int = 3
```

Example:
```json
{
  "request_id": "req_abc123",
  "user_id": "user_456",
  "agent_name": "lead_qualifier",
  "execution_mode": "stream",
  "timeout_ms": 30000,
  "retry_attempt": 0,
  "max_retries": 3
}
```

### Router Metadata

```python
@dataclass
class RouterMetadata:
    """Metadata for router logs."""
    intent: str
    confidence: float
    complexity: float
    handler: str
    stratum: Optional[str] = None
    routing_time_ms: float = 0.0
    classification_prompt_tokens: int = 0
```

Example:
```json
{
  "intent": "execute_task",
  "confidence": 0.92,
  "complexity": 0.65,
  "handler": "executor_rai",
  "stratum": "RAI",
  "routing_time_ms": 125.5,
  "classification_prompt_tokens": 350
}
```

### Planner Metadata

```python
@dataclass
class PlannerMetadata:
    """Metadata for planner logs."""
    goal: str
    plan_id: str
    step_count: int
    complexity_score: float
    estimated_tokens: int
    available_tools: list[str] = field(default_factory=list)
    dependencies_resolved: bool = True
```

Example:
```json
{
  "goal": "Qualify lead John from Acme Corp",
  "plan_id": "plan_xyz789",
  "step_count": 5,
  "complexity_score": 0.68,
  "estimated_tokens": 8500,
  "available_tools": ["websearch", "crm"],
  "dependencies_resolved": true
}
```

### Reasoning Metadata

```python
@dataclass
class ReasoningMetadata:
    """Metadata for reasoning logs."""
    strategy: str
    task_hash: str  # SHA256 hash for privacy
    complexity: float
    confidence: float
    branches_explored: int = 1
    depth_reached: int = 1
    fallback_used: bool = False
    reasoning_trace_length: int = 0
```

Example:
```json
{
  "strategy": "tree_of_thoughts",
  "task_hash": "sha256:a1b2c3...",
  "complexity": 0.72,
  "confidence": 0.86,
  "branches_explored": 4,
  "depth_reached": 3,
  "fallback_used": false,
  "reasoning_trace_length": 12
}
```

### Memory Metadata

```python
@dataclass
class MemoryMetadata:
    """Metadata for memory logs."""
    layer: str  # resources, items, categories
    operation_type: str  # read, write, search, consolidate
    retrieval_method: Optional[str] = None  # rag, llm, hybrid
    items_count: int = 0
    relevance_threshold: float = 0.7
    search_query_hash: Optional[str] = None
```

Example:
```json
{
  "layer": "items",
  "operation_type": "search",
  "retrieval_method": "hybrid",
  "items_count": 8,
  "relevance_threshold": 0.75,
  "search_query_hash": "sha256:d4e5f6..."
}
```

### Contracts Metadata

```python
@dataclass
class ContractsMetadata:
    """Metadata for contract logs."""
    contract_name: str
    contract_id: str
    validation_passed: bool
    deliverables_checked: list[str] = field(default_factory=list)
    deliverables_passed: list[str] = field(default_factory=list)
    deliverables_failed: list[str] = field(default_factory=list)
    retry_number: int = 0
    feedback_provided: bool = False
```

Example:
```json
{
  "contract_name": "lead_qualification",
  "contract_id": "contract_123",
  "validation_passed": false,
  "deliverables_checked": ["qualification_score", "bant_assessment", "recommended_action"],
  "deliverables_passed": ["qualification_score", "recommended_action"],
  "deliverables_failed": ["bant_assessment"],
  "retry_number": 1,
  "feedback_provided": true
}
```

### Tools Metadata

```python
@dataclass
class ToolsMetadata:
    """Metadata for tool invocation logs."""
    tool_name: str
    tool_server: Optional[str] = None  # MCP server name
    tool_params_hash: Optional[str] = None  # Hash for privacy
    execution_time_ms: float = 0.0
    result_size_bytes: int = 0
    cache_hit: bool = False
    timeout_used: bool = False
```

Example:
```json
{
  "tool_name": "websearch",
  "tool_server": "tavily-mcp",
  "tool_params_hash": "sha256:g7h8i9...",
  "execution_time_ms": 1250.0,
  "result_size_bytes": 4096,
  "cache_hit": false,
  "timeout_used": false
}
```

---

## Error Logging Patterns

### Standard Error Log Entry

```python
def log_error(
    component: ComponentId,
    operation: str,
    error: Exception,
    context: dict[str, Any],
    recoverable: bool = False,
    session_id: str = "",
    correlation_id: str = "",
) -> ExecutionLogEntry:
    """Create a standardized error log entry."""
    return ExecutionLogEntry(
        level=LogLevel.ERROR,
        component=component,
        operation=operation,
        message=f"{operation} failed: {type(error).__name__}",
        session_id=session_id,
        correlation_id=correlation_id,
        error_info=ErrorInfo.from_exception(error, recoverable=recoverable),
        metadata=context,
    )
```

### Error Categories

| Category | Error Code Prefix | Description |
|----------|------------------|-------------|
| Token Errors | `TOKEN_` | Budget exhaustion, limit exceeded |
| Contract Errors | `CONTRACT_` | Validation failures, deliverable missing |
| Tool Errors | `TOOL_` | MCP failures, timeout, unavailable |
| Memory Errors | `MEMORY_` | Retrieval failures, storage issues |
| Reasoning Errors | `REASONING_` | Strategy failures, complexity assessment |
| System Errors | `SYSTEM_` | Infrastructure, configuration issues |

### Error Code Registry

```python
class ErrorCode(str, Enum):
    """Standardized error codes for Sigil v2."""

    # Token errors
    TOKEN_BUDGET_EXHAUSTED = "TOKEN_001"
    TOKEN_LIMIT_EXCEEDED = "TOKEN_002"
    TOKEN_TRACKING_FAILED = "TOKEN_003"

    # Contract errors
    CONTRACT_VALIDATION_FAILED = "CONTRACT_001"
    CONTRACT_DELIVERABLE_MISSING = "CONTRACT_002"
    CONTRACT_RETRY_EXHAUSTED = "CONTRACT_003"
    CONTRACT_TIMEOUT = "CONTRACT_004"

    # Tool errors
    TOOL_NOT_FOUND = "TOOL_001"
    TOOL_EXECUTION_FAILED = "TOOL_002"
    TOOL_TIMEOUT = "TOOL_003"
    TOOL_MCP_CONNECTION_FAILED = "TOOL_004"
    TOOL_INVALID_PARAMS = "TOOL_005"

    # Memory errors
    MEMORY_RETRIEVAL_FAILED = "MEMORY_001"
    MEMORY_STORAGE_FAILED = "MEMORY_002"
    MEMORY_INDEX_CORRUPTED = "MEMORY_003"
    MEMORY_EMBEDDING_FAILED = "MEMORY_004"

    # Reasoning errors
    REASONING_STRATEGY_FAILED = "REASONING_001"
    REASONING_COMPLEXITY_ERROR = "REASONING_002"
    REASONING_TIMEOUT = "REASONING_003"
    REASONING_FALLBACK_EXHAUSTED = "REASONING_004"

    # System errors
    SYSTEM_INITIALIZATION_FAILED = "SYSTEM_001"
    SYSTEM_CONFIGURATION_INVALID = "SYSTEM_002"
    SYSTEM_DEPENDENCY_MISSING = "SYSTEM_003"
    SYSTEM_RESOURCE_UNAVAILABLE = "SYSTEM_004"
```

### Error Log Examples

#### Token Budget Exhausted

```python
ExecutionLogEntry(
    level=LogLevel.ERROR,
    component=ComponentId.ORCHESTRATOR,
    operation="execute_request",
    message="Token budget exhausted before completion",
    session_id="sess_abc123",
    correlation_id="corr_xyz789",
    tokens_used=TokenUsage(input_tokens=180000, output_tokens=76000),
    error_info=ErrorInfo(
        error_type="BudgetExhaustedError",
        error_message="Used 256,000/256,000 tokens. Cannot continue.",
        error_code="TOKEN_001",
        recoverable=False,
        context={
            "tokens_remaining": 0,
            "last_component": "reasoning",
            "last_operation": "tree_of_thoughts",
        }
    ),
    budget_snapshot=BudgetStatus(
        total_budget=256000,
        used=256000,
        by_component={
            "orchestrator": 5000,
            "router": 2000,
            "planner": 15000,
            "reasoning": 180000,
            "memory": 30000,
            "contracts": 14000,
            "tools": 10000,
        }
    ),
)
```

#### Contract Validation Failed

```python
ExecutionLogEntry(
    level=LogLevel.WARNING,
    component=ComponentId.CONTRACTS,
    operation="validate_output",
    message="Contract validation failed, will retry",
    session_id="sess_abc123",
    correlation_id="corr_xyz789",
    tokens_used=TokenUsage(input_tokens=2500, output_tokens=800),
    error_info=ErrorInfo(
        error_type="ContractValidationError",
        error_message="Missing required deliverable: bant_assessment",
        error_code="CONTRACT_002",
        recoverable=True,
        retry_after_ms=0,
        context={
            "deliverables_passed": ["qualification_score", "recommended_action"],
            "deliverables_failed": ["bant_assessment"],
        }
    ),
    metadata={
        "contract_name": "lead_qualification",
        "retry_number": 1,
        "max_retries": 2,
        "feedback_provided": True,
    },
)
```

#### Tool MCP Connection Failed

```python
ExecutionLogEntry(
    level=LogLevel.ERROR,
    component=ComponentId.TOOLS,
    operation="connect_mcp_server",
    message="Failed to connect to MCP server: tavily-mcp",
    session_id="sess_abc123",
    correlation_id="corr_xyz789",
    duration_ms=5000.0,
    error_info=ErrorInfo(
        error_type="MCPConnectionError",
        error_message="Connection timeout after 5000ms",
        error_code="TOOL_004",
        recoverable=True,
        retry_after_ms=2000,
        context={
            "server_name": "tavily-mcp",
            "connection_attempts": 3,
            "last_known_status": "unreachable",
        }
    ),
    metadata={
        "tool_name": "websearch",
        "tool_server": "tavily-mcp",
        "timeout_used": True,
    },
)
```

---

## Token Tracking Contract

### Per-Operation Token Reporting

Every LLM call MUST produce a `TokenUsage` record. Components collect these records and report aggregates.

```python
class TokenTrackingContract:
    """
    Contract for token tracking across Sigil v2.

    All components that make LLM calls MUST:
    1. Record TokenUsage for each call
    2. Include TokenUsage in log entries
    3. Accumulate usage across operations
    4. Report to orchestrator for budget tracking
    """

    def record_llm_call(
        self,
        response: dict,
        model: str,
        operation: str,
    ) -> TokenUsage:
        """
        Record token usage from an LLM response.

        Args:
            response: Raw LLM API response containing usage field.
            model: Model identifier (e.g., 'anthropic:claude-opus-4-5-20251101').
            operation: Operation name for attribution.

        Returns:
            TokenUsage object with extracted usage data.

        Raises:
            ValueError: If response lacks usage information.
        """
        usage = TokenUsage.from_anthropic_response(response, model)

        # Log the usage
        self.logger.info(
            f"LLM call completed: {operation}",
            extra={
                "component": self.component_id,
                "operation": operation,
                "tokens_used": usage,
            }
        )

        # Accumulate for budget tracking
        self.accumulator.add(usage)

        return usage
```

### Budget Tracking

```python
class BudgetTracker:
    """
    Tracks token consumption against the 256K budget.

    Usage:
        >>> tracker = BudgetTracker(total_budget=256_000)
        >>> tracker.consume(5000, "orchestrator")
        >>> tracker.get_status().percentage_used
        1.95
        >>> tracker.can_afford(300000)
        False
    """

    def __init__(self, total_budget: int = 256_000):
        self.status = BudgetStatus(total_budget=total_budget, used=0)
        self._lock = threading.Lock()

    def consume(self, tokens: int, component: str) -> None:
        """Consume tokens and update status."""
        with self._lock:
            self.status.consume(tokens, component)

            if self.status.critical_threshold_reached:
                self.logger.critical(
                    "Token budget critical: 95% consumed",
                    extra={
                        "component": "system",
                        "operation": "budget_tracking",
                        "budget_snapshot": self.status,
                    }
                )
            elif self.status.warning_threshold_reached:
                self.logger.warning(
                    "Token budget warning: 80% consumed",
                    extra={
                        "component": "system",
                        "operation": "budget_tracking",
                        "budget_snapshot": self.status,
                    }
                )

    def get_status(self) -> BudgetStatus:
        """Get current budget status snapshot."""
        with self._lock:
            return BudgetStatus(
                total_budget=self.status.total_budget,
                used=self.status.used,
                by_component=dict(self.status.by_component),
            )

    def can_afford(self, tokens: int) -> bool:
        """Check if budget can afford the specified tokens."""
        with self._lock:
            return self.status.can_afford(tokens)

    def get_remaining_for_component(self, component: str, allocation: float) -> int:
        """
        Get remaining tokens allocated to a component.

        Args:
            component: Component identifier.
            allocation: Percentage of total budget allocated (0.0-1.0).

        Returns:
            Remaining tokens available for this component.
        """
        component_budget = int(self.status.total_budget * allocation)
        component_used = self.status.by_component.get(component, 0)
        return max(0, component_budget - component_used)
```

### Component Budget Allocation

Default budget allocation per component:

| Component | Allocation | Max Tokens |
|-----------|------------|------------|
| reasoning | 45% | 115,200 |
| memory | 15% | 38,400 |
| planner | 12% | 30,720 |
| contracts | 10% | 25,600 |
| tools | 8% | 20,480 |
| router | 4% | 10,240 |
| context | 3% | 7,680 |
| orchestrator | 2% | 5,120 |
| evolution | 1% | 2,560 |

```python
DEFAULT_COMPONENT_ALLOCATIONS = {
    ComponentId.REASONING: 0.45,
    ComponentId.MEMORY: 0.15,
    ComponentId.PLANNER: 0.12,
    ComponentId.CONTRACTS: 0.10,
    ComponentId.TOOLS: 0.08,
    ComponentId.ROUTER: 0.04,
    ComponentId.CONTEXT: 0.03,
    ComponentId.ORCHESTRATOR: 0.02,
    ComponentId.EVOLUTION: 0.01,
}
```

---

## Streaming Protocol

### WebSocket Log Stream

```python
@dataclass
class StreamConfig:
    """Configuration for log streaming."""

    # Filtering
    filter: LogFilter = field(default_factory=LogFilter)

    # Buffering
    buffer_size: int = 100  # Max entries to buffer
    flush_interval_ms: int = 100  # Max time between flushes

    # Heartbeat
    heartbeat_interval_ms: int = 5000  # Heartbeat every 5s

    # Compression
    compress: bool = False  # Enable gzip compression


@dataclass
class StreamMessage:
    """A message in the log stream."""

    type: str  # "log", "heartbeat", "error", "buffer_flush"
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    entries: list[ExecutionLogEntry] = field(default_factory=list)
    heartbeat_data: Optional["HeartbeatData"] = None
    error: Optional[str] = None

    def to_json(self) -> str:
        """Serialize to JSON for WebSocket transmission."""
        data = {
            "type": self.type,
            "timestamp": self.timestamp.isoformat(),
        }

        if self.entries:
            data["entries"] = [e.to_dict() for e in self.entries]
        if self.heartbeat_data:
            data["heartbeat"] = self.heartbeat_data.to_dict()
        if self.error:
            data["error"] = self.error

        return json.dumps(data)


@dataclass
class HeartbeatData:
    """Data included in heartbeat messages."""

    session_id: str
    is_alive: bool = True
    current_operation: Optional[str] = None
    current_component: Optional[str] = None
    tokens_used: int = 0
    budget_remaining: int = 0
    logs_since_last_heartbeat: int = 0
    uptime_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "session_id": self.session_id,
            "is_alive": self.is_alive,
            "current_operation": self.current_operation,
            "current_component": self.current_component,
            "tokens_used": self.tokens_used,
            "budget_remaining": self.budget_remaining,
            "logs_since_last_heartbeat": self.logs_since_last_heartbeat,
            "uptime_seconds": self.uptime_seconds,
        }
```

### Stream Protocol Sequence

```
Client                                Server
  |                                      |
  |-- WS Connect ----------------------->|
  |                                      |
  |<-- Connection Accepted --------------|
  |                                      |
  |-- StreamConfig --------------------->|
  |                                      |
  |<-- Initial Heartbeat ----------------|
  |                                      |
  |<-- Log Entry 1 ----------------------|
  |<-- Log Entry 2 ----------------------|
  |<-- Log Entry 3 ----------------------|
  |                                      |
  |<-- Heartbeat (5s) -------------------|
  |                                      |
  |<-- Buffer Flush (100 entries) -------|
  |                                      |
  |-- Update Filter -------------------->|
  |                                      |
  |<-- Filter Updated -------------------|
  |                                      |
  |<-- Log Entry (filtered) -------------|
  |                                      |
  |-- Disconnect ----------------------->|
  |                                      |
```

### Buffer Flush Strategy

```python
class LogBuffer:
    """
    Buffers log entries for batch transmission.

    Flushes when:
    1. Buffer reaches capacity (buffer_size entries)
    2. Time since last flush exceeds flush_interval_ms
    3. High-priority entry received (ERROR or CRITICAL)
    """

    def __init__(
        self,
        buffer_size: int = 100,
        flush_interval_ms: int = 100,
    ):
        self.buffer: list[ExecutionLogEntry] = []
        self.buffer_size = buffer_size
        self.flush_interval_ms = flush_interval_ms
        self.last_flush = datetime.now(timezone.utc)
        self._lock = threading.Lock()
        self._flush_callback: Optional[Callable] = None

    def add(self, entry: ExecutionLogEntry) -> None:
        """Add entry to buffer, flush if needed."""
        with self._lock:
            self.buffer.append(entry)

            should_flush = (
                len(self.buffer) >= self.buffer_size
                or self._time_since_last_flush_ms() >= self.flush_interval_ms
                or entry.level in (LogLevel.ERROR, LogLevel.CRITICAL)
            )

            if should_flush:
                self._flush()

    def _flush(self) -> None:
        """Flush buffer to callback."""
        if not self.buffer:
            return

        entries = self.buffer.copy()
        self.buffer.clear()
        self.last_flush = datetime.now(timezone.utc)

        if self._flush_callback:
            self._flush_callback(entries)

    def _time_since_last_flush_ms(self) -> float:
        """Calculate milliseconds since last flush."""
        delta = datetime.now(timezone.utc) - self.last_flush
        return delta.total_seconds() * 1000
```

---

## Query and Filter Specification

### Query Operators

For metadata queries, the following operators are supported:

| Operator | Description | Example |
|----------|-------------|---------|
| `$eq` | Equals | `{"strategy": {"$eq": "tree_of_thoughts"}}` |
| `$ne` | Not equals | `{"retry_number": {"$ne": 0}}` |
| `$gt` | Greater than | `{"confidence": {"$gt": 0.8}}` |
| `$gte` | Greater than or equal | `{"tokens_used": {"$gte": 1000}}` |
| `$lt` | Less than | `{"duration_ms": {"$lt": 5000}}` |
| `$lte` | Less than or equal | `{"retry_number": {"$lte": 2}}` |
| `$in` | In array | `{"component": {"$in": ["reasoning", "memory"]}}` |
| `$nin` | Not in array | `{"level": {"$nin": ["trace", "debug"]}}` |
| `$exists` | Field exists | `{"error_info": {"$exists": true}}` |
| `$regex` | Regex match | `{"message": {"$regex": ".*failed.*"}}` |

### Query Examples

#### Find high-token operations

```json
{
  "min_tokens": 5000,
  "sort_by": "tokens_used",
  "sort_order": "desc",
  "limit": 20
}
```

#### Find errors in reasoning component

```json
{
  "components": ["reasoning"],
  "has_error": true,
  "start_time": "2026-01-11T00:00:00Z",
  "end_time": "2026-01-11T23:59:59Z"
}
```

#### Find contract validation failures

```json
{
  "components": ["contracts"],
  "operations": ["validate_output"],
  "metadata_query": {
    "validation_passed": {"$eq": false}
  }
}
```

#### Find slow tool executions

```json
{
  "components": ["tools"],
  "metadata_query": {
    "execution_time_ms": {"$gt": 2000}
  },
  "sort_by": "duration_ms",
  "sort_order": "desc"
}
```

---

## Complete Examples

### Example 1: Successful Orchestration Flow

```python
# 1. Orchestration starts
ExecutionLogEntry(
    id="log_001",
    level=LogLevel.INFO,
    component=ComponentId.ORCHESTRATOR,
    operation="start_execution",
    message="Starting agent execution: lead_qualifier",
    session_id="sess_abc123",
    correlation_id="corr_xyz789",
    operation_id="op_001",
    metadata={
        "request_id": "req_12345",
        "agent_name": "lead_qualifier",
        "execution_mode": "sync",
        "timeout_ms": 30000,
    },
    budget_snapshot=BudgetStatus(
        total_budget=256000,
        used=0,
    ),
)

# 2. Router classifies intent
ExecutionLogEntry(
    id="log_002",
    level=LogLevel.INFO,
    component=ComponentId.ROUTER,
    operation="classify_intent",
    message="Classified intent as 'execute_task' with stratum RAI",
    session_id="sess_abc123",
    correlation_id="corr_xyz789",
    operation_id="op_002",
    parent_operation_id="op_001",
    duration_ms=125.5,
    tokens_used=TokenUsage(input_tokens=350, output_tokens=50, model="anthropic:claude-opus-4-5-20251101"),
    metadata={
        "intent": "execute_task",
        "confidence": 0.92,
        "complexity": 0.65,
        "handler": "executor_rai",
        "stratum": "RAI",
    },
    budget_snapshot=BudgetStatus(
        total_budget=256000,
        used=400,
        by_component={"router": 400},
    ),
)

# 3. Planner creates plan
ExecutionLogEntry(
    id="log_003",
    level=LogLevel.INFO,
    component=ComponentId.PLANNER,
    operation="create_plan",
    message="Created plan with 5 steps for lead qualification",
    session_id="sess_abc123",
    correlation_id="corr_xyz789",
    operation_id="op_003",
    parent_operation_id="op_001",
    duration_ms=850.0,
    tokens_used=TokenUsage(input_tokens=2500, output_tokens=1200, model="anthropic:claude-opus-4-5-20251101"),
    metadata={
        "goal": "Qualify lead John from Acme Corp",
        "plan_id": "plan_xyz789",
        "step_count": 5,
        "complexity_score": 0.68,
        "estimated_tokens": 8500,
        "available_tools": ["websearch", "crm"],
    },
    budget_snapshot=BudgetStatus(
        total_budget=256000,
        used=4100,
        by_component={"router": 400, "planner": 3700},
    ),
)

# 4. Reasoning executes with Tree of Thoughts
ExecutionLogEntry(
    id="log_004",
    level=LogLevel.INFO,
    component=ComponentId.REASONING,
    operation="tree_of_thoughts",
    message="Reasoning completed: explored 4 branches, confidence 0.86",
    session_id="sess_abc123",
    correlation_id="corr_xyz789",
    operation_id="op_004",
    parent_operation_id="op_001",
    duration_ms=3200.0,
    tokens_used=TokenUsage(input_tokens=8500, output_tokens=2800, model="anthropic:claude-opus-4-5-20251101"),
    metadata={
        "strategy": "tree_of_thoughts",
        "task_hash": "sha256:a1b2c3d4e5",
        "complexity": 0.72,
        "confidence": 0.86,
        "branches_explored": 4,
        "depth_reached": 3,
    },
    budget_snapshot=BudgetStatus(
        total_budget=256000,
        used=15400,
        by_component={"router": 400, "planner": 3700, "reasoning": 11300},
    ),
)

# 5. Memory search
ExecutionLogEntry(
    id="log_005",
    level=LogLevel.INFO,
    component=ComponentId.MEMORY,
    operation="hybrid_retrieval",
    message="Retrieved 8 relevant memory items",
    session_id="sess_abc123",
    correlation_id="corr_xyz789",
    operation_id="op_005",
    parent_operation_id="op_001",
    duration_ms=450.0,
    tokens_used=TokenUsage(input_tokens=1200, output_tokens=300, model="anthropic:claude-opus-4-5-20251101"),
    metadata={
        "layer": "items",
        "operation_type": "search",
        "retrieval_method": "hybrid",
        "items_count": 8,
        "relevance_threshold": 0.75,
    },
    budget_snapshot=BudgetStatus(
        total_budget=256000,
        used=16900,
        by_component={"router": 400, "planner": 3700, "reasoning": 11300, "memory": 1500},
    ),
)

# 6. Tool invocation
ExecutionLogEntry(
    id="log_006",
    level=LogLevel.INFO,
    component=ComponentId.TOOLS,
    operation="websearch",
    message="Web search completed: found 5 relevant results",
    session_id="sess_abc123",
    correlation_id="corr_xyz789",
    operation_id="op_006",
    parent_operation_id="op_001",
    duration_ms=1250.0,
    tokens_used=TokenUsage(input_tokens=0, output_tokens=0),  # Tool doesn't consume LLM tokens
    metadata={
        "tool_name": "websearch",
        "tool_server": "tavily-mcp",
        "execution_time_ms": 1250.0,
        "result_size_bytes": 4096,
        "cache_hit": False,
    },
    budget_snapshot=BudgetStatus(
        total_budget=256000,
        used=16900,
        by_component={"router": 400, "planner": 3700, "reasoning": 11300, "memory": 1500},
    ),
)

# 7. Contract validation (first attempt fails)
ExecutionLogEntry(
    id="log_007",
    level=LogLevel.WARNING,
    component=ComponentId.CONTRACTS,
    operation="validate_output",
    message="Contract validation failed, will retry",
    session_id="sess_abc123",
    correlation_id="corr_xyz789",
    operation_id="op_007",
    parent_operation_id="op_001",
    duration_ms=50.0,
    tokens_used=TokenUsage(input_tokens=500, output_tokens=100, model="anthropic:claude-opus-4-5-20251101"),
    metadata={
        "contract_name": "lead_qualification",
        "validation_passed": False,
        "deliverables_checked": ["qualification_score", "bant_assessment", "recommended_action"],
        "deliverables_passed": ["qualification_score", "recommended_action"],
        "deliverables_failed": ["bant_assessment"],
        "retry_number": 1,
        "feedback_provided": True,
    },
    error_info=ErrorInfo(
        error_type="ContractValidationError",
        error_message="Missing required deliverable: bant_assessment",
        error_code="CONTRACT_002",
        recoverable=True,
    ),
    budget_snapshot=BudgetStatus(
        total_budget=256000,
        used=17500,
        by_component={"router": 400, "planner": 3700, "reasoning": 11300, "memory": 1500, "contracts": 600},
    ),
)

# 8. Retry reasoning
ExecutionLogEntry(
    id="log_008",
    level=LogLevel.INFO,
    component=ComponentId.REASONING,
    operation="chain_of_thought",
    message="Retry reasoning completed with feedback",
    session_id="sess_abc123",
    correlation_id="corr_xyz789",
    operation_id="op_008",
    parent_operation_id="op_001",
    duration_ms=1800.0,
    tokens_used=TokenUsage(input_tokens=4500, output_tokens=1800, model="anthropic:claude-opus-4-5-20251101"),
    metadata={
        "strategy": "chain_of_thought",
        "is_retry": True,
        "retry_feedback": "Missing bant_assessment deliverable",
        "confidence": 0.91,
    },
    budget_snapshot=BudgetStatus(
        total_budget=256000,
        used=23800,
        by_component={"router": 400, "planner": 3700, "reasoning": 17600, "memory": 1500, "contracts": 600},
    ),
)

# 9. Contract validation (second attempt succeeds)
ExecutionLogEntry(
    id="log_009",
    level=LogLevel.INFO,
    component=ComponentId.CONTRACTS,
    operation="validate_output",
    message="Contract validation passed",
    session_id="sess_abc123",
    correlation_id="corr_xyz789",
    operation_id="op_009",
    parent_operation_id="op_001",
    duration_ms=45.0,
    tokens_used=TokenUsage(input_tokens=500, output_tokens=80, model="anthropic:claude-opus-4-5-20251101"),
    metadata={
        "contract_name": "lead_qualification",
        "validation_passed": True,
        "deliverables_checked": ["qualification_score", "bant_assessment", "recommended_action"],
        "deliverables_passed": ["qualification_score", "bant_assessment", "recommended_action"],
        "deliverables_failed": [],
        "retry_number": 2,
    },
    budget_snapshot=BudgetStatus(
        total_budget=256000,
        used=24380,
        by_component={"router": 400, "planner": 3700, "reasoning": 17600, "memory": 1500, "contracts": 1180},
    ),
)

# 10. Orchestration completes
ExecutionLogEntry(
    id="log_010",
    level=LogLevel.INFO,
    component=ComponentId.ORCHESTRATOR,
    operation="complete_execution",
    message="Agent execution completed successfully",
    session_id="sess_abc123",
    correlation_id="corr_xyz789",
    operation_id="op_010",
    parent_operation_id="op_001",
    duration_ms=8650.0,
    tokens_used=TokenUsage(input_tokens=18050, output_tokens=6330, model="anthropic:claude-opus-4-5-20251101"),
    metadata={
        "request_id": "req_12345",
        "agent_name": "lead_qualifier",
        "success": True,
        "total_steps": 5,
        "retries": 1,
    },
    budget_snapshot=BudgetStatus(
        total_budget=256000,
        used=24380,
        by_component={
            "router": 400,
            "planner": 3700,
            "reasoning": 17600,
            "memory": 1500,
            "contracts": 1180,
        },
    ),
)
```

### Example 2: Token Budget Exhausted

```python
# Budget approaching limit
ExecutionLogEntry(
    id="log_warning",
    level=LogLevel.WARNING,
    component=ComponentId.SYSTEM,
    operation="budget_tracking",
    message="Token budget warning: 80% consumed",
    session_id="sess_def456",
    correlation_id="corr_uvw123",
    budget_snapshot=BudgetStatus(
        total_budget=256000,
        used=204800,
        by_component={
            "reasoning": 150000,
            "memory": 30000,
            "planner": 15000,
            "contracts": 5000,
            "router": 3000,
            "orchestrator": 1800,
        },
    ),
)

# Budget exhausted
ExecutionLogEntry(
    id="log_critical",
    level=LogLevel.CRITICAL,
    component=ComponentId.ORCHESTRATOR,
    operation="execute_request",
    message="Token budget exhausted, aborting execution",
    session_id="sess_def456",
    correlation_id="corr_uvw123",
    error_info=ErrorInfo(
        error_type="BudgetExhaustedError",
        error_message="Used 256,000/256,000 tokens. Cannot continue.",
        error_code="TOKEN_001",
        recoverable=False,
        context={
            "tokens_remaining": 0,
            "last_component": "reasoning",
            "last_operation": "mcts",
            "partial_result_available": True,
        }
    ),
    budget_snapshot=BudgetStatus(
        total_budget=256000,
        used=256000,
        by_component={
            "reasoning": 180000,
            "memory": 38000,
            "planner": 22000,
            "contracts": 10000,
            "router": 4000,
            "orchestrator": 2000,
        },
    ),
)
```

### Example 3: Streaming Log Session

```json
// Connection established
{
  "type": "heartbeat",
  "timestamp": "2026-01-11T14:30:00.000Z",
  "heartbeat": {
    "session_id": "sess_stream001",
    "is_alive": true,
    "current_operation": null,
    "current_component": null,
    "tokens_used": 0,
    "budget_remaining": 256000,
    "logs_since_last_heartbeat": 0,
    "uptime_seconds": 0.0
  }
}

// Log entries arrive
{
  "type": "log",
  "timestamp": "2026-01-11T14:30:01.125Z",
  "entries": [
    {
      "id": "log_stream_001",
      "timestamp": "2026-01-11T14:30:01.120Z",
      "level": "info",
      "component": "orchestrator",
      "operation": "start_execution",
      "message": "Starting agent execution: research_agent",
      "session_id": "sess_stream001",
      "correlation_id": "corr_stream_abc",
      "operation_id": "op_stream_001"
    }
  ]
}

// Buffer flush with multiple entries
{
  "type": "buffer_flush",
  "timestamp": "2026-01-11T14:30:02.500Z",
  "entries": [
    {
      "id": "log_stream_002",
      "timestamp": "2026-01-11T14:30:01.500Z",
      "level": "info",
      "component": "router",
      "operation": "classify_intent",
      "message": "Classified intent as 'research_task'",
      "tokens_used": {"input_tokens": 300, "output_tokens": 40, "total_tokens": 340}
    },
    {
      "id": "log_stream_003",
      "timestamp": "2026-01-11T14:30:02.100Z",
      "level": "info",
      "component": "planner",
      "operation": "create_plan",
      "message": "Created plan with 3 steps",
      "tokens_used": {"input_tokens": 1800, "output_tokens": 900, "total_tokens": 2700}
    }
  ]
}

// Heartbeat with activity
{
  "type": "heartbeat",
  "timestamp": "2026-01-11T14:30:05.000Z",
  "heartbeat": {
    "session_id": "sess_stream001",
    "is_alive": true,
    "current_operation": "tree_of_thoughts",
    "current_component": "reasoning",
    "tokens_used": 3040,
    "budget_remaining": 252960,
    "logs_since_last_heartbeat": 3,
    "uptime_seconds": 5.0
  }
}
```

---

## Implementation Notes

### Logger Configuration

```python
import logging
import json
from datetime import datetime, timezone


class SigilLogHandler(logging.Handler):
    """
    Custom log handler that produces ExecutionLogEntry objects.
    """

    def __init__(self, buffer: LogBuffer):
        super().__init__()
        self.buffer = buffer

    def emit(self, record: logging.LogRecord) -> None:
        """Convert LogRecord to ExecutionLogEntry and buffer."""
        entry = ExecutionLogEntry(
            level=self._map_level(record.levelno),
            component=ComponentId(getattr(record, 'component', 'system')),
            operation=getattr(record, 'operation', 'unknown'),
            message=record.getMessage(),
            session_id=getattr(record, 'session_id', ''),
            correlation_id=getattr(record, 'correlation_id', ''),
            operation_id=getattr(record, 'operation_id', ''),
            parent_operation_id=getattr(record, 'parent_operation_id', None),
            duration_ms=getattr(record, 'duration_ms', None),
            tokens_used=getattr(record, 'tokens_used', None),
            metadata=getattr(record, 'metadata', {}),
            error_info=getattr(record, 'error_info', None),
            budget_snapshot=getattr(record, 'budget_snapshot', None),
        )
        self.buffer.add(entry)

    def _map_level(self, levelno: int) -> LogLevel:
        """Map Python log levels to Sigil log levels."""
        if levelno <= logging.DEBUG:
            return LogLevel.DEBUG
        elif levelno <= logging.INFO:
            return LogLevel.INFO
        elif levelno <= logging.WARNING:
            return LogLevel.WARNING
        elif levelno <= logging.ERROR:
            return LogLevel.ERROR
        else:
            return LogLevel.CRITICAL


def configure_sigil_logging() -> logging.Logger:
    """Configure logging for Sigil v2."""
    logger = logging.getLogger('sigil')
    logger.setLevel(logging.DEBUG)

    buffer = LogBuffer(buffer_size=100, flush_interval_ms=100)
    handler = SigilLogHandler(buffer)
    logger.addHandler(handler)

    return logger
```

### Context Manager for Operations

```python
from contextlib import asynccontextmanager
import time


@asynccontextmanager
async def operation_context(
    logger: logging.Logger,
    component: ComponentId,
    operation: str,
    session_id: str,
    correlation_id: str,
    parent_operation_id: Optional[str] = None,
):
    """
    Context manager for tracking operations.

    Automatically logs start and completion, tracks duration,
    and handles errors.

    Usage:
        async with operation_context(logger, ComponentId.REASONING, "tree_of_thoughts", session_id, correlation_id) as ctx:
            result = await execute_reasoning()
            ctx.set_tokens(result.tokens_used)
            ctx.set_metadata({"confidence": result.confidence})
    """
    operation_id = str(uuid.uuid4())
    start_time = time.monotonic()

    class OperationContext:
        def __init__(self):
            self.tokens_used: Optional[TokenUsage] = None
            self.metadata: dict[str, Any] = {}

        def set_tokens(self, tokens: TokenUsage) -> None:
            self.tokens_used = tokens

        def set_metadata(self, metadata: dict[str, Any]) -> None:
            self.metadata.update(metadata)

    ctx = OperationContext()

    # Log start
    logger.info(
        f"Starting {operation}",
        extra={
            "component": component.value,
            "operation": operation,
            "session_id": session_id,
            "correlation_id": correlation_id,
            "operation_id": operation_id,
            "parent_operation_id": parent_operation_id,
        }
    )

    try:
        yield ctx

        # Log success
        duration_ms = (time.monotonic() - start_time) * 1000
        logger.info(
            f"Completed {operation}",
            extra={
                "component": component.value,
                "operation": operation,
                "session_id": session_id,
                "correlation_id": correlation_id,
                "operation_id": operation_id,
                "parent_operation_id": parent_operation_id,
                "duration_ms": duration_ms,
                "tokens_used": ctx.tokens_used,
                "metadata": ctx.metadata,
            }
        )
    except Exception as e:
        # Log error
        duration_ms = (time.monotonic() - start_time) * 1000
        logger.error(
            f"Failed {operation}: {type(e).__name__}",
            extra={
                "component": component.value,
                "operation": operation,
                "session_id": session_id,
                "correlation_id": correlation_id,
                "operation_id": operation_id,
                "parent_operation_id": parent_operation_id,
                "duration_ms": duration_ms,
                "error_info": ErrorInfo.from_exception(e),
                "metadata": ctx.metadata,
            }
        )
        raise
```

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2026-01-11 | Initial release |

---

## References

- [Sigil v2 Architecture Document](/Users/zidane/Bland-Agents-Dataset/Sigil_V2.md)
- [Token Budget System](/Users/zidane/Bland-Agents-Dataset/acti-agent-builder/sigil/telemetry/tokens.py)
- [Event System](/Users/zidane/Bland-Agents-Dataset/acti-agent-builder/sigil/config/schemas/events.py)
- [Event Store](/Users/zidane/Bland-Agents-Dataset/acti-agent-builder/sigil/state/store.py)
