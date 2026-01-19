# Monitoring Implementation Guide for Sigil v2

## Overview

This guide provides comprehensive instructions for integrating the logging and monitoring system into Sigil v2 components. It covers structured logging patterns, token tracking best practices, performance considerations, and component-specific implementation examples.

**Version:** 1.0.0
**Last Updated:** 2026-01-11
**Target Audience:** Sigil v2 developers and contributors

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Core Concepts](#core-concepts)
3. [Logger Setup](#logger-setup)
4. [Structured Logging Patterns](#structured-logging-patterns)
5. [Token Tracking Integration](#token-tracking-integration)
6. [Component-Specific Examples](#component-specific-examples)
7. [Performance Considerations](#performance-considerations)
8. [Testing Logging](#testing-logging)
9. [Troubleshooting](#troubleshooting)
10. [Reference](#reference)

---

## Quick Start

### Minimal Integration (5 minutes)

Add logging to any component in three steps:

```python
# Step 1: Import the logging utilities
from sigil.telemetry.logging import get_component_logger, operation_context
from sigil.telemetry.tokens import TokenUsage
from sigil.logging_contracts import ComponentId, LogLevel

# Step 2: Get a component-specific logger
logger = get_component_logger(ComponentId.REASONING)

# Step 3: Log operations with context
async def execute_reasoning(task: str, session_id: str) -> dict:
    async with operation_context(
        logger=logger,
        component=ComponentId.REASONING,
        operation="chain_of_thought",
        session_id=session_id,
        correlation_id=session_id,  # Use session_id or generate unique
    ) as ctx:
        # Execute your operation
        result = await self._do_reasoning(task)

        # Report token usage
        ctx.set_tokens(TokenUsage(
            input_tokens=result.usage.input_tokens,
            output_tokens=result.usage.output_tokens,
            model="anthropic:claude-opus-4-5-20251101",
        ))

        # Add operation-specific metadata
        ctx.set_metadata({
            "strategy": "chain_of_thought",
            "confidence": result.confidence,
            "reasoning_steps": len(result.trace),
        })

        return result.output
```

That's it! The context manager handles:
- Start/completion logging
- Duration tracking
- Error handling
- Token accumulation
- Budget checking

---

## Core Concepts

### The Logging Stack

```
┌─────────────────────────────────────────────────────────────────┐
│                     Application Code                             │
│  logger.info("Operation completed", extra={...})                │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   SigilLogHandler                                │
│  Converts LogRecord → ExecutionLogEntry                         │
│  Validates schema compliance                                    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      LogBuffer                                   │
│  Buffers entries for batch processing                           │
│  Flushes on capacity, timeout, or priority                      │
└─────────────────────────────────────────────────────────────────┘
                              │
                   ┌──────────┴──────────┐
                   ▼                      ▼
┌──────────────────────────┐  ┌──────────────────────────┐
│      File Store          │  │    WebSocket Stream      │
│  JSON/JSONL files        │  │  Real-time to clients    │
└──────────────────────────┘  └──────────────────────────┘
```

### Key Abstractions

| Abstraction | Purpose | Location |
|-------------|---------|----------|
| `ExecutionLogEntry` | Canonical log format | `sigil/logging_contracts.py` |
| `TokenUsage` | Per-operation token tracking | `sigil/telemetry/tokens.py` |
| `BudgetTracker` | Session-level budget management | `sigil/telemetry/budget.py` |
| `LogBuffer` | Batching and streaming | `sigil/telemetry/buffer.py` |
| `operation_context` | Automatic operation lifecycle | `sigil/telemetry/logging.py` |

### The Three Rules

1. **Every LLM call MUST report TokenUsage**
2. **Every operation MUST log start and completion**
3. **Every error MUST include ErrorInfo**

---

## Logger Setup

### Module-Level Configuration

```python
# sigil/telemetry/logging.py

import logging
from typing import Optional
from contextlib import asynccontextmanager
import uuid
import time

from sigil.logging_contracts import (
    ComponentId,
    LogLevel,
    ExecutionLogEntry,
    TokenUsage,
    ErrorInfo,
    BudgetStatus,
)


class SigilLogHandler(logging.Handler):
    """
    Custom log handler that produces ExecutionLogEntry objects.

    Converts standard Python logging calls to Sigil's structured format.
    """

    def __init__(self, buffer: "LogBuffer", budget_tracker: "BudgetTracker"):
        super().__init__()
        self.buffer = buffer
        self.budget_tracker = budget_tracker

    def emit(self, record: logging.LogRecord) -> None:
        """Convert LogRecord to ExecutionLogEntry and buffer."""
        try:
            entry = self._create_entry(record)
            self.buffer.add(entry)
        except Exception:
            self.handleError(record)

    def _create_entry(self, record: logging.LogRecord) -> ExecutionLogEntry:
        """Create ExecutionLogEntry from LogRecord."""
        # Extract structured data from extra
        tokens_used = getattr(record, 'tokens_used', None)
        error_info = getattr(record, 'error_info', None)
        metadata = getattr(record, 'metadata', {})

        # Get budget snapshot if tokens were used
        budget_snapshot = None
        if tokens_used:
            budget_snapshot = self.budget_tracker.get_status()

        return ExecutionLogEntry(
            level=self._map_level(record.levelno),
            component=ComponentId(getattr(record, 'component', 'system')),
            operation=getattr(record, 'operation', 'unknown'),
            message=record.getMessage(),
            session_id=getattr(record, 'session_id', ''),
            correlation_id=getattr(record, 'correlation_id', ''),
            operation_id=getattr(record, 'operation_id', str(uuid.uuid4())),
            parent_operation_id=getattr(record, 'parent_operation_id', None),
            duration_ms=getattr(record, 'duration_ms', None),
            tokens_used=tokens_used,
            metadata=metadata,
            error_info=error_info,
            budget_snapshot=budget_snapshot,
        )

    def _map_level(self, levelno: int) -> LogLevel:
        """Map Python log levels to Sigil log levels."""
        if levelno <= 5:  # Custom TRACE level
            return LogLevel.TRACE
        elif levelno <= logging.DEBUG:
            return LogLevel.DEBUG
        elif levelno <= logging.INFO:
            return LogLevel.INFO
        elif levelno <= logging.WARNING:
            return LogLevel.WARNING
        elif levelno <= logging.ERROR:
            return LogLevel.ERROR
        else:
            return LogLevel.CRITICAL


# Global instances
_buffer: Optional["LogBuffer"] = None
_budget_tracker: Optional["BudgetTracker"] = None
_loggers: dict[str, logging.Logger] = {}


def initialize_logging(
    buffer_size: int = 100,
    flush_interval_ms: int = 100,
    total_budget: int = 256_000,
) -> None:
    """
    Initialize the Sigil logging system.

    Call once at application startup before any logging occurs.

    Args:
        buffer_size: Maximum log entries to buffer before flush.
        flush_interval_ms: Maximum time between buffer flushes.
        total_budget: Total token budget for the session.
    """
    global _buffer, _budget_tracker

    from sigil.telemetry.buffer import LogBuffer
    from sigil.telemetry.budget import BudgetTracker

    _buffer = LogBuffer(
        buffer_size=buffer_size,
        flush_interval_ms=flush_interval_ms,
    )

    _budget_tracker = BudgetTracker(total_budget=total_budget)


def get_component_logger(component: ComponentId) -> logging.Logger:
    """
    Get a logger for a specific component.

    Args:
        component: The component identifier.

    Returns:
        A configured logger for the component.
    """
    global _buffer, _budget_tracker, _loggers

    if _buffer is None or _budget_tracker is None:
        initialize_logging()

    logger_name = f"sigil.{component.value}"

    if logger_name not in _loggers:
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.DEBUG)

        # Add our custom handler
        handler = SigilLogHandler(_buffer, _budget_tracker)
        logger.addHandler(handler)

        # Prevent propagation to root logger
        logger.propagate = False

        _loggers[logger_name] = logger

    return _loggers[logger_name]


def get_budget_tracker() -> "BudgetTracker":
    """Get the global budget tracker."""
    global _budget_tracker
    if _budget_tracker is None:
        initialize_logging()
    return _budget_tracker
```

### Operation Context Manager

```python
# sigil/telemetry/context.py

from contextlib import asynccontextmanager
from typing import Any, Optional
import uuid
import time

from sigil.logging_contracts import (
    ComponentId,
    TokenUsage,
    ErrorInfo,
)


class OperationContext:
    """Context for tracking operation lifecycle."""

    def __init__(self):
        self.tokens_used: Optional[TokenUsage] = None
        self.metadata: dict[str, Any] = {}
        self._accumulated_tokens: list[TokenUsage] = []

    def set_tokens(self, tokens: TokenUsage) -> None:
        """Set token usage for this operation."""
        self.tokens_used = tokens
        self._accumulated_tokens.append(tokens)

    def add_tokens(self, tokens: TokenUsage) -> None:
        """Add additional token usage (for operations with multiple LLM calls)."""
        self._accumulated_tokens.append(tokens)

        # Update total
        if self.tokens_used is None:
            self.tokens_used = tokens
        else:
            self.tokens_used = self.tokens_used + tokens

    def set_metadata(self, metadata: dict[str, Any]) -> None:
        """Set operation metadata."""
        self.metadata.update(metadata)

    def get_total_tokens(self) -> TokenUsage:
        """Get total tokens across all LLM calls."""
        if not self._accumulated_tokens:
            return TokenUsage.zero()

        total = self._accumulated_tokens[0]
        for usage in self._accumulated_tokens[1:]:
            total = total + usage
        return total


@asynccontextmanager
async def operation_context(
    logger: "logging.Logger",
    component: ComponentId,
    operation: str,
    session_id: str,
    correlation_id: str,
    parent_operation_id: Optional[str] = None,
    log_start: bool = True,
    log_completion: bool = True,
):
    """
    Context manager for tracking operations with automatic logging.

    Handles:
    - Start logging (optional)
    - Duration tracking
    - Token accumulation
    - Completion logging
    - Error handling with ErrorInfo
    - Budget tracking

    Args:
        logger: Component logger to use.
        component: Component identifier.
        operation: Operation name (snake_case).
        session_id: Session identifier.
        correlation_id: Correlation chain identifier.
        parent_operation_id: Parent operation for nesting.
        log_start: Whether to log operation start.
        log_completion: Whether to log operation completion.

    Yields:
        OperationContext for setting tokens and metadata.

    Example:
        async with operation_context(
            logger, ComponentId.REASONING, "tree_of_thoughts",
            session_id, correlation_id
        ) as ctx:
            result = await do_work()
            ctx.set_tokens(result.tokens)
            ctx.set_metadata({"confidence": result.confidence})
    """
    operation_id = str(uuid.uuid4())
    start_time = time.monotonic()
    ctx = OperationContext()

    # Log start
    if log_start:
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

        # Calculate duration
        duration_ms = (time.monotonic() - start_time) * 1000

        # Update budget tracker
        if ctx.tokens_used:
            budget_tracker = get_budget_tracker()
            budget_tracker.consume(
                ctx.tokens_used.total_tokens,
                component.value,
            )

        # Log completion
        if log_completion:
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
        # Calculate duration even on error
        duration_ms = (time.monotonic() - start_time) * 1000

        # Create error info
        error_info = ErrorInfo.from_exception(e, recoverable=False)

        # Log error
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
                "error_info": error_info,
                "metadata": ctx.metadata,
            }
        )

        # Re-raise the exception
        raise
```

---

## Structured Logging Patterns

### Pattern 1: Simple Operation Logging

For operations that don't involve LLM calls:

```python
from sigil.telemetry.logging import get_component_logger
from sigil.logging_contracts import ComponentId

logger = get_component_logger(ComponentId.MEMORY)

def load_memory_index(index_path: str) -> None:
    """Load memory index from disk."""
    logger.info(
        "Loading memory index",
        extra={
            "component": "memory",
            "operation": "load_index",
            "metadata": {
                "index_path": index_path,
            }
        }
    )

    try:
        # Load the index
        index = _load_faiss_index(index_path)

        logger.info(
            "Memory index loaded successfully",
            extra={
                "component": "memory",
                "operation": "load_index",
                "metadata": {
                    "index_path": index_path,
                    "vector_count": index.ntotal,
                    "dimension": index.d,
                }
            }
        )
    except Exception as e:
        logger.error(
            f"Failed to load memory index: {e}",
            extra={
                "component": "memory",
                "operation": "load_index",
                "error_info": ErrorInfo.from_exception(e),
                "metadata": {
                    "index_path": index_path,
                }
            }
        )
        raise
```

### Pattern 2: LLM Call with Token Tracking

For operations that make LLM calls:

```python
from sigil.telemetry.logging import get_component_logger, operation_context
from sigil.telemetry.tokens import TokenUsage
from sigil.logging_contracts import ComponentId

logger = get_component_logger(ComponentId.REASONING)

async def execute_chain_of_thought(
    task: str,
    context: dict,
    session_id: str,
    correlation_id: str,
) -> ReasoningResult:
    """Execute chain-of-thought reasoning."""

    async with operation_context(
        logger=logger,
        component=ComponentId.REASONING,
        operation="chain_of_thought",
        session_id=session_id,
        correlation_id=correlation_id,
    ) as ctx:
        # Build prompt
        prompt = build_cot_prompt(task, context)

        # Call LLM
        response = await llm.generate(
            messages=[{"role": "user", "content": prompt}],
            model="anthropic:claude-opus-4-5-20251101",
        )

        # Extract token usage from response
        ctx.set_tokens(TokenUsage.from_anthropic_response(
            response.raw_response,
            model="anthropic:claude-opus-4-5-20251101",
        ))

        # Parse result
        result = parse_cot_response(response.content)

        # Add metadata
        ctx.set_metadata({
            "strategy": "chain_of_thought",
            "reasoning_steps": len(result.steps),
            "confidence": result.confidence,
        })

        return result
```

### Pattern 3: Multi-Step Operation with Token Accumulation

For operations with multiple LLM calls:

```python
from sigil.telemetry.logging import get_component_logger, operation_context
from sigil.telemetry.tokens import TokenUsage
from sigil.logging_contracts import ComponentId

logger = get_component_logger(ComponentId.REASONING)

async def execute_tree_of_thoughts(
    task: str,
    context: dict,
    num_branches: int,
    session_id: str,
    correlation_id: str,
) -> ReasoningResult:
    """Execute tree-of-thoughts reasoning with multiple branches."""

    async with operation_context(
        logger=logger,
        component=ComponentId.REASONING,
        operation="tree_of_thoughts",
        session_id=session_id,
        correlation_id=correlation_id,
    ) as ctx:
        branches = []

        # Step 1: Generate initial thoughts
        for i in range(num_branches):
            logger.debug(
                f"Generating branch {i+1}/{num_branches}",
                extra={
                    "component": "reasoning",
                    "operation": "tree_of_thoughts",
                    "session_id": session_id,
                    "metadata": {"branch_index": i},
                }
            )

            response = await llm.generate(
                messages=[{"role": "user", "content": generate_branch_prompt(task, i)}],
            )

            # Add tokens from this call
            ctx.add_tokens(TokenUsage.from_anthropic_response(
                response.raw_response,
                model="anthropic:claude-opus-4-5-20251101",
            ))

            branches.append(parse_branch(response.content))

        # Step 2: Evaluate branches
        evaluation_prompt = build_evaluation_prompt(task, branches)
        evaluation_response = await llm.generate(
            messages=[{"role": "user", "content": evaluation_prompt}],
        )

        ctx.add_tokens(TokenUsage.from_anthropic_response(
            evaluation_response.raw_response,
            model="anthropic:claude-opus-4-5-20251101",
        ))

        # Step 3: Select best branch
        best_branch = select_best_branch(evaluation_response.content, branches)

        # Final metadata
        ctx.set_metadata({
            "strategy": "tree_of_thoughts",
            "branches_explored": num_branches,
            "best_branch_index": best_branch.index,
            "confidence": best_branch.confidence,
            "llm_calls": num_branches + 1,  # branches + evaluation
        })

        return best_branch.result
```

### Pattern 4: Nested Operations with Parent Tracking

For operations that contain sub-operations:

```python
from sigil.telemetry.logging import get_component_logger, operation_context
from sigil.logging_contracts import ComponentId

logger = get_component_logger(ComponentId.ORCHESTRATOR)

async def orchestrate_request(
    request: str,
    session_id: str,
) -> dict:
    """Orchestrate a complete request through all components."""

    correlation_id = str(uuid.uuid4())

    async with operation_context(
        logger=logger,
        component=ComponentId.ORCHESTRATOR,
        operation="orchestrate_request",
        session_id=session_id,
        correlation_id=correlation_id,
    ) as orchestrator_ctx:
        # Step 1: Route the request
        route_result = await route_request(
            request=request,
            session_id=session_id,
            correlation_id=correlation_id,
            parent_operation_id=orchestrator_ctx.operation_id,
        )

        # Step 2: Plan execution
        plan = await create_plan(
            goal=request,
            handler=route_result.handler,
            session_id=session_id,
            correlation_id=correlation_id,
            parent_operation_id=orchestrator_ctx.operation_id,
        )

        # Step 3: Execute plan steps
        results = []
        for step in plan.steps:
            step_result = await execute_step(
                step=step,
                session_id=session_id,
                correlation_id=correlation_id,
                parent_operation_id=orchestrator_ctx.operation_id,
            )
            results.append(step_result)

            # Accumulate tokens from sub-operations
            if step_result.tokens_used:
                orchestrator_ctx.add_tokens(step_result.tokens_used)

        # Final metadata
        orchestrator_ctx.set_metadata({
            "handler": route_result.handler,
            "plan_steps": len(plan.steps),
            "steps_completed": len(results),
        })

        return {"results": results}


async def route_request(
    request: str,
    session_id: str,
    correlation_id: str,
    parent_operation_id: str,
) -> RouteResult:
    """Route request to appropriate handler."""

    router_logger = get_component_logger(ComponentId.ROUTER)

    async with operation_context(
        logger=router_logger,
        component=ComponentId.ROUTER,
        operation="classify_intent",
        session_id=session_id,
        correlation_id=correlation_id,
        parent_operation_id=parent_operation_id,  # Link to parent
    ) as ctx:
        # Classification logic...
        result = await classify(request)
        ctx.set_tokens(result.tokens_used)
        ctx.set_metadata({
            "intent": result.intent,
            "confidence": result.confidence,
        })
        return result
```

### Pattern 5: Error Handling with Recovery

For operations that may fail and retry:

```python
from sigil.telemetry.logging import get_component_logger
from sigil.logging_contracts import ComponentId, ErrorInfo

logger = get_component_logger(ComponentId.CONTRACTS)

async def validate_with_retry(
    output: dict,
    contract: Contract,
    session_id: str,
    correlation_id: str,
    max_retries: int = 2,
) -> ValidationResult:
    """Validate output against contract with retry on failure."""

    for attempt in range(max_retries + 1):
        try:
            result = validate_output(output, contract)

            if result.passed:
                logger.info(
                    "Contract validation passed",
                    extra={
                        "component": "contracts",
                        "operation": "validate_output",
                        "session_id": session_id,
                        "correlation_id": correlation_id,
                        "metadata": {
                            "contract_name": contract.name,
                            "validation_passed": True,
                            "attempt": attempt + 1,
                        }
                    }
                )
                return result
            else:
                # Validation failed but no exception
                error_info = ErrorInfo(
                    error_type="ContractValidationError",
                    error_message=f"Missing deliverables: {result.missing}",
                    error_code="CONTRACT_002",
                    recoverable=attempt < max_retries,
                    retry_after_ms=0,
                    context={
                        "deliverables_passed": result.passed_deliverables,
                        "deliverables_failed": result.missing,
                    }
                )

                logger.warning(
                    f"Contract validation failed, attempt {attempt + 1}/{max_retries + 1}",
                    extra={
                        "component": "contracts",
                        "operation": "validate_output",
                        "session_id": session_id,
                        "correlation_id": correlation_id,
                        "error_info": error_info,
                        "metadata": {
                            "contract_name": contract.name,
                            "validation_passed": False,
                            "attempt": attempt + 1,
                            "will_retry": attempt < max_retries,
                        }
                    }
                )

                if attempt < max_retries:
                    # Retry with feedback
                    output = await regenerate_with_feedback(
                        output, result.feedback, contract
                    )
                    continue
                else:
                    return result

        except Exception as e:
            error_info = ErrorInfo.from_exception(e, recoverable=attempt < max_retries)

            logger.error(
                f"Contract validation error: {e}",
                extra={
                    "component": "contracts",
                    "operation": "validate_output",
                    "session_id": session_id,
                    "correlation_id": correlation_id,
                    "error_info": error_info,
                    "metadata": {
                        "contract_name": contract.name,
                        "attempt": attempt + 1,
                    }
                }
            )

            if attempt >= max_retries:
                raise

    # Should not reach here
    raise RuntimeError("Validation loop exited unexpectedly")
```

---

## Token Tracking Integration

### The Token Tracking Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                       LLM Call                                   │
│  response = await llm.generate(...)                             │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   Extract TokenUsage                             │
│  usage = TokenUsage.from_anthropic_response(response, model)    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                  Report to Context                               │
│  ctx.add_tokens(usage)                                          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│               Include in Log Entry                               │
│  Log entry includes tokens_used field                           │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│               Update Budget Tracker                              │
│  budget_tracker.consume(tokens, component)                      │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│             Check Budget Thresholds                              │
│  Warning at 80%, Critical at 95%                                │
└─────────────────────────────────────────────────────────────────┘
```

### Extracting TokenUsage from LLM Responses

#### Anthropic API

```python
from sigil.telemetry.tokens import TokenUsage

def extract_anthropic_tokens(response: dict, model: str) -> TokenUsage:
    """Extract token usage from Anthropic API response."""
    usage = response.get("usage", {})

    return TokenUsage(
        input_tokens=usage.get("input_tokens", 0),
        output_tokens=usage.get("output_tokens", 0),
        model=model,
        cache_read_tokens=usage.get("cache_read_input_tokens", 0),
        cache_write_tokens=usage.get("cache_creation_input_tokens", 0),
    )

# With the Anthropic SDK
from anthropic import Anthropic

client = Anthropic()
response = client.messages.create(
    model="claude-opus-4-5-20251101",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Hello"}],
)

usage = TokenUsage(
    input_tokens=response.usage.input_tokens,
    output_tokens=response.usage.output_tokens,
    model=f"anthropic:{response.model}",
    cache_read_tokens=getattr(response.usage, "cache_read_input_tokens", 0),
    cache_write_tokens=getattr(response.usage, "cache_creation_input_tokens", 0),
)
```

#### LangChain Integration

```python
from langchain_anthropic import ChatAnthropic
from langchain_core.callbacks import BaseCallbackHandler
from sigil.telemetry.tokens import TokenUsage

class TokenTrackingCallback(BaseCallbackHandler):
    """LangChain callback for token tracking."""

    def __init__(self):
        self.usage: TokenUsage = TokenUsage.zero()

    def on_llm_end(self, response, **kwargs):
        """Extract tokens when LLM completes."""
        if hasattr(response, 'llm_output') and response.llm_output:
            usage = response.llm_output.get('usage', {})
            self.usage = TokenUsage(
                input_tokens=usage.get('input_tokens', 0),
                output_tokens=usage.get('output_tokens', 0),
                model=response.llm_output.get('model', ''),
            )

# Usage
callback = TokenTrackingCallback()
llm = ChatAnthropic(model="claude-opus-4-5-20251101", callbacks=[callback])

result = await llm.ainvoke([{"role": "user", "content": "Hello"}])
print(f"Tokens used: {callback.usage.total_tokens}")
```

#### DeepAgents Integration

```python
from deepagents import Agent
from sigil.telemetry.tokens import TokenUsage

class TokenTrackingAgent(Agent):
    """Agent wrapper that tracks token usage."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._last_usage: TokenUsage = TokenUsage.zero()

    async def run(self, input: str) -> dict:
        """Run agent and track tokens."""
        result = await super().run(input)

        # Extract usage from the result
        if hasattr(result, 'usage'):
            self._last_usage = TokenUsage(
                input_tokens=result.usage.input_tokens,
                output_tokens=result.usage.output_tokens,
                model=self.model,
            )

        return result

    @property
    def last_usage(self) -> TokenUsage:
        """Get token usage from last run."""
        return self._last_usage
```

### Budget Management

```python
from sigil.telemetry.budget import BudgetTracker, BudgetExhaustedError
from sigil.logging_contracts import BudgetStatus

class BudgetTracker:
    """
    Tracks token consumption against session budget.

    Thread-safe for concurrent operations.
    """

    def __init__(
        self,
        total_budget: int = 256_000,
        warning_threshold: float = 0.80,
        critical_threshold: float = 0.95,
    ):
        self.status = BudgetStatus(
            total_budget=total_budget,
            used=0,
        )
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold
        self._lock = threading.Lock()
        self._warning_logged = False
        self._critical_logged = False

    def consume(self, tokens: int, component: str) -> None:
        """
        Consume tokens from budget.

        Args:
            tokens: Number of tokens to consume.
            component: Component consuming the tokens.

        Raises:
            BudgetExhaustedError: If budget is exceeded.
        """
        with self._lock:
            # Check if we can afford
            if tokens > self.status.remaining:
                raise BudgetExhaustedError(
                    f"Insufficient budget: need {tokens}, have {self.status.remaining}"
                )

            # Consume
            self.status.consume(tokens, component)

            # Check thresholds
            self._check_thresholds()

    def _check_thresholds(self) -> None:
        """Check and log budget threshold warnings."""
        utilization = self.status.used / self.status.total_budget

        if utilization >= self.critical_threshold and not self._critical_logged:
            logger.critical(
                f"Token budget CRITICAL: {utilization*100:.1f}% consumed",
                extra={
                    "component": "system",
                    "operation": "budget_tracking",
                    "budget_snapshot": self.status,
                }
            )
            self._critical_logged = True
        elif utilization >= self.warning_threshold and not self._warning_logged:
            logger.warning(
                f"Token budget warning: {utilization*100:.1f}% consumed",
                extra={
                    "component": "system",
                    "operation": "budget_tracking",
                    "budget_snapshot": self.status,
                }
            )
            self._warning_logged = True

    def can_afford(self, tokens: int) -> bool:
        """Check if budget can afford the specified tokens."""
        with self._lock:
            return self.status.remaining >= tokens

    def get_status(self) -> BudgetStatus:
        """Get current budget status snapshot."""
        with self._lock:
            return BudgetStatus(
                total_budget=self.status.total_budget,
                used=self.status.used,
                by_component=dict(self.status.by_component),
            )

    def reserve(self, tokens: int, component: str) -> "BudgetReservation":
        """
        Reserve tokens for a future operation.

        Useful when you need to ensure budget is available before
        starting an expensive operation.

        Args:
            tokens: Number of tokens to reserve.
            component: Component requesting reservation.

        Returns:
            BudgetReservation context manager.
        """
        return BudgetReservation(self, tokens, component)


class BudgetReservation:
    """Context manager for budget reservations."""

    def __init__(self, tracker: BudgetTracker, tokens: int, component: str):
        self.tracker = tracker
        self.tokens = tokens
        self.component = component
        self.consumed = 0

    def __enter__(self):
        # Check availability (but don't consume yet)
        if not self.tracker.can_afford(self.tokens):
            raise BudgetExhaustedError(
                f"Cannot reserve {self.tokens} tokens, only {self.tracker.status.remaining} available"
            )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # If operation succeeded and didn't consume, consume the reservation
        if exc_type is None and self.consumed == 0:
            self.tracker.consume(self.tokens, self.component)

    def consume_actual(self, actual_tokens: int) -> None:
        """Consume actual tokens used (may differ from reservation)."""
        self.tracker.consume(actual_tokens, self.component)
        self.consumed = actual_tokens
```

### Pre-Flight Budget Checks

Before starting expensive operations, check budget availability:

```python
async def execute_tree_of_thoughts(
    task: str,
    branches: int,
    session_id: str,
) -> ReasoningResult:
    """Execute tree-of-thoughts with budget pre-check."""

    # Estimate token cost
    estimated_tokens = estimate_tot_tokens(task, branches)

    # Get budget tracker
    budget = get_budget_tracker()

    # Pre-flight check
    if not budget.can_afford(estimated_tokens):
        logger.warning(
            "Insufficient budget for tree_of_thoughts, falling back to chain_of_thought",
            extra={
                "component": "reasoning",
                "operation": "strategy_selection",
                "session_id": session_id,
                "metadata": {
                    "original_strategy": "tree_of_thoughts",
                    "fallback_strategy": "chain_of_thought",
                    "estimated_tokens": estimated_tokens,
                    "available_tokens": budget.status.remaining,
                }
            }
        )
        return await execute_chain_of_thought(task, session_id)

    # Proceed with budget reservation
    with budget.reserve(estimated_tokens, "reasoning"):
        # Execute operation
        result = await _do_tree_of_thoughts(task, branches)

        # Consume actual tokens
        budget.consume_actual(result.tokens_used.total_tokens)

        return result


def estimate_tot_tokens(task: str, branches: int) -> int:
    """Estimate tokens for tree-of-thoughts operation."""
    # Base cost per branch
    base_per_branch = 3000

    # Evaluation cost
    evaluation_cost = 2000

    # Task complexity factor
    task_factor = min(2.0, 1.0 + len(task) / 1000)

    return int((base_per_branch * branches + evaluation_cost) * task_factor)
```

---

## Component-Specific Examples

### Router Component

```python
# sigil/routing/router.py

from sigil.telemetry.logging import get_component_logger, operation_context
from sigil.telemetry.tokens import TokenUsage
from sigil.logging_contracts import ComponentId

logger = get_component_logger(ComponentId.ROUTER)


class Router:
    """Routes requests to appropriate handlers."""

    async def route(
        self,
        request: str,
        context: dict,
        session_id: str,
        correlation_id: str,
    ) -> RouteDecision:
        """Route a request to the appropriate handler."""

        async with operation_context(
            logger=logger,
            component=ComponentId.ROUTER,
            operation="route_request",
            session_id=session_id,
            correlation_id=correlation_id,
        ) as ctx:
            # Step 1: Classify intent
            intent, intent_tokens = await self._classify_intent(request)
            ctx.add_tokens(intent_tokens)

            # Step 2: Assess complexity
            complexity, complexity_tokens = await self._assess_complexity(
                request, context
            )
            ctx.add_tokens(complexity_tokens)

            # Step 3: Select handler
            handler = self._select_handler(intent, complexity)

            # Set metadata
            ctx.set_metadata({
                "intent": intent,
                "complexity": complexity,
                "handler": handler,
                "stratum": self._get_stratum(handler),
            })

            return RouteDecision(
                handler=handler,
                intent=intent,
                complexity=complexity,
            )

    async def _classify_intent(self, request: str) -> tuple[str, TokenUsage]:
        """Classify the intent of a request."""

        prompt = f"""Classify the intent of this request:

Request: {request}

Intents: create_agent, execute_task, simple_query, modify_agent, other

Return only the intent name."""

        response = await self.llm.generate(
            messages=[{"role": "user", "content": prompt}],
        )

        usage = TokenUsage.from_anthropic_response(
            response.raw_response,
            model=self.model,
        )

        logger.debug(
            f"Intent classified: {response.content.strip()}",
            extra={
                "component": "router",
                "operation": "classify_intent",
                "tokens_used": usage,
                "metadata": {"intent": response.content.strip()},
            }
        )

        return response.content.strip(), usage

    async def _assess_complexity(
        self, request: str, context: dict
    ) -> tuple[float, TokenUsage]:
        """Assess the complexity of a request."""

        prompt = f"""Rate the complexity of this request from 0.0 to 1.0:

Request: {request}
Context keys: {list(context.keys())}

Consider:
- Number of steps required
- Ambiguity in requirements
- Need for real-world interaction
- Criticality of outcome

Return only a number."""

        response = await self.llm.generate(
            messages=[{"role": "user", "content": prompt}],
        )

        usage = TokenUsage.from_anthropic_response(
            response.raw_response,
            model=self.model,
        )

        complexity = float(response.content.strip())

        logger.debug(
            f"Complexity assessed: {complexity}",
            extra={
                "component": "router",
                "operation": "assess_complexity",
                "tokens_used": usage,
                "metadata": {"complexity": complexity},
            }
        )

        return complexity, usage
```

### Memory Component

```python
# sigil/memory/retrieval.py

from sigil.telemetry.logging import get_component_logger, operation_context
from sigil.telemetry.tokens import TokenUsage
from sigil.logging_contracts import ComponentId

logger = get_component_logger(ComponentId.MEMORY)


class MemoryRetriever:
    """Retrieves memories using RAG, LLM, or hybrid methods."""

    async def retrieve(
        self,
        query: str,
        method: str = "hybrid",
        k: int = 10,
        session_id: str = "",
        correlation_id: str = "",
    ) -> list[MemoryItem]:
        """Retrieve relevant memories."""

        async with operation_context(
            logger=logger,
            component=ComponentId.MEMORY,
            operation=f"{method}_retrieval",
            session_id=session_id,
            correlation_id=correlation_id,
        ) as ctx:
            if method == "rag":
                results = await self._rag_retrieve(query, k, ctx)
            elif method == "llm":
                results = await self._llm_retrieve(query, k, ctx)
            else:  # hybrid
                results = await self._hybrid_retrieve(query, k, ctx)

            ctx.set_metadata({
                "layer": "items",
                "operation_type": "search",
                "retrieval_method": method,
                "items_count": len(results),
                "k_requested": k,
            })

            return results

    async def _rag_retrieve(
        self,
        query: str,
        k: int,
        ctx: OperationContext,
    ) -> list[MemoryItem]:
        """Retrieve using embedding similarity search."""

        # Embed the query
        embedding, embed_tokens = await self._embed_query(query)
        ctx.add_tokens(embed_tokens)

        # Search vector store
        results = self.vector_store.search(embedding, k=k)

        logger.debug(
            f"RAG search found {len(results)} items",
            extra={
                "component": "memory",
                "operation": "rag_search",
                "metadata": {
                    "k": k,
                    "results": len(results),
                    "top_score": results[0].score if results else 0,
                }
            }
        )

        return [self._to_memory_item(r) for r in results]

    async def _llm_retrieve(
        self,
        query: str,
        k: int,
        ctx: OperationContext,
    ) -> list[MemoryItem]:
        """Retrieve using LLM-based reading of memory categories."""

        # Load memory categories
        categories = await self._load_categories()

        # Build prompt for LLM to select relevant memories
        prompt = self._build_llm_retrieval_prompt(query, categories, k)

        response = await self.llm.generate(
            messages=[{"role": "user", "content": prompt}],
        )

        usage = TokenUsage.from_anthropic_response(
            response.raw_response,
            model=self.model,
        )
        ctx.add_tokens(usage)

        results = self._parse_llm_retrieval_response(response.content)

        logger.debug(
            f"LLM retrieval found {len(results)} items",
            extra={
                "component": "memory",
                "operation": "llm_retrieval",
                "tokens_used": usage,
                "metadata": {
                    "categories_searched": len(categories),
                    "results": len(results),
                }
            }
        )

        return results

    async def _hybrid_retrieve(
        self,
        query: str,
        k: int,
        ctx: OperationContext,
    ) -> list[MemoryItem]:
        """Hybrid retrieval: RAG first, LLM if needed."""

        # Start with RAG
        rag_results = await self._rag_retrieve(query, k, ctx)

        # Check if results are sufficient
        if len(rag_results) >= k and self._results_quality_ok(rag_results):
            logger.debug(
                "Hybrid: RAG results sufficient, skipping LLM",
                extra={
                    "component": "memory",
                    "operation": "hybrid_retrieval",
                    "metadata": {
                        "rag_results": len(rag_results),
                        "llm_needed": False,
                    }
                }
            )
            return rag_results[:k]

        # RAG insufficient, use LLM
        logger.debug(
            "Hybrid: RAG results insufficient, using LLM",
            extra={
                "component": "memory",
                "operation": "hybrid_retrieval",
                "metadata": {
                    "rag_results": len(rag_results),
                    "llm_needed": True,
                }
            }
        )

        llm_results = await self._llm_retrieve(query, k, ctx)

        # Merge and deduplicate
        merged = self._merge_results(rag_results, llm_results, k)

        return merged
```

### Planning Component

```python
# sigil/planning/planner.py

from sigil.telemetry.logging import get_component_logger, operation_context
from sigil.telemetry.tokens import TokenUsage
from sigil.logging_contracts import ComponentId

logger = get_component_logger(ComponentId.PLANNER)


class Planner:
    """Generates execution plans from goals."""

    async def create_plan(
        self,
        goal: str,
        context: dict,
        available_tools: list[str],
        session_id: str,
        correlation_id: str,
    ) -> Plan:
        """Create an execution plan for a goal."""

        async with operation_context(
            logger=logger,
            component=ComponentId.PLANNER,
            operation="create_plan",
            session_id=session_id,
            correlation_id=correlation_id,
        ) as ctx:
            # Generate plan using LLM
            prompt = self._build_planning_prompt(goal, context, available_tools)

            response = await self.llm.generate(
                messages=[{"role": "user", "content": prompt}],
                output_schema=PlanSchema,
            )

            usage = TokenUsage.from_anthropic_response(
                response.raw_response,
                model=self.model,
            )
            ctx.set_tokens(usage)

            # Parse and validate plan
            plan = self._parse_plan(response.content)

            # Estimate token costs
            estimated_tokens = self._estimate_plan_tokens(plan)

            # Validate dependencies
            self._validate_dependencies(plan)

            ctx.set_metadata({
                "goal_hash": hashlib.sha256(goal.encode()).hexdigest()[:16],
                "plan_id": plan.id,
                "step_count": len(plan.steps),
                "complexity_score": plan.complexity,
                "estimated_tokens": estimated_tokens,
                "available_tools": available_tools,
            })

            logger.info(
                f"Plan created with {len(plan.steps)} steps",
                extra={
                    "component": "planner",
                    "operation": "create_plan",
                    "session_id": session_id,
                    "tokens_used": usage,
                    "metadata": {
                        "plan_id": plan.id,
                        "steps": [s.name for s in plan.steps],
                    }
                }
            )

            return plan

    def _estimate_plan_tokens(self, plan: Plan) -> int:
        """Estimate token cost for executing the plan."""
        total = 0

        for step in plan.steps:
            # Base cost by complexity
            complexity_costs = {
                "simple": 1000,
                "moderate": 3000,
                "complex": 8000,
                "critical": 15000,
            }
            total += complexity_costs.get(step.complexity, 3000)

            # Add tool costs
            if step.requires_tools:
                total += 500 * len(step.required_tools)

            # Add contract validation cost
            if step.contract:
                total += 1000

        return total
```

### Reasoning Component

```python
# sigil/reasoning/manager.py

from sigil.telemetry.logging import get_component_logger, operation_context
from sigil.telemetry.tokens import TokenUsage
from sigil.logging_contracts import ComponentId

logger = get_component_logger(ComponentId.REASONING)


class ReasoningManager:
    """Manages reasoning strategy selection and execution."""

    async def reason(
        self,
        task: str,
        context: dict,
        session_id: str,
        correlation_id: str,
        force_strategy: Optional[str] = None,
    ) -> ReasoningResult:
        """Execute reasoning on a task."""

        async with operation_context(
            logger=logger,
            component=ComponentId.REASONING,
            operation="execute_reasoning",
            session_id=session_id,
            correlation_id=correlation_id,
        ) as ctx:
            # Select strategy
            strategy_name, selection_tokens = await self._select_strategy(
                task, context, force_strategy
            )
            ctx.add_tokens(selection_tokens)

            logger.info(
                f"Selected reasoning strategy: {strategy_name}",
                extra={
                    "component": "reasoning",
                    "operation": "strategy_selection",
                    "session_id": session_id,
                    "tokens_used": selection_tokens,
                    "metadata": {
                        "strategy": strategy_name,
                        "was_forced": force_strategy is not None,
                    }
                }
            )

            # Get strategy implementation
            strategy = self._get_strategy(strategy_name)

            # Execute with fallback handling
            try:
                result = await strategy.execute(task, context)
                ctx.add_tokens(result.tokens_used)

                ctx.set_metadata({
                    "strategy": strategy_name,
                    "task_hash": hashlib.sha256(task.encode()).hexdigest()[:16],
                    "confidence": result.confidence,
                    "reasoning_steps": len(result.trace),
                })

                return result

            except StrategyError as e:
                # Log strategy failure
                logger.warning(
                    f"Strategy {strategy_name} failed, attempting fallback",
                    extra={
                        "component": "reasoning",
                        "operation": f"{strategy_name}_failed",
                        "session_id": session_id,
                        "error_info": ErrorInfo.from_exception(e, recoverable=True),
                        "metadata": {
                            "original_strategy": strategy_name,
                            "error_type": type(e).__name__,
                        }
                    }
                )

                # Attempt fallback
                return await self._fallback_reasoning(
                    task, context, strategy_name, session_id, correlation_id, ctx
                )

    async def _select_strategy(
        self,
        task: str,
        context: dict,
        force_strategy: Optional[str],
    ) -> tuple[str, TokenUsage]:
        """Select the appropriate reasoning strategy."""

        if force_strategy:
            return force_strategy, TokenUsage.zero()

        # Assess complexity
        prompt = f"""Rate the complexity of this task from 0.0 to 1.0:

Task: {task}
Context keys: {list(context.keys())}

Return only a number."""

        response = await self.llm.generate(
            messages=[{"role": "user", "content": prompt}],
        )

        usage = TokenUsage.from_anthropic_response(
            response.raw_response,
            model=self.model,
        )

        complexity = float(response.content.strip())

        # Map complexity to strategy
        if complexity < 0.3:
            strategy = "direct"
        elif complexity < 0.5:
            strategy = "chain_of_thought"
        elif complexity < 0.7:
            strategy = "tree_of_thoughts"
        elif complexity < 0.9:
            strategy = "react"
        else:
            strategy = "mcts"

        # Check budget constraints
        budget = get_budget_tracker()
        estimated_cost = self._estimate_strategy_cost(strategy)

        if not budget.can_afford(estimated_cost):
            # Downgrade to cheaper strategy
            strategy = self._downgrade_strategy(strategy, budget.status.remaining)

            logger.warning(
                f"Budget constraint: downgraded strategy to {strategy}",
                extra={
                    "component": "reasoning",
                    "operation": "strategy_selection",
                    "metadata": {
                        "original_strategy": strategy,
                        "downgraded_to": strategy,
                        "budget_remaining": budget.status.remaining,
                    }
                }
            )

        return strategy, usage

    async def _fallback_reasoning(
        self,
        task: str,
        context: dict,
        failed_strategy: str,
        session_id: str,
        correlation_id: str,
        ctx: OperationContext,
    ) -> ReasoningResult:
        """Execute fallback reasoning after strategy failure."""

        # Define fallback chain
        fallback_chain = {
            "mcts": "react",
            "react": "tree_of_thoughts",
            "tree_of_thoughts": "chain_of_thought",
            "chain_of_thought": "direct",
        }

        fallback_strategy = fallback_chain.get(failed_strategy)

        if not fallback_strategy:
            raise ReasoningError(f"No fallback available for {failed_strategy}")

        logger.info(
            f"Falling back from {failed_strategy} to {fallback_strategy}",
            extra={
                "component": "reasoning",
                "operation": "strategy_fallback",
                "session_id": session_id,
                "metadata": {
                    "original_strategy": failed_strategy,
                    "fallback_strategy": fallback_strategy,
                }
            }
        )

        strategy = self._get_strategy(fallback_strategy)
        result = await strategy.execute(task, context)
        ctx.add_tokens(result.tokens_used)

        ctx.set_metadata({
            "strategy": fallback_strategy,
            "was_fallback": True,
            "original_strategy": failed_strategy,
            "confidence": result.confidence,
        })

        return result
```

### Contracts Component

```python
# sigil/contracts/executor.py

from sigil.telemetry.logging import get_component_logger, operation_context
from sigil.telemetry.tokens import TokenUsage
from sigil.logging_contracts import ComponentId, ErrorInfo

logger = get_component_logger(ComponentId.CONTRACTS)


class ContractExecutor:
    """Executes agents with contract enforcement."""

    async def execute(
        self,
        agent: BaseAgent,
        task: str,
        contract: Contract,
        context: dict,
        session_id: str,
        correlation_id: str,
    ) -> ContractResult:
        """Execute agent with contract verification."""

        async with operation_context(
            logger=logger,
            component=ComponentId.CONTRACTS,
            operation="execute_with_contract",
            session_id=session_id,
            correlation_id=correlation_id,
        ) as ctx:
            attempt = 0
            last_error = None

            while attempt <= contract.max_retries:
                attempt += 1

                logger.info(
                    f"Contract execution attempt {attempt}/{contract.max_retries + 1}",
                    extra={
                        "component": "contracts",
                        "operation": "execute_attempt",
                        "session_id": session_id,
                        "metadata": {
                            "contract_name": contract.name,
                            "attempt": attempt,
                        }
                    }
                )

                try:
                    # Execute agent
                    output = await agent.run(task, context)

                    if hasattr(output, 'tokens_used'):
                        ctx.add_tokens(output.tokens_used)

                    # Validate output
                    validation_result = await self._validate(
                        output, contract, session_id
                    )

                    if validation_result.success:
                        ctx.set_metadata({
                            "contract_name": contract.name,
                            "validation_passed": True,
                            "attempts": attempt,
                        })

                        return ContractResult(
                            success=True,
                            output=output,
                            attempts=attempt,
                        )

                    # Validation failed
                    logger.warning(
                        f"Contract validation failed: {validation_result.errors}",
                        extra={
                            "component": "contracts",
                            "operation": "validate_output",
                            "session_id": session_id,
                            "error_info": ErrorInfo(
                                error_type="ContractValidationError",
                                error_message=", ".join(validation_result.errors),
                                error_code="CONTRACT_002",
                                recoverable=attempt < contract.max_retries,
                            ),
                            "metadata": {
                                "contract_name": contract.name,
                                "attempt": attempt,
                                "deliverables_passed": validation_result.passed,
                                "deliverables_failed": validation_result.failed,
                            }
                        }
                    )

                    # Prepare retry context
                    if attempt <= contract.max_retries:
                        context = self._add_retry_context(
                            context, validation_result
                        )

                    last_error = validation_result.errors

                except Exception as e:
                    logger.error(
                        f"Contract execution error: {e}",
                        extra={
                            "component": "contracts",
                            "operation": "execute_attempt",
                            "session_id": session_id,
                            "error_info": ErrorInfo.from_exception(
                                e, recoverable=attempt < contract.max_retries
                            ),
                            "metadata": {
                                "contract_name": contract.name,
                                "attempt": attempt,
                            }
                        }
                    )

                    if attempt > contract.max_retries:
                        raise

                    last_error = str(e)

            # All attempts exhausted
            ctx.set_metadata({
                "contract_name": contract.name,
                "validation_passed": False,
                "attempts": attempt,
                "final_errors": last_error,
            })

            return ContractResult(
                success=False,
                errors=last_error,
                attempts=attempt,
            )

    async def _validate(
        self,
        output: dict,
        contract: Contract,
        session_id: str,
    ) -> ValidationResult:
        """Validate output against contract deliverables."""

        passed = []
        failed = []
        errors = []

        for deliverable in contract.deliverables:
            if deliverable.name not in output:
                failed.append(deliverable.name)
                errors.append(f"Missing: {deliverable.name}")
                continue

            value = output[deliverable.name]

            # Type check
            if not isinstance(value, deliverable.type):
                failed.append(deliverable.name)
                errors.append(
                    f"Type mismatch: {deliverable.name} expected "
                    f"{deliverable.type.__name__}, got {type(value).__name__}"
                )
                continue

            # Custom validation
            if deliverable.validation:
                try:
                    if not deliverable.validation(value):
                        failed.append(deliverable.name)
                        errors.append(f"Validation failed: {deliverable.name}")
                        continue
                except Exception as e:
                    failed.append(deliverable.name)
                    errors.append(f"Validation error: {deliverable.name} - {e}")
                    continue

            passed.append(deliverable.name)

        success = len(failed) == 0 and contract.success_criteria(output)

        return ValidationResult(
            success=success,
            passed=passed,
            failed=failed,
            errors=errors,
        )
```

---

## Performance Considerations

### 1. Buffer Configuration

Configure buffering based on your use case:

```python
# High-throughput logging (batch processing)
initialize_logging(
    buffer_size=500,      # Larger buffer
    flush_interval_ms=500, # Less frequent flushes
)

# Low-latency logging (real-time monitoring)
initialize_logging(
    buffer_size=10,       # Smaller buffer
    flush_interval_ms=50,  # More frequent flushes
)

# Balanced (default)
initialize_logging(
    buffer_size=100,
    flush_interval_ms=100,
)
```

### 2. Log Level Selection

Use appropriate log levels to control volume:

```python
# Production: INFO and above
logging.getLogger("sigil").setLevel(logging.INFO)

# Development: DEBUG and above
logging.getLogger("sigil").setLevel(logging.DEBUG)

# Troubleshooting: TRACE (custom level)
logging.getLogger("sigil").setLevel(5)  # TRACE
```

### 3. Metadata Size Limits

Keep metadata concise to avoid performance issues:

```python
# GOOD: Concise metadata
ctx.set_metadata({
    "strategy": "tree_of_thoughts",
    "confidence": 0.86,
    "branches": 4,
})

# BAD: Large metadata
ctx.set_metadata({
    "full_reasoning_trace": [...],  # Avoid large arrays
    "raw_response": "...",  # Avoid large strings
})
```

### 4. Async Operations

Use async operations for logging to avoid blocking:

```python
# The operation_context is already async-aware
async with operation_context(...) as ctx:
    # Non-blocking logging
    pass
```

### 5. Sampling for High-Volume Operations

For very high-volume operations, consider sampling:

```python
import random

class SampledLogger:
    """Logger that samples low-priority logs."""

    def __init__(self, logger: logging.Logger, sample_rate: float = 0.1):
        self.logger = logger
        self.sample_rate = sample_rate

    def debug(self, message: str, **kwargs):
        """Sample debug logs at configured rate."""
        if random.random() < self.sample_rate:
            self.logger.debug(message, **kwargs)

    def info(self, message: str, **kwargs):
        """Always log info messages."""
        self.logger.info(message, **kwargs)
```

---

## Testing Logging

### Unit Testing Logs

```python
import pytest
from unittest.mock import MagicMock, patch
from sigil.telemetry.logging import get_component_logger, operation_context
from sigil.logging_contracts import ComponentId, TokenUsage


@pytest.fixture
def mock_buffer():
    """Create a mock log buffer for testing."""
    with patch('sigil.telemetry.logging._buffer') as mock:
        mock.entries = []
        mock.add = lambda entry: mock.entries.append(entry)
        yield mock


@pytest.fixture
def mock_budget():
    """Create a mock budget tracker for testing."""
    with patch('sigil.telemetry.logging._budget_tracker') as mock:
        mock.status = BudgetStatus(total_budget=256000, used=0)
        mock.get_status = lambda: mock.status
        mock.consume = MagicMock()
        yield mock


class TestLogging:
    """Tests for the logging system."""

    async def test_operation_context_logs_start_and_completion(
        self, mock_buffer, mock_budget
    ):
        """Test that operation_context logs start and completion."""
        logger = get_component_logger(ComponentId.REASONING)

        async with operation_context(
            logger=logger,
            component=ComponentId.REASONING,
            operation="test_operation",
            session_id="test_session",
            correlation_id="test_correlation",
        ) as ctx:
            ctx.set_metadata({"test": "value"})

        # Should have 2 log entries: start and completion
        assert len(mock_buffer.entries) == 2

        start_entry = mock_buffer.entries[0]
        assert start_entry.operation == "test_operation"
        assert "Starting" in start_entry.message

        completion_entry = mock_buffer.entries[1]
        assert "Completed" in completion_entry.message
        assert completion_entry.metadata["test"] == "value"

    async def test_operation_context_logs_errors(
        self, mock_buffer, mock_budget
    ):
        """Test that operation_context logs errors correctly."""
        logger = get_component_logger(ComponentId.REASONING)

        with pytest.raises(ValueError):
            async with operation_context(
                logger=logger,
                component=ComponentId.REASONING,
                operation="failing_operation",
                session_id="test_session",
                correlation_id="test_correlation",
            ):
                raise ValueError("Test error")

        # Should have 2 entries: start and error
        assert len(mock_buffer.entries) == 2

        error_entry = mock_buffer.entries[1]
        assert "Failed" in error_entry.message
        assert error_entry.error_info is not None
        assert error_entry.error_info.error_type == "ValueError"

    async def test_token_tracking(self, mock_buffer, mock_budget):
        """Test that tokens are tracked correctly."""
        logger = get_component_logger(ComponentId.REASONING)

        async with operation_context(
            logger=logger,
            component=ComponentId.REASONING,
            operation="test_operation",
            session_id="test_session",
            correlation_id="test_correlation",
        ) as ctx:
            ctx.set_tokens(TokenUsage(
                input_tokens=1000,
                output_tokens=500,
                model="test-model",
            ))

        completion_entry = mock_buffer.entries[1]
        assert completion_entry.tokens_used.total_tokens == 1500

        # Budget tracker should be called
        mock_budget.consume.assert_called_once_with(1500, "reasoning")
```

### Integration Testing

```python
import pytest
import asyncio
from sigil.telemetry.logging import initialize_logging, get_component_logger
from sigil.telemetry.buffer import LogBuffer


class TestLoggingIntegration:
    """Integration tests for the logging system."""

    @pytest.fixture
    def log_buffer(self):
        """Create a real log buffer with test flush callback."""
        flushed_entries = []

        buffer = LogBuffer(buffer_size=10, flush_interval_ms=100)
        buffer._flush_callback = lambda entries: flushed_entries.extend(entries)

        return buffer, flushed_entries

    async def test_buffer_flush_on_capacity(self, log_buffer):
        """Test that buffer flushes when capacity is reached."""
        buffer, flushed = log_buffer

        # Add entries up to capacity
        for i in range(10):
            entry = ExecutionLogEntry(
                level=LogLevel.INFO,
                component=ComponentId.SYSTEM,
                operation=f"test_{i}",
                message=f"Test message {i}",
            )
            buffer.add(entry)

        # Should have flushed
        assert len(flushed) == 10

    async def test_buffer_flush_on_timeout(self, log_buffer):
        """Test that buffer flushes after timeout."""
        buffer, flushed = log_buffer

        entry = ExecutionLogEntry(
            level=LogLevel.INFO,
            component=ComponentId.SYSTEM,
            operation="test",
            message="Test message",
        )
        buffer.add(entry)

        # Wait for flush interval
        await asyncio.sleep(0.15)

        # Should have flushed
        assert len(flushed) == 1

    async def test_buffer_immediate_flush_on_error(self, log_buffer):
        """Test that errors are flushed immediately."""
        buffer, flushed = log_buffer

        # Add info entry
        info_entry = ExecutionLogEntry(
            level=LogLevel.INFO,
            component=ComponentId.SYSTEM,
            operation="info_test",
            message="Info message",
        )
        buffer.add(info_entry)

        # Add error entry
        error_entry = ExecutionLogEntry(
            level=LogLevel.ERROR,
            component=ComponentId.SYSTEM,
            operation="error_test",
            message="Error message",
        )
        buffer.add(error_entry)

        # Should have flushed both immediately
        assert len(flushed) == 2
```

---

## Troubleshooting

### Common Issues

#### 1. Missing Log Entries

**Symptom:** Log entries are not appearing.

**Causes and Solutions:**

```python
# Cause 1: Logger not initialized
# Solution: Initialize logging at startup
from sigil.telemetry.logging import initialize_logging
initialize_logging()

# Cause 2: Log level too high
# Solution: Lower the log level
import logging
logging.getLogger("sigil").setLevel(logging.DEBUG)

# Cause 3: Buffer not flushing
# Solution: Reduce flush interval or buffer size
initialize_logging(buffer_size=10, flush_interval_ms=50)
```

#### 2. Token Tracking Not Working

**Symptom:** TokenUsage is always zero.

**Causes and Solutions:**

```python
# Cause 1: Not extracting from response
# Solution: Extract tokens from LLM response
usage = TokenUsage.from_anthropic_response(
    response.raw_response,  # Need raw response dict
    model=model,
)

# Cause 2: Not calling ctx.set_tokens()
# Solution: Always set tokens in context
async with operation_context(...) as ctx:
    result = await llm.generate(...)
    ctx.set_tokens(extract_tokens(result))  # Don't forget this!
```

#### 3. Budget Not Updating

**Symptom:** Budget status shows 0 consumption.

**Causes and Solutions:**

```python
# Cause 1: Budget tracker not initialized
# Solution: Initialize logging (creates budget tracker)
initialize_logging(total_budget=256_000)

# Cause 2: Not consuming tokens
# Solution: Budget is updated automatically by operation_context
# when tokens are set
async with operation_context(...) as ctx:
    ctx.set_tokens(usage)  # This updates the budget
```

#### 4. Performance Degradation

**Symptom:** Logging is slowing down operations.

**Causes and Solutions:**

```python
# Cause 1: Buffer too small causing frequent flushes
# Solution: Increase buffer size
initialize_logging(buffer_size=500, flush_interval_ms=500)

# Cause 2: Too much metadata
# Solution: Limit metadata size
ctx.set_metadata({
    "key": "value",  # Keep it small
    # Don't include large arrays or strings
})

# Cause 3: Too many debug logs
# Solution: Raise log level in production
logging.getLogger("sigil").setLevel(logging.INFO)
```

### Debug Mode

Enable debug mode for detailed logging diagnostics:

```python
# Enable debug mode
import os
os.environ["SIGIL_LOG_DEBUG"] = "1"

# This will:
# - Log buffer operations
# - Log budget updates
# - Log flush events
# - Include timing information
```

---

## Reference

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `SIGIL_LOG_LEVEL` | `INFO` | Default log level |
| `SIGIL_LOG_BUFFER_SIZE` | `100` | Buffer capacity |
| `SIGIL_LOG_FLUSH_INTERVAL_MS` | `100` | Flush interval |
| `SIGIL_LOG_DEBUG` | `0` | Enable debug mode |
| `SIGIL_TOKEN_BUDGET` | `256000` | Default token budget |
| `SIGIL_BUDGET_WARNING_THRESHOLD` | `0.80` | Warning threshold |
| `SIGIL_BUDGET_CRITICAL_THRESHOLD` | `0.95` | Critical threshold |

### Module Index

| Module | Description |
|--------|-------------|
| `sigil.telemetry.logging` | Logger setup and configuration |
| `sigil.telemetry.tokens` | TokenUsage and TokenTracker |
| `sigil.telemetry.budget` | BudgetTracker and reservations |
| `sigil.telemetry.buffer` | LogBuffer implementation |
| `sigil.telemetry.context` | Operation context manager |
| `sigil.logging_contracts` | Schema definitions |

### Related Documentation

- [Logging Contracts](/Users/zidane/Bland-Agents-Dataset/acti-agent-builder/docs/logging-contracts.md)
- [Monitoring API](/Users/zidane/Bland-Agents-Dataset/acti-agent-builder/docs/monitoring-api.yaml)
- [CLI Execution Flow](/Users/zidane/Bland-Agents-Dataset/acti-agent-builder/docs/cli-execution-flow.md)
- [Sigil v2 Architecture](/Users/zidane/Bland-Agents-Dataset/Sigil_V2.md)

---

**Version History**

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2026-01-11 | Initial release |
