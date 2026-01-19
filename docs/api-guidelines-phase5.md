# API Guidelines: Phase 5 Planning & Reasoning

## Overview

This document provides implementation guidance for the Phase 5 Planning & Reasoning API. It supplements the OpenAPI specification with Python SDK contracts, naming conventions, security notes, and example implementations.

---

## Table of Contents

1. [Python SDK Contracts](#1-python-sdk-contracts)
2. [Naming Conventions](#2-naming-conventions)
3. [Required Headers](#3-required-headers)
4. [Error Handling](#4-error-handling)
5. [Rate Limiting](#5-rate-limiting)
6. [Authentication](#6-authentication)
7. [Example Implementations](#7-example-implementations)
8. [Integration Patterns](#8-integration-patterns)
9. [Testing Guidelines](#9-testing-guidelines)

---

## 1. Python SDK Contracts

### 1.1 Planner Class

```python
from typing import Optional, Dict, Any, List
from datetime import datetime
from dataclasses import dataclass, field

class Planner:
    """Decomposes high-level goals into executable plans.

    The Planner uses an LLM to break down complex goals into a DAG
    (Directed Acyclic Graph) of steps that can be executed with
    appropriate parallelization and dependency management.

    Attributes:
        model: LLM model for plan generation
        tool_registry: Available tools for step assignment
        memory_manager: Memory system for context retrieval
        cache: Plan cache for similar goal reuse
    """

    def __init__(
        self,
        model: str = "anthropic:claude-sonnet-4-20250514",
        tool_registry: "ToolRegistry | None" = None,
        memory_manager: "MemoryManager | None" = None,
        cache_ttl_hours: int = 24,
        max_steps: int = 20,
    ) -> None:
        """Initialize the Planner.

        Args:
            model: LLM model identifier for plan generation.
            tool_registry: Registry of available tools. Auto-created if None.
            memory_manager: Memory system for context. Optional.
            cache_ttl_hours: Plan cache time-to-live in hours.
            max_steps: Maximum steps allowed per plan.
        """
        ...

    async def create_plan(
        self,
        goal: str,
        context: Optional[Dict[str, Any]] = None,
        tools: Optional[List[str]] = None,
        max_steps: int = 10,
    ) -> "Plan":
        """Generate an execution plan for a goal.

        Args:
            goal: User's desired outcome (required, 10-500 chars).
            context: Optional context dict from memory, previous runs, etc.
            tools: Available tools agent can use (optional).
            max_steps: Maximum steps in plan (default 10, range 1-50).

        Returns:
            Plan with DAG structure of steps.

        Raises:
            InvalidGoalError: If goal is empty, too short, or too vague.
            PlanningError: If plan generation fails after retries.
            TokenBudgetError: If not enough tokens to plan.
            DependencyError: If circular dependencies detected (regenerates).

        Side Effects:
            - Emits PlanCreatedEvent to event store
            - Stores plan in persistent cache for resumption
            - Updates token budget tracker

        Example:
            >>> planner = Planner()
            >>> plan = await planner.create_plan(
            ...     goal="Qualify John from Acme Corp for our SaaS",
            ...     context={
            ...         "lead_name": "John",
            ...         "company": "Acme Corp",
            ...         "budget": "known",
            ...         "authority": "unknown"
            ...     },
            ...     tools=["crm_lookup", "websearch", "send_email"],
            ...     max_steps=8
            ... )
            >>> print(f"Generated {len(plan.steps)} steps")
        """
        ...

    def validate_plan(self, plan: "Plan") -> "ValidationResult":
        """Validate plan structure and dependencies.

        Checks:
        - DAG is acyclic (no circular dependencies)
        - All dependencies reference valid step_ids
        - Step types are valid
        - Estimated tokens within budget

        Args:
            plan: Plan to validate.

        Returns:
            ValidationResult with is_valid and any errors.
        """
        ...

    async def optimize_plan(
        self,
        plan: "Plan",
        optimization_goals: List[str] = ["minimize_tokens", "maximize_parallelism"],
    ) -> "Plan":
        """Optimize plan structure.

        Optimizations:
        - Merge sequential reasoning steps when possible
        - Identify additional parallelization opportunities
        - Remove redundant steps
        - Estimate more accurate token costs

        Args:
            plan: Plan to optimize.
            optimization_goals: List of optimization objectives.

        Returns:
            Optimized plan (new instance, original unchanged).
        """
        ...
```

### 1.2 PlanExecutor Class

```python
class PlanExecutor:
    """Executes plans with dependency management and concurrency control.

    The executor handles:
    - Topological sorting for correct execution order
    - Bounded parallelism for concurrent step execution
    - Retry logic with exponential backoff
    - Checkpointing for resumption after failure
    - Progress tracking and event emission

    Attributes:
        max_concurrent: Maximum parallel step executions
        max_retries: Maximum retries per step
        checkpoint_store: Storage for execution checkpoints
    """

    def __init__(
        self,
        reasoning_manager: "ReasoningManager | None" = None,
        tool_executor: "ToolExecutor | None" = None,
        memory_manager: "MemoryManager | None" = None,
        max_concurrent: int = 3,
        max_retries: int = 3,
        checkpoint_dir: str = "outputs/checkpoints",
    ) -> None:
        """Initialize the PlanExecutor.

        Args:
            reasoning_manager: For reasoning steps. Auto-created if None.
            tool_executor: For tool call steps. Auto-created if None.
            memory_manager: For context and storage. Optional.
            max_concurrent: Max parallel steps (default 3).
            max_retries: Retries per step (default 3).
            checkpoint_dir: Checkpoint storage directory.
        """
        ...

    async def execute(self, plan: "Plan") -> "PlanResult":
        """Execute a plan to completion.

        Args:
            plan: Previously created plan to execute.

        Returns:
            PlanResult with all step results and final output.

        Raises:
            PlanNotFoundError: If plan doesn't exist.
            StepExecutionError: If step fails after all retries (continues).
            TokenBudgetError: If budget exhausted (returns partial result).

        Execution Guarantees:
            - Steps execute in dependency order (topological sort)
            - Failed steps retry up to 3 times
            - Failed dependencies cause step to be skipped (not aborted)
            - Always returns best partial result
            - Full audit trail in event store

        Events Emitted:
            - PlanExecutionStartedEvent(plan_id)
            - PlanStepStartedEvent(plan_id, step_id) per step
            - PlanStepCompletedEvent(plan_id, step_id, result) per step
            - PlanStepFailedEvent(plan_id, step_id, error, retry_count) on failure
            - PlanCompletedEvent(plan_id, total_tokens)
        """
        ...

    async def pause(self, plan_id: str) -> bool:
        """Pause execution after current step.

        Args:
            plan_id: ID of plan to pause.

        Returns:
            True if paused, False if not running.

        Events Emitted:
            - PlanPausedEvent(plan_id, last_completed_step)
        """
        ...

    async def resume(self, plan_id: str) -> "PlanResult":
        """Resume from last completed step.

        Args:
            plan_id: ID of plan to resume.

        Returns:
            Full result from resumed execution.

        Events Emitted:
            - PlanResumedEvent(plan_id, resuming_from)
        """
        ...

    async def abort(self, plan_id: str) -> bool:
        """Cancel all remaining steps.

        Args:
            plan_id: ID of plan to abort.

        Returns:
            True if aborted, False if already complete.

        Events Emitted:
            - PlanAbortedEvent(plan_id)
        """
        ...

    async def get_status(self, plan_id: str) -> "PlanStatus":
        """Get execution status of a plan.

        Args:
            plan_id: ID of plan to check.

        Returns:
            PlanStatus with overall status, current step, and progress.
        """
        ...
```

### 1.3 ReasoningManager Class

```python
class ReasoningManager:
    """Orchestrates reasoning strategy selection and execution.

    The ReasoningManager selects the appropriate reasoning strategy based on
    task complexity and executes it with fallback handling. It integrates
    with memory for context and tracks token usage.

    Strategy Selection Logic:
        complexity 0.0 - 0.3 -> DirectStrategy
        complexity 0.3 - 0.5 -> CoTStrategy
        complexity 0.5 - 0.7 -> ToTStrategy
        complexity 0.7 - 0.9 -> ReActStrategy
        complexity 0.9 - 1.0 -> MCTSStrategy

    Fallback Behavior:
        If selected strategy fails, try next lower complexity strategy.
        Ultimate fallback is always DirectStrategy.
    """

    def __init__(
        self,
        model: str = "anthropic:claude-sonnet-4-20250514",
        memory_manager: "MemoryManager | None" = None,
        tool_registry: "ToolRegistry | None" = None,
        cache_enabled: bool = True,
    ) -> None:
        """Initialize the ReasoningManager.

        Args:
            model: Default LLM model.
            memory_manager: Memory for context retrieval. Optional.
            tool_registry: Tools for ReAct strategy. Optional.
            cache_enabled: Enable result caching.
        """
        ...

    async def execute(
        self,
        task: str,
        context: Optional[Dict[str, Any]] = None,
        complexity: Optional[float] = None,
        strategy: Optional[str] = None,
    ) -> "StrategyResult":
        """Execute reasoning on a task.

        Args:
            task: Task to reason about (required).
            context: Background info from memory, etc. (optional).
            complexity: Override automatic complexity (0.0-1.0, optional).
            strategy: Force specific strategy (optional, overrides auto-selection).

        Returns:
            StrategyResult from selected strategy.
            If strategy fails, automatically tries fallback chain.
            Always returns result (DirectStrategy is ultimate fallback).

        Events Emitted:
            - ReasoningTaskStartedEvent(task_hash, complexity)
            - StrategySelectedEvent(strategy_name, complexity)
            - ReasoningCompletedEvent(strategy_name, confidence, tokens)
            - StrategyFallbackEvent(original, fallback, reason) on fallback
        """
        ...

    def select_strategy(self, complexity: float) -> str:
        """Select strategy based on complexity.

        Args:
            complexity: Task complexity score (0.0 - 1.0).

        Returns:
            Strategy name to use.

        Selection Logic:
            0.0 - 0.3: direct
            0.3 - 0.5: cot
            0.5 - 0.7: tot
            0.7 - 0.9: react
            0.9 - 1.0: mcts
        """
        ...

    def get_metrics(self) -> "StrategyMetrics":
        """Get strategy execution metrics.

        Returns:
            StrategyMetrics with per-strategy and overall statistics.
        """
        ...

    def clear_metrics(self) -> None:
        """Reset collected metrics to zero."""
        ...
```

### 1.4 BaseStrategy Abstract Class

```python
from abc import ABC, abstractmethod

class BaseStrategy(ABC):
    """Abstract base class for reasoning strategies.

    All strategies must implement the execute() method.
    Strategies are stateless - all context passed via parameters.

    Attributes:
        name: Strategy identifier (e.g., 'direct', 'chain_of_thought')
        complexity_range: (min, max) complexity this strategy is optimized for
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Strategy name."""
        ...

    @property
    @abstractmethod
    def complexity_range(self) -> tuple[float, float]:
        """(min, max) complexity range."""
        ...

    @abstractmethod
    async def execute(
        self,
        task: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> "StrategyResult":
        """Execute strategy on task and return result.

        Args:
            task: Task/question to reason about.
            context: Optional context (facts, history).

        Returns:
            StrategyResult with answer, trace, and confidence.
        """
        ...
```

### 1.5 Strategy Implementations

#### DirectStrategy

```python
class DirectStrategy(BaseStrategy):
    """Simple single-call reasoning for straightforward tasks.

    Characteristics:
        - Complexity range: 0.0-0.3
        - Token budget: 100-300
        - Reasoning trace: Always exactly 1 step: ["Single LLM call"]
        - Confidence: Moderate (0.4-0.7)
        - Best for: Factual queries, simple lookups
    """

    name = "direct"
    complexity_range = (0.0, 0.3)

    async def execute(
        self,
        task: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> "StrategyResult":
        """Execute direct single-call reasoning."""
        ...
```

#### ChainOfThoughtStrategy

```python
class ChainOfThoughtStrategy(BaseStrategy):
    """Step-by-step reasoning with explicit thought process.

    Characteristics:
        - Complexity range: 0.3-0.5
        - Token budget: 300-800
        - Reasoning trace: 3-7 steps, explicit "Let me think..."
        - Confidence: Higher (0.6-0.8)
        - Best for: Step-by-step reasoning, explanations
    """

    name = "cot"
    complexity_range = (0.3, 0.5)
```

#### TreeOfThoughtsStrategy

```python
class TreeOfThoughtsStrategy(BaseStrategy):
    """Explore multiple reasoning paths and select best.

    Characteristics:
        - Complexity range: 0.5-0.7
        - Token budget: 800-2000
        - Reasoning trace: Shows 3-5 explored paths with selection
        - Confidence: High (0.7-0.9)
        - Best for: Ambiguous problems, multiple valid answers
    """

    name = "tot"
    complexity_range = (0.5, 0.7)
    num_paths: int = 3
    max_depth: int = 3
```

#### ReActStrategy

```python
class ReActStrategy(BaseStrategy):
    """Interleaved reasoning and tool use.

    Characteristics:
        - Complexity range: 0.7-0.9
        - Token budget: 1000-3000
        - Reasoning trace: Interleaved Thought-Action-Observation
        - Confidence: Variable (0.5-0.9)
        - Best for: Tool-heavy tasks, real-world interaction
    """

    name = "react"
    complexity_range = (0.7, 0.9)
    max_iterations: int = 5
```

#### MCTSStrategy

```python
class MCTSStrategy(BaseStrategy):
    """Monte Carlo Tree Search for critical decisions.

    Characteristics:
        - Complexity range: 0.9-1.0
        - Token budget: 2000-5000
        - Reasoning trace: Tree of simulated outcomes with scores
        - Confidence: Highest (0.8-0.95)
        - Best for: Critical decisions, high-stakes tasks
    """

    name = "mcts"
    complexity_range = (0.9, 1.0)
    num_simulations: int = 10
    exploration_constant: float = 1.414  # sqrt(2)
    max_depth: int = 5
```

---

## 2. Naming Conventions

### 2.1 Identifiers

| Entity | Pattern | Example |
|--------|---------|---------|
| Plan ID | `plan_{uuid}` | `plan_550e8400-e29b-41d4-a716-446655440000` |
| Step ID | `step_{index}` | `step_1`, `step_2`, `step_3` |
| Event ID | `evt_{uuid}` | `evt_123e4567-e89b-12d3-a456-426614174000` |
| Checkpoint ID | `chk_{plan_id}_{timestamp}` | `chk_plan_550e8400_1704974400` |

### 2.2 Field Names

| Style | Usage | Examples |
|-------|-------|----------|
| snake_case | All field names | `plan_id`, `step_count`, `tokens_used` |
| PascalCase | Class names | `PlanExecutor`, `StrategyResult` |
| lowercase | Event types | `plan.created`, `reasoning.completed` |
| SCREAMING_SNAKE | Constants | `MAX_STEPS`, `DEFAULT_MODEL` |

### 2.3 Resource Names

| Resource | Singular | Plural | URI Pattern |
|----------|----------|--------|-------------|
| Plan | `plan` | `plans` | `/plans/{plan_id}` |
| Step | `step` | `steps` | `/plans/{plan_id}/steps/{step_id}` |
| Strategy | `strategy` | `strategies` | `/strategies/{strategy_name}` |

---

## 3. Required Headers

### 3.1 Request Headers

| Header | Required | Description | Example |
|--------|----------|-------------|---------|
| `Content-Type` | Yes | Media type | `application/json` |
| `Accept` | Recommended | Accepted response types | `application/json` |
| `Authorization` | Conditional | Bearer token (when exposed via HTTP) | `Bearer eyJhbG...` |
| `X-Request-ID` | Recommended | Client-generated request ID | `req_abc123` |
| `X-Session-ID` | Conditional | Session ID for tracking | `sess_xyz789` |

### 3.2 Response Headers

| Header | Description | Example |
|--------|-------------|---------|
| `Content-Type` | Response media type | `application/json` |
| `X-Request-ID` | Echo of client request ID | `req_abc123` |
| `X-Correlation-ID` | Server-generated correlation ID | `corr_456def` |
| `X-RateLimit-Limit` | Rate limit ceiling | `100` |
| `X-RateLimit-Remaining` | Remaining requests | `95` |
| `X-RateLimit-Reset` | Reset timestamp (Unix) | `1704974400` |
| `X-Tokens-Used` | Tokens consumed by request | `450` |
| `X-Tokens-Remaining` | Remaining token budget | `49550` |

---

## 4. Error Handling

### 4.1 Error Taxonomy

| Code | Name | Trigger | Recovery |
|------|------|---------|----------|
| P01 | PlanningError | Plan generation fails | Retry once, ask user |
| P02 | InvalidGoalError | Goal unclear or too broad | Ask user to clarify |
| P03 | DependencyError | Circular dependencies | Regenerate without cycles |
| P04 | StepExecutionError | Step fails | Retry up to 3x |
| P05 | PlanNotFoundError | Plan ID doesn't exist | Verify plan ID |
| R01 | StrategyExecutionError | Strategy fails | Fallback chain |
| R02 | AllStrategiesFailedError | All strategies fail | Manual intervention |
| R03 | TimeoutError | Operation too long | Return best result |
| R04 | TokenBudgetError | Out of tokens | Degrade strategy |
| R05 | UnknownStrategyError | Strategy not registered | Use auto-selection |

### 4.2 Error Response Format (RFC 9457)

```json
{
  "type": "https://sigil.io/errors/invalid-goal",
  "title": "Invalid Goal Error",
  "status": 400,
  "detail": "Goal is too vague. Provide a more specific objective with measurable outcomes.",
  "instance": "/plans",
  "code": "P02",
  "recovery_suggestions": [
    "Include specific outcome you want to achieve",
    "Add context about the target (lead name, company, etc.)",
    "Break down into smaller, specific goals"
  ]
}
```

### 4.3 Python Exception Classes

```python
class SigilError(Exception):
    """Base exception for all Sigil errors."""
    code: str = "SIG-000"
    recovery_suggestions: List[str] = []

class PlanningError(SigilError):
    """Generic planning failure."""
    code = "P01"

class InvalidGoalError(PlanningError):
    """Goal unclear or too broad."""
    code = "P02"

class DependencyError(PlanningError):
    """Circular dependencies in generated plan."""
    code = "P03"

class StepExecutionError(PlanningError):
    """Step failed after all retries."""
    code = "P04"

class PlanNotFoundError(PlanningError):
    """Plan ID doesn't exist."""
    code = "P05"

class ReasoningError(SigilError):
    """Base exception for reasoning errors."""
    code = "R00"

class StrategyExecutionError(ReasoningError):
    """Strategy failed to execute."""
    code = "R01"

class AllStrategiesFailedError(ReasoningError):
    """All reasoning strategies have failed."""
    code = "R02"

class TimeoutError(SigilError):
    """Operation exceeded timeout."""
    code = "R03"

class TokenBudgetError(SigilError):
    """Token budget exhausted."""
    code = "R04"

class UnknownStrategyError(ReasoningError):
    """Strategy name not registered."""
    code = "R05"
```

---

## 5. Rate Limiting

### 5.1 Default Limits

| Resource | Limit | Window | Scope |
|----------|-------|--------|-------|
| Plan creation | 10 | 1 minute | Per session |
| Plan execution | 5 concurrent | - | Per session |
| Reasoning tasks | 30 | 1 minute | Per session |
| Strategy fallbacks | 3 | Per task | Per task |

### 5.2 Rate Limit Headers

```http
HTTP/1.1 429 Too Many Requests
Content-Type: application/json
X-RateLimit-Limit: 10
X-RateLimit-Remaining: 0
X-RateLimit-Reset: 1704974460
Retry-After: 60

{
  "type": "https://sigil.io/errors/rate-limit-exceeded",
  "title": "Rate Limit Exceeded",
  "status": 429,
  "detail": "Plan creation rate limit exceeded. Maximum 10 plans per minute.",
  "retry_after": 60
}
```

---

## 6. Authentication

### 6.1 Internal SDK Usage

No authentication required for direct Python SDK usage within the Sigil framework.

### 6.2 HTTP API (When Exposed via FastAPI)

Bearer token authentication using JWT tokens from Phase 3 authentication:

```http
POST /api/v1/plans HTTP/1.1
Host: localhost:8000
Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
Content-Type: application/json

{
  "goal": "Qualify lead John from Acme Corp"
}
```

### 6.3 Security Considerations

1. **Goal Content**: Goals may contain sensitive information. Do not log full goal text in production.
2. **Context Data**: Context dictionaries may include PII. Apply data minimization.
3. **Reasoning Traces**: Traces may contain intermediate reasoning about sensitive topics. Consider access controls.
4. **Event Hashing**: Use SHA256 hashes for task content in events to preserve privacy while enabling correlation.

---

## 7. Example Implementations

### 7.1 Create and Execute a Plan

```python
from sigil.planning import Planner, PlanExecutor

async def qualify_lead(lead_name: str, company: str) -> str:
    """Complete lead qualification workflow."""

    # Initialize components
    planner = Planner()
    executor = PlanExecutor()

    # Create plan
    plan = await planner.create_plan(
        goal=f"Qualify lead {lead_name} from {company} for our SaaS",
        context={
            "lead_name": lead_name,
            "company": company,
            "budget": "unknown",
            "authority": "unknown",
            "need": "unknown",
            "timeline": "unknown",
        },
        tools=["crm_lookup", "websearch", "send_email"],
        max_steps=8,
    )

    print(f"Generated plan with {len(plan.steps)} steps")
    print(f"Estimated tokens: {plan.estimated_tokens}")

    # Execute plan
    result = await executor.execute(plan)

    if result.status == "completed":
        print(f"Plan completed successfully!")
        print(f"Total tokens: {result.total_tokens}")
        return result.final_output
    else:
        print(f"Plan finished with status: {result.status}")
        print(f"Errors: {result.errors}")
        return result.final_output or "Partial result"

# Usage
output = await qualify_lead("John", "Acme Corp")
print(output)
```

### 7.2 Reasoning with Strategy Selection

```python
from sigil.reasoning import ReasoningManager

async def analyze_pricing(company_size: int, budget: str) -> dict:
    """Recommend pricing plan with reasoning."""

    manager = ReasoningManager()

    task = f"""
    Recommend the best pricing plan for:
    - Company size: {company_size} employees
    - Budget situation: {budget}

    Available plans:
    - Basic: $50/month (up to 10 users)
    - Professional: $200/month (up to 100 users)
    - Enterprise: Custom pricing (unlimited users)
    """

    # Let manager auto-select strategy based on complexity
    result = await manager.execute(
        task=task,
        context={
            "facts": [
                "Basic plan includes core features only",
                "Professional plan includes advanced analytics",
                "Enterprise includes dedicated support",
            ]
        },
    )

    return {
        "recommendation": result.answer,
        "confidence": result.confidence,
        "reasoning": result.reasoning_trace,
        "strategy_used": result.metadata.get("strategy_name"),
        "tokens_used": result.tokens_used,
    }

# Usage
analysis = await analyze_pricing(50, "moderate")
print(f"Recommendation: {analysis['recommendation']}")
print(f"Confidence: {analysis['confidence']:.2f}")
print(f"Strategy: {analysis['strategy_used']}")
```

### 7.3 Forced Strategy Execution

```python
from sigil.reasoning import ReasoningManager

async def critical_decision(task: str) -> dict:
    """Execute high-stakes decision with MCTS."""

    manager = ReasoningManager()

    # Force MCTS for critical decisions
    result = await manager.execute(
        task=task,
        strategy="mcts",  # Override auto-selection
    )

    return {
        "decision": result.answer,
        "confidence": result.confidence,
        "simulations": result.metadata.get("simulations_run"),
        "paths_explored": result.metadata.get("paths_explored"),
    }
```

### 7.4 Plan with Pause and Resume

```python
from sigil.planning import Planner, PlanExecutor
import asyncio

async def long_running_plan():
    """Execute a plan with pause/resume capability."""

    planner = Planner()
    executor = PlanExecutor()

    plan = await planner.create_plan(
        goal="Comprehensive market analysis for enterprise CRM",
        max_steps=15,
    )

    # Start execution in background
    execution_task = asyncio.create_task(
        executor.execute(plan)
    )

    # Wait a bit, then pause
    await asyncio.sleep(5)
    paused = await executor.pause(plan.plan_id)

    if paused:
        print("Execution paused. Can resume later.")

        # Later... resume execution
        result = await executor.resume(plan.plan_id)
        return result

    # If not paused (already completed), get result
    return await execution_task
```

---

## 8. Integration Patterns

### 8.1 Router Integration (Phase 3)

```python
from sigil.routing import Router, RouteDecision
from sigil.planning import Planner, PlanExecutor
from sigil.reasoning import ReasoningManager

async def handle_request(message: str, session_id: str):
    """Route request to planning or direct reasoning."""

    router = Router()
    decision: RouteDecision = router.route(message)

    if decision.use_planning:
        # Complex task - use planning
        planner = Planner()
        executor = PlanExecutor()

        plan = await planner.create_plan(
            goal=message,
            session_id=session_id,
        )
        result = await executor.execute(plan)
        return result.final_output

    else:
        # Simple task - direct reasoning
        manager = ReasoningManager()
        result = await manager.execute(
            task=message,
            complexity=decision.complexity,
        )
        return result.answer
```

### 8.2 Memory Integration (Phase 4)

```python
from sigil.memory import MemoryManager
from sigil.planning import Planner
from sigil.reasoning import ReasoningManager

async def reason_with_memory(task: str):
    """Retrieve memory context before reasoning."""

    memory = MemoryManager()
    manager = ReasoningManager(memory_manager=memory)

    # Memory is automatically retrieved and injected
    result = await manager.execute(
        task=task,
        context={
            # Memory facts are added automatically when memory_manager is set
            "additional_context": "User prefers detailed explanations",
        },
    )

    return result
```

### 8.3 Contract Integration (Phase 6)

```python
from sigil.planning import Planner, PlanExecutor
from sigil.contracts import ContractExecutor, ContractSpec

async def verified_plan_execution():
    """Execute plan with contract verification on critical steps."""

    planner = Planner()
    executor = PlanExecutor()
    contracts = ContractExecutor()

    plan = await planner.create_plan(
        goal="Generate qualified lead report",
    )

    # Add contracts to critical steps
    for step in plan.steps:
        if "score" in step.description.lower():
            step.contract_spec = ContractSpec(
                required_fields=["score"],
                field_types={"score": int},
                validation_rules=["score >= 0 and score <= 100"],
            )

    # Execute with contract checking
    result = await executor.execute(
        plan=plan,
        contract_executor=contracts,
    )

    return result
```

---

## 9. Testing Guidelines

### 9.1 Unit Test Examples

```python
import pytest
from sigil.planning import Planner
from sigil.reasoning import ReasoningManager

@pytest.mark.asyncio
async def test_create_plan_success():
    """Test successful plan creation."""
    planner = Planner()

    plan = await planner.create_plan(
        goal="Simple test goal with enough detail",
        max_steps=5,
    )

    assert plan.plan_id.startswith("plan_")
    assert len(plan.steps) > 0
    assert len(plan.steps) <= 5
    assert plan.complexity >= 0.0
    assert plan.complexity <= 1.0

@pytest.mark.asyncio
async def test_create_plan_invalid_goal():
    """Test plan creation with invalid goal."""
    planner = Planner()

    with pytest.raises(InvalidGoalError):
        await planner.create_plan(goal="hi")  # Too short

@pytest.mark.asyncio
async def test_strategy_selection():
    """Test strategy selection by complexity."""
    manager = ReasoningManager()

    assert manager.select_strategy(0.1) == "direct"
    assert manager.select_strategy(0.4) == "cot"
    assert manager.select_strategy(0.6) == "tot"
    assert manager.select_strategy(0.8) == "react"
    assert manager.select_strategy(0.95) == "mcts"

@pytest.mark.asyncio
async def test_fallback_on_failure():
    """Test strategy fallback when primary fails."""
    manager = ReasoningManager()

    # Force MCTS and inject failure
    result = await manager.execute(
        task="Test task that should fallback",
        strategy="mcts",
    )

    # Should succeed with fallback
    assert result.answer is not None
    if result.metadata.get("fallback_used"):
        assert result.metadata.get("original_strategy") == "mcts"
```

### 9.2 Integration Test Examples

```python
@pytest.mark.asyncio
async def test_full_planning_workflow():
    """Test complete plan creation and execution."""
    planner = Planner()
    executor = PlanExecutor()

    # Create
    plan = await planner.create_plan(
        goal="Research competitor pricing for CRM market",
        tools=["websearch"],
        max_steps=5,
    )

    # Validate
    validation = planner.validate_plan(plan)
    assert validation.is_valid

    # Execute
    result = await executor.execute(plan)

    assert result.plan_id == plan.plan_id
    assert result.status in ["completed", "partial"]
    assert result.total_tokens > 0

@pytest.mark.asyncio
async def test_plan_pause_resume():
    """Test plan pause and resume functionality."""
    planner = Planner()
    executor = PlanExecutor()

    plan = await planner.create_plan(
        goal="Multi-step analysis task",
        max_steps=10,
    )

    # Start execution (non-blocking)
    import asyncio
    task = asyncio.create_task(executor.execute(plan))

    # Wait briefly, then pause
    await asyncio.sleep(0.5)
    paused = await executor.pause(plan.plan_id)

    if paused:
        # Get status
        status = await executor.get_status(plan.plan_id)
        assert status.overall_status == "paused"

        # Resume
        result = await executor.resume(plan.plan_id)
        assert result.status in ["completed", "partial"]
    else:
        # Already completed
        result = await task
        assert result.status == "completed"
```

### 9.3 Mock Fixtures

```python
import pytest
from unittest.mock import AsyncMock, MagicMock

@pytest.fixture
def mock_planner():
    """Create a mock planner for testing."""
    planner = MagicMock()
    planner.create_plan = AsyncMock(return_value=Plan(
        plan_id="plan_test_123",
        goal="Test goal",
        steps=[
            PlanStep(
                step_id="step_1",
                description="Test step",
                status="pending",
            )
        ],
        created_at=datetime.now(),
        complexity=0.5,
        estimated_tokens=500,
        metadata=PlanMetadata(),
    ))
    return planner

@pytest.fixture
def mock_reasoning_manager():
    """Create a mock reasoning manager for testing."""
    manager = MagicMock()
    manager.execute = AsyncMock(return_value=StrategyResult(
        answer="Test answer",
        confidence=0.85,
        reasoning_trace=["Step 1: Analyzed", "Step 2: Concluded"],
        tokens_used=300,
        execution_time_seconds=2.5,
        model="anthropic:claude-sonnet-4-20250514",
        metadata={"strategy_name": "cot"},
    ))
    return manager
```

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2026-01-11 | Initial API guidelines |

---

*Document maintained by: API Architecture Team*
*Last updated: 2026-01-11*
