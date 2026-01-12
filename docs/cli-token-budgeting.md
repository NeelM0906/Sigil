# Sigil v2 CLI Token Budgeting System

## Document Information

| Field | Value |
|-------|-------|
| Version | 1.0.0 |
| Status | Design Specification |
| Created | 2026-01-11 |
| Author | Systems Architecture Team |
| Scope | Token Budget Management |

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Token Budget Overview](#2-token-budget-overview)
3. [Budget Allocation Strategy](#3-budget-allocation-strategy)
4. [Component Budgets](#4-component-budgets)
5. [Token Tracking Implementation](#5-token-tracking-implementation)
6. [Budget Visualization](#6-budget-visualization)
7. [Warning Thresholds and Alerts](#7-warning-thresholds-and-alerts)
8. [Budget Exceeded Scenarios](#8-budget-exceeded-scenarios)
9. [Forecasting and Estimation](#9-forecasting-and-estimation)
10. [Budget Optimization](#10-budget-optimization)
11. [Multi-Session Budget Management](#11-multi-session-budget-management)
12. [API Reference](#12-api-reference)

---

## 1. Executive Summary

### 1.1 Purpose

This document defines the comprehensive token budgeting system for Sigil v2 CLI execution. Token budgets ensure predictable costs, prevent runaway consumption, and enable informed decision-making during pipeline execution.

### 1.2 Key Principles

| Principle | Description |
|-----------|-------------|
| Tokens as Currency | All costs measured in tokens, not USD |
| Hierarchical Allocation | Budgets allocated from session to component |
| Soft Limits | Warnings before hard enforcement |
| Transparency | Real-time visibility into consumption |
| Accountability | Every token attributed to a component |

### 1.3 Budget Summary

```
+-----------------------------------------------------------------------------+
|                       SIGIL v2 TOKEN BUDGET SUMMARY                          |
+-----------------------------------------------------------------------------+

Total Session Budget:  256,000 tokens (Claude's context window)

Component Allocation:
+-------------------+----------+------------+--------------------------------+
| Component         | Tokens   | Percentage | Purpose                        |
+-------------------+----------+------------+--------------------------------+
| Routing           | 10,000   | 3.9%       | Intent classification          |
| Memory            | 30,000   | 11.7%      | RAG retrieval + LLM reading    |
| Planning          | 20,000   | 7.8%       | Plan generation                |
| Reasoning         | 150,000  | 58.6%      | Main LLM operations            |
| Contracts         | 10,000   | 3.9%       | Output validation              |
| Validation        | 10,000   | 3.9%       | Final verification             |
| Reserve           | 26,000   | 10.2%      | Retries and overflow           |
+-------------------+----------+------------+--------------------------------+
| TOTAL             | 256,000  | 100%       |                                |
+-------------------+----------+------------+--------------------------------+

Warning Thresholds:
- 50% (128,000 tokens): ELEVATED - Monitor closely
- 80% (204,800 tokens): CRITICAL - Consider completing soon
- 95% (243,200 tokens): EXCEEDED - Hard stop on new operations
```

---

## 2. Token Budget Overview

### 2.1 Why Token Budgeting?

Token budgeting addresses several critical challenges:

```
+-----------------------------------------------------------------------------+
|                      CHALLENGES ADDRESSED BY BUDGETING                       |
+-----------------------------------------------------------------------------+

Challenge                    | Without Budgeting        | With Budgeting
-----------------------------|--------------------------|-------------------------
Cost predictability          | Unknown until bill       | Known before execution
Runaway consumption          | Unbounded                | Capped at limit
Resource allocation          | First-come, first-served | Fair allocation
Performance monitoring       | Post-hoc analysis        | Real-time tracking
Decision support             | None                     | Data-driven choices
Error recovery               | No budget for retries    | Reserved capacity
```

### 2.2 Token Budget Architecture

```
+-----------------------------------------------------------------------------+
|                       TOKEN BUDGET ARCHITECTURE                              |
+-----------------------------------------------------------------------------+

                        +---------------------------+
                        |     SESSION BUDGET        |
                        |      256,000 tokens       |
                        +---------------------------+
                                    |
        +---------------------------+---------------------------+
        |                           |                           |
        v                           v                           v
+---------------+           +---------------+           +---------------+
| PIPELINE RUN  |           | PIPELINE RUN  |           | PIPELINE RUN  |
| Budget: 85K   |           | Budget: 85K   |           | Budget: 85K   |
+---------------+           +---------------+           +---------------+
        |
        +-------+-------+-------+-------+-------+
        |       |       |       |       |       |
        v       v       v       v       v       v
     +-----+ +-----+ +-----+ +-----+ +-----+ +-----+
     |Rout | |Mem  | |Plan | |Reas | |Cont | |Val  |
     |10K  | |30K  | |20K  | |150K | |10K  | |10K  |
     +-----+ +-----+ +-----+ +-----+ +-----+ +-----+
        |       |       |       |       |       |
        v       v       v       v       v       v
    Actual  Actual  Actual  Actual  Actual  Actual
    Usage   Usage   Usage   Usage   Usage   Usage
     50     150      0      450      0       0

Hierarchy:
- Session: Total available tokens for all operations
- Pipeline Run: Tokens for a single task execution
- Component: Tokens for a specific subsystem
- Operation: Individual LLM call consumption
```

### 2.3 Budget Lifecycle

```
+-----------------------------------------------------------------------------+
|                          BUDGET LIFECYCLE                                    |
+-----------------------------------------------------------------------------+

Phase 1: Initialization
+-----------------------+
| Create session        |
| Allocate total budget |
| Set component limits  |
+-----------------------+
          |
          v
Phase 2: Allocation
+-----------------------+
| Route to components   |
| Reserve for retries   |
| Track allocations     |
+-----------------------+
          |
          v
Phase 3: Consumption
+-----------------------+
| Execute operations    |
| Track per-operation   |
| Accumulate totals     |
| Check thresholds      |
+-----------------------+
          |
          v
Phase 4: Monitoring
+-----------------------+
| Log consumption       |
| Update displays       |
| Trigger warnings      |
| Forecast completion   |
+-----------------------+
          |
          v
Phase 5: Finalization
+-----------------------+
| Summarize usage       |
| Generate report       |
| Archive metrics       |
+-----------------------+
```

---

## 3. Budget Allocation Strategy

### 3.1 Allocation Philosophy

The budget allocation strategy is designed around typical Sigil v2 workloads:

```
+-----------------------------------------------------------------------------+
|                      ALLOCATION PHILOSOPHY                                   |
+-----------------------------------------------------------------------------+

Principle 1: Reasoning Dominates
--------------------------------
Most tokens go to the reasoning component because:
- Multiple LLM calls for complex tasks
- Iterative refinement of outputs
- Tool use with context maintenance
- Chain-of-thought processing

Allocation: 58.6% (150,000 tokens)


Principle 2: Memory Requires Headroom
-------------------------------------
Memory retrieval can be unpredictable:
- RAG may return many results
- LLM reading for complex queries
- Cross-session context loading

Allocation: 11.7% (30,000 tokens)


Principle 3: Reserve for Resilience
-----------------------------------
Always keep tokens available for:
- Retry failed operations
- Handle unexpected complexity
- Error recovery prompts
- Graceful degradation

Allocation: 10.2% (26,000 tokens)


Principle 4: Overhead Must Be Bounded
-------------------------------------
Routing, planning, and validation are overhead:
- Should not consume main budget
- Have predictable token usage
- Can be optimized with caching

Allocation: 15.6% combined (40,000 tokens)
```

### 3.2 Default Allocations

```python
# sigil/telemetry/budget_defaults.py

from dataclasses import dataclass
from typing import Dict


@dataclass
class BudgetAllocation:
    """Default budget allocation configuration."""

    # Total session budget
    TOTAL_BUDGET: int = 256_000

    # Component allocations (must sum to TOTAL_BUDGET)
    COMPONENT_ALLOCATIONS: Dict[str, int] = None

    def __post_init__(self):
        if self.COMPONENT_ALLOCATIONS is None:
            self.COMPONENT_ALLOCATIONS = {
                "routing": 10_000,      # 3.9%
                "memory": 30_000,       # 11.7%
                "planning": 20_000,     # 7.8%
                "reasoning": 150_000,   # 58.6%
                "contracts": 10_000,    # 3.9%
                "validation": 10_000,   # 3.9%
                "reserve": 26_000,      # 10.2%
            }

    @property
    def allocated_total(self) -> int:
        """Sum of all allocations."""
        return sum(self.COMPONENT_ALLOCATIONS.values())

    @property
    def is_balanced(self) -> bool:
        """Check if allocations sum to total budget."""
        return self.allocated_total == self.TOTAL_BUDGET

    def get_allocation(self, component: str) -> int:
        """Get allocation for a component."""
        return self.COMPONENT_ALLOCATIONS.get(component, 0)

    def validate(self) -> bool:
        """Validate allocation configuration."""
        if not self.is_balanced:
            raise ValueError(
                f"Allocations ({self.allocated_total}) != "
                f"Total budget ({self.TOTAL_BUDGET})"
            )
        return True


# Pre-configured allocation profiles
class AllocationProfiles:
    """Pre-defined allocation profiles for different use cases."""

    @staticmethod
    def default() -> BudgetAllocation:
        """Standard allocation for general use."""
        return BudgetAllocation()

    @staticmethod
    def reasoning_heavy() -> BudgetAllocation:
        """Profile for complex reasoning tasks."""
        return BudgetAllocation(
            COMPONENT_ALLOCATIONS={
                "routing": 5_000,
                "memory": 20_000,
                "planning": 15_000,
                "reasoning": 180_000,  # 70.3%
                "contracts": 10_000,
                "validation": 10_000,
                "reserve": 16_000,
            }
        )

    @staticmethod
    def memory_heavy() -> BudgetAllocation:
        """Profile for memory-intensive operations."""
        return BudgetAllocation(
            COMPONENT_ALLOCATIONS={
                "routing": 10_000,
                "memory": 80_000,     # 31.3%
                "planning": 20_000,
                "reasoning": 100_000,
                "contracts": 10_000,
                "validation": 10_000,
                "reserve": 26_000,
            }
        )

    @staticmethod
    def conservative() -> BudgetAllocation:
        """Profile with large reserve for reliability."""
        return BudgetAllocation(
            COMPONENT_ALLOCATIONS={
                "routing": 8_000,
                "memory": 25_000,
                "planning": 15_000,
                "reasoning": 120_000,
                "contracts": 8_000,
                "validation": 8_000,
                "reserve": 72_000,   # 28.1%
            }
        )
```

### 3.3 Allocation Visualization

```
+-----------------------------------------------------------------------------+
|                      BUDGET ALLOCATION VISUALIZATION                         |
+-----------------------------------------------------------------------------+

DEFAULT PROFILE (256,000 tokens):

Routing     [====                                                    ]  3.9%
Memory      [============                                            ] 11.7%
Planning    [========                                                ]  7.8%
Reasoning   [============================================================] 58.6%
Contracts   [====                                                    ]  3.9%
Validation  [====                                                    ]  3.9%
Reserve     [==========                                              ] 10.2%
            |---------|---------|---------|---------|---------|---------|
            0%        16.7%     33.3%     50%       66.7%     83.3%     100%


REASONING-HEAVY PROFILE (256,000 tokens):

Routing     [==                                                      ]  2.0%
Memory      [========                                                ]  7.8%
Planning    [======                                                  ]  5.9%
Reasoning   [====================================================================] 70.3%
Contracts   [====                                                    ]  3.9%
Validation  [====                                                    ]  3.9%
Reserve     [======                                                  ]  6.3%


MEMORY-HEAVY PROFILE (256,000 tokens):

Routing     [====                                                    ]  3.9%
Memory      [================================                        ] 31.3%
Planning    [========                                                ]  7.8%
Reasoning   [========================================                ] 39.1%
Contracts   [====                                                    ]  3.9%
Validation  [====                                                    ]  3.9%
Reserve     [==========                                              ] 10.2%
```

---

## 4. Component Budgets

### 4.1 Routing Component

```
+-----------------------------------------------------------------------------+
|                      ROUTING COMPONENT BUDGET                                |
+-----------------------------------------------------------------------------+

Allocation: 10,000 tokens (3.9%)

Purpose:
- Classify user intent
- Determine pipeline configuration
- Select reasoning strategy
- Route to appropriate handlers

Typical Operations:
+----------------------------+--------+----------------------------------+
| Operation                  | Tokens | Description                      |
+----------------------------+--------+----------------------------------+
| Intent classification      | 50-200 | Classify request type            |
| Complexity assessment      | 50-150 | Determine task complexity        |
| Strategy selection         | 0-100  | Pick reasoning approach          |
| Handler routing            | 0-50   | Route to specific handler        |
+----------------------------+--------+----------------------------------+
| Typical total per request  | 100-500|                                  |
+----------------------------+--------+----------------------------------+

Budget Allows:
- 20-100 routing operations per session
- Headroom for complex classification
- Buffer for retry on ambiguous inputs

Warning Signs:
- Using >500 tokens per routing: Check prompt efficiency
- Multiple routing attempts: Intent unclear, may need user clarification
```

### 4.2 Memory Component

```
+-----------------------------------------------------------------------------+
|                       MEMORY COMPONENT BUDGET                                |
+-----------------------------------------------------------------------------+

Allocation: 30,000 tokens (11.7%)

Purpose:
- Retrieve relevant memories via RAG
- Read memory categories with LLM
- Extract facts from new resources
- Consolidate memory items

Typical Operations:
+----------------------------+--------+----------------------------------+
| Operation                  | Tokens | Description                      |
+----------------------------+--------+----------------------------------+
| RAG retrieval              | 100-500| Embedding search results         |
| LLM memory reading         | 500-2K | Deep memory analysis             |
| Fact extraction            | 200-1K | Extract from new data            |
| Memory consolidation       | 300-1K | Aggregate into categories        |
+----------------------------+--------+----------------------------------+
| Typical total per request  | 500-3K |                                  |
+----------------------------+--------+----------------------------------+

Budget Allows:
- 10-60 memory operations per session
- Mix of fast RAG and slow LLM reading
- Background consolidation tasks

Memory Retrieval Modes:
+-----------+--------+----------+--------------------------------------+
| Mode      | Tokens | Latency  | Use Case                             |
+-----------+--------+----------+--------------------------------------+
| RAG Only  | 100-500| 50-200ms | Simple factual queries               |
| LLM Only  | 500-2K | 1-3s     | Complex reasoning about memories     |
| Hybrid    | 300-2K | 0.5-3s   | Unknown complexity (start RAG)       |
+-----------+--------+----------+--------------------------------------+

Warning Signs:
- Frequent LLM reading: Consider pre-computing category summaries
- High token per retrieval: Memory content too verbose
- Repeated extractions: Check resource deduplication
```

### 4.3 Planning Component

```
+-----------------------------------------------------------------------------+
|                      PLANNING COMPONENT BUDGET                               |
+-----------------------------------------------------------------------------+

Allocation: 20,000 tokens (7.8%)

Purpose:
- Decompose tasks into steps
- Determine step dependencies
- Estimate resource requirements
- Create execution plan

Typical Operations:
+----------------------------+--------+----------------------------------+
| Operation                  | Tokens | Description                      |
+----------------------------+--------+----------------------------------+
| Task decomposition         | 500-2K | Break task into steps            |
| Dependency analysis        | 100-500| Order steps correctly            |
| Resource estimation        | 100-300| Estimate tokens per step         |
| Plan optimization          | 200-500| Reduce unnecessary steps         |
+----------------------------+--------+----------------------------------+
| Typical total per request  | 500-2K |                                  |
+----------------------------+--------+----------------------------------+

Budget Allows:
- 10-40 planning operations per session
- Complex multi-step task decomposition
- Plan revision if execution fails

Planning Strategies:
+-----------------+--------+------------------------------------------+
| Strategy        | Tokens | Description                              |
+-----------------+--------+------------------------------------------+
| Simple          | 0      | Direct execution, no planning            |
| Template        | 100-300| Use pre-defined plan template            |
| Generated       | 500-2K | LLM generates custom plan                |
| Adaptive        | 1K-3K  | Plan + runtime adjustments               |
+-----------------+--------+------------------------------------------+

Warning Signs:
- Plans with >10 steps: Task may be too complex
- Multiple plan revisions: Initial task unclear
- Zero planning tokens: Verify simple tasks don't need plans
```

### 4.4 Reasoning Component

```
+-----------------------------------------------------------------------------+
|                     REASONING COMPONENT BUDGET                               |
+-----------------------------------------------------------------------------+

Allocation: 150,000 tokens (58.6%)

Purpose:
- Execute main LLM reasoning
- Process tool calls and results
- Generate intermediate outputs
- Apply reasoning strategies

Typical Operations:
+----------------------------+---------+---------------------------------+
| Operation                  | Tokens  | Description                     |
+----------------------------+---------+---------------------------------+
| Direct response            | 100-500 | Simple, single-turn response    |
| Chain-of-thought           | 500-2K  | Step-by-step reasoning          |
| Tree-of-thoughts           | 2K-10K  | Explore multiple paths          |
| ReAct iteration            | 500-2K  | Per thought-action-observation  |
| MCTS simulation            | 5K-20K  | Monte Carlo tree search         |
+----------------------------+---------+---------------------------------+
| Typical total per request  | 1K-50K  |                                 |
+----------------------------+---------+---------------------------------+

Budget Allows:
- 3-150 reasoning operations depending on complexity
- Full reasoning chain for complex tasks
- Multiple iterations for quality

Strategy Token Estimates:
+-------------------+------------+----------+----------------------------+
| Strategy          | Per-Step   | Total    | Typical Use Case           |
+-------------------+------------+----------+----------------------------+
| Direct            | 100-500    | 100-500  | Simple queries             |
| Chain-of-Thought  | 500-2K     | 1K-5K    | Moderate reasoning         |
| Tree-of-Thoughts  | 2K-5K      | 5K-20K   | Complex with alternatives  |
| ReAct             | 500-2K     | 2K-20K   | Tool-using tasks           |
| MCTS              | 2K-10K     | 10K-50K  | Critical decisions         |
+-------------------+------------+----------+----------------------------+

Warning Signs:
- Single operation >10K tokens: Check for verbose context
- >5 ReAct iterations: Task may be poorly defined
- MCTS for simple tasks: Overkill, use simpler strategy
```

### 4.5 Contracts Component

```
+-----------------------------------------------------------------------------+
|                     CONTRACTS COMPONENT BUDGET                               |
+-----------------------------------------------------------------------------+

Allocation: 10,000 tokens (3.9%)

Purpose:
- Define output requirements
- Validate deliverables
- Enforce constraints
- Handle validation failures

Typical Operations:
+----------------------------+--------+----------------------------------+
| Operation                  | Tokens | Description                      |
+----------------------------+--------+----------------------------------+
| Schema validation          | 0      | Programmatic, no tokens          |
| Content validation         | 100-500| Check content requirements       |
| Quality assessment         | 200-1K | Score output quality             |
| Retry prompt generation    | 200-500| Generate feedback for retry      |
+----------------------------+--------+----------------------------------+
| Typical total per request  | 200-1K |                                  |
+----------------------------+--------+----------------------------------+

Budget Allows:
- 10-50 contract validations per session
- Multiple retries with feedback
- Complex quality assessments

Contract Types:
+-------------------+--------+------------------------------------------+
| Contract          | Tokens | Validation Focus                         |
+-------------------+--------+------------------------------------------+
| lead_qualification| 300-800| BANT score, recommended action           |
| agent_config      | 200-500| Schema compliance, tool validity         |
| research_report   | 500-1K | Citations, findings completeness         |
+-------------------+--------+------------------------------------------+

Warning Signs:
- Multiple validation failures: Output quality issue upstream
- High tokens per validation: Simplify contract requirements
- Skipped validations: Risk of bad outputs to user
```

### 4.6 Validation Component

```
+-----------------------------------------------------------------------------+
|                     VALIDATION COMPONENT BUDGET                              |
+-----------------------------------------------------------------------------+

Allocation: 10,000 tokens (3.9%)

Purpose:
- Final output verification
- Cross-check contract results
- Ensure coherent response
- Quality gate before user

Typical Operations:
+----------------------------+--------+----------------------------------+
| Operation                  | Tokens | Description                      |
+----------------------------+--------+----------------------------------+
| Output coherence check     | 100-300| Ensure response makes sense      |
| Fact verification          | 200-500| Verify claims against memory     |
| Format validation          | 0-100  | Check output structure           |
| Final quality score        | 100-300| Overall quality assessment       |
+----------------------------+--------+----------------------------------+
| Typical total per request  | 200-800|                                  |
+----------------------------+--------+----------------------------------+

Budget Allows:
- 12-50 validation operations per session
- Multiple quality checks
- Deep verification when needed

Validation Levels:
+-----------+--------+------------------------------------------------+
| Level     | Tokens | Checks Performed                               |
+-----------+--------+------------------------------------------------+
| Basic     | 0-100  | Format only (programmatic)                     |
| Standard  | 200-500| Format + coherence + basic facts               |
| Thorough  | 500-1K | All above + memory cross-reference             |
| Critical  | 1K-2K  | All above + independent verification           |
+-----------+--------+------------------------------------------------+

Warning Signs:
- Validation failures at end: Issues should catch earlier
- High token validation: Consider caching checks
- Skipped validation: Never skip for user-facing outputs
```

### 4.7 Reserve Budget

```
+-----------------------------------------------------------------------------+
|                        RESERVE BUDGET                                        |
+-----------------------------------------------------------------------------+

Allocation: 26,000 tokens (10.2%)

Purpose:
- Buffer for retries
- Handle unexpected complexity
- Error recovery
- Graceful degradation

Reserve Categories:
+----------------------------+--------+----------------------------------+
| Category                   | Tokens | Purpose                          |
+----------------------------+--------+----------------------------------+
| Retry buffer               | 15,000 | Failed operation retries         |
| Complexity overflow        | 5,000  | Tasks exceeding estimates        |
| Error recovery             | 3,000  | Error handling prompts           |
| Emergency reserve          | 3,000  | Critical operations only         |
+----------------------------+--------+----------------------------------+

Usage Policy:
- Retry buffer: Automatically accessed on transient failures
- Complexity overflow: Released when component budget exhausted
- Error recovery: For generating user-friendly error messages
- Emergency reserve: Manually authorized for critical completion

Reserve Trigger Conditions:
+----------------------------+------------------------------------------+
| Condition                  | Action                                   |
+----------------------------+------------------------------------------+
| Component budget exhausted | Release 20% of overflow buffer           |
| Retry needed               | Deduct from retry buffer                 |
| Unhandled error            | Use error recovery allocation            |
| Session completion at risk | Request emergency reserve authorization  |
+----------------------------+------------------------------------------+

Warning Signs:
- Reserve >50% used: Review component efficiency
- Emergency reserve accessed: Post-mortem required
- Reserve exhausted: Session must terminate
```

---

## 5. Token Tracking Implementation

### 5.1 Token Tracker Class

```python
# sigil/telemetry/token_tracker.py

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Optional, List, Callable
from enum import Enum
import threading


class BudgetWarningLevel(str, Enum):
    """Warning levels for budget consumption."""
    NORMAL = "normal"
    ELEVATED = "elevated"
    CRITICAL = "critical"
    EXCEEDED = "exceeded"


@dataclass
class TokenOperation:
    """Record of a single token-consuming operation."""
    operation_id: str
    component: str
    operation_type: str
    tokens_consumed: int
    timestamp: datetime
    metadata: Dict = field(default_factory=dict)


@dataclass
class ComponentBudget:
    """Budget tracking for a single component."""
    name: str
    allocated: int
    used: int = 0
    operations: int = 0
    history: List[TokenOperation] = field(default_factory=list)

    @property
    def remaining(self) -> int:
        return max(0, self.allocated - self.used)

    @property
    def utilization(self) -> float:
        if self.allocated == 0:
            return 0.0
        return (self.used / self.allocated) * 100

    @property
    def avg_per_operation(self) -> float:
        if self.operations == 0:
            return 0.0
        return self.used / self.operations

    def consume(self, tokens: int, operation: TokenOperation) -> bool:
        """Consume tokens and record operation."""
        self.used += tokens
        self.operations += 1
        self.history.append(operation)
        return self.remaining >= 0


@dataclass
class SessionBudget:
    """Complete budget state for a session."""
    session_id: str
    total: int
    used: int = 0
    components: Dict[str, ComponentBudget] = field(default_factory=dict)
    operations: List[TokenOperation] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)

    # Thresholds
    warning_threshold: float = 0.50
    critical_threshold: float = 0.80
    exceeded_threshold: float = 0.95

    @property
    def remaining(self) -> int:
        return max(0, self.total - self.used)

    @property
    def utilization(self) -> float:
        if self.total == 0:
            return 0.0
        return (self.used / self.total) * 100

    @property
    def warning_level(self) -> BudgetWarningLevel:
        util = self.utilization / 100
        if util >= self.exceeded_threshold:
            return BudgetWarningLevel.EXCEEDED
        elif util >= self.critical_threshold:
            return BudgetWarningLevel.CRITICAL
        elif util >= self.warning_threshold:
            return BudgetWarningLevel.ELEVATED
        return BudgetWarningLevel.NORMAL


class TokenTracker:
    """
    Thread-safe token tracking with comprehensive metrics.

    Features:
    - Hierarchical budget management
    - Real-time consumption tracking
    - Warning threshold monitoring
    - Operation history for auditing
    - Forecasting support
    """

    def __init__(
        self,
        session_id: str,
        total_budget: int = 256_000,
        allocations: Dict[str, int] = None,
        logger: "ExecutionLogger" = None,
    ) -> None:
        """
        Initialize the token tracker.

        Args:
            session_id: Unique session identifier
            total_budget: Total token budget
            allocations: Component allocations (uses defaults if None)
            logger: Optional execution logger
        """
        self.session_id = session_id
        self.logger = logger
        self._lock = threading.Lock()

        # Initialize budget
        self.budget = SessionBudget(
            session_id=session_id,
            total=total_budget,
        )

        # Set up component budgets
        default_allocations = {
            "routing": 10_000,
            "memory": 30_000,
            "planning": 20_000,
            "reasoning": 150_000,
            "contracts": 10_000,
            "validation": 10_000,
            "reserve": 26_000,
        }
        allocations = allocations or default_allocations

        for name, tokens in allocations.items():
            self.budget.components[name] = ComponentBudget(
                name=name,
                allocated=tokens,
            )

        # Warning callbacks
        self._warning_callbacks: List[Callable] = []
        self._last_warning_level = BudgetWarningLevel.NORMAL

    def track(
        self,
        component: str,
        tokens: int,
        operation_type: str = "unknown",
        metadata: Dict = None,
    ) -> bool:
        """
        Track token consumption for an operation.

        Args:
            component: Component name
            tokens: Tokens consumed
            operation_type: Type of operation
            metadata: Additional operation metadata

        Returns:
            True if consumption succeeded (within budget)
        """
        with self._lock:
            # Create operation record
            operation = TokenOperation(
                operation_id=f"{self.session_id}_{len(self.budget.operations)}",
                component=component,
                operation_type=operation_type,
                tokens_consumed=tokens,
                timestamp=datetime.now(),
                metadata=metadata or {},
            )

            # Update component budget
            if component in self.budget.components:
                self.budget.components[component].consume(tokens, operation)

            # Update session budget
            self.budget.used += tokens
            self.budget.operations.append(operation)

            # Check warning levels
            self._check_warnings()

            return self.budget.remaining > 0

    def _check_warnings(self) -> None:
        """Check and trigger warning callbacks if threshold crossed."""
        current_level = self.budget.warning_level

        if current_level != self._last_warning_level:
            self._last_warning_level = current_level
            for callback in self._warning_callbacks:
                callback(current_level, self.budget.used, self.budget.total)

    def on_warning(self, callback: Callable) -> None:
        """Register a warning callback."""
        self._warning_callbacks.append(callback)

    def can_consume(self, tokens: int, component: str = None) -> bool:
        """
        Check if tokens can be consumed without exceeding budget.

        Args:
            tokens: Tokens to check
            component: Optional component to check against

        Returns:
            True if consumption would succeed
        """
        with self._lock:
            # Check session budget
            if self.budget.remaining < tokens:
                return False

            # Check component budget if specified
            if component and component in self.budget.components:
                comp = self.budget.components[component]
                if comp.remaining < tokens:
                    return False

            return True

    def get_status(self) -> Dict:
        """Get complete budget status."""
        with self._lock:
            return {
                "session_id": self.session_id,
                "total": self.budget.total,
                "used": self.budget.used,
                "remaining": self.budget.remaining,
                "utilization": self.budget.utilization,
                "warning_level": self.budget.warning_level.value,
                "operations_count": len(self.budget.operations),
                "components": {
                    name: {
                        "allocated": comp.allocated,
                        "used": comp.used,
                        "remaining": comp.remaining,
                        "utilization": comp.utilization,
                        "operations": comp.operations,
                    }
                    for name, comp in self.budget.components.items()
                },
            }

    def get_component_status(self, component: str) -> Optional[Dict]:
        """Get status for a specific component."""
        with self._lock:
            if component not in self.budget.components:
                return None

            comp = self.budget.components[component]
            return {
                "name": comp.name,
                "allocated": comp.allocated,
                "used": comp.used,
                "remaining": comp.remaining,
                "utilization": comp.utilization,
                "operations": comp.operations,
                "avg_per_operation": comp.avg_per_operation,
                "recent_operations": [
                    {
                        "operation_id": op.operation_id,
                        "type": op.operation_type,
                        "tokens": op.tokens_consumed,
                        "timestamp": op.timestamp.isoformat(),
                    }
                    for op in comp.history[-5:]
                ],
            }

    def forecast(
        self,
        operations_remaining: int = None,
        time_remaining_seconds: float = None,
    ) -> Dict:
        """
        Forecast token usage based on current consumption.

        Args:
            operations_remaining: Expected remaining operations
            time_remaining_seconds: Expected remaining time

        Returns:
            Forecast metrics
        """
        with self._lock:
            if not self.budget.operations:
                return {
                    "forecast_total": 0,
                    "forecast_remaining": self.budget.total,
                    "will_exceed": False,
                    "confidence": "low",
                }

            # Calculate rates
            total_ops = len(self.budget.operations)
            avg_tokens_per_op = self.budget.used / total_ops

            elapsed = (
                datetime.now() - self.budget.created_at
            ).total_seconds()
            tokens_per_second = self.budget.used / elapsed if elapsed > 0 else 0

            # Forecast based on operations
            if operations_remaining is not None:
                ops_forecast = self.budget.used + (
                    avg_tokens_per_op * operations_remaining
                )
            else:
                ops_forecast = None

            # Forecast based on time
            if time_remaining_seconds is not None:
                time_forecast = self.budget.used + (
                    tokens_per_second * time_remaining_seconds
                )
            else:
                time_forecast = None

            # Use best available forecast
            if ops_forecast and time_forecast:
                forecast = (ops_forecast + time_forecast) / 2
                confidence = "high"
            elif ops_forecast:
                forecast = ops_forecast
                confidence = "medium"
            elif time_forecast:
                forecast = time_forecast
                confidence = "medium"
            else:
                forecast = self.budget.used * 2  # Double current usage
                confidence = "low"

            return {
                "forecast_total": int(forecast),
                "forecast_remaining": max(0, self.budget.total - int(forecast)),
                "will_exceed": forecast > self.budget.total,
                "avg_tokens_per_operation": avg_tokens_per_op,
                "tokens_per_second": tokens_per_second,
                "confidence": confidence,
            }
```

### 5.2 Token Counting Utilities

```python
# sigil/telemetry/token_counting.py

import tiktoken
from typing import List, Dict, Any, Optional


class TokenCounter:
    """
    Utility for counting tokens in text and messages.

    Uses tiktoken for accurate Claude token counting.
    """

    def __init__(self, model: str = "claude-3-sonnet") -> None:
        """
        Initialize the token counter.

        Args:
            model: Model name for tokenizer selection
        """
        # Use cl100k_base encoding (closest to Claude)
        self.encoding = tiktoken.get_encoding("cl100k_base")

    def count_text(self, text: str) -> int:
        """
        Count tokens in a text string.

        Args:
            text: Text to count

        Returns:
            Token count
        """
        if not text:
            return 0
        return len(self.encoding.encode(text))

    def count_messages(self, messages: List[Dict[str, str]]) -> int:
        """
        Count tokens in a list of messages.

        Args:
            messages: List of message dicts with 'role' and 'content'

        Returns:
            Total token count including message overhead
        """
        total = 0

        for message in messages:
            # Message overhead (~4 tokens per message)
            total += 4

            # Role token
            total += 1

            # Content tokens
            content = message.get("content", "")
            total += self.count_text(content)

        # Conversation overhead (~3 tokens)
        total += 3

        return total

    def count_with_system(
        self,
        system_prompt: str,
        messages: List[Dict[str, str]],
    ) -> int:
        """
        Count tokens including system prompt.

        Args:
            system_prompt: System prompt text
            messages: Conversation messages

        Returns:
            Total token count
        """
        system_tokens = self.count_text(system_prompt) + 4  # System overhead
        message_tokens = self.count_messages(messages)

        return system_tokens + message_tokens

    def estimate_response(
        self,
        prompt_tokens: int,
        max_tokens: int = 4096,
    ) -> Dict[str, int]:
        """
        Estimate response token usage.

        Args:
            prompt_tokens: Input prompt tokens
            max_tokens: Maximum response tokens

        Returns:
            Estimated usage metrics
        """
        # Typical response is 30-50% of max_tokens
        estimated_response = int(max_tokens * 0.4)

        return {
            "prompt_tokens": prompt_tokens,
            "max_response_tokens": max_tokens,
            "estimated_response_tokens": estimated_response,
            "estimated_total": prompt_tokens + estimated_response,
        }


# Convenience functions
_counter: Optional[TokenCounter] = None


def get_counter() -> TokenCounter:
    """Get or create singleton token counter."""
    global _counter
    if _counter is None:
        _counter = TokenCounter()
    return _counter


def count_tokens(text: str) -> int:
    """Count tokens in text."""
    return get_counter().count_text(text)


def count_message_tokens(messages: List[Dict[str, str]]) -> int:
    """Count tokens in messages."""
    return get_counter().count_messages(messages)
```

---

## 6. Budget Visualization

### 6.1 Console Display Formats

```
+-----------------------------------------------------------------------------+
|                      BUDGET VISUALIZATION FORMATS                            |
+-----------------------------------------------------------------------------+

FORMAT 1: Compact (Single Line)
-------------------------------
Tokens: 650 / 256,000 (0.25%) [######............................] 255,350 left


FORMAT 2: Component Breakdown
-----------------------------
[Memory]      150 tokens  ====
[Routing]      50 tokens  ==
[Planning]      0 tokens
[Reasoning]   450 tokens  ===========
[Contracts]     0 tokens
[Validation]    0 tokens
----------------------------------
Total:        650 tokens  ##.... (0.25% of 256K)
Remaining:    255,350 tokens


FORMAT 3: Dashboard View
------------------------
+==============================================================+
|                    TOKEN BUDGET STATUS                        |
+==============================================================+
| Session: test-1                                               |
| Status: Running                                               |
+--------------------------------------------------------------+
|                                                               |
| COMPONENT USAGE:                                              |
|                                                               |
| Routing    [====                    ]  50 / 10,000   (0.5%)   |
| Memory     [========                ] 150 / 30,000   (0.5%)   |
| Planning   [                        ]   0 / 20,000   (0.0%)   |
| Reasoning  [==                      ] 450 / 150,000  (0.3%)   |
| Contracts  [                        ]   0 / 10,000   (0.0%)   |
| Validation [                        ]   0 / 10,000   (0.0%)   |
|                                                               |
+--------------------------------------------------------------+
| BUDGET SUMMARY:                                               |
|                                                               |
| Total:     [##                                ] 650 / 256,000 |
| Used:      0.25%                                              |
| Remaining: 255,350 tokens                                     |
| Status:    NORMAL                                             |
|                                                               |
+--------------------------------------------------------------+
| FORECAST:                                                     |
|                                                               |
| Rate:      130 tokens/operation                               |
| If 10 more operations: ~1,950 total (0.76%)                   |
| Budget sufficient: YES                                        |
|                                                               |
+==============================================================+


FORMAT 4: Warning State
-----------------------
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!                      BUDGET WARNING                          !!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

Status: CRITICAL (83% used)

Used:      212,480 / 256,000 tokens
Remaining: 43,520 tokens

[##################################################...........]

Component Over Budget:
  - Reasoning: 145,000 / 150,000 (96.7%) <-- NEAR LIMIT

Recommended Actions:
  1. Complete current operation
  2. Avoid new complex reasoning tasks
  3. Consider session wrap-up

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
```

### 6.2 Visualization Implementation

```python
# sigil/interfaces/cli/budget_display.py

from dataclasses import dataclass
from typing import Dict, Optional
from enum import Enum


class DisplayStyle(str, Enum):
    """Display style options."""
    COMPACT = "compact"
    BREAKDOWN = "breakdown"
    DASHBOARD = "dashboard"
    MINIMAL = "minimal"


@dataclass
class BudgetDisplay:
    """
    Renders budget status in various formats.

    Supports multiple display styles for different contexts.
    """

    # Color codes
    RESET = "\033[0m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    RED = "\033[31m"
    BOLD = "\033[1m"
    DIM = "\033[2m"

    def render(
        self,
        status: Dict,
        style: DisplayStyle = DisplayStyle.BREAKDOWN,
    ) -> str:
        """
        Render budget status in specified style.

        Args:
            status: Budget status dictionary from TokenTracker
            style: Display style to use

        Returns:
            Formatted string for display
        """
        if style == DisplayStyle.COMPACT:
            return self._render_compact(status)
        elif style == DisplayStyle.BREAKDOWN:
            return self._render_breakdown(status)
        elif style == DisplayStyle.DASHBOARD:
            return self._render_dashboard(status)
        else:
            return self._render_minimal(status)

    def _render_minimal(self, status: Dict) -> str:
        """Render minimal single-value display."""
        used = status["used"]
        total = status["total"]
        pct = (used / total) * 100 if total > 0 else 0
        return f"Tokens: {used:,} / {total:,} ({pct:.2f}%)"

    def _render_compact(self, status: Dict) -> str:
        """Render compact single-line display."""
        used = status["used"]
        total = status["total"]
        remaining = status["remaining"]
        pct = status["utilization"]

        # Progress bar (30 chars)
        filled = int(pct / 100 * 30)
        bar = "#" * filled + "." * (30 - filled)

        # Color based on level
        color = self._color_for_level(status["warning_level"])

        return (
            f"Tokens: {used:,} / {total:,} ({pct:.2f}%) "
            f"{color}[{bar}]{self.RESET} {remaining:,} left"
        )

    def _render_breakdown(self, status: Dict) -> str:
        """Render component breakdown display."""
        lines = []

        # Component breakdown
        components = status.get("components", {})
        component_order = [
            "routing", "memory", "planning",
            "reasoning", "contracts", "validation"
        ]

        for name in component_order:
            if name in components:
                comp = components[name]
                used = comp["used"]
                bar = self._token_bar(used, 5000, 20)
                lines.append(f"[{name:12}] {used:6,} tokens  {bar}")

        # Separator
        lines.append("-" * 40)

        # Total
        used = status["used"]
        total = status["total"]
        remaining = status["remaining"]
        pct = status["utilization"]

        total_bar = self._progress_bar(pct, 30)
        lines.append(f"Total:        {used:6,} tokens  {total_bar} ({pct:.2f}% of {total//1000}K)")
        lines.append(f"Remaining:    {remaining:,} tokens")

        return "\n".join(lines)

    def _render_dashboard(self, status: Dict) -> str:
        """Render full dashboard display."""
        lines = []

        # Header
        lines.append("+" + "=" * 62 + "+")
        lines.append("|" + " " * 20 + "TOKEN BUDGET STATUS" + " " * 23 + "|")
        lines.append("+" + "=" * 62 + "+")
        lines.append(f"| Session: {status['session_id']:<52} |")

        warning = status["warning_level"]
        warning_color = self._color_for_level(warning)
        lines.append(f"| Status: {warning_color}{warning.upper():<53}{self.RESET} |")

        lines.append("+" + "-" * 62 + "+")
        lines.append("|" + " " * 62 + "|")
        lines.append("| COMPONENT USAGE:" + " " * 45 + "|")
        lines.append("|" + " " * 62 + "|")

        # Components
        components = status.get("components", {})
        for name in ["routing", "memory", "planning", "reasoning", "contracts", "validation"]:
            if name in components:
                comp = components[name]
                bar = self._progress_bar(comp["utilization"], 24)
                line = (
                    f"| {name:10} [{bar}] "
                    f"{comp['used']:>6,} / {comp['allocated']:>7,} "
                    f"({comp['utilization']:>5.1f}%) |"
                )
                lines.append(line)

        lines.append("|" + " " * 62 + "|")
        lines.append("+" + "-" * 62 + "+")
        lines.append("| BUDGET SUMMARY:" + " " * 46 + "|")
        lines.append("|" + " " * 62 + "|")

        # Summary
        used = status["used"]
        total = status["total"]
        remaining = status["remaining"]
        pct = status["utilization"]

        bar = self._progress_bar(pct, 36)
        lines.append(f"| Total:     [{bar}] {used:>6,} / {total:,} |")
        lines.append(f"| Used:      {pct:.2f}%" + " " * 49 + "|")
        lines.append(f"| Remaining: {remaining:,} tokens" + " " * (43 - len(str(remaining))) + "|")
        lines.append(f"| Status:    {warning.upper()}" + " " * (50 - len(warning)) + "|")
        lines.append("|" + " " * 62 + "|")
        lines.append("+" + "=" * 62 + "+")

        return "\n".join(lines)

    def _token_bar(self, tokens: int, scale: int, width: int) -> str:
        """Generate a token usage bar."""
        if scale == 0:
            return ""
        filled = min(width, int((tokens / scale) * width))
        return "=" * filled

    def _progress_bar(self, percentage: float, width: int) -> str:
        """Generate a progress bar with color."""
        filled = int(percentage / 100 * width)
        empty = width - filled

        if percentage < 50:
            color = self.GREEN
        elif percentage < 80:
            color = self.YELLOW
        else:
            color = self.RED

        return f"{color}{'#' * filled}{self.RESET}{'.' * empty}"

    def _color_for_level(self, level: str) -> str:
        """Get color for warning level."""
        colors = {
            "normal": self.GREEN,
            "elevated": self.YELLOW,
            "critical": self.RED,
            "exceeded": self.RED + self.BOLD,
        }
        return colors.get(level, self.RESET)


def format_budget_status(
    status: Dict,
    style: str = "breakdown",
) -> str:
    """
    Convenience function to format budget status.

    Args:
        status: Budget status from TokenTracker
        style: Display style name

    Returns:
        Formatted string
    """
    display = BudgetDisplay()
    return display.render(status, DisplayStyle(style))
```

---

## 7. Warning Thresholds and Alerts

### 7.1 Threshold Configuration

```
+-----------------------------------------------------------------------------+
|                      WARNING THRESHOLD CONFIGURATION                         |
+-----------------------------------------------------------------------------+

THRESHOLD LEVELS:

Level       | % Used  | Tokens Used  | Tokens Remaining | Action Required
------------|---------|--------------|------------------|------------------
NORMAL      | 0-50%   | 0-128,000    | 128,000-256,000  | None
ELEVATED    | 50-80%  | 128,000-204K | 52,000-128,000   | Monitor closely
CRITICAL    | 80-95%  | 204,800-243K | 12,800-51,200    | Prepare to complete
EXCEEDED    | >95%    | >243,200     | <12,800          | Stop new operations


THRESHOLD ACTIONS:

+-----------------------------------------------------------------------------+
| Level    | Log Event | Display Alert | Callback Trigger | Block Operations |
+-----------------------------------------------------------------------------+
| NORMAL   | No        | No            | No               | No               |
| ELEVATED | Yes       | Status change | Yes              | No               |
| CRITICAL | Yes       | Warning banner| Yes              | Soft (warn only) |
| EXCEEDED | Yes       | Error banner  | Yes              | Hard (block new) |
+-----------------------------------------------------------------------------+
```

### 7.2 Alert Implementation

```python
# sigil/telemetry/budget_alerts.py

from dataclasses import dataclass
from datetime import datetime
from typing import Callable, List, Optional, Dict
from enum import Enum


class AlertSeverity(str, Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class BudgetAlert:
    """A budget alert event."""
    alert_id: str
    severity: AlertSeverity
    warning_level: str
    message: str
    tokens_used: int
    tokens_total: int
    timestamp: datetime
    metadata: Dict = None

    @property
    def utilization(self) -> float:
        if self.tokens_total == 0:
            return 0.0
        return (self.tokens_used / self.tokens_total) * 100


class BudgetAlertManager:
    """
    Manages budget alerts and notifications.

    Features:
    - Configurable thresholds
    - Multiple alert handlers
    - Alert history
    - Deduplication
    """

    def __init__(
        self,
        warning_threshold: float = 0.50,
        critical_threshold: float = 0.80,
        exceeded_threshold: float = 0.95,
    ) -> None:
        """
        Initialize alert manager.

        Args:
            warning_threshold: % for elevated warning
            critical_threshold: % for critical warning
            exceeded_threshold: % for exceeded state
        """
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold
        self.exceeded_threshold = exceeded_threshold

        self._handlers: List[Callable[[BudgetAlert], None]] = []
        self._history: List[BudgetAlert] = []
        self._last_level: Optional[str] = None
        self._alert_count = 0

    def add_handler(self, handler: Callable[[BudgetAlert], None]) -> None:
        """Add an alert handler."""
        self._handlers.append(handler)

    def check_and_alert(
        self,
        tokens_used: int,
        tokens_total: int,
        metadata: Dict = None,
    ) -> Optional[BudgetAlert]:
        """
        Check budget status and generate alert if needed.

        Args:
            tokens_used: Current tokens used
            tokens_total: Total budget
            metadata: Additional context

        Returns:
            BudgetAlert if threshold crossed, None otherwise
        """
        utilization = tokens_used / tokens_total if tokens_total > 0 else 0

        # Determine current level
        if utilization >= self.exceeded_threshold:
            level = "exceeded"
            severity = AlertSeverity.CRITICAL
        elif utilization >= self.critical_threshold:
            level = "critical"
            severity = AlertSeverity.ERROR
        elif utilization >= self.warning_threshold:
            level = "elevated"
            severity = AlertSeverity.WARNING
        else:
            level = "normal"
            severity = AlertSeverity.INFO

        # Only alert on level change
        if level == self._last_level:
            return None

        self._last_level = level

        # Don't alert for normal
        if level == "normal":
            return None

        # Create alert
        self._alert_count += 1
        alert = BudgetAlert(
            alert_id=f"budget_alert_{self._alert_count}",
            severity=severity,
            warning_level=level,
            message=self._format_message(level, utilization, tokens_used, tokens_total),
            tokens_used=tokens_used,
            tokens_total=tokens_total,
            timestamp=datetime.now(),
            metadata=metadata,
        )

        # Store in history
        self._history.append(alert)

        # Notify handlers
        for handler in self._handlers:
            try:
                handler(alert)
            except Exception:
                pass  # Don't let handler failures break alerting

        return alert

    def _format_message(
        self,
        level: str,
        utilization: float,
        used: int,
        total: int,
    ) -> str:
        """Format alert message."""
        messages = {
            "elevated": (
                f"Budget usage elevated: {utilization*100:.1f}% "
                f"({used:,} / {total:,} tokens). Monitor consumption."
            ),
            "critical": (
                f"Budget CRITICAL: {utilization*100:.1f}% "
                f"({used:,} / {total:,} tokens). "
                f"Only {total - used:,} tokens remaining. "
                "Consider completing current task."
            ),
            "exceeded": (
                f"Budget EXCEEDED: {utilization*100:.1f}% "
                f"({used:,} / {total:,} tokens). "
                f"Only {total - used:,} tokens remaining. "
                "New operations will be blocked."
            ),
        }
        return messages.get(level, f"Budget alert: {level}")

    def get_history(self, limit: int = 10) -> List[BudgetAlert]:
        """Get recent alert history."""
        return list(reversed(self._history[-limit:]))

    def clear_history(self) -> None:
        """Clear alert history."""
        self._history.clear()


# Built-in alert handlers
def console_alert_handler(alert: BudgetAlert) -> None:
    """Print alert to console."""
    colors = {
        AlertSeverity.INFO: "\033[34m",
        AlertSeverity.WARNING: "\033[33m",
        AlertSeverity.ERROR: "\033[31m",
        AlertSeverity.CRITICAL: "\033[31m\033[1m",
    }
    reset = "\033[0m"
    color = colors.get(alert.severity, "")

    print(f"\n{color}[BUDGET ALERT - {alert.severity.value.upper()}]{reset}")
    print(f"{alert.message}")
    print()


def log_alert_handler(logger: "ExecutionLogger") -> Callable:
    """Create a logging alert handler."""
    async def handler(alert: BudgetAlert) -> None:
        await logger.log_event(
            event_type="budget_warning",
            component="budget",
            message=alert.message,
            tokens=alert.tokens_used,
            metadata={
                "warning_level": alert.warning_level,
                "severity": alert.severity.value,
                "utilization": alert.utilization,
            },
        )
    return handler
```

### 7.3 Alert Messages

```
+-----------------------------------------------------------------------------+
|                          ALERT MESSAGE TEMPLATES                             |
+-----------------------------------------------------------------------------+

ELEVATED (50% threshold):
-------------------------
[BUDGET ALERT - WARNING]
Budget usage elevated: 52.3% (133,888 / 256,000 tokens). Monitor consumption.


CRITICAL (80% threshold):
-------------------------
[BUDGET ALERT - ERROR]
Budget CRITICAL: 83.7% (214,272 / 256,000 tokens). Only 41,728 tokens
remaining. Consider completing current task.

Recommendations:
- Complete current operation before starting new ones
- Avoid initiating complex reasoning tasks
- Consider wrapping up the session


EXCEEDED (95% threshold):
-------------------------
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
[BUDGET ALERT - CRITICAL]
Budget EXCEEDED: 96.2% (246,272 / 256,000 tokens). Only 9,728 tokens
remaining. New operations will be blocked.

IMMEDIATE ACTIONS REQUIRED:
1. Complete any in-progress operations
2. No new reasoning or memory operations permitted
3. Only validation and finalization allowed
4. Session should be concluded

To continue working, start a new session.
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
```

---

## 8. Budget Exceeded Scenarios

### 8.1 Exceeded Handling Strategy

```
+-----------------------------------------------------------------------------+
|                    BUDGET EXCEEDED HANDLING STRATEGY                         |
+-----------------------------------------------------------------------------+

When budget is exceeded (>95%), the system must:

1. COMPLETE IN-PROGRESS OPERATIONS
   - Allow current LLM call to finish
   - Process pending tool results
   - Don't interrupt mid-operation

2. BLOCK NEW OPERATIONS
   - Reject new reasoning requests
   - Block new memory operations
   - Prevent new plan generation

3. ALLOW FINALIZATION
   - Permit validation of current output
   - Allow error message generation
   - Enable session summary creation

4. PRESERVE STATE
   - Save current progress
   - Log all consumption
   - Store partial results

5. NOTIFY USER
   - Display clear error message
   - Provide recommendations
   - Offer continuation options
```

### 8.2 Exceeded Scenarios

```
+-----------------------------------------------------------------------------+
|                      BUDGET EXCEEDED SCENARIOS                               |
+-----------------------------------------------------------------------------+

SCENARIO 1: Mid-Reasoning Exceeded
----------------------------------
Situation: Reasoning step consumes more than expected, budget exceeded mid-way

Handling:
1. Complete current LLM call (don't interrupt)
2. Skip remaining reasoning iterations
3. Return best result so far
4. Log partial completion
5. Notify user of truncated reasoning

Example:
  Before: 240,000 tokens used, reasoning step started
  During: Reasoning uses 20,000 tokens (260,000 > 256,000)
  After:  Return partial reasoning, mark as incomplete


SCENARIO 2: Memory Retrieval Exceeded
-------------------------------------
Situation: Large memory retrieval pushes over budget

Handling:
1. Complete retrieval (results already computed)
2. Block subsequent memory operations
3. Continue with retrieved memories only
4. No memory consolidation

Example:
  Before: 250,000 tokens, memory retrieval queued
  During: RAG returns results (8,000 tokens)
  After:  Use results, block further memory ops


SCENARIO 3: Planning Exceeded
-----------------------------
Situation: Plan generation itself exceeds budget

Handling:
1. Return partial plan
2. Mark uncompleted steps as "deferred"
3. Execute only completed steps
4. Log planning incomplete

Example:
  Plan has 5 steps, budget exceeded after step 3
  Execute: steps 1-3 only
  Defer: steps 4-5 for next session


SCENARIO 4: Validation Exceeded
-------------------------------
Situation: Final validation would exceed budget

Handling:
1. Skip LLM-based validation
2. Perform schema-only validation (free)
3. Mark output as "unverified"
4. Warn user about reduced quality assurance

Example:
  Output generated, 254,000 tokens used
  Validation needs 5,000 tokens
  Skip: LLM validation
  Do: Schema check only
```

### 8.3 Exceeded Response Implementation

```python
# sigil/telemetry/budget_exceeded.py

from dataclasses import dataclass
from typing import Optional, Dict, Any, List
from enum import Enum


class ExceededAction(str, Enum):
    """Actions when budget exceeded."""
    COMPLETE_CURRENT = "complete_current"
    RETURN_PARTIAL = "return_partial"
    SKIP_STEP = "skip_step"
    USE_FALLBACK = "use_fallback"
    ABORT = "abort"


@dataclass
class ExceededResponse:
    """Response to budget exceeded condition."""
    action: ExceededAction
    message: str
    partial_result: Optional[Any] = None
    skipped_steps: List[str] = None
    recommendations: List[str] = None


class BudgetExceededHandler:
    """
    Handles budget exceeded conditions gracefully.

    Ensures system can complete/finalize despite
    budget constraints.
    """

    def __init__(
        self,
        emergency_reserve: int = 5_000,
    ) -> None:
        """
        Initialize handler.

        Args:
            emergency_reserve: Tokens reserved for finalization
        """
        self.emergency_reserve = emergency_reserve

    def handle_exceeded(
        self,
        component: str,
        operation: str,
        tokens_needed: int,
        tokens_remaining: int,
        current_result: Any = None,
    ) -> ExceededResponse:
        """
        Handle a budget exceeded condition.

        Args:
            component: Component that triggered exceeded
            operation: Operation being attempted
            tokens_needed: Tokens the operation needs
            tokens_remaining: Tokens actually available
            current_result: Any partial result available

        Returns:
            ExceededResponse with handling instructions
        """
        # Check if we can use emergency reserve
        can_use_emergency = tokens_remaining >= self.emergency_reserve

        # Determine action based on component and availability
        if component == "reasoning":
            return self._handle_reasoning_exceeded(
                operation, tokens_needed, tokens_remaining,
                current_result, can_use_emergency
            )
        elif component == "memory":
            return self._handle_memory_exceeded(
                operation, tokens_needed, tokens_remaining,
                current_result
            )
        elif component == "validation":
            return self._handle_validation_exceeded(
                operation, tokens_needed, tokens_remaining,
                current_result
            )
        else:
            return self._handle_generic_exceeded(
                component, operation, tokens_needed,
                tokens_remaining, current_result
            )

    def _handle_reasoning_exceeded(
        self,
        operation: str,
        needed: int,
        remaining: int,
        result: Any,
        can_emergency: bool,
    ) -> ExceededResponse:
        """Handle exceeded during reasoning."""
        if result is not None:
            # We have a partial result, return it
            return ExceededResponse(
                action=ExceededAction.RETURN_PARTIAL,
                message=(
                    "Budget exceeded during reasoning. "
                    "Returning best result so far."
                ),
                partial_result=result,
                recommendations=[
                    "Review partial result for completeness",
                    "Start new session to continue reasoning",
                    "Consider breaking task into smaller pieces",
                ],
            )
        elif can_emergency:
            # Use emergency reserve for minimal response
            return ExceededResponse(
                action=ExceededAction.USE_FALLBACK,
                message=(
                    "Budget exceeded. Using emergency reserve "
                    "for basic response generation."
                ),
                recommendations=[
                    "Response may be less detailed",
                    "Start new session for full reasoning",
                ],
            )
        else:
            # Must abort
            return ExceededResponse(
                action=ExceededAction.ABORT,
                message=(
                    "Budget fully exhausted. Cannot continue "
                    "reasoning. Please start a new session."
                ),
                recommendations=[
                    "Review token consumption patterns",
                    "Consider using more efficient prompts",
                    "Break complex tasks into smaller sessions",
                ],
            )

    def _handle_memory_exceeded(
        self,
        operation: str,
        needed: int,
        remaining: int,
        result: Any,
    ) -> ExceededResponse:
        """Handle exceeded during memory operations."""
        return ExceededResponse(
            action=ExceededAction.SKIP_STEP,
            message=(
                "Budget exceeded during memory retrieval. "
                "Continuing without additional memory context."
            ),
            skipped_steps=["memory_retrieval", "memory_consolidation"],
            recommendations=[
                "Response may lack historical context",
                "Consider pre-loading important memories",
                "Review memory efficiency settings",
            ],
        )

    def _handle_validation_exceeded(
        self,
        operation: str,
        needed: int,
        remaining: int,
        result: Any,
    ) -> ExceededResponse:
        """Handle exceeded during validation."""
        return ExceededResponse(
            action=ExceededAction.USE_FALLBACK,
            message=(
                "Budget exceeded during validation. "
                "Using schema validation only (no LLM verification)."
            ),
            partial_result=result,
            recommendations=[
                "Output validated against schema only",
                "LLM quality check skipped",
                "Manually review output for correctness",
            ],
        )

    def _handle_generic_exceeded(
        self,
        component: str,
        operation: str,
        needed: int,
        remaining: int,
        result: Any,
    ) -> ExceededResponse:
        """Handle exceeded for other components."""
        if result is not None:
            return ExceededResponse(
                action=ExceededAction.RETURN_PARTIAL,
                message=f"Budget exceeded in {component}. Returning partial result.",
                partial_result=result,
            )
        else:
            return ExceededResponse(
                action=ExceededAction.ABORT,
                message=f"Budget exceeded in {component}. Cannot continue.",
                recommendations=["Start a new session to continue"],
            )


# Global handler instance
_handler: Optional[BudgetExceededHandler] = None


def get_exceeded_handler() -> BudgetExceededHandler:
    """Get or create exceeded handler."""
    global _handler
    if _handler is None:
        _handler = BudgetExceededHandler()
    return _handler
```

---

## 9. Forecasting and Estimation

### 9.1 Forecasting Methods

```
+-----------------------------------------------------------------------------+
|                         FORECASTING METHODS                                  |
+-----------------------------------------------------------------------------+

METHOD 1: Operation-Based Forecasting
-------------------------------------
Calculates expected total based on average tokens per operation.

Formula:
  forecast = current_used + (avg_tokens_per_op * remaining_operations)

Accuracy: High when operations are similar
Use when: Known number of remaining steps

Example:
  Current: 650 tokens after 5 operations (avg 130/op)
  Remaining: 3 operations expected
  Forecast: 650 + (130 * 3) = 1,040 tokens


METHOD 2: Time-Based Forecasting
--------------------------------
Calculates expected total based on consumption rate over time.

Formula:
  rate = current_used / elapsed_seconds
  forecast = current_used + (rate * remaining_seconds)

Accuracy: Moderate, depends on consistent activity
Use when: Known time budget for session

Example:
  Current: 650 tokens in 30 seconds (21.7 tokens/sec)
  Time remaining: 60 seconds
  Forecast: 650 + (21.7 * 60) = 1,950 tokens


METHOD 3: Component-Based Forecasting
-------------------------------------
Estimates based on component-specific patterns.

Formula:
  forecast = sum(component_forecasts)
  component_forecast = current_used + expected_additional

Accuracy: High when component patterns known
Use when: Detailed pipeline information available

Example:
  Routing: 50 done, expect 0 more = 50
  Memory: 150 done, expect 100 more = 250
  Reasoning: 450 done, expect 2000 more = 2,450
  Total forecast: 2,750 tokens


METHOD 4: Hybrid Forecasting
----------------------------
Combines multiple methods with weighting.

Formula:
  forecast = (w1 * op_forecast + w2 * time_forecast + w3 * comp_forecast) / (w1 + w2 + w3)

Accuracy: Highest overall
Use when: Multiple data sources available

Weights (configurable):
  - Operation-based: 0.4 (if ops known)
  - Time-based: 0.3 (always available)
  - Component-based: 0.3 (if details known)
```

### 9.2 Forecasting Implementation

```python
# sigil/telemetry/budget_forecasting.py

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Optional, List


@dataclass
class ForecastResult:
    """Result of budget forecasting."""
    forecast_total: int
    forecast_remaining: int
    will_exceed: bool
    confidence: str  # low, medium, high
    method: str
    details: Dict


class BudgetForecaster:
    """
    Forecasts token budget consumption.

    Supports multiple forecasting methods with
    confidence estimation.
    """

    def __init__(self) -> None:
        """Initialize forecaster."""
        self._history: List[Dict] = []

    def record_operation(
        self,
        component: str,
        tokens: int,
        timestamp: datetime = None,
    ) -> None:
        """Record an operation for forecasting."""
        self._history.append({
            "component": component,
            "tokens": tokens,
            "timestamp": timestamp or datetime.now(),
        })

    def forecast(
        self,
        current_used: int,
        total_budget: int,
        operations_remaining: int = None,
        time_remaining_seconds: float = None,
        component_estimates: Dict[str, int] = None,
    ) -> ForecastResult:
        """
        Generate budget forecast.

        Args:
            current_used: Tokens used so far
            total_budget: Total budget
            operations_remaining: Expected remaining ops
            time_remaining_seconds: Expected remaining time
            component_estimates: Per-component estimates

        Returns:
            ForecastResult with prediction
        """
        forecasts = []
        weights = []

        # Method 1: Operation-based
        if operations_remaining is not None and self._history:
            ops_forecast = self._forecast_by_operations(
                current_used, operations_remaining
            )
            if ops_forecast is not None:
                forecasts.append(ops_forecast)
                weights.append(0.4)

        # Method 2: Time-based
        if time_remaining_seconds is not None and self._history:
            time_forecast = self._forecast_by_time(
                current_used, time_remaining_seconds
            )
            if time_forecast is not None:
                forecasts.append(time_forecast)
                weights.append(0.3)

        # Method 3: Component-based
        if component_estimates:
            comp_forecast = self._forecast_by_components(
                current_used, component_estimates
            )
            forecasts.append(comp_forecast)
            weights.append(0.3)

        # Combine forecasts
        if not forecasts:
            # Fallback: assume 2x current usage
            forecast_total = current_used * 2
            method = "fallback"
            confidence = "low"
        elif len(forecasts) == 1:
            forecast_total = forecasts[0]
            method = "single"
            confidence = "medium"
        else:
            # Weighted average
            total_weight = sum(weights[:len(forecasts)])
            forecast_total = int(
                sum(f * w for f, w in zip(forecasts, weights))
                / total_weight
            )
            method = "hybrid"
            confidence = "high"

        return ForecastResult(
            forecast_total=forecast_total,
            forecast_remaining=max(0, total_budget - forecast_total),
            will_exceed=forecast_total > total_budget,
            confidence=confidence,
            method=method,
            details={
                "forecasts": forecasts,
                "weights": weights[:len(forecasts)],
                "current_used": current_used,
                "total_budget": total_budget,
            },
        )

    def _forecast_by_operations(
        self,
        current_used: int,
        ops_remaining: int,
    ) -> Optional[int]:
        """Forecast based on operations."""
        if not self._history:
            return None

        avg_tokens = sum(h["tokens"] for h in self._history) / len(self._history)
        return int(current_used + (avg_tokens * ops_remaining))

    def _forecast_by_time(
        self,
        current_used: int,
        time_remaining: float,
    ) -> Optional[int]:
        """Forecast based on time."""
        if not self._history or len(self._history) < 2:
            return None

        # Calculate elapsed time
        first = self._history[0]["timestamp"]
        last = self._history[-1]["timestamp"]
        elapsed = (last - first).total_seconds()

        if elapsed <= 0:
            return None

        rate = current_used / elapsed
        return int(current_used + (rate * time_remaining))

    def _forecast_by_components(
        self,
        current_used: int,
        estimates: Dict[str, int],
    ) -> int:
        """Forecast based on component estimates."""
        additional = sum(estimates.values())
        return current_used + additional

    def estimate_operations_remaining(
        self,
        current_used: int,
        total_budget: int,
    ) -> int:
        """
        Estimate how many more operations fit in budget.

        Args:
            current_used: Tokens used so far
            total_budget: Total budget

        Returns:
            Estimated remaining operations
        """
        if not self._history:
            return 0

        avg_tokens = sum(h["tokens"] for h in self._history) / len(self._history)
        remaining = total_budget - current_used

        if avg_tokens <= 0:
            return 0

        return int(remaining / avg_tokens)
```

---

## 10. Budget Optimization

### 10.1 Optimization Strategies

```
+-----------------------------------------------------------------------------+
|                       BUDGET OPTIMIZATION STRATEGIES                         |
+-----------------------------------------------------------------------------+

STRATEGY 1: Prompt Compression
------------------------------
Reduce tokens in system prompts and context.

Techniques:
- Remove redundant instructions
- Use shorter variable names
- Compress examples
- Lazy load context

Savings: 10-30% of prompt tokens


STRATEGY 2: Memory Efficiency
-----------------------------
Optimize memory retrieval and storage.

Techniques:
- Limit RAG result count
- Summarize long memories
- Use embedding-only search first
- Cache frequent queries

Savings: 20-50% of memory tokens


STRATEGY 3: Reasoning Strategy Selection
----------------------------------------
Choose appropriate reasoning for complexity.

Mapping:
- Simple: Direct (100-500 tokens)
- Moderate: Chain-of-thought (500-2K)
- Complex: Tree-of-thoughts (2K-10K)
- Critical: MCTS (10K-50K)

Savings: 50-80% when right strategy selected


STRATEGY 4: Early Termination
-----------------------------
Stop processing when sufficient quality reached.

Techniques:
- Quality thresholds for reasoning
- Early exit from iterations
- Sufficient confidence detection

Savings: 20-40% on complex tasks


STRATEGY 5: Caching
-------------------
Cache frequently used results.

Cacheable:
- Routing decisions
- Plan templates
- Common memory queries
- Validation schemas

Savings: 10-20% overall
```

### 10.2 Optimization Recommendations

```python
# sigil/telemetry/budget_optimizer.py

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class OptimizationRecommendation:
    """A budget optimization recommendation."""
    category: str
    description: str
    estimated_savings: str
    implementation: str
    priority: str  # high, medium, low


class BudgetOptimizer:
    """
    Analyzes budget usage and provides optimization recommendations.
    """

    def analyze(self, budget_status: Dict) -> List[OptimizationRecommendation]:
        """
        Analyze budget status and generate recommendations.

        Args:
            budget_status: Status from TokenTracker

        Returns:
            List of optimization recommendations
        """
        recommendations = []

        components = budget_status.get("components", {})

        # Check component-specific issues
        for name, comp in components.items():
            recs = self._analyze_component(name, comp)
            recommendations.extend(recs)

        # Check overall patterns
        overall_recs = self._analyze_overall(budget_status)
        recommendations.extend(overall_recs)

        # Sort by priority
        priority_order = {"high": 0, "medium": 1, "low": 2}
        recommendations.sort(key=lambda r: priority_order.get(r.priority, 2))

        return recommendations

    def _analyze_component(
        self,
        name: str,
        status: Dict,
    ) -> List[OptimizationRecommendation]:
        """Analyze a single component."""
        recs = []
        utilization = status.get("utilization", 0)
        avg_per_op = status.get("used", 0) / max(1, status.get("operations", 1))

        if name == "reasoning" and utilization > 80:
            recs.append(OptimizationRecommendation(
                category="reasoning",
                description="High reasoning token usage detected",
                estimated_savings="30-50%",
                implementation=(
                    "Consider using simpler reasoning strategy "
                    "(chain-of-thought vs tree-of-thoughts) for "
                    "moderate complexity tasks"
                ),
                priority="high",
            ))

        if name == "memory" and avg_per_op > 1000:
            recs.append(OptimizationRecommendation(
                category="memory",
                description="High tokens per memory operation",
                estimated_savings="20-40%",
                implementation=(
                    "Limit RAG results, use embedding-only search, "
                    "or summarize long memory content"
                ),
                priority="medium",
            ))

        if name == "routing" and avg_per_op > 200:
            recs.append(OptimizationRecommendation(
                category="routing",
                description="High tokens for routing decisions",
                estimated_savings="50-70%",
                implementation=(
                    "Cache common routing decisions, "
                    "simplify routing prompt"
                ),
                priority="low",
            ))

        return recs

    def _analyze_overall(self, status: Dict) -> List[OptimizationRecommendation]:
        """Analyze overall budget patterns."""
        recs = []

        utilization = status.get("utilization", 0)

        if utilization > 90:
            recs.append(OptimizationRecommendation(
                category="overall",
                description="Session approaching budget limit",
                estimated_savings="N/A",
                implementation=(
                    "Consider breaking large tasks into multiple sessions, "
                    "or increasing total budget allocation"
                ),
                priority="high",
            ))

        # Check for unbalanced usage
        components = status.get("components", {})
        used_ratios = {
            name: comp.get("used", 0) / max(1, comp.get("allocated", 1))
            for name, comp in components.items()
        }

        max_ratio = max(used_ratios.values()) if used_ratios else 0
        min_ratio = min(used_ratios.values()) if used_ratios else 0

        if max_ratio > 0.9 and min_ratio < 0.1:
            recs.append(OptimizationRecommendation(
                category="allocation",
                description="Unbalanced component budget usage",
                estimated_savings="10-20%",
                implementation=(
                    "Reallocate budget from underutilized components "
                    "to high-usage components"
                ),
                priority="medium",
            ))

        return recs
```

---

## 11. Multi-Session Budget Management

### 11.1 Cross-Session Budgeting

```
+-----------------------------------------------------------------------------+
|                    MULTI-SESSION BUDGET MANAGEMENT                           |
+-----------------------------------------------------------------------------+

SCENARIO: Long-running workflows spanning multiple sessions

Architecture:
+-------------------+     +-------------------+     +-------------------+
|   Session 1       |     |   Session 2       |     |   Session 3       |
|   Budget: 256K    |     |   Budget: 256K    |     |   Budget: 256K    |
|   Used: 240K      |     |   Used: 180K      |     |   Used: 150K      |
+-------------------+     +-------------------+     +-------------------+
         |                         |                         |
         v                         v                         v
+-----------------------------------------------------------------------+
|                      WORKFLOW BUDGET TRACKER                           |
|                                                                        |
|   Total Allocated: 768,000 tokens (3 sessions)                         |
|   Total Used: 570,000 tokens                                           |
|   Total Remaining: 198,000 tokens                                      |
|                                                                        |
+-----------------------------------------------------------------------+


BUDGET CARRYOVER STRATEGIES:

1. Fixed Per-Session
   - Each session gets fixed allocation
   - Unused tokens do not carry over
   - Simple but potentially wasteful

2. Pooled Budget
   - Total budget shared across sessions
   - Unused tokens carry forward
   - Requires cross-session tracking

3. Adaptive Allocation
   - Analyze previous session usage
   - Adjust next session allocation
   - Most efficient but complex
```

### 11.2 Workflow Budget Implementation

```python
# sigil/telemetry/workflow_budget.py

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional
from pathlib import Path
import json


@dataclass
class SessionSummary:
    """Summary of a completed session's budget usage."""
    session_id: str
    total_budget: int
    total_used: int
    components: Dict[str, int]
    started_at: datetime
    completed_at: datetime


@dataclass
class WorkflowBudget:
    """Budget management across multiple sessions."""
    workflow_id: str
    total_budget: int
    sessions: List[SessionSummary] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)

    @property
    def total_used(self) -> int:
        return sum(s.total_used for s in self.sessions)

    @property
    def total_remaining(self) -> int:
        return max(0, self.total_budget - self.total_used)

    @property
    def sessions_count(self) -> int:
        return len(self.sessions)

    @property
    def avg_per_session(self) -> float:
        if not self.sessions:
            return 0.0
        return self.total_used / len(self.sessions)


class WorkflowBudgetManager:
    """
    Manages budgets across multiple sessions in a workflow.

    Supports:
    - Cross-session budget tracking
    - Budget carryover
    - Session allocation recommendations
    """

    def __init__(self, storage_path: Path = None) -> None:
        """
        Initialize workflow budget manager.

        Args:
            storage_path: Path for persistent storage
        """
        self.storage_path = storage_path or Path.home() / ".sigil" / "workflows"
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self._workflows: Dict[str, WorkflowBudget] = {}

    def create_workflow(
        self,
        workflow_id: str,
        total_budget: int,
    ) -> WorkflowBudget:
        """Create a new workflow budget."""
        workflow = WorkflowBudget(
            workflow_id=workflow_id,
            total_budget=total_budget,
        )
        self._workflows[workflow_id] = workflow
        self._save_workflow(workflow)
        return workflow

    def get_workflow(self, workflow_id: str) -> Optional[WorkflowBudget]:
        """Get a workflow by ID."""
        if workflow_id in self._workflows:
            return self._workflows[workflow_id]

        # Try loading from disk
        return self._load_workflow(workflow_id)

    def record_session(
        self,
        workflow_id: str,
        session_summary: SessionSummary,
    ) -> None:
        """Record a completed session."""
        workflow = self.get_workflow(workflow_id)
        if workflow is None:
            raise ValueError(f"Workflow {workflow_id} not found")

        workflow.sessions.append(session_summary)
        self._save_workflow(workflow)

    def recommend_session_budget(
        self,
        workflow_id: str,
        operations_expected: int = None,
    ) -> int:
        """
        Recommend budget for next session.

        Args:
            workflow_id: Workflow identifier
            operations_expected: Expected operations in session

        Returns:
            Recommended token budget
        """
        workflow = self.get_workflow(workflow_id)
        if workflow is None:
            return 256_000  # Default

        remaining = workflow.total_remaining

        if not workflow.sessions:
            # First session: use default or portion of remaining
            return min(256_000, remaining)

        # Based on previous sessions
        avg_used = workflow.avg_per_session

        if operations_expected:
            # Scale by expected operations vs average
            avg_ops = sum(
                len(s.components) for s in workflow.sessions
            ) / len(workflow.sessions)

            if avg_ops > 0:
                scale = operations_expected / avg_ops
                recommended = int(avg_used * scale * 1.2)  # 20% buffer
            else:
                recommended = int(avg_used * 1.2)
        else:
            # Use average plus buffer
            recommended = int(avg_used * 1.2)

        # Cap at remaining budget
        return min(recommended, remaining, 256_000)

    def get_workflow_status(self, workflow_id: str) -> Dict:
        """Get complete workflow status."""
        workflow = self.get_workflow(workflow_id)
        if workflow is None:
            return {"error": "Workflow not found"}

        return {
            "workflow_id": workflow.workflow_id,
            "total_budget": workflow.total_budget,
            "total_used": workflow.total_used,
            "total_remaining": workflow.total_remaining,
            "sessions_count": workflow.sessions_count,
            "avg_per_session": workflow.avg_per_session,
            "sessions": [
                {
                    "session_id": s.session_id,
                    "used": s.total_used,
                    "started": s.started_at.isoformat(),
                    "completed": s.completed_at.isoformat(),
                }
                for s in workflow.sessions
            ],
        }

    def _save_workflow(self, workflow: WorkflowBudget) -> None:
        """Save workflow to disk."""
        path = self.storage_path / f"{workflow.workflow_id}.json"
        data = {
            "workflow_id": workflow.workflow_id,
            "total_budget": workflow.total_budget,
            "created_at": workflow.created_at.isoformat(),
            "sessions": [
                {
                    "session_id": s.session_id,
                    "total_budget": s.total_budget,
                    "total_used": s.total_used,
                    "components": s.components,
                    "started_at": s.started_at.isoformat(),
                    "completed_at": s.completed_at.isoformat(),
                }
                for s in workflow.sessions
            ],
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

    def _load_workflow(self, workflow_id: str) -> Optional[WorkflowBudget]:
        """Load workflow from disk."""
        path = self.storage_path / f"{workflow_id}.json"
        if not path.exists():
            return None

        with open(path, 'r') as f:
            data = json.load(f)

        workflow = WorkflowBudget(
            workflow_id=data["workflow_id"],
            total_budget=data["total_budget"],
            created_at=datetime.fromisoformat(data["created_at"]),
        )

        for s in data.get("sessions", []):
            workflow.sessions.append(SessionSummary(
                session_id=s["session_id"],
                total_budget=s["total_budget"],
                total_used=s["total_used"],
                components=s["components"],
                started_at=datetime.fromisoformat(s["started_at"]),
                completed_at=datetime.fromisoformat(s["completed_at"]),
            ))

        self._workflows[workflow_id] = workflow
        return workflow
```

---

## 12. API Reference

### 12.1 TokenTracker API

```python
# Complete API Reference for TokenTracker

class TokenTracker:
    """Thread-safe token tracking with comprehensive metrics."""

    def __init__(
        self,
        session_id: str,
        total_budget: int = 256_000,
        allocations: Dict[str, int] = None,
        logger: "ExecutionLogger" = None,
    ) -> None:
        """
        Initialize the token tracker.

        Parameters:
            session_id: Unique session identifier
            total_budget: Total token budget (default 256,000)
            allocations: Component allocations (uses defaults if None)
            logger: Optional execution logger for monitoring
        """

    def track(
        self,
        component: str,
        tokens: int,
        operation_type: str = "unknown",
        metadata: Dict = None,
    ) -> bool:
        """
        Track token consumption for an operation.

        Parameters:
            component: Component name (routing, memory, etc.)
            tokens: Tokens consumed
            operation_type: Type of operation
            metadata: Additional operation metadata

        Returns:
            True if consumption succeeded (within budget)
            False if budget exceeded

        Raises:
            None (fails silently, returns False)
        """

    def can_consume(
        self,
        tokens: int,
        component: str = None,
    ) -> bool:
        """
        Check if tokens can be consumed without exceeding budget.

        Parameters:
            tokens: Tokens to check
            component: Optional component to check against

        Returns:
            True if consumption would succeed
        """

    def get_status(self) -> Dict:
        """
        Get complete budget status.

        Returns:
            {
                "session_id": str,
                "total": int,
                "used": int,
                "remaining": int,
                "utilization": float,
                "warning_level": str,
                "operations_count": int,
                "components": {
                    "component_name": {
                        "allocated": int,
                        "used": int,
                        "remaining": int,
                        "utilization": float,
                        "operations": int,
                    }
                }
            }
        """

    def get_component_status(self, component: str) -> Optional[Dict]:
        """
        Get status for a specific component.

        Parameters:
            component: Component name

        Returns:
            Component status dict or None if not found
        """

    def forecast(
        self,
        operations_remaining: int = None,
        time_remaining_seconds: float = None,
    ) -> Dict:
        """
        Forecast token usage based on current consumption.

        Parameters:
            operations_remaining: Expected remaining operations
            time_remaining_seconds: Expected remaining time

        Returns:
            {
                "forecast_total": int,
                "forecast_remaining": int,
                "will_exceed": bool,
                "avg_tokens_per_operation": float,
                "tokens_per_second": float,
                "confidence": str,
            }
        """

    def on_warning(self, callback: Callable) -> None:
        """
        Register a warning callback.

        Parameters:
            callback: Function(level, used, total) to call on warnings
        """
```

### 12.2 BudgetDisplay API

```python
class BudgetDisplay:
    """Renders budget status in various formats."""

    def render(
        self,
        status: Dict,
        style: DisplayStyle = DisplayStyle.BREAKDOWN,
    ) -> str:
        """
        Render budget status in specified style.

        Parameters:
            status: Budget status dictionary from TokenTracker
            style: Display style (COMPACT, BREAKDOWN, DASHBOARD, MINIMAL)

        Returns:
            Formatted string for display
        """
```

### 12.3 BudgetAlertManager API

```python
class BudgetAlertManager:
    """Manages budget alerts and notifications."""

    def __init__(
        self,
        warning_threshold: float = 0.50,
        critical_threshold: float = 0.80,
        exceeded_threshold: float = 0.95,
    ) -> None:
        """Initialize alert manager with thresholds."""

    def add_handler(
        self,
        handler: Callable[[BudgetAlert], None],
    ) -> None:
        """Add an alert handler callback."""

    def check_and_alert(
        self,
        tokens_used: int,
        tokens_total: int,
        metadata: Dict = None,
    ) -> Optional[BudgetAlert]:
        """Check budget and generate alert if threshold crossed."""

    def get_history(self, limit: int = 10) -> List[BudgetAlert]:
        """Get recent alert history."""
```

---

## Document Revision History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0.0 | 2026-01-11 | Systems Architecture | Initial release |

---

*End of CLI Token Budgeting Document*
