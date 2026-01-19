# API Contract: Phase 5 Planning & Reasoning System

## Overview

This document defines the comprehensive system architecture and API contract for Sigil v2's Phase 5 Planning & Reasoning System. This phase adds two complex, interdependent subsystems that enable goal decomposition, multi-strategy reasoning, and intelligent task execution.

**Key Capabilities:**
- **Planning Subsystem**: Decompose high-level goals into executable DAG-structured plans
- **Reasoning Subsystem**: Five strategies (Direct, CoT, ToT, ReAct, MCTS) selected by complexity
- **Cross-Phase Integration**: Deep integration with Routing (Phase 3), Memory (Phase 4), and Contracts (Phase 6)
- **Token Budget Management**: Coordinated budget allocation across planning and reasoning operations
- **Event-Sourced Execution**: Full audit trail with replay capability for both subsystems

**Dependencies:**
- Phase 3: Router, ComplexityAssessor, TokenTracker, EventStore
- Phase 4: MemoryManager, retrieval system
- Phase 6: Contract verification (forward reference)

---

## Table of Contents

1. [Design Principles](#1-design-principles)
2. [Module Structure](#2-module-structure)
3. [System Integration Architecture](#3-system-integration-architecture)
4. [Planning Subsystem API](#4-planning-subsystem-api)
5. [Reasoning Subsystem API](#5-reasoning-subsystem-api)
6. [Token Budget Management](#6-token-budget-management)
7. [Event Contract](#7-event-contract)
8. [Concurrency and Parallelization](#8-concurrency-and-parallelization)
9. [Performance Optimization](#9-performance-optimization)
10. [Error Handling and Resilience](#10-error-handling-and-resilience)
11. [Monitoring and Observability](#11-monitoring-and-observability)
12. [Data Schema Contracts](#12-data-schema-contracts)
13. [Integration Examples](#13-integration-examples)
14. [Implementation Guidance](#14-implementation-guidance)

---

## 1. Design Principles

### 1.1 Core Philosophy

1. **Complexity-Adaptive**: Select reasoning depth based on task complexity (0.0-1.0 scale from ComplexityAssessor)
2. **Fail-Safe Execution**: Always produce a result - fallback to simpler strategies rather than fail completely
3. **Token-Conscious**: Every operation budgets tokens; graceful degradation when budget exhausted
4. **Traceable**: Full event trail from goal to result for debugging and optimization
5. **Memory-Grounded**: All reasoning can access memory for fact-grounding and context
6. **Resumable**: Checkpointed execution allows resumption after interruption

### 1.2 Integration Philosophy

```
Phase 5 is the "brain" that orchestrates intelligent behavior:
- Phase 3 (Routing) determines WHEN to engage planning/reasoning
- Phase 4 (Memory) provides CONTEXT for planning/reasoning
- Phase 5 (Planning/Reasoning) determines HOW to achieve goals
- Phase 6 (Contracts) validates WHAT was produced
```

---

## 2. Module Structure

```
sigil/planning/
|-- __init__.py                 # Module exports: Planner, PlanExecutor, Plan, PlanStep
|-- planner.py                  # Planner: goal decomposition via LLM
|-- executor.py                 # PlanExecutor: DAG execution with concurrency
|-- optimizer.py                # PlanOptimizer: plan simplification and caching
|-- schemas.py                  # Plan, PlanStep, PlanResult dataclasses

sigil/reasoning/
|-- __init__.py                 # Module exports: ReasoningManager, strategies
|-- manager.py                  # ReasoningManager: strategy selection and orchestration
|-- complexity.py               # ComplexityScorer: detailed complexity analysis
|-- cache.py                    # StrategyCache: caching for complexity and results
|-- strategies/
|   |-- __init__.py             # Strategy exports
|   |-- base.py                 # BaseStrategy ABC with execute() contract
|   |-- direct.py               # DirectStrategy: simple single-call reasoning
|   |-- chain_of_thought.py     # CoTStrategy: step-by-step reasoning
|   |-- tree_of_thoughts.py     # ToTStrategy: multi-path exploration
|   |-- react.py                # ReActStrategy: thought-action-observation loop
|   |-- mcts.py                 # MCTSStrategy: Monte Carlo Tree Search

sigil/orchestration/
|-- __init__.py                 # Module exports: Orchestrator
|-- orchestrator.py             # Orchestrator: coordinates planning + reasoning + memory
```

---

## 3. System Integration Architecture

### 3.1 High-Level Data Flow

```
User Request
    |
    v
+-------------------+
|  Router (Phase 3) |
|  - Intent classify |
|  - Complexity assess|
+--------+----------+
         |
         v
    [complexity score]
         |
    +----+----+
    |         |
    v         v
Simple    Complex
(< 0.5)   (>= 0.5)
    |         |
    v         v
+--------+ +------------------+
|Reasoning| |  Planner         |
|Manager  | |  (decompose goal)|
+----+----+ +--------+---------+
     |               |
     v               v
 [Direct/CoT]   [Plan with steps]
     |               |
     |        +------+------+
     |        |             |
     |        v             v
     |   [Parallel     [Sequential
     |    steps]        steps]
     |        |             |
     |        +------+------+
     |               |
     |               v
     |        +------+------+
     |        | Step executes|
     |        | via Reasoning|
     |        | Manager      |
     |        +------+------+
     |               |
     +-------+-------+
             |
             v
    +--------+--------+
    | Memory Manager   |
    | - Retrieve context|
    | - Store results  |
    +--------+--------+
             |
             v
    [Final Result + Events]
```

### 3.2 Integration with Phase 3 (Routing)

#### 3.2.1 Router Integration Points

```python
# Router determines when to trigger planning
class Router:
    def route(self, message: str) -> RouteDecision:
        """
        Integration points:
        1. ComplexityAssessor.assess() -> complexity score
        2. complexity > 0.5 AND use_planning=True -> triggers Planner
        3. complexity <= 0.5 -> direct to ReasoningManager
        """
```

#### 3.2.2 Complexity-to-Strategy Mapping

The Router's ComplexityAssessor output directly feeds into ReasoningManager's strategy selection:

```python
class RouteDecision:
    complexity: float  # 0.0 - 1.0
    use_planning: bool  # True if complexity > 0.5 AND feature enabled
    suggested_strategy: Optional[str]  # Router can suggest strategy override
```

#### 3.2.3 Strategy Override

The Router can override strategy selection for specific intents:

```python
INTENT_STRATEGY_OVERRIDES = {
    Intent.SYSTEM_COMMAND: "direct",  # Always use Direct for /commands
    Intent.QUERY_MEMORY: "cot",  # Use CoT for memory queries
}
```

### 3.3 Integration with Phase 4 (Memory)

#### 3.3.1 Memory Retrieval for Planning

```python
async def plan_with_context(
    goal: str,
    memory_manager: MemoryManager,
) -> Plan:
    """
    1. Retrieve relevant memory for goal context
    2. Include memory in plan generation prompt
    3. Memory informs step decomposition
    """
    # Retrieve relevant memories
    context = await memory_manager.retrieve(
        query=goal,
        mode=RetrievalMode.HYBRID,
        k=10,
    )

    # Include in plan generation
    plan = await planner.generate(
        goal=goal,
        context=format_memory_context(context.items),
        available_tools=tool_registry.list_tools(),
    )
    return plan
```

#### 3.3.2 Memory Retrieval for Reasoning

```python
async def reason_with_context(
    task: str,
    strategy: BaseStrategy,
    memory_manager: MemoryManager,
) -> StrategyResult:
    """
    1. Retrieve relevant facts for grounding
    2. Include facts in reasoning prompt
    3. Prevents hallucination with fact-checking
    """
    facts = await memory_manager.retrieve(
        query=task,
        mode=RetrievalMode.RAG,  # Fast for reasoning
        k=5,
    )

    result = await strategy.execute(
        task=task,
        context=ReasoningContext(
            facts=[f.content for f in facts.items],
            fact_sources=[f.source_resource_id for f in facts.items],
        ),
    )
    return result
```

#### 3.3.3 Plan Execution Results to Memory

```python
async def store_execution_results(
    plan: Plan,
    results: list[StepResult],
    memory_manager: MemoryManager,
    session_id: str,
) -> Resource:
    """
    Store plan execution as a resource for future learning.
    """
    execution_content = format_execution_log(plan, results)

    resource = await memory_manager.store_resource(
        content=execution_content,
        resource_type="plan_execution",
        metadata={
            "plan_id": plan.plan_id,
            "goal": plan.goal,
            "success": all(r.success for r in results),
            "total_tokens": sum(r.tokens_used for r in results),
            "session_id": session_id,
        },
    )

    # Extract insights from execution
    await memory_manager.extract_and_store(
        resource_id=resource.resource_id,
        category_hint="execution_patterns",
    )

    return resource
```

### 3.4 Integration with Phase 6 (Contracts)

#### 3.4.1 Plan Steps with Contract Specifications

```python
@dataclass
class PlanStep:
    step_id: str
    description: str
    step_type: StepType  # tool_call, reasoning, parallel_group
    dependencies: list[str]  # step_ids that must complete first
    contract_spec: Optional[ContractSpec] = None  # Output requirements

@dataclass
class ContractSpec:
    """Embedded contract for step output verification."""
    required_fields: list[str]
    field_types: dict[str, type]
    validation_rules: list[str]  # e.g., "score >= 0 and score <= 100"
```

#### 3.4.2 Contract Verification During Execution

```python
async def execute_step_with_contract(
    step: PlanStep,
    context: ExecutionContext,
    contract_executor: ContractExecutor,
) -> StepResult:
    """
    Execute step and verify output against contract if specified.
    """
    result = await execute_step(step, context)

    if step.contract_spec:
        validation = contract_executor.validate(
            output=result.output,
            contract=step.contract_spec,
        )

        if not validation.passed:
            # Trigger retry or fallback
            if result.retry_count < 3:
                return await retry_step_with_feedback(
                    step=step,
                    context=context,
                    errors=validation.errors,
                )
            else:
                result.status = StepStatus.FAILED_CONTRACT
                result.contract_errors = validation.errors

    return result
```

#### 3.4.3 Reasoning Result Contracts

```python
@dataclass
class StrategyResult:
    """Result from reasoning strategy execution."""
    answer: str
    reasoning_trace: list[str]
    confidence: float
    tokens_used: int
    strategy_name: str

    # Contract verification metadata
    meets_contract: bool = True
    contract_violations: list[str] = field(default_factory=list)
```

---

## 4. Planning Subsystem API

### 4.1 Planner Class

```python
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
```

#### Constructor

```python
def __init__(
    self,
    model: str = "anthropic:claude-sonnet-4-20250514",
    tool_registry: ToolRegistry | None = None,
    memory_manager: MemoryManager | None = None,
    cache_ttl_hours: int = 24,
    max_steps: int = 20,
) -> None:
```

**Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `model` | `str` | No | `"anthropic:claude-sonnet-4-20250514"` | LLM for plan generation |
| `tool_registry` | `ToolRegistry` | No | Auto-created | Available tools |
| `memory_manager` | `MemoryManager` | No | None | Memory for context |
| `cache_ttl_hours` | `int` | No | 24 | Plan cache TTL |
| `max_steps` | `int` | No | 20 | Maximum steps per plan |

#### Method: `generate`

```python
async def generate(
    self,
    goal: str,
    context: str | None = None,
    constraints: PlanConstraints | None = None,
    session_id: str | None = None,
) -> Plan:
    """Generate an execution plan for a goal.

    Args:
        goal: High-level goal to achieve (1-1000 chars)
        context: Optional context from memory or user
        constraints: Optional execution constraints
        session_id: Session ID for event tracking

    Returns:
        Plan with DAG structure of steps

    Raises:
        PlanGenerationError: If plan generation fails
        InvalidGoalError: If goal is empty or too vague

    Example:
        >>> plan = await planner.generate(
        ...     goal="Qualify lead John from Acme Corp",
        ...     context="Previous call notes: interested in enterprise plan"
        ... )
        >>> print(f"Generated {len(plan.steps)} steps")
    """
```

**Return Value:**

```python
Plan(
    plan_id="plan_550e8400-e29b-41d4-a716-446655440000",
    goal="Qualify lead John from Acme Corp",
    steps=[
        PlanStep(
            step_id="step_1",
            description="Retrieve existing information about John from memory",
            step_type=StepType.REASONING,
            dependencies=[],
            estimated_tokens=200,
        ),
        PlanStep(
            step_id="step_2",
            description="Identify BANT gaps from available information",
            step_type=StepType.REASONING,
            dependencies=["step_1"],
            estimated_tokens=300,
        ),
        # ... more steps
    ],
    metadata=PlanMetadata(
        complexity=0.72,
        estimated_total_tokens=1500,
        estimated_duration_seconds=15,
        parallel_groups=2,
    ),
    created_at=datetime(2026, 1, 11, 10, 30, 0),
)
```

**Events Emitted:**
- `PlanCreatedEvent(plan_id, goal, step_count)`

#### Method: `validate_plan`

```python
def validate_plan(self, plan: Plan) -> ValidationResult:
    """Validate plan structure and dependencies.

    Checks:
    - DAG is acyclic (no circular dependencies)
    - All dependencies reference valid step_ids
    - Step types are valid
    - Estimated tokens within budget

    Returns:
        ValidationResult with is_valid and any errors
    """
```

#### Method: `optimize_plan`

```python
async def optimize_plan(
    self,
    plan: Plan,
    optimization_goals: list[str] = ["minimize_tokens", "maximize_parallelism"],
) -> Plan:
    """Optimize plan structure.

    Optimizations:
    - Merge sequential reasoning steps when possible
    - Identify additional parallelization opportunities
    - Remove redundant steps
    - Estimate more accurate token costs
    """
```

### 4.2 PlanExecutor Class

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
```

#### Constructor

```python
def __init__(
    self,
    reasoning_manager: ReasoningManager | None = None,
    tool_executor: ToolExecutor | None = None,
    memory_manager: MemoryManager | None = None,
    max_concurrent: int = 3,
    max_retries: int = 3,
    checkpoint_dir: str = "outputs/checkpoints",
) -> None:
```

**Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `reasoning_manager` | `ReasoningManager` | No | Auto-created | For reasoning steps |
| `tool_executor` | `ToolExecutor` | No | Auto-created | For tool call steps |
| `memory_manager` | `MemoryManager` | No | None | For context and storage |
| `max_concurrent` | `int` | No | 3 | Max parallel steps |
| `max_retries` | `int` | No | 3 | Retries per step |
| `checkpoint_dir` | `str` | No | `"outputs/checkpoints"` | Checkpoint storage |

#### Method: `execute`

```python
async def execute(
    self,
    plan: Plan,
    session_id: str,
    token_budget: TokenBudget | None = None,
    context: ExecutionContext | None = None,
) -> ExecutionResult:
    """Execute a plan to completion.

    Args:
        plan: Plan to execute
        session_id: Session ID for tracking
        token_budget: Optional token budget constraint
        context: Optional execution context

    Returns:
        ExecutionResult with all step results

    Raises:
        PlanExecutionError: If unrecoverable error occurs
        TokenBudgetExceeded: If budget exhausted

    Example:
        >>> result = await executor.execute(plan, session_id="sess-123")
        >>> print(f"Completed: {result.success}")
        >>> print(f"Total tokens: {result.total_tokens}")
    """
```

**Return Value:**

```python
ExecutionResult(
    plan_id="plan_550e8400...",
    success=True,
    step_results=[
        StepResult(
            step_id="step_1",
            status=StepStatus.COMPLETED,
            output="Retrieved: John is CEO, interested in enterprise...",
            tokens_used=180,
            duration_ms=2100,
            retries=0,
        ),
        # ... more results
    ],
    total_tokens=1450,
    total_duration_ms=12500,
    final_output="Lead qualified with score 75...",
)
```

**Events Emitted:**
- `PlanExecutionStartedEvent(plan_id)`
- `PlanStepStartedEvent(plan_id, step_id)` per step
- `PlanStepCompletedEvent(plan_id, step_id, result)` per step
- `PlanStepFailedEvent(plan_id, step_id, error, retry_count)` on failure
- `PlanCompletedEvent(plan_id, total_tokens)`

#### Method: `resume`

```python
async def resume(
    self,
    plan_id: str,
    session_id: str,
) -> ExecutionResult:
    """Resume execution from last checkpoint.

    Args:
        plan_id: ID of plan to resume
        session_id: Session ID for tracking

    Returns:
        ExecutionResult from resumed execution

    Events Emitted:
        - PlanResumedEvent(plan_id, resuming_from)
    """
```

#### Method: `pause`

```python
async def pause(self, plan_id: str) -> bool:
    """Pause execution and create checkpoint.

    Args:
        plan_id: ID of plan to pause

    Returns:
        True if paused successfully

    Events Emitted:
        - PlanPausedEvent(plan_id, last_completed_step)
    """
```

### 4.3 Plan Data Structures

```python
class StepType(str, Enum):
    """Types of plan steps."""
    TOOL_CALL = "tool_call"          # Execute a tool
    REASONING = "reasoning"          # Use reasoning strategy
    PARALLEL_GROUP = "parallel_group"  # Group of parallel steps
    CONDITIONAL = "conditional"      # Conditional branching


@dataclass
class PlanStep:
    """A single step in an execution plan.

    Attributes:
        step_id: Unique identifier (format: step_{index})
        description: Human-readable step description
        step_type: Type of step execution
        dependencies: Step IDs that must complete first
        tool_name: Tool to call (if step_type=TOOL_CALL)
        tool_args: Arguments for tool (if step_type=TOOL_CALL)
        reasoning_task: Task description (if step_type=REASONING)
        strategy_hint: Suggested reasoning strategy
        contract_spec: Optional output contract
        estimated_tokens: Estimated token cost
        timeout_seconds: Step timeout (default 60)
    """
    step_id: str
    description: str
    step_type: StepType
    dependencies: list[str] = field(default_factory=list)
    tool_name: str | None = None
    tool_args: dict[str, Any] | None = None
    reasoning_task: str | None = None
    strategy_hint: str | None = None
    contract_spec: ContractSpec | None = None
    estimated_tokens: int = 500
    timeout_seconds: int = 60


@dataclass
class Plan:
    """Complete execution plan for a goal.

    Attributes:
        plan_id: Unique identifier (format: plan_{uuid})
        goal: Original goal that generated this plan
        steps: Ordered list of plan steps
        metadata: Plan metadata and estimates
        created_at: Creation timestamp
        is_immutable: True after creation (plans are immutable)
    """
    plan_id: str
    goal: str
    steps: list[PlanStep]
    metadata: PlanMetadata
    created_at: datetime
    is_immutable: bool = True

    def get_dependency_graph(self) -> dict[str, list[str]]:
        """Get step dependencies as adjacency list."""

    def get_execution_order(self) -> list[list[str]]:
        """Get topologically sorted execution order.

        Returns list of step ID lists, where each inner list
        can be executed in parallel.
        """

    def get_step(self, step_id: str) -> PlanStep | None:
        """Get step by ID."""


@dataclass
class PlanMetadata:
    """Metadata about plan complexity and estimates."""
    complexity: float  # 0.0 - 1.0
    estimated_total_tokens: int
    estimated_duration_seconds: float
    parallel_groups: int  # Number of parallelizable groups
    tool_calls: int  # Number of tool call steps
    reasoning_steps: int  # Number of reasoning steps


@dataclass
class PlanConstraints:
    """Constraints for plan generation."""
    max_steps: int = 20
    max_parallel: int = 5
    allowed_tools: list[str] | None = None
    forbidden_tools: list[str] | None = None
    max_tokens: int = 10000
    prefer_parallel: bool = True
```

---

## 5. Reasoning Subsystem API

### 5.1 ReasoningManager Class

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

    Attributes:
        strategies: Registry of available strategies
        complexity_scorer: For task complexity assessment
        cache: Strategy and result caching
        memory_manager: For fact grounding
    """
```

#### Constructor

```python
def __init__(
    self,
    model: str = "anthropic:claude-sonnet-4-20250514",
    memory_manager: MemoryManager | None = None,
    tool_registry: ToolRegistry | None = None,
    cache_enabled: bool = True,
) -> None:
```

**Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `model` | `str` | No | `"anthropic:claude-sonnet-4-20250514"` | Default LLM model |
| `memory_manager` | `MemoryManager` | No | None | Memory for context |
| `tool_registry` | `ToolRegistry` | No | None | Tools for ReAct |
| `cache_enabled` | `bool` | No | True | Enable result caching |

#### Method: `reason`

```python
async def reason(
    self,
    task: str,
    complexity: float | None = None,
    strategy_override: str | None = None,
    context: ReasoningContext | None = None,
    token_budget: TokenBudget | None = None,
    session_id: str | None = None,
) -> StrategyResult:
    """Execute reasoning on a task.

    Args:
        task: The task/question to reason about
        complexity: Pre-computed complexity (or auto-assess)
        strategy_override: Force specific strategy
        context: Additional context (facts, history)
        token_budget: Token budget constraint
        session_id: Session ID for tracking

    Returns:
        StrategyResult with answer, trace, and confidence

    Raises:
        ReasoningError: If all strategies fail
        TokenBudgetExceeded: If budget exhausted

    Example:
        >>> result = await manager.reason(
        ...     task="What pricing plan should we recommend for a 50-person company?",
        ...     context=ReasoningContext(facts=["Enterprise is $500/month"]),
        ... )
        >>> print(f"Answer: {result.answer}")
        >>> print(f"Confidence: {result.confidence}")
    """
```

**Return Value:**

```python
StrategyResult(
    answer="Based on the company size and typical needs...",
    reasoning_trace=[
        "Step 1: Analyze company size (50 people)",
        "Step 2: Consider typical enterprise needs",
        "Step 3: Compare pricing tiers",
        "Step 4: Recommend Professional plan",
    ],
    confidence=0.85,
    tokens_used=450,
    strategy_name="cot",
    duration_ms=3200,
)
```

**Events Emitted:**
- `ReasoningTaskStartedEvent(task, complexity)`
- `StrategySelectedEvent(strategy_name, complexity)`
- `ReasoningCompletedEvent(strategy_name, confidence, tokens)`
- `StrategyFallbackEvent(original_strategy, fallback_to)` on fallback

#### Method: `select_strategy`

```python
def select_strategy(
    self,
    complexity: float,
    task_type: str | None = None,
) -> BaseStrategy:
    """Select strategy based on complexity.

    Args:
        complexity: Task complexity score (0.0 - 1.0)
        task_type: Optional task type for learned optimization

    Returns:
        Strategy instance to use

    Selection Logic:
        0.0 - 0.3: DirectStrategy (simple factual)
        0.3 - 0.5: CoTStrategy (step-by-step)
        0.5 - 0.7: ToTStrategy (explore alternatives)
        0.7 - 0.9: ReActStrategy (tool-interleaved)
        0.9 - 1.0: MCTSStrategy (critical decisions)
    """
```

#### Method: `assess_complexity`

```python
async def assess_complexity(
    self,
    task: str,
    context: ReasoningContext | None = None,
) -> ComplexityAssessment:
    """Assess complexity of a reasoning task.

    More detailed than Router's ComplexityAssessor - specifically
    designed for reasoning strategy selection.

    Returns:
        ComplexityAssessment with score and factors
    """
```

### 5.2 BaseStrategy Abstract Class

```python
from abc import ABC, abstractmethod

class BaseStrategy(ABC):
    """Abstract base class for reasoning strategies.

    All strategies must implement the execute() method.
    Strategies are stateless - all context passed via parameters.

    Attributes:
        name: Strategy identifier
        model: LLM model to use
        max_tokens: Maximum tokens for this strategy
    """

    name: str
    model: str
    max_tokens: int

    @abstractmethod
    async def execute(
        self,
        task: str,
        context: ReasoningContext | None = None,
        token_budget: TokenBudget | None = None,
    ) -> StrategyResult:
        """Execute the reasoning strategy.

        Args:
            task: Task/question to reason about
            context: Optional context (facts, history)
            token_budget: Optional token constraint

        Returns:
            StrategyResult with answer and metadata
        """
        pass

    def get_prompt(self, task: str, context: ReasoningContext | None) -> str:
        """Generate strategy-specific prompt."""
        pass
```

### 5.3 Strategy Implementations

#### 5.3.1 DirectStrategy

```python
class DirectStrategy(BaseStrategy):
    """Simple single-call reasoning for straightforward tasks.

    Characteristics:
    - Tokens: 100-300
    - Latency: 1-2s
    - Paths explored: 1
    - Best for: Factual queries, simple lookups

    No multi-step reasoning - just prompt and respond.
    """

    name = "direct"
    max_tokens = 300

    async def execute(
        self,
        task: str,
        context: ReasoningContext | None = None,
        token_budget: TokenBudget | None = None,
    ) -> StrategyResult:
        """Execute direct single-call reasoning."""
```

#### 5.3.2 CoTStrategy (Chain-of-Thought)

```python
class CoTStrategy(BaseStrategy):
    """Step-by-step reasoning with explicit thought process.

    Characteristics:
    - Tokens: 300-800
    - Latency: 2-4s
    - Paths explored: 1
    - Best for: Multi-step problems, calculations

    Uses "Let's think step by step" prompting pattern.
    Captures reasoning trace for transparency.
    """

    name = "cot"
    max_tokens = 800

    async def execute(
        self,
        task: str,
        context: ReasoningContext | None = None,
        token_budget: TokenBudget | None = None,
    ) -> StrategyResult:
        """Execute chain-of-thought reasoning.

        Prompt includes:
        - Task description
        - Available context/facts
        - "Let's think step by step" instruction
        - Request for numbered steps

        Parsing extracts:
        - Individual reasoning steps
        - Final answer
        - Confidence estimation
        """
```

#### 5.3.3 ToTStrategy (Tree-of-Thoughts)

```python
class ToTStrategy(BaseStrategy):
    """Explore multiple reasoning paths and select best.

    Characteristics:
    - Tokens: 800-2000
    - Latency: 4-8s
    - Paths explored: 3-5
    - Best for: Ambiguous problems, creative tasks

    Generates multiple candidate solutions, evaluates each,
    and selects the best path with explanation.
    """

    name = "tot"
    max_tokens = 2000
    num_paths: int = 3
    max_depth: int = 3

    async def execute(
        self,
        task: str,
        context: ReasoningContext | None = None,
        token_budget: TokenBudget | None = None,
    ) -> StrategyResult:
        """Execute tree-of-thoughts reasoning.

        Algorithm:
        1. Generate num_paths initial approaches
        2. For each approach, generate next-step options
        3. Evaluate all paths using scoring prompt
        4. Select best path and synthesize answer
        5. Include alternative paths in trace
        """

    async def generate_paths(
        self,
        task: str,
        context: ReasoningContext | None,
    ) -> list[ReasoningPath]:
        """Generate multiple reasoning paths in parallel."""

    async def evaluate_paths(
        self,
        paths: list[ReasoningPath],
        task: str,
    ) -> list[tuple[ReasoningPath, float]]:
        """Evaluate and score each path."""
```

#### 5.3.4 ReActStrategy (Reasoning + Acting)

```python
class ReActStrategy(BaseStrategy):
    """Interleaved reasoning and tool use.

    Characteristics:
    - Tokens: 1000-3000
    - Latency: 5-15s
    - Paths explored: 1+ (with tool loops)
    - Best for: Tool-intensive tasks, information gathering

    Implements Thought -> Action -> Observation loop.
    Continues until answer found or max_iterations reached.
    """

    name = "react"
    max_tokens = 3000
    max_iterations: int = 5
    tool_registry: ToolRegistry

    async def execute(
        self,
        task: str,
        context: ReasoningContext | None = None,
        token_budget: TokenBudget | None = None,
    ) -> StrategyResult:
        """Execute ReAct reasoning loop.

        Loop:
        1. Thought: Reason about current state
        2. Action: Select and call tool (or finish)
        3. Observation: Process tool result
        4. Repeat until Finish action or max_iterations

        Tool calls are SEQUENTIAL (not parallel) to maintain
        coherent reasoning trace.
        """

    def parse_action(self, response: str) -> tuple[str, dict[str, Any]]:
        """Parse action from LLM response.

        Format: Action[tool_name]: {"arg": "value"}
        Or: Finish[answer]: Final answer text
        """
```

#### 5.3.5 MCTSStrategy (Monte Carlo Tree Search)

```python
class MCTSStrategy(BaseStrategy):
    """Monte Carlo Tree Search for critical decisions.

    Characteristics:
    - Tokens: 2000-5000
    - Latency: 10-30s
    - Paths explored: Tree structure
    - Best for: High-stakes decisions, game-like problems

    Simulates multiple decision paths, uses UCB1 for
    exploration/exploitation balance, and selects most
    promising path.
    """

    name = "mcts"
    max_tokens = 5000
    num_simulations: int = 10
    exploration_constant: float = 1.414  # sqrt(2)
    max_depth: int = 5

    async def execute(
        self,
        task: str,
        context: ReasoningContext | None = None,
        token_budget: TokenBudget | None = None,
    ) -> StrategyResult:
        """Execute MCTS reasoning.

        Algorithm:
        1. Create root node from task
        2. For each simulation:
           a. Select: Use UCB1 to select promising node
           b. Expand: Generate child nodes (possible next steps)
           c. Simulate: Run to terminal state
           d. Backpropagate: Update node scores
        3. Select best action from root
        4. Synthesize answer from best path
        """

    def ucb1_score(self, node: MCTSNode, parent_visits: int) -> float:
        """Calculate UCB1 score for node selection."""
        exploitation = node.value / node.visits
        exploration = self.exploration_constant * sqrt(log(parent_visits) / node.visits)
        return exploitation + exploration
```

### 5.4 Strategy Characteristics Summary

| Strategy | Complexity Range | Tokens | Latency | Paths | Best For |
|----------|------------------|--------|---------|-------|----------|
| Direct | 0.0 - 0.3 | 100-300 | 1-2s | 1 | Factual queries |
| CoT | 0.3 - 0.5 | 300-800 | 2-4s | 1 | Step-by-step problems |
| ToT | 0.5 - 0.7 | 800-2000 | 4-8s | 3-5 | Ambiguous, creative |
| ReAct | 0.7 - 0.9 | 1000-3000 | 5-15s | 1+ | Tool-intensive |
| MCTS | 0.9 - 1.0 | 2000-5000 | 10-30s | Tree | Critical decisions |

---

## 6. Token Budget Management

### 6.1 Cross-Phase Budget Allocation

```python
@dataclass
class SessionTokenBudget:
    """Token budget allocation across all phases.

    Default allocation for 50,000 total tokens:
    - Planning: 15% (7,500 tokens)
    - Reasoning: 50% (25,000 tokens)
    - Memory: 20% (10,000 tokens)
    - Routing/Overhead: 15% (7,500 tokens)
    """
    total_tokens: int = 50_000

    # Subsystem allocations (percentages)
    planning_pct: float = 0.15
    reasoning_pct: float = 0.50
    memory_pct: float = 0.20
    overhead_pct: float = 0.15

    @property
    def planning_budget(self) -> int:
        return int(self.total_tokens * self.planning_pct)

    @property
    def reasoning_budget(self) -> int:
        return int(self.total_tokens * self.reasoning_pct)

    @property
    def memory_budget(self) -> int:
        return int(self.total_tokens * self.memory_pct)

    @property
    def overhead_budget(self) -> int:
        return int(self.total_tokens * self.overhead_pct)
```

### 6.2 Phase 5 Token Budgeting

#### Planning Token Costs

| Operation | Typical Tokens | Notes |
|-----------|----------------|-------|
| Plan generation | 1000-2000 | Depends on goal complexity |
| Plan context (per step) | 300-500 | Memory context included |
| Plan optimization | 500-800 | Optional step |
| **Total per plan** | **2000-3500** | |

#### Reasoning Token Costs

| Strategy | Input Tokens | Output Tokens | Total Range |
|----------|--------------|---------------|-------------|
| Direct | 50-100 | 50-200 | 100-300 |
| CoT | 100-200 | 200-600 | 300-800 |
| ToT | 200-500 | 600-1500 | 800-2000 |
| ReAct | 300-800 | 700-2200 | 1000-3000 |
| MCTS | 500-1500 | 1500-3500 | 2000-5000 |

### 6.3 Budget Tracking Integration

```python
class Phase5TokenTracker:
    """Tracks token usage specifically for Phase 5 operations.

    Integrates with Phase 3 TokenTracker but provides detailed
    breakdown for planning and reasoning operations.
    """

    def __init__(self, session_tracker: TokenTracker):
        self.session_tracker = session_tracker
        self.planning_tokens = 0
        self.reasoning_tokens = 0
        self.by_strategy: dict[str, int] = defaultdict(int)

    def record_planning(self, input_tokens: int, output_tokens: int):
        """Record tokens used for planning operations."""
        total = input_tokens + output_tokens
        self.planning_tokens += total
        self.session_tracker.record_usage(input_tokens, output_tokens)

    def record_reasoning(
        self,
        strategy: str,
        input_tokens: int,
        output_tokens: int,
    ):
        """Record tokens used for reasoning operations."""
        total = input_tokens + output_tokens
        self.reasoning_tokens += total
        self.by_strategy[strategy] += total
        self.session_tracker.record_usage(input_tokens, output_tokens)

    def get_remaining_for_planning(self, budget: SessionTokenBudget) -> int:
        """Get remaining tokens available for planning."""
        return max(0, budget.planning_budget - self.planning_tokens)

    def get_remaining_for_reasoning(self, budget: SessionTokenBudget) -> int:
        """Get remaining tokens available for reasoning."""
        return max(0, budget.reasoning_budget - self.reasoning_tokens)

    def should_degrade_strategy(self, budget: SessionTokenBudget) -> bool:
        """Check if we should use cheaper strategy due to budget."""
        remaining = self.get_remaining_for_reasoning(budget)
        return remaining < 1000  # Switch to Direct/CoT only
```

### 6.4 Graceful Degradation

```python
async def reason_with_budget_awareness(
    manager: ReasoningManager,
    task: str,
    complexity: float,
    budget: SessionTokenBudget,
    tracker: Phase5TokenTracker,
) -> StrategyResult:
    """Execute reasoning with budget-aware strategy selection.

    Degradation Rules:
    - If < 500 tokens remaining: Force Direct strategy
    - If < 1000 tokens remaining: Force Direct or CoT
    - If < 2000 tokens remaining: Exclude MCTS
    - Normal selection otherwise
    """
    remaining = tracker.get_remaining_for_reasoning(budget)

    if remaining < 500:
        strategy_override = "direct"
    elif remaining < 1000:
        strategy_override = "direct" if complexity < 0.4 else "cot"
    elif remaining < 2000:
        # Exclude MCTS, cap at ReAct
        complexity = min(complexity, 0.89)
        strategy_override = None
    else:
        strategy_override = None

    return await manager.reason(
        task=task,
        complexity=complexity,
        strategy_override=strategy_override,
    )
```

---

## 7. Event Contract

### 7.1 Planning Events

```python
class PlanningEventType(str, Enum):
    """Planning subsystem event types."""
    PLAN_CREATED = "plan.created"
    PLAN_EXECUTION_STARTED = "plan.execution_started"
    PLAN_STEP_STARTED = "plan.step_started"
    PLAN_STEP_COMPLETED = "plan.step_completed"
    PLAN_STEP_FAILED = "plan.step_failed"
    PLAN_COMPLETED = "plan.completed"
    PLAN_PAUSED = "plan.paused"
    PLAN_RESUMED = "plan.resumed"
```

#### Event Payloads

```python
@dataclass
class PlanCreatedPayload:
    """Payload for PLAN_CREATED event."""
    plan_id: str
    goal: str
    step_count: int
    estimated_tokens: int
    complexity: float


@dataclass
class PlanExecutionStartedPayload:
    """Payload for PLAN_EXECUTION_STARTED event."""
    plan_id: str
    token_budget: int | None


@dataclass
class PlanStepStartedPayload:
    """Payload for PLAN_STEP_STARTED event."""
    plan_id: str
    step_id: str
    step_type: str
    description: str


@dataclass
class PlanStepCompletedPayload:
    """Payload for PLAN_STEP_COMPLETED event."""
    plan_id: str
    step_id: str
    success: bool
    output_preview: str  # First 200 chars
    tokens_used: int
    duration_ms: float


@dataclass
class PlanStepFailedPayload:
    """Payload for PLAN_STEP_FAILED event."""
    plan_id: str
    step_id: str
    error_type: str
    error_message: str
    retry_count: int
    will_retry: bool


@dataclass
class PlanCompletedPayload:
    """Payload for PLAN_COMPLETED event."""
    plan_id: str
    success: bool
    total_tokens: int
    total_duration_ms: float
    steps_completed: int
    steps_failed: int


@dataclass
class PlanPausedPayload:
    """Payload for PLAN_PAUSED event."""
    plan_id: str
    last_completed_step: str
    checkpoint_id: str


@dataclass
class PlanResumedPayload:
    """Payload for PLAN_RESUMED event."""
    plan_id: str
    resuming_from: str
    checkpoint_id: str
```

### 7.2 Reasoning Events

```python
class ReasoningEventType(str, Enum):
    """Reasoning subsystem event types."""
    REASONING_TASK_STARTED = "reasoning.task_started"
    STRATEGY_SELECTED = "reasoning.strategy_selected"
    REASONING_COMPLETED = "reasoning.completed"
    STRATEGY_FALLBACK = "reasoning.fallback"
```

#### Event Payloads

```python
@dataclass
class ReasoningTaskStartedPayload:
    """Payload for REASONING_TASK_STARTED event."""
    task_hash: str  # SHA256 hash for privacy
    complexity: float
    context_facts_count: int


@dataclass
class StrategySelectedPayload:
    """Payload for STRATEGY_SELECTED event."""
    strategy_name: str
    complexity: float
    was_override: bool
    reason: str  # Why this strategy was selected


@dataclass
class ReasoningCompletedPayload:
    """Payload for REASONING_COMPLETED event."""
    strategy_name: str
    confidence: float
    tokens_used: int
    duration_ms: float
    trace_length: int  # Number of reasoning steps


@dataclass
class StrategyFallbackPayload:
    """Payload for STRATEGY_FALLBACK event."""
    original_strategy: str
    fallback_strategy: str
    reason: str  # Why fallback occurred
```

### 7.3 Event Factory Functions

```python
def create_plan_created_event(
    session_id: str,
    plan_id: str,
    goal: str,
    step_count: int,
    estimated_tokens: int,
    complexity: float,
    correlation_id: str | None = None,
) -> Event:
    """Create a PlanCreatedEvent."""
    return Event(
        event_id=_generate_event_id(),
        event_type=EventType.PLAN_CREATED,
        timestamp=_get_utc_now(),
        session_id=session_id,
        correlation_id=correlation_id,
        payload={
            "plan_id": plan_id,
            "goal": goal,
            "step_count": step_count,
            "estimated_tokens": estimated_tokens,
            "complexity": complexity,
        },
    )


def create_reasoning_completed_event(
    session_id: str,
    strategy_name: str,
    confidence: float,
    tokens_used: int,
    duration_ms: float,
    trace_length: int,
    correlation_id: str | None = None,
) -> Event:
    """Create a ReasoningCompletedEvent."""
    return Event(
        event_id=_generate_event_id(),
        event_type=EventType.REASONING_COMPLETED,
        timestamp=_get_utc_now(),
        session_id=session_id,
        correlation_id=correlation_id,
        payload={
            "strategy_name": strategy_name,
            "confidence": confidence,
            "tokens_used": tokens_used,
            "duration_ms": duration_ms,
            "trace_length": trace_length,
        },
    )


# ... additional factory functions for all event types
```

---

## 8. Concurrency and Parallelization

### 8.1 Planning Concurrency

```python
@dataclass
class ConcurrencyConfig:
    """Configuration for concurrent execution."""
    max_plan_steps: int = 3          # Max parallel plan steps
    max_tot_paths: int = 5           # Max parallel ToT paths
    max_mcts_simulations: int = 10   # Max parallel MCTS sims
    total_concurrent_cap: int = 20   # Overall maximum
```

#### Plan Step Parallelization

```python
async def execute_parallel_steps(
    steps: list[PlanStep],
    context: ExecutionContext,
    max_concurrent: int = 3,
) -> list[StepResult]:
    """Execute independent plan steps in parallel.

    Uses semaphore to limit concurrent executions.
    Steps with no dependencies on each other run simultaneously.
    """
    semaphore = asyncio.Semaphore(max_concurrent)

    async def execute_with_limit(step: PlanStep) -> StepResult:
        async with semaphore:
            return await execute_step(step, context)

    results = await asyncio.gather(
        *[execute_with_limit(step) for step in steps],
        return_exceptions=True,
    )

    return [
        r if isinstance(r, StepResult) else StepResult.from_exception(r)
        for r in results
    ]
```

#### Dependency-Aware Execution

```python
async def execute_plan_with_dependencies(
    plan: Plan,
    context: ExecutionContext,
    max_concurrent: int = 3,
) -> ExecutionResult:
    """Execute plan respecting step dependencies.

    Uses topological sort to determine execution waves.
    Each wave contains steps that can run in parallel.
    """
    execution_order = plan.get_execution_order()
    all_results: dict[str, StepResult] = {}

    for wave in execution_order:
        # Get steps in this wave
        wave_steps = [plan.get_step(step_id) for step_id in wave]

        # Inject dependency results into context
        for step in wave_steps:
            step_context = context.with_dependency_results(
                {dep_id: all_results[dep_id] for dep_id in step.dependencies}
            )

        # Execute wave in parallel
        wave_results = await execute_parallel_steps(
            wave_steps, step_context, max_concurrent
        )

        # Store results
        for step, result in zip(wave_steps, wave_results):
            all_results[step.step_id] = result

    return ExecutionResult.from_step_results(plan, all_results)
```

### 8.2 Reasoning Concurrency

#### ToT Parallel Path Generation

```python
async def generate_tot_paths_parallel(
    task: str,
    context: ReasoningContext,
    num_paths: int = 3,
) -> list[ReasoningPath]:
    """Generate ToT reasoning paths in parallel.

    All path generations are independent and can run
    simultaneously for faster exploration.
    """
    path_tasks = [
        generate_single_path(task, context, path_index=i)
        for i in range(num_paths)
    ]

    paths = await asyncio.gather(*path_tasks)
    return paths
```

#### MCTS Parallel Simulation

```python
async def run_mcts_simulations_parallel(
    node: MCTSNode,
    num_simulations: int = 10,
    max_parallel: int = 10,
) -> list[SimulationResult]:
    """Run MCTS simulations in parallel batches.

    Simulations are independent and can run in parallel.
    Results are aggregated for backpropagation.
    """
    semaphore = asyncio.Semaphore(max_parallel)

    async def simulate_with_limit() -> SimulationResult:
        async with semaphore:
            return await simulate_from_node(node)

    results = await asyncio.gather(
        *[simulate_with_limit() for _ in range(num_simulations)]
    )

    return results
```

#### ReAct Sequential Execution

```python
async def execute_react_loop(
    task: str,
    context: ReasoningContext,
    tool_registry: ToolRegistry,
    max_iterations: int = 5,
) -> StrategyResult:
    """Execute ReAct loop SEQUENTIALLY.

    Tool calls must be sequential because:
    1. Each observation informs the next thought
    2. Tool state may change between calls
    3. Reasoning trace must be coherent

    NOT parallelized intentionally.
    """
    trace = []
    for iteration in range(max_iterations):
        # Thought
        thought = await generate_thought(task, context, trace)
        trace.append(f"Thought {iteration + 1}: {thought}")

        # Action
        action, args = parse_action(thought)

        if action == "Finish":
            return StrategyResult(answer=args, reasoning_trace=trace)

        # Observation (sequential tool call)
        observation = await execute_tool(tool_registry, action, args)
        trace.append(f"Observation: {observation}")

    # Max iterations reached
    return StrategyResult(
        answer="Could not complete within iteration limit",
        reasoning_trace=trace,
        confidence=0.3,
    )
```

---

## 9. Performance Optimization

### 9.1 Caching Strategy

```python
@dataclass
class CacheConfig:
    """Caching configuration for Phase 5."""
    complexity_score_ttl: int = 60          # 60 seconds
    similar_goal_ttl: int = 86400           # 24 hours
    strategy_effectiveness_ttl: int = 604800  # 7 days
    llm_response_ttl: int = 3600            # 1 hour


class Phase5Cache:
    """Multi-tier cache for Phase 5 operations.

    Cache Levels:
    1. Complexity scores: Short TTL, high hit rate
    2. Similar goals: Medium TTL, plan reuse
    3. Strategy effectiveness: Long TTL, learning
    4. LLM responses: Short TTL, identical prompts
    """

    def __init__(self, config: CacheConfig = CacheConfig()):
        self.config = config
        self.complexity_cache: dict[str, tuple[float, datetime]] = {}
        self.goal_cache: dict[str, tuple[Plan, datetime]] = {}
        self.strategy_cache: dict[str, tuple[str, datetime]] = {}
        self.response_cache: dict[str, tuple[str, datetime]] = {}

    def get_complexity(self, task_hash: str) -> float | None:
        """Get cached complexity score."""
        if task_hash in self.complexity_cache:
            score, timestamp = self.complexity_cache[task_hash]
            if (datetime.now() - timestamp).seconds < self.config.complexity_score_ttl:
                return score
        return None

    def get_similar_plan(self, goal_hash: str) -> Plan | None:
        """Get cached plan for similar goal."""
        if goal_hash in self.goal_cache:
            plan, timestamp = self.goal_cache[goal_hash]
            if (datetime.now() - timestamp).seconds < self.config.similar_goal_ttl:
                return plan
        return None

    def get_effective_strategy(self, task_type: str) -> str | None:
        """Get most effective strategy for task type."""
        if task_type in self.strategy_cache:
            strategy, timestamp = self.strategy_cache[task_type]
            if (datetime.now() - timestamp).seconds < self.config.strategy_effectiveness_ttl:
                return strategy
        return None

    def record_strategy_success(self, task_type: str, strategy: str, success: bool):
        """Record strategy outcome for learning."""
        # Update effectiveness tracking
        pass
```

### 9.2 Batching Strategies

```python
async def batch_tot_path_generation(
    tasks: list[str],
    context: ReasoningContext,
    batch_size: int = 5,
) -> list[list[ReasoningPath]]:
    """Batch ToT path generation for multiple tasks.

    More efficient than sequential calls when handling
    multiple reasoning tasks in a plan.
    """
    all_paths = []
    for i in range(0, len(tasks), batch_size):
        batch = tasks[i:i + batch_size]
        batch_paths = await asyncio.gather(
            *[generate_tot_paths(task, context) for task in batch]
        )
        all_paths.extend(batch_paths)
    return all_paths


async def batch_memory_retrieval(
    queries: list[str],
    memory_manager: MemoryManager,
    k: int = 5,
) -> dict[str, list[MemoryItem]]:
    """Batch memory retrieval for multiple queries.

    More efficient than individual retrievals due to
    embedding batch generation.
    """
    # Generate all embeddings at once
    embeddings = await memory_manager.item_layer.embedding_service.embed_batch(queries)

    # Search all at once
    results = {}
    for query, embedding in zip(queries, embeddings):
        items = await memory_manager.item_layer.search_by_embedding(embedding, k)
        results[query] = items

    return results
```

### 9.3 Pre-Generation and Warm-Up

```python
async def warm_up_common_plans(
    planner: Planner,
    common_goals: list[str],
) -> None:
    """Pre-generate plans for common goals at startup.

    Reduces latency for frequently requested operations.
    """
    for goal in common_goals:
        plan = await planner.generate(goal)
        planner.cache.store_plan(goal, plan)


# Common goals to pre-generate
COMMON_GOALS = [
    "Qualify a new lead",
    "Schedule a meeting",
    "Research a company",
    "Draft a follow-up email",
    "Analyze objection patterns",
]
```

### 9.4 Early Termination

```python
async def tot_with_early_termination(
    task: str,
    context: ReasoningContext,
    confidence_threshold: float = 0.9,
) -> StrategyResult:
    """ToT with early termination when high-confidence path found.

    Stops exploring additional paths if we find one with
    confidence >= threshold, saving tokens.
    """
    paths = []
    for i in range(MAX_PATHS):
        path = await generate_single_path(task, context, i)
        score = await evaluate_path(path, task)
        paths.append((path, score))

        if score >= confidence_threshold:
            # Early termination - found good enough answer
            return StrategyResult(
                answer=path.final_answer,
                confidence=score,
                reasoning_trace=path.steps,
                tokens_used=path.tokens_used,
            )

    # No early termination - return best path
    best_path, best_score = max(paths, key=lambda x: x[1])
    return StrategyResult(
        answer=best_path.final_answer,
        confidence=best_score,
        reasoning_trace=best_path.steps,
    )
```

### 9.5 Performance Characteristics

| Operation | Cold (ms) | Warm/Cached (ms) | Notes |
|-----------|-----------|------------------|-------|
| Plan generation | 3000-8000 | 50-100 | Cached plans for similar goals |
| Complexity assessment | 500-1000 | 10-20 | Cached scores |
| Direct reasoning | 1000-2000 | 800-1500 | Minimal caching benefit |
| CoT reasoning | 2000-4000 | 1500-3000 | Cached prompts help |
| ToT reasoning | 4000-8000 | 3000-6000 | Early termination helps |
| ReAct reasoning | 5000-15000 | 4000-12000 | Sequential, limited caching |
| MCTS reasoning | 10000-30000 | 8000-25000 | Parallel sims help |

---

## 10. Error Handling and Resilience

### 10.1 Planning Error Handling

```python
class PlanningError(SigilError):
    """Base exception for planning errors."""
    code = "PLAN-000"


class InvalidGoalError(PlanningError):
    """Goal is empty, too vague, or invalid."""
    code = "PLAN-001"

    def __init__(self, goal: str, reason: str):
        super().__init__(
            message=f"Invalid goal: {reason}",
            context={"goal": goal[:100], "reason": reason},
            recovery_suggestions=[
                "Provide a more specific goal",
                "Include what outcome you want to achieve",
                "Add context about the situation",
            ],
        )


class PlanGenerationError(PlanningError):
    """Failed to generate a valid plan."""
    code = "PLAN-002"

    recovery_suggestions = [
        "Try with a simpler goal",
        "Break goal into smaller pieces manually",
        "Retry - may be transient LLM error",
    ]


class DependencyCycleError(PlanningError):
    """Plan contains circular dependencies."""
    code = "PLAN-003"

    def __init__(self, cycle: list[str]):
        super().__init__(
            message=f"Circular dependency detected: {' -> '.join(cycle)}",
            context={"cycle": cycle},
            recovery_suggestions=[
                "Regenerating plan without cycle",
                "Manual step reordering may be needed",
            ],
        )


class StepExecutionError(PlanningError):
    """Step execution failed after all retries."""
    code = "PLAN-004"

    def __init__(self, step_id: str, error: Exception, retries: int):
        super().__init__(
            message=f"Step {step_id} failed after {retries} retries: {error}",
            context={"step_id": step_id, "retries": retries, "error": str(error)},
            recovery_suggestions=[
                "Check step dependencies are satisfied",
                "Verify tool availability",
                "Review step parameters",
            ],
        )
```

### 10.2 Reasoning Error Handling

```python
class ReasoningError(SigilError):
    """Base exception for reasoning errors."""
    code = "REASON-000"


class StrategyExecutionError(ReasoningError):
    """Strategy failed to execute."""
    code = "REASON-001"

    def __init__(self, strategy: str, error: Exception):
        super().__init__(
            message=f"Strategy {strategy} failed: {error}",
            context={"strategy": strategy, "error": str(error)},
            recovery_suggestions=[
                "Falling back to simpler strategy",
                "Check LLM API availability",
            ],
        )


class AllStrategiesFailedError(ReasoningError):
    """All reasoning strategies have failed."""
    code = "REASON-002"

    def __init__(self, strategies_tried: list[str]):
        super().__init__(
            message=f"All strategies failed: {strategies_tried}",
            context={"strategies_tried": strategies_tried},
            recovery_suggestions=[
                "Check LLM API status",
                "Try with simpler task formulation",
                "Manual intervention may be required",
            ],
        )


class ToolExecutionError(ReasoningError):
    """Tool call in ReAct failed."""
    code = "REASON-003"

    def __init__(self, tool_name: str, error: Exception):
        super().__init__(
            message=f"Tool {tool_name} failed: {error}",
            context={"tool_name": tool_name, "error": str(error)},
            recovery_suggestions=[
                "Check tool availability",
                "Verify tool parameters",
                "Try alternative tool if available",
            ],
        )
```

### 10.3 Fallback Chains

```python
class StrategyFallbackChain:
    """Manages fallback logic when strategies fail."""

    FALLBACK_ORDER = {
        "mcts": ["react", "tot", "cot", "direct"],
        "react": ["tot", "cot", "direct"],
        "tot": ["cot", "direct"],
        "cot": ["direct"],
        "direct": [],  # No fallback from direct
    }

    async def execute_with_fallback(
        self,
        task: str,
        initial_strategy: str,
        context: ReasoningContext,
    ) -> StrategyResult:
        """Execute with automatic fallback on failure."""
        current_strategy = initial_strategy
        errors = []

        while True:
            try:
                strategy = self.strategies[current_strategy]
                result = await strategy.execute(task, context)
                return result

            except StrategyExecutionError as e:
                errors.append((current_strategy, e))

                # Get next fallback
                fallbacks = self.FALLBACK_ORDER.get(current_strategy, [])
                if not fallbacks:
                    raise AllStrategiesFailedError(
                        strategies_tried=[s for s, _ in errors]
                    )

                current_strategy = fallbacks[0]

                # Emit fallback event
                emit_strategy_fallback_event(
                    original=errors[-1][0],
                    fallback=current_strategy,
                    reason=str(e),
                )
```

### 10.4 Recovery Guarantees

| Failure Mode | Recovery Action | Guarantee |
|--------------|----------------|-----------|
| Plan generation fails | Return empty plan, ask for manual steps | Always return something |
| Dependency cycle detected | Regenerate without cycles | Valid DAG |
| Step execution fails | Retry up to 3x, then skip | Continue execution |
| Strategy fails | Fallback to simpler strategy | DirectStrategy as last resort |
| Token budget exceeded | Return best result so far | Partial result |
| Timeout | Return best result with flag | Partial result |
| LLM API error | Retry once, then fallback | Best-effort result |

### 10.5 Checkpoint and Resume

```python
@dataclass
class ExecutionCheckpoint:
    """Checkpoint for plan execution resumption."""
    checkpoint_id: str
    plan_id: str
    completed_steps: list[str]
    step_results: dict[str, StepResult]
    current_step: str | None
    created_at: datetime
    token_usage: int


class CheckpointStore:
    """Stores and retrieves execution checkpoints."""

    def __init__(self, storage_dir: str = "outputs/checkpoints"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    def save_checkpoint(self, checkpoint: ExecutionCheckpoint) -> None:
        """Save checkpoint to disk."""
        path = self.storage_dir / f"{checkpoint.plan_id}.json"
        with open(path, "w") as f:
            json.dump(checkpoint.to_dict(), f)

    def load_checkpoint(self, plan_id: str) -> ExecutionCheckpoint | None:
        """Load checkpoint from disk."""
        path = self.storage_dir / f"{plan_id}.json"
        if not path.exists():
            return None
        with open(path) as f:
            data = json.load(f)
        return ExecutionCheckpoint.from_dict(data)

    def delete_checkpoint(self, plan_id: str) -> None:
        """Delete checkpoint after successful completion."""
        path = self.storage_dir / f"{plan_id}.json"
        if path.exists():
            path.unlink()
```

---

## 11. Monitoring and Observability

### 11.1 Metrics to Track

```python
@dataclass
class Phase5Metrics:
    """Metrics for Phase 5 monitoring."""

    # Planning metrics
    plans_generated: int = 0
    plans_executed: int = 0
    plan_generation_time_ms: list[float] = field(default_factory=list)
    plan_execution_time_ms: list[float] = field(default_factory=list)
    steps_per_plan: list[int] = field(default_factory=list)
    plan_success_rate: float = 0.0

    # Reasoning metrics
    reasoning_tasks: int = 0
    strategy_distribution: dict[str, int] = field(default_factory=dict)
    strategy_success_rate: dict[str, float] = field(default_factory=dict)
    fallback_rate: float = 0.0
    tokens_per_strategy: dict[str, list[int]] = field(default_factory=dict)

    # Performance metrics
    cache_hit_rate: float = 0.0
    avg_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0

    # Resource metrics
    token_budget_utilization: float = 0.0
    concurrent_executions: int = 0


class MetricsCollector:
    """Collects and aggregates Phase 5 metrics."""

    def __init__(self):
        self.metrics = Phase5Metrics()
        self._lock = asyncio.Lock()

    async def record_plan_generation(self, duration_ms: float, step_count: int):
        """Record plan generation metrics."""
        async with self._lock:
            self.metrics.plans_generated += 1
            self.metrics.plan_generation_time_ms.append(duration_ms)
            self.metrics.steps_per_plan.append(step_count)

    async def record_strategy_execution(
        self,
        strategy: str,
        success: bool,
        tokens: int,
        duration_ms: float,
    ):
        """Record strategy execution metrics."""
        async with self._lock:
            self.metrics.reasoning_tasks += 1
            self.metrics.strategy_distribution[strategy] = (
                self.metrics.strategy_distribution.get(strategy, 0) + 1
            )
            if strategy not in self.metrics.tokens_per_strategy:
                self.metrics.tokens_per_strategy[strategy] = []
            self.metrics.tokens_per_strategy[strategy].append(tokens)

    def get_summary(self) -> dict:
        """Get metrics summary for dashboard."""
        return {
            "planning": {
                "total_plans": self.metrics.plans_generated,
                "avg_generation_time_ms": self._avg(self.metrics.plan_generation_time_ms),
                "avg_steps_per_plan": self._avg(self.metrics.steps_per_plan),
                "success_rate": self.metrics.plan_success_rate,
            },
            "reasoning": {
                "total_tasks": self.metrics.reasoning_tasks,
                "strategy_distribution": self.metrics.strategy_distribution,
                "fallback_rate": self.metrics.fallback_rate,
            },
            "performance": {
                "cache_hit_rate": self.metrics.cache_hit_rate,
                "avg_latency_ms": self.metrics.avg_latency_ms,
                "p95_latency_ms": self.metrics.p95_latency_ms,
            },
        }
```

### 11.2 Dashboard Specifications

#### Dashboard 1: Phase 5 Overview

| Panel | Metric | Visualization |
|-------|--------|---------------|
| Plans/min | `plans_generated` rate | Line chart |
| Avg Plan Latency | `plan_generation_time_ms` avg | Gauge |
| Strategy Distribution | `strategy_distribution` | Pie chart |
| Success Rate | `plan_success_rate` | Gauge |

#### Dashboard 2: Reasoning Effectiveness

| Panel | Metric | Visualization |
|-------|--------|---------------|
| Strategy Accuracy | `strategy_success_rate` by strategy | Bar chart |
| Token Efficiency | `tokens_per_strategy` avg | Bar chart |
| Fallback Rate | `fallback_rate` | Gauge |
| Complexity Distribution | histogram of complexity scores | Histogram |

#### Dashboard 3: Resource Usage

| Panel | Metric | Visualization |
|-------|--------|---------------|
| Token Budget Util | `token_budget_utilization` | Gauge |
| Concurrent Tasks | `concurrent_executions` | Line chart |
| Cache Hit Rate | `cache_hit_rate` | Gauge |
| Memory Usage | system memory | Line chart |

### 11.3 Structured Logging

```python
import structlog

logger = structlog.get_logger(__name__)

# Planning logs
logger.info(
    "plan_generated",
    plan_id=plan.plan_id,
    goal=plan.goal[:50],
    step_count=len(plan.steps),
    complexity=plan.metadata.complexity,
    duration_ms=duration_ms,
)

logger.info(
    "plan_step_completed",
    plan_id=plan_id,
    step_id=step_id,
    success=result.success,
    tokens_used=result.tokens_used,
    duration_ms=result.duration_ms,
)

# Reasoning logs
logger.info(
    "strategy_selected",
    task_hash=hash(task)[:8],
    complexity=complexity,
    strategy=strategy.name,
    was_override=was_override,
)

logger.info(
    "reasoning_completed",
    strategy=strategy.name,
    confidence=result.confidence,
    tokens_used=result.tokens_used,
    duration_ms=duration_ms,
    trace_length=len(result.reasoning_trace),
)

logger.warning(
    "strategy_fallback",
    original_strategy=original,
    fallback_strategy=fallback,
    reason=str(error),
)
```

### 11.4 Debug Capabilities

```python
class DebugCapabilities:
    """Debug tools for Phase 5 troubleshooting."""

    async def replay_plan_execution(
        self,
        plan_id: str,
        event_store: EventStore,
    ) -> list[Event]:
        """Replay plan execution from stored events.

        Returns chronological list of events for analysis.
        """
        events = event_store.get_by_correlation(plan_id)
        return sorted(events, key=lambda e: e.timestamp)

    async def inspect_reasoning_trace(
        self,
        task: str,
        strategy: str,
    ) -> dict:
        """Execute reasoning with full trace capture.

        Returns detailed trace for debugging strategy behavior.
        """
        result = await self.reasoning_manager.reason(
            task=task,
            strategy_override=strategy,
            debug=True,
        )

        return {
            "task": task,
            "strategy": strategy,
            "trace": result.reasoning_trace,
            "confidence": result.confidence,
            "tokens": result.tokens_used,
            "intermediate_states": result.debug_info.get("states", []),
        }

    async def ab_test_strategies(
        self,
        tasks: list[str],
        strategies: list[str],
    ) -> dict:
        """A/B test strategies on tasks.

        Returns comparison of strategy performance.
        """
        results = {}
        for strategy in strategies:
            strategy_results = []
            for task in tasks:
                result = await self.reasoning_manager.reason(
                    task=task,
                    strategy_override=strategy,
                )
                strategy_results.append({
                    "task": task[:50],
                    "confidence": result.confidence,
                    "tokens": result.tokens_used,
                    "duration_ms": result.duration_ms,
                })
            results[strategy] = strategy_results

        return results

    async def simulate_budget_scenario(
        self,
        plan: Plan,
        budget: SessionTokenBudget,
    ) -> dict:
        """Simulate plan execution with different budgets.

        Returns predicted behavior at various budget levels.
        """
        scenarios = {}
        for pct in [0.25, 0.50, 0.75, 1.0]:
            simulated_budget = SessionTokenBudget(
                total_tokens=int(budget.total_tokens * pct)
            )
            scenarios[f"{int(pct * 100)}%"] = await self._simulate_execution(
                plan, simulated_budget
            )
        return scenarios
```

---

## 12. Data Schema Contracts

### 12.1 Plan Schemas

```python
@dataclass
class Plan:
    """Complete execution plan.

    Attributes:
        plan_id: Unique identifier (format: plan_{uuid})
        goal: Original goal
        steps: Ordered list of PlanStep
        metadata: Plan metadata
        created_at: Creation timestamp
        is_immutable: True after creation
    """
    plan_id: str
    goal: str
    steps: list[PlanStep]
    metadata: PlanMetadata
    created_at: datetime
    is_immutable: bool = True


@dataclass
class PlanStep:
    """Single step in execution plan.

    See Section 4.3 for full definition.
    """
    step_id: str
    description: str
    step_type: StepType
    dependencies: list[str] = field(default_factory=list)
    # ... additional fields


@dataclass
class PlanMetadata:
    """Plan metadata and estimates.

    See Section 4.3 for full definition.
    """
    complexity: float
    estimated_total_tokens: int
    estimated_duration_seconds: float
    parallel_groups: int
    tool_calls: int
    reasoning_steps: int
```

### 12.2 Reasoning Schemas

```python
@dataclass
class StrategyResult:
    """Result from strategy execution.

    Attributes:
        answer: Final answer text
        reasoning_trace: List of reasoning steps
        confidence: Confidence score (0.0-1.0)
        tokens_used: Total tokens consumed
        strategy_name: Strategy that produced result
        duration_ms: Execution duration
        fallback_used: True if fallback occurred
        original_strategy: Original strategy if fallback
    """
    answer: str
    reasoning_trace: list[str]
    confidence: float
    tokens_used: int
    strategy_name: str
    duration_ms: float = 0.0
    fallback_used: bool = False
    original_strategy: str | None = None


@dataclass
class ReasoningContext:
    """Context for reasoning execution.

    Attributes:
        facts: Known facts for grounding
        fact_sources: Source IDs for facts
        history: Conversation/reasoning history
        tools: Available tools for ReAct
        constraints: Any execution constraints
    """
    facts: list[str] = field(default_factory=list)
    fact_sources: list[str] = field(default_factory=list)
    history: list[str] = field(default_factory=list)
    tools: list[str] = field(default_factory=list)
    constraints: dict[str, Any] = field(default_factory=dict)


@dataclass
class ComplexityAssessment:
    """Detailed complexity assessment.

    Attributes:
        score: Overall complexity (0.0-1.0)
        factors: Individual factor scores
        suggested_strategy: Recommended strategy
        confidence: Confidence in assessment
    """
    score: float
    factors: dict[str, float]  # e.g., {"length": 0.3, "tools": 0.5}
    suggested_strategy: str
    confidence: float
```

### 12.3 Execution Schemas

```python
@dataclass
class ExecutionResult:
    """Result from plan execution.

    Attributes:
        plan_id: ID of executed plan
        success: True if all steps succeeded
        step_results: Results for each step
        total_tokens: Total tokens used
        total_duration_ms: Total execution time
        final_output: Aggregated final output
        checkpoints: Any checkpoints created
    """
    plan_id: str
    success: bool
    step_results: list[StepResult]
    total_tokens: int
    total_duration_ms: float
    final_output: str
    checkpoints: list[str] = field(default_factory=list)


@dataclass
class StepResult:
    """Result from step execution.

    Attributes:
        step_id: Step identifier
        status: Execution status
        output: Step output
        tokens_used: Tokens consumed
        duration_ms: Execution time
        retries: Number of retries
        error: Error if failed
    """
    step_id: str
    status: StepStatus
    output: Any
    tokens_used: int
    duration_ms: float
    retries: int = 0
    error: str | None = None


class StepStatus(str, Enum):
    """Step execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    FAILED_CONTRACT = "failed_contract"
```

---

## 13. Integration Examples

### 13.1 Full Planning + Reasoning Workflow

```python
from sigil.planning import Planner, PlanExecutor
from sigil.reasoning import ReasoningManager
from sigil.memory import MemoryManager
from sigil.routing import Router
from sigil.telemetry import TokenBudget, TokenTracker

async def execute_complex_task(
    user_request: str,
    session_id: str,
):
    """Complete workflow: Route -> Plan -> Execute -> Store."""

    # 1. Initialize components
    router = Router(get_settings())
    planner = Planner()
    executor = PlanExecutor()
    memory = MemoryManager()
    tracker = TokenTracker()
    budget = TokenBudget(max_total_tokens=50000)

    # 2. Route request
    decision = router.route(user_request)

    if decision.use_planning:
        # 3. Retrieve memory context
        context = await memory.retrieve(
            query=user_request,
            mode=RetrievalMode.HYBRID,
            k=10,
        )

        # 4. Generate plan
        plan = await planner.generate(
            goal=user_request,
            context=format_memory(context.items),
            session_id=session_id,
        )

        # 5. Execute plan
        result = await executor.execute(
            plan=plan,
            session_id=session_id,
            token_budget=budget,
        )

        # 6. Store execution to memory
        await memory.store_resource(
            content=format_execution_log(plan, result),
            resource_type="plan_execution",
            metadata={
                "plan_id": plan.plan_id,
                "goal": plan.goal,
                "success": result.success,
            },
        )

        return result.final_output

    else:
        # Simple task - direct reasoning
        reasoning = ReasoningManager(memory_manager=memory)
        result = await reasoning.reason(
            task=user_request,
            complexity=decision.complexity,
            session_id=session_id,
        )

        return result.answer
```

### 13.2 Custom Strategy Selection

```python
async def reason_with_custom_selection(
    task: str,
    prefer_accuracy: bool = False,
    max_tokens: int = 2000,
):
    """Custom strategy selection based on requirements."""
    reasoning = ReasoningManager()

    # Assess complexity
    assessment = await reasoning.assess_complexity(task)

    # Custom selection logic
    if prefer_accuracy:
        # Always use at least CoT
        if assessment.score < 0.3:
            strategy = "cot"
        elif assessment.score < 0.7:
            strategy = "tot"
        else:
            strategy = "mcts"
    else:
        # Budget-conscious selection
        if max_tokens < 500:
            strategy = "direct"
        elif max_tokens < 1000:
            strategy = "cot"
        else:
            strategy = None  # Auto-select

    result = await reasoning.reason(
        task=task,
        complexity=assessment.score,
        strategy_override=strategy,
    )

    return result
```

### 13.3 Plan with Contracts

```python
async def execute_verified_plan(goal: str, session_id: str):
    """Execute plan with contract verification on critical steps."""
    from sigil.contracts import ContractExecutor

    planner = Planner()
    executor = PlanExecutor()
    contracts = ContractExecutor()

    # Generate plan with contract specs
    plan = await planner.generate(
        goal=goal,
        constraints=PlanConstraints(
            require_contracts_for=["tool_call", "final_output"],
        ),
    )

    # Add contracts to critical steps
    for step in plan.steps:
        if step.step_type == StepType.TOOL_CALL:
            step.contract_spec = ContractSpec(
                required_fields=["result"],
                validation_rules=["result is not None"],
            )

    # Execute with contract checking
    result = await executor.execute(
        plan=plan,
        session_id=session_id,
        contract_executor=contracts,
    )

    return result
```

---

## 14. Implementation Guidance

### 14.1 Phase Order

**Phase 5.1 - Reasoning Strategies (Week 1):**
1. Implement `BaseStrategy` ABC
2. Implement `DirectStrategy` (simplest)
3. Implement `CoTStrategy` (step-by-step)
4. Write tests for both strategies

**Phase 5.2 - Additional Strategies (Week 1-2):**
1. Implement `ToTStrategy` (multi-path)
2. Implement `ReActStrategy` (tool loop)
3. Implement `MCTSStrategy` (tree search)
4. Write tests for all strategies

**Phase 5.3 - ReasoningManager (Week 2):**
1. Implement strategy selection logic
2. Implement fallback chain
3. Implement complexity assessment
4. Add caching layer
5. Write integration tests

**Phase 5.4 - Planner (Week 2-3):**
1. Implement `Planner` with LLM decomposition
2. Implement plan validation (DAG check)
3. Implement plan optimization
4. Add plan caching
5. Write tests

**Phase 5.5 - PlanExecutor (Week 3):**
1. Implement dependency resolution
2. Implement parallel execution
3. Implement retry logic
4. Implement checkpointing
5. Write integration tests

**Phase 5.6 - Integration (Week 3-4):**
1. Integrate with Router (Phase 3)
2. Integrate with Memory (Phase 4)
3. Add event emission
4. Add token tracking
5. Write end-to-end tests

### 14.2 Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `anthropic` | >= 0.20.0 | LLM calls |
| `networkx` | >= 3.0 | DAG operations |
| `asyncio` | stdlib | Concurrency |
| `structlog` | >= 24.0 | Structured logging |

### 14.3 File Locations

```
sigil/planning/
|-- __init__.py                 # Exports
|-- planner.py                  # Planner class
|-- executor.py                 # PlanExecutor class
|-- optimizer.py                # PlanOptimizer class
|-- schemas.py                  # Plan, PlanStep dataclasses

sigil/reasoning/
|-- __init__.py                 # Exports
|-- manager.py                  # ReasoningManager class
|-- complexity.py               # ComplexityScorer
|-- cache.py                    # StrategyCache
|-- strategies/
|   |-- __init__.py
|   |-- base.py                 # BaseStrategy ABC
|   |-- direct.py               # DirectStrategy
|   |-- chain_of_thought.py     # CoTStrategy
|   |-- tree_of_thoughts.py     # ToTStrategy
|   |-- react.py                # ReActStrategy
|   |-- mcts.py                 # MCTSStrategy

sigil/orchestration/
|-- __init__.py
|-- orchestrator.py             # Cross-phase orchestration

tests/unit/
|-- test_planning/
|   |-- test_planner.py
|   |-- test_executor.py
|-- test_reasoning/
|   |-- test_strategies.py
|   |-- test_manager.py

tests/integration/
|-- test_phase5_integration.py
|-- test_phase5_memory_integration.py
|-- test_phase5_routing_integration.py
```

### 14.4 Testing Requirements

```
Unit Tests:
- Each strategy in isolation
- Planner generation logic
- Executor dependency resolution
- Cache hit/miss behavior
- Error handling and fallback

Integration Tests:
- Router -> ReasoningManager flow
- Planner -> Executor flow
- Memory retrieval during reasoning
- Event emission completeness
- Token budget enforcement

E2E Tests:
- Full user request to result
- Multi-step plan execution
- Strategy fallback scenarios
- Checkpoint/resume functionality
```

---

## Version History

- **1.0.0** (2026-01-11): Initial API contract
  - Complete Planning subsystem design
  - Complete Reasoning subsystem design
  - Five strategy implementations
  - Cross-phase integration architecture
  - Token budget management
  - Event contract
  - Concurrency design
  - Performance optimization strategies
  - Error handling and resilience
  - Monitoring and observability
  - Implementation guidance
