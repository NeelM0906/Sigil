# Phase 5 Architecture Summary: Planning & Reasoning Integration

## Executive Summary

Phase 5 adds the "brain" of Sigil v2 - two interdependent subsystems that enable intelligent goal decomposition and adaptive reasoning. This document provides a high-level architectural overview for backend and API architects.

---

## System Context

```
+------------------------------------------------------------------+
|                        SIGIL v2 SYSTEM                           |
+------------------------------------------------------------------+
|                                                                  |
|  +--------------+     +--------------+     +--------------+      |
|  |   Phase 3    |     |   Phase 5    |     |   Phase 6    |      |
|  |   Routing    | --> |  Planning &  | --> |  Contracts   |      |
|  |              |     |  Reasoning   |     |              |      |
|  +--------------+     +--------------+     +--------------+      |
|         |                    |                    |              |
|         v                    v                    v              |
|  +--------------+     +--------------+     +--------------+      |
|  |Complexity    |     |Memory Context|     |  Validation  |      |
|  |Assessment    |     |  Retrieval   |     |  & Retry     |      |
|  +--------------+     +--------------+     +--------------+      |
|                              |                                   |
|                              v                                   |
|                       +--------------+                           |
|                       |   Phase 4    |                           |
|                       |    Memory    |                           |
|                       +--------------+                           |
|                                                                  |
+------------------------------------------------------------------+
```

---

## Core Components

### 1. Planning Subsystem

**Purpose**: Decompose high-level goals into executable DAG-structured plans

```
Goal: "Qualify lead John from Acme Corp"
            |
            v
    +-------+-------+
    |    Planner    |  <- LLM decomposes goal
    +-------+-------+
            |
            v
    +-------+-------+
    |     Plan      |
    |  (DAG Steps)  |
    +-------+-------+
            |
    +-------+-------+-------+
    |       |       |       |
    v       v       v       v
  Step1   Step2   Step3   Step4
  (RAG)   (CoT)  (tool)  (CoT)
```

**Key Classes**:
- `Planner`: Generates plans from goals using LLM
- `PlanExecutor`: Executes plans with dependency management
- `Plan`: Immutable DAG of `PlanStep` objects
- `PlanStep`: Individual executable unit with dependencies

**Characteristics**:
- Plan generation: 3-8 seconds
- Plan storage: ~1KB per plan
- Parallel execution: Up to 3 concurrent steps
- Checkpointing: Resume after failure

### 2. Reasoning Subsystem

**Purpose**: Execute reasoning tasks with complexity-adaptive strategy selection

```
        Task + Complexity Score
                |
                v
    +-----------+-----------+
    |   ReasoningManager    |
    +-----------+-----------+
                |
    +-----------+-----------+-----------+-----------+
    |           |           |           |           |
    v           v           v           v           v
 Direct       CoT         ToT       ReAct       MCTS
(0-0.3)    (0.3-0.5)   (0.5-0.7)  (0.7-0.9)  (0.9-1.0)
```

**Strategy Selection by Complexity**:

| Complexity | Strategy | Tokens | Latency | Use Case |
|------------|----------|--------|---------|----------|
| 0.0 - 0.3 | Direct | 100-300 | 1-2s | Factual queries |
| 0.3 - 0.5 | CoT | 300-800 | 2-4s | Step-by-step |
| 0.5 - 0.7 | ToT | 800-2000 | 4-8s | Ambiguity |
| 0.7 - 0.9 | ReAct | 1000-3000 | 5-15s | Tool-heavy |
| 0.9 - 1.0 | MCTS | 2000-5000 | 10-30s | Critical |

**Fallback Chain**: MCTS -> ReAct -> ToT -> CoT -> Direct

---

## Integration Points

### Phase 3 (Routing) Integration

```python
# Router output feeds Phase 5
RouteDecision(
    complexity=0.72,      # -> ReasoningManager.select_strategy()
    use_planning=True,    # -> Triggers Planner
    use_memory=True,      # -> Memory context retrieval
)
```

**Integration Rules**:
1. Router's `ComplexityAssessor` output determines strategy selection
2. `use_planning=True` when complexity > 0.5 AND feature enabled
3. Router can override strategy for specific intents (e.g., /commands -> Direct)

### Phase 4 (Memory) Integration

**Planning uses Memory for**:
- Context retrieval before plan generation
- Fact grounding during reasoning
- Storage of plan execution results
- Insight extraction from successful plans

```python
# Memory integration during planning
context = await memory_manager.retrieve(goal, k=10)
plan = await planner.generate(goal, context=format_memory(context))

# Memory integration during reasoning
facts = await memory_manager.retrieve(task, k=5)
result = await strategy.execute(task, context=ReasoningContext(facts=facts))

# Store execution results
await memory_manager.store_resource(execution_log, type="plan_execution")
```

### Phase 6 (Contracts) Integration

**Forward Reference**:
- Plan steps can include `ContractSpec` for output validation
- Contract failures trigger step retry or plan alternative
- ReasoningManager results include contract validation status

---

## Token Budget Architecture

### Session Budget Allocation (Default 50,000 tokens)

```
+-------------------+
|  Total: 50,000    |
+-------------------+
|                   |
| Planning: 15%     |  7,500 tokens
| (7,500)           |    - Plan generation: ~1,500/plan
|                   |    - Context per step: ~500
+-------------------+
|                   |
| Reasoning: 50%    |  25,000 tokens
| (25,000)          |    - Direct: ~150
|                   |    - CoT: ~400
|                   |    - ToT: ~1,000
|                   |    - ReAct: ~800
|                   |    - MCTS: ~2,000
+-------------------+
|                   |
| Memory: 20%       |  10,000 tokens
| (10,000)          |    - Retrieval: ~200/query
|                   |    - Extraction: ~500/resource
+-------------------+
|                   |
| Overhead: 15%     |  7,500 tokens
| (7,500)           |    - Routing: ~100
|                   |    - Events: ~50
+-------------------+
```

### Graceful Degradation

```
Remaining Tokens    Action
----------------    ------
< 500               Force Direct strategy only
< 1000              Force Direct or CoT only
< 2000              Exclude MCTS
80% used            Warning threshold
95% used            Critical warning
100%                Return best result so far
```

---

## Event Sourcing

### Planning Events

```
PlanCreatedEvent(plan_id, goal, step_count)
     |
     v
PlanExecutionStartedEvent(plan_id)
     |
     +---> PlanStepStartedEvent(plan_id, step_id)
     |           |
     |           v
     |     PlanStepCompletedEvent(plan_id, step_id, result)
     |     OR
     |     PlanStepFailedEvent(plan_id, step_id, error, retry)
     |
     v
PlanCompletedEvent(plan_id, total_tokens)
OR
PlanPausedEvent(plan_id, checkpoint)
```

### Reasoning Events

```
ReasoningTaskStartedEvent(task_hash, complexity)
     |
     v
StrategySelectedEvent(strategy, complexity, reason)
     |
     v
ReasoningCompletedEvent(strategy, confidence, tokens)
OR
StrategyFallbackEvent(original, fallback, reason)
```

---

## Concurrency Model

### Planning Concurrency

```
Plan Steps:
  Step1 --+
          |-- [Parallel Wave 1, max 3]
  Step2 --+
          |
  Step3 <-+ (depends on 1,2)
          |
  Step4 <-+ (depends on 3)
          |
  Step5 --+
          |-- [Parallel Wave 2, max 3]
  Step6 --+
```

- Maximum 3 concurrent plan steps
- Dependency-aware scheduling via topological sort
- Checkpoint after each step completion

### Reasoning Concurrency

| Strategy | Concurrency | Notes |
|----------|-------------|-------|
| Direct | 1 | Single call |
| CoT | 1 | Single call |
| ToT | 5 | Parallel path exploration |
| ReAct | 1 | Sequential tool calls |
| MCTS | 10 | Parallel simulations |

**Total Concurrency Cap**: 20 concurrent LLM/tool operations

---

## Error Handling Matrix

| Failure | Recovery | Guarantee |
|---------|----------|-----------|
| Plan generation fails | Return empty plan, ask for steps | Always return |
| Dependency cycle | Regenerate without cycles | Valid DAG |
| Step execution fails | Retry 3x, then skip | Continue |
| Strategy fails | Fallback chain | DirectStrategy final |
| Budget exceeded | Return best result | Partial result |
| Timeout | Return best result | Flagged partial |
| LLM error | Retry once, fallback | Best effort |

---

## Performance Targets

| Operation | Target | P95 |
|-----------|--------|-----|
| Plan generation (cold) | 5s | 8s |
| Plan generation (cached) | 100ms | 200ms |
| Direct reasoning | 1.5s | 2.5s |
| CoT reasoning | 3s | 5s |
| ToT reasoning | 6s | 10s |
| ReAct reasoning | 10s | 15s |
| MCTS reasoning | 20s | 35s |

---

## Caching Strategy

| Cache Type | TTL | Purpose |
|------------|-----|---------|
| Complexity scores | 60s | Avoid re-assessment |
| Similar goals | 24h | Reuse plans |
| Strategy effectiveness | 7d | Learning |
| LLM responses | 1h | Identical prompts |

**Cache Exclusions**:
- MCTS results (too specific)
- Tool call results (state changes)
- Memory-dependent reasoning (context changes)

---

## API Surface Summary

### Planner API

```python
class Planner:
    async def generate(goal, context?, constraints?) -> Plan
    def validate_plan(plan) -> ValidationResult
    async def optimize_plan(plan, goals?) -> Plan
```

### PlanExecutor API

```python
class PlanExecutor:
    async def execute(plan, session_id, token_budget?) -> ExecutionResult
    async def resume(plan_id, session_id) -> ExecutionResult
    async def pause(plan_id) -> bool
```

### ReasoningManager API

```python
class ReasoningManager:
    async def reason(task, complexity?, override?, context?) -> StrategyResult
    def select_strategy(complexity, task_type?) -> BaseStrategy
    async def assess_complexity(task, context?) -> ComplexityAssessment
```

### Strategy Interface

```python
class BaseStrategy(ABC):
    name: str
    max_tokens: int

    @abstractmethod
    async def execute(task, context?, budget?) -> StrategyResult
```

---

## Implementation Phases

```
Week 1:
  [x] DirectStrategy
  [x] CoTStrategy
  [ ] ToTStrategy

Week 1-2:
  [ ] ReActStrategy
  [ ] MCTSStrategy
  [ ] ReasoningManager

Week 2-3:
  [ ] Planner
  [ ] PlanExecutor
  [ ] Plan validation

Week 3-4:
  [ ] Phase 3 integration
  [ ] Phase 4 integration
  [ ] Token tracking
  [ ] Event emission
  [ ] End-to-end tests
```

---

## Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Strategy selection | Complexity-based | Optimize tokens vs capability |
| Plan structure | DAG | Enables parallelization |
| Fallback behavior | Chain down | Always produce result |
| Token tracking | Per-subsystem | Fine-grained budget control |
| Concurrency | Bounded | Prevent resource exhaustion |
| Caching | Multi-tier | Balance freshness vs speed |
| Events | Fine-grained | Full audit trail |

---

## Files Created

- `/docs/api-contract-phase5-planning-reasoning.md` - Full API contract (1800+ lines)
- `/docs/phase5-architecture-summary.md` - This summary document

---

## Next Steps for Backend Team

1. **Start with Strategies**: Implement DirectStrategy and CoTStrategy first
2. **Build ReasoningManager**: Add strategy selection and fallback
3. **Implement Planner**: LLM-based goal decomposition
4. **Add PlanExecutor**: Dependency resolution and parallel execution
5. **Integrate with Phase 3/4**: Memory context, routing decisions
6. **Add observability**: Events, metrics, logging

---

## Phase 7 Integration: Orchestrator

Phase 5 components (Planner, PlanExecutor, ReasoningManager) are integrated into the unified
`SigilOrchestrator` introduced in Phase 7. The orchestrator serves as the central entry point
for all request processing and coordinates Phase 5 components as follows:

### Orchestrator Integration Points

```
┌─────────────────────────────────────────────────────────────────┐
│                     SigilOrchestrator                           │
│                                                                 │
│  Request → ContextManager → Router → [Phase 5 Components]       │
│                                                                 │
│  Phase 5 Integration:                                           │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                                                         │   │
│  │  1. Router provides complexity score to orchestrator    │   │
│  │  2. Orchestrator invokes ReasoningManager.reason()      │   │
│  │  3. For complex tasks (complexity > 0.5):               │   │
│  │     - Orchestrator invokes Planner.generate()           │   │
│  │     - PlanExecutor.execute() runs plan steps            │   │
│  │  4. Results flow back through orchestrator              │   │
│  │                                                         │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Key Integration Methods

| Phase 5 Component | Orchestrator Method | Trigger |
|-------------------|---------------------|---------|
| ReasoningManager | `_execute_reasoning()` | All requests |
| Planner | `_generate_plan()` | complexity > 0.5 |
| PlanExecutor | `_execute_plan()` | Plan exists |

### Token Budget Coordination

The orchestrator allocates tokens to Phase 5 components from the total 256K budget:

- **Reasoning**: Up to 50% (128K tokens) depending on strategy
- **Planning**: Up to 15% (38.4K tokens) for plan generation
- **Per-step execution**: Tracked and enforced by orchestrator

### Event Flow

Phase 5 events are now emitted through the orchestrator's event pipeline:

```
OrchestratorRequestReceived
    └── ReasoningTaskStartedEvent
        └── StrategySelectedEvent
        └── [Strategy-specific events]
        └── ReasoningCompletedEvent
    └── PlanCreatedEvent (if planning)
        └── PlanStepStartedEvent (per step)
        └── PlanStepCompletedEvent (per step)
        └── PlanCompletedEvent
    └── OrchestratorResponseSent
```

### Related Documentation

- **Phase 7 Architecture**: `/docs/phase7-integration-architecture.md`
- **Phase 7 API Contract**: `/docs/api-contract-phase7-integration.md`
- **Phase 7 OpenAPI Spec**: `/docs/openapi-phase7-integration.yaml`
- **Phase 7 API Guidelines**: `/docs/api-guidelines-phase7.md`

---

*Document Version: 1.1.0*
*Last Updated: 2026-01-11*
