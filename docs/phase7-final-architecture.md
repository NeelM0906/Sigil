# Sigil v2 - Phase 7 Final Architecture Document

**Version:** 2.0.0
**Date:** 2026-01-11
**Status:** Integration & Polish Phase (Phase 7)
**Author:** Systems Architecture Review

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Architecture Overview](#2-architecture-overview)
3. [Implementation Status Review](#3-implementation-status-review)
4. [Core Subsystem Analysis](#4-core-subsystem-analysis)
5. [Integration Layer Analysis](#5-integration-layer-analysis)
6. [Data Flow Architecture](#6-data-flow-architecture)
7. [API Architecture](#7-api-architecture)
8. [Performance Architecture](#8-performance-architecture)
9. [Security Architecture](#9-security-architecture)
10. [Scalability Analysis](#10-scalability-analysis)
11. [Gap Analysis](#11-gap-analysis)
12. [Risk Assessment](#12-risk-assessment)
13. [Recommendations](#13-recommendations)
14. [Appendices](#14-appendices)

---

## 1. Executive Summary

### 1.1 Project Overview

Sigil v2 is a self-improving agent framework built on ACTi (Actualized Collective Transformational Intelligence) methodology. The framework enables creation of executable AI agents with:

- **Persistent Memory**: 3-layer architecture for knowledge retention and retrieval
- **Hierarchical Planning**: DAG-based task decomposition with topological validation
- **Multi-Strategy Reasoning**: 5 reasoning strategies selected by complexity (0.0-1.0)
- **Contract Verification**: Formal output specification with retry/fallback mechanisms
- **Event-Sourced State**: Full audit trail with replay capability
- **Token Budget Management**: Provider-agnostic cost control (256K token budget)

### 1.2 Implementation Progress Summary

| Phase | Component | Status | Completeness |
|-------|-----------|--------|--------------|
| Phase 3 | Memory System | **COMPLETE** | 100% |
| Phase 4 | Routing Layer | **COMPLETE** | 100% |
| Phase 5 | Planning & Reasoning | **COMPLETE** | 100% |
| Phase 6 | Contracts System | **COMPLETE** | 100% |
| Phase 6 | Integration Bridges | **COMPLETE** | 100% |
| Phase 7 | Context Manager | **NOT STARTED** | 0% |
| Phase 7 | Evolution Manager | **NOT STARTED** | 0% |
| Phase 7 | FastAPI Server | **NOT STARTED** | 0% |

### 1.3 Key Findings

**Strengths:**
1. Solid foundation with well-structured core modules
2. Complete contract system with comprehensive validation
3. All four integration bridges implemented (router, memory, plan, reasoning)
4. Robust exception hierarchy covering all error scenarios
5. Provider-agnostic token tracking system

**Gaps Requiring Attention:**
1. Context Manager (context assembly, compression) not implemented
2. Evolution Manager (self-improvement, TextGrad) not implemented
3. FastAPI REST API server not implemented
4. WebSocket streaming not implemented
5. SigilOrchestrator (unified orchestration) not implemented

### 1.4 256K Token Budget Validation

The system is designed to support a 256K token budget with:
- `TokenBudget` class with validation (max_total_tokens configurable)
- `TokenTracker` for cumulative usage tracking
- `ContractConstraints` with warn_threshold (default 0.8)
- Token remaining calculation in retry logic

**Verification Status:** The token tracking infrastructure supports 256K but requires integration testing with actual LLM providers to validate end-to-end budget enforcement.

---

## 2. Architecture Overview

### 2.1 High-Level System Architecture

```
                                    SIGIL v2 ARCHITECTURE
 ==================================================================================================

                                     USER INTERFACES
 +------------------------------------------------------------------------------------------------+
 |   CLI (Implemented)    |    REST API (TODO)     |    WebSocket (TODO)    |    SDK (TODO)      |
 +------------------------------------------------------------------------------------------------+
                                          |
                                          v
                                   ROUTING LAYER
 +------------------------------------------------------------------------------------------------+
 |                                                                                                |
 |   IntentClassifier                ComplexityAssessor                  Router                  |
 |   (Keyword + Regex)               (4-Factor Analysis)              (Handler Selection)        |
 |                                                                                                |
 |   Intents: CREATE_AGENT | RUN_AGENT | QUERY_MEMORY | MODIFY_AGENT | SYSTEM | GENERAL_CHAT     |
 +------------------------------------------------------------------------------------------------+
                                          |
                                          v
                              INTEGRATION BRIDGES (Phase 6)
 +------------------------------------------------------------------------------------------------+
 |                                                                                                |
 |   ContractSelector           ContractMemoryBridge        PlanContractBridge                   |
 |   (router_bridge.py)         (memory_bridge.py)          (plan_bridge.py)                     |
 |                                                                                                |
 |                          ContractAwareReasoningManager                                         |
 |                              (reasoning_bridge.py)                                             |
 |                                                                                                |
 +------------------------------------------------------------------------------------------------+
                                          |
                                          v
                                ORCHESTRATION LAYER
 +------------------------------------------------------------------------------------------------+
 |                                                                                                |
 |   +------------------+     +------------------+     +------------------+                       |
 |   |     Planner      |     |ReasoningManager  |     | ContextManager   |  (TODO)              |
 |   |                  |     |                  |     |                  |                       |
 |   | - Plan generation|     | - Strategy select|     | - Context assem. |                       |
 |   | - DAG validation |     | - 5 strategies   |     | - Compression    |                       |
 |   | - Caching (24h)  |     | - Fallback chain |     | - Token-aware    |                       |
 |   +------------------+     +------------------+     +------------------+                       |
 |                                                                                                |
 +------------------------------------------------------------------------------------------------+
                                          |
                                          v
                                 EXECUTION LAYER
 +------------------------------------------------------------------------------------------------+
 |                                                                                                |
 |   +------------------+     +------------------+     +------------------+                       |
 |   | ContractExecutor |     |    Agents        |     |     Tools        |                       |
 |   |                  |     |                  |     |                  |                       |
 |   | - Validation     |     | - Builder        |     | - MCP Client     |                       |
 |   | - Retry logic    |     | - Executor       |     | - Built-in tools |                       |
 |   | - Fallback mgmt  |     | - Validator      |     | - Memory tools   |                       |
 |   +------------------+     +------------------+     +------------------+                       |
 |                                                                                                |
 +------------------------------------------------------------------------------------------------+
                                          |
                                          v
                                PERSISTENCE LAYER
 +------------------------------------------------------------------------------------------------+
 |                                                                                                |
 |   +------------------+     +------------------+     +------------------+                       |
 |   |  MemoryManager   |     |    EventStore    |     |EvolutionManager  |  (TODO)              |
 |   |                  |     |                  |     |                  |                       |
 |   | - 3-layer memory |     | - JSON file store|     | - TextGrad opt.  |                       |
 |   | - RAG/LLM/Hybrid |     | - File locking   |     | - Eval/evolve    |                       |
 |   | - Consolidation  |     | - Event replay   |     | - Versioning     |                       |
 |   +------------------+     +------------------+     +------------------+                       |
 |                                                                                                |
 +------------------------------------------------------------------------------------------------+
                                          |
                                          v
                                  INFRASTRUCTURE
 +------------------------------------------------------------------------------------------------+
 |                                                                                                |
 |   TokenTracker    |    SigilSettings    |    Exception Hierarchy    |    Logging/Telemetry   |
 |                                                                                                |
 +------------------------------------------------------------------------------------------------+
```

### 2.2 Directory Structure Analysis

```
sigil/
|
+-- core/                          # COMPLETE - Foundation layer
|   +-- base.py                    # BaseAgent, BaseStrategy, BaseRetriever
|   +-- exceptions.py              # 15 exception classes, full hierarchy
|
+-- config/                        # COMPLETE - Configuration management
|   +-- settings.py                # Pydantic settings, feature flags
|   +-- schemas/                   # Pydantic models for all data types
|       +-- agent.py
|       +-- memory.py
|       +-- plan.py
|       +-- contract.py
|       +-- events.py
|
+-- memory/                        # COMPLETE - 3-layer memory system
|   +-- manager.py                 # MemoryManager orchestration
|   +-- layers/
|   |   +-- resources.py           # Layer 1: Raw data
|   |   +-- items.py               # Layer 2: Extracted facts
|   |   +-- categories.py          # Layer 3: Aggregated knowledge
|   +-- extraction.py              # Item extraction from resources
|   +-- retrieval.py               # RAG + LLM retrieval
|   +-- consolidation.py           # Category aggregation
|   +-- templates/
|       +-- acti.py                # ACTi-specific memory templates
|
+-- routing/                       # COMPLETE - Intent-based routing
|   +-- router.py                  # Router, IntentClassifier, ComplexityAssessor
|
+-- planning/                      # COMPLETE - Task decomposition
|   +-- planner.py                 # Plan generation, DAG validation
|   +-- executor.py                # Plan execution
|   +-- schemas.py                 # PlanStepConfig, StepType
|
+-- reasoning/                     # COMPLETE - Multi-strategy reasoning
|   +-- manager.py                 # ReasoningManager, strategy selection
|   +-- prompts.py                 # Strategy-specific prompts
|   +-- strategies/
|       +-- base.py                # BaseStrategy ABC
|       +-- direct.py              # Direct (0.0-0.3)
|       +-- chain_of_thought.py    # CoT (0.3-0.5)
|       +-- tree_of_thoughts.py    # ToT (0.5-0.7)
|       +-- react.py               # ReAct (0.7-0.9)
|       +-- mcts.py                # MCTS (0.9-1.0)
|
+-- contracts/                     # COMPLETE - Output verification
|   +-- schema.py                  # Contract, Deliverable, FailureStrategy
|   +-- validator.py               # ContractValidator
|   +-- executor.py                # ContractExecutor
|   +-- retry.py                   # RetryManager
|   +-- fallback.py                # FallbackManager, FallbackStrategy
|   +-- templates/
|   |   +-- acti.py                # 5 ACTi contract templates
|   +-- integration/               # COMPLETE - Phase 6 integration bridges
|       +-- router_bridge.py       # ContractSelector
|       +-- memory_bridge.py       # ContractMemoryBridge
|       +-- plan_bridge.py         # PlanContractBridge
|       +-- reasoning_bridge.py    # ContractAwareReasoningManager
|
+-- context/                       # NOT IMPLEMENTED - Placeholder only
|   +-- __init__.py                # TODO: ContextManager, ContextBuilder
|
+-- evolution/                     # NOT IMPLEMENTED - Placeholder only
|   +-- __init__.py                # TODO: EvolutionEngine
|   +-- optimizers/
|       +-- __init__.py            # TODO: TextGradOptimizer
|
+-- state/                         # COMPLETE - Event-sourced state
|   +-- events.py                  # Event types, factory functions
|   +-- store.py                   # EventStore with file locking
|   +-- session.py                 # Session management
|
+-- telemetry/                     # COMPLETE - Cost tracking
|   +-- tokens.py                  # TokenBudget, TokenTracker, TokenMetrics
|
+-- interfaces/                    # PARTIAL
|   +-- cli/                       # Exists (from v1)
|   +-- api/
|       +-- __init__.py            # TODO: FastAPI server
|
+-- tools/                         # PARTIAL
|   +-- mcp/                       # MCP client integration
|   +-- builtin/
|       +-- memory_tools.py        # Memory operations
|       +-- planning_tools.py      # Planning operations
|
+-- tests/                         # COMPLETE - Test coverage
    +-- memory/                    # 6 test files
    +-- planning/                  # 3 test files
    +-- reasoning/                 # 7 test files
    +-- contracts/                 # 6 test files
```

---

## 3. Implementation Status Review

### 3.1 Phase 3: Memory System (COMPLETE)

**Files Reviewed:**
- `/sigil/memory/manager.py` (441 lines)
- `/sigil/memory/layers/resources.py`
- `/sigil/memory/layers/items.py`
- `/sigil/memory/layers/categories.py`
- `/sigil/memory/retrieval.py`
- `/sigil/memory/consolidation.py`

**Implementation Assessment:**

| Component | Status | Quality | Notes |
|-----------|--------|---------|-------|
| ResourceLayer | Complete | High | Raw data storage with metadata |
| ItemLayer | Complete | High | Extracted facts with embeddings |
| CategoryLayer | Complete | High | Aggregated markdown knowledge |
| MemoryManager | Complete | High | Full orchestration |
| Extraction | Complete | High | LLM-based fact extraction |
| Retrieval | Complete | High | RAG/LLM/Hybrid modes |
| Consolidation | Complete | High | Category aggregation |

**3-Layer Architecture:**
```
+-------------------------+
|  LAYER 3: CATEGORIES    |  <- Aggregated knowledge (markdown files)
|  lead_preferences.md    |     Human/LLM readable summaries
+-------------------------+
          ^ aggregates
+-------------------------+
|  LAYER 2: ITEMS         |  <- Discrete facts with embeddings
|  MemoryItem with        |     RAG-searchable via vectors
|  source_resource_id     |     Links to source for traceability
+-------------------------+
          ^ extracts from
+-------------------------+
|  LAYER 1: RESOURCES     |  <- Raw source data
|  Conversations, docs,   |     Full content preserved
|  configs, transcripts   |
+-------------------------+
```

**Retrieval Modes:**
1. **RAG** (fast): Embedding-based vector search for simple queries
2. **LLM** (accurate): Full reading for complex reasoning
3. **Hybrid**: Start RAG, escalate to LLM if confidence < threshold

### 3.2 Phase 4: Routing Layer (COMPLETE)

**File Reviewed:** `/sigil/routing/router.py` (688 lines)

**Implementation Assessment:**

| Component | Status | Quality | Notes |
|-----------|--------|---------|-------|
| Intent Enum | Complete | High | 6 intent types defined |
| IntentClassifier | Complete | High | Keyword + regex matching |
| ComplexityAssessor | Complete | High | 4-factor analysis |
| RouteDecision | Complete | High | Full routing decision model |
| Router | Complete | High | Handler selection + flags |

**Intent Classification:**
```python
class Intent(Enum):
    CREATE_AGENT = "create_agent"    # -> "builder" handler
    RUN_AGENT = "run_agent"          # -> "executor" handler
    QUERY_MEMORY = "query_memory"    # -> "memory_query" handler
    MODIFY_AGENT = "modify_agent"    # -> "agent_modifier" handler
    SYSTEM_COMMAND = "system_command"# -> "system_command" handler
    GENERAL_CHAT = "general_chat"    # -> "chat" handler
```

**Complexity Assessment (4 Factors):**
1. Message length (0.25 weight) - Normalized to MAX_LENGTH=500
2. Tool requirements (0.25 weight) - 20+ tool keywords
3. Domain specificity (0.25 weight) - 30+ domain keywords
4. Decision complexity (0.25 weight) - 25+ decision keywords

**Subsystem Activation Rules:**
- Planning: `complexity > 0.5 AND use_planning=True`
- Memory: `intent in MEMORY_RELEVANT_INTENTS AND use_memory=True`
- Contracts: `complexity > 0.7 AND use_contracts=True`

### 3.3 Phase 5: Planning & Reasoning (COMPLETE)

**Planning Files Reviewed:**
- `/sigil/planning/planner.py`
- `/sigil/planning/executor.py`
- `/sigil/planning/schemas.py`

**Reasoning Files Reviewed:**
- `/sigil/reasoning/manager.py`
- `/sigil/reasoning/strategies/*.py`

**Planning Implementation:**

| Component | Status | Quality | Notes |
|-----------|--------|---------|-------|
| Planner | Complete | High | Plan generation with caching |
| PlanExecutor | Complete | High | DAG-based execution |
| DAG Validation | Complete | High | Kahn's algorithm |
| Plan Caching | Complete | High | 24-hour TTL |

**DAG Validation via Kahn's Algorithm:**
```python
def validate_dag(self, steps: list[PlanStepConfig]) -> bool:
    """
    Uses Kahn's algorithm for topological sorting.
    Detects circular dependencies by checking if all
    steps can be processed when in-degree reaches 0.
    Returns False if cycle detected.
    """
```

**Reasoning Implementation:**

| Strategy | Complexity Range | Status | Use Case |
|----------|-----------------|--------|----------|
| Direct | 0.0 - 0.3 | Complete | Simple queries, lookups |
| Chain-of-Thought | 0.3 - 0.5 | Complete | Step-by-step reasoning |
| Tree-of-Thoughts | 0.5 - 0.7 | Complete | Multiple valid paths |
| ReAct | 0.7 - 0.9 | Complete | Tool interaction loops |
| MCTS | 0.9 - 1.0 | Complete | Critical decisions |

**Strategy Selection:**
```python
STRATEGY_RANGES = {
    "direct": (0.0, 0.3),
    "chain_of_thought": (0.3, 0.5),
    "tree_of_thoughts": (0.5, 0.7),
    "react": (0.7, 0.9),
    "mcts": (0.9, 1.0),
}
```

**Fallback Chain:** MCTS -> ReAct -> ToT -> CoT -> Direct

### 3.4 Phase 6: Contracts System (COMPLETE)

**Files Reviewed:**
- `/sigil/contracts/schema.py`
- `/sigil/contracts/validator.py`
- `/sigil/contracts/executor.py`
- `/sigil/contracts/retry.py`
- `/sigil/contracts/fallback.py`
- `/sigil/contracts/templates/acti.py`

**Core Contract Components:**

| Component | Status | Quality | Notes |
|-----------|--------|---------|-------|
| Contract Schema | Complete | High | Full specification model |
| Deliverable | Complete | High | Field spec with validation |
| ContractConstraints | Complete | High | Token/tool/timeout limits |
| FailureStrategy | Complete | High | 6 strategies defined |
| ContractValidator | Complete | High | Type + rule validation |
| ContractExecutor | Complete | High | Full execution orchestration |
| RetryManager | Complete | High | Prompt refinement |
| FallbackManager | Complete | High | 5 fallback strategies |

**FailureStrategy Enum:**
```python
class FailureStrategy(str, Enum):
    RETRY = "retry"       # Retry with refined prompt
    FALLBACK = "fallback" # Use fallback output
    PARTIAL = "partial"   # Return partial output
    TEMPLATE = "template" # Use template defaults
    ESCALATE = "escalate" # Raise for human review
    FAIL = "fail"         # Raise ContractValidationError
```

**Contract Execution Flow:**
```
1. Execute agent to produce output
2. Validate output against contract deliverables
3. IF valid: Return SUCCESS result
4. IF invalid AND strategy != FAIL:
   a. IF retries_remaining AND tokens_remaining:
      - Refine prompt with errors
      - Re-execute (loop to step 1)
   b. ELSE: Apply fallback strategy
5. IF invalid AND strategy == FAIL:
   - Raise ContractValidationError
```

**ACTi Contract Templates (5):**
1. `lead_qualification` - BANT methodology scoring
2. `research_report` - Structured findings report
3. `appointment_booking` - Calendar event creation
4. `market_analysis` - Competitive analysis
5. `compliance_check` - Governance validation

### 3.5 Phase 6: Integration Bridges (COMPLETE)

**Files Reviewed:**
- `/sigil/contracts/integration/router_bridge.py` (381 lines)
- `/sigil/contracts/integration/memory_bridge.py` (610 lines)
- `/sigil/contracts/integration/plan_bridge.py` (762 lines)
- `/sigil/contracts/integration/reasoning_bridge.py` (623 lines)

**Router Bridge (ContractSelector):**

| Feature | Status | Notes |
|---------|--------|-------|
| Intent-to-Contract mapping | Complete | INTENT_CONTRACT_MAP |
| Stratum-to-Contract mapping | Complete | STRATUM_CONTRACT_MAP |
| Complexity-based adjustment | Complete | +/- retries, token budget |
| Failure strategy recommendation | Complete | Based on complexity |

**Stratum-to-Contract Mapping:**
```python
STRATUM_CONTRACT_MAP = {
    "RTI": "research_report",      # Reality & Truth Intelligence
    "RAI": "lead_qualification",   # Readiness & Agreement Intelligence
    "ZACS": "appointment_booking", # Zone Action & Conversion Systems
    "EEI": "market_analysis",      # Economic & Ecosystem Intelligence
    "IGE": "compliance_check",     # Integrity & Governance Engine
}
```

**Memory Bridge (ContractMemoryBridge):**

| Feature | Status | Notes |
|---------|--------|-------|
| Contract template storage | Complete | Versioned storage |
| Validation result storage | Complete | For pattern learning |
| Exemplar output storage | Complete | Few-shot learning |
| Error pattern analysis | Complete | Common error retrieval |
| Success rate tracking | Complete | Historical metrics |

**Plan Bridge (PlanContractBridge):**

| Feature | Status | Notes |
|---------|--------|-------|
| ContractSpec (lightweight) | Complete | Per-step contracts |
| Contract attachment rules | Complete | Final step, tool calls |
| Step execution with contract | Complete | StepContractResult |
| Violation handling | Complete | Replan/skip/abort |

**Reasoning Bridge (ContractAwareReasoningManager):**

| Feature | Status | Notes |
|---------|--------|-------|
| Contract-aware strategy selection | Complete | Complexity boost |
| Contract-aware prompt building | Complete | Output format spec |
| Common error integration | Complete | Warnings in prompt |
| Strategy recommendation | Complete | With explanation |

**Complexity Boost Rules:**
- FAIL strategy: +0.2 complexity
- Many deliverables (>5): +0.1 complexity
- Complex rules (>10 total): +0.1 complexity

### 3.6 State Management (COMPLETE)

**Files Reviewed:**
- `/sigil/state/events.py`
- `/sigil/state/store.py`
- `/sigil/state/session.py`

**Event Types Implemented:**
```python
class EventType(str, Enum):
    # Session lifecycle
    SESSION_STARTED = "session_started"
    SESSION_ENDED = "session_ended"

    # Message events
    MESSAGE_ADDED = "message_added"
    MESSAGE_UPDATED = "message_updated"

    # Agent events
    AGENT_STARTED = "agent_started"
    AGENT_COMPLETED = "agent_completed"

    # Tool events
    TOOL_CALLED = "tool_called"
    TOOL_RESULT = "tool_result"

    # Memory events
    MEMORY_EXTRACTED = "memory_extracted"
    MEMORY_RETRIEVED = "memory_retrieved"
    MEMORY_CONSOLIDATED = "memory_consolidated"

    # Planning events
    PLAN_CREATED = "plan_created"
    PLAN_STEP_STARTED = "plan_step_started"
    PLAN_STEP_COMPLETED = "plan_step_completed"

    # Reasoning events
    REASONING_STARTED = "reasoning_started"
    REASONING_COMPLETED = "reasoning_completed"

    # Contract events
    CONTRACT_VALIDATED = "contract_validated"
    CONTRACT_RETRIED = "contract_retried"
    CONTRACT_FALLBACK = "contract_fallback"

    # Error events
    ERROR_OCCURRED = "error_occurred"

    # Evolution events
    EVOLUTION_STARTED = "evolution_started"
    EVOLUTION_COMPLETED = "evolution_completed"
```

**EventStore Implementation:**
- JSON file-based storage
- File locking via `portalocker` for concurrency
- Append-only design for audit trail
- Event replay capability for state reconstruction

### 3.7 Token Tracking (COMPLETE)

**File Reviewed:** `/sigil/telemetry/tokens.py`

**Components:**

| Class | Purpose | Status |
|-------|---------|--------|
| TokenBudget | Immutable budget config | Complete |
| TokenTracker | Cumulative usage tracking | Complete |
| TokenMetrics | Per-call detailed metrics | Complete |

**TokenBudget Validation:**
```python
@dataclass(frozen=True)
class TokenBudget:
    max_input_tokens: Optional[int] = None
    max_output_tokens: Optional[int] = None
    max_total_tokens: Optional[int] = None
    warn_threshold: float = 0.8  # 80% warning

    def __post_init__(self):
        if self.max_total_tokens is not None:
            if self.max_total_tokens <= 0:
                raise ValueError("max_total_tokens must be positive")
```

**256K Token Support:**
```python
# Settings configuration
TokenBudget(max_total_tokens=256000)  # 256K tokens

# Usage tracking
tracker = TokenTracker(budget=budget)
tracker.add_tokens(input_tokens=1000, output_tokens=500)
remaining = budget.get_remaining_tokens(tracker.total_tokens)
```

### 3.8 Configuration System (COMPLETE)

**File Reviewed:** `/sigil/config/settings.py`

**Feature Flags:**
```python
class SigilSettings(BaseSettings):
    # Feature flags
    use_memory: bool = False     # SIGIL_USE_MEMORY
    use_planning: bool = False   # SIGIL_USE_PLANNING
    use_contracts: bool = False  # SIGIL_USE_CONTRACTS
    use_evolution: bool = False  # SIGIL_USE_EVOLUTION
    use_routing: bool = True     # SIGIL_USE_ROUTING

    # LLM settings
    llm: LLMSettings = Field(default_factory=LLMSettings)

    # Memory settings
    memory: MemorySettings = Field(default_factory=MemorySettings)

    # Contract settings
    contracts: ContractSettings = Field(default_factory=ContractSettings)

    def get_active_features(self) -> list[str]:
        """Return list of enabled feature names."""
```

### 3.9 Exception Hierarchy (COMPLETE)

**File Reviewed:** `/sigil/core/exceptions.py`

**Exception Classes (15):**
```
SigilError (base)
|
+-- ConfigurationError
|   +-- FeatureDisabledError
|
+-- AgentError
|   +-- AgentNotFoundError
|   +-- AgentExecutionError
|
+-- MemoryError
|   +-- MemoryStorageError
|   +-- MemoryRetrievalError
|
+-- PlanningError
|   +-- PlanExecutionError
|   +-- CyclicDependencyError
|
+-- ContractValidationError
|
+-- TokenBudgetExceeded
|
+-- ToolError
|   +-- ToolNotFoundError
|   +-- ToolExecutionError
|
+-- StateError
```

---

## 4. Core Subsystem Analysis

### 4.1 Memory Subsystem Deep Dive

**Architecture Strengths:**
1. Clean separation of concerns across 3 layers
2. Full traceability from categories to source resources
3. Dual retrieval modes (RAG/LLM) with hybrid escalation
4. Event emission for audit trail
5. Async-first design

**Memory Operations:**

```python
class MemoryManager:
    async def store_resource(
        resource_type: str,
        content: str,
        metadata: dict,
        session_id: str,
    ) -> Resource

    async def extract_and_store(
        resource_id: str,
        session_id: str,
    ) -> list[MemoryItem]

    async def retrieve(
        query: str,
        k: int = 5,
        mode: str = "hybrid",
    ) -> list[MemoryItem]

    async def consolidate_category(
        category: str,
        session_id: str,
    ) -> MemoryCategory

    async def get_resource(resource_id: str) -> Optional[Resource]
```

**Performance Considerations:**
- RAG retrieval: O(log n) with vector index
- LLM retrieval: O(n) linear scan - expensive at scale
- Consolidation: Batched aggregation to minimize LLM calls
- Caching: Category-level caching recommended

**Scalability Limits:**
- Current implementation uses file-based storage
- Vector database integration needed for production scale
- Recommend: Pinecone, Weaviate, or Qdrant for embeddings

### 4.2 Planning Subsystem Deep Dive

**Architecture Strengths:**
1. DAG validation prevents circular dependencies
2. Plan caching reduces LLM calls (24h TTL)
3. Clean step dependencies model
4. Event-driven execution tracking

**Plan Generation Flow:**
```
1. Check cache for similar goal
2. IF cache hit AND TTL valid:
   - Return cached plan
3. ELSE:
   - Decompose goal into steps (LLM)
   - Validate DAG (Kahn's algorithm)
   - Cache validated plan
   - Return plan
```

**Step Types:**
```python
class StepType(str, Enum):
    REASONING = "reasoning"    # LLM reasoning task
    TOOL_CALL = "tool_call"   # Tool invocation
    MEMORY_OP = "memory_op"   # Memory operation
    DECISION = "decision"     # Conditional branching
```

**Parallel Execution Opportunity:**
Steps with no dependencies can execute in parallel. Current implementation is sequential - parallel execution is a recommended optimization.

### 4.3 Reasoning Subsystem Deep Dive

**Architecture Strengths:**
1. Clean strategy abstraction via BaseStrategy
2. Automatic selection by complexity score
3. Fallback chain for robustness
4. Per-strategy metrics tracking

**Strategy Implementations:**

| Strategy | Key Feature | Token Cost |
|----------|------------|------------|
| Direct | Single LLM call | Low (~500) |
| CoT | Step-by-step chain | Medium (~1000) |
| ToT | Multiple paths explored | High (~3000) |
| ReAct | Thought-Action-Observation loop | Variable |
| MCTS | Monte Carlo tree search | Very High (~5000+) |

**ReAct Loop:**
```
while not done and iterations < max:
    thought = llm.reason(context)
    action = llm.select_action(thought)
    observation = execute_action(action)
    context.append(thought, action, observation)
    done = check_completion(observation)
```

**MCTS Implementation:**
```
1. Selection: UCB1 tree traversal
2. Expansion: Add child nodes (LLM)
3. Simulation: Rollout to terminal (LLM)
4. Backpropagation: Update values
5. Repeat until budget exhausted
6. Return best path
```

### 4.4 Contract Subsystem Deep Dive

**Architecture Strengths:**
1. Declarative contract specification
2. Pluggable validation rules
3. Multiple failure strategies
4. Rich fallback mechanisms
5. Full event emission

**Validation Process:**
```python
def validate(output: dict, contract: Contract) -> ValidationResult:
    errors = []

    # 1. Check required fields present
    for deliverable in contract.deliverables:
        if deliverable.required and deliverable.name not in output:
            errors.append(MissingFieldError(deliverable.name))

    # 2. Check field types
    for deliverable in contract.deliverables:
        if deliverable.name in output:
            actual_type = type(output[deliverable.name])
            if not matches_type(actual_type, deliverable.type):
                errors.append(TypeMismatchError(...))

    # 3. Apply validation rules
    for deliverable in contract.deliverables:
        if deliverable.name in output:
            value = output[deliverable.name]
            for rule in deliverable.validation_rules:
                if not evaluate_rule(rule, value):
                    errors.append(RuleViolationError(...))

    return ValidationResult(is_valid=len(errors)==0, errors=errors)
```

**Fallback Strategy Selection:**
```python
def select_strategy(
    partial_output: dict,
    contract: Contract,
    validation_result: ValidationResult,
) -> FallbackResult:

    # Priority order based on contract failure_strategy
    if contract.failure_strategy == FailureStrategy.PARTIAL:
        return _apply_partial_fill(partial_output, contract)

    elif contract.failure_strategy == FailureStrategy.TEMPLATE:
        return _apply_template_defaults(contract)

    elif contract.failure_strategy == FailureStrategy.FALLBACK:
        # Try partial first, then template
        result = _apply_partial_fill(partial_output, contract)
        if result.is_complete:
            return result
        return _apply_template_defaults(contract)

    elif contract.failure_strategy == FailureStrategy.ESCALATE:
        return FallbackResult(strategy=ESCALATE, output={})

    # Default: FAIL
    return FallbackResult(strategy=FAIL, output={})
```

---

## 5. Integration Layer Analysis

### 5.1 Contract-Router Integration

**ContractSelector** bridges routing decisions to contract selection:

```
RouteDecision                     Contract Selection
+----------------+                +------------------+
| intent         | -----+-------> | INTENT_MAP       |
| complexity     |      |         +------------------+
| handler_name   |      |
| use_contracts  |      |         +------------------+
| metadata       |      +-------> | STRATUM_MAP      |
+----------------+                +------------------+
                                          |
                                          v
                                  +------------------+
                                  | Complexity Adj.  |
                                  | - +1 retry @ 0.8 |
                                  | - 1.5x tokens    |
                                  | - -1 retry @ 0.5 |
                                  +------------------+
                                          |
                                          v
                                  +------------------+
                                  | Adjusted Contract|
                                  +------------------+
```

**Complexity Thresholds:**
```python
COMPLEXITY_NO_CONTRACT = 0.5      # Below: No contract needed
COMPLEXITY_OPTIONAL_CONTRACT = 0.7 # Below: Fallback strategy
COMPLEXITY_REQUIRED_CONTRACT = 0.9 # Below: Retry strategy
# Above 0.9: FAIL strategy (strict verification)
```

### 5.2 Contract-Memory Integration

**ContractMemoryBridge** enables learning from validation history:

```
Contract Execution                Memory Storage
+------------------+              +------------------+
| Contract         |              | CONTRACT_TEMPLATE|
| Validation       | -----------> | (versioned)      |
| Output           |              +------------------+
+------------------+
        |                         +------------------+
        +-----------------------> | VALIDATION_RESULT|
                                  | (pass/fail)      |
                                  +------------------+
                                          |
        +-----------------------> | EXEMPLAR_OUTPUT  |
        |                         | (successful)     |
        |                         +------------------+
        |
        v
+------------------+
| ContractContext  |
| - examples       | <----------- Retrieved for
| - common_errors  |              prompt enhancement
| - success_rate   |
+------------------+
```

**Learning Loop:**
1. Store validation results (pass/fail with errors)
2. Extract insights from failures (LLM extraction)
3. Retrieve common errors for future prompts
4. Track success rate per contract
5. Store successful outputs as exemplars

### 5.3 Contract-Plan Integration

**PlanContractBridge** attaches contracts to plan steps:

```
Plan Steps                        Contract Attachment
+------------------+              +------------------+
| Step 1           |              | No contract      |
| (simple)         | -----------> | (complexity low) |
+------------------+              +------------------+

+------------------+              +------------------+
| Step 2           |              | ContractSpec     |
| TOOL_CALL        | -----------> | (tool output)    |
| (crm_update)     |              +------------------+
+------------------+

+------------------+              +------------------+
| Step 3           |              | ContractSpec     |
| REASONING        | -----------> | (structured)     |
| (analyze + score)|              +------------------+
+------------------+

+------------------+              +------------------+
| Step N (FINAL)   |              | Main Contract    |
| Final output     | -----------> | (full validation)|
+------------------+              +------------------+
```

**Attachment Rules:**
1. Final step: Always gets main contract
2. TOOL_CALL steps: Contract if tool in STRUCTURED_OUTPUT_TOOLS
3. REASONING steps: Contract if keywords match STRUCTURED_OUTPUT_KEYWORDS

### 5.4 Contract-Reasoning Integration

**ContractAwareReasoningManager** adjusts strategy for contracts:

```
Contract Requirements             Strategy Adjustment
+------------------+              +------------------+
| failure_strategy |              | +0.2 complexity  |
| = FAIL           | -----------> | (more thorough)  |
+------------------+              +------------------+

+------------------+              +------------------+
| deliverables > 5 | -----------> | +0.1 complexity  |
+------------------+              +------------------+

+------------------+              +------------------+
| rules > 10       | -----------> | +0.1 complexity  |
+------------------+              +------------------+

Base Complexity: 0.5
        +
Adjustments: +0.4 (FAIL + many deliverables + complex rules)
        =
Effective: 0.9 -> Selects MCTS strategy
```

**Prompt Enhancement:**
```
TASK: {original_task}

OUTPUT REQUIREMENTS:
Your response must be a valid JSON object with the following structure:

  score*: int - Overall score from 0-100
  recommendation*: str - Recommended action
  analysis: str - Detailed analysis

(* = required field)

EXAMPLE OUTPUT:
{
  "score": 75,
  "recommendation": "Schedule demo",
  "analysis": "Strong budget, needs timeline clarity"
}

VALIDATION RULES:
- score: value >= 0, value <= 100

COMMON MISTAKES TO AVOID:
- Missing field: recommendation
- Type error: score must be int, not str

Return ONLY the JSON object, no explanatory text.
```

---

## 6. Data Flow Architecture

### 6.1 Request Processing Flow

```
+------------------+     +------------------+     +------------------+
|   User Request   | --> |     Router       | --> |  Intent/Complexity|
|                  |     |                  |     |  Assessment       |
+------------------+     +------------------+     +------------------+
                                                          |
                                                          v
+------------------+     +------------------+     +------------------+
|  Handler         | <-- | RouteDecision    | <-- | Subsystem Flags  |
|  Selection       |     | - use_planning   |     | - Planning       |
|                  |     | - use_memory     |     | - Memory         |
+------------------+     | - use_contracts  |     | - Contracts      |
        |                +------------------+     +------------------+
        v
+------------------+
| Contract Select  |
| (if applicable)  |
+------------------+
        |
        v
+------------------+     +------------------+     +------------------+
|    Planning      | --> |    Reasoning     | --> |    Execution     |
|    (if enabled)  |     |    (strategy)    |     |    (agents/tools)|
+------------------+     +------------------+     +------------------+
        |                        |                        |
        v                        v                        v
+------------------+     +------------------+     +------------------+
|  Plan Events     |     | Reasoning Events |     | Execution Events |
+------------------+     +------------------+     +------------------+
        |                        |                        |
        +------------------------+------------------------+
                                 |
                                 v
                        +------------------+
                        |    EventStore    |
                        |  (append-only)   |
                        +------------------+
```

### 6.2 Contract Execution Flow

```
+------------------+
| Execute Request  |
| - agent          |
| - task           |
| - contract       |
+------------------+
        |
        v
+------------------+     +------------------+
| Agent.run(task)  | --> |     Output       |
+------------------+     +------------------+
        |                        |
        |                        v
        |                +------------------+
        |                | ContractValidator|
        |                | .validate()      |
        |                +------------------+
        |                        |
        |               +--------+--------+
        |               |                 |
        |               v                 v
        |        +----------+      +----------+
        |        |  VALID   |      | INVALID  |
        |        +----------+      +----------+
        |               |                 |
        |               v                 |
        |        +----------+             |
        |        | SUCCESS  |             |
        |        | Result   |             |
        |        +----------+             |
        |                                 |
        |                    +------------+
        |                    |
        |                    v
        |            +------------------+
        |            | Check Retries    |
        |            | & Token Budget   |
        |            +------------------+
        |                    |
        |           +--------+--------+
        |           |                 |
        |           v                 v
        |    +----------+      +----------+
        |    | RETRY    |      | FALLBACK |
        |    +----------+      +----------+
        |           |                 |
        |           v                 v
        +-----> Refine Prompt    +----------+
                    |            | Apply    |
                    v            | Strategy |
               +----------+      +----------+
               | Re-execute|           |
               +----------+            v
                    |            +----------+
                    |            | FALLBACK |
                    +-+          | Result   |
                                 +----------+
```

### 6.3 Memory Data Flow

```
+------------------+     +------------------+     +------------------+
|  Raw Content     | --> |    Resources     | --> |   Extraction     |
|  (conversations, |     |    Layer 1       |     |   (LLM-based)    |
|   documents)     |     +------------------+     +------------------+
+------------------+                                      |
                                                          v
+------------------+     +------------------+     +------------------+
|   Categories     | <-- |  Consolidation   | <-- |      Items       |
|   Layer 3        |     |   (aggregation)  |     |    Layer 2       |
+------------------+     +------------------+     +------------------+
        |                                                 |
        v                                                 v
+------------------+                              +------------------+
|   Markdown       |                              |   Embeddings     |
|   (human-readable|                              |   (RAG search)   |
+------------------+                              +------------------+
        |                                                 |
        +-------------------------+------------------------+
                                  |
                                  v
                          +------------------+
                          |    Retrieval     |
                          | (RAG/LLM/Hybrid) |
                          +------------------+
                                  |
                                  v
                          +------------------+
                          |  Agent Context   |
                          +------------------+
```

### 6.4 Token Budget Flow

```
+------------------+     +------------------+
|  TokenBudget     | --> |  TokenTracker    |
| - max_input      |     | - input_tokens   |
| - max_output     |     | - output_tokens  |
| - max_total      |     | - total_tokens   |
| - warn_threshold |     +------------------+
+------------------+              |
        |                         v
        v                 +------------------+
+------------------+      | Check Budget     |
| Contract         | ---> | Before Execution |
| Constraints      |      +------------------+
+------------------+              |
                         +--------+--------+
                         |                 |
                         v                 v
                  +----------+      +----------+
                  | WITHIN   |      | EXCEEDED |
                  | BUDGET   |      +----------+
                  +----------+           |
                         |               v
                         v        +----------+
                  +----------+    | Raise    |
                  | Continue |    | TokenBudgetExceeded
                  | Execution|    +----------+
                  +----------+
                         |
                         v
                  +------------------+
                  | Update Tracker   |
                  | After Execution  |
                  +------------------+
```

---

## 7. API Architecture

### 7.1 API Specification Status

**Phase 5 OpenAPI Spec:** Complete (1563 lines)
- Planning endpoints: /plans, /plans/{plan_id}, /plans/execute
- Reasoning endpoints: /reasoning/execute, /reasoning/strategies

**Phase 6 OpenAPI Spec:** Complete (1026 lines)
- Contract endpoints: /contracts, /contracts/{name}, /contracts/validate
- Execution endpoint: /contracts/execute
- Template endpoints: /templates, /templates/{name}

**FastAPI Implementation:** NOT STARTED (placeholder only)

### 7.2 API Endpoint Summary

**Planning Endpoints:**
| Method | Path | Description |
|--------|------|-------------|
| POST | /plans | Create new plan |
| GET | /plans/{id} | Get plan details |
| POST | /plans/{id}/execute | Execute plan |
| GET | /plans/{id}/status | Get execution status |

**Reasoning Endpoints:**
| Method | Path | Description |
|--------|------|-------------|
| POST | /reasoning/execute | Execute reasoning task |
| GET | /reasoning/strategies | List available strategies |
| GET | /reasoning/strategies/{name} | Get strategy details |

**Contract Endpoints:**
| Method | Path | Description |
|--------|------|-------------|
| GET | /contracts | List contracts |
| POST | /contracts | Create contract |
| GET | /contracts/{name} | Get contract details |
| PUT | /contracts/{name} | Update contract |
| DELETE | /contracts/{name} | Delete contract |
| POST | /contracts/validate | Validate output |
| POST | /contracts/execute | Execute with contract |

**Template Endpoints:**
| Method | Path | Description |
|--------|------|-------------|
| GET | /templates | List templates |
| GET | /templates/{name} | Get template |
| GET | /templates/{name}/prompt-context | Get prompt context |

### 7.3 Request/Response Schemas

**ExecutionRequest:**
```yaml
ExecutionRequest:
  required:
    - agent_id
    - task
  properties:
    agent_id: string
    task: string
    contract_name: string
    contract_spec: Contract
    context: object
    force_strategy: enum[retry, fallback, partial, template, escalate, fail]
    session_id: string
```

**ExecutionResult:**
```yaml
ExecutionResult:
  properties:
    output: object
    is_valid: boolean
    attempts: integer
    tokens_used: integer
    applied_strategy: enum[success, retry, fallback, fail]
    validation_result: ValidationResult
    metadata:
      execution_time_ms: integer
      model_used: string
      retries_performed: integer
```

### 7.4 Error Response Format

Following RFC 7807 Problem Details:
```yaml
Error:
  required:
    - type
    - title
    - status
    - detail
  properties:
    type: uri          # Error type URI
    title: string      # Human-readable title
    status: integer    # HTTP status code
    detail: string     # Human-readable explanation
    code: string       # Application error code
    instance: uri      # Specific occurrence URI
    recovery_suggestions: array[string]
```

### 7.5 Authentication Design

**JWT Bearer Authentication:**
```yaml
securitySchemes:
  BearerAuth:
    type: http
    scheme: bearer
    bearerFormat: JWT
```

**Implementation Requirements (TODO):**
1. JWT token validation middleware
2. Token expiration handling
3. Role-based access control
4. API key alternative for service accounts
5. Rate limiting per token

---

## 8. Performance Architecture

### 8.1 Performance Budget

Based on 256K token context window:

| Operation | Target Latency | Token Budget |
|-----------|---------------|--------------|
| Intent Classification | < 50ms | 0 (local) |
| Complexity Assessment | < 50ms | 0 (local) |
| Routing Decision | < 100ms | 0 (combined) |
| Memory Retrieval (RAG) | < 200ms | 0 |
| Memory Retrieval (LLM) | < 2000ms | ~1000 |
| Plan Generation | < 3000ms | ~2000 |
| Contract Validation | < 100ms | 0 (local) |
| Strategy Execution | Variable | Strategy-dependent |

**Strategy Token Costs:**
| Strategy | Input Tokens | Output Tokens | Total |
|----------|-------------|---------------|-------|
| Direct | ~500 | ~200 | ~700 |
| CoT | ~800 | ~400 | ~1200 |
| ToT | ~2000 | ~1000 | ~3000 |
| ReAct | ~1500/iteration | ~500/iteration | Variable |
| MCTS | ~3000 | ~2000 | ~5000+ |

### 8.2 Caching Strategy

**Plan Cache:**
- TTL: 24 hours
- Key: Hash of (goal, context)
- Storage: In-memory dictionary
- Eviction: LRU when max_size reached

**Memory Cache (Recommended):**
- Category cache: 1 hour TTL
- Item cache: 15 minute TTL
- Embedding cache: 24 hour TTL

**Contract Template Cache:**
- TTL: Until restart
- Storage: In-memory
- Invalidation: On template update

### 8.3 Bottleneck Analysis

**Identified Bottlenecks:**

1. **LLM Calls** (Primary)
   - Plan generation: 1 call per plan
   - Reasoning strategies: 1-10+ calls per task
   - Memory extraction: 1 call per resource
   - Mitigation: Caching, batching, parallel execution

2. **Memory Retrieval** (Secondary)
   - LLM mode: Linear scan of resources
   - Mitigation: Vector database for RAG

3. **File I/O** (Minor)
   - EventStore: Append operations
   - Mitigation: Async I/O, batched writes

4. **Validation Rules** (Minor)
   - Rule evaluation: O(n) per deliverable
   - Mitigation: Compile rules once, cache

### 8.4 Optimization Recommendations

**High Priority:**
1. Implement parallel plan step execution for independent steps
2. Add vector database backend for memory retrieval
3. Implement LLM response streaming
4. Add request-level caching for repeated queries

**Medium Priority:**
1. Batch memory extraction operations
2. Implement connection pooling for external services
3. Add circuit breakers for LLM providers
4. Optimize regex compilation (already done in router)

**Low Priority:**
1. Implement lazy loading for strategy modules
2. Add metrics collection for bottleneck monitoring
3. Profile memory allocation patterns

---

## 9. Security Architecture

### 9.1 Security Layers

```
+------------------+
| Authentication   | <- JWT token validation
+------------------+
        |
        v
+------------------+
| Authorization    | <- Role-based access control (TODO)
+------------------+
        |
        v
+------------------+
| Input Validation | <- Pydantic models, sanitization
+------------------+
        |
        v
+------------------+
| Token Budget     | <- Resource exhaustion prevention
+------------------+
        |
        v
+------------------+
| Prompt Injection | <- Output filtering (TODO)
| Defense          |
+------------------+
```

### 9.2 Current Security Controls

**Implemented:**
1. Pydantic validation for all data models
2. Token budget enforcement
3. Exception hierarchy with safe error messages
4. File locking for concurrent access
5. Environment variable configuration (no hardcoded secrets)

**Not Implemented:**
1. JWT authentication middleware
2. Role-based access control
3. Rate limiting
4. Prompt injection filtering
5. Output sanitization
6. Audit logging with security events

### 9.3 Prompt Injection Attack Surface

**Risk Areas:**
1. **User task input** -> Planner, Reasoning strategies
2. **Memory content** -> Retrieval context injection
3. **Contract definitions** -> Validation rule injection
4. **Tool parameters** -> MCP tool invocation

**Mitigation Strategies (Recommended):**

1. **Input Sanitization:**
```python
def sanitize_task_input(task: str) -> str:
    # Remove potential injection patterns
    patterns_to_remove = [
        r"ignore previous instructions",
        r"disregard all prior",
        r"system prompt:",
    ]
    sanitized = task
    for pattern in patterns_to_remove:
        sanitized = re.sub(pattern, "", sanitized, flags=re.IGNORECASE)
    return sanitized
```

2. **Output Validation:**
```python
def validate_llm_output(output: str, expected_format: str) -> bool:
    # Validate output matches expected structure
    # Reject outputs with suspicious patterns
    pass
```

3. **Context Isolation:**
```python
def build_isolated_context(memory_items: list) -> str:
    # Wrap memory content in clear delimiters
    # Instruct LLM to treat as data, not instructions
    pass
```

### 9.4 API Key Handling

**Current Status:**
- Environment variables: ANTHROPIC_API_KEY, TAVILY_API_KEY, etc.
- Pydantic SecretStr type used
- No key rotation mechanism

**Recommendations:**
1. Implement key rotation support
2. Add key scoping (per-agent, per-session)
3. Implement key usage tracking
4. Add alerting for suspicious usage patterns

### 9.5 Security Checklist

| Control | Status | Priority |
|---------|--------|----------|
| Input validation | COMPLETE | - |
| Token budget enforcement | COMPLETE | - |
| JWT authentication | TODO | HIGH |
| Rate limiting | TODO | HIGH |
| Prompt injection defense | TODO | HIGH |
| Output sanitization | TODO | MEDIUM |
| Audit logging | TODO | MEDIUM |
| Key rotation | TODO | LOW |
| RBAC | TODO | MEDIUM |

---

## 10. Scalability Analysis

### 10.1 Current Architecture Limits

**Single-Instance Limits:**
| Resource | Estimated Limit | Bottleneck |
|----------|----------------|------------|
| Concurrent requests | ~50 | Async event loop |
| Memory items | ~100,000 | File-based storage |
| Events per session | ~10,000 | JSON file I/O |
| Plans cached | ~1,000 | Memory |

**Why These Limits:**
1. File-based storage doesn't scale horizontally
2. In-memory caches lost on restart
3. No request distribution mechanism
4. Single-threaded event loop for async operations

### 10.2 Horizontal Scaling Requirements

**To scale beyond single instance:**

1. **Stateless Application Layer:**
   - Move session state to Redis/Memcached
   - Use distributed locking (Redis SETNX)
   - Externalize caches to Redis

2. **Database Backend:**
   - Replace JSON files with PostgreSQL
   - Use vector database for embeddings
   - Implement connection pooling

3. **Load Balancing:**
   - Add sticky sessions for WebSocket
   - Implement health checks
   - Configure auto-scaling policies

4. **Message Queue:**
   - Add async job processing (Celery/RQ)
   - Queue plan execution steps
   - Enable retry with backoff

### 10.3 Scaling Architecture (Future)

```
                          +------------------+
                          |   Load Balancer  |
                          +------------------+
                                   |
              +--------------------+--------------------+
              |                    |                    |
              v                    v                    v
       +----------+         +----------+         +----------+
       |  API 1   |         |  API 2   |         |  API N   |
       | (Stateless)        | (Stateless)        | (Stateless)
       +----------+         +----------+         +----------+
              |                    |                    |
              +--------------------+--------------------+
                                   |
              +--------------------+--------------------+
              |                    |                    |
              v                    v                    v
       +----------+         +----------+         +----------+
       |  Redis   |         | PostgreSQL|        |  Qdrant  |
       | (Cache)  |         | (State)   |        | (Vectors)|
       +----------+         +----------+         +----------+
```

### 10.4 Cost Scaling Model

**Per-Request Cost Components:**
```
Total Cost = Routing (free) + Memory (depends) + Planning (depends) + Reasoning + Validation (free)

Where:
- Routing: 0 tokens (local processing)
- Memory: 0-1000 tokens (RAG free, LLM charged)
- Planning: ~2000 tokens (if enabled)
- Reasoning: 700-5000+ tokens (strategy-dependent)
- Validation: 0 tokens (local processing)
```

**Estimated Cost per Request (at $0.003/1K input, $0.015/1K output):**
| Complexity | Tokens | Estimated Cost |
|------------|--------|----------------|
| Simple (Direct) | ~700 | ~$0.004 |
| Medium (CoT) | ~1200 | ~$0.007 |
| Complex (ToT) | ~3000 | ~$0.02 |
| Critical (MCTS) | ~5000 | ~$0.03 |
| Full Pipeline (Complex) | ~8000 | ~$0.05 |

---

## 11. Gap Analysis

### 11.1 Critical Gaps (Phase 7 Blockers)

| Gap | Description | Impact | Effort |
|-----|-------------|--------|--------|
| ContextManager | Context assembly and compression not implemented | Cannot manage 256K context window efficiently | High (2-3 weeks) |
| EvolutionManager | Self-improvement via TextGrad not implemented | No automated agent improvement | High (3-4 weeks) |
| FastAPI Server | REST API server not implemented | No programmatic access | Medium (1-2 weeks) |
| WebSocket | Streaming responses not implemented | Poor UX for long-running tasks | Medium (1 week) |

### 11.2 High Priority Gaps

| Gap | Description | Impact | Effort |
|-----|-------------|--------|--------|
| SigilOrchestrator | Unified orchestration layer missing | Manual coordination required | Medium (1-2 weeks) |
| Parallel Execution | Plan steps execute sequentially | Suboptimal performance | Medium (1 week) |
| Vector Database | File-based embeddings | Won't scale | Medium (1-2 weeks) |
| Rate Limiting | No request throttling | DoS vulnerability | Low (3-5 days) |

### 11.3 Medium Priority Gaps

| Gap | Description | Impact | Effort |
|-----|-------------|--------|--------|
| Authentication | JWT middleware not implemented | No security | Low (3-5 days) |
| Prompt Injection Defense | No filtering | Security risk | Medium (1 week) |
| Metrics Collection | No observability | Debugging difficulty | Low (3-5 days) |
| Connection Pooling | Not implemented | Resource waste | Low (2-3 days) |

### 11.4 Gap Resolution Roadmap

**Week 1-2:**
- [ ] Implement FastAPI server with basic endpoints
- [ ] Add JWT authentication middleware
- [ ] Implement rate limiting

**Week 3-4:**
- [ ] Implement ContextManager with compression
- [ ] Add parallel plan step execution
- [ ] Integrate vector database

**Week 5-6:**
- [ ] Implement WebSocket streaming
- [ ] Add SigilOrchestrator
- [ ] Implement metrics collection

**Week 7-8:**
- [ ] Begin EvolutionManager implementation
- [ ] Add prompt injection defense
- [ ] Performance optimization

---

## 12. Risk Assessment

### 12.1 Technical Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Token budget overflow | Medium | High | Strict enforcement, early warnings |
| LLM provider outage | Low | Critical | Multi-provider support, fallback |
| Memory retrieval degradation | Medium | Medium | Caching, index optimization |
| Circular dependency in plans | Low | High | DAG validation (implemented) |
| Contract validation bypass | Low | Medium | Strict type checking (implemented) |

### 12.2 Security Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Prompt injection | High | High | Input sanitization (TODO) |
| API key exposure | Medium | Critical | SecretStr, key rotation (partial) |
| DoS via large requests | Medium | Medium | Rate limiting (TODO) |
| Data exfiltration | Low | High | Output filtering (TODO) |

### 12.3 Operational Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| State loss on restart | Medium | Medium | Persistent storage (implemented) |
| Event store corruption | Low | High | File locking (implemented) |
| Cache inconsistency | Medium | Low | TTL-based invalidation |
| Configuration drift | Low | Medium | Pydantic validation (implemented) |

### 12.4 Business Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| LLM cost overrun | Medium | Medium | Token budgets (implemented) |
| Unpredictable latency | Medium | Medium | Timeouts, circuit breakers |
| Incorrect agent outputs | Medium | High | Contracts (implemented) |
| Compliance violations | Low | Critical | Audit trail (implemented) |

---

## 13. Recommendations

### 13.1 Immediate Actions (This Sprint)

1. **Implement FastAPI Server**
   - Create basic CRUD endpoints
   - Add health check endpoint
   - Integrate with existing modules

2. **Add Authentication Middleware**
   - JWT validation
   - API key alternative
   - Basic rate limiting

3. **Create Integration Tests**
   - End-to-end contract execution
   - Memory retrieval accuracy
   - Plan execution correctness

### 13.2 Short-Term Actions (Next 2-4 Weeks)

1. **Implement ContextManager**
   - Context assembly from multiple sources
   - Token-aware compression
   - Priority-based truncation

2. **Add Parallel Execution**
   - Identify independent plan steps
   - Execute with asyncio.gather
   - Handle partial failures

3. **Integrate Vector Database**
   - Choose provider (Qdrant recommended)
   - Migrate embedding storage
   - Update retrieval logic

### 13.3 Medium-Term Actions (1-2 Months)

1. **Implement EvolutionManager**
   - Performance evaluation framework
   - TextGrad optimization
   - Prompt versioning

2. **Add WebSocket Streaming**
   - Real-time progress updates
   - Partial result streaming
   - Connection management

3. **Security Hardening**
   - Prompt injection defense
   - Output sanitization
   - Comprehensive audit logging

### 13.4 Long-Term Actions (3+ Months)

1. **Horizontal Scaling**
   - Externalize state to Redis
   - Database backend migration
   - Kubernetes deployment

2. **Advanced Features**
   - Multi-agent orchestration
   - Agent composition
   - Custom strategy plugins

3. **Observability**
   - Distributed tracing
   - Custom metrics dashboards
   - Automated alerting

---

## 14. Appendices

### 14.1 Appendix A: File Inventory

**Core Implementation Files (53 total):**

```
sigil/core/base.py                    # 180 lines
sigil/core/exceptions.py              # 220 lines
sigil/config/settings.py              # 350 lines
sigil/config/schemas/agent.py         # ~100 lines
sigil/config/schemas/memory.py        # ~150 lines
sigil/config/schemas/plan.py          # ~120 lines
sigil/config/schemas/contract.py      # ~100 lines
sigil/config/schemas/events.py        # ~80 lines
sigil/memory/manager.py               # 441 lines
sigil/memory/layers/resources.py      # ~200 lines
sigil/memory/layers/items.py          # ~250 lines
sigil/memory/layers/categories.py     # ~200 lines
sigil/memory/extraction.py            # ~150 lines
sigil/memory/retrieval.py             # ~300 lines
sigil/memory/consolidation.py         # ~180 lines
sigil/memory/templates/acti.py        # ~100 lines
sigil/routing/router.py               # 688 lines
sigil/planning/planner.py             # ~400 lines
sigil/planning/executor.py            # ~350 lines
sigil/planning/schemas.py             # ~200 lines
sigil/reasoning/manager.py            # ~450 lines
sigil/reasoning/prompts.py            # ~150 lines
sigil/reasoning/strategies/base.py    # ~100 lines
sigil/reasoning/strategies/direct.py  # ~150 lines
sigil/reasoning/strategies/chain_of_thought.py # ~200 lines
sigil/reasoning/strategies/tree_of_thoughts.py # ~300 lines
sigil/reasoning/strategies/react.py   # ~350 lines
sigil/reasoning/strategies/mcts.py    # ~400 lines
sigil/contracts/schema.py             # ~350 lines
sigil/contracts/validator.py          # ~300 lines
sigil/contracts/executor.py           # 628 lines
sigil/contracts/retry.py              # ~250 lines
sigil/contracts/fallback.py           # ~300 lines
sigil/contracts/templates/acti.py     # ~250 lines
sigil/contracts/integration/router_bridge.py    # 381 lines
sigil/contracts/integration/memory_bridge.py    # 610 lines
sigil/contracts/integration/plan_bridge.py      # 762 lines
sigil/contracts/integration/reasoning_bridge.py # 623 lines
sigil/state/events.py                 # ~400 lines
sigil/state/store.py                  # ~300 lines
sigil/state/session.py                # ~200 lines
sigil/telemetry/tokens.py             # ~280 lines

Total Implementation: ~12,000+ lines
```

**Test Files (19 total):**
```
sigil/tests/memory/test_*.py          # 6 files
sigil/tests/planning/test_*.py        # 3 files
sigil/tests/reasoning/test_*.py       # 7 files
sigil/tests/contracts/test_*.py       # 6 files

Total Tests: ~3,000+ lines
```

### 14.2 Appendix B: Configuration Reference

**Environment Variables:**
```bash
# Required
ANTHROPIC_API_KEY=sk-ant-...

# Feature Flags
SIGIL_USE_MEMORY=false
SIGIL_USE_PLANNING=false
SIGIL_USE_CONTRACTS=false
SIGIL_USE_EVOLUTION=false
SIGIL_USE_ROUTING=true

# LLM Settings
SIGIL_LLM_PROVIDER=anthropic
SIGIL_LLM_MODEL=claude-sonnet-4-20250514
SIGIL_LLM_TEMPERATURE=0.7
SIGIL_LLM_MAX_TOKENS=4096

# Memory Settings
SIGIL_MEMORY_EMBEDDING_MODEL=text-embedding-3-small
SIGIL_MEMORY_RETRIEVAL_K=5
SIGIL_MEMORY_HYBRID_THRESHOLD=0.7

# Contract Settings
SIGIL_CONTRACT_DEFAULT_MAX_RETRIES=2
SIGIL_CONTRACT_DEFAULT_STRATEGY=retry

# MCP Tools
ELEVENLABS_API_KEY=...
TAVILY_API_KEY=...
TWILIO_ACCOUNT_SID=...
TWILIO_AUTH_TOKEN=...
```

### 14.3 Appendix C: API Error Codes

| Code | HTTP Status | Description |
|------|-------------|-------------|
| CV-001 | 400 | Invalid contract specification |
| CV-002 | 400 | Invalid deliverable type |
| CV-003 | 400 | Invalid validation rule |
| CV-004 | 404 | Contract not found |
| CV-005 | 409 | Contract already exists |
| CV-006 | 403 | Cannot modify template |
| CV-007 | 422 | Contract validation failed |
| CV-008 | 422 | Contract violation (fail strategy) |
| CV-009 | 500 | Internal execution error |

### 14.4 Appendix D: Glossary

| Term | Definition |
|------|------------|
| ACTi | Actualized Collective Transformational Intelligence - Methodology for agent creation |
| BANT | Budget, Authority, Need, Timeline - Lead qualification framework |
| Contract | Formal specification of required agent outputs |
| Deliverable | Single output field in a contract |
| DAG | Directed Acyclic Graph - Used for plan step dependencies |
| RAG | Retrieval Augmented Generation - Vector-based search |
| TextGrad | Gradient descent optimization applied to text prompts |
| MCTS | Monte Carlo Tree Search - Decision-making algorithm |
| ReAct | Reasoning + Acting - Iterative thought-action-observation loop |
| ToT | Tree of Thoughts - Multi-path reasoning exploration |
| CoT | Chain of Thought - Step-by-step reasoning |
| MCP | Model Context Protocol - Tool integration standard |

### 14.5 Appendix E: References

1. Sigil v2 CLAUDE.md - Project documentation
2. Phase 5 OpenAPI Specification - docs/openapi-phase5-planning-reasoning.yaml
3. Phase 6 OpenAPI Specification - docs/openapi-phase6-contracts.yaml
4. Phase 6 Architecture Document - docs/phase6-contracts-system-architecture.md

---

**Document End**

*Generated: 2026-01-11*
*Architecture Review: Sigil v2 Phase 7*
*Status: Ready for Implementation of Identified Gaps*
