# Phase 7 Integration Architecture: System Design

## Document Information

| Field | Value |
|-------|-------|
| Version | 1.0.0 |
| Status | Draft |
| Last Updated | 2026-01-11 |
| Authors | Systems Architecture Team |

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [System Overview](#system-overview)
3. [Component Architecture](#component-architecture)
4. [Data Flow Architecture](#data-flow-architecture)
5. [Integration Patterns](#integration-patterns)
6. [Deployment Architecture](#deployment-architecture)
7. [Scalability Design](#scalability-design)
8. [Reliability & Fault Tolerance](#reliability--fault-tolerance)
9. [Observability Architecture](#observability-architecture)
10. [Security Architecture](#security-architecture)
11. [Performance Architecture](#performance-architecture)
12. [Migration Strategy](#migration-strategy)

---

## Executive Summary

Phase 7 represents the culmination of the Sigil v2 architecture, providing a unified integration layer that orchestrates all subsystems. This document provides comprehensive architectural documentation for the Integration & Polish layer.

### Key Objectives

1. **Unified Orchestration**: Single entry point for all agent operations
2. **Context Optimization**: Efficient context assembly and compression
3. **Continuous Evolution**: Automated agent evaluation and optimization
4. **Production Readiness**: Enterprise-grade reliability and observability

### Architecture Principles

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       ARCHITECTURE PRINCIPLES                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  1. SEPARATION OF CONCERNS                                                  │
│     └── Each component has a single, well-defined responsibility            │
│                                                                              │
│  2. LOOSE COUPLING                                                          │
│     └── Components communicate through well-defined interfaces              │
│                                                                              │
│  3. HIGH COHESION                                                           │
│     └── Related functionality is grouped together                           │
│                                                                              │
│  4. EVENT-DRIVEN ARCHITECTURE                                               │
│     └── State changes are propagated via events                             │
│                                                                              │
│  5. FAIL-SAFE DEFAULTS                                                      │
│     └── Systems degrade gracefully under failure                            │
│                                                                              │
│  6. OBSERVABILITY BY DESIGN                                                 │
│     └── Metrics, logs, and traces are first-class citizens                  │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## System Overview

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                              SIGIL v2 SYSTEM ARCHITECTURE                            │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                      │
│   EXTERNAL INTERFACES                                                                │
│   ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐               │
│   │   REST API  │  │  WebSocket  │  │     CLI     │  │   Webhooks  │               │
│   └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘               │
│          │                │                │                │                        │
│   ───────┴────────────────┴────────────────┴────────────────┴──────────────────     │
│                                    │                                                 │
│                                    ▼                                                 │
│   ┌─────────────────────────────────────────────────────────────────────────────┐   │
│   │                           PHASE 7: INTEGRATION LAYER                         │   │
│   │  ┌───────────────────────────────────────────────────────────────────────┐  │   │
│   │  │                        SigilOrchestrator                               │  │   │
│   │  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐   │  │   │
│   │  │  │   Request   │  │   Routing   │  │  Execution  │  │  Response   │   │  │   │
│   │  │  │  Validator  │→→│   Manager   │→→│   Engine    │→→│  Assembler  │   │  │   │
│   │  │  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘   │  │   │
│   │  └───────────────────────────────────────────────────────────────────────┘  │   │
│   │                                    │                                         │   │
│   │     ┌──────────────────────────────┼──────────────────────────────┐         │   │
│   │     │                              │                              │         │   │
│   │     ▼                              ▼                              ▼         │   │
│   │  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐          │   │
│   │  │  ContextManager  │  │   EventStore     │  │ EvolutionManager │          │   │
│   │  │  - Assembly      │  │  - Event Log     │  │  - Evaluation    │          │   │
│   │  │  - Compression   │  │  - Replay        │  │  - Optimization  │          │   │
│   │  │  - Budgeting     │  │  - Audit         │  │  - Versioning    │          │   │
│   │  └──────────────────┘  └──────────────────┘  └──────────────────┘          │   │
│   └─────────────────────────────────────────────────────────────────────────────┘   │
│                                    │                                                 │
│   ───────┬────────────────────────┬┴───────────────────────┬──────────────────     │
│          │                        │                        │                        │
│          ▼                        ▼                        ▼                        │
│   ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐                │
│   │    Phase 3      │    │    Phase 4      │    │    Phase 5      │                │
│   │    ROUTING      │    │    MEMORY       │    │   PLANNING &    │                │
│   │                 │    │                 │    │   REASONING     │                │
│   │ - Intent Class  │    │ - 3-Layer Mem   │    │ - Planner       │                │
│   │ - Complexity    │    │ - RAG/LLM       │    │ - Strategies    │                │
│   │ - Handler Sel   │    │ - Consolidation │    │ - Execution     │                │
│   └─────────────────┘    └─────────────────┘    └─────────────────┘                │
│          │                        │                        │                        │
│   ───────┴────────────────────────┴────────────────────────┴──────────────────     │
│                                    │                                                 │
│                                    ▼                                                 │
│   ┌─────────────────────────────────────────────────────────────────────────────┐   │
│   │                          Phase 6: CONTRACTS                                  │   │
│   │  - Output Validation    - Retry Management    - Fallback Strategies         │   │
│   └─────────────────────────────────────────────────────────────────────────────┘   │
│                                    │                                                 │
│   ───────┬────────────────────────┴────────────────────────┬──────────────────     │
│          │                                                  │                        │
│          ▼                                                  ▼                        │
│   ┌─────────────────┐                              ┌─────────────────┐              │
│   │  LLM Providers  │                              │    MCP Tools    │              │
│   │ - Anthropic     │                              │ - Voice         │              │
│   │ - OpenAI        │                              │ - WebSearch     │              │
│   │ - Other         │                              │ - CRM           │              │
│   └─────────────────┘                              └─────────────────┘              │
│                                                                                      │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

### Component Responsibilities

| Component | Primary Responsibility | Key Interfaces |
|-----------|----------------------|----------------|
| SigilOrchestrator | Request coordination | All subsystems |
| ContextManager | Context assembly & compression | Memory, Session |
| EvolutionManager | Agent improvement | Evaluation, Optimization |
| EventStore | Event persistence | All components |
| Router (Phase 3) | Intent classification | Orchestrator |
| MemoryManager (Phase 4) | Memory operations | Orchestrator, Context |
| Planner (Phase 5) | Goal decomposition | Orchestrator, Reasoning |
| ReasoningManager (Phase 5) | Strategy execution | Orchestrator |
| ContractExecutor (Phase 6) | Output validation | Orchestrator |

---

## Component Architecture

### SigilOrchestrator

The SigilOrchestrator is the central coordination point for all operations.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          SigilOrchestrator                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                        Request Pipeline                               │   │
│  │                                                                       │   │
│  │  Request → Validate → Authenticate → Route → Execute → Respond       │   │
│  │              │            │           │         │         │           │   │
│  │              ▼            ▼           ▼         ▼         ▼           │   │
│  │           Schema       Token      Handler   Subsystem  Metadata       │   │
│  │           Check        Verify     Select    Dispatch   Addition       │   │
│  │                                                                       │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐          │
│  │  Request State   │  │  Session State   │  │  Metrics State   │          │
│  │  ──────────────  │  │  ──────────────  │  │  ──────────────  │          │
│  │  - Active reqs   │  │  - Session map   │  │  - Request count │          │
│  │  - Request ctx   │  │  - Token usage   │  │  - Latency hist  │          │
│  │  - Semaphore     │  │  - History       │  │  - Error rates   │          │
│  └──────────────────┘  └──────────────────┘  └──────────────────┘          │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                        Handler Registry                               │   │
│  │                                                                       │   │
│  │  CREATE      → _handle_create()                                       │   │
│  │  RUN         → _handle_run()                                          │   │
│  │  PAUSE       → _handle_pause()                                        │   │
│  │  RESUME      → _handle_resume()                                       │   │
│  │  STATUS      → _handle_status()                                       │   │
│  │  MEMORY_*    → _handle_memory_*()                                     │   │
│  │  EVALUATE    → _handle_evaluate()                                     │   │
│  │  OPTIMIZE    → _handle_optimize()                                     │   │
│  │                                                                       │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### Orchestrator Internal Flow

```python
class SigilOrchestrator:
    """Internal flow diagram as code."""

    async def handle(self, request: OrchestratorRequest) -> OrchestratorResponse:
        """
        Main request handling flow.

        Flow:
        ┌─────────────────────────────────────────────────────────────┐
        │                     REQUEST FLOW                             │
        ├─────────────────────────────────────────────────────────────┤
        │                                                              │
        │  1. VALIDATION                                               │
        │     ├── Check request_id format (UUID)                       │
        │     ├── Validate operation type                              │
        │     ├── Validate payload matches operation                   │
        │     └── Validate config constraints                          │
        │                                                              │
        │  2. CONTEXT CREATION                                         │
        │     ├── Get/create session                                   │
        │     ├── Initialize token tracking                            │
        │     └── Set up request context                               │
        │                                                              │
        │  3. RESOURCE ACQUISITION                                     │
        │     ├── Acquire semaphore (concurrency control)              │
        │     ├── Register active request                              │
        │     └── Emit RequestStarted event                            │
        │                                                              │
        │  4. ROUTING & EXECUTION                                      │
        │     ├── Select handler based on operation                    │
        │     ├── Apply timeout wrapper                                │
        │     └── Execute handler                                      │
        │                                                              │
        │  5. RESPONSE ASSEMBLY                                        │
        │     ├── Calculate latency                                    │
        │     ├── Assemble metadata                                    │
        │     ├── Emit RequestCompleted event                          │
        │     └── Return response                                      │
        │                                                              │
        │  6. CLEANUP                                                  │
        │     ├── Unregister active request                            │
        │     ├── Release semaphore                                    │
        │     └── Update metrics                                       │
        │                                                              │
        └─────────────────────────────────────────────────────────────┘
        """
        pass
```

### ContextManager

The ContextManager handles all aspects of context assembly and optimization.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            ContextManager                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                      Context Assembly Pipeline                        │   │
│  │                                                                       │   │
│  │  Sources                                                              │   │
│  │  ┌─────────────┐                                                      │   │
│  │  │   System    │──┐                                                   │   │
│  │  │   Prompt    │  │                                                   │   │
│  │  └─────────────┘  │                                                   │   │
│  │  ┌─────────────┐  │    ┌─────────────┐    ┌─────────────┐            │   │
│  │  │   Memory    │──┼───▶│  Assembler  │───▶│   Budget    │            │   │
│  │  │   Items     │  │    │             │    │   Check     │            │   │
│  │  └─────────────┘  │    └─────────────┘    └──────┬──────┘            │   │
│  │  ┌─────────────┐  │                              │                    │   │
│  │  │   History   │──┤                              │ Over Budget?       │   │
│  │  │             │  │                              │                    │   │
│  │  └─────────────┘  │                    ┌─────────┴─────────┐         │   │
│  │  ┌─────────────┐  │                    │                   │         │   │
│  │  │   Tools     │──┤                    ▼                   ▼         │   │
│  │  │             │  │            ┌─────────────┐    ┌─────────────┐    │   │
│  │  └─────────────┘  │            │   Compress  │    │   Output    │    │   │
│  │  ┌─────────────┐  │            │   Context   │    │   Context   │    │   │
│  │  │   User      │──┘            └──────┬──────┘    └─────────────┘    │   │
│  │  │   Input     │                      │                              │   │
│  │  └─────────────┘                      ▼                              │   │
│  │                               ┌─────────────┐                        │   │
│  │                               │   Output    │                        │   │
│  │                               │   Context   │                        │   │
│  │                               └─────────────┘                        │   │
│  │                                                                       │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                      Compression Strategies                           │   │
│  │                                                                       │   │
│  │  Smart Compression (Default)                                          │   │
│  │  ┌──────────────────────────────────────────────────────────────┐    │   │
│  │  │ 1. Prune unused tools        (Priority: 0.3)                 │    │   │
│  │  │ 2. Summarize old history     (Priority: 0.4)                 │    │   │
│  │  │ 3. Prune low-relevance memory (Priority: 0.6)                │    │   │
│  │  │ 4. Truncate if still over    (Last resort)                   │    │   │
│  │  └──────────────────────────────────────────────────────────────┘    │   │
│  │                                                                       │   │
│  │  Truncate Strategy                                                    │   │
│  │  ┌──────────────────────────────────────────────────────────────┐    │   │
│  │  │ - Remove from end of each component                          │    │   │
│  │  │ - Preserve most recent content                               │    │   │
│  │  │ - Fast but loses information                                 │    │   │
│  │  └──────────────────────────────────────────────────────────────┘    │   │
│  │                                                                       │   │
│  │  Summarize Strategy                                                   │   │
│  │  ┌──────────────────────────────────────────────────────────────┐    │   │
│  │  │ - Use LLM to summarize sections                              │    │   │
│  │  │ - Preserves key information                                  │    │   │
│  │  │ - Slower, uses extra tokens                                  │    │   │
│  │  └──────────────────────────────────────────────────────────────┘    │   │
│  │                                                                       │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### Token Budget Allocation

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      TOKEN BUDGET: 256,000 TOKENS                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  INPUT BUDGET: 150,000 tokens                                               │
│  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━   │
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │ System Context                                                         │ │
│  │ ████████████████████                                     20,000 (13.3%)│ │
│  │ Agent prompts, configuration, role definition                          │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │ Memory Context                                                         │ │
│  │ ████████████████████████████████████████                  40,000 (26.7%)│ │
│  │ Retrieved memory items, category summaries                             │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │ Conversation History                                                   │ │
│  │ ██████████████████████████████████████████████████        50,000 (33.3%)│ │
│  │ Previous turns, assistant responses                                    │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │ Tool Definitions                                                       │ │
│  │ ███████████████                                           15,000 (10.0%)│ │
│  │ MCP tool schemas, function signatures                                  │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │ Current Request                                                        │ │
│  │ ████████████████████                                     20,000 (13.3%)│ │
│  │ User input, current turn                                               │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │ Reserved                                                               │ │
│  │ █████                                                      5,000 (3.3%)│ │
│  │ Safety margin, overhead                                                │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
│  OUTPUT BUDGET: 102,400 tokens                                              │
│  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━   │
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │ Response Content                                          80,000 (78.1%)│ │
│  │ ████████████████████████████████████████████████████████████████████  │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │ Tool Calls                                                15,000 (14.6%)│ │
│  │ ████████████████                                                       │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │ Metadata/Events                                            7,400 (7.2%)│ │
│  │ ████████                                                               │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
│  SAFETY MARGIN: 3,600 tokens (1.4%)                                         │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### EvolutionManager

The EvolutionManager handles continuous agent improvement.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           EvolutionManager                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                        Evaluation Pipeline                            │   │
│  │                                                                       │   │
│  │  Agent + TestCases                                                    │   │
│  │       │                                                               │   │
│  │       ▼                                                               │   │
│  │  ┌─────────────────┐                                                  │   │
│  │  │   Load Agent    │                                                  │   │
│  │  │   Config        │                                                  │   │
│  │  └────────┬────────┘                                                  │   │
│  │           │                                                           │   │
│  │           ▼                                                           │   │
│  │  ┌─────────────────┐    ┌─────────────────┐                          │   │
│  │  │  Execute Each   │───▶│  Score Each     │                          │   │
│  │  │  Test Case      │    │  Dimension      │                          │   │
│  │  └─────────────────┘    └────────┬────────┘                          │   │
│  │                                  │                                    │   │
│  │                                  ▼                                    │   │
│  │                         ┌─────────────────┐                          │   │
│  │                         │   Aggregate     │                          │   │
│  │                         │   Results       │                          │   │
│  │                         └────────┬────────┘                          │   │
│  │                                  │                                    │   │
│  │                                  ▼                                    │   │
│  │                         ┌─────────────────┐                          │   │
│  │                         │  Evaluation     │                          │   │
│  │                         │  Result         │                          │   │
│  │                         └─────────────────┘                          │   │
│  │                                                                       │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                       Optimization Pipeline                           │   │
│  │                                                                       │   │
│  │  Baseline Evaluation                                                  │   │
│  │       │                                                               │   │
│  │       ▼                                                               │   │
│  │  ┌─────────────────┐                                                  │   │
│  │  │  Analyze        │                                                  │   │
│  │  │  Failures       │                                                  │   │
│  │  └────────┬────────┘                                                  │   │
│  │           │                                                           │   │
│  │           ▼                                                           │   │
│  │  ┌─────────────────┐    ┌─────────────────┐                          │   │
│  │  │  Generate       │───▶│  Apply          │                          │   │
│  │  │  Improvements   │    │  Changes        │                          │   │
│  │  └─────────────────┘    └────────┬────────┘                          │   │
│  │                                  │                                    │   │
│  │                                  ▼                                    │   │
│  │                         ┌─────────────────┐                          │   │
│  │                         │  Re-evaluate    │                          │   │
│  │                         │  Agent          │                          │   │
│  │                         └────────┬────────┘                          │   │
│  │                                  │                                    │   │
│  │                         ┌────────┴────────┐                          │   │
│  │                         │  Improved?      │                          │   │
│  │                         └────────┬────────┘                          │   │
│  │                        Yes │          │ No                           │   │
│  │                            ▼          ▼                              │   │
│  │                    ┌─────────────┐  ┌─────────────┐                  │   │
│  │                    │   Create    │  │   Revert    │                  │   │
│  │                    │   Version   │  │   Changes   │                  │   │
│  │                    └─────────────┘  └─────────────┘                  │   │
│  │                                                                       │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                        Safety Constraints                             │   │
│  │                                                                       │   │
│  │  ┌─────────────────────────────────────────────────────────────────┐ │   │
│  │  │ 1. Max Prompt Change: 30%                                       │ │   │
│  │  │    └── Prevents drastic prompt modifications                    │ │   │
│  │  │                                                                 │ │   │
│  │  │ 2. Major Change Approval: Required for >20% change              │ │   │
│  │  │    └── Human-in-the-loop for significant modifications          │ │   │
│  │  │                                                                 │ │   │
│  │  │ 3. Minimum Improvement: 5%                                      │ │   │
│  │  │    └── Only apply changes that meaningfully improve             │ │   │
│  │  │                                                                 │ │   │
│  │  │ 4. Version Retention: Last 10 versions                          │ │   │
│  │  │    └── Always able to rollback                                  │ │   │
│  │  └─────────────────────────────────────────────────────────────────┘ │   │
│  │                                                                       │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Data Flow Architecture

### Request Lifecycle

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         REQUEST LIFECYCLE                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Client                                                                      │
│    │                                                                         │
│    │ POST /v7/agents/sales_qualifier/run                                    │
│    │ {"input": "Qualify lead John from Acme", "session_id": "sess_123"}     │
│    │                                                                         │
│    ▼                                                                         │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │ 1. REQUEST INGRESS                                                       ││
│  │    ├── Parse JSON body                                                   ││
│  │    ├── Extract headers (Authorization, X-Request-ID)                     ││
│  │    ├── Validate JWT token                                                ││
│  │    └── Create OrchestratorRequest                                        ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│    │                                                                         │
│    ▼                                                                         │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │ 2. ORCHESTRATOR VALIDATION                                               ││
│  │    ├── Validate request_id (UUID format)                                 ││
│  │    ├── Validate operation type                                           ││
│  │    ├── Validate payload against operation                                ││
│  │    └── Validate config constraints                                       ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│    │                                                                         │
│    ▼                                                                         │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │ 3. SESSION MANAGEMENT                                                    ││
│  │    ├── Get or create session from session_id                             ││
│  │    ├── Load session history                                              ││
│  │    ├── Initialize token tracking                                         ││
│  │    └── Create RequestContext                                             ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│    │                                                                         │
│    ▼                                                                         │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │ 4. CONTEXT ASSEMBLY                                                      ││
│  │    ├── Load agent configuration                                          ││
│  │    ├── Retrieve memory items (MemoryManager)                             ││
│  │    ├── Build system context                                              ││
│  │    ├── Build history context                                             ││
│  │    ├── Build tool context                                                ││
│  │    ├── Check token budget                                                ││
│  │    └── Compress if needed                                                ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│    │                                                                         │
│    ▼                                                                         │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │ 5. ROUTING                                                               ││
│  │    ├── Classify intent (Router)                                          ││
│  │    ├── Assess complexity                                                 ││
│  │    ├── Determine use_planning flag                                       ││
│  │    └── Select execution strategy                                         ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│    │                                                                         │
│    ├──────────────────────────────┐                                         │
│    │ use_planning=true            │ use_planning=false                      │
│    ▼                              ▼                                         │
│  ┌──────────────────────┐  ┌──────────────────────┐                        │
│  │ 6a. PLANNING         │  │ 6b. REASONING        │                        │
│  │  ├── Generate plan   │  │  ├── Select strategy │                        │
│  │  ├── Execute steps   │  │  ├── Execute task    │                        │
│  │  └── Aggregate       │  │  └── Return result   │                        │
│  └──────────────────────┘  └──────────────────────┘                        │
│    │                              │                                         │
│    └──────────────────────────────┘                                         │
│    │                                                                         │
│    ▼                                                                         │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │ 7. CONTRACT VALIDATION                                                   ││
│  │    ├── Get agent contract                                                ││
│  │    ├── Validate output against contract                                  ││
│  │    ├── If failed: retry with progressive refinement                      ││
│  │    └── If still failed: apply fallback                                   ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│    │                                                                         │
│    ▼                                                                         │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │ 8. MEMORY STORAGE                                                        ││
│  │    ├── Store conversation turn as resource                               ││
│  │    ├── Extract memory items                                              ││
│  │    └── Update relevant categories                                        ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│    │                                                                         │
│    ▼                                                                         │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │ 9. RESPONSE ASSEMBLY                                                     ││
│  │    ├── Calculate final token usage                                       ││
│  │    ├── Calculate latency                                                 ││
│  │    ├── Assemble metadata                                                 ││
│  │    ├── Emit RequestCompleted event                                       ││
│  │    └── Return OrchestratorResponse                                       ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│    │                                                                         │
│    ▼                                                                         │
│  Client                                                                      │
│    │                                                                         │
│    │ Response: 200 OK                                                       │
│    │ {"response": "...", "tool_calls": [...], "metadata": {...}}            │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### WebSocket Streaming Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       WEBSOCKET STREAMING FLOW                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Client                              Server                                  │
│    │                                    │                                    │
│    │  WS CONNECT                        │                                    │
│    │  /ws/agents/sales/run              │                                    │
│    │ ──────────────────────────────────▶│                                    │
│    │                                    │                                    │
│    │  101 Switching Protocols           │                                    │
│    │ ◀──────────────────────────────────│                                    │
│    │                                    │                                    │
│    │  {"type": "connection_ready"}      │                                    │
│    │ ◀──────────────────────────────────│                                    │
│    │                                    │                                    │
│    │  {"type": "run_request",           │                                    │
│    │   "input": "...",                  │                                    │
│    │   "session_id": "..."}             │                                    │
│    │ ──────────────────────────────────▶│                                    │
│    │                                    │  ┌────────────────────────────┐    │
│    │                                    │  │ Create RequestContext     │    │
│    │                                    │  │ Start streaming generator │    │
│    │                                    │  └────────────────────────────┘    │
│    │                                    │                                    │
│    │  {"type": "stream_start"}          │                                    │
│    │ ◀──────────────────────────────────│                                    │
│    │                                    │                                    │
│    │  {"type": "token",                 │  ┌────────────────────────────┐    │
│    │   "data": {"content": "Based"}}    │  │ LLM generates tokens      │    │
│    │ ◀──────────────────────────────────│  │ Stream to client          │    │
│    │                                    │  └────────────────────────────┘    │
│    │  {"type": "token",                 │                                    │
│    │   "data": {"content": " on"}}      │                                    │
│    │ ◀──────────────────────────────────│                                    │
│    │                                    │                                    │
│    │  ...more tokens...                 │                                    │
│    │ ◀──────────────────────────────────│                                    │
│    │                                    │                                    │
│    │  {"type": "tool_call",             │  ┌────────────────────────────┐    │
│    │   "data": {"tool_name": "crm"}}    │  │ Tool execution            │    │
│    │ ◀──────────────────────────────────│  └────────────────────────────┘    │
│    │                                    │                                    │
│    │  {"type": "tool_result",           │                                    │
│    │   "data": {"result": {...}}}       │                                    │
│    │ ◀──────────────────────────────────│                                    │
│    │                                    │                                    │
│    │  ...more tokens...                 │                                    │
│    │ ◀──────────────────────────────────│                                    │
│    │                                    │                                    │
│    │  {"type": "backpressure",          │  ┌────────────────────────────┐    │
│    │   "data": {"paused": true}}        │  │ Queue full                │    │
│    │ ◀──────────────────────────────────│  │ Signal backpressure       │    │
│    │                                    │  └────────────────────────────┘    │
│    │                                    │                                    │
│    │  {"type": "ready",                 │                                    │
│    │   "last_sequence": 42}             │                                    │
│    │ ──────────────────────────────────▶│                                    │
│    │                                    │  ┌────────────────────────────┐    │
│    │                                    │  │ Resume streaming          │    │
│    │                                    │  └────────────────────────────┘    │
│    │                                    │                                    │
│    │  ...more tokens...                 │                                    │
│    │ ◀──────────────────────────────────│                                    │
│    │                                    │                                    │
│    │  {"type": "complete",              │                                    │
│    │   "data": {"tokens_used": {...}}}  │                                    │
│    │ ◀──────────────────────────────────│                                    │
│    │                                    │                                    │
│    │  WS CLOSE                          │                                    │
│    │ ──────────────────────────────────▶│                                    │
│    │                                    │                                    │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Event Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            EVENT FLOW                                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Components emit events to EventStore                                        │
│  ──────────────────────────────────────                                     │
│                                                                              │
│  ┌──────────────────┐                                                       │
│  │  Orchestrator    │──┬──▶ OrchestratorInitializedEvent                    │
│  │                  │  ├──▶ RequestStartedEvent                             │
│  │                  │  ├──▶ RequestCompletedEvent                           │
│  │                  │  └──▶ RequestFailedEvent                              │
│  └──────────────────┘                                                       │
│                                                                              │
│  ┌──────────────────┐                                                       │
│  │  ContextManager  │──┬──▶ ContextAssembledEvent                           │
│  │                  │  ├──▶ ContextCompressedEvent                          │
│  │                  │  └──▶ BudgetWarningEvent                              │
│  └──────────────────┘                                                       │
│                                                                              │
│  ┌──────────────────┐                                                       │
│  │  EvolutionMgr    │──┬──▶ AgentEvaluatedEvent                             │
│  │                  │  ├──▶ AgentOptimizedEvent                             │
│  │                  │  └──▶ AgentRolledBackEvent                            │
│  └──────────────────┘                                                       │
│                                                                              │
│  ┌──────────────────┐                                                       │
│  │  Memory (Ph4)    │──┬──▶ MemoryStoredEvent                               │
│  │                  │  ├──▶ MemoryRetrievedEvent                            │
│  │                  │  └──▶ MemoryConsolidatedEvent                         │
│  └──────────────────┘                                                       │
│                                                                              │
│  ┌──────────────────┐                                                       │
│  │  Planning (Ph5)  │──┬──▶ PlanCreatedEvent                                │
│  │                  │  ├──▶ PlanStepCompletedEvent                          │
│  │                  │  └──▶ PlanCompletedEvent                              │
│  └──────────────────┘                                                       │
│                                                                              │
│                      │                                                       │
│                      ▼                                                       │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │                         EventStore                                       ││
│  │  ┌─────────────────────────────────────────────────────────────────┐    ││
│  │  │ Event Log (Append-Only)                                         │    ││
│  │  │                                                                 │    ││
│  │  │ seq_1: OrchestratorInitializedEvent {...}                       │    ││
│  │  │ seq_2: RequestStartedEvent {request_id: "req_001", ...}         │    ││
│  │  │ seq_3: ContextAssembledEvent {request_id: "req_001", ...}       │    ││
│  │  │ seq_4: PlanCreatedEvent {request_id: "req_001", ...}            │    ││
│  │  │ seq_5: PlanStepCompletedEvent {request_id: "req_001", ...}      │    ││
│  │  │ seq_6: RequestCompletedEvent {request_id: "req_001", ...}       │    ││
│  │  │ seq_7: RequestStartedEvent {request_id: "req_002", ...}         │    ││
│  │  │ ...                                                             │    ││
│  │  └─────────────────────────────────────────────────────────────────┘    ││
│  │                                                                          ││
│  │  Capabilities:                                                           ││
│  │  - Append events                                                         ││
│  │  - Query by request_id                                                   ││
│  │  - Query by session_id                                                   ││
│  │  - Query by time range                                                   ││
│  │  - Replay for state reconstruction                                       ││
│  │                                                                          ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│                      │                                                       │
│                      ▼                                                       │
│  Event consumers (async)                                                     │
│  ──────────────────────                                                     │
│                                                                              │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐          │
│  │  Metrics         │  │  Audit Logger    │  │  Webhook         │          │
│  │  Aggregator      │  │                  │  │  Dispatcher      │          │
│  └──────────────────┘  └──────────────────┘  └──────────────────┘          │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Integration Patterns

### Subsystem Integration

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     SUBSYSTEM INTEGRATION PATTERNS                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Pattern 1: Direct Invocation                                               │
│  ─────────────────────────────                                              │
│                                                                              │
│  Used when: Synchronous call required, result needed immediately             │
│                                                                              │
│  Orchestrator ──[await]──▶ Subsystem ──[return]──▶ Orchestrator             │
│                                                                              │
│  Example:                                                                    │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  # Context assembly                                                   │   │
│  │  context = await self._context_manager.assemble(...)                  │   │
│  │                                                                       │   │
│  │  # Routing decision                                                   │   │
│  │  decision = await self._router.route(input, agent, context)           │   │
│  │                                                                       │   │
│  │  # Memory retrieval                                                   │   │
│  │  items = await self._memory_manager.retrieve(query, k=10)             │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  Pattern 2: Event-Driven                                                    │
│  ──────────────────────                                                     │
│                                                                              │
│  Used when: Async notification needed, no immediate result required          │
│                                                                              │
│  Component ──[emit]──▶ EventStore ──[async]──▶ Subscribers                  │
│                                                                              │
│  Example:                                                                    │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  # Emit event after successful operation                              │   │
│  │  await self._emit_event(RequestCompletedEvent(                        │   │
│  │      request_id=request.request_id,                                   │   │
│  │      operation=request.operation,                                     │   │
│  │      tokens_used=context.tokens_used.total_tokens,                    │   │
│  │  ))                                                                   │   │
│  │                                                                       │   │
│  │  # Subscribers receive async                                          │   │
│  │  # - Metrics aggregator updates counters                              │   │
│  │  # - Audit logger records event                                       │   │
│  │  # - Webhook dispatcher sends to external systems                     │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  Pattern 3: Pipeline                                                        │
│  ─────────────────                                                          │
│                                                                              │
│  Used when: Sequential processing through multiple stages                    │
│                                                                              │
│  Input ──▶ Stage1 ──▶ Stage2 ──▶ Stage3 ──▶ Output                         │
│                                                                              │
│  Example:                                                                    │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  # Request pipeline                                                   │   │
│  │  request ──▶ validate ──▶ route ──▶ execute ──▶ contract ──▶ respond │   │
│  │                                                                       │   │
│  │  # Context pipeline                                                   │   │
│  │  sources ──▶ assemble ──▶ budget_check ──▶ compress ──▶ output       │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  Pattern 4: Fallback Chain                                                  │
│  ────────────────────────                                                   │
│                                                                              │
│  Used when: Graceful degradation needed                                      │
│                                                                              │
│  Primary ─[fail]─▶ Secondary ─[fail]─▶ Tertiary ─[fail]─▶ Default           │
│                                                                              │
│  Example:                                                                    │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  # Reasoning strategy fallback                                        │   │
│  │  MCTS ─[fail]─▶ ReAct ─[fail]─▶ ToT ─[fail]─▶ CoT ─[fail]─▶ Direct   │   │
│  │                                                                       │   │
│  │  # Memory retrieval fallback                                          │   │
│  │  Hybrid ─[fail]─▶ RAG ─[fail]─▶ LLM ─[fail]─▶ Empty                  │   │
│  │                                                                       │   │
│  │  # Contract fallback                                                  │   │
│  │  Full ─[fail]─▶ Partial ─[fail]─▶ Template ─[fail]─▶ Fail            │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Cross-Phase Communication

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    CROSS-PHASE COMMUNICATION                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Phase 7 (Orchestrator) ←→ Phase 3 (Router)                                 │
│  ───────────────────────────────────────────                                │
│                                                                              │
│  Orchestrator                                Router                          │
│       │                                          │                          │
│       │  RouteRequest(input, agent, context)     │                          │
│       │ ────────────────────────────────────────▶│                          │
│       │                                          │                          │
│       │                                          │ Classify intent          │
│       │                                          │ Assess complexity        │
│       │                                          │ Select handler           │
│       │                                          │                          │
│       │  RouteDecision(complexity, strategy,     │                          │
│       │                 use_planning, handler)   │                          │
│       │ ◀────────────────────────────────────────│                          │
│       │                                          │                          │
│                                                                              │
│  Phase 7 (Orchestrator) ←→ Phase 4 (Memory)                                 │
│  ────────────────────────────────────────────                               │
│                                                                              │
│  Orchestrator                                Memory                          │
│       │                                          │                          │
│       │  Retrieve(query, agent, k=10, method)    │                          │
│       │ ────────────────────────────────────────▶│                          │
│       │                                          │                          │
│       │                                          │ Search embeddings        │
│       │                                          │ Rank by relevance        │
│       │                                          │ Return items             │
│       │                                          │                          │
│       │  List[MemoryItem]                        │                          │
│       │ ◀────────────────────────────────────────│                          │
│       │                                          │                          │
│       │                                          │                          │
│       │  Store(content, agent, session, type)    │                          │
│       │ ────────────────────────────────────────▶│                          │
│       │                                          │                          │
│       │                                          │ Store resource           │
│       │                                          │ Extract items            │
│       │                                          │ Update categories        │
│       │                                          │                          │
│       │  StoreResult(resource_id, items_count)   │                          │
│       │ ◀────────────────────────────────────────│                          │
│       │                                          │                          │
│                                                                              │
│  Phase 7 (Orchestrator) ←→ Phase 5 (Planning/Reasoning)                     │
│  ───────────────────────────────────────────────────────                    │
│                                                                              │
│  Orchestrator                              Planner/Reasoning                 │
│       │                                          │                          │
│       │  (If use_planning=true)                  │                          │
│       │  GeneratePlan(goal, context)             │                          │
│       │ ────────────────────────────────────────▶│                          │
│       │                                          │                          │
│       │  Plan(steps, dependencies)               │                          │
│       │ ◀────────────────────────────────────────│                          │
│       │                                          │                          │
│       │  ExecutePlan(plan, session, budget)      │                          │
│       │ ────────────────────────────────────────▶│                          │
│       │                                          │                          │
│       │  ExecutionResult(outputs, tokens)        │                          │
│       │ ◀────────────────────────────────────────│                          │
│       │                                          │                          │
│       │                                          │                          │
│       │  (If use_planning=false)                 │                          │
│       │  Reason(task, complexity, context)       │                          │
│       │ ────────────────────────────────────────▶│                          │
│       │                                          │                          │
│       │  StrategyResult(result, strategy,        │                          │
│       │                 confidence, tokens)      │                          │
│       │ ◀────────────────────────────────────────│                          │
│       │                                          │                          │
│                                                                              │
│  Phase 7 (Orchestrator) ←→ Phase 6 (Contracts)                              │
│  ─────────────────────────────────────────────                              │
│                                                                              │
│  Orchestrator                              Contracts                         │
│       │                                          │                          │
│       │  Validate(output, contract)              │                          │
│       │ ────────────────────────────────────────▶│                          │
│       │                                          │                          │
│       │                                          │ Check deliverables       │
│       │                                          │ Validate constraints     │
│       │                                          │                          │
│       │  ValidationResult(passed, failures)      │                          │
│       │ ◀────────────────────────────────────────│                          │
│       │                                          │                          │
│       │  (If !passed && retry)                   │                          │
│       │  ExecuteWithContract(executor,           │                          │
│       │                      contract, context)  │                          │
│       │ ────────────────────────────────────────▶│                          │
│       │                                          │                          │
│       │  ContractResult(output, retries,         │                          │
│       │                 fallback_used)           │                          │
│       │ ◀────────────────────────────────────────│                          │
│       │                                          │                          │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Deployment Architecture

### Single-Node Deployment

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     SINGLE-NODE DEPLOYMENT                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Server (e.g., 8 vCPU, 32GB RAM)                                            │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │                                                                          ││
│  │  ┌─────────────────────────────────────────────────────────────────┐    ││
│  │  │                      Docker Compose                              │    ││
│  │  │                                                                  │    ││
│  │  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐              │    ││
│  │  │  │   sigil     │  │   redis     │  │  postgres   │              │    ││
│  │  │  │   (api)     │  │   (cache)   │  │  (events)   │              │    ││
│  │  │  │             │  │             │  │             │              │    ││
│  │  │  │  port:8000  │  │  port:6379  │  │  port:5432  │              │    ││
│  │  │  └─────────────┘  └─────────────┘  └─────────────┘              │    ││
│  │  │                                                                  │    ││
│  │  │  ┌─────────────┐  ┌─────────────┐                               │    ││
│  │  │  │  mcp-voice  │  │ mcp-search  │                               │    ││
│  │  │  │             │  │             │                               │    ││
│  │  │  │  port:9001  │  │  port:9002  │                               │    ││
│  │  │  └─────────────┘  └─────────────┘                               │    ││
│  │  │                                                                  │    ││
│  │  └─────────────────────────────────────────────────────────────────┘    ││
│  │                                                                          ││
│  │  Volumes:                                                                ││
│  │  - ./data/postgres:/var/lib/postgresql/data                             ││
│  │  - ./data/redis:/data                                                   ││
│  │  - ./agents:/app/agents                                                 ││
│  │  - ./logs:/app/logs                                                     ││
│  │                                                                          ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Multi-Node Deployment

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     MULTI-NODE KUBERNETES DEPLOYMENT                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│                           ┌─────────────────┐                               │
│                           │   Load Balancer │                               │
│                           │  (Ingress/ALB)  │                               │
│                           └────────┬────────┘                               │
│                                    │                                        │
│                    ┌───────────────┼───────────────┐                        │
│                    │               │               │                        │
│                    ▼               ▼               ▼                        │
│  ┌─────────────────────┐ ┌─────────────────────┐ ┌─────────────────────┐   │
│  │   API Pod 1         │ │   API Pod 2         │ │   API Pod 3         │   │
│  │   ┌─────────────┐   │ │   ┌─────────────┐   │ │   ┌─────────────┐   │   │
│  │   │ sigil-api   │   │ │   │ sigil-api   │   │ │   │ sigil-api   │   │   │
│  │   │ replicas: 3 │   │ │   │             │   │ │   │             │   │   │
│  │   └─────────────┘   │ │   └─────────────┘   │ │   └─────────────┘   │   │
│  │   resources:        │ │                     │ │                     │   │
│  │   cpu: 2            │ │                     │ │                     │   │
│  │   memory: 4Gi       │ │                     │ │                     │   │
│  └─────────────────────┘ └─────────────────────┘ └─────────────────────┘   │
│                    │               │               │                        │
│                    └───────────────┼───────────────┘                        │
│                                    │                                        │
│                                    ▼                                        │
│              ┌─────────────────────────────────────────┐                   │
│              │              Internal Services          │                   │
│              │                                         │                   │
│              │  ┌─────────────┐  ┌─────────────┐       │                   │
│              │  │   Redis     │  │  PostgreSQL │       │                   │
│              │  │  (Cluster)  │  │  (Primary/  │       │                   │
│              │  │             │  │   Replica)  │       │                   │
│              │  └─────────────┘  └─────────────┘       │                   │
│              │                                         │                   │
│              │  ┌─────────────┐  ┌─────────────┐       │                   │
│              │  │ Vector DB   │  │   MCP       │       │                   │
│              │  │ (Pgvector/  │  │   Tools     │       │                   │
│              │  │  Pinecone)  │  │             │       │                   │
│              │  └─────────────┘  └─────────────┘       │                   │
│              │                                         │                   │
│              └─────────────────────────────────────────┘                   │
│                                                                              │
│  Horizontal Pod Autoscaler:                                                 │
│  - Min replicas: 2                                                          │
│  - Max replicas: 10                                                         │
│  - Target CPU utilization: 70%                                              │
│  - Target memory utilization: 80%                                           │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Scalability Design

### Horizontal Scaling

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       HORIZONTAL SCALING                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Stateless Components (Scale Horizontally)                                  │
│  ─────────────────────────────────────────                                  │
│                                                                              │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │ API Servers                                                            │  │
│  │                                                                        │  │
│  │  ┌─────┐  ┌─────┐  ┌─────┐  ┌─────┐  ┌─────┐                         │  │
│  │  │ API │  │ API │  │ API │  │ API │  │ API │  ...                     │  │
│  │  │  1  │  │  2  │  │  3  │  │  4  │  │  5  │                         │  │
│  │  └─────┘  └─────┘  └─────┘  └─────┘  └─────┘                         │  │
│  │                                                                        │  │
│  │  Scaling triggers:                                                     │  │
│  │  - CPU > 70%                                                           │  │
│  │  - Memory > 80%                                                        │  │
│  │  - Request queue > 100                                                 │  │
│  │                                                                        │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
│  Stateful Components (Scale Vertically / Cluster)                           │
│  ────────────────────────────────────────────────                           │
│                                                                              │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │ Redis Cluster                                                          │  │
│  │                                                                        │  │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐                                │  │
│  │  │ Master  │  │ Master  │  │ Master  │                                │  │
│  │  │ Slot    │  │ Slot    │  │ Slot    │                                │  │
│  │  │ 0-5460  │  │ 5461-   │  │ 10923-  │                                │  │
│  │  │         │  │ 10922   │  │ 16383   │                                │  │
│  │  └────┬────┘  └────┬────┘  └────┬────┘                                │  │
│  │       │            │            │                                      │  │
│  │  ┌────▼────┐  ┌────▼────┐  ┌────▼────┐                                │  │
│  │  │ Replica │  │ Replica │  │ Replica │                                │  │
│  │  └─────────┘  └─────────┘  └─────────┘                                │  │
│  │                                                                        │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │ PostgreSQL (Read Replicas)                                             │  │
│  │                                                                        │  │
│  │  ┌─────────────┐                                                       │  │
│  │  │   Primary   │ ──────────┬──────────┬──────────┐                    │  │
│  │  │   (Write)   │           │          │          │                    │  │
│  │  └─────────────┘           ▼          ▼          ▼                    │  │
│  │                     ┌───────────┐ ┌───────────┐ ┌───────────┐         │  │
│  │                     │  Replica  │ │  Replica  │ │  Replica  │         │  │
│  │                     │  (Read)   │ │  (Read)   │ │  (Read)   │         │  │
│  │                     └───────────┘ └───────────┘ └───────────┘         │  │
│  │                                                                        │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Request Distribution

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      REQUEST DISTRIBUTION                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Load Balancing Strategy: Least Connections with Health Checks              │
│  ─────────────────────────────────────────────────────────────              │
│                                                                              │
│  Incoming Requests                                                          │
│       │                                                                      │
│       ▼                                                                      │
│  ┌─────────────────┐                                                        │
│  │  Load Balancer  │                                                        │
│  │                 │                                                        │
│  │  Health Check:  │                                                        │
│  │  /health        │                                                        │
│  │  interval: 10s  │                                                        │
│  │  timeout: 5s    │                                                        │
│  │  threshold: 3   │                                                        │
│  └────────┬────────┘                                                        │
│           │                                                                  │
│           │ Least connections                                               │
│           │                                                                  │
│  ┌────────┴────────┬────────────────┬────────────────┐                     │
│  │                 │                │                │                      │
│  ▼                 ▼                ▼                ▼                      │
│  ┌───────┐    ┌───────┐    ┌───────┐    ┌───────┐                          │
│  │ API 1 │    │ API 2 │    │ API 3 │    │ API 4 │                          │
│  │  (5)  │    │  (3)  │    │  (8)  │    │  (2)  │◀── Next request         │
│  └───────┘    └───────┘    └───────┘    └───────┘    (lowest conn)         │
│                                                                              │
│  Session Affinity: Disabled (stateless design)                              │
│  Sticky Sessions: Only for WebSocket connections                            │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Reliability & Fault Tolerance

### Failure Modes and Recovery

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    FAILURE MODES AND RECOVERY                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Failure Mode: LLM Provider Unavailable                                     │
│  ───────────────────────────────────────                                    │
│                                                                              │
│  Detection: Connection timeout, 5xx errors                                   │
│  Recovery:                                                                   │
│  1. Retry with exponential backoff (1s, 2s, 4s, 8s)                         │
│  2. Fallback to alternate provider if configured                            │
│  3. Return cached response if available and fresh                           │
│  4. Return partial result with degradation notice                           │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ Primary Provider ─[fail]─▶ Retry (3x) ─[fail]─▶ Fallback Provider   │    │
│  │                                                         │            │    │
│  │                                               ─[fail]─▶ Cache       │    │
│  │                                                         │            │    │
│  │                                               ─[fail]─▶ Degraded    │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  Failure Mode: Memory System Unavailable                                    │
│  ──────────────────────────────────────                                     │
│                                                                              │
│  Detection: Connection refused, query timeout                                │
│  Recovery:                                                                   │
│  1. Continue without memory context                                          │
│  2. Set include_memory=false in context                                     │
│  3. Log degradation for monitoring                                          │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ Memory Query ─[fail]─▶ Continue without memory                       │    │
│  │                        (context.memory_context = None)               │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  Failure Mode: Contract Validation Failure                                  │
│  ─────────────────────────────────────────                                  │
│                                                                              │
│  Detection: ValidationResult.passed = false                                  │
│  Recovery:                                                                   │
│  1. Retry with progressive refinement (3 levels)                            │
│  2. Apply partial fallback (return what validated)                          │
│  3. Apply template fallback (predefined response)                           │
│  4. Fail with detailed error                                                │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ Validate ─[fail]─▶ Retry L1 ─[fail]─▶ Retry L2 ─[fail]─▶ Retry L3  │    │
│  │                                                          │           │    │
│  │                                                ─[fail]─▶ Partial    │    │
│  │                                                          │           │    │
│  │                                                ─[fail]─▶ Template   │    │
│  │                                                          │           │    │
│  │                                                ─[fail]─▶ Error      │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  Failure Mode: Token Budget Exceeded                                        │
│  ───────────────────────────────────                                        │
│                                                                              │
│  Detection: TokenBudgetExceededError                                        │
│  Recovery:                                                                   │
│  1. Enable aggressive compression                                           │
│  2. Reduce memory_retrieval_k                                               │
│  3. Reduce history_turns                                                    │
│  4. Return error with budget status                                         │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ Budget Check ─[over]─▶ Compress ─[still over]─▶ Reduce k            │    │
│  │                                                    │                 │    │
│  │                                     ─[still over]─▶ Reduce turns    │    │
│  │                                                    │                 │    │
│  │                                     ─[still over]─▶ Error           │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Circuit Breaker Pattern

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      CIRCUIT BREAKER PATTERN                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│                       ┌──────────────────┐                                  │
│                       │     CLOSED       │                                  │
│                       │ (Normal Operation)│                                  │
│                       └────────┬─────────┘                                  │
│                                │                                             │
│                                │ Failure threshold exceeded                  │
│                                │ (5 failures in 30s)                         │
│                                ▼                                             │
│                       ┌──────────────────┐                                  │
│            ┌─────────│      OPEN        │◀──────────────┐                   │
│            │         │  (Fast Fail)     │               │                   │
│            │         └────────┬─────────┘               │                   │
│            │                  │                         │                   │
│            │                  │ Timeout (60s)           │                   │
│            │                  ▼                         │                   │
│            │         ┌──────────────────┐               │                   │
│            │         │   HALF-OPEN      │               │                   │
│            │         │ (Test Request)   │               │                   │
│            │         └────────┬─────────┘               │                   │
│            │                  │                         │                   │
│            │    Success       │       Failure           │                   │
│            │         ┌────────┴────────┐                │                   │
│            │         ▼                 ▼                │                   │
│            │  ┌──────────────┐  ┌──────────────┐        │                   │
│            └──│   CLOSED     │  │    OPEN      │────────┘                   │
│               └──────────────┘  └──────────────┘                            │
│                                                                              │
│  Implementation:                                                             │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  class CircuitBreaker:                                                │   │
│  │      def __init__(self, failure_threshold=5, recovery_timeout=60):    │   │
│  │          self.state = "closed"                                        │   │
│  │          self.failures = 0                                            │   │
│  │          self.last_failure = None                                     │   │
│  │                                                                       │   │
│  │      async def call(self, func, *args):                               │   │
│  │          if self.state == "open":                                     │   │
│  │              if self._should_attempt_reset():                         │   │
│  │                  self.state = "half-open"                             │   │
│  │              else:                                                    │   │
│  │                  raise CircuitOpenError()                             │   │
│  │                                                                       │   │
│  │          try:                                                         │   │
│  │              result = await func(*args)                               │   │
│  │              self._on_success()                                       │   │
│  │              return result                                            │   │
│  │          except Exception as e:                                       │   │
│  │              self._on_failure()                                       │   │
│  │              raise                                                    │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Observability Architecture

### Metrics Collection

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      METRICS COLLECTION                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Application Metrics (Prometheus)                                           │
│  ────────────────────────────────                                           │
│                                                                              │
│  Request Metrics:                                                            │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │ sigil_requests_total{operation, status}           Counter              │   │
│  │ sigil_request_duration_seconds{operation}         Histogram           │   │
│  │ sigil_requests_in_progress                        Gauge               │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  Token Metrics:                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │ sigil_tokens_used_total{component}                Counter              │   │
│  │ sigil_tokens_per_request{operation}               Histogram           │   │
│  │ sigil_budget_utilization{session}                 Gauge               │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  Subsystem Metrics:                                                          │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │ sigil_memory_retrieval_duration_seconds           Histogram           │   │
│  │ sigil_context_compression_ratio                   Histogram           │   │
│  │ sigil_reasoning_strategy_used{strategy}           Counter              │   │
│  │ sigil_contract_validation_passed{contract}        Counter              │   │
│  │ sigil_evolution_score{agent}                      Gauge               │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  Infrastructure Metrics:                                                     │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │ sigil_websocket_connections                       Gauge               │   │
│  │ sigil_event_store_size                            Gauge               │   │
│  │ sigil_cache_hit_ratio                             Gauge               │   │
│  │ sigil_llm_latency_seconds{provider}               Histogram           │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Logging Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       LOGGING ARCHITECTURE                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Structured Logging (structlog + JSON)                                       │
│  ────────────────────────────────────                                       │
│                                                                              │
│  Log Levels:                                                                 │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │ DEBUG   - Detailed diagnostic information                             │   │
│  │ INFO    - General operational events                                  │   │
│  │ WARNING - Unexpected but recoverable situations                       │   │
│  │ ERROR   - Errors that don't stop the application                      │   │
│  │ CRITICAL- Severe errors requiring immediate attention                 │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  Log Format:                                                                 │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │ {                                                                     │   │
│  │   "timestamp": "2026-01-11T12:00:00.000Z",                           │   │
│  │   "level": "INFO",                                                    │   │
│  │   "logger": "sigil.orchestrator",                                     │   │
│  │   "event": "request_completed",                                       │   │
│  │   "request_id": "550e8400-e29b-41d4-a716-446655440000",              │   │
│  │   "session_id": "session_abc123",                                     │   │
│  │   "operation": "run",                                                 │   │
│  │   "agent_name": "sales_qualifier",                                    │   │
│  │   "tokens_used": 1500,                                                │   │
│  │   "latency_ms": 3500,                                                 │   │
│  │   "success": true                                                     │   │
│  │ }                                                                     │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  Log Pipeline:                                                               │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                                                                      │    │
│  │  Application ──▶ structlog ──▶ JSON ──▶ stdout                      │    │
│  │                                            │                         │    │
│  │                                            ▼                         │    │
│  │                                      Log Collector                   │    │
│  │                                      (Fluentd/Vector)                │    │
│  │                                            │                         │    │
│  │                         ┌──────────────────┼──────────────────┐      │    │
│  │                         ▼                  ▼                  ▼      │    │
│  │                   Elasticsearch      CloudWatch       Datadog       │    │
│  │                                                                      │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Distributed Tracing

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      DISTRIBUTED TRACING                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Trace Example: Agent Run Request                                           │
│  ────────────────────────────────                                           │
│                                                                              │
│  trace_id: abc123                                                           │
│  │                                                                           │
│  ├─ span: http_request (parent)                                             │
│  │  ├─ http.method: POST                                                    │
│  │  ├─ http.url: /v7/agents/sales/run                                       │
│  │  ├─ http.status_code: 200                                                │
│  │  └─ duration: 3500ms                                                     │
│  │                                                                           │
│  │  ├─ span: orchestrator.validate                                          │
│  │  │  └─ duration: 5ms                                                     │
│  │  │                                                                        │
│  │  ├─ span: context_manager.assemble                                       │
│  │  │  ├─ tokens.total: 45000                                               │
│  │  │  ├─ compression.applied: true                                         │
│  │  │  └─ duration: 150ms                                                   │
│  │  │                                                                        │
│  │  │  ├─ span: memory.retrieve                                             │
│  │  │  │  ├─ items_retrieved: 10                                            │
│  │  │  │  └─ duration: 80ms                                                 │
│  │  │                                                                        │
│  │  ├─ span: router.route                                                   │
│  │  │  ├─ complexity: 0.65                                                  │
│  │  │  ├─ use_planning: false                                               │
│  │  │  └─ duration: 100ms                                                   │
│  │  │                                                                        │
│  │  ├─ span: reasoning.execute                                              │
│  │  │  ├─ strategy: chain_of_thought                                        │
│  │  │  └─ duration: 2800ms                                                  │
│  │  │                                                                        │
│  │  │  ├─ span: llm.call                                                    │
│  │  │  │  ├─ provider: anthropic                                            │
│  │  │  │  ├─ model: claude-3                                                │
│  │  │  │  ├─ tokens.input: 40000                                            │
│  │  │  │  ├─ tokens.output: 1200                                            │
│  │  │  │  └─ duration: 2500ms                                               │
│  │  │                                                                        │
│  │  ├─ span: contract.validate                                              │
│  │  │  ├─ contract: lead_qualification                                      │
│  │  │  ├─ passed: true                                                      │
│  │  │  └─ duration: 50ms                                                    │
│  │  │                                                                        │
│  │  └─ span: memory.store                                                   │
│  │     └─ duration: 45ms                                                    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Security Architecture

### Authentication Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      AUTHENTICATION FLOW                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Client                  API Gateway              Auth Service               │
│    │                         │                         │                     │
│    │  Request with JWT       │                         │                     │
│    │  Authorization: Bearer  │                         │                     │
│    │ ───────────────────────▶│                         │                     │
│    │                         │                         │                     │
│    │                         │  Validate JWT           │                     │
│    │                         │ ───────────────────────▶│                     │
│    │                         │                         │                     │
│    │                         │                         │ Verify signature    │
│    │                         │                         │ Check expiration    │
│    │                         │                         │ Check revocation    │
│    │                         │                         │                     │
│    │                         │  Validation Result      │                     │
│    │                         │ ◀───────────────────────│                     │
│    │                         │                         │                     │
│    │                         │ Extract claims:         │                     │
│    │                         │ - user_id               │                     │
│    │                         │ - tenant_id             │                     │
│    │                         │ - scopes                │                     │
│    │                         │                         │                     │
│    │                         │──▶ API Handler          │                     │
│    │                         │                         │                     │
│    │  Response               │                         │                     │
│    │ ◀───────────────────────│                         │                     │
│    │                         │                         │                     │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Data Protection

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        DATA PROTECTION                                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Encryption:                                                                 │
│  ───────────                                                                │
│                                                                              │
│  In Transit:                                                                 │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │ - TLS 1.3 for all API connections                                     │   │
│  │ - WSS for WebSocket connections                                       │   │
│  │ - mTLS for internal service communication                             │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  At Rest:                                                                    │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │ - AES-256 for database encryption                                     │   │
│  │ - KMS for key management                                              │   │
│  │ - Encrypted backups                                                   │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  Sensitive Data Handling:                                                    │
│  ────────────────────────                                                   │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │ Data Type              Storage           Retention                    │   │
│  │ ──────────────────────────────────────────────────────────────────   │   │
│  │ API Keys               Hashed (bcrypt)   Until revoked               │   │
│  │ User Sessions          Redis (encrypted) 24 hours                    │   │
│  │ Conversation History   PostgreSQL        30 days (configurable)      │   │
│  │ Memory Items           PostgreSQL        Until deleted               │   │
│  │ Audit Logs             PostgreSQL        90 days                     │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  PII Handling:                                                               │
│  ────────────                                                               │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │ - PII detection in inputs (opt-in)                                    │   │
│  │ - Automatic redaction in logs                                         │   │
│  │ - Data masking in non-production environments                         │   │
│  │ - Right to deletion support (GDPR)                                    │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Performance Architecture

### Latency Optimization

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     LATENCY OPTIMIZATION                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Request Path Optimization:                                                  │
│  ──────────────────────────                                                 │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                                                                       │   │
│  │  1. Connection Pooling                                                │   │
│  │     - HTTP/2 multiplexing to LLM providers                            │   │
│  │     - Redis connection pool (max: 100)                                │   │
│  │     - PostgreSQL connection pool (max: 50)                            │   │
│  │                                                                       │   │
│  │  2. Parallel Execution                                                │   │
│  │     - Memory retrieval || History loading                             │   │
│  │     - Tool calls executed in parallel (where safe)                    │   │
│  │     - Plan steps executed in dependency waves                         │   │
│  │                                                                       │   │
│  │  3. Caching                                                           │   │
│  │     - Agent config cache (TTL: 5 min)                                 │   │
│  │     - Context cache (TTL: 30 sec)                                     │   │
│  │     - Memory embedding cache (TTL: 1 hour)                            │   │
│  │                                                                       │   │
│  │  4. Early Termination                                                 │   │
│  │     - Fast-path for simple queries (Direct strategy)                  │   │
│  │     - Short-circuit on cache hit                                      │   │
│  │     - Streaming for immediate feedback                                │   │
│  │                                                                       │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  Latency Budget:                                                             │
│  ──────────────                                                             │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                                                                       │   │
│  │  Component              Target P50    Target P95    Budget %          │   │
│  │  ─────────────────────────────────────────────────────────────────   │   │
│  │  Request Parsing        5ms           10ms          0.2%             │   │
│  │  Authentication         10ms          20ms          0.4%             │   │
│  │  Context Assembly       50ms          100ms         2.0%             │   │
│  │  Memory Retrieval       100ms         300ms         4.0%             │   │
│  │  Routing                100ms         200ms         4.0%             │   │
│  │  LLM Execution          2000ms        4000ms        80.0%            │   │
│  │  Contract Validation    50ms          100ms         2.0%             │   │
│  │  Response Assembly      20ms          50ms          0.8%             │   │
│  │  ─────────────────────────────────────────────────────────────────   │   │
│  │  Total (Simple)         2500ms        5000ms        ~100%            │   │
│  │                                                                       │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Migration Strategy

### From Phase 6 to Phase 7

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     MIGRATION: PHASE 6 → PHASE 7                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Phase 1: Preparation (Week 1)                                              │
│  ───────────────────────────────                                            │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │ [ ] Deploy Phase 7 components alongside existing infrastructure       │   │
│  │ [ ] Set feature flags to disabled                                     │   │
│  │ [ ] Verify component initialization                                   │   │
│  │ [ ] Run smoke tests                                                   │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  Phase 2: Shadow Mode (Week 2)                                              │
│  ─────────────────────────────                                              │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │ [ ] Enable shadow routing (requests processed by both old & new)      │   │
│  │ [ ] Compare responses for consistency                                 │   │
│  │ [ ] Monitor latency and error rates                                   │   │
│  │ [ ] Fix discrepancies                                                 │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  Phase 3: Canary Rollout (Week 3)                                           │
│  ─────────────────────────────────                                          │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │ [ ] Route 5% of traffic to Phase 7                                    │   │
│  │ [ ] Monitor closely for 24 hours                                      │   │
│  │ [ ] Increase to 25% if metrics are healthy                            │   │
│  │ [ ] Increase to 50% if metrics are healthy                            │   │
│  │ [ ] Full rollout at 100%                                              │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  Phase 4: Cleanup (Week 4)                                                  │
│  ─────────────────────────                                                  │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │ [ ] Remove old endpoints from load balancer                           │   │
│  │ [ ] Deprecate old API versions                                        │   │
│  │ [ ] Update documentation                                              │   │
│  │ [ ] Archive old code                                                  │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  Rollback Plan:                                                              │
│  ──────────────                                                             │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │ Trigger: Error rate > 5% OR P95 latency > 2x baseline                 │   │
│  │                                                                       │   │
│  │ Steps:                                                                │   │
│  │ 1. Immediately route 100% traffic to old system                       │   │
│  │ 2. Disable Phase 7 feature flags                                      │   │
│  │ 3. Investigate root cause                                             │   │
│  │ 4. Fix and restart canary process                                     │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Appendix: Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Entry Point | Single Orchestrator | Unified error handling, consistent metrics |
| Token Budgeting | Per-component allocation | Fine-grained control, predictable costs |
| Context Compression | Smart multi-strategy | Balance quality vs. tokens |
| Evolution Safety | Approval thresholds | Prevent runaway optimization |
| Event Sourcing | All state changes | Full audit trail, replay capability |
| Streaming | WebSocket with backpressure | Real-time UX, prevent overload |
| Caching | Multi-layer (Redis + local) | Balance freshness vs. latency |
| Error Format | RFC 9457 | Industry standard, rich metadata |

---

## Detailed Component Implementation Specifications

### SigilOrchestrator Implementation

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    SIGILORCHESTRATOR IMPLEMENTATION DETAIL                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Class Structure:                                                            │
│  ────────────────                                                            │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │ class SigilOrchestrator:                                              │   │
│  │     """                                                               │   │
│  │     Central orchestration component for Sigil v2.                     │   │
│  │                                                                       │   │
│  │     Responsibilities:                                                 │   │
│  │     - Request validation and normalization                            │   │
│  │     - Context assembly with token budgeting                           │   │
│  │     - Phase routing (3→4→5→6)                                         │   │
│  │     - Response formatting and delivery                                │   │
│  │     - Error handling and recovery                                     │   │
│  │     - Metric emission                                                 │   │
│  │     """                                                               │   │
│  │                                                                       │   │
│  │     # Configuration                                                   │   │
│  │     config: OrchestratorConfig                                        │   │
│  │                                                                       │   │
│  │     # Dependencies (injected)                                         │   │
│  │     context_manager: ContextManager                                   │   │
│  │     router: Router                                                    │   │
│  │     memory_manager: MemoryManager                                     │   │
│  │     reasoning_manager: ReasoningManager                               │   │
│  │     planner: Planner                                                  │   │
│  │     contract_executor: ContractExecutor                               │   │
│  │     evolution_manager: EvolutionManager                               │   │
│  │     event_store: EventStore                                           │   │
│  │     metrics: MetricsCollector                                         │   │
│  │                                                                       │   │
│  │     # State                                                           │   │
│  │     _active_sessions: Dict[str, SessionState]                         │   │
│  │     _circuit_breaker: CircuitBreaker                                  │   │
│  │     _rate_limiter: RateLimiter                                        │   │
│  │                                                                       │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  Method Specifications:                                                      │
│  ──────────────────────                                                      │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │ async def handle(request: OrchestratorRequest) -> OrchestratorResponse│   │
│  │                                                                       │   │
│  │ Purpose: Process a single request through the full pipeline           │   │
│  │                                                                       │   │
│  │ Algorithm:                                                            │   │
│  │ 1. Validate request schema                                            │   │
│  │ 2. Check rate limits                                                  │   │
│  │ 3. Check circuit breaker                                              │   │
│  │ 4. Create/retrieve session                                            │   │
│  │ 5. Assemble context (via ContextManager)                              │   │
│  │ 6. Route to appropriate handler                                       │   │
│  │ 7. Execute handler with token tracking                                │   │
│  │ 8. Validate output (if contract specified)                            │   │
│  │ 9. Store execution results                                            │   │
│  │ 10. Emit metrics and events                                           │   │
│  │ 11. Return formatted response                                         │   │
│  │                                                                       │   │
│  │ Error Handling:                                                       │   │
│  │ - ValidationError → 400 response                                      │   │
│  │ - RateLimitExceeded → 429 response with retry-after                   │   │
│  │ - CircuitOpen → 503 response                                          │   │
│  │ - TokenBudgetExceeded → 402 response                                  │   │
│  │ - ContractViolation → retry or 422 response                           │   │
│  │ - InternalError → 500 response with correlation ID                    │   │
│  │                                                                       │   │
│  │ Performance Targets:                                                  │   │
│  │ - P50: 2.5s                                                           │   │
│  │ - P95: 5.0s                                                           │   │
│  │ - P99: 8.0s                                                           │   │
│  │                                                                       │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │ async def handle_stream(request: OrchestratorRequest)                 │   │
│  │     -> AsyncGenerator[StreamEvent, None]                              │   │
│  │                                                                       │   │
│  │ Purpose: Process request with streaming response                      │   │
│  │                                                                       │   │
│  │ Algorithm:                                                            │   │
│  │ 1. Same validation as handle()                                        │   │
│  │ 2. Yield status_start event                                           │   │
│  │ 3. Yield context_assembled event                                      │   │
│  │ 4. For each token from LLM:                                           │   │
│  │    a. Check backpressure signal                                       │   │
│  │    b. If backpressure, pause until ACK                                │   │
│  │    c. Yield token_delta event                                         │   │
│  │ 5. For each tool call:                                                │   │
│  │    a. Yield tool_call_start event                                     │   │
│  │    b. Execute tool                                                    │   │
│  │    c. Yield tool_call_result event                                    │   │
│  │ 6. Yield complete event with usage stats                              │   │
│  │                                                                       │   │
│  │ Backpressure Protocol:                                                │   │
│  │ - Client sends PAUSE message when buffer > 80%                        │   │
│  │ - Server queues events (max 1000)                                     │   │
│  │ - Client sends RESUME when buffer < 20%                               │   │
│  │ - Server resumes event emission                                       │   │
│  │ - If queue full, drop oldest non-critical events                      │   │
│  │                                                                       │   │
│  │ Event Types:                                                          │   │
│  │ - status: Processing state changes                                    │   │
│  │ - token_delta: Incremental text output                                │   │
│  │ - tool_call: Tool invocation and result                               │   │
│  │ - error: Error occurred (non-fatal)                                   │   │
│  │ - complete: Request finished                                          │   │
│  │                                                                       │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │ async def create_agent(config: AgentConfig) -> Agent                  │   │
│  │                                                                       │   │
│  │ Purpose: Create a new agent with specified configuration              │   │
│  │                                                                       │   │
│  │ Algorithm:                                                            │   │
│  │ 1. Validate agent configuration                                       │   │
│  │ 2. Generate unique agent ID                                           │   │
│  │ 3. Initialize agent state                                             │   │
│  │ 4. Configure tools and permissions                                    │   │
│  │ 5. Set up memory namespace                                            │   │
│  │ 6. Apply evolution settings                                           │   │
│  │ 7. Store agent metadata                                               │   │
│  │ 8. Emit AgentCreated event                                            │   │
│  │ 9. Return agent handle                                                │   │
│  │                                                                       │   │
│  │ Validation Rules:                                                     │   │
│  │ - name: 3-64 chars, alphanumeric + underscore                         │   │
│  │ - system_prompt: max 10,000 chars                                     │   │
│  │ - tools: must exist in registry                                       │   │
│  │ - token_budget: within allowed range                                  │   │
│  │                                                                       │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │ async def execute_task(                                               │   │
│  │     agent_id: str,                                                    │   │
│  │     task: str,                                                        │   │
│  │     context: Optional[Dict] = None                                    │   │
│  │ ) -> TaskResult                                                       │   │
│  │                                                                       │   │
│  │ Purpose: Execute a task with a specific agent                         │   │
│  │                                                                       │   │
│  │ Algorithm:                                                            │   │
│  │ 1. Load agent configuration                                           │   │
│  │ 2. Create execution session                                           │   │
│  │ 3. Retrieve relevant memories                                         │   │
│  │ 4. Assess task complexity                                             │   │
│  │ 5. Select reasoning strategy                                          │   │
│  │ 6. Generate plan (if complexity > 0.5)                                │   │
│  │ 7. Execute plan/task                                                  │   │
│  │ 8. Validate against contract (if specified)                           │   │
│  │ 9. Extract memories from execution                                    │   │
│  │ 10. Store execution record                                            │   │
│  │ 11. Return result                                                     │   │
│  │                                                                       │   │
│  │ Task Types:                                                           │   │
│  │ - simple: Direct execution, no planning                               │   │
│  │ - complex: Planning + multi-step execution                            │   │
│  │ - contracted: Execution with output validation                        │   │
│  │ - streaming: Progressive output delivery                              │   │
│  │                                                                       │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### ContextManager Implementation

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    CONTEXTMANAGER IMPLEMENTATION DETAIL                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Class Structure:                                                            │
│  ────────────────                                                            │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │ class ContextManager:                                                 │   │
│  │     """                                                               │   │
│  │     Manages context window assembly and compression.                  │   │
│  │                                                                       │   │
│  │     Responsibilities:                                                 │   │
│  │     - Assemble context from multiple sources                          │   │
│  │     - Track token usage per component                                 │   │
│  │     - Apply compression when budget exceeded                          │   │
│  │     - Maintain context quality metrics                                │   │
│  │     """                                                               │   │
│  │                                                                       │   │
│  │     # Configuration                                                   │   │
│  │     total_budget: int = 256_000                                       │   │
│  │     allocations: TokenAllocations                                     │   │
│  │     compression_threshold: float = 0.9                                │   │
│  │                                                                       │   │
│  │     # Components                                                      │   │
│  │     tokenizer: Tokenizer                                              │   │
│  │     compressor: ContextCompressor                                     │   │
│  │     memory_manager: MemoryManager                                     │   │
│  │                                                                       │   │
│  │     # Default Allocations (256K total)                                │   │
│  │     system_budget: int = 20_000      # 7.8%                           │   │
│  │     memory_budget: int = 40_000      # 15.6%                          │   │
│  │     history_budget: int = 50_000     # 19.5%                          │   │
│  │     tools_budget: int = 15_000       # 5.9%                           │   │
│  │     request_budget: int = 20_000     # 7.8%                           │   │
│  │     output_budget: int = 102_400     # 40.0%                          │   │
│  │     reserved_budget: int = 8_600     # 3.4%                           │   │
│  │                                                                       │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  Assembly Algorithm:                                                         │
│  ───────────────────                                                         │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │ async def assemble(request: ContextRequest) -> AssembledContext       │   │
│  │                                                                       │   │
│  │ Step 1: Initialize tracking                                           │   │
│  │ ┌────────────────────────────────────────────────────────────────┐   │   │
│  │ │ used_tokens = {                                                │   │   │
│  │ │     'system': 0, 'memory': 0, 'history': 0,                    │   │   │
│  │ │     'tools': 0, 'request': 0                                   │   │   │
│  │ │ }                                                              │   │   │
│  │ └────────────────────────────────────────────────────────────────┘   │   │
│  │                                                                       │   │
│  │ Step 2: Add system prompt (always first, highest priority)            │   │
│  │ ┌────────────────────────────────────────────────────────────────┐   │   │
│  │ │ system_tokens = tokenize(request.system_prompt)                │   │   │
│  │ │ if system_tokens > system_budget:                              │   │   │
│  │ │     raise SystemPromptTooLarge()                               │   │   │
│  │ │ used_tokens['system'] = system_tokens                          │   │   │
│  │ └────────────────────────────────────────────────────────────────┘   │   │
│  │                                                                       │   │
│  │ Step 3: Add current request                                           │   │
│  │ ┌────────────────────────────────────────────────────────────────┐   │   │
│  │ │ request_tokens = tokenize(request.user_message)                │   │   │
│  │ │ if request_tokens > request_budget:                            │   │   │
│  │ │     truncated = truncate_message(request.user_message,         │   │   │
│  │ │                                  request_budget)               │   │   │
│  │ │ used_tokens['request'] = min(request_tokens, request_budget)   │   │   │
│  │ └────────────────────────────────────────────────────────────────┘   │   │
│  │                                                                       │   │
│  │ Step 4: Add tool definitions                                          │   │
│  │ ┌────────────────────────────────────────────────────────────────┐   │   │
│  │ │ tool_tokens = tokenize(format_tools(request.tools))            │   │   │
│  │ │ if tool_tokens > tools_budget:                                 │   │   │
│  │ │     prioritized_tools = prioritize_tools(request.tools,        │   │   │
│  │ │                                          tools_budget)         │   │   │
│  │ │ used_tokens['tools'] = min(tool_tokens, tools_budget)          │   │   │
│  │ └────────────────────────────────────────────────────────────────┘   │   │
│  │                                                                       │   │
│  │ Step 5: Retrieve and add memories                                     │   │
│  │ ┌────────────────────────────────────────────────────────────────┐   │   │
│  │ │ memories = await memory_manager.retrieve(                      │   │   │
│  │ │     query=request.user_message,                                │   │   │
│  │ │     k=20,                                                      │   │   │
│  │ │     max_tokens=memory_budget                                   │   │   │
│  │ │ )                                                              │   │   │
│  │ │ memory_tokens = tokenize(format_memories(memories))            │   │   │
│  │ │ used_tokens['memory'] = memory_tokens                          │   │   │
│  │ └────────────────────────────────────────────────────────────────┘   │   │
│  │                                                                       │   │
│  │ Step 6: Add conversation history                                      │   │
│  │ ┌────────────────────────────────────────────────────────────────┐   │   │
│  │ │ remaining_budget = history_budget                              │   │   │
│  │ │ history_messages = []                                          │   │   │
│  │ │                                                                │   │   │
│  │ │ # Add messages from most recent to oldest                      │   │   │
│  │ │ for msg in reversed(request.history):                          │   │   │
│  │ │     msg_tokens = tokenize(msg)                                 │   │   │
│  │ │     if msg_tokens <= remaining_budget:                         │   │   │
│  │ │         history_messages.insert(0, msg)                        │   │   │
│  │ │         remaining_budget -= msg_tokens                         │   │   │
│  │ │     else:                                                      │   │   │
│  │ │         break                                                  │   │   │
│  │ │                                                                │   │   │
│  │ │ used_tokens['history'] = history_budget - remaining_budget     │   │   │
│  │ └────────────────────────────────────────────────────────────────┘   │   │
│  │                                                                       │   │
│  │ Step 7: Check if compression needed                                   │   │
│  │ ┌────────────────────────────────────────────────────────────────┐   │   │
│  │ │ total_used = sum(used_tokens.values())                         │   │   │
│  │ │ input_budget = total_budget - output_budget - reserved_budget  │   │   │
│  │ │                                                                │   │   │
│  │ │ if total_used / input_budget > compression_threshold:          │   │   │
│  │ │     context = await compress(context, target=input_budget*0.8) │   │   │
│  │ └────────────────────────────────────────────────────────────────┘   │   │
│  │                                                                       │   │
│  │ Step 8: Return assembled context                                      │   │
│  │ ┌────────────────────────────────────────────────────────────────┐   │   │
│  │ │ return AssembledContext(                                       │   │   │
│  │ │     messages=assembled_messages,                               │   │   │
│  │ │     token_usage=used_tokens,                                   │   │   │
│  │ │     available_output=output_budget,                            │   │   │
│  │ │     compression_applied=compression_applied,                   │   │   │
│  │ │     quality_score=quality_score                                │   │   │
│  │ │ )                                                              │   │   │
│  │ └────────────────────────────────────────────────────────────────┘   │   │
│  │                                                                       │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  Compression Strategies:                                                     │
│  ───────────────────────                                                     │   │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │ Strategy: smart (default)                                             │   │
│  │ ─────────────────────────                                             │   │
│  │                                                                       │   │
│  │ Algorithm:                                                            │   │
│  │ 1. Score each context segment by relevance to current request         │   │
│  │ 2. Identify low-relevance segments                                    │   │
│  │ 3. Apply targeted compression:                                        │   │
│  │    - Remove redundant information                                     │   │
│  │    - Summarize verbose explanations                                   │   │
│  │    - Keep key facts and decisions                                     │   │
│  │ 4. Preserve high-relevance segments intact                            │   │
│  │                                                                       │   │
│  │ Quality Preservation:                                                 │   │
│  │ - Minimum 80% of high-relevance content retained                      │   │
│  │ - Key entities always preserved                                       │   │
│  │ - Decision points never compressed                                    │   │
│  │                                                                       │   │
│  │ Token Reduction: 30-50%                                               │   │
│  │ Quality Impact: Low                                                   │   │
│  │ Latency: 200-500ms                                                    │   │
│  │                                                                       │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │ Strategy: truncate                                                    │   │
│  │ ────────────────────                                                  │   │
│  │                                                                       │   │
│  │ Algorithm:                                                            │   │
│  │ 1. Remove oldest messages first                                       │   │
│  │ 2. Keep most recent N messages                                        │   │
│  │ 3. Always preserve system prompt                                      │   │
│  │                                                                       │   │
│  │ Use When:                                                             │   │
│  │ - Speed is critical                                                   │   │
│  │ - Context is mostly conversational                                    │   │
│  │ - Historical accuracy not required                                    │   │
│  │                                                                       │   │
│  │ Token Reduction: 40-70%                                               │   │
│  │ Quality Impact: Medium                                                │   │
│  │ Latency: 10-50ms                                                      │   │
│  │                                                                       │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │ Strategy: summarize                                                   │   │
│  │ ─────────────────────                                                 │   │
│  │                                                                       │   │
│  │ Algorithm:                                                            │   │
│  │ 1. Group messages into conversation segments                          │   │
│  │ 2. Use LLM to summarize each segment                                  │   │
│  │ 3. Replace segments with summaries                                    │   │
│  │ 4. Keep most recent segment uncompressed                              │   │
│  │                                                                       │   │
│  │ Use When:                                                             │   │
│  │ - Long conversation history                                           │   │
│  │ - Historical context is valuable                                      │   │
│  │ - Quality is more important than speed                                │   │
│  │                                                                       │   │
│  │ Token Reduction: 60-80%                                               │   │
│  │ Quality Impact: Low-Medium                                            │   │
│  │ Latency: 1-3s (requires LLM call)                                     │   │
│  │                                                                       │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Testing Architecture

### Test Categories and Coverage Requirements

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         TESTING ARCHITECTURE                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Test Pyramid:                                                               │
│  ─────────────                                                               │
│                                                                              │
│                           /\                                                 │
│                          /  \                                                │
│                         / E2E \          5% - Critical user journeys         │
│                        /______\                                              │
│                       /        \                                             │
│                      /Integration\       25% - Component interactions        │
│                     /______________\                                         │
│                    /                \                                        │
│                   /    Unit Tests    \   70% - Individual functions          │
│                  /____________________\                                      │
│                                                                              │
│  Coverage Requirements:                                                      │
│  ──────────────────────                                                      │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │ Component              │ Min Coverage │ Critical Paths │ Target      │   │
│  │ ─────────────────────────────────────────────────────────────────────│   │
│  │ SigilOrchestrator      │ 90%          │ handle, stream │ 95%         │   │
│  │ ContextManager         │ 85%          │ assemble       │ 90%         │   │
│  │ EvolutionManager       │ 80%          │ optimize       │ 85%         │   │
│  │ MemoryManager          │ 85%          │ retrieve       │ 90%         │   │
│  │ ReasoningManager       │ 80%          │ reason         │ 85%         │   │
│  │ Planner                │ 80%          │ generate       │ 85%         │   │
│  │ ContractExecutor       │ 90%          │ validate       │ 95%         │   │
│  │ Router                 │ 85%          │ route          │ 90%         │   │
│  │ ─────────────────────────────────────────────────────────────────────│   │
│  │ Overall Target         │ 85%          │ N/A            │ 90%         │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Unit Test Specifications

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         UNIT TEST SPECIFICATIONS                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  SigilOrchestrator Unit Tests:                                               │
│  ─────────────────────────────                                               │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │ test_handle_valid_request:                                            │   │
│  │   Given: Valid OrchestratorRequest                                    │   │
│  │   When: handle() is called                                            │   │
│  │   Then: Returns OrchestratorResponse with status 200                  │   │
│  │   And: Response contains valid content                                │   │
│  │   And: Token usage is tracked                                         │   │
│  │   And: Events are emitted                                             │   │
│  │                                                                       │   │
│  │ test_handle_invalid_request:                                          │   │
│  │   Given: Request missing required fields                              │   │
│  │   When: handle() is called                                            │   │
│  │   Then: Raises ValidationError                                        │   │
│  │   And: Error contains field-level details                             │   │
│  │                                                                       │   │
│  │ test_handle_rate_limited:                                             │   │
│  │   Given: Rate limit exceeded for client                               │   │
│  │   When: handle() is called                                            │   │
│  │   Then: Returns 429 response                                          │   │
│  │   And: Response includes retry-after header                           │   │
│  │                                                                       │   │
│  │ test_handle_circuit_open:                                             │   │
│  │   Given: Circuit breaker is open                                      │   │
│  │   When: handle() is called                                            │   │
│  │   Then: Returns 503 response                                          │   │
│  │   And: Does not call downstream services                              │   │
│  │                                                                       │   │
│  │ test_handle_token_budget_exceeded:                                    │   │
│  │   Given: Session token budget exhausted                               │   │
│  │   When: handle() is called                                            │   │
│  │   Then: Returns 402 response                                          │   │
│  │   And: Response includes usage details                                │   │
│  │                                                                       │   │
│  │ test_handle_contract_violation:                                       │   │
│  │   Given: Output violates contract                                     │   │
│  │   And: Retry policy is enabled                                        │   │
│  │   When: handle() is called                                            │   │
│  │   Then: Retries up to max_retries                                     │   │
│  │   And: Returns best attempt if all retries fail                       │   │
│  │                                                                       │   │
│  │ test_handle_streaming:                                                │   │
│  │   Given: Valid streaming request                                      │   │
│  │   When: handle_stream() is called                                     │   │
│  │   Then: Yields sequence of StreamEvents                               │   │
│  │   And: First event is status_start                                    │   │
│  │   And: Last event is complete                                         │   │
│  │   And: Token deltas are properly chunked                              │   │
│  │                                                                       │   │
│  │ test_handle_stream_backpressure:                                      │   │
│  │   Given: Client sends PAUSE signal                                    │   │
│  │   When: Streaming is in progress                                      │   │
│  │   Then: Event emission pauses                                         │   │
│  │   And: Events are queued                                              │   │
│  │   When: Client sends RESUME signal                                    │   │
│  │   Then: Queued events are emitted                                     │   │
│  │   And: Normal streaming resumes                                       │   │
│  │                                                                       │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  ContextManager Unit Tests:                                                  │
│  ──────────────────────────                                                  │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │ test_assemble_basic:                                                  │   │
│  │   Given: Request with system prompt, user message, tools              │   │
│  │   When: assemble() is called                                          │   │
│  │   Then: Returns AssembledContext                                      │   │
│  │   And: All components are included                                    │   │
│  │   And: Token counts are accurate                                      │   │
│  │                                                                       │   │
│  │ test_assemble_with_memory:                                            │   │
│  │   Given: Request with memory retrieval enabled                        │   │
│  │   When: assemble() is called                                          │   │
│  │   Then: Relevant memories are included                                │   │
│  │   And: Memory token budget is respected                               │   │
│  │                                                                       │   │
│  │ test_assemble_compression_triggered:                                  │   │
│  │   Given: Context exceeds 90% of budget                                │   │
│  │   When: assemble() is called                                          │   │
│  │   Then: Compression is applied                                        │   │
│  │   And: Final size is within budget                                    │   │
│  │   And: compression_applied flag is True                               │   │
│  │                                                                       │   │
│  │ test_assemble_history_truncation:                                     │   │
│  │   Given: History exceeds history_budget                               │   │
│  │   When: assemble() is called                                          │   │
│  │   Then: Oldest messages are dropped                                   │   │
│  │   And: Most recent messages are preserved                             │   │
│  │                                                                       │   │
│  │ test_compress_smart_strategy:                                         │   │
│  │   Given: Context with mixed relevance segments                        │   │
│  │   When: compress(strategy='smart') is called                          │   │
│  │   Then: Low-relevance segments are compressed                         │   │
│  │   And: High-relevance segments are preserved                          │   │
│  │   And: Reduction is 30-50%                                            │   │
│  │                                                                       │   │
│  │ test_compress_truncate_strategy:                                      │   │
│  │   Given: Long conversation history                                    │   │
│  │   When: compress(strategy='truncate') is called                       │   │
│  │   Then: Oldest messages are removed                                   │   │
│  │   And: System prompt is preserved                                     │   │
│  │   And: Reduction is 40-70%                                            │   │
│  │                                                                       │   │
│  │ test_compress_summarize_strategy:                                     │   │
│  │   Given: Multi-segment conversation                                   │   │
│  │   When: compress(strategy='summarize') is called                      │   │
│  │   Then: Older segments are summarized                                 │   │
│  │   And: Recent segment is uncompressed                                 │   │
│  │   And: Reduction is 60-80%                                            │   │
│  │                                                                       │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  EvolutionManager Unit Tests:                                                │
│  ────────────────────────────                                                │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │ test_evaluate_agent:                                                  │   │
│  │   Given: Agent with execution history                                 │   │
│  │   When: evaluate() is called                                          │   │
│  │   Then: Returns EvaluationResult                                      │   │
│  │   And: Score is between 0 and 1                                       │   │
│  │   And: Dimension scores are included                                  │   │
│  │                                                                       │   │
│  │ test_evaluate_custom_criteria:                                        │   │
│  │   Given: Custom evaluation criteria                                   │   │
│  │   When: evaluate() is called with criteria                            │   │
│  │   Then: Custom criteria are evaluated                                 │   │
│  │   And: Results include custom dimension scores                        │   │
│  │                                                                       │   │
│  │ test_optimize_approved:                                               │   │
│  │   Given: Optimization with score improvement > 10%                    │   │
│  │   And: auto_approve_threshold = 0.1                                   │   │
│  │   When: optimize() is called                                          │   │
│  │   Then: New version is automatically approved                         │   │
│  │   And: Agent is updated                                               │   │
│  │                                                                       │   │
│  │ test_optimize_pending_approval:                                       │   │
│  │   Given: Optimization with score improvement < threshold              │   │
│  │   When: optimize() is called                                          │   │
│  │   Then: New version is created with pending status                    │   │
│  │   And: Agent is NOT updated                                           │   │
│  │   And: Approval request is generated                                  │   │
│  │                                                                       │   │
│  │ test_optimize_safety_violated:                                        │   │
│  │   Given: Optimization that violates safety constraints                │   │
│  │   When: optimize() is called                                          │   │
│  │   Then: Optimization is rejected                                      │   │
│  │   And: SafetyViolation error is raised                                │   │
│  │   And: No version is created                                          │   │
│  │                                                                       │   │
│  │ test_rollback:                                                        │   │
│  │   Given: Agent with multiple versions                                 │   │
│  │   When: rollback() is called with target version                      │   │
│  │   Then: Agent reverts to target version                               │   │
│  │   And: Rollback event is recorded                                     │   │
│  │                                                                       │   │
│  │ test_compare_versions:                                                │   │
│  │   Given: Two agent versions                                           │   │
│  │   When: compare_versions() is called                                  │   │
│  │   Then: Returns detailed comparison                                   │   │
│  │   And: Includes diff of configurations                                │   │
│  │   And: Includes performance delta                                     │   │
│  │                                                                       │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Integration Test Specifications

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     INTEGRATION TEST SPECIFICATIONS                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Cross-Phase Integration Tests:                                              │
│  ──────────────────────────────                                              │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │ test_orchestrator_to_router_integration:                              │   │
│  │   Setup: SigilOrchestrator with live Router                           │   │
│  │   Given: Request requiring routing decision                           │   │
│  │   When: Orchestrator processes request                                │   │
│  │   Then: Router receives properly formatted input                      │   │
│  │   And: Route decision is applied                                      │   │
│  │   And: Metrics are recorded for both components                       │   │
│  │                                                                       │   │
│  │ test_orchestrator_to_memory_integration:                              │   │
│  │   Setup: SigilOrchestrator with live MemoryManager                    │   │
│  │   Given: Request requiring memory retrieval                           │   │
│  │   When: Orchestrator processes request                                │   │
│  │   Then: Relevant memories are retrieved                               │   │
│  │   And: Memories are included in context                               │   │
│  │   And: Memory tokens are tracked                                      │   │
│  │                                                                       │   │
│  │ test_orchestrator_to_reasoning_integration:                           │   │
│  │   Setup: SigilOrchestrator with live ReasoningManager                 │   │
│  │   Given: Request with high complexity (> 0.7)                         │   │
│  │   When: Orchestrator processes request                                │   │
│  │   Then: Appropriate reasoning strategy is selected                    │   │
│  │   And: Strategy execution completes                                   │   │
│  │   And: Result is properly formatted                                   │   │
│  │                                                                       │   │
│  │ test_orchestrator_to_planner_integration:                             │   │
│  │   Setup: SigilOrchestrator with live Planner                          │   │
│  │   Given: Complex goal requiring planning                              │   │
│  │   When: Orchestrator processes request                                │   │
│  │   Then: Plan is generated                                             │   │
│  │   And: Plan steps are executed                                        │   │
│  │   And: Results are aggregated                                         │   │
│  │                                                                       │   │
│  │ test_orchestrator_to_contract_integration:                            │   │
│  │   Setup: SigilOrchestrator with live ContractExecutor                 │   │
│  │   Given: Request with contract specification                          │   │
│  │   When: Orchestrator processes request                                │   │
│  │   Then: Output is validated against contract                          │   │
│  │   And: Validation result is included in response                      │   │
│  │                                                                       │   │
│  │ test_orchestrator_to_evolution_integration:                           │   │
│  │   Setup: SigilOrchestrator with live EvolutionManager                 │   │
│  │   Given: Agent with evolution enabled                                 │   │
│  │   When: Execution completes                                           │   │
│  │   Then: Execution is recorded for evaluation                          │   │
│  │   And: Periodic optimization triggers                                 │   │
│  │                                                                       │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  Data Flow Integration Tests:                                                │
│  ────────────────────────────                                                │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │ test_full_request_lifecycle:                                          │   │
│  │   Given: Complete request with all features enabled                   │   │
│  │   When: Request flows through system                                  │   │
│  │   Then: Each phase receives correct input                             │   │
│  │   And: Each phase produces correct output                             │   │
│  │   And: Token tracking is consistent across phases                     │   │
│  │   And: Events are emitted in correct order                            │   │
│  │                                                                       │   │
│  │ test_memory_to_context_flow:                                          │   │
│  │   Given: Memories stored in MemoryManager                             │   │
│  │   When: ContextManager retrieves memories                             │   │
│  │   Then: Retrieved memories match query relevance                      │   │
│  │   And: Memory budget is respected                                     │   │
│  │   And: Token counts are accurate                                      │   │
│  │                                                                       │   │
│  │ test_routing_to_reasoning_flow:                                       │   │
│  │   Given: Router assesses complexity                                   │   │
│  │   When: Complexity is passed to ReasoningManager                      │   │
│  │   Then: Correct strategy is selected                                  │   │
│  │   And: Strategy parameters match complexity                           │   │
│  │                                                                       │   │
│  │ test_evolution_feedback_loop:                                         │   │
│  │   Given: Agent execution completes                                    │   │
│  │   When: Execution is evaluated                                        │   │
│  │   And: Optimization is triggered                                      │   │
│  │   Then: New version reflects improvements                             │   │
│  │   And: Future executions use optimized version                        │   │
│  │                                                                       │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  Event Flow Integration Tests:                                               │
│  ─────────────────────────────                                               │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │ test_event_emission_order:                                            │   │
│  │   Given: Request processing starts                                    │   │
│  │   When: Request flows through all phases                              │   │
│  │   Then: Events are emitted in expected order:                         │   │
│  │         1. RequestReceived                                            │   │
│  │         2. ContextAssembled                                           │   │
│  │         3. RouteDecided                                               │   │
│  │         4. ExecutionStarted                                           │   │
│  │         5. [Phase-specific events]                                    │   │
│  │         6. ExecutionCompleted                                         │   │
│  │         7. ResponseSent                                               │   │
│  │                                                                       │   │
│  │ test_event_store_persistence:                                         │   │
│  │   Given: Events emitted during request                                │   │
│  │   When: Request completes                                             │   │
│  │   Then: All events are persisted                                      │   │
│  │   And: Events can be replayed                                         │   │
│  │   And: State can be reconstructed from events                         │   │
│  │                                                                       │   │
│  │ test_event_subscription_delivery:                                     │   │
│  │   Given: External subscriber registered                               │   │
│  │   When: Events are emitted                                            │   │
│  │   Then: Subscriber receives all events                                │   │
│  │   And: Events arrive in order                                         │   │
│  │   And: Delivery is confirmed                                          │   │
│  │                                                                       │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### End-to-End Test Specifications

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       END-TO-END TEST SPECIFICATIONS                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Critical User Journeys:                                                     │
│  ───────────────────────                                                     │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │ E2E Test: Agent Creation and Task Execution                          │   │
│  │ ─────────────────────────────────────────────                        │   │
│  │                                                                       │   │
│  │ Scenario: User creates an agent and executes a task                   │   │
│  │                                                                       │   │
│  │ Steps:                                                                │   │
│  │ 1. POST /api/v1/agents with configuration                             │   │
│  │    Expected: 201 Created with agent_id                                │   │
│  │                                                                       │   │
│  │ 2. GET /api/v1/agents/{agent_id}                                      │   │
│  │    Expected: 200 OK with agent details                                │   │
│  │                                                                       │   │
│  │ 3. POST /api/v1/agents/{agent_id}/execute with task                   │   │
│  │    Expected: 200 OK with task result                                  │   │
│  │                                                                       │   │
│  │ 4. GET /api/v1/agents/{agent_id}/sessions                             │   │
│  │    Expected: Session with execution record                            │   │
│  │                                                                       │   │
│  │ Assertions:                                                           │   │
│  │ - Agent is persisted in database                                      │   │
│  │ - Task result contains expected output                                │   │
│  │ - Token usage is recorded                                             │   │
│  │ - Events are stored in event store                                    │   │
│  │                                                                       │   │
│  │ Duration: < 30s                                                       │   │
│  │ Priority: P0 (Critical)                                               │   │
│  │                                                                       │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │ E2E Test: Streaming Response with Backpressure                        │   │
│  │ ──────────────────────────────────────────────                        │   │
│  │                                                                       │   │
│  │ Scenario: Client receives streaming response with flow control        │   │
│  │                                                                       │   │
│  │ Steps:                                                                │   │
│  │ 1. Open WebSocket to /ws/v1/agents/{agent_id}/stream                  │   │
│  │    Expected: Connection established                                   │   │
│  │                                                                       │   │
│  │ 2. Send execute message with task                                     │   │
│  │    Expected: Receive status_start event                               │   │
│  │                                                                       │   │
│  │ 3. Receive token_delta events                                         │   │
│  │    Expected: Tokens arrive in order                                   │   │
│  │                                                                       │   │
│  │ 4. Client sends PAUSE signal                                          │   │
│  │    Expected: Server pauses event emission                             │   │
│  │                                                                       │   │
│  │ 5. Client sends RESUME signal                                         │   │
│  │    Expected: Queued events delivered                                  │   │
│  │                                                                       │   │
│  │ 6. Receive complete event                                             │   │
│  │    Expected: Total token count matches events                         │   │
│  │                                                                       │   │
│  │ Assertions:                                                           │   │
│  │ - No events lost during pause                                         │   │
│  │ - Event order preserved                                               │   │
│  │ - Connection gracefully closed                                        │   │
│  │                                                                       │   │
│  │ Duration: < 60s                                                       │   │
│  │ Priority: P0 (Critical)                                               │   │
│  │                                                                       │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │ E2E Test: Memory-Augmented Execution                                  │   │
│  │ ────────────────────────────────────                                  │   │
│  │                                                                       │   │
│  │ Scenario: Agent uses stored memories to complete task                 │   │
│  │                                                                       │   │
│  │ Steps:                                                                │   │
│  │ 1. POST /api/v1/memory/resources with test document                   │   │
│  │    Expected: 201 Created with resource_id                             │   │
│  │                                                                       │   │
│  │ 2. POST /api/v1/memory/extract with resource_id                       │   │
│  │    Expected: Memory items extracted                                   │   │
│  │                                                                       │   │
│  │ 3. POST /api/v1/agents/{agent_id}/execute with related task           │   │
│  │    Expected: Response references stored information                   │   │
│  │                                                                       │   │
│  │ 4. GET /api/v1/memory/items?query=test                                │   │
│  │    Expected: Items used in context returned                           │   │
│  │                                                                       │   │
│  │ Assertions:                                                           │   │
│  │ - Memory retrieval triggered                                          │   │
│  │ - Context includes relevant memories                                  │   │
│  │ - Response quality improved vs. no-memory baseline                    │   │
│  │                                                                       │   │
│  │ Duration: < 45s                                                       │   │
│  │ Priority: P1 (High)                                                   │   │
│  │                                                                       │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │ E2E Test: Contract-Validated Execution                                │   │
│  │ ──────────────────────────────────────                                │   │
│  │                                                                       │   │
│  │ Scenario: Agent output validated against contract                     │   │
│  │                                                                       │   │
│  │ Steps:                                                                │   │
│  │ 1. POST /api/v1/contracts with contract definition                    │   │
│  │    Expected: 201 Created with contract_id                             │   │
│  │                                                                       │   │
│  │ 2. POST /api/v1/agents with contract_id in config                     │   │
│  │    Expected: Agent created with contract                              │   │
│  │                                                                       │   │
│  │ 3. POST /api/v1/agents/{agent_id}/execute with task                   │   │
│  │    Expected: Response includes validation_result                      │   │
│  │                                                                       │   │
│  │ 4. Verify validation_result.valid = true                              │   │
│  │    Expected: All deliverables pass validation                         │   │
│  │                                                                       │   │
│  │ Assertions:                                                           │   │
│  │ - Contract retrieved before execution                                 │   │
│  │ - Output schema matches contract                                      │   │
│  │ - Retry triggered if initial output invalid                           │   │
│  │                                                                       │   │
│  │ Duration: < 60s                                                       │   │
│  │ Priority: P1 (High)                                                   │   │
│  │                                                                       │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │ E2E Test: Agent Evolution Cycle                                       │   │
│  │ ─────────────────────────────────                                     │   │
│  │                                                                       │   │
│  │ Scenario: Agent is evaluated and optimized                            │   │
│  │                                                                       │   │
│  │ Steps:                                                                │   │
│  │ 1. Execute multiple tasks with agent (10+ executions)                 │   │
│  │    Expected: Execution history stored                                 │   │
│  │                                                                       │   │
│  │ 2. POST /api/v1/evolution/evaluate with agent_id                      │   │
│  │    Expected: Evaluation score returned                                │   │
│  │                                                                       │   │
│  │ 3. POST /api/v1/evolution/optimize with agent_id                      │   │
│  │    Expected: New version created                                      │   │
│  │                                                                       │   │
│  │ 4. GET /api/v1/agents/{agent_id}/versions                             │   │
│  │    Expected: Multiple versions listed                                 │   │
│  │                                                                       │   │
│  │ 5. POST /api/v1/evolution/compare                                     │   │
│  │    Expected: Version comparison returned                              │   │
│  │                                                                       │   │
│  │ Assertions:                                                           │   │
│  │ - New version has higher score                                        │   │
│  │ - Changes are documented                                              │   │
│  │ - Rollback available if needed                                        │   │
│  │                                                                       │   │
│  │ Duration: < 120s                                                      │   │
│  │ Priority: P2 (Medium)                                                 │   │
│  │                                                                       │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │ E2E Test: Error Recovery and Circuit Breaker                          │   │
│  │ ────────────────────────────────────────────                          │   │
│  │                                                                       │   │
│  │ Scenario: System handles errors gracefully                            │   │
│  │                                                                       │   │
│  │ Steps:                                                                │   │
│  │ 1. Simulate LLM provider outage                                       │   │
│  │    Expected: Requests fail with 503                                   │   │
│  │                                                                       │   │
│  │ 2. Send 5 consecutive requests                                        │   │
│  │    Expected: Circuit breaker opens                                    │   │
│  │                                                                       │   │
│  │ 3. Send request while circuit open                                    │   │
│  │    Expected: Fast-fail 503 without LLM call                           │   │
│  │                                                                       │   │
│  │ 4. Restore LLM provider                                               │   │
│  │    Wait: 30 seconds (half-open timeout)                               │   │
│  │                                                                       │   │
│  │ 5. Send request                                                       │   │
│  │    Expected: Circuit closes on success                                │   │
│  │                                                                       │   │
│  │ 6. Send normal request                                                │   │
│  │    Expected: 200 OK                                                   │   │
│  │                                                                       │   │
│  │ Assertions:                                                           │   │
│  │ - Circuit state transitions correctly                                 │   │
│  │ - Metrics recorded for failures                                       │   │
│  │ - No cascade failures                                                 │   │
│  │                                                                       │   │
│  │ Duration: < 90s                                                       │   │
│  │ Priority: P1 (High)                                                   │   │
│  │                                                                       │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Capacity Planning

### Resource Requirements

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         CAPACITY PLANNING                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Workload Profiles:                                                          │
│  ──────────────────                                                          │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │ Profile: Light (Startup / Development)                                │   │
│  │ ──────────────────────────────────────                                │   │
│  │                                                                       │   │
│  │ Concurrent Users: 10-50                                               │   │
│  │ Requests/second: 5-20                                                 │   │
│  │ Agents: 10-100                                                        │   │
│  │ Memory Items: 10,000-100,000                                          │   │
│  │                                                                       │   │
│  │ Infrastructure:                                                       │   │
│  │ ┌────────────────────────────────────────────────────────────────┐   │   │
│  │ │ Component          │ Instances │ CPU    │ Memory │ Storage     │   │   │
│  │ │ ──────────────────────────────────────────────────────────────│   │   │
│  │ │ Application        │ 2         │ 2 core │ 4 GB   │ N/A         │   │   │
│  │ │ PostgreSQL         │ 1         │ 2 core │ 8 GB   │ 100 GB SSD  │   │   │
│  │ │ Redis              │ 1         │ 1 core │ 2 GB   │ N/A         │   │   │
│  │ │ Vector DB          │ 1         │ 2 core │ 4 GB   │ 50 GB SSD   │   │   │
│  │ │ ──────────────────────────────────────────────────────────────│   │   │
│  │ │ Total              │ 5         │ 7 core │ 18 GB  │ 150 GB      │   │   │
│  │ └────────────────────────────────────────────────────────────────┘   │   │
│  │                                                                       │   │
│  │ Estimated Cost: $500-800/month (cloud)                                │   │
│  │                                                                       │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │ Profile: Medium (Growth Stage)                                        │   │
│  │ ─────────────────────────────────                                     │   │
│  │                                                                       │   │
│  │ Concurrent Users: 100-500                                             │   │
│  │ Requests/second: 50-200                                               │   │
│  │ Agents: 500-2,000                                                     │   │
│  │ Memory Items: 1M-10M                                                  │   │
│  │                                                                       │   │
│  │ Infrastructure:                                                       │   │
│  │ ┌────────────────────────────────────────────────────────────────┐   │   │
│  │ │ Component          │ Instances │ CPU    │ Memory │ Storage     │   │   │
│  │ │ ──────────────────────────────────────────────────────────────│   │   │
│  │ │ Application        │ 4         │ 4 core │ 8 GB   │ N/A         │   │   │
│  │ │ PostgreSQL Primary │ 1         │ 4 core │ 16 GB  │ 500 GB SSD  │   │   │
│  │ │ PostgreSQL Replica │ 2         │ 2 core │ 8 GB   │ 500 GB SSD  │   │   │
│  │ │ Redis Cluster      │ 3         │ 2 core │ 4 GB   │ N/A         │   │   │
│  │ │ Vector DB Cluster  │ 3         │ 4 core │ 8 GB   │ 200 GB SSD  │   │   │
│  │ │ Load Balancer      │ 2         │ 1 core │ 2 GB   │ N/A         │   │   │
│  │ │ ──────────────────────────────────────────────────────────────│   │   │
│  │ │ Total              │ 15        │ 35 core│ 82 GB  │ 1.7 TB      │   │   │
│  │ └────────────────────────────────────────────────────────────────┘   │   │
│  │                                                                       │   │
│  │ Estimated Cost: $3,000-5,000/month (cloud)                            │   │
│  │                                                                       │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │ Profile: Heavy (Enterprise / Scale)                                   │   │
│  │ ───────────────────────────────────                                   │   │
│  │                                                                       │   │
│  │ Concurrent Users: 1,000-10,000                                        │   │
│  │ Requests/second: 500-2,000                                            │   │
│  │ Agents: 10,000-100,000                                                │   │
│  │ Memory Items: 100M+                                                   │   │
│  │                                                                       │   │
│  │ Infrastructure:                                                       │   │
│  │ ┌────────────────────────────────────────────────────────────────┐   │   │
│  │ │ Component          │ Instances │ CPU    │ Memory │ Storage     │   │   │
│  │ │ ──────────────────────────────────────────────────────────────│   │   │
│  │ │ Application        │ 10-20     │ 8 core │ 16 GB  │ N/A         │   │   │
│  │ │ PostgreSQL Primary │ 1         │ 16 core│ 64 GB  │ 2 TB SSD    │   │   │
│  │ │ PostgreSQL Replica │ 4         │ 8 core │ 32 GB  │ 2 TB SSD    │   │   │
│  │ │ Redis Cluster      │ 6         │ 4 core │ 16 GB  │ N/A         │   │   │
│  │ │ Vector DB Cluster  │ 6         │ 8 core │ 32 GB  │ 1 TB SSD    │   │   │
│  │ │ Load Balancer      │ 2         │ 2 core │ 4 GB   │ N/A         │   │   │
│  │ │ Message Queue      │ 3         │ 4 core │ 8 GB   │ 500 GB SSD  │   │   │
│  │ │ ──────────────────────────────────────────────────────────────│   │   │
│  │ │ Total              │ 32-42     │ 180+   │ 500+ GB│ 15+ TB      │   │   │
│  │ └────────────────────────────────────────────────────────────────┘   │   │
│  │                                                                       │   │
│  │ Estimated Cost: $15,000-30,000/month (cloud)                          │   │
│  │                                                                       │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  Scaling Triggers:                                                           │
│  ─────────────────                                                           │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │ Metric                        │ Scale Up When    │ Scale Down When   │   │
│  │ ──────────────────────────────────────────────────────────────────── │   │
│  │ CPU Utilization               │ > 70% for 5 min  │ < 30% for 15 min  │   │
│  │ Memory Utilization            │ > 80% for 5 min  │ < 40% for 15 min  │   │
│  │ Request Queue Depth           │ > 100 pending    │ < 10 pending      │   │
│  │ Response Latency (P95)        │ > 5s             │ < 2s              │   │
│  │ WebSocket Connections         │ > 80% capacity   │ < 40% capacity    │   │
│  │ Database Connection Pool      │ > 80% used       │ < 40% used        │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Disaster Recovery Architecture

### Recovery Objectives and Procedures

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     DISASTER RECOVERY ARCHITECTURE                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Recovery Objectives:                                                        │
│  ────────────────────                                                        │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │ Metric                    │ Target      │ Maximum     │ Notes        │   │
│  │ ────────────────────────────────────────────────────────────────────│   │
│  │ Recovery Time Objective   │ 15 minutes  │ 1 hour      │ Full service │   │
│  │ Recovery Point Objective  │ 5 minutes   │ 15 minutes  │ Data loss    │   │
│  │ Mean Time to Recovery     │ 30 minutes  │ 2 hours     │ Avg recovery │   │
│  │ Service Level Agreement   │ 99.9%       │ 99.5%       │ Monthly      │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  Backup Strategy:                                                            │
│  ────────────────                                                            │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │ Data Type           │ Frequency      │ Retention  │ Location         │   │
│  │ ────────────────────────────────────────────────────────────────────│   │
│  │ PostgreSQL Full     │ Daily          │ 30 days    │ S3 Cross-Region  │   │
│  │ PostgreSQL WAL      │ Continuous     │ 7 days     │ S3 Same-Region   │   │
│  │ Vector DB Snapshot  │ Daily          │ 14 days    │ S3 Cross-Region  │   │
│  │ Redis RDB           │ Hourly         │ 24 hours   │ S3 Same-Region   │   │
│  │ Event Store Archive │ Daily          │ 90 days    │ Glacier          │   │
│  │ Configuration       │ On Change      │ Unlimited  │ Git + S3         │   │
│  │ Agent Definitions   │ Daily          │ 365 days   │ S3 Cross-Region  │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  Failure Scenarios and Recovery:                                             │
│  ───────────────────────────────                                             │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │ Scenario: Single Application Instance Failure                         │   │
│  │ ────────────────────────────────────────────                          │   │
│  │                                                                       │   │
│  │ Detection: Health check fails (30 seconds)                            │   │
│  │ Impact: Minimal - load balancer routes to healthy instances           │   │
│  │                                                                       │   │
│  │ Recovery Steps:                                                       │   │
│  │ 1. Kubernetes automatically restarts pod (automatic)                  │   │
│  │ 2. Pod passes health checks                                           │   │
│  │ 3. Load balancer adds pod back to pool                                │   │
│  │                                                                       │   │
│  │ Recovery Time: 1-2 minutes (automatic)                                │   │
│  │                                                                       │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │ Scenario: Database Primary Failure                                    │   │
│  │ ────────────────────────────────────                                  │   │
│  │                                                                       │   │
│  │ Detection: Connection failures, replication lag spike                 │   │
│  │ Impact: Write operations fail until failover                          │   │
│  │                                                                       │   │
│  │ Recovery Steps:                                                       │   │
│  │ 1. Detect primary failure (30 seconds)                                │   │
│  │ 2. Promote replica to primary (1-2 minutes)                           │   │
│  │ 3. Update connection strings (automatic via PgBouncer)                │   │
│  │ 4. Verify replication from new primary                                │   │
│  │ 5. Create new replica from promoted primary                           │   │
│  │                                                                       │   │
│  │ Recovery Time: 3-5 minutes                                            │   │
│  │ Data Loss: < 5 seconds (synchronous replication)                      │   │
│  │                                                                       │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │ Scenario: Redis Cluster Node Failure                                  │   │
│  │ ───────────────────────────────────                                   │   │
│  │                                                                       │   │
│  │ Detection: Cluster health check fails                                 │   │
│  │ Impact: Cache misses increase, slight latency increase                │   │
│  │                                                                       │   │
│  │ Recovery Steps:                                                       │   │
│  │ 1. Redis Cluster detects node failure (15 seconds)                    │   │
│  │ 2. Replica promoted to master (automatic)                             │   │
│  │ 3. Cluster rebalances slots                                           │   │
│  │ 4. Replace failed node                                                │   │
│  │                                                                       │   │
│  │ Recovery Time: 1-2 minutes (automatic)                                │   │
│  │ Data Loss: None (with replicas)                                       │   │
│  │                                                                       │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │ Scenario: LLM Provider Outage                                         │   │
│  │ ────────────────────────────────                                      │   │
│  │                                                                       │   │
│  │ Detection: API errors, timeout rate increase                          │   │
│  │ Impact: Agent executions fail or degrade                              │   │
│  │                                                                       │   │
│  │ Recovery Steps:                                                       │   │
│  │ 1. Circuit breaker opens (automatic, 5 failures)                      │   │
│  │ 2. Failover to secondary provider (if configured)                     │   │
│  │ 3. Queue non-urgent requests                                          │   │
│  │ 4. Return cached responses where applicable                           │   │
│  │ 5. Monitor primary provider health                                    │   │
│  │ 6. Circuit breaker half-opens after timeout                           │   │
│  │ 7. Resume normal operation on success                                 │   │
│  │                                                                       │   │
│  │ Recovery Time: Depends on provider                                    │   │
│  │ Mitigation: Multi-provider configuration                              │   │
│  │                                                                       │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │ Scenario: Complete Region Failure                                     │   │
│  │ ───────────────────────────────                                       │   │
│  │                                                                       │   │
│  │ Detection: Multiple health check failures, no connectivity            │   │
│  │ Impact: Complete service outage until failover                        │   │
│  │                                                                       │   │
│  │ Recovery Steps:                                                       │   │
│  │ 1. Detect region failure (1-2 minutes)                                │   │
│  │ 2. DNS failover to DR region (5 minutes TTL)                          │   │
│  │ 3. Activate DR infrastructure                                         │   │
│  │ 4. Restore latest backups if needed                                   │   │
│  │ 5. Verify service functionality                                       │   │
│  │ 6. Update status page                                                 │   │
│  │ 7. Monitor DR region                                                  │   │
│  │                                                                       │   │
│  │ Recovery Time: 15-30 minutes                                          │   │
│  │ Data Loss: Up to RPO (5-15 minutes)                                   │   │
│  │                                                                       │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  DR Runbook Checklist:                                                       │
│  ─────────────────────                                                       │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │ [ ] 1. Confirm incident severity and scope                            │   │
│  │ [ ] 2. Page on-call engineer if not already                           │   │
│  │ [ ] 3. Open incident channel in Slack                                 │   │
│  │ [ ] 4. Begin status page incident                                     │   │
│  │ [ ] 5. Execute appropriate recovery procedure                         │   │
│  │ [ ] 6. Verify service restoration                                     │   │
│  │ [ ] 7. Run smoke tests                                                │   │
│  │ [ ] 8. Update status page                                             │   │
│  │ [ ] 9. Document incident timeline                                     │   │
│  │ [ ] 10. Schedule post-mortem                                          │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## API Versioning Strategy

### Version Management and Deprecation

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         API VERSIONING STRATEGY                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Versioning Scheme:                                                          │
│  ──────────────────                                                          │
│                                                                              │
│  URL Path Versioning: /api/v{major}/resource                                 │
│                                                                              │
│  Examples:                                                                   │
│  - /api/v1/agents                                                            │
│  - /api/v2/agents                                                            │
│  - /ws/v1/stream                                                             │
│                                                                              │
│  Version Lifecycle:                                                          │
│  ──────────────────                                                          │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                                                                       │   │
│  │  CURRENT ──────────► DEPRECATED ──────────► SUNSET ──────────► REMOVED│   │
│  │     │                     │                    │                  │    │   │
│  │     │                     │                    │                  │    │   │
│  │  Active              Warning Headers       Error Response      404     │   │
│  │  Development         Sunset Date           Limited Support    Gone     │   │
│  │                      Announced             Read-Only Mode              │   │
│  │                                                                       │   │
│  │  Timeline:                                                            │   │
│  │  ─────────                                                            │   │
│  │  Current: Ongoing until next major version                            │   │
│  │  Deprecated: 6 months minimum                                         │   │
│  │  Sunset: 3 months minimum                                             │   │
│  │  Removed: After sunset period                                         │   │
│  │                                                                       │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  Deprecation Headers:                                                        │
│  ────────────────────                                                        │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │ When endpoint is deprecated, include these headers:                   │   │
│  │                                                                       │   │
│  │ Deprecation: true                                                     │   │
│  │ Sunset: Sat, 01 Jul 2026 00:00:00 GMT                                 │   │
│  │ Link: </api/v2/agents>; rel="successor-version"                       │   │
│  │                                                                       │   │
│  │ Response body warning:                                                │   │
│  │ {                                                                     │   │
│  │   "warnings": [                                                       │   │
│  │     {                                                                 │   │
│  │       "code": "deprecated_endpoint",                                  │   │
│  │       "message": "This endpoint is deprecated...",                    │   │
│  │       "sunset_date": "2026-07-01T00:00:00Z",                          │   │
│  │       "migration_guide": "https://docs.sigil.ai/migration/v1-to-v2"   │   │
│  │     }                                                                 │   │
│  │   ],                                                                  │   │
│  │   "data": { ... }                                                     │   │
│  │ }                                                                     │   │
│  │                                                                       │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  Breaking vs Non-Breaking Changes:                                           │
│  ─────────────────────────────────                                           │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │ NON-BREAKING (Allowed in same major version):                         │   │
│  │ ─────────────────────────────────────────────                         │   │
│  │ - Adding new optional fields to requests                              │   │
│  │ - Adding new fields to responses                                      │   │
│  │ - Adding new endpoints                                                │   │
│  │ - Adding new enum values (if client handles unknown)                  │   │
│  │ - Increasing rate limits                                              │   │
│  │ - Improving error messages                                            │   │
│  │ - Adding new optional query parameters                                │   │
│  │                                                                       │   │
│  │ BREAKING (Requires major version bump):                               │   │
│  │ ──────────────────────────────────────                                │   │
│  │ - Removing fields from responses                                      │   │
│  │ - Changing field types                                                │   │
│  │ - Renaming fields                                                     │   │
│  │ - Removing endpoints                                                  │   │
│  │ - Changing authentication method                                      │   │
│  │ - Changing error response format                                      │   │
│  │ - Making optional fields required                                     │   │
│  │ - Reducing rate limits                                                │   │
│  │ - Changing endpoint URLs                                              │   │
│  │                                                                       │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  Version Support Matrix:                                                     │
│  ───────────────────────                                                     │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │ Version │ Status     │ Release Date │ Deprecation  │ Sunset Date    │   │
│  │ ────────────────────────────────────────────────────────────────────│   │
│  │ v1      │ Current    │ 2026-01-11   │ TBD          │ TBD            │   │
│  │ v2      │ Planned    │ Q3 2026      │ v1 + 6mo     │ v1 + 12mo      │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Document Revision History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0.0 | 2026-01-11 | Systems Architecture Team | Initial Phase 7 architecture |
| 1.1.0 | 2026-01-11 | Systems Architecture Team | Added detailed implementation specs, testing, capacity planning, DR |

---

*Document Version: 1.1.0*
*Last Updated: 2026-01-11*
