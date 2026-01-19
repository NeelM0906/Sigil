# CLI Execution Flow for Sigil v2

## Overview

This document provides a comprehensive specification of the end-to-end execution flow for the Sigil v2 CLI `orchestrate` command. It details token accumulation at each stage, log output locations, and how the `monitor.py` script consumes logs for real-time visibility.

**Version:** 1.0.0
**Last Updated:** 2026-01-11
**Audience:** Developers, operators, and integrators

---

## Table of Contents

1. [Command Overview](#command-overview)
2. [Execution Architecture](#execution-architecture)
3. [Phase-by-Phase Flow](#phase-by-phase-flow)
4. [Token Accumulation](#token-accumulation)
5. [Log Output Locations](#log-output-locations)
6. [Monitor Script Integration](#monitor-script-integration)
7. [Error Handling Flow](#error-handling-flow)
8. [Complete Execution Example](#complete-execution-example)
9. [Configuration Options](#configuration-options)

---

## Command Overview

### The `orchestrate` Command

The `orchestrate` command is the primary entry point for executing agent tasks through the Sigil v2 framework. It coordinates all subsystems to process a user request.

```bash
# Basic usage
sigil orchestrate "Qualify this lead: John Smith from Acme Corp"

# With agent specification
sigil orchestrate --agent lead_qualifier "Qualify this lead: John Smith from Acme Corp"

# With budget limit
sigil orchestrate --budget 50000 "Research market trends for AI startups"

# With monitoring enabled
sigil orchestrate --monitor "Create a report on competitor pricing"

# Verbose mode with full logging
sigil orchestrate -v --monitor "Analyze customer feedback data"
```

### Command Signature

```python
@click.command()
@click.argument('task', type=str)
@click.option('--agent', '-a', default=None, help='Specific agent to use')
@click.option('--budget', '-b', default=256000, help='Token budget limit')
@click.option('--monitor', '-m', is_flag=True, help='Enable real-time monitoring')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose output')
@click.option('--output', '-o', default=None, help='Output file for results')
@click.option('--session', '-s', default=None, help='Session ID (auto-generated if not provided)')
def orchestrate(task, agent, budget, monitor, verbose, output, session):
    """Execute a task through the Sigil v2 orchestration pipeline."""
    pass
```

---

## Execution Architecture

### High-Level Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           CLI ENTRY POINT                                    │
│  sigil orchestrate "Qualify lead John from Acme Corp"                       │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         PHASE 1: INITIALIZATION                              │
│  - Create session ID                                                         │
│  - Initialize budget tracker (256K)                                          │
│  - Initialize log buffer                                                     │
│  - Start monitoring stream (if --monitor)                                    │
│  Tokens: 0 | Budget Remaining: 256,000                                       │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           PHASE 2: ROUTING                                   │
│  - Classify intent (create_agent | execute_task | simple_query)              │
│  - Assess complexity (0.0 - 1.0)                                             │
│  - Select handler and reasoning strategy                                     │
│  Tokens: ~400 | Budget Remaining: ~255,600                                   │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           PHASE 3: PLANNING                                  │
│  - Generate execution plan from goal                                         │
│  - Decompose into steps with dependencies                                    │
│  - Assign reasoning strategies per step                                      │
│  - Estimate token cost for plan                                              │
│  Tokens: ~4,100 | Budget Remaining: ~251,900                                 │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        PHASE 4: MEMORY RETRIEVAL                             │
│  - Search for relevant memories (hybrid: RAG + LLM)                          │
│  - Load context from memory categories                                       │
│  - Build initial context window                                              │
│  Tokens: ~5,600 | Budget Remaining: ~250,400                                 │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                       PHASE 5: PLAN EXECUTION                                │
│  For each step in plan:                                                      │
│    - Execute reasoning strategy                                              │
│    - Invoke tools (MCP) if required                                          │
│    - Validate intermediate outputs                                           │
│  Tokens: ~20,000 | Budget Remaining: ~236,000                                │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                     PHASE 6: CONTRACT VALIDATION                             │
│  - Validate final output against contract                                    │
│  - Retry with feedback if validation fails                                   │
│  - Generate retry feedback prompt                                            │
│  Tokens: ~22,000 | Budget Remaining: ~234,000                                │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      PHASE 7: MEMORY EXTRACTION                              │
│  - Extract facts from conversation                                           │
│  - Store new memories                                                        │
│  - Update memory categories                                                  │
│  Tokens: ~24,000 | Budget Remaining: ~232,000                                │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         PHASE 8: COMPLETION                                  │
│  - Format output for display                                                 │
│  - Write to output file (if specified)                                       │
│  - Flush remaining logs                                                      │
│  - Close monitoring stream                                                   │
│  Final Tokens: ~24,000 | Budget Remaining: ~232,000                          │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Component Interaction Sequence

```
CLI          Orchestrator    Router    Planner    Memory    Reasoning    Contracts    Tools
 │                │            │          │          │           │            │          │
 │  orchestrate   │            │          │          │           │            │          │
 ├───────────────>│            │          │          │           │            │          │
 │                │            │          │          │           │            │          │
 │                │  route()   │          │          │           │            │          │
 │                ├───────────>│          │          │           │            │          │
 │                │   intent   │          │          │           │            │          │
 │                │<───────────┤          │          │           │            │          │
 │                │            │          │          │           │            │          │
 │                │      create_plan()    │          │           │            │          │
 │                ├───────────────────────>│          │           │            │          │
 │                │         plan          │          │           │            │          │
 │                │<───────────────────────┤          │           │            │          │
 │                │            │          │          │           │            │          │
 │                │            │          │ retrieve()│           │            │          │
 │                ├────────────────────────────────────>│           │            │          │
 │                │            │          │ memories │           │            │          │
 │                │<────────────────────────────────────┤           │            │          │
 │                │            │          │          │           │            │          │
 │                │  For each step:       │          │           │            │          │
 │                │            │          │          │  reason()  │            │          │
 │                ├─────────────────────────────────────────────────>│            │          │
 │                │            │          │          │   result   │            │          │
 │                │<─────────────────────────────────────────────────┤            │          │
 │                │            │          │          │           │            │          │
 │                │            │          │          │           │ invoke()   │          │
 │                ├───────────────────────────────────────────────────────────────────────>│
 │                │            │          │          │           │    result  │          │
 │                │<───────────────────────────────────────────────────────────────────────┤
 │                │            │          │          │           │            │          │
 │                │            │          │          │           │ validate() │          │
 │                ├──────────────────────────────────────────────────────────────>│          │
 │                │            │          │          │           │   result   │          │
 │                │<──────────────────────────────────────────────────────────────┤          │
 │                │            │          │          │           │            │          │
 │   result       │            │          │          │           │            │          │
 │<───────────────┤            │          │          │           │            │          │
 │                │            │          │          │           │            │          │
```

---

## Phase-by-Phase Flow

### Phase 1: Initialization

**Purpose:** Set up the execution environment and initialize tracking systems.

**Code Location:** `sigil/interfaces/cli/commands/orchestrate.py`

```python
async def initialize_execution(
    task: str,
    budget: int,
    session_id: Optional[str],
    monitor: bool,
) -> ExecutionContext:
    """Initialize execution environment."""

    # Generate session ID if not provided
    if session_id is None:
        session_id = f"sess_{uuid.uuid4().hex[:12]}"

    # Initialize logging system
    from sigil.telemetry.logging import initialize_logging
    initialize_logging(
        buffer_size=100,
        flush_interval_ms=100,
        total_budget=budget,
    )

    # Create execution context
    context = ExecutionContext(
        session_id=session_id,
        correlation_id=session_id,  # Use session as initial correlation
        task=task,
        budget=budget,
        started_at=datetime.now(timezone.utc),
    )

    # Start monitoring stream if requested
    if monitor:
        from sigil.telemetry.monitor import start_monitor_stream
        context.monitor_stream = start_monitor_stream(session_id)

    # Log initialization
    logger.info(
        "Execution initialized",
        extra={
            "component": "cli",
            "operation": "initialize",
            "session_id": session_id,
            "metadata": {
                "task_length": len(task),
                "budget": budget,
                "monitor_enabled": monitor,
            }
        }
    )

    return context
```

**Logs Generated:**
```json
{
  "id": "log_init_001",
  "timestamp": "2026-01-11T14:30:00.000Z",
  "level": "info",
  "component": "cli",
  "operation": "initialize",
  "message": "Execution initialized",
  "session_id": "sess_abc123def456",
  "tokens_used": null,
  "budget_snapshot": {
    "total_budget": 256000,
    "used": 0,
    "remaining": 256000,
    "percentage_used": 0.0
  }
}
```

**Token Impact:** 0 tokens (no LLM calls)

---

### Phase 2: Routing

**Purpose:** Classify the request intent and determine the appropriate handler.

**Code Location:** `sigil/routing/router.py`

```python
async def route_request(
    context: ExecutionContext,
) -> RouteDecision:
    """Route the request to the appropriate handler."""

    async with operation_context(
        logger=router_logger,
        component=ComponentId.ROUTER,
        operation="route_request",
        session_id=context.session_id,
        correlation_id=context.correlation_id,
    ) as ctx:
        # Step 1: Classify intent
        intent_response = await llm.generate(
            messages=[{
                "role": "user",
                "content": f"""Classify this request into one of:
- create_agent: User wants to create a new agent
- execute_task: User wants to execute a task with an agent
- simple_query: User has a simple question
- modify_agent: User wants to modify an existing agent

Request: {context.task}

Return only the intent name."""
            }]
        )

        intent_tokens = TokenUsage.from_anthropic_response(
            intent_response.raw_response,
            model="anthropic:claude-opus-4-5-20251101",
        )
        ctx.add_tokens(intent_tokens)

        intent = intent_response.content.strip()

        # Step 2: Assess complexity
        complexity_response = await llm.generate(
            messages=[{
                "role": "user",
                "content": f"""Rate the complexity of this request from 0.0 to 1.0:

Request: {context.task}

Consider: steps required, ambiguity, real-world interaction needs.
Return only a number."""
            }]
        )

        complexity_tokens = TokenUsage.from_anthropic_response(
            complexity_response.raw_response,
            model="anthropic:claude-opus-4-5-20251101",
        )
        ctx.add_tokens(complexity_tokens)

        complexity = float(complexity_response.content.strip())

        # Step 3: Select handler and strategy
        handler = select_handler(intent, complexity)
        strategy = select_strategy(complexity)

        ctx.set_metadata({
            "intent": intent,
            "complexity": complexity,
            "handler": handler,
            "strategy": strategy,
        })

        return RouteDecision(
            intent=intent,
            complexity=complexity,
            handler=handler,
            reasoning_strategy=strategy,
        )
```

**Logs Generated:**
```json
{
  "id": "log_route_001",
  "timestamp": "2026-01-11T14:30:00.150Z",
  "level": "info",
  "component": "router",
  "operation": "route_request",
  "message": "Starting route_request",
  "session_id": "sess_abc123def456"
}
```

```json
{
  "id": "log_route_002",
  "timestamp": "2026-01-11T14:30:00.275Z",
  "level": "info",
  "component": "router",
  "operation": "route_request",
  "message": "Completed route_request",
  "session_id": "sess_abc123def456",
  "duration_ms": 125.5,
  "tokens_used": {
    "input_tokens": 320,
    "output_tokens": 80,
    "total_tokens": 400,
    "model": "anthropic:claude-opus-4-5-20251101"
  },
  "metadata": {
    "intent": "execute_task",
    "complexity": 0.65,
    "handler": "executor_rai",
    "strategy": "tree_of_thoughts"
  },
  "budget_snapshot": {
    "total_budget": 256000,
    "used": 400,
    "remaining": 255600,
    "percentage_used": 0.16
  }
}
```

**Token Impact:** ~400 tokens
- Intent classification: ~200 tokens
- Complexity assessment: ~200 tokens

---

### Phase 3: Planning

**Purpose:** Generate an execution plan that decomposes the task into steps.

**Code Location:** `sigil/planning/planner.py`

```python
async def create_execution_plan(
    context: ExecutionContext,
    route: RouteDecision,
) -> Plan:
    """Create an execution plan for the task."""

    async with operation_context(
        logger=planner_logger,
        component=ComponentId.PLANNER,
        operation="create_plan",
        session_id=context.session_id,
        correlation_id=context.correlation_id,
    ) as ctx:
        # Get available tools
        available_tools = await get_available_tools()

        # Generate plan
        plan_response = await llm.generate(
            messages=[{
                "role": "user",
                "content": f"""Create a step-by-step execution plan for this goal:

Goal: {context.task}

Available tools: {[t.name for t in available_tools]}

For each step, provide:
1. name: Short identifier (snake_case)
2. description: What this step does
3. inputs: What it needs
4. outputs: What it produces
5. complexity: simple | moderate | complex | critical
6. dependencies: Which previous steps must complete first
7. tools_required: Which tools are needed
8. needs_contract: true if output needs verification

Return as JSON array."""
            }],
            output_schema=PlanSchema,
        )

        plan_tokens = TokenUsage.from_anthropic_response(
            plan_response.raw_response,
            model="anthropic:claude-opus-4-5-20251101",
        )
        ctx.set_tokens(plan_tokens)

        # Parse and validate plan
        plan = parse_plan(plan_response.content)
        plan.id = f"plan_{uuid.uuid4().hex[:12]}"

        # Assign reasoning strategies to steps
        for step in plan.steps:
            step.reasoning_strategy = COMPLEXITY_TO_STRATEGY[step.complexity]

        # Estimate total tokens
        estimated_tokens = estimate_plan_tokens(plan)

        ctx.set_metadata({
            "plan_id": plan.id,
            "step_count": len(plan.steps),
            "estimated_tokens": estimated_tokens,
            "steps": [s.name for s in plan.steps],
        })

        return plan
```

**Logs Generated:**
```json
{
  "id": "log_plan_001",
  "timestamp": "2026-01-11T14:30:00.400Z",
  "level": "info",
  "component": "planner",
  "operation": "create_plan",
  "message": "Completed create_plan",
  "session_id": "sess_abc123def456",
  "duration_ms": 850.0,
  "tokens_used": {
    "input_tokens": 2500,
    "output_tokens": 1200,
    "total_tokens": 3700,
    "model": "anthropic:claude-opus-4-5-20251101"
  },
  "metadata": {
    "plan_id": "plan_xyz789abc",
    "step_count": 5,
    "estimated_tokens": 18500,
    "steps": [
      "retrieve_context",
      "research_lead",
      "assess_bant",
      "generate_recommendation",
      "format_output"
    ]
  },
  "budget_snapshot": {
    "total_budget": 256000,
    "used": 4100,
    "remaining": 251900,
    "percentage_used": 1.60
  }
}
```

**Token Impact:** ~3,700 tokens
- Plan generation prompt: ~2,500 input tokens
- Plan response: ~1,200 output tokens

---

### Phase 4: Memory Retrieval

**Purpose:** Retrieve relevant memories to provide context for execution.

**Code Location:** `sigil/memory/retrieval.py`

```python
async def retrieve_execution_context(
    context: ExecutionContext,
    plan: Plan,
) -> MemoryContext:
    """Retrieve memories relevant to the task."""

    async with operation_context(
        logger=memory_logger,
        component=ComponentId.MEMORY,
        operation="retrieve_context",
        session_id=context.session_id,
        correlation_id=context.correlation_id,
    ) as ctx:
        # Step 1: RAG search for relevant items
        query = f"{context.task}"
        rag_results = await vector_store.search(
            query=query,
            k=20,
        )

        # Embedding doesn't consume LLM tokens directly
        # but we track the operation

        # Step 2: LLM-based relevance filtering
        if len(rag_results) > 10:
            filter_response = await llm.generate(
                messages=[{
                    "role": "user",
                    "content": f"""Select the 10 most relevant memories for this task:

Task: {context.task}

Memories:
{format_memories(rag_results)}

Return the indices of the 10 most relevant memories as a JSON array."""
                }]
            )

            filter_tokens = TokenUsage.from_anthropic_response(
                filter_response.raw_response,
                model="anthropic:claude-opus-4-5-20251101",
            )
            ctx.add_tokens(filter_tokens)

            selected_indices = json.loads(filter_response.content)
            memories = [rag_results[i] for i in selected_indices]
        else:
            memories = rag_results

        # Step 3: Load memory categories
        categories = await load_relevant_categories(context.task)

        ctx.set_metadata({
            "retrieval_method": "hybrid",
            "rag_results": len(rag_results),
            "filtered_results": len(memories),
            "categories_loaded": len(categories),
        })

        return MemoryContext(
            items=memories,
            categories=categories,
        )
```

**Logs Generated:**
```json
{
  "id": "log_mem_001",
  "timestamp": "2026-01-11T14:30:01.500Z",
  "level": "info",
  "component": "memory",
  "operation": "retrieve_context",
  "message": "Completed retrieve_context",
  "session_id": "sess_abc123def456",
  "duration_ms": 650.0,
  "tokens_used": {
    "input_tokens": 1200,
    "output_tokens": 300,
    "total_tokens": 1500,
    "model": "anthropic:claude-opus-4-5-20251101"
  },
  "metadata": {
    "retrieval_method": "hybrid",
    "rag_results": 20,
    "filtered_results": 10,
    "categories_loaded": 3
  },
  "budget_snapshot": {
    "total_budget": 256000,
    "used": 5600,
    "remaining": 250400,
    "percentage_used": 2.19
  }
}
```

**Token Impact:** ~1,500 tokens
- Relevance filtering: ~1,500 tokens

---

### Phase 5: Plan Execution

**Purpose:** Execute each step in the plan sequentially.

**Code Location:** `sigil/planning/executor.py`

```python
async def execute_plan(
    context: ExecutionContext,
    plan: Plan,
    memory_context: MemoryContext,
) -> ExecutionResult:
    """Execute all steps in the plan."""

    async with operation_context(
        logger=executor_logger,
        component=ComponentId.ORCHESTRATOR,
        operation="execute_plan",
        session_id=context.session_id,
        correlation_id=context.correlation_id,
    ) as plan_ctx:
        results = {}
        total_tokens = TokenUsage.zero()

        for step in plan.get_execution_order():
            step_result = await execute_step(
                step=step,
                context=context,
                memory_context=memory_context,
                previous_results=results,
                parent_operation_id=plan_ctx.operation_id,
            )

            results[step.name] = step_result
            total_tokens = total_tokens + step_result.tokens_used

            # Log step completion
            logger.info(
                f"Step completed: {step.name}",
                extra={
                    "component": "orchestrator",
                    "operation": "execute_step",
                    "session_id": context.session_id,
                    "tokens_used": step_result.tokens_used,
                    "metadata": {
                        "step_name": step.name,
                        "step_strategy": step.reasoning_strategy,
                        "success": step_result.success,
                    }
                }
            )

        plan_ctx.set_tokens(total_tokens)
        plan_ctx.set_metadata({
            "plan_id": plan.id,
            "steps_executed": len(results),
            "all_succeeded": all(r.success for r in results.values()),
        })

        return ExecutionResult(
            plan_id=plan.id,
            step_results=results,
            tokens_used=total_tokens,
        )


async def execute_step(
    step: PlanStep,
    context: ExecutionContext,
    memory_context: MemoryContext,
    previous_results: dict,
    parent_operation_id: str,
) -> StepResult:
    """Execute a single plan step."""

    async with operation_context(
        logger=reasoning_logger,
        component=ComponentId.REASONING,
        operation=step.reasoning_strategy,
        session_id=context.session_id,
        correlation_id=context.correlation_id,
        parent_operation_id=parent_operation_id,
    ) as step_ctx:
        # Build step context
        step_context = build_step_context(
            step, context, memory_context, previous_results
        )

        # Get reasoning strategy
        strategy = get_strategy(step.reasoning_strategy)

        # Execute reasoning
        reasoning_result = await strategy.execute(
            task=step.description,
            context=step_context,
        )
        step_ctx.add_tokens(reasoning_result.tokens_used)

        # Execute tools if required
        if step.tools_required:
            for tool_name in step.tools_required:
                tool_result = await execute_tool(
                    tool_name=tool_name,
                    params=reasoning_result.tool_params.get(tool_name, {}),
                    session_id=context.session_id,
                    correlation_id=context.correlation_id,
                    parent_operation_id=step_ctx.operation_id,
                )
                # Tools don't typically consume LLM tokens

        step_ctx.set_metadata({
            "step_name": step.name,
            "strategy": step.reasoning_strategy,
            "confidence": reasoning_result.confidence,
            "tools_used": step.tools_required,
        })

        return StepResult(
            step_name=step.name,
            output=reasoning_result.output,
            tokens_used=step_ctx.get_total_tokens(),
            success=True,
        )
```

**Logs Generated (per step):**

Step 1: retrieve_context
```json
{
  "id": "log_step_001",
  "timestamp": "2026-01-11T14:30:02.000Z",
  "level": "info",
  "component": "reasoning",
  "operation": "direct",
  "message": "Completed direct",
  "session_id": "sess_abc123def456",
  "duration_ms": 180.0,
  "tokens_used": {
    "input_tokens": 500,
    "output_tokens": 200,
    "total_tokens": 700
  },
  "metadata": {
    "step_name": "retrieve_context",
    "strategy": "direct",
    "confidence": 0.95
  }
}
```

Step 2: research_lead (with tool)
```json
{
  "id": "log_step_002",
  "timestamp": "2026-01-11T14:30:04.500Z",
  "level": "info",
  "component": "reasoning",
  "operation": "chain_of_thought",
  "message": "Completed chain_of_thought",
  "session_id": "sess_abc123def456",
  "duration_ms": 2500.0,
  "tokens_used": {
    "input_tokens": 2000,
    "output_tokens": 1200,
    "total_tokens": 3200
  },
  "metadata": {
    "step_name": "research_lead",
    "strategy": "chain_of_thought",
    "confidence": 0.88,
    "tools_used": ["websearch"]
  }
}
```

```json
{
  "id": "log_tool_001",
  "timestamp": "2026-01-11T14:30:03.500Z",
  "level": "info",
  "component": "tools",
  "operation": "websearch",
  "message": "Completed websearch",
  "session_id": "sess_abc123def456",
  "duration_ms": 1200.0,
  "tokens_used": null,
  "metadata": {
    "tool_name": "websearch",
    "tool_server": "tavily-mcp",
    "results_count": 5
  }
}
```

Step 3: assess_bant (complex reasoning)
```json
{
  "id": "log_step_003",
  "timestamp": "2026-01-11T14:30:08.000Z",
  "level": "info",
  "component": "reasoning",
  "operation": "tree_of_thoughts",
  "message": "Completed tree_of_thoughts",
  "session_id": "sess_abc123def456",
  "duration_ms": 3500.0,
  "tokens_used": {
    "input_tokens": 6000,
    "output_tokens": 2500,
    "total_tokens": 8500
  },
  "metadata": {
    "step_name": "assess_bant",
    "strategy": "tree_of_thoughts",
    "branches_explored": 4,
    "confidence": 0.86
  }
}
```

**Token Impact (all steps):** ~14,400 tokens
- retrieve_context: ~700 tokens
- research_lead: ~3,200 tokens
- assess_bant: ~8,500 tokens
- generate_recommendation: ~1,200 tokens
- format_output: ~800 tokens

---

### Phase 6: Contract Validation

**Purpose:** Validate the final output against the contract specification.

**Code Location:** `sigil/contracts/executor.py`

```python
async def validate_output(
    context: ExecutionContext,
    execution_result: ExecutionResult,
    contract: Contract,
) -> ValidationResult:
    """Validate execution output against contract."""

    async with operation_context(
        logger=contracts_logger,
        component=ComponentId.CONTRACTS,
        operation="validate_output",
        session_id=context.session_id,
        correlation_id=context.correlation_id,
    ) as ctx:
        # Extract final output
        final_output = execution_result.step_results["format_output"].output

        # Validate deliverables
        validation = validate_deliverables(final_output, contract)

        if not validation.success:
            # Log validation failure
            logger.warning(
                "Contract validation failed",
                extra={
                    "component": "contracts",
                    "operation": "validate_output",
                    "session_id": context.session_id,
                    "error_info": ErrorInfo(
                        error_type="ContractValidationError",
                        error_message=f"Missing: {validation.missing_deliverables}",
                        error_code="CONTRACT_002",
                        recoverable=True,
                    ),
                    "metadata": {
                        "contract_name": contract.name,
                        "passed": validation.passed_deliverables,
                        "failed": validation.missing_deliverables,
                    }
                }
            )

            # Generate retry feedback
            feedback_response = await llm.generate(
                messages=[{
                    "role": "user",
                    "content": f"""The output is missing these deliverables: {validation.missing_deliverables}

Contract requirements:
{format_contract_requirements(contract)}

Current output:
{json.dumps(final_output, indent=2)}

Provide specific instructions to fix the missing deliverables."""
                }]
            )

            feedback_tokens = TokenUsage.from_anthropic_response(
                feedback_response.raw_response,
                model="anthropic:claude-opus-4-5-20251101",
            )
            ctx.add_tokens(feedback_tokens)

            validation.feedback = feedback_response.content

        ctx.set_metadata({
            "contract_name": contract.name,
            "validation_passed": validation.success,
            "deliverables_checked": len(contract.deliverables),
            "deliverables_passed": len(validation.passed_deliverables),
        })

        return validation
```

**Logs Generated:**
```json
{
  "id": "log_contract_001",
  "timestamp": "2026-01-11T14:30:12.000Z",
  "level": "info",
  "component": "contracts",
  "operation": "validate_output",
  "message": "Completed validate_output",
  "session_id": "sess_abc123def456",
  "duration_ms": 150.0,
  "tokens_used": {
    "input_tokens": 800,
    "output_tokens": 200,
    "total_tokens": 1000
  },
  "metadata": {
    "contract_name": "lead_qualification",
    "validation_passed": true,
    "deliverables_checked": 4,
    "deliverables_passed": 4
  },
  "budget_snapshot": {
    "total_budget": 256000,
    "used": 21500,
    "remaining": 234500,
    "percentage_used": 8.40
  }
}
```

**Token Impact:** ~1,000-2,500 tokens
- Validation (if passes): ~1,000 tokens
- Validation with retry feedback: ~2,500 tokens

---

### Phase 7: Memory Extraction

**Purpose:** Extract facts from the execution and store new memories.

**Code Location:** `sigil/memory/extraction.py`

```python
async def extract_and_store_memories(
    context: ExecutionContext,
    execution_result: ExecutionResult,
) -> None:
    """Extract memories from execution and store them."""

    async with operation_context(
        logger=memory_logger,
        component=ComponentId.MEMORY,
        operation="extract_memories",
        session_id=context.session_id,
        correlation_id=context.correlation_id,
    ) as ctx:
        # Prepare conversation for extraction
        conversation = format_execution_as_conversation(execution_result)

        # Extract facts using LLM
        extraction_response = await llm.generate(
            messages=[{
                "role": "user",
                "content": f"""Extract key facts from this execution:

{conversation}

For each fact, provide:
- content: The fact text
- type: fact | preference | event | skill | pattern
- confidence: 0.0-1.0

Return as JSON array."""
            }]
        )

        extraction_tokens = TokenUsage.from_anthropic_response(
            extraction_response.raw_response,
            model="anthropic:claude-opus-4-5-20251101",
        )
        ctx.set_tokens(extraction_tokens)

        # Parse and store
        facts = json.loads(extraction_response.content)

        for fact in facts:
            await memory_store.add_item(
                content=fact["content"],
                memory_type=fact["type"],
                source_session_id=context.session_id,
                confidence=fact["confidence"],
            )

        ctx.set_metadata({
            "facts_extracted": len(facts),
            "fact_types": [f["type"] for f in facts],
        })
```

**Logs Generated:**
```json
{
  "id": "log_memory_extract_001",
  "timestamp": "2026-01-11T14:30:13.500Z",
  "level": "info",
  "component": "memory",
  "operation": "extract_memories",
  "message": "Completed extract_memories",
  "session_id": "sess_abc123def456",
  "duration_ms": 800.0,
  "tokens_used": {
    "input_tokens": 1800,
    "output_tokens": 700,
    "total_tokens": 2500
  },
  "metadata": {
    "facts_extracted": 5,
    "fact_types": ["fact", "fact", "preference", "pattern", "event"]
  },
  "budget_snapshot": {
    "total_budget": 256000,
    "used": 24000,
    "remaining": 232000,
    "percentage_used": 9.38
  }
}
```

**Token Impact:** ~2,500 tokens

---

### Phase 8: Completion

**Purpose:** Finalize execution, format output, and clean up.

**Code Location:** `sigil/interfaces/cli/commands/orchestrate.py`

```python
async def complete_execution(
    context: ExecutionContext,
    result: ExecutionResult,
    output_file: Optional[str],
) -> None:
    """Complete execution and clean up."""

    # Format output for display
    formatted_output = format_result_for_display(result)

    # Display to console
    console.print(formatted_output)

    # Write to file if specified
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(result.to_dict(), f, indent=2)

    # Log completion
    logger.info(
        "Execution completed",
        extra={
            "component": "cli",
            "operation": "complete_execution",
            "session_id": context.session_id,
            "metadata": {
                "success": result.success,
                "total_tokens": result.total_tokens,
                "duration_seconds": (datetime.now(timezone.utc) - context.started_at).total_seconds(),
                "output_file": output_file,
            }
        }
    )

    # Flush remaining logs
    await flush_log_buffer()

    # Close monitoring stream
    if context.monitor_stream:
        await context.monitor_stream.close()
```

**Logs Generated:**
```json
{
  "id": "log_complete_001",
  "timestamp": "2026-01-11T14:30:14.000Z",
  "level": "info",
  "component": "cli",
  "operation": "complete_execution",
  "message": "Execution completed",
  "session_id": "sess_abc123def456",
  "duration_ms": 14000.0,
  "metadata": {
    "success": true,
    "total_tokens": 24000,
    "duration_seconds": 14.0,
    "output_file": null
  },
  "budget_snapshot": {
    "total_budget": 256000,
    "used": 24000,
    "remaining": 232000,
    "percentage_used": 9.38
  }
}
```

**Token Impact:** 0 tokens (no LLM calls)

---

## Token Accumulation

### Token Budget Flow

```
Phase              Tokens Used    Cumulative    Budget Remaining    % Used
─────────────────────────────────────────────────────────────────────────────
1. Initialization        0              0           256,000          0.00%
2. Routing             400            400           255,600          0.16%
3. Planning          3,700          4,100           251,900          1.60%
4. Memory            1,500          5,600           250,400          2.19%
5. Execution        14,400         20,000           236,000          7.81%
6. Contracts         1,500         21,500           234,500          8.40%
7. Memory Extract    2,500         24,000           232,000          9.38%
8. Completion            0         24,000           232,000          9.38%
─────────────────────────────────────────────────────────────────────────────
TOTAL               24,000         24,000           232,000          9.38%
```

### Token Breakdown by Component

```
Component        Tokens    Percentage
────────────────────────────────────
reasoning       14,400       60.00%
planner          3,700       15.42%
memory           4,000       16.67%
contracts        1,500        6.25%
router             400        1.67%
────────────────────────────────────
TOTAL           24,000      100.00%
```

### Token Tracking Data Structure

```python
@dataclass
class ExecutionTokenSummary:
    """Summary of token usage for an execution."""

    session_id: str
    total_tokens: int
    input_tokens: int
    output_tokens: int
    budget_total: int
    budget_used: int
    budget_remaining: int
    percentage_used: float

    by_component: dict[str, int]
    by_phase: dict[str, int]
    by_operation: list[OperationTokenRecord]

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "total_tokens": self.total_tokens,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "budget": {
                "total": self.budget_total,
                "used": self.budget_used,
                "remaining": self.budget_remaining,
                "percentage_used": round(self.percentage_used, 2),
            },
            "by_component": self.by_component,
            "by_phase": self.by_phase,
            "operations": [op.to_dict() for op in self.by_operation],
        }
```

---

## Log Output Locations

### File-Based Logs

Logs are written to JSON files organized by session:

```
outputs/
├── sessions/
│   ├── sess_abc123def456.json       # Session event log
│   ├── sess_abc123def456.jsonl      # Streaming log (JSONL format)
│   └── sess_def789abc012.json
└── logs/
    ├── 2026-01-11.jsonl             # Daily aggregate log
    └── 2026-01-12.jsonl
```

### Session Log Format

```json
{
  "session_id": "sess_abc123def456",
  "created_at": "2026-01-11T14:30:00.000Z",
  "updated_at": "2026-01-11T14:30:14.000Z",
  "events": [
    {
      "id": "log_init_001",
      "timestamp": "2026-01-11T14:30:00.000Z",
      "level": "info",
      "component": "cli",
      "operation": "initialize",
      "message": "Execution initialized",
      "session_id": "sess_abc123def456",
      "tokens_used": null,
      "budget_snapshot": {
        "total_budget": 256000,
        "used": 0,
        "remaining": 256000
      }
    },
    // ... more events
  ],
  "summary": {
    "total_events": 25,
    "total_tokens": 24000,
    "duration_seconds": 14.0,
    "success": true
  }
}
```

### Streaming Log Format (JSONL)

Each line is a complete JSON object:

```jsonl
{"id":"log_init_001","timestamp":"2026-01-11T14:30:00.000Z","level":"info","component":"cli","operation":"initialize","message":"Execution initialized"}
{"id":"log_route_001","timestamp":"2026-01-11T14:30:00.150Z","level":"info","component":"router","operation":"route_request","message":"Starting route_request"}
{"id":"log_route_002","timestamp":"2026-01-11T14:30:00.275Z","level":"info","component":"router","operation":"route_request","message":"Completed route_request","tokens_used":{"total_tokens":400}}
```

### Log Rotation

```python
# Log rotation configuration
LOG_ROTATION_CONFIG = {
    "max_file_size_mb": 100,
    "max_files": 30,
    "compress_old": True,
    "retention_days": 30,
}
```

---

## Monitor Script Integration

### The `monitor.py` Script

The monitoring script provides real-time visibility into CLI execution.

```python
#!/usr/bin/env python3
"""Real-time monitoring for Sigil v2 CLI execution.

Usage:
    python monitor.py [--session SESSION_ID] [--filter COMPONENT] [--level LEVEL]

Examples:
    # Monitor all sessions
    python monitor.py

    # Monitor specific session
    python monitor.py --session sess_abc123def456

    # Filter to reasoning component
    python monitor.py --filter reasoning

    # Show only warnings and errors
    python monitor.py --level warning
"""

import asyncio
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import websockets
from rich.console import Console
from rich.live import Live
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn


console = Console()


class MonitorClient:
    """WebSocket client for real-time log monitoring."""

    def __init__(
        self,
        session_id: Optional[str] = None,
        component_filter: Optional[str] = None,
        level_filter: str = "info",
    ):
        self.session_id = session_id
        self.component_filter = component_filter
        self.level_filter = level_filter
        self.ws_url = "ws://localhost:8000/monitoring/v1/logs/stream"

        # State
        self.current_status = {
            "session_id": None,
            "current_operation": None,
            "current_component": None,
            "tokens_used": 0,
            "budget_remaining": 256000,
            "is_alive": False,
        }
        self.recent_logs = []
        self.errors = []

    async def connect(self):
        """Connect to the monitoring WebSocket."""
        params = []
        if self.session_id:
            params.append(f"session_id={self.session_id}")
        if self.component_filter:
            params.append(f"components={self.component_filter}")

        url = self.ws_url
        if params:
            url += "?" + "&".join(params)

        async with websockets.connect(url) as ws:
            # Send configuration
            config = {
                "type": "config",
                "filter": {
                    "levels": self._get_level_filter(),
                },
                "buffer_size": 10,
                "flush_interval_ms": 100,
                "heartbeat_interval_ms": 1000,
            }
            await ws.send(json.dumps(config))

            # Start receiving
            await self._receive_loop(ws)

    async def _receive_loop(self, ws):
        """Main receive loop for WebSocket messages."""
        with Live(self._render_dashboard(), refresh_per_second=4) as live:
            async for message in ws:
                data = json.loads(message)
                self._process_message(data)
                live.update(self._render_dashboard())

    def _process_message(self, data: dict):
        """Process incoming WebSocket message."""
        msg_type = data.get("type")

        if msg_type == "heartbeat":
            self.current_status.update(data.get("heartbeat", {}))

        elif msg_type == "log":
            for entry in data.get("entries", []):
                self._process_log_entry(entry)

        elif msg_type == "buffer_flush":
            for entry in data.get("entries", []):
                self._process_log_entry(entry)

        elif msg_type == "error":
            self.errors.append(data.get("error"))

    def _process_log_entry(self, entry: dict):
        """Process a single log entry."""
        # Add to recent logs (keep last 20)
        self.recent_logs.append(entry)
        if len(self.recent_logs) > 20:
            self.recent_logs.pop(0)

        # Track errors
        if entry.get("level") in ("error", "critical"):
            self.errors.append({
                "timestamp": entry.get("timestamp"),
                "component": entry.get("component"),
                "message": entry.get("message"),
            })

        # Update status from tokens_used
        if entry.get("budget_snapshot"):
            snapshot = entry["budget_snapshot"]
            self.current_status["tokens_used"] = snapshot.get("used", 0)
            self.current_status["budget_remaining"] = snapshot.get("remaining", 256000)

        if entry.get("operation"):
            self.current_status["current_operation"] = entry["operation"]
            self.current_status["current_component"] = entry["component"]

    def _render_dashboard(self) -> Panel:
        """Render the monitoring dashboard."""
        # Status table
        status_table = Table(title="Execution Status", expand=True)
        status_table.add_column("Metric", style="cyan")
        status_table.add_column("Value", style="green")

        status_table.add_row(
            "Session",
            self.current_status.get("session_id") or "Waiting..."
        )
        status_table.add_row(
            "Status",
            "[green]Active[/green]" if self.current_status.get("is_alive") else "[yellow]Idle[/yellow]"
        )
        status_table.add_row(
            "Current Operation",
            self.current_status.get("current_operation") or "-"
        )
        status_table.add_row(
            "Current Component",
            self.current_status.get("current_component") or "-"
        )

        # Budget display
        tokens_used = self.current_status.get("tokens_used", 0)
        budget_remaining = self.current_status.get("budget_remaining", 256000)
        percentage = (tokens_used / 256000) * 100

        budget_color = "green"
        if percentage > 80:
            budget_color = "red"
        elif percentage > 50:
            budget_color = "yellow"

        status_table.add_row(
            "Tokens Used",
            f"[{budget_color}]{tokens_used:,}[/{budget_color}] / 256,000 ({percentage:.1f}%)"
        )
        status_table.add_row(
            "Budget Remaining",
            f"{budget_remaining:,}"
        )

        # Recent logs table
        logs_table = Table(title="Recent Logs", expand=True)
        logs_table.add_column("Time", width=12)
        logs_table.add_column("Level", width=8)
        logs_table.add_column("Component", width=12)
        logs_table.add_column("Operation", width=20)
        logs_table.add_column("Tokens", width=8)
        logs_table.add_column("Message", overflow="fold")

        for log in self.recent_logs[-10:]:
            level = log.get("level", "info")
            level_style = {
                "debug": "dim",
                "info": "blue",
                "warning": "yellow",
                "error": "red",
                "critical": "red bold",
            }.get(level, "white")

            tokens = ""
            if log.get("tokens_used"):
                tokens = str(log["tokens_used"].get("total_tokens", ""))

            logs_table.add_row(
                log.get("timestamp", "")[-12:-5],  # HH:MM:SS
                f"[{level_style}]{level.upper()}[/{level_style}]",
                log.get("component", ""),
                log.get("operation", ""),
                tokens,
                log.get("message", "")[:50],
            )

        # Combine into dashboard
        from rich.layout import Layout
        layout = Layout()
        layout.split_column(
            Layout(status_table, name="status", size=10),
            Layout(logs_table, name="logs"),
        )

        return Panel(layout, title="[bold]Sigil v2 Monitor[/bold]", border_style="blue")

    def _get_level_filter(self) -> list[str]:
        """Get log levels based on filter setting."""
        all_levels = ["trace", "debug", "info", "warning", "error", "critical"]
        try:
            idx = all_levels.index(self.level_filter)
            return all_levels[idx:]
        except ValueError:
            return all_levels[2:]  # Default to info and above


async def main():
    parser = argparse.ArgumentParser(description="Sigil v2 Real-time Monitor")
    parser.add_argument("--session", "-s", help="Session ID to monitor")
    parser.add_argument("--filter", "-f", help="Filter by component")
    parser.add_argument("--level", "-l", default="info", help="Minimum log level")
    args = parser.parse_args()

    client = MonitorClient(
        session_id=args.session,
        component_filter=args.filter,
        level_filter=args.level,
    )

    console.print("[bold blue]Sigil v2 Monitor[/bold blue]")
    console.print("Connecting to monitoring stream...")

    try:
        await client.connect()
    except KeyboardInterrupt:
        console.print("\n[yellow]Monitor stopped[/yellow]")
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
```

### Monitor Display Output

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                             Sigil v2 Monitor                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│ Execution Status                                                             │
│ ┌─────────────────────────┬────────────────────────────────────────────────┐│
│ │ Metric                  │ Value                                          ││
│ ├─────────────────────────┼────────────────────────────────────────────────┤│
│ │ Session                 │ sess_abc123def456                              ││
│ │ Status                  │ Active                                         ││
│ │ Current Operation       │ tree_of_thoughts                               ││
│ │ Current Component       │ reasoning                                      ││
│ │ Tokens Used             │ 15,400 / 256,000 (6.0%)                        ││
│ │ Budget Remaining        │ 240,600                                        ││
│ └─────────────────────────┴────────────────────────────────────────────────┘│
│                                                                              │
│ Recent Logs                                                                  │
│ ┌──────────┬────────┬────────────┬──────────────────────┬────────┬─────────┐│
│ │ Time     │ Level  │ Component  │ Operation            │ Tokens │ Message ││
│ ├──────────┼────────┼────────────┼──────────────────────┼────────┼─────────┤│
│ │ 14:30:00 │ INFO   │ cli        │ initialize           │        │ Execu...││
│ │ 14:30:00 │ INFO   │ router     │ route_request        │ 400    │ Compl...││
│ │ 14:30:01 │ INFO   │ planner    │ create_plan          │ 3700   │ Compl...││
│ │ 14:30:02 │ INFO   │ memory     │ retrieve_context     │ 1500   │ Compl...││
│ │ 14:30:02 │ DEBUG  │ reasoning  │ tree_of_thoughts     │        │ Start...││
│ │ 14:30:05 │ DEBUG  │ reasoning  │ tree_of_thoughts     │        │ Branc...││
│ │ 14:30:07 │ DEBUG  │ reasoning  │ tree_of_thoughts     │        │ Branc...││
│ │ 14:30:08 │ INFO   │ reasoning  │ tree_of_thoughts     │ 8500   │ Compl...││
│ │ 14:30:09 │ INFO   │ tools      │ websearch            │        │ Compl...││
│ │ 14:30:11 │ INFO   │ contracts  │ validate_output      │ 1000   │ Compl...││
│ └──────────┴────────┴────────────┴──────────────────────┴────────┴─────────┘│
└─────────────────────────────────────────────────────────────────────────────┘
```

### File-Based Monitoring (Alternative)

For environments without WebSocket support, monitor logs via file tailing:

```bash
#!/bin/bash
# file_monitor.sh - Monitor logs via file tailing

SESSION_DIR="${1:-outputs/sessions}"
LOG_FILE="${2:-}"

if [ -n "$LOG_FILE" ]; then
    # Watch specific session
    tail -f "$SESSION_DIR/$LOG_FILE.jsonl" | while read line; do
        echo "$line" | jq -r '
            "\(.timestamp | split("T")[1] | split(".")[0]) [\(.level | ascii_upcase)] \(.component)/\(.operation): \(.message)"
        '
    done
else
    # Watch all sessions
    tail -f "$SESSION_DIR"/*.jsonl | while read line; do
        # Extract filename and log
        if [[ "$line" == *"==>"* ]]; then
            echo "--- $line ---"
        else
            echo "$line" | jq -r '
                "\(.timestamp | split("T")[1] | split(".")[0]) [\(.level | ascii_upcase)] \(.component)/\(.operation): \(.message)"
            ' 2>/dev/null
        fi
    done
fi
```

---

## Error Handling Flow

### Error Propagation Path

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Error Occurs                                       │
│  e.g., LLM API timeout, validation failure, tool error                      │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                     Component Error Handler                                  │
│  - Create ErrorInfo from exception                                           │
│  - Log error with full context                                               │
│  - Determine if recoverable                                                  │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                         ┌────────────┴────────────┐
                         ▼                          ▼
              ┌─────────────────────┐    ┌─────────────────────┐
              │    Recoverable      │    │   Non-Recoverable   │
              │  - Log WARNING      │    │  - Log ERROR        │
              │  - Attempt retry    │    │  - Propagate up     │
              │  - Track attempts   │    │  - Abort execution  │
              └─────────────────────┘    └─────────────────────┘
                         │                          │
                         ▼                          ▼
              ┌─────────────────────┐    ┌─────────────────────┐
              │  Retry Successful   │    │   Orchestrator      │
              │  - Continue         │    │  - Log CRITICAL     │
              │  - Log recovery     │    │  - Save partial     │
              └─────────────────────┘    │  - Return error     │
                                         └─────────────────────┘
```

### Error Log Examples

**Recoverable Error (Retry)**
```json
{
  "id": "log_error_001",
  "timestamp": "2026-01-11T14:30:08.000Z",
  "level": "warning",
  "component": "contracts",
  "operation": "validate_output",
  "message": "Contract validation failed, will retry",
  "session_id": "sess_abc123def456",
  "error_info": {
    "error_type": "ContractValidationError",
    "error_message": "Missing required deliverable: bant_assessment",
    "error_code": "CONTRACT_002",
    "recoverable": true,
    "retry_after_ms": 0
  },
  "metadata": {
    "attempt": 1,
    "max_attempts": 3,
    "will_retry": true
  }
}
```

**Non-Recoverable Error (Abort)**
```json
{
  "id": "log_error_002",
  "timestamp": "2026-01-11T14:30:15.000Z",
  "level": "critical",
  "component": "orchestrator",
  "operation": "execute_plan",
  "message": "Token budget exhausted, aborting execution",
  "session_id": "sess_abc123def456",
  "error_info": {
    "error_type": "BudgetExhaustedError",
    "error_message": "Used 256,000/256,000 tokens. Cannot continue.",
    "error_code": "TOKEN_001",
    "recoverable": false,
    "context": {
      "last_component": "reasoning",
      "last_operation": "tree_of_thoughts",
      "partial_result_available": true
    }
  },
  "budget_snapshot": {
    "total_budget": 256000,
    "used": 256000,
    "remaining": 0,
    "percentage_used": 100.0
  }
}
```

---

## Complete Execution Example

### CLI Command

```bash
sigil orchestrate --monitor "Qualify this lead: John Smith, CEO at Acme Corp, interested in enterprise plan"
```

### Complete Log Sequence

```json
// 1. Initialization
{"id":"log_001","timestamp":"2026-01-11T14:30:00.000Z","level":"info","component":"cli","operation":"initialize","message":"Execution initialized","session_id":"sess_abc123def456","budget_snapshot":{"total_budget":256000,"used":0}}

// 2. Routing
{"id":"log_002","timestamp":"2026-01-11T14:30:00.150Z","level":"info","component":"router","operation":"route_request","message":"Starting route_request","session_id":"sess_abc123def456"}
{"id":"log_003","timestamp":"2026-01-11T14:30:00.275Z","level":"info","component":"router","operation":"route_request","message":"Completed route_request","session_id":"sess_abc123def456","tokens_used":{"input_tokens":320,"output_tokens":80,"total_tokens":400},"metadata":{"intent":"execute_task","complexity":0.65,"handler":"executor_rai"}}

// 3. Planning
{"id":"log_004","timestamp":"2026-01-11T14:30:00.300Z","level":"info","component":"planner","operation":"create_plan","message":"Starting create_plan","session_id":"sess_abc123def456"}
{"id":"log_005","timestamp":"2026-01-11T14:30:01.150Z","level":"info","component":"planner","operation":"create_plan","message":"Completed create_plan","session_id":"sess_abc123def456","tokens_used":{"input_tokens":2500,"output_tokens":1200,"total_tokens":3700},"metadata":{"plan_id":"plan_xyz789","step_count":5}}

// 4. Memory Retrieval
{"id":"log_006","timestamp":"2026-01-11T14:30:01.200Z","level":"info","component":"memory","operation":"retrieve_context","message":"Starting retrieve_context","session_id":"sess_abc123def456"}
{"id":"log_007","timestamp":"2026-01-11T14:30:01.850Z","level":"info","component":"memory","operation":"retrieve_context","message":"Completed retrieve_context","session_id":"sess_abc123def456","tokens_used":{"input_tokens":1200,"output_tokens":300,"total_tokens":1500},"metadata":{"items_retrieved":10}}

// 5. Plan Execution - Step 1
{"id":"log_008","timestamp":"2026-01-11T14:30:01.900Z","level":"info","component":"reasoning","operation":"direct","message":"Starting direct","session_id":"sess_abc123def456","metadata":{"step_name":"retrieve_context"}}
{"id":"log_009","timestamp":"2026-01-11T14:30:02.080Z","level":"info","component":"reasoning","operation":"direct","message":"Completed direct","session_id":"sess_abc123def456","tokens_used":{"total_tokens":700},"metadata":{"step_name":"retrieve_context","confidence":0.95}}

// 5. Plan Execution - Step 2 (with tool)
{"id":"log_010","timestamp":"2026-01-11T14:30:02.100Z","level":"info","component":"reasoning","operation":"chain_of_thought","message":"Starting chain_of_thought","session_id":"sess_abc123def456","metadata":{"step_name":"research_lead"}}
{"id":"log_011","timestamp":"2026-01-11T14:30:03.000Z","level":"info","component":"tools","operation":"websearch","message":"Starting websearch","session_id":"sess_abc123def456"}
{"id":"log_012","timestamp":"2026-01-11T14:30:04.200Z","level":"info","component":"tools","operation":"websearch","message":"Completed websearch","session_id":"sess_abc123def456","duration_ms":1200,"metadata":{"results_count":5}}
{"id":"log_013","timestamp":"2026-01-11T14:30:04.600Z","level":"info","component":"reasoning","operation":"chain_of_thought","message":"Completed chain_of_thought","session_id":"sess_abc123def456","tokens_used":{"total_tokens":3200},"metadata":{"step_name":"research_lead","confidence":0.88}}

// 5. Plan Execution - Step 3 (complex reasoning)
{"id":"log_014","timestamp":"2026-01-11T14:30:04.650Z","level":"info","component":"reasoning","operation":"tree_of_thoughts","message":"Starting tree_of_thoughts","session_id":"sess_abc123def456","metadata":{"step_name":"assess_bant"}}
{"id":"log_015","timestamp":"2026-01-11T14:30:05.500Z","level":"debug","component":"reasoning","operation":"tree_of_thoughts","message":"Exploring branch 1/4","session_id":"sess_abc123def456"}
{"id":"log_016","timestamp":"2026-01-11T14:30:06.300Z","level":"debug","component":"reasoning","operation":"tree_of_thoughts","message":"Exploring branch 2/4","session_id":"sess_abc123def456"}
{"id":"log_017","timestamp":"2026-01-11T14:30:07.100Z","level":"debug","component":"reasoning","operation":"tree_of_thoughts","message":"Exploring branch 3/4","session_id":"sess_abc123def456"}
{"id":"log_018","timestamp":"2026-01-11T14:30:07.900Z","level":"debug","component":"reasoning","operation":"tree_of_thoughts","message":"Exploring branch 4/4","session_id":"sess_abc123def456"}
{"id":"log_019","timestamp":"2026-01-11T14:30:08.150Z","level":"info","component":"reasoning","operation":"tree_of_thoughts","message":"Completed tree_of_thoughts","session_id":"sess_abc123def456","tokens_used":{"total_tokens":8500},"metadata":{"step_name":"assess_bant","branches_explored":4,"confidence":0.86}}

// 5. Plan Execution - Steps 4 & 5
{"id":"log_020","timestamp":"2026-01-11T14:30:08.200Z","level":"info","component":"reasoning","operation":"chain_of_thought","message":"Completed chain_of_thought","session_id":"sess_abc123def456","tokens_used":{"total_tokens":1200},"metadata":{"step_name":"generate_recommendation"}}
{"id":"log_021","timestamp":"2026-01-11T14:30:08.550Z","level":"info","component":"reasoning","operation":"direct","message":"Completed direct","session_id":"sess_abc123def456","tokens_used":{"total_tokens":800},"metadata":{"step_name":"format_output"}}

// 6. Contract Validation
{"id":"log_022","timestamp":"2026-01-11T14:30:08.600Z","level":"info","component":"contracts","operation":"validate_output","message":"Starting validate_output","session_id":"sess_abc123def456"}
{"id":"log_023","timestamp":"2026-01-11T14:30:08.750Z","level":"info","component":"contracts","operation":"validate_output","message":"Completed validate_output","session_id":"sess_abc123def456","tokens_used":{"total_tokens":1000},"metadata":{"contract_name":"lead_qualification","validation_passed":true}}

// 7. Memory Extraction
{"id":"log_024","timestamp":"2026-01-11T14:30:08.800Z","level":"info","component":"memory","operation":"extract_memories","message":"Starting extract_memories","session_id":"sess_abc123def456"}
{"id":"log_025","timestamp":"2026-01-11T14:30:09.600Z","level":"info","component":"memory","operation":"extract_memories","message":"Completed extract_memories","session_id":"sess_abc123def456","tokens_used":{"total_tokens":2500},"metadata":{"facts_extracted":5}}

// 8. Completion
{"id":"log_026","timestamp":"2026-01-11T14:30:09.700Z","level":"info","component":"cli","operation":"complete_execution","message":"Execution completed","session_id":"sess_abc123def456","metadata":{"success":true,"total_tokens":24000,"duration_seconds":9.7}}
```

---

## Configuration Options

### CLI Configuration

```yaml
# ~/.sigil/config.yaml

cli:
  # Default budget for orchestrate command
  default_budget: 256000

  # Default monitoring settings
  monitor:
    enabled: false
    port: 8000

  # Output settings
  output:
    default_format: json
    include_logs: false

logging:
  # Log level for CLI output
  level: info

  # Buffer settings
  buffer_size: 100
  flush_interval_ms: 100

  # File logging
  file:
    enabled: true
    path: outputs/logs
    format: jsonl
    rotation:
      max_size_mb: 100
      max_files: 30

  # Streaming
  stream:
    enabled: true
    port: 8000
    heartbeat_interval_ms: 5000

budget:
  # Total context window
  total: 256000

  # Warning thresholds
  warning_threshold: 0.80
  critical_threshold: 0.95

  # Component allocations (percentage)
  allocations:
    reasoning: 0.45
    memory: 0.15
    planner: 0.12
    contracts: 0.10
    tools: 0.08
    router: 0.04
    context: 0.03
    orchestrator: 0.02
    evolution: 0.01
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `SIGIL_LOG_LEVEL` | `info` | Default log level |
| `SIGIL_LOG_FILE` | `outputs/logs/sigil.jsonl` | Log file path |
| `SIGIL_TOKEN_BUDGET` | `256000` | Default token budget |
| `SIGIL_MONITOR_PORT` | `8000` | Monitoring WebSocket port |
| `SIGIL_BUFFER_SIZE` | `100` | Log buffer size |
| `SIGIL_FLUSH_INTERVAL` | `100` | Buffer flush interval (ms) |

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2026-01-11 | Initial release |

---

## References

- [Logging Contracts](/Users/zidane/Bland-Agents-Dataset/acti-agent-builder/docs/logging-contracts.md)
- [Monitoring API](/Users/zidane/Bland-Agents-Dataset/acti-agent-builder/docs/monitoring-api.yaml)
- [Monitoring Implementation Guide](/Users/zidane/Bland-Agents-Dataset/acti-agent-builder/docs/monitoring-implementation-guide.md)
- [Sigil v2 Architecture](/Users/zidane/Bland-Agents-Dataset/Sigil_V2.md)
