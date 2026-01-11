# ACTi Agent Builder - Breaking Issues Analysis

Issues identified during Phase 2.5 CLI testing (2026-01-10). These are critical architectural issues that cause agent execution failures.

---

## Executive Summary

Two interconnected critical issues are causing agent failures:

1. **Infinite Loop / Non-Termination**: Agents complete their task but continue calling tools indefinitely instead of stopping
2. **GraphRecursionError**: After 25 iterations, LangGraph raises an error because no stop condition was reached

Root causes involve:
- Missing `recursion_limit` config propagation during streaming
- Lack of explicit termination instructions in agent prompts
- Builder agent re-invoking tools after successful execution
- No "task complete" signal mechanism

---

## Issue 15: GraphRecursionError - Recursion Limit of 25 Reached

### Symptoms

```
langgraph.errors.GraphRecursionError: Recursion limit of 25 reached without hitting a stop condition.
You can increase the limit by setting the `recursion_limit` config key.
For troubleshooting, visit: https://docs.langchain.com/oss/python/langgraph/errors/GRAPH_RECURSION_LIMIT
```

### Observed Behavior

1. User asks builder to demo an agent (e.g., "give me a demo for this agent")
2. Builder calls `execute_created_agent` which runs successfully
3. Agent performs websearch, gets results, drafts comprehensive report
4. Instead of stopping, agent/builder continues looping:
   - Calls more tools
   - Re-runs the same agent with variations
   - Updates todos
   - Eventually hits recursion limit

### Root Cause Analysis

#### Cause 1: Config Not Propagating to Streaming API

**Location:** `venv/lib/python3.12/site-packages/deepagents/graph.py:200`

```python
return create_agent(
    model,
    ...
).with_config({"recursion_limit": 1000})
```

The deepagents library sets `recursion_limit: 1000`, but the CLI uses `astream_events()` without passing config:

**Location:** `src/cli.py:2412`

```python
async for event in agent.astream_events(
    {"messages": patched_messages},
    version="v2",
):
```

**Problem:** `astream_events()` may not inherit the `with_config()` settings, falling back to LangGraph's default of 25.

**Evidence:** Error shows limit of 25 despite deepagents setting 1000.

#### Cause 2: No Explicit Termination Instructions

**Problem:** Created agents and the builder itself lack explicit instructions about WHEN to stop calling tools and provide a final response.

LangGraph agents stop when the LLM produces a response with NO tool_calls. But without explicit termination guidance:
- Agent keeps finding "more work to do"
- Calls write_todos to update status
- Searches for additional information
- Never emits a "final" message without tool calls

#### Cause 3: Builder Loops After Tool Success

**Observed in logs:**

```
[execute_created_agent complete]
...
The execution timed out - likely because this is a very comprehensive research task...
Let me try running it with a longer timeout:
[Calling execute_created_agent...]  <- LOOPS AGAIN
```

The builder interprets successful tool results and decides to call MORE tools instead of presenting results to the user.

---

## Issue 16: Agent Non-Termination (Infinite Tool Loop)

### Symptoms

Agent successfully completes task (websearch, report generation) but keeps running:

```
[tavily-search complete]
I now have comprehensive research data. Let me compile the full report:
[execute_created_agent complete]    <- Task completed!
...
The execution timed out... Let me try running it with a longer timeout:
[Calling execute_created_agent...]  <- Why is it running again?!
```

### Root Cause Analysis

#### Cause 1: LangGraph Stop Condition Requirements

Per [LangGraph documentation](https://docs.langchain.com/oss/python/langgraph/errors/GRAPH_RECURSION_LIMIT):

> A LangGraph StateGraph reaches its maximum step limit before completing execution when there's no proper termination condition.

**Stop condition requirement:** The LLM must emit a response WITHOUT any `tool_calls` to signal completion.

**Problem:** Neither the builder nor created agents have system prompt instructions that explicitly tell them:
1. When their task is complete
2. To stop calling tools and provide a final response
3. Not to re-run tools after successful completion

#### Cause 2: FilesystemMiddleware Injects Tools

**Location:** `venv/lib/python3.12/site-packages/deepagents/middleware/filesystem.py`

Even when subagents are defined with `tools: []`, the `FilesystemMiddleware` injects filesystem tools (read_file, write_file, glob, grep, etc.).

This means:
- Subagents that should be "tool-less" (like prompt-engineer) still have tools available
- They may call these tools indefinitely instead of just outputting text

#### Cause 3: TodoListMiddleware Encourages Continuous Work

**Location:** `venv/lib/python3.12/site-packages/deepagents/graph.py:132-134`

```python
subagent_middleware: list[AgentMiddleware] = [
    TodoListMiddleware(),
]
```

The `write_todos` tool encourages agents to:
- Track tasks
- Mark tasks complete
- Find new tasks
- Continue working

This creates a loop where completing one task leads to identifying the next.

---

## Technical Deep Dive

### How LangGraph Determines When to Stop

```
                    ┌─────────────────┐
                    │   LLM Response  │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │  Has tool_calls? │
                    └────────┬────────┘
                             │
              ┌──────────────┴──────────────┐
              │ YES                         │ NO
              ▼                             ▼
    ┌─────────────────┐           ┌─────────────────┐
    │  Execute Tools  │           │      STOP       │
    │  Continue Loop  │           │  Return Result  │
    └─────────────────┘           └─────────────────┘
```

**Key insight:** The ONLY way to stop is for the LLM to NOT include tool_calls in its response.

### Why Our Agents Don't Stop

1. **Builder has many tools**: `BUILDER_TOOLS` includes 5+ tools
2. **Created agents have MCP tools**: websearch, voice, calendar, etc.
3. **Middleware adds more tools**: TodoListMiddleware, FilesystemMiddleware
4. **No explicit "STOP NOW" instruction**: Prompts don't tell agents when to stop

The LLM sees available tools and thinks "I could do more!" instead of "I'm done."

---

## Proposed Fixes

### Fix 1: Propagate recursion_limit to Streaming API

**File:** `src/cli.py`

**Change:** Pass config explicitly to `astream_events()`

```python
# Current (broken)
async for event in agent.astream_events(
    {"messages": patched_messages},
    version="v2",
):

# Fixed
async for event in agent.astream_events(
    {"messages": patched_messages},
    version="v2",
    config={"recursion_limit": 100},  # Or inherit from agent config
):
```

### Fix 2: Add Explicit Termination Instructions to Builder Prompt

**File:** `src/prompts.py`

**Add to BUILDER_SYSTEM_PROMPT:**

```python
## CRITICAL: Task Completion Protocol

When you have completed a user's request:

1. **STOP calling tools** - Do not call execute_created_agent again
2. **Present the result** - Summarize what was accomplished
3. **Ask for feedback** - Only if explicitly needed

After execute_created_agent returns a successful result:
- DO NOT re-run the agent with different parameters
- DO NOT try to "improve" or "expand" the result
- Simply present the agent's response to the user

If a tool times out or fails:
- Report the error to the user
- Ask if they want to retry
- DO NOT automatically retry
```

### Fix 3: Add Termination Instructions to Created Agent Prompts

**File:** `src/prompts.py` (PROMPT_ENGINEER_SYSTEM_PROMPT)

**Ensure generated prompts include:**

```
## Task Completion

When your task is complete:
1. Provide your final response
2. DO NOT call any more tools
3. DO NOT look for additional work
4. Simply output your result and stop

You will know the task is complete when:
- You have answered the user's question
- You have gathered sufficient information
- You have completed the requested action
```

### Fix 4: Implement Max Turns / Iteration Limit

**File:** `src/mcp_integration.py`

**Add max_turns parameter:**

```python
async def create_agent_with_tools(
    agent_config: AgentConfig,
    *,
    skip_unavailable: bool = False,
    timeout: float = 30.0,
    max_turns: int = 10,  # NEW: Maximum tool call rounds
) -> CompiledStateGraph:
    ...
    # Set recursion limit based on max_turns
    # Each "turn" can involve multiple internal steps
    return agent.with_config({
        "recursion_limit": max_turns * 5,
        "max_assistant_turns": max_turns,
    })
```

### Fix 5: Add Conversation Length Check

**File:** `src/cli.py` or `src/tools.py`

**Add safeguard:**

```python
# In execute_created_agent or _stream_agent_response
if len(messages) > MAX_MESSAGES:
    logger.warning("Conversation exceeded %d messages, forcing termination", MAX_MESSAGES)
    return "Task terminated due to excessive message count. Please simplify your request."
```

### Fix 6: Use interrupt_before for Control

**Option:** Add human-in-the-loop after certain tool calls:

```python
agent = create_agent(
    ...
    interrupt_before=["execute_created_agent"],  # Pause before re-running
)
```

---

## References

- [GRAPH_RECURSION_LIMIT - LangChain Docs](https://docs.langchain.com/oss/python/langgraph/errors/GRAPH_RECURSION_LIMIT)
- [How to create and control loops - LangGraph](https://langchain-ai.github.io/langgraphjs/how-tos/recursion-limit/)
- [Prevent Infinite Tool Call Loop - GitHub Issue #26019](https://github.com/langchain-ai/langchain/issues/26019)
- [Agent Recursion Between Tools - GitHub Discussion #1725](https://github.com/langchain-ai/langgraph/discussions/1725)
- [DeepSeek V3 API Call Does Not Stop - GitHub Issue #3097](https://github.com/langchain-ai/langgraph/issues/3097)

---

## Priority Matrix

| Issue | Severity | Effort | Priority |
|-------|----------|--------|----------|
| Config propagation (Fix 1) | Critical | Low | P0 |
| Builder termination instructions (Fix 2) | Critical | Low | P0 |
| Created agent termination (Fix 3) | High | Medium | P1 |
| Max turns limit (Fix 4) | High | Medium | P1 |
| Conversation length check (Fix 5) | Medium | Low | P2 |
| Human-in-the-loop (Fix 6) | Low | High | P3 |

---

## Immediate Action Items

1. **[P0] Add recursion_limit to streaming config** - 5 min fix
2. **[P0] Add termination instructions to BUILDER_SYSTEM_PROMPT** - 10 min fix
3. **[P1] Update PROMPT_ENGINEER_SYSTEM_PROMPT to include termination guidance** - 15 min
4. **[P1] Add max_turns parameter to execute_created_agent** - 30 min

---

*Created: 2026-01-10*
*Status: Analysis Complete - Fixes Pending*

---

# SUPPLEMENTAL ANALYSIS: Why Previous Fixes Failed

**Date:** 2026-01-10 14:25 UTC
**Status:** Applied Fixes 1-4 and tested. Issue persists. Root cause analysis below.

---

## Evidence: Infinite Loop Still Occurs After Fixes 1-4

**CLI Test Output Observed:**

```
builder> perform brief search on the intersection of meme culture and AI trends...

⠋ Thinking...
I'll help you research these three related topics...
[Calling task...]
[Calling list_created_agents...]
[Calling list_available_tools...]
[list_available_tools complete]
[Calling write_todos...]
[write_todos complete]

[Calling create_agent_config...]
[create_agent_config complete]  ← Created early_teen_researcher

[Calling execute_created_agent...]
[tavily-search complete]  ← Search succeeded!

[Calling create_agent_config...]  ← Created teen_tech_researcher
[Calling execute_created_agent...]  ← Executed again

[Calling write_todos...]
[execute_created_agent complete]  ← Agent completed!

[Calling tavily-search...]  ← But searches AGAIN
[Calling execute_created_agent...]  ← Re-invokes AGAIN

[tavily-search complete]
[execute_created_agent complete]

[Calling create_agent_config...]  ← Creates ANOTHER agent
[Calling execute_created_agent...]  ← And runs AGAIN
...continues indefinitely...
```

**Pattern Analysis:**
1. First tool call: success ✓
2. Builder's response: "I see there's already an agent... Let me create another..."
3. Creates 2nd agent
4. Calls agent 2: success ✓
5. Builder's response: "Now let me execute this agent with comprehensive research..."
6. Calls agent 3, agent 4, search tools repeatedly
7. **Never stops calling tools**
8. Eventually hits timeout or recursion limit

**Key Observation:** Fixes 1-4 did NOT stop this behavior.

---

## Why Fixes 1-4 Are Insufficient

### Issue 1: Builder Agent Itself Has No max_turns Applied

**Current Code (src/builder.py:71-75):**

```python
def create_builder(
    model: str = DEFAULT_MODEL,
    root_dir: str | Path | None = None,
    virtual_mode: bool = True,
    # ← NO max_turns parameter exists
) -> CompiledStateGraph:
    # ...
    return create_agent(...).with_config({"recursion_limit": 1000})
```

**Problem:**
- Fix 4 added `max_turns` to `execute_created_agent()` function (a tool)
- But `execute_created_agent` is called BY the builder agent
- The **builder agent itself** has NO max_turns
- Builder runs with `recursion_limit: 1000` which it never hits because:
  - Not a hard limit on tool calls
  - Just a limit on graph recursion steps
  - Each tool call might consume 2-5 recursion steps
  - Builder can call 200+ tools before hitting 1000 steps
  - CLI timeout or connection error happens first

**Call Chain:**
```
create_builder() ← Creates builder with recursion_limit=1000
    ↓
Builder Agent Loop (UNLIMITED tool invocations)
    ├─ Tool 1: list_created_agents() → 2 steps
    ├─ Tool 2: list_available_tools() → 2 steps
    ├─ Tool 3: write_todos() → 2 steps
    ├─ Tool 4: create_agent_config() → 2 steps
    ├─ Tool 5: execute_created_agent() → 10 steps (timeout)
    ├─ LLM reasoning: "Let me try again..." → 5 steps
    ├─ Tool 6: execute_created_agent() retry → 10 steps
    ├─ ...continues forever or until real error
    └─ Eventually network timeout or manual interrupt
```

**The Fix Didn't Work Because:** The max_turns only applies to agents created with execute_created_agent, not to the builder agent itself.

### Issue 2: Termination Instructions Are Ignored by LLM

**What We Added (src/prompts.py):**

```python
BUILDER_SYSTEM_PROMPT += """
## CRITICAL: Task Completion Protocol

When you have completed a user's request:
1. STOP calling tools
2. Present the result
...
After execute_created_agent returns a successful result:
- DO NOT re-run the agent with different parameters
- DO NOT try to "improve" or "expand" the result
- Simply present the agent's response to the user
"""
```

**What Builder Did Instead (from logs):**

```
[tavily-search complete]
I now have comprehensive research data. Let me compile the full report:
[execute_created_agent complete]    ← SUCCEEDED!

2026-01-10 14:17:33,807 - src.mcp_integration - INFO - Creating agent...
2026-01-10 14:17:35,104 - src.tools - ERROR - Agent execution failed: Connection error
The websearch MCP tool connection failed. Let me provide you with a comprehensive research summary:
[Calling write_todos...]
[Calling tavily-search...]  ← Calls tools AGAIN despite instruction
[Calling execute_created_agent...]  ← Re-runs agent AGAIN
```

**Why The Instruction Was Ignored:**

1. **LLM saw a problem (timeout/connection error)**
2. **LLM's reasoning:** "Task incomplete due to connection error. I should retry."
3. **LLM ignored:** "DO NOT retry" instruction because the tool FAILED, not succeeded
4. **Pattern:** "If task failed, retry. If task succeeded, stop." — But LLM doesn't distinguish between failure and non-task-completion

**The Fix Didn't Work Because:**
- Prompt instructions are suggestions, not hard rules
- LLMs will find reasons to use available tools
- Failure + available tools = LLM decides to retry
- No enforcement mechanism exists

### Issue 3: max_turns on execute_created_agent Doesn't Prevent Builder Loop

**What We Added (src/tools.py, src/mcp_integration.py):**

```python
async def create_agent_with_tools(
    agent_config: AgentConfig,
    *,
    max_turns: int = 10,  # ← NEW
) -> CompiledStateGraph:
    recursion_limit = max_turns * 5
    return agent.with_config({
        "recursion_limit": recursion_limit,
    })

@tool
def execute_created_agent(
    agent_name: str,
    task: str,
    max_turns: int = 10,  # ← NEW
) -> str:
    ...
```

**The Problem:**

```
Builder Agent (UNLIMITED RECURSION)
    ├─ Tool: execute_created_agent(agent_name="web_researcher", max_turns=10)
    │   └─ Created agent runs for max 10 turns ✓
    │   └─ Returns result ✓
    ├─ Tool: execute_created_agent(agent_name="teen_researcher", max_turns=10)  ← 2nd call
    │   └─ Created agent runs for max 10 turns ✓
    │   └─ Returns result ✓
    ├─ Tool: execute_created_agent(agent_name="culture_analyst", max_turns=10)  ← 3rd call
    │   └─ Created agent runs for max 10 turns ✓
    ├─ Tool: execute_created_agent(agent_name="web_researcher", max_turns=10)  ← 4th call (retry)
    │   └─ Created agent runs for max 10 turns ✓
    └─ Builder keeps calling tools INDEFINITELY
       The max_turns only limits EACH individual agent, not the builder
```

**The Fix Didn't Work Because:**
- max_turns limits each individual agent execution
- Builder can call execute_created_agent multiple times
- Builder itself has no max_turns to limit how many times it calls tools
- Like having a timeout on each web request, but nothing limiting total requests

### Issue 4: Timeout Handling Creates Retry Loops

**From Logs:**

```
[execute_created_agent complete]
The execution timed out - likely because this is a very comprehensive research task...
Let me try running it with a longer timeout:
[Calling execute_created_agent...]
```

**Root Cause:**
- Agent times out or returns error
- Builder sees: "Task returned an error"
- Builder's reasoning: "Let me retry with different parameters"
- Calls tool again
- Timeout happens again
- Loops until recursion limit or manual interrupt

**Why No Protection:**
- Fixes 1-4 don't prevent timeout retry loops
- No detection of repeated failures
- No limit on retry attempts
- No feedback mechanism to user

---

## The Real Problem: Builder Agent Needs Hard Stops

**The fundamental issue:**

We cannot rely on LLM-based termination decisions when:
1. Multiple tools are available
2. Each tool returns some result (success, error, timeout)
3. LLM can interpret any result as "reason to try something else"
4. No enforcement mechanism exists

**Architectural Limitation:**
- LLMs are designed to keep trying until they succeed or exhaust context
- Prompts are advisory, not enforceable
- Reversing this requires **structural changes, not prompt changes**

---

## Solution Architecture

### Required Changes:

**1. Hard Recursion Limit on Builder Agent (Fix 7)**
- Add `max_turns` parameter to `create_builder()`
- Apply recursion_limit = max_turns * 5 to builder
- Causes hard stop after tool call limit reached
- User sees "Stopped: recursion limit reached" instead of hang

**2. Detection and Stopping of Retry Loops (Fix 8)**
- Detect when same tool called 3+ times consecutively
- Return error instead of looping
- Prevents runaway behavior from connection issues

**3. Message Count Safeguard (Fix 9)**
- Track messages in conversation
- Force stop after 100 messages
- Detects loop patterns

**4. Explicit Tool Success Signals (Fix 8)**
- execute_created_agent returns status="success" field
- Builder prompt tells LLM: "If status=success, STOP. Do NOT retry."
- Status field makes success unambiguous

**5. User Confirmation for Retries (Fix 8)**
- On tool error, prompt builder: "Ask user before retrying"
- Don't automatically retry
- Gives user control

---

## Next Steps

These detailed analysis and fixes are documented for implementation. **Do not attempt to solve by adding more prompts—that won't work.** Implement the structural fixes in the solution architecture above.


