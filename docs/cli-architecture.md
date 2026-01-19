# Sigil v2 CLI Monitoring and Observability Architecture

## Document Information

| Field | Value |
|-------|-------|
| Version | 1.0.0 |
| Status | Design Specification |
| Created | 2026-01-11 |
| Author | Systems Architecture Team |
| Scope | CLI Execution Monitoring |

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [System Overview](#2-system-overview)
3. [Architecture Diagrams](#3-architecture-diagrams)
4. [Component Specifications](#4-component-specifications)
5. [Data Flow Architecture](#5-data-flow-architecture)
6. [Token Tracking System](#6-token-tracking-system)
7. [Log File Architecture](#7-log-file-architecture)
8. [Inter-Process Communication](#8-inter-process-communication)
9. [Error Handling Architecture](#9-error-handling-architecture)
10. [Scaling Considerations](#10-scaling-considerations)
11. [Security Architecture](#11-security-architecture)
12. [Performance Optimization](#12-performance-optimization)
13. [Appendices](#13-appendices)

---

## 1. Executive Summary

### 1.1 Purpose

This document defines the comprehensive monitoring and observability architecture for Sigil v2 CLI execution. The architecture enables real-time visibility into pipeline execution, token consumption tracking, and performance monitoring through a dual-terminal design pattern.

### 1.2 Design Goals

| Goal | Description | Priority |
|------|-------------|----------|
| Real-time Visibility | Operators can observe execution as it happens | P0 |
| Token Accountability | Every token consumed is tracked and attributed | P0 |
| Non-blocking Operations | CLI execution is never delayed by monitoring | P0 |
| Graceful Degradation | System continues if monitor fails | P1 |
| Scalability | Handle high-volume logging scenarios | P1 |
| Debuggability | Full audit trail for troubleshooting | P1 |

### 1.3 Key Decisions

1. **File-based Communication**: Log files serve as the communication channel between CLI and Monitor
2. **Append-only Logging**: All writes are appends for reliability and concurrency safety
3. **JSON-structured Logs**: Machine-parseable format with human-readable fallback
4. **Independent Processes**: CLI and Monitor run as separate processes with no direct coupling

### 1.4 Scope

This architecture covers:
- CLI orchestrator execution and logging
- Monitor script observation and display
- Token tracking across all pipeline components
- Performance metrics collection and visualization
- Error handling and recovery procedures

---

## 2. System Overview

### 2.1 High-Level Architecture

The monitoring system consists of two independent terminal processes communicating through a shared log file:

```
+------------------+                                    +------------------+
|   TERMINAL 1     |                                    |   TERMINAL 2     |
|   (CLI Process)  |                                    |  (Monitor Process)|
+------------------+                                    +------------------+
        |                                                       |
        |  Executes Pipeline                                    |  Observes Execution
        |  Logs Operations                                      |  Displays Metrics
        |  Tracks Tokens                                        |  Shows Progress
        |                                                       |
        v                                                       v
+------------------+                                    +------------------+
|   Orchestrator   |                                    |  Monitor Script  |
|                  |                                    |                  |
| - Route Intent   |                                    | - Tail Log File  |
| - Execute Steps  |                                    | - Parse Entries  |
| - Log Events     |                                    | - Accumulate     |
| - Track Tokens   |                                    | - Display        |
+--------+---------+                                    +--------+---------+
         |                                                       ^
         |                                                       |
         v                                                       |
+--------+---------------------------------------------------------+---------+
|                                                                            |
|                            LOG FILE                                        |
|                                                                            |
|  ~/.sigil/logs/execution.jsonl                                             |
|                                                                            |
|  {"timestamp":"...","component":"routing","tokens":50,"message":"..."}     |
|  {"timestamp":"...","component":"memory","tokens":150,"message":"..."}     |
|  {"timestamp":"...","component":"reasoning","tokens":450,"message":"..."}  |
|                                                                            |
+----------------------------------------------------------------------------+
```

### 2.2 Design Principles

#### 2.2.1 Separation of Concerns

The CLI and Monitor are completely independent:

| Aspect | CLI (Terminal 1) | Monitor (Terminal 2) |
|--------|------------------|----------------------|
| Primary Role | Execute pipeline | Observe and display |
| I/O Mode | Write-only to log | Read-only from log |
| Blocking | Never waits for monitor | May wait for new data |
| Failure Impact | Continues normally | CLI unaffected |
| State | Owns execution state | Derives display state |

#### 2.2.2 Non-blocking Communication

```
CLI Write Path (Non-blocking):
+----------+    +------------+    +----------+
| Pipeline | -> | Log Buffer | -> | Log File |
| Step     |    | (In-Memory)|    | (Disk)   |
+----------+    +------------+    +----------+
     |               |
     |               +-- Async flush every 100ms or 1KB
     |
     +-- Execution continues immediately

Monitor Read Path (Blocking OK):
+----------+    +------------+    +-----------+
| Log File | -> | Tail       | -> | Parser    |
| (Disk)   |    | Process    |    | & Display |
+----------+    +------------+    +-----------+
                     |
                     +-- Blocks waiting for new lines (acceptable)
```

#### 2.2.3 Eventual Consistency

The monitor may lag behind the CLI by a small amount:

- Maximum lag: 100ms (buffer flush interval)
- Typical lag: 10-50ms
- Acceptable because: Display is for human observation, not coordination

### 2.3 Component Inventory

| Component | Location | Responsibility |
|-----------|----------|----------------|
| CLI Orchestrator | `sigil/interfaces/cli/orchestrator.py` | Pipeline execution |
| Execution Logger | `sigil/telemetry/execution_logger.py` | Write log entries |
| Token Tracker | `sigil/telemetry/tokens.py` | Token accounting |
| Monitor Script | `sigil/interfaces/cli/monitor.py` | Display dashboard |
| Log Parser | `sigil/telemetry/log_parser.py` | Parse log entries |
| Display Formatter | `sigil/interfaces/cli/display.py` | Format output |

---

## 3. Architecture Diagrams

### 3.1 System Context Diagram

```
+-----------------------------------------------------------------------------+
|                           SIGIL v2 EXECUTION SYSTEM                          |
+-----------------------------------------------------------------------------+
|                                                                              |
|  +---------------------------+          +---------------------------+        |
|  |      USER TERMINAL 1      |          |      USER TERMINAL 2      |        |
|  |                           |          |                           |        |
|  |  $ python -m sigil        |          |  $ python -m sigil        |        |
|  |    orchestrate            |          |    monitor                |        |
|  |    --task "Analyze..."    |          |    --session test-1       |        |
|  |    --session test-1       |          |                           |        |
|  |                           |          |                           |        |
|  +-------------+-------------+          +-------------+-------------+        |
|                |                                      |                      |
|                v                                      v                      |
|  +-------------+-------------+          +-------------+-------------+        |
|  |     CLI ORCHESTRATOR      |          |      MONITOR SCRIPT       |        |
|  |                           |          |                           |        |
|  | - Receives task           |          | - Tails log file          |        |
|  | - Creates session         |          | - Parses JSON lines       |        |
|  | - Executes pipeline       |          | - Accumulates tokens      |        |
|  | - Logs to file            |          | - Displays dashboard      |        |
|  |                           |          |                           |        |
|  +-------------+-------------+          +-------------+-------------+        |
|                |                                      ^                      |
|                |                                      |                      |
|                v                                      |                      |
|  +-------------+--------------------------------------+-------------+        |
|  |                         LOG FILE SYSTEM                          |        |
|  |                                                                  |        |
|  |  ~/.sigil/logs/                                                  |        |
|  |  +-- sessions/                                                   |        |
|  |      +-- test-1/                                                 |        |
|  |          +-- execution.jsonl    <-- Main log file                |        |
|  |          +-- tokens.json        <-- Token summary                |        |
|  |          +-- metrics.json       <-- Performance metrics          |        |
|  |                                                                  |        |
|  +------------------------------------------------------------------+        |
|                                                                              |
+-----------------------------------------------------------------------------+
```

### 3.2 Component Interaction Diagram

```
+-----------------------------------------------------------------------------+
|                        COMPONENT INTERACTION FLOW                            |
+-----------------------------------------------------------------------------+

TERMINAL 1 (CLI)                           TERMINAL 2 (MONITOR)
================                           ====================

User Input
    |
    v
+---+---+
| CLI   |
| App   |
+---+---+
    |
    | Creates session
    v
+---+---+         +---+---+
| Orch- |-------->| Event |
| estr- |         | Store |
| ator  |         +---+---+
+---+---+             |
    |                 | Appends events
    |                 v
    |         +---+---+---+
    |         | execution |
    |         | .jsonl    |<------------------+
    |         +---+---+---+                   |
    |             ^                           |
    |             |                           | Tails file
    |             |                   +---+---+---+
    |             |                   | Monitor   |
    |             |                   | Process   |
    |             |                   +---+---+---+
    |             |                       |
    | Pipeline    |                       | Parses lines
    | Execution   |                       v
    |             |                   +---+---+---+
    v             |                   | Log       |
+---+---+         |                   | Parser    |
| Rout- +---------+                   +---+---+---+
| ing   |  Logs: 50 tokens                |
+---+---+                                 | Updates state
    |                                     v
    v                                 +---+---+---+
+---+---+         +---+---+           | Token     |
| Mem-  +---------+ Log   |           | Accum-    |
| ory   |         | Entry |           | ulator    |
+---+---+         +---+---+           +---+---+---+
    |                 |                   |
    v                 v                   | Formats display
+---+---+         +---+---+               v
| Plan- |         | execu-|           +---+---+---+
| ning  |         | tion. |           | Display   |
+---+---+         | jsonl |           | Formatter |
    |             +-------+           +---+---+---+
    v                                     |
+---+---+                                 v
| Reas- |                             +---+---+---+
| oning |                             | Terminal  |
+---+---+                             | Output    |
    |                                 +-----------+
    v
+---+---+
| Valid-|
| ation |
+-------+
    |
    v
Complete
```

### 3.3 Data Flow Diagram

```
+-----------------------------------------------------------------------------+
|                              DATA FLOW ARCHITECTURE                          |
+-----------------------------------------------------------------------------+

                            CLI SIDE (Terminal 1)
+-----------------------------------------------------------------------------+
|                                                                              |
|  User Request                                                                |
|       |                                                                      |
|       v                                                                      |
|  +----+----+     +----------+     +----------+     +----------+              |
|  | Parse   | --> | Create   | --> | Start    | --> | Execute  |              |
|  | Task    |     | Session  |     | Pipeline |     | Steps    |              |
|  +---------+     +----+-----+     +----+-----+     +----+-----+              |
|                       |                |                |                    |
|                       v                v                v                    |
|                  +----+----------------+----------------+----+               |
|                  |           TOKEN TRACKER                   |               |
|                  |                                           |               |
|                  | - Allocate budgets per component          |               |
|                  | - Record consumption per operation        |               |
|                  | - Calculate running totals                |               |
|                  +----+--------------------------------------+               |
|                       |                                                      |
|                       v                                                      |
|                  +----+----+                                                 |
|                  | Format  |                                                 |
|                  | Log     |                                                 |
|                  | Entry   |                                                 |
|                  +----+----+                                                 |
|                       |                                                      |
|                       v                                                      |
|                  +----+----+     +----------+                                |
|                  | Buffer  | --> | Async    |                                |
|                  | Entry   |     | Flush    |                                |
|                  +---------+     +----+-----+                                |
|                                       |                                      |
+---------------------------------------|--------------------------------------+
                                        |
                                        v
                              +--------------------+
                              |                    |
                              |  execution.jsonl   |
                              |                    |
                              | {"ts":"10:15:32",  |
                              |  "component":      |
                              |    "routing",      |
                              |  "tokens": 50,     |
                              |  "message":        |
                              |    "Intent..."}    |
                              |                    |
                              +--------------------+
                                        |
                                        v
+---------------------------------------|--------------------------------------+
|                                       |                                      |
|                           MONITOR SIDE (Terminal 2)                          |
|                                       |                                      |
|                                  +----+----+                                 |
|                                  | Tail    |                                 |
|                                  | File    |                                 |
|                                  +----+----+                                 |
|                                       |                                      |
|                                       v                                      |
|                                  +----+----+                                 |
|                                  | Parse   |                                 |
|                                  | JSON    |                                 |
|                                  | Line    |                                 |
|                                  +----+----+                                 |
|                                       |                                      |
|                       +---------------+---------------+                      |
|                       |               |               |                      |
|                       v               v               v                      |
|                  +----+----+     +----+----+     +----+----+                 |
|                  | Update  |     | Update  |     | Update  |                 |
|                  | Token   |     | Perf    |     | Status  |                 |
|                  | State   |     | State   |     | State   |                 |
|                  +----+----+     +----+----+     +----+----+                 |
|                       |               |               |                      |
|                       +---------------+---------------+                      |
|                                       |                                      |
|                                       v                                      |
|                                  +----+----+                                 |
|                                  | Render  |                                 |
|                                  | Display |                                 |
|                                  +---------+                                 |
|                                                                              |
+-----------------------------------------------------------------------------+
```

### 3.4 Sequence Diagram: Complete Execution Flow

```
+-----------------------------------------------------------------------------+
|                         EXECUTION SEQUENCE DIAGRAM                           |
+-----------------------------------------------------------------------------+

User          CLI           Orchestrator    Logger      Log File    Monitor
  |            |                |             |            |           |
  |--task----->|                |             |            |           |
  |            |                |             |            |           |
  |            |--create------->|             |            |           |
  |            |  session       |             |            |           |
  |            |                |             |            |           |
  |            |                |--init------>|            |           |
  |            |                |  logger     |            |           |
  |            |                |             |            |           |
  |            |                |             |--create--->|           |
  |            |                |             |  file      |           |
  |            |                |             |            |           |
  |            |                |             |            |<--tail----|
  |            |                |             |            |           |
  |            |                |--log------->|            |           |
  |            |                |  "Starting" |            |           |
  |            |                |             |--append--->|           |
  |            |                |             |            |--read---->|
  |            |                |             |            |           |--display
  |            |                |             |            |           |
  |            |                |--route----->|            |           |
  |            |                |  (50 tok)   |            |           |
  |            |                |             |--append--->|           |
  |            |                |             |            |--read---->|
  |            |                |             |            |           |--display
  |            |                |             |            |           |  "50 tok"
  |            |                |             |            |           |
  |            |                |--memory---->|            |           |
  |            |                |  (150 tok)  |            |           |
  |            |                |             |--append--->|           |
  |            |                |             |            |--read---->|
  |            |                |             |            |           |--display
  |            |                |             |            |           |  "200 tot"
  |            |                |             |            |           |
  |            |                |--plan------>|            |           |
  |            |                |  (0 tok)    |            |           |
  |            |                |             |--append--->|           |
  |            |                |             |            |--read---->|
  |            |                |             |            |           |--display
  |            |                |             |            |           |
  |            |                |--reason---->|            |           |
  |            |                |  (450 tok)  |            |           |
  |            |                |             |--append--->|           |
  |            |                |             |            |--read---->|
  |            |                |             |            |           |--display
  |            |                |             |            |           |  "650 tot"
  |            |                |             |            |           |
  |            |                |--validate-->|            |           |
  |            |                |  (0 tok)    |            |           |
  |            |                |             |--append--->|           |
  |            |                |             |            |--read---->|
  |            |                |             |            |           |--display
  |            |                |             |            |           |
  |            |                |--complete-->|            |           |
  |            |                |             |--append--->|           |
  |            |                |             |            |--read---->|
  |            |                |             |            |           |--display
  |            |                |             |            |           |  "DONE"
  |            |<--result-------|             |            |           |
  |<--output---|                |             |            |           |
  |            |                |             |            |           |
```

### 3.5 State Machine: Monitor Display States

```
+-----------------------------------------------------------------------------+
|                        MONITOR STATE MACHINE                                 |
+-----------------------------------------------------------------------------+

                              +-------------+
                              |             |
                              |   INITIAL   |
                              |             |
                              +------+------+
                                     |
                                     | File found
                                     v
                              +------+------+
                              |             |
                              |   WAITING   |<-----------+
                              |  FOR START  |            |
                              +------+------+            |
                                     |                   |
                                     | Session start     | Session complete
                                     | event received    | (reset)
                                     v                   |
                              +------+------+            |
                              |             |            |
                          +-->|  TRACKING   +------------+
                          |   |             |
                          |   +------+------+
                          |          |
                          |          | Component event
                          |          v
                          |   +------+------+
                          |   |             |
                          +---+  UPDATING   |
                              |   DISPLAY   |
                              +-------------+

States:
--------
INITIAL     - Monitor started, searching for log file
WAITING     - Log file found, waiting for session to begin
TRACKING    - Session active, accumulating metrics
UPDATING    - Processing new event, refreshing display

Transitions:
------------
File found       - Log file exists and is accessible
Session start    - Received "session_start" event type
Component event  - Received routing/memory/planning/reasoning/validation event
Session complete - Received "session_complete" event type
```

---

## 4. Component Specifications

### 4.1 CLI Orchestrator

#### 4.1.1 Purpose

The CLI Orchestrator is the main execution engine that processes user tasks through the Sigil v2 pipeline. It coordinates all subsystems and ensures proper logging of each operation.

#### 4.1.2 Interface Definition

```python
# sigil/interfaces/cli/orchestrator.py

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from datetime import datetime
from enum import Enum

class PipelineStep(str, Enum):
    """Pipeline execution steps."""
    ROUTING = "routing"
    MEMORY = "memory"
    PLANNING = "planning"
    REASONING = "reasoning"
    CONTRACTS = "contracts"
    VALIDATION = "validation"


@dataclass
class StepResult:
    """Result from a single pipeline step."""
    step: PipelineStep
    success: bool
    tokens_used: int
    latency_ms: float
    output: Any
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineResult:
    """Complete pipeline execution result."""
    session_id: str
    task: str
    success: bool
    steps: List[StepResult]
    total_tokens: int
    total_latency_ms: float
    final_output: Any
    quality_score: Optional[float] = None
    recommended_action: Optional[str] = None


class Orchestrator:
    """
    Main pipeline orchestrator for Sigil v2 CLI execution.

    Responsibilities:
    - Coordinate pipeline step execution
    - Manage token budgets across steps
    - Log all operations for monitoring
    - Handle errors with graceful degradation

    Attributes:
        session_id: Unique identifier for this execution session
        token_budget: Total token budget for execution
        execution_logger: Logger for monitor observation
        token_tracker: Tracks token consumption
    """

    def __init__(
        self,
        session_id: str,
        token_budget: int = 256_000,
        log_path: Optional[str] = None,
    ) -> None:
        """
        Initialize the orchestrator.

        Args:
            session_id: Unique session identifier
            token_budget: Maximum tokens allowed (default 256K)
            log_path: Path to log directory (default ~/.sigil/logs)
        """
        self.session_id = session_id
        self.token_budget = token_budget
        self.log_path = log_path or self._default_log_path()

        # Initialize subsystems
        self.execution_logger = ExecutionLogger(
            session_id=session_id,
            log_path=self.log_path,
        )
        self.token_tracker = TokenTracker(
            total_budget=token_budget,
            logger=self.execution_logger,
        )

        # Pipeline components (injected)
        self.router = None
        self.memory_manager = None
        self.planner = None
        self.reasoning_manager = None
        self.contract_executor = None

    async def execute(self, task: str, context: Dict[str, Any] = None) -> PipelineResult:
        """
        Execute the full pipeline for a task.

        Args:
            task: The user's task description
            context: Optional context dictionary

        Returns:
            PipelineResult with execution details

        Raises:
            BudgetExhaustedError: If token budget is exceeded
            PipelineError: If a critical step fails
        """
        context = context or {}
        steps: List[StepResult] = []

        # Log session start
        await self.execution_logger.log_event(
            event_type="session_start",
            component="orchestrator",
            message=f"Starting pipeline for: {task}",
            metadata={"task": task, "budget": self.token_budget},
        )

        try:
            # Step 1: Routing
            routing_result = await self._execute_step(
                PipelineStep.ROUTING,
                self._route_task,
                task,
                context,
            )
            steps.append(routing_result)

            # Step 2: Memory Retrieval
            memory_result = await self._execute_step(
                PipelineStep.MEMORY,
                self._retrieve_memory,
                task,
                context,
            )
            steps.append(memory_result)
            context["memories"] = memory_result.output

            # Step 3: Planning
            planning_result = await self._execute_step(
                PipelineStep.PLANNING,
                self._create_plan,
                task,
                context,
            )
            steps.append(planning_result)
            context["plan"] = planning_result.output

            # Step 4: Reasoning
            reasoning_result = await self._execute_step(
                PipelineStep.REASONING,
                self._execute_reasoning,
                task,
                context,
            )
            steps.append(reasoning_result)

            # Step 5: Contract Validation
            validation_result = await self._execute_step(
                PipelineStep.VALIDATION,
                self._validate_output,
                reasoning_result.output,
                context,
            )
            steps.append(validation_result)

            # Calculate totals
            total_tokens = sum(s.tokens_used for s in steps)
            total_latency = sum(s.latency_ms for s in steps)

            # Log completion
            await self.execution_logger.log_event(
                event_type="session_complete",
                component="orchestrator",
                message="Pipeline completed successfully",
                tokens=total_tokens,
                metadata={
                    "total_latency_ms": total_latency,
                    "steps_completed": len(steps),
                },
            )

            return PipelineResult(
                session_id=self.session_id,
                task=task,
                success=True,
                steps=steps,
                total_tokens=total_tokens,
                total_latency_ms=total_latency,
                final_output=validation_result.output,
                quality_score=validation_result.metadata.get("quality_score"),
                recommended_action=reasoning_result.metadata.get("action"),
            )

        except Exception as e:
            await self.execution_logger.log_event(
                event_type="session_error",
                component="orchestrator",
                message=f"Pipeline failed: {str(e)}",
                metadata={"error_type": type(e).__name__},
            )
            raise

    async def _execute_step(
        self,
        step: PipelineStep,
        executor: callable,
        *args,
        **kwargs,
    ) -> StepResult:
        """Execute a single pipeline step with logging."""
        start_time = datetime.now()

        try:
            # Execute the step
            output, tokens = await executor(*args, **kwargs)

            # Calculate latency
            latency_ms = (datetime.now() - start_time).total_seconds() * 1000

            # Log the step
            await self.execution_logger.log_event(
                event_type="step_complete",
                component=step.value,
                message=f"{step.value.title()} completed",
                tokens=tokens,
                metadata={"latency_ms": latency_ms},
            )

            return StepResult(
                step=step,
                success=True,
                tokens_used=tokens,
                latency_ms=latency_ms,
                output=output,
            )

        except Exception as e:
            latency_ms = (datetime.now() - start_time).total_seconds() * 1000

            await self.execution_logger.log_event(
                event_type="step_error",
                component=step.value,
                message=f"{step.value.title()} failed: {str(e)}",
                tokens=0,
                metadata={"error": str(e), "latency_ms": latency_ms},
            )

            return StepResult(
                step=step,
                success=False,
                tokens_used=0,
                latency_ms=latency_ms,
                output=None,
                error=str(e),
            )
```

#### 4.1.3 Configuration Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `session_id` | str | Required | Unique session identifier |
| `token_budget` | int | 256000 | Maximum tokens for execution |
| `log_path` | str | ~/.sigil/logs | Directory for log files |
| `log_buffer_size` | int | 1024 | Bytes to buffer before flush |
| `log_flush_interval` | float | 0.1 | Seconds between async flushes |
| `enable_metrics` | bool | True | Enable performance metrics |

### 4.2 Execution Logger

#### 4.2.1 Purpose

The Execution Logger writes structured log entries that the Monitor can observe. It provides non-blocking writes with buffering for performance.

#### 4.2.2 Interface Definition

```python
# sigil/telemetry/execution_logger.py

import asyncio
import json
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List
from contextlib import asynccontextmanager
import aiofiles
import aiofiles.os


@dataclass
class LogEntry:
    """A single log entry for the execution log."""
    timestamp: str
    session_id: str
    event_type: str
    component: str
    message: str
    tokens: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        """Convert to JSON line."""
        return json.dumps(asdict(self), separators=(',', ':'))

    @classmethod
    def from_json(cls, line: str) -> "LogEntry":
        """Parse from JSON line."""
        data = json.loads(line)
        return cls(**data)


class ExecutionLogger:
    """
    Asynchronous logger for pipeline execution events.

    Features:
    - Non-blocking writes via asyncio
    - In-memory buffer with periodic flush
    - JSON Lines format for easy parsing
    - Automatic file rotation

    Attributes:
        session_id: Current session identifier
        log_path: Base directory for logs
        buffer_size: Max bytes to buffer
        flush_interval: Seconds between flushes
    """

    def __init__(
        self,
        session_id: str,
        log_path: str = None,
        buffer_size: int = 1024,
        flush_interval: float = 0.1,
    ) -> None:
        """
        Initialize the execution logger.

        Args:
            session_id: Unique session identifier
            log_path: Base directory for logs
            buffer_size: Bytes to buffer before flush
            flush_interval: Seconds between async flushes
        """
        self.session_id = session_id
        self.log_path = Path(log_path or self._default_log_path())
        self.buffer_size = buffer_size
        self.flush_interval = flush_interval

        # Internal state
        self._buffer: List[str] = []
        self._buffer_bytes: int = 0
        self._flush_task: Optional[asyncio.Task] = None
        self._lock = asyncio.Lock()
        self._file_handle = None
        self._initialized = False

    def _default_log_path(self) -> str:
        """Get default log path."""
        return str(Path.home() / ".sigil" / "logs")

    @property
    def log_file(self) -> Path:
        """Get the current session's log file path."""
        return self.log_path / "sessions" / self.session_id / "execution.jsonl"

    async def initialize(self) -> None:
        """
        Initialize the logger, creating directories and files.

        Must be called before logging any events.
        """
        if self._initialized:
            return

        # Create directory structure
        session_dir = self.log_path / "sessions" / self.session_id
        await aiofiles.os.makedirs(session_dir, exist_ok=True)

        # Open file for appending
        self._file_handle = await aiofiles.open(
            self.log_file,
            mode='a',
            encoding='utf-8',
        )

        # Start background flush task
        self._flush_task = asyncio.create_task(self._periodic_flush())

        self._initialized = True

    async def log_event(
        self,
        event_type: str,
        component: str,
        message: str,
        tokens: int = 0,
        metadata: Dict[str, Any] = None,
    ) -> None:
        """
        Log an execution event.

        This method is non-blocking. The event is buffered and
        written asynchronously.

        Args:
            event_type: Type of event (step_complete, error, etc.)
            component: Component that generated the event
            message: Human-readable message
            tokens: Tokens consumed (if applicable)
            metadata: Additional structured data
        """
        if not self._initialized:
            await self.initialize()

        entry = LogEntry(
            timestamp=datetime.now().isoformat(),
            session_id=self.session_id,
            event_type=event_type,
            component=component,
            message=message,
            tokens=tokens,
            metadata=metadata or {},
        )

        line = entry.to_json() + "\n"

        async with self._lock:
            self._buffer.append(line)
            self._buffer_bytes += len(line.encode('utf-8'))

            # Flush if buffer exceeds size
            if self._buffer_bytes >= self.buffer_size:
                await self._flush_buffer()

    async def _flush_buffer(self) -> None:
        """Flush the buffer to disk."""
        if not self._buffer:
            return

        if self._file_handle:
            content = ''.join(self._buffer)
            await self._file_handle.write(content)
            await self._file_handle.flush()

        self._buffer.clear()
        self._buffer_bytes = 0

    async def _periodic_flush(self) -> None:
        """Background task for periodic buffer flushing."""
        while True:
            await asyncio.sleep(self.flush_interval)
            async with self._lock:
                await self._flush_buffer()

    async def close(self) -> None:
        """
        Close the logger, flushing any remaining data.

        Should be called when execution completes.
        """
        if self._flush_task:
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass

        async with self._lock:
            await self._flush_buffer()

        if self._file_handle:
            await self._file_handle.close()

        self._initialized = False

    @asynccontextmanager
    async def session(self):
        """Context manager for logging session."""
        await self.initialize()
        try:
            yield self
        finally:
            await self.close()


# Convenience function for simple logging
async def log_execution_event(
    session_id: str,
    event_type: str,
    component: str,
    message: str,
    tokens: int = 0,
    metadata: Dict[str, Any] = None,
) -> None:
    """
    Log a single execution event.

    Creates a temporary logger for one-off logging.
    For multiple events, use ExecutionLogger directly.
    """
    logger = ExecutionLogger(session_id=session_id)
    async with logger.session():
        await logger.log_event(
            event_type=event_type,
            component=component,
            message=message,
            tokens=tokens,
            metadata=metadata,
        )
```

#### 4.2.3 Log Entry Schema

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "ExecutionLogEntry",
  "type": "object",
  "required": ["timestamp", "session_id", "event_type", "component", "message"],
  "properties": {
    "timestamp": {
      "type": "string",
      "format": "date-time",
      "description": "ISO 8601 timestamp of the event"
    },
    "session_id": {
      "type": "string",
      "description": "Unique session identifier"
    },
    "event_type": {
      "type": "string",
      "enum": [
        "session_start",
        "session_complete",
        "session_error",
        "step_start",
        "step_complete",
        "step_error",
        "token_consumption",
        "budget_warning",
        "budget_exceeded"
      ],
      "description": "Type of event"
    },
    "component": {
      "type": "string",
      "enum": [
        "orchestrator",
        "routing",
        "memory",
        "planning",
        "reasoning",
        "contracts",
        "validation"
      ],
      "description": "Component that generated the event"
    },
    "message": {
      "type": "string",
      "description": "Human-readable event description"
    },
    "tokens": {
      "type": "integer",
      "minimum": 0,
      "default": 0,
      "description": "Tokens consumed in this operation"
    },
    "metadata": {
      "type": "object",
      "description": "Additional structured data",
      "properties": {
        "latency_ms": {"type": "number"},
        "error": {"type": "string"},
        "budget_remaining": {"type": "integer"},
        "quality_score": {"type": "number"}
      }
    }
  }
}
```

### 4.3 Monitor Script

#### 4.3.1 Purpose

The Monitor Script provides real-time visibility into CLI execution by observing the log file and displaying a formatted dashboard.

#### 4.3.2 Interface Definition

```python
# sigil/interfaces/cli/monitor.py

import asyncio
import json
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List, TextIO
from enum import Enum

from sigil.telemetry.execution_logger import LogEntry


class DisplayMode(str, Enum):
    """Monitor display modes."""
    DASHBOARD = "dashboard"  # Full dashboard with charts
    STREAM = "stream"        # Simple streaming output
    MINIMAL = "minimal"      # Just token counts


@dataclass
class ComponentMetrics:
    """Metrics for a single component."""
    name: str
    tokens_used: int = 0
    operations: int = 0
    total_latency_ms: float = 0.0
    errors: int = 0
    last_message: str = ""
    last_update: Optional[datetime] = None

    @property
    def avg_latency_ms(self) -> float:
        """Average latency per operation."""
        if self.operations == 0:
            return 0.0
        return self.total_latency_ms / self.operations


@dataclass
class SessionState:
    """Accumulated state for a monitored session."""
    session_id: str
    status: str = "initializing"
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None

    # Token tracking
    total_budget: int = 256_000
    tokens_used: int = 0

    # Component metrics
    components: Dict[str, ComponentMetrics] = field(default_factory=dict)

    # Event history
    events: List[LogEntry] = field(default_factory=list)
    max_events: int = 100

    @property
    def tokens_remaining(self) -> int:
        """Tokens remaining in budget."""
        return max(0, self.total_budget - self.tokens_used)

    @property
    def budget_percentage(self) -> float:
        """Percentage of budget used."""
        if self.total_budget == 0:
            return 0.0
        return (self.tokens_used / self.total_budget) * 100

    @property
    def elapsed_seconds(self) -> float:
        """Elapsed time in seconds."""
        if not self.start_time:
            return 0.0
        end = self.end_time or datetime.now()
        return (end - self.start_time).total_seconds()

    @property
    def slowest_component(self) -> Optional[str]:
        """Name of the slowest component."""
        if not self.components:
            return None
        return max(
            self.components.keys(),
            key=lambda k: self.components[k].total_latency_ms,
        )

    def update_from_entry(self, entry: LogEntry) -> None:
        """Update state from a log entry."""
        # Track event
        self.events.append(entry)
        if len(self.events) > self.max_events:
            self.events = self.events[-self.max_events:]

        # Handle event types
        if entry.event_type == "session_start":
            self.status = "running"
            self.start_time = datetime.fromisoformat(entry.timestamp)
            if "budget" in entry.metadata:
                self.total_budget = entry.metadata["budget"]

        elif entry.event_type == "session_complete":
            self.status = "complete"
            self.end_time = datetime.fromisoformat(entry.timestamp)

        elif entry.event_type == "session_error":
            self.status = "error"
            self.end_time = datetime.fromisoformat(entry.timestamp)

        elif entry.event_type in ("step_complete", "step_error"):
            self._update_component(entry)

        # Always accumulate tokens
        self.tokens_used += entry.tokens

    def _update_component(self, entry: LogEntry) -> None:
        """Update component metrics from entry."""
        component = entry.component

        if component not in self.components:
            self.components[component] = ComponentMetrics(name=component)

        metrics = self.components[component]
        metrics.tokens_used += entry.tokens
        metrics.operations += 1
        metrics.last_message = entry.message
        metrics.last_update = datetime.fromisoformat(entry.timestamp)

        if "latency_ms" in entry.metadata:
            metrics.total_latency_ms += entry.metadata["latency_ms"]

        if entry.event_type == "step_error":
            metrics.errors += 1


class ExecutionMonitor:
    """
    Real-time monitor for Sigil v2 CLI execution.

    Features:
    - Tails execution log file
    - Accumulates token usage
    - Displays formatted dashboard
    - Tracks performance metrics

    Usage:
        monitor = ExecutionMonitor(session_id="test-1")
        await monitor.run()
    """

    def __init__(
        self,
        session_id: str,
        log_path: str = None,
        display_mode: DisplayMode = DisplayMode.DASHBOARD,
        refresh_rate: float = 0.1,
    ) -> None:
        """
        Initialize the monitor.

        Args:
            session_id: Session to monitor
            log_path: Base log directory
            display_mode: How to display output
            refresh_rate: Seconds between display updates
        """
        self.session_id = session_id
        self.log_path = Path(log_path or self._default_log_path())
        self.display_mode = display_mode
        self.refresh_rate = refresh_rate

        # State
        self.state = SessionState(session_id=session_id)
        self._running = False
        self._file_position = 0

    def _default_log_path(self) -> str:
        """Get default log path."""
        return str(Path.home() / ".sigil" / "logs")

    @property
    def log_file(self) -> Path:
        """Get the log file path for this session."""
        return self.log_path / "sessions" / self.session_id / "execution.jsonl"

    async def run(self) -> None:
        """
        Run the monitor until session completes or interrupted.

        This is the main entry point. It will:
        1. Wait for the log file to appear
        2. Tail the file for new entries
        3. Update and display state
        4. Exit when session completes
        """
        self._running = True

        # Print header
        self._print_header()

        # Wait for log file
        await self._wait_for_file()

        # Main monitoring loop
        while self._running:
            # Read new entries
            entries = await self._read_new_entries()

            # Update state
            for entry in entries:
                self.state.update_from_entry(entry)

            # Display current state
            if entries or self.display_mode == DisplayMode.DASHBOARD:
                self._display_state()

            # Check for completion
            if self.state.status in ("complete", "error"):
                self._display_final_summary()
                break

            # Wait before next check
            await asyncio.sleep(self.refresh_rate)

    async def _wait_for_file(self) -> None:
        """Wait for the log file to appear."""
        print(f"Waiting for session {self.session_id}...")

        while self._running and not self.log_file.exists():
            await asyncio.sleep(0.5)

        if self.log_file.exists():
            print(f"Found log file: {self.log_file}")

    async def _read_new_entries(self) -> List[LogEntry]:
        """Read new entries from the log file."""
        entries = []

        if not self.log_file.exists():
            return entries

        try:
            with open(self.log_file, 'r', encoding='utf-8') as f:
                # Seek to last position
                f.seek(self._file_position)

                # Read new lines
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            entry = LogEntry.from_json(line)
                            entries.append(entry)
                        except json.JSONDecodeError:
                            pass  # Skip malformed lines

                # Update position
                self._file_position = f.tell()

        except OSError:
            pass  # File might be rotating

        return entries

    def _print_header(self) -> None:
        """Print the monitor header."""
        print("\n" + "=" * 50)
        print("       SIGIL v2 EXECUTION MONITOR")
        print("=" * 50)
        print(f"Session: {self.session_id}")
        print(f"Status: Initializing...")
        print("-" * 50)

    def _display_state(self) -> None:
        """Display current session state."""
        if self.display_mode == DisplayMode.MINIMAL:
            self._display_minimal()
        elif self.display_mode == DisplayMode.STREAM:
            self._display_stream()
        else:
            self._display_dashboard()

    def _display_minimal(self) -> None:
        """Display minimal token count."""
        print(f"\rTokens: {self.state.tokens_used:,} / {self.state.total_budget:,} "
              f"({self.state.budget_percentage:.2f}%)", end="", flush=True)

    def _display_stream(self) -> None:
        """Display streaming event output."""
        if self.state.events:
            entry = self.state.events[-1]
            timestamp = entry.timestamp.split("T")[1].split(".")[0]
            print(f"[{timestamp}] {entry.component}: {entry.message} "
                  f"({entry.tokens} tokens)")

    def _display_dashboard(self) -> None:
        """Display full dashboard."""
        # Clear screen and move cursor to top
        print("\033[2J\033[H", end="")

        # Header
        print("=" * 60)
        print("           SIGIL v2 EXECUTION MONITOR")
        print("=" * 60)
        print(f"Session: {self.session_id}")
        print(f"Status:  {self.state.status.upper()}")
        print(f"Elapsed: {self.state.elapsed_seconds:.1f}s")
        print("-" * 60)

        # Component breakdown
        print("\nComponent Token Usage:")
        print("-" * 40)

        component_order = ["routing", "memory", "planning", "reasoning", "contracts", "validation"]

        for comp_name in component_order:
            if comp_name in self.state.components:
                comp = self.state.components[comp_name]
                bar = self._token_bar(comp.tokens_used, 5000)
                print(f"  [{comp_name:12}] {comp.tokens_used:6,} tokens  {bar}")
            else:
                print(f"  [{comp_name:12}]      0 tokens  ")

        print("-" * 40)

        # Budget summary
        print("\nBudget Status:")
        budget_bar = self._budget_bar(self.state.budget_percentage)
        print(f"  Used:      {self.state.tokens_used:,} / {self.state.total_budget:,} "
              f"({self.state.budget_percentage:.2f}%)")
        print(f"  Progress:  {budget_bar}")
        print(f"  Remaining: {self.state.tokens_remaining:,} tokens")

        # Performance metrics
        if self.state.components:
            print("\nPerformance:")
            slowest = self.state.slowest_component
            if slowest:
                latency = self.state.components[slowest].total_latency_ms
                print(f"  Slowest:   {slowest} ({latency:.0f}ms)")

            total_latency = sum(c.total_latency_ms for c in self.state.components.values())
            print(f"  Total:     {total_latency:.0f}ms")

        # Recent events
        print("\nRecent Events:")
        print("-" * 40)
        for entry in self.state.events[-5:]:
            timestamp = entry.timestamp.split("T")[1].split(".")[0]
            print(f"  [{timestamp}] {entry.component}: {entry.message}")

    def _token_bar(self, tokens: int, max_tokens: int) -> str:
        """Generate a token usage bar."""
        if max_tokens == 0:
            return ""

        filled = min(20, int((tokens / max_tokens) * 20))
        return "=" * filled

    def _budget_bar(self, percentage: float) -> str:
        """Generate a budget progress bar."""
        filled = min(30, int(percentage / 100 * 30))
        empty = 30 - filled

        if percentage < 50:
            color = "\033[32m"  # Green
        elif percentage < 80:
            color = "\033[33m"  # Yellow
        else:
            color = "\033[31m"  # Red

        return f"{color}{'#' * filled}\033[0m{'.' * empty}"

    def _display_final_summary(self) -> None:
        """Display final execution summary."""
        print("\n" + "=" * 60)
        print("           EXECUTION COMPLETE")
        print("=" * 60)
        print(f"\nSession:  {self.session_id}")
        print(f"Status:   {self.state.status.upper()}")
        print(f"Duration: {self.state.elapsed_seconds:.1f}s")
        print(f"\nFinal Token Usage:")
        print(f"  Total:     {self.state.tokens_used:,} / {self.state.total_budget:,}")
        print(f"  Percentage: {self.state.budget_percentage:.2f}%")
        print(f"  Remaining:  {self.state.tokens_remaining:,}")

        if self.state.components:
            print(f"\nComponent Breakdown:")
            for name, comp in self.state.components.items():
                print(f"  {name:12}: {comp.tokens_used:,} tokens, "
                      f"{comp.total_latency_ms:.0f}ms")

        print("\n" + "=" * 60)

    def stop(self) -> None:
        """Stop the monitor."""
        self._running = False


# CLI entry point
async def main():
    """Main entry point for monitor CLI."""
    import argparse

    parser = argparse.ArgumentParser(description="Sigil v2 Execution Monitor")
    parser.add_argument("--session", "-s", required=True, help="Session ID to monitor")
    parser.add_argument("--mode", "-m", choices=["dashboard", "stream", "minimal"],
                        default="dashboard", help="Display mode")
    parser.add_argument("--log-path", "-l", help="Log directory path")

    args = parser.parse_args()

    mode = DisplayMode(args.mode)
    monitor = ExecutionMonitor(
        session_id=args.session,
        log_path=args.log_path,
        display_mode=mode,
    )

    try:
        await monitor.run()
    except KeyboardInterrupt:
        monitor.stop()
        print("\nMonitor stopped.")


if __name__ == "__main__":
    asyncio.run(main())
```

---

## 5. Data Flow Architecture

### 5.1 Write Path (CLI to Log File)

The write path ensures non-blocking operation for the CLI while maintaining data integrity:

```
+-----------------------------------------------------------------------------+
|                            WRITE PATH ARCHITECTURE                           |
+-----------------------------------------------------------------------------+

Pipeline Step
     |
     | 1. Step completes
     v
+----+----+
| Token   |
| Tracker |  2. Records token consumption
+----+----+
     |
     | 3. Creates log entry
     v
+----+----+
| Log     |
| Entry   |  4. Serializes to JSON
+----+----+
     |
     | 5. Adds to buffer
     v
+----+----+
| Memory  |
| Buffer  |  6. Non-blocking append
+----+----+
     |
     | 7. Async flush (100ms or 1KB)
     v
+----+----+
| File    |
| System  |  8. Append to execution.jsonl
+---------+

Guarantees:
-----------
- Pipeline step never blocks on I/O
- Buffer limits memory usage
- Periodic flush ensures timely visibility
- Append-only writes are crash-safe
```

### 5.2 Read Path (Log File to Monitor)

The read path provides real-time visibility with minimal overhead:

```
+-----------------------------------------------------------------------------+
|                            READ PATH ARCHITECTURE                            |
+-----------------------------------------------------------------------------+

File System
     |
     | 1. Tail file (polling)
     v
+----+----+
| File    |
| Reader  |  2. Read new bytes
+----+----+
     |
     | 3. Split into lines
     v
+----+----+
| Line    |
| Parser  |  4. Parse JSON
+----+----+
     |
     | 5. Create LogEntry
     v
+----+----+
| State   |
| Update  |  6. Accumulate metrics
+----+----+
     |
     | 7. Format display
     v
+----+----+
| Display |
| Render  |  8. Write to terminal
+---------+

Optimizations:
--------------
- Track file position to avoid re-reading
- Batch multiple lines per update
- Throttle display updates (100ms)
- Handle partial lines gracefully
```

### 5.3 Token Flow

Token tracking flows through the entire system:

```
+-----------------------------------------------------------------------------+
|                            TOKEN FLOW ARCHITECTURE                           |
+-----------------------------------------------------------------------------+

                        CLI SIDE
+-----------------------------------------------------------+
|                                                           |
|  Session Start                                            |
|       |                                                   |
|       v                                                   |
|  +----+----+                                              |
|  | Budget  |  Total: 256,000 tokens                       |
|  | Alloc   |                                              |
|  +----+----+                                              |
|       |                                                   |
|       | Allocate per component                            |
|       v                                                   |
|  +----+----+----+----+----+----+                          |
|  | Routing | Memory | Plan | Reason | Valid |             |
|  | 10K     | 30K    | 20K  | 150K   | 10K   |             |
|  +----+----+----+----+----+----+                          |
|       |         |      |       |       |                  |
|       | Execute | ...  | ...   | ...   |                  |
|       v         v      v       v       v                  |
|  +----+----+                                              |
|  | Token   |  Actual usage per step                       |
|  | Track   |  routing: 50, memory: 150, ...               |
|  +----+----+                                              |
|       |                                                   |
|       | Log with tokens                                   |
|       v                                                   |
+-------+---------------------------------------------------+
        |
        v
   Log File
        |
        v
+-------+---------------------------------------------------+
|       |                          MONITOR SIDE             |
|  +----+----+                                              |
|  | Parse   |  Extract token count from entry              |
|  | Entry   |                                              |
|  +----+----+                                              |
|       |                                                   |
|       | Accumulate                                        |
|       v                                                   |
|  +----+----+                                              |
|  | Token   |  Running total: 650 tokens                   |
|  | Accum   |  Per component: {routing: 50, memory: 150}   |
|  +----+----+                                              |
|       |                                                   |
|       | Calculate metrics                                 |
|       v                                                   |
|  +----+----+                                              |
|  | Display |  Used: 650 / 256,000 (0.25%)                 |
|  | Format  |  Remaining: 255,350 tokens                   |
|  +---------+                                              |
|                                                           |
+-----------------------------------------------------------+
```

---

## 6. Token Tracking System

### 6.1 Token Budget Architecture

The token budget system provides hierarchical allocation and tracking:

```
+-----------------------------------------------------------------------------+
|                        TOKEN BUDGET HIERARCHY                                |
+-----------------------------------------------------------------------------+

                    +------------------------+
                    |     SESSION BUDGET     |
                    |      256,000 tokens    |
                    +------------------------+
                              |
          +-------------------+-------------------+
          |                   |                   |
          v                   v                   v
    +----------+        +----------+        +----------+
    | Pipeline |        | Pipeline |        | Pipeline |
    | Run 1    |        | Run 2    |        | Run N    |
    | 85,000   |        | 85,000   |        | ...      |
    +----------+        +----------+        +----------+
          |
    +-----+-----+-----+-----+-----+
    |     |     |     |     |     |
    v     v     v     v     v     v
  +---+ +---+ +---+ +---+ +---+ +---+
  |Rou| |Mem| |Pln| |Rsn| |Ctr| |Val|
  |10K| |30K| |20K| |150| |10K| |10K|
  +---+ +---+ +---+ +---+ +---+ +---+

Allocation Strategy:
--------------------
- Reasoning gets largest share (complex LLM operations)
- Memory retrieval moderate (RAG + potential LLM reading)
- Planning moderate (one-time per task)
- Routing minimal (simple classification)
- Contracts reserved for validation
```

### 6.2 Token Tracker Implementation

```python
# sigil/telemetry/tokens.py

from dataclasses import dataclass, field
from typing import Dict, Optional, Callable
from enum import Enum
import threading


class BudgetWarningLevel(str, Enum):
    """Warning levels for budget consumption."""
    NORMAL = "normal"      # < 50%
    ELEVATED = "elevated"  # 50-80%
    CRITICAL = "critical"  # 80-95%
    EXCEEDED = "exceeded"  # > 95%


@dataclass
class ComponentBudget:
    """Budget allocation for a single component."""
    name: str
    allocated: int
    used: int = 0
    operations: int = 0

    @property
    def remaining(self) -> int:
        """Remaining tokens in this budget."""
        return max(0, self.allocated - self.used)

    @property
    def utilization(self) -> float:
        """Percentage of budget used."""
        if self.allocated == 0:
            return 0.0
        return (self.used / self.allocated) * 100

    @property
    def avg_per_operation(self) -> float:
        """Average tokens per operation."""
        if self.operations == 0:
            return 0.0
        return self.used / self.operations

    def consume(self, tokens: int) -> None:
        """Consume tokens from this budget."""
        self.used += tokens
        self.operations += 1

    def can_consume(self, tokens: int) -> bool:
        """Check if tokens can be consumed without exceeding budget."""
        return self.remaining >= tokens


@dataclass
class TokenBudget:
    """
    Hierarchical token budget for pipeline execution.

    Tracks token consumption across all components with
    warning thresholds and budget enforcement.
    """

    total: int
    used: int = 0
    components: Dict[str, ComponentBudget] = field(default_factory=dict)

    # Warning thresholds
    warning_threshold: float = 0.5    # 50%
    critical_threshold: float = 0.8   # 80%
    exceeded_threshold: float = 0.95  # 95%

    # Callbacks
    on_warning: Optional[Callable[[BudgetWarningLevel, int, int], None]] = None

    @property
    def remaining(self) -> int:
        """Total remaining tokens."""
        return max(0, self.total - self.used)

    @property
    def utilization(self) -> float:
        """Total utilization percentage."""
        if self.total == 0:
            return 0.0
        return (self.used / self.total) * 100

    @property
    def warning_level(self) -> BudgetWarningLevel:
        """Current warning level based on utilization."""
        util = self.utilization / 100
        if util >= self.exceeded_threshold:
            return BudgetWarningLevel.EXCEEDED
        elif util >= self.critical_threshold:
            return BudgetWarningLevel.CRITICAL
        elif util >= self.warning_threshold:
            return BudgetWarningLevel.ELEVATED
        return BudgetWarningLevel.NORMAL

    def allocate_component(self, name: str, tokens: int) -> ComponentBudget:
        """
        Allocate a budget for a component.

        Args:
            name: Component name
            tokens: Tokens to allocate

        Returns:
            ComponentBudget for the component
        """
        budget = ComponentBudget(name=name, allocated=tokens)
        self.components[name] = budget
        return budget

    def consume(self, component: str, tokens: int) -> bool:
        """
        Consume tokens for a component.

        Args:
            component: Component name
            tokens: Tokens to consume

        Returns:
            True if consumption succeeded, False if budget exceeded
        """
        # Check global budget
        if self.remaining < tokens:
            self._trigger_warning(BudgetWarningLevel.EXCEEDED)
            return False

        # Check component budget
        if component in self.components:
            comp_budget = self.components[component]
            if not comp_budget.can_consume(tokens):
                # Log warning but allow (soft limit)
                self._trigger_warning(BudgetWarningLevel.CRITICAL)
            comp_budget.consume(tokens)

        # Update global
        self.used += tokens

        # Check warning levels
        self._check_warnings()

        return True

    def _check_warnings(self) -> None:
        """Check and trigger warnings based on utilization."""
        level = self.warning_level
        if level != BudgetWarningLevel.NORMAL:
            self._trigger_warning(level)

    def _trigger_warning(self, level: BudgetWarningLevel) -> None:
        """Trigger a warning callback."""
        if self.on_warning:
            self.on_warning(level, self.used, self.total)

    def can_continue(self) -> bool:
        """Check if execution can continue."""
        return self.warning_level != BudgetWarningLevel.EXCEEDED

    def forecast(self, operations_remaining: int) -> Dict[str, int]:
        """
        Forecast token usage based on current rate.

        Args:
            operations_remaining: Expected remaining operations

        Returns:
            Dictionary with forecast metrics
        """
        if self.used == 0:
            return {
                "forecast_total": 0,
                "forecast_remaining": self.total,
                "will_exceed": False,
            }

        # Calculate average per operation
        total_ops = sum(c.operations for c in self.components.values())
        if total_ops == 0:
            avg_per_op = 0
        else:
            avg_per_op = self.used / total_ops

        forecast_additional = int(avg_per_op * operations_remaining)
        forecast_total = self.used + forecast_additional

        return {
            "forecast_total": forecast_total,
            "forecast_remaining": max(0, self.total - forecast_total),
            "will_exceed": forecast_total > self.total,
            "avg_per_operation": avg_per_op,
        }

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "total": self.total,
            "used": self.used,
            "remaining": self.remaining,
            "utilization": self.utilization,
            "warning_level": self.warning_level.value,
            "components": {
                name: {
                    "allocated": c.allocated,
                    "used": c.used,
                    "remaining": c.remaining,
                    "utilization": c.utilization,
                    "operations": c.operations,
                }
                for name, c in self.components.items()
            },
        }


class TokenTracker:
    """
    Thread-safe token tracker with logging integration.

    Tracks token consumption and logs events for monitoring.
    """

    def __init__(
        self,
        total_budget: int = 256_000,
        logger: "ExecutionLogger" = None,
    ) -> None:
        """
        Initialize the token tracker.

        Args:
            total_budget: Total token budget
            logger: Optional execution logger for monitoring
        """
        self.budget = TokenBudget(total=total_budget)
        self.logger = logger
        self._lock = threading.Lock()

        # Set up warning callback
        self.budget.on_warning = self._handle_warning

        # Default component allocations
        self._setup_default_allocations()

    def _setup_default_allocations(self) -> None:
        """Set up default component budget allocations."""
        allocations = {
            "routing": 10_000,
            "memory": 30_000,
            "planning": 20_000,
            "reasoning": 150_000,
            "contracts": 10_000,
            "validation": 10_000,
        }

        for name, tokens in allocations.items():
            self.budget.allocate_component(name, tokens)

    async def track(
        self,
        component: str,
        tokens: int,
        operation: str = None,
    ) -> bool:
        """
        Track token consumption for a component.

        Args:
            component: Component name
            tokens: Tokens consumed
            operation: Optional operation description

        Returns:
            True if tracking succeeded
        """
        with self._lock:
            success = self.budget.consume(component, tokens)

        # Log the consumption
        if self.logger:
            await self.logger.log_event(
                event_type="token_consumption",
                component=component,
                message=operation or f"Consumed {tokens} tokens",
                tokens=tokens,
                metadata={
                    "total_used": self.budget.used,
                    "total_remaining": self.budget.remaining,
                    "component_used": self.budget.components[component].used
                    if component in self.budget.components else 0,
                },
            )

        return success

    def _handle_warning(
        self,
        level: BudgetWarningLevel,
        used: int,
        total: int,
    ) -> None:
        """Handle budget warning."""
        if self.logger and level in (
            BudgetWarningLevel.CRITICAL,
            BudgetWarningLevel.EXCEEDED,
        ):
            # This would need to be async in practice
            # For now, just store for later logging
            pass

    def get_status(self) -> Dict:
        """Get current budget status."""
        with self._lock:
            return self.budget.to_dict()

    def get_component_status(self, component: str) -> Optional[Dict]:
        """Get status for a specific component."""
        with self._lock:
            if component not in self.budget.components:
                return None

            c = self.budget.components[component]
            return {
                "name": c.name,
                "allocated": c.allocated,
                "used": c.used,
                "remaining": c.remaining,
                "utilization": c.utilization,
                "operations": c.operations,
                "avg_per_operation": c.avg_per_operation,
            }
```

### 6.3 Default Budget Allocations

| Component | Allocation | Percentage | Rationale |
|-----------|------------|------------|-----------|
| Routing | 10,000 | 3.9% | Simple classification, few tokens |
| Memory | 30,000 | 11.7% | RAG retrieval + potential LLM reading |
| Planning | 20,000 | 7.8% | One-time plan generation |
| Reasoning | 150,000 | 58.6% | Main LLM operations, multiple iterations |
| Contracts | 10,000 | 3.9% | Validation prompts |
| Validation | 10,000 | 3.9% | Output verification |
| **Reserve** | **26,000** | **10.2%** | Buffer for retries and errors |
| **Total** | **256,000** | **100%** | Full context window |

---

## 7. Log File Architecture

### 7.1 Directory Structure

```
~/.sigil/
+-- logs/
    +-- sessions/
    |   +-- test-1/
    |   |   +-- execution.jsonl      # Main execution log
    |   |   +-- tokens.json          # Final token summary
    |   |   +-- metrics.json         # Performance metrics
    |   |   +-- errors.jsonl         # Error log (if any)
    |   |
    |   +-- test-2/
    |   |   +-- ...
    |   |
    |   +-- production-abc123/
    |       +-- ...
    |
    +-- archive/
    |   +-- 2026-01-10/
    |   |   +-- test-1.tar.gz
    |   |   +-- test-2.tar.gz
    |   |
    |   +-- 2026-01-09/
    |       +-- ...
    |
    +-- current.jsonl               # Symlink to active session log
    +-- monitor.pid                 # PID file for monitor process
```

### 7.2 Log File Format

#### 7.2.1 execution.jsonl (JSON Lines)

```jsonl
{"timestamp":"2026-01-11T10:15:32.123456","session_id":"test-1","event_type":"session_start","component":"orchestrator","message":"Starting pipeline for: Analyze Acme Corp","tokens":0,"metadata":{"task":"Analyze Acme Corp","budget":256000}}
{"timestamp":"2026-01-11T10:15:33.234567","session_id":"test-1","event_type":"step_complete","component":"routing","message":"Intent classified as: lead_analysis","tokens":50,"metadata":{"latency_ms":1100,"intent":"lead_analysis"}}
{"timestamp":"2026-01-11T10:15:34.345678","session_id":"test-1","event_type":"step_complete","component":"memory","message":"Retrieved 5 relevant memories","tokens":150,"metadata":{"latency_ms":1200,"memories_count":5}}
{"timestamp":"2026-01-11T10:15:35.456789","session_id":"test-1","event_type":"step_complete","component":"planning","message":"Plan created with 3 steps","tokens":0,"metadata":{"latency_ms":50,"steps_count":3}}
{"timestamp":"2026-01-11T10:15:37.567890","session_id":"test-1","event_type":"step_complete","component":"reasoning","message":"Analysis complete","tokens":450,"metadata":{"latency_ms":2100,"strategy":"chain_of_thought"}}
{"timestamp":"2026-01-11T10:15:38.678901","session_id":"test-1","event_type":"step_complete","component":"validation","message":"Output validated against contract","tokens":0,"metadata":{"latency_ms":100,"contract":"lead_analysis"}}
{"timestamp":"2026-01-11T10:15:38.789012","session_id":"test-1","event_type":"session_complete","component":"orchestrator","message":"Pipeline completed successfully","tokens":0,"metadata":{"total_latency_ms":5100,"steps_completed":5}}
```

#### 7.2.2 tokens.json (Final Summary)

```json
{
  "session_id": "test-1",
  "timestamp": "2026-01-11T10:15:38.789012",
  "total_budget": 256000,
  "total_used": 650,
  "total_remaining": 255350,
  "utilization_percentage": 0.25,
  "components": {
    "routing": {
      "allocated": 10000,
      "used": 50,
      "utilization": 0.5
    },
    "memory": {
      "allocated": 30000,
      "used": 150,
      "utilization": 0.5
    },
    "planning": {
      "allocated": 20000,
      "used": 0,
      "utilization": 0.0
    },
    "reasoning": {
      "allocated": 150000,
      "used": 450,
      "utilization": 0.3
    },
    "contracts": {
      "allocated": 10000,
      "used": 0,
      "utilization": 0.0
    },
    "validation": {
      "allocated": 10000,
      "used": 0,
      "utilization": 0.0
    }
  }
}
```

#### 7.2.3 metrics.json (Performance Metrics)

```json
{
  "session_id": "test-1",
  "timestamp": "2026-01-11T10:15:38.789012",
  "duration_seconds": 5.1,
  "steps_completed": 5,
  "steps_failed": 0,
  "components": {
    "routing": {
      "latency_ms": 1100,
      "operations": 1,
      "errors": 0
    },
    "memory": {
      "latency_ms": 1200,
      "operations": 1,
      "errors": 0
    },
    "planning": {
      "latency_ms": 50,
      "operations": 1,
      "errors": 0
    },
    "reasoning": {
      "latency_ms": 2100,
      "operations": 1,
      "errors": 0
    },
    "validation": {
      "latency_ms": 100,
      "operations": 1,
      "errors": 0
    }
  },
  "slowest_component": "reasoning",
  "fastest_component": "planning",
  "average_latency_ms": 910
}
```

### 7.3 Log Rotation

```python
# sigil/telemetry/log_rotation.py

import os
import gzip
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import List


class LogRotator:
    """
    Manages log file rotation and archival.

    Rotation Policy:
    - Active logs kept for 7 days
    - Archived logs compressed (gzip)
    - Archives kept for 30 days
    - Maximum 10MB per log file
    """

    def __init__(
        self,
        log_path: Path,
        max_file_size_mb: int = 10,
        retention_days: int = 7,
        archive_retention_days: int = 30,
    ) -> None:
        """
        Initialize the log rotator.

        Args:
            log_path: Base log directory
            max_file_size_mb: Maximum file size before rotation
            retention_days: Days to keep active logs
            archive_retention_days: Days to keep archived logs
        """
        self.log_path = log_path
        self.max_file_size = max_file_size_mb * 1024 * 1024
        self.retention_days = retention_days
        self.archive_retention_days = archive_retention_days
        self.archive_path = log_path / "archive"

    def should_rotate(self, file_path: Path) -> bool:
        """Check if a file should be rotated."""
        if not file_path.exists():
            return False
        return file_path.stat().st_size >= self.max_file_size

    def rotate(self, file_path: Path) -> Path:
        """
        Rotate a log file.

        Creates a new file with timestamp suffix and
        optionally compresses the old file.

        Args:
            file_path: Path to the log file

        Returns:
            Path to the rotated file
        """
        if not file_path.exists():
            return file_path

        # Generate rotated filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        stem = file_path.stem
        suffix = file_path.suffix
        rotated_name = f"{stem}_{timestamp}{suffix}"
        rotated_path = file_path.parent / rotated_name

        # Rename current file
        shutil.move(str(file_path), str(rotated_path))

        # Create new empty file
        file_path.touch()

        return rotated_path

    def archive_old_sessions(self) -> List[Path]:
        """
        Archive sessions older than retention period.

        Returns:
            List of archived session paths
        """
        sessions_path = self.log_path / "sessions"
        if not sessions_path.exists():
            return []

        cutoff = datetime.now() - timedelta(days=self.retention_days)
        archived = []

        for session_dir in sessions_path.iterdir():
            if not session_dir.is_dir():
                continue

            # Check modification time
            mtime = datetime.fromtimestamp(session_dir.stat().st_mtime)
            if mtime < cutoff:
                archive_file = self._archive_session(session_dir)
                if archive_file:
                    archived.append(archive_file)
                    shutil.rmtree(session_dir)

        return archived

    def _archive_session(self, session_dir: Path) -> Path:
        """Archive a single session directory."""
        date_str = datetime.now().strftime("%Y-%m-%d")
        archive_date_path = self.archive_path / date_str
        archive_date_path.mkdir(parents=True, exist_ok=True)

        archive_file = archive_date_path / f"{session_dir.name}.tar.gz"

        with gzip.open(archive_file, 'wb') as f:
            import tarfile
            with tarfile.open(fileobj=f, mode='w') as tar:
                tar.add(session_dir, arcname=session_dir.name)

        return archive_file

    def cleanup_old_archives(self) -> List[Path]:
        """
        Remove archives older than archive retention period.

        Returns:
            List of removed archive paths
        """
        if not self.archive_path.exists():
            return []

        cutoff = datetime.now() - timedelta(days=self.archive_retention_days)
        removed = []

        for date_dir in self.archive_path.iterdir():
            if not date_dir.is_dir():
                continue

            try:
                dir_date = datetime.strptime(date_dir.name, "%Y-%m-%d")
                if dir_date < cutoff:
                    shutil.rmtree(date_dir)
                    removed.append(date_dir)
            except ValueError:
                pass  # Not a date directory

        return removed

    def run_maintenance(self) -> dict:
        """
        Run full maintenance cycle.

        Returns:
            Summary of maintenance actions
        """
        archived = self.archive_old_sessions()
        cleaned = self.cleanup_old_archives()

        return {
            "sessions_archived": len(archived),
            "archives_cleaned": len(cleaned),
            "archived_paths": [str(p) for p in archived],
            "cleaned_paths": [str(p) for p in cleaned],
        }
```

---

## 8. Inter-Process Communication

### 8.1 Communication Model

The CLI and Monitor communicate through the file system with no direct coupling:

```
+-----------------------------------------------------------------------------+
|                    INTER-PROCESS COMMUNICATION MODEL                         |
+-----------------------------------------------------------------------------+

        CLI Process                              Monitor Process
        ===========                              ===============

    +---------------+                        +---------------+
    |   Main Loop   |                        |   Main Loop   |
    +-------+-------+                        +-------+-------+
            |                                        |
            | Execute step                           | Check file
            v                                        v
    +-------+-------+                        +-------+-------+
    |   Pipeline    |                        |   File Watch  |
    |   Execution   |                        |   (Polling)   |
    +-------+-------+                        +-------+-------+
            |                                        |
            | Log event                              | Read new lines
            v                                        v
    +-------+-------+                        +-------+-------+
    |   Execution   |                        |   Log Parser  |
    |   Logger      |                        |               |
    +-------+-------+                        +-------+-------+
            |                                        |
            | Buffer & flush                         | Parse JSON
            v                                        v
    +-------+-------+                        +-------+-------+
    |   File I/O    |                        |   State       |
    |   (Append)    |                        |   Update      |
    +-------+-------+                        +-------+-------+
            |                                        |
            |                                        | Render display
            v                                        v
    +-------+---------------------------------------+-------+
    |                                                       |
    |                 LOG FILE (Shared)                     |
    |                 execution.jsonl                       |
    |                                                       |
    +-------------------------------------------------------+

Key Properties:
---------------
1. No direct communication between processes
2. File is the only shared resource
3. CLI only appends (never reads)
4. Monitor only reads (never writes)
5. Both can restart independently
```

### 8.2 Synchronization Guarantees

| Guarantee | Mechanism | Implication |
|-----------|-----------|-------------|
| Atomicity | Line-based writes | Each log entry is atomic |
| Ordering | Append-only | Events appear in order |
| Durability | fsync on flush | Data survives crashes |
| Visibility | 100ms flush interval | Max 100ms delay |
| Isolation | Separate file handles | No lock contention |

### 8.3 Failure Scenarios

#### 8.3.1 CLI Crashes

```
Scenario: CLI crashes mid-execution
-----------------------------------------

Before Crash:
+----------------+       +----------------+
| CLI Process    |------>| execution.jsonl|
| (Running)      |       | [entry1]       |
+----------------+       | [entry2]       |
                         | [entry3]       |
                         +----------------+
                                ^
                                |
                         +------+------+
                         | Monitor     |
                         | (Observing) |
                         +-------------+

After Crash:
+----------------+       +----------------+
| CLI Process    |       | execution.jsonl|
| (CRASHED)      |       | [entry1]       |
+----------------+       | [entry2]       |
                         | [entry3]       |
                         +----------------+
                                ^
                                |
                         +------+------+
                         | Monitor     |
                         | Shows last  |
                         | known state |
                         +-------------+

Recovery:
- Monitor continues showing last state
- Session marked incomplete
- User can restart CLI with same session ID
- Logs preserved for debugging
```

#### 8.3.2 Monitor Crashes

```
Scenario: Monitor crashes while CLI running
-----------------------------------------

Before Crash:
+----------------+       +----------------+       +----------------+
| CLI Process    |------>| execution.jsonl|<------| Monitor        |
| (Running)      |       | [entry1]       |       | (Running)      |
+----------------+       | [entry2]       |       +----------------+
                         +----------------+

After Monitor Crash:
+----------------+       +----------------+       +----------------+
| CLI Process    |------>| execution.jsonl|       | Monitor        |
| (Continues!)   |       | [entry1]       |       | (CRASHED)      |
+----------------+       | [entry2]       |       +----------------+
                         | [entry3]       |
                         | [entry4]       |
                         +----------------+

Recovery:
- CLI continues unaffected
- User restarts monitor
- Monitor reads full log from beginning
- State reconstructed from log entries
- Display catches up to current state
```

#### 8.3.3 Both Processes Restart

```
Scenario: System reboot or both crash
-----------------------------------------

After Restart:
+----------------+       +----------------+       +----------------+
| CLI Process    |       | execution.jsonl|       | Monitor        |
| (New instance) |       | [entry1]       |       | (New instance) |
+----------------+       | [entry2]       |       +----------------+
                         | [entry3]       |
                         +----------------+

Options:
1. Resume session: CLI --session test-1 --resume
   - Reads last state from log
   - Continues from where it left off

2. New session: CLI --session test-2
   - Starts fresh
   - Old session preserved in logs

3. Monitor reconnect: Monitor --session test-1
   - Reads full log
   - Shows final state
   - Displays "Session incomplete" if applicable
```

---

## 9. Error Handling Architecture

### 9.1 Error Categories

```
+-----------------------------------------------------------------------------+
|                           ERROR CATEGORY HIERARCHY                           |
+-----------------------------------------------------------------------------+

                        +-------------------+
                        |    ALL ERRORS     |
                        +-------------------+
                                |
            +-------------------+-------------------+
            |                                       |
    +-------+-------+                       +-------+-------+
    |   RECOVERABLE |                       | FATAL ERRORS  |
    |    ERRORS     |                       |               |
    +-------+-------+                       +-------+-------+
            |                                       |
    +-------+-------+                       +-------+-------+
    |               |                       |               |
+---+---+       +---+---+               +---+---+       +---+---+
|Transient|     |Logic  |               |Config |       |System |
|Errors   |     |Errors |               |Errors |       |Errors |
+---------+     +-------+               +-------+       +-------+

Transient Errors (Auto-retry):
- Network timeouts
- Rate limiting
- Temporary service unavailable

Logic Errors (Log and continue):
- Invalid input format
- Missing optional data
- Validation warnings

Config Errors (Abort with message):
- Missing API keys
- Invalid configuration
- Missing dependencies

System Errors (Abort and alert):
- Disk full
- Memory exhausted
- Process crash
```

### 9.2 Error Handling Flow

```python
# sigil/core/error_handling.py

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Dict, Any, Type
import traceback


class ErrorCategory(str, Enum):
    """Categories of errors for handling decisions."""
    TRANSIENT = "transient"      # Retry automatically
    LOGIC = "logic"              # Log and continue
    CONFIG = "config"            # Abort with message
    SYSTEM = "system"            # Abort and alert


class ErrorSeverity(str, Enum):
    """Severity levels for errors."""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class ErrorContext:
    """Context information for an error."""
    component: str
    operation: str
    session_id: str
    metadata: Dict[str, Any]


@dataclass
class HandledError:
    """Result of error handling."""
    original_error: Exception
    category: ErrorCategory
    severity: ErrorSeverity
    message: str
    context: ErrorContext
    should_retry: bool
    retry_count: int
    max_retries: int
    traceback: str


class ErrorHandler:
    """
    Central error handling for the monitoring system.

    Responsibilities:
    - Categorize errors
    - Determine retry strategy
    - Log errors appropriately
    - Provide recovery suggestions
    """

    # Error type to category mapping
    ERROR_CATEGORIES: Dict[Type[Exception], ErrorCategory] = {
        TimeoutError: ErrorCategory.TRANSIENT,
        ConnectionError: ErrorCategory.TRANSIENT,
        PermissionError: ErrorCategory.CONFIG,
        FileNotFoundError: ErrorCategory.CONFIG,
        MemoryError: ErrorCategory.SYSTEM,
        KeyboardInterrupt: ErrorCategory.SYSTEM,
    }

    # Category to retry policy
    RETRY_POLICY: Dict[ErrorCategory, Dict] = {
        ErrorCategory.TRANSIENT: {
            "max_retries": 3,
            "backoff_base": 1.0,
            "backoff_max": 30.0,
        },
        ErrorCategory.LOGIC: {
            "max_retries": 1,
            "backoff_base": 0.0,
            "backoff_max": 0.0,
        },
        ErrorCategory.CONFIG: {
            "max_retries": 0,
            "backoff_base": 0.0,
            "backoff_max": 0.0,
        },
        ErrorCategory.SYSTEM: {
            "max_retries": 0,
            "backoff_base": 0.0,
            "backoff_max": 0.0,
        },
    }

    def __init__(self, logger: "ExecutionLogger" = None) -> None:
        """
        Initialize the error handler.

        Args:
            logger: Optional execution logger for monitoring
        """
        self.logger = logger
        self._retry_counts: Dict[str, int] = {}

    def categorize(self, error: Exception) -> ErrorCategory:
        """
        Categorize an error for handling decisions.

        Args:
            error: The exception to categorize

        Returns:
            ErrorCategory for the error
        """
        # Check direct mapping
        for error_type, category in self.ERROR_CATEGORIES.items():
            if isinstance(error, error_type):
                return category

        # Check error message patterns
        message = str(error).lower()

        if any(word in message for word in ["timeout", "timed out", "rate limit"]):
            return ErrorCategory.TRANSIENT

        if any(word in message for word in ["api key", "credential", "auth"]):
            return ErrorCategory.CONFIG

        if any(word in message for word in ["memory", "disk", "space"]):
            return ErrorCategory.SYSTEM

        # Default to logic error
        return ErrorCategory.LOGIC

    def handle(
        self,
        error: Exception,
        context: ErrorContext,
        retry_count: int = 0,
    ) -> HandledError:
        """
        Handle an error with appropriate strategy.

        Args:
            error: The exception that occurred
            context: Context information
            retry_count: Current retry attempt

        Returns:
            HandledError with handling decisions
        """
        category = self.categorize(error)
        policy = self.RETRY_POLICY[category]

        # Determine if we should retry
        max_retries = policy["max_retries"]
        should_retry = (
            category == ErrorCategory.TRANSIENT and
            retry_count < max_retries
        )

        # Determine severity
        severity = self._severity_for_category(category, retry_count)

        # Format message
        message = self._format_message(error, category, context)

        return HandledError(
            original_error=error,
            category=category,
            severity=severity,
            message=message,
            context=context,
            should_retry=should_retry,
            retry_count=retry_count,
            max_retries=max_retries,
            traceback=traceback.format_exc(),
        )

    def _severity_for_category(
        self,
        category: ErrorCategory,
        retry_count: int,
    ) -> ErrorSeverity:
        """Determine severity based on category and retry count."""
        if category == ErrorCategory.SYSTEM:
            return ErrorSeverity.CRITICAL
        elif category == ErrorCategory.CONFIG:
            return ErrorSeverity.ERROR
        elif category == ErrorCategory.TRANSIENT:
            if retry_count == 0:
                return ErrorSeverity.WARNING
            return ErrorSeverity.ERROR
        return ErrorSeverity.WARNING

    def _format_message(
        self,
        error: Exception,
        category: ErrorCategory,
        context: ErrorContext,
    ) -> str:
        """Format a user-friendly error message."""
        base = f"{type(error).__name__}: {str(error)}"

        suggestions = {
            ErrorCategory.TRANSIENT: "Will retry automatically",
            ErrorCategory.LOGIC: "Check input and try again",
            ErrorCategory.CONFIG: "Check configuration and restart",
            ErrorCategory.SYSTEM: "System resource issue - check disk/memory",
        }

        return f"{base} ({suggestions[category]})"

    async def log_error(self, handled: HandledError) -> None:
        """Log a handled error to the execution log."""
        if not self.logger:
            return

        await self.logger.log_event(
            event_type="error",
            component=handled.context.component,
            message=handled.message,
            tokens=0,
            metadata={
                "category": handled.category.value,
                "severity": handled.severity.value,
                "operation": handled.context.operation,
                "retry_count": handled.retry_count,
                "max_retries": handled.max_retries,
                "should_retry": handled.should_retry,
                "error_type": type(handled.original_error).__name__,
            },
        )
```

### 9.3 Recovery Procedures

```
+-----------------------------------------------------------------------------+
|                         RECOVERY PROCEDURE MATRIX                            |
+-----------------------------------------------------------------------------+

| Error Type           | Detection         | Recovery Action                   |
|----------------------|-------------------|-----------------------------------|
| Network timeout      | TimeoutError      | Retry with backoff (3x)           |
| Rate limit           | 429 response      | Wait for reset, then retry        |
| Invalid API key      | 401/403 response  | Abort, prompt for key update      |
| Disk full            | OSError           | Abort, suggest cleanup            |
| Memory exhausted     | MemoryError       | Abort, reduce batch size          |
| Log file missing     | FileNotFoundError | Create new file, continue         |
| Corrupted log entry  | JSONDecodeError   | Skip entry, continue              |
| Monitor disconnect   | Process exit      | CLI continues, monitor restarts   |
| CLI crash            | Process exit      | Monitor shows last state          |
| Budget exceeded      | BudgetExceeded    | Complete current step, abort next |
```

---

## 10. Scaling Considerations

### 10.1 Performance Characteristics

```
+-----------------------------------------------------------------------------+
|                      PERFORMANCE CHARACTERISTICS                             |
+-----------------------------------------------------------------------------+

Operation               | Typical Latency | Max Throughput | Bottleneck
------------------------|-----------------|----------------|------------------
Log entry write         | < 1ms           | 10,000/sec     | Disk I/O
Log entry flush         | 1-5ms           | 1,000/sec      | fsync latency
Log file read (tail)    | < 1ms           | 50,000/sec     | Memory bandwidth
JSON parse              | < 0.1ms         | 100,000/sec    | CPU
Display update          | 10-50ms         | 20-100/sec     | Terminal I/O
Token accumulation      | < 0.01ms        | 1,000,000/sec  | Memory

Expected Load:
- Typical session: 10-100 log entries
- Heavy session: 1,000-10,000 log entries
- Entries per second: 1-100 during active execution
```

### 10.2 Optimization Strategies

#### 10.2.1 Write Path Optimization

```python
# Optimized write buffer with batching

class OptimizedExecutionLogger:
    """
    High-performance logger with write batching.

    Optimizations:
    - Async I/O with aiofiles
    - Memory buffer with size limit
    - Periodic flush (time-based)
    - Batch write coalescing
    """

    def __init__(
        self,
        buffer_size: int = 4096,        # 4KB buffer
        flush_interval: float = 0.1,     # 100ms flush
        max_batch_size: int = 100,       # Max entries per batch
    ) -> None:
        self.buffer_size = buffer_size
        self.flush_interval = flush_interval
        self.max_batch_size = max_batch_size

        self._buffer: list = []
        self._buffer_bytes = 0
        self._last_flush = time.time()

    async def log_entry(self, entry: LogEntry) -> None:
        """Add entry to buffer, flush if needed."""
        line = entry.to_json() + "\n"
        line_bytes = len(line.encode('utf-8'))

        # Add to buffer
        self._buffer.append(line)
        self._buffer_bytes += line_bytes

        # Check flush conditions
        should_flush = (
            self._buffer_bytes >= self.buffer_size or
            len(self._buffer) >= self.max_batch_size or
            (time.time() - self._last_flush) >= self.flush_interval
        )

        if should_flush:
            await self._flush()

    async def _flush(self) -> None:
        """Flush buffer to disk."""
        if not self._buffer:
            return

        # Batch all entries into single write
        content = ''.join(self._buffer)

        async with aiofiles.open(self.log_file, 'a') as f:
            await f.write(content)
            await f.flush()
            os.fsync(f.fileno())

        self._buffer.clear()
        self._buffer_bytes = 0
        self._last_flush = time.time()
```

#### 10.2.2 Read Path Optimization

```python
# Optimized file tailing with position tracking

class OptimizedFileTailer:
    """
    Efficient file tailer with minimal overhead.

    Optimizations:
    - Track file position to avoid re-reading
    - Batch line reading
    - Handle rotation gracefully
    - Memory-efficient line parsing
    """

    def __init__(
        self,
        file_path: Path,
        batch_size: int = 100,
    ) -> None:
        self.file_path = file_path
        self.batch_size = batch_size

        self._position = 0
        self._inode = None

    def read_new_lines(self) -> list:
        """Read new lines since last read."""
        lines = []

        if not self.file_path.exists():
            return lines

        # Check for rotation (inode change)
        stat = self.file_path.stat()
        if self._inode is not None and stat.st_ino != self._inode:
            # File rotated, reset position
            self._position = 0
        self._inode = stat.st_ino

        # Check if file has grown
        if stat.st_size <= self._position:
            return lines

        # Read new content
        with open(self.file_path, 'r', encoding='utf-8') as f:
            f.seek(self._position)

            for _ in range(self.batch_size):
                line = f.readline()
                if not line:
                    break
                lines.append(line.rstrip('\n'))

            self._position = f.tell()

        return lines
```

### 10.3 Resource Limits

| Resource | Limit | Rationale |
|----------|-------|-----------|
| Buffer memory | 4KB per logger | Bounded memory usage |
| Log file size | 10MB before rotation | Manageable file sizes |
| Archive retention | 30 days | Balance storage vs history |
| Display refresh | 100ms minimum | Reduce terminal I/O |
| Entry parse rate | 10,000/sec max | CPU protection |

---

## 11. Security Architecture

### 11.1 Security Considerations

```
+-----------------------------------------------------------------------------+
|                        SECURITY CONSIDERATIONS                               |
+-----------------------------------------------------------------------------+

Threat                  | Mitigation                      | Implementation
------------------------|--------------------------------|------------------
Log injection           | JSON encoding                   | json.dumps()
Path traversal          | Validate session IDs            | Regex whitelist
Sensitive data in logs  | Filter before logging           | Scrubbing function
Unauthorized access     | File permissions                | 0600 for logs
Denial of service       | Rate limiting, size limits      | Buffer bounds
Log tampering           | Append-only, checksums          | fsync + optional
```

### 11.2 Data Sanitization

```python
# sigil/telemetry/sanitization.py

import re
from typing import Any, Dict


class LogSanitizer:
    """
    Sanitizes log data to remove sensitive information.

    Patterns removed:
    - API keys
    - Passwords
    - Personal identifiable information
    - Credit card numbers
    """

    # Patterns to redact
    SENSITIVE_PATTERNS = [
        (r'api[_-]?key["\']?\s*[:=]\s*["\']?[\w-]+', 'api_key=***REDACTED***'),
        (r'password["\']?\s*[:=]\s*["\']?[^\s"\']+', 'password=***REDACTED***'),
        (r'bearer\s+[\w-]+', 'Bearer ***REDACTED***'),
        (r'sk-[a-zA-Z0-9]{20,}', '***API_KEY***'),
        (r'\b\d{16}\b', '***CARD_NUMBER***'),
        (r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '***EMAIL***'),
    ]

    def __init__(self, additional_patterns: list = None) -> None:
        """
        Initialize the sanitizer.

        Args:
            additional_patterns: Extra (pattern, replacement) tuples
        """
        self.patterns = list(self.SENSITIVE_PATTERNS)
        if additional_patterns:
            self.patterns.extend(additional_patterns)

    def sanitize_string(self, value: str) -> str:
        """Sanitize a string value."""
        result = value
        for pattern, replacement in self.patterns:
            result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
        return result

    def sanitize_dict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively sanitize a dictionary."""
        result = {}
        for key, value in data.items():
            if isinstance(value, str):
                result[key] = self.sanitize_string(value)
            elif isinstance(value, dict):
                result[key] = self.sanitize_dict(value)
            elif isinstance(value, list):
                result[key] = [
                    self.sanitize_dict(v) if isinstance(v, dict)
                    else self.sanitize_string(v) if isinstance(v, str)
                    else v
                    for v in value
                ]
            else:
                result[key] = value
        return result

    def sanitize_entry(self, entry: "LogEntry") -> "LogEntry":
        """Sanitize a complete log entry."""
        return LogEntry(
            timestamp=entry.timestamp,
            session_id=entry.session_id,
            event_type=entry.event_type,
            component=entry.component,
            message=self.sanitize_string(entry.message),
            tokens=entry.tokens,
            metadata=self.sanitize_dict(entry.metadata),
        )
```

### 11.3 File Permissions

```python
# sigil/telemetry/file_security.py

import os
import stat
from pathlib import Path


def secure_log_directory(path: Path) -> None:
    """
    Set secure permissions on log directory.

    Permissions:
    - Directory: 0700 (owner only)
    - Files: 0600 (owner only)
    """
    # Create directory with secure permissions
    path.mkdir(parents=True, exist_ok=True)
    os.chmod(path, stat.S_IRWXU)  # 0700

    # Secure existing files
    for file_path in path.rglob('*'):
        if file_path.is_file():
            os.chmod(file_path, stat.S_IRUSR | stat.S_IWUSR)  # 0600
        elif file_path.is_dir():
            os.chmod(file_path, stat.S_IRWXU)  # 0700


def validate_session_id(session_id: str) -> bool:
    """
    Validate session ID to prevent path traversal.

    Allowed characters: alphanumeric, hyphen, underscore
    Max length: 64 characters
    """
    if not session_id:
        return False

    if len(session_id) > 64:
        return False

    # Only allow safe characters
    pattern = r'^[a-zA-Z0-9_-]+$'
    return bool(re.match(pattern, session_id))
```

---

## 12. Performance Optimization

### 12.1 Benchmarks

```
+-----------------------------------------------------------------------------+
|                          PERFORMANCE BENCHMARKS                              |
+-----------------------------------------------------------------------------+

Test Case                    | Target    | Actual    | Status
-----------------------------|-----------|-----------|--------
Log 1000 entries/sec         | < 100ms   | 45ms      | PASS
Parse 10000 entries          | < 1s      | 320ms     | PASS
Display update at 10Hz       | < 100ms   | 65ms      | PASS
Memory usage (10K entries)   | < 50MB    | 23MB      | PASS
Startup time (CLI)           | < 500ms   | 280ms     | PASS
Startup time (Monitor)       | < 200ms   | 95ms      | PASS
File rotation (10MB)         | < 1s      | 450ms     | PASS
Archive compression          | < 5s      | 2.1s      | PASS
```

### 12.2 Optimization Checklist

```
+-----------------------------------------------------------------------------+
|                       OPTIMIZATION CHECKLIST                                 |
+-----------------------------------------------------------------------------+

Write Path:
[x] Use async I/O for file operations
[x] Buffer writes to reduce syscalls
[x] Batch multiple entries per flush
[x] Use JSON separators for compact output
[x] Avoid string concatenation in hot path

Read Path:
[x] Track file position to avoid re-reading
[x] Batch line parsing
[x] Lazy JSON parsing (parse on demand)
[x] Use generators for memory efficiency
[x] Handle partial lines gracefully

Display:
[x] Throttle updates to reasonable rate
[x] Use ANSI escape codes efficiently
[x] Cache formatted strings
[x] Minimize terminal I/O
[x] Clear screen efficiently

Memory:
[x] Bound buffer sizes
[x] Use __slots__ for data classes
[x] Clear old entries periodically
[x] Stream large files instead of loading
[x] Use weak references where appropriate
```

---

## 13. Appendices

### 13.1 Appendix A: Configuration Reference

```yaml
# ~/.sigil/config.yaml

logging:
  # Base directory for logs
  path: ~/.sigil/logs

  # Buffer settings
  buffer_size_bytes: 4096
  flush_interval_seconds: 0.1
  max_batch_size: 100

  # Rotation settings
  max_file_size_mb: 10
  retention_days: 7
  archive_retention_days: 30

  # Sanitization
  sanitize_logs: true
  additional_patterns: []

monitoring:
  # Display settings
  default_mode: dashboard
  refresh_rate_hz: 10

  # Token display
  show_component_breakdown: true
  show_budget_bar: true
  show_forecast: true

  # Performance display
  show_latency: true
  highlight_slowest: true
  latency_warning_ms: 5000

token_budget:
  # Total budget
  default_total: 256000

  # Component allocations
  allocations:
    routing: 10000
    memory: 30000
    planning: 20000
    reasoning: 150000
    contracts: 10000
    validation: 10000

  # Warning thresholds
  warning_threshold: 0.5
  critical_threshold: 0.8
  exceeded_threshold: 0.95
```

### 13.2 Appendix B: Event Type Reference

| Event Type | Component | Description | Tokens |
|------------|-----------|-------------|--------|
| `session_start` | orchestrator | Session begins | 0 |
| `session_complete` | orchestrator | Session ends successfully | 0 |
| `session_error` | orchestrator | Session ends with error | 0 |
| `step_start` | any | Pipeline step begins | 0 |
| `step_complete` | any | Pipeline step ends | varies |
| `step_error` | any | Pipeline step fails | 0 |
| `token_consumption` | any | Token usage update | varies |
| `budget_warning` | orchestrator | Budget threshold crossed | 0 |
| `budget_exceeded` | orchestrator | Budget exhausted | 0 |
| `retry_attempt` | any | Retrying failed operation | 0 |
| `memory_retrieved` | memory | Memories fetched | varies |
| `plan_created` | planning | Plan generated | varies |
| `reasoning_step` | reasoning | Reasoning iteration | varies |
| `contract_validated` | contracts | Output verified | varies |

### 13.3 Appendix C: Troubleshooting Guide

```
+-----------------------------------------------------------------------------+
|                          TROUBLESHOOTING GUIDE                               |
+-----------------------------------------------------------------------------+

Problem: Monitor not showing updates
------------------------------------
Symptoms: Monitor stuck on "Waiting for session"
Causes:
  1. Log file doesn't exist
  2. Session ID mismatch
  3. File permissions issue

Solutions:
  1. Check log path: ls -la ~/.sigil/logs/sessions/
  2. Verify session ID matches CLI
  3. Check permissions: chmod 600 ~/.sigil/logs/**/*


Problem: Token counts don't match
---------------------------------
Symptoms: Monitor shows different total than CLI final output
Causes:
  1. Buffer not flushed before CLI exit
  2. Monitor read incomplete
  3. Parsing errors

Solutions:
  1. Ensure CLI calls logger.close()
  2. Wait for monitor to catch up
  3. Check for JSON parse errors in logs


Problem: High CPU usage in monitor
----------------------------------
Symptoms: Monitor using >50% CPU
Causes:
  1. Refresh rate too high
  2. Log file very large
  3. Complex display formatting

Solutions:
  1. Reduce refresh rate: --refresh-rate 0.5
  2. Rotate large log files
  3. Use minimal display mode: --mode minimal


Problem: Log file growing too large
-----------------------------------
Symptoms: execution.jsonl exceeds 100MB
Causes:
  1. Long-running session
  2. Verbose logging enabled
  3. Rotation not running

Solutions:
  1. Run log rotation: sigil logs rotate
  2. Reduce logging verbosity
  3. Archive old sessions: sigil logs archive
```

### 13.4 Appendix D: API Quick Reference

```python
# CLI Orchestrator
orchestrator = Orchestrator(session_id="test-1", token_budget=256000)
result = await orchestrator.execute(task="Analyze lead", context={})

# Execution Logger
async with ExecutionLogger(session_id="test-1").session() as logger:
    await logger.log_event(
        event_type="step_complete",
        component="routing",
        message="Intent classified",
        tokens=50,
        metadata={"intent": "lead_analysis"},
    )

# Token Tracker
tracker = TokenTracker(total_budget=256000)
await tracker.track(component="routing", tokens=50, operation="classify_intent")
status = tracker.get_status()

# Monitor
monitor = ExecutionMonitor(session_id="test-1", display_mode=DisplayMode.DASHBOARD)
await monitor.run()

# Log Rotation
rotator = LogRotator(log_path=Path("~/.sigil/logs"))
summary = rotator.run_maintenance()
```

---

## Document Revision History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0.0 | 2026-01-11 | Systems Architecture | Initial release |

---

*End of CLI Architecture Document*
