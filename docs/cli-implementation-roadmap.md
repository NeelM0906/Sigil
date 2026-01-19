# Sigil v2 CLI Monitoring Implementation Roadmap

## Document Information

| Field | Value |
|-------|-------|
| Version | 1.0.0 |
| Status | Implementation Plan |
| Created | 2026-01-11 |
| Author | Systems Architecture Team |
| Timeline | 4 Weeks |

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Implementation Overview](#2-implementation-overview)
3. [Phase 1: Foundation](#3-phase-1-foundation)
4. [Phase 2: Token Tracking](#4-phase-2-token-tracking)
5. [Phase 3: Monitor Implementation](#5-phase-3-monitor-implementation)
6. [Phase 4: Integration and Polish](#6-phase-4-integration-and-polish)
7. [Testing Strategy](#7-testing-strategy)
8. [Rollout Plan](#8-rollout-plan)
9. [Backwards Compatibility](#9-backwards-compatibility)
10. [Risk Management](#10-risk-management)

---

## 1. Executive Summary

### 1.1 Objective

Implement a comprehensive monitoring and observability system for Sigil v2 CLI execution, enabling real-time visibility into pipeline execution, token consumption tracking, and performance monitoring.

### 1.2 Timeline Overview

```
+-----------------------------------------------------------------------------+
|                         IMPLEMENTATION TIMELINE                              |
+-----------------------------------------------------------------------------+

Week 1: Foundation
+------------------+------------------+------------------+------------------+
| Day 1-2          | Day 3-4          | Day 5            | Review           |
| Log Schema       | Execution Logger | Buffer System    | Checkpoint       |
+------------------+------------------+------------------+------------------+

Week 2: Token Tracking
+------------------+------------------+------------------+------------------+
| Day 1-2          | Day 3-4          | Day 5            | Review           |
| Token Counter    | Budget Manager   | Alert System     | Checkpoint       |
+------------------+------------------+------------------+------------------+

Week 3: Monitor Implementation
+------------------+------------------+------------------+------------------+
| Day 1-2          | Day 3            | Day 4-5          | Review           |
| File Tailer      | Log Parser       | Display Renderer | Checkpoint       |
+------------------+------------------+------------------+------------------+

Week 4: Integration and Polish
+------------------+------------------+------------------+------------------+
| Day 1-2          | Day 3            | Day 4            | Day 5            |
| CLI Integration  | Testing          | Documentation    | Release          |
+------------------+------------------+------------------+------------------+
```

### 1.3 Success Criteria

| Criterion | Measure | Target |
|-----------|---------|--------|
| Log write latency | 99th percentile | < 10ms |
| Monitor update rate | Updates per second | 10 Hz |
| Token accuracy | Deviation from actual | < 1% |
| CPU overhead | Additional CPU usage | < 5% |
| Memory overhead | Additional memory | < 50MB |

---

## 2. Implementation Overview

### 2.1 Component Dependency Graph

```
+-----------------------------------------------------------------------------+
|                      COMPONENT DEPENDENCY GRAPH                              |
+-----------------------------------------------------------------------------+

                    +------------------------+
                    |   CLI Orchestrator     |
                    | (existing, to modify)  |
                    +------------------------+
                              |
                              | depends on
                              v
                    +------------------------+
                    |   Execution Logger     |  <-- Phase 1
                    +------------------------+
                              |
              +---------------+---------------+
              |                               |
              v                               v
    +------------------+            +------------------+
    |   Log Schema     |            |  Buffer System   |  <-- Phase 1
    +------------------+            +------------------+
              |
              | used by
              v
    +------------------+
    |   Token Tracker  |  <-- Phase 2
    +------------------+
              |
              | uses
              v
    +------------------+
    |  Budget Manager  |  <-- Phase 2
    +------------------+
              |
              | triggers
              v
    +------------------+
    |   Alert System   |  <-- Phase 2
    +------------------+


                    +------------------------+
                    |    Monitor Script      |  <-- Phase 3
                    +------------------------+
                              |
              +---------------+---------------+
              |               |               |
              v               v               v
    +------------+    +------------+    +---------------+
    | File Tailer|    | Log Parser |    |Display Render |  <-- Phase 3
    +------------+    +------------+    +---------------+


                    +------------------------+
                    |    CLI Integration     |  <-- Phase 4
                    +------------------------+
                              |
              +---------------+---------------+
              |               |               |
              v               v               v
    +------------+    +------------+    +---------------+
    | Orchestrator|   |  Commands  |    |   Testing    |  <-- Phase 4
    | Integration |   | Integration|    |              |
    +------------+    +------------+    +---------------+
```

### 2.2 File Structure

```
sigil/
+-- telemetry/                      # NEW: Telemetry module
|   +-- __init__.py
|   +-- execution_logger.py         # Phase 1: Logging
|   +-- log_schema.py               # Phase 1: Schema
|   +-- buffer.py                   # Phase 1: Buffering
|   +-- tokens.py                   # Phase 2: Token tracking
|   +-- budget.py                   # Phase 2: Budget management
|   +-- alerts.py                   # Phase 2: Alert system
|   +-- token_counting.py           # Phase 2: Token counter
|   +-- log_parser.py               # Phase 3: Parser
|   +-- log_rotation.py             # Phase 4: Rotation
|
+-- interfaces/
|   +-- cli/
|       +-- monitor.py              # Phase 3: Monitor script
|       +-- display.py              # Phase 3: Display renderer
|       +-- budget_display.py       # Phase 3: Budget display
|       +-- orchestrator.py         # Phase 4: Modified orchestrator
|
+-- tests/
    +-- telemetry/                  # NEW: Telemetry tests
        +-- test_execution_logger.py
        +-- test_tokens.py
        +-- test_budget.py
        +-- test_monitor.py
```

---

## 3. Phase 1: Foundation

### 3.1 Overview

**Duration:** Week 1 (5 days)
**Goal:** Establish logging infrastructure

### 3.2 Tasks

```
+-----------------------------------------------------------------------------+
|                         PHASE 1 TASK BREAKDOWN                               |
+-----------------------------------------------------------------------------+

TASK 1.1: Log Schema Definition (Day 1)
---------------------------------------
Priority: P0
Effort: 4 hours
Dependencies: None

Deliverables:
- LogEntry dataclass
- Event type enumeration
- Metadata schema
- JSON serialization

Files:
- sigil/telemetry/log_schema.py


TASK 1.2: Execution Logger Core (Day 1-2)
-----------------------------------------
Priority: P0
Effort: 8 hours
Dependencies: Task 1.1

Deliverables:
- ExecutionLogger class
- Async write methods
- Session management
- File path handling

Files:
- sigil/telemetry/execution_logger.py


TASK 1.3: Buffer System (Day 2-3)
---------------------------------
Priority: P0
Effort: 6 hours
Dependencies: Task 1.2

Deliverables:
- In-memory buffer
- Periodic flush
- Size-based flush
- Thread-safe operations

Files:
- sigil/telemetry/buffer.py
- Update: sigil/telemetry/execution_logger.py


TASK 1.4: Unit Tests (Day 4)
----------------------------
Priority: P0
Effort: 6 hours
Dependencies: Tasks 1.1-1.3

Deliverables:
- Schema tests
- Logger tests
- Buffer tests
- Edge case coverage

Files:
- sigil/tests/telemetry/test_log_schema.py
- sigil/tests/telemetry/test_execution_logger.py
- sigil/tests/telemetry/test_buffer.py


TASK 1.5: Integration Verification (Day 5)
------------------------------------------
Priority: P0
Effort: 4 hours
Dependencies: Tasks 1.1-1.4

Deliverables:
- End-to-end logging test
- Performance benchmarks
- Documentation draft

Files:
- sigil/tests/telemetry/test_integration_logging.py
```

### 3.3 Implementation Details

#### 3.3.1 Log Schema (Task 1.1)

```python
# sigil/telemetry/log_schema.py

from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Dict, Any, Optional
from enum import Enum
import json


class EventType(str, Enum):
    """Types of execution events."""
    SESSION_START = "session_start"
    SESSION_COMPLETE = "session_complete"
    SESSION_ERROR = "session_error"
    STEP_START = "step_start"
    STEP_COMPLETE = "step_complete"
    STEP_ERROR = "step_error"
    TOKEN_CONSUMPTION = "token_consumption"
    BUDGET_WARNING = "budget_warning"
    BUDGET_EXCEEDED = "budget_exceeded"
    RETRY_ATTEMPT = "retry_attempt"


class Component(str, Enum):
    """Pipeline components."""
    ORCHESTRATOR = "orchestrator"
    ROUTING = "routing"
    MEMORY = "memory"
    PLANNING = "planning"
    REASONING = "reasoning"
    CONTRACTS = "contracts"
    VALIDATION = "validation"


@dataclass
class LogEntry:
    """A single log entry for execution logging."""

    timestamp: str
    session_id: str
    event_type: str
    component: str
    message: str
    tokens: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        session_id: str,
        event_type: EventType,
        component: Component,
        message: str,
        tokens: int = 0,
        **metadata,
    ) -> "LogEntry":
        """Create a new log entry with current timestamp."""
        return cls(
            timestamp=datetime.now().isoformat(),
            session_id=session_id,
            event_type=event_type.value,
            component=component.value,
            message=message,
            tokens=tokens,
            metadata=metadata,
        )

    def to_json(self) -> str:
        """Serialize to JSON line."""
        return json.dumps(asdict(self), separators=(',', ':'))

    @classmethod
    def from_json(cls, line: str) -> "LogEntry":
        """Deserialize from JSON line."""
        data = json.loads(line.strip())
        return cls(**data)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
```

#### 3.3.2 Execution Logger (Task 1.2)

```python
# sigil/telemetry/execution_logger.py

import asyncio
from pathlib import Path
from typing import Optional, Dict, Any
from contextlib import asynccontextmanager
import aiofiles

from .log_schema import LogEntry, EventType, Component
from .buffer import LogBuffer


class ExecutionLogger:
    """
    Async execution logger with buffered writes.

    Usage:
        async with ExecutionLogger(session_id="test").session() as logger:
            await logger.log_event(
                EventType.STEP_COMPLETE,
                Component.ROUTING,
                "Routing complete",
                tokens=50,
            )
    """

    DEFAULT_LOG_PATH = Path.home() / ".sigil" / "logs"

    def __init__(
        self,
        session_id: str,
        log_path: Optional[Path] = None,
        buffer_size: int = 4096,
        flush_interval: float = 0.1,
    ) -> None:
        self.session_id = session_id
        self.log_path = log_path or self.DEFAULT_LOG_PATH
        self.buffer = LogBuffer(
            size_limit=buffer_size,
            time_limit=flush_interval,
        )

        self._initialized = False
        self._file_handle = None
        self._flush_task: Optional[asyncio.Task] = None

    @property
    def session_dir(self) -> Path:
        """Get session directory path."""
        return self.log_path / "sessions" / self.session_id

    @property
    def log_file(self) -> Path:
        """Get log file path."""
        return self.session_dir / "execution.jsonl"

    async def initialize(self) -> None:
        """Initialize logger, create directories and files."""
        if self._initialized:
            return

        # Create directories
        self.session_dir.mkdir(parents=True, exist_ok=True)

        # Open file handle
        self._file_handle = await aiofiles.open(
            self.log_file, mode='a', encoding='utf-8'
        )

        # Start background flush task
        self._flush_task = asyncio.create_task(self._periodic_flush())

        self._initialized = True

    async def log_event(
        self,
        event_type: EventType,
        component: Component,
        message: str,
        tokens: int = 0,
        **metadata,
    ) -> None:
        """Log an execution event."""
        if not self._initialized:
            await self.initialize()

        entry = LogEntry.create(
            session_id=self.session_id,
            event_type=event_type,
            component=component,
            message=message,
            tokens=tokens,
            **metadata,
        )

        await self.buffer.add(entry)

        # Check if buffer needs flushing
        if self.buffer.should_flush():
            await self._flush()

    async def _flush(self) -> None:
        """Flush buffer to disk."""
        entries = await self.buffer.drain()
        if entries and self._file_handle:
            content = '\n'.join(e.to_json() for e in entries) + '\n'
            await self._file_handle.write(content)
            await self._file_handle.flush()

    async def _periodic_flush(self) -> None:
        """Background task for periodic flushing."""
        while True:
            await asyncio.sleep(self.buffer.time_limit)
            if self.buffer.has_data():
                await self._flush()

    async def close(self) -> None:
        """Close logger and flush remaining data."""
        if self._flush_task:
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass

        # Final flush
        await self._flush()

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
```

#### 3.3.3 Buffer System (Task 1.3)

```python
# sigil/telemetry/buffer.py

import asyncio
import time
from dataclasses import dataclass, field
from typing import List
from .log_schema import LogEntry


@dataclass
class LogBuffer:
    """
    Thread-safe buffer for log entries.

    Supports both size-based and time-based flushing.
    """

    size_limit: int = 4096  # bytes
    time_limit: float = 0.1  # seconds

    _entries: List[LogEntry] = field(default_factory=list)
    _bytes: int = 0
    _last_flush: float = field(default_factory=time.time)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    async def add(self, entry: LogEntry) -> None:
        """Add entry to buffer."""
        async with self._lock:
            self._entries.append(entry)
            self._bytes += len(entry.to_json().encode('utf-8'))

    def should_flush(self) -> bool:
        """Check if buffer should be flushed."""
        return (
            self._bytes >= self.size_limit or
            (time.time() - self._last_flush) >= self.time_limit
        )

    def has_data(self) -> bool:
        """Check if buffer has data."""
        return len(self._entries) > 0

    async def drain(self) -> List[LogEntry]:
        """Drain all entries from buffer."""
        async with self._lock:
            entries = self._entries
            self._entries = []
            self._bytes = 0
            self._last_flush = time.time()
            return entries

    @property
    def size(self) -> int:
        """Current buffer size in bytes."""
        return self._bytes

    @property
    def count(self) -> int:
        """Number of entries in buffer."""
        return len(self._entries)
```

### 3.4 Phase 1 Checklist

```
+-----------------------------------------------------------------------------+
|                         PHASE 1 COMPLETION CHECKLIST                         |
+-----------------------------------------------------------------------------+

[ ] Log schema defined with all event types
[ ] LogEntry serialization working (to_json, from_json)
[ ] ExecutionLogger creates directories
[ ] ExecutionLogger writes to correct path
[ ] Buffer accumulates entries
[ ] Buffer flushes on size limit
[ ] Buffer flushes on time limit
[ ] Async operations work correctly
[ ] Context manager cleans up properly
[ ] Unit tests pass (>90% coverage)
[ ] Performance benchmark meets target (<10ms write)
[ ] Documentation updated
```

---

## 4. Phase 2: Token Tracking

### 4.1 Overview

**Duration:** Week 2 (5 days)
**Goal:** Implement comprehensive token tracking

### 4.2 Tasks

```
+-----------------------------------------------------------------------------+
|                         PHASE 2 TASK BREAKDOWN                               |
+-----------------------------------------------------------------------------+

TASK 2.1: Token Counter (Day 1)
-------------------------------
Priority: P0
Effort: 4 hours
Dependencies: Phase 1 complete

Deliverables:
- TokenCounter class using tiktoken
- Message counting
- Estimation utilities

Files:
- sigil/telemetry/token_counting.py


TASK 2.2: Token Tracker (Day 1-2)
---------------------------------
Priority: P0
Effort: 8 hours
Dependencies: Task 2.1

Deliverables:
- TokenTracker class
- Component budgets
- Usage recording
- Status reporting

Files:
- sigil/telemetry/tokens.py


TASK 2.3: Budget Manager (Day 2-3)
----------------------------------
Priority: P0
Effort: 6 hours
Dependencies: Task 2.2

Deliverables:
- BudgetManager class
- Allocation strategies
- Forecasting
- Threshold checking

Files:
- sigil/telemetry/budget.py


TASK 2.4: Alert System (Day 3-4)
--------------------------------
Priority: P1
Effort: 6 hours
Dependencies: Task 2.3

Deliverables:
- BudgetAlertManager class
- Warning callbacks
- Alert history
- Console handler

Files:
- sigil/telemetry/alerts.py


TASK 2.5: Unit Tests (Day 4-5)
------------------------------
Priority: P0
Effort: 6 hours
Dependencies: Tasks 2.1-2.4

Deliverables:
- Token counter tests
- Tracker tests
- Budget tests
- Alert tests

Files:
- sigil/tests/telemetry/test_token_counting.py
- sigil/tests/telemetry/test_tokens.py
- sigil/tests/telemetry/test_budget.py
- sigil/tests/telemetry/test_alerts.py


TASK 2.6: Integration Verification (Day 5)
------------------------------------------
Priority: P0
Effort: 4 hours
Dependencies: Tasks 2.1-2.5

Deliverables:
- Token tracking integration test
- Logger integration
- Accuracy verification

Files:
- sigil/tests/telemetry/test_integration_tokens.py
```

### 4.3 Implementation Details

#### 4.3.1 Token Counter (Task 2.1)

```python
# sigil/telemetry/token_counting.py

import tiktoken
from typing import List, Dict, Optional


class TokenCounter:
    """Count tokens using tiktoken encoding."""

    def __init__(self, encoding_name: str = "cl100k_base") -> None:
        self.encoding = tiktoken.get_encoding(encoding_name)

    def count_text(self, text: str) -> int:
        """Count tokens in text."""
        if not text:
            return 0
        return len(self.encoding.encode(text))

    def count_messages(self, messages: List[Dict[str, str]]) -> int:
        """Count tokens in message list."""
        total = 0
        for msg in messages:
            total += 4  # Message overhead
            total += self.count_text(msg.get("content", ""))
        total += 3  # Conversation overhead
        return total


# Singleton instance
_counter: Optional[TokenCounter] = None


def get_counter() -> TokenCounter:
    """Get or create singleton counter."""
    global _counter
    if _counter is None:
        _counter = TokenCounter()
    return _counter


def count_tokens(text: str) -> int:
    """Count tokens in text string."""
    return get_counter().count_text(text)
```

#### 4.3.2 Token Tracker (Task 2.2)

Reference implementation in Section 5 of cli-token-budgeting.md.

#### 4.3.3 Budget Manager (Task 2.3)

Reference implementation in Section 3 of cli-token-budgeting.md.

#### 4.3.4 Alert System (Task 2.4)

Reference implementation in Section 7 of cli-token-budgeting.md.

### 4.4 Phase 2 Checklist

```
+-----------------------------------------------------------------------------+
|                         PHASE 2 COMPLETION CHECKLIST                         |
+-----------------------------------------------------------------------------+

[ ] TokenCounter accurately counts text tokens
[ ] TokenCounter handles messages with overhead
[ ] TokenTracker tracks per-component usage
[ ] TokenTracker records operation history
[ ] BudgetManager allocates component budgets
[ ] BudgetManager forecasts usage
[ ] BudgetManager detects threshold crossings
[ ] AlertManager triggers on threshold changes
[ ] AlertManager supports multiple handlers
[ ] AlertManager tracks history
[ ] Integration with ExecutionLogger works
[ ] Unit tests pass (>90% coverage)
[ ] Token accuracy < 1% deviation
[ ] Documentation updated
```

---

## 5. Phase 3: Monitor Implementation

### 5.1 Overview

**Duration:** Week 3 (5 days)
**Goal:** Implement real-time monitoring display

### 5.2 Tasks

```
+-----------------------------------------------------------------------------+
|                         PHASE 3 TASK BREAKDOWN                               |
+-----------------------------------------------------------------------------+

TASK 3.1: File Tailer (Day 1)
-----------------------------
Priority: P0
Effort: 6 hours
Dependencies: Phase 1 complete

Deliverables:
- FileTailer class
- Position tracking
- Rotation handling
- Efficient reading

Files:
- sigil/telemetry/log_parser.py (partial)


TASK 3.2: Log Parser (Day 1-2)
------------------------------
Priority: P0
Effort: 4 hours
Dependencies: Task 3.1

Deliverables:
- Line parsing
- Error handling
- Batch processing

Files:
- sigil/telemetry/log_parser.py


TASK 3.3: State Accumulator (Day 2)
-----------------------------------
Priority: P0
Effort: 4 hours
Dependencies: Task 3.2

Deliverables:
- SessionState class
- Event aggregation
- Metric calculation

Files:
- sigil/interfaces/cli/monitor.py (partial)


TASK 3.4: Display Renderer (Day 3)
----------------------------------
Priority: P0
Effort: 6 hours
Dependencies: Task 3.3

Deliverables:
- Dashboard mode
- Stream mode
- Minimal mode
- Color formatting

Files:
- sigil/interfaces/cli/display.py
- sigil/interfaces/cli/budget_display.py


TASK 3.5: Monitor Script (Day 3-4)
----------------------------------
Priority: P0
Effort: 6 hours
Dependencies: Tasks 3.1-3.4

Deliverables:
- ExecutionMonitor class
- CLI entry point
- Command options
- Signal handling

Files:
- sigil/interfaces/cli/monitor.py


TASK 3.6: Unit Tests (Day 4-5)
------------------------------
Priority: P0
Effort: 6 hours
Dependencies: Tasks 3.1-3.5

Deliverables:
- Tailer tests
- Parser tests
- Display tests
- Monitor tests

Files:
- sigil/tests/cli/test_monitor.py
- sigil/tests/cli/test_display.py


TASK 3.7: Manual Testing (Day 5)
--------------------------------
Priority: P0
Effort: 4 hours
Dependencies: Tasks 3.1-3.6

Deliverables:
- End-to-end testing
- Edge case verification
- Performance validation

Files:
- Manual test script
```

### 5.3 Implementation Details

#### 5.3.1 File Tailer (Task 3.1)

```python
# sigil/telemetry/log_parser.py

import os
from pathlib import Path
from typing import List, Optional


class FileTailer:
    """
    Efficient file tailer with position tracking.

    Handles:
    - Incremental reading
    - File rotation
    - Partial lines
    """

    def __init__(self, file_path: Path) -> None:
        self.file_path = file_path
        self._position = 0
        self._inode: Optional[int] = None
        self._partial_line = ""

    def read_new_lines(self) -> List[str]:
        """Read new complete lines since last read."""
        lines = []

        if not self.file_path.exists():
            return lines

        # Check for rotation
        stat = self.file_path.stat()
        if self._inode is not None and stat.st_ino != self._inode:
            self._position = 0
            self._partial_line = ""
        self._inode = stat.st_ino

        # Check if file has shrunk (truncated)
        if stat.st_size < self._position:
            self._position = 0

        # Read new content
        with open(self.file_path, 'r', encoding='utf-8') as f:
            f.seek(self._position)
            content = self._partial_line + f.read()
            self._position = f.tell()

        # Split into lines
        if content:
            parts = content.split('\n')
            # Last part might be incomplete
            self._partial_line = parts[-1]
            # Complete lines
            lines = [p for p in parts[:-1] if p.strip()]

        return lines

    def reset(self) -> None:
        """Reset to beginning of file."""
        self._position = 0
        self._partial_line = ""
```

#### 5.3.2 Monitor Script (Task 3.5)

Reference implementation in Section 4.3 of cli-architecture.md.

### 5.4 Phase 3 Checklist

```
+-----------------------------------------------------------------------------+
|                         PHASE 3 COMPLETION CHECKLIST                         |
+-----------------------------------------------------------------------------+

[ ] FileTailer tracks position correctly
[ ] FileTailer handles rotation
[ ] FileTailer handles truncation
[ ] LogParser parses valid JSON lines
[ ] LogParser skips malformed lines
[ ] SessionState accumulates tokens correctly
[ ] SessionState tracks component metrics
[ ] Dashboard display renders correctly
[ ] Stream display shows events
[ ] Minimal display shows token count
[ ] Monitor CLI accepts options
[ ] Monitor handles Ctrl+C gracefully
[ ] Monitor reconnects on file changes
[ ] Update rate meets 10 Hz target
[ ] Unit tests pass (>90% coverage)
[ ] Manual testing successful
[ ] Documentation updated
```

---

## 6. Phase 4: Integration and Polish

### 6.1 Overview

**Duration:** Week 4 (5 days)
**Goal:** Integrate with CLI and finalize

### 6.2 Tasks

```
+-----------------------------------------------------------------------------+
|                         PHASE 4 TASK BREAKDOWN                               |
+-----------------------------------------------------------------------------+

TASK 4.1: Orchestrator Integration (Day 1)
------------------------------------------
Priority: P0
Effort: 6 hours
Dependencies: Phases 1-3 complete

Deliverables:
- Modify Orchestrator to use ExecutionLogger
- Add token tracking to pipeline steps
- Add budget checking

Files:
- sigil/interfaces/cli/orchestrator.py


TASK 4.2: CLI Command Integration (Day 1-2)
-------------------------------------------
Priority: P0
Effort: 4 hours
Dependencies: Task 4.1

Deliverables:
- Add --session option
- Add --log-path option
- Add monitoring commands

Files:
- sigil/interfaces/cli/app.py
- sigil/interfaces/cli/commands/orchestrate.py


TASK 4.3: Log Rotation (Day 2)
------------------------------
Priority: P1
Effort: 4 hours
Dependencies: Phase 1 complete

Deliverables:
- LogRotator class
- Size-based rotation
- Archival

Files:
- sigil/telemetry/log_rotation.py


TASK 4.4: Integration Testing (Day 3)
-------------------------------------
Priority: P0
Effort: 6 hours
Dependencies: Tasks 4.1-4.3

Deliverables:
- End-to-end tests
- Two-terminal tests
- Performance tests

Files:
- sigil/tests/integration/test_monitoring.py


TASK 4.5: Documentation (Day 4)
-------------------------------
Priority: P1
Effort: 6 hours
Dependencies: Tasks 4.1-4.4

Deliverables:
- User guide updates
- API documentation
- Troubleshooting guide

Files:
- docs/cli-monitoring-guide.md (update)
- docs/api/telemetry.md


TASK 4.6: Release Preparation (Day 5)
-------------------------------------
Priority: P0
Effort: 4 hours
Dependencies: Tasks 4.1-4.5

Deliverables:
- Version bump
- Changelog
- Release notes
- Final testing

Files:
- pyproject.toml
- CHANGELOG.md
- docs/release-notes/v2.1.0.md
```

### 6.3 Implementation Details

#### 6.3.1 Orchestrator Integration (Task 4.1)

```python
# sigil/interfaces/cli/orchestrator.py (modifications)

from sigil.telemetry.execution_logger import ExecutionLogger
from sigil.telemetry.tokens import TokenTracker
from sigil.telemetry.log_schema import EventType, Component


class Orchestrator:
    """Pipeline orchestrator with monitoring integration."""

    def __init__(
        self,
        session_id: str,
        token_budget: int = 256_000,
        log_path: str = None,
    ) -> None:
        self.session_id = session_id
        self.token_budget = token_budget

        # Initialize monitoring components
        self.logger = ExecutionLogger(
            session_id=session_id,
            log_path=log_path,
        )
        self.tracker = TokenTracker(
            session_id=session_id,
            total_budget=token_budget,
            logger=self.logger,
        )

    async def execute(self, task: str, context: dict = None) -> dict:
        """Execute pipeline with monitoring."""
        async with self.logger.session():
            # Log session start
            await self.logger.log_event(
                EventType.SESSION_START,
                Component.ORCHESTRATOR,
                f"Starting pipeline for: {task}",
                budget=self.token_budget,
                task=task,
            )

            try:
                # Execute pipeline steps with tracking
                result = await self._execute_pipeline(task, context)

                # Log completion
                await self.logger.log_event(
                    EventType.SESSION_COMPLETE,
                    Component.ORCHESTRATOR,
                    "Pipeline completed successfully",
                    total_tokens=self.tracker.budget.used,
                )

                return result

            except Exception as e:
                await self.logger.log_event(
                    EventType.SESSION_ERROR,
                    Component.ORCHESTRATOR,
                    f"Pipeline failed: {str(e)}",
                    error_type=type(e).__name__,
                )
                raise

    async def _execute_step(
        self,
        step_name: str,
        component: Component,
        executor: callable,
        *args,
        **kwargs,
    ) -> tuple:
        """Execute a step with logging and tracking."""
        import time

        start = time.time()

        # Log step start
        await self.logger.log_event(
            EventType.STEP_START,
            component,
            f"Starting {step_name}",
        )

        try:
            # Execute
            result, tokens = await executor(*args, **kwargs)

            # Track tokens
            await self.tracker.track(
                component=component.value,
                tokens=tokens,
                operation_type=step_name,
            )

            # Log completion
            latency_ms = (time.time() - start) * 1000
            await self.logger.log_event(
                EventType.STEP_COMPLETE,
                component,
                f"{step_name} completed",
                tokens=tokens,
                latency_ms=latency_ms,
            )

            return result, tokens

        except Exception as e:
            await self.logger.log_event(
                EventType.STEP_ERROR,
                component,
                f"{step_name} failed: {str(e)}",
                error=str(e),
            )
            raise
```

#### 6.3.2 CLI Command Integration (Task 4.2)

```python
# sigil/interfaces/cli/commands/orchestrate.py (modifications)

import click
import asyncio
from sigil.interfaces.cli.orchestrator import Orchestrator


@click.command()
@click.option('--task', '-t', required=True, help='Task to execute')
@click.option('--session', '-s', required=True, help='Session ID for monitoring')
@click.option('--budget', '-b', default=256000, help='Token budget')
@click.option('--log-path', '-l', default=None, help='Log directory path')
def orchestrate(task: str, session: str, budget: int, log_path: str):
    """Execute a task through the pipeline."""

    click.echo(f"Processing request: {task}")
    click.echo(f"Session: {session}")
    click.echo("Pipeline starting...")

    orchestrator = Orchestrator(
        session_id=session,
        token_budget=budget,
        log_path=log_path,
    )

    result = asyncio.run(orchestrator.execute(task))

    click.echo("\n" + "=" * 40)
    click.echo("EXECUTION COMPLETE")
    click.echo("=" * 40)
    click.echo(f"Result: {result.get('summary', 'Complete')}")
    click.echo(f"Final Tokens: {result.get('tokens_used', 0):,} / {budget:,}")
```

### 6.4 Phase 4 Checklist

```
+-----------------------------------------------------------------------------+
|                         PHASE 4 COMPLETION CHECKLIST                         |
+-----------------------------------------------------------------------------+

[ ] Orchestrator uses ExecutionLogger
[ ] Orchestrator tracks tokens per step
[ ] Orchestrator checks budget limits
[ ] CLI accepts --session option
[ ] CLI accepts --log-path option
[ ] Monitor command works
[ ] Log rotation implemented
[ ] Archive retention works
[ ] Integration tests pass
[ ] Two-terminal workflow works
[ ] Performance targets met
[ ] Documentation complete
[ ] Release notes written
[ ] Version bumped
[ ] Final testing complete
```

---

## 7. Testing Strategy

### 7.1 Test Categories

```
+-----------------------------------------------------------------------------+
|                           TEST CATEGORIES                                    |
+-----------------------------------------------------------------------------+

CATEGORY 1: Unit Tests
----------------------
Coverage target: >90%
Scope: Individual functions and classes
Mocking: External dependencies mocked

Files:
- test_log_schema.py
- test_execution_logger.py
- test_buffer.py
- test_token_counting.py
- test_tokens.py
- test_budget.py
- test_alerts.py
- test_monitor.py
- test_display.py


CATEGORY 2: Integration Tests
-----------------------------
Coverage target: >80%
Scope: Component interactions
Mocking: Minimal, real file I/O

Files:
- test_integration_logging.py
- test_integration_tokens.py
- test_integration_monitoring.py


CATEGORY 3: End-to-End Tests
----------------------------
Coverage target: Critical paths
Scope: Full system behavior
Mocking: None

Files:
- test_e2e_cli_monitor.py


CATEGORY 4: Performance Tests
-----------------------------
Coverage target: Key operations
Scope: Latency and throughput

Files:
- test_performance.py
```

### 7.2 Test Cases

```
+-----------------------------------------------------------------------------+
|                         KEY TEST CASES                                       |
+-----------------------------------------------------------------------------+

LOGGING TESTS:
- Log entry serialization round-trip
- Logger creates directories
- Logger writes to correct file
- Buffer flushes on size limit
- Buffer flushes on time limit
- Async operations concurrent safe
- Context manager cleans up

TOKEN TESTS:
- Token counter accurate for text
- Token counter handles empty input
- Tracker accumulates correctly
- Tracker per-component tracking
- Budget manager allocation
- Budget manager thresholds
- Alert triggers on threshold

MONITOR TESTS:
- File tailer reads new lines
- File tailer handles rotation
- Parser skips malformed lines
- State accumulates correctly
- Display renders all modes
- Monitor CLI options work
- Monitor handles Ctrl+C

INTEGRATION TESTS:
- Logger + Tracker integration
- Monitor reads logger output
- Full pipeline logging
- Two-terminal workflow
```

### 7.3 Test Commands

```bash
# Run all tests
pytest sigil/tests/telemetry/ -v

# Run with coverage
pytest sigil/tests/telemetry/ --cov=sigil/telemetry --cov-report=html

# Run specific category
pytest sigil/tests/telemetry/test_execution_logger.py -v

# Run integration tests
pytest sigil/tests/integration/ -v --timeout=60

# Run performance tests
pytest sigil/tests/performance/ -v --benchmark-only

# Run end-to-end tests
pytest sigil/tests/e2e/ -v --timeout=120
```

---

## 8. Rollout Plan

### 8.1 Rollout Phases

```
+-----------------------------------------------------------------------------+
|                           ROLLOUT PHASES                                     |
+-----------------------------------------------------------------------------+

PHASE 1: Internal Testing (Week 4, Days 1-2)
--------------------------------------------
Audience: Development team
Environment: Local development
Features: All
Feedback: Direct communication

Actions:
- Deploy to dev environments
- Run full test suite
- Manual testing
- Fix critical issues


PHASE 2: Beta Release (Week 4, Day 3)
-------------------------------------
Audience: Internal users
Environment: Staging
Features: All
Feedback: Issue tracker

Actions:
- Deploy to staging
- Announce beta availability
- Monitor for issues
- Collect feedback


PHASE 3: Production Release (Week 4, Day 5)
-------------------------------------------
Audience: All users
Environment: Production
Features: All
Feedback: Issue tracker, support

Actions:
- Version bump to 2.1.0
- Deploy to production
- Publish release notes
- Monitor adoption
```

### 8.2 Rollback Plan

```
+-----------------------------------------------------------------------------+
|                           ROLLBACK PLAN                                      |
+-----------------------------------------------------------------------------+

TRIGGER CONDITIONS:
- Critical bug affecting >10% of users
- Performance degradation >20%
- Data corruption issues
- Security vulnerability

ROLLBACK STEPS:

1. Immediate (< 1 hour):
   - Revert to previous version
   - Disable monitoring features via flag
   - Notify affected users

2. Short-term (< 24 hours):
   - Identify root cause
   - Develop fix
   - Test fix in staging
   - Plan re-deployment

3. Re-deployment:
   - Deploy fix to staging
   - Verify fix
   - Deploy to production
   - Monitor closely

FEATURE FLAGS:
- SIGIL_ENABLE_MONITORING=true/false
- SIGIL_ENABLE_TOKEN_TRACKING=true/false
- SIGIL_ENABLE_BUDGET_ALERTS=true/false
```

---

## 9. Backwards Compatibility

### 9.1 Compatibility Matrix

```
+-----------------------------------------------------------------------------+
|                      BACKWARDS COMPATIBILITY MATRIX                          |
+-----------------------------------------------------------------------------+

Component          | v2.0 Compatible | Changes Required | Migration Path
-------------------|-----------------|------------------|------------------
CLI Commands       | Yes             | New options added| Optional adoption
Config Files       | Yes             | New keys added   | Defaults applied
Log Format         | N/A             | New feature      | No migration
API Interfaces     | Yes             | Additions only   | Optional adoption
Python API         | Yes             | New modules      | Import if needed
```

### 9.2 Breaking Changes

```
+-----------------------------------------------------------------------------+
|                         BREAKING CHANGES                                     |
+-----------------------------------------------------------------------------+

CHANGE: None planned

All changes are additive:
- New CLI options (--session, --log-path)
- New modules (sigil/telemetry/)
- New commands (monitor)
- New configuration keys

Existing functionality unchanged:
- CLI without --session works as before
- No monitoring overhead if not used
- Existing configs remain valid
```

### 9.3 Migration Guide

```
+-----------------------------------------------------------------------------+
|                         MIGRATION GUIDE                                      |
+-----------------------------------------------------------------------------+

FOR USERS NOT USING MONITORING:
-------------------------------
No action required. Everything works as before.


FOR USERS WANTING MONITORING:
-----------------------------

Step 1: Update to v2.1.0
$ pip install --upgrade sigil

Step 2: Add session ID to CLI calls
# Before:
$ python -m sigil.cli orchestrate --task "Analyze lead"

# After:
$ python -m sigil.cli orchestrate --task "Analyze lead" --session my-session

Step 3: Start monitor in separate terminal
$ python -m sigil.cli monitor --session my-session


FOR PROGRAMMATIC USERS:
-----------------------

Step 1: Import new modules
from sigil.telemetry import ExecutionLogger, TokenTracker

Step 2: Initialize with session
logger = ExecutionLogger(session_id="my-session")
tracker = TokenTracker(session_id="my-session")

Step 3: Use in pipeline
async with logger.session():
    await logger.log_event(...)
    tracker.track(...)
```

---

## 10. Risk Management

### 10.1 Risk Register

```
+-----------------------------------------------------------------------------+
|                           RISK REGISTER                                      |
+-----------------------------------------------------------------------------+

RISK 1: Performance Degradation
-------------------------------
Likelihood: Medium
Impact: High
Mitigation:
- Async I/O for logging
- Buffered writes
- Performance testing
- Feature flag to disable
Contingency: Disable monitoring via flag


RISK 2: Disk Space Issues
-------------------------
Likelihood: Medium
Impact: Medium
Mitigation:
- Log rotation
- Size limits
- Archival
- Monitoring disk usage
Contingency: Emergency cleanup script


RISK 3: Token Count Inaccuracy
------------------------------
Likelihood: Low
Impact: Medium
Mitigation:
- Use tiktoken (accurate)
- Validation testing
- Cross-check with API usage
Contingency: Manual adjustment, improved algorithm


RISK 4: Monitor Sync Issues
---------------------------
Likelihood: Medium
Impact: Low
Mitigation:
- Clear documentation
- Session ID validation
- Error messages
Contingency: Troubleshooting guide


RISK 5: Schedule Slippage
-------------------------
Likelihood: Medium
Impact: Medium
Mitigation:
- Buffer time in schedule
- Prioritized task list
- Daily progress checks
Contingency: Reduce scope to core features
```

### 10.2 Mitigation Actions

```
+-----------------------------------------------------------------------------+
|                        MITIGATION ACTIONS                                    |
+-----------------------------------------------------------------------------+

DAILY ACTIONS:
- Review progress against plan
- Run test suite
- Check performance metrics
- Address blockers immediately

WEEKLY ACTIONS:
- Phase completion review
- Risk reassessment
- Stakeholder update
- Scope adjustment if needed

PRE-RELEASE ACTIONS:
- Full regression testing
- Performance validation
- Documentation review
- Rollback plan verification
```

---

## Document Revision History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0.0 | 2026-01-11 | Systems Architecture | Initial release |

---

*End of CLI Implementation Roadmap Document*
