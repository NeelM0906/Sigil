# Sigil v2 CLI Monitoring User Guide

## Document Information

| Field | Value |
|-------|-------|
| Version | 1.0.0 |
| Status | User Guide |
| Created | 2026-01-11 |
| Author | Systems Architecture Team |
| Audience | Developers, Operators |

---

## Table of Contents

1. [Getting Started](#1-getting-started)
2. [Running CLI and Monitor Together](#2-running-cli-and-monitor-together)
3. [Understanding the Monitor Display](#3-understanding-the-monitor-display)
4. [Reading and Interpreting Logs](#4-reading-and-interpreting-logs)
5. [Common Issues and Debugging](#5-common-issues-and-debugging)
6. [Performance Optimization](#6-performance-optimization)
7. [Log File Management](#7-log-file-management)
8. [Advanced Usage](#8-advanced-usage)
9. [Troubleshooting Reference](#9-troubleshooting-reference)
10. [Best Practices](#10-best-practices)

---

## 1. Getting Started

### 1.1 Prerequisites

Before using the monitoring system, ensure you have:

```
REQUIREMENTS:
- Python 3.10 or higher
- Sigil v2 installed and configured
- Two terminal windows/tabs available
- Write access to ~/.sigil/logs/
```

### 1.2 Quick Start

The monitoring system uses two terminals: one for the CLI execution and one for the monitor.

**Terminal 1 (CLI):**
```bash
# Navigate to project
cd /Users/zidane/Bland-Agents-Dataset/acti-agent-builder

# Activate virtual environment
source venv/bin/activate

# Run CLI with a specific session ID
python -m sigil.interfaces.cli.app orchestrate \
    --task "Analyze Acme Corp lead" \
    --session my-session-001
```

**Terminal 2 (Monitor):**
```bash
# Navigate to project
cd /Users/zidane/Bland-Agents-Dataset/acti-agent-builder

# Activate virtual environment
source venv/bin/activate

# Start monitoring the same session
python -m sigil.interfaces.cli.monitor \
    --session my-session-001
```

### 1.3 Session Naming Conventions

Use meaningful session names for easy tracking:

```
RECOMMENDED NAMING PATTERNS:

Format: <type>-<identifier>-<timestamp>

Examples:
  lead-analysis-20260111-001     # Lead analysis task
  agent-build-acme-001           # Agent building for Acme
  research-market-2026q1-003     # Q1 market research
  test-routing-debug             # Test session for debugging

AVOID:
  test                           # Too generic
  session1                       # Not descriptive
  asdf                          # Meaningless
```

---

## 2. Running CLI and Monitor Together

### 2.1 Step-by-Step Workflow

```
+-----------------------------------------------------------------------------+
|                        MONITORING WORKFLOW                                   |
+-----------------------------------------------------------------------------+

STEP 1: Open Two Terminal Windows
---------------------------------
+-------------------+                    +-------------------+
| Terminal 1        |                    | Terminal 2        |
| (CLI Execution)   |                    | (Monitor Display) |
+-------------------+                    +-------------------+


STEP 2: Start Monitor First (Optional but Recommended)
------------------------------------------------------
# Terminal 2
$ python -m sigil.interfaces.cli.monitor --session my-session

Output:
========== SIGIL v2 EXECUTION MONITOR ==========
Session: my-session
Status: Waiting for session to start...
--------------------------------------------------


STEP 3: Start CLI Execution
---------------------------
# Terminal 1
$ python -m sigil.interfaces.cli.app orchestrate \
    --task "Analyze lead John Smith from Acme Corp" \
    --session my-session

Output:
Processing request: Analyze lead John Smith from Acme Corp
Session: my-session
Pipeline starting...


STEP 4: Observe Real-Time Updates (Monitor Terminal)
----------------------------------------------------
========== SIGIL v2 EXECUTION MONITOR ==========
Session: my-session
Status: In Progress...

[10:15:32] Routing... 50 tokens
[10:15:33] Memory retrieval... 150 tokens (Total: 200)
[10:15:34] Planning... 0 tokens (Total: 200)
[10:15:36] Reasoning... 450 tokens (Total: 650)
[10:15:37] Validation... 0 tokens (Total: 650)

Budget Status:
  Used:      650 / 256,000 (0.25%)
  Remaining: 255,350 tokens


STEP 5: View Completion (Both Terminals)
----------------------------------------
# Terminal 1 (CLI)
========== EXECUTION COMPLETE ==========
Result: Lead analysis completed
Quality Score: 0.92
Recommended Action: Schedule discovery call

Final Tokens: 650 / 256,000 (0.25%)
Total Time: 5.1 seconds

# Terminal 2 (Monitor)
========== EXECUTION COMPLETE ==========
Session: my-session completed successfully
Duration: 5.1 seconds
Final token usage: 650 / 256,000 (0.25%)
```

### 2.2 Starting Monitor After CLI

You can start the monitor after the CLI has begun execution:

```bash
# CLI already running with session "my-session"

# Start monitor - it will catch up by reading existing logs
python -m sigil.interfaces.cli.monitor --session my-session

# Monitor reads all existing events and displays current state
```

The monitor will:
1. Find the log file for the session
2. Read all existing entries
3. Calculate current totals
4. Display accumulated state
5. Continue watching for new events

### 2.3 Monitor Command Options

```bash
python -m sigil.interfaces.cli.monitor [OPTIONS]

OPTIONS:
  --session, -s TEXT     Session ID to monitor (required)
  --mode, -m TEXT        Display mode: dashboard, stream, minimal
                         Default: dashboard
  --log-path, -l TEXT    Custom log directory path
                         Default: ~/.sigil/logs
  --refresh-rate FLOAT   Display refresh rate in seconds
                         Default: 0.1 (10 updates/sec)

EXAMPLES:

# Full dashboard mode (default)
python -m sigil.interfaces.cli.monitor -s my-session

# Streaming mode (simple event list)
python -m sigil.interfaces.cli.monitor -s my-session -m stream

# Minimal mode (just token count)
python -m sigil.interfaces.cli.monitor -s my-session -m minimal

# Custom log path
python -m sigil.interfaces.cli.monitor -s my-session -l /var/log/sigil

# Slower refresh for remote terminals
python -m sigil.interfaces.cli.monitor -s my-session --refresh-rate 0.5
```

---

## 3. Understanding the Monitor Display

### 3.1 Dashboard Mode (Default)

The dashboard provides comprehensive real-time information:

```
+==============================================================+
|                    TOKEN BUDGET STATUS                        |
+==============================================================+
| Session: lead-analysis-001                                    |
| Status: RUNNING                                               |
+--------------------------------------------------------------+
|                                                               |
| COMPONENT USAGE:                                              |
|                                                               |
| Routing    [====                    ]   50 / 10,000   (0.5%)  |
| Memory     [========                ]  150 / 30,000   (0.5%)  |
| Planning   [                        ]    0 / 20,000   (0.0%)  |
| Reasoning  [==                      ]  450 / 150,000  (0.3%)  |
| Contracts  [                        ]    0 / 10,000   (0.0%)  |
| Validation [                        ]    0 / 10,000   (0.0%)  |
|                                                               |
+--------------------------------------------------------------+
| BUDGET SUMMARY:                                               |
|                                                               |
| Total:     [##                                ]  650 / 256,000|
| Used:      0.25%                                              |
| Remaining: 255,350 tokens                                     |
| Status:    NORMAL                                             |
|                                                               |
+--------------------------------------------------------------+
| PERFORMANCE:                                                  |
|                                                               |
| Routing:    1.2s                                              |
| Memory:     1.5s (SLOWEST)                                    |
| Planning:   0.1s                                              |
| Reasoning:  2.1s                                              |
| Total:      4.9s                                              |
|                                                               |
+--------------------------------------------------------------+
| RECENT EVENTS:                                                |
|                                                               |
| [10:15:32] routing: Intent classified as lead_analysis        |
| [10:15:33] memory: Retrieved 5 relevant memories              |
| [10:15:34] planning: Plan created with 3 steps                |
| [10:15:36] reasoning: Analysis complete                       |
| [10:15:37] validation: Output validated                       |
|                                                               |
+==============================================================+
```

**Section Breakdown:**

| Section | Description |
|---------|-------------|
| Header | Session ID and current status |
| Component Usage | Per-component token consumption with progress bars |
| Budget Summary | Overall budget status with visual indicator |
| Performance | Latency for each component, highlights slowest |
| Recent Events | Last 5 events with timestamps |

### 3.2 Stream Mode

Stream mode shows a simple list of events as they occur:

```
========== SIGIL v2 EXECUTION MONITOR ==========
Session: lead-analysis-001
Mode: Stream
--------------------------------------------------

[10:15:32.123] routing    | Intent classified          |    50 tokens | Total:     50
[10:15:33.456] memory     | Retrieved 5 memories       |   150 tokens | Total:    200
[10:15:34.012] planning   | Plan created (3 steps)     |     0 tokens | Total:    200
[10:15:36.234] reasoning  | Analysis complete          |   450 tokens | Total:    650
[10:15:37.567] validation | Output validated           |     0 tokens | Total:    650

--------------------------------------------------
Status: Running | Tokens: 650 / 256,000 (0.25%)
```

**Use Stream Mode When:**
- Logging to a file
- Running in CI/CD pipelines
- Debugging specific events
- Limited terminal width

### 3.3 Minimal Mode

Minimal mode shows just the essential token information:

```
Tokens: 650 / 256,000 (0.25%) [##.............................] 255,350 left
```

**Use Minimal Mode When:**
- Screen space is limited
- Running multiple monitors
- Only need token count
- Embedded in other scripts

### 3.4 Status Indicators

```
STATUS VALUES:

+-------------+------------------------------------------+
| Status      | Meaning                                  |
+-------------+------------------------------------------+
| WAITING     | Monitor ready, CLI not started yet       |
| RUNNING     | Pipeline actively executing              |
| COMPLETE    | Session finished successfully            |
| ERROR       | Session ended with an error              |
+-------------+------------------------------------------+


WARNING LEVELS (Budget):

+-------------+------------------------------------------+
| Level       | Meaning                                  |
+-------------+------------------------------------------+
| NORMAL      | Budget usage < 50%, all good             |
| ELEVATED    | Budget usage 50-80%, monitor closely     |
| CRITICAL    | Budget usage 80-95%, complete soon       |
| EXCEEDED    | Budget usage > 95%, new ops blocked      |
+-------------+------------------------------------------+


COLOR CODING:

Green  : Normal status, on track
Yellow : Warning, needs attention
Red    : Critical, immediate action needed
```

---

## 4. Reading and Interpreting Logs

### 4.1 Log File Location

```
LOG FILE STRUCTURE:

~/.sigil/logs/
+-- sessions/
|   +-- lead-analysis-001/
|   |   +-- execution.jsonl    # Main execution log
|   |   +-- tokens.json        # Final token summary
|   |   +-- metrics.json       # Performance metrics
|   |
|   +-- agent-build-acme/
|       +-- execution.jsonl
|       +-- tokens.json
|       +-- metrics.json
|
+-- archive/
    +-- 2026-01-10/
        +-- old-session.tar.gz
```

### 4.2 Log Entry Format

Each line in `execution.jsonl` is a JSON object:

```json
{
  "timestamp": "2026-01-11T10:15:32.123456",
  "session_id": "lead-analysis-001",
  "event_type": "step_complete",
  "component": "routing",
  "message": "Intent classified as lead_analysis",
  "tokens": 50,
  "metadata": {
    "latency_ms": 1200,
    "intent": "lead_analysis",
    "confidence": 0.95
  }
}
```

### 4.3 Event Types

```
EVENT TYPE REFERENCE:

+-------------------+------------------------------------------+
| Event Type        | Description                              |
+-------------------+------------------------------------------+
| session_start     | Session has begun                        |
| session_complete  | Session finished successfully            |
| session_error     | Session ended with error                 |
| step_start        | A pipeline step is starting              |
| step_complete     | A pipeline step completed                |
| step_error        | A pipeline step failed                   |
| token_consumption | Token usage update (detailed)            |
| budget_warning    | Budget threshold crossed                 |
| budget_exceeded   | Budget limit reached                     |
| retry_attempt     | Retrying a failed operation              |
+-------------------+------------------------------------------+
```

### 4.4 Reading Logs Manually

```bash
# View recent entries
tail -f ~/.sigil/logs/sessions/my-session/execution.jsonl

# Pretty print single entry
cat ~/.sigil/logs/sessions/my-session/execution.jsonl | head -1 | python -m json.tool

# Filter by component
cat ~/.sigil/logs/sessions/my-session/execution.jsonl | \
  grep '"component":"reasoning"'

# Calculate total tokens
cat ~/.sigil/logs/sessions/my-session/execution.jsonl | \
  python -c "import sys,json; print(sum(json.loads(l).get('tokens',0) for l in sys.stdin))"

# Find errors
cat ~/.sigil/logs/sessions/my-session/execution.jsonl | \
  grep '"event_type":".*error"'

# Get session duration
cat ~/.sigil/logs/sessions/my-session/execution.jsonl | \
  python -c "
import sys, json
from datetime import datetime
lines = [json.loads(l) for l in sys.stdin]
if len(lines) >= 2:
    start = datetime.fromisoformat(lines[0]['timestamp'])
    end = datetime.fromisoformat(lines[-1]['timestamp'])
    print(f'Duration: {(end-start).total_seconds():.1f}s')
"
```

### 4.5 Interpreting Patterns

```
HEALTHY PATTERNS:

+--------------------------------------------------------------+
| Pattern: Sequential component execution                       |
| Log shows: routing -> memory -> planning -> reasoning -> val |
| Meaning: Normal pipeline flow                                 |
+--------------------------------------------------------------+

+--------------------------------------------------------------+
| Pattern: Low token usage per component                        |
| Log shows: routing=50, memory=150, reasoning=500              |
| Meaning: Efficient operation                                  |
+--------------------------------------------------------------+

+--------------------------------------------------------------+
| Pattern: Consistent latencies                                 |
| Log shows: All steps < 3 seconds                              |
| Meaning: No bottlenecks                                       |
+--------------------------------------------------------------+


WARNING PATTERNS:

+--------------------------------------------------------------+
| Pattern: High reasoning tokens                                |
| Log shows: reasoning=15000+ per operation                     |
| Meaning: Complex task or verbose context                      |
| Action: Review reasoning strategy selection                   |
+--------------------------------------------------------------+

+--------------------------------------------------------------+
| Pattern: Multiple retry events                                |
| Log shows: retry_attempt events                               |
| Meaning: Transient failures occurring                         |
| Action: Check network, API status                             |
+--------------------------------------------------------------+

+--------------------------------------------------------------+
| Pattern: Memory taking long time                              |
| Log shows: memory latency > 5 seconds                         |
| Meaning: Large retrieval or LLM reading                       |
| Action: Limit RAG results, optimize queries                   |
+--------------------------------------------------------------+


ERROR PATTERNS:

+--------------------------------------------------------------+
| Pattern: session_error after routing                          |
| Log shows: routing -> session_error                           |
| Meaning: Intent classification failed                         |
| Action: Review task description clarity                       |
+--------------------------------------------------------------+

+--------------------------------------------------------------+
| Pattern: budget_exceeded mid-reasoning                        |
| Log shows: budget_warning -> budget_exceeded -> step_error    |
| Meaning: Task too complex for budget                          |
| Action: Break into smaller tasks                              |
+--------------------------------------------------------------+
```

---

## 5. Common Issues and Debugging

### 5.1 Issue: Monitor Not Showing Updates

```
SYMPTOM:
Monitor displays "Waiting for session..." but CLI is running

CAUSES AND SOLUTIONS:

1. Session ID Mismatch
   ----------------------
   Check: Compare session IDs in both terminals
   Fix: Use exact same session ID

   # CLI
   python -m sigil.interfaces.cli.app orchestrate --session MY-SESSION

   # Monitor (case sensitive!)
   python -m sigil.interfaces.cli.monitor --session MY-SESSION

2. Log Path Different
   -------------------
   Check: Verify log paths match
   Fix: Use same --log-path in both commands

   # CLI
   python -m sigil.interfaces.cli.app orchestrate --session s1 --log-path /custom/path

   # Monitor
   python -m sigil.interfaces.cli.monitor --session s1 --log-path /custom/path

3. Permission Issues
   ------------------
   Check: ls -la ~/.sigil/logs/
   Fix: chmod -R 755 ~/.sigil/logs/

4. Log File Not Created
   ---------------------
   Check: ls ~/.sigil/logs/sessions/MY-SESSION/
   Fix: Ensure CLI has write permissions

5. CLI Not Writing Logs
   ---------------------
   Check: CLI error messages
   Fix: Review CLI configuration
```

### 5.2 Issue: Token Counts Don't Match

```
SYMPTOM:
Monitor shows different total than CLI final output

CAUSES AND SOLUTIONS:

1. Buffer Not Flushed
   -------------------
   Cause: CLI exited before log buffer flushed
   Fix: Ensure proper CLI shutdown (Ctrl+C or completion)

2. Monitor Started Late
   --------------------
   Cause: Monitor missed early events
   Fix: Restart monitor to reread full log

   # Stop monitor (Ctrl+C)
   # Restart to reread from beginning
   python -m sigil.interfaces.cli.monitor --session my-session

3. Log Parsing Errors
   -------------------
   Cause: Malformed JSON lines
   Fix: Check for truncated lines

   # Find malformed lines
   cat ~/.sigil/logs/sessions/my-session/execution.jsonl | \
     python -c "
import sys, json
for i, line in enumerate(sys.stdin):
    try:
        json.loads(line)
    except:
        print(f'Line {i}: {line[:50]}...')
"

4. Race Condition
   ---------------
   Cause: CLI and monitor reading/writing simultaneously
   Fix: Wait a moment after CLI completion, then check monitor
```

### 5.3 Issue: High CPU Usage in Monitor

```
SYMPTOM:
Monitor process using >50% CPU

CAUSES AND SOLUTIONS:

1. Refresh Rate Too High
   ----------------------
   Cause: Updating display too frequently
   Fix: Lower refresh rate

   python -m sigil.interfaces.cli.monitor -s my-session --refresh-rate 0.5

2. Large Log File
   ---------------
   Cause: Re-reading large file
   Fix: Archive old sessions

   python -m sigil.cli.tools logs archive

3. Complex Display Rendering
   -------------------------
   Cause: Dashboard mode overhead
   Fix: Use minimal mode

   python -m sigil.interfaces.cli.monitor -s my-session -m minimal

4. Busy Loop
   ----------
   Cause: Polling too aggressively
   Fix: Increase sleep interval in monitor config
```

### 5.4 Issue: Monitor Crashes

```
SYMPTOM:
Monitor exits unexpectedly

CAUSES AND SOLUTIONS:

1. Memory Exhaustion
   ------------------
   Cause: Too many events in memory
   Fix: Monitor has built-in limits, but for very long sessions:

   # Start fresh session
   python -m sigil.interfaces.cli.monitor -s new-session

2. File Rotation During Read
   --------------------------
   Cause: Log rotated while monitor reading
   Fix: Monitor handles this, but if issues persist:

   # Disable rotation during critical sessions
   # Or use streaming mode which handles rotation better
   python -m sigil.interfaces.cli.monitor -s my-session -m stream

3. Terminal Resize
   ----------------
   Cause: Terminal resized during dashboard render
   Fix: Use stream or minimal mode in resizable environments
```

### 5.5 Debugging Checklist

```
+-----------------------------------------------------------------------------+
|                        DEBUGGING CHECKLIST                                   |
+-----------------------------------------------------------------------------+

[ ] 1. Verify session IDs match exactly (case sensitive)

[ ] 2. Check log directory exists and is writable
      ls -la ~/.sigil/logs/sessions/

[ ] 3. Verify log file is being created
      ls -la ~/.sigil/logs/sessions/YOUR-SESSION/

[ ] 4. Check log file has content
      wc -l ~/.sigil/logs/sessions/YOUR-SESSION/execution.jsonl

[ ] 5. Verify JSON format is valid
      head -1 ~/.sigil/logs/sessions/YOUR-SESSION/execution.jsonl | python -m json.tool

[ ] 6. Check for permission errors in CLI output

[ ] 7. Try stream mode to isolate display issues
      python -m sigil.interfaces.cli.monitor -s YOUR-SESSION -m stream

[ ] 8. Check system resources
      top -l 1 | grep python

[ ] 9. Review CLI error logs
      cat ~/.sigil/logs/error.log | tail -20

[ ] 10. Test with a fresh session
       python -m sigil.interfaces.cli.monitor -s test-$(date +%s)
```

---

## 6. Performance Optimization

### 6.1 Reducing Token Consumption

```
OPTIMIZATION STRATEGIES:

1. Choose Appropriate Reasoning Strategy
   -------------------------------------
   Task Complexity -> Strategy

   Simple query      -> Direct (100-500 tokens)
   Moderate analysis -> Chain-of-thought (500-2K tokens)
   Complex decision  -> Tree-of-thoughts (2K-10K tokens)

   Example:
   # For simple tasks, force direct strategy
   python -m sigil.interfaces.cli.app orchestrate \
     --task "What is John's budget?" \
     --strategy direct

2. Limit Memory Retrieval
   -----------------------
   # Default retrieves up to 20 memories
   # Reduce for simple queries
   python -m sigil.interfaces.cli.app orchestrate \
     --task "Analyze lead" \
     --memory-limit 5

3. Use Caching
   ------------
   # Enable routing cache for repeated patterns
   python -m sigil.interfaces.cli.app orchestrate \
     --task "Qualify lead" \
     --enable-cache

4. Compress Context
   -----------------
   # Summarize long conversation histories
   python -m sigil.interfaces.cli.app orchestrate \
     --task "Continue analysis" \
     --compress-history
```

### 6.2 Improving Latency

```
LATENCY OPTIMIZATION:

1. Monitor Component Latencies
   ----------------------------
   Dashboard shows per-component timing:

   Routing:    1.2s
   Memory:     1.5s (SLOWEST)  <-- Focus here
   Planning:   0.1s
   Reasoning:  2.1s

2. Memory Optimization
   --------------------
   If memory is slowest:
   - Use RAG-only retrieval for simple queries
   - Pre-compute category summaries
   - Index frequently accessed memories

3. Reasoning Optimization
   -----------------------
   If reasoning is slowest:
   - Use simpler strategy
   - Reduce context window
   - Enable streaming responses

4. Network Optimization
   ---------------------
   If all components slow:
   - Check network latency to API
   - Use regional endpoints if available
   - Consider connection pooling
```

### 6.3 Resource Management

```
RESOURCE OPTIMIZATION:

1. Monitor System Resources
   -------------------------
   # Watch memory and CPU during execution
   top -l 1 | grep -E "python|sigil"

   # Check disk I/O
   iostat 1 5

2. Limit Concurrent Sessions
   --------------------------
   # Each session uses ~50-100MB RAM
   # Recommended: max 5 concurrent sessions per machine

3. Clean Up Old Sessions
   ----------------------
   # Archive sessions older than 7 days
   python -m sigil.cli.tools logs cleanup --days 7

   # Remove archived logs older than 30 days
   python -m sigil.cli.tools logs prune --archive-days 30

4. Log Rotation
   -------------
   # Rotate logs larger than 10MB
   python -m sigil.cli.tools logs rotate --max-size 10
```

---

## 7. Log File Management

### 7.1 Log Retention Policy

```
DEFAULT RETENTION POLICY:

+-------------------+-------------------+----------------------------------+
| Log Type          | Retention Period  | Location                         |
+-------------------+-------------------+----------------------------------+
| Active sessions   | 7 days            | ~/.sigil/logs/sessions/          |
| Archived sessions | 30 days           | ~/.sigil/logs/archive/           |
| Error logs        | 90 days           | ~/.sigil/logs/error.log          |
+-------------------+-------------------+----------------------------------+
```

### 7.2 Manual Log Management

```bash
# List all sessions
ls -lt ~/.sigil/logs/sessions/

# Check session size
du -sh ~/.sigil/logs/sessions/*

# View session summary
cat ~/.sigil/logs/sessions/MY-SESSION/tokens.json | python -m json.tool

# Delete a specific session
rm -rf ~/.sigil/logs/sessions/MY-SESSION/

# Archive old sessions manually
tar -czf ~/backup/session-archive-$(date +%Y%m%d).tar.gz \
  ~/.sigil/logs/sessions/

# Clear all logs (CAUTION)
rm -rf ~/.sigil/logs/sessions/*
```

### 7.3 Automated Cleanup

```bash
# Create cleanup script
cat > ~/.sigil/cleanup.sh << 'EOF'
#!/bin/bash
# Sigil log cleanup script

LOG_DIR="$HOME/.sigil/logs"
SESSIONS_DIR="$LOG_DIR/sessions"
ARCHIVE_DIR="$LOG_DIR/archive"

# Archive sessions older than 7 days
find "$SESSIONS_DIR" -maxdepth 1 -type d -mtime +7 | while read dir; do
    session=$(basename "$dir")
    date_dir=$(date +%Y-%m-%d)
    mkdir -p "$ARCHIVE_DIR/$date_dir"
    tar -czf "$ARCHIVE_DIR/$date_dir/$session.tar.gz" -C "$SESSIONS_DIR" "$session"
    rm -rf "$dir"
    echo "Archived: $session"
done

# Remove archives older than 30 days
find "$ARCHIVE_DIR" -maxdepth 1 -type d -mtime +30 -exec rm -rf {} \;

echo "Cleanup complete"
EOF

chmod +x ~/.sigil/cleanup.sh

# Run manually
~/.sigil/cleanup.sh

# Or schedule with cron
# 0 2 * * * ~/.sigil/cleanup.sh >> ~/.sigil/cleanup.log 2>&1
```

### 7.4 Log Analysis Scripts

```bash
# Token usage report for all sessions
cat > ~/.sigil/token_report.sh << 'EOF'
#!/bin/bash
echo "Session Token Report"
echo "===================="
for dir in ~/.sigil/logs/sessions/*/; do
    session=$(basename "$dir")
    if [ -f "$dir/tokens.json" ]; then
        used=$(cat "$dir/tokens.json" | python -c "import sys,json; print(json.load(sys.stdin).get('total_used',0))")
        echo "$session: $used tokens"
    fi
done | sort -t: -k2 -n -r
EOF

chmod +x ~/.sigil/token_report.sh


# Find high-token sessions
cat > ~/.sigil/find_expensive.sh << 'EOF'
#!/bin/bash
THRESHOLD=${1:-10000}
echo "Sessions using > $THRESHOLD tokens:"
for dir in ~/.sigil/logs/sessions/*/; do
    session=$(basename "$dir")
    if [ -f "$dir/tokens.json" ]; then
        used=$(cat "$dir/tokens.json" | python -c "import sys,json; print(json.load(sys.stdin).get('total_used',0))")
        if [ "$used" -gt "$THRESHOLD" ]; then
            echo "$session: $used tokens"
        fi
    fi
done
EOF

chmod +x ~/.sigil/find_expensive.sh


# Error summary
cat > ~/.sigil/error_summary.sh << 'EOF'
#!/bin/bash
echo "Error Summary"
echo "============="
for dir in ~/.sigil/logs/sessions/*/; do
    session=$(basename "$dir")
    if [ -f "$dir/execution.jsonl" ]; then
        errors=$(grep '"event_type":".*error"' "$dir/execution.jsonl" | wc -l)
        if [ "$errors" -gt 0 ]; then
            echo "$session: $errors errors"
        fi
    fi
done
EOF

chmod +x ~/.sigil/error_summary.sh
```

---

## 8. Advanced Usage

### 8.1 Multiple Concurrent Sessions

```bash
# Run multiple CLI sessions with different monitors

# Terminal 1: CLI Session A
python -m sigil.interfaces.cli.app orchestrate \
    --task "Analyze lead A" \
    --session session-a

# Terminal 2: Monitor Session A
python -m sigil.interfaces.cli.monitor -s session-a

# Terminal 3: CLI Session B
python -m sigil.interfaces.cli.app orchestrate \
    --task "Analyze lead B" \
    --session session-b

# Terminal 4: Monitor Session B
python -m sigil.interfaces.cli.monitor -s session-b

# Terminal 5: Combined minimal view
watch -n 1 'for s in session-a session-b; do
    echo -n "$s: "
    if [ -f ~/.sigil/logs/sessions/$s/execution.jsonl ]; then
        cat ~/.sigil/logs/sessions/$s/execution.jsonl | \
          python -c "import sys,json; print(sum(json.loads(l).get(\"tokens\",0) for l in sys.stdin))"
    else
        echo "waiting"
    fi
done'
```

### 8.2 Remote Monitoring

```bash
# Monitor from a remote machine

# On remote machine, share log directory via SSH
# On local machine:
ssh user@remote-host "tail -f ~/.sigil/logs/sessions/my-session/execution.jsonl" | \
  python -c "
import sys, json
total = 0
for line in sys.stdin:
    try:
        entry = json.loads(line.strip())
        total += entry.get('tokens', 0)
        print(f'{entry[\"component\"]}: {entry[\"tokens\"]} (Total: {total})')
    except:
        pass
"

# Or use SSH tunnel for full monitor
# On local machine:
ssh -L 8080:localhost:8080 user@remote-host &

# Then access remote logs via mounted filesystem
sshfs user@remote-host:~/.sigil/logs ~/remote-logs

# Run monitor on mounted logs
python -m sigil.interfaces.cli.monitor \
    -s my-session \
    -l ~/remote-logs
```

### 8.3 Custom Log Processing

```python
#!/usr/bin/env python3
"""Custom log processor example."""

import json
import sys
from collections import defaultdict
from datetime import datetime


def process_session_logs(log_path: str):
    """Process session logs and generate report."""

    stats = {
        "total_tokens": 0,
        "component_tokens": defaultdict(int),
        "component_latency": defaultdict(float),
        "errors": [],
        "events": 0,
    }

    start_time = None
    end_time = None

    with open(log_path, 'r') as f:
        for line in f:
            try:
                entry = json.loads(line.strip())
                stats["events"] += 1

                # Track tokens
                tokens = entry.get("tokens", 0)
                stats["total_tokens"] += tokens
                stats["component_tokens"][entry["component"]] += tokens

                # Track latency
                latency = entry.get("metadata", {}).get("latency_ms", 0)
                stats["component_latency"][entry["component"]] += latency

                # Track timestamps
                ts = datetime.fromisoformat(entry["timestamp"])
                if start_time is None:
                    start_time = ts
                end_time = ts

                # Track errors
                if "error" in entry.get("event_type", ""):
                    stats["errors"].append({
                        "component": entry["component"],
                        "message": entry["message"],
                        "timestamp": entry["timestamp"],
                    })

            except json.JSONDecodeError:
                continue

    # Calculate duration
    if start_time and end_time:
        stats["duration_seconds"] = (end_time - start_time).total_seconds()

    return stats


def print_report(stats: dict):
    """Print formatted report."""
    print("\n" + "=" * 50)
    print("SESSION ANALYSIS REPORT")
    print("=" * 50)

    print(f"\nTotal Events: {stats['events']}")
    print(f"Total Tokens: {stats['total_tokens']:,}")
    print(f"Duration: {stats.get('duration_seconds', 0):.1f}s")

    print("\nTokens by Component:")
    for comp, tokens in sorted(stats["component_tokens"].items()):
        print(f"  {comp}: {tokens:,}")

    print("\nLatency by Component:")
    for comp, latency in sorted(stats["component_latency"].items()):
        print(f"  {comp}: {latency:.0f}ms")

    if stats["errors"]:
        print(f"\nErrors ({len(stats['errors'])}):")
        for err in stats["errors"][:5]:  # Show first 5
            print(f"  [{err['component']}] {err['message'][:50]}...")

    print("\n" + "=" * 50)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python process_logs.py <log_path>")
        sys.exit(1)

    stats = process_session_logs(sys.argv[1])
    print_report(stats)
```

### 8.4 Integration with External Tools

```bash
# Send metrics to Prometheus/Grafana
# (Example pushgateway integration)

cat > ~/.sigil/push_metrics.sh << 'EOF'
#!/bin/bash
SESSION=$1
PUSHGATEWAY=${2:-"localhost:9091"}

if [ -z "$SESSION" ]; then
    echo "Usage: push_metrics.sh <session_id> [pushgateway_host]"
    exit 1
fi

LOG_FILE="$HOME/.sigil/logs/sessions/$SESSION/execution.jsonl"

if [ ! -f "$LOG_FILE" ]; then
    echo "Log file not found: $LOG_FILE"
    exit 1
fi

# Calculate metrics
TOTAL_TOKENS=$(cat "$LOG_FILE" | python -c "import sys,json; print(sum(json.loads(l).get('tokens',0) for l in sys.stdin))")
EVENT_COUNT=$(wc -l < "$LOG_FILE")

# Push to Prometheus
cat <<METRICS | curl --data-binary @- "http://$PUSHGATEWAY/metrics/job/sigil/instance/$SESSION"
# HELP sigil_tokens_total Total tokens consumed
# TYPE sigil_tokens_total counter
sigil_tokens_total $TOTAL_TOKENS
# HELP sigil_events_total Total events logged
# TYPE sigil_events_total counter
sigil_events_total $EVENT_COUNT
METRICS

echo "Metrics pushed for session: $SESSION"
EOF

chmod +x ~/.sigil/push_metrics.sh


# Send alerts to Slack
cat > ~/.sigil/slack_alert.py << 'EOF'
#!/usr/bin/env python3
import json
import os
import sys
import requests

SLACK_WEBHOOK = os.environ.get("SLACK_WEBHOOK_URL")

def send_alert(session_id: str, message: str, severity: str = "warning"):
    if not SLACK_WEBHOOK:
        print("SLACK_WEBHOOK_URL not set")
        return

    color = {
        "info": "#36a64f",
        "warning": "#ffcc00",
        "error": "#ff0000",
    }.get(severity, "#cccccc")

    payload = {
        "attachments": [{
            "color": color,
            "title": f"Sigil Alert: {session_id}",
            "text": message,
            "footer": "Sigil Monitoring",
        }]
    }

    requests.post(SLACK_WEBHOOK, json=payload)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: slack_alert.py <session_id> <message> [severity]")
        sys.exit(1)

    send_alert(
        sys.argv[1],
        sys.argv[2],
        sys.argv[3] if len(sys.argv) > 3 else "warning"
    )
EOF

chmod +x ~/.sigil/slack_alert.py
```

---

## 9. Troubleshooting Reference

### 9.1 Error Code Reference

```
+-----------------------------------------------------------------------------+
|                          ERROR CODE REFERENCE                                |
+-----------------------------------------------------------------------------+

ERROR CODE          | DESCRIPTION                    | SOLUTION
--------------------|--------------------------------|---------------------------
E001_SESSION_NF     | Session not found              | Check session ID spelling
E002_LOG_PERM       | Log permission denied          | Fix: chmod 755 ~/.sigil/logs
E003_PARSE_FAIL     | JSON parse failure             | Check log file integrity
E004_NET_TIMEOUT    | Network timeout                | Retry, check connectivity
E005_BUDGET_EXCEED  | Token budget exceeded          | Start new session
E006_COMPONENT_FAIL | Component execution failed     | See component-specific error
E007_CONTRACT_FAIL  | Contract validation failed     | Review output requirements
E008_MEMORY_OOM     | Out of memory                  | Reduce session complexity
E009_DISK_FULL      | Disk full                      | Clean up old logs
E010_CONFIG_ERR     | Configuration error            | Check .env and settings
```

### 9.2 Symptom-Solution Matrix

```
+-----------------------------------------------------------------------------+
|                       SYMPTOM-SOLUTION MATRIX                                |
+-----------------------------------------------------------------------------+

SYMPTOM                          | LIKELY CAUSE              | SOLUTION
---------------------------------|---------------------------|-------------------
Monitor shows "Waiting..."       | Session ID mismatch       | Verify session IDs
Monitor shows old data           | Not reading new entries   | Restart monitor
CLI hangs at "Pipeline start"    | Network/API issue         | Check connectivity
Tokens much higher than expected | Complex task/context      | Use simpler strategy
Session completes but no output  | Validation failed         | Check contract errors
Monitor CPU high                 | Refresh rate too fast     | Use --refresh-rate 0.5
Log file very large              | Many operations           | Enable log rotation
No log file created              | Permission issue          | Check directory perms
JSON errors in logs              | Incomplete writes         | Ensure clean shutdown
Budget warning at start          | Previous session overflow | Use fresh session
```

### 9.3 Quick Fixes

```bash
# Fix 1: Reset log directory
rm -rf ~/.sigil/logs
mkdir -p ~/.sigil/logs/sessions

# Fix 2: Fix permissions
chmod -R 755 ~/.sigil/logs

# Fix 3: Clear stuck session
rm -rf ~/.sigil/logs/sessions/stuck-session

# Fix 4: Restart with clean state
pkill -f "sigil"
python -m sigil.interfaces.cli.app orchestrate --session new-$(date +%s)

# Fix 5: Debug logging
export SIGIL_DEBUG=true
python -m sigil.interfaces.cli.app orchestrate --session debug-session

# Fix 6: Validate log format
python -c "
import json
with open('$HOME/.sigil/logs/sessions/YOUR-SESSION/execution.jsonl') as f:
    for i, line in enumerate(f):
        try:
            json.loads(line)
        except:
            print(f'Invalid JSON at line {i+1}')
"
```

---

## 10. Best Practices

### 10.1 Session Management

```
+-----------------------------------------------------------------------------+
|                    SESSION MANAGEMENT BEST PRACTICES                         |
+-----------------------------------------------------------------------------+

DO:
---
[+] Use descriptive session names
    Good: lead-qualification-acme-20260111
    Bad:  test1

[+] One session per logical task
    Good: Separate session for each lead
    Bad:  All leads in one session

[+] Start monitor before CLI
    Ensures you capture all events from the start

[+] Review session summaries after completion
    cat ~/.sigil/logs/sessions/SESSION/tokens.json

[+] Archive important sessions
    Backup before cleanup for post-analysis


DON'T:
------
[-] Reuse session IDs
    Always use unique session names

[-] Run multiple CLIs with same session
    Causes log corruption

[-] Ignore budget warnings
    Leads to incomplete results

[-] Delete active session logs
    Wait for completion

[-] Use very long session names
    Keep under 64 characters
```

### 10.2 Monitoring Best Practices

```
+-----------------------------------------------------------------------------+
|                    MONITORING BEST PRACTICES                                 |
+-----------------------------------------------------------------------------+

DO:
---
[+] Match CLI and monitor session IDs exactly

[+] Use appropriate display mode for context
    - Dashboard: Interactive debugging
    - Stream: CI/CD pipelines
    - Minimal: Resource-constrained environments

[+] Review performance section for bottlenecks

[+] Set up alerts for long-running sessions

[+] Correlate events with CLI output


DON'T:
------
[-] Rely solely on monitor for session status
    Check CLI output directly for final results

[-] Ignore component latency warnings
    Indicates potential issues

[-] Run monitor on same machine as heavy CLI load
    Use remote monitoring for production

[-] Let monitor run indefinitely
    Stop when session completes
```

### 10.3 Log Management Best Practices

```
+-----------------------------------------------------------------------------+
|                    LOG MANAGEMENT BEST PRACTICES                             |
+-----------------------------------------------------------------------------+

DO:
---
[+] Set up automated cleanup
    Use cron job for daily cleanup

[+] Monitor disk usage
    Set alerts for log directory size

[+] Archive important sessions before cleanup

[+] Use compression for archives
    tar.gz reduces size significantly

[+] Review error logs weekly
    Identify recurring issues


DON'T:
------
[-] Let logs grow unbounded
    Causes disk issues

[-] Delete without archiving
    Lose debugging information

[-] Store logs on slow filesystems
    Impacts CLI performance

[-] Ignore log rotation settings
    Configure appropriately for usage
```

### 10.4 Performance Best Practices

```
+-----------------------------------------------------------------------------+
|                    PERFORMANCE BEST PRACTICES                                |
+-----------------------------------------------------------------------------+

DO:
---
[+] Start with conservative token budgets
    Increase based on actual usage patterns

[+] Use appropriate reasoning strategy
    Don't use MCTS for simple queries

[+] Monitor token consumption trends
    Identify optimization opportunities

[+] Cache frequently used routing decisions

[+] Pre-compute memory summaries


DON'T:
------
[-] Use maximum budget by default
    Wastes tokens on overhead

[-] Ignore the slowest component metric
    Focus optimization there

[-] Run complex tasks in constrained environments

[-] Skip validation to save tokens
    Quality suffers
```

---

## Document Revision History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0.0 | 2026-01-11 | Systems Architecture | Initial release |

---

*End of CLI Monitoring User Guide*
