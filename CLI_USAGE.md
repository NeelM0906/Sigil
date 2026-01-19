# Sigil v2 CLI - Quick Start Guide

## Overview

Sigil v2 is now fully accessible through the command-line interface with real-time execution monitoring and token tracking.

## Three Main Commands

### 1. Orchestrate - Run Full Pipeline

Execute any task through the complete Sigil v2 orchestration pipeline:

```bash
python -m sigil.interfaces.cli.app orchestrate \
  --task "Analyze Acme Corp as a prospect for CRM solutions" \
  --session-id test-1 \
  --verbose
```

**Options:**
- `--task TEXT` (required) - The task or message to process
- `--session-id TEXT` - Session identifier (auto-generated if not provided)
- `--user-id TEXT` - Optional user identifier
- `--contract TEXT` - Optional contract name to enforce
- `--strategy [direct|chain_of_thought|tree_of_thoughts|react|mcts]` - Force specific reasoning strategy
- `--verbose` - Show detailed pipeline output

**Output:**
- Routing decision (intent, complexity, handler)
- Reasoning strategy selected
- Token usage per component (Routing, Memory, Planning, Reasoning, Validation)
- Total tokens and budget percentage (out of 256,000)
- Execution time

### 2. Log Stream - Real-Time Monitoring

Tail the execution log file in real-time (best in separate terminal):

```bash
python -m sigil.interfaces.cli.app log-stream
```

**Output Format:**
```
[TIMESTAMP] [LEVEL] [COMPONENT] [TOKENS] Message
[2026-01-12 04:33:39] [INFO] [ROUTING] [50 tokens] Intent: general_chat, Complexity: 0.50
[2026-01-12 04:33:39] [INFO] [REASONING] [68 tokens] Completed with confidence: 0.7
[2026-01-12 04:33:39] [INFO] [COMPLETE] [118 tokens] Total: 118 tokens / 256,000 (0.05%)
```

### 3. Status - View Metrics

Check orchestrator status and accumulated metrics:

```bash
python -m sigil.interfaces.cli.app status
```

**Output:**
- Active features (memory, planning, contracts, evolution)
- Environment info (dev/prod, debug mode, log level)
- Log file path and size
- Orchestrator metrics (total requests, success rate, total tokens)
- Health status

## Dual-Terminal Setup (Recommended)

For a full observability experience, run the CLI and monitor in parallel:

**Terminal 1 - Start Monitor:**
```bash
source venv/bin/activate
python -m sigil.interfaces.cli.app log-stream
```

**Terminal 2 - Run Orchestrations:**
```bash
source venv/bin/activate
python -m sigil.interfaces.cli.app orchestrate \
  --task "Qualify lead John from Acme Corp" \
  --session-id lead-qualify-1 \
  --verbose
```

As you run tasks in Terminal 2, Terminal 1 will show real-time logs with token tracking.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                  Dual-Terminal Setup                    │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  Terminal 1 (Monitor)          Terminal 2 (CLI)        │
│  ──────────────────────        ──────────────          │
│                                                          │
│  python -m sigil...   ←→   Shared Log File    ←→   python -m sigil...
│  .log-stream          outputs/                    .orchestrate
│                       sigil-execution.log         --task "..."
│                                                          │
│  Displays:            Format:                    Executes:
│  • Real-time logs     • JSON Lines               • Pipeline
│  • Token counter      • Structured data          • Memory
│  • Token budget %     • Component tags           • Planning
│  • Color-coded        • Timestamps               • Reasoning
│                       • Token counts              • Validation
│                                                   │
└─────────────────────────────────────────────────────────┘
```

## Token Budget Tracking

All operations track tokens against a 256,000 token session budget:

- **Routing**: ~50 tokens per request (intent classification, complexity assessment)
- **Memory**: Variable (0-200 tokens depending on retrieval mode)
- **Planning**: Variable (0-500+ tokens for complex plans)
- **Reasoning**: Variable (50-2000+ tokens depending on strategy)
- **Validation**: Variable (0-500 tokens for contract validation)

**Status display shows:**
```
Total: 118 / 256,000 (0.05%)
```

## Pipeline Execution Flow

```
orchestrate command
       ↓
Input: task, session_id, context
       ↓
1. ROUTING LAYER
   • Intent classification
   • Complexity assessment (0.0-1.0)
   • Handler selection
       ↓
2. MEMORY LAYER (if enabled)
   • Retrieve relevant items
   • Hybrid RAG + LLM search
       ↓
3. PLANNING LAYER (if enabled)
   • Goal decomposition
   • DAG plan creation
       ↓
4. REASONING LAYER
   • Select strategy by complexity:
     - 0.0-0.3: Direct
     - 0.3-0.5: Chain-of-Thought
     - 0.5-0.7: Tree-of-Thoughts
     - 0.7-0.9: ReAct
     - 0.9-1.0: MCTS
   • Execute with LLM
       ↓
5. VALIDATION LAYER (if contract specified)
   • Verify output schema
   • Check constraints
   • Enforce success criteria
       ↓
Output: Result + Token Summary
```

## Example Workflows

### 1. Simple Analysis
```bash
python -m sigil.interfaces.cli.app orchestrate \
  --task "What are the key benefits of CRM systems?"
```

### 2. Lead Qualification
```bash
python -m sigil.interfaces.cli.app orchestrate \
  --task "Qualify John Smith, VP at Acme Corp, $500k budget" \
  --session-id acme-lead \
  --contract lead_qualification
```

### 3. Complex Reasoning (Force Tree-of-Thoughts)
```bash
python -m sigil.interfaces.cli.app orchestrate \
  --task "Develop competitive positioning against Salesforce" \
  --strategy tree_of_thoughts \
  --verbose
```

### 4. Monitor Specific Session
```bash
# Terminal 1
python -m sigil.interfaces.cli.app log-stream

# Terminal 2 - Run 3 different tasks
python -m sigil.interfaces.cli.app orchestrate --task "Task 1" --session-id session-1
python -m sigil.interfaces.cli.app orchestrate --task "Task 2" --session-id session-2
python -m sigil.interfaces.cli.app orchestrate --task "Task 3" --session-id session-3

# Terminal 1 shows all logs aggregated
```

## Files and Components

- **`sigil/interfaces/cli/app.py`** - Main CLI application with Click commands
- **`sigil/interfaces/cli/monitoring.py`** - Token tracking, logging, display utilities
- **`scripts/monitor.py`** - Standalone log monitor for separate terminal
- **`outputs/sigil-execution.log`** - Structured execution logs (JSON Lines format)

## Troubleshooting

### No logs appearing in log-stream?
- Ensure `outputs/` directory exists
- Check file permissions on `outputs/sigil-execution.log`
- Verify orchestrate command is running in another terminal

### Token count seems wrong?
- Different strategies use different token budgets
- Memory retrieval adds tokens
- Planning adds tokens
- Check verbose output for per-component breakdown

### Session not found?
- Sessions are identified by `session-id` parameter
- Each session creates a separate entry in logs
- Use unique session IDs to differentiate runs

## Next Steps

1. **Enable Memory System**: Set `SIGIL_USE_MEMORY=true` in environment
2. **Enable Planning**: Set `SIGIL_USE_PLANNING=true` in environment
3. **Add Contracts**: Use `--contract lead_qualification` for verified outputs
4. **Scale Monitoring**: Use log aggregation tools to capture all sessions

## References

- [Architecture Documentation](docs/cli-architecture.md)
- [Token Budgeting Guide](docs/cli-token-budgeting.md)
- [Monitoring Guide](docs/cli-monitoring-guide.md)
- [Implementation Roadmap](docs/cli-implementation-roadmap.md)
