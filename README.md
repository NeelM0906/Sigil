# Sigil: Meta-Agent Framework for Building Executable AI Agents

**Build AI agents with real tool capabilities using the ACTi methodology and Sigil's modular architecture.**

Sigil is a comprehensive AI agent framework that combines meta-agent design patterns with production-ready execution infrastructure. It enables both interactive CLI-based agent workflows and programmatic generation of task-specific agents through a builder framework.

## Core Vision

Sigil provides two complementary paradigms:

1. **Interactive Agent CLI** - Ready-to-use agent interface with planning, routing, memory management, and real-time tool execution
2. **Agent Builder Framework** - Meta-agent that designs and generates new agents for specific tasks

Both leverage the same underlying architecture: **Planning → Routing → Memory → Reasoning → Contract Validation**

---

## Architecture Overview

### System Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                    Sigil Agent System                         │
└──────────────────────────────────────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
        ▼                   ▼                   ▼
   ┌─────────────┐   ┌─────────────┐   ┌──────────────┐
   │  Interactive│   │   Agent     │   │   Tool       │
   │  CLI        │   │   Builder   │   │   Runtime    │
   │             │   │             │   │              │
   │ • Routing   │   │ • ACTi      │   │ • Tavily     │
   │ • Planning  │   │   Strata    │   │ • Executors  │
   │ • Memory    │   │ • Prompt    │   │ • Results    │
   │ • Reasoning │   │   Crafting  │   │   Mgmt       │
   └─────────────┘   └─────────────┘   └──────────────┘
        │                   │                   │
        └───────────────────┴───────────────────┘
                        │
            ┌───────────┴───────────┐
            │                       │
      ┌─────▼──────┐         ┌─────▼──────┐
      │   LLM      │         │   Context  │
      │  (Claude)  │         │   & State  │
      └────────────┘         └────────────┘
```

### Execution Pipeline

```
Request Entry Point
        │
        ▼
    ┌─────────────────────────────────┐
    │ 1. ROUTING                      │
    │ Intent classification &         │
    │ complexity assessment           │
    └────────┬────────────────────────┘
             │
             ▼
    ┌─────────────────────────────────┐
    │ 2. PLANNING (if complex)        │
    │ Goal decomposition &            │
    │ step sequencing                 │
    └────────┬────────────────────────┘
             │
             ▼
    ┌─────────────────────────────────┐
    │ 3. CONTEXT ASSEMBLY             │
    │ Memory retrieval & history      │
    │ synthesis                       │
    └────────┬────────────────────────┘
             │
             ▼
    ┌─────────────────────────────────┐
    │ 4. EXECUTION                    │
    │ Tool calls & reasoning steps    │
    │ with result integration         │
    └────────┬────────────────────────┘
             │
             ▼
    ┌─────────────────────────────────┐
    │ 5. VALIDATION (if contracted)   │
    │ Output contract verification    │
    └────────┬────────────────────────┘
             │
             ▼
        Response Output
```

### Component Responsibilities

#### **Router** (`sigil/routing/`)
- Classifies user intent from message content
- Assesses task complexity (0.0-1.0)
- Determines if planning, memory, contracts are needed
- **Keyword detection** for tool-triggering queries ("search", "news", "latest", etc.)

#### **Planner** (`sigil/planning/`)
- Decomposes goals into executable steps
- Creates step sequences with dependencies
- Attaches tool specifications (websearch, calendar, etc.)
- Extracts query arguments from user intent
- **Tool-aware planning**: Automatically includes search parameters in plan

#### **Tool Executor** (`sigil/planning/tool_executor.py`)
- Routes TOOL_CALL steps to appropriate executors:
  - **TavilyExecutor**: Direct Tavily API integration (~1s response)
  - **BuiltinExecutor**: Memory/planning builtin tools
- Formats tool results explicitly for LLM consumption
- Handles timeouts, errors, and fallback reasoning

#### **Memory Manager** (`sigil/memory/`)
- Retrieves relevant historical context (RAG)
- Stores memories from interactions
- Supports layered memory (short-term, semantic, event-based)

#### **Reasoning Manager** (`sigil/reasoning/`)
- Selects strategy based on complexity:
  - **0.0-0.3**: Direct (fastest, single-pass)
  - **0.3-0.5**: Chain of Thought (explicit reasoning)
  - **0.5-0.7**: Tree of Thoughts (multi-branch exploration)
  - **0.7-0.9**: ReAct (reasoning + action loops)
  - **0.9-1.0**: MCTS (Monte Carlo tree search)
- **Tool-aware prompting**: Integrates tool results into strategy decisions

#### **Contract System** (`sigil/contracts/`)
- Validates agent outputs against expected schemas
- Ensures structured compliance for specific intents
- Provides validation feedback

---

## Key Features

### 1. Real Tool Integration
- **Tavily Web Search**: Direct API + formatted result synthesis
- **ElevenLabs Voice**: Text-to-speech synthesis
- **Google Calendar**: Event scheduling and availability
- **Twilio**: SMS and phone communication
- **HubSpot**: CRM operations and contact management
- **Extensible Architecture**: Add new tools via custom executors

### 2. Planning & Tool Awareness
- Automatic query extraction for search tools
- Explicit tool result formatting in LLM prompts
- Result truncation with memory limits (prevent token bloat)
- Multi-step plans with dependency resolution

### 3. Intelligent Routing
- Intent-based message classification
- Tool keyword detection (automatically triggers planning)
- Complexity-driven strategy selection
- Fallback mechanisms for graceful degradation

### 4. Memory & Context Management
- Multi-layer memory system
- Semantic similarity retrieval
- Event-based memory indexing
- Customizable memory templates

### 5. Agent Builder Framework (ACTi)
Meta-agent system that creates task-specific agents by:
- Analyzing user requirements
- Selecting appropriate tools from available catalog
- Crafting optimized system prompts via subagent delegation
- Learning from reference patterns (Bland AI, N8N)
- Generating structured agent configurations

---

## Quick Start

### Prerequisites
- Python 3.11+
- Anthropic API key
- (Optional) Tool API keys (Tavily, ElevenLabs, etc.)

### Installation

```bash
# Clone and setup
git clone https://github.com/NeelM0906/Sigil.git
cd Sigil

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -e .

# Configure environment
cp .env.example .env
# Edit .env and add ANTHROPIC_API_KEY and tool-specific keys
```

### Interactive CLI Usage

```bash
# Launch interactive agent
python -m sigil.interfaces.cli interactive

# Example queries:
# >>> find me the latest news about Iran
# >>> search for AI developments in 2025
# >>> what happened in tech this week
```

The CLI automatically:
1. Routes your query (detects "news/search" keywords)
2. Plans multi-step execution if needed
3. Executes Tavily search (~1 second)
4. Synthesizes results into readable response

### Agent Builder Usage

```python
from sigil.orchestrator import SigilOrchestrator
from sigil.core.schemas import OrchestratorRequest

# Create orchestrator
orchestrator = SigilOrchestrator()

# Process a request
request = OrchestratorRequest(
    message="Create a lead qualification agent for B2B SaaS",
    session_id="builder-session"
)
response = await orchestrator.process(request)

# Access generated output
print(response.output["result"])
```

---

## Data Flow: Tool Search Example

**User Query**: "Tell me the latest news about Iran"

```
1. INPUT
   Message: "Tell me the latest news about Iran"

2. ROUTING
   Router detects keywords: "news", "latest"
   └─> Intent: run_agent
   └─> Complexity: 0.09
   └─> use_planning: TRUE (keyword-triggered)

3. PLANNING
   Planner.create_plan("Tell me the latest news about Iran")
   └─> Step 1: websearch.search
       └─> _tool_args: {"query": "Tell me the latest news about Iran", "max_results": 5}
   └─> Step 2: Reason about search results
   └─> Step 3: Generate final response

4. TOOL EXECUTION
   ToolStepExecutor._execute_tool_call(step_1)
   └─> Detects tool_name: "websearch.search"
   └─> Extracts _tool_args: {"query": "..."}
   └─> Routes to TavilyExecutor (direct API)
   └─> Returns: {
         "results": [
           {"title": "Iran News...", "url": "...", "content": "..."},
           ...
         ]
       }

5. REASONING CONTEXT INTEGRATION
   ToolStepExecutor._execute_reasoning(step_2)
   └─> Builds context with prior_outputs:
       {
         "prior_outputs": {
           "step-1": {
             "output": "<Tavily JSON>",
             "tokens_used": 0
           }
         }
       }
   └─> format_tool_results_section() detected as Tavily format
   └─> Formats as:
       "## Tool Results
        ### step-1
        Search query: Tell me the latest news about Iran

        1. Iran News Update
           Latest developments in Iran...
           Source: https://..."
   └─> Frames reasoning task:
       "Based on the following tool results, Generate final response:
        ## Tool Results
        [formatted results]

        Generate a comprehensive response for the user."

6. LLM REASONING
   DirectStrategy.execute(task, context_with_formatted_results)
   └─> Calls Claude with formatted tool results
   └─> Returns synthesized response with citations

7. OUTPUT AGGREGATION
   PlanResult.from_step_results()
   └─> Uses LAST step output (synthesis, not raw tool data)
   └─> Returns: "# Latest News About Iran - Summary\n..."

8. RESPONSE
   {
     "status": "success",
     "result": "# Latest News About Iran...",
     "tokens_used": 5397,
     "time_ms": 14548
   }
```

---

## Architecture Modules

### `sigil/routing/`
Intent classification and routing decisions. Single entry point for determining which subsystems activate.

### `sigil/planning/`
Task decomposition into executable steps with tool specifications and dependencies.

**Submodules:**
- `planner.py`: Goal decomposition engine
- `executor.py`: Plan step sequencer
- `tool_executor.py`: Tool call router and result formatter
- `executors/`: Specific executor implementations
  - `tavily_executor.py`: Direct Tavily API
  - `builtin_executor.py`: Memory/planning tools
- `schemas.py`: Plan and step data models

### `sigil/reasoning/`
Strategy selection and LLM reasoning. Complexity-based strategy routing.

**Strategies:**
- `direct.py`: Single-pass reasoning (0.0-0.3 complexity)
- `chain_of_thought.py`: Step-by-step reasoning (0.3-0.5)
- `tree_of_thoughts.py`: Multi-branch exploration (0.5-0.7)
- `react.py`: Reason-Act loops (0.7-0.9)
- `mcts.py`: Monte Carlo tree search (0.9-1.0)
- `utils.py`: Tool result formatting utilities

### `sigil/memory/`
Context retrieval and storage. Multi-layer memory with semantic indexing.

**Layers:**
- Short-term: Recent messages and interactions
- Semantic: Vector-indexed facts and insights
- Event-based: Historical events and milestones

### `sigil/contracts/`
Output validation and schema enforcement for specific intents.

### `sigil/interfaces/`
User interaction surfaces:
- `cli/`: Interactive command-line interface
- `api/`: REST API endpoints (planned)

---

## Advanced Configuration

### Feature Flags
Enable/disable subsystems via environment variables:

```bash
SIGIL_USE_PLANNING=true      # Enable task decomposition
SIGIL_USE_ROUTING=true       # Enable intent classification
SIGIL_USE_MEMORY=true        # Enable context retrieval
SIGIL_USE_CONTRACTS=false    # Disable output validation
```

### Model Selection
```bash
SIGIL_LLM_MODEL=claude-opus-4-5-20251101  # Default
```

### Token Management
```bash
SIGIL_MAX_INPUT_TOKENS=150000
SIGIL_MAX_OUTPUT_TOKENS=102400
```

---

## Testing & Validation

See `Claude.md` for comprehensive testing methodologies, test harnesses, and validation frameworks for:
- Memory system validation
- Context management engine testing
- Planning module integration tests
- TextGrad optimization evaluation

---

## ACTi Methodology: Agent Strata

The framework classifies agents into five intelligence strata:

| Stratum | Purpose | Key Capabilities | Recommended Tools |
|---------|---------|-----------------|-------------------|
| **RTI** | Reality & Truth Intelligence | Data gathering, fact verification, research | websearch, crm |
| **RAI** | Readiness & Agreement Intelligence | Lead qualification, rapport building, persuasion | communication, crm, voice |
| **ZACS** | Zone Action & Conversion Systems | Scheduling, follow-ups, conversion optimization | calendar, communication, voice |
| **EEI** | Economic & Ecosystem Intelligence | Market analysis, forecasting, optimization | websearch, crm, analytics |
| **IGE** | Integrity & Governance Engine | Compliance, audit, quality assurance | all tools + contracts |

---

## Roadmap

- [x] **Phase 1**: Core architecture (routing, planning, reasoning, memory)
- [x] **Phase 2**: Tool integration (Tavily, ElevenLabs, calendar, etc.)
- [x] **Phase 3**: Interactive CLI with real-time tool execution
- [x] **Phase 4**: Tool result synthesis and prompt integration
- [x] **Phase 5**: Agent builder meta-agent framework
- [ ] **Phase 6**: TextGrad optimization for prompt refinement
- [ ] **Phase 7**: REST API backend for web UIs
- [ ] **Phase 8**: Frontend dashboard for agent management

---

## Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/description`)
3. Make changes following the modular architecture
4. Add tests for new functionality
5. Submit a Pull Request with clear description

---

## License

MIT License - see [LICENSE](LICENSE) file

---

## Acknowledgments

- Built on [deepagents](https://github.com/deepagents/deepagents) framework
- Inspired by ACTi methodology and Unblinded Formula framework
- Reference patterns from Bland AI and N8N
