# Sigil

**A meta-agent framework for creating executable AI agents with real tool capabilities.**

Sigil is a Python-based agent builder that uses the ACTi methodology to design and generate AI agents. Unlike traditional workflow builders that produce static configurations, Sigil creates agents that can actually *do things* - make calls, search the web, schedule meetings, and more - through real MCP (Model Context Protocol) tool integrations.

## Features

- **Meta-Agent Architecture**: A builder agent that creates other agents through natural language conversation
- **ACTi Methodology**: Five-stratum framework for classifying and designing agents:
  - **RTI** (Reality & Truth Intelligence): Data gathering, fact verification
  - **RAI** (Readiness & Agreement Intelligence): Lead qualification, rapport building
  - **ZACS** (Zone Action & Conversion Systems): Scheduling, conversions
  - **EEI** (Economic & Ecosystem Intelligence): Analytics, optimization
  - **IGE** (Integrity & Governance Engine): Compliance, quality control
- **Real Tool Capabilities**: Agents connect to actual services via MCP servers:
  - Voice synthesis (ElevenLabs)
  - Web search (Tavily)
  - Calendar management (Google Calendar)
  - Communication (Twilio)
  - CRM integration (HubSpot)
- **Subagent Delegation**: Specialized subagents for prompt engineering and pattern analysis
- **Reference Pattern Learning**: Learn from Bland AI and N8N workflow patterns

## Architecture

```
                     ┌─────────────────────────────────┐
                     │         Sigil Builder           │
                     │    (deepagents meta-agent)      │
                     └───────────────┬─────────────────┘
                                     │
              ┌──────────────────────┼──────────────────────┐
              │                      │                      │
    ┌─────────▼─────────┐  ┌────────▼────────┐  ┌─────────▼─────────┐
    │  prompt-engineer  │  │ pattern-analyzer│  │   Builder Tools   │
    │    (subagent)     │  │   (subagent)    │  │ create_agent_config│
    └───────────────────┘  └─────────────────┘  │ list_available_tools│
                                                │ get_agent_config   │
                                                │ list_created_agents│
                                                └─────────────────────┘
                                                          │
                                                          ▼
                                               ┌─────────────────────┐
                                               │  Generated Agents   │
                                               │  (with MCP tools)   │
                                               └─────────────────────┘
```

## Quick Start

### Prerequisites

- Python 3.11 or higher
- An Anthropic API key

### Installation

```bash
# Clone the repository
git clone https://github.com/NeelM0906/Sigil.git
cd Sigil

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e .

# Set up environment variables
cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY
```

### Usage

#### Interactive CLI

```bash
# Run the builder in interactive mode
python -m src.builder
```

Example session:
```
You: Create a lead qualification agent for B2B SaaS companies

Builder: I'll create a lead qualification agent for you. Let me first analyze
the requirements and select appropriate tools...

[Agent analyzes requirements, selects tools, crafts system prompt]

Builder: I've created the lead_qualifier agent with the following configuration:
- Stratum: RAI (Readiness & Agreement Intelligence)
- Tools: communication, crm
- Saved to: outputs/agents/lead_qualifier.json
```

#### Programmatic Usage

```python
from src.builder import create_builder
from langchain_core.messages import HumanMessage

# Create the builder agent
builder = create_builder()

# Generate an agent configuration
result = builder.invoke({
    "messages": [HumanMessage(content="Create an appointment scheduling agent")]
})

# The agent config is saved to outputs/agents/
print(result["messages"][-1].content)
```

#### Using Builder Tools Directly

```python
from src.tools import create_agent_config, list_available_tools

# List available MCP tool categories
print(list_available_tools.invoke({}))

# Create an agent configuration
result = create_agent_config.invoke({
    "name": "research_assistant",
    "description": "Researches topics and gathers facts from the web",
    "system_prompt": "You are a research assistant that helps users find accurate information...",
    "tools": ["websearch"],
    "stratum": "RTI"
})
print(result)
```

## Project Structure

```
sigil/
├── src/
│   ├── __init__.py          # Package exports
│   ├── builder.py           # Main builder agent
│   ├── tools.py             # LangChain tools for agent creation
│   ├── schemas.py           # Pydantic data models
│   └── prompts.py           # System prompts for builder and subagents
├── tests/
│   ├── conftest.py          # Shared test fixtures
│   ├── test_schemas.py      # Schema validation tests
│   ├── test_tools.py        # Tool functionality tests
│   ├── test_prompts.py      # Prompt content tests
│   └── test_builder.py      # Builder creation tests
├── outputs/
│   └── agents/              # Generated agent configurations
├── docs/
│   ├── api-contract-tools.md    # Tools API documentation
│   └── api-contract-builder.md  # Builder API documentation
├── pyproject.toml           # Project configuration
└── README.md                # This file
```

## Available Tools

The builder can assign these MCP tool categories to generated agents:

| Category | Provider | Capabilities |
|----------|----------|--------------|
| `voice` | ElevenLabs | Text-to-speech, voice synthesis |
| `websearch` | Tavily | Web search, research, fact-finding |
| `calendar` | Google Calendar | Scheduling, availability checks |
| `communication` | Twilio | SMS, phone calls |
| `crm` | HubSpot | Contact management, deal tracking |

## ACTi Strata

Each agent is classified into one of five strata based on its primary function:

| Stratum | Purpose | Recommended Tools |
|---------|---------|-------------------|
| **RTI** | Data gathering, fact verification | websearch, crm |
| **RAI** | Lead qualification, rapport building | communication, crm |
| **ZACS** | Scheduling, conversions, follow-ups | calendar, communication, voice |
| **EEI** | Analytics, market research | websearch, crm |
| **IGE** | Compliance, quality control | all tools |

## Running Tests

```bash
# Activate virtual environment
source venv/bin/activate

# Run all tests
PYTHONPATH=. pytest tests/ -v

# Run specific test file
PYTHONPATH=. pytest tests/test_schemas.py -v

# Run with coverage
PYTHONPATH=. pytest tests/ --cov=src --cov-report=html
```

## Configuration

### Environment Variables

Create a `.env` file in the project root:

```bash
# Required
ANTHROPIC_API_KEY=sk-ant-...

# Optional (for MCP tool execution in Phase 2)
ELEVENLABS_API_KEY=...
TAVILY_API_KEY=...
TWILIO_ACCOUNT_SID=...
TWILIO_AUTH_TOKEN=...
HUBSPOT_API_KEY=...
```

### Model Configuration

The default model is `anthropic:claude-opus-4-5-20251101`. You can override this:

```python
from src.builder import create_builder

# Use a different model
builder = create_builder(model="anthropic:claude-sonnet-4-20250514")
```

## API Documentation

Detailed API documentation is available in the `docs/` directory:

- [Tools API Contract](docs/api-contract-tools.md) - Builder tools specification
- [Builder API Contract](docs/api-contract-builder.md) - Builder module specification

## Roadmap

- [x] **Phase 1: Foundation** - Builder agent with config generation
- [x] **Phase 2: MCP Integration** - Real tool execution via MCP servers
- [ ] **Phase 3: API Backend** - FastAPI server for frontend integration
- [ ] **Phase 4: Frontend** - Web UI for agent creation and management

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built on [deepagents](https://github.com/deepagents/deepagents) framework
- Inspired by ACTi methodology from the Unblinded Formula framework
- Reference patterns from Bland AI and N8N workflows
