# Sigil + Bland AI Workflow Knowledge Integration

> **Semantic search and RAG over your Bland AI conversation pathways**

This integration adds intelligent workflow knowledge capabilities to the Sigil framework, allowing you to search through Bland AI pathways and get instant answers based on your actual data.

---

## What This Does

**Search your Bland AI workflows semantically** - Ask questions like "How do I handle objections?" and get answers from YOUR actual pathways, not generic AI responses.

### Key Features

**Semantic Search** - Understands meaning, not just keywords  
 **RAG (Retrieval Augmented Generation)** - Answers based on your data  
 **Source Citations** - Shows which pathway each answer came from  
 **Relevance Scoring** - Ranks results by match quality (e.g., 89% match)  
 **Verbose Mode** - See the search process and what was found  
 **Integrated with Sigil** - Works with agent builder  

---



## 🚀 Quick Start

### 1. Clone & Setup

```bash
# Clone the repository
git clone https://github.com/NeelM0906/Sigil.git
cd Sigil

# Checkout the integration branch
git checkout sab_dev

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# Install dependencies
pip install openai numpy
```

### 2. Set Environment Variables

**Windows (CMD):**
```bash
set OPENAI_API_KEY=your_openai_key_here
```

**Windows (PowerShell):**
```powershell
$env:OPENAI_API_KEY="your_openai_key_here"
```

**Mac/Linux:**
```bash
export OPENAI_API_KEY=your_openai_key_here
```

### 3. Configure Data Path

Edit `sigil/tools/workflow_knowledge.py` (line 20):

```python
# Update this path to where your Bland data is located
ACTI_PATH = r"D:\ACTI\Sigil-neel_dev\Sigil-neel_dev\Autogen-Framework-autogen-bland-json"
```

Change it to your actual path where `Autogen-Framework-autogen-bland-json/data/` exists.

---

## 🧪 Testing Guide

### Test 1: Pure Workflow Search (No Sigil)

Test just the search functionality over your Bland data:

```bash
python workflow_search.py
```

**What it does:**
- Indexes your Bland pathways
- Performs semantic search
- Shows matching workflows with relevance scores
- Displays actual content from your data

**Example queries:**
```
🔍 Query: How do I handle price objections?
🔍 Query: Show me email collection examples
🔍 Query: What's a good cold call script?
🔍 Query: stats
🔍 Query: quit
```

**Expected output:**
```
SEARCHING YOUR WORKFLOW DATA
 Query: How do I handle price objections?
 Searching through: 47 pathways

️  Performing semantic search...
 Found 3 matching workflows!

 SEARCH RESULTS - RANKED BY RELEVANCE
======================================================================

🔹 MATCH #1
======================================================================
 Source: Brian Kent Pathway
 Relevance: 89%
 Content Preview:
   **Handle Objections** (Knowledge Base)
   When discussing pricing, use the feel-felt-found technique...
   [actual content from your pathway]
```

---

### Test 2: Integration Test Suite

Test the complete Sigil integration:

```bash
python tests\test_workflow_integration.py
```

**What it tests:**
1. Import workflow knowledge module
2.  Create tool instance and build index
3.  Search functionality with relevance scoring
4.  Question answering with formatted responses
5.  Example retrieval
6.  Router intent detection
7.  Tool executor integration
8.  End-to-end workflow

**Expected output:**
```
🧪 SIGIL + ACTi INTEGRATION TEST SUITE
============================================================
TEST 1: Import Workflow Knowledge Tool
============================================================
✅ Successfully imported workflow_knowledge module

============================================================
TEST 2: Create Tool Instance
============================================================
✅ Indexed 47 pathways

[... more tests ...]

============================================================
TOTAL: 7/8 tests passed
============================================================
```

---

### Test 3: Combined CLI (Workflow Search + Agent Builder)

The main interface that combines workflow knowledge with agent building:

```bash
python -m src.cli_combined
```

**What it does:**
- **Detects intent automatically** - Routes queries to the right handler
- **Workflow Knowledge Mode** - Answers "how to" questions from your data
- **Agent Builder Mode** - Creates agents, optionally searching workflows first
- **Verbose search** - Shows what was found before building

**Example Usage:**

**Query 1: Pure Workflow Knowledge**
```
You: How do I handle objections?

[🤖 Detected Intent: workflow_knowledge]
💡 [Workflow Knowledge Mode]

🔍 SEARCHING YOUR WORKFLOW DATA
📋 Query: How do I handle objections?
📂 Searching: 47 Bland pathways

✅ Found 3 matching workflows!

📊 MATCHING WORKFLOWS FROM YOUR DATA:
🔹 Match #1: Brian Kent Pathway (89% relevance)
   [shows actual content from your pathway]

📝 FORMATTED ANSWER:
## Answer: How do I handle objections?
[compiled answer from your data with sources]
```

**Query 2: Agent Building with Workflow Search**
```
You: Create a lead qualification agent

[🤖 Detected Intent: create_agent]
🔨 [Agent Builder Mode with Workflow Search]

📚 Step 1: Searching your existing workflows...
🔍 SEARCHING YOUR WORKFLOW DATA
✅ Found 3 relevant patterns!

🔹 Match #1: Brian Kent - Lead Scoring (85%)
   [shows content]

🔨 Step 2: BUILDING AGENT
✅ Using workflow knowledge as context for agent design...

Builder: Based on your existing workflows, I'll create a lead 
qualification agent using patterns from Brian Kent Pathway...
[builds agent using your proven techniques]
```

---

## 📁 Project Structure

```
Sigil/
├── sigil/
│   ├── tools/
│   │   └── workflow_knowledge.py          # Main workflow search tool
│   ├── routing/
│   │   └── router.py                      # Intent detection (modified)
│   └── planning/
│       └── tool_executor.py               # Tool execution (modified)
├── src/
│   └── cli_combined.py                    # Combined CLI interface
├── tests/
│   └── test_workflow_integration.py       # Integration tests
├── workflow_search.py                     # Standalone search tool
├── Autogen-Framework-autogen-bland-json/  # Your Bland data
│   └── data/
│       └── bland_dataset/                 # JSON pathway files
└── README.md
```

---

## 🔧 How It Works

### Architecture

```
User Query
    ↓
Router (detects intent)
    ↓
├─→ WORKFLOW_KNOWLEDGE intent
│   ↓
│   Workflow Knowledge Tool
│   ↓
│   Semantic Search (OpenAI embeddings)
│   ↓
│   Find relevant pathways
│   ↓
│   Extract content (RAG)
│   ↓
│   Return formatted answer with sources
│
└─→ CREATE_AGENT intent
    ↓
    1. Search workflows for patterns
    2. Show what was found
    3. Use findings as context
    4. Build agent with proven techniques
```


## 📝 Development Notes

### Modified Files

1. **sigil/tools/workflow_knowledge.py** (NEW)
   - Main workflow knowledge tool
   - Integrates with ACTi Router retrieval system

2. **sigil/routing/router.py** (MODIFIED)
   - Added `WORKFLOW_KNOWLEDGE` intent
   - Added keyword patterns for workflow queries

3. **sigil/planning/tool_executor.py** (MODIFIED)
   - Added `_execute_workflow_knowledge()` method
   - Routes `workflow_knowledge.*` tool calls

4. **src/cli_combined.py** (NEW)
   - Combined interface for workflow search + agent building
   - Verbose mode shows search process

5. **tests/test_workflow_integration.py** (NEW)
   - Comprehensive integration tests

---

