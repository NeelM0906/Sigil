# Unblinded Knowledge Base (Supplemental Memory)

Query the Pinecone ublib2 database containing Sean's teachings, medical protocols, and training materials as a **supplemental memory layer**.

## Memory Architecture

This skill implements a **3-layer memory system**:
```
┌──────────────────────────────────┐
│ Layer 1: Local Memory            │ ← Per-user context
│   - User preferences             │
│   - Conversation history         │
│   - Personal context             │
└──────────────┬───────────────────┘
               │
┌──────────────▼───────────────────┐
│ Layer 2: Global Memory           │ ← Bot knowledge
│   - System instructions          │
│   - General facts                │
│   - Bot personality              │
└──────────────┬───────────────────┘
               │
┌──────────────▼───────────────────┐
│ Layer 3: Unblinded Knowledge     │ ← THIS SKILL
│   - Sean's teachings (8,955 rec) │
│   - Medical protocols            │
│   - Training transcripts         │
└──────────────────────────────────┘

All 3 layers work together - Pinecone is SUPPLEMENTAL
```

## Usage

### Basic Query (Pinecone Only)
```bash
python ~/clawd/skills/unblinded-knowledge/query.py "What is emotional grounding?"
```

### With Context Integration
```bash
python ~/clawd/skills/unblinded-knowledge/query.py "Sean's method" --with-context
```

### JSON Output
```bash
python ~/clawd/skills/unblinded-knowledge/query.py "teaching" --json
```

## Testing
```bash
# Run full test suite
python ~/clawd/skills/unblinded-knowledge/test.py

# Expected output:
# ✅ Memory Scope Separation
# ✅ Embedding Model
# ✅ Supplemental Integration
# ✅ API Key Handling
```

## Integration with Bot

When the bot receives a query:

1. **Check local memory** (user context)
2. **Check global memory** (bot knowledge)
3. **Query Pinecone** (supplemental unblinded knowledge)
4. **Combine all 3 layers** for final response

Example:
```
User: "How does Sean handle difficult clients?"

Bot process:
1. Local: "User prefers direct answers"
2. Global: "Be helpful and professional"
3. Pinecone: "Sean uses emotional grounding first..."
4. Response: [Combines all 3 layers]
```

## Memory Scope Behavior

### What This Skill DOES:
- ✅ Queries Pinecone for supplemental knowledge
- ✅ Preserves existing local/global memory
- ✅ Returns combined context with all layers
- ✅ Indicates relevance scores

### What This Skill DOES NOT Do:
- ❌ Replace local user memory
- ❌ Overwrite global bot memory
- ❌ Modify existing conversation history
- ❌ Change user preferences

## API Keys Required
```bash
# Windows
set PINECONE_API_KEY=your-pinecone-key
set OPENAI_API_KEY=your-openai-key

# Linux/Mac
export PINECONE_API_KEY=your-pinecone-key
export OPENAI_API_KEY=your-openai-key
```

## Data Source

- **Database**: Pinecone ublib2
- **Records**: 8,955 entries
- **Content**: Sean's teachings, Kumar's protocols, training materials
- **Embedding Model**: text-embedding-3-small (1536 dimensions)

## Performance

- Query latency: ~0.5-1s
- Embedding generation: ~200ms
- Pinecone search: ~300ms
- Top-K results: 5 (configurable)

## Troubleshooting

### "Local memory was lost"
❌ This means the skill is replacing memory incorrectly
✅ Use the fixed `query.py` that preserves all layers

### "Wrong embedding size"
❌ Using wrong embedding model
✅ Use `text-embedding-3-small` (1536 dimensions)

### "No results found"
❌ Query too generic or Pinecone index empty
✅ Try more specific queries or verify Pinecone has data

## Architecture Compliance

This skill follows Sigil's memory architecture:
- **Scope**: `supplemental` (not `primary` or `global`)
- **Mode**: `additive` (not `replacement`)
- **Integration**: Works alongside existing memory layers
- **Testing**: Full test suite included