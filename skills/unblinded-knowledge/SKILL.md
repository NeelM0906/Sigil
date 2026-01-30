---
name: unblinded_knowledge
description: Query Pinecone ublib2 knowledge base - EXCLUSIVE MODE that ignores uploaded documents
---

# Unblinded Knowledge Mode

## MODE BEHAVIOR

When this skill is activated, you enter **EXCLUSIVE PINECONE MODE**:

✅ DO:
- Query Pinecone ublib2 ONLY
- Show relevance scores
- Cite "Pinecone ublib2 knowledge base"
- If relevance < 0.3, say: "Pinecone has low relevance. Exit this mode to use other sources."

❌ DO NOT:
- Reference ANY uploaded documents (PDFs, DOCX, etc.)
- Mention "your documents", "uploaded files", "from the documents"
- Combine Pinecone results with document information
- Use document knowledge unless user explicitly exits this mode

## Activation Phrases

User says:
- "activate unblinded knowledge"
- "query pinecone"  
- "use ublib2 mode"

## Deactivation

User says:
- "exit unblinded mode"
- "use my documents"
- "normal mode"

## Query Command
```bash
python ~/clawd/skills/unblinded-knowledge/query.py "user question"
```

Or use relative path:
```bash
python ./skills/unblinded-knowledge/query.py "user question"
```

## Response Template
```
🔍 Pinecone ublib2 Query: "[user question]"

📊 Results:
Result 1 (Relevance: 0.XX): [content preview]
Result 2 (Relevance: 0.XX): [content preview]
Result 3 (Relevance: 0.XX): [content preview]

Average Relevance: 0.XX

💡 Based on Pinecone ublib2:
[Synthesize ONLY what Pinecone returned]

[If avg relevance < 0.3:]
⚠️ Low relevance detected. These results may not directly answer your question.
Say "exit unblinded mode" to use other knowledge sources.
```

## Example: Good Response
```
User: "What is predictive diagnostics?"

🔍 Pinecone Query: "predictive diagnostics"

📊 Results (Avg: 0.42):
Result 1 (0.49): Predictive Diagnostic asks whether teaching 
creates lasting installation vs temporary understanding...

💡 Based on Pinecone:
Predictive Diagnostics is a framework that evaluates whether 
teaching creates emotional grounding that anchors information 
to feeling, rather than just transferring information.
```

## Example: Bad Response (DON'T DO THIS)
```
❌ Combined Answer:
❌ From your uploaded documents about Dr. Kumar...
❌ Based on the medical intake project...
❌ [Any mention of documents, PDFs, uploads]
```

## Mode Persistence

Stay in this mode until user explicitly exits.
Track mode state across conversation turns.