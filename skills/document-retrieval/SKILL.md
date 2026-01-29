---
name: knowledge-rag
description: "Search documents and the Pinecone knowledge base. Upload documents for RAG retrieval, search user documents with hybrid search, and query the always-available knowledge base."
metadata: {"sigil":{"emoji":"📚"}}
---

# Knowledge RAG Skill

Search through uploaded documents and the Pinecone knowledge base. The knowledge base is always available and provides domain-specific information.

## Available Tools

### Document Tools

**document_upload** - Store a document for later retrieval
```
Parameters:
- file_path: Path to the uploaded file (PDF, DOCX, TXT, MD)
- user_id: User ID for document isolation
```

**document_search** - Search through stored documents using hybrid search (keyword + semantic)
```
Parameters:
- query: Search query
- user_id: User ID
- top_k: Number of results (default: 5)
```

**document_list** - List all stored documents for a user
```
Parameters:
- user_id: User ID
```

### Knowledge Base Tools

**knowledge_search** - Search the Pinecone knowledge base
```
Parameters:
- query: Search query
- top_k: Number of results (default: 5)
```

**knowledge_status** - Check if the knowledge base is available and get statistics

## Auto-Recall

The knowledge base automatically injects relevant context before agent responses when:
- The user's query is longer than 10 characters
- The query doesn't start with "/" (not a command)
- Relevant results are found with score above the minimum threshold

## How It Works

1. **Document RAG**: When users upload documents, the text is extracted, chunked, and embedded. Search uses a hybrid approach combining TF-IDF keyword matching (40%) and semantic similarity (60%).

2. **Knowledge Base**: The Pinecone vector database stores domain-specific knowledge that's always available. Results are automatically injected into context when relevant.

## Example Usage

Search the knowledge base:
```
Use the knowledge_search tool to find information about [topic]
```

Search uploaded documents:
```
Use the document_search tool to search my documents for [query]
```

List my documents:
```
Use the document_list tool to show my uploaded documents
```
