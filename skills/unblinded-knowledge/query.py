#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import os
from pinecone import Pinecone
from openai import OpenAI

# Force UTF-8 encoding
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

# Get API keys
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')

if not PINECONE_API_KEY or not OPENAI_API_KEY:
    print("ERROR: Missing API keys")
    sys.exit(1)

# Initialize
pc = Pinecone(api_key=PINECONE_API_KEY)
openai_client = OpenAI(api_key=OPENAI_API_KEY)

# Get query
query = sys.argv[1] if len(sys.argv) > 1 else "test query"

try:
    print("=" * 60)
    print("🔍 UNBLINDED KNOWLEDGE MODE - PINECONE UBLIB2 EXCLUSIVE")
    print("=" * 60)
    print(f"\nQuery: \"{query}\"\n")

    # Generate embedding
    response = openai_client.embeddings.create(
        model='text-embedding-ada-002',
        input=query
    )
    embedding = response.data[0].embedding

    # Query Pinecone
    index = pc.Index('ublib2')
    results = index.query(
        vector=embedding,
        top_k=5,
        include_metadata=True
    )

    if results.matches:
        print("📊 RESULTS:\n")

        for i, match in enumerate(results.matches, 1):
            score = match.score
            text = match.metadata.get('text', 'No text')

            print(f"Result {i} (Relevance: {score:.3f})")
            print(f"{text[:300]}...\n")

        # Calculate average
        avg = sum(m.score for m in results.matches) / len(results.matches)
        print(f"Average Relevance: {avg:.3f}")

        if avg < 0.3:
            print("\n⚠️  LOW RELEVANCE WARNING")
            print("These results may not directly answer your question.")
            print("Say 'exit unblinded mode' to use other sources.\n")
        else:
            print("\n✅ Good relevance\n")

    else:
        print("❌ No results found in Pinecone ublib2\n")

    print("=" * 60)
    print("🔒 UPLOADED DOCUMENTS ARE IGNORED IN THIS MODE")
    print("=" * 60)

except Exception as e:
    print(f"❌ Error: {e}")
    sys.exit(1)
