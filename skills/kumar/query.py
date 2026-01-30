#!/usr/bin/env python3
"""
Query Kumar's knowledge - Simple version for Sigil
"""
import json
import os
import sys
import numpy as np
from openai import OpenAI


def query(question, return_context=False):
    """Query Kumar's knowledge base"""

    api_key = os.environ.get('OPENAI_API_KEY')
    if not api_key:
        return None

    # Load knowledge
    if not os.path.exists('kumar_knowledge.json'):
        return None

    with open('kumar_knowledge.json', 'r', encoding='utf-8') as f:
        knowledge = json.load(f)

    # Get question embedding
    client = OpenAI(api_key=api_key)
    try:
        response = client.embeddings.create(
            model='text-embedding-3-small',
            input=question
        )
        q_emb = np.array(response.data[0].embedding)
    except:
        return None

    # Find similar conversations
    results = []
    for item in knowledge:
        k_emb = np.array(item['embedding'])
        similarity = np.dot(q_emb, k_emb) / (np.linalg.norm(q_emb) * np.linalg.norm(k_emb))
        results.append({
            'similarity': float(similarity),
            'kumar_said': item['kumar_said'],
            'agent_responded': item['agent_responded'],
            'date': item['date'],
            'type': item.get('type', 'historical')
        })

    # Sort and get top 3
    results.sort(key=lambda x: x['similarity'], reverse=True)
    top_3 = results[:3]

    # If called programmatically, return data
    if return_context:
        return top_3

    # If called from command line, display
    print(f"\nSearching {len(knowledge)} Kumar examples...")
    print("\n" + "=" * 70)
    print("RELEVANT KUMAR KNOWLEDGE:")
    print("=" * 70)

    for i, r in enumerate(top_3, 1):
        print(f"\nMatch {i} (Similarity: {r['similarity']:.3f}) [{r['type']}]")
        print(f"Date: {r['date']}")
        print(f"\nKumar: {r['kumar_said'][:200]}...")
        print(f"\nAgent: {r['agent_responded'][:200]}...")
        print("-" * 70)

    return top_3


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python query.py 'your question'")
        sys.exit(1)

    question = ' '.join(sys.argv[1:])
    query(question)