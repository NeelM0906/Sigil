#!/usr/bin/env python3
"""
Simple query - Get complete Kumar answers
"""
import json
import sys
import os
import numpy as np
from openai import OpenAI


def search_kumar(question):
    """Search Kumar's knowledge and return complete answer"""

    api_key = os.environ.get('OPENAI_API_KEY')
    if not api_key:
        return None

    # Load knowledge
    if not os.path.exists('kumar_clean.json'):
        return None

    with open('kumar_clean.json', 'r', encoding='utf-8') as f:
        knowledge = json.load(f)

    # Generate question embedding
    client = OpenAI(api_key=api_key)

    try:
        response = client.embeddings.create(
            model='text-embedding-3-small',
            input=question
        )
        q_emb = np.array(response.data[0].embedding)
    except:
        return None

    # Find most similar
    best_match = None
    best_score = 0

    for item in knowledge:
        k_emb = np.array(item['embedding'])
        similarity = np.dot(q_emb, k_emb) / (np.linalg.norm(q_emb) * np.linalg.norm(k_emb))

        if similarity > best_score:
            best_score = similarity
            best_match = item

    if best_match and best_score > 0.4:
        return {
            'answer': best_match['agent_response'],
            'similarity': float(best_score),
            'original_question': best_match['user_message']
        }

    return None


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(1)

    question = ' '.join(sys.argv[1:])
    result = search_kumar(question)

    if result:
        print("\n" + "=" * 70)
        print(f"KUMAR'S ANSWER (Relevance: {result['similarity']:.2f})")
        print("=" * 70)
        print(f"\nOriginal Question: {result['original_question'][:200]}...")
        print(f"\n{result['answer']}")
        print("=" * 70)
    else:
        print("No relevant Kumar knowledge found.")