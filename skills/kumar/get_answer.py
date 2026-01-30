#!/usr/bin/env python3
"""
Get Kumar's answer for Sigil
Returns ONLY the answer, nothing else
"""
import json
import sys
import os


def get_answer(question):
    """Get Kumar's answer"""

    try:
        import numpy as np
    except ImportError:
        sys.stderr.write("ERROR: numpy not installed\n")
        sys.exit(1)

    try:
        from openai import OpenAI
    except ImportError:
        sys.stderr.write("ERROR: openai not installed\n")
        sys.exit(1)

    api_key = os.environ.get('OPENAI_API_KEY')
    if not api_key:
        sys.stderr.write("ERROR: OPENAI_API_KEY not set\n")
        sys.exit(1)

    knowledge_file = 'kumar_clean.json'
    if not os.path.exists(knowledge_file):
        sys.stderr.write(f"ERROR: {knowledge_file} not found\n")
        sys.exit(1)

    try:
        with open(knowledge_file, 'r', encoding='utf-8') as f:
            knowledge = json.load(f)
    except Exception as e:
        sys.stderr.write(f"ERROR: Failed to load knowledge: {e}\n")
        sys.exit(1)

    if not knowledge:
        sys.stderr.write("ERROR: Knowledge base is empty\n")
        sys.exit(1)

    client = OpenAI(api_key=api_key)

    try:
        response = client.embeddings.create(
            model='text-embedding-3-small',
            input=question
        )
        q_emb = np.array(response.data[0].embedding)
    except Exception as e:
        sys.stderr.write(f"ERROR: Failed to generate embedding: {e}\n")
        sys.exit(1)

    # Find best match
    best_match = None
    best_score = 0

    for item in knowledge:
        try:
            k_emb = np.array(item['embedding'])
            similarity = np.dot(q_emb, k_emb) / (np.linalg.norm(q_emb) * np.linalg.norm(k_emb))

            if similarity > best_score:
                best_score = similarity
                best_match = item
        except Exception as e:
            sys.stderr.write(f"WARNING: Error processing item: {e}\n")
            continue

    # Only return if good match
    if best_match and best_score > 0.5:
        return best_match['agent_response']

    return ""


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.stderr.write("ERROR: No question provided\n")
        sys.exit(1)

    question = ' '.join(sys.argv[1:])

    try:
        answer = get_answer(question)

        if answer:
            print(answer)
        else:
            sys.stderr.write("No relevant answer found\n")
            sys.exit(1)
    except Exception as e:
        sys.stderr.write(f"FATAL ERROR: {e}\n")
        sys.exit(1)