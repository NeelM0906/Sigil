#!/usr/bin/env python3
"""
Setup: Load Kumar's teaching conversations from CSV into embeddings
"""

import csv
import json
import os
import sys


def setup():
    from openai import OpenAI

    api_key = os.environ.get('OPENAI_API_KEY')
    if not api_key:
        print("ERROR: Set OPENAI_API_KEY")
        return False

    csv_file = 'messages_rows_filtered.csv'
    if not os.path.exists(csv_file):
        print(f"ERROR: {csv_file} not found")
        return False

    print("Loading Kumar's teaching conversations...")

    # Load CSV
    with open(csv_file, 'r', encoding='utf-8') as f:
        messages = list(csv.DictReader(f))

    # Extract conversation pairs (Kumar + Agent response)
    conversations = []
    i = 0
    while i < len(messages) - 1:
        if messages[i]['role'] == 'user' and messages[i + 1]['role'] == 'assistant':
            conversations.append({
                'kumar_said': messages[i]['message'],
                'agent_responded': messages[i + 1]['message'],
                'date': messages[i]['created_at']
            })
            i += 2
        else:
            i += 1

    print(f"Found {len(conversations)} teaching examples")

    # Generate embeddings
    print("Generating embeddings...")
    client = OpenAI(api_key=api_key)

    data = []
    for i, conv in enumerate(conversations, 1):
        # Combine both sides for context
        text = f"Kumar: {conv['kumar_said']}\nAgent: {conv['agent_responded']}"

        response = client.embeddings.create(
            model='text-embedding-3-small',
            input=text
        )

        data.append({
            'kumar_said': conv['kumar_said'],
            'agent_responded': conv['agent_responded'],
            'date': conv['date'],
            'embedding': response.data[0].embedding
        })

        print(f"  {i}/{len(conversations)}")

    # Save
    with open('kumar_knowledge.json', 'w') as f:
        json.dump(data, f, indent=2)

    print(f"\nDone! Created kumar_knowledge.json with {len(data)} examples")
    return True


if __name__ == "__main__":
    setup()