#!/usr/bin/env python3
"""
Clean setup - Extract ONLY Kumar's actual messages from CSV
"""
import csv
import json
import os
from openai import OpenAI


def clean_and_setup():
    """Extract clean Kumar conversations"""

    api_key = os.environ.get('OPENAI_API_KEY')
    if not api_key:
        print("ERROR: Set OPENAI_API_KEY")
        return False

    csv_file = 'messages_rows_filtered.csv'
    if not os.path.exists(csv_file):
        print(f"ERROR: {csv_file} not found")
        return False

    print("Loading CSV...")
    with open(csv_file, 'r', encoding='utf-8') as f:
        messages = list(csv.DictReader(f))

    print(f"Total messages: {len(messages)}")

    # Extract CLEAN conversation pairs
    conversations = []
    i = 0
    while i < len(messages) - 1:
        user_msg = messages[i]
        agent_msg = messages[i + 1]

        if user_msg['role'] == 'user' and agent_msg['role'] == 'assistant':
            # Clean the messages - remove system timestamps and noise
            user_text = user_msg['message'].strip()
            agent_text = agent_msg['message'].strip()

            # Skip if it's just system messages
            if user_text.startswith('System:') or len(user_text) < 20:
                i += 2
                continue

            # Skip if agent response is too short
            if len(agent_text) < 50:
                i += 2
                continue

            conversations.append({
                'user': user_text,
                'agent': agent_text,
                'date': user_msg['created_at']
            })
            i += 2
        else:
            i += 1

    print(f"Extracted {len(conversations)} clean conversations")

    if len(conversations) == 0:
        print("ERROR: No valid conversations found!")
        return False

    # Generate embeddings
    print("Generating embeddings...")
    client = OpenAI(api_key=api_key)

    knowledge = []
    for i, conv in enumerate(conversations, 1):
        # Combine for embedding
        text = f"Question: {conv['user']}\n\nAnswer: {conv['agent']}"

        try:
            response = client.embeddings.create(
                model='text-embedding-3-small',
                input=text
            )

            knowledge.append({
                'user_message': conv['user'],
                'agent_response': conv['agent'],
                'date': conv['date'],
                'embedding': response.data[0].embedding
            })

            if i % 5 == 0:
                print(f"  Processed {i}/{len(conversations)}")
        except Exception as e:
            print(f"  Error on conversation {i}: {e}")
            continue

    # Save
    print(f"\nSaving {len(knowledge)} conversations...")
    with open('kumar_clean.json', 'w', encoding='utf-8') as f:
        json.dump(knowledge, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Done! Created kumar_clean.json")
    print(f"   File size: {os.path.getsize('kumar_clean.json') / 1024 / 1024:.2f} MB")
    return True


if __name__ == "__main__":
    clean_and_setup()