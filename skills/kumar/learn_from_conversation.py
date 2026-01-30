#!/usr/bin/env python3
"""
AUTOMATIC: Learn from new Kumar conversations
Called automatically after each conversation
"""

import json
import os
from datetime import datetime
from openai import OpenAI


def learn_from_conversation(user_message, bot_response, user_is_kumar=False):
    """
    Automatically add new conversation to knowledge base

    Args:
        user_message: What user said
        bot_response: What bot replied
        user_is_kumar: True if this is Kumar chatting
    """

    api_key = os.environ.get('OPENAI_API_KEY')
    if not api_key:
        return False

    # Load existing knowledge
    knowledge_file = 'kumar_knowledge.json'
    if not os.path.exists(knowledge_file):
        print("Warning: kumar_knowledge.json not found, creating new")
        knowledge = []
    else:
        with open(knowledge_file, 'r') as f:
            knowledge = json.load(f)

    # Create new entry
    text = f"Kumar: {user_message}\nAgent: {bot_response}"

    # Generate embedding
    client = OpenAI(api_key=api_key)
    response = client.embeddings.create(
        model='text-embedding-3-small',
        input=text
    )

    # Add to knowledge
    new_entry = {
        'kumar_said': user_message,
        'agent_responded': bot_response,
        'date': datetime.now().isoformat(),
        'type': 'live_conversation' if user_is_kumar else 'user_conversation',
        'embedding': response.data[0].embedding
    }

    knowledge.append(new_entry)

    # Save back
    with open(knowledge_file, 'w') as f:
        json.dump(knowledge, f, indent=2)

    print(f"[AUTO-LEARN] Added conversation to knowledge base (Total: {len(knowledge)})")
    return True


if __name__ == "__main__":
    # Test
    import sys

    if len(sys.argv) >= 3:
        user_msg = sys.argv[1]
        bot_resp = sys.argv[2]
        is_kumar = '--kumar' in sys.argv
        learn_from_conversation(user_msg, bot_resp, is_kumar)