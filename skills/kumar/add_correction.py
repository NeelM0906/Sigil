#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Add Kumar's live corrections to the knowledge base
"""

import json
import sys
import os
from datetime import datetime


def add_correction(situation, kumar_correction):
    """Add a new correction from Kumar"""

    try:
        from openai import OpenAI
    except ImportError:
        print("❌ Missing: pip install openai")
        return False

    api_key = os.environ.get('OPENAI_API_KEY')
    if not api_key:
        print("❌ OPENAI_API_KEY not set")
        return False

    # Load existing embeddings
    embeddings_file = 'kumar_embeddings.json'
    if not os.path.exists(embeddings_file):
        print(f"❌ {embeddings_file} not found! Run setup.py first")
        return False

    with open(embeddings_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print("=" * 70)
    print("ADD KUMAR CORRECTION")
    print("=" * 70)

    # Create new entry
    text = f"Situation: {situation}\n\nKumar's Correction: {kumar_correction}"

    # Generate embedding
    print("\n1. Generating embedding...")
    openai_client = OpenAI(api_key=api_key)

    response = openai_client.embeddings.create(
        model='text-embedding-3-small',
        input=text
    )

    new_entry = {
        'id': f'correction_{int(datetime.now().timestamp())}',
        'text': text,
        'kumar_message': situation,
        'assistant_response': kumar_correction,
        'timestamp': datetime.now().isoformat(),
        'type': 'live_correction',
        'embedding': response.data[0].embedding
    }

    # Add to data
    data.append(new_entry)

    # Save back
    print("2. Saving to file...")
    with open(embeddings_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)

    print("\n" + "=" * 70)
    print("✓ Correction added successfully!")
    print(f"  ID: {new_entry['id']}")
    print(f"  Total entries: {len(data)}")
    print(f"  File: {embeddings_file}")
    print("=" * 70)

    return True


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python add_correction.py '<situation>' '<kumar_correction>'")
        print("\nExample:")
        print("  python add_correction.py 'Patient worried about cost' 'Validate emotion first, then explain value'")
        sys.exit(1)

    situation = sys.argv[1]
    correction = sys.argv[2]

    add_correction(situation, correction)
