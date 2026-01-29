#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unblinded Knowledge Query - SUPPLEMENTAL MEMORY SCOPE
Does NOT replace local/global memory - works alongside them
"""

import sys
import os
import json


def query_pinecone_supplemental(query, local_context=None, global_context=None):
    """
    Query Pinecone as SUPPLEMENTAL memory layer

    Args:
        query: User query string
        local_context: User-specific memory (optional)
        global_context: Bot global memory (optional)

    Returns:
        Combined memory context (all 3 layers)
    """

    try:
        from pinecone import Pinecone
        from openai import OpenAI
    except ImportError as e:
        return {
            'error': f'Missing dependency: {e}',
            'install': 'pip install pinecone-client openai'
        }

    # Check API keys
    pinecone_key = os.environ.get('PINECONE_API_KEY')
    openai_key = os.environ.get('OPENAI_API_KEY')

    if not pinecone_key or not openai_key:
        return {
            'error': 'API keys not set',
            'required': ['PINECONE_API_KEY', 'OPENAI_API_KEY']
        }

    # Initialize clients
    try:
        pc = Pinecone(api_key=pinecone_key)
        openai_client = OpenAI(api_key=openai_key)
    except Exception as e:
        return {'error': f'Client initialization failed: {e}'}

    # Generate embedding using correct model
    try:
        response = openai_client.embeddings.create(
            model='text-embedding-3-small',
            input=query
        )
        embedding = response.data[0].embedding
    except Exception as e:
        return {'error': f'Embedding generation failed: {e}'}

    # Query Pinecone
    try:
        index = pc.Index('ublib2')
        results = index.query(
            vector=embedding,
            top_k=5,
            include_metadata=True
        )
    except Exception as e:
        return {'error': f'Pinecone query failed: {e}'}

    # Calculate average relevance
    avg_relevance = sum(m.score for m in results.matches) / len(results.matches) if results.matches else 0

    # Format Pinecone results
    pinecone_context = {
        'source': 'unblinded_knowledge',
        'scope': 'supplemental',  # KEY: This is supplemental, not primary!
        'query': query,
        'avg_relevance': avg_relevance,
        'results': []
    }

    for i, match in enumerate(results.matches, 1):
        pinecone_context['results'].append({
            'rank': i,
            'score': match.score,
            'text': match.metadata.get('text', 'No text'),
            'id': match.id
        })

    # Build combined context (ALL 3 LAYERS - CRITICAL!)
    combined_context = {
        'memory_layers': {
            'local': local_context if local_context is not None else {'note': 'No local context provided'},
            'global': global_context if global_context is not None else {'note': 'No global context provided'},
            'unblinded': pinecone_context
        },
        'integration_mode': 'supplemental',
        'note': 'All memory layers preserved - Pinecone is supplemental'
    }

    return combined_context


if __name__ == "__main__":
    # Only print when running directly (not during tests)
    if len(sys.argv) < 2:
        print("Usage: python query.py <query> [--with-context]")
        print("\nExamples:")
        print("  python query.py 'What is emotional grounding?'")
        print("  python query.py 'Sean teaching method' --with-context")
        sys.exit(1)

    query_text = ' '.join([arg for arg in sys.argv[1:] if not arg.startswith('--')])
    with_context = '--with-context' in sys.argv

    # Mock contexts for testing
    mock_local = {'user': 'test_user', 'preference': 'technical'} if with_context else None
    mock_global = {'system': 'helpful assistant'} if with_context else None

    result = query_pinecone_supplemental(query_text, mock_local, mock_global)

    if result and 'error' not in result:
        # Display results
        print("=" * 70)
        print("UNBLINDED KNOWLEDGE QUERY (Supplemental Memory)")
        print("=" * 70)
        print(f"\nQuery: {query_text}\n")

        print("MEMORY LAYERS:")
        print(f"  Local:     {'Present' if mock_local else 'Not provided'}")
        print(f"  Global:    {'Present' if mock_global else 'Not provided'}")
        print(f"  Unblinded: {len(result['memory_layers']['unblinded']['results'])} results")

        print("\nPINECONE RESULTS:")
        for res in result['memory_layers']['unblinded']['results']:
            print(f"\n  Result {res['rank']} (Score: {res['score']:.3f}):")
            print(f"    {res['text'][:200]}...")

        if '--json' in sys.argv:
            print("\n\nJSON OUTPUT:")
            print(json.dumps(result, indent=2))
    else:
        print(f"\n❌ Query failed: {result.get('error', 'Unknown error')}")
        sys.exit(1)