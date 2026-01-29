#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Suite for Unblinded Knowledge Skill
Tests memory scope separation and proper integration
"""

import sys
import os
import json

# Force UTF-8
if sys.platform == "win32":
    import io

    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')


def test_memory_scope_separation():
    """Test that Pinecone doesn't overwrite local/global memory"""
    print("\nTest 1: Memory Scope Separation")
    print("-" * 60)

    from query import query_pinecone_supplemental

    # Create mock memory contexts
    local_context = {
        'user_id': 'test_user',
        'preferences': 'technical communication',
        'history': ['previous query 1', 'previous query 2']
    }

    global_context = {
        'system_role': 'helpful assistant',
        'knowledge_base': 'general facts'
    }

    # Query Pinecone
    query = "What is Sean's teaching method?"
    result = query_pinecone_supplemental(query, local_context, global_context)

    # Verify all layers exist
    assert result is not None, "Query returned None!"
    assert 'memory_layers' in result, "Missing memory_layers!"

    layers = result['memory_layers']
    assert 'local' in layers, "Local memory missing!"
    assert 'global' in layers, "Global memory missing!"
    assert 'unblinded' in layers, "Unblinded memory missing!"

    # Verify local memory wasn't overwritten
    assert layers['local'] == local_context, "Local memory was modified!"
    assert layers['global'] == global_context, "Global memory was modified!"

    print("✅ PASSED: All memory layers preserved")
    print(f"   - Local: {layers['local']['user_id']}")
    print(f"   - Global: {layers['global']['system_role']}")
    print(f"   - Unblinded: {layers['unblinded']['scope']}")

    return True


def test_embedding_model():
    """Test correct embedding model usage"""
    print("\nTest 2: Embedding Model")
    print("-" * 60)

    try:
        from openai import OpenAI
    except ImportError:
        print("⚠️  SKIPPED: OpenAI not installed")
        return True

    api_key = os.environ.get('OPENAI_API_KEY')
    if not api_key:
        print("⚠️  SKIPPED: OPENAI_API_KEY not set")
        return True

    client = OpenAI(api_key=api_key)

    # Test embedding generation
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input="test query"
    )

    embedding = response.data[0].embedding
    expected_size = 1536

    assert len(embedding) == expected_size, f"Wrong embedding size: {len(embedding)} != {expected_size}"

    print(f"✅ PASSED: Correct embedding model")
    print(f"   - Model: text-embedding-3-small")
    print(f"   - Vector size: {len(embedding)}")

    return True


def test_supplemental_integration():
    """Test that Pinecone works as supplemental layer"""
    print("\nTest 3: Supplemental Integration")
    print("-" * 60)

    from query import query_pinecone_supplemental

    # Test without context (should work)
    result1 = query_pinecone_supplemental("test query")
    assert result1 is not None, "Query without context failed!"
    assert result1['memory_layers']['unblinded']['scope'] == 'supplemental'

    print("✅ PASSED: Works as standalone")

    # Test with context (should combine)
    mock_context = {'existing': 'data'}
    result2 = query_pinecone_supplemental("test query", mock_context)
    assert result2['memory_layers']['local'] == mock_context

    print("✅ PASSED: Combines with existing context")

    return True


def test_api_key_handling():
    """Test graceful handling of missing API keys"""
    print("\nTest 4: API Key Handling")
    print("-" * 60)

    # Save original keys
    original_pinecone = os.environ.get('PINECONE_API_KEY')
    original_openai = os.environ.get('OPENAI_API_KEY')

    # Temporarily remove keys
    if original_pinecone:
        del os.environ['PINECONE_API_KEY']
    if original_openai:
        del os.environ['OPENAI_API_KEY']

    from query import query_pinecone_supplemental

    # Should return error dict (not None)
    result = query_pinecone_supplemental("test")

    # FIX: Check for error dict instead of None
    assert result is not None, "Should return result"
    assert 'error' in result, "Should have error key"
    assert 'API keys not set' in result['error'], "Wrong error message"

    # Restore keys
    if original_pinecone:
        os.environ['PINECONE_API_KEY'] = original_pinecone
    if original_openai:
        os.environ['OPENAI_API_KEY'] = original_openai

    print("✅ PASSED: Handles missing API keys gracefully")

    return True


def run_all_tests():
    """Run all tests"""
    print("=" * 70)
    print("UNBLINDED KNOWLEDGE - TEST SUITE")
    print("=" * 70)
    print("Testing memory scope separation and proper integration")

    tests = [
        ("Memory Scope Separation", test_memory_scope_separation),
        ("Embedding Model", test_embedding_model),
        ("Supplemental Integration", test_supplemental_integration),
        ("API Key Handling", test_api_key_handling),
    ]

    passed = 0
    failed = 0

    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"❌ FAILED: {test_name}")
            print(f"   Error: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print(f"Passed: {passed}/{len(tests)}")
    print(f"Failed: {failed}/{len(tests)}")

    if failed == 0:
        print("\n✅ ALL TESTS PASSED")
        print("=" * 70)
        return 0
    else:
        print(f"\n❌ {failed} TEST(S) FAILED")
        print("=" * 70)
        return 1


if __name__ == "__main__":
    sys.exit(run_all_tests())