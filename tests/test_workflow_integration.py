"""
Complete Test Suite for Sigil + ACTi Integration

Run this to test the complete integration end-to-end.

Usage:
    cd Sigil-neel_dev
    python tests/test_workflow_integration.py
"""

import asyncio
import sys
import os

# Add Sigil to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_1_import_workflow_knowledge():
    """Test 1: Can we import the workflow knowledge tool?"""
    print("=" * 60)
    print("TEST 1: Import Workflow Knowledge Tool")
    print("=" * 60)
    
    try:
        from sigil.tools.workflow_knowledge import create_workflow_knowledge_tool
        print("✅ Successfully imported workflow_knowledge module")
        return True
    except ImportError as e:
        print(f"❌ Failed to import: {e}")
        print("\nMake sure:")
        print("1. workflow_knowledge.py is in sigil/tools/")
        print("2. ACTi Router is in correct location")
        return False


def test_2_create_tool():
    """Test 2: Can we create the tool instance?"""
    print("\n" + "=" * 60)
    print("TEST 2: Create Tool Instance")
    print("=" * 60)
    
    try:
        from sigil.tools.workflow_knowledge import create_workflow_knowledge_tool
        
        print("Creating workflow knowledge tool...")
        tool = create_workflow_knowledge_tool(index_calls=False)
        
        print("✅ Tool created successfully")
        print(f"   Pathways indexed: {len(tool.retriever.pathway_data)}")
        
        return True, tool
    except Exception as e:
        print(f"❌ Failed to create tool: {e}")
        print("\nMake sure:")
        print("1. Data directory exists at: Autogen-Framework-autogen-bland-json/data/")
        print("2. Bland pathway files exist in data/bland_dataset/")
        print("3. OPENAI_API_KEY is set in environment")
        return False, None


def test_3_search(tool):
    """Test 3: Can we search for workflows?"""
    print("\n" + "=" * 60)
    print("TEST 3: Search Functionality")
    print("=" * 60)
    
    try:
        query = "How to handle objections?"
        print(f"Query: {query}")
        
        results = tool.search(query, top_k=3)
        
        print(f"\n✅ Search returned {len(results)} results:")
        for i, result in enumerate(results, 1):
            print(f"   {i}. {result.source_name}")
            print(f"      Relevance: {result.relevance_score:.0%}")
        
        return True
    except Exception as e:
        print(f"❌ Search failed: {e}")
        return False


def test_4_ask_question(tool):
    """Test 4: Can we ask questions?"""
    print("\n" + "=" * 60)
    print("TEST 4: Question Answering")
    print("=" * 60)
    
    try:
        question = "What's a good cold call opening?"
        print(f"Question: {question}")
        
        answer = tool.ask(question, max_results=2)
        
        print("\n✅ Question answered:")
        print(answer[:300] + "...\n")
        
        return True
    except Exception as e:
        print(f"❌ Question answering failed: {e}")
        return False


def test_5_get_examples(tool):
    """Test 5: Can we get examples?"""
    print("\n" + "=" * 60)
    print("TEST 5: Get Examples")
    print("=" * 60)
    
    try:
        topic = "email collection"
        print(f"Topic: {topic}")
        
        examples = tool.get_examples(topic, top_k=2)
        
        print("\n✅ Examples retrieved:")
        print(examples[:400] + "...\n")
        
        return True
    except Exception as e:
        print(f"❌ Get examples failed: {e}")
        return False


def test_6_router_integration():
    """Test 6: Does the router detect workflow knowledge queries?"""
    print("\n" + "=" * 60)
    print("TEST 6: Router Integration")
    print("=" * 60)
    
    try:
        from sigil.routing.router import IntentClassifier
        
        classifier = IntentClassifier()
        
        test_queries = [
            "How do I handle objections?",
            "Show me examples of email collection",
            "What's a good cold call script?"
        ]
        
        all_correct = True
        for query in test_queries:
            intent = classifier.classify(query)
            is_correct = intent == "WORKFLOW_KNOWLEDGE"
            status = "✅" if is_correct else "❌"
            
            print(f"{status} '{query}' -> {intent}")
            
            if not is_correct:
                all_correct = False
        
        if all_correct:
            print("\n✅ Router correctly detects workflow knowledge queries")
            return True
        else:
            print("\n⚠️  Router needs configuration")
            print("   See router_integration.py for instructions")
            return False
            
    except Exception as e:
        print(f"❌ Router test failed: {e}")
        print("   Router integration may not be complete")
        return False


def test_7_tool_executor_integration():
    """Test 7: Can tool executor call workflow knowledge?"""
    print("\n" + "=" * 60)
    print("TEST 7: Tool Executor Integration")
    print("=" * 60)
    
    try:
        from sigil.planning.tool_executor import ToolStepExecutor
        
        executor = ToolStepExecutor()
        
        # Check if method exists
        if hasattr(executor, '_execute_workflow_knowledge'):
            print("✅ Tool executor has workflow knowledge method")
            return True
        else:
            print("⚠️  Tool executor needs integration")
            print("   See tool_executor_integration.py for instructions")
            return False
            
    except Exception as e:
        print(f"❌ Tool executor test failed: {e}")
        print("   Tool executor integration may not be complete")
        return False


async def test_8_end_to_end():
    """Test 8: End-to-end workflow knowledge query"""
    print("\n" + "=" * 60)
    print("TEST 8: End-to-End Test")
    print("=" * 60)
    
    try:
        from sigil.tools.workflow_knowledge import create_workflow_knowledge_tool
        
        # Simulate complete flow
        print("Simulating user query: 'How to handle price objections?'")
        
        # Step 1: Create tool
        tool = create_workflow_knowledge_tool(index_calls=False)
        print("   1. Tool created ✓")
        
        # Step 2: Search
        results = tool.search("handle price objections", top_k=2)
        print(f"   2. Search completed ({len(results)} results) ✓")
        
        # Step 3: Format answer
        answer = tool.ask("How to handle price objections?", max_results=2)
        print(f"   3. Answer generated ({len(answer)} chars) ✓")
        
        # Show answer preview
        print("\n   Answer preview:")
        lines = answer.split('\n')[:10]
        for line in lines:
            print(f"   {line}")
        print("   ...")
        
        print("\n✅ End-to-end test passed!")
        return True
        
    except Exception as e:
        print(f"❌ End-to-end test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("\n" + "🧪" * 30)
    print("SIGIL + ACTi INTEGRATION TEST SUITE")
    print("🧪" * 30 + "\n")
    
    results = {}
    
    # Test 1: Import
    results['import'] = test_1_import_workflow_knowledge()
    if not results['import']:
        print("\n❌ Cannot proceed without successful import")
        return
    
    # Test 2: Create tool
    success, tool = test_2_create_tool()
    results['create_tool'] = success
    if not success:
        print("\n❌ Cannot proceed without tool creation")
        return
    
    # Test 3-5: Tool functionality
    results['search'] = test_3_search(tool)
    results['ask'] = test_4_ask_question(tool)
    results['examples'] = test_5_get_examples(tool)
    
    # Test 6-7: Sigil integration
    results['router'] = test_6_router_integration()
    results['executor'] = test_7_tool_executor_integration()
    
    # Test 8: End-to-end
    results['e2e'] = asyncio.run(test_8_end_to_end())
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test_name, passed_test in results.items():
        status = "✅ PASSED" if passed_test else "❌ FAILED"
        print(f"{status}: {test_name}")
    
    print("\n" + "=" * 60)
    print(f"TOTAL: {passed}/{total} tests passed")
    print("=" * 60)
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! Integration is complete.")
        print("\nYou can now use workflow knowledge in Sigil CLI:")
        print('  >>> How to handle objections?')
        print('  >>> Show me examples of email collection')
        print('  >>> What\'s a good cold call script?')
    else:
        print("\n⚠️  Some tests failed. See instructions above.")
        print("\nFiles to check:")
        print("  1. sigil/tools/workflow_knowledge.py")
        print("  2. sigil/planning/tool_executor.py")
        print("  3. sigil/routing/router.py")


if __name__ == "__main__":
    main()
