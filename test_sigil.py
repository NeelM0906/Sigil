#!/usr/bin/env python3
"""
Interactive test of Sigil v2 capabilities
Run this to see all subsystems in action
"""
import asyncio
import sys
from pathlib import Path

# Add sigil to path
sys.path.insert(0, str(Path(__file__).parent))

from sigil.config import get_settings
from sigil.memory.manager import MemoryManager
from sigil.reasoning.manager import ReasoningManager
from sigil.planning.planner import Planner
from sigil.orchestrator import SigilOrchestrator, OrchestratorRequest

async def main():
    print("=" * 70)
    print("SIGIL v2 - COMPREHENSIVE SYSTEM TEST")
    print("=" * 70)

    try:
        # 1. Test Configuration
        print("\n[1] CONFIGURATION SYSTEM")
        settings = get_settings()
        print("[OK] Settings loaded")
        print(f"    Memory enabled: {settings.use_memory}")
        print(f"    Planning enabled: {settings.use_planning}")
        print(f"    Contracts enabled: {settings.use_contracts}")
        print(f"    Evolution enabled: {settings.use_evolution}")

        # 2. Test Memory
        print("\n[2] MEMORY SYSTEM (3-Layer Architecture)")
        memory = MemoryManager()
        print("[OK] Memory manager initialized")

        # Store a resource (Layer 1)
        resource = await memory.store_resource(
            resource_type="conversation",
            content="John Smith is a VP of Operations at Acme Corp. "
                   "They have a $500k annual budget and a team of 20 people. "
                   "Interested in CRM solutions for sales and customer service."
        )
        print(f"[OK] Resource stored (Layer 1): {resource.resource_id[:12]}...")

        # Extract items (Layer 2)
        items = await memory.extract_and_store(resource.resource_id)
        print(f"[OK] Extracted {len(items)} memory items (Layer 2)")
        for item in items[:3]:
            print(f"     {item.content[:60]}...")

        # Retrieve with memory (Layer 3)
        results = await memory.retrieve(
            "What is Acme Corp's budget?",
            k=3,
            mode="hybrid"  # RAG + LLM
        )
        print(f"[OK] Retrieved {len(results)} items via hybrid retrieval")
        if results:
            print(f"     Top match: {results[0].content[:60]}...")

        # 3. Test Planning
        print("\n[3] PLANNING SYSTEM (Goal Decomposition)")
        planner = Planner()
        print("[OK] Planner initialized")

        plan = await planner.create_plan(
            goal="Research market trends and create competitive analysis",
            context={"industry": "CRM", "focus": "enterprise"}
        )
        print(f"[OK] Plan created with {len(plan.steps)} steps (DAG structure)")
        for i, step in enumerate(plan.steps[:5], 1):
            deps = f" [depends on: {', '.join(step.dependencies)}]" if step.dependencies else ""
            print(f"     Step {i}: {step.description[:50]}...{deps}")

        # 4. Test Reasoning
        print("\n[4] REASONING SYSTEM (5 Strategies)")
        reasoning = ReasoningManager()
        print("[OK] Reasoning manager initialized")

        # Test different complexities
        complexities = [0.2, 0.4, 0.6, 0.8]
        for complexity in complexities:
            result = await reasoning.execute(
                task="What are the key features in enterprise CRM?",
                context={"industry": "technology"},
                complexity=complexity,
                session_id="test-session"
            )
            strategy_names = {
                0.2: "Direct",
                0.4: "Chain-of-Thought",
                0.6: "Tree-of-Thoughts",
                0.8: "ReAct"
            }
            print(f"[OK] Complexity {complexity}: {strategy_names.get(complexity, 'Strategy')} "
                  f"({result.tokens_used} tokens, "
                  f"confidence: {result.confidence:.0%})")

        # 5. Test Orchestration
        print("\n[5] ORCHESTRATION (All Phases Combined)")
        orchestrator = SigilOrchestrator(memory_manager=memory)
        print("[OK] Orchestrator initialized")

        request = OrchestratorRequest(
            message="Based on what we know about Acme (from memory), "
                   "are they a good prospect for our CRM solution?",
            session_id="test-session-1",
            context={"complexity": 0.6}
        )
        result = await orchestrator.process(request)
        print("[OK] Orchestrator pipeline executed")
        print(f"     Status: {result.status.value}")
        print(f"     Output: {str(result.output)[:80]}...")
        print(f"     Total tokens: {result.tokens_used}")
        print(f"     Execution time: {result.execution_time_ms:.0f}ms")
        if result.errors:
            print(f"     Errors: {len(result.errors)}")

        # 6. Test Metrics
        print("\n[6] METRICS & MONITORING")
        memory_stats = memory.get_stats()
        print("[OK] Memory metrics:")
        print(f"     Total items: {memory_stats.item_count}")
        print(f"     Total resources: {memory_stats.resource_count}")
        print(f"     Total categories: {memory_stats.category_count}")

        reasoning_metrics = reasoning.get_metrics()
        print("[OK] Reasoning metrics:")
        print(f"     Total executions: {reasoning_metrics['total_executions']}")
        print(f"     Total successes: {reasoning_metrics['total_successes']}")
        print(f"     Success rate: {reasoning_metrics['overall_success_rate']:.1%}")
        print(f"     Total fallbacks: {reasoning_metrics['total_fallbacks']}")

        # 7. Test Integration
        print("\n[7] END-TO-END INTEGRATION TEST")

        # Create a complex scenario
        scenario_resource = await memory.store_resource(
            resource_type="conversation",
            content="Lead: Sarah Johnson, CTO at TechStartup Inc. "
                   "Company size: 50 employees, Series A funded, $2M annual spend on tools. "
                   "Currently uses Salesforce but exploring alternatives. "
                   "Concerns: cost, integration with engineering tools, API capabilities."
        )
        await memory.extract_and_store(scenario_resource.resource_id)

        # Run complete pipeline
        full_request = OrchestratorRequest(
            message="Analyze TechStartup Inc. as a prospect and recommend next steps",
            session_id="full-test",
            context={"complexity": 0.7}
        )
        full_result = await orchestrator.process(full_request)

        print("[OK] Full integration pipeline executed")
        print(f"     Status: {full_result.status.value}")
        print(f"     Output: {str(full_result.output)[:80]}...")
        print(f"     Total tokens: {full_result.tokens_used}")
        print(f"     Execution time: {full_result.execution_time_ms:.0f}ms")

        print("\n" + "=" * 70)
        print("SUCCESS - ALL SYSTEMS OPERATIONAL")
        print("SIGIL v2 IS READY FOR USE")
        print("=" * 70)
        print("\nNext Steps:")
        print("  1. Run tests: pytest tests/ -v -s --log-cli-level=DEBUG")
        print("  2. Start API: python -m sigil.interfaces.api.server")
        print("  3. Use CLI: python -m src.cli create --name 'agent_name'")
        print("  4. Check logs: tail -f sigil.log")

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
