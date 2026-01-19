"""
Combined CLI - Agent Builder + Workflow Knowledge
With verbose workflow search integration and auto-fixes

Usage:
    python -m src.cli_combined
"""

import os
import sys
import signal
from pathlib import Path
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

from sigil.routing.router import Router, Intent
from sigil.config import get_settings

# Import builder components
from src.builder import create_builder, extract_text_from_content

# Import workflow knowledge
try:
    from sigil.tools.workflow_knowledge import create_workflow_knowledge_tool
    WORKFLOW_AVAILABLE = True
except ImportError:
    WORKFLOW_AVAILABLE = False
    print("⚠️  Workflow knowledge not available - check ACTi Router path")


class CombinedCLI:
    """Combined CLI with agent builder and workflow knowledge."""

    def __init__(self):
        # Load environment
        env_path = Path(__file__).parent.parent / ".env"
        load_dotenv(env_path)

        # Verify API key
        if not os.getenv("ANTHROPIC_API_KEY"):
            print("ERROR: ANTHROPIC_API_KEY environment variable not set.")
            print("Please set it in your .env file or environment.")
            sys.exit(1)

        # Initialize router
        self.router = Router(get_settings())

        # Initialize builder
        print("Creating builder agent...")
        self.builder = create_builder()
        print("Builder ready.")

        # Workflow tool (lazy load)
        self.workflow_tool = None

        # Conversation history for builder
        self.messages = []

        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        # Print header
        print("\n" + "=" * 70)
        print("ACTi Combined CLI - Builder + Workflow Knowledge")
        print("=" * 70)
        print("\n🎯 Capabilities:")
        print("  1. 📚 Answer workflow questions from YOUR Bland/n8n data")
        print("  2. 🔨 Build agents using best practices from existing workflows")
        print("  3. 🔍 Show what was found in your workflows before building")
        print("\n💡 Examples:")
        print("  • 'How do I handle objections?' → Searches your workflows")
        print("  • 'Create an agent for...' → Searches workflows, THEN builds")
        print("\n⚙️  Features:")
        print("  • Verbose mode shows matching content from your data")
        print("  • Agent builder uses workflow knowledge as context")
        print("  • Type 'quit' or 'exit' to end")
        print("=" * 70 + "\n")

    def _signal_handler(self, sig, frame):
        """Handle shutdown signals."""
        print("\n\nShutting down gracefully...")
        sys.exit(0)

    def _get_workflow_tool(self):
        """Lazy load workflow knowledge tool."""
        if self.workflow_tool is None and WORKFLOW_AVAILABLE:
            print("\n[🔍 Loading workflow knowledge...]")
            try:
                self.workflow_tool = create_workflow_knowledge_tool(index_calls=False)
                print(f"[✅ Indexed {len(self.workflow_tool.retriever.pathway_data)} pathways from Bland/n8n]\n")
            except Exception as e:
                print(f"[❌ Failed to load workflow knowledge: {e}]")
                print("[💡 Tip: Check that ACTi Router path is correct in workflow_knowledge.py]")
                self.workflow_tool = None
        return self.workflow_tool

    def _search_workflows_verbose(self, query: str, context: str = ""):
        """
        Search workflows and display results verbosely.
        Returns the formatted search results as a string.
        """
        tool = self._get_workflow_tool()

        if not tool:
            print("\n⚠️  Workflow knowledge not available - proceeding without it")
            return None

        print("\n" + "🔍" * 35)
        print("🔍 SEARCHING YOUR WORKFLOW DATA")
        print("🔍" * 35)
        print(f"\n📋 Query: {query}")
        print(f"📂 Searching: {len(tool.retriever.pathway_data)} Bland pathways")

        # Perform semantic search
        print("\n⚙️  Performing semantic search...")
        try:
            results = tool.search(query, top_k=5, data_type="pathway")
        except Exception as e:
            print(f"❌ Search failed: {e}")
            return None

        if not results:
            print("❌ No matching workflows found")
            return None

        print(f"✅ Found {len(results)} matching workflows!")
        print("\n" + "=" * 70)
        print("📊 MATCHING WORKFLOWS FROM YOUR DATA:")
        print("=" * 70)

        # Display each result with details
        findings = []
        for i, result in enumerate(results, 1):
            print(f"\n🔹 Match #{i}: {result.source_name}")
            print(f"   📈 Relevance: {result.relevance_score:.0%}")
            print(f"   📁 Type: {result.source_type.upper()}")
            print(f"   📄 Content Preview:")
            print("   " + "-" * 66)

            # Show preview of content
            content_preview = result.content[:400].replace('\n', '\n   ')
            print(f"   {content_preview}")
            if len(result.content) > 400:
                print(f"   ... (truncated, {len(result.content)} chars total)")
            print("   " + "-" * 66)

            findings.append({
                'name': result.source_name,
                'relevance': result.relevance_score,
                'content': result.content
            })

        print("\n" + "=" * 70)
        print(f"✅ FOUND {len(findings)} RELEVANT PATTERNS IN YOUR DATA")
        print("=" * 70)

        # Format findings for builder context
        context_text = f"\n\n## Workflow Knowledge from Bland/n8n Data:\n\n"
        context_text += f"I found {len(findings)} relevant patterns in your existing workflows:\n\n"

        for finding in findings:
            context_text += f"### From: {finding['name']} (Relevance: {finding['relevance']:.0%})\n"
            context_text += f"{finding['content'][:600]}\n\n"

        context_text += "Please use these patterns as reference when designing the agent.\n"

        return context_text

    def run(self):
        """Run the combined CLI."""

        while True:
            try:
                user_input = input("\nYou: ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\nGoodbye!")
                break

            if not user_input:
                continue

            if user_input.lower() in ('quit', 'exit'):
                print("Goodbye!")
                break

            # Route the query
            decision = self.router.route(user_input)

            print(f"\n[🤖 Detected Intent: {decision.intent.value}]")

            # Handle workflow knowledge queries (standalone)
            if decision.intent == Intent.WORKFLOW_KNOWLEDGE and WORKFLOW_AVAILABLE:
                self._handle_workflow_query(user_input)

            # Handle agent creation WITH workflow search first
            elif decision.intent == Intent.CREATE_AGENT:
                self._handle_builder_with_workflow_search(user_input)

            # Handle everything else with builder
            else:
                self._handle_builder_query(user_input)

            print()

    def _handle_workflow_query(self, query: str):
        """Handle pure workflow knowledge query."""
        print("\n💡 [Workflow Knowledge Mode - Query Only]")

        tool = self._get_workflow_tool()

        if not tool:
            print("❌ Workflow knowledge not available")
            return

        # Search and display verbosely
        workflow_context = self._search_workflows_verbose(query)

        if not workflow_context:
            print("\n💬 No specific workflows found, providing general response...")

        print("\n" + "📝" * 35)
        print("📝 FORMATTED ANSWER:")
        print("📝" * 35)

        # Get formatted answer
        try:
            answer = tool.ask(query, max_results=3)
            print("\n" + answer)
        except Exception as e:
            print(f"\n❌ Error generating answer: {e}")

    def _handle_builder_with_workflow_search(self, query: str):
        """Handle agent building WITH workflow search first."""
        print("\n🔨 [Agent Builder Mode with Workflow Search]")

        # First, search workflows for relevant patterns
        print("\n📚 Step 1: Searching your existing workflows for relevant patterns...")

        workflow_context = self._search_workflows_verbose(query)

        # Now build agent with workflow context
        print("\n" + "🔨" * 35)
        print("🔨 Step 2: BUILDING AGENT")
        print("🔨" * 35)

        if workflow_context:
            print("\n✅ Using workflow knowledge as context for agent design...")
            enhanced_query = f"{query}\n\n{workflow_context}"
        else:
            print("\n💬 No specific workflow patterns found, building from general knowledge...")
            enhanced_query = query

        print("\nBuilder: ", end="", flush=True)

        # Add user message to history (with workflow context if available)
        self.messages.append(HumanMessage(content=enhanced_query))

        try:
            result = self.builder.invoke({"messages": self.messages})

            # Extract AI response messages
            ai_messages = [
                msg for msg in result.get("messages", [])
                if hasattr(msg, "type") and msg.type == "ai"
            ]

            if ai_messages:
                response = extract_text_from_content(ai_messages[-1].content)
                print(response)
                # Update messages with full conversation
                self.messages = result.get("messages", self.messages)
            else:
                print("[No response received]")

        except Exception as e:
            print(f"[Error: {e}]")
            import traceback
            traceback.print_exc()

    def _handle_builder_query(self, query: str):
        """Handle standard builder query."""
        print("\n🔨 [Agent Builder Mode]")
        print("\nBuilder: ", end="", flush=True)

        # Add user message to history
        self.messages.append(HumanMessage(content=query))

        try:
            result = self.builder.invoke({"messages": self.messages})

            # Extract AI response messages
            ai_messages = [
                msg for msg in result.get("messages", [])
                if hasattr(msg, "type") and msg.type == "ai"
            ]

            if ai_messages:
                response = extract_text_from_content(ai_messages[-1].content)
                print(response)
                # Update messages with full conversation
                self.messages = result.get("messages", self.messages)
            else:
                print("[No response received]")

        except Exception as e:
            print(f"[Error: {e}]")
            import traceback
            traceback.print_exc()


def main():
    """Main entry point."""
    cli = CombinedCLI()
    cli.run()


if __name__ == "__main__":
    main()