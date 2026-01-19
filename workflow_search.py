"""
Workflow Search Tool - Pure RAG over Bland/n8n Data

Searches through your Bland pathways and n8n workflows using semantic search
and returns relevant information using RAG (Retrieval Augmented Generation).

Usage:
    python workflow_search.py
"""

import os
import sys

# Set the correct path to ACTi Router
ACTI_PATH = r"D:\ACTI\Sigil-neel_dev\Sigil-neel_dev\Autogen-Framework-autogen-bland-json"
sys.path.insert(0, ACTI_PATH)

from src.acti_router.core.retrieval import EnhancedSemanticRetriever


class WorkflowSearchTool:
    """Pure search tool for Bland/n8n workflows."""

    def __init__(self, data_dir: str):
        """Initialize with data directory."""
        self.data_dir = data_dir
        print("=" * 70)
        print("🔍 WORKFLOW SEARCH TOOL - Pure RAG over Your Data")
        print("=" * 70)
        print(f"\n📂 Data Directory: {data_dir}")

        # Initialize retriever
        print("\n⚙️  Building semantic index...")
        self.retriever = EnhancedSemanticRetriever(data_dir=data_dir)
        self.retriever.build_index(include_calls=False)

        print(f"\n✅ Ready!")
        print(f"   📊 Indexed: {len(self.retriever.pathway_data)} Bland pathways")
        print(f"   📊 Total embeddings: {len(self.retriever.embeddings)}")
        print("\n" + "=" * 70)

    def search(self, query: str, top_k: int = 5, verbose: bool = True):
        """
        Search through workflows and display results.

        Args:
            query: Search query
            top_k: Number of results to return
            verbose: Show detailed search process
        """
        if verbose:
            print("\n" + "🔍" * 35)
            print("🔍 SEARCHING YOUR WORKFLOW DATA")
            print("🔍" * 35)
            print(f"\n📋 Query: {query}")
            print(f"📂 Searching through: {len(self.retriever.pathway_data)} pathways")
            print(f"🧠 Using semantic search (embeddings + cosine similarity)")

        # Perform semantic search
        print("\n⚙️  Performing semantic search...")
        results = self.retriever.retrieve(query, top_k=top_k, data_type="pathway")

        if not results:
            print("❌ No matching workflows found")
            return []

        print(f"✅ Found {len(results)} matching workflows!")

        if verbose:
            print("\n" + "=" * 70)
            print("📊 SEARCH RESULTS - RANKED BY RELEVANCE")
            print("=" * 70)

            for i, result in enumerate(results, 1):
                print(f"\n{'=' * 70}")
                print(f"🔹 MATCH #{i}")
                print(f"{'=' * 70}")
                print(f"📁 Source: {result['name']}")
                print(f"📈 Relevance: {result['similarity']:.0%}")
                print(f"🏷️  Type: {result['type'].upper()}")
                print(f"📄 File: {result['filepath']}")

                # Extract the pathway data
                pathway_data = result['data']

                print(f"\n📊 Pathway Info:")
                print(f"   • Pathway ID: {pathway_data.pathway_id}")
                print(f"   • Name: {pathway_data.pathway_name}")
                print(f"   • Total Nodes: {pathway_data.node_count}")

                # Show node details
                print(f"\n🔷 Nodes in this pathway:")
                for j, node in enumerate(pathway_data.nodes[:5], 1):  # Show first 5 nodes
                    print(f"\n   Node {j}: {node.node_name} ({node.node_type})")

                    if node.prompt:
                        print(f"   📝 Prompt Preview:")
                        preview = node.prompt[:300].replace('\n', '\n      ')
                        print(f"      {preview}")
                        if len(node.prompt) > 300:
                            print(f"      ... ({len(node.prompt)} chars total)")

                    if node.kb:
                        print(f"   📚 Knowledge Base Preview:")
                        kb_preview = node.kb[:200].replace('\n', '\n      ')
                        print(f"      {kb_preview}")
                        if len(node.kb) > 200:
                            print(f"      ... ({len(node.kb)} chars total)")

                    if node.global_label:
                        print(f"   🏷️  Label: {node.global_label}")

                if len(pathway_data.nodes) > 5:
                    print(f"\n   ... and {len(pathway_data.nodes) - 5} more nodes")

        return results

    def answer_query(self, query: str, max_results: int = 3):
        """
        Answer a query using RAG - search and synthesize information.

        Args:
            query: User question
            max_results: Maximum number of pathways to use
        """
        # Search for relevant workflows
        results = self.retriever.retrieve(query, top_k=max_results, data_type="pathway")

        if not results:
            return "❌ No relevant information found in your workflows."

        # Build comprehensive answer from results
        print("\n" + "📝" * 35)
        print("📝 ANSWER COMPILED FROM YOUR DATA")
        print("📝" * 35)

        answer = f"\n## Answer: {query}\n\n"
        answer += f"Based on **{len(results)} relevant workflows** from your Bland data:\n\n"

        for i, result in enumerate(results, 1):
            pathway_data = result['data']
            similarity = result['similarity']

            answer += f"### {i}. {pathway_data.pathway_name}\n"
            answer += f"**Relevance: {similarity:.0%}** | Type: {result['type'].upper()}\n\n"

            # Extract key information from nodes
            for node in pathway_data.nodes[:3]:  # Top 3 nodes
                if node.node_name:
                    answer += f"#### Node: {node.node_name}\n"

                if node.prompt:
                    answer += f"**Prompt:**\n```\n{node.prompt[:500]}\n```\n"
                    if len(node.prompt) > 500:
                        answer += f"*... (truncated, {len(node.prompt)} chars total)*\n"
                    answer += "\n"

                if node.kb:
                    answer += f"**Knowledge Base:**\n```\n{node.kb[:300]}\n```\n"
                    if len(node.kb) > 300:
                        answer += f"*... (truncated, {len(node.kb)} chars total)*\n"
                    answer += "\n"

            answer += "---\n\n"

        answer += f"\n**📊 Summary:**\n"
        answer += f"- Searched: {len(self.retriever.pathway_data)} pathways\n"
        answer += f"- Found: {len(results)} relevant matches\n"
        answer += f"- Top relevance: {results[0]['similarity']:.0%}\n"

        print(answer)
        return answer


def main():
    """Main interactive loop."""

    # Data directory
    data_dir = r"D:\ACTI\Sigil-neel_dev\Sigil-neel_dev\Autogen-Framework-autogen-bland-json\data"

    # Check if directory exists
    if not os.path.exists(data_dir):
        print(f"❌ Error: Data directory not found: {data_dir}")
        print("\nPlease update the data_dir path in this script.")
        return

    # Initialize search tool
    try:
        search_tool = WorkflowSearchTool(data_dir)
    except Exception as e:
        print(f"❌ Failed to initialize: {e}")
        import traceback
        traceback.print_exc()
        return

    print("\n💡 Commands:")
    print("   • Type your question to search")
    print("   • 'search <query>' - Detailed search with all results")
    print("   • 'answer <query>' - Get formatted answer")
    print("   • 'stats' - Show statistics")
    print("   • 'quit' - Exit")
    print()

    # Interactive loop
    while True:
        try:
            user_input = input("\n🔍 Query: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\nGoodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break

        # Handle commands
        if user_input.lower() == 'stats':
            print("\n📊 Statistics:")
            print(f"   Pathways indexed: {len(search_tool.retriever.pathway_data)}")
            print(f"   Total nodes: {sum(p.node_count for p in search_tool.retriever.pathway_data)}")
            print(f"   Embeddings: {len(search_tool.retriever.embeddings)}")
            continue

        # Search command
        if user_input.lower().startswith('search '):
            query = user_input[7:].strip()
            search_tool.search(query, top_k=5, verbose=True)

        # Answer command
        elif user_input.lower().startswith('answer '):
            query = user_input[7:].strip()
            search_tool.answer_query(query, max_results=3)

        # Default: treat as answer query
        else:
            search_tool.answer_query(user_input, max_results=3)


if __name__ == "__main__":
    main()