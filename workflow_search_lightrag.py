"""
Workflow Search with LightRAG - Graph-based RAG over Bland/n8n Data

Minimal version that works with any LightRAG API version.

Usage:
    python workflow_search_lightrag.py
"""

import os
import json
from pathlib import Path

# Minimal LightRAG import - just the core
from lightrag import LightRAG, QueryParam


class LightRAGWorkflowSearch:
    """LightRAG-powered workflow search - minimal version."""

    def __init__(self, data_dir: str, working_dir: str = "./lightrag_storage"):
        """
        Initialize LightRAG search.

        Args:
            data_dir: Directory with Bland/n8n data
            working_dir: Where LightRAG stores its graph/index
        """
        self.data_dir = data_dir
        self.working_dir = working_dir

        print("=" * 70)
        print("🔍 LIGHTRAG WORKFLOW SEARCH")
        print("=" * 70)
        print(f"\n📂 Data: {data_dir}")
        print(f"💾 Storage: {working_dir}")

        # Verify OpenAI API key
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY not set!")

        # Initialize LightRAG with minimal config (uses defaults)
        print("\n⚙️  Initializing LightRAG...")

        self.rag = LightRAG(
            working_dir=working_dir,
            # LightRAG will use default OpenAI settings
        )

        # Index workflows if not already done
        if not self._is_indexed():
            print("📚 Building knowledge graph from workflows...")
            self._index_workflows()
        else:
            print("✅ Using existing knowledge graph")

        print("\n✅ LightRAG ready!")
        print("=" * 70)

    def _is_indexed(self) -> bool:
        """Check if workflows are already indexed."""
        # Check for any files in working directory
        working_path = Path(self.working_dir)
        if not working_path.exists():
            return False
        return len(list(working_path.glob("*"))) > 0

    def _index_workflows(self):
        """Index all workflows into LightRAG knowledge graph."""
        bland_dir = Path(self.data_dir) / "bland_dataset"

        if not bland_dir.exists():
            print(f"❌ Bland dataset not found at: {bland_dir}")
            return

        pathway_files = list(bland_dir.glob("*.json"))
        print(f"Found {len(pathway_files)} pathway files")

        # Index in batches to avoid token limits
        for i, filepath in enumerate(pathway_files, 1):
            print(f"   [{i}/{len(pathway_files)}] Indexing: {filepath.name}")

            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    pathway_data = json.load(f)

                # Build text for this pathway (shortened)
                pathway_text = self._build_pathway_text(pathway_data, filepath.name)

                # Insert into LightRAG
                self.rag.insert(pathway_text)

            except Exception as e:
                print(f"      ⚠️  Error: {e}")

        print("\n✅ Knowledge graph built!")

    def _build_pathway_text(self, pathway_data: dict, filename: str) -> str:
        """Build text representation of pathway (shortened for token limits)."""
        parts = []

        # Pathway metadata
        pathway_name = pathway_data.get("pathway_name", filename.replace(".json", ""))
        pathway_id = pathway_data.get("pathway_id", "unknown")

        parts.append(f"Pathway: {pathway_name}")
        parts.append(f"File: {filename}")
        parts.append("")

        # Extract nodes (limit to first 3 to avoid token limits)
        nodes = pathway_data.get("nodes", [])[:3]

        for node in nodes:
            node_data = node.get("data", {})
            node_name = node_data.get("name", "Unnamed")

            parts.append(f"Node: {node_name}")

            # Prompt (first 500 chars)
            prompt = node_data.get("prompt", "")
            if prompt:
                parts.append(f"Prompt: {prompt[:500]}")

            # KB (first 300 chars)
            kb = node_data.get("kb", "")
            if kb:
                parts.append(f"KB: {kb[:300]}")

            parts.append("")

        return "\n".join(parts)

    def search(self, query: str, mode: str = "hybrid", verbose: bool = True):
        """Search using LightRAG."""
        if verbose:
            print("\n" + "🔍" * 35)
            print("🔍 LIGHTRAG SEARCH")
            print("🔍" * 35)
            print(f"\n📋 Query: {query}")
            print(f"🧠 Mode: {mode.upper()}")

        print("\n⚙️  Searching knowledge graph...")

        try:
            # Query LightRAG
            result = self.rag.query(query, param=QueryParam(mode=mode))

            print("\n" + "=" * 70)
            print("📊 LIGHTRAG RESULTS")
            print("=" * 70)
            print(f"\n{result}")
            print("\n" + "=" * 70)

            return result
        except Exception as e:
            print(f"❌ Search failed: {e}")
            print("\n💡 Try: pip install --upgrade lightrag-hku")
            return None


def main():
    """Main interactive loop."""

    # Configuration
    data_dir = r"D:\ACTI\Sigil-neel_dev\Sigil-neel_dev\Autogen-Framework-autogen-bland-json\data"
    working_dir = "./lightrag_workflow_storage"

    # Check data directory
    if not os.path.exists(data_dir):
        print(f"❌ Data directory not found: {data_dir}")
        return

    # Initialize LightRAG search
    try:
        search = LightRAGWorkflowSearch(data_dir, working_dir)
    except Exception as e:
        print(f"❌ Failed to initialize: {e}")
        import traceback
        traceback.print_exc()
        print("\n💡 Your current system (OpenAI embeddings) works great!")
        print("   LightRAG is optional and experimental.")
        return

    print("\n💡 Commands:")
    print("   • <query> - Search")
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

        # Search
        search.search(user_input, mode="hybrid", verbose=True)


if __name__ == "__main__":
    main()