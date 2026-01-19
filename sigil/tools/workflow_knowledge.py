"""
Workflow Knowledge Tool - Complete Integration
Semantic Q&A over Bland/n8n data integrated with Sigil.

Installation:
1. Copy this file to: Sigil-neel_dev/sigil/tools/workflow_knowledge.py
2. Ensure ACTi Router is in the correct relative path
3. Install dependencies: openai, numpy
"""

import sys
import os
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

# Add ACTi Router to Python path
ACTI_PATH = r"D:\ACTI\Sigil-neel_dev\Sigil-neel_dev\Autogen-Framework-autogen-bland-json"
sys.path.insert(0, ACTI_PATH)
# Import ACTi components
try:
    from src.acti_router.core.retrieval import EnhancedSemanticRetriever, PathwayData, CallData
except ImportError as e:
    print(f"Warning: Could not import ACTi Router components: {e}")
    print(f"Make sure ACTi Router is at: {ACTI_PATH}")
    EnhancedSemanticRetriever = None
    PathwayData = None
    CallData = None


@dataclass
class KnowledgeResult:
    """Result from knowledge search."""
    source_type: str  # "pathway" or "call"
    source_name: str
    relevance_score: float
    content: str
    metadata: Dict[str, Any]


class WorkflowKnowledgeTool:
    """
    Semantic Q&A tool over Bland workflow and call data.
    
    This connects ACTi Router's semantic index to Sigil,
    allowing Sigil agents to search through workflow patterns
    and call transcripts.
    
    Example usage:
        tool = WorkflowKnowledgeTool()
        answer = tool.ask("How to handle objections?")
        print(answer)
    """
    
    def __init__(self, data_dir: str = None, index_calls: bool = True):
        """
        Initialize workflow knowledge tool.
        
        Args:
            data_dir: Path to data directory (default: ACTi data/)
            index_calls: Whether to index call transcripts
        """
        if EnhancedSemanticRetriever is None:
            raise ImportError(
                "Could not import ACTi Router. Please ensure it's installed at: "
                f"{ACTI_PATH}"
            )
        
        if data_dir is None:
            data_dir = os.path.join(ACTI_PATH, "data")
        
        print(f"🔍 Initializing Workflow Knowledge Tool...")
        print(f"   Data directory: {data_dir}")
        
        # Create enhanced retriever
        self.retriever = EnhancedSemanticRetriever(data_dir=data_dir)
        
        # Build index
        print(f"   Building semantic index...")
        self.retriever.build_index(include_calls=index_calls)
        
        print(f"✅ Workflow Knowledge Tool ready!")
        print(f"   Indexed {len(self.retriever.pathway_data)} pathways")
        print(f"   Indexed {len(self.retriever.call_data)} calls")
    
    def search(
        self,
        query: str,
        top_k: int = 5,
        data_type: str = "all"  # "all", "pathway", "call"
    ) -> List[KnowledgeResult]:
        """
        Search for relevant knowledge.
        
        Args:
            query: Natural language search query
            top_k: Number of results to return
            data_type: Type of data to search ("all", "pathway", "call")
        
        Returns:
            List of knowledge results ranked by relevance
            
        Example:
            results = tool.search("handle objections", top_k=3)
            for result in results:
                print(f"{result.source_name}: {result.relevance_score:.0%}")
        """
        # Retrieve from semantic index
        raw_results = self.retriever.retrieve(
            query=query,
            top_k=top_k,
            data_type=data_type
        )
        
        # Convert to knowledge results
        results = []
        for raw in raw_results:
            result = self._convert_to_knowledge_result(raw, query)
            results.append(result)
        
        return results
    
    def ask(
        self,
        question: str,
        max_results: int = 3,
        include_calls: bool = False
    ) -> str:
        """
        Ask a question and get formatted answer.
        
        Args:
            question: Natural language question
            max_results: Maximum results to use for answer
            include_calls: Whether to include call transcripts
        
        Returns:
            Formatted answer with sources
            
        Example:
            answer = tool.ask("How do I handle price objections?")
            print(answer)
        """
        # Determine data type
        data_type = "all" if include_calls else "pathway"
        
        # Search
        results = self.search(
            query=question,
            top_k=max_results,
            data_type=data_type
        )
        
        if not results:
            return "I couldn't find relevant information in the workflow data."
        
        # Format answer
        answer = self._format_answer(question, results)
        return answer
    
    def get_examples(
        self,
        topic: str,
        top_k: int = 3,
        from_calls: bool = False
    ) -> str:
        """
        Get concrete examples from workflows or calls.
        
        Args:
            topic: What to get examples of
            top_k: Number of examples to return
            from_calls: Whether to get examples from call transcripts
        
        Returns:
            Formatted examples with sources
            
        Example:
            examples = tool.get_examples("email collection", top_k=3)
            print(examples)
        """
        data_type = "call" if from_calls else "pathway"
        
        results = self.search(
            query=topic,
            top_k=top_k,
            data_type=data_type
        )
        
        if not results:
            return f"No examples found for: {topic}"
        
        # Format examples
        output = f"## Examples of: {topic}\n\n"
        
        for i, result in enumerate(results, 1):
            output += f"### Example {i}: {result.source_name}\n"
            output += f"**Source:** {result.source_type.upper()}\n"
            output += f"**Relevance:** {result.relevance_score:.0%}\n\n"
            output += f"{result.content}\n\n"
            output += "---\n\n"
        
        return output
    
    def compare_approaches(
        self,
        topic: str,
        num_approaches: int = 3
    ) -> str:
        """
        Compare how different workflows handle a topic.
        
        Args:
            topic: What to compare (e.g., "lead qualification")
            num_approaches: Number of approaches to compare
        
        Returns:
            Formatted comparison
            
        Example:
            comparison = tool.compare_approaches("lead qualification")
            print(comparison)
        """
        results = self.search(
            query=topic,
            top_k=num_approaches,
            data_type="pathway"
        )
        
        if len(results) < 2:
            return f"Not enough examples found to compare: {topic}"
        
        # Format comparison
        output = f"## Comparison: {topic}\n\n"
        
        for i, result in enumerate(results, 1):
            output += f"### Approach {i}: {result.source_name}\n\n"
            output += f"**Relevance:** {result.relevance_score:.0%}\n\n"
            output += f"{result.content}\n\n"
            output += "---\n\n"
        
        return output
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about indexed data.
        
        Returns:
            Dictionary with statistics
        """
        return {
            "total_pathways": len(self.retriever.pathway_data),
            "total_calls": len(self.retriever.call_data),
            "total_embeddings": len(self.retriever.embeddings),
            "pathway_names": [p.pathway_name for p in self.retriever.pathway_data],
            "avg_nodes_per_pathway": sum(
                p.node_count for p in self.retriever.pathway_data
            ) / len(self.retriever.pathway_data) if self.retriever.pathway_data else 0
        }
    
    # Private methods
    
    def _convert_to_knowledge_result(
        self,
        raw_result: Dict[str, Any],
        query: str
    ) -> KnowledgeResult:
        """Convert raw search result to knowledge result."""
        
        source_type = raw_result["type"]
        data = raw_result["data"]
        
        # Extract content based on type
        if source_type == "pathway":
            content = self._extract_pathway_content(data, query)
        else:  # call
            content = self._extract_call_content(data, query)
        
        return KnowledgeResult(
            source_type=source_type,
            source_name=raw_result["name"],
            relevance_score=raw_result["similarity"],
            content=content,
            metadata={
                "id": raw_result["id"],
                "filepath": raw_result["filepath"]
            }
        )
    
    def _extract_pathway_content(
        self,
        pathway: PathwayData,
        query: str
    ) -> str:
        """Extract relevant content from pathway."""
        query_lower = query.lower()
        relevant_parts = []
        
        # Find nodes relevant to query
        for node in pathway.nodes:
            if self._is_relevant(node.prompt, query_lower):
                # Add node content
                node_text = f"**{node.node_name}** ({node.node_type})"
                
                if node.prompt:
                    prompt_preview = node.prompt[:400]
                    if len(node.prompt) > 400:
                        prompt_preview += "..."
                    node_text += f"\n{prompt_preview}"
                
                relevant_parts.append(node_text)
        
        # If no specific matches, return first few nodes
        if not relevant_parts:
            for node in pathway.nodes[:2]:
                if node.prompt:
                    relevant_parts.append(f"**{node.node_name}**: {node.prompt[:200]}...")
        
        return "\n\n".join(relevant_parts[:3])  # Top 3
    
    def _extract_call_content(
        self,
        call: CallData,
        query: str
    ) -> str:
        """Extract relevant content from call."""
        parts = []
        
        # Add call summary
        if call.summary:
            parts.append(f"**Summary:** {call.summary}")
        
        # Add relevant transcript excerpts
        if call.transcript:
            # Find relevant parts of transcript
            lines = call.transcript.split('\n')
            relevant_lines = [
                line for line in lines
                if self._is_relevant(line.lower(), query.lower())
            ]
            
            if relevant_lines:
                parts.append(f"**Transcript Excerpt:**")
                parts.append("\n".join(relevant_lines[:5]))  # Top 5 lines
        
        return "\n\n".join(parts)
    
    def _is_relevant(self, text: str, query: str) -> bool:
        """Check if text is relevant to query."""
        if not text:
            return False
        
        query_words = query.split()
        return any(word in text for word in query_words if len(word) > 3)
    
    def _format_answer(
        self,
        question: str,
        results: List[KnowledgeResult]
    ) -> str:
        """Format results into answer."""
        answer = f"## Answer: {question}\n\n"
        answer += "Based on the workflow data:\n\n"
        
        for i, result in enumerate(results, 1):
            answer += f"### {i}. From '{result.source_name}' ({result.source_type})\n"
            answer += f"*Relevance: {result.relevance_score:.0%}*\n\n"
            answer += f"{result.content}\n\n"
        
        answer += "---\n"
        answer += f"**Sources:** {len(results)} {results[0].source_type}(s)\n"
        
        return answer


# Convenience function for Sigil integration
def create_workflow_knowledge_tool(
    data_dir: str = None,
    index_calls: bool = True
) -> WorkflowKnowledgeTool:
    """
    Create workflow knowledge tool for use in Sigil.
    
    Args:
        data_dir: Path to data directory (default: auto-detect)
        index_calls: Whether to index call transcripts
    
    Returns:
        Initialized WorkflowKnowledgeTool
        
    Usage in Sigil:
        from sigil.tools.workflow_knowledge import create_workflow_knowledge_tool
        
        tool = create_workflow_knowledge_tool()
        answer = tool.ask("How to handle objections?")
        print(answer)
    """
    return WorkflowKnowledgeTool(
        data_dir=data_dir,
        index_calls=index_calls
    )


# Standalone test
if __name__ == "__main__":
    print("Testing Workflow Knowledge Tool...")
    
    try:
        tool = create_workflow_knowledge_tool(index_calls=False)
        
        # Test search
        print("\n" + "="*60)
        print("TEST: Search")
        print("="*60)
        results = tool.search("handle objections", top_k=2)
        for r in results:
            print(f"- {r.source_name} ({r.relevance_score:.0%})")
        
        # Test ask
        print("\n" + "="*60)
        print("TEST: Ask Question")
        print("="*60)
        answer = tool.ask("What's a good cold call opening?", max_results=1)
        print(answer[:500] + "...")
        
        print("\n✅ All tests passed!")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nMake sure:")
        print("1. ACTi Router is in the correct location")
        print("2. Data files exist in data/bland_dataset/")
        print("3. OpenAI API key is set (OPENAI_API_KEY)")
