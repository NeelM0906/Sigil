"""
Enhanced Semantic Retriever for ACTi Router.

This module provides semantic search over Bland AI pathways and n8n workflows
using OpenAI embeddings and cosine similarity.
"""

import os
import json
import glob
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

try:
    from openai import OpenAI
except ImportError:
    import openai
    OpenAI = None


@dataclass
class NodeData:
    """Complete node data structure."""
    node_id: str
    node_name: str
    node_type: str
    prompt: str = ""
    kb: str = ""
    temperature: float = 0.7
    position: Dict[str, float] = None
    global_label: str = ""
    
    def __post_init__(self):
        if self.position is None:
            self.position = {}


@dataclass
class PathwayData:
    """Complete pathway data structure."""
    pathway_id: str
    pathway_name: str
    nodes: List[NodeData]
    edges: List[Dict[str, Any]]
    node_count: int = 0
    
    def __post_init__(self):
        self.node_count = len(self.nodes)


@dataclass
class CallData:
    """Call data structure."""
    call_id: str
    pathway_id: str
    transcript: str
    summary: str
    duration: float
    status: str


class CompleteDataExtractor:
    """Extracts complete data from Bland pathway and n8n workflow files."""
    
    def extract_pathway_from_file(self, filepath: str) -> PathwayData:
        """Extract complete pathway data from JSON file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
        
        # Extract pathway metadata
        pathway_id = raw_data.get("pathway_id", os.path.basename(filepath))
        pathway_name = raw_data.get("pathway_name", os.path.basename(filepath).replace(".json", ""))
        
        # Extract nodes
        nodes = []
        for raw_node in raw_data.get("nodes", []):
            node_data_dict = raw_node.get("data", {})
            
            node = NodeData(
                node_id=raw_node.get("id", ""),
                node_name=node_data_dict.get("name", ""),
                node_type=raw_node.get("type", ""),
                prompt=node_data_dict.get("prompt", ""),
                kb=node_data_dict.get("kb", ""),
                temperature=node_data_dict.get("modelOptions", {}).get("temperature", 0.7),
                position=raw_node.get("position", {}),
                global_label=node_data_dict.get("globalLabel", "")
            )
            nodes.append(node)
        
        # Extract edges
        edges = raw_data.get("edges", [])
        
        return PathwayData(
            pathway_id=pathway_id,
            pathway_name=pathway_name,
            nodes=nodes,
            edges=edges
        )
    
    def extract_call_from_file(self, filepath: str) -> CallData:
        """Extract call data from JSON file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
        
        # Extract transcript
        transcript_parts = []
        for t in raw_data.get("transcripts", []):
            speaker = t.get("user", "AI")
            text = t.get("text", "")
            transcript_parts.append(f"{speaker}: {text}")
        
        return CallData(
            call_id=raw_data.get("call_id", ""),
            pathway_id=raw_data.get("pathway_id", ""),
            transcript="\n".join(transcript_parts),
            summary=raw_data.get("summary", ""),
            duration=raw_data.get("call_length", 0),
            status=raw_data.get("status", "")
        )


class SearchableTextBuilder:
    """Builds searchable text from extracted data."""
    
    def build_pathway_text(self, pathway: PathwayData) -> str:
        """Build comprehensive searchable text for pathway."""
        parts = [f"Pathway: {pathway.pathway_name}"]
        
        for node in pathway.nodes:
            node_text = f"Node: {node.node_name} ({node.node_type})"
            if node.prompt:
                node_text += f"\nPrompt: {node.prompt}"
            if node.kb:
                node_text += f"\nKnowledge Base: {node.kb}"
            if node.global_label:
                node_text += f"\nGlobal: {node.global_label}"
            parts.append(node_text)
        
        return "\n\n".join(parts)
    
    def build_call_text(self, call: CallData) -> str:
        """Build searchable text for call."""
        parts = [
            f"Call ID: {call.call_id}",
            f"Status: {call.status}",
            f"Duration: {call.duration}s"
        ]
        
        if call.summary:
            parts.append(f"Summary: {call.summary}")
        
        if call.transcript:
            parts.append(f"Transcript:\n{call.transcript}")
        
        return "\n\n".join(parts)


class EnhancedSemanticRetriever:
    """
    Enhanced semantic retriever with complete data extraction.
    
    Indexes Bland pathways and call transcripts using OpenAI embeddings
    for semantic search.
    """
    
    def __init__(self, data_dir: str = None):
        """Initialize retriever with data directory."""
        if data_dir is None:
            data_dir = os.path.join(os.path.dirname(__file__), "../../../data")
        
        self.data_dir = data_dir
        self.pathways_dir = os.path.join(data_dir, "bland_dataset")
        self.calls_dir = os.path.join(data_dir, "bland_calls")
        
        # Initialize OpenAI client
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        
        if OpenAI:
            self.client = OpenAI(api_key=api_key)
        else:
            openai.api_key = api_key
            self.client = None
        
        # Data storage
        self.pathway_data: List[PathwayData] = []
        self.call_data: List[CallData] = []
        self.embeddings: List[List[float]] = []
        self.metadata: List[Dict[str, Any]] = []
        
        # Extractors
        self.extractor = CompleteDataExtractor()
        self.text_builder = SearchableTextBuilder()
    
    def build_index(self, include_calls: bool = False):
        """Build semantic index from data files."""
        print(f"Building semantic index from: {self.data_dir}")
        
        # Index pathways
        self._index_pathways()
        
        # Index calls if requested
        if include_calls:
            self._index_calls()
        
        print(f"Index built: {len(self.pathway_data)} pathways, {len(self.call_data)} calls")
    
    def _index_pathways(self):
        """Index all pathway files."""
        if not os.path.exists(self.pathways_dir):
            print(f"Warning: Pathways directory not found: {self.pathways_dir}")
            return
        
        pathway_files = glob.glob(os.path.join(self.pathways_dir, "*.json"))
        print(f"Found {len(pathway_files)} pathway files")
        
        for filepath in pathway_files:
            try:
                # Extract pathway data
                pathway = self.extractor.extract_pathway_from_file(filepath)
                self.pathway_data.append(pathway)
                
                # Build searchable text
                searchable_text = self.text_builder.build_pathway_text(pathway)
                
                # Generate embedding
                embedding = self._get_embedding(searchable_text)
                self.embeddings.append(embedding)
                
                # Store metadata
                self.metadata.append({
                    "type": "pathway",
                    "id": pathway.pathway_id,
                    "name": pathway.pathway_name,
                    "filepath": filepath,
                    "data": pathway
                })
                
            except Exception as e:
                print(f"Error indexing {filepath}: {e}")
    
    def _index_calls(self):
        """Index all call files."""
        if not os.path.exists(self.calls_dir):
            print(f"Warning: Calls directory not found: {self.calls_dir}")
            return
        
        call_files = glob.glob(os.path.join(self.calls_dir, "*.json"))
        print(f"Found {len(call_files)} call files")
        
        for filepath in call_files:
            try:
                # Extract call data
                call = self.extractor.extract_call_from_file(filepath)
                self.call_data.append(call)
                
                # Build searchable text
                searchable_text = self.text_builder.build_call_text(call)
                
                # Generate embedding
                embedding = self._get_embedding(searchable_text)
                self.embeddings.append(embedding)
                
                # Store metadata
                self.metadata.append({
                    "type": "call",
                    "id": call.call_id,
                    "name": f"Call {call.call_id}",
                    "filepath": filepath,
                    "data": call
                })
                
            except Exception as e:
                print(f"Error indexing {filepath}: {e}")

    def _get_embedding(self, text: str) -> List[float]:
        """Generate embedding for text using OpenAI."""
        # Truncate text to fit model limits (8192 tokens ≈ 32,768 chars)
        # Use conservative estimate: 6000 tokens ≈ 24,000 chars
        MAX_CHARS = 24000
        if len(text) > MAX_CHARS:
            text = text[:MAX_CHARS]
            print(f"Warning: Truncated text from {len(text)} to {MAX_CHARS} chars")

        try:
            if self.client:
                # New OpenAI client
                response = self.client.embeddings.create(
                    model="text-embedding-3-small",
                    input=text
                )
                return response.data[0].embedding
            else:
                # Legacy OpenAI API
                response = openai.Embedding.create(
                    model="text-embedding-3-small",
                    input=text
                )
                return response['data'][0]['embedding']
        except Exception as e:
            print(f"Error generating embedding: {e}")
            # Return zero vector as fallback
            return [0.0] * 1536

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        data_type: str = "all"  # "all", "pathway", "call"
    ) -> List[Dict[str, Any]]:
        """
        Retrieve top-k most relevant items for query.
        
        Args:
            query: Search query
            top_k: Number of results to return
            data_type: Filter by type ("all", "pathway", "call")
        
        Returns:
            List of results with similarity scores
        """
        if not self.embeddings:
            return []
        
        # Generate query embedding
        query_embedding = self._get_embedding(query)
        
        # Calculate similarities
        similarities = []
        for idx, emb in enumerate(self.embeddings):
            # Filter by type if specified
            if data_type != "all" and self.metadata[idx]["type"] != data_type:
                continue
            
            sim = self._cosine_similarity(query_embedding, emb)
            similarities.append((idx, sim))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Get top-k
        top_results = similarities[:top_k]
        
        # Format results
        results = []
        for idx, sim in top_results:
            result = {
                "similarity": sim,
                "type": self.metadata[idx]["type"],
                "id": self.metadata[idx]["id"],
                "name": self.metadata[idx]["name"],
                "filepath": self.metadata[idx]["filepath"],
                "data": self.metadata[idx]["data"]
            }
            results.append(result)
        
        return results
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        import math
        
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude1 = math.sqrt(sum(a * a for a in vec1))
        magnitude2 = math.sqrt(sum(b * b for b in vec2))
        
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0
        
        return dot_product / (magnitude1 * magnitude2)


# Test function
if __name__ == "__main__":
    print("Testing Enhanced Semantic Retriever...")
    
    retriever = EnhancedSemanticRetriever()
    retriever.build_index(include_calls=False)
    
    print("\nTesting search...")
    results = retriever.retrieve("handle objections", top_k=3)
    
    for r in results:
        print(f"- {r['name']} ({r['similarity']:.2%})")
    
    print("\nDone!")
