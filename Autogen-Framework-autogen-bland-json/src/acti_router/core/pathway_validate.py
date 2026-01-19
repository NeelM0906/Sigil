from __future__ import annotations

from typing import Any, Dict, List, Set


class BlandPathwayValidationError(ValueError):
    """Raised when a generated Bland pathway payload fails local validation."""


def _node_id(node: Dict[str, Any]) -> str:
    nid = node.get("id")
    if nid is None:
        return ""
    return str(nid)


def validate_bland_pathway_payload(payload: Dict[str, Any]) -> None:
    """Lightweight, local validation for Bland pathway payloads.

    This does NOT guarantee Bland will accept the payload (their schema can change),
    but it catches the most common issues:
      - missing nodes/edges
      - edges referencing missing node IDs
      - no start node (data.isStart == True)

    Raises:
        BlandPathwayValidationError
    """

    nodes: List[Dict[str, Any]] = list(payload.get("nodes") or [])
    edges: List[Dict[str, Any]] = list(payload.get("edges") or [])

    if not nodes:
        raise BlandPathwayValidationError("Payload has no nodes")
    if not isinstance(nodes, list) or not all(isinstance(n, dict) for n in nodes):
        raise BlandPathwayValidationError("Payload nodes must be a list of objects")
    if not isinstance(edges, list) or not all(isinstance(e, dict) for e in edges):
        raise BlandPathwayValidationError("Payload edges must be a list of objects")

    node_ids: Set[str] = set()
    start_nodes: List[str] = []

    for n in nodes:
        nid = _node_id(n)
        if not nid:
            raise BlandPathwayValidationError("A node is missing a string 'id'")
        if nid in node_ids:
            raise BlandPathwayValidationError(f"Duplicate node id detected: {nid}")
        node_ids.add(nid)

        data = n.get("data")
        if isinstance(data, dict) and data.get("isStart") is True:
            start_nodes.append(nid)

    if not start_nodes:
        raise BlandPathwayValidationError("No start node found. Set data.isStart=true on exactly one node")
    if len(start_nodes) > 1:
        raise BlandPathwayValidationError(
            f"Multiple start nodes found ({len(start_nodes)}): {start_nodes}. Only one node should have data.isStart=true"
        )

    # Validate edges refer to existing nodes
    for e in edges:
        src = e.get("source")
        tgt = e.get("target")
        if src is None or tgt is None:
            raise BlandPathwayValidationError(f"Edge missing 'source' or 'target': {e}")
        src_s = str(src)
        tgt_s = str(tgt)
        if src_s not in node_ids:
            raise BlandPathwayValidationError(f"Edge source '{src_s}' does not exist as a node id")
        if tgt_s not in node_ids:
            raise BlandPathwayValidationError(f"Edge target '{tgt_s}' does not exist as a node id")
