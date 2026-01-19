from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from .catalog import _iter_json_files, _split_data_dirs

def _safe_load_json(path: str) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return None

def extract_components_from_bland(doc: Dict[str, Any]) -> Dict[str, Any]:
    # Bland agent JSONs vary; we keep a pragmatic, schema-light extraction.
    out: Dict[str, Any] = {"source": "bland"}
    for key in ["name", "agent_name", "voice", "language", "webhook_url", "webhook", "prompt", "system_prompt"]:
        if key in doc:
            out[key] = doc.get(key)
    # tool-ish fields
    for key in ["tools", "actions", "integrations", "functions"]:
        if key in doc:
            out[key] = doc.get(key)
    return out

def extract_components_from_n8n(doc: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {"source": "n8n"}
    out["name"] = doc.get("name")
    out["id"] = doc.get("id")
    # nodes contain most reusable patterns
    nodes = doc.get("nodes") or []
    conns = doc.get("connections") or {}
    out["nodes_summary"] = [{"name": n.get("name"), "type": n.get("type")} for n in nodes if isinstance(n, dict)]
    # heuristics: keep webhook/http/ai/llm/call/sms/calendar nodes as reusable
    reusable = []
    for n in nodes:
        if not isinstance(n, dict):
            continue
        t = (n.get("type") or "").lower()
        if any(k in t for k in ["webhook", "http", "openai", "llm", "twilio", "calendar", "gmail", "slack", "hubspot"]):
            reusable.append({"name": n.get("name"), "type": n.get("type"), "parameters": n.get("parameters")})
    out["reusable_nodes"] = reusable
    out["connections"] = conns
    return out

def extract_components(file_path: str, source_hint: Optional[str] = None) -> Dict[str, Any]:
    doc = _safe_load_json(file_path) or {}
    src = (source_hint or doc.get("source") or "").lower()
    # crude detection
    if src == "n8n" or "nodes" in doc and "connections" in doc:
        return extract_components_from_n8n(doc)
    return extract_components_from_bland(doc)

def build_component_library(data_dir: str, out_path: str) -> Dict[str, Any]:
    """Build a lightweight 'component library' from workflow JSONs.

    `data_dir` can be:
      - a directory
      - a comma/semicolon-separated list of directories
      - '.' (repo root) to auto-detect common dataset folders
    """
    items: List[Dict[str, Any]] = []
    for fp in sorted(_iter_json_files(_split_data_dirs(data_dir))):
        comp = extract_components(str(fp))
        comp["file_path"] = str(fp)
        items.append(comp)
    library = {"count": len(items), "items": items}
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(library, indent=2), encoding="utf-8")
    return library
