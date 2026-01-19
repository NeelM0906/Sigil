from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List

def _safe_text(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    return json.dumps(x, ensure_ascii=False)

def _extract_global_prompt(nodes: List[Dict[str, Any]]) -> str:
    for n in nodes:
        gc = (n or {}).get("globalConfig") or {}
        gp = gc.get("globalPrompt")
        if isinstance(gp, str) and gp.strip():
            return gp.strip()
    return ""

def _summarize_extract_vars(ev: Any) -> List[Dict[str, Any]]:
    out = []
    if not isinstance(ev, list):
        return out
    for row in ev:
        # expected: [name, type, instruction, required_bool]
        if isinstance(row, list) and len(row) >= 4:
            out.append(
                {
                    "var": str(row[0]),
                    "type": str(row[1]),
                    "instruction": str(row[2]),
                    "required": bool(row[3]),
                }
            )
    return out

def _node_archetype(n: Dict[str, Any]) -> Dict[str, Any]:
    t = n.get("type")
    data = n.get("data") or {}
    prompt = data.get("prompt") or data.get("message") or ""
    arche = {
        "type": t,
        "name": data.get("name") or n.get("id"),
        "has_prompt": bool(prompt),
        "prompt_len": len(_safe_text(prompt)),
        "uses_tags": bool(re.search(r"<[^>]+>", _safe_text(prompt))),
        "extract_vars": _summarize_extract_vars(data.get("extractVars")),
        "has_condition_text": bool(data.get("condition")),
        "model_options_keys": sorted(list((data.get("modelOptions") or {}).keys())),
    }

    if t == "Webhook":
        arche["webhook"] = {
            "method": data.get("method"),
            "has_body": bool(data.get("body")),
            "headers_shape": "list_pairs" if isinstance(data.get("headers"), list) else type(data.get("headers")).__name__,
            "has_responseData": bool(data.get("responseData")),
            "has_responsePathways": bool(data.get("responsePathways")),
            "isGlobal": bool(data.get("isGlobal")),
            "globalLabel": data.get("globalLabel"),
        }
    return arche

def _edge_archetype(e: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "type": e.get("type"),
        "label": ((e.get("data") or {}).get("label")),
        "source": e.get("source"),
        "target": e.get("target"),
    }

def compile_profiles(
    dataset_dir: Path,
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(dataset_dir.glob("*.json"))
    if not files:
        raise SystemExit(f"No .json files found in: {dataset_dir}")

    for fp in files:
        raw = json.loads(fp.read_text(encoding="utf-8"))
        nodes = raw.get("nodes") or []
        edges = raw.get("edges") or []

        profile = {
            "source_file": str(fp),
            "global_prompt": _extract_global_prompt(nodes),
            "node_archetypes": [_node_archetype(n) for n in nodes if isinstance(n, dict)],
            "edge_archetypes": [_edge_archetype(e) for e in edges if isinstance(e, dict)],
            # quick “style hints” for the model
            "style_hints": {
                "default_node_prompt_style": "long_scripted" if any((n.get("data") or {}).get("prompt") for n in nodes) else "short",
                "edges_use_custom_type": any((e.get("type") == "custom") for e in edges),
                "extract_vars_format": "list_of_4" if any(isinstance((n.get("data") or {}).get("extractVars"), list) for n in nodes) else "unknown",
            },
        }

        out_name = fp.stem.replace(" ", "_").replace("(", "").replace(")", "").replace(",", "")
        out_path = out_dir / f"{out_name}.profile.json"
        out_path.write_text(json.dumps(profile, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"✅ Compiled {len(files)} profiles -> {out_dir}")

if __name__ == "__main__":
    # change these if you want
    dataset = Path(r"data\bland_dataset")
    out = Path(r"outputs\bland_profiles")
    compile_profiles(dataset, out)
