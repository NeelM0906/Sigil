from __future__ import annotations

import glob
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List

from .schemas import WorkflowRecord


def _split_data_dirs(data_dir: str) -> List[str]:
    """Normalize dataset locations.

    Accepts:
      - a single directory: 'data'
      - multiple directories: 'bland_dataset,n8n_workflows'
      - '.' meaning "auto-detect common dataset folders under CWD"
    """
    raw = (data_dir or "").strip()
    if not raw:
        raw = "."

    # Support comma/semicolon-separated lists
    parts = [p.strip() for p in raw.replace(";", ",").split(",") if p.strip()]

    # Auto-detect mode: repo root
    if parts == ["."]:
        cwd = Path.cwd()
        candidates = [
            cwd / "data",
            cwd / "bland_dataset",
            cwd / "n8n_workflows",
            cwd / "datasets",
        ]
        detected = [str(p) for p in candidates if p.exists() and p.is_dir()]
        return detected or [str(cwd)]

    return parts


def _iter_json_files(dirs: Iterable[str]) -> Iterable[Path]:
    """Yield JSON files from one or more directories, recursively."""
    seen: set[str] = set()
    for d in dirs:
        p = Path(d)
        if p.is_file() and p.suffix.lower() == ".json":
            rp = str(p.resolve())
            if rp not in seen:
                seen.add(rp)
                yield p
            continue
        if not p.exists():
            continue
        if p.is_dir():
            for fp in p.rglob("*.json"):
                rp = str(fp.resolve())
                if rp not in seen:
                    seen.add(rp)
                    yield fp

def detect_source(obj: Dict[str, Any]) -> str:
    if isinstance(obj.get("nodes"), list) and "connections" in obj and "id" in obj:
        return "n8n"
    if isinstance(obj.get("nodes"), list) and "edges" in obj:
        return "bland"
    return "unknown"

def _safe_str(x: Any) -> str:
    return str(x) if x is not None else ""

def build_profile(obj: Dict[str, Any], source: str, workflow_id: str, workflow_name: str) -> str:
    parts: List[str] = [
        f"SOURCE: {source}",
        f"WORKFLOW_ID: {workflow_id}",
        f"NAME: {workflow_name}",
    ]

    if source == "n8n":
        nodes = obj.get("nodes", []) or []
        node_names = [n.get("name", "") for n in nodes if n.get("name")]
        node_types = [n.get("type", "") for n in nodes if n.get("type")]

        system_msgs: List[str] = []
        for n in nodes:
            params = n.get("parameters", {}) or {}
            opts = params.get("options") or {}
            sm = opts.get("systemMessage")
            if isinstance(sm, str) and sm.strip():
                system_msgs.append(sm.strip())

        parts.append("NODE_NAMES: " + ", ".join(node_names[:30]))
        parts.append("NODE_TYPES: " + ", ".join(node_types[:30]))
        if system_msgs:
            parts.append("SYSTEM_MESSAGE_SAMPLE: " + system_msgs[0][:1500])

    elif source == "bland":
        nodes = obj.get("nodes", []) or []
        prompts: List[str] = []
        urls: List[str] = []

        global_prompt = ""
        for n in nodes:
            gc = n.get("globalConfig")
            if isinstance(gc, dict):
                gp = gc.get("globalPrompt")
                if isinstance(gp, str) and gp.strip():
                    global_prompt = gp.strip()
                    break

        for n in nodes:
            d = n.get("data", {}) or {}
            p = d.get("prompt")
            if isinstance(p, str) and p.strip():
                prompts.append(p.strip())
            if "url" in d and d.get("url"):
                urls.append(_safe_str(d.get("url")))

        if urls:
            parts.append("WEBHOOK_URLS: " + ", ".join(urls[:3]))
        if global_prompt:
            parts.append("GLOBAL_PROMPT: " + global_prompt[:1200])
        if prompts:
            parts.append("NODE_PROMPTS_SAMPLE: " + " | ".join(p[:250] for p in prompts[:8]))

    return "\n".join(parts)

class Catalog:
    def __init__(self, records: List[WorkflowRecord]):
        self.records = records

    @classmethod
    def load_local(cls, data_dir: str = "data") -> "Catalog":
        """Load workflow configs from one or more folders.

        This is intentionally flexible because many repos store JSONs under
        nested folders (e.g. `bland_dataset/**.json` and `n8n_workflows/**.json`).

        `data_dir` may be:
          - a single folder
          - a comma/semicolon-separated list of folders
          - '.' (repo root) — we auto-detect common dataset folders if present
        """
        records: List[WorkflowRecord] = []

        data_dirs = _split_data_dirs(data_dir)
        json_paths: List[Path] = []
        for d in data_dirs:
            p = Path(d)
            if p.is_file() and p.suffix.lower() == ".json":
                json_paths.append(p)
                continue
            if p.is_dir():
                json_paths.extend([x for x in p.rglob("*.json") if x.is_file()])

        # Backwards compatible: if nothing found and the given dir doesn't exist,
        # fall back to the original shallow glob so existing scripts don't break.
        if not json_paths:
            for path in glob.glob(os.path.join(data_dir, "*.json")):
                json_paths.append(Path(path))

        for path in sorted(set(json_paths)):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    obj = json.load(f)
            except Exception:
                # Skip non-JSON / malformed files instead of crashing the whole catalog.
                continue

            source = detect_source(obj)

            if source == "n8n":
                workflow_id = _safe_str(obj.get("id", path.name))
                workflow_name = _safe_str(obj.get("name", path.name))
            else:
                workflow_id = path.name
                workflow_name = path.name

            profile = build_profile(obj, source, workflow_id, workflow_name)

            records.append(
                WorkflowRecord(
                    source=source,
                    workflow_id=workflow_id,
                    workflow_name=workflow_name,
                    profile=profile,
                    file_path=str(path),
                )
            )

        return cls(records)
