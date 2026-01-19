from __future__ import annotations
import asyncio, json, time
from pathlib import Path
from rich import print
from rich.panel import Panel

from .core.retrieval import EmbeddingIndex
from .agents.router_agent import route_with_autogen
from .agents.builder_agent import build_agent_spec

def main():
    print("[bold]ACTi Meta-Builder (AutoGen) — interactive mode[/bold]")
    print("This does: shortlist → route → build AgentSpec → save JSON\n")
    idx = EmbeddingIndex(outputs_dir="outputs")

    while True:
        q = input("You> ").strip()
        if q.lower() in {"exit","quit"}:
            break

        candidates = idx.shortlist(q, k=8)
        rr = asyncio.run(route_with_autogen(q, candidates, model="gpt-4o-mini"))

        selected = next((c for c in candidates if c.workflow_id == rr.workflow_id), candidates[0])
        selected_payload = {
            "source": selected.source,
            "workflow_id": selected.workflow_id,
            "workflow_name": selected.workflow_name,
            "file_path": selected.file_path,
            "profile_preview": selected.profile_preview,
            "similarity": selected.similarity,
        }
        shortlist_payload = [
            {
                "source": c.source,
                "workflow_id": c.workflow_id,
                "workflow_name": c.workflow_name,
                "profile_preview": c.profile_preview,
                "similarity": c.similarity,
                "file_path": c.file_path,
            }
            for c in candidates
        ]

        spec = asyncio.run(build_agent_spec(q, selected_payload, rr.model_dump(), shortlisted_workflows=shortlist_payload))
        out_dir = Path("outputs") / "agent_specs"
        out_dir.mkdir(parents=True, exist_ok=True)
        fname = f"{int(time.time())}_{spec.agent_name.replace(' ', '_')}.json"
        path = out_dir / fname
        path.write_text(json.dumps(spec.model_dump(), indent=2), encoding="utf-8")

        print(Panel(json.dumps(spec.model_dump(), indent=2), title=f"Saved → {path}", expand=True))

if __name__ == "__main__":
    main()
