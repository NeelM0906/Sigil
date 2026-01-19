from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

import typer
from rich import print
from rich.panel import Panel

from .core.catalog import Catalog
from .core.retrieval import EmbeddingIndex
from .agents.router_agent import route_with_autogen
from .agents.builder_agent import build_agent_spec
from .agents.pathway_builder_agent import build_bland_pathway_payload
from .core.pathway_schemas import BlandPathwayBuildResult
from .integrations.bland_api import BlandAPIClient

app = typer.Typer(help="ACTi AutoGen workflow router + meta-agent builder")


@app.command("build-index")
def build_index(
    data_dir: str = typer.Option(
        ".",
        help=(
            "Directory (or comma-separated directories) containing workflow JSON files. "
            "Use '.' to auto-detect common folders like 'data', 'bland_dataset', and 'n8n_workflows'."
        ),
    ),
    outputs_dir: str = typer.Option("outputs", help="Directory to store caches"),
    embed_model: str = typer.Option("text-embedding-3-small", help="Embedding model name"),
):
    cat = Catalog.load_local(data_dir)
    idx = EmbeddingIndex(embed_model=embed_model, outputs_dir=outputs_dir)
    idx.build_and_cache(cat.records)
    print(f"[green]OK[/green] Cached {len(cat.records)} workflows to {outputs_dir}/catalog.json and {outputs_dir}/embeddings.npz")


@app.command("route")
def route(
    query: str = typer.Option(..., "--query", "-q", help="User query to route"),
    k: int = typer.Option(8, help="Number of candidates to shortlist"),
    outputs_dir: str = typer.Option("outputs", help="Directory containing cached index"),
    model: str = typer.Option("gpt-4o-mini", help="LLM model to use for rerank"),
):
    idx = EmbeddingIndex(outputs_dir=outputs_dir)
    candidates = idx.shortlist(query, k=k)
    rr = asyncio.run(route_with_autogen(query, candidates, model=model))
    print(Panel(json.dumps(rr.model_dump(), indent=2), title="RouteResult", expand=True))


@app.command("build-agent")
def build_agent(
    query: str = typer.Option(..., "--query", "-q", help="Natural language request for a new agent"),
    k: int = typer.Option(8, help="Number of candidates to shortlist"),
    outputs_dir: str = typer.Option("outputs", help="Directory containing cached index"),
    model: str = typer.Option("gpt-4o-mini", help="LLM model to use"),
):
    """Query → shortlist → RouterAgent → BuilderAgent → save AgentSpec JSON."""
    idx = EmbeddingIndex(outputs_dir=outputs_dir)
    candidates = idx.shortlist(query, k=k)

    rr = asyncio.run(route_with_autogen(query, candidates, model=model))
    selected = next((c for c in candidates if c.workflow_id == rr.workflow_id), candidates[0])

    spec = asyncio.run(build_agent_spec(query, selected.model_dump(), rr.model_dump(), model=model))

    out_dir = Path(outputs_dir) / "agent_specs"
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = f"{int(time.time())}_{spec.agent_name.replace(' ', '_')}.json"
    out_path = out_dir / fname
    out_path.write_text(json.dumps(spec.model_dump(), indent=2), encoding="utf-8")

    print(Panel(json.dumps(spec.model_dump(), indent=2), title="AgentSpec", expand=True))
    print(f"[green]Saved[/green] {out_path}")


@app.command("build-bland-pathway")
def build_bland_pathway(
    query: str = typer.Option(..., "--query", "-q", help="Natural language request for a new Bland pathway"),
    k: int = typer.Option(8, help="Number of candidates to shortlist"),
    outputs_dir: str = typer.Option("outputs", help="Directory containing cached index"),
    model: str = typer.Option("gpt-4o-mini", help="LLM model to use"),
    push_to_bland: bool = typer.Option(
        False,
        "--push",
        help="If set, create + update the pathway in Bland. By default we only generate + save JSON locally.",
    ),
    name: str = typer.Option("", help="Optional: override pathway name"),
    description: str = typer.Option("", help="Optional: override pathway description"),
    extra_instructions: str = typer.Option("", help="Optional: additional instructions for the pathway builder"),
):
    """Query → shortlist → RouterAgent → PathwayBuilderAgent → save payload JSON (optional push to Bland)."""
    idx = EmbeddingIndex(outputs_dir=outputs_dir)
    candidates = idx.shortlist(query, k=k)

    rr = asyncio.run(route_with_autogen(query, candidates, model=model))
    selected = next((c for c in candidates if c.workflow_id == rr.workflow_id), candidates[0])

    # Load the selected template (nodes/edges) from disk
    template_path = Path(selected.file_path)
    if not template_path.exists():
        raise FileNotFoundError(f"Selected template file not found: {template_path}")
    template_raw = json.loads(template_path.read_text(encoding="utf-8"))

    payload = asyncio.run(
        build_bland_pathway_payload(
            user_query=query,
            template_pathway=template_raw,
            selected_workflow=selected.model_dump(),
            router_decision=rr.model_dump(),
            model=model,
            extra_instructions=(extra_instructions or None),
        )
    )

    # Apply CLI overrides
    if name.strip():
        payload.name = name.strip()
    if description.strip():
        payload.description = description.strip()

    # Save payload artifact
    out_dir = Path(outputs_dir) / "bland_created"
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = f"{int(time.time())}_{payload.name.replace(' ', '_')}.payload.json"
    out_path = out_dir / fname
    out_path.write_text(json.dumps(payload.model_dump(), indent=2), encoding="utf-8")

    print(Panel(json.dumps(payload.model_dump(), indent=2), title="BlandPathwayPayload", expand=True))
    print(f"[green]Saved payload[/green] {out_path}")

    if not push_to_bland:
        result = BlandPathwayBuildResult(
            status="saved_locally",
            name=payload.name,
            description=payload.description,
            template_workflow_id=selected.workflow_id,
            template_file_path=str(template_path),
        )
        print(Panel(json.dumps(result.model_dump(), indent=2), title="Result", expand=True))
        return

    # Push to Bland
    client = BlandAPIClient()
    pathway_id = client.create_pathway(payload.name, payload.description)
    api_resp = client.update_pathway(pathway_id, name=payload.name, description=payload.description, nodes=payload.nodes, edges=payload.edges)

    result = BlandPathwayBuildResult(
        status="success",
        pathway_id=pathway_id,
        name=payload.name,
        description=payload.description,
        template_workflow_id=selected.workflow_id,
        template_file_path=str(template_path),
        api_response=api_resp,
    )

    # Save result
    result_path = out_dir / f"{int(time.time())}_{payload.name.replace(' ', '_')}.result.json"
    result_path.write_text(json.dumps(result.model_dump(), indent=2), encoding="utf-8")
    print(Panel(json.dumps(result.model_dump(), indent=2), title="Result", expand=True))
    print(f"[green]Saved result[/green] {result_path}")


@app.command("chat")
def chat(
    outputs_dir: str = typer.Option("outputs"),
    model: str = typer.Option("gpt-4o-mini"),
    k: int = typer.Option(8),
):
    """Interactive chat mode (like `python -m src.builder`)."""
    print("[bold]ACTi Builder Chat[/bold] (type 'exit' to quit)\n")
    while True:
        q = typer.prompt("You")
        if q.strip().lower() in {"exit", "quit"}:
            break

        idx = EmbeddingIndex(outputs_dir=outputs_dir)
        candidates = idx.shortlist(q, k=k)

        rr = asyncio.run(route_with_autogen(q, candidates, model=model))
        selected = next((c for c in candidates if c.workflow_id == rr.workflow_id), candidates[0])

        spec = asyncio.run(build_agent_spec(q, selected.model_dump(), rr.model_dump(), model=model))

        out_dir = Path(outputs_dir) / "agent_specs"
        out_dir.mkdir(parents=True, exist_ok=True)
        fname = f"{int(time.time())}_{spec.agent_name.replace(' ', '_')}.json"
        out_path = out_dir / fname
        out_path.write_text(json.dumps(spec.model_dump(), indent=2), encoding="utf-8")

        print(Panel(json.dumps(spec.model_dump(), indent=2), title=f"Created: {spec.agent_name}", expand=True))
        print(f"[green]Saved[/green] {out_path}\n")


@app.command("build-components")
def build_components(data_dir: str = ".", outputs_dir: str = "outputs"):
    """Build a reusable component/snippet library from all JSON configs."""
    from pathlib import Path
    from .core.component_extractor import build_component_library
    out_path = str(Path(outputs_dir) / "catalog" / "components.json")
    build_component_library(data_dir=data_dir, out_path=out_path)
    print(f"[green]Wrote[/green] {out_path}")


@app.command("chat-build")
def chat_build():
    """Interactive meta-builder: shortlist → route → build AgentSpec."""
    from .builder import main
    main()


if __name__ == "__main__":
    app()
