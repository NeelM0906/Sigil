from __future__ import annotations

import json
import re
from typing import Any, Dict, Optional

from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient

from ..core.pathway_schemas import BlandPathwayPayload


PATHWAY_BUILDER_SYSTEM = """You are Sigil Pathway Builder.

You will be given:
- USER_QUERY
- TEMPLATE_PATHWAY: a JSON object with fields {nodes:[...], edges:[...]} (a working Bland pathway template)
- SELECTED_WORKFLOW metadata (source/workflow_id/workflow_name/profile_preview/file_path)
- ROUTER_DECISION (router reasoning, matched signals, confidence)

Your job: output STRICT JSON (no markdown, no code fences) that matches this schema exactly:

{
  "name": "<string>",
  "description": "<string>",
  "nodes": [...],
  "edges": [...]
}

Rules:
- Start from TEMPLATE_PATHWAY and MODIFY it. Do NOT invent a completely new structure unless necessary.
- Keep the node/edge format consistent with the template (ids, data fields, types, positions, etc.).
- Ensure the resulting pathway is coherent for USER_QUERY.
- If you add new nodes, make sure they are connected via edges so there are no dead ends.
- Keep output valid JSON only.
"""


def _extract_json(text: str) -> str:
    text = re.sub(r"^```(json)?\s*", "", text.strip(), flags=re.IGNORECASE)
    text = re.sub(r"```\s*$", "", text.strip())
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    return m.group(0) if m else text


def _normalize_payload(raw: Dict[str, Any]) -> Dict[str, Any]:
    data = dict(raw)

    # Some models wrap output
    if "payload" in data and isinstance(data["payload"], dict):
        data = dict(data["payload"])
    if "BlandPathway" in data and isinstance(data["BlandPathway"], dict):
        data = dict(data["BlandPathway"])

    # Alternate keys
    if "title" in data and "name" not in data:
        data["name"] = data.pop("title")
    if "desc" in data and "description" not in data:
        data["description"] = data.pop("desc")

    # Required fallbacks
    data.setdefault("name", "Generated Pathway")
    data.setdefault("description", "")
    data.setdefault("nodes", [])
    data.setdefault("edges", [])

    return data


async def build_bland_pathway_payload(
    user_query: str,
    template_pathway: Dict[str, Any],
    selected_workflow: Dict[str, Any],
    router_decision: Dict[str, Any],
    model: str = "gpt-4o-mini",
    extra_instructions: Optional[str] = None,
) -> BlandPathwayPayload:
    task_obj: Dict[str, Any] = {
        "USER_QUERY": user_query,
        "TEMPLATE_PATHWAY": template_pathway,
        "SELECTED_WORKFLOW": selected_workflow,
        "ROUTER_DECISION": router_decision,
    }
    if extra_instructions:
        task_obj["EXTRA_INSTRUCTIONS"] = extra_instructions

    task = json.dumps(task_obj, indent=2)

    model_client = OpenAIChatCompletionClient(model=model)
    agent = AssistantAgent(name="pathway_builder", model_client=model_client, system_message=PATHWAY_BUILDER_SYSTEM)

    try:
        res = await agent.run(task=task)
        content = res.messages[-1].content if res.messages else ""
        try:
            raw = json.loads(_extract_json(content))
        except Exception as e:
            raise ValueError(f"Pathway builder did not return valid JSON. Raw output:\n{content}") from e

        data = _normalize_payload(raw)
        return BlandPathwayPayload(**data)
    finally:
        try:
            await model_client.close()
        except Exception:
            pass


__all__ = ["build_bland_pathway_payload"]
