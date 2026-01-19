from __future__ import annotations

import json
import re
from typing import List

from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient

from ..core.schemas import CandidateWorkflow, RouteResult, ConsideredWorkflow

ROUTER_SYSTEM = """You are an ACTi workflow router.

You will be given:
- USER_QUERY
- A list of CANDIDATE_WORKFLOWS (each has: source, workflow_id, workflow_name, similarity, file_path, profile_preview)

Return STRICT JSON (no markdown, no code fences) with this schema:

{
  "source": "bland" | "n8n",
  "workflow_id": "<string>",
  "workflow_name": "<string>",
  "confidence": <float 0..1>,
  "reasoning": "<short explanation>",
  "matched_signals": ["<signal>", ...],
  "considered": [
    {"workflow_id": "<string>", "why_not": "<short>"}
  ]
}

Rules:
- Pick exactly ONE best workflow_id from the provided candidates.
- confidence reflects your certainty among the provided candidates, not global certainty.
- Keep reasoning concise and grounded in the candidate previews.
"""


def _extract_json(text: str) -> str:
    """Extract the first JSON object from a text blob."""
    # Remove common fences
    text = re.sub(r"^```(json)?\s*", "", text.strip(), flags=re.IGNORECASE)
    text = re.sub(r"```\s*$", "", text.strip())
    # Find first {...}
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    return m.group(0) if m else text


async def route_with_autogen(
    user_query: str,
    candidates: List[CandidateWorkflow],
    model: str = "gpt-4o-mini",
) -> RouteResult:
    payload = [c.model_dump() for c in candidates]
    task = (
        "USER_QUERY:\n" + user_query.strip() + "\n\n" +
        "CANDIDATE_WORKFLOWS:\n" + json.dumps(payload, indent=2)
    )

    model_client = OpenAIChatCompletionClient(model=model)
    router = AssistantAgent(name="router", model_client=model_client, system_message=ROUTER_SYSTEM)

    try:
        res = await router.run(task=task)
        content = res.messages[-1].content if res.messages else ""
        try:
            data = json.loads(_extract_json(content))
        except Exception as e:
            raise ValueError(f"Router did not return valid JSON. Raw output:\n{content}") from e

        # Enrich / normalize considered entries to satisfy schema while staying flexible.
        considered_raw = data.get("considered") or []
        considered: List[ConsideredWorkflow] = []
        by_id = {c.workflow_id: c for c in candidates}
        for item in considered_raw:
            if not isinstance(item, dict) or "workflow_id" not in item:
                continue
            wid = str(item.get("workflow_id"))
            why = str(item.get("why_not") or item.get("reason") or "").strip() or "Not selected."
            cw = by_id.get(wid)
            considered.append(
                ConsideredWorkflow(
                    workflow_id=wid,
                    why_not=why,
                    source=(cw.source if cw else None),
                    workflow_name=(cw.workflow_name if cw else None),
                    similarity=(cw.similarity if cw else None),
                    file_path=(cw.file_path if cw else None),
                    profile_preview=(cw.profile_preview if cw else None),
                )
            )
        data["considered"] = [c.model_dump() for c in considered] if considered else None

        return RouteResult(**data)
    finally:
        # Prevent Windows "Event loop is closed" noise from httpx transports.
        try:
            await model_client.close()
        except Exception:
            pass


__all__ = ["route_with_autogen"]
