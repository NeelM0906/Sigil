from __future__ import annotations

import json
import re
from typing import Any, Dict, Optional

from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient

from ..core.agent_spec import AgentSpec


BUILDER_SYSTEM = """You are Sigil Builder (a meta-agent) that generates an executable AgentSpec.

You will be given:
- USER_QUERY
- SELECTED_WORKFLOW (source/workflow_id/workflow_name/profile_preview)
- ROUTER_DECISION (router reasoning, matched signals, confidence)
- OPTIONAL: REUSABLE_COMPONENTS_HINTS (components/snippets extracted from older configs)

Your job: output STRICT JSON (no markdown, no code fences) matching this schema exactly:

{
  "agent_name": "...",
  "description": "...",
  "stratum": "RTI" | "RAI" | "ZACS" | "EEI" | "IGE",
  "tools": ["workflow","communication","calendar","voice","websearch","crm","storage"],
  "system_prompt": "...",
  "conversation_flow": ["...", "..."],
  "guardrails": ["...", "..."],
  "success_criteria": ["...", "..."],
  "version": "v1.0",

  "selected_workflow_id": "...",
  "selected_workflow_source": "bland" | "n8n",
  "selected_workflow_name": "...",
  "router_reasoning": "...",
  "matched_signals": ["..."],

  "reused_components": [],
  "base_workflows": ["<selected_workflow_id>"],
  "runtime_config_placeholders": {}
}

Rules:
- Must include agent_name + description (required).
- tools should be a flat list (NOT nested).
- base_workflows should always include the selected_workflow_id at minimum.
- Keep prompts + flows practical and specific.
"""


def _extract_json(text: str) -> str:
    text = re.sub(r"^```(json)?\s*", "", text.strip(), flags=re.IGNORECASE)
    text = re.sub(r"```\s*$", "", text.strip())
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    return m.group(0) if m else text


def _normalize_builder_payload(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Make builder output robust to slight format drift."""
    data = dict(raw)

    # Some models wrap output like {"AgentSpec": {...}}
    if "AgentSpec" in data and isinstance(data["AgentSpec"], dict):
        data = dict(data["AgentSpec"])

    # Map common alternative keys
    if "name" in data and "agent_name" not in data:
        data["agent_name"] = data.pop("name")
    if "agent" in data and isinstance(data["agent"], dict) and "agent_name" not in data:
        maybe = data["agent"]
        if "name" in maybe:
            data["agent_name"] = maybe.get("name")

    # Description fallback
    if "description" not in data or not str(data.get("description") or "").strip():
        # Try nested tools.description (we sometimes had this)
        tools_obj = data.get("tools")
        if isinstance(tools_obj, dict) and "description" in tools_obj:
            data["description"] = str(tools_obj.get("description") or "").strip()
        else:
            data["description"] = f"Agent for: {data.get('agent_name','') or 'user request'}".strip()

    # Flatten tools if nested {primary:[...]}
    tools_val = data.get("tools")
    if isinstance(tools_val, dict):
        primary = tools_val.get("primary") or tools_val.get("tool_categories") or []
        if isinstance(primary, list):
            data["tools"] = primary
        else:
            data["tools"] = []
    elif tools_val is None:
        data["tools"] = []

    return data


async def build_agent_spec(
    user_query: str,
    selected_workflow: Dict[str, Any],
    router_decision: Dict[str, Any],
    model: str = "gpt-4o-mini",
    components_hint: Optional[Dict[str, Any]] = None,
) -> AgentSpec:
    task_obj = {
        "USER_QUERY": user_query,
        "SELECTED_WORKFLOW": selected_workflow,
        "ROUTER_DECISION": router_decision,
        "REUSABLE_COMPONENTS_HINTS": components_hint or {},
    }
    task = json.dumps(task_obj, indent=2)

    model_client = OpenAIChatCompletionClient(model=model)
    builder = AssistantAgent(name="builder", model_client=model_client, system_message=BUILDER_SYSTEM)

    try:
        res = await builder.run(task=task)
        content = res.messages[-1].content if res.messages else ""
        try:
            raw = json.loads(_extract_json(content))
        except Exception as e:
            raise ValueError(f"Builder did not return valid JSON. Raw output:\n{content}") from e

        data = _normalize_builder_payload(raw)

        # Inject traceability fields from router decision + selected workflow
        data.setdefault("selected_workflow_id", selected_workflow.get("workflow_id"))
        data.setdefault("selected_workflow_source", selected_workflow.get("source"))
        data.setdefault("selected_workflow_name", selected_workflow.get("workflow_name"))
        data.setdefault("router_reasoning", router_decision.get("reasoning", ""))
        data.setdefault("matched_signals", router_decision.get("matched_signals") or [])

        # Ensure base_workflows includes the selected id
        bw = data.get("base_workflows")
        if not isinstance(bw, list):
            bw = []
        if data.get("selected_workflow_id") and data["selected_workflow_id"] not in bw:
            bw = [data["selected_workflow_id"], *bw]
        data["base_workflows"] = bw

        # Reused components from hints
        if components_hint:
            reused = components_hint.get("reusable_components") or []
            if isinstance(reused, list) and reused:
                data.setdefault("reused_components", reused)
        data.setdefault("reused_components", [])

        data.setdefault("runtime_config_placeholders", {})
        data.setdefault("version", "v1.0")

        return AgentSpec(**data)
    finally:
        try:
            await model_client.close()
        except Exception:
            pass


__all__ = ["build_agent_spec"]
