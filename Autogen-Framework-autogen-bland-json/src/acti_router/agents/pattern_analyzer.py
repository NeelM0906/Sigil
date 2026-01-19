from __future__ import annotations
import json, re
from typing import Dict, Any
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient

def _extract_json(text: str) -> str:
    text = (text or "").strip()
    text = re.sub(r"^```json\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^```\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        raise ValueError(f"No JSON found in:\n{text}")
    return m.group(0).strip()

async def _run(system_message: str, payload: Dict[str, Any], model: str) -> Dict[str, Any]:
    agent = AssistantAgent(
        name="subagent",
        model_client=OpenAIChatCompletionClient(model=model),
        system_message=system_message,
    )
    res = await agent.run(task=json.dumps(payload, indent=2))
    js = _extract_json(res.messages[-1].content)
    return json.loads(js)

from ..prompts import PATTERN_ANALYZER_SYSTEM

async def run(user_request: str, shortlisted_workflows, model: str = "gpt-4o-mini"):
    payload = {"user_request": user_request, "shortlisted_workflows": shortlisted_workflows, "ask": "Extract reusable patterns AND suggest reusable components/snippets to reuse."}
    return await _run(PATTERN_ANALYZER_SYSTEM, payload, model)
