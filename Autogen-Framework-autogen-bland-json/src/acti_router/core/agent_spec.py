from __future__ import annotations

from pydantic import BaseModel, Field
from typing import Dict, List, Literal

ActiStratum = Literal["RTI", "RAI", "ZACS", "EEI", "IGE"]
ToolCategory = Literal["voice", "websearch", "calendar", "communication", "crm", "workflow", "storage"]

class AgentSpec(BaseModel):
    agent_name: str
    description: str
    stratum: ActiStratum
    tools: List[ToolCategory] = Field(default_factory=list)

    system_prompt: str
    conversation_flow: List[str] = Field(default_factory=list)
    guardrails: List[str] = Field(default_factory=list)
    success_criteria: List[str] = Field(default_factory=list)
    version: str = "v1.0"

    # Traceability to routing decision / dataset
    selected_workflow_id: str
    selected_workflow_source: str
    selected_workflow_name: str
    router_reasoning: str
    matched_signals: List[str] = Field(default_factory=list)
    reused_components: list = Field(default_factory=list)
    base_workflows: list = Field(default_factory=list)

    # Execution placeholders (no secrets)
    runtime_config_placeholders: Dict[str, str] = Field(default_factory=dict)
