from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class BlandPathwayPayload(BaseModel):
    """Create/update payload for a Bland conversational pathway."""

    name: str
    description: str = ""
    nodes: List[Dict[str, Any]] = Field(default_factory=list)
    edges: List[Dict[str, Any]] = Field(default_factory=list)

    # Traceability (not sent to Bland unless you choose to)
    selected_workflow_id: Optional[str] = None
    selected_workflow_name: Optional[str] = None
    router_reasoning: Optional[str] = None
    matched_signals: List[str] = Field(default_factory=list)


class BlandPathwayBuildResult(BaseModel):
    """Result of building + pushing a pathway to Bland."""

    status: str
    pathway_id: Optional[str] = None
    name: Optional[str] = None
    description: Optional[str] = None
    template_workflow_id: Optional[str] = None
    template_file_path: Optional[str] = None
    api_response: Optional[Dict[str, Any]] = None

