from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field

SourceType = Literal["bland", "n8n"]


class WorkflowRecord(BaseModel):
    """Canonical record for a stored workflow/config (Bland or n8n)."""

    source: SourceType
    workflow_id: str
    workflow_name: str
    file_path: str

    # Free-form metadata
    description: Optional[str] = None
    tags: List[str] = Field(default_factory=list)

    # Profile text used for retrieval (embeddings). We keep BOTH names because earlier
    # iterations used `profile_preview` only.
    profile: str = ""
    profile_preview: str = ""

    # Original raw JSON (optional)
    raw: Optional[Dict[str, Any]] = None

    def model_post_init(self, __context: Any) -> None:
        # Keep preview in sync if not provided
        if not self.profile_preview and self.profile:
            self.profile_preview = self.profile[:800]


class CandidateWorkflow(BaseModel):
    """Workflow candidate returned by retrieval/shortlisting."""

    source: SourceType
    workflow_id: str
    workflow_name: str
    similarity: float
    file_path: str
    profile_preview: str


class ConsideredWorkflow(BaseModel):
    """Lightweight 'runner-up' explanation from the router model.

    The router LLM often returns only {workflow_id, why_not}. We therefore keep other
    fields optional to avoid brittle validation.
    """

    workflow_id: str
    why_not: str

    # Optional enrichment fields (when available)
    source: Optional[SourceType] = None
    workflow_name: Optional[str] = None
    similarity: Optional[float] = None
    file_path: Optional[str] = None
    profile_preview: Optional[str] = None


class RouteResult(BaseModel):
    source: SourceType
    workflow_id: str
    workflow_name: str
    confidence: float
    reasoning: str
    matched_signals: List[str] = Field(default_factory=list)
    considered: Optional[List[ConsideredWorkflow]] = None


# Re-export AgentSpec from its own module so older imports keep working.
try:
    from .agent_spec import AgentSpec  # noqa: F401
except Exception:  # pragma: no cover
    AgentSpec = None  # type: ignore


__all__ = [
    "SourceType",
    "WorkflowRecord",
    "CandidateWorkflow",
    "ConsideredWorkflow",
    "RouteResult",
    "AgentSpec",
]
