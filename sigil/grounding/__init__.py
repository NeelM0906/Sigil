"""Grounding phase for Sigil v2 - Information gathering and clarification.

The grounding phase sits between routing and planning in the orchestration pipeline.
It analyzes incoming goals to:
1. Identify information needs
2. Check existing knowledge (memory, context)
3. Identify gaps that require clarification
4. Enrich the request with gathered information

This helps ensure the planner has sufficient context to create effective plans.
"""

from sigil.grounding.schemas import (
    GroundingRequest,
    GroundingResult,
    InformationNeed,
    InformationGap,
    GapType,
    GroundingStatus,
)
from sigil.grounding.grounder import Grounder

__all__ = [
    "Grounder",
    "GroundingRequest",
    "GroundingResult",
    "InformationNeed",
    "InformationGap",
    "GapType",
    "GroundingStatus",
]
