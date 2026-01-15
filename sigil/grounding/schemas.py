"""Schemas for the grounding phase.

This module defines data models for information gathering and clarification
during the grounding phase of the orchestration pipeline.

Classes:
    GapType: Types of information gaps
    GroundingStatus: Status of grounding process
    InformationNeed: Identified information requirement
    InformationGap: Gap in available information
    GroundingRequest: Input to the grounding phase
    GroundingResult: Output from the grounding phase
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional


def generate_uuid() -> str:
    """Generate a UUID4 hex string."""
    return uuid.uuid4().hex


def utc_now() -> datetime:
    """Get the current UTC datetime."""
    return datetime.now(timezone.utc)


class GapType(str, Enum):
    """Types of information gaps that may require clarification.

    Attributes:
        AMBIGUOUS_ENTITY: Entity reference is ambiguous (e.g., "the meeting")
        MISSING_PARAMETER: Required parameter not specified (e.g., time, date)
        UNCLEAR_INTENT: User intent is unclear
        MISSING_CONTEXT: Background context is missing
        CONFLICTING_INFO: Conflicting information in the request
        INSUFFICIENT_DETAIL: Request lacks necessary detail
    """

    AMBIGUOUS_ENTITY = "ambiguous_entity"
    MISSING_PARAMETER = "missing_parameter"
    UNCLEAR_INTENT = "unclear_intent"
    MISSING_CONTEXT = "missing_context"
    CONFLICTING_INFO = "conflicting_info"
    INSUFFICIENT_DETAIL = "insufficient_detail"


class GroundingStatus(str, Enum):
    """Status of the grounding process.

    Attributes:
        COMPLETE: All information needs satisfied
        PARTIAL: Some gaps remain but can proceed
        NEEDS_CLARIFICATION: Critical gaps require user input
        FAILED: Grounding process failed
    """

    COMPLETE = "complete"
    PARTIAL = "partial"
    NEEDS_CLARIFICATION = "needs_clarification"
    FAILED = "failed"


@dataclass
class InformationNeed:
    """An identified information requirement for the task.

    Represents something the system needs to know to complete the task.

    Attributes:
        need_id: Unique identifier
        category: Category of information (entity, parameter, context, etc.)
        description: Description of what is needed
        priority: Priority level (1=critical, 2=important, 3=nice-to-have)
        source: Potential source (memory, context, user, tool)
        resolved: Whether this need has been satisfied
        resolved_value: The value if resolved
    """

    need_id: str = field(default_factory=generate_uuid)
    category: str = ""
    description: str = ""
    priority: int = 2
    source: str = "unknown"
    resolved: bool = False
    resolved_value: Optional[Any] = None

    def resolve(self, value: Any) -> None:
        """Mark this need as resolved with a value."""
        self.resolved = True
        self.resolved_value = value


@dataclass
class InformationGap:
    """A gap in available information that may need clarification.

    Represents missing or ambiguous information that could prevent
    successful task completion.

    Attributes:
        gap_id: Unique identifier
        gap_type: Type of gap
        description: Description of the gap
        affected_element: The element (entity, parameter) affected
        severity: How critical this gap is (1=blocking, 2=significant, 3=minor)
        suggestion: Suggested question to ask user
        can_infer: Whether the system can reasonably infer this
        inferred_value: Value if inference was attempted
    """

    gap_id: str = field(default_factory=generate_uuid)
    gap_type: GapType = GapType.MISSING_PARAMETER
    description: str = ""
    affected_element: str = ""
    severity: int = 2
    suggestion: Optional[str] = None
    can_infer: bool = False
    inferred_value: Optional[Any] = None


@dataclass
class GroundingRequest:
    """Input to the grounding phase.

    Attributes:
        message: The original user message
        session_id: Session identifier
        context: Additional context from the request
        route_intent: Classified intent from routing
        route_complexity: Complexity score from routing
        available_tools: List of available tools
        memory_results: Results from memory retrieval (if any)
    """

    message: str
    session_id: str
    context: dict[str, Any] = field(default_factory=dict)
    route_intent: Optional[str] = None
    route_complexity: float = 0.5
    available_tools: list[str] = field(default_factory=list)
    memory_results: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class GroundingResult:
    """Output from the grounding phase.

    Attributes:
        status: Status of the grounding process
        enriched_message: Original message enriched with context
        enriched_context: Context dictionary enriched with gathered info
        information_needs: List of identified information needs
        information_gaps: List of identified gaps
        clarification_questions: Questions to ask user (if needed)
        confidence: Confidence in the grounding (0.0-1.0)
        can_proceed: Whether planning can proceed
        tokens_used: Tokens used in grounding
        processing_time_ms: Time spent in grounding
    """

    status: GroundingStatus
    enriched_message: str
    enriched_context: dict[str, Any] = field(default_factory=dict)
    information_needs: list[InformationNeed] = field(default_factory=list)
    information_gaps: list[InformationGap] = field(default_factory=list)
    clarification_questions: list[str] = field(default_factory=list)
    confidence: float = 1.0
    can_proceed: bool = True
    tokens_used: int = 0
    processing_time_ms: float = 0.0

    @property
    def has_critical_gaps(self) -> bool:
        """Check if there are critical (blocking) gaps."""
        return any(gap.severity == 1 for gap in self.information_gaps)

    @property
    def unresolved_needs_count(self) -> int:
        """Count of unresolved information needs."""
        return sum(1 for need in self.information_needs if not need.resolved)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "status": self.status.value,
            "enriched_message": self.enriched_message,
            "enriched_context": self.enriched_context,
            "information_needs": [
                {
                    "need_id": n.need_id,
                    "category": n.category,
                    "description": n.description,
                    "resolved": n.resolved,
                }
                for n in self.information_needs
            ],
            "information_gaps": [
                {
                    "gap_id": g.gap_id,
                    "gap_type": g.gap_type.value,
                    "description": g.description,
                    "severity": g.severity,
                    "suggestion": g.suggestion,
                }
                for g in self.information_gaps
            ],
            "clarification_questions": self.clarification_questions,
            "confidence": self.confidence,
            "can_proceed": self.can_proceed,
            "tokens_used": self.tokens_used,
            "processing_time_ms": self.processing_time_ms,
        }


__all__ = [
    "GapType",
    "GroundingStatus",
    "InformationNeed",
    "InformationGap",
    "GroundingRequest",
    "GroundingResult",
]
