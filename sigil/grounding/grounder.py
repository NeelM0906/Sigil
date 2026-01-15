"""Grounder for Sigil v2 - Information gathering and clarification.

This module implements the Grounder class which analyzes incoming goals
to identify information needs, check existing knowledge, and enrich
requests before planning.

The Grounder sits between routing and planning in the pipeline:
    Route -> Ground -> Plan -> Execute

Example:
    >>> grounder = Grounder(memory_manager=mm)
    >>> result = await grounder.ground(
    ...     GroundingRequest(
    ...         message="Schedule a meeting with John",
    ...         session_id="sess-123",
    ...     )
    ... )
    >>> if result.can_proceed:
    ...     # Continue to planning
    ... else:
    ...     # Ask clarification questions
"""

from __future__ import annotations

import logging
import re
import time
from typing import Any, Optional, TYPE_CHECKING

from sigil.grounding.schemas import (
    GroundingRequest,
    GroundingResult,
    GroundingStatus,
    InformationNeed,
    InformationGap,
    GapType,
)
from sigil.state.events import Event, EventType, _generate_event_id, _get_utc_now
from sigil.state.store import EventStore

if TYPE_CHECKING:
    from sigil.memory.manager import MemoryManager


# =============================================================================
# Module Logger
# =============================================================================

logger = logging.getLogger(__name__)


# =============================================================================
# Constants
# =============================================================================

# Keywords that suggest specific information needs
TEMPORAL_KEYWORDS = [
    "when", "time", "date", "tomorrow", "today", "next week",
    "schedule", "meeting", "appointment", "deadline",
]

ENTITY_KEYWORDS = [
    "who", "contact", "person", "company", "client", "customer",
    "lead", "team", "manager",
]

LOCATION_KEYWORDS = [
    "where", "location", "place", "address", "room", "office",
]

QUANTITY_KEYWORDS = [
    "how many", "how much", "count", "number", "amount", "budget",
]

ACTION_KEYWORDS = [
    "search", "find", "create", "update", "delete", "send", "call",
    "email", "schedule", "book", "cancel",
]


# =============================================================================
# Event Creators
# =============================================================================


def create_grounding_started_event(
    session_id: str,
    message_preview: str,
    correlation_id: Optional[str] = None,
) -> Event:
    """Create a grounding started event."""
    return Event(
        event_id=_generate_event_id(),
        event_type=EventType.MESSAGE_ADDED,
        timestamp=_get_utc_now(),
        session_id=session_id,
        correlation_id=correlation_id,
        payload={
            "phase": "grounding",
            "action": "started",
            "message_preview": message_preview[:100],
        },
    )


def create_grounding_completed_event(
    session_id: str,
    status: GroundingStatus,
    needs_count: int,
    gaps_count: int,
    can_proceed: bool,
    correlation_id: Optional[str] = None,
) -> Event:
    """Create a grounding completed event."""
    return Event(
        event_id=_generate_event_id(),
        event_type=EventType.MESSAGE_ADDED,
        timestamp=_get_utc_now(),
        session_id=session_id,
        correlation_id=correlation_id,
        payload={
            "phase": "grounding",
            "action": "completed",
            "status": status.value,
            "needs_count": needs_count,
            "gaps_count": gaps_count,
            "can_proceed": can_proceed,
        },
    )


# =============================================================================
# Grounder
# =============================================================================


class Grounder:
    """Analyzes goals to gather information and identify gaps.

    The Grounder performs several key functions:
    1. Identifies information needs from the goal
    2. Checks existing knowledge (memory, context)
    3. Identifies gaps that may need clarification
    4. Enriches the request with gathered information

    Attributes:
        memory_manager: Optional memory manager for knowledge retrieval
        event_store: Optional event store for audit trails
        strict_mode: If True, require all critical info before proceeding

    Example:
        >>> grounder = Grounder()
        >>> result = await grounder.ground(request)
        >>> if not result.can_proceed:
        ...     for q in result.clarification_questions:
        ...         print(f"Please clarify: {q}")
    """

    def __init__(
        self,
        memory_manager: Optional["MemoryManager"] = None,
        event_store: Optional[EventStore] = None,
        strict_mode: bool = False,
    ) -> None:
        """Initialize the Grounder.

        Args:
            memory_manager: Optional memory manager for knowledge retrieval.
            event_store: Optional event store for audit trails.
            strict_mode: If True, require all critical info before proceeding.
        """
        self._memory_manager = memory_manager
        self._event_store = event_store or EventStore()
        self._strict_mode = strict_mode

    async def ground(
        self,
        request: GroundingRequest,
        correlation_id: Optional[str] = None,
    ) -> GroundingResult:
        """Ground a request by gathering information and identifying gaps.

        This is the main entry point for the grounding phase.

        Args:
            request: The grounding request
            correlation_id: Optional correlation ID for tracing

        Returns:
            GroundingResult with enriched context and any gaps identified
        """
        start_time = time.perf_counter()

        # Emit started event
        self._event_store.append(
            create_grounding_started_event(
                session_id=request.session_id,
                message_preview=request.message,
                correlation_id=correlation_id,
            )
        )

        try:
            # Step 1: Analyze the message to identify information needs
            information_needs = self._analyze_information_needs(request)

            # Step 2: Check existing knowledge to resolve needs
            await self._resolve_from_context(information_needs, request)
            await self._resolve_from_memory(information_needs, request)

            # Step 3: Identify remaining gaps
            information_gaps = self._identify_gaps(information_needs, request)

            # Step 4: Generate clarification questions for critical gaps
            clarification_questions = self._generate_clarification_questions(
                information_gaps
            )

            # Step 5: Enrich the context with gathered information
            enriched_context = self._build_enriched_context(
                request, information_needs, information_gaps
            )

            # Step 6: Determine status and whether to proceed
            status, can_proceed, confidence = self._determine_status(
                information_needs, information_gaps
            )

            # Build enriched message if we have additional context
            enriched_message = self._build_enriched_message(
                request.message, information_needs
            )

            processing_time_ms = (time.perf_counter() - start_time) * 1000

            result = GroundingResult(
                status=status,
                enriched_message=enriched_message,
                enriched_context=enriched_context,
                information_needs=information_needs,
                information_gaps=information_gaps,
                clarification_questions=clarification_questions,
                confidence=confidence,
                can_proceed=can_proceed,
                tokens_used=0,  # No LLM tokens in rule-based grounding
                processing_time_ms=processing_time_ms,
            )

            logger.info(
                f"Grounding completed: status={status.value}, "
                f"needs={len(information_needs)}, gaps={len(information_gaps)}, "
                f"can_proceed={can_proceed}"
            )

        except Exception as e:
            logger.error(f"Grounding failed: {e}")
            processing_time_ms = (time.perf_counter() - start_time) * 1000
            result = GroundingResult(
                status=GroundingStatus.FAILED,
                enriched_message=request.message,
                enriched_context=request.context.copy(),
                confidence=0.0,
                can_proceed=True,  # Allow proceeding on failure
                processing_time_ms=processing_time_ms,
            )

        # Emit completed event
        self._event_store.append(
            create_grounding_completed_event(
                session_id=request.session_id,
                status=result.status,
                needs_count=len(result.information_needs),
                gaps_count=len(result.information_gaps),
                can_proceed=result.can_proceed,
                correlation_id=correlation_id,
            )
        )

        return result

    def _analyze_information_needs(
        self,
        request: GroundingRequest,
    ) -> list[InformationNeed]:
        """Analyze the message to identify information needs.

        Uses keyword matching and pattern recognition to identify
        what information is needed to complete the task.

        Args:
            request: The grounding request

        Returns:
            List of identified information needs
        """
        needs: list[InformationNeed] = []
        message_lower = request.message.lower()

        # Check for temporal information needs
        if any(kw in message_lower for kw in TEMPORAL_KEYWORDS):
            # Check if specific time is mentioned
            has_specific_time = bool(
                re.search(r'\d{1,2}:\d{2}|\d{1,2}\s*(am|pm)|tomorrow|today|next\s+\w+day', message_lower)
            )
            if not has_specific_time:
                needs.append(InformationNeed(
                    category="temporal",
                    description="Specific time or date for the action",
                    priority=1 if "schedule" in message_lower or "meeting" in message_lower else 2,
                    source="user",
                ))

        # Check for entity information needs
        if any(kw in message_lower for kw in ENTITY_KEYWORDS):
            # Check for specific names - require at least two capitalized words
            # (first + last name pattern, not just sentence-starting words)
            # Also accept single capitalized word not at sentence start
            has_specific_name = bool(
                re.search(r'(?<=[.!?\s])\s*[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b', request.message)  # "John Smith"
                or re.search(r'\swith\s+[A-Z][a-z]+\b', request.message)  # "with John"
            )
            # Check for ambiguous references like "the team", "the client"
            has_ambiguous_reference = bool(
                re.search(r'\bthe\s+(team|client|customer|manager|lead|group|department)\b', message_lower)
            )
            if not has_specific_name or has_ambiguous_reference:
                needs.append(InformationNeed(
                    category="entity",
                    description="Specific person or entity to interact with",
                    priority=1,
                    source="user",
                ))

        # Check for location information needs
        if any(kw in message_lower for kw in LOCATION_KEYWORDS):
            needs.append(InformationNeed(
                category="location",
                description="Location or place information",
                priority=2,
                source="user",
            ))

        # Check for quantity information needs
        if any(kw in message_lower for kw in QUANTITY_KEYWORDS):
            has_number = bool(re.search(r'\d+', message_lower))
            if not has_number:
                needs.append(InformationNeed(
                    category="quantity",
                    description="Specific quantity or amount",
                    priority=2,
                    source="user",
                ))

        # Check for action-specific information needs
        if "search" in message_lower or "find" in message_lower:
            needs.append(InformationNeed(
                category="search_query",
                description="Search query or topic",
                priority=1,
                source="message",
                resolved=True,  # The message itself contains the query
                resolved_value=request.message,
            ))

        if "email" in message_lower or "send" in message_lower:
            # Check for recipient
            if not re.search(r'\bto\s+\w+', message_lower):
                needs.append(InformationNeed(
                    category="recipient",
                    description="Recipient of the communication",
                    priority=1,
                    source="user",
                ))

        # Check intent from routing
        if request.route_intent:
            if request.route_intent == "run_agent":
                needs.append(InformationNeed(
                    category="task_definition",
                    description="Clear task definition for the agent",
                    priority=1,
                    source="message",
                    resolved=True,
                    resolved_value=request.message,
                ))

        return needs

    async def _resolve_from_context(
        self,
        needs: list[InformationNeed],
        request: GroundingRequest,
    ) -> None:
        """Try to resolve information needs from provided context.

        Args:
            needs: List of information needs to resolve
            request: The grounding request with context
        """
        context = request.context

        for need in needs:
            if need.resolved:
                continue

            # Try to resolve from context based on category
            if need.category == "entity" and "contact" in context:
                need.resolve(context["contact"])
            elif need.category == "entity" and "person" in context:
                need.resolve(context["person"])
            elif need.category == "temporal" and "time" in context:
                need.resolve(context["time"])
            elif need.category == "temporal" and "date" in context:
                need.resolve(context["date"])
            elif need.category == "location" and "location" in context:
                need.resolve(context["location"])
            elif need.category == "recipient" and "to" in context:
                need.resolve(context["to"])
            elif need.category == "recipient" and "email" in context:
                need.resolve(context["email"])

    async def _resolve_from_memory(
        self,
        needs: list[InformationNeed],
        request: GroundingRequest,
    ) -> None:
        """Try to resolve information needs from memory.

        Args:
            needs: List of information needs to resolve
            request: The grounding request with memory results
        """
        if not self._memory_manager and not request.memory_results:
            return

        # Use provided memory results if available
        memory_results = request.memory_results

        # If no pre-fetched results but we have memory manager, query
        if not memory_results and self._memory_manager:
            try:
                memories = await self._memory_manager.retrieve(
                    query=request.message,
                    k=5,
                )
                memory_results = [
                    {"content": m.content, "category": m.category}
                    for m in memories
                ]
            except Exception as e:
                logger.warning(f"Memory retrieval failed: {e}")
                return

        # Try to resolve needs from memory
        for need in needs:
            if need.resolved:
                continue

            for memory in memory_results:
                content = memory.get("content", "")
                category = memory.get("category", "")

                # Check if memory is relevant to this need
                if need.category == "entity" and category in ["contacts", "people"]:
                    need.resolve(content)
                    need.source = "memory"
                    break
                elif need.category == "temporal" and "schedule" in category:
                    need.resolve(content)
                    need.source = "memory"
                    break

    def _identify_gaps(
        self,
        needs: list[InformationNeed],
        request: GroundingRequest,
    ) -> list[InformationGap]:
        """Identify gaps from unresolved information needs.

        Args:
            needs: List of information needs
            request: The grounding request

        Returns:
            List of identified gaps
        """
        gaps: list[InformationGap] = []

        for need in needs:
            if need.resolved:
                continue

            # Map need category to gap type
            gap_type = GapType.MISSING_PARAMETER
            if need.category == "entity":
                gap_type = GapType.AMBIGUOUS_ENTITY
            elif need.category in ["task_definition", "search_query"]:
                gap_type = GapType.UNCLEAR_INTENT
            elif need.category in ["context", "background"]:
                gap_type = GapType.MISSING_CONTEXT

            # Determine severity based on priority
            severity = need.priority

            # Generate suggestion for resolving the gap
            suggestion = self._generate_gap_suggestion(need, request)

            # Check if we can reasonably infer
            can_infer = need.priority >= 3  # Only low priority items can be inferred

            gaps.append(InformationGap(
                gap_type=gap_type,
                description=need.description,
                affected_element=need.category,
                severity=severity,
                suggestion=suggestion,
                can_infer=can_infer,
            ))

        return gaps

    def _generate_gap_suggestion(
        self,
        need: InformationNeed,
        request: GroundingRequest,
    ) -> str:
        """Generate a suggestion for resolving an information gap.

        Args:
            need: The unresolved information need
            request: The grounding request

        Returns:
            Suggested question or action
        """
        suggestions = {
            "temporal": "What date and time would you like?",
            "entity": "Who specifically would you like to interact with?",
            "location": "What location or place?",
            "quantity": "How many or how much?",
            "recipient": "Who should receive this?",
            "task_definition": "Could you clarify what you'd like to accomplish?",
        }
        return suggestions.get(need.category, f"Please provide: {need.description}")

    def _generate_clarification_questions(
        self,
        gaps: list[InformationGap],
    ) -> list[str]:
        """Generate clarification questions for critical gaps.

        Args:
            gaps: List of information gaps

        Returns:
            List of clarification questions
        """
        questions: list[str] = []

        # Sort by severity (most critical first)
        sorted_gaps = sorted(gaps, key=lambda g: g.severity)

        for gap in sorted_gaps:
            if gap.severity <= 2 and gap.suggestion:  # Critical or significant
                questions.append(gap.suggestion)

        return questions[:3]  # Limit to 3 questions max

    def _build_enriched_context(
        self,
        request: GroundingRequest,
        needs: list[InformationNeed],
        gaps: list[InformationGap],
    ) -> dict[str, Any]:
        """Build enriched context from gathered information.

        Args:
            request: The grounding request
            needs: List of information needs
            gaps: List of information gaps

        Returns:
            Enriched context dictionary
        """
        enriched = request.context.copy()

        # Add resolved information
        resolved_info = {}
        for need in needs:
            if need.resolved and need.resolved_value:
                resolved_info[need.category] = need.resolved_value

        if resolved_info:
            enriched["resolved_info"] = resolved_info

        # Add gap summary
        if gaps:
            enriched["information_gaps"] = [
                {"type": g.gap_type.value, "description": g.description}
                for g in gaps
                if g.severity <= 2  # Only include significant gaps
            ]

        # Add memory context if available
        if request.memory_results:
            enriched["memory_context"] = request.memory_results[:3]

        return enriched

    def _build_enriched_message(
        self,
        original_message: str,
        needs: list[InformationNeed],
    ) -> str:
        """Build enriched message with resolved information.

        Args:
            original_message: The original user message
            needs: List of information needs with resolved values

        Returns:
            Enriched message string
        """
        # For now, return original message
        # In the future, could augment with resolved context
        enriched = original_message

        # Add context annotations for resolved needs
        resolved = [n for n in needs if n.resolved and n.source == "memory"]
        if resolved:
            annotations = []
            for need in resolved:
                annotations.append(f"[{need.category}: {need.resolved_value}]")
            if annotations:
                enriched = f"{original_message}\n\nContext: {', '.join(annotations)}"

        return enriched

    def _determine_status(
        self,
        needs: list[InformationNeed],
        gaps: list[InformationGap],
    ) -> tuple[GroundingStatus, bool, float]:
        """Determine grounding status and whether to proceed.

        Args:
            needs: List of information needs
            gaps: List of information gaps

        Returns:
            Tuple of (status, can_proceed, confidence)
        """
        if not needs:
            # No information needs identified - complete
            return GroundingStatus.COMPLETE, True, 1.0

        # Count resolved needs
        resolved_count = sum(1 for n in needs if n.resolved)
        total_needs = len(needs)
        resolution_ratio = resolved_count / total_needs if total_needs > 0 else 1.0

        # Check for critical gaps
        critical_gaps = [g for g in gaps if g.severity == 1]
        significant_gaps = [g for g in gaps if g.severity == 2]

        if not gaps:
            return GroundingStatus.COMPLETE, True, resolution_ratio

        if critical_gaps:
            if self._strict_mode:
                return GroundingStatus.NEEDS_CLARIFICATION, False, resolution_ratio
            else:
                # Allow proceeding but with warning
                return GroundingStatus.PARTIAL, True, resolution_ratio * 0.7

        if significant_gaps:
            return GroundingStatus.PARTIAL, True, resolution_ratio * 0.8

        return GroundingStatus.COMPLETE, True, resolution_ratio


__all__ = ["Grounder"]
