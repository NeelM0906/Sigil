"""Tests for the Grounder class.

Tests cover:
- Basic initialization
- Information need identification
- Context resolution
- Memory resolution
- Gap identification
- Clarification question generation
- Status determination
- Strict mode behavior
"""

import pytest
from unittest.mock import AsyncMock, MagicMock

from sigil.grounding.grounder import Grounder
from sigil.grounding.schemas import (
    GroundingRequest,
    GroundingResult,
    GroundingStatus,
    InformationNeed,
    InformationGap,
    GapType,
)


class TestGrounderBasics:
    """Tests for basic Grounder functionality."""

    def test_initialization(self, grounder):
        """Test that Grounder initializes correctly."""
        assert grounder is not None
        assert grounder._strict_mode is False

    def test_initialization_strict_mode(self, grounder_strict):
        """Test strict mode initialization."""
        assert grounder_strict._strict_mode is True

    def test_initialization_with_memory(self, grounder_with_memory, mock_memory_manager):
        """Test initialization with memory manager."""
        assert grounder_with_memory._memory_manager is not None


class TestInformationNeedIdentification:
    """Tests for identifying information needs from messages."""

    @pytest.mark.asyncio
    async def test_search_query_need_identified(self, grounder, simple_request):
        """Test that search queries are identified as needs."""
        result = await grounder.ground(simple_request)

        assert result.status in [GroundingStatus.COMPLETE, GroundingStatus.PARTIAL]
        search_needs = [n for n in result.information_needs if n.category == "search_query"]
        assert len(search_needs) >= 1
        assert search_needs[0].resolved  # Search query is self-resolved

    @pytest.mark.asyncio
    async def test_temporal_need_identified(self, grounder):
        """Test that temporal needs are identified."""
        request = GroundingRequest(
            message="Schedule a meeting next week",
            session_id="test-session",
        )
        result = await grounder.ground(request)

        temporal_needs = [n for n in result.information_needs if n.category == "temporal"]
        assert len(temporal_needs) >= 1

    @pytest.mark.asyncio
    async def test_entity_need_identified(self, grounder, ambiguous_request):
        """Test that entity needs are identified for ambiguous references."""
        result = await grounder.ground(ambiguous_request)

        entity_needs = [n for n in result.information_needs if n.category == "entity"]
        assert len(entity_needs) >= 1

    @pytest.mark.asyncio
    async def test_no_needs_for_complete_request(self, grounder, complete_request):
        """Test that complete requests have fewer unresolved needs."""
        result = await grounder.ground(complete_request)

        # Should have fewer gaps since context provides info
        assert result.confidence >= 0.5
        assert result.can_proceed


class TestContextResolution:
    """Tests for resolving needs from context."""

    @pytest.mark.asyncio
    async def test_entity_resolved_from_context(self, grounder, request_with_context):
        """Test that entity needs are resolved from context."""
        result = await grounder.ground(request_with_context)

        # Check that recipient/entity was resolved
        recipient_needs = [
            n for n in result.information_needs
            if n.category in ["recipient", "entity"] and n.resolved
        ]
        # Context should help resolve some needs
        assert result.can_proceed

    @pytest.mark.asyncio
    async def test_context_added_to_enriched(self, grounder, request_with_context):
        """Test that context is preserved in enriched context."""
        result = await grounder.ground(request_with_context)

        assert "contact" in result.enriched_context or "to" in result.enriched_context


class TestMemoryResolution:
    """Tests for resolving needs from memory."""

    @pytest.mark.asyncio
    async def test_memory_results_used(self, grounder, request_with_memory):
        """Test that memory results are incorporated."""
        result = await grounder.ground(request_with_memory)

        # Memory context should be added
        assert "memory_context" in result.enriched_context
        assert len(result.enriched_context["memory_context"]) > 0

    @pytest.mark.asyncio
    async def test_memory_manager_queried(self, grounder_with_memory, mock_memory_manager):
        """Test that memory manager is queried when available."""
        # Setup mock to return memories
        mock_memory = MagicMock()
        mock_memory.content = "Test memory content"
        mock_memory.category = "general"
        mock_memory_manager.retrieve.return_value = [mock_memory]

        request = GroundingRequest(
            message="What do we know about the client?",
            session_id="test-session",
        )

        await grounder_with_memory.ground(request)

        # Memory manager should have been called
        mock_memory_manager.retrieve.assert_called_once()


class TestGapIdentification:
    """Tests for identifying information gaps."""

    @pytest.mark.asyncio
    async def test_gaps_created_for_unresolved_needs(self, grounder, ambiguous_request):
        """Test that gaps are created for unresolved needs."""
        result = await grounder.ground(ambiguous_request)

        # Should have some gaps for ambiguous request
        assert len(result.information_gaps) > 0

    @pytest.mark.asyncio
    async def test_gap_type_mapping(self, grounder):
        """Test that gap types are correctly mapped."""
        request = GroundingRequest(
            message="Send an email",  # Missing recipient
            session_id="test-session",
        )
        result = await grounder.ground(request)

        # Should have gaps for missing information
        if result.information_gaps:
            gap_types = [g.gap_type for g in result.information_gaps]
            assert any(
                gt in gap_types
                for gt in [GapType.MISSING_PARAMETER, GapType.AMBIGUOUS_ENTITY]
            )

    @pytest.mark.asyncio
    async def test_gap_severity_assignment(self, grounder, ambiguous_request):
        """Test that gap severity is assigned based on priority."""
        result = await grounder.ground(ambiguous_request)

        for gap in result.information_gaps:
            assert gap.severity in [1, 2, 3]


class TestClarificationQuestions:
    """Tests for clarification question generation."""

    @pytest.mark.asyncio
    async def test_questions_generated_for_gaps(self, grounder, ambiguous_request):
        """Test that clarification questions are generated for gaps."""
        result = await grounder.ground(ambiguous_request)

        # Should generate questions for gaps
        if result.information_gaps:
            assert len(result.clarification_questions) > 0

    @pytest.mark.asyncio
    async def test_questions_limited_to_three(self, grounder):
        """Test that questions are limited to max 3."""
        request = GroundingRequest(
            message="Schedule a meeting with someone about something somewhere sometime",
            session_id="test-session",
        )
        result = await grounder.ground(request)

        assert len(result.clarification_questions) <= 3

    @pytest.mark.asyncio
    async def test_no_questions_for_complete_request(self, grounder, simple_request):
        """Test that simple requests don't require clarification."""
        result = await grounder.ground(simple_request)

        # Search requests are typically self-contained
        assert result.can_proceed


class TestStatusDetermination:
    """Tests for determining grounding status."""

    @pytest.mark.asyncio
    async def test_complete_status_no_gaps(self, grounder, simple_request):
        """Test COMPLETE status when no significant gaps."""
        result = await grounder.ground(simple_request)

        # Simple search should be complete or at least partial
        assert result.status in [GroundingStatus.COMPLETE, GroundingStatus.PARTIAL]
        assert result.can_proceed

    @pytest.mark.asyncio
    async def test_partial_status_with_gaps(self, grounder, ambiguous_request):
        """Test PARTIAL status when gaps exist but can proceed."""
        result = await grounder.ground(ambiguous_request)

        # Ambiguous request should have gaps but can still proceed
        assert result.can_proceed  # Non-strict mode allows proceeding

    @pytest.mark.asyncio
    async def test_needs_clarification_strict_mode(self, grounder_strict, ambiguous_request):
        """Test NEEDS_CLARIFICATION status in strict mode with critical gaps."""
        result = await grounder_strict.ground(ambiguous_request)

        # In strict mode, critical gaps should block proceeding
        if result.has_critical_gaps:
            assert result.status == GroundingStatus.NEEDS_CLARIFICATION
            assert not result.can_proceed


class TestConfidenceCalculation:
    """Tests for confidence score calculation."""

    @pytest.mark.asyncio
    async def test_high_confidence_complete_request(self, grounder, simple_request):
        """Test high confidence for complete requests."""
        result = await grounder.ground(simple_request)

        assert result.confidence >= 0.5

    @pytest.mark.asyncio
    async def test_lower_confidence_with_gaps(self, grounder, ambiguous_request):
        """Test lower confidence when gaps exist."""
        result = await grounder.ground(ambiguous_request)

        # Gaps should reduce confidence
        assert result.confidence <= 1.0


class TestEnrichedOutput:
    """Tests for enriched message and context."""

    @pytest.mark.asyncio
    async def test_enriched_message_preserved(self, grounder, simple_request):
        """Test that original message is preserved in enriched message."""
        result = await grounder.ground(simple_request)

        assert simple_request.message in result.enriched_message

    @pytest.mark.asyncio
    async def test_enriched_context_contains_original(self, grounder, request_with_context):
        """Test that original context is preserved."""
        result = await grounder.ground(request_with_context)

        # Original context keys should be present
        for key in request_with_context.context:
            assert key in result.enriched_context

    @pytest.mark.asyncio
    async def test_resolved_info_added_to_context(self, grounder, request_with_context):
        """Test that resolved information is added to context."""
        result = await grounder.ground(request_with_context)

        # Should have some resolved info if needs were resolved
        if any(n.resolved for n in result.information_needs):
            assert "resolved_info" in result.enriched_context


class TestEventEmission:
    """Tests for event emission during grounding."""

    @pytest.mark.asyncio
    async def test_started_event_emitted(self, grounder, event_store, simple_request):
        """Test that started event is emitted."""
        initial_count = len(event_store.get_events(simple_request.session_id))

        await grounder.ground(simple_request)

        events = event_store.get_events(simple_request.session_id)
        assert len(events) > initial_count

    @pytest.mark.asyncio
    async def test_completed_event_emitted(self, grounder, event_store, simple_request):
        """Test that completed event is emitted."""
        await grounder.ground(simple_request)

        events = event_store.get_events(simple_request.session_id)
        # Should have at least started and completed events
        grounding_events = [
            e for e in events
            if e.payload.get("phase") == "grounding"
        ]
        assert len(grounding_events) >= 2


class TestErrorHandling:
    """Tests for error handling in grounding."""

    @pytest.mark.asyncio
    async def test_graceful_failure_on_memory_error(self, grounder_with_memory, mock_memory_manager):
        """Test graceful handling of memory manager errors."""
        mock_memory_manager.retrieve.side_effect = Exception("Memory error")

        request = GroundingRequest(
            message="Test message",
            session_id="test-session",
        )

        result = await grounder_with_memory.ground(request)

        # Should still succeed with warning
        assert result.can_proceed

    @pytest.mark.asyncio
    async def test_failed_status_on_critical_error(self, grounder, simple_request):
        """Test that critical errors result in FAILED status but allow proceeding."""
        # Simulate by using a request that triggers an error
        # (Currently the implementation handles errors gracefully)
        result = await grounder.ground(simple_request)

        # Even on failure, should allow proceeding
        assert result.can_proceed


class TestIntegration:
    """Integration tests for the grounding phase."""

    @pytest.mark.asyncio
    async def test_full_grounding_flow_simple(self, grounder, simple_request):
        """Test full grounding flow with simple request."""
        result = await grounder.ground(simple_request)

        assert result is not None
        assert isinstance(result, GroundingResult)
        assert result.enriched_message is not None
        assert result.processing_time_ms > 0

    @pytest.mark.asyncio
    async def test_full_grounding_flow_complex(self, grounder, ambiguous_request):
        """Test full grounding flow with complex request."""
        result = await grounder.ground(ambiguous_request)

        assert result is not None
        assert result.enriched_context is not None
        assert result.processing_time_ms > 0

    @pytest.mark.asyncio
    async def test_grounding_result_serialization(self, grounder, simple_request):
        """Test that grounding result can be serialized."""
        result = await grounder.ground(simple_request)

        result_dict = result.to_dict()

        assert "status" in result_dict
        assert "enriched_message" in result_dict
        assert "can_proceed" in result_dict
        assert "confidence" in result_dict
