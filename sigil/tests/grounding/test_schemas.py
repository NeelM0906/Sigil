"""Tests for grounding schemas.

Tests cover:
- Schema initialization
- Data validation
- Methods and properties
- Serialization
"""

import pytest

from sigil.grounding.schemas import (
    GapType,
    GroundingStatus,
    InformationNeed,
    InformationGap,
    GroundingRequest,
    GroundingResult,
)


class TestGapType:
    """Tests for GapType enum."""

    def test_all_gap_types_exist(self):
        """Test that all expected gap types exist."""
        assert GapType.AMBIGUOUS_ENTITY
        assert GapType.MISSING_PARAMETER
        assert GapType.UNCLEAR_INTENT
        assert GapType.MISSING_CONTEXT
        assert GapType.CONFLICTING_INFO
        assert GapType.INSUFFICIENT_DETAIL

    def test_gap_type_values(self):
        """Test gap type string values."""
        assert GapType.AMBIGUOUS_ENTITY.value == "ambiguous_entity"
        assert GapType.MISSING_PARAMETER.value == "missing_parameter"


class TestGroundingStatus:
    """Tests for GroundingStatus enum."""

    def test_all_statuses_exist(self):
        """Test that all expected statuses exist."""
        assert GroundingStatus.COMPLETE
        assert GroundingStatus.PARTIAL
        assert GroundingStatus.NEEDS_CLARIFICATION
        assert GroundingStatus.FAILED

    def test_status_values(self):
        """Test status string values."""
        assert GroundingStatus.COMPLETE.value == "complete"
        assert GroundingStatus.NEEDS_CLARIFICATION.value == "needs_clarification"


class TestInformationNeed:
    """Tests for InformationNeed dataclass."""

    def test_default_initialization(self):
        """Test default values."""
        need = InformationNeed()

        assert need.need_id is not None
        assert need.category == ""
        assert need.priority == 2
        assert need.resolved is False
        assert need.resolved_value is None

    def test_custom_initialization(self):
        """Test custom values."""
        need = InformationNeed(
            category="temporal",
            description="Meeting time",
            priority=1,
            source="user",
        )

        assert need.category == "temporal"
        assert need.description == "Meeting time"
        assert need.priority == 1
        assert need.source == "user"

    def test_resolve_method(self):
        """Test resolving an information need."""
        need = InformationNeed(
            category="entity",
            description="Contact person",
        )

        assert not need.resolved

        need.resolve("John Smith")

        assert need.resolved
        assert need.resolved_value == "John Smith"

    def test_unique_ids(self):
        """Test that each need gets a unique ID."""
        need1 = InformationNeed()
        need2 = InformationNeed()

        assert need1.need_id != need2.need_id


class TestInformationGap:
    """Tests for InformationGap dataclass."""

    def test_default_initialization(self):
        """Test default values."""
        gap = InformationGap()

        assert gap.gap_id is not None
        assert gap.gap_type == GapType.MISSING_PARAMETER
        assert gap.severity == 2
        assert gap.can_infer is False

    def test_custom_initialization(self):
        """Test custom values."""
        gap = InformationGap(
            gap_type=GapType.AMBIGUOUS_ENTITY,
            description="Who is 'the team'?",
            affected_element="entity",
            severity=1,
            suggestion="Who specifically should attend?",
        )

        assert gap.gap_type == GapType.AMBIGUOUS_ENTITY
        assert gap.severity == 1
        assert gap.suggestion is not None

    def test_unique_ids(self):
        """Test that each gap gets a unique ID."""
        gap1 = InformationGap()
        gap2 = InformationGap()

        assert gap1.gap_id != gap2.gap_id


class TestGroundingRequest:
    """Tests for GroundingRequest dataclass."""

    def test_required_fields(self):
        """Test required fields."""
        request = GroundingRequest(
            message="Test message",
            session_id="test-session",
        )

        assert request.message == "Test message"
        assert request.session_id == "test-session"

    def test_default_values(self):
        """Test default values."""
        request = GroundingRequest(
            message="Test",
            session_id="sess",
        )

        assert request.context == {}
        assert request.route_intent is None
        assert request.route_complexity == 0.5
        assert request.available_tools == []
        assert request.memory_results == []

    def test_full_initialization(self):
        """Test with all fields."""
        request = GroundingRequest(
            message="Schedule meeting",
            session_id="sess-123",
            context={"key": "value"},
            route_intent="run_agent",
            route_complexity=0.7,
            available_tools=["calendar.create"],
            memory_results=[{"content": "Memory", "category": "history"}],
        )

        assert request.route_intent == "run_agent"
        assert len(request.available_tools) == 1
        assert len(request.memory_results) == 1


class TestGroundingResult:
    """Tests for GroundingResult dataclass."""

    def test_minimal_initialization(self):
        """Test with minimal required fields."""
        result = GroundingResult(
            status=GroundingStatus.COMPLETE,
            enriched_message="Test message",
        )

        assert result.status == GroundingStatus.COMPLETE
        assert result.can_proceed is True
        assert result.confidence == 1.0

    def test_full_initialization(self):
        """Test with all fields."""
        needs = [InformationNeed(category="test")]
        gaps = [InformationGap(gap_type=GapType.MISSING_PARAMETER)]

        result = GroundingResult(
            status=GroundingStatus.PARTIAL,
            enriched_message="Test message",
            enriched_context={"key": "value"},
            information_needs=needs,
            information_gaps=gaps,
            clarification_questions=["What time?"],
            confidence=0.8,
            can_proceed=True,
            tokens_used=100,
            processing_time_ms=50.5,
        )

        assert result.status == GroundingStatus.PARTIAL
        assert len(result.information_needs) == 1
        assert len(result.information_gaps) == 1
        assert len(result.clarification_questions) == 1

    def test_has_critical_gaps_property(self):
        """Test has_critical_gaps property."""
        # No gaps
        result1 = GroundingResult(
            status=GroundingStatus.COMPLETE,
            enriched_message="Test",
        )
        assert not result1.has_critical_gaps

        # Non-critical gap
        result2 = GroundingResult(
            status=GroundingStatus.PARTIAL,
            enriched_message="Test",
            information_gaps=[InformationGap(severity=2)],
        )
        assert not result2.has_critical_gaps

        # Critical gap
        result3 = GroundingResult(
            status=GroundingStatus.NEEDS_CLARIFICATION,
            enriched_message="Test",
            information_gaps=[InformationGap(severity=1)],
        )
        assert result3.has_critical_gaps

    def test_unresolved_needs_count_property(self):
        """Test unresolved_needs_count property."""
        resolved = InformationNeed()
        resolved.resolve("value")
        unresolved = InformationNeed()

        result = GroundingResult(
            status=GroundingStatus.PARTIAL,
            enriched_message="Test",
            information_needs=[resolved, unresolved],
        )

        assert result.unresolved_needs_count == 1

    def test_to_dict_method(self):
        """Test serialization to dict."""
        result = GroundingResult(
            status=GroundingStatus.COMPLETE,
            enriched_message="Test message",
            enriched_context={"key": "value"},
            confidence=0.9,
            can_proceed=True,
            tokens_used=50,
            processing_time_ms=25.0,
        )

        result_dict = result.to_dict()

        assert result_dict["status"] == "complete"
        assert result_dict["enriched_message"] == "Test message"
        assert result_dict["enriched_context"] == {"key": "value"}
        assert result_dict["confidence"] == 0.9
        assert result_dict["can_proceed"] is True
        assert result_dict["tokens_used"] == 50
        assert result_dict["processing_time_ms"] == 25.0

    def test_to_dict_with_needs_and_gaps(self):
        """Test serialization includes needs and gaps."""
        result = GroundingResult(
            status=GroundingStatus.PARTIAL,
            enriched_message="Test",
            information_needs=[
                InformationNeed(
                    category="temporal",
                    description="Time needed",
                )
            ],
            information_gaps=[
                InformationGap(
                    gap_type=GapType.MISSING_PARAMETER,
                    description="Missing time",
                    severity=2,
                    suggestion="When?",
                )
            ],
        )

        result_dict = result.to_dict()

        assert len(result_dict["information_needs"]) == 1
        assert result_dict["information_needs"][0]["category"] == "temporal"

        assert len(result_dict["information_gaps"]) == 1
        assert result_dict["information_gaps"][0]["gap_type"] == "missing_parameter"
