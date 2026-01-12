"""Tests for fallback management."""

import pytest

from sigil.contracts.schema import (
    Contract,
    ContractConstraints,
    Deliverable,
    FailureStrategy,
)
from sigil.contracts.validator import ValidationResult, ValidationError
from sigil.contracts.fallback import (
    FallbackManager,
    FallbackResult,
    FallbackStrategy,
)


class TestFallbackManager:
    """Tests for FallbackManager class."""

    def test_can_build_partial_with_valid_output(
        self, fallback_manager, simple_contract, valid_output
    ):
        """Test can_build_partial with valid output."""
        result = fallback_manager.can_build_partial(valid_output, simple_contract)
        assert result is True

    def test_can_build_partial_with_none_output(
        self, fallback_manager, simple_contract
    ):
        """Test can_build_partial with None output."""
        result = fallback_manager.can_build_partial(None, simple_contract)
        assert result is False

    def test_can_build_partial_with_partial_output(
        self, fallback_manager, simple_contract, partial_output
    ):
        """Test can_build_partial with partial output."""
        # partial_output has score but invalid recommendation
        result = fallback_manager.can_build_partial(partial_output, simple_contract)
        # With 1/2 required fields valid (50%), should meet 0.5 threshold
        assert result is True

    def test_can_build_partial_insufficient_coverage(self, fallback_manager):
        """Test can_build_partial when coverage is too low."""
        manager = FallbackManager(min_partial_coverage=0.8)
        contract = Contract(
            name="test",
            description="Test",
            deliverables=[
                Deliverable(name="a", type="int", description="A", required=True),
                Deliverable(name="b", type="int", description="B", required=True),
                Deliverable(name="c", type="int", description="C", required=True),
            ],
        )
        # Only 1 of 3 required fields = 33% coverage
        output = {"a": 1}
        result = manager.can_build_partial(output, contract)
        assert result is False


class TestBuildPartialResult:
    """Tests for building partial results."""

    def test_build_partial_result_basic(
        self, fallback_manager, simple_contract, valid_output
    ):
        """Test building partial result from valid output."""
        result = fallback_manager.build_partial_result(valid_output, simple_contract)

        assert result.strategy == FallbackStrategy.PARTIAL
        assert result.output["score"] == 75
        assert result.output["recommendation"] == "schedule_demo"
        assert len(result.partial_fields) == 2
        assert len(result.template_fields) == 0

    def test_build_partial_result_with_missing_field(
        self, fallback_manager, simple_contract
    ):
        """Test building partial result with missing field."""
        output = {"score": 75}  # missing recommendation
        result = fallback_manager.build_partial_result(output, simple_contract)

        assert result.strategy == FallbackStrategy.PARTIAL
        assert result.output["score"] == 75
        # recommendation should come from template
        assert "recommendation" in result.output
        assert "score" in result.partial_fields
        assert "recommendation" in result.template_fields

    def test_build_partial_result_includes_warnings(
        self, fallback_manager, simple_contract
    ):
        """Test that partial result includes warnings."""
        output = {"score": 75}  # missing recommendation
        result = fallback_manager.build_partial_result(output, simple_contract)

        assert len(result.warnings) > 0
        assert any("recommendation" in w.lower() for w in result.warnings)

    def test_build_partial_result_uses_validation_result(
        self, fallback_manager, simple_contract, valid_output
    ):
        """Test that partial result uses validation result's partial_output."""
        validation_result = ValidationResult(
            is_valid=False,
            partial_output={"score": 80},  # Different from valid_output
        )
        result = fallback_manager.build_partial_result(
            valid_output, simple_contract, validation_result
        )

        # Should use validation_result's partial_output
        assert result.output["score"] == 80


class TestBuildTemplateResult:
    """Tests for building template results."""

    def test_build_template_result(self, fallback_manager, simple_contract):
        """Test building template-based result."""
        result = fallback_manager.build_template_result(simple_contract)

        assert result.strategy == FallbackStrategy.TEMPLATE
        assert "score" in result.output
        assert "recommendation" in result.output
        assert len(result.template_fields) == 2
        assert len(result.partial_fields) == 0

    def test_build_template_result_uses_examples(self, fallback_manager, simple_contract):
        """Test that template result uses example values."""
        result = fallback_manager.build_template_result(simple_contract)

        # score_deliverable has example=75
        assert result.output["score"] == 75
        # recommendation_deliverable has example="schedule_demo"
        assert result.output["recommendation"] == "schedule_demo"

    def test_build_template_result_includes_warnings(
        self, fallback_manager, simple_contract
    ):
        """Test that template result includes appropriate warnings."""
        result = fallback_manager.build_template_result(simple_contract)

        assert len(result.warnings) > 0
        assert any("template" in w.lower() for w in result.warnings)
        assert any("review" in w.lower() for w in result.warnings)


class TestSelectStrategy:
    """Tests for strategy selection."""

    def test_select_strategy_partial_preferred(
        self, fallback_manager, simple_contract, partial_output
    ):
        """Test that partial strategy is selected when viable."""
        result = fallback_manager.select_strategy(partial_output, simple_contract)
        assert result.strategy == FallbackStrategy.PARTIAL

    def test_select_strategy_template_fallback(self, fallback_manager, simple_contract):
        """Test that template strategy is used when partial not viable."""
        output = {}  # Empty output
        result = fallback_manager.select_strategy(output, simple_contract)
        assert result.strategy == FallbackStrategy.TEMPLATE

    def test_select_strategy_escalate_for_fail_strategy(
        self, fallback_manager, strict_contract, valid_output
    ):
        """Test that escalate is returned for fail strategy."""
        result = fallback_manager.select_strategy(valid_output, strict_contract)
        assert result.strategy == FallbackStrategy.ESCALATE

    def test_select_strategy_none_output(self, fallback_manager, simple_contract):
        """Test strategy selection with None output."""
        result = fallback_manager.select_strategy(None, simple_contract)
        assert result.strategy == FallbackStrategy.TEMPLATE


class TestMergeWithTemplate:
    """Tests for merge_with_template helper."""

    def test_merge_fills_missing_fields(self, fallback_manager, simple_contract):
        """Test that merge fills missing fields with template values."""
        output = {"score": 90}
        merged = fallback_manager.merge_with_template(output, simple_contract)

        assert merged["score"] == 90  # Original value preserved
        assert "recommendation" in merged  # Template value added

    def test_merge_preserves_existing_values(self, fallback_manager, simple_contract):
        """Test that merge preserves existing values."""
        output = {"score": 90, "recommendation": "custom_action"}
        merged = fallback_manager.merge_with_template(output, simple_contract)

        assert merged["score"] == 90
        assert merged["recommendation"] == "custom_action"


class TestFallbackResult:
    """Tests for FallbackResult dataclass."""

    def test_fallback_result_to_dict(self):
        """Test serialization to dictionary."""
        result = FallbackResult(
            output={"score": 75},
            strategy=FallbackStrategy.PARTIAL,
            warnings=["Missing field filled from template"],
            missing_fields=["recommendation"],
            partial_fields=["score"],
            template_fields=["recommendation"],
        )
        data = result.to_dict()

        assert data["output"] == {"score": 75}
        assert data["strategy"] == "partial"
        assert len(data["warnings"]) == 1
        assert "recommendation" in data["missing_fields"]


class TestFallbackStrategy:
    """Tests for FallbackStrategy enum."""

    def test_strategy_values(self):
        """Test fallback strategy enum values."""
        assert FallbackStrategy.PARTIAL.value == "partial"
        assert FallbackStrategy.TEMPLATE.value == "template"
        assert FallbackStrategy.ESCALATE.value == "escalate"
        assert FallbackStrategy.NONE.value == "none"


class TestStrategyDescription:
    """Tests for strategy description."""

    def test_get_strategy_description_partial(self, fallback_manager):
        """Test description for partial strategy."""
        desc = fallback_manager.get_strategy_description(FallbackStrategy.PARTIAL)
        assert "partial" in desc.lower()
        assert "valid fields" in desc.lower()

    def test_get_strategy_description_template(self, fallback_manager):
        """Test description for template strategy."""
        desc = fallback_manager.get_strategy_description(FallbackStrategy.TEMPLATE)
        assert "template" in desc.lower()

    def test_get_strategy_description_escalate(self, fallback_manager):
        """Test description for escalate strategy."""
        desc = fallback_manager.get_strategy_description(FallbackStrategy.ESCALATE)
        assert "escalat" in desc.lower() or "exception" in desc.lower()


class TestMeaningfulValue:
    """Tests for meaningful value checking."""

    def test_meaningful_value_none(self, fallback_manager):
        """Test that None is not meaningful."""
        assert fallback_manager._is_meaningful_value(None) is False

    def test_meaningful_value_empty_string(self, fallback_manager):
        """Test that empty string is not meaningful."""
        assert fallback_manager._is_meaningful_value("") is False
        assert fallback_manager._is_meaningful_value("   ") is False

    def test_meaningful_value_empty_list(self, fallback_manager):
        """Test that empty list is not meaningful."""
        assert fallback_manager._is_meaningful_value([]) is False

    def test_meaningful_value_empty_dict(self, fallback_manager):
        """Test that empty dict is not meaningful."""
        assert fallback_manager._is_meaningful_value({}) is False

    def test_meaningful_value_valid_values(self, fallback_manager):
        """Test that valid values are meaningful."""
        assert fallback_manager._is_meaningful_value("hello") is True
        assert fallback_manager._is_meaningful_value(0) is True
        assert fallback_manager._is_meaningful_value(False) is True
        assert fallback_manager._is_meaningful_value([1, 2]) is True
        assert fallback_manager._is_meaningful_value({"key": "value"}) is True
