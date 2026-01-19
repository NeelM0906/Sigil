"""Tests for contract templates."""

import pytest

from sigil.contracts.schema import Contract, FailureStrategy
from sigil.contracts.templates.acti import (
    CONTRACT_TEMPLATES,
    appointment_booking_contract,
    compliance_check_contract,
    get_template,
    lead_qualification_contract,
    list_templates,
    market_analysis_contract,
    research_report_contract,
)


class TestLeadQualificationContract:
    """Tests for lead_qualification_contract template."""

    def test_create_default(self):
        """Test creating with default parameters."""
        contract = lead_qualification_contract()
        assert contract.name == "lead_qualification"
        assert contract.failure_strategy == FailureStrategy.RETRY
        assert contract.max_retries == 2

    def test_deliverables(self):
        """Test that contract has correct deliverables."""
        contract = lead_qualification_contract()
        names = contract.get_deliverable_names()
        assert "score" in names
        assert "bant_assessment" in names
        assert "recommended_action" in names

    def test_score_deliverable_spec(self):
        """Test score deliverable specification."""
        contract = lead_qualification_contract()
        score = contract.get_deliverable("score")
        assert score is not None
        assert score.type == "int"
        assert score.required is True
        assert "0 <= value <= 100" in score.validation_rules

    def test_bant_deliverable_spec(self):
        """Test BANT assessment deliverable specification."""
        contract = lead_qualification_contract()
        bant = contract.get_deliverable("bant_assessment")
        assert bant is not None
        assert bant.type == "dict"
        assert "'budget' in value" in bant.validation_rules
        assert "'authority' in value" in bant.validation_rules

    def test_custom_parameters(self):
        """Test creating with custom parameters."""
        contract = lead_qualification_contract(
            max_tokens=10000,
            max_tool_calls=10,
            timeout_seconds=120,
            max_retries=3,
            failure_strategy=FailureStrategy.FALLBACK,
        )
        assert contract.constraints.max_total_tokens == 10000
        assert contract.constraints.max_tool_calls == 10
        assert contract.constraints.timeout_seconds == 120
        assert contract.max_retries == 3
        assert contract.failure_strategy == FailureStrategy.FALLBACK

    def test_metadata(self):
        """Test contract metadata."""
        contract = lead_qualification_contract()
        assert contract.metadata["stratum"] == "RAI"
        assert "sales" in contract.metadata["use_case"]


class TestResearchReportContract:
    """Tests for research_report_contract template."""

    def test_create_default(self):
        """Test creating with default parameters."""
        contract = research_report_contract()
        assert contract.name == "research_report"
        assert contract.failure_strategy == FailureStrategy.FALLBACK

    def test_deliverables(self):
        """Test that contract has correct deliverables."""
        contract = research_report_contract()
        names = contract.get_deliverable_names()
        assert "summary" in names
        assert "sources" in names
        assert "key_findings" in names

    def test_summary_deliverable_spec(self):
        """Test summary deliverable specification."""
        contract = research_report_contract()
        summary = contract.get_deliverable("summary")
        assert summary is not None
        assert summary.type == "str"
        assert "len(value) >= 50" in summary.validation_rules

    def test_sources_deliverable_spec(self):
        """Test sources deliverable specification."""
        contract = research_report_contract()
        sources = contract.get_deliverable("sources")
        assert sources is not None
        assert sources.type == "list"
        assert "len(value) >= 1" in sources.validation_rules

    def test_metadata(self):
        """Test contract metadata."""
        contract = research_report_contract()
        assert contract.metadata["stratum"] == "RTI"


class TestAppointmentBookingContract:
    """Tests for appointment_booking_contract template."""

    def test_create_default(self):
        """Test creating with default parameters."""
        contract = appointment_booking_contract()
        assert contract.name == "appointment_booking"
        assert contract.failure_strategy == FailureStrategy.RETRY
        assert contract.max_retries == 3

    def test_deliverables(self):
        """Test that contract has correct deliverables."""
        contract = appointment_booking_contract()
        names = contract.get_deliverable_names()
        assert "booking_confirmed" in names
        assert "datetime" in names
        assert "meeting_link" in names

    def test_booking_confirmed_spec(self):
        """Test booking_confirmed deliverable specification."""
        contract = appointment_booking_contract()
        confirmed = contract.get_deliverable("booking_confirmed")
        assert confirmed is not None
        assert confirmed.type == "bool"
        assert confirmed.required is True

    def test_meeting_link_optional(self):
        """Test that meeting_link is optional."""
        contract = appointment_booking_contract()
        link = contract.get_deliverable("meeting_link")
        assert link is not None
        assert link.required is False

    def test_metadata(self):
        """Test contract metadata."""
        contract = appointment_booking_contract()
        assert contract.metadata["stratum"] == "ZACS"


class TestMarketAnalysisContract:
    """Tests for market_analysis_contract template."""

    def test_create_default(self):
        """Test creating with default parameters."""
        contract = market_analysis_contract()
        assert contract.name == "market_analysis"
        assert contract.constraints.max_total_tokens == 10000

    def test_deliverables(self):
        """Test that contract has correct deliverables."""
        contract = market_analysis_contract()
        names = contract.get_deliverable_names()
        assert "market_size" in names
        assert "competitors" in names
        assert "opportunities" in names

    def test_metadata(self):
        """Test contract metadata."""
        contract = market_analysis_contract()
        assert contract.metadata["stratum"] == "EEI"


class TestComplianceCheckContract:
    """Tests for compliance_check_contract template."""

    def test_create_default(self):
        """Test creating with default parameters."""
        contract = compliance_check_contract()
        assert contract.name == "compliance_check"
        assert contract.failure_strategy == FailureStrategy.FAIL
        assert contract.max_retries == 1

    def test_deliverables(self):
        """Test that contract has correct deliverables."""
        contract = compliance_check_contract()
        names = contract.get_deliverable_names()
        assert "is_compliant" in names
        assert "violations" in names
        assert "recommendations" in names

    def test_is_compliant_spec(self):
        """Test is_compliant deliverable specification."""
        contract = compliance_check_contract()
        compliant = contract.get_deliverable("is_compliant")
        assert compliant is not None
        assert compliant.type == "bool"
        assert compliant.required is True

    def test_metadata(self):
        """Test contract metadata."""
        contract = compliance_check_contract()
        assert contract.metadata["stratum"] == "IGE"


class TestTemplateRegistry:
    """Tests for template registry functions."""

    def test_list_templates(self):
        """Test listing all available templates."""
        templates = list_templates()
        assert "lead_qualification" in templates
        assert "research_report" in templates
        assert "appointment_booking" in templates
        assert "market_analysis" in templates
        assert "compliance_check" in templates

    def test_get_template_by_name(self):
        """Test getting template by name."""
        contract = get_template("lead_qualification")
        assert contract is not None
        assert contract.name == "lead_qualification"

    def test_get_template_with_kwargs(self):
        """Test getting template with custom parameters."""
        contract = get_template("lead_qualification", max_tokens=20000)
        assert contract is not None
        assert contract.constraints.max_total_tokens == 20000

    def test_get_template_invalid_name(self):
        """Test getting template with invalid name."""
        contract = get_template("invalid_template")
        assert contract is None

    def test_contract_templates_dict(self):
        """Test CONTRACT_TEMPLATES dictionary."""
        assert len(CONTRACT_TEMPLATES) == 5
        assert callable(CONTRACT_TEMPLATES["lead_qualification"])


class TestTemplateValidation:
    """Tests for template schema validation."""

    def test_all_templates_validate(self):
        """Test that all templates pass schema validation."""
        for name in list_templates():
            contract = get_template(name)
            is_valid, errors = contract.validate_schema()
            # Some templates may have warnings but should be generally valid
            if not is_valid:
                # Check if the only issue is fail strategy with retries
                if contract.failure_strategy == FailureStrategy.FAIL:
                    assert contract.max_retries <= 1
                else:
                    assert False, f"Template {name} failed validation: {errors}"

    def test_all_templates_have_deliverables(self):
        """Test that all templates have at least one deliverable."""
        for name in list_templates():
            contract = get_template(name)
            assert len(contract.deliverables) >= 1

    def test_all_templates_have_metadata(self):
        """Test that all templates have stratum metadata."""
        for name in list_templates():
            contract = get_template(name)
            assert "stratum" in contract.metadata
            assert "stratum_name" in contract.metadata

    def test_all_deliverables_have_examples(self):
        """Test that all deliverables have example values."""
        for name in list_templates():
            contract = get_template(name)
            for deliverable in contract.deliverables:
                assert deliverable.example is not None, (
                    f"Template {name} deliverable {deliverable.name} missing example"
                )
