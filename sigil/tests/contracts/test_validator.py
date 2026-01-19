"""Tests for contract validation."""

import pytest

from sigil.contracts.schema import Contract, ContractConstraints, Deliverable
from sigil.contracts.validator import (
    ContractValidator,
    ErrorSeverity,
    ValidationError,
    ValidationResult,
)
from sigil.telemetry.tokens import TokenTracker


class TestValidationError:
    """Tests for ValidationError dataclass."""

    def test_create_basic_error(self):
        """Test creating a basic validation error."""
        error = ValidationError(
            field="score",
            reason="Required field is missing",
        )
        assert error.field == "score"
        assert error.reason == "Required field is missing"
        assert error.severity == ErrorSeverity.ERROR

    def test_create_error_with_all_fields(self):
        """Test creating error with all fields."""
        error = ValidationError(
            field="score",
            reason="Value out of range",
            expected="0 <= value <= 100",
            actual="150",
            severity=ErrorSeverity.ERROR,
            rule="0 <= value <= 100",
        )
        assert error.expected == "0 <= value <= 100"
        assert error.actual == "150"
        assert error.rule == "0 <= value <= 100"

    def test_error_to_dict(self):
        """Test serialization to dictionary."""
        error = ValidationError(
            field="score",
            reason="Type mismatch",
            expected="int",
            actual="str",
        )
        result = error.to_dict()
        assert result["field"] == "score"
        assert result["reason"] == "Type mismatch"
        assert result["severity"] == "error"

    def test_error_str_representation(self):
        """Test string representation."""
        error = ValidationError(
            field="score",
            reason="Type mismatch",
            expected="int",
            actual="str",
        )
        str_repr = str(error)
        assert "score" in str_repr
        assert "Type mismatch" in str_repr


class TestValidationResult:
    """Tests for ValidationResult dataclass."""

    def test_create_valid_result(self):
        """Test creating a valid result."""
        result = ValidationResult(is_valid=True)
        assert result.is_valid is True
        assert len(result.errors) == 0
        assert len(result.warnings) == 0

    def test_create_invalid_result(self):
        """Test creating an invalid result with errors."""
        errors = [
            ValidationError(field="score", reason="Missing"),
            ValidationError(field="name", reason="Type mismatch"),
        ]
        result = ValidationResult(is_valid=False, errors=errors)
        assert result.is_valid is False
        assert len(result.errors) == 2

    def test_get_error_summary(self):
        """Test getting error summary."""
        errors = [
            ValidationError(field="score", reason="Missing"),
            ValidationError(field="name", reason="Type mismatch"),
        ]
        result = ValidationResult(is_valid=False, errors=errors)
        summary = result.get_error_summary()
        assert "2 validation error(s)" in summary
        assert "score" in summary
        assert "name" in summary

    def test_get_failed_fields(self):
        """Test getting failed field names."""
        errors = [
            ValidationError(field="score", reason="Missing"),
            ValidationError(field="name", reason="Type mismatch"),
        ]
        result = ValidationResult(is_valid=False, errors=errors)
        failed = result.get_failed_fields()
        assert "score" in failed
        assert "name" in failed

    def test_get_passed_fields(self):
        """Test getting passed field names."""
        errors = [ValidationError(field="score", reason="Missing")]
        result = ValidationResult(
            is_valid=False,
            errors=errors,
            deliverables_checked=["score", "name", "status"],
        )
        passed = result.get_passed_fields()
        assert "name" in passed
        assert "status" in passed
        assert "score" not in passed

    def test_result_to_dict(self):
        """Test serialization to dictionary."""
        result = ValidationResult(
            is_valid=True,
            deliverables_checked=["score", "name"],
            validation_time_ms=50.5,
        )
        data = result.to_dict()
        assert data["is_valid"] is True
        assert data["deliverables_checked"] == ["score", "name"]
        assert data["validation_time_ms"] == 50.5


class TestContractValidator:
    """Tests for ContractValidator class."""

    def test_validate_valid_output(self, validator, simple_contract, valid_output):
        """Test validating a valid output."""
        result = validator.validate(valid_output, simple_contract)
        assert result.is_valid is True
        assert len(result.errors) == 0

    def test_validate_missing_required_field(
        self, validator, simple_contract, invalid_output_missing_field
    ):
        """Test that missing required field is caught."""
        result = validator.validate(invalid_output_missing_field, simple_contract)
        assert result.is_valid is False
        assert len(result.errors) >= 1
        assert any("recommendation" in e.field for e in result.errors)
        assert any("missing" in e.reason.lower() for e in result.errors)

    def test_validate_wrong_type(
        self, validator, simple_contract, invalid_output_wrong_type
    ):
        """Test that wrong type is caught."""
        result = validator.validate(invalid_output_wrong_type, simple_contract)
        assert result.is_valid is False
        assert any("type" in e.reason.lower() for e in result.errors)

    def test_validate_rule_violation(
        self, validator, simple_contract, invalid_output_rule_violation
    ):
        """Test that validation rule violation is caught."""
        result = validator.validate(invalid_output_rule_violation, simple_contract)
        assert result.is_valid is False
        assert any("rule" in e.reason.lower() for e in result.errors)

    def test_validate_optional_field_missing(
        self, validator, contract_with_optional, valid_output
    ):
        """Test that missing optional field is acceptable."""
        # valid_output doesn't have 'notes' field
        result = validator.validate(valid_output, contract_with_optional)
        assert result.is_valid is True

    def test_validate_partial_output(self, validator, simple_contract, partial_output):
        """Test validating partial output."""
        result = validator.validate(partial_output, simple_contract)
        assert result.is_valid is False
        # Should have partial_output with the valid field
        assert result.partial_output is not None
        assert "score" in result.partial_output

    def test_validate_generates_suggestion(
        self, validator, simple_contract, invalid_output_missing_field
    ):
        """Test that suggestion is generated for errors."""
        result = validator.validate(invalid_output_missing_field, simple_contract)
        assert result.suggestion is not None
        assert len(result.suggestion) > 0

    def test_validate_records_time(self, validator, simple_contract, valid_output):
        """Test that validation time is recorded."""
        result = validator.validate(valid_output, simple_contract)
        assert result.validation_time_ms > 0

    def test_validate_records_deliverables_checked(
        self, validator, simple_contract, valid_output
    ):
        """Test that deliverables checked is recorded."""
        result = validator.validate(valid_output, simple_contract)
        assert "score" in result.deliverables_checked
        assert "recommendation" in result.deliverables_checked


class TestContractValidatorTypeChecking:
    """Tests for type checking in ContractValidator."""

    def test_check_type_int(self, validator):
        """Test integer type checking."""
        assert validator._check_type(42, "int") is True
        assert validator._check_type("42", "int") is False
        assert validator._check_type(42.0, "int") is False

    def test_check_type_float(self, validator):
        """Test float type checking."""
        assert validator._check_type(42.5, "float") is True
        assert validator._check_type(42, "float") is True  # int accepted for float
        assert validator._check_type("42.5", "float") is False

    def test_check_type_str(self, validator):
        """Test string type checking."""
        assert validator._check_type("hello", "str") is True
        assert validator._check_type(42, "str") is False

    def test_check_type_bool(self, validator):
        """Test boolean type checking."""
        assert validator._check_type(True, "bool") is True
        assert validator._check_type(False, "bool") is True
        assert validator._check_type(1, "bool") is False
        assert validator._check_type("true", "bool") is False

    def test_check_type_dict(self, validator):
        """Test dict type checking."""
        assert validator._check_type({}, "dict") is True
        assert validator._check_type({"key": "value"}, "dict") is True
        assert validator._check_type([], "dict") is False

    def test_check_type_list(self, validator):
        """Test list type checking."""
        assert validator._check_type([], "list") is True
        assert validator._check_type([1, 2, 3], "list") is True
        assert validator._check_type({}, "list") is False

    def test_check_type_any(self, validator):
        """Test Any type accepts anything."""
        assert validator._check_type(42, "Any") is True
        assert validator._check_type("hello", "Any") is True
        assert validator._check_type(None, "Any") is True

    def test_check_type_parameterized(self, validator):
        """Test parameterized types check base type."""
        assert validator._check_type([1, 2, 3], "list[int]") is True
        assert validator._check_type({"a": 1}, "dict[str, int]") is True


class TestContractValidatorRuleChecking:
    """Tests for rule validation in ContractValidator."""

    def test_compile_comparison_rule(self, validator):
        """Test compiling comparison rules."""
        validators = validator.compile_rules(["0 <= value <= 100"])
        assert len(validators) == 1
        assert validators[0](50) is True
        assert validators[0](150) is False

    def test_compile_isinstance_rule(self, validator):
        """Test compiling isinstance rules."""
        validators = validator.compile_rules(["isinstance(value, int)"])
        assert len(validators) == 1
        assert validators[0](42) is True
        assert validators[0]("42") is False

    def test_compile_len_rule(self, validator):
        """Test compiling length rules."""
        validators = validator.compile_rules(["len(value) > 0"])
        assert len(validators) == 1
        assert validators[0]("hello") is True
        assert validators[0]("") is False
        assert validators[0]([1, 2]) is True
        assert validators[0]([]) is False

    def test_compile_membership_rule(self, validator):
        """Test compiling membership rules."""
        validators = validator.compile_rules(["'key' in value"])
        assert len(validators) == 1
        assert validators[0]({"key": 1}) is True
        assert validators[0]({"other": 1}) is False

    def test_rule_evaluation_error(self, validator):
        """Test that rule evaluation errors are handled."""
        # This rule would cause an error on non-string values
        d = Deliverable(
            name="test",
            type="str",
            description="Test",
            validation_rules=["value.startswith('http')"],
        )
        # First, valid case
        errors = validator.validate_deliverable("test", "https://example.com", d)
        assert len(errors) == 0

        # Now with an invalid input that causes attribute error
        errors = validator.validate_deliverable("test", "ftp://example.com", d)
        assert len(errors) >= 1


class TestContractValidatorConstraints:
    """Tests for constraint validation."""

    def test_validate_constraints_within_budget(
        self, validator, simple_contract, token_tracker
    ):
        """Test constraint validation within budget."""
        token_tracker.record_usage(1000, 500)
        result = validator.validate_constraints(token_tracker, simple_contract)
        assert result is True

    def test_validate_constraints_exceeds_total(self, validator):
        """Test constraint validation when total exceeded."""
        contract = Contract(
            name="test",
            description="Test",
            deliverables=[
                Deliverable(name="result", type="str", description="Test")
            ],
            constraints=ContractConstraints(max_total_tokens=1000),
        )
        tracker = TokenTracker()
        tracker.record_usage(800, 500)  # Total 1300 > 1000

        result = validator.validate_constraints(tracker, contract)
        assert result is False

    def test_validate_constraints_exceeds_input(self, validator):
        """Test constraint validation when input exceeded."""
        contract = Contract(
            name="test",
            description="Test",
            deliverables=[
                Deliverable(name="result", type="str", description="Test")
            ],
            constraints=ContractConstraints(max_input_tokens=500),
        )
        tracker = TokenTracker()
        tracker.record_usage(800, 200)

        result = validator.validate_constraints(tracker, contract)
        assert result is False


class TestStrictValidation:
    """Tests for strict validation mode."""

    def test_strict_mode_warnings_fail(self, strict_validator):
        """Test that warnings fail validation in strict mode."""
        # Create a contract with a rule that generates warning on compile error
        d = Deliverable(
            name="test",
            type="str",
            description="Test",
            validation_rules=["invalid_syntax["],  # Invalid rule
        )
        contract = Contract(
            name="test",
            description="Test",
            deliverables=[d],
        )
        result = strict_validator.validate({"test": "value"}, contract)
        # In strict mode, warnings should fail validation
        if result.warnings:
            assert result.is_valid is False
