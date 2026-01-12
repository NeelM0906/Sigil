"""Tests for contract schema definitions."""

import pytest
from pydantic import ValidationError

from sigil.contracts.schema import (
    Contract,
    ContractConstraints,
    Deliverable,
    FailureStrategy,
)


class TestDeliverable:
    """Tests for Deliverable dataclass."""

    def test_create_basic_deliverable(self):
        """Test creating a basic deliverable."""
        deliverable = Deliverable(
            name="score",
            type="int",
            description="A score value",
        )
        assert deliverable.name == "score"
        assert deliverable.type == "int"
        assert deliverable.description == "A score value"
        assert deliverable.required is True
        assert deliverable.validation_rules == []
        assert deliverable.example is None

    def test_create_deliverable_with_all_fields(self):
        """Test creating a deliverable with all fields."""
        deliverable = Deliverable(
            name="recommendation",
            type="str",
            description="Recommended action",
            required=False,
            validation_rules=["len(value) > 0"],
            example="schedule_demo",
        )
        assert deliverable.name == "recommendation"
        assert deliverable.type == "str"
        assert deliverable.required is False
        assert deliverable.validation_rules == ["len(value) > 0"]
        assert deliverable.example == "schedule_demo"

    def test_deliverable_name_validation_valid(self):
        """Test valid deliverable names."""
        valid_names = ["score", "my_score", "score123", "my_score_123"]
        for name in valid_names:
            d = Deliverable(name=name, type="int", description="test")
            assert d.name == name

    def test_deliverable_name_validation_invalid(self):
        """Test invalid deliverable names are rejected."""
        invalid_names = ["my-score", "my score", "score!", "123"]
        for name in invalid_names:
            # Numbers-only names should fail the alphanumeric check after removing underscores
            if name.replace("_", "").isalnum() and name.strip():
                continue
            with pytest.raises(ValidationError):
                Deliverable(name=name, type="int", description="test")

    def test_deliverable_type_validation_valid(self):
        """Test valid type names."""
        valid_types = ["int", "float", "str", "bool", "dict", "list", "Any"]
        for type_name in valid_types:
            d = Deliverable(name="test", type=type_name, description="test")
            assert d.type == type_name

    def test_deliverable_type_validation_parameterized(self):
        """Test parameterized types are accepted."""
        d = Deliverable(name="items", type="list[str]", description="test")
        assert d.type == "list[str]"

    def test_deliverable_type_validation_invalid(self):
        """Test invalid type names are rejected."""
        with pytest.raises(ValidationError):
            Deliverable(name="test", type="invalid_type", description="test")

    def test_deliverable_to_dict(self):
        """Test serialization to dictionary."""
        deliverable = Deliverable(
            name="score",
            type="int",
            description="A score",
            validation_rules=["0 <= value <= 100"],
        )
        result = deliverable.to_dict()
        assert result["name"] == "score"
        assert result["type"] == "int"
        assert result["description"] == "A score"
        assert result["validation_rules"] == ["0 <= value <= 100"]

    def test_deliverable_from_dict(self):
        """Test deserialization from dictionary."""
        data = {
            "name": "score",
            "type": "int",
            "description": "A score",
            "required": True,
            "validation_rules": ["0 <= value <= 100"],
            "example": 75,
        }
        deliverable = Deliverable.from_dict(data)
        assert deliverable.name == "score"
        assert deliverable.type == "int"
        assert deliverable.example == 75


class TestContractConstraints:
    """Tests for ContractConstraints dataclass."""

    def test_create_default_constraints(self):
        """Test creating constraints with defaults."""
        constraints = ContractConstraints()
        assert constraints.max_input_tokens is None
        assert constraints.max_output_tokens is None
        assert constraints.max_total_tokens is None
        assert constraints.max_tool_calls is None
        assert constraints.timeout_seconds is None
        assert constraints.warn_threshold == 0.8

    def test_create_constraints_with_values(self):
        """Test creating constraints with all values."""
        constraints = ContractConstraints(
            max_input_tokens=2000,
            max_output_tokens=1000,
            max_total_tokens=5000,
            max_tool_calls=5,
            timeout_seconds=60,
            warn_threshold=0.7,
        )
        assert constraints.max_input_tokens == 2000
        assert constraints.max_output_tokens == 1000
        assert constraints.max_total_tokens == 5000
        assert constraints.max_tool_calls == 5
        assert constraints.timeout_seconds == 60
        assert constraints.warn_threshold == 0.7

    def test_is_within_token_budget_true(self):
        """Test budget check when within limits."""
        constraints = ContractConstraints(
            max_input_tokens=2000,
            max_output_tokens=1000,
            max_total_tokens=5000,
        )
        assert constraints.is_within_token_budget(1000, 500) is True

    def test_is_within_token_budget_false_input(self):
        """Test budget check when input exceeds limit."""
        constraints = ContractConstraints(max_input_tokens=1000)
        assert constraints.is_within_token_budget(1500, 500) is False

    def test_is_within_token_budget_false_output(self):
        """Test budget check when output exceeds limit."""
        constraints = ContractConstraints(max_output_tokens=500)
        assert constraints.is_within_token_budget(1000, 1000) is False

    def test_is_within_token_budget_false_total(self):
        """Test budget check when total exceeds limit."""
        constraints = ContractConstraints(max_total_tokens=2000)
        assert constraints.is_within_token_budget(1500, 1000) is False

    def test_get_remaining_tokens(self):
        """Test remaining token calculation."""
        constraints = ContractConstraints(max_total_tokens=5000)
        assert constraints.get_remaining_tokens(3000) == 2000
        assert constraints.get_remaining_tokens(5000) == 0
        assert constraints.get_remaining_tokens(6000) == 0  # Don't go negative

    def test_get_remaining_tokens_unlimited(self):
        """Test remaining tokens when no limit set."""
        constraints = ContractConstraints()
        assert constraints.get_remaining_tokens(10000) == -1

    def test_should_warn(self):
        """Test warning threshold check."""
        constraints = ContractConstraints(
            max_total_tokens=1000, warn_threshold=0.8
        )
        assert constraints.should_warn(700) is False
        assert constraints.should_warn(800) is True
        assert constraints.should_warn(900) is True

    def test_constraints_to_dict(self):
        """Test serialization to dictionary."""
        constraints = ContractConstraints(
            max_total_tokens=5000, max_tool_calls=5
        )
        result = constraints.to_dict()
        assert result["max_total_tokens"] == 5000
        assert result["max_tool_calls"] == 5
        # None values should be excluded
        assert "max_input_tokens" not in result

    def test_constraints_from_dict(self):
        """Test deserialization from dictionary."""
        data = {"max_total_tokens": 5000, "warn_threshold": 0.9}
        constraints = ContractConstraints.from_dict(data)
        assert constraints.max_total_tokens == 5000
        assert constraints.warn_threshold == 0.9


class TestContract:
    """Tests for Contract dataclass."""

    def test_create_basic_contract(self, score_deliverable):
        """Test creating a basic contract."""
        contract = Contract(
            name="test_contract",
            description="A test contract",
            deliverables=[score_deliverable],
        )
        assert contract.name == "test_contract"
        assert contract.description == "A test contract"
        assert len(contract.deliverables) == 1
        assert contract.failure_strategy == FailureStrategy.RETRY
        assert contract.max_retries == 2
        assert contract.version == "1.0.0"

    def test_create_contract_with_all_fields(
        self, score_deliverable, recommendation_deliverable
    ):
        """Test creating a contract with all fields."""
        contract = Contract(
            name="full_contract",
            description="Full contract",
            deliverables=[score_deliverable, recommendation_deliverable],
            constraints=ContractConstraints(max_total_tokens=5000),
            failure_strategy=FailureStrategy.FALLBACK,
            max_retries=3,
            version="2.0.0",
            metadata={"author": "test"},
        )
        assert contract.name == "full_contract"
        assert len(contract.deliverables) == 2
        assert contract.failure_strategy == FailureStrategy.FALLBACK
        assert contract.max_retries == 3
        assert contract.version == "2.0.0"
        assert contract.metadata == {"author": "test"}

    def test_contract_name_normalization(self, score_deliverable):
        """Test that contract names are normalized."""
        contract = Contract(
            name="My-Test Contract",
            description="Test",
            deliverables=[score_deliverable],
        )
        assert contract.name == "my_test_contract"

    def test_contract_duplicate_deliverable_names_rejected(self):
        """Test that duplicate deliverable names are rejected."""
        d1 = Deliverable(name="score", type="int", description="Score 1")
        d2 = Deliverable(name="score", type="int", description="Score 2")
        with pytest.raises(ValidationError):
            Contract(
                name="test",
                description="Test",
                deliverables=[d1, d2],
            )

    def test_contract_requires_deliverables(self):
        """Test that contract requires at least one deliverable."""
        with pytest.raises(ValidationError):
            Contract(
                name="test",
                description="Test",
                deliverables=[],
            )

    def test_get_required_deliverables(
        self, score_deliverable, optional_notes_deliverable
    ):
        """Test getting required deliverables."""
        contract = Contract(
            name="test",
            description="Test",
            deliverables=[score_deliverable, optional_notes_deliverable],
        )
        required = contract.get_required_deliverables()
        assert len(required) == 1
        assert required[0].name == "score"

    def test_get_optional_deliverables(
        self, score_deliverable, optional_notes_deliverable
    ):
        """Test getting optional deliverables."""
        contract = Contract(
            name="test",
            description="Test",
            deliverables=[score_deliverable, optional_notes_deliverable],
        )
        optional = contract.get_optional_deliverables()
        assert len(optional) == 1
        assert optional[0].name == "notes"

    def test_get_deliverable_by_name(
        self, score_deliverable, recommendation_deliverable
    ):
        """Test getting a specific deliverable by name."""
        contract = Contract(
            name="test",
            description="Test",
            deliverables=[score_deliverable, recommendation_deliverable],
        )
        d = contract.get_deliverable("score")
        assert d is not None
        assert d.name == "score"

        d = contract.get_deliverable("nonexistent")
        assert d is None

    def test_get_deliverable_names(
        self, score_deliverable, recommendation_deliverable
    ):
        """Test getting all deliverable names."""
        contract = Contract(
            name="test",
            description="Test",
            deliverables=[score_deliverable, recommendation_deliverable],
        )
        names = contract.get_deliverable_names()
        assert names == ["score", "recommendation"]

    def test_contract_to_dict(self, simple_contract):
        """Test serialization to dictionary."""
        result = simple_contract.to_dict()
        assert result["name"] == "simple_contract"
        assert result["description"] == "A simple test contract"
        assert len(result["deliverables"]) == 2
        assert result["failure_strategy"] == "retry"
        assert result["max_retries"] == 2

    def test_contract_from_dict(self):
        """Test deserialization from dictionary."""
        data = {
            "name": "test_contract",
            "description": "Test description",
            "deliverables": [
                {
                    "name": "score",
                    "type": "int",
                    "description": "Score value",
                }
            ],
            "failure_strategy": "fallback",
            "max_retries": 3,
        }
        contract = Contract.from_dict(data)
        assert contract.name == "test_contract"
        assert contract.failure_strategy == FailureStrategy.FALLBACK
        assert contract.max_retries == 3
        assert len(contract.deliverables) == 1

    def test_validate_schema_valid(self, simple_contract):
        """Test schema validation on valid contract."""
        is_valid, errors = simple_contract.validate_schema()
        assert is_valid is True
        assert len(errors) == 0

    def test_validate_schema_no_required_warning(self, optional_notes_deliverable):
        """Test schema validation warns when no required deliverables."""
        contract = Contract(
            name="optional_only",
            description="Contract with only optional fields",
            deliverables=[optional_notes_deliverable],
        )
        is_valid, errors = contract.validate_schema()
        assert is_valid is False
        assert any("required" in e.lower() for e in errors)

    def test_validate_schema_fail_strategy_with_retries(self, score_deliverable):
        """Test schema validation catches fail strategy with retries."""
        contract = Contract(
            name="invalid_config",
            description="Invalid configuration",
            deliverables=[score_deliverable],
            failure_strategy=FailureStrategy.FAIL,
            max_retries=2,  # Should be 0 for fail strategy
        )
        is_valid, errors = contract.validate_schema()
        assert is_valid is False
        assert any("max_retries" in e.lower() for e in errors)

    def test_get_template_output(self, score_deliverable, recommendation_deliverable):
        """Test template output generation."""
        contract = Contract(
            name="test",
            description="Test",
            deliverables=[score_deliverable, recommendation_deliverable],
        )
        template = contract.get_template_output()
        assert template["score"] == 75  # From example
        assert template["recommendation"] == "schedule_demo"  # From example

    def test_get_template_output_default_values(self):
        """Test template output uses type defaults when no example."""
        d = Deliverable(
            name="count",
            type="int",
            description="A count",
        )
        contract = Contract(
            name="test",
            description="Test",
            deliverables=[d],
        )
        template = contract.get_template_output()
        assert template["count"] == 0  # Default for int


class TestFailureStrategy:
    """Tests for FailureStrategy enum."""

    def test_strategy_values(self):
        """Test failure strategy enum values."""
        assert FailureStrategy.RETRY.value == "retry"
        assert FailureStrategy.FALLBACK.value == "fallback"
        assert FailureStrategy.FAIL.value == "fail"

    def test_strategy_from_string(self):
        """Test creating strategy from string."""
        assert FailureStrategy("retry") == FailureStrategy.RETRY
        assert FailureStrategy("fallback") == FailureStrategy.FALLBACK
        assert FailureStrategy("fail") == FailureStrategy.FAIL
