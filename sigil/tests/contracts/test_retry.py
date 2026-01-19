"""Tests for retry management."""

import pytest

from sigil.contracts.schema import Contract, ContractConstraints, Deliverable, FailureStrategy
from sigil.contracts.validator import ValidationError, ErrorSeverity
from sigil.contracts.retry import RetryManager, RetryContext


class TestRetryManager:
    """Tests for RetryManager class."""

    def test_should_retry_first_attempt(self, retry_manager, simple_contract):
        """Test should_retry on first attempt."""
        errors = [
            ValidationError(
                field="score",
                reason="Required deliverable is missing",
            )
        ]
        result = retry_manager.should_retry(
            attempt=1,
            contract=simple_contract,
            tokens_remaining=5000,
            errors=errors,
        )
        assert result is True

    def test_should_retry_max_retries_exceeded(self, retry_manager, simple_contract):
        """Test should_retry when max retries exceeded."""
        errors = [
            ValidationError(field="score", reason="Required deliverable is missing")
        ]
        result = retry_manager.should_retry(
            attempt=3,  # simple_contract has max_retries=2
            contract=simple_contract,
            tokens_remaining=5000,
            errors=errors,
        )
        assert result is False

    def test_should_retry_insufficient_tokens(self, retry_manager, simple_contract):
        """Test should_retry when tokens insufficient."""
        errors = [
            ValidationError(field="score", reason="Required deliverable is missing")
        ]
        result = retry_manager.should_retry(
            attempt=1,
            contract=simple_contract,
            tokens_remaining=100,  # Less than MIN_TOKENS_FOR_RETRY
            errors=errors,
        )
        assert result is False

    def test_should_retry_recoverable_errors(self, retry_manager, simple_contract):
        """Test should_retry with recoverable errors."""
        recoverable_errors = [
            ValidationError(field="score", reason="Required deliverable is missing"),
            ValidationError(field="name", reason="Type mismatch"),
            ValidationError(field="status", reason="Validation rule failed"),
        ]
        for error in recoverable_errors:
            result = retry_manager.should_retry(
                attempt=1,
                contract=simple_contract,
                tokens_remaining=5000,
                errors=[error],
            )
            assert result is True, f"Error '{error.reason}' should be recoverable"


class TestPromptRefinement:
    """Tests for prompt refinement strategies."""

    def test_refine_prompt_iteration_1(self, retry_manager, simple_contract):
        """Test prompt refinement on first iteration."""
        errors = [
            ValidationError(
                field="score",
                reason="Required deliverable is missing",
                expected="int",
            )
        ]
        original = "Score this lead."
        refined = retry_manager.refine_prompt(original, errors, attempt=1)

        assert original in refined
        assert "validation error" in refined.lower()
        assert "score" in refined.lower()

    def test_refine_prompt_iteration_2(self, retry_manager, simple_contract):
        """Test prompt refinement on second iteration."""
        errors = [
            ValidationError(
                field="score",
                reason="Type mismatch",
                expected="int",
                actual="str",
            )
        ]
        original = "Score this lead."
        refined = retry_manager.refine_prompt(
            original, errors, attempt=2, contract=simple_contract
        )

        assert "SIMPLIFIED TASK" in refined
        assert "JSON" in refined
        assert "score" in refined.lower()

    def test_refine_prompt_iteration_3(self, retry_manager, simple_contract):
        """Test prompt refinement on third iteration."""
        errors = [
            ValidationError(field="score", reason="Rule violation")
        ]
        original = "Score this lead."
        refined = retry_manager.refine_prompt(
            original, errors, attempt=3, contract=simple_contract
        )

        assert "FINAL ATTEMPT" in refined
        assert "EXACT structure" in refined

    def test_refine_prompt_includes_error_details(self, retry_manager):
        """Test that prompt includes specific error details."""
        errors = [
            ValidationError(
                field="score",
                reason="Value out of range",
                expected="0 <= value <= 100",
                actual="150",
            )
        ]
        refined = retry_manager.refine_prompt("Test task", errors, attempt=1)

        assert "score" in refined
        assert "out of range" in refined.lower() or "0 <= value <= 100" in refined


class TestRetryContext:
    """Tests for retry context generation."""

    def test_get_retry_context_empty_errors(self, retry_manager, simple_contract):
        """Test retry context with no errors."""
        context = retry_manager.get_retry_context(simple_contract, [])
        assert "No validation errors" in context

    def test_get_retry_context_missing_deliverables(
        self, retry_manager, simple_contract
    ):
        """Test retry context with missing deliverables."""
        errors = [
            ValidationError(
                field="score",
                reason="Required deliverable is missing",
            )
        ]
        context = retry_manager.get_retry_context(simple_contract, errors)

        assert "Missing required deliverables" in context
        assert "score" in context

    def test_get_retry_context_type_mismatches(self, retry_manager, simple_contract):
        """Test retry context with type mismatches."""
        errors = [
            ValidationError(
                field="score",
                reason="Type mismatch",
                expected="int",
                actual="str",
            )
        ]
        context = retry_manager.get_retry_context(simple_contract, errors)

        assert "Type mismatch" in context
        assert "score" in context
        assert "int" in context

    def test_get_retry_context_rule_violations(self, retry_manager, simple_contract):
        """Test retry context with rule violations."""
        errors = [
            ValidationError(
                field="score",
                reason="Validation rule failed: 0 <= value <= 100",
                rule="0 <= value <= 100",
                actual="150",
            )
        ]
        context = retry_manager.get_retry_context(simple_contract, errors)

        assert "rule violation" in context.lower()
        assert "0 <= value <= 100" in context


class TestExpectedFormat:
    """Tests for expected format generation."""

    def test_get_expected_format(self, retry_manager, simple_contract):
        """Test expected format generation."""
        format_desc = retry_manager.get_expected_format(simple_contract)

        assert "score" in format_desc
        assert "recommendation" in format_desc
        assert "int" in format_desc
        assert "str" in format_desc

    def test_get_expected_format_includes_rules(self, retry_manager, simple_contract):
        """Test that format includes validation rules."""
        format_desc = retry_manager.get_expected_format(simple_contract)

        # Should include the validation rules
        assert "Constraints" in format_desc or "0 <= value <= 100" in format_desc

    def test_get_expected_format_marks_required(self, retry_manager, contract_with_optional):
        """Test that required fields are marked."""
        format_desc = retry_manager.get_expected_format(contract_with_optional)

        # Required fields should be marked with *
        assert "*" in format_desc
        assert "required" in format_desc.lower()


class TestTokenEstimation:
    """Tests for retry token estimation."""

    def test_estimate_retry_tokens_basic(self, retry_manager):
        """Test basic token estimation."""
        errors = [
            ValidationError(field="score", reason="Missing"),
        ]
        estimate = retry_manager.estimate_retry_tokens(
            "Score this lead", errors, attempt=1
        )
        assert estimate > 0

    def test_estimate_retry_tokens_increases_with_errors(self, retry_manager):
        """Test that token estimate increases with more errors."""
        one_error = [
            ValidationError(field="score", reason="Missing"),
        ]
        two_errors = [
            ValidationError(field="score", reason="Missing"),
            ValidationError(field="name", reason="Wrong type"),
        ]

        estimate_one = retry_manager.estimate_retry_tokens(
            "Test task", one_error, attempt=1
        )
        estimate_two = retry_manager.estimate_retry_tokens(
            "Test task", two_errors, attempt=1
        )

        assert estimate_two > estimate_one

    def test_estimate_retry_tokens_increases_with_attempt(self, retry_manager):
        """Test that token estimate increases with attempt number."""
        errors = [ValidationError(field="score", reason="Missing")]

        estimate_1 = retry_manager.estimate_retry_tokens("Test", errors, attempt=1)
        estimate_2 = retry_manager.estimate_retry_tokens("Test", errors, attempt=2)
        estimate_3 = retry_manager.estimate_retry_tokens("Test", errors, attempt=3)

        assert estimate_2 > estimate_1
        assert estimate_3 > estimate_2


class TestRetryContextDataclass:
    """Tests for RetryContext dataclass."""

    def test_create_retry_context(self):
        """Test creating a RetryContext."""
        errors = [ValidationError(field="score", reason="Missing")]
        context = RetryContext(
            attempt=1,
            max_retries=3,
            tokens_used=1000,
            tokens_remaining=4000,
            errors=errors,
            original_task="Score this lead",
            refined_task="Score this lead. Fix: score is missing",
        )

        assert context.attempt == 1
        assert context.max_retries == 3
        assert context.tokens_used == 1000
        assert context.tokens_remaining == 4000
        assert len(context.errors) == 1
        assert context.original_task == "Score this lead"
