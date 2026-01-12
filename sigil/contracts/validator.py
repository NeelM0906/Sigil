"""Contract validation for Sigil v2 framework.

This module implements contract validation logic to verify that agent
outputs meet contract specifications.

Key Components:
    - ValidationError: Single validation error with details
    - ValidationResult: Aggregate validation outcome
    - ContractValidator: Main validation engine

The validator supports:
    - Type checking against deliverable specifications
    - Custom validation rules with expressions
    - Constraint validation (tokens, tool calls)
    - Comprehensive error reporting

Example:
    >>> from sigil.contracts.validator import ContractValidator
    >>> from sigil.contracts.schema import Contract, Deliverable
    >>> validator = ContractValidator()
    >>> result = validator.validate(output, contract)
    >>> if result.is_valid:
    ...     print("Output meets contract requirements")
    ... else:
    ...     for error in result.errors:
    ...         print(f"Error: {error.field} - {error.reason}")
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional, Union

from sigil.contracts.schema import Contract, ContractConstraints, Deliverable
from sigil.telemetry.tokens import TokenTracker


class ErrorSeverity(str, Enum):
    """Severity levels for validation errors.

    Attributes:
        ERROR: Critical error that fails validation
        WARNING: Non-critical issue that allows validation to pass
        INFO: Informational message for debugging
    """

    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass
class ValidationError:
    """Single validation error with details.

    Represents a specific validation failure or warning discovered
    during contract validation.

    Attributes:
        field: Name of the deliverable or constraint that failed
        reason: Human-readable description of the failure
        expected: What was expected (type, value, pattern)
        actual: What was actually found
        severity: Error severity (error, warning, info)
        rule: The validation rule that failed (if applicable)

    Example:
        >>> error = ValidationError(
        ...     field="score",
        ...     reason="Value out of range",
        ...     expected="0 <= value <= 100",
        ...     actual="150",
        ...     severity=ErrorSeverity.ERROR
        ... )
    """

    field: str
    reason: str
    expected: Optional[str] = None
    actual: Optional[str] = None
    severity: ErrorSeverity = ErrorSeverity.ERROR
    rule: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize error to dictionary.

        Returns:
            Dictionary representation of the error.
        """
        return {
            "field": self.field,
            "reason": self.reason,
            "expected": self.expected,
            "actual": self.actual,
            "severity": self.severity.value,
            "rule": self.rule,
        }

    def __str__(self) -> str:
        """Format error as human-readable string."""
        parts = [f"[{self.severity.value.upper()}] {self.field}: {self.reason}"]
        if self.expected:
            parts.append(f"Expected: {self.expected}")
        if self.actual:
            parts.append(f"Actual: {self.actual}")
        return " | ".join(parts)


@dataclass
class ValidationResult:
    """Aggregate validation outcome.

    Contains the overall validation result along with all errors,
    warnings, and suggestions for fixing issues.

    Attributes:
        is_valid: Whether the output passes validation
        errors: List of validation errors
        warnings: List of validation warnings
        suggestion: Suggested fix for validation failures
        validation_time_ms: Time taken for validation in milliseconds
        deliverables_checked: List of deliverables that were validated
        partial_output: Partial output that passed validation (if any)

    Example:
        >>> result = ValidationResult(
        ...     is_valid=False,
        ...     errors=[error1, error2],
        ...     suggestion="Ensure score is between 0 and 100"
        ... )
    """

    is_valid: bool
    errors: list[ValidationError] = field(default_factory=list)
    warnings: list[ValidationError] = field(default_factory=list)
    suggestion: Optional[str] = None
    validation_time_ms: float = 0.0
    deliverables_checked: list[str] = field(default_factory=list)
    partial_output: Optional[dict[str, Any]] = None

    def get_error_summary(self) -> str:
        """Get a summary of all errors.

        Returns:
            Human-readable summary of validation errors.
        """
        if not self.errors:
            return "No errors"

        lines = [f"Found {len(self.errors)} validation error(s):"]
        for i, error in enumerate(self.errors, 1):
            lines.append(f"  {i}. {error}")

        return "\n".join(lines)

    def get_failed_fields(self) -> list[str]:
        """Get list of fields that failed validation.

        Returns:
            List of field names with errors.
        """
        return list({error.field for error in self.errors})

    def get_passed_fields(self) -> list[str]:
        """Get list of fields that passed validation.

        Returns:
            List of field names that passed.
        """
        failed = set(self.get_failed_fields())
        return [f for f in self.deliverables_checked if f not in failed]

    def to_dict(self) -> dict[str, Any]:
        """Serialize result to dictionary.

        Returns:
            Dictionary representation of the result.
        """
        return {
            "is_valid": self.is_valid,
            "errors": [e.to_dict() for e in self.errors],
            "warnings": [w.to_dict() for w in self.warnings],
            "suggestion": self.suggestion,
            "validation_time_ms": self.validation_time_ms,
            "deliverables_checked": self.deliverables_checked,
            "partial_output": self.partial_output,
        }


class ContractValidator:
    """Main validation engine for contract verification.

    Validates agent outputs against contract specifications, checking:
    - Required deliverables are present
    - Types match specifications
    - Custom validation rules pass
    - Resource constraints are met

    Example:
        >>> validator = ContractValidator()
        >>> result = validator.validate(
        ...     output={"score": 75, "recommendation": "schedule_demo"},
        ...     contract=contract
        ... )
        >>> print(result.is_valid)
        True
    """

    # Mapping of type names to Python types
    TYPE_MAP: dict[str, type | tuple[type, ...]] = {
        "int": int,
        "float": (int, float),  # Allow int for float
        "str": str,
        "bool": bool,
        "dict": dict,
        "list": list,
        "tuple": tuple,
        "set": set,
        "None": type(None),
        "Any": object,  # Accepts anything
    }

    def __init__(self, strict_mode: bool = False) -> None:
        """Initialize the validator.

        Args:
            strict_mode: If True, treat warnings as errors.
        """
        self.strict_mode = strict_mode
        self._compiled_rules: dict[str, list[Callable[[Any], bool]]] = {}

    def validate(
        self,
        output: dict[str, Any],
        contract: Contract,
        tracker: Optional[TokenTracker] = None,
    ) -> ValidationResult:
        """Validate output against contract.

        Performs comprehensive validation of the output dictionary
        against the contract specification.

        Args:
            output: Output dictionary to validate.
            contract: Contract to validate against.
            tracker: Optional token tracker for constraint checking.

        Returns:
            ValidationResult with validation outcome and details.
        """
        start_time = time.perf_counter()
        errors: list[ValidationError] = []
        warnings: list[ValidationError] = []
        deliverables_checked: list[str] = []
        partial_output: dict[str, Any] = {}

        # Validate each deliverable
        for deliverable in contract.deliverables:
            deliverables_checked.append(deliverable.name)
            deliverable_errors = self.validate_deliverable(
                deliverable.name,
                output.get(deliverable.name),
                deliverable,
            )

            # Separate errors from warnings
            for error in deliverable_errors:
                if error.severity == ErrorSeverity.ERROR:
                    errors.append(error)
                else:
                    warnings.append(error)
                    # If only warning, include in partial output
                    if deliverable.name in output:
                        partial_output[deliverable.name] = output[deliverable.name]

            # If no errors for this deliverable, add to partial output
            if not any(e.severity == ErrorSeverity.ERROR for e in deliverable_errors):
                if deliverable.name in output:
                    partial_output[deliverable.name] = output[deliverable.name]

        # Validate constraints if tracker provided
        if tracker is not None:
            constraint_errors = self._validate_constraints(tracker, contract)
            errors.extend(constraint_errors)

        # Calculate validation time
        validation_time_ms = (time.perf_counter() - start_time) * 1000

        # Determine if valid (in strict mode, warnings also fail)
        is_valid = len(errors) == 0
        if self.strict_mode and warnings:
            is_valid = False

        # Generate suggestion
        suggestion = self._generate_suggestion(errors, contract)

        return ValidationResult(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
            suggestion=suggestion,
            validation_time_ms=validation_time_ms,
            deliverables_checked=deliverables_checked,
            partial_output=partial_output if partial_output else None,
        )

    def validate_deliverable(
        self,
        name: str,
        value: Any,
        deliverable: Deliverable,
    ) -> list[ValidationError]:
        """Validate a single deliverable value.

        Args:
            name: Name of the deliverable being validated.
            value: The value to validate.
            deliverable: The deliverable specification.

        Returns:
            List of validation errors (empty if valid).
        """
        errors: list[ValidationError] = []

        # Check if required deliverable is missing
        if value is None:
            if deliverable.required:
                errors.append(
                    ValidationError(
                        field=name,
                        reason="Required deliverable is missing",
                        expected=f"Value of type {deliverable.type}",
                        actual="None/missing",
                        severity=ErrorSeverity.ERROR,
                    )
                )
            return errors

        # Check type
        if not self._check_type(value, deliverable.type):
            errors.append(
                ValidationError(
                    field=name,
                    reason="Type mismatch",
                    expected=deliverable.type,
                    actual=type(value).__name__,
                    severity=ErrorSeverity.ERROR,
                )
            )
            # Don't check rules if type is wrong
            return errors

        # Check validation rules
        rule_errors = self._check_rules(name, value, deliverable.validation_rules)
        errors.extend(rule_errors)

        return errors

    def validate_constraints(
        self,
        tracker: TokenTracker,
        contract: Contract,
    ) -> bool:
        """Check if execution constraints were met.

        Args:
            tracker: Token tracker with usage information.
            contract: Contract with constraints.

        Returns:
            True if within all constraints, False otherwise.
        """
        errors = self._validate_constraints(tracker, contract)
        return len(errors) == 0

    def _validate_constraints(
        self,
        tracker: TokenTracker,
        contract: Contract,
    ) -> list[ValidationError]:
        """Internal constraint validation with error details.

        Args:
            tracker: Token tracker with usage information.
            contract: Contract with constraints.

        Returns:
            List of constraint validation errors.
        """
        errors: list[ValidationError] = []
        constraints = contract.constraints

        # Check token limits
        if constraints.max_total_tokens is not None:
            if tracker.total_tokens > constraints.max_total_tokens:
                errors.append(
                    ValidationError(
                        field="_constraints.max_total_tokens",
                        reason="Total token limit exceeded",
                        expected=f"<= {constraints.max_total_tokens}",
                        actual=str(tracker.total_tokens),
                        severity=ErrorSeverity.ERROR,
                    )
                )

        if constraints.max_input_tokens is not None:
            if tracker.input_tokens > constraints.max_input_tokens:
                errors.append(
                    ValidationError(
                        field="_constraints.max_input_tokens",
                        reason="Input token limit exceeded",
                        expected=f"<= {constraints.max_input_tokens}",
                        actual=str(tracker.input_tokens),
                        severity=ErrorSeverity.ERROR,
                    )
                )

        if constraints.max_output_tokens is not None:
            if tracker.output_tokens > constraints.max_output_tokens:
                errors.append(
                    ValidationError(
                        field="_constraints.max_output_tokens",
                        reason="Output token limit exceeded",
                        expected=f"<= {constraints.max_output_tokens}",
                        actual=str(tracker.output_tokens),
                        severity=ErrorSeverity.ERROR,
                    )
                )

        return errors

    def compile_rules(self, rules: list[str]) -> list[Callable[[Any], bool]]:
        """Convert rule strings to validator functions.

        Compiles validation rule expressions into callable validators.
        Supports:
        - Comparison expressions: "0 <= value <= 100"
        - Type checks: "isinstance(value, int)"
        - Length checks: "len(value) > 0"
        - Membership checks: "value in ['a', 'b', 'c']"
        - Boolean expressions: "value.startswith('http')"

        Args:
            rules: List of validation rule expressions.

        Returns:
            List of compiled validator functions.

        Example:
            >>> validator = ContractValidator()
            >>> validators = validator.compile_rules(["0 <= value <= 100"])
            >>> validators[0](75)
            True
            >>> validators[0](150)
            False
        """
        validators: list[Callable[[Any], bool]] = []

        for rule in rules:
            validator = self._compile_single_rule(rule)
            if validator:
                validators.append(validator)

        return validators

    def _compile_single_rule(self, rule: str) -> Optional[Callable[[Any], bool]]:
        """Compile a single rule string to a validator function.

        Args:
            rule: Validation rule expression.

        Returns:
            Callable validator or None if compilation fails.
        """
        # Try to create a lambda from the expression
        try:
            # Create a safe evaluation context
            def make_validator(expr: str) -> Callable[[Any], bool]:
                def validator(value: Any) -> bool:
                    try:
                        # Safe eval with limited globals
                        safe_globals = {
                            "__builtins__": {},
                            "len": len,
                            "isinstance": isinstance,
                            "int": int,
                            "float": float,
                            "str": str,
                            "bool": bool,
                            "list": list,
                            "dict": dict,
                            "tuple": tuple,
                            "set": set,
                            "True": True,
                            "False": False,
                            "None": None,
                        }
                        local_vars = {"value": value}
                        return bool(eval(expr, safe_globals, local_vars))
                    except Exception:
                        return False

                return validator

            return make_validator(rule)

        except Exception:
            return None

    def _check_type(self, value: Any, expected_type: str) -> bool:
        """Check if value matches expected type.

        Args:
            value: Value to check.
            expected_type: Expected type as string.

        Returns:
            True if type matches, False otherwise.
        """
        # Handle parameterized types (e.g., list[str], dict[str, int])
        base_type = expected_type.split("[")[0].strip()

        # Look up the Python type
        python_type = self.TYPE_MAP.get(base_type)
        if python_type is None:
            # Unknown type, accept anything
            return True

        return isinstance(value, python_type)

    def _check_rules(
        self,
        field_name: str,
        value: Any,
        rules: list[str],
    ) -> list[ValidationError]:
        """Check all validation rules for a value.

        Args:
            field_name: Name of the field being validated.
            value: Value to check.
            rules: List of validation rule expressions.

        Returns:
            List of validation errors for failed rules.
        """
        errors: list[ValidationError] = []

        for rule in rules:
            validator = self._compile_single_rule(rule)
            if validator is None:
                # Rule compilation failed
                errors.append(
                    ValidationError(
                        field=field_name,
                        reason=f"Invalid validation rule",
                        expected="Valid expression",
                        actual=rule,
                        severity=ErrorSeverity.WARNING,
                        rule=rule,
                    )
                )
                continue

            try:
                if not validator(value):
                    errors.append(
                        ValidationError(
                            field=field_name,
                            reason=f"Validation rule failed: {rule}",
                            expected=rule,
                            actual=self._format_value(value),
                            severity=ErrorSeverity.ERROR,
                            rule=rule,
                        )
                    )
            except Exception as e:
                errors.append(
                    ValidationError(
                        field=field_name,
                        reason=f"Rule evaluation error: {str(e)}",
                        expected=rule,
                        actual=self._format_value(value),
                        severity=ErrorSeverity.ERROR,
                        rule=rule,
                    )
                )

        return errors

    def _format_value(self, value: Any, max_length: int = 100) -> str:
        """Format a value for error message display.

        Args:
            value: Value to format.
            max_length: Maximum string length.

        Returns:
            Formatted string representation.
        """
        str_value = repr(value)
        if len(str_value) > max_length:
            str_value = str_value[: max_length - 3] + "..."
        return str_value

    def _generate_suggestion(
        self,
        errors: list[ValidationError],
        contract: Contract,
    ) -> Optional[str]:
        """Generate a suggestion for fixing validation errors.

        Args:
            errors: List of validation errors.
            contract: The contract being validated against.

        Returns:
            Human-readable suggestion or None.
        """
        if not errors:
            return None

        suggestions: list[str] = []

        for error in errors[:3]:  # Limit to first 3 errors
            if "missing" in error.reason.lower():
                deliverable = contract.get_deliverable(error.field)
                if deliverable:
                    suggestions.append(
                        f"Include '{error.field}' ({deliverable.type}): {deliverable.description}"
                    )
            elif "type mismatch" in error.reason.lower():
                suggestions.append(
                    f"Ensure '{error.field}' is of type {error.expected}, not {error.actual}"
                )
            elif "rule failed" in error.reason.lower():
                suggestions.append(
                    f"For '{error.field}', ensure: {error.rule}"
                )
            elif "constraint" in error.field.lower():
                suggestions.append(
                    f"Reduce token usage to meet constraint: {error.expected}"
                )

        if not suggestions:
            return "Review the contract requirements and ensure all deliverables meet specifications."

        return " | ".join(suggestions)


__all__ = [
    "ErrorSeverity",
    "ValidationError",
    "ValidationResult",
    "ContractValidator",
]
