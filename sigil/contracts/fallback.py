"""Fallback management for contract execution in Sigil v2 framework.

This module implements fallback strategies when contract validation fails
after all retry attempts have been exhausted.

Key Components:
    - FallbackManager: Orchestrates fallback strategy selection and execution

Fallback Strategies:
    1. Partial Result: Return what passed validation + warnings
    2. Template Result: Return contract-compliant template output
    3. Escalation: Raise exception (for fail strategy)

Example:
    >>> from sigil.contracts.fallback import FallbackManager
    >>> manager = FallbackManager()
    >>> if manager.can_build_partial(output, contract):
    ...     result = manager.build_partial_result(output, contract)
    ... else:
    ...     result = manager.build_template_result(contract)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

from sigil.contracts.schema import Contract, Deliverable, FailureStrategy
from sigil.contracts.validator import ValidationError, ValidationResult


class FallbackStrategy(str, Enum):
    """Strategy used for generating fallback output.

    Attributes:
        PARTIAL: Use partial output that passed validation
        TEMPLATE: Use template-based output from contract
        ESCALATE: Raise an exception (no fallback)
        NONE: No fallback applied (success case)
    """

    PARTIAL = "partial"
    TEMPLATE = "template"
    ESCALATE = "escalate"
    NONE = "none"


@dataclass
class FallbackResult:
    """Result of a fallback operation.

    Attributes:
        output: The fallback output dictionary
        strategy: Which fallback strategy was used
        warnings: List of warnings about the fallback
        missing_fields: Fields that could not be provided
        partial_fields: Fields that came from partial output
        template_fields: Fields generated from template
    """

    output: dict[str, Any]
    strategy: FallbackStrategy
    warnings: list[str]
    missing_fields: list[str]
    partial_fields: list[str]
    template_fields: list[str]

    def to_dict(self) -> dict[str, Any]:
        """Serialize fallback result to dictionary.

        Returns:
            Dictionary representation.
        """
        return {
            "output": self.output,
            "strategy": self.strategy.value,
            "warnings": self.warnings,
            "missing_fields": self.missing_fields,
            "partial_fields": self.partial_fields,
            "template_fields": self.template_fields,
        }


class FallbackManager:
    """Manages fallback strategies when contract validation fails.

    Provides fallback logic that:
    - Determines if partial output is usable
    - Builds partial results from valid portions
    - Generates template-based fallback outputs
    - Selects the best fallback strategy

    Example:
        >>> manager = FallbackManager()
        >>> result = manager.select_strategy(output, contract, validation_result)
        >>> if result.strategy == FallbackStrategy.PARTIAL:
        ...     print("Using partial output")
    """

    # Minimum percentage of required fields needed for partial result
    MIN_PARTIAL_COVERAGE = 0.5

    def __init__(
        self,
        min_partial_coverage: float = MIN_PARTIAL_COVERAGE,
    ) -> None:
        """Initialize the fallback manager.

        Args:
            min_partial_coverage: Minimum coverage for partial results (0.0-1.0).
        """
        self.min_partial_coverage = min_partial_coverage

    def can_build_partial(
        self,
        output: Optional[dict[str, Any]],
        contract: Contract,
        validation_result: Optional[ValidationResult] = None,
    ) -> bool:
        """Check if partial output is usable.

        A partial result is usable if:
        - Output exists
        - At least MIN_PARTIAL_COVERAGE of required fields are valid
        - Valid fields are meaningful (not empty/None)

        Args:
            output: Output dictionary from agent.
            contract: Contract being validated.
            validation_result: Optional validation result with partial_output.

        Returns:
            True if partial result can be built, False otherwise.
        """
        if output is None:
            return False

        # If we have validation result, use its partial output info
        if validation_result and validation_result.partial_output:
            valid_fields = set(validation_result.partial_output.keys())
        else:
            valid_fields = set(output.keys())

        # Count required fields that are valid
        required = contract.get_required_deliverables()
        if not required:
            return True

        valid_required = 0
        for deliverable in required:
            if deliverable.name in valid_fields:
                value = output.get(deliverable.name)
                if self._is_meaningful_value(value):
                    valid_required += 1

        coverage = valid_required / len(required)
        return coverage >= self.min_partial_coverage

    def build_partial_result(
        self,
        output: dict[str, Any],
        contract: Contract,
        validation_result: Optional[ValidationResult] = None,
    ) -> FallbackResult:
        """Build a partial result from valid portions of output.

        Returns valid fields from output plus template values for
        missing/invalid fields, along with appropriate warnings.

        Args:
            output: Output dictionary from agent.
            contract: Contract being validated.
            validation_result: Optional validation result with partial_output.

        Returns:
            FallbackResult with partial output and metadata.
        """
        result_output: dict[str, Any] = {}
        warnings: list[str] = []
        partial_fields: list[str] = []
        template_fields: list[str] = []
        missing_fields: list[str] = []

        # Determine which fields are valid
        if validation_result and validation_result.partial_output:
            valid_output = validation_result.partial_output
        else:
            valid_output = output

        # Get template for fallback values
        template = contract.get_template_output()

        # Process each deliverable
        for deliverable in contract.deliverables:
            name = deliverable.name

            if name in valid_output:
                # Use the valid output value
                result_output[name] = valid_output[name]
                partial_fields.append(name)
            elif name in output and output[name] is not None:
                # Value exists but failed validation - use template
                result_output[name] = template.get(name)
                template_fields.append(name)
                warnings.append(
                    f"'{name}' failed validation, using template value"
                )
            elif name in template:
                # Missing field - use template
                result_output[name] = template.get(name)
                template_fields.append(name)
                if deliverable.required:
                    warnings.append(
                        f"Required field '{name}' is missing, using template value"
                    )
                    missing_fields.append(name)
            else:
                # No value available
                missing_fields.append(name)
                warnings.append(f"No value available for '{name}'")

        # Add overall warning about partial result
        if missing_fields or template_fields:
            warnings.insert(
                0,
                f"Partial result: {len(partial_fields)} valid, "
                f"{len(template_fields)} template, {len(missing_fields)} missing"
            )

        return FallbackResult(
            output=result_output,
            strategy=FallbackStrategy.PARTIAL,
            warnings=warnings,
            missing_fields=missing_fields,
            partial_fields=partial_fields,
            template_fields=template_fields,
        )

    def build_template_result(self, contract: Contract) -> FallbackResult:
        """Build a template-based fallback result.

        Creates output using template values from contract deliverables.
        All fields will be template values.

        Args:
            contract: Contract to build template from.

        Returns:
            FallbackResult with template output.
        """
        template = contract.get_template_output()
        template_fields = list(template.keys())

        warnings = [
            f"Template fallback: All {len(template_fields)} fields are template values",
            "This output should be reviewed before use",
        ]

        return FallbackResult(
            output=template,
            strategy=FallbackStrategy.TEMPLATE,
            warnings=warnings,
            missing_fields=[],
            partial_fields=[],
            template_fields=template_fields,
        )

    def select_strategy(
        self,
        output: Optional[dict[str, Any]],
        contract: Contract,
        validation_result: Optional[ValidationResult] = None,
    ) -> FallbackResult:
        """Select and execute the best fallback strategy.

        Strategy selection logic:
        1. If contract strategy is FAIL, return escalation
        2. If partial output is viable, use partial result
        3. Otherwise, use template result

        Args:
            output: Output dictionary from agent (may be None).
            contract: Contract being validated.
            validation_result: Optional validation result.

        Returns:
            FallbackResult with chosen strategy and output.
        """
        # Check if we should escalate (fail strategy)
        if contract.failure_strategy == FailureStrategy.FAIL:
            return FallbackResult(
                output={},
                strategy=FallbackStrategy.ESCALATE,
                warnings=["Contract failure strategy is FAIL - escalating error"],
                missing_fields=contract.get_deliverable_names(),
                partial_fields=[],
                template_fields=[],
            )

        # Try partial result first
        if self.can_build_partial(output, contract, validation_result):
            return self.build_partial_result(output, contract, validation_result)

        # Fall back to template
        return self.build_template_result(contract)

    def merge_with_template(
        self,
        output: dict[str, Any],
        contract: Contract,
    ) -> dict[str, Any]:
        """Merge output with template to fill missing fields.

        Useful for filling in gaps without full fallback handling.

        Args:
            output: Output dictionary (possibly incomplete).
            contract: Contract with template definitions.

        Returns:
            Merged output with all deliverable fields.
        """
        template = contract.get_template_output()
        merged = dict(template)  # Start with template
        merged.update(output)  # Override with actual output
        return merged

    def _is_meaningful_value(self, value: Any) -> bool:
        """Check if a value is meaningful (not empty/None).

        Args:
            value: Value to check.

        Returns:
            True if value is meaningful.
        """
        if value is None:
            return False
        if isinstance(value, str) and value.strip() == "":
            return False
        if isinstance(value, (list, dict)) and len(value) == 0:
            return False
        return True

    def get_strategy_description(self, strategy: FallbackStrategy) -> str:
        """Get human-readable description of a fallback strategy.

        Args:
            strategy: Fallback strategy to describe.

        Returns:
            Human-readable description.
        """
        descriptions = {
            FallbackStrategy.PARTIAL: (
                "Partial result: Returns valid fields from agent output combined "
                "with template values for invalid/missing fields."
            ),
            FallbackStrategy.TEMPLATE: (
                "Template result: Returns a complete output using template values "
                "from the contract when no valid output is available."
            ),
            FallbackStrategy.ESCALATE: (
                "Escalation: Raises an exception because the contract's failure "
                "strategy is set to 'fail'."
            ),
            FallbackStrategy.NONE: (
                "No fallback: Output passed validation and no fallback was needed."
            ),
        }
        return descriptions.get(strategy, "Unknown strategy")


__all__ = [
    "FallbackStrategy",
    "FallbackResult",
    "FallbackManager",
]
