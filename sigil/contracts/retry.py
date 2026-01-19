"""Retry management for contract execution in Sigil v2 framework.

This module implements intelligent retry logic for contract violations,
including prompt refinement strategies for each retry attempt.

Key Components:
    - RetryManager: Orchestrates retry decisions and prompt refinement

Retry Strategies:
    - Iteration 1: Add validation error details to prompt
    - Iteration 2: Simplify task, explicit format requirements
    - Iteration 3: Use template output structure

Example:
    >>> from sigil.contracts.retry import RetryManager
    >>> manager = RetryManager()
    >>> if manager.should_retry(attempt=1, contract=contract, tokens_remaining=5000, errors=errors):
    ...     refined_prompt = manager.refine_prompt(original_task, errors, attempt=1)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from sigil.contracts.schema import Contract, Deliverable
from sigil.contracts.validator import ValidationError, ValidationResult


@dataclass
class RetryContext:
    """Context information for a retry attempt.

    Attributes:
        attempt: Current attempt number (1-indexed)
        max_retries: Maximum retry attempts allowed
        tokens_used: Total tokens used so far
        tokens_remaining: Estimated remaining token budget
        errors: Validation errors from previous attempt
        original_task: Original task description
        refined_task: Refined task with error context
    """

    attempt: int
    max_retries: int
    tokens_used: int
    tokens_remaining: int
    errors: list[ValidationError]
    original_task: str
    refined_task: str


class RetryManager:
    """Manages retry decisions and prompt refinement for contract execution.

    Provides intelligent retry logic that:
    - Determines if retry is appropriate given errors and resources
    - Refines prompts progressively to help agent succeed
    - Tracks retry attempts and token usage

    Example:
        >>> manager = RetryManager()
        >>> if manager.should_retry(1, contract, 5000, errors):
        ...     new_prompt = manager.refine_prompt(task, errors, 1)
        ...     # Execute agent with new_prompt
    """

    # Default minimum tokens required for a retry attempt
    MIN_TOKENS_FOR_RETRY = 500

    # Error types that are considered recoverable
    RECOVERABLE_ERROR_REASONS = {
        "Required deliverable is missing",
        "Type mismatch",
        "Validation rule failed",
    }

    def __init__(
        self,
        min_tokens_for_retry: int = MIN_TOKENS_FOR_RETRY,
    ) -> None:
        """Initialize the retry manager.

        Args:
            min_tokens_for_retry: Minimum tokens required to attempt a retry.
        """
        self.min_tokens_for_retry = min_tokens_for_retry

    def should_retry(
        self,
        attempt: int,
        contract: Contract,
        tokens_remaining: int,
        errors: list[ValidationError],
    ) -> bool:
        """Determine if a retry should be attempted.

        Checks:
        - Attempt count vs max_retries
        - Token budget availability
        - Error recoverability

        Args:
            attempt: Current attempt number (1-indexed).
            contract: Contract being executed.
            tokens_remaining: Estimated remaining token budget.
            errors: Validation errors from current attempt.

        Returns:
            True if retry should be attempted, False otherwise.
        """
        # Check if we've exceeded max retries
        if attempt >= contract.max_retries:
            return False

        # Check if we have enough tokens for another attempt
        if tokens_remaining < self.min_tokens_for_retry:
            return False

        # Check if any errors are recoverable
        if not self._has_recoverable_errors(errors):
            return False

        return True

    def refine_prompt(
        self,
        original_task: str,
        errors: list[ValidationError],
        attempt: int,
        contract: Optional[Contract] = None,
    ) -> str:
        """Refine the task prompt based on validation errors.

        Progressive refinement strategy:
        - Iteration 1: Add validation error details
        - Iteration 2: Simplify task, explicit format
        - Iteration 3: Use template output structure

        Args:
            original_task: Original task description.
            errors: Validation errors from previous attempt.
            attempt: Current retry attempt number (1-indexed).
            contract: Optional contract for template generation.

        Returns:
            Refined task prompt.
        """
        if attempt == 1:
            return self._refine_with_error_details(original_task, errors)
        elif attempt == 2:
            return self._refine_with_explicit_format(original_task, errors, contract)
        else:
            return self._refine_with_template_structure(original_task, errors, contract)

    def get_retry_context(
        self,
        contract: Contract,
        errors: list[ValidationError],
    ) -> str:
        """Get human-readable error summary for context.

        Creates a formatted summary of validation errors that can be
        used to help the agent understand what went wrong.

        Args:
            contract: Contract that was being validated.
            errors: Validation errors to summarize.

        Returns:
            Human-readable error summary.
        """
        if not errors:
            return "No validation errors."

        lines = [f"Contract '{contract.name}' validation failed:"]

        # Group errors by type
        missing_deliverables = []
        type_mismatches = []
        rule_violations = []
        other_errors = []

        for error in errors:
            if "missing" in error.reason.lower():
                missing_deliverables.append(error)
            elif "type mismatch" in error.reason.lower():
                type_mismatches.append(error)
            elif "rule failed" in error.reason.lower():
                rule_violations.append(error)
            else:
                other_errors.append(error)

        if missing_deliverables:
            lines.append("\nMissing required deliverables:")
            for error in missing_deliverables:
                deliverable = contract.get_deliverable(error.field)
                if deliverable:
                    lines.append(
                        f"  - {error.field} ({deliverable.type}): {deliverable.description}"
                    )
                else:
                    lines.append(f"  - {error.field}")

        if type_mismatches:
            lines.append("\nType mismatches:")
            for error in type_mismatches:
                lines.append(
                    f"  - {error.field}: expected {error.expected}, got {error.actual}"
                )

        if rule_violations:
            lines.append("\nValidation rule violations:")
            for error in rule_violations:
                lines.append(f"  - {error.field}: {error.rule}")
                if error.actual:
                    lines.append(f"    (current value: {error.actual})")

        if other_errors:
            lines.append("\nOther issues:")
            for error in other_errors:
                lines.append(f"  - {error.field}: {error.reason}")

        return "\n".join(lines)

    def get_expected_format(self, contract: Contract) -> str:
        """Generate expected output format description.

        Args:
            contract: Contract to describe format for.

        Returns:
            Human-readable format description.
        """
        lines = ["Expected output format (JSON object with these fields):"]

        for deliverable in contract.deliverables:
            required_marker = "*" if deliverable.required else ""
            type_info = deliverable.type
            lines.append(f"  {deliverable.name}{required_marker}: {type_info}")
            lines.append(f"    - {deliverable.description}")
            if deliverable.validation_rules:
                lines.append(f"    - Constraints: {', '.join(deliverable.validation_rules)}")
            if deliverable.example is not None:
                lines.append(f"    - Example: {deliverable.example}")

        lines.append("\n(* = required field)")
        return "\n".join(lines)

    def _has_recoverable_errors(self, errors: list[ValidationError]) -> bool:
        """Check if any errors are potentially recoverable.

        Args:
            errors: List of validation errors.

        Returns:
            True if at least one error is recoverable.
        """
        for error in errors:
            # Check against known recoverable error patterns
            for recoverable_pattern in self.RECOVERABLE_ERROR_REASONS:
                if recoverable_pattern.lower() in error.reason.lower():
                    return True

        return True  # Default to recoverable if unknown error type

    def _refine_with_error_details(
        self,
        original_task: str,
        errors: list[ValidationError],
    ) -> str:
        """Refine prompt by adding validation error details.

        This is the first level of refinement - simply append
        error information to help the agent understand what failed.

        Args:
            original_task: Original task description.
            errors: Validation errors from previous attempt.

        Returns:
            Refined prompt with error details.
        """
        error_details = []
        for error in errors:
            detail = f"- {error.field}: {error.reason}"
            if error.expected:
                detail += f" (expected: {error.expected})"
            if error.actual:
                detail += f" (got: {error.actual})"
            error_details.append(detail)

        refined = f"""{original_task}

IMPORTANT - Previous attempt had validation errors that must be fixed:
{chr(10).join(error_details)}

Please ensure your response addresses all these issues and provides the correct format."""

        return refined

    def _refine_with_explicit_format(
        self,
        original_task: str,
        errors: list[ValidationError],
        contract: Optional[Contract],
    ) -> str:
        """Refine prompt with explicit format requirements.

        Second level of refinement - simplify the task and
        be very explicit about the expected output format.

        Args:
            original_task: Original task description.
            errors: Validation errors from previous attempt.
            contract: Contract for format specification.

        Returns:
            Refined prompt with explicit format.
        """
        format_spec = ""
        if contract:
            format_spec = self.get_expected_format(contract)

        error_summary = "Previous errors: " + ", ".join(
            f"{e.field} ({e.reason})" for e in errors
        )

        refined = f"""SIMPLIFIED TASK:
{original_task}

{error_summary}

YOUR RESPONSE MUST BE A VALID JSON OBJECT WITH THE FOLLOWING STRUCTURE:

{format_spec}

CRITICAL INSTRUCTIONS:
1. Return ONLY a valid JSON object
2. Include ALL required fields
3. Use the correct data types for each field
4. Ensure values meet the specified constraints
5. Do not include any explanatory text outside the JSON"""

        return refined

    def _refine_with_template_structure(
        self,
        original_task: str,
        errors: list[ValidationError],
        contract: Optional[Contract],
    ) -> str:
        """Refine prompt using template output structure.

        Third level of refinement - provide a concrete template
        that the agent just needs to fill in.

        Args:
            original_task: Original task description.
            errors: Validation errors from previous attempt.
            contract: Contract for template generation.

        Returns:
            Refined prompt with template structure.
        """
        template_json = "{}"
        if contract:
            template = contract.get_template_output()
            import json
            template_json = json.dumps(template, indent=2)

        refined = f"""FINAL ATTEMPT - Please complete this task carefully.

TASK: {original_task}

You MUST respond with a JSON object matching this EXACT structure:
```json
{template_json}
```

FILL IN THE VALUES based on the task. Replace the example values with your actual analysis/results.

CRITICAL:
- Copy the structure exactly
- Keep all field names unchanged
- Only modify the VALUES
- Return valid JSON only"""

        return refined

    def estimate_retry_tokens(
        self,
        original_task: str,
        errors: list[ValidationError],
        attempt: int,
    ) -> int:
        """Estimate tokens needed for a retry attempt.

        Provides a rough estimate of the additional tokens that
        will be consumed by a refined prompt.

        Args:
            original_task: Original task description.
            errors: Validation errors to include.
            attempt: Retry attempt number.

        Returns:
            Estimated additional tokens for retry.
        """
        # Base estimate from original task
        base_tokens = len(original_task.split()) * 2  # rough word-to-token ratio

        # Add overhead for error details
        error_overhead = len(errors) * 50  # ~50 tokens per error

        # Add overhead for format specification (increases with attempt)
        format_overhead = attempt * 100

        return base_tokens + error_overhead + format_overhead


__all__ = [
    "RetryContext",
    "RetryManager",
]
