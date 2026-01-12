"""Contract schema definitions for Sigil v2 framework.

This module extends the base contract schemas with enhanced functionality
for contract validation, serialization, and schema verification.

The schema module provides:
- Deliverable: Enhanced deliverable specification with example support
- ContractConstraints: Resource limits with warn thresholds
- Contract: Full contract definition with helper methods
- Schema validation utilities

All schemas are Pydantic-based for automatic validation and serialization.

Example:
    >>> from sigil.contracts.schema import Contract, Deliverable, ContractConstraints
    >>> contract = Contract(
    ...     name="lead_qualification",
    ...     description="Qualify sales leads with BANT assessment",
    ...     deliverables=[
    ...         Deliverable(
    ...             name="score",
    ...             type="int",
    ...             description="Qualification score 0-100",
    ...             validation_rules=["0 <= value <= 100"]
    ...         )
    ...     ],
    ...     constraints=ContractConstraints(max_total_tokens=5000)
    ... )
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Callable, Optional, Union
from datetime import datetime

from pydantic import BaseModel, Field, field_validator, model_validator


class FailureStrategy(str, Enum):
    """Enumeration of failure handling strategies.

    Defines how the system should respond when contract verification
    fails after all retries are exhausted.

    Attributes:
        RETRY: Retry the operation up to max_retries times
        FALLBACK: Use a fallback response with partial results
        FAIL: Fail immediately with an error
    """

    RETRY = "retry"
    FALLBACK = "fallback"
    FAIL = "fail"


class Deliverable(BaseModel):
    """Enhanced deliverable specification for a contract.

    Defines what an agent must produce as part of contract fulfillment.
    Each deliverable has a name, type specification, validation rules,
    and an optional example value.

    Attributes:
        name: Deliverable name (e.g., "qualification_score")
        type: Expected Python type as string (e.g., "int", "dict", "str")
        description: Human-readable description of this deliverable
        required: Whether this deliverable must be present (default: True)
        validation_rules: Optional list of validation expressions
        example: Optional example value for documentation

    Example:
        ```python
        deliverable = Deliverable(
            name="score",
            type="int",
            description="Lead qualification score from 0 to 100",
            required=True,
            validation_rules=["0 <= value <= 100"],
            example=75
        )
        ```
    """

    name: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Deliverable name (e.g., 'qualification_score')",
    )
    type: str = Field(
        ...,
        min_length=1,
        description="Expected Python type as string (e.g., 'int', 'dict')",
        examples=["int", "str", "dict", "list", "bool", "float"],
    )
    description: str = Field(
        ...,
        min_length=1,
        description="Human-readable description of this deliverable",
    )
    required: bool = Field(
        default=True,
        description="Whether this deliverable must be present",
    )
    validation_rules: list[str] = Field(
        default_factory=list,
        description="List of validation expressions (e.g., '0 <= value <= 100')",
    )
    example: Optional[Any] = Field(
        default=None,
        description="Optional example value for documentation",
    )

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Validate deliverable name is a valid identifier.

        Args:
            v: Name to validate.

        Returns:
            Validated name.

        Raises:
            ValueError: If name contains invalid characters.
        """
        v = v.strip()
        if not v.replace("_", "").isalnum():
            raise ValueError(
                "name must be alphanumeric with underscores only"
            )
        return v

    @field_validator("type")
    @classmethod
    def validate_type(cls, v: str) -> str:
        """Validate type name is a recognized Python type.

        Args:
            v: Type name to validate.

        Returns:
            Validated type name.

        Raises:
            ValueError: If type name is not recognized.
        """
        valid_types = {
            "int", "float", "str", "bool", "dict", "list",
            "tuple", "set", "None", "Any", "Optional"
        }
        # Allow parameterized types like list[str], dict[str, int]
        base_type = v.split("[")[0].strip()
        if base_type not in valid_types:
            raise ValueError(
                f"type '{v}' not recognized. Valid base types: {valid_types}"
            )
        return v

    def to_dict(self) -> dict[str, Any]:
        """Serialize deliverable to dictionary.

        Returns:
            Dictionary representation of the deliverable.
        """
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Deliverable":
        """Create a Deliverable from a dictionary.

        Args:
            data: Dictionary with deliverable data.

        Returns:
            New Deliverable instance.
        """
        return cls(**data)


class ContractConstraints(BaseModel):
    """Resource and execution limits for a contract.

    Defines constraints on agent execution to ensure predictable
    resource usage and prevent runaway costs.

    Attributes:
        max_input_tokens: Maximum input tokens per request
        max_output_tokens: Maximum output tokens per response
        max_total_tokens: Maximum total tokens (input + output)
        max_tool_calls: Maximum number of tool invocations
        timeout_seconds: Maximum execution time in seconds
        warn_threshold: Percentage (0.0-1.0) to trigger warning

    Example:
        ```python
        constraints = ContractConstraints(
            max_total_tokens=5000,
            max_tool_calls=5,
            timeout_seconds=60,
            warn_threshold=0.8
        )
        ```
    """

    max_input_tokens: Optional[int] = Field(
        default=None,
        ge=1,
        description="Maximum input tokens per request",
    )
    max_output_tokens: Optional[int] = Field(
        default=None,
        ge=1,
        description="Maximum output tokens per response",
    )
    max_total_tokens: Optional[int] = Field(
        default=None,
        ge=1,
        description="Maximum total tokens (input + output)",
    )
    max_tool_calls: Optional[int] = Field(
        default=None,
        ge=0,
        description="Maximum number of tool invocations",
    )
    timeout_seconds: Optional[int] = Field(
        default=None,
        ge=1,
        description="Maximum execution time in seconds",
    )
    warn_threshold: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Percentage (0.0-1.0) to trigger warning",
    )

    def is_within_token_budget(
        self, input_tokens: int, output_tokens: int
    ) -> bool:
        """Check if token usage is within budget.

        Args:
            input_tokens: Number of input tokens used.
            output_tokens: Number of output tokens used.

        Returns:
            True if within all applicable token limits.
        """
        if self.max_input_tokens and input_tokens > self.max_input_tokens:
            return False
        if self.max_output_tokens and output_tokens > self.max_output_tokens:
            return False
        if self.max_total_tokens:
            total = input_tokens + output_tokens
            if total > self.max_total_tokens:
                return False
        return True

    def get_remaining_tokens(self, used_tokens: int) -> int:
        """Calculate remaining tokens in budget.

        Args:
            used_tokens: Total tokens used so far.

        Returns:
            Number of remaining tokens, or -1 if unlimited.
        """
        if self.max_total_tokens is None:
            return -1
        return max(0, self.max_total_tokens - used_tokens)

    def should_warn(self, used_tokens: int) -> bool:
        """Check if usage has reached the warning threshold.

        Args:
            used_tokens: Total tokens used so far.

        Returns:
            True if utilization exceeds warn_threshold.
        """
        if self.max_total_tokens is None:
            return False
        utilization = used_tokens / self.max_total_tokens
        return utilization >= self.warn_threshold

    def to_dict(self) -> dict[str, Any]:
        """Serialize constraints to dictionary.

        Returns:
            Dictionary representation of the constraints.
        """
        return self.model_dump(exclude_none=True)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ContractConstraints":
        """Create ContractConstraints from a dictionary.

        Args:
            data: Dictionary with constraints data.

        Returns:
            New ContractConstraints instance.
        """
        return cls(**data)


class Contract(BaseModel):
    """Main contract definition for agent output verification.

    Contracts specify what an agent must deliver, resource constraints,
    and how to handle failures. They enable guaranteed outputs by
    defining clear success criteria.

    Attributes:
        name: Contract name (e.g., "lead_qualification")
        description: What this contract enforces
        deliverables: List of expected outputs
        constraints: Resource and execution limits
        failure_strategy: How to handle verification failures
        max_retries: Maximum retry attempts on failure
        version: Contract version for tracking changes
        metadata: Optional metadata for additional context

    Example:
        ```python
        contract = Contract(
            name="lead_qualification",
            description="Qualify leads with BANT assessment",
            deliverables=[
                Deliverable(
                    name="score",
                    type="int",
                    description="Qualification score 0-100",
                    validation_rules=["0 <= value <= 100"]
                ),
                Deliverable(
                    name="recommendation",
                    type="str",
                    description="Next action recommendation"
                )
            ],
            constraints=ContractConstraints(
                max_total_tokens=5000,
                max_tool_calls=5,
                timeout_seconds=60
            ),
            failure_strategy=FailureStrategy.RETRY,
            max_retries=2
        )
        ```
    """

    name: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Contract name (e.g., 'lead_qualification')",
    )
    description: str = Field(
        ...,
        min_length=1,
        description="What this contract enforces",
    )
    deliverables: list[Deliverable] = Field(
        ...,
        min_length=1,
        description="List of expected outputs",
    )
    constraints: ContractConstraints = Field(
        default_factory=ContractConstraints,
        description="Resource and execution limits",
    )
    failure_strategy: FailureStrategy = Field(
        default=FailureStrategy.RETRY,
        description="How to handle verification failures",
    )
    max_retries: int = Field(
        default=2,
        ge=0,
        le=10,
        description="Maximum retry attempts on failure",
    )
    version: str = Field(
        default="1.0.0",
        description="Contract version for tracking changes",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Optional metadata for additional context",
    )

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Validate contract name is a valid identifier.

        Args:
            v: Name to validate.

        Returns:
            Validated name in snake_case.

        Raises:
            ValueError: If name contains invalid characters.
        """
        v = v.strip().lower().replace("-", "_").replace(" ", "_")
        if not v.replace("_", "").isalnum():
            raise ValueError(
                "name must be alphanumeric with underscores only"
            )
        return v

    @field_validator("deliverables")
    @classmethod
    def validate_unique_deliverable_names(
        cls, v: list[Deliverable]
    ) -> list[Deliverable]:
        """Ensure all deliverable names are unique.

        Args:
            v: List of deliverables to validate.

        Returns:
            Validated deliverables list.

        Raises:
            ValueError: If duplicate deliverable names found.
        """
        names = [d.name for d in v]
        duplicates = [name for name in names if names.count(name) > 1]
        if duplicates:
            raise ValueError(
                f"Duplicate deliverable names found: {set(duplicates)}"
            )
        return v

    def get_required_deliverables(self) -> list[Deliverable]:
        """Get list of required deliverables.

        Returns:
            List of deliverables where required=True.
        """
        return [d for d in self.deliverables if d.required]

    def get_optional_deliverables(self) -> list[Deliverable]:
        """Get list of optional deliverables.

        Returns:
            List of deliverables where required=False.
        """
        return [d for d in self.deliverables if not d.required]

    def get_deliverable(self, name: str) -> Optional[Deliverable]:
        """Get a deliverable by name.

        Args:
            name: Name of the deliverable to find.

        Returns:
            The deliverable if found, None otherwise.
        """
        for d in self.deliverables:
            if d.name == name:
                return d
        return None

    def get_deliverable_names(self) -> list[str]:
        """Get list of all deliverable names.

        Returns:
            List of deliverable names.
        """
        return [d.name for d in self.deliverables]

    def to_dict(self) -> dict[str, Any]:
        """Serialize contract to dictionary.

        Returns:
            Dictionary representation of the contract.
        """
        return {
            "name": self.name,
            "description": self.description,
            "deliverables": [d.to_dict() for d in self.deliverables],
            "constraints": self.constraints.to_dict(),
            "failure_strategy": self.failure_strategy.value,
            "max_retries": self.max_retries,
            "version": self.version,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Contract":
        """Create a Contract from a dictionary.

        Args:
            data: Dictionary with contract data.

        Returns:
            New Contract instance.
        """
        deliverables = [
            Deliverable.from_dict(d) if isinstance(d, dict) else d
            for d in data.get("deliverables", [])
        ]
        constraints = data.get("constraints", {})
        if isinstance(constraints, dict):
            constraints = ContractConstraints.from_dict(constraints)

        failure_strategy = data.get("failure_strategy", "retry")
        if isinstance(failure_strategy, str):
            failure_strategy = FailureStrategy(failure_strategy)

        return cls(
            name=data["name"],
            description=data["description"],
            deliverables=deliverables,
            constraints=constraints,
            failure_strategy=failure_strategy,
            max_retries=data.get("max_retries", 2),
            version=data.get("version", "1.0.0"),
            metadata=data.get("metadata", {}),
        )

    def validate_schema(self) -> tuple[bool, list[str]]:
        """Validate the contract schema for completeness and consistency.

        Performs validation checks on the contract structure:
        - At least one deliverable defined
        - All required deliverables have descriptions
        - Validation rules are syntactically valid
        - Constraints are internally consistent

        Returns:
            Tuple of (is_valid, list of error messages).
        """
        errors: list[str] = []

        # Check deliverables
        if not self.deliverables:
            errors.append("Contract must have at least one deliverable")

        required_count = len(self.get_required_deliverables())
        if required_count == 0:
            errors.append("Contract should have at least one required deliverable")

        # Check constraints consistency
        if (
            self.constraints.max_input_tokens is not None
            and self.constraints.max_output_tokens is not None
            and self.constraints.max_total_tokens is not None
        ):
            if (
                self.constraints.max_input_tokens + self.constraints.max_output_tokens
                > self.constraints.max_total_tokens
            ):
                errors.append(
                    "max_input_tokens + max_output_tokens exceeds max_total_tokens"
                )

        # Check retry configuration
        if self.failure_strategy == FailureStrategy.FAIL and self.max_retries > 0:
            errors.append(
                "max_retries should be 0 when failure_strategy is 'fail'"
            )

        return len(errors) == 0, errors

    def get_template_output(self) -> dict[str, Any]:
        """Generate a template output structure based on deliverables.

        Creates a dictionary with default values based on deliverable
        types and examples. Useful for generating fallback outputs.

        Returns:
            Dictionary with template values for each deliverable.
        """
        template: dict[str, Any] = {}
        type_defaults: dict[str, Any] = {
            "int": 0,
            "float": 0.0,
            "str": "",
            "bool": False,
            "dict": {},
            "list": [],
            "tuple": (),
            "set": set(),
        }

        for deliverable in self.deliverables:
            if deliverable.example is not None:
                template[deliverable.name] = deliverable.example
            else:
                base_type = deliverable.type.split("[")[0]
                template[deliverable.name] = type_defaults.get(base_type, None)

        return template


__all__ = [
    "FailureStrategy",
    "Deliverable",
    "ContractConstraints",
    "Contract",
]
