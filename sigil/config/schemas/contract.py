"""Contract and deliverable schemas for Sigil v2.

This module defines Pydantic models for the contract-based verification
system, which ensures agent outputs meet specified requirements.

Contract System Overview:
    Contracts define what an agent must deliver and how to verify it.
    Each contract specifies:
    - Deliverables: What the agent must produce
    - Constraints: Resource limits (tokens, time, retries)
    - Failure strategy: How to handle verification failures

Classes:
    FailureStrategy: Enumeration of failure handling strategies
    ContractDeliverable: Expected output specification
    ContractConstraints: Resource and execution limits
    Contract: Main contract definition
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator


class FailureStrategy(str, Enum):
    """Enumeration of failure handling strategies.

    Defines how the system should respond when contract verification
    fails after all retries are exhausted.

    Attributes:
        RETRY: Retry the operation up to max_retries times
        FALLBACK: Use a fallback response or handler
        FAIL: Fail immediately with an error
    """

    RETRY = "retry"
    FALLBACK = "fallback"
    FAIL = "fail"


class ContractDeliverable(BaseModel):
    """Expected output specification for a contract.

    Defines what an agent must produce as part of contract fulfillment.
    Each deliverable has a name, type specification, and validation rules.

    Attributes:
        name: Deliverable name (e.g., "qualification_score")
        type_name: Expected Python type as string (e.g., "int", "dict", "str")
        description: Human-readable description of this deliverable
        required: Whether this deliverable must be present (default: True)
        validation_rules: Optional list of validation expressions

    Example:
        ```python
        deliverable = ContractDeliverable(
            name="qualification_score",
            type_name="int",
            description="Lead qualification score from 0 to 100",
            required=True,
            validation_rules=["value >= 0", "value <= 100"]
        )
        ```
    """

    name: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Deliverable name (e.g., 'qualification_score')",
    )
    type_name: str = Field(
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
    validation_rules: Optional[list[str]] = Field(
        default=None,
        description="Optional list of validation expressions",
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

    @field_validator("type_name")
    @classmethod
    def validate_type_name(cls, v: str) -> str:
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
                f"type_name '{v}' not recognized. Valid base types: {valid_types}"
            )
        return v


class ContractConstraints(BaseModel):
    """Resource and execution limits for a contract.

    Defines constraints on agent execution to ensure predictable
    resource usage and prevent runaway costs.

    Attributes:
        max_input_tokens: Maximum input tokens per request
        max_output_tokens: Maximum output tokens per response
        max_total_tokens: Maximum total tokens (input + output)
        max_tool_calls: Maximum number of tool invocations
        max_retries: Maximum retry attempts on failure (default: 2)
        timeout_seconds: Maximum execution time in seconds

    Example:
        ```python
        constraints = ContractConstraints(
            max_total_tokens=5000,
            max_tool_calls=5,
            max_retries=2,
            timeout_seconds=60
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
    max_retries: int = Field(
        default=2,
        ge=0,
        le=10,
        description="Maximum retry attempts on failure",
    )
    timeout_seconds: Optional[int] = Field(
        default=None,
        ge=1,
        description="Maximum execution time in seconds",
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

    Example:
        ```python
        contract = Contract(
            name="lead_qualification",
            description="Ensures lead qualification produces valid scores and recommendations",
            deliverables=[
                ContractDeliverable(
                    name="score",
                    type_name="int",
                    description="Qualification score 0-100",
                    validation_rules=["value >= 0", "value <= 100"]
                ),
                ContractDeliverable(
                    name="recommendation",
                    type_name="str",
                    description="Next action recommendation"
                )
            ],
            constraints=ContractConstraints(
                max_total_tokens=5000,
                max_tool_calls=5,
                timeout_seconds=60
            ),
            failure_strategy=FailureStrategy.RETRY
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
    deliverables: list[ContractDeliverable] = Field(
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
        cls, v: list[ContractDeliverable]
    ) -> list[ContractDeliverable]:
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

    def get_required_deliverables(self) -> list[ContractDeliverable]:
        """Get list of required deliverables.

        Returns:
            List of deliverables where required=True.
        """
        return [d for d in self.deliverables if d.required]

    def get_optional_deliverables(self) -> list[ContractDeliverable]:
        """Get list of optional deliverables.

        Returns:
            List of deliverables where required=False.
        """
        return [d for d in self.deliverables if not d.required]


__all__ = [
    "FailureStrategy",
    "ContractDeliverable",
    "ContractConstraints",
    "Contract",
]
