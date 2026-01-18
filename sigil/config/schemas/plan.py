"""Planning and task decomposition schemas for Sigil v2.

This module defines Pydantic models for the planning subsystem, which
handles task decomposition, step sequencing, and execution tracking.

Classes:
    PlanStatus: Enumeration of plan/step execution states
    PlanStep: Individual step in an execution plan
    Plan: Complete execution plan for a task
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


def generate_uuid() -> str:
    """Generate a UUID4 hex string for use as an identifier.

    Returns:
        A 32-character hex string UUID.
    """
    return uuid.uuid4().hex


def utc_now() -> datetime:
    """Get the current UTC datetime.

    Returns:
        Current datetime with UTC timezone.
    """
    return datetime.now(timezone.utc)


class PlanStatus(str, Enum):
    """Enumeration of plan/step execution states.

    States follow a linear progression with failure as an alternative
    terminal state:
        pending -> in_progress -> completed
                       |
                       v
                    failed

    Attributes:
        PENDING: Not yet started
        IN_PROGRESS: Currently executing
        COMPLETED: Successfully finished
        FAILED: Execution failed
    """

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"

    def is_terminal(self) -> bool:
        """Check if this status is a terminal state.

        Returns:
            True if status is COMPLETED or FAILED.
        """
        return self in (PlanStatus.COMPLETED, PlanStatus.FAILED)

    def can_transition_to(self, target: "PlanStatus") -> bool:
        """Check if transition to target status is valid.

        Valid transitions:
        - pending -> in_progress
        - in_progress -> completed
        - in_progress -> failed
        - pending -> failed (skip execution)

        Args:
            target: The target status to transition to.

        Returns:
            True if the transition is valid.
        """
        valid_transitions = {
            PlanStatus.PENDING: {PlanStatus.IN_PROGRESS, PlanStatus.FAILED},
            PlanStatus.IN_PROGRESS: {PlanStatus.COMPLETED, PlanStatus.FAILED},
            PlanStatus.COMPLETED: set(),  # Terminal
            PlanStatus.FAILED: set(),  # Terminal
        }
        return target in valid_transitions.get(self, set())


class PlanStep(BaseModel):
    """Individual step in an execution plan.

    Each step represents a discrete action to be executed as part
    of a larger plan. Steps can have dependencies on other steps
    and track their own execution status.

    Attributes:
        step_id: Unique step identifier within the plan
        description: Human-readable description of what this step does
        status: Current execution status
        dependencies: List of step IDs that must complete first
        tool_calls: Optional list of tools this step may use
        result: Output from execution (populated after completion)

    Example:
        ```python
        step = PlanStep(
            description="Retrieve customer information from CRM",
            dependencies=[],
            tool_calls=["crm.get_contact", "crm.get_history"]
        )
        ```
    """

    step_id: str = Field(
        default_factory=generate_uuid,
        description="Unique step identifier within the plan",
    )
    description: str = Field(
        ...,
        min_length=1,
        description="Human-readable description of what this step does",
    )
    status: PlanStatus = Field(
        default=PlanStatus.PENDING,
        description="Current execution status",
    )
    dependencies: list[str] = Field(
        default_factory=list,
        description="List of step IDs that must complete before this step",
    )
    tool_calls: Optional[list[str]] = Field(
        default=None,
        description="Optional list of tools this step may use",
    )
    tool_args: Optional[dict[str, Any]] = Field(
        default=None,
        description="Arguments for the tool (if step uses tools)",
    )
    result: Optional[str] = Field(
        default=None,
        description="Output from execution (populated after completion)",
    )

    def is_ready(self, completed_step_ids: set[str]) -> bool:
        """Check if this step is ready to execute.

        A step is ready when all its dependencies are in the
        completed_step_ids set and the step is still pending.

        Args:
            completed_step_ids: Set of step IDs that have completed.

        Returns:
            True if all dependencies are met and step is pending.
        """
        if self.status != PlanStatus.PENDING:
            return False
        return all(dep_id in completed_step_ids for dep_id in self.dependencies)

    def start(self) -> None:
        """Mark this step as in progress.

        Raises:
            ValueError: If transition from current status is invalid.
        """
        if not self.status.can_transition_to(PlanStatus.IN_PROGRESS):
            raise ValueError(
                f"Cannot start step: invalid transition from {self.status.value}"
            )
        self.status = PlanStatus.IN_PROGRESS

    def complete(self, result: str) -> None:
        """Mark this step as completed with a result.

        Args:
            result: The output from executing this step.

        Raises:
            ValueError: If transition from current status is invalid.
        """
        if not self.status.can_transition_to(PlanStatus.COMPLETED):
            raise ValueError(
                f"Cannot complete step: invalid transition from {self.status.value}"
            )
        self.status = PlanStatus.COMPLETED
        self.result = result

    def fail(self, error: str) -> None:
        """Mark this step as failed with an error message.

        Args:
            error: The error message describing the failure.

        Raises:
            ValueError: If transition from current status is invalid.
        """
        if not self.status.can_transition_to(PlanStatus.FAILED):
            raise ValueError(
                f"Cannot fail step: invalid transition from {self.status.value}"
            )
        self.status = PlanStatus.FAILED
        self.result = f"FAILED: {error}"


class Plan(BaseModel):
    """Complete execution plan for a task.

    A plan represents the decomposition of a high-level goal into
    a sequence of executable steps. Plans track dependencies between
    steps and overall execution progress.

    Attributes:
        plan_id: Unique plan identifier
        goal: High-level goal this plan achieves
        steps: Ordered list of plan steps
        created_at: Plan creation timestamp (UTC)
        completed_at: Plan completion timestamp (UTC, None if not completed)
        status: Current plan status

    Example:
        ```python
        plan = Plan(
            goal="Qualify lead John from Acme Corp",
            steps=[
                PlanStep(description="Retrieve existing information from memory"),
                PlanStep(description="Identify BANT gaps", dependencies=["step_0"]),
                PlanStep(description="Formulate discovery questions", dependencies=["step_1"]),
            ]
        )
        ```
    """

    plan_id: str = Field(
        default_factory=generate_uuid,
        description="Unique plan identifier",
    )
    goal: str = Field(
        ...,
        min_length=1,
        description="High-level goal this plan achieves",
    )
    steps: list[PlanStep] = Field(
        default_factory=list,
        description="Ordered list of plan steps",
    )
    created_at: datetime = Field(
        default_factory=utc_now,
        description="Plan creation timestamp (UTC)",
    )
    completed_at: Optional[datetime] = Field(
        default=None,
        description="Plan completion timestamp (UTC)",
    )
    status: PlanStatus = Field(
        default=PlanStatus.PENDING,
        description="Current plan status",
    )

    @model_validator(mode="after")
    def validate_dependencies(self) -> "Plan":
        """Validate that all step dependencies reference existing steps.

        Returns:
            The validated Plan instance.

        Raises:
            ValueError: If any dependency references a non-existent step.
        """
        step_ids = {step.step_id for step in self.steps}
        for step in self.steps:
            invalid_deps = [dep for dep in step.dependencies if dep not in step_ids]
            if invalid_deps:
                raise ValueError(
                    f"Step '{step.step_id}' has invalid dependencies: {invalid_deps}"
                )
        return self

    def get_ready_steps(self) -> list[PlanStep]:
        """Get steps that are ready to execute.

        Returns steps whose dependencies are all completed and
        which are still in pending status.

        Returns:
            List of steps ready for execution.
        """
        completed_ids = {
            step.step_id
            for step in self.steps
            if step.status == PlanStatus.COMPLETED
        }
        return [step for step in self.steps if step.is_ready(completed_ids)]

    def get_progress(self) -> float:
        """Get plan completion progress as a percentage.

        Returns:
            Percentage of steps completed (0.0 to 1.0).
        """
        if not self.steps:
            return 1.0  # Empty plan is "complete"
        completed_count = sum(
            1 for step in self.steps if step.status == PlanStatus.COMPLETED
        )
        return completed_count / len(self.steps)

    def update_status(self) -> None:
        """Update plan status based on step statuses.

        Sets plan status to:
        - COMPLETED if all steps are completed
        - FAILED if any step has failed
        - IN_PROGRESS if any step is in progress
        - PENDING otherwise
        """
        if not self.steps:
            self.status = PlanStatus.COMPLETED
            self.completed_at = utc_now()
            return

        # Check for any failures
        if any(step.status == PlanStatus.FAILED for step in self.steps):
            self.status = PlanStatus.FAILED
            self.completed_at = utc_now()
            return

        # Check if all completed
        if all(step.status == PlanStatus.COMPLETED for step in self.steps):
            self.status = PlanStatus.COMPLETED
            self.completed_at = utc_now()
            return

        # Check if any in progress
        if any(step.status == PlanStatus.IN_PROGRESS for step in self.steps):
            self.status = PlanStatus.IN_PROGRESS
            return

        # Otherwise still pending
        self.status = PlanStatus.PENDING

    @field_validator("steps")
    @classmethod
    def validate_steps_non_empty(cls, v: list[PlanStep]) -> list[PlanStep]:
        """Allow empty steps list but warn in documentation.

        An empty plan is technically valid but should be avoided.

        Args:
            v: List of plan steps.

        Returns:
            The validated steps list.
        """
        return v


__all__ = [
    "PlanStatus",
    "PlanStep",
    "Plan",
]
