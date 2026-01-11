"""Enhanced planning schemas for Phase 5 Planning & Reasoning.

This module extends the base plan schemas with additional fields required
for the Phase 5 planning and reasoning integration. It provides data models
for plan generation, execution tracking, and result reporting.

Classes:
    StepType: Enumeration of plan step types
    PlanStepConfig: Configuration for individual plan steps
    PlanMetadata: Metadata about plan complexity and estimates
    PlanConstraints: Constraints for plan generation
    ExecutionCheckpoint: Checkpoint for plan execution resumption
    StepResult: Result from step execution
    PlanResult: Complete execution result
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


# =============================================================================
# Constants
# =============================================================================

DEFAULT_MAX_STEPS = 20
"""Default maximum number of steps in a plan."""

DEFAULT_STEP_TIMEOUT = 60
"""Default step execution timeout in seconds."""

DEFAULT_MAX_CONCURRENT = 3
"""Default maximum concurrent step executions."""

DEFAULT_MAX_RETRIES = 3
"""Default maximum retries per step."""


def generate_uuid() -> str:
    """Generate a UUID4 hex string for use as an identifier."""
    return uuid.uuid4().hex


def utc_now() -> datetime:
    """Get the current UTC datetime."""
    return datetime.now(timezone.utc)


# =============================================================================
# Enumerations
# =============================================================================


class StepType(str, Enum):
    """Types of plan steps.

    Attributes:
        TOOL_CALL: Execute a tool
        REASONING: Use reasoning strategy
        PARALLEL_GROUP: Group of parallel steps
        CONDITIONAL: Conditional branching
        MEMORY_QUERY: Query memory system
    """

    TOOL_CALL = "tool_call"
    REASONING = "reasoning"
    PARALLEL_GROUP = "parallel_group"
    CONDITIONAL = "conditional"
    MEMORY_QUERY = "memory_query"


class StepStatus(str, Enum):
    """Step execution status.

    Attributes:
        PENDING: Not yet started
        RUNNING: Currently executing
        COMPLETED: Successfully finished
        FAILED: Execution failed
        SKIPPED: Step was skipped
        FAILED_CONTRACT: Failed contract validation
    """

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    FAILED_CONTRACT = "failed_contract"

    def is_terminal(self) -> bool:
        """Check if this status is a terminal state."""
        return self in (
            StepStatus.COMPLETED,
            StepStatus.FAILED,
            StepStatus.SKIPPED,
            StepStatus.FAILED_CONTRACT,
        )


class ExecutionState(str, Enum):
    """Plan execution state.

    Attributes:
        NOT_STARTED: Plan has not started execution
        RUNNING: Plan is currently executing
        PAUSED: Plan execution is paused
        COMPLETED: Plan completed successfully
        FAILED: Plan execution failed
        ABORTED: Plan was aborted
    """

    NOT_STARTED = "not_started"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    ABORTED = "aborted"


# =============================================================================
# Plan Step Configuration
# =============================================================================


class PlanStepConfig(BaseModel):
    """Configuration for a single plan step.

    This extends the basic PlanStep with additional fields for Phase 5
    execution including tool configuration, reasoning hints, and
    contract specifications.

    Attributes:
        step_id: Unique step identifier
        description: Human-readable step description
        step_type: Type of step execution
        dependencies: Step IDs that must complete first
        tool_name: Tool to call (if step_type=TOOL_CALL)
        tool_args: Arguments for tool (if step_type=TOOL_CALL)
        reasoning_task: Task description (if step_type=REASONING)
        strategy_hint: Suggested reasoning strategy
        estimated_tokens: Estimated token cost
        timeout_seconds: Step timeout
        retry_on_failure: Whether to retry on failure
        max_retries: Maximum retry attempts
    """

    step_id: str = Field(
        default_factory=generate_uuid,
        description="Unique step identifier",
    )
    description: str = Field(
        ...,
        min_length=1,
        description="Human-readable step description",
    )
    step_type: StepType = Field(
        default=StepType.REASONING,
        description="Type of step execution",
    )
    dependencies: list[str] = Field(
        default_factory=list,
        description="Step IDs that must complete first",
    )
    tool_name: Optional[str] = Field(
        default=None,
        description="Tool to call (if step_type=TOOL_CALL)",
    )
    tool_args: Optional[dict[str, Any]] = Field(
        default=None,
        description="Arguments for tool",
    )
    reasoning_task: Optional[str] = Field(
        default=None,
        description="Task description for reasoning",
    )
    strategy_hint: Optional[str] = Field(
        default=None,
        description="Suggested reasoning strategy",
    )
    estimated_tokens: int = Field(
        default=500,
        ge=0,
        description="Estimated token cost",
    )
    timeout_seconds: int = Field(
        default=DEFAULT_STEP_TIMEOUT,
        ge=1,
        description="Step timeout in seconds",
    )
    retry_on_failure: bool = Field(
        default=True,
        description="Whether to retry on failure",
    )
    max_retries: int = Field(
        default=DEFAULT_MAX_RETRIES,
        ge=0,
        description="Maximum retry attempts",
    )

    def model_post_init(self, __context: Any) -> None:
        """Validate step configuration after initialization."""
        if self.step_type == StepType.TOOL_CALL and not self.tool_name:
            raise ValueError("tool_name is required for TOOL_CALL steps")
        if self.step_type == StepType.REASONING and not self.reasoning_task:
            # Use description as reasoning task if not specified
            self.reasoning_task = self.description


# =============================================================================
# Plan Metadata
# =============================================================================


class PlanMetadata(BaseModel):
    """Metadata about plan complexity and estimates.

    Attributes:
        complexity: Overall complexity score (0.0-1.0)
        estimated_total_tokens: Estimated total token cost
        estimated_duration_seconds: Estimated execution time
        parallel_groups: Number of parallelizable step groups
        tool_calls: Number of tool call steps
        reasoning_steps: Number of reasoning steps
    """

    complexity: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Overall complexity score",
    )
    estimated_total_tokens: int = Field(
        default=0,
        ge=0,
        description="Estimated total token cost",
    )
    estimated_duration_seconds: float = Field(
        default=0.0,
        ge=0.0,
        description="Estimated execution time",
    )
    parallel_groups: int = Field(
        default=1,
        ge=1,
        description="Number of parallelizable groups",
    )
    tool_calls: int = Field(
        default=0,
        ge=0,
        description="Number of tool call steps",
    )
    reasoning_steps: int = Field(
        default=0,
        ge=0,
        description="Number of reasoning steps",
    )


# =============================================================================
# Plan Constraints
# =============================================================================


class PlanConstraints(BaseModel):
    """Constraints for plan generation.

    Attributes:
        max_steps: Maximum number of steps
        max_parallel: Maximum parallel step executions
        allowed_tools: Tools that can be used (None = all)
        forbidden_tools: Tools that cannot be used
        max_tokens: Maximum total token budget
        prefer_parallel: Whether to prefer parallel execution
        max_depth: Maximum dependency chain depth
    """

    max_steps: int = Field(
        default=DEFAULT_MAX_STEPS,
        ge=1,
        le=100,
        description="Maximum number of steps",
    )
    max_parallel: int = Field(
        default=5,
        ge=1,
        description="Maximum parallel executions",
    )
    allowed_tools: Optional[list[str]] = Field(
        default=None,
        description="Allowed tools (None = all)",
    )
    forbidden_tools: Optional[list[str]] = Field(
        default=None,
        description="Forbidden tools",
    )
    max_tokens: int = Field(
        default=10000,
        ge=100,
        description="Maximum token budget",
    )
    prefer_parallel: bool = Field(
        default=True,
        description="Prefer parallel execution",
    )
    max_depth: int = Field(
        default=10,
        ge=1,
        description="Maximum dependency depth",
    )


# =============================================================================
# Step Result
# =============================================================================


@dataclass
class StepResult:
    """Result from step execution.

    Attributes:
        step_id: Step identifier
        status: Execution status
        output: Step output
        tokens_used: Tokens consumed
        duration_ms: Execution time in milliseconds
        retries: Number of retries attempted
        error: Error message if failed
        started_at: Execution start time
        completed_at: Execution completion time
    """

    step_id: str
    status: StepStatus
    output: Any = None
    tokens_used: int = 0
    duration_ms: float = 0.0
    retries: int = 0
    error: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    @classmethod
    def from_exception(cls, step_id: str, exception: Exception) -> "StepResult":
        """Create a failed result from an exception."""
        return cls(
            step_id=step_id,
            status=StepStatus.FAILED,
            error=str(exception),
            completed_at=utc_now(),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "step_id": self.step_id,
            "status": self.status.value,
            "output": self.output if not callable(self.output) else str(self.output),
            "tokens_used": self.tokens_used,
            "duration_ms": self.duration_ms,
            "retries": self.retries,
            "error": self.error,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
        }


# =============================================================================
# Execution Checkpoint
# =============================================================================


@dataclass
class ExecutionCheckpoint:
    """Checkpoint for plan execution resumption.

    Allows plans to be paused and resumed by storing execution state.

    Attributes:
        checkpoint_id: Unique checkpoint identifier
        plan_id: ID of the plan being executed
        completed_steps: List of completed step IDs
        step_results: Results for completed steps
        current_step: Currently executing step (if any)
        created_at: Checkpoint creation time
        tokens_used: Total tokens used so far
    """

    checkpoint_id: str
    plan_id: str
    completed_steps: list[str]
    step_results: dict[str, StepResult]
    current_step: Optional[str] = None
    created_at: datetime = field(default_factory=utc_now)
    tokens_used: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert checkpoint to dictionary for serialization."""
        return {
            "checkpoint_id": self.checkpoint_id,
            "plan_id": self.plan_id,
            "completed_steps": self.completed_steps,
            "step_results": {k: v.to_dict() for k, v in self.step_results.items()},
            "current_step": self.current_step,
            "created_at": self.created_at.isoformat(),
            "tokens_used": self.tokens_used,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExecutionCheckpoint":
        """Create checkpoint from dictionary."""
        step_results = {}
        for step_id, result_data in data.get("step_results", {}).items():
            step_results[step_id] = StepResult(
                step_id=result_data["step_id"],
                status=StepStatus(result_data["status"]),
                output=result_data.get("output"),
                tokens_used=result_data.get("tokens_used", 0),
                duration_ms=result_data.get("duration_ms", 0.0),
                retries=result_data.get("retries", 0),
                error=result_data.get("error"),
            )

        return cls(
            checkpoint_id=data["checkpoint_id"],
            plan_id=data["plan_id"],
            completed_steps=data["completed_steps"],
            step_results=step_results,
            current_step=data.get("current_step"),
            created_at=datetime.fromisoformat(data["created_at"]),
            tokens_used=data.get("tokens_used", 0),
        )


# =============================================================================
# Plan Result
# =============================================================================


@dataclass
class PlanResult:
    """Complete result from plan execution.

    Attributes:
        plan_id: ID of the executed plan
        success: Whether all steps succeeded
        state: Final execution state
        step_results: Results for all steps
        total_tokens: Total tokens consumed
        total_duration_ms: Total execution time
        final_output: Aggregated final output
        checkpoints: Checkpoints created during execution
        started_at: Execution start time
        completed_at: Execution completion time
        errors: List of errors encountered
    """

    plan_id: str
    success: bool
    state: ExecutionState
    step_results: list[StepResult]
    total_tokens: int = 0
    total_duration_ms: float = 0.0
    final_output: Optional[str] = None
    checkpoints: list[str] = field(default_factory=list)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    errors: list[str] = field(default_factory=list)

    @classmethod
    def from_step_results(
        cls,
        plan_id: str,
        step_results: dict[str, StepResult],
    ) -> "PlanResult":
        """Create PlanResult from step results dictionary."""
        results_list = list(step_results.values())
        success = all(r.status == StepStatus.COMPLETED for r in results_list)
        total_tokens = sum(r.tokens_used for r in results_list)
        total_duration = sum(r.duration_ms for r in results_list)
        errors = [r.error for r in results_list if r.error]

        # Aggregate output from completed steps
        outputs = [
            str(r.output) for r in results_list
            if r.status == StepStatus.COMPLETED and r.output
        ]
        final_output = "\n".join(outputs) if outputs else None

        return cls(
            plan_id=plan_id,
            success=success,
            state=ExecutionState.COMPLETED if success else ExecutionState.FAILED,
            step_results=results_list,
            total_tokens=total_tokens,
            total_duration_ms=total_duration,
            final_output=final_output,
            errors=errors,
            completed_at=utc_now(),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "plan_id": self.plan_id,
            "success": self.success,
            "state": self.state.value,
            "step_results": [r.to_dict() for r in self.step_results],
            "total_tokens": self.total_tokens,
            "total_duration_ms": self.total_duration_ms,
            "final_output": self.final_output,
            "checkpoints": self.checkpoints,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "errors": self.errors,
        }


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Constants
    "DEFAULT_MAX_STEPS",
    "DEFAULT_STEP_TIMEOUT",
    "DEFAULT_MAX_CONCURRENT",
    "DEFAULT_MAX_RETRIES",
    # Utilities
    "generate_uuid",
    "utc_now",
    # Enumerations
    "StepType",
    "StepStatus",
    "ExecutionState",
    # Configuration
    "PlanStepConfig",
    "PlanMetadata",
    "PlanConstraints",
    # Results
    "StepResult",
    "ExecutionCheckpoint",
    "PlanResult",
]
