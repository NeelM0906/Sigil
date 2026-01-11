"""Shared fixtures for planning module tests."""

import asyncio
import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock, patch

from sigil.config.schemas.plan import Plan, PlanStep, PlanStatus
from sigil.planning.schemas import (
    StepType,
    StepStatus,
    ExecutionState,
    PlanStepConfig,
    PlanMetadata,
    PlanConstraints,
    StepResult,
    ExecutionCheckpoint,
    PlanResult,
    generate_uuid,
    utc_now,
)
from sigil.planning.planner import Planner, PlanningError, DAGValidationError
from sigil.planning.executor import PlanExecutor, ExecutionContext
from sigil.state.store import EventStore
from sigil.telemetry.tokens import TokenTracker


@pytest.fixture
def temp_storage_dir():
    """Create a temporary storage directory for tests."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def event_store(temp_storage_dir):
    """Create an EventStore for tests."""
    return EventStore(storage_dir=temp_storage_dir + "/events")


@pytest.fixture
def token_tracker():
    """Create a TokenTracker for tests."""
    return TokenTracker()


@pytest.fixture
def plan_constraints():
    """Create default PlanConstraints for tests."""
    return PlanConstraints(
        max_steps=10,
        max_parallel=3,
        max_tokens=5000,
        prefer_parallel=True,
        max_depth=5,
    )


@pytest.fixture
def planner(event_store, token_tracker, plan_constraints):
    """Create a Planner instance for tests."""
    return Planner(
        event_store=event_store,
        token_tracker=token_tracker,
        constraints=plan_constraints,
        cache_ttl_hours=1,
    )


@pytest.fixture
def plan_executor(event_store, token_tracker):
    """Create a PlanExecutor instance for tests."""
    return PlanExecutor(
        event_store=event_store,
        token_tracker=token_tracker,
        max_concurrent=3,
        max_retries=3,
    )


@pytest.fixture
def simple_plan():
    """Create a simple test plan with no dependencies."""
    return Plan(
        plan_id=generate_uuid(),
        goal="Test goal",
        steps=[
            PlanStep(
                step_id="step-1",
                description="First step",
                dependencies=[],
            ),
            PlanStep(
                step_id="step-2",
                description="Second step",
                dependencies=["step-1"],
            ),
            PlanStep(
                step_id="step-3",
                description="Third step",
                dependencies=["step-2"],
            ),
        ],
        status=PlanStatus.PENDING,
    )


@pytest.fixture
def parallel_plan():
    """Create a plan with parallel steps."""
    return Plan(
        plan_id=generate_uuid(),
        goal="Test parallel execution",
        steps=[
            PlanStep(
                step_id="step-1",
                description="Initial step",
                dependencies=[],
            ),
            PlanStep(
                step_id="step-2a",
                description="Parallel step A",
                dependencies=["step-1"],
            ),
            PlanStep(
                step_id="step-2b",
                description="Parallel step B",
                dependencies=["step-1"],
            ),
            PlanStep(
                step_id="step-2c",
                description="Parallel step C",
                dependencies=["step-1"],
            ),
            PlanStep(
                step_id="step-3",
                description="Final step",
                dependencies=["step-2a", "step-2b", "step-2c"],
            ),
        ],
        status=PlanStatus.PENDING,
    )


@pytest.fixture
def cyclic_plan():
    """Create a plan with circular dependencies (invalid)."""
    return Plan(
        plan_id=generate_uuid(),
        goal="Test cyclic detection",
        steps=[
            PlanStep(
                step_id="step-a",
                description="Step A",
                dependencies=["step-c"],
            ),
            PlanStep(
                step_id="step-b",
                description="Step B",
                dependencies=["step-a"],
            ),
            PlanStep(
                step_id="step-c",
                description="Step C",
                dependencies=["step-b"],
            ),
        ],
        status=PlanStatus.PENDING,
    )


@pytest.fixture
def mock_step_executor():
    """Create a mock step executor function."""
    async def executor(step: PlanStep, prior_results: dict) -> StepResult:
        await asyncio.sleep(0.01)  # Simulate some work
        return StepResult(
            step_id=step.step_id,
            status=StepStatus.COMPLETED,
            output=f"Completed: {step.description}",
            tokens_used=100,
            duration_ms=10.0,
            started_at=utc_now(),
            completed_at=utc_now(),
        )
    return executor


@pytest.fixture
def failing_step_executor():
    """Create a step executor that always fails."""
    async def executor(step: PlanStep, prior_results: dict) -> StepResult:
        raise Exception("Simulated step failure")
    return executor
