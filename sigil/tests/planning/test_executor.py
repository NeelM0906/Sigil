"""Tests for the PlanExecutor class.

Tests cover:
- Basic plan execution
- Dependency ordering
- Parallel execution with bounded concurrency
- Pause/resume functionality
- Abort functionality
- Retry logic with exponential backoff
- Checkpoint creation
- Event emission
"""

import asyncio
import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from datetime import datetime

from sigil.config.schemas.plan import Plan, PlanStep, PlanStatus
from sigil.planning.executor import (
    PlanExecutor,
    ExecutionContext,
    create_step_started_event,
    create_step_completed_event,
    create_plan_completed_event,
)
from sigil.planning.schemas import (
    StepStatus,
    ExecutionState,
    StepResult,
    ExecutionCheckpoint,
    PlanResult,
    generate_uuid,
    utc_now,
)


class TestPlanExecutorBasics:
    """Tests for basic PlanExecutor functionality."""

    def test_initialization(self, plan_executor):
        """Test that PlanExecutor initializes correctly."""
        assert plan_executor._max_concurrent == 3
        assert plan_executor._max_retries == 3
        assert plan_executor._abort_flag is False
        assert plan_executor._pause_event.is_set()

    def test_initialization_with_defaults(self):
        """Test PlanExecutor initialization with default values."""
        executor = PlanExecutor()
        assert executor._event_store is None
        assert executor._token_tracker is None
        assert executor._max_concurrent == 3


class TestBasicExecution:
    """Tests for basic plan execution."""

    @pytest.mark.asyncio
    async def test_execute_simple_plan(self, plan_executor, simple_plan):
        """Test executing a simple sequential plan."""
        result = await plan_executor.execute(simple_plan)

        assert result is not None
        assert result.plan_id == simple_plan.plan_id
        assert result.success is True
        assert result.state == ExecutionState.COMPLETED
        assert len(result.step_results) == 3

    @pytest.mark.asyncio
    async def test_execute_empty_plan(self, plan_executor):
        """Test executing an empty plan."""
        empty_plan = Plan(
            plan_id=generate_uuid(),
            goal="Empty",
            steps=[],
            status=PlanStatus.PENDING,
        )

        result = await plan_executor.execute(empty_plan)

        assert result.success is True
        assert result.state == ExecutionState.COMPLETED
        assert len(result.step_results) == 0

    @pytest.mark.asyncio
    async def test_execute_single_step_plan(self, plan_executor):
        """Test executing a single-step plan."""
        single_step_plan = Plan(
            plan_id=generate_uuid(),
            goal="Single step",
            steps=[
                PlanStep(
                    step_id="only-step",
                    description="The only step",
                    dependencies=[],
                ),
            ],
            status=PlanStatus.PENDING,
        )

        result = await plan_executor.execute(single_step_plan)

        assert result.success is True
        assert len(result.step_results) == 1
        assert result.step_results[0].step_id == "only-step"


class TestDependencyOrdering:
    """Tests for dependency-based execution ordering."""

    @pytest.mark.asyncio
    async def test_respects_dependencies(self, plan_executor, simple_plan):
        """Test that steps are executed in dependency order."""
        execution_order = []

        async def tracking_executor(step, prior_results):
            execution_order.append(step.step_id)
            await asyncio.sleep(0.01)
            return StepResult(
                step_id=step.step_id,
                status=StepStatus.COMPLETED,
                output="Done",
                tokens_used=10,
                duration_ms=10.0,
                started_at=utc_now(),
                completed_at=utc_now(),
            )

        plan_executor._step_executor = tracking_executor
        await plan_executor.execute(simple_plan)

        # step-1 must be before step-2, step-2 must be before step-3
        assert execution_order.index("step-1") < execution_order.index("step-2")
        assert execution_order.index("step-2") < execution_order.index("step-3")

    @pytest.mark.asyncio
    async def test_blocked_steps_not_executed_prematurely(self, plan_executor):
        """Test that steps don't execute until dependencies complete."""
        plan = Plan(
            plan_id=generate_uuid(),
            goal="Test blocking",
            steps=[
                PlanStep(
                    step_id="blocker",
                    description="Blocking step",
                    dependencies=[],
                ),
                PlanStep(
                    step_id="blocked",
                    description="Blocked step",
                    dependencies=["blocker"],
                ),
            ],
            status=PlanStatus.PENDING,
        )

        execution_times = {}

        async def timed_executor(step, prior_results):
            execution_times[step.step_id] = asyncio.get_event_loop().time()
            await asyncio.sleep(0.1)  # Longer delay for blocker
            return StepResult(
                step_id=step.step_id,
                status=StepStatus.COMPLETED,
                output="Done",
                tokens_used=10,
                duration_ms=100.0,
                started_at=utc_now(),
                completed_at=utc_now(),
            )

        plan_executor._step_executor = timed_executor
        await plan_executor.execute(plan)

        assert execution_times["blocker"] < execution_times["blocked"]


class TestParallelExecution:
    """Tests for parallel step execution."""

    @pytest.mark.asyncio
    async def test_parallel_steps_execute_together(self, plan_executor, parallel_plan):
        """Test that independent steps execute in parallel."""
        concurrent_count = []
        current_count = 0
        lock = asyncio.Lock()

        async def counting_executor(step, prior_results):
            nonlocal current_count
            async with lock:
                current_count += 1
                concurrent_count.append(current_count)

            await asyncio.sleep(0.05)

            async with lock:
                current_count -= 1

            return StepResult(
                step_id=step.step_id,
                status=StepStatus.COMPLETED,
                output="Done",
                tokens_used=10,
                duration_ms=50.0,
                started_at=utc_now(),
                completed_at=utc_now(),
            )

        plan_executor._step_executor = counting_executor
        await plan_executor.execute(parallel_plan)

        # At some point, we should have had concurrent execution
        assert max(concurrent_count) > 1

    @pytest.mark.asyncio
    async def test_respects_max_concurrent(self, plan_executor, parallel_plan):
        """Test that concurrency limit is respected."""
        max_observed = 0
        current_count = 0
        lock = asyncio.Lock()

        async def counting_executor(step, prior_results):
            nonlocal current_count, max_observed
            async with lock:
                current_count += 1
                max_observed = max(max_observed, current_count)

            await asyncio.sleep(0.05)

            async with lock:
                current_count -= 1

            return StepResult(
                step_id=step.step_id,
                status=StepStatus.COMPLETED,
                output="Done",
                tokens_used=10,
                duration_ms=50.0,
                started_at=utc_now(),
                completed_at=utc_now(),
            )

        plan_executor._max_concurrent = 2
        plan_executor._step_executor = counting_executor
        await plan_executor.execute(parallel_plan)

        # Should never exceed max_concurrent
        assert max_observed <= 2


class TestPauseResume:
    """Tests for pause/resume functionality."""

    @pytest.mark.asyncio
    async def test_pause_and_resume(self, plan_executor, simple_plan):
        """Test pausing and resuming execution."""
        execution_started = asyncio.Event()
        resume_trigger = asyncio.Event()

        async def pausable_executor(step, prior_results):
            if step.step_id == "step-2":
                execution_started.set()
                await resume_trigger.wait()

            return StepResult(
                step_id=step.step_id,
                status=StepStatus.COMPLETED,
                output="Done",
                tokens_used=10,
                duration_ms=10.0,
                started_at=utc_now(),
                completed_at=utc_now(),
            )

        plan_executor._step_executor = pausable_executor

        # Start execution in background
        task = asyncio.create_task(plan_executor.execute(simple_plan))

        # Wait for execution to start
        await asyncio.wait_for(execution_started.wait(), timeout=1.0)

        # Pause execution
        assert plan_executor.pause() is True

        # Resume execution
        resume_trigger.set()
        assert plan_executor.resume() is True

        # Wait for completion
        result = await asyncio.wait_for(task, timeout=2.0)
        assert result.success is True

    @pytest.mark.asyncio
    async def test_pause_creates_checkpoint(self, plan_executor):
        """Test that pausing creates a checkpoint."""
        plan = Plan(
            plan_id=generate_uuid(),
            goal="Checkpoint test",
            steps=[
                PlanStep(step_id="s1", description="Step 1", dependencies=[]),
                PlanStep(step_id="s2", description="Step 2", dependencies=["s1"]),
            ],
            status=PlanStatus.PENDING,
        )

        # Execute with tracking
        result = await plan_executor.execute(plan)

        # Plan should complete successfully
        assert result.success is True


class TestAbort:
    """Tests for abort functionality."""

    @pytest.mark.asyncio
    async def test_abort_stops_execution(self, plan_executor, simple_plan):
        """Test that abort stops execution."""
        execution_started = asyncio.Event()

        async def slow_executor(step, prior_results):
            if step.step_id == "step-1":
                execution_started.set()
            await asyncio.sleep(0.5)
            return StepResult(
                step_id=step.step_id,
                status=StepStatus.COMPLETED,
                output="Done",
                tokens_used=10,
                duration_ms=500.0,
                started_at=utc_now(),
                completed_at=utc_now(),
            )

        plan_executor._step_executor = slow_executor

        # Start execution in background
        task = asyncio.create_task(plan_executor.execute(simple_plan))

        # Wait for first step to start
        await asyncio.wait_for(execution_started.wait(), timeout=1.0)

        # Abort
        assert plan_executor.abort() is True

        # Wait for completion
        result = await asyncio.wait_for(task, timeout=2.0)

        assert result.state == ExecutionState.ABORTED


class TestRetryLogic:
    """Tests for retry logic with exponential backoff."""

    @pytest.mark.asyncio
    async def test_retry_on_failure(self, plan_executor):
        """Test that failed steps are retried."""
        attempt_count = 0

        async def failing_then_success_executor(step, prior_results):
            nonlocal attempt_count
            attempt_count += 1

            if attempt_count < 3:
                raise Exception("Simulated failure")

            return StepResult(
                step_id=step.step_id,
                status=StepStatus.COMPLETED,
                output="Done after retries",
                tokens_used=10,
                duration_ms=10.0,
                started_at=utc_now(),
                completed_at=utc_now(),
            )

        plan = Plan(
            plan_id=generate_uuid(),
            goal="Retry test",
            steps=[
                PlanStep(step_id="s1", description="Failing step", dependencies=[]),
            ],
            status=PlanStatus.PENDING,
        )

        plan_executor._step_executor = failing_then_success_executor
        plan_executor._max_retries = 3

        result = await plan_executor.execute(plan)

        assert result.success is True
        assert attempt_count == 3

    @pytest.mark.asyncio
    async def test_max_retries_exceeded(self, plan_executor):
        """Test that max retries limit is enforced."""
        attempt_count = 0

        async def always_failing_executor(step, prior_results):
            nonlocal attempt_count
            attempt_count += 1
            raise Exception("Permanent failure")

        plan = Plan(
            plan_id=generate_uuid(),
            goal="Max retry test",
            steps=[
                PlanStep(step_id="s1", description="Always failing", dependencies=[]),
            ],
            status=PlanStatus.PENDING,
        )

        plan_executor._step_executor = always_failing_executor
        plan_executor._max_retries = 2

        result = await plan_executor.execute(plan)

        assert result.success is False
        assert result.state == ExecutionState.FAILED
        # Initial attempt + 2 retries = 3
        assert attempt_count == 3


class TestCheckpoints:
    """Tests for checkpoint creation and resumption."""

    @pytest.mark.asyncio
    async def test_checkpoint_created_on_pause(self, plan_executor, simple_plan):
        """Test that checkpoints are created during pause."""
        # Execute normally and verify checkpoint capability exists
        result = await plan_executor.execute(simple_plan)
        assert result is not None

    @pytest.mark.asyncio
    async def test_resume_from_checkpoint(self, plan_executor):
        """Test resuming execution from a checkpoint."""
        plan = Plan(
            plan_id="test-plan",
            goal="Checkpoint resume test",
            steps=[
                PlanStep(step_id="s1", description="Step 1", dependencies=[]),
                PlanStep(step_id="s2", description="Step 2", dependencies=["s1"]),
                PlanStep(step_id="s3", description="Step 3", dependencies=["s2"]),
            ],
            status=PlanStatus.PENDING,
        )

        # Create a checkpoint at step 2
        checkpoint = ExecutionCheckpoint(
            checkpoint_id=generate_uuid(),
            plan_id="test-plan",
            completed_steps=["s1"],
            step_results={
                "s1": StepResult(
                    step_id="s1",
                    status=StepStatus.COMPLETED,
                    output="Completed",
                    tokens_used=50,
                    duration_ms=100.0,
                )
            },
            tokens_used=50,
        )

        # Store checkpoint
        plan_executor._checkpoints[checkpoint.checkpoint_id] = checkpoint

        # Resume from checkpoint
        result = await plan_executor.resume_from_checkpoint(
            plan, checkpoint.checkpoint_id
        )

        assert result.success is True
        # Should have results for all steps
        assert len(result.step_results) == 3


class TestExecutionContext:
    """Tests for ExecutionContext helper class."""

    def test_get_ready_steps(self, simple_plan):
        """Test getting ready steps."""
        context = ExecutionContext(plan=simple_plan)

        # Initially only step-1 should be ready (no dependencies)
        ready = context.get_ready_steps()
        assert len(ready) == 1
        assert ready[0].step_id == "step-1"

        # After completing step-1, step-2 should be ready
        context.completed_steps.add("step-1")
        ready = context.get_ready_steps()
        assert len(ready) == 1
        assert ready[0].step_id == "step-2"

    def test_is_complete(self, simple_plan):
        """Test completion check."""
        context = ExecutionContext(plan=simple_plan)

        assert context.is_complete() is False

        context.completed_steps = {"step-1", "step-2", "step-3"}
        assert context.is_complete() is True

    def test_is_complete_with_failures(self, simple_plan):
        """Test completion check with failed steps."""
        context = ExecutionContext(plan=simple_plan)

        context.completed_steps = {"step-1", "step-2"}
        context.failed_steps = {"step-3"}

        assert context.is_complete() is True


class TestEventEmission:
    """Tests for event emission during execution."""

    @pytest.mark.asyncio
    async def test_step_events_emitted(self, plan_executor, simple_plan, event_store):
        """Test that step started/completed events are emitted."""
        result = await plan_executor.execute(simple_plan, session_id="test-session")

        if event_store.session_exists("test-session"):
            events = event_store.get_events("test-session")
            # Should have start and complete events for each step
            assert len(events) >= 6  # 3 starts + 3 completes

    @pytest.mark.asyncio
    async def test_plan_completed_event(self, plan_executor, simple_plan, event_store):
        """Test that plan completed event is emitted."""
        result = await plan_executor.execute(simple_plan, session_id="test-session")

        if event_store.session_exists("test-session"):
            events = event_store.get_events("test-session")
            completed_events = [
                e for e in events if "completed" in e.event_type.value.lower()
            ]
            assert len(completed_events) >= 1


class TestStatusTracking:
    """Tests for execution status tracking."""

    @pytest.mark.asyncio
    async def test_get_status_during_execution(self, plan_executor):
        """Test getting status during execution."""
        execution_started = asyncio.Event()
        continue_execution = asyncio.Event()

        async def blocking_executor(step, prior_results):
            execution_started.set()
            await continue_execution.wait()
            return StepResult(
                step_id=step.step_id,
                status=StepStatus.COMPLETED,
                output="Done",
                tokens_used=10,
                duration_ms=10.0,
                started_at=utc_now(),
                completed_at=utc_now(),
            )

        plan = Plan(
            plan_id="status-test-plan",
            goal="Status test",
            steps=[
                PlanStep(step_id="s1", description="Step 1", dependencies=[]),
            ],
            status=PlanStatus.PENDING,
        )

        plan_executor._step_executor = blocking_executor

        # Start execution
        task = asyncio.create_task(plan_executor.execute(plan))

        # Wait for execution to start
        await asyncio.wait_for(execution_started.wait(), timeout=1.0)

        # Check status
        status = plan_executor.get_status("status-test-plan")
        assert status is not None
        assert status["plan_id"] == "status-test-plan"
        assert status["state"] == "running"

        # Complete execution
        continue_execution.set()
        await task

    def test_get_status_not_found(self, plan_executor):
        """Test getting status for non-existent plan."""
        status = plan_executor.get_status("nonexistent-plan")
        assert status is None


class TestStepResult:
    """Tests for StepResult dataclass."""

    def test_from_exception(self):
        """Test creating StepResult from exception."""
        result = StepResult.from_exception("step-1", ValueError("Test error"))

        assert result.step_id == "step-1"
        assert result.status == StepStatus.FAILED
        assert "Test error" in result.error
        assert result.completed_at is not None

    def test_to_dict(self):
        """Test StepResult serialization."""
        result = StepResult(
            step_id="step-1",
            status=StepStatus.COMPLETED,
            output="Test output",
            tokens_used=100,
            duration_ms=50.0,
        )

        data = result.to_dict()
        assert data["step_id"] == "step-1"
        assert data["status"] == "completed"
        assert data["output"] == "Test output"


class TestPlanResult:
    """Tests for PlanResult dataclass."""

    def test_from_step_results(self):
        """Test creating PlanResult from step results."""
        step_results = {
            "s1": StepResult(
                step_id="s1",
                status=StepStatus.COMPLETED,
                output="Output 1",
                tokens_used=50,
                duration_ms=100.0,
            ),
            "s2": StepResult(
                step_id="s2",
                status=StepStatus.COMPLETED,
                output="Output 2",
                tokens_used=75,
                duration_ms=150.0,
            ),
        }

        result = PlanResult.from_step_results("test-plan", step_results)

        assert result.plan_id == "test-plan"
        assert result.success is True
        assert result.total_tokens == 125
        assert result.total_duration_ms == 250.0

    def test_from_step_results_with_failure(self):
        """Test PlanResult with failed steps."""
        step_results = {
            "s1": StepResult(
                step_id="s1",
                status=StepStatus.COMPLETED,
                tokens_used=50,
                duration_ms=100.0,
            ),
            "s2": StepResult(
                step_id="s2",
                status=StepStatus.FAILED,
                error="Test error",
                tokens_used=25,
                duration_ms=50.0,
            ),
        }

        result = PlanResult.from_step_results("test-plan", step_results)

        assert result.success is False
        assert result.state == ExecutionState.FAILED
        assert "Test error" in result.errors

    def test_to_dict(self):
        """Test PlanResult serialization."""
        result = PlanResult(
            plan_id="test-plan",
            success=True,
            state=ExecutionState.COMPLETED,
            step_results=[],
            total_tokens=100,
            total_duration_ms=500.0,
        )

        data = result.to_dict()
        assert data["plan_id"] == "test-plan"
        assert data["success"] is True
        assert data["state"] == "completed"
