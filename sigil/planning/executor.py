"""Plan Executor for Sigil v2 Phase 5 Planning & Reasoning.

This module implements plan execution with dependency-ordered step execution,
bounded concurrency, retry logic, and pause/resume capabilities.

Classes:
    PlanExecutor: Executes plans with monitoring and control.

Example:
    >>> from sigil.planning.executor import PlanExecutor
    >>> from sigil.planning.planner import Planner
    >>>
    >>> planner = Planner()
    >>> plan = await planner.create_plan(goal="Research competitors")
    >>>
    >>> executor = PlanExecutor()
    >>> result = await executor.execute(plan)
    >>> print(f"Success: {result.success}, Tokens: {result.total_tokens}")
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Optional, Awaitable

from sigil.config.schemas.plan import Plan, PlanStep, PlanStatus
from sigil.config import get_settings
from sigil.core.exceptions import SigilError, PlanExecutionError
from sigil.state.events import Event, EventType, _generate_event_id, _get_utc_now
from sigil.state.store import EventStore
from sigil.telemetry.tokens import TokenTracker

from sigil.planning.schemas import (
    StepStatus,
    StepResult,
    PlanResult,
    ExecutionState,
    ExecutionCheckpoint,
    generate_uuid,
    utc_now,
    DEFAULT_MAX_CONCURRENT,
    DEFAULT_MAX_RETRIES,
)


# =============================================================================
# Module Logger
# =============================================================================

logger = logging.getLogger(__name__)


# =============================================================================
# Constants
# =============================================================================

EXPONENTIAL_BACKOFF_BASE = 1.0
"""Base delay for exponential backoff in seconds."""

EXPONENTIAL_BACKOFF_MAX = 30.0
"""Maximum delay for exponential backoff in seconds."""


# =============================================================================
# Event Creators
# =============================================================================


def create_step_started_event(
    session_id: str,
    plan_id: str,
    step_id: str,
    step_description: str,
    correlation_id: Optional[str] = None,
) -> Event:
    """Create a StepStartedEvent."""
    payload = {
        "plan_id": plan_id,
        "step_id": step_id,
        "step_description": step_description,
    }
    return Event(
        event_id=_generate_event_id(),
        event_type=EventType.STEP_STARTED,
        timestamp=_get_utc_now(),
        session_id=session_id,
        correlation_id=correlation_id,
        payload=payload,
    )


def create_step_completed_event(
    session_id: str,
    plan_id: str,
    step_id: str,
    status: str,
    tokens_used: int,
    duration_ms: float,
    error: Optional[str] = None,
    correlation_id: Optional[str] = None,
) -> Event:
    """Create a StepCompletedEvent."""
    payload = {
        "plan_id": plan_id,
        "step_id": step_id,
        "status": status,
        "tokens_used": tokens_used,
        "duration_ms": duration_ms,
        "error": error,
    }
    return Event(
        event_id=_generate_event_id(),
        event_type=EventType.STEP_COMPLETED,
        timestamp=_get_utc_now(),
        session_id=session_id,
        correlation_id=correlation_id,
        payload=payload,
    )


def create_plan_completed_event(
    session_id: str,
    plan_id: str,
    success: bool,
    state: str,
    total_tokens: int,
    total_duration_ms: float,
    step_count: int,
    errors: list[str],
    correlation_id: Optional[str] = None,
) -> Event:
    """Create a PlanCompletedEvent."""
    payload = {
        "plan_id": plan_id,
        "success": success,
        "state": state,
        "total_tokens": total_tokens,
        "total_duration_ms": total_duration_ms,
        "step_count": step_count,
        "error_count": len(errors),
        "errors": errors[:5],  # Limit errors in event
    }
    return Event(
        event_id=_generate_event_id(),
        event_type=EventType.PLAN_COMPLETED,
        timestamp=_get_utc_now(),
        session_id=session_id,
        correlation_id=correlation_id,
        payload=payload,
    )


# =============================================================================
# Step Executor Type
# =============================================================================

# Type for step execution function
StepExecutorFn = Callable[[PlanStep, dict[str, StepResult]], Awaitable[StepResult]]


# =============================================================================
# Execution Context
# =============================================================================


@dataclass
class ExecutionContext:
    """Context for plan execution.

    Attributes:
        plan: The plan being executed.
        step_results: Results for completed steps.
        current_steps: Currently executing step IDs.
        completed_steps: Completed step IDs.
        failed_steps: Failed step IDs.
        state: Current execution state.
        started_at: Execution start time.
        tokens_used: Total tokens consumed.
        checkpoints: Created checkpoint IDs.
    """

    plan: Plan
    step_results: dict[str, StepResult] = field(default_factory=dict)
    current_steps: set[str] = field(default_factory=set)
    completed_steps: set[str] = field(default_factory=set)
    failed_steps: set[str] = field(default_factory=set)
    state: ExecutionState = ExecutionState.NOT_STARTED
    started_at: Optional[datetime] = None
    tokens_used: int = 0
    checkpoints: list[str] = field(default_factory=list)

    def get_ready_steps(self) -> list[PlanStep]:
        """Get steps that are ready to execute.

        Returns steps whose dependencies are all completed and
        which are still pending.
        """
        ready = []
        for step in self.plan.steps:
            if step.step_id in self.completed_steps:
                continue
            if step.step_id in self.current_steps:
                continue
            if step.step_id in self.failed_steps:
                continue

            # Check all dependencies are completed
            deps_met = all(
                dep_id in self.completed_steps for dep_id in step.dependencies
            )
            if deps_met:
                ready.append(step)

        return ready

    def is_complete(self) -> bool:
        """Check if all steps are complete or failed."""
        return len(self.completed_steps) + len(self.failed_steps) == len(
            self.plan.steps
        )


# =============================================================================
# Plan Executor
# =============================================================================


class PlanExecutor:
    """Executes plans with dependency ordering and concurrency control.

    The PlanExecutor handles step execution with proper dependency ordering,
    bounded concurrency, retry logic with exponential backoff, and
    pause/resume/abort capabilities.

    Features:
        - Topological sort for dependency ordering
        - Bounded concurrency (default max 3 parallel steps)
        - Retry logic (3 retries with exponential backoff)
        - Pause/resume/abort control
        - Event emission for each step transition
        - Checkpoint creation for resumption

    Attributes:
        event_store: Event store for audit trails.
        token_tracker: Token tracker for budget management.
        max_concurrent: Maximum parallel step executions.
        max_retries: Maximum retries per step.

    Example:
        >>> executor = PlanExecutor(max_concurrent=3)
        >>> result = await executor.execute(plan, session_id="sess-123")
    """

    def __init__(
        self,
        event_store: Optional[EventStore] = None,
        token_tracker: Optional[TokenTracker] = None,
        max_concurrent: int = DEFAULT_MAX_CONCURRENT,
        max_retries: int = DEFAULT_MAX_RETRIES,
        step_executor: Optional[StepExecutorFn] = None,
    ) -> None:
        """Initialize the PlanExecutor.

        Args:
            event_store: Optional custom event store.
            token_tracker: Optional token tracker for budget.
            max_concurrent: Maximum concurrent step executions.
            max_retries: Maximum retries per step.
            step_executor: Optional custom step executor function.
        """
        self._event_store = event_store
        self._token_tracker = token_tracker
        self._max_concurrent = max_concurrent
        self._max_retries = max_retries
        self._step_executor = step_executor or self._default_step_executor
        self._settings = get_settings()

        # Execution control
        self._pause_event = asyncio.Event()
        self._pause_event.set()  # Not paused by default
        self._abort_flag = False
        self._execution_contexts: dict[str, ExecutionContext] = {}
        self._checkpoints: dict[str, ExecutionCheckpoint] = {}

    async def _default_step_executor(
        self,
        step: PlanStep,
        prior_results: dict[str, StepResult],
    ) -> StepResult:
        """Default step executor implementation.

        This simulates step execution. In production, this would integrate
        with the reasoning manager and tool execution system.

        Args:
            step: The step to execute.
            prior_results: Results from prior steps.

        Returns:
            StepResult with execution outcome.
        """
        started_at = utc_now()
        start_time = time.time()

        # Simulate execution based on step type
        await asyncio.sleep(0.1)  # Simulate some processing time

        # Generate output based on step description
        output = f"Completed: {step.description}"

        # Simulate token usage
        tokens_used = len(step.description) // 4 + 100

        duration_ms = (time.time() - start_time) * 1000

        return StepResult(
            step_id=step.step_id,
            status=StepStatus.COMPLETED,
            output=output,
            tokens_used=tokens_used,
            duration_ms=duration_ms,
            retries=0,
            started_at=started_at,
            completed_at=utc_now(),
        )

    async def _execute_step_with_retry(
        self,
        step: PlanStep,
        context: ExecutionContext,
        session_id: Optional[str] = None,
    ) -> StepResult:
        """Execute a step with retry logic.

        Args:
            step: The step to execute.
            context: Execution context.
            session_id: Optional session ID for events.

        Returns:
            StepResult with execution outcome.
        """
        retries = 0
        last_error: Optional[Exception] = None

        # Emit step started event
        if self._event_store and session_id:
            event = create_step_started_event(
                session_id=session_id,
                plan_id=context.plan.plan_id,
                step_id=step.step_id,
                step_description=step.description,
            )
            self._event_store.append(event)

        while retries <= self._max_retries:
            try:
                # Check for pause
                await self._pause_event.wait()

                # Check for abort
                if self._abort_flag:
                    return StepResult(
                        step_id=step.step_id,
                        status=StepStatus.SKIPPED,
                        error="Execution aborted",
                        completed_at=utc_now(),
                    )

                # Execute step
                result = await self._step_executor(step, context.step_results)
                result.retries = retries

                # Track tokens
                if self._token_tracker and result.tokens_used > 0:
                    self._token_tracker.record_usage(
                        input_tokens=result.tokens_used // 2,
                        output_tokens=result.tokens_used // 2,
                    )

                # Emit step completed event
                if self._event_store and session_id:
                    event = create_step_completed_event(
                        session_id=session_id,
                        plan_id=context.plan.plan_id,
                        step_id=step.step_id,
                        status=result.status.value,
                        tokens_used=result.tokens_used,
                        duration_ms=result.duration_ms,
                        error=result.error,
                    )
                    self._event_store.append(event)

                return result

            except Exception as e:
                last_error = e
                retries += 1

                if retries <= self._max_retries:
                    # Exponential backoff
                    delay = min(
                        EXPONENTIAL_BACKOFF_BASE * (2 ** (retries - 1)),
                        EXPONENTIAL_BACKOFF_MAX,
                    )
                    logger.warning(
                        f"Step {step.step_id} failed (attempt {retries}), "
                        f"retrying in {delay:.1f}s: {e}"
                    )
                    await asyncio.sleep(delay)

        # All retries exhausted
        error_msg = f"Step failed after {retries} retries: {last_error}"
        logger.error(f"Step {step.step_id} failed: {error_msg}")

        result = StepResult(
            step_id=step.step_id,
            status=StepStatus.FAILED,
            error=error_msg,
            retries=retries - 1,
            completed_at=utc_now(),
        )

        # Emit step completed event for failure
        if self._event_store and session_id:
            event = create_step_completed_event(
                session_id=session_id,
                plan_id=context.plan.plan_id,
                step_id=step.step_id,
                status=result.status.value,
                tokens_used=0,
                duration_ms=0,
                error=error_msg,
            )
            self._event_store.append(event)

        return result

    async def _execute_parallel_batch(
        self,
        steps: list[PlanStep],
        context: ExecutionContext,
        session_id: Optional[str] = None,
    ) -> list[StepResult]:
        """Execute a batch of steps in parallel.

        Args:
            steps: Steps to execute in parallel.
            context: Execution context.
            session_id: Optional session ID for events.

        Returns:
            List of StepResults.
        """
        # Mark steps as current
        for step in steps:
            context.current_steps.add(step.step_id)

        # Execute in parallel with bounded concurrency
        semaphore = asyncio.Semaphore(self._max_concurrent)

        async def execute_with_semaphore(step: PlanStep) -> StepResult:
            async with semaphore:
                return await self._execute_step_with_retry(step, context, session_id)

        tasks = [execute_with_semaphore(step) for step in steps]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        step_results: list[StepResult] = []
        for step, result in zip(steps, results):
            context.current_steps.discard(step.step_id)

            if isinstance(result, Exception):
                step_result = StepResult.from_exception(step.step_id, result)
            else:
                step_result = result

            step_results.append(step_result)
            context.step_results[step.step_id] = step_result
            context.tokens_used += step_result.tokens_used

            if step_result.status == StepStatus.COMPLETED:
                context.completed_steps.add(step.step_id)
            else:
                context.failed_steps.add(step.step_id)

        return step_results

    def _create_checkpoint(self, context: ExecutionContext) -> ExecutionCheckpoint:
        """Create an execution checkpoint.

        Args:
            context: Current execution context.

        Returns:
            ExecutionCheckpoint for resumption.
        """
        checkpoint_id = generate_uuid()
        checkpoint = ExecutionCheckpoint(
            checkpoint_id=checkpoint_id,
            plan_id=context.plan.plan_id,
            completed_steps=list(context.completed_steps),
            step_results=dict(context.step_results),
            current_step=list(context.current_steps)[0] if context.current_steps else None,
            tokens_used=context.tokens_used,
        )

        self._checkpoints[checkpoint_id] = checkpoint
        context.checkpoints.append(checkpoint_id)

        logger.debug(
            f"Created checkpoint {checkpoint_id} for plan {context.plan.plan_id}"
        )

        return checkpoint

    async def execute(
        self,
        plan: Plan,
        session_id: Optional[str] = None,
        checkpoint: Optional[ExecutionCheckpoint] = None,
    ) -> PlanResult:
        """Execute a plan.

        Executes all steps in dependency order with bounded concurrency.
        Emits events for each step transition and creates checkpoints
        for potential resumption.

        Args:
            plan: The plan to execute.
            session_id: Optional session ID for event tracking.
            checkpoint: Optional checkpoint to resume from.

        Returns:
            PlanResult with execution outcome.

        Example:
            >>> result = await executor.execute(plan)
            >>> if result.success:
            ...     print(f"Plan completed with {result.total_tokens} tokens")
        """
        # Reset abort flag
        self._abort_flag = False

        # Initialize or restore context
        if checkpoint:
            context = ExecutionContext(
                plan=plan,
                step_results=dict(checkpoint.step_results),
                completed_steps=set(checkpoint.completed_steps),
                tokens_used=checkpoint.tokens_used,
            )
            context.state = ExecutionState.RUNNING
            logger.info(
                f"Resuming plan {plan.plan_id} from checkpoint {checkpoint.checkpoint_id}"
            )
        else:
            context = ExecutionContext(plan=plan)
            context.state = ExecutionState.RUNNING
            context.started_at = utc_now()

        # Store context for external control
        self._execution_contexts[plan.plan_id] = context

        try:
            # Main execution loop
            while not context.is_complete():
                # Check for abort
                if self._abort_flag:
                    context.state = ExecutionState.ABORTED
                    break

                # Check for pause
                if not self._pause_event.is_set():
                    context.state = ExecutionState.PAUSED
                    self._create_checkpoint(context)
                    await self._pause_event.wait()
                    context.state = ExecutionState.RUNNING

                # Get ready steps
                ready_steps = context.get_ready_steps()

                if not ready_steps:
                    if context.current_steps:
                        # Wait for current steps to complete
                        await asyncio.sleep(0.1)
                        continue
                    else:
                        # No more steps can be executed (blocked or complete)
                        break

                # Execute batch in parallel
                await self._execute_parallel_batch(ready_steps, context, session_id)

            # Determine final state
            if context.state != ExecutionState.ABORTED:
                if context.failed_steps:
                    context.state = ExecutionState.FAILED
                else:
                    context.state = ExecutionState.COMPLETED

            # Create result
            result = PlanResult.from_step_results(plan.plan_id, context.step_results)
            result.state = context.state
            result.started_at = context.started_at
            result.checkpoints = context.checkpoints

            # Emit plan completed event
            if self._event_store and session_id:
                event = create_plan_completed_event(
                    session_id=session_id,
                    plan_id=plan.plan_id,
                    success=result.success,
                    state=result.state.value,
                    total_tokens=result.total_tokens,
                    total_duration_ms=result.total_duration_ms,
                    step_count=len(result.step_results),
                    errors=result.errors,
                )
                self._event_store.append(event)

            logger.info(
                f"Plan {plan.plan_id} completed: success={result.success}, "
                f"tokens={result.total_tokens}, duration={result.total_duration_ms:.0f}ms"
            )

            return result

        finally:
            # Cleanup
            if plan.plan_id in self._execution_contexts:
                del self._execution_contexts[plan.plan_id]

    def pause(self, plan_id: Optional[str] = None) -> bool:
        """Pause plan execution.

        Args:
            plan_id: Optional specific plan ID to pause.
                If None, pauses all executions.

        Returns:
            True if pause was initiated.
        """
        self._pause_event.clear()
        logger.info(f"Pausing execution for plan: {plan_id or 'all'}")
        return True

    def resume(self, plan_id: Optional[str] = None) -> bool:
        """Resume paused plan execution.

        Args:
            plan_id: Optional specific plan ID to resume.
                If None, resumes all executions.

        Returns:
            True if resume was initiated.
        """
        self._pause_event.set()
        logger.info(f"Resuming execution for plan: {plan_id or 'all'}")
        return True

    def abort(self, plan_id: Optional[str] = None) -> bool:
        """Abort plan execution.

        Args:
            plan_id: Optional specific plan ID to abort.
                If None, aborts all executions.

        Returns:
            True if abort was initiated.
        """
        self._abort_flag = True
        self._pause_event.set()  # Ensure not stuck in pause
        logger.info(f"Aborting execution for plan: {plan_id or 'all'}")
        return True

    def get_status(self, plan_id: str) -> Optional[dict[str, Any]]:
        """Get current execution status for a plan.

        Args:
            plan_id: The plan ID to check.

        Returns:
            Status dictionary or None if not found.
        """
        context = self._execution_contexts.get(plan_id)
        if not context:
            return None

        return {
            "plan_id": plan_id,
            "state": context.state.value,
            "completed_steps": len(context.completed_steps),
            "total_steps": len(context.plan.steps),
            "failed_steps": len(context.failed_steps),
            "current_steps": list(context.current_steps),
            "tokens_used": context.tokens_used,
            "progress": len(context.completed_steps) / len(context.plan.steps)
            if context.plan.steps
            else 1.0,
        }

    def get_checkpoint(self, checkpoint_id: str) -> Optional[ExecutionCheckpoint]:
        """Get a checkpoint by ID.

        Args:
            checkpoint_id: The checkpoint ID.

        Returns:
            ExecutionCheckpoint or None if not found.
        """
        return self._checkpoints.get(checkpoint_id)

    async def resume_from_checkpoint(
        self,
        plan: Plan,
        checkpoint_id: str,
        session_id: Optional[str] = None,
    ) -> PlanResult:
        """Resume execution from a checkpoint.

        Args:
            plan: The plan to execute.
            checkpoint_id: The checkpoint ID to resume from.
            session_id: Optional session ID for event tracking.

        Returns:
            PlanResult with execution outcome.

        Raises:
            PlanExecutionError: If checkpoint not found.
        """
        checkpoint = self._checkpoints.get(checkpoint_id)
        if not checkpoint:
            raise PlanExecutionError(
                f"Checkpoint not found: {checkpoint_id}",
                plan_id=plan.plan_id,
            )

        return await self.execute(plan, session_id, checkpoint)


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "PlanExecutor",
    "ExecutionContext",
    "create_step_started_event",
    "create_step_completed_event",
    "create_plan_completed_event",
]
