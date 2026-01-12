"""Contract execution for Sigil v2 framework.

This module implements the main contract execution engine that coordinates
validation, retry, and fallback logic to ensure contract compliance.

Key Components:
    - ContractResult: Complete result of contract execution
    - ContractExecutor: Main orchestration class

The executor implements the following flow:
1. Execute agent to produce output
2. Validate output against contract
3. If valid: Return success
4. If invalid: Apply failure strategy (retry/fallback/fail)

Example:
    >>> from sigil.contracts.executor import ContractExecutor
    >>> executor = ContractExecutor()
    >>> result = await executor.execute_with_contract(agent, task, contract, context)
    >>> if result.is_valid:
    ...     print("Contract fulfilled:", result.output)
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional, Protocol, TYPE_CHECKING

from sigil.contracts.schema import Contract, FailureStrategy
from sigil.contracts.validator import ContractValidator, ValidationResult
from sigil.contracts.retry import RetryManager
from sigil.contracts.fallback import FallbackManager, FallbackStrategy
from sigil.core.exceptions import ContractValidationError
from sigil.telemetry.tokens import TokenTracker

if TYPE_CHECKING:
    from sigil.state.store import EventStore


class AppliedStrategy(str, Enum):
    """Strategy that was applied to achieve the result.

    Attributes:
        SUCCESS: Output passed validation on first try
        RETRY: Output passed after one or more retries
        FALLBACK: Fallback output was generated
        FAIL: Contract failed (exception raised)
    """

    SUCCESS = "success"
    RETRY = "retry"
    FALLBACK = "fallback"
    FAIL = "fail"


@dataclass
class ContractResult:
    """Complete result of contract execution.

    Contains the output (validated or fallback), validation details,
    execution metrics, and strategy information.

    Attributes:
        output: The output dictionary (validated or fallback)
        is_valid: Whether the output passed validation
        attempts: Number of attempts made
        tokens_used: Total tokens consumed
        validation_result: Detailed validation outcome
        applied_strategy: Which strategy produced the result
        metadata: Additional execution metadata

    Example:
        >>> result = ContractResult(
        ...     output={"score": 75, "recommendation": "schedule_demo"},
        ...     is_valid=True,
        ...     attempts=1,
        ...     tokens_used=1500,
        ...     validation_result=validation_result,
        ...     applied_strategy=AppliedStrategy.SUCCESS
        ... )
    """

    output: dict[str, Any]
    is_valid: bool
    attempts: int
    tokens_used: int
    validation_result: ValidationResult
    applied_strategy: AppliedStrategy
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize result to dictionary.

        Returns:
            Dictionary representation.
        """
        return {
            "output": self.output,
            "is_valid": self.is_valid,
            "attempts": self.attempts,
            "tokens_used": self.tokens_used,
            "validation_result": self.validation_result.to_dict(),
            "applied_strategy": self.applied_strategy.value,
            "metadata": self.metadata,
        }

    @property
    def succeeded(self) -> bool:
        """Check if execution succeeded (valid or fallback).

        Returns:
            True if output is usable (not a failure).
        """
        return self.applied_strategy != AppliedStrategy.FAIL


class AgentProtocol(Protocol):
    """Protocol for agents that can be executed with contracts."""

    async def run(self, task: str, context: Optional[Any] = None) -> dict[str, Any]:
        """Run the agent with a task.

        Args:
            task: Task description.
            context: Optional execution context.

        Returns:
            Output dictionary.
        """
        ...


class ContractExecutor:
    """Main contract execution engine.

    Orchestrates the full contract execution flow including:
    - Agent execution
    - Output validation
    - Retry logic with prompt refinement
    - Fallback generation

    Example:
        >>> executor = ContractExecutor()
        >>> result = await executor.execute_with_contract(
        ...     agent=my_agent,
        ...     task="Qualify this lead",
        ...     contract=lead_qualification_contract(),
        ...     context=lead_info
        ... )
    """

    def __init__(
        self,
        validator: Optional[ContractValidator] = None,
        retry_manager: Optional[RetryManager] = None,
        fallback_manager: Optional[FallbackManager] = None,
        event_store: Optional["EventStore"] = None,
        emit_events: bool = True,
    ) -> None:
        """Initialize the contract executor.

        Args:
            validator: Custom validator instance.
            retry_manager: Custom retry manager instance.
            fallback_manager: Custom fallback manager instance.
            event_store: Optional event store for event emission.
            emit_events: Whether to emit events during execution.
        """
        self.validator = validator or ContractValidator()
        self.retry_manager = retry_manager or RetryManager()
        self.fallback_manager = fallback_manager or FallbackManager()
        self.event_store = event_store
        self.emit_events = emit_events

    async def execute_with_contract(
        self,
        agent: AgentProtocol,
        task: str,
        contract: Contract,
        context: Optional[Any] = None,
        token_tracker: Optional[TokenTracker] = None,
    ) -> ContractResult:
        """Execute an agent with contract enforcement.

        Main entry point for contract-verified execution. Handles
        the full lifecycle of validation, retry, and fallback.

        Args:
            agent: Agent instance to execute.
            task: Task description for the agent.
            contract: Contract specifying required outputs.
            context: Optional context for agent execution.
            token_tracker: Optional tracker for token usage.

        Returns:
            ContractResult with output and execution details.

        Raises:
            ContractValidationError: If failure_strategy is FAIL and
                validation fails after all retries.
        """
        start_time = time.perf_counter()
        tracker = token_tracker or TokenTracker()
        execution_id = str(uuid.uuid4())

        attempt = 0
        tokens_at_start = tracker.total_tokens
        validation_result: Optional[ValidationResult] = None
        last_output: Optional[dict[str, Any]] = None
        current_task = task
        current_context = context

        # Emit validation started event
        self._emit_validation_started(execution_id, contract)

        while attempt <= contract.max_retries:
            attempt += 1

            # Execute agent
            try:
                output = await agent.run(current_task, current_context)
                last_output = output
            except Exception as e:
                # Agent execution failed
                last_output = {}
                validation_result = ValidationResult(
                    is_valid=False,
                    errors=[],
                    suggestion=f"Agent execution failed: {str(e)}",
                )
                break

            # Track tokens (simulated if tracker doesn't auto-update)
            tokens_used = tracker.total_tokens - tokens_at_start

            # Validate output
            validation_result = self.validator.validate(
                output, contract, tracker
            )

            # Emit validation event
            self._emit_validation_completed(
                execution_id, contract, validation_result, attempt
            )

            if validation_result.is_valid:
                # Success!
                total_time = (time.perf_counter() - start_time) * 1000
                strategy = AppliedStrategy.SUCCESS if attempt == 1 else AppliedStrategy.RETRY

                self._emit_contract_completed(
                    execution_id, contract, strategy, attempt
                )

                return ContractResult(
                    output=output,
                    is_valid=True,
                    attempts=attempt,
                    tokens_used=tracker.total_tokens - tokens_at_start,
                    validation_result=validation_result,
                    applied_strategy=strategy,
                    metadata={
                        "execution_id": execution_id,
                        "execution_time_ms": total_time,
                        "contract_name": contract.name,
                    },
                )

            # Check if we should fail immediately
            if contract.failure_strategy == FailureStrategy.FAIL:
                self._emit_contract_failed(
                    execution_id, contract, validation_result
                )
                raise ContractValidationError(
                    message=f"Contract '{contract.name}' validation failed: {validation_result.get_error_summary()}",
                    contract_id=execution_id,
                    context={"errors": [e.to_dict() for e in validation_result.errors]},
                )

            # Determine if we should retry
            tokens_remaining = self._calculate_remaining_tokens(
                contract, tracker.total_tokens - tokens_at_start
            )

            if self.retry_manager.should_retry(
                attempt, contract, tokens_remaining, validation_result.errors
            ):
                # Emit retry event
                self._emit_retry(
                    execution_id, contract, attempt, validation_result
                )

                # Refine prompt
                current_task = self.retry_manager.refine_prompt(
                    task, validation_result.errors, attempt, contract
                )
                # Reset context for fresh attempt
                current_context = None
                continue

            # No more retries - apply fallback
            break

        # Apply fallback strategy
        fallback_result = self.fallback_manager.select_strategy(
            last_output, contract, validation_result
        )

        # Emit fallback event
        self._emit_fallback(
            execution_id, contract, fallback_result.strategy
        )

        # Handle escalation
        if fallback_result.strategy == FallbackStrategy.ESCALATE:
            self._emit_contract_failed(
                execution_id, contract, validation_result
            )
            raise ContractValidationError(
                message=f"Contract '{contract.name}' validation failed after {attempt} attempts",
                contract_id=execution_id,
                context={
                    "errors": [e.to_dict() for e in (validation_result.errors if validation_result else [])],
                    "attempts": attempt,
                },
            )

        total_time = (time.perf_counter() - start_time) * 1000

        self._emit_contract_completed(
            execution_id, contract, AppliedStrategy.FALLBACK, attempt
        )

        return ContractResult(
            output=fallback_result.output,
            is_valid=False,
            attempts=attempt,
            tokens_used=tracker.total_tokens - tokens_at_start,
            validation_result=validation_result or ValidationResult(is_valid=False),
            applied_strategy=AppliedStrategy.FALLBACK,
            metadata={
                "execution_id": execution_id,
                "execution_time_ms": total_time,
                "contract_name": contract.name,
                "fallback_strategy": fallback_result.strategy.value,
                "fallback_warnings": fallback_result.warnings,
            },
        )

    async def retry_with_refinement(
        self,
        agent: AgentProtocol,
        task: str,
        output: dict[str, Any],
        errors: list,
        attempt: int,
        contract: Contract,
    ) -> dict[str, Any]:
        """Execute a single retry attempt with refined prompt.

        Used for manual retry control or testing.

        Args:
            agent: Agent to execute.
            task: Original task description.
            output: Previous output that failed validation.
            errors: Validation errors from previous attempt.
            attempt: Current retry attempt number.
            contract: Contract being executed.

        Returns:
            New output from agent (may still be invalid).
        """
        refined_task = self.retry_manager.refine_prompt(
            task, errors, attempt, contract
        )
        return await agent.run(refined_task, None)

    def get_fallback_result(
        self,
        contract: Contract,
        partial_output: Optional[dict[str, Any]],
        validation_result: Optional[ValidationResult] = None,
    ) -> dict[str, Any]:
        """Generate fallback output for a contract.

        Used for manual fallback generation or testing.

        Args:
            contract: Contract to generate fallback for.
            partial_output: Partial output from failed execution.
            validation_result: Validation result with partial info.

        Returns:
            Fallback output dictionary.
        """
        result = self.fallback_manager.select_strategy(
            partial_output, contract, validation_result
        )
        return result.output

    def _calculate_remaining_tokens(
        self,
        contract: Contract,
        tokens_used: int,
    ) -> int:
        """Calculate remaining token budget.

        Args:
            contract: Contract with constraints.
            tokens_used: Tokens already consumed.

        Returns:
            Estimated remaining tokens.
        """
        if contract.constraints.max_total_tokens is None:
            return 10000  # Default large budget

        return contract.constraints.get_remaining_tokens(tokens_used)

    def _emit_validation_started(
        self,
        execution_id: str,
        contract: Contract,
    ) -> None:
        """Emit validation started event."""
        if not self.emit_events or not self.event_store:
            return

        # Import here to avoid circular imports
        from sigil.state.events import Event, EventType, _generate_event_id, _get_utc_now

        event = Event(
            event_id=_generate_event_id(),
            event_type=EventType.CONTRACT_VALIDATED,  # Will use specific type when added
            timestamp=_get_utc_now(),
            session_id=execution_id,
            payload={
                "action": "validation_started",
                "contract_name": contract.name,
                "deliverables": contract.get_deliverable_names(),
            },
        )
        try:
            self.event_store.append(event)
        except Exception:
            pass  # Don't fail execution on event errors

    def _emit_validation_completed(
        self,
        execution_id: str,
        contract: Contract,
        result: ValidationResult,
        attempt: int,
    ) -> None:
        """Emit validation completed event."""
        if not self.emit_events or not self.event_store:
            return

        from sigil.state.events import (
            create_contract_validated_event,
        )

        event = create_contract_validated_event(
            session_id=execution_id,
            contract_id=execution_id,
            contract_name=contract.name,
            passed=result.is_valid,
            deliverables_checked=result.deliverables_checked,
            validation_errors=[str(e) for e in result.errors],
            retry_count=attempt - 1,
        )
        try:
            self.event_store.append(event)
        except Exception:
            pass

    def _emit_retry(
        self,
        execution_id: str,
        contract: Contract,
        attempt: int,
        result: ValidationResult,
    ) -> None:
        """Emit retry event."""
        if not self.emit_events or not self.event_store:
            return

        from sigil.state.events import Event, EventType, _generate_event_id, _get_utc_now

        event = Event(
            event_id=_generate_event_id(),
            event_type=EventType.CONTRACT_VALIDATED,
            timestamp=_get_utc_now(),
            session_id=execution_id,
            payload={
                "action": "retry",
                "contract_name": contract.name,
                "attempt": attempt,
                "error_count": len(result.errors),
            },
        )
        try:
            self.event_store.append(event)
        except Exception:
            pass

    def _emit_fallback(
        self,
        execution_id: str,
        contract: Contract,
        strategy: FallbackStrategy,
    ) -> None:
        """Emit fallback event."""
        if not self.emit_events or not self.event_store:
            return

        from sigil.state.events import Event, EventType, _generate_event_id, _get_utc_now

        event = Event(
            event_id=_generate_event_id(),
            event_type=EventType.CONTRACT_VALIDATED,
            timestamp=_get_utc_now(),
            session_id=execution_id,
            payload={
                "action": "fallback",
                "contract_name": contract.name,
                "fallback_strategy": strategy.value,
            },
        )
        try:
            self.event_store.append(event)
        except Exception:
            pass

    def _emit_contract_completed(
        self,
        execution_id: str,
        contract: Contract,
        strategy: AppliedStrategy,
        attempts: int,
    ) -> None:
        """Emit contract completed event."""
        if not self.emit_events or not self.event_store:
            return

        from sigil.state.events import Event, EventType, _generate_event_id, _get_utc_now

        event = Event(
            event_id=_generate_event_id(),
            event_type=EventType.CONTRACT_VALIDATED,
            timestamp=_get_utc_now(),
            session_id=execution_id,
            payload={
                "action": "completed",
                "contract_name": contract.name,
                "applied_strategy": strategy.value,
                "total_attempts": attempts,
            },
        )
        try:
            self.event_store.append(event)
        except Exception:
            pass

    def _emit_contract_failed(
        self,
        execution_id: str,
        contract: Contract,
        result: Optional[ValidationResult],
    ) -> None:
        """Emit contract failed event."""
        if not self.emit_events or not self.event_store:
            return

        from sigil.state.events import Event, EventType, _generate_event_id, _get_utc_now

        event = Event(
            event_id=_generate_event_id(),
            event_type=EventType.ERROR_OCCURRED,
            timestamp=_get_utc_now(),
            session_id=execution_id,
            payload={
                "error_type": "ContractValidationError",
                "contract_name": contract.name,
                "error_count": len(result.errors) if result else 0,
            },
        )
        try:
            self.event_store.append(event)
        except Exception:
            pass


# Convenience function for simple execution
async def execute_with_contract(
    agent: AgentProtocol,
    task: str,
    contract: Contract,
    context: Optional[Any] = None,
) -> ContractResult:
    """Execute an agent with contract enforcement.

    Convenience function that creates a default executor and runs it.

    Args:
        agent: Agent instance to execute.
        task: Task description for the agent.
        contract: Contract specifying required outputs.
        context: Optional context for agent execution.

    Returns:
        ContractResult with output and execution details.
    """
    executor = ContractExecutor(emit_events=False)
    return await executor.execute_with_contract(agent, task, contract, context)


__all__ = [
    "AppliedStrategy",
    "ContractResult",
    "AgentProtocol",
    "ContractExecutor",
    "execute_with_contract",
]
