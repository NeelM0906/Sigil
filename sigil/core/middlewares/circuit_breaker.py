"""Circuit breaker middleware for Sigil orchestrator pipeline.

This middleware prevents cascade failures by tracking step failures
and temporarily disabling steps that are consistently failing.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

from sigil.core.middleware import BaseMiddleware
from sigil.core.exceptions import SigilError

logger = logging.getLogger(__name__)


class CircuitState(str, Enum):
    """State of a circuit breaker.

    Attributes:
        CLOSED: Normal operation, requests pass through
        OPEN: Failing, requests are blocked
        HALF_OPEN: Testing, limited requests allowed
    """

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitOpenError(SigilError):
    """Raised when a circuit is open and blocking requests."""

    def __init__(self, step_name: str, message: str = "") -> None:
        super().__init__(
            message or f"Circuit breaker open for step '{step_name}'",
            code="CIRCUIT_OPEN",
            context={"step_name": step_name},
            recoverable=True,
        )
        self.step_name = step_name


@dataclass
class CircuitBreaker:
    """Circuit breaker state for a single step.

    Attributes:
        step_name: Name of the step
        state: Current circuit state
        failure_count: Number of consecutive failures
        success_count: Number of successes in half-open state
        last_failure_time: Timestamp of last failure
        last_state_change: Timestamp of last state change
    """

    step_name: str
    state: CircuitState = CircuitState.CLOSED
    failure_count: int = 0
    success_count: int = 0
    last_failure_time: Optional[float] = None
    last_state_change: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "step_name": self.step_name,
            "state": self.state.value,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "last_failure_time": self.last_failure_time,
            "last_state_change": self.last_state_change,
        }


class CircuitBreakerMiddleware(BaseMiddleware):
    """Prevents cascade failures with circuit breaker pattern.

    This middleware tracks step failures and temporarily disables
    steps that are consistently failing, allowing the system to
    recover gracefully.

    Circuit states:
    - CLOSED: Normal operation, requests pass through
    - OPEN: Step is failing, requests are blocked for recovery_timeout
    - HALF_OPEN: Testing recovery, limited requests allowed

    Example:
        >>> middleware = CircuitBreakerMiddleware(
        ...     failure_threshold=5,
        ...     recovery_timeout=30.0,
        ...     half_open_max_calls=3,
        ... )
        >>> chain.add(middleware)

    Attributes:
        failure_threshold: Number of failures before opening circuit
        recovery_timeout: Seconds to wait before trying again
        half_open_max_calls: Max calls in half-open state before closing
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
        half_open_max_calls: int = 3,
        excluded_steps: Optional[set[str]] = None,
    ) -> None:
        """Initialize the circuit breaker middleware.

        Args:
            failure_threshold: Number of failures before opening circuit
            recovery_timeout: Seconds to wait before trying again
            half_open_max_calls: Max calls in half-open before closing
            excluded_steps: Steps to exclude from circuit breaking
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_calls = half_open_max_calls
        self.excluded_steps = excluded_steps or set()
        self._circuits: dict[str, CircuitBreaker] = {}

    @property
    def name(self) -> str:
        """Get the middleware name."""
        return "CircuitBreakerMiddleware"

    def _get_circuit(self, step_name: str) -> CircuitBreaker:
        """Get or create circuit breaker for a step."""
        if step_name not in self._circuits:
            self._circuits[step_name] = CircuitBreaker(
                step_name=step_name,
                last_state_change=time.time(),
            )
        return self._circuits[step_name]

    def _should_allow_request(self, circuit: CircuitBreaker) -> bool:
        """Determine if a request should be allowed through."""
        if circuit.state == CircuitState.CLOSED:
            return True

        if circuit.state == CircuitState.OPEN:
            # Check if recovery timeout has elapsed
            if circuit.last_failure_time:
                elapsed = time.time() - circuit.last_failure_time
                if elapsed >= self.recovery_timeout:
                    # Transition to half-open
                    circuit.state = CircuitState.HALF_OPEN
                    circuit.success_count = 0
                    circuit.last_state_change = time.time()
                    logger.info(
                        f"Circuit for '{circuit.step_name}' transitioning to half-open"
                    )
                    return True
            return False

        if circuit.state == CircuitState.HALF_OPEN:
            # Allow limited requests in half-open state
            return circuit.success_count < self.half_open_max_calls

        return False

    def _record_success(self, circuit: CircuitBreaker) -> None:
        """Record a successful call."""
        if circuit.state == CircuitState.HALF_OPEN:
            circuit.success_count += 1
            if circuit.success_count >= self.half_open_max_calls:
                # Enough successes, close the circuit
                circuit.state = CircuitState.CLOSED
                circuit.failure_count = 0
                circuit.success_count = 0
                circuit.last_state_change = time.time()
                logger.info(f"Circuit for '{circuit.step_name}' closed after recovery")
        elif circuit.state == CircuitState.CLOSED:
            # Reset failure count on success
            circuit.failure_count = 0

    def _record_failure(self, circuit: CircuitBreaker) -> None:
        """Record a failed call."""
        circuit.failure_count += 1
        circuit.last_failure_time = time.time()

        if circuit.state == CircuitState.HALF_OPEN:
            # Any failure in half-open state reopens the circuit
            circuit.state = CircuitState.OPEN
            circuit.success_count = 0
            circuit.last_state_change = time.time()
            logger.warning(
                f"Circuit for '{circuit.step_name}' reopened after failure in half-open state"
            )
        elif circuit.state == CircuitState.CLOSED:
            if circuit.failure_count >= self.failure_threshold:
                circuit.state = CircuitState.OPEN
                circuit.last_state_change = time.time()
                logger.warning(
                    f"Circuit for '{circuit.step_name}' opened after "
                    f"{circuit.failure_count} failures"
                )

    async def pre_step(self, step_name: str, ctx: Any) -> Any:
        """Check circuit state before allowing step execution.

        Args:
            step_name: Name of the step
            ctx: Pipeline context

        Returns:
            The context unchanged

        Raises:
            CircuitOpenError: If the circuit is open
        """
        if step_name in self.excluded_steps:
            return ctx

        circuit = self._get_circuit(step_name)

        if not self._should_allow_request(circuit):
            raise CircuitOpenError(step_name)

        return ctx

    async def post_step(self, step_name: str, ctx: Any, result: Any) -> Any:
        """Record successful completion.

        Args:
            step_name: Name of the step
            ctx: Pipeline context
            result: Step result

        Returns:
            The result unchanged
        """
        if step_name in self.excluded_steps:
            return result

        circuit = self._get_circuit(step_name)
        self._record_success(circuit)

        return result

    async def on_error(
        self, step_name: str, ctx: Any, error: Exception
    ) -> Optional[Any]:
        """Record failure and update circuit state.

        Args:
            step_name: Name of the step
            ctx: Pipeline context
            error: The exception

        Raises:
            Exception: Re-raises the error
        """
        if step_name not in self.excluded_steps:
            circuit = self._get_circuit(step_name)
            self._record_failure(circuit)

        raise error

    def get_circuit_state(self, step_name: str) -> CircuitState:
        """Get the current state of a circuit.

        Args:
            step_name: Name of the step

        Returns:
            The circuit state
        """
        circuit = self._get_circuit(step_name)
        return circuit.state

    def get_all_circuits(self) -> dict[str, dict[str, Any]]:
        """Get state of all circuits.

        Returns:
            Dictionary mapping step names to circuit state dicts
        """
        return {name: circuit.to_dict() for name, circuit in self._circuits.items()}

    def reset_circuit(self, step_name: str) -> None:
        """Manually reset a circuit to closed state.

        Args:
            step_name: Name of the step
        """
        if step_name in self._circuits:
            circuit = self._circuits[step_name]
            circuit.state = CircuitState.CLOSED
            circuit.failure_count = 0
            circuit.success_count = 0
            circuit.last_state_change = time.time()
            logger.info(f"Circuit for '{step_name}' manually reset")

    def reset_all(self) -> None:
        """Reset all circuits."""
        for circuit in self._circuits.values():
            circuit.state = CircuitState.CLOSED
            circuit.failure_count = 0
            circuit.success_count = 0
            circuit.last_state_change = time.time()
        logger.info("All circuits reset")
