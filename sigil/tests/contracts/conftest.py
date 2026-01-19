"""Shared fixtures for contracts module tests."""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock

from sigil.contracts.schema import (
    Contract,
    ContractConstraints,
    Deliverable,
    FailureStrategy,
)
from sigil.contracts.validator import ContractValidator, ValidationResult
from sigil.contracts.retry import RetryManager
from sigil.contracts.fallback import FallbackManager
from sigil.contracts.executor import ContractExecutor
from sigil.telemetry.tokens import TokenTracker


# =============================================================================
# Sample Deliverables
# =============================================================================


@pytest.fixture
def score_deliverable():
    """Create a score deliverable (0-100 integer)."""
    return Deliverable(
        name="score",
        type="int",
        description="Score from 0 to 100",
        required=True,
        validation_rules=["0 <= value <= 100"],
        example=75,
    )


@pytest.fixture
def recommendation_deliverable():
    """Create a recommendation deliverable (string)."""
    return Deliverable(
        name="recommendation",
        type="str",
        description="Recommended next action",
        required=True,
        validation_rules=["len(value) > 0"],
        example="schedule_demo",
    )


@pytest.fixture
def optional_notes_deliverable():
    """Create an optional notes deliverable."""
    return Deliverable(
        name="notes",
        type="str",
        description="Optional additional notes",
        required=False,
        validation_rules=[],
        example="High priority lead",
    )


@pytest.fixture
def bant_deliverable():
    """Create a BANT assessment deliverable (dict)."""
    return Deliverable(
        name="bant_assessment",
        type="dict",
        description="BANT assessment dictionary",
        required=True,
        validation_rules=[
            "isinstance(value, dict)",
            "'budget' in value",
            "'authority' in value",
        ],
        example={
            "budget": "confirmed",
            "authority": "decision_maker",
            "need": "high",
            "timeline": "Q1",
        },
    )


# =============================================================================
# Sample Contracts
# =============================================================================


@pytest.fixture
def simple_contract(score_deliverable, recommendation_deliverable):
    """Create a simple contract with two required deliverables."""
    return Contract(
        name="simple_contract",
        description="A simple test contract",
        deliverables=[score_deliverable, recommendation_deliverable],
        constraints=ContractConstraints(
            max_total_tokens=5000,
            max_tool_calls=5,
        ),
        failure_strategy=FailureStrategy.RETRY,
        max_retries=2,
    )


@pytest.fixture
def contract_with_optional(
    score_deliverable, recommendation_deliverable, optional_notes_deliverable
):
    """Create a contract with optional deliverable."""
    return Contract(
        name="contract_with_optional",
        description="Contract with optional fields",
        deliverables=[
            score_deliverable,
            recommendation_deliverable,
            optional_notes_deliverable,
        ],
        constraints=ContractConstraints(max_total_tokens=5000),
        failure_strategy=FailureStrategy.FALLBACK,
        max_retries=1,
    )


@pytest.fixture
def strict_contract(score_deliverable):
    """Create a contract with fail strategy."""
    return Contract(
        name="strict_contract",
        description="Contract that fails on validation errors",
        deliverables=[score_deliverable],
        constraints=ContractConstraints(max_total_tokens=3000),
        failure_strategy=FailureStrategy.FAIL,
        max_retries=0,
    )


@pytest.fixture
def complex_contract(
    score_deliverable, recommendation_deliverable, bant_deliverable
):
    """Create a complex contract with multiple deliverables."""
    return Contract(
        name="complex_contract",
        description="Complex contract with BANT assessment",
        deliverables=[
            score_deliverable,
            bant_deliverable,
            recommendation_deliverable,
        ],
        constraints=ContractConstraints(
            max_input_tokens=2000,
            max_output_tokens=1000,
            max_total_tokens=5000,
            max_tool_calls=5,
            timeout_seconds=60,
        ),
        failure_strategy=FailureStrategy.RETRY,
        max_retries=2,
    )


# =============================================================================
# Sample Outputs
# =============================================================================


@pytest.fixture
def valid_output():
    """Create a valid output for simple_contract."""
    return {
        "score": 75,
        "recommendation": "schedule_demo",
    }


@pytest.fixture
def invalid_output_missing_field():
    """Create output missing required field."""
    return {
        "score": 75,
        # recommendation is missing
    }


@pytest.fixture
def invalid_output_wrong_type():
    """Create output with wrong type."""
    return {
        "score": "seventy-five",  # Should be int
        "recommendation": "schedule_demo",
    }


@pytest.fixture
def invalid_output_rule_violation():
    """Create output that violates validation rule."""
    return {
        "score": 150,  # Out of range (0-100)
        "recommendation": "schedule_demo",
    }


@pytest.fixture
def partial_output():
    """Create output with one valid, one invalid field."""
    return {
        "score": 75,  # Valid
        "recommendation": "",  # Invalid (empty string fails len > 0)
    }


@pytest.fixture
def valid_complex_output():
    """Create valid output for complex_contract."""
    return {
        "score": 85,
        "bant_assessment": {
            "budget": "confirmed",
            "authority": "decision_maker",
            "need": "high",
            "timeline": "Q1 2026",
        },
        "recommendation": "schedule_demo",
    }


# =============================================================================
# Component Fixtures
# =============================================================================


@pytest.fixture
def validator():
    """Create a ContractValidator instance."""
    return ContractValidator()


@pytest.fixture
def strict_validator():
    """Create a ContractValidator with strict mode."""
    return ContractValidator(strict_mode=True)


@pytest.fixture
def retry_manager():
    """Create a RetryManager instance."""
    return RetryManager()


@pytest.fixture
def fallback_manager():
    """Create a FallbackManager instance."""
    return FallbackManager()


@pytest.fixture
def token_tracker():
    """Create a TokenTracker instance."""
    return TokenTracker()


@pytest.fixture
def executor():
    """Create a ContractExecutor instance without events."""
    return ContractExecutor(emit_events=False)


# =============================================================================
# Mock Agent Fixtures
# =============================================================================


@pytest.fixture
def mock_agent_success(valid_output):
    """Create a mock agent that returns valid output."""
    agent = MagicMock()
    agent.run = AsyncMock(return_value=valid_output)
    return agent


@pytest.fixture
def mock_agent_failure():
    """Create a mock agent that raises an exception."""
    agent = MagicMock()
    agent.run = AsyncMock(side_effect=Exception("Agent execution failed"))
    return agent


@pytest.fixture
def mock_agent_invalid(invalid_output_missing_field):
    """Create a mock agent that returns invalid output."""
    agent = MagicMock()
    agent.run = AsyncMock(return_value=invalid_output_missing_field)
    return agent


@pytest.fixture
def mock_agent_retry_success(invalid_output_missing_field, valid_output):
    """Create a mock agent that fails then succeeds."""
    agent = MagicMock()
    agent.run = AsyncMock(
        side_effect=[invalid_output_missing_field, valid_output]
    )
    return agent


# =============================================================================
# Utility Functions
# =============================================================================


def create_contract(
    name: str = "test_contract",
    deliverables: list = None,
    failure_strategy: FailureStrategy = FailureStrategy.RETRY,
    max_retries: int = 2,
) -> Contract:
    """Helper to create test contracts."""
    if deliverables is None:
        deliverables = [
            Deliverable(
                name="result",
                type="str",
                description="Test result",
                required=True,
            )
        ]
    return Contract(
        name=name,
        description=f"Test contract: {name}",
        deliverables=deliverables,
        failure_strategy=failure_strategy,
        max_retries=max_retries,
    )
