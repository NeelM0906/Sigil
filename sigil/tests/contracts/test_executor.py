"""Tests for contract execution."""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock

from sigil.contracts.schema import Contract, ContractConstraints, Deliverable, FailureStrategy
from sigil.contracts.validator import ValidationResult, ValidationError
from sigil.contracts.executor import (
    AppliedStrategy,
    ContractExecutor,
    ContractResult,
    execute_with_contract,
)
from sigil.core.exceptions import ContractValidationError
from sigil.telemetry.tokens import TokenTracker


class TestContractResult:
    """Tests for ContractResult dataclass."""

    def test_create_success_result(self, valid_output):
        """Test creating a successful result."""
        result = ContractResult(
            output=valid_output,
            is_valid=True,
            attempts=1,
            tokens_used=1500,
            validation_result=ValidationResult(is_valid=True),
            applied_strategy=AppliedStrategy.SUCCESS,
        )
        assert result.is_valid is True
        assert result.succeeded is True
        assert result.attempts == 1
        assert result.applied_strategy == AppliedStrategy.SUCCESS

    def test_create_retry_result(self, valid_output):
        """Test creating a result after retry."""
        result = ContractResult(
            output=valid_output,
            is_valid=True,
            attempts=2,
            tokens_used=3000,
            validation_result=ValidationResult(is_valid=True),
            applied_strategy=AppliedStrategy.RETRY,
        )
        assert result.is_valid is True
        assert result.succeeded is True
        assert result.attempts == 2
        assert result.applied_strategy == AppliedStrategy.RETRY

    def test_create_fallback_result(self):
        """Test creating a fallback result."""
        result = ContractResult(
            output={"score": 0, "recommendation": ""},
            is_valid=False,
            attempts=3,
            tokens_used=4500,
            validation_result=ValidationResult(is_valid=False),
            applied_strategy=AppliedStrategy.FALLBACK,
        )
        assert result.is_valid is False
        assert result.succeeded is True  # Fallback still succeeds
        assert result.applied_strategy == AppliedStrategy.FALLBACK

    def test_create_fail_result(self):
        """Test creating a fail result."""
        result = ContractResult(
            output={},
            is_valid=False,
            attempts=1,
            tokens_used=1000,
            validation_result=ValidationResult(is_valid=False),
            applied_strategy=AppliedStrategy.FAIL,
        )
        assert result.is_valid is False
        assert result.succeeded is False  # Fail does not succeed
        assert result.applied_strategy == AppliedStrategy.FAIL

    def test_result_to_dict(self, valid_output):
        """Test serialization to dictionary."""
        result = ContractResult(
            output=valid_output,
            is_valid=True,
            attempts=1,
            tokens_used=1500,
            validation_result=ValidationResult(is_valid=True),
            applied_strategy=AppliedStrategy.SUCCESS,
            metadata={"execution_id": "test-123"},
        )
        data = result.to_dict()

        assert data["output"] == valid_output
        assert data["is_valid"] is True
        assert data["attempts"] == 1
        assert data["applied_strategy"] == "success"
        assert data["metadata"]["execution_id"] == "test-123"


class TestContractExecutorSuccess:
    """Tests for successful contract execution."""

    @pytest.mark.asyncio
    async def test_execute_success_first_attempt(
        self, executor, mock_agent_success, simple_contract
    ):
        """Test successful execution on first attempt."""
        result = await executor.execute_with_contract(
            mock_agent_success,
            "Score this lead",
            simple_contract,
        )

        assert result.is_valid is True
        assert result.attempts == 1
        assert result.applied_strategy == AppliedStrategy.SUCCESS
        assert result.output["score"] == 75
        mock_agent_success.run.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_tracks_metadata(
        self, executor, mock_agent_success, simple_contract
    ):
        """Test that execution tracks metadata."""
        result = await executor.execute_with_contract(
            mock_agent_success,
            "Score this lead",
            simple_contract,
        )

        assert "execution_id" in result.metadata
        assert "execution_time_ms" in result.metadata
        assert result.metadata["contract_name"] == "simple_contract"


class TestContractExecutorRetry:
    """Tests for retry behavior in contract execution."""

    @pytest.mark.asyncio
    async def test_execute_retry_success(
        self, executor, mock_agent_retry_success, simple_contract
    ):
        """Test successful execution after retry."""
        result = await executor.execute_with_contract(
            mock_agent_retry_success,
            "Score this lead",
            simple_contract,
        )

        assert result.is_valid is True
        assert result.attempts == 2
        assert result.applied_strategy == AppliedStrategy.RETRY
        assert mock_agent_retry_success.run.call_count == 2

    @pytest.mark.asyncio
    async def test_execute_retry_exhausted(self, executor, simple_contract):
        """Test execution when all retries exhausted."""
        # Create agent that always returns invalid output
        invalid_output = {"score": 75}  # missing recommendation
        agent = MagicMock()
        agent.run = AsyncMock(return_value=invalid_output)

        result = await executor.execute_with_contract(
            agent,
            "Score this lead",
            simple_contract,
        )

        # Should fall back after exhausting retries
        assert result.applied_strategy == AppliedStrategy.FALLBACK
        # With max_retries=2 and retry_manager checking attempt >= max_retries,
        # we get initial + 1 retry = 2 attempts before fallback
        assert result.attempts >= 2
        assert agent.run.call_count >= 2


class TestContractExecutorFallback:
    """Tests for fallback behavior in contract execution."""

    @pytest.mark.asyncio
    async def test_execute_fallback_generates_output(
        self, executor, mock_agent_invalid, contract_with_optional
    ):
        """Test that fallback generates valid output."""
        result = await executor.execute_with_contract(
            mock_agent_invalid,
            "Score this lead",
            contract_with_optional,
        )

        assert result.applied_strategy == AppliedStrategy.FALLBACK
        # Output should have all required fields (from template)
        assert "score" in result.output
        assert "recommendation" in result.output
        assert "fallback_strategy" in result.metadata


class TestContractExecutorFail:
    """Tests for fail strategy in contract execution."""

    @pytest.mark.asyncio
    async def test_execute_fail_raises_exception(self, executor, strict_contract):
        """Test that fail strategy raises ContractValidationError."""
        # strict_contract requires score (0-100), provide invalid score
        invalid_output = {"score": 150}  # Out of range
        agent = MagicMock()
        agent.run = AsyncMock(return_value=invalid_output)

        with pytest.raises(ContractValidationError) as exc_info:
            await executor.execute_with_contract(
                agent,
                "Score this lead",
                strict_contract,
            )

        assert "strict_contract" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_execute_fail_includes_errors(self, executor, strict_contract):
        """Test that fail exception includes error details."""
        # strict_contract requires score (0-100), provide invalid score
        invalid_output = {"score": 150}  # Out of range
        agent = MagicMock()
        agent.run = AsyncMock(return_value=invalid_output)

        with pytest.raises(ContractValidationError) as exc_info:
            await executor.execute_with_contract(
                agent,
                "Score this lead",
                strict_contract,
            )

        assert exc_info.value.context is not None
        assert "errors" in exc_info.value.context


class TestContractExecutorAgentFailure:
    """Tests for agent execution failure handling."""

    @pytest.mark.asyncio
    async def test_execute_agent_exception(
        self, executor, mock_agent_failure, simple_contract
    ):
        """Test handling when agent raises exception."""
        result = await executor.execute_with_contract(
            mock_agent_failure,
            "Score this lead",
            simple_contract,
        )

        # Should fall back after agent failure
        assert result.applied_strategy == AppliedStrategy.FALLBACK
        assert "score" in result.output  # Template output


class TestContractExecutorTokenTracking:
    """Tests for token tracking during execution."""

    @pytest.mark.asyncio
    async def test_execute_with_token_tracker(
        self, executor, mock_agent_success, simple_contract
    ):
        """Test execution with external token tracker."""
        tracker = TokenTracker()
        tracker.record_usage(500, 200)  # Pre-existing usage

        result = await executor.execute_with_contract(
            mock_agent_success,
            "Score this lead",
            simple_contract,
            token_tracker=tracker,
        )

        assert result.is_valid is True
        # tokens_used should reflect new usage (0 in this mock case)
        assert result.tokens_used >= 0


class TestRetryWithRefinement:
    """Tests for manual retry with refinement."""

    @pytest.mark.asyncio
    async def test_retry_with_refinement(
        self, executor, mock_agent_success, simple_contract
    ):
        """Test manual retry with refinement."""
        errors = [ValidationError(field="score", reason="Missing")]

        result = await executor.retry_with_refinement(
            mock_agent_success,
            "Original task",
            {"incomplete": "output"},
            errors,
            attempt=1,
            contract=simple_contract,
        )

        assert result is not None
        mock_agent_success.run.assert_called_once()


class TestGetFallbackResult:
    """Tests for manual fallback result generation."""

    def test_get_fallback_result(self, executor, simple_contract):
        """Test manual fallback result generation."""
        partial_output = {"score": 75}

        result = executor.get_fallback_result(simple_contract, partial_output)

        assert "score" in result
        assert "recommendation" in result


class TestAppliedStrategy:
    """Tests for AppliedStrategy enum."""

    def test_strategy_values(self):
        """Test applied strategy enum values."""
        assert AppliedStrategy.SUCCESS.value == "success"
        assert AppliedStrategy.RETRY.value == "retry"
        assert AppliedStrategy.FALLBACK.value == "fallback"
        assert AppliedStrategy.FAIL.value == "fail"


class TestConvenienceFunction:
    """Tests for execute_with_contract convenience function."""

    @pytest.mark.asyncio
    async def test_execute_with_contract_function(
        self, mock_agent_success, simple_contract
    ):
        """Test the convenience function."""
        result = await execute_with_contract(
            mock_agent_success,
            "Score this lead",
            simple_contract,
        )

        assert result.is_valid is True
        assert result.applied_strategy == AppliedStrategy.SUCCESS


class TestComplexScenarios:
    """Tests for complex execution scenarios."""

    @pytest.mark.asyncio
    async def test_complex_contract_validation(self, executor, complex_contract):
        """Test execution with complex contract."""
        valid_complex_output = {
            "score": 85,
            "bant_assessment": {
                "budget": "confirmed",
                "authority": "decision_maker",
                "need": "high",
                "timeline": "Q1",
            },
            "recommendation": "schedule_demo",
        }
        agent = MagicMock()
        agent.run = AsyncMock(return_value=valid_complex_output)

        result = await executor.execute_with_contract(
            agent,
            "Qualify this enterprise lead",
            complex_contract,
        )

        assert result.is_valid is True
        assert result.output["bant_assessment"]["budget"] == "confirmed"

    @pytest.mark.asyncio
    async def test_progressive_improvement(self, executor, simple_contract):
        """Test that agent improves progressively through retries."""
        # Simulate agent improving on each retry
        outputs = [
            {"score": 75},  # First attempt: missing recommendation
            {"score": 75, "recommendation": "schedule_demo"},  # Second: valid
        ]
        agent = MagicMock()
        agent.run = AsyncMock(side_effect=outputs)

        result = await executor.execute_with_contract(
            agent,
            "Score this lead",
            simple_contract,
        )

        assert result.is_valid is True
        assert result.attempts == 2
        assert result.applied_strategy == AppliedStrategy.RETRY
