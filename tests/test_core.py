"""Tests for Sigil v2 core module (base classes and exceptions).

Task 3.5.3: Write tests for base classes

Test Coverage:
- Abstract methods raise NotImplementedError when not implemented
- Exception chaining works correctly
- Exception serialization for logging
- Exception hierarchy is correct
- Base class contracts are enforced
"""

from __future__ import annotations

import pytest
import asyncio
from typing import Any

from sigil.core import (
    # Base classes
    BaseAgent,
    BaseStrategy,
    BaseRetriever,
    # Result types
    StrategyResult,
    RetrievalResult,
    # Exceptions
    SigilError,
    ConfigurationError,
    AgentError,
    AgentInitializationError,
    AgentExecutionError,
    AgentTimeoutError,
    MemoryError,
    MemoryWriteError,
    MemoryRetrievalError,
    ReasoningError,
    StrategyNotFoundError,
    PlanExecutionError,
    RoutingError,
    ContractError,
    ContractValidationError,
    ContractViolation,
    ToolError,
    ToolNotFoundError,
    ToolExecutionError,
    MCPConnectionError,
    EvolutionError,
    OptimizationError,
    PromptMutationError,
    TokenBudgetExceeded,
    TokenBudgetWarning,
)


# =============================================================================
# Test: Base Classes - Abstract Methods
# =============================================================================


class TestBaseAgentAbstract:
    """Test that BaseAgent abstract methods raise properly."""

    def test_cannot_instantiate_directly(self):
        """Test that BaseAgent cannot be instantiated directly."""
        with pytest.raises(TypeError, match="abstract"):
            BaseAgent()

    def test_must_implement_name(self):
        """Test that subclass must implement name property."""

        class PartialAgent(BaseAgent):
            @property
            def config(self):
                return {}

            async def run(self, message: str):
                return {}

            def get_tools(self):
                return []

        with pytest.raises(TypeError, match="abstract"):
            PartialAgent()

    def test_must_implement_config(self):
        """Test that subclass must implement config property."""

        class PartialAgent(BaseAgent):
            @property
            def name(self) -> str:
                return "test"

            async def run(self, message: str):
                return {}

            def get_tools(self):
                return []

        with pytest.raises(TypeError, match="abstract"):
            PartialAgent()

    def test_must_implement_run(self):
        """Test that subclass must implement run method."""

        class PartialAgent(BaseAgent):
            @property
            def name(self) -> str:
                return "test"

            @property
            def config(self):
                return {}

            def get_tools(self):
                return []

        with pytest.raises(TypeError, match="abstract"):
            PartialAgent()

    def test_must_implement_get_tools(self):
        """Test that subclass must implement get_tools method."""

        class PartialAgent(BaseAgent):
            @property
            def name(self) -> str:
                return "test"

            @property
            def config(self):
                return {}

            async def run(self, message: str):
                return {}

        with pytest.raises(TypeError, match="abstract"):
            PartialAgent()

    def test_complete_implementation_works(self):
        """Test that a complete implementation can be instantiated."""

        class CompleteAgent(BaseAgent):
            @property
            def name(self) -> str:
                return "test-agent"

            @property
            def config(self):
                return {"model": "test"}

            async def run(self, message: str):
                return {"response": message}

            def get_tools(self):
                return ["tool1", "tool2"]

        agent = CompleteAgent()
        assert agent.name == "test-agent"
        assert agent.config == {"model": "test"}
        assert agent.get_tools() == ["tool1", "tool2"]
        assert agent.get_description() == "Agent: test-agent"


class TestBaseStrategyAbstract:
    """Test that BaseStrategy abstract methods raise properly."""

    def test_cannot_instantiate_directly(self):
        """Test that BaseStrategy cannot be instantiated directly."""
        with pytest.raises(TypeError, match="abstract"):
            BaseStrategy()

    def test_must_implement_all_methods(self):
        """Test that subclass must implement all abstract methods."""

        class PartialStrategy(BaseStrategy):
            @property
            def name(self) -> str:
                return "test"

            # Missing complexity_range and execute

        with pytest.raises(TypeError, match="abstract"):
            PartialStrategy()

    def test_complete_implementation_works(self):
        """Test that a complete implementation can be instantiated."""

        class CompleteStrategy(BaseStrategy):
            @property
            def name(self) -> str:
                return "direct"

            @property
            def complexity_range(self) -> tuple[float, float]:
                return (0.0, 0.3)

            async def execute(
                self, task: str, context: dict[str, Any]
            ) -> StrategyResult:
                return StrategyResult(
                    success=True,
                    output="done",
                    reasoning_trace=["Executed"]
                )

        strategy = CompleteStrategy()
        assert strategy.name == "direct"
        assert strategy.complexity_range == (0.0, 0.3)
        assert strategy.is_applicable(0.2) is True
        assert strategy.is_applicable(0.5) is False


class TestBaseRetrieverAbstract:
    """Test that BaseRetriever abstract methods raise properly."""

    def test_cannot_instantiate_directly(self):
        """Test that BaseRetriever cannot be instantiated directly."""
        with pytest.raises(TypeError, match="abstract"):
            BaseRetriever()

    def test_complete_implementation_works(self):
        """Test that a complete implementation can be instantiated."""

        class CompleteRetriever(BaseRetriever):
            @property
            def retrieval_type(self) -> str:
                return "rag"

            async def retrieve(self, query: str, k: int = 10) -> list[Any]:
                return [{"item": query}]

        retriever = CompleteRetriever()
        assert retriever.retrieval_type == "rag"


# =============================================================================
# Test: Result Dataclasses
# =============================================================================


class TestStrategyResult:
    """Test StrategyResult dataclass."""

    def test_create_result(self):
        """Test creating a StrategyResult."""
        result = StrategyResult(
            success=True,
            output="test output",
            reasoning_trace=["step1", "step2"],
            tokens_used=100,
            metadata={"key": "value"}
        )
        assert result.success is True
        assert result.output == "test output"
        assert result.reasoning_trace == ["step1", "step2"]
        assert result.tokens_used == 100
        assert result.metadata == {"key": "value"}

    def test_default_values(self):
        """Test StrategyResult default values."""
        result = StrategyResult(
            success=False,
            output=None,
            reasoning_trace=[]
        )
        assert result.tokens_used == 0
        assert result.metadata is None


class TestRetrievalResult:
    """Test RetrievalResult dataclass."""

    def test_create_result(self):
        """Test creating a RetrievalResult."""
        result = RetrievalResult(
            items=["item1", "item2"],
            scores=[0.9, 0.8],
            total_found=10,
            query_time_ms=5.5
        )
        assert result.items == ["item1", "item2"]
        assert result.scores == [0.9, 0.8]
        assert result.total_found == 10
        assert result.query_time_ms == 5.5

    def test_default_values(self):
        """Test RetrievalResult default values."""
        result = RetrievalResult(items=[])
        assert result.scores is None
        assert result.total_found == 0
        assert result.query_time_ms == 0.0


# =============================================================================
# Test: Exception Hierarchy
# =============================================================================


class TestExceptionHierarchy:
    """Test that exception hierarchy is correct."""

    def test_sigil_error_is_base(self):
        """Test that SigilError is the base for all Sigil exceptions."""
        assert issubclass(ConfigurationError, SigilError)
        assert issubclass(AgentError, SigilError)
        assert issubclass(MemoryError, SigilError)
        assert issubclass(ReasoningError, SigilError)
        assert issubclass(PlanExecutionError, SigilError)
        assert issubclass(RoutingError, SigilError)
        assert issubclass(ContractError, SigilError)
        assert issubclass(ToolError, SigilError)
        assert issubclass(EvolutionError, SigilError)
        assert issubclass(TokenBudgetExceeded, SigilError)

    def test_agent_error_subtypes(self):
        """Test AgentError subtypes."""
        assert issubclass(AgentInitializationError, AgentError)
        assert issubclass(AgentExecutionError, AgentError)
        assert issubclass(AgentTimeoutError, AgentError)

    def test_memory_error_subtypes(self):
        """Test MemoryError subtypes."""
        assert issubclass(MemoryWriteError, MemoryError)
        assert issubclass(MemoryRetrievalError, MemoryError)

    def test_tool_error_subtypes(self):
        """Test ToolError subtypes."""
        assert issubclass(ToolNotFoundError, ToolError)
        assert issubclass(ToolExecutionError, ToolError)
        assert issubclass(MCPConnectionError, ToolError)

    def test_evolution_error_subtypes(self):
        """Test EvolutionError subtypes."""
        assert issubclass(OptimizationError, EvolutionError)
        assert issubclass(PromptMutationError, EvolutionError)

    def test_contract_violation_alias(self):
        """Test ContractViolation is alias for ContractValidationError."""
        assert ContractViolation is ContractValidationError


# =============================================================================
# Test: Exception Creation and Attributes
# =============================================================================


class TestSigilError:
    """Test SigilError base exception."""

    def test_create_basic(self):
        """Test creating a basic SigilError."""
        error = SigilError("Something went wrong")
        assert error.message == "Something went wrong"
        assert error.code == "SIGIL_ERROR"
        assert error.context == {}
        assert error.recoverable is False

    def test_create_with_all_params(self):
        """Test creating SigilError with all parameters."""
        error = SigilError(
            message="Test error",
            code="TEST_ERROR",
            context={"key": "value"},
            recoverable=True
        )
        assert error.message == "Test error"
        assert error.code == "TEST_ERROR"
        assert error.context == {"key": "value"}
        assert error.recoverable is True

    def test_str_format(self):
        """Test SigilError string format."""
        error = SigilError("Test message", code="MY_CODE")
        assert str(error) == "[MY_CODE] Test message"

    def test_can_be_raised_and_caught(self):
        """Test that SigilError can be raised and caught."""
        with pytest.raises(SigilError) as exc_info:
            raise SigilError("Test", code="TEST")
        assert exc_info.value.code == "TEST"


class TestSpecificExceptions:
    """Test specific exception types."""

    def test_configuration_error(self):
        """Test ConfigurationError."""
        error = ConfigurationError("Invalid config", config_key="model")
        assert error.config_key == "model"
        assert error.code == "CONFIG_ERROR"

    def test_agent_error(self):
        """Test AgentError."""
        error = AgentError("Agent failed", agent_id="agent-123")
        assert error.agent_id == "agent-123"
        assert error.code == "AGENT_ERROR"

    def test_memory_error(self):
        """Test MemoryError."""
        error = MemoryError("Memory failed", layer="items")
        assert error.layer == "items"
        assert error.code == "MEMORY_ERROR"

    def test_reasoning_error(self):
        """Test ReasoningError."""
        error = ReasoningError("Reasoning failed", strategy="cot")
        assert error.strategy == "cot"
        assert error.code == "REASONING_ERROR"

    def test_plan_execution_error(self):
        """Test PlanExecutionError."""
        error = PlanExecutionError(
            "Step failed",
            plan_id="plan-123",
            step_index=2,
            step_name="Execute query"
        )
        assert error.plan_id == "plan-123"
        assert error.step_index == 2
        assert error.step_name == "Execute query"
        assert error.code == "PLAN_EXECUTION_ERROR"

    def test_routing_error(self):
        """Test RoutingError."""
        error = RoutingError(
            "Cannot route",
            intent="unknown",
            available_handlers=["handler1", "handler2"]
        )
        assert error.intent == "unknown"
        assert error.available_handlers == ["handler1", "handler2"]
        assert error.code == "ROUTING_ERROR"

    def test_contract_error(self):
        """Test ContractError."""
        error = ContractError(
            "Contract failed",
            contract_id="contract-123",
            deliverable="output"
        )
        assert error.contract_id == "contract-123"
        assert error.deliverable == "output"
        assert error.code == "CONTRACT_ERROR"

    def test_tool_error(self):
        """Test ToolError."""
        error = ToolError("Tool failed", tool_name="websearch")
        assert error.tool_name == "websearch"
        assert error.code == "TOOL_ERROR"

    def test_evolution_error(self):
        """Test EvolutionError."""
        error = EvolutionError("Optimization failed", optimization_step=5)
        assert error.optimization_step == 5
        assert error.code == "EVOLUTION_ERROR"


# =============================================================================
# Test: Exception Chaining
# =============================================================================


class TestExceptionChaining:
    """Test that exception chaining works correctly."""

    def test_chain_with_from(self):
        """Test chaining exceptions with 'from'."""
        original = ValueError("Original error")
        try:
            try:
                raise original
            except ValueError as e:
                raise SigilError("Wrapped error") from e
        except SigilError as e:
            assert e.__cause__ is original

    def test_chain_propagates_context(self):
        """Test that context propagates through chain."""
        try:
            try:
                raise ToolExecutionError(
                    "Tool failed",
                    tool_name="api_call",
                    context={"url": "http://example.com"}
                )
            except ToolError as e:
                raise AgentExecutionError(
                    "Agent failed due to tool error",
                    agent_id="agent-1"
                ) from e
        except AgentError as e:
            assert e.agent_id == "agent-1"
            assert e.__cause__.tool_name == "api_call"


# =============================================================================
# Test: Exception Serialization
# =============================================================================


class TestExceptionSerialization:
    """Test exception serialization for logging."""

    def test_sigil_error_to_str(self):
        """Test SigilError converts to string properly."""
        error = SigilError(
            "Test message",
            code="TEST_CODE",
            context={"key": "value"}
        )
        error_str = str(error)
        assert "[TEST_CODE]" in error_str
        assert "Test message" in error_str

    def test_token_budget_exceeded_to_str(self):
        """Test TokenBudgetExceeded has detailed string."""
        error = TokenBudgetExceeded(
            "Budget exceeded",
            current_input_tokens=5000,
            current_output_tokens=3000,
            max_input_tokens=4000,
            max_output_tokens=2000,
            max_total_tokens=6000,
            exceeded_limit="total"
        )
        error_str = str(error)
        assert "TOKEN_BUDGET_EXCEEDED" in error_str
        assert "8000" in error_str or "8,000" in error_str  # Total used
        assert "6000" in error_str or "6,000" in error_str  # Max total

    def test_exception_context_is_dict(self):
        """Test that exception context is a dict for serialization."""
        error = SigilError(
            "Test",
            context={"nested": {"data": [1, 2, 3]}}
        )
        assert isinstance(error.context, dict)
        assert error.context["nested"]["data"] == [1, 2, 3]


# =============================================================================
# Test: TokenBudgetExceeded
# =============================================================================


class TestTokenBudgetExceeded:
    """Test TokenBudgetExceeded exception in detail."""

    def test_calculates_utilization_total(self):
        """Test utilization calculation for total limit."""
        error = TokenBudgetExceeded(
            "Exceeded",
            current_input_tokens=4000,
            current_output_tokens=2000,
            max_total_tokens=5000,
            exceeded_limit="total"
        )
        assert error.current_total_tokens == 6000
        assert error.utilization == 1.2  # 6000/5000

    def test_calculates_utilization_input(self):
        """Test utilization calculation for input limit."""
        error = TokenBudgetExceeded(
            "Exceeded",
            current_input_tokens=5000,
            current_output_tokens=1000,
            max_input_tokens=4000,
            exceeded_limit="input"
        )
        assert error.utilization == 1.25  # 5000/4000

    def test_is_not_recoverable(self):
        """Test that TokenBudgetExceeded is not recoverable."""
        error = TokenBudgetExceeded("Exceeded")
        assert error.recoverable is False


# =============================================================================
# Test: TokenBudgetWarning
# =============================================================================


class TestTokenBudgetWarning:
    """Test TokenBudgetWarning."""

    def test_is_user_warning(self):
        """Test TokenBudgetWarning is a UserWarning."""
        assert issubclass(TokenBudgetWarning, UserWarning)

    def test_create_warning(self):
        """Test creating a TokenBudgetWarning."""
        warning = TokenBudgetWarning(
            "Approaching limit",
            current_input_tokens=3000,
            current_output_tokens=1500,
            max_total_tokens=6000,
            utilization=0.75,
            threshold=0.7
        )
        assert warning.current_total_tokens == 4500
        assert warning.utilization == 0.75
        assert warning.threshold == 0.7

    def test_can_issue_warning(self):
        """Test that TokenBudgetWarning can be issued."""
        import warnings

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            warnings.warn(
                TokenBudgetWarning(
                    "Test warning",
                    current_input_tokens=100,
                    current_output_tokens=50,
                    max_total_tokens=200,
                    utilization=0.75,
                    threshold=0.7
                )
            )
            assert len(w) == 1
            assert issubclass(w[0].category, TokenBudgetWarning)


# =============================================================================
# Test: Async Methods
# =============================================================================


class TestAsyncMethods:
    """Test that async methods work correctly in implementations."""

    @pytest.mark.asyncio
    async def test_agent_run_is_async(self):
        """Test that agent run method is async."""

        class AsyncAgent(BaseAgent):
            @property
            def name(self) -> str:
                return "async-agent"

            @property
            def config(self):
                return {}

            async def run(self, message: str):
                await asyncio.sleep(0.001)  # Simulate async work
                return {"response": message}

            def get_tools(self):
                return []

        agent = AsyncAgent()
        result = await agent.run("test")
        assert result == {"response": "test"}

    @pytest.mark.asyncio
    async def test_strategy_execute_is_async(self):
        """Test that strategy execute method is async."""

        class AsyncStrategy(BaseStrategy):
            @property
            def name(self) -> str:
                return "async-strategy"

            @property
            def complexity_range(self) -> tuple[float, float]:
                return (0.0, 1.0)

            async def execute(
                self, task: str, context: dict[str, Any]
            ) -> StrategyResult:
                await asyncio.sleep(0.001)
                return StrategyResult(
                    success=True,
                    output=task,
                    reasoning_trace=["async done"]
                )

        strategy = AsyncStrategy()
        result = await strategy.execute("test", {})
        assert result.success is True
        assert result.output == "test"

    @pytest.mark.asyncio
    async def test_retriever_retrieve_is_async(self):
        """Test that retriever retrieve method is async."""

        class AsyncRetriever(BaseRetriever):
            @property
            def retrieval_type(self) -> str:
                return "rag"

            async def retrieve(self, query: str, k: int = 10) -> list[Any]:
                await asyncio.sleep(0.001)
                return [{"query": query}]

        retriever = AsyncRetriever()
        items = await retriever.retrieve("test")
        assert len(items) == 1
        assert items[0]["query"] == "test"

    @pytest.mark.asyncio
    async def test_retrieve_with_scores(self):
        """Test retrieve_with_scores default implementation."""

        class TestRetriever(BaseRetriever):
            @property
            def retrieval_type(self) -> str:
                return "test"

            async def retrieve(self, query: str, k: int = 10) -> list[Any]:
                return [1, 2, 3]

        retriever = TestRetriever()
        result = await retriever.retrieve_with_scores("test")
        assert isinstance(result, RetrievalResult)
        assert result.items == [1, 2, 3]
        assert result.total_found == 3
        assert result.scores is None
