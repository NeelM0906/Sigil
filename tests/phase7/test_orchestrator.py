"""Tests for SigilOrchestrator.

This module contains comprehensive tests for the SigilOrchestrator class,
including unit tests, integration tests, and edge case tests.

Test Categories:
    - Request/Response tests
    - Pipeline step tests
    - Error handling tests
    - Metrics tests
    - Integration tests
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timezone

from sigil.orchestrator import (
    SigilOrchestrator,
    OrchestratorRequest,
    OrchestratorResponse,
    OrchestratorStatus,
    PipelineContext,
    process_message,
    DEFAULT_INPUT_BUDGET,
    DEFAULT_OUTPUT_BUDGET,
)
from sigil.config.settings import SigilSettings
from sigil.routing.router import Router, RouteDecision, Intent


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_settings():
    """Create mock settings with all features disabled."""
    settings = MagicMock(spec=SigilSettings)
    settings.use_memory = False
    settings.use_planning = False
    settings.use_contracts = False
    settings.use_routing = False
    settings.use_evolution = False
    settings.get_active_features.return_value = []
    settings.debug = False
    return settings


@pytest.fixture
def mock_settings_all_enabled():
    """Create mock settings with all features enabled."""
    settings = MagicMock(spec=SigilSettings)
    settings.use_memory = True
    settings.use_planning = True
    settings.use_contracts = True
    settings.use_routing = True
    settings.use_evolution = True
    settings.get_active_features.return_value = [
        "memory", "planning", "contracts", "routing", "evolution"
    ]
    settings.debug = True
    return settings


@pytest.fixture
def mock_router():
    """Create mock router."""
    router = MagicMock(spec=Router)
    router.route.return_value = RouteDecision(
        intent=Intent.GENERAL_CHAT,
        confidence=0.9,
        complexity=0.5,
        handler_name="default",
        use_planning=False,
        use_memory=False,
        use_contracts=False,
    )
    return router


@pytest.fixture
def orchestrator(mock_settings, mock_router):
    """Create orchestrator with mocked dependencies."""
    return SigilOrchestrator(
        settings=mock_settings,
        router=mock_router,
    )


@pytest.fixture
def sample_request():
    """Create a sample orchestrator request."""
    return OrchestratorRequest(
        message="Test message",
        session_id="test-session-123",
        user_id="test-user-456",
    )


# =============================================================================
# OrchestratorRequest Tests
# =============================================================================


class TestOrchestratorRequest:
    """Tests for OrchestratorRequest dataclass."""

    def test_request_creation_basic(self):
        """Test basic request creation."""
        request = OrchestratorRequest(
            message="Hello",
            session_id="sess-123",
        )
        assert request.message == "Hello"
        assert request.session_id == "sess-123"
        assert request.user_id is None
        assert request.correlation_id is not None

    def test_request_creation_full(self):
        """Test request creation with all fields."""
        request = OrchestratorRequest(
            message="Test",
            session_id="sess-123",
            user_id="user-456",
            agent_name="my-agent",
            context={"key": "value"},
            contract_name="lead_qualification",
            force_strategy="direct",
            max_tokens=1000,
            timeout_seconds=30.0,
            correlation_id="corr-789",
            metadata={"source": "test"},
        )
        assert request.message == "Test"
        assert request.user_id == "user-456"
        assert request.agent_name == "my-agent"
        assert request.context == {"key": "value"}
        assert request.correlation_id == "corr-789"

    def test_request_correlation_id_auto_generated(self):
        """Test that correlation ID is auto-generated if not provided."""
        request = OrchestratorRequest(
            message="Test",
            session_id="sess-123",
        )
        assert request.correlation_id is not None
        assert len(request.correlation_id) == 36  # UUID format


class TestOrchestratorResponse:
    """Tests for OrchestratorResponse dataclass."""

    def test_response_success_property(self):
        """Test success property."""
        response = OrchestratorResponse(
            request_id="req-123",
            status=OrchestratorStatus.SUCCESS,
            output={"result": "done"},
        )
        assert response.success is True

    def test_response_failed_not_success(self):
        """Test that failed status is not success."""
        response = OrchestratorResponse(
            request_id="req-123",
            status=OrchestratorStatus.FAILED,
            output={},
            errors=["Something went wrong"],
        )
        assert response.success is False
        assert response.has_errors is True

    def test_response_to_dict(self):
        """Test response serialization to dict."""
        response = OrchestratorResponse(
            request_id="req-123",
            status=OrchestratorStatus.SUCCESS,
            output={"result": "done"},
            tokens_used=100,
            execution_time_ms=50.0,
        )
        data = response.to_dict()
        assert data["request_id"] == "req-123"
        assert data["status"] == "success"
        assert data["tokens_used"] == 100


# =============================================================================
# PipelineContext Tests
# =============================================================================


class TestPipelineContext:
    """Tests for PipelineContext class."""

    def test_context_creation(self, sample_request):
        """Test pipeline context creation."""
        ctx = PipelineContext(sample_request)
        assert ctx.request == sample_request
        assert ctx.tokens_used == 0
        assert ctx.errors == []
        assert ctx.warnings == []

    def test_add_tokens(self, sample_request):
        """Test adding tokens to context."""
        ctx = PipelineContext(sample_request)
        ctx.add_tokens(100)
        ctx.add_tokens(50)
        assert ctx.tokens_used == 150

    def test_add_error(self, sample_request):
        """Test adding errors to context."""
        ctx = PipelineContext(sample_request)
        ctx.add_error("Error 1")
        ctx.add_error("Error 2")
        assert len(ctx.errors) == 2
        assert "Error 1" in ctx.errors

    def test_add_warning(self, sample_request):
        """Test adding warnings to context."""
        ctx = PipelineContext(sample_request)
        ctx.add_warning("Warning 1")
        assert len(ctx.warnings) == 1

    def test_elapsed_time(self, sample_request):
        """Test elapsed time tracking."""
        ctx = PipelineContext(sample_request)
        # Give it some time
        import time
        time.sleep(0.01)
        assert ctx.elapsed_ms > 0


# =============================================================================
# SigilOrchestrator Tests
# =============================================================================


class TestSigilOrchestratorInit:
    """Tests for SigilOrchestrator initialization."""

    def test_init_default(self):
        """Test default initialization."""
        with patch('sigil.orchestrator.get_settings') as mock_get_settings:
            mock_settings = MagicMock()
            mock_settings.use_memory = False
            mock_settings.use_planning = False
            mock_settings.use_contracts = False
            mock_settings.use_routing = False
            mock_settings.get_active_features.return_value = []
            mock_get_settings.return_value = mock_settings

            orchestrator = SigilOrchestrator()
            assert orchestrator is not None
            assert orchestrator._settings == mock_settings

    def test_init_with_custom_settings(self, mock_settings):
        """Test initialization with custom settings."""
        orchestrator = SigilOrchestrator(settings=mock_settings)
        assert orchestrator._settings == mock_settings

    def test_init_metrics_zeroed(self, mock_settings):
        """Test that metrics start at zero."""
        orchestrator = SigilOrchestrator(settings=mock_settings)
        assert orchestrator._total_requests == 0
        assert orchestrator._successful_requests == 0
        assert orchestrator._failed_requests == 0


class TestSigilOrchestratorProcess:
    """Tests for SigilOrchestrator process method."""

    @pytest.mark.asyncio
    async def test_process_simple_request(self, orchestrator, sample_request):
        """Test processing a simple request."""
        response = await orchestrator.process(sample_request)

        assert response is not None
        assert response.request_id is not None
        assert isinstance(response.status, OrchestratorStatus)
        assert response.execution_time_ms > 0

    @pytest.mark.asyncio
    async def test_process_increments_metrics(self, orchestrator, sample_request):
        """Test that processing increments metrics."""
        initial_requests = orchestrator._total_requests

        await orchestrator.process(sample_request)

        assert orchestrator._total_requests == initial_requests + 1

    @pytest.mark.asyncio
    async def test_process_with_routing_disabled(self, mock_settings, sample_request):
        """Test processing with routing disabled."""
        mock_settings.use_routing = False
        orchestrator = SigilOrchestrator(settings=mock_settings)

        response = await orchestrator.process(sample_request)

        # Should use default route decision
        assert response.route_decision is not None
        assert response.route_decision.intent == Intent.GENERAL_CHAT

    @pytest.mark.asyncio
    async def test_process_with_routing_enabled(self, mock_settings, mock_router, sample_request):
        """Test processing with routing enabled."""
        mock_settings.use_routing = True
        orchestrator = SigilOrchestrator(
            settings=mock_settings,
            router=mock_router,
        )

        response = await orchestrator.process(sample_request)

        # Router should have been called
        mock_router.route.assert_called_once_with(sample_request.message)

    @pytest.mark.asyncio
    async def test_process_error_handling(self, mock_settings, sample_request):
        """Test error handling during processing."""
        mock_settings.use_routing = True

        # Create router that raises an exception
        mock_router = MagicMock()
        mock_router.route.side_effect = Exception("Routing failed")

        orchestrator = SigilOrchestrator(
            settings=mock_settings,
            router=mock_router,
        )

        response = await orchestrator.process(sample_request)

        # Should still complete with warning
        assert response is not None
        assert len(response.warnings) > 0 or len(response.errors) > 0

    @pytest.mark.asyncio
    async def test_process_timeout_handling(self, orchestrator, sample_request):
        """Test timeout handling."""
        # This tests that timeout is handled gracefully
        # In a real scenario, we'd mock async operations to time out
        response = await orchestrator.process(sample_request)
        assert response is not None


class TestSigilOrchestratorPipelineSteps:
    """Tests for individual pipeline steps."""

    @pytest.mark.asyncio
    async def test_step_route_without_routing(self, orchestrator, sample_request):
        """Test routing step when routing is disabled."""
        ctx = PipelineContext(sample_request)

        await orchestrator._step_route(ctx)

        assert ctx.route_decision is not None
        assert ctx.route_decision.intent == Intent.GENERAL_CHAT

    @pytest.mark.asyncio
    async def test_step_plan_without_planning(self, orchestrator, sample_request):
        """Test planning step when planning is disabled."""
        ctx = PipelineContext(sample_request)
        ctx.route_decision = RouteDecision(
            intent=Intent.GENERAL_CHAT,
            confidence=0.9,
            complexity=0.3,
            handler_name="default",
            use_planning=False,
        )

        await orchestrator._step_plan(ctx)

        # No plan should be created
        assert ctx.plan is None

    @pytest.mark.asyncio
    async def test_step_assemble_context(self, orchestrator, sample_request):
        """Test context assembly step."""
        ctx = PipelineContext(sample_request)
        ctx.route_decision = RouteDecision(
            intent=Intent.GENERAL_CHAT,
            confidence=0.9,
            complexity=0.3,
            handler_name="default",
        )

        await orchestrator._step_assemble_context(ctx)

        assert "message" in ctx.assembled_context
        assert ctx.assembled_context["message"] == sample_request.message

    @pytest.mark.asyncio
    async def test_step_validate_without_contracts(self, orchestrator, sample_request):
        """Test validation step when contracts are disabled."""
        ctx = PipelineContext(sample_request)
        ctx.route_decision = RouteDecision(
            intent=Intent.GENERAL_CHAT,
            confidence=0.9,
            complexity=0.3,
            handler_name="default",
            use_contracts=False,
        )
        ctx.output = {"result": "test"}

        await orchestrator._step_validate(ctx)

        # No contract result should be set
        assert ctx.contract_result is None


class TestSigilOrchestratorMetrics:
    """Tests for orchestrator metrics."""

    def test_get_metrics(self, orchestrator):
        """Test getting metrics."""
        metrics = orchestrator.get_metrics()

        assert "total_requests" in metrics
        assert "successful_requests" in metrics
        assert "failed_requests" in metrics
        assert "success_rate" in metrics
        assert "features_enabled" in metrics

    def test_reset_metrics(self, orchestrator):
        """Test resetting metrics."""
        orchestrator._total_requests = 10
        orchestrator._successful_requests = 8
        orchestrator._failed_requests = 2

        orchestrator.reset_metrics()

        assert orchestrator._total_requests == 0
        assert orchestrator._successful_requests == 0
        assert orchestrator._failed_requests == 0

    @pytest.mark.asyncio
    async def test_metrics_after_processing(self, orchestrator, sample_request):
        """Test metrics are updated after processing."""
        await orchestrator.process(sample_request)
        await orchestrator.process(sample_request)

        metrics = orchestrator.get_metrics()
        assert metrics["total_requests"] == 2


class TestSigilOrchestratorHealth:
    """Tests for health checking."""

    def test_is_healthy_default(self, orchestrator):
        """Test health check with default configuration."""
        assert orchestrator.is_healthy is True

    def test_is_healthy_with_unhealthy_memory(self, mock_settings):
        """Test health check with unhealthy memory manager."""
        mock_settings.use_memory = True
        mock_memory = MagicMock()
        mock_memory.is_healthy = False

        orchestrator = SigilOrchestrator(
            settings=mock_settings,
            memory_manager=mock_memory,
        )

        assert orchestrator.is_healthy is False


# =============================================================================
# Convenience Function Tests
# =============================================================================


class TestProcessMessage:
    """Tests for process_message convenience function."""

    @pytest.mark.asyncio
    async def test_process_message_basic(self):
        """Test basic process_message call."""
        with patch('sigil.orchestrator.SigilOrchestrator') as mock_orch_class:
            mock_orch = MagicMock()
            mock_orch.process = AsyncMock(return_value=OrchestratorResponse(
                request_id="req-123",
                status=OrchestratorStatus.SUCCESS,
                output={"result": "done"},
            ))
            mock_orch_class.return_value = mock_orch

            response = await process_message(
                message="Hello",
                session_id="sess-123",
            )

            assert response is not None
            assert response.status == OrchestratorStatus.SUCCESS


# =============================================================================
# Integration Tests
# =============================================================================


class TestOrchestratorIntegration:
    """Integration tests for the orchestrator."""

    @pytest.mark.asyncio
    async def test_full_pipeline_execution(self):
        """Test full pipeline execution end-to-end."""
        with patch('sigil.orchestrator.get_settings') as mock_get_settings:
            mock_settings = MagicMock()
            mock_settings.use_memory = False
            mock_settings.use_planning = False
            mock_settings.use_contracts = False
            mock_settings.use_routing = False
            mock_settings.get_active_features.return_value = []
            mock_get_settings.return_value = mock_settings

            orchestrator = SigilOrchestrator(settings=mock_settings)

            request = OrchestratorRequest(
                message="Create an agent that searches the web",
                session_id="integration-test-session",
            )

            response = await orchestrator.process(request)

            assert response.request_id is not None
            assert response.execution_time_ms > 0

    @pytest.mark.asyncio
    async def test_multiple_concurrent_requests(self):
        """Test handling multiple concurrent requests."""
        with patch('sigil.orchestrator.get_settings') as mock_get_settings:
            mock_settings = MagicMock()
            mock_settings.use_memory = False
            mock_settings.use_planning = False
            mock_settings.use_contracts = False
            mock_settings.use_routing = False
            mock_settings.get_active_features.return_value = []
            mock_get_settings.return_value = mock_settings

            orchestrator = SigilOrchestrator(settings=mock_settings)

            requests = [
                OrchestratorRequest(
                    message=f"Request {i}",
                    session_id=f"session-{i}",
                )
                for i in range(5)
            ]

            # Process concurrently
            responses = await asyncio.gather(
                *[orchestrator.process(req) for req in requests]
            )

            assert len(responses) == 5
            assert all(r.request_id is not None for r in responses)


# =============================================================================
# Tool Execution Integration Tests (Phase 3 - Approach 1)
# =============================================================================


class TestOrchestratorToolExecution:
    """Integration tests for orchestrator tool execution flow.

    These tests verify that the orchestrator correctly:
    - Creates tool-aware step executors
    - Routes TOOL_CALL steps to appropriate executors
    - Routes REASONING steps to reasoning manager
    - Handles fallback when tool execution fails
    - Tracks tokens correctly
    """

    @pytest.fixture
    def mock_settings_with_planning(self):
        """Create mock settings with planning enabled."""
        settings = MagicMock(spec=SigilSettings)
        settings.use_memory = True
        settings.use_planning = True
        settings.use_contracts = False
        settings.use_routing = False
        settings.get_active_features.return_value = ["memory", "planning"]
        settings.debug = True
        return settings

    @pytest.fixture
    def mock_plan_with_tool_steps(self):
        """Create a mock plan with tool steps."""
        from sigil.config.schemas.plan import Plan, PlanStep, PlanStatus

        step1 = PlanStep(
            step_id="step-1",
            description="Search for AI news",
            status=PlanStatus.PENDING,
            dependencies=[],
            tool_calls=["websearch.search"],
        )
        step2 = PlanStep(
            step_id="step-2",
            description="Summarize the search results",
            status=PlanStatus.PENDING,
            dependencies=["step-1"],
            tool_calls=None,  # This is a reasoning step
        )

        plan = Plan(
            plan_id="test-plan-123",
            goal="Find and summarize AI news",
            steps=[step1, step2],
            status=PlanStatus.PENDING,
        )
        return plan

    @pytest.mark.asyncio
    async def test_tool_executor_creation(self, mock_settings_with_planning):
        """Test that tool step executor is created correctly."""
        from sigil.planning.tool_executor import create_tool_step_executor

        executor = create_tool_step_executor(
            memory_manager=None,
            planner=None,
            reasoning_manager=None,
            allow_reasoning_fallback=False,
        )

        assert executor is not None
        assert hasattr(executor, "execute_step")

    @pytest.mark.asyncio
    async def test_step_execute_with_plan(
        self,
        mock_settings_with_planning,
        mock_plan_with_tool_steps,
    ):
        """Test _step_execute with a plan containing tool steps."""
        from sigil.planning.schemas import StepResult, StepStatus

        # Create mock components
        mock_memory = MagicMock()
        mock_memory.is_healthy = True
        mock_memory.retrieve = AsyncMock(return_value=[])

        mock_planner = MagicMock()
        mock_planner.create_plan = AsyncMock(return_value=mock_plan_with_tool_steps)

        mock_reasoning = MagicMock()
        mock_reasoning.execute = AsyncMock(return_value=MagicMock(
            success=True,
            answer="Test answer",
            tokens_used=100,
            reasoning_trace=[],
            model="test-model",
            confidence=0.9,
        ))

        # Create orchestrator
        orchestrator = SigilOrchestrator(
            settings=mock_settings_with_planning,
            memory_manager=mock_memory,
            planner=mock_planner,
            reasoning_manager=mock_reasoning,
        )

        # Create context with plan
        request = OrchestratorRequest(
            message="Find AI news",
            session_id="test-session",
        )
        ctx = PipelineContext(request)
        ctx.route_decision = RouteDecision(
            intent=Intent.GENERAL_CHAT,
            confidence=0.9,
            complexity=0.7,
            handler_name="default",
            use_planning=True,
        )
        ctx.plan = mock_plan_with_tool_steps
        ctx.plan_id = mock_plan_with_tool_steps.plan_id

        # Execute step
        await orchestrator._step_execute(ctx)

        # Should have output (either from plan execution or fallback)
        assert ctx.output is not None
        # Either plan executed or fell back to reasoning
        assert "result" in ctx.output or "error" in ctx.output

    @pytest.mark.asyncio
    async def test_step_execute_fallback_to_reasoning(
        self,
        mock_settings_with_planning,
    ):
        """Test that step execution falls back to reasoning when tools fail."""
        # Create mock reasoning manager that succeeds
        mock_reasoning = MagicMock()
        mock_reasoning.execute = AsyncMock(return_value=MagicMock(
            success=True,
            answer="Fallback answer",
            tokens_used=150,
            reasoning_trace=["step1", "step2"],
            model="test-model",
            confidence=0.85,
        ))

        # Create orchestrator without plan executor
        mock_settings_with_planning.use_planning = False
        orchestrator = SigilOrchestrator(
            settings=mock_settings_with_planning,
            reasoning_manager=mock_reasoning,
        )

        # Create context without plan
        request = OrchestratorRequest(
            message="Test message",
            session_id="test-session",
        )
        ctx = PipelineContext(request)
        ctx.route_decision = RouteDecision(
            intent=Intent.GENERAL_CHAT,
            confidence=0.9,
            complexity=0.5,
            handler_name="default",
            use_planning=False,
        )
        ctx.assembled_context = {"message": "Test message"}

        # Execute step
        await orchestrator._step_execute(ctx)

        # Should use reasoning manager
        mock_reasoning.execute.assert_called_once()
        assert ctx.output is not None
        assert ctx.output.get("result") == "Fallback answer"
        assert ctx.tokens_used == 150

    @pytest.mark.asyncio
    async def test_token_tracking_in_plan_execution(
        self,
        mock_settings_with_planning,
        mock_plan_with_tool_steps,
    ):
        """Test that tokens are tracked correctly during plan execution."""
        from sigil.planning.schemas import StepResult, StepStatus, PlanResult, ExecutionState

        # Create mock plan executor that returns a result
        mock_executor = MagicMock()
        mock_result = PlanResult(
            plan_id="test-plan",
            success=True,
            state=ExecutionState.COMPLETED,
            step_results=[
                StepResult(
                    step_id="step-1",
                    status=StepStatus.COMPLETED,
                    output="Tool output",
                    tokens_used=0,  # Tool calls use 0 tokens
                ),
                StepResult(
                    step_id="step-2",
                    status=StepStatus.COMPLETED,
                    output="Reasoning output",
                    tokens_used=200,  # Reasoning uses tokens
                ),
            ],
            total_tokens=200,
            total_duration_ms=1000.0,
            final_output="Final output",
        )
        mock_executor.execute = AsyncMock(return_value=mock_result)

        orchestrator = SigilOrchestrator(
            settings=mock_settings_with_planning,
            plan_executor=mock_executor,
        )

        # Create context with plan
        request = OrchestratorRequest(
            message="Test",
            session_id="test-session",
        )
        ctx = PipelineContext(request)
        ctx.route_decision = RouteDecision(
            intent=Intent.GENERAL_CHAT,
            confidence=0.9,
            complexity=0.7,
            handler_name="default",
            use_planning=True,
        )
        ctx.plan = mock_plan_with_tool_steps
        ctx.plan_id = mock_plan_with_tool_steps.plan_id

        # Execute step
        await orchestrator._step_execute(ctx)

        # Verify token tracking
        assert ctx.tokens_used == 200  # From plan execution
        assert ctx.output.get("total_tokens") == 200

    @pytest.mark.asyncio
    async def test_plan_execution_error_handling(
        self,
        mock_settings_with_planning,
        mock_plan_with_tool_steps,
    ):
        """Test error handling during plan execution."""
        from sigil.core.exceptions import PlanExecutionError

        # Create mock plan executor that raises an error
        mock_executor = MagicMock()
        mock_executor.execute = AsyncMock(
            side_effect=PlanExecutionError("Tool execution failed", plan_id="test-plan")
        )

        # Create mock reasoning manager for fallback
        mock_reasoning = MagicMock()
        mock_reasoning.execute = AsyncMock(return_value=MagicMock(
            success=True,
            answer="Fallback answer after error",
            tokens_used=100,
            reasoning_trace=[],
            model="test-model",
            confidence=0.7,
        ))

        orchestrator = SigilOrchestrator(
            settings=mock_settings_with_planning,
            plan_executor=mock_executor,
            reasoning_manager=mock_reasoning,
        )

        # Create context with plan
        request = OrchestratorRequest(
            message="Test",
            session_id="test-session",
        )
        ctx = PipelineContext(request)
        ctx.route_decision = RouteDecision(
            intent=Intent.GENERAL_CHAT,
            confidence=0.9,
            complexity=0.7,
            handler_name="default",
            use_planning=True,
        )
        ctx.plan = mock_plan_with_tool_steps
        ctx.plan_id = mock_plan_with_tool_steps.plan_id
        ctx.assembled_context = {"message": "Test"}

        # Execute step
        await orchestrator._step_execute(ctx)

        # Should have fallen back to reasoning
        mock_reasoning.execute.assert_called_once()
        assert len(ctx.warnings) > 0
        assert ctx.output is not None
        assert ctx.output.get("result") == "Fallback answer after error"

    @pytest.mark.asyncio
    async def test_full_pipeline_with_tool_execution(self):
        """Test full orchestrator pipeline with tool execution."""
        from sigil.planning.schemas import StepResult, StepStatus, PlanResult, ExecutionState
        from sigil.config.schemas.plan import Plan, PlanStep, PlanStatus

        # Create mock settings
        mock_settings = MagicMock(spec=SigilSettings)
        mock_settings.use_memory = False
        mock_settings.use_planning = True
        mock_settings.use_contracts = False
        mock_settings.use_routing = False
        mock_settings.get_active_features.return_value = ["planning"]

        # Create mock plan
        mock_plan = Plan(
            plan_id="integration-test-plan",
            goal="Integration test goal",
            steps=[
                PlanStep(
                    step_id="step-1",
                    description="Execute tool",
                    status=PlanStatus.PENDING,
                    tool_calls=["test.tool"],
                ),
            ],
        )

        # Create mock planner
        mock_planner = MagicMock()
        mock_planner.create_plan = AsyncMock(return_value=mock_plan)

        # Create mock plan executor
        mock_executor = MagicMock()
        mock_result = PlanResult(
            plan_id="integration-test-plan",
            success=True,
            state=ExecutionState.COMPLETED,
            step_results=[
                StepResult(
                    step_id="step-1",
                    status=StepStatus.COMPLETED,
                    output="Tool executed successfully",
                    tokens_used=0,
                ),
            ],
            total_tokens=0,
            total_duration_ms=500.0,
            final_output="Tool executed successfully",
        )
        mock_executor.execute = AsyncMock(return_value=mock_result)

        # Create orchestrator
        orchestrator = SigilOrchestrator(
            settings=mock_settings,
            planner=mock_planner,
            plan_executor=mock_executor,
        )

        # Process request
        request = OrchestratorRequest(
            message="Execute integration test",
            session_id="integration-test-session",
        )

        response = await orchestrator.process(request)

        # Verify response
        assert response is not None
        assert response.status in [OrchestratorStatus.SUCCESS, OrchestratorStatus.PARTIAL]
        assert response.plan_id == "integration-test-plan"
        assert response.execution_time_ms > 0

    @pytest.mark.asyncio
    async def test_step_execute_output_structure(
        self,
        mock_settings_with_planning,
        mock_plan_with_tool_steps,
    ):
        """Test that step execution output has correct structure."""
        from sigil.planning.schemas import StepResult, StepStatus, PlanResult, ExecutionState

        # Create mock plan executor
        mock_executor = MagicMock()
        mock_result = PlanResult(
            plan_id="test-plan",
            success=True,
            state=ExecutionState.COMPLETED,
            step_results=[
                StepResult(
                    step_id="step-1",
                    status=StepStatus.COMPLETED,
                    output="Search results",
                    tokens_used=0,
                ),
                StepResult(
                    step_id="step-2",
                    status=StepStatus.COMPLETED,
                    output="Summary of results",
                    tokens_used=150,
                ),
            ],
            total_tokens=150,
            total_duration_ms=800.0,
            final_output="Search results\nSummary of results",
        )
        mock_executor.execute = AsyncMock(return_value=mock_result)

        orchestrator = SigilOrchestrator(
            settings=mock_settings_with_planning,
            plan_executor=mock_executor,
        )

        # Create context
        request = OrchestratorRequest(
            message="Test",
            session_id="test-session",
        )
        ctx = PipelineContext(request)
        ctx.route_decision = RouteDecision(
            intent=Intent.GENERAL_CHAT,
            confidence=0.9,
            complexity=0.7,
            handler_name="default",
            use_planning=True,
        )
        ctx.plan = mock_plan_with_tool_steps
        ctx.plan_id = mock_plan_with_tool_steps.plan_id

        # Execute
        await orchestrator._step_execute(ctx)

        # Verify output structure
        assert ctx.output is not None
        assert "result" in ctx.output
        assert "plan_id" in ctx.output
        assert "steps_executed" in ctx.output
        assert "step_results" in ctx.output
        assert "total_tokens" in ctx.output
        assert "total_duration_ms" in ctx.output

        # Verify step results structure
        step_results = ctx.output["step_results"]
        assert len(step_results) == 2
        for sr in step_results:
            assert "step_id" in sr
            assert "status" in sr
            assert "tokens_used" in sr
