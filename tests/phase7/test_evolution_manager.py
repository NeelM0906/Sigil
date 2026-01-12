"""Tests for EvolutionManager.

This module contains comprehensive tests for the evolution system,
including safety constraints, optimization, and version tracking.

Test Categories:
    - EvolutionConfig tests
    - SafetyChecker tests
    - Optimizer tests
    - EvolutionManager tests
    - Version tracking tests
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timezone

from sigil.evolution.manager import (
    EvolutionManager,
    EvolutionConfig,
    EvolutionStatus,
    EvolutionResult,
    OptimizationMethod,
    TestCase,
    TestResult,
    EvaluationResult,
    PromptVersion,
    SafetyChecker,
    TextGradOptimizer,
    EvolutionaryOptimizer,
    DEFAULT_MAX_GENERATIONS,
    DEFAULT_MIN_IMPROVEMENT_THRESHOLD,
    DEFAULT_MAX_PROMPT_DRIFT,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def evolution_config():
    """Create default evolution config."""
    return EvolutionConfig()


@pytest.fixture
def evolution_config_strict():
    """Create strict evolution config."""
    return EvolutionConfig(
        max_generations=3,
        min_improvement_threshold=0.1,
        max_prompt_drift=0.2,
        min_confidence=0.9,
        test_pass_threshold=0.95,
    )


@pytest.fixture
def safety_checker(evolution_config):
    """Create safety checker."""
    return SafetyChecker(evolution_config)


@pytest.fixture
def test_suite():
    """Create sample test suite."""
    return [
        TestCase(
            test_id="test-1",
            input="What is 2+2?",
            expected_output="4",
            weight=1.0,
        ),
        TestCase(
            test_id="test-2",
            input="Greet the user",
            expected_output="Hello!",
            weight=1.0,
        ),
        TestCase(
            test_id="test-3",
            input="Say goodbye",
            expected_output="Goodbye!",
            weight=1.0,
        ),
    ]


@pytest.fixture
def evolution_manager():
    """Create evolution manager."""
    return EvolutionManager()


@pytest.fixture
def mock_agent_runner():
    """Create mock agent runner."""
    async def runner(prompt: str, input: str) -> str:
        if "2+2" in input:
            return "4"
        elif "greet" in input.lower():
            return "Hello!"
        elif "goodbye" in input.lower():
            return "Goodbye!"
        return "Unknown"

    return runner


# =============================================================================
# EvolutionConfig Tests
# =============================================================================


class TestEvolutionConfig:
    """Tests for EvolutionConfig dataclass."""

    def test_config_defaults(self):
        """Test default configuration values."""
        config = EvolutionConfig()

        assert config.max_generations == DEFAULT_MAX_GENERATIONS
        assert config.min_improvement_threshold == DEFAULT_MIN_IMPROVEMENT_THRESHOLD
        assert config.max_prompt_drift == DEFAULT_MAX_PROMPT_DRIFT
        assert config.optimization_method == OptimizationMethod.TEXTGRAD

    def test_config_custom(self):
        """Test custom configuration."""
        config = EvolutionConfig(
            max_generations=5,
            min_improvement_threshold=0.05,
            max_prompt_drift=0.5,
            optimization_method=OptimizationMethod.EVOLUTIONARY,
        )

        assert config.max_generations == 5
        assert config.min_improvement_threshold == 0.05
        assert config.optimization_method == OptimizationMethod.EVOLUTIONARY

    def test_config_to_dict(self):
        """Test configuration serialization."""
        config = EvolutionConfig()
        data = config.to_dict()

        assert "max_generations" in data
        assert "optimization_method" in data
        assert data["optimization_method"] == "textgrad"


# =============================================================================
# TestCase and TestResult Tests
# =============================================================================


class TestTestCase:
    """Tests for TestCase dataclass."""

    def test_test_case_creation(self):
        """Test creating a test case."""
        tc = TestCase(
            test_id="test-1",
            input="Hello",
            expected_output="Hi",
        )

        assert tc.test_id == "test-1"
        assert tc.weight == 1.0

    def test_test_case_with_weight(self):
        """Test test case with custom weight."""
        tc = TestCase(
            test_id="test-1",
            input="Hello",
            expected_output="Hi",
            weight=2.0,
        )

        assert tc.weight == 2.0


class TestTestResult:
    """Tests for TestResult dataclass."""

    def test_test_result_passed(self):
        """Test passing result."""
        result = TestResult(
            test_id="test-1",
            passed=True,
            score=1.0,
            actual_output="Hi",
        )

        assert result.passed is True
        assert result.score == 1.0

    def test_test_result_failed(self):
        """Test failed result."""
        result = TestResult(
            test_id="test-1",
            passed=False,
            score=0.3,
            error="Output mismatch",
        )

        assert result.passed is False
        assert result.error == "Output mismatch"


# =============================================================================
# EvaluationResult Tests
# =============================================================================


class TestEvaluationResult:
    """Tests for EvaluationResult dataclass."""

    def test_evaluation_result_empty(self):
        """Test empty evaluation result."""
        result = EvaluationResult(overall_score=0.0)

        assert result.total_tests == 0
        assert result.pass_rate == 0.0

    def test_evaluation_result_with_tests(self):
        """Test evaluation result with test results."""
        result = EvaluationResult(
            overall_score=0.8,
            test_results=[
                TestResult(test_id="1", passed=True, score=1.0),
                TestResult(test_id="2", passed=True, score=0.9),
                TestResult(test_id="3", passed=False, score=0.5),
            ],
        )

        assert result.total_tests == 3
        assert result.passed_tests == 2
        assert result.failed_tests == 1
        assert result.pass_rate == pytest.approx(0.666, rel=0.01)


# =============================================================================
# PromptVersion Tests
# =============================================================================


class TestPromptVersion:
    """Tests for PromptVersion dataclass."""

    def test_version_creation(self):
        """Test creating a version."""
        version = PromptVersion(
            version_id="v1",
            prompt="You are an assistant.",
        )

        assert version.version_id == "v1"
        assert version.score == 0.0
        assert version.generation == 0

    def test_version_to_dict(self):
        """Test version serialization."""
        version = PromptVersion(
            version_id="v1",
            prompt="You are an assistant.",
            score=0.85,
            confidence=0.9,
        )

        data = version.to_dict()
        assert data["version_id"] == "v1"
        assert data["score"] == 0.85


# =============================================================================
# SafetyChecker Tests
# =============================================================================


class TestSafetyChecker:
    """Tests for SafetyChecker class."""

    def test_check_improvement_passed(self, safety_checker):
        """Test improvement check passes."""
        passed, reason = safety_checker.check_improvement(
            initial_score=0.7,
            new_score=0.8,
        )

        assert passed is True
        assert "meets threshold" in reason

    def test_check_improvement_failed(self, safety_checker):
        """Test improvement check fails for small improvement."""
        passed, reason = safety_checker.check_improvement(
            initial_score=0.7,
            new_score=0.705,  # Only 0.7% improvement
        )

        assert passed is False
        assert "below threshold" in reason

    def test_check_drift_passed(self, safety_checker):
        """Test drift check passes."""
        passed, reason = safety_checker.check_drift(drift=0.1)

        assert passed is True
        assert "within limits" in reason

    def test_check_drift_failed(self, safety_checker):
        """Test drift check fails for high drift."""
        passed, reason = safety_checker.check_drift(drift=0.5)

        assert passed is False
        assert "exceeds max" in reason

    def test_check_confidence_passed(self, safety_checker):
        """Test confidence check passes."""
        passed, reason = safety_checker.check_confidence(confidence=0.8)

        assert passed is True

    def test_check_confidence_failed(self, safety_checker):
        """Test confidence check fails for low confidence."""
        passed, reason = safety_checker.check_confidence(confidence=0.5)

        assert passed is False
        assert "below minimum" in reason

    def test_check_test_pass_rate_passed(self, safety_checker):
        """Test pass rate check passes."""
        passed, reason = safety_checker.check_test_pass_rate(pass_rate=0.95)

        assert passed is True

    def test_check_test_pass_rate_failed(self, safety_checker):
        """Test pass rate check fails."""
        passed, reason = safety_checker.check_test_pass_rate(pass_rate=0.7)

        assert passed is False

    def test_run_all_checks(self, safety_checker):
        """Test running all checks."""
        checks = safety_checker.run_all_checks(
            initial_score=0.7,
            new_score=0.85,
            drift=0.1,
            confidence=0.9,
            pass_rate=0.95,
        )

        assert "improvement" in checks
        assert "drift" in checks
        assert "confidence" in checks
        assert "test_pass_rate" in checks

    def test_all_passed_true(self, safety_checker):
        """Test all_passed returns True when all pass."""
        checks = {
            "improvement": (True, "ok"),
            "drift": (True, "ok"),
            "confidence": (True, "ok"),
            "test_pass_rate": (True, "ok"),
        }

        assert safety_checker.all_passed(checks) is True

    def test_all_passed_false(self, safety_checker):
        """Test all_passed returns False when any fails."""
        checks = {
            "improvement": (True, "ok"),
            "drift": (False, "failed"),
            "confidence": (True, "ok"),
            "test_pass_rate": (True, "ok"),
        }

        assert safety_checker.all_passed(checks) is False


# =============================================================================
# Optimizer Tests
# =============================================================================


class TestTextGradOptimizer:
    """Tests for TextGradOptimizer."""

    @pytest.mark.asyncio
    async def test_optimize_without_llm(self):
        """Test optimization without LLM produces variations."""
        optimizer = TextGradOptimizer(llm_call=None)

        candidates = await optimizer.optimize(
            current_prompt="You are an assistant.",
            evaluation_result=EvaluationResult(overall_score=0.7),
            failed_cases=[],
        )

        assert len(candidates) > 0
        # Should produce variations of the prompt
        assert all("assistant" in c or "expert" in c.lower() for c in candidates)

    @pytest.mark.asyncio
    async def test_optimize_with_llm(self):
        """Test optimization with LLM."""
        async def mock_llm(prompt: str) -> str:
            return "Improved prompt 1\n---\nImproved prompt 2\n---\nImproved prompt 3"

        optimizer = TextGradOptimizer(llm_call=mock_llm, num_candidates=3)

        candidates = await optimizer.optimize(
            current_prompt="You are an assistant.",
            evaluation_result=EvaluationResult(overall_score=0.7),
            failed_cases=[],
        )

        assert len(candidates) == 3
        assert "Improved prompt 1" in candidates


class TestEvolutionaryOptimizer:
    """Tests for EvolutionaryOptimizer."""

    @pytest.mark.asyncio
    async def test_optimize_produces_candidates(self):
        """Test that optimizer produces candidates."""
        optimizer = EvolutionaryOptimizer(
            mutation_rate=0.1,
            num_candidates=5,
        )

        candidates = await optimizer.optimize(
            current_prompt="You are an assistant.",
            evaluation_result=EvaluationResult(overall_score=0.7),
            failed_cases=[],
        )

        assert len(candidates) == 5


# =============================================================================
# EvolutionManager Tests
# =============================================================================


class TestEvolutionManagerInit:
    """Tests for EvolutionManager initialization."""

    def test_init_default(self):
        """Test default initialization."""
        manager = EvolutionManager()

        assert manager._config is not None
        assert manager._optimizer is not None
        assert manager._safety_checker is not None

    def test_init_custom_config(self, evolution_config_strict):
        """Test initialization with custom config."""
        manager = EvolutionManager(config=evolution_config_strict)

        assert manager._config.max_generations == 3

    def test_init_custom_optimizer(self):
        """Test initialization with custom optimizer."""
        optimizer = EvolutionaryOptimizer()
        manager = EvolutionManager(optimizer=optimizer)

        assert isinstance(manager._optimizer, EvolutionaryOptimizer)


class TestEvolutionManagerEvolve:
    """Tests for EvolutionManager evolve method."""

    @pytest.mark.asyncio
    async def test_evolve_basic(self, evolution_manager, test_suite, mock_agent_runner):
        """Test basic evolution."""
        result = await evolution_manager.evolve(
            agent_name="test-agent",
            current_prompt="You are a helpful assistant.",
            test_suite=test_suite,
            agent_runner=mock_agent_runner,
            generations=1,
        )

        assert isinstance(result, EvolutionResult)
        assert result.evolution_id is not None
        assert result.agent_name == "test-agent"
        assert result.initial_score > 0

    @pytest.mark.asyncio
    async def test_evolve_tracks_versions(self, evolution_manager, test_suite, mock_agent_runner):
        """Test that evolution tracks version history."""
        result = await evolution_manager.evolve(
            agent_name="test-agent",
            current_prompt="You are a helpful assistant.",
            test_suite=test_suite,
            agent_runner=mock_agent_runner,
            generations=2,
        )

        assert len(result.version_history) >= 1

    @pytest.mark.asyncio
    async def test_evolve_respects_max_generations(self, test_suite, mock_agent_runner):
        """Test that evolution respects max generations."""
        config = EvolutionConfig(max_generations=2)
        manager = EvolutionManager(config=config)

        result = await manager.evolve(
            agent_name="test-agent",
            current_prompt="You are a helpful assistant.",
            test_suite=test_suite,
            agent_runner=mock_agent_runner,
        )

        assert result.generations_run <= 2

    @pytest.mark.asyncio
    async def test_evolve_safety_blocks(self, test_suite):
        """Test that safety checks can block evolution."""
        config = EvolutionConfig(
            min_improvement_threshold=0.99,  # Very high threshold
            max_generations=1,
        )
        manager = EvolutionManager(config=config)

        # Agent that always returns the same thing
        async def bad_runner(prompt: str, input: str) -> str:
            return "wrong"

        result = await manager.evolve(
            agent_name="test-agent",
            current_prompt="Bad prompt",
            test_suite=test_suite,
            agent_runner=bad_runner,
        )

        # Should be blocked or no improvement
        assert result.status in [
            EvolutionStatus.SAFETY_BLOCKED,
            EvolutionStatus.NO_IMPROVEMENT,
            EvolutionStatus.FAILED,
        ]


class TestEvolutionManagerEvaluate:
    """Tests for evaluation."""

    @pytest.mark.asyncio
    async def test_evaluate_perfect_score(self, evolution_manager, test_suite, mock_agent_runner):
        """Test evaluation with perfect scores."""
        result = await evolution_manager._evaluate(
            prompt="Perfect prompt",
            test_suite=test_suite,
            agent_runner=mock_agent_runner,
        )

        assert result.overall_score > 0
        assert result.total_tests == 3


class TestEvolutionManagerScoring:
    """Tests for output scoring."""

    def test_score_exact_match(self, evolution_manager):
        """Test scoring exact matches."""
        score = evolution_manager._score_output(
            actual="Hello",
            expected="Hello",
        )

        assert score == 1.0

    def test_score_string_similarity(self, evolution_manager):
        """Test scoring similar strings."""
        score = evolution_manager._score_output(
            actual="hello world",
            expected="hello there",
        )

        assert 0 < score < 1.0

    def test_score_dict_matching(self, evolution_manager):
        """Test scoring dict matching."""
        score = evolution_manager._score_output(
            actual={"a": 1, "b": 2},
            expected={"a": 1, "b": 3},
        )

        assert score == 0.5  # 1 out of 2 values match


class TestEvolutionManagerDrift:
    """Tests for drift calculation."""

    def test_calculate_drift_identical(self, evolution_manager):
        """Test drift calculation for identical prompts."""
        drift = evolution_manager._calculate_drift(
            original="You are an assistant.",
            candidate="You are an assistant.",
        )

        assert drift == 0.0

    def test_calculate_drift_different(self, evolution_manager):
        """Test drift calculation for different prompts."""
        drift = evolution_manager._calculate_drift(
            original="You are a helpful assistant.",
            candidate="I am a robot.",
        )

        assert 0 < drift < 1.0


class TestEvolutionManagerVersionHistory:
    """Tests for version history management."""

    @pytest.mark.asyncio
    async def test_get_version_history(self, evolution_manager, test_suite, mock_agent_runner):
        """Test getting version history."""
        await evolution_manager.evolve(
            agent_name="test-agent",
            current_prompt="You are a helpful assistant.",
            test_suite=test_suite,
            agent_runner=mock_agent_runner,
            generations=1,
        )

        history = evolution_manager.get_version_history("test-agent")
        assert len(history) >= 1

    @pytest.mark.asyncio
    async def test_get_best_version(self, evolution_manager, test_suite, mock_agent_runner):
        """Test getting best version."""
        await evolution_manager.evolve(
            agent_name="test-agent",
            current_prompt="You are a helpful assistant.",
            test_suite=test_suite,
            agent_runner=mock_agent_runner,
            generations=1,
        )

        best = evolution_manager.get_best_version("test-agent")
        assert best is not None

    def test_get_best_version_not_found(self, evolution_manager):
        """Test getting best version for unknown agent."""
        best = evolution_manager.get_best_version("unknown-agent")
        assert best is None

    @pytest.mark.asyncio
    async def test_rollback(self, evolution_manager, test_suite, mock_agent_runner):
        """Test rollback to previous version."""
        await evolution_manager.evolve(
            agent_name="test-agent",
            current_prompt="You are a helpful assistant.",
            test_suite=test_suite,
            agent_runner=mock_agent_runner,
            generations=1,
        )

        history = evolution_manager.get_version_history("test-agent")
        if history:
            version = evolution_manager.rollback(
                agent_name="test-agent",
                version_id=history[0].version_id,
            )
            assert version is not None


class TestEvolutionManagerMetrics:
    """Tests for evolution metrics."""

    def test_get_metrics(self, evolution_manager):
        """Test getting metrics."""
        metrics = evolution_manager.get_metrics()

        assert "total_evolutions" in metrics
        assert "successful_evolutions" in metrics
        assert "safety_blocks" in metrics

    @pytest.mark.asyncio
    async def test_metrics_updated(self, evolution_manager, test_suite, mock_agent_runner):
        """Test metrics are updated after evolution."""
        await evolution_manager.evolve(
            agent_name="test-agent",
            current_prompt="You are a helpful assistant.",
            test_suite=test_suite,
            agent_runner=mock_agent_runner,
            generations=1,
        )

        metrics = evolution_manager.get_metrics()
        assert metrics["total_evolutions"] == 1
