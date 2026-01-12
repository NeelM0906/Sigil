"""EvolutionManager - Self-improvement with safety constraints for Sigil v2.

This module implements the EvolutionManager, which orchestrates agent
self-improvement through evaluation, optimization, and testing cycles
with strong safety rails.

Key Components:
    - EvolutionManager: Main orchestrator for evolution
    - EvolutionConfig: Configuration with safety constraints
    - EvolutionResult: Results of an evolution cycle
    - PromptVersion: Versioned prompt with metadata

Safety Constraints:
    - max_generations: Maximum evolution iterations
    - min_improvement_threshold: Minimum improvement to accept change
    - max_prompt_drift: Maximum allowed semantic drift from original
    - test_suite_required: Require passing test suite before commit

Evolution Loop:
    1. Evaluate: Score current agent on test suite
    2. Optimize: Generate improved prompt candidates
    3. Test: Validate candidates against test suite
    4. Commit: Accept improvement if passes safety checks

Example:
    >>> from sigil.evolution.manager import EvolutionManager
    >>>
    >>> manager = EvolutionManager()
    >>> result = await manager.evolve(
    ...     agent_name="lead_qualifier",
    ...     test_suite=test_cases,
    ...     generations=5,
    ... )
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
import uuid
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Optional, Awaitable

from sigil.config import get_settings
from sigil.config.settings import SigilSettings
from sigil.core.exceptions import EvolutionError, OptimizationError
from sigil.state.events import Event, EventType, _generate_event_id, _get_utc_now
from sigil.state.store import EventStore


# =============================================================================
# Module Logger
# =============================================================================

logger = logging.getLogger(__name__)


# =============================================================================
# Constants
# =============================================================================

# Default safety constraints
DEFAULT_MAX_GENERATIONS = 10
DEFAULT_MIN_IMPROVEMENT_THRESHOLD = 0.02  # 2% improvement required
DEFAULT_MAX_PROMPT_DRIFT = 0.3  # 30% semantic drift allowed
DEFAULT_MIN_CONFIDENCE = 0.7  # 70% confidence required
DEFAULT_TEST_PASS_THRESHOLD = 0.9  # 90% test pass rate required


# =============================================================================
# Enums
# =============================================================================


class EvolutionStatus(str, Enum):
    """Status of an evolution cycle.

    Attributes:
        PENDING: Not yet started
        EVALUATING: Evaluating current performance
        OPTIMIZING: Generating improvements
        TESTING: Running test suite
        IMPROVED: Successfully improved
        NO_IMPROVEMENT: No improvement found
        FAILED: Evolution failed
        SAFETY_BLOCKED: Blocked by safety constraints
    """
    PENDING = "pending"
    EVALUATING = "evaluating"
    OPTIMIZING = "optimizing"
    TESTING = "testing"
    IMPROVED = "improved"
    NO_IMPROVEMENT = "no_improvement"
    FAILED = "failed"
    SAFETY_BLOCKED = "safety_blocked"


class OptimizationMethod(str, Enum):
    """Optimization method for prompt improvement.

    Attributes:
        TEXTGRAD: TextGrad-style gradient descent
        EVOLUTIONARY: Evolutionary prompt mutation
        BEAM_SEARCH: Beam search with scoring
        RANDOM: Random sampling (baseline)
    """
    TEXTGRAD = "textgrad"
    EVOLUTIONARY = "evolutionary"
    BEAM_SEARCH = "beam_search"
    RANDOM = "random"


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class EvolutionConfig:
    """Configuration for evolution with safety constraints.

    Attributes:
        max_generations: Maximum evolution iterations
        min_improvement_threshold: Minimum improvement to accept
        max_prompt_drift: Maximum semantic drift allowed
        min_confidence: Minimum confidence score required
        test_pass_threshold: Minimum test pass rate required
        optimization_method: Method for optimization
        rollback_on_failure: Whether to rollback on test failure
        require_human_approval: Require human approval for changes
    """
    max_generations: int = DEFAULT_MAX_GENERATIONS
    min_improvement_threshold: float = DEFAULT_MIN_IMPROVEMENT_THRESHOLD
    max_prompt_drift: float = DEFAULT_MAX_PROMPT_DRIFT
    min_confidence: float = DEFAULT_MIN_CONFIDENCE
    test_pass_threshold: float = DEFAULT_TEST_PASS_THRESHOLD
    optimization_method: OptimizationMethod = OptimizationMethod.TEXTGRAD
    rollback_on_failure: bool = True
    require_human_approval: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "max_generations": self.max_generations,
            "min_improvement_threshold": self.min_improvement_threshold,
            "max_prompt_drift": self.max_prompt_drift,
            "min_confidence": self.min_confidence,
            "test_pass_threshold": self.test_pass_threshold,
            "optimization_method": self.optimization_method.value,
            "rollback_on_failure": self.rollback_on_failure,
            "require_human_approval": self.require_human_approval,
        }


@dataclass
class TestCase:
    """A test case for evolution evaluation.

    Attributes:
        test_id: Unique identifier
        input: Test input
        expected_output: Expected output (for comparison)
        weight: Weight for scoring (default 1.0)
        metadata: Additional metadata
    """
    test_id: str
    input: str
    expected_output: Any
    weight: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TestResult:
    """Result of running a test case.

    Attributes:
        test_id: Test case identifier
        passed: Whether the test passed
        score: Score achieved (0.0-1.0)
        actual_output: Actual output from agent
        error: Error message if failed
        execution_time_ms: Time to execute
    """
    test_id: str
    passed: bool
    score: float
    actual_output: Any = None
    error: Optional[str] = None
    execution_time_ms: float = 0.0


@dataclass
class EvaluationResult:
    """Result of evaluating an agent on test suite.

    Attributes:
        overall_score: Overall score (0.0-1.0)
        test_results: Individual test results
        pass_rate: Percentage of tests passed
        total_tests: Total number of tests
        passed_tests: Number of tests passed
        failed_tests: Number of tests failed
        evaluation_time_ms: Total evaluation time
    """
    overall_score: float
    test_results: list[TestResult] = field(default_factory=list)
    pass_rate: float = 0.0
    total_tests: int = 0
    passed_tests: int = 0
    failed_tests: int = 0
    evaluation_time_ms: float = 0.0

    def __post_init__(self) -> None:
        """Calculate statistics."""
        if self.test_results:
            self.total_tests = len(self.test_results)
            self.passed_tests = sum(1 for r in self.test_results if r.passed)
            self.failed_tests = self.total_tests - self.passed_tests
            self.pass_rate = self.passed_tests / self.total_tests if self.total_tests > 0 else 0.0


@dataclass
class PromptVersion:
    """Versioned prompt with metadata.

    Attributes:
        version_id: Unique version identifier
        prompt: The prompt text
        score: Score achieved on test suite
        confidence: Confidence in this version
        drift_from_original: Semantic drift from original
        parent_version: ID of parent version
        generation: Generation number
        created_at: When created
        metadata: Additional metadata
    """
    version_id: str
    prompt: str
    score: float = 0.0
    confidence: float = 1.0
    drift_from_original: float = 0.0
    parent_version: Optional[str] = None
    generation: int = 0
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "version_id": self.version_id,
            "prompt": self.prompt,
            "score": self.score,
            "confidence": self.confidence,
            "drift_from_original": self.drift_from_original,
            "parent_version": self.parent_version,
            "generation": self.generation,
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata,
        }


@dataclass
class EvolutionResult:
    """Result of an evolution cycle.

    Attributes:
        evolution_id: Unique evolution identifier
        agent_name: Name of evolved agent
        status: Final status
        generations_run: Number of generations completed
        initial_score: Score before evolution
        final_score: Score after evolution
        improvement: Absolute improvement
        improvement_percentage: Relative improvement
        best_version: Best prompt version found
        version_history: All versions tried
        safety_checks: Results of safety checks
        total_time_ms: Total evolution time
        errors: Any errors encountered
    """
    evolution_id: str
    agent_name: str
    status: EvolutionStatus
    generations_run: int = 0
    initial_score: float = 0.0
    final_score: float = 0.0
    improvement: float = 0.0
    improvement_percentage: float = 0.0
    best_version: Optional[PromptVersion] = None
    version_history: list[PromptVersion] = field(default_factory=list)
    safety_checks: dict[str, bool] = field(default_factory=dict)
    total_time_ms: float = 0.0
    errors: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Calculate improvement metrics."""
        if self.initial_score > 0:
            self.improvement = self.final_score - self.initial_score
            self.improvement_percentage = self.improvement / self.initial_score

    @property
    def improved(self) -> bool:
        """Check if evolution resulted in improvement."""
        return self.status == EvolutionStatus.IMPROVED

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "evolution_id": self.evolution_id,
            "agent_name": self.agent_name,
            "status": self.status.value,
            "generations_run": self.generations_run,
            "initial_score": self.initial_score,
            "final_score": self.final_score,
            "improvement": self.improvement,
            "improvement_percentage": self.improvement_percentage,
            "best_version": self.best_version.to_dict() if self.best_version else None,
            "safety_checks": self.safety_checks,
            "total_time_ms": self.total_time_ms,
            "errors": self.errors,
        }


# =============================================================================
# Safety Checker
# =============================================================================


class SafetyChecker:
    """Checks safety constraints for evolution.

    Verifies that prompt changes meet safety requirements:
    - Minimum improvement threshold
    - Maximum prompt drift
    - Minimum confidence score
    - Test suite pass rate
    """

    def __init__(self, config: EvolutionConfig) -> None:
        self._config = config

    def check_improvement(
        self,
        initial_score: float,
        new_score: float,
    ) -> tuple[bool, str]:
        """Check if improvement meets threshold.

        Args:
            initial_score: Original score
            new_score: New score

        Returns:
            Tuple of (passed, reason)
        """
        if initial_score == 0:
            return True, "No baseline score"

        improvement = (new_score - initial_score) / initial_score

        if improvement < self._config.min_improvement_threshold:
            return False, (
                f"Improvement {improvement:.2%} below threshold "
                f"{self._config.min_improvement_threshold:.2%}"
            )

        return True, f"Improvement {improvement:.2%} meets threshold"

    def check_drift(self, drift: float) -> tuple[bool, str]:
        """Check if prompt drift is within limits.

        Args:
            drift: Semantic drift from original (0.0-1.0)

        Returns:
            Tuple of (passed, reason)
        """
        if drift > self._config.max_prompt_drift:
            return False, (
                f"Drift {drift:.2%} exceeds max {self._config.max_prompt_drift:.2%}"
            )

        return True, f"Drift {drift:.2%} within limits"

    def check_confidence(self, confidence: float) -> tuple[bool, str]:
        """Check if confidence meets minimum.

        Args:
            confidence: Confidence score (0.0-1.0)

        Returns:
            Tuple of (passed, reason)
        """
        if confidence < self._config.min_confidence:
            return False, (
                f"Confidence {confidence:.2%} below minimum "
                f"{self._config.min_confidence:.2%}"
            )

        return True, f"Confidence {confidence:.2%} meets minimum"

    def check_test_pass_rate(self, pass_rate: float) -> tuple[bool, str]:
        """Check if test pass rate meets threshold.

        Args:
            pass_rate: Test pass rate (0.0-1.0)

        Returns:
            Tuple of (passed, reason)
        """
        if pass_rate < self._config.test_pass_threshold:
            return False, (
                f"Pass rate {pass_rate:.2%} below threshold "
                f"{self._config.test_pass_threshold:.2%}"
            )

        return True, f"Pass rate {pass_rate:.2%} meets threshold"

    def run_all_checks(
        self,
        initial_score: float,
        new_score: float,
        drift: float,
        confidence: float,
        pass_rate: float,
    ) -> dict[str, tuple[bool, str]]:
        """Run all safety checks.

        Args:
            initial_score: Original score
            new_score: New score
            drift: Semantic drift
            confidence: Confidence score
            pass_rate: Test pass rate

        Returns:
            Dictionary of check results
        """
        return {
            "improvement": self.check_improvement(initial_score, new_score),
            "drift": self.check_drift(drift),
            "confidence": self.check_confidence(confidence),
            "test_pass_rate": self.check_test_pass_rate(pass_rate),
        }

    def all_passed(self, checks: dict[str, tuple[bool, str]]) -> bool:
        """Check if all safety checks passed.

        Args:
            checks: Dictionary of check results

        Returns:
            True if all checks passed
        """
        return all(passed for passed, _ in checks.values())


# =============================================================================
# Prompt Optimizer (Abstract)
# =============================================================================


class PromptOptimizer:
    """Base class for prompt optimization.

    Subclasses implement specific optimization strategies.
    """

    async def optimize(
        self,
        current_prompt: str,
        evaluation_result: EvaluationResult,
        failed_cases: list[TestResult],
    ) -> list[str]:
        """Generate optimized prompt candidates.

        Args:
            current_prompt: Current prompt text
            evaluation_result: Recent evaluation result
            failed_cases: Failed test cases for feedback

        Returns:
            List of candidate prompts
        """
        raise NotImplementedError


class TextGradOptimizer(PromptOptimizer):
    """TextGrad-style prompt optimizer.

    Uses gradient-like feedback to improve prompts based on
    failure analysis.
    """

    def __init__(
        self,
        llm_call: Optional[Callable[[str], Awaitable[str]]] = None,
        num_candidates: int = 3,
    ) -> None:
        self._llm_call = llm_call
        self._num_candidates = num_candidates

    async def optimize(
        self,
        current_prompt: str,
        evaluation_result: EvaluationResult,
        failed_cases: list[TestResult],
    ) -> list[str]:
        """Generate improved prompts using TextGrad approach."""
        if not self._llm_call:
            # Without LLM, return minor variations
            return self._simple_variations(current_prompt)

        # Build optimization prompt
        failures_summary = "\n".join([
            f"- Input: {r.actual_output[:100]}... Expected: {r.error}"
            for r in failed_cases[:5]
        ])

        optimization_prompt = f"""Analyze this agent prompt and its failures, then generate {self._num_candidates} improved versions.

CURRENT PROMPT:
{current_prompt}

PERFORMANCE:
- Overall Score: {evaluation_result.overall_score:.2%}
- Pass Rate: {evaluation_result.pass_rate:.2%}

SAMPLE FAILURES:
{failures_summary}

Generate {self._num_candidates} improved prompt versions that address these failures.
Format: Return each version on a new line, separated by "---".

IMPROVED PROMPTS:"""

        try:
            response = await self._llm_call(optimization_prompt)
            candidates = [c.strip() for c in response.split("---") if c.strip()]
            return candidates[:self._num_candidates]
        except Exception as e:
            logger.warning(f"TextGrad optimization failed: {e}")
            return self._simple_variations(current_prompt)

    def _simple_variations(self, prompt: str) -> list[str]:
        """Generate simple variations without LLM."""
        # Add clarifying instructions
        variations = [
            prompt + "\n\nBe precise and thorough in your responses.",
            prompt + "\n\nEnsure all outputs follow the expected format exactly.",
            "You are an expert agent. " + prompt,
        ]
        return variations


class EvolutionaryOptimizer(PromptOptimizer):
    """Evolutionary prompt optimizer.

    Uses mutation and crossover to evolve prompts.
    """

    def __init__(
        self,
        mutation_rate: float = 0.1,
        num_candidates: int = 5,
    ) -> None:
        self._mutation_rate = mutation_rate
        self._num_candidates = num_candidates

    async def optimize(
        self,
        current_prompt: str,
        evaluation_result: EvaluationResult,
        failed_cases: list[TestResult],
    ) -> list[str]:
        """Generate mutated prompt candidates."""
        import random

        candidates = []
        words = current_prompt.split()

        for _ in range(self._num_candidates):
            # Apply mutations
            mutated_words = []
            for word in words:
                if random.random() < self._mutation_rate:
                    # Mutation: duplicate, remove, or swap
                    action = random.choice(["duplicate", "remove", "keep"])
                    if action == "duplicate":
                        mutated_words.extend([word, word])
                    elif action == "remove":
                        pass  # Skip word
                    else:
                        mutated_words.append(word)
                else:
                    mutated_words.append(word)

            candidates.append(" ".join(mutated_words))

        return candidates


# =============================================================================
# Evolution Manager
# =============================================================================


class EvolutionManager:
    """Orchestrates agent self-improvement.

    The EvolutionManager coordinates the evolution cycle:
    1. Evaluate current agent performance
    2. Generate optimized prompt candidates
    3. Test candidates against test suite
    4. Commit improvements if safety checks pass

    Features:
        - Multiple optimization strategies
        - Strong safety constraints
        - Version history tracking
        - Rollback capability
        - Event emission for audit

    Example:
        >>> manager = EvolutionManager()
        >>> result = await manager.evolve(
        ...     agent_name="sales_qualifier",
        ...     current_prompt="You are a sales agent...",
        ...     test_suite=[TestCase(...)],
        ... )
    """

    def __init__(
        self,
        config: Optional[EvolutionConfig] = None,
        optimizer: Optional[PromptOptimizer] = None,
        event_store: Optional[EventStore] = None,
        llm_call: Optional[Callable[[str], Awaitable[str]]] = None,
        settings: Optional[SigilSettings] = None,
    ) -> None:
        """Initialize the EvolutionManager.

        Args:
            config: Evolution configuration
            optimizer: Prompt optimizer instance
            event_store: Event store for audit
            llm_call: Async function for LLM calls
            settings: Framework settings
        """
        self._config = config or EvolutionConfig()
        self._event_store = event_store or EventStore()
        self._settings = settings or get_settings()
        self._llm_call = llm_call

        # Initialize optimizer based on config
        if optimizer:
            self._optimizer = optimizer
        elif self._config.optimization_method == OptimizationMethod.TEXTGRAD:
            self._optimizer = TextGradOptimizer(llm_call=llm_call)
        elif self._config.optimization_method == OptimizationMethod.EVOLUTIONARY:
            self._optimizer = EvolutionaryOptimizer()
        else:
            self._optimizer = TextGradOptimizer(llm_call=llm_call)

        # Safety checker
        self._safety_checker = SafetyChecker(self._config)

        # Version history
        self._version_history: dict[str, list[PromptVersion]] = {}

        # Metrics
        self._total_evolutions: int = 0
        self._successful_evolutions: int = 0
        self._safety_blocks: int = 0

    async def evolve(
        self,
        agent_name: str,
        current_prompt: str,
        test_suite: list[TestCase],
        agent_runner: Callable[[str, str], Awaitable[Any]],
        generations: Optional[int] = None,
    ) -> EvolutionResult:
        """Run evolution cycle for an agent.

        Args:
            agent_name: Name of the agent
            current_prompt: Current prompt to evolve
            test_suite: Test suite for evaluation
            agent_runner: Function to run agent with (prompt, input) -> output
            generations: Optional generation limit override

        Returns:
            EvolutionResult with evolution outcome

        Example:
            >>> async def runner(prompt, input):
            ...     # Run agent with prompt
            ...     return result
            >>> result = await manager.evolve(
            ...     agent_name="my_agent",
            ...     current_prompt="...",
            ...     test_suite=[...],
            ...     agent_runner=runner,
            ... )
        """
        evolution_id = str(uuid.uuid4())
        start_time = time.perf_counter()
        max_generations = generations or self._config.max_generations

        # Initialize result
        result = EvolutionResult(
            evolution_id=evolution_id,
            agent_name=agent_name,
            status=EvolutionStatus.EVALUATING,
        )

        # Initialize version history for this agent
        if agent_name not in self._version_history:
            self._version_history[agent_name] = []

        # Create initial version
        original_version = PromptVersion(
            version_id=str(uuid.uuid4()),
            prompt=current_prompt,
            generation=0,
        )

        try:
            # Step 1: Evaluate current performance
            logger.info(f"Evaluating baseline for {agent_name}")
            baseline_eval = await self._evaluate(
                current_prompt,
                test_suite,
                agent_runner,
            )
            original_version.score = baseline_eval.overall_score
            result.initial_score = baseline_eval.overall_score
            result.version_history.append(original_version)

            # Store original
            self._version_history[agent_name].append(original_version)

            # Step 2: Evolution loop
            best_version = original_version
            current_version = original_version

            for generation in range(max_generations):
                result.status = EvolutionStatus.OPTIMIZING
                result.generations_run = generation + 1

                logger.info(
                    f"Evolution {agent_name} generation {generation + 1}/{max_generations}"
                )

                # Get failed cases for feedback
                failed_cases = [
                    r for r in baseline_eval.test_results if not r.passed
                ]

                # Generate candidates
                candidates = await self._optimizer.optimize(
                    current_version.prompt,
                    baseline_eval,
                    failed_cases,
                )

                if not candidates:
                    logger.warning("No candidates generated")
                    continue

                # Step 3: Test candidates
                result.status = EvolutionStatus.TESTING

                for i, candidate_prompt in enumerate(candidates):
                    # Create version
                    candidate_version = PromptVersion(
                        version_id=str(uuid.uuid4()),
                        prompt=candidate_prompt,
                        parent_version=current_version.version_id,
                        generation=generation + 1,
                    )

                    # Calculate drift
                    candidate_version.drift_from_original = self._calculate_drift(
                        original_version.prompt,
                        candidate_prompt,
                    )

                    # Evaluate candidate
                    try:
                        eval_result = await self._evaluate(
                            candidate_prompt,
                            test_suite,
                            agent_runner,
                        )
                        candidate_version.score = eval_result.overall_score

                        # Calculate confidence based on consistency
                        candidate_version.confidence = self._calculate_confidence(
                            eval_result
                        )

                        # Run safety checks
                        safety_checks = self._safety_checker.run_all_checks(
                            initial_score=best_version.score,
                            new_score=candidate_version.score,
                            drift=candidate_version.drift_from_original,
                            confidence=candidate_version.confidence,
                            pass_rate=eval_result.pass_rate,
                        )

                        # Store version
                        result.version_history.append(candidate_version)

                        # Check if this is better
                        if (
                            candidate_version.score > best_version.score
                            and self._safety_checker.all_passed(safety_checks)
                        ):
                            best_version = candidate_version
                            current_version = candidate_version
                            logger.info(
                                f"Found improved version: {candidate_version.score:.2%}"
                            )

                    except Exception as e:
                        logger.warning(f"Candidate {i} evaluation failed: {e}")
                        result.errors.append(f"Candidate evaluation failed: {e}")

                # Update baseline for next iteration
                baseline_eval = await self._evaluate(
                    current_version.prompt,
                    test_suite,
                    agent_runner,
                )

            # Step 4: Final safety check and commit
            if best_version.version_id != original_version.version_id:
                final_checks = self._safety_checker.run_all_checks(
                    initial_score=original_version.score,
                    new_score=best_version.score,
                    drift=best_version.drift_from_original,
                    confidence=best_version.confidence,
                    pass_rate=baseline_eval.pass_rate,
                )

                result.safety_checks = {
                    check: passed for check, (passed, _) in final_checks.items()
                }

                if self._safety_checker.all_passed(final_checks):
                    result.status = EvolutionStatus.IMPROVED
                    result.best_version = best_version
                    result.final_score = best_version.score
                    self._successful_evolutions += 1
                else:
                    result.status = EvolutionStatus.SAFETY_BLOCKED
                    self._safety_blocks += 1
                    logger.warning("Evolution blocked by safety checks")
            else:
                result.status = EvolutionStatus.NO_IMPROVEMENT
                result.final_score = original_version.score

        except Exception as e:
            result.status = EvolutionStatus.FAILED
            result.errors.append(str(e))
            logger.exception(f"Evolution failed: {e}")

        # Finalize result
        result.total_time_ms = (time.perf_counter() - start_time) * 1000
        self._total_evolutions += 1

        # Emit completion event
        self._emit_evolution_event(result)

        return result

    async def _evaluate(
        self,
        prompt: str,
        test_suite: list[TestCase],
        agent_runner: Callable[[str, str], Awaitable[Any]],
    ) -> EvaluationResult:
        """Evaluate prompt against test suite.

        Args:
            prompt: Prompt to evaluate
            test_suite: Test cases to run
            agent_runner: Function to run agent

        Returns:
            EvaluationResult with scores
        """
        start_time = time.perf_counter()
        test_results = []
        total_score = 0.0
        total_weight = 0.0

        for test_case in test_suite:
            try:
                test_start = time.perf_counter()
                actual_output = await agent_runner(prompt, test_case.input)
                test_time = (time.perf_counter() - test_start) * 1000

                # Score the result
                score = self._score_output(
                    actual_output,
                    test_case.expected_output,
                )
                passed = score >= 0.8  # 80% threshold for pass

                test_results.append(TestResult(
                    test_id=test_case.test_id,
                    passed=passed,
                    score=score,
                    actual_output=actual_output,
                    execution_time_ms=test_time,
                ))

                total_score += score * test_case.weight
                total_weight += test_case.weight

            except Exception as e:
                test_results.append(TestResult(
                    test_id=test_case.test_id,
                    passed=False,
                    score=0.0,
                    error=str(e),
                ))
                total_weight += test_case.weight

        overall_score = total_score / total_weight if total_weight > 0 else 0.0
        evaluation_time = (time.perf_counter() - start_time) * 1000

        return EvaluationResult(
            overall_score=overall_score,
            test_results=test_results,
            evaluation_time_ms=evaluation_time,
        )

    def _score_output(
        self,
        actual: Any,
        expected: Any,
    ) -> float:
        """Score actual output against expected.

        Simple scoring based on string/value comparison.

        Args:
            actual: Actual output
            expected: Expected output

        Returns:
            Score between 0.0 and 1.0
        """
        if actual == expected:
            return 1.0

        # String similarity
        if isinstance(actual, str) and isinstance(expected, str):
            # Simple character overlap
            actual_set = set(actual.lower())
            expected_set = set(expected.lower())
            if not expected_set:
                return 0.0
            overlap = len(actual_set & expected_set) / len(expected_set)
            return overlap

        # Dict comparison
        if isinstance(actual, dict) and isinstance(expected, dict):
            if not expected:
                return 0.0
            matching_keys = set(actual.keys()) & set(expected.keys())
            matching_values = sum(
                1 for k in matching_keys if actual.get(k) == expected.get(k)
            )
            return matching_values / len(expected)

        # List comparison
        if isinstance(actual, list) and isinstance(expected, list):
            if not expected:
                return 0.0
            matches = sum(1 for a in actual if a in expected)
            return matches / len(expected)

        return 0.0

    def _calculate_drift(
        self,
        original: str,
        candidate: str,
    ) -> float:
        """Calculate semantic drift between prompts.

        Simple calculation based on word overlap.

        Args:
            original: Original prompt
            candidate: Candidate prompt

        Returns:
            Drift value (0.0 = identical, 1.0 = completely different)
        """
        original_words = set(original.lower().split())
        candidate_words = set(candidate.lower().split())

        if not original_words:
            return 1.0

        intersection = original_words & candidate_words
        union = original_words | candidate_words

        if not union:
            return 0.0

        jaccard = len(intersection) / len(union)
        return 1.0 - jaccard

    def _calculate_confidence(
        self,
        eval_result: EvaluationResult,
    ) -> float:
        """Calculate confidence in evaluation result.

        Based on consistency of scores and pass rate.

        Args:
            eval_result: Evaluation result

        Returns:
            Confidence score (0.0-1.0)
        """
        if not eval_result.test_results:
            return 0.0

        # Higher pass rate = higher confidence
        pass_confidence = eval_result.pass_rate

        # Lower variance in scores = higher confidence
        scores = [r.score for r in eval_result.test_results]
        if len(scores) > 1:
            mean_score = sum(scores) / len(scores)
            variance = sum((s - mean_score) ** 2 for s in scores) / len(scores)
            consistency = 1.0 - min(variance, 1.0)
        else:
            consistency = 1.0

        return (pass_confidence + consistency) / 2

    def _emit_evolution_event(self, result: EvolutionResult) -> None:
        """Emit evolution completion event."""
        event = Event(
            event_id=_generate_event_id(),
            event_type=EventType.EVOLUTION_COMPLETED,
            timestamp=_get_utc_now(),
            session_id=result.agent_name,
            payload=result.to_dict(),
        )
        self._event_store.append(event)

    def get_version_history(self, agent_name: str) -> list[PromptVersion]:
        """Get version history for an agent.

        Args:
            agent_name: Name of the agent

        Returns:
            List of prompt versions
        """
        return self._version_history.get(agent_name, [])

    def get_best_version(self, agent_name: str) -> Optional[PromptVersion]:
        """Get the best performing version for an agent.

        Args:
            agent_name: Name of the agent

        Returns:
            Best prompt version or None
        """
        history = self._version_history.get(agent_name, [])
        if not history:
            return None

        return max(history, key=lambda v: v.score)

    def rollback(
        self,
        agent_name: str,
        version_id: str,
    ) -> Optional[PromptVersion]:
        """Rollback to a specific version.

        Args:
            agent_name: Name of the agent
            version_id: Version to rollback to

        Returns:
            The version rolled back to, or None if not found
        """
        history = self._version_history.get(agent_name, [])

        for version in history:
            if version.version_id == version_id:
                logger.info(f"Rolling back {agent_name} to version {version_id}")
                return version

        return None

    def get_metrics(self) -> dict[str, Any]:
        """Get evolution manager metrics.

        Returns:
            Dictionary with metrics
        """
        return {
            "total_evolutions": self._total_evolutions,
            "successful_evolutions": self._successful_evolutions,
            "safety_blocks": self._safety_blocks,
            "success_rate": (
                self._successful_evolutions / max(self._total_evolutions, 1)
            ),
            "agents_tracked": len(self._version_history),
            "total_versions": sum(
                len(h) for h in self._version_history.values()
            ),
            "config": self._config.to_dict(),
        }


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Enums
    "EvolutionStatus",
    "OptimizationMethod",
    # Config
    "EvolutionConfig",
    # Data classes
    "TestCase",
    "TestResult",
    "EvaluationResult",
    "PromptVersion",
    "EvolutionResult",
    # Checker
    "SafetyChecker",
    # Optimizers
    "PromptOptimizer",
    "TextGradOptimizer",
    "EvolutionaryOptimizer",
    # Main class
    "EvolutionManager",
    # Constants
    "DEFAULT_MAX_GENERATIONS",
    "DEFAULT_MIN_IMPROVEMENT_THRESHOLD",
    "DEFAULT_MAX_PROMPT_DRIFT",
]
