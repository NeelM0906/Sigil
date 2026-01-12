"""Evolution module for Sigil v2 framework.

This module implements self-improvement through optimization:
- Prompt optimization via TextGrad
- Evolutionary prompt mutation
- Safety-constrained evolution
- Version history tracking

Key Components:
    - EvolutionManager: Orchestrates self-improvement
    - SafetyChecker: Enforces safety constraints
    - TextGradOptimizer: Gradient-based optimization
    - EvolutionaryOptimizer: Evolutionary optimization

Example:
    >>> from sigil.evolution import EvolutionManager, TestCase
    >>>
    >>> manager = EvolutionManager()
    >>> result = await manager.evolve(
    ...     agent_name="my_agent",
    ...     current_prompt="...",
    ...     test_suite=[TestCase(...)],
    ...     agent_runner=runner_fn,
    ... )
"""

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
    PromptOptimizer,
    TextGradOptimizer,
    EvolutionaryOptimizer,
    DEFAULT_MAX_GENERATIONS,
    DEFAULT_MIN_IMPROVEMENT_THRESHOLD,
    DEFAULT_MAX_PROMPT_DRIFT,
)

__all__ = [
    # Main class
    "EvolutionManager",
    # Config
    "EvolutionConfig",
    # Enums
    "EvolutionStatus",
    "OptimizationMethod",
    # Data classes
    "TestCase",
    "TestResult",
    "EvaluationResult",
    "PromptVersion",
    "EvolutionResult",
    # Utilities
    "SafetyChecker",
    "PromptOptimizer",
    "TextGradOptimizer",
    "EvolutionaryOptimizer",
    # Constants
    "DEFAULT_MAX_GENERATIONS",
    "DEFAULT_MIN_IMPROVEMENT_THRESHOLD",
    "DEFAULT_MAX_PROMPT_DRIFT",
]
