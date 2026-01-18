"""Pre-send context validation for Sigil.

This module provides validation and reduction strategies for context
before sending to the LLM, preventing context overflow failures.

Validation Flow:
    1. Estimate token count for assembled context
    2. Compare against effective maximum (max_tokens - buffer)
    3. If over budget, apply reduction strategies in priority order
    4. Return validated context with reduction details

Reduction Strategies (applied in order):
    1. REDUCE_MEMORY: Reduce memory items (5 -> 3 -> 1)
    2. REDUCE_PLAN_STEPS: Reduce plan steps (5 -> 3 -> 1)
    3. TRUNCATE_TOOL_RESULTS: Truncate tool results (4000 -> 2000 -> 1000 -> 500)

Example:
    >>> from sigil.context.validator import ContextValidator
    >>>
    >>> validator = ContextValidator(max_tokens=150000, buffer_tokens=5000)
    >>> result = validator.validate(context)
    >>> if not result.is_valid:
    ...     reduced_context, result = validator.validate_and_reduce(context)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional, TYPE_CHECKING
import logging

if TYPE_CHECKING:
    from sigil.config.settings import SigilSettings

logger = logging.getLogger(__name__)


# =============================================================================
# Constants
# =============================================================================

DEFAULT_MAX_TOKENS = 150000
"""Default maximum context tokens for Claude models."""

DEFAULT_BUFFER_TOKENS = 5000
"""Default safety buffer to reserve for response generation."""

DEFAULT_CHARS_PER_TOKEN = 4
"""Default character-to-token ratio estimate."""


# =============================================================================
# Reduction Strategy Enum
# =============================================================================


class ReductionStrategy(str, Enum):
    """Available context reduction strategies.

    Attributes:
        REDUCE_MEMORY: Reduce the number of memory items in context.
        REDUCE_PLAN_STEPS: Reduce the number of plan steps in context.
        TRUNCATE_TOOL_RESULTS: Truncate tool result outputs to a character limit.
        TRUNCATE_CONTEXT_VALUES: Truncate arbitrary context values.
        SUMMARIZE: Use LLM to summarize context (requires external call).
    """

    REDUCE_MEMORY = "reduce_memory"
    REDUCE_PLAN_STEPS = "reduce_plan_steps"
    TRUNCATE_TOOL_RESULTS = "truncate_tool_results"
    TRUNCATE_CONTEXT_VALUES = "truncate_context_values"
    SUMMARIZE = "summarize"


# =============================================================================
# Result Data Classes
# =============================================================================


@dataclass
class ValidationResult:
    """Result of context validation.

    Attributes:
        is_valid: Whether the context fits within the token budget.
        estimated_tokens: Estimated token count for the context.
        max_tokens: Maximum allowed tokens (after buffer).
        overflow_tokens: Number of tokens over budget (0 if valid).
        overflow_ratio: Ratio of estimated to max tokens.
        reductions_applied: List of reduction descriptions applied.
        warnings: Non-fatal warnings about context state.
    """

    is_valid: bool
    estimated_tokens: int
    max_tokens: int
    overflow_tokens: int = 0
    overflow_ratio: float = 1.0
    reductions_applied: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass
class ReductionResult:
    """Result of applying a reduction strategy.

    Attributes:
        success: Whether the reduction was applied successfully.
        tokens_saved: Estimated tokens saved by the reduction.
        strategy: The reduction strategy that was applied.
        details: Human-readable description of the reduction.
    """

    success: bool
    tokens_saved: int
    strategy: ReductionStrategy
    details: str = ""


# =============================================================================
# Context Validator
# =============================================================================


class ContextValidator:
    """Validates and reduces context to fit within token budget.

    This validator estimates token count and applies reduction strategies
    in priority order until the context fits within the budget.

    The validation process:
    1. Estimate total tokens in the context dictionary
    2. Compare against effective maximum (max_tokens - buffer_tokens)
    3. If over budget, apply reductions in priority order
    4. Return modified context and validation result

    Attributes:
        max_tokens: Maximum allowed tokens.
        buffer_tokens: Safety buffer to reserve for response.
        chars_per_token: Characters per token estimate.

    Example:
        >>> validator = ContextValidator(max_tokens=1000, buffer_tokens=100)
        >>> context = {"message": "Hello " * 500}
        >>> result = validator.validate(context)
        >>> print(f"Valid: {result.is_valid}, Tokens: {result.estimated_tokens}")
    """

    def __init__(
        self,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        buffer_tokens: int = DEFAULT_BUFFER_TOKENS,
        chars_per_token: int = DEFAULT_CHARS_PER_TOKEN,
    ) -> None:
        """Initialize the context validator.

        Args:
            max_tokens: Maximum allowed tokens (default: 150000).
            buffer_tokens: Safety buffer to reserve (default: 5000).
            chars_per_token: Character-to-token ratio (default: 4).
        """
        self.max_tokens = max_tokens
        self.buffer_tokens = buffer_tokens
        self.chars_per_token = chars_per_token
        self._effective_max = max_tokens - buffer_tokens

    def estimate_tokens(self, text: str) -> int:
        """Estimate token count for text.

        Uses character-based estimation with a small overhead buffer
        to account for tokenization overhead.

        Args:
            text: Text to estimate tokens for.

        Returns:
            Estimated token count.
        """
        if not text:
            return 0
        # Add small buffer for tokenization overhead
        return len(text) // self.chars_per_token + 50

    def estimate_context_tokens(self, context: dict[str, Any]) -> int:
        """Estimate total tokens for a context dictionary.

        Recursively estimates tokens for all string values in the context,
        including nested dictionaries and lists.

        Args:
            context: Context dictionary to estimate.

        Returns:
            Total estimated tokens.
        """
        total = 0

        def _estimate_value(value: Any) -> int:
            """Recursively estimate tokens for a value."""
            if isinstance(value, str):
                return self.estimate_tokens(value)
            elif isinstance(value, dict):
                return sum(_estimate_value(v) for v in value.values())
            elif isinstance(value, (list, tuple)):
                return sum(_estimate_value(v) for v in value)
            else:
                # Convert to string for estimation
                return self.estimate_tokens(str(value))

        for key, value in context.items():
            # Count key tokens
            total += self.estimate_tokens(key)
            # Count value tokens
            total += _estimate_value(value)

        return total

    def validate(self, context: dict[str, Any]) -> ValidationResult:
        """Validate context against token budget.

        Checks if the context fits within the effective maximum tokens
        (max_tokens - buffer_tokens) and generates warnings if approaching
        the limit.

        Args:
            context: Context dictionary to validate.

        Returns:
            ValidationResult with details about the validation.
        """
        estimated = self.estimate_context_tokens(context)
        overflow = max(0, estimated - self._effective_max)
        ratio = estimated / self._effective_max if self._effective_max > 0 else float("inf")

        warnings = []
        if ratio > 0.8:
            warnings.append(f"Context at {ratio:.1%} of budget")
        if ratio > 0.95:
            warnings.append("Context near maximum - consider reduction")

        return ValidationResult(
            is_valid=estimated <= self._effective_max,
            estimated_tokens=estimated,
            max_tokens=self._effective_max,
            overflow_tokens=overflow,
            overflow_ratio=ratio,
            warnings=warnings,
        )

    def reduce_memory_items(
        self,
        context: dict[str, Any],
        current_count: int,
        target_count: int,
    ) -> ReductionResult:
        """Reduce memory items in context.

        Keeps only the first target_count memory items, removing older
        or lower-priority memories.

        Args:
            context: Context to modify (modified in place).
            current_count: Current number of memory items.
            target_count: Target number of memory items to keep.

        Returns:
            ReductionResult with details about the reduction.
        """
        if "relevant_memories" not in context:
            return ReductionResult(
                success=False,
                tokens_saved=0,
                strategy=ReductionStrategy.REDUCE_MEMORY,
                details="No memories in context",
            )

        memories = context.get("relevant_memories", [])
        if len(memories) <= target_count:
            return ReductionResult(
                success=False,
                tokens_saved=0,
                strategy=ReductionStrategy.REDUCE_MEMORY,
                details=f"Already at {len(memories)} items",
            )

        # Estimate tokens before reduction
        before_tokens = self.estimate_tokens(str(memories))

        # Reduce to target count
        context["relevant_memories"] = memories[:target_count]

        # Estimate tokens after reduction
        after_tokens = self.estimate_tokens(str(context["relevant_memories"]))
        saved = before_tokens - after_tokens

        logger.debug(
            f"Reduced memories from {len(memories)} to {target_count}, "
            f"saved ~{saved} tokens"
        )

        return ReductionResult(
            success=True,
            tokens_saved=saved,
            strategy=ReductionStrategy.REDUCE_MEMORY,
            details=f"Reduced from {len(memories)} to {target_count} items",
        )

    def reduce_plan_steps(
        self,
        context: dict[str, Any],
        current_count: int,
        target_count: int,
    ) -> ReductionResult:
        """Reduce plan steps in context.

        Keeps only the first target_count plan steps. This preserves
        the initial steps which are typically most important.

        Args:
            context: Context to modify (modified in place).
            current_count: Current number of plan steps.
            target_count: Target number of steps to keep.

        Returns:
            ReductionResult with details about the reduction.
        """
        plan = context.get("plan", {})
        steps = plan.get("steps", [])

        if len(steps) <= target_count:
            return ReductionResult(
                success=False,
                tokens_saved=0,
                strategy=ReductionStrategy.REDUCE_PLAN_STEPS,
                details=f"Already at {len(steps)} steps",
            )

        # Estimate tokens before reduction
        before_tokens = self.estimate_tokens(str(steps))

        # Reduce to target count
        plan["steps"] = steps[:target_count]

        # Estimate tokens after reduction
        after_tokens = self.estimate_tokens(str(plan["steps"]))
        saved = before_tokens - after_tokens

        logger.debug(
            f"Reduced plan steps from {len(steps)} to {target_count}, "
            f"saved ~{saved} tokens"
        )

        return ReductionResult(
            success=True,
            tokens_saved=saved,
            strategy=ReductionStrategy.REDUCE_PLAN_STEPS,
            details=f"Reduced from {len(steps)} to {target_count} steps",
        )

    def truncate_tool_results(
        self,
        context: dict[str, Any],
        max_chars: int,
    ) -> ReductionResult:
        """Truncate tool results to specified character limit.

        Truncates each tool output in prior_outputs to max_chars,
        appending a "[truncated]" marker if truncation occurred.

        Args:
            context: Context to modify (modified in place).
            max_chars: Maximum characters per tool result.

        Returns:
            ReductionResult with details about the reduction.
        """
        prior_outputs = context.get("prior_outputs", {})
        if not prior_outputs:
            return ReductionResult(
                success=False,
                tokens_saved=0,
                strategy=ReductionStrategy.TRUNCATE_TOOL_RESULTS,
                details="No tool outputs in context",
            )

        total_saved = 0
        truncated_count = 0

        for step_id, output_data in prior_outputs.items():
            output = output_data.get("output", "")
            if len(output) > max_chars:
                before_len = len(output)
                output_data["output"] = output[:max_chars] + "\n[truncated]"
                saved = (before_len - max_chars) // self.chars_per_token
                total_saved += saved
                truncated_count += 1

        if truncated_count == 0:
            return ReductionResult(
                success=False,
                tokens_saved=0,
                strategy=ReductionStrategy.TRUNCATE_TOOL_RESULTS,
                details="No results exceeded limit",
            )

        logger.debug(
            f"Truncated {truncated_count} tool results to {max_chars} chars, "
            f"saved ~{total_saved} tokens"
        )

        return ReductionResult(
            success=True,
            tokens_saved=total_saved,
            strategy=ReductionStrategy.TRUNCATE_TOOL_RESULTS,
            details=f"Truncated {truncated_count} results to {max_chars} chars",
        )

    def validate_and_reduce(
        self,
        context: dict[str, Any],
        settings: Optional[Any] = None,
    ) -> tuple[dict[str, Any], ValidationResult]:
        """Validate context and apply reductions if needed.

        Applies reduction strategies in priority order until context
        fits within budget or all strategies are exhausted.

        Priority order:
        1. Reduce memory items (5 -> 3 -> 1)
        2. Reduce plan steps (5 -> 3 -> 1)
        3. Truncate tool results (4000 -> 2000 -> 1000 -> 500)

        Args:
            context: Context dictionary to validate and reduce.
            settings: Optional SigilSettings for additional configuration.

        Returns:
            Tuple of (modified context, final validation result).
        """
        # Initial validation
        result = self.validate(context)

        if result.is_valid:
            return context, result

        logger.info(
            f"Context exceeds budget by {result.overflow_tokens} tokens, "
            f"applying reductions"
        )

        reductions_applied = []

        # Strategy 1: Reduce memory items progressively
        for target in [3, 1]:
            if result.is_valid:
                break
            reduction = self.reduce_memory_items(
                context,
                len(context.get("relevant_memories", [])),
                target,
            )
            if reduction.success:
                reductions_applied.append(reduction.details)
                result = self.validate(context)

        # Strategy 2: Reduce plan steps progressively
        for target in [3, 1]:
            if result.is_valid:
                break
            plan = context.get("plan", {})
            reduction = self.reduce_plan_steps(
                context,
                len(plan.get("steps", [])),
                target,
            )
            if reduction.success:
                reductions_applied.append(reduction.details)
                result = self.validate(context)

        # Strategy 3: Truncate tool results progressively
        for max_chars in [4000, 2000, 1000, 500]:
            if result.is_valid:
                break
            reduction = self.truncate_tool_results(context, max_chars)
            if reduction.success:
                reductions_applied.append(reduction.details)
                result = self.validate(context)

        result.reductions_applied = reductions_applied

        if not result.is_valid:
            logger.warning(
                f"Could not reduce context below budget. "
                f"Still over by {result.overflow_tokens} tokens"
            )
        else:
            logger.info(
                f"Context reduced to {result.estimated_tokens} tokens "
                f"after {len(reductions_applied)} reductions"
            )

        return context, result


# =============================================================================
# Factory Functions
# =============================================================================


def create_validator_from_settings(settings: "SigilSettings") -> ContextValidator:
    """Create a ContextValidator from SigilSettings.

    Looks for context-related settings in the settings object and
    uses them to configure the validator. Falls back to defaults
    if context settings are not available.

    Args:
        settings: SigilSettings instance.

    Returns:
        Configured ContextValidator instance.
    """
    context_settings = getattr(settings, "context", None)

    if context_settings:
        return ContextValidator(
            max_tokens=getattr(
                context_settings, "max_context_tokens", DEFAULT_MAX_TOKENS
            ),
            buffer_tokens=getattr(
                context_settings, "validation_buffer_tokens", DEFAULT_BUFFER_TOKENS
            ),
            chars_per_token=getattr(
                context_settings, "chars_per_token_estimate", DEFAULT_CHARS_PER_TOKEN
            ),
        )

    # Fallback to defaults
    return ContextValidator()


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Enums
    "ReductionStrategy",
    # Data classes
    "ValidationResult",
    "ReductionResult",
    # Main class
    "ContextValidator",
    # Factory functions
    "create_validator_from_settings",
    # Constants
    "DEFAULT_MAX_TOKENS",
    "DEFAULT_BUFFER_TOKENS",
    "DEFAULT_CHARS_PER_TOKEN",
]
