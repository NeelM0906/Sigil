"""Context validation middleware for Sigil pipeline.

This middleware validates context size before the execute step and applies
reduction strategies if the context exceeds the token budget.

Integration Point:
    The middleware hooks into the pre_step phase for the "execute" step,
    validating and potentially reducing the assembled_context before it
    is sent to the LLM.

Usage:
    >>> from sigil.core.middlewares import ContextValidationMiddleware
    >>> from sigil.core.middleware import MiddlewareChain
    >>>
    >>> chain = MiddlewareChain()
    >>> chain.add(ContextValidationMiddleware(max_tokens=150000))

Example with settings:
    >>> middleware = ContextValidationMiddleware.from_settings(settings)
    >>> chain.add(middleware)
"""

from __future__ import annotations

from typing import Any, Optional, TYPE_CHECKING
import logging

from sigil.core.middleware import BaseMiddleware
from sigil.context.validator import (
    ContextValidator,
    ValidationResult,
    DEFAULT_MAX_TOKENS,
    DEFAULT_BUFFER_TOKENS,
    DEFAULT_CHARS_PER_TOKEN,
)

if TYPE_CHECKING:
    from sigil.config.settings import SigilSettings

logger = logging.getLogger(__name__)


class ContextValidationMiddleware(BaseMiddleware):
    """Middleware that validates context before execution.

    This middleware checks context size before the execute step
    and applies reduction strategies if the context exceeds the budget.

    The middleware operates at the pre_step hook for the "execute" step,
    which is where the assembled context is about to be sent to the LLM.
    If the context exceeds the token budget, reduction strategies are
    applied in priority order.

    Attributes:
        validator: The ContextValidator instance used for validation.
        enabled: Whether validation is enabled.

    Example:
        >>> middleware = ContextValidationMiddleware(
        ...     max_tokens=150000,
        ...     buffer_tokens=5000,
        ...     enabled=True,
        ... )
        >>> chain.add(middleware)
    """

    def __init__(
        self,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        buffer_tokens: int = DEFAULT_BUFFER_TOKENS,
        chars_per_token: int = DEFAULT_CHARS_PER_TOKEN,
        enabled: bool = True,
    ) -> None:
        """Initialize the context validation middleware.

        Args:
            max_tokens: Maximum allowed tokens (default: 150000).
            buffer_tokens: Safety buffer to reserve (default: 5000).
            chars_per_token: Character-to-token ratio (default: 4).
            enabled: Whether validation is enabled (default: True).
        """
        self.validator = ContextValidator(
            max_tokens=max_tokens,
            buffer_tokens=buffer_tokens,
            chars_per_token=chars_per_token,
        )
        self.enabled = enabled
        self._last_validation: Optional[ValidationResult] = None

    @property
    def name(self) -> str:
        """Get the middleware name."""
        return "ContextValidationMiddleware"

    @classmethod
    def from_settings(cls, settings: "SigilSettings") -> "ContextValidationMiddleware":
        """Create middleware from SigilSettings.

        Extracts context-related settings from the settings object
        to configure the middleware. Falls back to defaults if
        context settings are not available.

        Args:
            settings: SigilSettings instance.

        Returns:
            Configured ContextValidationMiddleware instance.
        """
        context_settings = getattr(settings, "context", None)

        if context_settings:
            return cls(
                max_tokens=getattr(
                    context_settings, "max_context_tokens", DEFAULT_MAX_TOKENS
                ),
                buffer_tokens=getattr(
                    context_settings, "validation_buffer_tokens", DEFAULT_BUFFER_TOKENS
                ),
                chars_per_token=getattr(
                    context_settings, "chars_per_token_estimate", DEFAULT_CHARS_PER_TOKEN
                ),
                enabled=getattr(
                    context_settings, "enable_pre_send_validation", True
                ),
            )

        return cls()

    async def pre_step(self, step_name: str, ctx: Any) -> Any:
        """Validate context before execute step.

        This hook is called before each pipeline step. It only performs
        validation for the "execute" step, which is when the context
        is about to be sent to the LLM.

        If the context exceeds the token budget, reduction strategies
        are applied and the context is modified in place.

        Args:
            step_name: Name of the step about to execute.
            ctx: Pipeline context object.

        Returns:
            The context, potentially with reduced assembled_context.
        """
        if not self.enabled:
            return ctx

        # Only validate before execute step
        if step_name != "execute":
            return ctx

        # Get assembled context from pipeline context
        assembled_context = getattr(ctx, "assembled_context", None)
        if not assembled_context:
            return ctx

        # Validate and reduce if needed
        reduced_context, result = self.validator.validate_and_reduce(assembled_context)
        self._last_validation = result

        # Update context with reduced version
        ctx.assembled_context = reduced_context

        # Log validation result
        if result.reductions_applied:
            logger.info(
                f"Context validation: {result.estimated_tokens}/{result.max_tokens} "
                f"tokens after {len(result.reductions_applied)} reductions"
            )
        elif result.warnings:
            for warning in result.warnings:
                logger.warning(f"Context validation: {warning}")

        return ctx

    async def post_step(self, step_name: str, ctx: Any, result: Any) -> Any:
        """Log validation stats after execute.

        This hook is called after each pipeline step. It logs validation
        statistics for the execute step to aid in monitoring and debugging.

        Args:
            step_name: Name of the step that executed.
            ctx: Pipeline context object.
            result: Result from the step.

        Returns:
            The result unchanged.
        """
        if step_name == "execute" and self._last_validation:
            v = self._last_validation
            logger.debug(
                f"Execute completed. Context was {v.estimated_tokens} tokens "
                f"({v.overflow_ratio:.1%} of budget)"
            )
        return result

    def get_last_validation(self) -> Optional[ValidationResult]:
        """Get the last validation result.

        Returns the most recent validation result, which can be used
        for monitoring, debugging, or metrics collection.

        Returns:
            The last ValidationResult, or None if no validation has occurred.
        """
        return self._last_validation

    def reset(self) -> None:
        """Reset the middleware state.

        Clears the last validation result. Useful for testing or when
        reusing the middleware across multiple requests.
        """
        self._last_validation = None


__all__ = ["ContextValidationMiddleware"]
