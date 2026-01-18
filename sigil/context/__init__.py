"""Context management for Sigil v2 framework.

This module provides context window management and compression
for agent invocations.

Key Components:
    - ContextManager: Assembles optimized context windows
    - ContextCompressor: Compresses context when over budget
    - ContextSource: Enumeration of context sources
    - ContextItem: Individual context items

Example:
    >>> from sigil.context import ContextManager, ContextCompressor
    >>>
    >>> manager = ContextManager(max_tokens=128000)
    >>> result = await manager.assemble(
    ...     task="Qualify lead",
    ...     session_id="sess-123",
    ... )
"""

from sigil.context.manager import (
    ContextManager,
    ContextSource,
    ContextItem,
    ContextAssemblyResult,
    ContextBudget,
    ContextProvider,
    MemoryContextProvider,
    ConversationContextProvider,
    DEFAULT_MAX_CONTEXT_TOKENS,
    DEFAULT_RESERVED_FOR_RESPONSE,
)

from sigil.context.compression import (
    ContextCompressor,
    CompressionStrategy,
    CompressionResult,
    TruncateStrategy,
    SummarizeStrategy,
    PrioritizeStrategy,
)

from sigil.context.validator import (
    ContextValidator,
    ValidationResult,
    ReductionResult,
    ReductionStrategy,
    create_validator_from_settings,
    DEFAULT_MAX_TOKENS,
    DEFAULT_BUFFER_TOKENS,
    DEFAULT_CHARS_PER_TOKEN,
)

__all__ = [
    # Manager
    "ContextManager",
    "ContextSource",
    "ContextItem",
    "ContextAssemblyResult",
    "ContextBudget",
    "ContextProvider",
    "MemoryContextProvider",
    "ConversationContextProvider",
    # Compression
    "ContextCompressor",
    "CompressionStrategy",
    "CompressionResult",
    "TruncateStrategy",
    "SummarizeStrategy",
    "PrioritizeStrategy",
    # Constants
    "DEFAULT_MAX_CONTEXT_TOKENS",
    "DEFAULT_RESERVED_FOR_RESPONSE",
    # Validator
    "ContextValidator",
    "ValidationResult",
    "ReductionResult",
    "ReductionStrategy",
    "create_validator_from_settings",
    "DEFAULT_MAX_TOKENS",
    "DEFAULT_BUFFER_TOKENS",
    "DEFAULT_CHARS_PER_TOKEN",
]
