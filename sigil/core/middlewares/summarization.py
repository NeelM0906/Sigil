"""Conversation summarization middleware for Sigil.

This middleware automatically summarizes old conversation history
when context approaches token limits, preserving recent messages.

The middleware monitors conversation context and automatically
summarizes older messages when approaching token limits. This helps
maintain context relevance while preventing token budget overruns.

Integration Point:
    The middleware hooks into the pre_step phase for the "execute" step,
    checking conversation history size and summarizing if necessary before
    the context is sent to the LLM.

Usage:
    >>> from sigil.core.middlewares import SummarizationMiddleware
    >>> from sigil.core.middleware import MiddlewareChain
    >>>
    >>> chain = MiddlewareChain()
    >>> chain.add(SummarizationMiddleware(threshold_tokens=120000))

Example with settings:
    >>> middleware = SummarizationMiddleware.from_settings(settings)
    >>> chain.add(middleware)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional, Protocol, TYPE_CHECKING
import logging

from sigil.core.middleware import BaseMiddleware

if TYPE_CHECKING:
    from sigil.config.settings import SigilSettings

logger = logging.getLogger(__name__)


@dataclass
class Message:
    """Represents a conversation message.

    Attributes:
        role: The role of the message sender ("user" or "assistant").
        content: The content of the message.
        timestamp: Optional ISO format timestamp of when the message was created.
        metadata: Additional metadata associated with the message.
    """

    role: str  # "user" or "assistant"
    content: str
    timestamp: Optional[str] = None
    metadata: dict = field(default_factory=dict)


@dataclass
class SummarizationResult:
    """Result of a summarization operation.

    Captures metrics about the summarization operation including
    message counts, token usage, and the generated summary.

    Attributes:
        original_message_count: Number of messages before summarization.
        preserved_message_count: Number of recent messages preserved verbatim.
        summarized_message_count: Number of messages that were summarized.
        summary: The generated summary text.
        tokens_before: Estimated token count before summarization.
        tokens_after: Estimated token count after summarization.
        tokens_saved: Number of tokens saved by summarization.
    """

    original_message_count: int
    preserved_message_count: int
    summarized_message_count: int
    summary: str
    tokens_before: int
    tokens_after: int
    tokens_saved: int


class LLMSummarizer(Protocol):
    """Protocol for LLM-based summarization.

    Defines the interface for LLM clients that can perform
    summarization. Implementations must provide a generate method.
    """

    async def generate(
        self,
        prompt: str,
        max_tokens: int = 500,
        temperature: float = 0.3,
    ) -> str:
        """Generate a response from the LLM.

        Args:
            prompt: The prompt to send to the LLM.
            max_tokens: Maximum tokens in the response.
            temperature: Sampling temperature.

        Returns:
            The generated text response.
        """
        ...


class ConversationSummarizer:
    """Summarizes conversation history to fit within token budgets.

    This class handles the logic of deciding when to summarize and
    how to partition messages between preserved and summarized.

    The summarizer uses a simple heuristic: preserve the most recent
    N messages verbatim and summarize everything before them. This
    ensures recent context is maintained while older context is
    compressed.

    Attributes:
        threshold_tokens: Token threshold for triggering summarization.
        preserve_recent: Number of recent messages to preserve verbatim.
        chars_per_token: Estimated characters per token.

    Example:
        >>> summarizer = ConversationSummarizer(
        ...     threshold_tokens=120000,
        ...     preserve_recent=6,
        ... )
        >>> if summarizer.should_summarize(messages):
        ...     to_summarize, to_preserve = summarizer.partition_messages(messages)
        ...     summary = summarizer.create_basic_summary(to_summarize)
    """

    def __init__(
        self,
        threshold_tokens: int = 120000,
        preserve_recent: int = 6,
        chars_per_token: int = 4,
    ):
        """Initialize the conversation summarizer.

        Args:
            threshold_tokens: Token threshold for triggering summarization.
            preserve_recent: Number of recent messages to preserve.
            chars_per_token: Estimated characters per token for size calculations.
        """
        self.threshold_tokens = threshold_tokens
        self.preserve_recent = preserve_recent
        self.chars_per_token = chars_per_token

    def estimate_tokens(self, text: str) -> int:
        """Estimate token count for text.

        Uses a simple character-to-token ratio estimation. This is
        a rough estimate and actual token counts may vary.

        Args:
            text: The text to estimate tokens for.

        Returns:
            Estimated token count.
        """
        if not text:
            return 0
        return len(text) // self.chars_per_token + 10

    def estimate_messages_tokens(self, messages: list[Message]) -> int:
        """Estimate total tokens for a list of messages.

        Accounts for message content, role labels, and structural
        overhead per message.

        Args:
            messages: List of conversation messages.

        Returns:
            Estimated total token count.
        """
        total = 0
        for msg in messages:
            total += self.estimate_tokens(msg.role)
            total += self.estimate_tokens(msg.content)
            total += 10  # Overhead for message structure
        return total

    def should_summarize(self, messages: list[Message]) -> bool:
        """Check if messages should be summarized.

        Summarization is triggered when:
        1. There are more messages than the preserve_recent count
        2. The estimated token count exceeds the threshold

        Args:
            messages: List of conversation messages.

        Returns:
            True if summarization should be triggered.
        """
        if len(messages) <= self.preserve_recent:
            return False

        tokens = self.estimate_messages_tokens(messages)
        return tokens > self.threshold_tokens

    def partition_messages(
        self,
        messages: list[Message],
    ) -> tuple[list[Message], list[Message]]:
        """Partition messages into preserved and to-summarize groups.

        The most recent N messages (where N = preserve_recent) are
        preserved verbatim. All earlier messages are candidates for
        summarization.

        Args:
            messages: All conversation messages.

        Returns:
            Tuple of (messages_to_summarize, messages_to_preserve).
        """
        if len(messages) <= self.preserve_recent:
            return [], messages

        # Preserve the most recent N messages
        to_summarize = messages[: -self.preserve_recent]
        to_preserve = messages[-self.preserve_recent :]

        return to_summarize, to_preserve

    def create_basic_summary(self, messages: list[Message]) -> str:
        """Create a basic summary without LLM (fallback).

        This creates a structured summary of the conversation
        when LLM summarization is not available. It extracts
        key information like message counts and topic bookends.

        Args:
            messages: Messages to summarize.

        Returns:
            Basic text summary.
        """
        if not messages:
            return "No earlier conversation history."

        # Count by role
        user_count = sum(1 for m in messages if m.role == "user")
        assistant_count = sum(1 for m in messages if m.role == "assistant")

        # Extract key topics from user messages (simple heuristic)
        user_messages = [m.content for m in messages if m.role == "user"]

        # Get first and last user messages as bookends
        first_topic = user_messages[0][:200] if user_messages else "N/A"
        last_topic = (
            user_messages[-1][:200] if len(user_messages) > 1 else first_topic
        )

        summary = f"""Summary of earlier conversation ({len(messages)} messages):
- {user_count} user messages, {assistant_count} assistant responses
- Started with: "{first_topic}..."
- Progressed to: "{last_topic}..."
"""
        return summary

    async def summarize_with_llm(
        self,
        messages: list[Message],
        llm_client: Any,
    ) -> str:
        """Summarize messages using LLM.

        Uses an LLM to generate a more intelligent summary that
        captures the key topics, decisions, and context from the
        conversation.

        Args:
            messages: Messages to summarize.
            llm_client: LLM client for generating summary.

        Returns:
            LLM-generated summary.
        """
        if not messages:
            return "No earlier conversation history."

        # Build conversation text
        conversation_text = ""
        for msg in messages:
            role_label = "User" if msg.role == "user" else "Assistant"
            conversation_text += f"{role_label}: {msg.content}\n\n"

        # Truncate if too long (keep within reasonable LLM context)
        max_chars = 50000  # ~12.5K tokens
        if len(conversation_text) > max_chars:
            conversation_text = (
                conversation_text[:max_chars] + "\n[Earlier messages truncated...]"
            )

        prompt = f"""Summarize the following conversation in 2-3 concise paragraphs.
Focus on:
1. Main topics discussed
2. Key decisions or conclusions reached
3. Any action items or follow-ups mentioned

Conversation:
{conversation_text}

Summary:"""

        try:
            # Use LLM to generate summary
            response = await llm_client.generate(
                prompt=prompt,
                max_tokens=500,
                temperature=0.3,
            )
            return response.strip()
        except Exception as e:
            logger.warning(f"LLM summarization failed, using basic summary: {e}")
            return self.create_basic_summary(messages)


class SummarizationMiddleware(BaseMiddleware):
    """Middleware that auto-summarizes long conversation history.

    This middleware monitors conversation context and automatically
    summarizes older messages when approaching token limits. It
    preserves recent messages verbatim to maintain immediate context.

    The middleware operates at the pre_step hook for the "execute" step,
    which is where the assembled context is about to be sent to the LLM.
    If the conversation history exceeds the token threshold, summarization
    is applied.

    Attributes:
        summarizer: ConversationSummarizer instance for handling summarization logic.
        enabled: Whether summarization is enabled.
        llm_client: Optional LLM client for more intelligent summaries.
        use_llm_summarization: Whether to use LLM for summarization.

    Example:
        >>> middleware = SummarizationMiddleware(
        ...     threshold_tokens=120000,
        ...     preserve_recent=6,
        ...     enabled=True,
        ... )
        >>> chain.add(middleware)

    Example with LLM:
        >>> middleware = SummarizationMiddleware(
        ...     threshold_tokens=120000,
        ...     llm_client=my_llm_client,
        ...     use_llm_summarization=True,
        ... )
    """

    def __init__(
        self,
        threshold_tokens: int = 120000,
        preserve_recent: int = 6,
        chars_per_token: int = 4,
        enabled: bool = True,
        llm_client: Any = None,
        use_llm_summarization: bool = False,
    ):
        """Initialize the summarization middleware.

        Args:
            threshold_tokens: Token threshold for triggering summarization.
            preserve_recent: Number of recent messages to preserve.
            chars_per_token: Estimated characters per token.
            enabled: Whether summarization is enabled.
            llm_client: Optional LLM client for better summaries.
            use_llm_summarization: Whether to use LLM for summarization.
        """
        self.summarizer = ConversationSummarizer(
            threshold_tokens=threshold_tokens,
            preserve_recent=preserve_recent,
            chars_per_token=chars_per_token,
        )
        self.enabled = enabled
        self.llm_client = llm_client
        self.use_llm_summarization = use_llm_summarization and llm_client is not None
        self._last_result: Optional[SummarizationResult] = None

    @property
    def name(self) -> str:
        """Get the middleware name."""
        return "SummarizationMiddleware"

    @classmethod
    def from_settings(
        cls,
        settings: "SigilSettings",
        llm_client: Any = None,
    ) -> "SummarizationMiddleware":
        """Create middleware from SigilSettings.

        Extracts context-related settings from the settings object
        to configure the middleware. Falls back to defaults if
        context settings are not available.

        Args:
            settings: SigilSettings instance.
            llm_client: Optional LLM client for summarization.

        Returns:
            Configured SummarizationMiddleware instance.
        """
        context_settings = getattr(settings, "context", None)

        if context_settings:
            return cls(
                threshold_tokens=getattr(
                    context_settings, "summarization_threshold_tokens", 120000
                ),
                preserve_recent=getattr(
                    context_settings, "preserve_recent_messages", 6
                ),
                chars_per_token=getattr(
                    context_settings, "chars_per_token_estimate", 4
                ),
                enabled=getattr(
                    context_settings, "enable_auto_summarization", True
                ),
                llm_client=llm_client,
            )

        return cls(llm_client=llm_client)

    def _extract_messages(self, ctx: Any) -> list[Message]:
        """Extract conversation messages from context.

        Handles multiple formats for conversation history storage
        in the assembled context.

        Args:
            ctx: Pipeline context object.

        Returns:
            List of Message objects extracted from context.
        """
        messages = []

        # Try to get from assembled context
        assembled = getattr(ctx, "assembled_context", {})

        # Check for conversation_history key
        history = assembled.get("conversation_history", [])

        for item in history:
            if isinstance(item, dict):
                messages.append(
                    Message(
                        role=item.get("role", "user"),
                        content=item.get("content", ""),
                        timestamp=item.get("timestamp"),
                        metadata=item.get("metadata", {}),
                    )
                )
            elif hasattr(item, "role") and hasattr(item, "content"):
                messages.append(
                    Message(
                        role=item.role,
                        content=item.content,
                        timestamp=getattr(item, "timestamp", None),
                        metadata=getattr(item, "metadata", {}),
                    )
                )

        return messages

    def _update_context_with_summary(
        self,
        ctx: Any,
        summary: str,
        preserved_messages: list[Message],
    ) -> None:
        """Update context with summarized history.

        Replaces the conversation_history in the assembled context
        with a summary entry followed by the preserved recent messages.

        Args:
            ctx: Pipeline context object.
            summary: The generated summary text.
            preserved_messages: List of messages to preserve.
        """
        assembled = getattr(ctx, "assembled_context", {})

        # Create summary message
        summary_entry = {
            "role": "system",
            "content": f"[CONVERSATION SUMMARY]\n{summary}",
            "is_summary": True,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        # Convert preserved messages back to dict format
        preserved_dicts = [
            {
                "role": msg.role,
                "content": msg.content,
                "timestamp": msg.timestamp,
                "metadata": msg.metadata,
            }
            for msg in preserved_messages
        ]

        # Replace conversation history with summary + preserved
        assembled["conversation_history"] = [summary_entry] + preserved_dicts
        assembled["conversation_summarized"] = True

        ctx.assembled_context = assembled

    async def pre_step(self, step_name: str, ctx: Any) -> Any:
        """Check and summarize conversation before assemble step.

        This hook is called before each pipeline step. It only performs
        summarization for the "execute" step, which is when the context
        is about to be sent to the LLM.

        If the conversation history exceeds the token threshold,
        summarization is applied and the context is modified in place.

        Args:
            step_name: Name of the step about to execute.
            ctx: Pipeline context object.

        Returns:
            The context, potentially with summarized conversation history.
        """
        if not self.enabled:
            return ctx

        # Only process before execute step (after assemble)
        if step_name != "execute":
            return ctx

        # Extract messages
        messages = self._extract_messages(ctx)

        if not messages:
            return ctx

        # Check if summarization needed
        if not self.summarizer.should_summarize(messages):
            return ctx

        logger.info(
            f"Conversation has {len(messages)} messages, triggering summarization"
        )

        # Partition messages
        to_summarize, to_preserve = self.summarizer.partition_messages(messages)

        tokens_before = self.summarizer.estimate_messages_tokens(messages)

        # Generate summary
        if self.use_llm_summarization and self.llm_client:
            summary = await self.summarizer.summarize_with_llm(
                to_summarize, self.llm_client
            )
        else:
            summary = self.summarizer.create_basic_summary(to_summarize)

        # Update context
        self._update_context_with_summary(ctx, summary, to_preserve)

        # Calculate tokens after
        tokens_after = self.summarizer.estimate_tokens(
            summary
        ) + self.summarizer.estimate_messages_tokens(to_preserve)

        # Store result
        self._last_result = SummarizationResult(
            original_message_count=len(messages),
            preserved_message_count=len(to_preserve),
            summarized_message_count=len(to_summarize),
            summary=summary,
            tokens_before=tokens_before,
            tokens_after=tokens_after,
            tokens_saved=tokens_before - tokens_after,
        )

        logger.info(
            f"Summarized {len(to_summarize)} messages, "
            f"preserved {len(to_preserve)}, "
            f"saved ~{self._last_result.tokens_saved} tokens"
        )

        return ctx

    def get_last_result(self) -> Optional[SummarizationResult]:
        """Get the last summarization result.

        Returns the most recent summarization result, which can be used
        for monitoring, debugging, or metrics collection.

        Returns:
            The last SummarizationResult, or None if no summarization has occurred.
        """
        return self._last_result

    def reset(self) -> None:
        """Reset the middleware state.

        Clears the last summarization result. Useful for testing or when
        reusing the middleware across multiple requests.
        """
        self._last_result = None


__all__ = [
    "Message",
    "SummarizationResult",
    "ConversationSummarizer",
    "SummarizationMiddleware",
]
