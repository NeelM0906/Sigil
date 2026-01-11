"""Memory extraction for the Sigil v2 3-layer memory architecture.

This module implements the MemoryExtractor class, which uses LLM to identify
and extract discrete facts from raw content (resources).

Classes:
    MemoryExtractor: LLM-based fact extraction from resources.
    ExtractionStrategy: Protocol for different extraction approaches.

Example:
    >>> from sigil.memory.extraction import MemoryExtractor
    >>> extractor = MemoryExtractor()
    >>> facts = await extractor.extract_from_content(
    ...     content="User: I prefer monthly billing\\nAgent: Noted!",
    ...     resource_type="conversation"
    ... )
    >>> for fact in facts:
    ...     print(f"{fact['category']}: {fact['content']}")
"""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional, Protocol

from sigil.config import get_settings
from sigil.config.schemas.memory import Resource, MemoryItem


# =============================================================================
# Extraction Result
# =============================================================================


@dataclass
class ExtractedFact:
    """A fact extracted from a resource.

    Attributes:
        content: The extracted fact content.
        category: Suggested category for the fact.
        confidence: Extraction confidence (0.0-1.0).
        context: Additional context about the extraction.
    """

    content: str
    category: Optional[str] = None
    confidence: float = 1.0
    context: Optional[dict[str, Any]] = None


# =============================================================================
# Extraction Strategies
# =============================================================================


class ExtractionStrategy(Protocol):
    """Protocol for extraction strategies."""

    async def extract(
        self,
        content: str,
        resource_type: str,
        metadata: Optional[dict[str, Any]] = None,
    ) -> list[ExtractedFact]:
        """Extract facts from content.

        Args:
            content: The raw content to extract from.
            resource_type: Type of the resource.
            metadata: Optional resource metadata.

        Returns:
            List of extracted facts.
        """
        ...


class ConversationExtractionStrategy:
    """Extraction strategy for conversation resources.

    This strategy identifies preferences, objections, decisions, and
    other significant information from conversation transcripts.
    """

    # Categories relevant to conversations
    CATEGORIES = [
        "lead_preferences",
        "objection_patterns",
        "buying_signals",
        "contact_information",
        "timeline_requirements",
        "budget_information",
        "decision_makers",
        "pain_points",
        "competitor_mentions",
        "successful_approaches",
    ]

    EXTRACTION_PROMPT = """Analyze the following conversation and extract discrete facts.

For each fact, provide:
1. content: The specific fact or piece of information
2. category: One of {categories}
3. confidence: How confident you are in this extraction (0.0-1.0)

Focus on:
- Customer preferences and requirements
- Objections raised and how they were handled
- Buying signals and interest indicators
- Contact details or timeline information
- Budget or authority information
- Pain points or challenges mentioned
- Competitive mentions
- Approaches that worked well

Conversation:
```
{content}
```

Respond with a JSON array of extracted facts. Example:
[
  {{"content": "Customer prefers email communication", "category": "lead_preferences", "confidence": 0.95}},
  {{"content": "Budget concern raised about annual pricing", "category": "objection_patterns", "confidence": 0.85}}
]

Only include meaningful, actionable facts. Do not include generic observations.
Respond ONLY with the JSON array, no other text."""

    async def extract(
        self,
        content: str,
        resource_type: str,
        metadata: Optional[dict[str, Any]] = None,
    ) -> list[ExtractedFact]:
        """Extract facts from conversation content."""
        if not content.strip():
            return []

        prompt = self.EXTRACTION_PROMPT.format(
            categories=", ".join(self.CATEGORIES),
            content=content,
        )

        try:
            response = await self._call_llm(prompt)
            return self._parse_response(response)
        except Exception as e:
            # Fall back to rule-based extraction
            return self._rule_based_extract(content)

    async def _call_llm(self, prompt: str) -> str:
        """Call the LLM for extraction."""
        try:
            import anthropic

            settings = get_settings()
            api_key = settings.api_keys.anthropic_api_key

            if not api_key:
                raise ValueError("No API key available")

            client = anthropic.Anthropic(api_key=api_key)
            message = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=2000,
                messages=[{"role": "user", "content": prompt}],
            )

            return message.content[0].text

        except ImportError:
            raise ValueError("Anthropic library not installed")

    def _parse_response(self, response: str) -> list[ExtractedFact]:
        """Parse LLM response into ExtractedFact objects."""
        # Try to extract JSON from response
        try:
            # Handle potential markdown code blocks
            json_match = re.search(r"\[[\s\S]*\]", response)
            if json_match:
                data = json.loads(json_match.group())
            else:
                data = json.loads(response)

            facts = []
            for item in data:
                if isinstance(item, dict) and "content" in item:
                    facts.append(
                        ExtractedFact(
                            content=item["content"],
                            category=item.get("category"),
                            confidence=float(item.get("confidence", 0.8)),
                        )
                    )
            return facts

        except (json.JSONDecodeError, KeyError):
            return []

    def _rule_based_extract(self, content: str) -> list[ExtractedFact]:
        """Simple rule-based extraction as fallback."""
        facts = []

        # Look for preference patterns
        preference_patterns = [
            (r"(?:I |we |customer )prefer[s]?\s+(.+?)(?:\.|$)", "lead_preferences"),
            (r"(?:I |we )want\s+(.+?)(?:\.|$)", "lead_preferences"),
            (r"budget (?:is |of )?(?:around |about )?(\$?[\d,]+)", "budget_information"),
            (r"(?:call|email|contact) (?:me |us )?(?:at |on )?(.+?)(?:\.|$)", "contact_information"),
            (r"decision.*?(?:by |before )(.+?)(?:\.|$)", "timeline_requirements"),
        ]

        for pattern, category in preference_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            for match in matches:
                if len(match.strip()) > 5:
                    facts.append(
                        ExtractedFact(
                            content=match.strip(),
                            category=category,
                            confidence=0.6,
                        )
                    )

        return facts


class DocumentExtractionStrategy:
    """Extraction strategy for document resources.

    This strategy identifies key information, facts, and relationships
    from document content.
    """

    CATEGORIES = [
        "product_knowledge",
        "pricing_information",
        "feature_details",
        "competitor_analysis",
        "market_insights",
        "customer_testimonials",
        "process_documentation",
        "policy_information",
    ]

    EXTRACTION_PROMPT = """Analyze the following document and extract key facts.

For each fact, provide:
1. content: The specific fact or piece of information
2. category: One of {categories}
3. confidence: How confident you are in this extraction (0.0-1.0)

Focus on:
- Product features and capabilities
- Pricing and packaging information
- Competitive differentiators
- Customer success stories
- Process or policy details
- Market or industry insights

Document:
```
{content}
```

Respond with a JSON array of extracted facts. Example:
[
  {{"content": "Product supports API integration", "category": "feature_details", "confidence": 0.95}},
  {{"content": "Enterprise plan starts at $500/month", "category": "pricing_information", "confidence": 0.9}}
]

Only include meaningful, factual information. Do not include opinions or vague statements.
Respond ONLY with the JSON array, no other text."""

    async def extract(
        self,
        content: str,
        resource_type: str,
        metadata: Optional[dict[str, Any]] = None,
    ) -> list[ExtractedFact]:
        """Extract facts from document content."""
        if not content.strip():
            return []

        prompt = self.EXTRACTION_PROMPT.format(
            categories=", ".join(self.CATEGORIES),
            content=content[:8000],  # Limit content length
        )

        try:
            response = await self._call_llm(prompt)
            return self._parse_response(response)
        except Exception:
            return self._rule_based_extract(content)

    async def _call_llm(self, prompt: str) -> str:
        """Call the LLM for extraction."""
        try:
            import anthropic

            settings = get_settings()
            api_key = settings.api_keys.anthropic_api_key

            if not api_key:
                raise ValueError("No API key available")

            client = anthropic.Anthropic(api_key=api_key)
            message = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=2000,
                messages=[{"role": "user", "content": prompt}],
            )

            return message.content[0].text

        except ImportError:
            raise ValueError("Anthropic library not installed")

    def _parse_response(self, response: str) -> list[ExtractedFact]:
        """Parse LLM response into ExtractedFact objects."""
        try:
            json_match = re.search(r"\[[\s\S]*\]", response)
            if json_match:
                data = json.loads(json_match.group())
            else:
                data = json.loads(response)

            facts = []
            for item in data:
                if isinstance(item, dict) and "content" in item:
                    facts.append(
                        ExtractedFact(
                            content=item["content"],
                            category=item.get("category"),
                            confidence=float(item.get("confidence", 0.8)),
                        )
                    )
            return facts

        except (json.JSONDecodeError, KeyError):
            return []

    def _rule_based_extract(self, content: str) -> list[ExtractedFact]:
        """Simple rule-based extraction as fallback."""
        facts = []

        # Extract sentences that look like facts
        sentences = re.split(r"[.!?]\s+", content)
        for sentence in sentences[:20]:  # Limit to first 20 sentences
            sentence = sentence.strip()
            if len(sentence) > 20 and len(sentence) < 200:
                # Simple heuristic: sentences with numbers or specific patterns
                if re.search(r"\d+|supports?|enables?|provides?|includes?", sentence, re.IGNORECASE):
                    facts.append(
                        ExtractedFact(
                            content=sentence,
                            category="product_knowledge",
                            confidence=0.5,
                        )
                    )

        return facts[:10]  # Limit results


class ConfigExtractionStrategy:
    """Extraction strategy for configuration resources.

    This strategy identifies settings, parameters, and configuration patterns.
    """

    async def extract(
        self,
        content: str,
        resource_type: str,
        metadata: Optional[dict[str, Any]] = None,
    ) -> list[ExtractedFact]:
        """Extract facts from configuration content."""
        facts = []

        # Try to parse as JSON or key-value pairs
        try:
            data = json.loads(content)
            facts.extend(self._extract_from_dict(data, ""))
        except json.JSONDecodeError:
            # Try key=value format
            for line in content.split("\n"):
                if "=" in line:
                    key, _, value = line.partition("=")
                    key = key.strip()
                    value = value.strip()
                    if key and value:
                        facts.append(
                            ExtractedFact(
                                content=f"{key} is set to {value}",
                                category="configuration",
                                confidence=0.9,
                            )
                        )

        return facts

    def _extract_from_dict(
        self,
        data: dict[str, Any],
        prefix: str,
    ) -> list[ExtractedFact]:
        """Recursively extract facts from a dictionary."""
        facts = []

        for key, value in data.items():
            full_key = f"{prefix}.{key}" if prefix else key

            if isinstance(value, dict):
                facts.extend(self._extract_from_dict(value, full_key))
            elif isinstance(value, list):
                facts.append(
                    ExtractedFact(
                        content=f"{full_key} contains {len(value)} items",
                        category="configuration",
                        confidence=0.9,
                    )
                )
            else:
                facts.append(
                    ExtractedFact(
                        content=f"{full_key} is set to {value}",
                        category="configuration",
                        confidence=0.9,
                    )
                )

        return facts


class FeedbackExtractionStrategy:
    """Extraction strategy for feedback resources.

    This strategy identifies patterns, sentiments, and actionable insights
    from user feedback.
    """

    CATEGORIES = [
        "positive_feedback",
        "improvement_suggestions",
        "bug_reports",
        "feature_requests",
        "usability_issues",
        "performance_concerns",
    ]

    EXTRACTION_PROMPT = """Analyze the following feedback and extract key insights.

For each insight, provide:
1. content: The specific insight or pattern
2. category: One of {categories}
3. confidence: How confident you are in this extraction (0.0-1.0)

Feedback:
```
{content}
```

Respond with a JSON array of extracted insights. Example:
[
  {{"content": "Users appreciate the intuitive interface", "category": "positive_feedback", "confidence": 0.9}},
  {{"content": "Request for dark mode support", "category": "feature_requests", "confidence": 0.95}}
]

Respond ONLY with the JSON array, no other text."""

    async def extract(
        self,
        content: str,
        resource_type: str,
        metadata: Optional[dict[str, Any]] = None,
    ) -> list[ExtractedFact]:
        """Extract insights from feedback content."""
        if not content.strip():
            return []

        prompt = self.EXTRACTION_PROMPT.format(
            categories=", ".join(self.CATEGORIES),
            content=content,
        )

        try:
            response = await self._call_llm(prompt)
            return self._parse_response(response)
        except Exception:
            return self._rule_based_extract(content)

    async def _call_llm(self, prompt: str) -> str:
        """Call the LLM for extraction."""
        try:
            import anthropic

            settings = get_settings()
            api_key = settings.api_keys.anthropic_api_key

            if not api_key:
                raise ValueError("No API key available")

            client = anthropic.Anthropic(api_key=api_key)
            message = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=2000,
                messages=[{"role": "user", "content": prompt}],
            )

            return message.content[0].text

        except ImportError:
            raise ValueError("Anthropic library not installed")

    def _parse_response(self, response: str) -> list[ExtractedFact]:
        """Parse LLM response into ExtractedFact objects."""
        try:
            json_match = re.search(r"\[[\s\S]*\]", response)
            if json_match:
                data = json.loads(json_match.group())
            else:
                data = json.loads(response)

            facts = []
            for item in data:
                if isinstance(item, dict) and "content" in item:
                    facts.append(
                        ExtractedFact(
                            content=item["content"],
                            category=item.get("category"),
                            confidence=float(item.get("confidence", 0.8)),
                        )
                    )
            return facts

        except (json.JSONDecodeError, KeyError):
            return []

    def _rule_based_extract(self, content: str) -> list[ExtractedFact]:
        """Simple sentiment-based extraction as fallback."""
        facts = []

        positive_words = ["love", "great", "excellent", "amazing", "helpful"]
        negative_words = ["hate", "terrible", "broken", "slow", "confusing"]
        request_words = ["wish", "would be nice", "please add", "need", "want"]

        content_lower = content.lower()

        for word in positive_words:
            if word in content_lower:
                facts.append(
                    ExtractedFact(
                        content=f"Positive sentiment: mentions '{word}'",
                        category="positive_feedback",
                        confidence=0.6,
                    )
                )
                break

        for word in negative_words:
            if word in content_lower:
                facts.append(
                    ExtractedFact(
                        content=f"Negative sentiment: mentions '{word}'",
                        category="improvement_suggestions",
                        confidence=0.6,
                    )
                )
                break

        for word in request_words:
            if word in content_lower:
                facts.append(
                    ExtractedFact(
                        content=f"Feature request detected: mentions '{word}'",
                        category="feature_requests",
                        confidence=0.6,
                    )
                )
                break

        return facts


# =============================================================================
# Memory Extractor
# =============================================================================


class MemoryExtractor:
    """LLM-based memory extraction from resources.

    This class orchestrates fact extraction from different resource types
    using appropriate strategies. It supports both LLM-based and rule-based
    extraction with graceful fallback.

    Attributes:
        strategies: Dictionary mapping resource types to extraction strategies.

    Example:
        >>> extractor = MemoryExtractor()
        >>> facts = await extractor.extract_from_content(
        ...     content="Customer wants monthly billing",
        ...     resource_type="conversation"
        ... )
    """

    def __init__(self) -> None:
        """Initialize the MemoryExtractor with default strategies."""
        self.strategies: dict[str, ExtractionStrategy] = {
            "conversation": ConversationExtractionStrategy(),
            "document": DocumentExtractionStrategy(),
            "config": ConfigExtractionStrategy(),
            "feedback": FeedbackExtractionStrategy(),
        }

    def register_strategy(
        self,
        resource_type: str,
        strategy: ExtractionStrategy,
    ) -> None:
        """Register a custom extraction strategy.

        Args:
            resource_type: The resource type this strategy handles.
            strategy: The extraction strategy implementation.
        """
        self.strategies[resource_type] = strategy

    async def extract_from_content(
        self,
        content: str,
        resource_type: str,
        metadata: Optional[dict[str, Any]] = None,
    ) -> list[ExtractedFact]:
        """Extract facts from raw content.

        Args:
            content: The raw content to extract from.
            resource_type: Type of the resource.
            metadata: Optional resource metadata.

        Returns:
            List of ExtractedFact objects.
        """
        strategy = self.strategies.get(resource_type)
        if strategy is None:
            # Use document strategy as default
            strategy = self.strategies.get("document", DocumentExtractionStrategy())

        return await strategy.extract(content, resource_type, metadata)

    async def extract_from_resource(
        self,
        resource: Resource,
    ) -> list[ExtractedFact]:
        """Extract facts from a Resource object.

        Args:
            resource: The Resource to extract from.

        Returns:
            List of ExtractedFact objects.
        """
        return await self.extract_from_content(
            content=resource.content,
            resource_type=resource.resource_type,
            metadata=resource.metadata,
        )

    async def extract_batch(
        self,
        resources: list[Resource],
    ) -> dict[str, list[ExtractedFact]]:
        """Extract facts from multiple resources.

        Args:
            resources: List of Resources to extract from.

        Returns:
            Dictionary mapping resource IDs to their extracted facts.
        """
        results: dict[str, list[ExtractedFact]] = {}

        for resource in resources:
            facts = await self.extract_from_resource(resource)
            results[resource.resource_id] = facts

        return results


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "MemoryExtractor",
    "ExtractedFact",
    "ExtractionStrategy",
    "ConversationExtractionStrategy",
    "DocumentExtractionStrategy",
    "ConfigExtractionStrategy",
    "FeedbackExtractionStrategy",
]
