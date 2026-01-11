"""Memory consolidation for the Sigil v2 3-layer memory architecture.

This module implements the MemoryConsolidator class, which aggregates memory
items into coherent markdown categories using LLM.

Classes:
    MemoryConsolidator: LLM-based consolidation of memory items.
    ConsolidationTrigger: Enum for consolidation trigger types.

Example:
    >>> from sigil.memory.consolidation import MemoryConsolidator
    >>> consolidator = MemoryConsolidator()
    >>> markdown = await consolidator.consolidate(
    ...     category_name="lead_preferences",
    ...     items=[item1, item2, item3],
    ...     existing_content="## Previous Knowledge\\n..."
    ... )
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

from sigil.config import get_settings
from sigil.config.schemas.memory import MemoryItem, MemoryCategory


# =============================================================================
# Enums and Types
# =============================================================================


class ConsolidationTrigger(Enum):
    """Types of events that trigger consolidation."""

    MANUAL = "manual"  # User-initiated consolidation
    ITEM_THRESHOLD = "item_threshold"  # After N new items
    SCHEDULED = "scheduled"  # Time-based schedule
    ON_DEMAND = "on_demand"  # When category is accessed


@dataclass
class ConsolidationResult:
    """Result of a consolidation operation.

    Attributes:
        category_name: Name of the consolidated category.
        markdown_content: The generated markdown content.
        items_consolidated: Number of items consolidated.
        is_incremental: Whether this was an incremental update.
        trigger: What triggered the consolidation.
    """

    category_name: str
    markdown_content: str
    items_consolidated: int
    is_incremental: bool
    trigger: ConsolidationTrigger


# =============================================================================
# Category Templates
# =============================================================================

CATEGORY_TEMPLATES: dict[str, dict[str, str]] = {
    "lead_preferences": {
        "description": "Customer preferences, requirements, and communication styles",
        "template": """# Lead Preferences

## Communication Preferences
<!-- How customers prefer to be contacted -->

## Product Preferences
<!-- What features/capabilities matter most -->

## Timeline & Urgency
<!-- Decision timelines and urgency levels -->

## Additional Insights
<!-- Other relevant preferences -->
""",
    },
    "objection_patterns": {
        "description": "Common objections and effective response strategies",
        "template": """# Objection Patterns

## Pricing Objections
<!-- Common pricing concerns and responses -->

## Timing Objections
<!-- "Not the right time" patterns -->

## Authority Objections
<!-- Decision-maker related objections -->

## Competitor Comparisons
<!-- How we compare and differentiate -->

## Effective Responses
<!-- Approaches that have worked well -->
""",
    },
    "product_knowledge": {
        "description": "Product features, capabilities, and technical details",
        "template": """# Product Knowledge

## Core Features
<!-- Main product capabilities -->

## Technical Details
<!-- Technical specifications and requirements -->

## Integration Capabilities
<!-- How the product integrates with other systems -->

## Pricing & Packaging
<!-- Pricing tiers and packages -->

## FAQs
<!-- Frequently asked questions -->
""",
    },
    "conversation_insights": {
        "description": "Patterns and insights from customer conversations",
        "template": """# Conversation Insights

## Buying Signals
<!-- Indicators of purchase intent -->

## Pain Points
<!-- Common customer challenges -->

## Decision Factors
<!-- What influences buying decisions -->

## Successful Techniques
<!-- Conversation approaches that work -->
""",
    },
    "successful_approaches": {
        "description": "Strategies and techniques that have proven effective",
        "template": """# Successful Approaches

## Discovery Techniques
<!-- Effective ways to understand needs -->

## Presentation Strategies
<!-- What resonates in product presentations -->

## Closing Techniques
<!-- Approaches that help close deals -->

## Relationship Building
<!-- Trust and rapport building methods -->
""",
    },
}


# =============================================================================
# Memory Consolidator
# =============================================================================


class MemoryConsolidator:
    """LLM-based consolidation of memory items into markdown categories.

    This class aggregates discrete memory items into coherent, human-readable
    markdown content. It supports both full rebuilds and incremental updates.

    Attributes:
        item_threshold: Number of items before auto-consolidation triggers.
        templates: Category templates for initialization.

    Example:
        >>> consolidator = MemoryConsolidator(item_threshold=50)
        >>> result = await consolidator.consolidate(
        ...     category_name="lead_preferences",
        ...     items=[item1, item2],
        ...     existing_content=""
        ... )
    """

    CONSOLIDATION_PROMPT = """You are consolidating memory items into a coherent markdown document.

Category: {category_name}
Description: {description}

{existing_section}

New items to incorporate:
{items_content}

Instructions:
1. Integrate the new items into the existing content
2. Group related information together
3. Use clear markdown formatting (headers, lists, emphasis)
4. Remove duplicates and consolidate similar points
5. Keep the content actionable and easy to scan
6. Preserve important details while being concise

Output the complete updated markdown content. Start directly with the content, no preamble."""

    INCREMENTAL_PROMPT = """Update the existing category content with new information.

Category: {category_name}

Existing Content:
```markdown
{existing_content}
```

New items to incorporate:
{items_content}

Instructions:
1. Add the new information to the appropriate sections
2. Update existing points if new items provide better information
3. Do not remove existing valuable content
4. Keep formatting consistent
5. Mark new or updated sections if helpful

Output the complete updated markdown content."""

    def __init__(
        self,
        item_threshold: int = 100,
        templates: Optional[dict[str, dict[str, str]]] = None,
    ) -> None:
        """Initialize the MemoryConsolidator.

        Args:
            item_threshold: Number of items before auto-consolidation triggers.
            templates: Custom category templates (uses defaults if not provided).
        """
        self.item_threshold = item_threshold
        self.templates = templates or CATEGORY_TEMPLATES

    def get_template(self, category_name: str) -> Optional[dict[str, str]]:
        """Get the template for a category.

        Args:
            category_name: The category name.

        Returns:
            Template dict with 'description' and 'template' keys, or None.
        """
        normalized = category_name.strip().lower().replace(" ", "_").replace("-", "_")
        return self.templates.get(normalized)

    async def consolidate(
        self,
        category_name: str,
        items: list[MemoryItem],
        existing_content: str = "",
        description: str = "",
        trigger: ConsolidationTrigger = ConsolidationTrigger.MANUAL,
    ) -> ConsolidationResult:
        """Consolidate memory items into markdown content.

        Args:
            category_name: Name of the category.
            items: List of MemoryItems to consolidate.
            existing_content: Existing category content (for incremental updates).
            description: Category description.
            trigger: What triggered this consolidation.

        Returns:
            ConsolidationResult with the generated markdown.
        """
        if not items:
            return ConsolidationResult(
                category_name=category_name,
                markdown_content=existing_content or self._get_initial_content(category_name),
                items_consolidated=0,
                is_incremental=bool(existing_content),
                trigger=trigger,
            )

        # Format items for the prompt
        items_content = self._format_items(items)

        # Get description from template if not provided
        if not description:
            template = self.get_template(category_name)
            if template:
                description = template.get("description", "")
            else:
                description = f"Knowledge about {category_name.replace('_', ' ')}"

        # Choose prompt based on whether this is incremental
        is_incremental = bool(existing_content.strip())

        if is_incremental:
            prompt = self.INCREMENTAL_PROMPT.format(
                category_name=category_name,
                existing_content=existing_content,
                items_content=items_content,
            )
        else:
            # For fresh consolidation, include template structure if available
            template = self.get_template(category_name)
            existing_section = ""
            if template:
                existing_section = f"Use this structure as a guide:\n```markdown\n{template['template']}\n```"

            prompt = self.CONSOLIDATION_PROMPT.format(
                category_name=category_name,
                description=description,
                existing_section=existing_section,
                items_content=items_content,
            )

        try:
            markdown_content = await self._call_llm(prompt)
        except Exception as e:
            # Fall back to simple concatenation
            markdown_content = self._fallback_consolidation(
                category_name, items, existing_content
            )

        return ConsolidationResult(
            category_name=category_name,
            markdown_content=markdown_content,
            items_consolidated=len(items),
            is_incremental=is_incremental,
            trigger=trigger,
        )

    async def consolidate_incremental(
        self,
        category_name: str,
        new_items: list[MemoryItem],
        existing_content: str,
        trigger: ConsolidationTrigger = ConsolidationTrigger.ITEM_THRESHOLD,
    ) -> ConsolidationResult:
        """Perform incremental consolidation (add new items to existing content).

        Args:
            category_name: Name of the category.
            new_items: New items to add.
            existing_content: Current category content.
            trigger: What triggered this consolidation.

        Returns:
            ConsolidationResult with updated markdown.
        """
        return await self.consolidate(
            category_name=category_name,
            items=new_items,
            existing_content=existing_content,
            trigger=trigger,
        )

    async def rebuild_category(
        self,
        category_name: str,
        all_items: list[MemoryItem],
        description: str = "",
    ) -> ConsolidationResult:
        """Rebuild a category from scratch.

        Args:
            category_name: Name of the category.
            all_items: All items to include.
            description: Category description.

        Returns:
            ConsolidationResult with fresh markdown.
        """
        return await self.consolidate(
            category_name=category_name,
            items=all_items,
            existing_content="",
            description=description,
            trigger=ConsolidationTrigger.MANUAL,
        )

    def should_consolidate(
        self,
        category: MemoryCategory,
        new_item_count: int,
    ) -> bool:
        """Check if a category should be consolidated.

        Args:
            category: The category to check.
            new_item_count: Number of new items since last consolidation.

        Returns:
            True if consolidation should be triggered.
        """
        # Trigger if we've accumulated enough new items
        return new_item_count >= self.item_threshold

    def _format_items(self, items: list[MemoryItem]) -> str:
        """Format memory items for the prompt.

        Args:
            items: List of MemoryItems.

        Returns:
            Formatted string representation.
        """
        lines = []
        for i, item in enumerate(items, 1):
            confidence = f" (confidence: {item.confidence:.0%})" if item.confidence < 1.0 else ""
            category = f" [{item.category}]" if item.category else ""
            lines.append(f"{i}. {item.content}{category}{confidence}")

        return "\n".join(lines)

    def _get_initial_content(self, category_name: str) -> str:
        """Get initial content for a new category.

        Args:
            category_name: The category name.

        Returns:
            Initial markdown content.
        """
        template = self.get_template(category_name)
        if template:
            return template["template"]

        # Generate a simple default
        title = category_name.replace("_", " ").title()
        return f"# {title}\n\n*No content yet. Items will be consolidated here.*\n"

    def _fallback_consolidation(
        self,
        category_name: str,
        items: list[MemoryItem],
        existing_content: str,
    ) -> str:
        """Simple fallback consolidation without LLM.

        Args:
            category_name: The category name.
            items: Items to consolidate.
            existing_content: Existing content.

        Returns:
            Consolidated markdown.
        """
        title = category_name.replace("_", " ").title()

        # Group items by their category if they have one
        grouped: dict[str, list[str]] = {}
        uncategorized: list[str] = []

        for item in items:
            if item.category:
                if item.category not in grouped:
                    grouped[item.category] = []
                grouped[item.category].append(item.content)
            else:
                uncategorized.append(item.content)

        # Build markdown
        lines = [f"# {title}", ""]

        if existing_content.strip():
            lines.append(existing_content.strip())
            lines.append("")
            lines.append("---")
            lines.append("")
            lines.append("## Recent Additions")
            lines.append("")

        for category, category_items in sorted(grouped.items()):
            section_title = category.replace("_", " ").title()
            lines.append(f"### {section_title}")
            for content in category_items:
                lines.append(f"- {content}")
            lines.append("")

        if uncategorized:
            lines.append("### Other")
            for content in uncategorized:
                lines.append(f"- {content}")
            lines.append("")

        return "\n".join(lines)

    async def _call_llm(self, prompt: str) -> str:
        """Call the LLM for consolidation.

        Args:
            prompt: The consolidation prompt.

        Returns:
            Generated markdown content.
        """
        try:
            import anthropic

            settings = get_settings()
            api_key = settings.api_keys.anthropic_api_key

            if not api_key:
                raise ValueError("No API key available")

            client = anthropic.Anthropic(api_key=api_key)
            message = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=4000,
                messages=[{"role": "user", "content": prompt}],
            )

            return message.content[0].text

        except ImportError:
            raise ValueError("Anthropic library not installed")


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "MemoryConsolidator",
    "ConsolidationTrigger",
    "ConsolidationResult",
    "CATEGORY_TEMPLATES",
]
