"""Memory tools for Sigil v2 agents.

This module provides LangChain-compatible tools that agents can use to
interact with the memory system. These tools enable agents to recall
information, store new facts, and access aggregated knowledge.

Tools:
    recall: Retrieve memories by query with mode selection.
    remember: Store important facts to memory.
    list_categories: List available memory categories.
    get_category: Get full markdown content of a category.

Example:
    >>> from sigil.tools.builtin.memory_tools import create_memory_tools
    >>> tools = create_memory_tools(memory_manager)
    >>> agent = create_agent(tools=tools)
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Callable, Optional, Type, Union

from pydantic import BaseModel, Field

# Try to import LangChain, but make it optional
try:
    from langchain_core.tools import BaseTool, StructuredTool
    from langchain_core.callbacks import CallbackManagerForToolRun, AsyncCallbackManagerForToolRun

    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    BaseTool = object
    StructuredTool = None


# =============================================================================
# Tool Input Schemas
# =============================================================================


class RecallInput(BaseModel):
    """Input schema for the recall tool."""

    query: str = Field(
        description="The search query to find relevant memories."
    )
    k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum number of memories to retrieve (1-20)."
    )
    category: Optional[str] = Field(
        default=None,
        description="Optional category to filter by (e.g., 'lead_preferences')."
    )
    mode: str = Field(
        default="hybrid",
        description="Retrieval mode: 'rag' (fast), 'llm' (accurate), or 'hybrid' (balanced)."
    )


class RememberInput(BaseModel):
    """Input schema for the remember tool."""

    content: str = Field(
        description="The fact or information to remember."
    )
    category: Optional[str] = Field(
        default=None,
        description="Category for this memory (e.g., 'lead_preferences', 'objection_patterns')."
    )
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Confidence level for this memory (0.0 to 1.0)."
    )


class GetCategoryInput(BaseModel):
    """Input schema for the get_category tool."""

    name: str = Field(
        description="Name of the category to retrieve (e.g., 'lead_preferences')."
    )


# =============================================================================
# Tool Implementations (Function-based)
# =============================================================================


def create_recall_tool(
    memory_manager: Any,
    session_id: Optional[str] = None,
) -> Callable:
    """Create a recall function for retrieving memories.

    Args:
        memory_manager: The MemoryManager instance.
        session_id: Optional session ID for context.

    Returns:
        A callable recall function.
    """

    async def recall(
        query: str,
        k: int = 5,
        category: Optional[str] = None,
        mode: str = "hybrid",
    ) -> str:
        """Retrieve relevant memories based on a query.

        Use this tool to recall information from memory. It searches through
        stored facts and returns the most relevant ones.

        Args:
            query: What you want to recall (e.g., "customer preferences").
            k: Maximum number of results (default: 5).
            category: Optional category filter.
            mode: 'rag' (fast), 'llm' (accurate), or 'hybrid' (balanced).

        Returns:
            JSON string with retrieved memories.
        """
        try:
            results = await memory_manager.recall(
                query=query,
                k=k,
                category=category,
                mode=mode,
            )

            if not results:
                return json.dumps({
                    "status": "success",
                    "memories": [],
                    "message": "No relevant memories found."
                })

            return json.dumps({
                "status": "success",
                "memories": results,
                "count": len(results),
            })

        except Exception as e:
            return json.dumps({
                "status": "error",
                "error": str(e),
            })

    return recall


def create_remember_tool(
    memory_manager: Any,
    session_id: Optional[str] = None,
) -> Callable:
    """Create a remember function for storing facts.

    Args:
        memory_manager: The MemoryManager instance.
        session_id: Optional session ID for context.

    Returns:
        A callable remember function.
    """

    async def remember(
        content: str,
        category: Optional[str] = None,
        confidence: float = 1.0,
    ) -> str:
        """Store an important fact in memory.

        Use this tool to remember important information for future reference.
        The fact will be stored with an embedding for later retrieval.

        Args:
            content: The fact to remember.
            category: Category for organization (e.g., 'lead_preferences').
            confidence: How confident you are in this fact (0.0-1.0).

        Returns:
            JSON string confirming storage.
        """
        try:
            item = await memory_manager.remember(
                content=content,
                category=category,
                session_id=session_id,
                confidence=confidence,
            )

            return json.dumps({
                "status": "success",
                "item_id": item.item_id,
                "message": f"Stored: {content[:50]}..." if len(content) > 50 else f"Stored: {content}",
            })

        except Exception as e:
            return json.dumps({
                "status": "error",
                "error": str(e),
            })

    return remember


def create_list_categories_tool(memory_manager: Any) -> Callable:
    """Create a function to list all categories.

    Args:
        memory_manager: The MemoryManager instance.

    Returns:
        A callable function.
    """

    async def list_categories() -> str:
        """List all available memory categories.

        Use this tool to see what categories of knowledge are available.
        Categories organize memories by topic (e.g., lead_preferences,
        objection_patterns, product_knowledge).

        Returns:
            JSON string with list of categories.
        """
        try:
            categories = await memory_manager.list_categories()

            if not categories:
                return json.dumps({
                    "status": "success",
                    "categories": [],
                    "message": "No categories yet. Use 'remember' to start building memory.",
                })

            return json.dumps({
                "status": "success",
                "categories": categories,
                "count": len(categories),
            })

        except Exception as e:
            return json.dumps({
                "status": "error",
                "error": str(e),
            })

    return list_categories


def create_get_category_tool(memory_manager: Any) -> Callable:
    """Create a function to get category content.

    Args:
        memory_manager: The MemoryManager instance.

    Returns:
        A callable function.
    """

    async def get_category(name: str) -> str:
        """Get the full content of a memory category.

        Use this tool to retrieve aggregated knowledge from a category.
        Categories contain consolidated insights in markdown format.

        Args:
            name: Category name (e.g., 'lead_preferences').

        Returns:
            The markdown content of the category.
        """
        try:
            content = await memory_manager.get_category_content(name)

            if content is None:
                return json.dumps({
                    "status": "not_found",
                    "message": f"Category '{name}' not found.",
                })

            return json.dumps({
                "status": "success",
                "name": name,
                "content": content,
            })

        except Exception as e:
            return json.dumps({
                "status": "error",
                "error": str(e),
            })

    return get_category


# =============================================================================
# LangChain Tool Classes
# =============================================================================


if LANGCHAIN_AVAILABLE:

    class RecallTool(BaseTool):
        """LangChain tool for recalling memories."""

        name: str = "recall"
        description: str = (
            "Retrieve relevant memories based on a query. "
            "Use this to recall information from memory. "
            "Input: query string, optional k (number of results), "
            "optional category filter, optional mode (rag/llm/hybrid)."
        )
        args_schema: Type[BaseModel] = RecallInput

        memory_manager: Any = None
        session_id: Optional[str] = None

        def _run(
            self,
            query: str,
            k: int = 5,
            category: Optional[str] = None,
            mode: str = "hybrid",
            run_manager: Optional[CallbackManagerForToolRun] = None,
        ) -> str:
            """Synchronous execution."""
            return asyncio.run(self._arun(query, k, category, mode))

        async def _arun(
            self,
            query: str,
            k: int = 5,
            category: Optional[str] = None,
            mode: str = "hybrid",
            run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
        ) -> str:
            """Asynchronous execution."""
            recall_func = create_recall_tool(self.memory_manager, self.session_id)
            return await recall_func(query, k, category, mode)

    class RememberTool(BaseTool):
        """LangChain tool for storing memories."""

        name: str = "remember"
        description: str = (
            "Store an important fact in memory for future reference. "
            "Use this to remember customer preferences, insights, or key information. "
            "Input: content (the fact), optional category, optional confidence."
        )
        args_schema: Type[BaseModel] = RememberInput

        memory_manager: Any = None
        session_id: Optional[str] = None

        def _run(
            self,
            content: str,
            category: Optional[str] = None,
            confidence: float = 1.0,
            run_manager: Optional[CallbackManagerForToolRun] = None,
        ) -> str:
            """Synchronous execution."""
            return asyncio.run(self._arun(content, category, confidence))

        async def _arun(
            self,
            content: str,
            category: Optional[str] = None,
            confidence: float = 1.0,
            run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
        ) -> str:
            """Asynchronous execution."""
            remember_func = create_remember_tool(self.memory_manager, self.session_id)
            return await remember_func(content, category, confidence)

    class ListCategoriesTool(BaseTool):
        """LangChain tool for listing categories."""

        name: str = "list_categories"
        description: str = (
            "List all available memory categories. "
            "Use this to see what organized knowledge is available. "
            "No input required."
        )

        memory_manager: Any = None

        def _run(
            self,
            run_manager: Optional[CallbackManagerForToolRun] = None,
        ) -> str:
            """Synchronous execution."""
            return asyncio.run(self._arun())

        async def _arun(
            self,
            run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
        ) -> str:
            """Asynchronous execution."""
            list_func = create_list_categories_tool(self.memory_manager)
            return await list_func()

    class GetCategoryTool(BaseTool):
        """LangChain tool for getting category content."""

        name: str = "get_category"
        description: str = (
            "Get the full content of a memory category. "
            "Use this to retrieve aggregated knowledge about a topic. "
            "Input: category name (e.g., 'lead_preferences')."
        )
        args_schema: Type[BaseModel] = GetCategoryInput

        memory_manager: Any = None

        def _run(
            self,
            name: str,
            run_manager: Optional[CallbackManagerForToolRun] = None,
        ) -> str:
            """Synchronous execution."""
            return asyncio.run(self._arun(name))

        async def _arun(
            self,
            name: str,
            run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
        ) -> str:
            """Asynchronous execution."""
            get_func = create_get_category_tool(self.memory_manager)
            return await get_func(name)


# =============================================================================
# Factory Functions
# =============================================================================


def create_memory_tools(
    memory_manager: Any,
    session_id: Optional[str] = None,
    as_langchain: bool = True,
) -> list[Any]:
    """Create all memory tools for an agent.

    Args:
        memory_manager: The MemoryManager instance.
        session_id: Optional session ID for context.
        as_langchain: If True, return LangChain tools; otherwise return functions.

    Returns:
        List of tools (LangChain BaseTool or callable functions).
    """
    if as_langchain and LANGCHAIN_AVAILABLE:
        return [
            RecallTool(memory_manager=memory_manager, session_id=session_id),
            RememberTool(memory_manager=memory_manager, session_id=session_id),
            ListCategoriesTool(memory_manager=memory_manager),
            GetCategoryTool(memory_manager=memory_manager),
        ]
    else:
        return [
            create_recall_tool(memory_manager, session_id),
            create_remember_tool(memory_manager, session_id),
            create_list_categories_tool(memory_manager),
            create_get_category_tool(memory_manager),
        ]


def create_recall_structured_tool(
    memory_manager: Any,
    session_id: Optional[str] = None,
) -> Any:
    """Create a structured recall tool using LangChain's StructuredTool.

    Args:
        memory_manager: The MemoryManager instance.
        session_id: Optional session ID.

    Returns:
        StructuredTool instance.
    """
    if not LANGCHAIN_AVAILABLE or StructuredTool is None:
        raise ImportError("LangChain is required for structured tools")

    recall_func = create_recall_tool(memory_manager, session_id)

    return StructuredTool.from_function(
        func=lambda **kwargs: asyncio.run(recall_func(**kwargs)),
        coroutine=recall_func,
        name="recall",
        description="Retrieve relevant memories based on a query.",
        args_schema=RecallInput,
    )


def create_remember_structured_tool(
    memory_manager: Any,
    session_id: Optional[str] = None,
) -> Any:
    """Create a structured remember tool using LangChain's StructuredTool.

    Args:
        memory_manager: The MemoryManager instance.
        session_id: Optional session ID.

    Returns:
        StructuredTool instance.
    """
    if not LANGCHAIN_AVAILABLE or StructuredTool is None:
        raise ImportError("LangChain is required for structured tools")

    remember_func = create_remember_tool(memory_manager, session_id)

    return StructuredTool.from_function(
        func=lambda **kwargs: asyncio.run(remember_func(**kwargs)),
        coroutine=remember_func,
        name="remember",
        description="Store an important fact in memory.",
        args_schema=RememberInput,
    )


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Input schemas
    "RecallInput",
    "RememberInput",
    "GetCategoryInput",
    # Factory functions
    "create_memory_tools",
    "create_recall_tool",
    "create_remember_tool",
    "create_list_categories_tool",
    "create_get_category_tool",
    "create_recall_structured_tool",
    "create_remember_structured_tool",
    # LangChain availability flag
    "LANGCHAIN_AVAILABLE",
]

# Conditionally export LangChain tool classes
if LANGCHAIN_AVAILABLE:
    __all__.extend([
        "RecallTool",
        "RememberTool",
        "ListCategoriesTool",
        "GetCategoryTool",
    ])
