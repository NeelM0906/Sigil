"""Planner for Sigil v2 Phase 5 Planning & Reasoning.

This module implements goal-to-plan decomposition using LLM-based planning.
The Planner creates execution plans with proper dependency ordering,
DAG validation, and caching for similar goals.

Classes:
    Planner: Creates execution plans from high-level goals.

Example:
    >>> from sigil.planning.planner import Planner
    >>> from sigil.state.store import EventStore
    >>>
    >>> planner = Planner(event_store=EventStore())
    >>> plan = await planner.create_plan(
    ...     goal="Qualify lead John from Acme Corp",
    ...     context={"lead_name": "John", "company": "Acme Corp"},
    ...     tools=["crm.get_contact", "crm.get_history"],
    ...     max_steps=10,
    ... )
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
import time
from datetime import datetime, timezone, timedelta
from typing import Any, Optional

from sigil.config.schemas.plan import Plan, PlanStep, PlanStatus
from sigil.config import get_settings
from sigil.core.exceptions import SigilError
from sigil.state.events import Event, EventType, _generate_event_id, _get_utc_now
from sigil.state.store import EventStore
from sigil.telemetry.tokens import TokenTracker

from sigil.planning.schemas import (
    PlanStepConfig,
    PlanMetadata,
    PlanConstraints,
    StepType,
    generate_uuid,
)
from sigil.planning.executors.builtin_executor import BuiltinToolExecutor


# =============================================================================
# Module Logger
# =============================================================================

logger = logging.getLogger(__name__)


# =============================================================================
# Constants
# =============================================================================

CACHE_TTL_HOURS = 24
"""Default cache TTL in hours for similar goal plans."""

DEFAULT_MAX_STEPS = 10
"""Default maximum steps per plan."""


# =============================================================================
# Event Creators
# =============================================================================


def create_plan_created_event(
    session_id: str,
    plan_id: str,
    goal: str,
    step_count: int,
    complexity: float,
    estimated_tokens: int,
    correlation_id: Optional[str] = None,
) -> Event:
    """Create a PlanCreatedEvent.

    Args:
        session_id: Session identifier.
        plan_id: Unique plan identifier.
        goal: The goal being planned.
        step_count: Number of steps in the plan.
        complexity: Assessed complexity (0.0-1.0).
        estimated_tokens: Estimated total tokens.
        correlation_id: Optional correlation ID.

    Returns:
        Event object.
    """
    payload = {
        "plan_id": plan_id,
        "goal": goal,
        "step_count": step_count,
        "complexity": complexity,
        "estimated_tokens": estimated_tokens,
    }

    return Event(
        event_id=_generate_event_id(),
        event_type=EventType.PLAN_CREATED,
        timestamp=_get_utc_now(),
        session_id=session_id,
        correlation_id=correlation_id,
        payload=payload,
    )


# =============================================================================
# Exceptions
# =============================================================================


class PlanningError(SigilError):
    """Error during plan creation."""

    def __init__(self, message: str, goal: Optional[str] = None, **kwargs: Any) -> None:
        super().__init__(message, code="PLANNING_ERROR", **kwargs)
        self.goal = goal


class DAGValidationError(PlanningError):
    """Error in dependency DAG validation (circular dependencies)."""

    def __init__(
        self,
        message: str,
        cycle_steps: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(message, **kwargs)
        self.cycle_steps = cycle_steps or []


# =============================================================================
# Cache Entry
# =============================================================================


class CacheEntry:
    """Cached plan entry with TTL tracking."""

    def __init__(self, plan: Plan, ttl_hours: int = CACHE_TTL_HOURS) -> None:
        self.plan = plan
        self.created_at = datetime.now(timezone.utc)
        self.ttl_hours = ttl_hours
        self.hits = 0

    @property
    def is_expired(self) -> bool:
        """Check if the cache entry has expired."""
        expiry = self.created_at + timedelta(hours=self.ttl_hours)
        return datetime.now(timezone.utc) > expiry

    def touch(self) -> None:
        """Record a cache hit."""
        self.hits += 1


# =============================================================================
# Tool Description Registry
# =============================================================================

# Builtin tool descriptions with argument schemas
BUILTIN_TOOL_DESCRIPTIONS: dict[str, dict[str, Any]] = {
    "memory.recall": {
        "description": "Retrieve relevant memories from the knowledge base using semantic search",
        "arguments": {
            "query": {"type": "string", "required": True, "description": "Memory search query"},
            "k": {"type": "integer", "required": False, "default": 5, "description": "Number of memories to retrieve"},
            "category": {"type": "string", "required": False, "description": "Filter by memory category"},
            "mode": {"type": "string", "required": False, "default": "hybrid", "description": "Search mode: 'semantic', 'keyword', or 'hybrid'"},
        },
    },
    "memory.retrieve": {
        "description": "Alias for memory.recall - retrieve memories by query",
        "arguments": {
            "query": {"type": "string", "required": True, "description": "Memory search query"},
            "k": {"type": "integer", "required": False, "default": 5, "description": "Number of memories to retrieve"},
        },
    },
    "memory.store": {
        "description": "Store a new fact or memory in the knowledge base",
        "arguments": {
            "content": {"type": "string", "required": True, "description": "Content to store"},
            "category": {"type": "string", "required": False, "description": "Memory category"},
            "confidence": {"type": "number", "required": False, "default": 1.0, "description": "Confidence score (0.0-1.0)"},
        },
    },
    "memory.remember": {
        "description": "Alias for memory.store - store new information",
        "arguments": {
            "content": {"type": "string", "required": True, "description": "Content to remember"},
            "category": {"type": "string", "required": False, "description": "Memory category"},
        },
    },
    "memory.list_categories": {
        "description": "List all available memory categories",
        "arguments": {},
    },
    "memory.get_category": {
        "description": "Get all memories from a specific category",
        "arguments": {
            "name": {"type": "string", "required": True, "description": "Category name"},
        },
    },
    "planning.create_plan": {
        "description": "Create a new execution plan for a sub-goal",
        "arguments": {
            "goal": {"type": "string", "required": True, "description": "Goal to plan for"},
            "context": {"type": "object", "required": False, "description": "Additional context"},
            "tools": {"type": "array", "required": False, "description": "Available tools"},
            "max_steps": {"type": "integer", "required": False, "default": 10, "description": "Maximum steps"},
        },
    },
    "planning.get_status": {
        "description": "Get the execution status of a plan",
        "arguments": {
            "plan_id": {"type": "string", "required": True, "description": "Plan identifier"},
        },
    },
}

# MCP tool descriptions (category-level with capabilities)
MCP_TOOL_DESCRIPTIONS: dict[str, dict[str, Any]] = {
    "websearch": {
        "description": "Tavily web search and research capabilities",
        "tools": {
            "search": {
                "description": "Search the web for information",
                "arguments": {
                    "query": {"type": "string", "required": True, "description": "Search query"},
                    "max_results": {"type": "integer", "required": False, "default": 5, "description": "Maximum results to return"},
                },
            },
            "extract": {
                "description": "Extract content from a specific URL",
                "arguments": {
                    "url": {"type": "string", "required": True, "description": "URL to extract content from"},
                },
            },
        },
    },
    "voice": {
        "description": "ElevenLabs voice synthesis and text-to-speech",
        "tools": {
            "text_to_speech": {
                "description": "Convert text to speech audio",
                "arguments": {
                    "text": {"type": "string", "required": True, "description": "Text to convert"},
                    "voice_id": {"type": "string", "required": False, "description": "Voice identifier"},
                },
            },
            "list_voices": {
                "description": "List available voices",
                "arguments": {},
            },
        },
    },
    "calendar": {
        "description": "Google Calendar integration for scheduling",
        "tools": {
            "list_events": {
                "description": "List calendar events",
                "arguments": {
                    "start_time": {"type": "string", "required": False, "description": "Start time (ISO format)"},
                    "end_time": {"type": "string", "required": False, "description": "End time (ISO format)"},
                    "max_results": {"type": "integer", "required": False, "default": 10, "description": "Maximum events"},
                },
            },
            "create_event": {
                "description": "Create a new calendar event",
                "arguments": {
                    "title": {"type": "string", "required": True, "description": "Event title"},
                    "start_time": {"type": "string", "required": True, "description": "Start time (ISO format)"},
                    "end_time": {"type": "string", "required": True, "description": "End time (ISO format)"},
                    "description": {"type": "string", "required": False, "description": "Event description"},
                    "attendees": {"type": "array", "required": False, "description": "List of attendee emails"},
                },
            },
            "get_freebusy": {
                "description": "Check availability for a time range",
                "arguments": {
                    "start_time": {"type": "string", "required": True, "description": "Start time"},
                    "end_time": {"type": "string", "required": True, "description": "End time"},
                },
            },
        },
    },
    "communication": {
        "description": "Twilio SMS and voice communication",
        "tools": {
            "send_sms": {
                "description": "Send an SMS message",
                "arguments": {
                    "to": {"type": "string", "required": True, "description": "Recipient phone number"},
                    "body": {"type": "string", "required": True, "description": "Message content"},
                    "from": {"type": "string", "required": False, "description": "Sender phone number"},
                },
            },
            "make_call": {
                "description": "Initiate a phone call",
                "arguments": {
                    "to": {"type": "string", "required": True, "description": "Recipient phone number"},
                    "from": {"type": "string", "required": True, "description": "Caller phone number"},
                    "message": {"type": "string", "required": False, "description": "Message to speak"},
                },
            },
        },
    },
    "crm": {
        "description": "HubSpot CRM for contact and deal management",
        "tools": {
            "get_contact": {
                "description": "Get contact information",
                "arguments": {
                    "email": {"type": "string", "required": False, "description": "Contact email"},
                    "contact_id": {"type": "string", "required": False, "description": "Contact ID"},
                },
            },
            "create_contact": {
                "description": "Create a new contact",
                "arguments": {
                    "email": {"type": "string", "required": True, "description": "Contact email"},
                    "first_name": {"type": "string", "required": False, "description": "First name"},
                    "last_name": {"type": "string", "required": False, "description": "Last name"},
                    "company": {"type": "string", "required": False, "description": "Company name"},
                },
            },
            "get_deals": {
                "description": "Get deals for a contact",
                "arguments": {
                    "contact_id": {"type": "string", "required": False, "description": "Contact ID"},
                    "stage": {"type": "string", "required": False, "description": "Deal stage filter"},
                },
            },
        },
    },
}


def get_tool_description(tool_name: str) -> Optional[dict[str, Any]]:
    """Get the description and arguments for a tool.

    Args:
        tool_name: Full tool name (e.g., "websearch.search", "memory.recall").

    Returns:
        Tool description dictionary with 'description' and 'arguments' keys,
        or None if tool is not found.
    """
    # Check builtin tools first
    if tool_name in BUILTIN_TOOL_DESCRIPTIONS:
        return BUILTIN_TOOL_DESCRIPTIONS[tool_name]

    # Parse tool name for MCP tools
    parts = tool_name.split(".", 1)
    if len(parts) != 2:
        return None

    category, operation = parts

    # Check MCP tool categories
    if category in MCP_TOOL_DESCRIPTIONS:
        category_info = MCP_TOOL_DESCRIPTIONS[category]
        tools = category_info.get("tools", {})
        if operation in tools:
            return tools[operation]

    return None


def format_tool_for_prompt(tool_name: str, tool_info: dict[str, Any]) -> str:
    """Format a tool description for inclusion in planning prompt.

    Args:
        tool_name: Full tool name.
        tool_info: Tool description dictionary.

    Returns:
        Formatted string for the planning prompt.
    """
    lines = [f"  Tool: {tool_name}"]
    lines.append(f"    Description: {tool_info.get('description', 'No description')}")

    arguments = tool_info.get("arguments", {})
    if arguments:
        lines.append("    Arguments:")
        for arg_name, arg_info in arguments.items():
            arg_type = arg_info.get("type", "any")
            required = "required" if arg_info.get("required", False) else "optional"
            default = f", default={arg_info['default']}" if "default" in arg_info else ""
            desc = arg_info.get("description", "")
            lines.append(f"      - {arg_name} ({arg_type}, {required}{default}): {desc}")

    return "\n".join(lines)


# =============================================================================
# Planner
# =============================================================================


class Planner:
    """Creates execution plans from high-level goals.

    The Planner uses LLM-based planning to decompose goals into a sequence
    of executable steps. It validates the dependency DAG to ensure no
    circular dependencies and caches similar goals for efficiency.

    Features:
        - LLM-based goal decomposition
        - DAG validation (no circular dependencies)
        - Caching of similar goals (24-hour TTL)
        - Token tracking for budget management
        - Event emission for audit trails

    Attributes:
        event_store: Event store for audit trails.
        token_tracker: Token tracker for budget management.
        constraints: Default plan constraints.

    Example:
        >>> planner = Planner()
        >>> plan = await planner.create_plan(
        ...     goal="Research competitors and create summary",
        ...     context={"industry": "SaaS"},
        ...     tools=["websearch.search"],
        ... )
    """

    def __init__(
        self,
        event_store: Optional[EventStore] = None,
        token_tracker: Optional[TokenTracker] = None,
        constraints: Optional[PlanConstraints] = None,
        cache_ttl_hours: int = CACHE_TTL_HOURS,
    ) -> None:
        """Initialize the Planner.

        Args:
            event_store: Optional custom event store.
            token_tracker: Optional token tracker for budget.
            constraints: Optional default constraints.
            cache_ttl_hours: Cache TTL in hours.
        """
        self._event_store = event_store
        self._token_tracker = token_tracker
        self._constraints = constraints or PlanConstraints()
        self._cache_ttl_hours = cache_ttl_hours
        self._plan_cache: dict[str, CacheEntry] = {}
        self._settings = get_settings()

    def _get_cache_key(
        self,
        goal: str,
        tools: list[str],
        context_hash: str,
    ) -> str:
        """Generate a cache key from goal, tools, and context.

        Args:
            goal: The goal string.
            tools: List of available tools.
            context_hash: Hash of the context dictionary.

        Returns:
            Cache key string.
        """
        key_data = f"{goal}:{sorted(tools)}:{context_hash}"
        return hashlib.sha256(key_data.encode()).hexdigest()[:16]

    def _hash_context(self, context: dict[str, Any]) -> str:
        """Create a hash of the context dictionary.

        Args:
            context: Context dictionary.

        Returns:
            Hash string.
        """
        try:
            serialized = json.dumps(context, sort_keys=True, default=str)
            return hashlib.md5(serialized.encode()).hexdigest()[:8]
        except (TypeError, ValueError):
            return "no_hash"

    def _check_cache(self, cache_key: str) -> Optional[Plan]:
        """Check if a valid cached plan exists.

        Args:
            cache_key: The cache key to look up.

        Returns:
            Cached Plan if found and not expired, None otherwise.
        """
        if cache_key not in self._plan_cache:
            return None

        entry = self._plan_cache[cache_key]
        if entry.is_expired:
            del self._plan_cache[cache_key]
            logger.debug(f"Cache entry expired for key {cache_key}")
            return None

        entry.touch()
        logger.debug(f"Cache hit for key {cache_key} (hits={entry.hits})")

        # Return a copy with new IDs
        cached_plan = entry.plan
        return Plan(
            plan_id=generate_uuid(),
            goal=cached_plan.goal,
            steps=[
                PlanStep(
                    step_id=generate_uuid(),
                    description=step.description,
                    status=PlanStatus.PENDING,
                    dependencies=[],  # Reset dependencies for fresh execution
                    tool_calls=step.tool_calls,
                )
                for step in cached_plan.steps
            ],
            status=PlanStatus.PENDING,
        )

    def _store_in_cache(self, cache_key: str, plan: Plan) -> None:
        """Store a plan in the cache.

        Args:
            cache_key: The cache key.
            plan: The plan to cache.
        """
        self._plan_cache[cache_key] = CacheEntry(plan, self._cache_ttl_hours)
        logger.debug(f"Cached plan for key {cache_key}")

    async def _generate_plan_with_llm(
        self,
        goal: str,
        context: dict[str, Any],
        tools: list[str],
        max_steps: int,
        constraints: PlanConstraints,
        use_tool_aware: bool = True,
    ) -> Plan:
        """Generate a plan using LLM decomposition.

        This method uses the LLM to decompose a goal into executable steps.
        When use_tool_aware=True, it generates plans with full tool metadata
        including tool names, arguments, and executor types.

        Args:
            goal: The goal to plan for.
            context: Context information.
            tools: Available tools.
            max_steps: Maximum number of steps.
            constraints: Planning constraints.
            use_tool_aware: Whether to use tool-aware planning (default True).

        Returns:
            Generated Plan with tool metadata populated.
        """
        # Expand tool categories to full tool names
        expanded_tools = self._expand_tool_categories(tools) if tools else []

        if use_tool_aware and expanded_tools:
            # Use tool-aware planning with detailed prompts
            prompt = self._build_tool_aware_planning_prompt(
                goal=goal,
                context=context,
                available_tools=expanded_tools,
                constraints=constraints,
            )

            # In production, this would call the LLM API with the prompt
            # For now, use structured decomposition with tool awareness
            llm_response = await self._simulate_tool_aware_llm_response(
                goal, context, expanded_tools, max_steps, constraints
            )

            # Parse the LLM response to extract tool metadata
            steps = self._parse_tool_aware_plan(llm_response, expanded_tools)

            # Validate and enrich with executor type information
            steps = self._validate_and_enrich_tools(steps, expanded_tools)
        else:
            # Fall back to legacy planning (no tool metadata)
            prompt = self._build_planning_prompt(goal, context, tools, constraints)
            steps = await self._structured_decomposition(
                goal, context, tools, max_steps, constraints
            )

        # Track tokens (simulated for now)
        tokens_used = len(prompt) // 4 + len(str(steps)) // 4
        if self._token_tracker:
            self._token_tracker.record_usage(
                input_tokens=len(prompt) // 4,
                output_tokens=len(str(steps)) // 4,
            )

        # Calculate metadata
        metadata = self._calculate_metadata(steps, constraints)

        # Create the plan
        plan = Plan(
            plan_id=generate_uuid(),
            goal=goal,
            steps=steps,
            status=PlanStatus.PENDING,
        )

        return plan

    async def _simulate_tool_aware_llm_response(
        self,
        goal: str,
        context: dict[str, Any],
        available_tools: list[str],
        max_steps: int,
        constraints: PlanConstraints,
    ) -> str:
        """Simulate an LLM response for tool-aware planning.

        This method generates a structured plan response in the expected format.
        In production, this would be replaced by actual LLM API calls.

        Args:
            goal: The goal to plan for.
            context: Context information.
            available_tools: List of available tool names.
            max_steps: Maximum number of steps.
            constraints: Planning constraints.

        Returns:
            Simulated LLM response string in the expected plan format.
        """
        goal_lower = goal.lower()
        response_parts = []
        step_num = 1
        dependencies = []

        # Check for memory tools
        has_memory = any("memory" in t for t in available_tools)
        # Check for search tools
        has_search = any("search" in t or "websearch" in t for t in available_tools)
        # Check for calendar tools
        has_calendar = any("calendar" in t for t in available_tools)
        # Check for communication tools
        has_comm = any("communication" in t or "sms" in t for t in available_tools)
        # Check for CRM tools
        has_crm = any("crm" in t for t in available_tools)

        # Step 1: Memory recall if available and relevant
        if has_memory:
            response_parts.append(f"""Step {step_num}: Retrieve relevant context from memory
  Tool: memory.recall
  Args: {{"query": "{goal[:100]}", "k": 5}}
  Depends: []""")
            dependencies.append(step_num)
            step_num += 1

        # Step 2: Web search if relevant keywords and tool available
        if has_search and any(kw in goal_lower for kw in ["research", "find", "search", "news", "latest", "investigate"]):
            # Build search query from goal
            search_query = goal.replace("Research ", "").replace("Find ", "").replace("Search for ", "")[:100]
            response_parts.append(f"""Step {step_num}: Search for relevant information
  Tool: websearch.search
  Args: {{"query": "{search_query}", "max_results": 5}}
  Depends: []""")
            dependencies.append(step_num)
            step_num += 1

        # Step 3: CRM lookup if relevant
        if has_crm and any(kw in goal_lower for kw in ["lead", "contact", "customer", "client", "deal"]):
            # Extract potential contact info from context
            email = context.get("email", context.get("contact_email", ""))
            if email:
                response_parts.append(f"""Step {step_num}: Look up contact information in CRM
  Tool: crm.get_contact
  Args: {{"email": "{email}"}}
  Depends: []""")
            else:
                response_parts.append(f"""Step {step_num}: Search for contact in CRM
  Tool: crm.get_contact
  Args: {{}}
  Depends: []""")
            dependencies.append(step_num)
            step_num += 1

        # Step 4: Calendar check if scheduling related
        if has_calendar and any(kw in goal_lower for kw in ["schedule", "meeting", "appointment", "calendar", "book"]):
            response_parts.append(f"""Step {step_num}: Check calendar availability
  Tool: calendar.list_events
  Args: {{"max_results": 10}}
  Depends: []""")
            dependencies.append(step_num)
            step_num += 1

        # Step 5: Reasoning/Analysis step - depends on previous steps
        if dependencies:
            deps_str = ", ".join(str(d) for d in dependencies)
            response_parts.append(f"""Step {step_num}: Analyze gathered information and determine next actions
  Tool: reasoning
  Args: {{"task": "Analyze the results from previous steps and synthesize findings for: {goal[:80]}"}}
  Depends: [{deps_str}]""")
            step_num += 1

        # Step 6: Communication step if needed
        if has_comm and any(kw in goal_lower for kw in ["contact", "reach", "notify", "send", "message"]):
            response_parts.append(f"""Step {step_num}: Send communication
  Tool: communication.send_sms
  Args: {{"to": "{{{{context.phone}}}}", "body": "Following up on {goal[:50]}"}}
  Depends: [{step_num - 1}]""")
            step_num += 1

        # Step 7: Calendar creation if scheduling
        if has_calendar and any(kw in goal_lower for kw in ["schedule", "book", "create meeting"]):
            response_parts.append(f"""Step {step_num}: Schedule meeting
  Tool: calendar.create_event
  Args: {{"title": "{goal[:50]}", "start_time": "{{{{computed.start_time}}}}", "end_time": "{{{{computed.end_time}}}}"}}
  Depends: [{step_num - 1}]""")
            step_num += 1

        # Step 8: Memory store if we should remember results
        if has_memory and len(response_parts) > 1:
            response_parts.append(f"""Step {step_num}: Store results in memory for future reference
  Tool: memory.store
  Args: {{"content": "Completed task: {goal[:80]}", "category": "task_results"}}
  Depends: [{step_num - 1}]""")
            step_num += 1

        # Final reasoning step for summary
        if len(response_parts) > 0:
            final_deps = list(range(1, step_num))
            deps_str = ", ".join(str(d) for d in final_deps[-3:])  # Depend on last 3 steps
            response_parts.append(f"""Step {step_num}: Generate final response and summary
  Tool: reasoning
  Args: {{"task": "Summarize the results of all previous steps and provide a comprehensive response to: {goal[:80]}"}}
  Depends: [{deps_str}]""")

        # If no steps were generated, create a basic reasoning plan
        if not response_parts:
            response_parts.append(f"""Step 1: Analyze the goal and formulate approach
  Tool: reasoning
  Args: {{"task": "Analyze and plan approach for: {goal}"}}
  Depends: []""")
            response_parts.append(f"""Step 2: Execute the plan
  Tool: reasoning
  Args: {{"task": "Execute the planned approach and generate response for: {goal}"}}
  Depends: [1]""")

        # Limit to max_steps
        if len(response_parts) > max_steps:
            response_parts = response_parts[:max_steps]

        return "\n\n".join(response_parts)

    def _build_planning_prompt(
        self,
        goal: str,
        context: dict[str, Any],
        tools: list[str],
        constraints: PlanConstraints,
    ) -> str:
        """Build the planning prompt for LLM.

        Args:
            goal: The goal to plan for.
            context: Context information.
            tools: Available tools.
            constraints: Planning constraints.

        Returns:
            Prompt string.
        """
        context_str = json.dumps(context, indent=2) if context else "No context"
        tools_str = ", ".join(tools) if tools else "No specific tools"

        return f"""You are a planning agent. Decompose the following goal into a sequence of executable steps.

GOAL: {goal}

CONTEXT:
{context_str}

AVAILABLE TOOLS: {tools_str}

CONSTRAINTS:
- Maximum steps: {constraints.max_steps}
- Maximum parallel executions: {constraints.max_parallel}
- Maximum token budget: {constraints.max_tokens}

For each step, provide:
1. A clear description of what to do
2. Any tools required
3. Dependencies on other steps (by step number)

Format each step as:
Step N: [Description]
  Tools: [tool1, tool2] or None
  Depends: [step numbers] or None

Generate the plan:"""

    def _build_tool_aware_planning_prompt(
        self,
        goal: str,
        context: dict[str, Any],
        available_tools: list[str],
        constraints: PlanConstraints,
    ) -> str:
        """Build a tool-aware planning prompt for LLM.

        This method generates a comprehensive prompt that includes:
        - The goal to achieve
        - Available tools with their descriptions and argument schemas
        - Example plan format with tool specifications
        - Constraints for plan generation

        Args:
            goal: The goal to plan for.
            context: Context information.
            available_tools: List of available tool names.
            constraints: Planning constraints.

        Returns:
            Formatted prompt string for tool-aware planning.
        """
        # Build context section
        context_str = json.dumps(context, indent=2) if context else "No additional context"

        # Build tool descriptions section
        tool_sections = []
        tool_index = 1

        for tool_name in available_tools:
            tool_info = get_tool_description(tool_name)
            if tool_info:
                formatted = format_tool_for_prompt(tool_name, tool_info)
                tool_sections.append(f"Tool {tool_index}:\n{formatted}")
                tool_index += 1
            else:
                # Unknown tool, add basic entry
                tool_sections.append(f"Tool {tool_index}:\n  Tool: {tool_name}\n    Description: External tool (no schema available)")
                tool_index += 1

        # Add reasoning as a special "tool"
        tool_sections.append(f"""Tool {tool_index}:
  Tool: reasoning
    Description: Use reasoning to analyze, synthesize, or generate responses without external tools
    Arguments:
      - task (string, required): Description of the reasoning task to perform""")

        tools_str = "\n\n".join(tool_sections) if tool_sections else "No tools available"

        # Build the complete prompt
        return f"""You are a planning agent. Your task is to decompose the given goal into a sequence of executable steps.
Each step should specify exactly which tool to use and with what arguments.

GOAL: {goal}

CONTEXT:
{context_str}

AVAILABLE TOOLS:
{tools_str}

CONSTRAINTS:
- Maximum steps: {constraints.max_steps}
- Maximum parallel executions: {constraints.max_parallel}
- Maximum token budget: {constraints.max_tokens}
- Steps should be atomic and focused on a single action

PLAN FORMAT:
For each step, provide the following on separate lines:
Step N: [Brief description of what this step accomplishes]
  Tool: [exact tool name, e.g., "websearch.search" or "reasoning"]
  Args: {{"arg1": "value1", "arg2": "value2"}} (valid JSON object)
  Depends: [list of step numbers this depends on, e.g., [1, 2] or []]

EXAMPLE PLAN:
Step 1: Search for recent news about the topic
  Tool: websearch.search
  Args: {{"query": "latest news 2026", "max_results": 5}}
  Depends: []

Step 2: Recall relevant information from memory
  Tool: memory.recall
  Args: {{"query": "related context", "k": 3}}
  Depends: []

Step 3: Analyze and synthesize the gathered information
  Tool: reasoning
  Args: {{"task": "Analyze the search results and memories to create a comprehensive summary"}}
  Depends: [1, 2]

IMPORTANT:
- Use exact tool names from the AVAILABLE TOOLS section
- Args must be valid JSON with proper quoting
- For reasoning steps, use Tool: reasoning with Args containing a "task" key
- Dependencies are step numbers (integers), not step IDs
- Steps with no dependencies should have Depends: []

Now generate a plan for the goal above:"""

    def _parse_tool_aware_plan(
        self,
        llm_response: str,
        available_tools: list[str],
    ) -> list[PlanStep]:
        """Parse an LLM-generated tool-aware plan into PlanStep objects.

        This method extracts tool metadata from the structured LLM response,
        including tool names, arguments, and dependencies.

        Args:
            llm_response: The LLM-generated plan text.
            available_tools: List of available tool names for validation.

        Returns:
            List of PlanStep objects with populated tool metadata.
        """
        steps: list[PlanStep] = []
        step_id_map: dict[int, str] = {}  # Map step numbers to step IDs

        # Pattern to match step blocks
        # Matches "Step N:" followed by description and subsequent lines
        step_pattern = re.compile(
            r"Step\s+(\d+):\s*(.+?)(?=(?:Step\s+\d+:|$))",
            re.DOTALL | re.IGNORECASE
        )

        matches = step_pattern.findall(llm_response)

        for step_num_str, step_content in matches:
            step_num = int(step_num_str)
            step_id = generate_uuid()
            step_id_map[step_num] = step_id

            # Parse the step content
            lines = step_content.strip().split("\n")
            description = lines[0].strip() if lines else f"Step {step_num}"

            # Extract tool name
            tool_name = None
            tool_match = re.search(r"^\s*Tool:\s*(.+?)\s*$", step_content, re.MULTILINE | re.IGNORECASE)
            if tool_match:
                tool_name = tool_match.group(1).strip()
                # Clean up quotes if present
                tool_name = tool_name.strip('"\'')

            # Extract arguments
            tool_args = {}
            args_match = re.search(r"^\s*Args:\s*(\{.+?\})\s*$", step_content, re.MULTILINE | re.IGNORECASE)
            if args_match:
                args_str = args_match.group(1).strip()
                try:
                    tool_args = json.loads(args_str)
                except json.JSONDecodeError:
                    # Try to fix common JSON issues
                    try:
                        # Replace single quotes with double quotes
                        fixed_args = args_str.replace("'", '"')
                        tool_args = json.loads(fixed_args)
                    except json.JSONDecodeError:
                        logger.warning(f"Failed to parse args for step {step_num}: {args_str}")
                        tool_args = {}

            # Extract dependencies
            dependencies: list[int] = []
            depends_match = re.search(r"^\s*Depends:\s*\[([^\]]*)\]", step_content, re.MULTILINE | re.IGNORECASE)
            if depends_match:
                depends_str = depends_match.group(1).strip()
                if depends_str:
                    # Parse comma-separated integers
                    try:
                        dependencies = [int(d.strip()) for d in depends_str.split(",") if d.strip()]
                    except ValueError:
                        logger.warning(f"Failed to parse dependencies for step {step_num}: {depends_str}")
                        dependencies = []

            # Create the step
            step = PlanStep(
                step_id=step_id,
                description=description,
                dependencies=[],  # Will be resolved after all steps are created
                tool_calls=[tool_name] if tool_name and tool_name != "reasoning" else None,
            )

            # Store parsed metadata for later enrichment
            step._parsed_tool_name = tool_name  # type: ignore
            step._parsed_tool_args = tool_args  # type: ignore
            step._parsed_dependencies = dependencies  # type: ignore

            steps.append(step)

        # Resolve dependencies using step_id_map
        for step in steps:
            parsed_deps = getattr(step, "_parsed_dependencies", [])
            resolved_deps = []
            for dep_num in parsed_deps:
                if dep_num in step_id_map:
                    resolved_deps.append(step_id_map[dep_num])
                else:
                    logger.warning(f"Step {step.step_id} references unknown dependency: Step {dep_num}")
            step.dependencies = resolved_deps

        return steps

    def _validate_and_enrich_tools(
        self,
        steps: list[PlanStep],
        available_tools: list[str],
    ) -> list[PlanStep]:
        """Validate and enrich plan steps with tool executor type information.

        This method:
        - Validates that tool names exist in available_tools
        - Sets tool_executor_type based on tool category (builtin vs mcp)
        - Logs warnings for invalid tools
        - Falls back to reasoning for steps with invalid tools

        Args:
            steps: List of PlanStep objects to validate.
            available_tools: List of available tool names.

        Returns:
            List of validated and enriched PlanStep objects.
        """
        enriched_steps = []

        for step in steps:
            # Get parsed metadata
            tool_name = getattr(step, "_parsed_tool_name", None)
            tool_args = getattr(step, "_parsed_tool_args", {})

            # Clean up temporary attributes
            if hasattr(step, "_parsed_tool_name"):
                delattr(step, "_parsed_tool_name")
            if hasattr(step, "_parsed_tool_args"):
                delattr(step, "_parsed_tool_args")
            if hasattr(step, "_parsed_dependencies"):
                delattr(step, "_parsed_dependencies")

            # Handle reasoning steps
            if tool_name == "reasoning" or tool_name is None:
                # This is a reasoning step, no tool execution needed
                # Store reasoning task in result field temporarily
                reasoning_task = tool_args.get("task", step.description) if tool_args else step.description
                # We can use tool_calls field to indicate this is a reasoning step
                step.tool_calls = None
                step._reasoning_task = reasoning_task  # type: ignore
                step._step_type = StepType.REASONING  # type: ignore
                step._tool_executor_type = None  # type: ignore
                enriched_steps.append(step)
                continue

            # Validate tool exists
            tool_valid = False
            tool_executor_type = None

            # Check if it's a builtin tool
            if BuiltinToolExecutor.is_builtin_tool(tool_name):
                tool_valid = True
                tool_executor_type = "builtin"
            # Check if it's in available tools
            elif tool_name in available_tools:
                tool_valid = True
                tool_executor_type = "mcp"
            # Check if it's a category.operation format matching available tools
            else:
                parts = tool_name.split(".", 1)
                if len(parts) == 2:
                    category = parts[0]
                    # Check if category is in available tools
                    for avail_tool in available_tools:
                        if avail_tool.startswith(category + ".") or avail_tool == category:
                            tool_valid = True
                            tool_executor_type = "mcp"
                            break

            if tool_valid:
                # Set tool metadata on step
                step.tool_calls = [tool_name]
                step._tool_name = tool_name  # type: ignore
                step._tool_args = tool_args  # type: ignore
                step._step_type = StepType.TOOL_CALL  # type: ignore
                step._tool_executor_type = tool_executor_type  # type: ignore
                logger.debug(f"Step {step.step_id}: tool={tool_name}, executor={tool_executor_type}")
            else:
                # Invalid tool - log warning and mark as reasoning fallback
                logger.warning(
                    f"Step {step.step_id}: tool '{tool_name}' not found in available tools. "
                    f"Falling back to reasoning."
                )
                step.tool_calls = None
                step._reasoning_task = f"[Fallback] {step.description}"  # type: ignore
                step._step_type = StepType.REASONING  # type: ignore
                step._tool_executor_type = None  # type: ignore

            enriched_steps.append(step)

        return enriched_steps

    def _expand_tool_categories(self, tools: list[str]) -> list[str]:
        """Expand tool category names to include all available operations.

        For example, "websearch" expands to ["websearch.search", "websearch.extract"].

        Args:
            tools: List of tool names (may include categories).

        Returns:
            Expanded list of tool names with full operations.
        """
        expanded = []

        for tool in tools:
            if "." in tool:
                # Already a full tool name
                expanded.append(tool)
            elif tool in MCP_TOOL_DESCRIPTIONS:
                # It's a category, expand to all operations
                category_info = MCP_TOOL_DESCRIPTIONS[tool]
                for operation in category_info.get("tools", {}).keys():
                    expanded.append(f"{tool}.{operation}")
            elif tool in ("memory", "planning"):
                # Builtin category
                for builtin_tool in BUILTIN_TOOL_DESCRIPTIONS:
                    if builtin_tool.startswith(tool + "."):
                        expanded.append(builtin_tool)
            else:
                # Unknown, keep as is
                expanded.append(tool)

        return expanded

    async def _structured_decomposition(
        self,
        goal: str,
        context: dict[str, Any],
        tools: list[str],
        max_steps: int,
        constraints: PlanConstraints,
    ) -> list[PlanStep]:
        """Perform structured decomposition of goal into steps.

        This is a heuristic-based decomposition that serves as a fallback
        or baseline implementation. In production, this would be enhanced
        or replaced by LLM-based decomposition.

        Args:
            goal: The goal to decompose.
            context: Context information.
            tools: Available tools.
            max_steps: Maximum number of steps.
            constraints: Planning constraints.

        Returns:
            List of PlanSteps.
        """
        goal_lower = goal.lower()
        steps: list[PlanStep] = []

        # Always start with understanding/analysis step
        steps.append(
            PlanStep(
                step_id=generate_uuid(),
                description=f"Analyze the goal and gather initial context: {goal}",
                dependencies=[],
                tool_calls=["memory.recall"] if "memory.recall" in tools else None,
            )
        )

        # Add steps based on goal keywords
        if any(kw in goal_lower for kw in ["research", "find", "search", "investigate", "news", "latest"]):
            search_step = PlanStep(
                step_id=generate_uuid(),
                description=f"Search for: {goal}",
                dependencies=[steps[0].step_id],
                tool_calls=[t for t in tools if "search" in t.lower()] or None,
            )
            # Extract query from goal for websearch tool
            search_step._tool_args = {"query": goal}  # type: ignore
            steps.append(search_step)

        if any(kw in goal_lower for kw in ["qualify", "assess", "evaluate"]):
            steps.append(
                PlanStep(
                    step_id=generate_uuid(),
                    description="Evaluate and assess gathered information",
                    dependencies=[steps[-1].step_id] if len(steps) > 1 else [steps[0].step_id],
                    tool_calls=None,
                )
            )

        if any(kw in goal_lower for kw in ["contact", "reach", "communicate", "call", "email"]):
            steps.append(
                PlanStep(
                    step_id=generate_uuid(),
                    description="Prepare and execute communication",
                    dependencies=[steps[-1].step_id] if len(steps) > 1 else [steps[0].step_id],
                    tool_calls=[t for t in tools if any(c in t.lower() for c in ["email", "sms", "call"])] or None,
                )
            )

        if any(kw in goal_lower for kw in ["schedule", "book", "meeting", "calendar"]):
            steps.append(
                PlanStep(
                    step_id=generate_uuid(),
                    description="Schedule meeting or appointment",
                    dependencies=[steps[-1].step_id] if len(steps) > 1 else [steps[0].step_id],
                    tool_calls=[t for t in tools if "calendar" in t.lower()] or None,
                )
            )

        if any(kw in goal_lower for kw in ["report", "summary", "summarize", "create"]):
            steps.append(
                PlanStep(
                    step_id=generate_uuid(),
                    description="Generate summary or report",
                    dependencies=[steps[-1].step_id] if len(steps) > 1 else [steps[0].step_id],
                    tool_calls=None,
                )
            )

        # Add verification step if we have multiple steps
        if len(steps) > 2:
            steps.append(
                PlanStep(
                    step_id=generate_uuid(),
                    description="Verify and validate results",
                    dependencies=[steps[-1].step_id],
                    tool_calls=None,
                )
            )

        # Ensure we don't exceed max_steps
        if len(steps) > max_steps:
            steps = steps[:max_steps]

        # Ensure at least one step exists
        if not steps:
            steps.append(
                PlanStep(
                    step_id=generate_uuid(),
                    description=f"Execute goal: {goal}",
                    dependencies=[],
                    tool_calls=None,
                )
            )

        return steps

    def _calculate_metadata(
        self,
        steps: list[PlanStep],
        constraints: PlanConstraints,
    ) -> PlanMetadata:
        """Calculate plan metadata from steps.

        Args:
            steps: List of plan steps.
            constraints: Planning constraints.

        Returns:
            PlanMetadata object.
        """
        # Count step types
        tool_calls = sum(1 for s in steps if s.tool_calls)
        reasoning_steps = len(steps) - tool_calls

        # Estimate tokens (rough heuristic)
        estimated_tokens = sum(len(s.description) // 4 + 200 for s in steps)

        # Calculate complexity based on step count and dependencies
        max_depth = self._calculate_max_depth(steps)
        complexity = min(
            (len(steps) / constraints.max_steps * 0.5)
            + (max_depth / constraints.max_depth * 0.3)
            + (tool_calls / len(steps) * 0.2)
            if steps
            else 0.0,
            1.0,
        )

        # Calculate parallel groups
        parallel_groups = self._calculate_parallel_groups(steps)

        return PlanMetadata(
            complexity=round(complexity, 3),
            estimated_total_tokens=estimated_tokens,
            estimated_duration_seconds=len(steps) * 2.0,  # Rough estimate
            parallel_groups=parallel_groups,
            tool_calls=tool_calls,
            reasoning_steps=reasoning_steps,
        )

    def _calculate_max_depth(self, steps: list[PlanStep]) -> int:
        """Calculate the maximum dependency chain depth.

        Args:
            steps: List of plan steps.

        Returns:
            Maximum depth.
        """
        if not steps:
            return 0

        # Build step lookup
        step_map = {s.step_id: s for s in steps}
        depth_cache: dict[str, int] = {}

        def get_depth(step_id: str) -> int:
            if step_id in depth_cache:
                return depth_cache[step_id]

            step = step_map.get(step_id)
            if not step or not step.dependencies:
                depth_cache[step_id] = 0
                return 0

            max_dep_depth = 0
            for dep_id in step.dependencies:
                if dep_id in step_map:
                    max_dep_depth = max(max_dep_depth, get_depth(dep_id) + 1)

            depth_cache[step_id] = max_dep_depth
            return max_dep_depth

        return max(get_depth(s.step_id) for s in steps)

    def _calculate_parallel_groups(self, steps: list[PlanStep]) -> int:
        """Calculate number of parallelizable step groups.

        Args:
            steps: List of plan steps.

        Returns:
            Number of parallel groups.
        """
        if not steps:
            return 0

        # Group by dependency level
        levels: dict[int, list[str]] = {}
        step_map = {s.step_id: s for s in steps}
        level_cache: dict[str, int] = {}

        def get_level(step_id: str) -> int:
            if step_id in level_cache:
                return level_cache[step_id]

            step = step_map.get(step_id)
            if not step or not step.dependencies:
                level_cache[step_id] = 0
                return 0

            max_dep_level = 0
            for dep_id in step.dependencies:
                if dep_id in step_map:
                    max_dep_level = max(max_dep_level, get_level(dep_id) + 1)

            level_cache[step_id] = max_dep_level
            return max_dep_level

        for step in steps:
            level = get_level(step.step_id)
            if level not in levels:
                levels[level] = []
            levels[level].append(step.step_id)

        return len(levels)

    def validate_dag(self, plan: Plan) -> bool:
        """Validate that the plan has no circular dependencies.

        Uses Kahn's algorithm for topological sorting to detect cycles.

        Args:
            plan: The plan to validate.

        Returns:
            True if valid (no cycles).

        Raises:
            DAGValidationError: If circular dependencies are detected.
        """
        if not plan.steps:
            return True

        # Build adjacency list and in-degree count
        in_degree: dict[str, int] = {s.step_id: 0 for s in plan.steps}
        adjacency: dict[str, list[str]] = {s.step_id: [] for s in plan.steps}
        step_ids = set(in_degree.keys())

        for step in plan.steps:
            for dep_id in step.dependencies:
                if dep_id not in step_ids:
                    logger.warning(f"Unknown dependency {dep_id} in step {step.step_id}")
                    continue
                adjacency[dep_id].append(step.step_id)
                in_degree[step.step_id] += 1

        # Kahn's algorithm
        queue = [step_id for step_id, degree in in_degree.items() if degree == 0]
        visited_count = 0

        while queue:
            current = queue.pop(0)
            visited_count += 1

            for neighbor in adjacency[current]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        if visited_count != len(plan.steps):
            # Find steps involved in cycles
            cycle_steps = [
                step_id for step_id, degree in in_degree.items() if degree > 0
            ]
            raise DAGValidationError(
                f"Circular dependencies detected in plan: {cycle_steps}",
                cycle_steps=cycle_steps,
            )

        return True

    async def create_plan(
        self,
        goal: str,
        context: Optional[dict[str, Any]] = None,
        tools: Optional[list[str]] = None,
        max_steps: int = DEFAULT_MAX_STEPS,
        session_id: Optional[str] = None,
        constraints: Optional[PlanConstraints] = None,
        use_cache: bool = True,
    ) -> Plan:
        """Create an execution plan from a goal.

        Args:
            goal: High-level goal to achieve.
            context: Optional context information.
            tools: Optional list of available tool names.
            max_steps: Maximum number of plan steps.
            session_id: Optional session ID for event tracking.
            constraints: Optional planning constraints.
            use_cache: Whether to use/check cache.

        Returns:
            Plan object with steps, complexity, and estimated tokens.

        Raises:
            PlanningError: If plan generation fails.
            DAGValidationError: If plan has circular dependencies.

        Example:
            >>> plan = await planner.create_plan(
            ...     goal="Schedule meeting with John",
            ...     tools=["calendar.create_event"],
            ... )
        """
        context = context or {}
        tools = tools or []
        constraints = constraints or self._constraints

        # Update max_steps in constraints
        if max_steps != constraints.max_steps:
            constraints = PlanConstraints(
                max_steps=max_steps,
                max_parallel=constraints.max_parallel,
                allowed_tools=constraints.allowed_tools,
                forbidden_tools=constraints.forbidden_tools,
                max_tokens=constraints.max_tokens,
                prefer_parallel=constraints.prefer_parallel,
                max_depth=constraints.max_depth,
            )

        # Check cache
        context_hash = self._hash_context(context)
        cache_key = self._get_cache_key(goal, tools, context_hash)

        if use_cache:
            cached_plan = self._check_cache(cache_key)
            if cached_plan:
                logger.info(f"Using cached plan for goal: {goal[:50]}...")
                # Still emit event for cached plan
                if self._event_store and session_id:
                    event = create_plan_created_event(
                        session_id=session_id,
                        plan_id=cached_plan.plan_id,
                        goal=goal,
                        step_count=len(cached_plan.steps),
                        complexity=0.5,  # Cached complexity
                        estimated_tokens=500,  # Cached estimate
                    )
                    self._event_store.append(event)
                return cached_plan

        # Generate plan
        start_time = time.time()
        try:
            plan = await self._generate_plan_with_llm(
                goal=goal,
                context=context,
                tools=tools,
                max_steps=max_steps,
                constraints=constraints,
            )
        except Exception as e:
            raise PlanningError(f"Failed to generate plan: {e}", goal=goal)

        generation_time = time.time() - start_time
        logger.info(
            f"Generated plan for goal '{goal[:50]}...' "
            f"with {len(plan.steps)} steps in {generation_time:.2f}s"
        )

        # Validate DAG
        self.validate_dag(plan)

        # Calculate metadata
        metadata = self._calculate_metadata(plan.steps, constraints)

        # Store in cache
        if use_cache:
            self._store_in_cache(cache_key, plan)

        # Emit event
        if self._event_store and session_id:
            event = create_plan_created_event(
                session_id=session_id,
                plan_id=plan.plan_id,
                goal=goal,
                step_count=len(plan.steps),
                complexity=metadata.complexity,
                estimated_tokens=metadata.estimated_total_tokens,
            )
            self._event_store.append(event)

        return plan

    def clear_cache(self) -> int:
        """Clear the plan cache.

        Returns:
            Number of entries cleared.
        """
        count = len(self._plan_cache)
        self._plan_cache.clear()
        logger.info(f"Cleared {count} cached plans")
        return count

    def get_cache_stats(self) -> dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dictionary with cache statistics.
        """
        total_entries = len(self._plan_cache)
        total_hits = sum(e.hits for e in self._plan_cache.values())
        expired = sum(1 for e in self._plan_cache.values() if e.is_expired)

        return {
            "total_entries": total_entries,
            "total_hits": total_hits,
            "expired_entries": expired,
            "active_entries": total_entries - expired,
            "ttl_hours": self._cache_ttl_hours,
        }


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "Planner",
    "PlanningError",
    "DAGValidationError",
    "create_plan_created_event",
    # Tool description utilities
    "BUILTIN_TOOL_DESCRIPTIONS",
    "MCP_TOOL_DESCRIPTIONS",
    "get_tool_description",
    "format_tool_for_prompt",
]
