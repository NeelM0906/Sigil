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
    ) -> Plan:
        """Generate a plan using LLM decomposition.

        This method uses the LLM to decompose a goal into executable steps.
        In production, this would call the actual LLM API. For now, we provide
        a structured decomposition based on goal analysis.

        Args:
            goal: The goal to plan for.
            context: Context information.
            tools: Available tools.
            max_steps: Maximum number of steps.
            constraints: Planning constraints.

        Returns:
            Generated Plan.
        """
        # Build the planning prompt
        prompt = self._build_planning_prompt(goal, context, tools, constraints)

        # In production, this would call the LLM
        # For now, generate a structured plan based on goal analysis
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
        if any(kw in goal_lower for kw in ["research", "find", "search", "investigate"]):
            steps.append(
                PlanStep(
                    step_id=generate_uuid(),
                    description="Search for relevant information",
                    dependencies=[steps[0].step_id],
                    tool_calls=[t for t in tools if "search" in t.lower()] or None,
                )
            )

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
]
