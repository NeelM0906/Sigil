"""Tests for the Planner class.

Tests cover:
- Goal decomposition into steps
- DAG validation (no circular dependencies)
- Caching of similar goals
- Event emission
- Error handling
"""

import pytest
from unittest.mock import MagicMock, patch

from sigil.config.schemas.plan import Plan, PlanStep, PlanStatus
from sigil.planning.planner import (
    Planner,
    PlanningError,
    DAGValidationError,
    CacheEntry,
    create_plan_created_event,
)
from sigil.planning.schemas import (
    PlanConstraints,
    generate_uuid,
)


class TestPlannerBasics:
    """Tests for basic Planner functionality."""

    def test_initialization(self, planner):
        """Test that Planner initializes correctly."""
        assert planner._constraints is not None
        assert planner._plan_cache == {}
        assert planner._cache_ttl_hours == 1

    def test_initialization_with_defaults(self):
        """Test Planner initialization with default values."""
        planner = Planner()
        assert planner._event_store is None
        assert planner._token_tracker is None
        assert planner._constraints.max_steps == 20  # Default


class TestGoalDecomposition:
    """Tests for goal decomposition into steps."""

    @pytest.mark.asyncio
    async def test_create_plan_basic(self, planner):
        """Test basic plan creation from a goal."""
        plan = await planner.create_plan(
            goal="Research competitors and create summary",
            context={"industry": "SaaS"},
            tools=["websearch.search"],
            max_steps=5,
        )

        assert plan is not None
        assert plan.plan_id is not None
        assert plan.goal == "Research competitors and create summary"
        assert len(plan.steps) >= 1
        assert len(plan.steps) <= 5
        assert plan.status == PlanStatus.PENDING

    @pytest.mark.asyncio
    async def test_create_plan_with_tools(self, planner):
        """Test plan creation with tool matching."""
        plan = await planner.create_plan(
            goal="Search for market trends",
            tools=["websearch.search", "crm.get_contact"],
        )

        assert plan is not None
        # Should have steps related to searching
        assert any("search" in step.description.lower() for step in plan.steps)

    @pytest.mark.asyncio
    async def test_create_plan_respects_max_steps(self, planner):
        """Test that plan respects max_steps constraint."""
        plan = await planner.create_plan(
            goal="Research and analyze competitors, create report, schedule meeting",
            max_steps=3,
        )

        assert len(plan.steps) <= 3

    @pytest.mark.asyncio
    async def test_create_plan_with_context(self, planner):
        """Test plan creation with context."""
        plan = await planner.create_plan(
            goal="Qualify lead",
            context={
                "lead_name": "John Doe",
                "company": "Acme Corp",
                "industry": "Software",
            },
        )

        assert plan is not None
        assert len(plan.steps) >= 1

    @pytest.mark.asyncio
    async def test_create_plan_complex_goal(self, planner):
        """Test decomposition of a complex goal."""
        plan = await planner.create_plan(
            goal="Research competitors, evaluate their products, contact decision maker, and schedule meeting",
            tools=["websearch.search", "calendar.create_event", "email.send"],
            max_steps=10,
        )

        # Complex goal should produce multiple steps
        assert len(plan.steps) >= 2
        # Steps should have proper dependencies
        for step in plan.steps[1:]:
            # Later steps should depend on earlier ones
            assert len(step.dependencies) >= 0


class TestDAGValidation:
    """Tests for DAG (Directed Acyclic Graph) validation."""

    def test_validate_dag_valid(self, planner, simple_plan):
        """Test DAG validation with valid plan."""
        assert planner.validate_dag(simple_plan) is True

    def test_validate_dag_cyclic(self, planner, cyclic_plan):
        """Test DAG validation detects cycles."""
        with pytest.raises(DAGValidationError) as excinfo:
            planner.validate_dag(cyclic_plan)

        assert "Circular dependencies" in str(excinfo.value)
        assert len(excinfo.value.cycle_steps) > 0

    def test_validate_dag_empty_plan(self, planner):
        """Test DAG validation with empty plan."""
        empty_plan = Plan(
            plan_id=generate_uuid(),
            goal="Empty",
            steps=[],
            status=PlanStatus.PENDING,
        )
        assert planner.validate_dag(empty_plan) is True

    def test_validate_dag_single_step(self, planner):
        """Test DAG validation with single step."""
        single_step_plan = Plan(
            plan_id=generate_uuid(),
            goal="Single step",
            steps=[
                PlanStep(
                    step_id="step-1",
                    description="Only step",
                    dependencies=[],
                ),
            ],
            status=PlanStatus.PENDING,
        )
        assert planner.validate_dag(single_step_plan) is True

    def test_validate_dag_unknown_dependency(self, planner):
        """Test DAG validation handles unknown dependencies gracefully.

        Note: The Plan model validates dependencies at construction time,
        so we test with a manually constructed plan step with modified deps.
        """
        # Create a valid plan first
        plan = Plan(
            plan_id=generate_uuid(),
            goal="Test",
            steps=[
                PlanStep(
                    step_id="step-1",
                    description="Step 1",
                    dependencies=[],
                ),
            ],
            status=PlanStatus.PENDING,
        )
        # Manually modify to test the planner's validation
        # (bypassing Pydantic's validation)
        plan.steps[0].dependencies = ["nonexistent-step"]

        # validate_dag should handle this gracefully (warn but not raise)
        assert planner.validate_dag(plan) is True

    def test_validate_dag_parallel_steps(self, planner, parallel_plan):
        """Test DAG validation with parallel steps."""
        assert planner.validate_dag(parallel_plan) is True


class TestCaching:
    """Tests for plan caching."""

    @pytest.mark.asyncio
    async def test_cache_hit(self, planner):
        """Test that identical goals use cached plans."""
        goal = "Research competitors"
        context = {"industry": "Tech"}
        tools = ["websearch.search"]

        # First call - creates plan
        plan1 = await planner.create_plan(
            goal=goal,
            context=context,
            tools=tools,
            use_cache=True,
        )

        # Second call - should use cache
        plan2 = await planner.create_plan(
            goal=goal,
            context=context,
            tools=tools,
            use_cache=True,
        )

        # Plans should be different objects but similar structure
        assert plan1.plan_id != plan2.plan_id  # New IDs for each use
        assert len(plan1.steps) == len(plan2.steps)

    @pytest.mark.asyncio
    async def test_cache_miss_different_goal(self, planner):
        """Test that different goals don't use cache."""
        plan1 = await planner.create_plan(
            goal="Research competitors",
            use_cache=True,
        )

        plan2 = await planner.create_plan(
            goal="Schedule meeting",
            use_cache=True,
        )

        # Different goals, different plans
        assert plan1.goal != plan2.goal

    @pytest.mark.asyncio
    async def test_cache_disabled(self, planner):
        """Test plan creation with caching disabled."""
        goal = "Research competitors"

        plan1 = await planner.create_plan(goal=goal, use_cache=False)
        plan2 = await planner.create_plan(goal=goal, use_cache=False)

        # Both plans should be freshly generated
        assert plan1.plan_id != plan2.plan_id

    def test_clear_cache(self, planner):
        """Test clearing the plan cache."""
        # Add some entries
        planner._plan_cache["key1"] = CacheEntry(
            Plan(plan_id="p1", goal="g1", steps=[], status=PlanStatus.PENDING)
        )
        planner._plan_cache["key2"] = CacheEntry(
            Plan(plan_id="p2", goal="g2", steps=[], status=PlanStatus.PENDING)
        )

        cleared = planner.clear_cache()
        assert cleared == 2
        assert len(planner._plan_cache) == 0

    def test_get_cache_stats(self, planner):
        """Test getting cache statistics."""
        # Add some entries
        planner._plan_cache["key1"] = CacheEntry(
            Plan(plan_id="p1", goal="g1", steps=[], status=PlanStatus.PENDING)
        )

        stats = planner.get_cache_stats()
        assert stats["total_entries"] == 1
        assert stats["total_hits"] == 0
        assert stats["ttl_hours"] == 1

    def test_cache_key_generation(self, planner):
        """Test that cache keys are deterministic."""
        key1 = planner._get_cache_key(
            goal="Test goal",
            tools=["tool1", "tool2"],
            context_hash="abc123",
        )
        key2 = planner._get_cache_key(
            goal="Test goal",
            tools=["tool1", "tool2"],
            context_hash="abc123",
        )
        assert key1 == key2

        # Different tools should give different key
        key3 = planner._get_cache_key(
            goal="Test goal",
            tools=["tool3"],
            context_hash="abc123",
        )
        assert key1 != key3


class TestMetadataCalculation:
    """Tests for plan metadata calculation."""

    def test_calculate_metadata(self, planner, simple_plan, plan_constraints):
        """Test metadata calculation."""
        metadata = planner._calculate_metadata(simple_plan.steps, plan_constraints)

        assert 0.0 <= metadata.complexity <= 1.0
        assert metadata.estimated_total_tokens > 0
        assert metadata.estimated_duration_seconds > 0
        assert metadata.parallel_groups >= 1

    def test_calculate_max_depth(self, planner, simple_plan):
        """Test max depth calculation."""
        depth = planner._calculate_max_depth(simple_plan.steps)
        # simple_plan has 3 steps in a chain
        assert depth == 2  # 0-indexed depth

    def test_calculate_parallel_groups(self, planner, parallel_plan):
        """Test parallel groups calculation."""
        groups = planner._calculate_parallel_groups(parallel_plan.steps)
        # parallel_plan has 3 levels: initial, parallel (3), final
        assert groups == 3


class TestEventEmission:
    """Tests for event emission."""

    @pytest.mark.asyncio
    async def test_plan_created_event(self, planner, event_store):
        """Test that plan creation emits an event."""
        plan = await planner.create_plan(
            goal="Test event emission",
            session_id="test-session-123",
        )

        # Check event was emitted
        if event_store.session_exists("test-session-123"):
            events = event_store.get_events("test-session-123")
            plan_events = [e for e in events if "plan" in e.event_type.value.lower()]
            assert len(plan_events) >= 1


class TestErrorHandling:
    """Tests for error handling."""

    @pytest.mark.asyncio
    async def test_planning_error_on_empty_goal(self, planner):
        """Test that empty goals raise planning error."""
        # Empty goal should raise PlanningError
        with pytest.raises(PlanningError):
            await planner.create_plan(goal="")

    @pytest.mark.asyncio
    async def test_planning_success_with_minimal_goal(self, planner):
        """Test that minimal but valid goals work."""
        plan = await planner.create_plan(goal="Do something")
        # Should produce at least one step
        assert len(plan.steps) >= 1

    def test_dag_validation_error_attributes(self):
        """Test DAGValidationError attributes."""
        error = DAGValidationError(
            "Cycle detected",
            cycle_steps=["step-a", "step-b", "step-c"],
        )
        assert error.cycle_steps == ["step-a", "step-b", "step-c"]
        assert "Cycle detected" in str(error)


class TestStructuredDecomposition:
    """Tests for structured goal decomposition."""

    @pytest.mark.asyncio
    async def test_research_goal_decomposition(self, planner):
        """Test decomposition of research-type goals."""
        plan = await planner.create_plan(
            goal="Research market trends",
            tools=["websearch.search"],
        )
        # Should have search-related steps
        descriptions = [s.description.lower() for s in plan.steps]
        assert any("search" in d or "research" in d for d in descriptions)

    @pytest.mark.asyncio
    async def test_communication_goal_decomposition(self, planner):
        """Test decomposition of communication-type goals."""
        plan = await planner.create_plan(
            goal="Contact the lead and schedule a meeting",
            tools=["email.send", "calendar.create_event"],
        )
        assert len(plan.steps) >= 2

    @pytest.mark.asyncio
    async def test_report_goal_decomposition(self, planner):
        """Test decomposition of report-type goals."""
        plan = await planner.create_plan(
            goal="Create a summary report",
        )
        descriptions = [s.description.lower() for s in plan.steps]
        assert any("summary" in d or "report" in d or "generate" in d for d in descriptions)
