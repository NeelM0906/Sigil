"""Tests for the middleware architecture.

This module tests:
- BaseMiddleware default behavior
- MiddlewareChain execution order
- StepRegistry registration and ordering
- MiddlewareConfig per-request configuration
- PipelineRunner execution with middleware
"""

import pytest
import asyncio
from typing import Any, Optional
from unittest.mock import AsyncMock, MagicMock, patch

from sigil.core.middleware import (
    BaseMiddleware,
    MiddlewareChain,
    MiddlewareConfig,
    PipelineRunner,
    SigilMiddleware,
    StepDefinition,
    StepName,
    StepRegistry,
)


# =============================================================================
# Test Fixtures
# =============================================================================


class RecordingMiddleware(BaseMiddleware):
    """Middleware that records all calls for testing."""

    def __init__(self, name: str = "recording"):
        self._name = name
        self.pre_step_calls: list[str] = []
        self.post_step_calls: list[str] = []
        self.error_calls: list[tuple[str, Exception]] = []

    @property
    def name(self) -> str:
        return self._name

    async def pre_step(self, step_name: str, ctx: Any) -> Any:
        self.pre_step_calls.append(step_name)
        return ctx

    async def post_step(self, step_name: str, ctx: Any, result: Any) -> Any:
        self.post_step_calls.append(step_name)
        return result

    async def on_error(
        self, step_name: str, ctx: Any, error: Exception
    ) -> Optional[Any]:
        self.error_calls.append((step_name, error))
        raise error


class RecoveringMiddleware(BaseMiddleware):
    """Middleware that can recover from errors."""

    def __init__(self, recovery_value: Any = "recovered"):
        self.recovery_value = recovery_value
        self.recovered_steps: list[str] = []

    @property
    def name(self) -> str:
        return "RecoveringMiddleware"

    async def on_error(
        self, step_name: str, ctx: Any, error: Exception
    ) -> Optional[Any]:
        self.recovered_steps.append(step_name)
        return self.recovery_value


class ModifyingMiddleware(BaseMiddleware):
    """Middleware that modifies context."""

    def __init__(self, key: str, value: Any):
        self.key = key
        self.value = value

    @property
    def name(self) -> str:
        return f"ModifyingMiddleware({self.key})"

    async def pre_step(self, step_name: str, ctx: Any) -> Any:
        if isinstance(ctx, dict):
            ctx[self.key] = self.value
        elif hasattr(ctx, "__dict__"):
            setattr(ctx, self.key, self.value)
        return ctx


@pytest.fixture
def recording_middleware():
    """Create a recording middleware."""
    return RecordingMiddleware()


@pytest.fixture
def step_registry():
    """Create a step registry with test steps."""
    registry = StepRegistry()
    return registry


# =============================================================================
# BaseMiddleware Tests
# =============================================================================


class TestBaseMiddleware:
    """Tests for BaseMiddleware default behavior."""

    @pytest.mark.asyncio
    async def test_default_pre_step_passes_through(self):
        """Default pre_step should return context unchanged."""
        middleware = BaseMiddleware.__new__(BaseMiddleware)
        ctx = {"test": "value"}
        result = await middleware.pre_step("test_step", ctx)
        assert result is ctx

    @pytest.mark.asyncio
    async def test_default_post_step_passes_through(self):
        """Default post_step should return result unchanged."""
        middleware = BaseMiddleware.__new__(BaseMiddleware)
        result = {"output": "data"}
        returned = await middleware.post_step("test_step", {}, result)
        assert returned is result

    @pytest.mark.asyncio
    async def test_default_on_error_reraises(self):
        """Default on_error should re-raise the exception."""
        middleware = BaseMiddleware.__new__(BaseMiddleware)
        error = ValueError("test error")

        with pytest.raises(ValueError) as exc_info:
            await middleware.on_error("test_step", {}, error)

        assert exc_info.value is error


# =============================================================================
# MiddlewareChain Tests
# =============================================================================


class TestMiddlewareChain:
    """Tests for MiddlewareChain."""

    def test_add_middleware(self):
        """Test adding middleware to chain."""
        chain = MiddlewareChain()
        mw1 = RecordingMiddleware("mw1")
        mw2 = RecordingMiddleware("mw2")

        chain.add(mw1).add(mw2)

        assert len(chain) == 2
        assert list(chain) == [mw1, mw2]

    def test_insert_middleware(self):
        """Test inserting middleware at specific position."""
        chain = MiddlewareChain()
        mw1 = RecordingMiddleware("mw1")
        mw2 = RecordingMiddleware("mw2")
        mw3 = RecordingMiddleware("mw3")

        chain.add(mw1).add(mw3)
        chain.insert(1, mw2)

        assert [mw.name for mw in chain] == ["mw1", "mw2", "mw3"]

    def test_remove_middleware(self):
        """Test removing middleware by name."""
        chain = MiddlewareChain()
        mw1 = RecordingMiddleware("mw1")
        mw2 = RecordingMiddleware("mw2")

        chain.add(mw1).add(mw2)
        removed = chain.remove("mw1")

        assert removed is True
        assert len(chain) == 1
        assert list(chain)[0].name == "mw2"

    def test_remove_nonexistent_middleware(self):
        """Test removing middleware that doesn't exist."""
        chain = MiddlewareChain()
        removed = chain.remove("nonexistent")
        assert removed is False

    def test_clear_chain(self):
        """Test clearing all middleware."""
        chain = MiddlewareChain()
        chain.add(RecordingMiddleware()).add(RecordingMiddleware())
        chain.clear()

        assert len(chain) == 0

    @pytest.mark.asyncio
    async def test_pre_step_execution_order(self):
        """Test that pre_step runs in order (first to last)."""
        chain = MiddlewareChain()
        mw1 = RecordingMiddleware("mw1")
        mw2 = RecordingMiddleware("mw2")
        mw3 = RecordingMiddleware("mw3")

        chain.add(mw1).add(mw2).add(mw3)

        await chain.run_pre_step("test_step", {})

        # All should have recorded the step
        assert mw1.pre_step_calls == ["test_step"]
        assert mw2.pre_step_calls == ["test_step"]
        assert mw3.pre_step_calls == ["test_step"]

    @pytest.mark.asyncio
    async def test_post_step_execution_order(self):
        """Test that post_step runs in reverse order (last to first)."""
        chain = MiddlewareChain()
        order = []

        class OrderRecordingMiddleware(BaseMiddleware):
            def __init__(self, name: str):
                self._name = name

            @property
            def name(self) -> str:
                return self._name

            async def post_step(self, step_name: str, ctx: Any, result: Any) -> Any:
                order.append(self._name)
                return result

        chain.add(OrderRecordingMiddleware("mw1"))
        chain.add(OrderRecordingMiddleware("mw2"))
        chain.add(OrderRecordingMiddleware("mw3"))

        await chain.run_post_step("test_step", {}, None)

        # Should be reverse order
        assert order == ["mw3", "mw2", "mw1"]

    @pytest.mark.asyncio
    async def test_on_error_execution_order(self):
        """Test that on_error runs in reverse order."""
        chain = MiddlewareChain()
        order = []

        class OrderRecordingMiddleware(BaseMiddleware):
            def __init__(self, name: str):
                self._name = name

            @property
            def name(self) -> str:
                return self._name

            async def on_error(
                self, step_name: str, ctx: Any, error: Exception
            ) -> Optional[Any]:
                order.append(self._name)
                raise error

        chain.add(OrderRecordingMiddleware("mw1"))
        chain.add(OrderRecordingMiddleware("mw2"))
        chain.add(OrderRecordingMiddleware("mw3"))

        with pytest.raises(ValueError):
            await chain.run_on_error("test_step", {}, ValueError("test"))

        # Should be reverse order
        assert order == ["mw3", "mw2", "mw1"]

    @pytest.mark.asyncio
    async def test_on_error_recovery(self):
        """Test that middleware can recover from errors."""
        chain = MiddlewareChain()
        mw1 = RecordingMiddleware("mw1")
        mw2 = RecoveringMiddleware("recovered_value")

        chain.add(mw1).add(mw2)

        result = await chain.run_on_error("test_step", {}, ValueError("test"))

        assert result == "recovered_value"
        # mw1's on_error should not have been called (recovery stops chain)
        assert len(mw1.error_calls) == 0

    @pytest.mark.asyncio
    async def test_context_modification(self):
        """Test that middleware can modify context."""
        chain = MiddlewareChain()
        chain.add(ModifyingMiddleware("key1", "value1"))
        chain.add(ModifyingMiddleware("key2", "value2"))

        ctx = {}
        result = await chain.run_pre_step("test_step", ctx)

        assert result["key1"] == "value1"
        assert result["key2"] == "value2"


# =============================================================================
# StepRegistry Tests
# =============================================================================


class TestStepRegistry:
    """Tests for StepRegistry."""

    @pytest.mark.asyncio
    async def test_register_step(self):
        """Test basic step registration."""
        registry = StepRegistry()

        async def my_handler(ctx):
            pass

        registry.register("my_step", my_handler)

        assert "my_step" in registry
        assert len(registry) == 1

    @pytest.mark.asyncio
    async def test_register_with_ordering(self):
        """Test step registration with ordering."""
        registry = StepRegistry()

        async def handler(ctx):
            pass

        registry.register("step1", handler)
        registry.register("step3", handler)
        registry.register("step2", handler, after="step1")

        names = registry.step_names
        assert names == ["step1", "step2", "step3"]

    @pytest.mark.asyncio
    async def test_register_before(self):
        """Test step registration with 'before' ordering."""
        registry = StepRegistry()

        async def handler(ctx):
            pass

        registry.register("step1", handler)
        registry.register("step3", handler)
        registry.register("step2", handler, before="step3")

        names = registry.step_names
        assert names == ["step1", "step2", "step3"]

    def test_enable_disable_step(self):
        """Test enabling and disabling steps."""
        registry = StepRegistry()

        async def handler(ctx):
            pass

        registry.register("my_step", handler)
        assert registry.is_enabled("my_step")

        registry.disable("my_step")
        assert not registry.is_enabled("my_step")

        registry.enable("my_step")
        assert registry.is_enabled("my_step")

    def test_cannot_disable_required_step(self):
        """Test that required steps cannot be disabled."""
        registry = StepRegistry()

        async def handler(ctx):
            pass

        registry.register("required_step", handler, required=True)

        with pytest.raises(ValueError):
            registry.disable("required_step")

    def test_get_enabled_steps(self):
        """Test getting only enabled steps."""
        registry = StepRegistry()

        async def handler(ctx):
            pass

        registry.register("step1", handler, enabled=True)
        registry.register("step2", handler, enabled=False)
        registry.register("step3", handler, enabled=True)

        enabled = registry.get_enabled_steps()
        enabled_names = [s.name for s in enabled]

        assert enabled_names == ["step1", "step3"]

    def test_reorder_steps(self):
        """Test reordering steps."""
        registry = StepRegistry()

        async def handler(ctx):
            pass

        registry.register("a", handler)
        registry.register("b", handler)
        registry.register("c", handler)

        registry.reorder(["c", "a", "b"])

        assert registry.step_names == ["c", "a", "b"]

    def test_move_after(self):
        """Test moving a step after another."""
        registry = StepRegistry()

        async def handler(ctx):
            pass

        registry.register("a", handler)
        registry.register("b", handler)
        registry.register("c", handler)

        registry.move_after("a", "b")

        assert registry.step_names == ["b", "a", "c"]

    def test_move_before(self):
        """Test moving a step before another."""
        registry = StepRegistry()

        async def handler(ctx):
            pass

        registry.register("a", handler)
        registry.register("b", handler)
        registry.register("c", handler)

        registry.move_before("c", "b")

        assert registry.step_names == ["a", "c", "b"]

    def test_unregister_step(self):
        """Test unregistering a step."""
        registry = StepRegistry()

        async def handler(ctx):
            pass

        registry.register("my_step", handler)
        removed = registry.unregister("my_step")

        assert removed is True
        assert "my_step" not in registry

    def test_cannot_unregister_required_step(self):
        """Test that required steps cannot be unregistered."""
        registry = StepRegistry()

        async def handler(ctx):
            pass

        registry.register("required_step", handler, required=True)

        with pytest.raises(ValueError):
            registry.unregister("required_step")


# =============================================================================
# MiddlewareConfig Tests
# =============================================================================


class TestMiddlewareConfig:
    """Tests for MiddlewareConfig."""

    def test_disable_step(self):
        """Test disabling a step in config."""
        config = MiddlewareConfig()
        config.disable_step("plan")

        assert "plan" in config.disabled_steps

    def test_enable_step(self):
        """Test re-enabling a step in config."""
        config = MiddlewareConfig()
        config.disable_step("plan")
        config.enable_step("plan")

        assert "plan" not in config.disabled_steps

    def test_override_step(self):
        """Test overriding a step handler."""
        config = MiddlewareConfig()

        async def custom_handler(ctx):
            pass

        config.override_step("plan", custom_handler)

        assert "plan" in config.step_overrides
        assert config.step_overrides["plan"] is custom_handler

    def test_add_middleware(self):
        """Test adding middleware to config."""
        config = MiddlewareConfig()
        mw = RecordingMiddleware()

        config.add_middleware(mw)

        assert config.chain is not None
        assert len(config.chain) == 1

    def test_method_chaining(self):
        """Test that config methods support chaining."""
        config = MiddlewareConfig()
        mw = RecordingMiddleware()

        result = (
            config.disable_step("plan")
            .disable_step("validate")
            .add_middleware(mw)
        )

        assert result is config
        assert len(config.disabled_steps) == 2
        assert len(config.chain) == 1


# =============================================================================
# PipelineRunner Tests
# =============================================================================


class TestPipelineRunner:
    """Tests for PipelineRunner."""

    @pytest.mark.asyncio
    async def test_runs_all_enabled_steps(self):
        """Test that runner executes all enabled steps."""
        registry = StepRegistry()
        executed = []

        async def make_handler(name):
            async def handler(ctx):
                executed.append(name)

            return handler

        registry.register("step1", await make_handler("step1"))
        registry.register("step2", await make_handler("step2"))
        registry.register("step3", await make_handler("step3"))

        runner = PipelineRunner(registry)

        # Create a mock context
        ctx = MagicMock()
        ctx.request.context = {}

        await runner.run(ctx)

        assert executed == ["step1", "step2", "step3"]

    @pytest.mark.asyncio
    async def test_skips_disabled_steps(self):
        """Test that runner skips disabled steps."""
        registry = StepRegistry()
        executed = []

        async def make_handler(name):
            async def handler(ctx):
                executed.append(name)

            return handler

        registry.register("step1", await make_handler("step1"))
        registry.register("step2", await make_handler("step2"), enabled=False)
        registry.register("step3", await make_handler("step3"))

        runner = PipelineRunner(registry)

        ctx = MagicMock()
        ctx.request.context = {}

        await runner.run(ctx)

        assert executed == ["step1", "step3"]

    @pytest.mark.asyncio
    async def test_per_request_disabled_steps(self):
        """Test that per-request config can disable steps."""
        registry = StepRegistry()
        executed = []

        async def make_handler(name):
            async def handler(ctx):
                executed.append(name)

            return handler

        registry.register("step1", await make_handler("step1"))
        registry.register("step2", await make_handler("step2"))
        registry.register("step3", await make_handler("step3"))

        runner = PipelineRunner(registry)

        ctx = MagicMock()
        ctx.request.context = {}

        config = MiddlewareConfig()
        config.disable_step("step2")

        await runner.run(ctx, config)

        assert executed == ["step1", "step3"]

    @pytest.mark.asyncio
    async def test_per_request_step_override(self):
        """Test that per-request config can override step handlers."""
        registry = StepRegistry()
        executed = []

        async def original_handler(ctx):
            executed.append("original")

        async def override_handler(ctx):
            executed.append("override")

        registry.register("step1", original_handler)

        runner = PipelineRunner(registry)

        ctx = MagicMock()
        ctx.request.context = {}

        config = MiddlewareConfig()
        config.override_step("step1", override_handler)

        await runner.run(ctx, config)

        assert executed == ["override"]

    @pytest.mark.asyncio
    async def test_middleware_hooks_called(self):
        """Test that middleware hooks are called around steps."""
        registry = StepRegistry()

        async def handler(ctx):
            pass

        registry.register("test_step", handler)

        mw = RecordingMiddleware()
        chain = MiddlewareChain()
        chain.add(mw)

        runner = PipelineRunner(registry, chain)

        ctx = MagicMock()
        ctx.request.context = {}

        await runner.run(ctx)

        assert "test_step" in mw.pre_step_calls
        assert "test_step" in mw.post_step_calls

    @pytest.mark.asyncio
    async def test_error_recovery_via_middleware(self):
        """Test that middleware can recover from step errors."""
        registry = StepRegistry()

        async def failing_handler(ctx):
            raise ValueError("step failed")

        registry.register("failing_step", failing_handler)

        mw = RecoveringMiddleware("recovered")
        chain = MiddlewareChain()
        chain.add(mw)

        runner = PipelineRunner(registry, chain)

        ctx = MagicMock()
        ctx.request.context = {}

        # Should not raise - middleware recovered
        await runner.run(ctx)

        assert "failing_step" in mw.recovered_steps


# =============================================================================
# Integration Tests
# =============================================================================


class TestMiddlewareIntegration:
    """Integration tests for the middleware system."""

    @pytest.mark.asyncio
    async def test_full_pipeline_with_middleware(self):
        """Test a complete pipeline with multiple middleware."""
        registry = StepRegistry()
        results = []

        async def step1(ctx):
            results.append(("step1", ctx.get("value", 0)))

        async def step2(ctx):
            results.append(("step2", ctx.get("value", 0)))

        async def step3(ctx):
            results.append(("step3", ctx.get("value", 0)))

        registry.register("step1", step1)
        registry.register("step2", step2)
        registry.register("step3", step3)

        chain = MiddlewareChain()
        chain.add(ModifyingMiddleware("value", 42))
        chain.add(RecordingMiddleware("recorder"))

        runner = PipelineRunner(registry, chain)

        # Use a dict as context
        ctx = {"request": MagicMock()}
        ctx["request"].context = {}

        # Mock the pre_step to work with dict
        original_pre_step = chain.run_pre_step

        async def dict_aware_pre_step(step_name, context):
            for mw in chain.middlewares:
                context = await mw.pre_step(step_name, context)
            return context

        # Run with dict context
        steps = registry.get_enabled_steps()
        for step in steps:
            ctx = await dict_aware_pre_step(step.name, ctx)
            await step.handler(ctx)

        # Each step should have seen the modified value
        assert results == [("step1", 42), ("step2", 42), ("step3", 42)]


# =============================================================================
# StepName Enum Tests
# =============================================================================


class TestStepName:
    """Tests for StepName enum."""

    def test_standard_step_names(self):
        """Test that standard step names are defined."""
        assert StepName.ROUTE.value == "route"
        assert StepName.GROUND.value == "ground"
        assert StepName.PLAN.value == "plan"
        assert StepName.ASSEMBLE.value == "assemble"
        assert StepName.EXECUTE.value == "execute"
        assert StepName.VALIDATE.value == "validate"

    def test_step_name_string_comparison(self):
        """Test that StepName can be compared to strings."""
        assert StepName.ROUTE == "route"
        assert StepName.PLAN == "plan"
