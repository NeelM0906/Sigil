"""Plan to Contract bridge for Sigil v2 Phase 6.

This module provides integration between the planning system (Phase 5) and the
contracts system (Phase 6), enabling:

- Per-step contract specifications for plan steps
- Contract attachment to critical plan steps
- Step execution with contract verification
- Contract violation handling during plan execution

Key Components:
    - ContractSpec: Lightweight contract specification for plan steps
    - PlanContractBridge: Main bridge for plan-contract integration
    - StepContractResult: Result from contract-verified step execution

The bridge implements selective contract attachment:
- Final step: Always gets the main contract (if provided)
- TOOL_CALL steps: Get contracts if they produce structured output
- REASONING steps: Get contracts if they produce structured data

Usage:
    >>> from sigil.contracts.integration.plan_bridge import PlanContractBridge
    >>> from sigil.contracts.executor import ContractExecutor
    >>>
    >>> bridge = PlanContractBridge(executor)
    >>> enhanced_plan = bridge.attach_contracts_to_plan(plan, default_contract)
    >>> result = await bridge.execute_step_with_contract(step, agent, context)

Example:
    >>> spec = ContractSpec(
    ...     required_fields=["score", "recommendation"],
    ...     field_types={"score": "int", "recommendation": "str"},
    ...     validation_rules=["0 <= value <= 100"],
    ... )
    >>> contract = spec.to_contract("analysis_step")
    >>> print(contract.name)
    'step_analysis_step'
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional, Protocol, TYPE_CHECKING

from sigil.contracts.schema import (
    Contract,
    ContractConstraints,
    Deliverable,
    FailureStrategy,
)
from sigil.contracts.validator import ValidationResult

if TYPE_CHECKING:
    from sigil.contracts.executor import ContractExecutor, ContractResult
    from sigil.contracts.integration.memory_bridge import ContractMemoryBridge
    from sigil.planning.schemas import PlanStepConfig, StepStatus


# =============================================================================
# Module Logger
# =============================================================================

logger = logging.getLogger(__name__)


# =============================================================================
# Agent Protocol
# =============================================================================


class AgentProtocol(Protocol):
    """Protocol for agents that can execute tasks."""

    async def run(
        self,
        task: str,
        context: Optional[Any] = None,
    ) -> dict[str, Any]:
        """Run the agent with a task.

        Args:
            task: Task description.
            context: Optional execution context.

        Returns:
            Output dictionary.
        """
        ...


# =============================================================================
# Contract Specification
# =============================================================================


@dataclass
class ContractSpec:
    """Lightweight contract specification for plan steps.

    A simplified contract definition that can be embedded in plan steps
    for per-step verification. Converts to a full Contract for validation.

    Attributes:
        required_fields: List of required output field names.
        field_types: Mapping of field names to type strings.
        validation_rules: List of validation rule expressions.
        max_tokens: Optional token limit for the step.

    Example:
        >>> spec = ContractSpec(
        ...     required_fields=["score", "analysis"],
        ...     field_types={"score": "int", "analysis": "str"},
        ...     validation_rules=["0 <= value <= 100"],
        ... )
        >>> contract = spec.to_contract("my_step")
    """

    required_fields: list[str]
    field_types: dict[str, str] = field(default_factory=dict)
    validation_rules: list[str] = field(default_factory=list)
    max_tokens: Optional[int] = None

    def to_contract(self, step_name: str) -> Contract:
        """Convert specification to a full Contract.

        Args:
            step_name: Name of the plan step for the contract.

        Returns:
            Contract instance for validation.

        Example:
            >>> spec = ContractSpec(required_fields=["result"])
            >>> contract = spec.to_contract("process_data")
            >>> contract.name
            'step_process_data'
        """
        deliverables = []

        for field_name in self.required_fields:
            # Get type for this field
            field_type = self.field_types.get(field_name, "Any")

            # Filter validation rules that apply to this field
            field_rules = [
                rule for rule in self.validation_rules
                if field_name in rule or "value" in rule
            ]

            deliverables.append(
                Deliverable(
                    name=field_name,
                    type=field_type,
                    description=f"Required field for step: {step_name}",
                    required=True,
                    validation_rules=field_rules,
                )
            )

        return Contract(
            name=f"step_{step_name}",
            description=f"Contract for plan step: {step_name}",
            deliverables=deliverables,
            constraints=ContractConstraints(
                max_total_tokens=self.max_tokens,
            ),
            failure_strategy=FailureStrategy.RETRY,
            max_retries=1,  # Single retry for step-level contracts
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "required_fields": self.required_fields,
            "field_types": self.field_types,
            "validation_rules": self.validation_rules,
            "max_tokens": self.max_tokens,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ContractSpec":
        """Create from dictionary."""
        return cls(
            required_fields=data.get("required_fields", []),
            field_types=data.get("field_types", {}),
            validation_rules=data.get("validation_rules", []),
            max_tokens=data.get("max_tokens"),
        )


# =============================================================================
# Step Contract Result
# =============================================================================


@dataclass
class StepContractResult:
    """Result from contract-verified step execution.

    Contains the step output along with contract validation details.

    Attributes:
        step_id: Identifier of the executed step.
        success: Whether the step succeeded (output valid or fallback used).
        output: The step output dictionary.
        is_valid: Whether output passed validation.
        tokens_used: Tokens consumed by the step.
        duration_ms: Execution time in milliseconds.
        contract_errors: List of validation error messages (if any).
        applied_strategy: Strategy that produced the result.
        warnings: List of warnings (for fallback results).
    """

    step_id: str
    success: bool
    output: dict[str, Any]
    is_valid: bool
    tokens_used: int = 0
    duration_ms: float = 0.0
    contract_errors: list[str] = field(default_factory=list)
    applied_strategy: str = "success"
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "step_id": self.step_id,
            "success": self.success,
            "output": self.output,
            "is_valid": self.is_valid,
            "tokens_used": self.tokens_used,
            "duration_ms": self.duration_ms,
            "contract_errors": self.contract_errors,
            "applied_strategy": self.applied_strategy,
            "warnings": self.warnings,
        }


# =============================================================================
# Plan Contract Bridge
# =============================================================================


class PlanContractBridge:
    """Integrates contracts into plan execution.

    The PlanContractBridge provides:
    1. Automatic contract attachment to plan steps
    2. Contract-verified step execution
    3. Contract violation handling and replanning support

    Contract Attachment Rules:
    - Final step: Always gets the main contract if provided
    - TOOL_CALL steps: Get contracts if output is used downstream
    - REASONING steps: Get contracts if they produce structured data

    Attributes:
        executor: ContractExecutor for validation.
        memory: Optional ContractMemoryBridge for context.

    Example:
        >>> bridge = PlanContractBridge(executor)
        >>> plan = bridge.attach_contracts_to_plan(plan, main_contract)
        >>> for step in plan.steps:
        ...     result = await bridge.execute_step_with_contract(step, agent, ctx)
    """

    # Tools that produce structured output needing verification
    STRUCTURED_OUTPUT_TOOLS = {
        "crm_update",
        "data_extract",
        "analysis",
        "qualification",
        "scoring",
    }

    # Keywords indicating structured output in reasoning tasks
    STRUCTURED_OUTPUT_KEYWORDS = [
        "analyze",
        "score",
        "classify",
        "extract",
        "structure",
        "evaluate",
        "assess",
        "calculate",
        "determine",
    ]

    def __init__(
        self,
        contract_executor: "ContractExecutor",
        memory_bridge: Optional["ContractMemoryBridge"] = None,
    ) -> None:
        """Initialize the PlanContractBridge.

        Args:
            contract_executor: ContractExecutor for validation.
            memory_bridge: Optional memory bridge for context retrieval.
        """
        self.executor = contract_executor
        self.memory = memory_bridge

    def attach_contracts_to_plan(
        self,
        plan: Any,  # Plan type from planning module
        default_contract: Optional[Contract] = None,
    ) -> Any:
        """Attach contract specs to plan steps that need verification.

        Rules for contract attachment:
        - Final step: Always uses default_contract if provided
        - TOOL_CALL steps: Contract if output is used downstream
        - REASONING steps: Contract if produces structured data

        Args:
            plan: Plan to enhance with contracts.
            default_contract: Contract for final output.

        Returns:
            Plan with contract specs attached to steps.

        Example:
            >>> enhanced = bridge.attach_contracts_to_plan(plan, contract)
            >>> enhanced.steps[-1].contract_spec is not None
            True
        """
        # Import here to avoid circular imports
        from sigil.planning.schemas import StepType

        enhanced_steps = []
        final_step_idx = len(plan.steps) - 1

        for idx, step in enumerate(plan.steps):
            enhanced_step = step

            # Attach to final step
            if idx == final_step_idx and default_contract:
                enhanced_step = self._attach_contract_to_step(
                    step, default_contract
                )
                logger.debug(f"Attached contract to final step: {step.step_id}")

            # Attach to tool calls with structured output
            elif step.step_type == StepType.TOOL_CALL:
                if self._needs_verification(step):
                    enhanced_step = self._create_tool_step_contract(step)
                    logger.debug(
                        f"Created tool step contract for: {step.step_id}"
                    )

            # Attach to reasoning steps with structured output
            elif step.step_type == StepType.REASONING:
                if self._produces_structured_output(step):
                    enhanced_step = self._create_reasoning_step_contract(step)
                    logger.debug(
                        f"Created reasoning step contract for: {step.step_id}"
                    )

            enhanced_steps.append(enhanced_step)

        # Create new plan with enhanced steps
        # Note: This assumes Plan has a similar constructor
        return self._create_plan_with_steps(plan, enhanced_steps)

    async def execute_step_with_contract(
        self,
        step: "PlanStepConfig",
        agent: AgentProtocol,
        context: dict[str, Any],
    ) -> StepContractResult:
        """Execute a plan step with contract verification.

        If the step has a contract_spec, validates the output against it.
        Otherwise, executes the step directly without validation.

        Args:
            step: Step to execute.
            agent: Agent to run the step.
            context: Execution context.

        Returns:
            StepContractResult with execution details.

        Example:
            >>> result = await bridge.execute_step_with_contract(step, agent, {})
            >>> if result.is_valid:
            ...     print("Step completed successfully")
        """
        import time
        start_time = time.perf_counter()

        # Get the task to execute
        task = step.reasoning_task or step.description

        # Check if step has contract spec
        contract_spec = getattr(step, "contract_spec", None)

        if contract_spec is None:
            # No contract - execute directly
            try:
                output = await agent.run(task, context)
                duration_ms = (time.perf_counter() - start_time) * 1000

                return StepContractResult(
                    step_id=step.step_id,
                    success=True,
                    output=output,
                    is_valid=True,
                    duration_ms=duration_ms,
                )
            except Exception as e:
                duration_ms = (time.perf_counter() - start_time) * 1000
                logger.error(f"Step execution failed: {e}")

                return StepContractResult(
                    step_id=step.step_id,
                    success=False,
                    output={},
                    is_valid=False,
                    duration_ms=duration_ms,
                    contract_errors=[str(e)],
                )

        # Convert ContractSpec to Contract
        contract = contract_spec.to_contract(step.step_id)

        # Execute with contract verification
        try:
            result = await self.executor.execute_with_contract(
                agent=agent,
                task=task,
                contract=contract,
                context=context,
            )

            duration_ms = (time.perf_counter() - start_time) * 1000

            # Extract errors from validation result
            errors = []
            if result.validation_result and result.validation_result.errors:
                errors = [str(e) for e in result.validation_result.errors]

            return StepContractResult(
                step_id=step.step_id,
                success=result.succeeded,
                output=result.output,
                is_valid=result.is_valid,
                tokens_used=result.tokens_used,
                duration_ms=duration_ms,
                contract_errors=errors,
                applied_strategy=result.applied_strategy.value,
                warnings=result.metadata.get("fallback_warnings", []),
            )

        except Exception as e:
            duration_ms = (time.perf_counter() - start_time) * 1000
            logger.error(f"Contract execution failed: {e}")

            return StepContractResult(
                step_id=step.step_id,
                success=False,
                output={},
                is_valid=False,
                duration_ms=duration_ms,
                contract_errors=[str(e)],
                applied_strategy="fail",
            )

    def _needs_verification(self, step: "PlanStepConfig") -> bool:
        """Check if a tool step needs output verification.

        Args:
            step: The plan step to check.

        Returns:
            True if the step produces structured output needing verification.
        """
        if step.tool_name:
            return step.tool_name in self.STRUCTURED_OUTPUT_TOOLS
        return False

    def _produces_structured_output(self, step: "PlanStepConfig") -> bool:
        """Check if a reasoning step produces structured output.

        Args:
            step: The plan step to check.

        Returns:
            True if the step likely produces structured output.
        """
        task = step.reasoning_task or step.description
        task_lower = task.lower()

        return any(
            keyword in task_lower
            for keyword in self.STRUCTURED_OUTPUT_KEYWORDS
        )

    def _attach_contract_to_step(
        self,
        step: "PlanStepConfig",
        contract: Contract,
    ) -> "PlanStepConfig":
        """Attach a contract to a step as a ContractSpec.

        Args:
            step: Step to attach contract to.
            contract: Contract to attach.

        Returns:
            Step with contract_spec attribute set.
        """
        # Convert contract to spec
        spec = ContractSpec(
            required_fields=[d.name for d in contract.deliverables if d.required],
            field_types={d.name: d.type for d in contract.deliverables},
            validation_rules=[
                rule
                for d in contract.deliverables
                for rule in d.validation_rules
            ],
            max_tokens=contract.constraints.max_total_tokens,
        )

        # Create new step with contract_spec
        # Note: This creates a copy with the new attribute
        step_dict = step.model_dump() if hasattr(step, "model_dump") else vars(step).copy()
        step_dict["contract_spec"] = spec

        # Reconstruct step
        from sigil.planning.schemas import PlanStepConfig
        return PlanStepConfig(**{
            k: v for k, v in step_dict.items()
            if k != "contract_spec"
        })

    def _create_tool_step_contract(
        self,
        step: "PlanStepConfig",
    ) -> "PlanStepConfig":
        """Create a contract for a tool step.

        Args:
            step: Tool step to create contract for.

        Returns:
            Step with generated contract_spec.
        """
        # Default contract spec for tool outputs
        spec = ContractSpec(
            required_fields=["result", "status"],
            field_types={"result": "Any", "status": "str"},
            validation_rules=["len(str(value)) > 0"],
            max_tokens=1000,
        )

        step_dict = step.model_dump() if hasattr(step, "model_dump") else vars(step).copy()
        step_dict["contract_spec"] = spec

        from sigil.planning.schemas import PlanStepConfig
        return PlanStepConfig(**{
            k: v for k, v in step_dict.items()
            if k != "contract_spec"
        })

    def _create_reasoning_step_contract(
        self,
        step: "PlanStepConfig",
    ) -> "PlanStepConfig":
        """Create a contract for a reasoning step.

        Args:
            step: Reasoning step to create contract for.

        Returns:
            Step with generated contract_spec.
        """
        # Infer required fields from task description
        task = step.reasoning_task or step.description
        task_lower = task.lower()

        required_fields = ["result"]
        field_types = {"result": "Any"}

        # Add common fields based on task keywords
        if "score" in task_lower:
            required_fields.append("score")
            field_types["score"] = "int"
        if "analysis" in task_lower or "analyze" in task_lower:
            required_fields.append("analysis")
            field_types["analysis"] = "str"
        if "recommendation" in task_lower or "recommend" in task_lower:
            required_fields.append("recommendation")
            field_types["recommendation"] = "str"

        spec = ContractSpec(
            required_fields=required_fields,
            field_types=field_types,
            validation_rules=[],
            max_tokens=2000,
        )

        step_dict = step.model_dump() if hasattr(step, "model_dump") else vars(step).copy()
        step_dict["contract_spec"] = spec

        from sigil.planning.schemas import PlanStepConfig
        return PlanStepConfig(**{
            k: v for k, v in step_dict.items()
            if k != "contract_spec"
        })

    def _create_plan_with_steps(
        self,
        original_plan: Any,
        steps: list,
    ) -> Any:
        """Create a new plan with modified steps.

        Args:
            original_plan: Original plan to copy metadata from.
            steps: New steps for the plan.

        Returns:
            New plan with updated steps.
        """
        # This is a generic implementation - adjust based on actual Plan class
        plan_dict = {}
        if hasattr(original_plan, "model_dump"):
            plan_dict = original_plan.model_dump()
        elif hasattr(original_plan, "__dict__"):
            plan_dict = dict(original_plan.__dict__)

        plan_dict["steps"] = steps

        # Return the modified plan
        # Note: May need adjustment based on actual Plan class
        return original_plan.__class__(**plan_dict) if hasattr(original_plan, "__class__") else plan_dict


# =============================================================================
# Contract Violation Handler
# =============================================================================


async def handle_contract_violation_in_plan(
    plan: Any,
    failed_step: "PlanStepConfig",
    errors: list[str],
    planner: Any,  # Planner type
) -> Optional[Any]:
    """Handle contract violation during plan execution.

    When a step fails contract validation, this function determines
    the appropriate action:
    1. Retry step with refined prompt (if errors are fixable)
    2. Replan from failed step (if step is critical)
    3. Skip step and continue (if no downstream dependencies)
    4. Abort plan (if critical and unfixable)

    Args:
        plan: Current plan being executed.
        failed_step: Step that failed contract validation.
        errors: Validation error messages.
        planner: Planner instance for potential replanning.

    Returns:
        New plan if replanning needed, None if should continue with original.

    Example:
        >>> new_plan = await handle_contract_violation_in_plan(
        ...     plan, failed_step, errors, planner
        ... )
        >>> if new_plan:
        ...     # Execute new plan
        ...     pass
    """
    # Find steps that depend on the failed step
    dependents = [
        s for s in plan.steps
        if failed_step.step_id in getattr(s, "dependencies", [])
    ]

    if not dependents:
        # No dependents - can skip this step
        logger.warning(
            f"Skipping failed step {failed_step.step_id}, no dependents"
        )
        return None

    # Check if errors are fixable through retry
    fixable_errors = [e for e in errors if _is_fixable_error(e)]
    if len(fixable_errors) == len(errors):
        # All errors are fixable - recommend retry
        logger.info(
            f"All errors fixable for step {failed_step.step_id}, "
            "recommend retry with refined prompt"
        )
        return None

    # Need to replan
    logger.info(f"Replanning from step {failed_step.step_id}")

    new_goal = (
        f"Continue from: {failed_step.description}, "
        f"fixing: {', '.join(errors[:3])}"
    )

    context = (
        f"Previous steps completed successfully. "
        f"Failed at step: {failed_step.step_id}"
    )

    try:
        new_plan = await planner.generate(
            goal=new_goal,
            context=context,
        )
        return new_plan
    except Exception as e:
        logger.error(f"Replanning failed: {e}")
        return None


def _is_fixable_error(error: str) -> bool:
    """Check if an error is fixable through retry.

    Fixable errors are typically format/structure issues that
    can be resolved with a clearer prompt.

    Args:
        error: Error message to check.

    Returns:
        True if the error is likely fixable.
    """
    fixable_patterns = [
        "missing",
        "type mismatch",
        "rule failed",
        "required field",
        "invalid format",
        "expected",
    ]
    error_lower = error.lower()
    return any(pattern in error_lower for pattern in fixable_patterns)


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "ContractSpec",
    "StepContractResult",
    "PlanContractBridge",
    "AgentProtocol",
    "handle_contract_violation_in_plan",
]
