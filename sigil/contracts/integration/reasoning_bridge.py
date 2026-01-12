"""Reasoning to Contract awareness bridge for Sigil v2 Phase 6.

This module provides integration between the reasoning system (Phase 5) and
the contracts system (Phase 6), enabling:

- Contract-aware reasoning strategy selection
- Contract-informed prompt construction
- Reasoning strategy adjustment based on contract strictness
- Common error pattern integration into prompts

Key Components:
    - ContractAwareReasoningManager: Extended ReasoningManager with contract support
    - ContractReasoningContext: Context for contract-aware reasoning

Contract requirements influence reasoning:
- Strict contracts (FAIL strategy) trigger more thorough strategies
- Many deliverables increase effective complexity
- Complex validation rules require clearer prompts

Usage:
    >>> from sigil.contracts.integration.reasoning_bridge import (
    ...     ContractAwareReasoningManager,
    ... )
    >>> from sigil.reasoning.manager import ReasoningManager
    >>>
    >>> base_manager = ReasoningManager()
    >>> aware_manager = ContractAwareReasoningManager(base_manager)
    >>>
    >>> strategy = aware_manager.select_strategy_for_contract(
    ...     task="Qualify lead",
    ...     contract=lead_qualification_contract(),
    ...     base_complexity=0.6,
    ... )

Example:
    >>> manager = ContractAwareReasoningManager(base_reasoning_manager)
    >>> prompt = manager.build_contract_aware_prompt(
    ...     task="Score this lead",
    ...     contract=qualification_contract,
    ... )
    >>> print("OUTPUT REQUIREMENTS:" in prompt)
    True
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Optional, TYPE_CHECKING

from sigil.contracts.schema import Contract, FailureStrategy

if TYPE_CHECKING:
    from sigil.reasoning.manager import ReasoningManager
    from sigil.reasoning.strategies.base import StrategyResult


# =============================================================================
# Module Logger
# =============================================================================

logger = logging.getLogger(__name__)


# =============================================================================
# Constants
# =============================================================================

# Complexity adjustments for contract features
STRICT_CONTRACT_COMPLEXITY_BOOST = 0.2
MANY_DELIVERABLES_COMPLEXITY_BOOST = 0.1
COMPLEX_RULES_COMPLEXITY_BOOST = 0.1

# Thresholds
MANY_DELIVERABLES_THRESHOLD = 5
COMPLEX_RULES_THRESHOLD = 10


# =============================================================================
# Contract Reasoning Context
# =============================================================================


@dataclass
class ContractReasoningContext:
    """Context for contract-aware reasoning.

    Contains information about the contract that influences
    reasoning strategy selection and prompt construction.

    Attributes:
        contract: The contract being executed.
        common_errors: Frequent validation errors for this contract.
        examples: Successful output examples.
        success_rate: Historical success rate.
        adjusted_complexity: Complexity adjusted for contract requirements.
    """

    contract: Contract
    common_errors: list[dict[str, Any]] = field(default_factory=list)
    examples: list[dict[str, Any]] = field(default_factory=list)
    success_rate: float = 0.0
    adjusted_complexity: float = 0.5

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "contract_name": self.contract.name,
            "common_errors": self.common_errors,
            "examples_count": len(self.examples),
            "success_rate": self.success_rate,
            "adjusted_complexity": self.adjusted_complexity,
        }


# =============================================================================
# Contract Aware Reasoning Manager
# =============================================================================


class ContractAwareReasoningManager:
    """Extends ReasoningManager with contract awareness.

    Provides contract-informed reasoning by:
    1. Adjusting strategy selection based on contract strictness
    2. Building prompts that include contract requirements
    3. Incorporating common errors into prompts as warnings

    Contract requirements influence reasoning:
    - FAIL strategy contracts get complexity boost (+0.2)
    - Many deliverables (>5) get complexity boost (+0.1)
    - Complex validation rules (>10) get complexity boost (+0.1)

    Attributes:
        _base_manager: The underlying ReasoningManager.

    Example:
        >>> manager = ContractAwareReasoningManager(reasoning_manager)
        >>> strategy = manager.select_strategy_for_contract(
        ...     "Score lead", contract, 0.5
        ... )
        >>> # Higher effective complexity due to strict contract
        >>> strategy
        'tree_of_thoughts'
    """

    def __init__(self, base_manager: "ReasoningManager") -> None:
        """Initialize the ContractAwareReasoningManager.

        Args:
            base_manager: The base ReasoningManager to extend.
        """
        self._base_manager = base_manager

    def select_strategy_for_contract(
        self,
        task: str,
        contract: Optional[Contract],
        base_complexity: float,
    ) -> str:
        """Select strategy considering contract requirements.

        Adjusts the base complexity based on contract features:
        - Strict contracts (FAIL strategy): +0.2 complexity
        - Many deliverables (>5): +0.1 complexity
        - Complex validation rules (>10 total): +0.1 complexity

        Args:
            task: The reasoning task.
            contract: Optional contract for output.
            base_complexity: Base complexity from router.

        Returns:
            Strategy name.

        Example:
            >>> manager.select_strategy_for_contract(
            ...     "Analyze market",
            ...     strict_contract,
            ...     0.5,
            ... )
            'tree_of_thoughts'  # Boosted from 0.5 to 0.7
        """
        if not contract:
            return self._base_manager.select_strategy(base_complexity)

        # Calculate adjusted complexity
        adjusted_complexity = self._calculate_adjusted_complexity(
            base_complexity, contract
        )

        logger.debug(
            f"Strategy selection for contract '{contract.name}': "
            f"base={base_complexity:.2f}, adjusted={adjusted_complexity:.2f}"
        )

        return self._base_manager.select_strategy(adjusted_complexity)

    def _calculate_adjusted_complexity(
        self,
        base_complexity: float,
        contract: Contract,
    ) -> float:
        """Calculate adjusted complexity based on contract.

        Args:
            base_complexity: Base complexity score.
            contract: Contract to analyze.

        Returns:
            Adjusted complexity (capped at 1.0).
        """
        adjusted = base_complexity

        # Strict contracts need more thorough reasoning
        if contract.failure_strategy == FailureStrategy.FAIL:
            adjusted += STRICT_CONTRACT_COMPLEXITY_BOOST
            logger.debug(
                f"Contract '{contract.name}' has FAIL strategy, "
                f"boosting complexity by {STRICT_CONTRACT_COMPLEXITY_BOOST}"
            )

        # Many deliverables = more complex
        deliverable_count = len(contract.deliverables)
        if deliverable_count > MANY_DELIVERABLES_THRESHOLD:
            adjusted += MANY_DELIVERABLES_COMPLEXITY_BOOST
            logger.debug(
                f"Contract '{contract.name}' has {deliverable_count} "
                f"deliverables, boosting complexity"
            )

        # Complex validation rules = more complex
        total_rules = sum(
            len(d.validation_rules) for d in contract.deliverables
        )
        if total_rules > COMPLEX_RULES_THRESHOLD:
            adjusted += COMPLEX_RULES_COMPLEXITY_BOOST
            logger.debug(
                f"Contract '{contract.name}' has {total_rules} "
                f"validation rules, boosting complexity"
            )

        return min(adjusted, 1.0)

    def build_contract_aware_prompt(
        self,
        task: str,
        contract: Contract,
        context: Optional[ContractReasoningContext] = None,
    ) -> str:
        """Build a prompt that includes contract requirements.

        Constructs a prompt with:
        1. The original task
        2. Output format specification from contract
        3. Example output from contract deliverables
        4. Validation rules summary
        5. Common errors to avoid (if context provided)

        Args:
            task: Base task description.
            contract: Contract with requirements.
            context: Optional context with common errors/examples.

        Returns:
            Enhanced prompt with contract requirements.

        Example:
            >>> prompt = manager.build_contract_aware_prompt(
            ...     "Score this lead",
            ...     qualification_contract,
            ... )
            >>> "OUTPUT REQUIREMENTS:" in prompt
            True
        """
        # Build format specification
        format_spec = self._build_format_spec(contract)

        # Build examples
        examples = self._build_examples(contract, context)

        # Build validation rules
        validation_rules = self._build_validation_rules(contract)

        # Construct the enhanced prompt
        prompt_parts = [
            task,
            "",
            "OUTPUT REQUIREMENTS:",
            "Your response must be a valid JSON object with the following structure:",
            "",
            format_spec,
            "",
            "EXAMPLE OUTPUT:",
            examples,
        ]

        # Add validation rules if present
        if validation_rules:
            prompt_parts.extend([
                "",
                "VALIDATION RULES:",
                validation_rules,
            ])

        # Add common errors to avoid (if available)
        if context and context.common_errors:
            prompt_parts.extend([
                "",
                "COMMON MISTAKES TO AVOID:",
            ])
            for error_info in context.common_errors[:3]:
                prompt_parts.append(f"- {error_info.get('error', 'Unknown error')}")

        # Add final instruction
        prompt_parts.extend([
            "",
            "Return ONLY the JSON object, no explanatory text.",
        ])

        return "\n".join(prompt_parts)

    def _build_format_spec(self, contract: Contract) -> str:
        """Build format specification from contract.

        Args:
            contract: Contract to extract format from.

        Returns:
            Formatted specification string.
        """
        lines = []
        for d in contract.deliverables:
            required_marker = "*" if d.required else ""
            lines.append(
                f"  {d.name}{required_marker}: {d.type} - {d.description}"
            )

        lines.append("")
        lines.append("(* = required field)")

        return "\n".join(lines)

    def _build_examples(
        self,
        contract: Contract,
        context: Optional[ContractReasoningContext] = None,
    ) -> str:
        """Build examples from contract deliverables.

        Uses actual examples from context if available,
        otherwise uses deliverable examples from contract.

        Args:
            contract: Contract with deliverable examples.
            context: Optional context with real examples.

        Returns:
            JSON-formatted example string.
        """
        # Prefer real examples from context
        if context and context.examples:
            # Use the first real example
            return json.dumps(context.examples[0], indent=2)

        # Fall back to contract deliverable examples
        example = {}
        for d in contract.deliverables:
            if d.example is not None:
                example[d.name] = d.example

        return json.dumps(example, indent=2)

    def _build_validation_rules(self, contract: Contract) -> str:
        """Build validation rules summary.

        Args:
            contract: Contract with validation rules.

        Returns:
            Formatted validation rules string.
        """
        lines = []
        for d in contract.deliverables:
            if d.validation_rules:
                rules_str = ", ".join(d.validation_rules)
                lines.append(f"- {d.name}: {rules_str}")

        if not lines:
            return ""

        return "\n".join(lines)

    async def execute_with_contract_context(
        self,
        task: str,
        contract: Contract,
        context: Optional[ContractReasoningContext] = None,
        session_id: Optional[str] = None,
    ) -> "StrategyResult":
        """Execute reasoning with contract-aware configuration.

        Combines strategy selection, prompt enhancement, and execution
        in a single method.

        Args:
            task: The task to reason about.
            contract: Contract for output.
            context: Optional contract context.
            session_id: Optional session ID for events.

        Returns:
            StrategyResult from execution.

        Example:
            >>> result = await manager.execute_with_contract_context(
            ...     "Qualify this lead",
            ...     qualification_contract,
            ... )
        """
        # Calculate adjusted complexity
        base_complexity = 0.5  # Default - should come from router
        adjusted_complexity = self._calculate_adjusted_complexity(
            base_complexity, contract
        )

        # Build enhanced prompt
        enhanced_task = self.build_contract_aware_prompt(
            task, contract, context
        )

        # Execute with the base manager
        return await self._base_manager.execute(
            task=enhanced_task,
            context={
                "contract_name": contract.name,
                "deliverables": contract.get_deliverable_names(),
            },
            complexity=adjusted_complexity,
            session_id=session_id,
        )

    def get_strategy_recommendation(
        self,
        contract: Contract,
        base_complexity: float,
    ) -> dict[str, Any]:
        """Get strategy recommendation with explanation.

        Returns the recommended strategy along with an explanation
        of why it was selected.

        Args:
            contract: Contract being executed.
            base_complexity: Base complexity from router.

        Returns:
            Dictionary with strategy and explanation.

        Example:
            >>> rec = manager.get_strategy_recommendation(contract, 0.5)
            >>> print(rec["strategy"])
            'tree_of_thoughts'
            >>> print(rec["reason"])
            'Complexity boosted due to strict contract...'
        """
        adjusted_complexity = self._calculate_adjusted_complexity(
            base_complexity, contract
        )

        strategy = self._base_manager.select_strategy(adjusted_complexity)

        # Build explanation
        reasons = []
        if contract.failure_strategy == FailureStrategy.FAIL:
            reasons.append(
                f"Contract uses FAIL strategy (+{STRICT_CONTRACT_COMPLEXITY_BOOST} complexity)"
            )

        deliverable_count = len(contract.deliverables)
        if deliverable_count > MANY_DELIVERABLES_THRESHOLD:
            reasons.append(
                f"Contract has {deliverable_count} deliverables "
                f"(+{MANY_DELIVERABLES_COMPLEXITY_BOOST} complexity)"
            )

        total_rules = sum(
            len(d.validation_rules) for d in contract.deliverables
        )
        if total_rules > COMPLEX_RULES_THRESHOLD:
            reasons.append(
                f"Contract has {total_rules} validation rules "
                f"(+{COMPLEX_RULES_COMPLEXITY_BOOST} complexity)"
            )

        if not reasons:
            reasons.append("No contract-specific complexity adjustments")

        return {
            "strategy": strategy,
            "base_complexity": base_complexity,
            "adjusted_complexity": adjusted_complexity,
            "reason": "; ".join(reasons),
            "contract_name": contract.name,
        }

    def create_reasoning_context(
        self,
        contract: Contract,
        common_errors: Optional[list[dict[str, Any]]] = None,
        examples: Optional[list[dict[str, Any]]] = None,
        success_rate: float = 0.0,
        base_complexity: float = 0.5,
    ) -> ContractReasoningContext:
        """Create a ContractReasoningContext.

        Convenience method for creating context objects.

        Args:
            contract: The contract being executed.
            common_errors: Frequent validation errors.
            examples: Successful output examples.
            success_rate: Historical success rate.
            base_complexity: Base complexity from router.

        Returns:
            ContractReasoningContext instance.
        """
        adjusted = self._calculate_adjusted_complexity(
            base_complexity, contract
        )

        return ContractReasoningContext(
            contract=contract,
            common_errors=common_errors or [],
            examples=examples or [],
            success_rate=success_rate,
            adjusted_complexity=adjusted,
        )


# =============================================================================
# Prompt Enhancement Utilities
# =============================================================================


def enhance_prompt_with_contract(
    task: str,
    contract: Contract,
    common_errors: Optional[list[str]] = None,
) -> str:
    """Enhance a prompt with contract requirements.

    Standalone function for simple prompt enhancement without
    a full ContractAwareReasoningManager.

    Args:
        task: Original task description.
        contract: Contract with output requirements.
        common_errors: Optional list of common error messages.

    Returns:
        Enhanced prompt string.

    Example:
        >>> prompt = enhance_prompt_with_contract(
        ...     "Score this lead",
        ...     qualification_contract,
        ...     common_errors=["Missing budget field"],
        ... )
    """
    manager = ContractAwareReasoningManager(None)  # type: ignore

    # Create minimal context if errors provided
    context = None
    if common_errors:
        context = ContractReasoningContext(
            contract=contract,
            common_errors=[{"error": e} for e in common_errors],
        )

    return manager.build_contract_aware_prompt(task, contract, context)


def get_output_format_instruction(contract: Contract) -> str:
    """Get a simple output format instruction from contract.

    Generates a concise instruction about required output format.

    Args:
        contract: Contract to extract format from.

    Returns:
        Format instruction string.

    Example:
        >>> instruction = get_output_format_instruction(contract)
        >>> print(instruction)
        'Return JSON with: score (int, required), recommendation (str, required)'
    """
    parts = []
    for d in contract.deliverables:
        req = "required" if d.required else "optional"
        parts.append(f"{d.name} ({d.type}, {req})")

    return f"Return JSON with: {', '.join(parts)}"


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "ContractAwareReasoningManager",
    "ContractReasoningContext",
    "enhance_prompt_with_contract",
    "get_output_format_instruction",
    "STRICT_CONTRACT_COMPLEXITY_BOOST",
    "MANY_DELIVERABLES_COMPLEXITY_BOOST",
    "COMPLEX_RULES_COMPLEXITY_BOOST",
]
