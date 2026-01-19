"""Router to Contract selection bridge for Sigil v2 Phase 6.

This module provides integration between the routing layer (Phase 3) and the
contracts system (Phase 6), enabling automatic contract selection based on
routing decisions.

Key Components:
    - ContractSelector: Selects appropriate contracts based on RouteDecision
    - INTENT_CONTRACT_MAP: Maps intents to contract templates
    - STRATUM_CONTRACT_MAP: Maps ACTi strata to contract templates

The bridge implements complexity-based contract adjustment, where higher
complexity tasks receive more lenient constraints (more retries, higher
token budgets) while simpler tasks use stricter constraints.

Usage:
    >>> from sigil.contracts.integration.router_bridge import ContractSelector
    >>> from sigil.routing.router import RouteDecision, Intent
    >>>
    >>> selector = ContractSelector()
    >>> decision = RouteDecision(
    ...     intent=Intent.RUN_AGENT,
    ...     confidence=0.9,
    ...     complexity=0.75,
    ...     handler_name="executor",
    ...     use_contracts=True,
    ... )
    >>> contract = selector.select_for_route(decision, stratum="RAI")
    >>> print(contract.name)
    'lead_qualification'

Example:
    >>> selector = ContractSelector()
    >>> # High complexity gets more retries
    >>> contract_high = selector.select_for_route(decision_high_complexity)
    >>> contract_high.max_retries
    3
    >>> # Low complexity gets fewer retries
    >>> contract_low = selector.select_for_route(decision_low_complexity)
    >>> contract_low.max_retries
    1
"""

from __future__ import annotations

import logging
from typing import Optional

from sigil.routing.router import RouteDecision, Intent
from sigil.contracts.schema import Contract, FailureStrategy
from sigil.contracts.templates.acti import CONTRACT_TEMPLATES, get_template


# =============================================================================
# Module Logger
# =============================================================================

logger = logging.getLogger(__name__)


# =============================================================================
# Intent to Contract Mappings
# =============================================================================

# Maps intents to contract template names.
# None indicates no default contract for that intent - contracts
# may still be selected based on stratum for RUN_AGENT intent.
INTENT_CONTRACT_MAP: dict[Intent, Optional[str]] = {
    Intent.CREATE_AGENT: None,  # Future: "agent_config" template
    Intent.RUN_AGENT: None,  # Dynamic based on agent stratum
    Intent.QUERY_MEMORY: None,  # Memory queries don't need contracts
    Intent.MODIFY_AGENT: None,  # Future: "agent_config" template
    Intent.SYSTEM_COMMAND: None,  # System commands are simple
    Intent.GENERAL_CHAT: None,  # Chat doesn't need contracts
}

# Maps ACTi strata to contract template names.
# Each stratum has a specific contract template optimized for its use case:
# - RTI: Research and truth verification
# - RAI: Lead qualification and readiness
# - ZACS: Action and conversion (appointments)
# - EEI: Economic and ecosystem analysis
# - IGE: Integrity and governance (compliance)
STRATUM_CONTRACT_MAP: dict[str, str] = {
    "RTI": "research_report",
    "RAI": "lead_qualification",
    "ZACS": "appointment_booking",
    "EEI": "market_analysis",
    "IGE": "compliance_check",
}


# =============================================================================
# Complexity-Based Selection Rules
# =============================================================================

# Complexity ranges for contract behavior
COMPLEXITY_NO_CONTRACT = 0.5  # Below this: no contract needed
COMPLEXITY_OPTIONAL_CONTRACT = 0.7  # Below this: contract with fallback
COMPLEXITY_REQUIRED_CONTRACT = 0.9  # Below this: contract with retry
# Above 0.9: strict contract with fail strategy


# =============================================================================
# Contract Selector
# =============================================================================


class ContractSelector:
    """Selects appropriate contracts based on routing decisions.

    The ContractSelector bridges the routing layer with the contracts system,
    providing automatic contract selection based on:

    1. Intent type (CREATE_AGENT, RUN_AGENT, etc.)
    2. ACTi stratum (RTI, RAI, ZACS, EEI, IGE)
    3. Task complexity (adjusts constraints)

    Contract adjustment rules:
    - Complexity >= 0.8: +1 retry, 50% more tokens (very complex)
    - Complexity <= 0.5: -1 retry (simple tasks)

    Attributes:
        _templates: Reference to CONTRACT_TEMPLATES registry.

    Example:
        >>> selector = ContractSelector()
        >>> contract = selector.select_for_route(decision, stratum="RAI")
        >>> print(contract.name, contract.max_retries)
        'lead_qualification' 2
    """

    def __init__(self) -> None:
        """Initialize the ContractSelector."""
        self._templates = CONTRACT_TEMPLATES

    def select_for_route(
        self,
        decision: RouteDecision,
        stratum: Optional[str] = None,
    ) -> Optional[Contract]:
        """Select a contract based on route decision.

        Selection logic:
        1. If use_contracts is False, return None
        2. Check intent-based mapping for direct contract
        3. For RUN_AGENT intent with stratum, use stratum mapping
        4. Adjust contract based on complexity

        Args:
            decision: The routing decision from Router.
            stratum: Optional ACTi stratum for RUN_AGENT intent.

        Returns:
            Contract if applicable, None if no contract needed.

        Example:
            >>> selector = ContractSelector()
            >>> decision = RouteDecision(
            ...     intent=Intent.RUN_AGENT,
            ...     confidence=0.9,
            ...     complexity=0.75,
            ...     handler_name="executor",
            ...     use_contracts=True,
            ... )
            >>> contract = selector.select_for_route(decision, stratum="RAI")
            >>> contract.name
            'lead_qualification'
        """
        # Check if contracts are enabled
        if not decision.use_contracts:
            logger.debug("Contracts disabled for this route decision")
            return None

        # Get contract name from intent mapping
        contract_name = INTENT_CONTRACT_MAP.get(decision.intent)

        # Special handling for RUN_AGENT - use stratum
        if decision.intent == Intent.RUN_AGENT and stratum:
            stratum_upper = stratum.upper()
            if stratum_upper in STRATUM_CONTRACT_MAP:
                contract_name = STRATUM_CONTRACT_MAP[stratum_upper]
                logger.debug(
                    f"Selected contract '{contract_name}' for stratum '{stratum}'"
                )

        if not contract_name:
            logger.debug(
                f"No contract template for intent {decision.intent.value}"
            )
            return None

        # Get template
        contract = get_template(contract_name)
        if contract is None:
            logger.warning(f"Contract template not found: {contract_name}")
            return None

        # Adjust based on complexity
        contract = self._adjust_for_complexity(contract, decision.complexity)

        logger.info(
            f"Selected contract '{contract.name}' for intent "
            f"{decision.intent.value} (complexity={decision.complexity:.2f})"
        )

        return contract

    def _adjust_for_complexity(
        self,
        contract: Contract,
        complexity: float,
    ) -> Contract:
        """Adjust contract parameters based on complexity.

        Higher complexity tasks get more lenient constraints (more retries,
        higher token budget) because they're expected to need more attempts.
        Lower complexity tasks get stricter constraints.

        Adjustment rules:
        - Complexity >= 0.8: max_retries += 1 (capped at 5), tokens * 1.5
        - Complexity <= 0.5: max_retries -= 1 (min 1)

        Args:
            contract: Base contract to adjust.
            complexity: Complexity score (0.0-1.0).

        Returns:
            Adjusted contract (new instance).
        """
        # Convert to dict for modification
        contract_data = contract.to_dict()

        if complexity >= 0.8:
            # Very complex - more retries, more tokens
            contract_data["max_retries"] = min(contract.max_retries + 1, 5)

            if contract_data["constraints"].get("max_total_tokens"):
                contract_data["constraints"]["max_total_tokens"] = int(
                    contract_data["constraints"]["max_total_tokens"] * 1.5
                )

            logger.debug(
                f"Increased limits for complexity {complexity:.2f}: "
                f"max_retries={contract_data['max_retries']}"
            )

        elif complexity <= 0.5:
            # Simple - fewer retries, stricter
            contract_data["max_retries"] = max(contract.max_retries - 1, 1)

            logger.debug(
                f"Decreased retries for complexity {complexity:.2f}: "
                f"max_retries={contract_data['max_retries']}"
            )

        return Contract.from_dict(contract_data)

    def get_contract_for_stratum(
        self,
        stratum: str,
        complexity: float = 0.5,
    ) -> Optional[Contract]:
        """Get a contract template for a specific ACTi stratum.

        Convenience method for getting stratum-specific contracts
        without a full RouteDecision.

        Args:
            stratum: ACTi stratum name (RTI, RAI, ZACS, EEI, IGE).
            complexity: Optional complexity for adjustment.

        Returns:
            Contract if stratum has a template, None otherwise.

        Example:
            >>> selector = ContractSelector()
            >>> contract = selector.get_contract_for_stratum("RAI", 0.6)
            >>> contract.name
            'lead_qualification'
        """
        stratum_upper = stratum.upper()
        contract_name = STRATUM_CONTRACT_MAP.get(stratum_upper)

        if not contract_name:
            return None

        contract = get_template(contract_name)
        if contract and complexity != 0.5:
            contract = self._adjust_for_complexity(contract, complexity)

        return contract

    def should_require_contract(
        self,
        complexity: float,
        intent: Intent,
    ) -> bool:
        """Determine if a contract should be required for a task.

        Based on complexity and intent, determines whether contract
        verification is recommended.

        Args:
            complexity: Task complexity (0.0-1.0).
            intent: The classified intent.

        Returns:
            True if contract verification is recommended.

        Example:
            >>> selector = ContractSelector()
            >>> selector.should_require_contract(0.8, Intent.RUN_AGENT)
            True
            >>> selector.should_require_contract(0.3, Intent.GENERAL_CHAT)
            False
        """
        # System commands and chat don't need contracts
        if intent in (Intent.SYSTEM_COMMAND, Intent.GENERAL_CHAT):
            return False

        # Memory queries rarely need contracts
        if intent == Intent.QUERY_MEMORY:
            return False

        # Complex tasks should use contracts
        return complexity >= COMPLEXITY_OPTIONAL_CONTRACT

    def get_recommended_failure_strategy(
        self,
        complexity: float,
    ) -> FailureStrategy:
        """Get recommended failure strategy based on complexity.

        Higher complexity tasks get more forgiving strategies.

        Args:
            complexity: Task complexity (0.0-1.0).

        Returns:
            Recommended FailureStrategy.

        Example:
            >>> selector = ContractSelector()
            >>> selector.get_recommended_failure_strategy(0.95)
            <FailureStrategy.FAIL: 'fail'>
            >>> selector.get_recommended_failure_strategy(0.6)
            <FailureStrategy.FALLBACK: 'fallback'>
        """
        if complexity >= COMPLEXITY_REQUIRED_CONTRACT:
            # Critical tasks should fail rather than provide wrong output
            return FailureStrategy.FAIL
        elif complexity >= COMPLEXITY_OPTIONAL_CONTRACT:
            # Complex tasks should retry
            return FailureStrategy.RETRY
        else:
            # Simple tasks can fall back
            return FailureStrategy.FALLBACK

    def list_available_contracts(self) -> list[str]:
        """List all available contract templates.

        Returns:
            List of contract template names.
        """
        return list(self._templates.keys())


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "ContractSelector",
    "INTENT_CONTRACT_MAP",
    "STRATUM_CONTRACT_MAP",
    "COMPLEXITY_NO_CONTRACT",
    "COMPLEXITY_OPTIONAL_CONTRACT",
    "COMPLEXITY_REQUIRED_CONTRACT",
]
