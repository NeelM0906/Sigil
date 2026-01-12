"""Contract integration bridges for cross-phase communication.

This module provides integration bridges that connect the Phase 6 contracts
system with other phases in the Sigil v2 framework:

- Phase 3 (Routing): ContractSelector for route-based contract selection
- Phase 4 (Memory): ContractMemoryBridge for template storage and validation history
- Phase 5 (Planning): PlanContractBridge for per-step contract verification
- Phase 5 (Reasoning): ContractAwareReasoningManager for contract-aware prompts

Usage:
    >>> from sigil.contracts.integration import (
    ...     ContractSelector,
    ...     ContractMemoryBridge,
    ...     PlanContractBridge,
    ...     ContractAwareReasoningManager,
    ... )

    >>> # Select contract based on route decision
    >>> selector = ContractSelector()
    >>> contract = selector.select_for_route(route_decision, stratum="RAI")

    >>> # Store validation results in memory
    >>> memory_bridge = ContractMemoryBridge(memory_manager)
    >>> await memory_bridge.store_validation_result(...)

Example:
    >>> from sigil.contracts.integration import ContractSelector
    >>> from sigil.routing.router import Router, RouteDecision
    >>>
    >>> selector = ContractSelector()
    >>> decision = RouteDecision(intent=Intent.RUN_AGENT, ...)
    >>> contract = selector.select_for_route(decision, stratum="RAI")
    >>> print(contract.name)
    'lead_qualification'
"""

from __future__ import annotations

from sigil.contracts.integration.router_bridge import (
    ContractSelector,
    INTENT_CONTRACT_MAP,
    STRATUM_CONTRACT_MAP,
)
from sigil.contracts.integration.memory_bridge import (
    ContractMemoryBridge,
    CONTRACT_RESOURCE_TYPE,
    VALIDATION_RESULT_TYPE,
)
from sigil.contracts.integration.plan_bridge import (
    ContractSpec,
    PlanContractBridge,
    StepContractResult,
)
from sigil.contracts.integration.reasoning_bridge import (
    ContractAwareReasoningManager,
    ContractReasoningContext,
)


__all__ = [
    # Router bridge
    "ContractSelector",
    "INTENT_CONTRACT_MAP",
    "STRATUM_CONTRACT_MAP",
    # Memory bridge
    "ContractMemoryBridge",
    "CONTRACT_RESOURCE_TYPE",
    "VALIDATION_RESULT_TYPE",
    # Plan bridge
    "ContractSpec",
    "PlanContractBridge",
    "StepContractResult",
    # Reasoning bridge
    "ContractAwareReasoningManager",
    "ContractReasoningContext",
]
