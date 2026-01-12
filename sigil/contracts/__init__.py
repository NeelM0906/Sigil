"""Contracts module for Sigil v2 framework.

This module implements contract-based verification for guaranteed outputs:
- Contract definition and validation
- Multi-validator verification pipelines
- Retry logic with prompt refinement
- Fallback handling strategies
- Cross-phase integration bridges (Phase 6.5)

Key Components:
    - Contract, Deliverable, ContractConstraints: Schema definitions
    - ContractValidator: Validates outputs against contracts
    - ContractExecutor: Orchestrates contract-enforced execution
    - RetryManager: Handles retry logic with prompt refinement
    - FallbackManager: Generates fallback outputs

Integration Bridges:
    - ContractSelector: Route-based contract selection (Phase 3)
    - ContractMemoryBridge: Contract template/result storage (Phase 4)
    - PlanContractBridge: Per-step contract verification (Phase 5)
    - ContractAwareReasoningManager: Contract-aware prompts (Phase 5)

Usage:
    >>> from sigil.contracts import Contract, Deliverable, ContractExecutor
    >>> from sigil.contracts.templates.acti import lead_qualification_contract
    >>>
    >>> # Use a pre-built template
    >>> contract = lead_qualification_contract()
    >>>
    >>> # Or build custom contract
    >>> contract = Contract(
    ...     name="my_contract",
    ...     description="Custom output verification",
    ...     deliverables=[
    ...         Deliverable(
    ...             name="result",
    ...             type="dict",
    ...             description="The result dictionary"
    ...         )
    ...     ]
    ... )
    >>>
    >>> # Execute with contract
    >>> executor = ContractExecutor()
    >>> result = await executor.execute_with_contract(agent, task, contract)
    >>>
    >>> # Use integration bridges
    >>> from sigil.contracts.integration import ContractSelector
    >>> selector = ContractSelector()
    >>> contract = selector.select_for_route(route_decision, stratum="RAI")
"""

# Schema exports
from sigil.contracts.schema import (
    Contract,
    ContractConstraints,
    Deliverable,
    FailureStrategy,
)

# Validator exports
from sigil.contracts.validator import (
    ContractValidator,
    ErrorSeverity,
    ValidationError,
    ValidationResult,
)

# Retry exports
from sigil.contracts.retry import (
    RetryContext,
    RetryManager,
)

# Fallback exports
from sigil.contracts.fallback import (
    FallbackManager,
    FallbackResult,
    FallbackStrategy,
)

# Executor exports
from sigil.contracts.executor import (
    AgentProtocol,
    AppliedStrategy,
    ContractExecutor,
    ContractResult,
    execute_with_contract,
)

# Integration bridge exports (Phase 6.5)
from sigil.contracts.integration import (
    # Router bridge
    ContractSelector,
    INTENT_CONTRACT_MAP,
    STRATUM_CONTRACT_MAP,
    # Memory bridge
    ContractMemoryBridge,
    CONTRACT_RESOURCE_TYPE,
    VALIDATION_RESULT_TYPE,
    # Plan bridge
    ContractSpec,
    PlanContractBridge,
    StepContractResult,
    # Reasoning bridge
    ContractAwareReasoningManager,
    ContractReasoningContext,
)

__all__ = [
    # Schema
    "Contract",
    "ContractConstraints",
    "Deliverable",
    "FailureStrategy",
    # Validator
    "ContractValidator",
    "ErrorSeverity",
    "ValidationError",
    "ValidationResult",
    # Retry
    "RetryContext",
    "RetryManager",
    # Fallback
    "FallbackManager",
    "FallbackResult",
    "FallbackStrategy",
    # Executor
    "AgentProtocol",
    "AppliedStrategy",
    "ContractExecutor",
    "ContractResult",
    "execute_with_contract",
    # Integration bridges (Phase 6.5)
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
