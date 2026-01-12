"""Memory to Contract bridge for Sigil v2 Phase 6.

This module provides integration between the memory system (Phase 4) and the
contracts system (Phase 6), enabling:

- Contract template storage and versioning in memory
- Validation result storage for learning
- Successful output examples for few-shot learning
- Common error pattern retrieval for proactive prompt enhancement

Key Components:
    - ContractMemoryBridge: Main bridge class for memory operations
    - CONTRACT_RESOURCE_TYPE: Resource type for contract templates
    - VALIDATION_RESULT_TYPE: Resource type for validation results

The bridge supports the contract learning loop:
1. Store validation results (pass/fail with errors)
2. Analyze common failure patterns
3. Retrieve examples for prompt enhancement
4. Track success rates per contract

Usage:
    >>> from sigil.contracts.integration.memory_bridge import ContractMemoryBridge
    >>> from sigil.memory.manager import MemoryManager
    >>>
    >>> manager = MemoryManager()
    >>> bridge = ContractMemoryBridge(manager)
    >>>
    >>> # Store validation result
    >>> await bridge.store_validation_result(
    ...     contract_name="lead_qualification",
    ...     output={"score": 75, "bant_assessment": {...}},
    ...     validation_passed=True,
    ...     errors=[],
    ...     session_id="sess-123",
    ... )
    >>>
    >>> # Get common errors for a contract
    >>> errors = await bridge.get_common_validation_errors("lead_qualification")

Example:
    >>> bridge = ContractMemoryBridge(memory_manager)
    >>> context = await bridge.get_contract_context("lead_qualification")
    >>> print(context["success_rate"])
    0.85
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional, TYPE_CHECKING

from sigil.contracts.schema import Contract

if TYPE_CHECKING:
    from sigil.memory.manager import MemoryManager


# =============================================================================
# Module Logger
# =============================================================================

logger = logging.getLogger(__name__)


# =============================================================================
# Constants
# =============================================================================

CONTRACT_RESOURCE_TYPE = "contract_template"
"""Resource type identifier for contract templates in memory."""

VALIDATION_RESULT_TYPE = "contract_validation"
"""Resource type identifier for validation results in memory."""

EXEMPLAR_OUTPUT_TYPE = "contract_example"
"""Resource type identifier for successful output examples."""

# Maximum characters for truncated output storage
MAX_OUTPUT_CHARS = 1000


# =============================================================================
# Contract Context
# =============================================================================


@dataclass
class ContractContext:
    """Context information for contract execution.

    Aggregated information from memory to enhance contract execution.

    Attributes:
        template: The contract template if found in memory.
        examples: List of successful output examples.
        common_errors: Frequent validation error patterns.
        success_rate: Historical success rate (0.0-1.0).
        total_executions: Total number of executions tracked.
    """

    template: Optional[Contract]
    examples: list[dict[str, Any]]
    common_errors: list[dict[str, Any]]
    success_rate: float
    total_executions: int

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "template": self.template.to_dict() if self.template else None,
            "examples": self.examples,
            "common_errors": self.common_errors,
            "success_rate": self.success_rate,
            "total_executions": self.total_executions,
        }


# =============================================================================
# Contract Memory Bridge
# =============================================================================


class ContractMemoryBridge:
    """Bridge between contracts and the memory system.

    Provides integration for:
    - Storing and retrieving contract templates from memory
    - Recording validation results for learning
    - Analyzing common validation errors
    - Storing successful outputs as exemplars

    The bridge enables contracts to improve over time by:
    1. Tracking success/failure patterns
    2. Identifying common errors
    3. Providing historical context for prompt enhancement

    Attributes:
        memory: The MemoryManager instance.

    Example:
        >>> bridge = ContractMemoryBridge(memory_manager)
        >>> await bridge.store_validation_result(...)
        >>> errors = await bridge.get_common_validation_errors("lead_qual")
    """

    def __init__(self, memory_manager: "MemoryManager") -> None:
        """Initialize the ContractMemoryBridge.

        Args:
            memory_manager: The MemoryManager instance for storage.
        """
        self.memory = memory_manager

    # =========================================================================
    # Contract Template Storage
    # =========================================================================

    async def store_contract_template(
        self,
        contract: Contract,
        version: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> str:
        """Store a contract template in memory.

        Stores the contract definition for versioning and retrieval.
        Multiple versions of the same contract can be stored.

        Args:
            contract: Contract to store.
            version: Optional version override.
            session_id: Optional session ID for event tracking.

        Returns:
            Resource ID of stored contract.

        Example:
            >>> resource_id = await bridge.store_contract_template(contract)
            >>> print(resource_id)
            'res-abc123...'
        """
        contract_version = version or contract.version

        resource = await self.memory.store_resource(
            resource_type=CONTRACT_RESOURCE_TYPE,
            content=json.dumps(contract.to_dict(), indent=2),
            metadata={
                "contract_name": contract.name,
                "version": contract_version,
                "deliverables": contract.get_deliverable_names(),
                "failure_strategy": contract.failure_strategy.value,
            },
            session_id=session_id,
        )

        logger.info(
            f"Stored contract template '{contract.name}' v{contract_version} "
            f"as resource {resource.resource_id}"
        )

        return resource.resource_id

    async def retrieve_contract_template(
        self,
        contract_name: str,
        version: Optional[str] = None,
    ) -> Optional[Contract]:
        """Retrieve a contract template from memory.

        Searches for a stored contract by name and optionally version.

        Args:
            contract_name: Name of the contract.
            version: Optional specific version.

        Returns:
            Contract if found, None otherwise.

        Example:
            >>> contract = await bridge.retrieve_contract_template("lead_qual")
            >>> print(contract.name)
            'lead_qualification'
        """
        # Use memory retrieval to find matching contracts
        results = await self.memory.retrieve(
            query=f"contract template {contract_name}",
            k=10,
        )

        for item in results:
            # Check if this item matches our criteria
            # Items don't have direct metadata, so we need to check content
            if item.category == CONTRACT_RESOURCE_TYPE:
                try:
                    # Get the source resource
                    resource = await self.memory.get_resource(
                        item.source_resource_id
                    )
                    if resource:
                        content = json.loads(resource.content)
                        if content.get("name") == contract_name:
                            if version is None or content.get("version") == version:
                                return Contract.from_dict(content)
                except (json.JSONDecodeError, KeyError) as e:
                    logger.warning(f"Failed to parse contract: {e}")
                    continue

        return None

    # =========================================================================
    # Validation Result Storage
    # =========================================================================

    async def store_validation_result(
        self,
        contract_name: str,
        output: dict[str, Any],
        validation_passed: bool,
        errors: list[str],
        session_id: str,
    ) -> str:
        """Store validation result for learning.

        Records the outcome of contract validation to enable
        pattern analysis and learning over time.

        Args:
            contract_name: Name of the contract validated.
            output: The output that was validated.
            validation_passed: Whether validation passed.
            errors: List of validation error messages.
            session_id: Session where validation occurred.

        Returns:
            Resource ID of stored result.

        Example:
            >>> resource_id = await bridge.store_validation_result(
            ...     contract_name="lead_qualification",
            ...     output={"score": 75},
            ...     validation_passed=True,
            ...     errors=[],
            ...     session_id="sess-123",
            ... )
        """
        content = {
            "contract_name": contract_name,
            "output_sample": self._truncate_output(output),
            "passed": validation_passed,
            "errors": errors,
            "session_id": session_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        resource = await self.memory.store_resource(
            resource_type=VALIDATION_RESULT_TYPE,
            content=json.dumps(content, indent=2),
            metadata={
                "contract_name": contract_name,
                "passed": validation_passed,
                "error_count": len(errors),
            },
            session_id=session_id,
        )

        logger.debug(
            f"Stored validation result for '{contract_name}': "
            f"passed={validation_passed}, errors={len(errors)}"
        )

        # Extract insights if validation failed (for pattern learning)
        if not validation_passed and errors:
            try:
                await self.memory.extract_and_store(
                    resource_id=resource.resource_id,
                    session_id=session_id,
                )
            except Exception as e:
                logger.warning(f"Failed to extract validation insights: {e}")

        return resource.resource_id

    async def store_successful_output_as_example(
        self,
        contract_name: str,
        output: dict[str, Any],
        session_id: Optional[str] = None,
    ) -> str:
        """Store successful output as example for future reference.

        When an output passes validation on first try, it becomes
        a positive example that can be used for:
        - Prompt enhancement
        - Few-shot learning
        - Contract template refinement

        Args:
            contract_name: Name of the contract.
            output: The successful output.
            session_id: Optional session ID.

        Returns:
            Resource ID of stored example.

        Example:
            >>> await bridge.store_successful_output_as_example(
            ...     "lead_qualification",
            ...     {"score": 85, "bant_assessment": {...}},
            ... )
        """
        example = {
            "contract_name": contract_name,
            "example_output": output,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "is_exemplar": True,
        }

        resource = await self.memory.store_resource(
            resource_type=EXEMPLAR_OUTPUT_TYPE,
            content=json.dumps(example, indent=2),
            metadata={
                "contract_name": contract_name,
                "is_exemplar": True,
            },
            session_id=session_id,
        )

        logger.info(
            f"Stored exemplar output for '{contract_name}': {resource.resource_id}"
        )

        return resource.resource_id

    # =========================================================================
    # Error Pattern Analysis
    # =========================================================================

    async def get_common_validation_errors(
        self,
        contract_name: str,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Get common validation errors for a contract.

        Analyzes stored validation results to identify frequent
        error patterns. Useful for proactive prompt enhancement.

        Args:
            contract_name: Name of the contract.
            limit: Maximum errors to return.

        Returns:
            List of common error patterns with counts.

        Example:
            >>> errors = await bridge.get_common_validation_errors("lead_qual")
            >>> print(errors[0])
            {'error': 'Missing field: bant_assessment', 'count': 15}
        """
        # Retrieve validation failures for this contract
        results = await self.memory.retrieve(
            query=f"validation errors {contract_name} failed",
            k=50,  # Get more results for better analysis
        )

        all_errors: list[str] = []

        for item in results:
            if item.category == VALIDATION_RESULT_TYPE:
                try:
                    resource = await self.memory.get_resource(
                        item.source_resource_id
                    )
                    if resource:
                        content = json.loads(resource.content)
                        if (
                            content.get("contract_name") == contract_name
                            and not content.get("passed", True)
                        ):
                            all_errors.extend(content.get("errors", []))
                except (json.JSONDecodeError, KeyError):
                    continue

        # Count and rank errors
        error_counts = Counter(all_errors)

        return [
            {"error": error, "count": count}
            for error, count in error_counts.most_common(limit)
        ]

    # =========================================================================
    # Context Retrieval
    # =========================================================================

    async def get_contract_context(
        self,
        contract_name: str,
    ) -> ContractContext:
        """Retrieve full context for contract execution.

        Aggregates all available information about a contract
        from memory for enhanced execution.

        Args:
            contract_name: Name of the contract.

        Returns:
            ContractContext with template, examples, errors, and stats.

        Example:
            >>> context = await bridge.get_contract_context("lead_qual")
            >>> print(f"Success rate: {context.success_rate:.1%}")
            'Success rate: 85.0%'
        """
        # Get template
        template = await self.retrieve_contract_template(contract_name)

        # Get exemplar outputs
        examples = await self._get_exemplar_outputs(contract_name, limit=3)

        # Get common errors
        common_errors = await self.get_common_validation_errors(
            contract_name, limit=5
        )

        # Calculate success rate
        success_rate, total_executions = await self._calculate_success_rate(
            contract_name
        )

        return ContractContext(
            template=template,
            examples=examples,
            common_errors=common_errors,
            success_rate=success_rate,
            total_executions=total_executions,
        )

    async def _get_exemplar_outputs(
        self,
        contract_name: str,
        limit: int = 3,
    ) -> list[dict[str, Any]]:
        """Get exemplar (successful) outputs for a contract.

        Args:
            contract_name: Name of the contract.
            limit: Maximum examples to return.

        Returns:
            List of successful output dictionaries.
        """
        results = await self.memory.retrieve(
            query=f"exemplar output {contract_name} successful",
            k=limit * 2,  # Get extra in case some don't match
        )

        examples: list[dict[str, Any]] = []

        for item in results:
            if len(examples) >= limit:
                break

            if item.category == EXEMPLAR_OUTPUT_TYPE:
                try:
                    resource = await self.memory.get_resource(
                        item.source_resource_id
                    )
                    if resource:
                        content = json.loads(resource.content)
                        if (
                            content.get("contract_name") == contract_name
                            and content.get("is_exemplar")
                        ):
                            examples.append(content.get("example_output", {}))
                except (json.JSONDecodeError, KeyError):
                    continue

        return examples

    async def _calculate_success_rate(
        self,
        contract_name: str,
        window_days: int = 7,
    ) -> tuple[float, int]:
        """Calculate success rate for a contract.

        Args:
            contract_name: Name of the contract.
            window_days: Days to look back for calculation.

        Returns:
            Tuple of (success_rate, total_executions).
        """
        results = await self.memory.retrieve(
            query=f"validation result {contract_name}",
            k=100,  # Get substantial history
        )

        passed = 0
        total = 0

        for item in results:
            if item.category == VALIDATION_RESULT_TYPE:
                try:
                    resource = await self.memory.get_resource(
                        item.source_resource_id
                    )
                    if resource:
                        content = json.loads(resource.content)
                        if content.get("contract_name") == contract_name:
                            total += 1
                            if content.get("passed", False):
                                passed += 1
                except (json.JSONDecodeError, KeyError):
                    continue

        if total == 0:
            return 0.0, 0

        return passed / total, total

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def _truncate_output(
        self,
        output: dict[str, Any],
        max_chars: int = MAX_OUTPUT_CHARS,
    ) -> dict[str, Any]:
        """Truncate output for storage.

        Prevents storing excessively large outputs in memory.

        Args:
            output: Output dictionary to truncate.
            max_chars: Maximum characters per value.

        Returns:
            Truncated output dictionary.
        """
        truncated = {}
        for key, value in output.items():
            str_val = str(value)
            if len(str_val) > max_chars:
                truncated[key] = str_val[:max_chars] + "..."
            else:
                truncated[key] = value
        return truncated


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "ContractMemoryBridge",
    "ContractContext",
    "CONTRACT_RESOURCE_TYPE",
    "VALIDATION_RESULT_TYPE",
    "EXEMPLAR_OUTPUT_TYPE",
]
