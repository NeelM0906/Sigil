# Phase 6 Contracts Integration: System Architecture

## Executive Summary

Phase 6 adds formal output verification with retry and fallback mechanisms to Sigil v2. This document provides the comprehensive system architecture for integrating contracts with the existing Phases 3-5 infrastructure.

**Current State:** Phases 3-5 are implemented with:
- Phase 3: Routing (IntentClassifier, ComplexityAssessor, Router)
- Phase 4: Memory (3-layer architecture with RAG/LLM retrieval)
- Phase 5: Planning & Reasoning (5 strategies: Direct, CoT, ToT, ReAct, MCTS)
- Phase 6: Contracts foundation exists (schema, validator, retry, fallback, executor, templates)

**Target State:** Full cross-phase integration enabling:
- Router-driven contract selection based on intent and complexity
- Memory-backed contract templates with validation result learning
- Plan step contracts with per-step verification
- Reasoning strategy awareness of contract requirements
- Complete event sourcing for audit trails

---

## Table of Contents

1. [System Integration Architecture](#1-system-integration-architecture)
2. [Cross-Phase Data Flow](#2-cross-phase-data-flow)
3. [Router Integration (Phase 3)](#3-router-integration-phase-3)
4. [Memory Integration (Phase 4)](#4-memory-integration-phase-4)
5. [Planning & Reasoning Integration (Phase 5)](#5-planning--reasoning-integration-phase-5)
6. [Token Budget Integration](#6-token-budget-integration)
7. [Event Store Integration](#7-event-store-integration)
8. [Enhanced Contract Executor](#8-enhanced-contract-executor)
9. [Validation System Architecture](#9-validation-system-architecture)
10. [Retry System Architecture](#10-retry-system-architecture)
11. [Fallback System Architecture](#11-fallback-system-architecture)
12. [Contract Storage Architecture](#12-contract-storage-architecture)
13. [Concurrency and Performance](#13-concurrency-and-performance)
14. [Error Handling and Recovery](#14-error-handling-and-recovery)
15. [Learning Integration](#15-learning-integration)
16. [Monitoring and Observability](#16-monitoring-and-observability)
17. [Configuration](#17-configuration)
18. [Implementation Checklist](#18-implementation-checklist)

---

## 1. System Integration Architecture

### 1.1 High-Level Component Diagram

```
+------------------------------------------------------------------+
|                        SIGIL v2 SYSTEM                           |
+------------------------------------------------------------------+
|                                                                  |
|  +------------+     +-------------+     +----------------+       |
|  |   Phase 3  |     |   Phase 5   |     |    Phase 6     |       |
|  |   Routing  | --> |  Planning & | --> |   Contracts    |       |
|  |            |     |  Reasoning  |     |                |       |
|  +-----+------+     +------+------+     +--------+-------+       |
|        |                   |                     |               |
|        v                   v                     v               |
|  +------------+     +-------------+     +----------------+       |
|  |Complexity  |     |   Memory    |     |   Validator    |       |
|  |Assessor    |     |   Context   |     |   RetryMgr     |       |
|  |IntentClass |     |  Retrieval  |     |   FallbackMgr  |       |
|  +------------+     +------+------+     +--------+-------+       |
|                            |                     |               |
|                            v                     v               |
|                     +--------------+     +----------------+      |
|                     |   Phase 4    |     |   EventStore   |      |
|                     |    Memory    |     | (Phase 3 infra)|      |
|                     +--------------+     +----------------+      |
|                                                                  |
+------------------------------------------------------------------+
```

### 1.2 Module Dependencies

```
sigil/contracts/
|-- __init__.py              # Exports: ContractExecutor, Contract, etc.
|-- schema.py                # Contract, Deliverable, ContractConstraints
|-- validator.py             # ContractValidator, ValidationResult
|-- retry.py                 # RetryManager, RetryContext
|-- fallback.py              # FallbackManager, FallbackResult
|-- executor.py              # ContractExecutor, ContractResult
|-- templates/
|   |-- __init__.py
|   |-- acti.py              # ACTi stratum-specific contracts
|-- integration/             # NEW: Cross-phase integration
|   |-- __init__.py
|   |-- router_bridge.py     # Router -> Contract selection
|   |-- memory_bridge.py     # Memory -> Contract storage/retrieval
|   |-- plan_bridge.py       # Plan steps -> Contract verification
|   |-- reasoning_bridge.py  # Reasoning -> Contract awareness
```

### 1.3 Integration Points Summary

| Source Phase | Integration Point | Target Phase | Data Flow |
|--------------|-------------------|--------------|-----------|
| Phase 3 | RouteDecision.use_contracts | Phase 6 | Contract selection trigger |
| Phase 3 | RouteDecision.complexity | Phase 6 | Determines contract strictness |
| Phase 3 | EventStore | Phase 6 | Contract events stored |
| Phase 4 | MemoryManager | Phase 6 | Contract templates, validation results |
| Phase 5 | PlanStep.contract_spec | Phase 6 | Per-step verification |
| Phase 5 | ReasoningManager | Phase 6 | Strategy selection awareness |
| Phase 6 | ValidationResult | Phase 4 | Stored for learning |
| Phase 6 | ContractResult | Phase 5 | Plan execution feedback |

---

## 2. Cross-Phase Data Flow

### 2.1 Complete Request Lifecycle

```
User Request
    |
    v
+-------------------+
|  Router (Phase 3) |
|  - classify_intent() -> Intent.CREATE_AGENT
|  - assess_complexity() -> 0.72 (high)
|  - _should_use_contracts() -> True (complexity > 0.7)
+--------+----------+
         |
         | RouteDecision(use_contracts=True, complexity=0.72)
         v
+-------------------+
|Contract Selection |
|  - select_contract_for_intent(Intent.CREATE_AGENT)
|  - adjust_strictness(complexity=0.72)
|  - Returns: agent_config_contract(max_retries=2)
+--------+----------+
         |
         | Contract + RouteDecision
         v
+-------------------+
|Planning (Phase 5) |
|  - Planner.generate(goal, context)
|  - Attaches contract_spec to critical steps
|  - Returns: Plan with ContractSpec per step
+--------+----------+
         |
         | Plan
         v
+-------------------+
|Memory (Phase 4)   |
|  - Retrieve relevant context
|  - Retrieve past validation patterns
|  - Returns: Context for reasoning
+--------+----------+
         |
         | Context + Plan
         v
+-------------------+
|Reasoning (Phase 5)|
|  - select_strategy() -> considers contract complexity
|  - execute() -> produces output
|  - Returns: StrategyResult with output dict
+--------+----------+
         |
         | StrategyResult.output
         v
+-------------------+
|ContractExecutor   |
|  (Phase 6)        |
|  - validate(output, contract)
|  - if invalid: retry or fallback
|  - emit_events() to EventStore
|  - store_result() to Memory
+--------+----------+
         |
         | ContractResult
         v
+-------------------+
|Result + Metadata  |
|  - Output (validated or fallback)
|  - Validation details
|  - Token usage
|  - Event trail
+-------------------+
```

### 2.2 State Machine: Contract Execution

```
                        START
                          |
                          v
                   +------+------+
                   |Execute Agent|
                   +------+------+
                          |
                          v
                   +------+------+
                   |  Validate   |
                   |   Output    |
                   +------+------+
                          |
              +-----------+-----------+
              |                       |
           VALID?                  INVALID
              |                       |
              v                       v
    +---------+--------+    +--------+--------+
    |ContractValidated |    | Decide Action   |
    |    Event         |    +--------+--------+
    +---------+--------+             |
              |              +-------+-------+
              v              |               |
           RETURN      Can Retry?      Can't Retry
                             |               |
                             v               v
                   +--------+--------+ +-----+-----+
                   | RetryAttempt   | |Apply      |
                   | - refine prompt| |Fallback   |
                   | - increment n  | +-----+-----+
                   +--------+--------+       |
                            |                v
                            |       +--------+--------+
                            |       |Strategy=FAIL?  |
                            |       +--------+--------+
                            |                |
                            |       +--------+--------+
                            |       |  YES   |   NO   |
                            |       +--------+--------+
                            |           |         |
                            |           v         v
                            |    +------+---+ +---+------+
                            |    |  RAISE   | | Build    |
                            |    |Exception | | Partial/ |
                            |    +----------+ | Template |
                            |                 +----+-----+
                            |                      |
                            v                      v
                   +--------+--------+    +--------+--------+
                   | Loop back to   |    | Fallback Event  |
                   | Execute Agent  |    +--------+--------+
                   +----------------+             |
                                                  v
                                              RETURN
```

---

## 3. Router Integration (Phase 3)

### 3.1 Enhanced Route Decision

The existing `RouteDecision` already includes `use_contracts`. We extend the Router to select appropriate contracts.

```python
# sigil/contracts/integration/router_bridge.py

from sigil.routing.router import Router, RouteDecision, Intent
from sigil.contracts.schema import Contract
from sigil.contracts.templates.acti import CONTRACT_TEMPLATES, get_template

# Intent to contract mapping
INTENT_CONTRACT_MAP: dict[Intent, str] = {
    Intent.CREATE_AGENT: "agent_config",  # Future template
    Intent.RUN_AGENT: None,  # Dynamic based on agent type
    Intent.QUERY_MEMORY: None,  # No contract needed
    Intent.MODIFY_AGENT: "agent_config",
    Intent.SYSTEM_COMMAND: None,
    Intent.GENERAL_CHAT: None,
}

# Stratum to contract mapping (for RUN_AGENT)
STRATUM_CONTRACT_MAP: dict[str, str] = {
    "RTI": "research_report",
    "RAI": "lead_qualification",
    "ZACS": "appointment_booking",
    "EEI": "market_analysis",
    "IGE": "compliance_check",
}


class ContractSelector:
    """Selects appropriate contracts based on routing decisions."""

    def __init__(self):
        self._templates = CONTRACT_TEMPLATES

    def select_for_route(
        self,
        decision: RouteDecision,
        stratum: str | None = None,
    ) -> Contract | None:
        """Select a contract based on route decision.

        Args:
            decision: The routing decision from Router.
            stratum: Optional ACTi stratum for RUN_AGENT intent.

        Returns:
            Contract if applicable, None if no contract needed.
        """
        if not decision.use_contracts:
            return None

        # Check intent-based mapping
        contract_name = INTENT_CONTRACT_MAP.get(decision.intent)

        # Special handling for RUN_AGENT - use stratum
        if decision.intent == Intent.RUN_AGENT and stratum:
            contract_name = STRATUM_CONTRACT_MAP.get(stratum)

        if not contract_name:
            return None

        # Get template and adjust based on complexity
        contract = get_template(contract_name)
        if contract:
            contract = self._adjust_for_complexity(contract, decision.complexity)

        return contract

    def _adjust_for_complexity(
        self,
        contract: Contract,
        complexity: float,
    ) -> Contract:
        """Adjust contract parameters based on complexity.

        Higher complexity -> more retries, looser constraints.
        Lower complexity -> fewer retries, stricter constraints.

        Args:
            contract: Base contract to adjust.
            complexity: Complexity score (0.0-1.0).

        Returns:
            Adjusted contract.
        """
        # Create a copy to avoid mutating original
        adjusted_data = contract.to_dict()

        if complexity >= 0.8:
            # Very complex - more retries, more tokens
            adjusted_data["max_retries"] = min(contract.max_retries + 1, 5)
            if adjusted_data["constraints"].get("max_total_tokens"):
                adjusted_data["constraints"]["max_total_tokens"] = int(
                    adjusted_data["constraints"]["max_total_tokens"] * 1.5
                )
        elif complexity <= 0.5:
            # Simple - fewer retries, stricter
            adjusted_data["max_retries"] = max(contract.max_retries - 1, 1)

        return Contract.from_dict(adjusted_data)
```

### 3.2 Router Contract Awareness

```python
# Enhanced Router.route() method signature
class EnhancedRouteDecision(RouteDecision):
    """Extended routing decision with contract information."""

    selected_contract: Contract | None = None
    contract_strictness: str = "normal"  # "strict", "normal", "relaxed"
```

### 3.3 Complexity-Based Contract Selection Rules

| Complexity Range | Contract Behavior |
|------------------|-------------------|
| 0.0 - 0.5 | No contract (simple tasks) |
| 0.5 - 0.7 | Optional contract with fallback strategy |
| 0.7 - 0.9 | Required contract with retry strategy |
| 0.9 - 1.0 | Strict contract with fail strategy (critical) |

---

## 4. Memory Integration (Phase 4)

### 4.1 Contract Template Storage

Contract templates can be stored and versioned in the memory system for dynamic updates.

```python
# sigil/contracts/integration/memory_bridge.py

from sigil.memory.manager import MemoryManager
from sigil.memory.layers.resources import ResourceType
from sigil.contracts.schema import Contract

CONTRACT_RESOURCE_TYPE = "contract_template"
VALIDATION_RESULT_TYPE = "contract_validation"


class ContractMemoryBridge:
    """Bridge between contracts and memory system."""

    def __init__(self, memory_manager: MemoryManager):
        self.memory = memory_manager

    async def store_contract_template(
        self,
        contract: Contract,
        version: str | None = None,
    ) -> str:
        """Store a contract template in memory.

        Args:
            contract: Contract to store.
            version: Optional version override.

        Returns:
            Resource ID of stored contract.
        """
        resource = await self.memory.store_resource(
            content=json.dumps(contract.to_dict(), indent=2),
            resource_type=CONTRACT_RESOURCE_TYPE,
            metadata={
                "contract_name": contract.name,
                "version": version or contract.version,
                "deliverables": contract.get_deliverable_names(),
                "failure_strategy": contract.failure_strategy.value,
            },
        )
        return resource.resource_id

    async def retrieve_contract_template(
        self,
        contract_name: str,
        version: str | None = None,
    ) -> Contract | None:
        """Retrieve a contract template from memory.

        Args:
            contract_name: Name of the contract.
            version: Optional specific version.

        Returns:
            Contract if found, None otherwise.
        """
        # Search for matching contract
        results = await self.memory.search(
            query=f"contract template {contract_name}",
            filters={"resource_type": CONTRACT_RESOURCE_TYPE},
            k=5,
        )

        for item in results.items:
            if item.metadata.get("contract_name") == contract_name:
                if version is None or item.metadata.get("version") == version:
                    # Load from resource
                    resource = await self.memory.get_resource(item.source_resource_id)
                    return Contract.from_dict(json.loads(resource.content))

        return None

    async def store_validation_result(
        self,
        contract_name: str,
        output: dict,
        validation_passed: bool,
        errors: list[str],
        session_id: str,
    ) -> str:
        """Store validation result for learning.

        Args:
            contract_name: Name of the contract validated.
            output: The output that was validated.
            validation_passed: Whether validation passed.
            errors: List of validation error messages.
            session_id: Session where validation occurred.

        Returns:
            Resource ID of stored result.
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
            content=json.dumps(content, indent=2),
            resource_type=VALIDATION_RESULT_TYPE,
            metadata={
                "contract_name": contract_name,
                "passed": validation_passed,
                "error_count": len(errors),
            },
        )

        # Extract insights if validation failed
        if not validation_passed and errors:
            await self.memory.extract_and_store(
                resource_id=resource.resource_id,
                category_hint="contract_validation_patterns",
            )

        return resource.resource_id

    async def get_common_validation_errors(
        self,
        contract_name: str,
        limit: int = 10,
    ) -> list[dict]:
        """Get common validation errors for a contract.

        Useful for proactive prompt enhancement.

        Args:
            contract_name: Name of the contract.
            limit: Maximum errors to return.

        Returns:
            List of common error patterns.
        """
        results = await self.memory.search(
            query=f"validation errors for {contract_name}",
            filters={
                "resource_type": VALIDATION_RESULT_TYPE,
                "passed": False,
            },
            k=limit,
        )

        errors = []
        for item in results.items:
            resource = await self.memory.get_resource(item.source_resource_id)
            content = json.loads(resource.content)
            errors.extend(content.get("errors", []))

        # Count and rank errors
        error_counts = Counter(errors)
        return [
            {"error": error, "count": count}
            for error, count in error_counts.most_common(limit)
        ]

    def _truncate_output(self, output: dict, max_chars: int = 1000) -> dict:
        """Truncate output for storage."""
        truncated = {}
        for key, value in output.items():
            str_val = str(value)
            if len(str_val) > max_chars:
                truncated[key] = str_val[:max_chars] + "..."
            else:
                truncated[key] = value
        return truncated
```

### 4.2 Successful Outputs as Example Cases

```python
async def store_successful_output_as_example(
    self,
    contract_name: str,
    output: dict,
) -> None:
    """Store successful output as example for future reference.

    When an output passes validation on first try, it becomes
    a positive example that can be used for:
    - Prompt enhancement
    - Few-shot learning
    - Contract template refinement

    Args:
        contract_name: Name of the contract.
        output: The successful output.
    """
    example = {
        "contract_name": contract_name,
        "example_output": output,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "is_exemplar": True,
    }

    await self.memory.store_resource(
        content=json.dumps(example, indent=2),
        resource_type="contract_example",
        metadata={
            "contract_name": contract_name,
            "is_exemplar": True,
        },
    )
```

### 4.3 Memory Retrieval for Contract Context

```python
async def get_contract_context(
    self,
    contract_name: str,
) -> dict:
    """Retrieve full context for a contract execution.

    Returns:
        Dictionary with:
        - template: Contract template
        - examples: Successful output examples
        - common_errors: Frequent validation errors
        - success_rate: Historical success rate
    """
    template = await self.retrieve_contract_template(contract_name)
    examples = await self._get_exemplar_outputs(contract_name, limit=3)
    common_errors = await self.get_common_validation_errors(contract_name, limit=5)
    success_rate = await self._calculate_success_rate(contract_name)

    return {
        "template": template,
        "examples": examples,
        "common_errors": common_errors,
        "success_rate": success_rate,
    }
```

---

## 5. Planning & Reasoning Integration (Phase 5)

### 5.1 Plan Steps with Contract Specifications

The existing `PlanStep` schema already supports `contract_spec`. We enhance this integration.

```python
# sigil/contracts/integration/plan_bridge.py

from sigil.planning.schemas import Plan, PlanStep, StepType
from sigil.contracts.schema import Contract, Deliverable, ContractConstraints

@dataclass
class ContractSpec:
    """Embedded contract specification for plan step output verification.

    Lighter-weight than full Contract - used for per-step validation.
    """
    required_fields: list[str]
    field_types: dict[str, str]
    validation_rules: list[str]
    max_tokens: int | None = None

    def to_contract(self, step_name: str) -> Contract:
        """Convert to full Contract for validation."""
        deliverables = [
            Deliverable(
                name=field,
                type=self.field_types.get(field, "Any"),
                description=f"Required field for step: {step_name}",
                validation_rules=[r for r in self.validation_rules if field in r],
            )
            for field in self.required_fields
        ]

        return Contract(
            name=f"step_{step_name}",
            description=f"Contract for plan step: {step_name}",
            deliverables=deliverables,
            constraints=ContractConstraints(
                max_total_tokens=self.max_tokens,
            ),
            failure_strategy=FailureStrategy.RETRY,
            max_retries=1,  # Single retry for steps
        )


class PlanContractBridge:
    """Integrates contracts into plan execution."""

    def __init__(
        self,
        contract_executor: ContractExecutor,
        memory_bridge: ContractMemoryBridge,
    ):
        self.executor = contract_executor
        self.memory = memory_bridge

    def attach_contracts_to_plan(
        self,
        plan: Plan,
        default_contract: Contract | None = None,
    ) -> Plan:
        """Attach contract specs to plan steps that need verification.

        Rules for contract attachment:
        - TOOL_CALL steps: Contract if output is used downstream
        - REASONING steps: Contract if produces structured data
        - Final step: Always use contract if provided

        Args:
            plan: Plan to enhance.
            default_contract: Contract for final output.

        Returns:
            Plan with contract specs attached.
        """
        enhanced_steps = []
        final_step_idx = len(plan.steps) - 1

        for idx, step in enumerate(plan.steps):
            enhanced_step = step

            # Attach to final step
            if idx == final_step_idx and default_contract:
                enhanced_step = self._attach_contract_to_step(step, default_contract)

            # Attach to tool calls with structured output
            elif step.step_type == StepType.TOOL_CALL:
                if self._needs_verification(step):
                    enhanced_step = self._create_tool_step_contract(step)

            # Attach to reasoning steps with structured output
            elif step.step_type == StepType.REASONING:
                if self._produces_structured_output(step):
                    enhanced_step = self._create_reasoning_step_contract(step)

            enhanced_steps.append(enhanced_step)

        return Plan(
            plan_id=plan.plan_id,
            goal=plan.goal,
            steps=enhanced_steps,
            metadata=plan.metadata,
            created_at=plan.created_at,
        )

    async def execute_step_with_contract(
        self,
        step: PlanStep,
        agent: AgentProtocol,
        context: dict,
    ) -> StepResult:
        """Execute a plan step with contract verification.

        Args:
            step: Step to execute.
            agent: Agent to run the step.
            context: Execution context.

        Returns:
            StepResult with contract validation info.
        """
        if not step.contract_spec:
            # No contract - execute directly
            output = await agent.run(step.reasoning_task or step.description, context)
            return StepResult(
                step_id=step.step_id,
                status=StepStatus.COMPLETED,
                output=output,
                tokens_used=0,  # Would need tracking
                duration_ms=0,
            )

        # Convert ContractSpec to Contract
        contract = step.contract_spec.to_contract(step.step_id)

        # Execute with contract
        result = await self.executor.execute_with_contract(
            agent=agent,
            task=step.reasoning_task or step.description,
            contract=contract,
            context=context,
        )

        return StepResult(
            step_id=step.step_id,
            status=StepStatus.COMPLETED if result.is_valid else StepStatus.FAILED_CONTRACT,
            output=result.output,
            tokens_used=result.tokens_used,
            duration_ms=result.metadata.get("execution_time_ms", 0),
            contract_errors=result.validation_result.get_failed_fields() if not result.is_valid else [],
        )

    def _needs_verification(self, step: PlanStep) -> bool:
        """Check if a tool step needs output verification."""
        # Tools that produce structured data need verification
        structured_tools = {"crm_update", "data_extract", "analysis"}
        return step.tool_name in structured_tools if step.tool_name else False

    def _produces_structured_output(self, step: PlanStep) -> bool:
        """Check if a reasoning step produces structured output."""
        structured_keywords = ["analyze", "score", "classify", "extract", "structure"]
        description = (step.reasoning_task or step.description).lower()
        return any(kw in description for kw in structured_keywords)
```

### 5.2 Reasoning Strategy Contract Awareness

```python
# sigil/contracts/integration/reasoning_bridge.py

from sigil.reasoning.manager import ReasoningManager
from sigil.contracts.schema import Contract

class ContractAwareReasoningManager(ReasoningManager):
    """Extends ReasoningManager with contract awareness.

    Contract requirements can influence:
    - Strategy selection (stricter contracts -> more thorough strategies)
    - Prompt construction (include contract requirements)
    - Confidence thresholds (contracts define success criteria)
    """

    def select_strategy_for_contract(
        self,
        task: str,
        contract: Contract | None,
        base_complexity: float,
    ) -> str:
        """Select strategy considering contract requirements.

        Args:
            task: The reasoning task.
            contract: Optional contract for output.
            base_complexity: Base complexity from router.

        Returns:
            Strategy name.
        """
        if not contract:
            return self.select_strategy(base_complexity)

        # Adjust complexity based on contract strictness
        adjusted_complexity = base_complexity

        # Strict contracts (FAIL strategy) need more thorough reasoning
        if contract.failure_strategy == FailureStrategy.FAIL:
            adjusted_complexity = min(adjusted_complexity + 0.2, 1.0)

        # Many deliverables = more complex
        deliverable_count = len(contract.deliverables)
        if deliverable_count > 5:
            adjusted_complexity = min(adjusted_complexity + 0.1, 1.0)

        # Complex validation rules = more complex
        total_rules = sum(len(d.validation_rules) for d in contract.deliverables)
        if total_rules > 10:
            adjusted_complexity = min(adjusted_complexity + 0.1, 1.0)

        return self.select_strategy(adjusted_complexity)

    def build_contract_aware_prompt(
        self,
        task: str,
        contract: Contract,
        context: ReasoningContext | None = None,
    ) -> str:
        """Build a prompt that includes contract requirements.

        Args:
            task: Base task description.
            contract: Contract with requirements.
            context: Additional reasoning context.

        Returns:
            Enhanced prompt with contract requirements.
        """
        # Get expected format from contract
        format_spec = self._build_format_spec(contract)

        # Get examples from contract deliverables
        examples = self._build_examples(contract)

        prompt = f"""{task}

OUTPUT REQUIREMENTS:
Your response must be a valid JSON object with the following structure:

{format_spec}

EXAMPLES:
{examples}

VALIDATION RULES:
{self._build_validation_rules(contract)}

Return ONLY the JSON object, no explanatory text."""

        return prompt

    def _build_format_spec(self, contract: Contract) -> str:
        """Build format specification from contract."""
        lines = []
        for d in contract.deliverables:
            required = "*" if d.required else ""
            lines.append(f"  {d.name}{required}: {d.type} - {d.description}")
        lines.append("\n(* = required)")
        return "\n".join(lines)

    def _build_examples(self, contract: Contract) -> str:
        """Build examples from deliverable examples."""
        example = {}
        for d in contract.deliverables:
            if d.example is not None:
                example[d.name] = d.example
        return json.dumps(example, indent=2)

    def _build_validation_rules(self, contract: Contract) -> str:
        """Build validation rules summary."""
        lines = []
        for d in contract.deliverables:
            if d.validation_rules:
                lines.append(f"- {d.name}: {', '.join(d.validation_rules)}")
        return "\n".join(lines) if lines else "No specific validation rules."
```

### 5.3 Contract Violations Trigger Replanning

```python
async def handle_contract_violation_in_plan(
    self,
    plan: Plan,
    failed_step: PlanStep,
    errors: list[str],
    planner: Planner,
) -> Plan | None:
    """Handle contract violation during plan execution.

    Options:
    1. Retry step with refined prompt
    2. Replan from failed step
    3. Skip step and continue
    4. Abort plan

    Args:
        plan: Current plan being executed.
        failed_step: Step that failed contract validation.
        errors: Validation error messages.
        planner: Planner instance for replanning.

    Returns:
        New plan if replanning, None if should continue with original.
    """
    # Check if step is critical (has downstream dependencies)
    dependents = [
        s for s in plan.steps
        if failed_step.step_id in s.dependencies
    ]

    if not dependents:
        # No dependents - can skip
        logger.warning(f"Skipping failed step {failed_step.step_id}, no dependents")
        return None

    # Check if errors are fixable
    fixable_errors = [e for e in errors if self._is_fixable_error(e)]
    if len(fixable_errors) == len(errors):
        # All errors fixable - retry with better prompt
        logger.info(f"Retrying step {failed_step.step_id} with refined prompt")
        return None

    # Need to replan
    logger.info(f"Replanning from step {failed_step.step_id}")
    new_goal = f"Continue from: {failed_step.description}, fixing: {errors}"
    new_plan = await planner.generate(
        goal=new_goal,
        context=f"Previous steps completed successfully. Failed at: {failed_step.step_id}",
    )

    return new_plan

def _is_fixable_error(self, error: str) -> bool:
    """Check if an error is fixable through retry."""
    fixable_patterns = ["missing", "type mismatch", "rule failed"]
    return any(pattern in error.lower() for pattern in fixable_patterns)
```

---

## 6. Token Budget Integration

### 6.1 Contract Token Budget Allocation

```python
# Token budget distribution for contract execution

@dataclass
class ContractTokenBudget:
    """Token budget allocation for contract execution.

    Default allocation from session budget:
    - Initial execution: 60%
    - Retry 1: 25%
    - Retry 2: 10%
    - Fallback: 5%
    """
    total_tokens: int
    initial_pct: float = 0.60
    retry1_pct: float = 0.25
    retry2_pct: float = 0.10
    fallback_pct: float = 0.05

    @property
    def initial_budget(self) -> int:
        return int(self.total_tokens * self.initial_pct)

    @property
    def retry1_budget(self) -> int:
        return int(self.total_tokens * self.retry1_pct)

    @property
    def retry2_budget(self) -> int:
        return int(self.total_tokens * self.retry2_pct)

    @property
    def fallback_budget(self) -> int:
        return int(self.total_tokens * self.fallback_pct)

    def budget_for_attempt(self, attempt: int) -> int:
        """Get token budget for specific attempt."""
        if attempt == 1:
            return self.initial_budget
        elif attempt == 2:
            return self.retry1_budget
        elif attempt == 3:
            return self.retry2_budget
        else:
            return self.fallback_budget

    def has_budget_for_retry(self, used: int, attempt: int) -> bool:
        """Check if there's budget for another retry."""
        remaining = self.total_tokens - used
        needed = self.budget_for_attempt(attempt + 1)
        return remaining >= needed
```

### 6.2 Token Tracking During Contract Execution

```python
class ContractTokenTracker:
    """Tracks token usage during contract execution."""

    def __init__(
        self,
        session_tracker: TokenTracker,
        budget: ContractTokenBudget,
    ):
        self.session_tracker = session_tracker
        self.budget = budget
        self.by_attempt: dict[int, int] = {}
        self.total_used = 0

    def record_attempt(
        self,
        attempt: int,
        input_tokens: int,
        output_tokens: int,
    ) -> None:
        """Record tokens for an attempt."""
        total = input_tokens + output_tokens
        self.by_attempt[attempt] = self.by_attempt.get(attempt, 0) + total
        self.total_used += total
        self.session_tracker.record_usage(input_tokens, output_tokens)

    def should_warn(self) -> bool:
        """Check if usage is approaching limit."""
        return self.total_used >= self.budget.total_tokens * 0.8

    def should_force_fallback(self) -> bool:
        """Check if we should force fallback due to budget."""
        return self.total_used >= self.budget.total_tokens * 0.95

    def get_remaining(self) -> int:
        """Get remaining tokens."""
        return max(0, self.budget.total_tokens - self.total_used)

    def get_utilization_report(self) -> dict:
        """Get utilization breakdown by attempt."""
        return {
            "total_budget": self.budget.total_tokens,
            "total_used": self.total_used,
            "utilization_pct": self.total_used / self.budget.total_tokens,
            "by_attempt": self.by_attempt,
            "remaining": self.get_remaining(),
        }
```

### 6.3 Budget Exhaustion Handling

```python
async def execute_with_budget_awareness(
    self,
    agent: AgentProtocol,
    task: str,
    contract: Contract,
    budget: ContractTokenBudget,
) -> ContractResult:
    """Execute with token budget awareness.

    If budget is exhausted before completion:
    1. Emit warning event
    2. Force fallback strategy
    3. Return partial/template result
    """
    tracker = ContractTokenTracker(self.session_tracker, budget)

    for attempt in range(1, contract.max_retries + 2):
        # Check budget before attempt
        if tracker.should_force_fallback():
            return self._force_fallback(contract, "budget_exhausted")

        if tracker.should_warn():
            self._emit_budget_warning(contract, tracker)

        # Execute with attempt budget
        attempt_budget = budget.budget_for_attempt(attempt)
        result = await self._execute_single_attempt(
            agent, task, contract, attempt_budget
        )

        # Record usage
        tracker.record_attempt(attempt, result.input_tokens, result.output_tokens)

        if result.is_valid:
            return result

        # Check if we can afford another retry
        if not tracker.has_budget_for_retry(tracker.total_used, attempt):
            return self._force_fallback(contract, "insufficient_budget_for_retry")

    return self._apply_fallback(contract, result)
```

---

## 7. Event Store Integration

### 7.1 Contract Event Types (Already Defined)

The existing `sigil/state/events.py` already defines contract event types:

```python
class EventType(Enum):
    # Contract events (Phase 6)
    CONTRACT_VALIDATION_STARTED = "contract.validation_started"
    CONTRACT_VALIDATED = "contract.validated"
    CONTRACT_VALIDATION_FAILED = "contract.validation_failed"
    CONTRACT_RETRY = "contract.retry"
    CONTRACT_FALLBACK = "contract.fallback"
    CONTRACT_COMPLETED = "contract.completed"
```

### 7.2 Enhanced Event Factory Functions

The existing factory functions are comprehensive. We ensure they're used consistently:

```python
# Complete event trail for a contract execution:

async def emit_contract_lifecycle_events(
    self,
    execution_id: str,
    contract: Contract,
    session_id: str,
    event_store: EventStore,
) -> None:
    """Emit complete event trail for contract execution.

    Event sequence:
    1. CONTRACT_VALIDATION_STARTED
    2. For each attempt:
       - CONTRACT_VALIDATED or CONTRACT_VALIDATION_FAILED
       - CONTRACT_RETRY (if retrying)
    3. CONTRACT_FALLBACK (if applicable)
    4. CONTRACT_COMPLETED
    """
    # Example emission pattern
    events = [
        create_contract_validation_started_event(
            session_id=session_id,
            contract_id=execution_id,
            contract_name=contract.name,
            deliverables=contract.get_deliverable_names(),
        )
    ]

    # ... execution happens ...

    # After completion
    events.append(
        create_contract_completed_event(
            session_id=session_id,
            contract_id=execution_id,
            contract_name=contract.name,
            applied_strategy="success",
            is_valid=True,
            attempts=1,
            tokens_used=1500,
            execution_time_ms=3200,
        )
    )

    # Batch append for efficiency
    event_store.append_batch(events)
```

### 7.3 Event Correlation

```python
def correlate_contract_events(
    events: list[Event],
    contract_id: str,
) -> list[Event]:
    """Get all events for a specific contract execution.

    Uses correlation_id to link related events.

    Args:
        events: All events to search.
        contract_id: Contract execution ID.

    Returns:
        Events related to this contract execution.
    """
    return [
        e for e in events
        if e.correlation_id == contract_id
        or e.payload.get("contract_id") == contract_id
    ]


def build_contract_execution_timeline(
    events: list[Event],
    contract_id: str,
) -> dict:
    """Build a timeline of a contract execution.

    Returns:
        Timeline with:
        - started_at: When validation started
        - attempts: List of attempt details
        - fallback_applied: Whether fallback was used
        - completed_at: When execution completed
        - total_duration_ms: Total execution time
    """
    related = correlate_contract_events(events, contract_id)
    related.sort(key=lambda e: e.timestamp)

    timeline = {
        "started_at": None,
        "attempts": [],
        "fallback_applied": False,
        "completed_at": None,
        "total_duration_ms": 0,
    }

    for event in related:
        if event.event_type == EventType.CONTRACT_VALIDATION_STARTED:
            timeline["started_at"] = event.timestamp

        elif event.event_type == EventType.CONTRACT_VALIDATED:
            timeline["attempts"].append({
                "attempt": event.payload.get("retry_count", 0) + 1,
                "passed": event.payload.get("passed"),
                "errors": event.payload.get("validation_errors", []),
                "timestamp": event.timestamp,
            })

        elif event.event_type == EventType.CONTRACT_VALIDATION_FAILED:
            timeline["attempts"].append({
                "attempt": event.payload.get("attempt"),
                "passed": False,
                "errors": event.payload.get("errors", []),
                "timestamp": event.timestamp,
            })

        elif event.event_type == EventType.CONTRACT_FALLBACK:
            timeline["fallback_applied"] = True
            timeline["fallback_strategy"] = event.payload.get("fallback_strategy")

        elif event.event_type == EventType.CONTRACT_COMPLETED:
            timeline["completed_at"] = event.timestamp
            timeline["total_duration_ms"] = event.payload.get("execution_time_ms")

    return timeline
```

---

## 8. Enhanced Contract Executor

### 8.1 Integrated Executor

The existing `ContractExecutor` is well-designed. We enhance it with cross-phase integration:

```python
# sigil/contracts/executor_enhanced.py

from sigil.contracts.executor import ContractExecutor, ContractResult
from sigil.contracts.integration.router_bridge import ContractSelector
from sigil.contracts.integration.memory_bridge import ContractMemoryBridge
from sigil.contracts.integration.plan_bridge import PlanContractBridge
from sigil.contracts.integration.reasoning_bridge import ContractAwareReasoningManager

class IntegratedContractExecutor(ContractExecutor):
    """Contract executor with full cross-phase integration.

    Enhances base executor with:
    - Router-based contract selection
    - Memory-backed context
    - Plan-aware execution
    - Reasoning strategy adaptation
    """

    def __init__(
        self,
        validator: ContractValidator | None = None,
        retry_manager: RetryManager | None = None,
        fallback_manager: FallbackManager | None = None,
        event_store: EventStore | None = None,
        memory_manager: MemoryManager | None = None,
        reasoning_manager: ReasoningManager | None = None,
    ):
        super().__init__(validator, retry_manager, fallback_manager, event_store)

        # Cross-phase bridges
        self.memory_bridge = ContractMemoryBridge(memory_manager) if memory_manager else None
        self.contract_selector = ContractSelector()
        self.reasoning_bridge = ContractAwareReasoningManager(reasoning_manager) if reasoning_manager else None

    async def execute_with_full_integration(
        self,
        agent: AgentProtocol,
        task: str,
        route_decision: RouteDecision,
        stratum: str | None = None,
        context: Any = None,
    ) -> ContractResult:
        """Execute with full cross-phase integration.

        Flow:
        1. Select contract based on route decision
        2. Retrieve memory context (common errors, examples)
        3. Build contract-aware prompt
        4. Execute with validation
        5. Store results in memory
        6. Emit all events

        Args:
            agent: Agent to execute.
            task: Task description.
            route_decision: Routing decision from Phase 3.
            stratum: Optional ACTi stratum.
            context: Additional context.

        Returns:
            ContractResult with full metadata.
        """
        # 1. Select contract
        contract = self.contract_selector.select_for_route(route_decision, stratum)

        if not contract:
            # No contract - execute directly
            output = await agent.run(task, context)
            return ContractResult(
                output=output,
                is_valid=True,
                attempts=1,
                tokens_used=0,
                validation_result=ValidationResult(is_valid=True),
                applied_strategy=AppliedStrategy.SUCCESS,
            )

        # 2. Retrieve memory context
        memory_context = None
        if self.memory_bridge:
            memory_context = await self.memory_bridge.get_contract_context(contract.name)

        # 3. Build enhanced prompt with contract awareness
        enhanced_task = task
        if self.reasoning_bridge and memory_context:
            enhanced_task = self.reasoning_bridge.build_contract_aware_prompt(
                task, contract, memory_context
            )

            # Add common error warnings
            if memory_context.get("common_errors"):
                enhanced_task += "\n\nCOMMON MISTAKES TO AVOID:\n"
                for err in memory_context["common_errors"][:3]:
                    enhanced_task += f"- {err['error']}\n"

        # 4. Execute with contract validation
        result = await self.execute_with_contract(
            agent=agent,
            task=enhanced_task,
            contract=contract,
            context=context,
        )

        # 5. Store results in memory
        if self.memory_bridge:
            await self.memory_bridge.store_validation_result(
                contract_name=contract.name,
                output=result.output,
                validation_passed=result.is_valid,
                errors=[str(e) for e in result.validation_result.errors],
                session_id=result.metadata.get("execution_id", "unknown"),
            )

            # Store successful outputs as examples
            if result.is_valid and result.applied_strategy == AppliedStrategy.SUCCESS:
                await self.memory_bridge.store_successful_output_as_example(
                    contract_name=contract.name,
                    output=result.output,
                )

        return result
```

---

## 9. Validation System Architecture

### 9.1 Existing Validator (Well-Designed)

The existing `ContractValidator` in `sigil/contracts/validator.py` provides:

- Type checking with `TYPE_MAP`
- Rule compilation with safe eval
- Constraint validation
- Error aggregation with severity levels
- Suggestion generation

### 9.2 Enhanced Validation Flow

```
Output received
     |
     v
+----+----+
|Deliverable 1|
+----+----+
     |
     +---> Check presence (required?)
     |     |
     |     v
     +---> Check type (TYPE_MAP)
     |     |
     |     v
     +---> Run validation rules
     |     |
     |     v
     +---> Collect errors/warnings
     |
     v
+----+----+
|Deliverable 2|
+----+----+
     |
     (same flow)
     |
     v
+----+----+
|Constraints|
+----+----+
     |
     +---> Token limits
     +---> Tool call limits
     +---> Timeout (if applicable)
     |
     v
+----+----+
|ValidationResult|
|  - is_valid     |
|  - errors       |
|  - warnings     |
|  - suggestion   |
|  - partial_output|
+----------------+
```

### 9.3 Rule Engine Details

```python
# Safe evaluation context for validation rules
SAFE_BUILTINS = {
    "__builtins__": {},
    "len": len,
    "isinstance": isinstance,
    "int": int,
    "float": float,
    "str": str,
    "bool": bool,
    "list": list,
    "dict": dict,
    "tuple": tuple,
    "set": set,
    "True": True,
    "False": False,
    "None": None,
    "min": min,
    "max": max,
    "abs": abs,
    "all": all,
    "any": any,
}

# Example validation rules and their compilation:
RULE_EXAMPLES = {
    "0 <= value <= 100": lambda v: 0 <= v <= 100,
    "len(value) > 0": lambda v: len(v) > 0,
    "value.startswith('http')": lambda v: v.startswith('http'),
    "'budget' in value": lambda v: 'budget' in v,
    "isinstance(value, dict)": lambda v: isinstance(v, dict),
}
```

---

## 10. Retry System Architecture

### 10.1 Existing RetryManager (Well-Designed)

The existing `RetryManager` in `sigil/contracts/retry.py` provides:

- Progressive prompt refinement (3 levels)
- Token estimation for retries
- Recoverable error detection
- Error summary generation

### 10.2 Retry Decision Matrix

| Condition | Retry? | Reason |
|-----------|--------|--------|
| attempt >= max_retries | No | Retry limit reached |
| tokens_remaining < MIN_TOKENS | No | Budget exhausted |
| No recoverable errors | No | Errors can't be fixed |
| All conditions met | Yes | Worth trying again |

### 10.3 Prompt Refinement Progression

```
Level 1 (First Retry):
+------------------------+
| Original Task          |
+------------------------+
| IMPORTANT - Previous   |
| attempt had validation |
| errors:                |
| - field: reason        |
| Please fix these issues|
+------------------------+

Level 2 (Second Retry):
+------------------------+
| SIMPLIFIED TASK        |
+------------------------+
| Previous errors summary|
+------------------------+
| MUST BE VALID JSON:    |
| - field1*: type        |
| - field2*: type        |
+------------------------+
| CRITICAL INSTRUCTIONS  |
| 1. Return ONLY JSON    |
| 2. Include ALL fields  |
| 3. Correct types       |
+------------------------+

Level 3 (Third Retry):
+------------------------+
| FINAL ATTEMPT          |
+------------------------+
| Task description       |
+------------------------+
| EXACT STRUCTURE:       |
| ```json                |
| {                      |
|   "field1": example,   |
|   "field2": example    |
| }                      |
| ```                    |
+------------------------+
| FILL IN VALUES         |
| Keep structure exact   |
+------------------------+
```

---

## 11. Fallback System Architecture

### 11.1 Existing FallbackManager (Well-Designed)

The existing `FallbackManager` in `sigil/contracts/fallback.py` provides:

- Partial result building
- Template result generation
- Strategy selection logic
- Meaningful value detection

### 11.2 Fallback Strategy Selection

```
+------------------------+
| Contract Strategy Check|
+------------------------+
         |
    +---------+
    | FAIL?   |
    +---------+
     |       |
    YES     NO
     |       |
     v       v
+-------+ +----------+
|ESCALATE| | Can Build|
|Return  | | Partial? |
+-------+ +----------+
             |    |
           YES   NO
             |    |
             v    v
        +-------+ +--------+
        |PARTIAL| |TEMPLATE|
        +-------+ +--------+
```

### 11.3 Partial vs Template Coverage

| Scenario | Partial Fields | Template Fields | Strategy |
|----------|----------------|-----------------|----------|
| 80%+ valid | 4/5 | 1/5 | PARTIAL |
| 50% valid | 2/5 | 3/5 | PARTIAL |
| <50% valid | 1/5 | 4/5 | TEMPLATE |
| 0% valid | 0/5 | 5/5 | TEMPLATE |
| Strategy=FAIL | N/A | N/A | ESCALATE |

---

## 12. Contract Storage Architecture

### 12.1 Storage Layers

```
Layer 1: Built-in Templates (Code)
+----------------------------------+
| sigil/contracts/templates/acti.py|
| - lead_qualification_contract()  |
| - research_report_contract()     |
| - appointment_booking_contract() |
| - market_analysis_contract()     |
| - compliance_check_contract()    |
+----------------------------------+
              |
              v (load at startup)
Layer 2: Runtime Cache (Memory)
+----------------------------------+
| ContractCache                    |
| - TTL: 1 hour                    |
| - LRU eviction                   |
| - Key: contract_name + version   |
+----------------------------------+
              |
              v (persist for versioning)
Layer 3: Memory System (Phase 4)
+----------------------------------+
| MemoryManager                    |
| - Resource type: contract_template|
| - Versioned storage              |
| - Historical tracking            |
+----------------------------------+
```

### 12.2 Template Lifecycle

```
1. DEFINE (Development)
   - Code in templates/acti.py
   - Includes deliverables, constraints, examples

2. LOAD (Runtime)
   - Lazy loading via get_template()
   - Cached in ContractCache

3. CUSTOMIZE (Per-Request)
   - Adjusted for complexity
   - Constraints modified

4. VERSION (Memory)
   - Stored via memory_bridge
   - Old versions archived

5. EVOLVE (Learning)
   - Analyze validation failures
   - Suggest improvements
   - A/B test new versions
```

---

## 13. Concurrency and Performance

### 13.1 Performance Characteristics

| Operation | Expected Time | Notes |
|-----------|---------------|-------|
| Contract validation | 50-100ms | Type checks + rule eval |
| Rule compilation | 5-10ms | Cached after first compile |
| Prompt refinement | <50ms | String manipulation |
| Re-execution | 1-5s | Same as original agent run |
| Fallback build | 20-50ms | Dictionary operations |
| Template generation | <100ms | From contract structure |

### 13.2 Caching Strategy

```python
class ContractCache:
    """In-memory cache for compiled contracts and rules."""

    def __init__(self, ttl_hours: int = 1):
        self.ttl = timedelta(hours=ttl_hours)
        self._contracts: dict[str, tuple[Contract, datetime]] = {}
        self._compiled_rules: dict[str, list[Callable]] = {}
        self._lock = threading.Lock()

    def get_contract(self, name: str) -> Contract | None:
        """Get cached contract."""
        with self._lock:
            if name in self._contracts:
                contract, cached_at = self._contracts[name]
                if datetime.now() - cached_at < self.ttl:
                    return contract
                else:
                    del self._contracts[name]
        return None

    def set_contract(self, name: str, contract: Contract) -> None:
        """Cache a contract."""
        with self._lock:
            self._contracts[name] = (contract, datetime.now())

    def get_compiled_rules(self, rule_key: str) -> list[Callable] | None:
        """Get cached compiled rules."""
        return self._compiled_rules.get(rule_key)

    def set_compiled_rules(self, rule_key: str, rules: list[Callable]) -> None:
        """Cache compiled rules."""
        self._compiled_rules[rule_key] = rules
```

### 13.3 Concurrency Constraints

```
Contract Execution: SEQUENTIAL
- Each attempt depends on previous result
- Validation depends on output
- Prompt refinement depends on errors

Validation: NO PARALLELIZATION NEEDED
- Fast operation (<100ms)
- CPU-bound, not I/O-bound

Fallback: SEQUENTIAL
- Single code path
- Fast operation (<50ms)
```

---

## 14. Error Handling and Recovery

### 14.1 Error Categories

```python
class ContractErrorCategory(Enum):
    """Categories of contract errors for handling."""

    VALIDATION = "validation"      # Output doesn't match spec
    EXECUTION = "execution"        # Agent failed to produce output
    BUDGET = "budget"              # Token budget exceeded
    TIMEOUT = "timeout"            # Execution timed out
    SYSTEM = "system"              # Internal system error
```

### 14.2 Error Recovery Matrix

| Error Type | Recovery Action | Guarantee |
|------------|----------------|-----------|
| Missing deliverable | Retry with explicit requirement | Retry up to max |
| Type mismatch | Retry with format example | Retry up to max |
| Rule violation | Retry with rule explanation | Retry up to max |
| Agent execution error | Apply fallback | Return template |
| Token budget exceeded | Force fallback | Return partial/template |
| Timeout | Return best so far | Partial result |
| Template generation fails | Return empty + warnings | Always returns |

### 14.3 Recovery Guarantees

```python
async def execute_with_guaranteed_result(
    self,
    agent: AgentProtocol,
    task: str,
    contract: Contract,
) -> ContractResult:
    """Execute with guarantee of returning a result.

    NEVER raises exceptions - always returns ContractResult.
    Worst case: Returns template output with warnings.
    """
    try:
        return await self.execute_with_contract(agent, task, contract)

    except ContractValidationError as e:
        # Expected failure - return with metadata
        return ContractResult(
            output=self.fallback_manager.build_template_result(contract).output,
            is_valid=False,
            attempts=e.context.get("attempts", 0),
            tokens_used=e.context.get("tokens_used", 0),
            validation_result=ValidationResult(
                is_valid=False,
                errors=[ValidationError(field="_", reason=str(e))],
            ),
            applied_strategy=AppliedStrategy.FAIL,
            metadata={"error": str(e)},
        )

    except Exception as e:
        # Unexpected error - still return something
        logger.error(f"Unexpected error in contract execution: {e}")
        return ContractResult(
            output=contract.get_template_output(),
            is_valid=False,
            attempts=0,
            tokens_used=0,
            validation_result=ValidationResult(
                is_valid=False,
                errors=[ValidationError(field="_system", reason=f"System error: {e}")],
            ),
            applied_strategy=AppliedStrategy.FALLBACK,
            metadata={
                "error": str(e),
                "fallback_reason": "system_error",
            },
        )
```

---

## 15. Learning Integration

### 15.1 Learning Signals

```python
@dataclass
class ContractLearningSignal:
    """Learning signal from contract execution."""

    contract_name: str
    success_first_try: bool
    retry_count: int
    error_types: list[str]  # ["missing", "type_mismatch", "rule_violation"]
    strategy_used: str
    tokens_used: int
    execution_time_ms: float
    timestamp: datetime


class ContractLearningCollector:
    """Collects and analyzes contract execution signals."""

    async def record_execution(
        self,
        result: ContractResult,
        contract: Contract,
    ) -> None:
        """Record execution for learning."""
        signal = ContractLearningSignal(
            contract_name=contract.name,
            success_first_try=result.applied_strategy == AppliedStrategy.SUCCESS,
            retry_count=result.attempts - 1,
            error_types=self._categorize_errors(result.validation_result.errors),
            strategy_used=result.applied_strategy.value,
            tokens_used=result.tokens_used,
            execution_time_ms=result.metadata.get("execution_time_ms", 0),
            timestamp=datetime.now(timezone.utc),
        )

        await self._store_signal(signal)

    async def analyze_contract_performance(
        self,
        contract_name: str,
        window_days: int = 7,
    ) -> dict:
        """Analyze performance of a contract over time.

        Returns:
            Performance metrics:
            - first_try_success_rate
            - avg_retries
            - common_error_types
            - avg_tokens
            - suggested_improvements
        """
        signals = await self._get_signals(contract_name, window_days)

        if not signals:
            return {"status": "no_data"}

        first_try_success = sum(1 for s in signals if s.success_first_try) / len(signals)
        avg_retries = sum(s.retry_count for s in signals) / len(signals)

        error_types = Counter()
        for s in signals:
            error_types.update(s.error_types)

        return {
            "first_try_success_rate": first_try_success,
            "avg_retries": avg_retries,
            "common_error_types": error_types.most_common(5),
            "avg_tokens": sum(s.tokens_used for s in signals) / len(signals),
            "suggested_improvements": self._suggest_improvements(
                first_try_success, error_types
            ),
        }

    def _suggest_improvements(
        self,
        success_rate: float,
        error_types: Counter,
    ) -> list[str]:
        """Suggest contract improvements based on data."""
        suggestions = []

        if success_rate < 0.5:
            suggestions.append("Consider adding more explicit examples to deliverables")

        if error_types.get("type_mismatch", 0) > error_types.total() * 0.3:
            suggestions.append("Type mismatches common - add explicit type examples")

        if error_types.get("missing", 0) > error_types.total() * 0.3:
            suggestions.append("Missing fields common - simplify required deliverables")

        if error_types.get("rule_violation", 0) > error_types.total() * 0.3:
            suggestions.append("Rule violations common - review validation rules")

        return suggestions
```

### 15.2 Contract Evolution (Phase 7 Preview)

```python
async def suggest_contract_improvements(
    self,
    contract: Contract,
    performance_data: dict,
) -> Contract:
    """Suggest improved contract based on performance data.

    Phase 7 feature - uses learning to evolve contracts.
    """
    # Analyze common failures
    if performance_data["common_error_types"]:
        top_error = performance_data["common_error_types"][0][0]

        if top_error == "missing":
            # Add better examples
            for d in contract.deliverables:
                if d.example is None:
                    d.example = self._generate_example(d)

        elif top_error == "type_mismatch":
            # Add explicit type hints to descriptions
            for d in contract.deliverables:
                d.description = f"[{d.type}] {d.description}"

        elif top_error == "rule_violation":
            # Relax overly strict rules
            for d in contract.deliverables:
                d.validation_rules = [
                    r for r in d.validation_rules
                    if self._rule_pass_rate(r) > 0.3
                ]

    return contract
```

---

## 16. Monitoring and Observability

### 16.1 Metrics to Track

```python
@dataclass
class ContractMetrics:
    """Phase 6 contract metrics."""

    # Validation metrics
    total_validations: int = 0
    first_try_pass_rate: float = 0.0
    retry_rate: float = 0.0
    fallback_rate: float = 0.0
    fail_rate: float = 0.0

    # Performance metrics
    avg_attempts: float = 0.0
    avg_tokens_per_contract: float = 0.0
    avg_validation_time_ms: float = 0.0

    # Error distribution
    error_by_type: dict[str, int] = field(default_factory=dict)
    error_by_contract: dict[str, int] = field(default_factory=dict)

    # Per-contract metrics
    contract_success_rates: dict[str, float] = field(default_factory=dict)
```

### 16.2 Dashboard Specifications

**Dashboard 1: Phase 6 Overview**

| Panel | Metric | Visualization |
|-------|--------|---------------|
| Validation Pass Rate | first_try_pass_rate | Gauge (0-100%) |
| Retry/Fallback/Fail | retry_rate, fallback_rate, fail_rate | Stacked bar |
| Avg Attempts | avg_attempts | Line chart over time |
| Token Efficiency | avg_tokens_per_contract | Line chart |

**Dashboard 2: Contract Performance**

| Panel | Metric | Visualization |
|-------|--------|---------------|
| Per-Contract Success | contract_success_rates | Heatmap |
| Error Distribution | error_by_type | Pie chart |
| Retry Progression | retries by level | Funnel |

**Dashboard 3: Error Analysis**

| Panel | Metric | Visualization |
|-------|--------|---------------|
| Top Errors | error_by_type | Bar chart |
| Error Trends | errors over time | Line chart |
| Recovery Success | retry success after error type | Grouped bar |

### 16.3 Structured Logging

```python
import structlog

logger = structlog.get_logger(__name__)

# Validation logging
logger.info(
    "contract_validation",
    contract_name=contract.name,
    passed=result.is_valid,
    errors=len(result.errors),
    validation_time_ms=result.validation_time_ms,
    deliverables_checked=result.deliverables_checked,
)

# Retry logging
logger.info(
    "contract_retry",
    contract_name=contract.name,
    attempt=attempt,
    max_retries=contract.max_retries,
    error_count=len(errors),
    refinement_level=attempt,
)

# Fallback logging
logger.warning(
    "contract_fallback",
    contract_name=contract.name,
    strategy=fallback_result.strategy.value,
    partial_fields=fallback_result.partial_fields,
    template_fields=fallback_result.template_fields,
    warnings=fallback_result.warnings,
)

# Completion logging
logger.info(
    "contract_completed",
    contract_name=contract.name,
    applied_strategy=result.applied_strategy.value,
    attempts=result.attempts,
    tokens_used=result.tokens_used,
    execution_time_ms=result.metadata.get("execution_time_ms"),
)
```

---

## 17. Configuration

### 17.1 Global Settings

```python
# sigil/config/settings.py additions

class ContractSettings(BaseModel):
    """Contract system configuration."""

    # Global defaults
    default_failure_strategy: str = "retry"  # retry, fallback, fail
    default_max_retries: int = 2
    warn_threshold: float = 0.8  # 80% of budget triggers warning
    strict_mode: bool = False  # Treat warnings as errors

    # Retry configuration
    min_tokens_for_retry: int = 500
    retry_delay_base_ms: int = 0  # No delay by default
    retry_delay_multiplier: float = 1.0  # Exponential backoff

    # Fallback configuration
    min_partial_coverage: float = 0.5  # 50% valid = can use partial

    # Cache configuration
    cache_ttl_hours: int = 1
    cache_max_entries: int = 100

    # Learning configuration
    enable_learning: bool = True
    learning_window_days: int = 7
```

### 17.2 Per-Contract Overrides

```python
# Contracts can override global settings via metadata

contract = Contract(
    name="critical_compliance",
    # ... deliverables ...
    failure_strategy=FailureStrategy.FAIL,  # Override global
    max_retries=0,  # No retries for compliance
    metadata={
        "override_strict_mode": True,
        "custom_warn_threshold": 0.5,
    },
)
```

### 17.3 Runtime Overrides

```python
# At execution time, can override contract behavior

result = await executor.execute_with_contract(
    agent=agent,
    task=task,
    contract=contract,
    token_tracker=tracker,
    overrides={
        "skip_validation": False,  # Never skip
        "force_strategy": None,  # Use contract default
        "max_retries_override": None,  # Use contract default
    },
)
```

---

## 18. Implementation Checklist

### 18.1 Must Have (Phase 6 Complete)

- [x] Contract schema definitions (`schema.py`)
- [x] ContractValidator with type checking and rules (`validator.py`)
- [x] RetryManager with progressive refinement (`retry.py`)
- [x] FallbackManager with partial/template strategies (`fallback.py`)
- [x] ContractExecutor orchestration (`executor.py`)
- [x] ACTi contract templates (`templates/acti.py`)
- [x] Event types for contracts (`state/events.py`)
- [x] Contract-related event factory functions

### 18.2 Integration Layer (Phase 6.5)

- [ ] Router -> Contract selection bridge (`integration/router_bridge.py`)
- [ ] Memory -> Contract storage bridge (`integration/memory_bridge.py`)
- [ ] Plan step -> Contract verification bridge (`integration/plan_bridge.py`)
- [ ] Reasoning -> Contract awareness bridge (`integration/reasoning_bridge.py`)
- [ ] Token budget integration for contracts
- [ ] Enhanced event emission in executor

### 18.3 Nice to Have (Future)

- [ ] Contract templates in memory system with versioning
- [ ] Learning signals collection and analysis
- [ ] A/B testing different contract versions
- [ ] Contract migration utilities
- [ ] Contract performance dashboard

---

## Summary

This architecture ensures Phase 6 contracts are:

1. **Reliable**: Recovery guarantees ensure always returning a result (never crash)
2. **Efficient**: Token budget awareness, caching, early fallback when appropriate
3. **Observable**: Complete event trail, structured logging, metrics collection
4. **Learnable**: Validation results stored for pattern analysis and improvement
5. **Integrated**: Seamless interaction with Routing, Memory, Planning, and Reasoning phases

The existing implementation provides a solid foundation. The integration layer detailed in this document connects contracts to the broader Sigil v2 ecosystem, enabling intelligent contract selection, memory-backed context, and reasoning-aware execution.

---

## Phase 7 Integration: Orchestrator

Phase 6 contracts are integrated into the unified `SigilOrchestrator` introduced in Phase 7.
The orchestrator coordinates contract validation as part of the request processing pipeline.

### Orchestrator Contract Integration

```
┌─────────────────────────────────────────────────────────────────┐
│                     SigilOrchestrator                           │
│                                                                 │
│  Request Processing Pipeline:                                   │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                                                         │   │
│  │  1. Request Validation                                  │   │
│  │  2. Context Assembly (ContextManager)                   │   │
│  │  3. Routing Decision (Router)                           │   │
│  │  4. Execution (Reasoning/Planning)                      │   │
│  │  5. CONTRACT VALIDATION (Phase 6)  <── Integration      │   │
│  │     - ContractExecutor.execute()                        │   │
│  │     - ValidationResult checked                          │   │
│  │     - Retry if contract violated                        │   │
│  │     - Fallback if retries exhausted                     │   │
│  │  6. Response Assembly                                   │   │
│  │                                                         │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Contract Validation Flow

```
Orchestrator.handle()
    │
    ├── Execute task/reasoning
    │   └── Raw output generated
    │
    ├── Check if contract specified
    │   └── if agent.contract_id is set
    │
    ├── ContractExecutor.execute()
    │   ├── Validate output against contract
    │   ├── If valid: return result
    │   └── If invalid:
    │       ├── Retry (up to max_retries)
    │       └── Fallback (if retries exhausted)
    │
    └── Include validation_result in response
```

### Key Integration Points

| Phase 6 Component | Orchestrator Usage | When Invoked |
|-------------------|-------------------|--------------|
| ContractExecutor | `_validate_output()` | After execution, if contract specified |
| ValidationResult | Included in response | Always (if contract exists) |
| RetryManager | Automatic retry | On validation failure |
| FallbackManager | Graceful degradation | When retries exhausted |

### Token Budget for Contracts

The orchestrator reserves tokens for contract-related operations:

- **Validation overhead**: 500-1000 tokens per validation
- **Retry attempts**: Each retry uses additional reasoning tokens
- **Fallback generation**: Up to 2000 tokens for fallback output

Total contract budget: Up to 5% of session budget (12.8K tokens for 256K session)

### Contract Events in Orchestrator Pipeline

```
OrchestratorRequestReceived
    └── [Execution events...]
    └── ContractValidationStarted
        ├── ContractValidationSucceeded
        │   └── OrchestratorResponseSent
        └── ContractValidationFailed
            └── ContractRetryAttempted (if retrying)
                └── ContractValidationSucceeded (on retry success)
            └── ContractFallbackTriggered (if retries exhausted)
                └── OrchestratorResponseSent (with fallback)
```

### API Endpoints

Contract management is exposed through the Phase 7 REST API:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/contracts` | POST | Create new contract |
| `/api/v1/contracts/{id}` | GET | Get contract details |
| `/api/v1/contracts/{id}` | PUT | Update contract |
| `/api/v1/contracts/{id}` | DELETE | Delete contract |
| `/api/v1/agents/{id}/contract` | PUT | Assign contract to agent |

### Related Documentation

- **Phase 7 Architecture**: `/docs/phase7-integration-architecture.md`
- **Phase 7 API Contract**: `/docs/api-contract-phase7-integration.md`
- **Phase 7 OpenAPI Spec**: `/docs/openapi-phase7-integration.yaml`
- **Phase 7 API Guidelines**: `/docs/api-guidelines-phase7.md`

---

*Document Version: 1.1.0*
*Last Updated: 2026-01-11*
*Author: Systems Architecture Team*
