# API Contract: Phase 6 Contracts System

## Overview

This document defines the comprehensive API contract for Sigil v2's Phase 6 Contracts System. The Contracts system provides formal specifications for agent outputs, ensuring deliverables meet requirements through validation, retry logic, and graceful degradation.

**Key Capabilities:**
- **Contract Definition**: Formal specifications for agent output structure and constraints
- **Contract Validation**: Runtime verification of outputs against specifications
- **Retry Management**: Intelligent retry with progressive refinement on validation failures
- **Fallback Strategies**: Graceful degradation when validation cannot be satisfied
- **Template Factory**: Pre-built contract templates for common ACTi workflows
- **Tool Integration**: LangChain-compatible tools for agent-driven contract enforcement

**Dependencies:**
- Phase 3: Router, TokenTracker, EventStore
- Phase 4: MemoryManager (for context retrieval during retry refinement)
- Phase 5: Planning, ReasoningManager (contract execution orchestration)

---

## Table of Contents

1. [Design Principles](#1-design-principles)
2. [Module Structure](#2-module-structure)
3. [Contract Definition API](#3-contract-definition-api)
4. [Contract Validation API](#4-contract-validation-api)
5. [Contract Execution API](#5-contract-execution-api)
6. [Retry Management API](#6-retry-management-api)
7. [Fallback Strategy API](#7-fallback-strategy-api)
8. [Template Factory API](#8-template-factory-api)
9. [Event Contract](#9-event-contract)
10. [Error Taxonomy](#10-error-taxonomy)
11. [Tool Integration](#11-tool-integration)
12. [Agent Integration](#12-agent-integration)
13. [Configuration API](#13-configuration-api)
14. [Data Schema Contracts](#14-data-schema-contracts)
15. [Integration Examples](#15-integration-examples)
16. [Implementation Guidance](#16-implementation-guidance)

---

## 1. Design Principles

### 1.1 Core Philosophy

1. **Explicit Over Implicit**: Every output requirement is formally specified
2. **Fail-Safe Execution**: Always return something useful - fallback over failure
3. **Progressive Refinement**: Retry with increasing specificity on validation failures
4. **Traceability**: Full audit trail of validation attempts and strategies applied
5. **Agent-Agnostic**: Contracts work with any agent implementation
6. **Composable**: Contracts can be combined and extended

### 1.2 When to Use Contracts

| Scenario | Use Contract? | Rationale |
|----------|---------------|-----------|
| Output used downstream | YES | Downstream systems need predictable structure |
| User-facing deliverable | YES | Quality assurance for end users |
| Critical business logic | YES | Compliance and audit requirements |
| Internal/recoverable | OPTIONAL | Lower stakes, can handle varied outputs |
| Exploratory/creative | NO | Contracts constrain creative freedom |

### 1.3 Contract vs. Schema Validation

```
Contracts are MORE than schema validation:

Schema Validation: "Is this JSON valid?"
Contract Validation: "Does this output satisfy the business requirement?"

Contracts include:
- Type checking (like schemas)
- Business rule validation
- Constraint enforcement (tokens, time, etc.)
- Semantic validation rules
- Recovery strategies
```

---

## 2. Module Structure

```
sigil/contracts/
|-- __init__.py                 # Module exports: Contract, ContractExecutor, etc.
|-- schema.py                   # Contract, Deliverable, Constraints dataclasses
|-- validator.py                # ContractValidator: validates outputs against contracts
|-- executor.py                 # ContractExecutor: executes with validation and retry
|-- retry.py                    # RetryManager: handles retry logic and refinement
|-- fallback.py                 # FallbackHandler: graceful degradation strategies
|-- templates/
|   |-- __init__.py             # Template exports
|   |-- factory.py              # get_contract_template() factory function
|   |-- acti.py                 # ACTi-specific templates
|   |-- common.py               # Common reusable templates
|-- tools/
|   |-- __init__.py             # Tool exports
|   |-- langchain.py            # LangChain tool implementations
```

---

## 3. Contract Definition API

### 3.1 Deliverable Schema

```python
from dataclasses import dataclass, field
from typing import Optional, Any, Callable
from enum import Enum


class DeliverableType(str, Enum):
    """Supported deliverable types."""
    STRING = "str"
    INTEGER = "int"
    FLOAT = "float"
    BOOLEAN = "bool"
    LIST = "list"
    DICT = "dict"
    ANY = "any"


@dataclass
class Deliverable:
    """Specification for a single output field.

    A deliverable defines what an agent must produce as part of its output.
    Each deliverable has a name, type, and optional validation rules.

    Attributes:
        name: Field name in output dict (e.g., "qualification_score")
        type: Expected data type (e.g., "int", "str", "dict")
        description: Human-readable description for documentation
        required: Whether this field must be present (default True)
        validation_rules: List of validation expressions (e.g., "value >= 0")
        example: Example value for documentation and fallback
        default: Default value if missing and not required
        nested_schema: For dict/list types, nested deliverable specs

    Example:
        >>> score = Deliverable(
        ...     name="qualification_score",
        ...     type="int",
        ...     description="Lead qualification score from 0-100",
        ...     required=True,
        ...     validation_rules=["value >= 0", "value <= 100"],
        ...     example=75,
        ... )
    """
    name: str
    type: str  # DeliverableType value
    description: str
    required: bool = True
    validation_rules: list[str] = field(default_factory=list)
    example: Optional[Any] = None
    default: Optional[Any] = None
    nested_schema: Optional[list["Deliverable"]] = None

    def validate_type(self, value: Any) -> bool:
        """Check if value matches expected type."""
        type_map = {
            "str": str,
            "int": int,
            "float": (int, float),
            "bool": bool,
            "list": list,
            "dict": dict,
            "any": object,
        }
        expected = type_map.get(self.type, object)
        return isinstance(value, expected)

    def validate_rules(self, value: Any) -> list[str]:
        """Validate value against all rules. Returns list of failed rules."""
        failed = []
        for rule in self.validation_rules:
            try:
                # Safe evaluation with value in scope
                if not eval(rule, {"value": value, "__builtins__": {}}):
                    failed.append(rule)
            except Exception as e:
                failed.append(f"{rule} (evaluation error: {e})")
        return failed
```

### 3.2 ContractConstraints Schema

```python
@dataclass
class ContractConstraints:
    """Resource constraints for contract execution.

    Constraints define the resource limits within which the contract
    must be satisfied. Exceeding constraints triggers appropriate
    handling (warning at threshold, failure at limit).

    Attributes:
        max_input_tokens: Maximum input tokens per attempt
        max_output_tokens: Maximum output tokens per attempt
        max_total_tokens: Maximum total tokens across all attempts
        max_tool_calls: Maximum tool calls allowed
        timeout_seconds: Maximum execution time
        warn_threshold: Percentage of limit to trigger warning (default 0.8)

    Example:
        >>> constraints = ContractConstraints(
        ...     max_total_tokens=5000,
        ...     max_tool_calls=5,
        ...     timeout_seconds=60,
        ...     warn_threshold=0.8,
        ... )
    """
    max_input_tokens: Optional[int] = None
    max_output_tokens: Optional[int] = None
    max_total_tokens: Optional[int] = None
    max_tool_calls: Optional[int] = None
    timeout_seconds: Optional[int] = None
    warn_threshold: float = 0.8

    def check_tokens(self, used: int) -> tuple[bool, bool]:
        """Check token usage. Returns (exceeded, warning)."""
        if self.max_total_tokens is None:
            return False, False
        exceeded = used > self.max_total_tokens
        warning = used > (self.max_total_tokens * self.warn_threshold)
        return exceeded, warning

    def check_tool_calls(self, count: int) -> tuple[bool, bool]:
        """Check tool call count. Returns (exceeded, warning)."""
        if self.max_tool_calls is None:
            return False, False
        exceeded = count > self.max_tool_calls
        warning = count > (self.max_tool_calls * self.warn_threshold)
        return exceeded, warning
```

### 3.3 Contract Schema

```python
class FailureStrategy(str, Enum):
    """Strategies for handling validation failures."""
    RETRY = "retry"          # Retry with refined prompt
    FALLBACK = "fallback"    # Return partial/template result
    PARTIAL = "partial"      # Same as fallback
    TEMPLATE = "template"    # Generate from contract template
    ESCALATE = "escalate"    # Raise exception
    FAIL = "fail"            # Same as escalate


@dataclass
class Contract:
    """Complete contract specification for agent output.

    A Contract defines the full specification for what an agent must
    produce, including structure, constraints, and failure handling.

    Attributes:
        name: Unique contract identifier (e.g., "lead_qualification")
        description: Human-readable contract description
        deliverables: List of required output fields
        constraints: Resource constraints for execution
        failure_strategy: How to handle validation failures
        max_retries: Maximum retry attempts (default 2)
        version: Contract version for compatibility
        metadata: Additional metadata (tags, owner, etc.)

    Example:
        >>> contract = Contract(
        ...     name="lead_qualification",
        ...     description="Qualify a sales lead using BANT methodology",
        ...     deliverables=[
        ...         Deliverable(
        ...             name="qualification_score",
        ...             type="int",
        ...             description="0-100 score",
        ...             validation_rules=["value >= 0", "value <= 100"],
        ...         ),
        ...         Deliverable(
        ...             name="bant_assessment",
        ...             type="dict",
        ...             description="BANT breakdown",
        ...         ),
        ...         Deliverable(
        ...             name="recommended_action",
        ...             type="str",
        ...             description="Next step recommendation",
        ...         ),
        ...     ],
        ...     constraints=ContractConstraints(
        ...         max_total_tokens=5000,
        ...         max_tool_calls=5,
        ...     ),
        ...     failure_strategy=FailureStrategy.RETRY,
        ...     max_retries=2,
        ... )
    """
    name: str
    description: str
    deliverables: list[Deliverable]
    constraints: ContractConstraints = field(default_factory=ContractConstraints)
    failure_strategy: FailureStrategy = FailureStrategy.RETRY
    max_retries: int = 2
    version: str = "1.0.0"
    metadata: dict[str, Any] = field(default_factory=dict)

    def get_deliverable(self, name: str) -> Optional[Deliverable]:
        """Get deliverable by name."""
        for d in self.deliverables:
            if d.name == name:
                return d
        return None

    def get_required_fields(self) -> list[str]:
        """Get list of required field names."""
        return [d.name for d in self.deliverables if d.required]

    def get_optional_fields(self) -> list[str]:
        """Get list of optional field names."""
        return [d.name for d in self.deliverables if not d.required]

    def to_prompt_context(self) -> str:
        """Generate prompt context describing expected output format."""
        lines = [
            f"## Required Output Format",
            f"",
            f"Your output must be a JSON object with the following fields:",
            f"",
        ]

        for d in self.deliverables:
            req = "REQUIRED" if d.required else "OPTIONAL"
            lines.append(f"- **{d.name}** ({d.type}, {req}): {d.description}")
            if d.validation_rules:
                lines.append(f"  - Validation: {', '.join(d.validation_rules)}")
            if d.example is not None:
                lines.append(f"  - Example: {d.example}")

        return "\n".join(lines)

    def generate_template_output(self) -> dict[str, Any]:
        """Generate a template output from contract spec."""
        output = {}
        for d in self.deliverables:
            if d.example is not None:
                output[d.name] = d.example
            elif d.default is not None:
                output[d.name] = d.default
            else:
                # Generate type-appropriate default
                defaults = {
                    "str": "",
                    "int": 0,
                    "float": 0.0,
                    "bool": False,
                    "list": [],
                    "dict": {},
                    "any": None,
                }
                output[d.name] = defaults.get(d.type)
        return output
```

---

## 4. Contract Validation API

### 4.1 ValidationError Schema

```python
class ValidationSeverity(str, Enum):
    """Severity levels for validation errors."""
    ERROR = "error"      # Fatal - output is invalid
    WARNING = "warning"  # Non-fatal - output is usable but suboptimal


class ValidationErrorType(str, Enum):
    """Types of validation errors."""
    MISSING = "missing"           # Required field missing
    TYPE_MISMATCH = "type"        # Wrong data type
    RULE_VIOLATION = "rule"       # Business rule failed
    CONSTRAINT = "constraint"     # Resource constraint exceeded


@dataclass
class ValidationError:
    """A single validation error.

    Attributes:
        field: Field name that failed validation (or None for global errors)
        error_type: Type of validation error
        reason: Human-readable explanation
        expected: What was expected
        actual: What was received
        severity: Error or warning
        rule: The specific rule that failed (if applicable)
    """
    field: Optional[str]
    error_type: ValidationErrorType
    reason: str
    expected: Any
    actual: Any
    severity: ValidationSeverity = ValidationSeverity.ERROR
    rule: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "field": self.field,
            "error_type": self.error_type.value,
            "reason": self.reason,
            "expected": str(self.expected),
            "actual": str(self.actual)[:100],  # Truncate long values
            "severity": self.severity.value,
            "rule": self.rule,
        }
```

### 4.2 ValidationResult Schema

```python
@dataclass
class ValidationResult:
    """Complete result from contract validation.

    Attributes:
        is_valid: True if output satisfies contract
        errors: List of validation errors
        warnings: List of warning messages
        suggestion: Suggested fix for validation failures
        validation_time_ms: Time taken to validate
        contract_name: Name of contract validated against
        contract_version: Version of contract
    """
    is_valid: bool
    errors: list[ValidationError]
    warnings: list[str]
    suggestion: Optional[str] = None
    validation_time_ms: int = 0
    contract_name: str = ""
    contract_version: str = ""

    @property
    def error_count(self) -> int:
        """Count of errors (not warnings)."""
        return len([e for e in self.errors if e.severity == ValidationSeverity.ERROR])

    @property
    def warning_count(self) -> int:
        """Count of warnings."""
        return len(self.warnings) + len([
            e for e in self.errors if e.severity == ValidationSeverity.WARNING
        ])

    def get_missing_fields(self) -> list[str]:
        """Get list of missing required fields."""
        return [
            e.field for e in self.errors
            if e.error_type == ValidationErrorType.MISSING and e.field
        ]

    def get_type_mismatches(self) -> list[str]:
        """Get list of fields with type mismatches."""
        return [
            e.field for e in self.errors
            if e.error_type == ValidationErrorType.TYPE_MISMATCH and e.field
        ]

    def get_rule_violations(self) -> list[tuple[str, str]]:
        """Get list of (field, rule) tuples that failed."""
        return [
            (e.field, e.rule) for e in self.errors
            if e.error_type == ValidationErrorType.RULE_VIOLATION and e.field and e.rule
        ]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "is_valid": self.is_valid,
            "errors": [e.to_dict() for e in self.errors],
            "warnings": self.warnings,
            "suggestion": self.suggestion,
            "validation_time_ms": self.validation_time_ms,
            "contract_name": self.contract_name,
            "contract_version": self.contract_version,
        }

    def to_summary(self) -> str:
        """Generate human-readable summary."""
        if self.is_valid:
            return f"Validation passed for {self.contract_name}"

        parts = [f"Validation failed for {self.contract_name}:"]
        for error in self.errors:
            parts.append(f"  - {error.field}: {error.reason}")

        if self.suggestion:
            parts.append(f"\nSuggestion: {self.suggestion}")

        return "\n".join(parts)
```

### 4.3 ContractValidator Class

```python
class ContractValidator:
    """Validates agent outputs against contract specifications.

    The validator checks:
    1. All required fields are present
    2. Field types match specifications
    3. Business rules are satisfied
    4. Resource constraints are within limits

    Attributes:
        strict_mode: If True, fail on first error; if False, collect all errors
    """

    def __init__(self, strict_mode: bool = False):
        """Initialize validator.

        Args:
            strict_mode: If True, stop at first error; if False, collect all errors.
        """
        self.strict_mode = strict_mode

    def validate(
        self,
        output: dict[str, Any],
        contract: Contract,
        context: Optional[dict[str, Any]] = None,
    ) -> ValidationResult:
        """Validate output against contract.

        Args:
            output: Agent output to validate (dict)
            contract: Contract specification
            context: Optional execution context (for constraint checking)

        Returns:
            ValidationResult with is_valid and any errors

        Example:
            >>> validator = ContractValidator()
            >>> result = validator.validate(
            ...     output={"qualification_score": 75, "bant_assessment": {}},
            ...     contract=lead_qualification_contract,
            ... )
            >>> if not result.is_valid:
            ...     print(result.to_summary())
        """
        import time
        start = time.time()

        errors: list[ValidationError] = []
        warnings: list[str] = []

        # 1. Check required fields
        for deliverable in contract.deliverables:
            if deliverable.required and deliverable.name not in output:
                errors.append(ValidationError(
                    field=deliverable.name,
                    error_type=ValidationErrorType.MISSING,
                    reason=f"Required field '{deliverable.name}' is missing",
                    expected=f"Field of type {deliverable.type}",
                    actual="<missing>",
                ))

                if self.strict_mode:
                    break

        # 2. Check types and rules for present fields
        for deliverable in contract.deliverables:
            if deliverable.name not in output:
                continue

            value = output[deliverable.name]

            # Type check
            if not deliverable.validate_type(value):
                errors.append(ValidationError(
                    field=deliverable.name,
                    error_type=ValidationErrorType.TYPE_MISMATCH,
                    reason=f"Expected type '{deliverable.type}', got '{type(value).__name__}'",
                    expected=deliverable.type,
                    actual=type(value).__name__,
                ))

                if self.strict_mode:
                    break
                continue  # Skip rule validation for wrong types

            # Rule validation
            failed_rules = deliverable.validate_rules(value)
            for rule in failed_rules:
                errors.append(ValidationError(
                    field=deliverable.name,
                    error_type=ValidationErrorType.RULE_VIOLATION,
                    reason=f"Rule '{rule}' failed for value '{value}'",
                    expected=rule,
                    actual=value,
                    rule=rule,
                ))

                if self.strict_mode:
                    break

        # 3. Check constraints (from context)
        if context:
            tokens_used = context.get("tokens_used", 0)
            tool_calls = context.get("tool_calls", 0)

            exceeded, warning = contract.constraints.check_tokens(tokens_used)
            if exceeded:
                errors.append(ValidationError(
                    field=None,
                    error_type=ValidationErrorType.CONSTRAINT,
                    reason=f"Token limit exceeded: {tokens_used} > {contract.constraints.max_total_tokens}",
                    expected=f"<= {contract.constraints.max_total_tokens}",
                    actual=tokens_used,
                ))
            elif warning:
                warnings.append(
                    f"Token usage at {tokens_used}/{contract.constraints.max_total_tokens} "
                    f"({tokens_used/contract.constraints.max_total_tokens:.0%})"
                )

            exceeded, warning = contract.constraints.check_tool_calls(tool_calls)
            if exceeded:
                errors.append(ValidationError(
                    field=None,
                    error_type=ValidationErrorType.CONSTRAINT,
                    reason=f"Tool call limit exceeded: {tool_calls} > {contract.constraints.max_tool_calls}",
                    expected=f"<= {contract.constraints.max_tool_calls}",
                    actual=tool_calls,
                ))

        # Generate suggestion if validation failed
        suggestion = None
        if errors:
            suggestion = self._generate_suggestion(errors, contract)

        elapsed_ms = int((time.time() - start) * 1000)

        return ValidationResult(
            is_valid=len([e for e in errors if e.severity == ValidationSeverity.ERROR]) == 0,
            errors=errors,
            warnings=warnings,
            suggestion=suggestion,
            validation_time_ms=elapsed_ms,
            contract_name=contract.name,
            contract_version=contract.version,
        )

    def _generate_suggestion(
        self,
        errors: list[ValidationError],
        contract: Contract,
    ) -> str:
        """Generate actionable suggestion from validation errors."""
        suggestions = []

        missing = [e.field for e in errors if e.error_type == ValidationErrorType.MISSING]
        if missing:
            suggestions.append(f"Add missing fields: {', '.join(missing)}")

        type_errors = [e.field for e in errors if e.error_type == ValidationErrorType.TYPE_MISMATCH]
        if type_errors:
            for field in type_errors:
                deliverable = contract.get_deliverable(field)
                if deliverable:
                    suggestions.append(f"Convert '{field}' to type '{deliverable.type}'")

        rule_errors = [(e.field, e.rule) for e in errors if e.error_type == ValidationErrorType.RULE_VIOLATION]
        if rule_errors:
            for field, rule in rule_errors:
                suggestions.append(f"Ensure '{field}' satisfies: {rule}")

        return "; ".join(suggestions) if suggestions else "Review output against contract specification"
```

---

## 5. Contract Execution API

### 5.1 ExecutionRequest Schema

```python
@dataclass
class ContractExecutionRequest:
    """Request to execute a task with contract enforcement.

    Attributes:
        agent_id: ID of agent to execute (or agent instance)
        task: Task description for the agent
        contract_name: Name of contract template to use
        contract: Full contract spec (alternative to contract_name)
        context: Optional execution context
        force_strategy: Override contract's failure strategy
        session_id: Session ID for tracking
    """
    agent_id: str
    task: str
    contract_name: Optional[str] = None
    contract: Optional[Contract] = None
    context: Optional[dict[str, Any]] = None
    force_strategy: Optional[FailureStrategy] = None
    session_id: Optional[str] = None

    def __post_init__(self):
        """Validate request has either contract_name or contract."""
        if not self.contract_name and not self.contract:
            raise ValueError("Either contract_name or contract must be provided")
```

### 5.2 ExecutionResult Schema

```python
@dataclass
class ContractExecutionResult:
    """Result from contract-enforced execution.

    Attributes:
        output: Validated or fallback output
        is_valid: True if output passed validation (or acceptable fallback)
        attempts: Number of execution attempts (1 = first try succeeded)
        tokens_used: Total tokens consumed across all attempts
        applied_strategy: Strategy that produced final result
        validation_result: Full validation details
        metadata: Execution metadata
    """
    output: dict[str, Any]
    is_valid: bool
    attempts: int
    tokens_used: int
    applied_strategy: str  # "success" | "retry" | "fallback" | "fail"
    validation_result: ValidationResult
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def retries_performed(self) -> int:
        """Number of retries (attempts - 1)."""
        return max(0, self.attempts - 1)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "output": self.output,
            "is_valid": self.is_valid,
            "attempts": self.attempts,
            "tokens_used": self.tokens_used,
            "applied_strategy": self.applied_strategy,
            "validation_result": self.validation_result.to_dict(),
            "metadata": self.metadata,
        }
```

### 5.3 ContractExecutor Class

```python
class ContractExecutor:
    """Executes agents with contract enforcement.

    The executor:
    1. Runs the agent with the task
    2. Validates output against contract
    3. If validation fails, applies retry/fallback strategy
    4. Returns best result with full metadata

    Attributes:
        validator: ContractValidator instance
        retry_manager: RetryManager for retry logic
        fallback_handler: FallbackHandler for degradation
        template_factory: Factory for contract templates
        event_store: EventStore for audit trail
    """

    def __init__(
        self,
        validator: Optional["ContractValidator"] = None,
        retry_manager: Optional["RetryManager"] = None,
        fallback_handler: Optional["FallbackHandler"] = None,
        template_factory: Optional["ContractTemplateFactory"] = None,
        event_store: Optional["EventStore"] = None,
        token_tracker: Optional["TokenTracker"] = None,
    ):
        """Initialize executor with dependencies."""
        self.validator = validator or ContractValidator()
        self.retry_manager = retry_manager or RetryManager()
        self.fallback_handler = fallback_handler or FallbackHandler()
        self.template_factory = template_factory or ContractTemplateFactory()
        self.event_store = event_store
        self.token_tracker = token_tracker

    async def execute(
        self,
        request: ContractExecutionRequest,
    ) -> ContractExecutionResult:
        """Execute task with contract enforcement.

        Args:
            request: Execution request with agent, task, and contract

        Returns:
            ContractExecutionResult with validated output

        Raises:
            ContractViolation: If strategy is FAIL and validation fails

        Execution Flow:
            1. Resolve contract (from name or spec)
            2. Execute agent
            3. Validate output
            4. If invalid:
               - Apply retry strategy (up to max_retries)
               - If still invalid, apply fallback strategy
            5. Return ContractExecutionResult with metadata

        Events Emitted:
            - ContractValidationStartedEvent
            - ContractValidatedEvent
            - ContractValidationFailedEvent (on failure)
            - ContractRetryEvent (on retry)
            - ContractFallbackEvent (on fallback)
            - ContractCompletedEvent

        Example:
            >>> executor = ContractExecutor()
            >>> result = await executor.execute(
            ...     ContractExecutionRequest(
            ...         agent_id="lead_qualifier",
            ...         task="Qualify John from Acme Corp",
            ...         contract_name="lead_qualification",
            ...     )
            ... )
            >>> print(f"Valid: {result.is_valid}, Attempts: {result.attempts}")
        """
        # Resolve contract
        contract = request.contract
        if not contract and request.contract_name:
            contract = self.template_factory.get_template(request.contract_name)

        if not contract:
            raise ValueError(f"Contract not found: {request.contract_name}")

        # Determine failure strategy
        strategy = request.force_strategy or contract.failure_strategy

        # Track state
        attempts = 0
        total_tokens = 0
        best_output = None
        best_validation = None

        # Emit start event
        self._emit_validation_started(request, contract)

        # Initial execution
        attempts += 1
        output, tokens = await self._execute_agent(request)
        total_tokens += tokens

        # Validate
        validation = self.validator.validate(
            output=output,
            contract=contract,
            context={"tokens_used": total_tokens},
        )

        best_output = output
        best_validation = validation

        if validation.is_valid:
            self._emit_validated(contract, validation)
            return ContractExecutionResult(
                output=output,
                is_valid=True,
                attempts=attempts,
                tokens_used=total_tokens,
                applied_strategy="success",
                validation_result=validation,
                metadata={"execution_time_ms": validation.validation_time_ms},
            )

        # Validation failed - apply strategy
        self._emit_validation_failed(contract, validation)

        # RETRY strategy
        if strategy in (FailureStrategy.RETRY,):
            for retry in range(contract.max_retries):
                attempts += 1

                # Refine prompt based on errors
                refined_task = self.retry_manager.refine_prompt(
                    original_task=request.task,
                    errors=validation.errors,
                    contract=contract,
                    level=retry + 1,
                )

                self._emit_retry(contract, attempts, validation)

                # Re-execute
                output, tokens = await self._execute_agent(
                    request, task_override=refined_task
                )
                total_tokens += tokens

                # Re-validate
                validation = self.validator.validate(
                    output=output,
                    contract=contract,
                    context={"tokens_used": total_tokens},
                )

                if validation.error_count < best_validation.error_count:
                    best_output = output
                    best_validation = validation

                if validation.is_valid:
                    self._emit_validated(contract, validation)
                    return ContractExecutionResult(
                        output=output,
                        is_valid=True,
                        attempts=attempts,
                        tokens_used=total_tokens,
                        applied_strategy="retry",
                        validation_result=validation,
                        metadata={"retries_performed": attempts - 1},
                    )

        # FALLBACK strategies
        if strategy in (FailureStrategy.FALLBACK, FailureStrategy.PARTIAL, FailureStrategy.TEMPLATE):
            fallback_result = self.fallback_handler.generate_fallback(
                output=best_output,
                contract=contract,
                validation=best_validation,
                strategy=strategy,
            )

            self._emit_fallback(contract, strategy, best_validation)

            return ContractExecutionResult(
                output=fallback_result.output,
                is_valid=False,  # Clearly marked as fallback
                attempts=attempts,
                tokens_used=total_tokens,
                applied_strategy=strategy.value,
                validation_result=best_validation,
                metadata={
                    "filled_from": fallback_result.filled_from,
                    "missing_deliverables": fallback_result.missing_deliverables,
                },
            )

        # FAIL/ESCALATE strategy
        if strategy in (FailureStrategy.FAIL, FailureStrategy.ESCALATE):
            self._emit_completed(contract, "fail", attempts, total_tokens)
            raise ContractViolation(
                contract_name=contract.name,
                errors=best_validation.errors,
                attempts=attempts,
            )

        # Should not reach here
        return ContractExecutionResult(
            output=best_output,
            is_valid=False,
            attempts=attempts,
            tokens_used=total_tokens,
            applied_strategy="unknown",
            validation_result=best_validation,
        )

    async def _execute_agent(
        self,
        request: ContractExecutionRequest,
        task_override: Optional[str] = None,
    ) -> tuple[dict[str, Any], int]:
        """Execute agent and return (output, tokens_used)."""
        # Implementation depends on agent registry
        # This is a placeholder
        task = task_override or request.task
        # agent = self.agent_registry.get(request.agent_id)
        # result = await agent.execute(task, context=request.context)
        # return result.output, result.tokens_used
        raise NotImplementedError("Agent execution depends on agent registry")

    def _emit_validation_started(self, request, contract):
        """Emit ContractValidationStartedEvent."""
        if self.event_store:
            # Implementation details
            pass

    # ... other event emission methods
```

---

## 6. Retry Management API

### 6.1 RetryConfiguration Schema

```python
@dataclass
class RetryConfiguration:
    """Configuration for retry behavior.

    Attributes:
        max_retries: Maximum retry attempts (default 2)
        retry_delays: Delays between retries in milliseconds
        retry_on_errors: Error types that trigger retry
        dont_retry: Error types that should not retry
        refinement_level: Initial refinement level (1-3)
    """
    max_retries: int = 2
    retry_delays: list[int] = field(default_factory=lambda: [0, 1000, 2000])
    retry_on_errors: list[str] = field(default_factory=lambda: ["missing", "type", "rule"])
    dont_retry: list[str] = field(default_factory=list)
    refinement_level: int = 1


class RefinementLevel(int, Enum):
    """Levels of prompt refinement aggressiveness."""
    LIGHT = 1    # Add error context to prompt
    MEDIUM = 2   # Add explicit format instructions
    HEAVY = 3    # Provide template structure
```

### 6.2 RetryManager Class

```python
class RetryManager:
    """Manages retry logic and prompt refinement.

    The retry manager:
    1. Analyzes validation errors
    2. Determines if retry is appropriate
    3. Refines the prompt with error context
    4. Tracks retry history

    Refinement Levels:
        Level 1 (Light): Add error context
            "Note: Previous attempt was missing 'qualification_score'"

        Level 2 (Medium): Add explicit format
            "Output MUST include: qualification_score (int, 0-100)"

        Level 3 (Heavy): Provide template
            "Output format: {'qualification_score': <int>, ...}"
    """

    def __init__(self, config: Optional[RetryConfiguration] = None):
        """Initialize retry manager."""
        self.config = config or RetryConfiguration()

    def should_retry(
        self,
        errors: list[ValidationError],
        attempt: int,
    ) -> bool:
        """Determine if retry is appropriate.

        Args:
            errors: Validation errors from last attempt
            attempt: Current attempt number (1-indexed)

        Returns:
            True if retry should be attempted
        """
        if attempt >= self.config.max_retries + 1:
            return False

        # Check if any errors are in dont_retry list
        for error in errors:
            if error.error_type.value in self.config.dont_retry:
                return False

        # Check if any errors are retryable
        retryable = any(
            error.error_type.value in self.config.retry_on_errors
            for error in errors
        )

        return retryable

    def classify_errors(
        self,
        errors: list[ValidationError],
    ) -> dict[str, list[ValidationError]]:
        """Classify errors by type for targeted refinement.

        Returns:
            Dict mapping error type to list of errors
        """
        classified: dict[str, list[ValidationError]] = {}
        for error in errors:
            key = error.error_type.value
            if key not in classified:
                classified[key] = []
            classified[key].append(error)
        return classified

    def refine_prompt(
        self,
        original_task: str,
        errors: list[ValidationError],
        contract: Contract,
        level: int = 1,
    ) -> str:
        """Refine prompt based on validation errors.

        Args:
            original_task: Original task description
            errors: Validation errors to address
            contract: Contract specification
            level: Refinement level (1-3)

        Returns:
            Refined task prompt
        """
        classified = self.classify_errors(errors)
        refinements = []

        # Level 1: Add error context
        if level >= 1:
            if "missing" in classified:
                missing_fields = [e.field for e in classified["missing"] if e.field]
                refinements.append(
                    f"IMPORTANT: Your response MUST include these fields: {', '.join(missing_fields)}"
                )

            if "type" in classified:
                for error in classified["type"]:
                    deliverable = contract.get_deliverable(error.field)
                    if deliverable:
                        refinements.append(
                            f"Field '{error.field}' must be of type {deliverable.type}"
                        )

            if "rule" in classified:
                for error in classified["rule"]:
                    refinements.append(
                        f"Field '{error.field}' must satisfy: {error.rule}"
                    )

        # Level 2: Add explicit format instructions
        if level >= 2:
            format_lines = ["", "REQUIRED OUTPUT FORMAT:", ""]
            for deliverable in contract.deliverables:
                if deliverable.required:
                    rules = ", ".join(deliverable.validation_rules) if deliverable.validation_rules else "none"
                    format_lines.append(
                        f"- {deliverable.name}: {deliverable.type} (rules: {rules})"
                    )
            refinements.append("\n".join(format_lines))

        # Level 3: Provide template structure
        if level >= 3:
            template = contract.generate_template_output()
            refinements.append(
                f"\nEXACT OUTPUT STRUCTURE REQUIRED:\n```json\n{json.dumps(template, indent=2)}\n```"
            )

        # Combine
        refined = original_task
        if refinements:
            refined = original_task + "\n\n" + "\n\n".join(refinements)

        return refined

    def get_delay(self, attempt: int) -> int:
        """Get delay in milliseconds before retry.

        Args:
            attempt: Attempt number (1-indexed)

        Returns:
            Delay in milliseconds
        """
        delays = self.config.retry_delays
        if attempt <= len(delays):
            return delays[attempt - 1]
        return delays[-1] if delays else 0
```

---

## 7. Fallback Strategy API

### 7.1 FallbackResult Schema

```python
@dataclass
class FallbackResult:
    """Result from fallback generation.

    Attributes:
        output: Generated fallback output
        filled_from: Source of fallback data ("partial" | "template")
        missing_deliverables: Fields that could not be filled
        warnings: Warning messages about the fallback
        note: Explanation of fallback result
    """
    output: dict[str, Any]
    filled_from: str  # "partial" | "template"
    missing_deliverables: list[str]
    warnings: list[str]
    note: str = ""
```

### 7.2 FallbackHandler Class

```python
class FallbackHandler:
    """Handles graceful degradation when validation fails.

    Fallback Strategies:
        - FALLBACK/PARTIAL: Return best-effort partial result with warnings
        - TEMPLATE: Generate from contract template
        - (FAIL/ESCALATE are not handled here - they raise exceptions)
    """

    def generate_fallback(
        self,
        output: dict[str, Any],
        contract: Contract,
        validation: ValidationResult,
        strategy: FailureStrategy,
    ) -> FallbackResult:
        """Generate fallback result.

        Args:
            output: Original (invalid) output
            contract: Contract specification
            validation: Validation result with errors
            strategy: Fallback strategy to apply

        Returns:
            FallbackResult with best-effort output
        """
        if strategy == FailureStrategy.TEMPLATE:
            return self._generate_template_fallback(contract)
        else:
            # FALLBACK or PARTIAL
            return self._generate_partial_fallback(output, contract, validation)

    def _generate_partial_fallback(
        self,
        output: dict[str, Any],
        contract: Contract,
        validation: ValidationResult,
    ) -> FallbackResult:
        """Generate partial result from available valid data.

        Takes valid fields from output, fills missing with defaults.
        """
        result = {}
        missing = []
        warnings = []

        for deliverable in contract.deliverables:
            if deliverable.name in output:
                value = output[deliverable.name]

                # Check if this field is valid
                is_valid = deliverable.validate_type(value)
                if is_valid:
                    failed_rules = deliverable.validate_rules(value)
                    is_valid = len(failed_rules) == 0

                if is_valid:
                    result[deliverable.name] = value
                else:
                    # Use default or example
                    if deliverable.default is not None:
                        result[deliverable.name] = deliverable.default
                        warnings.append(f"Used default for invalid '{deliverable.name}'")
                    elif deliverable.example is not None:
                        result[deliverable.name] = deliverable.example
                        warnings.append(f"Used example for invalid '{deliverable.name}'")
                    else:
                        missing.append(deliverable.name)
            else:
                # Field missing
                if deliverable.default is not None:
                    result[deliverable.name] = deliverable.default
                    warnings.append(f"Used default for missing '{deliverable.name}'")
                elif deliverable.example is not None:
                    result[deliverable.name] = deliverable.example
                    warnings.append(f"Used example for missing '{deliverable.name}'")
                elif deliverable.required:
                    missing.append(deliverable.name)

        return FallbackResult(
            output=result,
            filled_from="partial",
            missing_deliverables=missing,
            warnings=warnings,
            note="Fallback result: some deliverables were inferred from defaults/examples",
        )

    def _generate_template_fallback(
        self,
        contract: Contract,
    ) -> FallbackResult:
        """Generate result purely from contract template."""
        template = contract.generate_template_output()

        return FallbackResult(
            output=template,
            filled_from="template",
            missing_deliverables=[],  # Template has all fields
            warnings=["Result generated entirely from template - no agent output used"],
            note="Fallback result: generated from contract template due to validation failures",
        )
```

---

## 8. Template Factory API

### 8.1 ContractTemplateFactory Class

```python
class ContractTemplateFactory:
    """Factory for retrieving and creating contract templates.

    Provides pre-built contracts for common ACTi workflows:
    - lead_qualification: BANT-based lead scoring
    - research_report: Structured research output
    - appointment_booking: Calendar event creation
    - market_analysis: Competitive analysis
    - compliance_check: Governance validation
    """

    _templates: dict[str, Contract] = {}

    def __init__(self):
        """Initialize with default templates."""
        self._register_default_templates()

    def _register_default_templates(self):
        """Register built-in templates."""
        # Lead Qualification
        self._templates["lead_qualification"] = Contract(
            name="lead_qualification",
            description="Qualify a sales lead using BANT methodology",
            deliverables=[
                Deliverable(
                    name="qualification_score",
                    type="int",
                    description="Overall qualification score from 0-100",
                    required=True,
                    validation_rules=["value >= 0", "value <= 100"],
                    example=75,
                ),
                Deliverable(
                    name="bant_assessment",
                    type="dict",
                    description="BANT breakdown with scores for Budget, Authority, Need, Timeline",
                    required=True,
                    example={
                        "budget": {"score": 80, "notes": "Budget approved"},
                        "authority": {"score": 70, "notes": "Decision maker identified"},
                        "need": {"score": 90, "notes": "Strong pain point"},
                        "timeline": {"score": 60, "notes": "Q2 implementation"},
                    },
                ),
                Deliverable(
                    name="recommended_action",
                    type="str",
                    description="Recommended next step",
                    required=True,
                    validation_rules=["len(value) > 0"],
                    example="Schedule product demo",
                ),
                Deliverable(
                    name="confidence",
                    type="float",
                    description="Confidence in assessment (0.0-1.0)",
                    required=False,
                    validation_rules=["value >= 0.0", "value <= 1.0"],
                    example=0.85,
                ),
            ],
            constraints=ContractConstraints(
                max_total_tokens=5000,
                max_tool_calls=5,
                timeout_seconds=60,
            ),
            failure_strategy=FailureStrategy.RETRY,
            max_retries=2,
            version="1.0.0",
        )

        # Research Report
        self._templates["research_report"] = Contract(
            name="research_report",
            description="Structured research report with findings and recommendations",
            deliverables=[
                Deliverable(
                    name="title",
                    type="str",
                    description="Report title",
                    required=True,
                    validation_rules=["len(value) > 0", "len(value) < 200"],
                ),
                Deliverable(
                    name="summary",
                    type="str",
                    description="Executive summary (2-3 sentences)",
                    required=True,
                    validation_rules=["len(value) >= 50", "len(value) <= 500"],
                ),
                Deliverable(
                    name="findings",
                    type="list",
                    description="List of key findings",
                    required=True,
                    validation_rules=["len(value) >= 1"],
                ),
                Deliverable(
                    name="recommendations",
                    type="list",
                    description="List of actionable recommendations",
                    required=True,
                    validation_rules=["len(value) >= 1"],
                ),
                Deliverable(
                    name="sources",
                    type="list",
                    description="List of sources used",
                    required=False,
                ),
            ],
            constraints=ContractConstraints(
                max_total_tokens=8000,
                max_tool_calls=10,
            ),
            failure_strategy=FailureStrategy.FALLBACK,
            max_retries=1,
        )

        # Appointment Booking
        self._templates["appointment_booking"] = Contract(
            name="appointment_booking",
            description="Calendar appointment creation details",
            deliverables=[
                Deliverable(
                    name="event_title",
                    type="str",
                    description="Event title/subject",
                    required=True,
                ),
                Deliverable(
                    name="start_time",
                    type="str",
                    description="Start time in ISO 8601 format",
                    required=True,
                ),
                Deliverable(
                    name="end_time",
                    type="str",
                    description="End time in ISO 8601 format",
                    required=True,
                ),
                Deliverable(
                    name="attendees",
                    type="list",
                    description="List of attendee email addresses",
                    required=True,
                    validation_rules=["len(value) >= 1"],
                ),
                Deliverable(
                    name="description",
                    type="str",
                    description="Event description",
                    required=False,
                ),
                Deliverable(
                    name="location",
                    type="str",
                    description="Meeting location or video link",
                    required=False,
                ),
            ],
            constraints=ContractConstraints(
                max_total_tokens=3000,
                max_tool_calls=3,
            ),
            failure_strategy=FailureStrategy.RETRY,
            max_retries=2,
        )

        # Market Analysis
        self._templates["market_analysis"] = Contract(
            name="market_analysis",
            description="Competitive market analysis report",
            deliverables=[
                Deliverable(
                    name="market_overview",
                    type="str",
                    description="Overview of the market landscape",
                    required=True,
                ),
                Deliverable(
                    name="competitors",
                    type="list",
                    description="List of competitor analysis objects",
                    required=True,
                    validation_rules=["len(value) >= 1"],
                ),
                Deliverable(
                    name="market_size",
                    type="dict",
                    description="Market size data (total, addressable, obtainable)",
                    required=False,
                ),
                Deliverable(
                    name="trends",
                    type="list",
                    description="Key market trends",
                    required=True,
                ),
                Deliverable(
                    name="opportunities",
                    type="list",
                    description="Identified opportunities",
                    required=True,
                ),
                Deliverable(
                    name="threats",
                    type="list",
                    description="Identified threats",
                    required=True,
                ),
            ],
            constraints=ContractConstraints(
                max_total_tokens=10000,
                max_tool_calls=15,
            ),
            failure_strategy=FailureStrategy.FALLBACK,
            max_retries=1,
        )

        # Compliance Check
        self._templates["compliance_check"] = Contract(
            name="compliance_check",
            description="Governance and compliance validation",
            deliverables=[
                Deliverable(
                    name="is_compliant",
                    type="bool",
                    description="Overall compliance status",
                    required=True,
                ),
                Deliverable(
                    name="compliance_score",
                    type="int",
                    description="Compliance score 0-100",
                    required=True,
                    validation_rules=["value >= 0", "value <= 100"],
                ),
                Deliverable(
                    name="violations",
                    type="list",
                    description="List of identified violations",
                    required=True,
                ),
                Deliverable(
                    name="recommendations",
                    type="list",
                    description="Remediation recommendations",
                    required=True,
                ),
                Deliverable(
                    name="reviewed_items",
                    type="list",
                    description="Items that were reviewed",
                    required=True,
                ),
            ],
            constraints=ContractConstraints(
                max_total_tokens=6000,
                max_tool_calls=8,
            ),
            failure_strategy=FailureStrategy.FAIL,  # Compliance must not fallback
            max_retries=2,
        )

    def get_template(self, name: str) -> Optional[Contract]:
        """Get contract template by name.

        Args:
            name: Template name (e.g., "lead_qualification")

        Returns:
            Contract instance or None if not found
        """
        return self._templates.get(name)

    def list_templates(self) -> list[str]:
        """List all available template names."""
        return list(self._templates.keys())

    def register_template(self, contract: Contract) -> None:
        """Register a custom contract template.

        Args:
            contract: Contract to register
        """
        self._templates[contract.name] = contract

    def create_custom(
        self,
        name: str,
        description: str,
        deliverables: list[Deliverable],
        constraints: Optional[ContractConstraints] = None,
        failure_strategy: FailureStrategy = FailureStrategy.RETRY,
        max_retries: int = 2,
    ) -> Contract:
        """Create and optionally register a custom contract.

        Args:
            name: Contract name
            description: Contract description
            deliverables: List of deliverables
            constraints: Optional constraints
            failure_strategy: Failure handling strategy
            max_retries: Maximum retry attempts

        Returns:
            Created Contract instance
        """
        contract = Contract(
            name=name,
            description=description,
            deliverables=deliverables,
            constraints=constraints or ContractConstraints(),
            failure_strategy=failure_strategy,
            max_retries=max_retries,
        )

        return contract


# Convenience function
def get_contract_template(name: str) -> Optional[Contract]:
    """Get a contract template by name.

    Convenience function for accessing the global template factory.

    Args:
        name: Template name

    Returns:
        Contract instance or None

    Example:
        >>> contract = get_contract_template("lead_qualification")
        >>> result = await executor.execute(task, contract=contract)
    """
    factory = ContractTemplateFactory()
    return factory.get_template(name)
```

---

## 9. Event Contract

### 9.1 Contract Event Types

```python
class ContractEventType(str, Enum):
    """Contract subsystem event types."""
    VALIDATION_STARTED = "contract.validation_started"
    VALIDATED = "contract.validated"
    VALIDATION_FAILED = "contract.validation_failed"
    RETRY = "contract.retry"
    FALLBACK = "contract.fallback"
    COMPLETED = "contract.completed"
```

### 9.2 Event Payloads

```python
@dataclass
class ContractValidationStartedPayload:
    """Payload for VALIDATION_STARTED event."""
    contract_id: str
    contract_name: str
    output_hash: str  # SHA256 hash for correlation
    timestamp: str
    session_id: Optional[str] = None


@dataclass
class ContractValidatedPayload:
    """Payload for VALIDATED event."""
    contract_id: str
    contract_name: str
    is_valid: bool
    error_count: int
    warning_count: int
    validation_time_ms: int


@dataclass
class ContractValidationFailedPayload:
    """Payload for VALIDATION_FAILED event."""
    contract_id: str
    contract_name: str
    error_types: list[str]  # ["missing", "type", "rule"]
    error_count: int
    reason: str


@dataclass
class ContractRetryPayload:
    """Payload for RETRY event."""
    contract_id: str
    contract_name: str
    attempt_number: int
    error_summary: str
    refined_prompt_hash: str  # SHA256 for correlation
    refinement_level: int
    tokens_used_so_far: int


@dataclass
class ContractFallbackPayload:
    """Payload for FALLBACK event."""
    contract_id: str
    contract_name: str
    fallback_type: str  # "partial" | "template"
    reason: str
    missing_deliverables: list[str]


@dataclass
class ContractCompletedPayload:
    """Payload for COMPLETED event."""
    contract_id: str
    contract_name: str
    applied_strategy: str  # "success" | "retry" | "fallback" | "fail"
    attempts: int
    tokens_used: int
    is_valid: bool
    execution_time_ms: int
```

### 9.3 Event Factory Functions

```python
def create_contract_validation_started_event(
    session_id: str,
    contract_id: str,
    contract_name: str,
    output: dict[str, Any],
    correlation_id: Optional[str] = None,
) -> Event:
    """Create ContractValidationStartedEvent."""
    import hashlib
    output_hash = hashlib.sha256(json.dumps(output, sort_keys=True).encode()).hexdigest()[:16]

    return Event(
        event_id=_generate_event_id(),
        event_type=ContractEventType.VALIDATION_STARTED.value,
        timestamp=_get_utc_now(),
        session_id=session_id,
        correlation_id=correlation_id,
        payload={
            "contract_id": contract_id,
            "contract_name": contract_name,
            "output_hash": output_hash,
            "timestamp": _get_utc_now(),
        },
    )


def create_contract_retry_event(
    session_id: str,
    contract_id: str,
    contract_name: str,
    attempt_number: int,
    error_summary: str,
    refined_prompt: str,
    refinement_level: int,
    tokens_used_so_far: int,
    correlation_id: Optional[str] = None,
) -> Event:
    """Create ContractRetryEvent."""
    import hashlib
    prompt_hash = hashlib.sha256(refined_prompt.encode()).hexdigest()[:16]

    return Event(
        event_id=_generate_event_id(),
        event_type=ContractEventType.RETRY.value,
        timestamp=_get_utc_now(),
        session_id=session_id,
        correlation_id=correlation_id,
        payload={
            "contract_id": contract_id,
            "contract_name": contract_name,
            "attempt_number": attempt_number,
            "error_summary": error_summary,
            "refined_prompt_hash": prompt_hash,
            "refinement_level": refinement_level,
            "tokens_used_so_far": tokens_used_so_far,
        },
    )


def create_contract_completed_event(
    session_id: str,
    contract_id: str,
    contract_name: str,
    applied_strategy: str,
    attempts: int,
    tokens_used: int,
    is_valid: bool,
    execution_time_ms: int,
    correlation_id: Optional[str] = None,
) -> Event:
    """Create ContractCompletedEvent."""
    return Event(
        event_id=_generate_event_id(),
        event_type=ContractEventType.COMPLETED.value,
        timestamp=_get_utc_now(),
        session_id=session_id,
        correlation_id=correlation_id,
        payload={
            "contract_id": contract_id,
            "contract_name": contract_name,
            "applied_strategy": applied_strategy,
            "attempts": attempts,
            "tokens_used": tokens_used,
            "is_valid": is_valid,
            "execution_time_ms": execution_time_ms,
        },
    )
```

---

## 10. Error Taxonomy

### 10.1 Contract Error Codes

| Code | Name | Trigger | Recovery |
|------|------|---------|----------|
| CV-001 | ValidationError | Output doesn't match schema | Retry or fallback |
| CV-002 | MissingDeliverable | Required field missing | Retry with context |
| CV-003 | TypeMismatch | Wrong data type | Retry with explicit type |
| CV-004 | RuleViolation | Validation rule failed | Retry or fallback |
| CV-005 | ConstraintViolation | Token/tool limits exceeded | Escalate to fail |
| CV-006 | AgentExecutionError | Agent failed before validation | Retry or escalate |
| CV-007 | TimeoutError | Execution exceeded timeout | Fallback |
| CV-008 | ContractViolation | User requested strategy=fail | Raise exception |
| CV-009 | ContractNotFound | Template name doesn't exist | Return error |
| CV-010 | InvalidContractSpec | Contract definition is invalid | Return error |

### 10.2 Exception Classes

```python
class ContractError(SigilError):
    """Base exception for contract errors."""
    code = "CV-000"


class ValidationError(ContractError):
    """Output validation failed."""
    code = "CV-001"

    def __init__(self, validation_result: ValidationResult):
        self.validation_result = validation_result
        message = f"Validation failed: {validation_result.error_count} errors"
        super().__init__(
            message=message,
            context={
                "contract_name": validation_result.contract_name,
                "error_count": validation_result.error_count,
            },
        )


class MissingDeliverableError(ContractError):
    """Required deliverable is missing from output."""
    code = "CV-002"

    def __init__(self, field: str, contract_name: str):
        super().__init__(
            message=f"Required field '{field}' missing from output",
            context={
                "field": field,
                "contract_name": contract_name,
            },
            recovery_suggestions=[
                f"Ensure agent output includes '{field}'",
                "Retry with more explicit instructions",
            ],
        )


class TypeMismatchError(ContractError):
    """Field has wrong type."""
    code = "CV-003"

    def __init__(self, field: str, expected: str, actual: str):
        super().__init__(
            message=f"Field '{field}' has type '{actual}', expected '{expected}'",
            context={
                "field": field,
                "expected_type": expected,
                "actual_type": actual,
            },
            recovery_suggestions=[
                f"Convert '{field}' to type '{expected}'",
            ],
        )


class RuleViolationError(ContractError):
    """Business rule validation failed."""
    code = "CV-004"

    def __init__(self, field: str, rule: str, value: Any):
        super().__init__(
            message=f"Field '{field}' violates rule: {rule}",
            context={
                "field": field,
                "rule": rule,
                "value": str(value)[:100],
            },
            recovery_suggestions=[
                f"Ensure '{field}' satisfies: {rule}",
            ],
        )


class ConstraintViolationError(ContractError):
    """Resource constraint exceeded."""
    code = "CV-005"

    def __init__(self, constraint: str, limit: int, actual: int):
        super().__init__(
            message=f"Constraint '{constraint}' exceeded: {actual} > {limit}",
            context={
                "constraint": constraint,
                "limit": limit,
                "actual": actual,
            },
            recovery_suggestions=[
                "Reduce complexity of task",
                "Increase resource limits",
            ],
        )


class ContractViolation(ContractError):
    """Contract validation failed with fail strategy."""
    code = "CV-008"

    def __init__(
        self,
        contract_name: str,
        errors: list[ValidationError],
        attempts: int,
    ):
        self.errors = errors
        self.attempts = attempts

        error_summary = "; ".join([e.reason for e in errors[:3]])

        super().__init__(
            message=f"Contract '{contract_name}' violated after {attempts} attempts: {error_summary}",
            context={
                "contract_name": contract_name,
                "error_count": len(errors),
                "attempts": attempts,
            },
            recovery_suggestions=[
                "Review agent output format",
                "Consider using fallback strategy",
                "Simplify contract requirements",
            ],
        )


class ContractNotFoundError(ContractError):
    """Contract template not found."""
    code = "CV-009"

    def __init__(self, contract_name: str, available: list[str]):
        super().__init__(
            message=f"Contract template '{contract_name}' not found",
            context={
                "contract_name": contract_name,
                "available_templates": available,
            },
            recovery_suggestions=[
                f"Use one of: {', '.join(available)}",
                "Create custom contract with Contract(...)",
            ],
        )
```

### 10.3 Error Response Format

```python
@dataclass
class ContractErrorResponse:
    """Structured error response for contract errors.

    Follows RFC 9457 Problem Details format.
    """
    code: str
    message: str
    details: dict[str, Any]
    recoverable: bool
    suggested_action: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "error": {
                "code": self.code,
                "message": self.message,
                "details": self.details,
                "recoverable": self.recoverable,
                "suggested_action": self.suggested_action,
            }
        }
```

---

## 11. Tool Integration

### 11.1 LangChain Tool: validate_contract

```python
from langchain.tools import Tool
from pydantic import BaseModel, Field


class ValidateContractInput(BaseModel):
    """Input schema for validate_contract tool."""
    output: dict = Field(description="JSON object to validate")
    contract_name: str = Field(description="Contract template name")


def create_validate_contract_tool(
    template_factory: ContractTemplateFactory,
    validator: ContractValidator,
) -> Tool:
    """Create LangChain tool for contract validation.

    Tool Name: validate_contract
    Description: Validate output against a contract specification

    Returns:
        LangChain Tool instance
    """

    def validate_contract(output: dict, contract_name: str) -> str:
        """Validate output against contract."""
        contract = template_factory.get_template(contract_name)
        if not contract:
            return f"Error: Contract '{contract_name}' not found"

        result = validator.validate(output, contract)

        if result.is_valid:
            return f"Validation PASSED for {contract_name}"
        else:
            errors = "; ".join([e.reason for e in result.errors[:3]])
            return f"Validation FAILED: {errors}"

    return Tool(
        name="validate_contract",
        description=(
            "Validate a JSON output against a contract specification. "
            "Use this to check if output meets requirements before returning."
        ),
        func=validate_contract,
        args_schema=ValidateContractInput,
    )
```

### 11.2 LangChain Tool: get_contract_template

```python
class GetContractTemplateInput(BaseModel):
    """Input schema for get_contract_template tool."""
    template_name: str = Field(description="Contract template name")


def create_get_contract_template_tool(
    template_factory: ContractTemplateFactory,
) -> Tool:
    """Create LangChain tool for retrieving contract templates.

    Tool Name: get_contract_template
    Description: Get specifications for a contract template
    """

    def get_template(template_name: str) -> str:
        """Get contract template specification."""
        contract = template_factory.get_template(template_name)
        if not contract:
            available = template_factory.list_templates()
            return f"Template '{template_name}' not found. Available: {', '.join(available)}"

        # Format as readable spec
        lines = [
            f"Contract: {contract.name}",
            f"Description: {contract.description}",
            "",
            "Required Fields:",
        ]

        for d in contract.deliverables:
            req = "REQUIRED" if d.required else "optional"
            lines.append(f"  - {d.name} ({d.type}, {req}): {d.description}")

        return "\n".join(lines)

    return Tool(
        name="get_contract_template",
        description=(
            "Get the specification for a contract template. "
            "Use this to understand what output format is required."
        ),
        func=get_template,
        args_schema=GetContractTemplateInput,
    )
```

### 11.3 LangChain Tool: enforce_contract

```python
class EnforceContractInput(BaseModel):
    """Input schema for enforce_contract tool."""
    task: str = Field(description="Task to execute")
    contract_name: str = Field(description="Contract to enforce")
    max_retries: int = Field(default=2, description="Maximum retry attempts")


def create_enforce_contract_tool(
    executor: ContractExecutor,
) -> Tool:
    """Create LangChain tool for contract-enforced execution.

    Tool Name: enforce_contract
    Description: Execute task with contract enforcement and auto-retry
    """

    async def enforce_contract(
        task: str,
        contract_name: str,
        max_retries: int = 2,
    ) -> str:
        """Execute with contract enforcement."""
        try:
            result = await executor.execute(
                ContractExecutionRequest(
                    agent_id="self",  # Use calling agent
                    task=task,
                    contract_name=contract_name,
                )
            )

            status = "VALID" if result.is_valid else "FALLBACK"
            return (
                f"Execution {status} after {result.attempts} attempt(s). "
                f"Strategy: {result.applied_strategy}. "
                f"Tokens: {result.tokens_used}"
            )
        except ContractViolation as e:
            return f"FAILED: {e.message}"
        except Exception as e:
            return f"Error: {str(e)}"

    return Tool(
        name="enforce_contract",
        description=(
            "Execute a task with contract enforcement and automatic retry. "
            "Use when you need guaranteed output format."
        ),
        func=enforce_contract,
        args_schema=EnforceContractInput,
        coroutine=enforce_contract,  # Mark as async
    )
```

---

## 12. Agent Integration

### 12.1 Agent with Contract (Config-Based)

```python
@dataclass
class AgentConfigWithContract:
    """Agent configuration with contract specification.

    Agents can have contracts defined in their configuration,
    which are automatically enforced on every execution.

    Example:
        >>> config = AgentConfigWithContract(
        ...     name="lead_qualifier",
        ...     instructions="You are a sales lead qualifier...",
        ...     contract_name="lead_qualification",
        ...     tools=["crm_lookup", "websearch"],
        ... )
    """
    name: str
    instructions: str
    contract_name: Optional[str] = None  # References template
    contract: Optional[Contract] = None  # Or inline contract
    tools: list[str] = field(default_factory=list)
    failure_strategy_override: Optional[FailureStrategy] = None
```

### 12.2 Runtime Contract Override

```python
async def execute_with_contract(
    executor: ContractExecutor,
    agent: Any,  # Agent instance
    task: str,
    contract: Optional[Contract] = None,
    strategy: Optional[FailureStrategy] = None,
) -> ContractExecutionResult:
    """Execute agent with runtime contract override.

    Allows overriding the agent's default contract at runtime.

    Args:
        executor: ContractExecutor instance
        agent: Agent to execute
        task: Task description
        contract: Override contract (uses agent default if None)
        strategy: Override failure strategy

    Returns:
        ContractExecutionResult

    Example:
        >>> # Override with stricter contract
        >>> strict_contract = Contract(
        ...     name="strict_qualification",
        ...     deliverables=[...],
        ...     failure_strategy=FailureStrategy.FAIL,
        ... )
        >>> result = await execute_with_contract(
        ...     executor=executor,
        ...     agent=lead_qualifier,
        ...     task="Qualify John from Acme",
        ...     contract=strict_contract,
        ... )
    """
    request = ContractExecutionRequest(
        agent_id=agent.name,
        task=task,
        contract=contract or agent.contract,
        force_strategy=strategy,
    )

    return await executor.execute(request)
```

### 12.3 Execution Flow

```
1. Route detects complex task (Phase 3)
2. Router suggests/selects contract based on intent
3. Agent executes task
4. ContractExecutor validates output
5. If invalid:
   - Apply retry strategy (up to max_retries)
   - Refine prompt with error context
   - Re-execute and re-validate
   - If still invalid, apply fallback strategy
6. Return ContractResult with metadata

Flow Diagram:

User Request
    |
    v
+-------------------+
|  Router (Phase 3) |
|  - Classify intent |
|  - Suggest contract|
+--------+----------+
         |
         v
+-------------------+
|  Agent Executor   |
|  - Execute task   |
+--------+----------+
         |
         v
+-------------------+
| ContractValidator |
|  - Validate output|
+--------+----------+
         |
    +----+----+
    |         |
  Valid    Invalid
    |         |
    v         v
 Return   +--------+
 Result   | Retry? |
          +----+---+
               |
          +----+----+
          |         |
         Yes       No
          |         |
          v         v
      Refine    +--------+
      Prompt    | Fallback|
          |     +----+---+
          v          |
      Re-execute     v
          |       Generate
          v       Result
      Re-validate    |
          |         v
          +----+----+
               |
               v
         Return Result
```

---

## 13. Configuration API

### 13.1 ContractSettings Schema

```python
@dataclass
class ContractSettings:
    """Configuration for contract system.

    Attributes:
        enabled: Whether contract system is active
        default_strategy: Default failure strategy
        max_retries: Default maximum retries
        warn_threshold: Default warning threshold
        strict_mode: Fail on first error vs. collect all
        event_emission: Whether to emit events
        validation_timeout_seconds: Timeout for validation
    """
    enabled: bool = True
    default_strategy: FailureStrategy = FailureStrategy.RETRY
    max_retries: int = 2
    warn_threshold: float = 0.8
    strict_mode: bool = False
    event_emission: bool = True
    validation_timeout_seconds: int = 10

    # Template registry settings
    auto_register_templates: bool = True
    custom_templates_dir: Optional[str] = None

    # Retry settings
    retry_delays: list[int] = field(default_factory=lambda: [0, 1000, 2000])
    refinement_start_level: int = 1
```

### 13.2 Integration with SigilSettings

```python
@dataclass
class SigilSettings:
    """Main Sigil configuration.

    Includes settings for all phases including contracts.
    """
    # ... other settings ...

    contracts: ContractSettings = field(default_factory=ContractSettings)

    # Usage example:
    # settings = SigilSettings(
    #     contracts=ContractSettings(
    #         enabled=True,
    #         default_strategy=FailureStrategy.FALLBACK,
    #         max_retries=3,
    #     )
    # )
```

### 13.3 Runtime Configuration

```python
class ContractRuntimeConfig:
    """Runtime configuration that can be modified per-request.

    Allows overriding settings for specific executions without
    changing the global configuration.
    """

    def __init__(self, base_settings: ContractSettings):
        self._base = base_settings
        self._overrides: dict[str, Any] = {}

    def override(self, **kwargs) -> "ContractRuntimeConfig":
        """Create new config with overrides."""
        new_config = ContractRuntimeConfig(self._base)
        new_config._overrides = {**self._overrides, **kwargs}
        return new_config

    @property
    def enabled(self) -> bool:
        return self._overrides.get("enabled", self._base.enabled)

    @property
    def default_strategy(self) -> FailureStrategy:
        return self._overrides.get("default_strategy", self._base.default_strategy)

    # ... other properties ...
```

---

## 14. Data Schema Contracts

### 14.1 Request/Response Schemas

#### Validation Request
```python
@dataclass
class ValidationRequest:
    """Request to validate output against contract.

    Used for direct validation API calls.
    """
    output: dict[str, Any]  # Output to validate
    contract_name: Optional[str] = None  # Template name
    contract_spec: Optional[dict[str, Any]] = None  # Or inline spec
    strict: bool = True  # Fail on first error vs. collect all
    context: Optional[dict[str, Any]] = None  # Execution context
```

#### Validation Response
```python
@dataclass
class ValidationResponse:
    """Response from validation API."""
    is_valid: bool
    errors: list[dict[str, Any]]  # Serialized ValidationErrors
    warnings: list[str]
    suggestion: Optional[str]
    validation_time_ms: int
    contract_name: str
    contract_version: str
```

#### Execution Request
```python
@dataclass
class ExecutionRequest:
    """Request to execute with contract enforcement."""
    agent_id: str
    task: str
    contract_name: Optional[str] = None
    contract_spec: Optional[dict[str, Any]] = None
    context: Optional[dict[str, Any]] = None
    force_strategy: Optional[str] = None  # Serialized FailureStrategy
    session_id: Optional[str] = None
```

#### Execution Response
```python
@dataclass
class ExecutionResponse:
    """Response from execution API."""
    output: dict[str, Any]
    is_valid: bool
    attempts: int
    tokens_used: int
    applied_strategy: str
    validation_result: ValidationResponse
    metadata: dict[str, Any]
```

### 14.2 JSON Serialization

```python
def contract_to_dict(contract: Contract) -> dict[str, Any]:
    """Serialize Contract to dictionary."""
    return {
        "name": contract.name,
        "description": contract.description,
        "deliverables": [
            {
                "name": d.name,
                "type": d.type,
                "description": d.description,
                "required": d.required,
                "validation_rules": d.validation_rules,
                "example": d.example,
                "default": d.default,
            }
            for d in contract.deliverables
        ],
        "constraints": {
            "max_input_tokens": contract.constraints.max_input_tokens,
            "max_output_tokens": contract.constraints.max_output_tokens,
            "max_total_tokens": contract.constraints.max_total_tokens,
            "max_tool_calls": contract.constraints.max_tool_calls,
            "timeout_seconds": contract.constraints.timeout_seconds,
            "warn_threshold": contract.constraints.warn_threshold,
        },
        "failure_strategy": contract.failure_strategy.value,
        "max_retries": contract.max_retries,
        "version": contract.version,
        "metadata": contract.metadata,
    }


def contract_from_dict(data: dict[str, Any]) -> Contract:
    """Deserialize Contract from dictionary."""
    deliverables = [
        Deliverable(
            name=d["name"],
            type=d["type"],
            description=d["description"],
            required=d.get("required", True),
            validation_rules=d.get("validation_rules", []),
            example=d.get("example"),
            default=d.get("default"),
        )
        for d in data["deliverables"]
    ]

    constraints_data = data.get("constraints", {})
    constraints = ContractConstraints(
        max_input_tokens=constraints_data.get("max_input_tokens"),
        max_output_tokens=constraints_data.get("max_output_tokens"),
        max_total_tokens=constraints_data.get("max_total_tokens"),
        max_tool_calls=constraints_data.get("max_tool_calls"),
        timeout_seconds=constraints_data.get("timeout_seconds"),
        warn_threshold=constraints_data.get("warn_threshold", 0.8),
    )

    return Contract(
        name=data["name"],
        description=data["description"],
        deliverables=deliverables,
        constraints=constraints,
        failure_strategy=FailureStrategy(data.get("failure_strategy", "retry")),
        max_retries=data.get("max_retries", 2),
        version=data.get("version", "1.0.0"),
        metadata=data.get("metadata", {}),
    )
```

---

## 15. Integration Examples

### 15.1 Basic Contract Validation

```python
from sigil.contracts import (
    Contract, Deliverable, ContractConstraints,
    ContractValidator, get_contract_template
)

# Using template
contract = get_contract_template("lead_qualification")
validator = ContractValidator()

output = {
    "qualification_score": 75,
    "bant_assessment": {
        "budget": {"score": 80},
        "authority": {"score": 70},
        "need": {"score": 90},
        "timeline": {"score": 60},
    },
    "recommended_action": "Schedule product demo",
}

result = validator.validate(output, contract)

if result.is_valid:
    print("Output is valid!")
else:
    print("Validation failed:")
    for error in result.errors:
        print(f"  - {error.field}: {error.reason}")
```

### 15.2 Contract-Enforced Execution

```python
from sigil.contracts import (
    ContractExecutor, ContractExecutionRequest,
    ContractTemplateFactory
)

executor = ContractExecutor()

request = ContractExecutionRequest(
    agent_id="lead_qualifier",
    task="Qualify John Smith from Acme Corp for our enterprise SaaS",
    contract_name="lead_qualification",
    context={
        "lead_info": "John is CTO, company has 200 employees",
    },
)

result = await executor.execute(request)

print(f"Valid: {result.is_valid}")
print(f"Strategy: {result.applied_strategy}")
print(f"Attempts: {result.attempts}")
print(f"Tokens: {result.tokens_used}")

if result.is_valid:
    print(f"Score: {result.output['qualification_score']}")
```

### 15.3 Custom Contract Creation

```python
from sigil.contracts import (
    Contract, Deliverable, ContractConstraints,
    FailureStrategy
)

# Create custom contract for email drafting
email_contract = Contract(
    name="email_draft",
    description="Structured email draft with subject and body",
    deliverables=[
        Deliverable(
            name="subject",
            type="str",
            description="Email subject line",
            required=True,
            validation_rules=["len(value) > 0", "len(value) < 100"],
        ),
        Deliverable(
            name="body",
            type="str",
            description="Email body content",
            required=True,
            validation_rules=["len(value) >= 50"],
        ),
        Deliverable(
            name="call_to_action",
            type="str",
            description="Primary CTA",
            required=False,
        ),
    ],
    constraints=ContractConstraints(
        max_total_tokens=2000,
    ),
    failure_strategy=FailureStrategy.FALLBACK,
    max_retries=1,
)

# Use with executor
request = ContractExecutionRequest(
    agent_id="email_writer",
    task="Draft a follow-up email to John about our product demo",
    contract=email_contract,
)

result = await executor.execute(request)
```

### 15.4 Integration with Planning (Phase 5)

```python
from sigil.planning import PlanExecutor
from sigil.contracts import ContractExecutor, get_contract_template

# Create plan executor with contract support
contract_executor = ContractExecutor()
plan_executor = PlanExecutor(contract_executor=contract_executor)

# Execute plan with contract on final step
plan = await planner.generate(
    goal="Qualify and respond to lead John from Acme Corp"
)

# Add contract to final output step
final_step = plan.steps[-1]
final_step.contract_spec = get_contract_template("lead_qualification")

# Execute with contract enforcement
result = await plan_executor.execute(plan)
```

### 15.5 Agent Tools Integration

```python
from sigil.contracts.tools import (
    create_validate_contract_tool,
    create_get_contract_template_tool,
    create_enforce_contract_tool,
)
from sigil.contracts import ContractExecutor, ContractTemplateFactory, ContractValidator

# Create tools
factory = ContractTemplateFactory()
validator = ContractValidator()
executor = ContractExecutor()

tools = [
    create_validate_contract_tool(factory, validator),
    create_get_contract_template_tool(factory),
    create_enforce_contract_tool(executor),
]

# Use with LangChain agent
from langchain.agents import AgentExecutor

agent = AgentExecutor(
    agent=...,
    tools=tools,
)

# Agent can now self-validate outputs
response = agent.invoke({
    "input": "Qualify the lead and validate your output against lead_qualification contract"
})
```

---

## 16. Implementation Guidance

### 16.1 Phase Order

**Phase 6.1 - Core Schemas (Days 1-2):**
1. Implement `Deliverable` dataclass
2. Implement `ContractConstraints` dataclass
3. Implement `Contract` dataclass
4. Write unit tests

**Phase 6.2 - Validation (Days 3-4):**
1. Implement `ValidationError` and `ValidationResult`
2. Implement `ContractValidator`
3. Write validation tests

**Phase 6.3 - Retry & Fallback (Days 5-6):**
1. Implement `RetryManager` with refinement levels
2. Implement `FallbackHandler`
3. Write retry/fallback tests

**Phase 6.4 - Executor (Days 7-8):**
1. Implement `ContractExecutor`
2. Integrate validator, retry, fallback
3. Write integration tests

**Phase 6.5 - Templates (Days 9-10):**
1. Implement `ContractTemplateFactory`
2. Create ACTi templates
3. Write template tests

**Phase 6.6 - Tools & Integration (Days 11-12):**
1. Implement LangChain tools
2. Integrate with Phase 5 (Planning)
3. Write end-to-end tests

### 16.2 File Locations

```
sigil/contracts/
|-- __init__.py                 # Module exports
|-- schema.py                   # Contract, Deliverable, Constraints
|-- validator.py                # ContractValidator
|-- executor.py                 # ContractExecutor
|-- retry.py                    # RetryManager
|-- fallback.py                 # FallbackHandler
|-- errors.py                   # Exception classes
|-- templates/
|   |-- __init__.py
|   |-- factory.py              # ContractTemplateFactory
|   |-- acti.py                 # ACTi templates
|-- tools/
|   |-- __init__.py
|   |-- langchain.py            # LangChain tools

tests/unit/test_contracts/
|-- test_schema.py
|-- test_validator.py
|-- test_executor.py
|-- test_retry.py
|-- test_fallback.py
|-- test_templates.py

tests/integration/
|-- test_contracts_integration.py
|-- test_contracts_planning_integration.py
```

### 16.3 Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `pydantic` | >= 2.0 | Schema validation |
| `langchain` | >= 0.1.0 | Tool integration |
| `structlog` | >= 24.0 | Structured logging |

### 16.4 Testing Requirements

```
Unit Tests:
- Deliverable type validation
- Contract serialization/deserialization
- Validator error detection
- Retry refinement levels
- Fallback generation

Integration Tests:
- Full validation flow
- Retry with agent execution
- Fallback strategy application
- Template factory usage

E2E Tests:
- Complete contract execution
- Integration with Planning (Phase 5)
- Tool usage by agents
```

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2026-01-11 | Initial API contract |

---

*Document maintained by: API Architecture Team*
*Last updated: 2026-01-11*
