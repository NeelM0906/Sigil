# API Guidelines: Phase 6 Contracts System

## Overview

This document provides implementation guidance for the Phase 6 Contracts API. It supplements the OpenAPI specification with Python SDK contracts, naming conventions, security notes, and example implementations.

---

## Table of Contents

1. [Python SDK Contracts](#1-python-sdk-contracts)
2. [Naming Conventions](#2-naming-conventions)
3. [Required Headers](#3-required-headers)
4. [Error Handling](#4-error-handling)
5. [Rate Limiting](#5-rate-limiting)
6. [Authentication](#6-authentication)
7. [Example Implementations](#7-example-implementations)
8. [Integration Patterns](#8-integration-patterns)
9. [Testing Guidelines](#9-testing-guidelines)

---

## 1. Python SDK Contracts

### 1.1 ContractValidator Class

```python
from typing import Optional, Dict, Any
from datetime import datetime

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

    def __init__(self, strict_mode: bool = False) -> None:
        """Initialize validator.

        Args:
            strict_mode: If True, stop at first error; if False, collect all errors.
        """
        ...

    def validate(
        self,
        output: Dict[str, Any],
        contract: "Contract",
        context: Optional[Dict[str, Any]] = None,
    ) -> "ValidationResult":
        """Validate output against contract.

        Args:
            output: Agent output to validate (dict)
            contract: Contract specification
            context: Optional execution context (for constraint checking)

        Returns:
            ValidationResult with is_valid and any errors

        Raises:
            ContractNotFoundError: If contract_name provided but not found

        Example:
            >>> validator = ContractValidator()
            >>> result = validator.validate(
            ...     output={"qualification_score": 75, "bant_assessment": {}},
            ...     contract=lead_qualification_contract,
            ... )
            >>> if not result.is_valid:
            ...     print(result.to_summary())
        """
        ...

    def validate_deliverable(
        self,
        value: Any,
        deliverable: "Deliverable",
    ) -> list["ValidationError"]:
        """Validate a single deliverable.

        Args:
            value: Value to validate
            deliverable: Deliverable specification

        Returns:
            List of validation errors (empty if valid)
        """
        ...
```

### 1.2 ContractExecutor Class

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
    ) -> None:
        """Initialize executor with dependencies.

        Args:
            validator: ContractValidator instance. Auto-created if None.
            retry_manager: RetryManager for retry logic. Auto-created if None.
            fallback_handler: FallbackHandler for degradation. Auto-created if None.
            template_factory: Factory for contract templates. Auto-created if None.
            event_store: EventStore for audit trail. Optional.
            token_tracker: TokenTracker for usage tracking. Optional.
        """
        ...

    async def execute(
        self,
        request: "ContractExecutionRequest",
    ) -> "ContractExecutionResult":
        """Execute task with contract enforcement.

        Args:
            request: Execution request with agent, task, and contract

        Returns:
            ContractExecutionResult with validated output

        Raises:
            ContractViolation: If strategy is FAIL and validation fails
            ContractNotFoundError: If contract_name not found

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
        ...

    async def execute_and_validate(
        self,
        agent: Any,
        task: str,
        contract: "Contract",
    ) -> tuple[Dict[str, Any], "ValidationResult", int]:
        """Execute agent and validate output.

        Lower-level method for custom execution flows.

        Args:
            agent: Agent instance to execute
            task: Task description
            contract: Contract specification

        Returns:
            Tuple of (output, validation_result, tokens_used)
        """
        ...
```

### 1.3 RetryManager Class

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
        Level 2 (Medium): Add explicit format instructions
        Level 3 (Heavy): Provide template structure
    """

    def __init__(
        self,
        config: Optional["RetryConfiguration"] = None,
    ) -> None:
        """Initialize retry manager.

        Args:
            config: Retry configuration. Uses defaults if None.
        """
        ...

    def should_retry(
        self,
        errors: list["ValidationError"],
        attempt: int,
    ) -> bool:
        """Determine if retry is appropriate.

        Args:
            errors: Validation errors from last attempt
            attempt: Current attempt number (1-indexed)

        Returns:
            True if retry should be attempted

        Logic:
            - Returns False if attempt >= max_retries + 1
            - Returns False if any error type is in dont_retry list
            - Returns True if any error type is in retry_on_errors list
        """
        ...

    def classify_errors(
        self,
        errors: list["ValidationError"],
    ) -> Dict[str, list["ValidationError"]]:
        """Classify errors by type for targeted refinement.

        Args:
            errors: List of validation errors

        Returns:
            Dict mapping error type to list of errors
        """
        ...

    def refine_prompt(
        self,
        original_task: str,
        errors: list["ValidationError"],
        contract: "Contract",
        level: int = 1,
    ) -> str:
        """Refine prompt based on validation errors.

        Args:
            original_task: Original task description
            errors: Validation errors to address
            contract: Contract specification
            level: Refinement level (1-3)

        Returns:
            Refined task prompt with error context

        Refinement Levels:
            Level 1: Add notes about missing/invalid fields
            Level 2: Add explicit format specification
            Level 3: Include full template structure
        """
        ...

    def get_delay(self, attempt: int) -> int:
        """Get delay in milliseconds before retry.

        Args:
            attempt: Attempt number (1-indexed)

        Returns:
            Delay in milliseconds (from retry_delays config)
        """
        ...
```

### 1.4 FallbackHandler Class

```python
class FallbackHandler:
    """Handles graceful degradation when validation fails.

    Fallback Strategies:
        - FALLBACK/PARTIAL: Return best-effort partial result
        - TEMPLATE: Generate from contract template
    """

    def generate_fallback(
        self,
        output: Dict[str, Any],
        contract: "Contract",
        validation: "ValidationResult",
        strategy: "FailureStrategy",
    ) -> "FallbackResult":
        """Generate fallback result.

        Args:
            output: Original (invalid) output
            contract: Contract specification
            validation: Validation result with errors
            strategy: Fallback strategy to apply

        Returns:
            FallbackResult with best-effort output

        Strategy Behavior:
            FALLBACK/PARTIAL:
                - Keep valid fields from output
                - Fill missing with defaults/examples
                - Mark missing fields that couldn't be filled

            TEMPLATE:
                - Generate entirely from contract template
                - Use examples and defaults
                - Mark result as template-generated
        """
        ...
```

### 1.5 ContractTemplateFactory Class

```python
class ContractTemplateFactory:
    """Factory for retrieving and creating contract templates.

    Built-in Templates:
        - lead_qualification: BANT-based lead scoring
        - research_report: Structured research output
        - appointment_booking: Calendar event creation
        - market_analysis: Competitive analysis
        - compliance_check: Governance validation
    """

    def __init__(self) -> None:
        """Initialize with default templates."""
        ...

    def get_template(self, name: str) -> Optional["Contract"]:
        """Get contract template by name.

        Args:
            name: Template name (e.g., "lead_qualification")

        Returns:
            Contract instance or None if not found

        Example:
            >>> factory = ContractTemplateFactory()
            >>> contract = factory.get_template("lead_qualification")
            >>> print(contract.deliverables)
        """
        ...

    def list_templates(self) -> list[str]:
        """List all available template names.

        Returns:
            List of template names
        """
        ...

    def register_template(self, contract: "Contract") -> None:
        """Register a custom contract template.

        Args:
            contract: Contract to register
        """
        ...

    def create_custom(
        self,
        name: str,
        description: str,
        deliverables: list["Deliverable"],
        constraints: Optional["ContractConstraints"] = None,
        failure_strategy: "FailureStrategy" = "retry",
        max_retries: int = 2,
    ) -> "Contract":
        """Create a custom contract.

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
        ...
```

---

## 2. Naming Conventions

### 2.1 Identifiers

| Entity | Pattern | Example |
|--------|---------|---------|
| Contract Name | `snake_case` | `lead_qualification` |
| Deliverable Name | `snake_case` | `qualification_score` |
| Event ID | `evt_{uuid}` | `evt_123e4567-e89b-12d3-a456` |
| Session ID | `sess_{uuid}` | `sess_abc123def456` |
| Hash (output/prompt) | `sha256_{first16}` | `sha256_a1b2c3d4` |

### 2.2 Field Names

| Style | Usage | Examples |
|-------|-------|----------|
| snake_case | All field names | `qualification_score`, `bant_assessment` |
| PascalCase | Class names | `ContractExecutor`, `ValidationResult` |
| lowercase | Event types | `contract.validated`, `contract.retry` |
| SCREAMING_SNAKE | Constants | `MAX_RETRIES`, `DEFAULT_STRATEGY` |

### 2.3 Error Codes

| Pattern | Meaning |
|---------|---------|
| CV-001 | Contract Validation error (schema) |
| CV-002 | Missing Deliverable |
| CV-003 | Type Mismatch |
| CV-004 | Rule Violation |
| CV-005 | Constraint Violation |
| CV-006 | Agent Execution Error |
| CV-007 | Timeout Error |
| CV-008 | Contract Violation (fail strategy) |
| CV-009 | Contract Not Found |
| CV-010 | Invalid Contract Spec |

---

## 3. Required Headers

### 3.1 Request Headers

| Header | Required | Description | Example |
|--------|----------|-------------|---------|
| `Content-Type` | Yes | Media type | `application/json` |
| `Accept` | Recommended | Accepted response types | `application/json` |
| `Authorization` | Conditional | Bearer token (HTTP API) | `Bearer eyJhbG...` |
| `X-Request-ID` | Recommended | Client-generated request ID | `req_abc123` |
| `X-Session-ID` | Conditional | Session ID for tracking | `sess_xyz789` |

### 3.2 Response Headers

| Header | Description | Example |
|--------|-------------|---------|
| `Content-Type` | Response media type | `application/json` |
| `X-Request-ID` | Echo of client request ID | `req_abc123` |
| `X-Correlation-ID` | Server-generated correlation ID | `corr_456def` |
| `X-Contract-Name` | Contract used for validation | `lead_qualification` |
| `X-Validation-Time-Ms` | Time spent in validation | `15` |
| `X-Tokens-Used` | Tokens consumed by request | `3200` |

---

## 4. Error Handling

### 4.1 Error Taxonomy

| Code | Name | HTTP Status | Trigger | Recovery |
|------|------|-------------|---------|----------|
| CV-001 | ValidationError | 400 | Output doesn't match schema | Retry or fallback |
| CV-002 | MissingDeliverable | 400 | Required field missing | Retry with context |
| CV-003 | TypeMismatch | 400 | Wrong data type | Retry with explicit type |
| CV-004 | RuleViolation | 400 | Validation rule failed | Retry or fallback |
| CV-005 | ConstraintViolation | 400 | Token/tool limits exceeded | Escalate to fail |
| CV-006 | AgentExecutionError | 500 | Agent failed before validation | Retry or escalate |
| CV-007 | TimeoutError | 504 | Execution exceeded timeout | Fallback |
| CV-008 | ContractViolation | 422 | User requested strategy=fail | Return error |
| CV-009 | ContractNotFound | 404 | Template name doesn't exist | Check name |
| CV-010 | InvalidContractSpec | 400 | Contract definition invalid | Fix spec |

### 4.2 Error Response Format (RFC 9457)

```json
{
  "type": "https://sigil.io/errors/contract-violation",
  "title": "Contract Violation",
  "status": 422,
  "detail": "Contract 'lead_qualification' violated after 3 attempts: missing required field 'recommended_action'",
  "instance": "/api/v1/contracts/execute",
  "code": "CV-008",
  "contract_name": "lead_qualification",
  "attempts": 3,
  "errors": [
    {
      "field": "recommended_action",
      "error_type": "missing",
      "reason": "Required field 'recommended_action' is missing"
    }
  ],
  "recovery_suggestions": [
    "Review agent output format",
    "Consider using fallback strategy",
    "Simplify contract requirements"
  ]
}
```

### 4.3 Python Exception Classes

```python
class ContractError(SigilError):
    """Base exception for contract errors."""
    code = "CV-000"


class ValidationError(ContractError):
    """Output validation failed."""
    code = "CV-001"


class MissingDeliverableError(ContractError):
    """Required deliverable missing."""
    code = "CV-002"


class TypeMismatchError(ContractError):
    """Field has wrong type."""
    code = "CV-003"


class RuleViolationError(ContractError):
    """Business rule validation failed."""
    code = "CV-004"


class ConstraintViolationError(ContractError):
    """Resource constraint exceeded."""
    code = "CV-005"


class ContractViolation(ContractError):
    """Contract validation failed with fail strategy."""
    code = "CV-008"


class ContractNotFoundError(ContractError):
    """Contract template not found."""
    code = "CV-009"
```

---

## 5. Rate Limiting

### 5.1 Default Limits

| Resource | Limit | Window | Scope |
|----------|-------|--------|-------|
| Validation requests | 100 | 1 minute | Per session |
| Execution requests | 20 | 1 minute | Per session |
| Contract creation | 10 | 1 minute | Per session |
| Concurrent executions | 5 | - | Per session |

### 5.2 Rate Limit Headers

```http
HTTP/1.1 429 Too Many Requests
Content-Type: application/json
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 0
X-RateLimit-Reset: 1704974460
Retry-After: 60

{
  "type": "https://sigil.io/errors/rate-limit-exceeded",
  "title": "Rate Limit Exceeded",
  "status": 429,
  "detail": "Validation request rate limit exceeded. Maximum 100 requests per minute.",
  "retry_after": 60
}
```

---

## 6. Authentication

### 6.1 Internal SDK Usage

No authentication required for direct Python SDK usage within the Sigil framework.

### 6.2 HTTP API

Bearer token authentication using JWT tokens:

```http
POST /api/v1/contracts/execute HTTP/1.1
Host: localhost:8000
Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
Content-Type: application/json

{
  "agent_id": "lead_qualifier",
  "task": "Qualify John from Acme Corp",
  "contract_name": "lead_qualification"
}
```

### 6.3 Security Considerations

1. **Output Validation**: Outputs may contain sensitive information. Apply data minimization.
2. **Validation Rules**: Rules are evaluated using safe eval. Never allow user-provided rules without sanitization.
3. **Retry Prompts**: Refined prompts include error context. Consider PII implications.
4. **Event Hashing**: Use SHA256 hashes for output/prompt content to preserve privacy while enabling correlation.

---

## 7. Example Implementations

### 7.1 Basic Contract Validation

```python
from sigil.contracts import (
    Contract, Deliverable, ContractConstraints,
    ContractValidator, get_contract_template
)

async def validate_lead_qualification():
    """Example: Validate lead qualification output."""

    # Get template
    contract = get_contract_template("lead_qualification")
    validator = ContractValidator()

    # Output from agent
    output = {
        "qualification_score": 75,
        "bant_assessment": {
            "budget": {"score": 80, "notes": "Budget approved"},
            "authority": {"score": 70, "notes": "Decision maker identified"},
            "need": {"score": 90, "notes": "Strong pain point"},
            "timeline": {"score": 60, "notes": "Q2 implementation"},
        },
        "recommended_action": "Schedule product demo",
        "confidence": 0.85,
    }

    # Validate
    result = validator.validate(output, contract)

    if result.is_valid:
        print("Validation PASSED")
    else:
        print("Validation FAILED:")
        for error in result.errors:
            print(f"  - {error.field}: {error.reason}")

        if result.suggestion:
            print(f"\nSuggestion: {result.suggestion}")

    return result
```

### 7.2 Contract-Enforced Execution

```python
from sigil.contracts import (
    ContractExecutor,
    ContractExecutionRequest,
    FailureStrategy,
)

async def execute_with_contract():
    """Example: Execute agent with contract enforcement."""

    executor = ContractExecutor()

    request = ContractExecutionRequest(
        agent_id="lead_qualifier",
        task="Qualify John Smith from Acme Corp for our enterprise SaaS platform",
        contract_name="lead_qualification",
        context={
            "lead_info": "John is CTO, company has 200 employees",
            "previous_interactions": "Attended webinar last week",
        },
        session_id="sess_abc123",
    )

    try:
        result = await executor.execute(request)

        print(f"Execution completed:")
        print(f"  Valid: {result.is_valid}")
        print(f"  Strategy: {result.applied_strategy}")
        print(f"  Attempts: {result.attempts}")
        print(f"  Tokens: {result.tokens_used}")

        if result.is_valid:
            print(f"\nOutput:")
            print(f"  Score: {result.output['qualification_score']}")
            print(f"  Action: {result.output['recommended_action']}")
        else:
            print(f"\nFallback output (not fully validated)")

    except ContractViolation as e:
        print(f"Contract violated: {e.message}")
        print(f"Attempts: {e.attempts}")
        for error in e.errors[:3]:
            print(f"  - {error.field}: {error.reason}")

    return result
```

### 7.3 Custom Contract Creation

```python
from sigil.contracts import (
    Contract,
    Deliverable,
    ContractConstraints,
    FailureStrategy,
)

def create_email_contract():
    """Example: Create custom email drafting contract."""

    email_contract = Contract(
        name="email_draft",
        description="Structured email draft with subject, body, and CTA",
        deliverables=[
            Deliverable(
                name="subject",
                type="str",
                description="Email subject line",
                required=True,
                validation_rules=[
                    "len(value) > 0",
                    "len(value) < 100",
                ],
                example="Follow-up: Your Product Demo Request",
            ),
            Deliverable(
                name="body",
                type="str",
                description="Email body content",
                required=True,
                validation_rules=[
                    "len(value) >= 50",
                    "len(value) <= 2000",
                ],
            ),
            Deliverable(
                name="call_to_action",
                type="str",
                description="Primary call-to-action",
                required=False,
                example="Schedule a call",
            ),
            Deliverable(
                name="tone",
                type="str",
                description="Email tone",
                required=False,
                validation_rules=[
                    "value in ['formal', 'friendly', 'urgent', 'casual']",
                ],
                default="friendly",
            ),
        ],
        constraints=ContractConstraints(
            max_total_tokens=2000,
            max_tool_calls=2,
            timeout_seconds=30,
        ),
        failure_strategy=FailureStrategy.FALLBACK,
        max_retries=1,
        version="1.0.0",
        metadata={
            "author": "sales_team",
            "use_case": "follow_up_emails",
        },
    )

    return email_contract
```

### 7.4 Retry with Custom Configuration

```python
from sigil.contracts import (
    ContractExecutor,
    RetryManager,
    RetryConfiguration,
)

async def execute_with_aggressive_retry():
    """Example: Execute with aggressive retry configuration."""

    # Custom retry configuration
    retry_config = RetryConfiguration(
        max_retries=3,
        retry_delays=[0, 500, 1000, 2000],
        retry_on_errors=["missing", "type", "rule"],
        dont_retry=["constraint"],  # Don't retry constraint violations
        refinement_level=2,  # Start at medium refinement
    )

    retry_manager = RetryManager(config=retry_config)
    executor = ContractExecutor(retry_manager=retry_manager)

    request = ContractExecutionRequest(
        agent_id="researcher",
        task="Generate comprehensive market analysis for CRM software",
        contract_name="market_analysis",
    )

    result = await executor.execute(request)

    print(f"Completed in {result.attempts} attempt(s)")
    print(f"Strategy applied: {result.applied_strategy}")

    return result
```

### 7.5 Manual Fallback Handling

```python
from sigil.contracts import (
    ContractValidator,
    FallbackHandler,
    FailureStrategy,
    get_contract_template,
)

async def handle_validation_with_fallback():
    """Example: Manual validation with explicit fallback handling."""

    contract = get_contract_template("research_report")
    validator = ContractValidator()
    fallback = FallbackHandler()

    # Partially valid output
    output = {
        "title": "Market Research Report",
        "summary": "This is a brief summary.",  # Too short!
        "findings": ["Finding 1", "Finding 2"],
        # Missing: recommendations, sources
    }

    result = validator.validate(output, contract)

    if not result.is_valid:
        print(f"Validation failed with {result.error_count} errors")

        # Generate fallback
        fallback_result = fallback.generate_fallback(
            output=output,
            contract=contract,
            validation=result,
            strategy=FailureStrategy.PARTIAL,
        )

        print(f"\nFallback generated:")
        print(f"  Filled from: {fallback_result.filled_from}")
        print(f"  Missing: {fallback_result.missing_deliverables}")
        print(f"  Warnings: {fallback_result.warnings}")

        return fallback_result.output

    return output
```

---

## 8. Integration Patterns

### 8.1 Integration with Planning (Phase 5)

```python
from sigil.planning import Planner, PlanExecutor
from sigil.contracts import ContractExecutor, get_contract_template

async def plan_with_contract_verification():
    """Execute plan with contract on final step."""

    planner = Planner()
    contract_executor = ContractExecutor()
    plan_executor = PlanExecutor(contract_executor=contract_executor)

    # Generate plan
    plan = await planner.generate(
        goal="Qualify and respond to lead John from Acme Corp"
    )

    # Add contract to final output step
    final_step = plan.steps[-1]
    final_step.contract_spec = get_contract_template("lead_qualification")

    # Execute with contract enforcement
    result = await plan_executor.execute(plan)

    return result
```

### 8.2 Integration with Router (Phase 3)

```python
from sigil.routing import Router
from sigil.contracts import ContractTemplateFactory

class ContractAwareRouter(Router):
    """Router that suggests contracts based on intent."""

    INTENT_CONTRACTS = {
        "qualify_lead": "lead_qualification",
        "research": "research_report",
        "schedule_meeting": "appointment_booking",
        "analyze_market": "market_analysis",
        "check_compliance": "compliance_check",
    }

    def __init__(self):
        super().__init__()
        self.template_factory = ContractTemplateFactory()

    def route(self, message: str) -> "RouteDecision":
        decision = super().route(message)

        # Suggest contract based on intent
        if decision.intent in self.INTENT_CONTRACTS:
            contract_name = self.INTENT_CONTRACTS[decision.intent]
            decision.suggested_contract = contract_name

        return decision
```

### 8.3 Integration with Memory (Phase 4)

```python
from sigil.memory import MemoryManager
from sigil.contracts import ContractExecutor, ContractExecutionRequest

async def execute_with_memory_context():
    """Execute contract with memory-enhanced context."""

    memory = MemoryManager()
    executor = ContractExecutor()

    # Retrieve relevant context from memory
    context = await memory.retrieve(
        query="John Smith Acme Corp",
        mode="hybrid",
        k=10,
    )

    # Include in contract execution
    request = ContractExecutionRequest(
        agent_id="lead_qualifier",
        task="Qualify John Smith from Acme Corp",
        contract_name="lead_qualification",
        context={
            "memory_facts": [item.content for item in context.items],
            "memory_sources": [item.source_resource_id for item in context.items],
        },
    )

    result = await executor.execute(request)

    # Store result in memory for future reference
    if result.is_valid:
        await memory.store_resource(
            content=json.dumps(result.output),
            resource_type="qualification_result",
            metadata={
                "lead_name": "John Smith",
                "company": "Acme Corp",
                "score": result.output["qualification_score"],
            },
        )

    return result
```

### 8.4 LangChain Agent Tools

```python
from sigil.contracts.tools import (
    create_validate_contract_tool,
    create_get_contract_template_tool,
    create_enforce_contract_tool,
)
from langchain.agents import AgentExecutor

def create_contract_aware_agent():
    """Create LangChain agent with contract tools."""

    from sigil.contracts import (
        ContractExecutor,
        ContractTemplateFactory,
        ContractValidator,
    )

    # Initialize contract components
    factory = ContractTemplateFactory()
    validator = ContractValidator()
    executor = ContractExecutor()

    # Create tools
    tools = [
        create_validate_contract_tool(factory, validator),
        create_get_contract_template_tool(factory),
        create_enforce_contract_tool(executor),
    ]

    # Create agent with tools
    agent = AgentExecutor(
        agent=...,  # Your agent
        tools=tools,
        verbose=True,
    )

    return agent


# Usage
agent = create_contract_aware_agent()
response = agent.invoke({
    "input": (
        "Qualify the lead John from Acme Corp. "
        "Make sure to validate your output against the lead_qualification contract."
    )
})
```

---

## 9. Testing Guidelines

### 9.1 Unit Test Examples

```python
import pytest
from sigil.contracts import (
    Contract,
    Deliverable,
    ContractConstraints,
    ContractValidator,
    ValidationError,
    ValidationErrorType,
)


class TestDeliverable:
    """Tests for Deliverable class."""

    def test_validate_type_success(self):
        """Test type validation with correct type."""
        deliverable = Deliverable(
            name="score",
            type="int",
            description="Test score",
        )
        assert deliverable.validate_type(42) is True
        assert deliverable.validate_type("42") is False

    def test_validate_rules_success(self):
        """Test rule validation with passing rules."""
        deliverable = Deliverable(
            name="score",
            type="int",
            description="Test score",
            validation_rules=["value >= 0", "value <= 100"],
        )
        failed = deliverable.validate_rules(50)
        assert failed == []

    def test_validate_rules_failure(self):
        """Test rule validation with failing rules."""
        deliverable = Deliverable(
            name="score",
            type="int",
            description="Test score",
            validation_rules=["value >= 0", "value <= 100"],
        )
        failed = deliverable.validate_rules(150)
        assert "value <= 100" in failed


class TestContractValidator:
    """Tests for ContractValidator class."""

    @pytest.fixture
    def simple_contract(self):
        """Create simple contract for testing."""
        return Contract(
            name="test_contract",
            description="Test contract",
            deliverables=[
                Deliverable(
                    name="required_field",
                    type="str",
                    description="Required string",
                    required=True,
                ),
                Deliverable(
                    name="optional_field",
                    type="int",
                    description="Optional integer",
                    required=False,
                ),
            ],
        )

    def test_validate_valid_output(self, simple_contract):
        """Test validation with valid output."""
        validator = ContractValidator()
        output = {"required_field": "hello", "optional_field": 42}

        result = validator.validate(output, simple_contract)

        assert result.is_valid is True
        assert result.error_count == 0

    def test_validate_missing_required(self, simple_contract):
        """Test validation with missing required field."""
        validator = ContractValidator()
        output = {"optional_field": 42}

        result = validator.validate(output, simple_contract)

        assert result.is_valid is False
        assert result.error_count == 1
        assert "required_field" in result.get_missing_fields()

    def test_validate_type_mismatch(self, simple_contract):
        """Test validation with type mismatch."""
        validator = ContractValidator()
        output = {"required_field": 123}  # Should be str

        result = validator.validate(output, simple_contract)

        assert result.is_valid is False
        assert "required_field" in result.get_type_mismatches()

    def test_strict_mode_stops_on_first_error(self, simple_contract):
        """Test strict mode stops at first error."""
        validator = ContractValidator(strict_mode=True)
        output = {"required_field": 123}  # Type error AND missing optional

        result = validator.validate(output, simple_contract)

        assert result.error_count == 1  # Only first error


class TestRetryManager:
    """Tests for RetryManager class."""

    def test_should_retry_within_limit(self):
        """Test retry allowed within limit."""
        from sigil.contracts import RetryManager, ValidationError, ValidationErrorType

        manager = RetryManager()
        errors = [
            ValidationError(
                field="score",
                error_type=ValidationErrorType.MISSING,
                reason="Missing",
                expected="int",
                actual="<missing>",
            )
        ]

        assert manager.should_retry(errors, attempt=1) is True
        assert manager.should_retry(errors, attempt=2) is True
        assert manager.should_retry(errors, attempt=3) is False  # max_retries=2

    def test_refine_prompt_level_1(self):
        """Test prompt refinement at level 1."""
        from sigil.contracts import RetryManager, Contract, Deliverable

        manager = RetryManager()
        contract = Contract(
            name="test",
            description="Test",
            deliverables=[
                Deliverable(
                    name="score",
                    type="int",
                    description="Score",
                )
            ],
        )
        errors = [
            ValidationError(
                field="score",
                error_type=ValidationErrorType.MISSING,
                reason="Missing",
                expected="int",
                actual="<missing>",
            )
        ]

        refined = manager.refine_prompt(
            original_task="Calculate score",
            errors=errors,
            contract=contract,
            level=1,
        )

        assert "Calculate score" in refined
        assert "score" in refined  # Error context added
```

### 9.2 Integration Test Examples

```python
import pytest
from sigil.contracts import (
    ContractExecutor,
    ContractExecutionRequest,
    ContractTemplateFactory,
    FailureStrategy,
)


@pytest.mark.asyncio
async def test_full_execution_flow():
    """Test complete contract execution flow."""
    executor = ContractExecutor()

    request = ContractExecutionRequest(
        agent_id="test_agent",
        task="Generate test output",
        contract_name="lead_qualification",
    )

    result = await executor.execute(request)

    assert result.output is not None
    assert result.attempts >= 1
    assert result.tokens_used > 0
    assert result.applied_strategy in ["success", "retry", "fallback"]


@pytest.mark.asyncio
async def test_retry_on_validation_failure():
    """Test retry behavior on validation failure."""
    executor = ContractExecutor()

    # Use a contract that will likely fail first attempt
    request = ContractExecutionRequest(
        agent_id="test_agent",
        task="Generate incomplete output",  # Likely to fail validation
        contract_name="lead_qualification",
    )

    result = await executor.execute(request)

    # Should have attempted retries
    if not result.validation_result.is_valid:
        assert result.attempts > 1 or result.applied_strategy == "fallback"


@pytest.mark.asyncio
async def test_fail_strategy_raises_exception():
    """Test that fail strategy raises ContractViolation."""
    from sigil.contracts import ContractViolation, Contract, Deliverable

    executor = ContractExecutor()

    # Create impossible-to-satisfy contract with fail strategy
    impossible_contract = Contract(
        name="impossible",
        description="Impossible contract",
        deliverables=[
            Deliverable(
                name="impossible_field",
                type="str",
                description="Must satisfy impossible rule",
                required=True,
                validation_rules=["len(value) > 1000000"],  # Impossible
            )
        ],
        failure_strategy=FailureStrategy.FAIL,
        max_retries=0,
    )

    request = ContractExecutionRequest(
        agent_id="test_agent",
        task="Generate output",
        contract=impossible_contract,
    )

    with pytest.raises(ContractViolation) as exc_info:
        await executor.execute(request)

    assert exc_info.value.contract_name == "impossible"
```

### 9.3 Mock Fixtures

```python
import pytest
from unittest.mock import AsyncMock, MagicMock


@pytest.fixture
def mock_validator():
    """Create mock validator for testing."""
    validator = MagicMock()
    validator.validate = MagicMock(
        return_value=ValidationResult(
            is_valid=True,
            errors=[],
            warnings=[],
            validation_time_ms=10,
            contract_name="test_contract",
            contract_version="1.0.0",
        )
    )
    return validator


@pytest.fixture
def mock_retry_manager():
    """Create mock retry manager for testing."""
    manager = MagicMock()
    manager.should_retry = MagicMock(return_value=False)
    manager.refine_prompt = MagicMock(side_effect=lambda task, **kwargs: task)
    manager.get_delay = MagicMock(return_value=0)
    return manager


@pytest.fixture
def mock_executor(mock_validator, mock_retry_manager):
    """Create mock executor with dependencies."""
    executor = ContractExecutor(
        validator=mock_validator,
        retry_manager=mock_retry_manager,
    )
    return executor
```

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2026-01-11 | Initial API guidelines |

---

*Document maintained by: API Architecture Team*
*Last updated: 2026-01-11*
