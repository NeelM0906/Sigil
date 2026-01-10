# API Contract: Error Handling & Recovery (Task 2.5.7)

## Overview

This document defines the comprehensive error handling and recovery specification for the ACTi Agent Builder (Sigil). It establishes a consistent error taxonomy, structured error response formats, recovery strategies, logging standards, and user-facing error display guidelines.

The error handling system is designed to:
- Provide consistent, actionable error messages across all CLI commands
- Enable programmatic error handling via structured error codes
- Support automatic and user-guided recovery from transient failures
- Maintain detailed logs for debugging and audit purposes
- Degrade gracefully when non-critical components fail

---

## Design Principles

1. **Fail-Fast, Recover Gracefully**: Detect errors early, but provide recovery paths where possible
2. **User-First Messages**: Error messages prioritize clarity and actionability over technical detail
3. **Structured for Machines, Readable for Humans**: All errors have codes for programmatic handling and clear messages for users
4. **Defense in Depth**: Multiple layers of error handling (validation, execution, recovery)
5. **Transparent Debugging**: Verbose mode exposes full context without cluttering normal output
6. **No Silent Failures**: All errors are logged and reported; nothing fails silently

---

## Error Categories

### 1. MCP Errors (MCP-xxx)

Errors related to MCP (Model Context Protocol) server connections and tool execution.

| Code | Name | Description | Severity |
|------|------|-------------|----------|
| `MCP-001` | `MCPConnectionError` | Failed to establish connection to MCP server | ERROR |
| `MCP-002` | `MCPTimeoutError` | MCP server connection or operation timed out | ERROR |
| `MCP-003` | `MCPCredentialError` | Required API key or credential is missing | ERROR |
| `MCP-004` | `MCPConfigurationError` | Invalid MCP server configuration | ERROR |
| `MCP-005` | `MCPToolNotFoundError` | Requested tool category does not exist in registry | ERROR |
| `MCP-006` | `MCPToolExecutionError` | Tool invocation failed during execution | ERROR |
| `MCP-007` | `MCPServerCrashError` | MCP server process terminated unexpectedly | ERROR |
| `MCP-008` | `MCPProtocolError` | Invalid MCP protocol message received | ERROR |
| `MCP-009` | `MCPPartialFailureWarning` | Some tools connected, others failed | WARNING |

### 2. Agent Errors (AGT-xxx)

Errors related to agent creation, loading, and execution.

| Code | Name | Description | Severity |
|------|------|-------------|----------|
| `AGT-001` | `AgentNotFoundError` | Agent configuration file does not exist | ERROR |
| `AGT-002` | `AgentConfigInvalidError` | Agent configuration JSON is malformed or invalid | ERROR |
| `AGT-003` | `AgentExecutionError` | Agent failed during task execution | ERROR |
| `AGT-004` | `AgentInstantiationError` | Failed to create agent instance from config | ERROR |
| `AGT-005` | `AgentTimeoutError` | Agent response exceeded timeout threshold | ERROR |
| `AGT-006` | `AgentToolMismatchError` | Agent requested unavailable tools | WARNING |
| `AGT-007` | `AgentHistoryCorruptError` | Conversation history file is corrupted | WARNING |
| `AGT-008` | `AgentVersionMismatchError` | Config version incompatible with current builder | WARNING |
| `AGT-009` | `AgentDuplicateNameError` | Agent name already exists (on create) | ERROR |

### 3. CLI Errors (CLI-xxx)

Errors related to command parsing and CLI operation.

| Code | Name | Description | Severity |
|------|------|-------------|----------|
| `CLI-001` | `UnknownCommandError` | Command not found in command registry | ERROR |
| `CLI-002` | `InvalidArgumentError` | Command argument is invalid or malformed | ERROR |
| `CLI-003` | `MissingArgumentError` | Required argument not provided | ERROR |
| `CLI-004` | `InvalidFlagError` | Unknown or invalid flag used | ERROR |
| `CLI-005` | `SessionStateError` | Invalid session state for operation | ERROR |
| `CLI-006` | `InterruptedError` | User interrupted operation (Ctrl+C) | INFO |
| `CLI-007` | `InputValidationError` | User input failed validation | ERROR |
| `CLI-008` | `ModeTransitionError` | Invalid mode transition attempted | ERROR |

### 4. File Errors (FIL-xxx)

Errors related to file system operations.

| Code | Name | Description | Severity |
|------|------|-------------|----------|
| `FIL-001` | `FileNotFoundError` | Specified file does not exist | ERROR |
| `FIL-002` | `FilePermissionError` | Insufficient permissions to read/write file | ERROR |
| `FIL-003` | `FileParseError` | Failed to parse file content (JSON, YAML, etc.) | ERROR |
| `FIL-004` | `DirectoryNotFoundError` | Required directory does not exist | ERROR |
| `FIL-005` | `DiskSpaceError` | Insufficient disk space for operation | ERROR |
| `FIL-006` | `FileCorruptError` | File exists but content is corrupted | ERROR |
| `FIL-007` | `BackupNotFoundError` | No backup exists for rollback | ERROR |
| `FIL-008` | `FileWriteError` | Failed to write file to disk | ERROR |
| `FIL-009` | `FileLockError` | File is locked by another process | ERROR |

### 5. Network Errors (NET-xxx)

Errors related to network operations and external API calls.

| Code | Name | Description | Severity |
|------|------|-------------|----------|
| `NET-001` | `NetworkUnreachableError` | No network connectivity | ERROR |
| `NET-002` | `APIRateLimitError` | External API rate limit exceeded | ERROR |
| `NET-003` | `ServiceUnavailableError` | External service returned 503 | ERROR |
| `NET-004` | `AuthenticationError` | API authentication failed (401) | ERROR |
| `NET-005` | `AuthorizationError` | API authorization failed (403) | ERROR |
| `NET-006` | `BadGatewayError` | Upstream service error (502) | ERROR |
| `NET-007` | `SSLCertificateError` | SSL/TLS certificate validation failed | ERROR |
| `NET-008` | `DNSResolutionError` | DNS lookup failed for host | ERROR |
| `NET-009` | `ConnectionResetError` | Connection was reset by remote host | ERROR |

### 6. Builder Errors (BLD-xxx)

Errors specific to the agent builder process.

| Code | Name | Description | Severity |
|------|------|-------------|----------|
| `BLD-001` | `BuilderNotInitializedError` | Builder agent not ready | ERROR |
| `BLD-002` | `BuilderPromptError` | Failed to process user prompt | ERROR |
| `BLD-003` | `BuilderToolError` | Builder tool execution failed | ERROR |
| `BLD-004` | `ConfigGenerationError` | Failed to generate valid agent config | ERROR |
| `BLD-005` | `StratumClassificationError` | Could not determine agent stratum | WARNING |
| `BLD-006` | `PromptTooLongError` | User prompt exceeds maximum length | ERROR |
| `BLD-007` | `AmbiguousRequestError` | Request is too vague to process | WARNING |

---

## Error Response Format

### Standard Error Object

All errors in the ACTi system conform to this structure:

```python
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional, Any
import traceback
import uuid


class ErrorSeverity(str, Enum):
    """Severity levels for errors."""
    DEBUG = "debug"       # Development-only information
    INFO = "info"         # Informational (e.g., user cancellation)
    WARNING = "warning"   # Non-fatal issue, operation continues
    ERROR = "error"       # Fatal error, operation failed
    CRITICAL = "critical" # System-level failure


@dataclass
class ACTiError:
    """Standardized error response for all ACTi operations.

    Attributes:
        code: Unique error code (e.g., "MCP-001")
        category: Error category (e.g., "MCP", "CLI", "AGT")
        name: Human-readable error name
        message: User-friendly error description
        severity: Error severity level
        recovery_suggestions: List of actionable recovery steps
        context: Additional context data for debugging
        timestamp: When the error occurred (ISO 8601)
        error_id: Unique identifier for this error instance
        stack_trace: Python stack trace (verbose mode only)
        documentation_url: Link to relevant documentation
    """

    code: str
    category: str
    name: str
    message: str
    severity: ErrorSeverity = ErrorSeverity.ERROR
    recovery_suggestions: list[str] = field(default_factory=list)
    context: dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    error_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    stack_trace: Optional[str] = None
    documentation_url: Optional[str] = None

    def __post_init__(self):
        """Extract category from code if not provided."""
        if not self.category and self.code:
            self.category = self.code.split("-")[0]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "code": self.code,
            "category": self.category,
            "name": self.name,
            "message": self.message,
            "severity": self.severity.value,
            "recovery_suggestions": self.recovery_suggestions,
            "context": self.context,
            "timestamp": self.timestamp,
            "error_id": self.error_id,
            "stack_trace": self.stack_trace,
            "documentation_url": self.documentation_url,
        }

    @classmethod
    def from_exception(
        cls,
        exception: Exception,
        code: str,
        name: str,
        recovery_suggestions: list[str] = None,
        include_trace: bool = False,
    ) -> "ACTiError":
        """Create ACTiError from a Python exception."""
        return cls(
            code=code,
            category=code.split("-")[0],
            name=name,
            message=str(exception),
            recovery_suggestions=recovery_suggestions or [],
            stack_trace=traceback.format_exc() if include_trace else None,
        )
```

### JSON Error Response Format

When errors are serialized (for logging or API responses):

```json
{
  "error": {
    "code": "MCP-003",
    "category": "MCP",
    "name": "MCPCredentialError",
    "message": "Missing credentials for 'calendar': GOOGLE_OAUTH_CREDENTIALS environment variable not set.",
    "severity": "error",
    "recovery_suggestions": [
      "Set the GOOGLE_OAUTH_CREDENTIALS environment variable",
      "Run: export GOOGLE_OAUTH_CREDENTIALS=/path/to/credentials.json",
      "See documentation: https://acti.dev/docs/setup/google-calendar"
    ],
    "context": {
      "tool_name": "calendar",
      "missing_vars": ["GOOGLE_OAUTH_CREDENTIALS"],
      "command": "/load appointment_scheduler"
    },
    "timestamp": "2025-01-09T15:30:45.123456",
    "error_id": "a1b2c3d4",
    "documentation_url": "https://acti.dev/docs/errors/MCP-003"
  }
}
```

---

## Exception Hierarchy

### Base Exception Classes

```python
class ACTiException(Exception):
    """Base exception for all ACTi errors.

    All custom exceptions in the ACTi system inherit from this class,
    ensuring consistent error handling and reporting.
    """

    code: str = "ACT-000"
    name: str = "ACTiException"
    default_message: str = "An unexpected error occurred"
    severity: ErrorSeverity = ErrorSeverity.ERROR
    recovery_suggestions: list[str] = []
    documentation_url: Optional[str] = None

    def __init__(
        self,
        message: str = None,
        context: dict[str, Any] = None,
        recovery_suggestions: list[str] = None,
    ):
        self.message = message or self.default_message
        self.context = context or {}
        if recovery_suggestions:
            self.recovery_suggestions = recovery_suggestions
        super().__init__(self.message)

    def to_acti_error(self, include_trace: bool = False) -> ACTiError:
        """Convert exception to ACTiError for structured handling."""
        return ACTiError(
            code=self.code,
            category=self.code.split("-")[0],
            name=self.name,
            message=self.message,
            severity=self.severity,
            recovery_suggestions=self.recovery_suggestions,
            context=self.context,
            documentation_url=self.documentation_url,
            stack_trace=traceback.format_exc() if include_trace else None,
        )


class MCPException(ACTiException):
    """Base exception for MCP-related errors."""
    code = "MCP-000"
    name = "MCPException"


class AgentException(ACTiException):
    """Base exception for agent-related errors."""
    code = "AGT-000"
    name = "AgentException"


class CLIException(ACTiException):
    """Base exception for CLI-related errors."""
    code = "CLI-000"
    name = "CLIException"


class FileException(ACTiException):
    """Base exception for file-related errors."""
    code = "FIL-000"
    name = "FileException"


class NetworkException(ACTiException):
    """Base exception for network-related errors."""
    code = "NET-000"
    name = "NetworkException"


class BuilderException(ACTiException):
    """Base exception for builder-related errors."""
    code = "BLD-000"
    name = "BuilderException"
```

### MCP Exceptions (Complete Hierarchy)

```python
class MCPConnectionError(MCPException):
    """Failed to establish connection to MCP server."""
    code = "MCP-001"
    name = "MCPConnectionError"
    default_message = "Failed to connect to MCP server"
    recovery_suggestions = [
        "Check that the MCP server command is available (uvx, npx)",
        "Verify network connectivity",
        "Try running the MCP server manually to check for errors",
        "Check ~/.acti/logs/error.log for detailed error information",
    ]

    def __init__(self, tool_name: str, original_error: Exception = None):
        self.tool_name = tool_name
        self.original_error = original_error
        message = f"Failed to connect to MCP server for '{tool_name}'"
        if original_error:
            message += f": {original_error}"
        super().__init__(
            message=message,
            context={"tool_name": tool_name, "original_error": str(original_error)},
        )


class MCPTimeoutError(MCPException):
    """MCP server connection or operation timed out."""
    code = "MCP-002"
    name = "MCPTimeoutError"
    default_message = "MCP operation timed out"
    recovery_suggestions = [
        "Try again - the server may be temporarily slow",
        "Increase the timeout with /config mcp_timeout 60",
        "Check if the MCP server is running correctly",
        "Verify there are no network issues",
    ]

    def __init__(self, tool_name: str, timeout: float):
        self.tool_name = tool_name
        self.timeout = timeout
        super().__init__(
            message=f"Connection to MCP server '{tool_name}' timed out after {timeout}s",
            context={"tool_name": tool_name, "timeout_seconds": timeout},
        )


class MCPCredentialError(MCPException):
    """Required API key or credential is missing."""
    code = "MCP-003"
    name = "MCPCredentialError"
    default_message = "Missing required credentials"

    def __init__(self, tool_name: str, missing_vars: list[str]):
        self.tool_name = tool_name
        self.missing_vars = missing_vars

        # Generate specific recovery suggestions based on tool
        suggestions = [
            f"Set the following environment variable(s): {', '.join(missing_vars)}",
        ]

        # Tool-specific setup guidance
        tool_docs = {
            "voice": "See: https://elevenlabs.io/api - Get your API key from the dashboard",
            "websearch": "See: https://tavily.com - Sign up for a free API key",
            "calendar": "See: https://console.cloud.google.com - Set up OAuth credentials",
            "communication": "See: https://console.twilio.com - Get your Account SID and Auth Token",
            "crm": "See: https://developers.hubspot.com - Generate a private app access token",
        }
        if tool_name in tool_docs:
            suggestions.append(tool_docs[tool_name])

        suggestions.append("Add credentials to .env file or export in shell")

        super().__init__(
            message=f"Missing credentials for '{tool_name}': {', '.join(missing_vars)}",
            context={"tool_name": tool_name, "missing_vars": missing_vars},
            recovery_suggestions=suggestions,
        )
        self.documentation_url = f"https://acti.dev/docs/setup/{tool_name}"


class MCPConfigurationError(MCPException):
    """Invalid MCP server configuration."""
    code = "MCP-004"
    name = "MCPConfigurationError"
    default_message = "Invalid MCP configuration"
    recovery_suggestions = [
        "Check MCP server configuration in src/tool_registry.py",
        "Verify the MCP server package is installed correctly",
        "Run the MCP server manually to check for configuration errors",
    ]

    def __init__(self, tool_name: str, message: str):
        self.tool_name = tool_name
        super().__init__(
            message=f"Configuration error for '{tool_name}': {message}",
            context={"tool_name": tool_name},
        )


class MCPToolNotFoundError(MCPException):
    """Requested tool category does not exist in registry."""
    code = "MCP-005"
    name = "MCPToolNotFoundError"
    default_message = "Tool not found in MCP registry"

    def __init__(self, tool_name: str, available_tools: list[str]):
        self.tool_name = tool_name
        self.available_tools = available_tools
        super().__init__(
            message=f"Unknown tool category '{tool_name}'. Available tools: {', '.join(available_tools)}",
            context={"tool_name": tool_name, "available_tools": available_tools},
            recovery_suggestions=[
                f"Use one of the available tools: {', '.join(available_tools)}",
                "Check spelling of the tool name",
                "Use /tools to see all available tools and their status",
            ],
        )


class MCPToolExecutionError(MCPException):
    """Tool invocation failed during execution."""
    code = "MCP-006"
    name = "MCPToolExecutionError"
    default_message = "Tool execution failed"
    recovery_suggestions = [
        "Check the tool inputs for validity",
        "Verify the external service is operational",
        "Try the operation again",
        "Check ~/.acti/logs/error.log for details",
    ]

    def __init__(self, tool_name: str, operation: str, original_error: Exception = None):
        self.tool_name = tool_name
        self.operation = operation
        self.original_error = original_error
        message = f"Tool '{tool_name}' failed during '{operation}'"
        if original_error:
            message += f": {original_error}"
        super().__init__(
            message=message,
            context={
                "tool_name": tool_name,
                "operation": operation,
                "original_error": str(original_error),
            },
        )


class MCPServerCrashError(MCPException):
    """MCP server process terminated unexpectedly."""
    code = "MCP-007"
    name = "MCPServerCrashError"
    default_message = "MCP server crashed unexpectedly"
    recovery_suggestions = [
        "Try the operation again - the server will be restarted",
        "Check MCP server logs for crash details",
        "Verify the MCP server package is up to date",
        "Report persistent crashes to the MCP server maintainers",
    ]

    def __init__(self, tool_name: str, exit_code: int = None):
        self.tool_name = tool_name
        self.exit_code = exit_code
        message = f"MCP server for '{tool_name}' terminated unexpectedly"
        if exit_code is not None:
            message += f" (exit code: {exit_code})"
        super().__init__(
            message=message,
            context={"tool_name": tool_name, "exit_code": exit_code},
        )


class MCPProtocolError(MCPException):
    """Invalid MCP protocol message received."""
    code = "MCP-008"
    name = "MCPProtocolError"
    default_message = "Invalid MCP protocol message"
    recovery_suggestions = [
        "Update the MCP server to the latest version",
        "Update langchain-mcp-adapters package",
        "Report this issue if it persists",
    ]


class MCPPartialFailureWarning(MCPException):
    """Some tools connected successfully, but others failed."""
    code = "MCP-009"
    name = "MCPPartialFailureWarning"
    severity = ErrorSeverity.WARNING
    default_message = "Some MCP tools failed to connect"

    def __init__(
        self,
        connected_tools: list[str],
        failed_tools: list[str],
        errors: dict[str, str],
    ):
        self.connected_tools = connected_tools
        self.failed_tools = failed_tools
        self.errors = errors

        message = f"Connected: {', '.join(connected_tools) or 'none'}. "
        message += f"Failed: {', '.join(failed_tools) or 'none'}."

        super().__init__(
            message=message,
            context={
                "connected_tools": connected_tools,
                "failed_tools": failed_tools,
                "errors": errors,
            },
            recovery_suggestions=[
                f"Check credentials for failed tools: {', '.join(failed_tools)}",
                "The agent will operate with reduced capabilities",
                "Use /tools to see tool status",
            ],
        )
```

### Agent Exceptions (Key Examples)

```python
class AgentNotFoundError(AgentException):
    """Agent configuration file does not exist."""
    code = "AGT-001"
    name = "AgentNotFoundError"

    def __init__(self, agent_name: str, search_path: str = None):
        self.agent_name = agent_name
        self.search_path = search_path
        super().__init__(
            message=f"Agent '{agent_name}' not found",
            context={"agent_name": agent_name, "search_path": search_path},
            recovery_suggestions=[
                "Use /list to see available agents",
                "Check the agent name spelling",
                f"Verify the agent exists in {search_path or 'outputs/agents/'}",
            ],
        )


class AgentConfigInvalidError(AgentException):
    """Agent configuration JSON is malformed or invalid."""
    code = "AGT-002"
    name = "AgentConfigInvalidError"

    def __init__(self, agent_name: str, validation_errors: list[str]):
        self.agent_name = agent_name
        self.validation_errors = validation_errors
        super().__init__(
            message=f"Invalid configuration for agent '{agent_name}': {'; '.join(validation_errors)}",
            context={"agent_name": agent_name, "validation_errors": validation_errors},
            recovery_suggestions=[
                "Check the agent's JSON file for syntax errors",
                "Recreate the agent with /create",
                "Manually fix the configuration file",
            ],
        )


class AgentExecutionError(AgentException):
    """Agent failed during task execution."""
    code = "AGT-003"
    name = "AgentExecutionError"

    def __init__(self, agent_name: str, task: str, original_error: Exception = None):
        self.agent_name = agent_name
        self.task = task
        self.original_error = original_error
        message = f"Agent '{agent_name}' failed executing task"
        if original_error:
            message += f": {original_error}"
        super().__init__(
            message=message,
            context={
                "agent_name": agent_name,
                "task_preview": task[:100] if task else None,
                "original_error": str(original_error),
            },
            recovery_suggestions=[
                "Try rephrasing the task",
                "Check if the agent's tools are working (/tools)",
                "Reload the agent (/load {agent_name})",
                "Check error logs for more details",
            ],
        )
```

---

## Recovery Strategies

### 1. Automatic Retry for Transient Failures

```python
from dataclasses import dataclass
from typing import Callable, TypeVar, Awaitable
import asyncio
import random


@dataclass
class RetryConfig:
    """Configuration for automatic retry behavior."""
    max_attempts: int = 3
    initial_delay: float = 1.0      # seconds
    max_delay: float = 30.0         # seconds
    exponential_base: float = 2.0
    jitter: bool = True             # Add randomness to prevent thundering herd
    retryable_errors: tuple = (
        MCPConnectionError,
        MCPTimeoutError,
        MCPServerCrashError,
        NetworkException,
    )


T = TypeVar("T")


async def with_retry(
    operation: Callable[[], Awaitable[T]],
    config: RetryConfig = None,
    operation_name: str = "operation",
) -> T:
    """Execute an async operation with automatic retry on transient failures.

    Args:
        operation: Async callable to execute.
        config: Retry configuration.
        operation_name: Name for logging purposes.

    Returns:
        Result of the operation.

    Raises:
        Last exception if all retries exhausted.
    """
    config = config or RetryConfig()
    last_error: Exception = None

    for attempt in range(1, config.max_attempts + 1):
        try:
            return await operation()

        except config.retryable_errors as e:
            last_error = e

            if attempt == config.max_attempts:
                logger.error(
                    f"{operation_name} failed after {config.max_attempts} attempts: {e}"
                )
                raise

            # Calculate delay with exponential backoff
            delay = min(
                config.initial_delay * (config.exponential_base ** (attempt - 1)),
                config.max_delay,
            )

            # Add jitter
            if config.jitter:
                delay = delay * (0.5 + random.random())

            logger.warning(
                f"{operation_name} failed (attempt {attempt}/{config.max_attempts}): {e}. "
                f"Retrying in {delay:.1f}s..."
            )

            await asyncio.sleep(delay)

    raise last_error
```

### 2. Graceful Degradation

When non-critical tools are unavailable, agents should continue with reduced capabilities.

```python
@dataclass
class DegradationPolicy:
    """Policy for handling partial tool availability."""

    # Tools that are required - agent won't start without them
    required_tools: set[str] = field(default_factory=set)

    # Tools that are optional - agent degrades if missing
    optional_tools: set[str] = field(default_factory=set)

    # Whether to warn user about degraded state
    warn_on_degradation: bool = True

    # Whether to log degradation events
    log_degradation: bool = True


async def create_agent_with_degradation(
    agent_config: AgentConfig,
    policy: DegradationPolicy = None,
) -> tuple[CompiledStateGraph, list[str]]:
    """Create agent with graceful degradation for missing tools.

    Args:
        agent_config: Agent configuration.
        policy: Degradation policy (defaults to all tools optional).

    Returns:
        Tuple of (agent, list of unavailable tools).

    Raises:
        MCPCredentialError: If a required tool is missing credentials.
    """
    policy = policy or DegradationPolicy()

    # Validate required tools
    validation = await validate_agent_tools(agent_config)

    missing_required = set(validation.missing) & policy.required_tools
    if missing_required:
        tool = list(missing_required)[0]
        missing_vars = _check_credentials(tool)
        raise MCPCredentialError(tool, missing_vars)

    # Create agent with available tools only
    agent = await create_agent_with_tools(
        agent_config,
        skip_unavailable=True,
    )

    # Return agent and list of what's missing
    return agent, validation.missing
```

### 3. User-Guided Recovery

For errors requiring user action, provide clear guidance:

```python
@dataclass
class RecoveryAction:
    """A suggested recovery action for the user."""
    action: str                     # What to do
    command: Optional[str] = None   # CLI command to run
    explanation: str = ""           # Why this might help
    risk_level: str = "safe"        # safe, moderate, risky


def get_recovery_actions(error: ACTiError) -> list[RecoveryAction]:
    """Get contextual recovery actions for an error."""

    actions = []

    if error.code == "MCP-003":  # Credential error
        tool_name = error.context.get("tool_name", "unknown")
        missing_vars = error.context.get("missing_vars", [])

        for var in missing_vars:
            actions.append(RecoveryAction(
                action=f"Set {var} environment variable",
                command=f"export {var}=your_api_key_here",
                explanation=f"This API key is required for {tool_name} functionality",
                risk_level="safe",
            ))

        actions.append(RecoveryAction(
            action="Add credentials to .env file",
            command=f"echo '{missing_vars[0]}=your_key' >> .env",
            explanation="Persists credentials across sessions",
            risk_level="safe",
        ))

    elif error.code == "MCP-002":  # Timeout
        actions.append(RecoveryAction(
            action="Retry the operation",
            explanation="Transient network issues may have caused the timeout",
            risk_level="safe",
        ))
        actions.append(RecoveryAction(
            action="Increase timeout duration",
            command="/config mcp_timeout 60",
            explanation="Some operations require more time",
            risk_level="safe",
        ))

    elif error.code == "AGT-001":  # Agent not found
        agent_name = error.context.get("agent_name", "unknown")
        actions.append(RecoveryAction(
            action="List available agents",
            command="/list",
            explanation="View all agents that have been created",
            risk_level="safe",
        ))
        actions.append(RecoveryAction(
            action="Create the agent",
            command=f"/create {agent_name}",
            explanation="Create a new agent with this name",
            risk_level="safe",
        ))

    return actions
```

### 4. Recovery Suggestion Templates

| Error Code | Recovery Suggestions |
|------------|---------------------|
| `MCP-001` | Verify network; restart MCP server; check server logs |
| `MCP-002` | Retry operation; increase timeout; check server health |
| `MCP-003` | Set missing env vars; check .env file; verify API key validity |
| `MCP-004` | Check tool_registry.py; verify MCP package installed |
| `MCP-005` | Use /tools to see available; check spelling |
| `AGT-001` | Use /list; check spelling; verify file exists |
| `AGT-002` | Check JSON syntax; recreate agent; manual fix |
| `CLI-001` | Use /help; check spelling |
| `FIL-002` | Check permissions; run with appropriate access |
| `NET-002` | Wait and retry; check rate limit documentation |

---

## Error Logging

### Log Configuration

Errors are logged to `~/.acti/logs/error.log` with the following configuration:

```python
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path


# Log locations
ACTI_LOG_DIR = Path.home() / ".acti" / "logs"
ERROR_LOG_FILE = ACTI_LOG_DIR / "error.log"
DEBUG_LOG_FILE = ACTI_LOG_DIR / "debug.log"

# Log configuration
LOG_CONFIG = {
    "max_file_size": 10 * 1024 * 1024,  # 10 MB
    "backup_count": 5,                    # Keep 5 rotated files
    "encoding": "utf-8",
}


def setup_error_logging(
    verbose: bool = False,
    log_level: str = "INFO",
) -> logging.Logger:
    """Configure error logging for ACTi CLI.

    Args:
        verbose: If True, also log to console with DEBUG level.
        log_level: Minimum log level (DEBUG, INFO, WARNING, ERROR).

    Returns:
        Configured logger instance.
    """
    # Ensure log directory exists
    ACTI_LOG_DIR.mkdir(parents=True, exist_ok=True)

    # Create logger
    logger = logging.getLogger("acti")
    logger.setLevel(logging.DEBUG)  # Capture all, filter at handler level

    # Error log handler (rotating file)
    error_handler = RotatingFileHandler(
        ERROR_LOG_FILE,
        maxBytes=LOG_CONFIG["max_file_size"],
        backupCount=LOG_CONFIG["backup_count"],
        encoding=LOG_CONFIG["encoding"],
    )
    error_handler.setLevel(logging.WARNING)  # Only WARNING and above
    error_handler.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s\n"
        "    Context: %(context)s\n"
        "    Error ID: %(error_id)s\n",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    logger.addHandler(error_handler)

    # Debug log handler (when verbose)
    if verbose:
        debug_handler = RotatingFileHandler(
            DEBUG_LOG_FILE,
            maxBytes=LOG_CONFIG["max_file_size"],
            backupCount=LOG_CONFIG["backup_count"],
            encoding=LOG_CONFIG["encoding"],
        )
        debug_handler.setLevel(logging.DEBUG)
        debug_handler.setFormatter(logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        ))
        logger.addHandler(debug_handler)

    return logger
```

### Log Entry Format

Standard log entries include:

```
2025-01-09 15:30:45 | ERROR    | acti.mcp | MCPCredentialError: Missing credentials for 'calendar'
    Context: {"tool_name": "calendar", "missing_vars": ["GOOGLE_OAUTH_CREDENTIALS"], "command": "/load scheduler"}
    Error ID: a1b2c3d4
```

### Structured Log Entry (JSON Lines format)

For machine parsing, logs can also be written in JSON Lines format:

```json
{"timestamp": "2025-01-09T15:30:45.123456", "level": "ERROR", "logger": "acti.mcp", "code": "MCP-003", "name": "MCPCredentialError", "message": "Missing credentials for 'calendar': GOOGLE_OAUTH_CREDENTIALS", "context": {"tool_name": "calendar", "missing_vars": ["GOOGLE_OAUTH_CREDENTIALS"]}, "error_id": "a1b2c3d4", "session_id": "sess_xyz123"}
```

### Log Functions

```python
def log_error(
    error: ACTiError,
    include_trace: bool = False,
    extra_context: dict = None,
) -> None:
    """Log an error with full context.

    Args:
        error: The ACTiError to log.
        include_trace: Whether to include stack trace.
        extra_context: Additional context to merge.
    """
    logger = logging.getLogger("acti")

    context = {**error.context}
    if extra_context:
        context.update(extra_context)

    extra = {
        "context": json.dumps(context),
        "error_id": error.error_id,
    }

    log_message = f"{error.name}: {error.message}"

    if error.severity == ErrorSeverity.WARNING:
        logger.warning(log_message, extra=extra)
    elif error.severity == ErrorSeverity.CRITICAL:
        logger.critical(log_message, extra=extra)
    else:
        logger.error(log_message, extra=extra)

    if include_trace and error.stack_trace:
        logger.debug(f"Stack trace for {error.error_id}:\n{error.stack_trace}")
```

---

## User-Facing Error Display

### Terminal Display Formatting

```python
from dataclasses import dataclass


@dataclass
class DisplayConfig:
    """Configuration for error display."""
    use_color: bool = True
    show_error_code: bool = True
    show_recovery_suggestions: bool = True
    show_documentation_link: bool = True
    max_suggestion_count: int = 3
    verbose: bool = False


class ErrorDisplay:
    """Formats errors for terminal display."""

    # ANSI color codes
    COLORS = {
        "reset": "\033[0m",
        "bold": "\033[1m",
        "dim": "\033[2m",
        "red": "\033[31m",
        "yellow": "\033[33m",
        "green": "\033[32m",
        "cyan": "\033[36m",
        "magenta": "\033[35m",
    }

    def __init__(self, config: DisplayConfig = None):
        self.config = config or DisplayConfig()
        if not self.config.use_color:
            self.COLORS = {k: "" for k in self.COLORS}

    def format_error(self, error: ACTiError) -> str:
        """Format an error for terminal display.

        Args:
            error: The error to format.

        Returns:
            Formatted string for terminal output.
        """
        c = self.COLORS
        lines = []

        # Header line with severity-appropriate color
        color = c["red"] if error.severity == ErrorSeverity.ERROR else c["yellow"]
        severity_label = error.severity.value.upper()

        if self.config.show_error_code:
            header = f"{color}{c['bold']}{severity_label}:{c['reset']} {error.message}"
            lines.append(header)
            lines.append(f"{c['dim']}Error Code: {error.code} ({error.name}){c['reset']}")
        else:
            lines.append(f"{color}{c['bold']}{severity_label}:{c['reset']} {error.message}")

        # Recovery suggestions
        if self.config.show_recovery_suggestions and error.recovery_suggestions:
            lines.append("")
            lines.append(f"{c['cyan']}How to fix:{c['reset']}")
            for i, suggestion in enumerate(error.recovery_suggestions[:self.config.max_suggestion_count]):
                lines.append(f"  {c['green']}{i+1}.{c['reset']} {suggestion}")

            remaining = len(error.recovery_suggestions) - self.config.max_suggestion_count
            if remaining > 0:
                lines.append(f"  {c['dim']}... and {remaining} more suggestions{c['reset']}")

        # Documentation link
        if self.config.show_documentation_link and error.documentation_url:
            lines.append("")
            lines.append(f"{c['dim']}Documentation: {error.documentation_url}{c['reset']}")

        # Debug info (verbose mode only)
        if self.config.verbose:
            lines.append("")
            lines.append(f"{c['dim']}Error ID: {error.error_id}{c['reset']}")
            lines.append(f"{c['dim']}Timestamp: {error.timestamp}{c['reset']}")
            if error.context:
                lines.append(f"{c['dim']}Context: {json.dumps(error.context, indent=2)}{c['reset']}")
            if error.stack_trace:
                lines.append(f"{c['dim']}Stack Trace:{c['reset']}")
                lines.append(error.stack_trace)

        return "\n".join(lines)

    def format_warning(self, error: ACTiError) -> str:
        """Format a warning for terminal display."""
        c = self.COLORS
        return f"{c['yellow']}{c['bold']}WARNING:{c['reset']} {error.message}"


# Convenience functions matching existing CLI style
def display_error(error: ACTiError, verbose: bool = False) -> None:
    """Display an error to the user."""
    display = ErrorDisplay(DisplayConfig(verbose=verbose))
    print(display.format_error(error))


def display_warning(message: str) -> None:
    """Display a warning message."""
    print(f"\033[33m\033[1mWARNING:\033[0m {message}")
```

### Example Error Displays

**Standard Error (MCP Credential Error):**

```
ERROR: Missing credentials for 'calendar': GOOGLE_OAUTH_CREDENTIALS environment variable not set.
Error Code: MCP-003 (MCPCredentialError)

How to fix:
  1. Set the following environment variable(s): GOOGLE_OAUTH_CREDENTIALS
  2. See: https://console.cloud.google.com - Set up OAuth credentials
  3. Add credentials to .env file or export in shell

Documentation: https://acti.dev/docs/errors/MCP-003
```

**Warning (Partial Tool Failure):**

```
WARNING: Some MCP tools failed to connect. Connected: websearch, voice. Failed: calendar.
Error Code: MCP-009 (MCPPartialFailureWarning)

How to fix:
  1. Check credentials for failed tools: calendar
  2. The agent will operate with reduced capabilities
  3. Use /tools to see tool status
```

**Verbose Mode (includes debug info):**

```
ERROR: Agent 'appointment_scheduler' failed executing task
Error Code: AGT-003 (AgentExecutionError)

How to fix:
  1. Try rephrasing the task
  2. Check if the agent's tools are working (/tools)
  3. Reload the agent (/load appointment_scheduler)

Documentation: https://acti.dev/docs/errors/AGT-003

Error ID: a1b2c3d4
Timestamp: 2025-01-09T15:30:45.123456
Context: {
  "agent_name": "appointment_scheduler",
  "task_preview": "Schedule a meeting with John...",
  "original_error": "Calendar API returned 401 Unauthorized"
}
```

---

## Integration Points

### Command Handler Error Handling Pattern

```python
async def handle_command_with_errors(
    handler: Callable,
    session: SessionState,
    args: list[str],
    config: CLIConfig,
) -> CommandResult:
    """Wrapper that provides consistent error handling for all commands.

    Args:
        handler: The command handler function.
        session: Current session state.
        args: Command arguments.
        config: CLI configuration.

    Returns:
        CommandResult with success or error information.
    """
    try:
        return await handler(session, args, config)

    except ACTiException as e:
        # Known error - display with recovery suggestions
        error = e.to_acti_error(include_trace=config.verbose)
        log_error(error, include_trace=config.verbose)
        display_error(error, verbose=config.verbose)

        return CommandResult(
            success=False,
            message=error.message,
            data={"error": error.to_dict()},
        )

    except KeyboardInterrupt:
        # User cancelled
        print(f"\n{Colors.DIM}Operation cancelled.{Colors.RESET}")
        return CommandResult(
            success=False,
            message="Operation cancelled by user",
        )

    except Exception as e:
        # Unexpected error - log full trace, show generic message
        error = ACTiError.from_exception(
            e,
            code="ACT-999",
            name="UnexpectedError",
            recovery_suggestions=[
                "Try the operation again",
                "Check ~/.acti/logs/error.log for details",
                "Report this issue if it persists",
            ],
            include_trace=True,
        )
        log_error(error, include_trace=True)

        if config.verbose:
            display_error(error, verbose=True)
        else:
            print(f"{Colors.RED}ERROR:{Colors.RESET} An unexpected error occurred: {e}")
            print(f"{Colors.DIM}Use --verbose for more details or check ~/.acti/logs/error.log{Colors.RESET}")

        return CommandResult(
            success=False,
            message=f"Unexpected error: {e}",
            data={"error": error.to_dict()},
        )
```

### MCP Integration Error Handling

```python
async def create_agent_with_tools_safe(
    agent_config: AgentConfig,
    skip_unavailable: bool = False,
    timeout: float = 30.0,
    verbose: bool = False,
) -> tuple[Optional[CompiledStateGraph], Optional[ACTiError]]:
    """Create agent with comprehensive error handling.

    Returns:
        Tuple of (agent, error). One will be None.
    """
    try:
        # Try with retry for transient failures
        agent = await with_retry(
            lambda: create_agent_with_tools(
                agent_config,
                skip_unavailable=skip_unavailable,
                timeout=timeout,
            ),
            config=RetryConfig(max_attempts=2),
            operation_name="create_agent",
        )
        return agent, None

    except MCPCredentialError as e:
        error = e.to_acti_error()
        return None, error

    except MCPConnectionError as e:
        error = e.to_acti_error()
        return None, error

    except MCPTimeoutError as e:
        error = e.to_acti_error()
        return None, error

    except Exception as e:
        error = ACTiError.from_exception(
            e,
            code="AGT-004",
            name="AgentInstantiationError",
            recovery_suggestions=[
                "Check agent configuration for errors",
                "Verify MCP tools are available (/tools)",
                "Try creating a simpler agent first",
            ],
        )
        return None, error
```

---

## Testing Error Handling

### Test Patterns

```python
import pytest
from unittest.mock import AsyncMock, patch


@pytest.mark.asyncio
async def test_mcp_credential_error_formatting():
    """Test that credential errors have proper recovery suggestions."""
    error = MCPCredentialError("calendar", ["GOOGLE_OAUTH_CREDENTIALS"])

    assert error.code == "MCP-003"
    assert "calendar" in error.message
    assert "GOOGLE_OAUTH_CREDENTIALS" in error.message
    assert len(error.recovery_suggestions) >= 2
    assert any("environment variable" in s for s in error.recovery_suggestions)


@pytest.mark.asyncio
async def test_retry_on_transient_failure():
    """Test that transient failures trigger retry."""
    call_count = 0

    async def flaky_operation():
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise MCPTimeoutError("websearch", 30.0)
        return "success"

    result = await with_retry(
        flaky_operation,
        config=RetryConfig(max_attempts=3, initial_delay=0.1),
    )

    assert result == "success"
    assert call_count == 3


@pytest.mark.asyncio
async def test_graceful_degradation():
    """Test agent creation with missing optional tools."""
    config = AgentConfig(
        name="test_agent",
        description="Test agent",
        system_prompt="You are a test agent.",
        tools=["websearch", "calendar"],  # Assume calendar not configured
    )

    with patch("src.mcp_integration.validate_agent_tools") as mock_validate:
        mock_validate.return_value = ToolValidationResult(
            valid=False,
            available=["websearch"],
            missing=["calendar"],
            messages=["calendar: Missing GOOGLE_OAUTH_CREDENTIALS"],
        )

        agent, missing = await create_agent_with_degradation(
            config,
            policy=DegradationPolicy(optional_tools={"calendar"}),
        )

        assert agent is not None
        assert "calendar" in missing


@pytest.mark.asyncio
async def test_error_display_formatting():
    """Test error display output."""
    error = ACTiError(
        code="MCP-003",
        category="MCP",
        name="MCPCredentialError",
        message="Missing credentials for 'calendar'",
        recovery_suggestions=[
            "Set GOOGLE_OAUTH_CREDENTIALS",
            "Check .env file",
        ],
    )

    display = ErrorDisplay(DisplayConfig(use_color=False))
    output = display.format_error(error)

    assert "ERROR:" in output
    assert "MCP-003" in output
    assert "Missing credentials" in output
    assert "How to fix:" in output
    assert "Set GOOGLE_OAUTH_CREDENTIALS" in output


@pytest.mark.asyncio
async def test_error_logging():
    """Test that errors are properly logged."""
    with patch("logging.Logger.error") as mock_log:
        error = MCPCredentialError("calendar", ["GOOGLE_OAUTH_CREDENTIALS"])
        log_error(error.to_acti_error())

        mock_log.assert_called_once()
        call_args = mock_log.call_args
        assert "MCPCredentialError" in call_args[0][0]
```

---

## Migration Guide

### Updating Existing Exception Handling

Replace existing exception handling with the new structured approach:

**Before:**

```python
try:
    agent = await create_agent_with_tools(config)
except Exception as e:
    print(f"Error: {e}")
```

**After:**

```python
try:
    agent = await create_agent_with_tools(config)
except MCPCredentialError as e:
    display_error(e.to_acti_error())
except MCPConnectionError as e:
    display_error(e.to_acti_error())
except MCPTimeoutError as e:
    display_error(e.to_acti_error())
except Exception as e:
    error = ACTiError.from_exception(e, "AGT-004", "AgentInstantiationError")
    log_error(error, include_trace=True)
    display_error(error)
```

---

## Summary

This error handling specification provides:

1. **Comprehensive Error Taxonomy**: 40+ error codes across 6 categories
2. **Structured Error Format**: Consistent `ACTiError` object with codes, messages, and recovery suggestions
3. **Exception Hierarchy**: Python exception classes that map to error codes
4. **Recovery Strategies**: Automatic retry, graceful degradation, and user-guided recovery
5. **Logging System**: Rotating file logs with structured format at `~/.acti/logs/`
6. **User Display**: Colored, actionable error messages with documentation links
7. **Integration Patterns**: Ready-to-use wrappers for command handlers

All error handling should follow the patterns in this document to ensure a consistent, user-friendly experience across the ACTi CLI.
