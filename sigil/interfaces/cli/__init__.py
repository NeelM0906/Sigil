"""CLI for Sigil v2 framework.

This module implements the command-line interface for Sigil v2:
- Modular command handlers for orchestration, memory, planning, reasoning
- Interactive session management with EventStore persistence
- Token usage tracking and visualization
- Real-time execution log monitoring

CLI Commands:
    sigil orchestrate <task>         Run task through full pipeline
    sigil memory <subcommand>        Memory system operations
    sigil sessions <subcommand>      Session management
    sigil history                    View conversation history
    sigil planning <subcommand>      Plan creation and execution
    sigil reasoning <subcommand>     Reasoning strategy management
    sigil budget <subcommand>        Token budget management
    sigil contracts <subcommand>     Contract management
    sigil help [command]             Show help information
    sigil status                     Show system status
    sigil version                    Show version information
    sigil config                     Show active configuration
    sigil clear                      Clear the screen
    sigil log-stream                 Real-time log viewer
    sigil interactive                Start interactive session

Key Components:
    - cli: Main CLI application (click-based)
    - ColoredOutput: Color-coded terminal output
    - TokenMeter: Token usage visualization
    - InteractiveSession: Session management with EventStore
    - Command handlers in sigil/interfaces/cli/commands/

Usage:
    # Run orchestration
    sigil orchestrate "Qualify lead John from Acme"

    # Memory operations
    sigil memory query "customer preferences"
    sigil memory store "Lead prefers email" --category preferences

    # Session management
    sigil sessions list
    sigil sessions new --name "Research Session"

    # Reasoning
    sigil reasoning strategies
    sigil reasoning test "2+2" --strategy chain_of_thought

    # Watch logs in real-time
    sigil log-stream

    # Start interactive mode
    sigil interactive
"""

from sigil.interfaces.cli.monitoring import (
    TokenDisplay,
    PipelineTokenTracker,
    SigilLogFormatter,
    SigilLogAdapter,
    RealTimeTokenCounter,
    setup_execution_logging,
    get_execution_logger,
    get_log_file_path,
    log_pipeline_step,
    log_token_summary,
    LOG_FILE_PATH,
    TOTAL_TOKEN_BUDGET,
)

from sigil.interfaces.cli.config import (
    CLIConfig,
    get_cli_config,
    reset_cli_config,
    set_cli_config,
    ANSI_COLORS,
    SEMANTIC_COLORS,
    TOKEN_THRESHOLDS,
    DEFAULT_TOKEN_BUDGET,
)

from sigil.interfaces.cli.formatter import (
    ColoredOutput,
    TokenMeter,
    ResultFormatter,
    colorize,
)

from sigil.interfaces.cli.session import (
    Message,
    InteractiveSession,
    get_active_session,
    reset_active_session,
    DEFAULT_SESSION_DIR,
)

from sigil.interfaces.cli.app import (
    cli,
    main,
)

from sigil.interfaces.cli.logging import (
    CliLogEntry,
    CliLogFormatter,
    CliLogger,
    setup_cli_logging,
    get_cli_logger,
    get_cli_logger_wrapper,
    reset_cli_logging,
    get_cli_log_file_path,
    log_command,
    CLI_LOG_FILE_PATH,
    CLI_TOKEN_BUDGET,
)

from sigil.interfaces.cli.schemas import (
    LogLevel,
    CommandStatus,
    BudgetStatus,
    CliLogEntry as CliLogEntrySchema,
    TokenUsageStats,
    SessionStats,
    MonitorEntry,
    LogFilter,
)


__all__ = [
    # CLI application
    "cli",
    "main",

    # Configuration
    "CLIConfig",
    "get_cli_config",
    "reset_cli_config",
    "set_cli_config",
    "ANSI_COLORS",
    "SEMANTIC_COLORS",
    "TOKEN_THRESHOLDS",
    "DEFAULT_TOKEN_BUDGET",

    # Formatter
    "ColoredOutput",
    "TokenMeter",
    "ResultFormatter",
    "colorize",

    # Session management
    "Message",
    "InteractiveSession",
    "get_active_session",
    "reset_active_session",
    "DEFAULT_SESSION_DIR",

    # Monitoring
    "TokenDisplay",
    "PipelineTokenTracker",
    "SigilLogFormatter",
    "SigilLogAdapter",
    "RealTimeTokenCounter",
    "setup_execution_logging",
    "get_execution_logger",
    "get_log_file_path",
    "log_pipeline_step",
    "log_token_summary",
    "LOG_FILE_PATH",
    "TOTAL_TOKEN_BUDGET",

    # CLI Logging (JSON Lines format)
    "CliLogEntry",
    "CliLogFormatter",
    "CliLogger",
    "setup_cli_logging",
    "get_cli_logger",
    "get_cli_logger_wrapper",
    "reset_cli_logging",
    "get_cli_log_file_path",
    "log_command",
    "CLI_LOG_FILE_PATH",
    "CLI_TOKEN_BUDGET",

    # CLI Schemas
    "LogLevel",
    "CommandStatus",
    "BudgetStatus",
    "CliLogEntrySchema",
    "TokenUsageStats",
    "SessionStats",
    "MonitorEntry",
    "LogFilter",
]
