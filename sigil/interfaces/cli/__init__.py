"""CLI for Sigil v2 framework.

This module implements the command-line interface for Sigil v2:
- Orchestrated pipeline execution (Route -> Plan -> Reason -> Validate)
- Real-time execution log monitoring
- Token usage tracking and reporting
- System status and health checks

CLI Commands:
    - sigil orchestrate --task "..." --session-id "...": Run through pipeline
    - sigil log-stream: Tail execution logs in real-time
    - sigil status: Show orchestrator status and metrics

Key Components:
    - cli: Main CLI application (click-based)
    - orchestrate: Execute complete orchestration pipeline
    - log_stream: Real-time log viewer
    - status: System status display
    - TokenDisplay: Format token usage information
    - PipelineTokenTracker: Track tokens per pipeline step
    - SigilLogFormatter: Custom log formatter with token counts

Usage:
    # Run orchestration
    python -m sigil.interfaces.cli.app orchestrate --task "Qualify lead" --session test-1

    # Watch logs in real-time
    python -m sigil.interfaces.cli.app log-stream

    # Check status
    python -m sigil.interfaces.cli.app status

    # Run standalone monitor (separate terminal)
    python scripts/monitor.py
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

from sigil.interfaces.cli.app import (
    cli,
    main,
)


__all__ = [
    # CLI commands
    "cli",
    "main",
    # Monitoring
    "TokenDisplay",
    "PipelineTokenTracker",
    "SigilLogFormatter",
    "SigilLogAdapter",
    "RealTimeTokenCounter",
    # Logging utilities
    "setup_execution_logging",
    "get_execution_logger",
    "get_log_file_path",
    "log_pipeline_step",
    "log_token_summary",
    # Constants
    "LOG_FILE_PATH",
    "TOTAL_TOKEN_BUDGET",
]
